import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.LinearAlgebra.Basic
import Mathlib.Algebra.LinearEq
import Mathlib.Algebra.LinearEquiv
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Logarithm
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finite.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Rat
import Mathlib.NumberTheory.Factorial
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real

namespace median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l81_81152

-- Definition of points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- The problem statements as Lean theorems
theorem median_on_AB_eq : ∀ (A B : ℝ × ℝ), A = (4, 0) ∧ B = (6, 7) → ∃ (x y : ℝ), x - 10 * y + 30 = 0 := by
  intros
  sorry

theorem altitude_on_BC_eq : ∀ (B C : ℝ × ℝ), B = (6, 7) ∧ C = (0, 3) → ∃ (x y : ℝ), 3 * x + 2 * y - 12 = 0 := by
  intros
  sorry

theorem perp_bisector_on_AC_eq : ∀ (A C : ℝ × ℝ), A = (4, 0) ∧ C = (0, 3) → ∃ (x y : ℝ), 8 * x - 6 * y - 7 = 0 := by
  intros
  sorry

end median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l81_81152


namespace distance_between_centers_l81_81424

open EuclideanGeometry

def triangle_XYZ (X Y Z : Point) := 
  right_triangle X Y Z ∧ dist X Y = 100 ∧ dist X Z = 140 ∧ dist Y Z = 180

def construction_AB_CD {X Y Z A B C D : Point} 
  (C1 C2 C3 : Circle) :=
  inscribed_circle C1 (triangle X Y Z) ∧
  on_line_segment A X Z ∧ on_line_segment B Y Z ∧
  perpendicular (line_segment A B) (line_segment X Z) ∧ tangent_to A B C1 ∧
  on_line_segment C X Y ∧ on_line_segment D Y Z ∧
  perpendicular (line_segment C D) (line_segment X Y) ∧ tangent_to C D C1 ∧
  inscribed_circle C2 (triangle X A B) ∧
  inscribed_circle C3 (triangle Y C D)

theorem distance_between_centers 
  {X Y Z A B C D : Point} {C1 C2 C3 : Circle}
  (h1 : triangle_XYZ X Y Z)
  (h2 : construction_AB_CD C1 C2 C3) :
  let O2 := center C2 in
  let O3 := center C3 in
  dist O2 O3 = real.sqrt (10 * 980) :=
sorry

end distance_between_centers_l81_81424


namespace possible_values_of_beta_l81_81422

noncomputable def find_possible_values_of_beta (β : ℂ) := 
  β ≠ 1 ∧ (|β^3 - 1| = 3 * |β - 1|) ∧ (|β^6 - 1| = 6 * |β - 1|) :=
  β = Complex.I * Real.sqrt 2 ∨ β = -Complex.I * Real.sqrt 2

theorem possible_values_of_beta (β : ℂ) : 
  β ≠ 1 ∧ (|β^3 - 1| = 3 * |β - 1|) ∧ (|β^6 - 1| = 6 * |β - 1|) ↔ 
  (β = Complex.I * Real.sqrt 2 ∨ β = -Complex.I * Real.sqrt 2) := sorry

end possible_values_of_beta_l81_81422


namespace cos_graph_shift_l81_81514

theorem cos_graph_shift :
  ∀ (x : ℝ), cos (2 * (x + π / 6)) = cos (2 * x + π / 3) :=
by
  intros
  sorry

end cos_graph_shift_l81_81514


namespace no_real_roots_of_quadratic_l81_81496

theorem no_real_roots_of_quadratic (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) :
  b^2 - 4 * a * c < 0 :=
by
  -- Use the given values
  rw [h1, h2, h3],
  -- Calculate discriminant for the equation x^2 + 2x + 3 = 0
  have h_discriminant : (2 : ℝ)^2 - 4 * (1 : ℝ) * (3 : ℝ) = -8 := by norm_num,
  -- Conclude that the discriminant is less than 0
  rw h_discriminant,
  exact neg_lt_zero.mpr (by norm_num : (0 : ℝ) < 8)

end no_real_roots_of_quadratic_l81_81496


namespace isosceles_triangle_perimeter_l81_81135

-- We define a structure for an isosceles triangle
structure IsoscelesTriangle where
  a b c : ℝ
  isosceles : a = b ∨ b = c ∨ a = c

-- We use the given conditions to declare the lengths of the sides
def side1 : ℝ := 4
def side2 : ℝ := 7

-- We then define the isosceles triangle with these side lengths
def triangle1 := IsoscelesTriangle.mk side1 side1 side2 (Or.inl rfl)
def triangle2 := IsoscelesTriangle.mk side1 side2 side2 (Or.inr rfl)

-- Finally, we state the theorem with the conditions and the proof goal
theorem isosceles_triangle_perimeter :
  let p1 := triangle1.a + triangle1.b + triangle1.c in
  let p2 := triangle2.a + triangle2.b + triangle2.c in
  p1 = 15 ∨ p2 = 18 :=
by
  sorry

end isosceles_triangle_perimeter_l81_81135


namespace arithmetic_sequence_m_value_l81_81785

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

noncomputable def find_m (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  (a (m + 1) + a (m - 1) - a m ^ 2 = 0) → (S (2 * m - 1) = 38) → m = 10

-- Problem Statement
theorem arithmetic_sequence_m_value :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ),
    arithmetic_sequence a → 
    sum_of_first_n_terms S a → 
    find_m a S m :=
by
  intros a S m ha hs h₁ h₂
  sorry

end arithmetic_sequence_m_value_l81_81785


namespace geralds_average_speed_l81_81468

theorem geralds_average_speed (poly_circuits : ℕ) (poly_time : ℝ) (track_length : ℝ) (gerald_speed_ratio : ℝ) :
  poly_circuits = 12 →
  poly_time = 0.5 →
  track_length = 0.25 →
  gerald_speed_ratio = 0.5 →
  let poly_speed :=  poly_circuits * track_length / poly_time in
  let gerald_speed :=  gerald_speed_ratio * poly_speed in
  gerald_speed = 3 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end geralds_average_speed_l81_81468


namespace insufficient_information_l81_81390

def members_count : ℕ := 22
def older_members_count : ℕ := 21

theorem insufficient_information :
  ∀ (ages : fin members_count → ℕ),
  members_count = 22 ∧ older_members_count = 21 ∧
  (∀ i ∈ {0 .. 21}, ages i > some_age) →
  ∃ (age_average : ℕ), age_average = ages.sum / members_count → false :=
by
  sorry

end insufficient_information_l81_81390


namespace max_distance_on_ellipse_l81_81893

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l81_81893


namespace problem_statement_l81_81500

def repeated_digits (d : ℕ) (n : ℕ) : ℕ :=
  let digits := d % 1000
  List.foldr (λ _ acc, acc * 1000 + digits) 0 (List.range n)

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def digit_sum (a b : ℕ) : ℕ :=
  let product := a * b
  let C := tens_digit product
  let D := units_digit product
  C + D

theorem problem_statement : digit_sum (repeated_digits 707 101) (repeated_digits 909 101) = 9 := by
  sorry

end problem_statement_l81_81500


namespace boat_ratio_l81_81214

theorem boat_ratio (b c d1 d2 : ℝ) 
  (h1 : b = 20) 
  (h2 : c = 4) 
  (h3 : d1 = 4) 
  (h4 : d2 = 2) : 
  (d1 + d2) / ((d1 / (b + c)) + (d2 / (b - c))) / b = 36 / 35 :=
by 
  sorry

end boat_ratio_l81_81214


namespace probability_of_non_defective_3_pencils_l81_81389

noncomputable def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

def probability_non_defective (total_pencils defective_pencils selected_pencils : ℕ) : ℚ :=
  let non_defective_pencils := total_pencils - defective_pencils
  let total_combinations := combination total_pencils selected_pencils
  let non_defective_combinations := combination non_defective_pencils selected_pencils
  non_defective_combinations / total_combinations

theorem probability_of_non_defective_3_pencils : 
  probability_non_defective 7 2 3 = 2 / 7 :=
by sorry

end probability_of_non_defective_3_pencils_l81_81389


namespace max_ratio_is_99_over_41_l81_81430

noncomputable def max_ratio (x y : ℕ) (h1 : x > y) (h2 : x + y = 140) : ℚ :=
  if h : y ≠ 0 then (x / y : ℚ) else 0

theorem max_ratio_is_99_over_41 : ∃ (x y : ℕ), x > y ∧ x + y = 140 ∧ max_ratio x y (by sorry) (by sorry) = (99 / 41 : ℚ) :=
by
  sorry

end max_ratio_is_99_over_41_l81_81430


namespace insulation_cost_of_rectangular_tank_l81_81700

theorem insulation_cost_of_rectangular_tank
  (l w h cost_per_sq_ft : ℕ)
  (hl : l = 4) (hw : w = 5) (hh : h = 3) (hc : cost_per_sq_ft = 20) :
  2 * l * w + 2 * l * h + 2 * w * h * 20 = 1880 :=
by
  sorry

end insulation_cost_of_rectangular_tank_l81_81700


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81635

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81635


namespace container_capacity_l81_81213

theorem container_capacity 
  (C : ℝ)
  (h1 : 0.75 * C - 0.30 * C = 45) :
  C = 100 := by
  sorry

end container_capacity_l81_81213


namespace number_of_ways_to_select_4_parents_with_one_couple_included_l81_81980

theorem number_of_ways_to_select_4_parents_with_one_couple_included 
  (P : Fin 12) 
  (C : Fin 6)
  (students : Fin 6) 
  (parents : students → Fin 2) 
  (couples : Finset (Fin 2 × Fin 2)) 
  (chosen_parents : Finset (Fin 4)) :
  (couples.card = 6) →
  (chosen_parents.card = 4) →
  (∃ pair ∈ couples, pair.fst ∈ chosen_parents ∧ pair.snd ∈ chosen_parents) →
  ∃ c : ℕ, c = 240 := 
by
  sorry

end number_of_ways_to_select_4_parents_with_one_couple_included_l81_81980


namespace max_PB_distance_l81_81884

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l81_81884


namespace well_performing_student_net_pay_l81_81198

def base_salary : ℤ := 25000
def bonus : ℤ := 5000
def tax_rate : ℝ := 0.13

def total_earnings (base_salary bonus : ℤ) : ℤ :=
  base_salary + bonus

def income_tax (total_earnings : ℤ) (tax_rate : ℝ) : ℤ :=
  total_earnings * (Real.toRat tax_rate)

def net_pay (total_earnings income_tax: ℤ) : ℤ :=
  total_earnings - income_tax

theorem well_performing_student_net_pay :
  net_pay (total_earnings base_salary bonus) (income_tax (total_earnings base_salary bonus) tax_rate) = 26100 := by
  sorry

end well_performing_student_net_pay_l81_81198


namespace largest_among_four_theorem_l81_81836

noncomputable def largest_among_four (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) : Prop :=
  (a^2 + b^2 > 1) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a^2 + b^2 > a)

theorem largest_among_four_theorem (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) :
  largest_among_four a b h1 h2 :=
sorry

end largest_among_four_theorem_l81_81836


namespace well_performing_student_take_home_pay_l81_81195

theorem well_performing_student_take_home_pay : 
  ∃ (base_salary bonus : ℕ) (income_tax_rate : ℝ),
      (base_salary = 25000) ∧ (bonus = 5000) ∧ (income_tax_rate = 0.13) ∧
      let total_earnings := base_salary + bonus in
      let income_tax := total_earnings * income_tax_rate in
      total_earnings - income_tax = 26100 :=
by
  use 25000
  use 5000
  use 0.13
  intros
  sorry

end well_performing_student_take_home_pay_l81_81195


namespace inscribed_rectangle_centers_l81_81287

theorem inscribed_rectangle_centers
  (A B C : Point)
  (h_acute : is_acute_angled_triangle A B C) :
  ∃ curvilinear_triangle : Set Point, 
    (∀ (P : Point), P ∈ curvilinear_triangle ↔ ∃ (K L M : Point), is_rectangle AKLM ∧ 
      (B ∈ seg KL ∧ C ∈ seg LM) ∧ center AKLM = P)  ∨
    (∃ arcs : Set Segment, (∀ (P : Point), P ∈ arcs ↔ ∃ (K L M : Point), is_rectangle AKLM ∧ 
      (B ∈ seg KL ∧ C ∈ seg LM) ∧ center AKLM = P)) :=
by sorry

end inscribed_rectangle_centers_l81_81287


namespace vector_dot_product_l81_81803

variables {V : Type*} [inner_product_space ℝ V]

variables (a b c : V)
  (h1 : ⟪a, b⟫ = 0)  -- a and b are perpendicular
  (h2 : ∥a∥ = 1)    -- a is a unit vector
  (h3 : ∥b∥ = 1)    -- b is a unit vector
  (h4 : ⟪c, a⟫ = -1)
  (h5 : ⟪c, b⟫ = -1)

theorem vector_dot_product (a b c : V) : 
  ⟪3 • a - b + 5 • c, b⟫ = -6 :=
by
  sorry

end vector_dot_product_l81_81803


namespace smallest_side_of_triangle_l81_81026

theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) 
  (hA : A = 60) (hC : C = 45) (hb : b = 4) (h_sum : A + B + C = 180) : 
  c = 4 * Real.sqrt 3 - 4 := 
sorry

end smallest_side_of_triangle_l81_81026


namespace chosen_number_probability_factorial_5_l81_81659

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81659


namespace sum_of_angles_equilateral_l81_81942

-- Definition of equilateral triangle and points division
structure EquilateralTriangle (A B C : Type) :=
  (triangle : Triangle A B C)
  (is_equilateral : triangle.isEquilateral)

structure PointsDivideSide (B C : Type) (n : ℕ) :=
  (points : Fin n → Point)
  (equal_division : ∀ i, segmentLength B (points i) = segmentLength (points i) (points (i + 1)))
  (equal_division_end : segmentLength (points (Fin.last n)) C = segmentLength B C / n)

-- Position of M on AC such that AM = BP1
structure PointOnSide (A C : Type) (P1 : Type) :=
  (M : Point)
  (equal_distance : segmentLength A M = segmentLength B P1)

-- Lean statement of the theorem
theorem sum_of_angles_equilateral {A B C : Type} [EquilateralTriangle A B C]
  (n : ℕ) (h_div : PointsDivideSide B C n) (h_M : PointOnSide A C (h_div.points 0)) :
  ∑ (i : Fin (n - 1)), ∠(A, h_div.points i, h_M.M) = 30 :=
  sorry

end sum_of_angles_equilateral_l81_81942


namespace price_of_baseball_cards_l81_81875

theorem price_of_baseball_cards 
    (packs_Digimon : ℕ)
    (price_per_pack : ℝ)
    (total_spent : ℝ)
    (total_cost_Digimon : ℝ) 
    (price_baseball_deck : ℝ) 
    (h1 : packs_Digimon = 4) 
    (h2 : price_per_pack = 4.45) 
    (h3 : total_spent = 23.86) 
    (h4 : total_cost_Digimon = packs_Digimon * price_per_pack) 
    (h5 : price_baseball_deck = total_spent - total_cost_Digimon) : 
    price_baseball_deck = 6.06 :=
sorry

end price_of_baseball_cards_l81_81875


namespace sunset_time_correct_l81_81938

theorem sunset_time_correct :
  let length_of_daylight := 11 * 60 + 10 in
  let sunrise_in_minutes := 7 * 60 + 30 in
  let sunset_in_minutes := sunrise_in_minutes + length_of_daylight in
  let sunset_hours := sunset_in_minutes / 60 in
  let sunset_minutes := sunset_in_minutes % 60 in
  sunset_hours = 18 ∧ sunset_minutes = 40 :=
by
  sorry

end sunset_time_correct_l81_81938


namespace digit_sum_inequality1_digit_sum_inequality2_l81_81554

noncomputable def digitSum (n : ℕ) : ℕ :=
  n.digits.sum

theorem digit_sum_inequality1 (k : ℕ) :
  digitSum(k) ≤ 8 * digitSum(8 * k) :=
sorry

theorem digit_sum_inequality2 (N : ℕ) :
  digitSum(N) ≤ 5 * digitSum(5^5 * N) :=
sorry

end digit_sum_inequality1_digit_sum_inequality2_l81_81554


namespace tiling_problem_l81_81757

theorem tiling_problem (m n : ℕ) : 
  (¬ ∃ board_cover : (fin m × fin n) → Prop, 
    ∀ pos, board_cover pos -> 
      pos.1 < m ∧ pos.2 < n ∧
      (∃ i : fin 3, pos.1 + i.val < m ∨ ∃ j : fin 5, pos.2 + j.val < n)) ↔ 
  (m = 4 ∧ n = 4) ∨ 
  (m = 2 ∧ n = 2) ∨ 
  (m = 2 ∧ n = 4) ∨ 
  (m = 2 ∧ n = 7) ∨ 
  (m = 1 ∧ ∃ k : ℕ, n = 3 * k + 1) ∨ 
  (m = 1 ∧ ∃ k : ℕ, n = 3 * k + 2) :=
sorry

end tiling_problem_l81_81757


namespace prob_factorial_5_l81_81668

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81668


namespace area_of_square_II_l81_81964

theorem area_of_square_II (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let diagonal_I := (sqrt 3) * a * b
  let side_I := diagonal_I / (sqrt 2)
  let area_I := (side_I ^ 2)
  let area_II := 3 * area_I
  area_II = (9 * (a * b) ^ 2) / 2 :=
by
  sorry

end area_of_square_II_l81_81964


namespace knights_liars_puzzle_solved_l81_81460

def is_knight (board : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop := board pos

def is_liar (board : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop := ¬ board pos

def adjacent_positions (pos : ℕ × ℕ) : List (ℕ × ℕ) :=
  let (x, y) := pos
  [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def exactly_one_knight_adjacent (board : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop :=
  (adjacent_positions pos).countp (is_knight board) = 1

def knight_condition (board : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop :=
  is_knight board pos → exactly_one_knight_adjacent board pos

def liar_condition (board : ℕ × ℕ → bool) (pos : ℕ × ℕ) : Prop :=
  is_liar board pos → ¬ exactly_one_knight_adjacent board pos

def valid_board (board : ℕ × ℕ → bool) : Prop :=
  (∃ (knights liars : List (ℕ × ℕ)),
    knights.length = 8 ∧ liars.length = 8 ∧
    List.all knights (λ pos => pos.1 < 4 ∧ pos.2 < 4 ∧ is_knight board pos) ∧
    List.all liars (λ pos => pos.1 < 4 ∧ pos.2 < 4 ∧ is_liar board pos) ∧
    List.all knights (knight_condition board) ∧
    List.all liars (liar_condition board))

theorem knights_liars_puzzle_solved : ∃ (board : ℕ × ℕ → bool), valid_board board :=
by
  sorry

end knights_liars_puzzle_solved_l81_81460


namespace minimum_value_y_range_of_a_l81_81205

-- Part 1
theorem minimum_value_y (x : ℝ) (h : x > -1) : 
  let y := (x^2 + 7 * x + 10) / (x + 1) 
  in y ≥ 9 :=
sorry

-- Part 2
theorem range_of_a (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 8 * y - x * y = 0) 
  (a : ℝ) :
  a ≤ x + y → a ≤ 18 :=
sorry

end minimum_value_y_range_of_a_l81_81205


namespace valid_lineups_count_l81_81122

theorem valid_lineups_count {n : ℕ} (George Alex Sam : ℕ) (Hoopers : Fin n) 
  (total_players := 15)
  (players := Finset.range total_players)
  (lineup_size := 6)
  (without_George_Alex := players.erase George ∩ players.erase Alex)
  (case1 := players.erase Alex ∩ players.erase Sam)
  (case2 := players.erase George)
  (case3 := without_George_Alex) :
  (Finset.card (Finset.filter (λ (x : Finset (Fin n)), George ∈ x ∧ Alex ∉ x ∧ Sam ∉ x) (Finset.powerset_len lineup_size players)) +
  Finset.card (Finset.filter (λ (x : Finset (Fin n)), Alex ∈ x ∧ George ∉ x) (Finset.powerset_len lineup_size players)) +
  Finset.card (Finset.filter (λ (x : Finset (Fin n)), George ∉ x ∧ Alex ∉ x) (Finset.powerset_len lineup_size players))) = 3795 :=
  sorry

end valid_lineups_count_l81_81122


namespace solution_set_l81_81441

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
variable {x : ℝ}

-- Assume f is differentiable and f' is its derivative
def differentiable_f (x : ℝ) := ∃ f', differentiable ℝ f' x

-- Conditions
axiom diff_f : differentiable_f x
axiom f'_def : ∀ x, deriv f x = f'(x)
axiom condition : ∀ x < 0, (2 * f(x) + x * f'(x)) > 0

-- Prove the solution set of the inequality (x+2016)^2 * f(x+2016) - 9 * f(-3) > 0 is (-∞, -2019)
theorem solution_set :
  {x : ℝ | (x + 2016) ^ 2 * f (x + 2016) - 9 * f (-3) > 0} = {x : ℝ | x < -2019} :=
sorry

end solution_set_l81_81441


namespace pyramid_volume_l81_81125

noncomputable def volume_of_pyramid (a α β : ℝ) : ℝ :=
  (a^3 * Real.sin (α / 2) * Real.tan β) / 6

theorem pyramid_volume (a α β : ℝ) : (volume_of_pyramid a α β ) = (a^3 * Real.sin (α / 2) * Real.tan β) / 6 :=
by sorry

end pyramid_volume_l81_81125


namespace union_of_A_and_B_l81_81360

def A : set ℤ := {-1, 0, 1}
def B : set ℤ := {-2, -1, 0}

theorem union_of_A_and_B :
  A ∪ B = {-2, -1, 0, 1} := 
by 
  sorry

end union_of_A_and_B_l81_81360


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81638

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81638


namespace probability_factor_of_5_factorial_l81_81693

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81693


namespace power_function_value_l81_81328

theorem power_function_value (f : ℝ → ℝ) (α : ℝ) (h1 : ∀ x, f(x) = x^α) (h2 : f 4 = 0.5) :
  f (1/4) = 2 :=
sorry

end power_function_value_l81_81328


namespace calc_sqrt_mult_l81_81525

theorem calc_sqrt_mult : 
  ∀ (a b c : ℕ), a = 256 → b = 64 → c = 16 → 
  (nat.sqrt (nat.sqrt a) * nat.cbrt b * nat.sqrt c = 64) :=
by 
  intros a b c h1 h2 h3
  rw [h1, nat.sqrt_eq, nat.sqrt_eq, h2, nat.cbrt_eq, h3, nat.sqrt_eq]
  sorry

end calc_sqrt_mult_l81_81525


namespace probability_is_13_over_30_l81_81682

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81682


namespace find_f_9_f_27_find_range_a_l81_81800

variables {α : Type*} [ordered_semiring α] {f : α → α}

-- Given conditions
axiom increasing (f_increasing : ∀ {x y : α}, (0 < x ∧ 0 < y ∧ x < y) → f(x) < f(y))
axiom functional_eq (f_mul_eq_add : ∀ {x y : α}, 0 < x → 0 < y → f(x * y) = f(x) + f(y))
axiom specific_value (f_three : f(3) = 1)

-- We prove that f(9) = 2 and f(27) = 3
theorem find_f_9_f_27 : 
  f(9) = 2 ∧ f(27) = 3 :=
sorry

-- Proving the range of a such that f(3) + f(a-8) < 2
theorem find_range_a (a : α) (h₁ : f(3) + f(a - 8) < 2) : 
  8 < a ∧ a < 11 :=
sorry

end find_f_9_f_27_find_range_a_l81_81800


namespace factor_probability_l81_81591

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81591


namespace min_value_frac_l81_81129

open Real

/-- Lean theorem formalizing the problem statement -/
theorem min_value_frac {a m n : ℝ} (h_a_pos : a > 0) (h_a_ne : a ≠ 1) (h_m_pos : m > 0) (h_n_pos : n > 0) 
    (h_func : (∀ x, y = log a (x + 3) - 1) (h_point : (x, y) = (-2, -1)) (h_line : m * -2 + n * -1 + 1 = 0)) :
    (min_val : ℝ) := by
  sorry

end min_value_frac_l81_81129


namespace mean_proportional_AC_is_correct_l81_81300

-- Definitions based on conditions
def AB := 4
def BC (AC : ℝ) := AB - AC

-- Lean theorem
theorem mean_proportional_AC_is_correct (AC : ℝ) :
  AC > 0 ∧ AC^2 = AB * BC AC ↔ AC = 2 * Real.sqrt 5 - 2 := 
sorry

end mean_proportional_AC_is_correct_l81_81300


namespace josiah_yards_per_game_l81_81077

open Nat

theorem josiah_yards_per_game :
  (let malik_yards_per_game := 18 in
  let malik_games := 4 in
  let darnell_yards_per_game := 11 in
  let darnell_games := 4 in
  let total_yards := 204 in
  let josiah_games := 4 in
  let malik_total_yards := malik_yards_per_game * malik_games in
  let darnell_total_yards := darnell_yards_per_game * darnell_games in
  let josiah_total_yards := total_yards - (malik_total_yards + darnell_total_yards) in
  let josiah_yards_per_game := josiah_total_yards / josiah_games in
  josiah_yards_per_game = 22) :=
by
  sorry

end josiah_yards_per_game_l81_81077


namespace chosen_number_probability_factorial_5_l81_81657

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81657


namespace smallest_sum_B_c_l81_81012

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end smallest_sum_B_c_l81_81012


namespace probability_factor_of_120_in_range_l81_81617

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81617


namespace hotel_weekly_loss_l81_81221

theorem hotel_weekly_loss :
  let operations_expenses := 5000 in
  let meetings_income := 5 / 8 * operations_expenses in
  let events_income := 3 / 10 * operations_expenses in
  let rooms_income := 11 / 20 * operations_expenses in
  let total_income := meetings_income + events_income + rooms_income in
  let taxes := 1200 in
  let employee_salaries := 2500 in
  let total_expenses := operations_expenses + taxes + employee_salaries in
  let profit_or_loss := total_income - total_expenses in
  profit_or_loss = -1325 :=
by
  sorry

end hotel_weekly_loss_l81_81221


namespace false_statement_is_propC_l81_81017

def isosceles_pyramid : Prop := ∀ p : Pyramid, (∀ e1 e2 : Edge, e1.length = e2.length ∧ e1.is_lateral ∧ e2.is_lateral → e1.is_leg ∧ e2.is_leg)

-- Proposition A: The angles between the legs and the base of an isosceles pyramid are all equal
def propA (p : Pyramid) : Prop := 
  ∀ (leg1 leg2 : Edge) (base1 base2 : Face),
    leg1.is_leg ∧ leg2.is_leg ∧ leg1.angleWith base1 = leg2.angleWith base2

-- Proposition B: The base quadrilateral of an isosceles pyramid must have a circumscribed circle
def propB (p : Pyramid) : Prop := 
  ∃ circ_circle : Circle, ∀ v : Vertex, v.is_base_vertex → v ∈ circ_circle

-- Proposition C: The dihedral angles formed by the lateral faces and the base of an isosceles pyramid are all equal or supplementary
def propC (p : Pyramid) : Prop := 
  ∀ (lat_face1 lat_face2 base_face : Face),
    lat_face1.dihedralAngleWith base_face = lat_face2.dihedralAngleWith base_face ∨ 
    lat_face1.dihedralAngleWith base_face + lat_face2.dihedralAngleWith base_face = π

-- Proposition D: All vertices of an isosceles pyramid must lie on the same sphere
def propD (p : Pyramid) : Prop := 
  ∃ circ_sphere : Sphere, ∀ v : Vertex, v ∈ p.vertices → v ∈ circ_sphere

-- The mathematical problem to prove
theorem false_statement_is_propC : ∀ p : Pyramid, isosceles_pyramid p → ¬ propC p :=
by
  sorry

end false_statement_is_propC_l81_81017


namespace evaluate_expression_is_41_l81_81273

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end evaluate_expression_is_41_l81_81273


namespace range_of_a_l81_81839

theorem range_of_a (a : ℝ) :
  ( ∀ x : ℝ, 3^(x^2 - 2*a*x) > (1/3)^(x + 1) ) ↔ -1/2 < a ∧ a < 3/2 :=
sorry

end range_of_a_l81_81839


namespace janice_homework_time_l81_81872

variable (H : ℝ)
variable (cleaning_room walk_dog take_trash : ℝ)

-- Conditions from the problem translated directly
def cleaning_room_time : cleaning_room = H / 2 := sorry
def walk_dog_time : walk_dog = H + 5 := sorry
def take_trash_time : take_trash = H / 6 := sorry
def total_time_before_movie : 35 + (H + cleaning_room + walk_dog + take_trash) = 120 := sorry

-- The main theorem to prove
theorem janice_homework_time (H : ℝ)
        (cleaning_room : ℝ := H / 2)
        (walk_dog : ℝ := H + 5)
        (take_trash : ℝ := H / 6) :
    H + cleaning_room + walk_dog + take_trash + 35 = 120 → H = 30 :=
by
  sorry

end janice_homework_time_l81_81872


namespace parabola_directrix_l81_81023

theorem parabola_directrix (p : ℝ) (h : p > 0) (h_directrix : -p / 2 = -4) : p = 8 :=
by
  sorry

end parabola_directrix_l81_81023


namespace polynomial_sum_of_coefficients_l81_81297

theorem polynomial_sum_of_coefficients :
  let p := (x^2 + 1) * (2 * x + 1)^9
  let q := Σ (i : ℕ) in finset.range 12, a i * (x + 2)^i
  ∀ (a : ℕ → ℤ), p = q → (Σ (i : ℕ) in finset.range 12, a i) = -2 :=
begin
  sorry
end

end polynomial_sum_of_coefficients_l81_81297


namespace max_value_is_one_l81_81066

open Complex

theorem max_value_is_one (α β : ℂ) (hβ : β^2 = 1) (hαβ : ¬ (conjugate α * β = 1)) :
  ∃ M : ℝ, (∀ β : ℂ, (β^2 = 1) → ¬ (conjugate α * β = 1) →
                 abs ((β - α) / (1 - conjugate α * β)) ≤ M) ∧ M = 1 :=
by
  existsi 1
  split
  { intros β hβ' hαβ'
    sorry }
  { refl }

end max_value_is_one_l81_81066


namespace perpendicular_condition_l81_81773

variable (λ : ℝ)
def a : ℝ × ℝ := (3, λ)
def b : ℝ × ℝ := (λ - 1, 2)

theorem perpendicular_condition (λ : ℝ) :
  (a λ).1 * (b λ).1 + (a λ).2 * (b λ).2 = 0 ↔ λ = 3 / 5 := 
by
  unfold a b
  sorry

end perpendicular_condition_l81_81773


namespace sphere_diameter_triple_volume_l81_81759

theorem sphere_diameter_triple_volume :
  let r₁ := 7 in
  let V₁ := (4 / 3) * Real.pi * r₁ ^ 3 in
  let V₂ := 3 * V₁ in
  let r₂ := (3 * r₁^3).cbrt in
  let d := 2 * r₂ in
  let (c, d) := if cbrt_factors := 14 * Real.root3 3 then (14, 3) else (0, 0) in
  c + d = 17 :=
by sorry

end sphere_diameter_triple_volume_l81_81759


namespace area_ratio_DEY_ECY_l81_81048

noncomputable def triangle_DEY_ECY_area_ratio
  (DE EC DC : ℕ) (bisects_angle : Prop) : ℚ :=
  if bisects_angle ∧ DE = 36 ∧ EC = 32 ∧ DC = 40 then (9 / 8) else 0

theorem area_ratio_DEY_ECY {DE EC DC : ℕ} {bisects_angle : Prop}
  (h_conditions : bisects_angle ∧ DE = 36 ∧ EC = 32 ∧ DC = 40) :
  triangle_DEY_ECY_area_ratio DE EC DC bisects_angle = 9 / 8 :=
  by simp [triangle_DEY_ECY_area_ratio, h_conditions]; sorry

end area_ratio_DEY_ECY_l81_81048


namespace part_one_part_two_l81_81419

def M (n : ℤ) : ℤ := n - 3
def M_frac (n : ℚ) : ℚ := - (1 / n^2)

theorem part_one 
    : M 28 * M_frac (1/5) = -1 :=
by {
  sorry
}

theorem part_two 
    : -1 / M 39 / (- M_frac (1/6)) = -1 :=
by {
  sorry
}

end part_one_part_two_l81_81419


namespace coffee_ratio_is_one_to_five_l81_81739

-- Given conditions
def thermos_capacity : ℕ := 20 -- capacity in ounces
def times_filled_per_day : ℕ := 2
def school_days_per_week : ℕ := 5
def new_weekly_coffee_consumption : ℕ := 40 -- in ounces

-- Definitions based on the conditions
def old_daily_coffee_consumption := thermos_capacity * times_filled_per_day
def old_weekly_coffee_consumption := old_daily_coffee_consumption * school_days_per_week

-- Theorem: The ratio of the new weekly coffee consumption to the old weekly coffee consumption is 1:5
theorem coffee_ratio_is_one_to_five : 
  new_weekly_coffee_consumption / old_weekly_coffee_consumption = 1 / 5 := 
by
  -- Proof is omitted
  sorry

end coffee_ratio_is_one_to_five_l81_81739


namespace deepak_age_l81_81245

theorem deepak_age (A D : ℕ)
  (h1 : A / D = 2 / 3)
  (h2 : A + 5 = 25) :
  D = 30 := 
by
  sorry

end deepak_age_l81_81245


namespace find_x_l81_81143

noncomputable def base23_repr (a : ℕ) (m : ℕ) : ℕ :=
  a * (23^0 + 23^1 + ... + 23^(2*m-1))

theorem find_x (a b x : ℕ) (m : ℕ) (h1 : a * (23^0 + 23^1 + ... + 23^(2*m-1)) = x)
  (h2 : x^2 = b * (1 + 23^(2*(2*m)-1))) :
  x = 13 * (23^0 + 23^1 + ... + 23^(2*m-1)) := by
  sorry

end find_x_l81_81143


namespace liza_final_balance_l81_81091

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end liza_final_balance_l81_81091


namespace max_min_x_plus_y_on_circle_l81_81865

-- Define the conditions
def polar_eq (ρ θ : Real) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the standard form of the circle
def circle_eq (x y : Real) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the parametric equations of the circle
def parametric_eq (α : Real) (x y : Real) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos α ∧ y = 2 + Real.sqrt 2 * Real.sin α

-- Define the problem in Lean
theorem max_min_x_plus_y_on_circle :
  (∀ (ρ θ : Real), polar_eq ρ θ → circle_eq (ρ * Real.cos θ) (ρ * Real.sin θ)) →
  (∀ (α : Real), parametric_eq α (2 + Real.sqrt 2 * Real.cos α) (2 + Real.sqrt 2 * Real.sin α)) →
  (∀ (P : Real × Real), circle_eq P.1 P.2 → 2 ≤ P.1 + P.2 ∧ P.1 + P.2 ≤ 6) :=
by
  intros hpolar hparam P hcircle
  sorry

end max_min_x_plus_y_on_circle_l81_81865


namespace simplify_cbrt_8000_eq_21_l81_81181

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l81_81181


namespace equal_distances_from_I_l81_81407

variables {A B C D E F I M N D' : Type*}
variables {triangle_ABC : Triangle A B C}
variables {incircle_I : Circle I}
variables {tangent_D : Point D}
variables {tangent_E : Point E}
variables {tangent_F : Point F}
variables {diameter_DD' : Line D D'}
variables {perpendicular_AD' : Line A D'}
variables {intersect_perpendicular_I_DE : Line I E}
variables {intersect_perpendicular_I_DF : Line I F}

-- Assume the necessary conditions
axiom incircle_tangent_BC : tangent_D ∈ incircle_I ∧ tangent_D lies on BC
axiom incircle_tangent_CA : tangent_E ∈ incircle_I ∧ tangent_E lies on CA
axiom incircle_tangent_AB : tangent_F ∈ incircle_I ∧ tangent_F lies on AB
axiom diameter_by_D : diameter_DD' = diameter through D
axiom perpendicular_line_AD' : is_perpendicular AD' I

axiom perpendicular_intersection_DE : intersect_perpendicular_I_DE ∩ DE = M
axiom perpendicular_intersection_DF : intersect_perpendicular_I_DF ∩ DF = N

-- Prove that IM = IN given the conditions
theorem equal_distances_from_I (triangle_ABC : Triangle A B C)
  (incircle_I : Circle I)
  (tangent_D : Point D) (tangent_E : Point E) (tangent_F : Point F)
  (diameter_DD' : Line D D') (perpendicular_AD' : Line A D')
  (intersect_perpendicular_I_DE intersect_perpendicular_I_DF : Point I)
  (incircle_tangent_BC : tangent_D ∈ incircle_I ∧ tangent_D lies on BC)
  (incircle_tangent_CA : tangent_E ∈ incircle_I ∧ tangent_E lies on CA)
  (incircle_tangent_AB : tangent_F ∈ incircle_I ∧ tangent_F lies on AB)
  (diameter_by_D : diameter_DD' = diameter through D)
  (perpendicular_line_AD' : is_perpendicular AD' I)
  (perpendicular_intersection_DE : intersect_perpendicular_I_DE ∩ DE = M)
  (perpendicular_intersection_DF : intersect_perpendicular_I_DF ∩ DF = N) :
  dist I M = dist I N :=
sorry

end equal_distances_from_I_l81_81407


namespace duration_of_loan_l81_81948

namespace SimpleInterest

variables (P SI R : ℝ) (T : ℝ)

-- Defining the conditions
def principal := P = 1500
def simple_interest := SI = 735
def rate := R = 7 / 100

-- The question: Prove the duration (T) of the loan
theorem duration_of_loan (hP : principal P) (hSI : simple_interest SI) (hR : rate R) :
  T = 7 :=
sorry

end SimpleInterest

end duration_of_loan_l81_81948


namespace smallest_sum_B_c_l81_81011

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end smallest_sum_B_c_l81_81011


namespace circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l81_81264

-- Circle 1 with center (8, -3) and passing through point (5, 1)
theorem circle_centered_at_8_neg3_passing_through_5_1 :
  ∃ r : ℝ, (r = 5) ∧ ((x - 8: ℝ)^2 + (y + 3)^2 = r^2) := by
  sorry

-- Circle passing through points A(-1, 5), B(5, 5), and C(6, -2)
theorem circle_passing_through_ABC :
  ∃ D E F : ℝ, (D = -4) ∧ (E = -2) ∧ (F = -20) ∧
    ( ∀ (x : ℝ) (y : ℝ), (x = -1 ∧ y = 5) 
      ∨ (x = 5 ∧ y = 5) 
      ∨ (x = 6 ∧ y = -2) 
      → (x^2 + y^2 + D*x + E*y + F = 0)) := by
  sorry

end circle_centered_at_8_neg3_passing_through_5_1_circle_passing_through_ABC_l81_81264


namespace solve_l81_81815

noncomputable def hyperbola_asymptote_problem :=
∀ a b : ℝ, (0 < a) → (0 < b) →
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 + y^2 - 6 * x + 5 = 0) →
   (∃ k : ℝ, (k = 2 * real.sqrt 5 / 5) ∧ (∀ x : ℝ, y = k * x ∨ y = -k * x)))

namespace hyperbola_asymptote_problem

open Real

theorem solve :
hyperbola_asymptote_problem :=
sorry

end hyperbola_asymptote_problem

end solve_l81_81815


namespace distance_origin_point1_distance_origin_midpoint_l81_81393

noncomputable def distance (p1 p2: (ℝ × ℝ)) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def point1 : ℝ × ℝ := (8, -15)
def origin : ℝ × ℝ := (0, 0)

def midpoint (p1 p2: (ℝ × ℝ)) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def midpoint_origin_point1 := midpoint origin point1

theorem distance_origin_point1 : distance origin point1 = 17 :=
by sorry

theorem distance_origin_midpoint : distance origin midpoint_origin_point1 = 8.5 :=
by sorry

end distance_origin_point1_distance_origin_midpoint_l81_81393


namespace prob_factorial_5_l81_81674

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81674


namespace inequality_amgm_l81_81437

variable {a b c : ℝ}

theorem inequality_amgm (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) : 
  (1 / 2) * ((a - b)^2 + (b - c)^2 + (c - a)^2) <= a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) ∧ 
  a^2 + b^2 + c^2 - 3 * (a^2 * b^2 * c^2)^(1 / 3) <= (a - b)^2 + (b - c)^2 + (c - a)^2 := 
by 
  sorry

end inequality_amgm_l81_81437


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81637

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81637


namespace only_PropositionB_is_correct_l81_81370

-- Define propositions as functions for clarity
def PropositionA (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) : Prop :=
  (1 / a) > (1 / b)

def PropositionB (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : Prop :=
  a ^ 3 < a

def PropositionC (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  (b + 1) / (a + 1) < b / a

def PropositionD (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : Prop :=
  c * b^2 < a * b^2

-- The main theorem stating that the only correct proposition is Proposition B
theorem only_PropositionB_is_correct :
  (∀ a b : ℝ, (a * b ≠ 0 ∧ a < b → ¬ PropositionA a b (a * b ≠ 0) (a < b))) ∧
  (∀ a : ℝ, (0 < a ∧ a < 1 → PropositionB a (0 < a) (a < 1))) ∧
  (∀ a b : ℝ, (a > b ∧ b > 0 → ¬ PropositionC a b (a > b) (b > 0))) ∧
  (∀ a b c : ℝ, (c < b ∧ b < a ∧ a * c < 0 → ¬ PropositionD a b c (c < b) (b < a) (a * c < 0))) :=
by
  -- Proof of the theorem
  sorry

end only_PropositionB_is_correct_l81_81370


namespace floor_multiple_of_floor_l81_81878

noncomputable def r : ℝ := sorry

theorem floor_multiple_of_floor (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : ∃ k, n = k * m) (hr : r ≥ 1) 
  (floor_multiple : ∀ (m n : ℕ), (∃ k : ℕ, n = k * m) → ∃ l, ⌊n * r⌋ = l * ⌊m * r⌋) :
  ∃ k : ℤ, r = k := 
sorry

end floor_multiple_of_floor_l81_81878


namespace radical_product_is_64_l81_81529

theorem radical_product_is_64:
  real.sqrt (16:ℝ) * real.sqrt (real.sqrt 256) * real.n_root 64 3 = 64 :=
sorry

end radical_product_is_64_l81_81529


namespace find_fifth_score_l81_81873

-- Define the known scores
def score1 : ℕ := 90
def score2 : ℕ := 93
def score3 : ℕ := 85
def score4 : ℕ := 97

-- Define the average of all scores
def average : ℕ := 92

-- Define the total number of scores
def total_scores : ℕ := 5

-- Define the total sum of all scores using the average
def total_sum : ℕ := total_scores * average

-- Define the sum of the four known scores
def known_sum : ℕ := score1 + score2 + score3 + score4

-- Define the fifth score
def fifth_score : ℕ := 95

-- Theorem statement: The fifth score plus the known sum equals the total sum.
theorem find_fifth_score : fifth_score + known_sum = total_sum := by
  sorry

end find_fifth_score_l81_81873


namespace smallest_square_area_l81_81564

theorem smallest_square_area (a b c d : ℕ) (hsquare : ∃ s : ℕ, s ≥ a + c ∧ s * s = a * b + c * d) :
    (a = 3) → (b = 5) → (c = 4) → (d = 6) → ∃ s : ℕ, s * s = 49 :=
by
  intros h1 h2 h3 h4
  cases hsquare with s hs
  use s
  -- Here we need to ensure s * s = 49
  sorry

end smallest_square_area_l81_81564


namespace num_factors_of_2_pow_20_minus_1_l81_81364

/-- 
Prove that the number of positive two-digit integers 
that are factors of \(2^{20} - 1\) is 5.
-/
theorem num_factors_of_2_pow_20_minus_1 :
  ∃ (n : ℕ), n = 5 ∧ (∀ (k : ℕ), k ∣ (2^20 - 1) → 10 ≤ k ∧ k < 100 → k = 33 ∨ k = 15 ∨ k = 27 ∨ k = 41 ∨ k = 45) 
  :=
sorry

end num_factors_of_2_pow_20_minus_1_l81_81364


namespace probability_factor_of_120_in_range_l81_81611

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81611


namespace sum_last_two_digits_fibonacci_factorial_series_l81_81734

theorem sum_last_two_digits_fibonacci_factorial_series :
  let fib_factorial n := (factorial n) % 100,
      series := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
      last_two_digits := series.map fib_factorial
  in last_two_digits.sum % 100 = 5 :=
by 
  -- Need to define fib_factorial, map over the series,
  -- calculate the sum of last two digits, and prove
  sorry

end sum_last_two_digits_fibonacci_factorial_series_l81_81734


namespace probability_of_spade_then_king_l81_81517

theorem probability_of_spade_then_king :
  ( (24 / 104) * (8 / 103) + (2 / 104) * (7 / 103) ) = 103 / 5356 :=
sorry

end probability_of_spade_then_king_l81_81517


namespace paul_total_vertical_distance_l81_81457

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end paul_total_vertical_distance_l81_81457


namespace theta_plus_phi_eq_l81_81395

-- Definitions corresponding to the problem conditions
variables {A B C D M N : Type} [decidable_eq A] [decidable_eq B] [decidable_eq C]
angles : A → B → C → ℝ
acute_ABC : A → B → C → Prop
bisector_of_A : A → B → C → D → Prop
circle_center_B_radius_BD : B → D → Prop
circle_center_C_radius_CD : C → D → Prop
intersect_side_AB_at_M : B → D → M → Prop
intersect_side_AC_at_N : C → D → N → Prop
BM_eq_BD_and_BMD_eq_2theta : B → M → D → ℝ → Prop
CN_eq_CD_and_CND_eq_2phi : C → N → D → ℝ → Prop

theorem theta_plus_phi_eq :
  acute_ABC A B C →
  bisector_of_A A B C D →
  circle_center_B_radius_BD B D →
  intersect_side_AB_at_M B D M →
  circle_center_C_radius_CD C D →
  intersect_side_AC_at_N C D N →
  BM_eq_BD_and_BMD_eq_2theta B M D (2 * θ) →
  CN_eq_CD_and_CND_eq_2phi C N D (2 * φ) →
  θ + φ = 90 - (1 / 4) * (angles A B C) :=
sorry

end theta_plus_phi_eq_l81_81395


namespace bounded_sequence_l81_81931

def is_triplet (x y z : ℕ) : Prop :=
  y = (x + z) / 2

def construct_sequence : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := Nat.find (λ m, m > construct_sequence (n+1) ∧ ∀ i j, i < j ∧ j < n + 2 → ¬ is_triplet (construct_sequence i) (construct_sequence j) m)

theorem bounded_sequence :
  construct_sequence 2023 ≤ 100000 :=
sorry

end bounded_sequence_l81_81931


namespace arithmetic_seq_general_term_l81_81323

theorem arithmetic_seq_general_term {d : ℝ} (h_d : d ≠ 0) 
  {a : ℕ → ℝ} (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_S : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_seq_arith : ∀ n, sqrt (8 * S (n + 1) + 2 * (n + 1)) - sqrt (8 * S n + 2 * n) = d) :
  ∀ n, a n = 4 * n - 9 / 4 := 
sorry

end arithmetic_seq_general_term_l81_81323


namespace option_A_option_B_option_D_l81_81340

-- Definitions of sequences
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a_1 + n * d

def geometric_seq (b_1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  b_1 * q ^ n

-- Option A: Prove that there exist d and q such that a_n = b_n
theorem option_A : ∃ (d q : ℤ), ∀ (a_1 b_1 : ℤ) (n : ℕ), 
  (arithmetic_seq a_1 d n = geometric_seq b_1 q n) := sorry

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Option B: Prove the differences form an arithmetic sequence
theorem option_B (a_1 : ℤ) (d : ℤ) :
  ∀ n k : ℕ, k > 0 → 
  (sum_arithmetic_seq a_1 d ((k + 1) * n) - sum_arithmetic_seq a_1 d (k * n) =
   (sum_arithmetic_seq a_1 d n + k * n * n * d)) := sorry

-- Option D: Prove there exist real numbers A and a such that A * a^a_n = b_n
theorem option_D (a_1 : ℤ) (d : ℤ) (b_1 : ℤ) (q : ℤ) :
  ∀ n : ℕ, b_1 > 0 → q > 0 → 
  ∃ A a : ℝ, A * a^ (arithmetic_seq a_1 d n) = (geometric_seq b_1 q n) := sorry

end option_A_option_B_option_D_l81_81340


namespace problem_1_problem_2_l81_81443

variables (A B C : ℝ) (a b c : ℝ)
variable (h1 : (Real.sin C + Real.sin B) * (c - b) = a * (Real.sin A - Real.sin B))
variable (h2 : C = Real.pi / 3)
variable (CD AD BD : ℝ)
variables (h3 : CD = 2) (h4 : AD = 2 * BD)

noncomputable theory

def measure_angle_C : Prop :=
  C = Real.pi / 3

def area_triangle_ABC : Prop :=
  let geometric_area := (sqrt 3 / 2) * a * b in
  let target_area := sqrt 3 * (Real.pow (a + b + c) 2) / 4 in
  target_area = 3 * (sqrt 3) / 2

theorem problem_1 : measure_angle_C A B C a b c h1 :=
by sorry

theorem problem_2 : area_triangle_ABC A B C a b c CD AD BD h3 h4 :=
by sorry

end problem_1_problem_2_l81_81443


namespace calc_sqrt_mult_l81_81523

theorem calc_sqrt_mult : 
  ∀ (a b c : ℕ), a = 256 → b = 64 → c = 16 → 
  (nat.sqrt (nat.sqrt a) * nat.cbrt b * nat.sqrt c = 64) :=
by 
  intros a b c h1 h2 h3
  rw [h1, nat.sqrt_eq, nat.sqrt_eq, h2, nat.cbrt_eq, h3, nat.sqrt_eq]
  sorry

end calc_sqrt_mult_l81_81523


namespace operations_to_achieve_perimeter_less_than_one_l81_81976

-- The initial perimeter of the rectangle
def initial_perimeter: ℝ := 32

-- The perimeter after n operations
def perimeter_after_n_operations (n : ℕ) : ℝ := initial_perimeter / (2 ^ n)

-- The main theorem stating the required number of operations
theorem operations_to_achieve_perimeter_less_than_one : ∃ n : ℕ, n = 11 ∧ perimeter_after_n_operations n < 1 := 
by
  sorry

end operations_to_achieve_perimeter_less_than_one_l81_81976


namespace three_digit_numbers_div_by_17_l81_81829

theorem three_digit_numbers_div_by_17 : ∃ n : ℕ, n = 53 ∧ 
  let min_k := Nat.ceil (100 / 17)
  let max_k := Nat.floor (999 / 17)
  min_k = 6 ∧ max_k = 58 ∧ (max_k - min_k + 1) = n :=
by
  sorry

end three_digit_numbers_div_by_17_l81_81829


namespace total_computers_sold_l81_81084

theorem total_computers_sold (T : ℕ) (h_half_sales_laptops : 2 * T / 2 = T)
        (h_third_sales_netbooks : 3 * T / 3 = T)
        (h_desktop_sales : T - T / 2 - T / 3 = 12) : T = 72 :=
by
  sorry

end total_computers_sold_l81_81084


namespace f_even_f_mono_decreasing_l81_81807

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 1 / (2 * x^2)

theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  simp [f, pow_two, neg_sq]

theorem f_mono_decreasing : ∀ ⦃x1 x2 : ℝ⦄, 0 < x1 → x1 < x2 → x2 < real.sqrt 2 / 2 → f x1 > f x2 := by
  sorry

end f_even_f_mono_decreasing_l81_81807


namespace equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l81_81150

-- Definitions for supply and demand functions
def Qs (p : ℝ) : ℝ := 2 + 8 * p
def Qd (p : ℝ) : ℝ := -2 * p + 12

-- Equilibrium without subsidy
theorem equilibrium_price_without_subsidy : (∃ p q, Qs p = q ∧ Qd p = q ∧ p = 1 ∧ q = 10) :=
sorry

-- New supply function with subsidy
def Qs_with_subsidy (p : ℝ) : ℝ := 10 + 8 * p

-- Increase in quantity sold due to subsidy
theorem increase_in_quantity_due_to_subsidy : 
  (∃ Δq, Δq = Qd 0.2 - Qd 1 ∧ Δq = 1.6) :=
sorry

end equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l81_81150


namespace max_tan_A_theorem_l81_81047

noncomputable def max_tan_A (AB BC : ℝ) (h1 : AB = 20) (h2 : BC = 15) : ℝ :=
  (3 * real.sqrt 7) / 7

theorem max_tan_A_theorem : max_tan_A 20 15 (by rfl) (by rfl) = (3 * real.sqrt 7) / 7 :=
sorry

end max_tan_A_theorem_l81_81047


namespace distance_PF_equilateral_l81_81063

-- Given conditions as definitions
def F : ℝ × ℝ := (1/2, 0)
def directrix l : ℝ := -1/2
def parabola (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 2 * P.1
def lies_on_directrix (Q : ℝ × ℝ) : Prop := Q.1 = -1/2
def parallel_to_x_axis (PQ : ℝ × ℝ) : Prop := PQ.2 = 0
def equidistant (PQ QF : ℝ) : Prop := PQ = QF

-- The key property we want to prove
theorem distance_PF_equilateral (P Q : ℝ × ℝ) (hP : parabola P) (hQ : lies_on_directrix Q) (h1 : parallel_to_x_axis (P.1 - Q.1, 0)) (h2 : equidistant ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ((Q.1 - F.1)^2 + (Q.2 - F.2)^2)) : 
  ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 2^2 :=
by sorry

end distance_PF_equilateral_l81_81063


namespace probability_factor_of_120_l81_81649

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81649


namespace probability_is_13_over_30_l81_81679

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81679


namespace liza_final_balance_l81_81090

def initial_balance : ℕ := 800
def rent : ℕ := 450
def paycheck : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

theorem liza_final_balance :
  initial_balance - rent + paycheck - (electricity_bill + internet_bill) - phone_bill = 1563 := by
  sorry

end liza_final_balance_l81_81090


namespace binom_28_5_l81_81795

theorem binom_28_5 : 
    (nat.choose 26 3 = 2600) → 
    (nat.choose 26 4 = 14950) →
    (nat.choose 26 5 = 65780) →
    nat.choose 28 5 = 98280 :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end binom_28_5_l81_81795


namespace find_B_find_sin_C_l81_81045

-- Definitions
variables {A B C a b c : ℝ}
variable (h : a * sin (2 * B) = sqrt 3 * b * sin A)
variable (h_cos_A : cos A = 1 / 3)

-- Theorem Statements
theorem find_B (h : a * sin (2 * B) = sqrt 3 * b * sin A) : B = π / 6 :=
sorry

theorem find_sin_C (h_cos_A : cos A = 1 / 3) (B : ℝ) (hB : B = π / 6) : 
  sin C = (2 * sqrt 6 + 1) / 6 :=
sorry

end find_B_find_sin_C_l81_81045


namespace triangle_YG_calculation_l81_81027

theorem triangle_YG_calculation :
  ∃ (d e f : ℕ), coprime d f ∧ YG = (d - (√e)) / f ∧ d + e + f = 69 := by
  -- Let XYZ be a triangle with angles XYZ = 60° and XZY = 30°, and side XZ = 2
  -- Let N be the midpoint of segment XZ
  -- Let point B lie on side ZY such that XB ⊥ YN
  -- Let segment ZY be extended through Y to point G such that BG = BG
  sorry

end triangle_YG_calculation_l81_81027


namespace evaluate_expression_l81_81272

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end evaluate_expression_l81_81272


namespace max_points_without_isosceles_trapezium_l81_81241

-- Definition of points and their bounds
def is_within_bounds (p: ℕ × ℕ) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 101 ∧ 1 ≤ p.2 ∧ p.2 ≤ 101
  
-- Predicate to check if four points form an isosceles trapezium with the base parallel to the x or y axis
def forms_isosceles_trapezium (p1 p2 p3 p4: ℕ × ℕ) : Prop :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  let (x3, y3) := p3 in
  let (x4, y4) := p4 in
  (y1 = y2 ∧ y3 = y4 ∧ (x1 - x2).natAbs = (x3 - x4).natAbs) ∨
  (x1 = x2 ∧ x3 = x4 ∧ (y1 - y2).natAbs = (y3 - y4).natAbs)

-- The main theorem statement
theorem max_points_without_isosceles_trapezium : ∃ S : Finset (ℕ × ℕ),
  (∀ p ∈ S, is_within_bounds p) ∧
  (∀ p1 p2 p3 p4 ∈ S, ¬forms_isosceles_trapezium p1 p2 p3 p4) ∧
  S.card = 201 :=
sorry

end max_points_without_isosceles_trapezium_l81_81241


namespace sum_of_A_coordinates_l81_81879

variables (A B C : ℝ × ℝ)
variable (h : dist C A / dist B A = 1 / 3 ∧ dist C B / dist B A = 1 / 3)

def B_coords : ℝ × ℝ := (2, -3)
def C_coords : ℝ × ℝ := (-2, 6)

theorem sum_of_A_coordinates (h : dist C A / dist B A = 1 / 3) (B = B_coords) (C = C_coords) :
  (A.1 + A.2) = -22 :=
sorry

end sum_of_A_coordinates_l81_81879


namespace find_k_l81_81280

open Real

def vector_sub (k : ℝ) : ℝ × ℝ := (3 * k - 6, k + 2)

def vector_norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_k (k : ℝ) (hk : k ≠ 0) :
  vector_norm (vector_sub k) = 5 * sqrt 2 ↔
  k = (3.2 + sqrt 14.24) / 2 ∨ k = (3.2 - sqrt 14.24) / 2 :=
by sorry

end find_k_l81_81280


namespace sum_of_cube_roots_l81_81945

noncomputable def cube_root (x : ℝ) : ℝ := Real.cbrt x

theorem sum_of_cube_roots (a : ℝ) (h : a ≥ -3/4) :
  (cube_root ((a+1)/2 + (a+3)/6 * Real.sqrt ((4*a+3)/3)) + 
   cube_root ((a+1)/2 - (a+3)/6 * Real.sqrt ((4*a+3)/3)) = 1) :=
sorry

end sum_of_cube_roots_l81_81945


namespace total_games_played_l81_81392

-- Define the number of teams and games per matchup condition
def num_teams : ℕ := 10
def games_per_matchup : ℕ := 5

-- Calculate total games played during the season
theorem total_games_played : 
  5 * ((num_teams * (num_teams - 1)) / 2) = 225 := by 
  sorry

end total_games_played_l81_81392


namespace line_equation_l81_81779

/-- Given a line l that passes through the point M(1,2). If the length of the segment intercepted 
by two parallel lines 4x + 3y + 1 = 0 and 4x + 3y + 6 = 0 is √2, then the equation of the line l 
is either x + 7y = 15 or 7x - y = 5. -/
theorem line_equation (M : Point) (l : Line) (k : ℝ) :
  M = (1, 2) →
  (∀ (P : Point), (4 * P.1 + 3 * P.2 + 1 = 0) → P ∈ l) →
  (∀ (Q : Point), (4 * Q.1 + 3 * Q.2 + 6 = 0) → Q ∈ l) →
  distance (projection l ⟨4, 3, 1⟩) (projection l ⟨4, 3, 6⟩) = sqrt 2 →
  (∀ x y : ℝ, (x + 7 * y = 15) ∨ (7 * x - y = 5)). 

end line_equation_l81_81779


namespace complement_A_in_R_l81_81794

open Set

variable (R : Set ℝ) (A : Set ℝ)
noncomputable def C_R (s : Set ℝ) : Set ℝ := {x | x ∈ R ∧ x ∉ s}

theorem complement_A_in_R :
  A = {x | (1 - x) * (x + 2) ≤ 0} →
  C_R R A = {x | -2 < x ∧ x < 1} :=
by
  intros hA
  unfold C_R
  rw hA
  sorry

end complement_A_in_R_l81_81794


namespace sum_log_identity_l81_81255

open Real

noncomputable def log2 (x : ℝ) := log x / log 2

theorem sum_log_identity :
  (∑ k in Finset.range 98 \\.map (\x -> x + 3), log2(1 + 1 / k) * log2 k * log2 (k + 1)) =
  (1 / log2 3) - (1 / log2 101) :=
by
  sorry

end sum_log_identity_l81_81255


namespace rectangle_length_l81_81973

variable (w l : ℝ)

def perimeter (w l : ℝ) : ℝ := 2 * w + 2 * l

theorem rectangle_length (h1 : l = w + 2) (h2 : perimeter w l = 20) : l = 6 :=
by sorry

end rectangle_length_l81_81973


namespace jogging_track_circumference_l81_81134

/-- 
Given:
- Deepak's speed = 20 km/hr
- His wife's speed = 12 km/hr
- They meet for the first time in 32 minutes

Then:
The circumference of the jogging track is 17.0667 km.
-/
theorem jogging_track_circumference (deepak_speed : ℝ) (wife_speed : ℝ) (meet_time : ℝ)
  (h1 : deepak_speed = 20)
  (h2 : wife_speed = 12)
  (h3 : meet_time = (32 / 60) ) : 
  ∃ circumference : ℝ, circumference = 17.0667 :=
by
  sorry

end jogging_track_circumference_l81_81134


namespace area_of_square_I_l81_81132

theorem area_of_square_I (area_A area_B : ℕ) (hA : area_A = 1) (hB : area_B = 81) : 
    let side_A := Nat.sqrt area_A
    let side_B := Nat.sqrt area_B
    let side_G := side_B - side_A
    let side_C := side_B + side_A
    let side_F := side_G - side_A
    let side_H := side_G + side_F
    let side_E := 4
    let side_D := side_C + side_E
    let side_I := side_D + side_E
    let area_I := side_I * side_I
    area_I = 324 :=
by 
  intros
  rw [hA, hB]
  let side_A := Nat.sqrt 1
  let side_B := Nat.sqrt 81
  let side_G := 9 - 1
  let side_C := 9 + 1
  let side_F := 8 - 1
  let side_H := 8 + 7
  let side_D := 10 + 4
  let side_I := 14 + 4
  let area_I := 18 * 18
  show 324 = 324
  exact Eq.refl 324

end area_of_square_I_l81_81132


namespace max_distance_on_ellipse_l81_81912

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l81_81912


namespace chosen_number_probability_factorial_5_l81_81660

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81660


namespace sqrt_three_pow_divisible_l81_81289

/-- For any non-negative integer n, (1 + sqrt 3)^(2*n + 1) is divisible by 2^(n + 1) -/
theorem sqrt_three_pow_divisible (n : ℕ) :
  ∃ k : ℕ, (⌊(1 + Real.sqrt 3)^(2 * n + 1)⌋ : ℝ) = k * 2^(n + 1) :=
sorry

end sqrt_three_pow_divisible_l81_81289


namespace circumcircle_radius_l81_81036

theorem circumcircle_radius (A B C : ℝ) (r : ℝ) 
  (h_isosceles : A = B)
  (h_base : AB = 6)
  (h_angle : ∠ BCA = 45) : 
  r = 3 * Real.sqrt 2 := by 
sorry

end circumcircle_radius_l81_81036


namespace triangle_tangent_ratios_and_value_l81_81848

variables {A B C : ℝ} -- Angles of the triangle
variables {AB BC CA : ℝ} -- Side lengths of the triangle
variables (a b c : ℝ) -- Vectors representing the triangle sides

-- Conditions given in the problem
def condition1 : Prop := (a • b / 3 = b • c / 2) ∧ (b • c / 2 = c • a / 1)

-- Desired Outcomes From Solution
def target_ratios : Prop := ∃ (k : ℝ), ∀ (A B C : ℝ), 
  tan A = k * √11 ∧ tan B = k * (1/3 * √11) ∧ tan C = k * (1/2 * √11) 

theorem triangle_tangent_ratios_and_value 
  (h : condition1) : 
  target_ratios := sorry

end triangle_tangent_ratios_and_value_l81_81848


namespace divide_into_groups_l81_81232

theorem divide_into_groups (k : ℕ) (people : ℕ) (favs : people → (writer : ℕ) × (artist : ℕ) × (composer : ℕ))
  (h_each_fav_k : ∀ w a c, (∑ p in finset.univ, if favs p = (w, a, c) then 1 else 0) = k) :
  ∃ (groups : fin (3 * k - 2) → finset (fin people)),
    (∀ i j, i ≠ j → groups i ∩ groups j = ∅) ∧ -- Groups are disjoint
    (∀ g, ∀ p1 p2 ∈ groups g, p1 ≠ p2 →
      (favs p1).1 ≠ (favs p2).1 ∧ (favs p1).2 ≠ (favs p2).2 ∧ (favs p1).3 ≠ (favs p2).3) := -- Different tastes in the same group
sorry

end divide_into_groups_l81_81232


namespace probability_is_13_over_30_l81_81684

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81684


namespace neil_more_candy_percentage_l81_81822

def maggieCandyCount : ℕ := 50
def harperCandyCount : ℕ := maggieCandyCount + (maggieCandyCount * 30 / 100)
def neilCandyCount : ℕ := 91

theorem neil_more_candy_percentage :
  ((neilCandyCount - harperCandyCount) / harperCandyCount.toFloat) * 100 = 40 := 
sorry

end neil_more_candy_percentage_l81_81822


namespace candidate_percentage_l81_81576

noncomputable def percentage_of_votes_candidate (votes_cast: ℕ) (rival_won_by: ℕ): ℕ :=
  let P := 30 in
  -- Converting P percent to integer form for verification
  let candidate_votes := votes_cast * P / 100 in
  let rival_votes := candidate_votes + rival_won_by in
  if candidate_votes + rival_votes = votes_cast then P else 0

theorem candidate_percentage : percentage_of_votes_candidate 4400 1760 = 30 :=
by
  -- The steps of the proof would be included here
  sorry

end candidate_percentage_l81_81576


namespace school_choir_robe_cost_l81_81703

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end school_choir_robe_cost_l81_81703


namespace tangent_circles_condition_l81_81043

theorem tangent_circles_condition (Gamma Delta : Circle) (R : ℝ) (n : ℕ) (p : ℕ -> ℝ) :
  tangent Gamma Delta ∧
  contains Gamma Delta ∧
  (Δ.radius = R / 2) ∧
  (Γ.radius = R) ∧
  (n ≥ 3) →
  ((p 1 - p n) ^ 2 = (n - 1) ^ 2 * (2 * (p 1 + p n) - (n - 1) ^ 2 - 8)) ↔
  ∃ (Upsilon : ℕ -> Circle),
    ∀ i, (1 ≤ i ∧ i ≤ n) →
    tangent Upsilon[i] Gamma ∧
    tangent Upsilon[i] Delta ∧
    vecino Upsilon[i] Upsilon[i + 1] ∧
    (Upsilon 1).radius = (R / (p 1)) ∧
    (Upsilon n).radius = (R / (p n)) :=
sorry

end tangent_circles_condition_l81_81043


namespace fraction_of_traditionalists_equal_one_fourth_l81_81216

variables (T P : ℕ) (h1 : ∀ (i : ℕ), i ∈ (finset.range 4) → T = P / 12)

theorem fraction_of_traditionalists_equal_one_fourth 
  (h2 : ∀ x, x = 4 * T) 
  (h3 : T = P / 12) :
  (4 * T) / (P + 4 * T) = 1 / 4 :=
by
  rw [h3, ← nat.cast_add, nat.cast_mul]
  norm_num
  sorry

end fraction_of_traditionalists_equal_one_fourth_l81_81216


namespace max_distance_on_ellipse_l81_81914

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l81_81914


namespace find_radius_of_circle_l81_81420

noncomputable def radius_of_circle (P O : Point) (r : ℝ) : Prop :=
  let shortest_distance := 2
  let longest_distance := 6
  (shortest_distance + longest_distance) / 2 = r

theorem find_radius_of_circle
  (P O : Point)
  (shortest_distance longest_distance : ℝ)
  (h1 : shortest_distance = 2)
  (h2 : longest_distance = 6) :
  radius_of_circle P O 2 :=
by
  unfold radius_of_circle
  rw [h1, h2]
  norm_num
  exact eq.refl _

end find_radius_of_circle_l81_81420


namespace return_trip_avg_speed_l81_81233

noncomputable def avg_speed_return_trip : ℝ := 
  let distance_ab_to_sy := 120
  let rate_ab_to_sy := 50
  let total_time := 5.5
  let time_ab_to_sy := distance_ab_to_sy / rate_ab_to_sy
  let time_return_trip := total_time - time_ab_to_sy
  distance_ab_to_sy / time_return_trip

theorem return_trip_avg_speed 
  (distance_ab_to_sy : ℝ := 120)
  (rate_ab_to_sy : ℝ := 50)
  (total_time : ℝ := 5.5) 
  : avg_speed_return_trip = 38.71 :=
by
  sorry

end return_trip_avg_speed_l81_81233


namespace solve_inequality_l81_81508

def solution_set_of_inequality : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

theorem solve_inequality (x : ℝ) (h : (2 - x) / (x + 4) > 0) : x ∈ solution_set_of_inequality :=
by
  sorry

end solve_inequality_l81_81508


namespace find_x_solution_l81_81819

theorem find_x_solution (x : ℝ) 
  (h : ∑' n:ℕ, ((-1)^(n+1)) * (2 * n + 1) * x^n = 16) : 
  x = -15/16 :=
sorry

end find_x_solution_l81_81819


namespace probability_factor_of_120_l81_81645

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81645


namespace max_a_is_one_plus_sqrt_two_l81_81919

noncomputable def max_a (a b c : ℝ) : ℝ :=
if h : a + b + c = 3 ∧ ab + ac + bc = 3 then 1 + Real.sqrt 2 else 0

theorem max_a_is_one_plus_sqrt_two (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) : 
  a ≤ 1 + Real.sqrt 2 :=
sorry

end max_a_is_one_plus_sqrt_two_l81_81919


namespace card_at_42_is_8_spade_l81_81269

-- Conditions Definition
def cards_sequence : List String := 
  ["A♥", "A♠", "2♥", "2♠", "3♥", "3♠", "4♥", "4♠", "5♥", "5♠", "6♥", "6♠", "7♥", "7♠", "8♥", "8♠",
   "9♥", "9♠", "10♥", "10♠", "J♥", "J♠", "Q♥", "Q♠", "K♥", "K♠"]

-- Proposition to be proved
theorem card_at_42_is_8_spade :
  cards_sequence[(41 % 26)] = "8♠" :=
by sorry

end card_at_42_is_8_spade_l81_81269


namespace factor_probability_l81_81596

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81596


namespace coordinate_of_point_A_l81_81966

theorem coordinate_of_point_A (a b : ℝ) 
    (h1 : |b| = 3) 
    (h2 : |a| = 4) 
    (h3 : a > b) : 
    (a, b) = (4, 3) ∨ (a, b) = (4, -3) :=
by
    sorry

end coordinate_of_point_A_l81_81966


namespace kho_kho_only_players_l81_81561

theorem kho_kho_only_players :
  ∃ (H : ℕ), ∃ (K : ℕ), K + 5 = 10 ∧ K + H + 5 = 40 ∧ H = 30 :=
by
  use 30
  use 5
  split
  · exact (by norm_num : 5 + 5 = 10)
  split
  · exact (by norm_num : 5 + 30 + 5 = 40)
  · exact (by norm_num : 30 = 30)

end kho_kho_only_players_l81_81561


namespace back_wheel_revolutions_l81_81939

noncomputable def radiusFront : ℝ := 3
noncomputable def radiusBack : ℝ := 0.5
noncomputable def numRevolutionsFront : ℕ := 50

theorem back_wheel_revolutions 
  (rFront : ℝ) (rBack : ℝ) (revolutionsFront : ℕ)
  (h_rFront : rFront = radiusFront) 
  (h_rBack : rBack = radiusBack) 
  (h_revolutionsFront : revolutionsFront = numRevolutionsFront) :
  let distanceFront := 2 * Real.pi * rFront * revolutionsFront in
  let circumferenceBack := 2 * Real.pi * rBack in
  distanceFront / circumferenceBack = 300 := sorry


end back_wheel_revolutions_l81_81939


namespace cost_of_10_pound_bag_is_correct_l81_81220

noncomputable def cost_of_5_pound_bag : ℝ := 13.80
noncomputable def cost_of_25_pound_bag : ℝ := 32.25
noncomputable def min_pounds_needed : ℝ := 65
noncomputable def max_pounds_allowed : ℝ := 80
noncomputable def least_possible_cost : ℝ := 98.73

def min_cost_10_pound_bag : ℝ := 1.98

theorem cost_of_10_pound_bag_is_correct :
  ∀ (x : ℝ), (x >= min_pounds_needed / cost_of_25_pound_bag ∧ x <= max_pounds_allowed / cost_of_5_pound_bag ∧ least_possible_cost = (3 * cost_of_25_pound_bag + x)) → x = min_cost_10_pound_bag :=
by
  sorry

end cost_of_10_pound_bag_is_correct_l81_81220


namespace jack_jill_meeting_distance_thm_l81_81411

def jack_jill_meeting_distance (head_start : ℝ) (jack_uphill_speed : ℝ) (jack_downhill_speed : ℝ) (jill_uphill_speed : ℝ) (jill_downhill_speed : ℝ) : ℝ : = 
  let uphill_distance: ℝ := 7 
  let start_point := 0
  let jack_start := 8 / 60 -- Convert 8 minutes to hours
  let jack_position (t : ℝ) := if t < jack_start then start_point
                           else if t < (jack_start + (uphill_distance / jack_uphill_speed)) 
                             then uphill_distance - (jack_uphill_speed * (t - jack_start))
                           else uphill_distance - (jack_downhill_speed * (t - (jack_start + (uphill_distance / jack_uphill_speed))))
  let jill_position (t : ℝ) := if t < (start_point) 
                           then start_point
                           else if t < ((uphill_distance / jill_uphill_speed)) 
                             then jill_uphill_speed * t
                           else uphill_distance - (jill_downhill_speed * (t - (uphill_distance / jill_uphill_speed)))
  ∃ t : ℝ, jack_position t = jill_position t ∧ jack_position t = 7 - 2

theorem jack_jill_meeting_distance_thm : 
  jack_jill_meeting_distance (2 / 15) 12 18 14 20 = 2 :=
begin
  sorry
end

end jack_jill_meeting_distance_thm_l81_81411


namespace twenty_yuan_banknotes_count_l81_81989

theorem twenty_yuan_banknotes_count (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
                                    (total_banknotes : x + y + z = 24)
                                    (total_amount : 10 * x + 20 * y + 50 * z = 1000) :
                                    y = 4 := 
sorry

end twenty_yuan_banknotes_count_l81_81989


namespace solve_arcsin_arccos_equation_l81_81146

noncomputable def arcsin : ℝ → ℝ := sorry
noncomputable def arccos : ℝ → ℝ := sorry

theorem solve_arcsin_arccos_equation :
  ∀ (x : ℝ), 
    (x ∈ Icc (-1 : ℝ) 1) →
    (arcsin x + arcsin (2 * x) = arccos x + arccos (2 * x)) ↔ 
    x = (sqrt 5) / 5 := sorry

end solve_arcsin_arccos_equation_l81_81146


namespace find_a_b_exists_ran_m_log_inequality_l81_81811

-- Conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 - b * Real.log x
def f_tangent_condition (a b : ℝ) : Prop := (f 1 a b = 1) ∧ ((2*a - b) = 0)

-- Proof (I)
theorem find_a_b : ∃ (a b : ℝ), f_tangent_condition a b := sorry

-- Proof (II)
def g (x : ℝ) (m : ℝ) : ℝ := m * (x - 1) - 2 * Real.log x
theorem exists_ran_m : ∃ (m : ℝ), (∀ x ∈ Ioo 0 1, g x m ≥ 0) ∧ (g 1 m = 0) ∧ (m ≤ 2) := sorry

-- Proof (III)
theorem log_inequality (x1 x2 : ℝ) (hx : 0 < x1 ∧ x1 < x2) : (x2 - x1) / (Real.log x2 - Real.log x1) < 2 * x2 := sorry

end find_a_b_exists_ran_m_log_inequality_l81_81811


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81636

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81636


namespace probability_factor_of_120_l81_81648

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81648


namespace perimeter_of_triangle_l81_81958

-- The given condition about the average length of the triangle sides.
def average_side_length (a b c : ℝ) (h : (a + b + c) / 3 = 12) : Prop :=
  a + b + c = 36

-- The theorem to prove the perimeter of triangle ABC.
theorem perimeter_of_triangle (a b c : ℝ) (h : (a + b + c) / 3 = 12) : a + b + c = 36 :=
  by
    sorry

end perimeter_of_triangle_l81_81958


namespace John_scored_24point5_goals_l81_81754

theorem John_scored_24point5_goals (T G : ℝ) (n : ℕ) (A : ℝ)
  (h1 : T = 65)
  (h2 : n = 9)
  (h3 : A = 4.5) :
  G = T - (n * A) :=
by
  sorry

end John_scored_24point5_goals_l81_81754


namespace max_min_g_l81_81413

noncomputable def f (x : ℝ) : ℝ :=
(x^2 + 2 * x - 1) / (x^2 + 1)

noncomputable def g (x : ℝ) : ℝ :=
f(x) * f(1 - x)

theorem max_min_g :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g x ≤ 1) ∧ (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ g x) :=
by {
  sorry
}

end max_min_g_l81_81413


namespace total_lives_l81_81556

theorem total_lives (initial_friends : ℕ) (lives_per_player : ℕ) (additional_players : ℕ) :
  initial_friends = 8 → lives_per_player = 6 → additional_players = 2 →
  (initial_friends * lives_per_player + additional_players * lives_per_player) = 60 := 
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_lives_l81_81556


namespace range_of_q_l81_81786

variable (a_n : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
variable (hg_seq : ∀ n : ℕ, n > 0 → ∃ a_1 : ℝ, S_n n = a_1 * (1 - q ^ n) / (1 - q))
variable (pos_sum : ∀ n : ℕ, n > 0 → S_n n > 0)

theorem range_of_q : q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ) := sorry

end range_of_q_l81_81786


namespace solution_set_of_inequality_l81_81798

-- Definition of an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) = f (-x)

-- Definition of a function decreasing on (-∞, 0)
def decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → f(x) ≥ f(y)

-- Statement of the theorem to be proved
theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : even_function f) (h2 : decreasing_on_negative f) :
  {x : ℝ | f x ≤ f 3} =  {x : ℝ | -3 ≤ x ∧ x ≤ 3} :=
by 
  sorry

end solution_set_of_inequality_l81_81798


namespace probability_factor_of_5_factorial_l81_81694

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81694


namespace coordinates_of_point_A_l81_81343

def f (x : ℝ) : ℝ := x^2 + 3 * x

theorem coordinates_of_point_A (a : ℝ) (b : ℝ) 
    (slope_condition : deriv f a = 7) 
    (point_condition : f a = b) : 
    a = 2 ∧ b = 10 := 
by {
    sorry
}

end coordinates_of_point_A_l81_81343


namespace domain_of_k_l81_81263

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 7)) + (1 / (x^2 + 1)) + (1 / (x^4 + 16))

theorem domain_of_k : {x : ℝ | k x ∈ set.univ} = {x : ℝ | x ≠ -7} :=
by
  ext
  simp [k]
  split
  · intro h
    simp at h
    sorry
  · intro h
    simp at h
    sorry

end domain_of_k_l81_81263


namespace expense_of_three_yuan_l81_81835

def isIncome (x : Int) : Prop := x > 0
def isExpense (x : Int) : Prop := x < 0
def incomeOfTwoYuan : Int := 2

theorem expense_of_three_yuan : isExpense (-3) :=
by
  -- Assuming the conditions:
  -- Income is positive: isIncome incomeOfTwoYuan (which is 2)
  -- Expenses are negative
  -- Expenses of 3 yuan should be denoted as -3 yuan
  sorry

end expense_of_three_yuan_l81_81835


namespace coefficients_divisible_by_seven_l81_81052

theorem coefficients_divisible_by_seven {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  a % 7 = 0 ∧ b % 7 = 0 ∧ c % 7 = 0 ∧ d % 7 = 0 ∧ e % 7 = 0 := 
  sorry

end coefficients_divisible_by_seven_l81_81052


namespace arrow_reading_l81_81965

-- Define the interval and values within it
def in_range (x : ℝ) : Prop := 9.75 ≤ x ∧ x ≤ 10.00
def closer_to_990 (x : ℝ) : Prop := |x - 9.90| < |x - 9.875|

-- The main theorem statement expressing the problem
theorem arrow_reading (x : ℝ) (hx1 : in_range x) (hx2 : closer_to_990 x) : x = 9.90 :=
by sorry

end arrow_reading_l81_81965


namespace smallest_area_square_l81_81570

theorem smallest_area_square (a b u v : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : u = 4) (h₄ : v = 6) :
  ∃ s : ℕ, s^2 = 81 ∧ 
    (∀ xa ya xb yb xu yu xv yv : ℕ, 
      (xa + a ≤ s) ∧ (ya + b ≤ s) ∧ (xb + u ≤ s) ∧ (yb + v ≤ s) ∧ 
      ─xa < xb → xb < xa + a → ─ya < yb → yb < ya + b →
      ─xu < xv → xv < xu + u → ─yu < yv → yv < yu + v ∧
      (ya + b ≤ yv ∨ yu + v ≤ yb))
    := sorry

end smallest_area_square_l81_81570


namespace shapes_with_equal_perimeter_l81_81079

theorem shapes_with_equal_perimeter {s : ℝ} (shapes : List (List ℝ)) 
  (h_shapes_conditions : ∀ shape ∈ shapes, (∀ len ∈ shape, len = s ∨ len = 2 * s ∨ len = 3 * s ∨ len = 4 * s)) :
  ∃ equal_shapes, equal_shapes.length = 4 ∧ 
  (∀ shape ∈ equal_shapes, (shape.sum = 4 * s)) ∧ 
  (∀ shape ∉ equal_shapes, shape.sum ≠ 4 * s) :=
by
  sorry

end shapes_with_equal_perimeter_l81_81079


namespace number_of_subsets_number_of_nonempty_proper_subsets_number_of_nonempty_subsets_sum_of_elements_of_nonempty_subsets_l81_81358

def A : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem number_of_subsets : A.powerset.card = 64 := by
  sorry

theorem number_of_nonempty_proper_subsets : (A.powerset.filter (λ s, s ≠ ∅ ∧ s ≠ A)).card = 62 := by
  sorry

theorem number_of_nonempty_subsets : (A.powerset.filter (λ s, s ≠ ∅)).card = 63 := by
  sorry

theorem sum_of_elements_of_nonempty_subsets : ((A.powerset.filter (λ s, s ≠ ∅)).sum (λ s, s.sum id)) = 672 := by
  sorry

end number_of_subsets_number_of_nonempty_proper_subsets_number_of_nonempty_subsets_sum_of_elements_of_nonempty_subsets_l81_81358


namespace domain_of_log_one_minus_tan_l81_81490

theorem domain_of_log_one_minus_tan :
  (∀ x : ℝ, 1 - tan x > 0 → -((π / 2) + (k : ℤ) * π) < x ∧ x < ((π / 4) + (k : ℤ) * π)) :=
sorry

end domain_of_log_one_minus_tan_l81_81490


namespace find_a_l81_81774

theorem find_a (a : ℝ) : (∀ x : ℝ, (x + 1) * (x - 3) = x^2 + a * x - 3) → a = -2 :=
  by
    sorry

end find_a_l81_81774


namespace Liu_Wei_parts_per_day_l81_81714

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end Liu_Wei_parts_per_day_l81_81714


namespace contrapositive_of_x_squared_lt_one_is_true_l81_81126

variable {x : ℝ}

theorem contrapositive_of_x_squared_lt_one_is_true
  (h : ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) :
  ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1 :=
by
  sorry

end contrapositive_of_x_squared_lt_one_is_true_l81_81126


namespace max_intersection_points_of_fifth_degree_polynomials_l81_81175

noncomputable def max_intersection_5th_degree_polynomials (p q : polynomial ℝ) (h₁ : p.degree = 5) (h₂ : q.degree = 5) (h₃ : p.leading_coeff = 1) (h₄ : q.leading_coeff = 1) (h_diff : p ≠ q) : ℕ :=
begin
  sorry
end

theorem max_intersection_points_of_fifth_degree_polynomials (p q : polynomial ℝ) (h₁ : p.degree = 5) (h₂ : q.degree = 5) (h₃ : p.leading_coeff = 1) (h₄ : q.leading_coeff = 1) (h_diff : p ≠ q) :
  ∃ n, n ≤ 4 ∧ max_intersection_5th_degree_polynomials p q h₁ h₂ h₃ h₄ h_diff = n := 
by {
  sorry
}

end max_intersection_points_of_fifth_degree_polynomials_l81_81175


namespace chosen_number_probability_factorial_5_l81_81662

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81662


namespace johns_average_speed_last_hour_l81_81057

theorem johns_average_speed_last_hour
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (distance_last_hour : ℕ)
  (average_speed_last_hour : ℕ)
  (H1 : total_distance = 120)
  (H2 : total_time = 3)
  (H3 : speed_first_hour = 40)
  (H4 : speed_second_hour = 50)
  (H5 : distance_last_hour = total_distance - (speed_first_hour + speed_second_hour))
  (H6 : average_speed_last_hour = distance_last_hour / 1)
  : average_speed_last_hour = 30 := 
by
  -- Placeholder for the proof
  sorry

end johns_average_speed_last_hour_l81_81057


namespace collinearity_of_P_Q_R_l81_81310

-- Definitions of the given conditions
variables {Point : Type} [MetricSpace Point]
variable (circle : Point → ℝ → Prop)
variable (diameter : Point → Point → ℝ) -- Diameter between points B and D
variable {A B C D P R Q : Point}
variable (line : Point → Point → Prop)
variable (intersection : Point → Point → Point)

-- Assumptions based on the problem conditions
axiom circle_diameter : circle B (diameter B D)
axiom point_A_on_circle : circle A (diameter B D / 2)
axiom point_C_on_circle : circle C (diameter B D / 2)
axiom AB_CD_intersect_P : intersection (line A B) (line C D) = P
axiom AD_BC_intersect_R : intersection (line A D) (line B C) = R
axiom tangents_at_A_C_intersect_Q : 
  ∃ Q, Q = intersection (tangent A (circle A (diameter B D / 2))) (tangent C (circle C (diameter B D / 2)))

-- Theorem statement based on the translation requirement
theorem collinearity_of_P_Q_R : collinear P Q R :=
sorry  -- Proof to be provided

end collinearity_of_P_Q_R_l81_81310


namespace angle_C_45_cos_B_sqrt6_div3_l81_81046

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) / 4

-- Proof Problem 1: Prove that angle C = 45°
theorem angle_C_45 (a b c : ℝ) (h1 : triangle_area a b c = (a^2 + b^2 - c^2) / 4) :
  ∃ C : ℝ, C = 45 :=
  sorry

-- Proof Problem 2: Prove that cos B = sqrt(6)/3
theorem cos_B_sqrt6_div3 (a : ℝ) (b : ℝ := 2) (c : ℝ := Real.sqrt 6) (C : ℝ := 45) :
  ∃ B : ℝ, cos B = Real.sqrt 6 / 3 :=
  sorry

end angle_C_45_cos_B_sqrt6_div3_l81_81046


namespace probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l81_81708

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 1
def total_outcomes_plan1 := 15
def winning_outcomes_plan1 := 6
def probability_plan1 : ℚ := winning_outcomes_plan1 / total_outcomes_plan1

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 2
def total_outcomes_plan2 := 36
def winning_outcomes_plan2 := 11
def probability_plan2 : ℚ := winning_outcomes_plan2 / total_outcomes_plan2

-- Statements to prove
theorem probability_of_winning_plan1_is_2_over_5 : probability_plan1 = 2 / 5 :=
by sorry

theorem probability_of_winning_plan2_is_11_over_36 : probability_plan2 = 11 / 36 :=
by sorry

theorem choose_plan1 : probability_plan1 > probability_plan2 :=
by sorry

end probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l81_81708


namespace well_performing_student_take_home_pay_l81_81194

theorem well_performing_student_take_home_pay : 
  ∃ (base_salary bonus : ℕ) (income_tax_rate : ℝ),
      (base_salary = 25000) ∧ (bonus = 5000) ∧ (income_tax_rate = 0.13) ∧
      let total_earnings := base_salary + bonus in
      let income_tax := total_earnings * income_tax_rate in
      total_earnings - income_tax = 26100 :=
by
  use 25000
  use 5000
  use 0.13
  intros
  sorry

end well_performing_student_take_home_pay_l81_81194


namespace division_of_mixed_numbers_l81_81521

noncomputable def mixed_to_improper (n : ℕ) (a b : ℕ) : ℚ :=
  n + (a / b)

theorem division_of_mixed_numbers : 
  (mixed_to_improper 7 1 3) / (mixed_to_improper 2 1 2) = 44 / 15 :=
by
  sorry

end division_of_mixed_numbers_l81_81521


namespace num_divisors_m_sq_less_than_m_not_divide_m_l81_81924

def m : ℕ := 2 ^ 35 * 5 ^ 21

theorem num_divisors_m_sq_less_than_m_not_divide_m : 
  let m_sq := m ^ 2 in
  let total_divisors_m_sq := (70 + 1) * (42 + 1) in
  let divisors_less_than_m := (total_divisors_m_sq - 1) / 2 in
  let total_divisors_m := (35 + 1) * (21 + 1) in
  let divisors_m_less_than_m := total_divisors_m - 1 in
  divisors_less_than_m - divisors_m_less_than_m = 735 :=
by
  let m := 2 ^ 35 * 5 ^ 21
  let m_sq := m ^ 2
  let total_divisors_m_sq := (70 + 1) * (42 + 1)
  let divisors_less_than_m := (total_divisors_m_sq - 1) / 2
  let total_divisors_m := (35 + 1) * (21 + 1)
  let divisors_m_less_than_m := total_divisors_m - 1
  show divisors_less_than_m - divisors_m_less_than_m = 735
  sorry

end num_divisors_m_sq_less_than_m_not_divide_m_l81_81924


namespace state_6_9_not_occur_l81_81957

theorem state_6_9_not_occur : 
  ¬ ∃ (a b : ℕ), 
    (∃ k, a = 168 - k * 93 ∨ a = 93 - k * 75 ∨ a = 75 - k * 18 ∨ a = 57 - k * 39 ∨ a = 39 - k * 21 ∨ a = 21 - k * 3 ∨ a = 18 - k * 15 ∨ a = 15 - k * 12 ∨ a = 12 - k * 9 ∨ a = 9 - k * 6 ∨ a = 6 - k * 3) ∧ 
    (∃ k, b = 93 ∨ b = 75 ∨ b = 18 ∨ b = 57 ∨ b = 39 ∨ b = 21 ∨ b = 3 ∨ b = 15 ∨ b = 12 ∨ b = 9 ∨ b = 6 ∨ b = 3) ∧ 
    (a, b) = (6, 9) := 
by {
  sorry
}

end state_6_9_not_occur_l81_81957


namespace probability_number_is_factor_of_120_l81_81627

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81627


namespace jogging_meetings_l81_81192

/-- 甲 and 乙 are jogging back and forth on a straight road that is 400 meters long,
with speeds of 3 meters per second and 2.5 meters per second, respectively.
They start simultaneously from opposite ends and head towards each other.
We need to prove that the number of times they meet within 20 minutes is 8. -/
theorem jogging_meetings (time_minutes : ℕ) (road_length speed_甲 speed_乙 : ℝ) (time_seconds meetings : ℝ)
  (h_time_minutes : time_minutes = 20)
  (h_road_length : road_length = 400)
  (h_speed_甲 : speed_甲 = 3)
  (h_speed_乙 : speed_乙 = 2.5)
  (h_time_seconds : time_seconds = time_minutes * 60) :
  meetings = ⌊(speed_甲 + speed_乙) * time_seconds / road_length ⌋ :=
by
  sorry

end jogging_meetings_l81_81192


namespace total_employees_l81_81855

-- definition of employees with sets T (truth-tellers) and L (liars)
def is_truth_teller (e : ℕ) : Prop := true -- Dummy definitions for type correctness
def is_liar (e : ℕ) : Prop := true -- Dummy definitions for type correctness

-- condition 1: If e is a truth-teller, then there are fewer than 10 people who work more than e
def cond1_truth_teller (e : ℕ) (employees : Finset ℕ) : Prop := 
  is_truth_teller e → ∃ (workload : ℕ → ℕ), employees.filter (λ x => workload x > workload e).card < 10

-- condition 2: If e is a truth-teller, then at least 100 people have a salary greater than e
def cond2_truth_teller (e : ℕ) (employees : Finset ℕ) : Prop := 
  is_truth_teller e → ∃ (salary : ℕ → ℕ), employees.filter (λ x => salary x > salary e).card ≥ 100

-- condition 3: If e is a liar, then there are at least 10 people who work more than e
def cond1_liar (e : ℕ) (employees : Finset ℕ) : Prop := 
  is_liar e → ∃ (workload : ℕ → ℕ), employees.filter (λ x => workload x > workload e).card ≥ 10

-- condition 4: If e is a liar, then there are fewer than 100 people who have a salary greater than e
def cond2_liar (e : ℕ) (employees : Finset ℕ) : Prop := 
  is_liar e → ∃ (salary : ℕ → ℕ), employees.filter (λ x => salary x > salary e).card < 100

-- Main theorem: Prove that the total number of employees is 110
theorem total_employees (employees : Finset ℕ) : 
  (∃ (t_truth : ℕ) (l_liar : ℕ), 
    (∑ e in employees, if is_truth_teller e then 1 else 0) = t_truth ∧ 
    (∑ e in employees, if is_liar e then 1 else 0) = l_liar ∧ 
    t_truth = 10 ∧ l_liar = 100 ∧ 
    ∀ e ∈ employees, 
      cond1_truth_teller e employees ∧ 
      cond2_truth_teller e employees ∧ 
      cond1_liar e employees ∧ 
      cond2_liar e employees) → 
  employees.card = 110 :=
sorry

end total_employees_l81_81855


namespace min_value_expression_l81_81470

theorem min_value_expression : ∃ (a b : ℕ), (1 ≤ a ∧ a < 10) ∧ (1 ≤ b ∧ b < 10) ∧ (2 * a - a * b + a ^ 2 = -6) :=
by
  sorry

end min_value_expression_l81_81470


namespace alex_border_material_l81_81488

noncomputable def border_material_needed (π_estimate : ℚ) (area_circle : ℚ) (extra_border : ℚ) : ℚ :=
  let r := (area_circle * 7 / 22).sqrt
  let circumference := 2 * π_estimate * r
  circumference + extra_border

theorem alex_border_material : 
  border_material_needed (22/7) 176 3 = 50.1 :=
by
  -- Calculation steps as per the provided problem and solution
  have r_squared : ℚ := 176 * 7 / 22
  have r := r_squared.sqrt
  have circumference := 2 * (22/7) * r
  have adjusted_circumference := circumference + 3
  exact adjusted_circumference


end alex_border_material_l81_81488


namespace max_statements_true_l81_81918

noncomputable def max_true_statements (a b : ℝ) : ℕ :=
  (if (a^2 > b^2) then 1 else 0) +
  (if (a < b) then 1 else 0) +
  (if (a < 0) then 1 else 0) +
  (if (b < 0) then 1 else 0) +
  (if (1 / a < 1 / b) then 1 else 0)

theorem max_statements_true : ∀ (a b : ℝ), max_true_statements a b ≤ 4 :=
by
  intro a b
  sorry

end max_statements_true_l81_81918


namespace measure_of_A_length_of_c_l81_81849

variable {α β γ : Prop}

-- Definitions for the problem conditions
def sides (a b c : ℝ) (A B C : ℝ) : Prop :=
a * Real.cos C + (1 / 2) * c = b

def given_a_b (a b : ℝ) : Prop :=
a = Real.sqrt 15 ∧ b = 4

-- Statement for the measure of angle A
theorem measure_of_A (a b c : ℝ) (A B C : ℝ) (h1 : sides a b c A B C) : A = Real.pi / 3 :=
sorry

-- Statement for the length of side c given specific a and b values
theorem length_of_c (a b c : ℝ) 
(h1 : sides a b c (Real.pi / 3) _ _) 
(h2 : given_a_b a b) : c = 2 + Real.sqrt 3 :=
sorry

end measure_of_A_length_of_c_l81_81849


namespace find_n_l81_81070

theorem find_n 
  (n : ℕ)
  (h1 : 0 < n)
  (h2 : n < real.sqrt 2)
  (h3 : real.sqrt 2 < n + 1) : 
  n = 1 :=
sorry

end find_n_l81_81070


namespace area_of_triangle_COB_l81_81098

theorem area_of_triangle_COB (p : ℝ) (h : 0 ≤ p ∧ p ≤ 15) : 
    ∃ (area : ℝ), area = (15 * p) / 2 :=
by 
  use (15 * p) / 2
  sorry

end area_of_triangle_COB_l81_81098


namespace harvest_duration_l81_81446

theorem harvest_duration (n : ℕ) : 
  ∑ k in range n, (16 + k * 8 - 12) = 1216 → n = 17 :=
by sorry

end harvest_duration_l81_81446


namespace range_of_a_l81_81808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l81_81808


namespace potato_bag_weight_l81_81574

theorem potato_bag_weight :
  ∃ w : ℝ, w = 16 / (w / 4) ∧ w = 16 := 
by
  sorry

end potato_bag_weight_l81_81574


namespace correct_diagnosis_l81_81551

-- Define the statements of the doctors
structure DoctorStatements :=
  (H1 : Prop)  -- The patient has a strong astigmatism.
  (H2 : Prop)  -- The patient smokes too much.
  (H3 : Prop)  -- The patient has a tropical fever.
  (T1 : Prop)  -- The patient has a strong astigmatism.
  (T2 : Prop)  -- The patient doesn’t eat well.
  (T3 : Prop)  -- The patient suffers from high blood pressure.
  (O1 : Prop)  -- The patient has a strong astigmatism.
  (O2 : Prop)  -- The patient is near-sighted.
  (O3 : Prop)  -- The patient has no signs of retinal detachment.

-- Define the given conditions
axiom H2_true : DoctorStatements → Prop
axiom T1_true : DoctorStatements → Prop
axiom O1_true : DoctorStatements → Prop

-- Define the correct diagnosis
def Diagnosis (s : DoctorStatements) : Prop :=
  s.H1 ∧ s.H2 ∧ s.T2 ∧ ¬s.H3

-- The final theorem to be proved
theorem correct_diagnosis (s : DoctorStatements) (h2 : H2_true s) (t1 : T1_true s) (o1 : O1_true s) : Diagnosis s :=
by
  split
  -- sorry, proof would go here
  sorry

end correct_diagnosis_l81_81551


namespace probability_number_is_factor_of_120_l81_81625

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81625


namespace probability_factor_of_5_factorial_l81_81687

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81687


namespace probability_factorial_five_l81_81606

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81606


namespace large_pizza_size_l81_81005

-- Definitions based on given conditions
def small_pizza_side_length : ℝ := 12
def small_pizza_cost : ℝ := 10
def large_pizza_cost : ℝ := 20
def individual_budget : ℝ := 30
def pooled_extra_area : ℝ := 36

-- Definition of side length of large pizza
def large_pizza_side_length : ℝ := 10 * Real.sqrt 3

-- Conditions
def area_of_small_pizza : ℝ := small_pizza_side_length^2
def individual_area_total : ℝ := 3 * area_of_small_pizza
def pooled_total_area : ℝ := individual_area_total + individual_area_total + pooled_extra_area

-- The theorem to be proven
theorem large_pizza_size (x : ℝ) (h : 3 * x^2 = pooled_total_area) : x = large_pizza_side_length :=
by
  -- Proof not required, so we use sorry
  sorry

end large_pizza_size_l81_81005


namespace limit_of_a_n_exists_and_is_zero_equality_for_a_n_l81_81412

section
variable {a_n : ℕ → ℝ}

/- Define a_n as given in the problem -/
def a_n (n : ℕ) : ℝ :=
  (∏ k in Finset.range n, (2 * (k + 1) - 1) : ℝ) /
  (∏ k in Finset.range n, 2 * (k + 1) : ℝ)

/- Prove that the limit of a_n exists and equals 0 -/
theorem limit_of_a_n_exists_and_is_zero :
  tendsto (λ n, a_n n) at_top (𝓝 0) := 
sorry

/- Prove the given equality for a_n holds -/
theorem equality_for_a_n (n : ℕ) :
  a_n n = (∏ k in Finset.range n, (1 - (1 / ((2 * (k + 1)) ^ 2)) : ℝ)) / (2 * n + 1) / a_n n :=
sorry

end

end limit_of_a_n_exists_and_is_zero_equality_for_a_n_l81_81412


namespace least_number_of_equal_cubes_l81_81581

def cuboid_dimensions := (18, 27, 36)
def ratio := (1, 2, 3)

theorem least_number_of_equal_cubes :
  ∃ n, n = 648 ∧
  ∃ a b c : ℕ,
    (a, b, c) = (3, 6, 9) ∧
    (18 % a = 0 ∧ 27 % b = 0 ∧ 36 % c = 0) ∧
    18 * 27 * 36 = n * (a * b * c) :=
sorry

end least_number_of_equal_cubes_l81_81581


namespace average_distinct_u_l81_81337

theorem average_distinct_u :
  ∀ (u : ℕ), (∃ a b : ℕ, a + b = 6 ∧ ab = u) →
  {u | ∃ a b : ℕ, a + b = 6 ∧ ab = u}.to_finset.val.sum / 3 = 22 / 3 :=
sorry

end average_distinct_u_l81_81337


namespace rational_roots_count_l81_81259

theorem rational_roots_count (b₄ b₃ b₂ b₁ : ℚ) :
  let divisors := {1, -1, 2, -2, 4, -4, 5, -5, 10, -10, 20, -20, 1/2, -1/2, 1/4, -1/4, 1/8, -1/8, 2/5, -2/5, 4/5, -4/5, 1/5, -1/5, 5/2, -5/2, 10/1, -10/1, 1/2, -1/2, 1/4, -1/4, 1/8, -1/8, 1.25, -1.25}
  in divisors.card = 28 :=
by
  sorry

end rational_roots_count_l81_81259


namespace count_valid_z_l81_81427

-- Definitions from the problem conditions
def f (z : ℂ) : ℂ := z^2 + 2 * complex.I * z + 2
def real_int_limit : ℤ := 20

-- The translation of the proof problem to a Lean 4 statement
theorem count_valid_z :
  ∀ z : ℂ, 
    (0 < z.im ∧ 
     abs (f(z).re) ≤ real_int_limit ∧ 
     abs (f(z).im) ≤ real_int_limit) → 
    nat :=

-- Replace _count_ with the specific count from the detailed solution
N := _count_
sorry -- proof not required

end count_valid_z_l81_81427


namespace sad_children_count_l81_81087

theorem sad_children_count (total_children happy_children neither_happy_nor_sad children sad_children : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_neither : neither_happy_nor_sad = 20)
  (boys girls happy_boys sad_girls neither_boys : ℕ)
  (h_boys : boys = 17)
  (h_girls : girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4)
  (h_neither_boys : neither_boys = 5) :
  sad_children = total_children - happy_children - neither_happy_nor_sad :=
by sorry

end sad_children_count_l81_81087


namespace point_coordinates_sum_l81_81330

noncomputable def f (x : ℝ) : ℝ := sorry

theorem point_coordinates_sum :
  f 3 = 8 →
  (∃ x y : ℝ, 2 * y = 4 * f (3 * x - 1) + 6 ∧ x + y = 21) :=
by
  intro h
  use 2
  use 19
  split
  { rw h, sorry },
  { refl }

end point_coordinates_sum_l81_81330


namespace prob_factorial_5_l81_81669

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81669


namespace percentage_excess_of_first_side_l81_81397

variable (L W : ℝ) -- Denoting the original length and width
variable (x : ℝ) -- Denoting the percentage in excess for the first side

def true_area : ℝ := L * W
def measured_area : ℝ := L * (1 + x / 100) * W * 0.95

theorem percentage_excess_of_first_side (h : measured_area L W x = true_area L W * 1.026) : x = 8 :=
by
  sorry

end percentage_excess_of_first_side_l81_81397


namespace infinite_primes_dividing_nsq_add_n_add_one_l81_81260

theorem infinite_primes_dividing_nsq_add_n_add_one :
  ∃ (seq : ℕ → ℕ), ∀ i : ℕ, prime (seq i) ∧ ∃ n : ℕ, seq i ∣ n^2 + n + 1 :=
sorry

end infinite_primes_dividing_nsq_add_n_add_one_l81_81260


namespace not_divisible_by_1980_divisible_by_1981_l81_81480

open Nat

theorem not_divisible_by_1980 (x : ℕ) : ¬ (2^100 * x - 1) % 1980 = 0 := by
sorry

theorem divisible_by_1981 : ∃ x : ℕ, (2^100 * x - 1) % 1981 = 0 := by
sorry

end not_divisible_by_1980_divisible_by_1981_l81_81480


namespace radius_order_l81_81738

structure Circle :=
(radius : ℝ)

def circle_A := Circle.mk (3 : ℝ)
def circle_B := Circle.mk (4 : ℝ)
def circle_C := Circle.mk (5 : ℝ)

theorem radius_order : circle_A.radius < circle_B.radius ∧ circle_B.radius < circle_C.radius :=
by 
  have ra : circle_A.radius = 3 := rfl
  have rb : circle_B.radius = 4 := rfl
  have rc : circle_C.radius = 5 := rfl
  exact ⟨by norm_num, by norm_num⟩

end radius_order_l81_81738


namespace f_pi_six_eq_neg_sqrt_three_div_two_l81_81968

def f : ℝ → ℝ := λ x, 1 - 2 * (Real.sin (x + Real.pi / 4))^2

theorem f_pi_six_eq_neg_sqrt_three_div_two : f (Real.pi / 6) = - Real.sqrt 3 / 2 :=
by
  sorry

end f_pi_six_eq_neg_sqrt_three_div_two_l81_81968


namespace hyperbola_eccentricity_l81_81355

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (c : ℝ) (h3 : c^2 = a^2 + b^2) :
  let e := c / a in
  e = 1 + Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l81_81355


namespace hexagonal_napkins_nailing_possible_pentagonal_napkins_nailing_impossible_l81_81188

-- Part (a): Proving it is always possible to nail down all hexagonal napkins with one nail each.
theorem hexagonal_napkins_nailing_possible :
  ∀ napkins : set (set.point_in_plane), (∀ n ∈ napkins, is_regular_hexagon n ∧ aligned_to_same_line napkins) →
  ∃ nails : set point_in_plane, (∀ n ∈ napkins, ∃ nail ∈ nails, nail_in_napkin nail n) :=
by admit

-- Part (b): Proving it is not always possible to nail down all pentagonal napkins with one nail each.
theorem pentagonal_napkins_nailing_impossible :
  ∀ napkins : set (set.point_in_plane), (∀ n ∈ napkins, is_regular_pentagon n ∧ aligned_to_same_line napkins) →
  ¬ (∃ nails : set point_in_plane, (∀ n ∈ napkins, ∃ nail ∈ nails, nail_in_napkin nail n)) :=
by admit

end hexagonal_napkins_nailing_possible_pentagonal_napkins_nailing_impossible_l81_81188


namespace max_distance_on_ellipse_l81_81892

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l81_81892


namespace statement_cost_per_rose_l81_81086

/-
Theorem statement:
Given:
- n_tables: the number of tables (20).
- cost_tablecloth: the cost to rent one tablecloth ($25).
- n_place_settings: the number of place settings per table (4).
- cost_place_setting: the cost to rent one place setting ($10).
- n_roses_per_table: the number of roses per table (10).
- n_lilies_per_table: the number of lilies per table (15).
- cost_lily: the cost of one lily ($4).
- total_cost: the total cost of decorations ($3500).

Prove that the cost per rose is $5.
-/

theorem cost_per_rose 
  (n_tables : ℕ)
  (cost_tablecloth : ℕ)
  (n_place_settings : ℕ)
  (cost_place_setting : ℕ)
  (n_roses_per_table : ℕ)
  (n_lilies_per_table : ℕ)
  (cost_lily : ℕ)
  (total_cost : ℕ) :
  let cost_roses := total_cost - (n_tables * cost_tablecloth + n_tables * n_place_settings * cost_place_setting + n_tables * n_lilies_per_table * cost_lily) in
  let total_roses := n_tables * n_roses_per_table in
  let cost_per_rose := cost_roses / total_roses in
  n_tables = 20 → 
  cost_tablecloth = 25 → 
  n_place_settings = 4 → 
  cost_place_setting = 10 → 
  n_roses_per_table = 10 → 
  n_lilies_per_table = 15 → 
  cost_lily = 4 → 
  total_cost = 3500 → 
  cost_per_rose = 5 :=
by {
  intros,
  sorry
}

end statement_cost_per_rose_l81_81086


namespace bisector_twice_altitude_l81_81095

theorem bisector_twice_altitude 
  (A B C : Type) 
  [h : Nonempty (Triangle A B C)] 
  (angle_A angle_B angle_C : ℝ)
  (h_angleC : angle_C = 120 + angle_A) 
  (altitude_BD bisector_BE : ℝ) 
  (h_angleB : angle_B = 60 - 2 * angle_A) 
  (right_angle : angle_BD = 30):
  bisector_BE = 2 * altitude_BD :=
sorry

end bisector_twice_altitude_l81_81095


namespace find_cos_alpha_l81_81322

-- Define that α is an acute angle
def is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Define the given conditions
def given_conditions (α : ℝ) : Prop :=
  is_acute α ∧ cos (α + π / 6) = 3 / 5

-- Prove that cos α = (3 * sqrt 3 + 4) / 10 given the conditions
theorem find_cos_alpha (α : ℝ) (h : given_conditions α) : 
  cos α = (3 * Real.sqrt 3 + 4) / 10 :=
sorry

end find_cos_alpha_l81_81322


namespace smallest_sum_B_c_l81_81013

theorem smallest_sum_B_c (B : ℕ) (c : ℕ) (hB : B < 5) (hc : c > 6) :
  31 * B = 4 * c + 4 → (B + c) = 34 :=
by
  sorry

end smallest_sum_B_c_l81_81013


namespace conner_ties_sydney_l81_81956

def sydney_initial_collect := 837
def conner_initial_collect := 723

def sydney_collect_day_one := 4
def conner_collect_day_one := 8 * sydney_collect_day_one / 2

def sydney_collect_day_two := (sydney_initial_collect + sydney_collect_day_one) - ((sydney_initial_collect + sydney_collect_day_one) / 10)
def conner_collect_day_two := conner_initial_collect + conner_collect_day_one + 123

def sydney_collect_day_three := sydney_collect_day_two + 2 * conner_collect_day_one
def conner_collect_day_three := (conner_collect_day_two - (123 / 4))

theorem conner_ties_sydney :
  sydney_collect_day_three <= conner_collect_day_three :=
by
  sorry

end conner_ties_sydney_l81_81956


namespace package_weights_l81_81449

theorem package_weights (a b c : ℕ) 
  (h1 : a + b = 108) 
  (h2 : b + c = 132) 
  (h3 : c + a = 138) 
  (h4 : a ≥ 40) 
  (h5 : b ≥ 40) 
  (h6 : c ≥ 40) : 
  a + b + c = 189 :=
sorry

end package_weights_l81_81449


namespace factor_probability_l81_81595

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81595


namespace digit_divisibility_l81_81552

-- Define the main problem statement
theorem digit_divisibility (a b c : ℕ) (n : ℕ) : 
  a ∈ {0,1,2,3,4,5,6,7,8,9} ∧ b ∈ {0,1,2,3,4,5,6,7,8,9} ∧ c ∈ {0,1,2,3,4,5,6,7,8,9} 
  ∧ n = 387000 + 10 * (10 * a + b) + c 
  ∧ (n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0) 
  ↔ n ∈ {387030, 387240, 387450, 387660, 387870} :=
by
  sorry

end digit_divisibility_l81_81552


namespace doubled_arithmetic_mean_of_first_four_primes_l81_81248

-- Declare noncomputable only when necessary
noncomputable def doubled_arithmetic_mean_reciprocal_first_four_primes : ℚ :=
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, (1 : ℚ) / p)
  let arithmetic_mean := (reciprocals.sum / reciprocals.length)
  2 * arithmetic_mean

theorem doubled_arithmetic_mean_of_first_four_primes : doubled_arithmetic_mean_reciprocal_first_four_primes = 247 / 420 := by
  sorry

end doubled_arithmetic_mean_of_first_four_primes_l81_81248


namespace perpendicular_bisector_tangent_l81_81493

-- Definitions for the conditions
variable {F : Point} -- Focus of the parabola
variable {D : Point} -- Point on the directrix

-- Definition of a parabola
def is_parabola (P : Point) (F : Point) (D : Point) : Prop :=
  distance P F = distance P (perpendicular_projection D)

-- Definition of perpendicular bisector
def perpendicular_bisector (F : Point) (D : Point) : Line :=
  -- Suppose here we have a function that constructs the perpendicular bisector
  
noncomputable def tangent_at (P : Point) (l : Line) (C : Curve) : Prop :=
  -- Suppose here we have a definition of tangency of a line l to a curve C at point P
  
-- Statement to prove
theorem perpendicular_bisector_tangent (F D: Point) :
  tangent_at (perpendicular_bisector F D).point (perpendicular_bisector F D) (parabola F D) := 
sorry

end perpendicular_bisector_tangent_l81_81493


namespace distance_from_center_of_square_to_vertex_l81_81454

theorem distance_from_center_of_square_to_vertex (a b : ℕ) (ha : a = 3) (hb : b = 5) :
  ∃ (d : ℝ), d = sqrt ((a^2 + b^2) / 2) ∧ d = sqrt (34 / 4) :=
by {
  existsi sqrt (34 / 4),
  split,
  {
    rw [ha, hb],
    rw [pow_two, pow_two, add_comm _ (5:ℝ)],
    norm_num,
  },
  {
    norm_num,
  },
}

end distance_from_center_of_square_to_vertex_l81_81454


namespace m_necessary_not_sufficient_cond_l81_81985

theorem m_necessary_not_sufficient_cond (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0) → m ≤ 2 :=
sorry

end m_necessary_not_sufficient_cond_l81_81985


namespace volume_of_geometric_body_l81_81314

def cube_volume (s : ℝ) : ℝ := 
  if 0 ≤ s ∧ s ≤ 3 then 1 / 6 else 0

theorem volume_of_geometric_body (s : ℝ) (h0 : 0 ≤ s) (h1 : s ≤ 3) :
  ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 1 ∧ x + y + z = s → (cube_volume s = 1 / 6) :=
sorry

end volume_of_geometric_body_l81_81314


namespace vector_sum_closed_polygon_l81_81558

-- Definitions for vectors AB, BC, CF, and FA
variables (AB BC CF FA : ℝ^3)

-- Condition stating that the vectors form a closed polygon (polygon law of vectors)
def isClosedPolygon (AB BC CF FA : ℝ^3) : Prop :=
  AB + BC + CF + FA = 0

-- Theorem to prove that the sum of these vectors is zero given they form a closed polygon
theorem vector_sum_closed_polygon (h : isClosedPolygon AB BC CF FA) :
  AB + CF + BC + FA = 0 :=
by {
  sorry, -- Proof goes here
}

end vector_sum_closed_polygon_l81_81558


namespace dogs_not_liking_any_l81_81035

variables (totalDogs : ℕ) (dogsLikeWatermelon : ℕ) (dogsLikeSalmon : ℕ) (dogsLikeBothSalmonWatermelon : ℕ)
          (dogsLikeChicken : ℕ) (dogsLikeWatermelonNotSalmon : ℕ) (dogsLikeSalmonChickenNotWatermelon : ℕ)

theorem dogs_not_liking_any : totalDogs = 80 → dogsLikeWatermelon = 21 → dogsLikeSalmon = 58 →
  dogsLikeBothSalmonWatermelon = 12 → dogsLikeChicken = 15 →
  dogsLikeWatermelonNotSalmon = 7 → dogsLikeSalmonChickenNotWatermelon = 10 →
  (totalDogs - ((dogsLikeSalmon - (dogsLikeBothSalmonWatermelon + dogsLikeSalmonChickenNotWatermelon)) +
                (dogsLikeWatermelon - (dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon)) +
                (dogsLikeChicken - (dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) +
                dogsLikeBothSalmonWatermelon + dogsLikeWatermelonNotSalmon + dogsLikeSalmonChickenNotWatermelon)) = 13 :=
by
  intros h_totalDogs h_dogsLikeWatermelon h_dogsLikeSalmon h_dogsLikeBothSalmonWatermelon 
         h_dogsLikeChicken h_dogsLikeWatermelonNotSalmon h_dogsLikeSalmonChickenNotWatermelon
  sorry

end dogs_not_liking_any_l81_81035


namespace swim_back_time_l81_81222

theorem swim_back_time (v_s v_w t : ℝ) (h_vs : v_s = 10) (h_vw : v_w = 8) (h_t : t = 8) :
  (t * (v_s + v_w)) / (v_s - v_w) = 72 :=
by
  -- definitions for effective speeds and distance
  let v_swim_with := v_s + v_w
  let v_swim_against := v_s - v_w
  let distance := t * v_swim_with
  have v_swim_with_def : v_swim_with = v_s + v_w := rfl
  have v_swim_against_def : v_swim_against = v_s - v_w := rfl
  have distance_def : distance = t * v_swim_with := rfl
  
  -- using the given conditions
  rw [h_vs, h_vw, h_t, v_swim_with_def, v_swim_against_def, distance_def]
  -- calculation
  sorry

end swim_back_time_l81_81222


namespace not_possible_1998_points_l81_81160

theorem not_possible_1998_points (n : ℕ) : ∀ k : ℕ, (add_points n k) ≠ 1998 :=
by sorry

-- Helper function implementing the process of adding points
-- This function is given but not required to implement steps in detail.
def add_points : ℕ → ℕ → ℕ
| n 0 := n
| n (m+1) := add_points (2*n - 1) m

-- The theorem states that for any initial number of points n and any number of iterations k,
-- it is impossible to end up with 1998 points.

end not_possible_1998_points_l81_81160


namespace point_in_second_quadrant_l81_81978

def P : ℝ × ℝ := (-5, 4)

theorem point_in_second_quadrant (p : ℝ × ℝ) (hx : p.1 = -5) (hy : p.2 = 4) : p.1 < 0 ∧ p.2 > 0 :=
by
  sorry

example : P.1 < 0 ∧ P.2 > 0 :=
  point_in_second_quadrant P rfl rfl

end point_in_second_quadrant_l81_81978


namespace meteorological_observations_l81_81934

open Set

variables (α : Type*) [Fintype α]
variables (rained_in_morning rained_in_evening clear_morning clear_evening : α → Prop)

theorem meteorological_observations (h1 : ∀ d, rained_in_morning d → clear_evening d)
                                    (h2 : ∀ d, rained_in_evening d → clear_morning d)
                                    (h3 : card ({d | rained_in_morning d} ∪ {d | rained_in_evening d}) = 9)
                                    (h4 : card {d | clear_evening d} = 6)
                                    (h5 : card {d | clear_morning d} = 7) :
  Fintype.card α = 11 :=
sorry

end meteorological_observations_l81_81934


namespace arrangement_count_l81_81236

theorem arrangement_count {α : Type*} (A B C : α) (P₁ P₂ P₃ : α) :
  (∀ l : list α, l.perm [A, B, C, P₁, P₂, P₃] →
                  (l.indexOf A < l.indexOf C ∧ l.indexOf B > l.indexOf C) ∨
                  (l.indexOf A > l.indexOf C ∧ l.indexOf B < l.indexOf C)) →
  (2 * 3! = 240) :=
by simp; norm_num

end arrangement_count_l81_81236


namespace complex_calculation_l81_81249

theorem complex_calculation (i : ℂ) (hi : i * i = -1) : (1 - i)^2 * i = 2 :=
by
  sorry

end complex_calculation_l81_81249


namespace max_distance_on_ellipse_l81_81894

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l81_81894


namespace probability_number_is_factor_of_120_l81_81622

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81622


namespace permutation_probability_l81_81064

theorem permutation_probability (T : set (fin 6 → fin 6)) 
  (hT : ∀ σ ∈ T, σ 0 ≠ 0 ∧ σ 0 ≠ 1) :
  let favorable_permutations := {σ ∈ T | σ 2 = 2},
      num_favorable := fintype.card favorable_permutations,
      num_total := fintype.card T,
      prob := num_favorable / num_total in
  let a := prob.num,
      b := prob.denom in
  a + b = 23 :=
by
  -- The proof steps will go here, but for now we skip the proof
  sorry

end permutation_probability_l81_81064


namespace candy_in_one_bowl_l81_81730

theorem candy_in_one_bowl (total_candies : ℕ) (eaten_candies : ℕ) (bowls : ℕ) (taken_per_bowl : ℕ) 
  (h1 : total_candies = 100) (h2 : eaten_candies = 8) (h3 : bowls = 4) (h4 : taken_per_bowl = 3) :
  (total_candies - eaten_candies) / bowls - taken_per_bowl = 20 :=
by
  sorry

end candy_in_one_bowl_l81_81730


namespace chosen_number_probability_factorial_5_l81_81655

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81655


namespace probability_factor_of_5_factorial_l81_81688

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81688


namespace shaded_area_eq_seven_half_problem_I4_2_problem_I4_3_problem_I4_4_l81_81367

theorem shaded_area_eq_seven_half (a : ℚ) : 
    (∃ (A B C : ℝ × ℝ),
      A = (0, 3) ∧ B = (3, 0) ∧ C = (0, 0) ∧
      let area_large := 1/2 * (B.1 - A.1) * (A.2 - B.2) 
      in  ∃ (D E F : ℝ × ℝ),
      D = (0, 1) ∧ E = (1, 2) ∧ F = (3, 0) ∧
      let area_small := 1/2 * (E.1 - D.1) * (E.2 - D.2) 
      in a = area_large - area_small) →
    a = 7/2 := 
sorry

theorem problem_I4_2 (a : ℚ) : a = 3.5 → (b : ℚ) → 8^b = 4^a - 4^3 → b = 2 := 
sorry

theorem problem_I4_3 (b : ℚ) : b = 2 → 
    (c : ℝ) → (∃ (x : ℝ), c = x ∧ x > 0 ∧ x^2 - 100 * b + 10000 / x^2 = 0) → 
    c = 10 := 
sorry

theorem problem_I4_4 (c : ℚ) : c = 10 → 
    (d : ℚ) → 
    (d = ∑ n in finset.range (c-1), 1/(n * (n+1))) → 
    d = 9/10 := 
sorry

end shaded_area_eq_seven_half_problem_I4_2_problem_I4_3_problem_I4_4_l81_81367


namespace find_r_s_l81_81423

def N : Matrix (Fin 2) (Fin 2) Int := ![![3, 4], ![-2, 0]]
def I : Matrix (Fin 2) (Fin 2) Int := ![![1, 0], ![0, 1]]

theorem find_r_s :
  ∃ (r s : Int), (N * N = r • N + s • I) ∧ (r = 3) ∧ (s = 16) :=
by
  sorry

end find_r_s_l81_81423


namespace lizas_final_balance_l81_81088

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end lizas_final_balance_l81_81088


namespace factor_probability_l81_81594

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81594


namespace math_problem_proof_l81_81922

noncomputable def problem_statement (x y z v w : ℝ) : Prop :=
  (x^2 + y^2 + z^2 + v^2 + w^2 = 4032 ∧ 
   x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧ w > 0 → 
   let N := (xz + 2yz + 2z^2 + 3zv + 7zw) in 
   N + x + y + z + v + w = 52 + 2018 * sqrt 67)

-- Set up the theorem proving that the given conditions imply the required property
theorem math_problem_proof :
  ∀ (x y z v w : ℝ), 
    x > 0 → y > 0 → z > 0 → v > 0 → w > 0 →
    x^2 + y^2 + z^2 + v^2 + w^2 = 4032 →
    let N := (xz + 2yz + 2z^2 + 3zv + 7zw) in 
    N + x + y + z + v + w = 52 + 2018 * sqrt 67 := 
    sorry

end math_problem_proof_l81_81922


namespace num_divisible_by_5_l81_81243

theorem num_divisible_by_5 : 
  ∃ (s : Finset (Fin 4 → ℕ)), 
  s.card = 6 ∧ 
  ∀ f ∈ s, 
    (Set.ofFinset s ∈ { n | n % 10 = 5 } ∧ 
    Function.Injective f ∧ 
    (∀ n, n ∈ Set.range f → n ∈ {1, 2, 3, 5})) := 
sorry

end num_divisible_by_5_l81_81243


namespace travel_time_from_B_to_A_l81_81502

example (d : ℝ) (u_speed d_speed t_ab t_ba : ℝ)
  (h1 : d = 21)
  (h2 : u_speed = 4)
  (h3 : d_speed = 6)
  (h4 : t_ab = 4.25)
  (h5 : t_ab = (u_dist / u_speed) + ((d - u_dist) / d_speed) := by
    have h1d := h1
    rw [h1] at h1d
    let u_dist := d * (u_speed * d_speed) / (d_speed * u_speed)
    have := t_ba = ((d - u_dist) / u_speed) + (u_dist / d_speed)
    sorry

theorem travel_time_from_B_to_A :
  ∀ d u_speed d_speed t_ab t_ba : ℝ,
    d = 21 → 
    u_speed = 4 → 
    d_speed = 6 → 
    t_ab = 4.25 →
    t_ab = (u_dist / u_speed) + ((d - u_dist) / d_speed) →
    t_ba = 4.5 :=
begin
  intros d u_speed d_speed t_ab t_ba h1 h2 h3 h4 h5,
  let u_dist := 9,
  have h6 : u_dist = 9, from rfl,
  have h7 : d = 21, from h1,
  have h8 : u_speed = 4, from h2,
  have h9 : d_speed = 6, from h3,
  have h10 : t_ab = 4.25, from h4,
  have h11 : t_ab = (u_dist / u_speed) + ((d - u_dist) / d_speed), from h5,
  have h12 : (21 - u_dist = 12), from rfl,
  calc
    t_ba = (12 / 4) + (9 / 6) : by sorry  
        ... = 3 + 1.5 : by sorry
        ... = 4.5 : by sorry,
end

end travel_time_from_B_to_A_l81_81502


namespace exists_one_to_one_function_l81_81204

variable {A : Type*}
variable {S : set (A × A × A)}

-- Define the conditions as hypotheses
def cond1 (a b c : A) : Prop := (a, b, c) ∈ S ↔ (b, c, a) ∈ S
def cond2 (a b c : A) : Prop := (a, b, c) ∈ S ↔ (c, b, a) ∉ S
def cond3 (a b c d : A) : Prop := (a, b, c) ∈ S ∧ (c, d, a) ∈ S ↔ (b, c, d) ∈ S ∧ (d, a, b) ∈ S

theorem exists_one_to_one_function
  (h1 : ∀ a b c : A, cond1 a b c)
  (h2 : ∀ a b c : A, cond2 a b c)
  (h3 : ∀ a b c d : A, cond3 a b c d) :
  ∃ (g : A → ℝ), function.injective g ∧ (∀ a b c : A, g a < g b < g c → (a, b, c) ∈ S) :=
sorry

end exists_one_to_one_function_l81_81204


namespace solve_system_of_eqs_l81_81118

theorem solve_system_of_eqs :
  ∃ (x y z : ℝ), (x = -1) ∧ (y = 1) ∧ (z = 2) ∧
    (x^2 - 2 * y + 1 = 0) ∧
    (y^2 - 4 * z + 7 = 0) ∧
    (z^2 + 2 * x - 2 = 0) :=
by
  use [-1, 1, 2]
  simp only [pow_two, add_left_eq_self, eq_self_iff_true, sub_eq_add_neg, add_zero, mul_one, mul_neg, true_and, add_neg_cancel_right, sq]
  sorry

end solve_system_of_eqs_l81_81118


namespace roger_tray_capacity_l81_81949

theorem roger_tray_capacity (T : ℕ) (trip_count : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ) 
  (h_trip_count : trip_count = 3)
  (h_trays_table1 : trays_table1 = 10)
  (h_trays_table2 : trays_table2 = 2)
  (h_total_trays : 3 * T = trays_table1 + trays_table2) : 
  T = 4 := 
by 
  rw [h_trip_count, h_trays_table1, h_trays_table2] at h_total_trays
  exact Nat.div_eq_of_eq_mul' h_total_trays.symm sorry

end roger_tray_capacity_l81_81949


namespace probability_factor_of_120_in_range_l81_81610

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81610


namespace impossible_cube_formation_l81_81824

def is_valid_cube_formation (n : ℕ) (corner_pieces : ℕ) (individual_cubes : ℕ) : Prop :=
  ∃ (C : set (ℕ × ℕ × ℕ)), ∀ x ∈ C, by sorry -- Outline the conditions for valid cube formation
  -- Placeholder for the detailed conditions and definition
  C.nonempty ∧ C.size = n^3

theorem impossible_cube_formation :
  ¬ is_valid_cube_formation 3 7 6 :=
by
  intro h
  sorry

end impossible_cube_formation_l81_81824


namespace unique_rational_line_through_irrational_point_l81_81039

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def rational_point (p : ℝ × ℝ) : Prop := is_rational p.1 ∧ is_rational p.2

theorem unique_rational_line_through_irrational_point (a : ℝ) (ha : ¬ is_rational a) :
  ∃! l : ℝ → ℝ, (∃ p1 p2 : ℝ × ℝ, rational_point p1 ∧ rational_point p2 ∧ p1 ≠ p2 ∧ l = λ x, 0) ∧ l a = 0 :=
by
  sorry

end unique_rational_line_through_irrational_point_l81_81039


namespace fixed_point_of_lines_l81_81136

theorem fixed_point_of_lines
  (k : ℝ) : ∃ (P : ℝ × ℝ), P = (2, -1) ∧ ∀ k:ℝ, ∃ P, k * P.1 + P.2 + 1 = 2 * k := sorry

end fixed_point_of_lines_l81_81136


namespace scientific_notation_of_750000_l81_81697

theorem scientific_notation_of_750000 : 750000 = 7.5 * 10^5 :=
by
  sorry

end scientific_notation_of_750000_l81_81697


namespace unique_real_exists_l81_81442

theorem unique_real_exists 
  (a : ℕ → ℕ) 
  (h_a_nonneg : ∀ n, n ≤ 1997 → 0 ≤ a n)
  (h_inequality : ∀ i j, 1 ≤ i → 1 ≤ j → i + j ≤ 1997 
                  → a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1) :
  ∃! x : ℝ, ∀ n, 1 ≤ n ∧ n ≤ 1997 → a n = floor (n * x) :=
begin
  sorry
end

end unique_real_exists_l81_81442


namespace four_digit_numbers_formed_is_114_l81_81511

theorem four_digit_numbers_formed_is_114 :
  let cards := [1, 1, 1, 2, 2, 3, 4]
  in 
  let sets_of_4 := {s | s ⊆ cards ∧ s.card = 4}
  in
  let perm_count s := multiset.permutations (multiset.of_list s.to_list).to_finset.card
  in
  ∑ s in sets_of_4, perm_count s = 114 :=
by
  sorry

end four_digit_numbers_formed_is_114_l81_81511


namespace correct_propositions_count_l81_81512

-- Definitions of propositional conditions
def proposition1 : Prop := ∃ (p : ℕ), p = 5 ∧ (∃ (a b : ℕ), a + b ≤ p ∧ a ≥ 2 ∧ b ≤ 3)
def proposition2 : Prop := ∃ (d : ℕ), d = 12 ∧ (d * (d - 3)) / 2 = 54
def proposition3 : Prop := ∃ (n : ℕ), (n - 2) * 180 = 360

-- The primary property to prove: the number of correct propositions
def num_correct_propositions : ℕ := if proposition1 ∧ proposition2 ∧ proposition3 then 3 else 0

-- Lean theorem asserting the correctness of the number of propositions is 3
theorem correct_propositions_count : num_correct_propositions = 3 :=
by
  unfold num_correct_propositions
  split_ifs
  case pos h =>
    have h1 : proposition1 := sorry
    have h2 : proposition2 := sorry
    have h3 : proposition3 := sorry
    exact rfl
  case neg h =>
    exfalso
    intro hp
    contradiction

end correct_propositions_count_l81_81512


namespace student_net_pay_l81_81199

theorem student_net_pay (base_salary bonus : ℕ) (tax_rate : ℝ) (h₁ : base_salary = 25000) (h₂ : bonus = 5000)
  (h₃ : tax_rate = 0.13) : (base_salary + bonus - (base_salary + bonus) * tax_rate) = 26100 :=
by 
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end student_net_pay_l81_81199


namespace perimeter_of_inner_polygon_le_outer_polygon_l81_81102

-- Definitions of polygons (for simplicity considered as list of points or sides)
structure Polygon where
  sides : List ℝ  -- assuming sides lengths are given as list of real numbers
  convex : Prop   -- a property stating that the polygon is convex

-- Definition of the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := p.sides.sum

-- Conditions from the problem
variable {P_in P_out : Polygon}
variable (h_convex_in : P_in.convex) (h_convex_out : P_out.convex)
variable (h_inside : ∀ s ∈ P_in.sides, s ∈ P_out.sides) -- simplifying the "inside" condition

-- The theorem statement
theorem perimeter_of_inner_polygon_le_outer_polygon :
  perimeter P_in ≤ perimeter P_out :=
by {
  sorry
}

end perimeter_of_inner_polygon_le_outer_polygon_l81_81102


namespace nat_power_digit_condition_l81_81279

theorem nat_power_digit_condition (n k : ℕ) : 
  (10^(k-1) < n^n ∧ n^n < 10^k) → (10^(n-1) < k^k ∧ k^k < 10^n) → 
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end nat_power_digit_condition_l81_81279


namespace intersection_point_l81_81127

theorem intersection_point :
  ∃ (x y : ℝ), (y = 2 * x) ∧ (x + y = 3) ∧ (x = 1) ∧ (y = 2) := 
by
  sorry

end intersection_point_l81_81127


namespace product_constant_angle_case1_angle_case2_l81_81309

-- Define the elements given in the problem
variable (O A A' B B' C D : Point)
variable (R : ℝ) -- Radius
variable (circle : Circle O R) -- Circle with center O and radius R
variable (diameter1 : Diameter O A A') -- Diameter AA'
variable (diameter2 : Diameter O B B') -- Diameter BB' perpendicular to AA'
variable (chordAD : Chord A D O) -- Chord AD intersects BB' at C

-- The conditions provided in the problem
variable (CO_eq_CD : CO = CD) -- Case 1
variable (CD_eq_DA' : CD = DA') -- Case 2

-- The constant part of the product AC * AD
theorem product_constant (AC_AD_constant : AC * AD = 2 * R ^ 2) : 
  AC * AD = 2 * R ^ 2 := 
sorry -- Proof not required

-- Angle calculations for case 1
theorem angle_case1 (case1_condition : CO_eq_CD) :
  ∠D A A' = 30 :=
sorry -- Proof not required

-- Angle calculations for case 2
theorem angle_case2 (case2_condition : CD_eq_DA') :
  ∠D A A' = 22 + 30 / 60 :=
sorry -- Proof not required

end product_constant_angle_case1_angle_case2_l81_81309


namespace prob_factorial_5_l81_81673

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81673


namespace interval_increasing_l81_81350

noncomputable theory

def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

def shifted_and_symmetric (f : ℝ → ℝ) : Prop :=
∃ g : ℝ → ℝ, (∀ x, f (x + π/6) = g x) ∧ (∀ x, g (-x) = g x)

theorem interval_increasing :
  (∃ φ : ℝ, |φ| < π/2 ∧ shifted_and_symmetric (λ x, sin (2 * x + φ)))
  → is_increasing (λ x, sin (2 * x + π/6)) (set.Icc (-π/3) (π/6)) :=
by
  intro h
  sorry

end interval_increasing_l81_81350


namespace equal_distances_l81_81857

variable {ABC : Triangle}
variables (X_B X_C Q : Point)

-- Conditions
axiom nonisosceles_triangle (h1 : ¬ is_isosceles_triangle ABC)
axiom excenter_X_B (h2 : excenter X_B ABC B)
axiom excenter_X_C (h3 : excenter X_C ABC C)
axiom external_angle_bisector (h4 : external_angle_bisector_intersects_circumcircle_A ABC Q)

-- Statement
theorem equal_distances (h1 h2 h3 h4 : Prop) : dist Q X_B = dist Q B ∧ dist Q B = dist Q C ∧ dist Q C = dist Q X_C :=
sorry

end equal_distances_l81_81857


namespace geometric_progression_l81_81987

theorem geometric_progression (b q : ℝ) :
  (b + b*q + b*q^2 + b*q^3 = -40) ∧ 
  (b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280) →
  (b = 2 ∧ q = -3) ∨ (b = -54 ∧ q = -1/3) :=
by sorry

end geometric_progression_l81_81987


namespace sin_2theta_eq_neg_one_half_max_value_of_m3_squared_plus_n_squared_l81_81317

-- Define points A, B, and C with given coordinates
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (1, -1)
def C (θ : ℝ) : ℝ × ℝ := (sqrt 2 * cos θ, sqrt 2 * sin θ)

-- Define vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- Norm of a vector
def norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Problem 1: Prove that sin 2θ = -1/2 given the conditions
theorem sin_2theta_eq_neg_one_half (θ : ℝ) 
  (h : norm (vec B (C θ) - vec B A) = sqrt 2) : 
  sin (2 * θ) = -1 / 2 := 
sorry

-- Problem 2: Prove the maximum value of (m-3)^2 + n^2 is 16 given the conditions
theorem max_value_of_m3_squared_plus_n_squared 
  (θ θ_real : ℝ) 
  (m n : ℝ)
  (h : m * (vec 0 A).1 + n * (vec 0 B).1 = (vec 0 (C θ)).1 
       ∧ m * (vec 0 A).2 + n * (vec 0 B).2 = (vec 0 (C θ)).2) : 
  ∃ m n : ℝ, 
  (m - 3) ^ 2 + n ^ 2 = 16 :=
sorry

end sin_2theta_eq_neg_one_half_max_value_of_m3_squared_plus_n_squared_l81_81317


namespace number_of_organizations_in_foundation_l81_81699

def company_raised : ℕ := 2500
def donation_percentage : ℕ := 80
def each_organization_receives : ℕ := 250
def total_donated : ℕ := (donation_percentage * company_raised) / 100

theorem number_of_organizations_in_foundation : total_donated / each_organization_receives = 8 :=
by
  sorry

end number_of_organizations_in_foundation_l81_81699


namespace chosen_number_probability_factorial_5_l81_81654

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81654


namespace product_alpha_implies_product_beta_l81_81788

variable {α : Type} [LinearOrderedField α]

/-- Given distinct real numbers a₁, a₂, ..., aₙ and real numbers b₁, b₂, ..., bₙ
such that there exists α in ℝ fulfilling ∏_{1 ≤ k ≤ n} (a_i + b_k) = α for i = 1, 2, ..., n,
prove that there exists β in ℝ fulfilling ∏_{1 ≤ k ≤ n} (a_k + b_j) = β for j = 1, 2, ..., n. -/
theorem product_alpha_implies_product_beta (a b : ℕ → α) (n : ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_exists_alpha : ∃ α, ∀ i, 1 ≤ i ∧ i ≤ n → (∏ k in Finset.range n, a i + b k) = α) :
  ∃ β, ∀ j, 1 ≤ j ∧ j ≤ n → (∏ k in Finset.range n, a k + b j) = β :=
by
  let α := Classical.choose h_exists_alpha
  let β := (-1 : α) ^ (n + 1) * α
  use β
  sorry

end product_alpha_implies_product_beta_l81_81788


namespace problem1_proof_problem2_proof_l81_81736

-- Definitions for Problem 1
def problem1_expr : Int := (1) - (-4) + (-1) - (+5)
def problem1_answer : Int := -1

theorem problem1_proof : problem1_expr = problem1_answer := by
  -- Proof is omitted
  sorry

-- Definitions for Problem 2
def problem2_expr : Int := -1 ^ 4 + abs (5 - 8) + 27 / (-3) * (1 / 3)
def problem2_answer : Int := -1

theorem problem2_proof : problem2_expr = problem2_answer := by
  -- Proof is omitted
  sorry

end problem1_proof_problem2_proof_l81_81736


namespace find_n_l81_81772

theorem find_n (n : ℕ) (a : ℕ → ℕ) 
  (H1 : (1 + x)^n = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + ... + a n * x^n)
  (H2 : a 2 / a 3 = 1 / 3) : n = 5 := by sorry

end find_n_l81_81772


namespace probability_factor_of_120_l81_81650

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81650


namespace number_of_special_pairs_is_even_l81_81092

-- Definitions for the problem context
def Polygon (V : Type) [Finite V] :=
{ sides : list (V × V) // ∀ (u v w : V), (u, v) ∈ sides → (v, w) ∈ sides → u ≠ w }

-- Definition of special pairs
def is_special_pair {V : Type} [Finite V] (p : Polygon V) (a b c d : V) :=
  (a, b) ∈ p.1.sides ∧ (c, d) ∈ p.1.sides ∧
  ¬(b == c ∨ d == a) ∧ -- non-adjacent segments
  (∃ (x : V), x != a ∧ x != c ∧ 
              (a, x) ∈ p.1.sides ∧ (c, x) ∈ p.1.sides)

-- Formal statement of the theorem
theorem number_of_special_pairs_is_even {V : Type} [Finite V] 
  (p : Polygon V) (h1 : Closed_Polygonal_Chain p) (h2 : No_Three_Verts_Collinear p) : 
  ∃ n : ℕ, n % 2 = 0 ∧ 
           n = (List.cardinality (List.filter (λ (quad: (V × V) × (V × V)), is_special_pair p quad.fst.fst quad.fst.snd quad.snd.fst quad.snd.snd) p.1.sides)) := 
sorry

end number_of_special_pairs_is_even_l81_81092


namespace chosen_number_probability_factorial_5_l81_81661

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81661


namespace max_distance_on_ellipse_l81_81905

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l81_81905


namespace algebraic_expression_domain_l81_81018

theorem algebraic_expression_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 2)) ↔ (x ≠ -2) := 
sorry

end algebraic_expression_domain_l81_81018


namespace relationship_among_m_n_p_l81_81307

noncomputable def m : ℝ := 0.9 ^ 5.1
noncomputable def n : ℝ := 5.1 ^ 0.9
noncomputable def p : ℝ := Real.logBase 0.9 5.1

theorem relationship_among_m_n_p : p < m ∧ m < n :=
by
  -- Directly use definitions and state the relationships to be proved
  have h_cond_m : 0 < m ∧ m < 1 := sorry
  have h_cond_n : n > 1 := sorry
  have h_cond_p : p < 0 := sorry
  sorry

end relationship_among_m_n_p_l81_81307


namespace prob_equals_two_yellow_marbles_l81_81585

noncomputable def probability_two_yellow_marbles : ℚ :=
  let total_marbles : ℕ := 3 + 4 + 8
  let yellow_marbles : ℕ := 4
  let first_draw_prob : ℚ := yellow_marbles / total_marbles
  let second_total_marbles : ℕ := total_marbles - 1
  let second_yellow_marbles : ℕ := yellow_marbles - 1
  let second_draw_prob : ℚ := second_yellow_marbles / second_total_marbles
  first_draw_prob * second_draw_prob

theorem prob_equals_two_yellow_marbles :
  probability_two_yellow_marbles = 2 / 35 :=
by
  sorry

end prob_equals_two_yellow_marbles_l81_81585


namespace coefficient_of_x5_in_expansion_l81_81489

-- Define the polynomial expansion of (x-1)(x+1)^8
def polynomial_expansion (x : ℚ) : ℚ :=
  (x - 1) * (x + 1) ^ 8

-- Define the binomial coefficient function
def binom_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem: The coefficient of x^5 in the expansion of (x-1)(x+1)^8 is 14
theorem coefficient_of_x5_in_expansion :
  binom_coeff 8 4 - binom_coeff 8 5 = 14 :=
sorry

end coefficient_of_x5_in_expansion_l81_81489


namespace midpoint_center_tangents_perpendicular_to_diameter_l81_81776

-- Assuming the existence of a circle and equilateral hyperbola intersecting at four points.
variables {α : Type*} [field α]
variables (a D E F : α)

-- Conditions on the hyperbola and circle
def hyperbola (x y : α) := x * y = a
def circle (x y : α) := x^2 + y^2 + D * x + E * y + F = 0

-- Intersection points
variables (x1 y1 x2 y2 x3 y3 x4 y4 : α)
def A1 := (x1, y1)
def A2 := (x2, y2)
def A3 := (x3, y3)
def A4 := (x4, y4)

-- Given conditions: A1 and A2 are endpoints of a diameter of the circle
axiom diameter_cond : x1 + x2 = -D

-- Midpoint of A3 and A4
def midpoint (A3 A4 : α × α) := ((A3.1 + A4.1) / 2, (A3.2 + A4.2) / 2)

-- Correct answer for part 1
theorem midpoint_center : midpoint (A3 x3 y3) (A4 x4 y4) = (0, 0) := sorry

-- Correct answer for part 2
theorem tangents_perpendicular_to_diameter : 
    let k1 := -a / (x3 * x3), 
        k2 := -a / (x1 * x2) in k1 * k2 = -1 := sorry

end midpoint_center_tangents_perpendicular_to_diameter_l81_81776


namespace well_performing_student_take_home_pay_l81_81193

theorem well_performing_student_take_home_pay : 
  ∃ (base_salary bonus : ℕ) (income_tax_rate : ℝ),
      (base_salary = 25000) ∧ (bonus = 5000) ∧ (income_tax_rate = 0.13) ∧
      let total_earnings := base_salary + bonus in
      let income_tax := total_earnings * income_tax_rate in
      total_earnings - income_tax = 26100 :=
by
  use 25000
  use 5000
  use 0.13
  intros
  sorry

end well_performing_student_take_home_pay_l81_81193


namespace decreasing_function_range_l81_81812

noncomputable def f (a x : ℝ) := a * (x^3) - x + 1

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ 0 := by
  sorry

end decreasing_function_range_l81_81812


namespace accurate_value_l81_81034

noncomputable def D : ℝ := 3.72581
noncomputable def error : ℝ := 0.00229

theorem accurate_value (D_upper D_lower : ℝ) (hD_upper : D_upper = D + error) (hD_lower : D_lower = D - error) : 
  round (D_upper * 10) / 10 = 3.7 ∧ round (D_lower * 10) / 10 = 3.7 :=
by
  sorry

end accurate_value_l81_81034


namespace smallest_among_neg2_cube_neg3_square_neg_neg1_l81_81244

def smallest_among (a b c : ℤ) : ℤ :=
if a < b then
  if a < c then a else c
else
  if b < c then b else c

theorem smallest_among_neg2_cube_neg3_square_neg_neg1 :
  smallest_among ((-2)^3) (-(3^2)) (-(-1)) = -(3^2) :=
by
  sorry

end smallest_among_neg2_cube_neg3_square_neg_neg1_l81_81244


namespace eddy_time_to_B_l81_81268

-- Definitions
def distance_A_to_B : ℝ := 570
def distance_A_to_C : ℝ := 300
def time_C : ℝ := 4
def speed_ratio : ℝ := 2.5333333333333333

-- Theorem Statement
theorem eddy_time_to_B : 
  (distance_A_to_B / (distance_A_to_C / time_C * speed_ratio)) = 3 := 
by
  sorry

end eddy_time_to_B_l81_81268


namespace minimum_product_of_chessboard_labels_l81_81743

theorem minimum_product_of_chessboard_labels :
  let f := λ i j => 1 / (i + j - 1 : ℝ),
      chosen := [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9)]
  in (∏ p in chosen, f p.1 p.2) = (1 / (Nat.factorial 9 : ℝ)) :=
by
  sorry

end minimum_product_of_chessboard_labels_l81_81743


namespace sum_of_coeffs_sum_indexed_coeffs_l81_81303

noncomputable def polynomial_expansion := 
  λ (x : ℝ), (2 - x)^10

noncomputable def polynomial_coeffs := 
  λ (x : ℝ), ∑ i in Finset.range 11, a i * x ^ i

theorem sum_of_coeffs {a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ} :
  (polynomial_expansion 1 = polynomial_coeffs 1) → ∀ x, 
  (2 - x)^10 = ∑ i in Finset.range 11, a i * x ^ i → 
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10}) = 1 :=
by sorry

theorem sum_indexed_coeffs {a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ} :
  (∀ (x : ℝ), ((2 - x)^10).derivative x = polynomial_coeffs.derivative x) → 
  ((10 * (-1)^9) = 
   (a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 + 6*a_6 + 7*a_7 + 8*a_8 + 9*a_9 + 10*a_{10})) :=
by sorry

end sum_of_coeffs_sum_indexed_coeffs_l81_81303


namespace first_player_wins_l81_81156

theorem first_player_wins : 
  ∃ (strategy: ℕ → bool), -- strategy is a function that decides which pile to take from
  ∀ (turn: ℕ), -- turn is the number of turns taken
  let remaining_nuts := 10 - turn in
  if remaining_nuts = 3 then -- when 3 nuts are left
    ∀ (pile1 pile2 pile3 : ℕ), pile1 + pile2 + pile3 = 3 -> -- three piles sum up to 3
    ¬ (pile1 = 1 ∧ pile2 = 1 ∧ pile3 = 1) -- these should not be in three separate piles 
  else
    (turn % 2 = 0 → strategy turn = true) ∧ -- first player's turns follow the strategy
    (turn % 2 = 1 → strategy turn = false)  -- second player's turns follow the strategy
:= sorry

end first_player_wins_l81_81156


namespace geralds_average_speed_l81_81466

theorem geralds_average_speed :
  ∀ (track_length : ℝ) (pollys_laps : ℕ) (pollys_time : ℝ) (geralds_factor : ℝ),
  track_length = 0.25 →
  pollys_laps = 12 →
  pollys_time = 0.5 →
  geralds_factor = 0.5 →
  (geralds_factor * (pollys_laps * track_length / pollys_time)) = 3 :=
by
  intro track_length pollys_laps pollys_time geralds_factor
  intro h_track_len h_pol_lys_laps h_pollys_time h_ger_factor
  sorry

end geralds_average_speed_l81_81466


namespace area_of_PQRS_l81_81462

noncomputable def length_square_EFGH := 6
noncomputable def height_equilateral_triangle := 3 * Real.sqrt 3
noncomputable def diagonal_PQRS := length_square_EFGH + 2 * height_equilateral_triangle
noncomputable def area_PQRS := (1 / 2) * (diagonal_PQRS * diagonal_PQRS)

theorem area_of_PQRS :
  (area_PQRS = 72 + 36 * Real.sqrt 3) :=
sorry

end area_of_PQRS_l81_81462


namespace g_satisfies_functional_eq_l81_81434

-- Define the function g
def g (x : ℝ) : ℝ := x + 3

-- Statement of the proof
theorem g_satisfies_functional_eq (x y : ℝ) :
  (g(x) * g(y) - g(x * y)) / 5 = x + y + 2 :=
by
  -- Proof goes here
  sorry

end g_satisfies_functional_eq_l81_81434


namespace sequence_inequality_l81_81926

open Complex

theorem sequence_inequality (a b : ℝ) (hapos : 0 < a) (hbpos : 0 < b)
  (a_n b_n : ℕ → ℝ) (n : ℕ) (hnpos : 0 < n)
  (h_seq : ∀ n, (a + b * I) ^ n = complex.ofReal (a_n n) + complex.ofReal (b_n n) * I) :
  (|complex.ofReal (a_n (n+1))| + |complex.ofReal (b_n (n+1))|) / 
  (|complex.ofReal (a_n n)| + |complex.ofReal (b_n n)|) ≥ 
  (a^2 + b^2) / (a + b) :=
sorry

end sequence_inequality_l81_81926


namespace miranda_saved_per_month_l81_81082

-- Definition of the conditions and calculation in the problem
def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def months : ℕ := 3
def miranda_savings : ℕ := total_cost - sister_contribution
def saved_per_month : ℕ := miranda_savings / months

-- Theorem statement with the expected answer
theorem miranda_saved_per_month : saved_per_month = 70 :=
by
  sorry

end miranda_saved_per_month_l81_81082


namespace equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l81_81151

-- Definitions for supply and demand functions
def Qs (p : ℝ) : ℝ := 2 + 8 * p
def Qd (p : ℝ) : ℝ := -2 * p + 12

-- Equilibrium without subsidy
theorem equilibrium_price_without_subsidy : (∃ p q, Qs p = q ∧ Qd p = q ∧ p = 1 ∧ q = 10) :=
sorry

-- New supply function with subsidy
def Qs_with_subsidy (p : ℝ) : ℝ := 10 + 8 * p

-- Increase in quantity sold due to subsidy
theorem increase_in_quantity_due_to_subsidy : 
  (∃ Δq, Δq = Qd 0.2 - Qd 1 ∧ Δq = 1.6) :=
sorry

end equilibrium_price_without_subsidy_increase_in_quantity_due_to_subsidy_l81_81151


namespace tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l81_81930

def tight_sequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → (1/2 : ℚ) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)

noncomputable def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = a 1 * q ^ (n - 1)

theorem tight_sequence_from_sum_of_terms (S : ℕ → ℚ) (a : ℕ → ℚ) : 
  (∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)) →
  (∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) →
  tight_sequence a :=
sorry

theorem range_of_q_for_tight_sequences (a : ℕ → ℚ) (S : ℕ → ℚ) (q : ℚ) :
  geometric_sequence a q →
  tight_sequence a →
  tight_sequence S →
  (1 / 2 : ℚ) ≤ q ∧ q < 1 :=
sorry

end tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l81_81930


namespace product_of_extremes_of_multiples_l81_81377

theorem product_of_extremes_of_multiples (x : ℤ)
  (h1 : ∀ i : ℕ, i < 15 → (3 * x + 3 * i)) -- Sequence of 15 consecutive multiples of 3
  (h2 : (∑ i in finset.range 15, (3 * x + 3 * i)) / 15 = 45) -- Average is 45
  : (3 * x) * (3 * x + 3 * 14) = 1584 := 
sorry

end product_of_extremes_of_multiples_l81_81377


namespace monthly_earnings_l81_81114

-- Defining the initial conditions and known information
def current_worth : ℝ := 90
def months : ℕ := 5

-- Let I be the initial investment, and E be the earnings per month.

noncomputable def initial_investment (I : ℝ) := I * 3 = current_worth
noncomputable def earned_twice_initial (E : ℝ) (I : ℝ) := E * months = 2 * I

-- Proving the monthly earnings
theorem monthly_earnings (I E : ℝ) (h1 : initial_investment I) (h2 : earned_twice_initial E I) : E = 12 :=
sorry

end monthly_earnings_l81_81114


namespace inscribed_octagon_diameter_l81_81720

theorem inscribed_octagon_diameter (a b c d e f g h : ℝ) (r : ℝ) 
  (h1 : a = 4) (h2 : b = 6) 
  (h3 : a = d) (h4 : a = f) (h5 : a = h) 
  (h6 : b = c) (h7 : b = e) (h8 : b = g) :
  2 * r = 6 * (sqrt (2 + sqrt 2)) :=
sorry

end inscribed_octagon_diameter_l81_81720


namespace flowers_bees_butterflies_comparison_l81_81988

def num_flowers : ℕ := 12
def num_bees : ℕ := 7
def num_butterflies : ℕ := 4
def difference_flowers_bees : ℕ := num_flowers - num_bees

theorem flowers_bees_butterflies_comparison :
  difference_flowers_bees - num_butterflies = 1 :=
by
  -- The proof will go here
  sorry

end flowers_bees_butterflies_comparison_l81_81988


namespace sector_perimeter_l81_81503

noncomputable def perimeter_of_sector (θ : ℝ) (r : ℝ) : ℝ :=
  let L := (θ / 360) * 2 * Real.pi * r
  in L + 2 * r

theorem sector_perimeter :
  perimeter_of_sector 180 28.000000000000004 = 143.96459430079216 :=
by
  simp [perimeter_of_sector]
  have pi_approx : Real.pi ≈ 3.141592653589793 := by norm_num
  calc
    (180 / 360) * 2 * Real.pi * 28.000000000000004 + 2 * 28.000000000000004
    = 1 * Real.pi * 28.000000000000004 + 2 * 28.000000000000004 : by norm_num
    ... ≈ 3.141592653589793 * 28.000000000000004 + 2 * 28.000000000000004 : by rw [pi_approx]
    ... ≈ 87.96459430079215 + 2 * 28.000000000000004 : by norm_num
    ... ≈ 87.96459430079216 + 56.00000000000001 : by norm_num
    ... ≈ 143.96459430079216 : by norm_num

end sector_perimeter_l81_81503


namespace problem_statement_l81_81767

def diamond (x y : ℝ) : ℝ := (x + y) ^ 2 * (x - y) ^ 2

theorem problem_statement : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end problem_statement_l81_81767


namespace parabola_focus_distance_l81_81060

theorem parabola_focus_distance
  (F P Q : ℝ × ℝ)
  (hF : F = (1 / 2, 0))
  (hP : ∃ y, P = (2 * y^2, y))
  (hQ : Q = (1 / 2, Q.2))
  (h_parallel : P.2 = Q.2)
  (h_distance : dist P Q = dist Q F) :
  dist P F = 2 :=
by
  sorry

end parabola_focus_distance_l81_81060


namespace num_neg_nums_in_set_l81_81242

theorem num_neg_nums_in_set : 
  let S := {-8, 0, -3^2, -(-5.7)}
  in (S.filter (λ x, x < 0)).card = 2 := 
by
  let S := {-8, 0, -3^2, -(-5.7)}
  have hs1 : -8 < 0 := by norm_num
  have hs2 : 0 ≥ 0 := by apply ge_of_eq; norm_num  
  have hs3 : -3^2 < 0 := by norm_num
  have hs4 : -(-5.7) ≥ 0 := by norm_num
  let N := S.filter (λ x, x < 0)
  have hn1 : N = {-8, -3^2} := by simp [hs1, hs3]
  show N.card = 2, from sorry

end num_neg_nums_in_set_l81_81242


namespace find_maximum_b_l81_81796

-- Given conditions
variables (a b : ℝ)

-- Hypotheses
hypothesis h1 : a > 0
hypothesis h2 : b > 0
hypothesis h3 : a + 3 * b = 1 / b - 1 / a

-- The statement to be proven
theorem find_maximum_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1 / b - 1 / a) : b <= 1 / 3 :=
by
  sorry -- Proof to be filled in;

end find_maximum_b_l81_81796


namespace min_length_segment_cut_l81_81972

noncomputable def ellipse_equation : Prop := 
  ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 → true

noncomputable def curve_C (t : ℝ) : Prop := 
  0 < t ∧ t ≤ Real.sqrt 2 / 2 → 
  ∀ x y : ℝ, (x - t)^2 + y^2 = (t^2 + 2 * t)^2 → true

theorem min_length_segment_cut
  (hx : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1) 
  (ht : ∀ t : ℝ, 0 < t ∧ t ≤ Real.sqrt 2 / 2 → 
      ∀ x y : ℝ, (x - t)^2 + y^2 = (t^2 + 2 * t)^2)
  (A : ℝ → ℝ → Prop)
  : ∃ k : ℝ, (0 < k^2 ∧ k^2 ≤ 1) → 
    let n := Real.sqrt (k^2 + 1) in 
    (1 < n ∧ n ≤ Real.sqrt 2) → 
    (12 * n / (4 * n - 1 / n) = 12 * Real.sqrt 2 / 7) :=
sorry

end min_length_segment_cut_l81_81972


namespace ball_placement_problem_l81_81097

/-- 
  Prove that the number of ways to place four balls labeled A, B, C, and D
  into three boxes numbered 1, 2, and 3, such that each box contains at least one ball,
  and balls A and B cannot be placed in the same box, is 30.
 -/
theorem ball_placement_problem : 
  let S := { A, B, C, D }
  let B := { 1, 2, 3 }
  (∀ (f : S → B), (∀ b ∈ B, ∃ a ∈ S, f a = b) ∧ (f A ≠ f B)) → 
  (number_of_ways = 30) :=
by
  sorry

end ball_placement_problem_l81_81097


namespace magnitude_of_angle_A_max_value_l81_81305

-- Define the given conditions
variables (A B C : ℝ) (a b c : ℝ)
def vec_m := (a, -2 * b - c)
def vec_n := (Real.cos A, Real.cos C)
def parallel (m n : ℝ × ℝ) := ∃ k : ℝ, m = (k * n.1, k * n.2)

-- Define the proof problems
theorem magnitude_of_angle_A (h : parallel (vec_m a b c) (vec_n A C)) : A = 2 * Real.pi / 3 :=
sorry

theorem max_value (A_eq : A = 2 * Real.pi / 3) :
  ∀ B C : ℝ, 0 < C ∧ C < Real.pi / 3 →
  (B = Real.pi / 3 - C) →
  2 * Real.sqrt 3 * Real.cos C / 2 ^ 2 - Real.sin (B - Real.pi / 3) ≤ Real.sqrt 3 + 2 :=
sorry

end magnitude_of_angle_A_max_value_l81_81305


namespace max_distance_on_ellipse_l81_81904

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l81_81904


namespace seven_digit_pos_integers_start_end_composite_count_l81_81256

theorem seven_digit_pos_integers_start_end_composite_count :
  let composite_digits := {4, 6, 8, 9}
  let count := (4 * 10^6) + (9 * 10^5 * 4) - (4 * 10^5 * 4)
  count = 600000 := by
    let composite_digits := {4, 6, 8, 9}
    let count_start := 4 * 10^6
    let count_end := 9 * 10^5 * 4
    let count_both := 4 * 10^5 * 4
    let count := count_start + count_end - count_both
    have : count = 600000 := sorry
    exact this

end seven_digit_pos_integers_start_end_composite_count_l81_81256


namespace integral_1_eq_integral_2_eq_l81_81732

variable (f : ℝ → ℝ)
noncomputable def integral_1 : ℝ :=
  ∫ x in -3..2, abs (x + 1)

theorem integral_1_eq :
  integral_1 (λ x, abs (x + 1)) = 13 / 2 :=
by sorry

noncomputable def f_piecewise (x: ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x ^ 2 else if 1 ≤ x ∧ x ≤ 2 then 2 - x else 0

noncomputable def integral_2 : ℝ :=
  ∫ x in (0:ℝ)..2, f_piecewise x

theorem integral_2_eq :
  integral_2 f_piecewise = 5 / 6 :=
by sorry

end integral_1_eq_integral_2_eq_l81_81732


namespace percent_blue_marbles_l81_81721

theorem percent_blue_marbles (total_items buttons red_marbles : ℝ) 
  (H1 : buttons = 0.30 * total_items)
  (H2 : red_marbles = 0.50 * (total_items - buttons)) :
  (total_items - buttons - red_marbles) / total_items = 0.35 :=
by 
  sorry

end percent_blue_marbles_l81_81721


namespace probability_factor_of_5_factorial_l81_81695

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81695


namespace sufficiency_not_necessity_l81_81428

def l1 : Type := sorry
def l2 : Type := sorry

def skew_lines (l1 l2 : Type) : Prop := sorry
def do_not_intersect (l1 l2 : Type) : Prop := sorry

theorem sufficiency_not_necessity (p q : Prop) 
  (hp : p = skew_lines l1 l2)
  (hq : q = do_not_intersect l1 l2) :
  (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end sufficiency_not_necessity_l81_81428


namespace slope_of_intersection_points_l81_81961

theorem slope_of_intersection_points :
  ∀ (x y : ℝ), (x^2 + y^2 - 6 * x + 4 * y - 20 = 0) ∧ (x^2 + y^2 - 8 * x + 18 * y + 40 = 0) →
    (∃ (m : ℝ), m = 1 / 7) :=
by
  intro x y h
  use 1 / 7
  sorry

end slope_of_intersection_points_l81_81961


namespace number_of_AB_students_correct_l81_81584

-- Definitions of conditions
def total_students : ℕ := 500
def blood_type_AB_students : ℕ := 50
def sample_size : ℕ := 60
def selection_probability : ℚ := sample_size / total_students

-- Definition and statement of the proof
def number_of_AB_students_to_be_drawn : ℕ :=
  blood_type_AB_students * selection_probability

theorem number_of_AB_students_correct :
  number_of_AB_students_to_be_drawn = 6 :=
by
  -- Proof using the given conditions and calculations (skipped here with "sorry")
  sorry

end number_of_AB_students_correct_l81_81584


namespace count_palindromic_times_on_12_hour_clock_l81_81210

theorem count_palindromic_times_on_12_hour_clock : 
  ∃ n : ℕ, n = 56 ∧ (∀ h m : ℕ, 1 ≤ h ∧ h ≤ 12 → 0 ≤ m ∧ m < 60 →
  (let s := if h < 10 then (h.repr ++ ":" ++ m.repr) else ((h.repr) ++ ":" ++ (m.repr)) in 
  ∀ s1 s2 s3 s4, (s = s1 ++ s2 ++ ":" ++ s3 ++ s4) → s1 == s4 ∧ s2 == s3)) sorry

end count_palindromic_times_on_12_hour_clock_l81_81210


namespace dried_grapes_water_percentage_l81_81294

def percentage_water_in_dried_grapes (fresh_grapes_weight dry_grapes_weight : ℚ) (water_percentage_in_fresh_grapes : ℚ) : ℚ :=
  let water_weight_in_fresh := fresh_grapes_weight * water_percentage_in_fresh_grapes in
  let solid_weight_in_fresh := fresh_grapes_weight - water_weight_in_fresh in
  let water_weight_in_dried := dry_grapes_weight - solid_weight_in_fresh in
  (water_weight_in_dried / dry_grapes_weight) * 100

theorem dried_grapes_water_percentage : percentage_water_in_dried_grapes 25 3.125 0.9 = 20 := by
  sorry

end dried_grapes_water_percentage_l81_81294


namespace tan_identity_l81_81020

open Real

-- Definition of conditions
def isPureImaginary (z : Complex) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem tan_identity (theta : ℝ) :
  isPureImaginary ((cos theta - 4/5) + (sin theta - 3/5) * Complex.I) →
  tan (theta - π / 4) = -7 :=
by
  sorry

end tan_identity_l81_81020


namespace subtract_and_convert_l81_81481

theorem subtract_and_convert : (3/4 - 1/16 : ℚ) = 0.6875 :=
by
  sorry

end subtract_and_convert_l81_81481


namespace tessellate_groups_correct_l81_81042

-- Definitions for the groups of polygons based on the given problem conditions
def Group1 := (polygon : String) -> polygon = "equilateral_triangle" ∨ polygon = "square"
def Group2 := (polygon : String) -> polygon = "equilateral_triangle" ∨ polygon = "regular_hexagon"
def Group3 := (polygon : String) -> polygon = "regular_hexagon" ∨ polygon = "square"
def Group4 := (polygon : String) -> polygon = "regular_octagon" ∨ polygon = "square"

-- Define the tessellation condition based on interior angle fitting
def can_tessellate (group : (String -> Prop)) : Prop :=
  -- Interior angles fitting rules must be detailed here, e.g., for equilateral_triangle = 60°, square = 90°, etc.
  sorry

-- Theorem to be proved
theorem tessellate_groups_correct : 
  (can_tessellate Group1) → (can_tessellate Group2) → (¬ can_tessellate Group3) → (can_tessellate Group4) → 
  ∀ group, (group = Group1 ∨ group = Group2 ∨ group = Group4) :=
begin
  sorry -- Proof not required
end

end tessellate_groups_correct_l81_81042


namespace ratio_x_to_w_as_percentage_l81_81075

theorem ratio_x_to_w_as_percentage (x y z w : ℝ) 
    (h1 : x = 1.20 * y) 
    (h2 : y = 0.30 * z) 
    (h3 : z = 1.35 * w) : 
    (x / w) * 100 = 48.6 := 
by sorry

end ratio_x_to_w_as_percentage_l81_81075


namespace max_PB_distance_l81_81881

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l81_81881


namespace find_counterfeit_coins_l81_81941

-- Definitions and assumptions based on the conditions
axiom five_coins (coins : Fin 5 → ℝ) : Prop
axiom two_counterfeit (coins : Fin 5 → ℝ) : Prop
axiom one_lighter_one_heavier (coins : Fin 5 → ℝ) : Prop

-- The actual theorem statement
theorem find_counterfeit_coins (coins : Fin 5 → ℝ) 
    (H1 : five_coins coins) 
    (H2 : two_counterfeit coins)
    (H3 : one_lighter_one_heavier coins) : 
    ∃ (c1 c2 : Fin 5), c1 ≠ c2 ∧ 
    (weights_equality c1 c2 ∨ weights_inequality c1 c2) := 
sorry

-- Additional definitions that represent the balance scale weighing condition
def weights_equality (c1 c2 : Fin 5) : Prop :=
-- Define equality condition based on the coins weight
sorry

def weights_inequality (c1 c2 : Fin 5) : Prop :=
-- Define inequality condition based on the coins weight
sorry

end find_counterfeit_coins_l81_81941


namespace intersection_A_B_l81_81357

def A := {-1, 0, 1}
def B := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_A_B_l81_81357


namespace find_m_l81_81375

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : (2 : ℝ) ≠ m) (h3 : 2 * m = (m - 4) / (2 - m)) : 
  m = (3 + Real.sqrt 41) / 4 :=
by
  sorry

end find_m_l81_81375


namespace probability_factor_of_120_l81_81647

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81647


namespace probability_number_is_factor_of_120_l81_81629

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81629


namespace find_x_l81_81144

noncomputable def base23_repr (a : ℕ) (m : ℕ) : ℕ :=
  a * (23^0 + 23^1 + ... + 23^(2*m-1))

theorem find_x (a b x : ℕ) (m : ℕ) (h1 : a * (23^0 + 23^1 + ... + 23^(2*m-1)) = x)
  (h2 : x^2 = b * (1 + 23^(2*(2*m)-1))) :
  x = 13 * (23^0 + 23^1 + ... + 23^(2*m-1)) := by
  sorry

end find_x_l81_81144


namespace tom_stops_at_two_houses_l81_81166

def tom_position_over_time (time : ℝ) : ℝ := sorry -- add the actual function representation here

def is_stationary (time : ℝ) : Prop :=
∃ t_start t_end : ℝ, t_start < time ∧ time < t_end ∧ 
                      ∀ t, t_start ≤ t ∧ t ≤ t_end → tom_position_over_time t = tom_position_over_time time

theorem tom_stops_at_two_houses :
  ∃ times : list ℝ, list.length times = 2 ∧
    ∀ t ∈ times, is_stationary t :=
sorry

end tom_stops_at_two_houses_l81_81166


namespace probability_dice_face_5_l81_81950

theorem probability_dice_face_5 :
  let fair_dice := True -- condition that dice is fair
  let num_faces := 6 -- condition that dice has 6 faces, each numbered 1 to 6
  P (face 5) = 1 / num_faces := by
begin
  sorry -- Proof is not required
end

end probability_dice_face_5_l81_81950


namespace average_of_u_l81_81331

theorem average_of_u :
  (∃ u : ℕ, ∀ r1 r2 : ℕ, (r1 + r2 = 6) ∧ (r1 * r2 = u) → r1 > 0 ∧ r2 > 0) →
  (∃ distinct_u : Finset ℕ, distinct_u = {5, 8, 9} ∧ (distinct_u.sum / distinct_u.card) = 22 / 3) :=
sorry

end average_of_u_l81_81331


namespace set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l81_81473

variable {U : Type} [DecidableEq U]
variables (A B C K : Set U)

theorem set_theorem_1 : (A \ K) ∪ (B \ K) = (A ∪ B) \ K := sorry
theorem set_theorem_2 : A \ (B \ C) = (A \ B) ∪ (A ∩ C) := sorry
theorem set_theorem_3 : A \ (A \ B) = A ∩ B := sorry
theorem set_theorem_4 : (A \ B) \ C = (A \ C) \ (B \ C) := sorry
theorem set_theorem_5 : A \ (B ∩ C) = (A \ B) ∪ (A \ C) := sorry
theorem set_theorem_6 : A \ (B ∪ C) = (A \ B) ∩ (A \ C) := sorry
theorem set_theorem_7 : A \ B = (A ∪ B) \ B ∧ A \ B = A \ (A ∩ B) := sorry

end set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l81_81473


namespace final_book_prices_l81_81575

-- Definitions for initial prices
def initial_price_A : ℝ := 20
def initial_price_B : ℝ := 30
def initial_price_C : ℝ := 40
def initial_price_D : ℝ := 50
def initial_price_E : ℝ := 60
def initial_price_F : ℝ := 70

-- Price adjustments
def price_adjustment_A : ℝ := initial_price_A - (0.35 * initial_price_A)
def price_adjustment_B : ℝ := initial_price_B - (0.25 * initial_price_B)
def price_adjustment_C : ℝ := initial_price_C + (0.45 * initial_price_C)
def price_adjustment_D : ℝ := initial_price_D + (0.15 * initial_price_D)
def price_adjustment_E : ℝ := (price_adjustment_A + initial_price_E) / 2
def price_adjustment_B_final : ℝ := Real.sqrt (price_adjustment_B * initial_price_F)

-- Theorem to establish the final prices
theorem final_book_prices :
    price_adjustment_A = 13 ∧
    price_adjustment_B_final ≈ 39.686 ∧ -- Approximation (use ≈ to denote approximately equal to in Lean)
    price_adjustment_C = 58 ∧
    price_adjustment_D = 57.5 ∧
    price_adjustment_E = 36.5 ∧
    initial_price_F = 70 :=
by sorry

end final_book_prices_l81_81575


namespace number_of_tricycles_l81_81158

def num_bicycles : Nat := 24
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3
def total_wheels : Nat := 90

theorem number_of_tricycles : ∃ T : Nat, (wheels_per_bicycle * num_bicycles) + (wheels_per_tricycle * T) = total_wheels ∧ T = 14 := by
  sorry

end number_of_tricycles_l81_81158


namespace unit_vector_collinear_l81_81858

-- Define the vectors a and e
def vec_a : ℝ × ℝ × ℝ := (3, 0, -4)

-- Define the proof statement
theorem unit_vector_collinear (e : ℝ × ℝ × ℝ) : 
  (e = (3/5, 0, -4/5) ∨ e = (-3/5, 0, 4/5)) ∧
  ((∃ k : ℝ, e = (k * 3, k * 0, k * -4)) ∧ (∥e∥ = 1)) :=
sorry

end unit_vector_collinear_l81_81858


namespace mutually_exclusive_not_opposite_l81_81266

namespace CardDistribution

-- Definitions based on conditions
def person := {A, B, C, D}
def card := {red, black, blue, white}
def distribution (p : person) : card := sorry -- Placeholder for mapping each person to a card

-- Events
def person_A_gets_red : Prop := distribution 'A' = red
def person_B_gets_red : Prop := distribution 'B' = red

-- Theorem to be proved
theorem mutually_exclusive_not_opposite :
  person_A_gets_red ∧ person_B_gets_red = False ∧ ¬ (person_A_gets_red ↔ person_B_gets_red) :=
sorry

end CardDistribution

end mutually_exclusive_not_opposite_l81_81266


namespace perp_lines_l81_81927

/-- Let points A, B, and C lie on a circle, and let line b be tangent to this circle at point B. From point P, lying on line b, perpendiculars PA₁ and PC₁ are dropped onto lines AB and BC respectively. Points A₁ and C₁ lie on segments AB and BC respectively. -/
theorem perp_lines (A B C P A₁ C₁ : Point) (h_circle : on_circle A B C)
  (h_tangent : tangent_to_circle B b) (h_point_on_line : on_line P b)
  (h_perp_PA₁ : perp P A₁ (line_through A B)) 
  (h_perp_PC₁ : perp P C₁ (line_through B C))  :
  perp A₁ C₁ (line_through A C) := sorry

end perp_lines_l81_81927


namespace number_of_polynomials_satisfying_Q1_eq_10_l81_81745

theorem number_of_polynomials_satisfying_Q1_eq_10 :
  let Q : (ℕ × ℕ × ℕ × ℕ × ℕ) → ℕ := λ ⟨a, b, c, d, e⟩, a*1^4 + b*1^3 + c*1^2 + d*1 + e in
  (finset.univ.filter $ λ p : ℕ × ℕ × ℕ × ℕ × ℕ,
    let ⟨a, b, c, d, e⟩ := p in
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧ e ≤ 9 ∧ Q p = 10).card = 1001 :=
begin
  sorry
end

end number_of_polynomials_satisfying_Q1_eq_10_l81_81745


namespace chosen_number_probability_factorial_5_l81_81663

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81663


namespace factor_probability_l81_81593

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81593


namespace abc_triangle_geometry_l81_81877

theorem abc_triangle_geometry
  (ABC : Triangle)
  (A B C W X Y X1 X2 Y1 Y2 Z P U V : Point)
  (HA : WA = WX ∧ WA = WY)
  (HX1 : LineOnPointLine X_A_ X1 ∧ AngleEqual AXX1 90)
  (HX2 : LineOnPointLine X1_X2 ∧ AngleEqual AX1X2 90)
  (HY1 : LineOnPointLine Y_A_ Y1 ∧ AngleEqual AYY1 90)
  (HY2 : LineOnPointLine Y1_Y2 ∧ AngleEqual AY1Y2 90)
  (HZ : Intersect AW XY Z)
  (HP : FootPerpendicular A (LinePeriod X2Y2) P)
  (HU1 : Intersect ZP BC U)
  (HU2 : Intersect ZP PerpendicularBisector BC V)
  (HCU : CBetween B U)
  (H_AB : AB = x + 1)
  (H_AC : AC = 3)
  (H_AV : AV = x)
  (H_BC_CU : BC / CU = x)
  (H_x : x = (sqrt 2641 - 1) / 33):
  100 * 2641 + 10 * 1 + 33 = 264143 :=
by sorry

end abc_triangle_geometry_l81_81877


namespace smallest_square_area_l81_81563

theorem smallest_square_area (a b c d : ℕ) (hsquare : ∃ s : ℕ, s ≥ a + c ∧ s * s = a * b + c * d) :
    (a = 3) → (b = 5) → (c = 4) → (d = 6) → ∃ s : ℕ, s * s = 49 :=
by
  intros h1 h2 h3 h4
  cases hsquare with s hs
  use s
  -- Here we need to ensure s * s = 49
  sorry

end smallest_square_area_l81_81563


namespace value_of_a2a16_over_a9_l81_81862

noncomputable def a_n (n : ℕ) : ℝ := sorry

axiom geometric_sequence (n m k : ℕ) : a_n (n + m) * a_n k = a_n (n + k) * a_n m

axiom quadratic_roots (x y : ℝ) : (x^2 + 6 * x + 2) = 0 → (y^2 + 6 * y + 2) = 0 → a_n 2 = x → a_n 16 = y

theorem value_of_a2a16_over_a9 :
  let a2 := a_n 2,
      a16 := a_n 16,
      a9 := a_n 9 in
  (a2 + a16 = -6 ∧ a2 * a16 = 2) →
  (a2 * a16) / a9 = a9 :=
begin
  intros,
  -- This is the point where the proof starts but we leave it as "sorry"
  sorry,
end

end value_of_a2a16_over_a9_l81_81862


namespace area_of_PQRS_l81_81463

noncomputable def length_square_EFGH := 6
noncomputable def height_equilateral_triangle := 3 * Real.sqrt 3
noncomputable def diagonal_PQRS := length_square_EFGH + 2 * height_equilateral_triangle
noncomputable def area_PQRS := (1 / 2) * (diagonal_PQRS * diagonal_PQRS)

theorem area_of_PQRS :
  (area_PQRS = 72 + 36 * Real.sqrt 3) :=
sorry

end area_of_PQRS_l81_81463


namespace cubic_of_cubic_roots_correct_l81_81501

variable (a b c : ℝ) (α β γ : ℝ)

-- Vieta's formulas conditions
axiom vieta1 : α + β + γ = -a
axiom vieta2 : α * β + β * γ + γ * α = b
axiom vieta3 : α * β * γ = -c

-- Define the polynomial whose roots are α³, β³, and γ³
def cubic_of_cubic_roots (x : ℝ) : ℝ :=
  x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3

-- Prove that this polynomial has α³, β³, γ³ as roots
theorem cubic_of_cubic_roots_correct :
  ∀ x : ℝ, cubic_of_cubic_roots a b c x = 0 ↔ (x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
sorry

end cubic_of_cubic_roots_correct_l81_81501


namespace find_selling_price_functional_relationship_and_max_find_value_of_a_l81_81578

section StoreProduct

variable (x : ℕ) (y : ℕ) (a k b : ℝ)

-- Definitions for the given conditions
def cost_price : ℝ := 50
def selling_price := x 
def sales_quantity := y 
def future_cost_increase := a

-- Given points
def point1 : ℝ × ℕ := (55, 90) 
def point2 : ℝ × ℕ := (65, 70)

-- Linear relationship between selling price and sales quantity
def linearfunc := y = k * x + b

-- Proof of the first statement
theorem find_selling_price (k := -2) (b := 200) : 
    (profit = 800 → (x = 60 ∨ x = 90)) :=
by
  -- People prove the theorem here
  sorry

-- Proof for the functional relationship between W and x
theorem functional_relationship_and_max (x := 75) : 
    W = -2*x^2 + 300*x - 10000 ∧ W_max = 1250 :=
by
  -- People prove the theorem here
  sorry

-- Proof for the value of a when the cost price increases
theorem find_value_of_a (cost_increase := 4) : 
    (W'_max = 960 → a = 4) :=
by
  -- People prove the theorem here
  sorry

end StoreProduct

end find_selling_price_functional_relationship_and_max_find_value_of_a_l81_81578


namespace sally_fries_count_l81_81108

theorem sally_fries_count (sally_initial_fries mark_initial_fries : ℕ) 
  (mark_gave_fraction : ℤ) 
  (h_sally_initial : sally_initial_fries = 14) 
  (h_mark_initial : mark_initial_fries = 36) 
  (h_mark_give : mark_gave_fraction = 1 / 3) :
  sally_initial_fries + (mark_initial_fries * mark_gave_fraction).natAbs = 26 :=
by
  sorry

end sally_fries_count_l81_81108


namespace morse_code_max_5_symbols_l81_81083

theorem morse_code_max_5_symbols : 
  let num_symbols := 5
  let num_combinations (n : ℕ) := 2^n
  ∑ n in finset.range (num_symbols + 1).filter (λ x, x ≠ 0), num_combinations n = 62 :=
by
  sorry

end morse_code_max_5_symbols_l81_81083


namespace probability_not_same_level_is_four_fifths_l81_81715

-- Definitions of the conditions
def nobility_levels := 5
def total_outcomes := nobility_levels * nobility_levels
def same_level_outcomes := nobility_levels

-- Definition of the probability
def probability_not_same_level := 1 - (same_level_outcomes / total_outcomes : ℚ)

-- The theorem statement
theorem probability_not_same_level_is_four_fifths :
  probability_not_same_level = 4 / 5 := 
  by sorry

end probability_not_same_level_is_four_fifths_l81_81715


namespace minimum_value_of_z_l81_81362

theorem minimum_value_of_z 
  (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : 2 * x - y - 2 ≤ 0) 
  (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x + y ∧ z = -6 :=
sorry

end minimum_value_of_z_l81_81362


namespace marbles_total_l81_81992

theorem marbles_total (yellow blue red total : ℕ)
  (hy : yellow = 5)
  (h_ratio : blue / red = 3 / 4)
  (h_red : red = yellow + 3)
  (h_total : total = yellow + blue + red) : total = 19 :=
by
  sorry

end marbles_total_l81_81992


namespace BHM_perpendicular_l81_81044

variables {P Q R S A B C D M H : Type}
variables [AffineSpace P]
variables [AddCancelCommGroup P]
variables [Module ℝ P]

def is_midpoint (A : P) (X Y : P) := A = (X +ᵥ Y) / 2

-- Given statements as Lean conditions
axiom pqrs_quadrilateral : True
axiom A_midpoint : is_midpoint A P Q
axiom B_midpoint : is_midpoint B Q R
axiom C_midpoint : is_midpoint C R S
axiom D_midpoint : is_midpoint D S P
axiom M_midpoint : is_midpoint M C D
axiom H_condition : dist H C = dist B C ∧ collinear A M H

-- The goal statement to prove
theorem BHM_perpendicular : ∠ B H M = 90 := sorry

end BHM_perpendicular_l81_81044


namespace average_distinct_u_l81_81338

theorem average_distinct_u :
  ∀ (u : ℕ), (∃ a b : ℕ, a + b = 6 ∧ ab = u) →
  {u | ∃ a b : ℕ, a + b = 6 ∧ ab = u}.to_finset.val.sum / 3 = 22 / 3 :=
sorry

end average_distinct_u_l81_81338


namespace inequality_solution_set_l81_81982

theorem inequality_solution_set (a b : ℝ) (h₀ : (1 : ℝ) + b = 3) (h₁ : (1 : ℝ) * b = -6 / a) 
(h₂ : ∀ x : ℝ, ax^2 - 3ax - 6 < 0 -> x < 1 ∨ x > b) : a + b = -1 := 
by 
  sorry

end inequality_solution_set_l81_81982


namespace find_f1_l81_81069

noncomputable def f : ℤ → ℤ
| x := if x ≥ 3 then 2 * x + 4 else f (x + 1) - 1

theorem find_f1 : f 1 = 8 :=
by
  sorry

end find_f1_l81_81069


namespace pure_imaginary_complex_l81_81841

def is_pure_imaginary (z : Complex) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_complex (a : ℝ) :
  is_pure_imaginary ((a + 2 * Complex.i) * (1 + 3 * Complex.i)) ↔ a = 6 := by
  sorry

end pure_imaginary_complex_l81_81841


namespace pure_imaginary_complex_solution_l81_81837

theorem pure_imaginary_complex_solution (a : Real) :
  (a ^ 2 - 1 = 0) ∧ ((a - 1) ≠ 0) → a = -1 := by
  sorry

end pure_imaginary_complex_solution_l81_81837


namespace shaded_area_is_correct_l81_81724

-- Conditions definition
def shaded_numbers : ℕ := 2015
def boundary_properties (segment : ℕ) : Prop := 
  segment = 1 ∨ segment = 2

theorem shaded_area_is_correct : ∀ n : ℕ, n = shaded_numbers → boundary_properties n → 
  (∃ area : ℚ, area = 47.5) :=
by
  sorry

end shaded_area_is_correct_l81_81724


namespace find_tangents_point_l81_81094

noncomputable def parabola (x : ℝ) : ℝ := (x^2) / 2

def line (x₀ : ℝ) : ℝ := -15 / 2

theorem find_tangents_point (x₀ : ℝ) :
  ( ∃ k1 k2 : ℝ, k1 + k2 = 2 * x₀ ∧ k1 * k2 = -15 ∧
    (k2 - k1) / (1 + k1 * k2) = 1 / Real.sqrt 3 ) ↔
  x₀ =Real.sqrt 3 /3:= sorry

end find_tangents_point_l81_81094


namespace sum_of_fractions_l81_81438

theorem sum_of_fractions (z : ℂ) (hz1 : z = complex.cos (2 * real.pi / 5) + complex.sin (2 * real.pi / 5) * complex.I) 
  (hz2 : z^5 = 1) : 
  (z / (1 + z^2)) + (z^2 / (1 + z^4)) + (z^3 / (1 + z^6)) + (z^4 / (1 + z^8)) = 1 := 
by 
  sorry

end sum_of_fractions_l81_81438


namespace price_of_first_tea_l81_81869

theorem price_of_first_tea (x : ℝ) (price_second : ℝ) (price_mixture : ℝ) (ratio : ℝ) :
    price_second = 74 → price_mixture = 69 → ratio = 1 →
    69 - x = 5 → x = 64 := 
by
  intro h1 h2 h3 h4
  rw h4
  norm_num
  sorry

end price_of_first_tea_l81_81869


namespace part_a_part_b_part_c_l81_81071

-- Let d and l be functions N -> N representing partitions into distinct summands and odd summands respectively
def d : ℕ → ℕ 
def l : ℕ → ℕ 

-- Assume given condition d(0) = l(0) = 1
axiom d_zero : d 0 = 1
axiom l_zero : l 0 = 1

-- Prove part (a)
theorem part_a :
  \[ ∑ n in (finset.range ∞), d n * x^n = ( ∏ k in (finset.range ∞), (1 + x^k)) \]

-- Prove part (b)
theorem part_b :
  \[ ∑ n in (finset.range ∞), l n * x^n = ( ∏ k in (finset.range ∞), (1 - x^(2*k-1))⁻¹) \]

-- Prove part (c)
theorem part_c (n : ℕ) : 
  d n = l n :=
sorry 

end part_a_part_b_part_c_l81_81071


namespace four_digit_numbers_l81_81826

theorem four_digit_numbers (a b c d : ℕ) (h_digits : multiset a b c d = {0, 0, 3, 9})
    (h_positive : ∀ x ∈ {0, 0, 3, 9}, x ≥ 0) (h_first_digit_non_zero : a ≠ 0): 
    ∃ n, n = 6 :=
by
  sorry

end four_digit_numbers_l81_81826


namespace radical_product_is_64_l81_81527

theorem radical_product_is_64:
  real.sqrt (16:ℝ) * real.sqrt (real.sqrt 256) * real.n_root 64 3 = 64 :=
sorry

end radical_product_is_64_l81_81527


namespace solve_expression_l81_81436

noncomputable def x : ℂ := complex.exp (2 * real.pi * complex.I / 9)

theorem solve_expression : 
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) =
  22 - 9 * x^5 - 9 * x^2 + 3 * x^6 + 4 * x^3 + 3 * x :=
sorry

end solve_expression_l81_81436


namespace additional_sugar_is_correct_l81_81163

def sugar_needed : ℝ := 450
def sugar_in_house : ℝ := 287
def sugar_in_basement_kg : ℝ := 50
def kg_to_lbs : ℝ := 2.20462

def sugar_in_basement : ℝ := sugar_in_basement_kg * kg_to_lbs
def total_sugar : ℝ := sugar_in_house + sugar_in_basement
def additional_sugar_needed : ℝ := sugar_needed - total_sugar

theorem additional_sugar_is_correct : additional_sugar_needed = 52.769 := by
  sorry

end additional_sugar_is_correct_l81_81163


namespace not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l81_81049

theorem not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles : 
  ¬ ∃ (rectangles : ℕ × ℕ), rectangles.1 = 1 ∧ rectangles.2 = 7 ∧ rectangles.1 * 4 + rectangles.2 * 3 = 25 :=
by
  sorry

end not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l81_81049


namespace find_length_DF_l81_81868

-- Definitions based on the problem statement
def triangle (A B C : Type) := ∀ (points : set A), set.finite points
def midpoint (A : Type) (a b : A) := (a + b) / 2 -- Idealized definition for midpoint, assuming A is a type supporting these operations
def median_len_square {A : Type} [normed_group A] [normed_space ℝ A] (d n : A) := ∥d - midpoint ↝∥^2

-- Statement of the problem
theorem find_length_DF {A : Type} [normed_group A] [normed_space ℝ A] (D E F N : A)
  (DE EF : ℝ) (DN : ℝ) (hDE : ∥D - E∥ = DE) (hEF : ∥E - F∥ = EF) (hDN : ∥D - N∥ = DN)
  (N_mid : N = midpoint E F) : 
  ∥D - F∥ = √(EF^2 - DE^2) := by
  sorry

end find_length_DF_l81_81868


namespace geometric_common_ratio_of_arithmetic_seq_l81_81859

theorem geometric_common_ratio_of_arithmetic_seq 
  (a : ℕ → ℝ) (d q : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 2)
  (h_nonzero_diff : d ≠ 0)
  (h_geo_seq : a 1 = 2 ∧ a 3 = 2 * q ∧ a 11 = 2 * q^2) : 
  q = 4 := 
by
  sorry

end geometric_common_ratio_of_arithmetic_seq_l81_81859


namespace total_shirts_correct_l81_81161

def machine_A_production_rate := 6
def machine_A_yesterday_minutes := 12
def machine_A_today_minutes := 10

def machine_B_production_rate := 8
def machine_B_yesterday_minutes := 10
def machine_B_today_minutes := 15

def machine_C_production_rate := 5
def machine_C_yesterday_minutes := 20
def machine_C_today_minutes := 0

def total_shirts_produced : Nat :=
  (machine_A_production_rate * machine_A_yesterday_minutes +
  machine_A_production_rate * machine_A_today_minutes) +
  (machine_B_production_rate * machine_B_yesterday_minutes +
  machine_B_production_rate * machine_B_today_minutes) +
  (machine_C_production_rate * machine_C_yesterday_minutes +
  machine_C_production_rate * machine_C_today_minutes)

theorem total_shirts_correct : total_shirts_produced = 432 :=
by 
  sorry 

end total_shirts_correct_l81_81161


namespace probability_number_is_factor_of_120_l81_81628

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81628


namespace petya_mistaken_l81_81096

-- Definitions for train configurations
def train11 : Type := { c : list ℕ // c.length = 11 }
def train12 : Type := { c : list ℕ // c.length = 12 }

-- Problem statement: proving that the number of distinct 11-car trains is at least 
--  as large as the number of distinct 12-car trains.

theorem petya_mistaken (distinct_trains11 train11 ≥ distinct_trains12 train12) :
  distinct_trains11 train11 ≥ distinct_trains12 train12 := 
by
sorry

end petya_mistaken_l81_81096


namespace fastest_car_80_088_mph_l81_81737

theorem fastest_car_80_088_mph :
  let km_to_miles := 0.621371
  let hours_A := 4.5
  let hours_B := 6.75
  let hours_C := 10
  let miles_A := 360
  let miles_B := 870 * km_to_miles
  let miles_C := 1150 * km_to_miles
  (miles_A / hours_A < miles_B / hours_B) ∧ (miles_C / hours_C < miles_B / hours_B) :=
by
  let km_to_miles := 0.621371
  let hours_A := 4.5
  let hours_B := 6.75
  let hours_C := 10
  let miles_A := 360
  let miles_B := 870 * km_to_miles
  let miles_C := 1150 * km_to_miles
  have speed_A := miles_A / hours_A
  have speed_B := miles_B / hours_B
  have speed_C := miles_C / hours_C
  calc
    speed_A < speed_B := sorry
    speed_C < speed_B := sorry

end fastest_car_80_088_mph_l81_81737


namespace problem_1_problem_2_l81_81723

variables {A B C P D E F G H K M N : Point} (triangle : IsTriangle A B C)
          (P_in_triangle : P ∈ triangle)

-- Conditions
axiom exterior_angle_bisectors_PBC : IsExteriorAngleBisector P B C D E
axiom exterior_angle_bisectors_PAC : IsExteriorAngleBisector P A C F G
axiom exterior_angle_bisectors_PAB : IsExteriorAngleBisector P A B H K
axiom intersection_FK_DE : Line FK ∩ Line DE = M
axiom intersection_HG_DE : Line HG ∩ Line DE = N

-- Statements to prove:
def statement1 : Prop :=
  1/segment_length P M - 1/segment_length P N = 1/segment_length P D - 1/segment_length P E 

def statement2 (midpoint_PD_PE : Midpoint P D E) : Prop :=
  segment_length P M = segment_length P N

-- The proof problems
theorem problem_1 : statement1 :=
  sorry

theorem problem_2 (midpoint_PD_PE : Midpoint P D E) : statement2 midpoint_PD_PE :=
  sorry

end problem_1_problem_2_l81_81723


namespace average_distinct_u_l81_81339

theorem average_distinct_u :
  ∀ (u : ℕ), (∃ a b : ℕ, a + b = 6 ∧ ab = u) →
  {u | ∃ a b : ℕ, a + b = 6 ∧ ab = u}.to_finset.val.sum / 3 = 22 / 3 :=
sorry

end average_distinct_u_l81_81339


namespace ratio_of_slices_l81_81752

theorem ratio_of_slices
  (initial_slices : ℕ)
  (slices_eaten_for_lunch : ℕ)
  (remaining_slices_after_lunch : ℕ)
  (slices_left_for_tomorrow : ℕ)
  (slices_eaten_for_dinner : ℕ)
  (ratio : ℚ) :
  initial_slices = 12 → 
  slices_eaten_for_lunch = initial_slices / 2 →
  remaining_slices_after_lunch = initial_slices - slices_eaten_for_lunch →
  slices_left_for_tomorrow = 4 →
  slices_eaten_for_dinner = remaining_slices_after_lunch - slices_left_for_tomorrow →
  ratio = (slices_eaten_for_dinner : ℚ) / remaining_slices_after_lunch →
  ratio = 1 / 3 :=
by sorry

end ratio_of_slices_l81_81752


namespace simplify_cbrt_8000_eq_21_l81_81183

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l81_81183


namespace range_of_m_l81_81440

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f'' (x : ℝ) : ℝ := sorry

-- Conditions
axiom cond1 : ∀ x : ℝ, f(x) + f(-x) = x^2
axiom cond2 : ∀ x : ℝ, (0 < x) → (f''(x) - x < 0)

-- Statement
theorem range_of_m :
  ∀ (m : ℝ), 
  (2 ≤ m) → 
  (f(4 - m) - f(m) ≥ 8 - 4 * m) :=
sorry

end range_of_m_l81_81440


namespace area_of_Gamma_rectangle_l81_81041

theorem area_of_Gamma_rectangle
  (AE BF : ℝ)
  (h1 : AE = 30)
  (h2 : BF = 25) :
  let DE := AE,
      DF := BF,
      area := DE * DF
  in area = 750 := by
  {
    sorry
  }

end area_of_Gamma_rectangle_l81_81041


namespace smallest_area_of_square_containing_rectangles_l81_81567

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end smallest_area_of_square_containing_rectangles_l81_81567


namespace percentage_of_solution_x_in_mixture_l81_81955

-- Definitions for the conditions:
-- Solution x has 40% chemical a, 60% chemical b
def solution_x_chemical_a_percentage : ℝ := 0.40
def solution_y_chemical_a_percentage : ℝ := 0.50
def mixture_chemical_a_percentage : ℝ := 0.47

-- Proving the percentage of solution x in the mixture
theorem percentage_of_solution_x_in_mixture : 
  ∃ (x : ℝ), 0.40 * x + 0.50 * (100 - x) = 47 ∧ x = 30 :=
begin
  -- Actual proof will go here
  sorry
end

end percentage_of_solution_x_in_mixture_l81_81955


namespace ratio_books_Pete_Matt_l81_81459

-- Definitions for the number of books read by Pete and Matt last year.
variable {P M : ℕ}

-- Conditions
variable (h1 : 3 * P = 300) -- Pete read 300 books in total over the two years.
variable (h2 : 3 / 2 * M = 75) -- Matt read 75 books in his second year, which is 50% more than last year.

-- Proof statement
theorem ratio_books_Pete_Matt : (P : ℚ) / M = 2 :=
by 
  sorry

end ratio_books_Pete_Matt_l81_81459


namespace correct_subtraction_l81_81007

theorem correct_subtraction (x : ℕ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_subtraction_l81_81007


namespace problem_r_value_l81_81435

theorem problem_r_value (n : ℕ) (h : n = 3) : 
  let s := 2^n - 1 in
  let r := 3^s - s in
  r = 2180 := by
  sorry

end problem_r_value_l81_81435


namespace product_of_square_and_neighbor_is_divisible_by_12_l81_81944

theorem product_of_square_and_neighbor_is_divisible_by_12 (n : ℤ) : 12 ∣ (n^2 * (n - 1) * (n + 1)) :=
sorry

end product_of_square_and_neighbor_is_divisible_by_12_l81_81944


namespace probability_is_13_over_30_l81_81675

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81675


namespace range_of_expression_l81_81845

-- Define the given conditions and statement
variables {A B C P : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space P]

-- The values for lengths and angle
constant AB : A × B → ℝ
constant AC : A × C → ℝ
constant angle_BAC : ℝ

-- Definitions for vector calculations
constant BC : B × C → ℝ
constant lambda : ℝ
constant AP : A × P → ℝ
constant BP : B × P → ℝ

theorem range_of_expression (h_angle: angle_BAC = 120)
                            (h_AB: AB = 2)
                            (h_AC: AC = 1)
                            (h_lambda: 0 ≤ lambda ∧ lambda ≤ 1)
                            : (∃ lower upper, lower = 13 / 4 ∧ upper = 5 ∧
                              lower ≤ BP ^ 2 - AP • BC ∧ BP ^ 2 - AP • BC ≤ upper) :=
sorry

end range_of_expression_l81_81845


namespace max_distance_B_P_l81_81899

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l81_81899


namespace power_identity_l81_81484

theorem power_identity (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 := by
  sorry

end power_identity_l81_81484


namespace max_distance_on_ellipse_l81_81913

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l81_81913


namespace max_distance_on_ellipse_l81_81908

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l81_81908


namespace smallest_area_square_l81_81571

theorem smallest_area_square (a b u v : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : u = 4) (h₄ : v = 6) :
  ∃ s : ℕ, s^2 = 81 ∧ 
    (∀ xa ya xb yb xu yu xv yv : ℕ, 
      (xa + a ≤ s) ∧ (ya + b ≤ s) ∧ (xb + u ≤ s) ∧ (yb + v ≤ s) ∧ 
      ─xa < xb → xb < xa + a → ─ya < yb → yb < ya + b →
      ─xu < xv → xv < xu + u → ─yu < yv → yv < yu + v ∧
      (ya + b ≤ yv ∨ yu + v ≤ yb))
    := sorry

end smallest_area_square_l81_81571


namespace simplify_and_multiply_roots_l81_81534

theorem simplify_and_multiply_roots :
  (256 = 4^4) →
  (64 = 4^3) →
  (16 = 4^2) →
  ∜256 * ∛64 * sqrt 16 = 64 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end simplify_and_multiply_roots_l81_81534


namespace find_remainder_mod_1000_l81_81925

-- Define the polynomial q(x)
def q(x : ℤ) := x^(2010) + x^(2009) + x^(2008) + x^(2007) + x^(2006) + x^(2005) + x^(2004) + x^(2003) + x^(2002) + x^(2001) + 
                  x^(2000) + x^(1999) + x^(1998) + x^(1997) + x^(1996) + x^(1995) + x^(1994) + x^(1993) + x^(1992) + x^(1991) + 
                  x^(1990) + x^(1989) + x^(1988) + x^(1987) + x^(1986) + x^(1985)  + x^(1984) + x^(1983) + x^(1982) + x^(1981) +
                  -- This continues all the way down to...
                  x^4 + x^3 + x^2 + x + 1

-- Define the divisor polynomial
def divisor(x : ℤ) := x^4 + x^2 + 2 * x + 1

-- Polynomial remainder of q(x) when divided by the divisor
def remainder_poly := Polynomial.modByMonic q divisor

-- Evaluate the remainder polynomial at x = 2010 and its absolute value
noncomputable def s_value := Int.absolute $ remainder_poly.eval 2010

-- Prove the final result
theorem find_remainder_mod_1000 : s_value % 1000 = 890 :=
by
    sorry

end find_remainder_mod_1000_l81_81925


namespace smallest_positive_period_of_f_l81_81809

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + π / 3) + Real.sin x ^ 2

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ (∀ T' > 0, (∀ x, f(x + T') = f(x)) → T' = T) := sorry

end smallest_positive_period_of_f_l81_81809


namespace find_two_digit_numbers_l81_81984

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_two_digit_numbers :
  { A : ℕ |
    10 ≤ A ∧ A < 100 ∧
    let a := A / 10,
        b := A % 10 in
    a ∈ {1, 2, ..., 9} ∧ b ∈ {0, 1, ..., 9} ∧
    (a + b) ^ 2 = sum_of_digits (A^2)
  } = {10, 20, 11, 30, 21, 12, 31, 22, 13} :=
by
  sorry

end find_two_digit_numbers_l81_81984


namespace max_PB_distance_l81_81883

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l81_81883


namespace Inequality_Pi_Inequality_Delta_l81_81553

noncomputable def edges_squared_sum (a b c a1 b1 c1 : ℝ) : ℝ :=
  a^2 + b^2 + c^2 + a1^2 + b1^2 + c1^2

theorem Inequality_Pi (a b c a1 b1 c1 π : ℝ) :
  let δ := edges_squared_sum a b c a1 b1 c1
  in π ≤ (Real.sqrt (6 * δ)) / 2 := sorry

theorem Inequality_Delta (δ Δ : ℝ) :
  Δ ≤ (Real.sqrt 3 * δ) / 6 := sorry

end Inequality_Pi_Inequality_Delta_l81_81553


namespace find_value_of_x_l81_81479

theorem find_value_of_x (x y z : ℤ) (h1 : x > y) (h2 : y > z) (h3 : z = 3)
  (h4 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h5 : (x = y + 1) ∧ (y = z + 1)) :
  x = 5 := 
sorry

end find_value_of_x_l81_81479


namespace math_problem_l81_81522

def sum1 : ℕ := 3 + 5 + 7
def sum2 : ℕ := 2 + 4 + 6

theorem math_problem :
  ((sum1.toRat / sum2.toRat) * 2) - (sum2.toRat / sum1.toRat) = 17 / 10 := by
  sorry

end math_problem_l81_81522


namespace ratio_x_y_l81_81382

theorem ratio_x_y (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1 / 2) : 
  x / y = 3 / (6 * x - 1) := 
sorry

end ratio_x_y_l81_81382


namespace sum_T_n_l81_81290

def g (x : ℕ) : ℕ := if x % 2 = 1 then 1 else g (x / 2) * 2

def T_n (n m : ℕ) : ℕ := ∑ k in (finset.range (2 ^ (n - 1))).map (λ k, k + 1), g (2 * k * m)

def p (m : ℕ) : ℕ := ∃ (j : ℕ), 2^j ∣ m ∧ ∀ (k : ℕ), 2^k ∣ m → j ≤ k

theorem sum_T_n (n m : ℕ) (hn : 0 < n) (hm : 1 ≤ m) : 
  T_n n m = 2^(p m + n) - 2^(p m) := by
  sorry

end sum_T_n_l81_81290


namespace totalWaysToCutGrid_l81_81037

/-
Problem: In how many ways can the given grid be cut into 1 × 2 rectangles if the side length of one cell is 1?
Given:
1. The side length of one cell is 1.
2. The grid is a composite of three 2 × 3 grids.
-/

def numberOfWaysToCutCompositeGrid (sideLength : ℕ) (grids : ℕ) : ℕ :=
by
    /- Assume side length of one cell is 1 and grid is composed of three 2x3 grids -/
    have h1 : sideLength = 1 := rfl
    have h2 : grids = 3 := rfl
    /- Total number of ways to cut each 2x3 grid into 1x2 rectangles is 3. -/
    have ways_to_cut_each_2x3_grid : ℕ := 3
    /- Calculate the total number of ways -/
    let total_ways := ways_to_cut_each_2x3_grid ^ grids
    /- Simplify the exponentiation -/
    have total_ways_eq : total_ways = 27 := by norm_num
    exact total_ways_eq

theorem totalWaysToCutGrid : numberOfWaysToCutCompositeGrid 1 3 = 27 := 
by
  exact numberOfWaysToCutCompositeGrid 1 3

end totalWaysToCutGrid_l81_81037


namespace red_balls_in_bag_l81_81828

theorem red_balls_in_bag : ∃ x : ℕ, (3 : ℚ) / (4 + (x : ℕ)) = 1 / 2 ∧ x = 2 := sorry

end red_balls_in_bag_l81_81828


namespace find_m_of_quadratic_root_zero_l81_81781

theorem find_m_of_quadratic_root_zero (m : ℝ) (h : ∃ x, (m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ x = 0) : m = 2 :=
sorry

end find_m_of_quadratic_root_zero_l81_81781


namespace minimum_colors_needed_l81_81165

-- Define the statements specific to the problem
def spaced_exactly (n m : ℕ) (d : ℕ) : Prop := abs (n - m) = d

theorem minimum_colors_needed 
  (N : ℕ)
  (coloring : ℕ → ℕ) 
  (h2 : ∀ i j, spaced_exactly i j 2 → coloring i ≠ coloring j)
  (h3 : ∀ i j, spaced_exactly i j 3 → coloring i ≠ coloring j)
  (h5 : ∀ i j, spaced_exactly i j 5 → coloring i ≠ coloring j) :
  ∃ c, c = 3 :=
sorry

end minimum_colors_needed_l81_81165


namespace square_area_from_circles_l81_81771

theorem square_area_from_circles :
  (∀ (r : ℝ), r = 7 → ∀ (n : ℕ), n = 4 → (∃ (side_length : ℝ), side_length = 2 * (2 * r))) →
  ∀ (side_length : ℝ), side_length = 28 →
  (∃ (area : ℝ), area = side_length * side_length ∧ area = 784) :=
sorry

end square_area_from_circles_l81_81771


namespace intersection_A_B_l81_81359

variable (A : Set ℝ) (B : Set ℝ)

def A_def : A = {x | -2 ≤ x ∧ x ≤ 3} := by
  ext x
  simp
  constructor
  intro h
  exact h
  intro h
  exact h

def B_def : B = {y | ∃ x, y = x^2 + 2} := by
  ext y
  simp
  constructor
  intro h
  exact h
  intro h
  exact h

theorem intersection_A_B :
  {x | x ∈ A ∧ x ∈ B} = {x | 2 ≤ x ∧ x ≤ 3} := by
  rw [A_def, B_def]
  sorry

end intersection_A_B_l81_81359


namespace percentage_reduction_in_sugar_consumption_l81_81840

theorem percentage_reduction_in_sugar_consumption (X : ℝ) (P_i P_n : ℝ) (h_i : P_i = 10) (h_n : P_n = 13) :
  let Y := (P_i * X) / P_n in
  (X - Y) / X * 100 = 23.08 := by
  sorry

end percentage_reduction_in_sugar_consumption_l81_81840


namespace sequence_general_term_l81_81252

def sequence (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 3 → a n = a (n - 1) - a (n - 2)

def sum_first_terms (a : ℕ → ℤ) (n : ℕ) (s : ℤ) : Prop :=
(∑ i in finset.range n, a i.succ) = s

theorem sequence_general_term (a : ℕ → ℤ) :
  (sequence a) →
  (sum_first_terms a 1492 1985) →
  (sum_first_terms a 1985 1492) →
  ∀ n, n ≥ 3 → a n = 
    let ω := complex.exp (2 * complex.pi * complex.I / 3) in
    let ω2 := ω^2 in
    (-ω)^(n-1) / (ω - ω2) * ((ω^(n-1) - 1) * 493 - (ω^(n-1) - ω) * ω * 999) := 
by
  intro h_sequence h_sum1492 h_sum1985 n hn,
  sorry

end sequence_general_term_l81_81252


namespace proof_of_parallelogram_reassembling_l81_81823

noncomputable def cut_quadrilateral_into_parallelogram 
  (Q : Type) [is_quadrilateral Q] : 
  Prop :=
∃ (cut1 cut2 : Q → Q),
  (is_valid_cut Q cut1) ∧ (is_valid_cut Q cut2) ∧
  (can_be_reassembled cut1 cut2 → is_parallelogram (reassemble cut1 cut2))

-- Assume the necessary predicates and properties are defined:
-- is_quadrilateral : Q → Prop :: indicates whether the shape is a quadrilateral
-- is_valid_cut : Q → (Q → Q) → Prop :: validates if a cut divides the quadrilateral properly
-- can_be_reassembled : (Q → Q) → (Q → Q) → Prop :: checks if cuts can be reassembled well
-- is_parallelogram : Q → Prop :: validates if the reassembled shape is a parallelogram
-- reassemble : (Q → Q) → (Q → Q) → Q :: function to reassemble cuts

theorem proof_of_parallelogram_reassembling (Q : Type) [is_quadrilateral Q] :
  cut_quadrilateral_into_parallelogram Q :=
sorry

end proof_of_parallelogram_reassembling_l81_81823


namespace student_net_pay_l81_81200

theorem student_net_pay (base_salary bonus : ℕ) (tax_rate : ℝ) (h₁ : base_salary = 25000) (h₂ : bonus = 5000)
  (h₃ : tax_rate = 0.13) : (base_salary + bonus - (base_salary + bonus) * tax_rate) = 26100 :=
by 
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end student_net_pay_l81_81200


namespace max_PB_distance_l81_81880

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l81_81880


namespace square_not_end_with_four_identical_digits_l81_81110

theorem square_not_end_with_four_identical_digits (n : ℕ) (d : ℕ) :
  n = d * d → ¬ (d ≠ 0 ∧ (n % 10000 = d ^ 4)) :=
by
  sorry

end square_not_end_with_four_identical_digits_l81_81110


namespace map_scale_l81_81247

theorem map_scale (d_map : ℕ) (time : ℕ) (speed : ℕ) (d_actual : ℕ) (scale: ℕ) :
  d_map = 5 →
  time = 5 →
  speed = 60 →
  d_actual = speed * time →
  scale = d_actual / d_map →
  scale = 60 :=
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end map_scale_l81_81247


namespace average_of_distinct_u_l81_81334

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end average_of_distinct_u_l81_81334


namespace prism_volume_and_CF_l81_81495

-- Define necessary variables and constants
variables (h : ℝ) (r : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (V : ℝ) (CF : ℝ)

-- Assuming necessary conditions from the problem
def prism_condition : Prop :=
  h = 5 ∧ r = sqrt 33 / 4 ∧ CF = 2 ∨ CF = 3

-- The theorem we want to prove
theorem prism_volume_and_CF (h := 5) (r := sqrt 33 / 4) (CF : ℝ) :
  prism_condition h r CF →
  (V = (495 * sqrt 3) / 16) ∧ (CF = 2 ∨ CF = 3) :=
sorry -- Proof goes here

end prism_volume_and_CF_l81_81495


namespace fraction_of_juniors_studying_japanese_l81_81726

variable (J S : ℕ) 
variable (fraction_juniors fraction_seniors : ℚ)

-- Conditions
def condition1 (J S : ℕ) : Prop := S = 2 * J
def condition2 (fraction_seniors : ℚ) : Prop := fraction_seniors = 1 / 8
def condition3 (fraction_total : ℚ) : Prop := fraction_total = 1 / 3

-- Question: What fraction of juniors study Japanese?
theorem fraction_of_juniors_studying_japanese 
  (h1 : condition1 J S) 
  (h2 : condition2 fraction_seniors) 
  (h3 : condition3 1/3) :
  fraction_juniors = 3 / 4 :=
sorry

end fraction_of_juniors_studying_japanese_l81_81726


namespace find_whole_number_M_l81_81713

theorem find_whole_number_M (M : ℕ) (h : 8 < M / 4 ∧ M / 4 < 9) : M = 33 :=
sorry

end find_whole_number_M_l81_81713


namespace dave_winfield_home_runs_l81_81237

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end dave_winfield_home_runs_l81_81237


namespace probability_factor_of_5_factorial_l81_81686

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81686


namespace equilibrium_price_quantity_quantity_increase_due_to_subsidy_l81_81148

theorem equilibrium_price_quantity (p Q : ℝ) :
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (Q^D(2) = 8 ∧ Q^D(3) = 6) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ p Q, Q^D(p) = Q^S(p) ∧ Q = 10 :=
by
  intros
  have h₁ : Q^D(p) = -2 * p + 12 := sorry
  have h₂ : Q^S(p) = 2 + 8 * p := sorry
  use 1
  use 10
  simp [Q^D, Q^S]
  split
  sorry -- detailed steps to show Q^D(1) = 10 and Q^S(1) = 10

theorem quantity_increase_due_to_subsidy (p Q : ℝ) (s : ℝ) :
  s = 1 →
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ ΔQ, ΔQ = 1.6 :=
by
  intros
  have Q_s : Q^S(p + s) = 2 + 8 * (p + 1) := sorry
  have Q_d : Q^D(p) = -2 * p + 12 := sorry
  have new_p : p = 0.2 := sorry
  have new_Q : Q^S(0.2) = 11.6 := sorry
  use 1.6
  simp
  sorry -- detailed steps to show ΔQ = 1.6.

end equilibrium_price_quantity_quantity_increase_due_to_subsidy_l81_81148


namespace correct_statement_l81_81542

namespace MathProof

-- Definitions based on conditions
def statement_a (a : ℤ) : Prop := -a < 0
def statement_b (x : ℤ) : Prop := |x| = x → x > 0
def statement_c : Prop := true -- Placeholder, condition irrelevant as it involves specific number handling not in Lean scope.
def statement_d : Prop := (degree (X ^ 2 * Y) = 2 + 2)

-- Main statement
theorem correct_statement : ∃ (d : bool), 
    (statement_a = false → statement_b = false →
    statement_c = false → statement_d = true) := by 
  sorry

end MathProof

end correct_statement_l81_81542


namespace paul_total_vertical_distance_l81_81458

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end paul_total_vertical_distance_l81_81458


namespace polynomial_expansion_l81_81016

theorem polynomial_expansion :
  ∀ (a0 a1 a2 a3 a4 : ℝ),
  (∀ x : ℝ, (2 * x + real.sqrt 3) ^ 4 = a0 + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4) →
  (a0 + a2 + a4) ^ 2 - (a1 + a3) ^ 2 = 1 := by
  sorry

end polynomial_expansion_l81_81016


namespace angle_AOB_equals_70_l81_81169

-- Variables definitions
variables (P A B O : Type)
variables (angle : Type → Type) -- angle as a function over pairs of points
variables (tangent_to_circle : Type → Type → Prop) -- relation to represent tangency to circle

-- Conditions
variables (h1 : triangle P A B) -- Triangle PAB is formed by three tangents to circle O
variables (h2 : tangent_to_circle A O)
variables (h3 : tangent_to_circle B O)
variables (h4 : tangent_to_circle P O)
variables (angle_APB : angle A P B = 40)

-- Proof goal
theorem angle_AOB_equals_70 : angle A O B = 70 :=
sorry -- Proof goes here

end angle_AOB_equals_70_l81_81169


namespace vector_magnitude_ratio_l81_81038

variables {V : Type*} [InnerProductSpace ℝ V]

-- Definitions
def O : V := 0
variables {A B C : V}
def origin_condition : Prop :=
  C = (3/4 : ℝ) • A + (1/4 : ℝ) • B

theorem vector_magnitude_ratio (h : origin_condition) :
  ∥C - B∥ / ∥C - A∥ = 3 :=
sorry

end vector_magnitude_ratio_l81_81038


namespace probability_factor_of_5_factorial_l81_81692

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81692


namespace club_members_count_l81_81391

theorem club_members_count : 
  (∀ (P R J G : ℕ), 
    P = 50 → 
    R = 22 → 
    J = 35 → 
    ∃ N : ℕ, 5 = N ∧  
    G = 10 → 
    x = 4 → 
    ∀ x, P = (x + (8 - x) + (14 - x) + (J - x - 8 + x) + N + ((P - (R + N + J - N - x - G + x)))) → 
    (14 + 32 + x) = P → 
    x = 4) :=
begin 
  sorry 
end

end club_members_count_l81_81391


namespace vertical_distance_l81_81456

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end vertical_distance_l81_81456


namespace rankings_are_correct_l81_81452

-- Define teams:
inductive Team
| A | B | C | D

-- Define the type for ranking
structure Ranking :=
  (first : Team)
  (second : Team)
  (third : Team)
  (last : Team)

-- Define the predictions of Jia, Yi, and Bing
structure Predictions := 
  (Jia : Ranking)
  (Yi : Ranking)
  (Bing : Ranking)

-- Define the condition that each prediction is half right, half wrong
def isHalfRightHalfWrong (pred : Ranking) (actual : Ranking) : Prop :=
  (pred.first = actual.first ∨ pred.second = actual.second ∨ pred.third = actual.third ∨ pred.last = actual.last) ∧
  (pred.first ≠ actual.first ∨ pred.second ≠ actual.second ∨ pred.third ≠ actual.third ∨ pred.last ≠ actual.last)

-- Define the actual rankings
def actualRanking : Ranking := { first := Team.C, second := Team.A, third := Team.D, last := Team.B }

-- Define Jia's Predictions 
def JiaPrediction : Ranking := { first := Team.C, second := Team.C, third := Team.D, last := Team.D }

-- Define Yi's Predictions 
def YiPrediction : Ranking := { first := Team.B, second := Team.A, third := Team.C, last := Team.D }

-- Define Bing's Predictions 
def BingPrediction : Ranking := { first := Team.C, second := Team.B, third := Team.A, last := Team.D }

-- Create an instance of predictions
def pred : Predictions := { Jia := JiaPrediction, Yi := YiPrediction, Bing := BingPrediction }

-- The theorem to be proved
theorem rankings_are_correct :
  isHalfRightHalfWrong pred.Jia actualRanking ∧ 
  isHalfRightHalfWrong pred.Yi actualRanking ∧ 
  isHalfRightHalfWrong pred.Bing actualRanking →
  actualRanking.first = Team.C ∧ actualRanking.second = Team.A ∧ actualRanking.third = Team.D ∧ 
  actualRanking.last = Team.B :=
by
  sorry -- Proof is not required.

end rankings_are_correct_l81_81452


namespace value_of_a_l81_81383

theorem value_of_a {a : ℝ} (h : ∀ x y : ℝ, (a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) → x = y) : a = 0 ∨ a = 1 := 
  sorry

end value_of_a_l81_81383


namespace chosen_number_probability_factorial_5_l81_81653

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81653


namespace prob_factorial_5_l81_81665

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81665


namespace find_n_l81_81834

theorem find_n (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 :=
by
  sorry

end find_n_l81_81834


namespace calc_expression_l81_81015

noncomputable def x := (3 + Real.sqrt 5) / 2 -- chosen from one of the roots of the quadratic equation x^2 - 3x + 1

theorem calc_expression (h : x + 1 / x = 3) : 
  (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 7 + 3 * Real.sqrt 5 := 
by 
  sorry

end calc_expression_l81_81015


namespace alpha_not_periodic_beta_not_periodic_sequences_not_periodic_l81_81067

noncomputable def alpha (n : ℕ) : ℕ :=
  nat.floor (real.sqrt 10 ^ n)

noncomputable def beta (n : ℕ) : ℕ :=
  nat.floor (real.sqrt 2 ^ n)

theorem alpha_not_periodic : ¬∃ p N : ℕ, ∀ n ≥ N, alpha (n + p) = alpha n :=
sorry

theorem beta_not_periodic : ¬∃ p N : ℕ, ∀ n ≥ N, beta (n + p) = beta n :=
sorry

theorem sequences_not_periodic : ¬ (∃ p N : ℕ, ∀ n ≥ N, (alpha (n + p) = alpha n) ∧ (beta (n + p) = beta n)) :=
sorry

end alpha_not_periodic_beta_not_periodic_sequences_not_periodic_l81_81067


namespace range_of_a_l81_81775

noncomputable def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0

noncomputable def q (x a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0

def sufficient_but_not_necessary_condition (a : ℝ) : Prop :=
  ∀ x, p x → q x a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : sufficient_but_not_necessary_condition a) :
  9 ≤ a :=
sorry

end range_of_a_l81_81775


namespace sequence_properties_l81_81804

variables {a : ℕ → ℕ} {b : ℕ → ℕ} {c : ℕ → ℝ} {d : ℕ} {q : ℕ} {S : ℕ → ℕ}

noncomputable def arithmetic_seq (n : ℕ) : ℕ := 2 * n - 1
noncomputable def geometric_seq (n : ℕ) : ℕ := 2^(n-1)
noncomputable def c_seq (n : ℕ) : ℝ := (2 * n - 1) * (1 / 2)^(n-1)

theorem sequence_properties :
  (a 3 = 5) →
  (a 2 + a 5 = 6 * b 2) →
  (a 1 = b 1) →
  (d > 1) →
  (d = q) →
  (∀ n, a n = 2 * n - 1) ∧ 
  (∀ n, b n = 2^(n-1)) ∧
  (∀ n, (∑ k in finset.range n, c_seq k) = 6 - (2 * n + 3) * (1 / 2)^(n-1)) :=
sorry

end sequence_properties_l81_81804


namespace isosceles_triangle_hypotenuse_sq_l81_81254

theorem isosceles_triangle_hypotenuse_sq (u v w : ℂ) (s t k : ℂ) 
  (huvw : u + v + w = -2) 
  (hpoly : u * v + v * w + w * u = s)
  (hz : u * v * w = -t)
  (hsum : |u| ^ 2 + |v| ^ 2 + |w| ^ 2 = 350)
  (hisosceles : |u - v| = |u - w| ∧ |u - v| = |v - w|) :
  k ^ 2 = 525 := 
sorry

end isosceles_triangle_hypotenuse_sq_l81_81254


namespace xiaomin_house_position_l81_81184

-- Define the initial position of the school at the origin
def school_pos : ℝ × ℝ := (0, 0)

-- Define the movement east and south from the school's position
def xiaomin_house_pos (east_distance south_distance : ℝ) : ℝ × ℝ :=
  (school_pos.1 + east_distance, school_pos.2 - south_distance)

-- The given conditions
def east_distance := 200
def south_distance := 150

-- The theorem stating Xiaomin's house position
theorem xiaomin_house_position :
  xiaomin_house_pos east_distance south_distance = (200, -150) :=
by
  -- Skipping the proof steps
  sorry

end xiaomin_house_position_l81_81184


namespace small_cone_altitude_l81_81582

theorem small_cone_altitude (h_f: ℝ) (a_lb: ℝ) (a_ub: ℝ) : 
  h_f = 24 → a_lb = 225 * Real.pi → a_ub = 25 * Real.pi → ∃ h_s, h_s = 12 := 
by
  intros h1 h2 h3
  sorry

end small_cone_altitude_l81_81582


namespace parallel_not_coincident_lines_l81_81024

theorem parallel_not_coincident_lines (a : ℝ) :
  (∃ b1 b2 : ℝ, (∀ x y : ℝ, (ax + 2y + 6 = 0 → x + (a-1)*y + b1 = 0) 
  ∧ (¬ ∃ c : ℝ, ax + 2y + 6 = 0 ↔ x + (a-1)*y + c = 0))) → a = -1 :=
by
  sorry

end parallel_not_coincident_lines_l81_81024


namespace chicken_rabbit_problem_l81_81557

theorem chicken_rabbit_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chicken_rabbit_problem_l81_81557


namespace sum_cotangents_triangles_eq_l81_81946

theorem sum_cotangents_triangles_eq (a b c : ℝ) (S : ℝ) (ma mb mc Smed : ℝ)
  (h1 : cot a + cot b + cot c = (a^2 + b^2 + c^2) / (4 * S))
  (h2 : ma^2 + mb^2 + mc^2 = 3 * ((a^2 + b^2 + c^2) / 4))
  (h3 : Smed = (3 / 4) * S) :
  cot a + cot b + cot c = cot ma + cot mb + cot mc :=
sorry

end sum_cotangents_triangles_eq_l81_81946


namespace smallest_three_digit_solution_l81_81537

theorem smallest_three_digit_solution :
  ∃ n : ℤ, 45 * n ≡ 90 [MOD 315] ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 100 :=
by
  sorry

end smallest_three_digit_solution_l81_81537


namespace max_distance_on_ellipse_l81_81907

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l81_81907


namespace number_and_sum_f1_product_l81_81421

/-!
Let S be the set of all nonzero real numbers.
The function f : S → S satisfies the following properties:
  1. f(1 / x) = x * f(x) for all x ∈ S.
  2. f(1 / x) + f(1 / y) = 1 + f(1 / (x + y)) for all x, y ∈ S such that x + y ∈ S.
-/
noncomputable def S := { x : ℝ // x ≠ 0 }

def f (x : S) : ℝ := sorry

axiom f_property1 : ∀ (x : S), f ⟨1 / x.val, div_ne_zero one_ne_zero x.property⟩ = x.val * f x
axiom f_property2 : ∀ (x y : S), x.val + y.val ≠ 0 →
  f ⟨1 / x.val, div_ne_zero one_ne_zero x.property⟩ +
  f ⟨1 / y.val, div_ne_zero one_ne_zero y.property⟩ =
  1 + f ⟨1 / (x.val + y.val), div_ne_zero one_ne_zero (add_ne_zero x.property y.property)⟩

theorem number_and_sum_f1_product : 
  let n := {(f ⟨1, one_ne_zero⟩)}.to_finset.card,
      s := {(f ⟨1, one_ne_zero⟩)}.to_finset.sum id
  in n * s = 2 := 
sorry

end number_and_sum_f1_product_l81_81421


namespace certain_number_l81_81373

theorem certain_number (x y a : ℤ) (h1 : 4 * x + y = a) (h2 : 2 * x - y = 20) 
  (h3 : y ^ 2 = 4) : a = 46 :=
sorry

end certain_number_l81_81373


namespace last_score_is_80_l81_81450

def entered_scores : List ℕ := [50, 55, 60, 65, 70, 80]

theorem last_score_is_80 (s : List ℕ) (h₁ : s.length = 6) (h₂ : entered_scores = s) 
  (h₃ : ∀ n, 1 ≤ n ∧ n ≤ 6 → (∑ k in (s.take n), k) % n = 0) : 
  s.last = 80 :=
sorry

end last_score_is_80_l81_81450


namespace prob_factorial_5_l81_81664

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81664


namespace consecutive_non_prime_powers_l81_81474

theorem consecutive_non_prime_powers (k : ℕ) (hk: k > 0) :
  ∃ (n : ℕ), ∀ i : ℕ, (i < k) → ¬ ∃ (p : ℕ) (m : ℕ), Nat.prime p ∧ n + i = p ^ m :=
sorry

end consecutive_non_prime_powers_l81_81474


namespace max_PB_distance_l81_81885

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l81_81885


namespace find_x_in_arithmetic_sequence_l81_81784

theorem find_x_in_arithmetic_sequence (x : ℝ) (d : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = -2 →
  a_n 2 = 0 →
  (∀ n, a_n n = -2 + (n-1) * 2) →
  let a_1' := a_n 1 + x in
  let a_4' := a_n 4 + x in
  let a_5' := a_n 5 + x in
  2 * a_4' = a_5' * a_1' ↔
  x satisfies the added number criteria :=
by {
  assume h1 h2 h3,
  sorry
}

end find_x_in_arithmetic_sequence_l81_81784


namespace gcd_8_10_factorial_l81_81761

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10_factorial : Nat.gcd (factorial 8) (factorial 10) = 40320 := by
  let fac8 : ℕ := factorial 8
  let fac10 : ℕ := 10 * 9 * fac8
  have h1 : fac8 = 40320 := by sorry
  have h2 : fac10 = 10 * 9 * fac8 := by sorry
  rw [h1]
  sorry -- final proof step ensuring gcd (40320) (10 * 9 * 40320) is 40320

end gcd_8_10_factorial_l81_81761


namespace number_of_friends_l81_81722

/- Define the conditions -/
def sandwiches_per_friend : Nat := 3
def total_sandwiches : Nat := 12

/- Define the mathematical statement to be proven -/
theorem number_of_friends : (total_sandwiches / sandwiches_per_friend) = 4 :=
by
  sorry

end number_of_friends_l81_81722


namespace all_chameleons_green_l81_81562

-- Define initial counts
def initial_chameleons (yellow red green : ℕ) : Prop := 
  yellow = 7 ∧ red = 10 ∧ green = 17 ∧ yellow + red + green = 34

-- Define the transformation rules
def transform (yellow red green : ℕ) : Prop := 
  ∀ (y r g : ℕ), 
    (∃ a b c, 
      ((a = y ∧ b = r - 1 ∧ c = g - 1 ∧ yellow = y + 2) ∨ 
       (a = y - 1 ∧ b = r ∧ c = g - 1 ∧ yellow + 1 = y) ∨ 
       (a = y - 1 ∧ b = r - 1 ∧ c = g ∧ red = r + 2))) → 
    initial_chameleons y r g

-- Proving that all chameleons will be green at the end
theorem all_chameleons_green : 
  ∀ yellow red green, initial_chameleons yellow red green → transform yellow red green → green = 34 :=
by 
  intros yellow red green h_init h_trans,
  sorry

end all_chameleons_green_l81_81562


namespace hyperbola_focal_length_4sqrt3_l81_81492

def hyperbola_focal_length (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  in 2 * c

theorem hyperbola_focal_length_4sqrt3 (a b : ℝ) (ha : a^2 = 10) (hb : b^2 = 2) : 
  hyperbola_focal_length a b = 4 * Real.sqrt 3 := by
  sorry

end hyperbola_focal_length_4sqrt3_l81_81492


namespace max_divisible_by_11_pairs_l81_81234

open Nat

theorem max_divisible_by_11_pairs : 
  ∃ (pairs : List (Nat × Nat)), 
  (∀ p ∈ pairs, (1 ≤ p.1 ∧ p.1 ≤ 20) ∧ (1 ≤ p.2 ∧ p.2 ≤ 20) ∧ p.1 ≠ p.2) ∧ 
  List.length pairs = 10 ∧
  (∃ s : List ℕ, 
     (∀ (p : Nat × Nat) ∈ pairs, p.1 + p.2 = s.head) ∧ 
     (List.length (List.filter (λ x, x % 11 = 0) s) = 9)) := sorry

end max_divisible_by_11_pairs_l81_81234


namespace radical_product_is_64_l81_81530

theorem radical_product_is_64:
  real.sqrt (16:ℝ) * real.sqrt (real.sqrt 256) * real.n_root 64 3 = 64 :=
sorry

end radical_product_is_64_l81_81530


namespace probability_is_13_over_30_l81_81676

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81676


namespace difference_in_pennies_l81_81717

theorem difference_in_pennies (p : ℤ) : 
  let alice_nickels := 3 * p + 2
  let bob_nickels := 2 * p + 6
  let difference_nickels := alice_nickels - bob_nickels
  let difference_in_pennies := difference_nickels * 5
  difference_in_pennies = 5 * p - 20 :=
by
  sorry

end difference_in_pennies_l81_81717


namespace probability_factor_of_120_in_range_l81_81614

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81614


namespace non_rain_hours_correct_l81_81295

def total_hours : ℕ := 9
def rain_hours : ℕ := 4

theorem non_rain_hours_correct : (total_hours - rain_hours) = 5 := 
by
  sorry

end non_rain_hours_correct_l81_81295


namespace part1_part2_l81_81326

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x * (1 + x)
  else 0 -- placeholder, will be defined properly in theorems

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

variable (h_odd : odd_function f)
variable (h_pos : ∀ x : ℝ, x > 0 → f x = x * (1 + x))

theorem part1 : f (-2) = -6 :=
by
  have h1 : f (2) = 2 * (1 + 2) := h_pos 2 (by linarith)
  have h2 : f (-2) = -f 2 := h_odd 2
  rw h1 at h2
  exact h2

theorem part2 (x : ℝ) (h : x < 0) : f x = x * (1 - x) :=
by
  have h1 : f (-x) = -x * (1 + -x) := h_pos (-x) (by linarith)
  have h2 : f x = -f (-x) := h_odd x
  rw h1 at h2
  exact h2

end part1_part2_l81_81326


namespace probability_is_13_over_30_l81_81683

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81683


namespace range_of_a_l81_81509

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x - 2| > a ^ 2 + a + 1) ↔ (a ∈ Ioo (-1 : ℝ) 0) :=
by
  sorry

end range_of_a_l81_81509


namespace evaluate_expression_l81_81735

theorem evaluate_expression : 2 ^ 3 / 2 ^ (-2) + (-2) ^ 3 - (- (1 / 3)) ^ (-1) * 3 ^ 2 = 51 :=
by
  sorry

end evaluate_expression_l81_81735


namespace sum_of_min_x_y_l81_81076

theorem sum_of_min_x_y : ∃ (x y : ℕ), 
  (∃ a b c : ℕ, 180 = 2^a * 3^b * 5^c) ∧
  (∃ u v w : ℕ, 180 * x = 2^u * 3^v * 5^w ∧ u % 4 = 0 ∧ v % 4 = 0 ∧ w % 4 = 0) ∧
  (∃ p q r : ℕ, 180 * y = 2^p * 3^q * 5^r ∧ p % 6 = 0 ∧ q % 6 = 0 ∧ r % 6 = 0) ∧
  (x + y = 4054500) :=
sorry

end sum_of_min_x_y_l81_81076


namespace number_of_uninvited_students_l81_81852

def students := Fin 25

-- Define conditions
def isolated_group_8 (s : students → Prop) : Prop := ∃ (A : Finset students), A.card = 8 ∧ (∀ a ∈ A, ∀ b ∈ A, a ≠ b → (friendship a b : Prop))

def no_friends (s : students → Prop) : Prop := ∃ (B : Finset students), B.card = 2 ∧ (∀ b ∈ B, ∀ a : students, ¬(friendship b a))

def friends_with_isolated (s : students → Prop) : Prop := 
  ∃ (C : Finset students), C.card = 2 ∧ (∀ c ∈ C, ∃ (A : Finset students), A.card = 8 ∧ (∀ a ∈ A, (friendship c a)))

-- Define friendship relation
def friendship (a b : students) : Prop := sorry

-- Main theorem stating the number of uninvited students
theorem number_of_uninvited_students (s : students → Prop) :
  isolated_group_8 s ∧ no_friends s ∧ friends_with_isolated s →
  ∃ (D : Finset students), D.card = 12 ∧ (∀ d ∈ D, ¬(invited_to_tony_study_group d)) :=
sorry

end number_of_uninvited_students_l81_81852


namespace correct_option_is_C_l81_81541

theorem correct_option_is_C (x y : ℝ) :
  ¬(3 * x + 4 * y = 12 * x * y) ∧
  ¬(x^9 / x^3 = x^3) ∧
  ((x^2)^3 = x^6) ∧
  ¬((x - y)^2 = x^2 - y^2) :=
by
  sorry

end correct_option_is_C_l81_81541


namespace number_of_whole_numbers_between_200_and_500_containing_3_or_4_is_232_l81_81365

def contains_digit_3_or_4 (n : ℕ) : Prop :=
  (n / 100 = 3 ∨ n / 100 = 4) ∨
  ((n % 100) / 10 = 3 ∨ (n % 100) / 10 = 4) ∨
  (n % 10 = 3 ∨ n % 10 = 4)

theorem number_of_whole_numbers_between_200_and_500_containing_3_or_4_is_232 :
  ∃ count, count = (Finset.card (Finset.filter contains_digit_3_or_4 (Finset.range 500 \ Finset.range 200))) ∧ count = 232 :=
by
  sorry

end number_of_whole_numbers_between_200_and_500_containing_3_or_4_is_232_l81_81365


namespace integral_eq_log_div_l81_81741

open Real

noncomputable def integral_result (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : ℝ :=
  ∫ x in 0..1, (x ^ m - x ^ n) / log x

theorem integral_eq_log_div (m n : ℝ) (hm : 0 < m) (hn : 0 < n) :
  integral_result m n hm hn = log (abs ((m + 1) / (n + 1))) :=
sorry

end integral_eq_log_div_l81_81741


namespace probability_is_13_over_30_l81_81677

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81677


namespace max_distance_l81_81887

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l81_81887


namespace find_value_of_f_l81_81425

noncomputable theory
open Real

variable {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x

theorem find_value_of_f (h : f (-3) = 7) : f 3 = -7 :=
by sorry

end find_value_of_f_l81_81425


namespace refrigerator_sale_price_l81_81226

theorem refrigerator_sale_price :
  ∀ (original_price : ℝ) (first_discount second_discount : ℝ),
    original_price = 250.0 →
    first_discount = 0.20 →
    second_discount = 0.15 →
    let price_after_first_discount := original_price * (1 - first_discount) in
    let final_price := price_after_first_discount * (1 - second_discount) in
    final_price = original_price * 0.68 :=
by
  intros original_price first_discount second_discount h1 h2 h3,
  let price_after_first_discount := original_price * (1 - first_discount),
  let final_price := price_after_first_discount * (1 - second_discount),
  rw [h1, h2, h3],
  sorry

end refrigerator_sale_price_l81_81226


namespace probability_factor_of_120_l81_81644

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81644


namespace z_in_fourth_quadrant_l81_81342

noncomputable def z (x : ℂ) (h : (1 - Complex.i) / x = (4 : ℂ) + 2 * Complex.i) : ℂ :=
  x

theorem z_in_fourth_quadrant (x : ℂ) (h : (1 - Complex.i) / x = (4 : ℂ) + 2 * Complex.i) :
    x.re > 0 ∧ x.im < 0 :=
sorry

end z_in_fourth_quadrant_l81_81342


namespace distance_BC_is_correct_l81_81993

noncomputable def distance_between_B_and_C
  (n : ℝ) (A B C : Type)
  (AB : ℝ) (angle_A angle_B : ℝ) :
  ℝ :=
  if h1 : AB = 10 * n 
  if h2 : angle_A = 60 
  if h3 : angle_B = 75 
  then 
    5 * Real.sqrt 6 * n
  else 
    0

theorem distance_BC_is_correct
  (n : ℝ) (A B C : Type)
  (hAB : AB = 10 * n) 
  (hA : angle_A = 60) 
  (hB : angle_B = 75) : 
  distance_between_B_and_C n A B C AB angle_A angle_B = 5 * Real.sqrt 6 * n := by
  -- Proof omitted
  sorry

end distance_BC_is_correct_l81_81993


namespace small_bonsai_sold_eq_l81_81932

-- Define the conditions
def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- The proof problem: Prove that the number of small bonsai sold is 3
theorem small_bonsai_sold_eq : ∃ x : ℕ, 30 * x + 20 * 5 = 190 ∧ x = 3 :=
by
  sorry

end small_bonsai_sold_eq_l81_81932


namespace all_seven_boys_are_siblings_l81_81747

variable {Boy : Type} [Finite Boy] [DecidableEq Boy] (isBrother : Boy → Boy → Prop) 

-- Condition 1: There are exactly 7 boys
noncomputable def seven_boys := {b : Boy | true}
#check (seven_boys : Set Boy)
axiom card_seven : seven_boys.toFinset.card = 7

-- Condition 2: Each boy has at least 3 brothers among the other 6 boys
axiom has_at_least_three_brothers (b : Boy) : 3 ≤ (seven_boys.toFinset.erase b).filter (isBrother b).card

-- We need to prove that: all 7 boys are siblings
theorem all_seven_boys_are_siblings : ∀ ⦃a b : Boy⦄, isBrother a b := by
  sorry

end all_seven_boys_are_siblings_l81_81747


namespace find_q_l81_81137

-- Given conditions
noncomputable def digits_non_zero (p q r : Nat) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0

noncomputable def three_digit_number (p q r : Nat) : Nat :=
  100 * p + 10 * q + r

noncomputable def two_digit_number (q r : Nat) : Nat :=
  10 * q + r

noncomputable def one_digit_number (r : Nat) : Nat := r

noncomputable def numbers_sum_to (p q r sum : Nat) : Prop :=
  three_digit_number p q r + two_digit_number q r + one_digit_number r = sum

-- The theorem to prove
theorem find_q (p q r : Nat) (hpq : digits_non_zero p q r)
  (hsum : numbers_sum_to p q r 912) : q = 5 := sorry

end find_q_l81_81137


namespace probability_factorial_five_l81_81601

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81601


namespace probability_number_is_factor_of_120_l81_81624

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81624


namespace m_plus_n_is_one_l81_81816

-- Given polynomials with parameters m and n
def A (m : ℤ) : ℤ → ℤ → ℤ := λ x y, 2 * x^2 + 2 * x * y + m * y - 8
def B (n : ℤ) : ℤ → ℤ → ℤ := λ x y, - n * x^2 + x * y + y + 7

-- Define 'A - 2B' expression
def A_minus_2B (m n : ℤ) : ℤ → ℤ → ℤ := 
  λ x y, A m x y - 2 * B n x y 

-- Define the condition for A_minus_2B to have no x^2 or y term
def no_x2_or_y_terms (m n : ℤ) : Prop :=
  ∀ x y : ℤ, (A_minus_2B m n x y = -22)

theorem m_plus_n_is_one (m n : ℤ) (h : no_x2_or_y_terms m n) : m + n = 1 :=
  sorry

end m_plus_n_is_one_l81_81816


namespace bounded_area_l81_81140

open Real

theorem bounded_area (a : ℝ) (h : a > 0) :
  let f := (x y : ℝ) → (x + (a + 1) * y) ^ 2 = 9 * a ^ 2
  let g := (x y : ℝ) → (a * x - y) ^ 2 = 4 * a ^ 2
  let area := 40 * a ^ 2 / sqrt ((1 + a ^ 2 + 2 * a) * (a ^ 2 + 1))
  ∃ x y, f x y ∧ g x y → area = 40 * a ^ 2 / sqrt ((1 + a ^ 2 + 2 * a) * (a ^ 2 + 1)) :=
by
  sorry

end bounded_area_l81_81140


namespace find_M_coordinate_l81_81866

-- Definitions of the given points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨1, -3, 1⟩
def M (y : ℝ) : Point3D := ⟨0, y, 0⟩

-- Definition for the squared distance between two points
def dist_sq (p1 p2 : Point3D) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2

-- Main theorem statement
theorem find_M_coordinate (y : ℝ) : 
  dist_sq (M y) A = dist_sq (M y) B → y = -1 :=
by
  simp [dist_sq, A, B, M]
  sorry

end find_M_coordinate_l81_81866


namespace cubic_roots_l81_81746

theorem cubic_roots (a x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 1) (h₃ : x₃ = a)
  (cond : (2 / x₁) + (2 / x₂) = (3 / x₃)) :
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = a ∧ (a = 2 ∨ a = 3 / 4)) :=
by
  sorry

end cubic_roots_l81_81746


namespace area_largest_sum_others_l81_81403

variables {A B C G I : Type*}
variables [triangle ABC : Type*] [centroid G ABC] [incenter I ABC]
variables (S_AGI S_BGI S_CGI : ℝ)

def area_largest_equals_sum_others (S_AGI S_BGI S_CGI : ℝ) : Prop :=
  ∃ (S_max : ℝ), 
    S_max = max S_AGI (max S_BGI S_CGI) ∧ 
    S_max = S_AGI + S_BGI + S_CGI - S_max

theorem area_largest_sum_others :
  ∀ {A B C G I : Type*} [β : triangle ABC] [γ : centroid G ABC] [ι : incenter I ABC],
  area_largest_equals_sum_others S_AGI S_BGI S_CGI := 
by sorry

end area_largest_sum_others_l81_81403


namespace range_translation_l81_81810

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := f(x - π / 6) + sqrt 3 / 2

theorem range_translation (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  ∃ y, y ∈ set.range (λ x, g x) ∧ 0 ≤ y ∧ y ≤ 1 + sqrt 3 / 2 :=
begin
  sorry
end

end range_translation_l81_81810


namespace relationship_only_sometimes_l81_81719

-- Representing the input data
variables (B s t b : ℝ)
variables (θ φ : ℝ)
-- Conditions for the triangles being isosceles with distinct sides
hypothesis (base_distinct : B ≠ b)
hypothesis (side_distinct : s ≠ t)
-- Definitions for perimeters, areas, and circumradius
def P := B + 2 * s
def p := b + 2 * t
def K := (1/2) * B * sqrt (s^2 - (B/2)^2)
def k := (1/2) * b * sqrt (t^2 - (b/2)^2)
def R := s / (2 * sin θ)
def r := t / (2 * sin φ)

-- The statement to be proved that the given relationship holds only sometimes
theorem relationship_only_sometimes :
  (R / r) = (K / k) :=
sorry

end relationship_only_sometimes_l81_81719


namespace FG_parallel_DE_l81_81408

noncomputable def triangle := Type*

variable {ABC : triangle}
variables {A B C D E F G : ABC}
variables (AB AC BC : ℝ)
variables (BF CG : ℝ)

-- Conditions:
-- 1. D and E are intersections of angle bisectors from C and B with sides AB and AC, respectively.
-- 2. F and G such that BF = CG = BC.

axiom angle_bisector_intersection_D : ∃ (D : ABC), ∀ (A : ABC), ∀ (B : ABC), ∀ (C : ABC), ∃ (angle_bisector_C : ABC), (D ∈ line_segment A B)
axiom angle_bisector_intersection_E : ∃ (E : ABC), ∀ (A : ABC), ∀ (B : ABC), ∀ (C : ABC), ∃ (angle_bisector_B : ABC), (E ∈ line_segment A C)
axiom extension_BF : ∃ (F : ABC), (BF = BC)
axiom extension_CG : ∃ (G : ABC), (CG = BC)

theorem FG_parallel_DE :
  ∃ (D : ABC) (E : ABC) (F : ABC) (G : ABC), (F ∉ line_segment B C) ∧ (G ∉ line_segment C B) → 
  (let BF := BC) → (let CG := BC) → (FG_parallel_DE : Prop) := sorry

end FG_parallel_DE_l81_81408


namespace compute_b1_b2_b3_squared_l81_81920

theorem compute_b1_b2_b3_squared :
  ∀ (b1 b2 b3 : ℝ),
  (∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + b1 * x + 1) * (x^2 + b2 * x + 1) * (x^2 + b3 * x + 1)) →
  b1^2 + b2^2 + b3^2 = 1 :=
by
sory

end compute_b1_b2_b3_squared_l81_81920


namespace factor_probability_l81_81587

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81587


namespace probability_factor_of_120_l81_81646

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81646


namespace closest_vertex_l81_81384

theorem closest_vertex (a : ℝ) : 
  let parabola_vertex : ℝ × ℝ := (0, 0) in
  let point_A : ℝ × ℝ := (0, a) in
  let distance (P : ℝ × ℝ) : ℝ := (P.1)^2 + (P.2 - a)^2 in
  let parabola (x : ℝ) : ℝ := x^2 / 2 in
  ∀ (P : ℝ × ℝ), parabola_vertex = P ∧ P.2 = parabola P.1 → distance P ≤ distance point_A → a ≤ 1 :=
by 
   -- The main theorem body will contain the actual proof
   sorry

end closest_vertex_l81_81384


namespace smallest_area_of_triangle_ABC_l81_81417

noncomputable def vector_cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2.1 * v.2.2 - u.2.2 * v.2.1, u.2.2 * v.1 - u.1 * v.2.2, u.1 * v.2.1 - u.2.1 * v.1)

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

noncomputable def triangle_area (a b c : ℝ × ℝ × ℝ) : ℝ :=
0.5 * vector_magnitude (vector_cross_product (b.1 - a.1, b.2.1 - a.2.1, b.2.2 - a.2.2)
                                            (c.1 - a.1, c.2.1 - a.2.1, c.2.2 - a.2.2))

theorem smallest_area_of_triangle_ABC :
  ∀ t : ℝ, triangle_area (-1, (1, 2)) (1, (2, 3)) (t, (1, 1)) = sqrt(3) / 2 :=
sorry

end smallest_area_of_triangle_ABC_l81_81417


namespace max_distance_B_P_l81_81901

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l81_81901


namespace gcd_of_polynomial_l81_81797

def multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem gcd_of_polynomial (b : ℕ) (h : multiple_of b 456) :
  Nat.gcd (4 * b^3 + b^2 + 6 * b + 152) b = 152 := sorry

end gcd_of_polynomial_l81_81797


namespace an_bn_odd_l81_81433

noncomputable def a (n : ℕ) : ℝ :=
  (2 + Real.sqrt 7) ^ (2 * n + 1)

noncomputable def b (n : ℕ) : ℝ :=
  a n - Real.floor (a n)

theorem an_bn_odd (n : ℕ) (h : 0 < n) :
  (a n) * (b n) % 2 = 1 :=
by
  sorry

end an_bn_odd_l81_81433


namespace complex_square_l81_81439

def z : ℂ := 2 - complex.I

theorem complex_square :
  z^2 = 3 - 4 * complex.I :=
by sorry

end complex_square_l81_81439


namespace length_of_AC_l81_81299

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end length_of_AC_l81_81299


namespace hyperbola_distance_l81_81354

-- Defining the hyperbola equation and the condition
def is_on_hyperbola (p : ℝ × ℝ) : Prop := (p.1 ^ 2 / 6) - (p.2 ^ 2 / 3) = 1

-- Defining the foci positions (these would typically be computed from the hyperbola equation)
def F1 : ℝ × ℝ := (a, 0) -- Placeholder for actual coordinates
def F2 : ℝ × ℝ := (-a, 0) -- Placeholder for actual coordinates

-- Defining point M and the orthogonal condition
def is_orthogonal (M F1 F2 : ℝ × ℝ) : Prop := 
  let MF1 := (M.1 - F1.1, M.2 - F1.2) in
  let F1F2 := (F2.1 - F1.1, F2.2 - F1.2) in
  MF1.1 * F1F2.1 + MF1.2 * F1F2.2 = 0

-- The distance calculation (to be used in the theorem)
def distance_to_line (F1 : ℝ × ℝ) (M : ℝ × ℝ) (F2 : ℝ × ℝ) : ℝ :=
  abs ((F2.2 - M.2) * F1.1 - (F2.1 - M.1) * F1.2) / sqrt ((F2.1 - M.1)^2 + (F2.2 - M.2)^2)

-- The main theorem stating the distance is 6/5
theorem hyperbola_distance (M : ℝ × ℝ) (hM : is_on_hyperbola M) (hM_orthogonal : is_orthogonal M F1 F2) :
  distance_to_line F1 M F2 = 6 / 5 :=
sorry

end hyperbola_distance_l81_81354


namespace y_coordinate_of_second_point_l81_81867

variable {m n k : ℝ}

theorem y_coordinate_of_second_point (h1 : m = 2 * n + 5) (h2 : k = 0.5) : (n + k) = n + 0.5 := 
by
  sorry

end y_coordinate_of_second_point_l81_81867


namespace bob_investments_difference_l81_81728

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem bob_investments_difference :
  let A_A := future_value 2000 0.12 12 2 in
  let A_B := future_value 1000 0.30 4 2 in
  A_A - A_B = 724.87 :=
by
  -- Proof skipped
  sorry

end bob_investments_difference_l81_81728


namespace prop_for_real_l81_81817

theorem prop_for_real (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end prop_for_real_l81_81817


namespace area_triangle_ABD_is_24_l81_81123

noncomputable def area_triangle_ABC := 60
noncomputable def length_BD := 8
noncomputable def length_DC := 12

theorem area_triangle_ABD_is_24 :
  let BC := length_BD + length_DC,
      h := 2 * area_triangle_ABC / BC,
      area_ABD := 1 / 2 * length_BD * h
  in area_ABD = 24 := by
  intros BC h area_ABD
  have : BC = length_BD + length_DC := rfl
  have : h = 2 * area_triangle_ABC / BC := rfl
  have : area_ABD = 1 / 2 * length_BD * h := rfl
  sorry

end area_triangle_ABD_is_24_l81_81123


namespace first_player_wins_l81_81155

noncomputable def game_win_guarantee : Prop :=
  ∃ (first_can_guarantee_win : Bool),
    first_can_guarantee_win = true

theorem first_player_wins :
  ∀ (nuts : ℕ) (players : (ℕ × ℕ)) (move : ℕ → ℕ) (end_condition : ℕ → Prop),
    nuts = 10 →
    players = (1, 2) →
    (∀ n, 0 < n ∧ n ≤ nuts → move n = n - 1) →
    (end_condition 3 = true) →
    (∀ x y z, x + y + z = 3 ↔ end_condition (x + y + z)) → 
    game_win_guarantee :=
by
  intros nuts players move end_condition H1 H2 H3 H4 H5
  sorry

end first_player_wins_l81_81155


namespace factor_probability_l81_81588

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81588


namespace find_two_digit_numbers_l81_81983

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_two_digit_numbers :
  { A : ℕ |
    10 ≤ A ∧ A < 100 ∧
    let a := A / 10,
        b := A % 10 in
    a ∈ {1, 2, ..., 9} ∧ b ∈ {0, 1, ..., 9} ∧
    (a + b) ^ 2 = sum_of_digits (A^2)
  } = {10, 20, 11, 30, 21, 12, 31, 22, 13} :=
by
  sorry

end find_two_digit_numbers_l81_81983


namespace find_S_l81_81206

theorem find_S (R S T : ℝ) (c : ℝ)
  (h1 : R = c * (S / T))
  (h2 : R = 2) (h3 : S = 1/2) (h4 : T = 4/3) (h_c : c = 16/3)
  (h_R : R = Real.sqrt 75) (h_T : T = Real.sqrt 32) :
  S = 45/4 := by
  sorry

end find_S_l81_81206


namespace sum_is_28_l81_81401

-- Define digits and uniqueness constraints
variables (Y E A M B : ℕ)
variables [fact (Y ≠ E)] [fact (Y ≠ A)] [fact (Y ≠ M)] [fact (Y ≠ B)]
variables [fact (E ≠ A)] [fact (E ≠ M)] [fact (E ≠ B)]
variables [fact (A ≠ M)] [fact (A ≠ B)]
variables [fact (M ≠ B)]

-- Define digits constraints as base ten digits
variables [fact (Y < 10)] [fact (E < 10)] [fact (A < 10)] [fact (M < 10)] [fact (B < 10)]

-- Define YE, AM, and BBB
def YE := 10 * Y + E
def AM := 10 * A + M
def BBB := 100 * B + 10 * B + B

-- Provide equation constraint
variables [fact (YE * AM = BBB)]

-- Prove the sum constraint
theorem sum_is_28 : Y + E + A + M + B = 28 := by
  sorry

end sum_is_28_l81_81401


namespace sum_of_non_palindromes_between_100_and_200_take_four_steps_to_become_palindromes_l81_81768

/-- 
For positive integers between 100 and 200, sum the non-palindrome integers that take 
exactly four steps to become palindromes by repeatedly reversing and adding the original 
number to its reverse.
-/
theorem sum_of_non_palindromes_between_100_and_200_take_four_steps_to_become_palindromes :
  ∑ n in (Finset.filter (λ n : ℕ, ¬is_palindrome n ∧ takes_four_steps_to_become_palindrome n) 
    (Finset.range 200).filter (λ n, 100 ≤ n ∧ n < 200)), n = 190 :=
by sorry

/-- Checks if a given number is a palindrome. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  in s = s.reverse

/-- 
Takes a number and determines if it becomes a palindrome in exactly four steps 
by repeatedly reversing the number and adding the original number to its reverse.
-/
def takes_four_steps_to_become_palindrome (n : ℕ) : Prop :=
  let step := λ x, x + (x.to_string.reverse.to_nat (λ _, 0))
  let s1 := step n
  let s2 := step s1
  let s3 := step s2
  let s4 := step s3
  in is_palindrome s4 ∧ ¬(is_palindrome s1 ∨ is_palindrome s2 ∨ is_palindrome s3)

end sum_of_non_palindromes_between_100_and_200_take_four_steps_to_become_palindromes_l81_81768


namespace vector_dot_product_l81_81405

variables {A B C O : Euclidean_Space ℝ (Fin 2)}

def CircumscribedCircle (r: ℝ) := |O - A| = r ∧ |O - B| = r ∧ |O - C| = r

theorem vector_dot_product :
  CircumscribedCircle 1 →
  2 • (O - A) + (B - A) + (C - A) = 0 →
  |O - A| = |O - B| →
  (C - A) • (C - B) = 3 := 
by
  intros hCircle hVec hDist
  sorry

end vector_dot_product_l81_81405


namespace fraction_simplification_l81_81954

theorem fraction_simplification :
  (20 / 21) * (35 / 54) * (63 / 50) = (7 / 9) :=
by
  sorry

end fraction_simplification_l81_81954


namespace ratio_of_shaded_to_non_shaded_area_l81_81168

noncomputable def equilateral_triangle :=
{a b c : ℝ // a = b ∧ b = c}

noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
(√3 / 4) * s^2

theorem ratio_of_shaded_to_non_shaded_area :
  ∀ (abc : equilateral_triangle),
  let a := 12 in
  let d := midpoint 12 0 in let e := midpoint 12 12 in let f := midpoint 0 12 in
  let g := midpoint d f in let h := midpoint f e in let i := midpoint e d in
  let area_abc := area_equilateral_triangle a in
  let area_def := area_equilateral_triangle (midpoint 12 0) in
  let area_ghi := area_equilateral_triangle (midpoint d f) in
  let shaded_area := area_def - area_ghi in
  let non_shaded_area := area_abc - shaded_area in
  shaded_area / non_shaded_area = (3 : ℝ) / 13 := sorry

end ratio_of_shaded_to_non_shaded_area_l81_81168


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81633

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81633


namespace second_player_wins_or_draws_l81_81172

def player_can_guarantee_win_or_draw(halves : (list ℕ × list ℕ)) : Prop :=
  let player1 := halves.1
  let player2 := halves.2
  player1.sum ≤ 10000 ∧ player2.sum ≤ 10000 ∧ (player2.sum < player1.sum ∨
  (player2.sum < 10000 ∧ player2.sum + halved_sum player1 player2 = 10000))

-- Given conditions
def game_conditions : Prop :=
  ∀ (halves : list ℕ × list ℕ), 
      (halves.1.sum + halves.2.sum = 10000) →
      player_can_guarantee_win_or_draw(halves)

theorem second_player_wins_or_draws : game_conditions :=
sorry

end second_player_wins_or_draws_l81_81172


namespace geese_left_in_the_field_l81_81400

theorem geese_left_in_the_field 
  (initial_geese : ℕ) 
  (geese_flew_away : ℕ) 
  (geese_joined : ℕ)
  (h1 : initial_geese = 372)
  (h2 : geese_flew_away = 178)
  (h3 : geese_joined = 57) :
  initial_geese - geese_flew_away + geese_joined = 251 := by
  sorry

end geese_left_in_the_field_l81_81400


namespace root_of_quadratic_l81_81292

theorem root_of_quadratic (a : ℝ) (h : IsRoot (λ x : ℝ => x^2 - a * x + 6) 2) : a = 5 :=
by sorry

end root_of_quadratic_l81_81292


namespace solve_for_y_l81_81115

theorem solve_for_y (y : ℝ) : 16 ^ (3 * y - 4) = (1 / 4) ^ (2 * y + 6) → y = 1 / 4 :=
by
 sorry

end solve_for_y_l81_81115


namespace partial_2_z_partial_x2_partial_2_z_partial_y_partial_x_l81_81278

-- Definitions and conditions from problem 3.32
def partial_z_partial_x (z : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ := z x y / (z x y ^ 2 - 1)
def partial_z_partial_y (z : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ := - z x y / (y * (z x y ^ 2 - 1))

-- Theorems proving the second partial derivatives
theorem partial_2_z_partial_x2 (z : ℝ → ℝ → ℝ) (x y : ℝ)
  (hx : partial_z_partial_x z x y = z x y / (z x y ^ 2 - 1)) :
  (∂^2 z / ∂ x ^ 2 x y) = z x y * (z x y ^ 4 - z x y ^ 2 + 2) / (x ^ 2 * (z x y ^ 2 - 1) ^ 3) :=
sorry

theorem partial_2_z_partial_y_partial_x (z : ℝ → ℝ → ℝ) (x y : ℝ)
  (hy : partial_z_partial_y z x y = - z x y / (y * (z x y ^ 2 - 1))) :
  (∂^2 z / ∂ y ∂ x x y) = - (z x y ^ 2 + 1) * z x y / (y * x * (z x y ^ 2 - 1) ^ 3) :=
sorry

end partial_2_z_partial_x2_partial_2_z_partial_y_partial_x_l81_81278


namespace exists_xy_in_unit_interval_l81_81368

variable (f g : ℝ → ℝ)

theorem exists_xy_in_unit_interval :
  ∃ x y ∈ Icc 0 1, |x * y - f x - g y| ≥ 1 / 4 :=
sorry

end exists_xy_in_unit_interval_l81_81368


namespace vertical_asymptotes_of_function_l81_81770

theorem vertical_asymptotes_of_function (k : ℝ) : 
  (f : ℝ → ℝ := λ x, (x^2 - x + k) / (x^2 + x - 18)) →
  (k = -6 ∨ k = -42) →
  ∃ (asymptote : ℝ), ∀ x, (f(x) -> asymptote ↗ ∞ ∨ f(x) -> asymptote ↘ -∞) :=
by
  sorry

end vertical_asymptotes_of_function_l81_81770


namespace calc_sqrt_mult_l81_81526

theorem calc_sqrt_mult : 
  ∀ (a b c : ℕ), a = 256 → b = 64 → c = 16 → 
  (nat.sqrt (nat.sqrt a) * nat.cbrt b * nat.sqrt c = 64) :=
by 
  intros a b c h1 h2 h3
  rw [h1, nat.sqrt_eq, nat.sqrt_eq, h2, nat.cbrt_eq, h3, nat.sqrt_eq]
  sorry

end calc_sqrt_mult_l81_81526


namespace largest_non_congruent_non_similar_set_l81_81230

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a 

def is_valid_side (n : ℕ) : Prop :=
  n < 8

def is_congruent {a1 b1 c1 a2 b2 c2 : ℕ} : Prop :=
  a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∨ a1 = b2 ∧ b1 = c2 ∧ c1 = a2 ∨ a1 = c2 ∧ b1 = a2 ∧ c1 = b2

def is_similar {a1 b1 c1 a2 b2 c2 : ℕ} : Prop :=
  ∃ k : ℕ, k ≠ 0 ∧ (a1 = k * a2 ∧ b1 = k * b2 ∧ c1 = k * c2 ∨ a1 = k * b2 ∧ b1 = k * c2 ∧ c1 = k * a2 ∨ a1 = k * c2 ∧ b1 = k * a2 ∧ c1 = k * b2)

def triangle_set : set (ℕ × ℕ × ℕ) :=
  { t | let (a, b, c) := t in a ≥ b ∧ b ≥ c ∧ is_valid_side a ∧ is_valid_side b ∧ is_valid_side c ∧ is_triangle a b c }

theorem largest_non_congruent_non_similar_set :
  ∃ S : set (ℕ × ℕ × ℕ), S ⊆ triangle_set ∧ 
  (∀ t1 t2 ∈ S, ¬ is_congruent t1 t2 ∧ ¬ is_similar t1 t2) ∧ 
  ∀ T : set (ℕ × ℕ × ℕ), T ⊆ triangle_set ∧ 
  (∀ t1 t2 ∈ T, ¬ is_congruent t1 t2 ∧ ¬ is_similar t1 t2) → T.card ≤ 15 :=
sorry

end largest_non_congruent_non_similar_set_l81_81230


namespace wall_width_and_cost_l81_81133

theorem wall_width_and_cost (w h l : ℝ) 
  (h_eq : h = 6 * w) 
  (l_eq : l = 7 * h)
  (volume_eq : 42 * w * 6 * w * 0.825 * w = 16128)
  (cost_per_cubic_meter : ℝ := 50) :
  w ≈ 4.26 ∧ (16128 * cost_per_cubic_meter = 806400) :=
by
  sorry

end wall_width_and_cost_l81_81133


namespace probability_factor_of_5_factorial_l81_81691

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81691


namespace solve_equation1_solve_equation2_l81_81477

-- Define the first quadratic equation
def equation1 := λ x : ℝ, x^2 - x - 2

-- Define the solutions for the first quadratic equation
noncomputable def solution1 : set ℝ := {2, -1}

-- Prove that the solutions of equation1 are 2 and -1
theorem solve_equation1 : ∀ x : ℝ, equation1 x = 0 ↔ x ∈ solution1 := by
  sorry

-- Define the second quadratic equation
def equation2 := λ x : ℝ, 2 * x^2 + 2 * x - 1

-- Define the solutions for the second quadratic equation
noncomputable def solution2 : set ℝ := {(-1 + Real.sqrt 3) / 2, (-1 - Real.sqrt 3) / 2}

-- Prove that the solutions of equation2 are (-1 + sqrt(3))/2 and (-1 - sqrt(3))/2
theorem solve_equation2 : ∀ x : ℝ, equation2 x = 0 ↔ x ∈ solution2 := by
  sorry

end solve_equation1_solve_equation2_l81_81477


namespace nina_total_miles_l81_81936

noncomputable def kilometer_to_mile : ℝ := 0.621371
noncomputable def yard_to_mile : ℝ := 0.000568182

def initial_run : ℝ := 0.08
def run_3km_twice : ℝ := 2 * 3 * kilometer_to_mile
def run_1200_yards : ℝ := 1200 * yard_to_mile
def long_run_6km : ℝ := 6 * kilometer_to_mile

def total_miles_ran : ℝ := initial_run + run_3km_twice + run_1200_yards + long_run_6km

theorem nina_total_miles : total_miles_ran ≈ 8.2182704 := by sorry

end nina_total_miles_l81_81936


namespace probability_number_is_factor_of_120_l81_81620

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81620


namespace quadratic_zero_count_four_l81_81311

-- Definitions translated from conditions
variables (a b c : ℝ)
variable (f : ℝ → ℝ)
variable h_f : f = λ x, a * x^2 + b * x + c
variable h_a_pos : a > 0
variable h_f_neg : f (1 / a) < 0

-- Statement of proof problem
theorem quadratic_zero_count_four :
  let g := λ x : ℝ, f (f x) in
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ 
                       x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
                       g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧ g x₄ = 0 :=
sorry

end quadratic_zero_count_four_l81_81311


namespace support_percentage_correct_l81_81394

-- Define the total number of government employees and the percentage supporting the project
def num_gov_employees : ℕ := 150
def perc_gov_support : ℝ := 0.70

-- Define the total number of citizens and the percentage supporting the project
def num_citizens : ℕ := 800
def perc_citizens_support : ℝ := 0.60

-- Calculate the number of supporters among government employees
def gov_supporters : ℝ := perc_gov_support * num_gov_employees

-- Calculate the number of supporters among citizens
def citizens_supporters : ℝ := perc_citizens_support * num_citizens

-- Calculate the total number of people surveyed and the total number of supporters
def total_surveyed : ℝ := num_gov_employees + num_citizens
def total_supporters : ℝ := gov_supporters + citizens_supporters

-- Define the expected correct answer percentage
def correct_percentage_supporters : ℝ := 61.58

-- Prove that the percentage of overall supporters is equal to the expected correct percentage 
theorem support_percentage_correct :
  (total_supporters / total_surveyed * 100) = correct_percentage_supporters :=
by
  sorry

end support_percentage_correct_l81_81394


namespace sum_of_possible_values_of_q_p_l81_81921

def p_domain := {-2, -1, 0, 1}
def p_range := {-1, 1, 3, 5}
def q_domain := {0, 1, 2, 3}
def q (x : ℕ) := x + 2

theorem sum_of_possible_values_of_q_p :
  let valid_ranges := (p_range ∩ q_domain)
  sum (valid_ranges.map q) = 8 := by sorry

end sum_of_possible_values_of_q_p_l81_81921


namespace evaluate_expression_l81_81270

theorem evaluate_expression (a : ℕ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 :=
by
  -- Here would be the proof which is omitted as per instructions
  sorry

end evaluate_expression_l81_81270


namespace average_of_u_l81_81332

theorem average_of_u :
  (∃ u : ℕ, ∀ r1 r2 : ℕ, (r1 + r2 = 6) ∧ (r1 * r2 = u) → r1 > 0 ∧ r2 > 0) →
  (∃ distinct_u : Finset ℕ, distinct_u = {5, 8, 9} ∧ (distinct_u.sum / distinct_u.card) = 22 / 3) :=
sorry

end average_of_u_l81_81332


namespace cone_surface_area_l81_81019

theorem cone_surface_area (a h r l : ℝ)
  (axial_section_eq_triangle : ∀ (a : ℝ), (√3 / 4) * a^2 = √3 → a = 2)
  (height_of_cone : ∀ (h a : ℝ), h^2 + (a / 2)^2 = a^2 → h = √3)
  (radius_of_cone_base : ∀ (a : ℝ), r = a / 2 → r = 1)
  (slant_height_of_cone : ∀ (a : ℝ), l = a → l = 2)
  (lateral_surface_area : ∀ (r l : ℝ), (1 / 2) * 2 * π * r * l = 2 * π)
  (base_area : ∀ (r : ℝ), π * r^2 = π)
  (total_surface_area : ∀ (base_area lateral_surface_area : ℝ), base_area + lateral_surface_area = 3 * π),
  total_surface_area (π * r^2) ((1 / 2) * 2 * π * r * l) = 3 * π :=
sorry

end cone_surface_area_l81_81019


namespace jason_daily_charge_l81_81054

theorem jason_daily_charge 
  (total_cost_eric : ℕ) (days_eric : ℕ) (daily_charge : ℕ)
  (h1 : total_cost_eric = 800) (h2 : days_eric = 20)
  (h3 : daily_charge = total_cost_eric / days_eric) :
  daily_charge = 40 := 
by
  sorry

end jason_daily_charge_l81_81054


namespace alicia_total_payment_l81_81718

def daily_rent_cost : ℕ := 30
def miles_cost_per_mile : ℝ := 0.25
def rental_days : ℕ := 5
def driven_miles : ℕ := 500

def total_cost (daily_rent_cost : ℕ) (rental_days : ℕ)
               (miles_cost_per_mile : ℝ) (driven_miles : ℕ) : ℝ :=
  (daily_rent_cost * rental_days) + (miles_cost_per_mile * driven_miles)

theorem alicia_total_payment :
  total_cost daily_rent_cost rental_days miles_cost_per_mile driven_miles = 275 := by
  sorry

end alicia_total_payment_l81_81718


namespace determine_range_of_m_l81_81318

variable (x m : ℝ)

def p : Prop := abs (x - 4) ≤ 6
def q : Prop := x^2 - m^2 - 2 * x + 1 ≤ 0

theorem determine_range_of_m (h : ¬p → ¬q) (h_m : 0 < m) : 9 ≤ m :=
by
  sorry

end determine_range_of_m_l81_81318


namespace area_PQRS_l81_81464

-- Define the conditions
def EFGH_area := 36
def side_length_EFGH := Real.sqrt EFGH_area
def side_length_equilateral := side_length_EFGH
def displacement := (side_length_EFGH * Real.sqrt 3) / 2
def side_length_PQRS := side_length_EFGH + 2 * displacement

-- Prove the question (area of PQRS)
theorem area_PQRS : 
  (side_length_PQRS)^2 = 144 + 72 * Real.sqrt 3 := by
  sorry

end area_PQRS_l81_81464


namespace probability_factorial_five_l81_81600

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81600


namespace geralds_average_speed_l81_81469

theorem geralds_average_speed (poly_circuits : ℕ) (poly_time : ℝ) (track_length : ℝ) (gerald_speed_ratio : ℝ) :
  poly_circuits = 12 →
  poly_time = 0.5 →
  track_length = 0.25 →
  gerald_speed_ratio = 0.5 →
  let poly_speed :=  poly_circuits * track_length / poly_time in
  let gerald_speed :=  gerald_speed_ratio * poly_speed in
  gerald_speed = 3 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end geralds_average_speed_l81_81469


namespace maximum_profit_l81_81579

def L1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
def L2 (x : ℝ) : ℝ := 2 * (15 - x)
def S (x : ℝ) : ℝ := L1 x + L2 x

theorem maximum_profit :
  (∀ x, 0 ≤ x → x ≤ 15 → S x ≤ 45.6) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 15 ∧ S x = 45.6) :=
begin
  sorry
end

end maximum_profit_l81_81579


namespace probability_factorial_five_l81_81604

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81604


namespace mean_proportional_AC_is_correct_l81_81301

-- Definitions based on conditions
def AB := 4
def BC (AC : ℝ) := AB - AC

-- Lean theorem
theorem mean_proportional_AC_is_correct (AC : ℝ) :
  AC > 0 ∧ AC^2 = AB * BC AC ↔ AC = 2 * Real.sqrt 5 - 2 := 
sorry

end mean_proportional_AC_is_correct_l81_81301


namespace proposition_true_l81_81749

theorem proposition_true (a b c : ℝ) (h : a > b) : ac^2 > bc^2 := sorry

end proposition_true_l81_81749


namespace logarithmic_product_inequality_l81_81321

theorem logarithmic_product_inequality (y : ℝ) : 
  (y = log 6 / log 5 * log 7 / log 6 * log 8 / log 7 * log 9 / log 8 * log 10 / log 9) 
  → 1 < y ∧ y < 2 :=
begin
  sorry
end

end logarithmic_product_inequality_l81_81321


namespace range_of_m_l81_81325

theorem range_of_m (f : ℝ → ℝ) 
  (h_inc : ∀ a b, -2 ≤ a → a ≤ b → b ≤ 2 → f(a) ≤ f(b)) 
  (h_condition : ∀ m, -2 ≤ 1 - m → 1 - m ≤ m → m ≤ 2 → f(1 - m) < f(m)) : 
  ∀ m, (1 / 2) < m ↔ (m ≤ 2) :=
sorry

end range_of_m_l81_81325


namespace sum_of_first_three_cards_l81_81053

theorem sum_of_first_three_cards :
  ∀ (G Y : ℕ → ℕ) (cards : ℕ → ℕ),
  (∀ n, G n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) →
  (∀ n, Y n ∈ ({4, 5, 6, 7, 8} : Set ℕ)) →
  (∀ n, cards (2 * n) = G (cards n) → cards (2 * n + 1) = Y (cards n + 1)) →
  (∀ n, Y n = G (n + 1) ∨ ∃ k, Y n = k * G (n + 1)) →
  (cards 0 + cards 1 + cards 2 = 14) :=
by
  sorry

end sum_of_first_three_cards_l81_81053


namespace participants_with_exactly_five_problems_l81_81030

theorem participants_with_exactly_five_problems (n : ℕ) 
  (p : Fin 6 → Fin 6 → ℕ)
  (h1 : ∀ i j : Fin 6, i ≠ j → p i j > 2 * n / 5)
  (h2 : ¬ ∃ i : Fin 6, ∀ j : Fin 6, j ≠ i → p i j = n)
  : ∃ k1 k2 : Fin n, k1 ≠ k2 ∧ (∀ i : Fin 6, (p i k1 = 5) ∧ (p i k2 = 5)) :=
sorry

end participants_with_exactly_five_problems_l81_81030


namespace number_of_people_in_range_l81_81031

open ProbabilityTheory

noncomputable def normal_distribution (μ σ : ℝ) (X : ℝ → Measure ℝ) : Prop :=
  ∀ a b, X a b = (1 / (σ * sqrt (2 * π))) * exp (-((b - μ)^2) / (2 * σ^2))

theorem number_of_people_in_range :
  let μ := 72
  let σ := 8
  let n := 20000
  let prob_1σ := 0.6827
  let prob_2σ := 0.9545
  
  let z₁ := (80 - μ) / σ
  let z₂ := (88 - μ) / σ
  
  let prob_80_88 := (prob_2σ - prob_1σ) / 2
  let num_people := prob_80_88 * n
  
  (num_people ≈ 2718) :=
sorry

end number_of_people_in_range_l81_81031


namespace part1_part2_part3_l81_81805

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end part1_part2_part3_l81_81805


namespace height_percentage_B_l81_81482

variable (r h: ℝ)

-- Volume of the original cylinder and water filled initially
def V_original (h r: ℝ) : ℝ := π * r^2 * h
def V_water (h r: ℝ) : ℝ := π * r^2 * (5 / 6 * h)

-- Radius of Cylinder A and its water volume when 3/5 full
def r_A (r: ℝ) : ℝ := 1.25 * r
def V_A_filled (r h: ℝ) : ℝ := π * (r_A r)^2 * (3 / 5 * h)

-- Radius of Cylinder B and remaining water volume
def r_B (r: ℝ) : ℝ := 0.60 * r
def V_B (r h: ℝ) : ℝ := (2 / 5) * V_water h r

-- Height of water in Cylinder B
def h_B (r h: ℝ) : ℝ := V_B r h / (π * (r_B r)^2)

theorem height_percentage_B :
  (h_B r h / h) * 100 ≈ 92.59 := by sorry

end height_percentage_B_l81_81482


namespace probability_no_shaded_square_l81_81572

theorem probability_no_shaded_square :
  let total_rectangles := binom 1002 2 * 2 -- Total rectangles in one row and multiply by 3
  let shaded_rectangles := 501 * 501 * 3 -- Shaded rectangles in all rows
  let probability := (total_rectangles - shaded_rectangles) / total_rectangles
  probability = 500 / 1001 :=
by
  sorry

end probability_no_shaded_square_l81_81572


namespace inscribed_angle_subtended_by_chord_l81_81217

theorem inscribed_angle_subtended_by_chord
  (ratio : ℝ) (total_angle : ℝ)
  (h_ratio : ratio = 7 / 11)
  (h_total_angle : total_angle = 360) :
  ∃ (angle1 angle2 : ℝ), angle1 = 70 ∧ angle2 = 110 := by
suffices h1 : total_angle / (7 + 11) = 20, from
have h2 : (total_angle / (7 + 11)) * 7 = 140, from
have h3 : (total_angle / (7 + 11)) * 11 = 220, from
have h4 : 140 / 2 = 70, from
have h5 : 220 / 2 = 110, from
⟨70, 110, h4, h5⟩
sorry

end inscribed_angle_subtended_by_chord_l81_81217


namespace probability_factorial_five_l81_81608

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81608


namespace probability_factor_of_120_in_range_l81_81616

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81616


namespace value_of_algebraic_expression_l81_81324

variable {a b : ℝ}

theorem value_of_algebraic_expression (h : b = 4 * a + 3) : 4 * a - b - 2 = -5 := 
by
  sorry

end value_of_algebraic_expression_l81_81324


namespace minimum_cooking_time_l81_81543

noncomputable def wash_pot_and_fill_water_time : ℕ := 2
noncomputable def wash_vegetables_time : ℕ := 6
noncomputable def prepare_noodles_and_seasonings_time : ℕ := 2
noncomputable def boil_water_time : ℕ := 10
noncomputable def cook_noodles_and_vegetables_time : ℕ := 3

theorem minimum_cooking_time :
  wash_pot_and_fill_water_time + boil_water_time + cook_noodles_and_vegetables_time = 15 :=
by 
  have wash_pot_time_plus_boiling_time := wash_pot_and_fill_water_time
  have boiling_and_other_ingredients := boil_water_time
  have combined_tasks_time := (wash_vegetables_time + prepare_noodles_and_seasonings_time)
  
  have total_during_boiling := combine_tasks_time +
  have final_cooking_time := cook_noodles_and_vegetables_time
  have overall_time := total_boiling_time
  
  sorry

end minimum_cooking_time_l81_81543


namespace election_valid_votes_l81_81396

noncomputable def total_valid_votes 
  (majority : ℝ) 
  (p_win : ℝ) 
  (p_lose : ℝ) : ℝ := (majority / (p_win - p_lose))

theorem election_valid_votes (majority : ℝ) 
  (p_win : ℝ) 
  (p_lose : ℝ)
  (h_majority : majority = 174)
  (h_p_win : p_win = 0.70) 
  (h_p_lose : p_lose = 0.30) :
  total_valid_votes majority p_win p_lose = 435 :=
by
  unfold total_valid_votes
  rw [h_majority, h_p_win, h_p_lose]
  norm_num
  sorry

end election_valid_votes_l81_81396


namespace gcd_104_156_l81_81173

theorem gcd_104_156 : Nat.gcd 104 156 = 52 :=
by
  -- the proof steps will go here, but we can use sorry to skip it
  sorry

end gcd_104_156_l81_81173


namespace triangle_area_of_perpendicular_medians_l81_81447

theorem triangle_area_of_perpendicular_medians 
  (X Y Z F E G : Point) 
  (M : Triangle X Y Z) 
  (hG : is_centroid G X Y Z)
  (h_A_F : is_median AF M)
  (h_B_E : is_median BE M)
  (h_perp : is_perpendicular AF BE)
  (h_AF : AF.length = 10)
  (h_BE : BE.length = 15) 
  : M.area = 100 := 
sorry

end triangle_area_of_perpendicular_medians_l81_81447


namespace triangle_angle_b_triangle_area_l81_81846

-- Define the problem conditions and proofs
theorem triangle_angle_b {A B C : ℝ} {a b c : ℝ} (h1 : 2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : B = π / 3 := 
sorry

theorem triangle_area {A B C : ℝ} {a b c : ℝ} (h1 : 2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1) 
  (h2 : a + c = 3 * sqrt 3 / 2) (h3 : b = sqrt 3) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (1 / 2) * a * c * sin (π / 3) = 5 * sqrt 3 / 16 :=
sorry

end triangle_angle_b_triangle_area_l81_81846


namespace max_distance_l81_81888

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l81_81888


namespace sum_of_first_10_terms_l81_81356

def sequence : ℕ → ℝ
| 1 := 1
| 2 := 2
| n + 2 := (1 + (Real.cos (n * Real.pi / 2))^2) * (sequence n) + (Real.sin (n * Real.pi / 2))^2

def sum_first_10_terms : ℝ := (Finset.range 10).sum (λ i, sequence (i + 1))

theorem sum_of_first_10_terms : sum_first_10_terms = 77 := by
  sorry

end sum_of_first_10_terms_l81_81356


namespace factor_probability_l81_81597

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81597


namespace icosahedron_paths_count_l81_81227

theorem icosahedron_paths_count :
  let n := 810 in
  ∃ top bottom upper_pentagon lower_pentagon : Finset Vertex,
  no_vertex_repeated (
    (top.to_upper_paths_card * 
      (upper_pentagon_horizontal_paths_card) +
    top.to_upper_paths_card * 
      upper_to_lower_paths_card * 
      lower_pentagon_horizontal_paths_card) 
      = n

end icosahedron_paths_count_l81_81227


namespace probability_factorial_five_l81_81599

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81599


namespace modemBDownloadTime_l81_81547

variable (timeA : ℝ) (speedRatio : ℝ)

-- Condition: it takes 25.5 minutes to download a file using modem A
def modemATime : ℝ := 25.5

-- Condition: modem B works at 17% of the speed of modem A
def modemBSpeedRatio : ℝ := 0.17

-- Define the time taken by modem B
def modemBTime (timeA : ℝ) (speedRatio : ℝ) : ℝ :=
  timeA / speedRatio

-- The theorem we want to prove
theorem modemBDownloadTime : modemBTime modemATime modemBSpeedRatio = 150 := by 
  calc
    modemBTime modemATime modemBSpeedRatio
        = 25.5 / 0.17 : by rfl
    ... = 150 : by sorry

end modemBDownloadTime_l81_81547


namespace probability_factorial_five_l81_81598

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81598


namespace bob_total_calories_l81_81729

def total_calories (slices_300 : ℕ) (calories_300 : ℕ) (slices_400 : ℕ) (calories_400 : ℕ) : ℕ :=
  slices_300 * calories_300 + slices_400 * calories_400

theorem bob_total_calories 
  (slices_300 : ℕ := 3)
  (calories_300 : ℕ := 300)
  (slices_400 : ℕ := 4)
  (calories_400 : ℕ := 400) :
  total_calories slices_300 calories_300 slices_400 calories_400 = 2500 := 
by 
  sorry

end bob_total_calories_l81_81729


namespace total_peaches_l81_81246

theorem total_peaches (initial_peaches_Audrey : ℕ) (multiplier_Audrey : ℕ)
                      (initial_peaches_Paul : ℕ) (multiplier_Paul : ℕ)
                      (initial_peaches_Maya : ℕ) (additional_peaches_Maya : ℕ) :
                      initial_peaches_Audrey = 26 →
                      multiplier_Audrey = 3 →
                      initial_peaches_Paul = 48 →
                      multiplier_Paul = 2 →
                      initial_peaches_Maya = 57 →
                      additional_peaches_Maya = 20 →
                      (initial_peaches_Audrey + multiplier_Audrey * initial_peaches_Audrey) +
                      (initial_peaches_Paul + multiplier_Paul * initial_peaches_Paul) +
                      (initial_peaches_Maya + additional_peaches_Maya) = 325 :=
by
  sorry

end total_peaches_l81_81246


namespace largest_5_digit_integer_congruent_15_mod_17_l81_81174

theorem largest_5_digit_integer_congruent_15_mod_17 :
  ∃ x : ℕ, (x < 100000) ∧ (x ≥ 10000) ∧ (x % 17 = 15) ∧ ∀ y : ℕ, (y < 100000) ∧ (y % 17 = 15) → y ≤ x := 
by
  use 99977
  split
  exact sorry -- proof that 99977 < 100000
  split
  exact sorry -- proof that 99977 ≥ 10000
  split
  exact sorry -- proof that 99977 % 17 = 15
  intros y hy
  exact sorry -- proof that for all y, if y < 100000 and y % 17 = 15, then y ≤ 99977

end largest_5_digit_integer_congruent_15_mod_17_l81_81174


namespace calc_sqrt_mult_l81_81524

theorem calc_sqrt_mult : 
  ∀ (a b c : ℕ), a = 256 → b = 64 → c = 16 → 
  (nat.sqrt (nat.sqrt a) * nat.cbrt b * nat.sqrt c = 64) :=
by 
  intros a b c h1 h2 h3
  rw [h1, nat.sqrt_eq, nat.sqrt_eq, h2, nat.cbrt_eq, h3, nat.sqrt_eq]
  sorry

end calc_sqrt_mult_l81_81524


namespace first_discount_percentage_l81_81974

theorem first_discount_percentage (x : ℝ) 
  (h₁ : ∀ (p : ℝ), p = 70) 
  (h₂ : ∀ (d₁ d₂ : ℝ), d₁ = x / 100 ∧ d₂ = 0.01999999999999997 )
  (h₃ : ∀ (final_price : ℝ), final_price = 61.74):
  x = 10 := 
by
  sorry

end first_discount_percentage_l81_81974


namespace imaginary_part_of_z_l81_81208

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_z (z : ℂ) (h : 1 + z = 2 + 3 * complex.I) : imaginary_part z = 3 := by
  sorry

end imaginary_part_of_z_l81_81208


namespace compute_S6_l81_81863

variable (a q : ℕ)
variable (geom_seq : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Define the geometric sequence and its properties
axiom geom_seq_def : geom_seq 1 = 1
axiom geom_seq_step : geom_seq 4 = 8
axiom sum_geom_seq : S n = (geom_seq 1 * (1 - q ^ n)) / (1 - q)

theorem compute_S6 : geom_seq 1 = 1 → geom_seq 4 = 8 → S 6 = 63 :=
by 
  intro h1 h4
  have q_value : q = 2 := by sorry
  have sum_value : S 6 = 63 := by sorry
  exact sum_value

end compute_S6_l81_81863


namespace equilibrium_price_quantity_quantity_increase_due_to_subsidy_l81_81149

theorem equilibrium_price_quantity (p Q : ℝ) :
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (Q^D(2) = 8 ∧ Q^D(3) = 6) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ p Q, Q^D(p) = Q^S(p) ∧ Q = 10 :=
by
  intros
  have h₁ : Q^D(p) = -2 * p + 12 := sorry
  have h₂ : Q^S(p) = 2 + 8 * p := sorry
  use 1
  use 10
  simp [Q^D, Q^S]
  split
  sorry -- detailed steps to show Q^D(1) = 10 and Q^S(1) = 10

theorem quantity_increase_due_to_subsidy (p Q : ℝ) (s : ℝ) :
  s = 1 →
  (∀ p, Q^S(p) = 2 + 8 * p) →
  (∀ p, Q^D(p) = -2 * p + 12) →
  ∃ ΔQ, ΔQ = 1.6 :=
by
  intros
  have Q_s : Q^S(p + s) = 2 + 8 * (p + 1) := sorry
  have Q_d : Q^D(p) = -2 * p + 12 := sorry
  have new_p : p = 0.2 := sorry
  have new_Q : Q^S(0.2) = 11.6 := sorry
  use 1.6
  simp
  sorry -- detailed steps to show ΔQ = 1.6.

end equilibrium_price_quantity_quantity_increase_due_to_subsidy_l81_81149


namespace midpoint_of_PK_is_incenter_l81_81959

open EuclideanGeometry

theorem midpoint_of_PK_is_incenter
  (A B C P K : Point)
  (S : Circle)
  (h_iso : isIsoscelesTriangle A B C)
  (h_tangent_AB : Circle.isTangentToCircleAt S (Segment.mk B A) P)
  (h_tangent_BC : Circle.isTangentToCircleAt S (Segment.mk B C) K)
  (h_internally_tangent : S.isInternallyTangentTo (circumcircle A B C)) :
  let M := midpoint (Segment.mk P K) in
  isIncenter M (Triangle.mk A B C) :=
by
  sorry

end midpoint_of_PK_is_incenter_l81_81959


namespace probability_factor_of_120_in_range_l81_81612

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81612


namespace amelia_wins_l81_81235

noncomputable def amelia_wins_prob (am_prob : ℝ) (bl_prob : ℝ): ℝ :=
  let p := am_prob in
  let q := 1 - am_prob in
  let r := 1 - bl_prob in
  p + q * r * p / (1 - q * r)

theorem amelia_wins : amelia_wins_prob (1 / 4) (1 / 3) = 1 / 2 :=
by
  sorry

end amelia_wins_l81_81235


namespace vectors_and_angle_l81_81790

open Real

def vector (α : Type*) := (α × α)

def parallel (a b : vector ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

def perpendicular (a b : vector ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

def magnitude (v : vector ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vectors_and_angle 
  (a : vector ℝ := (3, -4))
  (b : vector ℝ := (2, -8 / 3))
  (c : vector ℝ := (2, 3 / 2))
  (h_parallel : parallel a b)
  (h_perpendicular : perpendicular a c) :
  b = (2, -8 / 3) ∧ c = (2, 3 / 2) ∧
  ∃ θ, 0 ≤ θ ∧ θ ≤ 180 ∧ Real.cos θ = 0 ∧ θ = 90 :=
by 
  sorry

end vectors_and_angle_l81_81790


namespace find_dividend_l81_81548

theorem find_dividend (quotient divisor remainder : ℕ) : quotient = 40 → divisor = 72 → remainder = 64 → divisor * quotient + remainder = 2944 :=
by
  intros hq hd hr
  rw [hq, hd, hr]
  exact rfl

end find_dividend_l81_81548


namespace find_f_l81_81352

theorem find_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 + x) :
  ∀ x : ℤ, f x = x^2 - x :=
by
  intro x
  sorry

end find_f_l81_81352


namespace beckett_younger_than_olaf_l81_81727

-- Define variables for ages
variables (O B S J : ℕ) (x : ℕ)

-- Express conditions as Lean hypotheses
def conditions :=
  B = O - x ∧  -- Beckett's age
  B = 12 ∧    -- Beckett is 12 years old
  S = O - 2 ∧ -- Shannen's age
  J = 2 * S + 5 ∧ -- Jack's age
  O + B + S + J = 71 -- Sum of ages
  
-- The theorem stating that Beckett is 8 years younger than Olaf
theorem beckett_younger_than_olaf (h : conditions O B S J x) : x = 8 :=
by
  -- The proof is omitted (using sorry)
  sorry

end beckett_younger_than_olaf_l81_81727


namespace volume_of_rotated_solid_l81_81969

theorem volume_of_rotated_solid :
  let f₁ : ℝ → ℝ := abs
  let f₂ : ℝ → ℝ := λ x, (6 - x) / 5
  let A := (-3 / 2, 3 / 2)
  let B := (1, 1)
  let θ := (2 / 3) * Real.pi
  volume_of_solid (triangle_formed_by f₁ f₂) θ = 5 / 6 := 
sorry

end volume_of_rotated_solid_l81_81969


namespace total_students_l81_81189

theorem total_students 
  (T : ℕ) 
  (below_8 : ℕ := 0.2 * T) 
  (age_8 : ℕ := 24) 
  (above_8 : ℕ := 2 / 3 * age_8) 
  (h1 : 0.2 * T = below_8)
  (h2 : age_8 = 24) 
  (h3 : above_8 = 2 / 3 * age_8) 
  (h4 : above_8 + age_8 = T - below_8) 
  : T = 50 :=
sorry

end total_students_l81_81189


namespace sin_x1_minus_x2_l81_81001

def vector_dot_product (m n : ℝ × ℝ) : ℝ :=
  m.1 * n.1 + m.2 * n.2

def m (x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin x, Real.sqrt 3 * (Real.cos x)^2)

def n (x : ℝ) : ℝ × ℝ :=
  (Real.cos x, -2)

theorem sin_x1_minus_x2 :
  ∀ (x1 x2 : ℝ), 
    0 < x1 ∧ x1 < x2 ∧ x2 < Real.pi ∧ 
    vector_dot_product (m x1) (n x1) = (1 / 2) - Real.sqrt 3 ∧ 
    vector_dot_product (m x2) (n x2) = (1 / 2) - Real.sqrt 3 -> 
  Real.sin (x1 - x2) = -(Real.sqrt 15 / 4) :=
by
  intros
  sorry

end sin_x1_minus_x2_l81_81001


namespace ada_original_seat_l81_81476

theorem ada_original_seat :
  ∃ (original_seat : ℕ), 
    original_seat ∈ {1, 2, 3, 4, 5, 6} ∧
    (∀ (Bea Ceci Dee Edie : ℕ) (Fara Gail : ℕ × ℕ),
      Bea = (original_seat + 3) % 6 + 1 ∧
      Ceci = (original_seat + 1) % 6 + 1 ∧
      Dee = (original_seat - 2 + 6 - 1) % 6 + 1 ∧ 
      Edie = (original_seat - 1 + 6 - 1) % 6 + 1 ∧
      (Fara.1, Fara.2) = (original_seat, (original_seat + 1) % 6 + 1) ∧
      (Gail.1, Gail.2) = (original_seat + 2) % 6 + 1, original_seat + 3) ∧
      (original_seat - 1 + 6 - 1) % 6 + 1 = 2)
    → original_seat = 3 :=
  sorry

end ada_original_seat_l81_81476


namespace find_AD_l81_81544

-- Definitions
variables {A B C D E : Type}
variables [triangle_ABC : Triangle A B C]
variables (AB BC CA : ℝ)
variables (on_line : A → B → C → Prop) (on_circle : A → B → C → Prop)
variables (circumcircle : A → D → Prop)

-- Conditions
def side_lengths : Prop := (AB = 15) ∧ (BC = 14) ∧ (CA = 13)
def altitude_meets_circumcircle : Prop := ∃ (D: Point), (on_line A B C) ∧ (on_circle A B C) ∧ (D ∈ circumcircle A)
def ad_value : ℝ := 63 / 4

-- Proof Problem
theorem find_AD (h1 : side_lengths AB BC CA) (h2 : altitude_meets_circumcircle D) :
  AD = ad_value :=
sorry

end find_AD_l81_81544


namespace total_expenditure_l81_81705

-- Define the conditions.
def singers : ℕ := 30
def current_robes : ℕ := 12
def robe_cost : ℕ := 2

-- Define the statement.
theorem total_expenditure (singers current_robes robe_cost : ℕ) : 
  (singers - current_robes) * robe_cost = 36 := by
  sorry

end total_expenditure_l81_81705


namespace max_value_a_squared_in_acute_triangle_l81_81854

theorem max_value_a_squared_in_acute_triangle 
    (a b c : ℝ)
    (A B C : ℝ) 
    (acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
    (sides_opposite : a = c * sin A / sin C ∧ b = c * sin B / sin C)
    (cond1 : b^2 + 4 * c^2 = 8)
    (cond2 : sin B + 2 * sin C = 6 * b * sin A * sin C) :
    a^2 = (15 - 8 * sqrt 2) / 3 :=
  sorry

end max_value_a_squared_in_acute_triangle_l81_81854


namespace percent_freshmen_liberal_arts_l81_81725

variable (T : ℝ) -- Total number of students

-- Conditions
def freshmen : ℝ := 0.40 * T
def freshmen_psych_majors : ℝ := 0.10 * T
def psych_majors_percent : ℝ := 0.50

-- Number of freshmen in the school of liberal arts
def F : ℝ := freshmen_psych_majors / psych_majors_percent

-- Percent of freshmen enrolled in the school of liberal arts
def percent_of_freshmen_in_liberal_arts : ℝ := F / freshmen

theorem percent_freshmen_liberal_arts : percent_of_freshmen_in_liberal_arts T = 0.50 :=
by sorry

end percent_freshmen_liberal_arts_l81_81725


namespace find_valid_x_base23_l81_81142

-- Define the conditions for x and x^2 in base-23
def valid_x (x : ℕ) (m : ℕ) : Prop :=
  let a := 13 in
  let expr := a * ((23^m + 23^(m - 1)) : ℕ) in
  x = expr

def valid_x_squared (x : ℕ) (m : ℕ) : Prop :=
  let a := 13 in
  x^2 =  a * (1 + 23^(2*m - 1))

theorem find_valid_x_base23 (x : ℕ) (m : ℕ) :
  valid_x x m -> valid_x_squared x m -> x = 13 * (23^m + 23^(m - 1)) := 
by
  intros
  sorry

end find_valid_x_base23_l81_81142


namespace correct_options_l81_81540

theorem correct_options :
  (∀ x : ℝ, x < 1/2 → 2*x + 1/(2*x - 1) ≤ -1) ∧
  (∃ y : ℝ, ∀ x : ℝ, y = (|x| + 5) / sqrt(|x| + 4) → y ≥ 2 → false) ∧
  (∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 2 → sqrt(2) ≤ sqrt(5*x - 2) / x ∧ sqrt(5*x - 2) / x ≤ 5/4*sqrt(2)) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 → 3 ≤ x/(y + 1) + 3/x) :=
by 
  repeat { sorry }

end correct_options_l81_81540


namespace only_PropositionB_is_correct_l81_81369

-- Define propositions as functions for clarity
def PropositionA (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) : Prop :=
  (1 / a) > (1 / b)

def PropositionB (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : Prop :=
  a ^ 3 < a

def PropositionC (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  (b + 1) / (a + 1) < b / a

def PropositionD (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : Prop :=
  c * b^2 < a * b^2

-- The main theorem stating that the only correct proposition is Proposition B
theorem only_PropositionB_is_correct :
  (∀ a b : ℝ, (a * b ≠ 0 ∧ a < b → ¬ PropositionA a b (a * b ≠ 0) (a < b))) ∧
  (∀ a : ℝ, (0 < a ∧ a < 1 → PropositionB a (0 < a) (a < 1))) ∧
  (∀ a b : ℝ, (a > b ∧ b > 0 → ¬ PropositionC a b (a > b) (b > 0))) ∧
  (∀ a b c : ℝ, (c < b ∧ b < a ∧ a * c < 0 → ¬ PropositionD a b c (c < b) (b < a) (a * c < 0))) :=
by
  -- Proof of the theorem
  sorry

end only_PropositionB_is_correct_l81_81369


namespace average_of_distinct_u_l81_81335

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end average_of_distinct_u_l81_81335


namespace find_valid_x_base23_l81_81141

-- Define the conditions for x and x^2 in base-23
def valid_x (x : ℕ) (m : ℕ) : Prop :=
  let a := 13 in
  let expr := a * ((23^m + 23^(m - 1)) : ℕ) in
  x = expr

def valid_x_squared (x : ℕ) (m : ℕ) : Prop :=
  let a := 13 in
  x^2 =  a * (1 + 23^(2*m - 1))

theorem find_valid_x_base23 (x : ℕ) (m : ℕ) :
  valid_x x m -> valid_x_squared x m -> x = 13 * (23^m + 23^(m - 1)) := 
by
  intros
  sorry

end find_valid_x_base23_l81_81141


namespace range_of_a_l81_81348

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1
noncomputable def f' (a x : ℝ) : ℝ := x^2 - a * x + a - 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → f' a x ≤ 0) ∧ (∀ x, 6 < x → f' a x ≥ 0) ↔ 5 ≤ a ∧ a ≤ 7 :=
by
  sorry

end range_of_a_l81_81348


namespace school_choir_robe_cost_l81_81704

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end school_choir_robe_cost_l81_81704


namespace f_leq_x_squared_l81_81483

variable (f : ℝ → ℝ)
variable (A : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → f(x) * f(y) ≤ y^2 * f(x / 2) + x^2 * f(y / 2))
variable (B : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f(x) ≤ 2016)

theorem f_leq_x_squared (x : ℝ) (H : 0 ≤ x) : f(x) ≤ x^2 := 
by 
	sorry

end f_leq_x_squared_l81_81483


namespace gcd_of_powers_l81_81250

-- Define the problem conditions
variables {a b m n : ℕ}
  (h_coprime : Nat.coprime a b)
  (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_m_pos : m > 0) (h_n_pos : n > 0)

-- Define the theorem statement
theorem gcd_of_powers (a b m n : ℕ)
  (h_coprime : Nat.coprime a b)
  (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_m_pos : m > 0) (h_n_pos : n > 0) :
  Nat.gcd (a^n - b^n) (a^m - b^m) = Nat.gcd a^Nat.gcd(m, n) b^Nat.gcd(m, n) :=
sorry

end gcd_of_powers_l81_81250


namespace sum_of_squares_of_roots_eq_46_l81_81742

theorem sum_of_squares_of_roots_eq_46 :
  let r s t : Real in
  (r + s + t = 8 ∧ r * s + s * t + r * t = 9) ∧ 
  (Polynomial.root (Polynomial.mk [ -2, 9, -8, 1 ]) r ∧
   Polynomial.root (Polynomial.mk [ -2, 9, -8, 1 ]) s ∧
   Polynomial.root (Polynomial.mk [ -2, 9, -8, 1 ]) t)
  → r^2 + s^2 + t^2 = 46 :=
by
  sorry

end sum_of_squares_of_roots_eq_46_l81_81742


namespace angle_A_value_cos_A_minus_2x_value_l81_81406

open Real

-- Let A, B, and C be the internal angles of triangle ABC.
variable {A B C x : ℝ}

-- Given conditions
axiom triangle_angles : A + B + C = π
axiom sinC_eq_2sinAminusB : sin C = 2 * sin (A - B)
axiom B_is_pi_over_6 : B = π / 6
axiom cosAplusx_is_neg_third : cos (A + x) = -1 / 3

-- Proof goals
theorem angle_A_value : A = π / 3 := by sorry

theorem cos_A_minus_2x_value : cos (A - 2 * x) = 7 / 9 := by sorry

end angle_A_value_cos_A_minus_2x_value_l81_81406


namespace train_speed_l81_81186

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 1600) (h2 : time = 40) : length / time = 40 := 
by
  -- use the given conditions here
  sorry

end train_speed_l81_81186


namespace remaining_integers_in_set_T_l81_81504

theorem remaining_integers_in_set_T : 
  let T := { 1 .. 100 } in
  let multiples_of (n : ℕ) := { k | k ∈ T ∧ k % n = 0 } in
  let multiples_of_4 := multiples_of 4 in
  let multiples_of_5 := multiples_of 5 in
  let multiples_of_20 := multiples_of 20 in
  let to_remove := multiples_of_4 ∪ multiples_of_5 \ multiples_of_20 in
  T.card - to_remove.card = 60 :=
by
  let T := {1 .. 100}
  let multiples_of (n : ℕ) := { k | k ∈ T ∧ k % n = 0 }
  let multiples_of_4 := multiples_of 4
  let multiples_of_5 := multiples_of 5
  let multiples_of_20 := multiples_of 20
  let to_remove := multiples_of_4 ∪ multiples_of_5 \ multiples_of_20
  have tm_4 : multiples_of_4.card = 25 := sorry
  have tm_5 : multiples_of_5.card = 20 := sorry
  have tm_20 : multiples_of_20.card = 5 := sorry
  have total_to_remove : to_remove.card = 40 := sorry
  have remaining : T.card - to_remove.card = 60 := by
    rw [total_to_remove]
    exact sorry
  exact remaining

end remaining_integers_in_set_T_l81_81504


namespace probability_factor_of_120_in_range_l81_81618

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81618


namespace max_distance_on_ellipse_l81_81897

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l81_81897


namespace probability_number_is_factor_of_120_l81_81626

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81626


namespace mary_max_earnings_l81_81080

theorem mary_max_earnings
  (max_hours : ℕ)
  (regular_rate : ℕ)
  (overtime_rate_increase_percent : ℕ)
  (first_hours : ℕ)
  (total_max_hours : ℕ)
  (total_hours_payable : ℕ) :
  max_hours = 60 →
  regular_rate = 8 →
  overtime_rate_increase_percent = 25 →
  first_hours = 20 →
  total_max_hours = 60 →
  total_hours_payable = 560 →
  ((first_hours * regular_rate) + ((total_max_hours - first_hours) * (regular_rate + (regular_rate * overtime_rate_increase_percent / 100)))) = total_hours_payable :=
by
  intros
  sorry

end mary_max_earnings_l81_81080


namespace DEF_right_angle_l81_81960

noncomputable theory

open EuclideanGeometry

-- Variables for circles and their properties
variables (S₁ S₂ S₃ : Circle)
variables {A B C : Point}
variables {D E F : Point}

-- Assumptions based on given conditions
axiom touches_externally (S₁ S₂ S₃ : Circle) : 
  touches S₁ S₂ ∧ touches S₁ S₃ ∧ touches S₂ S₃

axiom common_points (S₁ S₂ S₃ : Circle) (A B C : Point) :
  (S₁.contains A ∧ S₂.contains A) ∧ (S₁.contains B ∧ S₃.contains B) ∧ (S₂.contains C ∧ S₃.contains C)

axiom intersection_AB (A B : Point) (S₂ S₃ : Circle) (D E : Point) :
  intersects_line_circle_twice (Line.mk A B) S₂ D ∧ intersects_line_circle_twice (Line.mk A B) S₃ E
  
axiom intersection_DC_F (D C : Point) (S₃ : Circle) (F : Point) :
  intersects_line_circle_twice (Line.mk D C) S₃ F ∧ D ≠ F

-- Theorem statement
theorem DEF_right_angle (S₁ S₂ S₃ : Circle) (A B C D E F : Point)
  (Ht : touches_externally S₁ S₂ S₃)
  (Hcp : common_points S₁ S₂ S₃ A B C)
  (HintAB : intersection_AB A B S₂ S₃ D E)
  (HintDC : intersection_DC_F D C S₃ F) :
  is_right_triangle D E F :=
sorry

end DEF_right_angle_l81_81960


namespace sequence_formula_l81_81733

theorem sequence_formula (n : ℕ) : 
  (2 * (∏ i in Finset.range (n + 1), (1 - (1 / (i + 2)^2)))) = (n + 2) / (n + 1) :=
by
  induction n with k hk
  -- base case
  case zero {
    sorry
  }
  -- inductive step
  case succ {
    sorry
  }

end sequence_formula_l81_81733


namespace net_growth_rate_is_2_1_l81_81128

def birth_rate : ℕ := 32
def death_rate : ℕ := 11
def initial_population : ℕ := 1000

def net_growth_rate (birth_rate death_rate : ℕ) : ℕ :=
  birth_rate - death_rate

def net_growth_rate_perc (net_growth_rate initial_population : ℕ) : ℚ :=
  (net_growth_rate / initial_population.toRat) * 100

theorem net_growth_rate_is_2_1 :
  net_growth_rate_perc (net_growth_rate birth_rate death_rate) initial_population = 2.1 :=
by
  -- proof to be filled in
  sorry

end net_growth_rate_is_2_1_l81_81128


namespace total_students_standing_committee_ways_different_grade_pairs_ways_l81_81986

-- Given conditions
def freshmen : ℕ := 5
def sophomores : ℕ := 6
def juniors : ℕ := 4

-- Proofs (statements only, no proofs provided)
theorem total_students : freshmen + sophomores + juniors = 15 :=
by sorry

theorem standing_committee_ways : freshmen * sophomores * juniors = 120 :=
by sorry

theorem different_grade_pairs_ways :
  freshmen * sophomores + sophomores * juniors + juniors * freshmen = 74 :=
by sorry

end total_students_standing_committee_ways_different_grade_pairs_ways_l81_81986


namespace smallest_a_l81_81507

theorem smallest_a :
  ∃ a : ℕ, (∀ n : ℕ, n > 0 → ∑ i in finset.range (n+1), 1 / (n + 1 + i : ℝ) < a - 2007 * (1 / 3 : ℝ)) ∧
  (∀ b : ℕ, b > 0 → (∀ n : ℕ, n > 0 → ∑ i in finset.range (n+1), 1 / (n + 1 + i : ℝ) < b - 2007 * (1 / 3 : ℝ)) → a ≤ b) :=
sorry

end smallest_a_l81_81507


namespace length_of_train_B_l81_81519

-- Given conditions
def lengthTrainA := 125  -- in meters
def speedTrainA := 54    -- in km/hr
def speedTrainB := 36    -- in km/hr
def timeToCross := 11    -- in seconds

-- Conversion factor from km/hr to m/s
def kmhr_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Relative speed of the trains in m/s
def relativeSpeed := kmhr_to_mps (speedTrainA + speedTrainB)

-- Distance covered in the given time
def distanceCovered := relativeSpeed * timeToCross

-- Proof statement
theorem length_of_train_B : distanceCovered - lengthTrainA = 150 := 
by
  -- Proof will go here
  sorry

end length_of_train_B_l81_81519


namespace porter_previous_painting_price_l81_81100

variable (P : ℝ)

-- Conditions
def condition1 : Prop := 3.5 * P - 1000 = 49000

-- Correct Answer
def answer : ℝ := 14285.71

-- Theorem stating that the answer holds given the conditions
theorem porter_previous_painting_price (h : condition1 P) : P = answer :=
sorry

end porter_previous_painting_price_l81_81100


namespace equation_of_line_through_P_and_Q_l81_81329

open EuclideanGeometry

def point := (ℝ × ℝ)

def P : point := (-2, 5)
def Q : point := (4, 1/2)

theorem equation_of_line_through_P_and_Q :
  ∃ A B C : ℝ, (A, B, C) = (3, 4, -14) ∧ ∀ x y, (x, y) ∈ euclidean.line_with_params P Q ↔ A * x + B * y + C = 0 :=
sorry

end equation_of_line_through_P_and_Q_l81_81329


namespace parallelogram_area_greater_than_one_l81_81498

open Real

-- Defining a function to check if a point is a lattice point
def isLatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p.1 = x ∧ p.2 = y

-- Defining the condition that vertices of parallelogram are lattice points
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (lattice_A : isLatticePoint A)
  (lattice_B : isLatticePoint B)
  (lattice_C : isLatticePoint C)
  (lattice_D : isLatticePoint D)

-- Defining the area of the parallelogram
noncomputable def area (P : Parallelogram) : ℝ :=
  let (A, B, C, D) := (P.A, P.B, P.C, P.D)
  in (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Condition that there's at least one additional lattice point inside or on the sides
def hasAdditionalLatticePoint (P : Parallelogram) : Prop :=
  ∃ (P' : ℝ × ℝ), isLatticePoint P' ∧ (
    (P'.1 < maximum [P.A.1, P.B.1, P.C.1, P.D.1]) ∧ (P'.1 > minimum [P.A.1, P.B.1, P.C.1, P.D.1]) ∧
    (P'.2 < maximum [P.A.2, P.B.2, P.C.2, P.D.2]) ∧ (P'.2 > minimum [P.A.2, P.B.2, P.C.2, P.D.2])
  ) ∨ (P' = P.A) ∨ (P' = P.B) ∨ (P' = P.C) ∨ (P' = P.D)

-- Lean statement for the proof problem
theorem parallelogram_area_greater_than_one (P : Parallelogram) (h : hasAdditionalLatticePoint P) :
  area P > 1 :=
sorry

end parallelogram_area_greater_than_one_l81_81498


namespace sum_of_reciprocals_l81_81119

theorem sum_of_reciprocals (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 31 / 21) :=
sorry

end sum_of_reciprocals_l81_81119


namespace original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l81_81138

variable (a_n : ℕ → ℝ) (n : ℕ+)

-- To prove the original proposition
theorem original_proposition : (a_n n + a_n (n + 1)) / 2 < a_n n → (∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the inverse proposition
theorem inverse_proposition : ((a_n n + a_n (n + 1)) / 2 ≥ a_n n → ¬ ∀ m, a_n m ≥ a_n (m + 1)) := 
sorry

-- To prove the converse proposition
theorem converse_proposition : (∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 < a_n n := 
sorry

-- To prove the contrapositive proposition
theorem contrapositive_proposition : (¬ ∀ m, a_n m ≥ a_n (m + 1)) → (a_n n + a_n (n + 1)) / 2 ≥ a_n n :=
sorry

end original_proposition_inverse_proposition_converse_proposition_contrapositive_proposition_l81_81138


namespace max_distance_l81_81889

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l81_81889


namespace abs_diff_ps_pds_eq_31_100_l81_81215

-- Defining the conditions
def num_red : ℕ := 500
def num_black : ℕ := 700
def num_blue : ℕ := 800
def total_marbles : ℕ := num_red + num_black + num_blue
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculating P_s and P_d
def ways_same_color : ℕ := choose num_red 2 + choose num_black 2 + choose num_blue 2
def total_ways : ℕ := choose total_marbles 2
def P_s : ℚ := ways_same_color / total_ways

def ways_different_color : ℕ := num_red * num_black + num_red * num_blue + num_black * num_blue
def P_d : ℚ := ways_different_color / total_ways

-- Proving the statement
theorem abs_diff_ps_pds_eq_31_100 : |P_s - P_d| = (31 : ℚ) / 100 := by
  sorry

end abs_diff_ps_pds_eq_31_100_l81_81215


namespace probability_factorial_five_l81_81603

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81603


namespace sum_of_squares_not_square_l81_81147

theorem sum_of_squares_not_square (a : ℕ) : 
  ¬ ∃ b : ℕ, (a - 1)^2 + a^2 + (a + 1)^2 = b^2 := 
by {
  sorry
}

end sum_of_squares_not_square_l81_81147


namespace distance_to_line_l81_81760

open Real

noncomputable def distance_from_point_to_line : ℝ :=
  let a := (3 : ℝ, -2, 5) in
  let p1 := (1 : ℝ, 4, 0) in
  let p2 := (4 : ℝ, 0, 2) in
  let v := (4 - 1, 0 - 4, 2 - 0) in -- direction vector
  let t := ((29:ℝ)⁻¹ * 40) in
  let closest_point := (1 + 3 * t, 4 - 4 * t, 2 * t) in
  let distance_vector := (closest_point.1 - a.1, closest_point.2 - a.2, closest_point.3 - a.3) in
  sqrt (distance_vector.1^2 + distance_vector.2^2 + distance_vector.3^2)

theorem distance_to_line :
  distance_from_point_to_line = 70 * sqrt 3 / 29 := by
  sorry

end distance_to_line_l81_81760


namespace copy_pages_l81_81051

theorem copy_pages (total_cents : ℕ) (cost_per_page : ℕ) (pages : ℕ)
  (h₁ : total_cents = 3000) 
  (h₂ : cost_per_page = 3) : 
  pages = total_cents / cost_per_page → pages = 1000 :=
by
  intros h
  rw [h₁, h₂, h]
  simp
  sorry

end copy_pages_l81_81051


namespace fraction_to_terminating_decimal_l81_81276

theorem fraction_to_terminating_decimal :
  (53 : ℚ)/160 = 0.33125 :=
by sorry

end fraction_to_terminating_decimal_l81_81276


namespace range_of_a_l81_81971

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3a) ↔ (a ≤ -1 ∨ 4 ≤ a) :=
by 
  sorry

end range_of_a_l81_81971


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81632

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81632


namespace problem_l81_81825

def is_valid_number (n : ℕ) : Prop :=
  -- n should be a 4-digit number
  1000 ≤ n ∧ n < 10000 ∧
  -- n should use the digits 2, 0, 3, 3 exactly once
  (∃ (a b c d : ℕ),
    n = a * 10^3 + b * 10^2 + c * 10 + d ∧
    ({a, b, c, d} ⊆ {2, 0, 3} ∧
     {2, 0, 3}.count a = 1 ∧
     {2, 0, 3}.count b = 1 ∧
     {2, 3}.count c = 1 ∧
     {2, 3}.count d = 1))

def count_valid_numbers : ℕ :=
  {n | is_valid_number n ∧ n > 2000}.to_finset.card

def answer : ℕ := 5

theorem problem :
  count_valid_numbers = answer :=
by 
  -- The proof will go here.
  sorry

end problem_l81_81825


namespace percent_increase_in_area_l81_81546

variables (r : ℝ)

def medium_area : ℝ := π * r ^ 2
def large_area : ℝ := π * (1.60 * r) ^ 2

theorem percent_increase_in_area :
  (large_area r - medium_area r) / medium_area r * 100 = 156 :=
by sorry

end percent_increase_in_area_l81_81546


namespace prob_factorial_5_l81_81671

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81671


namespace cube_root_simplification_l81_81180

theorem cube_root_simplification : 
  ∃ (a b : ℕ), (∃ c : ℕ, 8000 = c ^ 3) ∧ a * b ^ c = 8000 ∧ b = 1 ∧ a + b = 21 :=
by
  use 20
  use 1
  use 20
  sorry

end cube_root_simplification_l81_81180


namespace cube_root_simplification_l81_81178

theorem cube_root_simplification : 
  ∃ (a b : ℕ), (∃ c : ℕ, 8000 = c ^ 3) ∧ a * b ^ c = 8000 ∧ b = 1 ∧ a + b = 21 :=
by
  use 20
  use 1
  use 20
  sorry

end cube_root_simplification_l81_81178


namespace right_triangle_perimeter_l81_81228

-- Conditions
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (h_area : 1 / 2 * 15 * b = 150)
variable (h_pythagorean : a^2 + b^2 = c^2)
variable (h_a : a = 15)

-- The theorem to prove the perimeter is 60 units
theorem right_triangle_perimeter : a + b + c = 60 := by
  sorry

end right_triangle_perimeter_l81_81228


namespace count_distinct_triangles_l81_81860

def is_natural_coordinate (p : (ℕ × ℕ)) : Prop :=
  true  -- This simplifies to always true since every natural coordinate is valid.

def is_centroid (G : (ℚ × ℚ)) (A B : (ℕ × ℕ)) : Prop :=
  G = ((1/3 * (A.1 + B.1) : ℚ), (1/3 * (A.2 + B.2) : ℚ))

def distinct_triangles (G : (ℚ × ℚ)) (n : ℕ) : Prop := 
  n = 90

theorem count_distinct_triangles :
  ∃ G : (ℚ × ℚ), G = (19/3, 11/3) ∧ 
  ∃ n : ℕ, distinct_triangles G n :=
by
  exists (19/3, 11/3)
  exists 90
  split
  . reflexivity
  . sorry -- Proof omitted for now

end count_distinct_triangles_l81_81860


namespace max_distance_on_ellipse_l81_81910

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l81_81910


namespace kids_attended_saturday_l81_81253

theorem kids_attended_saturday (x : ℕ) (h1 : ∀ x, 10 * x + 10 * (x / 2) = 300) : x = 20 :=
by
  have h2 : 15 * x = 300 := by rw [← h1 x]
  have h3 : x = 300 / 15 := by rw [Nat.mul_div_cancel_left _ (Nat.div_pos (by decide) (by decide))]
  exact h3

end kids_attended_saturday_l81_81253


namespace probability_two_girls_l81_81032

-- Define the conditions
def total_students := 8
def total_girls := 5
def total_boys := 3
def choose_two_from_n (n : ℕ) := n * (n - 1) / 2

-- Define the question as a statement that the probability equals 5/14
theorem probability_two_girls
    (h1 : choose_two_from_n total_students = 28)
    (h2 : choose_two_from_n total_girls = 10) :
    (choose_two_from_n total_girls : ℚ) / choose_two_from_n total_students = 5 / 14 :=
by
  sorry

end probability_two_girls_l81_81032


namespace parabola_y_intercepts_l81_81004

theorem parabola_y_intercepts : 
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x : ℝ), x = 0 → 
  ∃ (y : ℝ), 3 * y^2 - 5 * y - 2 = 0 :=
sorry

end parabola_y_intercepts_l81_81004


namespace maximize_area_l81_81219

noncomputable def optimal_fencing (L W : ℝ) : Prop :=
  (2 * L + W = 1200) ∧ (∀ L1 W1, 2 * L1 + W1 = 1200 → L * W ≥ L1 * W1)

theorem maximize_area : ∃ L W, optimal_fencing L W ∧ L + W = 900 := sorry

end maximize_area_l81_81219


namespace solve_quadratic_eq_l81_81117

theorem solve_quadratic_eq (x y : ℝ) :
  (x = 3 ∧ y = 1) ∨ (x = -1 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) ∨ (x = -1 ∧ y = -5) ↔
  x ^ 2 - x * y + y ^ 2 - x + 3 * y - 7 = 0 := sorry

end solve_quadratic_eq_l81_81117


namespace volume_of_rotated_square_cylinder_l81_81288

-- Definitions from conditions
def side_length : ℝ := 20
def radius := side_length / (2 * Real.pi)
def height := side_length

-- Theorem statement for the volume of the cylinder
theorem volume_of_rotated_square_cylinder :
  let V := Real.pi * radius^2 * height
  V = 2000 / Real.pi :=
by
  let V := Real.pi * radius^2 * height
  sorry

end volume_of_rotated_square_cylinder_l81_81288


namespace picture_distance_l81_81223

theorem picture_distance (wall_width picture_width x y : ℝ)
  (h_wall : wall_width = 25)
  (h_picture : picture_width = 5)
  (h_relation : x = 2 * y)
  (h_total : x + picture_width + y = wall_width) :
  x = 13.34 :=
by
  sorry

end picture_distance_l81_81223


namespace tan_identity_solution_l81_81265

theorem tan_identity_solution (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 360) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 270 :=
by
  sorry

end tan_identity_solution_l81_81265


namespace exists_two_distinct_primes_l81_81780

theorem exists_two_distinct_primes (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.prime p) : 
  ∃ q r : ℕ, 2 ≤ q ∧ q < p ∧ Nat.prime q ∧ 2 ≤ r ∧ r < p ∧ Nat.prime r ∧ q ≠ r ∧ 
  ¬ q ^ (p - 1) ≡ 1 [MOD p ^ 2] ∧ ¬ r ^ (p - 1) ≡ 1 [MOD p ^ 2] := 
sorry

end exists_two_distinct_primes_l81_81780


namespace student_net_pay_l81_81201

theorem student_net_pay (base_salary bonus : ℕ) (tax_rate : ℝ) (h₁ : base_salary = 25000) (h₂ : bonus = 5000)
  (h₃ : tax_rate = 0.13) : (base_salary + bonus - (base_salary + bonus) * tax_rate) = 26100 :=
by 
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end student_net_pay_l81_81201


namespace sum_squares_condition_l81_81843

theorem sum_squares_condition
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 75)
  (h2 : ab + bc + ca = 40)
  (h3 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 :=
by sorry

end sum_squares_condition_l81_81843


namespace digit_150_of_22_over_70_l81_81535

theorem digit_150_of_22_over_70 : 
    let r := periodic_decimal 11 35 
    (↑r.numerator == 22) ∧ (↑r.denominator == 70) →
    repeat_block r == 6 ∧ r.digits = [3, 1, 4, 2, 8, 5] →
  (nth_digit r 150 == 5) := 
by {
  sorry
}

end digit_150_of_22_over_70_l81_81535


namespace total_ladybugs_and_ants_l81_81159

def num_leaves : ℕ := 84
def ladybugs_per_leaf : ℕ := 139
def ants_per_leaf : ℕ := 97

def total_ladybugs := ladybugs_per_leaf * num_leaves
def total_ants := ants_per_leaf * num_leaves
def total_insects := total_ladybugs + total_ants

theorem total_ladybugs_and_ants : total_insects = 19824 := by
  sorry

end total_ladybugs_and_ants_l81_81159


namespace max_interesting_in_five_consecutive_l81_81074

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

def is_interesting (n : ℕ) : Prop :=
  is_prime (sum_of_digits n)

theorem max_interesting_in_five_consecutive :
  ∀ n : ℕ, ∃ (a b c d e : ℕ), 
    [n, n+1, n+2, n+3, n+4] = [a, b, c, d, e] →
    [is_interesting a, is_interesting b, is_interesting c, is_interesting d, is_interesting e].count true = 4 :=
sorry

end max_interesting_in_five_consecutive_l81_81074


namespace part_a_part_b_l81_81555

-- Definition of subset A = {1, ..., n}
def A (n : ℕ) : Set ℕ := { x | x ∈ Finset.range n }

-- Definition of a permutation function
def is_perm (f : ℕ → ℕ) := ∃ g : ℕ → ℕ, ∀ x, g (f x) = x ∧ f (g x) = x

-- Definition of good function for a permutation
def is_good (n : ℕ) (f σ : ℕ → ℕ) : Prop :=
  ∃ k ∈ (A n), (∀ i j, 1 ≤ i → i < j → j ≤ k → (f (σ i)) < (f (σ j))) ∧ 
  (∀ i j, k ≤ i → i < j → j ≤ n → (f (σ i)) > (f (σ j)))

-- Definition of set S_σ
def S_sigma (n : ℕ) (σ : ℕ → ℕ) : Set (ℕ → ℕ) :=
  { f | is_perm f ∧ is_good n f σ }

-- Theorem for part (a)
theorem part_a (n : ℕ) (σ : ℕ → ℕ) (h_perm_σ : is_perm σ) : 
  |S_sigma n σ| = 2^(n-1) := 
  sorry

-- Theorem for part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) : 
  ∃ (σ τ : ℕ → ℕ), is_perm σ ∧ is_perm τ ∧ 
  S_sigma n σ ∩ S_sigma n τ = ∅ := 
  sorry

end part_a_part_b_l81_81555


namespace ball_hits_ground_at_4_5_l81_81211

def height (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 180

theorem ball_hits_ground_at_4_5 :
  ∃ t : ℝ, height t = 0 ∧ t = 4.5 :=
sorry

end ball_hits_ground_at_4_5_l81_81211


namespace triangle_side_range_l81_81783

-- The conditions for the problem
variable (a b : ℝ)
variable (c : ℝ := 2)
variable (B : ℝ)
variable (h1 : b^2 - a^2 = a * c)
variable (h2 : ∀ x, (0 < x ∧ x < π/2) → cos x ∈ (0, 1))

-- The range of 'a' given the conditions
theorem triangle_side_range (hB : B ∈ (0, π/2)) : a ∈ (2/3, 2) :=
begin
  have h_cos : cos B ∈ (0, 1), from h2 B (and.intro (lt_trans zero_lt_one (half_pi_pos.mpr hB.left)) hB.right),
  have h_range : 2 + 4 * cos B ∈ (2, 6), from sorry,
  have h_a_range : a = 4 / (2 + 4 * cos B), from sorry,
  sorry
end

end triangle_side_range_l81_81783


namespace parametric_to_standard_equation_l81_81139

theorem parametric_to_standard_equation (x y t : ℝ) 
(h1 : x = 4 * t + 1) 
(h2 : y = -2 * t - 5) : 
x + 2 * y + 9 = 0 :=
by
  sorry

end parametric_to_standard_equation_l81_81139


namespace equilateral_triangle_area_l81_81486

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 3) :
  (sqrt (3:ℝ)) * real.sqrt 3 = 9 * real.sqrt 3 / 2 :=
by
  have H : 2 * real.sqrt 3 = 2 * real.sqrt 3 := by sorry 
  sorry

end equilateral_triangle_area_l81_81486


namespace number_of_truthful_monkeys_l81_81040

theorem number_of_truthful_monkeys 
  (tigers foxes monkeys : ℕ)
  (total_groups : ℕ)
  (animals_per_group : ℕ)
  (responses_tiger : ℕ)
  (responses_fox : ℕ)
  (x y m n z : ℕ) :
  tigers = 100 →
  foxes = 100 →
  monkeys = 100 →
  total_groups = 100 →
  animals_per_group = 3 →
  responses_tiger = 138 →
  responses_fox = 188 →
  (x - m + (100 - y) + n = 138) →
  (m + (100 - x - n) + (100 - z) = 188) →
  x ≤ 100 →
  y ≤ 100 →
  m ≤ x →
  n ≤ 100 - x →
  z = 12 - m + x + n →
  100 - x = 76 :=
begin
  intros,
  sorry,
end

end number_of_truthful_monkeys_l81_81040


namespace polar_eqn_hyperbola_l81_81963

theorem polar_eqn_hyperbola : 
  ∀ (ρ θ : ℝ), (ρ^2 * real.cos (2 * θ) = 1) → (False) :=
begin
  intros ρ θ h,
  -- Proof is omitted
  sorry
end

end polar_eqn_hyperbola_l81_81963


namespace simplify_expr_l81_81538

theorem simplify_expr : 
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = (5 : ℚ) / 4 := 
by
  sorry

end simplify_expr_l81_81538


namespace purchase_probability_l81_81212

/--
A batch of products from a company has packages containing 10 components each.
Each package has either 1 or 2 second-grade components. 10% of the packages
contain 2 second-grade components. Xiao Zhang will decide to purchase
if all 4 randomly selected components from a package are first-grade.

We aim to prove the probability that Xiao Zhang decides to purchase the company's
products is \( \frac{43}{75} \).
-/
theorem purchase_probability : true := sorry

end purchase_probability_l81_81212


namespace sum_S_n_div_n_arithmetic_a_n_formula_b_n_sum_formula_l81_81312

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else 4 * n - 3

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ n

def S_n (n : ℕ) : ℕ := (finset.range (n+1)).sum a_n

theorem sum_S_n_div_n_arithmetic (n : ℕ) : 2 * n - 1 = (S_n n) / n := sorry

theorem a_n_formula (n : ℕ) : a_n n = 4 * n - 3 := sorry

theorem b_n_sum_formula (n : ℕ) : finset.range (n+1).sum b_n = 2 ^ n - 1 := sorry

end sum_S_n_div_n_arithmetic_a_n_formula_b_n_sum_formula_l81_81312


namespace num_possible_sums_l81_81928

theorem num_possible_sums (A : Finset ℕ) (hA : A.card = 60) (hA_sub : ∀ x ∈ A, x ∈ Finset.range 121) :
  ∃ n : ℕ, n = 3601 ∧ ∀ S : ℕ, (∃ B : Finset ℕ, B ⊆ A ∧ B.card = 60 ∧ S = B.sum) → S ∈ Finset.range (1831, 5431) :=
by {
  sorry
}

end num_possible_sums_l81_81928


namespace calculate_expression_l81_81251

theorem calculate_expression :
  |1 - Real.sqrt 2| + (1/2)^(-2 : ℤ) - (Real.pi - 2023)^0 = Real.sqrt 2 + 2 := 
by
  sorry

end calculate_expression_l81_81251


namespace a_pow_b_eq_neg_one_l81_81371

theorem a_pow_b_eq_neg_one (a b : ℤ) (h : |a + 1| = -(b - 3)^2) : a^b = -1 := by
  sorry

end a_pow_b_eq_neg_one_l81_81371


namespace combined_weight_of_Meg_and_Chris_cats_l81_81979

-- Definitions based on the conditions
def ratio (M A C : ℕ) : Prop := 13 * A = 21 * M ∧ 13 * C = 28 * M 
def half_anne (M A : ℕ) : Prop := M = 20 + A / 2
def total_weight (M A C T : ℕ) : Prop := T = M + A + C

-- Theorem statement
theorem combined_weight_of_Meg_and_Chris_cats (M A C T : ℕ) 
  (h1 : ratio M A C) 
  (h2 : half_anne M A) 
  (h3 : total_weight M A C T) : 
  M + C = 328 := 
sorry

end combined_weight_of_Meg_and_Chris_cats_l81_81979


namespace parabola_focus_distance_l81_81061

theorem parabola_focus_distance
  (F P Q : ℝ × ℝ)
  (hF : F = (1 / 2, 0))
  (hP : ∃ y, P = (2 * y^2, y))
  (hQ : Q = (1 / 2, Q.2))
  (h_parallel : P.2 = Q.2)
  (h_distance : dist P Q = dist Q F) :
  dist P F = 2 :=
by
  sorry

end parabola_focus_distance_l81_81061


namespace combined_perimeter_of_squares_false_l81_81518

-- Define the terms and conditions
def square_side_length (a : ℝ) := a > 0
def combined_perimeter_is_sum_of_squares (a : ℝ) :=
  let square_perimeter := 4 * a in
  let rectangle_perimeter := 6 * a in
  rectangle_perimeter = 2 * square_perimeter

-- Problem statement
theorem combined_perimeter_of_squares_false (a : ℝ) (h : square_side_length a) : 
  ¬ combined_perimeter_is_sum_of_squares a :=
sorry

end combined_perimeter_of_squares_false_l81_81518


namespace picture_area_l81_81006

theorem picture_area (x y : ℕ) (hx : 1 < x) (hy : 1 < y) 
  (h_area : (3 * x + 4) * (y + 3) = 60) : x * y = 15 := 
by 
  sorry

end picture_area_l81_81006


namespace projection_of_sum_on_a_l81_81000

variable (a b : ℝ × ℝ)
variable (k : ℝ)
variable (h : a = (-1, 2) ∧ b = (k, 1))
variable (orth : a.1 * b.1 + a.2 * b.2 = 0)

theorem projection_of_sum_on_a : 
  (let a : ℝ × ℝ := (-1, 2);
       b : ℝ × ℝ := (k, 1);
       k := 2
   in (a.1 + b.1, a.2 + b.2) • a / ‖a‖ = sqrt 5) :=
by
  sorry

end projection_of_sum_on_a_l81_81000


namespace triangle_angle_A_l81_81387

theorem triangle_angle_A (a c C A : Real) (h1 : a = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) 
(h4 : Real.sin A = 1 / 2) : A = Real.pi / 6 :=
sorry

end triangle_angle_A_l81_81387


namespace fraction_comparison_and_differences_l81_81740

theorem fraction_comparison_and_differences :
  (1/3 < 0.5) ∧ (0.5 < 3/5) ∧ 
  (0.5 - 1/3 = 1/6) ∧ 
  (3/5 - 0.5 = 1/10) :=
by
  sorry

end fraction_comparison_and_differences_l81_81740


namespace B_gt_A_l81_81432

noncomputable def A :=
  let s := (fin 2006^2006) × (fin 2006^2006) × (fin 2006^2006) × (fin 2006^2006)
  set.count {t : s | (t.1.1.1 ^ 3 + t.1.1.2 ^ 2 = t.1.2.1 ^ 3 + t.2 ^ 2 + 1 : nat)}

noncomputable def B :=
  let s := (fin 2006^2006) × (fin 2006^2006) × (fin 2006^2006) × (fin 2006^2006)
  set.count {t : s | (t.1.1.1 ^ 3 + t.1.1.2 ^ 2 = t.1.2.1 ^ 3 + t.2 ^ 2 : nat)}

theorem B_gt_A : B > A :=
by
  sorry

end B_gt_A_l81_81432


namespace prob_factorial_5_l81_81670

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81670


namespace dave_winfield_home_runs_l81_81238

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end dave_winfield_home_runs_l81_81238


namespace prob_factorial_5_l81_81672

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81672


namespace sin_A_over_1_minus_cos_A_l81_81782

variable {a b c : ℝ} -- Side lengths of the triangle
variable {A B C : ℝ} -- Angles opposite to the sides

theorem sin_A_over_1_minus_cos_A 
  (h_area : 0.5 * b * c * Real.sin A = a^2 - (b - c)^2) :
  Real.sin A / (1 - Real.cos A) = 3 :=
sorry

end sin_A_over_1_minus_cos_A_l81_81782


namespace max_consecutive_interesting_integers_l81_81445

noncomputable def isInteresting (k : ℕ) : Prop :=
  let primes := List.range k |>.map Nat.prime
  let prod := primes.foldl (· * ·) 1
  prod % k = 0

theorem max_consecutive_interesting_integers :
  ∃ n : ℕ, n = 7 ∧ (∀ k : ℕ, k ≤ n → isInteresting k) :=
begin
  sorry
end

end max_consecutive_interesting_integers_l81_81445


namespace cosine_squared_identity_l81_81304

theorem cosine_squared_identity (α : ℝ) (h : sin α - cos α = 1/3) :
  cos (π/4 - α) ^ 2 = 17/18 := by
  sorry

end cosine_squared_identity_l81_81304


namespace sum_of_distinct_mn_eq_32_l81_81257

theorem sum_of_distinct_mn_eq_32 :
  (∑ (m n : ℕ) in finset.filter 
    (λ (p : ℕ × ℕ), (p.1 > 0) ∧ (p.2 > 0) ∧ (nat.lcm p.1 p.2 + nat.gcd p.1 p.2 = 2 * (p.1 + p.2) + 11))
    (finset.product (finset.range 100) (finset.range 100)), 
    m + n) = 32 :=
by
  sorry

end sum_of_distinct_mn_eq_32_l81_81257


namespace min_value_of_function_l81_81923

noncomputable def f (y : ℝ) : ℝ := 9 * y^4 + 4 * y^(-5)

theorem min_value_of_function :
  ∃ y > 0, f y = 13 ∧ ∀ z > 0, f z ≥ 13 :=
begin
  sorry
end

end min_value_of_function_l81_81923


namespace james_fish_tanks_l81_81871

theorem james_fish_tanks (n t1 t2 t3 : ℕ) (h1 : t1 = 20) (h2 : t2 = 2 * t1) (h3 : t3 = 2 * t1) (h4 : t1 + t2 + t3 = 100) : n = 3 :=
sorry

end james_fish_tanks_l81_81871


namespace initial_pile_counts_l81_81990

def pile_transfers (A B C : ℕ) : Prop :=
  (A + B + C = 48) ∧
  ∃ (A' B' C' : ℕ), 
    (A' = A + B) ∧ (B' = B + C) ∧ (C' = C + A) ∧
    (A' = 2 * 16) ∧ (B' = 2 * 12) ∧ (C' = 2 * 14)

theorem initial_pile_counts :
  ∃ A B C : ℕ, pile_transfers A B C ∧ A = 22 ∧ B = 14 ∧ C = 12 :=
by
  sorry

end initial_pile_counts_l81_81990


namespace disk_color_alignment_l81_81876

theorem disk_color_alignment (n m : ℕ) (C : Fin n → Fin m) :
  n = 36 → m = 2 → (∀ x, C x = 18) → 
  ∃ k : Fin n, ∑ i : Fin (36), if C i = C (i + k) then 1 else 0 ≥ 18 := 
by
  intro h1 h2 h3
  sorry

end disk_color_alignment_l81_81876


namespace sum_a99_a100_l81_81818

def a_n : ℕ → ℚ 
| 1     := 1
| (n+2) := 
  let k := nat.find (λ k, n < k * (k + 1) / 2) in
  let i := k * (k + 1) / 2 - n in
  ((k : ℚ) - (i : ℚ) + 1) / i

theorem sum_a99_a100 : a_n 99 + a_n 100 = 37 / 24 := 
by
  sorry

end sum_a99_a100_l81_81818


namespace max_distance_l81_81886

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l81_81886


namespace induction_sequence_l81_81520

-- Given conditions
def is_positive_nat (n : ℕ) := n > 0

-- Theorem to prove
theorem induction_sequence (k : ℕ) (h : is_positive_nat k) :
  (k + 1) * (k + 2) * ... * (2 * k) * (2 * k + 1) * (2 * k + 2) / (k + 1) = 
  2 * ((2 * k + 1)) :=
sorry

end induction_sequence_l81_81520


namespace scrap_cookie_radius_l81_81709

-- Define the side length of the square cookie dough.
def side_length_square_dough : ℝ := 6

-- Define the radius of the large cookies.
def radius_large_cookie : ℝ := 1

-- Define the radius of the small cookies.
def radius_small_cookie : ℝ := 0.5

-- Number of large cookies.
def num_large_cookies : ℕ := 4

-- Number of small cookies.
def num_small_cookies : ℕ := 5

-- Define the area of the square cookie dough.
def area_square_dough : ℝ := side_length_square_dough ^ 2

-- Define the area of one large cookie.
def area_large_cookie : ℝ := π * radius_large_cookie ^ 2

-- Define the area of one small cookie.
def area_small_cookie : ℝ := π * radius_small_cookie ^ 2

-- Define the total area of the large cookies.
def total_area_large_cookies : ℝ := num_large_cookies * area_large_cookie

-- Define the total area of the small cookies.
def total_area_small_cookies : ℝ := num_small_cookies * area_small_cookie

-- Define the total area of all small cookies.
def total_area_all_cookies: ℝ := total_area_large_cookies + total_area_small_cookies

-- Define the area of the leftover scrap.
def area_scrap : ℝ := area_square_dough - total_area_all_cookies

-- Define a theorem to prove the radius of the scrap cookie.
theorem scrap_cookie_radius : sqrt area_scrap = sqrt 30.75 :=
by
  sorry

end scrap_cookie_radius_l81_81709


namespace existence_of_fractions_l81_81751

-- Definitions for problem conditions
def is_irreducible (a b : ℕ) : Prop := Nat.gcd a b = 1

def different_denominators (fractions : List (ℕ × ℕ)) : Prop :=
  fractions.map Prod.snd = (List.range 2018).map (λ n => n + 1)

def denominator_of_difference_less (fractions : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ) (hi : i < 2018) (hj : j < 2018) (i ≠ j), 
  let diff_denominator := (fractions[i].snd * fractions[j].snd) in
  diff_denominator < fractions[i].snd ∧ diff_denominator < fractions[j].snd

-- Main theorem statement
theorem existence_of_fractions :
  ∃ (fractions : List (ℕ × ℕ)),
    List.length fractions = 2018 ∧
    (∀ (i : ℕ) (hi : i < 2018), 
      is_irreducible (fractions[i].fst) (fractions[i].snd)
    ) ∧ 
    different_denominators fractions ∧
    denominator_of_difference_less fractions := 
sorry

end existence_of_fractions_l81_81751


namespace triangle_geom_seq_l81_81847

variable {α : Type} [LinearOrderedField α] [Real α]
variables (a b c : α)
variables (A B : Real.Angle)

-- Conditions
def is_geom_seq (a b c : α) := b ^ 2 = a * c
def eqn_given (a b c : α) := a^2 - c^2 = a * c - b * c

-- Translation of the problem into a Lean 4 statement
theorem triangle_geom_seq 
  (h1 : is_geom_seq a b c) 
  (h2 : eqn_given a b c)
  (sin_B : Real.Angle.sin B = b * Real.Angle.sin A / a) :
  b * Real.Angle.sin B / c = Real.sqrt (3 / 4) := sorry

end triangle_geom_seq_l81_81847


namespace tori_pass_grade_l81_81167

theorem tori_pass_grade : 
  let total_problems := 90
  let arithmetic_problems := 20
  let algebra_problems := 30
  let geometry_problems := 40
  let arithmetic_correct := 0.8 * arithmetic_problems
  let algebra_correct := 0.5 * algebra_problems
  let geometry_correct := 0.55 * geometry_problems
  let total_correct := arithmetic_correct + algebra_correct + geometry_correct
  let passing_grade := 0.65 * total_problems
  let needed_correct := Nat.ceil passing_grade
  total_problems = 90 →
  arithmetic_problems = 20 →
  algebra_problems = 30 →
  geometry_problems = 40 →
  arithmetic_correct = 16 →
  algebra_correct = 15 →
  geometry_correct = 22 →
  needed_correct = 59 →
  needed_correct - total_correct = 6 := 
by
  sorry

end tori_pass_grade_l81_81167


namespace relationship_between_a_b_l81_81426

noncomputable def a : ℝ := 0.2 ^ 0.1
noncomputable def b : ℝ := real.log 0.1 / real.log 0.2

theorem relationship_between_a_b :
  b > 1 ∧ 1 > a :=
by
  sorry

end relationship_between_a_b_l81_81426


namespace age_of_B_l81_81191

variables (A B C : ℝ)

theorem age_of_B :
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 :=
by
  intro h1 h2
  sorry

end age_of_B_l81_81191


namespace periodic_sequence_l81_81981

noncomputable def a : ℕ → ℚ
| 0     := 2
| (n+1) := (1 + a n) / (1 - a n)

def T : ℕ → ℚ 
| 0     := 1
| (n+1) := T n * a n

theorem periodic_sequence :
  (∀ k < 4, a (k + 4) = a k) ∧ (a 0 * a 1 * a 2 * a 3 = 1) → T 2014 = -6 :=
by 
  sorry

end periodic_sequence_l81_81981


namespace f_at_4_l81_81131

-- Define the conditions on the function f
variable (f : ℝ → ℝ)
variable (h_domain : true) -- All ℝ → ℝ functions have ℝ as their domain.

-- f is an odd function
axiom h_odd : ∀ x : ℝ, f (-x) = -f x

-- Given functional equation
axiom h_eqn : ∀ x : ℝ, f (2 * x - 3) - 2 * f (3 * x - 10) + f (x - 3) = 28 - 6 * x 

-- The goal is to determine the value of f(4), which should be 8.
theorem f_at_4 : f 4 = 8 :=
sorry

end f_at_4_l81_81131


namespace PerpendicularityTheorem_PerpendicularAndInclinedLines_ThreePerpendiculars_DihedralAngles_EqualityInequalityDihedralAngles_IntersectionLineTheorem_l81_81085

-- Define the Perpendicularity Theorem
theorem PerpendicularityTheorem (P: Point) (Π: Plane) : ∃! line, line ⊥ Π ∧ P ∈ line := sorry

-- Define the Theorem on Perpendicular and Inclined Lines
theorem PerpendicularAndInclinedLines (P: Point) (Π: Plane) : 
  (∀ l, l ⊥ Π → l.length ≤ l'.length ∧ l' ∈ inclined_lines(P, Π)) ∧ 
  (∀ l l', equal_projection(l, l') → l.length = l'.length) ∧ 
  (∀ l l', ¬equal_length(l, l') → (longer_projection(l, l') → l.length < l'.length)) := sorry

-- Define the Theorem of Three Perpendiculars
theorem ThreePerpendiculars (P: Point) (l1 l2: Line) (Π: Plane) : 
  l1 ⊥ Π ∧ l2 ⊥ Π ∧ P ∈ l1 ∧ P ∈ l2 → l1 ⊥ l2 := sorry

-- Define the Theorem on Dihedral Angles
theorem DihedralAngles (α β: DihedralAngle) : 
  α = β → linear_angle(α) = linear_angle(β) := sorry

-- Define the Theorems on Equality and Inequality of Dihedral Angles
theorem EqualityInequalityDihedralAngles (α β: DihedralAngle) : 
  α = β ↔ linear_angle(α) = linear_angle(β) ∧ 
  (α < β ↔ linear_angle(α) < linear_angle(β)) := sorry

-- Define the Intersection Line Theorem
theorem IntersectionLineTheorem (Π1 Π2 Π3: Plane) : 
  Π1 ⊥ Π3 ∧ Π2 ⊥ Π3 → (line_of_intersection(Π1, Π2) ⊥ Π3) := sorry

end PerpendicularityTheorem_PerpendicularAndInclinedLines_ThreePerpendiculars_DihedralAngles_EqualityInequalityDihedralAngles_IntersectionLineTheorem_l81_81085


namespace modular_inverse_7_mod_26_l81_81283

/-- Find the modular inverse of 7 mod 26. -/
theorem modular_inverse_7_mod_26 :
  ∃ a : ℤ, 0 ≤ a ∧ a ≤ 25 ∧ (7 * a ≡ 1 [ZMOD 26]) :=
by
  use 15
  split
  · exact dec_trivial
  split
  · exact dec_trivial
  · exact dec_trivial

end modular_inverse_7_mod_26_l81_81283


namespace vector_magnitude_example_l81_81014

noncomputable def unit_vector (v : ℝ → ℝ) : Prop :=
  ∥v∥ = 1

noncomputable def angle (v₁ v₂ : ℝ → ℝ) (θ : ℝ) : Prop :=
  (v₁ ⋅ v₂) = ∥v₁∥ * ∥v₂∥ * real.cos θ

noncomputable def vector_magnitude (v : ℝ → ℝ) : ℝ :=
  real.sqrt (v ⋅ v)

theorem vector_magnitude_example
  (e₁ e₂ : ℝ → ℝ)
  (he₁ : unit_vector e₁)
  (he₂ : unit_vector e₂)
  (h_angle : angle e₁ e₂ (real.pi / 3))
  (a := (λ i, 2 * e₁ i - e₂ i)) :
  vector_magnitude a = real.sqrt 3 :=
sorry

end vector_magnitude_example_l81_81014


namespace determine_f_one_l81_81917

noncomputable def S := {x : ℝ // x ≠ 0}

variable (k : ℝ) (hk : k ≠ 0)
variable (f : S → S)

axiom f_property1 : ∀ x : S, f ⟨1 / x.val, by simp [x.property]⟩ = Real.cos (k * x.val) * x.val * f x
axiom f_property2 : ∀ x y : S, (x.val + y.val ≠ 0) → f ⟨1 / x.val, by simp [x.property]⟩ + f ⟨1 / y.val, by simp [y.property]⟩ = 1 + f ⟨1 / (x.val + y.val), by simp [x.property, y.property]⟩

theorem determine_f_one : f ⟨1, by norm_num⟩ = ⟨1, by norm_num⟩ :=
sorry

end determine_f_one_l81_81917


namespace min_area_of_square_projection_l81_81497

theorem min_area_of_square_projection (a : ℝ) (θ : ℝ) (ha : a = 4 ∨ a = 4 / cos θ) (hcos : 0 < cos θ ∧ cos θ ≤ 1) : (∃ (min_area : ℝ), min_area = 16) :=
by
  use 16
  sorry

end min_area_of_square_projection_l81_81497


namespace log_tower_6_l81_81073

def tower : ℕ → ℕ 
| 1       => 3
| (n + 1) => 3 ^ tower n

def C : ℕ := tower 5 ^ tower 5
def D : ℕ := tower 5 ^ C

theorem log_tower_6 : 
  ∃ m : ℕ, (∀ k : ℕ, k ≤ m → k > 0 → has_log3 k D) ∧ m = 6 := 
sorry

end log_tower_6_l81_81073


namespace medians_form_right_triangle_medians_inequality_l81_81485

variable {α : Type*}
variables {a b c : ℝ}
variables {m_a m_b m_c : ℝ}
variable (orthogonal_medians : m_a * m_b = 0)

-- Part (a)
theorem medians_form_right_triangle
  (orthogonal_medians : m_a * m_b = 0) :
  m_a^2 + m_b^2 = m_c^2 :=
sorry

-- Part (b)
theorem medians_inequality
  (orthogonal_medians : m_a * m_b = 0)
  (triangle_sides : a^2 + b^2 = 5 * c^2): 
  5 * (a^2 + b^2 - c^2) ≥ 8 * a * b :=
sorry

end medians_form_right_triangle_medians_inequality_l81_81485


namespace find_f_10_l81_81778

def f (x : ℝ) : ℝ := sorry

axiom f_mul (x y : ℝ) : f(x) * f(y) = f(x + y)
axiom f_1 : f(1) = 2

theorem find_f_10 : f(10) = 1024 :=
by
repeat sorry

end find_f_10_l81_81778


namespace probability_is_13_over_30_l81_81680

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81680


namespace time_addition_example_proof_l81_81870

noncomputable def time_addition_result (initial_time : Nat × Nat × Nat := (3, 0, 0))
  (hours : Nat := 315) (minutes : Nat := 58) (seconds : Nat := 16) : Nat :=
let (h, m, s) := initial_time in
let total_seconds := h * 3600 + m * 60 + s + hours * 3600 + minutes * 60 + seconds in
let new_h := ((total_seconds / 3600) % 12) in
let new_m := (total_seconds % 3600) / 60 in
let new_s := total_seconds % 60 in
new_h + new_m + new_s

theorem time_addition_example_proof :
  time_addition_result () = 77 :=
by
  sorry

end time_addition_example_proof_l81_81870


namespace distance_PF_equilateral_l81_81062

-- Given conditions as definitions
def F : ℝ × ℝ := (1/2, 0)
def directrix l : ℝ := -1/2
def parabola (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 2 * P.1
def lies_on_directrix (Q : ℝ × ℝ) : Prop := Q.1 = -1/2
def parallel_to_x_axis (PQ : ℝ × ℝ) : Prop := PQ.2 = 0
def equidistant (PQ QF : ℝ) : Prop := PQ = QF

-- The key property we want to prove
theorem distance_PF_equilateral (P Q : ℝ × ℝ) (hP : parabola P) (hQ : lies_on_directrix Q) (h1 : parallel_to_x_axis (P.1 - Q.1, 0)) (h2 : equidistant ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ((Q.1 - F.1)^2 + (Q.2 - F.2)^2)) : 
  ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 2^2 :=
by sorry

end distance_PF_equilateral_l81_81062


namespace integral_of_piecewise_function_l81_81346

def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 else 2 - x

theorem integral_of_piecewise_function :
  ∫ x in 0..2, f x = 5 / 6 :=
by
  sorry

end integral_of_piecewise_function_l81_81346


namespace find_a_given_coefficient_l81_81379

theorem find_a_given_coefficient (a : ℝ) (h : (5.choose 2) * ((-1 / 2 : ℝ) ^ 2) * (a ^ 3) = 20) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l81_81379


namespace marble_ratio_l81_81996

theorem marble_ratio (W L M : ℕ) (h1 : W = 16) (h2 : L = W + W / 4) (h3 : W + L + M = 60) :
  M / (W + L) = 2 / 3 := 
sorry

end marble_ratio_l81_81996


namespace discount_for_multiple_rides_l81_81185

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end discount_for_multiple_rides_l81_81185


namespace difference_of_fractions_l81_81374

theorem difference_of_fractions (p q : ℕ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (5/6) - (1/7) = 29/42 := by
sorrr

end difference_of_fractions_l81_81374


namespace g_strictly_increasing_on_0_pi_div_4_l81_81716

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x)

theorem g_strictly_increasing_on_0_pi_div_4 :
  StrictMonoOn g (Set.Ioo 0 (π / 4)) :=
by
  sorry

end g_strictly_increasing_on_0_pi_div_4_l81_81716


namespace product_of_digits_of_next_palindromic_year_l81_81510

theorem product_of_digits_of_next_palindromic_year (start_year : ℕ) (gap : ℕ) (next_palindromic_year : ℕ) :
  start_year = 2992 ∧ gap = 300 ∧ next_palindromic_year = 3322 → 
  (start_year + gap = 3292 ∧ 
  next_palindromic_year > (start_year + gap) ∧ 
  (string.reverse (next_palindromic_year.repr) = next_palindromic_year.repr) ∧ 
  (next_palindromic_year / 1000) * ((next_palindromic_year % 1000) / 100) * ((next_palindromic_year % 100) / 10) * (next_palindromic_year % 10) = 36) :=
by
  sorry

end product_of_digits_of_next_palindromic_year_l81_81510


namespace factory_production_exceeds_60000_from_2022_l81_81851

theorem factory_production_exceeds_60000_from_2022 (lg : ℝ → ℝ) (h_lg2 : lg 2 = 0.3010) (h_lg3 : lg 3 = 0.4771)  :
  ∀ n : ℕ, n ≥ 7 → 20000 * (1.2 ^ n) > 60000 :=
by
  intro n hn
  -- The actual proof steps would go here to demonstrate the inequality,
  -- but they are omitted as per the problem statement.
  sorry

end factory_production_exceeds_60000_from_2022_l81_81851


namespace minimum_elements_in_Z_l81_81505

variables {Z : Type} {A B : fin n → set Z}
  (h_partitionA : ∀ i j, i ≠ j → disjoint (A i) (A j))
  (h_partitionB : ∀ i j, i ≠ j → disjoint (B i) (B j))
  (h_union_ge_n : ∀ i j, (A i ∪ B j).card ≥ n)
  (h_nonemptyA : ∀ i, (A i).nonempty)
  (h_nonemptyB : ∀ i, (B i).nonempty)

theorem minimum_elements_in_Z (h_disjoint : (∀ i j, i ≠ j → disjoint (A i) (A j)) ∧ 
    (∀ i j, i ≠ j → disjoint (B i) (B j)) ∧ 
    ∀ i j, (A i ∪ B j).card ≥ n ∧ 
    ∀ i, (A i).nonempty ∧ 
    ∀ i, (B i).nonempty) : 
    ∃ Z, Z.card ≥ n^2/2 ∧ Z.card = n^2/2 :=
begin
  sorry
end

end minimum_elements_in_Z_l81_81505


namespace convex_polygon_quadrilateral_division_l81_81545

open Nat

theorem convex_polygon_quadrilateral_division (n : ℕ) : ℕ :=
  if h : n > 0 then
    1 / (2 * n - 1) * (Nat.choose (3 * n - 3) (n - 1))
  else
    0

end convex_polygon_quadrilateral_division_l81_81545


namespace evaluate_expression_at_two_l81_81177

theorem evaluate_expression_at_two: 
  (3 * 2^2 - 4 * 2 + 2) = 6 := 
by 
  sorry

end evaluate_expression_at_two_l81_81177


namespace cube_root_simplification_l81_81179

theorem cube_root_simplification : 
  ∃ (a b : ℕ), (∃ c : ℕ, 8000 = c ^ 3) ∧ a * b ^ c = 8000 ∧ b = 1 ∧ a + b = 21 :=
by
  use 20
  use 1
  use 20
  sorry

end cube_root_simplification_l81_81179


namespace vertical_asymptote_singleton_l81_81769

theorem vertical_asymptote_singleton (c : ℝ) :
  (∃ x, (x^2 - 2 * x + c) = 0 ∧ ((x - 1) * (x + 3) = 0) ∧ (x ≠ 1 ∨ x ≠ -3)) 
  ↔ (c = 1 ∨ c = -15) :=
by
  sorry

end vertical_asymptote_singleton_l81_81769


namespace prob_factorial_5_l81_81666

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81666


namespace complex_number_quadrant_l81_81021

def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_quadrant (z : ℂ) (h : z * complex.i = 2 + 3 * complex.i) :
  in_fourth_quadrant z :=
by
  sorry

end complex_number_quadrant_l81_81021


namespace children_left_l81_81994

-- Define the initial problem constants and conditions
def totalGuests := 50
def halfGuests := totalGuests / 2
def numberOfMen := 15
def numberOfWomen := halfGuests
def numberOfChildren := totalGuests - (numberOfWomen + numberOfMen)
def proportionMenLeft := numberOfMen / 5
def totalPeopleStayed := 43
def totalPeopleLeft := totalGuests - totalPeopleStayed

-- Define the proposition to prove
theorem children_left : 
  totalPeopleLeft - proportionMenLeft = 4 := by 
    sorry

end children_left_l81_81994


namespace composition_of_rotations_is_rotation_l81_81472

variable {Point Line Plane : Type}
variable (intersect_at : Line → Line → Point → Prop)
variable (perpendicular : Plane → Line → Prop)
variable (angle_between : Plane → Plane → ℝ)

-- Definitions for conditions
def rotation_about (axis : Line) : Type := sorry
def composition (r1 r2 : Type) : Type := sorry

-- Given hypotheses
axiom l1 l2 l : Line
axiom O : Point
axiom α1 β2 : Plane
axiom intersect_axes : intersect_at l1 l2 O
axiom perp_α1_l1 : perpendicular α1 l1
axiom perp_β2_l2 : perpendicular β2 l2
axiom l_intersection : l = Plane.intersection α1 β2

-- Theorem Statement
theorem composition_of_rotations_is_rotation 
  (r1 : rotation_about l1) 
  (r2 : rotation_about l2) :
  ∃ (r : rotation_about l),
    composition r1 r2 = r ∧ rotation_angle r = 2 * angle_between α1 β2 := sorry

end composition_of_rotations_is_rotation_l81_81472


namespace no_fascinating_412_414_451_l81_81130

noncomputable def F : ℤ → ℤ := sorry

def fascinating (F : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, F(x) = F(a - x)

axiom F_property (F : ℤ → ℤ) : ∀ c : ℤ, ∃ x : ℤ, F(x) ≠ c

theorem no_fascinating_412_414_451 : ¬fascinating F 412 ∧ ¬fascinating F 414 ∧ ¬fascinating F 451 :=
by
  sorry

end no_fascinating_412_414_451_l81_81130


namespace probability_factor_of_120_l81_81643

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81643


namespace find_a_for_max_value_l81_81494

theorem find_a_for_max_value :
  (∃ a : ℝ, (∀ x ∈ Icc (-3 : ℝ) 2, ax³ + 2 * a * x + 1 ≤ 4) ∧ 
  (∃ x ∈ Icc (-3 : ℝ) 2, ax³ + 2 * a * x + 1 = 4)) ↔ 
    (a = 1/4 ∨ a = -1/11) := 
sorry

end find_a_for_max_value_l81_81494


namespace market_price_correct_l81_81025

-- Definitions based on conditions
def initial_tax_rate : ℝ := 3.5 / 100
def new_tax_rate : ℝ := 3.333 / 100
def tax_savings : ℝ := 14
def tax_rate_difference : ℝ := initial_tax_rate - new_tax_rate

-- Market price calculation
def market_price := tax_savings / tax_rate_difference

-- The statement we want to prove
theorem market_price_correct :
  market_price = 8235.29 :=
by
  sorry

end market_price_correct_l81_81025


namespace chosen_number_probability_factorial_5_l81_81656

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81656


namespace points_in_triangle_l81_81559

theorem points_in_triangle (points : Finset (ℝ × ℝ)) (h : points.card = 2015)
  (area_cond : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points → 
    triangle_area A B C ≤ 1) :
  ∃ (A B C : ℝ × ℝ), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ triangle_area A B C = 4 := 
sorry

end points_in_triangle_l81_81559


namespace isabella_non_yellow_houses_l81_81409

variable (Green Yellow Red Blue Pink : ℕ)

axiom h1 : 3 * Yellow = Green
axiom h2 : Red = Yellow + 40
axiom h3 : Green = 90
axiom h4 : Blue = (Green + Yellow) / 2
axiom h5 : Pink = (Red / 2) + 15

theorem isabella_non_yellow_houses : (Green + Red + Blue + Pink - Yellow) = 270 :=
by 
  sorry

end isabella_non_yellow_houses_l81_81409


namespace star_polygon_points_l81_81744

theorem star_polygon_points (n : ℕ) (A B : ℕ → ℝ) 
  (h_angles_congruent_A : ∀ i j, A i = A j)
  (h_angles_congruent_B : ∀ i j, B i = B j)
  (h_angle_relation : ∀ i, A i = B i - 15) :
  n = 24 :=
by
  sorry

end star_polygon_points_l81_81744


namespace steel_scrap_problem_l81_81162

theorem steel_scrap_problem 
  (x y : ℝ)
  (h1 : x + y = 140)
  (h2 : 0.05 * x + 0.40 * y = 42) :
  x = 40 ∧ y = 100 :=
by
  -- Solution steps are not required here
  sorry

end steel_scrap_problem_l81_81162


namespace Sally_fries_total_l81_81106

theorem Sally_fries_total 
  (sally_fries_initial : ℕ)
  (mark_fries_initial : ℕ)
  (fries_given_by_mark : ℕ)
  (one_third_of_mark_fries : mark_fries_initial = 36 → fries_given_by_mark = mark_fries_initial / 3) :
  sally_fries_initial = 14 → mark_fries_initial = 36 → fries_given_by_mark = 12 →
  let sally_fries_final := sally_fries_initial + fries_given_by_mark
  in sally_fries_final = 26 := 
by
  intros h1 h2 h3
  unfold sally_fries_final
  rw [h1, h3]
  exact rfl

end Sally_fries_total_l81_81106


namespace quadrilateral_property_l81_81224

noncomputable def cyclic_tangential_quadrilateral (a b c d x y : ℝ) :=
  -- Conditions of the problem
  a = 80 ∧ b = 100 ∧ c = 140 ∧ d = 120 ∧ 
  -- The quadrilateral is cyclic and tangential
  let s := (a + b + c + d) / 2 in
  let area := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) in
  let r := area / s in
  let n1 := y in
  let n2 := x in
  -- Using symmetry and point of tangency properties:
  x = (c + d - b - a) / 2 ∧ y = (c + d + b + a) / 2 - c

-- The proof statement
theorem quadrilateral_property : 
  ∀ (a b c d x y : ℝ),
  cyclic_tangential_quadrilateral a b c d x y →
  |x - y| = 50.726 :=
by sorry

end quadrilateral_property_l81_81224


namespace max_distance_on_ellipse_l81_81915

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l81_81915


namespace simplify_and_multiply_roots_l81_81532

theorem simplify_and_multiply_roots :
  (256 = 4^4) →
  (64 = 4^3) →
  (16 = 4^2) →
  ∜256 * ∛64 * sqrt 16 = 64 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end simplify_and_multiply_roots_l81_81532


namespace vertical_distance_l81_81455

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end vertical_distance_l81_81455


namespace max_distance_B_P_l81_81900

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l81_81900


namespace well_performing_student_net_pay_l81_81197

def base_salary : ℤ := 25000
def bonus : ℤ := 5000
def tax_rate : ℝ := 0.13

def total_earnings (base_salary bonus : ℤ) : ℤ :=
  base_salary + bonus

def income_tax (total_earnings : ℤ) (tax_rate : ℝ) : ℤ :=
  total_earnings * (Real.toRat tax_rate)

def net_pay (total_earnings income_tax: ℤ) : ℤ :=
  total_earnings - income_tax

theorem well_performing_student_net_pay :
  net_pay (total_earnings base_salary bonus) (income_tax (total_earnings base_salary bonus) tax_rate) = 26100 := by
  sorry

end well_performing_student_net_pay_l81_81197


namespace tax_percentage_excess_income_l81_81028

theorem tax_percentage_excess_income :
  ∀ (rate : ℝ) (total_tax income : ℝ), 
  rate = 0.15 →
  total_tax = 8000 →
  income = 50000 →
  (total_tax - income * rate) / (income - 40000) = 0.2 :=
by
  intros rate total_tax income hrate htotal hincome
  -- proof omitted
  sorry

end tax_percentage_excess_income_l81_81028


namespace prob_factorial_5_l81_81667

theorem prob_factorial_5! :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 30 }
  let factors_120 := { n | n ∣ 120 }
  let favorable_outcomes := S ∩ factors_120
  let probability := (favorable_outcomes.card * 15) = (S.card * 8)
  probability := true :=
by
  sorry

end prob_factorial_5_l81_81667


namespace simplify_and_multiply_roots_l81_81531

theorem simplify_and_multiply_roots :
  (256 = 4^4) →
  (64 = 4^3) →
  (16 = 4^2) →
  ∜256 * ∛64 * sqrt 16 = 64 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end simplify_and_multiply_roots_l81_81531


namespace find_other_endpoint_l81_81975

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ) 
  (h_mid_x : x_m = (x_1 + x_2) / 2)
  (h_mid_y : y_m = (y_1 + y_2) / 2)
  (h_x_m : x_m = 3)
  (h_y_m : y_m = 4)
  (h_x_1 : x_1 = 0)
  (h_y_1 : y_1 = -1) :
  (x_2, y_2) = (6, 9) :=
sorry

end find_other_endpoint_l81_81975


namespace average_of_u_l81_81333

theorem average_of_u :
  (∃ u : ℕ, ∀ r1 r2 : ℕ, (r1 + r2 = 6) ∧ (r1 * r2 = u) → r1 > 0 ∧ r2 > 0) →
  (∃ distinct_u : Finset ℕ, distinct_u = {5, 8, 9} ∧ (distinct_u.sum / distinct_u.card) = 22 / 3) :=
sorry

end average_of_u_l81_81333


namespace find_coordinates_of_c_l81_81820

-- Define vectors in space
def vector_a : ℝ × ℝ × ℝ := (0, 1, -1)
def vector_b : ℝ × ℝ × ℝ := (1, 2, 3)

-- Vector \(\overrightarrow{c}\) given the conditions
def vector_c : ℝ × ℝ × ℝ := (3 * fst vector_a, 3 * (vector_a.snd), 3 * (vector_a.snd.snd)) - vector_b

-- Proof statement
theorem find_coordinates_of_c : vector_c = (-1, 1, -6) := by
  -- Proof goes here
  sorry

end find_coordinates_of_c_l81_81820


namespace EFZY_is_cyclic_l81_81970

open EuclideanGeometry

-- Define the points and their properties
variables {A B C X D E F Y Z : Point}

-- The conditions of the given problem
axiom incircle_TOUCHES_triangle_ABC (h : Triangle A B C) :
  ∃ D E F, (IncircleTouchesAt D B C) ∧ (IncircleTouchesAt E C A) ∧ (IncircleTouchesAt F A B)

axiom point_X_within_triangle (hx : PointInTriangle X A B C) :
  ∃ Y Z, (IncircleTouchesAt D B C) ∧ (IncircleTouchesAt Y C X) ∧ (IncircleTouchesAt Z X B)

-- Main statement to prove
theorem EFZY_is_cyclic : CyclicQuadrilateral E F Z Y :=
sorry

end EFZY_is_cyclic_l81_81970


namespace max_distance_B_P_l81_81902

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l81_81902


namespace ratio_of_x_intercepts_l81_81171

theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) (h1 : s = -b / 8) (h2 : t = -b / 4) : s / t = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l81_81171


namespace max_distance_on_ellipse_l81_81911

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l81_81911


namespace negate_proposition_l81_81792

theorem negate_proposition :
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0)) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) :=
by
  sorry

end negate_proposition_l81_81792


namespace optimal_position_theorem_l81_81296

noncomputable def optimal_position (a b a1 b1 : ℝ) : ℝ :=
  (b / 2) + (b1 / (2 * a1)) * (a - a1)

theorem optimal_position_theorem 
  (a b a1 b1 : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0) :
  ∃ x, x = optimal_position a b a1 b1 := by
  sorry

end optimal_position_theorem_l81_81296


namespace triangle_area_percentage_difference_l81_81999

variables (b h : ℝ)
noncomputable def area_triangle (base height : ℝ) : ℝ := (1/2) * base * height

theorem triangle_area_percentage_difference :
  let base_B : ℝ := b,
      height_B : ℝ := h,
      base_A : ℝ := 1.10 * b,
      height_A : ℝ := 0.90 * h,
      area_B : ℝ := area_triangle base_B height_B,
      area_A : ℝ := area_triangle base_A height_A in
  ((area_A - area_B) / area_B) * 100 = -1 :=
by
  let base_B := b
  let height_B := h
  let base_A := 1.10 * b
  let height_A := 0.90 * h
  let area_B := area_triangle base_B height_B
  let area_A := area_triangle base_A height_A
  have h1 : area_A = (1/2) * 1.10 * b * 0.90 * h := by sorry
  have h2 : area_A = 0.99 * area_B := by sorry
  have h3 : (area_A - area_B) / area_B = -0.01 := by sorry
  show ((area_A - area_B) / area_B) * 100 = -1 from by sorry

end triangle_area_percentage_difference_l81_81999


namespace find_a_solve_inequality_l81_81802

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem find_a (h : ∀ x : ℝ, f x a = -f (-x) a) : a = 1 := sorry

theorem solve_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : f x 1 > 3 := sorry

end find_a_solve_inequality_l81_81802


namespace shaded_region_area_l81_81145

-- Given conditions
def diagonal_PQ : ℝ := 10
def number_of_squares : ℕ := 20

-- Definition of the side length of the squares
noncomputable def side_length := diagonal_PQ / (4 * Real.sqrt 2)

-- Area of one smaller square
noncomputable def one_square_area := side_length * side_length

-- Total area of the shaded region
noncomputable def total_area_of_shaded_region := number_of_squares * one_square_area

-- The theorem to be proven
theorem shaded_region_area : total_area_of_shaded_region = 62.5 := by
  sorry

end shaded_region_area_l81_81145


namespace impossible_sum_999999999_l81_81787

def isRearrangement (n m : ℕ) : Prop :=
  let digits (k : ℕ) : List ℕ :=
    if k = 0 then []
    else k.digits
  List.perm (digits n) (digits m)

theorem impossible_sum_999999999 (n m : ℕ) :
  isRearrangement n m → m + n ≠ 999999999 :=
by
  sorry

end impossible_sum_999999999_l81_81787


namespace solve_system_of_equations_l81_81284

theorem solve_system_of_equations :
  ∃ x y : ℚ, 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -259 / 29 ∧ y = -38 / 29 := by {
  use [-259 / 29, -38 / 29],
  split,
  { -- Shoot, you need to fill in the proof for this piece
    sorry },
  split,
  { -- Same for this part
    sorry },
  split,
  { -- And this one
    sorry },
  { -- Finally this one
    sorry }
}

end solve_system_of_equations_l81_81284


namespace second_interest_rate_l81_81951

theorem second_interest_rate (P1 P2 : ℝ) (r : ℝ) (total_amount total_income: ℝ) (h1 : total_amount = 2500)
  (h2 : P1 = 1500.0000000000007) (h3 : total_income = 135) :
  P2 = total_amount - P1 →
  P1 * 0.05 = 75 →
  P2 * r = 60 →
  r = 0.06 :=
sorry

end second_interest_rate_l81_81951


namespace factor_probability_l81_81589

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81589


namespace quiz_show_valid_guesses_l81_81580

theorem quiz_show_valid_guesses :
  let digits := [2, 2, 2, 4, 4, 4, 4],
      valid_prices : Fin 10000 := to .succ_to_fin_set,
      valid_guess_count := 
        (finset.attach $ finset.powerset_len 3 (finset.of_sorted_list digits)).card *
        (finset.powerset_len 3 (finset.range 7)).card - 3 :=
  valid_guess_count = 420 :=
by
  sorry

end quiz_show_valid_guesses_l81_81580


namespace probability_factorial_five_l81_81605

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81605


namespace non_negative_dot_product_l81_81789

theorem non_negative_dot_product
  (a b c d e f g h : ℝ) :
  (a * c + b * d ≥ 0) ∨ (a * e + b * f ≥ 0) ∨ (a * g + b * h ≥ 0) ∨
  (c * e + d * f ≥ 0) ∨ (c * g + d * h ≥ 0) ∨ (e * g + f * h ≥ 0) :=
sorry

end non_negative_dot_product_l81_81789


namespace inequality_l81_81777

variable {f : ℝ → ℝ}

theorem inequality (h1 : ∀ x ∈ Ioc 0 (π / 2), f x * Real.tan x + (deriv f) x < 0) :
  √3 * f (π / 3) < f (π / 6) :=
sorry

end inequality_l81_81777


namespace can_transform_1220_to_2012_cannot_transform_1220_to_2021_l81_81414

def can_transform (abcd : ℕ) (wxyz : ℕ) : Prop :=
  ∀ a b c d w x y z, 
  abcd = a*1000 + b*100 + c*10 + d ∧ 
  wxyz = w*1000 + x*100 + y*10 + z →
  (∃ (k : ℕ) (m : ℕ), 
    (k = a ∧ a ≠ d  ∧ m = c  ∧ c ≠ w ∧ 
     w = b + (k - b) ∧ x = c + (m - c)) ∨
    (k = w ∧ w ≠ x  ∧ m = y  ∧ y ≠ z ∧ 
     z = a + (k - a) ∧ x = d + (m - d)))
          
theorem can_transform_1220_to_2012 : can_transform 1220 2012 :=
sorry

theorem cannot_transform_1220_to_2021 : ¬ can_transform 1220 2021 :=
sorry

end can_transform_1220_to_2012_cannot_transform_1220_to_2021_l81_81414


namespace transformed_graph_l81_81998

theorem transformed_graph (x : ℝ) : 
    (∀ x, f(x) = 2 * sin (4 * x - π/3)) :=
by
  sorry

end transformed_graph_l81_81998


namespace CM_eq_EN_l81_81402

-- Definitions for the conditions
structure RegularOctagon (A B C D E F G H : Type) extends EuclideanGeometry :=
  (sides_eq : ∀ {P Q}, P ≠ Q → distances_eq P Q)
  (cyclic : Cyclic A B C D E F G H)

variables {A B C D E F G H : Type} [RegularOctagon A B C D E F G H]

-- Points M and N
noncomputable def PointM (A D C E : Type) : Type :=
  intersection (line_through A D) (line_through C E)

noncomputable def PointN (C D E G : Type) : Type :=
  intersection (line_through C D) (line_through E G)

-- Problem statement
theorem CM_eq_EN (A B C D E F G H : Type) [RegularOctagon A B C D E F G H]
  (M : Type := PointM A D C E) (N : Type := PointN C D E G) :
  distance C M = distance E N :=
sorry

end CM_eq_EN_l81_81402


namespace select_two_integers_divisibility_l81_81916

open Polynomial

theorem select_two_integers_divisibility
  (F : Polynomial ℤ)
  (m : ℕ)
  (a : Fin m → ℤ)
  (H : ∀ n : ℤ, ∃ i : Fin m, a i ∣ F.eval n) :
  ∃ i j : Fin m, i ≠ j ∧ ∀ n : ℤ, ∃ k : Fin m, k = i ∨ k = j ∧ a k ∣ F.eval n :=
by
  sorry

end select_two_integers_divisibility_l81_81916


namespace complex_fraction_simplification_l81_81111

theorem complex_fraction_simplification :
  (4 + 7 * complex.i) / (4 - 7 * complex.i) + (4 - 7 * complex.i) / (4 + 7 * complex.i) = -66 / 65 :=
by
  sorry

end complex_fraction_simplification_l81_81111


namespace exists_interval_and_polynomial_l81_81059

theorem exists_interval_and_polynomial (p q : ℤ) :
  ∃ (I : set ℝ) (P : polynomial ℤ),
    (∃ a b : ℝ, a < b ∧ (I = set.Icc a b) ∧ (b - a = 1 / q)) ∧
    (∀ x ∈ I, abs (P.eval x - (p : ℚ) / q) < 1 / (q : ℚ)^2) :=
sorry

end exists_interval_and_polynomial_l81_81059


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81634

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81634


namespace probability_is_13_over_30_l81_81681

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81681


namespace range_of_g_l81_81351

theorem range_of_g (a b : ℝ) (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = a * x + b)
  (h₂ : ∀ x ∈ (set.Icc (a-4) a), f x = - f (-x))
  (h₃ : ∀ x, g x = b * x + a / x) :
  set.range (λ x : ℝ, if x ∈ set.Icc (-4) (-1) then g x else 0) = set.Icc (-2 : ℝ) (-1 / 2) := 
sorry

end range_of_g_l81_81351


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81641

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81641


namespace midpoint_moves_along_circle_l81_81104

open EuclideanGeometry

theorem midpoint_moves_along_circle
  {APBQ : Type}
  [InscribedQuadrilateral APBQ]
  (ω : Circle)
  (P Q A B : ω.Point)
  (h1 : ∠P = 90°)
  (h2 : ∠Q = 90°)
  (h3 : dist A P = dist A Q)
  (h4 : dist A P < dist B P)
  (X : Segment P Q)
  (h5 : IsVariablePoint X (Segment P Q))
  (S : ω.Point)
  (h6 : LineThrough A X ∩ ω = {A, S})
  (T : Arc A Q B)
  (h7 : Perpendicular (Segment X T) (LineThrough A X))
  (M : Midpoint (Chord S T)) :
  ∃ (C : Circle), ∀ x ∈ Segment P Q, Midpoint (Chord (S x) (T x)) ∈ C :=
sorry

end midpoint_moves_along_circle_l81_81104


namespace probability_factor_of_120_l81_81652

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81652


namespace julie_bought_boxes_l81_81874

-- Definitions for the conditions
def packages_per_box := 5
def sheets_per_package := 250
def sheets_per_newspaper := 25
def newspapers := 100

-- Calculations based on conditions
def total_sheets_needed := newspapers * sheets_per_newspaper
def sheets_per_box := packages_per_box * sheets_per_package

-- The goal: to prove that the number of boxes of paper Julie bought is 2
theorem julie_bought_boxes : total_sheets_needed / sheets_per_box = 2 :=
  by
    sorry

end julie_bought_boxes_l81_81874


namespace evaluate_expression_l81_81271

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end evaluate_expression_l81_81271


namespace cosine_B_value_range_of_b_l81_81850

theorem cosine_B_value (A B C a b c : ℝ)
  (hSides : a + c = 1)
  (hCosine : cos C + cos A * cos B = sqrt 3 * sin A * cos B) :
  cos B = 1 / 2 := 
sorry

theorem range_of_b (A B C a b c : ℝ)
  (hSides : a + c = 1)
  (hCosine : cos C + cos A * cos B = sqrt 3 * sin A * cos B)
  (hCosB : cos B = 1 / 2)
  (h0a1 : 0 < a ∧ a < 1) :
  1 / 2 ≤ b ∧ b < 1 :=
sorry

end cosine_B_value_range_of_b_l81_81850


namespace ratio_cone_to_cylinder_l81_81285

def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

def volume_cone (r h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

theorem ratio_cone_to_cylinder (r : ℝ) (h_cyl h_cone : ℝ) (h_cyl_eq : h_cyl = 10) (h_cone_eq : h_cone = 5) (r_eq : r = 5) :
  (volume_cone r h_cone) / (volume_cylinder r h_cyl) = 1 / 6 :=
by
  -- proof goes here
  sorry

end ratio_cone_to_cylinder_l81_81285


namespace area_PCD_eq_l81_81748

/-- Define the points P, D, and C as given in the conditions. -/
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 18⟩
def D : Point := ⟨3, 18⟩
def C (q : ℝ) : Point := ⟨0, q⟩

/-- Define the function to compute the area of triangle PCD given q. -/
noncomputable def area_triangle_PCD (q : ℝ) : ℝ :=
  1 / 2 * (D.x - P.x) * (P.y - q)

theorem area_PCD_eq (q : ℝ) : 
  area_triangle_PCD q = 27 - 3 / 2 * q := 
by 
  sorry

end area_PCD_eq_l81_81748


namespace digits_equal_l81_81103

theorem digits_equal {n : ℕ} : 
  (1000^n < 1974^n) ∧ (10^(3*n) ≤ 1974^n) ∧ (1974^n + 2^n < 10^(3*n + 1)) ∧((3^n + 1) % 8 ≠ 0) :=
begin
  sorry
end

end digits_equal_l81_81103


namespace six_points_two_triangles_l81_81308

/-- Given 6 points on a plane such that no 3 of them are collinear, 
    there exist two (not necessarily disjoint) sets of three points each, 
    such that the smallest angle in the two triangles determined by these triplets is different. 
-/
theorem six_points_two_triangles (P : Fin 6 → ℝ × ℝ)
  (h_nocollinear : ∀ i j k : Fin 6, i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear ({P i, P j, P k} : Set (ℝ × ℝ))) :
  ∃ (A B C D E F : Fin 6), A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F ∧
    (∃ α1 α2 : ℝ, α1 ≠ α2 ∧
      (∃ Δ1 Δ2 : Triangle, Δ1 = Triangle.mk (P A) (P B) (P C) ∧ Δ2 = Triangle.mk (P D) (P E) (P F) ∧
        α1 = minAngle Δ1 ∧ α2 = minAngle Δ2)) :=
begin
  sorry
end

end six_points_two_triangles_l81_81308


namespace factor_1_factor_2_factor_3_l81_81277

-- Consider the variables a, b, x, y
variable (a b x y : ℝ)

-- Statement 1: Factorize 3a^3 - 6a^2 + 3a
theorem factor_1 : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
by
  sorry
  
-- Statement 2: Factorize a^2(x - y) + b^2(y - x)
theorem factor_2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a^2 - b^2) :=
by
  sorry

-- Statement 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factor_3 : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
by
  sorry

end factor_1_factor_2_factor_3_l81_81277


namespace combined_age_of_teachers_janitor_principal_l81_81124

theorem combined_age_of_teachers_janitor_principal 
  (avg_students_age : ℕ)
  (num_students : ℕ)
  (avg_students_teachers_age : ℕ)
  (num_students_teachers : ℕ)
  (avg_all_age : ℕ)
  (num_all : ℕ)
  (total_age_students : ℕ)
  (total_age_students_teachers : ℕ)
  (total_age_all : ℕ)
  (combined_age_teachers : ℕ)
  (combined_age_janitor_principal : ℕ) :
  avg_students_age = 18 →
  num_students = 30 →
  avg_students_teachers_age = 19 →
  num_students_teachers = 32 →
  avg_all_age = 20 →
  num_all = 34 →
  total_age_students = avg_students_age * num_students →
  total_age_students_teachers = avg_students_teachers_age * num_students_teachers →
  total_age_all = avg_all_age * num_all →
  combined_age_teachers = total_age_students_teachers - total_age_students →
  combined_age_janitor_principal = total_age_all - total_age_students_teachers →
  (combined_age_teachers + combined_age_janitor_principal) = 140 :=
by
  intros
  combine
  sorry

end combined_age_of_teachers_janitor_principal_l81_81124


namespace platform_length_l81_81573

theorem platform_length
  (train_length : ℕ)
  (crossing_platform_time : ℕ)
  (crossing_pole_time : ℕ)
  (train_speed : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 300)
  (h2 : crossing_platform_time = 45)
  (h3 : crossing_pole_time = 18)
  (h4 : train_speed = train_length.to_real / crossing_pole_time.to_real)
  (h5 : platform_length = train_speed * crossing_platform_time.to_real - train_length.to_real) :
  platform_length = 450 := by
  sorry

end platform_length_l81_81573


namespace train_time_to_pass_platform_l81_81711

noncomputable def total_distance (train_length platform_length : ℕ) : ℕ :=
train_length + platform_length

noncomputable def speed_in_meters_per_second (speed_km_per_hr : ℝ) : ℝ :=
(speed_km_per_hr * 1000) / 3600

noncomputable def time_to_pass_platform (distance speed : ℝ) : ℝ :=
distance / speed

theorem train_time_to_pass_platform :
  ∀ (train_length platform_length : ℕ) (speed_km_per_hr : ℝ),
    train_length = 360 →
    platform_length = 150 →
    speed_km_per_hr = 45 →
    time_to_pass_platform (total_distance train_length platform_length) (speed_in_meters_per_second speed_km_per_hr) = 40.8 :=
by
  intros train_length platform_length speed_km_per_hr h_train_length h_platform_length h_speed_km_per_hr
  have h1: total_distance train_length platform_length = 510 :=
    by rw [h_train_length, h_platform_length]; exact rfl
  have h2: speed_in_meters_per_second speed_km_per_hr = 12.5 :=
    by rw h_speed_km_per_hr; exact rfl
  rw [h1, h2]
  exact rfl


end train_time_to_pass_platform_l81_81711


namespace gcd_problem_l81_81550

theorem gcd_problem : 
  let a := 690
  let b := 875
  let r1 := 10
  let r2 := 25
  let n1 := a - r1
  let n2 := b - r2
  gcd n1 n2 = 170 :=
by
  sorry

end gcd_problem_l81_81550


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81631

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81631


namespace average_of_distinct_u_l81_81336

theorem average_of_distinct_u :
  let u_values := { u : ℕ | ∃ (r_1 r_2 : ℕ), r_1 + r_2 = 6 ∧ r_1 * r_2 = u }
  u_values = {5, 8, 9} ∧ (5 + 8 + 9) / 3 = 22 / 3 :=
by
  sorry

end average_of_distinct_u_l81_81336


namespace k_squared_minus_3k_minus_4_l81_81302

theorem k_squared_minus_3k_minus_4 (a b c d k : ℚ)
  (h₁ : (2 * a) / (b + c + d) = k)
  (h₂ : (2 * b) / (a + c + d) = k)
  (h₃ : (2 * c) / (a + b + d) = k)
  (h₄ : (2 * d) / (a + b + c) = k) :
  k^2 - 3 * k - 4 = -50 / 9 ∨ k^2 - 3 * k - 4 = 6 :=
  sorry

end k_squared_minus_3k_minus_4_l81_81302


namespace problem_statement_l81_81444

-- Definition of arithmetic sequence condition
def arithmetic_seq_condition (S_n a_n : ℕ → ℕ) : Prop :=
∀ n : ℕ, 2 * a_n = S_n + 1

-- Definition of b_n sequence
def b_n (a_n : ℕ → ℕ) (n : ℕ) : ℚ :=
(a_n(n + 1) : ℚ) / ((a_n(n + 1) - 1) * (a_n(n + 2) - 1))

-- The main theorem statement
theorem problem_statement (S_n a_n : ℕ → ℕ)
  (h₁ : arithmetic_seq_condition S_n a_n)
  (h₂ : a_n 1 = 1)
  (h₃ : ∀ n : ℕ, n ≥ 2 → 2 * a_n(n - 1) = S_n(n - 1) + 1) :
  (∀ n : ℕ, a_n n = 2^(n-1)) ∧ (∀ n : ℕ, (2/3 : ℚ) ≤ ∑ i in finset.range n, b_n a_n i ∧ ∑ i in finset.range n, b_n a_n i < (1 : ℚ)) :=
by
  sorry

end problem_statement_l81_81444


namespace maria_borrowed_336_l81_81078

def maria_hourly_wage (hour : ℕ) : ℕ :=
  2 * (hour % 6 + 1)

noncomputable def total_earnings (hours : ℕ) : ℕ :=
  (finset.range hours).sum maria_hourly_wage

theorem maria_borrowed_336 :
  total_earnings 48 = 336 :=
by
  sorry

end maria_borrowed_336_l81_81078


namespace symmetric_line_equation_l81_81491

theorem symmetric_line_equation (x y : ℝ) :
  let l := λ (x y : ℝ), x + 3 * y - 2 = 0,
      A := (2, 0),
      B := (0, 2 / 3),
      B' := (0, -2 / 3),
      symmetric_line := λ (x y : ℝ), x - 3 * y - 2 = 0
  in ∀ x y, symmetric_line x y ↔ l x y := sorry

end symmetric_line_equation_l81_81491


namespace maximum_k_value_l81_81416

def k (G : SimpleGraph (Fin n)) : ℕ :=
  sorry  -- Placeholder for the minimal value of k for boxes representation

def M (n : ℕ) : ℕ :=
  if n > 1 then
    let graphs := {G : SimpleGraph (Fin n) // true} in
    graphs.foldl (λ acc G => max acc (k G)) 0
  else
    0 -- Set a minimum of zero for n <= 1 for consistency

theorem maximum_k_value (n : ℕ) (h : n > 1) : M n = n / 2 :=
  sorry

end maximum_k_value_l81_81416


namespace smallest_square_area_l81_81565

theorem smallest_square_area (a b c d : ℕ) (hsquare : ∃ s : ℕ, s ≥ a + c ∧ s * s = a * b + c * d) :
    (a = 3) → (b = 5) → (c = 4) → (d = 6) → ∃ s : ℕ, s * s = 49 :=
by
  intros h1 h2 h3 h4
  cases hsquare with s hs
  use s
  -- Here we need to ensure s * s = 49
  sorry

end smallest_square_area_l81_81565


namespace find_other_number_l81_81190

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 192) (h_hcf : Nat.gcd A B = 16) (h_A : A = 48) : B = 64 :=
by
  sorry

end find_other_number_l81_81190


namespace iron_water_weight_equality_l81_81209

theorem iron_water_weight_equality :
  ∀ (iron water : ℝ), (iron = 1) → (water = 1) → (iron = water) :=
by intros iron water h_iron h_water
   rw [h_iron, h_water]
   exact eq.refl 1

end iron_water_weight_equality_l81_81209


namespace smallest_area_of_square_containing_rectangles_l81_81566

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end smallest_area_of_square_containing_rectangles_l81_81566


namespace has_zero_in_interval_l81_81806

noncomputable def f (x : ℝ) : ℝ := 2^x + 2*x - 3

theorem has_zero_in_interval : 
  ∃ k : ℤ, ∃ x : ℝ, (x ∈ Ioo k (k + 1)) ∧ (f x = 0) ∧ k = 0 :=
by
  have h₀ : f 0 = -2 := by
    dsimp [f]
    norm_num
  have h₁ : f 1 = 1 := by
    dsimp [f]
    norm_num
  have h_mono : ∀ x y : ℝ, x < y → f x < f y := sorry    -- Assuming increasing over ℝ, needs proof
  sorry

end has_zero_in_interval_l81_81806


namespace unspent_portion_is_43_over_48_l81_81952

-- Definitions for conditions:
variable (G : ℝ) -- Spending limit of the gold card
def platinum_limit : ℝ := 2 * G
def diamond_limit : ℝ := 6 * G

def gold_balance : ℝ := G / 4
def platinum_balance : ℝ := platinum_limit G / 8
def diamond_balance : ℝ := diamond_limit G / 16

def new_platinum_balance : ℝ := gold_balance G + platinum_balance G
def transferred_to_diamond : ℝ := new_platinum_balance G / 2
def new_diamond_balance : ℝ := diamond_balance G + transferred_to_diamond G

-- Unspent portion of diamond card limit
def unspent_portion : ℝ := (diamond_limit G - new_diamond_balance G) / diamond_limit G

-- Proof statement
theorem unspent_portion_is_43_over_48 (G: ℝ) (hG : 0 < G) :
  unspent_portion G = 43 / 48 :=
sorry

end unspent_portion_is_43_over_48_l81_81952


namespace find_f2_l81_81306

theorem find_f2 (a b : ℝ) (h : -8 * a - 2 * b - 4 = 2) : a * (2:ℝ)^3 + b * (2:ℝ) - 4 = -10 :=
by
  have h₁ : -8 * a - 2 * b = 6 := by linarith
  have h₂ : 8 * a + 2 * b - 4 = -10 := sorry
  exact h₂

end find_f2_l81_81306


namespace probability_factorial_five_l81_81602

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81602


namespace divisible_by_13_l81_81101

theorem divisible_by_13 (n : ℤ) : 13 ∣ (1 + 3^(3*n+1) + 9^(3*n+1)) := sorry

end divisible_by_13_l81_81101


namespace tom_apple_fraction_l81_81997

theorem tom_apple_fraction (initial_oranges initial_apples oranges_sold_fraction oranges_remaining total_fruits_remaining apples_initial apples_sold_fraction : ℕ→ℚ) :
  initial_oranges = 40 →
  initial_apples = 70 →
  oranges_sold_fraction = 1 / 4 →
  oranges_remaining = initial_oranges - initial_oranges * oranges_sold_fraction →
  total_fruits_remaining = 65 →
  total_fruits_remaining = oranges_remaining + (initial_apples - initial_apples * apples_sold_fraction) →
  apples_sold_fraction = 1 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tom_apple_fraction_l81_81997


namespace count_integers_with_properties_l81_81827

/-- Prove that the number of integers between 1500 and 2500 such that 
    the units digit equals the sum of the other digits, and the first (thousands) 
    digit is a prime number, is 51. -/
theorem count_integers_with_properties : 
  let primes := [2, 5] in
  (∀ n : ℕ, 1500 ≤ n ∧ n ≤ 2500 ∧ 
    (n % 10 = (n / 1000 + (n % 1000) / 100 + (n % 100) / 10 )) ∧
    (n / 1000 ∈ primes)) →
  ∃ (count : ℕ), count = 51 :=
by
  sorry

end count_integers_with_properties_l81_81827


namespace simplify_expression_l81_81113

theorem simplify_expression {a b : ℝ} :
  ( ((a - b) ^ 2 + a * b) / ((a + b) ^ 2 - a * b) / 
    ((a ^ 5 + b ^ 5 + a ^ 2 * b ^ 3 + a ^ 3 * b ^ 2) / 
     ((a ^ 3 + b ^ 3 + a ^ 2 * b + a * b ^ 2) * (a ^ 3 - b ^ 3))) = a - b :=
by
  sorry

end simplify_expression_l81_81113


namespace geralds_average_speed_l81_81467

theorem geralds_average_speed :
  ∀ (track_length : ℝ) (pollys_laps : ℕ) (pollys_time : ℝ) (geralds_factor : ℝ),
  track_length = 0.25 →
  pollys_laps = 12 →
  pollys_time = 0.5 →
  geralds_factor = 0.5 →
  (geralds_factor * (pollys_laps * track_length / pollys_time)) = 3 :=
by
  intro track_length pollys_laps pollys_time geralds_factor
  intro h_track_len h_pol_lys_laps h_pollys_time h_ger_factor
  sorry

end geralds_average_speed_l81_81467


namespace max_lateral_surface_area_of_cylinder_l81_81801

-- Definitions based on conditions
def sphere_volume (R : ℝ) : ℝ := (4 / 3) * π * R^3
def lateral_surface_area (r l : ℝ) : ℝ := 2 * π * r * l
def satisfies_conditions (r l R : ℝ) : Prop :=
  r^2 + (l / 2)^2 = R^2 ∧ sphere_volume R = (4 / 3) * π

-- Theorem statement
theorem max_lateral_surface_area_of_cylinder (r l R : ℝ) 
  (h_conditions : satisfies_conditions r l R) : 
  lateral_surface_area r l ≤ 2 * π :=
sorry

end max_lateral_surface_area_of_cylinder_l81_81801


namespace area_PQRS_l81_81465

-- Define the conditions
def EFGH_area := 36
def side_length_EFGH := Real.sqrt EFGH_area
def side_length_equilateral := side_length_EFGH
def displacement := (side_length_EFGH * Real.sqrt 3) / 2
def side_length_PQRS := side_length_EFGH + 2 * displacement

-- Prove the question (area of PQRS)
theorem area_PQRS : 
  (side_length_PQRS)^2 = 144 + 72 * Real.sqrt 3 := by
  sorry

end area_PQRS_l81_81465


namespace find_a_l81_81022

def z (a : Real) : Complex := (a * Complex.I) / (1 + Complex.I)

theorem find_a (a : Real) (h : Complex.im (z a) = -1) : a = -2 :=
by
  sorry

end find_a_l81_81022


namespace answered_neither_question_correctly_l81_81833

variable (A B : Type)
variables [ProbabilitySpace A] [ProbabilitySpace B]

-- The conditions
lemma answered_first_question_correctly (P_A : ℝ) : P_A = 0.75 := by
  sorry

lemma answered_second_question_correctly (P_B : ℝ) : P_B = 0.7 := by
  sorry

lemma answered_both_questions_correctly (P_AB : ℝ) : P_AB = 0.65 := by
  sorry

-- The statement to prove
theorem answered_neither_question_correctly 
  (P_A : ℝ) (P_B : ℝ) (P_AB : ℝ) (P_neither : ℝ) :
  P_A = 0.75 → P_B = 0.7 → P_AB = 0.65 → P_neither = 1 - (P_A + P_B - P_AB) := by
  sorry

end answered_neither_question_correctly_l81_81833


namespace area_of_sine_triangle_l81_81341

-- We define the problem conditions and the statement we want to prove
theorem area_of_sine_triangle (A B C : Real) (area_ABC : ℝ) (unit_circle : ℝ) :
  unit_circle = 1 → area_ABC = 1 / 2 →
  let a := 2 * Real.sin A
  let b := 2 * Real.sin B
  let c := 2 * Real.sin C
  let s := (a + b + c) / 2
  let area_sine_triangle := 
    (s * (s - a) * (s - b) * (s - c)).sqrt / 4 
  area_sine_triangle = 1 / 8 :=
by
  intros
  sorry -- Proof is left as an exercise

end area_of_sine_triangle_l81_81341


namespace license_plate_count_l81_81366

-- Define the various components and conditions of the problem
def numLetters : ℕ := 26
def numDigits : ℕ := 10

-- Different middle characters can either be both letters or both digits
def middleChoices : ℕ := numLetters + numDigits

-- Define the number of possible license plates given the conditions
theorem license_plate_count :
  (numLetters * numDigits * middleChoices) = 9360 := by
  -- This simply states the mathematical formula found in the solution
  calc
    numLetters * numDigits * middleChoices
      = 26 * 10 * 36 : by rw [numLetters, numDigits, middleChoices]
  ... = 9360 : by norm_num

end license_plate_count_l81_81366


namespace trajectory_equation_circle_equation_l81_81398

-- Define the variables
variables {x y r : ℝ}

-- Prove the trajectory equation of the circle center P
theorem trajectory_equation (h1 : x^2 + r^2 = 2) (h2 : y^2 + r^2 = 3) : y^2 - x^2 = 1 :=
sorry

-- Prove the equation of the circle P given the distance to the line y = x
theorem circle_equation (h : (|x - y| / Real.sqrt 2) = (Real.sqrt 2) / 2) : 
  (x = y + 1 ∨ x = y - 1) → 
  ((y + 1)^2 + x^2 = 3 ∨ (y - 1)^2 + x^2 = 3) :=
sorry

end trajectory_equation_circle_equation_l81_81398


namespace painting_area_l81_81109

theorem painting_area
  (wall_height : ℝ) (wall_length : ℝ)
  (window_height : ℝ) (window_length : ℝ)
  (door_height : ℝ) (door_length : ℝ)
  (cond1 : wall_height = 10) (cond2 : wall_length = 15)
  (cond3 : window_height = 3) (cond4 : window_length = 5)
  (cond5 : door_height = 2) (cond6 : door_length = 7) :
  wall_height * wall_length - window_height * window_length - door_height * door_length = 121 := 
by
  simp [cond1, cond2, cond3, cond4, cond5, cond6]
  sorry

end painting_area_l81_81109


namespace point_M_coordinates_l81_81099

theorem point_M_coordinates :
  ∃ M : ℝ × ℝ × ℝ, 
    M.1 = 0 ∧ M.2.1 = 0 ∧  
    (dist (1, 0, 2) (M.1, M.2.1, M.2.2) = dist (1, -3, 1) (M.1, M.2.1, M.2.2)) ∧ 
    M = (0, 0, -3) :=
by
  sorry

end point_M_coordinates_l81_81099


namespace diplomats_not_speaking_russian_l81_81937

-- Definitions to formalize the problem
def total_diplomats : ℕ := 150
def speak_french : ℕ := 17
def speak_both_french_and_russian : ℕ := (10 * total_diplomats) / 100
def speak_neither_french_nor_russian : ℕ := (20 * total_diplomats) / 100

-- Theorem to prove the desired quantity
theorem diplomats_not_speaking_russian : 
  speak_neither_french_nor_russian + (speak_french - speak_both_french_and_russian) = 32 := by
  sorry

end diplomats_not_speaking_russian_l81_81937


namespace complement_U_A_l81_81361

open Set

def U : Set ℤ := univ
def A : Set ℤ := { x | x^2 - x - 2 ≥ 0 }

theorem complement_U_A :
  (U \ A) = { 0, 1 } := by
  sorry

end complement_U_A_l81_81361


namespace line_relationships_l81_81316

/-- Given a point P(a, b) inside the circle x^2 + y^2 = r^2 such that ab ≠ 0,
let line m be the chord passing through P and line l be given by the equation ax + by = r^2.
Prove that line m is parallel to line l and line l does not intersect the circle. -/
theorem line_relationships
  (a b r : ℝ)
  (h_nonzero : a * b ≠ 0)
  (h_inside : a^2 + b^2 < r^2) :
  let P := (a, b),
      circle_eq := λ (x y : ℝ), x^2 + y^2 = r^2,
      line_l := λ (x y : ℝ), a * x + b * y = r^2
  in (∃ m_slope l_slope : ℝ, m_slope = l_slope) ∧ (∀ O : ℝ × ℝ, O = (0,0) → ¬(∃ x y : ℝ, circle_eq x y ∧ line_l x y)) := 
sorry

end line_relationships_l81_81316


namespace max_distance_l81_81890

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l81_81890


namespace total_action_figures_l81_81055

-- Definitions based on conditions
def initial_figures : ℕ := 8
def figures_per_set : ℕ := 5
def added_sets : ℕ := 2
def total_added_figures : ℕ := added_sets * figures_per_set
def total_figures : ℕ := initial_figures + total_added_figures

-- Theorem statement with conditions and expected result
theorem total_action_figures : total_figures = 18 := by
  sorry

end total_action_figures_l81_81055


namespace probability_factor_of_5_factorial_l81_81696

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81696


namespace solve_for_k_l81_81116

open Real

theorem solve_for_k (x k : ℝ) (b c : ℝ) (h₁ : b = x^k) (h₂ : c = x^(1/k))
  (h₃ : 3 * (log b x) ^ 2 + 5 * (log c x) ^ 2 = (12 * (log x) ^ 2 / (log b * log c))) :
  k = sqrt((6 + sqrt 21) / 5) ∨ k = sqrt((6 - sqrt 21) / 5) := by
  sorry

end solve_for_k_l81_81116


namespace find_angle_BAC_l81_81386

variable (A B C : Type) [InnerProductSpace ℝ A] (P Q R : A)
variable (h1 : dist P Q = 2)
variable (h2 : dist P R = 3)
variable (h3 : inner (Q - P) (R - P) < 0)
variable (h4 : 1 / 2 * ∥Q - P∥ * ∥R - P∥ * Real.sin (Real.angle (Q - P) (R - P)) = 3 / 2)

theorem find_angle_BAC : Real.angle (Q - P) (R - P) = (5 / 6 : ℝ) * Real.pi :=
sorry

end find_angle_BAC_l81_81386


namespace lizas_final_balance_l81_81089

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end lizas_final_balance_l81_81089


namespace closest_point_is_correct_l81_81764

noncomputable def closest_point_on_plane (p : ℝ × ℝ × ℝ) : Prop :=
  let plane := (3 * p.1 - p.2 + 2 * p.3 = 18) in
  let origin := (0, 0, 0 : ℝ × ℝ × ℝ) in
  let distance := λ (p1 p2 : ℝ × ℝ × ℝ), real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2) in
  ∀ q, 3 * q.1 - q.2 + 2 * q.3 = 18 → distance p origin ≤ distance q origin

theorem closest_point_is_correct :
  closest_point_on_plane (27 / 7, -9 / 7, 18 / 7) :=
by
  sorry

end closest_point_is_correct_l81_81764


namespace max_distance_on_ellipse_l81_81896

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l81_81896


namespace sin_cos_15_eq_quarter_l81_81207

theorem sin_cos_15_eq_quarter :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
by 
  sorry

end sin_cos_15_eq_quarter_l81_81207


namespace maximum_triangle_area_l81_81344

theorem maximum_triangle_area (l : ℝ → ℝ) (h_line_not_through_A : ¬ ∃ x, l x = 1)
  (h_intersect1 : ∃ x, (x^2 / 3) + (l x)^2 = 1) (h_intersect2 : ∃ x, (x^2 / 3) + (l x)^2 = 1)
  (h_perpendicular : ∀ (P Q : ℝ × ℝ), (P snd - 1) * (Q snd - 1) = - (P fst) * (Q fst)) :
  ∃ P Q : ℝ × ℝ, (P snd = l P fst) ∧ (Q snd = l Q fst) ∧ 
  (P ≠ Q) ∧ 
  (P ≠ (0, 1)) ∧ 
  (Q ≠ (0, 1)) ∧ 
  2 * (P.fst * Q.snd - P.snd * Q.fst) = 9 / 4 :=
sorry

end maximum_triangle_area_l81_81344


namespace total_expenditure_l81_81706

-- Define the conditions.
def singers : ℕ := 30
def current_robes : ℕ := 12
def robe_cost : ℕ := 2

-- Define the statement.
theorem total_expenditure (singers current_robes robe_cost : ℕ) : 
  (singers - current_robes) * robe_cost = 36 := by
  sorry

end total_expenditure_l81_81706


namespace nancy_apples_l81_81081

/-- Define the total number of apples picked by Mike, Keith, and the total number -/
variable (apples_picked_by_mike : ℕ)
variable (apples_picked_by_keith : ℕ)
variable (total_apples_picked : ℕ)

/-- State the problem as a theorem that proves Nancy picked 3 apples -/
theorem nancy_apples (apples_picked_by_mike = 7) 
                    (apples_picked_by_keith = 6)
                    (total_apples_picked = 16) :
                    (total_apples_picked - apples_picked_by_mike - apples_picked_by_keith = 3) := 
by
  sorry

end nancy_apples_l81_81081


namespace triangle_side_length_l81_81762

theorem triangle_side_length (a b p : ℝ) (H_perimeter : a + b + 10 = p) (H_a : a = 7) (H_b : b = 15) (H_p : p = 32) : 10 = 10 :=
by
  sorry

end triangle_side_length_l81_81762


namespace simplify_cbrt_8000_eq_21_l81_81182

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l81_81182


namespace Sally_fries_total_l81_81105

theorem Sally_fries_total 
  (sally_fries_initial : ℕ)
  (mark_fries_initial : ℕ)
  (fries_given_by_mark : ℕ)
  (one_third_of_mark_fries : mark_fries_initial = 36 → fries_given_by_mark = mark_fries_initial / 3) :
  sally_fries_initial = 14 → mark_fries_initial = 36 → fries_given_by_mark = 12 →
  let sally_fries_final := sally_fries_initial + fries_given_by_mark
  in sally_fries_final = 26 := 
by
  intros h1 h2 h3
  unfold sally_fries_final
  rw [h1, h3]
  exact rfl

end Sally_fries_total_l81_81105


namespace algebraic_identity_l81_81176

theorem algebraic_identity (a b : ℕ) (h1 : a = 753) (h2 : b = 247)
  (identity : ∀ a b, (a^2 + b^2 - a * b) / (a^3 + b^3) = 1 / (a + b)) : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 0.001 := 
by
  sorry

end algebraic_identity_l81_81176


namespace remainder_when_nm_div_61_l81_81842

theorem remainder_when_nm_div_61 (n m : ℕ) (k j : ℤ):
  n = 157 * k + 53 → m = 193 * j + 76 → (n + m) % 61 = 7 := by
  intros h1 h2
  sorry

end remainder_when_nm_div_61_l81_81842


namespace evaluate_expression_is_41_l81_81274

noncomputable def evaluate_expression : ℚ :=
  (121 * (1 / 13 - 1 / 17) + 169 * (1 / 17 - 1 / 11) + 289 * (1 / 11 - 1 / 13)) /
  (11 * (1 / 13 - 1 / 17) + 13 * (1 / 17 - 1 / 11) + 17 * (1 / 11 - 1 / 13))

theorem evaluate_expression_is_41 : evaluate_expression = 41 := 
by
  sorry

end evaluate_expression_is_41_l81_81274


namespace exists_partition_generating_sigma_algebra_l81_81065

variable (Ω : Type) [Countable Ω]
variable (𝓕 : Set (Set Ω)) [sigma_algebra Ω 𝓕]

theorem exists_partition_generating_sigma_algebra :
  ∃ (D : ℕ → Set Ω), (∀ i j, i ≠ j → D i ∩ D j = ∅) ∧ (⋃ n, D n = Set.univ) ∧ (𝓕 = { S | ∃ N : Set ℕ, S = ⋃ n ∈ N, D n }) :=
sorry

end exists_partition_generating_sigma_algebra_l81_81065


namespace work_completion_days_approx_l81_81933

noncomputable def work_rate_mary : ℝ := 1 / 11
noncomputable def work_rate_rosy : ℝ := 1.10 * work_rate_mary
noncomputable def work_rate_john : ℝ := 0.75 * work_rate_mary
noncomputable def work_rate_alex : ℝ := 1.40 * work_rate_rosy

noncomputable def combined_work_rate : ℝ := work_rate_mary + work_rate_rosy + work_rate_john + work_rate_alex
noncomputable def days_to_complete_work : ℝ := 1 / combined_work_rate

theorem work_completion_days_approx :
  days_to_complete_work ≈ 2.51 := sorry

end work_completion_days_approx_l81_81933


namespace decreasing_intervals_of_reciprocal_l81_81008

theorem decreasing_intervals_of_reciprocal :
  ∀ x : ℝ, x ≠ 0 → (deriv (λ x, 1/x) x < 0) :=
by
  intro x hx
  have deriv_y : deriv (λ x, 1/x) x = -1 / (x ^ 2) := sorry
  rw deriv_y
  have x_sq_pos : x^2 > 0 := by
    apply pow_pos
    exact ne_of_gt (lt_of_le_of_ne (le_of_lt (lt_or_gt_of_ne hx)) hx.symm)
  linarith

end decreasing_intervals_of_reciprocal_l81_81008


namespace possible_orderings_l81_81856

noncomputable def total_orderings : Nat :=
  ∑ k in Finset.range 6, Nat.choose 5 k * (k + 1) * (k + 2)

theorem possible_orderings : total_orderings = 552 := by
  sorry

end possible_orderings_l81_81856


namespace proof_problem_l81_81731

noncomputable def problem_expr : ℝ :=
  (1 / 8) ^ (- 2 / 3) + real.logb 3 6 + real.logb 3 (9 / 2) - 10 ^ (1 + real.log 1/2)

theorem proof_problem : problem_expr = 2 :=
by
  sorry

end proof_problem_l81_81731


namespace connie_remaining_marbles_l81_81258

def initial_marbles : ℕ := 73
def marbles_given : ℕ := 70

theorem connie_remaining_marbles : initial_marbles - marbles_given = 3 := by
  sorry

end connie_remaining_marbles_l81_81258


namespace Margo_paired_with_Irma_probability_l81_81516

noncomputable def probability_Margo_paired_with_Irma : ℚ :=
  1 / 29

theorem Margo_paired_with_Irma_probability :
  let total_students := 30
  let number_of_pairings := total_students - 1
  probability_Margo_paired_with_Irma = 1 / number_of_pairings := 
by
  sorry

end Margo_paired_with_Irma_probability_l81_81516


namespace jo_blair_30th_term_l81_81056

-- Definitions based on the conditions
def jo_blair_sequence : ℕ → ℕ
| 0     := 1
| (n+1) := if n % 2 = 0 then jo_blair_sequence n + 2 else jo_blair_sequence n + 1

-- Main theorem statement
theorem jo_blair_30th_term : jo_blair_sequence 29 = 88 :=
by
  sorry

end jo_blair_30th_term_l81_81056


namespace sin_negative_angle_sin_pi_between_sin_special_angle_sin_negative_seven_pi_over_six_l81_81750

theorem sin_negative_angle : 
  ∀ (x : ℝ), sin (-x) = -sin x := by sorry

theorem sin_pi_between : 
  sin (π + π / 6) = -sin (π / 6) := by sorry

theorem sin_special_angle : 
  sin (π / 6) = 1 / 2 := by sorry

theorem sin_negative_seven_pi_over_six :
  sin (-7 * π / 6) = 1 / 2 :=
  by have h1 := sin_negative_angle (7 * π / 6)
     have h2 := sin_pi_between
     have h3 := sin_special_angle
     rw [h1, h2, h3]
     sorry

end sin_negative_angle_sin_pi_between_sin_special_angle_sin_negative_seven_pi_over_six_l81_81750


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81639

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81639


namespace problem_statement_l81_81187

-- Define the figure Φ as the set of points (x, y) satisfying the given inequalities
def figure_Φ (x y : ℝ) : Prop :=
  x^2 - y^2 ≤ 2 * (x - y) ∧ x^2 + y^2 ≤ 4 * (x + y - 1)

-- Define the area of figure Φ
def area_figure_Φ : ℝ := 2 * real.pi

-- Define the distance from point T(0, 4) to the nearest point of the figure Φ
def distance_T_to_figure_Φ : ℝ := 2 * real.sqrt 2 - 2

-- The main statement to prove
theorem problem_statement :
  (∃ S : ℝ, S = area_figure_Φ ∧ S = 2 * real.pi) ∧ 
  (∃ ρ : ℝ, ρ = distance_T_to_figure_Φ ∧ ρ = 2 * real.sqrt 2 - 2) :=
by { sorry }

end problem_statement_l81_81187


namespace range_of_a_l81_81793

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1 / 2) * Real.log x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := (2 * a * x^2 + 1) / (2 * x)

def p (a : ℝ) : Prop := ∀ x, 1 ≤ x → f_prime (a) (x) ≤ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1

theorem range_of_a (a : ℝ) : (p a ∧ q a) → -1 < a ∧ a ≤ -1 / 2 :=
by
  sorry

end range_of_a_l81_81793


namespace min_value_of_f_range_of_a_l81_81068

def f (x : ℝ) : ℝ := 2 * |x - 2| - x + 5

theorem min_value_of_f : ∃ (m : ℝ), m = 3 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use 3
  sorry

theorem range_of_a (a : ℝ) : (|a + 2| ≥ 3 ↔ a ≤ -5 ∨ a ≥ 1) :=
sorry

end min_value_of_f_range_of_a_l81_81068


namespace parabola_focal_distance_hyperbola_equation_point_plane_distance_ellipse_extreme_value_l81_81560

open Real

-- Problem 1
theorem parabola_focal_distance :
  let P := parabola_focus (λ x, 4 * x^2 = y) in
  P = (0, 1/16) →
  parabola_focal_distance (λ x, 4 * x^2 = y) = 1 / 8 :=
begin
  sorry
end

-- Problem 2
theorem hyperbola_equation :
  let H1 := hyperbola_has_same_asymptotes (λ x y, x^2 / 2 - y^2 = 1) (2,0) in
  H1 = (λ x y, x^2 / 4 - y^2 / 2 = 1) :=
begin
  sorry
end

-- Problem 3
theorem point_plane_distance :
  let D := distance_to_plane (0, 1, 3) (λ x y z, x + 2 * y + 3 * z + 3 = 0) in
  D = sqrt 14 :=
begin
  sorry
end

-- Problem 4
theorem ellipse_extreme_value :
  let M := max_distance_sum (1, 1) (lower_focus_of_ellipse (λ x y, 5 * y^2 + 9 * x^2 = 45)) in
  let N := min_distance_sum (1, 1) (lower_focus_of_ellipse (λ x y, 5 * y^2 + 9 * x^2 = 45)) in
  M - N = 2 * sqrt 2 :=
begin
  sorry
end

end parabola_focal_distance_hyperbola_equation_point_plane_distance_ellipse_extreme_value_l81_81560


namespace score_recording_l81_81378

theorem score_recording (avg : ℤ) (h : avg = 0) : 
  (9 = avg + 9) ∧ (-18 = avg - 18) ∧ (-2 = avg - 2) :=
by
  -- Proof steps go here
  sorry

end score_recording_l81_81378


namespace factor_probability_l81_81590

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81590


namespace cuboid_inequality_l81_81471

theorem cuboid_inequality 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end cuboid_inequality_l81_81471


namespace hexagon_coverage_is_50_percent_l81_81499

def equilateral_triangle_area (a : ℝ) : ℝ := 
  (sqrt 3 / 4) * a^2

def hexagon_area (a : ℝ) : ℝ := 
  6 * equilateral_triangle_area a

def covered_area (a : ℝ) : ℝ := 
  3 * equilateral_triangle_area a

def fraction_covered (a : ℝ) : ℝ := 
  covered_area a / hexagon_area a

def percent_covered (a : ℝ) : ℝ := 
  fraction_covered a * 100

theorem hexagon_coverage_is_50_percent (a : ℝ) (h : a > 0) : 
  percent_covered a = 50 :=
by
  sorry

end hexagon_coverage_is_50_percent_l81_81499


namespace probability_product_greater_than_zero_l81_81170

open Set Finset

def chosen_integers := {-3, -6, 5, 2, -1}

def different_integers_chosen (x y : ℤ) : Prop := 
  x ∈ chosen_integers ∧ y ∈ chosen_integers ∧ x ≠ y

def product_greater_than_zero (x y : ℤ) : Prop := 
  x * y > 0

theorem probability_product_greater_than_zero : 
  ∃ (total_pairs favorable_pairs : ℕ), 
  total_pairs = Nat.choose (chosen_integers.toFinset.card) 2 ∧
  favorable_pairs = 4 ∧ 
  (favorable_pairs : ℚ) / total_pairs = 2 / 5 :=
by
  sorry

end probability_product_greater_than_zero_l81_81170


namespace acute_angle_sum_l81_81376

variable (α β : ℝ)

def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2
def tan_sum_satisfies (α β : ℝ) : Prop := (1 + sqrt 3 * tan α) * (1 + sqrt 3 * tan β) = 4

theorem acute_angle_sum (hα : is_acute α) (hβ : is_acute β) (h : tan_sum_satisfies α β) : α + β = π / 3 :=
sorry

end acute_angle_sum_l81_81376


namespace tetrahedron_face_equal_iff_conditions_l81_81943

-- Define the conditions for proving tetrahedron equilateral
structure Tetrahedron :=
  (vertices : Fin 4 → Point)
  (edges_eq : ∀ i j, ¬(i = j) → length (vertices i - vertices j) = length (vertices (σ i) - vertices (σ j))) 

def vertices_of_tetrahedron_face_equal (T : Tetrahedron) : Prop := 
  ∀ i j, ¬(i = j) → length (vertices T i - vertices T j) = length (vertices T i - vertices T j)

theorem tetrahedron_face_equal_iff_conditions (T : Tetrahedron) :
  (∃ v : Fin 4, (∑ i : Fin 4, plane_angle i v = π) ∧ (∃ u : Fin 4 → Boolean, ∀ i j, u i = u j)) ∨
  (∃ A B, A = B) ∨
  (∃ r : ℝ, (∀ i, circumscribed_circle_radius (face T i) = r)) ∨
  (∃ G:Point, ∀ j : Fin 4, distance G (vertices T j) = distance G (centroid T))
  ↔ vertices_of_tetrahedron_face_equal T :=
by
  sorry

end tetrahedron_face_equal_iff_conditions_l81_81943


namespace river_flow_rate_l81_81701

/--
Given:
- The depth of the river is 2 meters.
- The width of the river is 45 meters.
- The volume flow rate of the river is 3000 cubic meters per minute.

Prove that the rate at which the river is flowing is 2 kilometers per hour.
-/
theorem river_flow_rate (depth width flow_rate : ℝ) 
    (h_depth : depth = 2) 
    (h_width : width = 45) 
    (h_flow_rate : flow_rate = 3000) : 
    let A := depth * width in
    let Q := flow_rate / 60 in
    let V := Q / A in
    (V * 3.6) = 2 := 
by 
  sorry

end river_flow_rate_l81_81701


namespace max_distance_on_ellipse_l81_81895

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l81_81895


namespace triangle_medians_and_bisectors_l81_81753

theorem triangle_medians_and_bisectors
  {A B C A1 A2 C' B' : Point}
  (h_median : midpoint A1 B C)
  (h_angle_bisector : angle_bisector A A2 B C)
  (h_circle : circle_through A A1 A2 intersects_side AB C' AC B') :
  segment_length B C' = segment_length B' C :=
sorry

end triangle_medians_and_bisectors_l81_81753


namespace convex_quadrilateral_intersections_l81_81033

theorem convex_quadrilateral_intersections
  (A B C D M H L H' L' : Point)
  (quadrilateral_ABCD : ConvexQuadrilateral A B C D)
  (extension_intersection : SameLine A B M ∧ SameLine C D M)
  (perpendicular_AD_BC : PerpendicularLine A D B C)
  (line_through_M_HL : ∃ line : Line, OnLine M line ∧ OnLine H line ∧ OnLine L line)
  (line_through_M_HL_prime : ∃ line' : Line, OnLine M line' ∧ OnLine H' line' ∧ OnLine L' line') :
  1 / (distance M H) + 1 / (distance M L) = 1 / (distance M H') + 1 / (distance M L') := 
sorry

end convex_quadrilateral_intersections_l81_81033


namespace smallest_area_of_square_containing_rectangles_l81_81568

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end smallest_area_of_square_containing_rectangles_l81_81568


namespace first_player_wins_l81_81154

noncomputable def game_win_guarantee : Prop :=
  ∃ (first_can_guarantee_win : Bool),
    first_can_guarantee_win = true

theorem first_player_wins :
  ∀ (nuts : ℕ) (players : (ℕ × ℕ)) (move : ℕ → ℕ) (end_condition : ℕ → Prop),
    nuts = 10 →
    players = (1, 2) →
    (∀ n, 0 < n ∧ n ≤ nuts → move n = n - 1) →
    (end_condition 3 = true) →
    (∀ x y z, x + y + z = 3 ↔ end_condition (x + y + z)) → 
    game_win_guarantee :=
by
  intros nuts players move end_condition H1 H2 H3 H4 H5
  sorry

end first_player_wins_l81_81154


namespace sum_of_ti_l81_81261

noncomputable def f (x : ℝ) : ℝ := if x < 0 then 2 - f (-x) else sorry

theorem sum_of_ti (ω : ℝ) (m : ℕ) (x y : ℕ → ℝ) 
  (hx : ∀ i, i < m → y i = sin (ω * x i) + 1)
  (hf : ∀ x, f (-x) = 2 - f x)
  (hi : ∀ i, i < m → (f (x i) = y i ∨ f (-x i) = 2 - y i))
  : ∑ i in Finset.range m, (x i + y i) = m := 
by
  sorry

end sum_of_ti_l81_81261


namespace probability_factor_of_5_factorial_l81_81690

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81690


namespace second_percentage_increase_l81_81977

theorem second_percentage_increase (P : ℝ) (x : ℝ) :
  1.25 * P * (1 + x / 100) = 1.625 * P ↔ x = 30 :=
by
  sorry

end second_percentage_increase_l81_81977


namespace perimeter_div_b_eq_simplified_radical_l81_81231

theorem perimeter_div_b_eq_simplified_radical (b : ℝ) (hb : 0 < b)
    (y_eq_2x : ∀ (x y : ℝ), y = 2 * x ↔ (x = y / 2)) :
  let p := 4 + Real.sqrt 13 + Real.sqrt 5 in
  ∃ P : ℝ, 
  (∃ (x1 x2 y1 y2 : ℝ), 
    (x1, y1) = (-b, -2 * b) ∧
    (x2, y2) = (b, 2 * b)) ∧
  let side1 := 2 * b in
  let side2 := 2 * b in
  let diag1 := Real.sqrt ((2 * b)^2 + (3 * b)^2) in
  let diag2 := Real.sqrt ((2 * b)^2 + b^2) in
  P = side1 + side2 + diag1 + diag2 ∧
  P / b = p :=
  let side1 := 2 * b in
  let side2 := 2 * b in
  let diag1 := Real.sqrt ((2 * b)^2 + (3 * b)^2) in
  let diag2 := Real.sqrt ((2 * b)^2 + b^2) in
  P = side1 + side2 + diag1 + diag2 ∧
  P / b = 4 + Real.sqrt 13 + Real.sqrt 5 :=
sorry

end perimeter_div_b_eq_simplified_radical_l81_81231


namespace dave_winfield_home_runs_l81_81239

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end dave_winfield_home_runs_l81_81239


namespace primes_and_six_divisibility_l81_81415

open Int

theorem primes_and_six_divisibility (n : ℕ) (p : Fin n → ℕ)
  (h1 : ∀ i, Prime (p i) ∧ 5 < p i)
  (h2 : 6 ∣ ∑ i, (p i)^2) :
  6 ∣ n :=
sorry

end primes_and_six_divisibility_l81_81415


namespace min_value_f_part1_range_of_a_part2_extreme_points_part3_l81_81929

-- Part (I)
theorem min_value_f_part1 (a : ℝ) (h_a : a = 0) : 
  ∀ x ∈ set.Icc (1 : ℝ) (Real.exp 1), Real.log (x + (x - a)^2) ≥ 1 :=
sorry

-- Part (II)
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ set.Icc (0.5 : ℝ) 2, 
  (2 * x^2 - 2 * a * x + 1) > 0 ) → a < 9 / 4 :=
sorry

-- Part (III)
theorem extreme_points_part3 (a : ℝ) :
  (a ≤ Real.sqrt 2 → (∀ x > 0, (Real.log (x + (x - a)^2)).deriv x > 0)) ∧
  (a > Real.sqrt 2 → 
    (∃ x1 x2,
      (x1 = (a - Real.sqrt (a^2 - 2))/2 ∧ x2 = (a + Real.sqrt (a^2 - 2))/2 ) ∧
        (Real.log (x1 + (x1 - a)^2) = Real.log (x2 + (x2 - a)^2).deriv x = 0))) :=
sorry

end min_value_f_part1_range_of_a_part2_extreme_points_part3_l81_81929


namespace older_brother_pocket_money_l81_81451

-- Definitions of the conditions
axiom sum_of_pocket_money (O Y : ℕ) : O + Y = 12000
axiom older_brother_more (O Y : ℕ) : O = Y + 1000

-- The statement to prove
theorem older_brother_pocket_money (O Y : ℕ) (h1 : O + Y = 12000) (h2 : O = Y + 1000) : O = 6500 :=
by
  exact sorry  -- Placeholder for the proof

end older_brother_pocket_money_l81_81451


namespace sum_of_reflected_coordinates_l81_81461

noncomputable def sum_of_coordinates (C D : ℝ × ℝ) : ℝ :=
  C.1 + C.2 + D.1 + D.2

theorem sum_of_reflected_coordinates (y : ℝ) :
  let C := (3, y)
  let D := (3, -y)
  sum_of_coordinates C D = 6 :=
by
  sorry

end sum_of_reflected_coordinates_l81_81461


namespace mika_stickers_l81_81935

theorem mika_stickers 
    (initial_stickers : ℝ := 20.5)
    (bought_stickers : ℝ := 26.25)
    (birthday_stickers : ℝ := 19.75)
    (friend_stickers : ℝ := 7.5)
    (sister_stickers : ℝ := 6.3)
    (greeting_card_stickers : ℝ := 58.5)
    (yard_sale_stickers : ℝ := 3.2) :
    initial_stickers + bought_stickers + birthday_stickers + friend_stickers
    - sister_stickers - greeting_card_stickers - yard_sale_stickers = 6 := 
by
    sorry

end mika_stickers_l81_81935


namespace expectation_is_four_thirds_l81_81707

-- Define the probability function
def P_ξ (k : ℕ) : ℚ :=
  if k = 0 then (1/2)^2 * (2/3)
  else if k = 1 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3)
  else if k = 2 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3) + (1/2) * (1/2) * (1/3)
  else if k = 3 then (1/2) * (1/2) * (1/3)
  else 0

-- Define the expected value function
def E_ξ : ℚ :=
  0 * P_ξ 0 + 1 * P_ξ 1 + 2 * P_ξ 2 + 3 * P_ξ 3

-- Formal statement of the problem
theorem expectation_is_four_thirds : E_ξ = 4 / 3 :=
  sorry

end expectation_is_four_thirds_l81_81707


namespace smallest_integer_of_inequality_l81_81536

theorem smallest_integer_of_inequality :
  ∃ x : ℤ, (8 - 7 * x ≥ 4 * x - 3) ∧ (∀ y : ℤ, (8 - 7 * y ≥ 4 * y - 3) → y ≥ x) ∧ x = 1 :=
sorry

end smallest_integer_of_inequality_l81_81536


namespace simple_interest_time_l81_81506

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem simple_interest_time (SI CI : ℝ) (SI_given CI_given P_simp P_comp r_simp r_comp t_comp : ℝ) :
  SI = CI / 2 →
  CI = compound_interest P_comp r_comp 1 t_comp - P_comp →
  SI = simple_interest P_simp r_simp t_comp →
  P_simp = 1272 →
  r_simp = 0.10 →
  P_comp = 5000 →
  r_comp = 0.12 →
  t_comp = 2 →
  t_comp = 5 :=
by
  intros
  sorry

end simple_interest_time_l81_81506


namespace mr_green_expected_yield_l81_81448

theorem mr_green_expected_yield :
  ∀ (steps_length steps_width foot_per_step yield_rate yield_increase : ℝ),
    steps_length = 18 →
    steps_width = 25 →
    foot_per_step = 3 →
    yield_rate = 0.5 →
    yield_increase = 0.1 →
    let length_in_feet := steps_length * foot_per_step,
        width_in_feet := steps_width * foot_per_step,
        area := length_in_feet * width_in_feet,
        initial_yield := area * yield_rate,
        final_yield := initial_yield * (1 + yield_increase)
    in final_yield = 2227.5 :=
begin
  intros steps_length steps_width foot_per_step yield_rate yield_increase,
  intros h1 h2 h3 h4 h5,
  let length_in_feet := steps_length * foot_per_step,
  let width_in_feet := steps_width * foot_per_step,
  let area := length_in_feet * width_in_feet,
  let initial_yield := area * yield_rate,
  let final_yield := initial_yield * (1 + yield_increase),
  have h_length_feet : length_in_feet = 54, by { rw [h1, h3], norm_num },
  have h_width_feet : width_in_feet = 75, by { rw [h2, h3], norm_num },
  have h_area : area = 4050, by { rw [h_length_feet, h_width_feet], norm_num },
  have h_initial_yield : initial_yield = 2025, by { rw [h_area, h4], norm_num },
  have h_final_yield : final_yield = 2227.5, by { rw [h_initial_yield, h5], norm_num },
  exact h_final_yield,
end

end mr_green_expected_yield_l81_81448


namespace hyperbola_eccentricity_l81_81967

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 2) = 1

-- Define the semi-major axis a and semi-minor axis b for the hyperbola
def a : ℝ := 1
def b : ℝ := Real.sqrt 2

-- Calculate c using a and b
def c : ℝ := Real.sqrt (a^2 + b^2)

-- Eccentricity e
def e : ℝ := c / a

-- Prove that the eccentricity of the given hyperbola is sqrt(3)
theorem hyperbola_eccentricity : e = Real.sqrt 3 :=
by
  -- Provide the proof later (if required)
  sorry

end hyperbola_eccentricity_l81_81967


namespace well_performing_student_net_pay_l81_81196

def base_salary : ℤ := 25000
def bonus : ℤ := 5000
def tax_rate : ℝ := 0.13

def total_earnings (base_salary bonus : ℤ) : ℤ :=
  base_salary + bonus

def income_tax (total_earnings : ℤ) (tax_rate : ℝ) : ℤ :=
  total_earnings * (Real.toRat tax_rate)

def net_pay (total_earnings income_tax: ℤ) : ℤ :=
  total_earnings - income_tax

theorem well_performing_student_net_pay :
  net_pay (total_earnings base_salary bonus) (income_tax (total_earnings base_salary bonus) tax_rate) = 26100 := by
  sorry

end well_performing_student_net_pay_l81_81196


namespace vector_odot_not_symmetric_l81_81262

-- Define the vector operation ⊛
def vector_odot (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

-- Statement: Prove that the operation is not symmetric
theorem vector_odot_not_symmetric (a b : ℝ × ℝ) : vector_odot a b ≠ vector_odot b a := by
  sorry

end vector_odot_not_symmetric_l81_81262


namespace range_f_eq_l81_81765

noncomputable def f (x : ℝ) : ℝ := (sin x)^3 + 2 * (sin x)^2 - 4 * sin x + 3 * cos x + 3 * (cos x)^2 - 2 / (sin x - 1)

theorem range_f_eq :
  let f := λ x : ℝ, (sin x)^3 + 2 * (sin x)^2 - 4 * sin x + 3 * cos x + 3 * (cos x)^2 - 2 / (sin x - 1) in
  {y | ∃ x : ℝ, sin x ≠ 1 ∧ y = f x} = set.Icc 1 (1 + 3 * sqrt 2) :=
by
  sorry

end range_f_eq_l81_81765


namespace minimum_sum_of_denominators_eq_36_l81_81861

theorem minimum_sum_of_denominators_eq_36 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ (1 / x.to_nat + 4 / y.to_nat + 9 / z.to_nat = 1) ∧ (x + y + z = 36) :=
sorry

end minimum_sum_of_denominators_eq_36_l81_81861


namespace initially_marked_points_l81_81953

theorem initially_marked_points (k : ℕ) (h : 4 * k - 3 = 101) : k = 26 :=
by
  sorry

end initially_marked_points_l81_81953


namespace first_player_wins_l81_81157

theorem first_player_wins : 
  ∃ (strategy: ℕ → bool), -- strategy is a function that decides which pile to take from
  ∀ (turn: ℕ), -- turn is the number of turns taken
  let remaining_nuts := 10 - turn in
  if remaining_nuts = 3 then -- when 3 nuts are left
    ∀ (pile1 pile2 pile3 : ℕ), pile1 + pile2 + pile3 = 3 -> -- three piles sum up to 3
    ¬ (pile1 = 1 ∧ pile2 = 1 ∧ pile3 = 1) -- these should not be in three separate piles 
  else
    (turn % 2 = 0 → strategy turn = true) ∧ -- first player's turns follow the strategy
    (turn % 2 = 1 → strategy turn = false)  -- second player's turns follow the strategy
:= sorry

end first_player_wins_l81_81157


namespace Simson_line_intersection_with_nine_point_circle_l81_81418

theorem Simson_line_intersection_with_nine_point_circle
  (ABC : Triangle)
  (H : Point) (F : Point)
  (on_circumcircle : onCircumcircle F ABC)
  (H_is_orthocenter : isOrthocenter H ABC)
  (simson_line : Line)
  (simson_line_def : isSimsonLine F ABC simson_line)
  (nine_point_circle : Circle)
  (nine_point_circle_def : isNinePointCircle nine_point_circle ABC)
  (M : Point)
  (M_def : midpoint M F H)
  (M_on_nine_point_circle : onCircle M nine_point_circle) :
  ∃ I : Point, 
    (intersection_point FH nine_point_circle I) ∧ 
    (onLine I simson_line) := 
sorry

end Simson_line_intersection_with_nine_point_circle_l81_81418


namespace radical_product_is_64_l81_81528

theorem radical_product_is_64:
  real.sqrt (16:ℝ) * real.sqrt (real.sqrt 256) * real.n_root 64 3 = 64 :=
sorry

end radical_product_is_64_l81_81528


namespace number_of_ways_to_spend_1000_euros_l81_81698

theorem number_of_ways_to_spend_1000_euros :
  (∃ x y : ℕ, 11 * x + 13 * y = 1000) → 
  (∀ x y : ℕ, (11 * x + 13 * y = 1000) → x ≥ 0 ∧ y ≥ 0) → 
  ∃ n : ℕ, n = 7 :=
begin
  sorry
end

end number_of_ways_to_spend_1000_euros_l81_81698


namespace probability_factor_of_120_l81_81642

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81642


namespace max_distance_on_ellipse_l81_81906

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l81_81906


namespace final_price_correct_l81_81940

noncomputable def price_cucumbers : ℝ := 5
noncomputable def price_tomatoes : ℝ := price_cucumbers - 0.20 * price_cucumbers
noncomputable def total_cost_before_discount : ℝ := 2 * price_tomatoes + 3 * price_cucumbers
noncomputable def discount : ℝ := 0.10 * total_cost_before_discount
noncomputable def final_price : ℝ := total_cost_before_discount - discount

theorem final_price_correct : final_price = 20.70 := by
  sorry

end final_price_correct_l81_81940


namespace ratio_DE_EF_l81_81404

-- Uses noncomputable theory as we are dealing with specific proportional segments and vectors
noncomputable theory

-- Define points in a triangle
variables (A B C D E F : Type)
-- Define the vectors representing the points
variables [vector_space ℝ (A → ℝ)]
variables [vector_space ℝ (B → ℝ)]
variables [vector_space ℝ (C → ℝ)]
variables [vector_space ℝ (D → ℝ)]
variables [vector_space ℝ (E → ℝ)]
variables [vector_space ℝ (F → ℝ)]

-- Assume some helper functions which describe the points in the segments
variables (a b c d e f : vector_space ℝ)
variables (AD_DB : ℝ) (BE_EC : ℝ)

-- Conditions given in the problem
axiom in_triangle : ∃ ABC : Type, A B C
axiom D_on_AB : ∃ AB : Type, D (AB) AD_DB = 2 / 5
axiom E_on_BC : ∃ BC : Type, E (BC) BE_EC = 2 / 5

-- The theorem to prove
theorem ratio_DE_EF :
  in_triangle A B C →
  D_on_AB D → 
  E_on_BC E → 
  (lines_intersect (D E) (A C) F) →
  (DE / EF) = (1 / 2) :=
begin
  sorry
end

end ratio_DE_EF_l81_81404


namespace geometric_sequence_sum_l81_81327

theorem geometric_sequence_sum (q a₁ : ℝ) (hq : q > 1) (h₁ : a₁ + a₁ * q^3 = 18) (h₂ : a₁^2 * q^3 = 32) :
  (a₁ * (1 - q^8) / (1 - q) = 510) :=
by
  sorry

end geometric_sequence_sum_l81_81327


namespace compute_a_l81_81319

theorem compute_a 
  (a b : ℚ) 
  (h : ∃ (x : ℝ), x^3 + (a : ℝ) * x^2 + (b : ℝ) * x - 37 = 0 ∧ x = 2 - 3 * Real.sqrt 3) : 
  a = -55 / 23 :=
by 
  sorry

end compute_a_l81_81319


namespace age_difference_28_l81_81995

variable (li_lin_age_father_sum li_lin_age_future father_age_future : ℕ)

theorem age_difference_28 
    (h1 : li_lin_age_father_sum = 50)
    (h2 : ∀ x, li_lin_age_future = x → father_age_future = 3 * x - 2)
    (h3 : li_lin_age_future + 4 = li_lin_age_father_sum + 8 - (father_age_future + 4))
    : li_lin_age_father_sum - li_lin_age_future = 28 :=
sorry

end age_difference_28_l81_81995


namespace altitude_le_geometric_mean_l81_81475

theorem altitude_le_geometric_mean {A B C : Type} [Trivial] 
  {a b c T s r m_c r_a r_b : ℝ}
  (h_eq_a : a = dist B C) (h_eq_b : b = dist A C)
  (h_eq_c : c = dist A B) 
  (h_alt : m_c = altitude C A B)
  (h_inradii_a : r_a = inradius B C A)
  (h_inradii_b : r_b = inradius A C B)
  (h_semiperimeter : s = (a + b + c) / 2) 
  (h_area : T = triangle_area A B C)
  (h_inradius : r = T / s)
  (h_alt_formula : m_c = 2 * T / c) :
  m_c ≤ sqrt (r_a * r_b) :=
begin
  sorry
end

end altitude_le_geometric_mean_l81_81475


namespace max_distance_B_P_l81_81903

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l81_81903


namespace perimeter_DEF_l81_81399

-- Definitions of the geometric conditions
variables (radius : ℝ) (X Y Z D E F : Point) (DEF : Triangle)
variables [Equilateral DEF] (circle1 circle2 circle3 : Circle)
variables [radius = 2] [Tangent circle1 circle2] [Tangent circle2 circle3]
variables [Tangent circle1 circle3] [Tangent circle1 DE]
variables [Tangent circle2 DE] [Tangent circle3 DE]
variables [Aligned X Y Z DE]
variables [CircleRadius circle1 = radius] [CircleCenter circle1 = X]
variables [CircleRadius circle2 = radius] [CircleCenter circle2 = Y]
variables [CircleRadius circle3 = radius] [CircleCenter circle3 = Z]

-- The theorem statement
theorem perimeter_DEF : perimeter DEF = 60 := 
sorry -- Proof is not required

end perimeter_DEF_l81_81399


namespace find_a_parallel_find_a_perpendicular_l81_81315

-- Define the lines l1 and l2 with their respective equations
def line_l1 (a : ℝ) : ℝ × ℝ → Prop :=
λ p, p.1 + a * p.2 - 2 * a - 2 = 0

def line_l2 (a : ℝ) : ℝ × ℝ → Prop :=
λ p, a * p.1 + p.2 - 1 - a = 0

-- Parallel condition
def is_parallel (a : ℝ) : Prop :=
-1 / a = -a → a = 1

-- Perpendicular condition
def is_perpendicular (a : ℝ) : Prop :=
(-1 / a) * (-a) = -1 → a = 0

theorem find_a_parallel :
  ∀ (a : ℝ), is_parallel a :=
begin
  intros a h,
  sorry -- Proof goes here
end

theorem find_a_perpendicular :
  ∀ (a : ℝ), is_perpendicular a :=
begin
  intros a h,
  sorry -- Proof goes here
end

end find_a_parallel_find_a_perpendicular_l81_81315


namespace probability_factor_of_5_factorial_is_8_over_15_l81_81640

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_of_factors (n : ℕ) : ℕ :=
  ( ∏ (p: ℕ) in (finset.range (n+1)).filter (λ x, nat.prime x ∧ n % x = 0), x.count_divisors n ).to_nat

def probability_factor_of_5_factorial : ℚ :=
  (number_of_factors (factorial 5) : ℚ) / 30

theorem probability_factor_of_5_factorial_is_8_over_15 :
  probability_factor_of_5_factorial = 8 / 15 := by
  sorry

end probability_factor_of_5_factorial_is_8_over_15_l81_81640


namespace grunters_win_probability_l81_81121

def probability_of_winning (win_prob : ℚ) (games_played : ℕ) (at_least_wins : ℕ) : ℚ :=
  let binomial (n k : ℕ) : ℕ := Nat.choose n k
  let win_k_games (k : ℕ) : ℚ :=
    ((binomial games_played k) : ℚ) * (win_prob ^ k) * ((1 - win_prob) ^ (games_played - k))
  (List.range at_least_wins).foldr (λ k acc, acc + win_k_games (games_played - k)) 0

theorem grunters_win_probability :
  probability_of_winning (4/5) 5 2 = 2304/3125 :=
by
  sorry

end grunters_win_probability_l81_81121


namespace total_amount_is_correct_l81_81710

def share_of_y_per_unit := 0.45
def share_of_z_per_unit := 0.50
def share_of_y := 45
def number_of_units := share_of_y / share_of_y_per_unit
def share_of_x_per_unit := 1

theorem total_amount_is_correct :
  share_of_x_per_unit * number_of_units + 
  share_of_y_per_unit * number_of_units + 
  share_of_z_per_unit * number_of_units = 195 := by
  sorry

end total_amount_is_correct_l81_81710


namespace red_blue_odd_vertices_even_l81_81093

-- Definitions based on the problem conditions
structure Map :=
  (countries : Type)
  (colors : countries → Fin 4)  -- red, yellow, blue, green represented as 0, 1, 2, 3
  (edges : countries → countries → Prop)  -- adjacency relation
  
def is_vertex (m : Map) (c1 c2 c3 : m.countries) : Prop :=
  m.edges c1 c2 ∧ m.edges c2 c3 ∧ m.edges c3 c1

-- Given conditions
axiom condition1 (m : Map) (p : m.countries) : ∃ c1 c2, m.edges c1 c2 ∨ ∃ c3, is_vertex m c1 c2 c3
axiom condition2 (m : Map) (p : m.countries) : ∃ c1 c2 c3, is_vertex m c1 c2 c3
axiom condition3 (m : Map ) (p : m.countries): ∀ c c', c ≠ c' → m.edges c c' → is_vertex m c c' = false
axiom condition4 (m : Map ) (v : m.countries): ∃ c1 c2 c3, is_vertex m c1 c2 c3
axiom condition5 (m : Map ) (c1 c2 : m.countries): m.edges c1 c2 → c1 ≠ c2
axiom condition6 (m : Map ): ∀ (c1 c2 : m.countries), m.edges c1 c2 → m.colors c1 ≠ m.colors c2

-- Definition of the number of vertices for a given country color
def vertices_count (m : Map) (color : Fin 4) : ℕ :=
  fintype.card {p : m.countries // m.colors p = color}

-- Number of countries with odd number of vertices of a given color
def odd_vertices_count (m : Map) (color : Fin 4) : ℕ :=
  finset.filter (λ x, x % 2 = 1) (finset.range (vertices_count m color)).card

-- The theorem to be proved
theorem red_blue_odd_vertices_even (m : Map) :
  (odd_vertices_count m 0 + odd_vertices_count m 2) % 2 = 0 := 
sorry

end red_blue_odd_vertices_even_l81_81093


namespace green_pairs_count_l81_81702

theorem green_pairs_count 
  (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) (total_pairs : ℕ) 
  (red_red_pairs : ℕ) 
  (H1 : red_students = 63)
  (H2 : green_students = 81)
  (H3 : total_students = red_students + green_students)
  (H4 : total_students = 144)
  (H5 : total_pairs = 72)
  (H6 : red_red_pairs = 27)
  : (total_pairs - red_red_pairs - 9 = 36) :=
begin
  -- Proof steps will go here
  sorry
end

end green_pairs_count_l81_81702


namespace area_of_triangle_A1B1C1_l81_81203

theorem area_of_triangle_A1B1C1
  (A B C C1 A1 B1 : Type)
  [geometry A B C]
  [geometry C1 A1 B1 A B C]
  (r : Rat)
  (h1 : r = 2)
  (h2 : area (triangle A B C) = 1)
  (h3 : segment_ratio (A, C1, B) = r)
  (h4 : segment_ratio (B, A1, C) = r)
  (h5 : segment_ratio (C, B1, A) = r) :
  area (triangle A1 B1 C1) = 1 / 3 :=
by 
  sorry

end area_of_triangle_A1B1C1_l81_81203


namespace probability_factor_of_120_in_range_l81_81615

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81615


namespace seeds_in_pots_l81_81755

theorem seeds_in_pots (x : ℕ) (total_seeds : ℕ) (seeds_fourth_pot : ℕ) 
  (h1 : total_seeds = 10) (h2 : seeds_fourth_pot = 1) 
  (h3 : 3 * x + seeds_fourth_pot = total_seeds) : x = 3 :=
by
  sorry

end seeds_in_pots_l81_81755


namespace find_b_l81_81345

-- Definitions based on the conditions in the problem
def eq1 (a : ℝ) := 3 * a + 3 = 0
def eq2 (a b : ℝ) := 2 * b - a = 4

-- Statement of the proof problem
theorem find_b (a b : ℝ) (h1 : eq1 a) (h2 : eq2 a b) : b = 3 / 2 :=
by
  sorry

end find_b_l81_81345


namespace probability_of_drawing_two_different_colors_l81_81029

variable (α : Type) [Fintype α]

def red_balls : Finset α := {A1, A2}
def black_balls : Finset α := {B1, B2, B3}
def all_balls : Finset α := red_balls ∪ black_balls

-- Proving the desired probability
theorem probability_of_drawing_two_different_colors
  (h : all_balls.card = 5) :
  (∑ x in (all_balls.product all_balls), ite (x.1 ≠ x.2) 1 0)
  * (2 / 1)
  = 3 / 5 := sorry

end probability_of_drawing_two_different_colors_l81_81029


namespace geometry_example_l81_81225

noncomputable def pyramid_intersection_area : ℝ :=
  let A := (0, 0, 0)
  let B := (6, 0, 0)
  let C := (6, 8, 0)
  let D := (0, 8, 0)
  let E := (3, 4, 5 * Real.sqrt 3)
  let P := ((6 + 3) / 2, (0 + 4) / 2, (0 + (5 * Real.sqrt 3)) / 2)
  let Q := (3, 8, 0)
  let plane := Plane.ofPoints A P Q
  let intersection_points := pyramidEdges.map (Plane.intersect plane)
  -- Calculate the actual area (requires vector cross product and determinant methods)
  let area := vectorCrossProductArea intersection_points -- Placeholder for cross product method
  (Real.sqrt area) -- We need to prove this equals sqrt(p) for some p

theorem geometry_example :
  ∃ p : ℝ, (pyramid_intersection_area = Real.sqrt p) := by
  sorry

end geometry_example_l81_81225


namespace probability_number_is_factor_of_120_l81_81623

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81623


namespace smallest_area_square_l81_81569

theorem smallest_area_square (a b u v : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : u = 4) (h₄ : v = 6) :
  ∃ s : ℕ, s^2 = 81 ∧ 
    (∀ xa ya xb yb xu yu xv yv : ℕ, 
      (xa + a ≤ s) ∧ (ya + b ≤ s) ∧ (xb + u ≤ s) ∧ (yb + v ≤ s) ∧ 
      ─xa < xb → xb < xa + a → ─ya < yb → yb < ya + b →
      ─xu < xv → xv < xu + u → ─yu < yv → yv < yu + v ∧
      (ya + b ≤ yv ∨ yu + v ≤ yb))
    := sorry

end smallest_area_square_l81_81569


namespace non_similar_triangles_with_arith_prog_angles_and_prime_factor_l81_81363

theorem non_similar_triangles_with_arith_prog_angles_and_prime_factor :
  ∃ (s : finset (ℕ × ℕ × ℕ)), 
    (∀ a b c ∈ s, 
      a ≠ b → 
      let ⟨α₁, β₁, γ₁⟩ := a in 
      let ⟨α₂, β₂, γ₂⟩ := b in 
      (α₁ + β₁ + γ₁ = 180) ∧ 
      (α₂ + β₂ + γ₂ = 180) ∧ 
      (α₁ < β₁) ∧ (β₁ < γ₁) ∧
      (β₁ - α₁ = γ₁ - β₁) ∧
      (β₂ - α₂ = γ₂ - β₂) ∧
      (α₁ % 5 = 0) ∧ 
      (α₂ % 5 = 0) ∧
      (α₁ = α₂ ∨ α₁ ≠ α₂)
    ) ∧
    (s.card = 4) := 
by
  sorry

end non_similar_triangles_with_arith_prog_angles_and_prime_factor_l81_81363


namespace angle_FDO_is_30_l81_81058

-- Define the triangle ABC with given angles
variable (A B C D E F O : Type) [Geometry A B C]
noncomputable def angle_B := 55
noncomputable def angle_C := 65
noncomputable def midpoint_D := midpoint B C = D
noncomputable def circumcircle_intersections :=
  (circumcircle A C D ∩ circumcircle A B D → AB = F) ∧ 
  (circumcircle A C D ∩ circumcircle A B D → AC = E)
noncomputable def circumcenter_O := circumcenter A E F = O

-- The main theorem we're proving
theorem angle_FDO_is_30 : angle F D O = 30 := by
  sorry

end angle_FDO_is_30_l81_81058


namespace dave_winfield_home_runs_l81_81240

theorem dave_winfield_home_runs (W : ℕ) (h : 755 = 2 * W - 175) : W = 465 :=
by
  sorry

end dave_winfield_home_runs_l81_81240


namespace smallest_middle_ring_number_is_106_l81_81202

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := 
  n > 1 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def letter_encoding : list ℕ :=
  list.range' 1 26  -- [1, 2, ..., 26]

def is_middle_ring_number (n : ℕ) : Prop :=
  is_composite n ∧ (∀ w : list ℕ, (∀ l ∈ w, l ∈ letter_encoding) → 
  (list.prod w ≠ n))

theorem smallest_middle_ring_number_is_106 : ∀ n : ℕ, 
  is_middle_ring_number n → n = 106 :=
begin
  sorry
end

end smallest_middle_ring_number_is_106_l81_81202


namespace initial_pile_counts_l81_81991

def pile_transfers (A B C : ℕ) : Prop :=
  (A + B + C = 48) ∧
  ∃ (A' B' C' : ℕ), 
    (A' = A + B) ∧ (B' = B + C) ∧ (C' = C + A) ∧
    (A' = 2 * 16) ∧ (B' = 2 * 12) ∧ (C' = 2 * 14)

theorem initial_pile_counts :
  ∃ A B C : ℕ, pile_transfers A B C ∧ A = 22 ∧ B = 14 ∧ C = 12 :=
by
  sorry

end initial_pile_counts_l81_81991


namespace hyperbola_foci_l81_81758

/-- Define a hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- Definition of the foci of the hyperbola -/
def foci_coords (c : ℝ) : Prop := c = Real.sqrt 29

/-- Proof that the foci of the hyperbola 4y^2 - 25x^2 = 100 are (0, -sqrt(29)) and (0, sqrt(29)) -/
theorem hyperbola_foci (x y : ℝ) (c : ℝ) (hx : hyperbola_eq x y) (hc : foci_coords c) :
  (x = 0 ∧ (y = -c ∨ y = c)) :=
sorry

end hyperbola_foci_l81_81758


namespace number_of_zeros_of_g_l81_81583
noncomputable theory
open Real

/-- Define the function f as an odd function on ℝ. -/
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f(x)

/-- Define the condition on f -/
def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x > 0, f(x) > -x * (deriv f x)

/-- Define the given function g -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := x * f(x) + log (abs (x + 1))

/-- The main theorem stating the number of zeros of g is 3. -/
theorem number_of_zeros_of_g (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : f 3 = 0) 
  (h3 : satisfies_condition f) :
  ∃ n : ℕ, n = 3 ∧ ∀ y : ℝ, g f y = 0 → y ∈ { -y | y ∈ ℝ } :=
by
  sorry

end number_of_zeros_of_g_l81_81583


namespace probability_factor_of_120_in_range_l81_81613

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81613


namespace bridge_length_l81_81712

theorem bridge_length (train_length : ℕ) (time_to_cross : ℕ) (speed_kmh : ℝ) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let total_distance := speed_ms * time_to_cross
  total_distance = train_length + 150 :=
begin
  let speed_ms := speed_kmh * (1000 / 3600),
  let total_distance := speed_ms * time_to_cross,
  calc
    total_distance = speed_kmh * (1000 / 3600) * time_to_cross : by rw total_distance
                ... = 16 * 25 : by sorry    -- Replace this placeholder with actual conversion
                ... = train_length + 150 : by sorry
end

end bridge_length_l81_81712


namespace max_area_height_l81_81293

theorem max_area_height (h : ℝ) (x : ℝ) 
  (right_trapezoid : True) 
  (angle_30_deg : True) 
  (perimeter_eq_6 : 3 * (x + h) = 6) : 
  h = 1 :=
by 
  sorry

end max_area_height_l81_81293


namespace probability_calculation_l81_81515

noncomputable def probability_of_event_A : ℚ := 
  let total_ways := 35 
  let favorable_ways := 6 
  favorable_ways / total_ways

theorem probability_calculation (A_team B_team : Type) [Fintype A_team] [Fintype B_team] [DecidableEq A_team] [DecidableEq B_team] :
  let total_players := 7 
  let selected_players := 4 
  let seeded_A := 2 
  let nonseeded_A := 1 
  let seeded_B := 2 
  let nonseeded_B := 2 
  let event_total_ways := Nat.choose total_players selected_players 
  let event_A_ways := Nat.choose seeded_A 2 * Nat.choose nonseeded_A 2 + Nat.choose seeded_B 2 * Nat.choose nonseeded_B 2 
  probability_of_event_A = 6 / 35 := 
sorry

end probability_calculation_l81_81515


namespace max_distance_l81_81891

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l81_81891


namespace inverse_proportional_k_value_l81_81838

theorem inverse_proportional_k_value (k : ℝ) :
  (∃ x y : ℝ, y = k / x ∧ x = - (Real.sqrt 2) / 2 ∧ y = Real.sqrt 2) → 
  k = -1 :=
by
  sorry

end inverse_proportional_k_value_l81_81838


namespace part1_part2_l81_81353

def f (x m : ℝ) : ℝ := x^2 - 4 * m * x + 6 * m

theorem part1 (m : ℝ) : 
  ∃ (a b : ℝ), 
    a < b ∧ 
    f(x, m) = 0 ↔ m ∈ set.Ioo a 0 ∨ m ∈ set.Ioo (3 / 2) b := 
sorry

theorem part2 (m : ℝ) :
  ∃ (min_val : ℝ), 
    (m ≤ 0 → min_val = f 0 m) ∧
    (0 < m ∧ m < 3 / 2 → min_val = f (2 * m) m) ∧
    (m ≥ 3 / 2 → min_val = f 3 m) := 
sorry

end part1_part2_l81_81353


namespace sin_alpha_beta_half_l81_81320

variables (α β : ℝ)

theorem sin_alpha_beta_half :
  (sin (π / 3 + α / 6) = -3 / 5 ∧ 
   cos (π / 6 + β / 2) = -12 / 13 ∧ 
   -5 * π < α ∧ α < -2 * π ∧ 
   -π / 3 < β ∧ β < 5 * π / 3) →
  sin (α / 6 + β / 2) = 33 / 65 :=
by
  sorry

end sin_alpha_beta_half_l81_81320


namespace pq_rs_sum_l81_81429

variable {ℝ : Type*}

theorem pq_rs_sum (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ p ∧ p ≠ r ∧ q ≠ s)
  (h_roots1 : ∀ x, x^2 - 12 * p * x - 13 * q = 0 → (x = r ∨ x = s))
  (h_roots2 : ∀ x, x^2 - 12 * r * x - 13 * s = 0 → (x = p ∨ x = q))
  (h_sum : p + q + r + s = 201) :
  pq + rs = -28743 / 12 :=
begin 
  sorry 
end

end pq_rs_sum_l81_81429


namespace length_of_AC_l81_81298

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end length_of_AC_l81_81298


namespace everett_weeks_worked_l81_81275

theorem everett_weeks_worked (daily_hours : ℕ) (total_hours : ℕ) (days_in_week : ℕ) 
  (h1 : daily_hours = 5) (h2 : total_hours = 140) (h3 : days_in_week = 7) : 
  (total_hours / (daily_hours * days_in_week) = 4) :=
by
  sorry

end everett_weeks_worked_l81_81275


namespace pages_and_cost_calculation_l81_81050

noncomputable def copy_pages_cost (cents_per_5_pages : ℕ) (total_cents : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
if total_cents < discount_threshold * (cents_per_5_pages / 5) then
  total_cents / (cents_per_5_pages / 5)
else
  let num_pages_before_discount := discount_threshold
  let remaining_pages := total_cents / (cents_per_5_pages / 5) - num_pages_before_discount
  let cost_before_discount := num_pages_before_discount * (cents_per_5_pages / 5)
  let discounted_cost := remaining_pages * (cents_per_5_pages / 5) * (1 - discount_rate)
  cost_before_discount + discounted_cost

theorem pages_and_cost_calculation :
  let cents_per_5_pages := 10
  let total_cents := 5000
  let discount_threshold := 1000
  let discount_rate := 0.10
  let num_pages := (cents_per_5_pages * 2500) / 5
  let cost := copy_pages_cost cents_per_5_pages total_cents discount_threshold discount_rate
  (num_pages = 2500) ∧ (cost = 4700) :=
by
  sorry

end pages_and_cost_calculation_l81_81050


namespace probability_factor_of_120_l81_81651

noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def numFactors (n : ℕ) : ℕ := 
  (List.range (n+1)).filter (λ d => d > 0 ∧ n % d = 0).length

def probability (num total : ℕ) : ℚ := num / total

theorem probability_factor_of_120 :
  probability (numFactors 120) 30 = 8 / 15 := 
by {
    sorry
}

end probability_factor_of_120_l81_81651


namespace solve_for_a_l81_81831

variables (a : ℂ) (Z1 Z2 : ℂ)

def Z1_def := Z1 = a + 2 * complex.I
def Z2_def := Z2 = complex.determinant ![![1, 2 * complex.I], ![2, 3]]

theorem solve_for_a
  (h1 : Z1_def a Z1)
  (h2 : Z2_def Z2)
  (h3 : is_real (Z1 / Z2)) :
  a = -3 / 2 :=
by
  sorry

end solve_for_a_l81_81831


namespace probability_factor_of_120_in_range_l81_81619

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81619


namespace tan_alpha_equal_neg_4_over_3_l81_81832

theorem tan_alpha_equal_neg_4_over_3 (α : ℝ)
  (h1 : α ∈ Ioo (π / 2) π)
  (h2 : 5 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) :
  Real.tan α = -4 / 3 :=
sorry

end tan_alpha_equal_neg_4_over_3_l81_81832


namespace ratio_of_common_differences_l81_81372

variable (x y d1 d2 : ℝ)

theorem ratio_of_common_differences (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0) 
  (seq1 : x + 4 * d1 = y) (seq2 : x + 5 * d2 = y) : d1 / d2 = 5 / 4 := 
sorry

end ratio_of_common_differences_l81_81372


namespace find_number_l81_81844

-- Define the number x and state the condition 55 + x = 88
def x := 33

-- State the theorem to be proven: if 55 + x = 88, then x = 33
theorem find_number (h : 55 + x = 88) : x = 33 :=
by
  sorry

end find_number_l81_81844


namespace smallest_sum_B_c_l81_81010

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end smallest_sum_B_c_l81_81010


namespace distance_between_hyperbola_vertices_l81_81282

theorem distance_between_hyperbola_vertices :
  let eqn := (16 * x^2 + 32 * x - 4 * y^2 - 8 * y - 3 = 0)
  in (distance_between_vertices eqn) = (sqrt 15 / 2) :=
by
  sorry

end distance_between_hyperbola_vertices_l81_81282


namespace max_distance_on_ellipse_l81_81909

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l81_81909


namespace grandpa_age_next_year_l81_81002

theorem grandpa_age_next_year (grandpa_age_current : ℕ) (mingming_age_current : ℕ) 
    (h1 : grandpa_age_current = 65) (h2 : mingming_age_current = 5) :
    (grandpa_age_current + 1) = 11 * (mingming_age_current + 1) :=
by
  rw [h1, h2]
  sorry

end grandpa_age_next_year_l81_81002


namespace fill_up_mini_vans_l81_81853

/--
In a fuel station, the service costs $2.20 per vehicle and every liter of fuel costs $0.70.
Assume that mini-vans have a tank size of 65 liters, and trucks have a tank size of 143 liters.
Given that 2 trucks were filled up and the total cost was $347.7,
prove the number of mini-vans filled up is 3.
-/
theorem fill_up_mini_vans (m : ℝ) (t : ℝ) 
    (service_cost_per_vehicle fuel_cost_per_liter : ℝ)
    (van_tank_size truck_tank_size total_cost : ℝ):
    service_cost_per_vehicle = 2.20 →
    fuel_cost_per_liter = 0.70 →
    van_tank_size = 65 →
    truck_tank_size = 143 →
    t = 2 →
    total_cost = 347.7 →
    (service_cost_per_vehicle * m + service_cost_per_vehicle * t) + (fuel_cost_per_liter * van_tank_size * m) + (fuel_cost_per_liter * truck_tank_size * t) = total_cost →
    m = 3 :=
by
  intros
  sorry

end fill_up_mini_vans_l81_81853


namespace probability_yellow_or_blue_twice_l81_81385

theorem probability_yellow_or_blue_twice :
  let total_faces := 12
  let yellow_faces := 4
  let blue_faces := 2
  let probability_yellow_or_blue := (yellow_faces / total_faces) + (blue_faces / total_faces)
  (probability_yellow_or_blue * probability_yellow_or_blue) = 1 / 4 := 
by
  sorry

end probability_yellow_or_blue_twice_l81_81385


namespace complex_fraction_simplification_l81_81112

theorem complex_fraction_simplification :
  (4 + 7 * complex.i) / (4 - 7 * complex.i) + (4 - 7 * complex.i) / (4 + 7 * complex.i) = -66 / 65 :=
by
  sorry

end complex_fraction_simplification_l81_81112


namespace find_sample_size_l81_81164

noncomputable def total_population := 120 + 80 + 60
noncomputable def elderly_population := 60
noncomputable def sample_elderly := 3
noncomputable def sampling_ratio := sample_elderly / elderly_population

theorem find_sample_size : (∃ n : ℕ, n / total_population = sample_elderly / elderly_population) :=
  by
  let n := 13
  have h_ratio : n / total_population = sample_elderly / elderly_population := by
    calc
      n / total_population = 13 / 260       : rfl
                      ... = 3 / 60         : by norm_num
  exact ⟨n, h_ratio⟩
  sorry

end find_sample_size_l81_81164


namespace polynomial_has_roots_l81_81286

-- Define the polynomial
def polynomial (x : ℂ) : ℂ := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7

-- Theorem to prove the existence of roots for the polynomial equation
theorem polynomial_has_roots : ∃ x : ℂ, polynomial x = 0 := by
  sorry

end polynomial_has_roots_l81_81286


namespace probability_number_is_factor_of_120_l81_81621

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81621


namespace no_positive_integer_solutions_l81_81009

def f (x : ℕ) : ℕ := x*x + x

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), a > 0 → b > 0 → 4 * (f a) ≠ (f b) :=
by
  intro a b a_pos b_pos
  sorry

end no_positive_integer_solutions_l81_81009


namespace trajectory_equation_and_max_area_l81_81864

/-- Given that the polar coordinate equation of the curve C₁ is ρ cos θ = 4,
  and a point M moves on C₁ such that |OM| * |OP| = 16 for point P on segment OM,
  the rectangular coordinate equation of the trajectory of point P is (x - 2)² + y² = 4,
  and the maximal area of triangle OAB given point A has polar coordinates (2, π/3)
  and point B lies on C₂ is √3 + 2. -/
theorem trajectory_equation_and_max_area :
  ∀ (θ ρ x y : ℝ),
  -- Curve C₁ in polar coordinates
  (ρ * cos θ = 4) → 
  -- Moving point M on C₁ with M having coordinates (4 * cos θ, 4 * sin θ)
  ((x = 4 * cos θ) ∧ (ρ = |OM|) ∧ (|OM| * |OP| = 16) → 
  -- Trajectory of point P
  (((x - 2) ^ 2) + y ^ 2 = 4)) ∧
  -- Maximum area of triangle OAB
  ((|A| = (1, sqrt 3)) ∧ (B ∈ (x - 2) ^ 2 + y ^ 2 = 4) →
  ∃ B, ∀ (A B : ℝ × ℝ), (max_area (O A B) = (sqrt 3) + 2)) :=
begin
  intros θ ρ x y hρθ hM,
  sorry  -- Proof is omitted
end

end trajectory_equation_and_max_area_l81_81864


namespace larger_number_problem_l81_81830

theorem larger_number_problem
  (x y : ℕ)
  (h1 : x * y = 40)
  (h2 : x + y = 13)
  (h3 : (Even x ∨ Even y)) :
  max x y = 8 :=
begin
  sorry
end

end larger_number_problem_l81_81830


namespace sum_of_values_of_x_l81_81291

def nabla (a b : ℝ) : ℝ := a * b - b * a^2

theorem sum_of_values_of_x : 
  (∑ x in ({x : ℝ | nabla 2 x - 8 = nabla x 6}), x) = -1 := 
sorry

end sum_of_values_of_x_l81_81291


namespace value_of_absolute_difference_value_of_cubic_sum_l81_81431

variable a b c : ℚ
variable (x₁ x₂ : ℚ)
variable h1 : 2 * x₁ ^ 2 + 7 * x₁ - 4 = 0
variable h2 : 2 * x₂ ^ 2 + 7 * x₂ - 4 = 0

theorem value_of_absolute_difference :
  |x₁ - x₂| = (9 / 2) :=
by
  sorry

theorem value_of_cubic_sum :
  x₁ ^ 3 + x₂ ^ 3 = (-511 / 8) :=
by
  sorry

end value_of_absolute_difference_value_of_cubic_sum_l81_81431


namespace necessarily_true_statement_l81_81229

-- Define the four statements as propositions
def Statement1 (d : ℕ) : Prop := d = 2
def Statement2 (d : ℕ) : Prop := d ≠ 3
def Statement3 (d : ℕ) : Prop := d = 5
def Statement4 (d : ℕ) : Prop := d % 2 = 0

-- The main theorem stating that given one of the statements is false, Statement3 is necessarily true
theorem necessarily_true_statement (d : ℕ) 
  (h1 : Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ ¬ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ ¬ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ ¬ Statement2 d ∧ Statement3 d ∧ Statement4 d)):
  Statement2 d :=
sorry

end necessarily_true_statement_l81_81229


namespace lambda_value_l81_81821

theorem lambda_value (λ : ℝ) :
  let m := (λ + 1, 1)
  let n := (λ + 2, 2)
  (λ m n : m.1 + n.1, m.2 + n.2) • (λ m n : m.1 - n.1, m.2 - n.2) = 0
  → λ = -3 := 
by
  sorry

end lambda_value_l81_81821


namespace solve_positive_integer_l81_81072

theorem solve_positive_integer (n : ℕ) (h : ∀ m : ℕ, m > 0 → n^m ≥ m^n) : n = 3 :=
sorry

end solve_positive_integer_l81_81072


namespace inequality_proof_l81_81791

theorem inequality_proof
  (n : ℕ)
  (x : Fin n → ℝ)
  (p q : ℝ)
  (h_pos : 0 < p)
  (h_q_ge_1 : 1 ≤ q)
  (h_pq_range : -1 < p - q ∧ p - q < 0)
  (h_x_pos : ∀ i : Fin n, 0 < x i ∧ x i < 1)
  (h_sum_x : ∑ i, x i = 1) :
  (∑ i, 1 / (x i ^ p - x i ^ q)) ≥ n ^ (q + 1) / (n ^ (q - p) - 1) := 
sorry


end inequality_proof_l81_81791


namespace probability_is_13_over_30_l81_81685

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81685


namespace terrorist_raid_participation_l81_81388

noncomputable def terrorist_raids : Prop :=
  (∃ (terrorists : Fin 101 → Set ℕ),
    (∀ i j, i ≠ j → ∃ (r : ℕ), r ∈ terrorists i ∧ r ∈ terrorists j) ∧
    ∃ (i : Fin 101), (Finset.card (Finset.filter (λ (r : ℕ), r ∈ terrorists i) Finset.univ) ≥ 11))

theorem terrorist_raid_participation : terrorist_raids :=
sorry

end terrorist_raid_participation_l81_81388


namespace chosen_number_probability_factorial_5_l81_81658

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_factors (n : ℕ) : ℕ :=
(nat.divisors n).length

def probability (favorable total : ℕ) : ℚ :=
favorable / total

theorem chosen_number_probability_factorial_5 :
  let n := 30 in
  let set := finset.range (n + 1) in
  let favorable_num := (finset.filter (λ x, is_factor x 120) set).card in
  let probability := probability favorable_num n in
  probability = (8 / 15 : ℚ) :=
by
  sorry

end chosen_number_probability_factorial_5_l81_81658


namespace total_people_informed_correct_l81_81120

-- Define the initial number of executive committee members
def exec_committee_members : ℕ := 6

-- Each committee member calls 6 different people
def first_round_calls (members : ℕ) : ℕ := members * 6

-- Each person called in the first round calls 6 other people
def second_round_calls (first_round : ℕ) : ℕ := first_round * 6

-- The total number of people who will know about the meeting
def total_informed_people : ℕ := 
  let members := exec_committee_members in
  let first_round := first_round_calls members in
  let second_round := second_round_calls first_round in
  members + first_round + second_round

theorem total_people_informed_correct : total_informed_people = 258 := 
  by 
    -- (Proof steps would go here)
    sorry

end total_people_informed_correct_l81_81120


namespace sally_fries_count_l81_81107

theorem sally_fries_count (sally_initial_fries mark_initial_fries : ℕ) 
  (mark_gave_fraction : ℤ) 
  (h_sally_initial : sally_initial_fries = 14) 
  (h_mark_initial : mark_initial_fries = 36) 
  (h_mark_give : mark_gave_fraction = 1 / 3) :
  sally_initial_fries + (mark_initial_fries * mark_gave_fraction).natAbs = 26 :=
by
  sorry

end sally_fries_count_l81_81107


namespace jenny_friends_count_l81_81153

theorem jenny_friends_count : 
  ∀ (cost_per_night_per_person : ℕ) (nights : ℕ) (total_cost : ℕ),
  cost_per_night_per_person = 40 →
  nights = 3 →
  total_cost = 360 →
  let F := (total_cost / (cost_per_night_per_person * nights)) - 1 in
  F = 2 :=
by {
  intros cost_per_night_per_person nights total_cost h1 h2 h3,
  let F := (total_cost / (cost_per_night_per_person * nights)) - 1,
  have hF : F = 2, {
    rw [h1, h2, h3],
    norm_num,
  },
  exact hF,
}

end jenny_friends_count_l81_81153


namespace sufficient_but_not_necessary_condition_l81_81799

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) (h : x > 0) : (∃ y : ℝ, (y < -3 ∨ y > -1) ∧ y > 0) := by
  sorry

end sufficient_but_not_necessary_condition_l81_81799


namespace distinct_positive_integer_triplets_l81_81313

theorem distinct_positive_integer_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hprod : a * b * c = 72^3) : 
  ∃ n, n = 1482 :=
by
  sorry

end distinct_positive_integer_triplets_l81_81313


namespace limit_example_l81_81947

theorem limit_example:
  ∀ ε > 0, ∃ δ > 0, ∀ x, (0 < |x - 1| ∧ |x - 1| < δ) → |((5 * x ^ 2 - 4 * x - 1) / (x - 1)) - 6| < ε :=
by
  -- The definition of delta.
  let δ (ε : Real) := ε / 5
  -- Introduce eps > 0
  assume ε hε
  -- Prove the existence of such a delta
  use δ ε, by
    intro h
    exact hε / 5, by
      intros x hx
      calc |((5 * x ^ 2 - 4 * x - 1) / (x - 1)) - 6|
        = |5 * x - 5| : sorry
      ... = 5 * |x - 1|   : sorry
      ... < ε             : by linarith [hx.left]

end limit_example_l81_81947


namespace area_of_triangle_range_of_a_l81_81814

def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 1)

def g (x : ℝ) : ℝ := (x^2 - a * x + 4) / x

theorem area_of_triangle (h : ∀ x, f x = abs (x + 1) - 2 * abs (x - 1)) : 
  (1/2) * 2 * (8/3) = (8 / 3) :=
sorry

theorem range_of_a {a : ℝ} (h : ∀ s t, s ∈ Ioi 0 → t ∈ Ioi 0 → g(s) ≥ f(t)) : 
  a ≤ 2 :=
sorry

end area_of_triangle_range_of_a_l81_81814


namespace probability_factorial_five_l81_81607

noncomputable def probability_factor_of_factorial_five : Prop :=
  let n := 30
  let factorial5 := 120
  let s : Finset ℕ := Finset.range (n + 1) -- This gives {0, 1, 2, ..., 30} in Lean, we can manually shift it to match {1, 2, ..., 30}
  let factors_of_120 := {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120} -- Precomputed factors of 120 within {1, 2, ..., 30}
  let favorable_outcomes := factors_of_120.filter (fun x => x ≤ n)
  let total_outcomes := s.filter (fun x => x ≠ 0)
  let probability := (favorable_outcomes.card : ℚ) / (total_outcomes.card)
  probability = 8 / 15

theorem probability_factorial_five : probability_factor_of_factorial_five :=
  by sorry

end probability_factorial_five_l81_81607


namespace general_formula_sequence_l81_81410

def arithmetic_sequence {a : ℕ → ℕ} (d : ℕ) : Prop :=
∀ n : ℕ, a (2 * n + 1) = a 1 + n * d

def geometric_sequence {a : ℕ → ℕ} (r : ℕ) : Prop :=
∀ n : ℕ, a (2 * n + 2) = a 2 * r^n

def a_mn {a : ℕ → ℕ} : Prop :=
∀ (m n : ℕ), m + n ≤ 5 → a m + a n = a (m + n)

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
(arithmetic_sequence 2) ∧ (geometric_sequence 2) ∧ a_mn

def b_n (a : ℕ → ℕ) (n : ℕ) : ℚ :=
if n % 2 = 1 then
  (a n + 1) / (a n * a n * a (n + 2) * a (n + 2))
else
  1 / (a n * a n)

def sum_b_n (a : ℕ → ℕ) (n : ℕ) : ℚ :=
(do Tnodd ← (1 to n by 2).map (b_n a)   -- sum for odd n
         (Ton ← (2 to n by 2).map (b_n a)   -- sum for even n
          Tnodd + Ton)

theorem general_formula_sequence :
  ∃ a : ℕ → ℕ,
  sequence a ∧
  (∀ n : ℕ, (n % 2 = 1 → a n = n) ∧ (n % 2 = 0 → a n = 2^(n / 2))) 
  ∧
  (∀ n : ℕ, 
       (n % 2 = 0 → sum_b_n a n = 7 / 12 - 1 / (4 * (n + 1) * (n + 1)) - 1 / (3 * 2 ^ n)) ∧
       (n % 2 = 1  → sum_b_n a n = 7 / 12 - 1 / (4 * (n + 2) * (n + 2)) - 1 / (3 * 2 ^ (n - 1))))
:= sorry

end general_formula_sequence_l81_81410


namespace richter_scale_frequency_l81_81453

theorem richter_scale_frequency (y : ℝ) : 10 ^ (5 - y) = 100 → y = 3 :=
by
  intro h
  have h1 : 10 ^ 2 = 100 := rfl -- acknowledging that 100 = 10^2
  rw [← h1] at h -- rewriting 100 as 10^2 in our condition
  rw [← eq_sub_iff_add_eq] -- rewriting equality for subtraction
  exact (eq_of_pow_eq_pow (zero_lt_ten) h).symm

end richter_scale_frequency_l81_81453


namespace probability_number_is_factor_of_120_l81_81630

theorem probability_number_is_factor_of_120:
  let S := {n | 1 ≤ n ∧ n ≤ 30} in
  let factorial_5 := 120 in
  let factors_of_120 := {n | n ∣ factorial_5} in
  let number_factors_120_in_S := (S ∩ factors_of_120).card in
  number_factors_120_in_S / 30 = 8 / 15 :=
by
  sorry

end probability_number_is_factor_of_120_l81_81630


namespace car_travel_distance_l81_81577

noncomputable def distance_in_miles (b t : ℝ) : ℝ :=
  (25 * b) / (1320 * t)

theorem car_travel_distance (b t : ℝ) : 
  let distance_in_feet := (b / 3) * (300 / t)
  let distance_in_miles' := distance_in_feet / 5280
  distance_in_miles' = distance_in_miles b t := 
by
  sorry

end car_travel_distance_l81_81577


namespace coefficient_of_x2_in_expansion_is_192_l81_81281

noncomputable def coefficient_of_x2_in_expansion : ℕ :=
  let n := 6
  let a := -2
  let b := -1
  let k := 1
  nat_binom n k * a^(n-k) * b^k

theorem coefficient_of_x2_in_expansion_is_192 :
  coefficient_of_x2_in_expansion = 192 :=
sorry

end coefficient_of_x2_in_expansion_is_192_l81_81281


namespace part_I_part_II_l81_81813

def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 1| - |x + 1|

theorem part_I (a : ℝ) (h : a = 2) :
  {x : ℝ | f a x ≥ 3} = {x : ℝ | x ≤ -2/3 } ∪ {x : ℝ | x ≥ 6} := 
sorry

theorem part_II (A : ℝ) (hA : A = 27 / 8) :
  ∃ a : ℝ, a > 0 ∧ let area := λ a, 
    let x1 := (a - 2) / (a + 1) in
    let x2 := (a + 2) / (a - 1) in
    (1 / 2) * ((x2 - x1) * 3) = A ∧ a = 3 := 
sorry

end part_I_part_II_l81_81813


namespace unreacted_moles_l81_81763

theorem unreacted_moles (m_CH4 m_Cl2 : ℝ) (yield : ℝ) :
  m_CH4 = 3 → m_Cl2 = 3 → yield = 0.80 →
  unreacted_methane = m_CH4 * (1 - yield) ∧ unreacted_chlorine = m_Cl2 * (1 - yield) →
  unreacted_methane = 0.6 ∧ unreacted_chlorine = 0.6 := 
by
  intros h1 h2 h3 h4;
  rw [h1, h2, h3] at h4;
  exact h4

end unreacted_moles_l81_81763


namespace max_distance_B_P_l81_81898

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l81_81898


namespace probability_factor_of_5_factorial_l81_81689

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (m n : ℕ) : Prop :=
  ∃ k, m * k = n

theorem probability_factor_of_5_factorial :
  let S := finset.range 31
  let fact_5 := factorial 5
  let num_factors := S.filter (is_factor fact_5)
  (num_factors.card : ℚ) / S.card = 8 / 15 :=
by
  sorry

end probability_factor_of_5_factorial_l81_81689


namespace simplify_and_multiply_roots_l81_81533

theorem simplify_and_multiply_roots :
  (256 = 4^4) →
  (64 = 4^3) →
  (16 = 4^2) →
  ∜256 * ∛64 * sqrt 16 = 64 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end simplify_and_multiply_roots_l81_81533


namespace slope_of_CD_l81_81962

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 2 * y + 40 = 0

-- Theorem statement
theorem slope_of_CD :
  ∃ C D : ℝ × ℝ,
    (circle1 C.1 C.2) ∧ (circle2 C.1 C.2) ∧ (circle1 D.1 D.2) ∧ (circle2 D.1 D.2) ∧
    (∃ m : ℝ, m = -2 / 3) := 
  sorry

end slope_of_CD_l81_81962


namespace johns_share_l81_81549

theorem johns_share (total_amount : ℕ) (r1 r2 r3 : ℕ) (h : total_amount = 6000) (hr1 : r1 = 2) (hr2 : r2 = 4) (hr3 : r3 = 6) :
  let total_ratio := r1 + r2 + r3
  let johns_ratio := r1
  let johns_share := (johns_ratio * total_amount) / total_ratio
  johns_share = 1000 :=
by
  sorry

end johns_share_l81_81549


namespace probability_is_13_over_30_l81_81678

def set_of_numbers : Finset ℕ := Finset.range 31
def factorial_5 : ℕ := nat.factorial 5
def factors_of_120_set : Finset ℕ := 
  (Finset.range 31).filter (λ x, x ∣ factorial_5)
def favorable_outcomes : ℕ := (factors_of_120_set).card
def total_outcomes : ℕ := (set_of_numbers \ {0}).card
def probability_of_being_factor : ℚ := favorable_outcomes / total_outcomes

theorem probability_is_13_over_30 : probability_of_being_factor = 13 / 30 := 
  by sorry

end probability_is_13_over_30_l81_81678


namespace pure_imaginary_a_value_l81_81380

theorem pure_imaginary_a_value (a : ℝ) (h : (2 + a * I) * (1 - I)).re = 0 : a = -2 :=
by
  sorry

end pure_imaginary_a_value_l81_81380


namespace chord_length_slope_angle_l81_81487

open Real

def eq_slope_angle (k : ℝ) : Prop :=
  ∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) ∧ k = tan θ

theorem chord_length_slope_angle :
  (∀ k : ℝ, (∃ θ : ℝ, (θ = π / 6 ∨ θ = 5 * π / 6) ∧ k = tan θ) ↔
    (∃ k : ℝ,
      (∀ x y : ℝ, (x - 2)^2 + (y - 3)^2 = 4 → ∃ y' : ℝ, y' = k * x + 3) ∧
      (2 * sqrt 3 = (2 * sqrt 3) * sqrt (k^2 + 1) / abs 2)) ) sorry

end chord_length_slope_angle_l81_81487


namespace closest_perfect_square_to_500_l81_81539

theorem closest_perfect_square_to_500 : 
  ∀ n : ℕ, (n * n = 484 ∨ n * n = 529) → 
  |500 - n * n| ≥ 16 → 
  n * n = 484 :=
by
  sorry

end closest_perfect_square_to_500_l81_81539


namespace system_of_equations_solution_l81_81478

theorem system_of_equations_solution :
  ∀ (x y z : ℤ),
    ((x = 38 ∧ y = 4 ∧ z = 9) ∨ (x = 110 ∧ y = 2 ∧ z = 33)) ↔
      (x * y - 2 * y = x + 106) ∧
      (y * z + 3 * y = z + 39) ∧
      (z * x + 3 * x = 2 * z + 438) :=
by
  intros x y z
  split
  {
    intro h
    cases h with h1 h2
    {
      cases h1
      simp [h1]
    },
    {
      cases h2
      simp [h2]
    }
  },
  {
    intro h
    have hf1 : x * y - 2 * y = x + 106 := h.left
    have hf2 : y * z + 3 * y = z + 39 := h.right.left
    have hf3 : z * x + 3 * x = 2 * z + 438 := h.right.right
    sorry -- The detailed proof logic can go here.
  }

end system_of_equations_solution_l81_81478


namespace probability_factor_of_120_in_range_l81_81609

theorem probability_factor_of_120_in_range :
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  ∃ (p : ℚ), p = ↑(factors_of_target.card) / ↑n ∧ p = 8 / 15 :=
by
  let n := 30
  let set := Finset.range (n + 1)
  let target := 120
  let factors_of_target := {d ∈ set | target % d = 0}
  have h_card : factors_of_target.card = 16 := sorry  -- Factor count derived
  have h_prob : ↑(factors_of_target.card) / ↑n = 8 / 15 := sorry
  exact ⟨8 / 15, h_prob, rfl⟩

end probability_factor_of_120_in_range_l81_81609


namespace math_proof_problem_l81_81003

noncomputable def certain_number : ℕ := 35

theorem math_proof_problem : ∀ (x : ℕ), x + 36 = 71 → x + 10 = 45 :=
by
  intros x h
  have hx : x = 71 - 36 := by sorry
  rw [hx]
  exact by norm_num

end math_proof_problem_l81_81003


namespace factor_probability_l81_81592

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def number_set : finset ℕ := finset.range 31

def factors (n : ℕ) : finset ℕ :=
  (finset.range (n + 1)).filter (λ x, n % x = 0)

theorem factor_probability :
  let n := 5
  let total_elements := 30
  let factorial_value := factorial n
  let factors_set := factors factorial_value
  let probability := (factors_set.filter (λ x, number_set ∈ x)).card.to_rat / total_elements.to_rat
  in probability = 8 / 15 :=
by {
  sorry
}

end factor_probability_l81_81592


namespace max_PB_distance_l81_81882

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l81_81882


namespace find_radius_min_OQ_value_l81_81586

-- Part 1: r = 4
theorem find_radius (r : ℝ) (A : ℝ × ℝ) (hA : A = (-3, 4)) (h_distance : ∀ B, (B.1^2 + B.2^2 = r^2) → (dist A B = 3)) : r = 4 := 
sorry

-- Part 2: Minimum value of |OQ| = 8
theorem min_OQ_value (r : ℝ) (h_r : r = 4) :
    ∀ P : ℝ × ℝ, P.1^2 + P.2^2 = r^2 →
    ∃ C D : ℝ × ℝ, C.1 > 0 ∧ C.2 = 0 ∧ D.1 = 0 ∧ D.2 > 0 ∧ 
    let O := (0, 0) in
    let Q := (C.1 + D.1, C.2 + D.2) in
    abs (Q.1^2 + Q.2^2) ≥ 8 :=
sorry

end find_radius_min_OQ_value_l81_81586


namespace line_ellipse_intersection_single_l81_81381

theorem line_ellipse_intersection_single (m : ℝ) :
  (∀ x : ℝ, has_single_solution $ x^2 + 6 * (m * x + 2)^2 - 4 = 0) → m^2 = 5 / 6 :=
by {
  sorry
}

end line_ellipse_intersection_single_l81_81381


namespace total_distance_between_first_and_fifth_poles_l81_81766

noncomputable def distance_between_poles (n : ℕ) (d : ℕ) : ℕ :=
  d / n

theorem total_distance_between_first_and_fifth_poles :
  ∀ (n : ℕ) (d : ℕ), (n = 3 ∧ d = 90) → (4 * distance_between_poles n d = 120) :=
by
  sorry

end total_distance_between_first_and_fifth_poles_l81_81766


namespace Tim_scores_expected_value_l81_81513

theorem Tim_scores_expected_value :
  let LAIMO := 15
  let FARML := 10
  let DOMO := 50
  let p := 1 / 3
  let expected_LAIMO := LAIMO * p
  let expected_FARML := FARML * p
  let expected_DOMO := DOMO * p
  expected_LAIMO + expected_FARML + expected_DOMO = 25 :=
by
  -- The Lean proof would go here
  sorry

end Tim_scores_expected_value_l81_81513


namespace find_x_for_prime_power_l81_81756

theorem find_x_for_prime_power (x : ℤ) :
  (∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ (2 * x * x + x - 6 = p ^ k)) → (x = -3 ∨ x = 2 ∨ x = 5) := by
  sorry

end find_x_for_prime_power_l81_81756


namespace cylinder_height_same_volume_as_cone_l81_81218

theorem cylinder_height_same_volume_as_cone
    (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V : ℝ)
    (h_volume_cone_eq : V = (1 / 3) * Real.pi * r_cone ^ 2 * h_cone)
    (r_cone_val : r_cone = 2)
    (h_cone_val : h_cone = 6)
    (r_cylinder_val : r_cylinder = 1) :
    ∃ h_cylinder : ℝ, (V = Real.pi * r_cylinder ^ 2 * h_cylinder) ∧ h_cylinder = 8 :=
by
  -- Here you would provide the proof for the theorem.
  sorry

end cylinder_height_same_volume_as_cone_l81_81218


namespace monotonicity_of_f_range_of_a_l81_81349

noncomputable def f (x a : ℝ) := Real.log x + Real.log a + (a - 1) * x + 2

theorem monotonicity_of_f (a : ℝ) (x : ℝ) (h_pos : a > 0) :
  (a ≥ 1 → ∀ x > 0, (f x a).deriv > 0) ∧
  (0 < a ∧ a < 1 → ∃ c : ℝ, ∀ x > 0, (x < c → (f x a).deriv > 0) ∧ (x > c → (f x a).deriv < 0)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, Real.exp (x - 2) ≥ f x a) → (0 < a ∧ a ≤ 1 / Real.exp 1) :=
sorry

end monotonicity_of_f_range_of_a_l81_81349


namespace problem_statement_l81_81347

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

axiom φ_cond (φ : ℝ) : 0 < φ ∧ φ < Real.pi / 2 

axiom symmetry_cond (x φ : ℝ) : f x φ = f (-Real.pi / 6 - (x + Real.pi / 6)) φ

theorem problem_statement (φ : ℝ) (hφ : φ_cond φ) (h_sym : ∀ x, symmetry_cond x φ) :
  f (Real.pi / 6) φ = -1 / 2 ∧ ∃! x ∈ (Set.Ioo (-Real.pi / 2) (Real.pi / 2)), 
  ∀ y ∈ (Set.Ioo (-Real.pi / 2) (Real.pi / 2)), f x φ ≥ f y φ :=
sorry

end problem_statement_l81_81347


namespace painting_problem_equation_l81_81267

def dougPaintingRate := 1 / 3
def davePaintingRate := 1 / 4
def combinedPaintingRate := dougPaintingRate + davePaintingRate
def timeRequiredToComplete (t : ℝ) : Prop := 
  (t - 1) * combinedPaintingRate = 2 / 3

theorem painting_problem_equation : ∃ t : ℝ, timeRequiredToComplete t :=
sorry

end painting_problem_equation_l81_81267
