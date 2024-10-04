import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Archimedean
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Polynomial.Bernstein
import Mathlib.Analysis.SpecialFunctions.Ellipse
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.Basic
import Mathlib.Probability.Bernoulli
import Mathlib.Probability.ProbabilityMeasure
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Real

namespace angle_BDC_is_15_l72_72902

-- Define the problem conditions
variables (A B C D : Point)
variables (triangle_ABC : Triangle A B C)
variables (triangle_ACD : Triangle A C D)
variable (congruent_triangles : Congruent triangle_ABC triangle_ACD)
variable (AB_eq_AC : A.distanceTo B = A.distanceTo C)
variable (AC_eq_AD : A.distanceTo C = A.distanceTo D)
variable (angle_BAC : angle A B C = 30)

-- State the problem to be proved
theorem angle_BDC_is_15 :
  angle B D C = 15 :=
sorry

end angle_BDC_is_15_l72_72902


namespace waffles_divisible_by_seven_l72_72497

theorem waffles_divisible_by_seven (x : ℕ) :
  let initial_waffles := 14 * x in
  ∃ n : ℕ, initial_waffles = 7 * n := 
by
  existsi (2 * x)
  rw [mul_assoc, ←mul_assoc 7]
  simp
  sorry

end waffles_divisible_by_seven_l72_72497


namespace percentage_increase_example_l72_72453

def percentage_increase (o n : ℝ) : ℝ :=
  ((n - o) / o) * 100

theorem percentage_increase_example : percentage_increase 40 60 = 50 :=
by {
  -- Skipping the proof
  sorry
}

end percentage_increase_example_l72_72453


namespace distance_covered_downstream_l72_72265

def speed_boat_still_water := 10 -- in kmph
def speed_current := 2 -- in kmph
def time_downstream := 17.998560115190788 -- in seconds

-- Converting speed from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

def effective_speed_downstream_mps : ℝ := kmph_to_mps (speed_boat_still_water + speed_current)

theorem distance_covered_downstream :
  let distance := effective_speed_downstream_mps * time_downstream in
  abs (distance - 59.99520038396929) < 0.001 := 
by
  sorry

end distance_covered_downstream_l72_72265


namespace angle_POQ_eq_angle_BAD_iff_ABCD_eq_ADBC_l72_72520

variable {O A B C D P Q : Type}
variable [Circle O] [Quadrilateral A B C D] [Point E P Q A B C D]
variable [Line AC] [Line OE] [Perpendicular E OE] [Segment E P AC] [Segment E Q AC]

-- Conditions from the problem description
def inscribed_quadrilateral (ABCD : Quadrilateral) (O : Circle) := 
  InscribedQuadrilateral ABCD O

def not_diameter (AC : Line) (O : Circle) : Prop := 
  ¬Diameter AC O

def point_on_segment_AC (E : Point) (AC : Line) (ratio : ℝ) : Prop := 
  segmentRatio E AC ratio = 4

def perpendicular_from_point (E : Point) (OE : Line) (P Q : Point) := 
  PerpendicularFrom E OE P Q

-- Main theorem statement
theorem angle_POQ_eq_angle_BAD_iff_ABCD_eq_ADBC 
  (h1 : inscribed_quadrilateral ABCD O)
  (h2 : not_diameter AC O)
  (h3 : point_on_segment_AC E AC 4)
  (h4 : perpendicular_from_point E OE P Q) :
  ∠(P, O, Q) = ∠(B, A, D) ↔ (length (A, B) * length (C, D) = length (A, D) * length (B, C)) :=
sorry

end angle_POQ_eq_angle_BAD_iff_ABCD_eq_ADBC_l72_72520


namespace number_of_bonnies_l72_72573

theorem number_of_bonnies (B blueberries apples : ℝ) 
  (h1 : blueberries = 3 / 4 * B) 
  (h2 : apples = 3 * blueberries)
  (h3 : B + blueberries + apples = 240) : 
  B = 60 :=
by
  sorry

end number_of_bonnies_l72_72573


namespace accurate_to_ten_thousandth_l72_72542

/-- Define the original number --/
def original_number : ℕ := 580000

/-- Define the accuracy of the number represented by 5.8 * 10^5 --/
def is_accurate_to_ten_thousandth_place (n : ℕ) : Prop :=
  n = 5 * 100000 + 8 * 10000

/-- The statement to be proven --/
theorem accurate_to_ten_thousandth : is_accurate_to_ten_thousandth_place original_number :=
by
  sorry

end accurate_to_ten_thousandth_l72_72542


namespace angle_E_in_parallelogram_EFGH_l72_72519

theorem angle_E_in_parallelogram_EFGH
  (EFGH_parallelogram : Parallelogram E F G H)
  (exterior_angle_F : angle (line_through F H) (line_through E F) = 150) :
  angle E = 30 :=
by
  sorry

end angle_E_in_parallelogram_EFGH_l72_72519


namespace parabola_eqn_min_distance_l72_72816

theorem parabola_eqn (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0) :
  (∀ x : ℝ,  y = a * x^2 + b * x) ↔ (∀ x : ℝ, y = (1/3) * x^2 - (2/3) * x) :=
by
  sorry

theorem min_distance (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0)
  (line_eq : ∀ x, (y : ℝ) = x - 25/4) :
  (∀ P : ℝ × ℝ, ∃ P_min : ℝ × ℝ, P_min = (5/2, 5/12)) :=
by
  sorry

end parabola_eqn_min_distance_l72_72816


namespace disprove_prime_statement_l72_72635

theorem disprove_prime_statement : ∃ n : ℕ, ((¬ Nat.Prime n) ∧ Nat.Prime (n + 2)) ∨ (Nat.Prime n ∧ ¬ Nat.Prime (n + 2)) :=
sorry

end disprove_prime_statement_l72_72635


namespace count_valid_even_numbers_l72_72041

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_between (n : ℕ) (a b : ℕ) : Prop :=
  a ≤ n ∧ n ≤ b

def digits_are_unique_and_in_set (n : ℕ) (s : set ℕ) : Prop :=
  let digits := (nat.digits 10 n) in
  (∀ d ∈ digits, d ∈ s) ∧ (list.nodup digits)

def valid_numbers (s : set ℕ) (a b : ℕ) : finset ℕ :=
  by sorry -- Implement a function that generates the set of valid numbers

theorem count_valid_even_numbers : (valid_numbers {1, 3, 4, 5, 7, 8} 300 800).card = 21 :=
  by sorry

end count_valid_even_numbers_l72_72041


namespace unique_f_solution_l72_72958

noncomputable def f : ℕ → ℝ := sorry

theorem unique_f_solution (f : ℕ → ℝ) (h1 : f 1 > 0)
    (h2 : ∀ n : ℕ, 1 ≤ n → ∑ d in divisors n, f d * f (n / d) = 1) :
    f (2018 ^ 2019) = (choose 4038 2019 ^ 2) / (2 ^ 8076) :=
sorry

end unique_f_solution_l72_72958


namespace zane_total_payment_l72_72991

open Real

noncomputable def shirt1_price := 50.0
noncomputable def shirt2_price := 50.0
noncomputable def discount1 := 0.4 * shirt1_price
noncomputable def discount2 := 0.3 * shirt2_price
noncomputable def price1_after_discount := shirt1_price - discount1
noncomputable def price2_after_discount := shirt2_price - discount2
noncomputable def total_before_tax := price1_after_discount + price2_after_discount
noncomputable def sales_tax := 0.08 * total_before_tax
noncomputable def total_cost := total_before_tax + sales_tax

-- We want to prove:
theorem zane_total_payment : total_cost = 70.20 := by sorry

end zane_total_payment_l72_72991


namespace cos_a5_value_l72_72348

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a(n + 1) = a(n) + d

theorem cos_a5_value (a : ℕ → ℝ) (d : ℝ):
  arithmetic_sequence a d →
  a(1) + a(2) + a(3) = π / 2 →
  a(7) + a(8) + a(9) = π →
  cos (a(5)) = sqrt 2 / 2 :=
by
  intro ha hb hc
  sorry

end cos_a5_value_l72_72348


namespace remainder_when_divided_by_95_l72_72624

theorem remainder_when_divided_by_95 (x : ℤ) (h1 : x % 19 = 12) :
  x % 95 = 12 := 
sorry

end remainder_when_divided_by_95_l72_72624


namespace binomial_coefficient_a_l72_72056

theorem binomial_coefficient_a (a : ℝ) : 
  let coeff := (nat.choose 6 3 : ℝ) * a^3 in
  coeff = -160 →
  a = -2 :=
by
  intro h
  sorry

end binomial_coefficient_a_l72_72056


namespace binary_sum_correct_l72_72209

theorem binary_sum_correct : 
  let x1 := 1111111111
  let x2 := 1010101010
  let x3 := 11110000
  let b2d := λ (n : ℕ), (List.range (Nat.Num.digits 2 n).length).foldr (fun i acc => acc + if n.test_bit i then 2^i else 0) 0
  b2d x1 + b2d x2 + b2d x3 = 1945 := 
by
  sorry

end binary_sum_correct_l72_72209


namespace presidency_meeting_combinations_l72_72644

noncomputable def choose : ℕ → ℕ → ℕ
| n, k => Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem presidency_meeting_combinations :
  let schools : ℕ := 3;
  let students_per_school : ℕ := 6;
  let host_combinations := choose students_per_school 3;
  let non_host_combinations := choose students_per_school 1;
  2160 = schools * (host_combinations * non_host_combinations * non_host_combinations) :=
by
  -- compute choose 6 3
  have h1 : choose 6 3 = 20 := by
    simp [choose, Nat.factorial, Nat.div]
  -- compute choose 6 1
  have h2 : choose 6 1 = 6 := by
    simp [choose, Nat.factorial, Nat.div]
  -- compute total combinations for one choice of host
  have h3 : 20 * 6 * 6 = 720 := by
    norm_num
  -- compute total combinations for all choices of host
  have h4 : 3 * 720 = 2160 := by
    norm_num
  -- prove the final equality
  exact Eq.trans (congr_arg (fun x => 3 * x) h3) h4
  sorry

end presidency_meeting_combinations_l72_72644


namespace platform_length_l72_72664

noncomputable def speed_kmph_to_mps (v_kmph : ℕ) : ℝ := v_kmph * 1000 / 3600

noncomputable def total_distance_covered (speed_mps : ℝ) (time_s : ℝ) : ℝ := speed_mps * time_s

def length_of_platform (total_distance : ℝ) (train_length : ℝ) : ℝ := total_distance - train_length

theorem platform_length
  (train_length : ℝ)
  (speed_kmph : ℕ)
  (time_s : ℝ) :
  let speed_mps := speed_kmph_to_mps speed_kmph
  in
  let total_distance := total_distance_covered speed_mps time_s
  in
  length_of_platform total_distance train_length = 290 :=
by
  have speed_mps := speed_kmph_to_mps speed_kmph
  have total_distance := total_distance_covered speed_mps time_s
  have platform_len := length_of_platform total_distance train_length
  -- Placeholder for the actual proof steps
  sorry

end platform_length_l72_72664


namespace number_of_whistlers_l72_72826

def fireworks_equation (W : ℕ) : Prop :=
  2 * 3 + 2 * W + 8 + 9 = 33

theorem number_of_whistlers :
  ∃ W : ℕ, fireworks_equation W ∧ W = 5 :=
by
  use 5
  constructor
  · unfold fireworks_equation
    norm_num
  · refl

end number_of_whistlers_l72_72826


namespace area_ring_shaped_region_l72_72578

-- Define the radii of the innermost and outermost circles
def radius_innermost := 4
def radius_outermost := 15

-- Define the area of a circle given its radius
def area_circle (r : ℝ) := Real.pi * r ^ 2

-- Theorem statement: Area of the ring-shaped region
theorem area_ring_shaped_region : 
  area_circle radius_outermost - area_circle radius_innermost = 209 * Real.pi := 
by
  sorry

end area_ring_shaped_region_l72_72578


namespace complex_squared_result_l72_72741

-- Define the imaginary unit 'i'
noncomputable def i : ℂ := complex.I

-- Given condition
def a : ℂ := (-3 - i) / (1 + 2 * i)

-- Goal to prove
theorem complex_squared_result : a^2 = -2 * i := by
  sorry

end complex_squared_result_l72_72741


namespace right_triangle_log_hypotenuse_l72_72688

theorem right_triangle_log_hypotenuse :
  let u := log 2 5 in
  let a := log 8 125 in
  let b := log 2 25 in
  let h := sqrt 5 * u in
  a = u ∧ b = 2 * u → 8^h = 5^sqrt(5) :=
by
  sorry

end right_triangle_log_hypotenuse_l72_72688


namespace find_initial_value_l72_72267

noncomputable def initial_value : ℕ :=
859560 - (859560 % 456)

theorem find_initial_value :
  ∃ (x : ℕ), x + (859560 % 456) = 859560 ∧ x = 859376 :=
begin
  use initial_value,
  split,
  { rw initial_value,
    rw ← nat.add_sub_assoc,
    exact nat.mod_le 859560 456 },
  { rw initial_value,
    reflexivity }
end

end find_initial_value_l72_72267


namespace cyclic_inequality_l72_72530

theorem cyclic_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^3 + y^3 + z^3 ≥ x^2 * (sqrt (y * z)) + y^2 * (sqrt (z * x)) + z^2 * (sqrt (x * y)) :=
by
  sorry

end cyclic_inequality_l72_72530


namespace smaller_angle_at_3_20_correct_l72_72979

noncomputable def smaller_angle_at_3_20 : Float :=
  let degrees_per_minute_for_minute_hand := 360 / 60
  let degrees_per_minute_for_hour_hand := 360 / (60 * 12)
  let initial_hour_hand_position := 90.0  -- 3 o'clock position
  let minute_past_three := 20
  let minute_hand_movement := minute_past_three * degrees_per_minute_for_minute_hand
  let hour_hand_movement := minute_past_three * degrees_per_minute_for_hour_hand
  let current_hour_hand_position := initial_hour_hand_position + hour_hand_movement
  let angle_between_hands := minute_hand_movement - current_hour_hand_position
  if angle_between_hands < 0 then
    -angle_between_hands
  else
    angle_between_hands

theorem smaller_angle_at_3_20_correct : smaller_angle_at_3_20 = 20.0 := by
  sorry

end smaller_angle_at_3_20_correct_l72_72979


namespace min_selections_for_product_238_l72_72873

theorem min_selections_for_product_238 : 
  ∀ (s : Finset ℕ), (∀ n ∈ s, 1 ≤ n ∧ n ≤ 200) → s.card ≥ 198 → 
  ∃ (a b ∈ s), a ≠ b ∧ a * b = 238 :=
by
  intros s hs hcard
  sorry

end min_selections_for_product_238_l72_72873


namespace max_area_of_triangle_ABC_l72_72108

open Real

-- Define the points A, B, C as given in the problem
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (4, 4)
def C (p : ℝ) : ℝ × ℝ := (p, p^2 - 4 * p + 4)

-- Define the function to calculate the area of triangle ABC using the Shoelace Theorem
def triangle_area (p : ℝ) : ℝ :=
  (1 / 2) * abs (0 * (p^2 - 4 * p + 4) + 4 * (p^2 - 4 * p + 4) + p * 4 - 4 * 4 - 4 * p)

-- The main theorem stating the maximum area
theorem max_area_of_triangle_ABC : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 4 ∧ triangle_area p = 8 :=
by
  use 2
  split
  { norm_num }
  split
  { norm_num }
  calc
    triangle_area 2 = 8 : by
      unfold triangle_area
      norm_num
      rw abs_of_nonneg
      { norm_num }
      { calc
          16 - 16 ≥ 0 : by norm_num
      }
  sorry

end max_area_of_triangle_ABC_l72_72108


namespace problem_statement_l72_72368

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (sqrt (x ^ 2 + 1) - x + 1) * (sqrt (y ^ 2 + 1) - y + 1) = 2) : 
  xy - 2 = -1 :=
by 
  sorry

end problem_statement_l72_72368


namespace proof_projections_equal_l72_72865

variables {S1 S2 : Type} [Sphere S1] [Sphere S2]
variables {P Q : Point}
variables {A B : S1.Point} {C D : S2.Point}
variables {E F : Point} {AC BD : Line}
variables {proj : (Line × Segment) → Line}

open Segment Line

noncomputable def projections_equal (S1 S2 : Type) [Sphere S1] [Sphere S2] 
  (P Q : Point) (A B : S1.Point) (C D : S2.Point) (E F : Point) 
  (AC BD : Line) (proj : (Line × Segment) → Line) : Prop :=
  (proj (AC, AB)) = (proj (AC, CD))

theorem proof_projections_equal {S1 S2 : Type} [Sphere S1] [Sphere S2] 
  {P Q : Point} {A B : S1.Point} {C D : S2.Point} {E F : Point} 
  {AC BD : Line} {proj : (Line × Segment) → Line} 
  (h1 : SegmentContains AC A C E) (h2 : SegmentContains BD B D F) 
  (h3 : Parallel (LineThrough P Q) BD) : 
  projections_equal S1 S2 P Q A B C D E F AC BD proj :=
begin
  sorry
end

end proof_projections_equal_l72_72865


namespace find_monic_quadratic_polynomial_l72_72712

-- Define the conditions
def is_monic {R : Type*} [CommRing R] (p : Polynomial R) : Prop :=
  p.leadingCoeff = 1

def has_real_coefficients {R : Type*} [CommRing R] (p : Polynomial R) : Prop :=
  ∀ c ∈ p.coeff.support, is_real (p.coeff c)

def has_root {R : Type*} [CommRing R] (p : Polynomial R) (z : R) : Prop :=
  p.eval z = 0

-- Define the complex polynomial we want to prove is the solution
def monic_quadratic_polynomial : Polynomial ℂ :=
  Polynomial.X^2 - 4 * Polynomial.X + 5

-- Theorem statement
theorem find_monic_quadratic_polynomial :
  is_monic monic_quadratic_polynomial ∧
  has_real_coefficients monic_quadratic_polynomial ∧
  has_root monic_quadratic_polynomial (2 + -1 * complex.i) :=
by
  sorry

end find_monic_quadratic_polynomial_l72_72712


namespace triangle_division_l72_72722

theorem triangle_division (n : ℕ) (h : n ≥ 2) :
  ∃ (A : fin n → Type) (similar : ∀ i j, A i ≃ A j → i = j), 
  ∀ i, ∃ (B : fin n → Type), (∀ j, B j ≃ A (j % n)) ∧ (∀ j k, j ≠ k → B j ≠ B k) :=
sorry

end triangle_division_l72_72722


namespace find_m_l72_72366

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (m x : ℝ) := (1 / 2) * x ^ 2 + m * x + (7 / 2)

theorem find_m :
  ∃ (m : ℝ), m < 0 ∧
  (∀ l : ℝ → ℝ, l = λ x, x - 1 →
    (∃ (x₁ y₁ : ℝ), x₁ = 1 ∧ y₁ = f x₁ ∧ (∀ x, f x - l x ≤ 0) ∧ (∀ x, g m x - l x ≤ 0) ∧
     (∃ (x₂ y₂ : ℝ), x₂ ≠ x₁ ∧ g m x₂ = l x₂)) →
   m = -2) := 
sorry

end find_m_l72_72366


namespace triangle_area_comparison_triangle_perimeter_comparison_l72_72199

theorem triangle_area_comparison (A B C1 C2 : Point) (AB : Line)
  (common_base : A ≠ B ∧ AB ∈ Line.through A B)
  (equal_angle : ∠ A C1 B = ∠ A C2 B)
  (side_length_diff : |dist A C1 - dist C1 B| < |dist A C2 - dist C2 B|) :
   area (triangle A B C1) > area (triangle A B C2) :=
sorry

theorem triangle_perimeter_comparison (A B C1 C2 : Point) (AB : Line)
  (common_base : A ≠ B ∧ AB ∈ Line.through A B)
  (equal_angle : ∠ A C1 B = ∠ A C2 B)
  (side_length_diff : |dist A C1 - dist C1 B| < |dist A C2 - dist C2 B|) :
   perimeter (triangle A B C1) > perimeter (triangle A B C2) :=
sorry

end triangle_area_comparison_triangle_perimeter_comparison_l72_72199


namespace total_packs_sold_l72_72225

def packs_sold_village_1 : ℕ := 23
def packs_sold_village_2 : ℕ := 28

theorem total_packs_sold : packs_sold_village_1 + packs_sold_village_2 = 51 :=
by
  -- We acknowledge the correctness of the calculation.
  sorry

end total_packs_sold_l72_72225


namespace count_is_68_l72_72139

def books : List String := ["Book of Songs", "Book of Documents", "Book of Rites", "Book of Changes", "Spring and Autumn Annals"]

def condition1 (permutation : List String) : Bool :=
  books.allDifferent permutation

def condition2 (permutation : List String) : Bool :=
  not ((permutation.indexOf! "Book of Songs" + 1 == permutation.indexOf! "Book of Rites") ∨ 
       (permutation.indexOf! "Book of Rites" + 1 == permutation.indexOf! "Book of Songs"))

def condition3 (permutation : List String) : Bool :=
  permutation.head ≠ "Book of Changes"

def valid_arrangements : List (List String) :=
  books.permutations.filter (λp => condition1 p ∧ condition2 p ∧ condition3 p)

def count_valid_arrangements : Nat :=
  valid_arrangements.length

theorem count_is_68 : count_valid_arrangements = 68 := by
  sorry

end count_is_68_l72_72139


namespace monkey_farm_l72_72072

theorem monkey_farm (x y : ℕ) 
  (h1 : y = 14 * x + 48) 
  (h2 : y = 18 * x - 64) : 
  x = 28 ∧ y = 440 := 
by 
  sorry

end monkey_farm_l72_72072


namespace total_pages_correct_average_page_count_per_chapter_correct_percentage_chapter1_correct_percentage_chapter2_correct_percentage_chapter3_correct_percentage_chapter4_correct_percentage_chapter5_correct_percentage_chapter6_correct_percentage_chapter7_correct_l72_72255

def page_count_chapter1 : ℕ := 66
def page_count_chapter2 : ℕ := 35
def page_count_chapter3 : ℕ := 24
def page_count_chapter4 : ℕ := 52
def page_count_chapter5 : ℕ := 48
def page_count_chapter6 : ℕ := 39
def page_count_chapter7 : ℕ := 58

def number_of_chapters : ℕ := 7

def total_pages : ℕ := page_count_chapter1 + page_count_chapter2 + page_count_chapter3 + page_count_chapter4 + page_count_chapter5 + page_count_chapter6 + page_count_chapter7

theorem total_pages_correct : total_pages = 322 := sorry

def average_page_count_per_chapter : ℝ := total_pages / number_of_chapters

theorem average_page_count_per_chapter_correct : average_page_count_per_chapter ≈ 46 := sorry

def percentage (count : ℕ) (total : ℕ) : ℝ := (count : ℝ) / (total : ℝ) * 100

theorem percentage_chapter1_correct : percentage page_count_chapter1 total_pages ≈ 20.50 := sorry
theorem percentage_chapter2_correct : percentage page_count_chapter2 total_pages ≈ 10.87 := sorry
theorem percentage_chapter3_correct : percentage page_count_chapter3 total_pages ≈ 7.45 := sorry
theorem percentage_chapter4_correct : percentage page_count_chapter4 total_pages ≈ 16.15 := sorry
theorem percentage_chapter5_correct : percentage page_count_chapter5 total_pages ≈ 14.91 := sorry
theorem percentage_chapter6_correct : percentage page_count_chapter6 total_pages ≈ 12.11 := sorry
theorem percentage_chapter7_correct : percentage page_count_chapter7 total_pages ≈ 18.01 := sorry

end total_pages_correct_average_page_count_per_chapter_correct_percentage_chapter1_correct_percentage_chapter2_correct_percentage_chapter3_correct_percentage_chapter4_correct_percentage_chapter5_correct_percentage_chapter6_correct_percentage_chapter7_correct_l72_72255


namespace length_of_each_train_l72_72971

def trains_length (L : ℝ) : Prop :=
  let v_faster := 46 * (1000 / 3600)
  let v_slower := 36 * (1000 / 3600)
  let relative_speed := v_faster - v_slower
  let time := 72
  let total_distance := relative_speed * time
  2 * L = total_distance

theorem length_of_each_train : ∃ L : ℝ, trains_length L :=
  exists.intro 100 sorry

end length_of_each_train_l72_72971


namespace arithmetic_sequence_sum_mul_three_eq_3480_l72_72298

theorem arithmetic_sequence_sum_mul_three_eq_3480 :
  let a := 50
  let d := 3
  let l := 95
  let n := ((l - a) / d + 1 : ℕ)
  let sum := n * (a + l) / 2
  3 * sum = 3480 := by
  sorry

end arithmetic_sequence_sum_mul_three_eq_3480_l72_72298


namespace number_line_distance_problem_l72_72935

theorem number_line_distance_problem :
  (∀ x : ℝ, abs (x - 0) = √5 → (x = √5 ∨ x = -√5)) ∧
  (∀ y : ℝ, abs (y - √5) = 2 * √5 → (y = 3 * √5 ∨ y = -√5)) :=
  by
  sorry

end number_line_distance_problem_l72_72935


namespace weight_loss_target_l72_72087

variable (J S : ℝ)

/-- Given Jake's current weight and their combined weight, Jake needs to lose 8 pounds to weigh
twice as much as his sister. -/
theorem weight_loss_target (h1 : J = 196) (h2 : J + S = 290) : J - 2 * S = 8 := by
  -- Load the assumptions
  have hS : S = 290 - J := by
    rw [← h2]
  rw [h1, hS]
  -- Simplify the expression
  norm_num
  -- The goal is verified
  sorry

end weight_loss_target_l72_72087


namespace selling_price_is_correct_l72_72999

-- Definitions based on conditions
def cost_price : ℝ := 280
def profit_percentage : ℝ := 0.3
def profit_amount : ℝ := cost_price * profit_percentage

-- Selling price definition
def selling_price : ℝ := cost_price + profit_amount

-- Theorem statement
theorem selling_price_is_correct : selling_price = 364 := by
  sorry

end selling_price_is_correct_l72_72999


namespace equal_radii_l72_72904

noncomputable theory

variables {P T : Point}
variables {c1 c2 : Circle}
variables {l1 l2 p1 p2 p3 p4 : Line}
variables {A1 B1 C1 D1 A2 B2 C2 D2 : Point}

-- Hypotheses: conditions from the problem
axiom common_tangents_intersect (h1 : Tangent c1 L1) (h2 : Tangent c2 L1) (h3 : Tangent c1 L2) (h4 : Tangent c2 L2) : Intersect at point P
axiom tangents_from_T (h5 : Tangent p1 c1) (h6 : Tangent p2 c1) (h7 : Tangent p3 c2) (h8 : Tangent p4 c2)
axiom intersections_l1 (ha1 : Intersection l1 p1 = A1) (hb1 : Intersection l1 p2 = B1) (hc1 : Intersection l1 p3 = C1) (hd1 : Intersection l1 p4 = D1) (order_A1B1PC1D1 : Order A1 B1 P C1 D1)
axiom intersections_l2 (ha2 : Intersection l2 p1 = A2) (hb2 : Intersection l2 p2 = B2) (hc2 : Intersection l2 p3 = C2) (hd2 : Intersection l2 p4 = D2)
axiom cyclic_quadrangles (h_cyclic1 : Cyclic A1 A2 D1 D2) (h_cyclic2 : Cyclic B1 B2 C1 C2)

-- Goal: prove that the radii of c1 and c2 are equal
theorem equal_radii : c1.radius = c2.radius :=
by 
  sorry

end equal_radii_l72_72904


namespace intersection_of_sets_l72_72373

theorem intersection_of_sets :
  let A := {x : ℝ | x > -1}
  let B := {-1, 0, 1, 2}
  A ∩ B = {0, 1, 2} :=
by
  let A := {x : ℝ | x > -1}
  let B := {-1, 0, 1, 2}
  have h1: A = {x : ℝ | x > -1}, from rfl,
  have h2: B = {-1, 0, 1, 2}, from rfl,
  sorry

end intersection_of_sets_l72_72373


namespace find_a_for_tangent_l72_72416

theorem find_a_for_tangent (a : ℤ) (x : ℝ) (h : ∀ x, 3*x^2 - 4*a*x + 2*a > 0) : a = 1 :=
sorry

end find_a_for_tangent_l72_72416


namespace simplify_expression_I_simplify_expression_II_l72_72894

-- Problem I: Simplify √(𝑎^(1/4)) ⋅ √(𝑎 ⋅ √𝑎) = √𝑎
theorem simplify_expression_I (a : ℝ) (h : 0 < a) : sqrt (a ^ (1 / 4)) * sqrt (a * sqrt a) = sqrt a :=
by
  sorry

-- Problem II: Simplify log2(3) ⋅ log3(5) ⋅ log5(4) = 2
theorem simplify_expression_II : log 3 / log 2 * log 5 / log 3 * log 4 / log 5 = 2 :=
by
  sorry

end simplify_expression_I_simplify_expression_II_l72_72894


namespace teddy_bear_cost_l72_72856

-- Definitions for the given conditions
def num_toys : ℕ := 28
def toy_price : ℕ := 10
def num_teddy_bears : ℕ := 20
def total_money : ℕ := 580

-- The theorem we want to prove
theorem teddy_bear_cost :
  (num_teddy_bears * 15 + num_toys * toy_price = total_money) :=
by
  sorry

end teddy_bear_cost_l72_72856


namespace equilateral_division_l72_72288

theorem equilateral_division (k : ℕ) :
  (k = 1 ∨ k = 3 ∨ k = 4 ∨ k = 9 ∨ k = 12 ∨ k = 36) ↔
  (k ∣ 36 ∧ ¬ (k = 2 ∨ k = 6 ∨ k = 18)) := by
  sorry

end equilateral_division_l72_72288


namespace smallest_m_for_integral_roots_l72_72984

theorem smallest_m_for_integral_roots :
  ∃ m : ℕ, (∀ x : ℚ, 12 * x^2 - m * x + 360 = 0 → x.den = 1) ∧ 
           (∀ k : ℕ, k < m → ¬∀ x : ℚ, 12 * x^2 - k * x + 360 = 0 → x.den = 1) :=  
begin
  sorry
end

end smallest_m_for_integral_roots_l72_72984


namespace inequality_not_always_correct_l72_72038

variable (x y z : ℝ)

-- Conditions
axiom Hx_pos : x > 0
axiom Hy_pos : y > 0
axiom Hx_gt_y : x > y
axiom Hz_pos : z > 0

-- Statement to prove: inequality (D) is not always correct
theorem inequality_not_always_correct : 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ x > y ∧ z > 0 ∧ (abs((x / z) - (y / z)) ≠ (x - y) / z)) :=
by {
  use [x, y, z],
  split; [exact Hx_pos, split; [exact Hy_pos, split; [exact Hx_gt_y, split; [exact Hz_pos, sorry]]]]
}

end inequality_not_always_correct_l72_72038


namespace triangle_area_is_64_l72_72592

/-- Define the vertices of the triangle --/
def vertex_A : ℝ × ℝ := (8, 8)
def vertex_B : ℝ × ℝ := (-8, 8)
def origin : ℝ × ℝ := (0, 0)

/-- Define the computation for the area of the triangle --/
noncomputable def triangle_area (A B O : ℝ × ℝ) : ℝ :=
  by
    let base := dist A B
    let height := (A.snd - O.snd).abs
    exact (1 / 2) * base * height

/-- The area of the triangle bounded by the lines y = x, y = -x, and y = 8 is 64 --/
theorem triangle_area_is_64 : triangle_area vertex_A vertex_B origin = 64 := by
  sorry

end triangle_area_is_64_l72_72592


namespace find_monic_polynomial_l72_72636

-- Define the original polynomial
def polynomial_1 (x : ℝ) := x^3 - 4 * x^2 + 9

-- Define the monic polynomial we are seeking
def polynomial_2 (x : ℝ) := x^3 - 12 * x^2 + 243

theorem find_monic_polynomial :
  ∀ (r1 r2 r3 : ℝ), 
    polynomial_1 r1 = 0 → 
    polynomial_1 r2 = 0 → 
    polynomial_1 r3 = 0 → 
    polynomial_2 (3 * r1) = 0 ∧ polynomial_2 (3 * r2) = 0 ∧ polynomial_2 (3 * r3) = 0 :=
by
  intros r1 r2 r3 h1 h2 h3
  sorry

end find_monic_polynomial_l72_72636


namespace placemat_length_is_correct_l72_72263

-- Define the conditions
def side_length_hexagon := 5
def number_of_placemats := 8
def width_placemat := 2
def perimeter_hexagon := 6 * side_length_hexagon

-- Define the length y of each placemat (to be proved)
def length_of_placemat := perimeter_hexagon / number_of_placemats

-- Prove that the length of each placemat is 3.75 units
theorem placemat_length_is_correct : length_of_placemat = 3.75 := by
  sorry

end placemat_length_is_correct_l72_72263


namespace angle_PQR_measure_l72_72439

theorem angle_PQR_measure (R S P Q : Type*) 
  (RSP_line : exists A : Type*, A = R ∨ A = S ∨ A = P)
  (angle_QSP : ∠ Q S P = 70)
  (RS_neq_SQ : R ≠ S)
  (PS_neq_SQ : P ≠ S) :
  ∠ P Q R = 170 := 
sorry

end angle_PQR_measure_l72_72439


namespace find_a₈_l72_72010

noncomputable def a₃ : ℝ := -11 / 6
noncomputable def a₅ : ℝ := -13 / 7

theorem find_a₈ (h : ∃ d : ℝ, ∀ n : ℕ, (1 / (a₃ + 2)) + (n-2) * d = (1 / (a_n + 2)))
  : a_n = -32 / 17 := sorry

end find_a₈_l72_72010


namespace min_area_triangle_DEF_is_correct_l72_72946

-- Define the conditions and the problem in Lean 4
noncomputable def least_area_triangle_DEF : ℝ :=
  let z := 5 + 2 * Complex.cis (2 * Real.pi * 0 / 12);
  let w := 5 + 2 * Complex.cis (2 * Real.pi * 1 / 12);
  let u := 5 + 2 * Complex.cis (2 * Real.pi * 2 / 12);
  let D := Complex.real_part z;
  let E := Complex.real_part w + Complex.imag_part w * Complex.I;
  let F := Complex.real_part u + Complex.imag_part u * Complex.I;
  let base := Complex.abs (E - D);
  let height := Complex.abs ((F - D) - (F - E)) / 2;
  
  base * height / 2

theorem min_area_triangle_DEF_is_correct :
  least_area_triangle_DEF = (sqrt(5 - 4 * sqrt(3)) * sqrt(3)) / 4 :=
sorry

end min_area_triangle_DEF_is_correct_l72_72946


namespace boxes_tickets_l72_72400

theorem boxes_tickets (tickets_per_box : ℕ) (total_tickets : ℕ) (boxes : ℕ) : 
  (tickets_per_box = 5 ∧ total_tickets = 45) → boxes = total_tickets / tickets_per_box :=
by {
  intros h,
  cases h with tickets_per_box_eq total_tickets_eq,
  rw [tickets_per_box_eq, total_tickets_eq],
  norm_num,
  exact boxes,
  sorry,
}

end boxes_tickets_l72_72400


namespace sum_squares_mod_13_l72_72606

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 10 := by
  sorry

end sum_squares_mod_13_l72_72606


namespace packages_count_l72_72223

/-- White t-shirts can be purchased in packages of 13. -/
def t_shirts_per_package : Nat := 13

/-- Mom buys 39 white t-shirts. -/
def total_t_shirts : Nat := 39

/-- The number of packages mom will have. -/
def packages : Nat := total_t_shirts / t_shirts_per_package

/-- Proof that the number of packages is 3 given the conditions. -/
theorem packages_count : packages = 3 := by
  unfold packages
  rw [total_t_shirts, t_shirts_per_package]
  sorry

end packages_count_l72_72223


namespace range_of_m_l72_72027

   theorem range_of_m (m : ℝ) :
     (∀ x : ℝ, ((x - 1) ^ 2 < 1) → (x ∈ set.Ioo (1 - m) (1 + m))) →
     (m ∈ set.Ioo 0 1) :=
   by
     intros h
     sorry
   
end range_of_m_l72_72027


namespace mouse_distance_furthest_point_l72_72656

noncomputable def chocolate : ℝ × ℝ := (14, 13)
noncomputable def mouse_initial : ℝ × ℝ := (2, -5)
noncomputable def mouse_path (x : ℝ) : ℝ := -4 * x + 6

noncomputable def perpendicular_to_path (x : ℝ) : ℝ := (1/4) * x + (45/2)

theorem mouse_distance_furthest_point :
  let a := -66 / 17 in
  let b := 366 / 17 in
  (perpendicular_to_path a = -4 * a + 6) ∧ (a + b = 300 / 17) :=
by
  sorry

end mouse_distance_furthest_point_l72_72656


namespace sally_seashells_l72_72580

theorem sally_seashells (T S: ℕ) (hT : T = 37) (h_total : T + S = 50) : S = 13 := by
  -- Skip the proof
  sorry

end sally_seashells_l72_72580


namespace water_level_equilibrium_l72_72203

noncomputable def h_initial : ℝ := 40
noncomputable def rho_water : ℝ := 1000
noncomputable def rho_oil : ℝ := 700

-- The mathematical problem is proving the final water level (h1) is approximately 16.47 cm
theorem water_level_equilibrium :
  ∃ h_1 h_2: ℝ, (rho_water * h_1 = rho_oil * h_2) ∧ (h_1 + h_2 = h_initial) ∧ (h_1 ≈ 16.47) :=
begin
  sorry
end

end water_level_equilibrium_l72_72203


namespace frank_remaining_money_l72_72342

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l72_72342


namespace smallest_m_for_integral_roots_l72_72982

theorem smallest_m_for_integral_roots :
  ∃ (m : ℕ), (∃ (p q : ℤ), p * q = 30 ∧ m = 12 * (p + q)) ∧ m = 132 := by
  sorry

end smallest_m_for_integral_roots_l72_72982


namespace sphere_radius_condition_l72_72570

theorem sphere_radius_condition {r : ℝ} (h : (4 * real.pi * r^3) / 3 = 8 * real.pi * r^2) : r = 6 := 
sorry

end sphere_radius_condition_l72_72570


namespace correct_operation_l72_72221

variable (a b : ℝ)

-- Conditions from the problem
def option_A : Prop := (2 * a + 3 * b = 5 * a * b)
def option_B : Prop := ((a - b)^2 = a^2 - b^2)
def option_C : Prop := ((a * b^2)^3 = a^3 * b^5)
def option_D : Prop := (3 * a^3 * (-4 * a^2) = -12 * a^5)

-- Prove that option D is correct and others are incorrect
theorem correct_operation : ¬option_A ∧ ¬option_B ∧ ¬option_C ∧ option_D :=
by
  sorry

end correct_operation_l72_72221


namespace same_percentage_loss_as_profit_l72_72560

theorem same_percentage_loss_as_profit (CP SP L : ℝ) (h_prof : SP = 1720)
  (h_loss : L = CP - (14.67 / 100) * CP)
  (h_25_prof : 1.25 * CP = 1875) :
  L = 1280 := 
  sorry

end same_percentage_loss_as_profit_l72_72560


namespace shell_placements_l72_72452

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem shell_placements : factorial 14 / 7 = 10480142147302400 := by
  sorry

end shell_placements_l72_72452


namespace line_intersects_curve_slope_when_intersects_l72_72029

noncomputable def polar_to_cartesian (θ : ℝ) : ℝ × ℝ := 
let ρ := 2 * cos θ - 4 * sin θ in (ρ * cos θ, ρ * sin θ)

def curve_C (p : ℝ × ℝ) : Prop :=
let (x, y) := p in (x - 1)^2 + (y + 2)^2 = 5

def line_l (t α : ℝ) : ℝ × ℝ := 
(1 + t * cos α, -1 + t * sin α)

def intersects (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) : Prop :=
∃ p : ℝ × ℝ, l p ∧ C p  

def distance (A B : ℝ × ℝ) : ℝ := 
let (x1, y1) := A in let (x2, y2) := B in 
real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def slope_of_l (α : ℝ) : ℝ := sin α / cos α

theorem line_intersects_curve (α : ℝ) :
  (∃ t : ℝ, curve_C (line_l t α)) → True :=
sorry -- proof is not required, so using sorry

theorem slope_when_intersects (A B : ℝ × ℝ) (α : ℝ) (h : distance A B = 3 * real.sqrt 2) :
  slope_of_l α = 1 ∨ slope_of_l α = -1 :=
sorry -- proof is not required, so using sorry

end line_intersects_curve_slope_when_intersects_l72_72029


namespace probability_drop_l72_72422

open Real

noncomputable def probability_of_oil_drop_falling_in_hole (c : ℝ) : ℝ :=
  (0.25 * c^2) / (π * (c^2 / 4))

theorem probability_drop (c : ℝ) (hc : c > 0) : 
  probability_of_oil_drop_falling_in_hole c = 0.25 / π :=
by
  sorry

end probability_drop_l72_72422


namespace product_not_48_l72_72222

theorem product_not_48 (a b : ℝ) (h : a = -1/2 ∧ b = 96) : a * b ≠ 48 :=
by
  cases h with ha hb
  rw [ha, hb]
  norm_num
  sorry

end product_not_48_l72_72222


namespace find_derivative_value_l72_72753

theorem find_derivative_value (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x ^ 2 + 2 * x * (f' 2)) :
  f' 3 = -6 :=
by
  sorry

end find_derivative_value_l72_72753


namespace num_subsets_of_P_eq_four_l72_72394

theorem num_subsets_of_P_eq_four : 
  let P := {1, 2} in 
  set.finite (P : set ℕ) ∧ set.countable (P : set ℕ) ∧ (finset.powerset P).card = 4 := 
by 
  sorry

end num_subsets_of_P_eq_four_l72_72394


namespace odd_function_iff_phi_eq_pi_div_4_l72_72004

def f (x φ : ℝ) : ℝ := sin (x - φ) + cos (x - φ)

theorem odd_function_iff_phi_eq_pi_div_4 :
  (∀ x : ℝ, f (-x) φ = -f x φ) ↔ φ = π / 4 :=
by
  sorry

end odd_function_iff_phi_eq_pi_div_4_l72_72004


namespace number_of_solutions_l72_72714

theorem number_of_solutions :
  (∃ (θ : ℝ), 0 < θ ∧ θ < 2 * real.pi ∧ tan (7 * real.pi * real.cos θ) = cot (7 * real.pi * real.sin θ)) →
  ∃ (n : ℕ), n = 36 :=
by
  sorry

end number_of_solutions_l72_72714


namespace eval_expression_l72_72702

theorem eval_expression (k : ℤ) : 2^(-(2*k + 2)) - 2^(-(2*k)) + 2^(-(2*k + 1)) + 2^(-(2*k - 1)) = (7 / 4) * 2^(-(2*k)) := 
sorry

end eval_expression_l72_72702


namespace triangle_area_bounded_by_lines_l72_72590

theorem triangle_area_bounded_by_lines : 
  let A := (8, 8)
      B := (-8, 8)
      O := (0, 0)
  in
  let base_length : ℝ := 16
  let height : ℝ := 8
  let area : ℝ := (1 / 2) * base_length * height
  in
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l72_72590


namespace trig_identity_l72_72167

theorem trig_identity (x : ℝ) : 
  let a := 4
      b := 8
      c := 4
      d := 2
  in 
  (cos (2 * x) + cos (6 * x) + cos (10 * x) + cos (14 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) ∧
  (a + b + c + d = 18) :=
by
  let a := 4
  let b := 8
  let c := 4
  let d := 2
  sorry

end trig_identity_l72_72167


namespace diagonal_length_of_rhombus_l72_72158

-- Definitions for the conditions
def side_length_of_square : ℝ := 8
def area_of_square : ℝ := side_length_of_square ^ 2
def area_of_rhombus : ℝ := 64
def d2 : ℝ := 8
-- Question
theorem diagonal_length_of_rhombus (d1 : ℝ) : (d1 * d2) / 2 = area_of_rhombus ↔ d1 = 16 := by
  sorry

end diagonal_length_of_rhombus_l72_72158


namespace value_of_a9_l72_72494

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem value_of_a9 (a_n S_n : ℝ) (a1 d : ℝ) :
  a_n a1 d 3 + a_n a1 d 6 = 12 ∧ S_n a1 d 4 = 8 → a_n a1 d 9 = 15 :=
by
  sorry

end value_of_a9_l72_72494


namespace cos_double_angle_l72_72752

variables {θ : ℝ}

-- Define vectors AB and BC
def vector_AB : ℝ × ℝ := (-1, -3)
def vector_BC : ℝ × ℝ := (2 * Real.sin θ, 2)

-- Define the collinearity condition
def collinear (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Define the problem statement
theorem cos_double_angle :
  collinear vector_AB vector_BC →
  Real.cos (2 * θ) = 7 / 9 :=
begin
  assume h_collinear,
  sorry
end

end cos_double_angle_l72_72752


namespace sum_of_arithmetic_sequence_l72_72047

-- Given conditions in the problem
axiom arithmetic_sequence (a : ℕ → ℤ): Prop
axiom are_roots (a b : ℤ): ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a = p ∨ a = q) ∧ (b = p ∨ b = q)

-- The equivalent proof problem statement
theorem sum_of_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a 2 = p ∨ a 2 = q) ∧ (a 11 = p ∨ a 11 = q)):
  a 5 + a 8 = 3 :=
sorry

end sum_of_arithmetic_sequence_l72_72047


namespace quadrilateral_equivalent_conditions_l72_72917

variable {AB CD BC AD BK BL DK DL AK CL AL CK : ℝ}
variable {A B C D K L : Type} [has_dist A B] [has_dist B C] [has_dist C D] [has_dist D A] [has_dist B L] [has_dist D K]

theorem quadrilateral_equivalent_conditions 
  (h1: AB + CD = BC + AD ∨ BK + BL = DK + DL ∨ AK + CL = AL + CK):
  (AB + CD = BC + AD ∧ BK + BL = DK + DL ∧ AK + CL = AL + CK) := 
sorry

end quadrilateral_equivalent_conditions_l72_72917


namespace range_of_k_l72_72382

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -x^2 + (1 / 2) * x else Real.exp x - 1

theorem range_of_k :
  {k : ℝ | ∀ x, f(x) - k * x = 0 → x = 0 ∨ x = 1 ∨ x = 2} = (1, +∞) := 
sorry

end range_of_k_l72_72382


namespace part1_part2_part3_l72_72734

-- Define the sequences a_n and b_n as described in the problem
def X_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → (a n = 0 ∨ a n = 1))

def accompanying_sequence (a b : ℕ → ℝ) : Prop :=
  (b 1 = 1) ∧ (∀ n : ℕ, n > 0 → b (n + 1) = abs (a n - (a (n + 1) / 2)) * b n)

-- 1. Prove the values of b_2, b_3, and b_4
theorem part1 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  a 2 = 1 → a 3 = 0 → a 4 = 1 →
  b 2 = 1 / 2 ∧ b 3 = 1 / 2 ∧ b 4 = 1 / 4 := 
sorry

-- 2. Prove the equivalence for geometric sequence and constant sequence
theorem part2 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  (∀ n : ℕ, n > 0 → a n = 1) ↔ (∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) = r * b n) := 
sorry

-- 3. Prove the maximum value of b_2019
theorem part3 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  b 2019 ≤ 1 / 2^1009 := 
sorry

end part1_part2_part3_l72_72734


namespace pure_imaginary_number_l72_72118

open Complex -- Use the Complex module for complex numbers

theorem pure_imaginary_number (a : ℝ) (h : (a - 1 : ℂ).re = 0) : a = 1 :=
by
  -- This part of the proof is omitted hence we put sorry
  sorry

end pure_imaginary_number_l72_72118


namespace sum_squares_of_roots_of_quadratic_l72_72334

theorem sum_squares_of_roots_of_quadratic:
  ∀ (s_1 s_2 : ℝ),
  (s_1 + s_2 = 20) ∧ (s_1 * s_2 = 32) →
  (s_1^2 + s_2^2 = 336) :=
by
  intros s_1 s_2 h
  sorry

end sum_squares_of_roots_of_quadratic_l72_72334


namespace intersecting_lines_at_BC_l72_72359

theorem intersecting_lines_at_BC 
  {A B C D E F G : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]
  (triangle_ABC : Triangle A B C)
  (h1 : abs (dist B A) + abs (dist C A) = 3 * abs (dist B C))
  (parallelogram_BCDA : Parallelogram B C D A)
  (parallelogram_CBEA : Parallelogram C B E A)
  (F_on_AC : Point F ∈ LineSegment A C)
  (G_on_AB : Point G ∈ LineSegment A B)
  (h2 : dist A F = dist A G = dist B C) :
  ∃ Z : Point, Z ∈ LineSegment B C ∧ IntersectingLines (Line D F) (Line E G) Z :=
sorry

end intersecting_lines_at_BC_l72_72359


namespace evaluate_expression_l72_72901

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

axiom g_inverse : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x

axiom g_1_2 : g 1 = 2
axiom g_4_7 : g 4 = 7
axiom g_3_8 : g 3 = 8

theorem evaluate_expression :
  g_inv (g_inv 8 * g_inv 2) = 3 :=
by
  sorry

end evaluate_expression_l72_72901


namespace find_pairs_l72_72692

noncomputable def diamond (a b : ℝ) : ℝ :=
  a^2 * b^2 - a^3 * b - a * b^3

theorem find_pairs (x y : ℝ) :
  diamond x y = diamond y x ↔
  x = 0 ∨ y = 0 ∨ x = y ∨ x = -y :=
by
  sorry

end find_pairs_l72_72692


namespace towel_price_l72_72293

theorem towel_price (x : ℝ)
    (guest_set_count : ℝ = 2)
    (guest_set_price : ℝ = 40)
    (master_set_count : ℝ = 4)
    (discount : ℝ = 0.20)
    (total_spending : ℝ = 224) :
    x = 50 :=
by
  sorry

end towel_price_l72_72293


namespace friends_travelled_distance_l72_72854

theorem friends_travelled_distance :
  let lionel_distance : ℝ := 4 * 5280
  let esther_distance : ℝ := 975 * 3
  let niklaus_distance : ℝ := 1287
  let isabella_distance : ℝ := 18 * 1000 * 3.28084
  let sebastian_distance : ℝ := 2400 * 3.28084
  let total_distance := lionel_distance + esther_distance + niklaus_distance + isabella_distance + sebastian_distance
  total_distance = 91261.136 := 
by
  sorry

end friends_travelled_distance_l72_72854


namespace length_MN_constant_circle_intersects_directrix_l72_72747

-- Definitions for the given conditions
def fixed_point_A (p : ℝ) (h : p > 0) : ℝ × ℝ := (0, p)

def parabola (p : ℝ) : set (ℝ × ℝ) := {O' | ∃ x₀ y₀, O' = (x₀, y₀) ∧ x₀^2 = 2*p*y₀}

def circle (O' : ℝ × ℝ) (p : ℝ) : set (ℝ × ℝ) :=
  {P | ∃ x₀ y₀, O' = (x₀, y₀) ∧ (P.1 - x₀)^2 + (P.2 - y₀)^2 = x₀^2 + (y₀ - p)^2}

def chord_x_axis (O' : ℝ × ℝ) (p : ℝ) : ℝ × ℝ := 
  let x₀ := O'.1 in 
  (x₀ - p, 0), (x₀ + p, 0)

-- Lean 4 statements for Part (1)
theorem length_MN_constant (p : ℝ) (h : p > 0) (O' : ℝ × ℝ) (hO' : O' ∈ parabola p) :
  let (M, N) := chord_x_axis O' p in 
  |M.1 - N.1| = 2 * p := by 
  sorry

-- Lean 4 statements for Part (2)
theorem circle_intersects_directrix (p : ℝ) (h : p > 0) (O' : ℝ × ℝ) (hO' : O' ∈ parabola p) :
  let AA : ℝ × ℝ := fixed_point_A p h in 
  let directrix : set (ℝ × ℝ) := {P | P.2 = -p / 2} in 
  let circle_set := circle O' p in 
  ∃ P ∈ circle_set, P ∈ directrix := by 
  sorry

end length_MN_constant_circle_intersects_directrix_l72_72747


namespace smallest_m_for_integral_roots_l72_72985

theorem smallest_m_for_integral_roots :
  ∃ m : ℕ, (∀ x : ℚ, 12 * x^2 - m * x + 360 = 0 → x.den = 1) ∧ 
           (∀ k : ℕ, k < m → ¬∀ x : ℚ, 12 * x^2 - k * x + 360 = 0 → x.den = 1) :=  
begin
  sorry
end

end smallest_m_for_integral_roots_l72_72985


namespace sum_of_squares_mod_13_l72_72603

theorem sum_of_squares_mod_13 :
  ((∑ i in Finset.range 16, i^2) % 13) = 3 :=
by 
  sorry

end sum_of_squares_mod_13_l72_72603


namespace slope_of_tangent_at_point_l72_72716

variables (a b x0 y0 : ℝ)
-- Condition 1: The ellipse equation evaluated at (x0, y0)
axiom ellipse_at_point : (x0^2 / a^2 + y0^2 / b^2 = 1)
-- Condition 2: y0 is not zero
axiom y0_nonzero : y0 ≠ 0

-- The theorem to be proved
theorem slope_of_tangent_at_point :
  ∀ a b x0 y0 : ℝ, (x0^2 / a^2 + y0^2 / b^2 = 1) → y0 ≠ 0 →
  (deriv (λ y x, x^2 / a^2 + y^2 / b^2) (x0, y0) = -b^2 * x0 / (a^2 * y0)) := by
sorry

end slope_of_tangent_at_point_l72_72716


namespace area_ADFE_proof_l72_72674

-- Definitions of the geometrical setup and the areas of the triangles.
variables {A B C D E F : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

-- Triangles and their areas.
def triangle_CDF (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Real := 3
def triangle_BFE (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Real := 4
def triangle_BCF (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Real := 5

-- The problem statement: Proving the area of quadrilateral ADFE
theorem area_ADFE_proof :
  let area_ADFE (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Real := 3 + 4 + 5 - triangle_CDF A B C D E F - triangle_BFE A B C D E F in
  area_ADFE A B C D E F = 204 / 13 := 
begin
  sorry
end

end area_ADFE_proof_l72_72674


namespace proof_1_proof_2_l72_72299

theorem proof_1 : 364 = 4 :=
sorry

theorem proof_2 : 4 ^ (Real.log 3 / Real.log 2) = 9 :=
sorry

end proof_1_proof_2_l72_72299


namespace find_original_length_l72_72852
noncomputable def original_rectangle_length (x : ℝ) (d : ℝ) : ℝ :=
  let orig_length := 2 * x
  let orig_breadth := x
  let new_length := 2 * x - 5
  let new_breadth := x + 4
  let orig_area := orig_length * orig_breadth
  let new_area := new_length * new_breadth
  have area_increase_h : new_area = orig_area + 75 := sorry
  have diag_relation_h : d = x * Real.sqrt 5 := sorry
  have x_val : x = 95 / 3 := sorry
  orig_length

theorem find_original_length : 
  ∃ x : ℝ, ∃ d : ℝ, original_rectangle_length x d = 190 / 3 :=
by {
  use 95 / 3,
  use (95 / 3) * Real.sqrt 5,
  rw original_rectangle_length,
  sorry
}

end find_original_length_l72_72852


namespace find_initial_price_student_ticket_l72_72535

-- Definitions for the problem
def initial_price_conditions (S T A : ℝ) : Prop :=
  (4 * S + 3 * T + 2 * A = 120) ∧
  (9 * S + 8 * T + 5 * A = 360) ∧
  (15 * S + 12 * T + 8 * A = 587) ∧
  (7 * (S - 3) + 10 * (T - 2) + 6 * (A - 4) = 300) ∧
  (8 * (S + 4) + 6 * (T + 3) + 4 * (A + 5) = 257)

theorem find_initial_price_student_ticket :
  ∀ (S T A : ℝ), initial_price_conditions S T A → abs (T - 8.83) < 0.01 :=
by
  intros S T A conds
  -- This is where the steps of the solution would go
  left sorry -- we reaffirm the goal has the correctly specified components

end find_initial_price_student_ticket_l72_72535


namespace sum_squares_mod_13_l72_72617

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 5 :=
by sorry

end sum_squares_mod_13_l72_72617


namespace domain_of_f_l72_72164

def f (x : ℝ) : ℝ := 3 * x^2 / sqrt (1 - x) + log (3 * x + 1)

theorem domain_of_f :
  {x : ℝ | 1 - x > 0 ∧ 3 * x + 1 > 0} = { x : ℝ | -1/3 < x ∧ x < 1 } :=
by
  ext x
  simp
  split
  { intro h
    cases h with h1 h2
    split
    { linarith }
    { linarith } }
  { intro h
    cases h with h1 h2
    split
    { linarith }
    { linarith } }

end domain_of_f_l72_72164


namespace decimal_expansion_period_mod_eq_one_l72_72107

theorem decimal_expansion_period_mod_eq_one (p : ℕ) (r : ℕ) 
  (hp : prime p) (h_gt_5 : p > 5)
  (h_period : ∃ a : ℕ, 1 / p = 0.a1a2 ⋯ a r ∘. ) :
    10^r ≡ 1 [MOD p] :=
begin
  sorry
end

end decimal_expansion_period_mod_eq_one_l72_72107


namespace frank_money_remaining_l72_72345

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l72_72345


namespace trapezoid_area_l72_72081

open Real

theorem trapezoid_area 
  (r : ℝ) (BM CD AB : ℝ) (radius_nonneg : 0 ≤ r) 
  (BM_positive : 0 < BM) (CD_positive : 0 < CD) (AB_positive : 0 < AB)
  (circle_radius : r = 4) (BM_length : BM = 16) (CD_length : CD = 3) :
  let height := 2 * r
  let base_sum := AB + CD
  let area := height * base_sum / 2
  AB = BM + 8 → area = 108 :=
by
  intro hyp
  sorry

end trapezoid_area_l72_72081


namespace range_of_m_l72_72919

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_m:
  ∀ m : ℝ, 
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ -3) ∧ 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ f x = -4) → 
  1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l72_72919


namespace find_p_no_linear_term_l72_72791

theorem find_p_no_linear_term (p : ℝ) :
  let expansion := (x - 3) * (x^2 + p * x - 1) in
  (∃ (x : ℝ), true) → -- To introduce x
  (∀ x : ℝ, (expansion : ℝ[x]) - x * 0 = 0) →
  p = -1/3 := by
  sorry

end find_p_no_linear_term_l72_72791


namespace max_value_of_x2_plus_y2_l72_72481

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + y^2

theorem max_value_of_x2_plus_y2 {x y : ℝ} (h : 5*x^2 + 4*y^2 = 10*x) : max_value x y ≤ 4 := sorry

end max_value_of_x2_plus_y2_l72_72481


namespace g_at_4_l72_72155

def f (x : ℝ) : ℝ := 4 / (3 - x)
def f_inv (x : ℝ) : ℝ := 3 - 4 / x
def g (x : ℝ) : ℝ := 1 / f_inv(x) + 7

theorem g_at_4 : g 4 = 7.5 := by
  sorry

end g_at_4_l72_72155


namespace adam_completes_remaining_work_in_10_days_l72_72859

variables (W : ℝ) -- total work

-- Conditions
def michael_days_to_complete_work := 40
def michael_work_rate := W / michael_days_to_complete_work
def combined_work_rate := W / 20
def worked_days := 15

-- Deduce Adam's work rate
def adam_work_rate := combined_work_rate - michael_work_rate

-- remaining work after Michael and Adam work together for 15 days
def remaining_work := W - worked_days * combined_work_rate

-- Final problem statement
theorem adam_completes_remaining_work_in_10_days :
  (remaining_work / adam_work_rate) = 10 := by
  sorry

end adam_completes_remaining_work_in_10_days_l72_72859


namespace smallest_n_l72_72148

def in_interval (x y z : ℝ) (n : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ n ∧ 2 ≤ y ∧ y ≤ n ∧ 2 ≤ z ∧ z ≤ n

def no_two_within_one_unit (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 1 ∧ abs (y - z) ≥ 1 ∧ abs (z - x) ≥ 1

def more_than_two_units_apart (x y z : ℝ) (n : ℕ) : Prop :=
  x > 2 ∧ x < n - 2 ∧ y > 2 ∧ y < n - 2 ∧ z > 2 ∧ z < n - 2

def probability_condition (n : ℕ) : Prop :=
  (n-4)^3 / (n-2)^3 > 1/3

theorem smallest_n (n : ℕ) : 11 = n → (∃ x y z : ℝ, in_interval x y z n ∧ no_two_within_one_unit x y z ∧ more_than_two_units_apart x y z n ∧ probability_condition n) :=
by
  sorry

end smallest_n_l72_72148


namespace bill_profit_difference_l72_72295

theorem bill_profit_difference 
  (SP : ℝ) 
  (hSP : SP = 1.10 * (SP / 1.10)) 
  (hSP_val : SP = 989.9999999999992) 
  (NP : ℝ) 
  (hNP : NP = 0.90 * (SP / 1.10)) 
  (NSP : ℝ) 
  (hNSP : NSP = 1.30 * NP) 
  : NSP - SP = 63.0000000000008 := 
by 
  sorry

end bill_profit_difference_l72_72295


namespace xyz_cubic_expression_l72_72768

theorem xyz_cubic_expression (x y z a b c : ℝ) (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0) (h7 : a ≠ 0) (h8 : b ≠ 0) (h9 : c ≠ 0) :
  x^3 + y^3 + z^3 = (a^3 + b^3 + c^3) / (a * b * c) :=
by
  sorry

end xyz_cubic_expression_l72_72768


namespace sum_of_first_5_terms_l72_72181

noncomputable def seq (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n a_n, a_n + 3)

theorem sum_of_first_5_terms : (seq 1) + (seq 2) + (seq 3) + (seq 4) + (seq 5) = 35 :=
by
  sorry

end sum_of_first_5_terms_l72_72181


namespace cos_A_value_l72_72820

namespace TriangleProof

-- Define the conditions from part (a)
variables {A B C : ℝ} -- Angles
variables {a b c S : ℝ} -- Sides and area

-- Given conditions
axiom sides_angles : ∀ {a b c A B C : ℝ}, a ≠ 0 → b ≠ 0 → c ≠ 0 → A + B + C = π
axiom area_condition : (a^2 + b^2) * Real.tan C = 8 * S
axiom sine_cosine_condition : Real.sin A * Real.cos B = 2 * Real.cos A * Real.sin B

-- Main theorem to be proven
theorem cos_A_value : Real.cos A = (Real.sqrt 30) / 15 := by
  sorry

end TriangleProof

end cos_A_value_l72_72820


namespace mirasol_account_balance_l72_72125

theorem mirasol_account_balance :
  ∀ (initial_amount spent_coffee spent_tumbler : ℕ), 
  initial_amount = 50 → 
  spent_coffee = 10 → 
  spent_tumbler = 30 → 
  initial_amount - (spent_coffee + spent_tumbler) = 10 :=
by
  intros initial_amount spent_coffee spent_tumbler
  intro h_initial_amount
  intro h_spent_coffee
  intro h_spent_tumbler
  rw [h_initial_amount, h_spent_coffee, h_spent_tumbler]
  simp
  done

end mirasol_account_balance_l72_72125


namespace old_stereo_cost_250_l72_72096
noncomputable def original_cost_old_stereo (X : ℝ) : Prop :=
  let trade_in_value := 0.80 * X in
  let new_system_cost := 600.0 in
  let discount := 0.25 * new_system_cost in
  let discounted_price := new_system_cost - discount in
  let out_of_pocket := 250 in
  trade_in_value + out_of_pocket = discounted_price

theorem old_stereo_cost_250 : ∃ X, original_cost_old_stereo X ∧ X = 250 :=
by {
  use 250,
  sorry
}

end old_stereo_cost_250_l72_72096


namespace number_of_bottle_caps_l72_72098

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end number_of_bottle_caps_l72_72098


namespace first_1200_positive_integers_expressible_count_l72_72404

def g (x : ℝ) : ℤ := ⌊3 * x⌋ + ⌊6 * x⌋ + ⌊9 * x⌋ + ⌊12 * x⌋

theorem first_1200_positive_integers_expressible_count :
  (∃(count : ℕ), count = 1200 ∧ (∃s : set ℕ, s ⊆ {i | 1 ≤ i ∧ i ≤ 1200} ∧ s.card = count ∧ 
   (∀ n ∈ s, ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ g(x) = n))) ∧ count * 26 / 30 = 1040 := 
sorry

end first_1200_positive_integers_expressible_count_l72_72404


namespace common_difference_sequence_sum_proof_l72_72365

variable {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom arithmetic_sequence (d : ℕ) : ∀ n, a (n + 1) = a n + d
axiom sum_of_first_n_terms : ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2
axiom given_condition : S 4 = 2 * S 2 + 8

-- Part I: Prove the common difference d
theorem common_difference : (d = 2) :=
by 
  sorry

-- Part II: Prove the sum of the first n terms of the sequence { 1/(a_n a_(n+1)) }
noncomputable def a_n (n : ℕ) := 1 + (n - 1) * 2
noncomputable def seq_term (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))
noncomputable def sequence_sum (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (2 * n + 1))

theorem sequence_sum_proof (a1 : ℕ) (d : ℕ) (an : ℕ → ℕ) : 
  (∀n, a n = 1 + (n - 1) * 2) ∧ T n = ∑ k in range n, seq_term k :=
by 
  sorry

end common_difference_sequence_sum_proof_l72_72365


namespace f_of_2016_is_2017_l72_72415

-- Defining the conditions as given in the problem
def f : ℕ → ℕ := sorry
  
axiom h1 : ∀ n : ℕ, f(f(n)) + f(n) = 2 * n + 3
axiom h2 : f(0) = 1

-- Proving the equivalence of the conditions and the answer to the question
theorem f_of_2016_is_2017 : f(2016) = 2017 := by
  sorry

end f_of_2016_is_2017_l72_72415


namespace integral_difference_l72_72363

variables {f : ℝ → ℝ} 
variables {A B : ℝ}

-- Given conditions
def condition1 : Prop := ∫ x in 0..1, f x = A
def condition2 : Prop := ∫ x in 0..2, f x = B

-- Proof problem statement
theorem integral_difference (h1 : condition1) (h2 : condition2) : ∫ x in 1..2, f x = B - A := 
by
  sorry

end integral_difference_l72_72363


namespace derivative_value_l72_72015

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * f' 1

theorem derivative_value : (∀ x : ℝ, deriv f x = 2 * x + 3 * deriv f 1) → deriv f 1 = -1 := sorry

end derivative_value_l72_72015


namespace perimeter_triangle_PQS_l72_72814

variables (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S]

def is_isosceles (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] :=
  ∃ (PQ PR : ℝ), PQ = PR

def extended_point (R Q S : Type) [metric_space R] [metric_space Q] [metric_space S] :=
  ∃ (QR RS QS : ℝ), QS = QR + RS

theorem perimeter_triangle_PQS 
  (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
  (h1 : is_isosceles P Q R)
  (QR PR RS : ℝ)
  (h2 : QR = 8)
  (h3 : PR = 10)
  (h4 : RS = 4)
  (h5 : extended_point R Q S) :
  let PQ := PR,
      PS := PR,
      QS := QR + RS in
  PQ + PS + QS = 32 := 
by
  sorry

end perimeter_triangle_PQS_l72_72814


namespace truncated_cube_edges_l72_72321

theorem truncated_cube_edges :
  ∃ P : Polyhedron, (∃ c : Cube, truncate_each_vertex c = P) → P.edges = 16 :=
by
  sorry

end truncated_cube_edges_l72_72321


namespace value_of_PQRS_l72_72000

theorem value_of_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 :=
by
  sorry

end value_of_PQRS_l72_72000


namespace find_number_l72_72266

theorem find_number (N : ℕ) (h : N / 16 = 16 * 8) : N = 2048 :=
sorry

end find_number_l72_72266


namespace number_of_distinct_c_values_l72_72475

-- Define complex numbers and the distinct count function.
noncomputable def distinct_count {α : Type*} [DecidableEq α] (s : Set α) : Nat :=
  (s.toFinset.card : ℕ)

-- Define the Lean 4 proof statement
theorem number_of_distinct_c_values :
  ∃ r s t c : ℂ, r ≠ s ∧ s ≠ t ∧ t ≠ r ∧
  (∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2 * c * r) * (z - 2 * c * s) * (z - 2 * c * t)) →
  distinct_count {c : ℂ | ∃ r s t : ℂ, r ≠ s ∧ s ≠ t ∧ t ≠ r ∧
  (∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2 * c * r) * (z - 2 * c * s) * (z - 2 * c * t)) } = 3 :=
by
  sorry -- The proof is not required.

end number_of_distinct_c_values_l72_72475


namespace four_digit_multiples_of_4_not_5_l72_72042

theorem four_digit_multiples_of_4_not_5 :
  let
    -- four digit range
    lower_limit := 1000
    upper_limit := 9999
    -- multiples of 4
    first_multiple_of_4 := 1004
    last_multiple_of_4 := 9996
    n := (last_multiple_of_4 - first_multiple_of_4) / 4 + 1
    -- multiples of both 4 and 5 (i.e., multiples of 20)
    first_multiple_of_20 := 1020
    last_multiple_of_20 := 9980
    m := (last_multiple_of_20 - first_multiple_of_20) / 20 + 1
  in
    n - m = 114 :=
by
  sorry

end four_digit_multiples_of_4_not_5_l72_72042


namespace original_triangle_sides_l72_72563

noncomputable def semi_perimeter (a1 b1 c1 : ℝ) : ℝ :=
  (a1 + b1 + c1) / 2

noncomputable def cosine_alpha (s1 b1 c1 : ℝ) : ℝ :=
  real.sqrt ((s1 - b1) * (s1 - c1) / (b1 * c1))

theorem original_triangle_sides (a1 b1 c1 s1 : ℝ) (ha1 : a1 > 0) (hb1 : b1 > 0) (hc1 : c1 > 0) (hs1 : s1 = semi_perimeter a1 b1 c1) :
  a1 * real.sqrt (b1 * c1 / ((s1 - b1) * (s1 - c1))) > 0 :=
begin
  -- Proof omitted
  sorry
end

end original_triangle_sides_l72_72563


namespace drink_total_is_150_ounces_l72_72998

def percentage (part total : ℝ) : ℝ := (part / total) * 100

theorem drink_total_is_150_ounces :
  ∀ (T G O W : ℝ),
    O = 0.35 * T →
    W = 0.35 * T →
    G = 45 →
    G = 0.30 * T →
    T = 150 :=
by
  intros T G O W hO hW hG hG_eq
  sorry

end drink_total_is_150_ounces_l72_72998


namespace count_M_lt_500_with_3_values_of_k_l72_72778

theorem count_M_lt_500_with_3_values_of_k : 
  ∃ (M : list ℕ), (∀ m ∈ M, m < 500) ∧ (∀ m ∈ M, (∃ (k_values : list ℕ), (∀ k ∈ k_values, sum_of_k_odd_integers m k) ∧ (k_values.length = 3))) ∧ (M.length = 6) := 
sorry

end count_M_lt_500_with_3_values_of_k_l72_72778


namespace cost_of_crayon_l72_72506

theorem cost_of_crayon (cost_per_pack : ℝ) :
  let num_packs := 6 in
  (num_packs * cost_per_pack = 15) → (cost_per_pack = 2.5) :=
by
  intro h
  exact sorry

end cost_of_crayon_l72_72506


namespace player1_wins_n_eq_4_l72_72205

theorem player1_wins_n_eq_4 :
  ∀ (board : Fin 2018 × Fin 2018 → option ℕ) (n : ℕ), n = 4 →
  (∀ i j : Fin 2018, board (i, j) = some 1 → 
    ∃ i0 j0, board (i0, j0) = none ∧
    (∃ k, (k < 4 ∧ (board (i, j+k) = some 2 ∨ board (i+k, j) = some 2)))) →
  ∀ i j : Fin 2018, board (i, j) = some 2 →
  ∃ (winning_placement : Fin 2018 × Fin 2018), board winning_placement = some 1 ∧
  ∃ k, (k < 4 ∧ (board (winning_placement.1, winning_placement.2+k) = some 1 ∨ 
                   board (winning_placement.1+k, winning_placement.2) = some 1)) :=
sorry

end player1_wins_n_eq_4_l72_72205


namespace inequality_solution_l72_72122

noncomputable def solution_set {f : ℝ → ℝ} (h_diff : ∀ x, x < 0 → differentiable_at ℝ f x) 
  (h_deriv : ∀ x, x < 0 → deriv f x = f' x) (h_cond : ∀ x, x < 0 → 3 * f x + x * (f' x) > 0) : 
  set ℝ := {x | -2018 < x ∧ x < -2015}

theorem inequality_solution {f : ℝ → ℝ}
  (h_diff : ∀ x, x < 0 → differentiable_at ℝ f x)
  (h_deriv : ∀ x, x < 0 → deriv f x = f' x)
  (h_cond : ∀ x, x < 0 → 3 * f x + x * (f' x) > 0)
  (x : ℝ) :
  ((x + 2015) ^ 3 * f (x + 2015) + 27 * f (-3) > 0) ↔ (x ∈ solution_set h_diff h_deriv h_cond) :=
by sorry

end inequality_solution_l72_72122


namespace eccentricity_range_l72_72349

theorem eccentricity_range (a b : ℝ) (e1 e2 : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : e1 = (sqrt (a^2 - b^2)) / a) (h4 : e2 = (sqrt (a^2 - b^2)) / b) 
  (h5 : e1 * e2 < 1) : (sqrt 2) < (a / b) ∧ (a / b) < (1 + sqrt 5) / 2 :=
by
  sorry

end eccentricity_range_l72_72349


namespace like_terms_exponents_l72_72780

theorem like_terms_exponents (m n : ℤ) (h1 : 2 * n - 1 = m) (h2 : m = 3) : m = 3 ∧ n = 2 :=
by
  sorry

end like_terms_exponents_l72_72780


namespace youngest_string_cheese_per_day_l72_72824

theorem youngest_string_cheese_per_day :
  (total_days_per_week * total_weeks * oldest_string_cheese_per_day + youngest_string_cheese_per_weeks = 
  total_packages * cheese_per_package) →
  (youngest_string_cheese_per_weeks / total_days_per_week = youngest_string_cheese_per_day) →
  youngest_string_cheese_per_day = 1 :=
by
  sorry

-- Define constants for problem conditions
constant total_days_per_week : ℕ := 5
constant total_weeks : ℕ := 4
constant oldest_string_cheese_per_day : ℕ := 2
constant cheese_per_package : ℕ := 30
constant total_packages : ℕ := 2

-- Define the amount of string cheeses youngest eats per week and per day
constant youngest_string_cheese_per_weeks : ℕ
constant youngest_string_cheese_per_day : ℕ :=
  youngest_string_cheese_per_weeks / total_days_per_week

end youngest_string_cheese_per_day_l72_72824


namespace tina_final_balance_l72_72581

noncomputable def monthlyIncome : ℝ := 1000
noncomputable def juneBonusRate : ℝ := 0.1
noncomputable def investmentReturnRate : ℝ := 0.05
noncomputable def taxRate : ℝ := 0.1

-- Savings rates
noncomputable def juneSavingsRate : ℝ := 0.25
noncomputable def julySavingsRate : ℝ := 0.20
noncomputable def augustSavingsRate : ℝ := 0.30

-- Expenses
noncomputable def juneRent : ℝ := 200
noncomputable def juneGroceries : ℝ := 100
noncomputable def juneBookRate : ℝ := 0.05

noncomputable def julyRent : ℝ := 250
noncomputable def julyGroceries : ℝ := 150
noncomputable def julyShoesRate : ℝ := 0.15

noncomputable def augustRent : ℝ := 300
noncomputable def augustGroceries : ℝ := 175
noncomputable def augustMiscellaneousRate : ℝ := 0.1

theorem tina_final_balance :
  let juneIncome := monthlyIncome * (1 + juneBonusRate)
  let juneSavings := juneIncome * juneSavingsRate
  let juneExpenses := juneRent + juneGroceries + juneIncome * juneBookRate
  let juneRemaining := juneIncome - juneSavings - juneExpenses

  let julyIncome := monthlyIncome
  let julyInvestmentReturn := juneSavings * investmentReturnRate
  let julyTotalIncome := julyIncome + julyInvestmentReturn
  let julySavings := julyTotalIncome * julySavingsRate
  let julyExpenses := julyRent + julyGroceries + julyIncome * julyShoesRate
  let julyRemaining := julyTotalIncome - julySavings - julyExpenses

  let augustIncome := monthlyIncome
  let augustInvestmentReturn := julySavings * investmentReturnRate
  let augustTotalIncome := augustIncome + augustInvestmentReturn
  let augustSavings := augustTotalIncome * augustSavingsRate
  let augustExpenses := augustRent + augustGroceries + augustIncome * augustMiscellaneousRate
  let augustRemaining := augustTotalIncome - augustSavings - augustExpenses

  let totalInvestmentReturn := julyInvestmentReturn + augustInvestmentReturn
  let totalTaxOnInvestment := totalInvestmentReturn * taxRate

  let finalBalance := juneRemaining + julyRemaining + augustRemaining - totalTaxOnInvestment

  finalBalance = 860.7075 := by
  sorry

end tina_final_balance_l72_72581


namespace DF_perpendicular_G_l72_72070

open real euclidean_geometry 

section 
variables {A B C D E F G M N: Point}
variables [triangle A B C] (circumcircle_AC: Midpoint D A C) (circumcircle_BC: Midpoint E B C) 

-- Given Conditions
axiom h1 : is_acute_triangle A B C 
axiom h2 : length A B < length A C 
axiom h3 : midpoint M B C
axiom h4 : on_circumcircle D circumcircle_AC A C
axiom h5 : on_circumcircle E circumcircle_BC B C
axiom h6 : tangency_point_incircle F A B C In_circle_tri
axiom h7 : intersect G A E B C
axiom h8 : on_line_seg N E F ∧ perp NB A B
axiom h9 : length B N = length E M

-- To be proven
theorem DF_perpendicular_G : perp D F F G := 
sorry
end 

end DF_perpendicular_G_l72_72070


namespace sequence_formula_l72_72353

-- Definitions of the sequence and conditions
variable {n : ℕ+} -- definition of a positive natural number

def a (n : ℕ+) : ℝ := sqrt n - sqrt (n - 1)
def S (n : ℕ+) : ℝ := ∑ i in finset.range n, a ⟨i + 1, nat.succ_pos _⟩ -- sum of the first n terms of the sequence

-- Given conditions
axiom a_pos_seq : ∀ n : ℕ+, 0 < a n
axiom Sn_condition : ∀ n : ℕ+, 2 * S n = a n + (1 / a n)

-- The theorem to prove
theorem sequence_formula : ∀ n : ℕ+, a n = sqrt n - sqrt (n - 1) :=
sorry

end sequence_formula_l72_72353


namespace area_of_right_triangle_l72_72944

variable (a : ℝ) (r : ℝ) (R : ℝ)

-- Given conditions in Lean 4
def condition1 := R = (5 / 2) * r
def condition2 := ∃ b c : ℝ, b^2 + c^2 = (2 * R)^2 ∧ (a = b ∨ a = c)

-- Define equivalent theorem
theorem area_of_right_triangle (h1 : condition1) (h2 : condition2) :
  ∃ S : ℝ, (S = (√21 * a^2) / 6 ∨ S = (√19 * a^2) / 22) := by
  sorry

end area_of_right_triangle_l72_72944


namespace no_sunday_on_seventh_l72_72445

theorem no_sunday_on_seventh (x : ℕ) (h1 : 1 ≤ x ∧ x ≤ 7) : 
  ∃ k, k ∈ {x % 7, (x + 3) % 7, (x + 5) % 7, (x + 1) % 7, (x + 4) % 7, (x + 6) % 7, (x + 2) % 7} ∧ k = 0 :=
sorry

end no_sunday_on_seventh_l72_72445


namespace sum_of_arithmetic_sequence_l72_72160

variable (n : ℕ)
variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions: 
-- 1. The sequence {a_n} has a common difference of 3.
def common_difference (d : ℤ) := d = 3

-- 2. The terms a_2, a_4, and a_8 form a geometric sequence.
def geometric_sequence (a : ℕ → ℤ) :=
  a 4 ^ 2 = a 2 * a 8

-- Question and answer:
-- Prove that the sum of the first 2n terms is 3n(2n+1).
theorem sum_of_arithmetic_sequence 
  (h1 : common_difference 3)
  (h2 : geometric_sequence a) :
  ∑ i in Finset.range (2 * n), a (i + 1) = 3 * n * (2 * n + 1) :=
sorry

end sum_of_arithmetic_sequence_l72_72160


namespace trapezium_area_correct_l72_72326

-- Define the lengths of the parallel sides and the distance between them
def parallel_side1 : ℝ := 20
def parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 17

-- Define the formula for the area of a trapezium
def trapezium_area (a b d : ℝ) := (1 / 2) * (a + b) * d

-- State the theorem that we need to prove
theorem trapezium_area_correct :
  trapezium_area parallel_side1 parallel_side2 distance_between_sides = 323 :=
by
  -- Proof to be filled in
  sorry

end trapezium_area_correct_l72_72326


namespace frank_remaining_money_l72_72341

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l72_72341


namespace ratio_of_areas_l72_72184

def side_length_C : ℝ := 12.5
def side_length_D : ℝ := 18.5
def area_ratio := (side_length_C / side_length_D) ^ 2

theorem ratio_of_areas : area_ratio = (625 : ℝ) / 1369 :=
by
  sorry

end ratio_of_areas_l72_72184


namespace sequence_terms_integer_l72_72183

theorem sequence_terms_integer:
  ∀ n : ℕ, n > 0 → ∃ x: ℕ → ℕ, (x 1 = 2) ∧ (∀ n > 0, n * x n = 2 * (2 * n - 1) * x (n - 1)) → x n ∈ ℤ :=
 by
  sorry

end sequence_terms_integer_l72_72183


namespace maximal_k_ineq_l72_72106

noncomputable def maximal_k (n : ℕ) : ℝ :=
if n % 2 = 0 then (Real.sqrt n) / 2 else (Real.sqrt (n + 5)) / 2

theorem maximal_k_ineq (n : ℕ) (x : Fin n → ℝ) : 
  Real.sqrt (Finset.sum (Finset.range n) (λ i, (x i) ^ 2)) ≥ 
  maximal_k n * (Finset.min' 
    (Finset.image (λ (i : Fin n), |x i - x ((i + 1) % n)|) (Finset.univ : Finset (Fin n)))
    (by {apply Finset.nonempty_image_iff.mpr, exact Finset.univ_nonempty})) :=
sorry

end maximal_k_ineq_l72_72106


namespace remaining_balance_is_correct_l72_72129

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l72_72129


namespace smallest_natural_number_with_digits_sum_1981_l72_72698

-- Definitions for the conditions
def digits_sum (n : ℕ) : ℕ := (n.digits 10).sum

def nat_smaller (n m : ℕ) : Prop := n < m

-- The theorem statement
theorem smallest_natural_number_with_digits_sum_1981 
    (n : ℕ) 
    (h : digits_sum n = 1981) 
    (∀ m : ℕ, (digits_sum m = 1981) → nat_smaller n m = false) : 
  n = (10^220 + 10^219 + 10^218 + ... + 10^0) :=
sorry

end smallest_natural_number_with_digits_sum_1981_l72_72698


namespace height_percentage_difference_l72_72206

theorem height_percentage_difference 
  (r1 h1 r2 h2 : ℝ) 
  (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2)
  (r2_eq_1_2_r1 : r2 = (6 / 5) * r1) :
  h1 = (36 / 25) * h2 :=
by
  sorry

end height_percentage_difference_l72_72206


namespace proof_l72_72112

noncomputable def main : Prop :=
  let p := 0 in
  let q := - (5 / 2) in
  ∀ p q : ℝ, (x-6)*(3*x+12) = x^2 + 2*x - 72 → (p + 2) * (q + 2) = -1

theorem proof : main :=
sorry

end proof_l72_72112


namespace find_a_l72_72470

noncomputable def proof_problem :=
  let f : ℝ → ℝ := λ x, 4 + Real.log x / Real.log 2
  let f' : ℝ → ℝ := λ x, 1 / (x * Real.log 2)
  let F : ℝ → ℝ := λ x, f x - f' x - 4
  ∃ x0 : ℝ, F x0 = 0 ∧ ∃ a : ℕ, a > 0 ∧ x0 ∈ Set.Ioo (a : ℝ) (a + 1)

theorem find_a : proof_problem := 
by 
  -- Details omitted, since proof is not required by the problem statement.
  sorry

end find_a_l72_72470


namespace max_acute_triangles_l72_72864

noncomputable def max_triangles_with_acute_angle (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem max_acute_triangles (n : ℕ) : 
    ∀ (a : ℕ), 
    (a = 2 * n + 1) → 
    (∃ (t : ℕ), t ≤ max_triangles_with_acute_angle n) :=
begin
  intros a ha,
  use max_triangles_with_acute_angle n,
  sorry,
end

end max_acute_triangles_l72_72864


namespace sugar_for_cake_l72_72256

-- Definitions of given values
def sugar_for_frosting : ℝ := 0.6
def total_sugar_required : ℝ := 0.8

-- Proof statement
theorem sugar_for_cake : (total_sugar_required - sugar_for_frosting) = 0.2 :=
by
  sorry

end sugar_for_cake_l72_72256


namespace sum_lent_is_1000_l72_72629

theorem sum_lent_is_1000
    (P : ℝ)
    (r : ℝ)
    (t : ℝ)
    (I : ℝ)
    (h1 : r = 5)
    (h2 : t = 5)
    (h3 : I = P - 750)
    (h4 : I = P * r * t / 100) :
  P = 1000 :=
by sorry

end sum_lent_is_1000_l72_72629


namespace decreasing_interval_range_of_f_l72_72384

noncomputable def f (x : ℝ) : ℝ :=
  (x^3) / 3 + x^2

theorem decreasing_interval :
  ∀ x : ℝ, -2 < x ∧ x < 0 → deriv f x < 0 :=
sorry

theorem range_of_f :
  set.range (λ x, f x) (set.Icc (-1 : ℝ) 1) = set.Icc (0 : ℝ) (4 / 3) :=
sorry

end decreasing_interval_range_of_f_l72_72384


namespace jane_fraction_inspected_l72_72828

theorem jane_fraction_inspected (J N : ℕ) (h1 : 0.005 * J + 0.009 * N = 0.0075 * (J + N)) :
  N / (J + N) = 5 / 8 := by
  sorry

end jane_fraction_inspected_l72_72828


namespace ants_meet_after_laps_l72_72670

/--
Given two circles with radii 33 cm and 9 cm respectively, and two ants starting from the same point \(A\) at the same time and moving at the same speed along each circle, prove that the ant on the smaller circle will complete 11 laps before the ants meet again at point \(A\).
-/
theorem ants_meet_after_laps (r_large r_small : ℝ) (h_large : r_large = 33) (h_small : r_small = 9) (speed : ℝ) :
  ∃ n, n = 11 ∧ (let t_large := (2 * real.pi * r_large) / speed in
                 let t_small := (2 * real.pi * r_small) / speed in
                 let lcm_time := nat.lcm (nat.ceil t_large) (nat.ceil t_small) in
                 lcm_time / (nat.ceil t_small) = n) :=
by
  use 11
  split
  · rfl
  · sorry

end ants_meet_after_laps_l72_72670


namespace relationships_between_A_B_C_l72_72766

def A := {0, 1}
def B := { x | x ∈ A ∧ x ∈ SetOf (fun n => n ∈ ℕ ∧ n > 0) }
def C := { x | x ⊆ A }

theorem relationships_between_A_B_C : (B ⊂ A) ∧ (A ∈ C) ∧ (B ∈ C) :=
    by
    -- Given conditions:
    have hA : A = {0, 1} := rfl
    have hB : B = {1} := by sorry
    have hC : C = { ∅, {1}, {0}, A } := by sorry
    
    -- Relationships
    sorry -- (B ⊂ A) ∧ (A ∈ C) ∧ (B ∈ C)

end relationships_between_A_B_C_l72_72766


namespace area_of_γ1_l72_72310

theorem area_of_γ1 :
  ∀ (ABCDEF : Hexagon)  -- Define a hexagon ABCDEF
  (γ1 γ2 γ3 : Circle)  -- Define three circles γ1, γ2, γ3
  (AB CD EF BC DE FA : ℝ)  -- Define sides of the hexagon
  (r : ℝ)  -- Define radius of the circles
  (h_AB : AB = 1) (h_CD : CD = 1) (h_EF : EF = 1)
  (h_BC : BC = 4) (h_DE : DE = 4) (h_FA : FA = 4)
  (h_tangent_1 : tangent γ1 AB ∧ tangent γ1 BC)
  (h_tangent_2 : tangent γ2 CD ∧ tangent γ2 DE)
  (h_tangent_3 : tangent γ3 EF ∧ tangent γ3 FA)
  (h_pairwise_tangent : tangent γ1 γ2 ∧ tangent γ2 γ3 ∧ tangent γ3 γ1), -- Conditions on tangency
  ∃ (m n : ℕ), 
    (m.gcd n = 1) ∧ -- m and n are relatively prime
    γ1.area = (↑m * Real.pi) / ↑n ∧ -- Area of γ1 is (mπ/n)
    100 * m + n = 14800 :=  -- Check that 100m + n equals 14800
begin
  sorry  -- placeholder for proof
end

end area_of_γ1_l72_72310


namespace range_of_k_l72_72792

def f (x k : ℝ) : ℝ := 4 * x^2 - k * x - 8

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  (∀ x1 x2, a ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ b → f x1 ≤ f x2 ∨ f x1 ≥ f x2)

theorem range_of_k (k : ℝ) : is_monotonic (f (·) k) 5 8 ↔ k ∈ Set.Iic 40 ∪ Set.Ici 64 :=
sorry

end range_of_k_l72_72792


namespace square_side_length_l72_72137

-- Define the pentagon and square within it with given side lengths and positioning
def pentagon_with_square 
  (A B C D E W X Y Z: Point)
  (AB BC CD DE EA: ℝ )
  (WXYZ_square: Square) : Prop :=
  (AB = 30) ∧
  (DE = 30 * (sqrt 3) - 15) ∧
  (W ∈ Segment BC) ∧
  (X ∈ Segment CD) ∧
  (Y ∈ Segment DE) ∧
  (Z ∈ Segment EA)

-- Hypothesis for the side length of the square is as calculated above
theorem square_side_length (A B C D E W X Y Z: Point)
  (s: ℝ)
  (h: pentagon_with_square A B C D E W X Y Z) :
  s = 20 * sqrt 3 - 10 := 
sorry

end square_side_length_l72_72137


namespace points_relation_on_parabola_l72_72026

theorem points_relation_on_parabola :
  let f (x : ℝ) := -(x - 2) ^ 2 in
  let y1 := f (-1) in
  let y2 := f 1 in
  let y3 := f 4 in
  y1 < y3 ∧ y3 < y2 :=
by
  -- Proof to be completed
  sorry

end points_relation_on_parabola_l72_72026


namespace problem_statement_l72_72121

variable {a b c : ℝ}

theorem problem_statement (h : a^2 + 2 * b^2 + 3 * c^2 = 3 / 2) : 3^(-a) + 9^(-b) + 27^(-c) ≥ 1 := by
  sorry

end problem_statement_l72_72121


namespace sum_of_three_distinct_roots_eq_zero_l72_72936

theorem sum_of_three_distinct_roots_eq_zero 
  (a b : ℝ)
  (h_discriminant1 : a^2 - 4 * b > 0)
  (h_discriminant2 : b^2 - 4 * a > 0)
  (h_three_roots : 
    (∃ x1 x2 x3 : ℝ, 
      ∀ x : ℝ, 
        (polynomial.eval x (polynomial.C 1 * polynomial.X^2 + polynomial.C a * polynomial.X + polynomial.C b) *
         polynomial.eval x (polynomial.C 1 * polynomial.X^2 + polynomial.C b * polynomial.X + polynomial.C a)) =
          (x - x1) * (x - x2) * (x - x3))) :
  x1 + x2 + x3 = 0 := 
sorry

end sum_of_three_distinct_roots_eq_zero_l72_72936


namespace tony_water_drink_l72_72195

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end tony_water_drink_l72_72195


namespace volume_of_inscribed_sphere_l72_72662

theorem volume_of_inscribed_sphere (edge_length : ℝ) (h_edge_length : edge_length = 8) : 
  let d := edge_length in
  let r := d / 2 in
  let V := (4/3) * Real.pi * r^3 in
  V = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l72_72662


namespace additional_matches_l72_72642

theorem additional_matches 
  (avg_runs_first_25 : ℕ → ℚ) 
  (avg_runs_additional : ℕ → ℚ) 
  (avg_runs_all : ℚ) 
  (total_matches_first_25 : ℕ) 
  (total_matches_all : ℕ) 
  (total_runs_first_25 : ℚ) 
  (total_runs_all : ℚ) 
  (x : ℕ)
  (h1 : avg_runs_first_25 25 = 45)
  (h2 : avg_runs_additional x = 15)
  (h3 : avg_runs_all = 38.4375)
  (h4 : total_matches_first_25 = 25)
  (h5 : total_matches_all = 32)
  (h6 : total_runs_first_25 = avg_runs_first_25 25 * 25)
  (h7 : total_runs_all = avg_runs_all * 32)
  (h8 : total_runs_first_25 + avg_runs_additional x * x = total_runs_all) :
  x = 7 :=
sorry

end additional_matches_l72_72642


namespace student_arrangement_l72_72247

theorem student_arrangement (students : Fin 6 → Prop)
  (A : (students 0) ∨ (students 5) → False)
  (females_adj : ∃ (i : Fin 6), i < 5 ∧ students i → students (i + 1))
  : ∃! n, n = 96 := by
  sorry

end student_arrangement_l72_72247


namespace eval_expression_l72_72703

theorem eval_expression : (49^2 - 25^2 + 10^2) = 1876 := by
  sorry

end eval_expression_l72_72703


namespace two_nm_to_m_scientific_notation_l72_72861

def nm_to_m : Float := 1e-9

theorem two_nm_to_m_scientific_notation : 2 * nm_to_m = 2 * 10⁻⁹ :=
by
  simp [nm_to_m]
  sorry

end two_nm_to_m_scientific_notation_l72_72861


namespace triangle_inequality_proof_l72_72538

theorem triangle_inequality_proof 
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_proof_l72_72538


namespace complex_fraction_calculation_l72_72059

theorem complex_fraction_calculation (z : ℂ) (h : z = 2 + 1 * complex.I) : (2 * complex.I) / (z - 1) = 1 + complex.I :=
by
  sorry

end complex_fraction_calculation_l72_72059


namespace number_of_bottle_caps_l72_72099

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end number_of_bottle_caps_l72_72099


namespace intersection_complement_eq_l72_72035

def U : Set Int := { -2, -1, 0, 1, 2, 3 }
def M : Set Int := { 0, 1, 2 }
def N : Set Int := { 0, 1, 2, 3 }

noncomputable def C_U (A : Set Int) := U \ A

theorem intersection_complement_eq :
  (C_U M ∩ N) = {3} :=
by
  sorry

end intersection_complement_eq_l72_72035


namespace problem_f_l72_72440

/-- Coefficient of the term x^m * y^n in the expansion of (1+x)^6 * (1+y)^4 --/
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

theorem problem_f :
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  -- We skip the proof
  sorry

end problem_f_l72_72440


namespace sum_squares_mod_13_l72_72608

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 10 := by
  sorry

end sum_squares_mod_13_l72_72608


namespace solve_ordered_pair_max_value_l72_72456

noncomputable def g (x : ℝ) : ℝ := real.sqrt (x * (100 - x)) + real.sqrt (x * (10 - x))

theorem solve_ordered_pair_max_value :
  ∃ (x1 N : ℝ), (g(x1) = N) ∧ (∀ x, 0 ≤ x ∧ x ≤ 10 → g(x) ≤ N) ∧ (x1 = 10) ∧ (N = 30 * real.sqrt 2) :=
by
  sorry

end solve_ordered_pair_max_value_l72_72456


namespace problem_z99_z100_sum_l72_72910

noncomputable def i : ℂ := complex.I

noncomputable def z : ℕ → ℂ
| 1       := 3 + 2 * i
| (n + 1) := complex.conj (z n) * (i ^ n)

theorem problem_z99_z100_sum :
  (z 99) + (z 100) = -5 + 5 * i :=
by
  sorry

end problem_z99_z100_sum_l72_72910


namespace inequality_proof_l72_72885

variable (x y : ℝ)
variable (hx : 0 < x) (hy : 0 < y)

theorem inequality_proof :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y :=
sorry

end inequality_proof_l72_72885


namespace angle_ab_da_eq_60deg_distance_cb_ad_eq_half_bc_l72_72831

-- Define the problem setting
variables {A B C D A' B' C' D' : Type}
variables {O O' : Type}
variables {u v w : Type}

-- Given conditions in Lean
-- AB (A, B, C, D) and A'B'C'D' form a truncated regular pyramid
-- BC' and DA' are perpendicular
-- The projection of B' on (ABC) is the center of the incircle of triangle ABC

-- Prove statements in Lean
theorem angle_ab_da_eq_60deg 
  (h_pyramid: truncated_regular_pyramid A B C D A' B' C' D')
  (h_orthogonal: orthogonal BC' DA'):
  ∠ (AB' DA') = 60 := sorry

theorem distance_cb_ad_eq_half_bc 
  (h_projection: projection_center_incircle B' (triangle ABC)) :
  distance (CB' AD') = 1 / 2 * BC' := sorry

end angle_ab_da_eq_60deg_distance_cb_ad_eq_half_bc_l72_72831


namespace participant_avg_eq_reciprocal_l72_72660

open_locale big_operators

def binom (n k : ℕ) := nat.choose n k

theorem participant_avg_eq_reciprocal (n T : ℕ) (hₙ : 9 ≤ n ∧ n ≤ 2017)
  (h_avg_9 := (T * binom (n-5) 4) / binom n 9)
  (h_avg_8 := (T * binom (n-5) 3) / binom n 8) :
  ((T * binom (n-5) 4) / binom n 9 = 1 / ((T * binom (n-5) 3) / binom n 8))
  → (∃ S : finset ℕ, S.card = 557 ∧ ∀ x ∈ S, 9 ≤ x ∧ x ≤ 2017 ∧ odd x) :=
sorry

end participant_avg_eq_reciprocal_l72_72660


namespace vertex_of_parabola_l72_72911

open Function

noncomputable def vertex_coordinates (a b c : ℝ) : ℝ × ℝ :=
let h := -b / (2 * a), k := (4 * a * c - b^2) / (4 * a) 
in (h, k)

theorem vertex_of_parabola : 
  vertex_coordinates 1 (-2) 3 = (1, 2) :=
by
  unfold vertex_coordinates
  simp
  split
  all_goals {norm_num}

end vertex_of_parabola_l72_72911


namespace quadratic_roots_find_m_l72_72794

theorem quadratic_roots_find_m (m : ℚ) :
  (∀ x : ℂ, 10 * x^2 - 6 * x + m = 0 → x = (3 + complex.I * real.sqrt 191) / 10 ∨ x = (3 - complex.I * real.sqrt 191) / 10) →
  m = 227 / 40 :=
sorry

end quadratic_roots_find_m_l72_72794


namespace probability_heads_equals_7_over_11_l72_72489

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l72_72489


namespace big_al_ate_40_bananas_on_june_7_l72_72294

-- Given conditions
def bananas_eaten_on_day (initial_bananas : ℕ) (day : ℕ) : ℕ :=
  initial_bananas + 4 * (day - 1)

def total_bananas_eaten (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 1 +
  bananas_eaten_on_day initial_bananas 2 +
  bananas_eaten_on_day initial_bananas 3 +
  bananas_eaten_on_day initial_bananas 4 +
  bananas_eaten_on_day initial_bananas 5 +
  bananas_eaten_on_day initial_bananas 6 +
  bananas_eaten_on_day initial_bananas 7

noncomputable def final_bananas_on_june_7 (initial_bananas : ℕ) : ℕ :=
  bananas_eaten_on_day initial_bananas 7

-- Theorem to be proved
theorem big_al_ate_40_bananas_on_june_7 :
  ∃ initial_bananas, total_bananas_eaten initial_bananas = 196 ∧ final_bananas_on_june_7 initial_bananas = 40 :=
sorry

end big_al_ate_40_bananas_on_june_7_l72_72294


namespace two_pow_n_minus_one_prime_imp_n_prime_l72_72477

theorem two_pow_n_minus_one_prime_imp_n_prime (n : ℕ) (h : Nat.Prime (2^n - 1)) : Nat.Prime n := 
sorry

end two_pow_n_minus_one_prime_imp_n_prime_l72_72477


namespace tangent_line_l72_72637

theorem tangent_line (x y : ℝ) (h_curve : y = x + Real.log x) (h_point: x = 1 ∧ y = 1): 
  ∃ k b, tangent_line_eq : y = k * x + b ∧ k = 2 ∧ b = -1 := 
sorry

end tangent_line_l72_72637


namespace min_workers_to_make_profit_l72_72260

theorem min_workers_to_make_profit 
  (maintenance_fees : ℕ := 700) 
  (wage_per_hour_per_worker : ℕ := 20) 
  (hours_per_day : ℕ := 8)
  (widgets_per_hour_per_worker : ℕ := 4) 
  (price_per_widget : ℕ := 4) : 
  ∃ n : ℕ, n = 22 ∧ 128 * n > 700 + 160 * n :=
by
  use 22
  have h : 128 * 22 > 700 + 160 * 22 := by linarith
  exact ⟨rfl, h⟩
  sorry

end min_workers_to_make_profit_l72_72260


namespace equation_of_ellipse_fixed_point_intersection_l72_72483

section EllipseProblem

-- Conditions
variable (a b : ℝ)
variable (e : ℝ) (b_pos : b > 0) (a_b_relation : a > b)
variable (short_axis : b = sqrt 3)
variable (eccentricity : e = 1 / 2)

-- Definitions based on conditions
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def slope_of_line (k t : ℝ) (x y : ℝ) : Prop :=
  y = k * x + t

def point_A : ℝ × ℝ := (0, -sqrt 3)

-- Proof targets
theorem equation_of_ellipse :
  ellipse_equation 0 0 ↔ (a^2 = 4 ∧ b^2 = 3) := sorry

theorem fixed_point_intersection (k t : ℝ) :
  ∀ P Q : ℝ × ℝ,
    P ≠ point_A ∧ Q ≠ point_A ∧ 
    slope_of_line k t P.1 P.2 ∧ slope_of_line k t Q.1 Q.2 ∧
    (P.2 - point_A.2) / (P.1 - point_A.1) + (Q.2 - point_A.2) / (Q.1 - point_A.1) = 2 →
    ∃ fp : ℝ × ℝ, fp = (sqrt 3, sqrt 3) := sorry

end EllipseProblem

end equation_of_ellipse_fixed_point_intersection_l72_72483


namespace number_of_harmonious_sets_l72_72697

def is_harmonious_set (G : Type) (op : G → G → G) (e : G) : Prop :=
  (∀ a b : G, op a b = a ∧ op a b = b) ∧ (∀ a : G, op a e = a ∧ op e a = a)

def harmonious_sets : Nat := 
  let non_negative_integers := {n : Nat // n ≥ 0}
  let even_numbers := {n : Int // n % 2 = 0}
  let plane_vectors := {vec : ℝ × ℝ}

  let op_add (a b : non_negative_integers) : non_negative_integers := ⟨a.val + b.val, sorry⟩
  let op_mul (a b : even_numbers) : even_numbers := ⟨a.val * b.val, sorry⟩
  let op_vec_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.fst + b.fst, a.snd + b.snd)

  let identity_add := ⟨0, sorry⟩ : non_negative_integers
  let identity_mul := ⟨1, sorry⟩ : even_numbers
  let identity_vec := (0, 0) : ℝ × ℝ

  let set1 := is_harmonious_set non_negative_integers op_add identity_add
  let set2 := is_harmonious_set even_numbers op_mul identity_mul
  let set3 := is_harmonious_set plane_vectors op_vec_add identity_vec

  (if set1 then 1 else 0) + (if set2 then 1 else 0) + (if set3 then 1 else 0)

theorem number_of_harmonious_sets : harmonious_sets = 2 := sorry

end number_of_harmonious_sets_l72_72697


namespace opposite_of_neg_quarter_l72_72940

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end opposite_of_neg_quarter_l72_72940


namespace solve_equation_1_solve_equation_2_l72_72533

theorem solve_equation_1 (x : ℝ) (h : 0.5 * x + 1.1 = 6.5 - 1.3 * x) : x = 3 :=
  by sorry

theorem solve_equation_2 (x : ℝ) (h : (1 / 6) * (3 * x - 9) = (2 / 5) * x - 3) : x = -15 :=
  by sorry

end solve_equation_1_solve_equation_2_l72_72533


namespace range_of_f_triangle_area_l72_72020

noncomputable def f (x : ℝ) : ℝ :=
  let m := Matrix.of ![[sqrt 3 * cos x ^ 2, - sin x], [cos x, 1]]
  Matrix.det m

-- Proving the range of f(x)
theorem range_of_f :
  ∀ x ∈ set.Icc 0 (Real.pi / 2), f x ∈ set.Icc 0 (1 + sqrt 3 / 2) :=
sorry

-- Defining the conditions for the triangle problem
def A (triangle : ℝ × ℝ × ℝ) : ℝ := triangle.1
def a (triangle : ℝ × ℝ × ℝ) : ℝ := 4
def b_plus_c (b c : ℝ) : Prop := b + c = 5

-- Proving the area
theorem triangle_area (A b c : ℝ) (hA : A ∈ set.Ioo 0 Real.pi)
  (hb_plus_c : b_plus_c b c) (hf : f (A / 2) = sqrt 3)
  : 1 / 2 * b * c * sin A = 3 * sqrt 3 / 4 :=
sorry

end range_of_f_triangle_area_l72_72020


namespace collinear_P1_P2_P3_l72_72036

-- Define the points and conditions
variables {A0 A1 A2 A3 P1 P2 P3 : Type}
variables [EuclideanGeometry]

-- conditions given in the problem
variables (h1 : Perpendicular (LineThrough A1 A2) (LineThrough A0 P1))
variables (h2 : Perpendicular (LineThrough A2 A3) (LineThrough A0 P2))
variables (h3 : Perpendicular (LineThrough A3 A1) (LineThrough A0 P3))

-- proof goal: Prove that P1, P2, and P3 are collinear
theorem collinear_P1_P2_P3
  (h1 : Perpendicular (LineThrough A1 A2) (LineThrough A0 P1))
  (h2 : Perpendicular (LineThrough A2 A3) (LineThrough A0 P2))
  (h3 : Perpendicular (LineThrough A3 A1) (LineThrough A0 P3)) :
  Collinear {P1, P2, P3} :=
by
  sorry

end collinear_P1_P2_P3_l72_72036


namespace waiter_earnings_l72_72240

theorem waiter_earnings
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (customers_tipped : total_customers - no_tip_customers = 3)
  (tips_per_customer : tip_amount = 9) :
  (total_customers - no_tip_customers) * tip_amount = 27 := by
  sorry

end waiter_earnings_l72_72240


namespace parallel_if_perpendicular_to_same_line_l72_72149

-- Define the geometrical concepts in play
variables {ℓ₁ ℓ₂ m : Type} [Plane ℓ₁] [Plane ℓ₂] [Line m]

-- Define the condition
def perpendicular_to_same_line (ℓ₁ ℓ₂ m : Type) [Plane ℓ₁] [Plane ℓ₂] [Line m] : Prop :=
  (perpendicular ℓ₁ m) ∧ (perpendicular ℓ₂ m)

-- State the theorem to prove
theorem parallel_if_perpendicular_to_same_line
  (ℓ₁ ℓ₂ m : Type) [Plane ℓ₁] [Plane ℓ₂] [Line m]
  (h : perpendicular_to_same_line ℓ₁ ℓ₂ m) :
  parallel ℓ₁ ℓ₂ :=
sorry

end parallel_if_perpendicular_to_same_line_l72_72149


namespace lesser_solution_is_minus_15_l72_72215

noncomputable def lesser_solution : ℤ := -15

theorem lesser_solution_is_minus_15 :
  ∃ x y : ℤ, x^2 + 10 * x - 75 = 0 ∧ y^2 + 10 * y - 75 = 0 ∧ x < y ∧ x = lesser_solution :=
by 
  sorry

end lesser_solution_is_minus_15_l72_72215


namespace sum_of_squares_mod_13_l72_72601

theorem sum_of_squares_mod_13 :
  ((∑ i in Finset.range 16, i^2) % 13) = 3 :=
by 
  sorry

end sum_of_squares_mod_13_l72_72601


namespace NONNA_permutations_MATHEMATICS_permutations_l72_72401

theorem NONNA_permutations : 
  let n := 5 
  let r := 3 
  (n.fact / r.fact) = 20 := 
by 
  sorry

theorem MATHEMATICS_permutations : 
  let total_letters := 10 
  let count_M := 2 
  let count_A := 3 
  let count_T := 2 
  let count_E := 1 
  let count_I := 1 
  let count_K := 1 
  (total_letters.fact / (count_M.fact * count_A.fact * count_T.fact * count_E.fact * count_I.fact * count_K.fact)) = 151200 := 
by 
  sorry

end NONNA_permutations_MATHEMATICS_permutations_l72_72401


namespace line_curve_intersect_curve_value_range_l72_72379

-- Part 1
theorem line_curve_intersect (m : ℝ) (h : ∃ A B : ℝ × ℝ, 
  (A.1 - 2)^2 + A.2^2 = 4 ∧ (B.1 - 2)^2 + B.2^2 = 4 ∧
  A.1 - A.2 - m = 0 ∧ B.1 - B.2 - m = 0 ∧ 
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = real.sqrt 14) : 
  m = 1 ∨ m = 3 :=
sorry

-- Part 2
theorem curve_value_range (x y : ℝ) (h : ∃ θ : ℝ, 
  x = 2 + 2 * real.cos θ ∧ y = 2 * real.sin θ) : 
  2 - 2 * real.sqrt 5 ≤ x + 2 * y ∧ x + 2 * y ≤ 2 + 2 * real.sqrt 5 :=
sorry

end line_curve_intersect_curve_value_range_l72_72379


namespace problem_l72_72707

-- Define the matrix
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 0], ![0, 2, 3], ![3, 0, 2]]

-- Define the condition that there exists a nonzero vector v such that A * v = k * v
def exists_eigenvector (k : ℝ) : Prop :=
  ∃ (v : Fin 3 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v

theorem problem : ∀ (k : ℝ), exists_eigenvector k ↔ (k = 2 + (45)^(1/3)) :=
sorry

end problem_l72_72707


namespace number_line_steps_l72_72863

theorem number_line_steps (total_distance steps : ℕ) (distance_per_step x : ℕ) 
  (h1 : total_distance = 25) (h2 : steps = 5) 
  (h3 : distance_per_step = total_distance / steps) 
  (h4 : x = 4 * distance_per_step) : 
  x = 20 :=
by 
  subst h1 
  subst h2
  subst h3
  subst h4
  simp
  sorry

end number_line_steps_l72_72863


namespace intersection_A_B_l72_72395

open Set

def universal_set : Set ℕ := {0, 1, 3, 5, 7, 9}
def complement_A : Set ℕ := {0, 5, 9}
def B : Set ℕ := {3, 5, 7}
def A : Set ℕ := universal_set \ complement_A

theorem intersection_A_B :
  A ∩ B = {3, 7} :=
by
  sorry

end intersection_A_B_l72_72395


namespace zero_point_of_f_l72_72571

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

theorem zero_point_of_f : f 3 = 0 :=
by
  unfold f
  rw [Real.log_eq_zero, sub_eq_self]
  exact rfl

# To verify our theorem, we need to check conditions under which the function is zero.
# According to the problem, we need to show that 3 is a zero point of our function.

end zero_point_of_f_l72_72571


namespace measure_of_angle_C_l72_72082

variable (a b c : ℝ) (S : ℝ)

-- Conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom area_equation : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)

-- The problem
theorem measure_of_angle_C (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.arctan (Real.sqrt 3) ∧ C = Real.pi / 3 :=
by
  sorry

end measure_of_angle_C_l72_72082


namespace a5_equals_17_l72_72079

noncomputable def seq : ℕ → ℕ
| 0 := 0 -- for convenience since list is 1-based in the problem
| 1 := 1
| 2 := 2
| (n + 3) := 2 * seq (n + 2) - seq (n + 1) + 2

theorem a5_equals_17 : seq 5 = 17 :=
by
  sorry

end a5_equals_17_l72_72079


namespace fraction_complex_eq_l72_72058

theorem fraction_complex_eq (z : ℂ) (h : z = 2 + I) : 2 * I / (z - 1) = 1 + I := by
  sorry

end fraction_complex_eq_l72_72058


namespace pentagon_angles_l72_72633

theorem pentagon_angles (ABCDE : Type) [convex_pentagon ABCDE]
  (B C A E D : Point ABCDE)
  (angle_BCA : Angle B C A)
  (angle_BEA : Angle B E A)
  (angle_BDA : Angle B D A)
  (angle_BDC : Angle B D C)
  (angle_EDA : Angle E D A)
  (h1: angle_BCA = angle_BEA)
  (h2: angle_BCA = (angle_BDA) / 2)
  (h3: angle_BDC = angle_EDA) :
  ∠ E D B = ∠ D A C :=
sorry

end pentagon_angles_l72_72633


namespace endpoints_of_diameter_l72_72577

universe u
variables {k : Type u} [Field k]
variables {S1 S2 S3: Type*} [Circumference S1 k] [Circumference S2 k] [Circumference S3 k]
variables {A B C: k}

-- Given three circles touching each other pairwise at distinct points A, B, and C.
structure TouchingCircles (S1 S2 S3: Type*) (k: Type u) [Field k]
  (A B C: k) :=
  (touching1 : TangencyPoint S2 S3 A)
  (touching2 : TangencyPoint S3 S1 B)
  (touching3 : TangencyPoint S1 S2 C)

-- Prove the desired intersection property
theorem endpoints_of_diameter
  (S1 S2 S3: Type*) [Circumference S1 k] [Circumference S2 k] [Circumference S3 k]
  (A B C : k) [TouchingCircles S1 S2 S3 k A B C] :
  ∃ (A1 B1: k), Line.connects_through_diameter S3 A B A1 B1 :=
sorry

end endpoints_of_diameter_l72_72577


namespace wobbly_divisibility_l72_72282

noncomputable def isWobbly (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, (i % 2 = 0 → digits.get i ≠ 0) ∧ (i % 2 = 1 → digits.get i = 0)

theorem wobbly_divisibility (n : ℕ) : 
  (∀ w : ℕ, isWobbly w → ¬ (n ∣ w)) ↔ (10 ∣ n ∨ 25 ∣ n) := 
sorry

end wobbly_divisibility_l72_72282


namespace radian_measure_of_acute_angle_l72_72193

def radius_largest := 4
def radius_middle := 3
def radius_smallest := 2

def area_largest := 16 * Real.pi
def area_middle := 9 * Real.pi
def area_smallest := 4 * Real.pi

def total_area := area_largest + area_middle + area_smallest

-- Let S be the area of the shaded region and U be the area of the unshaded region
def S := (3/4: ℝ) * U
def total_area_eq (U : ℝ) : S + U = total_area := 
  calc S + U = (3/4 * U) + U : by rw S 
         ... = (7/4) * U : by ring 
         ... = total_area : sorry -- this follows from substituting and solving

-- Given conditions
axiom shaded_relation (U : ℝ) (S : ℝ) : S = (3/4) * U

-- Prove the radian measure of the acute angle
theorem radian_measure_of_acute_angle (U : ℝ) (θ : ℝ) 
  (hU : U = 116 * Real.pi / 7) 
  (hS : shaded_relation U S) 
  (htotal_area_eq : total_area_eq U)
  : θ = 6 * Real.pi / 77 := 
  sorry

end radian_measure_of_acute_angle_l72_72193


namespace number_of_meetings_is_zero_l72_72200

structure TrackConditions where
  circumference : ℝ
  speed_boy1 : ℝ
  speed_boy2 : ℝ

noncomputable def numberOfMeetings (conds : TrackConditions) : ℕ :=
  if h : conds.speed_boy1 + conds.speed_boy2 ≠ 0 then
    let time_to_meet_at_A : ℝ := conds.circumference / (conds.speed_boy1 + conds.speed_boy2)
    let possible_times : List ℝ := List.range 7 |>.map (fun n => (n : ℝ) * (conds.circumference / |conds.speed_boy1 - conds.speed_boy2|))
    possible_times.filter (fun t => t < time_to_meet_at_A && t ≠ 0).length
  else 0

theorem number_of_meetings_is_zero :
  ∀ (conds : TrackConditions),
    conds.circumference = 120 →
    conds.speed_boy1 = 6 →
    conds.speed_boy2 = 10 →
    numberOfMeetings conds = 0 :=
by
  intros conds circ_eq speed1_eq speed2_eq
  sorry

end number_of_meetings_is_zero_l72_72200


namespace oblique_coordinates_vector_properties_l72_72771

variable {θ : ℝ}
variable {λ x1 y1 x2 y2 : ℝ}

-- Conditions
axiom unit_vectors : ∀ (e1 e2 : ℝ), e1 * e1 = 1 ∧ e2 * e2 = 1 
axiom non_right_angle : θ ≠ π / 2

-- Definitions in oblique coordinates
def oblique_vector (e1 e2 : ℝ) (x y : ℝ) : ℝ := x * e1 + y * e2
def vector_sub (a1 a2 b1 b2 : ℝ) : ℝ × ℝ := (a1 - b1, a2 - b2)
def scalar_mul (λ x y : ℝ) : ℝ × ℝ := (λ * x, λ * y)

-- Proof Problem Statement
theorem oblique_coordinates_vector_properties (e1 e2 : ℝ) (h1 : e1 * e1 = 1) (h2 : e2 * e2 = 1) (h3 : θ ≠ π / 2) :
  vector_sub (x1 * e1) (y1 * e2) (x2 * e1) (y2 * e2) = (x1 - x2, y1 - y2)
  ∧ scalar_mul λ x1 y1 = (λ * x1, λ * y1) :=
by sorry

end oblique_coordinates_vector_properties_l72_72771


namespace largest_divisor_of_n4_sub_4n2_is_4_l72_72315

theorem largest_divisor_of_n4_sub_4n2_is_4 (n : ℤ) : 4 ∣ (n^4 - 4 * n^2) :=
sorry

end largest_divisor_of_n4_sub_4n2_is_4_l72_72315


namespace cubic_eq_has_natural_roots_l72_72325

theorem cubic_eq_has_natural_roots (p : ℝ) :
  (∃ (x y z : ℕ), 5*x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p ∧ 
                 5*y^3 - 5*(p+1)*y^2 + (71*p-1)*y + 1 = 66*p ∧ 
                 5*z^3 - 5*(p+1)*z^2 + (71*p-1)*z + 1 = 66*p) → p = 76 :=
begin
  sorry
end

end cubic_eq_has_natural_roots_l72_72325


namespace locus_of_M_l72_72358

noncomputable def Z (A B C D : ℝ) : ℝ := sorry -- Some definition for the intersection of lines AB and CD

theorem locus_of_M (A B C D : ℝ) :
  ∃ Z, ∀ M,
  circle (center := Z) (radius := real.sqrt (Z * A * Z * B)) =
  { M : ℝ | circumcircle_tangent (Z M A B) (Z M C D) } :=
sorry

end locus_of_M_l72_72358


namespace multiplicative_inverse_exists_l72_72117

def P : ℕ := 123321
def Q : ℕ := 246642
def N : ℕ := 1_000_003
def PQ_mod_N : ℕ := (P * Q) % N

theorem multiplicative_inverse_exists :
  ∃ M : ℕ, (PQ_mod_N * M) % N = 1 ∧ 0 < M := by
  have hPQ_mod_N : PQ_mod_N = 330 := by
    unfold PQ_mod_N
    norm_num
  use 69788
  rw [hPQ_mod_N]
  norm_num
  sorry

end multiplicative_inverse_exists_l72_72117


namespace mirasol_balance_l72_72131

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l72_72131


namespace problem_remainder_P2017_mod_1000_l72_72105

def P (x : ℤ) : ℤ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem problem_remainder_P2017_mod_1000 :
  (P 2017) % 1000 = 167 :=
by
  -- this proof examines \( P(2017) \) modulo 1000
  sorry

end problem_remainder_P2017_mod_1000_l72_72105


namespace number_of_levels_l72_72073

theorem number_of_levels (total_capacity : ℕ) (additional_cars : ℕ) (already_parked_cars : ℕ) (n : ℕ) :
  total_capacity = 425 →
  additional_cars = 62 →
  already_parked_cars = 23 →
  n = total_capacity / (already_parked_cars + additional_cars) →
  n = 5 :=
by
  intros
  sorry

end number_of_levels_l72_72073


namespace quadratic_inequality_solution_l72_72564

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 10 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 5} :=
by
  sorry

end quadratic_inequality_solution_l72_72564


namespace average_temperature_l72_72906

theorem average_temperature (T_tue T_wed T_thu : ℝ) 
  (h1 : (42 + T_tue + T_wed + T_thu) / 4 = 48)
  (T_fri : ℝ := 34) :
  ((T_tue + T_wed + T_thu + T_fri) / 4 = 46) :=
by
  sorry

end average_temperature_l72_72906


namespace length_of_platform_l72_72995

noncomputable def train_length : ℝ := 300
noncomputable def time_to_cross_platform : ℝ := 39
noncomputable def time_to_cross_pole : ℝ := 9

theorem length_of_platform : ∃ P : ℝ, P = 1000 :=
by
  let train_speed := train_length / time_to_cross_pole
  let total_distance_cross_platform := train_length + 1000
  let platform_length := total_distance_cross_platform - train_length
  existsi platform_length
  sorry

end length_of_platform_l72_72995


namespace choose_blue_pair_l72_72802

/-- In a drawer, there are 12 distinguishable socks: 5 white, 3 brown, and 4 blue socks.
    Prove that the number of ways to choose a pair of socks such that both socks are blue is 6. -/
theorem choose_blue_pair (total_socks white_socks brown_socks blue_socks : ℕ)
  (h_total : total_socks = 12) (h_white : white_socks = 5) (h_brown : brown_socks = 3) (h_blue : blue_socks = 4) :
  (blue_socks.choose 2) = 6 :=
by
  sorry

end choose_blue_pair_l72_72802


namespace locus_of_midpoints_circle_l72_72461

-- Definitions and conditions
section CircleLocus

variables (O Q : Point) (r : ℝ)
def circle_center_radius (O : Point) (r : ℝ) := { X : Point | dist O X = r }
def point_in_circle (Q O : Point) (r : ℝ) := dist O Q < r

-- Given: A circle of radius 10 units centered at O, and Q inside at 3 units from O
parameter (C : set Point)
parameter (radius_OQ : dist O Q = 3)
parameter (radius_CO : dist O = 10)
axiom (circ_def : C = circle_center_radius O 10)
axiom (Q_in_C : point_in_circle Q O 10)

-- Conclusion: The locus of midpoints of all chords passing through Q forms a circle
theorem locus_of_midpoints_circle :
  ∃ (M : Point) (radius_OM: ℝ), 
    (forall (X Y : Point), X ∈ C ∧ Y ∈ C ∧ 
    dist Q (X, Y) = (3) ∧ midpoint X Y = M) → (dist O M = 1.5 ) :=
sorry

end CircleLocus

end locus_of_midpoints_circle_l72_72461


namespace find_third_angle_l72_72815

-- Definitions from the problem conditions
def triangle_angle_sum (a b c : ℝ) : Prop := a + b + c = 180

-- Statement of the proof problem
theorem find_third_angle (a b x : ℝ) (h1 : a = 50) (h2 : b = 45) (h3 : triangle_angle_sum a b x) : x = 85 := sorry

end find_third_angle_l72_72815


namespace root_fraction_power_l72_72916

theorem root_fraction_power (a : ℝ) (ha : a = 5) : 
  (a^(1/3)) / (a^(1/5)) = a^(2/15) := by
  sorry

end root_fraction_power_l72_72916


namespace male_students_count_l72_72527

theorem male_students_count (x : ℕ) (h1 : 0 < x)
  (h2 : x > 10 - x)                               -- More male than female students
  (h3 : nat.choose x 2 * nat.choose (10 - x) 2 * 6 * 6 = 3240) : -- Given condition
  x = 6 :=
sorry

end male_students_count_l72_72527


namespace triangle_area_bounded_by_lines_l72_72591

theorem triangle_area_bounded_by_lines : 
  let A := (8, 8)
      B := (-8, 8)
      O := (0, 0)
  in
  let base_length : ℝ := 16
  let height : ℝ := 8
  let area : ℝ := (1 / 2) * base_length * height
  in
  area = 64 :=
by
  sorry

end triangle_area_bounded_by_lines_l72_72591


namespace geometric_mean_of_negatives_l72_72920

theorem geometric_mean_of_negatives :
  ∃ x : ℝ, x^2 = (-2) * (-8) ∧ (x = 4 ∨ x = -4) := by
  sorry

end geometric_mean_of_negatives_l72_72920


namespace none_are_perfect_squares_l72_72219

-- Definitions of the factorial expressions
def expr_A := factorial 100 * factorial 101
def expr_B := factorial 101 * factorial 102
def expr_C := factorial 102 * factorial 103
def expr_D := factorial 103 * factorial 104
def expr_E := factorial 104 * factorial 105

-- The theorem with a proof obligation that none of the expressions are perfect squares
theorem none_are_perfect_squares :
  ¬(∃ x, x^2 = expr_A) ∧ ¬(∃ x, x^2 = expr_B) ∧ ¬(∃ x, x^2 = expr_C) ∧ ¬(∃ x, x^2 = expr_D) ∧ ¬(∃ x, x^2 = expr_E) :=
by
  sorry

end none_are_perfect_squares_l72_72219


namespace batsman_average_runs_l72_72163

theorem batsman_average_runs
  (average_20_matches : ℕ → ℕ)
  (average_10_matches : ℕ → ℕ)
  (h1 : average_20_matches = 20 * 40)
  (h2 : average_10_matches = 10 * 13) :
  (average_20_matches + average_10_matches) / 30 = 31 := 
by 
  sorry

end batsman_average_runs_l72_72163


namespace sin_alpha_plus_beta_l72_72739

theorem sin_alpha_plus_beta (α β : ℝ) 
    (h1 : sin α + cos β = 1) 
    (h2 : cos α + sin β = 0) : 
    sin (α + β) = -1 / 2 := 
by 
  sorry

end sin_alpha_plus_beta_l72_72739


namespace card_B_l72_72033

open Set

def A : Set ℤ := {0, 1, 2}

def B : Set ℤ := {z | ∃ (x ∈ A) (y ∈ A), z = x - y}

theorem card_B : B.to_finset.card = 5 := by
  sorry

end card_B_l72_72033


namespace number_of_juniors_l72_72809

theorem number_of_juniors (total_students j_percentage s_percentage : ℚ) (debate_team_ratio : ℚ):
  total_students = 40 →
  j_percentage = 1/5 →
  s_percentage = 1/4 →
  debate_team_ratio = 2 →
  ∃ J S, J + S = total_students ∧ S = debate_team_ratio * j_percentage * J / s_percentage ∧ J = 11 :=
by 
  intros h1 h2 h3 h4
  use 11
  use 18
  split
  exact h1
  split
  calc 18 = (2 : ℚ) * (1 / 5) * 11 / (1 / 4) : by 
    rw [h2, h3, h4]
    ring
  exact rfl
  exact rfl

end number_of_juniors_l72_72809


namespace min_sum_b_l72_72355

-- Define the sequence a_n
def seq_a (n : ℕ) : ℕ → ℕ := λ n, if n = 0 then 0 else if n = 1 then 0 else if n = 2 then 10 else sorry

-- Define the sum of first n terms S_n
def sum_a (n : ℕ) : ℕ := sorry

axiom a3_eq_10 : seq_a 2 = 10
axiom S6_eq_72 : sum_a 6 = 72

-- Define the new sequence b_n
def seq_b (n : ℕ) : ℕ := seq_a n / 2 - 30

-- Define the sum of first n terms T_n of sequence b_n
def sum_b (n : ℕ) : ℕ := sorry

-- Prove that the minimum value of the sum of the first n terms of sequence b_n is -225
theorem min_sum_b : ∃ n, sum_b 15 = -225 :=
by {
    --express the sum_b computation and put the necessary logic to derive it
    sorry
}

end min_sum_b_l72_72355


namespace smallest_battleship_grid_l72_72523

/-- Definition of the ship placements and requirements. -/
def ship_placements (n : ℕ) : Prop :=
  ∃ (ship1 ship2 ship3 ship4 : Finset (Fin n × Fin n)),
  disjoint ship1 ship2 ∧ disjoint ship1 ship3 ∧ disjoint ship1 ship4 ∧
  disjoint ship2 ship3 ∧ disjoint ship2 ship4 ∧ disjoint ship3 ship4 ∧
  ship1.card = 4 ∧
  (ship2.card = 3 ∧ ship3.card = 3) ∧
  (ship4.card = 2 ∧ ship4.card = 1 ∧ ship4.card = 1 ∧ ship4.card = 1) ∧
  pairwise (λ s1 s2, ∀ p ∈ s1, ∀ q ∈ s2, ¬ (adjacent p q or corner_touch p q))

-- Definition of what it means for points to be adjacent
def adjacent (p q : Fin n × Fin n) : Prop :=
  (p.fst = q.fst ∧ (p.snd = q.snd + 1 ∨ q.snd = p.snd + 1) ∨
  (p.snd = q.snd ∧ (p.fst = q.fst + 1 ∨ q.fst = p.fst + 1)))

-- Definition of what it means for points to be touching at the corners
def corner_touch (p q : Fin n × Fin n) : Prop :=
  abs (p.fst - q.fst) = 1 ∧ abs (p.snd - q.snd) = 1

/-- The main theorem statement: Smallest grid size for the Battleship game adhering to the rules. -/
theorem smallest_battleship_grid : ∃ (n : ℕ), n = 7 ∧ ship_placements n :=
by
  use 7
  sorry

end smallest_battleship_grid_l72_72523


namespace hours_l72_72652

def mechanic_hours_charged (h : ℕ) : Prop :=
  45 * h + 225 = 450

theorem hours (h : ℕ) : mechanic_hours_charged h → h = 5 :=
by
  intro h_eq
  have : 45 * h + 225 = 450 := h_eq
  sorry

end hours_l72_72652


namespace triangle_square_ratio_l72_72960

theorem triangle_square_ratio (n : ℝ) (h₁ : ∀ (A B C D E F : ℝ),
    A^2 + B^2 = C^2 ∧ D^2 + E^2 = F^2 → 
    (B = n * A) → 
    (A = 1) →
    (B + E = C) ∧ (B ∧ E ≤ F) :=
begin
  sorry
end

end triangle_square_ratio_l72_72960


namespace numOf25CentCoinsMore_l72_72655

-- Define the number of 5-cent coins
def num5CentCoins : Nat := x

-- Define the number of 10-cent coins in terms of the number of 5-cent coins
def num10CentCoins : Nat := x + 3

-- Define the total number of coins as given
def totalCoins : Nat := 23

-- Define the number of 25-cent coins in terms of 5-cent and 10-cent coins
def num25CentCoins : Nat := 23 - (x + (x + 3))

-- Define the total value constraint
def totalValue : Nat := 5 * x + 10 * (x + 3) + 25 * (20 - 2 * x)

-- State the proof problem
theorem numOf25CentCoinsMore : ∀ x : Nat, 5 * x + 10 * (x + 3) + 25 * (20 - 2 * x) = 320 → 20 - 2 * x = 8 → (20 - 2 * x) - x = 2 := by
  intros
  sorry

end numOf25CentCoinsMore_l72_72655


namespace b_divisible_by_8_l72_72458

variable (b : ℕ) (n : ℕ)
variable (hb_even : b % 2 = 0) (hb_pos : b > 0) (hn_gt1 : n > 1)
variable (h_square : ∃ k : ℕ, k^2 = (b^n - 1) / (b - 1))

theorem b_divisible_by_8 : b % 8 = 0 :=
by
  sorry

end b_divisible_by_8_l72_72458


namespace data_center_connections_l72_72192

-- Define the total number of switches and their connections
def num_switches : ℕ := 30
def connections_per_switch : ℕ := 4

-- Define the expected number of connections
def expected_connections : ℕ := 60

-- The theorem to prove the number of connections needed
theorem data_center_connections : 
  (num_switches * connections_per_switch) / 2 = expected_connections := 
begin
  -- We start with displaying the mathematical equivalence
  let total_connections := num_switches * connections_per_switch,
  have h : total_connections = 120, from sorry,
  show total_connections / 2 = expected_connections, from sorry,
end

end data_center_connections_l72_72192


namespace range_of_a_l72_72836

def A (x : ℝ) : Prop := x^2 - 4 * x + 3 ≤ 0
def B (x : ℝ) (a : ℝ) : Prop := x^2 - a * x < x - a

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) ∧ ∃ x, ¬ (A x → B x a) ↔ 1 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l72_72836


namespace circumcircle_of_triangle_FPQ_l72_72942

theorem circumcircle_of_triangle_FPQ :
  ∀ (F Q P : Π (x y : ℝ), Prop),
    (∀ (x y : ℝ), F x y ↔ x = 0 ∧ y = 1) →
    (∀ (x y : ℝ), Q x y ↔ x = 0 ∧ y = -1) →
    (P 2 1 ∨ P (-2) 1) →
    (P = λ x y, x^2 = 4 * y) →
    ((x - 1)^2 + y^2 = 2) ∨ ((x + 1)^2 + y^2 = 2) :=
begin
  intros F Q P hF hQ hP hParabola,
  sorry
end

end circumcircle_of_triangle_FPQ_l72_72942


namespace additional_spending_required_l72_72961

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end additional_spending_required_l72_72961


namespace avg_eq_pos_diff_l72_72790

theorem avg_eq_pos_diff (y : ℝ) (h : (35 + y) / 2 = 42) : |35 - y| = 14 := 
sorry

end avg_eq_pos_diff_l72_72790


namespace find_a_value_l72_72189

noncomputable def collinear (points : List (ℚ × ℚ)) := 
  ∃ a b c, ∀ (x y : ℚ), (x, y) ∈ points → a * x + b * y + c = 0

theorem find_a_value (a : ℚ) :
  collinear [(3, -5), (-a + 2, 3), (2*a + 3, 2)] → a = -7 / 23 :=
by
  sorry

end find_a_value_l72_72189


namespace not_hyperbola_condition_l72_72280

theorem not_hyperbola_condition (m : ℝ) (x y : ℝ) (h1 : 1 ≤ m) (h2 : m ≤ 3) :
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m) :=
sorry

end not_hyperbola_condition_l72_72280


namespace equidistant_points_ellipse_hyperbola_ray_l72_72147

variables {α : Type*} [metric_space α]

def equidistant_set (A O : α) (R : ℝ) (X : α) : Prop :=
  dist X O = dist X A + R ∨  -- A is outside the circle
  dist X O + dist X A = R ∨  -- A is inside the circle
  (dist X O = dist X A ∧ ∃ R : set α, is_ray R O A X)  -- A is on the circle

theorem equidistant_points_ellipse_hyperbola_ray {α : Type*} [metric_space α] 
(A O : α) (R : ℝ) : ∀ X : α, equidistant_set A O R X → 
(is_ellipse O A R X ∨ is_hyperbola O A R X ∨ is_ray (ray O A) X) :=
begin
  sorry
end

end equidistant_points_ellipse_hyperbola_ray_l72_72147


namespace sum_squares_mod_eq_6_l72_72599

def squares_mod (n : ℕ) : ℕ :=
  (List.range n).map (λ x => (x.succ * x.succ % 13))

noncomputable def sum_squares_mod : ℕ :=
  (squares_mod 15).sum 

theorem sum_squares_mod_eq_6 : sum_squares_mod % 13 = 6 := 
by
  sorry

end sum_squares_mod_eq_6_l72_72599


namespace extra_cubes_needed_to_complete_structure_l72_72822

theorem extra_cubes_needed_to_complete_structure :
  ∀ (num_initial_cubes : ℕ) (num_faces_per_cube : ℕ),
  num_initial_cubes = 7 →
  num_faces_per_cube = 6 →
  -- Initial structure is a single cube with one cube stuck to each of its faces.
  let final_structure_cubes := num_initial_cubes + num_faces_per_cube * (num_faces_per_cube - 1) 
  in final_structure_cubes - num_initial_cubes = 18 :=
by sorry

end extra_cubes_needed_to_complete_structure_l72_72822


namespace not_perfect_square_l72_72658

theorem not_perfect_square (n : ℕ) (h_len : n.digits 10 = 1000) 
    (h_digit : ∀ i, i < 1000 → n.digits 10 i = 5 ∨ (∃ j, j < 1000 ∧ n.digits 10 j = 6 ∧ i ≠ j)) : ¬ ∃ k, k * k = n :=
sorry

end not_perfect_square_l72_72658


namespace revive_Ivan_Tsarevich_after_4_hours_l72_72239

theorem revive_Ivan_Tsarevich_after_4_hours : 
  ∀ (D W R S : ℕ) (v_w v_r : ℕ) (r_w : ℚ),
  D = 20 ∧ 
  W = 1 ∧ 
  R = 0.5 ∧ 
  S = 0.25 ∧ 
  v_w = 3 ∧ 
  v_r = 6 →
  (4 = 2 / R + (D - 2 * v_w) / (v_w + v_r)) := 
by
  sorry

end revive_Ivan_Tsarevich_after_4_hours_l72_72239


namespace jenna_hourly_wage_l72_72092

noncomputable def calculate_hourly_wage (ticket_cost : ℝ)
                                       (drink_ticket_cost : ℝ)
                                       (drink_tickets_count : ℕ)
                                       (weekly_hours : ℝ)
                                       (monthly_spending_percentage : ℝ)
                                       (weeks_per_month : ℝ)
                                       (monthly_spending : ℝ) : ℝ :=
  let total_cost := ticket_cost + drink_ticket_cost * drink_tickets_count
  let monthly_salary := monthly_spending / monthly_spending_percentage
  let hours_per_month := weekly_hours * weeks_per_month
  (monthly_salary / hours_per_month)

theorem jenna_hourly_wage : 
  calculate_hourly_wage 181 7 5 30 0.1 4.33 216 ≈ 16.63 := 
by
  sorry

end jenna_hourly_wage_l72_72092


namespace average_age_of_two_new_men_l72_72161

theorem average_age_of_two_new_men :
  ∀ (A N : ℕ), 
    (∀ n : ℕ, n = 12) → 
    (N = 21 + 23 + 12) → 
    (A = N / 2) → 
    A = 28 :=
by
  intros A N twelve men_replace_eq_avg men_avg_eq
  sorry

end average_age_of_two_new_men_l72_72161


namespace probability_A_B_different_rooms_l72_72006

open Classical
noncomputable theory

-- Define the rooms, persons, choices and the probability
def rooms := {1, 2}
def persons := {A, B}

def numChoices : ℕ := 2
def totalOutcomes : ℕ := numChoices * numChoices

def differentRoomOutcomes : ℕ := 2

def probability_different_rooms := differentRoomOutcomes / totalOutcomes

theorem probability_A_B_different_rooms : 
  probability_different_rooms = 1 / 2 :=
by
  sorry

end probability_A_B_different_rooms_l72_72006


namespace farmer_milk_production_l72_72248

-- Definitions based on conditions
def total_cows (c : ℕ) : Prop := 0.4 * c = 50
def female_cows (c : ℕ) : ℕ := (0.6 * c).toNat
def milk_per_day (f : ℕ) : ℕ := 2 * f

-- Theorem to prove the farmer gets 150 gallons of milk a day
theorem farmer_milk_production : ∀ (c : ℕ), total_cows c → milk_per_day (female_cows c) = 150 := by
  intros c hc
  sorry

end farmer_milk_production_l72_72248


namespace average_monthly_balance_l72_72283

theorem average_monthly_balance :
  let balances := [100, 200, 250, 50, 300, 300]
  (balances.sum / balances.length : ℕ) = 200 :=
by
  sorry

end average_monthly_balance_l72_72283


namespace quadrilateral_area_l72_72270

theorem quadrilateral_area (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * |a - b| * |a + b| = 32) : a + b = 8 :=
by
  sorry

end quadrilateral_area_l72_72270


namespace reasoning_correct_l72_72574

theorem reasoning_correct :
  (∀ (T : Type) [topological_space T] (A B C D : T) 
    (area : T → T → T → T → ℝ),
      ∀ (P : T), P = (λ t, t) →
      area A B C P + area A B D P + area B C D P > area A B C D) ∧
  (∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →
    let an_arithmetic := (∀ n, (∑ i in finset.range (10 - 6 + 1), a (i + 6)) / 5 = (∑ i in finset.range 15, a (i + 1)) / 15) in
    (∀ (b : ℕ → ℝ), 
      (∀ n, b (n + 1) / b n = b 2 / b 1) →
      let b_geometric := ∀ n, real.sqrt (b 6 * b 7 * b 8 * b 9 * b 10) = real.sqrt (b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 * b 8 * b 9 * b 10 * b 11 * b 12 * b 13 * b 14 * b 15) in
    an_arithmetic ∧ b_geometric)) :=
by sorry

end reasoning_correct_l72_72574


namespace points_per_question_l72_72988

theorem points_per_question
    (questions_first_half : ℕ)
    (questions_second_half : ℕ)
    (final_score : ℕ)
    (h1 : questions_first_half = 6)
    (h2 : questions_second_half = 4)
    (h3 : final_score = 30) :
    (final_score / (questions_first_half + questions_second_half)) = 3 :=
by
  dsimp
  rw [h1, h2, h3]
  norm_num
  simp
  sorry

end points_per_question_l72_72988


namespace equal_distances_sum_red_blue_l72_72510

-- Definitions
variables (A B M : Point) (n : ℕ) (points : List Point)
noncomputable def segment_midpoint : Prop :=
  ∀ p ∈ points, (dist A p + dist B p) = dist A B

noncomputable def symmetrical_points : Prop :=
  points.length = 2 * n ∧
  ∃ M, ∀ p ∈ points, ∃ q ∈ points, q ≠ p ∧ dist M p = dist M q

noncomputable def coloring_points : Prop :=
  ∃ red_points blue_points : List Point,
  red_points.length = n ∧ blue_points.length = n ∧
  (∀ p ∈ red_points, p ∈ points) ∧ (∀ p ∈ blue_points, p ∈ points) ∧
  disjoint red_points blue_points ∧ append red_points blue_points = points

-- Theorem to prove
theorem equal_distances_sum_red_blue 
  (H1 : segment_midpoint A B M)
  (H2 : symmetrical_points points n)
  (H3 : coloring_points points n) : 
    (sum (dist A) (red_points)) = (sum (dist B) (blue_points)) :=
sorry

end equal_distances_sum_red_blue_l72_72510


namespace coefficient_of_m_degree_of_m_l72_72547

variable (a b : ℚ) -- Declaring a and b as rational numbers

/-- Let m be the monomial -5/8 * a * b^3 --/
def m := - (5 / 8) * a * b ^ 3

/-- prove that the coefficient is -5/8 --/
theorem coefficient_of_m : is_coeff m (-5 / 8) :=
sorry

/-- prove that the degree of m is 4 --/
theorem degree_of_m : is_degree m 4 :=
sorry

end coefficient_of_m_degree_of_m_l72_72547


namespace range_of_a_l72_72018

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then -x + 3 * a else a ^ x

theorem range_of_a (a : ℝ) : 
  is_decreasing (f a) ↔ (1 / 3 ≤ a ∧ a < 1) :=
by sorry

end range_of_a_l72_72018


namespace sum_squares_mod_13_l72_72607

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 10 := by
  sorry

end sum_squares_mod_13_l72_72607


namespace sum_of_coordinates_eq_six_l72_72513

-- Define the coordinates of points C and D
structure Point where
  x : ℝ
  y : ℝ

def C : Point := { x := 3, y := y }
def D : Point := { x := 3, y := -y }

-- Statement of the theorem
theorem sum_of_coordinates_eq_six (y : ℝ) : 
  C.x + C.y + D.x + D.y = 6 :=
by
  -- sum the coordinates and verify the statement.
  simp [C, D, Point]
  sorry

end sum_of_coordinates_eq_six_l72_72513


namespace cars_entered_l72_72575

theorem cars_entered (initial_cars cars_left current_cars more_cars_entered : ℕ) 
  (h1 : initial_cars = 80)
  (h2 : cars_left = 13)
  (h3 : current_cars = 85)
  (h4 : more_cars_entered = current_cars - (initial_cars - cars_left)) : 
  more_cars_entered = 18 :=
by {
  rw [h1, h2, h3],
  sorry
}

end cars_entered_l72_72575


namespace avg_weight_of_class_is_approx_58_89_l72_72236

-- Define the conditions
def section_A_students := 50
def section_B_students := 40
def avg_weight_A := 50.0
def avg_weight_B := 70.0

-- Calculate the total weight of each section
def total_weight_A := avg_weight_A * section_A_students
def total_weight_B := avg_weight_B * section_B_students

-- Calculate the total weight and total number of students of the whole class
def total_weight := total_weight_A + total_weight_B
def total_students := section_A_students + section_B_students

-- Calculate the average weight of the whole class
def avg_weight := total_weight / total_students

-- The theorem to prove
theorem avg_weight_of_class_is_approx_58_89 : abs (avg_weight - 58.89) < 0.01 :=
by
  sorry

end avg_weight_of_class_is_approx_58_89_l72_72236


namespace sum_of_cubes_l72_72467

open Real

theorem sum_of_cubes (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
(h_eq : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 := 
by sorry

end sum_of_cubes_l72_72467


namespace complex_fraction_calculation_l72_72060

theorem complex_fraction_calculation (z : ℂ) (h : z = 2 + 1 * complex.I) : (2 * complex.I) / (z - 1) = 1 + complex.I :=
by
  sorry

end complex_fraction_calculation_l72_72060


namespace kendra_sites_visited_l72_72135

theorem kendra_sites_visited (x : ℕ) 
  (mon_birds : x * 7) 
  (tue_birds : x * 5)
  (wed_birds : 10 * 8 = 80)
  (avg_birds : (7 * x + 5 * x + 80) / (2 * x + 10) = 7) :
  2 * x = 10 :=
by
  sorry

end kendra_sites_visited_l72_72135


namespace gg5_of_3_l72_72476

def g (x : ℝ) : ℝ := 1 / (1 - x)

theorem gg5_of_3
  : g (g (g (g (g 3)))) = 3 := by
  sorry

end gg5_of_3_l72_72476


namespace range_of_a_for_zero_l72_72749

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_zero (a : ℝ) : a ≤ 2 * Real.log 2 - 2 → ∃ x : ℝ, f a x = 0 := by
  sorry

end range_of_a_for_zero_l72_72749


namespace range_of_a_l72_72837

open Set

variable (U : Set ℝ := @univ ℝ) 
variable (A : Set ℝ := { x | x > 1 })
variable (B : Set ℝ := { x | x > a })

noncomputable
def complement_A : Set ℝ := U \ A
noncomputable
def union_complement_A_B : Set ℝ := complement_A U A ∪ B

theorem range_of_a (a : ℝ) : union_complement_A_B U A B = @univ ℝ ↔ a ∈ Iic 1 := 
sorry

end range_of_a_l72_72837


namespace cube_edge_length_l72_72914

-- Define edge length and surface area
variables (edge_length surface_area : ℝ)

-- Given condition
def surface_area_condition : Prop := surface_area = 294

-- Cube surface area formula
def cube_surface_area : Prop := surface_area = 6 * edge_length^2

-- Proof statement
theorem cube_edge_length (h1: surface_area_condition surface_area) (h2: cube_surface_area edge_length surface_area) : edge_length = 7 := 
by
  sorry

end cube_edge_length_l72_72914


namespace verify_integer_count_l72_72402

noncomputable def count_integers_in_interval : ℤ :=
  let lower_bound := Int.ceil (-15 * Real.pi)
  let upper_bound := Int.floor (12 * Real.pi)
  upper_bound - lower_bound + 1

theorem verify_integer_count :
  count_integers_in_interval = 85 :=
by
  -- Definitions of -15*pi and 12*pi, their ceiling, and floor values
  have h1 : Int.ceil (-15 * Real.pi) = -47 := sorry -- Proof to determine the ceil value
  have h2 : Int.floor (12 * Real.pi) = 37 := sorry -- Proof to determine the floor value

  -- Use these ceil and floor values to count the number of integers
  rw [count_integers_in_interval, h1, h2]
  norm_num
  sorry

end verify_integer_count_l72_72402


namespace trig_identity_l72_72347

theorem trig_identity (α : ℝ) (h : Real.sin (π + α) = 1 / 2) : Real.cos (α - 3 / 2 * π) = 1 / 2 :=
  sorry

end trig_identity_l72_72347


namespace matt_total_score_l72_72503

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end matt_total_score_l72_72503


namespace smaller_circle_rolling_in_larger_circle_l72_72883

open EuclideanGeometry

-- Definition of points, circles, radius, and tangency
def Point : Type := ℝ × ℝ
def radius (R : ℝ) (center : Point) (p : Point) : Prop := dist center p = R

-- Prove that if a circle with radius R rolls inside another circle with radius 2R, 
-- any point on the smaller circle describes a straight line.
theorem smaller_circle_rolling_in_larger_circle (R : ℝ) (O A : Point) :
  (∀ p : Point, radius R A p) →
  (∀ p : Point, radius (2 * R) O p) →
  (∀ p : Point, tangent O (2 * R) A p) →
  ∃ L : Line, ∀ (B : Point), (radius R A B) → on_line B L :=
sorry

end smaller_circle_rolling_in_larger_circle_l72_72883


namespace f_increasing_on_m_eq_1_find_range_of_m_l72_72389

-- Definitions used in conditions
def f (x : ℝ) (m : ℝ) : ℝ := m / x + x * Real.log x
def g (x : ℝ) : ℝ := Real.log x - 2

-- Part 1:
theorem f_increasing_on_m_eq_1 : ∃ (s : Set ℝ), (1 < s) → (∀ x ∈ s, - (1 / x^2) + Real.log x + 1 > 0) :=
sorry

-- Part 2:
theorem find_range_of_m : ∃ (m : ℝ), (1 / 2 ≤ m ∧ m ≤ Real.exp 1) ∧
  ∀ (x1 x2 : ℝ), x1 ∈ Set.Icc 1 (Real.exp 1) → x2 ∈ Set.Icc 1 (Real.exp 1) →
  (f x1 m / x1) * (g x2 / x2) = -1 :=
sorry

end f_increasing_on_m_eq_1_find_range_of_m_l72_72389


namespace area_triangle_ABC_l72_72509

noncomputable theory
open_locale classical

structure Point :=
  (x : ℝ)
  (y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem area_triangle_ABC : area_of_triangle ⟨0, 0⟩ ⟨3, 0⟩ ⟨0, 4⟩ = 6 :=
begin
  sorry
end

end area_triangle_ABC_l72_72509


namespace find_second_number_l72_72543

theorem find_second_number 
  (x : ℕ)
  (h1 : (55 + x + 507 + 2 + 684 + 42) / 6 = 223)
  : x = 48 := 
by 
  sorry

end find_second_number_l72_72543


namespace dot_product_condition_l72_72398

variable {V : Type*} [InnerProductSpace ℝ V]

theorem dot_product_condition 
  (a b : V) 
  (ha : ‖a‖ = 3) 
  (hb : ‖b‖ = 8) 
  (hab : ‖(5/3 : ℝ) • a - b‖ = 7) : 
  inner a b = 12 :=
sorry

end dot_product_condition_l72_72398


namespace saving_is_zero_cents_l72_72651

-- Define the in-store and online prices
def in_store_price : ℝ := 129.99
def online_payment_per_installment : ℝ := 29.99
def shipping_and_handling : ℝ := 11.99

-- Define the online total price
def online_total_price : ℝ := 4 * online_payment_per_installment + shipping_and_handling

-- Define the saving in cents
def saving_in_cents : ℝ := (in_store_price - online_total_price) * 100

-- State the theorem to prove the number of cents saved
theorem saving_is_zero_cents : saving_in_cents = 0 := by
  sorry

end saving_is_zero_cents_l72_72651


namespace total_sleep_time_is_correct_l72_72645

-- Define the sleeping patterns of the animals
def cougar_sleep_even_days : ℕ := 4
def cougar_sleep_odd_days : ℕ := 6
def zebra_sleep_more : ℕ := 2

-- Define the distribution of even and odd days in a week
def even_days_in_week : ℕ := 3
def odd_days_in_week : ℕ := 4

-- Define the total weekly sleep time for the cougar
def cougar_total_weekly_sleep : ℕ := 
  (cougar_sleep_even_days * even_days_in_week) + 
  (cougar_sleep_odd_days * odd_days_in_week)

-- Define the total weekly sleep time for the zebra
def zebra_total_weekly_sleep : ℕ := 
  ((cougar_sleep_even_days + zebra_sleep_more) * even_days_in_week) + 
  ((cougar_sleep_odd_days + zebra_sleep_more) * odd_days_in_week)

-- Define the total weekly sleep time for both the cougar and the zebra
def total_weekly_sleep : ℕ := 
  cougar_total_weekly_sleep + zebra_total_weekly_sleep

-- Prove that the total weekly sleep time for both animals is 86 hours
theorem total_sleep_time_is_correct : total_weekly_sleep = 86 :=
by
  -- skipping proof
  sorry

end total_sleep_time_is_correct_l72_72645


namespace sum_of_solutions_l72_72369

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : x1 + log x1 = 3) (h2 : x2 + 10^x2 = 3) :
  x1 + x2 = 6 :=
sorry

end sum_of_solutions_l72_72369


namespace parity_equivalence_l72_72246

theorem parity_equivalence (p q : ℕ) :
  (Even (p^3 - q^3)) ↔ (Even (p + q)) :=
by
  sorry

end parity_equivalence_l72_72246


namespace find_k_l72_72657

theorem find_k (n : ℕ) (h1 : finset.card (finset.filter (λ d, d ∣ n) (finset.Icc 1 n)) = 72) (h2 : finset.card (finset.filter (λ d, d ∣ (5 * n)) (finset.Icc 1 (5 * n))) = 90) : 
  ∃ k : ℕ, (k = 3) ∧ (∃ m : ℕ, (m % 5 ≠ 0) ∧ (n = 5^k * m)) :=
by
  sorry

end find_k_l72_72657


namespace sum_of_a_vals_l72_72333

theorem sum_of_a_vals :
  (∑ a in {a : ℝ | ∃ x : ℝ, (x - a)^2 + (x^2 - 3 * x + 2)^2 = 0}, a) = 3 :=
by sorry

end sum_of_a_vals_l72_72333


namespace sum_squares_mod_13_l72_72610

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 10 := by
  sorry

end sum_squares_mod_13_l72_72610


namespace total_pictures_l72_72777

theorem total_pictures :
  let Randy_pictures := 5
  let Peter_pictures := Randy_pictures + 3
  let Quincy_pictures := Peter_pictures + 20
  let Susan_pictures := 2 * Quincy_pictures - 7
  let Thomas_pictures := Randy_pictures ^ 3
  Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by 
    let Randy_pictures := 5
    let Peter_pictures := Randy_pictures + 3
    let Quincy_pictures := Peter_pictures + 20
    let Susan_pictures := 2 * Quincy_pictures - 7
    let Thomas_pictures := Randy_pictures ^ 3
    sorry

end total_pictures_l72_72777


namespace min_M_equals_log2_l72_72844

noncomputable def a (x y z : ℝ) : ℝ := log 10 z + log 10 ((x / (y * z)) + 1)
noncomputable def b (x y z : ℝ) : ℝ := log 10 (1 / x) + log 10 (x * y * z + 1)
noncomputable def c (x y z : ℝ) : ℝ := log 10 y + log 10 ((1 / (x * y * z)) + 1)

def M (x y z : ℝ) : ℝ := max (a x y z) (max (b x y z) (c x y z))

theorem min_M_equals_log2 : (∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 → M x y z ≥ log 10 2) ∧
                           (M 1 1 1 = log 10 2) := 
sorry

end min_M_equals_log2_l72_72844


namespace probability_heads_equals_7_over_11_l72_72487

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l72_72487


namespace circumpasses_orthocenter_l72_72881

variable {A B C H A₁ B₁ C₁ : Type}
variables [points_notation : Point A₁] [points_notation : Point B₁] [points_notation : Point C₁]
variables [triangle_notation : Triangle A B C]

def in_triangle (p : Point) (tri : Triangle) : Prop := 
  ∃ α β γ : ℝ, α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧ α * tri.a + β * tri.b + γ * tri.c = p

def area (tri : Triangle) : ℝ := 
  let ⟨a, b, c⟩ := tri.angles in 
  1 / 2 * ℝ.abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

def circumpasses (tri : Triangle) (p : Point) : Prop := 
  let ⟨a, b, c⟩ := tri.angles in 
  Distance a p = Distance b p ∧ Distance b p = Distance c p

theorem circumpasses_orthocenter 
  (h₁ : ∀ (A₁ : Point), in_triangle A₁ (altitude A))
  (h₂ : ∀ (B₁ : Point), in_triangle B₁ (altitude B))
  (h₃ : ∀ (C₁ : Point), in_triangle C₁ (altitude C))
  (h₄ : area (triangle ABC₁) + area (triangle BCA₁) + area (triangle CAB₁) = area (triangle ABC)) :
  circumpasses (triangle A₁ B₁ C₁) H :=
by sorry

end circumpasses_orthocenter_l72_72881


namespace proof_problem_l72_72529

theorem proof_problem (n : ℕ) : 
  let lhs := 4 * (10^(2*n-1)) + 4 * (10^(2*n-2)) + ... + 4 * (10^n) + 8 * (10^(n-1)) + ... + 8 * (10) + 9,
      rhs := (6 * (10^(n-1)) + 6 * (10^(n-2)) + ... + 6 * 10 + 7) ^ 2 in
  lhs = rhs := 
   sorry

end proof_problem_l72_72529


namespace garden_max_area_l72_72454

theorem garden_max_area (l w : ℝ) (h_fence : l + 2 * w = 160) :
  (∀ A : ℝ, (A = l * w) → A ≤ (3200 : ℝ)) :=
by
  unfold
  sorry

end garden_max_area_l72_72454


namespace triangle_inequality_l72_72866

theorem triangle_inequality (A B C D E : Point)
  (AB AC BC DE : ℝ)
  (hD : is_on_segment A B D)
  (hE : is_on_segment A C E)
  (h_parallel : DE ∥ BC)
  (h_tangent : tangent_to_incircle DE (triangle A B C)) :
  AB + BC + CA ≥ 8 * DE :=
begin
  sorry
end

end triangle_inequality_l72_72866


namespace factorize_one_factorize_two_l72_72323

variable (m x y : ℝ)

-- Problem statement for Question 1
theorem factorize_one (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := 
sorry

-- Problem statement for Question 2
theorem factorize_two (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := 
sorry

end factorize_one_factorize_two_l72_72323


namespace rent_expense_per_month_l72_72524

def salary_per_month := 5000
def tax_rate := 0.10
def net_salary (salary : ℕ) (tax_rate : ℕ) : ℕ :=
  salary - (salary * tax_rate)
def fraction_of_salary (fraction : ℚ) (salary : ℕ) : ℕ :=
  fraction.num * salary / fraction.denom
def total_rent_after_tax (salary : ℕ) (fraction : ℚ) (tax_rate : ℚ) : ℕ :=
  fraction_of_salary fraction (net_salary salary tax_rate)

-- Given conditions
def late_rent_fraction := 3 / 5
def late_rent_instances := 2

-- The final proof statement
theorem rent_expense_per_month :
  let net_salary := net_salary salary_per_month tax_rate
  let total_rent := total_rent_after_tax salary_per_month late_rent_fraction tax_rate
  let monthly_rent := total_rent / late_rent_instances
  monthly_rent = 1350 := 
by {
  -- In this part we ideally solve the above step by step as provided in the example
  sorry
}

end rent_expense_per_month_l72_72524


namespace sum_of_squares_even_2_to_14_l72_72216

theorem sum_of_squares_even_2_to_14 : (∑ n in {2, 4, 6, 8, 10, 12, 14}, n^2) = 560 := by
  -- sum of squares of the given even numbers
  sorry

end sum_of_squares_even_2_to_14_l72_72216


namespace imaginary_part_of_z_l72_72008

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + complex.i) = 4) : complex.im z = -2 :=
sorry

end imaginary_part_of_z_l72_72008


namespace twenty_percent_of_x_l72_72787

noncomputable def x := 1800 / 1.2

theorem twenty_percent_of_x (h : 1.2 * x = 1800) : 0.2 * x = 300 :=
by
  -- The proof would go here, but we'll replace it with sorry.
  sorry

end twenty_percent_of_x_l72_72787


namespace sum_of_squares_mod_13_l72_72602

theorem sum_of_squares_mod_13 :
  ((∑ i in Finset.range 16, i^2) % 13) = 3 :=
by 
  sorry

end sum_of_squares_mod_13_l72_72602


namespace kyunghoon_time_to_go_down_l72_72102

theorem kyunghoon_time_to_go_down (d : ℕ) (t_up t_down total_time : ℕ) : 
  ((t_up = d / 3) ∧ (t_down = (d + 2) / 4) ∧ (total_time = 4) → (t_up + t_down = total_time) → (t_down = 2)) := 
by
  sorry

end kyunghoon_time_to_go_down_l72_72102


namespace regular_tetrahedron_conditions_l72_72120

-- Definition of the problem in mathematical terms
variable {S A B C : Type} [tetrahedron S A B C]

-- Existence of spheres that are tangent to the edges of the tetrahedron
axiom exists_tangent_spheres {S A B C : Type} [tetrahedron S A B C] :
  ∃ (σ₁ σ₂ σ₃ σ₄ σ₅ : sphere), tangent σ₁ S A ∧ tangent σ₂ S B ∧ tangent σ₃ S C ∧ 
  tangent σ₄ A B ∧ tangent σ₅ B C ∧ tangent σ₆ C A

-- Theorem to prove (I) and (II)
theorem regular_tetrahedron_conditions {S A B C : Type} [tetrahedron S A B C]
  (h₁ : ∃ (σ₁ σ₂ σ₃ σ₄ σ₅ : sphere), tangent σ₁ S A ∧ tangent σ₂ S B ∧ tangent σ₃ S C ∧ 
  tangent σ₄ A B ∧ tangent σ₅ B C ∧ tangent σ₆ C A) :
  regular S A B C ∧
  (∀ S A B C : Type, regular S A B C → ∃ (σ₁ σ₂ σ₃ σ₄ σ₅ : sphere), 
    tangent σ₁ S A ∧ tangent σ₂ S B ∧ tangent σ₃ S C ∧ 
    tangent σ₄ A B ∧ tangent σ₅ B C ∧ tangent σ₆ C A) :=
sorry

end regular_tetrahedron_conditions_l72_72120


namespace part1_max_min_part2_min_g_l72_72305

-- Part 1
theorem part1_max_min (a b c M m : ℝ) (f : ℝ → ℝ) :
  (∀ x ∈ [-2, 2], f x = a * x^2 + b * x + c) →
  (A = {x : ℝ | f x = x}) →
  (A = {1, 2}) →
  (f 0 = 2) →
  (m = f 1) →
  (M = f 2) →
  (m = 1) ∧ (M = 10) :=
begin
  sorry
end

-- Part 2
theorem part2_min_g (a b c M m : ℝ) (g : ℝ → ℝ) :
  (∀ x ∈ [-2, 2], f x = a * x^2 + b * x + c) →
  (A = {x : ℝ | f x = x}) →
  (A = {1}) →
  (a ≥ 1) →
  (f 0 = 2) →
  (m = f 1) →
  (M = f 2) →
  g a = M + m →
  (∀ a ≥ 1, g a = 9 * a - 1 - 1/(4 * a)) →
  g 1 = 31/4 :=
begin
  sorry
end

end part1_max_min_part2_min_g_l72_72305


namespace classroom_problem_l72_72233

noncomputable def classroom_problem_statement : Prop :=
  ∀ (B G : ℕ) (b g : ℝ),
    b > 0 →
    g > 0 →
    B > 0 →
    G > 0 →
    ¬ ((B * g + G * b) / (B + G) = b + g ∧ b > 0 ∧ g > 0)

theorem classroom_problem : classroom_problem_statement :=
  by
    intros B G b g hb_gt0 hg_gt0 hB_gt0 hG_gt0
    sorry

end classroom_problem_l72_72233


namespace circumference_of_semicircle_is_correct_l72_72235

-- Define the given conditions
def length : ℝ := 20
def breadth : ℝ := 16
def rectangle_perimeter : ℝ := 2 * (length + breadth)
def side_of_square : ℝ := rectangle_perimeter / 4
def diameter_of_semicircle : ℝ := side_of_square

-- Define the value of π (pi)
def pi : ℝ := 3.14

-- Define the expected circumference of the semicircle
def expected_circumference : ℝ := 46.26

-- Prove the problem statement
theorem circumference_of_semicircle_is_correct :
  (pi * diameter_of_semicircle) / 2 + diameter_of_semicircle = expected_circumference :=
by
  -- Add proof details here
  sorry

end circumference_of_semicircle_is_correct_l72_72235


namespace water_level_after_valve_opened_l72_72201

-- Given conditions
def h : ℝ := 40  -- initial height in cm
def ρ_water : ℝ := 1000  -- density of water in kg/m^3
def ρ_oil : ℝ := 700  -- density of oil in kg/m^3

-- Lean statement to prove
theorem water_level_after_valve_opened :
  let h1 := (ρ_oil * h) / (ρ_water + ρ_oil) in
  h1 = 280 / 17 :=
by
  sorry

end water_level_after_valve_opened_l72_72201


namespace frank_money_remaining_l72_72346

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l72_72346


namespace remaining_unit_area_l72_72907

theorem remaining_unit_area {
  total_units : ℕ,
  total_area : ℕ,
  unit_area : ℕ,
  specific_units : ℕ,
  remaining_units : ℕ
} (h1 : total_units = 42)
  (h2 : total_area = 5040)
  (h3 : unit_area = 8 * 4)
  (h4 : specific_units = 20)
  (h5 : remaining_units = total_units - specific_units) :
  total_area - specific_units * unit_area = 200 * remaining_units :=
by sorry

end remaining_unit_area_l72_72907


namespace sum_of_first_10_terms_l72_72745

variable (a b : ℕ → ℕ)
variable (d1 d2 : ℕ)

-- Definitions for conditions
def is_arithmetic_sequence (s : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, s (n + 1) = s n + d

def seq_coeffs : Prop :=
  is_arithmetic_sequence a d1 ∧ is_arithmetic_sequence b d2 ∧ a 1 = 25 ∧ b 1 = 75 ∧ a 100 + b 100 = 100

-- The theorem we need to prove
theorem sum_of_first_10_terms :
  seq_coeffs a b d1 d2 →
  (∑ i in Finset.range 10, (a i + b i)) = 1000 :=
by
  sorry

end sum_of_first_10_terms_l72_72745


namespace number_of_positive_real_solutions_l72_72043

noncomputable def p (x : ℝ) : ℝ := x^12 + 5 * x^11 + 20 * x^10 + 1300 * x^9 - 1105 * x^8

theorem number_of_positive_real_solutions : ∃! x : ℝ, 0 < x ∧ p x = 0 :=
sorry

end number_of_positive_real_solutions_l72_72043


namespace peanut_raising_ratio_l72_72643

theorem peanut_raising_ratio
  (initial_peanuts : ℝ)
  (remove_peanuts_1 : ℝ)
  (add_raisins_1 : ℝ)
  (remove_mixture : ℝ)
  (add_raisins_2 : ℝ)
  (final_peanuts : ℝ)
  (final_raisins : ℝ)
  (ratio : ℝ) :
  initial_peanuts = 10 ∧
  remove_peanuts_1 = 2 ∧
  add_raisins_1 = 2 ∧
  remove_mixture = 2 ∧
  add_raisins_2 = 2 ∧
  final_peanuts = initial_peanuts - remove_peanuts_1 - (remove_mixture * (initial_peanuts - remove_peanuts_1) / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) ∧
  final_raisins = add_raisins_1 - (remove_mixture * add_raisins_1 / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) + add_raisins_2 ∧
  ratio = final_peanuts / final_raisins →
  ratio = 16 / 9 := by
  sorry

end peanut_raising_ratio_l72_72643


namespace rect_eq_C2_max_dist_C2_C1_l72_72374

def polar_eq_C1 (rho θ : ℝ) : Prop := rho * cos(θ - π / 3) = -1

def polar_eq_C2 (rho θ : ℝ) : Prop := rho = 2 * sqrt 2 * cos (θ - π / 4)

theorem rect_eq_C2 (x y : ℝ) :
  (∃ (θ : ℝ), polar_eq_C2 (sqrt (x^2 + y^2)) θ ∧ x = sqrt (x^2 + y^2) * cos θ ∧ y = sqrt (x^2 + y^2) * sin θ) ↔
  (x - 1)^2 + (y - 1)^2 = 2 := by
  sorry

theorem max_dist_C2_C1 :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 →
  ∃ (d : ℝ), (∀ (x1 y1 : ℝ), polar_eq_C1 (sqrt (x1^2 + y1^2)) (atan2 y1 x1) →
  d = (|x1 + sqrt 3 * y1 + 2| / sqrt (1 + 3)) + sqrt 2)
  ∧ d = (3 + sqrt 3 + 2 * sqrt 2) / 2) := by
  sorry

end rect_eq_C2_max_dist_C2_C1_l72_72374


namespace max_puzzle_sets_l72_72860

theorem max_puzzle_sets 
  (total_logic : ℕ) (total_visual : ℕ) (total_word : ℕ)
  (h1 : total_logic = 36) (h2 : total_visual = 27) (h3 : total_word = 15)
  (x y : ℕ)
  (h4 : 7 ≤ 4 * x + 3 * x + y ∧ 4 * x + 3 * x + y ≤ 12)
  (h5 : 4 * x / 3 * x = 4 / 3)
  (h6 : y ≥ 3 * x / 2) :
  5 ≤ total_logic / (4 * x) ∧ 5 ≤ total_visual / (3 * x) ∧ 5 ≤ total_word / y :=
sorry

end max_puzzle_sets_l72_72860


namespace gain_percentage_is_four_l72_72665

theorem gain_percentage_is_four 
  (CP : ℝ) 
  (hCP : CP = 1200) 
  (loss_percentage : ℝ)
  (h_loss_percentage : loss_percentage = 0.1) 
  (additional_price : ℝ)
  (h_additional_price : additional_price = 168) 
  (gain_percentage : ℝ) :
  let initial_SP := CP - (loss_percentage * CP),
      new_SP := initial_SP + additional_price,
      gain_amount := new_SP - CP in
  gain_percentage = (gain_amount / CP) * 100 :=
by
  sorry

end gain_percentage_is_four_l72_72665


namespace volume_of_circumscribed_sphere_of_cone_correct_l72_72817

noncomputable def volume_of_circumscribed_sphere_of_cone (P A B C : ℝ) (h1 : P - A - B = 2) (h2 : A - C = 1) (h3 : P ⊥ A - B - C) (h4 : A - C ⊥ A - B) : ℝ :=
  (4 / 3) * π * ((sqrt (2^2 + 1^2 + 2^2)) / 2)^3

theorem volume_of_circumscribed_sphere_of_cone_correct (P A B C : ℝ) (h1 : P - A - B = 2) (h2 : A - C = 1) (h3 : P ⊥ A - B - C) (h4 : A - C ⊥ A - B) :
  volume_of_circumscribed_sphere_of_cone P A B C h1 h2 h3 h4 = (9 / 2) * π :=
by
  sorry

end volume_of_circumscribed_sphere_of_cone_correct_l72_72817


namespace tan_half_product_values_l72_72408

theorem tan_half_product_values (a b : ℝ) (h : 3 * (Real.sin a + Real.sin b) + 2 * (Real.sin a * Real.sin b + 1) = 0) : 
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = -4 ∨ x = -1) := sorry

end tan_half_product_values_l72_72408


namespace return_journey_steps_l72_72424

-- Definitions of prime and composite
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

-- Steps function definition
def steps (n : ℕ) : ℤ :=
  if n = 1 then 0
  else if is_prime n then 2
  else if is_composite n then -3
  else 0

-- Sum of steps
def total_steps : ℤ :=
  (Finset.range 31).sum steps

theorem return_journey_steps : |total_steps| = 37 :=
  sorry

end return_journey_steps_l72_72424


namespace sum_abc_eq_8_l72_72900

theorem sum_abc_eq_8 (a b c : ℝ) 
  (h : (a - 5) ^ 2 + (b - 6) ^ 2 + (c - 7) ^ 2 - 2 * (a - 5) * (b - 6) = 0) : 
  a + b + c = 8 := 
sorry

end sum_abc_eq_8_l72_72900


namespace teddy_bear_cost_l72_72857

theorem teddy_bear_cost : 
  ∀ (n : ℕ) (cost_per_toy : ℕ) 
  (total_cost : ℕ) (num_teddy_bears : ℕ) 
  (amount_in_wallet : ℕ) (cost_per_bear : ℕ),
  n = 28 → 
  cost_per_toy = 10 → 
  num_teddy_bears = 20 → 
  amount_in_wallet = 580 → 
  total_cost = 280 → 
  total_cost = n * cost_per_toy →
  (amount_in_wallet - total_cost) = num_teddy_bears * cost_per_bear →
  cost_per_bear = 15 :=
by 
  intros n cost_per_toy total_cost num_teddy_bears amount_in_wallet cost_per_bear 
         hn hcost_per_toy hnum_teddy_bears hamount_in_wallet htotal_cost htotal_cost_eq
        hbear_cost_eq,
  sorry

end teddy_bear_cost_l72_72857


namespace fraction_equivalent_to_decimal_l72_72208

theorem fraction_equivalent_to_decimal : 
  ∃ (x : ℚ), x = 0.6 + 0.0037 * (1 / (1 - 0.01)) ∧ x = 631 / 990 :=
by
  sorry

end fraction_equivalent_to_decimal_l72_72208


namespace seed_selection_valid_l72_72965

def seeds : List Nat := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07]

def extractValidSeeds (lst : List Nat) (startIndex : Nat) (maxValue : Nat) (count : Nat) : List Nat :=
  lst.drop startIndex
  |>.filter (fun n => n < maxValue)
  |>.take count

theorem seed_selection_valid :
  extractValidSeeds seeds 10 850 4 = [169, 555, 671, 105] :=
by
  sorry

end seed_selection_valid_l72_72965


namespace find_m_l72_72751

theorem find_m (S : ℕ → ℝ) (m : ℝ) (h : ∀ n, S n = m * 2^(n-1) - 3) : m = 6 :=
by
  sorry

end find_m_l72_72751


namespace chromium_percentage_in_second_alloy_l72_72433

-- Define the conditions
def first_alloy_chromium_percentage : ℝ := 10 / 100
def first_alloy_weight : ℝ := 15
def third_alloy_chromium_percentage : ℝ := 8.6 / 100
def total_weight : ℝ := first_alloy_weight + 35

-- Define the amount of chromium from the first alloy
def chromium_in_first_alloy : ℝ := first_alloy_chromium_percentage * first_alloy_weight

-- Define the equation based on the amount of chromium in the new alloy
theorem chromium_percentage_in_second_alloy (x : ℝ) :
  chromium_in_first_alloy + (x / 100 * 35) = third_alloy_chromium_percentage * total_weight → 
  x = 8 := 
by 
  sorry

end chromium_percentage_in_second_alloy_l72_72433


namespace trigonometric_identity_l72_72994

variable (α : ℝ)

theorem trigonometric_identity :
  4.9 * (Real.sin (7 * Real.pi / 8 - 2 * α))^2 - (Real.sin (9 * Real.pi / 8 - 2 * α))^2 = 
  Real.sin (4 * α) / Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l72_72994


namespace complement_not_subset_l72_72726

-- Define sets A and B
def A : set ℕ := {0, 1, 2}
def B : set ℕ := {0, 1}

-- Prove that the complement of B relative to A is not a subset of B
theorem complement_not_subset :
  (A \ B) ⊆ B = false := by
{
  sorry
}

end complement_not_subset_l72_72726


namespace sum_of_integers_l72_72562

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 103) 
                        (h2 : Nat.gcd a b = 1) 
                        (h3 : a < 20) 
                        (h4 : b < 20) : 
                        a + b = 19 :=
  by sorry

end sum_of_integers_l72_72562


namespace cylinder_unfolded_curve_is_sine_l72_72646

noncomputable def unfolded_cylinder_curve : curve :=
  sorry

theorem cylinder_unfolded_curve_is_sine :
  (unfolded_cylinder_curve cylinder base_radius angle) = sine_curve :=
sorry

end cylinder_unfolded_curve_is_sine_l72_72646


namespace coefficient_of_pi_x_over_5_l72_72546

-- Definition of the function where we find the coefficient
def coefficient_of_fraction (expr : ℝ) : ℝ := sorry

-- Statement with proof obligation
theorem coefficient_of_pi_x_over_5 :
  coefficient_of_fraction (π * x / 5) = π / 5 :=
sorry

end coefficient_of_pi_x_over_5_l72_72546


namespace sum_of_squares_mod_13_l72_72611

theorem sum_of_squares_mod_13 :
  (∑ k in Finset.range 16, k^2) % 13 = 10 :=
by
  sorry

end sum_of_squares_mod_13_l72_72611


namespace weight_of_lightest_dwarf_l72_72896

noncomputable def weight_of_dwarf (n : ℕ) (x : ℝ) : ℝ := 5 - (n - 1) * x

theorem weight_of_lightest_dwarf :
  ∃ x : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 101 → weight_of_dwarf 1 x = 5) ∧
    (weight_of_dwarf 76 x + weight_of_dwarf 77 x + weight_of_dwarf 78 x + weight_of_dwarf 79 x + weight_of_dwarf 80 x =
     weight_of_dwarf 96 x + weight_of_dwarf 97 x + weight_of_dwarf 98 x + weight_of_dwarf 99 x + weight_of_dwarf 100 x + weight_of_dwarf 101 x) →
    weight_of_dwarf 101 x = 2.5 :=
by
  sorry

end weight_of_lightest_dwarf_l72_72896


namespace second_investment_percentage_l72_72066

-- Define the conditions
def total_investment : ℝ := 2000
def investment_at_10_percent : ℝ := 1400
def investment_at_p_percent : ℝ := 600
def income_difference : ℝ := 92

-- Define the annual incomes
def income_from_10_percent_investment : ℝ := 0.10 * investment_at_10_percent
def income_from_p_percent_investment (P : ℝ) : ℝ := (P / 100) * investment_at_p_percent

-- Lean theorem statement
theorem second_investment_percentage : 
  ∃ P : ℝ, income_from_10_percent_investment - income_from_p_percent_investment P = income_difference ∧ P = 8 :=
by 
  sorry

end second_investment_percentage_l72_72066


namespace find_students_contrib_l72_72262

-- Variables and conditions
variables (total_collected : ℕ) (months : ℕ) (total_monthly : ℕ)
variables (n_students : ℕ) (contribution : ℕ)

-- Given data
def given_data : Prop :=
  total_collected = 49685 ∧ months = 5 ∧ total_monthly = 9937 ∧ (9937 = 19 * 523)

-- Prove the group size and contribution per student
theorem find_students_contrib (h : given_data) : 
  ∃ n_students contribution, (n_students * contribution * months = total_collected)
  ∧ (n_students = 19 ∧ contribution = 523) := 
sorry

end find_students_contrib_l72_72262


namespace saltwater_solution_l72_72232

theorem saltwater_solution (x : ℝ) (h1 : ∃ v : ℝ, v = x ∧ v * 0.2 = 0.20 * x)
(h2 : 3 / 4 * x = 3 / 4 * x)
(h3 : ∃ v' : ℝ, v' = 3 / 4 * x + 6 + 12)
(h4 : (0.20 * x + 12) / (3 / 4 * x + 18) = 1 / 3) : x = 120 :=
by 
  sorry

end saltwater_solution_l72_72232


namespace num_divisors_of_sum_of_consecutive_odd_primes_at_least_four_l72_72830

theorem num_divisors_of_sum_of_consecutive_odd_primes_at_least_four (p q : ℕ) (hp : Prime p) (hq : Prime q) (hc : is_consecutive_odd_primes p q) :
  ∃ d : ℕ, d ≥ 4 ∧ d = num_divisors (p + q) :=
sorry

-- Let's define the helper concepts that are required, but note we do not define their proofs.

-- is_consecutive_odd_primes checks if two primes are consecutive odd primes.
def is_consecutive_odd_primes (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ odd p ∧ odd q ∧ ∃ r, Prime r ∧ odd r ∧ p < r ∧ r < q

-- num_divisors calculates the number of positive divisors of a number
def num_divisors (n : ℕ) : ℕ :=
  if n > 0 then (nat.divisors n).length else 0

end num_divisors_of_sum_of_consecutive_odd_primes_at_least_four_l72_72830


namespace find_analytic_expression_l72_72003

-- Definitions based on the conditions
def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

def condition_f0 (f : ℝ → ℝ) : Prop := 
  f 0 = 1

def condition_diff (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (x + 1) - f x = 2 * x

-- The target function
def target_function : ℝ → ℝ := 
  λ x, x^2 - x + 1

-- The theorem to prove
theorem find_analytic_expression : 
  is_quadratic target_function ∧ condition_f0 target_function ∧ condition_diff target_function := 
by 
  -- Proof goes here
  sorry

end find_analytic_expression_l72_72003


namespace scalene_triangle_third_side_l72_72171

theorem scalene_triangle_third_side (a b c : ℕ) (h : (a - 3)^2 + (b - 2)^2 = 0) : 
  a = 3 ∧ b = 2 → c = 2 ∨ c = 3 ∨ c = 4 := 
by {
  sorry
}

end scalene_triangle_third_side_l72_72171


namespace num_nickels_l72_72540

-- Let n represent the number of nickels and q the number of quarters.
variable (n q : ℕ)

-- Conditions: There are 9 more nickels than quarters.
def condition1 := n = q + 9

-- The total amount of money in cents is 625.
def condition2 := 5 * n + 25 * q = 625

-- The statement we want to prove: n = 28.
theorem num_nickels : condition1 → condition2 → n = 28 := by
  sorry

end num_nickels_l72_72540


namespace solve_quadratic_for_negative_integer_l72_72187

theorem solve_quadratic_for_negative_integer (N : ℤ) (h_neg : N < 0) (h_eq : 2 * N^2 + N = 20) : N = -4 :=
sorry

end solve_quadratic_for_negative_integer_l72_72187


namespace probability_A_and_C_adjacent_given_A_and_B_adjacent_l72_72340

theorem probability_A_and_C_adjacent_given_A_and_B_adjacent :
  let students := ["A", "B", "C", "D"]
  let arrangements := permutations students
  let AB_adjacent := filter (λ l, (list.indexOf l "A" + 1 == list.indexOf l "B") ∨ (list.indexOf l "A" - 1 == list.indexOf l "B")) arrangements
  let AC_given_AB_adjacent := filter (λ l, (list.indexOf l "A" + 1 == list.indexOf l "C") ∨ (list.indexOf l "A" - 1 == list.indexOf l "C")) AB_adjacent
  (AC_given_AB_adjacent.card.toFloat / AB_adjacent.card.toFloat = 1 / 3) :=
by
  sorry

end probability_A_and_C_adjacent_given_A_and_B_adjacent_l72_72340


namespace inequality_solution_l72_72313

theorem inequality_solution (x : ℝ) (h_pos : 0 < x) :
  (3 / 8 + |x - 14 / 24| < 8 / 12) ↔ x ∈ Set.Ioo (7 / 24) (7 / 8) :=
by
  sorry

end inequality_solution_l72_72313


namespace monotonic_intervals_a_eq_1_no_zero_points_in_interval_zero_to_one_l72_72760

open Real

-- Define the function f(x)
def f (a x : ℝ) : ℝ := a * (x - 1) - 2 * log x

-- Problem I
theorem monotonic_intervals_a_eq_1 :
  (∀ x ∈ Ioi 2, deriv (f 1) x > 0) ∧
  (∀ x ∈ Ioo 0 2, deriv (f 1) x < 0) := 
sorry

-- Problem II
theorem no_zero_points_in_interval_zero_to_one (a : ℝ) :
  (∀ x ∈ Ioo 0 1, f a x ≠ 0) → a ≤ 2 :=
sorry

end monotonic_intervals_a_eq_1_no_zero_points_in_interval_zero_to_one_l72_72760


namespace range_a_inequality_l72_72317

theorem range_a_inequality (a : ℝ) :
  (∀ x : ℝ, (a-2) * x^2 + 2 * (a-2) * x - 4 < 0) ↔ a ∈ set.Icc (-2 : ℝ) 2 :=
sorry

end range_a_inequality_l72_72317


namespace tan_value_l72_72364

variable (θ : ℝ)

theorem tan_value (h : sin (12/5 * real.pi + θ) + 2 * sin (11/10 * real.pi - θ) = 0) :
  tan (2/5 * real.pi + θ) = 2 :=
sorry

end tan_value_l72_72364


namespace probability_heads_equals_7_over_11_l72_72488

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l72_72488


namespace regression_line_equation_chi_squared_test_l72_72588

open scoped Real

-- Define the linear regression line problem
theorem regression_line_equation :
    let x : List ℝ := [1, 2, 3, 4]
    let y : List ℝ := [1250, 1050, 1000, 900]
    let n : ℝ := 4
    let x_bar := (1 + 2 + 3 + 4) / n
    let y_bar := (1250 + 1050 + 1000 + 900) / n
    let b := (1250 + 2100 + 3000 + 3600 - n * x_bar * y_bar) / (1 + 4 + 9 + 16 - n * x_bar^2)
    let a := y_bar - b * x_bar
    let regression_y (x : ℝ) := b * x + a
    in b = -110 ∧ a = 1325 ∧ regression_y 5 = 775 := by
    sorry

-- Define the chi-squared test problem
theorem chi_squared_test :
    let a : ℕ := 7
    let b : ℕ := 13
    let c : ℕ := 3
    let d : ℕ := 27
    let n : ℕ := a + b + c + d
    let K2 := (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))
    let critical_value : ℝ := 3.841
    in K2 = 4.6875 ∧ K2 > critical_value := by
    sorry

end regression_line_equation_chi_squared_test_l72_72588


namespace matt_total_score_l72_72504

-- Definitions from the conditions
def num_2_point_shots : ℕ := 4
def num_3_point_shots : ℕ := 2
def score_per_2_point_shot : ℕ := 2
def score_per_3_point_shot : ℕ := 3

-- Proof statement
theorem matt_total_score : 
  (num_2_point_shots * score_per_2_point_shot) + 
  (num_3_point_shots * score_per_3_point_shot) = 14 := 
by 
  sorry  -- placeholder for the actual proof

end matt_total_score_l72_72504


namespace number_of_recipes_needed_l72_72290

def numStudents : ℕ := 150
def avgCookiesPerStudent : ℕ := 3
def cookiesPerRecipe : ℕ := 18
def attendanceDrop : ℝ := 0.40

theorem number_of_recipes_needed (n : ℕ) (c : ℕ) (r : ℕ) (d : ℝ) : 
  n = numStudents →
  c = avgCookiesPerStudent →
  r = cookiesPerRecipe →
  d = attendanceDrop →
  ∃ (recipes : ℕ), recipes = 15 :=
by
  intros
  sorry

end number_of_recipes_needed_l72_72290


namespace find_area_CDE_l72_72868

-- Define the areas of the given triangles
variable {R : Type} [LinearOrderedField R]

-- Points D on AC and E on BC of triangle ABC
variables {A B C D E F : R}

-- The areas of the given triangles
variables (area_ABF area_ADF area_BEF : R)

-- The specific values of the areas given as conditions
def area_ABF := 1
def area_ADF := 1 / 4
def area_BEF := 1 / 5

-- Define the overall area of each triangle
def area_ABC (area_CDE : R) : R :=
  area_ABF + area_ADF + area_BEF + area_CDE

-- The relationship based on given conditions
theorem find_area_CDE (area_CDE : R) :
  area_CDE = (3 : R) / 38 := sorry

end find_area_CDE_l72_72868


namespace true_statements_l72_72464

-- Definitions of planes and line
variables {α β γ : Type*}
variable {l : α}

-- Conditions for each statement:
-- Statement 1
def statement1 (α β γ : Type*) (h1 : α ∥ β) (h2 : α ⟂ γ) : Prop :=
  β ⟂ γ

-- Statement 2
def statement2 (α β γ : Type*) (h1 : α ⟂ γ) (h2 : β ⟂ γ) (h3 : α ∩ β = l) : Prop :=
  l ⟂ γ

-- Statement 3 condition
def statement3_false (α : Type*) (l : α) : Prop :=
  ∀ (countless_lines : set α), (∀ x ∈ countless_lines, l ⟂ x) → ¬ (l ⟂ α)

-- Statement 4 condition
def statement4_false (α β : Type*) (three_points : set (α × α × α)) : Prop :=
  (∀ p1 p2 p3 ∈ three_points, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 → ∀ q ∈ β, dist p1 q = dist p2 q = dist p3 q) → ¬ (α ∥ β)

-- The main theorem combining all statements
theorem true_statements (α β γ : Type*) (l : α) 
  (h1 : α ∥ β) (h2 : α ⟂ γ) (h3 : α ⟂ γ) (h4 : β ⟂ γ) (h5 : α ∩ β = l)
  (countless_lines : set α) 
  (three_points : set (α × α × α)) :
  statement1 α β γ h1 h2 ∧ statement2 α β γ h3 h4 h5 ∧ statement3_false α l ∧ statement4_false α β three_points :=
by 
  -- As required, no proof is provided here
  sorry

end true_statements_l72_72464


namespace are_names_possible_l72_72285

-- Define the structure to hold names
structure Person where
  first_name  : String
  middle_name : String
  last_name   : String

-- List of 4 people
def people : List Person :=
  [{ first_name := "Ivan", middle_name := "Ivanovich", last_name := "Ivanov" },
   { first_name := "Ivan", middle_name := "Petrovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Ivanovich", last_name := "Petrov" },
   { first_name := "Petr", middle_name := "Petrovich", last_name := "Ivanov" }]

-- Define the problem theorem
theorem are_names_possible :
  ∃ (people : List Person), 
    (∀ (p1 p2 p3 : Person), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → (p1.first_name ≠ p2.first_name ∨ p1.first_name ≠ p3.first_name ∨ p2.first_name ≠ p3.first_name) ∧
    (p1.middle_name ≠ p2.middle_name ∨ p1.middle_name ≠ p3.middle_name ∨ p2.middle_name ≠ p3.middle_name) ∧
    (p1.last_name ≠ p2.last_name ∨ p1.last_name ≠ p3.last_name ∨ p2.last_name ≠ p3.last_name)) ∧
    (∀ (p1 p2 : Person), p1 ≠ p2 → (p1.first_name = p2.first_name ∨ p1.middle_name = p2.middle_name ∨ p1.last_name = p2.last_name)) :=
by
  -- Place proof here
  sorry

end are_names_possible_l72_72285


namespace simplify_correct_l72_72667

theorem simplify_correct (x y : ℝ) :
  x - 2 * (y - 1) = x - 2 * y + 2 :=
by
  rw [sub_mul, mul_one, sub_sub, sub_add_eq_sub_sub]
  sorry

end simplify_correct_l72_72667


namespace highest_throw_among_them_l72_72683

variable (Christine_throw1 Christine_throw2 Christine_throw3 Janice_throw1 Janice_throw2 Janice_throw3 : ℕ)
 
-- Conditions given in the problem
def conditions :=
  Christine_throw1 = 20 ∧
  Janice_throw1 = Christine_throw1 - 4 ∧
  Christine_throw2 = Christine_throw1 + 10 ∧
  Janice_throw2 = Janice_throw1 * 2 ∧
  Christine_throw3 = Christine_throw2 + 4 ∧
  Janice_throw3 = Christine_throw1 + 17

-- The proof statement
theorem highest_throw_among_them : conditions -> max (max (max (max (max Christine_throw1 Christine_throw2) Christine_throw3) Janice_throw1) Janice_throw2) Janice_throw3 = 37 :=
sorry

end highest_throw_among_them_l72_72683


namespace determine_polynomial_l72_72846

-- Define a function f and its conditions
def polynomial_condition (f : ℤ → ℤ → ℤ) : Prop :=
  f 1 2 = 2 ∧ ∀ x y, y * f x (f x y) = (f x y) ^ 2 ∧
                    x * f (f x y) y = (f x y) ^ 2
  
theorem determine_polynomial (f : ℤ → ℤ → ℤ) :
  polynomial_condition f → f = (λ x y, x * y) :=
by
  sorry

end determine_polynomial_l72_72846


namespace intersecting_lines_l72_72172

theorem intersecting_lines (a b : ℝ) (h1 : 1 = 1 / 4 * 2 + a) (h2 : 2 = 1 / 4 * 1 + b) : 
  a + b = 9 / 4 := 
sorry

end intersecting_lines_l72_72172


namespace glass_cannot_all_be_upright_l72_72953

def glass_flip_problem :=
  ∀ (g : Fin 6 → ℤ),
    g 0 = 1 ∧ g 1 = 1 ∧ g 2 = 1 ∧ g 3 = 1 ∧ g 4 = 1 ∧ g 5 = -1 →
    (∀ (flip : Fin 4 → Fin 6 → ℤ),
      (∃ (i1 i2 i3 i4: Fin 6), 
        flip 0 = g i1 * -1 ∧ 
        flip 1 = g i2 * -1 ∧
        flip 2 = g i3 * -1 ∧
        flip 3 = g i4 * -1) →
      ∃ j, g j ≠ 1)

theorem glass_cannot_all_be_upright : glass_flip_problem :=
  sorry

end glass_cannot_all_be_upright_l72_72953


namespace paul_sold_11_books_l72_72511

variable (initial_books : ℕ) (books_given : ℕ) (books_left : ℕ) (books_sold : ℕ)

def number_of_books_sold (initial_books books_given books_left books_sold : ℕ) : Prop :=
  initial_books - books_given - books_left = books_sold

theorem paul_sold_11_books : number_of_books_sold 108 35 62 11 :=
by
  sorry

end paul_sold_11_books_l72_72511


namespace fruit_pie_apple_peach_pie_fruits_l72_72446

theorem fruit_pie_apple_peach_pie_fruits (a_f p_f a_ap e_ap n_f n_ap : ℕ) (hf : a_f = 4) (hf_p : p_f = 3) (hap : a_ap = 6) (hap_e : e_ap = 2) (hnf : n_f = 357) (hnap : n_ap = 712) :
  let A := a_f * n_f + a_ap * n_ap in
  let P := p_f * n_f in
  let E := e_ap * n_ap in
  A = 5700 ∧ P = 1071 ∧ E = 1424 := 
by
  sorry

end fruit_pie_apple_peach_pie_fruits_l72_72446


namespace monotonic_decreasing_interval_l72_72930

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∃ t ∈ Ioo (-1 : ℝ) 11, t = x} ⊆ {x : ℝ | ∃ t, f'(t) < 0} :=
sorry

end monotonic_decreasing_interval_l72_72930


namespace find_b_given_a_l72_72521

-- Definitions based on the conditions
def varies_inversely (a b : ℝ) (k : ℝ) : Prop := a * b = k
def k_value : ℝ := 400

-- The proof statement
theorem find_b_given_a (a b : ℝ) (h1 : varies_inversely 800 0.5 k_value) (h2 : a = 3200) : b = 0.125 :=
by
  -- skipped proof
  sorry

end find_b_given_a_l72_72521


namespace hyperbola_standard_eqn_l72_72332

theorem hyperbola_standard_eqn (x y : ℝ) (a b : ℝ) : 
  (∀ (x y : ℝ), (x^2 / 3 - y^2 = 1 → a^2 / 3 - b^2 = λ) 
  (λ : ℝ) = ( (sqrt 3)^2 / 3 - 2^2 ) → λ = -3 → ((abs λ = 1 / 3)  ) :=
by sorry

end hyperbola_standard_eqn_l72_72332


namespace circumscribed_cyclic_quad_perpendicular_chords_perpendicular_chords_cyclic_quad_l72_72515

-- Statement 1
theorem circumscribed_cyclic_quad_perpendicular_chords
  (A B C D M N P Q : Point)
  [Circle touch_points A B C D M N P Q] :
  circumscribed (A B C D) ∧ cyclic (A B C D) →
  perpendicular (chord (connect M N)) (chord (connect P Q)) :=
sorry

-- Statement 2
theorem perpendicular_chords_cyclic_quad
  (A B C D M N P Q : Point)
  [Circle touch_points A B C D M N P Q] :
  perpendicular (chord (connect M N)) (chord (connect P Q)) →
  cyclic (A B C D) :=
sorry

end circumscribed_cyclic_quad_perpendicular_chords_perpendicular_chords_cyclic_quad_l72_72515


namespace largest_integer_solution_of_inequality_l72_72170

theorem largest_integer_solution_of_inequality :
  ∃ x : ℤ, x < 2 ∧ (∀ y : ℤ, y < 2 → y ≤ x) ∧ -x + 3 > 1 :=
sorry

end largest_integer_solution_of_inequality_l72_72170


namespace cylinder_surface_area_l72_72273

/-- A right cylinder with radius 3 inches and height twice the radius has a total surface area of 54π square inches. -/
theorem cylinder_surface_area (r : ℝ) (h : ℝ) (A_total : ℝ) (π : ℝ) : r = 3 → h = 2 * r → π = Real.pi → A_total = 54 * π :=
by
  sorry

end cylinder_surface_area_l72_72273


namespace circumscribed_area_ge_theorem_l72_72471

noncomputable def circumscribed_area_ge (n : ℕ) (r : ℝ) (M : EuclideanGeometry.Polygon ℝ) (M_n : EuclideanGeometry.RegularPolygon n ℝ) : Prop :=
  M.circumscribed ℝ (EuclideanGeometry.Circle.make O r) →
  M_n.circumscribed ℝ (EuclideanGeometry.Circle.make O r) →
  M.area ≥ M_n.area

theorem circumscribed_area_ge_theorem {n : ℕ} {r : ℝ} {M : EuclideanGeometry.Polygon ℝ} {M_n : EuclideanGeometry.RegularPolygon n ℝ}
  (circ_M : M.circumscribed ℝ (EuclideanGeometry.Circle.make O r))
  (circ_M_n : M_n.circumscribed ℝ (EuclideanGeometry.Circle.make O r)) :
  circumscribed_area_ge n r M M_n :=
begin
  sorry
end

end circumscribed_area_ge_theorem_l72_72471


namespace perpendicular_triangle_equation_l72_72878

variables {A B C A1 B1 C1 : Type}{A1, B1, C1 : A \to B \to C}

-- Definitions of the necessary concepts
noncomputable def triangle_area (ABC : Triangle) : ℝ := sorry
noncomputable def perpendicular_from_vertex (A A1 AB B B1 BC C C1 CA : Type) : Prop := sorry

-- Definition of the main goal
theorem perpendicular_triangle_equation
  (ABC : Triangle)
  (h_perpend_A : perpendicular_from_vertex A A1 ABC)
  (h_perpend_B : perpendicular_from_vertex B B1 ABC)
  (h_perpend_C : perpendicular_from_vertex C C1 ABC) :
  (C1 * ABC - A * BC - B * CA = 2 * triangle_area ABC) :=
sorry

end perpendicular_triangle_equation_l72_72878


namespace sum_of_sequence_l72_72238

noncomputable def x : ℕ → ℚ
| 1       := 1 / 2
| (n + 1) := x n / (2 * (n + 1) * x n + 1)
  -- We use (n+1) here because Lean’s natural numbers start from 0 internally.

theorem sum_of_sequence :
  ∑ k in Finset.range 2018, x (k + 1) = 2018 / 2019 :=
sorry

end sum_of_sequence_l72_72238


namespace order_of_m_n_p_q_l72_72412

variable {m n p q : ℝ} -- Define the variables as real numbers

theorem order_of_m_n_p_q (h1 : m < n) 
                         (h2 : p < q) 
                         (h3 : (p - m) * (p - n) < 0) 
                         (h4 : (q - m) * (q - n) < 0) : 
    m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_m_n_p_q_l72_72412


namespace ratio_of_numbers_l72_72793

theorem ratio_of_numbers (a b : ℕ) (ha : a = 45) (hb : b = 60) (lcm_ab : Nat.lcm a b = 180) : (a : ℚ) / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l72_72793


namespace problem_solution_l72_72762

theorem problem_solution (a c : ℝ) (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 3 → ax^2 + x + c > 0) →
  (a = -1/4 ∧ c = -3/4) →
  (∀ x : ℝ, 2 < x ∧ x < 6 → a*x^2 + 2*x + 4*c > 0) →
  (3*a*x + c*m < 0) →
  (∀ x ∈ set.Ioo 2 6, x > -m) →
  m ≥ -2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end problem_solution_l72_72762


namespace difference_of_radii_l72_72179

noncomputable def diffRadii (r : ℝ) (R : ℝ) : ℝ := R - r

theorem difference_of_radii (r : ℝ) (R : ℝ) (h1 : π * R^2 / (π * r^2) = 4) : diffRadii r R = r :=
by
  have h2 : R^2 / r^2 = 4 := by
    have : π ≠ 0 := by linarith
    field_simp at h1
    assumption
  have h3 : R / r = 2 := by
    exact (eq_div_iff (show r^2 ≠ 0 from pow_ne_zero 2 (ne_of_gt (show r > 0 from by linarith)))).1 (by linarith)
  have h4 : R = 2 * r := by
    field_simp at h3
    assumption
  have h5 : diffRadii r R = (2 * r) - r := by
    rw h4
  linarith

end difference_of_radii_l72_72179


namespace length_BD_leq_12_l72_72993

variable (A B C D : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

noncomputable
def length_AB : ℝ := 15
noncomputable
def length_BC : ℝ := 10
noncomputable def length_BD : ℝ := sorry -- unknown length to define

theorem length_BD_leq_12 (triangle_ABC : MetricSpace.triangle A B C)
  (BD_bisector : MetricSpace.bisector B A C D)
  (H1 : length AB = 15)
  (H2 : length BC = 10) :
  length BD ≤ 12 :=
sorry

end length_BD_leq_12_l72_72993


namespace compute_alpha_powers_l72_72838

variable (α1 α2 α3 : ℂ)

open Complex

-- Given conditions
def condition1 : Prop := α1 + α2 + α3 = 2
def condition2 : Prop := α1^2 + α2^2 + α3^2 = 6
def condition3 : Prop := α1^3 + α2^3 + α3^3 = 14

-- The required proof statement
theorem compute_alpha_powers (h1 : condition1 α1 α2 α3) (h2 : condition2 α1 α2 α3) (h3 : condition3 α1 α2 α3) :
  α1^7 + α2^7 + α3^7 = 46 := by
  sorry

end compute_alpha_powers_l72_72838


namespace calculate_profit_percentage_l72_72661

variable (P : ℝ)

def calculate_overall_loss (P : ℝ) := (P / 100) * 2500 - 500

theorem calculate_profit_percentage 
  (total_worth : ℝ)
  (percent_profit_stock : ℝ)
  (remaining_percent_loss : ℝ)
  (overall_loss : ℝ)
  (worth_20_percent : ℝ)
  (worth_80_percent : ℝ)
  (loss_80_percent : ℝ):
  calculate_overall_loss P = overall_loss → P = 10 :=
by
  intros
  sorry

-- Given conditions
#check calculate_profit_percentage 12_499.999999999998 0.2 0.8 (-250) 2500 10000 500

end calculate_profit_percentage_l72_72661


namespace find_point_on_x_axis_l72_72879

theorem find_point_on_x_axis (a : ℝ) (h : abs (3 * a + 6) = 30) : (a = -12) ∨ (a = 8) :=
sorry

end find_point_on_x_axis_l72_72879


namespace sum_greater_than_one_point_one_l72_72986

theorem sum_greater_than_one_point_one :
  let numbers := [1.4, 9/10, 1.2, 0.5, 13/10]
  (numbers.filter (λ x => x >= 1.1)).sum = 3.9 :=
by
  let numbers := [1.4, 9/10, 1.2, 0.5, 13/10]
  have h : numbers.filter (λ x => x >= 1.1) = [1.4, 1.2, 13/10] := by decide
  have h1 : ([1.4, 1.2, 13/10].sum = 3.9) := by decide
  rw [h]
  exact h1

end sum_greater_than_one_point_one_l72_72986


namespace fill_half_jar_in_18_days_l72_72159

-- Define the doubling condition and the days required to fill half the jar
variable (area : ℕ → ℕ)
variable (doubling : ∀ t, area (t + 1) = 2 * area t)
variable (full_jar : area 19 = 2^19)
variable (half_jar : area 18 = 2^18)

theorem fill_half_jar_in_18_days :
  ∃ n, n = 18 ∧ area n = 2^18 :=
by {
  -- The proof is omitted, but we state the goal
  sorry
}

end fill_half_jar_in_18_days_l72_72159


namespace ABC_is_acute_l72_72356

theorem ABC_is_acute (S A B C : Point)
  (h₁ : ∀ (S A B C : Point), is_triangular_pyramid S A B C)
  (h₂ : ∀ (S A : Line), (SB : Line), (SC : Line), mutually_perpendicular SA SB SC) : 
  is_acute_angled_triangle A B C :=
sorry

end ABC_is_acute_l72_72356


namespace sum_of_squares_mod_13_l72_72605

theorem sum_of_squares_mod_13 :
  ((∑ i in Finset.range 16, i^2) % 13) = 3 :=
by 
  sorry

end sum_of_squares_mod_13_l72_72605


namespace speed_of_first_train_l72_72641

noncomputable def length_of_first_train : ℝ := 280
noncomputable def speed_of_second_train_kmph : ℝ := 80
noncomputable def length_of_second_train : ℝ := 220.04
noncomputable def time_to_cross : ℝ := 9

noncomputable def relative_speed_mps := (length_of_first_train + length_of_second_train) / time_to_cross

noncomputable def relative_speed_kmph := relative_speed_mps * (3600 / 1000)

theorem speed_of_first_train :
  (relative_speed_kmph - speed_of_second_train_kmph) = 120.016 :=
by
  sorry

end speed_of_first_train_l72_72641


namespace mb_plus_mc_eq_ma_l72_72114

variables {A B C M : Point}

-- condition 1: A, B, and C form an equilateral triangle
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- condition 2: M is on the circumcircle of triangle ABC, on the arc between B and C not containing A
def on_circumcircle_not_containing (M A B C : Point) (O : Point) : Prop :=
  on_circumcircle M A B C ∧ on_arc_not_containing M B C A

theorem mb_plus_mc_eq_ma (O : Point)
  (h1 : equilateral_triangle A B C)
  (h2 : on_circumcircle_not_containing M A B C O) :
  dist M B + dist M C = dist M A :=
sorry

end mb_plus_mc_eq_ma_l72_72114


namespace center_of_the_hyperbola_l72_72314

def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

structure Point where
  x : ℝ
  y : ℝ

def center_of_hyperbola_is (p : Point) : Prop :=
  hyperbola_eq (p.x + 3) (p.y + 4)

theorem center_of_the_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → center_of_hyperbola_is {x := 3, y := 4} :=
by
  intros x y h
  sorry

end center_of_the_hyperbola_l72_72314


namespace domain_of_rational_function_l72_72708

def rational_function (x : ℝ) : ℝ := (x^3 - 3 * x^2 + 2 * x + 5) / (x^3 - 6 * x^2 + 11 * x - 6)

theorem domain_of_rational_function :
  ∀ x : ℝ, x ∈ ((set.Ioo (-1 : ℝ) 1) ∪ (set.Ioo 1 2) ∪ (set.Ioo 2 3) ∪ (set.Ioc 3) ∪ (set.Ioi 3))  ↔
  rational_function x ≠ 0 := 
sorry

end domain_of_rational_function_l72_72708


namespace percentage_difference_l72_72406

theorem percentage_difference : 0.70 * 100 - 0.60 * 80 = 22 := 
by
  sorry

end percentage_difference_l72_72406


namespace conjugate_of_z_l72_72377

noncomputable def z : ℂ := 1 - 2 * complex.I

theorem conjugate_of_z : complex.conj z = 1 + 2 * complex.I := by
  have h : z + complex.I = (1 + complex.I) / complex.I := by sorry
  have hz : z = 1 - 2 * complex.I := by sorry
  show complex.conj z = 1 + 2 * complex.I from sorry

end conjugate_of_z_l72_72377


namespace smallest_even_number_l72_72186

theorem smallest_even_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6)) = 140) : x = 32 :=
by
  sorry

end smallest_even_number_l72_72186


namespace malvina_correct_l72_72451
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end malvina_correct_l72_72451


namespace max_value_sum_l72_72949

-- Given four distinct integers f, g, h, j, each being one of 4, 5, 6, and 7,
-- prove that the largest value of the sum fg + gh + hj + fj is 120.

theorem max_value_sum (f g h j : ℕ)
  (hf : f ∈ {4, 5, 6, 7})
  (hg : g ∈ {4, 5, 6, 7})
  (hh : h ∈ {4, 5, 6, 7})
  (hj : j ∈ {4, 5, 6, 7})
  (distinct : f ≠ g ∧ g ≠ h ∧ h ≠ j ∧ j ≠ f ∧ f ≠ h ∧ g ≠ j) :
  fg + gh + hj + fj ≤ 120 :=
sorry

end max_value_sum_l72_72949


namespace max_value_of_y_period_of_y_l72_72331

def y (x : ℝ) := 3 * Real.sin (2 * x) + 4 * Real.cos (2 * x)

theorem max_value_of_y : 
  ∃ M, (∀ x, y x ≤ M) ∧ (∃ x, y x = M) ∧ M = 5 := by
  sorry

theorem period_of_y :
  ∃ T > 0, (∀ x, y (x + T) = y x) ∧ T = Real.pi := by
  sorry

end max_value_of_y_period_of_y_l72_72331


namespace tan_alpha_tan_m_l72_72244

section Problem1
variable {α : ℝ}
hypothesis h1 : Real.sin α = 1 / 3
hypothesis h2 : (π / 2 < α) ∧ (α < π) -- α in the second quadrant

theorem tan_alpha : Real.tan α = -Real.sqrt 2 / 4 :=
sorry
end Problem1

section Problem2
variable {m : ℝ}
hypothesis h3 : m ≠ 0
hypothesis h4 : m ≠ 1
hypothesis h5 : m ≠ -1
hypothesis h6 : Real.sin α = m

theorem tan_m : Real.tan α = m / Real.sqrt(1 - m ^ 2) ∨ Real.tan α = -m / Real.sqrt(1 - m ^ 2) :=
sorry
end Problem2

end tan_alpha_tan_m_l72_72244


namespace reflection_of_P_in_AB_l72_72254

noncomputable theory
open_locale classical

-- Define the circle and the points
variables {X : Type*} [metric_space X] [normed_group X] [normed_space ℝ X]
variables (O A B P Q R : X)
variables (circle : metric.sphere O 1)
variables (line_AB : affine_subspace ℝ X := affine_span ℝ {A, B})

-- Conditions as definitions
def on_circle (x : X) : Prop := x ∈ circle
def minor_arc (x : X) : Prop := on_circle x ∧ dist x A < dist x B
def equidistant {x y z : X} : Prop := dist x y = dist z y

-- Problem Statement
theorem reflection_of_P_in_AB 
(hA : on_circle A)
(hB : on_circle B)
(hP : minor_arc P)
(hQ : on_circle Q)
(hR : on_circle R)
(hPQ : equidistant P Q A)
(hPR : equidistant P R B)
:
  let X := classical.some (exists_intersection (line_through A R) (line_through B Q))
  in reflection_in_line AB P = X :=
begin
  sorry,
end

end reflection_of_P_in_AB_l72_72254


namespace vector_problem_solution_l72_72396

-- Define the conditions given in the problem
variables {a b : ℝ} -- Represents the two unit vectors
variables {m : ℝ} -- Represents the scalar multiple

axiom unit_vectors : ∥a∥ = 1 ∧ ∥b∥ = 1
axiom angle_between : real.angle_cos 30 = (a • b) / (∥a∥ * ∥b∥)
axiom dot_product_zero : (b • (m * a + (1 - m) * b)) = 0

-- Prove that m = 4 + 2 * real.sqrt 3
theorem vector_problem_solution : 
  ∃ m : ℝ, (m * (∥a∥ * ∥b∥ * real.cos 30) + (1 - m) * ∥b∥^2 = 0) → 
  m = 4 + 2 * real.sqrt 3 :=
by
  -- This is where the proof would go
  sorry

end vector_problem_solution_l72_72396


namespace angle_OAB_of_inscribed_decagon_l72_72231

theorem angle_OAB_of_inscribed_decagon (O A B : ℝ -> ℝ) (r : ℝ) (h1 : distance O A = r) (h2 : distance O B = r) (h3 : A ≠ B) :
  let θ := 360 / 10 in
  let central_angle := θ in
  let angle_OAB := central_angle / 2 in
  angle_OAB = 18 :=
sorry

end angle_OAB_of_inscribed_decagon_l72_72231


namespace statement_A_statement_B_statement_C_statement_D_correct_statements_l72_72625

theorem statement_A : 
  let line := λ x y, sqrt 3 * x + y + 1 = 0 in 
  slope line = - sqrt 3 → angle_inclination slope = 120 :=
sorry

theorem statement_B :
  let line := λ x y, x - y - 1 = 0 in 
  ¬(passes_through line (2, 1)) :=
sorry 

theorem statement_C :
  let line1 := λ x y, x + 2 * y - 4 = 0 in 
  let line2 := λ x y, 2 * x + 4 * y + 1 = 0 in
  distance line1 line2 = 9 * sqrt 5 / 10 :=
sorry

theorem statement_D :
  let l1 := λ x y, a * x + 2 * a * y + 1 = 0 in
  let l2 := λ x y, (a - 1) * x - (a + 1) * y - 4 = 0 in
  perpendicular l1 l2 → a = -3 :=
sorry

theorem correct_statements : 
  (statement_A ∧ statement_C ∧ statement_D) ∧ ¬ statement_B :=
sorry

end statement_A_statement_B_statement_C_statement_D_correct_statements_l72_72625


namespace find_S6_and_a1_range_of_d_l72_72468

-- Define the arithmetic sequence and sum of the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_S6_and_a1 (a₁ d : ℝ) (S₅ S₆ : ℝ) 
  (h1 : S₅ * S₆ + 15 = 0) (h2 : S₅ ≠ 5) :
  S₆ = -3 ∧ a₁ = 7 :=
by
  sorry

theorem range_of_d (a₁ d : ℝ) (S₅ S₆ : ℝ) 
  (h1 : S₅ * S₆ + 15 = 0) :
  d ≤ -2 * Real.sqrt 2 ∨ d ≥ 2 * Real.sqrt 2 :=
by
  sorry

end find_S6_and_a1_range_of_d_l72_72468


namespace percent_students_with_B_l72_72071

theorem percent_students_with_B :
  let scores := [91, 82, 56, 99, 86, 95, 88, 79, 77, 68, 83, 81, 65, 84, 93, 72, 89, 78]
  let total_students := 18
  let B_grades := (85, 93)
  let count_B := scores.count (λ score => score ≥ B_grades.1 ∧ score ≤ B_grades.2)
  (count_B * 100) / total_students = 27.78 := by
  sorry

end percent_students_with_B_l72_72071


namespace probability_heads_equals_7_over_11_l72_72486

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l72_72486


namespace mirasol_balance_l72_72133

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l72_72133


namespace sphere_in_cube_l72_72275

noncomputable def radius (edge: ℕ) : ℝ := (edge : ℝ) / 2
noncomputable def volume (r: ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def surface_area (r: ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_in_cube (edge: ℕ) (h: edge = 10) : 
  volume (radius edge) = (500 / 3) * Real.pi ∧ surface_area (radius edge) = 100 * Real.pi := 
by
  obtain h_radius : radius edge = 5 := by
    rw [radius, h]
    norm_num 
  split
  · rw [volume, h_radius]
    norm_num
  · rw [surface_area, h_radius]
    norm_num
  sorry

end sphere_in_cube_l72_72275


namespace sequence_eventually_periodic_iff_odd_l72_72835

theorem sequence_eventually_periodic_iff_odd (u : ℚ) (m : ℕ) (h_u_pos : 0 < u) :
  (∃ c t : ℕ, ∀ n : ℕ, n ≥ c → sequence (u, m) n = sequence (u, m) (n+t)) ↔ odd m := 
begin
  sorry
end

end sequence_eventually_periodic_iff_odd_l72_72835


namespace not_divisible_1978_1000_l72_72143

theorem not_divisible_1978_1000 (m : ℕ) : ¬ ∃ m : ℕ, (1000^m - 1) ∣ (1978^m - 1) := sorry

end not_divisible_1978_1000_l72_72143


namespace largest_tan_B_l72_72444

-- The context of the problem involves a triangle with given side lengths
variables (ABC : Triangle) -- A triangle ABC

-- Define the lengths of sides AB and BC
variables (AB BC : ℝ) 
-- Define the value of tan B
variable (tanB : ℝ)

-- The given conditions
def condition_1 := AB = 25
def condition_2 := BC = 20

-- Define the actual statement we need to prove
theorem largest_tan_B (ABC : Triangle) (AB BC tanB : ℝ) : 
  AB = 25 → BC = 20 → tanB = 3 / 4 := sorry

end largest_tan_B_l72_72444


namespace tony_belinda_combined_age_l72_72194

/-- Tony and Belinda have a combined age. Belinda is 8 more than twice Tony's age. 
Tony is 16 years old and Belinda is 40 years old. What is their combined age? -/
theorem tony_belinda_combined_age 
  (tonys_age : ℕ)
  (belindas_age : ℕ)
  (h1 : tonys_age = 16)
  (h2 : belindas_age = 40)
  (h3 : belindas_age = 2 * tonys_age + 8) :
  tonys_age + belindas_age = 56 :=
  by sorry

end tony_belinda_combined_age_l72_72194


namespace highest_throw_is_37_feet_l72_72686

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l72_72686


namespace smallest_constant_for_triangle_l72_72717

theorem smallest_constant_for_triangle 
  (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)  
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 := 
  sorry

end smallest_constant_for_triangle_l72_72717


namespace problem_condition_l72_72049

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l72_72049


namespace inequality_solution_range_l72_72028

theorem inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ (k ∈ set.Iic (Real.sqrt (6)/6)) := 
sorry

end inequality_solution_range_l72_72028


namespace area_inside_rectangle_outside_circles_approx_l72_72886

def rectangle_area (CD DA : ℝ) : ℝ :=
  CD * DA

def circle_area (radius : ℝ) : ℝ :=
  π * radius * radius

def quarter_circle_area (radius : ℝ) : ℝ :=
  (π * radius * radius) / 4

constant pi_approx : ℝ := 3.14

theorem area_inside_rectangle_outside_circles_approx :
  let rect_area := rectangle_area 4 6 in
  let circle_A_area := quarter_circle_area 2 in
  let circle_B_area := quarter_circle_area 3 in
  let circle_C_area := quarter_circle_area 4 in
  let total_quarter_circles_area := circle_A_area + circle_B_area + circle_C_area in
  let exact_area := rect_area - total_quarter_circles_area in
  let approx_area := rect_area - (total_quarter_circles_area * pi_approx / π) in
  approx_area ≈ 1.235 :=
by
  sorry

end area_inside_rectangle_outside_circles_approx_l72_72886


namespace decreasing_interval_range_of_a_l72_72381

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (2 * x^2 - 3 * x) * Real.exp x

-- Monotonically decreasing interval
theorem decreasing_interval : ∀ x, x ∈ Ioo (-1.5) 1 → (4 * x + 2 * x^2 - 3) * Real.exp x < 0 := sorry

-- Range of 'a' for exactly one real root
theorem range_of_a (a : ℝ) : 
  ((∀ x ≠ 0, (2 * x - 3) * Real.exp x ≠ a / x) ∧ (∃! x, (2 * x - 3) * Real.exp x = a / x)) ↔ 
  a ∈ {-Real.exp 1, 0} ∪ Ioc (9 * Real.exp (-1.5)) ∞ := sorry

end decreasing_interval_range_of_a_l72_72381


namespace no_consecutive_primes_sum_65_l72_72297

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p q : ℕ) : Prop := 
  is_prime p ∧ is_prime q ∧ (q = p + 2 ∨ q = p - 2)

theorem no_consecutive_primes_sum_65 : 
  ¬ ∃ p q : ℕ, consecutive_primes p q ∧ p + q = 65 :=
by 
  sorry

end no_consecutive_primes_sum_65_l72_72297


namespace find_p_value_l72_72370

theorem find_p_value (p : ℝ) 
  (h_root1 : is_root (λ x : ℂ, 3 * x^2 + p * x - 8) (2 + complex.i)) 
  (h_real : ∀ x : ℂ, x.im = 0 → is_root (λ x, 3 * x^2 + p * x - 8) x → is_root (λ x, 3 * x^2 + p * x - 8) x.conj)
  : p = -12 := 
by
  sorry

end find_p_value_l72_72370


namespace part_I_part_II_l72_72390

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2 * x|

-- Part I
theorem part_I : ∀ x : ℝ, g x > -4 → -5 < x ∧ x < -3 :=
by
  sorry

-- Part II
theorem part_II : 
  (∃ x1 x2 : ℝ, f x1 a = g x2) → -6 ≤ a ∧ a ≤ -4 :=
by
  sorry

end part_I_part_II_l72_72390


namespace magpies_triangle_l72_72676

/-- In a regular n-gon with n vertices, magpies start at each vertex, fly away, and possibly return to different vertices. 
Prove that for n ≥ 3, n ≠ 5, there will always exist a triangle formed by three magpies which is acute, right, or obtuse 
after they return to any vertices. -/
theorem magpies_triangle (n : ℕ) (h : n ≥ 3) (hne : n ≠ 5) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
   (∀ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ C ≠ A →
    ((AB² + BC² > CA²) ∨ (AB² + BC² = CA²) ∨ (AB² + BC² < CA²))) :=
by
  sorry

end magpies_triangle_l72_72676


namespace arithmetic_mean_16_24_40_32_l72_72594

theorem arithmetic_mean_16_24_40_32 : (16 + 24 + 40 + 32) / 4 = 28 :=
by
  sorry

end arithmetic_mean_16_24_40_32_l72_72594


namespace mirasol_account_balance_l72_72127

theorem mirasol_account_balance :
  ∀ (initial_amount spent_coffee spent_tumbler : ℕ), 
  initial_amount = 50 → 
  spent_coffee = 10 → 
  spent_tumbler = 30 → 
  initial_amount - (spent_coffee + spent_tumbler) = 10 :=
by
  intros initial_amount spent_coffee spent_tumbler
  intro h_initial_amount
  intro h_spent_coffee
  intro h_spent_tumbler
  rw [h_initial_amount, h_spent_coffee, h_spent_tumbler]
  simp
  done

end mirasol_account_balance_l72_72127


namespace number_of_ways_to_tile_dominos_l72_72669

-- Define the dimensions of the shapes and the criteria for the tiling problem
def L_shaped_area := 24
def size_of_square := 4
def size_of_rectangles := 2 * 10
def number_of_ways_to_tile := 208

-- Theorem statement
theorem number_of_ways_to_tile_dominos :
  (L_shaped_area = size_of_square + size_of_rectangles) →
  number_of_ways_to_tile = 208 :=
by
  intros h
  sorry

end number_of_ways_to_tile_dominos_l72_72669


namespace fraction_division_l72_72987

theorem fraction_division : (12345 : ℕ) / (1 + 2 + 3 + 4 + 5) = 823 := by
  have h1 := Nat.add_assoc 1 2 (3 + 4 + 5)
  have h2 := Nat.add_assoc (1 + 2) 3 (4 + 5)
  have h3 := Nat.add_assoc (1 + 2 + 3) 4 5
  have h4 : (1 + 2 + 3 + 4 + 5) = 15 := by
    rw [h3, Nat.add_comm, ←Nat.add_assoc, Nat.add_comm 4 5, Nat.add_add, Nat.add_comm 3]
    norm_num
  rw h4
  norm_num
  sorry

end fraction_division_l72_72987


namespace pagoda_top_story_lanterns_l72_72437

/--
Given a 7-story pagoda where each story has twice as many lanterns as the one above it, 
and a total of 381 lanterns across all stories, prove the number of lanterns on the top (7th) story is 3.
-/
theorem pagoda_top_story_lanterns (a : ℕ) (n : ℕ) (r : ℚ) (sum_lanterns : ℕ) :
  n = 7 → r = 1 / 2 → sum_lanterns = 381 →
  (a * (1 - r^n) / (1 - r) = sum_lanterns) → (a * r^(n - 1) = 3) :=
by
  intros h_n h_r h_sum h_geo_sum
  let a_val := 192 -- from the solution steps
  rw [h_n, h_r, h_sum] at h_geo_sum
  have h_a : a = a_val := by sorry
  rw [h_a, h_n, h_r]
  exact sorry

end pagoda_top_story_lanterns_l72_72437


namespace peter_total_dogs_l72_72150

def num_german_shepherds_sam : ℕ := 3
def num_french_bulldogs_sam : ℕ := 4
def num_german_shepherds_peter := 3 * num_german_shepherds_sam
def num_french_bulldogs_peter := 2 * num_french_bulldogs_sam

theorem peter_total_dogs : num_german_shepherds_peter + num_french_bulldogs_peter = 17 :=
by {
  -- adding proofs later
  sorry
}

end peter_total_dogs_l72_72150


namespace range_of_m_l72_72386

def f (x : ℝ) : ℝ := 2 * x + 1 / x ^ 2 - 4
def g (x m : ℝ) : ℝ := (Real.log x) / x - m

theorem range_of_m (m : ℝ) : 
  (∀ x1 ∈ Set.Icc (1 : ℝ) (2 : ℝ), ∃ x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2), f x1 ≤ g x2 m) ↔
  m ≤ 1 / Real.exp 1 - 1 / 4 :=
by
  sorry

end range_of_m_l72_72386


namespace trig_identity_l72_72168

theorem trig_identity (x : ℝ) : 
  let a := 4
      b := 8
      c := 4
      d := 2
  in 
  (cos (2 * x) + cos (6 * x) + cos (10 * x) + cos (14 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) ∧
  (a + b + c + d = 18) :=
by
  let a := 4
  let b := 8
  let c := 4
  let d := 2
  sorry

end trig_identity_l72_72168


namespace minimize_variance_l72_72354

def values := [4, 6, 8, 9, x, y, 11, 12, 14, 16]

def median := 10

theorem minimize_variance :
  (values.nth_le 5 (by decide) = median) ∧ (x ≤ 10) ∧ y = 10 → x * y = 100 := 
by
  sorry

end minimize_variance_l72_72354


namespace owen_profit_l72_72877

theorem owen_profit
  (num_boxes : ℕ)
  (cost_per_box : ℕ)
  (pieces_per_box : ℕ)
  (sold_boxes : ℕ)
  (price_per_25_pieces : ℕ)
  (remaining_pieces : ℕ)
  (price_per_10_pieces : ℕ) :
  num_boxes = 12 →
  cost_per_box = 9 →
  pieces_per_box = 50 →
  sold_boxes = 6 →
  price_per_25_pieces = 5 →
  remaining_pieces = 300 →
  price_per_10_pieces = 3 →
  sold_boxes * 2 * price_per_25_pieces + (remaining_pieces / 10) * price_per_10_pieces - num_boxes * cost_per_box = 42 :=
by
  intros h_num h_cost h_pieces h_sold h_price_25 h_remain h_price_10
  sorry

end owen_profit_l72_72877


namespace factory_produces_6500_toys_per_week_l72_72648

theorem factory_produces_6500_toys_per_week
    (days_per_week : ℕ)
    (toys_per_day : ℕ)
    (h1 : days_per_week = 5)
    (h2 : toys_per_day = 1300) :
    days_per_week * toys_per_day = 6500 := 
by 
  sorry

end factory_produces_6500_toys_per_week_l72_72648


namespace series_sum_equals_four_l72_72687

/-- 
  Proof of the sum of the series: 
  ∑ (n=1 to ∞) (6n² - n + 1) / (n⁵ - n⁴ + n³ - n² + n) = 4 
--/
theorem series_sum_equals_four :
  (∑' n : ℕ, (if n > 0 then (6 * n^2 - n + 1 : ℝ) / (n^5 - n^4 + n^3 - n^2 + n) else 0)) = 4 :=
by
  sorry

end series_sum_equals_four_l72_72687


namespace exists_graph_girth_chromatic_l72_72951

-- Theorem 11.2.2 (Erdős, 1959)
theorem exists_graph_girth_chromatic (k : ℕ) (hk : k ≥ 3) :
  ∃ (H : SimpleGraph ℕ), H.girth > k ∧ H.chromaticNumber > k := 
sorry

end exists_graph_girth_chromatic_l72_72951


namespace find_monic_quadratic_polynomial_l72_72713

-- Define the conditions
def is_monic {R : Type*} [CommRing R] (p : Polynomial R) : Prop :=
  p.leadingCoeff = 1

def has_real_coefficients {R : Type*} [CommRing R] (p : Polynomial R) : Prop :=
  ∀ c ∈ p.coeff.support, is_real (p.coeff c)

def has_root {R : Type*} [CommRing R] (p : Polynomial R) (z : R) : Prop :=
  p.eval z = 0

-- Define the complex polynomial we want to prove is the solution
def monic_quadratic_polynomial : Polynomial ℂ :=
  Polynomial.X^2 - 4 * Polynomial.X + 5

-- Theorem statement
theorem find_monic_quadratic_polynomial :
  is_monic monic_quadratic_polynomial ∧
  has_real_coefficients monic_quadratic_polynomial ∧
  has_root monic_quadratic_polynomial (2 + -1 * complex.i) :=
by
  sorry

end find_monic_quadratic_polynomial_l72_72713


namespace mildred_initial_oranges_l72_72507

theorem mildred_initial_oranges (final_oranges : ℕ) (added_oranges : ℕ) 
  (final_oranges_eq : final_oranges = 79) (added_oranges_eq : added_oranges = 2) : 
  final_oranges - added_oranges = 77 :=
by
  -- proof steps would go here
  sorry

end mildred_initial_oranges_l72_72507


namespace correct_statements_in_math_problem_l72_72287

theorem correct_statements_in_math_problem :
  ∃ (s2 s3 s6 : Prop),
    (s2 ∧ s3 ∧ s6) ∧
    (s2 ↔ ∀ {r : ℝ}, |r| > 0 → ∀ {x y : ℝ}, x ≠ 0 ∧ y ≠ 0 → (|r| = 1 ↔ linear_dep x y)) ∧
    (s3 ↔ ∀ (c : ℝ) (data : list ℝ), (variance data = variance (list.map (λ x, x + c) data))) ∧
    (s6 ↔ ∀ {S1 S2 S3 S4 r : ℝ}, r > 0 →  (volume_tetrahedron S1 S2 S3 S4 r = (1/3) * (S1 + S2 + S3 + S4) * r)) :=
sorry

end correct_statements_in_math_problem_l72_72287


namespace nl_equal_twice_median_l72_72136

namespace Geometry

-- Define the structure of a triangle with vertices A, B, C
structure Triangle :=
(A B C : Point)

-- Define the midpoint function
def midpoint (p q : Point) : Point :=
{ -- midpoint definition }

-- Define the length of a segment
def length (p q : Point) : ℝ :=
{ -- length definition }

-- Define that squares are constructed externally on sides AB, BC of triangle ABC
structure SquareOnSideExternally (t : Triangle) (p1 p2 : Point) :=
(BAMN : Square t.A t.B p1 p2)
(BCKL : Square t.B t.C p1 p2)

-- Definition of the points N and L from the squares BAMN and BCKL, respectively
def PointN (sq : Square t.A t.B) : Point := { -- Point N definition }
def PointL (sq : Square t.B t.C) : Point := { -- Point L definition }

-- Median definition in triangle from vertex B to midpoint Q of the opposite side
def median (t : Triangle) (B : t.B) (Q : midpoint t.A t.C) : Point := { -- median definition }

-- Define the proof problem
theorem nl_equal_twice_median (t : Triangle) (sqs: SquareOnSideExternally t (PointN sqs.BAMN) (PointL sqs.BCKL)) :
  2 * length t.B (midpoint t.A t.C) = length (PointN sqs.BAMN) (PointL sqs.BCKL) :=
sorry

end Geometry

end nl_equal_twice_median_l72_72136


namespace jaime_saves_enough_l72_72322

-- Definitions of the conditions
def weekly_savings : ℕ := 50
def bi_weekly_expense : ℕ := 46
def target_savings : ℕ := 135

-- The proof goal
theorem jaime_saves_enough : ∃ weeks : ℕ, 2 * ((weeks * weekly_savings - bi_weekly_expense) / 2) = target_savings := 
sorry

end jaime_saves_enough_l72_72322


namespace highest_throw_christine_janice_l72_72678

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l72_72678


namespace isosceles_triangle_k_eq_l72_72431

theorem isosceles_triangle_k_eq (k : ℝ) : 
  (∃ x1 x2 : ℝ, (x1^2 - 4*x1 + k = 0) ∧ (x2^2 - 4*x2 + k = 0) ∧ 
  (x1 = x2 ∨ x1 = 3 ∨ x2 = 3) ∧ (x1 + x2 = 4) ∧ (3 ∈ {x1, x2})) → 
  (k = 3 ∨ k = 4) :=
by 
  sorry

end isosceles_triangle_k_eq_l72_72431


namespace range_of_a_l72_72719

def f (x : ℝ) : ℝ := x - 2 / x
def g (x a : ℝ) : ℝ := -x^2 + a * x - 5

theorem range_of_a :
  (∀ x1 : ℝ, x1 ∈ set.Icc 1 2 → ∃ x2 : ℝ, x2 ∈ set.Icc 2 4 ∧ g x2 a ≤ f x1) ↔ a ≤ 5 := by
suffices : sorry
exact this

end range_of_a_l72_72719


namespace find_a_and_b_l72_72927

variable {x : ℝ}

/-- The problem statement: Given the function y = b + a * sin x (with a < 0), and the maximum value is -1, and the minimum value is -5,
    find the values of a and b. --/
theorem find_a_and_b (a b : ℝ) (h : a < 0) 
  (h1 : ∀ x, b + a * Real.sin x ≤ -1)
  (h2 : ∀ x, b + a * Real.sin x ≥ -5) : 
  a = -2 ∧ b = -3 := sorry

end find_a_and_b_l72_72927


namespace ice_cream_arrangements_is_correct_l72_72888

-- Let us define the problem: counting the number of unique stacks of ice cream flavors
def ice_cream_scoops_arrangements : ℕ :=
  let total_scoops := 5
  let vanilla_scoops := 2
  Nat.factorial total_scoops / Nat.factorial vanilla_scoops

-- Assertion that needs to be proved
theorem ice_cream_arrangements_is_correct : ice_cream_scoops_arrangements = 60 := by
  -- Proof to be filled in; current placeholder
  sorry

end ice_cream_arrangements_is_correct_l72_72888


namespace red_ball_count_l72_72952

theorem red_ball_count 
  (bags : List ℕ) 
  (H_distinct : bags = [7, 15, 16, 10, 23]) 
  (H_one_red : ∃ (red_bag : ℕ), red_bag ∈ bags ∧ ∀ other_bag ∈ bags, other_bag ≠ red_bag → red_bag = other_bag) 
  (H_yellow_blue_ratio : ∃ (yellow : ℕ) (blue : ℕ), yellow = 2 * blue ∧ ∑ b in (bags \ [red_bag]), b = yellow + blue) :
  ∃ red_bag ∈ bags, red_bag = 23 :=
sorry

end red_ball_count_l72_72952


namespace solve_equation_l72_72720

def euler_totient (n : ℕ) : ℕ := sorry  -- Placeholder, Euler's φ function definition
def sigma_function (n : ℕ) : ℕ := sorry  -- Placeholder, σ function definition

theorem solve_equation (x : ℕ) : euler_totient (sigma_function (2^x)) = 2^x → x = 1 := by
  sorry

end solve_equation_l72_72720


namespace sum_squares_mod_eq_6_l72_72597

def squares_mod (n : ℕ) : ℕ :=
  (List.range n).map (λ x => (x.succ * x.succ % 13))

noncomputable def sum_squares_mod : ℕ :=
  (squares_mod 15).sum 

theorem sum_squares_mod_eq_6 : sum_squares_mod % 13 = 6 := 
by
  sorry

end sum_squares_mod_eq_6_l72_72597


namespace projection_correct_l72_72729

variables (a b : ℝ^3)

-- Given conditions
def norm_a : real := 3
def norm_b : real := 5
def dot_ab : real := 12

-- Projection of a onto b
def projection_of_a_on_b : real := (dot_ab) / (norm_b)

theorem projection_correct : projection_of_a_on_b = 12 / 5 := by
  sorry

end projection_correct_l72_72729


namespace arithmetic_sequence_common_difference_l72_72435

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h1 : a 1 = 1) (h3 : a 3 = 4) :
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 / 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l72_72435


namespace radius_of_inscribed_circle_lt_one_third_longer_leg_l72_72516

theorem radius_of_inscribed_circle_lt_one_third_longer_leg {a b c r : ℝ} 
  (hABC : ∀ {A B C : ℝ}, ∃ (a : ℝ) (b : ℝ) (c : ℝ) (r : ℝ), (a + b = c ∧ r = (a + b - c) / 2 ∧ (r < b / 3)) ) :  
  (∀ (a b : ℝ), a ≤ 1 ∧ b = 1 → ∀ {r : ℝ}, r < b / 3) := 
by 
  intro a b hab 
  have h1 : a + b = √(a^2 + b^2) := sorry
  have h2 : r = (a + b - √(a^2 + b^2)) / 2 := sorry
  have h3 : (a + b) / 2 - (√(a^2 + b^2)) / 2 ≤ a / 2 + b / 2 - 1 / √(2) (a + b) / 2 := sorry 
  have h4 : a ≤ 1  := sorry
  have h5 : b = 1 := sorry
  have h6 : r < b / 3 := sorry
  exact h6

end radius_of_inscribed_circle_lt_one_third_longer_leg_l72_72516


namespace problem_l72_72798

open Real

-- Definitions of the variables and constants
variables {A B C : ℝ} {a b c : ℝ} (f : ℝ → ℝ)

-- Given conditions
def conditions (A B C a b c : ℝ) (f : ℝ → ℝ) : Prop :=
  (f x = sin (3 * x + B) + cos (3 * x + B)) ∧
  (∀ x, f x = f (-x)) ∧
  (b = f (π / 12)) ∧
  (a = sqrt 2 / 2)

-- Proof problem: we need to prove these statements given the conditions
theorem problem (A B C a b c : ℝ) (f : ℝ → ℝ) (h : conditions A B C a b c f) :
  b = 1 ∧ C = 7 * π / 12 :=
by
  obtain ⟨_, _, _⟩ := h
  sorry

end problem_l72_72798


namespace matrix_power_identity_l72_72302

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -Real.sqrt 2], ![Real.sqrt 2, 2]]

theorem matrix_power_identity :
  A ^ 4 = ![![(-64 : ℝ), 0], ![0, (-64 : ℝ)]] :=
by
  sorry

end matrix_power_identity_l72_72302


namespace binary_sum_in_base_10_l72_72975

theorem binary_sum_in_base_10 :
  (255 : ℕ) + (63 : ℕ) = 318 :=
sorry

end binary_sum_in_base_10_l72_72975


namespace eccentricity_hyperbola_l72_72552

theorem eccentricity_hyperbola : 
  let a2 := 4
  let b2 := 5
  let e := Real.sqrt (1 + (b2 / a2))
  e = 3 / 2 := by
    apply sorry

end eccentricity_hyperbola_l72_72552


namespace opal_total_savings_l72_72872

-- Definitions based on conditions
def initial_winnings : ℝ := 100.00
def first_saving : ℝ := initial_winnings / 2
def first_bet : ℝ := initial_winnings / 2
def profit_percentage : ℝ := 0.60
def second_profit : ℝ := profit_percentage * first_bet
def total_second_earnings : ℝ := first_bet + second_profit
def second_saving : ℝ := total_second_earnings / 2

-- Theorem statement
theorem opal_total_savings : first_saving + second_saving = 90.00 := by
  sorry

end opal_total_savings_l72_72872


namespace area_of_equilateral_triangle_l72_72360

noncomputable def hyperbola : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 1}

noncomputable def A : ℝ × ℝ := (-1, 0)

noncomputable def on_hyperbola (p : ℝ × ℝ) : Prop := (p.1^2 - p.2^2 = 1) ∧ (p.1 > 0)

axiom B_def : ∃ B : ℝ × ℝ, on_hyperbola B
axiom C_def : ∃ C : ℝ × ℝ, on_hyperbola C

noncomputable def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d := λ p q, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = d A C ∧ d A B = d B C

theorem area_of_equilateral_triangle :
  ∀ (B C : ℝ × ℝ),
    on_hyperbola B → on_hyperbola C → equilateral_triangle A B C →
    let d := λ p q, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    let side := d A B
    S = (Real.sqrt 3) / 4 * (5 + 2 * Real.sqrt 7) :=
sorry

end area_of_equilateral_triangle_l72_72360


namespace find_divisor_of_115_l72_72328

theorem find_divisor_of_115 (x : ℤ) (N : ℤ)
  (hN : N = 115)
  (h1 : N % 38 = 1)
  (h2 : N % x = 1) :
  x = 57 :=
by
  sorry

end find_divisor_of_115_l72_72328


namespace sheila_hourly_pay_l72_72890

theorem sheila_hourly_pay :
  ∃ R : ℝ, let regular_hours := 3 * 8 + 2 * 6,
               overtime_hours := 3 + 2,
               overtime_pay := 1.5 * R * overtime_hours,
               weekly_task_pay := 45,
               total_earnings := 36 * R + 7.5 * R + 45
           in total_earnings = 535 ∧ R ≈ 11.26 :=
begin
  sorry
end

end sheila_hourly_pay_l72_72890


namespace mean_of_remaining_students_l72_72801

theorem mean_of_remaining_students {n : ℕ} (h : n > 15) (overall_mean : 10) (group_mean : 15 * 16) :
  ((10 * n) - 240) / (n - 15) = (10n - 240) / (n - 15) := 
by
  sorry

end mean_of_remaining_students_l72_72801


namespace number_of_valid_outfits_l72_72990

theorem number_of_valid_outfits 
  (shirts : ℕ) (pants : ℕ) (hats : ℕ) (colors_pants : ℕ) (colors_shirts_hats : ℕ)
  (different_colors_needed : ∀ color: ℕ, (color = colors_pants) → (color ≠ colors_shirts_hats))
  (colors_shirts : fin colors_shirts_hats)
  (colors_pants_h : fin colors_pants) 
  (colors_hats : fin colors_shirts_hats) :
  shirts = 8 → pants = 4 → hats = 8 → colors_pants = 4 → colors_shirts_hats = 8 →
  (∀ (cc : ℕ), (cc ∈ {0, 1, 2, 3}) → ((colors_shirts cc) ≠ (colors_hats cc))) →
  ∃ outfit_combinations : ℕ, outfit_combinations = 252 := sorry

end number_of_valid_outfits_l72_72990


namespace exists_infinite_n_block_zeros_l72_72518

theorem exists_infinite_n_block_zeros :
  ∃ᶠ n : ℕ, ∃ m k : ℕ, k > 0 ∧ n > 0 ∧ 5^n = 10^(m + 1976) * k :=
sorry

end exists_infinite_n_block_zeros_l72_72518


namespace intersection_A_B_l72_72361

def A : Set ℤ := {-2, -1, 1, 2, 4}
def B : Set ℝ := {x | -2 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = {-1, 1, 2} := sorry

end intersection_A_B_l72_72361


namespace maximize_area_of_triangle_PAB_l72_72175

def parabola (x : ℝ) : ℝ := 4 - x^2
def line (x : ℝ) : ℝ := 4 * x

theorem maximize_area_of_triangle_PAB :
  let A := ⟨-2, 0⟩
  let B := ⟨2, 8⟩
  let P := ⟨-2, 0⟩
  in ∀ P, (P.1, P.2) = (-2, 0) :=
sorry

end maximize_area_of_triangle_PAB_l72_72175


namespace question_1_question_2_l72_72110

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem question_1 (m : ℝ) :
  (∀ x : ℝ, f x ≤ -m^2 + 6 * m) ↔ (1 ≤ m ∧ m ≤ 5) :=
by
  sorry

theorem question_2 (a b c : ℝ) (h1 : 3 * a + 4 * b + 5 * c = 5) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ 1 / 2 :=
by
  sorry

end question_1_question_2_l72_72110


namespace apple_stack_count_l72_72650

theorem apple_stack_count : 
  let base := 4 in
  let first_layer := base * base * base in
  let second_layer := (base - 1) * (base - 1) * (base - 1) in
  let third_layer := (base - 2) * (base - 2) * (base - 2) in
  let fourth_layer := (base - 3) * (base - 3) * (base - 3) in
  first_layer + second_layer + third_layer + fourth_layer = 100 :=
by
  let base := 4
  let first_layer := base * base * base
  let second_layer := (base - 1) * (base - 1) * (base - 1)
  let third_layer := (base - 2) * (base - 2) * (base - 2)
  let fourth_layer := (base - 3) * (base - 3) * (base - 3)
  show first_layer + second_layer + third_layer + fourth_layer = 100
  sorry

end apple_stack_count_l72_72650


namespace find_sets_C_l72_72362

theorem find_sets_C (B C : Set ℕ) :
  (B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
  (B ∩ C = ∅) ∧
  (∑ x in B, x = ∏ x in C, x) →
  C = {6, 7} ∨ C = {1, 4, 10} ∨ C = {1, 2, 3, 7} :=
by 
  sorry

end find_sets_C_l72_72362


namespace min_value_of_f_l72_72622

noncomputable def f (x : ℝ) : ℝ := (cos x)^2 / (cos x * sin x - (sin x)^2)

theorem min_value_of_f : 0 < x ∧ x < π / 4 → Min (f x) 4 := 
by sorry

end min_value_of_f_l72_72622


namespace sum_of_ratios_eq_n_l72_72632

-- Define the problem statement
theorem sum_of_ratios_eq_n
  {n : ℕ}
  (A : Fin n → ℝ × ℝ)
  (B : Fin n → ℝ × ℝ)
  (G : ℝ × ℝ)
  (starts_on_line1 : ∀ i, line1 (A i))
  (ends_on_line2 : ∀ i, line2 (B i))
  (passes_through_G : ∀ i, collinear (A i) G (B i))
  (centroid_condition : G = centroid (finset.univ.image A)) :
  (finset.univ.sum (λ i, distance (A i) G / distance G (B i))) = n := sorry

end sum_of_ratios_eq_n_l72_72632


namespace problem_condition_l72_72048

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l72_72048


namespace simplify_sequence_product_l72_72531

theorem simplify_sequence_product : ( ∏ n in ( Finset.range 402).image (λ n, (5 * (n + 1) + 5) / (5 * (n + 1))),, 402 := sorry

end simplify_sequence_product_l72_72531


namespace zero_point_not_suff_and_nec_for_opposite_signs_l72_72743

-- Definitions of the conditions
variables {α : Type*} [LinearOrder α] [TopologicalSpace α] 
variables (f : α → ℝ) (a b : α)

-- Given conditions:
-- 1. Continuous function
-- 2. Zero point in the interval (a,b)
-- 3. f(a) * f(b) < 0
axiom continuous_f : Continuous f

def has_zero_in_interval : Prop :=
  ∃ x ∈ Ioo a b, f x = 0

def opposite_signs_at_ends : Prop :=
  f a * f b < 0

-- The theorem to be proved
theorem zero_point_not_suff_and_nec_for_opposite_signs (continuous_f : Continuous f) :
  ¬ (has_zero_in_interval f a b ↔ opposite_signs_at_ends f a b) :=
sorry

end zero_point_not_suff_and_nec_for_opposite_signs_l72_72743


namespace B_starts_6_hours_after_A_l72_72281

theorem B_starts_6_hours_after_A 
    (A_walk_speed : ℝ) (B_cycle_speed : ℝ) (catch_up_distance : ℝ)
    (hA : A_walk_speed = 10) (hB : B_cycle_speed = 20) (hD : catch_up_distance = 120) :
    ∃ t : ℝ, t = 6 :=
by
  sorry

end B_starts_6_hours_after_A_l72_72281


namespace find_slope_k_l72_72016

noncomputable theory

open Real

variables {x y k : ℝ}

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Define the slope of the line intersecting the ellipse
def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y1 - y2) / (x1 - x2)

-- Define points A and B intersections
def is_intersection (x y : ℝ) : Prop := ellipse x y

-- Define midpoint M of segment AB
def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

-- Define angle θ with tanθ = 2√2
def tan_theta : ℝ := 2 * sqrt 2

-- Define the theorem to prove
theorem find_slope_k (x1 y1 x2 y2 : ℝ) 
  (hx1 : is_intersection x1 y1) (hx2 : is_intersection x2 y2) 
  (hm : midpoint x1 y1 x2 y2) :
  (tan (atan (line_slope x1 y1 x2 y2) + π - atan ((snd hm) / (fst hm)))) = tan_theta →
  k = ±(sqrt 2 / 2) :=
sorry

end find_slope_k_l72_72016


namespace max_value_of_symmetric_function_l72_72420

noncomputable def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) → ∃ x : ℝ, ∀ y : ℝ, f x a b ≥ f y a b ∧ f x a b = 16 :=
sorry

end max_value_of_symmetric_function_l72_72420


namespace highest_throw_is_37_feet_l72_72684

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l72_72684


namespace carson_speed_l72_72447

def Jerry_one_way_time : ℕ := 15 -- Jerry takes 15 minutes one-way
def school_distance : ℕ := 4 -- Distance to the school is 4 miles

theorem carson_speed :
  let carson_time_minutes := Jerry_one_way_time * 2,
      carson_time_hours := carson_time_minutes / 60,
      carson_speed := school_distance / carson_time_hours
  in carson_speed = 8 := by
  sorry

end carson_speed_l72_72447


namespace range_of_a_l72_72391

-- Define the inequality condition
def inequality (x a : ℝ) : Prop :=
  2 * x^2 + a * x - a^2 > 0

-- State the main problem
theorem range_of_a (a: ℝ) : 
  inequality 2 a -> (-2 < a) ∧ (a < 4) :=
by
  sorry

end range_of_a_l72_72391


namespace max_distance_on_20_gallons_l72_72279

def highway_mpg := 12.2
def city_mpg := 7.6
def gallons := 20

theorem max_distance_on_20_gallons : (highway_mpg * gallons = 244) :=
by sorry

end max_distance_on_20_gallons_l72_72279


namespace sin_sum_leq_floor_square_div_4_l72_72737

theorem sin_sum_leq_floor_square_div_4 {θ : ℕ → ℝ} {n : ℕ} 
  (h_sum_zero : ∑ i in finset.range n, sin (θ i) = 0) :
  abs (∑ i in finset.range n, (i + 1) * sin (θ i)) ≤ ⌊(n^2 : ℝ) / 4⌋ :=
sorry

end sin_sum_leq_floor_square_div_4_l72_72737


namespace general_term_correct_l72_72182

-- Define the sequence a_n
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2^n

-- Define the general term formula for the sequence a_n
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = n * 2^(n - 1)

-- Theorem statement: the general term formula holds for the sequence a_n
theorem general_term_correct (a : ℕ → ℕ) (h_seq : seq a) : general_term a :=
by
  sorry

end general_term_correct_l72_72182


namespace min_sandwiches_left_Cm_l72_72300

-- Definitions and variables
variable (Cm Uf Pp S : Type)

-- Conditions to capture the initial settings and patterns
def initial_sandwiches (x : Type) : ℕ := 15
def sandwiches_left_5_minutes (x : Type) [x = Uf] : ℕ := 8
def total_sandwiches_eaten : ℕ := 4 * 5
def thefts_5_minutes : ℕ := 5
def sandwiches_by_own (x : Type) : ℕ := 5

-- Statement to be proven
theorem min_sandwiches_left_Cm (u_eq_8 : sandwiches_left_5_minutes Uf = 8) :
  ∃ remaining : ℕ, remaining = 7 := by
  sorry

end min_sandwiches_left_Cm_l72_72300


namespace K_value_of_sphere_volume_l72_72501

theorem K_value_of_sphere_volume :
  let side_length : ℝ := 3
  let cube_surface_area : ℝ := 6 * side_length^2
  let sphere_surface_area : ℝ := cube_surface_area
  let sphere_radius : ℝ := sqrt (sphere_surface_area / (4 * Real.pi))
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  let K := 54
  sphere_volume = K * sqrt(18) / sqrt(Real.pi) :=
by
  sorry

end K_value_of_sphere_volume_l72_72501


namespace probability_of_odd_sums_l72_72800

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def grid_valid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ r : Fin 3, is_odd (grid.rowSum r) ∧ 
               ∀ c : Fin 3, is_odd (grid.colSum c) ∧
               is_odd (grid.diagonalSum 0) ∧ 
               is_odd (grid.diagonalSum 1) ∧
               permutations (grid) = factorial 9

theorem probability_of_odd_sums:
  (∃ (grid : Matrix (Fin 3) (Fin 3) ℕ), 
    (∀ n, 1 ≤ n ∧ n ≤ 9 → (∃ r : Fin 3, ∃ c : Fin 3, grid[r, c] = n)) ∧
    grid_valid grid) →
  (48 / 362880 = 1 / 7560) := 
sorry

end probability_of_odd_sums_l72_72800


namespace bridge_length_l72_72230

def length_of_bridge (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (time / 60)

theorem bridge_length (speed : ℝ) (time : ℝ) (h_speed : speed = 10) (h_time : time = 3) : length_of_bridge speed time = 1 / 2 :=
by
  rw [length_of_bridge, h_speed, h_time]
  norm_num
  rfl

end bridge_length_l72_72230


namespace find_missing_fraction_l72_72567

def f1 := 1/3
def f2 := 1/2
def f3 := 1/5
def f4 := 1/4
def f5 := -9/20
def f6 := -9/20
def total_sum := 45/100
def missing_fraction := 1/15

theorem find_missing_fraction : f1 + f2 + f3 + f4 + f5 + f6 + missing_fraction = total_sum :=
by
  sorry

end find_missing_fraction_l72_72567


namespace limit_product_at_infinity_l72_72482

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Assuming that the domain of f and g is \mathbb{R}_{\ge 0}
axiom h1 : Continuous f
axiom h2 : ∀ x y, x ≤ y → g(x) ≤ g(y)
axiom h3 : ∀ M : ℝ, ∃ x : ℝ, 0 ≤ x ∧ g(x) ≥ M
axiom h4 : ∀ (x_n : ℕ → ℚ), x_n ⟶ ∞ → (f ∘ x_n) * (g ∘ x_n) ⟶ 1

theorem limit_product_at_infinity :
  tendsto (λ x => f(x) * g(x)) at_top (𝓝 1) :=
sorry

end limit_product_at_infinity_l72_72482


namespace length_D_to_B_l72_72427

/- 
  Parameters:
    - Triangle with vertices D, E, F where DE and EF are legs and DF is the hypotenuse.
    - The angle at D is 90 degrees.
    - DF (hypotenuse) = 10 units.
    - The radius of the inscribed circle is 2 units.
  To prove:
    - The length x from point D to point B along line DF is either 4 or 6.
-/

theorem length_D_to_B {D E F : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (DE : D → E → ℝ) (EF : E → F → ℝ) (DF : D → F → ℝ) (A B C : D → E → F → ℝ)
  (hypotenuse_len : DF = 10) (radius_len : 2)
  (r_triangle : right_triangle D E F)
  (inscribed_circle : touches_circle_at A B C)
  (angle_at_D : ∠DEF = 90):
  (x = 4 ∨ x = 6) :=
begin
  sorry
end

end length_D_to_B_l72_72427


namespace min_value_g_in_interval_l72_72009

-- Define the power function f and the point condition
def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

-- Define the function g using the power function
def g (x : ℝ) (α : ℝ) : ℝ := (2 * x - 1) * f x α

-- Define the condition that the function passes through the point (3, 1/3)
def passes_through (α : ℝ) : Prop := f 3 α = 1 / 3

-- Theorem stating that the minimum value of g in the interval [1/2, 2] is 0,
-- given the condition that the function passes through the point (3, 1/3)
theorem min_value_g_in_interval (α : ℝ) (h : passes_through α) :
  infi (λ x, g x α) (set.Icc (1/2 : ℝ) (2 : ℝ)) = 0 := sorry

end min_value_g_in_interval_l72_72009


namespace socks_same_color_l72_72044

theorem socks_same_color :
  (∑ c in Finset.range 3, if c = 0 then Nat.choose 5 2 else if c = 1 then Nat.choose 5 2 else Nat.choose 2 2) = 21 := by
  sorry

end socks_same_color_l72_72044


namespace simplify_expression_l72_72781

variable (x y : ℝ)

def A := x^2 + 3 * x * y + y^2
def B := x^2 - 3 * x * y + y^2

theorem simplify_expression : A - (B + 2 * B - (A + B)) = 12 * x * y :=
by
  sorry

end simplify_expression_l72_72781


namespace sum_squares_mod_13_l72_72619

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 5 :=
by sorry

end sum_squares_mod_13_l72_72619


namespace plane_equation_l72_72496

theorem plane_equation (a b c d : ℤ)
  (h1 : a = 19) (h2 : b = -67) (h3 : c = 109) (h4 : d = -362)
  (lineM : ∀ (x y z : ℝ), 2 * x - y + 2 * z = 4 ∧ 3 * x + y - z = 6)
  (planeQ : ∀ (x y z : ℝ), a * x + b * y + c * z + d = 0)
  (distance : ∀ {x y z : ℝ}, (x, y, z) = (2, 0, 3) → (|a * 2 + b * 0 + c * 3 + d| / Real.sqrt ((a ^ 2 + b ^ 2 + c ^ 2)) = 3 / Real.sqrt 2))
  : ∃ (A B C D : ℤ), 
    (A = a) ∧ (B = b) ∧ (C = c) ∧ (D = d) ∧ 
    (A > 0) ∧ (Int.gcd A B C D = 1) :=
sorry

end plane_equation_l72_72496


namespace triangle_area_l72_72067

open Real

-- Define the angles A and C, side a, and state the goal as proving the area
theorem triangle_area (A C : ℝ) (a : ℝ) (hA : A = 30 * (π / 180)) (hC : C = 45 * (π / 180)) (ha : a = 2) : 
  (1 / 2) * ((sqrt 6 + sqrt 2) * (2 * sqrt 2) * sin (30 * (π / 180))) = sqrt 3 + 1 := 
by
  sorry

end triangle_area_l72_72067


namespace LaShawn_twice_Kymbrea_after_25_months_l72_72827

theorem LaShawn_twice_Kymbrea_after_25_months : 
  ∀ (x : ℕ), (10 + 6 * x = 2 * (30 + 2 * x)) → x = 25 :=
by
  intro x
  sorry

end LaShawn_twice_Kymbrea_after_25_months_l72_72827


namespace sum_squares_mod_eq_6_l72_72596

def squares_mod (n : ℕ) : ℕ :=
  (List.range n).map (λ x => (x.succ * x.succ % 13))

noncomputable def sum_squares_mod : ℕ :=
  (squares_mod 15).sum 

theorem sum_squares_mod_eq_6 : sum_squares_mod % 13 = 6 := 
by
  sorry

end sum_squares_mod_eq_6_l72_72596


namespace min_value_of_fx_l72_72052

theorem min_value_of_fx (x : ℝ) (h : x * Real.log 2 / Real.log 5 ≥ -1) :
  ∃ t : ℝ, (t = 2^x) ∧ (t ≥ 1/5) ∧ ∀ y : ℝ, (y = 4^x - 2^(x+1) - 3) → y ≥ -4 :=
by
  sorry

end min_value_of_fx_l72_72052


namespace problem_statement_l72_72174

-- Define the floor function
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the problem statement as a Lean theorem
theorem problem_statement (n : ℕ) :
  let I := (n + 1) ^ 2 + n - (floor (real.sqrt ((n + 1) ^ 2 + n + 1))) ^ 2
  in I > 0 := 
by
  sorry

end problem_statement_l72_72174


namespace salary_january_l72_72545

theorem salary_january
  (J F M A May : ℝ)  -- declare the salaries as real numbers
  (h1 : (J + F + M + A) / 4 = 8000)  -- condition 1
  (h2 : (F + M + A + May) / 4 = 9500)  -- condition 2
  (h3 : May = 6500) :  -- condition 3
  J = 500 := 
by
  sorry

end salary_january_l72_72545


namespace find_area_CDE_l72_72869

-- Define the areas of the given triangles
variable {R : Type} [LinearOrderedField R]

-- Points D on AC and E on BC of triangle ABC
variables {A B C D E F : R}

-- The areas of the given triangles
variables (area_ABF area_ADF area_BEF : R)

-- The specific values of the areas given as conditions
def area_ABF := 1
def area_ADF := 1 / 4
def area_BEF := 1 / 5

-- Define the overall area of each triangle
def area_ABC (area_CDE : R) : R :=
  area_ABF + area_ADF + area_BEF + area_CDE

-- The relationship based on given conditions
theorem find_area_CDE (area_CDE : R) :
  area_CDE = (3 : R) / 38 := sorry

end find_area_CDE_l72_72869


namespace calc_3_power_a_mul_27_power_b_l72_72409

theorem calc_3_power_a_mul_27_power_b (a b : ℤ) (h : a + 3 * b - 2 = 0) : 3^a * 27^b = 9 :=
by
  sorry

end calc_3_power_a_mul_27_power_b_l72_72409


namespace determinant_roots_cubic_eq_l72_72848

noncomputable def determinant_of_matrix (a b c : ℝ) : ℝ :=
  a * (b * c - 1) - (c - 1) + (1 - b)

theorem determinant_roots_cubic_eq {a b c p q r : ℝ}
  (h1 : a + b + c = p)
  (h2 : a * b + b * c + c * a = q)
  (h3 : a * b * c = r) :
  determinant_of_matrix a b c = r - p + 2 :=
by {
  sorry
}

end determinant_roots_cubic_eq_l72_72848


namespace sum_of_roots_is_zero_l72_72463

theorem sum_of_roots_is_zero
  (Q : Polynomial ℝ)
  (h_monic : Q.monic)
  (h_deg : Q.natDegree = 4)
  {θ : ℝ}
  (h_theta_range : 0 < θ ∧ θ < π / 6)
  (h_roots : ∃ z1 z2 : ℂ, 
    z1 = complex.ofReal (cos θ) + complex.I * complex.ofReal (sin (2 * θ)) ∧ 
    z2 = -complex.ofReal (cos θ) + complex.I * complex.ofReal (sin (2 * θ)) ∧
    Q.roots = [z1, z2, conj z1, conj z2])
  (h_area : ∀ z1 z2 : ℂ, 
    z1 = complex.ofReal (cos θ) + complex.I * complex.ofReal (sin (2 * θ)) → 
    z2 = -complex.ofReal (cos θ) + complex.I * complex.ofReal (sin (2 * θ)) →
    4 * abs (cos θ * sin (2 * θ)) = Q.eval 0) :
  Q.roots.sum = 0 :=
begin
  sorry
end

end sum_of_roots_is_zero_l72_72463


namespace pebble_pile_impossibility_l72_72587

theorem pebble_pile_impossibility :
  ∀(piles : List ℕ), 
  piles = [51, 49, 5] ∧
  (∀piles', (piles' = merge_two_piles piles ∨ piles' = divide_pile piles) → merge_or_divide_steps piles piles') →
  ¬(∃piles'', piles''.length = 105 ∧ ∀ x, x ∈ piles'' → x = 1) :=
begin
  sorry
end

end pebble_pile_impossibility_l72_72587


namespace opposite_of_neg_quarter_l72_72939

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end opposite_of_neg_quarter_l72_72939


namespace vector_magnitude_l72_72775

noncomputable theory

variables (a b : V) (V : Type*) [inner_product_space ℝ V]

-- Given conditions
axiom norm_a : ∥a∥ = 3
axiom norm_b : ∥b∥ = 4
axiom angle_120 : real.angle a b = real.angle.pi_div_3

-- Theorem statement
theorem vector_magnitude (a b : V) [inner_product_space ℝ V] (ha : ∥a∥ = 3) (hb : ∥b∥ = 4)
  (hab : ⟪a, b⟫ = -6) : ∥a + (2 : ℝ) • b∥ = 7 :=
by { sorry }

end vector_magnitude_l72_72775


namespace range_c_x0_1_value_c_x0_half_l72_72022

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem range_c_x0_1 : 
  (∀ (x : ℝ), 0 < x → x < ⊤ → f(x) - f(1) ≥ c * (x - 1)) ↔ (c ∈ set.Icc (-1 : ℝ) 1) := 
by
  sorry

theorem value_c_x0_half : 
  (∀ (x : ℝ), 0 < x → x < ⊤ → f(x) - f(1 / 2) ≥ c * (x - 1 / 2)) ↔ (c = -2) := 
by
  sorry

end range_c_x0_1_value_c_x0_half_l72_72022


namespace sum_of_squares_mod_13_l72_72612

theorem sum_of_squares_mod_13 :
  (∑ k in Finset.range 16, k^2) % 13 = 10 :=
by
  sorry

end sum_of_squares_mod_13_l72_72612


namespace alice_savings_by_end_of_third_month_l72_72086

variable (B P : ℝ)

theorem alice_savings_by_end_of_third_month :
  let savings := 180 + B * (1 - P / 100) in
  true := 
begin
  sorry,
end

end alice_savings_by_end_of_third_month_l72_72086


namespace geometric_sequence_from_second_term_l72_72818

theorem geometric_sequence_from_second_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  S 1 = 1 ∧ S 2 = 2 ∧ (∀ n, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
  (∀ n, n ≥ 2 → a (n + 1) = 2 * a n) :=
by
  sorry

end geometric_sequence_from_second_term_l72_72818


namespace monic_quadratic_with_root_real_l72_72710
noncomputable def quadratic_polynomial_with_root : polynomial ℝ :=
  polynomial.X ^ 2 - 4 * polynomial.X + 5

theorem monic_quadratic_with_root_real (α : ℂ) (hα : α = 2 - complex.I) :
    polynomial.eval α quadratic_polynomial_with_root = 0 ∧ 
    polynomial.monic quadratic_polynomial_with_root :=
by
  -- Show that α = 2 - i implies conjugate root 2 + i
  have h_conj : complex.conj α = 2 + complex.I,
  {
    rw [hα, complex.conj_sub, complex.conj_I],
    norm_num,
  }
  -- Verify roots and monic property
  sorry

end monic_quadratic_with_root_real_l72_72710


namespace weights_divisible_into_three_groups_l72_72973

theorem weights_divisible_into_three_groups (n : ℕ) (h : n ≥ 3) : 
  (∃ k : ℕ, n = 3 * k + 2) ∨ (∃ k : ℕ, n = 3 * k + 3) ↔ 
  (∃ g1 g2 g3 : set ℕ, 
     (∀ x ∈ g1, x ≤ n ∧ 1 ≤ x) ∧ 
     (∀ x ∈ g2, x ≤ n ∧ 1 ≤ x) ∧ 
     (∀ x ∈ g3, x ≤ n ∧ 1 ≤ x) ∧ 
     g1 ∪ g2 ∪ g3 = {1, 2, ..., n} ∧ 
     g1 ∩ g2 = ∅ ∧
     g2 ∩ g3 = ∅ ∧
     g1 ∩ g3 = ∅ ∧
     (∑ x in g1, x = ∑ x in g2, x) ∧ 
     (∑ x in g2, x = ∑ x in g3, x)
   ) :=
by sorry

end weights_divisible_into_three_groups_l72_72973


namespace second_carpenter_days_l72_72229

variable {W : Type} [LinearOrderedField W]

def work_done (days : ℕ) (w_per_day : W) : W :=
  days * w_per_day

theorem second_carpenter_days (W_1 W_2 : W) (h1 : W_1 = W_2) :
  (work_done 1 W_1 + work_done 7 W_1 = work_done 4 (W_1 + W_2)) → 
  let total_days := 8 in
  let carpenter2_days := total_days / (W_2 / W_2) in
  carpenter2_days = total_days :=
by
  intros h proof sorry

end second_carpenter_days_l72_72229


namespace part_i_part_ii_l72_72834

-- Define the problem data
noncomputable def a_0 : ℝ := 0
variables {k : ℕ} (a b : Fin k.succ → ℝ)

-- Existence of such polynomials p_n that meet the conditions in part (i)
theorem part_i (n : ℕ) (h : n > k) :
  ∃ (p : ℝ[X]), degree p ≤ n ∧
    (∀ i : Fin k.succ, (derivative^[i] p).eval (-1) = a i) ∧
    (∀ i : Fin k.succ, (derivative^[i] p).eval 1 = b i) ∧
    (∀ x : ℝ, abs x ≤ 1 → abs (p.eval x) ≤ c / n^2) :=
sorry

-- Impossibility of the relation in part (ii)
theorem part_ii :
  ¬ (∀ (n : ℕ), ∃ (p : ℝ[X]), degree p = n ∧
      (∀ i : Fin k.succ, (derivative^[i] p).eval (-1) = a i) ∧
      (∀ i : Fin k.succ, (derivative^[i] p).eval 1 = b i) ∧
      (tendsto (λ n, n^2 * (⨆ x, abs x ≤ 1 → abs (p.eval x))) at_top (nhds 0))) :=
sorry

end part_i_part_ii_l72_72834


namespace teddy_bear_cost_l72_72858

theorem teddy_bear_cost : 
  ∀ (n : ℕ) (cost_per_toy : ℕ) 
  (total_cost : ℕ) (num_teddy_bears : ℕ) 
  (amount_in_wallet : ℕ) (cost_per_bear : ℕ),
  n = 28 → 
  cost_per_toy = 10 → 
  num_teddy_bears = 20 → 
  amount_in_wallet = 580 → 
  total_cost = 280 → 
  total_cost = n * cost_per_toy →
  (amount_in_wallet - total_cost) = num_teddy_bears * cost_per_bear →
  cost_per_bear = 15 :=
by 
  intros n cost_per_toy total_cost num_teddy_bears amount_in_wallet cost_per_bear 
         hn hcost_per_toy hnum_teddy_bears hamount_in_wallet htotal_cost htotal_cost_eq
        hbear_cost_eq,
  sorry

end teddy_bear_cost_l72_72858


namespace josh_bottle_caps_l72_72101

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end josh_bottle_caps_l72_72101


namespace probability_equality_l72_72474

variables {Ω : Type*} [ProbabilitySpace Ω]
variables {N : ℕ} (p q : ℝ) (hp : p > 0) (hq : p + q = 1)

-- Defining the Bernoulli random variables
noncomputable def xi (n : ℕ) : Ω → Bool := 
  λ ω, (Bernoulli (MeasureTheory.probMeasure p)).val ω

-- Sum of i.i.d Bernoulli random variables
noncomputable def S (n : ℕ) : Ω → ℕ
| 0 := 0
| (n+1) := S n + if xi (n+1) then 1 else 0

-- Probability of S_n being equal to k
noncomputable def P_n (n k : ℕ) : ℝ :=
  MeasureTheory.prob (λ ω, S n ω = k)

-- Statement to be proven
theorem probability_equality (n k : ℕ) (hn : n < N) (hk : k ≥ 1) :
  P_n (n+1) k = p * P_n n (k-1) + q * P_n n k :=
sorry

end probability_equality_l72_72474


namespace tomatoes_on_last_plant_l72_72812

theorem tomatoes_on_last_plant (n k : ℕ) (a : ℕ → ℕ) 
  (h1 : n = 12) 
  (h2 : ∀ i, 1 ≤ i → i ≤ n → a i = a (i - 1) + k) 
  (h3 : ∑ i in Finset.range n, a (i + 1) = 186) : 
  a n = 21 := 
sorry

end tomatoes_on_last_plant_l72_72812


namespace area_of_triangle_PST_l72_72443

noncomputable def triangleArea (a b c : ℝ) (angle : ℝ) : ℝ :=
  (1 / 2) * a * b * real.sin angle

noncomputable def trianglePSTArea
  (PQ PS PT : ℝ) (sineP : ℝ) : ℝ :=
  (1 / 2) * PS * PT * sineP

theorem area_of_triangle_PST :
  let PQ := 8
  let QR := 12
  let PR := 10
  let PS := 3
  let PT := 6
  let s := (PQ + QR + PR) / 2
  let areaPQR := real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  let sineP := areaPQR * 2 / (QR * PR)
  trianglePSTArea PQ PS PT sineP = (9 * real.sqrt 7) / 4 :=
by
  let PQ := 8
  let QR := 12
  let PR := 10
  let PS := 3
  let PT := 6
  let s := (PQ + QR + PR) / 2
  let areaPQR := real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  let sineP := areaPQR * 2 / (QR * PR)
  have h : trianglePSTArea PQ PS PT sineP = (9 * real.sqrt 7) / 4 := by
    sorry
  exact h


end area_of_triangle_PST_l72_72443


namespace lcm_140_225_is_6300_l72_72214

def lcm_140_225 : ℕ := Nat.lcm 140 225

theorem lcm_140_225_is_6300 : lcm_140_225 = 6300 :=
by
  sorry

end lcm_140_225_is_6300_l72_72214


namespace meets_standard_writing_requirements_l72_72218

-- Definitions derived from conditions in the problem
def A : ℚ × Var := (-1, x)
def B : ℚ × Var × Var := (7 / 6, x, y) -- improper fraction
def C : ℚ × Var := (0.8, x)
def D : ℚ × Var := (-7 / 2, a)

-- Problem statement
theorem meets_standard_writing_requirements (d_standard : (ℚ × Var) → Prop) :
  d_standard D :=
sorry

end meets_standard_writing_requirements_l72_72218


namespace deal_saves_customer_two_dollars_l72_72653

-- Define the conditions of the problem
def movie_ticket_price : ℕ := 8
def popcorn_price : ℕ := movie_ticket_price - 3
def drink_price : ℕ := popcorn_price + 1
def candy_price : ℕ := drink_price / 2

def normal_total_price : ℕ := movie_ticket_price + popcorn_price + drink_price + candy_price
def deal_price : ℕ := 20

-- Prove the savings
theorem deal_saves_customer_two_dollars : normal_total_price - deal_price = 2 :=
by
  -- We will fill in the proof here
  sorry

end deal_saves_customer_two_dollars_l72_72653


namespace sin_A_of_right_triangle_l72_72797

theorem sin_A_of_right_triangle
  {A B C: Type} [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (triangleABC: Triangle A B C)
  (h1: angle C A B = π / 2)
  (h2: BC / CA = 3 / 4) : sin A = 3 / 5 :=
sorry

end sin_A_of_right_triangle_l72_72797


namespace find_m_l72_72561

noncomputable def h (m : ℤ) : Polynomial ℤ :=
  Polynomial.C (3 * m ^ 2 + 6 * m + 3) +
  Polynomial.C (- (m ^ 2 + m)).comp Polynomial.X * Polynomial.X +
  Polynomial.C (-1).comp Polynomial.X ^ 2 +
  Polynomial.C (1).comp Polynomial.X ^ 3

theorem find_m (m : ℤ) : 3 * m + 21 = 0 → ∀ x : ℤ, h m = (Polynomial.C 1) * (Polynomial.C (x - 3 + 0)) * (Polynomial.C (x - (-6) + 0)) * (Polynomial.C (x - 6 + 0)) → m = -7 :=
sorry

end find_m_l72_72561


namespace determine_month_salary_l72_72544

variables (J F M A May : ℕ)

-- Condition: Average salary for January, February, March, and April is Rs. 8000
def avg1 : Prop := (J + F + M + A) / 4 = 8000

-- Condition: Average salary for February, March, April, and May is Rs. 8700
def avg2 : Prop := (F + M + A + May) / 4 = 8700

-- Condition: Salary for the month of January is Rs. 3700
def jan_salary : Prop := J = 3700

-- Condition: Salary for some month is Rs. 6500
def some_month_salary : Prop := May = 6500

-- Question: In which month did he earn Rs. 6500?
theorem determine_month_salary
  (h1 : avg1)
  (h2 : avg2)
  (h3 : jan_salary) :
  some_month_salary :=
begin
  sorry
end

end determine_month_salary_l72_72544


namespace root_exists_between_0_and_1_l72_72884

theorem root_exists_between_0_and_1 (a b c : ℝ) (m : ℝ) (hm : 0 < m)
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x ^ 2 + b * x + c = 0 :=
by
  sorry

end root_exists_between_0_and_1_l72_72884


namespace draw_dots_l72_72228

theorem draw_dots (line_length interval : ℕ) (no_start_end : Prop) : 
  line_length = 30 → interval = 5 → no_start_end → 
  ∃ n, n = 5 :=
by
  intro h1 h2 h3
  use 5
  sorry

end draw_dots_l72_72228


namespace solution_inequality_l72_72565

theorem solution_inequality (x : ℝ) :
  ((1 / 2) ^ x > 34) ↔ x < - Real.log 34 / Real.log 2 :=
by
  sorry

end solution_inequality_l72_72565


namespace max_distance_is_correct_l72_72867

-- Define the post position and the rope length
def post_position : ℝ × ℝ := (5, 5)
def rope_length : ℝ := 12

-- The function to calculate the distance from the origin to the point (x, y)
def distance_from_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

-- The maximum distance the dog can travel from the origin
def max_distance_from_origin : ℝ :=
  let (x, y) := post_position in
  distance_from_origin x y + rope_length

-- The theorem to prove
theorem max_distance_is_correct :
  max_distance_from_origin = 12 + 5 * Real.sqrt 2 :=
by
  sorry

end max_distance_is_correct_l72_72867


namespace james_dancing_calories_l72_72090

def walking_calories_per_hour : ℕ := 300
def sessions_per_day : ℕ := 2
def hours_per_session : ℝ := 0.5
def days_per_week : ℕ := 4

def dancing_calories_per_hour : ℕ := 2 * walking_calories_per_hour
def calories_per_session : ℝ := dancing_calories_per_hour * hours_per_session
def calories_per_day : ℝ := calories_per_session * sessions_per_day
def calories_per_week : ℝ := calories_per_day * days_per_week

theorem james_dancing_calories : calories_per_week = 2400 := 
by 
  rw [calories_per_week, calories_per_day, calories_per_session, dancing_calories_per_hour],
  simp,
  norm_num,
  sorry

end james_dancing_calories_l72_72090


namespace smallest_positive_period_of_f_monotonically_increasing_interval_of_f_max_min_values_of_f_on_interval_l72_72385

noncomputable def f (x : ℝ) : ℝ :=
  √3 * sin x * cos x - sin x ^ 2 + 1 / 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, T = π ∧ ∀ x : ℝ, f (x + T) = f x := sorry

theorem monotonically_increasing_interval_of_f :
  ∀ k : ℤ, ∀ x : ℝ,
    ( -π / 3 - k * π ≤ x ∧ x ≤ π / 6 - k * π ) →
    (∀ y z, ( -π / 3 - k * π ≤ y ∧ y ≤ π / 6 - k * π ) →
      ( y ≤ z → f y ≤ f z )) := sorry

theorem max_min_values_of_f_on_interval :
  ∃ x₁ x₂ : ℝ,
    0 ≤ x₁ ∧ x₁ ≤ π / 2 ∧ f x₁ = 1 ∧
    0 ≤ x₂ ∧ x₂ ≤ π / 2 ∧ f x₂ = -1 :=
  sorry

end smallest_positive_period_of_f_monotonically_increasing_interval_of_f_max_min_values_of_f_on_interval_l72_72385


namespace fold_line_straight_midpoints_form_parallelogram_midpoints_form_rectangle_midpoints_form_rhombus_midpoints_form_square_l72_72145

theorem fold_line_straight (A B: Type) (l: A → B → Prop) (M: A) :
  (∀ x y: A, l x y → ∃ M: A, (dist M x = dist M y)) →
  (∀ x y: A, l x y → collinear M x y) := sorry

theorem midpoints_form_parallelogram (A B C D K L M N : Type) 
  (AB : A → B → Prop) (BC : B → C → Prop)
  (CD : C → D → Prop) (DA : D → A → Prop)
  (midpoint : ∀ x y z : Type, midpoint x y z → Prop)
  (dist : ∀ x y : Type, dist x y → Prop)
  :
  ((midpoint A B K) ∧ (midpoint B C L) ∧ (midpoint C D M) ∧ (midpoint D A N)) →
  parallelogram K L M N := sorry

theorem midpoints_form_rectangle (A B C D K L M N : Type) 
  (AB : A → B → Prop) (BC : B → C → Prop)
  (CD : C → D → Prop) (DA : D → A → Prop)
  (midpoint : ∀ x y z : Type, midpoint x y z → Prop)
  (dist : ∀ x y : Type, dist x y → Prop)
  :
  ((midpoint A B K) ∧ (midpoint B C L) ∧ (midpoint C D M) ∧ (midpoint D A N)) ∧ 
  (perpendicular AC BD) →
  rectangle K L M N := sorry

theorem midpoints_form_rhombus (A B C D K L M N : Type) 
  (AB : A → B → Prop) (BC : B → C → Prop)
  (CD : C → D → Prop) (DA : D → A → Prop)
  (midpoint : ∀ x y z : Type, midpoint x y z → Prop)
  (dist : ∀ x y : Type, dist x y → Prop)
  :
  ((midpoint A B K) ∧ (midpoint B C L) ∧ (midpoint C D M) ∧ (midpoint D A N)) ∧ 
  (equal_length AC BD) →
  rhombus K L M N := sorry

theorem midpoints_form_square (A B C D K L M N : Type) 
  (AB : A → B → Prop) (BC : B → C → Prop)
  (CD : C → D → Prop) (DA : D → A → Prop)
  (midpoint : ∀ x y z : Type, midpoint x y z → Prop)
  (dist : ∀ x y : Type, dist x y → Prop) 
  :
  ((midpoint A B K) ∧ (midpoint B C L) ∧ (midpoint C D M) ∧ (midpoint D A N)) ∧ 
  (equal_length AC BD ∧ perpendicular AC BD) →
  square K L M N := sorry

end fold_line_straight_midpoints_form_parallelogram_midpoints_form_rectangle_midpoints_form_rhombus_midpoints_form_square_l72_72145


namespace g100_value_l72_72903

-- Define the function g and its properties
def g (x : ℝ) : ℝ := sorry

theorem g100_value 
  (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g (x / y) + x - y) : 
  g 100 = 99 / 2 := 
sorry

end g100_value_l72_72903


namespace cos_identity_l72_72166

theorem cos_identity :
  (∃ a b c d : ℕ, a * ∀ b c d : ℕ, cos (b * x) * cos (c * x) * cos (d * x) = 4 * cos (6 * x) * cos (5 * x) * cos (3 * x)) →
  a + b + c + d = 18 :=
by
  sorry

end cos_identity_l72_72166


namespace n_not_prime_l72_72478

theorem n_not_prime 
  (p : ℕ) (a b c n : ℕ) 
  (prime_p : nat.prime p) 
  (a_lt_p : a < p) (b_lt_p : b < p) (c_lt_p : c < p) 
  (h1 : p^2 ∣ a + (n-1) * b) 
  (h2 : p^2 ∣ b + (n-1) * c) 
  (h3 : p^2 ∣ c + (n-1) * a) 
  : ¬(nat.prime n) := sorry

end n_not_prime_l72_72478


namespace overall_average_score_l72_72134

theorem overall_average_score 
  (M : ℝ) (E : ℝ) (m e : ℝ)
  (hM : M = 82)
  (hE : E = 75)
  (hRatio : m / e = 5 / 3) :
  (M * m + E * e) / (m + e) = 79.375 := 
by
  sorry

end overall_average_score_l72_72134


namespace necessary_but_not_sufficient_condition_l72_72119

variable (x y : ℝ)

theorem necessary_but_not_sufficient_condition :
  (x ≠ 1 ∨ y ≠ 1) ↔ (xy ≠ 1) :=
sorry

end necessary_but_not_sufficient_condition_l72_72119


namespace distance_focus_line_l72_72550

theorem distance_focus_line : 
  let focus := (2 : ℝ, 0 : ℝ)
  let line_eq := (1 : ℝ) * focus.1 + (- √3) * focus.2 + 0 = 0
  let dist := abs(line_eq) / √((1 : ℝ)^2 + (- √3)^2)
  dist = 1 :=
by {
  sorry -- proof steps go here
}

end distance_focus_line_l72_72550


namespace female_employees_sample_probability_A_middle_BC_adjacent_binomial_expansion_x3_range_of_a_l72_72243

-- Problem 1
theorem female_employees_sample (total employees male_employees total_sample female_employees_sample : ℕ) 
  (h_total : employees = 750) (h_male : male_employees = 300) (h_sample : total_sample = 45) 
  (h_females : female_employees = employees - male_employees) : 
  female_employees_sample = 27 :=
sorry

-- Problem 2
theorem probability_A_middle_BC_adjacent (total_persons : ℕ) : 
  (total_persons = 9) → (probA_middle_BC_adjacent = 1 / 42) :=
sorry

-- Problem 3
theorem binomial_expansion_x3 (x : ℂ) : coeff_of_x3 = -180 :=
sorry

-- Problem 4
theorem range_of_a (a b x : ℝ) (f : ℝ → ℝ) (h1 : ∀ b, b ≤ 0 → ∀ x, e < x ∧ x ≤ e^2 → f(x) ≥ x)
  (h2 : ∀ x, e < x ∧ x ≤ e^2 → f(x) = a * log x - b * x^2) : 
  a ∈ [ (e^2 / 2), +∞ ) :=
sorry

end female_employees_sample_probability_A_middle_BC_adjacent_binomial_expansion_x3_range_of_a_l72_72243


namespace infinite_primes_equal_divisors_l72_72517

theorem infinite_primes_equal_divisors :
  ∃ᶠ n : ℕ in Filter.atTop, (n > 0) ∧
  let p := Nat.largest_prime_divisor (n^4 + n^2 + 1)
  let q := Nat.largest_prime_divisor ((n+1)^4 + (n+1)^2 + 1)
  in p = q :=
sorry

end infinite_primes_equal_divisors_l72_72517


namespace complement_of_supplement_of_30_degrees_l72_72211

def supplementary_angle (x : ℕ) : ℕ := 180 - x
def complementary_angle (x : ℕ) : ℕ := if x > 90 then x - 90 else 90 - x

theorem complement_of_supplement_of_30_degrees : complementary_angle (supplementary_angle 30) = 60 := by
  sorry

end complement_of_supplement_of_30_degrees_l72_72211


namespace PQ_midpoint_DG_l72_72459

-- Definitions of the points and properties given in the problem
variables {A B C D E F G P Q : Type*}

-- Declaring the points are on a plane
variable [EuclideanGeometry ℝ A B C D E F G P Q]

-- D, E, F are the feet of the perpendiculars from A, B, C to BC, CA, AB respectively
def feet_perpendicular (A B C D E F : Type*) : Prop :=
  perpendicular (line_through A B) (line_through D E) ∧
  perpendicular (line_through B C) (line_through E F) ∧
  perpendicular (line_through C A) (line_through F D)

-- G is the intersection point of AD and EF
def intersection_AD_EF (A D E F G : Type*) : Prop :=
  line_through A D = line_through E F

-- P is the intersection point of the circumcircle of triangle DFG with AB (not F)
def intersection_circumcircle_DFG_AB (D F G P A B : Type*) : Prop :=
  P ∈ circumcircle D F G ∧ P ∉ {F} ∧ P ∈ (line_through A B)

-- Q is the intersection point of the circumcircle of triangle DEG with AC (not E)
def intersection_circumcircle_DEG_AC (D E G Q A C : Type*) : Prop :=
  Q ∈ circumcircle D E G ∧ Q ∉ {E} ∧ Q ∈ (line_through A C)

-- PQ passes through the midpoint of DG
theorem PQ_midpoint_DG (A B C D E F G P Q : Type*) [EuclideanGeometry ℝ A B C D E F G P Q] :
  feet_perpendicular A B C D E F →
  intersection_AD_EF A D E F G →
  intersection_circumcircle_DFG_AB D F G P A B →
  intersection_circumcircle_DEG_AC D E G Q A C →
  midpoint (line_through P Q) (line_through D G) :=
by
  sorry

end PQ_midpoint_DG_l72_72459


namespace james_dancing_calories_l72_72089

theorem james_dancing_calories
  (calories_per_hour_walking : ℕ)
  (calories_per_hour_dancing : ℕ)
  (hours_per_session : ℝ)
  (sessions_per_day : ℕ)
  (days_per_week : ℕ)
  (cal_per_hour_walking : calories_per_hour_walking = 300)
  (cal_per_hour_dancing : calories_per_hour_dancing = 2 * calories_per_hour_walking)
  (hours_per_session_def : hours_per_session = 0.5)
  (sessions_per_day_def : sessions_per_day = 2)
  (days_per_week_def : days_per_week = 4)
  : calories_per_hour_dancing * (hours_per_session * sessions_per_day * days_per_week).natAbs = 2400 := by
  sorry

end james_dancing_calories_l72_72089


namespace birds_are_crows_l72_72069

theorem birds_are_crows (total_birds pigeons crows sparrows parrots non_pigeons: ℕ)
    (h1: pigeons = 20)
    (h2: crows = 40)
    (h3: sparrows = 15)
    (h4: parrots = total_birds - pigeons - crows - sparrows)
    (h5: total_birds = pigeons + crows + sparrows + parrots)
    (h6: non_pigeons = total_birds - pigeons) :
    (crows * 100 / non_pigeons = 50) :=
by sorry

end birds_are_crows_l72_72069


namespace square_area_tangent_circle_l72_72689

theorem square_area_tangent_circle (s r : ℝ) (A B C D : ℝ) (tangent : A = C ∧ B = C ∧ D = C) (midpoint_AC : C / 2 = B) (diagonal_AC: AC = s * Real.sqrt 2) : 
  square_area := s = 4*r → area := s^2 = 16*r^2 := 
begin
  sorry,
end

end square_area_tangent_circle_l72_72689


namespace base7_to_base10_l72_72274

/-- Conversion of a base 7 number to base 10 -/
theorem base7_to_base10 (n : Nat) (h : n = 2 * 7^2 + 1 * 7^1 + 5 * 7^0) : 
    n = 110 := 
by
  rw [h]
  norm_num
  sorry

end base7_to_base10_l72_72274


namespace xy_pairs_l72_72037

/-- Let x, y, and z be three numbers. Let absolute differences be defined
    as x1 = |x - y|, y1 = |y - z|, z1 = |z - x| proceeding recursively: 
    x_{k+1} = |x_k - y_k|, y_{k+1} = |y_k - z_k|, z_{k+1} = |z_k - x_k|.
    Given that eventually the sequence stabilizes and z = 1, x and y 
    can only be (0, 1) or (1, 0). -/
theorem xy_pairs (x y z : ℕ) (n : ℕ) (h : ∀ k ≤ n, let x_k := abs (x - y)
                                           let y_k := abs (y - z)
                                           let z_k := abs (z - x)
                                           let x_k1 := abs (x_k - y_k)
                                           let y_k1 := abs (y_k - z_k)
                                           let z_k1 := abs (z_k - x_k)
                                           in x_k1 = x ∧ y_k1 = y ∧ z_k1 = z)
                                           (hz : z = 1) :
                                           (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
sorry

end xy_pairs_l72_72037


namespace rows_of_seats_l72_72278

theorem rows_of_seats (students sections_per_row students_per_section : ℕ) (h1 : students_per_section = 2) (h2 : sections_per_row = 2) (h3 : students = 52) :
  (students / students_per_section / sections_per_row) = 13 :=
sorry

end rows_of_seats_l72_72278


namespace problem1_problem2_l72_72378

namespace ArithmeticSequence

-- Part (1)
theorem problem1 (a1 : ℚ) (d : ℚ) (S_n : ℚ) (n : ℕ) (a_n : ℚ) 
  (h1 : a1 = 5 / 6) 
  (h2 : d = -1 / 6) 
  (h3 : S_n = -5) 
  (h4 : S_n = n * (2 * a1 + (n - 1) * d) / 2) 
  (h5 : a_n = a1 + (n - 1) * d) : 
  (n = 15) ∧ (a_n = -3 / 2) :=
sorry

-- Part (2)
theorem problem2 (d : ℚ) (n : ℕ) (a_n : ℚ) (a1 : ℚ) (S_n : ℚ)
  (h1 : d = 2) 
  (h2 : n = 15) 
  (h3 : a_n = -10) 
  (h4 : a_n = a1 + (n - 1) * d) 
  (h5 : S_n = n * (2 * a1 + (n - 1) * d) / 2) : 
  (a1 = -38) ∧ (S_n = -360) :=
sorry

end ArithmeticSequence

end problem1_problem2_l72_72378


namespace find_quadratic_fn_find_number_of_intersections_l72_72085

noncomputable def quadratic_fn (a c : ℝ) := λ x : ℝ, a * x ^ 2 + c

open Function

-- Given conditions
def conditions (a c : ℝ) :=
  quadratic_fn a c 0 = -2 ∧ quadratic_fn a c 1 = -1 ∧ Symmetric (quadratic_fn a c : ℝ → ℝ) (0 : ℝ)

-- Correct answers
def answer_fn := quadratic_fn 1 (-2)
def number_of_intersections := 2
def intersection_points := [Real.sqrt 2, -Real.sqrt 2]

-- The proof problem statements
theorem find_quadratic_fn (a c : ℝ) (h : conditions a c) :
  quadratic_fn a c = answer_fn :=
sorry

theorem find_number_of_intersections :
  length (filter (λ x, answer_fn x = 0) intersection_points) = number_of_intersections :=
sorry

end find_quadratic_fn_find_number_of_intersections_l72_72085


namespace initial_calculated_average_l72_72162

theorem initial_calculated_average (S : ℕ) (A_correct : ℕ) (A_correct = 15) (S + 36 = 150) :
  (S + 26) / 10 = 14 :=
by
  sorry

end initial_calculated_average_l72_72162


namespace amount_needed_for_free_delivery_l72_72964

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end amount_needed_for_free_delivery_l72_72964


namespace work_completion_days_l72_72996

theorem work_completion_days (T : ℕ) (hT : T = 8) : 
  (∀ (a b : ℝ), (a = 12) → (b = 24) → (1 / a + 1 / b) = 1 / T) :=
begin
  intros a b ha hb,
  rw [ha, hb],
  calc
    (1 / 12) + (1 / 24) = (2 / 24) + (1 / 24) : by linarith
    ... = 3 / 24 : by simp
    ... = 1 / 8 : by norm_num,
  exact hT.symm,
end

end work_completion_days_l72_72996


namespace q_const_term_l72_72472

def polynomial (R : Type*) := R → R

variables {R : Type*} [CommRing R]
variable {p q r : polynomial R}

-- Given conditions
axiom p_const_term : ∀ (x : R), p(x) = x + 5
axiom r_eq_p_mul_q : ∀ (x : R), r(x) = p(x) * q(x)
axiom r_const_term : ∀ (x : R), r(0) = -10

-- Theorem statement to prove
theorem q_const_term : q(0) = -2 :=
  sorry

end q_const_term_l72_72472


namespace find_angle_A_find_range_of_f_l72_72002

-- Definition of an acute triangle
structure AcuteTriangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (sides_angles : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (acute_angles : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (law_of_sines : a / sin A = b / sin B = c / sin C)

-- Given conditions for Problem (I)
def given_condition (t : AcuteTriangle) : Prop :=
  (t.a + t.b) * (sin t.A - sin t.B) = (t.c - t.b) * sin t.C

-- (I) Prove that angle A = π / 3
theorem find_angle_A (t : AcuteTriangle) (hc : given_condition t) : t.A = π / 3 :=
sorry

-- Function definition for problem (II)
def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (x / 2) * cos (x / 2) + cos x ^ 2

-- (II) Prove the range of f(B)
theorem find_range_of_f (t : AcuteTriangle) (hB : π/6 < t.B ∧ t.B < π/2) : 
  ∃ I : Set ℝ, I = (λ x : ℝ, (1 + sqrt 3) / 2 < x ∧ x <= 3 / 2) ∧ 
  (∀ b ∈ (λ x : ℝ, π/6 < x ∧ x < π/2), f b ∈ I) :=
sorry

end find_angle_A_find_range_of_f_l72_72002


namespace PQ_perpendicular_BC_l72_72912

-- Declare the variables and types
variables {A B C D E M N F P Q : Type}
variables {AB CD MF NF BP CP AQ DQ : ℝ}
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space E] [metric_space M]
variables [metric_space N] [metric_space F] [metric_space P]
variables [metric_space Q]

-- Define midpoints and intersections
def is_midpoint (M : Type) (A B : Type) : Prop := (distance M A = distance M B)
def diagonals_intersect (E : Type) (AC BD : Type) : Prop := 
  ∃ (E : Type), E ∈ AC ∧ E ∈ BD
def perp_bisector_meet (F : Type) (AB CD : Type) : Prop :=
  ∃ (F : Type), is_midpoint F AB ∧ is_midpoint F CD
def line_meets (X Y Z : Type) (L : Type) : Prop :=
  ∃ (P : Type), P ∈ line_segment X Y ∧ P ∈ line_segment Y Z

-- Given conditions
axiom givens (MF CD NF AB DQ BP AQ CP : ℝ) :
  MF * CD = NF * AB ∧ DQ * BP = AQ * CP

-- Hypothesis for the diagonals of the quadrilateral
axiom diagonal_condition (E : Type) (AC BD : Type) : diagonals_intersect E AC BD

-- Hypothesis for the midpoints of AB and CD
axiom midpoint_M (M : Type) : is_midpoint M A B
axiom midpoint_N (N : Type) : is_midpoint N C D

-- Hypothesis for the perpendicular bisectors meeting at F
axiom bisectors_meet_F (F : Type) : perp_bisector_meet F AB CD

-- Hypothesis for EF intersecting BC and AD at P and Q
axiom EF_intersect_PQ (EF BC AD : Type) : 
  line_meets E F P ∧ line_meets E F Q

-- The Main Theorem Statement
theorem PQ_perpendicular_BC (PQ : Type) : 
  PQ ⊥ BC :=
sorry -- Proof is omitted as instructed.

end PQ_perpendicular_BC_l72_72912


namespace hyperbola_eccentricity_proof_l72_72761

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (focus : ℝ) (hfocus : focus = real.sqrt (a^2 + b^2)) : ℝ :=
let c : ℝ := focus in
let symmetric_point := (2 * c - 7 * a, 0) in
if (symmetric_point.fst^2 / a^2) = 1 then
  let e : ℝ := real.sqrt (1 + b^2 / a^2) in
  e
else
  e

theorem hyperbola_eccentricity_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : ∃ (e : ℝ), e = 3 ∨ e = 4 :=
begin
  let focus := real.sqrt (a^2 + b^2),
  have hfocus : focus = real.sqrt (a^2 + b^2) := rfl,
  let e := hyperbola_eccentricity a b ha hb focus hfocus,
  sorry
end

end hyperbola_eccentricity_proof_l72_72761


namespace isosceles_triangle_altitude_l72_72144

theorem isosceles_triangle_altitude
  (a : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (triangle : isosceles ∧ acute (angle_at_vertex A))
  (AM : ℝ := altitude A a) :
  AM = a / (Complex.tan α) :=
  sorry

end isosceles_triangle_altitude_l72_72144


namespace largest_base_digit_sum_not_equal_9_l72_72623

theorem largest_base_digit_sum_not_equal_9 :
  ∃ (b : ℕ), (∀ n : ℕ, 1 ≤ n ∧ n ≤ b → (12^3 : ℕ).digits n.sum_digits ≠ 3^2) ∧ b = 9 :=
begin
  sorry
end

end largest_base_digit_sum_not_equal_9_l72_72623


namespace sum_of_squares_mod_13_l72_72615

theorem sum_of_squares_mod_13 :
  (∑ k in Finset.range 16, k^2) % 13 = 10 :=
by
  sorry

end sum_of_squares_mod_13_l72_72615


namespace sum_squares_mod_eq_6_l72_72598

def squares_mod (n : ℕ) : ℕ :=
  (List.range n).map (λ x => (x.succ * x.succ % 13))

noncomputable def sum_squares_mod : ℕ :=
  (squares_mod 15).sum 

theorem sum_squares_mod_eq_6 : sum_squares_mod % 13 = 6 := 
by
  sorry

end sum_squares_mod_eq_6_l72_72598


namespace initial_trucks_l72_72151

def trucks_given_to_Jeff : ℕ := 13
def trucks_left_with_Sarah : ℕ := 38

theorem initial_trucks (initial_trucks_count : ℕ) :
  initial_trucks_count = trucks_given_to_Jeff + trucks_left_with_Sarah → initial_trucks_count = 51 :=
by
  sorry

end initial_trucks_l72_72151


namespace shaded_square_area_l72_72918

noncomputable def Pythagorean_area (a b c : ℕ) (area_a area_b area_c : ℕ) : Prop :=
  area_a = a^2 ∧ area_b = b^2 ∧ area_c = c^2 ∧ a^2 + b^2 = c^2

theorem shaded_square_area 
  (area1 area2 area3 : ℕ)
  (area_unmarked : ℕ)
  (h1 : area1 = 5)
  (h2 : area2 = 8)
  (h3 : area3 = 32)
  (h_unmarked: area_unmarked = area2 + area3)
  (h_shaded : area1 + area_unmarked = 45) :
  area1 + area_unmarked = 45 :=
by
  exact h_shaded

end shaded_square_area_l72_72918


namespace smallest_n_interval_not_contains_quadratic_number_l72_72640

noncomputable def is_quadratic_number (x : ℝ) : Prop :=
  ∃ (a b c : ℤ), abs a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
                 abs b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
                 abs c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
                 x^2 * (a : ℝ) + x * (b : ℝ) + (c : ℝ) = 0

theorem smallest_n_interval_not_contains_quadratic_number :
  ∃ n : ℕ, 0 < n ∧ n = 11 ∧ 
  (∀ x : ℝ, is_quadratic_number x → ¬((n - 1 / 3 < x ∧ x < n) ∨ (n < x ∧ x < n + 1 / 3))) :=
begin
  sorry
end

end smallest_n_interval_not_contains_quadratic_number_l72_72640


namespace tenth_term_is_correct_l72_72976

-- Define the conditions
def first_term : ℚ := 3
def last_term : ℚ := 88
def num_terms : ℕ := 30
def common_difference : ℚ := (last_term - first_term) / (num_terms - 1)

-- Define the function for the n-th term of the arithmetic sequence
def nth_term (n : ℕ) : ℚ := first_term + (n - 1) * common_difference

-- Prove that the 10th term is 852/29
theorem tenth_term_is_correct : nth_term 10 = 852 / 29 := 
by 
  -- Add the proof later, the statement includes the setup and conditions
  sorry

end tenth_term_is_correct_l72_72976


namespace geometric_loci_l72_72392

noncomputable def quadratic_discriminant (x y : ℝ) : ℝ :=
  x^2 + 4 * y^2 - 4

-- Conditions:
def real_and_distinct (x y : ℝ) := 
  ((x^2) / 4 + y^2 > 1) 

def equal_and_real (x y : ℝ) := 
  ((x^2) / 4 + y^2 = 1) 

def complex_roots (x y : ℝ) := 
  ((x^2) / 4 + y^2 < 1)

def both_roots_positive (x y : ℝ) := 
  (x < 0) ∧ (-1 < y) ∧ (y < 1)

def both_roots_negative (x y : ℝ) := 
  (x > 0) ∧ (-1 < y) ∧ (y < 1)

def opposite_sign_roots (x y : ℝ) := 
  (y > 1) ∨ (y < -1)

theorem geometric_loci (x y : ℝ) :
  (real_and_distinct x y ∨ equal_and_real x y ∨ complex_roots x y) ∧ 
  ((real_and_distinct x y ∧ both_roots_positive x y) ∨
   (real_and_distinct x y ∧ both_roots_negative x y) ∨
   (real_and_distinct x y ∧ opposite_sign_roots x y)) := 
sorry

end geometric_loci_l72_72392


namespace joshua_finishes_after_malcolm_l72_72499

def time_difference_between_runners
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (malcolm_finish_time : ℕ := malcolm_speed * race_length)
  (joshua_finish_time : ℕ := joshua_speed * race_length) : ℕ :=
joshua_finish_time - malcolm_finish_time

theorem joshua_finishes_after_malcolm
  (race_length : ℕ)
  (malcolm_speed : ℕ)
  (joshua_speed : ℕ)
  (h_race_length : race_length = 12)
  (h_malcolm_speed : malcolm_speed = 7)
  (h_joshua_speed : joshua_speed = 9) : time_difference_between_runners race_length malcolm_speed joshua_speed = 24 :=
by 
  subst h_race_length
  subst h_malcolm_speed
  subst h_joshua_speed
  rfl

#print joshua_finishes_after_malcolm

end joshua_finishes_after_malcolm_l72_72499


namespace quadratics_root_k_value_l72_72755

theorem quadratics_root_k_value :
  (∀ k : ℝ, (∀ x : ℝ, x^2 + k * x + 6 = 0 → (x = 2 ∨ ∃ x1 : ℝ, x1 * 2 = 6 ∧ x1 + 2 = k)) → 
  (x = 2 → ∃ x1 : ℝ, x1 = 3 ∧ k = -5)) := 
sorry

end quadratics_root_k_value_l72_72755


namespace Owen_profit_l72_72874

/-- 
Owen bought 12 boxes of face masks, each box costing $9 and containing 50 masks. 
He repacked 6 boxes into smaller packs sold for $5 per 25 masks and sold the remaining masks in baggies of 10 pieces for $3 each.
Prove that Owen's profit amounts to $42.
 -/
theorem Owen_profit :
  let box_count := 12
  let cost_per_box := 9
  let masks_per_box := 50
  let repacked_boxes := 6
  let repack_price := 5
  let repack_size := 25
  let baggy_price := 3
  let baggy_size := 10 in
  let total_cost := box_count * cost_per_box in
  let total_masks := box_count * masks_per_box in
  let masks_repacked := repacked_boxes * masks_per_box in
  let repacked_revenue := (masks_repacked / repack_size) * repack_price in
  let remaining_masks := total_masks - masks_repacked in
  let baggy_revenue := (remaining_masks / baggy_size) * baggy_price in
  let total_revenue := repacked_revenue + baggy_revenue in
  let profit := total_revenue - total_cost in
  profit = 42 := by
  sorry

end Owen_profit_l72_72874


namespace students_difference_l72_72093

theorem students_difference :
  let lower_grades := 325
  let middle_upper_grades := 4 * lower_grades
  in middle_upper_grades - lower_grades = 975 := by
  sorry

end students_difference_l72_72093


namespace smallest_positive_q_l72_72318

theorem smallest_positive_q :
  ∃ (q : ℕ), (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 → ∃ (n : ℤ), 
    (m : ℚ) / 1007 * q < n ∧ n < (m + 1) / 1008 * q) ∧ 
    (∀ (q' : ℕ), (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1006 → ∃ (n : ℤ), 
      (m : ℚ) / 1007 * q' < n ∧ n < (m + 1) / 1008 * q') → q ≤ q')) :=
sorry

end smallest_positive_q_l72_72318


namespace coefficient_x_term_expansion_l72_72909

theorem coefficient_x_term_expansion :
  let f := λ x : ℝ, (x^2 - x + 1)
  coeff (expand (f x) 3) 1 = -3 :=
by
  sorry

end coefficient_x_term_expansion_l72_72909


namespace proof_of_truth_values_l72_72941

open Classical

variables (x : ℝ)

-- Original proposition: If x = 1, then x^2 = 1.
def original_proposition : Prop := (x = 1) → (x^2 = 1)

-- Converse of the original proposition: If x^2 = 1, then x = 1.
def converse_proposition : Prop := (x^2 = 1) → (x = 1)

-- Inverse of the original proposition: If x ≠ 1, then x^2 ≠ 1.
def inverse_proposition : Prop := (x ≠ 1) → (x^2 ≠ 1)

-- Contrapositive of the original proposition: If x^2 ≠ 1, then x ≠ 1.
def contrapositive_proposition : Prop := (x^2 ≠ 1) → (x ≠ 1)

-- Negation of the original proposition: If x = 1, then x^2 ≠ 1.
def negation_proposition : Prop := (x = 1) → (x^2 ≠ 1)

theorem proof_of_truth_values :
  (original_proposition x) ∧
  (converse_proposition x = False) ∧
  (inverse_proposition x = False) ∧
  (contrapositive_proposition x) ∧
  (negation_proposition x = False) := by
  sorry

end proof_of_truth_values_l72_72941


namespace lottery_probability_l72_72926

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem lottery_probability : 
  let MegaBall_prob : ℚ := 1 / 27
  let WinnerBall_prob : ℚ := 1 / (combination 44 5)
  let Total_prob : ℚ := MegaBall_prob * WinnerBall_prob
  in Total_prob = 1 / 29322216 :=
sorry

end lottery_probability_l72_72926


namespace oblique_coordinates_vector_properties_l72_72772

variable {θ : ℝ}
variable {λ x1 y1 x2 y2 : ℝ}

-- Conditions
axiom unit_vectors : ∀ (e1 e2 : ℝ), e1 * e1 = 1 ∧ e2 * e2 = 1 
axiom non_right_angle : θ ≠ π / 2

-- Definitions in oblique coordinates
def oblique_vector (e1 e2 : ℝ) (x y : ℝ) : ℝ := x * e1 + y * e2
def vector_sub (a1 a2 b1 b2 : ℝ) : ℝ × ℝ := (a1 - b1, a2 - b2)
def scalar_mul (λ x y : ℝ) : ℝ × ℝ := (λ * x, λ * y)

-- Proof Problem Statement
theorem oblique_coordinates_vector_properties (e1 e2 : ℝ) (h1 : e1 * e1 = 1) (h2 : e2 * e2 = 1) (h3 : θ ≠ π / 2) :
  vector_sub (x1 * e1) (y1 * e2) (x2 * e1) (y2 * e2) = (x1 - x2, y1 - y2)
  ∧ scalar_mul λ x1 y1 = (λ * x1, λ * y1) :=
by sorry

end oblique_coordinates_vector_properties_l72_72772


namespace sum_n_10_terms_progression_l72_72947

noncomputable def sum_arith_progression (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_n_10_terms_progression :
  ∃ (a : ℕ), (∃ (n : ℕ), sum_arith_progression n a 3 = 220) ∧
  (2 * a + (10 - 1) * 3) = 43 ∧
  sum_arith_progression 10 a 3 = 215 :=
by sorry

end sum_n_10_terms_progression_l72_72947


namespace distance_between_bakery_and_butcher_shop_l72_72292

variables (v1 v2 : ℝ) -- speeds of the butcher's and baker's son respectively
variables (x : ℝ) -- distance covered by the baker's son by the time they meet
variable (distance : ℝ) -- distance between the bakery and the butcher shop

-- Given conditions
def butcher_walks_500_more := x + 0.5
def butcher_time_left := 10 / 60
def baker_time_left := 22.5 / 60

-- Equivalent relationships
def v1_def := v1 = 6 * x
def v2_def := v2 = (8/3) * (x + 0.5)

-- Final proof problem
theorem distance_between_bakery_and_butcher_shop :
  (x + 0.5 + x) = 2.5 :=
sorry

end distance_between_bakery_and_butcher_shop_l72_72292


namespace bees_directions_at_15_feet_l72_72584

-- Define starting positions for Bee A and Bee B
structure Position :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)

def start : Position := ⟨0, 0, 0⟩

-- Define the movement patterns for Bee A and Bee B
def beeA_moves : List Position :=
  [⟨0, 1, 0⟩, ⟨2, 1, 0⟩, ⟨2, 1, 1⟩, ⟨2, 0, 1⟩]

def beeB_moves : List Position :=
  [⟨0, -1, 0⟩, ⟨-2, -1, 0⟩, ⟨-2, -1, -1⟩]

-- Function to calculate the position of the bee after n cycles for a given movement pattern
def position_after_n_cycles (moves : List Position) (n : ℕ) : Position :=
  let cycle_length := moves.length
  let cycle_pos := n % cycle_length
  let full_cycles := n / cycle_length
  let full_moves_sum := moves.sum
  let partial_moves_sum := (moves.take cycle_pos).sum
  ⟨full_cycles * full_moves_sum.x + partial_moves_sum.x,
   full_cycles * full_moves_sum.y + partial_moves_sum.y,
   full_cycles * full_moves_sum.z + partial_moves_sum.z⟩

-- Function to calculate Euclidean distance squared
def distance_squared (p1 p2 : Position) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

-- Lean statement for the problem
theorem bees_directions_at_15_feet : 
  ∃ nA nB : ℕ, 
  distance_squared (position_after_n_cycles beeA_moves nA) (position_after_n_cycles beeB_moves nB) = 225 
  ∧ (nA % beeA_moves.length = 1) 
  ∧ (nB % beeB_moves.length = 1) := 
sorry

end bees_directions_at_15_feet_l72_72584


namespace james_dancing_calories_l72_72091

def walking_calories_per_hour : ℕ := 300
def sessions_per_day : ℕ := 2
def hours_per_session : ℝ := 0.5
def days_per_week : ℕ := 4

def dancing_calories_per_hour : ℕ := 2 * walking_calories_per_hour
def calories_per_session : ℝ := dancing_calories_per_hour * hours_per_session
def calories_per_day : ℝ := calories_per_session * sessions_per_day
def calories_per_week : ℝ := calories_per_day * days_per_week

theorem james_dancing_calories : calories_per_week = 2400 := 
by 
  rw [calories_per_week, calories_per_day, calories_per_session, dancing_calories_per_hour],
  simp,
  norm_num,
  sorry

end james_dancing_calories_l72_72091


namespace syrup_cost_per_week_l72_72261

theorem syrup_cost_per_week (gallons_per_week : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) 
  (h1 : gallons_per_week = 180) 
  (h2 : gallons_per_box = 30) 
  (h3 : cost_per_box = 40) : 
  (gallons_per_week / gallons_per_box) * cost_per_box = 240 := 
by
  sorry

end syrup_cost_per_week_l72_72261


namespace teddy_bear_cost_l72_72855

-- Definitions for the given conditions
def num_toys : ℕ := 28
def toy_price : ℕ := 10
def num_teddy_bears : ℕ := 20
def total_money : ℕ := 580

-- The theorem we want to prove
theorem teddy_bear_cost :
  (num_teddy_bears * 15 + num_toys * toy_price = total_money) :=
by
  sorry

end teddy_bear_cost_l72_72855


namespace cupcake_combinations_l72_72677

/-- 
Bill needs to purchase exactly seven cupcakes, and the bakery has five types of cupcakes. 
Bill is required to get at least one of each of the first four types. 
We need to prove that the number of ways for Bill to complete his order is 35.
-/
theorem cupcake_combinations : 
  (∑ x in finset.Ico 4 8, (finset.Ico 4 8).choose x) = 35 := by
begin
  sorry
end

end cupcake_combinations_l72_72677


namespace part1_part2_l72_72484

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part1 (x : ℝ) : (f x 2 ≥ 7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) : a = 1 :=
by sorry

end part1_part2_l72_72484


namespace nth_term_206_l72_72441

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 10 ∧ a 1 = -10 ∧ ∀ n, a (n + 2) = -a n

theorem nth_term_206 (a : ℕ → ℝ) (h : geometric_sequence a) : a 205 = -10 :=
by
  -- Utilizing the sequence property to determine the 206th term
  sorry

end nth_term_206_l72_72441


namespace roja_alone_time_l72_72498

theorem roja_alone_time (W : ℝ) (R : ℝ) :
  (1 / 60 + 1 / R = 1 / 35) → (R = 210) :=
by
  intros
  -- Proof goes here
  sorry

end roja_alone_time_l72_72498


namespace original_cookies_l72_72320

noncomputable def initial_cookies (final_cookies : ℝ) (ratio : ℝ) (days : ℕ) : ℝ :=
  final_cookies / ratio^days

theorem original_cookies :
  ∀ (final_cookies : ℝ) (ratio : ℝ) (days : ℕ),
  final_cookies = 28 →
  ratio = 0.7 →
  days = 3 →
  initial_cookies final_cookies ratio days = 82 :=
by
  intros final_cookies ratio days h_final h_ratio h_days
  rw [initial_cookies, h_final, h_ratio, h_days]
  norm_num
  sorry

end original_cookies_l72_72320


namespace estimated_total_fish_l72_72226

-- Let's define the conditions first
def total_fish_marked := 100
def second_catch_total := 200
def marked_in_second_catch := 5

-- The variable representing the total number of fish in the pond
variable (x : ℕ)

-- The theorem stating that given the conditions, the total number of fish is 4000
theorem estimated_total_fish
  (h1 : total_fish_marked = 100)
  (h2 : second_catch_total = 200)
  (h3 : marked_in_second_catch = 5)
  (h4 : (marked_in_second_catch : ℝ) / second_catch_total = (total_fish_marked : ℝ) / x) :
  x = 4000 := 
sorry

end estimated_total_fish_l72_72226


namespace perimeter_triangle_gt_twice_oc_l72_72880

theorem perimeter_triangle_gt_twice_oc
  {O X Y C A B : Point}
  (hC_in_XOY : lies_within_angle O X Y C)
  (hA_on_OX : lies_on_ray O X A)
  (hB_on_OY : lies_on_ray O Y B)
  : perimeter (triangle A B C) > 2 * distance O C :=
sorry

end perimeter_triangle_gt_twice_oc_l72_72880


namespace maximum_angles_lt_150_l72_72978

open_locale classical

variable (n : ℕ)

-- Define the sum of interior angles for a simple polygon with n sides.
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Define a simple polygon with 2017 sides where all angles are less than 180 degrees.
structure simple_polygon :=
  (sides : ℕ)
  (angles : finset ℝ)
  (angles_size : angles.card = sides)
  (angle_bound : ∀ θ ∈ angles, θ < 180)

-- Define the specific polygon we're interested in.
def polygon_2017 : simple_polygon :=
  { sides := 2017,
    angles := finset.range 2017.map (λ i, _), -- Place-holder for range of angles
    angles_size := by sorry, -- omitted for brevity but should assert card = 2017,
    angle_bound := by sorry } -- omitted for brevity but should assert ∀ θ, θ < 180

-- Statement of the problem
theorem maximum_angles_lt_150 (p : simple_polygon) (h₁ : p.sides = 2017) (h₂ : ∀ θ ∈ p.angles, θ < 180) : 
  ∃ k, (∀ θ ∈ p.angles, θ ≤ 150 → θ.card = k) ∧ k ≤ 12 :=
begin
  sorry -- Proof omitted
end

end maximum_angles_lt_150_l72_72978


namespace triangle_side_relationship_l72_72832

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry

theorem triangle_side_relationship 
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = 40 * Real.pi / 180)
  (hβ : β = 60 * Real.pi / 180)
  (hγ : γ = 80 * Real.pi / 180)
  (h_angle_sum : α + β + γ = Real.pi) : 
  a * (a + b + c) = b * (b + c) :=
sorry

end triangle_side_relationship_l72_72832


namespace original_cost_of_dolls_l72_72311

theorem original_cost_of_dolls 
  (x : ℝ) -- original cost of each Russian doll
  (savings : ℝ) -- total savings of Daniel
  (h1 : savings = 15 * x) -- Daniel saves enough to buy 15 dolls at original price
  (h2 : savings = 20 * 3) -- with discounted price, he can buy 20 dolls
  : x = 4 :=
by
  sorry

end original_cost_of_dolls_l72_72311


namespace duty_frequency_not_possible_l72_72803

theorem duty_frequency_not_possible : 
  ¬ ∃ (G : Type), 
  (∃ (V: List (Fin 10)) (E: List (Fin 10 × Fin 10)), 
    (∀ (x ∈ V), (∃ (y ∈ V), (x, y) ∈ E) ∧ (x ≠ y)) ∧
    (∀ (x y : Fin 10), (x, y) ∈ E → (y, x) ∉ E) ∧
    (length (filter (λ (v : Fin 10), v = 9 ∨ v = 8 ∨ v = 5 ∨ v = 3 ∨ v = 1) 
    (map (λ (v : Fin 10), List.length (filter (λ (e : Fin 10 × Fin 10), 
      e.fst = v ∨ e.snd = v) E)) V)) = 10)) := 
  sorry

end duty_frequency_not_possible_l72_72803


namespace isosceles_triangle_sides_l72_72811

theorem isosceles_triangle_sides (m n : ℝ) (h1 : is_isosceles m n) :
  ∃ x y : ℝ, 
    x = 2 * m ^ 2 / Real.sqrt (4 * m ^ 2 - n ^ 2) ∧ 
    y = 2 * m * n / Real.sqrt (4 * m ^ 2 - n ^ 2) :=
sorry

def is_isosceles (m n : ℝ) : Prop :=
  ∃ (x y : ℝ),
    x = 2 * m ^ 2 / Real.sqrt (4 * m ^ 2 - n ^ 2) ∧
    y = 2 * m * n / Real.sqrt (4 * m ^ 2 - n ^ 2)

end isosceles_triangle_sides_l72_72811


namespace highest_throw_christine_janice_l72_72680

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l72_72680


namespace integral_solution_l72_72046

noncomputable def integral_eq_zero : Prop :=
  ∃ k : ℝ, (∫ (2 * x - 3 * x^2) in set.Icc 0 k, 1) = 0 ∧ (k = 0 ∨ k = 1)

theorem integral_solution : integral_eq_zero :=
sorry

end integral_solution_l72_72046


namespace distinct_values_of_g_l72_72691

def g (x : ℝ) : ℝ := ∑ k in finset.range (15 - 1 - 1 + 1), ⌊(k + 2) * fract x⌋

-- Helper function to calculate Euler's Totient Function.
noncomputable def phi (n : ℕ) : ℕ := (finset.range n).filter (nat.coprime n).card

-- Sum of the Euler's Totient Function values from 2 to 15
def sum_euler_totient : ℕ := ∑ k in finset.range (15 - 2 + 1), phi (k + 2)

theorem distinct_values_of_g : sum_euler_totient + 1 = 72 := by
  sorry

end distinct_values_of_g_l72_72691


namespace div_sum_lt_n_squared_and_divides_iff_prime_l72_72176

theorem div_sum_lt_n_squared_and_divides_iff_prime {n : ℕ} (hn : n > 1)
  (divisors : List ℕ) (hdiv : ∀ d ∈ divisors, d ∣ n) (sorted_div : divisors = divisors.sorted)
  (d1 : divisors.head = 1) (dk : divisors.getLast ((List.headI divisors).get <| λ _ => 0) = n) :
  let d := (List.map (λ ⟨d1, d2⟩, d1 * d2)
                          (List.zip (divisors.init) (divisors.tail))).sum in
  (d < n^2) ∧ (d ∣ n^2 ↔ Nat.Prime n) := sorry

end div_sum_lt_n_squared_and_divides_iff_prime_l72_72176


namespace mirasol_balance_l72_72132

/-- Given Mirasol initially had $50, spends $10 on coffee beans, and $30 on a tumbler,
    prove that the remaining balance in her account is $10. -/
theorem mirasol_balance (initial_balance spent_coffee spent_tumbler remaining_balance : ℕ)
  (h1 : initial_balance = 50)
  (h2 : spent_coffee = 10)
  (h3 : spent_tumbler = 30)
  (h4 : remaining_balance = initial_balance - (spent_coffee + spent_tumbler)) :
  remaining_balance = 10 :=
sorry

end mirasol_balance_l72_72132


namespace highest_score_l72_72630

theorem highest_score
  (avg : ℕ): avg = 60 →
  (innings : ℕ): innings = 46 →
  (diff : ℕ ): diff = 150 →
  (excluded_avg : ℕ): excluded_avg = 58 →
  (H L : ℕ): (H - L = diff) →
  (total_runs : ℕ): total_runs = avg * innings →
  (excluded_total_runs : ℕ): excluded_total_runs = excluded_avg * (innings - 2) →
  ((H + L = total_runs - excluded_total_runs) →
  H = 179) :=
by
  intros
  sorry

end highest_score_l72_72630


namespace number_of_valid_arrangements_is_68_l72_72142

-- Definitions of the constraints.
def Books : Type := {classics : List String // classics.length = 5}
def valid_book_order (order : List String) : Prop :=
  order.length = 4 ∧
  (order.nodup) ∧
  ("Book of Songs" ∈ order) ∧
  ("Book of Documents" ∈ order) ∧
  ("Book of Rites" ∈ order) ∧
  ("Book of Changes" ∈ order) ∧
  ("Spring and Autumn Annals" ∈ order) ∧
  ∀ (i : ℕ) (h : i < order.length - 1),
    ¬((order.nth_le i (by linarith) = "Book of Songs" ∧ order.nth_le (i + 1) (by linarith) = "Book of Rites") ∨
      (order.nth_le i (by linarith) = "Book of Rites" ∧ order.nth_le (i + 1) (by linarith) = "Book of Songs")) ∧
  ¬(order.head = some "Book of Changes")

-- The main statement to be proved.
theorem number_of_valid_arrangements_is_68 :
  ∃ (orders : List (List String)), (∀ o ∈ orders, valid_book_order o) ∧ orders.length = 68 :=
sorry

end number_of_valid_arrangements_is_68_l72_72142


namespace Joe_fair_spending_l72_72915

theorem Joe_fair_spending :
  let entrance_fee_under_18 := 5 in
  let entrance_fee_over_18 := 5 + (5 * 20 / 100) in
  let group_discount := 15 / 100 in
  let ride_cost := 0.5 in
  let num_people := 3 in
  let joe_age := 30 in
  let twins_age := 6 in
  let joe_rides := 4 in
  let twinA_rides := 3 in
  let twinB_rides := 5 in
  let total_entrance_fee_without_discount := entrance_fee_over_18 + (2 * entrance_fee_under_18) in
  let discount := total_entrance_fee_without_discount * group_discount in
  let total_entrance_fee_with_discount := total_entrance_fee_without_discount - discount in
  let total_ride_cost := (joe_rides * ride_cost) + (twinA_rides * ride_cost) + (twinB_rides * ride_cost) in
  let total_cost := total_entrance_fee_with_discount + total_ride_cost in
  total_cost = 19.60 := by
  sorry

end Joe_fair_spending_l72_72915


namespace circle_radius_l72_72943

noncomputable def radius_of_circle : ℝ :=
  let center_x := -29/16 in
  let point1 : ℝ × ℝ := (2, 3) in
  let center : ℝ × ℝ := (center_x, 0) in
  real.sqrt ((center.1 - point1.1)^2 + (center.2 - point1.2)^2)

theorem circle_radius :
  (radius_of_circle = 65 / 16) :=
by 
    -- Calculate distance step and simplify
    let center_x := -29 / 16
    let point1 : ℝ × ℝ := (2, 3)
    have center : ℝ × ℝ := (center_x, 0)
    have radius := real.sqrt ((center.1 - point1.1) ^ 2 + (center.2 - point1.2) ^ 2)
    show radius = 65 / 16, by sorry

end circle_radius_l72_72943


namespace Matthew_initial_cakes_l72_72505

theorem Matthew_initial_cakes (n_cakes : ℕ) (n_crackers : ℕ) (n_friends : ℕ) (crackers_per_person : ℕ) :
  n_friends = 4 →
  n_crackers = 32 →
  crackers_per_person = 8 →
  n_crackers = n_friends * crackers_per_person →
  n_cakes = n_friends * crackers_per_person →
  n_cakes = 32 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1, h3] at h5
  exact h5

end Matthew_initial_cakes_l72_72505


namespace problem_statement_l72_72839

open Real

variable (e : ℝ)

def a : ℝ × ℝ × ℝ := (2, -7, 3)
def b : ℝ × ℝ × ℝ := (-4, e, 1)
def c : ℝ × ℝ × ℝ := (0, -3, 8)

def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def vector_dot (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem problem_statement : 
  vector_dot 
    (vector_sub a b) 
    (vector_cross (vector_sub b c) (vector_sub c a)) = 28 * e + 196 := by
  sorry

end problem_statement_l72_72839


namespace number_exceeds_percent_l72_72234

theorem number_exceeds_percent (x : ℝ) (h : x = 0.12 * x + 52.8) : x = 60 :=
by {
  sorry
}

end number_exceeds_percent_l72_72234


namespace find_x_l72_72730

noncomputable def A (S : List ℝ) : List ℝ :=
  (List.map (λ (x : ℝ × ℝ), (x.1 + x.2) / 2) (S.zip S.tail!)).tail!

noncomputable def A_iter (m : ℕ) (S : List ℝ) : List ℝ :=
  match m with
  | 0 => S
  | (n+1) => A (A_iter n S)

theorem find_x (x : ℝ) (hx : x > 0) (S : List ℝ) (hS : S = List.range 101 |>.map (λ n, x^n)) 
  (hA : A_iter 100 S = [1 / 2^50]) : x = Real.sqrt 2 - 1 :=
sorry

end find_x_l72_72730


namespace find_integers_l72_72694

def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem find_integers (x : ℤ) (h : isPerfectSquare (x^2 + 19 * x + 95)) : x = -14 ∨ x = -5 := by
  sorry

end find_integers_l72_72694


namespace additional_hours_due_to_leak_l72_72259

noncomputable def fill_rate_no_leak : ℝ := 1 / 14
noncomputable def leak_rate : ℝ := 1 / 112

theorem additional_hours_due_to_leak :
  let effective_fill_rate := fill_rate_no_leak - leak_rate in
  let time_fill_leak := 1 / effective_fill_rate in
  let additional_time := time_fill_leak - 14 in
  additional_time = 2 :=
by
  sorry

end additional_hours_due_to_leak_l72_72259


namespace parabola_intersection_radius_sqr_l72_72559

theorem parabola_intersection_radius_sqr {x y : ℝ} :
  (y = (x - 2)^2) →
  (x - 3 = (y + 2)^2) →
  ∃ r, r^2 = 9 / 2 :=
by
  intros h1 h2
  sorry

end parabola_intersection_radius_sqr_l72_72559


namespace relationship_y1_y2_y3_l72_72023

noncomputable def y (x : ℝ) : ℝ := -(x - 2) ^ 2

def A : ℝ × ℝ := (-1, y (-1))
def B : ℝ × ℝ := (1, y (1))
def C : ℝ × ℝ := (4, y (4))

theorem relationship_y1_y2_y3 :
  let y1 := y (-1)
  let y2 := y (1)
  let y3 := y (4)
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_y1_y2_y3_l72_72023


namespace period_tan_transformation_l72_72980

noncomputable def period_of_tan : ℝ :=
  π

noncomputable def period_of_transformed_tan (k : ℝ) : ℝ :=
  period_of_tan / k

theorem period_tan_transformation : 
  ∀ x : ℝ, ∀ k : ℝ, k = (3/4) → period_of_transformed_tan k = (4 * π / 3) :=
by
  intros x k h
  rw [h]
  unfold period_of_transformed_tan
  unfold period_of_tan
  rw [div_eq_mul_inv, mul_assoc, mul_comm _ (4 : ℝ), mul_inv_cancel]
  norm_num
  sorry

end period_tan_transformation_l72_72980


namespace expansion_constant_term_l72_72017

theorem expansion_constant_term (a x : ℝ) (hx_pos : 0 < x) (expansion_coeff_sum : (1 + a / x) * (2 * x - 1 / x) ^ 5 = 2) :
  let x_one := 1
  let a_one := a = 1
  let general_term := ∀ r : ℕ, r ≤ 5 → 
    rfl
    C(5,r) * (2 * x)^ (5 - r) * (-1 / x)^ r = (-1)^ r * 2^(5 - r) * C(5,r) * x^(5 - 2 * r)
  (2^3 * C(5, 2)) = 80 := 
sorry

end expansion_constant_term_l72_72017


namespace length_of_train_l72_72925

-- Define the conditions
def bridge_length : ℕ := 200
def train_crossing_time : ℕ := 60
def train_speed : ℕ := 5

-- Define the total distance traveled by the train while crossing the bridge
def total_distance : ℕ := train_speed * train_crossing_time

-- The problem is to show the length of the train
theorem length_of_train :
  total_distance - bridge_length = 100 :=
by sorry

end length_of_train_l72_72925


namespace find_fx_l72_72350

variable (f : ℝ → ℝ)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), 3 * f(x) + 2 * f(1 - x) = 4 * x

theorem find_fx (h : satisfies_condition f) : ∀ (x : ℝ), f(x) = 4 * x - 8 / 5 :=
by
  sorry

end find_fx_l72_72350


namespace systematic_sampling_counts_to_10_l72_72585

theorem systematic_sampling_counts_to_10 :
  ∀ (n : ℕ), (n ≥ 16 ∧ n ≤ 25) → 960 > 32 ∧ n ∈ ℤ → 451 ≤ 30 * n - 21 ∧ 30 * n - 21 ≤ 750 → n = 10 :=
by
  sorry

end systematic_sampling_counts_to_10_l72_72585


namespace solve_log_problem_l72_72706

theorem solve_log_problem (x : ℝ) : 9^(2 * log x / log 5) = 81 ↔ x = 5 := sorry

end solve_log_problem_l72_72706


namespace min_integral_achieved_at_3_minus_2_sqrt_2_l72_72480

-- Define the function and its integral
def F (a : ℝ) : ℝ := ∫ x in a..a^2, 1 / (x + sqrt x)

theorem min_integral_achieved_at_3_minus_2_sqrt_2 (a : ℝ) (ha : a > 0) :
  (∀ x > 0, F x ≥ F (3 - 2 * sqrt 2)) :=
sorry

end min_integral_achieved_at_3_minus_2_sqrt_2_l72_72480


namespace a_n_formula_T_n_formula_l72_72032

-- Defining the sequence Sn
def S (n : ℕ) : ℚ := (3 / 2) * n^2 - (123 / 2) * n

-- Defining the sequence an based on Sn
def a (n : ℕ) : ℚ := if n = 1 then S 1 else S n - S (n - 1)

-- Defining the absolute sum Tn
def T (n : ℕ) : ℚ :=
  if n ≤ 20 then
    (123 / 2) * n - (3 / 2) * n^2
  else
    (3 / 2) * n^2 - (123 / 2) * n + 1260

-- Proofs to be provided
theorem a_n_formula (n : ℕ) (hn : 0 < n) : a n = 3 * n - 63 :=
by sorry

theorem T_n_formula (n : ℕ) (hn : 0 < n) : 
  T n = if n ≤ 20 then 
          (123 / 2) * n - (3 / 2) * n^2 
        else 
          (3 / 2) * n^2 - (123 / 2) * n + 1260 :=
by sorry

end a_n_formula_T_n_formula_l72_72032


namespace height_of_model_l72_72522

noncomputable def original_monument_height : ℝ := 100
noncomputable def original_monument_radius : ℝ := 20
noncomputable def original_monument_volume : ℝ := 125600
noncomputable def model_volume : ℝ := 1.256

theorem height_of_model : original_monument_height / (original_monument_volume / model_volume)^(1/3) = 1 :=
by
  sorry

end height_of_model_l72_72522


namespace max_subsets_no_containment_l72_72628

theorem max_subsets_no_containment (n : ℕ) (h_pos : 0 < n) :
  ∃ A : Finset (Finset (Fin n)), 
    (∀ (a b ∈ A), a ⊆ b → a = b) ∧ 
    A.card = Nat.choose n (n / 2) :=
by sorry

end max_subsets_no_containment_l72_72628


namespace max_gcd_11n_3_6n_1_l72_72289

theorem max_gcd_11n_3_6n_1 : ∃ n : ℕ+, ∀ k : ℕ+,  11 * n + 3 = 7 * k + 1 ∧ 6 * n + 1 = 7 * k + 2 → ∃ d : ℕ, d = Nat.gcd (11 * n + 3) (6 * n + 1) ∧ d = 7 :=
by
  sorry

end max_gcd_11n_3_6n_1_l72_72289


namespace absolute_value_condition_l72_72054

theorem absolute_value_condition (x : ℝ) (h : |x| = 32) : x = 32 ∨ x = -32 :=
sorry

end absolute_value_condition_l72_72054


namespace midpoint_coordinates_l72_72065

-- Define the line and the parabola
def line (x y : ℝ) := x - y = 2
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define the midpoint function
def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) := 
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- Main theorem statement
theorem midpoint_coordinates : 
  ∃ A B : (ℝ × ℝ), (line A.fst A.snd) ∧ (parabola A.fst A.snd) ∧ 
                      (line B.fst B.snd) ∧ (parabola B.fst B.snd) ∧ 
                      midpoint A B = (4, 2) :=
by {
    sorry -- The proof details go here
}

end midpoint_coordinates_l72_72065


namespace tony_water_intake_l72_72197

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end tony_water_intake_l72_72197


namespace max_value_of_x_plus_y_plus_z_l72_72924

theorem max_value_of_x_plus_y_plus_z : ∀ (x y z : ℤ), (∃ k : ℤ, x = 5 * k ∧ 6 = y * k ∧ z = 2 * k) → x + y + z ≤ 43 :=
by
  intros x y z h
  rcases h with ⟨k, hx, hy, hz⟩
  sorry

end max_value_of_x_plus_y_plus_z_l72_72924


namespace angle_DAB_is_45_degrees_l72_72084

theorem angle_DAB_is_45_degrees 
  (A B C D E : Type)
  (triangle_ABC : Triangle A B C)
  (square_BCDE : Square B C D E)
  (isosceles : length CA = length CB)
  (angle_DAB : Angle D A B = 45) :
  angle_DAB = 45 := 
sorry

end angle_DAB_is_45_degrees_l72_72084


namespace all_stones_weigh_the_same_l72_72572

theorem all_stones_weigh_the_same (x : Fin 13 → ℕ)
  (h : ∀ (i : Fin 13), ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧
    i ∉ A ∧ i ∉ B ∧ ∀ (j k : Fin 13), j ∈ A → k ∈ B → x j = x k): 
  ∀ i j : Fin 13, x i = x j := 
sorry

end all_stones_weigh_the_same_l72_72572


namespace oblique_coordinates_properties_l72_72773

variables {α : Type*} [inner_product_space ℝ α]

def unit_vectors (e1 e2 : α) : Prop :=
  ∥e1∥ = 1 ∧ ∥e2∥ = 1 ∧ (⟪e1, e2⟫ / (∥e1∥ * ∥e2∥) ≠ 0)

variables (e1 e2 : α) (θ : ℝ) (h_unit_vectors : unit_vectors e1 e2)
          (x1 x2 y1 y2 : ℝ) (a b : α)

axiom vector_a_def : a = x1 • e1 + y1 • e2
axiom vector_b_def : b = x2 • e1 + y2 • e2

theorem oblique_coordinates_properties :
  (a - b = (x1 - x2) • e1 + (y1 - y2) • e2) ∧
  (∀ λ : ℝ, λ • a = λ * x1 • e1 + λ * y1 • e2) :=
sorry

end oblique_coordinates_properties_l72_72773


namespace coefficient_of_friction_l72_72968

-- Let P be the weight of the block
variable (P : Real)

-- Define the forces in terms of Real numbers
def F_up (F_down : Real) : Real := 3 * F_down
def F_sum (F_up F_down : Real) : Real := F_up + F_down

-- Define the coefficient of friction mu
def mu (P : Real) : Real := (1 / 2) * (1 / (Real.sqrt 3))

theorem coefficient_of_friction (F_down : Real) 
  (h1 : F_up F_down = 3 * F_down) 
  (h2 : F_sum (F_up F_down) F_down = P) :
  mu P = Real.sqrt 3 / 6 :=
by
  sorry

end coefficient_of_friction_l72_72968


namespace median_of_data_set_l72_72555

theorem median_of_data_set :
  let data := [5, 7, 5, 8, 6, 13, 5]
  let sorted_data := data.sorted
  let median := sorted_data.get! (sorted_data.length / 2)
  median = 6 := by
  sorry

end median_of_data_set_l72_72555


namespace maria_drove_578_miles_l72_72500

noncomputable def MariaDrivingDistance : ℕ :=
let gas_per_mile := 1 / 28
let full_tank_gallons := 16
let initial_distance := 420
let additional_gas := 10
let final_tank_fraction := 1 / 3 in 
let gas_used_first_leg := initial_distance * gas_per_mile in
let remaining_gas_after_first_leg := full_tank_gallons - gas_used_first_leg in
let total_gas_after_refuel := remaining_gas_after_first_leg + additional_gas in
let final_gas_remaining := full_tank_gallons * final_tank_fraction in
let gas_used_second_leg := total_gas_after_refuel - final_gas_remaining in
let distance_second_leg := gas_used_second_leg / gas_per_mile in
initial_distance + distance_second_leg

theorem maria_drove_578_miles : MariaDrivingDistance = 578 := sorry

end maria_drove_578_miles_l72_72500


namespace sum_squares_mod_13_l72_72616

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 5 :=
by sorry

end sum_squares_mod_13_l72_72616


namespace magnitude_a_plus_3b_parallel_condition_l72_72399

-- Define the vectors
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

-- Magnitude of a + 3b
theorem magnitude_a_plus_3b : 
  ‖(1 : ℝ, 0) + 3 • (2, 1)‖ = Real.sqrt 58 :=
by  sorry

-- Parallel condition for vectors
theorem parallel_condition (k : ℝ) : 
  (1 - 2 * k, -k) = (1 - 2 * 3, -3) ↔ k = 3 :=
by  sorry

end magnitude_a_plus_3b_parallel_condition_l72_72399


namespace monotonic_decreasing_interval_l72_72931

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l72_72931


namespace farmer_milk_production_l72_72250

theorem farmer_milk_production
  (total_cattle : ℕ)
  (male_cows : ℕ)
  (female_percentage : ℚ)
  (milk_per_day_per_female : ℕ)
  (males_to_total_ratio : ℚ)
  (total_cattle_eq : total_cattle = male_cows / males_to_total_ratio)
  (female_cows_eq : nat.cast (total_cattle) * female_percentage = nat.cast (total_cattle) - nat.cast (male_cows))
  (males_count : male_cows = 50)
  (f_perc : female_percentage = 0.60)
  (milk_per_female: milk_per_day_per_female = 2)
  (males_ratio : males_to_total_ratio = 0.40)
  : total_cattle * female_percentage * milk_per_day_per_female = 150 :=
by
  sorry

end farmer_milk_production_l72_72250


namespace probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l72_72825

def TianJi_top {α : Type} [LinearOrder α] (a1 a2 : α) (b1 : α) : Prop :=
  a2 < b1 ∧ b1 < a1

def TianJi_middle {α : Type} [LinearOrder α] (a3 a2 : α) (b2 : α) : Prop :=
  a3 < b2 ∧ b2 < a2

def TianJi_bottom {α : Type} [LinearOrder α] (a3 : α) (b3 : α) : Prop :=
  b3 < a3

def without_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning without knowing the order
  1 / 6

theorem probability_without_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  without_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 6 :=
sorry

def with_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning with specific group knowledge
  1 / 2

theorem probability_with_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  with_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 2 :=
sorry

end probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l72_72825


namespace notebook_problem_l72_72097

theorem notebook_problem :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 2 * x + 5 * y + 6 * z = 62 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x = 14 :=
by
  sorry

end notebook_problem_l72_72097


namespace amount_needed_for_free_delivery_l72_72963

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end amount_needed_for_free_delivery_l72_72963


namespace max_a_plus_b_l72_72428

open Classical
noncomputable theory

variable {V : Type} (G : SimpleGraph V)

/--
For a graph G, the chromatic number χ(G) is the minimum number of teams such that any two students in the same team are always friends.
-/
def chromaticNumber (G : SimpleGraph V) : ℕ := Classical.choice G.chromatic_number.exists_is_greedy_coloring

theorem max_a_plus_b (n : ℕ) (G : SimpleGraph (Fin n)) :
  (chromaticNumber G) + (chromaticNumber G.compl) <= n + 1 := sorry

end max_a_plus_b_l72_72428


namespace vector_c_representation_l72_72796

variable (a b c : Vector ℝ)

/-- Define the vectors a, b, and c --/
def vector_a := (1.0, 1.0 : ℝ × ℝ)
def vector_b := (1.0, -1.0 : ℝ × ℝ)
def vector_c := (-1.0, -2.0 : ℝ × ℝ)

/-- State the equivalence to be proven --/
theorem vector_c_representation : 
  vector_c = - (3 / 2 : ℝ) • (vector_a : ℝ × ℝ) + (1 / 2 : ℝ) • (vector_b : ℝ × ℝ) :=
sorry

end vector_c_representation_l72_72796


namespace other_girl_age_l72_72548

theorem other_girl_age (x : ℕ) (h1 : 13 + x = 27) : x = 14 := by
  sorry

end other_girl_age_l72_72548


namespace find_eccentricity_of_ellipse_l72_72748

theorem find_eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c = sqrt (a^2 - b^2)) (h4 : ∃ (M : ℝ × ℝ), M.1 = c ∧ M.2 = 2 / 3 * b) :
  eccentricity a b c = sqrt 5 / 3 :=
by
  sorry

end find_eccentricity_of_ellipse_l72_72748


namespace find_even_increasing_function_l72_72668

theorem find_even_increasing_function :
  ∃! (f : ℝ → ℝ), 
    (∀ x : ℝ, f x = |x|) ∨ (f x = x^3) ∨ (f x = x^2 + 2) ∨ (f x = -x^2) ∧ 
    (∀ x : ℝ, f (-x) = f x) ∧ 
    (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → f x1 < f x2) :=
begin
  sorry
end

end find_even_increasing_function_l72_72668


namespace exists_five_non_neg_reals_sum_to_one_min_product_ge_one_ninth_all_five_non_neg_reals_sum_to_one_max_product_le_one_ninth_l72_72638

-- Part 1: Existence of configuration with minimum product of adjacent pairs being at least 1/9.
theorem exists_five_non_neg_reals_sum_to_one_min_product_ge_one_ninth :
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ a + b + c + d + e = 1 ∧
  ∀ (perm : list ℝ), perm.perm [a, b, c, d, e] → ∃ (i : ℕ), 0 ≤ i ∧ i < 5 ∧
    (perm.nth i * perm.nth ((i + 1) % 5) ≥ 1/9) :=
by sorry

-- Part 2: Arrangement ensuring maximum product of adjacent pairs being at most 1/9.
theorem all_five_non_neg_reals_sum_to_one_max_product_le_one_ninth :
  ∀ (a b c d e : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ a + b + c + d + e = 1 →
  ∃ (perm : list ℝ), perm.perm [a, b, c, d, e] ∧
  ∀ (i : ℕ), 0 ≤ i ∧ i < 5 → (perm.nth i * perm.nth ((i + 1) % 5) ≤ 1/9) :=
by sorry

end exists_five_non_neg_reals_sum_to_one_min_product_ge_one_ninth_all_five_non_neg_reals_sum_to_one_max_product_le_one_ninth_l72_72638


namespace baseEight_conversion_l72_72977

-- Base-eight number is given as 1563
def baseEight : Nat := 1563

-- Function to convert a base-eight number to base-ten
noncomputable def baseEightToBaseTen (n : Nat) : Nat :=
  let digit3 := (n / 1000) % 10
  let digit2 := (n / 100) % 10
  let digit1 := (n / 10) % 10
  let digit0 := n % 10
  digit3 * 8^3 + digit2 * 8^2 + digit1 * 8^1 + digit0 * 8^0

theorem baseEight_conversion :
  baseEightToBaseTen baseEight = 883 := by
  sorry

end baseEight_conversion_l72_72977


namespace students_not_in_biology_l72_72055

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) (enrolled_in_biology : ℚ) :
  total_students = 880 →
  biology_percentage = 27.5 / 100 →
  enrolled_in_biology = total_students * biology_percentage →
  total_students - enrolled_in_biology.toNat = 638 := 
by
  intro h1 h2 h3
  simp [h1, h2, h3]
  sorry

end students_not_in_biology_l72_72055


namespace problem_2002_multiples_l72_72405

theorem problem_2002_multiples :
  ∃ (n : ℕ), 
    n = 1800 ∧
    (∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 149 →
      2002 ∣ (10^j - 10^i) ↔ j - i ≡ 0 [MOD 6]) :=
sorry

end problem_2002_multiples_l72_72405


namespace rectangle_shaded_area_fraction_l72_72882

-- Defining necessary parameters and conditions
variables {R : Type} [LinearOrderedField R]

noncomputable def shaded_fraction (length width : R) : R :=
  let P : R × R := (0, width / 2)
  let Q : R × R := (length / 2, width)
  let rect_area := length * width
  let tri_area := (1 / 2) * (length / 2) * (width / 2)
  let shaded_area := rect_area - tri_area
  shaded_area / rect_area

-- The theorem stating our desired proof goal
theorem rectangle_shaded_area_fraction (length width : R) (h_length : 0 < length) (h_width : 0 < width) :
  shaded_fraction length width = 7 / 8 := by
  sorry

end rectangle_shaded_area_fraction_l72_72882


namespace water_level_equilibrium_l72_72204

noncomputable def h_initial : ℝ := 40
noncomputable def rho_water : ℝ := 1000
noncomputable def rho_oil : ℝ := 700

-- The mathematical problem is proving the final water level (h1) is approximately 16.47 cm
theorem water_level_equilibrium :
  ∃ h_1 h_2: ℝ, (rho_water * h_1 = rho_oil * h_2) ∧ (h_1 + h_2 = h_initial) ∧ (h_1 ≈ 16.47) :=
begin
  sorry
end

end water_level_equilibrium_l72_72204


namespace temperature_difference_l72_72553

variable (T_high T_low diff : ℝ)
variable (T_high_value : T_high = 25)
variable (T_low_value : T_low = 15)

theorem temperature_difference : T_high - T_low = diff → diff = 10 :=
by
  intro h
  rw [T_high_value, T_low_value] at h
  -- Use the fact that 25 - 15 = 10
  have : 25 - 15 = 10 := by norm_num
  rw this at h
  exact h

end temperature_difference_l72_72553


namespace solve_for_n_l72_72532

theorem solve_for_n :
  ∃ n : ℚ, 16 ^ (n + 1) * 16 ^ n * 16 ^ n = 256 ^ 4 ∧ n = 7 / 3 :=
by
  sorry

end solve_for_n_l72_72532


namespace find_lambda_l72_72397

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : vector := (1, -3)
def b : vector := (4, -2)

def lambda_condition (λ : ℝ) : Prop :=
  dot_product (λ • a + b) a = 0

theorem find_lambda (λ : ℝ) : lambda_condition λ → λ = -1 :=
by sorry

end find_lambda_l72_72397


namespace tan_diff_id_l72_72376

theorem tan_diff_id {θ : ℝ} (h : ∃ t : ℝ × ℝ, t = (1, -2) ∧ θ = real.atan2 t.2 t.1) :
  real.tan (π / 4 - θ) = -3 :=
sorry

end tan_diff_id_l72_72376


namespace monotonic_interval_range_of_composite_function_l72_72556

noncomputable def is_log3_of_t (t : ℝ) : ℝ := log (1/3) t

theorem monotonic_interval_range_of_composite_function (x : ℝ) :
  (∀ t > 0, ∀ y, y = is_log3_of_t t → monotonic_decreasing is_log3_of_t) →
  (-x^2 + 2*x + 8 > 0) →
  ((∀ x, x ∈ (1, 4) ∨ x ∈ [1, 4]) ∧ (t ∈ [0, 9] → y ∈ [-2, +∞))) :=
sorry

end monotonic_interval_range_of_composite_function_l72_72556


namespace arithmetic_series_first_term_l72_72335

theorem arithmetic_series_first_term :
  ∃ a d : ℚ, 
    (30 * (2 * a + 59 * d) = 240) ∧
    (30 * (2 * a + 179 * d) = 3240) ∧
    a = - (247 / 12) :=
by
  sorry

end arithmetic_series_first_term_l72_72335


namespace remaining_balance_is_correct_l72_72128

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l72_72128


namespace arithmetic_seq_problem_l72_72436

theorem arithmetic_seq_problem (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 := 
sorry

end arithmetic_seq_problem_l72_72436


namespace sum_first_n_terms_l72_72372

-- State the necessary definitions and the final theorem to be proved
noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem sum_first_n_terms (n : ℕ) :
  (∑ k in Finset.range n, 1 / f (k + 1)) = n / (n + 1) := by
  sorry

end sum_first_n_terms_l72_72372


namespace simplify_expression_l72_72893

variable (y : ℤ)

theorem simplify_expression : 5 * y + 7 * y - 3 * y = 9 * y := by
  sorry

end simplify_expression_l72_72893


namespace monic_quadratic_with_root_real_l72_72711
noncomputable def quadratic_polynomial_with_root : polynomial ℝ :=
  polynomial.X ^ 2 - 4 * polynomial.X + 5

theorem monic_quadratic_with_root_real (α : ℂ) (hα : α = 2 - complex.I) :
    polynomial.eval α quadratic_polynomial_with_root = 0 ∧ 
    polynomial.monic quadratic_polynomial_with_root :=
by
  -- Show that α = 2 - i implies conjugate root 2 + i
  have h_conj : complex.conj α = 2 + complex.I,
  {
    rw [hα, complex.conj_sub, complex.conj_I],
    norm_num,
  }
  -- Verify roots and monic property
  sorry

end monic_quadratic_with_root_real_l72_72711


namespace bombardiers_shots_l72_72576

theorem bombardiers_shots (x y z : ℕ) :
  x + y = z + 26 →
  x + y + 38 = y + z →
  x + z = y + 24 →
  x = 25 ∧ y = 64 ∧ z = 63 := by
  sorry

end bombardiers_shots_l72_72576


namespace population_growth_percent_l72_72799

theorem population_growth_percent (p q r : ℕ) 
  (hp : p^2 = 441) 
  (hq : q^2 - p^2 = 184) 
  (hr : r^2 - p^2 = 500) : 
  (r^2 - p^2 : ℚ) / p^2 * 100 ≈ 227 := 
by 
  have hp' : p^2 = 441 := hp
  sorry

end population_growth_percent_l72_72799


namespace expression_value_l72_72001

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) :
  (a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4 := 
by
  sorry

end expression_value_l72_72001


namespace largest_n_value_l72_72213

theorem largest_n_value (n : ℕ) (h1: n < 100000) (h2: (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0) : n = 99996 := 
sorry

end largest_n_value_l72_72213


namespace fraction_complex_eq_l72_72057

theorem fraction_complex_eq (z : ℂ) (h : z = 2 + I) : 2 * I / (z - 1) = 1 + I := by
  sorry

end fraction_complex_eq_l72_72057


namespace find_four_digit_number_l72_72649

theorem find_four_digit_number (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (h1 : 10 * x + x * y = 100 * x + y) (h2 : ((100 * x + y) % 10) = 5) :
  100 * x + y = 1995 :=
begin
  sorry,
end

end find_four_digit_number_l72_72649


namespace rescue_center_animals_heavier_l72_72039

open Nat Real

theorem rescue_center_animals_heavier :
  let puppy_weights := [6.5, 7.2, 8.0, 9.5]
  let cat_weights := 16 * 2.8
  let bird_weights := [120 / 1000, 150 / 1000, 100 / 1000, 200 / 1000, 175 / 1000, 190 / 1000, 145 / 1000, 160 / 1000, 105 / 1000, 135 / 1000]
  let total_puppy_weight := puppy_weights.sum
  let total_cat_weight := cat_weights
  let total_bird_weight := bird_weights.sum
  let total_rescue_center_weight := total_cat_weight + total_bird_weight
  total_rescue_center_weight - total_puppy_weight = 15.080 :=
by
  let puppy_weights := [6.5, 7.2, 8.0, 9.5]
  let cat_weights := 16 * 2.8
  let bird_weights := [120 / 1000, 150 / 1000, 100 / 1000, 200 / 1000, 175 / 1000, 190 / 1000, 145 / 1000, 160 / 1000, 105 / 1000, 135 / 1000]
  let total_puppy_weight := puppy_weights.sum
  let total_cat_weight := cat_weights
  let total_bird_weight := bird_weights.sum
  let total_rescue_center_weight := total_cat_weight + total_bird_weight
  sorry

end rescue_center_animals_heavier_l72_72039


namespace ce_bisects_bd_l72_72104

variable {A B C D E : Point}
variable {CAB BCA ECD DEC AEC : Angle}

-- Assume that we have a convex pentagon ABCDE
axiom convex_pentagon : ConvexPentagon A B C D E

-- Given angles are equal
axiom equal_angles : CAB = BCA ∧ BCA = ECD ∧ ECD = DEC ∧ DEC = AEC

-- Prove that CE bisects BD
theorem ce_bisects_bd 
  (CP : ConvexPentagon A B C D E) 
  (EA : CAB = BCA ∧ BCA = ECD ∧ ECD = DEC ∧ DEC = AEC) : 
  Bisects C E B D := 
sorry

end ce_bisects_bd_l72_72104


namespace find_theta_l72_72007

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)
hypothesis h_mag_a : ‖a‖ = 3
hypothesis h_mag_b : ‖b‖ = 4
hypothesis h_mag_a_plus_b : ‖a + b‖ = 5
hypothesis h_angle : real.cos theta = 0

theorem find_theta : theta = real.pi / 2 := 
sorry

end find_theta_l72_72007


namespace problem_one_problem_two_l72_72068

-- Defining the sides and angles of the triangle
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Condition 1: cos A = 1/3
def cosA_condition : Prop := cos A = 1 / 3

-- Problem 1: Prove sin^2((B + C) / 2) + cos 2A = -1/9
theorem problem_one (cosA : cosA_condition) : sin^2 ((B + C) / 2) + cos (2 * A) = -1 / 9 := by sorry

-- Additional condition for Problem 2: a = sqrt 3
def a_is_sqrt_3 : Prop := a = √3

-- Problem 2: Prove the maximum area of triangle ABC is 3√2/4 when a = √3
theorem problem_two (cosA : cosA_condition) (a_sqrt3 : a_is_sqrt_3) : 
  let max_area := 3 * √2 / 4 in
  ∀ b c, a = (b * c) * sin A / 2 * max_area := by sorry

end problem_one_problem_two_l72_72068


namespace equal_to_2r_l72_72473

variable (α β γ : Type) [LinearOrderedField α] 
          [MetricSpace β] [HasDist β α] 
          [AddGroup γ] [AddActionSE β γ]

structure Triangle (α : Type) [LinearOrderedField α] : Type :=
(a b c : α)

structure Point (β : Type) : Type :=
(x y : β)

structure Projection (β : Type) extends Point β :=
()

variable (ABC : Triangle α)
variable (A₁ B₁ C₁ : Projection β)
variable (O : Point β)
variable (r : α)

axiom projections_on_altitudes : (ABC.v₀).x = A₁.x ∧ (ABC.v₁).y = B₁.y ∧ (ABC.v₂).z = C₁.z

axiom equal_lengths : dist (ABC.a) A₁ = dist (ABC.b) B₁ ∧ dist (ABC.b) B₁ = dist (ABC.c) C₁

theorem equal_to_2r : equal_lengths ABC A₁ B₁ C₁ → dist (ABC.a) A₁ = 2 * r := sorry

end equal_to_2r_l72_72473


namespace sum_squares_mod_eq_6_l72_72600

def squares_mod (n : ℕ) : ℕ :=
  (List.range n).map (λ x => (x.succ * x.succ % 13))

noncomputable def sum_squares_mod : ℕ :=
  (squares_mod 15).sum 

theorem sum_squares_mod_eq_6 : sum_squares_mod % 13 = 6 := 
by
  sorry

end sum_squares_mod_eq_6_l72_72600


namespace ratio_of_radii_l72_72269

theorem ratio_of_radii (α : Real) (β γ : Real) (h1 : β + γ = 120)
  (h2 : 2 * β + 3 * γ = α) :
  ( 
    let r := (R : Real) in 
    let S := (P : Real) in 
    let P := R * (Real.sin (360 - α) + Real.sin (α - 240) + Real.sin 60) in
    let S1 := Real.sqrt 3 * R^2 * (Real.sin α) * (Real.sin (α + 120)) in
    let S2 := (P * r) / 2 in
    S1 = S2 -> 
    r / R = Real.cos (α + 60) - (1 / 2)
  )

end ratio_of_radii_l72_72269


namespace radius_I_l72_72579

noncomputable def radius_O1 : ℝ := 3
noncomputable def radius_O2 : ℝ := 3
noncomputable def radius_O3 : ℝ := 3

axiom O1_O2_tangent : ∀ (O1 O2 : ℝ), O1 + O2 = radius_O1 + radius_O2
axiom O2_O3_tangent : ∀ (O2 O3 : ℝ), O2 + O3 = radius_O2 + radius_O3
axiom O3_O1_tangent : ∀ (O3 O1 : ℝ), O3 + O1 = radius_O3 + radius_O1

axiom I_O1_tangent : ∀ (I O1 : ℝ), I + O1 = radius_O1 + I
axiom I_O2_tangent : ∀ (I O2 : ℝ), I + O2 = radius_O2 + I
axiom I_O3_tangent : ∀ (I O3 : ℝ), I + O3 = radius_O3 + I

theorem radius_I : ∀ (I : ℝ), I = radius_O1 :=
by
  sorry

end radius_I_l72_72579


namespace final_doll_count_l72_72821

noncomputable def jazmin_initial_dolls : ℝ := 1209
noncomputable def geraldine_initial_dolls : ℝ := 2186
noncomputable def mariana_initial_dolls : ℝ := 3451.5

noncomputable def jazmin_dolls_left : ℤ := Int.ofNat (Float.ceil (jazmin_initial_dolls * (2 / 3)))
noncomputable def geraldine_dolls_left : ℤ := Int.ofNat (Float.ceil (geraldine_initial_dolls * (1 - 0.158)))
noncomputable def mariana_dolls_left : ℤ := Int.ofNat (Float.ceil (mariana_initial_dolls - 987))

noncomputable def total_dolls_left : ℤ := jazmin_dolls_left + geraldine_dolls_left + mariana_dolls_left

theorem final_doll_count :
  total_dolls_left = 5111 :=
by
  sorry

end final_doll_count_l72_72821


namespace permutation_count_condition_l72_72330

theorem permutation_count_condition :
  let perms := {a : Fin 7 → Fin 8 // (∀ i, 1 ≤ a i ∧ a i ≤ 7) ∧ Function.Injective a} in
  (Finset.filter
     (λ (a : perms),
        ((a.1 0 + 1) / 2) * ((a.1 1 + 2) / 2) * ((a.1 2 + 3) / 2) *
        ((a.1 3 + 4) / 2) * ((a.1 4 + 5) / 2) * ((a.1 5 + 6) / 2) *
        ((a.1 6 + 7) / 2) = 5040 ∧
        a.1 6 > a.1 0)
     (Finset.univ : Finset perms)).card = 1 :=
sorry

end permutation_count_condition_l72_72330


namespace find_number_l72_72252

theorem find_number (x : ℝ) : 8050 * x = 80.5 → x = 0.01 :=
by
  sorry

end find_number_l72_72252


namespace main_theorem_l72_72666

variables {A B C M X Y : Point}
variables {AB AC XB CY : ℝ}

-- Defining the conditions mentioned
def is_isosceles_triangle (A B C : Point) : Prop := dist B A = dist C A

def midpoint (M : Point) (B C : Point) : Prop := dist B M = dist C M

def tangent_circle (M : Point) (AB AC : Line) : Prop :=
  ∃ r : ℝ, circle M r ∧ tangent circle (A,B) ∧ tangent circle (A,C)

-- Points X and Y lie on segments AB and AC respectively
def point_on_segment (P Q R : Point) : Prop :=
  colinear P Q R ∧ dist P Q + dist Q R = dist P R

-- Main theorem statement
theorem main_theorem
  (h1 : is_isosceles_triangle A B C)
  (h2 : midpoint M B C)
  (h3 : tangent_circle M (line_through A B) (line_through A C))
  (hX : point_on_segment A X B)
  (hY : point_on_segment A Y C) :
  (tangent (line_segment X Y) (circle M radius)) ↔ (4 * (dist X B) * (dist Y C) = dist B C ^ 2) :=
sorry

end main_theorem_l72_72666


namespace missing_number_is_eight_l72_72895

theorem missing_number_is_eight (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  intro h
  sorry

end missing_number_is_eight_l72_72895


namespace trigo_identity_l72_72782

variable (α : ℝ)

theorem trigo_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (Real.pi / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end trigo_identity_l72_72782


namespace probability_factor_72_lt_5_l72_72595

theorem probability_factor_72_lt_5 : 
  let factors := {x : ℕ | x > 0 ∧ x ∣ 72} in
  let count_factors := (factors.filter (λ x, x < 5)).card in
  count_factors / factors.card = 1 / 3 :=
by
  sorry

end probability_factor_72_lt_5_l72_72595


namespace area_of_fifteen_sided_figure_is_17_l72_72169

-- Define the fifteen-sided figure on 1 cm x 1 cm graph paper
def fifteen_sided_figure_on_graph_paper : Type := sorry

-- State the theorem regarding its area
theorem area_of_fifteen_sided_figure_is_17 : area fifteen_sided_figure_on_graph_paper = 17 := 
sorry

end area_of_fifteen_sided_figure_is_17_l72_72169


namespace problem_equivalence_l72_72434

noncomputable def parametric_circle_point (t : ℝ) : ℝ × ℝ :=
  (-5 + (Real.sqrt 2) * Real.cos t, 3 + (Real.sqrt 2) * Real.sin t)

def polar_to_cartesian_point (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def point_A : ℝ × ℝ := polar_to_cartesian_point 2 (Real.pi / 2)

def point_B : ℝ × ℝ := polar_to_cartesian_point 2 Real.pi

def line_l_cartesian (ρ θ : ℝ) : ℝ :=
  ρ * Real.cos θ - ρ * Real.sin θ

theorem problem_equivalence :
  (∀ t : ℝ, let (x, y) := parametric_circle_point t in (x + 5)^2 + (y - 3)^2 = 2) ∧
  (line_l_cartesian (2 * Real.sqrt 2) (Real.pi / 4) = -2) ∧
  (let (Aₓ, Aᵧ) := point_A, (Bₓ, Bᵧ) := point_B,
       d_min := (4 / Real.sqrt 2) in
     1 / 2 * 2 * Real.sqrt 2 * d_min = 4) :=
by
  sorry

end problem_equivalence_l72_72434


namespace field_trip_seniors_l72_72905

theorem field_trip_seniors (n : ℕ) 
  (h1 : n < 300) 
  (h2 : n % 17 = 15) 
  (h3 : n % 19 = 12) : 
  n = 202 :=
  sorry

end field_trip_seniors_l72_72905


namespace multiples_of_9_ending_in_5_l72_72779

theorem multiples_of_9_ending_in_5 (n : ℕ) :
  (∃ k : ℕ, n = 9 * k ∧ 0 < n ∧ n < 600 ∧ n % 10 = 5) → 
  ∃ l, l = 7 := 
by
sorry

end multiples_of_9_ending_in_5_l72_72779


namespace roots_ab_a_plus_b_eq_one_l72_72466

noncomputable def roots_of_polynomial (x : ℝ) : Polynomial ℝ := x ^ 4 - 6 * x - 1

theorem roots_ab_a_plus_b_eq_one (a b : ℝ) (h_roots : roots_of_polynomial a = 0 ∧ roots_of_polynomial b = 0) :
    a * b + a + b = 1 :=
sorry

end roots_ab_a_plus_b_eq_one_l72_72466


namespace total_length_of_race_l72_72892

-- Define the lengths of each part
def l1 := 15.5
def l2 := 21.5
def l3 := 21.5
def l4 := 16

-- Define the statement to be proved
theorem total_length_of_race : l1 + l2 + l3 + l4 = 74.5 :=
by 
  -- Placeholder for actual proof
  sorry

end total_length_of_race_l72_72892


namespace number_of_subsets_A_l72_72934

-- Define the set A
def A : Set ℕ := {1, 2}

-- Lean statement to prove the number of subsets of A is 4
theorem number_of_subsets_A : (Finset.powerset (Finset.fromSet A)).card = 4 :=
by
  sorry

end number_of_subsets_A_l72_72934


namespace binomial_coefficient_largest_middle_l72_72076

theorem binomial_coefficient_largest_middle
  (n : ℕ) (h : n > 0) :
  (∀ k, binomial n k <= binomial n 5) → n = 10 :=
by sorry

end binomial_coefficient_largest_middle_l72_72076


namespace smallest_n_7770_l72_72847

theorem smallest_n_7770 (n : ℕ) 
  (h1 : ∀ d ∈ n.digits 10, d = 0 ∨ d = 7)
  (h2 : 15 ∣ n) : 
  n = 7770 := 
sorry

end smallest_n_7770_l72_72847


namespace construct_line_e_l72_72807

variables {ABC : Type} [triangle ABC] {e : line} {B1 C1 : point}

-- Assume side b and c with relations to B1 and C1 on the line e, intersecting sides b and c.
theorem construct_line_e (ABC : Triangle) (e : Line) (B1 C1 : Point)
  (A B C : Point)
  (b : Side) (c : Side)
  (h1 : B ∈ b) (h2 : C ∈ c)
  (h3: e.intersect b = B1) (h4: e.intersect c = C1) :
  (A - B1).length = (B1 - C1).length ∧ (B1 - C1).length = (C1 - B).length :=
sorry

end construct_line_e_l72_72807


namespace cups_of_flour_required_l72_72502

/-- Define the number of cups of sugar and salt required by the recipe. --/
def sugar := 14
def salt := 7
/-- Define the number of cups of flour already added. --/
def flour_added := 2
/-- Define the additional requirement of flour being 3 more cups than salt. --/
def additional_flour_requirement := 3

/-- Main theorem to prove the total amount of flour the recipe calls for. --/
theorem cups_of_flour_required : total_flour = 10 :=
by
  sorry

end cups_of_flour_required_l72_72502


namespace Owen_profit_l72_72875

/-- 
Owen bought 12 boxes of face masks, each box costing $9 and containing 50 masks. 
He repacked 6 boxes into smaller packs sold for $5 per 25 masks and sold the remaining masks in baggies of 10 pieces for $3 each.
Prove that Owen's profit amounts to $42.
 -/
theorem Owen_profit :
  let box_count := 12
  let cost_per_box := 9
  let masks_per_box := 50
  let repacked_boxes := 6
  let repack_price := 5
  let repack_size := 25
  let baggy_price := 3
  let baggy_size := 10 in
  let total_cost := box_count * cost_per_box in
  let total_masks := box_count * masks_per_box in
  let masks_repacked := repacked_boxes * masks_per_box in
  let repacked_revenue := (masks_repacked / repack_size) * repack_price in
  let remaining_masks := total_masks - masks_repacked in
  let baggy_revenue := (remaining_masks / baggy_size) * baggy_price in
  let total_revenue := repacked_revenue + baggy_revenue in
  let profit := total_revenue - total_cost in
  profit = 42 := by
  sorry

end Owen_profit_l72_72875


namespace highest_throw_among_them_l72_72681

variable (Christine_throw1 Christine_throw2 Christine_throw3 Janice_throw1 Janice_throw2 Janice_throw3 : ℕ)
 
-- Conditions given in the problem
def conditions :=
  Christine_throw1 = 20 ∧
  Janice_throw1 = Christine_throw1 - 4 ∧
  Christine_throw2 = Christine_throw1 + 10 ∧
  Janice_throw2 = Janice_throw1 * 2 ∧
  Christine_throw3 = Christine_throw2 + 4 ∧
  Janice_throw3 = Christine_throw1 + 17

-- The proof statement
theorem highest_throw_among_them : conditions -> max (max (max (max (max Christine_throw1 Christine_throw2) Christine_throw3) Janice_throw1) Janice_throw2) Janice_throw3 = 37 :=
sorry

end highest_throw_among_them_l72_72681


namespace opposite_of_neg_one_fourth_l72_72938

def opposite_of (x : ℝ) : ℝ := -x

theorem opposite_of_neg_one_fourth :
  opposite_of (-1/4) = 1/4 :=
by
  sorry

end opposite_of_neg_one_fourth_l72_72938


namespace number_of_mappings_l72_72063

theorem number_of_mappings (M N : Type) (m n : ℕ) 
  (hM : Fintype.card M = m) (hN : Fintype.card N = n) : 
  (Fintype.card (M → N)) = n ^ m := 
sorry

end number_of_mappings_l72_72063


namespace BE_parallel_AD_l72_72103

-- Definitions and conditions
variables {A B C D E : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables (AB AC AD BE CE : ℝ)
variables (Γ : set α)

-- Assume necessary conditions
[A_isosceles : AB = AC]
[Γ_is_circumcircle : ∀ {X Y : α}, X ∈ Γ ∧ Y ∈ Γ → metric_ℝ.dist X Y = metric_ℝ.dist (circumcenter_tr X Y) (circumradius X Y)]
[D_on_arc_AB : ∃ (P : α), P ∈ arc AB ∧ ¬ (P_mem segment AC)]
[E_on_arc_AC : ∃ (Q : α), Q ∈ arc AC ∧ ¬ (Q_mem segment AB)]
[AD_eq_CE : AD = CE]

-- The statement to be proven
theorem BE_parallel_AD : parallel BE AD :=
sorry

end BE_parallel_AD_l72_72103


namespace remaining_balance_is_correct_l72_72130

def initial_balance : ℕ := 50
def spent_coffee : ℕ := 10
def spent_tumbler : ℕ := 30

theorem remaining_balance_is_correct : initial_balance - (spent_coffee + spent_tumbler) = 10 := by
  sorry

end remaining_balance_is_correct_l72_72130


namespace walt_age_l72_72557

theorem walt_age (T W : ℕ) 
  (h1 : T = 3 * W)
  (h2 : T + 12 = 2 * (W + 12)) : 
  W = 12 :=
by
  sorry

end walt_age_l72_72557


namespace multiple_of_michael_trophies_l72_72819

-- Conditions
def michael_current_trophies : ℕ := 30
def michael_trophies_increse : ℕ := 100
def total_trophies_in_three_years : ℕ := 430

-- Proof statement
theorem multiple_of_michael_trophies (x : ℕ) :
  (michael_current_trophies + michael_trophies_increse) + (michael_current_trophies * x) = total_trophies_in_three_years → x = 10 := 
by
  sorry

end multiple_of_michael_trophies_l72_72819


namespace measure_inf_eq_zero_l72_72840

noncomputable def extended_random_variable (Ω : Type*) [MeasurableSpace Ω] (μ : Measure Ω) : Type* := sorry

variables {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} (ξ : extended_random_variable Ω μ)
variable (I : ℝ)

axiom finite_integral : I = ∫ ω in Ω, |ξ| ∂μ < ∞

theorem measure_inf_eq_zero (h : finite_integral I ξ) : μ {ω | |ξ ω| = ∞} = 0 := 
sorry

end measure_inf_eq_zero_l72_72840


namespace liquid_Y_radius_correct_l72_72264

noncomputable def liquid_Y_radius (V : ℝ) (thickness : ℝ) : ℝ :=
  real.sqrt (V / (thickness * real.pi))

theorem liquid_Y_radius_correct :
  liquid_Y_radius 320 0.15 = real.sqrt (2133.33 / real.pi) :=
by
  sorry

end liquid_Y_radius_correct_l72_72264


namespace new_dwelling_points_relationship_l72_72659

def g (x : ℝ) : ℝ := Real.sin x
def h (x : ℝ) : ℝ := Real.log x
def φ (x : ℝ) : ℝ := x^3

theorem new_dwelling_points_relationship : 
  let a := π / 4,
      b := exists! (λ x, Real.log x = 1 / x ∧ x > 1 ∧ x < Real.exp 1),
      c := 3 in
   a < (Classical.choose b) ∧ (Classical.choose b) < c :=
begin
  sorry
end

end new_dwelling_points_relationship_l72_72659


namespace G_n_zero_l72_72479

theorem G_n_zero (x y z A B C : ℝ) (h1 : A + B + C = Real.pi)
  (G : ℕ+ → ℝ) (h2 : G 1 = 0) (h3 : G 2 = 0)
  (h4 : ∀ n : ℕ+, G n = x^n * Real.sin (n * A) + y^n * Real.sin (n * B) + z^n * Real.sin (n * C)) :
  ∀ n : ℕ+, G n = 0 :=
sorry

end G_n_zero_l72_72479


namespace neg_p_equivalence_l72_72031

theorem neg_p_equivalence:
  (∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
sorry

end neg_p_equivalence_l72_72031


namespace part1_part2_l72_72732

noncomputable def quadratic_eq (m x : ℝ) : Prop := m * x^2 - 2 * x + 1 = 0

theorem part1 (m : ℝ) : 
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 ≠ x2) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

theorem part2 (m : ℝ) (x1 x2 : ℝ) : 
  (quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ x1 * x2 - x1 - x2 = 1/2) ↔ (m = -2) :=
by sorry

end part1_part2_l72_72732


namespace find_q_l72_72770

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 6) : q = 3 + Real.sqrt 3 :=
by
  sorry

end find_q_l72_72770


namespace true_statements_count_l72_72537

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem true_statements_count :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + 
  (if s2 then 1 else 0) + 
  (if s3 then 1 else 0) + 
  (if s4 then 1 else 0) = 2 :=
by
  sorry

end true_statements_count_l72_72537


namespace smallest_digit_A_divisible_by_3_l72_72442

theorem smallest_digit_A_divisible_by_3:
  ∃ A : ℕ, A < 10 ∧ (4 + A + 8 + 8 + 5 + 1) % 3 = 0 ∧ ∀ A' < 10, ((4 + A' + 8 + 8 + 5 + 1) % 3 = 0 → 1 ≤ A) :=
begin
  use 1,
  split,
  { norm_num },
  { split,
    { norm_num },
    { intros A' A'_bound h,
      have : (2 + A') % 3 = 0 := by norm_num [show 26 % 3 = 2 by norm_num, nat.add_mod],
      norm_num [this] at h,
      exact le_of_eq h,
    }
  },
end

end smallest_digit_A_divisible_by_3_l72_72442


namespace center_of_incircle_on_line_passing_through_midpoints_l72_72891

-- Definitions of the points
variables {A B C D E F L M N : Type*}

-- Triangle and points conditions
variables (triangle_ABC : A ∈ triangle B C A)
variables (incircle_property : touches_incircle B C A D E F)
variables (excircle_property : touches_excircle_opposite A B C L M N)

-- Assert the proof problem
def proof_problem : Prop :=
  let incircle_center := center_of_incircle A B C in
  let midpoint_BC := midpoint B C in
  let midpoint_AD := midpoint A D in
  lies_on_line incircle_center (line_through midpoint_BC midpoint_AD)

-- Proof of the problem
theorem center_of_incircle_on_line_passing_through_midpoints :
  proof_problem triangle_ABC incircle_property excircle_property := sorry

end center_of_incircle_on_line_passing_through_midpoints_l72_72891


namespace min_abc_value_l72_72845

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem min_abc_value
  (a b c : ℕ)
  (h1: is_prime a)
  (h2 : is_prime b)
  (h3 : is_prime c)
  (h4 : a^5 ∣ (b^2 - c))
  (h5 : ∃ k : ℕ, (b + c) = k^2) :
  a * b * c = 1958 := sorry

end min_abc_value_l72_72845


namespace find_AC_l72_72083

theorem find_AC (A B C : ℝ) (hA : A = 120 * real.pi / 180) (BC : ℝ) (AB : ℝ)
  (hBC : BC = real.sqrt 19) (hAB : AB = 2) : AC = 3 :=
by
  -- Defining cosine of 120 degrees 
  have cos_120 : real.cos (120 * real.pi / 180) = -1 / 2 := by sorry
  -- Applying the Law of Cosines
  let b := AB
  let c := BC
  let a := AC
  have h_cos : a^2 = b^2 + c^2 - 2 * b * c * cos_120 := by sorry
  -- Solving for AC
  exact sorry

end find_AC_l72_72083


namespace ellipse_equation_min_OB_value_l72_72380

noncomputable def eccentricity := (c : ℝ) (a : ℝ) := c / a

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity (real.sqrt(2/3) * a) a = real.sqrt(6)/3)
  (h4 : ∃ x y : ℝ, x = real.sqrt(3) ∧ y = 1 ∧ (x^2 / a^2 + y^2 / b^2 = 1)) : 
  (∃ (a b : ℝ), (b^2 = 1/3 * a^2) ∧ (a^2 = 6) ∧ (b^2 = 2)) ∧ 
    (∀ (x y : ℝ), (x^2 / 6 + y^2 / 2 = 1) := 
sorry

theorem min_OB_value (a b x_0 y_0 : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity (real.sqrt(2/3) * a) a = real.sqrt(6)/3)
  (h4 : ∃ x y : ℝ, x = real.sqrt(3) ∧ y = 1 ∧ (x^2 / a^2 + y^2 / b^2 = 1))
  (h5 : x_0^2 / 6 + y_0^2 / 2 = 1) :
  ∃ y_0 : ℝ, y_0 ≠ 0 ∧ min_value (λ (y_0 : ℝ), |y_0| + (3 / 2) * (1 / |y_0|)) = real.sqrt(6) := 
sorry

end ellipse_equation_min_OB_value_l72_72380


namespace regular_hexagon_and_circle_same_perimeter_l72_72272

theorem regular_hexagon_and_circle_same_perimeter (P : ℝ) (hP : 0 < P) :
  let A := π * (P / 6) ^ 2 
      B := π * (P / (2 * π)) ^ 2 
  in A / B = π / 9 :=
by
  -- Definitions for area
  let side_length_hex := P / 6
  let radius_circum_hex := side_length_hex
  let area_circum_hex := π * radius_circum_hex ^ 2
  let radius_circle := P / (2 * π)
  let area_circle := π * radius_circle ^ 2

  -- Calculate the ratio
  let ratio := area_circum_hex / area_circle

  -- Final conclusion
  have h1 : A = π * (P / 6)^2 := rfl
  have h2 : B = π * (P / (2 * π))^2 := rfl
  have h3 : ratio = (A / B) := rfl

  exact /-
    Include the necessary steps to show the equivalence,
    leading to: 
      ratio = π / 9 
    as required.
  -/
  sorry

end regular_hexagon_and_circle_same_perimeter_l72_72272


namespace total_emails_received_l72_72673

theorem total_emails_received (E : ℝ)
    (h1 : (3/5) * (3/4) * E = 180) :
    E = 400 :=
sorry

end total_emails_received_l72_72673


namespace cars_to_hours_l72_72957

def car_interval := 20 -- minutes
def num_cars := 30
def minutes_per_hour := 60

theorem cars_to_hours :
  (car_interval * num_cars) / minutes_per_hour = 10 := by
  sorry

end cars_to_hours_l72_72957


namespace angle_measure_l72_72418

-- Defining the type of angles in degrees
def is_complementary (a b : ℝ) : Prop := a + b = 90
def is_supplementary (a b : ℝ) : Prop := a + b = 180

-- Defining the conditions 
def conditions (x : ℝ) : Prop :=
  is_complementary (90 - x) (180 - x)

-- Main theorem statement
theorem angle_measure (x : ℝ) (h : conditions x) : x = 45 :=
  sorry

end angle_measure_l72_72418


namespace minimum_positive_s_n_at_n_19_l72_72357

noncomputable theory

namespace ArithmeticSequence

open Function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

variables (a : ℕ → ℝ)

#check ∑ i in finset.range 11, a i -- to ensure the Lean sum function for sequences

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i

axiom condition_1 : is_arithmetic_sequence a 
axiom condition_2 : a 11 / a 10 < -1
axiom condition_3 : ∃ n, S_n a n = real.Sup (set.range (S_n a))

theorem minimum_positive_s_n_at_n_19 : 
  (∀ n, S_n a n = real.Sup (set.Icc 0 (real.Sup (set.range (S_n a)))) → n = 19) :=
sorry

end ArithmeticSequence

end minimum_positive_s_n_at_n_19_l72_72357


namespace sum_squares_mod_13_l72_72609

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 10 := by
  sorry

end sum_squares_mod_13_l72_72609


namespace sum_f_1_to_2010_l72_72759

def f (x : ℝ) : ℝ := Real.sin (π * x / 3)

lemma period_of_f : ∀ x, f(x) = f(x + 6) :=
begin
  intro x,
  -- proof about the periodicity
  sorry
end

theorem sum_f_1_to_2010 : (∑ k in Finset.range 2010, f (k + 1)) = 0 :=
begin
  -- proof about the sum of first 2010 values of f(x)
  sorry
end

end sum_f_1_to_2010_l72_72759


namespace angle_CAP_eq_angle_BAQ_l72_72970

-- Define basic geometrical setup and points
variables {C1 C2 : Circle} {A B P Q C : Point}

-- Define the conditions of the problem
def conditions : Prop := 
  C1 ∩ C2 = {A, B} ∧  -- C_1 and C_2 intersect at points A and B
  tangent_at P C1 Q ∧ -- External common tangent touches C1 at P and C2 at Q
  reflection B PQ C    -- C is the reflection of B in line PQ

-- Theorem to be proven
theorem angle_CAP_eq_angle_BAQ (h : conditions) : 
  angle C A P = angle B A Q :=
  sorry

end angle_CAP_eq_angle_BAQ_l72_72970


namespace must_be_true_if_not_all_electric_l72_72277

variable (P : Type) (ElectricCar : P → Prop)

theorem must_be_true_if_not_all_electric (h : ¬ ∀ x : P, ElectricCar x) : 
  ∃ x : P, ¬ ElectricCar x :=
by 
sorry

end must_be_true_if_not_all_electric_l72_72277


namespace solution_correctness_l72_72897

noncomputable def solve_equation : ℝ :=
  let b := (9:ℝ) / (5:ℝ) in 
  Real.log (9^5) / Real.log b

theorem solution_correctness (x : ℝ) (condition : 9^(x + 5) = 5^x) : 
  x = solve_equation :=
sorry

end solution_correctness_l72_72897


namespace coeff_x10_expansion_l72_72908

-- Definitions based on conditions of the problem
def polynomial := (x + 2)^10 * (x^2 - 1)

-- Main theorem statement
theorem coeff_x10_expansion : (polynomial.coeff 10) = 179 :=
by
  sorry

end coeff_x10_expansion_l72_72908


namespace relationship_between_x_and_z_l72_72495

-- Definitions of the given conditions
variable {x y z : ℝ}

-- Statement of the theorem
theorem relationship_between_x_and_z (h1 : x = 1.027 * y) (h2 : y = 0.45 * z) : x = 0.46215 * z :=
by
  sorry

end relationship_between_x_and_z_l72_72495


namespace frank_remaining_money_l72_72343

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end frank_remaining_money_l72_72343


namespace probability_solution_l72_72490

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l72_72490


namespace farmer_milk_production_l72_72251

theorem farmer_milk_production
  (total_cattle : ℕ)
  (male_cows : ℕ)
  (female_percentage : ℚ)
  (milk_per_day_per_female : ℕ)
  (males_to_total_ratio : ℚ)
  (total_cattle_eq : total_cattle = male_cows / males_to_total_ratio)
  (female_cows_eq : nat.cast (total_cattle) * female_percentage = nat.cast (total_cattle) - nat.cast (male_cows))
  (males_count : male_cows = 50)
  (f_perc : female_percentage = 0.60)
  (milk_per_female: milk_per_day_per_female = 2)
  (males_ratio : males_to_total_ratio = 0.40)
  : total_cattle * female_percentage * milk_per_day_per_female = 150 :=
by
  sorry

end farmer_milk_production_l72_72251


namespace functional_inequality_l72_72352

def f (n : ℕ) : ℝ := sorry

theorem functional_inequality (n : ℕ) (hn : 0 < n) : 
  f (n + 1) ≥ f n + Real.log2 ((n + 1) / n) := 
sorry

end functional_inequality_l72_72352


namespace find_f_e_plus_f_2_minus_e_l72_72750

def f : ℝ → ℝ := sorry

def g (x : ℝ) : ℝ := f(x + 1) + 5

axiom h_odd_g : ∀ x : ℝ, g(-x) = -g(x)

theorem find_f_e_plus_f_2_minus_e : f(real.e) + f(2 - real.e) = -10 := by
  have h1 : g(real.e - 1) = f(real.e) + 5 := sorry
  have h2 : g(1 - real.e) = f(2 - real.e) + 5 := sorry
  have h3 : g(real.e - 1) + g(1 - real.e) = 0 := h_odd_g (real.e - 1)
  sorry

end find_f_e_plus_f_2_minus_e_l72_72750


namespace find_z_l72_72351

-- Define the problem in Lean
noncomputable def target_z := -1 + complex.I * real.sqrt 3

-- Main theorem statement
theorem find_z {z : ℂ} (h1 : complex.arg (z + 2) = real.pi / 3) (h2 : complex.arg (z - 2) = 5 * real.pi / 6) :
  z = target_z :=
sorry

end find_z_l72_72351


namespace range_of_a_l72_72740

noncomputable def a := Real
def f (a : a) (x : Real) := x + a^2 / x
def g (x : Real) := x - log x

theorem range_of_a (a : Real) (x1 x2 : Real) (h1 : a > 0)
  (h2 : x1 ∈ set.Icc (1/e) 1)
  (h3 : x2 ∈ set.Icc (1/e) 1) :
  (∀ x2 ∈ set.Icc (1/e) 1, ∃ x1 ∈ set.Icc (1/e) 1, f a x1 ≥ g x2) ↔
  a ∈ set.Icc (1/2) + ∞ ∪ set.Icc (sqrt (e-1) / e) (1 / e) :=
sorry

end range_of_a_l72_72740


namespace opposite_of_neg_one_fourth_l72_72937

def opposite_of (x : ℝ) : ℝ := -x

theorem opposite_of_neg_one_fourth :
  opposite_of (-1/4) = 1/4 :=
by
  sorry

end opposite_of_neg_one_fourth_l72_72937


namespace haley_total_lives_l72_72634

-- Define initial conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def lives_gained : ℕ := 36

-- Definition to calculate total lives
def total_lives (initial_lives lives_lost lives_gained : ℕ) : ℕ :=
  initial_lives - lives_lost + lives_gained

-- The theorem statement we want to prove
theorem haley_total_lives : total_lives initial_lives lives_lost lives_gained = 46 :=
by 
  sorry

end haley_total_lives_l72_72634


namespace who_received_q_first_l72_72959

-- Definitions based on the conditions given in the problem.
variables (p q r : ℕ) -- p, q, and r are natural numbers
variable h₀ : 0 < p ∧ p < q ∧ q < r -- Condition 1
variable h₁ : 20 + 10 + 9 = (p + q + r) * 3 -- Condition 2
variable h₂ : ∑ (i : fin 3), (i ∈ [20, 10, 9] ↔ (B_last := r)) -- Condition 3
variable h₃ : ∃ n, n ≥ 2 -- Condition 4

theorem who_received_q_first (h₀ : 0 < p ∧ p < q ∧ q < r)
                            (h₁ : 20 + 10 + 9 = (p + q + r) * 3)
                            (h₂ : B = r)
                            (h₃ : ∃ n, n ≥ 2) :
  (C_first := q) :=
by
  sorry

end who_received_q_first_l72_72959


namespace triangle_area_is_correct_l72_72307

-- Define the vertices of the triangle
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 3 }
def B : Point := { x := 5, y := -1 }
def C : Point := { x := 2, y := 6 }

-- Define the formula for the area of a triangle given its vertices
def triangle_area (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

-- Prove that the area of the triangle with vertices A, B, and C is 18.5
theorem triangle_area_is_correct : triangle_area A B C = 18.5 := by
  sorry

end triangle_area_is_correct_l72_72307


namespace expand_polynomial_l72_72704

theorem expand_polynomial :
  (5 * x^2 + 3 * x - 4) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 12 * x^3 := 
by
  sorry

end expand_polynomial_l72_72704


namespace fraction_nonnegative_for_interval_l72_72339

theorem fraction_nonnegative_for_interval :
  {x : ℝ | x - 4 * x^2 + 16 * x^3 ≥ 0} ∩ {x : ℝ | 12 - x^3 ≥ 0} = {x : ℝ | x ∈ Icc 0 (2 * Real.cbrt 3)} :=
sorry

end fraction_nonnegative_for_interval_l72_72339


namespace hyperbola_condition_l72_72411

-- Define the condition and the problem
def hyperbola_condition_suff_not_necessary (k : ℝ) : Prop :=
  k > 3 ↔ ∃ f : ∀ (x y : ℝ), (k = 4) ∧ (k-3) * (k+3) > 0

theorem hyperbola_condition (k : ℝ) (hk : k ∈ ℝ) :
  (k > 3) ↔ hyperbola_condition_suff_not_necessary k := 
sorry

end hyperbola_condition_l72_72411


namespace part1_part2_part3_exists_AB_l72_72851

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
axiom f_finv_ineq : ∀ x : ℝ, f x + f_inv x < (5/2) * x

def seq_a : ℕ → ℝ
| 0       := 8
| 1       := 10
| (n + 2) := f (seq_a (n + 1))

def seq_b (n : ℕ) : ℝ := seq_a (n + 1) - 2 * seq_a n

theorem part1 (n : ℕ) (h : n ≥ 1) : seq_a (n + 1) + seq_a (n - 1) < (5/2) * seq_a n :=
sorry

theorem part2 (n : ℕ) (h : n ∈ ℕ*): seq_b n < -6 * (1/2)^n :=
sorry

theorem part3_exists_AB :
  (∃ A B : ℝ, (seq_a 0 = (A + B)) ∧ (seq_a 1 = (4 * A + B) / 2) ∧ 
  (∀ n ≥ 2, seq_a n < (A * 4^n + B) / 2^n))
  → ((A = 4 ∧ B = 4)) :=
sorry


end part1_part2_part3_exists_AB_l72_72851


namespace simple_interest_rate_l72_72675

theorem simple_interest_rate (P A : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) :
  P = 800 → A = 950 → T = 5 → SI = A - P → SI = (P * R * T) / 100 → R = 3.75 :=
  by
  intros hP hA hT hSI h_formula
  sorry

end simple_interest_rate_l72_72675


namespace omega_terms_sum_to_zero_l72_72465

theorem omega_terms_sum_to_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 :=
by sorry

end omega_terms_sum_to_zero_l72_72465


namespace unoccupied_volume_correct_l72_72526

-- Define the conditions given in the problem
def tank_length := 12 -- inches
def tank_width := 8 -- inches
def tank_height := 10 -- inches
def water_fraction := 1 / 3
def ice_cube_side := 1 -- inches
def num_ice_cubes := 12

-- Calculate the occupied volume
noncomputable def tank_volume : ℝ := tank_length * tank_width * tank_height
noncomputable def water_volume : ℝ := tank_volume * water_fraction
noncomputable def ice_cube_volume : ℝ := ice_cube_side^3
noncomputable def total_ice_volume : ℝ := ice_cube_volume * num_ice_cubes
noncomputable def total_occupied_volume : ℝ := water_volume + total_ice_volume

-- Calculate the unoccupied volume
noncomputable def unoccupied_volume : ℝ := tank_volume - total_occupied_volume

-- State the problem
theorem unoccupied_volume_correct : unoccupied_volume = 628 := by
  sorry

end unoccupied_volume_correct_l72_72526


namespace max_value_of_f_l72_72718

def f (x : ℝ) : ℝ := min (3 * x + 4) (min (- (1/3) * x + 2) (- (1/2) * x + 8))

theorem max_value_of_f (x : ℝ) : ∃! value : ℝ, value = 11 / 5 ∧ 
  (∀ x, f x ≤ value) :=
by
  sorry

end max_value_of_f_l72_72718


namespace village_population_l72_72207

variable (Px : ℕ)
variable (py : ℕ := 42000)
variable (years : ℕ := 16)
variable (rate_decrease_x : ℕ := 1200)
variable (rate_increase_y : ℕ := 800)

theorem village_population (Px : ℕ) (py : ℕ := 42000)
  (years : ℕ := 16) (rate_decrease_x : ℕ := 1200)
  (rate_increase_y : ℕ := 800) :
  Px - rate_decrease_x * years = py + rate_increase_y * years → Px = 74000 := by
  sorry

end village_population_l72_72207


namespace sum_of_reciprocals_of_roots_l72_72853

theorem sum_of_reciprocals_of_roots :
  let f := polynomial.C 2 * polynomial.X^3 + polynomial.C 3 * polynomial.X^2 + polynomial.C 5 * polynomial.X + polynomial.C 7 in
  let roots := (f.roots : multiset ℂ) in
  (roots.map (λ r, (1 : ℂ) / r)).sum = -5 / 7 :=
by
  sorry

end sum_of_reciprocals_of_roots_l72_72853


namespace HunterScoreIs45_l72_72776

variable (G J H : ℕ)
variable (h1 : G = J + 10)
variable (h2 : J = 2 * H)
variable (h3 : G = 100)

theorem HunterScoreIs45 : H = 45 := by
  sorry

end HunterScoreIs45_l72_72776


namespace probability_solution_l72_72493

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l72_72493


namespace point_2023_coordinates_l72_72014

theorem point_2023_coordinates :
  let x_coord := λ (n : ℕ), 2 * n - 1
  let y_coord := λ (n : ℕ), (-1)^(n-1) * 2^n
  x_coord 2023 = 4045 ∧ y_coord 2023 = 2^2023 :=
by
  sorry

end point_2023_coordinates_l72_72014


namespace find_number_l72_72889

theorem find_number (x : ℝ) (h : 7 * x = 3 * x + 12) : x = 3 :=
by
  sorry

end find_number_l72_72889


namespace determinant_matrix_A_l72_72301

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ := 
  !![-5, 3; 4, -4]

theorem determinant_matrix_A :
  matrix.det matrix_A = 8 := by
  sorry

end determinant_matrix_A_l72_72301


namespace grouping_factorization_splitting_factorization_perimeter_of_triangle_l72_72586

-- Definitions for part (1) Grouping Method
def grouping_factorize (x y : ℝ) : (ℝ × ℝ) :=
  (3 * x + 1 + y, 3 * x + 1 - y)

theorem grouping_factorization (x y : ℝ) :
  (grouping_factorize x y).fst * (grouping_factorize x y).snd = 9 * x^2 + 6 * x - y^2 + 1 :=
by
  sorry

-- Definitions for part (2) Splitting Method
def splitting_factorize (x : ℝ) : (ℝ × ℝ) :=
  (x - 2, x - 4)

theorem splitting_factorization (x : ℝ) :
  (splitting_factorize x).fst * (splitting_factorize x).snd = x^2 - 6 * x + 8 :=
by
  sorry

-- Definitions for part (3) Finding Perimeter
def sides_of_triangle := {a b c : ℝ // a^2 + 5 * b^2 + c^2 - 4 * a * b - 6 * b - 10 * c + 34 = 0}

def perimeter (t : sides_of_triangle) : ℝ :=
  t.val.1 + t.val.2 + t.val.3

theorem perimeter_of_triangle (t : sides_of_triangle) : perimeter t = 14 :=
by
  sorry

end grouping_factorization_splitting_factorization_perimeter_of_triangle_l72_72586


namespace all_numbers_positive_l72_72806

theorem all_numbers_positive 
  (a : Fin 21 → ℝ)
  (h : ∀ (S : Finset (Fin 21)), 
        (S.card = 10) → 
        (∑ i in S, a i < ∑ j in (Finset.univ \ S), a j)) :
  ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l72_72806


namespace prob_Y_gt_4_l72_72062

noncomputable def p : ℝ := 1 - (0.343)^(1/3)

def X : ℕ → ℤ := λ n, Binomial n p

def Y : ℝ → ℝ := λ y, Normal 2 (δ ^ 2)

axiom P_X_ge_1 : (X ≥ 1) = 0.657
axiom P_0_lt_Y_lt_2 : (0 < Y < 2) = p

theorem prob_Y_gt_4 : (Y > 4) = 0.2 := by
  sorry

end prob_Y_gt_4_l72_72062


namespace max_trig_expression_is_5_over_2_l72_72329

noncomputable def max_trig_expression : ℝ :=
  (λ θ : ℝ, 5 * (1 / 2) * real.sin (2 * θ)).sup

theorem max_trig_expression_is_5_over_2 :
  max_trig_expression = 5 / 2 := 
by sorry

end max_trig_expression_is_5_over_2_l72_72329


namespace parabola_pass_through_fixed_point_l72_72746

theorem parabola_pass_through_fixed_point
  (p : ℝ) (hp : p > 0)
  (xM yM : ℝ) (hM : (xM, yM) = (1, -2))
  (hMp : yM^2 = 2 * p * xM)
  (xA yA xC yC xB yB xD yD : ℝ)
  (hxA : xA = xC ∨ xA ≠ xC)
  (hxB : xB = xD ∨ xB ≠ xD)
  (x2 y0 : ℝ) (h : (x2, y0) = (2, 0))
  (m1 m2 : ℝ) (hm1m2 : m1 * m2 = -1)
  (l1_intersect_A : xA = m1 * yA + 2)
  (l1_intersect_C : xC = m1 * yC + 2)
  (l2_intersect_B : xB = m2 * yB + 2)
  (l2_intersect_D : xD = m2 * yD + 2)
  (hMidM : (2 * xA + 2 * xC = 4 * xM ∧ 2 * yA + 2 * yC = 4 * yM))
  (hMidN : (2 * xB + 2 * xD = 4 * xM ∧ 2 * yB + 2 * yD = 4 * yM)) :
  (yM^2 = 4 * xM) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, y = k * x ↔ y = xM / (m1 + m2) ∧ y = m1) :=
sorry

end parabola_pass_through_fixed_point_l72_72746


namespace sum_a_terms_l72_72045

theorem sum_a_terms :
  ∀ (a : ℕ → ℤ),
  (∀ x : ℤ, (x^2 + 1) * (2 * x + 1)^9 = 
    ∑ i in Finset.range 12, a i * (x + 2)^i)
  → (∑ i in Finset.range 12, a i = -2) :=
by
  intros a h
  have h_eval := h (-1)
  simp at h_eval
  rw [Finset.sum_range_succ] at h_eval
  sorry

end sum_a_terms_l72_72045


namespace domain_of_f_range_of_f_l72_72758

-- Define the function f
def f (x : ℝ) : ℝ := (cos (2 * x)) / (sin x + cos x)

-- Domain definition: x should not be kπ - π/4 where k is an integer
def domain_f (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * π - π / 4

-- Range definition: f(x) should be within (-sqrt(2), sqrt(2))
def range_f (y : ℝ) : Prop := - real.sqrt 2 < y ∧ y < real.sqrt 2

theorem domain_of_f :
  ∀ x : ℝ, 
  (sin x + cos x ≠ 0) ↔ domain_f x :=
sorry

theorem range_of_f :
  ∀ x : ℝ, 
  domain_f x →
  range_f (f x) :=
sorry

end domain_of_f_range_of_f_l72_72758


namespace ninety_percent_of_population_is_expected_number_l72_72253

/-- Define the total population of the village -/
def total_population : ℕ := 9000

/-- Define the percentage rate as a fraction -/
def percentage_rate : ℕ := 90

/-- Define the expected number of people representing 90% of the population -/
def expected_number : ℕ := 8100

/-- The proof problem: Prove that 90% of the total population is 8100 -/
theorem ninety_percent_of_population_is_expected_number :
  (percentage_rate * total_population / 100) = expected_number :=
by
  sorry

end ninety_percent_of_population_is_expected_number_l72_72253


namespace log_cot_square_l72_72783

noncomputable def cot (θ : ℝ) : ℝ := (Real.cos θ) / (Real.sin θ)

theorem log_cot_square (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) :
  Real.logBase (Real.sin θ) (1 + cot θ ^ 2) = -2 :=
by
  sorry

end log_cot_square_l72_72783


namespace Dave_has_more_money_than_Derek_l72_72312

theorem Dave_has_more_money_than_Derek:
    ∀ (Derek_has Dave_has Derek_lunch1 Derek_lunch2 Derek_lunch3 Derek_dessert Derek_discount 
       Dave_lunch1 Dave_lunch2 Dave_snacks Dave_tax) 
    (H1: Derek_has = 40)
    (H2: Derek_lunch1 = 14)
    (H3: Derek_lunch2 = 11)
    (H4: Derek_lunch3 = 5)
    (H5: Derek_dessert = 8)
    (H6: Derek_discount = 0.10)
    (H7: Dave_has = 50)
    (H8: Dave_lunch1 = 7)
    (H9: Dave_lunch2 = 12)
    (H10: Dave_snacks = 9)
    (H11: Dave_tax = 0.08),
    let Derek_expense := Derek_lunch1 + Derek_lunch2 + Derek_lunch3 + Derek_dessert in
    let Derek_final := Derek_expense - (Derek_discount * Derek_expense) in
    let Derek_left := Derek_has - Derek_final in

    let Dave_expense := Dave_lunch1 + Dave_lunch2 + Dave_snacks in
    let Dave_final := Dave_expense + (Dave_tax * Dave_expense) in
    let Dave_left := Dave_has - Dave_final in
    
    (Dave_left - Derek_left = 13.96) :=
by {
    intros,
    let Derek_expense := Derek_lunch1 + Derek_lunch2 + Derek_lunch3 + Derek_dessert,
    let Derek_final := Derek_expense - (Derek_discount * Derek_expense),
    let Derek_left := Derek_has - Derek_final,

    let Dave_expense := Dave_lunch1 + Dave_lunch2 + Dave_snacks,
    let Dave_final := Dave_expense + (Dave_tax * Dave_expense),
    let Dave_left := Dave_has - Dave_final,

    have H_Derek_expense: Derek_expense = 38 := sorry,
    have H_Derek_discount_amount: Derek_discount * Derek_expense = 3.80 := sorry,
    have H_Derek_final: Derek_final = 34.20 := sorry,
    have H_Derek_left: Derek_left = 5.80 := sorry,

    have H_Dave_expense: Dave_expense = 28 := sorry,
    have H_Dave_tax_amount: Dave_tax * Dave_expense = 2.24 := sorry,
    have H_Dave_final: Dave_final = 30.24 := sorry,
    have H_Dave_left: Dave_left = 19.76 := sorry,

    rw [H_Derek_left, H_Dave_left],
    exact H_Derek_left,

    norm_num at H_Derek_left H_Dave_left,
}

end Dave_has_more_money_than_Derek_l72_72312


namespace sum_squares_mod_13_l72_72620

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 5 :=
by sorry

end sum_squares_mod_13_l72_72620


namespace monotonic_decreasing_interval_l72_72929

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∃ t ∈ Ioo (-1 : ℝ) 11, t = x} ⊆ {x : ℝ | ∃ t, f'(t) < 0} :=
sorry

end monotonic_decreasing_interval_l72_72929


namespace circle_through_points_eq_center_radius_l72_72709

theorem circle_through_points_eq_center_radius :
  ∃ (D E F : ℝ), 
    (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
    (∀ (x y: ℝ), (x = 1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
    (∀ (x y: ℝ), (x = 4 ∧ y = 2) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧ 
    ((x: ℝ)pow 2 - 4*x + 3*y = 0 → ((x - 4)^2 + (y - (-3))^2 = 25)) :=
by {
  sorry
}

end circle_through_points_eq_center_radius_l72_72709


namespace cos_graph_shift_left_l72_72967

/-- To obtain the graph of the function y = cos (x + π/3), 
    the graph of y = cos x needs to be shifted π/3 units to the left. -/
theorem cos_graph_shift_left (x : ℝ) : 
  ∃ c : ℝ, (∀ x : ℝ, cos(x) = cos(x + c)) ∧ c = -π/3 := 
sorry

end cos_graph_shift_left_l72_72967


namespace oblique_coordinates_properties_l72_72774

variables {α : Type*} [inner_product_space ℝ α]

def unit_vectors (e1 e2 : α) : Prop :=
  ∥e1∥ = 1 ∧ ∥e2∥ = 1 ∧ (⟪e1, e2⟫ / (∥e1∥ * ∥e2∥) ≠ 0)

variables (e1 e2 : α) (θ : ℝ) (h_unit_vectors : unit_vectors e1 e2)
          (x1 x2 y1 y2 : ℝ) (a b : α)

axiom vector_a_def : a = x1 • e1 + y1 • e2
axiom vector_b_def : b = x2 • e1 + y2 • e2

theorem oblique_coordinates_properties :
  (a - b = (x1 - x2) • e1 + (y1 - y2) • e2) ∧
  (∀ λ : ℝ, λ • a = λ * x1 • e1 + λ * y1 • e2) :=
sorry

end oblique_coordinates_properties_l72_72774


namespace seating_unique_ways_l72_72432

theorem seating_unique_ways (n : Nat) (h : n = 7) :
  (n.factorial / n) = 720 := by
  rw [h]
  simp
  exact Nat.factorial_succ 6

end seating_unique_ways_l72_72432


namespace find_hyperbola_equation_l72_72731

noncomputable theory

def foci (h : {x // x^2 / 16 - y^2 / 9 = (1 : ℝ)}) : set ℝ := {5, -5}

def hyperbola_condition (a b : ℝ) : Prop :=
  (a^2 + b^2 = 25) ∧
  (5 / (4 * a^2) - 6 / b^2 = 1)

theorem find_hyperbola_equation :
  ∃ (a b : ℝ), hyperbola_condition a b ∧
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 24 = 1) :=
begin
  use [1, 24],
  split,
  { simp [hyperbola_condition],
    norm_num,
    field_simp,
    ring },
  sorry
end

end find_hyperbola_equation_l72_72731


namespace math_problem_l72_72050

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end math_problem_l72_72050


namespace sum_of_possible_remainders_l72_72549

theorem sum_of_possible_remainders :
  ∀ n : ℕ, 
  (∀ (k : ℕ), n = 1000*k + 100*(k+2) + 10*(k+4) + (k+6) ∧ 0 ≤ k ∧ k+6 ≤ 9) →
  ∑ (h : ℕ) in finset.filter (λ k, n = 1111*k + 206 ∧ k ≤ 3) (finset.range 4), (n % 31) = 35 := 
by sorry

end sum_of_possible_remainders_l72_72549


namespace select_three_numbers_l72_72152

open Finset

theorem select_three_numbers :
  let s := range 15
  ∑ (a1 in s.filter (λ a1, a1 < 12)) (a2 in s.filter (λ a2, a1 + 3 ≤ a2 ∧ a2 < 14)) (a3 in s.filter (λ a3, a2 + 2 ≤ a3)), 1 = 165 :=
by
  sorry

end select_three_numbers_l72_72152


namespace sum_of_distinct_prime_factors_of_2550_l72_72621

theorem sum_of_distinct_prime_factors_of_2550 :
  let distinct_prime_factors := {2, 3, 5, 17}
  sum (distinct_prime_factors : set ℕ) = 27 := by
  sorry

end sum_of_distinct_prime_factors_of_2550_l72_72621


namespace problem_1_problem_2_problem_3_l72_72754

def ellipse_eq (m : ℝ) (x y : ℝ) : Prop := x^2 / (m + 1) + y^2 / m = 1
def line_eq (k x y : ℝ) : Prop := y = k * (x + 1)
def point_D : Prop := -1 = 0

theorem problem_1 (m k : ℝ) (M N : ℝ × ℝ) : 
  m = 1 → k = 1 → 
  (∀ (x y : ℝ), ellipse_eq 1 x y → line_eq 1 x y) →
  M = (0, 1) ∧ N = (-4/3, -1/3) :=
sorry

theorem problem_2 (m : ℝ) (λ μ : ℝ) (M N E D : ℝ × ℝ) :
  m = 2 → 
  ellipse_eq 2 (M.1) (M.2) ∧ ellipse_eq 2 (N.1) (N.2) →
  λ * (E.1 - D.1) = M.1 ∧ μ * (E.1 - D.1) = N.1 →
  λ + μ = 3 :=
sorry

theorem problem_3 (m : ℝ) (k2 : ℝ) (l_eq x y : ℝ) (M N F : ℝ × ℝ) : 
  m = 3 →
  let l : line_eq k2 x y := λ _, y = k2 * (x + 1) in
  (ellipse_eq 3 (M.1) (M.2) ∧ ellipse_eq 3 (N.1) (N.2)) →
  let incircle_area := 18 / 49 * π in
  let DF := 2 in
  |k2 * (M.1 - N.1)| = 12 * sqrt 2 / 7 →
  k2 = ±1 :=
sorry

end problem_1_problem_2_problem_3_l72_72754


namespace tony_water_drink_l72_72196

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end tony_water_drink_l72_72196


namespace derivative_at_neg_one_l72_72061

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

theorem derivative_at_neg_one : deriv f (-1) = -1 :=
by
  -- definition of the function
  -- proof of the statement
  sorry

end derivative_at_neg_one_l72_72061


namespace sum_of_squares_eq_l72_72462

theorem sum_of_squares_eq (a b c R OH : ℝ) (O: Type*) [circumcenter O] (H: Type*) [orthocenter H] (H1 : circumradius ABC = R) : 
  a^2 + b^2 + c^2 = 9 * R^2 - OH^2 := 
sorry

end sum_of_squares_eq_l72_72462


namespace path_count_l72_72308

structure PathNetwork :=
(fromM_to_A : ℕ)
(fromM_to_B : ℕ)
(fromM_to_E : ℕ)
(fromA_to_C : ℕ)
(fromA_to_D : ℕ)
(fromB_to_N : ℕ)
(fromB_to_C : ℕ)
(fromC_to_N : ℕ)
(fromD_to_N : ℕ)
(fromE_to_B : ℕ)
(fromE_to_D : ℕ)

theorem path_count (p : PathNetwork)
  (h1 : p.fromM_to_A = 1)
  (h2 : p.fromM_to_B = 1)
  (h3 : p.fromM_to_E = 1)
  (h4 : p.fromA_to_C = 1)
  (h5 : p.fromA_to_D = 1)
  (h6 : p.fromB_to_N = 1)
  (h7 : p.fromB_to_C = 1)
  (h8 : p.fromC_to_N = 1)
  (h9 : p.fromD_to_N = 1)
  (h10 : p.fromE_to_B = 1)
  (h11 : p.fromE_to_D = 1)
  :
  let paths_from_C_to_N := 1 in
  let paths_from_D_to_N := 1 in
  let paths_from_A_to_N := paths_from_C_to_N + paths_from_D_to_N in
  let paths_from_B_to_N := 1 + paths_from_A_to_N + paths_from_C_to_N in
  let paths_from_E_to_N := paths_from_B_to_N + paths_from_D_to_N in
  let total_paths := paths_from_A_to_N + paths_from_B_to_N + paths_from_E_to_N in
  total_paths = 11 :=
by
  -- Proof elided
  sorry

end path_count_l72_72308


namespace count_items_in_U_l72_72954

theorem count_items_in_U (A B U : Set α) (hAU : Uncountable U) (hA : A ⊆ U) (hB : B ⊆ U) 
  (nA : fintype.card A = 105) (nB : fintype.card B = 49) (nAB : fintype.card (A ∩ B) = 23) 
  (n_not_A_B : fintype.card (U \ (A ∪ B)) = 59) : fintype.card U = 190 :=
by
  -- We need to prove the cardinality of the universal set U:
  have h1 : fintype.card U = fintype.card (A ∪ B ∪ (U \ (A ∪ B))) := by
    sorry
  -- Using the principle of inclusion-exclusion allows to find this number:
  have h2 : fintype.card U = fintype.card A + fintype.card B - fintype.card (A ∩ B) + fintype.card (U \ (A ∪ B)) := by
    sorry
  -- Substituting the given values:
  have h3 : fintype.card U = 105 + 49 - 23 + 59 := by
    sorry
  -- Summing up the values to get the final result:
  have h4 : 105 + 49 - 23 + 59 = 190 := by
    sorry
  exact h4

end count_items_in_U_l72_72954


namespace book_distribution_l72_72955

theorem book_distribution (chinese_books : ℕ) (math_books : ℕ) (students : ℕ) (at_least_one : ∀ i : ℕ, i < students → ℕ → Bool) :
  chinese_books = 3 ∧ math_books = 1 ∧ students = 3 ∧ (∀ i, i < students → at_least_one i 1) →
  (∃ n, n = 9) :=
by
  intros hc
  cases hc with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  existsi 9,
  sorry

end book_distribution_l72_72955


namespace arithmetic_sequence_sum_l72_72795

theorem arithmetic_sequence_sum {a_n : ℕ → ℤ} (d : ℤ) (S : ℕ → ℤ) 
  (h_seq : ∀ n, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_condition : a_n 1 = 2 * a_n 3 - 3) : 
  S 9 = 27 :=
sorry

end arithmetic_sequence_sum_l72_72795


namespace monotonic_decreasing_interval_l72_72933

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l72_72933


namespace min_distance_to_water_all_trees_l72_72190

/-- Proof that the minimum distance Xiao Zhang must walk to water all 10 trees is 410 meters -/
def minimum_distance_to_water_trees (num_trees : ℕ) (distance_between_trees : ℕ) : ℕ := 
  (sorry) -- implementation to calculate the minimum distance

theorem min_distance_to_water_all_trees (num_trees distance_between_trees : ℕ) :
  num_trees = 10 → 
  distance_between_trees = 10 →
  minimum_distance_to_water_trees num_trees distance_between_trees = 410 :=
by
  intros h_num_trees h_distance_between_trees
  rw [h_num_trees, h_distance_between_trees]
  -- Add proof here that the distance is 410
  sorry

end min_distance_to_water_all_trees_l72_72190


namespace relationship_y1_y2_y3_l72_72024

noncomputable def y (x : ℝ) : ℝ := -(x - 2) ^ 2

def A : ℝ × ℝ := (-1, y (-1))
def B : ℝ × ℝ := (1, y (1))
def C : ℝ × ℝ := (4, y (4))

theorem relationship_y1_y2_y3 :
  let y1 := y (-1)
  let y2 := y (1)
  let y3 := y (4)
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_y1_y2_y3_l72_72024


namespace functional_equation_solution_l72_72695

theorem functional_equation_solution (f : ℝ → ℝ) 
  (hcont : ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), continuous_at f x)
  (feq : ∀ x y : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) → y ∈ Icc (-1 : ℝ) (1 : ℝ) → x + y ∈ Icc (-1 : ℝ) (1 : ℝ) → f (x + y) = (f x + f y) / (1 - f x * f y)) :
  ∃ a : ℝ, abs a ≤ π / 2 ∧ (∀ x, f x = real.tan (a * x)) :=
sorry

end functional_equation_solution_l72_72695


namespace sara_movie_tickets_l72_72862

theorem sara_movie_tickets (T : ℕ) (h1 : 10.62 * T + 1.59 + 13.95 = 36.78) : T = 2 :=
sorry

end sara_movie_tickets_l72_72862


namespace highest_throw_is_37_feet_l72_72685

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l72_72685


namespace find_a_l72_72021

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * x^2 + a * x + 2

-- Define the derivative of the function
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 6 * x + a

-- Define the tangent line equation at point (0,2)
def tangent_line (a : ℝ) (x : ℝ) : ℝ := a * x + 2

theorem find_a (a : ℝ) :
  (∀ x, f' a 0 = a) ∧ (tangent_line a (-2) = 0) → a = 1 :=
by {
  sorry
}

end find_a_l72_72021


namespace log4_T_l72_72116

noncomputable def T : ℝ :=
  let f := (1 + 2 * complex.I * x) ^ 2011 in
  (f.coeffs ℝ).sum

theorem log4_T (T : ℝ) (hT : T = (1 + 2 * complex.I).geom_sum 2011 + (1 - 2 * complex.I).geom_sum 2011 / 2) :
  log 4 T = 502.5 :=
  sorry

end log4_T_l72_72116


namespace tangent_ellipse_hyperbola_l72_72165

theorem tangent_ellipse_hyperbola (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 4) → m = 5 / 9 :=
by
  sorry

end tangent_ellipse_hyperbola_l72_72165


namespace smallest_angle_satisfying_trig_eqn_l72_72304

theorem smallest_angle_satisfying_trig_eqn :
  ∃ x : ℝ, 0 < x ∧ 8 * (Real.sin x)^2 * (Real.cos x)^4 - 8 * (Real.sin x)^4 * (Real.cos x)^2 = 1 ∧ x = 10 :=
by
  sorry

end smallest_angle_satisfying_trig_eqn_l72_72304


namespace hyperbola_parameters_sum_l72_72843

theorem hyperbola_parameters_sum :
  let F1 := (-2, 2 - real.sqrt 3 / 2)
  let F2 := (-2, 2 + real.sqrt 3 / 2)
  let h := -2
  let k := 2
  let a := real.sqrt 2 / 2
  let c := real.sqrt 3 / 2
  let b := real.sqrt (c * c - a * a)
  (b > 0)
  (|F1.1 - h| = 0 → |F1.2 - k| = real.sqrt 3 / 2)
  (|F2.1 - h| = 0 → |F2.2 - k| = real.sqrt 3 / 2)
  (h + k + a + b = (real.sqrt 2 + 1) / 2) :=
begin
  sorry
end

end hyperbola_parameters_sum_l72_72843


namespace percentage_marks_D_l72_72291

-- Definitions based on conditions:
def full_marks : ℕ := 500
def A_marks : ℕ := 360

-- Define the relationships given in the conditions:
def B_marks := A_marks / 0.90
def C_marks := B_marks / 1.25
def D_marks := C_marks / 0.80

-- The theorem to prove the percentage of full marks obtained by D
theorem percentage_marks_D : (D_marks / full_marks) * 100 = 80 :=
by
  sorry

end percentage_marks_D_l72_72291


namespace total_toothpicks_after_removal_l72_72582

/-- Total number of toothpicks calculation -/
theorem total_toothpicks_after_removal (original_length : ℕ) (original_width : ℕ) (remove_length : ℕ) (remove_width : ℕ) :
  let total_original := (original_length + 1) * original_width + (original_width + 1) * original_length
      total_removed := (remove_length + 1) * remove_width + (remove_width + 1) * remove_length
  in original_length = 70 ∧ original_width = 45 ∧ remove_length = 5 ∧ remove_width = 5 →
     (total_original - total_removed) = 6295 :=
by
  intros original_length original_width remove_length remove_width
  intro h
  let total_original := (original_length + 1) * original_width + (original_width + 1) * original_length
  let total_removed := (remove_length + 1) * remove_width + (remove_width + 1) * remove_length
  rw [and_assoc, ←and_assoc] at h
  cases h with h_length h_rest1
  cases h_rest1 with h_width h_rest2
  cases h_rest2 with h_remove_length h_remove_width
  rw [h_length, h_width, h_remove_length, h_remove_width]
  sorry

end total_toothpicks_after_removal_l72_72582


namespace multiplicative_inverse_of_550_mod_4319_l72_72763

theorem multiplicative_inverse_of_550_mod_4319 :
  (48^2 + 275^2 = 277^2) → ((550 * 2208) % 4319 = 1) := by
  intro h
  sorry

end multiplicative_inverse_of_550_mod_4319_l72_72763


namespace real_number_a_l72_72367

theorem real_number_a (a : ℝ) (i : ℂ) (h : |(a-2 : ℂ) + ((4+3*i) / (1+2*i))| = real.sqrt 3 * a) : 
  a = real.sqrt 2 / 2 :=
by
  sorry

end real_number_a_l72_72367


namespace average_salary_of_technicians_l72_72430

theorem average_salary_of_technicians:
  (total_workers avg_salary_all avg_salary_non_tech : ℕ)
  (workers_tech workers_non_tech : ℕ)
  (total_salary : ℕ) :
  total_workers = 12 →
  avg_salary_all = 9000 →
  avg_salary_non_tech = 6000 →
  workers_tech = 6 →
  workers_non_tech = 6 →
  total_salary = total_workers * avg_salary_all →
  (
    (total_salary - (workers_non_tech * avg_salary_non_tech)) / workers_tech
  ) = 12000 :=
begin
  -- Proof goes here
  sorry
end

end average_salary_of_technicians_l72_72430


namespace smallest_m_for_integral_roots_l72_72983

theorem smallest_m_for_integral_roots :
  ∃ (m : ℕ), (∃ (p q : ℤ), p * q = 30 ∧ m = 12 * (p + q)) ∧ m = 132 := by
  sorry

end smallest_m_for_integral_roots_l72_72983


namespace larger_number_is_sixty_three_l72_72185

theorem larger_number_is_sixty_three (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : y = 63 :=
  sorry

end larger_number_is_sixty_three_l72_72185


namespace monotonic_decreasing_interval_l72_72928

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∃ t ∈ Ioo (-1 : ℝ) 11, t = x} ⊆ {x : ℝ | ∃ t, f'(t) < 0} :=
sorry

end monotonic_decreasing_interval_l72_72928


namespace probability_solution_l72_72491

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l72_72491


namespace additional_spending_required_l72_72962

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end additional_spending_required_l72_72962


namespace exists_term_lt_sqrt_l72_72833

-- Define the problem with all necessary conditions and show there exists a term a_i less than sqrt(m)
theorem exists_term_lt_sqrt {m : ℕ} {a : ℕ → ℕ} (h0 : ∀ i < n, a i ∣ m) 
  (h_diff : a 0 ≠ a n) (h_seq : ∀ i, 1 ≤ i ∧ i < n → a (i + 1) = |a i - a (i - 1)|) 
  (h_gcd : Nat.gcd (Finset.range (n + 1)).gcd (λ i, a i) = 1) :
  ∃ i, a i < Nat.sqrt m := 
sorry

end exists_term_lt_sqrt_l72_72833


namespace stops_time_proof_l72_72224

variable (departure_time arrival_time driving_time stop_time_in_minutes : ℕ)
variable (h_departure : departure_time = 7 * 60)
variable (h_arrival : arrival_time = 20 * 60)
variable (h_driving : driving_time = 12 * 60)
variable (total_minutes := arrival_time - departure_time)

theorem stops_time_proof :
  stop_time_in_minutes = (total_minutes - driving_time) := by
  sorry

end stops_time_proof_l72_72224


namespace farmer_milk_production_l72_72249

-- Definitions based on conditions
def total_cows (c : ℕ) : Prop := 0.4 * c = 50
def female_cows (c : ℕ) : ℕ := (0.6 * c).toNat
def milk_per_day (f : ℕ) : ℕ := 2 * f

-- Theorem to prove the farmer gets 150 gallons of milk a day
theorem farmer_milk_production : ∀ (c : ℕ), total_cows c → milk_per_day (female_cows c) = 150 := by
  intros c hc
  sorry

end farmer_milk_production_l72_72249


namespace circle_center_and_radius_l72_72696

def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2) ^ 2 + y ^ 2 = 4) →
  (exists (h k r : ℝ), (h, k) = (2, 0) ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l72_72696


namespace water_level_after_valve_opened_l72_72202

-- Given conditions
def h : ℝ := 40  -- initial height in cm
def ρ_water : ℝ := 1000  -- density of water in kg/m^3
def ρ_oil : ℝ := 700  -- density of oil in kg/m^3

-- Lean statement to prove
theorem water_level_after_valve_opened :
  let h1 := (ρ_oil * h) / (ρ_water + ρ_oil) in
  h1 = 280 / 17 :=
by
  sorry

end water_level_after_valve_opened_l72_72202


namespace duchess_is_thief_l72_72626

/-- The possible suspects. -/
inductive Suspect
| Duchess
| Cheshire_Cat
| Cook

open Suspect

/-- The Duchess's statement is that Cheshire Cat stole the cookbook. -/
def Duchess_statement := Cheshire_Cat = (stole : Suspect)

/-- The Cheshire Cat admits to stealing the cookbook. -/
def Cheshire_Cat_statement := Cheshire_Cat = stole

/-- The Cook asserts they did not steal the cookbook. -/
def Cook_statement := Cook ≠ stole

/-- The thief is lying about being the thief. -/
def thief_lies (s : Suspect) : Prop :=
  (s = Duchess → ¬ Duchess_statement) ∧ 
  (s = Cheshire_Cat → ¬ Cheshire_Cat_statement) ∧ 
  (s = Cook → ¬ Cook_statement)

/-- At least one non-thief tells the truth. -/
def non_thief_truth (s : Suspect) : Prop :=
  (s ≠ Duchess → Duchess_statement ∨ Cheshire_Cat_statement ∨ Cook_statement) ∧
  (s ≠ Cheshire_Cat → Duchess_statement ∨ Cheshire_Cat_statement ∨ Cook_statement) ∧
  (s ≠ Cook → Duchess_statement ∨ Cheshire_Cat_statement ∨ Cook_statement)

/-- The main theorem to prove: The Duchess is the thief. -/
theorem duchess_is_thief :
  ∃ (s : Suspect), thief_lies s ∧ non_thief_truth s ∧ s = Duchess := 
sorry

end duchess_is_thief_l72_72626


namespace part1_width_of_tunnel_part2_minimize_earthwork_l72_72690

-- Assume a coordinate system with ellipse parameters
def ellipse_eq (a b x y : ℝ) : Prop := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1
def point_on_ellipse (a b : ℝ) : Prop := ellipse_eq a b 11 4.5

theorem part1_width_of_tunnel :
  ∀ a b : ℝ,
  b = 6 → point_on_ellipse a b → 2 * a = 33.3 :=
by
  sorry

theorem part2_minimize_earthwork :
  ∀ a b : ℝ,
  b ≥ 6 → point_on_ellipse a b → (a := 15.55 ∧ b = 6.4) :=
by
  sorry

end part1_width_of_tunnel_part2_minimize_earthwork_l72_72690


namespace angle_between_vectors_l72_72005

variables (a b : ℝ^3) (θ : ℝ)

-- Declare the conditions given in the problem
def condition1 : ∥a∥ = 1 := sorry
def condition2 : ∥b∥ = 1 := sorry
def condition3 : a ⬝ b = 1 / 2 := sorry

-- The main theorem to prove
theorem angle_between_vectors (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : a ⬝ b = 1 / 2) : θ = π / 3 := 
sorry

end angle_between_vectors_l72_72005


namespace third_measurement_multiple_of_one_l72_72554

-- Define the lengths in meters
def length1_meter : ℕ := 6
def length2_meter : ℕ := 5

-- Convert lengths to centimeters
def length1_cm := length1_meter * 100
def length2_cm := length2_meter * 100

-- Define that the greatest common divisor (gcd) of lengths in cm is 100 cm
def gcd_length : ℕ := Nat.gcd length1_cm length2_cm

-- Given that the gcd is 100 cm
theorem third_measurement_multiple_of_one
  (h1 : gcd_length = 100) :
  ∃ n : ℕ, n = 1 :=
sorry

end third_measurement_multiple_of_one_l72_72554


namespace triangle_centroid_value_l72_72969

-- Define the coordinates of the points P, Q, and R
def P : (ℝ × ℝ) := (2, 3)
def Q : (ℝ × ℝ) := (-1, -6)
def R : (ℝ × ℝ) := (7, 0)

-- Compute the centroid S of triangle PQR
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def S : ℝ × ℝ := centroid P Q R

-- Calculate 10x + y for the centroid S
def ten_x_plus_y (S : ℝ × ℝ) : ℝ :=
  10 * S.1 + S.2

-- Statement of the theorem
theorem triangle_centroid_value :
  ten_x_plus_y S = 77 / 3 :=
by
  sorry

end triangle_centroid_value_l72_72969


namespace elaine_earnings_l72_72455

variable (E P : ℝ)
variable (H1 : 0.30 * E * (1 + P / 100) = 2.025 * 0.20 * E)

theorem elaine_earnings : P = 35 :=
by
  -- We assume the conditions here and the proof is skipped by sorry.
  sorry

end elaine_earnings_l72_72455


namespace points_relation_on_parabola_l72_72025

theorem points_relation_on_parabola :
  let f (x : ℝ) := -(x - 2) ^ 2 in
  let y1 := f (-1) in
  let y2 := f 1 in
  let y3 := f 4 in
  y1 < y3 ∧ y3 < y2 :=
by
  -- Proof to be completed
  sorry

end points_relation_on_parabola_l72_72025


namespace monotonic_decreasing_interval_l72_72932

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l72_72932


namespace sum_first_n_terms_l72_72393

open Nat

def a : ℕ → ℕ
| 0 := 0  -- Note: The sequence starts from a_1 = 8 at n=1
| 1 := 8
| (n+1) := a n + 2^(n+1)

noncomputable def S (n : ℕ) : ℕ := ∑ i in range (n+1), a i

theorem sum_first_n_terms (n : ℕ) : S n = 2^(n+2) + 4 * n - 4 :=
by sorry

end sum_first_n_terms_l72_72393


namespace count_nonnegative_integer_solutions_eq_one_l72_72403

open Set

noncomputable def count_nonnegative_solutions : ℕ :=
  {x : ℕ | x^2 + 3 * x - 18 = 0}.toFinset.card

theorem count_nonnegative_integer_solutions_eq_one :
  count_nonnegative_solutions = 1 := 
sorry

end count_nonnegative_integer_solutions_eq_one_l72_72403


namespace locus_of_P_is_circular_arc_l72_72237

variable (A B C M N P : Type)
variable [IsoscelesTriangle A B C] (BC AB AC AM AN BN CM b a : ℝ)
variable (BM CN : Line)
variable [Intersects BM CN P]

theorem locus_of_P_is_circular_arc :
  BC = a → AB = b → AC = b → 
  a^2 * AM * AN = b^2 * BN * CM →
  ∃ (arc : Arc), 
    arc.chord = BC ∧ 
    arc.subtendedAngle = 180 - ∠ ABC ∧ 
    P ∈ arc :=
sorry

end locus_of_P_is_circular_arc_l72_72237


namespace wall_width_is_correct_l72_72258

-- Definitions based on the conditions
def brick_length : ℝ := 25  -- in cm
def brick_height : ℝ := 11.25  -- in cm
def brick_width : ℝ := 6  -- in cm
def num_bricks : ℝ := 5600
def wall_length : ℝ := 700  -- 7 m in cm
def wall_height : ℝ := 600  -- 6 m in cm
def total_volume : ℝ := num_bricks * (brick_length * brick_height * brick_width)

-- Prove that the inferred width of the wall is correct
theorem wall_width_is_correct : (total_volume / (wall_length * wall_height)) = 22.5 := by
  sorry

end wall_width_is_correct_l72_72258


namespace committee_with_treasurer_l72_72074

theorem committee_with_treasurer {α : Type} (club : Finset α) (T : α) (hT : T ∈ club) (h_card : club.card = 12) :
  (∃ S : Finset α, S.card = 5 ∧ T ∈ S) ↔ (Fintype.card {S : Finset α // S.card = 5 ∧ T ∈ S} = 330) :=
by
  sorry

end committee_with_treasurer_l72_72074


namespace exists_points_X_Y_l72_72309

theorem exists_points_X_Y (A B C X Y : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB : MetricSegment A B) (BC : MetricSegment B C) (AC : MetricSegment A C) :
  ∃ X Y, (X ∈ AB) ∧ (Y ∈ BC) ∧ (dist A X = dist B Y) ∧ Parallels (MetricSegment (X, Y)) (MetricSegment (A, C)) := 
sorry

end exists_points_X_Y_l72_72309


namespace intersection_of_S_and_T_l72_72767

-- Define S and T based on given conditions
def S : Set ℝ := { x | x^2 + 2 * x = 0 }
def T : Set ℝ := { x | x^2 - 2 * x = 0 }

-- Prove the intersection of S and T
theorem intersection_of_S_and_T : S ∩ T = {0} :=
sorry

end intersection_of_S_and_T_l72_72767


namespace fifty_times_reciprocal_of_eight_times_number_three_l72_72788

theorem fifty_times_reciprocal_of_eight_times_number_three (x : ℚ) 
  (h : 8 * x = 3) : 50 * (1 / x) = 133 + 1 / 3 :=
sorry

end fifty_times_reciprocal_of_eight_times_number_three_l72_72788


namespace circle_path_length_l72_72992

/--
Given a triangle ABC with sides of lengths 6, 8, and 10, and a circle
with radius 2 centered at P that rolls around the inside of the triangle.
When P first returns to its original position, the distance it has traveled is 8.
-/
theorem circle_path_length (A B C P : Type) [MetricSpace A B] [MetricSpace B C] 
  [MetricSpace C A] [MetricSpace A P] [MetricSpace B P] [MetricSpace C P]
  (radius : ℝ) (side1 side2 side3 : ℝ)
  (h_side1 : side1 = 6) (h_side2 : side2 = 8) 
  (h_side3 : side3 = 10)
  (h_radius : radius = 2) : 
  let triangle := triangle (A, B, C) in
  let circle := circle P radius in
  path_length P side1 side2 side3 radius = 8 := 
begin
  sorry
end

end circle_path_length_l72_72992


namespace sum_first_2016_terms_l72_72123

-- Define sequence and its sum
def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * n

def S (n : ℕ) : ℕ :=
  nat.rec_on n 0 (λ k IH, IH + sequence_a (k+1))

-- Given conditions
axiom Sn_relationship (n : ℕ) : 
  S (n + 1) = (n + 2) / (n + 1) * S n

-- Define the sequence of interest
def sequence_b (n : ℕ) : ℝ :=
  1 / (sequence_a n * sequence_a (n + 1))

-- Define the sum of the first 2016 terms of the sequence_b
def sum_sequence_b (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ k, sequence_b (k + 1))

-- The theorem to prove
theorem sum_first_2016_terms : 
  sum_sequence_b 2016 = 504 / 2017 :=
sorry

end sum_first_2016_terms_l72_72123


namespace solve_log_equation_l72_72898

theorem solve_log_equation : ∃ x : ℝ, (log (2 * (x + 10)) - log ((x + 10)^3) = 4) ∧ (x = 9990 ∨ x = -9.9) := 
by
  sorry

end solve_log_equation_l72_72898


namespace max_ab_value_l72_72410

theorem max_ab_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∃ x : ℝ, x = 0 ∧ ∀ y : ℝ, f y = a * exp y + (b^2 - 3) * y ∧ has_deriv_at f (a * exp x + b^2 - 3) x):
  ab = 2 :=
by 
  -- Define the function f(x)
  let f := λ x, a * exp x + (b^2 - 3) * x
  -- Compute the derivative of f(x)
  have deriv_f : ∀ x, deriv f x = a * exp x + (b^2 - 3) := sorry
  -- Use the condition that f attains an extreme value at x = 0
  have h4 : deriv f 0 = a + b^2 - 3 := sorry
  -- Substitute in the extreme value condition
  have h5 : a + b^2 - 3 = 0 := sorry
  -- Solve for a in terms of b
  have h6 : a = 3 - b^2 := by linarith
  -- Define g(b) = (3 - b^2) * b
  let g := λ b, (3 - b^2) * b
  -- Prove that the maximum value of g(b) occurs at b = 1
  have max_g : ∀ b ∈ Ioo (0 : ℝ) (sqrt 3), g(b) ≤ g(1) := sorry
  -- Since g(1) = 2, conclude that the maximum value of ab is 2
  have h7 : g(1) = 2 := by norm_num
  show ab = 2, from sorry

end max_ab_value_l72_72410


namespace odd_power_sum_mod_eight_l72_72784

theorem odd_power_sum_mod_eight (n : ℕ) (hn : n % 2 = 1) (hpos : 0 < n) :
  (6 ^ n + ∑ i in Finset.range n, Nat.choose n i * 6^(n-i)) % 8 = 6 :=
by
  sorry

end odd_power_sum_mod_eight_l72_72784


namespace find_m_l72_72034

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

-- The intersection condition
def intersect_condition (m : ℝ) : Prop := A m ∩ B = {3}

-- The statement to prove
theorem find_m : ∃ m : ℝ, intersect_condition m → m = 3 :=
by {
  use 3,
  sorry
}

end find_m_l72_72034


namespace vector_addition_l72_72705

theorem vector_addition : 
  (\(\begin{pmatrix} 5 \\ -3 \end{pmatrix} + \(\begin{pmatrix} -8 \\ 14 \end{pmatrix}) = \(\begin{pmatrix} -3 \\ 11 \end{pmatrix}) :=
by
  sorry

end vector_addition_l72_72705


namespace upper_limit_l72_72421

noncomputable def upper_limit_Arun (w : ℝ) (X : ℝ) : Prop :=
  (w > 66 ∧ w < X) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 69) ∧ ((66 + X) / 2 = 68)

theorem upper_limit (w : ℝ) (X : ℝ) (h : upper_limit_Arun w X) : X = 69 :=
by sorry

end upper_limit_l72_72421


namespace hyperbola_standard_equation_l72_72923

theorem hyperbola_standard_equation (a b : ℝ) (h₁ : (sqrt 2, sqrt 3) ∈ {p : ℝ × ℝ | (p.2^2 / a^2) - (p.1^2 / b^2) = 1}) 
  (h₂ : sqrt 2 = real.sqrt (1 + (b / a)^2)) : 
  a = 1 ∧ b = 1 ∧ ∀ x y : ℝ, y^2 - x^2 = 1 → (y^2 / a^2) - (x^2 / b^2) = 1 :=
by
  sorry

end hyperbola_standard_equation_l72_72923


namespace building_height_l72_72997

theorem building_height
  (H_f : ℝ) (L_f : ℝ) (L_b : ℝ)
  (H_f_nonneg : 0 < H_f)
  (L_f_nonneg : 0 < L_f)
  (L_b_nonneg : 0 < L_b)
  (ratio_eq : H_f / L_f = H_f / L_f * (L_b / 28.75)) :
  ∃ H_b : ℝ, H_b ≈ 12.44 :=
by
  let H_b := (H_f / L_f) * L_b
  have : 12.44 ≈ (17.5 / 40.25) * 28.75,
  sorry

end building_height_l72_72997


namespace stripe_length_l72_72647

theorem stripe_length
  (cylinder_circumference : ℝ)
  (c : cylinder_circumference = 10)
  (cylinder_height : ℝ)
  (h : cylinder_height = 24)
  (spiral_turns : ℕ)
  (t : spiral_turns = 2)
  : (real.sqrt (cylinder_circumference^2 + (cylinder_height * spiral_turns)^2) = 49) :=
by
  sorry

end stripe_length_l72_72647


namespace fg_midpoint_mn_l72_72539

open Lean

theorem fg_midpoint_mn 
  {A B C D E F G M N : Point}
  (O O1 : Circle)
  (h1 : tangent A (O.touching B C M))
  (h2 : on_line D E A B)
  (h3 : on_line D E A C)
  (h4 : parallel D E B C)
  (h5 : tangent ADE_O1.touches D E N)
  (h6 : intersection (line B O1) (line D O) = F)
  (h7 : intersection (line C O1) (line E O) = G) :
  mid_point (line F G) (line M N) :=
by
  sorry

end fg_midpoint_mn_l72_72539


namespace range_of_k_l72_72053

theorem range_of_k 
  (h : ∀ x : ℝ, x^2 + 2 * k * x - (k - 2) > 0) : -2 < k ∧ k < 1 := 
sorry

end range_of_k_l72_72053


namespace solution_set_of_inequality_l72_72913

section

variable (f : ℝ → ℝ)
variable (hf_domain : ∀ x, f x ∈ ℝ)
variable (hf_at_neg2 : f (-2) = 2018)
variable (h_derivative : ∀ x, deriv f x < 2 * x)

def g (x : ℝ) : ℝ := f x - x^2 - 2014

theorem solution_set_of_inequality :
  {x : ℝ | f x < x^2 + 2014} = Ioi (-2) := by
  sorry

end

end solution_set_of_inequality_l72_72913


namespace lines_ell_and_n_on_same_plane_l72_72113

-- Definitions for the given conditions
def hexagonal_pyramid (O A B C D E F : Type) : Prop := 
  -- Define the structure of the hexagonal pyramid here
  sorry

def circumscribed_around_sphere (base: Type) (sphere: Type) : Prop := 
  -- Define the property of the base being circumscribed around the sphere here
  sorry

def tangency_plane (A1 B1 C1 D1 E1 F1 : Type) : Prop := 
  -- Define the property of the tangency plane passing through tangency points
  sorry

def lines_on_same_plane (ℓ m n : Type) : Prop := 
  -- Define the lines ℓ, m, n lying on the same plane
  sorry

-- Theorem statement
theorem lines_ell_and_n_on_same_plane 
  (O A B C D E F A1 B1 C1 D1 E1 F1 ℓ m n : Type)
  (hpyramid : hexagonal_pyramid O A B C D E F)
  (hcircum : circumscribed_around_sphere (A, B, C, D, E, F) ω)
  (htangency : tangency_plane A1 B1 C1 D1 E1 F1)
  (hlnoplane : lines_on_same_plane ℓ m n ⨯ lines_on_same_plane m n) :
  lines_on_same_plane ℓ n :=
sorry

end lines_ell_and_n_on_same_plane_l72_72113


namespace sum_of_possible_M_values_l72_72177

theorem sum_of_possible_M_values :
  ∃ (x y z : ℕ), z = x + y ∧ xyz = 4 * (x + y + z) ∧ (set.sum (set_of_possible_M_values) = 120) := by sorry

end sum_of_possible_M_values_l72_72177


namespace megatek_manufacturing_percentage_proof_l72_72157

def megatek_employee_percentage
  (total_degrees_in_circle : ℕ)
  (manufacturing_degrees : ℕ) : ℚ :=
  (manufacturing_degrees / total_degrees_in_circle : ℚ) * 100

theorem megatek_manufacturing_percentage_proof (h1 : total_degrees_in_circle = 360)
  (h2 : manufacturing_degrees = 54) :
  megatek_employee_percentage total_degrees_in_circle manufacturing_degrees = 15 := 
by
  sorry

end megatek_manufacturing_percentage_proof_l72_72157


namespace largest_prime_factor_of_4519_l72_72212

theorem largest_prime_factor_of_4519 : 
  ∃ p : ℕ, prime p ∧ p ∣ 4519 ∧ ∀ q : ℕ, prime q ∧ q ∣ 4519 → q ≤ p :=
begin
  sorry
end

end largest_prime_factor_of_4519_l72_72212


namespace quadratic_intersects_once_l72_72338

theorem quadratic_intersects_once (c : ℝ) : (∀ x : ℝ, x^2 - 6 * x + c = 0 → x = 3 ) ↔ c = 9 :=
by
  sorry

end quadratic_intersects_once_l72_72338


namespace percentage_increase_shannon_l72_72508

noncomputable def shannon_formula (W S N : ℝ) : ℝ := W * (Real.log (1 + S / N) / Real.log 2)

noncomputable def simplified_shannon_formula (W S N : ℝ) : ℝ := W * (Real.log (S / N) / Real.log 2)

theorem percentage_increase_shannon (W : ℝ) (log2_10 : ℝ) (log2_approx : Real.log 2 ≈ 0.301)
  (hno : log2_10 ≈ 3.32193) (Wi: ℝ) (S_ini N_ini S_fin N_fin : ℝ)
  (h1 : S_ini = 1000) (h2 : N_ini = 1) (h3 : S_fin = 5000) (h4 : N_fin = 1)
  (hW : Wi = W) :
  let C_i := simplified_shannon_formula Wi S_ini N_ini in
  let C_f := simplified_shannon_formula Wi S_fin N_fin in
  let percentage_increase := (C_f - C_i) / C_i * 100 in
  percentage_increase ≈ 23 := 
sorry

end percentage_increase_shannon_l72_72508


namespace percentage_increase_formula_l72_72154

theorem percentage_increase_formula (A B C : ℝ) (h1 : A = 3 * B) (h2 : C = B - 30) :
  100 * ((A - C) / C) = 200 + 9000 / C := 
by 
  sorry

end percentage_increase_formula_l72_72154


namespace compare_f_m_plus_2_l72_72727

theorem compare_f_m_plus_2 (a : ℝ) (ha : a > 0) (m : ℝ) 
  (hf : (a * m^2 + 2 * a * m + 1) < 0) : 
  (a * (m + 2)^2 + 2 * a * (m + 2) + 1) > 1 :=
sorry

end compare_f_m_plus_2_l72_72727


namespace dozen_chocolate_chip_baked_proof_l72_72672

-- Definition of conditions
variables 
  (dozensOatmeal : ℕ) (dozensSugar : ℕ) (dozensChocolateBaked : ℕ) 
  (dozensOatmealGiven : ℕ) (dozensSugarGiven : ℕ) (dozensChocolateGiven : ℕ) 
  (cookiesKept : ℕ)

-- Setting the specific values for the conditions
def conditions := 
  dozensOatmeal = 3 ∧ dozensSugar = 2 ∧ dozensOatmealGiven = 2 ∧ 
  dozensSugarGiven = 1.5 ∧ dozensChocolateBaked * 12 + dozensSugar * 12 +
  dozensOatmeal * 12 - dozensOatmealGiven * 12 - dozensSugarGiven * 12 - 
  dozensChocolateGiven * 12 = cookiesKept ∧ cookiesKept = 36

-- Lean statement
theorem dozen_chocolate_chip_baked_proof : 
  conditions -> dozensChocolateBaked = 4 :=
sorry

end dozen_chocolate_chip_baked_proof_l72_72672


namespace area_CDE_l72_72870

variable (A B C D E F : Type) [RealInnerProductSpace ℝ A] [T2Space A]

def point_on_triangle_AC (D : A) (AC : Set A) (D_on_AC : D ∈ AC) := sorry
def point_on_triangle_BC (E : A) (BC : Set A) (E_on_BC : E ∈ BC) := sorry
def intersection_point (AE : Set A) (BD : Set A) (F : A) (F_int : F ∈ (AE ∩ BD)) := sorry

theorem area_CDE :
  (point_on_triangle_AC D (segment A C) sorry) →
  (point_on_triangle_BC E (segment B C) sorry) →
  (intersection_point (segment A E) (segment B D) F sorry) →
  (area_of_triangle A B F = 1) →
  (area_of_triangle A D F = 1 / 4) →
  (area_of_triangle B E F = 1 / 5) →
  area_of_triangle C D E = 3 / 38 := sorry

end area_CDE_l72_72870


namespace triangle_angle_ratios_l72_72945

theorem triangle_angle_ratios 
    (a b c : ℕ) 
    (h_ratio : (a, b, c) = (2, 4, 3)) 
    (h_sum : a + b + c = 9) 
    (h_total : 180 = 180) : 
  let largest_angle := 180 * 4 / 9 in
  let smallest_angle := 180 * 2 / 9 in
  largest_angle = 80 ∧ smallest_angle = 40 :=
by
  sorry

end triangle_angle_ratios_l72_72945


namespace number_condition_l72_72899

theorem number_condition (x : ℤ) (h : x - 7 = 9) : 5 * x = 80 := by
  sorry

end number_condition_l72_72899


namespace xiao_li_place_l72_72989

def guess_A (place : String) : Prop :=
  place ≠ "first" ∧ place ≠ "second"

def guess_B (place : String) : Prop :=
  place ≠ "first" ∧ place = "third"

def guess_C (place : String) : Prop :=
  place ≠ "third" ∧ place = "first"

def correct_guesses (guess : String → Prop) (place : String) : Prop :=
  guess place

def half_correct_guesses (guess : String → Prop) (place : String) : Prop :=
  (guess "first" = (place = "first")) ∨
  (guess "second" = (place = "second")) ∨
  (guess "third" = (place = "third"))

theorem xiao_li_place :
  ∃ (place : String),
  (correct_guesses guess_A place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_B place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_B place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_A place) :=
sorry

end xiao_li_place_l72_72989


namespace five_letter_word_count_l72_72095

theorem five_letter_word_count : 
  let num_letters := 26 in
  (num_letters ^ 1) * (num_letters ^ 3) = 456976 := 
by
  sorry

end five_letter_word_count_l72_72095


namespace compute_expression_l72_72303

theorem compute_expression (x : ℕ) (h : x = 3) : (x^8 + 8 * x^4 + 16) / (x^4 - 4) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l72_72303


namespace batsman_avg_after_17th_inning_l72_72423

variable A : ℕ -- definition of the initial average
variable runs_scored_in_17th : ℕ := 66   -- the runs he scored in the 17th inning
variable innings_played_before : ℕ := 16 -- innings before 17th inning
variable increased_avg : ℕ := 3 

theorem batsman_avg_after_17th_inning (A : ℕ) (runs_scored_in_17th : ℕ) (innings_played_before : ℕ) (increased_avg : ℕ) : 
  (A * innings_played_before + runs_scored_in_17th) / (innings_played_before + 1) = A + increased_avg → 
  A = 15 → 
  (A + increased_avg) = 18 := 
by 
  intro h
  intro ha
  sorry

end batsman_avg_after_17th_inning_l72_72423


namespace find_BC_in_quadrilateral_l72_72438

open Real

theorem find_BC_in_quadrilateral 
  (A B C D : Point) 
  (h_convex: ConvexQuadrilateral A B C D)
  (h_diagAC: Diagonal A C) 
  (h_diagBD: Diagonal B D)
  (h_AD: dist A D = 2)
  (h_angleABD: angle A B D = 90) 
  (h_angleACD: angle A C D = 90)
  (h_inscribedCentersDist: dist (inscribedCircleCenter A B D) (inscribedCircleCenter A C D) = sqrt 2) : 
  dist B C = sqrt 3 := 
sorry

end find_BC_in_quadrilateral_l72_72438


namespace squirrel_cannot_catch_nut_l72_72724

section SquirrelNutProblem

variable (a : ℝ := 3.75)  -- distance to squirrel in meters
variable (V₀ : ℝ := 2.5)  -- speed of the nut in m/s
variable (g : ℝ := 10)    -- acceleration due to gravity in m/s^2
variable (max_jump : ℝ := 2.7) -- maximum jump distance of the squirrel in meters

-- The function representing the squared distance between the squirrel and the nut
def distance_sq (t : ℝ) : ℝ :=
  (V₀ * t - a)^2 + (g * t^2 / 2)^2

theorem squirrel_cannot_catch_nut : 
  (∀ t, distance_sq V₀ a g t >= max_jump^2) :=
by
  sorry

end SquirrelNutProblem

end squirrel_cannot_catch_nut_l72_72724


namespace jellybean_problem_l72_72631

theorem jellybean_problem :
  ∃ (X Y : ℕ), X + Y = 1200 ∧ X = 3 * Y - 400 ∧ X = 800 :=
by
  -- Define the number of jelly beans in jars X and Y
  let X : ℕ := 800
  let Y : ℕ := 400
  exists X, Y
  split
  -- Verify the total number of jelly beans is 1200
  · exact by linarith
  split
  -- Verify the relationship between X and Y
  · exact by linarith
  -- Verify that X is indeed 800
  · exact rfl

end jellybean_problem_l72_72631


namespace gcd_of_polynomial_and_multiple_l72_72742

-- Definitions based on given conditions
def multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- The main statement of the problem
theorem gcd_of_polynomial_and_multiple (y : ℕ) (h : multiple_of y 56790) :
  Nat.gcd ((3 * y + 2) * (5 * y + 3) * (11 * y + 7) * (y + 17)) y = 714 :=
sorry

end gcd_of_polynomial_and_multiple_l72_72742


namespace books_shelves_l72_72663

def initial_books : ℝ := 40.0
def additional_books : ℝ := 20.0
def books_per_shelf : ℝ := 4.0

theorem books_shelves :
  (initial_books + additional_books) / books_per_shelf = 15 :=
by 
  sorry

end books_shelves_l72_72663


namespace min_sum_equals_nine_l72_72375

theorem min_sum_equals_nine (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 4 * a + b - a * b = 0) : a + b = 9 :=
by
  sorry

end min_sum_equals_nine_l72_72375


namespace count_is_68_l72_72140

def books : List String := ["Book of Songs", "Book of Documents", "Book of Rites", "Book of Changes", "Spring and Autumn Annals"]

def condition1 (permutation : List String) : Bool :=
  books.allDifferent permutation

def condition2 (permutation : List String) : Bool :=
  not ((permutation.indexOf! "Book of Songs" + 1 == permutation.indexOf! "Book of Rites") ∨ 
       (permutation.indexOf! "Book of Rites" + 1 == permutation.indexOf! "Book of Songs"))

def condition3 (permutation : List String) : Bool :=
  permutation.head ≠ "Book of Changes"

def valid_arrangements : List (List String) :=
  books.permutations.filter (λp => condition1 p ∧ condition2 p ∧ condition3 p)

def count_valid_arrangements : Nat :=
  valid_arrangements.length

theorem count_is_68 : count_valid_arrangements = 68 := by
  sorry

end count_is_68_l72_72140


namespace find_largest_n_l72_72336

def g (x : ℕ) : ℕ := 
  if x = 0 then 0 else 2 ^ (Nat.find (λ n, x % (2 ^ (n + 1)) ≠ 0) + 1)

def S_n (n : ℕ) : ℕ := 
  (List.range (2 ^ (n-1) + 1)).tail.sum (λ k, g (2 * k))

theorem find_largest_n : ∃ n < 1000, S_n n = (n + 1) * 2^(n - 1) ∧ is_square (S_n n) ∧ ∀ m < 1000, is_square (S_n m) → m ≤ n :=
  by
  let n := 511
  have h1 : n < 1000 := by decide
  have h2 : S_n n = (n + 1) * 2^(n - 1) := by sorry
  have h3 : is_square (S_n n) := by sorry
  have h4 : ∀ m < 1000, is_square (S_n m) → m ≤ n := by sorry
  exact ⟨n, h1, h2, h3, h4⟩

end find_largest_n_l72_72336


namespace find_f_at_1_l72_72387

noncomputable def f (f'3 : ℝ) (x : ℝ) : ℝ := 2 * f'3 * x - 2 * x^2 + 3 * Real.log x

theorem find_f_at_1 (f'3 : ℝ) (h : f'3 = 11) : f f'3 1 = 20 := by
  rw [h]
  simp [f, Real.log_one]
  norm_num
  sorry

end find_f_at_1_l72_72387


namespace median_BC_eq_area_ABC_eq_17_perp_bisector_BC_eq_l72_72948

-- Definitions of the points A, B, and C
def A : (ℝ × ℝ) := (4, 0)
def B : (ℝ × ℝ) := (6, 7)
def C : (ℝ × ℝ) := (0, 3)

-- 1. Prove the equation of the median on side BC is 5x + y - 20 = 0
theorem median_BC_eq :
  let E := (B.1 + C.1) / 2
  let F := (B.2 + C.2) / 2
  let M := (E, F)
  -- Point A is (4,0) and M is midpoint of BC
  (5 * x + y - 20 = 0) :=
by
  sorry

-- 2. Prove the area of triangle ABC is 17
theorem area_ABC_eq_17 :
  let AB := Math.sqrt((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Math.sqrt((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Math.sqrt((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + CA) / 2
  let area := Math.sqrt(s * (s - AB) * (s - BC) * (s - CA))
  (area = 17) :=
by 
  sorry

-- 3. Prove the equation of the perpendicular bisector of side BC is 3x + 2y - 19 = 0
theorem perp_bisector_BC_eq :
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let slopeBC := (C.2 - B.2) / (C.1 - B.1)
  let perp_slope := -1 / slopeBC
  -- Midpoint M is calculated, use point-slope form for perpendicular bisector
  (3 * x + 2 * y - 19 = 0) :=
by 
  sorry

end median_BC_eq_area_ABC_eq_17_perp_bisector_BC_eq_l72_72948


namespace gcd_pow_sum_l72_72841

open Nat

def gcd_expr (p q m n : ℕ) : ℕ :=
  gcd (p^m + q^m) (p^n + q^n)

theorem gcd_pow_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (h_gcd : gcd m n = 1)
  (p q : ℕ) (h_primep : Prime p) (h_primeq : Prime q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) (h_distinct: p ≠ q) :
  gcd_expr p q m n =
  if (m + n) % 2 = 1 then
    2
  else
    p + q :=
sorry

end gcd_pow_sum_l72_72841


namespace smallest_angle_l72_72808

theorem smallest_angle (largest_angle : ℝ) (a b : ℝ) (h1 : largest_angle = 120) (h2 : 3 * a = 2 * b) (h3 : largest_angle + a + b = 180) : b = 24 := by
  sorry

end smallest_angle_l72_72808


namespace f_neg_15_pi_over_4_eq_l72_72469

noncomputable def f (x : ℝ) : ℝ :=
if -π / 2 ≤ x ∧ x ≤ 0 then cos x
else if 0 ≤ x ∧ x ≤ π then sin x
else 0

theorem f_neg_15_pi_over_4_eq : f (-15 * π / 4) = sqrt 2 / 2 :=
by
  sorry

end f_neg_15_pi_over_4_eq_l72_72469


namespace determine_a_zeros_l72_72383

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x = 3 then a else 2 / |x - 3|

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

theorem determine_a_zeros (a : ℝ) : (∃ c d, c ≠ 3 ∧ d ≠ 3 ∧ c ≠ d ∧ y c a = 0 ∧ y d a = 0 ∧ y 3 a = 0) → a = 4 :=
sorry

end determine_a_zeros_l72_72383


namespace sqrt_2023_irrational_l72_72220

theorem sqrt_2023_irrational : ¬ ∃ (r : ℚ), r^2 = 2023 := by
  sorry

end sqrt_2023_irrational_l72_72220


namespace mirasol_account_balance_l72_72126

theorem mirasol_account_balance :
  ∀ (initial_amount spent_coffee spent_tumbler : ℕ), 
  initial_amount = 50 → 
  spent_coffee = 10 → 
  spent_tumbler = 30 → 
  initial_amount - (spent_coffee + spent_tumbler) = 10 :=
by
  intros initial_amount spent_coffee spent_tumbler
  intro h_initial_amount
  intro h_spent_coffee
  intro h_spent_tumbler
  rw [h_initial_amount, h_spent_coffee, h_spent_tumbler]
  simp
  done

end mirasol_account_balance_l72_72126


namespace check_conclusions_l72_72413

def periodic_seq (a : ℕ → ℝ) (T : ℕ) :=
  ∀ n : ℕ, a (n + T) = a n

def sequence_rule (a : ℕ → ℝ) :=
  ∃ m > 0, a 1 = m ∧
  (∀ n, (a n > 1 → a (n+1) = a n - 1) ∧ (0 < a n ∧ a n ≤ 1 → a (n+1) = 1 / a n))

theorem check_conclusions (a : ℕ → ℝ) :
  sequence_rule a →
  (¬(∃ m > 0, m = 4 / 5 ∧ a 3 = 3)) ∧
  (∀ m > 0, a 3 = 2 → (m = 4 ∨ m = 3 ∨ m = 3 / 2)) ∧
  (∃ m > 0, m = real.sqrt 2 ∧ periodic_seq a 3) ∧
  (¬(∃ m ∈ ℚ, m ≥ 2 ∧ periodic_seq a k)) :=
  sorry

end check_conclusions_l72_72413


namespace one_add_i_cubed_eq_one_sub_i_l72_72589

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
sorry

end one_add_i_cubed_eq_one_sub_i_l72_72589


namespace girls_in_art_class_l72_72178

theorem girls_in_art_class (g b : ℕ) (h_ratio : 4 * b = 3 * g) (h_total : g + b = 70) : g = 40 :=
by {
  sorry
}

end girls_in_art_class_l72_72178


namespace solver_inequality_l72_72534

theorem solver_inequality (x : ℝ) :
  (2 * x - 1 ≥ x + 2) ∧ (x + 5 < 4 * x - 1) → (x ≥ 3) :=
by
  intro h
  sorry

end solver_inequality_l72_72534


namespace quadratic_function_from_table_l72_72078

theorem quadratic_function_from_table :
  ∃ (a b c : ℝ), 
    ( ∀ x ∈ {-2, -1, 0, 1, 2}, 
        (-6.5, -4, -2.5, -2, -2.5).nth x ≠ none 
        → (a * x^2 + b * x + c = (-6.5, -4, -2.5, -2, -2.5).get x)) :=
sorry

end quadratic_function_from_table_l72_72078


namespace fraction_meaningful_l72_72966

theorem fraction_meaningful (a : ℝ) : (a + 3 ≠ 0) ↔ (a ≠ -3) :=
by
  sorry

end fraction_meaningful_l72_72966


namespace perfect_square_1_add_4a_l72_72407

theorem perfect_square_1_add_4a
    (a n : ℕ)
    (h : (sqrt (n + sqrt n) % 1 = sqrt a % 1)) :
    ∃ t : ℕ, 1 + 4 * a = (2 * t + 1) ^ 2 := sorry

end perfect_square_1_add_4a_l72_72407


namespace misha_discharges_before_vasya_l72_72972

theorem misha_discharges_before_vasya
  (initial_charge : ℕ) (initial_vasya : ℕ := 15) (initial_misha : ℕ := 15)
  (final_vasya : ℕ := 11) (final_misha : ℕ := 12)
  (uniform_discharge : ∀ t t' : ℕ, (initial_vasya - final_vasya) = (initial_misha - final_misha)) :
  (final_misha - final_vasya) < 0 :=
by
  let discharge_vasya := initial_vasya - final_vasya
  let discharge_misha := initial_misha - final_misha
  have hv : discharge_vasya = 15 - 11 := rfl
  have hm : discharge_misha = 15 - 12 := rfl
  have h_discharge_correct : discharge_vasya = 4 := rfl
  have h_discharge_correct' : discharge_misha = 3 := rfl
  linarith

end misha_discharges_before_vasya_l72_72972


namespace owen_profit_l72_72876

theorem owen_profit
  (num_boxes : ℕ)
  (cost_per_box : ℕ)
  (pieces_per_box : ℕ)
  (sold_boxes : ℕ)
  (price_per_25_pieces : ℕ)
  (remaining_pieces : ℕ)
  (price_per_10_pieces : ℕ) :
  num_boxes = 12 →
  cost_per_box = 9 →
  pieces_per_box = 50 →
  sold_boxes = 6 →
  price_per_25_pieces = 5 →
  remaining_pieces = 300 →
  price_per_10_pieces = 3 →
  sold_boxes * 2 * price_per_25_pieces + (remaining_pieces / 10) * price_per_10_pieces - num_boxes * cost_per_box = 42 :=
by
  intros h_num h_cost h_pieces h_sold h_price_25 h_remain h_price_10
  sorry

end owen_profit_l72_72876


namespace maria_final_bottle_count_l72_72124

-- Define the initial conditions
def initial_bottles : ℕ := 14
def bottles_drunk : ℕ := 8
def bottles_bought : ℕ := 45

-- State the theorem to prove
theorem maria_final_bottle_count : initial_bottles - bottles_drunk + bottles_bought = 51 :=
by
  sorry

end maria_final_bottle_count_l72_72124


namespace distance_from_point_to_line_l72_72327

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def point_a : Point3D := ⟨1, -1, 2⟩
def point_b : Point3D := ⟨-2, 2, 1⟩
def point_c : Point3D := ⟨-1, -1, 3⟩

def direction_vector (p1 p2 : Point3D) := ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

noncomputable def distance (p : Point3D) (l1 l2 : Point3D) : ℝ := 
  let d := direction_vector l1 l2
  let t := (d.x * (p.x - l1.x) + d.y * (p.y - l1.y) + d.z * (p.z - l1.z)) / (d.x * d.x + d.y * d.y + d.z * d.z)
  let p_proj := ⟨l1.x + t * d.x, l1.y + t * d.y, l1.z + t * d.z⟩
  real.sqrt ((p_proj.x - p.x) ^ 2 + (p_proj.y - p.y) ^ 2 + (p_proj.z - p.z) ^ 2)

theorem distance_from_point_to_line :
  distance point_a point_b point_c = real.sqrt 5 := 
sorry

end distance_from_point_to_line_l72_72327


namespace tip_percentage_approximately_15_l72_72568

noncomputable def totalBill : ℝ := 211.0
noncomputable def numPeople : ℕ := 8
noncomputable def individualShare : ℝ := 30.33125

theorem tip_percentage_approximately_15 :
    let totalBillWithTip := numPeople * individualShare in
    let tip := totalBillWithTip - totalBill in
    let tipPercentage := (tip / totalBill) * 100 in
    abs (tipPercentage - 15) < 0.1 :=
by
  let totalBillWithTip := numPeople * individualShare
  let tip := totalBillWithTip - totalBill
  let tipPercentage := (tip / totalBill) * 100
  show abs (tipPercentage - 15) < 0.1
  sorry

end tip_percentage_approximately_15_l72_72568


namespace exists_free_subset_l72_72974

-- Define what it means for a set of points to be free
def isFree (points : set (ℝ × ℝ)) : Prop :=
  ∀ (a b c : ℝ × ℝ), a ∈ points → b ∈ points → c ∈ points → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a → ¬(dist a b = dist b c ∧ dist b c = dist c a))

-- Define the theorem statement
theorem exists_free_subset (n : ℕ) (h : 0 < n) (points : set (ℝ × ℝ)) (H : points.card = n) :
  ∃ (subset : set (ℝ × ℝ)), subset ⊆ points ∧ subset.card ≥ (⌊real.sqrt n⌋₊) ∧ isFree subset :=
sorry

end exists_free_subset_l72_72974


namespace mean_and_variance_subtracted_by_constant_l72_72414

noncomputable def mean (s : Set ℝ) [Nonempty s] : ℝ :=
  (∑ x in s.toFinset, x) / s.toFinset.card

noncomputable def variance (s : Set ℝ) [h : Nonempty s] : ℝ :=
  let m := mean s
  (∑ x in s.toFinset, (x - m) ^ 2) / s.toFinset.card

theorem mean_and_variance_subtracted_by_constant
  (S : Set ℝ) [Nonempty S] (c : ℝ) (h : c ≠ 0) :
  mean (S.image (λ x, x - c)) ≠ mean S ∧ variance (S.image (λ x, x - c)) = variance S :=
by
  sorry

end mean_and_variance_subtracted_by_constant_l72_72414


namespace car_speed_l72_72257

noncomputable def speed := 900 / 1 -- 900 km/h
noncomputable def time_900 := 1 / speed -- Time for 1 km at 900 km/h in hours
noncomputable def time_900_seconds := time_900 * 3600 -- Convert time to seconds

-- Given conditions
axiom takes_5_seconds_longer : ∀ v : ℝ, (time_900_seconds + 5 / 3600) = (1 / v)

theorem car_speed : ∃ v : ℝ, takes_5_seconds_longer v ∧ v = 400 := 
by
  use 400
  simp [takes_5_seconds_longer, time_900_seconds]
  -- We skip the detailed proof steps here.
  sorry

end car_speed_l72_72257


namespace arithmetic_sequence_a1_range_l72_72850

theorem arithmetic_sequence_a1_range {a_n : ℕ → ℝ} (d : ℝ) 
  (h₁ : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h₂ : d ∈ Ioo (-1) 0)
  (h₃ : (∀ n, S n = n * a_n 1 + d * (n * (n - 1) / 2)) ∧ (∀ n, n = 9 → S n = (S 9)))
  (h₄ : (sin(a_n 4 + a_n 5) ≠ 0))
  (H : (sin^2 (a_n 3) * cos^2 (a_n 6) - sin^2 (a_n 6) * cos^2 (a_n 3) / sin (a_n 4 + a_n 5)) = 1) :
  (a_n 1) ∈ Ioo (4 * π / 3) (3 * π / 2) :=
sorry

end arithmetic_sequence_a1_range_l72_72850


namespace tan_alpha_l72_72013

-- Define the given point on the unit circle
def point_on_unit_circle : ℝ × ℝ := (-4/5, 3/5)

-- Define the tangent of the angle α problem
theorem tan_alpha (x y : ℝ) (h : (x, y) = point_on_unit_circle) : 
  real.tan (real.atan2 x y) = -3/4 :=
by
  sorry

end tan_alpha_l72_72013


namespace squirrel_cannot_catch_nut_l72_72723

section SquirrelNutProblem

variable (a : ℝ := 3.75)  -- distance to squirrel in meters
variable (V₀ : ℝ := 2.5)  -- speed of the nut in m/s
variable (g : ℝ := 10)    -- acceleration due to gravity in m/s^2
variable (max_jump : ℝ := 2.7) -- maximum jump distance of the squirrel in meters

-- The function representing the squared distance between the squirrel and the nut
def distance_sq (t : ℝ) : ℝ :=
  (V₀ * t - a)^2 + (g * t^2 / 2)^2

theorem squirrel_cannot_catch_nut : 
  (∀ t, distance_sq V₀ a g t >= max_jump^2) :=
by
  sorry

end SquirrelNutProblem

end squirrel_cannot_catch_nut_l72_72723


namespace find_smallest_number_l72_72981

theorem find_smallest_number 
  : ∃ x : ℕ, (x - 18) % 14 = 0 ∧ (x - 18) % 26 = 0 ∧ (x - 18) % 28 = 0 ∧ (x - 18) / Nat.lcm 14 (Nat.lcm 26 28) = 746 ∧ x = 271562 := by
  sorry

end find_smallest_number_l72_72981


namespace josh_bottle_caps_l72_72100

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end josh_bottle_caps_l72_72100


namespace domain_of_f_l72_72551

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - x)

theorem domain_of_f :
  {x : ℝ | 2 - x > 0} = {x : ℝ | x < 2} :=
by {
  ext,
  rw [set.mem_set_of_eq, set.mem_set_of_eq],
  exact ⟨fun h => lt_of_sub_pos h, fun h => sub_pos_of_lt h⟩
}

end domain_of_f_l72_72551


namespace symmedian_line_of_triangle_l72_72849

-- Define the problem's conditions and statement in Lean 4
theorem symmedian_line_of_triangle 
  (ABC : Triangle) 
  (B D E F : Point) 
  (BD_internal_bisector : ∠ABD = ∠DBC) 
  (D_on_BC : D ∈ lineSegment B C)
  (BD_intersects_circumcircle_Omega : BD ∩ circumcircle ABC = {B, E})
  (circle_omega_DE : Circle (midpoint D E) (segment D E))
  (F_on_Omega : F ∈ circumcircle ABC ∧ F ∈ circle_omega_DE)
  : isSymmedianLine B F ABC := 
sorry

end symmedian_line_of_triangle_l72_72849


namespace sin_cos_alpha_value_l72_72769

theorem sin_cos_alpha_value (α : ℝ) (h : (4 : ℝ) / sin α = (3 : ℝ) / cos α) :
  sin α * cos α = 12 / 25 :=
by
  sorry

end sin_cos_alpha_value_l72_72769


namespace unique_solution_l72_72715

def satisfies_equation (m n : ℕ) : Prop :=
  15 * m * n = 75 - 5 * m - 3 * n

theorem unique_solution : satisfies_equation 1 6 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → satisfies_equation m n → (m, n) = (1, 6) :=
by {
  sorry
}

end unique_solution_l72_72715


namespace PQRS_parallelogram_and_area_l72_72457

variables {A B C D E P Q R S : Point}
variables (convex_pentagon_ABCDE : ConvexPentagon A B C D E)
variables (centroid_A_BE : CentroidTriangle A B E P)
variables (centroid_B_CE : CentroidTriangle B C E Q)
variables (centroid_C_DE : CentroidTriangle C D E R)
variables (centroid_D_AE : CentroidTriangle D A E S)

theorem PQRS_parallelogram_and_area :
  (is_parallelogram PQRS) ∧ (area PQRS = 2 / 9 * area (Quadrilateral.mk A B C D)) :=
sorry

end PQRS_parallelogram_and_area_l72_72457


namespace Petya_wins_optimally_l72_72138

theorem Petya_wins_optimally :
  (∀ digits : Finset ℕ,
    (∀ x ∈ digits, x ≠ 0 ∧ x < 10) ∧ 
    (∀ x y ∈ digits, x ≠ y → x ∣ y → false) →
    (∃ moves : ℕ → ℕ, 
      moves 0 ∈ digits ∧ 
      moves 0 ≠ 0 ∧ 
      moves 0 < 10 ∧ 
      (∀ i, moves (i + 1) ∈ digits ∧ 
            moves (i + 1) ≠ 0 ∧ 
            moves (i + 1) < 10 ∧ 
            (∀ j < i, moves (i + 1) ∣ moves j → false)) → 
      Petya_wins moves))
:= sorry

end Petya_wins_optimally_l72_72138


namespace triangle_area_is_64_l72_72593

/-- Define the vertices of the triangle --/
def vertex_A : ℝ × ℝ := (8, 8)
def vertex_B : ℝ × ℝ := (-8, 8)
def origin : ℝ × ℝ := (0, 0)

/-- Define the computation for the area of the triangle --/
noncomputable def triangle_area (A B O : ℝ × ℝ) : ℝ :=
  by
    let base := dist A B
    let height := (A.snd - O.snd).abs
    exact (1 / 2) * base * height

/-- The area of the triangle bounded by the lines y = x, y = -x, and y = 8 is 64 --/
theorem triangle_area_is_64 : triangle_area vertex_A vertex_B origin = 64 := by
  sorry

end triangle_area_is_64_l72_72593


namespace find_a_1994_l72_72180

variable (a : ℝ) -- a is a real number

def sequence : ℕ → ℝ
| 0       := a
| (n + 1) := (sequence n * Real.sqrt 3 + 1) / (Real.sqrt 3 - sequence n)

theorem find_a_1994 : sequence a 1994 = (a + Real.sqrt 3) / (1 - a * Real.sqrt 3) :=
by
  sorry

end find_a_1994_l72_72180


namespace proof_problem_l72_72736

-- Definitions of the propositions
def p : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → y = 5 - 3 * x
def q : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → 2 * x + 6 * y - 4 = 0

-- Translate the mathematical proof problem into a Lean theorem
theorem proof_problem : 
  (p ∧ ¬q) ∧ ¬((¬p) ∧ q) :=
by
  -- You can fill in the exact proof steps here
  sorry

end proof_problem_l72_72736


namespace largest_binomial_coefficient_seventh_term_l72_72188

theorem largest_binomial_coefficient_seventh_term :
  ∀ (x : ℝ), let expansion := (1 - 2 * x) ^ 8 in
  let binom_coeff := λ (n k : ℕ), Nat.choose n k in
  ∃ term : ℕ, term = 7 ∧ 
  (∀ k : ℕ, k ≠ 7 → binom_coeff 8 k < binom_coeff 8 6) := 
sorry

end largest_binomial_coefficient_seventh_term_l72_72188


namespace admission_rate_major_B_higher_than_major_A_admission_rate_males_higher_than_females_l72_72627

-- Conditions
def num_applicants_major_A_male : ℕ := 100
def num_applicants_major_A_female : ℕ := 300
def num_applicants_major_B_male : ℕ := 400
def num_applicants_major_B_female : ℕ := 100

def rate_admission_major_A_male : ℝ := 0.25
def rate_admission_major_A_female : ℝ := 0.30
def rate_admission_major_B_male : ℝ := 0.45
def rate_admission_major_B_female : ℝ := 0.50

-- Correct Answers
def overall_admission_rate_major_A : ℝ :=
  ((num_applicants_major_A_male * rate_admission_major_A_male) +
  (num_applicants_major_A_female * rate_admission_major_A_female)) /
  (num_applicants_major_A_male + num_applicants_major_A_female)

def overall_admission_rate_major_B : ℝ :=
  ((num_applicants_major_B_male * rate_admission_major_B_male) +
  (num_applicants_major_B_female * rate_admission_major_B_female)) /
  (num_applicants_major_B_male + num_applicants_major_B_female)

def overall_admission_rate_male : ℝ :=
  ((num_applicants_major_A_male * rate_admission_major_A_male) +
  (num_applicants_major_B_male * rate_admission_major_B_male)) /
  (num_applicants_major_A_male + num_applicants_major_B_male)

def overall_admission_rate_female : ℝ :=
  ((num_applicants_major_A_female * rate_admission_major_A_female) +
  (num_applicants_major_B_female * rate_admission_major_B_female)) /
  (num_applicants_major_A_female + num_applicants_major_B_female)

theorem admission_rate_major_B_higher_than_major_A : overall_admission_rate_major_B > overall_admission_rate_major_A :=
by sorry

theorem admission_rate_males_higher_than_females : overall_admission_rate_male > overall_admission_rate_female :=
by sorry

end admission_rate_major_B_higher_than_major_A_admission_rate_males_higher_than_females_l72_72627


namespace probability_line_not_third_quadrant_l72_72738

noncomputable def A : Set ℤ := {-1, 1, 2}
noncomputable def B : Set ℤ := {-2, 1, 2}

def favorable_pairs : Set (ℤ × ℤ) := {p ∈ (A ×ˢ B) | p.1 < 0 ∧ p.2 > 0}
def total_pairs : Set (ℤ × ℤ) := A ×ˢ B

theorem probability_line_not_third_quadrant :
  (favorable_pairs.to_finset.card : ℚ) / (total_pairs.to_finset.card : ℚ) = 2 / 9 :=
by
  sorry

end probability_line_not_third_quadrant_l72_72738


namespace sum_squares_mod_13_l72_72618

theorem sum_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 5 :=
by sorry

end sum_squares_mod_13_l72_72618


namespace simplify_trig_expression_l72_72153

def trigonometric_identity (x : ℝ) : Prop :=
  let sin_half_x := sin (x / 2)
  let cos_half_x := cos (x / 2)
  let lhs := 2 - (2 * sin_half_x * cos_half_x) - (1 - 2 * sin_half_x^2)
  let rhs := 2 + (2 * sin_half_x * cos_half_x) - (1 - 2 * sin_half_x^2)
  (lhs / rhs) = (tan (x / 2) * tan ((x / 2) - (π / 4)))

theorem simplify_trig_expression (x : ℝ) : trigonometric_identity x :=
  sorry

end simplify_trig_expression_l72_72153


namespace meals_for_dinner_l72_72426

theorem meals_for_dinner (lunch_meals prepared_dinner_meals sold_lunch_meals remaining_lunch_meals total_meals : ℕ) 
  (h1 : lunch_meals = 17) 
  (h2 : sold_lunch_meals = 12) 
  (h3 : prepared_dinner_meals = 5) 
  (h4 : remaining_lunch_meals = lunch_meals - sold_lunch_meals) 
  (h5 : total_meals = remaining_lunch_meals + prepared_dinner_meals) : 
  total_meals = 10 := 
begin
  sorry
end

end meals_for_dinner_l72_72426


namespace real_parts_product_l72_72316

noncomputable def product_of_real_parts (c : ℂ) : ℂ :=
  let x1 := -1 + real.sqrt (complex.re (c / 2)) in
  let x2 := -1 - real.sqrt (complex.re (c / 2)) in
  x1 * x2

theorem real_parts_product : product_of_real_parts (1 + complex.I) = -1 / 2 := by
  sorry

end real_parts_product_l72_72316


namespace picture_size_l72_72512

theorem picture_size (total_pics_A : ℕ) (size_A : ℕ) (total_pics_B : ℕ) (C : ℕ)
  (hA : total_pics_A * size_A = C) (hB : total_pics_B = 3000) : 
  (C / total_pics_B = 8) :=
by
  sorry

end picture_size_l72_72512


namespace raised_bed_height_l72_72296

theorem raised_bed_height : 
  ∀ (total_planks : ℕ) (num_beds : ℕ) (planks_per_bed : ℕ) (height : ℚ),
  total_planks = 50 →
  num_beds = 10 →
  planks_per_bed = 4 * height →
  (total_planks = num_beds * planks_per_bed) →
  height = 5 / 4 :=
by
  intros total_planks num_beds planks_per_bed H
  intros h1 h2 h3 h4
  sorry

end raised_bed_height_l72_72296


namespace possible_k_values_l72_72425

theorem possible_k_values :
  ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 10) →
   (∃ (colors : set (set (ℕ × ℕ))),
    (∀ (p : set ℕ), p.card = 10 →
     (∀ (q : set ℕ), q ⊆ p → q.card = k →
      ∀ (i j : ℕ), i ∈ q → j ∈ q → i < j →
      ∃ (c ∈ colors), {i, j} ∈ c)) → k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10)
:= 
  sorry

end possible_k_values_l72_72425


namespace stadium_length_in_yards_l72_72558

theorem stadium_length_in_yards (length_in_feet : ℕ) (conversion_factor : ℕ) : ℕ :=
    length_in_feet / conversion_factor

example : stadium_length_in_yards 240 3 = 80 :=
by sorry

end stadium_length_in_yards_l72_72558


namespace algebraic_expression_value_l72_72786

theorem algebraic_expression_value (x : ℝ) (h : x = 2 * Real.sqrt 3 - 1) : x^2 + 2 * x - 3 = 8 :=
by 
  sorry

end algebraic_expression_value_l72_72786


namespace problem_conditions_l72_72388

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 6)

theorem problem_conditions (ω : ℝ) (ϕ : ℝ) (h1 : ω > 0) (h2 : -π / 2 ≤ ϕ) (h3 : ϕ < π / 2)
                           (h4 : (λ x, sqrt 3 * sin (ω * x + ϕ)) = (λ x, sqrt 3 * sin (2 * x - π / 6))) : Prop :=
  (∀ k : ℤ, (λ x, sqrt 3 * sin (2 * (k * π / 2 + π / 12) - π / 6)) (k * π / 2 + π / 12) = 0) ∧
  (∀ k : ℤ, (λ x, sqrt 3 * sin (2 * x - π / 6)) (2 * (k * π) - π / 6) ≤
           (λ x, sqrt 3 * sin (2 * x - π / 6)) (2 * x - π / 6) ∧
           (λ x, sqrt 3 * sin (2 * x - π / 6)) (2 * x - π / 6) ≤
           (λ x, sqrt 3 * sin (2 * x - π / 6)) (2 * (k * π) + π / 2 - π / 6))

end problem_conditions_l72_72388


namespace mapped_set_eq_l72_72528

theorem mapped_set_eq {A B : set ℝ} (f : ℝ → ℝ) (h₁ : ∀ x ∈ A, f x = x^3 - x + 1)
  (h₂ : ∀ x ∈ B, f x = 1 → x = 0 ∨ x = 1 ∨ x = -1) :
  {x ∈ A | f x = 1} = {0, 1, -1} :=
by
  sorry

end mapped_set_eq_l72_72528


namespace part1_part2_l72_72765

open Set

variable (m : ℝ) (A B : Set ℝ)

def A (m : ℝ) : Set ℝ := { x : ℝ | 1 - m ≤ x ∧ x ≤ 2 * m + 1 }

def B : Set ℝ := { x : ℝ | 1 / 9 ≤ 3^x ∧ 3^x ≤ 81 }

theorem part1 : A 2 ∪ B = { x : ℝ | -2 ≤ x ∧ x ≤ 5 } :=
sorry

theorem part2 : (B ⊆ A m) → m ≥ 3 :=
sorry

end part1_part2_l72_72765


namespace find_a_value_l72_72757

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (-x) = -f (x)

theorem find_a_value :
  ∃ a : ℝ, (∀ x : ℝ, a - 2 / (2^x + 1) = -(a - 2 / (2^(-x) + 1))) → a = 1 :=
begin
  sorry
end

end find_a_value_l72_72757


namespace fraction_multiplied_by_3_l72_72789

theorem fraction_multiplied_by_3 (a b : ℚ) (h : a ≠ b) : 
  (3 * (a * b) / (3 * a - 3 * b) = 3 * (a * b / (a - b))) := 
by 
  let num := 3 * (a * b)
  let denom := 3 * a - 3 * b
  have h1 : denom = 3 * (a - b),
  { ring },
  rw h1,
  suffices : num / 3 * (a - b) = 3 * (a * b / (a - b)),
  { rw this },
  rw mul_div_cancel,
  sorry

end fraction_multiplied_by_3_l72_72789


namespace part_a_l72_72764

structure Point :=
  (x : ℝ)
  (y : ℝ)

def parabola (p : ℝ) := { pt : Point | pt.y^2 = 4 * p * pt.x }

noncomputable def focus (p : ℝ) : Point :=
  ⟨p, 0⟩

def projection_on_directrix (p : ℝ) (M : Point) : Point :=
  ⟨-p, M.y⟩

def line_through_focus (F : Point) (m : ℝ) : Point → Prop :=
  fun P : Point => P.y = m * (P.x - F.x) + F.y

noncomputable def line_intersects_parabola_at : parabola 1 → parabola 1 → Prop
| ⟨x1, y1, _⟩ , ⟨x2, y2, _⟩ := (y1 = 2 * √x1 ∨ y1 = -2 * √x1) ∧ (y2 = 2 * √x2 ∨ y2 = -2 * √x2) 

noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem part_a (x1 x2 : ℝ) (y1 y2 : ℝ) (h : x1 + x2 = 6) (intersects : line_intersects_parabola_at ⟨x1, y1, rfl⟩ ⟨x2, y2, rfl⟩) : distance ⟨x1, y1⟩ ⟨x2, y2⟩ = 8 := 
sorry

end part_a_l72_72764


namespace find_angle_between_generatrix_and_base_l72_72276

-- Definitions to represent the problem conditions
def is_truncated_cone_with_inscribed_sphere
  (V1 V2 : ℝ) (H R R1 R2 : ℝ) (α : ℝ) : Prop :=
  V2 = 2 / 3 * real.pi * R^3 ∧
  V1 = 1 / 3 * real.pi * H * (R1^2 + R1 * R2 + R2^2) ∧
  V2 = 1 / 2 * V1 ∧
  H = 2 * R ∧
  H = (R1 - R2) * real.tan α

-- The main theorem to prove
theorem find_angle_between_generatrix_and_base
  (V1 V2 : ℝ) (H R R1 R2 : ℝ) (α : ℝ)
  (h : is_truncated_cone_with_inscribed_sphere V1 V2 H R R1 R2 α)
  : α = real.arctan 2 := 
sorry

end find_angle_between_generatrix_and_base_l72_72276


namespace common_point_exists_l72_72829

-- Definitions based on the problem conditions
variable {k k' : Circle} (O : Point) (A B E F : Point)
variable [concentric : concentric_circles k k' O]
variable [larger : larger_circle k' k]
variable [line_OAB : line_through O A B k k']
variable [line_OEF : line_through O E F k k']

-- Points conditions
variable [separate_O_A_B : separates_point O A B]
variable [separate_E_O_F : separates_point E O F]

-- Define the circles based on diameters
def circumcircle_OAE := circumcircle (triangle O A E)
def circle_diameter_AB := circle_with_diameter A B
def circle_diameter_EF := circle_with_diameter E F

-- Proposition Statement
theorem common_point_exists :
  ∃ P : Point, P ∈ circumcircle_OAE ∧ P ∈ circle_diameter_AB ∧ P ∈ circle_diameter_EF :=
sorry -- The proof would go here

-- Definitions
class concentric_circles (k k' : Circle) (O : Point) : Prop := sorry
class larger_circle (k' k : Circle) : Prop := sorry
class line_through (O A B : Point) (k k' : Circle) : Prop := sorry
class separates_point (X Y Z : Point) : Prop := sorry
def circumcircle (T : Triangle) : Circle := sorry
def triangle (O A E : Point) : Triangle := sorry
def circle_with_diameter (A B : Point) : Circle := sorry

end common_point_exists_l72_72829


namespace highest_throw_among_them_l72_72682

variable (Christine_throw1 Christine_throw2 Christine_throw3 Janice_throw1 Janice_throw2 Janice_throw3 : ℕ)
 
-- Conditions given in the problem
def conditions :=
  Christine_throw1 = 20 ∧
  Janice_throw1 = Christine_throw1 - 4 ∧
  Christine_throw2 = Christine_throw1 + 10 ∧
  Janice_throw2 = Janice_throw1 * 2 ∧
  Christine_throw3 = Christine_throw2 + 4 ∧
  Janice_throw3 = Christine_throw1 + 17

-- The proof statement
theorem highest_throw_among_them : conditions -> max (max (max (max (max Christine_throw1 Christine_throw2) Christine_throw3) Janice_throw1) Janice_throw2) Janice_throw3 = 37 :=
sorry

end highest_throw_among_them_l72_72682


namespace complex_number_real_implies_value_of_a_l72_72419

theorem complex_number_real_implies_value_of_a (a : ℝ) : 
  (∃ (a_real : ℝ), (1 - complex.i) * (a_real + complex.i) ∈ SetOf (λ z : ℂ, z.im = 0)) → 
  a = 1 :=
by
  sorry

end complex_number_real_implies_value_of_a_l72_72419


namespace tony_water_intake_l72_72198

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end tony_water_intake_l72_72198


namespace frank_money_remaining_l72_72344

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l72_72344


namespace single_elimination_matches_l72_72429

theorem single_elimination_matches (players : Nat) (h_start : players = 512) 
  (h_elimination : ∀ match, match = players - 1) :
  matches = 511 :=
by 
  sorry

end single_elimination_matches_l72_72429


namespace digits_of_n_are_9_l72_72842

noncomputable def number_of_digits (n : ℕ) : ℕ :=
  (n : ℚ).toReal.decDigits.length

theorem digits_of_n_are_9 :
  ∃ n : ℕ, 
    (∀ m : ℕ, (m % 30 = 0 ∧ (∃ k1 : ℕ, m^2 = k1^3) ∧ (∃ k2 : ℕ, m^5 = k2^2)) → n ≤ m) ∧
    (n % 30 = 0) ∧
    (∃ k1 : ℕ, n^2 = k1^3) ∧
    (∃ k2 : ℕ, n^5 = k2^2) ∧
    (number_of_digits n = 9) :=
begin
  sorry
end

end digits_of_n_are_9_l72_72842


namespace restaurant_bill_l72_72227

theorem restaurant_bill (cost_taco : ℝ) (cost_friends_bill : ℝ) (cost_friend_tacos : ℝ) (cost_friend_enchiladas : ℝ)
  (num_your_tacos num_your_enchiladas num_friend_tacos num_friend_enchiladas : ℕ)
  (H1 : num_your_tacos = 2) (H2 : num_your_enchiladas = 3) (H3 : num_friend_tacos = 3) (H4 : num_friend_enchiladas = 5)
  (H5 : cost_taco = 0.9) (H6 : cost_friends_bill = 12.7) 
  (H7 : cost_friend_tacos = num_friend_tacos * cost_taco) 
  (H8 : cost_friend_enchiladas = cost_friends_bill - cost_friend_tacos) :
  let cost_enchilada := cost_friend_enchiladas / num_friend_enchiladas,
      cost_your_tacos := num_your_tacos * cost_taco,
      cost_your_enchiladas := num_your_enchiladas * cost_enchilada,
      total_your_bill := cost_your_tacos + cost_your_enchiladas in
  total_your_bill = 7.8 :=
by
  -- Proof to be filled in
  sorry

end restaurant_bill_l72_72227


namespace max_value_negative_one_l72_72785

theorem max_value_negative_one (f : ℝ → ℝ) (hx : ∀ x, x < 1 → f x ≤ -1) :
  ∀ x, x < 1 → ∃ M, (∀ y, y < 1 → f y ≤ M) ∧ f x = M :=
sorry

end max_value_negative_one_l72_72785


namespace arithmetic_seq_term_six_l72_72337

theorem arithmetic_seq_term_six {a : ℕ → ℝ} (a1 : ℝ) (S3 : ℝ) (h1 : a1 = 2) (h2 : S3 = 12) :
  a 6 = 12 :=
sorry

end arithmetic_seq_term_six_l72_72337


namespace probability_solution_l72_72492

noncomputable def binom_10_7 := Nat.choose 10 7
noncomputable def binom_10_6 := Nat.choose 10 6

theorem probability_solution (p q : ℝ) (h₁ : q = 1 - p) (h₂ : binom_10_7 = 120) (h₃ : binom_10_6 = 210)
  (h₄ : 120 * p ^ 7 * q ^ 3 = 210 * p ^ 6 * q ^ 4) : p = 7 / 11 := 
sorry

end probability_solution_l72_72492


namespace angle_between_tangents_from_y_eq_x_to_circle_is_60_l72_72700

/-- 
Prove that the angle between two tangent lines from any point on the line y = x 
to the circle (x - 5)^2 + (y - 1)^2 = 2, when the lines are symmetric about y = x,
is 60 degrees.
-/
theorem angle_between_tangents_from_y_eq_x_to_circle_is_60 :
  ∀ (P : ℝ × ℝ), (P.1 = P.2) → 
  let C := (5, 1) in 
  let r := real.sqrt 2 in 
  geom.angle_between_two_tangents P C r = 60 :=
by sorry

end angle_between_tangents_from_y_eq_x_to_circle_is_60_l72_72700


namespace round_product_eq_156_l72_72699

theorem round_product_eq_156 :
  ∃ x : ℝ, (ceil x) * x = 156 ∧ x = 12 := by
  sorry

end round_product_eq_156_l72_72699


namespace pen_distribution_l72_72319

theorem pen_distribution:
  (∃ (fountain: ℕ) (ballpoint: ℕ), fountain = 2 ∧ ballpoint = 3) ∧
  (∃ (students: ℕ), students = 4) →
  (∀ (s: ℕ), s ≥ 1 → s ≤ 4) →
  ∃ (ways: ℕ), ways = 28 :=
by
  sorry

end pen_distribution_l72_72319


namespace sam_nickels_count_l72_72525

-- Variables for the conditions
variable (initial_nickels given_nickels taken_nickels : ℕ)

-- Definition and theorem statement
def final_nickels (initial_nickels given_nickels taken_nickels : ℕ) : ℕ :=
  (initial_nickels + given_nickels) - taken_nickels

theorem sam_nickels_count :
  initial_nickels = 29 →
  given_nickels = 24 →
  taken_nickels = 13 →
  final_nickels initial_nickels given_nickels taken_nickels = 40 :=
by
  intros h_initial h_given h_taken
  rw [h_initial, h_given, h_taken]
  calc
    (29 + 24) - 13 = 53 - 13 := by norm_num
                 ... = 40   := by norm_num

end sam_nickels_count_l72_72525


namespace product_abc_l72_72744

theorem product_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b^3 = 180) : a * b * c = 60 * c := 
sorry

end product_abc_l72_72744


namespace third_number_eq_l72_72569

theorem third_number_eq :
  ∃ x : ℝ, (0.625 * 0.0729 * x) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ x = 2.33075 := 
by
  sorry

end third_number_eq_l72_72569


namespace cannot_rearrange_to_satisfy_divisibility_condition_l72_72030

noncomputable def A (x : ℕ) : ℕ :=
∑ i in Finset.range 100, (2 * i + 1) * x ^ i

def B (b : Fin 100 → ℕ) (x : ℕ) : ℕ :=
∑ i in Finset.range 100, b i * x ^ i

theorem cannot_rearrange_to_satisfy_divisibility_condition (b : Fin 100 → ℕ) :
  (∀ k ≥ 2, ¬ (199 ∣ (A k - B b k))) → false :=
 sorry

end cannot_rearrange_to_satisfy_divisibility_condition_l72_72030


namespace nonagon_blue_quadrilateral_l72_72805

theorem nonagon_blue_quadrilateral :
  ∀ (vertices : Finset ℕ) (red blue : ℕ → ℕ → Prop),
    (vertices.card = 9) →
    (∀ a b, red a b ∨ blue a b) →
    (∀ a b c, (red a b ∧ red b c ∧ red c a) → False) →
    (∃ A B C D, blue A B ∧ blue B C ∧ blue C D ∧ blue D A ∧ blue A C ∧ blue B D) := 
by
  -- Proof goes here
  sorry

end nonagon_blue_quadrilateral_l72_72805


namespace area_equivalence_l72_72242

-- Definitions for the vertices and the points of intersection.
variables {A A₁ B B₁ C C₁ D D₁ E E₁ K : Point}

-- Conditions for the problem
axiom regular_star_pentagon : regular_star_shaped_pentagon A A₁ B B₁ C C₁ D D₁ E E₁
axiom lines_intersect_at_K : extends_to_intersect A B D E K

-- Definition of polygon and quadrilateral
def polygon_A_B_B₁_C_C₁_D_E_D₁ : Polygon := ⟨[A, B, B₁, C, C₁, D, E, D₁], sorry⟩
def quadrilateral_A_D₁_E_K : Quadrilateral := ⟨[A, D₁, E, K], sorry⟩

-- Theorem to be proved
theorem area_equivalence :
  area polygon_A_B_B₁_C_C₁_D_E_D₁ = area quadrilateral_A_D₁_E_K :=
sorry

end area_equivalence_l72_72242


namespace perimeter_of_isosceles_right_triangle_l72_72813

theorem perimeter_of_isosceles_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_equal_angles : a = b) (h_hypotenuse : c = 10 * real.sqrt 2) :
  a + b + c = 20 + 10 * real.sqrt 2 :=
by
  sorry

end perimeter_of_isosceles_right_triangle_l72_72813


namespace math_problem_l72_72051

theorem math_problem 
  (x y : ℝ) 
  (h : x^2 + y^2 - x * y = 1) 
  : (-2 ≤ x + y) ∧ (x^2 + y^2 ≤ 2) :=
by
  sorry

end math_problem_l72_72051


namespace bojan_wins_strategy_l72_72671

theorem bojan_wins_strategy (a b : ℕ) (h1: ∀ i, 1 ≤ i ≤ 2016 → ∃ k, a + i = k) (h2: ∀ j, 1 ≤ j ≤ 2016 → ∃ m, b + j = m) :
  let pieces_ana := {a + i | i in (1:ℕ)..2016}
  let pieces_bojan := {b + j | j in (1:ℕ)..2016}
  ∀ x ∈ pieces_ana, ∃ y ∈ pieces_bojan, (x + y) % 2017 = (a + b) % 2017 :=
by
  sorry

end bojan_wins_strategy_l72_72671


namespace value_of_n_l72_72721

theorem value_of_n : ∃ n : ℕ, 4^3 - 8 = 5^2 + n ∧ n = 31 :=
by
  use 31
  split
  { norm_num }
  { rfl }

# The proof is omitted since we only need the statement

end value_of_n_l72_72721


namespace stephen_speed_l72_72536

theorem stephen_speed (v : ℝ) 
  (time : ℝ := 0.25)
  (speed_second_third : ℝ := 12)
  (speed_last_third : ℝ := 20)
  (total_distance : ℝ := 12) :
  (v * time + speed_second_third * time + speed_last_third * time = total_distance) → 
  v = 16 :=
by
  intro h
  -- introducing the condition h: v * 0.25 + 3 + 5 = 12
  sorry

end stephen_speed_l72_72536


namespace james_dancing_calories_l72_72088

theorem james_dancing_calories
  (calories_per_hour_walking : ℕ)
  (calories_per_hour_dancing : ℕ)
  (hours_per_session : ℝ)
  (sessions_per_day : ℕ)
  (days_per_week : ℕ)
  (cal_per_hour_walking : calories_per_hour_walking = 300)
  (cal_per_hour_dancing : calories_per_hour_dancing = 2 * calories_per_hour_walking)
  (hours_per_session_def : hours_per_session = 0.5)
  (sessions_per_day_def : sessions_per_day = 2)
  (days_per_week_def : days_per_week = 4)
  : calories_per_hour_dancing * (hours_per_session * sessions_per_day * days_per_week).natAbs = 2400 := by
  sorry

end james_dancing_calories_l72_72088


namespace sum_of_squares_mod_13_l72_72613

theorem sum_of_squares_mod_13 :
  (∑ k in Finset.range 16, k^2) % 13 = 10 :=
by
  sorry

end sum_of_squares_mod_13_l72_72613


namespace sum_of_squares_mod_13_l72_72614

theorem sum_of_squares_mod_13 :
  (∑ k in Finset.range 16, k^2) % 13 = 10 :=
by
  sorry

end sum_of_squares_mod_13_l72_72614


namespace max_product_sum_2015_l72_72887

-- Define the main problem and conditions
def sum_to_2015 (S : Set ℕ) := ∃ l : List ℕ, l.product = S ∧ l.sum = 2015 ∧ ∀ s ∈ l, s > 0

def largest_product (S : Set ℕ) := 
  ∀ T : Set ℕ, sum_to_2015 T → (∏ x in S, x) ≥ (∏ x in T, x)

-- Define the answer stated in the solution
def answer_set : Set ℕ := {a | (a = 3 ∧ a.count = 671) ∨ (a = 2 ∧ a.count = 1)}

-- Prove the equivalence
theorem max_product_sum_2015 : largest_product answer_set := 
by
  sorry

end max_product_sum_2015_l72_72887


namespace triangle_probability_YD_gt_6sqrt2_and_P_closer_to_Q_l72_72583

theorem triangle_probability_YD_gt_6sqrt2_and_P_closer_to_Q 
  (XYZ : Type*) [metric_space XYZ]
  (X Y Z Q P D : XYZ)
  (hXYZ_right : angle XY Z = π/2)
  (hYXZ_60 : angle YX Z = π/3)
  (hXY_12 : dist X Y = 12)
  (hXQ_8 : dist X Q = 8)
  (point_P_in_triangle : ∃ P : XYZ, P ∈ triangle XY Z)
  (D_on_XZ : ∃ D : XYZ, collinear X Z D)
  : probability (dist Y D > 6 * sqrt 2) = (3 - sqrt 3) / 3 ∧ 
    probability (dist P Q < dist P Y) = by {
  sorry
}

end triangle_probability_YD_gt_6sqrt2_and_P_closer_to_Q_l72_72583


namespace birch_count_is_87_l72_72639

def num_trees : ℕ := 130
def incorrect_signs (B L : ℕ) : Prop := B + L = num_trees ∧ L + 1 = num_trees - 1 ∧ B = 87

theorem birch_count_is_87 (B L : ℕ) (h1 : B + L = num_trees) (h2 : L + 1 = num_trees - 1) :
  B = 87 :=
sorry

end birch_count_is_87_l72_72639


namespace find_vector_b_l72_72956

open Matrix

noncomputable def vector_b (a b : Vector 3 ℝ) : Prop :=
  a + b = ![8, -4, 0] ∧ 
  (∃ t : ℝ, a = t • ![2, -1, 1]) ∧ 
  (b ⬝ ![2, -1, 1] = 0)

theorem find_vector_b : ∃ b : Vector 3 ℝ, vector_b ![2, -1, -3] b :=
sorry

end find_vector_b_l72_72956


namespace comparison_problem_l72_72728

-- Define the three variables according to their given expressions
def a : ℝ := log 2 / log 5
def b : ℝ := Real.sin (55 * Real.pi / 180)
def c : ℝ := (1 / 2)^0.6

-- State the theorem
theorem comparison_problem : b > c ∧ c > a := by
  sorry

end comparison_problem_l72_72728


namespace highest_throw_christine_janice_l72_72679

theorem highest_throw_christine_janice
  (c1 : ℕ) -- Christine's first throw
  (j1 : ℕ) -- Janice's first throw
  (c2 : ℕ) -- Christine's second throw
  (j2 : ℕ) -- Janice's second throw
  (c3 : ℕ) -- Christine's third throw
  (j3 : ℕ) -- Janice's third throw
  (h1 : c1 = 20)
  (h2 : j1 = c1 - 4)
  (h3 : c2 = c1 + 10)
  (h4 : j2 = j1 * 2)
  (h5 : c3 = c2 + 4)
  (h6 : j3 = c1 + 17) :
  max c1 (max c2 (max c3 (max j1 (max j2 j3)))) = 37 := by
  sorry

end highest_throw_christine_janice_l72_72679


namespace jodi_walked_miles_per_day_l72_72450

theorem jodi_walked_miles_per_day (x : ℕ) 
  (h1 : 6 * x + 12 + 18 + 24 = 60) : 
  x = 1 :=
by
  sorry

end jodi_walked_miles_per_day_l72_72450


namespace george_flips_probability_l72_72725

open ProbabilityTheory

/-- Define the probability space for George's 5 coin flips -/
def george_coin (n : ℕ) := 
  {flips_vec : vector (option bool) n // 
    ∀ i, i < n → (flips_vec.nth i = some tt ∧ ∃p, p = 1/4) ∨ 
         (flips_vec.nth i = some ff ∧ ∃q, q = 3/4)}

/-- Define the probability of getting exactly k tails in n flips -/
noncomputable def prob_k_tails (k n : ℕ) (p : ℚ) : ℚ := 
  ∑ wk in (finset.filter (λ s : finset (fin (n : ℕ)), s.card = k) finset.univ),
  p ^ k * (1 - p) ^ (n - k)

/-- Calculation of final probability -/
theorem george_flips_probability : 
  prob_k_tails 3 5 (3/4) * (3/4) * (3/4) = 567/1024 :=
by 
  sorry

end george_flips_probability_l72_72725


namespace naval_formation_l72_72654

theorem naval_formation :
  let num_ways_submarines := 2!
  let num_ways_arrange_ships := (Nat.choose 6 3) * (3! * 3!)
  let num_invalid_arrangements := 2 * 2!
  num_ways_submarines * (num_ways_arrange_ships - num_invalid_arrangements) = 1296 :=
by
  sorry

end naval_formation_l72_72654


namespace plane_α_perpendicular_to_plane_β_l72_72735

variables {Point : Type*} [AffineSpace Point ℝ]
variables {α β : Affine.Subspace ℝ Point}
variables {l m : Affine.Subspace ℝ Point}

def is_parallel (s t : Affine.Subspace ℝ Point) : Prop := 
  ∃ (v : ℝ), s.direction = v • t.direction

def is_perpendicular (s t : Affine.Subspace ℝ Point) : Prop := 
  ∀ (v ∈ s.direction) (w ∈ t.direction), @InnerProductSpace.inner ℝ _ _ ⟨AffineSpace.vector_space ℝ Point⟩ v w = 0

-- Conditions
axiom line_l_in_plane_α : l ≤ α
axiom line_m_in_plane_β : m ≤ β
axiom l_perpendicular_to_β : is_perpendicular l β

-- Goal
theorem plane_α_perpendicular_to_plane_β :
  is_perpendicular α β :=
sorry

end plane_α_perpendicular_to_plane_β_l72_72735


namespace DM_eq_DN_l72_72115

theorem DM_eq_DN 
  (A B C D P M E F N : Point)
  (hD_on_BC : lies_on D (line_segment B C))
  (hP_on_AD : lies_on P (line_segment A D))
  (h_line_through_D : straight_line_through D)
  (hM_on_AB : lies_on M (line_segment A B))
  (hE_on_PB : lies_on E (line_segment P B))
  (hF_on_AC : lies_on F (extension_of_line_segment A C))
  (hN_on_PC : lies_on N (extension_of_line_segment P C))
  (hDE_eq_DF : dist D E = dist D F) :
  dist D M = dist D N := 
by
  sorry

end DM_eq_DN_l72_72115


namespace equivalent_proof_problem_l72_72701

-- Definition of curve C in polar coordinates and its conversion to Cartesian coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ^2 * Float.cos (2 * θ) + 4 = 0
def curve_C_Cartesian (x y : ℝ) : Prop := y^2 - x^2 = 4

-- Parametric equation of line l
def line_l (t : ℝ) : ℝ × ℝ := (t, Float.sqrt(5) + 2 * t)

-- Point A
def point_A : ℝ × ℝ := (0, Float.sqrt 5)

-- M, N are intersection points between line l and curve C, we need to find the value of 1/|AM| + 1/|AN|
def find_value_AM_AN (A : ℝ × ℝ) (M N: ℝ × ℝ) : ℝ :=
  let AM := Math.sqrt ((fst A - fst M)^2 + (snd A - snd M)^2)
  let AN := Math.sqrt ((fst A - fst N)^2 + (snd A - snd N)^2)
  1 / (Math.abs AM) + 1 / (Math.abs AN)

theorem equivalent_proof_problem :
    (∀ (ρ θ : ℝ), curve_C ρ θ ↔ ∃ (x y : ℝ), ρ = Math.sqrt (x^2 + y^2) ∧ θ = Float.atan2 y x ∧ curve_C_Cartesian x y) ∧
    (∀ (A : ℝ × ℝ), ∃ (M N: ℝ × ℝ), M = line_l t ∧ N = line_l t₁ → find_value_AM_AN A M N = 4) :=
    by
    sorry

end equivalent_proof_problem_l72_72701


namespace negation_of_existence_l72_72173

variable (Triangle : Type) (has_circumcircle : Triangle → Prop)

theorem negation_of_existence :
  ¬ (∃ t : Triangle, ¬ has_circumcircle t) ↔ ∀ t : Triangle, has_circumcircle t :=
by sorry

end negation_of_existence_l72_72173


namespace dual_of_regular_polyhedron_is_regular_l72_72146

theorem dual_of_regular_polyhedron_is_regular
  (T : Type) [regular_polyhedron T] (T′ : Type) [dual_polyhedron T T′] [regular_polyhedron T′] :
  regular_polyhedron T' :=
sorry

end dual_of_regular_polyhedron_is_regular_l72_72146


namespace number_of_valid_arrangements_is_68_l72_72141

-- Definitions of the constraints.
def Books : Type := {classics : List String // classics.length = 5}
def valid_book_order (order : List String) : Prop :=
  order.length = 4 ∧
  (order.nodup) ∧
  ("Book of Songs" ∈ order) ∧
  ("Book of Documents" ∈ order) ∧
  ("Book of Rites" ∈ order) ∧
  ("Book of Changes" ∈ order) ∧
  ("Spring and Autumn Annals" ∈ order) ∧
  ∀ (i : ℕ) (h : i < order.length - 1),
    ¬((order.nth_le i (by linarith) = "Book of Songs" ∧ order.nth_le (i + 1) (by linarith) = "Book of Rites") ∨
      (order.nth_le i (by linarith) = "Book of Rites" ∧ order.nth_le (i + 1) (by linarith) = "Book of Songs")) ∧
  ¬(order.head = some "Book of Changes")

-- The main statement to be proved.
theorem number_of_valid_arrangements_is_68 :
  ∃ (orders : List (List String)), (∀ o ∈ orders, valid_book_order o) ∧ orders.length = 68 :=
sorry

end number_of_valid_arrangements_is_68_l72_72141


namespace sum_of_squares_mod_13_l72_72604

theorem sum_of_squares_mod_13 :
  ((∑ i in Finset.range 16, i^2) % 13) = 3 :=
by 
  sorry

end sum_of_squares_mod_13_l72_72604


namespace total_cost_of_fireworks_is_159_remaining_fireworks_are_13_l72_72040

-- Define the conditions
def small_firework_cost : ℕ := 12
def large_firework_cost : ℕ := 25
def henrys_small_fireworks : ℕ := 3
def henrys_large_fireworks : ℕ := 2
def friends_small_fireworks : ℕ := 4
def friends_large_fireworks : ℕ := 1
def saved_fireworks : ℕ := 6
def used_saved_fireworks : ℕ := 3

-- Define the statements to prove
theorem total_cost_of_fireworks_is_159 :
  (henrys_small_fireworks * small_firework_cost +
   henrys_large_fireworks * large_firework_cost +
   friends_small_fireworks * small_firework_cost +
   friends_large_fireworks * large_firework_cost) = 159 :=
by simp [small_firework_cost, large_firework_cost, henrys_small_fireworks,
         henrys_large_fireworks, friends_small_fireworks, friends_large_fireworks]; norm_num; sorry

theorem remaining_fireworks_are_13 :
  (henrys_small_fireworks + henrys_large_fireworks +
   friends_small_fireworks + friends_large_fireworks +
   saved_fireworks - used_saved_fireworks) = 13 :=
by simp [henrys_small_fireworks, henrys_large_fireworks, friends_small_fireworks,
         friends_large_fireworks, saved_fireworks, used_saved_fireworks]; norm_num; sorry 

end total_cost_of_fireworks_is_159_remaining_fireworks_are_13_l72_72040


namespace ray_equation_and_distance_l72_72271

def P : ℝ × ℝ := (-6, 7)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8 * x - 6 * y + 21 = 0
def line_of_ray (x y : ℝ) : Prop := (3 * x + 4 * y - 10 = 0) ∨ (4 * x + 3 * y + 3 = 0)
def distance_to_center (x y : ℝ) := real.sqrt ((x + 6)^2 + (y + 7)^2)

theorem ray_equation_and_distance :
  (∃ x y, circle_eq x y ∧ distance_to_center x y = 14) →
  (line_of_ray (-6) (-7)) :=
sorry

end ray_equation_and_distance_l72_72271


namespace b_greater_than_neg3_l72_72011

def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem b_greater_than_neg3 (b : ℝ) :
  (∀ (n : ℕ), 0 < n → a_n (n + 1) b > a_n n b) → b > -3 :=
by
  sorry

end b_greater_than_neg3_l72_72011


namespace quadrilateral_is_kite_l72_72804

-- Definitions based on the given conditions
structure Quadrilateral :=
  (a b c d : ℝ)  -- sides of the quadrilateral
  (diag1 diag2 : ℝ) -- diagonals of the quadrilateral
  (diagonals_perpendicular : diag1 ⊥ diag2)
  (adjacent_sides_equal : a = b ∧ c = d ∧ a ≠ c)

-- The theorem stating the quadrilateral is a kite
theorem quadrilateral_is_kite (q : Quadrilateral) : 
  ∃ k : Quadrilateral, k = q ∧ (∀ a b c d : ℝ, q.a = q.b ∧ q.c = q.d ∧ q.a ≠ q.c) :=
  sorry

end quadrilateral_is_kite_l72_72804


namespace division_of_cubes_l72_72210

theorem division_of_cubes (a c : ℤ) (h_a : a = 6) (h_c : c = 3) : 
\((a^3 + c^3) \div (a^2 - ac + c^2) = 9\) := sorry

end division_of_cubes_l72_72210


namespace no_tangent_slope_2_l72_72064

theorem no_tangent_slope_2 (x : ℝ) (hx : x > 1) : 
  ¬(deriv (λ x : ℝ, 1 / (Real.log x)) x = 2) :=
sorry

end no_tangent_slope_2_l72_72064


namespace joan_bought_72_eggs_l72_72094

def dozen := 12
def joan_eggs (dozens: Nat) := dozens * dozen

theorem joan_bought_72_eggs : joan_eggs 6 = 72 :=
by
  sorry

end joan_bought_72_eggs_l72_72094


namespace rectangle_of_incenters_l72_72460

-- Cyclic quadrilateral and respective incircle centers.
variables {A B C D I_A I_B I_C I_D : Type}
variables [CyclicQuadrilateral ABCD]
variables [Incenter I_A (Triangle BCD)]
variables [Incenter I_B (Triangle DCA)]
variables [Incenter I_C (Triangle ADB)]
variables [Incenter I_D (Triangle BAC)]

-- Prove that I_A I_B I_C I_D forms a rectangle.
theorem rectangle_of_incenters
  (h_cyclic: CyclicQuadrilateral ABCD)
  (h_I_A: Incenter I_A (Triangle BCD))
  (h_I_B: Incenter I_B (Triangle DCA))
  (h_I_C: Incenter I_C (Triangle ADB))
  (h_I_D: Incenter I_D (Triangle BAC)) :
  Rectangle I_A I_B I_C I_D :=
sorry

end rectangle_of_incenters_l72_72460


namespace a_2012_eq_6_l72_72080

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧
  a 2 = 6 ∧
  ∀ n ≥ 2, a (n + 1) = a n + a (n - 1)

theorem a_2012_eq_6 (a : ℕ → ℤ) (h : sequence a) : a 2012 = 6 :=
sorry

end a_2012_eq_6_l72_72080


namespace number_of_correct_propositions_is_zero_l72_72286

-- Definitions of vector properties and the conditions
def are_collinear (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2, k * b.3)
def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)
def are_skew (a b : ℝ × ℝ × ℝ) : Prop := ¬ ∃ p : ℝ × ℝ × ℝ, ∃ q : ℝ × ℝ × ℝ, (a + p = b + q)
def are_coplanar (a b c : ℝ × ℝ × ℝ) : Prop := det ![
  [a.1, a.2, a.3],
  [b.1, b.2, b.3],
  [c.1, c.2, c.3]
] = 0

-- Conditions translations
axiom condition1 (a b : ℝ × ℝ × ℝ) : are_collinear a b → are_parallel a b
axiom condition2 (a b : ℝ × ℝ × ℝ) : are_skew a b → ¬ are_coplanar a b
axiom condition3 (a b c : ℝ × ℝ × ℝ) : are_coplanar a b c → ∀ p q : ℝ × ℝ × ℝ, (a = p ∨ a = q) ∧ (b = p ∨ b = q) ∧ (c = p ∨ c = q) → are_coplanar a b c
axiom condition4 (a b c p : ℝ × ℝ × ℝ) : ∀ x y z : ℝ, p = (x * a.1 + y * a.2 + z * a.3, x * b.1 + y * b.2 + z * b.3) → ¬ are_coplanar a b c

-- Proving the number of correct propositions
theorem number_of_correct_propositions_is_zero : 0 = 0 :=
by
  have incorrect1 : ¬ condition1 := sorry
  have incorrect2 : ¬ condition2 := sorry
  have incorrect3 : ¬ condition3 := sorry
  have incorrect4 : ¬ condition4 := sorry
  have correct_props_count : Nat := 0
  show 0 = correct_props_count from rfl

end number_of_correct_propositions_is_zero_l72_72286


namespace find_norm_a_l72_72371

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given Conditions
axiom angle_ab : ∀ ⦃a b : EuclideanSpace ℝ (Fin 3)⦄, ∠a b = π / 3
axiom norm_b : ∥b∥ = 2
axiom norm_b_minus_2a : ∥b - 2 • a∥ = 2

-- Proof Statement
theorem find_norm_a : ∥a∥ = 1 :=
sorry

end find_norm_a_l72_72371


namespace find_f_neg_2016_l72_72019

noncomputable def f : ℝ → ℝ
| x => if h : x > 2 then f (x+5)
       else if h : x < -2 then f (-x)
       else Real.exp x

theorem find_f_neg_2016 : f (-2016) = Real.exp 1 :=
by
  sorry

end find_f_neg_2016_l72_72019


namespace fixed_point_through_BC_l72_72514

variables {Point : Type} [MetricSpace Point]

def OnCircle (O : Circle Point) (A : Point) : Prop := A ∈ O.points
def OnLine (l : Line Point) (P : Point) : Prop := P ∈ l.points
def Intersect (seg1 seg2 : Segment Point) : Point := sorry  -- Intersecting point (abstracted)

theorem fixed_point_through_BC 
  (O : Circle Point) (l : Line Point)
  (A E F G B D C H : Point)
  (hA_on_O : OnCircle O A) 
  (hE_on_l : OnLine l E) 
  (hF_on_l : OnLine l F) 
  (hG_on_l : OnLine l G)
  (hB_intersect : Intersect (Segment.mk A E) (O.points) = B)
  (hD_intersect : Intersect (Segment.mk A G) (O.points) = D)
  (hC_intersect : Intersect (Segment.mk F D) (O.points) = C)
  (hH_on_l : OnLine l H) : 
  ∃ H : Point, OnLine l H ∧ ∀ A', OnCircle O A' → 
  PassThrough (LineThroughPoints B C) H :=
sorry

end fixed_point_through_BC_l72_72514


namespace land_area_correct_l72_72268

-- Define the parameters of the trapezoid
structure Trapezoid where
  a b c d : ℝ -- sides of the trapezoid
  h : ℝ       -- height of the trapezoid
  
-- Given conditions: The sides of the trapezoid
def trapezoid_land : Trapezoid :=
  { a := 2100, b := 1500, c := 613, d := 37, h := 35 }
  
-- Conversion factor for square meters to négyszögöl
noncomputable def conversion_factor : ℝ := 3.596

-- The target area in négyszögöl
noncomputable def target_area : ℝ := 17519

-- Theorem: Prove the area of the land in négyszögöl
theorem land_area_correct (t : Trapezoid) (cf : ℝ) : 
  t.a = 2100 → 
  t.b = 1500 → 
  t.c = 613 → 
  t.d = 37 → 
  t.h = 35 → 
  cf = 3.596 → 
  ((t.a + t.b) / 2) * t.h / cf = target_area := 
by
  sorry

end land_area_correct_l72_72268


namespace geometric_sequence_a5_l72_72077

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6)
  (h2 : a 3 + a 5 + a 7 = 78)
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  a 5 = 18 :=
by sorry

end geometric_sequence_a5_l72_72077


namespace number_of_correct_propositions_l72_72693

open Set

def power_set (A : Set α) : Set (Set α) := { B | B ⊆ A }

def number_of_elements (A : Set α) [Finite A] : Nat := Finset.card (toFinset A)

theorem number_of_correct_propositions :
  (∀ (A : Set α), A ⊆ power_set A) ∧
  ¬(∃ (A : Set α) [Finite A], number_of_elements (power_set A) = 3) ∧
  (∀ (A B : Set α), A ∩ B = ∅ → power_set A ∩ power_set B = ∅) ∧
  (∀ (A B : Set α), A ⊆ B → power_set A ⊆ power_set B) ∧
  (∀ (A B : Set α) [Finite A] [Finite B], number_of_elements A - number_of_elements B = 1 →
    number_of_elements (power_set A) = 2 * number_of_elements (power_set B)) → 
  3 :=
by
  sorry

end number_of_correct_propositions_l72_72693


namespace function_passes_through_fixed_point_l72_72921

variables (a : ℝ) (x y : ℝ)

noncomputable def f : ℝ → ℝ := λ x, a^(x - 1) + 1

theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) : 
  f a 1 = 2 := by
sorry

end function_passes_through_fixed_point_l72_72921


namespace select_14_languages_l72_72156

theorem select_14_languages 
  (n : ℕ) 
  (h_n_ge_9 : n ≥ 9) 
  (employees : Finset (Finset ℕ)) 
  (langs : Finset ℕ) 
  (h_card_employees : employees.card = 500)
  (h_card_langs : langs.card = 2 * n)
  (h_employee_speaks_n_langs : ∀ e ∈ employees, (e ∩ langs).card ≥ n) :
  ∃ s ⊆ langs, s.card = 14 ∧ ∀ e ∈ employees, (e ∩ s).nonempty :=
sorry

end select_14_languages_l72_72156


namespace second_is_trululu_l72_72284

-- Problem statement definitions
def first_brother_says (s : String) : Prop := s = "I am Tweedledum"
def second_brother_says (s : String) : Prop := s = "That's him!"
def is_trululu (person : String) : Prop  -- This is the property of being Trululu, who always lies

-- Conditions provided by the problem
def first_brother : String := "brother_1"
def second_brother : String := "brother_2"
axiom firstsays : first_brother_says "I am Tweedledum"
axiom secondsays : second_brother_says "That's him!"

-- Assert the conclusion that second brother is Trululu based on the conditions
theorem second_is_trululu : is_trululu second_brother :=
by
  sorry -- The proof is omitted as per instruction

end second_is_trululu_l72_72284


namespace unique_solution_3x_4y_5z_l72_72324

theorem unique_solution_3x_4y_5z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  intro h
  sorry

end unique_solution_3x_4y_5z_l72_72324


namespace initial_amount_owed_l72_72449

theorem initial_amount_owed (interest_rate : ℝ := 0.10) (compounding_periods : ℕ := 12) (interest_second_month : ℝ := 22) :
  let monthly_rate := interest_rate / compounding_periods in
  let P := interest_second_month / (monthly_rate) in
  abs (P - 2640.96) < 0.01 :=
by
  let monthly_rate := interest_rate / compounding_periods
  let P := interest_second_month / monthly_rate
  sorry

end initial_amount_owed_l72_72449


namespace rectangle_length_decreased_by_28_57_percent_l72_72950

theorem rectangle_length_decreased_by_28_57_percent (w l : Real) (h1 : w > 0) (h2 : l > 0) :
  let l' := l / 1.4 in 
  let percentage_decrease := (1 - (l' / l)) * 100 in
  percentage_decrease ≈ 28.57 :=
by
  sorry

end rectangle_length_decreased_by_28_57_percent_l72_72950


namespace total_number_of_wheels_l72_72823

-- Define the conditions as hypotheses
def cars := 2
def wheels_per_car := 4

def bikes := 2
def trashcans := 1
def wheels_per_bike_or_trashcan := 2

def roller_skates_pair := 1
def wheels_per_skate := 4

def tricycle := 1
def wheels_per_tricycle := 3

-- Prove the total number of wheels
theorem total_number_of_wheels :
  cars * wheels_per_car +
  (bikes + trashcans) * wheels_per_bike_or_trashcan +
  (roller_skates_pair * 2) * wheels_per_skate +
  tricycle * wheels_per_tricycle 
  = 25 :=
by
  sorry

end total_number_of_wheels_l72_72823


namespace g_passing_point_iff_t_neg_e_extremum_at_2_for_t_le_0_two_extrema_in_0_to_2_iff_t_range_l72_72485

noncomputable def g (x : ℝ) (t : ℝ) : ℝ := (Real.exp x) / (x^2) - t * (2 / x + Real.log x)

theorem g_passing_point_iff_t_neg_e (t : ℝ) :
  g 1 t = 3 * Real.exp 1 ↔ t = -Real.exp 1 := by
  sorry

theorem extremum_at_2_for_t_le_0 (t : ℝ) (ht : t ≤ 0) :
  ∃ x, x = 2 ∧ deriv (λ x, g x t) x = 0 := by
  sorry

theorem two_extrema_in_0_to_2_iff_t_range (t : ℝ) :
  (∃ x1 x2, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ x1 ≠ x2 ∧ deriv (λ x, g x t) x1 = 0 ∧ deriv (λ x, g x t) x2 = 0) ↔
  e < t ∧ t < (Real.exp (2 : ℝ)) / 2 := by
  sorry

end g_passing_point_iff_t_neg_e_extremum_at_2_for_t_le_0_two_extrema_in_0_to_2_iff_t_range_l72_72485


namespace variance_transformed_list_l72_72012

noncomputable def stddev (xs : List ℝ) : ℝ := sorry
noncomputable def variance (xs : List ℝ) : ℝ := sorry

theorem variance_transformed_list :
  ∀ (a_1 a_2 a_3 a_4 a_5 : ℝ),
  stddev [a_1, a_2, a_3, a_4, a_5] = 2 →
  variance [3 * a_1 - 2, 3 * a_2 - 2, 3 * a_3 - 2, 3 * a_4 - 2, 3 * a_5 - 2] = 36 :=
by
  intros
  sorry

end variance_transformed_list_l72_72012


namespace prob_X_ge_2_l72_72733

open ProbabilityTheory

noncomputable def normal_dist_X : Measure ℝ := measure_theory.measure_gaussian (3 : ℝ) (σ^2 : ℝ)

theorem prob_X_ge_2 : (ProbabilityTheory.probability (normal_dist_X (set.Ici (2 : ℝ)))) = 0.85 :=
begin
  sorry
end

end prob_X_ge_2_l72_72733


namespace part_a_part_b_l72_72245

-- Part (a)
theorem part_a :
  (inf { abs (x * Real.sqrt 2 + y * Real.sqrt 3 + z * Real.sqrt 5) |
    x y z : ℤ, x^2 + y^2 + z^2 > 0 } = 0) :=
sorry

-- Part (b)
theorem part_b :
  ∃ (a b c : ℚ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (∀ ε > 0, ∃ x y z : ℤ, abs (x*a + y*b + z*c) < ε → abs (x*a + y*b + z*c) = 0) :=
sorry

end part_a_part_b_l72_72245


namespace find_k_l72_72417

theorem find_k (k : ℝ) (h : (2 * (7:ℝ)^2) + 3 * 7 - k = 0) : k = 119 := by
  sorry

end find_k_l72_72417


namespace carson_speed_l72_72448

def Jerry_one_way_time : ℕ := 15 -- Jerry takes 15 minutes one-way
def school_distance : ℕ := 4 -- Distance to the school is 4 miles

theorem carson_speed :
  let carson_time_minutes := Jerry_one_way_time * 2,
      carson_time_hours := carson_time_minutes / 60,
      carson_speed := school_distance / carson_time_hours
  in carson_speed = 8 := by
  sorry

end carson_speed_l72_72448


namespace ratio_of_ages_l72_72541

theorem ratio_of_ages (x y : ℕ) (h1 : 8 * x + y = 245) (h2 : y - 3 * x = 80) :
  (5 * x) : y = 3 : 5 :=
by
  sorry

end ratio_of_ages_l72_72541


namespace sine_of_smaller_angle_and_admissible_k_l72_72922

theorem sine_of_smaller_angle_and_admissible_k
  (α : ℝ) (k : ℝ) (h₁ : 0 < α) (h₂ : 2 * α < π) (h₃ : k > 2) :
  (real.sin (π / 2 - 2 * α) = 1 / (k - 1)) ∧ (2 < k) :=
by
  sorry

end sine_of_smaller_angle_and_admissible_k_l72_72922


namespace mary_cut_roses_l72_72191

theorem mary_cut_roses (initial_roses add_roses total_roses : ℕ) (h1 : initial_roses = 6) (h2 : total_roses = 16) (h3 : total_roses = initial_roses + add_roses) : add_roses = 10 :=
by
  sorry

end mary_cut_roses_l72_72191


namespace speed_ratio_l72_72241

variable (v_A v_B : ℝ)

def equidistant_3min : Prop := 3 * v_A = abs (-800 + 3 * v_B)
def equidistant_8min : Prop := 8 * v_A = abs (-800 + 8 * v_B)
def speed_ratio_correct : Prop := v_A / v_B = 1 / 2

theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_8min v_A v_B) : speed_ratio_correct v_A v_B :=
by
  sorry

end speed_ratio_l72_72241


namespace find_sum_u_v_l72_72756

theorem find_sum_u_v : ∃ (u v : ℚ), 5 * u - 6 * v = 35 ∧ 3 * u + 5 * v = -10 ∧ u + v = -40 / 43 :=
by
  sorry

end find_sum_u_v_l72_72756


namespace right_triangle_hypotenuse_l72_72109

theorem right_triangle_hypotenuse (A B C P Q : Point) (x y BC : ℝ)
  (hABC : right_triangle A B C)
  (hBAC : angle_eq B A C (π / 2))
  (hP : midpoint P A B)
  (hQ : midpoint Q A C)
  (hQC : dist Q C = 25)
  (hPB : dist P B = 28)
  (hx : dist A P = x)
  (hy : dist A Q = y)
  (h1 : y^2 + x^2 = 281.8)
  (hBC : BC = sqrt (y^2 + x^2)) :
  BC = 34 :=
by
  sorry

end right_triangle_hypotenuse_l72_72109


namespace even_internal_triangles_l72_72306

noncomputable def is_internal_triangle (A : Finset (EuclideanSpace ℝ (Fin 2))) (T : Finset (EuclideanSpace ℝ (Fin 2))) : Prop :=
  T ⊆ A ∧ T.card = 3

theorem even_internal_triangles (A : Finset (EuclideanSpace ℝ (Fin 2)))
  (h_card : A.card = 2009)
  (h_no_collinear : ∀ (T : Finset (EuclideanSpace ℝ (Fin 2))), T ⊆ A ∧ T.card = 3 → ¬ collinear T) :
  ∀ P ∈ A, even (Finset.card {T : Finset (EuclideanSpace ℝ (Fin 2)) | is_internal_triangle A T ∧ P ∈ T}) :=
by
  sorry

end even_internal_triangles_l72_72306


namespace area_CDE_l72_72871

variable (A B C D E F : Type) [RealInnerProductSpace ℝ A] [T2Space A]

def point_on_triangle_AC (D : A) (AC : Set A) (D_on_AC : D ∈ AC) := sorry
def point_on_triangle_BC (E : A) (BC : Set A) (E_on_BC : E ∈ BC) := sorry
def intersection_point (AE : Set A) (BD : Set A) (F : A) (F_int : F ∈ (AE ∩ BD)) := sorry

theorem area_CDE :
  (point_on_triangle_AC D (segment A C) sorry) →
  (point_on_triangle_BC E (segment B C) sorry) →
  (intersection_point (segment A E) (segment B D) F sorry) →
  (area_of_triangle A B F = 1) →
  (area_of_triangle A D F = 1 / 4) →
  (area_of_triangle B E F = 1 / 5) →
  area_of_triangle C D E = 3 / 38 := sorry

end area_CDE_l72_72871


namespace find_c_and_root_situation_l72_72217

theorem find_c_and_root_situation (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 4) (h₃ : c - 2) (x : ℝ) (hx : x = -1) : 
  c = 5 ∧ (b^2 - 4 * a * c < 0) :=
by
  sorry

end find_c_and_root_situation_l72_72217


namespace find_s_l72_72111

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem find_s (p q r s : ℝ)
  (h1 : ∀ (x : ℝ), g x p q r s = (x + 1) * (x + 10) * (x + 10) * (x + 10))
  (h2 : p + q + r + s = 2673) :
  s = 1000 := 
  sorry

end find_s_l72_72111


namespace average_trees_planted_l72_72810

def A := 225
def B := A + 48
def C := A - 24
def total_trees := A + B + C
def average := total_trees / 3

theorem average_trees_planted :
  average = 233 := by
  sorry

end average_trees_planted_l72_72810


namespace find_speed_of_boat_l72_72566

noncomputable def speed_of_boat_in_still_water 
  (v : ℝ) 
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) : Prop :=
  v = 42

theorem find_speed_of_boat 
  (v : ℝ)
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) 
  (h1 : v + current_speed = distance / (time_in_minutes / 60)) : 
  speed_of_boat_in_still_water v :=
by
  sorry

end find_speed_of_boat_l72_72566


namespace EF_squared_eq_AE_squared_plus_BF_squared_l72_72075

variable {A B C D E F : Type*}
variable [metric_space E] [metric_space F] [add_comm_group F] [vector_space ℝ F]

-- Definitions based on the given problem conditions
def is_right_angle_triangle (A B C : E) : Prop :=
  ∃ (t : ℝ), dist A C ^ 2 + dist B C ^ 2 = dist A B ^ 2

def midpoint (D : E) (A B : E) : Prop :=
  2 * dist D A = dist A B ∧ 2 * dist D B = dist A B

def perpendicular (D E F : E) : Prop :=
  dist D E ^ 2 + dist D F ^ 2 = dist E F ^ 2

-- Problem statement in Lean 4
theorem EF_squared_eq_AE_squared_plus_BF_squared
  (A B C D E F : E)
  (h1 : is_right_angle_triangle A B C)
  (h2 : midpoint D A B)
  (h3 : D = midpoint_point A B)
  (h4 : E ∈ line_segment A C)
  (h5 : F ∈ line_segment B C)
  (h6 : perpendicular D E F) :
  dist E F ^ 2 = dist A E ^ 2 + dist B F ^ 2 :=
sorry

end EF_squared_eq_AE_squared_plus_BF_squared_l72_72075
