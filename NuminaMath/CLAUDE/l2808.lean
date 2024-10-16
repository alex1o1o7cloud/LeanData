import Mathlib

namespace NUMINAMATH_CALUDE_cosine_symmetric_minimum_l2808_280836

open Real

theorem cosine_symmetric_minimum (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, 0 < x → x < 2 → f x = cos (π * x)) →
  0 < a → a < 2 →
  0 < b → b < 2 →
  a ≠ b →
  f a = f b →
  (∀ x y, 0 < x → x < 2 → 0 < y → y < 2 → x ≠ y → f x = f y → 1/x + 4/y ≥ 9/2) →
  ∃ x y, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x ≠ y ∧ f x = f y ∧ 1/x + 4/y = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetric_minimum_l2808_280836


namespace NUMINAMATH_CALUDE_root_between_roots_l2808_280859

theorem root_between_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + b = 0)
  (h2 : ∃ y : ℝ, y^2 - a*y + b = 0) :
  ∃ (x_1 y_1 z : ℝ), x_1^2 + a*x_1 + b = 0 ∧
                     y_1^2 - a*y_1 + b = 0 ∧
                     z^2 + 2*a*z + 2*b = 0 ∧
                     ((x_1 < z ∧ z < y_1) ∨ (y_1 < z ∧ z < x_1)) :=
by sorry

end NUMINAMATH_CALUDE_root_between_roots_l2808_280859


namespace NUMINAMATH_CALUDE_gabrielles_peaches_l2808_280806

theorem gabrielles_peaches (martine benjy gabrielle : ℕ) 
  (h1 : martine = 2 * benjy + 6)
  (h2 : benjy = gabrielle / 3)
  (h3 : martine = 16) :
  gabrielle = 15 := by
  sorry

end NUMINAMATH_CALUDE_gabrielles_peaches_l2808_280806


namespace NUMINAMATH_CALUDE_root_difference_l2808_280807

theorem root_difference (x : ℝ) : 
  let equation := fun x => x^2 + 42*x + 420 + 48
  let roots := {r : ℝ | equation r = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 6 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_l2808_280807


namespace NUMINAMATH_CALUDE_star_seven_two_l2808_280850

def star (a b : ℤ) : ℤ := 4 * a - 4 * b

theorem star_seven_two : star 7 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_two_l2808_280850


namespace NUMINAMATH_CALUDE_electronic_dogs_distance_l2808_280854

/-- Represents a vertex of a cube --/
inductive Vertex
| A | B | C | D | A1 | B1 | C1 | D1

/-- Represents the position of an electronic dog on the cube --/
structure DogPosition where
  vertex : Vertex
  segments_completed : Nat

/-- The cube with edge length 1 --/
def unitCube : Set Vertex := {Vertex.A, Vertex.B, Vertex.C, Vertex.D, Vertex.A1, Vertex.B1, Vertex.C1, Vertex.D1}

/-- The distance between two vertices of the unit cube --/
def distance (v1 v2 : Vertex) : Real := sorry

/-- The movement rule for the dogs --/
def validMove (v1 v2 v3 : Vertex) : Prop := sorry

/-- The final position of the black dog after 2008 segments --/
def blackDogFinalPosition : DogPosition := ⟨Vertex.A, 2008⟩

/-- The final position of the yellow dog after 2009 segments --/
def yellowDogFinalPosition : DogPosition := ⟨Vertex.A1, 2009⟩

theorem electronic_dogs_distance :
  distance blackDogFinalPosition.vertex yellowDogFinalPosition.vertex = 1 := by sorry

end NUMINAMATH_CALUDE_electronic_dogs_distance_l2808_280854


namespace NUMINAMATH_CALUDE_sum_largest_smallest_even_le_49_l2808_280892

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def largest_even_le_49 : ℕ := 48

def smallest_even_gt_0_le_49 : ℕ := 2

theorem sum_largest_smallest_even_le_49 :
  largest_even_le_49 + smallest_even_gt_0_le_49 = 50 ∧
  is_even largest_even_le_49 ∧
  is_even smallest_even_gt_0_le_49 ∧
  largest_even_le_49 ≤ 49 ∧
  smallest_even_gt_0_le_49 > 0 ∧
  smallest_even_gt_0_le_49 ≤ 49 ∧
  ∀ n, is_even n ∧ n > 0 ∧ n ≤ 49 → n ≤ largest_even_le_49 ∧ n ≥ smallest_even_gt_0_le_49 :=
by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_even_le_49_l2808_280892


namespace NUMINAMATH_CALUDE_unique_n_with_divisor_property_l2808_280864

def has_ten_divisors (n : ℕ) : Prop :=
  ∃ (d : Fin 10 → ℕ), d 0 = 1 ∧ d 9 = n ∧
    (∀ i : Fin 9, d i < d (i + 1)) ∧
    (∀ m : ℕ, m ∣ n ↔ ∃ i : Fin 10, d i = m)

theorem unique_n_with_divisor_property :
  ∀ n : ℕ, n > 0 →
    has_ten_divisors n →
    (∃ (d : Fin 10 → ℕ), 2 * n = (d 4)^2 + (d 5)^2 - 1) →
    n = 272 :=
sorry

end NUMINAMATH_CALUDE_unique_n_with_divisor_property_l2808_280864


namespace NUMINAMATH_CALUDE_third_person_contribution_l2808_280814

theorem third_person_contribution
  (total : ℕ)
  (h_total : total = 1040)
  (x : ℕ)
  (h_brittany : 3 * x = Brittany)
  (h_angela : 3 * Brittany = Angela)
  (h_sum : x + Brittany + Angela = total) :
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_third_person_contribution_l2808_280814


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2808_280813

theorem stratified_sample_size 
  (population_ratio_A : ℚ) 
  (population_ratio_B : ℚ) 
  (population_ratio_C : ℚ) 
  (sample_size_A : ℕ) 
  (total_sample_size : ℕ) :
  population_ratio_A = 3 / 14 →
  population_ratio_B = 4 / 14 →
  population_ratio_C = 7 / 14 →
  sample_size_A = 15 →
  population_ratio_A = sample_size_A / total_sample_size →
  total_sample_size = 70 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2808_280813


namespace NUMINAMATH_CALUDE_range_of_a_l2808_280835

def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

theorem range_of_a :
  ∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2808_280835


namespace NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l2808_280823

theorem sqrt_31_between_5_and_6 : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l2808_280823


namespace NUMINAMATH_CALUDE_rational_equality_l2808_280827

theorem rational_equality (n : ℕ) (x y : ℚ) 
  (h_odd : Odd n) 
  (h_pos : 0 < n) 
  (h_eq : x^n + 2*y = y^n + 2*x) : 
  x = y := by sorry

end NUMINAMATH_CALUDE_rational_equality_l2808_280827


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2808_280890

theorem matrix_inverse_proof : 
  let M : Matrix (Fin 3) (Fin 3) ℚ := !![4/11, 3/11, 0; -1/11, 2/11, 0; 0, 0, 1/3]
  let A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -3, 0; 1, 4, 0; 0, 0, 3]
  M * A = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2808_280890


namespace NUMINAMATH_CALUDE_train_length_l2808_280883

/-- The length of a train crossing a bridge -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 200 →
  crossing_time = 60 →
  train_speed = 5 →
  bridge_length + (train_speed * crossing_time - bridge_length) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2808_280883


namespace NUMINAMATH_CALUDE_symmetric_axis_of_shifted_function_l2808_280822

/-- Given a function f(x) = √3 * sin(2x) - cos(2x), prove that when shifted right by π/3 units,
    one of its symmetric axes is given by the equation x = π/6 -/
theorem symmetric_axis_of_shifted_function :
  ∃ (f : ℝ → ℝ) (g : ℝ → ℝ),
    (∀ x, f x = Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)) ∧
    (∀ x, g x = f (x - π / 3)) ∧
    (∀ x, g x = g (π / 3 - x)) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_shifted_function_l2808_280822


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2808_280867

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 2 * (b + c) →
  b = 5 * c →
  a * b * c = 2500 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2808_280867


namespace NUMINAMATH_CALUDE_multiplicative_inverse_problem_l2808_280858

theorem multiplicative_inverse_problem :
  let A : ℕ := 111112
  let B : ℕ := 142858
  let M : ℕ := 1000003
  let N : ℕ := 513487
  (A * B * N) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_problem_l2808_280858


namespace NUMINAMATH_CALUDE_apartments_can_decrease_l2808_280898

/-- Represents a building configuration -/
structure Building where
  entrances : ℕ
  floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in a building -/
def total_apartments (b : Building) : ℕ :=
  b.entrances * b.floors * b.apartments_per_floor

/-- Represents the modifications made to a building -/
structure Modification where
  entrances_removed : ℕ
  floors_added : ℕ

/-- Applies a modification to a building -/
def apply_modification (b : Building) (m : Modification) : Building :=
  { entrances := b.entrances - m.entrances_removed,
    floors := b.floors + m.floors_added,
    apartments_per_floor := b.apartments_per_floor }

/-- Theorem: It's possible for the number of apartments to decrease after modifications -/
theorem apartments_can_decrease (initial : Building) (mod1 mod2 : Modification) :
  ∃ (final : Building),
    final = apply_modification (apply_modification initial mod1) mod2 ∧
    total_apartments final < total_apartments initial :=
  sorry


end NUMINAMATH_CALUDE_apartments_can_decrease_l2808_280898


namespace NUMINAMATH_CALUDE_largest_m_binomial_sum_l2808_280888

theorem largest_m_binomial_sum (m : ℕ) : (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_m_binomial_sum_l2808_280888


namespace NUMINAMATH_CALUDE_sequence_periodicity_l2808_280879

def sequence_rule (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, n > 1 → x (n + 1) = |x n| - x (n - 1)

theorem sequence_periodicity (x : ℤ → ℝ) (h : sequence_rule x) :
  ∀ k : ℤ, x (k + 9) = x k ∧ x (k + 8) = x (k - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l2808_280879


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_norm_l2808_280899

/-- Two vectors in ℝ² -/
def a (x : ℝ) : Fin 2 → ℝ := ![x + 1, 2]
def b : Fin 2 → ℝ := ![1, -1]

/-- Parallel vectors have proportional components -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * w i

theorem parallel_vectors_sum_norm (x : ℝ) :
  parallel (a x) b → ‖(a x) + b‖ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_norm_l2808_280899


namespace NUMINAMATH_CALUDE_johns_goals_l2808_280818

theorem johns_goals (total_goals : ℝ) (teammate_count : ℕ) (avg_teammate_goals : ℝ) :
  total_goals = 65 ∧
  teammate_count = 9 ∧
  avg_teammate_goals = 4.5 →
  total_goals - (teammate_count : ℝ) * avg_teammate_goals = 24.5 :=
by sorry

end NUMINAMATH_CALUDE_johns_goals_l2808_280818


namespace NUMINAMATH_CALUDE_c_grazing_months_l2808_280881

/-- Represents the number of oxen-months for each person -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Represents the total rent of the pasture -/
def total_rent : ℕ := 175

/-- Represents c's share of the rent -/
def c_share : ℕ := 45

/-- Theorem stating that c put his oxen for grazing for 3 months -/
theorem c_grazing_months :
  ∃ (x : ℕ),
    x = 3 ∧
    c_share * (oxen_months 10 7 + oxen_months 12 5 + oxen_months 15 x) =
    total_rent * oxen_months 15 x :=
by sorry

end NUMINAMATH_CALUDE_c_grazing_months_l2808_280881


namespace NUMINAMATH_CALUDE_a_union_b_iff_c_l2808_280819

-- Define sets A, B, and C
def A : Set ℝ := {x | x - 2 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- Theorem statement
theorem a_union_b_iff_c : ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C := by sorry

end NUMINAMATH_CALUDE_a_union_b_iff_c_l2808_280819


namespace NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l2808_280815

/-- The area of a triangle given two sides and the angle bisector between them. -/
theorem triangle_area_with_angle_bisector 
  (a b f_c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf_c : f_c > 0) 
  (h_triangle : 4 * a^2 * b^2 > (a + b)^2 * f_c^2) : 
  ∃ t : ℝ, t = ((a + b) * f_c) / (4 * a * b) * Real.sqrt (4 * a^2 * b^2 - (a + b)^2 * f_c^2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_with_angle_bisector_l2808_280815


namespace NUMINAMATH_CALUDE_star_to_square_ratio_is_three_fifths_l2808_280809

/-- Represents a square with side length 5 cm containing a star formed by four identical isosceles triangles, each with height 1 cm -/
structure StarInSquare where
  square_side : ℝ
  triangle_height : ℝ
  square_side_eq : square_side = 5
  triangle_height_eq : triangle_height = 1

/-- Calculates the ratio of the star area to the square area -/
def star_to_square_ratio (s : StarInSquare) : ℚ :=
  3 / 5

/-- Theorem stating that the ratio of the star area to the square area is 3/5 -/
theorem star_to_square_ratio_is_three_fifths (s : StarInSquare) :
  star_to_square_ratio s = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_star_to_square_ratio_is_three_fifths_l2808_280809


namespace NUMINAMATH_CALUDE_net_pay_calculation_l2808_280804

/-- Calculate net pay given gross pay and taxes paid -/
def net_pay (gross_pay taxes_paid : ℕ) : ℕ :=
  gross_pay - taxes_paid

/-- Theorem: Given the conditions, prove that the net pay is 315 dollars -/
theorem net_pay_calculation (gross_pay taxes_paid : ℕ) 
  (h1 : gross_pay = 450)
  (h2 : taxes_paid = 135) :
  net_pay gross_pay taxes_paid = 315 := by
  sorry

end NUMINAMATH_CALUDE_net_pay_calculation_l2808_280804


namespace NUMINAMATH_CALUDE_parcera_triples_l2808_280852

def isParcera (p q r : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  p ∣ (q^2 - 4) ∧ q ∣ (r^2 - 4) ∧ r ∣ (p^2 - 4)

theorem parcera_triples :
  ∀ p q r : Nat, isParcera p q r ↔ 
    ((p, q, r) = (2, 2, 2) ∨ 
     (p, q, r) = (5, 3, 7) ∨ 
     (p, q, r) = (7, 5, 3) ∨ 
     (p, q, r) = (3, 7, 5)) :=
by sorry

end NUMINAMATH_CALUDE_parcera_triples_l2808_280852


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l2808_280868

theorem doctors_lawyers_ratio (d l : ℕ) (h_total : d + l > 0) :
  (45 * d + 55 * l) / (d + l) = 47 →
  d = 4 * l :=
by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l2808_280868


namespace NUMINAMATH_CALUDE_f_g_3_eq_6_l2808_280894

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := x^2 - 8

theorem f_g_3_eq_6 : f (g 3) = 6 := by sorry

end NUMINAMATH_CALUDE_f_g_3_eq_6_l2808_280894


namespace NUMINAMATH_CALUDE_sue_necklace_purple_beads_l2808_280895

theorem sue_necklace_purple_beads :
  ∀ (purple blue green : ℕ),
    purple + blue + green = 46 →
    blue = 2 * purple →
    green = blue + 11 →
    purple = 7 := by
  sorry

end NUMINAMATH_CALUDE_sue_necklace_purple_beads_l2808_280895


namespace NUMINAMATH_CALUDE_gold_silver_board_theorem_l2808_280877

/-- A board configuration with gold and silver cells -/
structure Board :=
  (size : Nat)
  (is_gold : Fin size → Fin size → Bool)

/-- Count gold cells in a rectangle -/
def count_gold (b : Board) (x y w h : Nat) : Nat :=
  (Finset.range w).sum (λ i =>
    (Finset.range h).sum (λ j =>
      if b.is_gold ⟨x + i, sorry⟩ ⟨y + j, sorry⟩ then 1 else 0))

/-- Property that each 3x3 square has A gold cells -/
def three_by_three_property (b : Board) (A : Nat) : Prop :=
  ∀ x y, x + 3 ≤ b.size → y + 3 ≤ b.size →
    count_gold b x y 3 3 = A

/-- Property that each 2x4 or 4x2 rectangle has Z gold cells -/
def two_by_four_property (b : Board) (Z : Nat) : Prop :=
  (∀ x y, x + 2 ≤ b.size → y + 4 ≤ b.size →
    count_gold b x y 2 4 = Z) ∧
  (∀ x y, x + 4 ≤ b.size → y + 2 ≤ b.size →
    count_gold b x y 4 2 = Z)

/-- The main theorem -/
theorem gold_silver_board_theorem :
  ∀ (b : Board) (A Z : Nat),
    b.size = 2016 →
    three_by_three_property b A →
    two_by_four_property b Z →
    ((A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8)) :=
sorry

end NUMINAMATH_CALUDE_gold_silver_board_theorem_l2808_280877


namespace NUMINAMATH_CALUDE_point_side_line_range_l2808_280875

/-- Given that points (3,1) and (-4,6) are on the same side of the line 3x-2y+a=0,
    the range of values for a is a < -7 or a > 24 -/
theorem point_side_line_range (a : ℝ) : 
  ((3 * 3 - 2 * 1 + a) * (-4 * 3 - 2 * 6 + a) > 0) ↔ (a < -7 ∨ a > 24) := by
  sorry

end NUMINAMATH_CALUDE_point_side_line_range_l2808_280875


namespace NUMINAMATH_CALUDE_min_value_sum_sqrt_ratios_equality_condition_l2808_280829

theorem min_value_sum_sqrt_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 / b^2) + Real.sqrt (b^2 / c^2) + Real.sqrt (c^2 / a^2) ≥ 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 / b^2) + Real.sqrt (b^2 / c^2) + Real.sqrt (c^2 / a^2) = 3 ↔ a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_sqrt_ratios_equality_condition_l2808_280829


namespace NUMINAMATH_CALUDE_elephant_drinking_problem_l2808_280805

/-- The number of days it takes for one elephant to drink a lake dry -/
def days_to_drink_lake (V C K : ℝ) : ℝ :=
  365

/-- Theorem stating the conditions and the result for the elephant drinking problem -/
theorem elephant_drinking_problem (V C K : ℝ) 
  (h1 : 183 * C = V + K)
  (h2 : 37 * 5 * C = V + 5 * K)
  (h3 : V > 0)
  (h4 : C > 0)
  (h5 : K > 0) :
  ∃ (t : ℝ), t * C = V + t * K ∧ t = days_to_drink_lake V C K :=
by
  sorry

#check elephant_drinking_problem

end NUMINAMATH_CALUDE_elephant_drinking_problem_l2808_280805


namespace NUMINAMATH_CALUDE_existence_of_distinct_integers_l2808_280826

theorem existence_of_distinct_integers (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n^2 < a ∧ a < (n+1)^2 ∧
    n^2 < b ∧ b < (n+1)^2 ∧
    n^2 < c ∧ c < (n+1)^2 ∧
    (c ∣ a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_distinct_integers_l2808_280826


namespace NUMINAMATH_CALUDE_wendy_polished_110_glasses_l2808_280870

/-- The number of small glasses Wendy polished -/
def small_glasses : ℕ := 50

/-- The additional number of large glasses compared to small glasses -/
def additional_large_glasses : ℕ := 10

/-- The total number of glasses Wendy polished -/
def total_glasses : ℕ := small_glasses + (small_glasses + additional_large_glasses)

/-- Proves that Wendy polished 110 glasses in total -/
theorem wendy_polished_110_glasses : total_glasses = 110 := by
  sorry

end NUMINAMATH_CALUDE_wendy_polished_110_glasses_l2808_280870


namespace NUMINAMATH_CALUDE_bowler_previous_wickets_l2808_280833

/-- Bowling average calculation -/
def bowling_average (runs : ℚ) (wickets : ℚ) : ℚ := runs / wickets

theorem bowler_previous_wickets 
  (initial_average : ℚ)
  (last_match_wickets : ℚ)
  (last_match_runs : ℚ)
  (average_decrease : ℚ)
  (h1 : initial_average = 12.4)
  (h2 : last_match_wickets = 7)
  (h3 : last_match_runs = 26)
  (h4 : average_decrease = 0.4) :
  ∃ (previous_wickets : ℚ),
    previous_wickets = 145 ∧
    bowling_average (initial_average * previous_wickets + last_match_runs) (previous_wickets + last_match_wickets) = initial_average - average_decrease :=
sorry

end NUMINAMATH_CALUDE_bowler_previous_wickets_l2808_280833


namespace NUMINAMATH_CALUDE_christophers_to_gabrielas_age_ratio_l2808_280862

/-- Proves that the ratio of Christopher's age to Gabriela's age is 2:1 given the conditions -/
theorem christophers_to_gabrielas_age_ratio :
  ∀ (c g : ℕ),
  c = 24 →  -- Christopher is now 24 years old
  c - 9 = 5 * (g - 9) →  -- Nine years ago, Christopher was 5 times as old as Gabriela
  c / g = 2 :=  -- The ratio of Christopher's age to Gabriela's age is 2:1
by
  sorry

#check christophers_to_gabrielas_age_ratio

end NUMINAMATH_CALUDE_christophers_to_gabrielas_age_ratio_l2808_280862


namespace NUMINAMATH_CALUDE_root_product_cubic_l2808_280842

theorem root_product_cubic (p q r : ℂ) : 
  (3 * p^3 - 8 * p^2 + p - 9 = 0) →
  (3 * q^3 - 8 * q^2 + q - 9 = 0) →
  (3 * r^3 - 8 * r^2 + r - 9 = 0) →
  p * q * r = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_product_cubic_l2808_280842


namespace NUMINAMATH_CALUDE_kilometer_to_leaps_l2808_280846

/-- Conversion between units of length -/
theorem kilometer_to_leaps 
  (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (h1 : p * (1 : ℝ) = q * (1 : ℝ))  -- p strides = q leaps
  (h2 : r * (1 : ℝ) = s * (1 : ℝ))  -- r bounds = s strides
  (h3 : t * (1 : ℝ) = u * (1 : ℝ))  -- t bounds = u kilometers
  : (1 : ℝ) * (1 : ℝ) = (t * s * q) / (u * r * p) * (1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_kilometer_to_leaps_l2808_280846


namespace NUMINAMATH_CALUDE_divisibility_by_101_l2808_280841

theorem divisibility_by_101 (n : ℕ) : 
  (101 ∣ (10^n - 1)) ↔ (4 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l2808_280841


namespace NUMINAMATH_CALUDE_outfit_count_l2808_280834

/-- The number of outfits that can be made with different colored shirts and hats -/
def number_of_outfits : ℕ :=
  let red_shirts := 7
  let blue_shirts := 5
  let green_shirts := 8
  let pants := 10
  let green_hats := 10
  let red_hats := 6
  let blue_hats := 7
  (red_shirts * pants * (green_hats + blue_hats)) +
  (blue_shirts * pants * (green_hats + red_hats)) +
  (green_shirts * pants * (red_hats + blue_hats))

theorem outfit_count : number_of_outfits = 3030 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l2808_280834


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l2808_280812

theorem consecutive_integers_product (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + d = 109) → (b * c = 2970) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l2808_280812


namespace NUMINAMATH_CALUDE_some_number_value_l2808_280844

theorem some_number_value (n : ℝ) : 
  (0.47 * 1442 - 0.36 * n) + 63 = 3 → n = 2049.28 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2808_280844


namespace NUMINAMATH_CALUDE_material_used_calculation_l2808_280880

-- Define the materials and their quantities
def first_material_bought : ℚ := 4/9
def second_material_bought : ℚ := 2/3
def third_material_bought : ℚ := 5/6

def first_material_left : ℚ := 8/18
def second_material_left : ℚ := 3/9
def third_material_left : ℚ := 2/12

-- Define conversion factors
def sq_meter_to_sq_yard : ℚ := 1196/1000

-- Define the theorem
theorem material_used_calculation :
  let first_used := first_material_bought - first_material_left
  let second_used := second_material_bought - second_material_left
  let third_used := (third_material_bought - third_material_left) * sq_meter_to_sq_yard
  let total_used := first_used + second_used + third_used
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ abs (total_used - 1130666/1000000) < ε :=
by sorry

end NUMINAMATH_CALUDE_material_used_calculation_l2808_280880


namespace NUMINAMATH_CALUDE_plane_speed_l2808_280838

/-- The speed of a plane in still air, given its performance with and against wind. -/
theorem plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) :
  distance_with_wind = 400 →
  distance_against_wind = 320 →
  wind_speed = 20 →
  ∃ (plane_speed : ℝ),
    distance_with_wind / (plane_speed + wind_speed) = distance_against_wind / (plane_speed - wind_speed) ∧
    plane_speed = 180 :=
by sorry

end NUMINAMATH_CALUDE_plane_speed_l2808_280838


namespace NUMINAMATH_CALUDE_gcd_120_168_l2808_280873

theorem gcd_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_120_168_l2808_280873


namespace NUMINAMATH_CALUDE_pool_visitors_l2808_280872

theorem pool_visitors (women : ℕ) (women_students : ℕ) (men_more : ℕ) (men_nonstudents : ℕ) 
  (h1 : women = 1518)
  (h2 : women_students = 536)
  (h3 : men_more = 525)
  (h4 : men_nonstudents = 1257) :
  women_students + ((women + men_more) - men_nonstudents) = 1322 := by
  sorry

end NUMINAMATH_CALUDE_pool_visitors_l2808_280872


namespace NUMINAMATH_CALUDE_solution_value_l2808_280800

/-- Represents a 2x3 augmented matrix --/
def AugmentedMatrix := Matrix (Fin 2) (Fin 3) ℝ

/-- Given augmented matrix --/
def givenMatrix : AugmentedMatrix := !![1, 0, 3; 1, 1, 4]

/-- Theorem: For the system of linear equations represented by the given augmented matrix,
    the value of x + 2y is equal to 5 --/
theorem solution_value (x y : ℝ) 
  (hx : givenMatrix 0 0 * x + givenMatrix 0 1 * y = givenMatrix 0 2)
  (hy : givenMatrix 1 0 * x + givenMatrix 1 1 * y = givenMatrix 1 2) :
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2808_280800


namespace NUMINAMATH_CALUDE_problem_solution_l2808_280878

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 3|
def g (m x : ℝ) : ℝ := m - 2*|x - 11|

-- State the theorem
theorem problem_solution :
  (∀ x m : ℝ, 2 * f x ≥ g m (x + 4)) →
  (∃ t : ℝ, t = 20 ∧ ∀ m : ℝ, (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) → m ≤ t) ∧
  (∀ a : ℝ, a > 0 →
    (∃ x y z : ℝ, 2*x^2 + 3*y^2 + 6*z^2 = a ∧
      ∀ x' y' z' : ℝ, 2*x'^2 + 3*y'^2 + 6*z'^2 = a → x' + y' + z' ≤ 1) →
    a = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2808_280878


namespace NUMINAMATH_CALUDE_function_properties_l2808_280803

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x ∈ Set.Ioo (-3 : ℝ) 2, f a b x > 0) ∧
    (∀ x ∈ Set.Iic (-3 : ℝ) ∪ Set.Ici 2, f a b x < 0) ∧
    (f a b (-3) = 0 ∧ f a b 2 = 0) →
    (∀ x, f a b x = -3 * x^2 - 3 * x + 18) ∧
    (∀ c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) ↔ c ≤ -25/12) ∧
    (∃ y_max : ℝ, y_max = -3 ∧
      ∀ x > -1, (f a b x - 21) / (x + 1) ≤ y_max) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2808_280803


namespace NUMINAMATH_CALUDE_circus_ticket_sales_l2808_280817

theorem circus_ticket_sales (lower_price upper_price : ℕ) 
  (total_tickets total_revenue : ℕ) : 
  lower_price = 30 → 
  upper_price = 20 → 
  total_tickets = 80 → 
  total_revenue = 2100 → 
  ∃ (lower_seats upper_seats : ℕ), 
    lower_seats + upper_seats = total_tickets ∧ 
    lower_price * lower_seats + upper_price * upper_seats = total_revenue ∧ 
    lower_seats = 50 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_sales_l2808_280817


namespace NUMINAMATH_CALUDE_product_101_101_l2808_280811

theorem product_101_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_product_101_101_l2808_280811


namespace NUMINAMATH_CALUDE_mara_crayon_count_l2808_280876

theorem mara_crayon_count : ∀ (mara_crayons : ℕ),
  (mara_crayons : ℚ) * (1 / 10 : ℚ) + (50 : ℚ) * (1 / 5 : ℚ) = 14 →
  mara_crayons = 40 := by
  sorry

end NUMINAMATH_CALUDE_mara_crayon_count_l2808_280876


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2808_280886

theorem greatest_divisor_with_remainders : 
  Nat.gcd (690 - 10) (875 - 25) = 170 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2808_280886


namespace NUMINAMATH_CALUDE_folded_rectangle_long_side_l2808_280851

/-- A rectangle with a specific folding property -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  folded_congruent : Bool

/-- The theorem stating the relationship between short and long sides in the folded rectangle -/
theorem folded_rectangle_long_side 
  (rect : FoldedRectangle) 
  (h1 : rect.short_side = 8) 
  (h2 : rect.folded_congruent = true) : 
  rect.long_side = 12 := by
  sorry

#check folded_rectangle_long_side

end NUMINAMATH_CALUDE_folded_rectangle_long_side_l2808_280851


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2808_280839

/-- A continuous function satisfying the given functional equation -/
structure FunctionalEquation where
  f : ℝ → ℝ
  continuous : Continuous f
  equation : ∀ x y, f (x + y) = f x + f y + f x * f y

/-- The theorem stating the form of the function satisfying the equation -/
theorem functional_equation_solution (fe : FunctionalEquation) :
  ∃ a : ℝ, a ≥ 1 ∧ ∀ x, fe.f x = a^x - 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2808_280839


namespace NUMINAMATH_CALUDE_cubic_equation_solution_mean_l2808_280855

theorem cubic_equation_solution_mean :
  let f : ℝ → ℝ := λ x => x^3 + 5*x^2 - 14*x
  let solutions := {x : ℝ | f x = 0}
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, f x = 0) ∧ 
    (s.sum id) / s.card = -5/3 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_mean_l2808_280855


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_over_b_l2808_280840

-- Define the function f(x) = ax^2 + b
def f (a b x : ℝ) : ℝ := a * x^2 + b

-- Define the derivative of f
def f_derivative (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * a * x

theorem tangent_slope_implies_a_over_b (a b : ℝ) : 
  f a b 1 = 3 ∧ f_derivative a 1 = 2 → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_over_b_l2808_280840


namespace NUMINAMATH_CALUDE_bill_calculation_l2808_280802

def original_bill : ℝ := 500
def late_charge_rate : ℝ := 0.02

def final_bill : ℝ :=
  original_bill * (1 + late_charge_rate) * (1 + late_charge_rate) * (1 + late_charge_rate)

theorem bill_calculation :
  final_bill = 530.604 := by sorry

end NUMINAMATH_CALUDE_bill_calculation_l2808_280802


namespace NUMINAMATH_CALUDE_characterization_of_n_l2808_280869

-- Define the type of positive integers
def PositiveInt := { n : ℕ | n > 0 }

-- Define a function to get all positive divisors of a number
def positiveDivisors (n : PositiveInt) : List PositiveInt := sorry

-- Define a function to check if a list forms a geometric sequence
def isGeometricSequence (l : List ℝ) : Prop := sorry

-- Define the conditions for n
def satisfiesConditions (n : PositiveInt) : Prop :=
  let divisors := positiveDivisors n
  (divisors.length ≥ 4) ∧
  (isGeometricSequence (List.zipWith (λ a b => b - a) divisors (List.tail divisors)))

-- Define the form pᵃ where p is prime and a ≥ 3
def isPrimePower (n : PositiveInt) : Prop :=
  ∃ (p : ℕ) (a : ℕ), Prime p ∧ a ≥ 3 ∧ n = p^a

-- The main theorem
theorem characterization_of_n (n : PositiveInt) :
  satisfiesConditions n ↔ isPrimePower n := by sorry

end NUMINAMATH_CALUDE_characterization_of_n_l2808_280869


namespace NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l2808_280848

-- Define the pizza and pepperoni properties
def pizza_diameter : ℝ := 12
def pepperoni_across_diameter : ℕ := 6
def total_pepperoni : ℕ := 24

-- Theorem statement
theorem pepperoni_coverage_fraction :
  let pepperoni_diameter : ℝ := pizza_diameter / pepperoni_across_diameter
  let pepperoni_radius : ℝ := pepperoni_diameter / 2
  let pepperoni_area : ℝ := π * pepperoni_radius ^ 2
  let total_pepperoni_area : ℝ := pepperoni_area * total_pepperoni
  let pizza_radius : ℝ := pizza_diameter / 2
  let pizza_area : ℝ := π * pizza_radius ^ 2
  total_pepperoni_area / pizza_area = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_fraction_l2808_280848


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2808_280837

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ → ℝ), 
    (x^4 - 3*x^2 + 1 = (x^2 - x + 1) * q x + r x) ∧ 
    (∀ y, r y = -3*y^2 + y + 1) ∧
    (∀ z, z^2 - z + 1 = 0 → r z = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2808_280837


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2808_280857

theorem absolute_value_inequality (x : ℝ) :
  (abs (x + 2) + abs (x - 1) ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2808_280857


namespace NUMINAMATH_CALUDE_charlie_feather_count_l2808_280861

/-- The number of feathers Charlie already has -/
def feathers_already_has : ℕ := 387

/-- The number of feathers Charlie needs to collect -/
def feathers_to_collect : ℕ := 513

/-- The total number of feathers Charlie needs for his wings -/
def total_feathers_needed : ℕ := feathers_already_has + feathers_to_collect

theorem charlie_feather_count : total_feathers_needed = 900 := by
  sorry

end NUMINAMATH_CALUDE_charlie_feather_count_l2808_280861


namespace NUMINAMATH_CALUDE_right_triangle_conditions_l2808_280816

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- Define the conditions from the problem
def condition_A (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ t.a = 5*x ∧ t.b = 12*x ∧ t.c = 13*x

def condition_B (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ t.a = 2*x ∧ t.b = 3*x ∧ t.c = 5*x

def condition_C (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 9*k ∧ t.b = 40*k ∧ t.c = 41*k

def condition_D (t : Triangle) : Prop :=
  t.a = 3^2 ∧ t.b = 4^2 ∧ t.c = 5^2

-- Theorem statement
theorem right_triangle_conditions :
  (∀ t : Triangle, condition_A t → is_right_triangle t) ∧
  (∀ t : Triangle, condition_B t → is_right_triangle t) ∧
  (∀ t : Triangle, condition_C t → is_right_triangle t) ∧
  (∃ t : Triangle, condition_D t ∧ ¬is_right_triangle t) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_conditions_l2808_280816


namespace NUMINAMATH_CALUDE_tunnel_length_l2808_280820

/-- Given a train and a tunnel, calculate the length of the tunnel. -/
theorem tunnel_length
  (train_length : ℝ)
  (exit_time : ℝ)
  (train_speed : ℝ)
  (h1 : train_length = 2)
  (h2 : exit_time = 4)
  (h3 : train_speed = 120) :
  let distance_traveled := train_speed / 60 * exit_time
  let tunnel_length := distance_traveled - train_length
  tunnel_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l2808_280820


namespace NUMINAMATH_CALUDE_discount_rate_calculation_l2808_280824

def marked_price : ℝ := 240
def selling_price : ℝ := 120

theorem discount_rate_calculation : 
  (marked_price - selling_price) / marked_price * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_discount_rate_calculation_l2808_280824


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_product_l2808_280832

theorem pure_imaginary_complex_product (a : ℝ) : 
  (Complex.im ((1 + a * Complex.I) * (3 - Complex.I)) ≠ 0 ∧ 
   Complex.re ((1 + a * Complex.I) * (3 - Complex.I)) = 0) → 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_product_l2808_280832


namespace NUMINAMATH_CALUDE_unique_real_root_l2808_280830

theorem unique_real_root : 
  (∃ x : ℝ, x^2 + 3 = 0) = false ∧ 
  (∃ x : ℝ, x^3 + 3 = 0) = true ∧ 
  (∃ x : ℝ, |1 / (x^2 - 3)| = 0) = false ∧ 
  (∃ x : ℝ, |x| + 3 = 0) = false :=
by sorry

end NUMINAMATH_CALUDE_unique_real_root_l2808_280830


namespace NUMINAMATH_CALUDE_binomial_seven_one_l2808_280808

theorem binomial_seven_one : (7 : ℕ).choose 1 = 7 := by sorry

end NUMINAMATH_CALUDE_binomial_seven_one_l2808_280808


namespace NUMINAMATH_CALUDE_intersection_nonempty_range_union_equals_B_l2808_280887

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 1 = 2}

-- Theorem for part (1)
theorem intersection_nonempty_range (m : ℝ) : 
  (A ∩ B m).Nonempty → m = -Real.sqrt 3 :=
sorry

-- Theorem for part (2)
theorem union_equals_B (m : ℝ) : 
  A ∪ B m = B m → m = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_range_union_equals_B_l2808_280887


namespace NUMINAMATH_CALUDE_honey_work_days_l2808_280831

/-- Proves that Honey worked for 20 days given her daily earnings and total spent and saved amounts. -/
theorem honey_work_days (daily_earnings : ℕ) (total_spent : ℕ) (total_saved : ℕ) :
  daily_earnings = 80 →
  total_spent = 1360 →
  total_saved = 240 →
  (total_spent + total_saved) / daily_earnings = 20 :=
by sorry

end NUMINAMATH_CALUDE_honey_work_days_l2808_280831


namespace NUMINAMATH_CALUDE_equation_solution_l2808_280821

theorem equation_solution (x : ℝ) :
  x ≠ -4 →
  -x^2 = (4*x + 2) / (x + 4) →
  x = -2 ∨ x = -1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2808_280821


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l2808_280865

theorem tangent_line_to_exp_curve (x y : ℝ) :
  (∃ (m b : ℝ), y = m * x + b ∧ 
    (∀ (x₀ : ℝ), Real.exp x₀ = m * x₀ + b → x₀ = 1 ∨ x₀ = x) ∧
    0 = m * 1 + b) →
  Real.exp 2 * x - y - Real.exp 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_curve_l2808_280865


namespace NUMINAMATH_CALUDE_tessellation_theorem_l2808_280889

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  interiorAngle : ℝ

/-- Checks if two regular polygons can tessellate -/
def canTessellate (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n m : ℕ), n * p1.interiorAngle + m * p2.interiorAngle = 360

theorem tessellation_theorem :
  let triangle : RegularPolygon := ⟨3, 60⟩
  let square : RegularPolygon := ⟨4, 90⟩
  let hexagon : RegularPolygon := ⟨6, 120⟩
  let octagon : RegularPolygon := ⟨8, 135⟩
  
  (canTessellate triangle square ∧
   canTessellate triangle hexagon ∧
   canTessellate octagon square) ∧
  ¬(canTessellate hexagon square) :=
by sorry

end NUMINAMATH_CALUDE_tessellation_theorem_l2808_280889


namespace NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l2808_280863

/-- The minimum number of socks needed to ensure at least n pairs of the same color
    when randomly picking from a set of socks with m different colors. -/
def min_socks (n : ℕ) (m : ℕ) : ℕ :=
  m + 1 + 2 * (n - 1)

/-- Theorem: Given a set of socks with 4 different colors, 
    the minimum number of socks that must be randomly picked 
    to ensure at least 15 pairs of the same color is 33. -/
theorem min_socks_for_fifteen_pairs : min_socks 15 4 = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l2808_280863


namespace NUMINAMATH_CALUDE_star_operation_proof_l2808_280828

-- Define the ※ operation
def star (a b : ℕ) : ℚ :=
  (b : ℚ) / 2 * (2 * (a : ℚ) / 10 + ((b : ℚ) - 1) / 10)

-- State the theorem
theorem star_operation_proof (a : ℕ) :
  star 1 2 = (3 : ℚ) / 10 ∧
  star 2 3 = (9 : ℚ) / 10 ∧
  star 5 4 = (26 : ℚ) / 10 ∧
  star a 15 = (165 : ℚ) / 10 →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_operation_proof_l2808_280828


namespace NUMINAMATH_CALUDE_min_value_of_f_l2808_280825

/-- The quadratic function f(x) = (x-2)^2 - 3 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The minimum value of f(x) is -3 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2808_280825


namespace NUMINAMATH_CALUDE_circle_circumference_limit_l2808_280885

open Real

theorem circle_circumference_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * π * (C / n) - C| < ε :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_limit_l2808_280885


namespace NUMINAMATH_CALUDE_min_value_ab_l2808_280860

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3/a + 2/b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3/x + 2/y = 2 → a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l2808_280860


namespace NUMINAMATH_CALUDE_athletes_arrival_time_l2808_280884

/-- Proves that the number of hours new athletes arrived is 7, given the initial conditions and the final difference in the number of athletes. -/
theorem athletes_arrival_time (
  initial_athletes : ℕ)
  (leaving_rate : ℕ)
  (leaving_hours : ℕ)
  (arriving_rate : ℕ)
  (final_difference : ℕ)
  (h1 : initial_athletes = 300)
  (h2 : leaving_rate = 28)
  (h3 : leaving_hours = 4)
  (h4 : arriving_rate = 15)
  (h5 : final_difference = 7)
  : ∃ (x : ℕ), 
    initial_athletes - (leaving_rate * leaving_hours) + (arriving_rate * x) = 
    initial_athletes - final_difference ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_athletes_arrival_time_l2808_280884


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l2808_280882

/-- The number of times Terrell lifts the 20-pound weights -/
def original_lifts : ℕ := 12

/-- The weight of each dumbbell in the original set (in pounds) -/
def original_weight : ℕ := 20

/-- The weight of each dumbbell in the new set (in pounds) -/
def new_weight : ℕ := 10

/-- The number of dumbbells Terrell lifts each time -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted -/
def total_weight (weight : ℕ) (lifts : ℕ) : ℕ :=
  num_dumbbells * weight * lifts

/-- The number of times Terrell needs to lift the new weights to achieve the same total weight -/
def required_lifts : ℕ := total_weight original_weight original_lifts / (num_dumbbells * new_weight)

theorem terrell_weight_lifting :
  required_lifts = 24 ∧
  total_weight new_weight required_lifts = total_weight original_weight original_lifts :=
by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l2808_280882


namespace NUMINAMATH_CALUDE_equation_solution_l2808_280874

theorem equation_solution : 
  ∃ x : ℝ, (Real.sqrt (x^2 + 6*x + 10) + Real.sqrt (x^2 - 6*x + 10) = 8) ↔ 
  (x = (4 * Real.sqrt 42) / 7 ∨ x = -(4 * Real.sqrt 42) / 7) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2808_280874


namespace NUMINAMATH_CALUDE_remaining_score_is_40_l2808_280891

/-- Represents the score of a dodgeball player -/
structure PlayerScore where
  hitting : ℕ
  catching : ℕ
  eliminating : ℕ

/-- Calculates the total score for a player -/
def totalScore (score : PlayerScore) : ℕ :=
  2 * score.hitting + 5 * score.catching + 10 * score.eliminating

/-- Represents the scores of all players in the game -/
structure GameScores where
  paige : PlayerScore
  brian : PlayerScore
  karen : PlayerScore
  jennifer : PlayerScore
  michael : PlayerScore

/-- The main theorem to prove -/
theorem remaining_score_is_40 (game : GameScores) : 
  totalScore game.paige = 21 →
  totalScore game.brian = 20 →
  game.karen.eliminating = 0 →
  game.jennifer.eliminating = 0 →
  game.michael.eliminating = 0 →
  totalScore game.paige + totalScore game.brian + 
  totalScore game.karen + totalScore game.jennifer + totalScore game.michael = 81 →
  totalScore game.karen + totalScore game.jennifer + totalScore game.michael = 40 := by
  sorry

#check remaining_score_is_40

end NUMINAMATH_CALUDE_remaining_score_is_40_l2808_280891


namespace NUMINAMATH_CALUDE_acute_triangle_condition_l2808_280893

/-- 
Given a unit circle with diameter AB, where A(-1, 0) and B(1, 0),
and a point D(x, 0) on AB, prove that AD, BD, and CD form an acute triangle
if and only if x is in the open interval (2 - √5, √5 - 2),
where C is the point where DC ⊥ AB intersects the circle.
-/
theorem acute_triangle_condition (x : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((x + 1)^2)
  let BD := Real.sqrt ((1 - x)^2)
  let CD := Real.sqrt (1 - x^2)
  (AD^2 + BD^2 > CD^2 ∧ AD^2 + CD^2 > BD^2 ∧ BD^2 + CD^2 > AD^2) ↔ 
  (x > 2 - Real.sqrt 5 ∧ x < Real.sqrt 5 - 2) :=
by sorry


end NUMINAMATH_CALUDE_acute_triangle_condition_l2808_280893


namespace NUMINAMATH_CALUDE_no_solutions_cyclotomic_equation_l2808_280843

theorem no_solutions_cyclotomic_equation :
  ∀ (x y : ℕ), x > 1 → (x^7 - 1) / (x - 1) ≠ y^5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_cyclotomic_equation_l2808_280843


namespace NUMINAMATH_CALUDE_select_five_from_eight_l2808_280853

/-- The number of ways to select k items from n items without considering order -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: Selecting 5 books from 8 books without order consideration yields 56 ways -/
theorem select_five_from_eight : combination 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l2808_280853


namespace NUMINAMATH_CALUDE_soccer_league_games_l2808_280871

/-- The number of games played in a league with a given number of teams and games per pair of teams. -/
def games_played (n : ℕ) (g : ℕ) : ℕ := n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team, 
    the total number of games played is 180. -/
theorem soccer_league_games : games_played 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l2808_280871


namespace NUMINAMATH_CALUDE_sled_total_distance_l2808_280866

/-- The distance traveled by a sled in n seconds, given initial distance and acceleration -/
def sledDistance (initialDistance : ℕ) (acceleration : ℕ) (n : ℕ) : ℕ :=
  n * (2 * initialDistance + (n - 1) * acceleration) / 2

/-- Theorem stating the total distance traveled by the sled -/
theorem sled_total_distance :
  sledDistance 8 10 40 = 8120 := by
  sorry

end NUMINAMATH_CALUDE_sled_total_distance_l2808_280866


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_five_l2808_280801

theorem irrationality_of_sqrt_five :
  ¬ (∃ (q : ℚ), q * q = 5) ∧
  (∃ (a : ℚ), a * a = 4) ∧
  (∃ (b : ℚ), b * b = 9) ∧
  (∃ (c : ℚ), c * c = 16) :=
sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_five_l2808_280801


namespace NUMINAMATH_CALUDE_trig_sum_equals_negative_sqrt3_over_6_trig_fraction_sum_simplification_l2808_280810

-- Part I
theorem trig_sum_equals_negative_sqrt3_over_6 :
  Real.sin (5 * Real.pi / 3) + Real.cos (11 * Real.pi / 2) + Real.tan (-11 * Real.pi / 6) = -Real.sqrt 3 / 6 := by
  sorry

-- Part II
theorem trig_fraction_sum_simplification (θ : Real) 
  (h1 : Real.tan θ ≠ 0) (h2 : Real.tan θ ≠ 1) :
  (Real.sin θ / (1 - 1 / Real.tan θ)) + (Real.cos θ / (1 - Real.tan θ)) = Real.sin θ + Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_negative_sqrt3_over_6_trig_fraction_sum_simplification_l2808_280810


namespace NUMINAMATH_CALUDE_trajectory_equation_l2808_280847

/-- The trajectory of point M satisfying the distance ratio condition -/
theorem trajectory_equation (x y : ℝ) :
  (((x - 5)^2 + y^2).sqrt / |x - 9/5| = 5/3) →
  (x^2 / 9 - y^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2808_280847


namespace NUMINAMATH_CALUDE_mrs_hilt_bug_count_l2808_280897

/-- The number of bugs Mrs. Hilt saw -/
def num_bugs : ℕ := 3

/-- The number of flowers each bug eats -/
def flowers_per_bug : ℕ := 2

/-- The total number of flowers eaten -/
def total_flowers : ℕ := 6

/-- Theorem: The number of bugs is correct given the conditions -/
theorem mrs_hilt_bug_count : 
  num_bugs * flowers_per_bug = total_flowers :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_bug_count_l2808_280897


namespace NUMINAMATH_CALUDE_bc_length_l2808_280856

-- Define the triangle
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 0) -- Exact coordinates don't matter for this proof
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (0, 0)

-- Define the given lengths
def AD : ℝ := 47
def CD : ℝ := 25
def AC : ℝ := 24

-- Define the theorem
theorem bc_length :
  Triangle A B C →
  Triangle A B D →
  D.1 < C.1 →
  D.2 = B.2 →
  C.2 = B.2 →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = AD^2 →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = CD^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 20.16^2 := by
  sorry

end NUMINAMATH_CALUDE_bc_length_l2808_280856


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2808_280849

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 3) : 2 * π * r^2 + π * r^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2808_280849


namespace NUMINAMATH_CALUDE_race_result_l2808_280896

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- The race setup -/
def Race (sasha lesha kolya : Runner) : Prop :=
  sasha.speed > 0 ∧ lesha.speed > 0 ∧ kolya.speed > 0 ∧
  sasha.speed ≠ lesha.speed ∧ sasha.speed ≠ kolya.speed ∧ lesha.speed ≠ kolya.speed ∧
  sasha.distance = 100 ∧
  lesha.distance = 90 ∧
  kolya.distance = 81

theorem race_result (sasha lesha kolya : Runner) 
  (h : Race sasha lesha kolya) : 
  sasha.distance - kolya.distance = 19 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l2808_280896


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2808_280845

theorem weekend_rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.3)
  (h2 : p_sunday = 0.6)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l2808_280845
