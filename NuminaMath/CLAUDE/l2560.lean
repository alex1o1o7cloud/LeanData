import Mathlib

namespace NUMINAMATH_CALUDE_line_equation_proof_l2560_256081

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y + 3 = 0
def line2 (x y : ℝ) : Prop := 4*x + 3*y + 1 = 0
def line3 (x y : ℝ) : Prop := 2*x - 3*y + 4 = 0
def line_result (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_equation_proof :
  ∃ x y : ℝ, 
    intersection_point x y ∧ 
    line_result x y ∧
    perpendicular (3/2) (-2/3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2560_256081


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2560_256025

theorem constant_term_expansion (x : ℝ) : ∃ c : ℝ, c = 24 ∧ 
  (∃ f : ℝ → ℝ, (λ x => (2*x + 1/x)^4) = f + λ _ => c) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2560_256025


namespace NUMINAMATH_CALUDE_min_value_ab_l2560_256019

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3/a + 2/b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3/x + 2/y = 2 → a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l2560_256019


namespace NUMINAMATH_CALUDE_rectangle_area_l2560_256074

/-- Given a rectangle PQRS with specified coordinates, prove its area is 40400 -/
theorem rectangle_area (y : ℤ) : 
  let P : ℝ × ℝ := (10, -30)
  let Q : ℝ × ℝ := (2010, 170)
  let S : ℝ × ℝ := (12, y)
  let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let PS := Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2)
  PQ * PS = 40400 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2560_256074


namespace NUMINAMATH_CALUDE_tower_surface_area_l2560_256082

-- Define the volumes of the cubes
def cube_volumes : List ℝ := [1, 8, 27, 64, 125, 216, 343]

-- Function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Function to calculate the surface area of a cube given its side length
def surface_area (side : ℝ) : ℝ := 6 * side^2

-- Function to calculate the exposed surface area of a cube in the tower
def exposed_surface_area (side : ℝ) (is_bottom : Bool) : ℝ :=
  if is_bottom then surface_area side else surface_area side - side^2

-- Theorem statement
theorem tower_surface_area :
  let sides := cube_volumes.map side_length
  let exposed_areas := List.zipWith exposed_surface_area sides [true, false, false, false, false, false, false]
  exposed_areas.sum = 701 := by sorry

end NUMINAMATH_CALUDE_tower_surface_area_l2560_256082


namespace NUMINAMATH_CALUDE_quadratic_ratio_l2560_256094

/-- The quadratic function f(x) = x^2 + 1600x + 1607 -/
def f (x : ℝ) : ℝ := x^2 + 1600*x + 1607

/-- The constant b in the completed square form (x+b)^2 + c -/
def b : ℝ := 800

/-- The constant c in the completed square form (x+b)^2 + c -/
def c : ℝ := -638393

/-- Theorem stating that c/b equals -797.99125 for the given quadratic -/
theorem quadratic_ratio : c / b = -797.99125 := by sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l2560_256094


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2560_256010

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2560_256010


namespace NUMINAMATH_CALUDE_special_cone_vertex_angle_l2560_256091

/-- A right circular cone with three pairwise perpendicular generatrices -/
structure SpecialCone where
  /-- The angle at the vertex of the axial section -/
  vertex_angle : ℝ
  /-- The condition that three generatrices are pairwise perpendicular -/
  perpendicular_generatrices : Prop

/-- Theorem: The angle at the vertex of the axial section of a special cone is 2 * arcsin(√6 / 3) -/
theorem special_cone_vertex_angle (cone : SpecialCone) :
  cone.perpendicular_generatrices →
  cone.vertex_angle = 2 * Real.arcsin (Real.sqrt 6 / 3) := by
  sorry

end NUMINAMATH_CALUDE_special_cone_vertex_angle_l2560_256091


namespace NUMINAMATH_CALUDE_floor_of_4_7_l2560_256021

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l2560_256021


namespace NUMINAMATH_CALUDE_junior_score_l2560_256057

theorem junior_score (n : ℕ) (h : n > 0) : 
  let junior_count : ℝ := 0.2 * n
  let senior_count : ℝ := 0.8 * n
  let total_score : ℝ := 85 * n
  let senior_score : ℝ := 82 * senior_count
  let junior_total_score : ℝ := total_score - senior_score
  junior_total_score / junior_count = 97 := by sorry

end NUMINAMATH_CALUDE_junior_score_l2560_256057


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l2560_256031

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b is in the set (-∞, -10] ∪ [10, ∞). -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_real_root_condition_l2560_256031


namespace NUMINAMATH_CALUDE_seven_rows_five_seats_l2560_256067

-- Define a movie ticket as a pair of natural numbers
def MovieTicket : Type := ℕ × ℕ

-- Define a function to create a movie ticket representation
def createTicket (rows : ℕ) (seats : ℕ) : MovieTicket := (rows, seats)

-- Theorem statement
theorem seven_rows_five_seats :
  createTicket 7 5 = (7, 5) := by sorry

end NUMINAMATH_CALUDE_seven_rows_five_seats_l2560_256067


namespace NUMINAMATH_CALUDE_problem_solution_l2560_256098

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : y / 100 * y + 6 = 10) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2560_256098


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2560_256088

theorem value_of_a_minus_b (a b c : ℚ) 
  (eq1 : 2011 * a + 2015 * b + c = 2021)
  (eq2 : 2013 * a + 2017 * b + c = 2023)
  (eq3 : 2012 * a + 2016 * b + 2 * c = 2026) :
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2560_256088


namespace NUMINAMATH_CALUDE_flash_overtakes_ace_l2560_256036

/-- The distance Flash needs to jog to overtake Ace -/
def overtake_distance (v y t : ℝ) : ℝ :=
  2 * (y + 60 * v * t)

/-- Theorem stating the distance Flash needs to jog to overtake Ace -/
theorem flash_overtakes_ace (v y t : ℝ) (hv : v > 0) (hy : y ≥ 0) (ht : t ≥ 0) :
  ∃ d : ℝ, d = overtake_distance v y t ∧ d > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_flash_overtakes_ace_l2560_256036


namespace NUMINAMATH_CALUDE_correct_calculation_l2560_256039

theorem correct_calculation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2560_256039


namespace NUMINAMATH_CALUDE_icosikaipentagon_diagonals_l2560_256047

/-- The number of diagonals that can be drawn from a single vertex of an n-sided polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

theorem icosikaipentagon_diagonals :
  diagonals_from_vertex 25 = 22 :=
by sorry

end NUMINAMATH_CALUDE_icosikaipentagon_diagonals_l2560_256047


namespace NUMINAMATH_CALUDE_negative_one_squared_and_one_are_opposite_l2560_256000

-- Define opposite numbers
def are_opposite (a b : ℤ) : Prop := a + b = 0

-- Theorem statement
theorem negative_one_squared_and_one_are_opposite : 
  are_opposite (-(1^2)) 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_squared_and_one_are_opposite_l2560_256000


namespace NUMINAMATH_CALUDE_sequence_sum_l2560_256087

theorem sequence_sum (seq : Fin 10 → ℝ) 
  (h1 : seq 2 = 5)
  (h2 : ∀ i : Fin 8, seq i + seq (i + 1) + seq (i + 2) = 25) :
  seq 0 + seq 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2560_256087


namespace NUMINAMATH_CALUDE_train_length_l2560_256027

/-- The length of a train crossing a bridge -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 200 →
  crossing_time = 60 →
  train_speed = 5 →
  bridge_length + (train_speed * crossing_time - bridge_length) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2560_256027


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2560_256015

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5 * x - 4 < 3 - 2 * x := by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2560_256015


namespace NUMINAMATH_CALUDE_dog_weight_ratio_l2560_256013

/-- Given the weights of two dogs, prove the ratio of their weights -/
theorem dog_weight_ratio 
  (evan_dog_weight : ℕ) 
  (total_weight : ℕ) 
  (h1 : evan_dog_weight = 63)
  (h2 : total_weight = 72)
  (h3 : ∃ k : ℕ, k * (total_weight - evan_dog_weight) = evan_dog_weight) :
  evan_dog_weight / (total_weight - evan_dog_weight) = 7 := by
sorry

end NUMINAMATH_CALUDE_dog_weight_ratio_l2560_256013


namespace NUMINAMATH_CALUDE_max_hawthorns_l2560_256048

theorem max_hawthorns (x : ℕ) : 
  x > 100 ∧
  x % 3 = 1 ∧
  x % 4 = 2 ∧
  x % 5 = 3 ∧
  x % 6 = 4 →
  x ≤ 178 ∧ 
  ∃ y : ℕ, y > 100 ∧ 
    y % 3 = 1 ∧ 
    y % 4 = 2 ∧ 
    y % 5 = 3 ∧ 
    y % 6 = 4 ∧ 
    y = 178 :=
by sorry

end NUMINAMATH_CALUDE_max_hawthorns_l2560_256048


namespace NUMINAMATH_CALUDE_candy_distribution_l2560_256037

/-- Given 27.5 candy bars divided among 8.3 people, each person receives approximately 3.313 candy bars -/
theorem candy_distribution (total_candy : ℝ) (num_people : ℝ) (candy_per_person : ℝ) 
  (h1 : total_candy = 27.5)
  (h2 : num_people = 8.3)
  (h3 : candy_per_person = total_candy / num_people) :
  ∃ ε > 0, |candy_per_person - 3.313| < ε :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2560_256037


namespace NUMINAMATH_CALUDE_sum_angles_S_and_R_l2560_256005

-- Define the circle and points
variable (circle : Type) (E F R G H : circle)

-- Define the measure of an arc
variable (arc_measure : circle → circle → ℝ)

-- Define the measure of an angle
variable (angle_measure : circle → ℝ)

-- State the theorem
theorem sum_angles_S_and_R (h1 : arc_measure F R = 60)
                           (h2 : arc_measure R G = 48) :
  angle_measure S + angle_measure R = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_angles_S_and_R_l2560_256005


namespace NUMINAMATH_CALUDE_remainder_calculation_l2560_256073

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_calculation : rem (-1/3) (4/7) = 5/21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_calculation_l2560_256073


namespace NUMINAMATH_CALUDE_back_sides_average_l2560_256063

def is_prime_or_one (n : ℕ) : Prop := n = 1 ∨ Nat.Prime n

theorem back_sides_average (a b c : ℕ) : 
  is_prime_or_one a ∧ is_prime_or_one b ∧ is_prime_or_one c →
  28 + a = 40 + b ∧ 40 + b = 49 + c →
  (a + b + c) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_back_sides_average_l2560_256063


namespace NUMINAMATH_CALUDE_not_necessarily_true_squared_l2560_256049

theorem not_necessarily_true_squared (x y : ℝ) (h : x > y) : 
  ¬ (∀ x y : ℝ, x > y → x^2 > y^2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_true_squared_l2560_256049


namespace NUMINAMATH_CALUDE_subset_sum_partition_l2560_256022

theorem subset_sum_partition (n : ℕ) (S : Finset ℝ) (h_pos : ∀ x ∈ S, 0 < x) (h_card : S.card = n) :
  ∃ (P : Finset (Finset ℝ)), 
    P.card = n ∧ 
    (∀ X ∈ P, ∃ (min max : ℝ), 
      (∀ y ∈ X, min ≤ y ∧ y ≤ max) ∧ 
      max < 2 * min) ∧
    (∀ A : Finset ℝ, A.Nonempty → A ⊆ S → ∃ X ∈ P, (A.sum id) ∈ X) :=
sorry

end NUMINAMATH_CALUDE_subset_sum_partition_l2560_256022


namespace NUMINAMATH_CALUDE_pencils_per_row_l2560_256097

theorem pencils_per_row (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) 
  (h1 : packs = 28) 
  (h2 : pencils_per_pack = 24) 
  (h3 : rows = 42) :
  (packs * pencils_per_pack) / rows = 16 := by
  sorry

#check pencils_per_row

end NUMINAMATH_CALUDE_pencils_per_row_l2560_256097


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2560_256024

def box_values : List ℕ := [10, 50, 100, 500, 1000, 5000, 50000, 75000, 200000, 400000, 500000, 1000000]

def total_boxes : ℕ := 16

def high_value_boxes : ℕ := (box_values.filter (λ x => x ≥ 500000)).length

theorem deal_or_no_deal_probability (boxes_to_eliminate : ℕ) :
  boxes_to_eliminate = 10 ↔ 
  (high_value_boxes : ℚ) / (total_boxes - boxes_to_eliminate : ℚ) ≥ 1/2 ∧
  ∀ n : ℕ, n < boxes_to_eliminate → 
    (high_value_boxes : ℚ) / (total_boxes - n : ℚ) < 1/2 :=
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l2560_256024


namespace NUMINAMATH_CALUDE_daughters_to_sons_ratio_l2560_256017

theorem daughters_to_sons_ratio (total_children : ℕ) (sons : ℕ) (daughters : ℕ) : 
  total_children = 21 → sons = 3 → daughters = total_children - sons → 
  (daughters : ℚ) / (sons : ℚ) = 6 / 1 := by
  sorry

end NUMINAMATH_CALUDE_daughters_to_sons_ratio_l2560_256017


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l2560_256086

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l2560_256086


namespace NUMINAMATH_CALUDE_largest_p_value_l2560_256016

theorem largest_p_value (m n p : ℕ) : 
  m ≥ 3 → n ≥ 3 → p ≥ 3 →
  (1 : ℚ) / m + (1 : ℚ) / n + (1 : ℚ) / p = (1 : ℚ) / 2 →
  p ≤ 42 :=
sorry

end NUMINAMATH_CALUDE_largest_p_value_l2560_256016


namespace NUMINAMATH_CALUDE_bottle_cap_count_l2560_256064

theorem bottle_cap_count : 
  ∀ (cost_per_cap total_cost num_caps : ℕ),
  cost_per_cap = 2 →
  total_cost = 12 →
  total_cost = cost_per_cap * num_caps →
  num_caps = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_count_l2560_256064


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2560_256053

/-- An arithmetic sequence with a_6 = 1 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ a1 d : ℚ, ∀ n : ℕ, a n = a1 + (n - 1) * d) ∧ a 6 = 1

/-- For any arithmetic sequence with a_6 = 1, a_2 + a_10 = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  a 2 + a 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2560_256053


namespace NUMINAMATH_CALUDE_cube_split_with_31_l2560_256080

/-- For a natural number m > 1, if 31 is one of the odd numbers in the sum that equals m^3, then m = 6. -/
theorem cube_split_with_31 (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ) (l : List ℕ), 
    (∀ n ∈ l, Odd n) ∧ 
    (List.sum l = m^3) ∧
    (31 ∈ l) ∧
    (List.length l = m)) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_split_with_31_l2560_256080


namespace NUMINAMATH_CALUDE_distinct_integer_quadruple_l2560_256065

theorem distinct_integer_quadruple : 
  ∀ a b c d : ℕ+, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a + b = c * d →
    a * b = c + d →
    ((a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
     (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
     (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
     (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
     (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_integer_quadruple_l2560_256065


namespace NUMINAMATH_CALUDE_min_abs_sum_l2560_256096

theorem min_abs_sum (x : ℝ) : 
  ∃ (l : ℝ), l = 45 ∧ ∀ y : ℝ, |y - 2| + |y - 47| ≥ l :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l2560_256096


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_positive_m_for_unique_solution_l2560_256006

theorem unique_quadratic_solution (m : ℝ) :
  (∃! x : ℝ, 16 * x^2 + m * x + 4 = 0) ↔ m = 16 ∨ m = -16 :=
by sorry

theorem positive_m_for_unique_solution :
  ∃ m : ℝ, m > 0 ∧ (∃! x : ℝ, 16 * x^2 + m * x + 4 = 0) ∧ m = 16 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_positive_m_for_unique_solution_l2560_256006


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2560_256043

theorem problem_1 (m : ℤ) (h : m = -3) : 4 * (m + 1)^2 - (2*m + 5) * (2*m - 5) = 5 := by sorry

theorem problem_2 (x : ℚ) (h : x = 2) : (x^2 - 1) / (x^2 + 2*x) / ((x - 1) / x) = 3/4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2560_256043


namespace NUMINAMATH_CALUDE_quadratic_real_roots_when_ac_negative_l2560_256066

theorem quadratic_real_roots_when_ac_negative 
  (a b c : ℝ) (h : a * c < 0) : 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_when_ac_negative_l2560_256066


namespace NUMINAMATH_CALUDE_magic_square_x_value_l2560_256095

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℤ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The theorem stating the value of x in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a = x)
  (h2 : ms.b = 19)
  (h3 : ms.c = 96)
  (h4 : ms.d = 1) :
  x = 200 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l2560_256095


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l2560_256042

/-- Represents the rowing scenario in a river with current --/
structure RowingScenario where
  stream_rate : ℝ
  rowing_speed : ℝ
  time_ratio : ℝ

/-- Checks if the rowing scenario satisfies the given conditions --/
def is_valid_scenario (s : RowingScenario) : Prop :=
  s.stream_rate = 18 ∧ 
  s.time_ratio = 3 ∧
  (1 / (s.rowing_speed - s.stream_rate)) = s.time_ratio * (1 / (s.rowing_speed + s.stream_rate))

/-- Theorem stating that the man's rowing speed in still water is 36 kmph --/
theorem mans_rowing_speed (s : RowingScenario) : 
  is_valid_scenario s → s.rowing_speed = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_mans_rowing_speed_l2560_256042


namespace NUMINAMATH_CALUDE_probability_of_sunflower_seed_l2560_256029

def sunflower_seeds : ℕ := 2
def green_bean_seeds : ℕ := 3
def pumpkin_seeds : ℕ := 4

def total_seeds : ℕ := sunflower_seeds + green_bean_seeds + pumpkin_seeds

theorem probability_of_sunflower_seed :
  (sunflower_seeds : ℚ) / total_seeds = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_sunflower_seed_l2560_256029


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2560_256008

theorem greatest_divisor_with_remainders : 
  Nat.gcd (690 - 10) (875 - 25) = 170 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2560_256008


namespace NUMINAMATH_CALUDE_sample_is_weights_l2560_256060

/-- Represents a student in the survey -/
structure Student where
  weight : ℝ

/-- Represents the survey conducted by the city -/
structure Survey where
  students : Finset Student
  grade : Nat

/-- Definition of a sample in this context -/
def Sample (survey : Survey) : Set ℝ :=
  {w | ∃ s ∈ survey.students, w = s.weight}

/-- The theorem stating that the sample is the weight of 100 students -/
theorem sample_is_weights (survey : Survey) 
    (h1 : survey.grade = 9) 
    (h2 : survey.students.card = 100) : 
  Sample survey = {w | ∃ s ∈ survey.students, w = s.weight} := by
  sorry

end NUMINAMATH_CALUDE_sample_is_weights_l2560_256060


namespace NUMINAMATH_CALUDE_acid_mixture_concentration_l2560_256034

/-- Calculates the final acid concentration when replacing part of a solution with another -/
def finalAcidConcentration (initialConcentration replacementConcentration : ℚ) 
  (replacementFraction : ℚ) : ℚ :=
  (1 - replacementFraction) * initialConcentration + 
  replacementFraction * replacementConcentration

/-- Proves that replacing half of a 50% acid solution with a 30% acid solution results in a 40% solution -/
theorem acid_mixture_concentration : 
  finalAcidConcentration (1/2) (3/10) (1/2) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_concentration_l2560_256034


namespace NUMINAMATH_CALUDE_ten_digit_number_divisibility_l2560_256056

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem ten_digit_number_divisibility (a b : ℕ) :
  a < 10 → b < 10 →
  is_divisible_by_99 (2016 * 10000 + a * 1000 + b * 100 + 2017) →
  a + b = 8 := by sorry

end NUMINAMATH_CALUDE_ten_digit_number_divisibility_l2560_256056


namespace NUMINAMATH_CALUDE_zoo_meat_amount_l2560_256090

/-- The amount of meat (in kg) that lasts for a given number of days for a lion and a tiger -/
def meatAmount (lionConsumption tigerConsumption daysLasting : ℕ) : ℕ :=
  (lionConsumption + tigerConsumption) * daysLasting

theorem zoo_meat_amount :
  meatAmount 25 20 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_zoo_meat_amount_l2560_256090


namespace NUMINAMATH_CALUDE_percentage_relationship_l2560_256004

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 28.57142857142857 / 100)) :
  y = x * (1 + 28.57142857142857 / 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2560_256004


namespace NUMINAMATH_CALUDE_sin_value_given_tan_and_range_l2560_256030

theorem sin_value_given_tan_and_range (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) 
  (h2 : Real.tan α = Real.sqrt 2) : 
  Real.sin α = -(Real.sqrt 6 / 3) := by
sorry

end NUMINAMATH_CALUDE_sin_value_given_tan_and_range_l2560_256030


namespace NUMINAMATH_CALUDE_arithmetic_operations_l2560_256014

theorem arithmetic_operations : 
  (24 - (-16) + (-25) - 32 = -17) ∧
  ((-1/2) * 2 / 2 * (-1/2) = 1/4) ∧
  (-2^2 * 5 - (-2)^3 * (1/8) + 1 = -18) ∧
  ((-1/4 - 5/6 + 8/9) / (-1/6)^2 + (-2)^2 * (-6) = -31) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l2560_256014


namespace NUMINAMATH_CALUDE_geometric_sequence_bounded_l2560_256068

theorem geometric_sequence_bounded (n k : ℕ) (a : ℕ → ℝ) : 
  n > 0 → k > 0 → 
  (∀ i ∈ Finset.range (k+1), n^k ≤ a i ∧ a i ≤ (n+1)^k) →
  (∀ i ∈ Finset.range k, ∃ q : ℝ, a (i+1) = a i * q) →
  (∀ i ∈ Finset.range (k+1), a i = n^k * ((n+1)/n)^i ∨ a i = (n+1)^k * (n/(n+1))^i) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_bounded_l2560_256068


namespace NUMINAMATH_CALUDE_garden_division_theorem_l2560_256011

/-- Represents a rectangular garden -/
structure Garden where
  width : ℕ
  height : ℕ
  trees : ℕ

/-- Represents a division of the garden -/
structure Division where
  parts : ℕ
  matches_used : ℕ
  trees_per_part : ℕ

/-- Checks if a division is valid for a given garden -/
def is_valid_division (g : Garden) (d : Division) : Prop :=
  d.parts = 4 ∧
  d.matches_used = 12 ∧
  d.trees_per_part * d.parts = g.trees ∧
  d.trees_per_part = 3

theorem garden_division_theorem (g : Garden) 
  (h1 : g.width = 4)
  (h2 : g.height = 3)
  (h3 : g.trees = 12) :
  ∃ d : Division, is_valid_division g d :=
sorry

end NUMINAMATH_CALUDE_garden_division_theorem_l2560_256011


namespace NUMINAMATH_CALUDE_volume_removed_percentage_l2560_256051

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ :=
  side ^ 3

/-- Theorem: The percentage of volume removed from a box with dimensions 20x15x10,
    by removing a 4cm cube from each of its 8 corners, is equal to (512/3000) * 100% -/
theorem volume_removed_percentage :
  let originalBox : BoxDimensions := ⟨20, 15, 10⟩
  let removedCubeSide : ℝ := 4
  let numCorners : ℕ := 8
  let originalVolume := boxVolume originalBox
  let removedVolume := numCorners * (cubeVolume removedCubeSide)
  (removedVolume / originalVolume) * 100 = (512 / 3000) * 100 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_percentage_l2560_256051


namespace NUMINAMATH_CALUDE_athletes_arrival_time_l2560_256028

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

end NUMINAMATH_CALUDE_athletes_arrival_time_l2560_256028


namespace NUMINAMATH_CALUDE_unique_single_digit_square_l2560_256044

theorem unique_single_digit_square (A : ℕ) : A < 10 ∧ (10 * A + A) * (10 * A + A) = 5929 ↔ A = 7 := by sorry

end NUMINAMATH_CALUDE_unique_single_digit_square_l2560_256044


namespace NUMINAMATH_CALUDE_negative_division_equals_nine_l2560_256032

theorem negative_division_equals_nine : (-81) / (-9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_equals_nine_l2560_256032


namespace NUMINAMATH_CALUDE_circular_track_length_l2560_256072

-- Define the track length
def track_length : ℝ := 350

-- Define the constants given in the problem
def first_meeting_distance : ℝ := 80
def second_meeting_distance : ℝ := 140

-- Theorem statement
theorem circular_track_length :
  ∀ (brenda_speed sally_speed : ℝ),
  brenda_speed > 0 ∧ sally_speed > 0 →
  ∃ (t₁ t₂ : ℝ),
  t₁ > 0 ∧ t₂ > 0 ∧
  brenda_speed * t₁ = first_meeting_distance ∧
  sally_speed * t₁ = track_length / 2 - first_meeting_distance ∧
  brenda_speed * (t₁ + t₂) = track_length / 2 + first_meeting_distance ∧
  sally_speed * (t₁ + t₂) = track_length / 2 + second_meeting_distance →
  track_length = 350 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_circular_track_length_l2560_256072


namespace NUMINAMATH_CALUDE_segment_ratio_l2560_256012

/-- Given two line segments a and b, where a is 2 meters and b is 40 centimeters,
    prove that the ratio of a to b is 5:1. -/
theorem segment_ratio (a b : ℝ) : a = 2 → b = 40 / 100 → a / b = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l2560_256012


namespace NUMINAMATH_CALUDE_max_sum_product_sqrt_l2560_256046

theorem max_sum_product_sqrt (x₁ x₂ x₃ x₄ : ℝ) 
  (non_neg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0) 
  (sum_one : x₁ + x₂ + x₃ + x₄ = 1) :
  (x₁ + x₂) * Real.sqrt (x₁ * x₂) +
  (x₁ + x₃) * Real.sqrt (x₁ * x₃) +
  (x₁ + x₄) * Real.sqrt (x₁ * x₄) +
  (x₂ + x₃) * Real.sqrt (x₂ * x₃) +
  (x₂ + x₄) * Real.sqrt (x₂ * x₄) +
  (x₃ + x₄) * Real.sqrt (x₃ * x₄) ≤ 3/4 ∧
  (x₁ = 1/4 ∧ x₂ = 1/4 ∧ x₃ = 1/4 ∧ x₄ = 1/4 →
    (x₁ + x₂) * Real.sqrt (x₁ * x₂) +
    (x₁ + x₃) * Real.sqrt (x₁ * x₃) +
    (x₁ + x₄) * Real.sqrt (x₁ * x₄) +
    (x₂ + x₃) * Real.sqrt (x₂ * x₃) +
    (x₂ + x₄) * Real.sqrt (x₂ * x₄) +
    (x₃ + x₄) * Real.sqrt (x₃ * x₄) = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_product_sqrt_l2560_256046


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2560_256038

/-- Given three points on a line, prove the value of k --/
theorem collinear_points_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 8 = m * 2 + b ∧ k = m * 10 + b ∧ 2 = m * 16 + b) → k = 32/7 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2560_256038


namespace NUMINAMATH_CALUDE_brothers_age_sum_l2560_256093

theorem brothers_age_sum : 
  ∀ (older_age younger_age : ℕ),
  younger_age = 27 →
  younger_age = older_age / 3 + 10 →
  older_age + younger_age = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_age_sum_l2560_256093


namespace NUMINAMATH_CALUDE_expedition_duration_proof_l2560_256023

theorem expedition_duration_proof (first_expedition : ℕ) 
  (h1 : first_expedition = 3)
  (second_expedition : ℕ) 
  (h2 : second_expedition = first_expedition + 2)
  (third_expedition : ℕ) 
  (h3 : third_expedition = 2 * second_expedition) : 
  (first_expedition + second_expedition + third_expedition) * 7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_expedition_duration_proof_l2560_256023


namespace NUMINAMATH_CALUDE_playground_area_is_4200_l2560_256062

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The landscape satisfies the given conditions -/
def is_valid_landscape (l : Landscape) : Prop :=
  l.breadth = 6 * l.length ∧
  l.breadth = 420 ∧
  l.playground_area = (1 / 7) * (l.length * l.breadth)

theorem playground_area_is_4200 (l : Landscape) (h : is_valid_landscape l) :
  l.playground_area = 4200 := by
  sorry

#check playground_area_is_4200

end NUMINAMATH_CALUDE_playground_area_is_4200_l2560_256062


namespace NUMINAMATH_CALUDE_people_per_bus_l2560_256050

/-- Given a field trip with vans and buses, calculate the number of people per bus -/
theorem people_per_bus 
  (total_people : ℕ) 
  (num_vans : ℕ) 
  (people_per_van : ℕ) 
  (num_buses : ℕ) 
  (h1 : total_people = 342)
  (h2 : num_vans = 9)
  (h3 : people_per_van = 8)
  (h4 : num_buses = 10)
  : (total_people - num_vans * people_per_van) / num_buses = 27 := by
  sorry

end NUMINAMATH_CALUDE_people_per_bus_l2560_256050


namespace NUMINAMATH_CALUDE_equation_solution_l2560_256099

theorem equation_solution (x : ℝ) : 
  (x + 1)^5 + (x + 1)^4 * (x - 1) + (x + 1)^3 * (x - 1)^2 + 
  (x + 1)^2 * (x - 1)^3 + (x + 1) * (x - 1)^4 + (x - 1)^5 = 0 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2560_256099


namespace NUMINAMATH_CALUDE_prime_square_sum_l2560_256045

theorem prime_square_sum (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
  (∃ (n : ℕ), p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_l2560_256045


namespace NUMINAMATH_CALUDE_lower_limit_proof_l2560_256077

theorem lower_limit_proof (x : ℤ) (y : ℝ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : y < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + 2 < 4)
  (h6 : x = 1) :
  y < 1 := by
sorry

end NUMINAMATH_CALUDE_lower_limit_proof_l2560_256077


namespace NUMINAMATH_CALUDE_factor_x8_minus_81_l2560_256059

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_x8_minus_81_l2560_256059


namespace NUMINAMATH_CALUDE_ellipse_equation_l2560_256001

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > 0, b > 0),
    if the line 2x + y - 2 = 0 passes through its upper vertex and right focus,
    then the equation of the ellipse is x²/5 + y²/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ 2*x + y = 2 ∧
   ((x = a ∧ y = 0) ∨ (x = 0 ∧ y = b))) →
  a^2 = 5 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2560_256001


namespace NUMINAMATH_CALUDE_range_of_f_l2560_256069

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2560_256069


namespace NUMINAMATH_CALUDE_undetermined_disjunction_l2560_256092

theorem undetermined_disjunction (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), (¬p ∧ ¬(p ∧ q)) → (p ∨ q) := by
sorry

end NUMINAMATH_CALUDE_undetermined_disjunction_l2560_256092


namespace NUMINAMATH_CALUDE_sequential_search_comparisons_l2560_256026

/-- Represents a sequential search on an unordered array. -/
structure SequentialSearch where
  array_size : Nat
  element_not_present : Bool
  unordered : Bool

/-- The number of comparisons needed for a sequential search. -/
def comparisons_needed (search : SequentialSearch) : Nat :=
  search.array_size

/-- Theorem: The number of comparisons for a sequential search on an unordered array
    of 100 elements, where the element is not present, is 100. -/
theorem sequential_search_comparisons :
  ∀ (search : SequentialSearch),
    search.array_size = 100 →
    search.element_not_present = true →
    search.unordered = true →
    comparisons_needed search = 100 := by
  sorry

end NUMINAMATH_CALUDE_sequential_search_comparisons_l2560_256026


namespace NUMINAMATH_CALUDE_divisibility_relation_l2560_256018

theorem divisibility_relation (p q r s : ℤ) 
  (h_s : s % 5 ≠ 0)
  (h_a : ∃ a : ℤ, (p * a^3 + q * a^2 + r * a + s) % 5 = 0) :
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2560_256018


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2560_256071

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ (f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2560_256071


namespace NUMINAMATH_CALUDE_alcohol_concentration_second_vessel_l2560_256085

/-- Proves that the initial concentration of alcohol in the second vessel is 60% --/
theorem alcohol_concentration_second_vessel :
  let vessel1_capacity : ℝ := 2
  let vessel1_alcohol_percentage : ℝ := 40
  let vessel2_capacity : ℝ := 6
  let total_liquid : ℝ := 8
  let final_vessel_capacity : ℝ := 10
  let final_mixture_percentage : ℝ := 44
  let vessel2_alcohol_percentage : ℝ := 
    (final_mixture_percentage * final_vessel_capacity - vessel1_alcohol_percentage * vessel1_capacity) / vessel2_capacity
  vessel2_alcohol_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_alcohol_concentration_second_vessel_l2560_256085


namespace NUMINAMATH_CALUDE_infinite_representable_theorem_l2560_256084

-- Define an increasing sequence of positive integers
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- Define the property we want to prove
def InfinitelyRepresentable (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, ∀ k : ℕ, ∃ n > k, ∃ j > i, ∃ r s : ℕ+, a n = r * a i + s * a j

-- State the theorem
theorem infinite_representable_theorem (a : ℕ → ℕ) (h : IncreasingSequence a) :
  InfinitelyRepresentable a := by
  sorry

end NUMINAMATH_CALUDE_infinite_representable_theorem_l2560_256084


namespace NUMINAMATH_CALUDE_charlie_feather_count_l2560_256020

/-- The number of feathers Charlie already has -/
def feathers_already_has : ℕ := 387

/-- The number of feathers Charlie needs to collect -/
def feathers_to_collect : ℕ := 513

/-- The total number of feathers Charlie needs for his wings -/
def total_feathers_needed : ℕ := feathers_already_has + feathers_to_collect

theorem charlie_feather_count : total_feathers_needed = 900 := by
  sorry

end NUMINAMATH_CALUDE_charlie_feather_count_l2560_256020


namespace NUMINAMATH_CALUDE_range_of_a_l2560_256009

-- Define the system of inequalities
def inequality_system (x a : ℝ) : Prop :=
  3 * x - a > x + 1 ∧ (3 * x - 2) / 2 < 1 + x

-- Define the condition of having exactly 3 integer solutions
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, inequality_system x a

-- The main theorem
theorem range_of_a (a : ℝ) :
  has_three_integer_solutions a → -1 ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2560_256009


namespace NUMINAMATH_CALUDE_triangle_problem_l2560_256089

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (2 * cos C * (a * cos B + b * cos A) = c) →
  (c = Real.sqrt 7) →
  (1/2 * a * b * sin C = 3 * Real.sqrt 3 / 2) →
  (C = π/3 ∧ a + b + c = 5 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2560_256089


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l2560_256052

theorem sine_cosine_sum (α : Real) : 
  (∃ (x y : Real), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ 
    Real.cos α = x ∧ Real.sin α = y) → 
  Real.sin α + 2 * Real.cos α = 2 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l2560_256052


namespace NUMINAMATH_CALUDE_simplify_expression_l2560_256003

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2560_256003


namespace NUMINAMATH_CALUDE_equation_solution_l2560_256041

theorem equation_solution : ∃ x : ℝ, 61 + x * 12 / (180 / 3) = 62 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2560_256041


namespace NUMINAMATH_CALUDE_original_number_proof_l2560_256078

theorem original_number_proof (x : ℕ) : (10 * x + 9) + 2 * x = 633 → x = 52 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2560_256078


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2560_256058

theorem solve_linear_equation :
  ∀ x : ℚ, -3 * x - 8 = 4 * x + 3 → x = -11/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2560_256058


namespace NUMINAMATH_CALUDE_shorter_side_length_l2560_256075

-- Define the circle and rectangle
def circle_radius : ℝ := 6

-- Define the relationship between circle and rectangle areas
def rectangle_area (circle_area : ℝ) : ℝ := 3 * circle_area

-- Define the theorem
theorem shorter_side_length (circle_area : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_area = π * circle_radius ^ 2)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : rectangle_area = (2 * circle_radius) * shorter_side) :
  shorter_side = 9 * π := by
  sorry


end NUMINAMATH_CALUDE_shorter_side_length_l2560_256075


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2560_256076

theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := fun x y => x^2 / 4 - y^2 / 9 = 1
  ∀ x y : ℝ, (∃ t : ℝ, t ≠ 0 ∧ h (t * x) (t * y)) ↔ y = (3/2) * x ∨ y = -(3/2) * x :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2560_256076


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_l2560_256070

theorem simple_interest_rate_problem (P A T : ℕ) (h1 : P = 25000) (h2 : A = 35500) (h3 : T = 12) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 35 / 10 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_l2560_256070


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l2560_256079

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hitting_ground_time : 
  ∃ t : ℝ, t = 1 + (Real.sqrt 19) / 2 ∧ 
  (∀ y : ℝ, y = -16 * t^2 + 32 * t + 60 → y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l2560_256079


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2560_256055

theorem mean_equality_implies_x_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + x) / 2
  mean1 = mean2 → x = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2560_256055


namespace NUMINAMATH_CALUDE_painting_theorem_l2560_256002

/-- Represents the portion of a wall painted in a given time -/
def paint_portion (rate : ℚ) (time : ℚ) : ℚ := rate * time

/-- The combined painting rate of two painters -/
def combined_rate (rate1 : ℚ) (rate2 : ℚ) : ℚ := rate1 + rate2

theorem painting_theorem (heidi_rate liam_rate : ℚ) 
  (h1 : heidi_rate = 1 / 60)
  (h2 : liam_rate = 1 / 90)
  (time : ℚ)
  (h3 : time = 15) :
  paint_portion (combined_rate heidi_rate liam_rate) time = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_painting_theorem_l2560_256002


namespace NUMINAMATH_CALUDE_black_friday_tv_sales_increase_black_friday_tv_sales_increase_proof_l2560_256040

theorem black_friday_tv_sales_increase : ℕ → Prop :=
  fun increase =>
    ∃ (T : ℕ),
      T + increase = 327 ∧
      T + 3 * increase = 477 ∧
      increase = 75

-- The proof would go here, but we'll use sorry as instructed
theorem black_friday_tv_sales_increase_proof :
  ∃ increase, black_friday_tv_sales_increase increase :=
by sorry

end NUMINAMATH_CALUDE_black_friday_tv_sales_increase_black_friday_tv_sales_increase_proof_l2560_256040


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2560_256054

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 5*x + 6 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 5 ∧ s₁ * s₂ = 6 ∧ s₁^2 + s₂^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2560_256054


namespace NUMINAMATH_CALUDE_functions_equality_l2560_256061

theorem functions_equality (x : ℝ) : 2 * |x| = Real.sqrt (4 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_functions_equality_l2560_256061


namespace NUMINAMATH_CALUDE_xy_range_l2560_256033

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.exp x = x * y * (2 * Real.log x + Real.log y)) : 
  x * y ≥ Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_xy_range_l2560_256033


namespace NUMINAMATH_CALUDE_rope_fraction_proof_l2560_256035

theorem rope_fraction_proof (total_ropes : ℕ) (avg_length total_length : ℝ) 
  (group1_avg group2_avg : ℝ) (f : ℝ) :
  total_ropes = 6 →
  avg_length = 80 →
  total_length = avg_length * total_ropes →
  group1_avg = 70 →
  group2_avg = 85 →
  total_length = group1_avg * (f * total_ropes) + group2_avg * ((1 - f) * total_ropes) →
  f = 1 / 3 := by
  sorry

#check rope_fraction_proof

end NUMINAMATH_CALUDE_rope_fraction_proof_l2560_256035


namespace NUMINAMATH_CALUDE_rabbit_exchange_l2560_256083

/-- The exchange problem between two rabbits --/
theorem rabbit_exchange (white_carrots gray_cabbages : ℕ) 
  (h1 : white_carrots = 180) 
  (h2 : gray_cabbages = 120) : 
  ∃ (x : ℕ), x > 0 ∧ x < gray_cabbages ∧ 
  (gray_cabbages - x + 3 * x = (white_carrots + gray_cabbages) / 2) ∧
  (white_carrots - 3 * x + x = (white_carrots + gray_cabbages) / 2) := by
sorry

#eval (180 + 120) / 2  -- Expected output: 150

end NUMINAMATH_CALUDE_rabbit_exchange_l2560_256083


namespace NUMINAMATH_CALUDE_circle_circumference_limit_l2560_256007

open Real

theorem circle_circumference_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * π * (C / n) - C| < ε :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_limit_l2560_256007
