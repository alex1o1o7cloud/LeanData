import Mathlib

namespace harry_fish_count_l637_63723

/-- Given three friends with fish, prove Harry's fish count -/
theorem harry_fish_count (sam joe harry : ℕ) : 
  sam = 7 →
  joe = 8 * sam →
  harry = 4 * joe →
  harry = 224 := by
  sorry

end harry_fish_count_l637_63723


namespace prob_at_least_six_heads_in_nine_flips_prob_at_least_six_heads_in_nine_flips_proof_l637_63778

/-- The probability of getting at least 6 heads in 9 fair coin flips -/
theorem prob_at_least_six_heads_in_nine_flips : ℝ :=
  130 / 512

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The probability of getting exactly k heads in n fair coin flips -/
def prob_exactly_k_heads (n k : ℕ) : ℝ := sorry

/-- The probability of getting at least k heads in n fair coin flips -/
def prob_at_least_k_heads (n k : ℕ) : ℝ := sorry

theorem prob_at_least_six_heads_in_nine_flips_proof :
  prob_at_least_k_heads 9 6 = prob_at_least_six_heads_in_nine_flips :=
by sorry

end prob_at_least_six_heads_in_nine_flips_prob_at_least_six_heads_in_nine_flips_proof_l637_63778


namespace systematic_sampling_fourth_element_l637_63785

/-- Systematic sampling function -/
def systematicSample (totalSize : Nat) (sampleSize : Nat) : Nat → Nat :=
  fun i => i * (totalSize / sampleSize) + 1

theorem systematic_sampling_fourth_element
  (totalSize : Nat)
  (sampleSize : Nat)
  (h1 : totalSize = 36)
  (h2 : sampleSize = 4)
  (h3 : systematicSample totalSize sampleSize 0 = 6)
  (h4 : systematicSample totalSize sampleSize 2 = 24)
  (h5 : systematicSample totalSize sampleSize 3 = 33) :
  systematicSample totalSize sampleSize 1 = 15 := by
sorry

end systematic_sampling_fourth_element_l637_63785


namespace median_distance_product_sum_l637_63729

/-- Given a triangle with medians of lengths s₁, s₂, s₃ and a point P with 
    distances d₁, d₂, d₃ to these medians respectively, prove that 
    s₁d₁ + s₂d₂ + s₃d₃ = 0 -/
theorem median_distance_product_sum (s₁ s₂ s₃ d₁ d₂ d₃ : ℝ) : 
  s₁ * d₁ + s₂ * d₂ + s₃ * d₃ = 0 := by
  sorry

end median_distance_product_sum_l637_63729


namespace second_box_capacity_l637_63725

/-- Represents the dimensions and capacity of a rectangular box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.height * b.width * b.length

theorem second_box_capacity (box1 box2 : Box) : 
  box1.height = 1.5 ∧ 
  box1.width = 4 ∧ 
  box1.length = 6 ∧ 
  box1.capacity = 72 ∧
  box2.height = 3 * box1.height ∧
  box2.width = 2 * box1.width ∧
  box2.length = 0.5 * box1.length →
  box2.capacity = 216 := by
  sorry

end second_box_capacity_l637_63725


namespace stating_solutions_depend_on_angle_l637_63721

/-- Represents a plane in 3D space --/
structure Plane where
  s₁ : ℝ
  s₂ : ℝ

/-- Represents an axis in 3D space --/
structure Axis where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the number of solutions --/
inductive NumSolutions
  | zero
  | one
  | two

/-- 
Given a plane S and an angle α₁ between S and the horizontal plane,
determines the number of possible x₁,₄ axes such that S is perpendicular
to the bisectors of the first and fourth quadrants.
--/
def num_solutions (S : Plane) (α₁ : ℝ) : NumSolutions :=
  sorry

/-- 
Theorem stating that the number of solutions depends on α₁
--/
theorem solutions_depend_on_angle (S : Plane) (α₁ : ℝ) :
  (α₁ > 45 → num_solutions S α₁ = NumSolutions.two) ∧
  (α₁ ≤ 45 → num_solutions S α₁ = NumSolutions.one ∨ num_solutions S α₁ = NumSolutions.zero) :=
  sorry

end stating_solutions_depend_on_angle_l637_63721


namespace line_parameterization_l637_63714

/-- Given a line y = 5x - 7 parameterized by [x; y] = [s; 2] + t[3; h], 
    prove that s = 9/5 and h = 15 -/
theorem line_parameterization (x y s h t : ℝ) : 
  y = 5 * x - 7 ∧ 
  ∃ (v : ℝ × ℝ), v.1 = x ∧ v.2 = y ∧ v = (s, 2) + t • (3, h) →
  s = 9/5 ∧ h = 15 := by
  sorry

end line_parameterization_l637_63714


namespace dog_division_theorem_l637_63792

def number_of_dogs : ℕ := 12
def group_sizes : List ℕ := [4, 5, 3]

def ways_to_divide_dogs (n : ℕ) (sizes : List ℕ) : ℕ :=
  sorry

theorem dog_division_theorem :
  ways_to_divide_dogs number_of_dogs group_sizes = 4200 :=
by sorry

end dog_division_theorem_l637_63792


namespace f_difference_l637_63740

/-- The function f(x) = 2x^3 - 3x^2 + 4x - 5 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4 * x - 5

/-- Theorem stating that f(x + h) - f(x) equals the given expression -/
theorem f_difference (x h : ℝ) : 
  f (x + h) - f x = 6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h + 4 := by
  sorry

end f_difference_l637_63740


namespace frank_jim_speed_difference_l637_63716

theorem frank_jim_speed_difference : 
  ∀ (jim_distance frank_distance : ℝ) (time : ℝ),
    jim_distance = 16 →
    frank_distance = 20 →
    time = 2 →
    (frank_distance / time) - (jim_distance / time) = 2 := by
  sorry

end frank_jim_speed_difference_l637_63716


namespace multiply_72519_9999_l637_63798

theorem multiply_72519_9999 : 72519 * 9999 = 725117481 := by
  sorry

end multiply_72519_9999_l637_63798


namespace min_value_theorem_l637_63769

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * (2^x) + Real.log 2 * (8^y) = Real.log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log 2 * (2^a) + Real.log 2 * (8^b) = Real.log 2 → 
    1/x + 1/(3*y) ≤ 1/a + 1/(3*b)) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.log 2 * (2^x) + Real.log 2 * (8^y) = Real.log 2 ∧ 
    1/x + 1/(3*y) = 4) := by
  sorry

end min_value_theorem_l637_63769


namespace inequality_equivalence_l637_63730

-- Define the logarithm base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, (0 < x ∧ |x + log3 x| < |x| + |log3 x|) ↔ (0 < x ∧ x < 1) :=
sorry

end inequality_equivalence_l637_63730


namespace roots_sum_less_than_two_l637_63752

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the theorem
theorem roots_sum_less_than_two (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ = m → f x₂ = m → x₁ + x₂ < 2 := by
  sorry

end roots_sum_less_than_two_l637_63752


namespace at_least_one_quadratic_has_solution_l637_63760

theorem at_least_one_quadratic_has_solution (a b c : ℝ) : 
  (∃ x : ℝ, x^2 + (a-b)*x + (b-c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b-c)*x + (c-a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c-a)*x + (a-b) = 0) := by
  sorry

end at_least_one_quadratic_has_solution_l637_63760


namespace trapezoid_equal_area_segment_l637_63706

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_segment : ℝ
  equal_area_segment : ℝ
  midpoint_area_ratio : ℝ × ℝ
  longer_base_diff : longer_base = shorter_base + 150
  midpoint_segment_def : midpoint_segment = shorter_base + 75
  midpoint_area_ratio_def : midpoint_area_ratio = (3, 4)

/-- The main theorem about the trapezoid -/
theorem trapezoid_equal_area_segment (t : Trapezoid) :
  ⌊(t.equal_area_segment^2) / 150⌋ = 187 := by
  sorry

end trapezoid_equal_area_segment_l637_63706


namespace intersection_complement_equality_l637_63726

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by
  sorry

end intersection_complement_equality_l637_63726


namespace robbery_participants_l637_63763

-- Define the suspects
variable (A B V G : Prop)

-- A: Alexey is guilty
-- B: Boris is guilty
-- V: Veniamin is guilty
-- G: Grigory is guilty

-- Define the conditions
variable (h1 : ¬G → (B ∧ ¬A))
variable (h2 : V → (¬A ∧ ¬B))
variable (h3 : G → B)
variable (h4 : B → (A ∨ V))

-- Theorem statement
theorem robbery_participants : A ∧ B ∧ G ∧ ¬V := by
  sorry

end robbery_participants_l637_63763


namespace solution_set_of_inequality_l637_63700

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x - 1) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-7) ∪ Set.Ioi (-2) :=
sorry

end solution_set_of_inequality_l637_63700


namespace geometric_sequence_a5_l637_63767

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 3 * a 7 = 64 →
  a 5 = 8 := by sorry

end geometric_sequence_a5_l637_63767


namespace point_distance_to_line_l637_63772

theorem point_distance_to_line (m : ℝ) : 
  let M : ℝ × ℝ := (1, 4)
  let l := {(x, y) : ℝ × ℝ | m * x + y - 1 = 0}
  (abs (m * M.1 + M.2 - 1) / Real.sqrt (m^2 + 1) = 3) → (m = 0 ∨ m = 3/4) := by
sorry

end point_distance_to_line_l637_63772


namespace unique_prime_perfect_square_l637_63754

theorem unique_prime_perfect_square : 
  ∃! p : ℕ, Prime p ∧ ∃ n : ℕ, 5^p + 4*p^4 = n^2 ∧ p = 31 :=
sorry

end unique_prime_perfect_square_l637_63754


namespace oldest_babysat_age_l637_63770

/-- Represents Jane's babysitting career and age information -/
structure BabysittingCareer where
  current_age : ℕ
  years_since_stopped : ℕ
  start_age : ℕ

/-- Calculates the maximum age of a child Jane could babysit at a given time -/
def max_child_age (jane_age : ℕ) : ℕ :=
  jane_age / 2

/-- Theorem stating the current age of the oldest person Jane could have babysat -/
theorem oldest_babysat_age (jane : BabysittingCareer)
  (h1 : jane.current_age = 34)
  (h2 : jane.years_since_stopped = 10)
  (h3 : jane.start_age = 18) :
  jane.current_age - jane.years_since_stopped - max_child_age (jane.current_age - jane.years_since_stopped) + jane.years_since_stopped = 22 :=
by
  sorry

#check oldest_babysat_age

end oldest_babysat_age_l637_63770


namespace max_value_on_parabola_l637_63786

/-- The maximum value of m + n where (m,n) lies on y = -x^2 + 3 is 13/4 -/
theorem max_value_on_parabola :
  ∀ m n : ℝ, n = -m^2 + 3 → (∀ x y : ℝ, y = -x^2 + 3 → m + n ≥ x + y) → m + n = 13/4 := by
  sorry

end max_value_on_parabola_l637_63786


namespace quadruple_cylinder_volume_l637_63738

/-- Theorem: Quadrupling Cylinder Dimensions -/
theorem quadruple_cylinder_volume (V : ℝ) (V' : ℝ) :
  V > 0 →  -- Assume positive initial volume
  V' = 64 * V →  -- Definition of V' based on problem conditions
  V' = (4^3) * V  -- Conclusion to prove
  := by sorry

end quadruple_cylinder_volume_l637_63738


namespace sum_of_reciprocals_find_b_find_c_find_d_l637_63746

-- Problem 1
theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 10) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem find_b (b : ℝ) (h1 : b > 0) (h2 : b^2 - 1 = 135 * 137) :
  b = 136 := by sorry

-- Problem 3
theorem find_c (c : ℝ) 
  (h : (1 : ℝ) * (-1 / 2) * (-c / 3) = -1) :
  c = -6 := by sorry

-- Problem 4
theorem find_d (c d : ℝ) 
  (h : (d - 1) / c = -1) (h2 : c = 2) :
  d = 7 := by sorry

end sum_of_reciprocals_find_b_find_c_find_d_l637_63746


namespace rectangular_solid_diagonal_l637_63707

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 54)
  (h2 : 4 * (a + b + c) = 40)
  (h3 : c = a + b) :
  a^2 + b^2 + c^2 = 46 := by
sorry

end rectangular_solid_diagonal_l637_63707


namespace arithmetic_sequence_2023_l637_63791

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_sequence_2023 (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d →
  a 1 = 2 →
  GeometricSequence (a 1) (a 3) (a 7) →
  a 2023 = 2024 :=
by sorry

end arithmetic_sequence_2023_l637_63791


namespace multiples_of_ten_range_l637_63787

theorem multiples_of_ten_range (start : ℕ) : 
  (∃ n : ℕ, n = 991) ∧ 
  (start ≤ 10000) ∧
  (∀ x ∈ Set.Icc start 10000, x % 10 = 0 → x ∈ Finset.range 992) ∧
  (10000 ∈ Finset.range 992) →
  start = 90 := by
sorry

end multiples_of_ten_range_l637_63787


namespace solution_set_equivalence_l637_63705

theorem solution_set_equivalence (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (h : ∀ x : ℝ, mx + n > 0 ↔ x > 2/5) : 
  ∀ x : ℝ, nx - m < 0 ↔ x > -5/2 := by
sorry

end solution_set_equivalence_l637_63705


namespace a_can_be_any_real_l637_63735

theorem a_can_be_any_real : ∀ (a b c d : ℝ), 
  b * (3 * d + 2) ≠ 0 → 
  a / b < -c / (3 * d + 2) → 
  ∃ (a₁ a₂ a₃ : ℝ), a₁ > 0 ∧ a₂ < 0 ∧ a₃ = 0 ∧ 
    (a₁ / b < -c / (3 * d + 2)) ∧ 
    (a₂ / b < -c / (3 * d + 2)) ∧ 
    (a₃ / b < -c / (3 * d + 2)) := by
  sorry

end a_can_be_any_real_l637_63735


namespace min_adventurers_l637_63728

structure AdventurerGroup where
  rubies : Finset Nat
  emeralds : Finset Nat
  sapphires : Finset Nat
  diamonds : Finset Nat

def AdventurerGroup.valid (g : AdventurerGroup) : Prop :=
  g.rubies.card = 5 ∧
  g.emeralds.card = 11 ∧
  g.sapphires.card = 10 ∧
  g.diamonds.card = 6 ∧
  (∀ a ∈ g.diamonds, (a ∈ g.emeralds ∨ a ∈ g.sapphires) ∧ ¬(a ∈ g.emeralds ∧ a ∈ g.sapphires)) ∧
  (∀ a ∈ g.emeralds, (a ∈ g.rubies ∨ a ∈ g.diamonds) ∧ ¬(a ∈ g.rubies ∧ a ∈ g.diamonds))

theorem min_adventurers (g : AdventurerGroup) (h : g.valid) :
  (g.rubies ∪ g.emeralds ∪ g.sapphires ∪ g.diamonds).card ≥ 16 := by
  sorry

#check min_adventurers

end min_adventurers_l637_63728


namespace point_in_second_quadrant_l637_63731

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -2
  let y : ℝ := 3
  second_quadrant x y :=
by sorry

end point_in_second_quadrant_l637_63731


namespace percent_of_x_is_y_l637_63773

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end percent_of_x_is_y_l637_63773


namespace transformation_matrix_exists_and_unique_l637_63795

open Matrix

theorem transformation_matrix_exists_and_unique :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ, 
    ∀ A : Matrix (Fin 2) (Fin 2) ℝ, 
      N * A = !![4 * A 0 0, 4 * A 0 1; A 1 0, A 1 1] := by
  sorry

end transformation_matrix_exists_and_unique_l637_63795


namespace array_transformation_theorem_l637_63742

/-- Represents an 8x8 array of +1 and -1 -/
def Array8x8 := Fin 8 → Fin 8 → Int

/-- Represents a move in the array -/
structure Move where
  row : Fin 8
  col : Fin 8

/-- Applies a move to an array -/
def applyMove (arr : Array8x8) (m : Move) : Array8x8 :=
  fun i j => if i = m.row ∨ j = m.col then -arr i j else arr i j

/-- Checks if an array is all +1 -/
def isAllPlusOne (arr : Array8x8) : Prop :=
  ∀ i j, arr i j = 1

theorem array_transformation_theorem :
  ∀ (initial : Array8x8),
  (∀ i j, initial i j = 1 ∨ initial i j = -1) →
  ∃ (moves : List Move),
  isAllPlusOne (moves.foldl applyMove initial) :=
sorry

end array_transformation_theorem_l637_63742


namespace isosceles_trapezoid_shorter_base_l637_63775

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the longer base
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  is_isosceles : a > b -- Condition for isosceles trapezoid

/-- 
  Theorem: In an isosceles trapezoid, if the foot of the height from a vertex 
  of the shorter base divides the longer base into two segments with a 
  difference of 10 units, then the length of the shorter base is 10 units.
-/
theorem isosceles_trapezoid_shorter_base 
  (t : IsoscelesTrapezoid) 
  (h : (t.a + t.b) / 2 = (t.a - t.b) / 2 + 10) : 
  t.b = 10 := by
  sorry

end isosceles_trapezoid_shorter_base_l637_63775


namespace not_square_expression_l637_63715

theorem not_square_expression (n : ℕ) (a : ℕ) (h1 : n > 2) (h2 : Odd a) (h3 : a > 0) : 
  let b := 2^(2^n)
  a ≤ b ∧ b ≤ 2*a → ¬ ∃ (k : ℕ), a^2 + b^2 - a*b = k^2 := by
  sorry

end not_square_expression_l637_63715


namespace isosceles_triangle_perimeter_l637_63774

/-- An isosceles triangle with side lengths 1 and 2 has perimeter 5 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) →
  (a = b ∨ b = c ∨ a = c) →
  a + b + c = 5 := by
  sorry


end isosceles_triangle_perimeter_l637_63774


namespace shift_theorem_l637_63765

/-- Represents a quadratic function of the form a(x-h)^2 + k --/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function horizontally --/
def horizontal_shift (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h + d, k := f.k }

/-- Shifts a quadratic function vertically --/
def vertical_shift (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h, k := f.k + d }

/-- The original quadratic function y = 2(x-2)^2 - 5 --/
def original_function : QuadraticFunction :=
  { a := 2, h := 2, k := -5 }

/-- The resulting function after shifts --/
def shifted_function : QuadraticFunction :=
  { a := 2, h := 4, k := -2 }

theorem shift_theorem :
  (vertical_shift (horizontal_shift original_function 2) 3) = shifted_function := by
  sorry

end shift_theorem_l637_63765


namespace friend_ate_two_slices_l637_63749

/-- Calculates the number of slices James's friend ate given the initial number of slices,
    the number James ate, and the fact that James ate half of the remaining slices. -/
def friend_slices (total : ℕ) (james_ate : ℕ) : ℕ :=
  total - 2 * james_ate

theorem friend_ate_two_slices :
  let total := 8
  let james_ate := 3
  friend_slices total james_ate = 2 := by
  sorry

end friend_ate_two_slices_l637_63749


namespace sector_area_proof_l637_63709

/-- Given a circle where a central angle of 2 radians corresponds to an arc length of 2 cm,
    prove that the area of the sector formed by this central angle is 1 cm². -/
theorem sector_area_proof (r : ℝ) (θ : ℝ) (l : ℝ) (A : ℝ) : 
  θ = 2 → l = 2 → l = r * θ → A = (1/2) * r^2 * θ → A = 1 := by
  sorry

end sector_area_proof_l637_63709


namespace cylinder_from_equation_l637_63718

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def isCylinder (S : Set CylindricalPoint) (d : ℝ) : Prop :=
  d > 0 ∧ S = {p : CylindricalPoint | p.r = d}

/-- The main theorem: the set of points satisfying r = d forms a cylinder -/
theorem cylinder_from_equation (d : ℝ) :
  let S := {p : CylindricalPoint | p.r = d}
  d > 0 → isCylinder S d := by
  sorry


end cylinder_from_equation_l637_63718


namespace function_domain_implies_m_range_l637_63751

theorem function_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m ≤ 4 :=
by sorry

end function_domain_implies_m_range_l637_63751


namespace light_ray_equation_l637_63732

/-- A light ray is emitted from point A(-3, 3), hits the x-axis, gets reflected, and is tangent to a circle. This theorem proves that the equation of the line on which the light ray lies is either 3x + 4y - 3 = 0 or 4x + 3y + 3 = 0. -/
theorem light_ray_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-3, 3)
  let circle (x y : ℝ) := x^2 + y^2 - 4*x - 4*y + 7 = 0
  let ray_hits_x_axis : Prop := ∃ (t : ℝ), t * (A.1 + 3) = -3 ∧ t * (A.2 - 3) = 0
  let is_tangent_to_circle : Prop := ∃ (x₀ y₀ : ℝ), circle x₀ y₀ ∧ 
    ((x - x₀) * (2*x₀ - 4) + (y - y₀) * (2*y₀ - 4) = 0)
  ray_hits_x_axis → is_tangent_to_circle → 
    (3*x + 4*y - 3 = 0) ∨ (4*x + 3*y + 3 = 0) :=
by sorry


end light_ray_equation_l637_63732


namespace correct_speeds_l637_63703

/-- Two points moving uniformly along a circumference -/
structure MovingPoints where
  circumference : ℝ
  time_difference : ℝ
  coincidence_interval : ℝ

/-- The speeds of the two points -/
def speeds (mp : MovingPoints) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct speeds for the given conditions -/
theorem correct_speeds (mp : MovingPoints) 
  (h1 : mp.circumference = 60)
  (h2 : mp.time_difference = 5)
  (h3 : mp.coincidence_interval = 60) :
  speeds mp = (3, 4) :=
sorry

end correct_speeds_l637_63703


namespace lawn_mowing_price_l637_63750

def sneaker_cost : ℕ := 92
def lawns_to_mow : ℕ := 3
def figures_to_sell : ℕ := 2
def figure_price : ℕ := 9
def job_hours : ℕ := 10
def hourly_rate : ℕ := 5

theorem lawn_mowing_price : 
  (sneaker_cost - (figures_to_sell * figure_price + job_hours * hourly_rate)) / lawns_to_mow = 8 := by
  sorry

end lawn_mowing_price_l637_63750


namespace binomial_inequality_l637_63766

theorem binomial_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) : 
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k : ℝ) * ((n-k)^(n-k) : ℝ)) < 
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n-k).factorial : ℝ)) ∧
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n-k).factorial : ℝ)) < 
  (n^n : ℝ) / ((k^k : ℝ) * ((n-k)^(n-k) : ℝ)) :=
by sorry

end binomial_inequality_l637_63766


namespace darker_tile_fraction_is_three_fourths_l637_63764

/-- Represents a floor with a repeating tile pattern -/
structure Floor :=
  (pattern_size : Nat)
  (corner_size : Nat)
  (dark_tiles_in_corner : Nat)

/-- The fraction of darker tiles in the floor -/
def darker_tile_fraction (f : Floor) : Rat :=
  let total_tiles := f.pattern_size * f.pattern_size
  let corner_tiles := f.corner_size * f.corner_size
  let num_corners := (f.pattern_size / f.corner_size) ^ 2
  let total_dark_tiles := f.dark_tiles_in_corner * num_corners
  total_dark_tiles / total_tiles

/-- Theorem stating that for a floor with a 4x4 repeating pattern and 3 darker tiles in each 2x2 corner,
    the fraction of darker tiles is 3/4 -/
theorem darker_tile_fraction_is_three_fourths (f : Floor)
  (h1 : f.pattern_size = 4)
  (h2 : f.corner_size = 2)
  (h3 : f.dark_tiles_in_corner = 3) :
  darker_tile_fraction f = 3/4 := by
  sorry

end darker_tile_fraction_is_three_fourths_l637_63764


namespace rooster_earnings_l637_63758

def price_per_kg : ℚ := 1/2

def rooster1_weight : ℚ := 30
def rooster2_weight : ℚ := 40

def total_earnings : ℚ := price_per_kg * (rooster1_weight + rooster2_weight)

theorem rooster_earnings : total_earnings = 35 := by
  sorry

end rooster_earnings_l637_63758


namespace tickets_to_buy_l637_63733

def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def multiple_ride_discount : ℝ := 1.0
def newspaper_coupon : ℝ := 1.0

theorem tickets_to_buy :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - newspaper_coupon = 7.0 := by
  sorry

end tickets_to_buy_l637_63733


namespace measles_cases_1987_l637_63734

/-- Calculates the number of measles cases in a given year assuming a linear decrease --/
def measlesCases (initialYear finalYear targetYear : ℕ) (initialCases finalCases : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let targetYears := targetYear - initialYear
  let totalDecrease := initialCases - finalCases
  let decrease := (targetYears * totalDecrease) / totalYears
  initialCases - decrease

/-- Theorem stating that the number of measles cases in 1987 would be 112,875 --/
theorem measles_cases_1987 :
  measlesCases 1960 1996 1987 450000 500 = 112875 := by
  sorry

#eval measlesCases 1960 1996 1987 450000 500

end measles_cases_1987_l637_63734


namespace square_root_problem_l637_63741

theorem square_root_problem (a b : ℝ) 
  (h1 : Real.sqrt (a + 9) = -5)
  (h2 : (2 * b - a) ^ (1/3 : ℝ) = -2) :
  Real.sqrt (2 * a + b) = 6 := by
sorry

end square_root_problem_l637_63741


namespace geometric_sequence_properties_l637_63789

-- Define a geometric sequence
def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

-- Define the problem statement
theorem geometric_sequence_properties
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b) :
  (is_geometric_sequence (λ n => a n * b n)) ∧
  ¬(∀ x y : ℕ → ℝ, is_geometric_sequence x → is_geometric_sequence y →
    is_geometric_sequence (λ n => x n + y n)) :=
by sorry

end geometric_sequence_properties_l637_63789


namespace right_triangle_circle_intersection_l637_63779

-- Define the triangle and circle
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem right_triangle_circle_intersection
  (triangle : RightTriangle)
  (circle : Circle)
  (D : ℝ × ℝ)
  (h1 : circle.center = ((triangle.B.1 + triangle.C.1) / 2, (triangle.B.2 + triangle.C.2) / 2))
  (h2 : circle.radius = Real.sqrt ((triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2) / 2)
  (h3 : D.1 = triangle.A.1 + 2 * (triangle.C.1 - triangle.A.1) / (triangle.C.1 - triangle.A.1 + triangle.C.2 - triangle.A.2))
  (h4 : D.2 = triangle.A.2 + 2 * (triangle.C.2 - triangle.A.2) / (triangle.C.1 - triangle.A.1 + triangle.C.2 - triangle.A.2))
  (h5 : Real.sqrt ((D.1 - triangle.A.1)^2 + (D.2 - triangle.A.2)^2) = 2)
  (h6 : Real.sqrt ((D.1 - triangle.B.1)^2 + (D.2 - triangle.B.2)^2) = 3)
  : Real.sqrt ((D.1 - triangle.C.1)^2 + (D.2 - triangle.C.2)^2) = 4.5 := by
  sorry

end right_triangle_circle_intersection_l637_63779


namespace expression_simplification_l637_63708

theorem expression_simplification 
  (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) : 
  a * (1/b + 1/c) + b * (1/a + 1/c) + c * (1/a + 1/b) = -3 := by
sorry

end expression_simplification_l637_63708


namespace range_of_x_when_a_is_one_range_of_a_for_not_p_necessary_not_sufficient_l637_63702

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Theorem for the first part of the problem
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → x ∈ Set.Ioo 2 4 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_not_p_necessary_not_sufficient :
  ∀ a : ℝ, (∀ x : ℝ, ¬(q x) → ¬(p x a)) ∧ (∃ x : ℝ, ¬(p x a) ∧ q x) → a ∈ Set.Ioo 1 2 := by sorry

end range_of_x_when_a_is_one_range_of_a_for_not_p_necessary_not_sufficient_l637_63702


namespace exponential_function_condition_l637_63796

theorem exponential_function_condition (x₁ x₂ : ℝ) :
  (x₁ + x₂ > 0) ↔ ((1/2 : ℝ)^x₁ * (1/2 : ℝ)^x₂ < 1) := by
  sorry

end exponential_function_condition_l637_63796


namespace problem_solution_l637_63781

theorem problem_solution (x : ℚ) : x = (1 / x) * (-x) - 3 * x + 4 → x = 3 / 4 := by
  sorry

end problem_solution_l637_63781


namespace train_length_l637_63747

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 18 seconds,
    and the platform length is 175 meters, the length of the train is 150 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 39)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 175) :
    ∃ train_length : ℝ, train_length = 150 := by
  sorry

end train_length_l637_63747


namespace multiply_x_equals_5_l637_63762

theorem multiply_x_equals_5 (x y : ℝ) (h1 : x * y ≠ 0) 
  (h2 : (1/5 * x) / (1/6 * y) = 0.7200000000000001) : 
  ∃ n : ℝ, n * x = 3 * y ∧ n = 5 := by
  sorry

end multiply_x_equals_5_l637_63762


namespace linear_function_properties_l637_63788

-- Define the linear function
def f (k x : ℝ) : ℝ := (3 - k) * x - 2 * k^2 + 18

theorem linear_function_properties :
  -- Part 1: The function passes through (0, -2) when k = ±√10
  (∃ k : ℝ, k^2 = 10 ∧ f k 0 = -2) ∧
  -- Part 2: The function is parallel to y = -x when k = 4
  (f 4 1 - f 4 0 = -1) ∧
  -- Part 3: The function decreases as x increases when k > 3
  (∀ k : ℝ, k > 3 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f k x₁ > f k x₂) :=
by sorry

end linear_function_properties_l637_63788


namespace largest_n_for_trig_inequality_l637_63799

theorem largest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / m) ∧
  n = 8 := by
  sorry

end largest_n_for_trig_inequality_l637_63799


namespace probability_same_color_is_15_364_l637_63755

def total_marbles : ℕ := 14
def red_marbles : ℕ := 3
def white_marbles : ℕ := 4
def blue_marbles : ℕ := 5
def green_marbles : ℕ := 2

def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem probability_same_color_is_15_364 :
  probability_same_color = 15 / 364 := by sorry

end probability_same_color_is_15_364_l637_63755


namespace negation_of_forall_geq_zero_is_exists_lt_zero_l637_63761

theorem negation_of_forall_geq_zero_is_exists_lt_zero :
  (¬ ∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) := by
  sorry

end negation_of_forall_geq_zero_is_exists_lt_zero_l637_63761


namespace root_difference_theorem_l637_63719

theorem root_difference_theorem (k : ℝ) : 
  (∃ α β : ℝ, (α^2 + k*α + 8 = 0 ∧ β^2 + k*β + 8 = 0) ∧
              ((α+3)^2 - k*(α+3) + 12 = 0 ∧ (β+3)^2 - k*(β+3) + 12 = 0)) →
  k = 3 := by
sorry

end root_difference_theorem_l637_63719


namespace divisibility_implies_equality_l637_63782

theorem divisibility_implies_equality (a b : ℕ+) (h : (a * b : ℕ) ∣ (a ^ 2 + b ^ 2 : ℕ)) : a = b := by
  sorry

end divisibility_implies_equality_l637_63782


namespace expression_evaluation_l637_63794

theorem expression_evaluation :
  let x : ℚ := -1/4
  (x - 1)^2 - 3*x*(1 - x) - (2*x - 1)*(2*x + 1) = 13/4 := by
  sorry

end expression_evaluation_l637_63794


namespace quadratic_two_zeros_l637_63745

/-- A quadratic function f(x) = ax² + bx + c has exactly two distinct real zeros when a·c < 0 -/
theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_two_zeros_l637_63745


namespace price_per_chicken_l637_63712

/-- Given Alan's market purchases, prove the price per chicken --/
theorem price_per_chicken (num_eggs : ℕ) (price_per_egg : ℕ) (num_chickens : ℕ) (total_spent : ℕ) :
  num_eggs = 20 →
  price_per_egg = 2 →
  num_chickens = 6 →
  total_spent = 88 →
  (total_spent - num_eggs * price_per_egg) / num_chickens = 8 := by
  sorry

end price_per_chicken_l637_63712


namespace cosine_even_and_decreasing_l637_63711

-- Define the properties of evenness and decreasing for a function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem cosine_even_and_decreasing :
  IsEven Real.cos ∧ IsDecreasingOn Real.cos 0 3 := by sorry

end cosine_even_and_decreasing_l637_63711


namespace age_sum_problem_l637_63710

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a * b * c = 256 → a + b + c = 20 :=
by sorry

end age_sum_problem_l637_63710


namespace problem_solution_l637_63790

theorem problem_solution : 
  ((-54 : ℚ) * (-1/2 + 2/3 - 4/9) = 15) ∧ 
  (-2 / (4/9) * (-2/3)^2 = -2) := by
sorry

end problem_solution_l637_63790


namespace saucer_area_l637_63776

/-- The area of a circular saucer with radius 3 centimeters is 9π square centimeters. -/
theorem saucer_area (π : ℝ) (h : π > 0) : 
  let r : ℝ := 3
  let area : ℝ := π * r^2
  area = 9 * π := by sorry

end saucer_area_l637_63776


namespace power_function_through_point_l637_63701

/-- A power function that passes through (2, 2√2) and evaluates to 27 at x = 9 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →  -- f is a power function
  f 2 = 2 * Real.sqrt 2 →  -- f passes through (2, 2√2)
  f 9 = 27 :=  -- prove that f(9) = 27
by sorry

end power_function_through_point_l637_63701


namespace final_amount_calculation_l637_63783

def monthly_salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def utility_bill_rate : ℝ := 0.25

theorem final_amount_calculation : 
  let tax := monthly_salary * tax_rate
  let insurance := monthly_salary * insurance_rate
  let after_deductions := monthly_salary - (tax + insurance)
  let utility_bills := after_deductions * utility_bill_rate
  monthly_salary - (tax + insurance + utility_bills) = 1125 := by
sorry

end final_amount_calculation_l637_63783


namespace triangle_altitude_l637_63777

theorem triangle_altitude (A b : ℝ) (h : A = 900 ∧ b = 45) :
  ∃ h : ℝ, A = (1/2) * b * h ∧ h = 40 := by
  sorry

end triangle_altitude_l637_63777


namespace predict_grain_demand_2012_l637_63757

/-- Regression equation for grain demand -/
def grain_demand (x : ℝ) : ℝ := 6.5 * (x - 2006) + 261

/-- Theorem: The predicted grain demand for 2012 is 300 ten thousand tons -/
theorem predict_grain_demand_2012 : grain_demand 2012 = 300 := by
  sorry

end predict_grain_demand_2012_l637_63757


namespace estimate_keyboard_warriors_opposition_l637_63736

/-- Estimates the number of people with a certain characteristic in a population based on a sample. -/
def estimatePopulation (totalPopulation : ℕ) (sampleSize : ℕ) (sampleOpposed : ℕ) : ℕ :=
  (totalPopulation * sampleOpposed) / sampleSize

/-- Theorem stating that the estimated number of people opposed to "keyboard warriors" is 6912. -/
theorem estimate_keyboard_warriors_opposition :
  let totalPopulation : ℕ := 9600
  let sampleSize : ℕ := 50
  let sampleOpposed : ℕ := 36
  estimatePopulation totalPopulation sampleSize sampleOpposed = 6912 := by
  sorry

#eval estimatePopulation 9600 50 36

end estimate_keyboard_warriors_opposition_l637_63736


namespace sum_of_integers_l637_63724

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
sorry

end sum_of_integers_l637_63724


namespace cubic_equation_solution_sum_l637_63780

/-- Given r, s, and t are solutions of x^3 - 6x^2 + 11x - 16 = 0, prove that (r+s)/t + (s+t)/r + (t+r)/s = 11/8 -/
theorem cubic_equation_solution_sum (r s t : ℝ) : 
  r^3 - 6*r^2 + 11*r - 16 = 0 →
  s^3 - 6*s^2 + 11*s - 16 = 0 →
  t^3 - 6*t^2 + 11*t - 16 = 0 →
  (r+s)/t + (s+t)/r + (t+r)/s = 11/8 := by
  sorry

end cubic_equation_solution_sum_l637_63780


namespace divisible_by_nine_l637_63748

theorem divisible_by_nine : ∃ (B : ℕ), B < 10 ∧ (7000 + 600 + 20 + B) % 9 = 0 := by
  sorry

end divisible_by_nine_l637_63748


namespace vasya_numbers_l637_63759

theorem vasya_numbers (x y : ℝ) : (x - 1) * (y - 1) = x * y → x + y = 1 := by
  sorry

end vasya_numbers_l637_63759


namespace more_trucks_than_buses_l637_63768

/-- Given 17 trucks and 9 buses, prove that there are 8 more trucks than buses. -/
theorem more_trucks_than_buses :
  let num_trucks : ℕ := 17
  let num_buses : ℕ := 9
  num_trucks - num_buses = 8 :=
by sorry

end more_trucks_than_buses_l637_63768


namespace point_movement_l637_63793

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moves a point on the number line -/
def move (p : Point) (units : ℤ) : Point :=
  ⟨p.value + units⟩

theorem point_movement (A B : Point) :
  A.value = -3 →
  B = move A 7 →
  B.value = 4 :=
by
  sorry

end point_movement_l637_63793


namespace product_sum_base_k_l637_63717

theorem product_sum_base_k (k : ℕ) (hk : k > 0) :
  (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5 →
  (3 * k + 14).digits k = [5, 0] :=
by sorry

end product_sum_base_k_l637_63717


namespace fourth_term_is_negative_twenty_l637_63713

def sequence_term (n : ℕ) : ℤ := (-1)^(n+1) * n * (n+1)

theorem fourth_term_is_negative_twenty : sequence_term 4 = -20 := by sorry

end fourth_term_is_negative_twenty_l637_63713


namespace euro_calculation_l637_63727

-- Define the € operation
def euro (x y : ℝ) : ℝ := 3 * x * y

-- Theorem statement
theorem euro_calculation : euro 3 (euro 4 5) = 540 := by
  sorry

end euro_calculation_l637_63727


namespace constant_c_value_l637_63771

theorem constant_c_value (b c : ℝ) :
  (∀ x : ℝ, 4 * (x + 2) * (x + b) = x^2 + c*x + 12) →
  c = 14 := by
sorry

end constant_c_value_l637_63771


namespace solve_timmys_orange_problem_l637_63797

/-- Represents the problem of calculating Timmy's remaining money after buying oranges --/
def timmys_orange_problem (calories_per_orange : ℕ) (oranges_per_pack : ℕ) 
  (price_per_orange : ℚ) (initial_money : ℚ) (calorie_goal : ℕ) (tax_rate : ℚ) : Prop :=
  let packs_needed : ℕ := ((calorie_goal + calories_per_orange - 1) / calories_per_orange + oranges_per_pack - 1) / oranges_per_pack
  let total_cost : ℚ := price_per_orange * (packs_needed * oranges_per_pack : ℚ)
  let tax_amount : ℚ := total_cost * tax_rate
  let final_cost : ℚ := total_cost + tax_amount
  let remaining_money : ℚ := initial_money - final_cost
  remaining_money = 244/100

/-- Theorem stating the solution to Timmy's orange problem --/
theorem solve_timmys_orange_problem : 
  timmys_orange_problem 80 3 (120/100) 10 400 (5/100) :=
by
  sorry

end solve_timmys_orange_problem_l637_63797


namespace tangent_line_at_point_one_l637_63744

/-- The function f(x) = x^2 + x - 1 -/
def f (x : ℝ) : ℝ := x^2 + x - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2*x + 1

theorem tangent_line_at_point_one :
  ∃ (m b : ℝ), 
    (f 1 = 1) ∧ 
    (f' 1 = m) ∧ 
    (∀ x y : ℝ, y = m * (x - 1) + 1 ↔ m * x - y + b = 0) ∧
    (3 * 1 - 1 + b = 0) ∧
    (∀ x y : ℝ, y = m * (x - 1) + 1 ↔ 3 * x - y - 2 = 0) :=
by sorry

end tangent_line_at_point_one_l637_63744


namespace second_integer_value_l637_63784

theorem second_integer_value (n : ℤ) : (n - 2) + (n + 2) = 132 → n = 66 := by
  sorry

end second_integer_value_l637_63784


namespace smallest_integer_with_remainder_two_l637_63720

theorem smallest_integer_with_remainder_two : ∃ n : ℕ, 
  (n > 20) ∧ 
  (∀ m : ℕ, m > 20 → 
    ((m % 3 = 2) ∧ (m % 4 = 2) ∧ (m % 5 = 2) ∧ (m % 6 = 2)) → 
    (n ≤ m)) ∧
  (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) :=
by
  -- The proof goes here
  sorry

#eval Nat.lcm (Nat.lcm 3 4) (Nat.lcm 5 6)  -- This should output 60

end smallest_integer_with_remainder_two_l637_63720


namespace maximum_marks_calculation_l637_63737

theorem maximum_marks_calculation (victor_percentage : ℝ) (victor_marks : ℝ) : 
  victor_percentage = 92 → 
  victor_marks = 460 → 
  (victor_marks / (victor_percentage / 100)) = 500 := by
sorry

end maximum_marks_calculation_l637_63737


namespace factor_expression_l637_63704

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) := by
  sorry

end factor_expression_l637_63704


namespace sum_of_x_and_y_l637_63756

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : x^2 * y^3 + y^2 * x^3 = 27) 
  (h2 : x * y = 3) : 
  x + y = 3 := by
sorry

end sum_of_x_and_y_l637_63756


namespace candy_bar_profit_l637_63753

/-- Calculates the profit from selling candy bars given the number of boxes, bars per box, selling price, and cost price. -/
def calculate_profit (boxes : ℕ) (bars_per_box : ℕ) (selling_price : ℚ) (cost_price : ℚ) : ℚ :=
  let total_bars := boxes * bars_per_box
  let revenue := total_bars * selling_price
  let cost := total_bars * cost_price
  revenue - cost

/-- Proves that the profit from selling 5 boxes of candy bars, with 10 bars per box,
    selling price of $1.50 per bar, and cost price of $1 per bar, is equal to $25. -/
theorem candy_bar_profit :
  calculate_profit 5 10 (3/2) 1 = 25 := by
  sorry

end candy_bar_profit_l637_63753


namespace acute_triangle_theorem_l637_63722

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

/-- Main theorem about the acute triangle -/
theorem acute_triangle_theorem (t : AcuteTriangle) :
  (Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A) →
  (t.c = Real.sqrt 7 ∧ t.a * t.b = 6) →
  (t.C = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7) :=
by sorry


end acute_triangle_theorem_l637_63722


namespace sarah_molly_groups_l637_63739

def chess_club_size : ℕ := 12
def group_size : ℕ := 6

theorem sarah_molly_groups (sarah molly : Fin chess_club_size) 
  (h_distinct : sarah ≠ molly) : 
  (Finset.univ.filter (λ s : Finset (Fin chess_club_size) => 
    s.card = group_size ∧ sarah ∈ s ∧ molly ∈ s)).card = 210 := by
  sorry

end sarah_molly_groups_l637_63739


namespace marias_sister_bottles_l637_63743

/-- Given the initial number of water bottles, the number Maria drank, and the number left,
    calculate the number of bottles Maria's sister drank. -/
theorem marias_sister_bottles (initial : ℝ) (maria_drank : ℝ) (left : ℝ) :
  initial = 45.0 →
  maria_drank = 14.0 →
  left = 23.0 →
  initial - maria_drank - left = 8.0 := by
  sorry

end marias_sister_bottles_l637_63743
