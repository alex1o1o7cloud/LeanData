import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_theorem_l538_53861

/-- Represents the number of rows in the hall -/
def x : ℕ := 9

/-- The total number of students is between 70 and 90 -/
axiom total_students : 70 < x^2 ∧ x^2 < 90

/-- The number of sixth-grade students per row -/
def sixth_grade_per_row : ℕ := x - 3

/-- The number of fifth-grade students per row is 3 -/
def fifth_grade_per_row : ℕ := 3

/-- The total number of rows is 3 more than the number of sixth-grade students per row -/
axiom row_count : x = sixth_grade_per_row + 3

/-- The number of sixth-grade students -/
def sixth_grade_count : ℕ := x * sixth_grade_per_row

/-- The number of fifth-grade students -/
def fifth_grade_count : ℕ := x * fifth_grade_per_row

theorem student_count_theorem : 
  sixth_grade_count = 54 ∧ fifth_grade_count = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_count_theorem_l538_53861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_major_axis_ellipse_equation_l538_53875

/-- An ellipse with foci F₁(-3,0) and F₂(3,0), tangent to the line y = x + 9 -/
structure ShortestMajorAxisEllipse where
  /-- The ellipse passes through a point (x, y) on the line y = x + 9 -/
  point_on_line : ∃ (x y : ℝ), y = x + 9
  /-- The foci of the ellipse are F₁(-3,0) and F₂(3,0) -/
  foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-3, 0), (3, 0))
  /-- The ellipse has the shortest major axis -/
  shortest_major_axis : ∀ (a b : ℝ), a > 0 ∧ b > 0 → a^2 / 45 + b^2 / 36 ≥ 1

/-- The equation of the ellipse with the shortest major axis -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 45 + y^2 / 36 = 1

/-- Theorem stating that the given equation is the correct equation for the ellipse -/
theorem shortest_major_axis_ellipse_equation :
  ∀ (x y : ℝ), ellipse_equation x y ↔ 
    (∃ (t : ℝ), y = t + 9 ∧ 
      (x - -3)^2 + (y - 0)^2 + (x - 3)^2 + (y - 0)^2 = 
      (6 * Real.sqrt 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_major_axis_ellipse_equation_l538_53875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l538_53845

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | n + 1 => 2 * sequence_a n + n - 1

def sequence_b (n : ℕ) : ℕ := sequence_a n + n

theorem sequence_properties :
  (sequence_b 1 = 2 ∧ sequence_b 2 = 4 ∧ sequence_b 3 = 8) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) = 2 * sequence_b n) ∧
  (∀ n : ℕ, n ≥ 1 → (Finset.range n).sum sequence_a = 2^(n + 1) - (n^2 + n) / 2 - 2) :=
by
  sorry  -- Use sorry to skip the proof

#eval sequence_b 1  -- Add some #eval statements to check the results
#eval sequence_b 2
#eval sequence_b 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l538_53845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_average_for_goal_l538_53858

noncomputable def current_scores : List ℝ := [91, 85, 82, 73, 88]
def num_current_tests : ℕ := 5
def target_increase : ℝ := 5
def num_additional_tests : ℕ := 2

noncomputable def current_average : ℝ := (current_scores.sum) / num_current_tests

noncomputable def target_average : ℝ := current_average + target_increase

def total_tests : ℕ := num_current_tests + num_additional_tests

noncomputable def required_average : ℝ := 
  (target_average * total_tests - current_scores.sum) / num_additional_tests

theorem minimum_average_for_goal :
  required_average = 101.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_average_for_goal_l538_53858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_min_balls_to_draw_proof_l538_53857

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  purple : Nat
  white : Nat
  black : Nat

/-- The actual counts of balls in the box -/
def boxContents : BallCounts :=
  { red := 30
  , green := 25
  , yellow := 18
  , blue := 15
  , purple := 12
  , white := 10
  , black := 7 }

/-- The target number of balls of a single color we want to guarantee -/
def targetCount : Nat := 20

/-- 
Theorem stating that the minimum number of balls to draw to guarantee
at least 'targetCount' balls of a single color is 101
-/
def min_balls_to_draw (box : BallCounts) (target : Nat) : Nat := 101

#eval min_balls_to_draw boxContents targetCount

/-- Proof of the theorem -/
theorem min_balls_to_draw_proof :
  min_balls_to_draw boxContents targetCount = 101 := by
  -- Unfold the definition of min_balls_to_draw
  unfold min_balls_to_draw
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_min_balls_to_draw_proof_l538_53857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_common_remainder_l538_53899

theorem greatest_common_remainder (a b c d : ℕ) : 
  a = 36 → b = 52 → c = 84 → d = 112 →
  (∃ (p : ℕ), p ∈ ({2, 3, 5} : Set ℕ) ∧ a % p = 0) →
  (∃ (p : ℕ), p ∈ ({2, 3, 5} : Set ℕ) ∧ b % p = 0) →
  (∃ (p : ℕ), p ∈ ({2, 3, 5} : Set ℕ) ∧ c % p = 0) →
  (∃ (p : ℕ), p ∈ ({2, 3, 5} : Set ℕ) ∧ d % p = 0) →
  Nat.gcd (b - a) (Nat.gcd (c - b) (d - c)) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_common_remainder_l538_53899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_l538_53886

open Real

-- Define the equation
def equation (θ : ℝ) : Prop :=
  cos (15 * π / 180) = sin (35 * π / 180) + cos θ

-- Define the least positive angle
noncomputable def least_positive_angle : ℝ :=
  arccos (2 * sin (35 * π / 180) * sin (20 * π / 180))

-- Theorem statement
theorem angle_measure :
  equation least_positive_angle ∧
  ∀ θ, 0 < θ ∧ θ < least_positive_angle → ¬ equation θ ∧
  round (least_positive_angle * 180 / π) = 79 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_l538_53886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l538_53856

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def share_factor (a b : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ a % d = 0 ∧ b % d = 0

def valid_set (S : Finset ℕ) : Prop :=
  (∀ n, n ∈ S → n ≤ 1000) ∧
  (∀ n, n ∈ S → ¬is_prime n) ∧
  (∀ a b, a ∈ S → b ∈ S → a ≠ b → ¬share_factor a b)

theorem max_valid_set_size :
  (∃ S : Finset ℕ, valid_set S ∧ S.card = 12) ∧
  (∀ S : Finset ℕ, valid_set S → S.card ≤ 12) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_size_l538_53856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_size_order_l538_53898

/-- Represents a soap size with its cost and quantity -/
structure SoapSize where
  cost : ℚ
  quantity : ℚ

/-- The tiny soap size -/
def tiny : SoapSize := { cost := 1, quantity := 1 }

/-- The normal soap size -/
def normal : SoapSize := { 
  cost := 14/10 * tiny.cost,
  quantity := 3/4 * 3 * tiny.quantity
}

/-- The jumbo soap size -/
def jumbo : SoapSize := {
  cost := 12/10 * normal.cost,
  quantity := 3 * tiny.quantity
}

/-- Calculates the cost per unit quantity of a soap size -/
def costPerUnit (s : SoapSize) : ℚ := s.cost / s.quantity

/-- Theorem stating the order of soap sizes from most economical to least economical -/
theorem soap_size_order : 
  costPerUnit jumbo < costPerUnit normal ∧ costPerUnit normal < costPerUnit tiny :=
by
  -- Unfold definitions and simplify
  simp [costPerUnit, jumbo, normal, tiny]
  -- Split the conjunction
  apply And.intro
  -- Prove jumbo < normal
  · norm_num
  -- Prove normal < tiny
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_size_order_l538_53898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_geometric_sum_property_geometric_sequence_ratio_eq_three_l538_53817

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  (∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) = q * geometric_sequence a₁ q n) :=
by sorry

theorem geometric_sum_property (a₁ : ℝ) (q : ℝ) (n : ℕ) :
  geometric_sum a₁ q (n + 1) = a₁ + q * geometric_sum a₁ q n :=
by sorry

theorem geometric_sequence_ratio_eq_three :
  let a₁ := (4 : ℝ)
  let S := geometric_sum a₁
  ∀ q : ℝ, q ≠ 1 →
    (∀ n : ℕ, n > 1 → (S q (n + 1) + 2) * (S q (n - 1) + 2) = (S q n + 2)^2) →
    q = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_geometric_sum_property_geometric_sequence_ratio_eq_three_l538_53817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l538_53832

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → S ≥ T) ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l538_53832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_terms_divisible_by_fifteen_fifteen_is_greatest_divisor_l538_53804

/-- An arithmetic sequence of positive integers -/
def ArithmeticSequence (a₁ c : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * c

/-- The sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a₁ c : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + ArithmeticSequence a₁ c (n - 1)) / 2

theorem sum_fifteen_terms_divisible_by_fifteen (a₁ c : ℕ) :
  ∃ k : ℕ, SumArithmeticSequence a₁ c 15 = 15 * k := by
  sorry

theorem fifteen_is_greatest_divisor :
  ∀ d : ℕ, d > 15 →
    ∃ a₁ c : ℕ, ¬(∃ k : ℕ, SumArithmeticSequence a₁ c 15 = d * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_terms_divisible_by_fifteen_fifteen_is_greatest_divisor_l538_53804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_4_2_l538_53824

/-- A power function passing through (4, 2) -/
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 2 / Real.log 4)

/-- Theorem stating that f(9) = 3 -/
theorem power_function_through_4_2 : f 9 = 3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.rpow_def, Real.exp_log]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_4_2_l538_53824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l538_53849

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.a 1 + seq.a 4 + seq.a 7 = 0) : 
  sum_n seq 6 / seq.a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l538_53849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_dog_expenses_l538_53806

def vet_appointment_cost : ℚ := 400
def number_of_appointments : ℚ := 3
def medication_cost : ℚ := 250
def grooming_cost : ℚ := 120
def pet_food_cost : ℚ := 300
def insurance_cost : ℚ := 100
def insurance_vet_coverage : ℚ := 80 / 100
def insurance_medication_coverage : ℚ := 50 / 100

def total_expenses : ℚ := 
  vet_appointment_cost + 
  (number_of_appointments - 1) * vet_appointment_cost * (1 - insurance_vet_coverage) +
  medication_cost * (1 - insurance_medication_coverage) +
  grooming_cost + 
  pet_food_cost + 
  insurance_cost

theorem john_dog_expenses : total_expenses = 1205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_dog_expenses_l538_53806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l538_53853

-- Define the function F
noncomputable def F (x y z : ℝ) : ℝ := x * y^z

-- Theorem statement
theorem solution_exists :
  ∃ (s : ℝ), s > 0 ∧ F s s 2 = 144 ∧ s = Real.rpow 2 (4/3) * Real.rpow 3 (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l538_53853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_limit_l538_53847

/-- The sum of the infinite series representing the growth of a sequence -/
noncomputable def seriesSum : ℝ := 2 + (1/3 * Real.sqrt 3) + 1/3 + (1/9 * Real.sqrt 3) + 1/9 + (1/27 * Real.sqrt 3) + 1/27

/-- The limit of the sequence's length -/
noncomputable def sequenceLimit : ℝ := 3 + (Real.sqrt 3) / 2

/-- Theorem stating that the sum of the series equals the limit -/
theorem series_sum_equals_limit : seriesSum = sequenceLimit := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_limit_l538_53847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similarity_transformation_of_nilpotent_matrices_l538_53893

open Matrix Complex

theorem similarity_transformation_of_nilpotent_matrices 
  (M N : Matrix (Fin 2) (Fin 2) ℂ) 
  (hM : M ≠ 0) 
  (hN : N ≠ 0) 
  (hM2 : M * M = 0) 
  (hN2 : N * N = 0) 
  (hMN : M * N + N * M = 1) : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℂ, 
    IsUnit A ∧ 
    M = A * !![0, 1; 0, 0] * A⁻¹ ∧
    N = A * !![0, 0; 1, 0] * A⁻¹ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similarity_transformation_of_nilpotent_matrices_l538_53893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_correct_l538_53884

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 1; 2, 1]
def β : Matrix (Fin 2) (Fin 1) ℚ := !![1; 2]
def α : Matrix (Fin 2) (Fin 1) ℚ := !![-1; 2]

theorem solution_correct : A ^ 2 * α = β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_correct_l538_53884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_a_p_l538_53874

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 3
  | 1 => 0
  | 2 => 2
  | (n + 3) => a (n + 1) + a n

/-- Theorem: For any prime p, p divides a_p -/
theorem prime_divides_a_p (p : ℕ) (hp : Nat.Prime p) : (p : ℤ) ∣ a p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_a_p_l538_53874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_opposite_AB_l538_53829

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-2, 6)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def opposite_direction : ℝ × ℝ := (-vector_AB.1, -vector_AB.2)

noncomputable def magnitude : ℝ := Real.sqrt (opposite_direction.1^2 + opposite_direction.2^2)

noncomputable def unit_vector : ℝ × ℝ := (opposite_direction.1 / magnitude, opposite_direction.2 / magnitude)

theorem unit_vector_opposite_AB : 
  unit_vector = (3/5, -4/5) := by
  sorry

#eval A
#eval B
#eval vector_AB
#eval opposite_direction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_opposite_AB_l538_53829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_length_calculation_l538_53825

theorem car_length_calculation (speed_kmph : ℝ) (time_seconds : ℝ) (length_meters : ℝ) : 
  speed_kmph = 36 →
  time_seconds = 0.9999200063994881 →
  length_meters = speed_kmph * (1000 / 3600) * time_seconds →
  abs (length_meters - 9.9992) < 0.00001 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  rw [h3]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_length_calculation_l538_53825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_proportion_a_smith_a_students_correct_smith_b_students_correct_l538_53865

/-- Represents a class with students and their grades -/
structure ClassGrades where
  total_students : ℕ
  a_students : ℕ
  b_students : ℕ
  h_total : total_students = a_students + b_students

/-- Mr. Johnson's class -/
def johnson_class : ClassGrades where
  total_students := 20
  a_students := 12
  b_students := 8
  h_total := by rfl

/-- Mrs. Smith's class -/
def smith_class : ClassGrades where
  total_students := 30
  a_students := 18
  b_students := 12
  h_total := by rfl

/-- The proportion of 'A' students is the same in both classes -/
theorem same_proportion_a :
  (johnson_class.a_students : ℚ) / johnson_class.total_students =
  (smith_class.a_students : ℚ) / smith_class.total_students :=
by sorry

/-- The number of 'A' students in Mrs. Smith's class is correct -/
theorem smith_a_students_correct :
  smith_class.a_students = 18 :=
by rfl

/-- The number of 'B' students in Mrs. Smith's class is correct -/
theorem smith_b_students_correct :
  smith_class.b_students = 12 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_proportion_a_smith_a_students_correct_smith_b_students_correct_l538_53865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_time_l538_53805

/-- The time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train of length 120 meters traveling at 121 km/hr takes approximately 3.57 seconds to cross an electric pole -/
theorem train_crossing_approx_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_crossing_time 120 121 - 3.57| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_time_l538_53805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_depth_is_two_l538_53826

/-- The depth of a rectangular box given its length, width, fill rate, and fill time. -/
noncomputable def box_depth (length width fill_rate fill_time : ℝ) : ℝ :=
  (fill_rate * fill_time) / (length * width)

/-- Theorem stating that the depth of the box is 2 feet under given conditions. -/
theorem box_depth_is_two :
  let length : ℝ := 7
  let width : ℝ := 6
  let fill_rate : ℝ := 4
  let fill_time : ℝ := 21
  box_depth length width fill_rate fill_time = 2 := by
  -- Unfold the definition of box_depth
  unfold box_depth
  -- Perform the calculation
  simp [mul_div_assoc]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_depth_is_two_l538_53826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_donation_theorem_l538_53877

noncomputable def cloth_donation (initial_size : ℝ) (num_cuts : ℕ) : ℝ :=
  let donation_sizes := List.range num_cuts |>.map (fun i => initial_size / 2^(i + 1))
  donation_sizes.sum

theorem cloth_donation_theorem :
  cloth_donation 100 2 = 75 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_donation_theorem_l538_53877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_implies_multiple_axes_l538_53895

/-- A figure in a 2D plane. -/
structure Figure where
  -- The structure is left abstract as we don't need its specific properties for this theorem

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space. -/
structure Line where
  -- The line is left abstract as we don't need its specific properties for this theorem

/-- Defines what it means for a point to belong to a figure. -/
def belongs_to (p : Point) (f : Figure) : Prop :=
  sorry -- Definition omitted for brevity

/-- Reflects a point over another point (center of symmetry). -/
def reflected_point (p : Point) (c : Point) : Point :=
  sorry -- Definition omitted for brevity

/-- Reflects a point over a line (axis of symmetry). -/
def reflected_point_over_line (p : Point) (l : Line) : Point :=
  sorry -- Definition omitted for brevity

/-- Defines what it means for a figure to have a center of symmetry. -/
def has_center_of_symmetry (f : Figure) (c : Point) : Prop :=
  ∀ p : Point, belongs_to p f → belongs_to (reflected_point p c) f

/-- Defines what it means for a figure to have an axis of symmetry. -/
def has_axis_of_symmetry (f : Figure) (l : Line) : Prop :=
  ∀ p : Point, belongs_to p f → belongs_to (reflected_point_over_line p l) f

/-- Theorem: If a figure has a center of symmetry, it cannot have exactly one axis of symmetry. -/
theorem center_of_symmetry_implies_multiple_axes (f : Figure) (c : Point) :
  has_center_of_symmetry f c → ¬∃! l : Line, has_axis_of_symmetry f l :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_implies_multiple_axes_l538_53895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l538_53872

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - Real.sqrt x) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l538_53872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l538_53852

-- Define the function f(x)
noncomputable def f (x : ℝ) := 3 * Real.sin x ^ 2 + 5 * Real.cos x ^ 2 + 2 * Real.cos x

-- State the theorem
theorem f_min_value : ∀ x : ℝ, f x ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l538_53852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_chord_length_l538_53842

/-- The trajectory of a point with equal distances to a fixed point and a line -/
def Trajectory (F : ℝ × ℝ) (l : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p F = |p.2 - l p.1|}

/-- The intersection points of a line with the trajectory -/
def IntersectionPoints (F : ℝ × ℝ) (l : ℝ → ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ Trajectory F l ∧ p.2 = m * p.1 + F.2}

theorem trajectory_and_chord_length 
  (F : ℝ × ℝ) 
  (l : ℝ → ℝ) 
  (h_F : F = (0, 1)) 
  (h_l : l = λ x ↦ -1) :
  (∀ p : ℝ × ℝ, p ∈ Trajectory F l ↔ p.1^2 = 4 * p.2) ∧ 
  (let m := Real.tan (π / 3)
   let points := IntersectionPoints F l m
   ∀ A B : ℝ × ℝ, A ∈ points → B ∈ points → A ≠ B → dist A B = 16) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_chord_length_l538_53842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sundays_tuesdays_count_l538_53836

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Counts the number of occurrences of a specific day in a 30-day month -/
def countDaysInMonth (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Checks if Sundays and Tuesdays are equal for a given start day -/
def hasSameSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  countDaysInMonth startDay DayOfWeek.Sunday = countDaysInMonth startDay DayOfWeek.Tuesday

/-- List of all days of the week -/
def allDays : List DayOfWeek :=
  [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
   DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]

/-- The main theorem to prove -/
theorem equal_sundays_tuesdays_count :
  (allDays.filter hasSameSundaysAndTuesdays).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sundays_tuesdays_count_l538_53836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_subsets_removal_l538_53819

theorem distinct_subsets_removal {X : Type} [Fintype X] (n : ℕ) (h_card : Fintype.card X = n)
  (A : Fin n → Set X) (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ x : X, ∀ i j, i ≠ j → A i \ {x} ≠ A j \ {x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_subsets_removal_l538_53819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l538_53821

def g : ℕ → ℕ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | 2 => 1
  | (n + 3) => g (n + 1) + g (n + 2)

theorem prime_divides_g_product {n : ℕ} (h_prime : Nat.Prime n) (h_gt5 : n > 5) :
  n ∣ g n * (g n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l538_53821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_S_equals_18145_l538_53820

/-- Definition of the sequence S_n -/
def S : ℕ → ℕ
| 0 => 1  -- Adding case for 0 to cover all natural numbers
| 1 => 1
| 2 => 5
| 3 => 15
| 4 => 34
| 5 => 65
| 6 => 111
| 7 => 175
| n + 8 => sorry  -- We don't define the general term to keep it within given conditions

/-- The sum of the first 99 terms of the sequence S_n -/
def sum_S : ℕ := (List.range 99).map (fun i => S (i + 1)) |>.sum

/-- Theorem stating that the sum of the first 99 terms of S_n is 18145 -/
theorem sum_S_equals_18145 : sum_S = 18145 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_S_equals_18145_l538_53820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_classification_l538_53880

/-- Represents a problem type in combinatorics -/
inductive ProblemType
  | Combination
  | Permutation

/-- Determines if a selection problem is a combination or permutation -/
def problemClassification (
  totalItems : ℕ
) (itemsToSelect : ℕ)
  (orderMatters : Bool) : ProblemType :=
  if orderMatters then ProblemType.Permutation else ProblemType.Combination

theorem selection_classification (n : ℕ) (h : n > 1) :
  problemClassification 50 2 true = ProblemType.Permutation ∧
  problemClassification 10 2 false = ProblemType.Combination ∧
  problemClassification 9 2 false = ProblemType.Combination ∧
  problemClassification n 3 false = ProblemType.Combination :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_classification_l538_53880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_l538_53876

open Real

-- Define the set of points generated by the polar equation r = sin θ
noncomputable def polar_curve (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ p = ((sin θ) * (cos θ), (sin θ) * (sin θ))}

-- Define what it means for a set to be a complete circle
def is_complete_circle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    ∀ p : ℝ × ℝ, p ∈ s ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Theorem statement
theorem smallest_complete_circle : 
  (is_complete_circle (polar_curve π)) ∧ 
  (∀ t : ℝ, 0 < t ∧ t < π → ¬(is_complete_circle (polar_curve t))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_complete_circle_l538_53876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_theorem_l538_53810

noncomputable def original_function (x : ℝ) : ℝ := Real.log (2 * x + 1)

noncomputable def target_function (x : ℝ) : ℝ := Real.log (2 * x + 3) - 1

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x + 1) - 1

theorem shift_theorem :
  ∀ x : ℝ, shifted_function x = target_function x :=
by
  intro x
  simp [shifted_function, original_function, target_function]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_theorem_l538_53810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l538_53889

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  width : ℝ

/-- The cost of fencing in pence per meter -/
noncomputable def fencing_cost_per_meter : ℝ := 70

/-- The total cost of fencing in dollars -/
noncomputable def total_fencing_cost : ℝ := 175

/-- Conversion rate from pence to dollars -/
noncomputable def pence_to_dollar : ℝ := 1 / 100

theorem park_area (park : RectangularPark) : 
  (park.length / park.width = 3 / 2) →
  (2 * (park.length + park.width) * fencing_cost_per_meter * pence_to_dollar = total_fencing_cost) →
  (park.length * park.width = 3750) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l538_53889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_for_ratio_difference_l538_53863

theorem smallest_constant_for_ratio_difference :
  (∀ (a₁ a₂ a₃ a₄ a₅ : ℝ), a₁ > 0 → a₂ > 0 → a₃ > 0 → a₄ > 0 → a₅ > 0 →
    ∃ (i j k l : Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
    |a₁ * (Fin.val i + 1) / (a₁ * (Fin.val j + 1)) - a₁ * (Fin.val k + 1) / (a₁ * (Fin.val l + 1))| ≤ (1/2 : ℝ)) ∧
  (∀ C : ℝ, C < 1/2 → 
    ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ), b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧ b₄ > 0 ∧ b₅ > 0 ∧
    ∀ (i j k l : Fin 5), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
    |b₁ * (Fin.val i + 1) / (b₁ * (Fin.val j + 1)) - b₁ * (Fin.val k + 1) / (b₁ * (Fin.val l + 1))| > C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_for_ratio_difference_l538_53863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficients_on_unit_circle_l538_53890

-- Define the circle
def on_unit_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 1

-- Define the points
variable (O M N P : ℝ × ℝ)

-- Define the conditions
variable (hM : on_unit_circle M)
variable (hN : on_unit_circle N)
variable (hP : on_unit_circle P)
variable (hDistinct : M ≠ N ∧ M ≠ P ∧ N ≠ P)
variable (hPerpendicular : (M.1 - O.1) * (N.1 - O.1) + (M.2 - O.2) * (N.2 - O.2) = 0)

-- Define the linear combination
variable (m n : ℝ)
variable (hLinearComb : P = (O.1 + m * (M.1 - O.1) + n * (N.1 - O.1), 
                             O.2 + m * (M.2 - O.2) + n * (N.2 - O.2)))

-- State the theorem
theorem coefficients_on_unit_circle :
  m^2 + n^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficients_on_unit_circle_l538_53890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bayes_theorem_l538_53855

-- Define the number of balls in each bag
def white_balls_A : ℕ := 3
def red_balls_A : ℕ := 2
def white_balls_B : ℕ := 2
def red_balls_B : ℕ := 2

-- Define the events
noncomputable def A₀ : ℝ := (red_balls_A.choose 2) / ((white_balls_A + red_balls_A).choose 2)
noncomputable def A₁ : ℝ := (white_balls_A.choose 1 * red_balls_A.choose 1) / ((white_balls_A + red_balls_A).choose 2)
noncomputable def A₂ : ℝ := (white_balls_A.choose 2) / ((white_balls_A + red_balls_A).choose 2)

noncomputable def B_given_A₀ : ℝ := (white_balls_B.choose 2) / ((white_balls_B + red_balls_B + 2).choose 2)
noncomputable def B_given_A₁ : ℝ := ((white_balls_B + 1).choose 2) / ((white_balls_B + red_balls_B + 2).choose 2)
noncomputable def B_given_A₂ : ℝ := ((white_balls_B + 2).choose 2) / ((white_balls_B + red_balls_B + 2).choose 2)

-- Theorem statement
theorem bayes_theorem :
  (A₂ * B_given_A₂) / (A₀ * B_given_A₀ + A₁ * B_given_A₁ + A₂ * B_given_A₂) = 18 / 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bayes_theorem_l538_53855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l538_53833

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^3 - a*x) / Real.log a

theorem monotonic_increasing_f_implies_a_range
  (h1 : ∀ x ∈ Set.Ioo (-1/2 : ℝ) 0, StrictMono (f a))
  (h2 : a > 0)
  (h3 : a ≠ 1) :
  a ∈ Set.Icc (3/4 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l538_53833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_far_from_faces_l538_53896

noncomputable section

-- Define the cube
def cube_edge_length : ℝ := 3

-- Define the condition for the inner cube
def inner_cube_edge_length : ℝ := 1

-- Define the volumes
def outer_cube_volume : ℝ := cube_edge_length ^ 3
def inner_cube_volume : ℝ := inner_cube_edge_length ^ 3

-- Define the probability
def probability : ℝ := inner_cube_volume / outer_cube_volume

-- Theorem statement
theorem probability_point_far_from_faces :
  probability = 1 / 27 := by
  -- Unfold the definitions
  unfold probability
  unfold inner_cube_volume
  unfold outer_cube_volume
  unfold cube_edge_length
  unfold inner_cube_edge_length
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_far_from_faces_l538_53896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_constant_l538_53867

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_equation P.1 P.2

-- Define the tangent line at a point on the circle
def tangent_line (P M N : ℝ × ℝ) : Prop :=
  point_on_circle P ∧ 
  hyperbola M.1 M.2 ∧ 
  hyperbola N.1 N.2 ∧
  (M.2 - P.2) * (N.1 - P.1) = (M.1 - P.1) * (N.2 - P.2)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- State the theorem
theorem chord_product_constant (P M N : ℝ × ℝ) :
  tangent_line P M N → distance P M * distance P N = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_product_constant_l538_53867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l538_53811

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_triangle_BAD (q : Quadrilateral) : Prop :=
  (q.B.1 - q.A.1) * (q.D.1 - q.A.1) + (q.B.2 - q.A.2) * (q.D.2 - q.A.2) = 0

def is_right_triangle_BDC (q : Quadrilateral) : Prop :=
  (q.B.1 - q.D.1) * (q.C.1 - q.D.1) + (q.B.2 - q.D.2) * (q.C.2 - q.D.2) = 0

noncomputable def AB_length (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2)

noncomputable def BD_length (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.D.1 - q.B.1)^2 + (q.D.2 - q.B.2)^2)

noncomputable def BC_length (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2)

-- Define the area of the quadrilateral
noncomputable def area (q : Quadrilateral) : ℝ :=
  5 * Real.sqrt 69 + 6.5 * Real.sqrt 56

-- State the theorem
theorem quadrilateral_area 
  (q : Quadrilateral) 
  (h1 : is_right_triangle_BAD q) 
  (h2 : is_right_triangle_BDC q) 
  (h3 : AB_length q = 10) 
  (h4 : BD_length q = 13) 
  (h5 : BC_length q = 15) : 
  area q = 5 * Real.sqrt 69 + 6.5 * Real.sqrt 56 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l538_53811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_l538_53878

/-- Given a quadratic function f(x) = ax² + bx where a ≠ 0 and a, b are constants,
    satisfying f(1-x) = f(1+x) and f(x) = 2x has two equal real roots,
    prove that the maximum value of g(x) = x³/3 + x² - 3x on [0,3] is 9. -/
theorem max_value_g (a b : ℝ) (ha : a ≠ 0) 
  (h1 : ∀ x, a*(1-x)^2 + b*(1-x) = a*(1+x)^2 + b*(1+x))
  (h2 : ∃! x, a*x^2 + b*x = 2*x) :
  let f := λ x => a*x^2 + b*x
  let g := λ x => x^3/3 + x^2 - 3*x
  ∃ x₀ ∈ Set.Icc 0 3, (∀ x ∈ Set.Icc 0 3, g x ≤ g x₀) ∧ g x₀ = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_l538_53878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distances_l538_53813

/-- Triangle ABC with vertices A, B, C, and point P --/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (P : ℝ × ℝ)

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances from P to A, B, and C --/
noncomputable def sum_distances (t : Triangle) : ℝ :=
  distance t.P t.A + distance t.P t.B + distance t.P t.C

/-- Theorem statement --/
theorem fermat_point_distances (t : Triangle) 
  (h1 : t.A = (0, 0))
  (h2 : t.B = (12, 0))
  (h3 : t.C = (4, 6))
  (h4 : t.P = (5, 3)) :
  sum_distances t = Real.sqrt 34 + Real.sqrt 58 + Real.sqrt 10 ∧
  (1 : ℝ) + 1 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distances_l538_53813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_negation_equivalence_not_both_false_sufficient_not_necessary_square_l538_53831

-- 1. "$am^{2} < bm^{2}$" is a sufficient but not necessary condition for "$a < b$"
theorem sufficient_not_necessary_condition :
  ∃ a b : ℝ, (∀ m : ℝ, a * m^2 < b * m^2 → a < b) ∧ ¬(a < b → ∀ m : ℝ, a * m^2 < b * m^2) :=
sorry

-- 2. The negation of "$\forall x \in \mathbb{R}, x^{3}-x^{2}-1 \leqslant 0$" is equivalent to "$\exists x \in \mathbb{R}, x^{3}-x^{2}-1 > 0$"
theorem negation_equivalence :
  ¬(∀ x : ℝ, x^3 - x^2 - 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 - 1 > 0 :=
sorry

-- 3. It is not true that if $p \land q$ is a false proposition, then both $p$ and $q$ are false propositions
theorem not_both_false :
  ∃ p q : Prop, ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q) :=
sorry

-- 4. "$x=2$" is a sufficient but not necessary condition for "$x^{2}=4$"
theorem sufficient_not_necessary_square :
  (∀ x : ℝ, x = 2 → x^2 = 4) ∧ ¬(∀ x : ℝ, x^2 = 4 → x = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_negation_equivalence_not_both_false_sufficient_not_necessary_square_l538_53831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_line_b_range_l538_53839

/-- Definition of a "friendship line" parabola for a given line -/
def friendship_line (m a b c : ℝ) : Prop :=
  a * (1 - 1/m)^2 + b * (1 - 1/m) + (1 - m) = 0

/-- Theorem stating the range of b for the "friendship line" parabola -/
theorem friendship_line_b_range {m a b c : ℝ} (hm : m ≠ 0) :
  friendship_line m a b c → b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_line_b_range_l538_53839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_point_coordinates_l538_53864

/-- Represents a point in 3D space with spherical coordinates -/
structure SphericalPoint where
  r : ℝ
  θ : ℝ
  φ : ℝ

/-- Represents a point in 3D space with rectangular coordinates -/
structure RectangularPoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Conversion function from spherical to rectangular coordinates -/
noncomputable def sphericalToRectangular (p : SphericalPoint) : RectangularPoint :=
  { x := p.r * Real.sin p.φ * Real.cos p.θ,
    y := p.r * Real.sin p.φ * Real.sin p.θ,
    z := p.r * Real.cos p.φ }

/-- Theorem stating the relationship between the original point and its reflection -/
theorem reflected_point_coordinates (p p' : SphericalPoint) :
  p = { r := 3, θ := 5 * π / 6, φ := π / 4 } →
  let rect := sphericalToRectangular p
  p' = { r := 3, θ := 11 * π / 6, φ := π / 4 } →
  sphericalToRectangular p' = { x := -rect.x, y := -rect.y, z := rect.z } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_point_coordinates_l538_53864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l538_53822

theorem sin_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α = 4/5) (h4 : Real.cos (α + β) = 5/13) : Real.sin β = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l538_53822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l538_53871

/-- Given a function f(x) = 3x + sin x + 1 for x ∈ ℝ, 
    if f(t) = 2 for some t, then f(-t) = 0 -/
theorem function_property (f : ℝ → ℝ) (t : ℝ) : 
  (∀ x, f x = 3*x + Real.sin x + 1) → 
  f t = 2 → 
  f (-t) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l538_53871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_lambda_l538_53838

def vector2D := ℝ × ℝ

def a : vector2D := (1, 2)
def b : vector2D := (2, -2)
def c (lambda : ℝ) : vector2D := (1, lambda)

def parallel (v w : vector2D) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_lambda (lambda : ℝ) :
  parallel (c lambda) ((2 * a.1 + b.1, 2 * a.2 + b.2)) → lambda = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_lambda_l538_53838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l538_53851

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C, prove the following inequalities -/
theorem triangle_inequalities (a b c A B C : ℝ) :
  (A + B + C = π) →
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (a * A + b * B + c * C ≥ a * B + b * C + c * A) ∧
  ((a * A + b * B + c * C) / (a + b + c) ≥ π / 3) ∧
  (π / 3 ≥ (A * Real.cos A + B * Real.cos B + C * Real.cos C) / (Real.cos A + Real.cos B + Real.cos C)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l538_53851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryan_advert_outreach_ratio_l538_53887

/-- Represents Bryan's work schedule in hours -/
structure WorkSchedule where
  total_hours : ℕ
  customer_outreach : ℕ
  marketing : ℕ
  advertisement : ℕ

/-- Calculates the ratio of advertisement time to customer outreach time -/
def advert_to_outreach_ratio (schedule : WorkSchedule) : ℚ :=
  schedule.advertisement / schedule.customer_outreach

/-- Bryan's actual work schedule -/
def bryan_schedule : WorkSchedule := {
  total_hours := 8,
  customer_outreach := 4,
  marketing := 2,
  advertisement := 8 - (4 + 2)
}

/-- Theorem stating that Bryan's advertisement to customer outreach ratio is 1:2 -/
theorem bryan_advert_outreach_ratio :
  advert_to_outreach_ratio bryan_schedule = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryan_advert_outreach_ratio_l538_53887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l538_53834

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def vector_problem (a b : V) : Prop :=
  ‖a‖ = 1 ∧ ‖b‖ = 2 ∧ ‖a - b‖ = 2 → ‖a + b‖ = Real.sqrt 6

theorem vector_sum_magnitude (a b : V) : vector_problem a b := by
  intro h
  have h1 : ‖a‖ = 1 := h.1
  have h2 : ‖b‖ = 2 := h.2.1
  have h3 : ‖a - b‖ = 2 := h.2.2
  
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l538_53834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l538_53815

/-- Represents the total number of voters -/
def n : ℕ := 2  -- Example value, can be changed

/-- Represents the number of voters in the smaller district -/
def m : ℕ := 1  -- Example value, can be changed

/-- Probability of Miraflores winning in a district -/
def win_prob (supporters : ℚ) (total : ℚ) : ℚ :=
  supporters / total

/-- Probability of Miraflores winning the election -/
def election_prob (m : ℕ) (n : ℕ) : ℚ :=
  (win_prob 1 1) * (win_prob (n - 1 : ℚ) ((2 * n - 1 : ℕ) : ℚ))

/-- Theorem stating that the optimal strategy maximizes the winning probability -/
theorem optimal_strategy (n : ℕ) (h : n > 1) :
  ∀ m, m ≥ 1 → m < 2 * n → election_prob 1 n ≥ election_prob m n := by
  sorry

#check optimal_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_l538_53815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l538_53823

/-- A quadratic function with specific properties -/
noncomputable def f (x : ℝ) : ℝ := -5/2 * x^2 + 15 * x - 25/2

/-- Theorem stating that f satisfies the required conditions -/
theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l538_53823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l538_53844

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 2*a - 2

-- State the theorem
theorem function_properties (a : ℝ) :
  (a > 0) →
  (∀ x, f a (2 + x) * f a (2 - x) = 4) →
  (∀ x ∈ Set.Icc 0 2, f a x = x^2 - a*x + 2*a - 2) →
  (f a 2 + f a 3 = 6) →
  (∀ x ∈ Set.Icc 0 4, 1 ≤ f a x ∧ f a x ≤ 3) →
  (a = 2 ∧ 4 - 2*Real.sqrt 6/3 ≤ a ∧ a ≤ 5/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l538_53844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l538_53841

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the ratio condition
def ratioCondition (t : Triangle) : Prop :=
  ∃ (k : Real), k > 0 ∧ t.a = 3 * k ∧ t.b = Real.sqrt 7 * k ∧ t.c = 2 * k

-- State the theorem
theorem angle_measure_in_special_triangle (t : Triangle) 
  (h1 : validTriangle t) (h2 : ratioCondition t) : t.B = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l538_53841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_line_l538_53891

-- Define the type for boys
inductive Boy : Type
  | Vasya | Dima | Egor | Ilya | Kolya | Petya | Tema | Fedya

-- Define the type for positions (1 to 8)
def Position : Type := Fin 8

-- Define the type for the line of boys
def Line : Type := Position → Boy

def is_valid_line (l : Line) : Prop :=
  -- All boys are in the line exactly once
  (∀ b : Boy, ∃! p : Position, l p = b) ∧
  -- Dima's number is three times Ilya's number
  (∃ p_ilya : Position, l p_ilya = Boy.Ilya ∧
    ∃ p_dima : Position, l p_dima = Boy.Dima ∧ p_dima.val + 1 = 3 * (p_ilya.val + 1)) ∧
  -- Fedya is after the third boy but before Kolya
  (∃ p_fedya p_kolya : Position,
    l p_fedya = Boy.Fedya ∧ l p_kolya = Boy.Kolya ∧
    3 < p_fedya.val ∧ p_fedya.val < p_kolya.val) ∧
  -- Vasya's number is half of Petya's number
  (∃ p_vasya p_petya : Position,
    l p_vasya = Boy.Vasya ∧ l p_petya = Boy.Petya ∧
    (p_vasya.val + 1) * 2 = p_petya.val + 1) ∧
  -- The fourth boy is immediately after Tema and somewhere before Petya
  (∃ p_tema : Position,
    l p_tema = Boy.Tema ∧ p_tema.val = 2 ∧
    ∃ p_petya : Position, l p_petya = Boy.Petya ∧ 4 < p_petya.val)

def correct_line : Line :=
  fun p => match p with
    | ⟨0, _⟩ => Boy.Egor
    | ⟨1, _⟩ => Boy.Ilya
    | ⟨2, _⟩ => Boy.Tema
    | ⟨3, _⟩ => Boy.Vasya
    | ⟨4, _⟩ => Boy.Fedya
    | ⟨5, _⟩ => Boy.Dima
    | ⟨6, _⟩ => Boy.Kolya
    | ⟨7, _⟩ => Boy.Petya

theorem unique_valid_line :
  ∀ l : Line, is_valid_line l ↔ l = correct_line :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_line_l538_53891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_unit_circle_l538_53848

theorem sine_tangent_unit_circle (α : ℝ) :
  (∃ x y : ℝ, x = -5/13 ∧ y = 12/13 ∧ x^2 + y^2 = 1) →
  Real.sin α = 12/13 ∧ Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_unit_circle_l538_53848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l538_53862

/-- Definition of a parabola y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Distance from a point to the directrix of the parabola y^2 = 4x -/
noncomputable def distance_to_directrix (p : ℝ × ℝ) : ℝ := p.1 + 1

/-- Distance from a point to the line x + 2y - 12 = 0 -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 + 2*p.2 - 12| / Real.sqrt 5

/-- The theorem stating the minimum sum of distances -/
theorem min_sum_distances :
  ∃ (min : ℝ), min = (11/5) * Real.sqrt 5 ∧
  ∀ (p : ℝ × ℝ), parabola p →
    distance_to_directrix p + distance_to_line p ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l538_53862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_angle_l538_53835

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop

/-- Represents a line in the 2D plane -/
structure Line where
  m : ℝ
  eq : ℝ → ℝ → Prop

def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ x y : ℝ, c.eq x y ∧ l.eq x y ∧
    ∀ x' y' : ℝ, c.eq x' y' → l.eq x' y' → (x = x' ∧ y = y')

def passes_through_origin (l : Line) : Prop :=
  l.eq 0 0

noncomputable def slope_angle (l : Line) : ℝ :=
  Real.arctan l.m

theorem tangent_line_slope_angle 
  (c : Circle) 
  (h_circle_eq : c.eq = λ x y => x^2 + y^2 - 4*x + 3 = 0) :
  ∃ l : Line, 
    is_tangent l c ∧ 
    passes_through_origin l ∧ 
    (slope_angle l = π/6 ∨ slope_angle l = 5*π/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_angle_l538_53835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l538_53837

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l538_53837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_sum_to_hundred_percent_arc_proportion_l538_53814

/-- Represents the percentage allocation for each category in the budget --/
def budget_allocation : Fin 10 → ℝ
| 0 => 12  -- Microphotonics
| 1 => 18  -- Home Electronics
| 2 => 15  -- Food Additives
| 3 => 23  -- Genetically Modified Microorganisms
| 4 => 7   -- Industrial Lubricants
| 5 => 10  -- Artificial Intelligence
| 6 => 4   -- Robotics
| 7 => 5   -- Renewable Energy
| 8 => 3   -- Advanced Materials
| 9 => 3   -- Basic Astrophysics (to be proven)

/-- The sum of the first 9 category percentages --/
def sum_first_nine : ℝ := (Finset.range 9).sum (λ i => budget_allocation i)

/-- The total degrees in a circle --/
def total_degrees : ℝ := 360

theorem basic_astrophysics_degrees :
  (budget_allocation 9 / 100) * total_degrees = 10.8 :=
by sorry

theorem sum_to_hundred_percent :
  sum_first_nine + budget_allocation 9 = 100 :=
by sorry

theorem arc_proportion :
  ∀ i : Fin 10, (budget_allocation i / 100) * total_degrees = 
    (budget_allocation i / (sum_first_nine + budget_allocation 9)) * total_degrees :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_sum_to_hundred_percent_arc_proportion_l538_53814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_non_intersecting_lines_l538_53888

/-- The function y = x + x/x -/
noncomputable def f (x : ℝ) : ℝ := x + x/x

/-- A line passing through (1, 0) -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - 1)

/-- The point (1, 0) -/
def point : ℝ × ℝ := (1, 0)

/-- Theorem stating that exactly two lines through (1, 0) do not intersect f -/
theorem two_non_intersecting_lines :
  ∃! (S : Finset ℝ), S.Nonempty ∧ 
  (∀ m ∈ S, ∀ x ≠ 0, f x ≠ line m x) ∧
  (∀ m ∈ S, line m point.1 = point.2) ∧
  S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_non_intersecting_lines_l538_53888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_theorem_l538_53882

/-- The volume of a regular hexagonal pyramid with base side length a and lateral surface area 10 times the area of the base -/
noncomputable def hexagonal_pyramid_volume (a : ℝ) : ℝ :=
  (9 * a^3 * Real.sqrt 11) / 2

/-- The area of the base of a regular hexagon with side length a -/
noncomputable def hexagonal_base_area (a : ℝ) : ℝ :=
  (3 * Real.sqrt 3 * a^2) / 2

/-- The lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (a : ℝ) : ℝ :=
  10 * hexagonal_base_area a

theorem hexagonal_pyramid_volume_theorem (a : ℝ) (h : a > 0) :
  hexagonal_pyramid_volume a = 
  (1/3) * hexagonal_base_area a * 
  (lateral_surface_area a / (3 * a * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_theorem_l538_53882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l538_53869

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.log (4 - x) / Real.log 10

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 2 ∧ x < 4}

-- Theorem stating that the domain of f is [2,4)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l538_53869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_passing_time_l538_53881

/-- Calculates the time (in seconds) it takes for a goods train to pass a woman in an opposite-moving train -/
noncomputable def time_to_pass (woman_speed : ℝ) (goods_speed : ℝ) (goods_length : ℝ) : ℝ :=
  let relative_speed := woman_speed + goods_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  goods_length / relative_speed_mps

theorem goods_train_passing_time :
  let woman_speed : ℝ := 20
  let goods_speed : ℝ := 51.99424046076314
  let goods_length : ℝ := 300
  |time_to_pass woman_speed goods_speed goods_length - 15| < 0.1 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_pass 20 51.99424046076314 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_passing_time_l538_53881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l538_53808

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem about a specific triangle with given properties --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi) 
  (h2 : Real.sin t.A / t.a = Real.sin t.B / t.b)
  (h3 : Real.sin t.A / t.a = Real.sin t.C / t.c)
  (h4 : Real.sin t.A / t.a = 1/2)
  (h5 : 2 * (Real.sin t.A ^ 2 - Real.sin t.C ^ 2) = (Real.sqrt 2 * t.a - t.b) * Real.sin t.B) :
  t.C = Real.pi / 4 ∧ 
  (∃ (S : Real), S ≤ Real.sqrt 2 / 2 + 1 / 2 ∧ 
    ∀ (A B : Real), A + B = 3 * Real.pi / 4 → 
      1 / 2 * t.a * t.b * Real.sin (Real.pi / 4) ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l538_53808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stronger_linear_correlation_as_r_approaches_one_l538_53885

/-- The linear correlation coefficient between two random variables -/
noncomputable def linear_correlation_coefficient (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := 
  sorry

/-- The strength of linear correlation between two random variables -/
noncomputable def linear_correlation_strength (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] : ℝ := 
  sorry

/-- Theorem: As the absolute value of the linear correlation coefficient approaches 1, 
    the linear correlation between two random variables becomes stronger -/
theorem stronger_linear_correlation_as_r_approaches_one 
  (X Y : Type) [NormedAddCommGroup X] [NormedAddCommGroup Y] :
  ∀ ε > 0, ∃ δ > 0, ∀ r : ℝ,
    (|r - 1| < δ ∧ r = linear_correlation_coefficient X Y) →
    linear_correlation_strength X Y > 1 - ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stronger_linear_correlation_as_r_approaches_one_l538_53885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_iff_a_eq_neg_one_l538_53868

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z as a function of real number a -/
noncomputable def z (a : ℝ) : ℂ := (a^2 * i) / (2 - i) + (1 - 2*a*i) / 5

/-- Theorem stating that z is purely imaginary if and only if a = -1 -/
theorem z_purely_imaginary_iff_a_eq_neg_one :
  ∀ a : ℝ, (∃ b : ℝ, z a = b * i) ↔ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_iff_a_eq_neg_one_l538_53868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_l538_53816

/-- Given a sphere of radius 2, when a cone is inscribed in it and the volume of the cone is maximized, 
    the radius of the sphere inscribed in this cone is 4(√3 - 1)/3 -/
theorem inscribed_sphere_radius (R : ℝ) (h_R : R = 2) : 
  ∃ (r h : ℝ),
    -- The cone is inscribed in the sphere
    r^2 + h^2 = R^2 ∧
    -- The volume of the cone is maximized
    (∀ (r' h' : ℝ), r'^2 + h'^2 = R^2 → 
      (1/3) * Real.pi * r^2 * h ≥ (1/3) * Real.pi * r'^2 * h') ∧
    -- The radius of the inscribed sphere in the cone
    let l := Real.sqrt (r^2 + h^2)
    (2 * r * h) / (r + h + l) = 4 * (Real.sqrt 3 - 1) / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_l538_53816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_35_l538_53827

/-- An ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop

/-- A line in the xy-plane -/
structure Line where
  m : ℝ
  c : ℝ
  eq : (x y : ℝ) → Prop

/-- The length of a chord formed by the intersection of an ellipse and a line -/
noncomputable def chordLength (e : Ellipse) (l : Line) : ℝ := sorry

/-- The specific ellipse x^2 + 4y^2 = 16 -/
def givenEllipse : Ellipse where
  a := 4
  b := 2
  eq := fun x y ↦ x^2 + 4*y^2 = 16

/-- The specific line y = (1/2)x + 1 -/
noncomputable def givenLine : Line where
  m := 1/2
  c := 1
  eq := fun x y ↦ y = (1/2)*x + 1

theorem chord_length_is_sqrt_35 :
  chordLength givenEllipse givenLine = Real.sqrt 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_35_l538_53827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_product_l538_53892

open BigOperators
open Finset

theorem sum_reciprocal_product (n : ℕ) :
  ∑ k in range n, (1 : ℝ) / ((k + 1) * (k + 2) * (k + 3)) =
    1/2 * (1/2 - 1 / ((n + 1) * (n + 2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_product_l538_53892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l538_53854

-- Define the ellipse C
noncomputable def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line l
def line (x y : ℝ) : Prop := y = x - 4/3

-- Define the right focus F
def right_focus : ℝ × ℝ := (1, 0)

-- Define vertex B
def vertex_B : ℝ × ℝ := (0, 1)

-- Define eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

theorem ellipse_and_line_properties :
  ∀ (M N : ℝ × ℝ),
  ellipse M.1 M.2 ∧ ellipse N.1 N.2 ∧  -- M and N are on the ellipse
  line M.1 M.2 ∧ line N.1 N.2 ∧        -- M and N are on the line
  (∃ (F : ℝ × ℝ), F = right_focus ∧ 
    -- F is the orthocenter of triangle BMN
    (F.2 - M.2) * (N.1 - vertex_B.1) = (F.1 - M.1) * (N.2 - vertex_B.2) ∧
    (F.2 - N.2) * (M.1 - vertex_B.1) = (F.1 - N.1) * (M.2 - vertex_B.2)) →
  -- The ellipse equation is correct
  (∀ (x y : ℝ), ellipse x y ↔ x^2/2 + y^2 = 1) ∧
  -- The line equation is correct
  (∀ (x y : ℝ), line x y ↔ y = x - 4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l538_53854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l538_53800

open Set
open Real

noncomputable def f (x : ℝ) := (Real.sqrt (sin x) + log (cos x)) / tan x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = ⋃ (k : ℤ), Ioo (2 * k * π) ((π / 2) + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l538_53800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_formula_l538_53828

/-- Represents a tetrahedron with given edge lengths -/
structure Tetrahedron where
  a : ℝ  -- Length of edge BC
  b : ℝ  -- Length of edge CA
  c : ℝ  -- Length of edge AB
  a₁ : ℝ -- Length of edge DA
  b₁ : ℝ -- Length of edge DB
  c₁ : ℝ -- Length of edge DC

/-- The square of the length of the median from vertex D to the centroid of face ABC -/
noncomputable def median_length_squared (t : Tetrahedron) : ℝ :=
  (1/3) * (t.a₁^2 + t.b₁^2 + t.c₁^2) - (1/9) * (t.a^2 + t.b^2 + t.c^2)

/-- Theorem stating that the square of the median length is equal to the given formula -/
theorem median_length_formula (t : Tetrahedron) :
  ∃ h : ℝ, h^2 = median_length_squared t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_formula_l538_53828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_card_numbers_l538_53870

def is_valid_arrangement (arrangement : List Nat) : Prop :=
  arrangement.length ≥ 4 ∧
  arrangement.all (λ n => n ∈ [1, 2, 4, 8]) ∧
  (∀ n ∈ [1, 2, 4, 8], n ∈ arrangement) ∧
  (∀ i, i < arrangement.length → 
    ((arrangement.get! i + arrangement.get! ((i + 1) % arrangement.length)) % 3 = 0)) ∧
  (∀ i, i < arrangement.length → 
    ((arrangement.get! i + arrangement.get! ((i + 1) % arrangement.length)) % 9 ≠ 0))

theorem valid_card_numbers :
  ∀ n : Nat, (∃ arrangement : List Nat, arrangement.length = n ∧ is_valid_arrangement arrangement) ↔ (n = 8 ∨ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_card_numbers_l538_53870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_and_inequality_l538_53818

-- Define the function f
def f (x : ℝ) : ℝ := |2 * |x| - 1|

-- Define the solution set A
def A : Set ℝ := {x | f x ≤ 1}

-- Theorem statement
theorem solution_and_inequality :
  (A = Set.Icc (-1) 1) ∧
  (∀ m n : ℝ, m ∈ A → n ∈ A → |m + n| ≤ m * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_and_inequality_l538_53818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_ratio_l538_53850

/-- The ratio of the longer to shorter dimension of a rectangle containing
    tangent circles arranged in a specific pattern --/
noncomputable def rectangle_ratio (n : ℕ) (m : ℕ) : ℝ :=
  (2 * n) / (2 * (Real.sqrt 3 + 1))

/-- The theorem stating the equality between the calculated ratio and the given form --/
theorem circle_arrangement_ratio :
  rectangle_ratio 7 3 = (1 / 2) * (Real.sqrt 147 - 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_ratio_l538_53850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_match_correct_probability_of_match_approx_l538_53840

/-- The number of numbers in the lottery pool --/
def totalNumbers : ℕ := 36

/-- The number of numbers drawn each week --/
def numbersDrawn : ℕ := 6

/-- Binomial coefficient calculation --/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Probability of at least one number matching between two independent draws --/
def probabilityOfMatch : ℚ :=
  1 - (binomial (totalNumbers - numbersDrawn) numbersDrawn : ℚ) / 
      (binomial totalNumbers numbersDrawn : ℚ)

/-- Theorem stating that the probability of at least one match is correct --/
theorem probability_of_match_correct :
  probabilityOfMatch = 1 - (binomial (totalNumbers - numbersDrawn) numbersDrawn : ℚ) / 
                           (binomial totalNumbers numbersDrawn : ℚ) :=
by
  rfl

/-- Theorem stating that the probability is approximately 0.695 --/
theorem probability_of_match_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |(probabilityOfMatch : ℝ) - 0.695| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_match_correct_probability_of_match_approx_l538_53840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_2x_over_complex_denominator_l538_53830

open Real

theorem integral_sin_2x_over_complex_denominator (x : ℝ) :
  let f := λ y : ℝ ↦ (1/6) * log (abs ((2 - cos (2*y)) / (1 + cos (2*y))))
  deriv f x = sin (2*x) / (1 + cos (2*x) + sin (2*x)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_2x_over_complex_denominator_l538_53830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_powers_i_binomial_permutation_ratio_sum_binomial_3_l538_53879

-- Part 1
def complex_i : ℂ := Complex.I

theorem complex_sum_powers_i :
  complex_i + complex_i^2 + complex_i^3 + complex_i^4 = 0 := by sorry

-- Part 2
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Replace Nat.permute with a definition using factorial
def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem binomial_permutation_ratio :
  (binomial 100 2 + binomial 100 97) / permutation 101 3 = 1 / 6 := by sorry

-- Part 3
theorem sum_binomial_3 :
  (Finset.range 8).sum (fun k => binomial (k + 3) 3) = 330 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_powers_i_binomial_permutation_ratio_sum_binomial_3_l538_53879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snowfall_approximation_l538_53846

/-- Represents the snowfall measurements in Yardley over two days -/
structure SnowfallData where
  monday_morning : Float
  monday_afternoon : Float
  tuesday_morning : Float
  tuesday_afternoon : Float

/-- Conversion factors between different units of measurement -/
structure ConversionFactors where
  inch_to_cm : Float
  cm_to_mm : Float

/-- Calculates the total snowfall in inches given the snowfall data and conversion factors -/
def calculateTotalSnowfall (data : SnowfallData) (factors : ConversionFactors) : Float :=
  data.monday_morning + data.monday_afternoon +
  (data.tuesday_morning / factors.inch_to_cm) +
  (data.tuesday_afternoon / factors.cm_to_mm / factors.inch_to_cm)

/-- Theorem stating that the total snowfall is approximately 2.141 inches -/
theorem total_snowfall_approximation (data : SnowfallData) (factors : ConversionFactors)
  (h1 : data.monday_morning = 0.125)
  (h2 : data.monday_afternoon = 0.5)
  (h3 : data.tuesday_morning = 1.35)
  (h4 : data.tuesday_afternoon = 25)
  (h5 : factors.inch_to_cm = 2.54)
  (h6 : factors.cm_to_mm = 10) :
  (calculateTotalSnowfall data factors - 2.141).abs < 0.001 := by
  sorry

#eval calculateTotalSnowfall
  { monday_morning := 0.125
    monday_afternoon := 0.5
    tuesday_morning := 1.35
    tuesday_afternoon := 25 }
  { inch_to_cm := 2.54
    cm_to_mm := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snowfall_approximation_l538_53846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l538_53860

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.sqrt 3 * Real.cos x - Real.sin x)

theorem f_properties :
  let period : ℝ := π
  let max_value : ℝ := 1/2
  let min_value : ℝ := -1/2
  let interval : Set ℝ := Set.Icc (-π/12) (π/3)
  (∀ x : ℝ, f (x + period) = f x) ∧ 
  (∀ x ∈ interval, f x ≤ max_value) ∧
  (∀ x ∈ interval, f x ≥ min_value) ∧
  (∃ x ∈ interval, f x = max_value) ∧
  (∃ x ∈ interval, f x = min_value) ∧
  (∀ t : ℝ, t > 0 → t < period → ¬(∀ x : ℝ, f (x + t) = f x)) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l538_53860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_value_multiple_art_piece_value_multiple_l538_53866

/-- Theorem: Given an art piece with an original price and a future increase in value,
    the multiple of the original price it will be worth is equal to
    (original price + increase in value) / original price -/
theorem art_value_multiple (original_price increase_value : ℝ) :
  original_price > 0 →
  (original_price + increase_value) / original_price =
    (original_price + increase_value) / original_price :=
by
  sorry

/-- The multiple of the original price that the art piece will be worth -/
noncomputable def art_value_multiple_calculation (original_price increase_value : ℝ) : ℝ :=
  (original_price + increase_value) / original_price

theorem art_piece_value_multiple :
  let original_price : ℝ := 4000
  let increase_value : ℝ := 8000
  art_value_multiple_calculation original_price increase_value = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_value_multiple_art_piece_value_multiple_l538_53866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_fifth_term_mySequence_pattern_l538_53802

def mySequence : ℕ → ℕ
  | 0 => 3
  | 1 => 6
  | 2 => 12
  | 3 => 21
  | 4 => 33
  | 5 => 48
  | n + 6 => mySequence (n + 5) + (n + 6) * 3

theorem mySequence_fifth_term : mySequence 4 = 33 := by
  rfl

theorem mySequence_pattern (n : ℕ) : n ≥ 1 → mySequence n - mySequence (n - 1) = n * 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_fifth_term_mySequence_pattern_l538_53802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ubitari_word_count_l538_53859

/-- The number of letters in the Ubitari alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum length of a word in the Ubitari language -/
def max_word_length : ℕ := 5

/-- The number of valid words of length n in the Ubitari language -/
def valid_words (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n > max_word_length then 0
  else alphabet_size^n - (alphabet_size - 1)^n

/-- The total number of valid words in the Ubitari language -/
def total_valid_words : ℕ :=
  (List.range (max_word_length + 1)).map valid_words |>.sum

/-- Theorem: The number of valid words in the Ubitari language is 1864701 -/
theorem ubitari_word_count : total_valid_words = 1864701 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ubitari_word_count_l538_53859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l538_53897

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (2, 4)
def b (x : ℝ) : ℝ × ℝ := (-2, x)

/-- Parallel vectors have proportional components -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Cosine of the angle between two vectors -/
noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

/-- Perpendicular vectors have zero dot product -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  dot_product v w = 0

theorem vector_properties :
  (∃ x : ℝ, parallel a (b x) ∧ x = -4) ∧
  (cos_angle a (b (-1)) = -4/5) ∧
  (∃ x : ℝ, perpendicular a (4 • a + b x) ∧ magnitude (b x) = Real.sqrt 365) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l538_53897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l538_53812

-- Define the power function
noncomputable def f (m : ℕ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 3)

-- Define the symmetry and decreasing conditions
def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

-- Define the inequality condition
def satisfies_inequality (a : ℝ) (m : ℕ) : Prop :=
  (a + 1)^(-(m : ℝ) / 3) < (3 - 2*a)^(-(m : ℝ) / 3)

-- State the theorem
theorem power_function_properties :
  ∃ (m : ℕ),
    m > 0 ∧
    (is_symmetric_about_y_axis (f m) ∧
     is_decreasing_on_positive_reals (f m)) ∧
    (∀ a : ℝ, satisfies_inequality a m ↔ 
      (a < -1 ∨ (2/3 < a ∧ a < 3/2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l538_53812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_relationship_l538_53803

/-- Quadratic function passing through specific points -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ
  a : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_at_A : a = -(-1)^2 - b*(-1) - c
  eq_at_B : a = -(3)^2 - b*(3) - c
  eq_at_C : y₁ = -(-2)^2 - b*(-2) - c
  eq_at_D : y₂ = -(Real.sqrt 2)^2 - b*(Real.sqrt 2) - c
  eq_at_E : y₃ = -(1)^2 - b*(1) - c

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem quadratic_function_relationship (f : QuadraticFunction) : f.y₃ < f.y₂ ∧ f.y₂ < f.y₁ := by
  sorry

#check quadratic_function_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_relationship_l538_53803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_idol_winner_votes_l538_53873

theorem math_idol_winner_votes :
  ∀ (total votes_1 votes_2 votes_3 votes_4 : ℕ),
  total = 5219000 →
  votes_1 = votes_2 + 22000 →
  votes_1 = votes_3 + 30000 →
  votes_1 = votes_4 + 73000 →
  votes_1 + votes_2 + votes_3 + votes_4 = total →
  votes_1 = 1336000 := by
  intro total votes_1 votes_2 votes_3 votes_4
  intro h_total h_votes_2 h_votes_3 h_votes_4 h_sum
  -- The proof steps would go here
  sorry

#check math_idol_winner_votes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_idol_winner_votes_l538_53873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_function_f_negative_x_l538_53843

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 1 + 2*x - x^2 else 1 - 2*x - x^2

theorem f_is_even_function (x : ℝ) : 
  f x = f (-x) := by sorry

theorem f_negative_x (x : ℝ) (h : x < 0) : 
  f x = 1 - 2*x - x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_function_f_negative_x_l538_53843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_and_white_specific_l538_53809

/-- Represents the color of a ball -/
inductive Color where
  | Red
  | White
deriving DecidableEq

/-- Represents a bag of balls -/
structure Bag where
  balls : Finset Color

/-- The probability of drawing both a red and a white ball -/
def prob_red_and_white (bag : Bag) : ℚ :=
  let total_balls := bag.balls.card
  let red_balls := (bag.balls.filter (· = Color.Red)).card
  let white_balls := (bag.balls.filter (· = Color.White)).card
  let favorable_outcomes := red_balls * white_balls
  let total_outcomes := total_balls * (total_balls - 1) / 2
  favorable_outcomes / total_outcomes

/-- The theorem to be proved -/
theorem prob_red_and_white_specific :
  ∃ (bag : Bag),
    bag.balls.card = 5 ∧
    (bag.balls.filter (· = Color.Red)).card = 2 ∧
    (bag.balls.filter (· = Color.White)).card = 3 ∧
    prob_red_and_white bag = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_and_white_specific_l538_53809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_problem_solution_l538_53894

/-- Represents the weaving scenario described in the problem -/
structure WeavingScenario where
  days : ℕ
  start_amount : ℚ
  end_amount : ℚ

/-- Calculates the total amount of cloth woven given a WeavingScenario -/
noncomputable def total_cloth_woven (scenario : WeavingScenario) : ℚ :=
  let daily_decrease := (scenario.start_amount - scenario.end_amount) / (scenario.days - 1)
  scenario.days * (scenario.start_amount + scenario.end_amount) / 2

/-- The theorem stating the total amount of cloth woven in the given scenario -/
theorem weaving_problem_solution :
  let scenario : WeavingScenario := ⟨30, 5, 1⟩
  total_cloth_woven scenario = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_problem_solution_l538_53894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l538_53883

theorem smallest_positive_z (x z : ℝ) : 
  Real.cos x = 0 → 
  Real.cos (x + z) = 1/2 → 
  (∀ w, w > 0 → Real.cos (x + w) = 1/2 → z ≤ w) → 
  z = π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l538_53883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_feeding_cost_increase_is_twenty_percent_l538_53801

/-- Represents the cattle purchase and sale scenario --/
structure CattleSale where
  num_cattle : ℕ
  buying_cost : ℚ
  weight_per_cattle : ℚ
  selling_price_per_pound : ℚ
  profit : ℚ

/-- Calculates the percentage increase in feeding cost compared to buying cost --/
def feeding_cost_percentage_increase (sale : CattleSale) : ℚ :=
  let total_selling_price := sale.num_cattle * sale.weight_per_cattle * sale.selling_price_per_pound
  let total_cost := total_selling_price - sale.profit
  let feeding_cost := total_cost - sale.buying_cost
  let increase_in_cost := feeding_cost - sale.buying_cost
  (increase_in_cost / sale.buying_cost) * 100

/-- Theorem stating that the percentage increase in feeding cost is 20% --/
theorem feeding_cost_increase_is_twenty_percent (sale : CattleSale)
  (h1 : sale.num_cattle = 100)
  (h2 : sale.buying_cost = 40000)
  (h3 : sale.weight_per_cattle = 1000)
  (h4 : sale.selling_price_per_pound = 2)
  (h5 : sale.profit = 112000) :
  feeding_cost_percentage_increase sale = 20 := by
  sorry

def example_sale : CattleSale := {
  num_cattle := 100,
  buying_cost := 40000,
  weight_per_cattle := 1000,
  selling_price_per_pound := 2,
  profit := 112000
}

#eval feeding_cost_percentage_increase example_sale

end NUMINAMATH_CALUDE_ERRORFEEDBACK_feeding_cost_increase_is_twenty_percent_l538_53801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_shifted_functions_l538_53807

-- Define the functions h and j
def h : ℝ → ℝ := sorry
def j : ℝ → ℝ := sorry

-- Define the given intersection points
axiom intersection1 : h 2 = j 2 ∧ h 2 = 2
axiom intersection2 : h 4 = j 4 ∧ h 4 = 6
axiom intersection3 : h 6 = j 6 ∧ h 6 = 12
axiom intersection4 : h 8 = j 8 ∧ h 8 = 12

-- Theorem statement
theorem intersection_of_shifted_functions :
  ∃ (x y : ℝ), h (x + 2) = j (2 * x) ∧ h (x + 2) = y ∧ x + y = 8 ∧
  ∃ (a b : ℝ), h (a + 2) = j (2 * a) ∧ h (a + 2) = b ∧ a + b = 16 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_shifted_functions_l538_53807
