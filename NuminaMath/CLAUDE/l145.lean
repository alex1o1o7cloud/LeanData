import Mathlib

namespace NUMINAMATH_CALUDE_irrational_numbers_in_set_l145_14537

theorem irrational_numbers_in_set : 
  let S : Set ℝ := {1/3, Real.pi, 0, Real.sqrt 5}
  ∀ x ∈ S, Irrational x ↔ (x = Real.pi ∨ x = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_irrational_numbers_in_set_l145_14537


namespace NUMINAMATH_CALUDE_triangle_circumradius_l145_14591

/-- Given a triangle with side lengths 8, 15, and 17, its circumradius is 8.5 -/
theorem triangle_circumradius : ∀ (a b c : ℝ), 
  a = 8 ∧ b = 15 ∧ c = 17 →
  (a^2 + b^2 = c^2) →
  (c / 2 = 8.5) := by
  sorry

#check triangle_circumradius

end NUMINAMATH_CALUDE_triangle_circumradius_l145_14591


namespace NUMINAMATH_CALUDE_four_correct_statements_l145_14571

theorem four_correct_statements (a b m : ℝ) : 
  -- Statement 1
  (∀ m, a * m^2 > b * m^2 → a > b) ∧
  -- Statement 2
  (a > b → a * |a| > b * |b|) ∧
  -- Statement 3
  (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b) ∧
  -- Statement 4
  (a > b ∧ b > 0 ∧ |Real.log a| = |Real.log b| → 2 * a + b > 3) :=
by sorry

end NUMINAMATH_CALUDE_four_correct_statements_l145_14571


namespace NUMINAMATH_CALUDE_propositions_truth_l145_14538

-- Define the necessary geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- Define the propositions
def proposition_1 (p1 p2 p3 : Plane) (l1 l2 : Line) : Prop :=
  line_in_plane l1 p1 → line_in_plane l2 p1 →
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2

def proposition_2 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2

def proposition_3 (l1 l2 l3 : Line) : Prop :=
  perpendicular_line_plane l1 l3 → perpendicular_line_plane l2 l3 → parallel l1 l2

def proposition_4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular p1 p2 →
  line_in_plane l p1 →
  ¬perpendicular_line_plane l (line_of_intersection p1 p2) →
  ¬perpendicular_line_plane l p2

-- Theorem stating which propositions are true and which are false
theorem propositions_truth : 
  (∃ p1 p2 p3 : Plane, ∃ l1 l2 : Line, ¬proposition_1 p1 p2 p3 l1 l2) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_2 p1 p2 l) ∧
  (∃ l1 l2 l3 : Line, ¬proposition_3 l1 l2 l3) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_4 p1 p2 l) :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_l145_14538


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l145_14574

theorem relationship_between_a_and_b (a b : ℝ) 
  (ha : a > 0) (hb : b < 0) (hab : a + b < 0) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l145_14574


namespace NUMINAMATH_CALUDE_natural_number_solution_system_l145_14599

theorem natural_number_solution_system (x y z t a b : ℕ) : 
  x^2 + y^2 = a ∧ 
  z^2 + t^2 = b ∧ 
  (x^2 + t^2) * (z^2 + y^2) = 50 →
  ((x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
   (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
   (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1)) :=
by sorry

#check natural_number_solution_system

end NUMINAMATH_CALUDE_natural_number_solution_system_l145_14599


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l145_14576

-- Problem 1
theorem problem_1 : 3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27 = 6 * Real.sqrt 3 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12) = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by sorry

-- Problem 3
theorem problem_3 : (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6)^2 = 3 + 2 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l145_14576


namespace NUMINAMATH_CALUDE_interesting_factor_exists_l145_14545

/-- A natural number is interesting if it can be represented both as the sum of two consecutive integers and as the sum of three consecutive integers. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ k m : ℤ, n = (k + (k + 1)) ∧ n = (m - 1 + m + (m + 1))

/-- The theorem states that if the product of five different natural numbers is interesting,
    then at least one of these natural numbers is interesting. -/
theorem interesting_factor_exists (a b c d e : ℕ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
    (h_interesting : is_interesting (a * b * c * d * e)) :
    is_interesting a ∨ is_interesting b ∨ is_interesting c ∨ is_interesting d ∨ is_interesting e :=
  sorry

end NUMINAMATH_CALUDE_interesting_factor_exists_l145_14545


namespace NUMINAMATH_CALUDE_complex_number_location_l145_14528

theorem complex_number_location : ∃ (z : ℂ), z = Complex.I * (Complex.I - 1) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l145_14528


namespace NUMINAMATH_CALUDE_product_equals_one_l145_14557

theorem product_equals_one :
  16 * 0.5 * 4 * 0.0625 / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l145_14557


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l145_14516

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (x : ℝ) (n : ℕ), x ∈ Set.Icc 0 1 → n > 0 → x^k * (1-x)^n < 1 / (1+n:ℝ)^3) ∧
  (∀ (k' : ℕ), k' > 0 → k' < k → 
    ∃ (x : ℝ) (n : ℕ), x ∈ Set.Icc 0 1 ∧ n > 0 ∧ x^k' * (1-x)^n ≥ 1 / (1+n:ℝ)^3) ∧
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l145_14516


namespace NUMINAMATH_CALUDE_specific_participants_match_probability_l145_14509

/-- The number of participants in the tournament -/
def n : ℕ := 26

/-- The probability that two specific participants will play against each other -/
def probability : ℚ := 1 / 13

/-- Theorem stating the probability of two specific participants playing against each other -/
theorem specific_participants_match_probability :
  (n - 1 : ℚ) / (n * (n - 1) / 2) = probability := by sorry

end NUMINAMATH_CALUDE_specific_participants_match_probability_l145_14509


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l145_14592

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length : ℝ := Real.sqrt area
  let perimeter : ℝ := 4 * side_length
  let total_cost : ℝ := perimeter * price_per_foot
  total_cost = 3944 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l145_14592


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l145_14555

/-- Acme's cost function -/
def acme_cost (x : ℕ) : ℕ := 75 + 10 * x

/-- Beta's cost function -/
def beta_cost (x : ℕ) : ℕ :=
  if x < 30 then 15 * x else 14 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme : ℕ := 20

theorem acme_cheaper_at_min_shirts :
  (∀ x < min_shirts_for_acme, beta_cost x ≤ acme_cost x) ∧
  (beta_cost min_shirts_for_acme > acme_cost min_shirts_for_acme) := by
  sorry

#check acme_cheaper_at_min_shirts

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l145_14555


namespace NUMINAMATH_CALUDE_max_value_expression_l145_14524

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + b^2)) ≤ a^2 + b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + b^2)) = a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l145_14524


namespace NUMINAMATH_CALUDE_larger_region_area_unit_circle_chord_l145_14515

/-- The area of the larger region formed by a chord of length 1 on a unit circle -/
theorem larger_region_area_unit_circle_chord (chord_length : Real) (h : chord_length = 1) :
  let circle_area : Real := π
  let triangle_area : Real := (Real.sqrt 3) / 4
  let sector_area : Real := π / 6
  let segment_area : Real := sector_area - triangle_area
  let larger_region_area : Real := circle_area - segment_area
  larger_region_area = 5 * π / 6 + (Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_larger_region_area_unit_circle_chord_l145_14515


namespace NUMINAMATH_CALUDE_polygon_diagonals_equal_sides_l145_14573

theorem polygon_diagonals_equal_sides : ∃ (n : ℕ), n > 0 ∧ n * (n - 3) / 2 = n := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_equal_sides_l145_14573


namespace NUMINAMATH_CALUDE_division_remainder_l145_14530

theorem division_remainder : ∃ r : ℕ, 
  12401 = 163 * 76 + r ∧ r < 163 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_remainder_l145_14530


namespace NUMINAMATH_CALUDE_tshirt_cost_l145_14549

-- Define the problem parameters
def initial_amount : ℕ := 26
def jumper_cost : ℕ := 9
def heels_cost : ℕ := 5
def remaining_amount : ℕ := 8

-- Define the theorem to prove
theorem tshirt_cost : 
  initial_amount - jumper_cost - heels_cost - remaining_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l145_14549


namespace NUMINAMATH_CALUDE_satellite_sensor_ratio_l145_14519

theorem satellite_sensor_ratio (total_sensors : ℝ) (non_upgraded_per_unit : ℝ) : 
  total_sensors > 0 →
  non_upgraded_per_unit ≥ 0 →
  (24 * non_upgraded_per_unit + 0.25 * total_sensors = total_sensors) →
  (non_upgraded_per_unit / (0.25 * total_sensors) = 1 / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_satellite_sensor_ratio_l145_14519


namespace NUMINAMATH_CALUDE_B_elements_l145_14503

def B : Set ℤ := {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 3}

theorem B_elements : B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_B_elements_l145_14503


namespace NUMINAMATH_CALUDE_calculate_expression_l145_14596

theorem calculate_expression : 12 - (-18) + (-7) = 23 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l145_14596


namespace NUMINAMATH_CALUDE_expression_satisfies_conditions_l145_14594

def original_expression : ℕ := 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

def transformed_expression : ℕ := 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1

theorem expression_satisfies_conditions :
  (original_expression = 11) ∧
  (transformed_expression = 11) :=
by
  sorry

#eval original_expression
#eval transformed_expression

end NUMINAMATH_CALUDE_expression_satisfies_conditions_l145_14594


namespace NUMINAMATH_CALUDE_difference_of_squares_l145_14552

theorem difference_of_squares (a b : ℝ) : (2*a + b) * (b - 2*a) = b^2 - 4*a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l145_14552


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l145_14541

theorem right_triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_n_ge_3 : n ≥ 3) : 
  a^n + b^n < c^n := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l145_14541


namespace NUMINAMATH_CALUDE_inequality_proofs_l145_14534

theorem inequality_proofs 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) 
  (x : ℝ) (hx : x ≥ 0) 
  (a b p q : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) (hq : q > 0)
  (hpq : 1 / p + 1 / q = 1) : 
  (x^α - α*x ≤ 1 - α) ∧ (a * b ≤ (1/p) * a^p + (1/q) * b^q) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l145_14534


namespace NUMINAMATH_CALUDE_circle_center_l145_14569

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center (x y : ℝ) : 
  (3 * x + 4 * y = 24) →  -- First tangent line
  (3 * x + 4 * y = -16) →  -- Second tangent line
  (x - 2 * y = 0) →  -- Line containing the center
  (x = 4/5 ∧ y = 2/5)  -- Center coordinates
  :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l145_14569


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l145_14575

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_left (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y ∧ y ≤ -1 → f x ≤ f y

-- State the theorem
theorem even_increasing_inequality 
  (h_even : is_even f) 
  (h_incr : increasing_on_left f) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) := by
sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l145_14575


namespace NUMINAMATH_CALUDE_inequality_proof_l145_14567

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 1) :
  (x^(n-1) - 1) / (n-1 : ℝ) ≤ (x^n - 1) / n :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l145_14567


namespace NUMINAMATH_CALUDE_sale_price_lower_than_original_l145_14550

theorem sale_price_lower_than_original (x : ℝ) (h : x > 0) :
  0.75 * (1.30 * x) < x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_lower_than_original_l145_14550


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_l145_14544

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to reverse the digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

-- Define a function to check if a number is a two-digit number with 2 as the tens digit
def isTwoDigitWithTensTwo (n : ℕ) : Prop := n ≥ 20 ∧ n < 30

-- Main theorem
theorem smallest_two_digit_prime_with_reversed_composite :
  ∃ (n : ℕ), 
    isPrime n ∧ 
    isTwoDigitWithTensTwo n ∧ 
    ¬(isPrime (reverseDigits n)) ∧
    (∀ m : ℕ, m < n → ¬(isPrime m ∧ isTwoDigitWithTensTwo m ∧ ¬(isPrime (reverseDigits m)))) ∧
    n = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_l145_14544


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l145_14535

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -2753 [ZMOD 8] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l145_14535


namespace NUMINAMATH_CALUDE_boys_not_adjacent_girls_adjacent_girls_not_at_ends_l145_14590

/-- The number of boys in the group -/
def num_boys : Nat := 3

/-- The number of girls in the group -/
def num_girls : Nat := 2

/-- The total number of people in the group -/
def total_people : Nat := num_boys + num_girls

/-- Calculates the number of ways to arrange n distinct objects -/
def permutations (n : Nat) : Nat := Nat.factorial n

/-- Theorem stating the number of ways boys are not adjacent -/
theorem boys_not_adjacent : 
  permutations num_girls * permutations num_boys = 12 := by sorry

/-- Theorem stating the number of ways girls are adjacent -/
theorem girls_adjacent : 
  permutations (total_people - num_girls + 1) * permutations num_girls = 48 := by sorry

/-- Theorem stating the number of ways girls are not at the ends -/
theorem girls_not_at_ends : 
  (total_people - 2) * permutations num_boys = 36 := by sorry

end NUMINAMATH_CALUDE_boys_not_adjacent_girls_adjacent_girls_not_at_ends_l145_14590


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l145_14556

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Triangle ABC on a grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  (1/2 : ℚ) * ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) : ℤ)

theorem area_of_triangle_ABC (t : GridTriangle) :
  t.A = ⟨0, 0⟩ →
  t.B = ⟨0, 2⟩ →
  t.C = ⟨3, 0⟩ →
  triangleArea t.A t.B t.C = 1/2 := by
  sorry

#check area_of_triangle_ABC

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l145_14556


namespace NUMINAMATH_CALUDE_product_equals_two_l145_14517

theorem product_equals_two : 10 * (1/5) * 4 * (1/16) * (1/2) * 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_two_l145_14517


namespace NUMINAMATH_CALUDE_parrots_left_on_branch_l145_14597

/-- Represents the number of birds on a tree branch -/
structure BirdCount where
  parrots : ℕ
  crows : ℕ

/-- The initial state of birds on the branch -/
def initialState : BirdCount where
  parrots := 7
  crows := 13 - 7

/-- The number of birds that flew away -/
def flownAway : ℕ :=
  initialState.crows - 1

/-- The final state of birds on the branch -/
def finalState : BirdCount where
  parrots := initialState.parrots - flownAway
  crows := 1

theorem parrots_left_on_branch :
  finalState.parrots = 2 :=
sorry

end NUMINAMATH_CALUDE_parrots_left_on_branch_l145_14597


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l145_14522

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -7; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![44/7, -57/7; -39/14, 51/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l145_14522


namespace NUMINAMATH_CALUDE_flowchart_result_for_6_l145_14543

-- Define the function that represents the flowchart logic
def flowchart_program (n : ℕ) : ℕ :=
  -- The actual implementation is not provided, so we'll use a placeholder
  sorry

-- Theorem statement
theorem flowchart_result_for_6 : flowchart_program 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_flowchart_result_for_6_l145_14543


namespace NUMINAMATH_CALUDE_servant_served_nine_months_l145_14554

/-- Represents the compensation and service time of a servant --/
structure ServantCompensation where
  fullYearSalary : ℕ  -- Salary for a full year in Rupees
  uniformPrice : ℕ    -- Price of the uniform in Rupees
  receivedSalary : ℕ  -- Salary actually received in Rupees
  monthsServed : ℕ    -- Number of months served

/-- Calculates the total compensation for a full year --/
def fullYearCompensation (s : ServantCompensation) : ℕ :=
  s.fullYearSalary + s.uniformPrice

/-- Calculates the total compensation received --/
def totalReceived (s : ServantCompensation) : ℕ :=
  s.receivedSalary + s.uniformPrice

/-- Theorem stating that under given conditions, the servant served for 9 months --/
theorem servant_served_nine_months (s : ServantCompensation)
  (h1 : s.fullYearSalary = 600)
  (h2 : s.uniformPrice = 200)
  (h3 : s.receivedSalary = 400)
  : s.monthsServed = 9 := by
  sorry


end NUMINAMATH_CALUDE_servant_served_nine_months_l145_14554


namespace NUMINAMATH_CALUDE_max_marks_proof_l145_14529

/-- Given a student needs 60% to pass, got 220 marks, and failed by 50 marks, prove the maximum marks are 450. -/
theorem max_marks_proof (passing_percentage : Real) (student_marks : ℕ) (failing_margin : ℕ) 
  (h1 : passing_percentage = 0.60)
  (h2 : student_marks = 220)
  (h3 : failing_margin = 50) :
  (student_marks + failing_margin) / passing_percentage = 450 := by
  sorry

end NUMINAMATH_CALUDE_max_marks_proof_l145_14529


namespace NUMINAMATH_CALUDE_two_x_is_equal_mean_value_function_l145_14589

/-- A function is an "equal mean value function" if it satisfies two conditions:
    1) For any x in its domain, f(x) + f(-x) = 0
    2) For any x₁ in its domain, there exists x₂ such that (f(x₁) + f(x₂))/2 = (x₁ + x₂)/2 -/
def is_equal_mean_value_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x₁, ∃ x₂, (f x₁ + f x₂) / 2 = (x₁ + x₂) / 2)

/-- The function f(x) = 2x is an "equal mean value function" -/
theorem two_x_is_equal_mean_value_function :
  is_equal_mean_value_function (λ x ↦ 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_two_x_is_equal_mean_value_function_l145_14589


namespace NUMINAMATH_CALUDE_magnitude_of_2_plus_3i_l145_14598

theorem magnitude_of_2_plus_3i :
  Complex.abs (2 + 3 * Complex.I) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_2_plus_3i_l145_14598


namespace NUMINAMATH_CALUDE_product_of_tangents_plus_one_l145_14564

theorem product_of_tangents_plus_one (α : ℝ) :
  (1 + Real.tan (α * π / 12)) * (1 + Real.tan (α * π / 6)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_tangents_plus_one_l145_14564


namespace NUMINAMATH_CALUDE_a_less_than_one_sufficient_not_necessary_l145_14531

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 + 2 * a * x + 6 * y + 5 * a = 0

-- Define what it means for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

-- Theorem stating that a < 1 is sufficient but not necessary
theorem a_less_than_one_sufficient_not_necessary :
  (∀ a : ℝ, a < 1 → is_circle a) ∧
  ¬(∀ a : ℝ, is_circle a → a < 1) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_one_sufficient_not_necessary_l145_14531


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l145_14558

theorem least_number_with_remainder_four (n : ℕ) : n = 184 ↔ 
  (n > 0) ∧ 
  (∀ m : ℕ, 0 < m ∧ m < n → 
    (m % 5 ≠ 4 ∨ m % 6 ≠ 4 ∨ m % 9 ≠ 4 ∨ m % 12 ≠ 4)) ∧
  (n % 5 = 4) ∧ (n % 6 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l145_14558


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l145_14582

def trailing_zeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeroes :
  trailing_zeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l145_14582


namespace NUMINAMATH_CALUDE_sum_of_decimals_l145_14532

theorem sum_of_decimals : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l145_14532


namespace NUMINAMATH_CALUDE_monomial_properties_l145_14579

/-- Represents a monomial with coefficient and variables -/
structure Monomial where
  coeff : ℚ
  a_exp : ℕ
  b_exp : ℕ

/-- The given monomial 3a²b/2 -/
def given_monomial : Monomial := { coeff := 3/2, a_exp := 2, b_exp := 1 }

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℚ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.a_exp + m.b_exp

theorem monomial_properties :
  coefficient given_monomial = 3/2 ∧ degree given_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l145_14579


namespace NUMINAMATH_CALUDE_no_solution_exists_l145_14577

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (heq : x^y + 1 = z^2) : False :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l145_14577


namespace NUMINAMATH_CALUDE_right_triangle_sets_l145_14540

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 2 ∧ b = 3 ∧ c = 4) ∧
  ¬(a^2 + b^2 = c^2) ∧
  (3^2 + 4^2 = 5^2) ∧
  (6^2 + 8^2 = 10^2) ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l145_14540


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_250_l145_14559

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factors (n : ℕ) : Set ℕ :=
  {p : ℕ | is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q → q ∣ n → p ≤ q}

theorem sum_two_smallest_prime_factors_250 :
  ∃ (a b : ℕ), a ∈ smallest_prime_factors 250 ∧ 
               b ∈ smallest_prime_factors 250 ∧ 
               a ≠ b ∧
               a + b = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_250_l145_14559


namespace NUMINAMATH_CALUDE_green_ball_probability_l145_14533

-- Define the number of balls of each color
def green_balls : ℕ := 2
def black_balls : ℕ := 3
def red_balls : ℕ := 6

-- Define the total number of balls
def total_balls : ℕ := green_balls + black_balls + red_balls

-- Define the probability of drawing a green ball
def prob_green_ball : ℚ := green_balls / total_balls

-- Theorem stating the probability of drawing a green ball
theorem green_ball_probability : prob_green_ball = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l145_14533


namespace NUMINAMATH_CALUDE_existence_of_even_floor_l145_14595

theorem existence_of_even_floor (n : ℕ) : ∃ k ∈ Finset.range (n + 1), Even (⌊(2 ^ (n + k) : ℝ) * Real.sqrt 2⌋) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_even_floor_l145_14595


namespace NUMINAMATH_CALUDE_average_difference_l145_14581

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 60) : 
  c - a = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l145_14581


namespace NUMINAMATH_CALUDE_distance_between_locations_l145_14561

/-- The distance between two locations given the speeds of two vehicles traveling towards each other and the time they take to meet. -/
theorem distance_between_locations (car_speed truck_speed : ℝ) (time : ℝ) : 
  car_speed > 0 → truck_speed > 0 → time > 0 →
  (car_speed + truck_speed) * time = 1925 → car_speed = 100 → truck_speed = 75 → time = 11 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l145_14561


namespace NUMINAMATH_CALUDE_hcf_problem_l145_14565

theorem hcf_problem (a b h : ℕ) (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b) :
  (Nat.gcd a b = h) →
  (∃ k : ℕ, Nat.lcm a b = 10 * 15 * k) →
  (max a b = 450) →
  h = 30 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l145_14565


namespace NUMINAMATH_CALUDE_one_black_one_white_probability_l145_14546

/-- The probability of picking one black ball and one white ball from a jar -/
theorem one_black_one_white_probability (black_balls white_balls : ℕ) : 
  black_balls = 5 → white_balls = 2 → 
  (black_balls * white_balls : ℚ) / ((black_balls + white_balls) * (black_balls + white_balls - 1) / 2) = 10/21 := by
sorry

end NUMINAMATH_CALUDE_one_black_one_white_probability_l145_14546


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l145_14568

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  trace_effect_to_cause : Bool
  start_from_inequality : Bool

/-- A condition in the context of inequality proofs -/
inductive Condition
  | necessary
  | sufficient
  | necessary_and_sufficient
  | necessary_or_sufficient

/-- The condition sought by the analysis method -/
def condition_sought (method : AnalysisMethod) : Condition :=
  Condition.sufficient

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition (method : AnalysisMethod) 
  (h1 : method.trace_effect_to_cause = true) 
  (h2 : method.start_from_inequality = true) : 
  condition_sought method = Condition.sufficient := by sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l145_14568


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l145_14504

theorem product_of_one_plus_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l145_14504


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l145_14511

theorem solve_exponential_equation :
  ∃ x : ℝ, (8 : ℝ) ^ (4 * x - 6) = (1 / 2 : ℝ) ^ (x + 5) ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l145_14511


namespace NUMINAMATH_CALUDE_youngest_child_age_l145_14536

def arithmetic_progression (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem youngest_child_age 
  (children : ℕ) 
  (ages : List ℕ) 
  (h_children : children = 8)
  (h_ages : ages = arithmetic_progression 2 3 7)
  : ages.head? = some 2 := by
  sorry

#eval arithmetic_progression 2 3 7

end NUMINAMATH_CALUDE_youngest_child_age_l145_14536


namespace NUMINAMATH_CALUDE_specific_pyramid_height_l145_14510

/-- Represents a right pyramid with a rectangular base -/
structure RightPyramid where
  basePerimeter : ℝ
  baseLength : ℝ
  baseBreadth : ℝ
  apexToVertexDistance : ℝ

/-- Calculates the height of a right pyramid -/
def pyramidHeight (p : RightPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  let p : RightPyramid := {
    basePerimeter := 40,
    baseLength := 40 / 3,
    baseBreadth := 20 / 3,
    apexToVertexDistance := 15
  }
  pyramidHeight p = 10 * Real.sqrt 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_height_l145_14510


namespace NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l145_14525

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
sorry

end NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l145_14525


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l145_14570

theorem range_of_2a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 2) (hb : 2 < b ∧ b < 3) :
  ∀ x, (∃ a b, (-2 < a ∧ a < 2) ∧ (2 < b ∧ b < 3) ∧ x = 2*a - b) ↔ -7 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l145_14570


namespace NUMINAMATH_CALUDE_power_of_64_l145_14520

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by
  have h1 : (64 : ℝ) = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l145_14520


namespace NUMINAMATH_CALUDE_floor_sufficiency_not_necessity_l145_14542

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_sufficiency_not_necessity :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) :=
by sorry

end NUMINAMATH_CALUDE_floor_sufficiency_not_necessity_l145_14542


namespace NUMINAMATH_CALUDE_function_inequality_constraint_l145_14523

theorem function_inequality_constraint (x a : ℝ) : 
  x > 0 → (2 * x + 1 > a * x) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_constraint_l145_14523


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l145_14505

theorem smallest_five_digit_multiple : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (15 ∣ n) ∧ (32 ∣ n) ∧ (9 ∣ n) ∧ (5 ∣ n) ∧ (54 ∣ n) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ 
    (15 ∣ m) ∧ (32 ∣ m) ∧ (9 ∣ m) ∧ (5 ∣ m) ∧ (54 ∣ m) → n ≤ m) ∧
  n = 17280 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l145_14505


namespace NUMINAMATH_CALUDE_direct_variation_theorem_y_value_at_negative_ten_l145_14588

/-- A function representing direct variation --/
def DirectVariation (k : ℝ) : ℝ → ℝ := fun x ↦ k * x

theorem direct_variation_theorem (k : ℝ) :
  (DirectVariation k 5 = 15) → (DirectVariation k (-10) = -30) := by
  sorry

/-- Main theorem proving the relationship between y and x --/
theorem y_value_at_negative_ten :
  ∃ k : ℝ, (DirectVariation k 5 = 15) ∧ (DirectVariation k (-10) = -30) := by
  sorry

end NUMINAMATH_CALUDE_direct_variation_theorem_y_value_at_negative_ten_l145_14588


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l145_14508

/-- The number of emails Jack received in different parts of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- Given information about Jack's email count -/
def jack_emails : EmailCount where
  morning := 5
  afternoon := 13 - 5
  evening := 72

/-- Theorem stating that Jack received 8 emails in the afternoon -/
theorem jack_afternoon_emails :
  jack_emails.afternoon = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l145_14508


namespace NUMINAMATH_CALUDE_total_highlighters_l145_14514

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 9) (h2 : yellow = 8) (h3 : blue = 5) :
  pink + yellow + blue = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l145_14514


namespace NUMINAMATH_CALUDE_right_angles_in_two_days_l145_14593

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_hand : ℕ)
  (minute_hand : ℕ)

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the number of right angles formed by clock hands in one day -/
def right_angles_per_day : ℕ := 44

/-- Checks if the clock hands form a right angle -/
def is_right_angle (c : Clock) : Prop :=
  (c.minute_hand - c.hour_hand) % 60 = 15 ∨ (c.hour_hand - c.minute_hand) % 60 = 15

/-- The main theorem: In 2 days, clock hands form a right angle 88 times -/
theorem right_angles_in_two_days :
  (2 * right_angles_per_day = 88) ∧
  (∀ t : ℕ, t < 2 * minutes_per_day →
    (∃ c : Clock, c.hour_hand = t % 720 ∧ c.minute_hand = t % 60 ∧
      is_right_angle c) ↔ t % (minutes_per_day / right_angles_per_day) = 0) :=
sorry

end NUMINAMATH_CALUDE_right_angles_in_two_days_l145_14593


namespace NUMINAMATH_CALUDE_distance_between_points_l145_14587

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-2, -3)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l145_14587


namespace NUMINAMATH_CALUDE_simplify_expression_l145_14563

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l145_14563


namespace NUMINAMATH_CALUDE_range_of_a_l145_14560

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (9 - 5*x) / 4 > 1 ∧ x < a

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x, inequality_system x a ↔ solution_set x) → a ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l145_14560


namespace NUMINAMATH_CALUDE_age_problem_l145_14547

/-- Given the ages of five people a, b, c, d, and e satisfying certain conditions,
    prove that b is 16 years old. -/
theorem age_problem (a b c d e : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = c / 2 →
  e = d - 3 →
  a + b + c + d + e = 52 →
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l145_14547


namespace NUMINAMATH_CALUDE_imma_fraction_is_83_125_l145_14539

/-- Represents the rose distribution problem --/
structure RoseDistribution where
  total_money : ℕ
  rose_price : ℕ
  roses_to_friends : ℕ
  jenna_fraction : ℚ

/-- Calculates the fraction of roses Imma receives --/
def imma_fraction (rd : RoseDistribution) : ℚ :=
  sorry

/-- Theorem stating the fraction of roses Imma receives --/
theorem imma_fraction_is_83_125 (rd : RoseDistribution) 
  (h1 : rd.total_money = 300)
  (h2 : rd.rose_price = 2)
  (h3 : rd.roses_to_friends = 125)
  (h4 : rd.jenna_fraction = 1/3) :
  imma_fraction rd = 83/125 :=
sorry

end NUMINAMATH_CALUDE_imma_fraction_is_83_125_l145_14539


namespace NUMINAMATH_CALUDE_original_price_l145_14500

-- Define the discount rate
def discount_rate : ℝ := 0.4

-- Define the discounted price
def discounted_price : ℝ := 120

-- Theorem stating the original price
theorem original_price : 
  ∃ (price : ℝ), price * (1 - discount_rate) = discounted_price ∧ price = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_l145_14500


namespace NUMINAMATH_CALUDE_pyramid_height_l145_14562

theorem pyramid_height (perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : perimeter = 32) (h_apex : apex_to_vertex = 12) :
  let side := perimeter / 4
  let half_diagonal := side * Real.sqrt 2 / 2
  let height := Real.sqrt (apex_to_vertex ^ 2 - half_diagonal ^ 2)
  height = 4 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_l145_14562


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l145_14501

/-- Represents a repeating decimal of the form 0.abab̄ab -/
def repeating_decimal_2 (a b : ℕ) : ℚ :=
  (100 * a + 10 * b + a + b : ℚ) / 9999

/-- Represents a repeating decimal of the form 0.abcabc̄abc -/
def repeating_decimal_3 (a b c : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c : ℚ) / 999999

/-- The main theorem stating that if the sum of the two repeating decimals
    equals 33/37, then abc must be 447 -/
theorem repeating_decimal_sum (a b c : ℕ) 
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10) 
  (h_sum : repeating_decimal_2 a b + repeating_decimal_3 a b c = 33/37) :
  100 * a + 10 * b + c = 447 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l145_14501


namespace NUMINAMATH_CALUDE_baseball_hits_theorem_l145_14512

def total_hits : ℕ := 50
def home_runs : ℕ := 3
def triples : ℕ := 2
def doubles : ℕ := 10

def singles : ℕ := total_hits - (home_runs + triples + doubles)

theorem baseball_hits_theorem :
  singles = 35 ∧ (singles : ℚ) / total_hits * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_baseball_hits_theorem_l145_14512


namespace NUMINAMATH_CALUDE_exists_node_not_on_line_l145_14578

/-- Represents a node on the grid --/
structure Node :=
  (x : Nat) (y : Nat)

/-- Represents a polygonal line on the grid --/
structure Line :=
  (nodes : List Node)

/-- The grid size --/
def gridSize : Nat := 100

/-- Checks if a node is on the boundary of the grid --/
def isOnBoundary (n : Node) : Bool :=
  n.x = 0 || n.x = gridSize || n.y = 0 || n.y = gridSize

/-- Checks if a node is a corner of the grid --/
def isCorner (n : Node) : Bool :=
  (n.x = 0 && n.y = 0) || (n.x = 0 && n.y = gridSize) ||
  (n.x = gridSize && n.y = 0) || (n.x = gridSize && n.y = gridSize)

/-- Theorem: There exists a non-corner node not on any line --/
theorem exists_node_not_on_line (lines : List Line) : 
  ∃ (n : Node), !isCorner n ∧ ∀ (l : Line), l ∈ lines → n ∉ l.nodes :=
sorry


end NUMINAMATH_CALUDE_exists_node_not_on_line_l145_14578


namespace NUMINAMATH_CALUDE_polynomial_not_equal_77_l145_14548

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_77_l145_14548


namespace NUMINAMATH_CALUDE_divide_600_in_ratio_1_2_l145_14521

def divide_in_ratio (total : ℚ) (ratio1 ratio2 : ℕ) : ℚ :=
  total * ratio1 / (ratio1 + ratio2)

theorem divide_600_in_ratio_1_2 :
  divide_in_ratio 600 1 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_divide_600_in_ratio_1_2_l145_14521


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l145_14585

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l145_14585


namespace NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l145_14580

theorem prime_divides_mn_minus_one (m n p : ℕ) 
  (h_m_pos : 0 < m) 
  (h_n_pos : 0 < n) 
  (h_p_prime : Nat.Prime p) 
  (h_m_lt_n : m < n) 
  (h_n_lt_p : n < p) 
  (h_p_div_m_sq : p ∣ (m^2 + 1)) 
  (h_p_div_n_sq : p ∣ (n^2 + 1)) : 
  p ∣ (m * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_mn_minus_one_l145_14580


namespace NUMINAMATH_CALUDE_no_solution_for_a_l145_14586

theorem no_solution_for_a (x : ℝ) (h : x = 4) :
  ¬∃ a : ℝ, a / (x + 4) + a / (x - 4) = a / (x - 4) :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_a_l145_14586


namespace NUMINAMATH_CALUDE_total_wheels_count_l145_14572

/-- Represents the total number of wheels in Jordan's driveway -/
def total_wheels : ℕ :=
  let cars := 2
  let car_wheels := 4
  let bikes := 3
  let bike_wheels := 2
  let trash_can_wheels := 2
  let tricycle_wheels := 3
  let roller_skate_wheels := 4
  let wheelchair_wheels := 6
  let wagon_wheels := 4
  
  cars * car_wheels +
  (bikes - 1) * bike_wheels + 1 +
  trash_can_wheels +
  tricycle_wheels +
  (roller_skate_wheels - 1) +
  wheelchair_wheels +
  wagon_wheels

theorem total_wheels_count : total_wheels = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l145_14572


namespace NUMINAMATH_CALUDE_product_of_roots_equals_32_l145_14566

theorem product_of_roots_equals_32 : 
  (256 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_equals_32_l145_14566


namespace NUMINAMATH_CALUDE_strictly_increasing_quadratic_function_l145_14507

theorem strictly_increasing_quadratic_function (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → (x^2 - a*x) < (y^2 - a*y)) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_quadratic_function_l145_14507


namespace NUMINAMATH_CALUDE_valid_XY_for_divisibility_by_72_l145_14526

/- Define a function to represent the number 42X4Y -/
def number (X Y : ℕ) : ℕ := 42000 + X * 100 + 40 + Y

/- Define the property of being a single digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/- Define the main theorem -/
theorem valid_XY_for_divisibility_by_72 :
  ∀ X Y : ℕ, 
  is_single_digit X → is_single_digit Y →
  (number X Y % 72 = 0 ↔ ((X = 8 ∧ Y = 0) ∨ (X = 0 ∧ Y = 8))) :=
by sorry

end NUMINAMATH_CALUDE_valid_XY_for_divisibility_by_72_l145_14526


namespace NUMINAMATH_CALUDE_g_range_l145_14502

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x ^ 4 + 3 * Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 4 * Real.sin x + 3 * Real.cos x ^ 2 - 9) /
  (Real.sin x - 1)

theorem g_range :
  ∀ x : ℝ, Real.sin x ≠ 1 → 2 ≤ g x ∧ g x < 15 := by
  sorry

end NUMINAMATH_CALUDE_g_range_l145_14502


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l145_14584

/-- Given that R varies directly as S and inversely as T, 
    prove that S = 20/3 when R = 5 and T = 1/3, 
    given that R = 2, T = 1/2, and S = 4 in another case. -/
theorem direct_inverse_variation (R S T : ℚ) : 
  (∃ k : ℚ, ∀ R S T, R = k * S / T) →  -- R varies directly as S and inversely as T
  (2 : ℚ) = (4 : ℚ) / (1/2 : ℚ) →      -- When R = 2, S = 4, and T = 1/2
  (5 : ℚ) = S / (1/3 : ℚ) →            -- When R = 5 and T = 1/3
  S = 20/3 := by
sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l145_14584


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l145_14553

/-- Represents the job titles in the school --/
inductive JobTitle
| Senior
| Intermediate
| Clerk

/-- Represents the school staff distribution --/
structure StaffDistribution where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  clerk : ℕ
  sum_eq_total : senior + intermediate + clerk = total

/-- Represents the sample distribution --/
structure SampleDistribution where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  clerk : ℕ
  sum_eq_total : senior + intermediate + clerk = total

/-- Checks if a sample distribution is correctly stratified --/
def isCorrectlySampled (staff : StaffDistribution) (sample : SampleDistribution) : Prop :=
  sample.senior * staff.total = staff.senior * sample.total ∧
  sample.intermediate * staff.total = staff.intermediate * sample.total ∧
  sample.clerk * staff.total = staff.clerk * sample.total

/-- The main theorem to prove --/
theorem stratified_sampling_correct 
  (staff : StaffDistribution)
  (sample : SampleDistribution)
  (h_staff : staff = { 
    total := 150, 
    senior := 45, 
    intermediate := 90, 
    clerk := 15, 
    sum_eq_total := by norm_num
  })
  (h_sample : sample = {
    total := 10,
    senior := 3,
    intermediate := 6,
    clerk := 1,
    sum_eq_total := by norm_num
  }) : 
  isCorrectlySampled staff sample := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l145_14553


namespace NUMINAMATH_CALUDE_rebus_solution_l145_14518

theorem rebus_solution :
  ∃! (A B G D V : ℕ),
    A * B + 8 = 3 * B ∧
    G * D + B = V ∧
    G * B + 3 = A * D ∧
    A = 2 ∧ B = 7 ∧ G = 1 ∧ D = 0 ∧ V = 15 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l145_14518


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l145_14506

/-- The function f(x) = (3 - x^2)e^x is monotonically increasing on the interval (-3, 1) -/
theorem monotonic_increasing_interval (x : ℝ) : 
  StrictMonoOn (fun x => (3 - x^2) * Real.exp x) (Set.Ioo (-3) 1) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l145_14506


namespace NUMINAMATH_CALUDE_ratio_sum_to_base_l145_14583

theorem ratio_sum_to_base (x y : ℝ) (h : y / x = 3 / 7) : (x + y) / x = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_base_l145_14583


namespace NUMINAMATH_CALUDE_coloring_existence_l145_14513

theorem coloring_existence : ∃ (f : ℕ → Bool), 
  ∀ (a : ℕ → ℕ) (d : ℕ),
    (∀ i : Fin 18, a i < a (i + 1)) →
    (∀ i j : Fin 18, a j - a i = d * (j - i)) →
    1 ≤ a 0 → a 17 ≤ 1986 →
    ∃ i j : Fin 18, f (a i) ≠ f (a j) := by
  sorry

end NUMINAMATH_CALUDE_coloring_existence_l145_14513


namespace NUMINAMATH_CALUDE_vince_monthly_savings_l145_14527

/-- Calculate Vince's monthly savings given the salon conditions --/
theorem vince_monthly_savings :
  let haircut_price : ℝ := 18
  let coloring_price : ℝ := 30
  let treatment_price : ℝ := 40
  let fixed_expenses : ℝ := 280
  let product_cost_per_customer : ℝ := 2
  let commission_rate : ℝ := 0.05
  let recreation_rate : ℝ := 0.20
  let haircut_customers : ℕ := 45
  let coloring_customers : ℕ := 25
  let treatment_customers : ℕ := 10

  let total_earnings : ℝ := 
    haircut_price * haircut_customers + 
    coloring_price * coloring_customers + 
    treatment_price * treatment_customers

  let total_customers : ℕ := 
    haircut_customers + coloring_customers + treatment_customers

  let variable_expenses : ℝ := 
    product_cost_per_customer * total_customers + 
    commission_rate * total_earnings

  let total_expenses : ℝ := fixed_expenses + variable_expenses

  let net_earnings : ℝ := total_earnings - total_expenses

  let recreation_amount : ℝ := recreation_rate * total_earnings

  let monthly_savings : ℝ := net_earnings - recreation_amount

  monthly_savings = 1030 := by
    sorry

end NUMINAMATH_CALUDE_vince_monthly_savings_l145_14527


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_over_32_l145_14551

theorem ten_thousandths_place_of_5_over_32 : 
  ∃ (a b c d : ℕ), (5 : ℚ) / 32 = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (6 : ℚ) / 10000 + (d : ℚ) / 100000 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_over_32_l145_14551
