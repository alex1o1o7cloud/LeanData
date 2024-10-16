import Mathlib

namespace NUMINAMATH_CALUDE_integer_pair_condition_l1286_128661

theorem integer_pair_condition (m n : ℕ+) :
  (∃ k : ℤ, (3 * n.val ^ 2 : ℚ) / m.val = k) ∧
  (∃ l : ℕ, (n.val ^ 2 + m.val : ℕ) = l ^ 2) →
  ∃ a : ℕ+, n = a ∧ m = 3 * a ^ 2 := by
sorry

end NUMINAMATH_CALUDE_integer_pair_condition_l1286_128661


namespace NUMINAMATH_CALUDE_divide_fractions_l1286_128696

theorem divide_fractions (a b c : ℚ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 4 / 7) : 
  c / a = 21 / 20 := by
sorry

end NUMINAMATH_CALUDE_divide_fractions_l1286_128696


namespace NUMINAMATH_CALUDE_tangent_segment_length_l1286_128662

theorem tangent_segment_length (r : ℝ) (a b : ℝ) : 
  r = 15 ∧ a = 6 ∧ b = 3 →
  ∃ x : ℝ, x = 12 ∧
    r^2 = x^2 + ((x + r - a - b) / 2)^2 ∧
    x + r = a + b + x + r - a - b :=
by sorry

end NUMINAMATH_CALUDE_tangent_segment_length_l1286_128662


namespace NUMINAMATH_CALUDE_coin_sum_bounds_l1286_128604

def coin_values : List ℕ := [1, 1, 1, 5, 10, 10, 25, 50]

theorem coin_sum_bounds (coins : List ℕ) (h : coins = coin_values) :
  (∃ (a b : ℕ), a ∈ coins ∧ b ∈ coins ∧ a + b = 2) ∧
  (∃ (c d : ℕ), c ∈ coins ∧ d ∈ coins ∧ c + d = 75) ∧
  (∀ (x y : ℕ), x ∈ coins → y ∈ coins → 2 ≤ x + y ∧ x + y ≤ 75) :=
by sorry

end NUMINAMATH_CALUDE_coin_sum_bounds_l1286_128604


namespace NUMINAMATH_CALUDE_tan_sum_identity_l1286_128659

theorem tan_sum_identity : 
  Real.tan (25 * π / 180) + Real.tan (35 * π / 180) + 
  Real.sqrt 3 * Real.tan (25 * π / 180) * Real.tan (35 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l1286_128659


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1286_128655

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | 12 * x^2 - a * x - a^2 < 0}
  if a > 0 then
    solution_set = {x : ℝ | -a/4 < x ∧ x < a/3}
  else if a = 0 then
    solution_set = ∅
  else
    solution_set = {x : ℝ | a/3 < x ∧ x < -a/4} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1286_128655


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1286_128699

theorem quadratic_equation_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (m - 1) * x₁^2 - 4 * x₁ + 1 = 0 ∧ 
   (m - 1) * x₂^2 - 4 * x₂ + 1 = 0) ↔ 
  (m < 5 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l1286_128699


namespace NUMINAMATH_CALUDE_three_fractions_l1286_128678

-- Define the list of expressions
def expressions : List String := [
  "3/a",
  "(a+b)/7",
  "x^2 + (1/2)y^2",
  "5",
  "1/(x-1)",
  "x/(8π)",
  "x^2/x"
]

-- Define what constitutes a fraction
def is_fraction (expr : String) : Prop :=
  ∃ (num denom : String), 
    expr = num ++ "/" ++ denom ∧ 
    denom ≠ "1" ∧
    ¬∃ (simplified : String), simplified ≠ expr ∧ ¬(∃ (n d : String), simplified = n ++ "/" ++ d)

-- Theorem stating that exactly 3 expressions are fractions
theorem three_fractions : 
  ∃ (fracs : List String), 
    fracs.length = 3 ∧ 
    (∀ expr ∈ fracs, expr ∈ expressions ∧ is_fraction expr) ∧
    (∀ expr ∈ expressions, is_fraction expr → expr ∈ fracs) :=
sorry

end NUMINAMATH_CALUDE_three_fractions_l1286_128678


namespace NUMINAMATH_CALUDE_complex_fraction_product_l1286_128677

theorem complex_fraction_product (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 : ℂ) + 7 * Complex.I = (a + b * Complex.I) * (2 - Complex.I) →
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l1286_128677


namespace NUMINAMATH_CALUDE_max_faces_limited_neighbor_tri_neighbor_is_tetrahedron_l1286_128649

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  edge_face_relation : edges = 2 * faces

/-- A convex polyhedron where each face has at most 4 neighboring faces. -/
structure LimitedNeighborPolyhedron extends ConvexPolyhedron where
  max_neighbors : edges ≤ 2 * faces

/-- A convex polyhedron where each face has exactly 3 neighboring faces. -/
structure TriNeighborPolyhedron extends ConvexPolyhedron where
  tri_neighbors : edges = 3 * faces / 2

/-- Theorem: The maximum number of faces in a LimitedNeighborPolyhedron is 6. -/
theorem max_faces_limited_neighbor (P : LimitedNeighborPolyhedron) : P.faces ≤ 6 := by
  sorry

/-- Theorem: A TriNeighborPolyhedron must be a tetrahedron (4 faces). -/
theorem tri_neighbor_is_tetrahedron (P : TriNeighborPolyhedron) : P.faces = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_faces_limited_neighbor_tri_neighbor_is_tetrahedron_l1286_128649


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l1286_128658

-- Define a right-angled triangle with integer side lengths
def RightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the perimeter constraint
def Perimeter (a b c : ℕ) : Prop :=
  a + b + c = 48

-- Define the area of a triangle
def Area (a b : ℕ) : ℕ :=
  a * b / 2

-- Theorem statement
theorem max_area_right_triangle :
  ∀ a b c : ℕ,
  RightTriangle a b c →
  Perimeter a b c →
  Area a b ≤ 288 :=
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l1286_128658


namespace NUMINAMATH_CALUDE_blackboard_problem_l1286_128687

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The operation of replacing a set of numbers with their sum modulo m -/
def replace_with_sum_mod (m : ℕ) (s : Finset ℕ) : ℕ := (s.sum id) % m

theorem blackboard_problem :
  ∀ (s : Finset ℕ),
  s.card = 2 →
  999 ∈ s →
  (∃ (t : Finset ℕ), t.card = 2004 ∧ Finset.range 2004 = t ∧
   replace_with_sum_mod 167 t = replace_with_sum_mod 167 s) →
  ∃ x, x ∈ s ∧ x ≠ 999 ∧ x = 3 := by
  sorry

#check blackboard_problem

end NUMINAMATH_CALUDE_blackboard_problem_l1286_128687


namespace NUMINAMATH_CALUDE_number_difference_l1286_128674

theorem number_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1286_128674


namespace NUMINAMATH_CALUDE_fault_line_movement_l1286_128624

/-- Fault line movement problem -/
theorem fault_line_movement 
  (total_movement : ℝ) 
  (past_year_movement : ℝ) 
  (h1 : total_movement = 6.5)
  (h2 : past_year_movement = 1.25) :
  total_movement - past_year_movement = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l1286_128624


namespace NUMINAMATH_CALUDE_polygon_count_l1286_128620

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The number of distinct convex polygons with 4 or more sides -/
def num_polygons : ℕ := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3)

theorem polygon_count :
  num_polygons = 32192 :=
sorry

end NUMINAMATH_CALUDE_polygon_count_l1286_128620


namespace NUMINAMATH_CALUDE_probability_of_defective_product_l1286_128685

/-- Given a set of products with some defective ones, calculate the probability of selecting a defective product -/
theorem probability_of_defective_product 
  (total : ℕ) 
  (defective : ℕ) 
  (h1 : total = 10) 
  (h2 : defective = 3) 
  (h3 : defective ≤ total) : 
  (defective : ℚ) / total = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_defective_product_l1286_128685


namespace NUMINAMATH_CALUDE_friday_work_proof_l1286_128647

/-- The time Mr. Willson worked on Friday in minutes -/
def friday_work_minutes : ℚ := 75

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

theorem friday_work_proof (monday : ℚ) (tuesday : ℚ) (wednesday : ℚ) (thursday : ℚ) 
  (h_monday : monday = 3/4)
  (h_tuesday : tuesday = 1/2)
  (h_wednesday : wednesday = 2/3)
  (h_thursday : thursday = 5/6)
  (h_total : monday + tuesday + wednesday + thursday + friday_work_minutes / 60 = 4) :
  friday_work_minutes = 75 := by
  sorry


end NUMINAMATH_CALUDE_friday_work_proof_l1286_128647


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l1286_128606

/-- The cubic equation coefficients -/
def a : ℝ := 3
def b : ℝ := 9
def c : ℝ := -135

/-- The cubic equation has a double root -/
def has_double_root (x y : ℝ) : Prop :=
  x = 2 * y ∨ y = 2 * x

/-- The value of k for which the statement holds -/
def k : ℝ := 525

/-- The main theorem -/
theorem cubic_equation_with_double_root :
  ∃ (x y : ℝ),
    a * x^3 + b * x^2 + c * x + k = 0 ∧
    a * y^3 + b * y^2 + c * y + k = 0 ∧
    has_double_root x y ∧
    k > 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l1286_128606


namespace NUMINAMATH_CALUDE_counterexample_exists_l1286_128690

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem counterexample_exists : ∃ n : ℕ, 
  ¬(is_prime n) ∧ ¬(is_prime (n - 5)) ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1286_128690


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_one_l1286_128642

theorem cube_root_of_a_plus_one (a : ℕ) (x : ℝ) (h : x ^ 2 = a) :
  (a + 1 : ℝ) ^ (1/3) = (x ^ 2 + 1) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_one_l1286_128642


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1286_128633

/-- A geometric sequence with common ratio q > 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 + a 6 = 33 →
  a 2 * a 5 = 32 →
  a 3 + a 8 = 132 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1286_128633


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1286_128635

theorem solve_linear_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1286_128635


namespace NUMINAMATH_CALUDE_union_eq_right_iff_complement_subset_l1286_128646

variable {U : Type*} -- Universal set
variable (A B : Set U) -- Sets A and B

theorem union_eq_right_iff_complement_subset :
  A ∪ B = B ↔ (Bᶜ : Set U) ⊆ (Aᶜ : Set U) := by sorry

end NUMINAMATH_CALUDE_union_eq_right_iff_complement_subset_l1286_128646


namespace NUMINAMATH_CALUDE_largest_A_value_l1286_128686

theorem largest_A_value : ∃ (A : ℝ),
  (∀ (x y : ℝ), x * y = 1 →
    ((x + y)^2 + 4) * ((x + y)^2 - 2) ≥ A * (x - y)^2) ∧
  (∀ (B : ℝ), (∀ (x y : ℝ), x * y = 1 →
    ((x + y)^2 + 4) * ((x + y)^2 - 2) ≥ B * (x - y)^2) → B ≤ A) ∧
  A = 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_A_value_l1286_128686


namespace NUMINAMATH_CALUDE_water_break_frequency_l1286_128615

theorem water_break_frequency
  (total_work_time : ℕ)
  (sitting_break_interval : ℕ)
  (water_break_excess : ℕ)
  (h1 : total_work_time = 240)
  (h2 : sitting_break_interval = 120)
  (h3 : water_break_excess = 10)
  : ℕ :=
  by
  -- Proof goes here
  sorry

#check water_break_frequency

end NUMINAMATH_CALUDE_water_break_frequency_l1286_128615


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1286_128697

theorem triangle_abc_properties (A B C : ℝ) (AB AC BC : ℝ) 
  (h_triangle : A + B + C = π)
  (h_AB : AB = 2)
  (h_AC : AC = 3)
  (h_BC : BC = Real.sqrt 7) : 
  A = π / 3 ∧ Real.cos (B - C) = 11 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1286_128697


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1286_128637

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_second_term 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1/4) 
  (h_a3a5 : a 3 * a 5 = 4 * (a 4 - 1)) :
  a 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1286_128637


namespace NUMINAMATH_CALUDE_no_answer_paradox_correct_answer_is_no_l1286_128630

/-- Represents the possible answers Alice can give to the Black Queen's question -/
inductive Answer
  | Yes
  | No

/-- Represents the possible outcomes of Alice's exam -/
inductive ExamResult
  | Pass
  | Fail

/-- Represents the Black Queen's judgment based on Alice's answer -/
def blackQueenJudgment (answer : Answer) : ExamResult → Prop :=
  match answer with
  | Answer.Yes => fun result => 
      (result = ExamResult.Pass → False) ∧ 
      (result = ExamResult.Fail → False)
  | Answer.No => fun result => 
      (result = ExamResult.Pass → False) ∧ 
      (result = ExamResult.Fail → False)

/-- Theorem stating that answering "No" creates an unresolvable paradox -/
theorem no_answer_paradox : 
  ∀ (result : ExamResult), blackQueenJudgment Answer.No result → False :=
by
  sorry

/-- Theorem stating that "No" is the correct answer to avoid failing the exam -/
theorem correct_answer_is_no : 
  ∀ (answer : Answer), 
    (∀ (result : ExamResult), blackQueenJudgment answer result → False) → 
    answer = Answer.No :=
by
  sorry

end NUMINAMATH_CALUDE_no_answer_paradox_correct_answer_is_no_l1286_128630


namespace NUMINAMATH_CALUDE_first_pass_bubble_sort_l1286_128640

def bubbleSortPass (list : List Int) : List Int :=
  list.zipWith (λ a b => if a > b then b else a) (list.drop 1 ++ [0])

theorem first_pass_bubble_sort :
  bubbleSortPass [8, 23, 12, 14, 39, 11] = [8, 12, 14, 23, 11, 39] := by
  sorry

end NUMINAMATH_CALUDE_first_pass_bubble_sort_l1286_128640


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l1286_128613

theorem unique_digit_divisibility : ∃! A : ℕ, A < 10 ∧ 41 % A = 0 ∧ (273100 + A * 10 + 8) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l1286_128613


namespace NUMINAMATH_CALUDE_max_squares_covered_proof_l1286_128682

/-- The side length of a checkerboard square in inches -/
def checkerboard_square_side : ℝ := 1.25

/-- The side length of the square card in inches -/
def card_side : ℝ := 1.75

/-- The maximum number of checkerboard squares that can be covered by the card -/
def max_squares_covered : ℕ := 9

/-- Theorem stating the maximum number of squares that can be covered by the card -/
theorem max_squares_covered_proof :
  ∀ (card_placement : ℝ × ℝ → Bool),
  (∃ (covered_squares : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ covered_squares →
      ∃ (x y : ℝ), 0 ≤ x ∧ x < card_side ∧ 0 ≤ y ∧ y < card_side ∧
        card_placement (x + i * checkerboard_square_side, y + j * checkerboard_square_side)) ∧
    covered_squares.card ≤ max_squares_covered) ∧
  (∃ (optimal_placement : ℝ × ℝ → Bool) (optimal_covered_squares : Finset (ℕ × ℕ)),
    (∀ (i j : ℕ), (i, j) ∈ optimal_covered_squares →
      ∃ (x y : ℝ), 0 ≤ x ∧ x < card_side ∧ 0 ≤ y ∧ y < card_side ∧
        optimal_placement (x + i * checkerboard_square_side, y + j * checkerboard_square_side)) ∧
    optimal_covered_squares.card = max_squares_covered) :=
by sorry

end NUMINAMATH_CALUDE_max_squares_covered_proof_l1286_128682


namespace NUMINAMATH_CALUDE_max_value_theorem_l1286_128665

theorem max_value_theorem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  3 * x * z * Real.sqrt 2 + 5 * x * y ≤ Real.sqrt 43 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1286_128665


namespace NUMINAMATH_CALUDE_larger_number_problem_l1286_128610

theorem larger_number_problem (x y : ℕ) : 
  x * y = 40 → x + y = 13 → max x y = 8 := by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1286_128610


namespace NUMINAMATH_CALUDE_total_balloons_is_eighteen_l1286_128609

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := fred_balloons + sam_balloons + mary_balloons

theorem total_balloons_is_eighteen : total_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_is_eighteen_l1286_128609


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1286_128629

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1286_128629


namespace NUMINAMATH_CALUDE_wire_division_l1286_128664

theorem wire_division (total_feet : ℕ) (total_inches : ℕ) (num_parts : ℕ) 
  (h1 : total_feet = 5) 
  (h2 : total_inches = 4) 
  (h3 : num_parts = 4) 
  (h4 : ∀ (feet : ℕ), feet * 12 = feet * (1 : ℕ) * 12) :
  (total_feet * 12 + total_inches) / num_parts = 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_division_l1286_128664


namespace NUMINAMATH_CALUDE_triangle_area_l1286_128669

theorem triangle_area (a b c A B C S : ℝ) : 
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a > 0 →
  a = 4 →
  A = π / 4 →
  B = π / 3 →
  S = (1 / 2) * a * b * Real.sin C →
  S = 6 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1286_128669


namespace NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l1286_128698

theorem rhombus_area_from_quadratic_roots : ∀ (d₁ d₂ : ℝ),
  d₁^2 - 10*d₁ + 24 = 0 →
  d₂^2 - 10*d₂ + 24 = 0 →
  d₁ ≠ d₂ →
  (1/2) * d₁ * d₂ = 12 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l1286_128698


namespace NUMINAMATH_CALUDE_not_in_E_iff_perfect_square_l1286_128617

/-- The set E of floor values of n + √n + 1/2 for natural numbers n -/
def E : Set ℕ := {m | ∃ n : ℕ, m = ⌊(n : ℝ) + Real.sqrt n + 1/2⌋}

/-- A positive integer m is not in set E if and only if it's a perfect square -/
theorem not_in_E_iff_perfect_square (m : ℕ) (hm : m > 0) : 
  m ∉ E ↔ ∃ k : ℕ, m = k^2 := by sorry

end NUMINAMATH_CALUDE_not_in_E_iff_perfect_square_l1286_128617


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1286_128689

theorem simplify_fraction_product : (2 / (2 + Real.sqrt 3)) * (2 / (2 - Real.sqrt 3)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1286_128689


namespace NUMINAMATH_CALUDE_marcella_shoes_l1286_128614

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

theorem marcella_shoes :
  max_remaining_pairs 20 9 = 11 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_l1286_128614


namespace NUMINAMATH_CALUDE_one_zero_in_interval_l1286_128621

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

theorem one_zero_in_interval (a : ℝ) (h : a > 3) :
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_zero_in_interval_l1286_128621


namespace NUMINAMATH_CALUDE_triangle_angle_value_l1286_128695

/-- Theorem: In a triangle with angles 40°, 3x, and x, the value of x is 35°. -/
theorem triangle_angle_value (x : ℝ) : 
  40 + 3 * x + x = 180 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l1286_128695


namespace NUMINAMATH_CALUDE_max_value_2sin_l1286_128619

theorem max_value_2sin (x : ℝ) : ∃ (M : ℝ), M = 2 ∧ ∀ y : ℝ, 2 * Real.sin x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_2sin_l1286_128619


namespace NUMINAMATH_CALUDE_room_length_proof_l1286_128693

theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 29925 →
  paving_rate = 900 →
  (total_cost / paving_rate) / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l1286_128693


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1286_128681

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_of_M_and_N : M ∩ N = {(1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1286_128681


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1286_128688

theorem quadratic_inequality_empty_solution (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1286_128688


namespace NUMINAMATH_CALUDE_three_inequality_propositions_l1286_128660

theorem three_inequality_propositions (a b c d : ℝ) :
  (∃ (f g h : Prop),
    (f = (a * b > 0)) ∧
    (g = (c / a > d / b)) ∧
    (h = (b * c > a * d)) ∧
    ((f ∧ g → h) ∧ (f ∧ h → g) ∧ (g ∧ h → f)) ∧
    (∀ (p q r : Prop),
      ((p = f ∨ p = g ∨ p = h) ∧
       (q = f ∨ q = g ∨ q = h) ∧
       (r = f ∨ r = g ∨ r = h) ∧
       (p ≠ q) ∧ (q ≠ r) ∧ (p ≠ r) ∧
       (p ∧ q → r)) →
      ((p = f ∧ q = g ∧ r = h) ∨
       (p = f ∧ q = h ∧ r = g) ∨
       (p = g ∧ q = h ∧ r = f)))) :=
by sorry

end NUMINAMATH_CALUDE_three_inequality_propositions_l1286_128660


namespace NUMINAMATH_CALUDE_sales_goals_calculation_l1286_128652

/-- Represents the sales data for a candy store employee over three days. -/
structure SalesData :=
  (jetBarGoal : ℕ)
  (zippyBarGoal : ℕ)
  (candyCloudGoal : ℕ)
  (mondayJetBars : ℕ)
  (mondayZippyBars : ℕ)
  (mondayCandyClouds : ℕ)
  (tuesdayJetBarsDiff : ℤ)
  (tuesdayZippyBarsDiff : ℕ)
  (wednesdayCandyCloudsMultiplier : ℕ)

/-- Calculates the remaining sales needed to reach the weekly goals. -/
def remainingSales (data : SalesData) : ℤ × ℤ × ℤ :=
  let totalJetBars := data.mondayJetBars + (data.mondayJetBars : ℤ) + data.tuesdayJetBarsDiff
  let totalZippyBars := data.mondayZippyBars + data.mondayZippyBars + data.tuesdayZippyBarsDiff
  let totalCandyClouds := data.mondayCandyClouds + data.mondayCandyClouds * data.wednesdayCandyCloudsMultiplier
  ((data.jetBarGoal : ℤ) - totalJetBars,
   (data.zippyBarGoal : ℤ) - (totalZippyBars : ℤ),
   (data.candyCloudGoal : ℤ) - (totalCandyClouds : ℤ))

theorem sales_goals_calculation (data : SalesData)
  (h1 : data.jetBarGoal = 90)
  (h2 : data.zippyBarGoal = 70)
  (h3 : data.candyCloudGoal = 50)
  (h4 : data.mondayJetBars = 45)
  (h5 : data.mondayZippyBars = 34)
  (h6 : data.mondayCandyClouds = 16)
  (h7 : data.tuesdayJetBarsDiff = -16)
  (h8 : data.tuesdayZippyBarsDiff = 8)
  (h9 : data.wednesdayCandyCloudsMultiplier = 2) :
  remainingSales data = (16, -6, 2) :=
by sorry


end NUMINAMATH_CALUDE_sales_goals_calculation_l1286_128652


namespace NUMINAMATH_CALUDE_common_rational_root_exists_l1286_128632

theorem common_rational_root_exists :
  ∃ (r : ℚ) (a b c d e f g : ℚ),
    (60 * r^4 + a * r^3 + b * r^2 + c * r + 20 = 0) ∧
    (20 * r^5 + d * r^4 + e * r^3 + f * r^2 + g * r + 60 = 0) ∧
    (r > 0) ∧
    (∀ n : ℤ, r ≠ n) ∧
    (r = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_common_rational_root_exists_l1286_128632


namespace NUMINAMATH_CALUDE_convoy_vehicles_specific_convoy_problem_l1286_128653

/-- Proves the number of vehicles in a convoy given specific conditions -/
theorem convoy_vehicles (bridge_length : ℕ) (convoy_speed : ℕ) (crossing_time : ℕ)
                        (vehicle_length : ℕ) (vehicle_gap : ℕ) : ℕ :=
  let total_distance := convoy_speed * crossing_time
  let convoy_length := total_distance - bridge_length
  let n := (convoy_length + vehicle_gap) / (vehicle_length + vehicle_gap)
  n

/-- The specific convoy problem -/
theorem specific_convoy_problem : convoy_vehicles 298 4 115 6 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_convoy_vehicles_specific_convoy_problem_l1286_128653


namespace NUMINAMATH_CALUDE_certain_number_proof_l1286_128607

theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, 213 * x = 340.8 ∧ x = 1.6 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1286_128607


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l1286_128612

theorem complex_arithmetic_expression : 
  ∃ ε > 0, ε < 0.0001 ∧ 
  |(3.5 / 0.7) * (5/3 : ℝ) + (7.2 / 0.36) - ((5/3 : ℝ) * 0.75 / 0.25) - 23.3335| < ε := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l1286_128612


namespace NUMINAMATH_CALUDE_ellipse_semi_major_axis_l1286_128627

theorem ellipse_semi_major_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →  -- Ellipse equation
  (m > 4) →                            -- Semi-major axis > Semi-minor axis
  (m = (2 : ℝ)^2 + 4) →                -- Relationship between a^2, b^2, and c^2
  (m = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_semi_major_axis_l1286_128627


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l1286_128600

theorem square_plus_inverse_square (x : ℝ) (h : x + (1/x) = 2) : x^2 + (1/x^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l1286_128600


namespace NUMINAMATH_CALUDE_tony_investment_rate_l1286_128663

/-- Calculates the investment rate given the investment amount and annual income. -/
def investment_rate (investment : ℚ) (annual_income : ℚ) : ℚ :=
  (annual_income / investment) * 100

/-- Proves that the investment rate is 7.8125% for the given scenario. -/
theorem tony_investment_rate :
  let investment := 3200
  let annual_income := 250
  investment_rate investment annual_income = 7.8125 := by
  sorry

end NUMINAMATH_CALUDE_tony_investment_rate_l1286_128663


namespace NUMINAMATH_CALUDE_shopping_trip_solution_l1286_128605

/-- The exchange rate from USD to CAD -/
def exchange_rate : ℚ := 8 / 5

/-- The amount spent in CAD -/
def amount_spent : ℕ := 80

/-- The function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating the solution to the problem -/
theorem shopping_trip_solution (d : ℕ) : 
  (exchange_rate * d - amount_spent = d) → sum_of_digits d = 7 := by
  sorry

#eval sum_of_digits 133  -- This should output 7

end NUMINAMATH_CALUDE_shopping_trip_solution_l1286_128605


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l1286_128668

/-- Contractor's daily wage problem -/
theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_pay : ℚ) :
  total_days = 30 →
  absent_days = 2 →
  fine_per_day = 15/2 →
  total_pay = 685 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - absent_days : ℚ) - fine_per_day * absent_days = total_pay ∧
    daily_wage = 25 :=
by sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l1286_128668


namespace NUMINAMATH_CALUDE_max_value_of_function_l1286_128634

theorem max_value_of_function (x : ℝ) (h : x < 1/3) :
  3 * x + 1 / (3 * x - 1) ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1286_128634


namespace NUMINAMATH_CALUDE_no_snow_no_rain_probability_l1286_128656

theorem no_snow_no_rain_probability 
  (prob_snow : ℚ) 
  (prob_rain : ℚ) 
  (days : ℕ) 
  (h1 : prob_snow = 2/3) 
  (h2 : prob_rain = 1/2) 
  (h3 : days = 5) : 
  (1 - prob_snow) * (1 - prob_rain) ^ days = 1/7776 :=
sorry

end NUMINAMATH_CALUDE_no_snow_no_rain_probability_l1286_128656


namespace NUMINAMATH_CALUDE_line_equation_k_l1286_128645

/-- Given a line passing through points (m, n) and (m + 2, n + 0.4), 
    with equation x = ky + 5, prove that k = 5 -/
theorem line_equation_k (m n : ℝ) : 
  let p : ℝ := 0.4
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + 2, n + p)
  let k : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_line_equation_k_l1286_128645


namespace NUMINAMATH_CALUDE_reciprocal_roots_condition_l1286_128648

/-- The quadratic equation 5x^2 + 7x + k = 0 has reciprocal roots if and only if k = 5 -/
theorem reciprocal_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 5 * x^2 + 7 * x + k = 0 ∧ 5 * y^2 + 7 * y + k = 0 ∧ x * y = 1) ↔ 
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_condition_l1286_128648


namespace NUMINAMATH_CALUDE_dress_original_price_l1286_128680

/-- The original price of a dress given shopping conditions --/
theorem dress_original_price (shoe_discount : ℚ) (dress_discount : ℚ) 
  (shoe_original_price : ℚ) (shoe_quantity : ℕ) (total_spent : ℚ) :
  shoe_discount = 40 / 100 →
  dress_discount = 20 / 100 →
  shoe_original_price = 50 →
  shoe_quantity = 2 →
  total_spent = 140 →
  ∃ (dress_original_price : ℚ),
    dress_original_price = 100 ∧
    total_spent = shoe_quantity * (shoe_original_price * (1 - shoe_discount)) +
                  dress_original_price * (1 - dress_discount) :=
by sorry

end NUMINAMATH_CALUDE_dress_original_price_l1286_128680


namespace NUMINAMATH_CALUDE_area_ratio_in_special_triangle_l1286_128694

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define the properties of the triangle and point D
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

def on_side (D A C : ℝ × ℝ) : Prop := sorry

def angle_measure (B D C : ℝ × ℝ) : ℝ := sorry

def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_in_special_triangle 
  (h_equilateral : is_equilateral A B C)
  (h_on_side : on_side D A C)
  (h_angle : angle_measure B D C = 30) :
  triangle_area A D B / triangle_area C D B = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_in_special_triangle_l1286_128694


namespace NUMINAMATH_CALUDE_cube_power_inequality_l1286_128628

theorem cube_power_inequality (a b c : ℕ+) :
  (a^(a:ℕ) * b^(b:ℕ) * c^(c:ℕ))^3 ≥ (a*b*c)^((a:ℕ)+(b:ℕ)+(c:ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_cube_power_inequality_l1286_128628


namespace NUMINAMATH_CALUDE_range_of_a_l1286_128625

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x + a * (y - 2 * Real.exp 1 * x) * (Real.log y - Real.log x) = 0) : 
  a < 0 ∨ a ≥ 2 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1286_128625


namespace NUMINAMATH_CALUDE_relationship_xyz_l1286_128667

theorem relationship_xyz (x y z : ℝ) 
  (h1 : x - y > x + z) 
  (h2 : x + y < y + z) : 
  y < -z ∧ x < z := by
sorry

end NUMINAMATH_CALUDE_relationship_xyz_l1286_128667


namespace NUMINAMATH_CALUDE_perimeter_pedal_ratio_l1286_128608

/-- A triangle in a 2D plane -/
structure Triangle where
  -- Define the triangle structure (you may need to adjust this based on your specific needs)
  -- For example, you could define it using three points or side lengths

/-- The pedal triangle of a given triangle -/
def pedal_triangle (t : Triangle) : Triangle :=
  sorry -- Definition of pedal triangle

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  sorry -- Definition of perimeter

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ :=
  sorry -- Definition of circumradius

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ :=
  sorry -- Definition of inradius

/-- Theorem: The ratio of a triangle's perimeter to its pedal triangle's perimeter
    is equal to the ratio of its circumradius to its inradius -/
theorem perimeter_pedal_ratio (t : Triangle) :
  (perimeter t) / (perimeter (pedal_triangle t)) = (circumradius t) / (inradius t) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_pedal_ratio_l1286_128608


namespace NUMINAMATH_CALUDE_max_value_sum_of_sines_l1286_128603

open Real

theorem max_value_sum_of_sines :
  ∃ (x : ℝ), ∀ (y : ℝ), sin y + sin (y - π/3) ≤ sqrt 3 ∧
  sin x + sin (x - π/3) = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_of_sines_l1286_128603


namespace NUMINAMATH_CALUDE_percent_of_number_l1286_128657

theorem percent_of_number : (25 : ℝ) / 100 * 280 = 70 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_l1286_128657


namespace NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_measure_l1286_128672

-- Part 1
theorem triangle_side_length (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = π / 6 →
  C = 3 * π / 4 →
  a = Real.sqrt 6 - Real.sqrt 2 :=
sorry

-- Part 2
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  S = (1 / 4) * (a^2 + b^2 - c^2) →
  C = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_triangle_angle_measure_l1286_128672


namespace NUMINAMATH_CALUDE_sin_1050_degrees_l1286_128644

theorem sin_1050_degrees : Real.sin (1050 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1050_degrees_l1286_128644


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l1286_128673

def M : ℕ := 33 * 38 * 58 * 462

/-- The sum of odd divisors of a natural number n -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- The sum of even divisors of a natural number n -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 14 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l1286_128673


namespace NUMINAMATH_CALUDE_students_walking_home_l1286_128639

theorem students_walking_home (total : ℚ) (bus carpool scooter walk : ℚ) : 
  bus = 1/3 * total →
  carpool = 1/5 * total →
  scooter = 1/8 * total →
  walk = total - (bus + carpool + scooter) →
  walk = 41/120 * total := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l1286_128639


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l1286_128622

/-- Represents the total number of seats in an airplane -/
def total_seats : ℕ := 180

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Represents the fraction of total seats in Business Class -/
def business_class_fraction : ℚ := 1/5

/-- Represents the fraction of total seats in Economy Class -/
def economy_class_fraction : ℚ := 3/5

/-- Theorem stating that the total number of seats is correct given the conditions -/
theorem airplane_seats_theorem :
  (first_class_seats : ℚ) + 
  business_class_fraction * total_seats + 
  economy_class_fraction * total_seats = total_seats := by sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l1286_128622


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l1286_128651

/-- Given an amount of simple interest, time period, and rate, prove that the rate percent is correct. -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (rate : ℝ) 
  (h1 : interest = 400) 
  (h2 : time = 4) 
  (h3 : rate = 0.1) : 
  rate * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_simple_interest_rate_percent_l1286_128651


namespace NUMINAMATH_CALUDE_problem_statement_l1286_128691

theorem problem_statement :
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧
  (¬ ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ Real.sin x > x) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1286_128691


namespace NUMINAMATH_CALUDE_best_fit_model_l1286_128638

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |1 - model.r_squared| ≤ |1 - m.r_squared|

theorem best_fit_model :
  let models : List RegressionModel := [
    ⟨"Model 1", 0.25⟩,
    ⟨"Model 2", 0.50⟩,
    ⟨"Model 3", 0.98⟩,
    ⟨"Model 4", 0.80⟩
  ]
  let model3 : RegressionModel := ⟨"Model 3", 0.98⟩
  has_best_fit model3 models := by sorry

end NUMINAMATH_CALUDE_best_fit_model_l1286_128638


namespace NUMINAMATH_CALUDE_hotline_probabilities_l1286_128611

theorem hotline_probabilities (p1 p2 p3 p4 : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p4 = 0.35) :
  (p1 + p2 + p3 + p4 = 0.95) ∧ (1 - (p1 + p2 + p3 + p4) = 0.05) := by
  sorry

end NUMINAMATH_CALUDE_hotline_probabilities_l1286_128611


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1286_128666

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 2*a - 5 = 0) → (b^2 + 2*b - 5 = 0) → (a^2 + a*b + 2*a = 0) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1286_128666


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l1286_128679

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 47)
  (h3 : max_ac_no_stripes = 47)
  (h4 : cars_without_ac ≤ total_cars)
  (h5 : max_ac_no_stripes ≤ total_cars - cars_without_ac) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = total_cars - cars_without_ac - max_ac_no_stripes ∧ 
    min_cars_with_stripes = 6 :=
by
  sorry

#check min_cars_with_racing_stripes

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l1286_128679


namespace NUMINAMATH_CALUDE_unique_fraction_condition_l1286_128601

def is_simplest_proper_fraction (n d : ℤ) : Prop :=
  0 < n ∧ n < d ∧ Nat.gcd n.natAbs d.natAbs = 1

def is_improper_fraction (n d : ℤ) : Prop :=
  n ≥ d

theorem unique_fraction_condition (x : ℤ) : 
  (is_simplest_proper_fraction x 8 ∧ is_improper_fraction x 6) ↔ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_condition_l1286_128601


namespace NUMINAMATH_CALUDE_team_incorrect_answers_contest_result_l1286_128631

theorem team_incorrect_answers 
  (total_questions : Nat) 
  (riley_incorrect : Nat) 
  (ofelia_correct_addition : Nat) : Nat :=
  let riley_correct := total_questions - riley_incorrect
  let ofelia_correct := riley_correct / 2 + ofelia_correct_addition
  let ofelia_incorrect := total_questions - ofelia_correct
  riley_incorrect + ofelia_incorrect

#check @team_incorrect_answers

theorem contest_result : 
  team_incorrect_answers 35 3 5 = 17 := by
  sorry

#check @contest_result

end NUMINAMATH_CALUDE_team_incorrect_answers_contest_result_l1286_128631


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_third_l1286_128650

theorem at_least_one_not_greater_than_third (a b c : ℝ) (h : a + b + c = 1) :
  min a (min b c) ≤ 1/3 := by sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_third_l1286_128650


namespace NUMINAMATH_CALUDE_max_books_with_23_dollars_l1286_128636

/-- Represents the available book purchasing options -/
inductive BookOption
  | Single
  | Set4
  | Set7

/-- Returns the cost of a given book option -/
def cost (option : BookOption) : ℕ :=
  match option with
  | BookOption.Single => 2
  | BookOption.Set4 => 7
  | BookOption.Set7 => 12

/-- Returns the number of books in a given book option -/
def books (option : BookOption) : ℕ :=
  match option with
  | BookOption.Single => 1
  | BookOption.Set4 => 4
  | BookOption.Set7 => 7

/-- Represents a combination of book purchases -/
structure Purchase where
  singles : ℕ
  sets4 : ℕ
  sets7 : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * cost BookOption.Single +
  p.sets4 * cost BookOption.Set4 +
  p.sets7 * cost BookOption.Set7

/-- Calculates the total number of books in a purchase -/
def totalBooks (p : Purchase) : ℕ :=
  p.singles * books BookOption.Single +
  p.sets4 * books BookOption.Set4 +
  p.sets7 * books BookOption.Set7

/-- Theorem: The maximum number of books that can be purchased with $23 is 13 -/
theorem max_books_with_23_dollars :
  ∃ (p : Purchase), totalCost p ≤ 23 ∧
  totalBooks p = 13 ∧
  ∀ (q : Purchase), totalCost q ≤ 23 → totalBooks q ≤ 13 := by
  sorry


end NUMINAMATH_CALUDE_max_books_with_23_dollars_l1286_128636


namespace NUMINAMATH_CALUDE_part_1_part_2_part_3_part_3_unique_l1286_128670

-- Define the algebraic expression
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Part 1
theorem part_1 : f 2 = -1 := by sorry

-- Part 2
theorem part_2 : 
  ∃ x₁ x₂ : ℝ, f x₁ = 4 ∧ f x₂ = 4 ∧ x₁^2 + x₂^2 = 18 := by sorry

-- Part 3
theorem part_3 :
  ∃ m : ℕ, 
    (∃ n₁ n₂ : ℝ, 
      f (m + 2) = n₁ ∧ 
      f (2*m + 1) = n₂ ∧ 
      ∃ k : ℤ, n₂ / n₁ = k) ∧
    m = 3 := by sorry

-- Additional theorem to show the uniqueness of m in part 3
theorem part_3_unique :
  ∀ m : ℕ, 
    (∃ n₁ n₂ : ℝ, 
      f (m + 2) = n₁ ∧ 
      f (2*m + 1) = n₂ ∧ 
      ∃ k : ℤ, n₂ / n₁ = k) →
    m = 3 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_part_3_part_3_unique_l1286_128670


namespace NUMINAMATH_CALUDE_man_double_son_age_l1286_128618

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2 -/
theorem man_double_son_age 
  (son_age : ℕ) 
  (age_difference : ℕ) 
  (h1 : son_age = 18) 
  (h2 : age_difference = 20) : 
  years_until_double_age son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_double_son_age_l1286_128618


namespace NUMINAMATH_CALUDE_total_cleaner_needed_l1286_128623

def cleaner_per_dog : ℕ := 6
def cleaner_per_cat : ℕ := 4
def cleaner_per_rabbit : ℕ := 1

def num_dogs : ℕ := 6
def num_cats : ℕ := 3
def num_rabbits : ℕ := 1

theorem total_cleaner_needed :
  cleaner_per_dog * num_dogs + cleaner_per_cat * num_cats + cleaner_per_rabbit * num_rabbits = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cleaner_needed_l1286_128623


namespace NUMINAMATH_CALUDE_not_perfect_square_l1286_128643

theorem not_perfect_square (n : ℕ) (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 
  ¬ ∃ k : ℕ, a * 10^(n+1) + 9 = k^2 :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1286_128643


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1286_128641

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 2 * i) / (2 + 5 * i) = (-4 : ℝ) / 29 - (19 : ℝ) / 29 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1286_128641


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l1286_128683

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedronVolume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 12

/-- Theorem: The volume ratio of two regular tetrahedrons with edge lengths a and 2a is 1:8 -/
theorem tetrahedron_volume_ratio (a : ℝ) (h : a > 0) :
  tetrahedronVolume (2 * a) / tetrahedronVolume a = 8 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l1286_128683


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_in_6_minutes_l1286_128616

/-- Represents a playlist of songs with increasing durations -/
structure Playlist where
  num_songs : ℕ
  duration_increment : ℕ
  shortest_duration : ℕ
  favorite_duration : ℕ

/-- Calculates the probability of not hearing the entire favorite song 
    within a given time limit -/
def probability_not_hearing_favorite (p : Playlist) (time_limit : ℕ) : ℚ :=
  sorry

/-- The specific playlist described in the problem -/
def marcel_playlist : Playlist :=
  { num_songs := 12
  , duration_increment := 30
  , shortest_duration := 60
  , favorite_duration := 300 }

/-- The main theorem to prove -/
theorem probability_not_hearing_favorite_in_6_minutes :
  probability_not_hearing_favorite marcel_playlist 360 = 1813 / 1980 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_in_6_minutes_l1286_128616


namespace NUMINAMATH_CALUDE_three_kopeck_count_l1286_128626

/-- Represents the denomination of a coin -/
inductive Denomination
| One
| Two
| Three

/-- Represents a row of coins -/
def CoinRow := List Denomination

/-- Checks if there's at least one coin between any two one-kopeck coins -/
def validOneKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Checks if there's at least two coins between any two two-kopeck coins -/
def validTwoKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Checks if there's at least three coins between any two three-kopeck coins -/
def validThreeKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Counts the number of three-kopeck coins in the row -/
def countThreeKopecks (row : CoinRow) : Nat := sorry

theorem three_kopeck_count (row : CoinRow) :
  row.length = 101 →
  validOneKopeckSpacing row →
  validTwoKopeckSpacing row →
  validThreeKopeckSpacing row →
  (countThreeKopecks row = 25 ∨ countThreeKopecks row = 26) :=
by sorry

end NUMINAMATH_CALUDE_three_kopeck_count_l1286_128626


namespace NUMINAMATH_CALUDE_john_paintball_cost_l1286_128692

/-- John's monthly expenditure on paintballs -/
def monthly_paintball_cost (plays_per_month : ℕ) (boxes_per_play : ℕ) (cost_per_box : ℕ) : ℕ :=
  plays_per_month * boxes_per_play * cost_per_box

/-- Theorem: John spends $225 a month on paintballs -/
theorem john_paintball_cost :
  monthly_paintball_cost 3 3 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_paintball_cost_l1286_128692


namespace NUMINAMATH_CALUDE_gridiron_club_members_l1286_128675

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a helmet in dollars -/
def helmet_cost : ℕ := 2 * tshirt_cost

/-- The cost of equipment for one member in dollars -/
def member_cost : ℕ := sock_cost + tshirt_cost + helmet_cost

/-- The total expenditure for all members in dollars -/
def total_expenditure : ℕ := 4680

/-- The number of members in the club -/
def club_members : ℕ := total_expenditure / member_cost

theorem gridiron_club_members :
  club_members = 104 :=
sorry

end NUMINAMATH_CALUDE_gridiron_club_members_l1286_128675


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1286_128602

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1286_128602


namespace NUMINAMATH_CALUDE_monkey_swing_theorem_l1286_128654

/-- The distance a monkey swings in a given time -/
def monkey_swing_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 60

/-- Theorem: A monkey swinging at 1.2 m/s for 30 minutes travels 2160 meters -/
theorem monkey_swing_theorem :
  monkey_swing_distance 1.2 30 = 2160 :=
by sorry

end NUMINAMATH_CALUDE_monkey_swing_theorem_l1286_128654


namespace NUMINAMATH_CALUDE_total_games_is_32_l1286_128676

/-- The number of games won by Jerry -/
def jerry_wins : ℕ := 7

/-- The number of games won by Dave -/
def dave_wins : ℕ := jerry_wins + 3

/-- The number of games won by Ken -/
def ken_wins : ℕ := dave_wins + 5

/-- The total number of games played -/
def total_games : ℕ := jerry_wins + dave_wins + ken_wins

theorem total_games_is_32 : total_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_32_l1286_128676


namespace NUMINAMATH_CALUDE_dot_product_CA_CB_l1286_128684

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Define the center of the circle C
def center_C : ℝ × ℝ := (0, 2)

-- Define a point B on the circle C
def point_B : ℝ × ℝ := sorry

-- State that line l is tangent to circle C at point B
axiom tangent_line : (point_B.1 - point_A.1) * (point_B.1 - center_C.1) + 
                     (point_B.2 - point_A.2) * (point_B.2 - center_C.2) = 0

-- The main theorem
theorem dot_product_CA_CB : 
  (point_A.1 - center_C.1) * (point_B.1 - center_C.1) + 
  (point_A.2 - center_C.2) * (point_B.2 - center_C.2) = 5 :=
sorry

end NUMINAMATH_CALUDE_dot_product_CA_CB_l1286_128684


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1286_128671

theorem sum_of_squares_and_products (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 52)
  (sum_of_products : x*y + y*z + z*x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1286_128671
