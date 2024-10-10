import Mathlib

namespace absolute_value_solution_set_l3150_315070

theorem absolute_value_solution_set (a b : ℝ) : 
  (∀ x, |x - a| < b ↔ 2 < x ∧ x < 4) → a - b = 2 := by
  sorry

end absolute_value_solution_set_l3150_315070


namespace min_marked_cells_l3150_315065

/-- Represents a marked cell on the board -/
structure MarkedCell :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a board with marked cells -/
structure Board :=
  (size : ℕ)
  (marked_cells : List MarkedCell)

/-- Checks if a sub-board contains a marked cell on both diagonals -/
def subBoardContainsMarkedDiagonals (board : Board) (m : ℕ) (topLeft : MarkedCell) : Prop :=
  ∃ (c1 c2 : MarkedCell),
    c1 ∈ board.marked_cells ∧
    c2 ∈ board.marked_cells ∧
    c1.row - topLeft.row = c1.col - topLeft.col ∧
    c2.row - topLeft.row = topLeft.col + m - 1 - c2.col ∧
    c1.row ≥ topLeft.row ∧ c1.row < topLeft.row + m ∧
    c1.col ≥ topLeft.col ∧ c1.col < topLeft.col + m ∧
    c2.row ≥ topLeft.row ∧ c2.row < topLeft.row + m ∧
    c2.col ≥ topLeft.col ∧ c2.col < topLeft.col + m

/-- The main theorem stating the minimum number of marked cells -/
theorem min_marked_cells (n : ℕ) :
  ∃ (board : Board),
    board.size = n ∧
    board.marked_cells.length = n ∧
    (∀ (m : ℕ) (topLeft : MarkedCell),
      m > n / 2 →
      topLeft.row + m ≤ n →
      topLeft.col + m ≤ n →
      subBoardContainsMarkedDiagonals board m topLeft) ∧
    (∀ (board' : Board),
      board'.size = n →
      board'.marked_cells.length < n →
      ∃ (m : ℕ) (topLeft : MarkedCell),
        m > n / 2 ∧
        topLeft.row + m ≤ n ∧
        topLeft.col + m ≤ n ∧
        ¬subBoardContainsMarkedDiagonals board' m topLeft) := by
  sorry

end min_marked_cells_l3150_315065


namespace simplify_expression_l3150_315038

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (97 * x + 45) = 100 * x + 60 := by
  sorry

end simplify_expression_l3150_315038


namespace geometric_sequence_fifth_term_l3150_315083

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
    (a : ℕ → ℝ)
    (h_geometric : IsGeometricSequence a)
    (h_sum1 : a 1 + a 2 = 4)
    (h_sum2 : a 2 + a 3 = 12) :
    a 5 = 81 := by
  sorry

end geometric_sequence_fifth_term_l3150_315083


namespace max_player_salary_l3150_315069

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 25 →
  min_salary = 15000 →
  total_cap = 800000 →
  (n - 1) * min_salary + (total_cap - (n - 1) * min_salary) = 440000 :=
by sorry

end max_player_salary_l3150_315069


namespace exam_venue_problem_l3150_315058

/-- Given a group of students, calculates the number not good at either of two subjects. -/
def students_not_good_at_either (total : ℕ) (good_at_english : ℕ) (good_at_chinese : ℕ) (good_at_both : ℕ) : ℕ :=
  total - (good_at_english + good_at_chinese - good_at_both)

/-- Proves that in a group of 45 students, if 35 are good at English, 31 are good at Chinese,
    and 24 are good at both, then 3 students are not good at either subject. -/
theorem exam_venue_problem :
  students_not_good_at_either 45 35 31 24 = 3 := by
  sorry

end exam_venue_problem_l3150_315058


namespace complex_equality_l3150_315040

theorem complex_equality (z : ℂ) : z = -1 + I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end complex_equality_l3150_315040


namespace prob_monochromatic_triangle_in_hexagon_l3150_315059

/-- A regular hexagon with randomly colored edges -/
structure ColoredHexagon where
  /-- The number of sides in a regular hexagon -/
  numSides : Nat
  /-- The number of diagonals in a regular hexagon -/
  numDiagonals : Nat
  /-- The total number of edges (sides + diagonals) -/
  numEdges : Nat
  /-- The number of possible triangles in a hexagon -/
  numTriangles : Nat
  /-- The probability of an edge being a specific color -/
  probEdgeColor : ℚ
  /-- The probability of a triangle not being monochromatic -/
  probNonMonochromatic : ℚ

/-- The probability of having at least one monochromatic triangle in a colored hexagon -/
def probMonochromaticTriangle (h : ColoredHexagon) : ℚ :=
  1 - (h.probNonMonochromatic ^ h.numTriangles)

/-- Theorem stating the probability of a monochromatic triangle in a randomly colored hexagon -/
theorem prob_monochromatic_triangle_in_hexagon :
  ∃ (h : ColoredHexagon),
    h.numSides = 6 ∧
    h.numDiagonals = 9 ∧
    h.numEdges = 15 ∧
    h.numTriangles = 20 ∧
    h.probEdgeColor = 1/2 ∧
    h.probNonMonochromatic = 3/4 ∧
    probMonochromaticTriangle h = 253/256 := by
  sorry

end prob_monochromatic_triangle_in_hexagon_l3150_315059


namespace union_of_A_and_B_l3150_315008

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 3*x + a = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + b = 0}

-- State the theorem
theorem union_of_A_and_B (a b : ℝ) :
  (∃ (x : ℝ), A a ∩ B b = {x}) →
  (∃ (y z : ℝ), A a ∪ B b = {y, z, 2}) :=
sorry

end union_of_A_and_B_l3150_315008


namespace negation_of_existence_is_forall_l3150_315090

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0) := by
  sorry

end negation_of_existence_is_forall_l3150_315090


namespace probability_colored_ball_l3150_315054

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of blue balls
def blue_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Theorem: The probability of drawing a colored ball is 7/10
theorem probability_colored_ball :
  (red_balls + blue_balls : ℚ) / total_balls = 7 / 10 := by
  sorry

end probability_colored_ball_l3150_315054


namespace inequality_proof_l3150_315021

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1)^2 / b + (b + 1)^2 / a ≥ 8 := by
  sorry

end inequality_proof_l3150_315021


namespace average_headcount_l3150_315002

def spring_05_06 : ℕ := 11200
def fall_05_06 : ℕ := 11100
def spring_06_07 : ℕ := 10800
def fall_06_07 : ℕ := 11000  -- approximated due to report error

def total_headcount : ℕ := spring_05_06 + fall_05_06 + spring_06_07 + fall_06_07
def num_terms : ℕ := 4

theorem average_headcount : 
  (total_headcount : ℚ) / num_terms = 11025 := by sorry

end average_headcount_l3150_315002


namespace seminar_selection_l3150_315066

theorem seminar_selection (boys girls : ℕ) (total_select : ℕ) : 
  boys = 4 → girls = 3 → total_select = 4 →
  (Nat.choose (boys + girls) total_select) - (Nat.choose boys total_select) = 34 := by
sorry

end seminar_selection_l3150_315066


namespace coexisting_expression_coexisting_negation_l3150_315082

/-- Definition of coexisting rational number pairs -/
def is_coexisting (a b : ℚ) : Prop := a * b = a - b - 1

/-- Theorem 1: For coexisting pairs, the given expression equals 1/2 -/
theorem coexisting_expression (a b : ℚ) (h : is_coexisting a b) :
  3 * a * b - a + (1/2) * (a + b - 5 * a * b) + 1 = 1/2 := by sorry

/-- Theorem 2: If (a,b) is coexisting, then (-b,-a) is also coexisting -/
theorem coexisting_negation (a b : ℚ) (h : is_coexisting a b) :
  is_coexisting (-b) (-a) := by sorry

end coexisting_expression_coexisting_negation_l3150_315082


namespace subset_ratio_eight_elements_l3150_315062

theorem subset_ratio_eight_elements : 
  let n : ℕ := 8
  let total_subsets : ℕ := 2^n
  let three_elem_subsets : ℕ := n.choose 3
  (three_elem_subsets : ℚ) / total_subsets = 7/32 := by sorry

end subset_ratio_eight_elements_l3150_315062


namespace probability_all_digits_different_l3150_315087

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

def count_three_digit_numbers : ℕ := 999 - 100 + 1

def count_three_digit_same_digits : ℕ := 9

theorem probability_all_digits_different :
  (count_three_digit_numbers - count_three_digit_same_digits : ℚ) / count_three_digit_numbers = 99/100 :=
sorry

end probability_all_digits_different_l3150_315087


namespace equation_one_solutions_equation_two_solution_l3150_315017

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  5 * x^2 = 15 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 3)^3 = -64 ↔ x = -7 := by sorry

end equation_one_solutions_equation_two_solution_l3150_315017


namespace common_root_is_negative_one_l3150_315067

/-- Given two equations with a common root and a condition on the coefficients,
    prove that the common root is -1. -/
theorem common_root_is_negative_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ x : ℝ, x^2 + a*x + b = 0 ∧ x^3 + b*x + a = 0 → x = -1 := by
  sorry

#check common_root_is_negative_one

end common_root_is_negative_one_l3150_315067


namespace max_a_for_function_inequality_l3150_315036

theorem max_a_for_function_inequality (f : ℝ → ℝ) (h : ∀ x, x ∈ [3, 5] → f x = 2 * x / (x - 1)) :
  (∃ a : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ a) ∧ 
   (∀ b : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ b) → b ≤ a)) →
  (∃ a : ℝ, a = 5/2 ∧ 
   (∀ x, x ∈ [3, 5] → f x ≥ a) ∧
   (∀ b : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ b) → b ≤ a)) :=
by sorry

end max_a_for_function_inequality_l3150_315036


namespace solution_characterization_l3150_315092

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1/3, 1/3, 1/3), (0, 0, 1), (2/3, -1/3, 2/3), (0, 1, 0), (1, 0, 0), (-1, 1, 1)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x + y + z = 1 ∧
  x^2*y + y^2*z + z^2*x = x*y^2 + y*z^2 + z*x^2 ∧
  x^3 + y^2 + z = y^3 + z^2 + x

theorem solution_characterization :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end solution_characterization_l3150_315092


namespace circles_common_chord_l3150_315004

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y - 40 = 0

-- Define the line
def common_chord (x y : ℝ) : Prop := x + 3*y - 10 = 0

-- Theorem statement
theorem circles_common_chord :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    circle1 p1.1 p1.2 ∧ circle1 p2.1 p2.2 ∧
    circle2 p1.1 p1.2 ∧ circle2 p2.1 p2.2 →
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y :=
sorry

end circles_common_chord_l3150_315004


namespace almond_weight_in_mixture_l3150_315003

/-- Given a mixture of nuts where the ratio of almonds to walnuts is 5:1 by weight,
    and the total weight is 140 pounds, the weight of almonds is 116.67 pounds. -/
theorem almond_weight_in_mixture (almond_parts : ℕ) (walnut_parts : ℕ) (total_weight : ℝ) :
  almond_parts = 5 →
  walnut_parts = 1 →
  total_weight = 140 →
  (almond_parts * total_weight) / (almond_parts + walnut_parts) = 116.67 := by
  sorry

end almond_weight_in_mixture_l3150_315003


namespace quadratic_function_properties_l3150_315098

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 11

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  (∀ x, f x ≤ 13) ∧  -- Maximum value is 13
  f 3 = 5 ∧          -- f(3) = 5
  f (-1) = 5 ∧       -- f(-1) = 5
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) -- f is a quadratic function
  :=
by
  sorry

#check quadratic_function_properties

end quadratic_function_properties_l3150_315098


namespace leadership_choices_l3150_315076

/-- The number of ways to choose leadership in an organization --/
def choose_leadership (total_members : ℕ) (president_count : ℕ) (vp_count : ℕ) (managers_per_vp : ℕ) : ℕ :=
  let remaining_after_president := total_members - president_count
  let remaining_after_vps := remaining_after_president - vp_count
  let remaining_after_vp1_managers := remaining_after_vps - managers_per_vp
  total_members *
  remaining_after_president *
  (remaining_after_president - 1) *
  (Nat.choose remaining_after_vps managers_per_vp) *
  (Nat.choose remaining_after_vp1_managers managers_per_vp)

/-- Theorem stating the number of ways to choose leadership in the given organization --/
theorem leadership_choices :
  choose_leadership 12 1 2 2 = 554400 :=
by sorry

end leadership_choices_l3150_315076


namespace warehouse_analysis_l3150_315011

/-- Represents the daily record of material movement --/
structure MaterialRecord where
  quantity : Int
  times : Nat

/-- Calculates the net change in material quantity --/
def netChange (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r => acc + r.quantity * r.times) 0

/-- Calculates the transportation cost for Option 1 --/
def costOption1 (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r =>
    acc + (if r.quantity > 0 then 5 else 8) * r.quantity.natAbs * r.times
  ) 0

/-- Calculates the transportation cost for Option 2 --/
def costOption2 (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r => acc + 6 * r.quantity.natAbs * r.times) 0

theorem warehouse_analysis (records : List MaterialRecord) :
  records = [
    { quantity := -3, times := 2 },
    { quantity := 4, times := 1 },
    { quantity := -1, times := 3 },
    { quantity := 2, times := 3 },
    { quantity := -5, times := 2 }
  ] →
  netChange records = -9 ∧ costOption2 records < costOption1 records := by
  sorry

end warehouse_analysis_l3150_315011


namespace pure_imaginary_complex_number_l3150_315014

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, a - (10 : ℂ) / (3 - Complex.I) = b * Complex.I) → a = 3 := by
  sorry

end pure_imaginary_complex_number_l3150_315014


namespace cricket_count_l3150_315032

theorem cricket_count (initial : Float) (additional : Float) :
  initial = 7.0 → additional = 11.0 → initial + additional = 18.0 := by sorry

end cricket_count_l3150_315032


namespace no_infinite_sequence_exists_l3150_315093

theorem no_infinite_sequence_exists : 
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k n ≠ 0) ∧ 
    (∀ n, k (n + 1) = k n - 1 / k n) ∧ 
    (∀ n, k n * k (n + 1) ≥ 0) := by
  sorry

end no_infinite_sequence_exists_l3150_315093


namespace divisibility_property_l3150_315020

theorem divisibility_property (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) :
  (a - c) ∣ (a * d + b * c) := by
  sorry

end divisibility_property_l3150_315020


namespace consecutive_points_length_l3150_315081

/-- Given 6 consecutive points on a straight line, prove that af = 25 -/
theorem consecutive_points_length (a b c d e f : ℝ) : 
  (c - b) = 3 * (d - c) →
  (e - d) = 8 →
  (b - a) = 5 →
  (c - a) = 11 →
  (f - e) = 4 →
  (f - a) = 25 := by
  sorry

end consecutive_points_length_l3150_315081


namespace flower_basket_problem_l3150_315045

theorem flower_basket_problem (o y p : ℕ) 
  (h1 : y + p = 7)   -- All but 7 are orange
  (h2 : o + p = 10)  -- All but 10 are yellow
  (h3 : o + y = 5)   -- All but 5 are purple
  : o + y + p = 11 := by
  sorry

end flower_basket_problem_l3150_315045


namespace james_tylenol_frequency_l3150_315064

/-- Proves that James takes Tylenol tablets every 6 hours given the conditions --/
theorem james_tylenol_frequency 
  (tablets_per_dose : ℕ)
  (mg_per_tablet : ℕ)
  (total_mg_per_day : ℕ)
  (hours_per_day : ℕ)
  (h1 : tablets_per_dose = 2)
  (h2 : mg_per_tablet = 375)
  (h3 : total_mg_per_day = 3000)
  (h4 : hours_per_day = 24) :
  (hours_per_day : ℚ) / ((total_mg_per_day : ℚ) / ((tablets_per_dose : ℚ) * mg_per_tablet)) = 6 := by
  sorry

#check james_tylenol_frequency

end james_tylenol_frequency_l3150_315064


namespace quadratic_roots_sum_reciprocal_l3150_315046

theorem quadratic_roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 7 = 0 →
  x₂^2 - 4*x₂ - 7 = 0 →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  1/x₁ + 1/x₂ = -4/7 := by
sorry

end quadratic_roots_sum_reciprocal_l3150_315046


namespace tech_class_avg_age_l3150_315013

def avg_age_arts : ℝ := 21
def num_arts_classes : ℕ := 8
def num_tech_classes : ℕ := 5
def overall_avg_age : ℝ := 19.846153846153847

theorem tech_class_avg_age :
  let total_classes := num_arts_classes + num_tech_classes
  let total_age := overall_avg_age * total_classes
  let arts_total_age := avg_age_arts * num_arts_classes
  (total_age - arts_total_age) / num_tech_classes = 990.4000000000002 := by
sorry

end tech_class_avg_age_l3150_315013


namespace quadratic_roots_problem_l3150_315023

theorem quadratic_roots_problem (p q : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (p + 5 * Complex.I) ^ 2 - (12 + 8 * Complex.I) * (p + 5 * Complex.I) + (20 + 40 * Complex.I) = 0 →
  (q + 3 * Complex.I) ^ 2 - (12 + 8 * Complex.I) * (q + 3 * Complex.I) + (20 + 40 * Complex.I) = 0 →
  p = 10 ∧ q = 2 := by
sorry

end quadratic_roots_problem_l3150_315023


namespace decreasing_interval_minimum_a_l3150_315099

noncomputable section

/-- The function f(x) = (2 - a)(x - 1) - 2ln(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

/-- The function g(x) = f(x) + x -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

/-- The derivative of g(x) -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 - a - 2 / x

theorem decreasing_interval (a : ℝ) :
  (g' a 1 = -1 ∧ g a 1 = 1) →
  ∀ x, 0 < x → x < 2 → g' a x < 0 :=
sorry

theorem minimum_a :
  (∀ x, 0 < x → x < 1/2 → f a x > 0) →
  a ≥ 2 - 4 * Real.log 2 :=
sorry

end decreasing_interval_minimum_a_l3150_315099


namespace smallest_number_of_cubes_for_given_box_l3150_315034

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes needed to fill a box -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem: The smallest number of identical cubes needed to fill a box with 
    dimensions 36x45x18 inches is 40 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨36, 45, 18⟩ = 40 := by
  sorry

end smallest_number_of_cubes_for_given_box_l3150_315034


namespace infinite_pairs_geometric_progression_l3150_315097

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (seq : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, seq (i + 1) = seq i * r

/-- There are infinitely many pairs of real numbers (a,b) such that 12, a, b, ab form a geometric progression. -/
theorem infinite_pairs_geometric_progression :
  {(a, b) : ℝ × ℝ | IsGeometricProgression (λ i => match i with
    | 0 => 12
    | 1 => a
    | 2 => b
    | 3 => a * b)} = Set.univ := by
  sorry


end infinite_pairs_geometric_progression_l3150_315097


namespace lucy_cookie_sales_l3150_315042

/-- Given that Robyn sold 16 packs of cookies and together with Lucy they sold 35 packs,
    prove that Lucy sold 19 packs. -/
theorem lucy_cookie_sales (robyn_sales : ℕ) (total_sales : ℕ) (h1 : robyn_sales = 16) (h2 : total_sales = 35) :
  total_sales - robyn_sales = 19 := by
  sorry

end lucy_cookie_sales_l3150_315042


namespace vessel_base_length_l3150_315094

/-- Given a cube and a rectangular vessel, proves the length of the vessel's base --/
theorem vessel_base_length 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 16) 
  (h2 : vessel_width = 15) 
  (h3 : water_rise = 13.653333333333334) : 
  (cube_edge ^ 3) / (vessel_width * water_rise) = 20 := by
  sorry

end vessel_base_length_l3150_315094


namespace largest_number_after_removal_l3150_315057

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def concatenated_primes : Nat :=
  first_ten_primes.foldl (fun acc n => acc * (10 ^ (Nat.digits 10 n).length) + n) 0

def remove_six_digits (n : Nat) : Set Nat :=
  { m | ∃ (digits : List Nat), 
    digits.length = (Nat.digits 10 n).length - 6 ∧
    (Nat.digits 10 m) = digits ∧
    (∀ d ∈ digits, d ∈ Nat.digits 10 n) }

theorem largest_number_after_removal :
  7317192329 ∈ remove_six_digits concatenated_primes ∧
  ∀ m ∈ remove_six_digits concatenated_primes, m ≤ 7317192329 := by
  sorry

end largest_number_after_removal_l3150_315057


namespace total_values_count_l3150_315088

theorem total_values_count (initial_mean correct_mean : ℝ) 
  (incorrect_value correct_value : ℝ) (n : ℕ) : 
  initial_mean = 150 →
  correct_mean = 151.25 →
  incorrect_value = 135 →
  correct_value = 160 →
  (n : ℝ) * initial_mean = (n : ℝ) * correct_mean - (correct_value - incorrect_value) →
  n = 20 := by
sorry

end total_values_count_l3150_315088


namespace number_line_essential_elements_l3150_315025

/-- Represents the essential elements of a number line -/
inductive NumberLineElement
  | PositiveDirection
  | Origin
  | UnitLength

/-- The set of essential elements of a number line -/
def essentialElements : Set NumberLineElement :=
  {NumberLineElement.PositiveDirection, NumberLineElement.Origin, NumberLineElement.UnitLength}

/-- Theorem stating that the essential elements of a number line are precisely
    positive direction, origin, and unit length -/
theorem number_line_essential_elements :
  ∀ (e : NumberLineElement), e ∈ essentialElements ↔
    (e = NumberLineElement.PositiveDirection ∨
     e = NumberLineElement.Origin ∨
     e = NumberLineElement.UnitLength) :=
by sorry

end number_line_essential_elements_l3150_315025


namespace factorial_1500_trailing_zeros_l3150_315073

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range 13).sum fun i => n / (5 ^ (i + 1))

/-- 1500! has 374 trailing zeros -/
theorem factorial_1500_trailing_zeros :
  trailingZeros 1500 = 374 := by
  sorry

end factorial_1500_trailing_zeros_l3150_315073


namespace sets_intersection_theorem_l3150_315047

def A (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | q*x^2 + p*x + 1 = 0}

theorem sets_intersection_theorem (p q : ℝ) :
  p ≠ 0 ∧ q ≠ 0 ∧ (A p q ∩ B p q).Nonempty ∧ (-2 ∈ A p q) →
  ((p = 1 ∧ q = -2) ∨ (p = 3 ∧ q = 2) ∨ (p = 5/2 ∧ q = 1)) :=
by sorry

end sets_intersection_theorem_l3150_315047


namespace circle_path_in_triangle_l3150_315037

/-- The path length of the center of a circle rolling inside a triangle --/
def circle_path_length (a b c r : ℝ) : ℝ :=
  (a - 2*r) + (b - 2*r) + (c - 2*r)

/-- Theorem stating the path length of a circle's center rolling inside a specific triangle --/
theorem circle_path_in_triangle : 
  let a : ℝ := 8
  let b : ℝ := 10
  let c : ℝ := 12.5
  let r : ℝ := 1.5
  circle_path_length a b c r = 21.5 := by
  sorry

#check circle_path_in_triangle

end circle_path_in_triangle_l3150_315037


namespace triangle_area_from_squares_l3150_315089

theorem triangle_area_from_squares (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2 : ℝ) * a * b = 24 :=
sorry

end triangle_area_from_squares_l3150_315089


namespace red_white_red_probability_l3150_315095

/-- The probability of drawing a red marble, then a white marble, and finally a red marble
    from a bag containing 4 red marbles and 6 white marbles, without replacement. -/
theorem red_white_red_probability :
  let total_marbles : ℕ := 10
  let red_marbles : ℕ := 4
  let white_marbles : ℕ := 6
  let prob_first_red : ℚ := red_marbles / total_marbles
  let prob_second_white : ℚ := white_marbles / (total_marbles - 1)
  let prob_third_red : ℚ := (red_marbles - 1) / (total_marbles - 2)
  prob_first_red * prob_second_white * prob_third_red = 1 / 10 := by
sorry

end red_white_red_probability_l3150_315095


namespace min_side_c_value_l3150_315096

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the minimum value of c is approximately 2.25 -/
theorem min_side_c_value (a b c : ℝ) (A B C : ℝ) : 
  b = 2 →
  c * Real.cos B + b * Real.cos C = 4 * a * Real.sin B * Real.sin C →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ c ≥ 2.25 - ε :=
sorry

end min_side_c_value_l3150_315096


namespace parabola_symmetry_point_l3150_315053

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a parabola -/
def onParabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = p.a * (pt.x - p.h)^2 + p.k

theorem parabola_symmetry_point (p : Parabola) :
  ∃ (m : ℝ),
    onParabola p ⟨-1, 2⟩ ∧
    onParabola p ⟨1, -2⟩ ∧
    onParabola p ⟨3, 2⟩ ∧
    onParabola p ⟨-2, m⟩ ∧
    onParabola p ⟨4, m⟩ := by
  sorry

end parabola_symmetry_point_l3150_315053


namespace combined_return_percentage_l3150_315027

theorem combined_return_percentage 
  (investment1 : ℝ) 
  (investment2 : ℝ) 
  (return1 : ℝ) 
  (return2 : ℝ) 
  (h1 : investment1 = 500)
  (h2 : investment2 = 1500)
  (h3 : return1 = 0.07)
  (h4 : return2 = 0.09) :
  (investment1 * return1 + investment2 * return2) / (investment1 + investment2) = 0.085 := by
sorry

end combined_return_percentage_l3150_315027


namespace triple_hash_twenty_l3150_315019

/-- The # operation defined on real numbers -/
def hash (N : ℝ) : ℝ := 0.75 * N + 3

/-- Theorem stating that applying the hash operation three times to 20 results in 15.375 -/
theorem triple_hash_twenty : hash (hash (hash 20)) = 15.375 := by sorry

end triple_hash_twenty_l3150_315019


namespace point_transformation_l3150_315052

def rotate90CounterClockwise (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CounterClockwise a b 2 3
  let (x₂, y₂) := reflectAboutYEqualX x₁ y₁
  (x₂ = 5 ∧ y₂ = -1) → b - a = 2 := by
  sorry

end point_transformation_l3150_315052


namespace train_crossing_time_l3150_315001

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 105 →
  train_speed_kmh = 54 →
  crossing_time = 7 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l3150_315001


namespace brick_surface_area_l3150_315068

/-- The surface area of a rectangular prism. -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a rectangular prism with dimensions 8 cm x 4 cm x 2 cm is 112 square centimeters. -/
theorem brick_surface_area :
  surface_area 8 4 2 = 112 := by
  sorry

end brick_surface_area_l3150_315068


namespace hcf_from_lcm_and_product_l3150_315000

/-- Given two positive integers with LCM 750 and product 18750, their HCF is 25 -/
theorem hcf_from_lcm_and_product (A B : ℕ+) 
  (h1 : Nat.lcm A B = 750) 
  (h2 : A * B = 18750) : 
  Nat.gcd A B = 25 := by
  sorry

end hcf_from_lcm_and_product_l3150_315000


namespace intersection_slope_l3150_315024

/-- Given two lines that intersect at a point, prove the slope of one line. -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y, y = -2 * x + 3 → y = m * x + 4) → -- Line p: y = -2x + 3, Line q: y = mx + 4
  1 = -2 * 1 + 3 →                         -- Point (1, 1) satisfies line p
  1 = m * 1 + 4 →                          -- Point (1, 1) satisfies line q
  m = -3 := by sorry

end intersection_slope_l3150_315024


namespace fraction_evaluation_l3150_315033

theorem fraction_evaluation : (3 : ℚ) / (1 - 3 / 4) = 12 := by sorry

end fraction_evaluation_l3150_315033


namespace repeating_decimal_value_l3150_315041

-- Define the repeating decimal 0.454545...
def repeating_decimal : ℚ := 0.454545

-- Theorem statement
theorem repeating_decimal_value : repeating_decimal = 5 / 11 := by
  sorry

end repeating_decimal_value_l3150_315041


namespace quadratic_factorization_l3150_315051

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l3150_315051


namespace certain_number_bound_l3150_315074

theorem certain_number_bound (n : ℝ) : 
  (∀ x : ℝ, x ≤ 2 → 6.1 * 10^x < n) → n > 610 := by sorry

end certain_number_bound_l3150_315074


namespace rectangle_area_l3150_315063

/-- Given a rectangle with length L and width W, if increasing the length by 10
    and decreasing the width by 6 doesn't change the area, and the perimeter is 76,
    then the area of the original rectangle is 360 square meters. -/
theorem rectangle_area (L W : ℝ) : 
  (L + 10) * (W - 6) = L * W → 2 * L + 2 * W = 76 → L * W = 360 := by
  sorry

end rectangle_area_l3150_315063


namespace factor_calculation_l3150_315012

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 18 → 
  factor * (2 * initial_number + 5) = 123 → 
  factor = 3 := by sorry

end factor_calculation_l3150_315012


namespace ron_has_two_friends_l3150_315055

/-- The number of Ron's friends eating pizza -/
def num_friends (total_slices : ℕ) (slices_per_person : ℕ) : ℕ :=
  total_slices / slices_per_person - 1

/-- Theorem: Given a 12-slice pizza and 4 slices per person, Ron has 2 friends -/
theorem ron_has_two_friends : num_friends 12 4 = 2 := by
  sorry

end ron_has_two_friends_l3150_315055


namespace sum_divisors_cube_lt_n_fourth_l3150_315029

def S (n : ℕ) : ℕ := sorry

theorem sum_divisors_cube_lt_n_fourth {n : ℕ} (h_odd : Odd n) (h_gt_one : n > 1) :
  (S n)^3 < n^4 := by sorry

end sum_divisors_cube_lt_n_fourth_l3150_315029


namespace expression_value_l3150_315009

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = -3) :
  ((2 * x - y)^2 - (x - y) * (x + y) - 2 * y^2) / x = 18 := by
  sorry

end expression_value_l3150_315009


namespace max_pieces_theorem_l3150_315080

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.eraseDups.length

def max_pieces : ℕ := 7

theorem max_pieces_theorem :
  ∀ n : ℕ, n > max_pieces →
    ¬∃ (A B : ℕ), is_five_digit A ∧ is_five_digit B ∧ has_distinct_digits A ∧ A = B * n :=
by sorry

end max_pieces_theorem_l3150_315080


namespace restaurant_ratio_change_l3150_315031

theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (hired_waiters : ℕ) :
  initial_cooks = 9 →
  initial_cooks * 11 = initial_waiters * 3 →
  hired_waiters = 12 →
  initial_cooks * 5 = (initial_waiters + hired_waiters) * 1 :=
by sorry

end restaurant_ratio_change_l3150_315031


namespace fraction_simplification_l3150_315039

theorem fraction_simplification : (5 * 6) / 10 = 3 := by
  sorry

end fraction_simplification_l3150_315039


namespace equation_roots_l3150_315085

/-- The equation in question -/
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

/-- The condition for having exactly two distinct complex roots -/
def has_two_distinct_roots (k : ℂ) : Prop :=
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    equation x₁ k ∧ equation x₂ k ∧
    ∀ x, equation x k → (x = x₁ ∨ x = x₂)

/-- The main theorem -/
theorem equation_roots (k : ℂ) :
  has_two_distinct_roots k ↔ (k = 2*I ∨ k = -2*I) :=
sorry

end equation_roots_l3150_315085


namespace parallelepiped_volume_l3150_315030

/-- A rectangular parallelepiped with face diagonals √3, √5, and 2 has volume √6 -/
theorem parallelepiped_volume (a b c : ℝ) 
  (h1 : a^2 + b^2 = 3)
  (h2 : a^2 + c^2 = 5)
  (h3 : b^2 + c^2 = 4) :
  a * b * c = Real.sqrt 6 := by
  sorry

end parallelepiped_volume_l3150_315030


namespace square_area_rational_l3150_315071

theorem square_area_rational (s : ℚ) : ∃ (a : ℚ), a = s^2 := by
  sorry

end square_area_rational_l3150_315071


namespace investment_ratio_l3150_315006

/-- Given two investors p and q, where p invested 60000 and the profit is divided in the ratio 4:6,
    prove that q invested 90000. -/
theorem investment_ratio (p q : ℕ) (h1 : p = 60000) (h2 : 4 * q = 6 * p) : q = 90000 := by
  sorry

end investment_ratio_l3150_315006


namespace product_of_roots_squared_minus_three_l3150_315060

theorem product_of_roots_squared_minus_three (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = -35) := by
sorry

end product_of_roots_squared_minus_three_l3150_315060


namespace sequence_inequality_l3150_315005

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n + a (2 * n) ≥ 3 * n)
  (h2 : ∀ n : ℕ, a (n + 1) + n ≤ 2 * Real.sqrt (a n * (n + 1)))
  (h3 : ∀ n : ℕ, 0 ≤ a n) :
  ∀ n : ℕ, a n ≥ n :=
by sorry

end sequence_inequality_l3150_315005


namespace boat_production_three_months_l3150_315079

def boat_production (initial : ℕ) (months : ℕ) : ℕ :=
  if months = 0 then 0
  else if months = 1 then initial
  else initial + boat_production (initial * 3) (months - 1)

theorem boat_production_three_months :
  boat_production 5 3 = 65 := by
  sorry

end boat_production_three_months_l3150_315079


namespace henri_total_miles_l3150_315091

-- Define the variables
def gervais_average_miles : ℕ := 315
def gervais_days : ℕ := 3
def additional_miles : ℕ := 305

-- Define the theorem
theorem henri_total_miles :
  let gervais_total := gervais_average_miles * gervais_days
  let henri_total := gervais_total + additional_miles
  henri_total = 1250 := by
  sorry

end henri_total_miles_l3150_315091


namespace number_selection_game_probability_l3150_315077

/-- The probability of not winning a prize in the number selection game -/
def prob_not_win : ℚ := 2499 / 2500

/-- The number of options to choose from -/
def num_options : ℕ := 50

theorem number_selection_game_probability :
  prob_not_win = 1 - (1 / num_options^2) :=
sorry

end number_selection_game_probability_l3150_315077


namespace cake_ratio_correct_l3150_315072

/-- The ratio of cakes made each day compared to the previous day -/
def cake_ratio : ℝ := 2

/-- The number of cakes made on the first day -/
def first_day_cakes : ℕ := 10

/-- The number of cakes made on the sixth day -/
def sixth_day_cakes : ℕ := 320

/-- Theorem stating that the cake ratio is correct given the conditions -/
theorem cake_ratio_correct :
  (first_day_cakes : ℝ) * cake_ratio ^ 5 = sixth_day_cakes := by sorry

end cake_ratio_correct_l3150_315072


namespace always_possible_largest_to_smallest_exists_impossible_smallest_to_largest_l3150_315049

-- Define the grid
def Grid := Fin 10 → Fin 10 → Bool

-- Define ship sizes
inductive ShipSize
| one
| two
| three
| four

-- Define the list of ships to be placed
def ships : List ShipSize :=
  [ShipSize.four] ++ List.replicate 2 ShipSize.three ++
  List.replicate 3 ShipSize.two ++ List.replicate 4 ShipSize.one

-- Define a valid placement
def isValidPlacement (g : Grid) (s : ShipSize) (x y : Fin 10) (horizontal : Bool) : Prop :=
  sorry

-- Define the theorem for part a
theorem always_possible_largest_to_smallest :
  ∀ (g : Grid),
  ∃ (g' : Grid),
    (∀ s ∈ ships, ∃ x y h, isValidPlacement g' s x y h) ∧
    (∀ x y, g' x y → g x y) :=
  sorry

-- Define the theorem for part b
theorem exists_impossible_smallest_to_largest :
  ∃ (g : Grid),
    (∀ s ∈ (ships.reverse.take (ships.length - 1)),
      ∃ x y h, isValidPlacement g s x y h) ∧
    (∀ x y h, ¬isValidPlacement g ShipSize.four x y h) :=
  sorry

end always_possible_largest_to_smallest_exists_impossible_smallest_to_largest_l3150_315049


namespace competition_results_l3150_315028

/-- Represents the categories of safety questions -/
inductive Category
  | TrafficSafety
  | FireSafety
  | WaterSafety

/-- Represents the scoring system for the competition -/
structure ScoringSystem where
  correct_points : ℕ
  incorrect_points : ℕ

/-- Represents the correct rates for each category -/
def correct_rates : Category → ℚ
  | Category.TrafficSafety => 2/3
  | Category.FireSafety => 1/2
  | Category.WaterSafety => 1/3

/-- The scoring system used in the competition -/
def competition_scoring : ScoringSystem :=
  { correct_points := 5, incorrect_points := 1 }

/-- Calculates the probability of scoring at least 6 points for two questions -/
def prob_at_least_6_points (s : ScoringSystem) : ℚ :=
  let p_traffic := correct_rates Category.TrafficSafety
  let p_fire := correct_rates Category.FireSafety
  p_traffic * p_fire + p_traffic * (1 - p_fire) + (1 - p_traffic) * p_fire

/-- Calculates the expected value of the total score for three questions from different categories -/
def expected_score_three_questions (s : ScoringSystem) : ℚ :=
  let p_traffic := correct_rates Category.TrafficSafety
  let p_fire := correct_rates Category.FireSafety
  let p_water := correct_rates Category.WaterSafety
  let p_all_correct := p_traffic * p_fire * p_water
  let p_two_correct := p_traffic * p_fire * (1 - p_water) +
                       p_traffic * (1 - p_fire) * p_water +
                       (1 - p_traffic) * p_fire * p_water
  let p_one_correct := p_traffic * (1 - p_fire) * (1 - p_water) +
                       (1 - p_traffic) * p_fire * (1 - p_water) +
                       (1 - p_traffic) * (1 - p_fire) * p_water
  let p_all_incorrect := (1 - p_traffic) * (1 - p_fire) * (1 - p_water)
  3 * s.correct_points * p_all_correct +
  (2 * s.correct_points + s.incorrect_points) * p_two_correct +
  (s.correct_points + 2 * s.incorrect_points) * p_one_correct +
  3 * s.incorrect_points * p_all_incorrect

theorem competition_results :
  prob_at_least_6_points competition_scoring = 5/6 ∧
  expected_score_three_questions competition_scoring = 9 := by
  sorry

end competition_results_l3150_315028


namespace f_properties_l3150_315043

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - 2*a| + a^2 - 3*a

/-- Theorem stating the properties of the function f and its zeros -/
theorem f_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) →
  (3/2 < a ∧ a < 3) ∧
  (2*(Real.sqrt 2 + 1)/3 < 1/x₁ + 1/x₂ + 1/x₃) :=
by sorry


end f_properties_l3150_315043


namespace product_inequality_l3150_315007

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end product_inequality_l3150_315007


namespace platform_length_l3150_315018

/-- Given a train's speed and crossing times, calculate the platform length -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- 72 kmph
  (h2 : platform_crossing_time = 32)  -- 32 seconds
  (h3 : man_crossing_time = 18)  -- 18 seconds
  : ∃ (platform_length : ℝ), platform_length = 280 :=
by
  sorry


end platform_length_l3150_315018


namespace average_marks_chemistry_mathematics_l3150_315086

/-- Given that the total marks in physics, chemistry, and mathematics is 150 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 75. -/
theorem average_marks_chemistry_mathematics (P C M : ℝ)
  (h : P + C + M = P + 150) :
  (C + M) / 2 = 75 := by
  sorry

end average_marks_chemistry_mathematics_l3150_315086


namespace cookout_attendance_l3150_315022

theorem cookout_attendance (kids_2004 kids_2005 kids_2006 : ℕ) : 
  kids_2005 = kids_2004 / 2 →
  kids_2006 = (2 * kids_2005) / 3 →
  kids_2006 = 20 →
  kids_2004 = 60 := by
sorry

end cookout_attendance_l3150_315022


namespace students_playing_at_least_one_sport_l3150_315026

/-- The number of students who like to play basketball -/
def B : ℕ := 7

/-- The number of students who like to play cricket -/
def C : ℕ := 10

/-- The number of students who like to play soccer -/
def S : ℕ := 8

/-- The number of students who like to play all three sports -/
def BCS : ℕ := 2

/-- The number of students who like to play both basketball and cricket -/
def BC : ℕ := 5

/-- The number of students who like to play both basketball and soccer -/
def BS : ℕ := 4

/-- The number of students who like to play both cricket and soccer -/
def CS : ℕ := 3

/-- The theorem stating that the number of students who like to play at least one sport is 21 -/
theorem students_playing_at_least_one_sport : 
  B + C + S - ((BC - BCS) + (BS - BCS) + (CS - BCS)) + BCS = 21 := by
  sorry

end students_playing_at_least_one_sport_l3150_315026


namespace total_crackers_bought_l3150_315075

/-- The number of boxes of crackers Darren bought -/
def darren_boxes : ℕ := 4

/-- The number of crackers in each box -/
def crackers_per_box : ℕ := 24

/-- The number of boxes Calvin bought -/
def calvin_boxes : ℕ := 2 * darren_boxes - 1

/-- The total number of crackers bought by both Darren and Calvin -/
def total_crackers : ℕ := darren_boxes * crackers_per_box + calvin_boxes * crackers_per_box

theorem total_crackers_bought :
  total_crackers = 264 := by
  sorry

end total_crackers_bought_l3150_315075


namespace first_round_games_count_l3150_315061

/-- A tennis tournament with specific conditions -/
structure TennisTournament where
  total_rounds : Nat
  second_round_games : Nat
  third_round_games : Nat
  final_games : Nat
  cans_per_game : Nat
  balls_per_can : Nat
  total_balls_used : Nat

/-- The number of games in the first round of the tournament -/
def first_round_games (t : TennisTournament) : Nat :=
  ((t.total_balls_used - (t.second_round_games + t.third_round_games + t.final_games) * 
    t.cans_per_game * t.balls_per_can) / (t.cans_per_game * t.balls_per_can))

/-- Theorem stating the number of games in the first round -/
theorem first_round_games_count (t : TennisTournament) 
  (h1 : t.total_rounds = 4)
  (h2 : t.second_round_games = 4)
  (h3 : t.third_round_games = 2)
  (h4 : t.final_games = 1)
  (h5 : t.cans_per_game = 5)
  (h6 : t.balls_per_can = 3)
  (h7 : t.total_balls_used = 225) :
  first_round_games t = 8 := by
  sorry

end first_round_games_count_l3150_315061


namespace cosine_calculation_l3150_315078

theorem cosine_calculation : Real.cos (π/3) - 2⁻¹ + Real.sqrt ((-2)^2) - (π-3)^0 = 1 := by
  sorry

end cosine_calculation_l3150_315078


namespace roy_sports_hours_l3150_315015

/-- Calculates the total hours spent on sports in school for a week with missed days -/
def sports_hours_in_week (daily_hours : ℕ) (school_days : ℕ) (missed_days : ℕ) : ℕ :=
  (school_days - missed_days) * daily_hours

/-- Proves that Roy spent 6 hours on sports in school for the given week -/
theorem roy_sports_hours :
  let daily_hours : ℕ := 2
  let school_days : ℕ := 5
  let missed_days : ℕ := 2
  sports_hours_in_week daily_hours school_days missed_days = 6 := by
  sorry

end roy_sports_hours_l3150_315015


namespace solution_set_inequality_l3150_315050

theorem solution_set_inequality (x : ℝ) : 
  (x ≠ 2) → ((2 * x + 5) / (x - 2) < 1 ↔ -7 < x ∧ x < 2) := by sorry

end solution_set_inequality_l3150_315050


namespace diophantine_equation_solution_l3150_315044

theorem diophantine_equation_solution (a : ℕ+) :
  ∃ (x y : ℕ+), (x^3 + x + a^2 : ℤ) = y^2 ∧
  x = 4 * a^2 * (16 * a^4 + 2) ∧
  y = 2 * a * (16 * a^4 + 2) * (16 * a^4 + 1) - a :=
by sorry

end diophantine_equation_solution_l3150_315044


namespace triangle_area_theorem_l3150_315016

/-- Given a triangle ABC with the following properties:
  1. sin C + sin(B-A) = 3 sin(2A)
  2. c = 2
  3. ∠C = π/3
  Prove that the area of triangle ABC is either 2√3/3 or 3√3/7 -/
theorem triangle_area_theorem (A B C : ℝ) (h1 : Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A))
    (h2 : 2 = 2) (h3 : C = π / 3) :
  let S := Real.sqrt 3 / 3 * 2
  let S' := Real.sqrt 3 * 3 / 7
  let area := (Real.sin C) * 2 / 2
  area = S ∨ area = S' := by
sorry

end triangle_area_theorem_l3150_315016


namespace line_parallelism_l3150_315035

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallelism 
  (a b : Line) 
  (α β : Plane) 
  (l : Line) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : intersection α β = l) 
  (h4 : parallel_lines a l) 
  (h5 : subset_line_plane b β) 
  (h6 : parallel_line_plane b α) : 
  parallel_lines a b :=
sorry

end line_parallelism_l3150_315035


namespace tetrahedron_inequality_l3150_315010

/-- Represents a tetrahedron with base edge lengths a, b, c, 
    lateral edge lengths x, y, z, and d being the distance from 
    the top vertex to the centroid of the base. -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  d : ℝ

/-- Theorem stating that for any tetrahedron, the sum of lateral edge lengths
    is less than or equal to the sum of base edge lengths plus three times
    the distance from the top vertex to the centroid of the base. -/
theorem tetrahedron_inequality (t : Tetrahedron) : 
  t.x + t.y + t.z ≤ t.a + t.b + t.c + 3 * t.d := by
  sorry

end tetrahedron_inequality_l3150_315010


namespace union_M_N_l3150_315084

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1}

theorem union_M_N : M ∪ N = {x | x > -1} := by
  sorry

end union_M_N_l3150_315084


namespace smallest_digit_divisible_by_six_l3150_315048

theorem smallest_digit_divisible_by_six : 
  ∃ (N : ℕ), N < 10 ∧ (1453 * 10 + N) % 6 = 0 ∧ 
  ∀ (M : ℕ), M < N → M < 10 → (1453 * 10 + M) % 6 ≠ 0 :=
by sorry

end smallest_digit_divisible_by_six_l3150_315048


namespace twenty_fifth_digit_sum_eighths_quarters_l3150_315056

theorem twenty_fifth_digit_sum_eighths_quarters : ∃ (s : ℚ), 
  (s = 1/8 + 1/4) ∧ 
  (∃ (d : ℕ → ℕ), (∀ n, d n < 10) ∧ 
    (s = ∑' n, (d n : ℚ) / 10^(n+1)) ∧ 
    (d 24 = 0)) := by
  sorry

end twenty_fifth_digit_sum_eighths_quarters_l3150_315056
