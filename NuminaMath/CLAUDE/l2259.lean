import Mathlib

namespace NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l2259_225973

-- Define a structure for triangles
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a triangle to be acute
def isAcute (t : Triangle) : Prop :=
  t.a < 90 ∧ t.b < 90 ∧ t.c < 90

-- Define what it means for a triangle to have prime angles
def hasPrimeAngles (t : Triangle) : Prop :=
  isPrime t.a ∧ isPrime t.b ∧ isPrime t.c

-- Define what it means for a triangle to be valid (sum of angles is 180)
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c = 180

-- Theorem statement
theorem unique_acute_prime_angled_triangle :
  ∃! t : Triangle, isAcute t ∧ hasPrimeAngles t ∧ isValidTriangle t ∧
  t.a = 2 ∧ t.b = 89 ∧ t.c = 89 :=
sorry

end NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l2259_225973


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2259_225946

theorem least_number_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((5432 + m) % 5 = 0 ∧ (5432 + m) % 6 = 0 ∧ (5432 + m) % 4 = 0 ∧ (5432 + m) % 3 = 0)) ∧ 
  ((5432 + n) % 5 = 0 ∧ (5432 + n) % 6 = 0 ∧ (5432 + n) % 4 = 0 ∧ (5432 + n) % 3 = 0) →
  n = 28 := by
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2259_225946


namespace NUMINAMATH_CALUDE_stating_standard_representation_of_point_l2259_225947

/-- 
Given a point in spherical coordinates (ρ, θ, φ), this function returns its standard representation
where 0 ≤ θ < 2π and 0 ≤ φ ≤ π.
-/
def standardSphericalRepresentation (ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- 
Theorem stating that the standard representation of the point (5, 3π/5, 9π/5) 
in spherical coordinates is (5, 8π/5, π/5).
-/
theorem standard_representation_of_point : 
  standardSphericalRepresentation 5 (3 * Real.pi / 5) (9 * Real.pi / 5) = 
    (5, 8 * Real.pi / 5, Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_stating_standard_representation_of_point_l2259_225947


namespace NUMINAMATH_CALUDE_nested_triple_op_result_l2259_225955

def triple_op (a b c : ℚ) : ℚ := (2 * a + b) / c

def nested_triple_op (x y z : ℚ) : ℚ :=
  triple_op (triple_op 30 60 90) (triple_op 3 6 9) (triple_op 6 12 18)

theorem nested_triple_op_result : nested_triple_op 30 60 90 = 4 := by
  sorry

end NUMINAMATH_CALUDE_nested_triple_op_result_l2259_225955


namespace NUMINAMATH_CALUDE_range_of_a_l2259_225941

theorem range_of_a (x a : ℝ) : 
  (∀ x, x^2 + 2*x - 3 ≤ 0 → x ≤ a) ∧ 
  (∃ x, x^2 + 2*x - 3 ≤ 0 ∧ x > a) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2259_225941


namespace NUMINAMATH_CALUDE_candidate_A_percentage_l2259_225939

def total_votes : ℕ := 560000
def invalid_vote_percentage : ℚ := 15 / 100
def valid_votes_for_A : ℕ := 380800

theorem candidate_A_percentage :
  (valid_votes_for_A : ℚ) / ((1 - invalid_vote_percentage) * total_votes) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_candidate_A_percentage_l2259_225939


namespace NUMINAMATH_CALUDE_derivative_f_l2259_225985

noncomputable def f (x : ℝ) := x * Real.sin x + Real.cos x

theorem derivative_f :
  deriv f = fun x ↦ x * Real.cos x := by sorry

end NUMINAMATH_CALUDE_derivative_f_l2259_225985


namespace NUMINAMATH_CALUDE_system_solution_l2259_225999

theorem system_solution (x y z t : ℝ) : 
  (x = (1/2) * (y + 1/y) ∧
   y = (1/2) * (z + 1/z) ∧
   z = (1/2) * (t + 1/t) ∧
   t = (1/2) * (x + 1/x)) →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2259_225999


namespace NUMINAMATH_CALUDE_hcl_formed_equals_c2h6_available_l2259_225938

-- Define the chemical reaction
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2h5cl : ℝ
  hcl : ℝ

-- Define the stoichiometric coefficients
def stoichiometric_ratio : Reaction :=
  { c2h6 := 1, cl2 := 1, c2h5cl := 1, hcl := 1 }

-- Define the available moles of reactants
def available_reactants : Reaction :=
  { c2h6 := 3, cl2 := 6, c2h5cl := 0, hcl := 0 }

-- Theorem: The number of moles of HCl formed is equal to the number of moles of C2H6 available
theorem hcl_formed_equals_c2h6_available :
  available_reactants.hcl = available_reactants.c2h6 :=
by
  sorry


end NUMINAMATH_CALUDE_hcl_formed_equals_c2h6_available_l2259_225938


namespace NUMINAMATH_CALUDE_overtime_probability_l2259_225987

theorem overtime_probability (p_chen p_li p_both : ℝ) : 
  p_chen = 1/3 →
  p_li = 1/4 →
  p_both = 1/6 →
  p_both / p_li = 2/3 := by
sorry

end NUMINAMATH_CALUDE_overtime_probability_l2259_225987


namespace NUMINAMATH_CALUDE_count_99_is_stone_10_l2259_225923

/-- Represents the number of stones in the circle -/
def num_stones : ℕ := 14

/-- Represents the length of a full counting cycle -/
def cycle_length : ℕ := 2 * num_stones - 1

/-- Maps a count to its corresponding stone number -/
def count_to_stone (count : ℕ) : ℕ :=
  let adjusted_count := (count - 1) % cycle_length + 1
  if adjusted_count ≤ num_stones
  then adjusted_count
  else 2 * num_stones - adjusted_count

/-- Theorem stating that the 99th count corresponds to the 10th stone -/
theorem count_99_is_stone_10 : count_to_stone 99 = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_99_is_stone_10_l2259_225923


namespace NUMINAMATH_CALUDE_problem_solution_l2259_225970

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 2) (h2 : b^2 / c = 3) (h3 : c^2 / a = 4) :
  a = 576^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2259_225970


namespace NUMINAMATH_CALUDE_path_combinations_l2259_225906

theorem path_combinations (ways_AB ways_BC : ℕ) (h1 : ways_AB = 2) (h2 : ways_BC = 3) :
  ways_AB * ways_BC = 6 := by
sorry

end NUMINAMATH_CALUDE_path_combinations_l2259_225906


namespace NUMINAMATH_CALUDE_correct_sum_after_error_l2259_225954

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a before multiplying by b and adding 35 results in 226,
    then the correct sum of ab + 35 is 54. -/
theorem correct_sum_after_error (a b : ℕ+) : 
  (a.val ≥ 10 ∧ a.val ≤ 99) →
  (((10 * (a.val % 10) + (a.val / 10)) * b.val) + 35 = 226) →
  (a.val * b.val + 35 = 54) :=
by sorry

end NUMINAMATH_CALUDE_correct_sum_after_error_l2259_225954


namespace NUMINAMATH_CALUDE_training_effect_l2259_225917

/-- Represents the scores and their frequencies in a test --/
structure TestScores :=
  (scores : List Nat)
  (frequencies : List Nat)

/-- Calculates the median of a list of scores --/
def median (scores : List Nat) : Nat :=
  sorry

/-- Calculates the mode of a list of scores --/
def mode (scores : List Nat) (frequencies : List Nat) : Nat :=
  sorry

/-- Calculates the average score --/
def average (scores : List Nat) (frequencies : List Nat) : Real :=
  sorry

/-- Calculates the number of students with scores greater than or equal to a threshold --/
def countExcellent (scores : List Nat) (frequencies : List Nat) (threshold : Nat) : Nat :=
  sorry

theorem training_effect (baselineScores simExamScores : TestScores)
  (totalStudents sampleSize : Nat)
  (hTotalStudents : totalStudents = 800)
  (hSampleSize : sampleSize = 50)
  (hBaselineScores : baselineScores = ⟨[6, 7, 8, 9, 10], [16, 8, 9, 9, 8]⟩)
  (hSimExamScores : simExamScores = ⟨[6, 7, 8, 9, 10], [5, 8, 6, 12, 19]⟩) :
  (median baselineScores.scores = 8 ∧ 
   mode simExamScores.scores simExamScores.frequencies = 10) ∧
  (average simExamScores.scores simExamScores.frequencies - 
   average baselineScores.scores baselineScores.frequencies = 0.94) ∧
  (totalStudents * (countExcellent simExamScores.scores simExamScores.frequencies 9) / sampleSize = 496) :=
by sorry

end NUMINAMATH_CALUDE_training_effect_l2259_225917


namespace NUMINAMATH_CALUDE_handshake_count_l2259_225953

theorem handshake_count (n : ℕ) (h : n = 6) : 
  n * 2 * (n * 2 - 2) / 2 = 60 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l2259_225953


namespace NUMINAMATH_CALUDE_train_distance_problem_l2259_225977

/-- The distance between two points P and Q, given the conditions of two trains traveling towards each other --/
theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 50) (h2 : v2 = 40) (h3 : d = 100) : 
  v1 * (d / (v1 - v2) + d / v2) = 900 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2259_225977


namespace NUMINAMATH_CALUDE_advertising_department_size_l2259_225932

theorem advertising_department_size 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (selected_from_ad : ℕ) 
  (h1 : total_employees = 1000)
  (h2 : sample_size = 80)
  (h3 : selected_from_ad = 4) :
  (selected_from_ad : ℚ) / (sample_size : ℚ) = (50 : ℚ) / (total_employees : ℚ) :=
by
  sorry

#check advertising_department_size

end NUMINAMATH_CALUDE_advertising_department_size_l2259_225932


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2259_225920

def B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 2, 1, 2; 3, 2, 1]

theorem matrix_equation_solution :
  ∃ (a b c : ℚ), 
    B^3 + a • B^2 + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 ∧ 
    a = 0 ∧ b = -283/13 ∧ c = 902/13 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2259_225920


namespace NUMINAMATH_CALUDE_expression_value_l2259_225916

theorem expression_value (x y z : ℚ) 
  (h1 : 3 * x - 2 * y - 2 * z = 0)
  (h2 : x - 4 * y + 8 * z = 0)
  (h3 : z ≠ 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2259_225916


namespace NUMINAMATH_CALUDE_certain_fraction_is_two_fifths_l2259_225993

theorem certain_fraction_is_two_fifths :
  ∀ (x y : ℚ),
    (x ≠ 0 ∧ y ≠ 0) →
    ((1 : ℚ) / 7) / (x / y) = ((3 : ℚ) / 7) / ((6 : ℚ) / 5) →
    x / y = (2 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_fraction_is_two_fifths_l2259_225993


namespace NUMINAMATH_CALUDE_coin_problem_l2259_225956

theorem coin_problem (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let recovered := (3 : ℚ) / 4 * lost
  x - (x - lost + recovered) = x / 6 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l2259_225956


namespace NUMINAMATH_CALUDE_maya_total_pages_l2259_225981

/-- The total number of pages Maya read in two weeks -/
def total_pages (books_last_week : ℕ) (pages_per_book : ℕ) (reading_increase : ℕ) : ℕ :=
  let pages_last_week := books_last_week * pages_per_book
  let pages_this_week := reading_increase * pages_last_week
  pages_last_week + pages_this_week

/-- Theorem stating that Maya read 4500 pages in total -/
theorem maya_total_pages :
  total_pages 5 300 2 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_maya_total_pages_l2259_225981


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l2259_225964

theorem max_perimeter_special_triangle :
  ∀ x : ℕ,
    x > 0 →
    x ≤ 20 →
    x + 4*x > 20 →
    x + 20 > 4*x →
    4*x + 20 > x →
    (∀ y : ℕ, 
      y > 0 →
      y ≤ 20 →
      y + 4*y > 20 →
      y + 20 > 4*y →
      4*y + 20 > y →
      x + 4*x + 20 ≥ y + 4*y + 20) →
    x + 4*x + 20 = 50 :=
by sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l2259_225964


namespace NUMINAMATH_CALUDE_network_paths_count_l2259_225909

-- Define the network structure
structure Network where
  hasDirectPath : (Char × Char) → Prop
  
-- Define the number of paths between two points
def numPaths (net : Network) (start finish : Char) : ℕ := sorry

-- Theorem statement
theorem network_paths_count (net : Network) :
  (net.hasDirectPath ('E', 'B')) →
  (net.hasDirectPath ('F', 'B')) →
  (net.hasDirectPath ('F', 'A')) →
  (net.hasDirectPath ('M', 'A')) →
  (net.hasDirectPath ('M', 'B')) →
  (net.hasDirectPath ('M', 'E')) →
  (net.hasDirectPath ('M', 'F')) →
  (net.hasDirectPath ('A', 'C')) →
  (net.hasDirectPath ('A', 'D')) →
  (net.hasDirectPath ('B', 'A')) →
  (net.hasDirectPath ('B', 'C')) →
  (net.hasDirectPath ('B', 'N')) →
  (net.hasDirectPath ('C', 'N')) →
  (net.hasDirectPath ('D', 'N')) →
  (numPaths net 'M' 'N' = 16) :=
by sorry

end NUMINAMATH_CALUDE_network_paths_count_l2259_225909


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2259_225922

/-- The repeating decimal 0.37246̄ expressed as a rational number -/
def repeating_decimal : ℚ := 
  37 / 100 + (246 / 999900)

/-- The target fraction -/
def target_fraction : ℚ := 3718740 / 999900

/-- Theorem stating that the repeating decimal 0.37246̄ is equal to 3718740/999900 -/
theorem repeating_decimal_equals_fraction : 
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2259_225922


namespace NUMINAMATH_CALUDE_max_radius_third_jar_l2259_225957

theorem max_radius_third_jar (pot_diameter : ℝ) (jar1_radius : ℝ) (jar2_radius : ℝ) :
  pot_diameter = 36 →
  jar1_radius = 6 →
  jar2_radius = 12 →
  ∃ (max_radius : ℝ),
    max_radius = 36 / 7 ∧
    ∀ (r : ℝ), r > max_radius →
      ¬ (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
        (x1^2 + y1^2 ≤ (pot_diameter/2)^2) ∧
        (x2^2 + y2^2 ≤ (pot_diameter/2)^2) ∧
        (x3^2 + y3^2 ≤ (pot_diameter/2)^2) ∧
        ((x1 - x2)^2 + (y1 - y2)^2 ≥ (jar1_radius + jar2_radius)^2) ∧
        ((x1 - x3)^2 + (y1 - y3)^2 ≥ (jar1_radius + r)^2) ∧
        ((x2 - x3)^2 + (y2 - y3)^2 ≥ (jar2_radius + r)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_max_radius_third_jar_l2259_225957


namespace NUMINAMATH_CALUDE_milk_composition_equation_l2259_225926

/-- Represents the nutritional composition of a bottle of milk -/
structure MilkComposition where
  protein : ℝ
  fat : ℝ
  carbohydrate : ℝ

/-- The total content of carbohydrates, protein, and fat in grams -/
def total_content : ℝ := 30

/-- Theorem stating the correct equation for the milk composition -/
theorem milk_composition_equation (m : MilkComposition) 
  (h1 : m.carbohydrate = 1.5 * m.protein)
  (h2 : m.carbohydrate + m.protein + m.fat = total_content) :
  (5/2) * m.protein + m.fat = total_content := by
  sorry

end NUMINAMATH_CALUDE_milk_composition_equation_l2259_225926


namespace NUMINAMATH_CALUDE_gradient_and_magnitude_at_point_l2259_225945

/-- The function z(x, y) = 3x^2 - 2y^2 -/
def z (x y : ℝ) : ℝ := 3 * x^2 - 2 * y^2

/-- The gradient of z at point (x, y) -/
def grad_z (x y : ℝ) : ℝ × ℝ := (6 * x, -4 * y)

theorem gradient_and_magnitude_at_point :
  let p : ℝ × ℝ := (1, 2)
  (grad_z p.1 p.2 = (6, -8)) ∧
  (Real.sqrt ((grad_z p.1 p.2).1^2 + (grad_z p.1 p.2).2^2) = 10) := by
  sorry

end NUMINAMATH_CALUDE_gradient_and_magnitude_at_point_l2259_225945


namespace NUMINAMATH_CALUDE_marble_ratio_l2259_225951

/-- Represents the number of marbles each person has -/
structure Marbles where
  atticus : ℕ
  jensen : ℕ
  cruz : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  3 * (m.atticus + m.jensen + m.cruz) = 60 ∧
  m.atticus = 4 ∧
  m.cruz = 8

/-- The theorem stating the ratio of Atticus's marbles to Jensen's marbles -/
theorem marble_ratio (m : Marbles) :
  marble_conditions m → m.atticus * 2 = m.jensen := by
  sorry

#check marble_ratio

end NUMINAMATH_CALUDE_marble_ratio_l2259_225951


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_count_l2259_225959

/-- Number of elements in a row of Pascal's Triangle -/
def pascal_row_count (n : ℕ) : ℕ := n + 1

/-- Sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of elements in the first 30 rows of Pascal's Triangle is 465 -/
theorem pascal_triangle_30_rows_count : sum_first_n 30 = 465 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_count_l2259_225959


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2259_225950

/-- Given a train of length 300 meters that crosses a platform in 51 seconds
    and a signal pole in 18 seconds, prove that the length of the platform
    is approximately 550.17 meters. -/
theorem platform_length_calculation (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 51)
  (h3 : pole_crossing_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 550.17) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l2259_225950


namespace NUMINAMATH_CALUDE_special_collection_books_l2259_225900

/-- The number of books in the special collection at the beginning of the month -/
def initial_books : ℕ := 75

/-- The percentage of loaned books that are returned -/
def return_rate : ℚ := 65 / 100

/-- The number of books in the special collection at the end of the month -/
def final_books : ℕ := 54

/-- The number of books loaned out during the month -/
def loaned_books : ℚ := 60.00000000000001

theorem special_collection_books :
  initial_books = final_books + (loaned_books - loaned_books * return_rate).ceil :=
sorry

end NUMINAMATH_CALUDE_special_collection_books_l2259_225900


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2259_225961

theorem isosceles_triangle_perimeter (base height : ℝ) (h1 : base = 10) (h2 : height = 6) :
  let side := Real.sqrt (height ^ 2 + (base / 2) ^ 2)
  2 * side + base = 2 * Real.sqrt 61 + 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2259_225961


namespace NUMINAMATH_CALUDE_scientific_notation_260000_l2259_225928

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Check if a ScientificNotation represents a given real number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  x = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number 260000 in scientific notation -/
def n : ScientificNotation :=
  { coefficient := 2.6
    exponent := 5
    h1 := by sorry }

theorem scientific_notation_260000 :
  represents n 260000 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_260000_l2259_225928


namespace NUMINAMATH_CALUDE_bird_cage_problem_l2259_225942

theorem bird_cage_problem (initial_birds : ℕ) : 
  (1 / 3 : ℚ) * (3 / 5 : ℚ) * (1 / 3 : ℚ) * initial_birds = 8 →
  initial_birds = 60 := by
sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l2259_225942


namespace NUMINAMATH_CALUDE_some_number_value_l2259_225949

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * some_number * 7) :
  some_number = 105 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2259_225949


namespace NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l2259_225952

theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : ℕ :=
  let sentences_per_page := paragraphs_per_page * sentences_per_paragraph
  let total_sentences := sentences_per_page * total_pages
  total_sentences / reading_speed

theorem gwendolyn_reading_time : 
  reading_time_calculation 300 40 20 150 = 400 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_gwendolyn_reading_time_l2259_225952


namespace NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l2259_225991

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | (x - (a + 5)) / (x - a) > 0}

-- Theorem for part 1
theorem intersection_when_a_neg_two :
  A ∩ B (-2) = {x | 3 < x ∧ x ≤ 4} := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  A ⊆ B a ↔ a < -6 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l2259_225991


namespace NUMINAMATH_CALUDE_masters_percentage_is_76_l2259_225937

/-- Represents a sports team with juniors and masters -/
structure Team where
  juniors : ℕ
  masters : ℕ

/-- Calculates the percentage of masters in a team -/
def percentageMasters (team : Team) : ℚ :=
  (team.masters : ℚ) / ((team.juniors + team.masters) : ℚ) * 100

/-- Theorem stating that under the given conditions, the percentage of masters is 76% -/
theorem masters_percentage_is_76 (team : Team) 
  (h1 : 22 * team.juniors + 47 * team.masters = 41 * (team.juniors + team.masters)) :
  percentageMasters team = 76 := by
  sorry

#eval (76 : ℚ)

end NUMINAMATH_CALUDE_masters_percentage_is_76_l2259_225937


namespace NUMINAMATH_CALUDE_min_value_theorem_l2259_225905

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + a*b + a*c + b*c = 4) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 → 2*a + b + c ≤ 2*x + y + z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2259_225905


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l2259_225974

theorem rightmost_three_digits_of_7_to_1987 :
  7^1987 % 1000 = 543 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l2259_225974


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_base_neq_slant_l2259_225984

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  /-- The edge length of the base -/
  baseEdge : ℝ
  /-- The slant height of the pyramid -/
  slantHeight : ℝ
  /-- The apex angle of each lateral face -/
  apexAngle : ℝ
  /-- Condition: baseEdge and slantHeight are positive -/
  baseEdge_pos : baseEdge > 0
  slantHeight_pos : slantHeight > 0
  /-- Condition: The apex angle is determined by the baseEdge and slantHeight -/
  apexAngle_eq : apexAngle = 2 * Real.arcsin (baseEdge / (2 * slantHeight))

/-- Theorem: It's impossible for a regular hexagonal pyramid to have its base edge length equal to its slant height -/
theorem hexagonal_pyramid_base_neq_slant (p : RegularHexagonalPyramid) : 
  p.baseEdge ≠ p.slantHeight := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_base_neq_slant_l2259_225984


namespace NUMINAMATH_CALUDE_fraction_calculation_l2259_225933

theorem fraction_calculation : 
  (((1 : ℚ) / 6 - (1 : ℚ) / 8 + (1 : ℚ) / 9) / 
   ((1 : ℚ) / 3 - (1 : ℚ) / 4 + (1 : ℚ) / 5)) * 3 = 55 / 34 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2259_225933


namespace NUMINAMATH_CALUDE_f_properties_l2259_225913

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem f_properties :
  (∀ x > 0, f x = x^2 + x - Real.log x) →
  (∃ m b : ℝ, ∀ x : ℝ, (x = 1 → f x = m * x + b) ∧ m = 2 ∧ b = 0) ∧
  (∃ x_min : ℝ, x_min > 0 ∧ ∀ x > 0, f x ≥ f x_min ∧ f x_min = 3/4 + Real.log 2) ∧
  (¬ ∃ x_max : ℝ, x_max > 0 ∧ ∀ x > 0, f x ≤ f x_max) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2259_225913


namespace NUMINAMATH_CALUDE_total_money_l2259_225998

/-- Given that r has two-thirds of the total amount and r has $2800, 
    prove that the total amount of money p, q, and r have among themselves is $4200. -/
theorem total_money (r_share : ℚ) (r_amount : ℕ) (total : ℕ) : 
  r_share = 2/3 → r_amount = 2800 → total = r_amount * 3/2 → total = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2259_225998


namespace NUMINAMATH_CALUDE_gnomon_shadow_length_l2259_225971

/-- Given a candle and a gnomon, this theorem calculates the length of the shadow cast by the gnomon. -/
theorem gnomon_shadow_length 
  (h : ℝ) -- height of the candle
  (H : ℝ) -- height of the gnomon
  (d : ℝ) -- distance between the bases of the candle and gnomon
  (h_pos : h > 0)
  (H_pos : H > 0)
  (d_pos : d > 0)
  (H_gt_h : H > h) :
  ∃ x : ℝ, x = (h * d) / (H - h) ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_gnomon_shadow_length_l2259_225971


namespace NUMINAMATH_CALUDE_intersection_sum_is_eight_l2259_225980

noncomputable def P : ℝ × ℝ := (0, 8)
noncomputable def Q : ℝ × ℝ := (0, 0)
noncomputable def R : ℝ × ℝ := (10, 0)

noncomputable def G : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
noncomputable def H : ℝ × ℝ := ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

noncomputable def line_PH (x : ℝ) : ℝ := 
  (H.2 - P.2) / (H.1 - P.1) * (x - P.1) + P.2

theorem intersection_sum_is_eight : 
  ∃ (I : ℝ × ℝ), I.1 = G.1 ∧ I.2 = line_PH I.1 ∧ I.1 + I.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_is_eight_l2259_225980


namespace NUMINAMATH_CALUDE_movie_profit_calculation_l2259_225918

def movie_profit (actor_cost food_cost_per_person num_people equipment_rental_factor selling_price : ℚ) : ℚ :=
  let food_cost := food_cost_per_person * num_people
  let total_food_and_actors := actor_cost + food_cost
  let equipment_cost := equipment_rental_factor * total_food_and_actors
  let total_cost := actor_cost + food_cost + equipment_cost
  selling_price - total_cost

theorem movie_profit_calculation :
  movie_profit 1200 3 50 2 10000 = 5950 :=
by sorry

end NUMINAMATH_CALUDE_movie_profit_calculation_l2259_225918


namespace NUMINAMATH_CALUDE_net_profit_calculation_l2259_225979

def basil_seed_cost : ℝ := 2
def mint_seed_cost : ℝ := 3
def zinnia_seed_cost : ℝ := 7
def potting_soil_cost : ℝ := 15

def basil_yield : ℕ := 20
def mint_yield : ℕ := 15
def zinnia_yield : ℕ := 10

def basil_germination_rate : ℝ := 0.8
def mint_germination_rate : ℝ := 0.75
def zinnia_germination_rate : ℝ := 0.7

def healthy_basil_price : ℝ := 5
def healthy_mint_price : ℝ := 6
def healthy_zinnia_price : ℝ := 10

def small_basil_price : ℝ := 3
def small_mint_price : ℝ := 4
def small_zinnia_price : ℝ := 7

def healthy_basil_sold : ℕ := 12
def small_basil_sold : ℕ := 8
def healthy_mint_sold : ℕ := 10
def small_mint_sold : ℕ := 4
def healthy_zinnia_sold : ℕ := 5
def small_zinnia_sold : ℕ := 2

def total_cost : ℝ := basil_seed_cost + mint_seed_cost + zinnia_seed_cost + potting_soil_cost

def total_revenue : ℝ :=
  healthy_basil_price * healthy_basil_sold +
  small_basil_price * small_basil_sold +
  healthy_mint_price * healthy_mint_sold +
  small_mint_price * small_mint_sold +
  healthy_zinnia_price * healthy_zinnia_sold +
  small_zinnia_price * small_zinnia_sold

theorem net_profit_calculation : 
  total_revenue - total_cost = 197 := by sorry

end NUMINAMATH_CALUDE_net_profit_calculation_l2259_225979


namespace NUMINAMATH_CALUDE_problem_1_l2259_225996

theorem problem_1 (x y : ℝ) : (x - 2*y)^2 - x*(x + 3*y) - 4*y^2 = -7*x*y := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2259_225996


namespace NUMINAMATH_CALUDE_parabola_vertex_l2259_225966

/-- A parabola is defined by the equation y = 3(x-7)^2 + 5. -/
def parabola (x y : ℝ) : Prop := y = 3 * (x - 7)^2 + 5

/-- The vertex of a parabola is the point where it reaches its minimum or maximum. -/
def is_vertex (x y : ℝ) : Prop := parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- The vertex of the parabola y = 3(x-7)^2 + 5 has coordinates (7, 5). -/
theorem parabola_vertex : is_vertex 7 5 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2259_225966


namespace NUMINAMATH_CALUDE_stamp_costs_l2259_225975

theorem stamp_costs (a b c d : ℝ) : 
  a + b + c + d = 84 →                   -- sum is 84
  b - a = c - b ∧ c - b = d - c →        -- arithmetic progression
  d = 2.5 * a →                          -- largest is 2.5 times smallest
  a = 12 ∧ b = 18 ∧ c = 24 ∧ d = 30 :=   -- prove the values
by sorry

end NUMINAMATH_CALUDE_stamp_costs_l2259_225975


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2259_225992

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) :
  ∃ (max : ℝ), max = 1 + Real.sqrt 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 6*y' + 12 = 0 → x' - y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2259_225992


namespace NUMINAMATH_CALUDE_central_angle_alice_bob_l2259_225901

/-- Represents a point on the Earth's surface with latitude and longitude -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on a spherical Earth -/
noncomputable def centralAngle (a b : EarthPoint) : Real :=
  sorry

/-- The location of Alice near Quito, Ecuador -/
def alice : EarthPoint :=
  { latitude := 0, longitude := -78 }

/-- The location of Bob near Vladivostok, Russia -/
def bob : EarthPoint :=
  { latitude := 43, longitude := 132 }

/-- Theorem stating that the central angle between Alice and Bob is 150 degrees -/
theorem central_angle_alice_bob :
  centralAngle alice bob = 150 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_alice_bob_l2259_225901


namespace NUMINAMATH_CALUDE_max_product_sum_l2259_225919

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) ∧
  A * M * C + A * M + M * C + C * A = 200 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_l2259_225919


namespace NUMINAMATH_CALUDE_quadratic_roots_l2259_225965

theorem quadratic_roots (a : ℝ) : 
  (3 : ℝ) ^ 2 - 2 * 3 + a = 0 → 
  (-1 : ℝ) ^ 2 - 2 * (-1) + a = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2259_225965


namespace NUMINAMATH_CALUDE_karen_fern_leaves_l2259_225968

/-- Calculates the number of leaves per frond given the total number of leaves,
    number of ferns, and number of fronds per fern. -/
def leaves_per_frond (total_leaves : ℕ) (num_ferns : ℕ) (fronds_per_fern : ℕ) : ℕ :=
  total_leaves / (num_ferns * fronds_per_fern)

/-- Proves that given Karen's fern arrangement, each frond has 30 leaves. -/
theorem karen_fern_leaves :
  let total_leaves : ℕ := 1260
  let num_ferns : ℕ := 6
  let fronds_per_fern : ℕ := 7
  leaves_per_frond total_leaves num_ferns fronds_per_fern = 30 := by
  sorry

end NUMINAMATH_CALUDE_karen_fern_leaves_l2259_225968


namespace NUMINAMATH_CALUDE_problem_statement_l2259_225914

theorem problem_statement (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 2/3 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → a^2 + 2*b^2 ≥ min) ∧
  (a*x + b*y) * (a*y + b*x) ≥ x*y := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2259_225914


namespace NUMINAMATH_CALUDE_colony_leadership_arrangements_l2259_225927

def colony_size : ℕ := 12
def num_deputies : ℕ := 2
def subordinates_per_deputy : ℕ := 3

def leadership_arrangements : ℕ :=
  colony_size *
  (colony_size - 1) *
  (colony_size - 2) *
  (Nat.choose (colony_size - num_deputies - 1) subordinates_per_deputy) *
  (Nat.choose (colony_size - num_deputies - 1 - subordinates_per_deputy) subordinates_per_deputy)

theorem colony_leadership_arrangements :
  leadership_arrangements = 2209600 :=
by sorry

end NUMINAMATH_CALUDE_colony_leadership_arrangements_l2259_225927


namespace NUMINAMATH_CALUDE_sum_b_n_equals_572_l2259_225934

def b_n (n : ℕ) : ℕ :=
  if n % 17 = 0 ∧ n % 19 = 0 then 15
  else if n % 19 = 0 ∧ n % 13 = 0 then 18
  else if n % 13 = 0 ∧ n % 17 = 0 then 17
  else 0

theorem sum_b_n_equals_572 :
  (Finset.range 2999).sum b_n = 572 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_n_equals_572_l2259_225934


namespace NUMINAMATH_CALUDE_blue_eyes_count_l2259_225960

/-- The number of people in the theater -/
def total_people : ℕ := 100

/-- The number of people with brown eyes -/
def brown_eyes : ℕ := total_people / 2

/-- The number of people with black eyes -/
def black_eyes : ℕ := total_people / 4

/-- The number of people with green eyes -/
def green_eyes : ℕ := 6

/-- The number of people with blue eyes -/
def blue_eyes : ℕ := total_people - (brown_eyes + black_eyes + green_eyes)

theorem blue_eyes_count : blue_eyes = 19 := by
  sorry

end NUMINAMATH_CALUDE_blue_eyes_count_l2259_225960


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l2259_225983

theorem square_minus_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (product_eq : a * b = 11) : 
  a^2 - a*b + b^2 = 67 := by
sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l2259_225983


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l2259_225921

theorem quadratic_perfect_square (x : ℝ) : ∃ (a b : ℝ), x^2 - 20*x + 100 = (a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l2259_225921


namespace NUMINAMATH_CALUDE_parabola_directrix_theorem_l2259_225972

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- A parabola opens upward if a > 0 -/
def opens_upward (p : Parabola) : Prop := p.a > 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_directrix_theorem (p : Parabola) :
  p.a = 1/4 ∧ p.b = 0 ∧ p.c = 0 ∧ opens_upward p ∧ vertex p = (0, 0) →
  directrix p = -1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_theorem_l2259_225972


namespace NUMINAMATH_CALUDE_circle_distance_bounds_specific_circle_distances_l2259_225990

/-- Given a circle with radius r and a point M at distance d from the center,
    returns a pair of the minimum and maximum distances from M to any point on the circle -/
def minMaxDistances (r d : ℝ) : ℝ × ℝ :=
  (r - d, r + d)

theorem circle_distance_bounds (r d : ℝ) (hr : r > 0) (hd : 0 ≤ d ∧ d < r) :
  let (min, max) := minMaxDistances r d
  ∀ p : ℝ × ℝ, (p.1 - r)^2 + p.2^2 = r^2 →
    min^2 ≤ (p.1 - d)^2 + p.2^2 ∧ (p.1 - d)^2 + p.2^2 ≤ max^2 :=
by sorry

theorem specific_circle_distances :
  minMaxDistances 10 3 = (7, 13) :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_bounds_specific_circle_distances_l2259_225990


namespace NUMINAMATH_CALUDE_output_for_three_l2259_225958

def f (a : ℤ) : ℤ :=
  if a < 10 then 2 * a else a + 1

theorem output_for_three :
  f 3 = 6 :=
by sorry

end NUMINAMATH_CALUDE_output_for_three_l2259_225958


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2259_225911

-- Define the function f(x) = x^3 + x
def f (x : ℝ) := x^3 + x

-- Define the derivative of f(x)
def f' (x : ℝ) := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (4 * x - y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2259_225911


namespace NUMINAMATH_CALUDE_f_composition_l2259_225936

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the domain of x
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 5

-- State the theorem
theorem f_composition (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (2 * x - 3) = 4 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_l2259_225936


namespace NUMINAMATH_CALUDE_square_of_two_times_sqrt_three_l2259_225924

theorem square_of_two_times_sqrt_three : (2 * Real.sqrt 3) ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_times_sqrt_three_l2259_225924


namespace NUMINAMATH_CALUDE_skateboard_and_graffiti_l2259_225967

/-- Skateboard distance traveled in n seconds -/
def skateboardDistance (n : ℕ) : ℕ := 
  n * (8 + (n - 1) * 5)

/-- Visible graffiti area after n seconds -/
def graffitiArea (n : ℕ) : ℕ := 
  2^(n + 1) - 2

theorem skateboard_and_graffiti : 
  skateboardDistance 20 = 2060 ∧ graffitiArea 20 = 2^21 - 2 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_and_graffiti_l2259_225967


namespace NUMINAMATH_CALUDE_factor_expression_l2259_225997

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2259_225997


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l2259_225908

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The total number of possible subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets that cannot form polygons (0, 1, or 2 points) -/
def non_polygon_subsets : ℕ := (n.choose 0) + (n.choose 1) + (n.choose 2)

/-- The number of distinct convex polygons with three or more sides -/
def num_polygons : ℕ := total_subsets - non_polygon_subsets

theorem distinct_polygons_count :
  num_polygons = 4017 :=
by sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l2259_225908


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l2259_225963

theorem quadratic_equation_conversion :
  ∀ x : ℝ, (x - 8)^2 = 5 ↔ x^2 - 16*x + 59 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l2259_225963


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2259_225925

-- Define the sets
def U : Set ℕ := Set.univ
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 3, 6, 8}

-- State the theorem
theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2259_225925


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_two_point_five_l2259_225904

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A point is inside a square --/
def isInside (p : ℝ × ℝ) (s : Square) : Prop :=
  s.bottomLeft.1 ≤ p.1 ∧ p.1 ≤ s.topRight.1 ∧
  s.bottomLeft.2 ≤ p.2 ∧ p.2 ≤ s.topRight.2

/-- The probability of an event for a uniformly distributed point in a square --/
def probability (s : Square) (event : ℝ × ℝ → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_two_point_five :
  let s : Square := { bottomLeft := (0, 0), topRight := (3, 3) }
  probability s (fun p => p.1 + p.2 < 2.5) = 125 / 360 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_two_point_five_l2259_225904


namespace NUMINAMATH_CALUDE_people_left_line_l2259_225976

theorem people_left_line (initial : ℕ) (joined : ℕ) (final : ℕ) (left : ℕ) : 
  initial = 9 → joined = 3 → final = 6 → initial - left + joined = final → left = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_left_line_l2259_225976


namespace NUMINAMATH_CALUDE_students_only_swimming_l2259_225915

/-- The number of students only participating in swimming in a sports day scenario --/
theorem students_only_swimming (total : ℕ) (swimming : ℕ) (track : ℕ) (ball : ℕ) 
  (swim_track : ℕ) (swim_ball : ℕ) : 
  total = 28 → 
  swimming = 15 → 
  track = 8 → 
  ball = 14 → 
  swim_track = 3 → 
  swim_ball = 3 → 
  swimming - (swim_track + swim_ball) = 9 := by
  sorry

#check students_only_swimming

end NUMINAMATH_CALUDE_students_only_swimming_l2259_225915


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l2259_225929

-- Define the ellipse equation
def ellipse (x y n : ℝ) : Prop := x^2 + n*(y-1)^2 = n

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - 4*(y+3)^2 = 4

-- Define the tangency condition (discriminant = 0)
def tangent_condition (n : ℝ) : Prop := (24-2*n)^2 - 4*(4+n)*40 = 0

-- Theorem statement
theorem ellipse_hyperbola_tangency :
  ∃ n₁ n₂ : ℝ, 
    (abs (n₁ - 62.20625) < 0.00001) ∧ 
    (abs (n₂ - 1.66875) < 0.00001) ∧
    (∀ x y : ℝ, ellipse x y n₁ ∧ hyperbola x y → tangent_condition n₁) ∧
    (∀ x y : ℝ, ellipse x y n₂ ∧ hyperbola x y → tangent_condition n₂) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l2259_225929


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l2259_225989

/-- Given a mixture where the volume fraction of alcohol is 2/7 and the volume fraction of water is 3/7,
    the ratio of the volume of alcohol to the volume of water is 2:3. -/
theorem alcohol_water_ratio (mixture : ℚ → ℚ) (h1 : mixture 1 = 2/7) (h2 : mixture 2 = 3/7) :
  (mixture 1) / (mixture 2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l2259_225989


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2259_225986

/-- Given a trapezoid ABCD where the ratio of the area of triangle ABC to the area of triangle ADC
    is 7:3, and AB + CD = 280, prove that AB = 196. -/
theorem trapezoid_segment_length (AB CD : ℝ) (h : ℝ) : 
  (AB * h / 2) / (CD * h / 2) = 7 / 3 →
  AB + CD = 280 →
  AB = 196 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2259_225986


namespace NUMINAMATH_CALUDE_bill_face_value_l2259_225982

/-- The face value of a bill given its true discount, due time, and discount rate -/
def face_value (true_discount : ℚ) (due_time_months : ℚ) (annual_rate : ℚ) : ℚ :=
  let rate_time := annual_rate * (due_time_months / 12)
  true_discount * (100 + 100 * rate_time) / (100 * rate_time)

/-- Theorem stating that the face value of the bill is 1764 given the specified conditions -/
theorem bill_face_value :
  face_value 189 9 (16/100) = 1764 := by
  sorry

end NUMINAMATH_CALUDE_bill_face_value_l2259_225982


namespace NUMINAMATH_CALUDE_cube_root_square_root_l2259_225907

theorem cube_root_square_root (x : ℝ) : (2 * x)^3 = 216 → (x + 6)^(1/2) = 3 ∨ (x + 6)^(1/2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_root_l2259_225907


namespace NUMINAMATH_CALUDE_max_value_theorem_l2259_225969

theorem max_value_theorem (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) 
  (hb : -1 ≤ b ∧ b ≤ 1) 
  (hc : -1 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 1 ∧ 
  ∃ (x y z : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ -1 ≤ z ∧ z ≤ 1 ∧ 
    Real.sqrt (x^2 * y^2 * z^2) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2259_225969


namespace NUMINAMATH_CALUDE_translation_complex_plane_l2259_225995

/-- A translation in the complex plane that takes 1 + 3i to 5 + 7i also takes 2 - i to 6 + 3i -/
theorem translation_complex_plane : 
  ∀ (f : ℂ → ℂ), 
  (∀ z : ℂ, ∃ w : ℂ, f z = z + w) → -- f is a translation
  (f (1 + 3*I) = 5 + 7*I) →         -- f takes 1 + 3i to 5 + 7i
  (f (2 - I) = 6 + 3*I) :=          -- f takes 2 - i to 6 + 3i
by sorry

end NUMINAMATH_CALUDE_translation_complex_plane_l2259_225995


namespace NUMINAMATH_CALUDE_least_valid_number_l2259_225994

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ) (p : ℕ), 
    d ≥ 1 ∧ d ≤ 9 ∧
    n = 10^p * d + (n % 10^p) ∧
    10^p * d + (n % 10^p) = 17 * (n % 10^p)

theorem least_valid_number : 
  is_valid_number 10625 ∧ 
  ∀ (m : ℕ), m < 10625 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l2259_225994


namespace NUMINAMATH_CALUDE_system_solution_l2259_225978

theorem system_solution (x y a : ℝ) : 
  x = 3 ∧ 
  4 * x + y = a ∧ 
  3 * x + 4 * y^2 = 3 * a → 
  a = 15 ∨ a = 9.75 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2259_225978


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2259_225943

theorem polynomial_evaluation (f : ℝ → ℝ) :
  (∀ x, f (x^2 + 2) = x^4 + 6*x^2 + 4) →
  (∀ x, f (x^2 - 2) = x^4 - 2*x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2259_225943


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_129_l2259_225948

theorem first_nonzero_digit_of_one_over_129 :
  ∃ (n : ℕ) (r : ℚ), (1 : ℚ) / 129 = (n : ℚ) / 10^(n+1) + r ∧ 0 ≤ r ∧ r < 1 / 10^(n+1) ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_129_l2259_225948


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2259_225962

theorem sum_of_three_consecutive_cubes_divisible_by_nine (a : ℤ) :
  ∃ k : ℤ, a^3 + (a+1)^3 + (a+2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2259_225962


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2259_225930

theorem concentric_circles_area_ratio : 
  let small_diameter : ℝ := 2
  let large_diameter : ℝ := 4
  let small_radius : ℝ := small_diameter / 2
  let large_radius : ℝ := large_diameter / 2
  let small_area : ℝ := π * small_radius^2
  let large_area : ℝ := π * large_radius^2
  let area_between : ℝ := large_area - small_area
  area_between / small_area = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2259_225930


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2259_225944

/-- Given a circle with the following properties:
  - A chord of length 18
  - The chord is intersected by a diameter at a point
  - The intersection point is 7 units from the center
  - The intersection point divides the chord in the ratio 2:1
  Prove that the radius of the circle is 11 -/
theorem circle_radius_proof (chord_length : ℝ) (intersection_distance : ℝ) 
  (h1 : chord_length = 18)
  (h2 : intersection_distance = 7)
  (h3 : ∃ (a b : ℝ), a + b = chord_length ∧ a = 2 * b) :
  ∃ (radius : ℝ), radius = 11 ∧ radius^2 = intersection_distance^2 + (chord_length^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2259_225944


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l2259_225940

/-- The sum of the first n odd natural numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The sum of the first n even natural numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The number of odd terms from 1 to 2023 -/
def n_odd : ℕ := (2023 - 1) / 2 + 1

/-- The number of even terms from 2 to 2022 -/
def n_even : ℕ := (2022 - 2) / 2 + 1

theorem odd_even_sum_difference : 
  sum_odd n_odd - sum_even n_even = 22 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l2259_225940


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2259_225935

theorem problem_1 : (-1/2)⁻¹ + (3 - Real.pi)^0 + (-3)^2 = 8 := by sorry

theorem problem_2 (a : ℝ) : a^2 * a^4 - (-2*a^2)^3 - 3*a^2 + a^2 = 9*a^6 - 2*a^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2259_225935


namespace NUMINAMATH_CALUDE_nut_mixture_ratio_l2259_225910

/-- Given a mixture of nuts where the ratio of almonds to walnuts is x:2 by weight,
    and there are 200 pounds of almonds in 280 pounds of the mixture,
    prove that the ratio of almonds to walnuts is 2.5:1. -/
theorem nut_mixture_ratio (x : ℝ) : 
  x / 2 = 200 / 80 → x / 2 = 2.5 := by sorry

end NUMINAMATH_CALUDE_nut_mixture_ratio_l2259_225910


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_l2259_225903

/-- Given a truncated pyramid with base areas S₁ and S₂ (S₁ < S₂) and volume V,
    the volume of the complete pyramid is (V * S₂ * √S₂) / (S₂ * √S₂ - S₁ * √S₁) -/
theorem truncated_pyramid_volume 
  (S₁ S₂ V : ℝ) 
  (h₁ : 0 < S₁) 
  (h₂ : 0 < S₂) 
  (h₃ : S₁ < S₂) 
  (h₄ : 0 < V) : 
  ∃ (V_full : ℝ), V_full = (V * S₂ * Real.sqrt S₂) / (S₂ * Real.sqrt S₂ - S₁ * Real.sqrt S₁) := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_l2259_225903


namespace NUMINAMATH_CALUDE_bookstore_discount_theorem_l2259_225988

variable (Book : Type)
variable (bookstore : Set Book)
variable (discounted_by_20_percent : Book → Prop)

theorem bookstore_discount_theorem 
  (h : ¬ ∀ b ∈ bookstore, discounted_by_20_percent b) : 
  (∃ b ∈ bookstore, ¬ discounted_by_20_percent b) ∧ 
  (¬ ∀ b ∈ bookstore, discounted_by_20_percent b) := by
  sorry

end NUMINAMATH_CALUDE_bookstore_discount_theorem_l2259_225988


namespace NUMINAMATH_CALUDE_right_triangle_with_specific_median_l2259_225912

/-- A right triangle with a median to the hypotenuse -/
structure RightTriangleWithMedian where
  a : ℝ  -- First leg
  b : ℝ  -- Second leg
  c : ℝ  -- Hypotenuse
  m : ℝ  -- Median to the hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  median_property : m = c / 2  -- Property of the median to the hypotenuse
  perimeter_difference : b - a = 1  -- Difference in perimeters of smaller triangles

/-- The sides of the triangle satisfy the given conditions -/
theorem right_triangle_with_specific_median (t : RightTriangleWithMedian) 
  (h1 : t.a + t.m = 8)  -- Perimeter of one smaller triangle
  (h2 : t.b + t.m = 9)  -- Perimeter of the other smaller triangle
  : t.a = 3 ∧ t.b = 4 ∧ t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_with_specific_median_l2259_225912


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2259_225902

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 9)
  (h_a5 : a 5 = 33) :
  ∃ d : ℝ, d = 8 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2259_225902


namespace NUMINAMATH_CALUDE_truck_to_car_ratio_l2259_225931

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The total number of people needed to lift 6 cars and 3 trucks -/
def total_people : ℕ := 60

/-- The number of cars that can be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks that can be lifted -/
def num_trucks : ℕ := 3

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := (total_people - num_cars * people_per_car) / num_trucks

theorem truck_to_car_ratio :
  (people_per_truck : ℚ) / people_per_car = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_truck_to_car_ratio_l2259_225931
