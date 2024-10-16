import Mathlib

namespace NUMINAMATH_CALUDE_anil_tomato_production_l2201_220182

/-- Represents the number of tomatoes in a square backyard -/
def TomatoCount (side : ℕ) : ℕ := side * side

/-- Proves that given the conditions of Anil's tomato garden, he produced 4356 tomatoes this year -/
theorem anil_tomato_production : 
  ∃ (last_year current_year : ℕ),
    TomatoCount current_year = TomatoCount last_year + 131 ∧
    current_year > last_year ∧
    TomatoCount current_year = 4356 := by
  sorry


end NUMINAMATH_CALUDE_anil_tomato_production_l2201_220182


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2201_220163

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 →
      (3 * x + 1) / ((x - 4) * (x - 2)^2) =
      P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 13/4 ∧ Q = -13/4 ∧ R = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2201_220163


namespace NUMINAMATH_CALUDE_odd_function_inequality_l2201_220149

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the condition for x > 0
def positive_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (deriv f x) + 2 * f x > 0

-- State the theorem
theorem odd_function_inequality (h1 : is_odd f) (h2 : positive_condition f) :
  4 * f 2 < 9 * f 3 :=
sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l2201_220149


namespace NUMINAMATH_CALUDE_series_sum_l2201_220108

/-- The sum of the series $\sum_{n=1}^{\infty} \frac{3^n}{9^n - 1}$ is equal to $\frac{1}{2}$ -/
theorem series_sum : ∑' n, (3^n : ℝ) / (9^n - 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_l2201_220108


namespace NUMINAMATH_CALUDE_parentheses_placement_l2201_220167

theorem parentheses_placement :
  (7 * (9 + 12 / 3) = 91) ∧
  ((7 * 9 + 12) / 3 = 25) ∧
  (7 * (9 + 12) / 3 = 49) ∧
  ((48 * 6) / (48 * 6) = 1) := by
  sorry

end NUMINAMATH_CALUDE_parentheses_placement_l2201_220167


namespace NUMINAMATH_CALUDE_dividing_line_theorem_l2201_220177

/-- Represents a disk in 2D space -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of five disks -/
structure DiskConfiguration where
  disks : Fin 5 → Disk
  square_vertices : Fin 4 → ℝ × ℝ
  aligned_centers : Fin 3 → ℝ × ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The center of a square given its vertices -/
def square_center (vertices : Fin 4 → ℝ × ℝ) : ℝ × ℝ := sorry

/-- Calculates the area of the figure formed by the disks on one side of a line -/
def area_on_side (config : DiskConfiguration) (line : Line) : ℝ := sorry

/-- States that the line passing through the square center and the fifth disk's center
    divides the total area of the five disks into two equal parts -/
theorem dividing_line_theorem (config : DiskConfiguration) :
  let square_center := square_center config.square_vertices
  let fifth_disk_center := (config.disks 4).center
  let dividing_line := Line.mk square_center fifth_disk_center
  area_on_side config dividing_line = (area_on_side config dividing_line) / 2 := by sorry

end NUMINAMATH_CALUDE_dividing_line_theorem_l2201_220177


namespace NUMINAMATH_CALUDE_triangle_angle_solution_l2201_220192

theorem triangle_angle_solution (x : ℝ) : 
  40 + 3 * x + (x + 10) = 180 → x = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_solution_l2201_220192


namespace NUMINAMATH_CALUDE_infinitely_many_special_triangles_l2201_220125

/-- A triangle with integer side lengths and area, where one side is 4 and the difference between the other two sides is 2. -/
structure SpecialTriangle where
  a : ℕ+  -- First side length
  b : ℕ+  -- Second side length
  c : ℕ+  -- Third side length (always 4)
  area : ℕ+  -- Area of the triangle
  h_c : c = 4  -- One side is 4
  h_diff : a - b = 2 ∨ b - a = 2  -- Difference between other two sides is 2
  h_triangle : a + b > c ∧ b + c > a ∧ a + c > b  -- Triangle inequality
  h_area : 4 * area ^ 2 = (a + b + c) * (a + b - c) * (b + c - a) * (a + c - b)  -- Heron's formula

/-- There are infinitely many special triangles. -/
theorem infinitely_many_special_triangles : ∀ n : ℕ, ∃ m > n, ∃ t : SpecialTriangle, m = t.a.val := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_special_triangles_l2201_220125


namespace NUMINAMATH_CALUDE_hash_3_7_l2201_220126

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem
theorem hash_3_7 : hash 3 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_hash_3_7_l2201_220126


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2201_220124

/-- The sum of three specific repeating decimals is 2 -/
theorem sum_of_repeating_decimals : ∃ (x y z : ℚ),
  (∀ n : ℕ, (10 * x - x) * 10^n = 3 * 10^n) ∧
  (∀ n : ℕ, (10 * y - y) * 10^n = 6 * 10^n) ∧
  (∀ n : ℕ, (10 * z - z) * 10^n = 9 * 10^n) ∧
  x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2201_220124


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l2201_220199

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Part 1: Range of m when p is a sufficient condition for q
theorem range_of_m : 
  (∀ x, p x → ∀ m, q x m) → 
  {m : ℝ | m ≥ 4} = {m : ℝ | m > 0} := 
by sorry

-- Part 2: Range of x when m=5, p ∨ q is true, and p ∧ q is false
theorem range_of_x : 
  {x : ℝ | (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5)} = 
  {x : ℝ | -4 ≤ x ∧ x < -1} ∪ {x : ℝ | 5 < x ∧ x ≤ 6} := 
by sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l2201_220199


namespace NUMINAMATH_CALUDE_unique_solution_for_system_l2201_220174

theorem unique_solution_for_system :
  ∃! (x y : ℝ), (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x + 1) + (y + 1)) ∧ x = 8 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l2201_220174


namespace NUMINAMATH_CALUDE_simplify_expression_l2201_220132

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2201_220132


namespace NUMINAMATH_CALUDE_tan_thirty_degrees_l2201_220178

theorem tan_thirty_degrees : Real.tan (π / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirty_degrees_l2201_220178


namespace NUMINAMATH_CALUDE_remainder_of_n_l2201_220119

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 1) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l2201_220119


namespace NUMINAMATH_CALUDE_container_weights_l2201_220107

theorem container_weights (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (w1 : x + y = 110) (w2 : y + z = 130) (w3 : z + x = 150) :
  x + y + z = 195 := by
  sorry

end NUMINAMATH_CALUDE_container_weights_l2201_220107


namespace NUMINAMATH_CALUDE_four_variable_equation_consecutive_evens_l2201_220102

theorem four_variable_equation_consecutive_evens :
  ∃ (x y z w : ℕ), 
    (x + y + z + w = 100) ∧ 
    (∃ (k : ℕ), x = 2 * k) ∧
    (∃ (l : ℕ), y = 2 * l) ∧
    (∃ (m : ℕ), z = 2 * m) ∧
    (∃ (n : ℕ), w = 2 * n) ∧
    (y = x + 2) ∧
    (z = x + 4) ∧
    (w = x + 6) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (w > 0) := by
  sorry

end NUMINAMATH_CALUDE_four_variable_equation_consecutive_evens_l2201_220102


namespace NUMINAMATH_CALUDE_mower_blades_cost_l2201_220183

def total_earned : ℕ := 104
def num_games : ℕ := 7
def game_price : ℕ := 9

theorem mower_blades_cost (remaining : ℕ) 
  (h1 : remaining = num_games * game_price) 
  (h2 : remaining + (total_earned - remaining) = total_earned) : 
  total_earned - remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_mower_blades_cost_l2201_220183


namespace NUMINAMATH_CALUDE_device_working_prob_correct_l2201_220151

/-- A device with two components, each having a probability of failure --/
structure Device where
  /-- The probability of a single component being damaged --/
  component_failure_prob : ℝ
  /-- Assumption that the component failure probability is between 0 and 1 --/
  h_prob_range : 0 ≤ component_failure_prob ∧ component_failure_prob ≤ 1

/-- The probability of the device working --/
def device_working_prob (d : Device) : ℝ :=
  (1 - d.component_failure_prob) * (1 - d.component_failure_prob)

/-- Theorem stating that for a device with component failure probability of 0.1,
    the probability of the device working is 0.81 --/
theorem device_working_prob_correct (d : Device) 
    (h : d.component_failure_prob = 0.1) : 
    device_working_prob d = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_device_working_prob_correct_l2201_220151


namespace NUMINAMATH_CALUDE_trapezoid_properties_l2201_220166

-- Define the trapezoid and its properties
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  h : ℝ
  parallel_AB_CD : AB ≠ CD

-- Define the midpoints
def midpoint_M (t : Trapezoid) : ℝ × ℝ := sorry
def midpoint_N (t : Trapezoid) : ℝ × ℝ := sorry
def midpoint_P (t : Trapezoid) : ℝ × ℝ := sorry

-- Define the length of MN
def length_MN (t : Trapezoid) : ℝ := sorry

-- Define the area of triangle MNP
def area_MNP (t : Trapezoid) : ℝ := sorry

-- Theorem statement
theorem trapezoid_properties (t : Trapezoid) 
  (h_AB : t.AB = 15) 
  (h_CD : t.CD = 24) 
  (h_h : t.h = 14) : 
  length_MN t = 4.5 ∧ area_MNP t = 15.75 := by sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l2201_220166


namespace NUMINAMATH_CALUDE_quadratic_solution_absolute_value_l2201_220172

theorem quadratic_solution_absolute_value : ∃ (x : ℝ), x^2 + 18*x + 81 = 0 ∧ |x| = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_absolute_value_l2201_220172


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2201_220161

theorem quadratic_function_property (a b : ℝ) (h1 : a ≠ b) : 
  let f := fun x => x^2 + a*x + b
  (f a = f b) → f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2201_220161


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2201_220118

theorem complex_equation_sum (a b : ℝ) : (1 + 2*I)*I = a + b*I → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2201_220118


namespace NUMINAMATH_CALUDE_equal_surface_area_implies_L_value_l2201_220165

/-- Given a cube with edge length 30 and a rectangular solid with edge lengths 20, 30, and L,
    if their surface areas are equal, then L = 42. -/
theorem equal_surface_area_implies_L_value (L : ℝ) : 
  (6 * 30 * 30 = 2 * 20 * 30 + 2 * 20 * L + 2 * 30 * L) → L = 42 := by
  sorry

#check equal_surface_area_implies_L_value

end NUMINAMATH_CALUDE_equal_surface_area_implies_L_value_l2201_220165


namespace NUMINAMATH_CALUDE_min_x_minus_y_l2201_220145

theorem min_x_minus_y (x y : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) →
  y ∈ Set.Icc 0 (2 * Real.pi) →
  2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1/2 →
  ∃ (z : Real), z = x - y ∧ ∀ (w : Real), w = x - y → z ≤ w ∧ z = -Real.pi/2 :=
by sorry

end NUMINAMATH_CALUDE_min_x_minus_y_l2201_220145


namespace NUMINAMATH_CALUDE_flour_measurement_l2201_220113

theorem flour_measurement (required : ℚ) (container : ℚ) (excess : ℚ) : 
  required = 15/4 ∧ container = 4/3 ∧ excess = 2/3 → 
  ∃ (n : ℕ), n * container = required - excess ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_flour_measurement_l2201_220113


namespace NUMINAMATH_CALUDE_arccos_sin_one_point_five_l2201_220164

theorem arccos_sin_one_point_five :
  Real.arccos (Real.sin 1.5) = π / 2 - 1.5 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_one_point_five_l2201_220164


namespace NUMINAMATH_CALUDE_gcd_g_x_l2201_220173

def g (x : ℤ) : ℤ := (3*x+5)*(9*x+4)*(11*x+8)*(x+11)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 34914 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 1760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l2201_220173


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2201_220154

theorem rectangular_box_surface_area 
  (x y z : ℝ) 
  (h1 : x > 0 ∧ y > 0 ∧ z > 0)
  (h2 : 4 * (x + y + z) = 140)
  (h3 : x^2 + y^2 + z^2 = 21^2) :
  2 * (x*y + x*z + y*z) = 784 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2201_220154


namespace NUMINAMATH_CALUDE_equation_solution_range_l2201_220120

theorem equation_solution_range (x a : ℝ) : 
  (2 * x + a) / (x - 1) = 1 → x > 0 → x ≠ 1 → a < -1 ∧ a ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2201_220120


namespace NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l2201_220187

theorem root_in_interval_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + x + a) 
  (h2 : a < 0) 
  (h3 : ∃ x ∈ Set.Ioo 0 1, f x = 0) : 
  -2 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_a_range_l2201_220187


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2201_220197

/-- Given two concentric circles with radii A and B (A > B), if the area between
    the circles is 15π square meters, then the length of a chord of the larger
    circle that is tangent to the smaller circle is 2√15 meters. -/
theorem chord_length_concentric_circles (A B : ℝ) (h1 : A > B) (h2 : A > 0) (h3 : B > 0)
    (h4 : π * A^2 - π * B^2 = 15 * π) :
    ∃ (c : ℝ), c^2 = 4 * 15 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2201_220197


namespace NUMINAMATH_CALUDE_stratified_sample_grade12_l2201_220129

/-- Represents the number of students in each grade and in the sample -/
structure SchoolSample where
  total : ℕ
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ
  sample10 : ℕ
  sample12 : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem stratified_sample_grade12 (s : SchoolSample) 
  (h_total : s.total = 1290)
  (h_grade10 : s.grade10 = 480)
  (h_grade_diff : s.grade11 = s.grade12 + 30)
  (h_sum : s.grade10 + s.grade11 + s.grade12 = s.total)
  (h_sample10 : s.sample10 = 96)
  (h_prop : s.sample10 / s.grade10 = s.sample12 / s.grade12) :
  s.sample12 = 78 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_grade12_l2201_220129


namespace NUMINAMATH_CALUDE_transmitted_word_is_parokhod_l2201_220137

/-- Represents a 5-digit binary number -/
def BinaryCode := Fin 32

/-- Represents a letter in the Russian alphabet -/
structure RussianLetter where
  code : BinaryCode
  char : Char

/-- Represents a word in Russian -/
def RussianWord := List RussianLetter

/-- Represents the set of received letters -/
def ReceivedSet := List RussianLetter

/-- Counts the number of 1s in a binary code -/
def countOnes (code : BinaryCode) : Nat :=
  sorry

/-- Checks if a word is valid given the received set -/
def isValidWord (word : RussianWord) (receivedSet : ReceivedSet) : Prop :=
  sorry

/-- The received set of letters -/
def receivedLetters : ReceivedSet :=
  sorry

/-- The theorem to prove -/
theorem transmitted_word_is_parokhod :
  ∃! (word : RussianWord),
    isValidWord word receivedLetters ∧
    word.map (λ l => l.char) = ['П', 'А', 'Р', 'О', 'Х', 'О', 'Д'] :=
  sorry

end NUMINAMATH_CALUDE_transmitted_word_is_parokhod_l2201_220137


namespace NUMINAMATH_CALUDE_jerry_shelf_items_l2201_220141

/-- Represents the number of items on Jerry's shelf --/
structure ShelfItems where
  actionFigures : ℕ
  books : ℕ
  videoGames : ℕ

/-- Calculates the total number of items on the shelf --/
def totalItems (items : ShelfItems) : ℕ :=
  items.actionFigures + items.books + items.videoGames

/-- Represents the changes made to the shelf items --/
structure ItemChanges where
  actionFiguresAdded : ℕ
  booksRemoved : ℕ
  videoGamesAdded : ℕ

/-- Applies changes to the shelf items --/
def applyChanges (items : ShelfItems) (changes : ItemChanges) : ShelfItems :=
  { actionFigures := items.actionFigures + changes.actionFiguresAdded,
    books := items.books - changes.booksRemoved,
    videoGames := items.videoGames + changes.videoGamesAdded }

theorem jerry_shelf_items :
  let initialItems : ShelfItems := { actionFigures := 4, books := 22, videoGames := 10 }
  let changes : ItemChanges := { actionFiguresAdded := 6, booksRemoved := 5, videoGamesAdded := 3 }
  let finalItems := applyChanges initialItems changes
  totalItems finalItems = 40 := by sorry

end NUMINAMATH_CALUDE_jerry_shelf_items_l2201_220141


namespace NUMINAMATH_CALUDE_parallel_sides_equal_or_complementary_l2201_220156

/-- Two angles in space -/
structure AngleInSpace where
  -- Define the necessary components of an angle in space
  -- This is a simplified representation
  measure : ℝ

/-- Predicate to check if two angles have parallel sides -/
def has_parallel_sides (a b : AngleInSpace) : Prop :=
  -- This is a placeholder for the actual condition of parallel sides
  True

/-- Predicate to check if two angles are equal -/
def are_equal (a b : AngleInSpace) : Prop :=
  a.measure = b.measure

/-- Predicate to check if two angles are complementary -/
def are_complementary (a b : AngleInSpace) : Prop :=
  a.measure + b.measure = 90

/-- Theorem: If two angles in space have parallel sides, 
    then they are either equal or complementary -/
theorem parallel_sides_equal_or_complementary (a b : AngleInSpace) :
  has_parallel_sides a b → (are_equal a b ∨ are_complementary a b) := by
  sorry

end NUMINAMATH_CALUDE_parallel_sides_equal_or_complementary_l2201_220156


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l2201_220157

/-- Represents the number of non-student tickets sold at an annual concert --/
def non_student_tickets : ℕ := 60

/-- Represents the number of student tickets sold at an annual concert --/
def student_tickets : ℕ := 150 - non_student_tickets

/-- The price of a student ticket in dollars --/
def student_price : ℕ := 5

/-- The price of a non-student ticket in dollars --/
def non_student_price : ℕ := 8

/-- The total revenue from ticket sales in dollars --/
def total_revenue : ℕ := 930

/-- The total number of tickets sold --/
def total_tickets : ℕ := 150

theorem concert_ticket_sales :
  (student_tickets * student_price + non_student_tickets * non_student_price = total_revenue) ∧
  (student_tickets + non_student_tickets = total_tickets) :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l2201_220157


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonals_l2201_220104

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midline : ℝ
  height : ℝ
  diagonal_angle : ℝ

/-- The diagonals of a trapezoid -/
def trapezoid_diagonals (t : Trapezoid) : ℝ × ℝ := sorry

/-- Theorem stating the diagonals of a specific trapezoid -/
theorem specific_trapezoid_diagonals :
  let t : Trapezoid := {
    midline := 7,
    height := 15 * Real.sqrt 3 / 7,
    diagonal_angle := 2 * π / 3  -- 120° in radians
  }
  trapezoid_diagonals t = (6, 10) := by sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonals_l2201_220104


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_l2201_220136

theorem complex_root_modulus_one (z : ℂ) (h : z^2 - z + 1 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_l2201_220136


namespace NUMINAMATH_CALUDE_solution_set_f_x_minus_one_gt_two_min_value_x_plus_2y_plus_2z_l2201_220105

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Theorem for part I
theorem solution_set_f_x_minus_one_gt_two :
  {x : ℝ | f (x - 1) > 2} = {x : ℝ | x < -1 ∨ x > 3} :=
sorry

-- Theorem for part II
theorem min_value_x_plus_2y_plus_2z (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) :
  ∃ (m : ℝ), m = -9 ∧ ∀ (x' y' z' : ℝ), f x' ^ 2 + y' ^ 2 + z' ^ 2 = 9 → x' + 2 * y' + 2 * z' ≥ m :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_x_minus_one_gt_two_min_value_x_plus_2y_plus_2z_l2201_220105


namespace NUMINAMATH_CALUDE_charley_beads_problem_l2201_220184

theorem charley_beads_problem (white_beads black_beads : ℕ) 
  (black_fraction : ℚ) (total_pulled : ℕ) :
  white_beads = 51 →
  black_beads = 90 →
  black_fraction = 1 / 6 →
  total_pulled = 32 →
  ∃ (white_fraction : ℚ),
    white_fraction * white_beads + black_fraction * black_beads = total_pulled ∧
    white_fraction = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_charley_beads_problem_l2201_220184


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2201_220140

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2201_220140


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2201_220170

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = x + f (f y)

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x + c for some real constant c -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2201_220170


namespace NUMINAMATH_CALUDE_total_soda_bottles_l2201_220193

/-- The number of regular soda bottles -/
def regular_soda : ℕ := 49

/-- The number of diet soda bottles -/
def diet_soda : ℕ := 40

/-- Theorem: The total number of regular and diet soda bottles is 89 -/
theorem total_soda_bottles : regular_soda + diet_soda = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_soda_bottles_l2201_220193


namespace NUMINAMATH_CALUDE_function_minimum_value_l2201_220147

/-- The function f(x) = x + a / (x - 2) where x > 2 and f(3) = 7 has a minimum value of 6 -/
theorem function_minimum_value (a : ℝ) : 
  (∀ x > 2, ∃ y, y = x + a / (x - 2)) → 
  (3 + a / (3 - 2) = 7) → 
  (∃ m : ℝ, ∀ x > 2, x + a / (x - 2) ≥ m ∧ ∃ x₀ > 2, x₀ + a / (x₀ - 2) = m) →
  (∀ x > 2, x + a / (x - 2) ≥ 6) ∧ ∃ x₀ > 2, x₀ + a / (x₀ - 2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_value_l2201_220147


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l2201_220144

/-- Given a quadratic function f(x) = ax^2 + 3x - 2, 
    if the slope of its tangent line at x = 2 is 7, then a = 1 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 + 3 * x - 2
  let f' : ℝ → ℝ := λ x => 2 * a * x + 3
  f' 2 = 7 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l2201_220144


namespace NUMINAMATH_CALUDE_grandsons_age_l2201_220128

theorem grandsons_age (grandson_age grandfather_age : ℕ) : 
  grandfather_age = 6 * grandson_age →
  grandfather_age + 4 + grandson_age + 4 = 78 →
  grandson_age = 10 := by
sorry

end NUMINAMATH_CALUDE_grandsons_age_l2201_220128


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l2201_220142

theorem parallelogram_perimeter (a b c d : ℕ) : 
  a^2 + b^2 = 130 →  -- sum of squares of diagonals
  c^2 + d^2 = 65 →   -- sum of squares of sides
  c + d = 11 →       -- sum of sides
  c * d = 28 →       -- product of sides
  2 * (c + d) = 22   -- perimeter
  := by sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l2201_220142


namespace NUMINAMATH_CALUDE_probability_B_outscores_A_is_correct_l2201_220101

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : Rat

/-- The probability that team B finishes with more points than team A -/
def probability_B_outscores_A (tournament : SoccerTournament) : Rat :=
  793 / 2048

/-- Theorem stating the probability that team B finishes with more points than team A -/
theorem probability_B_outscores_A_is_correct (tournament : SoccerTournament) 
  (h1 : tournament.num_teams = 8)
  (h2 : tournament.games_per_team = 7)
  (h3 : tournament.win_probability = 1 / 2) : 
  probability_B_outscores_A tournament = 793 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_B_outscores_A_is_correct_l2201_220101


namespace NUMINAMATH_CALUDE_min_value_theorem_l2201_220106

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → (x + y + 1) / (x * y) ≥ (a + b + 1) / (a * b)) →
  (a + b + 1) / (a * b) = 4 * Real.sqrt 3 + 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2201_220106


namespace NUMINAMATH_CALUDE_field_trip_bus_occupancy_l2201_220146

/-- Proves that given the conditions from the field trip problem, 
    the number of people in each bus is 18.0 --/
theorem field_trip_bus_occupancy 
  (num_vans : ℝ) 
  (num_buses : ℝ) 
  (people_per_van : ℝ) 
  (additional_people_in_buses : ℝ) 
  (h1 : num_vans = 6.0)
  (h2 : num_buses = 8.0)
  (h3 : people_per_van = 6.0)
  (h4 : additional_people_in_buses = 108.0) :
  (num_vans * people_per_van + additional_people_in_buses) / num_buses = 18.0 := by
  sorry

#eval (6.0 * 6.0 + 108.0) / 8.0  -- This should evaluate to 18.0

end NUMINAMATH_CALUDE_field_trip_bus_occupancy_l2201_220146


namespace NUMINAMATH_CALUDE_soccer_league_teams_l2201_220171

theorem soccer_league_teams (n : ℕ) : n * (n - 1) / 2 = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l2201_220171


namespace NUMINAMATH_CALUDE_sandy_change_theorem_l2201_220123

def cappuccino_price : ℝ := 2
def iced_tea_price : ℝ := 3
def cafe_latte_price : ℝ := 1.5
def espresso_price : ℝ := 1

def sandy_order_cappuccinos : ℕ := 3
def sandy_order_iced_teas : ℕ := 2
def sandy_order_cafe_lattes : ℕ := 2
def sandy_order_espressos : ℕ := 2

def paid_amount : ℝ := 20

theorem sandy_change_theorem :
  paid_amount - (cappuccino_price * sandy_order_cappuccinos +
                 iced_tea_price * sandy_order_iced_teas +
                 cafe_latte_price * sandy_order_cafe_lattes +
                 espresso_price * sandy_order_espressos) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_theorem_l2201_220123


namespace NUMINAMATH_CALUDE_point_not_on_line_l2201_220159

theorem point_not_on_line (m b : ℝ) (h : m + b < 0) :
  ¬(∃ (x y : ℝ), y = m * x + b ∧ x = 0 ∧ y = 20) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l2201_220159


namespace NUMINAMATH_CALUDE_solve_for_a_when_x_is_zero_range_of_a_when_x_is_one_l2201_220188

-- Define the equation
def equation (a : ℚ) (x : ℚ) : Prop :=
  |a| * x = |a + 1| - x

-- Theorem 1
theorem solve_for_a_when_x_is_zero :
  ∀ a : ℚ, equation a 0 → a = -1 :=
sorry

-- Theorem 2
theorem range_of_a_when_x_is_one :
  ∀ a : ℚ, equation a 1 → a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_solve_for_a_when_x_is_zero_range_of_a_when_x_is_one_l2201_220188


namespace NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_for_ln_l2201_220148

theorem sqrt_necessary_not_sufficient_for_ln :
  (∀ x y, x > 0 ∧ y > 0 → (Real.log x > Real.log y → Real.sqrt x > Real.sqrt y)) ∧
  (∃ x y, Real.sqrt x > Real.sqrt y ∧ ¬(Real.log x > Real.log y)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_for_ln_l2201_220148


namespace NUMINAMATH_CALUDE_remainder_seven_fourth_mod_hundred_l2201_220155

theorem remainder_seven_fourth_mod_hundred : 7^4 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_fourth_mod_hundred_l2201_220155


namespace NUMINAMATH_CALUDE_stating_max_s_value_l2201_220191

/-- Represents the dimensions of the large rectangle to be tiled -/
def large_rectangle : ℕ × ℕ := (1993, 2000)

/-- Represents the area of a 2 × 2 square -/
def square_area : ℕ := 4

/-- Represents the area of a P-rectangle -/
def p_rectangle_area : ℕ := 5

/-- Represents the area of an S-rectangle -/
def s_rectangle_area : ℕ := 4

/-- Represents the total area of the large rectangle -/
def total_area : ℕ := large_rectangle.1 * large_rectangle.2

/-- 
Theorem stating that the maximum value of s (sum of 2 × 2 squares and S-rectangles) 
used to tile the large rectangle is 996500
-/
theorem max_s_value : 
  ∀ a b c : ℕ, 
  a * square_area + b * p_rectangle_area + c * s_rectangle_area = total_area →
  a + c ≤ 996500 :=
sorry

end NUMINAMATH_CALUDE_stating_max_s_value_l2201_220191


namespace NUMINAMATH_CALUDE_no_preimage_set_l2201_220180

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem no_preimage_set (p : ℝ) : 
  (∀ x : ℝ, f x ≠ p) ↔ p ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_no_preimage_set_l2201_220180


namespace NUMINAMATH_CALUDE_drop_recording_l2201_220153

/-- Represents the change in water level in meters -/
def WaterLevelChange : Type := ℝ

/-- Records a rise in water level -/
def recordRise (meters : ℝ) : WaterLevelChange := meters

/-- Records a drop in water level -/
def recordDrop (meters : ℝ) : WaterLevelChange := -meters

/-- The theorem stating how a drop in water level should be recorded -/
theorem drop_recording (rise : ℝ) (drop : ℝ) :
  recordRise rise = rise → recordDrop drop = -drop :=
by sorry

end NUMINAMATH_CALUDE_drop_recording_l2201_220153


namespace NUMINAMATH_CALUDE_complement_of_intersection_l2201_220198

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_of_intersection (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l2201_220198


namespace NUMINAMATH_CALUDE_number_value_relationship_l2201_220194

theorem number_value_relationship (n v : ℝ) : 
  n > 0 → n = 7 → n - 4 = 21 * v → v = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_number_value_relationship_l2201_220194


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_13_l2201_220152

theorem largest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 13 = 0 → n ≤ 987 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_13_l2201_220152


namespace NUMINAMATH_CALUDE_derivative_value_at_two_l2201_220110

theorem derivative_value_at_two (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, HasDerivAt f (f' x) x) →
  (∀ x, f x = x^2 + 3*x*(f' 2)) →
  f' 2 = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_value_at_two_l2201_220110


namespace NUMINAMATH_CALUDE_solutions_to_equation_unique_solutions_l2201_220168

-- Define the equation
def equation (s : ℝ) : ℝ := 12 * s^2 + 2 * s

-- Theorem stating that 0.5 and -2/3 are solutions to the equation when t = 4
theorem solutions_to_equation :
  equation (1/2) = 4 ∧ equation (-2/3) = 4 :=
by sorry

-- Theorem stating that these are the only solutions
theorem unique_solutions (s : ℝ) :
  equation s = 4 ↔ s = 1/2 ∨ s = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_unique_solutions_l2201_220168


namespace NUMINAMATH_CALUDE_total_amount_theorem_l2201_220139

/-- Calculate the selling price of an item given its purchase price and loss percentage -/
def sellingPrice (purchasePrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  purchasePrice * (1 - lossPercentage / 100)

/-- Calculate the total amount received from selling three items -/
def totalAmountReceived (price1 price2 price3 : ℚ) (loss1 loss2 loss3 : ℚ) : ℚ :=
  sellingPrice price1 loss1 + sellingPrice price2 loss2 + sellingPrice price3 loss3

theorem total_amount_theorem (price1 price2 price3 loss1 loss2 loss3 : ℚ) :
  price1 = 600 ∧ price2 = 800 ∧ price3 = 1000 ∧
  loss1 = 20 ∧ loss2 = 25 ∧ loss3 = 30 →
  totalAmountReceived price1 price2 price3 loss1 loss2 loss3 = 1780 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l2201_220139


namespace NUMINAMATH_CALUDE_tree_height_ratio_l2201_220111

/-- Given three trees with specific height relationships, prove that the height of the smallest tree
    is 1/4 of the height of the middle-sized tree. -/
theorem tree_height_ratio :
  ∀ (h₁ h₂ h₃ : ℝ),
  h₁ = 108 →
  h₂ = h₁ / 2 - 6 →
  h₃ = 12 →
  h₃ / h₂ = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tree_height_ratio_l2201_220111


namespace NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l2201_220186

theorem quadratic_root_implies_v_value : ∀ v : ℝ,
  ((-25 - Real.sqrt 361) / 12 : ℝ) ∈ {x : ℝ | 6 * x^2 + 25 * x + v = 0} →
  v = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_v_value_l2201_220186


namespace NUMINAMATH_CALUDE_polynomial_condition_implies_linear_l2201_220196

/-- A polynomial with real coefficients -/
def RealPolynomial : Type := ℝ → ℝ

/-- The condition that P(x + y) is rational when P(x) and P(y) are rational -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, (∃ q₁ q₂ : ℚ, P x = q₁ ∧ P y = q₂) → ∃ q : ℚ, P (x + y) = q

/-- The theorem stating that polynomials satisfying the condition must be linear with rational coefficients -/
theorem polynomial_condition_implies_linear
  (P : RealPolynomial)
  (h : SatisfiesCondition P) :
  ∃ a b : ℚ, ∀ x : ℝ, P x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_polynomial_condition_implies_linear_l2201_220196


namespace NUMINAMATH_CALUDE_divisibility_equations_solutions_l2201_220189

theorem divisibility_equations_solutions :
  (∀ x : ℤ, (x - 1 ∣ x + 3) ↔ x ∈ ({-3, -1, 0, 2, 3, 5} : Set ℤ)) ∧
  (∀ x : ℤ, (x + 2 ∣ x^2 + 2) ↔ x ∈ ({-8, -5, -4, -3, -1, 0, 1, 4} : Set ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equations_solutions_l2201_220189


namespace NUMINAMATH_CALUDE_lcm_18_36_l2201_220138

theorem lcm_18_36 : Nat.lcm 18 36 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_36_l2201_220138


namespace NUMINAMATH_CALUDE_k_range_l2201_220121

open Real

theorem k_range (k : ℝ) : 
  (∀ x > 1, k * (exp (k * x) + 1) - (1 / x + 1) * log x > 0) → 
  k > 1 / exp 1 :=
by sorry

end NUMINAMATH_CALUDE_k_range_l2201_220121


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l2201_220181

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relationship between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relationship between planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the "within" relationship between lines and planes
variable (within : Line → Plane → Prop)

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (b : Line) (α β : Plane) 
  (h1 : parallel_line_plane b α) 
  (h2 : parallel_plane α β) : 
  parallel_line_plane b β ∨ within b β := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l2201_220181


namespace NUMINAMATH_CALUDE_line_through_P_parallel_to_tangent_at_M_l2201_220190

/-- The curve y = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 6 * x - 4

/-- Point P -/
def P : ℝ × ℝ := (-1, 2)

/-- Point M -/
def M : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at point M -/
def k : ℝ := f' M.1

/-- The equation of the line passing through P and parallel to the tangent line at M -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem line_through_P_parallel_to_tangent_at_M :
  line_equation P.1 P.2 ∧
  ∀ x y, line_equation x y → (y - P.2) = k * (x - P.1) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_parallel_to_tangent_at_M_l2201_220190


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l2201_220160

/-- Given a triangle ABC with vertices A(x₁, y₁), B(x₂, y₂), C(x₃, y₃),
    and points E on AC and F on AB such that AE:EC = n:l and AF:FB = m:l,
    prove that the intersection point P of BE and CF has coordinates
    ((lx₁ + mx₂ + nx₃)/(l + m + n), (ly₁ + my₂ + ny₃)/(l + m + n)) -/
theorem intersection_point_coordinates
  (x₁ y₁ x₂ y₂ x₃ y₃ l m n : ℝ)
  (h₁ : m ≠ -l)
  (h₂ : n ≠ -l)
  (h₃ : l + m + n ≠ 0) :
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let C := (x₃, y₃)
  let E := ((l * x₁ + n * x₃) / (l + n), (l * y₁ + n * y₃) / (l + n))
  let F := ((l * x₁ + m * x₂) / (l + m), (l * y₁ + m * y₂) / (l + m))
  let P := ((l * x₁ + m * x₂ + n * x₃) / (l + m + n), (l * y₁ + m * y₂ + n * y₃) / (l + m + n))
  ∃ (t : ℝ), (P.1 - E.1) / (B.1 - E.1) = t ∧ (P.2 - E.2) / (B.2 - E.2) = t ∧
             (P.1 - F.1) / (C.1 - F.1) = (1 - t) ∧ (P.2 - F.2) / (C.2 - F.2) = (1 - t) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l2201_220160


namespace NUMINAMATH_CALUDE_tangent_point_bounds_l2201_220134

/-- A point (a,b) through which two distinct tangent lines can be drawn to the curve y = e^x -/
structure TangentPoint where
  a : ℝ
  b : ℝ
  two_tangents : ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    b = Real.exp t₁ * (a - t₁ + 1) ∧
    b = Real.exp t₂ * (a - t₂ + 1)

/-- If two distinct tangent lines to y = e^x can be drawn through (a,b), then 0 < b < e^a -/
theorem tangent_point_bounds (p : TangentPoint) : 0 < p.b ∧ p.b < Real.exp p.a := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_bounds_l2201_220134


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2201_220114

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / (rate * time)

/-- Theorem: Given a loan with 12% p.a. simple interest rate, if the interest
    amount after 10 years is Rs. 1500, then the principal amount borrowed was Rs. 1250. -/
theorem loan_principal_calculation (rate : ℚ) (time : ℕ) (interest : ℕ) :
  rate = 12 → time = 10 → interest = 1500 →
  calculate_principal rate time interest = 1250 := by
  sorry

#eval calculate_principal 12 10 1500

end NUMINAMATH_CALUDE_loan_principal_calculation_l2201_220114


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2201_220103

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 2*x - 8 = 0) :
  x^3 - 2*x^2 - 8*x + 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2201_220103


namespace NUMINAMATH_CALUDE_distance_traveled_l2201_220127

/-- Given a person traveling at 6 km/h for 10 minutes, prove that the distance traveled is 1000 meters. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 1/6 → speed * time * 1000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2201_220127


namespace NUMINAMATH_CALUDE_extraneous_root_implies_a_value_l2201_220162

/-- The equation has an extraneous root if x = 3 is a solution to the polynomial form of the equation -/
def has_extraneous_root (a : ℚ) : Prop :=
  ∃ x : ℚ, x = 3 ∧ x - 2*a = 2*(x - 3)

/-- The original equation -/
def original_equation (x a : ℚ) : Prop :=
  x / (x - 3) - 2*a / (x - 3) = 2

theorem extraneous_root_implies_a_value :
  ∀ a : ℚ, has_extraneous_root a → a = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_extraneous_root_implies_a_value_l2201_220162


namespace NUMINAMATH_CALUDE_change_is_five_l2201_220133

/-- Given a meal cost, drink cost, tip percentage, and payment amount, 
    calculate the change received. -/
def calculate_change (meal_cost drink_cost tip_percentage payment : ℚ) : ℚ :=
  let total_before_tip := meal_cost + drink_cost
  let tip_amount := total_before_tip * (tip_percentage / 100)
  let total_with_tip := total_before_tip + tip_amount
  payment - total_with_tip

/-- Theorem stating that given the specified costs and payment, 
    the change received is $5. -/
theorem change_is_five :
  calculate_change 10 2.5 20 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_change_is_five_l2201_220133


namespace NUMINAMATH_CALUDE_jenny_recycling_payment_jenny_gets_three_cents_per_can_l2201_220112

/-- Calculates the amount Jenny gets paid per can given the recycling conditions -/
theorem jenny_recycling_payment (bottle_weight : ℕ) (can_weight : ℕ) (total_weight : ℕ) 
  (num_cans : ℕ) (bottle_payment : ℕ) (total_payment : ℕ) : ℕ :=
  let remaining_weight := total_weight - (num_cans * can_weight)
  let num_bottles := remaining_weight / bottle_weight
  let bottle_total_payment := num_bottles * bottle_payment
  let can_total_payment := total_payment - bottle_total_payment
  can_total_payment / num_cans

/-- Proves that Jenny gets paid 3 cents per can under the given conditions -/
theorem jenny_gets_three_cents_per_can : 
  jenny_recycling_payment 6 2 100 20 10 160 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jenny_recycling_payment_jenny_gets_three_cents_per_can_l2201_220112


namespace NUMINAMATH_CALUDE_subtraction_problem_l2201_220109

theorem subtraction_problem (minuend : ℝ) (difference : ℝ) (subtrahend : ℝ)
  (h1 : minuend = 98.2)
  (h2 : difference = 17.03)
  (h3 : subtrahend = minuend - difference) :
  subtrahend = 81.17 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2201_220109


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l2201_220117

theorem estimate_larger_than_original 
  (x y a b ε : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x > y) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hab : a > b) 
  (hε : ε > 0) : 
  (a * x + ε) - (b * y - ε) > a * x - b * y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l2201_220117


namespace NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l2201_220175

theorem distinct_naturals_reciprocal_sum (x y z : ℕ) : 
  x < y ∧ y < z ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  ∃ (a : ℕ), (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = a
  →
  x = 2 ∧ y = 3 ∧ z = 6 := by
sorry

end NUMINAMATH_CALUDE_distinct_naturals_reciprocal_sum_l2201_220175


namespace NUMINAMATH_CALUDE_probability_two_in_same_box_is_12_25_l2201_220143

def num_balls : ℕ := 3
def num_boxes : ℕ := 5

def total_placements : ℕ := num_boxes ^ num_balls

def two_in_same_box_placements : ℕ := 
  (num_balls.choose 2) * (num_boxes.choose 1) * (num_boxes - 1)

def probability_two_in_same_box : ℚ := 
  two_in_same_box_placements / total_placements

theorem probability_two_in_same_box_is_12_25 : 
  probability_two_in_same_box = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_probability_two_in_same_box_is_12_25_l2201_220143


namespace NUMINAMATH_CALUDE_only_prop2_and_prop4_true_l2201_220135

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def line_parallel (a b : Line) : Prop := sorry
def line_perpendicular (a b : Line) : Prop := sorry
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the propositions
def proposition1 (m n : Line) (α β : Plane) : Prop :=
  parallel α β ∧ contained_in m β ∧ contained_in n α → line_parallel m n

def proposition2 (m n : Line) (α β : Plane) : Prop :=
  parallel α β ∧ line_perpendicular_plane m β ∧ line_parallel_plane n α → line_perpendicular m n

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  perpendicular α β ∧ line_perpendicular_plane m α ∧ line_parallel_plane n β → line_parallel m n

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular α β ∧ line_perpendicular_plane m α ∧ line_perpendicular_plane n β → line_perpendicular m n

-- Theorem stating that only propositions 2 and 4 are true
theorem only_prop2_and_prop4_true (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) (h_diff_planes : α ≠ β) : 
  (¬ proposition1 m n α β) ∧ 
  proposition2 m n α β ∧ 
  (¬ proposition3 m n α β) ∧ 
  proposition4 m n α β := by
  sorry

end NUMINAMATH_CALUDE_only_prop2_and_prop4_true_l2201_220135


namespace NUMINAMATH_CALUDE_problem_solution_l2201_220195

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 2*y^3) / 8 = 46.375 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2201_220195


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2201_220158

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2)
  f (-2) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2201_220158


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2201_220131

/-- 
Given that x·(4x + 3) < d if and only when x ∈ (-5/2, 1), prove that d = 10
-/
theorem quadratic_inequality_solution (d : ℝ) : 
  (∀ x : ℝ, x * (4 * x + 3) < d ↔ -5/2 < x ∧ x < 1) → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2201_220131


namespace NUMINAMATH_CALUDE_min_sum_abc_l2201_220179

theorem min_sum_abc (a b c : ℕ+) (h : (a : ℚ) / 77 + (b : ℚ) / 91 + (c : ℚ) / 143 = 1) :
  ∃ (a' b' c' : ℕ+), (a' : ℚ) / 77 + (b' : ℚ) / 91 + (c' : ℚ) / 143 = 1 ∧
    (∀ (x y z : ℕ+), (x : ℚ) / 77 + (y : ℚ) / 91 + (z : ℚ) / 143 = 1 → 
      a' + b' + c' ≤ x + y + z) ∧
    a' + b' + c' = 79 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abc_l2201_220179


namespace NUMINAMATH_CALUDE_log_product_equation_l2201_220116

theorem log_product_equation (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 10) = 4 → x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equation_l2201_220116


namespace NUMINAMATH_CALUDE_triangle_equality_l2201_220115

-- Define the triangle ADC
structure TriangleADC where
  AD : ℝ
  DC : ℝ
  D : ℝ
  h1 : AD = DC
  h2 : D = 100

-- Define the triangle CAB
structure TriangleCAB where
  CA : ℝ
  AB : ℝ
  A : ℝ
  h3 : CA = AB
  h4 : A = 20

-- Define the theorem
theorem triangle_equality (ADC : TriangleADC) (CAB : TriangleCAB) :
  CAB.AB = ADC.DC + CAB.AB - CAB.CA :=
sorry

end NUMINAMATH_CALUDE_triangle_equality_l2201_220115


namespace NUMINAMATH_CALUDE_negation_equivalence_l2201_220100

theorem negation_equivalence :
  (¬ ∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x + 2 = 0) ↔
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → a * x^2 - 3 * x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2201_220100


namespace NUMINAMATH_CALUDE_square_equation_solution_l2201_220176

theorem square_equation_solution :
  ∃ x : ℝ, (3000 + x)^2 = x^2 ∧ x = -1500 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2201_220176


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l2201_220122

theorem no_primes_divisible_by_45 : ∀ p : ℕ, Nat.Prime p → ¬(45 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l2201_220122


namespace NUMINAMATH_CALUDE_divides_a_iff_divides_n_l2201_220169

/-- Sequence defined by a(n) = 2a(n-1) + a(n-2) for n > 1, with a(0) = 0 and a(1) = 1 -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

/-- For all natural numbers k and n, 2^k divides a(n) if and only if 2^k divides n -/
theorem divides_a_iff_divides_n (k n : ℕ) : (2^k : ℤ) ∣ a n ↔ 2^k ∣ n := by sorry

end NUMINAMATH_CALUDE_divides_a_iff_divides_n_l2201_220169


namespace NUMINAMATH_CALUDE_power_inequality_l2201_220130

theorem power_inequality (a : ℝ) (n : ℕ) :
  (a > 1 → a^n > 1) ∧ (a < 1 → a^n < 1) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2201_220130


namespace NUMINAMATH_CALUDE_induction_sum_terms_l2201_220185

theorem induction_sum_terms (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end NUMINAMATH_CALUDE_induction_sum_terms_l2201_220185


namespace NUMINAMATH_CALUDE_sum_of_three_roots_is_zero_l2201_220150

/-- Given two quadratic polynomials with coefficients a and b, 
    where each has two distinct roots and their product has exactly three distinct roots,
    prove that the sum of these three roots is 0. -/
theorem sum_of_three_roots_is_zero (a b : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (∀ x : ℝ, x^2 + b*x + a = 0 ↔ (x = x₃ ∨ x = x₄))) →
  (∃! y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃ ∧
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = y₁ ∨ x = y₂ ∨ x = y₃))) →
  ∃ y₁ y₂ y₃ : ℝ, y₁ + y₂ + y₃ = 0 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_three_roots_is_zero_l2201_220150
