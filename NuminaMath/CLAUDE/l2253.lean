import Mathlib

namespace NUMINAMATH_CALUDE_product_of_separated_evens_l2253_225320

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def circular_arrangement (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 16

theorem product_of_separated_evens (a d : ℕ) : 
  circular_arrangement a → 
  circular_arrangement d → 
  is_even a → 
  is_even d → 
  (∃ b c, circular_arrangement b ∧ circular_arrangement c ∧ 
    ((a < b ∧ b < c ∧ c < d) ∨ (d < a ∧ a < b ∧ b < c) ∨ 
     (c < d ∧ d < a ∧ a < b) ∨ (b < c ∧ c < d ∧ d < a))) →
  a * d = 120 :=
sorry

end NUMINAMATH_CALUDE_product_of_separated_evens_l2253_225320


namespace NUMINAMATH_CALUDE_school_paper_usage_theorem_l2253_225305

/-- The number of sheets of paper used by a school in a week -/
def school_paper_usage (sheets_per_class_per_day : ℕ) (school_days_per_week : ℕ) (num_classes : ℕ) : ℕ :=
  sheets_per_class_per_day * school_days_per_week * num_classes

/-- Theorem stating that under given conditions, the school uses 9000 sheets of paper per week -/
theorem school_paper_usage_theorem :
  school_paper_usage 200 5 9 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_school_paper_usage_theorem_l2253_225305


namespace NUMINAMATH_CALUDE_julia_cakes_remaining_l2253_225332

/-- 
Given:
- Julia bakes one less than 5 cakes per day
- Julia bakes for 6 days
- Clifford eats one cake every other day

Prove that Julia has 21 cakes remaining after 6 days
-/
theorem julia_cakes_remaining (cakes_per_day : ℕ) (days : ℕ) (clifford_eats : ℕ) : 
  cakes_per_day = 5 - 1 → 
  days = 6 → 
  clifford_eats = days / 2 → 
  cakes_per_day * days - clifford_eats = 21 := by
sorry

end NUMINAMATH_CALUDE_julia_cakes_remaining_l2253_225332


namespace NUMINAMATH_CALUDE_expression_evaluation_l2253_225388

theorem expression_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  ((((x+2)^2 * (x^2-2*x+4)^2) / (x^3+8)^2)^2) * ((((x-2)^2 * (x^2+2*x+4)^2) / (x^3-8)^2)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2253_225388


namespace NUMINAMATH_CALUDE_hundredth_figure_squares_l2253_225328

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem hundredth_figure_squares :
  f 100 = 30301 := by sorry

end NUMINAMATH_CALUDE_hundredth_figure_squares_l2253_225328


namespace NUMINAMATH_CALUDE_correct_initial_distribution_l2253_225331

/-- Represents the initial and final coin counts for each person -/
structure CoinCounts where
  initial_gold : ℕ
  initial_silver : ℕ
  final_gold : ℕ
  final_silver : ℕ

/-- Represents the treasure distribution problem -/
def treasure_distribution (k : CoinCounts) (v : CoinCounts) : Prop :=
  -- Křemílek loses half of his gold coins
  k.initial_gold / 2 = k.final_gold - v.initial_gold / 3 ∧
  -- Vochomůrka loses half of his silver coins
  v.initial_silver / 2 = v.final_silver - k.initial_silver / 4 ∧
  -- Vochomůrka gives one-third of his remaining gold coins to Křemílek
  v.initial_gold * 2 / 3 = v.final_gold ∧
  -- Křemílek gives one-quarter of his silver coins to Vochomůrka
  k.initial_silver * 3 / 4 = k.final_silver ∧
  -- After exchanges, each has exactly 12 gold coins and 18 silver coins
  k.final_gold = 12 ∧ k.final_silver = 18 ∧
  v.final_gold = 12 ∧ v.final_silver = 18

/-- Theorem stating the correct initial distribution of coins -/
theorem correct_initial_distribution :
  ∃ (k v : CoinCounts),
    treasure_distribution k v ∧
    k.initial_gold = 12 ∧ k.initial_silver = 24 ∧
    v.initial_gold = 18 ∧ v.initial_silver = 24 :=
  sorry

end NUMINAMATH_CALUDE_correct_initial_distribution_l2253_225331


namespace NUMINAMATH_CALUDE_consecutive_numbers_equation_l2253_225354

theorem consecutive_numbers_equation :
  ∃ (a b c d : ℕ), (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1) ∧ (a * c - b * d = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_equation_l2253_225354


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l2253_225385

/-- The number of bowl colors -/
def numBowls : ℕ := 5

/-- The number of glass colors -/
def numGlasses : ℕ := 4

/-- The total number of possible pairings without restrictions -/
def totalPairings : ℕ := numBowls * numGlasses

/-- The number of restricted pairings (purple bowl with green glass) -/
def restrictedPairings : ℕ := 1

/-- The number of valid pairings -/
def validPairings : ℕ := totalPairings - restrictedPairings

theorem bowl_glass_pairings :
  validPairings = 19 :=
sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l2253_225385


namespace NUMINAMATH_CALUDE_min_disks_for_lilas_problem_l2253_225312

/-- Represents the storage problem with given file sizes and quantities --/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  large_files : ℕ
  large_file_size : ℚ
  medium_files : ℕ
  medium_file_size : ℚ
  small_file_size : ℚ

/-- Calculates the minimum number of disks required for the given storage problem --/
def min_disks_required (problem : StorageProblem) : ℕ :=
  sorry

/-- The specific storage problem instance --/
def lilas_problem : StorageProblem :=
  { total_files := 40
  , disk_capacity := 2
  , large_files := 4
  , large_file_size := 1.2
  , medium_files := 10
  , medium_file_size := 1
  , small_file_size := 0.6 }

/-- Theorem stating that the minimum number of disks required for Lila's problem is 16 --/
theorem min_disks_for_lilas_problem :
  min_disks_required lilas_problem = 16 :=
sorry

end NUMINAMATH_CALUDE_min_disks_for_lilas_problem_l2253_225312


namespace NUMINAMATH_CALUDE_complex_equation_l2253_225345

theorem complex_equation : (2 * Complex.I) * (1 + Complex.I)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l2253_225345


namespace NUMINAMATH_CALUDE_difference_ones_zeros_235_l2253_225383

def base_10_to_base_2 (n : ℕ) : List ℕ :=
  sorry

def count_zeros (l : List ℕ) : ℕ :=
  sorry

def count_ones (l : List ℕ) : ℕ :=
  sorry

theorem difference_ones_zeros_235 :
  let binary_235 := base_10_to_base_2 235
  let w := count_ones binary_235
  let z := count_zeros binary_235
  w - z = 2 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_235_l2253_225383


namespace NUMINAMATH_CALUDE_negative_division_rule_div_negative_64_negative_32_l2253_225379

theorem negative_division_rule (x y : ℤ) (hy : y ≠ 0) : (-x) / (-y) = x / y := by sorry

theorem div_negative_64_negative_32 : (-64) / (-32) = 2 := by sorry

end NUMINAMATH_CALUDE_negative_division_rule_div_negative_64_negative_32_l2253_225379


namespace NUMINAMATH_CALUDE_binary_multiplication_correct_l2253_225335

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0. 
    The least significant bit is at the head of the list. -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNumber) : BinaryNumber :=
  sorry -- Implementation details omitted

theorem binary_multiplication_correct :
  let a : BinaryNumber := [true, true, false, true]  -- 1011₂
  let b : BinaryNumber := [true, false, true]        -- 101₂
  let result : BinaryNumber := [true, true, true, false, true, true]  -- 110111₂
  binary_multiply a b = result ∧ 
  binary_to_decimal (binary_multiply a b) = binary_to_decimal a * binary_to_decimal b :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_correct_l2253_225335


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2253_225377

/-- The eccentricity of an ellipse satisfies the given range -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let B := (0, b)
  let e := Real.sqrt (1 - (b^2 / a^2))
  (∀ p ∈ C, Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2) ≤ 2*b) →
  0 < e ∧ e ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2253_225377


namespace NUMINAMATH_CALUDE_equal_distribution_of_sweets_l2253_225397

/-- Proves that each student receives 4 sweet treats given the conditions -/
theorem equal_distribution_of_sweets
  (cookies : ℕ) (cupcakes : ℕ) (brownies : ℕ) (students : ℕ)
  (h_cookies : cookies = 20)
  (h_cupcakes : cupcakes = 25)
  (h_brownies : brownies = 35)
  (h_students : students = 20)
  : (cookies + cupcakes + brownies) / students = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_sweets_l2253_225397


namespace NUMINAMATH_CALUDE_equal_face_parallelepiped_implies_rhombus_l2253_225380

/-- A parallelepiped with equal parallelogram faces -/
structure EqualFaceParallelepiped where
  /-- The length of the first edge -/
  a : ℝ
  /-- The length of the second edge -/
  b : ℝ
  /-- The length of the third edge -/
  c : ℝ
  /-- All edges have positive length -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  /-- All faces have equal area -/
  equal_faces : a * b = b * c ∧ b * c = a * c

/-- A rhombus is a quadrilateral with all sides equal -/
def is_rhombus (s₁ s₂ s₃ s₄ : ℝ) : Prop :=
  s₁ = s₂ ∧ s₂ = s₃ ∧ s₃ = s₄

/-- If all 6 faces of a parallelepiped are equal parallelograms, then they are rhombuses -/
theorem equal_face_parallelepiped_implies_rhombus (P : EqualFaceParallelepiped) :
  is_rhombus P.a P.a P.a P.a ∧
  is_rhombus P.b P.b P.b P.b ∧
  is_rhombus P.c P.c P.c P.c :=
sorry

end NUMINAMATH_CALUDE_equal_face_parallelepiped_implies_rhombus_l2253_225380


namespace NUMINAMATH_CALUDE_integer_root_iff_a_value_l2253_225395

def polynomial (a x : ℤ) : ℤ := x^4 + 4*x^3 + a*x^2 + 8

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_iff_a_value :
  ∀ a : ℤ, has_integer_root a ↔ a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2 :=
sorry

end NUMINAMATH_CALUDE_integer_root_iff_a_value_l2253_225395


namespace NUMINAMATH_CALUDE_set_equality_implies_a_value_l2253_225309

/-- Given two sets are equal, prove that a must be either 1 or -1 -/
theorem set_equality_implies_a_value (a : ℝ) : 
  ({0, -1, 2*a} : Set ℝ) = ({a-1, -abs a, a+1} : Set ℝ) → 
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_value_l2253_225309


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2253_225317

theorem cost_price_calculation (markup_percentage : ℝ) (discount_percentage : ℝ) (profit : ℝ) : 
  markup_percentage = 0.2 →
  discount_percentage = 0.1 →
  profit = 40 →
  ∃ (cost_price : ℝ), 
    cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price = profit ∧
    cost_price = 500 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2253_225317


namespace NUMINAMATH_CALUDE_ratio_equality_l2253_225304

theorem ratio_equality (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2253_225304


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l2253_225387

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point2D
  p2 : Point2D

/-- Checks if a point lies on the x-axis -/
def isOnXAxis (p : Point2D) : Prop :=
  p.y = 0

/-- Checks if a point lies on a given line -/
def isOnLine (l : Line) (p : Point2D) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

theorem line_intersects_x_axis (l : Line) : 
  l.p1 = ⟨3, -1⟩ → l.p2 = ⟨7, 3⟩ → 
  ∃ p : Point2D, isOnLine l p ∧ isOnXAxis p ∧ p = ⟨4, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l2253_225387


namespace NUMINAMATH_CALUDE_movie_length_after_cut_l2253_225347

/-- Calculates the final length of a movie after cutting a scene -/
theorem movie_length_after_cut (original_length cut_length : ℕ) : 
  original_length = 60 → cut_length = 3 → original_length - cut_length = 57 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cut_l2253_225347


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l2253_225350

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l2253_225350


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_max_l2253_225307

theorem triangle_cosine_sum_max (A B C : ℝ) (h : Real.sin C = 2 * Real.cos A * Real.cos B) :
  ∃ (max : ℝ), max = (Real.sqrt 2 + 1) / 2 ∧ 
    ∀ (A' B' C' : ℝ), Real.sin C' = 2 * Real.cos A' * Real.cos B' →
      Real.cos A' ^ 2 + Real.cos B' ^ 2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_max_l2253_225307


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2253_225356

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that among the given sets, only (6, 8, 10) forms a right triangle --/
theorem right_triangle_sets :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 6 8 10) ∧
  ¬(is_right_triangle 5 8 13) ∧
  ¬(is_right_triangle 12 13 14) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2253_225356


namespace NUMINAMATH_CALUDE_problem_statement_l2253_225381

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^2 + 3*b^2 ≥ 2*b*(a + b)) ∧ 
  ((1/a + 2/b = 1) → (2*a + b ≥ 8)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2253_225381


namespace NUMINAMATH_CALUDE_min_value_f_and_m_plus_2n_l2253_225368

-- Define the function f
def f (x a : ℝ) : ℝ := x + |x - a|

-- State the theorem
theorem min_value_f_and_m_plus_2n :
  ∃ (a : ℝ),
    (∀ x, (f x a - 2)^4 ≥ 0 ∧ f x a ≤ 4) →
    (∃ x₀, ∀ x, f x a ≥ f x₀ a ∧ f x₀ a = 2) ∧
    (∀ m n : ℕ+, 1 / (m : ℝ) + 2 / (n : ℝ) = 2 →
      (m : ℝ) + 2 * (n : ℝ) ≥ 9/2) ∧
    (∃ m₀ n₀ : ℕ+, 1 / (m₀ : ℝ) + 2 / (n₀ : ℝ) = 2 ∧
      (m₀ : ℝ) + 2 * (n₀ : ℝ) = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_f_and_m_plus_2n_l2253_225368


namespace NUMINAMATH_CALUDE_gold_medals_count_l2253_225359

theorem gold_medals_count (total : ℕ) (silver : ℕ) (bronze : ℕ) (h1 : total = 67) (h2 : silver = 32) (h3 : bronze = 16) :
  total - silver - bronze = 19 := by
  sorry

end NUMINAMATH_CALUDE_gold_medals_count_l2253_225359


namespace NUMINAMATH_CALUDE_cd_length_is_nine_l2253_225351

/-- A tetrahedron with specific edge lengths -/
structure Tetrahedron where
  edges : Finset ℝ
  edge_count : edges.card = 6
  edge_values : edges = {9, 15, 22, 35, 40, 44}
  ab_length : 44 ∈ edges

/-- The length of CD in the tetrahedron -/
def cd_length (t : Tetrahedron) : ℝ := 9

/-- Theorem stating that CD length is 9 in the given tetrahedron -/
theorem cd_length_is_nine (t : Tetrahedron) : cd_length t = 9 := by
  sorry

end NUMINAMATH_CALUDE_cd_length_is_nine_l2253_225351


namespace NUMINAMATH_CALUDE_min_values_theorem_l2253_225372

theorem min_values_theorem (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : (r + s - r * s) * (r + s + r * s) = r * s) : 
  (∃ (r' s' : ℝ), r' > 0 ∧ s' > 0 ∧ 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' ∧
    r + s - r * s ≥ -3 + 2 * Real.sqrt 3 ∧
    r + s + r * s ≥ 3 + 2 * Real.sqrt 3) ∧
  (r + s - r * s = -3 + 2 * Real.sqrt 3 ∨ r + s + r * s = 3 + 2 * Real.sqrt 3 → 
    r = Real.sqrt 3 ∧ s = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l2253_225372


namespace NUMINAMATH_CALUDE_triangle_area_double_l2253_225336

theorem triangle_area_double (halved_area : ℝ) :
  halved_area = 7 → 2 * halved_area = 14 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_double_l2253_225336


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2253_225373

/-- An arithmetic sequence with given first four terms -/
def arithmetic_sequence (x y : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => y
  | 2 => 3*x + y
  | 3 => x + 2*y + 2
  | n + 4 => arithmetic_sequence x y 3 + (n + 1) * (arithmetic_sequence x y 1 - arithmetic_sequence x y 0)

/-- The theorem stating that y - x = 2 for the given arithmetic sequence -/
theorem arithmetic_sequence_difference (x y : ℝ) :
  let a := arithmetic_sequence x y
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →
  y - x = 2 := by
  sorry

#check arithmetic_sequence_difference

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2253_225373


namespace NUMINAMATH_CALUDE_sandy_average_price_per_book_l2253_225389

/-- Represents a bookshop visit with the number of books bought and the total price paid -/
structure BookshopVisit where
  books : ℕ
  price : ℚ

/-- Calculates the average price per book given a list of bookshop visits -/
def averagePricePerBook (visits : List BookshopVisit) : ℚ :=
  (visits.map (λ v => v.price)).sum / (visits.map (λ v => v.books)).sum

/-- The theorem statement for Sandy's bookshop visits -/
theorem sandy_average_price_per_book :
  let visits : List BookshopVisit := [
    { books := 65, price := 1080 },
    { books := 55, price := 840 },
    { books := 45, price := 765 },
    { books := 35, price := 630 }
  ]
  averagePricePerBook visits = 16575 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_sandy_average_price_per_book_l2253_225389


namespace NUMINAMATH_CALUDE_inequality_proof_l2253_225346

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  2 * (x^2 + y^2) ≥ (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2253_225346


namespace NUMINAMATH_CALUDE_yellow_square_area_percentage_l2253_225362

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  /-- Side length of the square flag -/
  side : ℝ
  /-- Width of each arm of the cross (equal to side length of yellow square) -/
  crossWidth : ℝ
  /-- Assumption that the cross width is positive and less than the flag side -/
  crossWidthValid : 0 < crossWidth ∧ crossWidth < side

/-- The area of the entire flag -/
def SquareFlag.area (flag : SquareFlag) : ℝ := flag.side ^ 2

/-- The area of the cross (including yellow center) -/
def SquareFlag.crossArea (flag : SquareFlag) : ℝ :=
  4 * flag.side * flag.crossWidth - 3 * flag.crossWidth ^ 2

/-- The area of the yellow square at the center -/
def SquareFlag.yellowArea (flag : SquareFlag) : ℝ := flag.crossWidth ^ 2

/-- Theorem stating that if the cross occupies 49% of the flag's area, 
    then the yellow square occupies 12.25% of the flag's area -/
theorem yellow_square_area_percentage (flag : SquareFlag) 
  (h : flag.crossArea = 0.49 * flag.area) : 
  flag.yellowArea / flag.area = 0.1225 := by
  sorry

end NUMINAMATH_CALUDE_yellow_square_area_percentage_l2253_225362


namespace NUMINAMATH_CALUDE_vector_problem_l2253_225343

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -3)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_problem (c : ℝ × ℝ) :
  perpendicular c (a.1 + b.1, a.2 + b.2) ∧
  parallel b (a.1 - c.1, a.2 - c.2) →
  c = (7/9, 7/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l2253_225343


namespace NUMINAMATH_CALUDE_factorization_equality_l2253_225326

theorem factorization_equality (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2253_225326


namespace NUMINAMATH_CALUDE_cube_edge_length_is_ten_l2253_225374

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- Checks if a point is inside a cube -/
def isInside (p : Point3D) (c : Cube) : Prop :=
  0 < p.x ∧ p.x < c.edgeLength ∧
  0 < p.y ∧ p.y < c.edgeLength ∧
  0 < p.z ∧ p.z < c.edgeLength

/-- Theorem: If there exists an interior point with specific distances from four vertices of a cube,
    then the edge length of the cube is 10 -/
theorem cube_edge_length_is_ten (c : Cube) (p : Point3D) 
    (v1 v2 v3 v4 : Point3D) : 
    isInside p c →
    squaredDistance p v1 = 50 →
    squaredDistance p v2 = 70 →
    squaredDistance p v3 = 90 →
    squaredDistance p v4 = 110 →
    (v1.x = 0 ∨ v1.x = c.edgeLength) ∧ 
    (v1.y = 0 ∨ v1.y = c.edgeLength) ∧ 
    (v1.z = 0 ∨ v1.z = c.edgeLength) →
    (v2.x = 0 ∨ v2.x = c.edgeLength) ∧ 
    (v2.y = 0 ∨ v2.y = c.edgeLength) ∧ 
    (v2.z = 0 ∨ v2.z = c.edgeLength) →
    (v3.x = 0 ∨ v3.x = c.edgeLength) ∧ 
    (v3.y = 0 ∨ v3.y = c.edgeLength) ∧ 
    (v3.z = 0 ∨ v3.z = c.edgeLength) →
    (v4.x = 0 ∨ v4.x = c.edgeLength) ∧ 
    (v4.y = 0 ∨ v4.y = c.edgeLength) ∧ 
    (v4.z = 0 ∨ v4.z = c.edgeLength) →
    (v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4) →
    c.edgeLength = 10 := by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_is_ten_l2253_225374


namespace NUMINAMATH_CALUDE_property_P_lower_bound_l2253_225364

/-- Property P for a function f: ℝ → ℝ -/
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, Real.sqrt (2 * f x) - Real.sqrt (2 * f x - f (2 * x)) ≥ 2

/-- The theorem stating that if f has property P, then f(x) ≥ 12 + 8√2 for all real x -/
theorem property_P_lower_bound (f : ℝ → ℝ) (h : has_property_P f) :
  ∀ x : ℝ, f x ≥ 12 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_property_P_lower_bound_l2253_225364


namespace NUMINAMATH_CALUDE_john_works_five_days_week_l2253_225314

/-- Represents John's work schedule and patient count --/
structure DoctorSchedule where
  patients_hospital1 : ℕ
  patients_hospital2 : ℕ
  total_patients_year : ℕ
  weeks_per_year : ℕ

/-- Calculates the number of days John works per week --/
def days_per_week (s : DoctorSchedule) : ℚ :=
  s.total_patients_year / (s.weeks_per_year * (s.patients_hospital1 + s.patients_hospital2))

/-- Theorem stating that John works 5 days a week --/
theorem john_works_five_days_week (s : DoctorSchedule)
  (h1 : s.patients_hospital1 = 20)
  (h2 : s.patients_hospital2 = 24)
  (h3 : s.total_patients_year = 11000)
  (h4 : s.weeks_per_year = 50) :
  days_per_week s = 5 := by
  sorry

#eval days_per_week { patients_hospital1 := 20, patients_hospital2 := 24, total_patients_year := 11000, weeks_per_year := 50 }

end NUMINAMATH_CALUDE_john_works_five_days_week_l2253_225314


namespace NUMINAMATH_CALUDE_vet_donation_amount_l2253_225363

/-- Calculates the amount donated by the vet to an animal shelter during a pet adoption event. -/
theorem vet_donation_amount (dog_fee cat_fee : ℕ) (dog_adoptions cat_adoptions : ℕ) (donation_fraction : ℚ) : 
  dog_fee = 15 →
  cat_fee = 13 →
  dog_adoptions = 8 →
  cat_adoptions = 3 →
  donation_fraction = 1/3 →
  (dog_fee * dog_adoptions + cat_fee * cat_adoptions) * donation_fraction = 53 := by
  sorry

end NUMINAMATH_CALUDE_vet_donation_amount_l2253_225363


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2253_225371

/-- Given two vectors a and b in R², prove that if they are perpendicular
    and a = (1, -1) and b = (m+1, 2m-4), then m = 5. -/
theorem perpendicular_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![m+1, 2*m-4]
  (∀ i, a i * b i = 0) → m = 5 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2253_225371


namespace NUMINAMATH_CALUDE_remainder_theorem_l2253_225301

def polynomial (x : ℝ) : ℝ := 8*x^4 - 18*x^3 + 27*x^2 - 31*x + 14

def divisor (x : ℝ) : ℝ := 4*x - 8

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2253_225301


namespace NUMINAMATH_CALUDE_star_calculation_l2253_225334

-- Define the ⋆ operation
def star (a b : ℚ) : ℚ := a + 2 / b

-- Theorem statement
theorem star_calculation :
  (star (star 3 4) 5) - (star 3 (star 4 5)) = 49 / 110 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2253_225334


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2253_225353

/-- Given a quadratic function f(x) = ax² + bx + c, where a, b, c are constants,
    and its derivative f'(x), if f(x) ≥ f'(x) for all x ∈ ℝ,
    then the maximum value of b²/(a² + c²) is 2√2 - 2. -/
theorem quadratic_function_inequality (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≥ 2 * a * x + b) → 
  a > 0 → 
  (∃ M, M = 2 * Real.sqrt 2 - 2 ∧ 
    b^2 / (a^2 + c^2) ≤ M ∧ 
    ∀ N, (∀ a' b' c', (∀ x, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') → 
      a' > 0 → b'^2 / (a'^2 + c'^2) ≤ N) → 
    M ≤ N) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2253_225353


namespace NUMINAMATH_CALUDE_prob_non_defective_product_l2253_225398

theorem prob_non_defective_product (prob_grade_b prob_grade_c : ℝ) 
  (h1 : prob_grade_b = 0.03)
  (h2 : prob_grade_c = 0.01)
  (h3 : 0 ≤ prob_grade_b ∧ prob_grade_b ≤ 1)
  (h4 : 0 ≤ prob_grade_c ∧ prob_grade_c ≤ 1)
  (h5 : prob_grade_b + prob_grade_c ≤ 1) :
  1 - (prob_grade_b + prob_grade_c) = 0.96 := by
sorry

end NUMINAMATH_CALUDE_prob_non_defective_product_l2253_225398


namespace NUMINAMATH_CALUDE_H_perimeter_is_36_l2253_225325

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Represents the H-shaped figure -/
structure HShape where
  largeRectLength : ℝ
  largeRectWidth : ℝ
  smallRectLength : ℝ
  smallRectWidth : ℝ

/-- Calculates the perimeter of the H-shaped figure -/
def HPerimeter (h : HShape) : ℝ :=
  2 * rectanglePerimeter h.largeRectLength h.largeRectWidth +
  rectanglePerimeter h.smallRectLength h.smallRectWidth -
  2 * 2 * h.smallRectLength

theorem H_perimeter_is_36 :
  let h : HShape := {
    largeRectLength := 3,
    largeRectWidth := 5,
    smallRectLength := 1,
    smallRectWidth := 3
  }
  HPerimeter h = 36 := by
  sorry

end NUMINAMATH_CALUDE_H_perimeter_is_36_l2253_225325


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2253_225310

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- The unique solution is z = -11
  use (-11 : ℚ)
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2253_225310


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2253_225302

/-- Proves that given specific simple interest conditions, the principal amount is 2000 --/
theorem simple_interest_principal :
  ∀ (rate : ℚ) (interest : ℚ) (time : ℚ) (principal : ℚ),
    rate = 25/2 →
    interest = 500 →
    time = 2 →
    principal * rate * time / 100 = interest →
    principal = 2000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2253_225302


namespace NUMINAMATH_CALUDE_fair_attendance_this_year_l2253_225357

def fair_attendance (this_year next_year last_year : ℕ) : Prop :=
  (next_year = 2 * this_year) ∧
  (last_year = next_year - 200) ∧
  (this_year + next_year + last_year = 2800)

theorem fair_attendance_this_year :
  ∃ (this_year next_year last_year : ℕ),
    fair_attendance this_year next_year last_year ∧ this_year = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_this_year_l2253_225357


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2253_225361

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = Real.cos 1 + 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2253_225361


namespace NUMINAMATH_CALUDE_sine_function_period_l2253_225352

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ k : ℤ, a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * π / 5) + c) + d) →
  b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_period_l2253_225352


namespace NUMINAMATH_CALUDE_average_of_ten_numbers_l2253_225344

theorem average_of_ten_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (seventh_num : ℝ) 
  (h1 : first_six_avg = 68)
  (h2 : last_six_avg = 75)
  (h3 : seventh_num = 258) :
  (6 * first_six_avg + 6 * last_six_avg - seventh_num) / 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_ten_numbers_l2253_225344


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_of_squares_l2253_225355

theorem polynomial_equality_implies_sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_of_squares_l2253_225355


namespace NUMINAMATH_CALUDE_valid_arrangements_l2253_225339

/-- The number of ways to arrange 2 black, 3 white, and 4 red balls in a row such that no black ball is next to a white ball. -/
def arrangeBalls : ℕ := 200

/-- The number of black balls -/
def blackBalls : ℕ := 2

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of red balls -/
def redBalls : ℕ := 4

/-- Theorem stating that the number of valid arrangements is equal to arrangeBalls -/
theorem valid_arrangements :
  (∃ (f : ℕ → ℕ → ℕ → ℕ), f blackBalls whiteBalls redBalls = arrangeBalls) :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2253_225339


namespace NUMINAMATH_CALUDE_space_diagonals_of_Q_l2253_225386

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - (2 * Q.quadrilateral_faces)

/-- Theorem: The number of space diagonals in the given polyhedron Q is 315 -/
theorem space_diagonals_of_Q :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 72 ∧
    Q.faces = 44 ∧
    Q.triangular_faces = 20 ∧
    Q.quadrilateral_faces = 24 ∧
    space_diagonals Q = 315 :=
sorry

end NUMINAMATH_CALUDE_space_diagonals_of_Q_l2253_225386


namespace NUMINAMATH_CALUDE_square_circle_relation_l2253_225367

theorem square_circle_relation (s : ℝ) (h : s > 0) :
  (4 * s = π * (s / Real.sqrt 2)^2) → s = 8 / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_relation_l2253_225367


namespace NUMINAMATH_CALUDE_club_size_l2253_225300

/-- The number of committees in the club -/
def num_committees : ℕ := 5

/-- A member of the club -/
structure Member where
  committees : Finset (Fin num_committees)
  mem_two_committees : committees.card = 2

/-- The club -/
structure Club where
  members : Finset Member
  unique_pair_member : ∀ (c1 c2 : Fin num_committees), c1 ≠ c2 → 
    (members.filter (λ m => c1 ∈ m.committees ∧ c2 ∈ m.committees)).card = 1

theorem club_size (c : Club) : c.members.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_club_size_l2253_225300


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l2253_225348

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_300_trailing_zeros :
  trailing_zeros 300 = 74 := by sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l2253_225348


namespace NUMINAMATH_CALUDE_opposite_pairs_l2253_225330

theorem opposite_pairs : 
  (-((-2)^3) ≠ -|((-2)^3)|) ∧ 
  ((-2)^3 ≠ -(2^3)) ∧ 
  (-2^2 = -(((-2)^2))) ∧ 
  (-(-2) ≠ -|(-2)|) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l2253_225330


namespace NUMINAMATH_CALUDE_point_on_line_l2253_225393

/-- The line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_through_points (x₁ y₁ x₂ y₂ : ℚ) (x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The theorem stating that (-3/7, 8) lies on the line through (3, 0) and (0, 7) -/
theorem point_on_line : line_through_points 3 0 0 7 (-3/7) 8 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2253_225393


namespace NUMINAMATH_CALUDE_simplify_expression_l2253_225369

theorem simplify_expression (x y : ℝ) : (x - y) * (x + y) + (x - y)^2 = 2*x^2 - 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2253_225369


namespace NUMINAMATH_CALUDE_cube_congruence_implies_sum_divisibility_l2253_225316

theorem cube_congruence_implies_sum_divisibility (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p → 
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
sorry

end NUMINAMATH_CALUDE_cube_congruence_implies_sum_divisibility_l2253_225316


namespace NUMINAMATH_CALUDE_polynomial_roots_theorem_l2253_225375

-- Define the polynomial
def P (a b c : ℂ) (x : ℂ) : ℂ := x^4 - a*x^3 - b*x + c

-- Define the set of solutions
def SolutionSet : Set (ℂ × ℂ × ℂ) :=
  {(a, 0, 0) | a : ℂ} ∪
  {((-1 + Complex.I * Real.sqrt 3) / 2, 1, (-1 + Complex.I * Real.sqrt 3) / 2),
   ((-1 - Complex.I * Real.sqrt 3) / 2, 1, (-1 - Complex.I * Real.sqrt 3) / 2),
   ((1 - Complex.I * Real.sqrt 3) / 2, -1, (1 + Complex.I * Real.sqrt 3) / 2),
   ((1 + Complex.I * Real.sqrt 3) / 2, -1, (1 - Complex.I * Real.sqrt 3) / 2)}

-- The main theorem
theorem polynomial_roots_theorem (a b c : ℂ) :
  (∃ d : ℂ, {a, b, c, d} ⊆ {x : ℂ | P a b c x = 0} ∧ (a, b, c) ∈ SolutionSet) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_theorem_l2253_225375


namespace NUMINAMATH_CALUDE_max_distance_to_line_l2253_225384

/-- The maximum distance from the point (1, 1) to the line x*cos(θ) + y*sin(θ) = 2 is 2 + √2 -/
theorem max_distance_to_line : 
  let P : ℝ × ℝ := (1, 1)
  let line (θ : ℝ) (x y : ℝ) := x * Real.cos θ + y * Real.sin θ = 2
  ∃ (d : ℝ), d = 2 + Real.sqrt 2 ∧ 
    ∀ (θ : ℝ), d ≥ Real.sqrt ((P.1 * Real.cos θ + P.2 * Real.sin θ - 2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l2253_225384


namespace NUMINAMATH_CALUDE_fruits_in_buckets_l2253_225382

/-- The number of fruits in three buckets -/
def total_fruits (a b c : ℕ) : ℕ := a + b + c

/-- Theorem: The total number of fruits in three buckets is 37 -/
theorem fruits_in_buckets :
  ∀ (a b c : ℕ),
  c = 9 →
  b = c + 3 →
  a = b + 4 →
  total_fruits a b c = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_fruits_in_buckets_l2253_225382


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l2253_225327

/-- The area of Andrea's living room floor, given that 20% is covered by a 4ft by 9ft carpet -/
theorem andreas_living_room_area : 
  ∀ (carpet_length carpet_width carpet_area total_area : ℝ),
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_area = carpet_length * carpet_width →
  carpet_area / total_area = 1/5 →
  total_area = 180 := by
sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l2253_225327


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_real_l2253_225329

theorem inequality_solution_implies_a_real : 
  (∃ x : ℝ, x^2 - a*x + a ≤ 1) → a ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_real_l2253_225329


namespace NUMINAMATH_CALUDE_fraction_equality_l2253_225358

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 2 / 3) :
  t / q = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2253_225358


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l2253_225376

theorem gcd_lcm_sum_for_special_case (a b : ℕ) (h : a = 1999 * b) :
  Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l2253_225376


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l2253_225391

theorem different_color_chips_probability :
  let total_chips : ℕ := 9
  let blue_chips : ℕ := 6
  let yellow_chips : ℕ := 3
  let prob_blue_then_yellow : ℚ := (blue_chips / total_chips) * (yellow_chips / (total_chips - 1))
  let prob_yellow_then_blue : ℚ := (yellow_chips / total_chips) * (blue_chips / (total_chips - 1))
  prob_blue_then_yellow + prob_yellow_then_blue = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l2253_225391


namespace NUMINAMATH_CALUDE_square_perimeter_l2253_225321

theorem square_perimeter : ∀ (x₁ x₂ : ℝ),
  x₁^2 + 4*x₁ + 3 = 7 →
  x₂^2 + 4*x₂ + 3 = 7 →
  x₁ ≠ x₂ →
  4 * |x₂ - x₁| = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l2253_225321


namespace NUMINAMATH_CALUDE_projection_onto_plane_l2253_225370

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ :=
  sorry

theorem projection_onto_plane (P : Plane) :
  project (2, 4, 7) P = (1, 3, 3) →
  project (6, -3, 8) P = (41/9, -40/9, 20/9) := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_plane_l2253_225370


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l2253_225311

theorem multiply_mixed_number : 8 * (9 + 2/5) = 75 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l2253_225311


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2253_225323

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ -- semi-perimeter
  R : ℝ -- circumradius
  r : ℝ -- inradius
  S : ℝ -- area

-- State the theorem
theorem triangle_inequalities (t : Triangle) : 
  (Real.cos t.A + Real.cos t.B + Real.cos t.C ≤ 3/2) ∧
  (Real.sin (t.A/2) * Real.sin (t.B/2) * Real.sin (t.C/2) ≤ 1/8) ∧
  (t.a * t.b * t.c ≥ 8 * (t.p - t.a) * (t.p - t.b) * (t.p - t.c)) ∧
  (t.R ≥ 2 * t.r) ∧
  (t.S ≤ (1/2) * t.R * t.p) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2253_225323


namespace NUMINAMATH_CALUDE_vasya_has_winning_strategy_l2253_225360

/-- Represents a game state -/
structure GameState where
  board : List Nat
  currentPlayer : Bool  -- true for Petya, false for Vasya

/-- Checks if a list of numbers contains an arithmetic progression -/
def hasArithmeticProgression (numbers : List Nat) : Bool :=
  sorry

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Nat) : Bool :=
  move ≤ 2018 ∧ move ∉ state.board

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Nat) : GameState :=
  { board := move :: state.board
  , currentPlayer := ¬state.currentPlayer }

/-- Represents a strategy for a player -/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (strategy : Strategy) (player : Bool) : Prop :=
  ∀ (initialState : GameState),
    initialState.currentPlayer = player →
    ∃ (finalState : GameState),
      (finalState.board.length ≥ 3 ∧
       hasArithmeticProgression finalState.board) ∧
      finalState.currentPlayer = player

/-- The main theorem stating that Vasya (the second player) has a winning strategy -/
theorem vasya_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy strategy false :=
sorry

end NUMINAMATH_CALUDE_vasya_has_winning_strategy_l2253_225360


namespace NUMINAMATH_CALUDE_product_remainder_l2253_225349

theorem product_remainder (a b c : ℕ) (ha : a = 2457) (hb : b = 6273) (hc : c = 91409) :
  (a * b * c) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2253_225349


namespace NUMINAMATH_CALUDE_attraction_visit_orders_l2253_225308

theorem attraction_visit_orders (n : ℕ) (h : n = 5) : 
  (n! / 2 : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_attraction_visit_orders_l2253_225308


namespace NUMINAMATH_CALUDE_intersection_M_N_l2253_225365

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2253_225365


namespace NUMINAMATH_CALUDE_principal_calculation_l2253_225340

/-- Prove that given the conditions, the principal amount is 1500 --/
theorem principal_calculation (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 15 → P = 1500 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2253_225340


namespace NUMINAMATH_CALUDE_employed_females_percentage_proof_l2253_225322

/-- The percentage of employed people in town X -/
def employed_percentage : ℝ := 64

/-- The percentage of employed males in town X -/
def employed_males_percentage : ℝ := 48

/-- The percentage of employed females out of the total employed population in town X -/
def employed_females_percentage : ℝ := 25

theorem employed_females_percentage_proof :
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = employed_females_percentage :=
by sorry

end NUMINAMATH_CALUDE_employed_females_percentage_proof_l2253_225322


namespace NUMINAMATH_CALUDE_final_week_hours_l2253_225392

def hours_worked : List ℕ := [14, 10, 13, 9, 12, 11]
def total_weeks : ℕ := 7
def required_average : ℕ := 12

theorem final_week_hours :
  ∃ (x : ℕ), (List.sum hours_worked + x) / total_weeks = required_average :=
by sorry

end NUMINAMATH_CALUDE_final_week_hours_l2253_225392


namespace NUMINAMATH_CALUDE_pencils_per_row_l2253_225306

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 32)
  (h2 : num_rows = 4)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2253_225306


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2253_225390

/-- Calculates the selling price of an item given its cost price and gain percentage. -/
def selling_price (cost : ℕ) (gain_percent : ℕ) : ℕ :=
  cost + (cost * gain_percent) / 100

/-- Theorem: If a cycle is bought for Rs. 1000 and sold with a 100% gain, the selling price is Rs. 2000. -/
theorem cycle_selling_price :
  selling_price 1000 100 = 2000 := by
  sorry

#eval selling_price 1000 100

end NUMINAMATH_CALUDE_cycle_selling_price_l2253_225390


namespace NUMINAMATH_CALUDE_smallest_number_l2253_225313

theorem smallest_number (a b c d : ℤ) 
  (ha : a = 2023) 
  (hb : b = 2022) 
  (hc : c = -2023) 
  (hd : d = -2022) : 
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l2253_225313


namespace NUMINAMATH_CALUDE_units_digit_17_pow_28_l2253_225366

theorem units_digit_17_pow_28 : (17^28) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_28_l2253_225366


namespace NUMINAMATH_CALUDE_boys_height_ratio_l2253_225341

theorem boys_height_ratio (total_students : ℕ) (boys_under_6ft : ℕ) 
  (h1 : total_students = 100)
  (h2 : boys_under_6ft = 10) :
  (boys_under_6ft : ℚ) / (total_students / 2 : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_height_ratio_l2253_225341


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2253_225394

/-- For a > 0 and a ≠ 1, the function f(x) = a^(x-2) - 3 passes through the point (2, -2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, a^(x - 2) - 3 = x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2253_225394


namespace NUMINAMATH_CALUDE_student_rank_theorem_l2253_225378

/-- Given a group of students, calculate the rank from left based on total students and rank from right -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem stating that in a group of 10 students, the 6th from right is 5th from left -/
theorem student_rank_theorem :
  let totalStudents : ℕ := 10
  let rankFromRight : ℕ := 6
  rankFromLeft totalStudents rankFromRight = 5 := by
  sorry


end NUMINAMATH_CALUDE_student_rank_theorem_l2253_225378


namespace NUMINAMATH_CALUDE_ninth_group_number_l2253_225333

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_employees : ℕ
  sample_size : ℕ
  group_size : ℕ
  fifth_group_number : ℕ

/-- The number drawn from the nth group in a systematic sampling -/
def number_drawn (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.group_size * (n - 1) + (s.fifth_group_number - s.group_size * 4)

/-- Theorem stating the relationship between the 5th and 9th group numbers -/
theorem ninth_group_number (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.sample_size = 40)
  (h3 : s.group_size = 5)
  (h4 : s.fifth_group_number = 22) :
  number_drawn s 9 = 42 := by
  sorry


end NUMINAMATH_CALUDE_ninth_group_number_l2253_225333


namespace NUMINAMATH_CALUDE_probability_of_different_digits_l2253_225337

/-- The number of integers from 100 to 999 inclusive -/
def total_integers : ℕ := 999 - 100 + 1

/-- The number of integers from 100 to 999 with all different digits -/
def integers_with_different_digits : ℕ := 9 * 9 * 8

/-- The probability of selecting an integer with all different digits from 100 to 999 -/
def probability : ℚ := integers_with_different_digits / total_integers

theorem probability_of_different_digits : probability = 18 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_different_digits_l2253_225337


namespace NUMINAMATH_CALUDE_largest_two_digit_with_digit_product_12_l2253_225303

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem largest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_with_digit_product_12_l2253_225303


namespace NUMINAMATH_CALUDE_function_identity_l2253_225399

def StrictlyIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ m n, m < n → f m < f n

theorem function_identity (f : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing f)
  (h_two : f 2 = 2)
  (h_coprime : ∀ m n, Nat.Coprime m n → f (m * n) = f m * f n) :
  ∀ n, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_identity_l2253_225399


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2253_225315

theorem polynomial_factorization (k : ℤ) :
  let N : ℕ := (4 * k^4 - 8 * k^2 + 2).toNat
  let p (x : ℝ) := x^8 + N * x^4 + 1
  let f (x : ℝ) := x^4 - 2*k*x^3 + 2*k^2*x^2 - 2*k*x + 1
  let g (x : ℝ) := x^4 + 2*k*x^3 + 2*k^2*x^2 + 2*k*x + 1
  ∀ x, p x = f x * g x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2253_225315


namespace NUMINAMATH_CALUDE_jasper_sold_31_drinks_l2253_225396

/-- Represents the number of items sold by Jasper -/
structure JasperSales where
  chips : ℕ
  hot_dogs : ℕ
  drinks : ℕ

/-- Calculates the number of drinks sold by Jasper -/
def calculate_drinks (sales : JasperSales) : ℕ :=
  sales.chips - 8 + 12

/-- Theorem stating that Jasper sold 31 drinks -/
theorem jasper_sold_31_drinks (sales : JasperSales) 
  (h1 : sales.chips = 27)
  (h2 : sales.hot_dogs = sales.chips - 8)
  (h3 : sales.drinks = sales.hot_dogs + 12) :
  sales.drinks = 31 := by
  sorry

end NUMINAMATH_CALUDE_jasper_sold_31_drinks_l2253_225396


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2253_225342

theorem complex_fraction_evaluation : 
  let f1 : ℚ := 7 / 18
  let f2 : ℚ := 9 / 2  -- 4 1/2 as improper fraction
  let f3 : ℚ := 1 / 6
  let f4 : ℚ := 40 / 3  -- 13 1/3 as improper fraction
  let f5 : ℚ := 15 / 4  -- 3 3/4 as improper fraction
  let f6 : ℚ := 5 / 16
  let f7 : ℚ := 23 / 8  -- 2 7/8 as improper fraction
  (((f1 * f2 + f3) / (f4 - f5 / f6)) * f7) = 529 / 128 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2253_225342


namespace NUMINAMATH_CALUDE_unclaimed_candy_fraction_verify_actual_taken_l2253_225318

/-- Represents the fraction of candy taken by each person -/
structure CandyFraction where
  al : Rat
  bert : Rat
  carl : Rat

/-- The intended ratio for candy distribution -/
def intended_ratio : CandyFraction :=
  { al := 4/9, bert := 1/3, carl := 2/9 }

/-- The actual amount of candy taken by each person -/
def actual_taken : CandyFraction :=
  { al := 4/9, bert := 5/27, carl := 20/243 }

/-- The theorem stating the fraction of candy that goes unclaimed -/
theorem unclaimed_candy_fraction :
  1 - (actual_taken.al + actual_taken.bert + actual_taken.carl) = 230/243 := by
  sorry

/-- Verify that the actual taken amounts are correct based on the problem description -/
theorem verify_actual_taken :
  actual_taken.al = intended_ratio.al ∧
  actual_taken.bert = intended_ratio.bert * (1 - actual_taken.al) ∧
  actual_taken.carl = intended_ratio.carl * (1 - actual_taken.al - actual_taken.bert) := by
  sorry

end NUMINAMATH_CALUDE_unclaimed_candy_fraction_verify_actual_taken_l2253_225318


namespace NUMINAMATH_CALUDE_carbon_dioxide_formation_l2253_225324

-- Define the chemical reaction
def chemical_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = 1 ∧ NaHCO3 = 1 ∧ NaNO3 = 1 ∧ CO2 = 1 ∧ H2O = 1

-- Theorem statement
theorem carbon_dioxide_formation :
  ∀ (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ),
    chemical_reaction HNO3 NaHCO3 NaNO3 CO2 H2O →
    CO2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_carbon_dioxide_formation_l2253_225324


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_four_l2253_225338

theorem arithmetic_sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_four_l2253_225338


namespace NUMINAMATH_CALUDE_three_solutions_sum_l2253_225319

theorem three_solutions_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_solutions : ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ = b ∧
    (∀ x : ℝ, Real.sqrt (|x|) + Real.sqrt (|x + a|) = b ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  a + b = 144 := by
sorry

end NUMINAMATH_CALUDE_three_solutions_sum_l2253_225319
