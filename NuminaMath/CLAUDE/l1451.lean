import Mathlib

namespace NUMINAMATH_CALUDE_card_combination_problem_l1451_145179

theorem card_combination_problem : Nat.choose 60 8 = 7580800000 := by
  sorry

end NUMINAMATH_CALUDE_card_combination_problem_l1451_145179


namespace NUMINAMATH_CALUDE_ac_length_l1451_145110

/-- Two triangles ABC and ADE are similar with given side lengths. -/
structure SimilarTriangles where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  AD : ℝ
  DE : ℝ
  EA : ℝ
  similar : True  -- Represents that the triangles are similar
  h_AB : AB = 18
  h_BC : BC = 24
  h_CA : CA = 20
  h_AD : AD = 9
  h_DE : DE = 12
  h_EA : EA = 15

/-- The length of AC in the similar triangles is 20. -/
theorem ac_length (t : SimilarTriangles) : t.CA = 20 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l1451_145110


namespace NUMINAMATH_CALUDE_vector_angle_theorem_l1451_145143

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def angle (a b : E) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem vector_angle_theorem (a b : E) (k : ℝ) (hk : k ≠ 0) 
  (h : norm (a + k • b) = norm (a - b)) : 
  (k = -1 → angle a b = Real.pi / 2) ∧ 
  (k ≠ -1 → angle a b = Real.arccos (-1 / (k + 1))) :=
sorry

end

end NUMINAMATH_CALUDE_vector_angle_theorem_l1451_145143


namespace NUMINAMATH_CALUDE_no_solution_PP_QQ_l1451_145129

-- Define the type of polynomials over ℝ
variable (P Q : ℝ → ℝ)

-- Hypothesis: P and Q are polynomials
axiom P_polynomial : Polynomial ℝ
axiom Q_polynomial : Polynomial ℝ

-- Hypothesis: ∀x ∈ ℝ, P(Q(x)) = Q(P(x))
axiom functional_equality : ∀ x : ℝ, P (Q x) = Q (P x)

-- Hypothesis: P(x) = Q(x) has no solutions
axiom no_solution_PQ : ∀ x : ℝ, P x ≠ Q x

-- Theorem: P(P(x)) = Q(Q(x)) has no solutions
theorem no_solution_PP_QQ : ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_PP_QQ_l1451_145129


namespace NUMINAMATH_CALUDE_number_list_difference_l1451_145180

theorem number_list_difference (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : (x₁ + x₂ + x₃) / 3 = -3)
  (h2 : (x₁ + x₂ + x₃ + x₄) / 4 = 4)
  (h3 : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = -5) :
  x₄ - x₅ = 66 := by
sorry

end NUMINAMATH_CALUDE_number_list_difference_l1451_145180


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l1451_145160

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l1451_145160


namespace NUMINAMATH_CALUDE_reciprocal_of_complex_l1451_145176

theorem reciprocal_of_complex (z : ℂ) (h : z = 5 + I) : 
  z⁻¹ = 5 / 26 - (1 / 26) * I :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_complex_l1451_145176


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1451_145141

/-- Given a geometric sequence {a_n} where a_1 = 1/9 and a_4 = 3, 
    the product of the first five terms is equal to 1 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 4 / a 1)^(1/3)) → -- Geometric sequence condition
  a 1 = 1/9 →                                  -- First term condition
  a 4 = 3 →                                    -- Fourth term condition
  a 1 * a 2 * a 3 * a 4 * a 5 = 1 :=            -- Product of first five terms
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1451_145141


namespace NUMINAMATH_CALUDE_proportional_difference_theorem_l1451_145167

theorem proportional_difference_theorem (x y z k₁ k₂ : ℝ) 
  (h1 : y - z = k₁ * x)
  (h2 : z - x = k₂ * y)
  (h3 : k₁ ≠ k₂)
  (h4 : z = 3 * (x - y))
  (h5 : x ≠ 0)
  (h6 : y ≠ 0) :
  (k₁ + 3) * (k₂ + 3) = 8 := by
sorry

end NUMINAMATH_CALUDE_proportional_difference_theorem_l1451_145167


namespace NUMINAMATH_CALUDE_lcm_problem_l1451_145174

theorem lcm_problem (n : ℕ) (h1 : n > 0) (h2 : Nat.lcm 24 n = 72) (h3 : Nat.lcm n 27 = 108) :
  n = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1451_145174


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1451_145188

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 5

-- State the theorem
theorem quadratic_minimum :
  ∃ (x_min : ℝ), (∀ (x : ℝ), f x ≥ f x_min) ∧ (x_min = 2) ∧ (f x_min = 13) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1451_145188


namespace NUMINAMATH_CALUDE_proposition_d_is_true_l1451_145135

theorem proposition_d_is_true (a b : ℝ) : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_d_is_true_l1451_145135


namespace NUMINAMATH_CALUDE_series_sum_l1451_145157

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The series term -/
noncomputable def series_term (k : ℕ) : ℤ := 
  floor ((1 + Real.sqrt (2000000 / 4^k)) / 2)

/-- The theorem statement -/
theorem series_sum : ∑' k, series_term k = 1414 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1451_145157


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l1451_145100

theorem smallest_proportional_part (total : ℚ) (p1 p2 p3 : ℚ) (h1 : total = 130) 
  (h2 : p1 = 1) (h3 : p2 = 1/4) (h4 : p3 = 1/5) 
  (h5 : ∃ x : ℚ, x * p1 + x * p2 + x * p3 = total) : 
  min (x * p1) (min (x * p2) (x * p3)) = 2600/145 :=
sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l1451_145100


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1451_145112

/-- The side length of a square inscribed in a right triangle with sides 6, 8, and 10 -/
def inscribedSquareSideLength : ℚ := 60 / 31

/-- Right triangle ABC with square XYZW inscribed -/
structure InscribedSquare where
  -- Triangle side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Square side length
  s : ℝ
  -- Conditions
  right_triangle : AB^2 + BC^2 = AC^2
  AB_eq : AB = 6
  BC_eq : BC = 8
  AC_eq : AC = 10
  inscribed : s > 0 -- The square is inscribed (side length is positive)
  on_AC : s ≤ AC -- X and Y are on AC
  on_AB : s ≤ AB -- W is on AB
  on_BC : s ≤ BC -- Z is on BC

/-- The side length of the inscribed square is equal to 60/31 -/
theorem inscribed_square_side_length (square : InscribedSquare) :
  square.s = inscribedSquareSideLength := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1451_145112


namespace NUMINAMATH_CALUDE_smallest_value_of_expression_l1451_145185

theorem smallest_value_of_expression (a b c : ℤ) (ω : ℂ) 
  (h1 : ω^4 = 1) 
  (h2 : ω ≠ 1) 
  (h3 : a = 2*b - c) : 
  ∃ (a₀ b₀ c₀ : ℤ), ∀ (a' b' c' : ℤ), 
    |Complex.abs (a₀ + b₀*ω + c₀*ω^3)| ≤ |Complex.abs (a' + b'*ω + c'*ω^3)| ∧ 
    |Complex.abs (a₀ + b₀*ω + c₀*ω^3)| = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_of_expression_l1451_145185


namespace NUMINAMATH_CALUDE_solution_of_system_l1451_145184

theorem solution_of_system (x y : ℚ) :
  (x + 5) / (x - 4) = (x - 7) / (x + 3) ∧ x + y = 20 →
  x = 13 / 19 ∧ y = 367 / 19 := by
sorry


end NUMINAMATH_CALUDE_solution_of_system_l1451_145184


namespace NUMINAMATH_CALUDE_jones_wardrobe_count_l1451_145173

/-- Represents the clothing items of Mr. Jones -/
structure Wardrobe where
  pants : ℕ
  shirts : ℕ
  ties : ℕ
  socks : ℕ

/-- Calculates the total number of clothing items -/
def total_clothes (w : Wardrobe) : ℕ :=
  w.pants + w.shirts + w.ties + w.socks

/-- Theorem stating the total number of clothes Mr. Jones owns -/
theorem jones_wardrobe_count :
  ∃ (w : Wardrobe),
    w.pants = 40 ∧
    w.shirts = 6 * w.pants ∧
    w.ties = (3 * w.shirts) / 2 ∧
    w.socks = w.ties ∧
    total_clothes w = 1000 := by
  sorry

#check jones_wardrobe_count

end NUMINAMATH_CALUDE_jones_wardrobe_count_l1451_145173


namespace NUMINAMATH_CALUDE_zig_book_count_l1451_145139

/-- Given that Zig wrote four times as many books as Flo and they wrote 75 books in total,
    prove that Zig wrote 60 books. -/
theorem zig_book_count (flo_books : ℕ) (zig_books : ℕ) : 
  zig_books = 4 * flo_books →  -- Zig wrote four times as many books as Flo
  zig_books + flo_books = 75 →  -- They wrote 75 books altogether
  zig_books = 60 :=  -- Prove that Zig wrote 60 books
by sorry

end NUMINAMATH_CALUDE_zig_book_count_l1451_145139


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1451_145142

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1451_145142


namespace NUMINAMATH_CALUDE_contractor_work_completion_l1451_145126

/-- Represents the problem of determining when 1/4 of the work was completed. -/
theorem contractor_work_completion (total_days : ℕ) (initial_workers : ℕ) (remaining_days : ℕ) (fired_workers : ℕ) : 
  total_days = 100 →
  initial_workers = 10 →
  remaining_days = 75 →
  fired_workers = 2 →
  ∃ (x : ℕ), 
    (x * initial_workers = remaining_days * (initial_workers - fired_workers)) ∧
    x = 60 :=
by sorry

end NUMINAMATH_CALUDE_contractor_work_completion_l1451_145126


namespace NUMINAMATH_CALUDE_handbag_price_adjustment_l1451_145183

/-- Calculates the final price of a handbag after a price increase followed by a discount -/
theorem handbag_price_adjustment (initial_price : ℝ) : 
  initial_price = 50 →
  (initial_price * 1.2) * 0.8 = 48 := by sorry

end NUMINAMATH_CALUDE_handbag_price_adjustment_l1451_145183


namespace NUMINAMATH_CALUDE_C_power_50_l1451_145186

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1451_145186


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l1451_145158

theorem smallest_multiple_of_45_and_75_not_20 : 
  ∃ (n : ℕ), n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ (m : ℕ), m > 0 → 45 ∣ m → 75 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l1451_145158


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l1451_145161

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) : 
  x + y ≤ 31 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l1451_145161


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l1451_145125

theorem matching_shoes_probability (n : ℕ) (h : n = 100) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 199 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l1451_145125


namespace NUMINAMATH_CALUDE_frequency_distribution_forms_l1451_145162

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable

/-- Represents a frequency distribution histogram -/
structure FrequencyDistributionHistogram

/-- Represents a set of data -/
structure DataSet

/-- A frequency distribution form for a set of data -/
class FrequencyDistributionForm (α : Type) where
  represents : α → DataSet → Prop

/-- Accuracy property for frequency distribution forms -/
class Accurate (α : Type) where
  is_accurate : α → Prop

/-- Intuitiveness property for frequency distribution forms -/
class Intuitive (α : Type) where
  is_intuitive : α → Prop

instance : FrequencyDistributionForm FrequencyDistributionTable where
  represents := sorry

instance : FrequencyDistributionForm FrequencyDistributionHistogram where
  represents := sorry

instance : Accurate FrequencyDistributionTable where
  is_accurate := sorry

instance : Intuitive FrequencyDistributionHistogram where
  is_intuitive := sorry

/-- Theorem stating that frequency distribution tables and histograms are two forms of frequency distribution for a set of data, with tables being accurate and histograms being intuitive -/
theorem frequency_distribution_forms :
  (∃ (t : FrequencyDistributionTable) (h : FrequencyDistributionHistogram) (d : DataSet),
    FrequencyDistributionForm.represents t d ∧
    FrequencyDistributionForm.represents h d) ∧
  (∀ (t : FrequencyDistributionTable), Accurate.is_accurate t) ∧
  (∀ (h : FrequencyDistributionHistogram), Intuitive.is_intuitive h) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_forms_l1451_145162


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1451_145136

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 - 4 * x - 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x - 3)^2 = 3 * x * (x - 3)
  let sol1 : Set ℝ := {(2 + Real.sqrt 6) / 2, (2 - Real.sqrt 6) / 2}
  let sol2 : Set ℝ := {3, -3/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) := by
  sorry

#check quadratic_equations_solutions

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1451_145136


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_equals_5_sqrt_3_l1451_145127

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (side_length : ℝ)

-- Define a point inside the triangle
structure PointInTriangle :=
  (triangle : EquilateralTriangle)
  (inside : Bool)

-- Define the sum of perpendiculars
def sum_of_perpendiculars (p : PointInTriangle) : ℝ := sorry

-- Theorem statement
theorem sum_of_perpendiculars_equals_5_sqrt_3 
  (p : PointInTriangle) 
  (h : p.triangle.side_length = 10) :
  sum_of_perpendiculars p = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_equals_5_sqrt_3_l1451_145127


namespace NUMINAMATH_CALUDE_square_sum_is_one_l1451_145138

/-- Given two real numbers A and B, we define two functions f and g. -/
def f (A B x : ℝ) : ℝ := A * x^2 + B

def g (A B x : ℝ) : ℝ := B * x^2 + A

/-- The main theorem stating that under certain conditions, A^2 + B^2 = 1 -/
theorem square_sum_is_one (A B : ℝ) (h1 : A ≠ B) 
    (h2 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A^2 + B^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_square_sum_is_one_l1451_145138


namespace NUMINAMATH_CALUDE_smallest_M_inequality_l1451_145148

theorem smallest_M_inequality (a b c : ℝ) : ∃ (M : ℝ), 
  (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ M*(x^2 + y^2 + z^2)^2) ∧ 
  (M = (9 * Real.sqrt 2) / 64) ∧
  (∀ (N : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) → M ≤ N) :=
by sorry

end NUMINAMATH_CALUDE_smallest_M_inequality_l1451_145148


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l1451_145134

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 7

def has_both_digits (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 5 = 0 ∧
    m % 7 = 0 ∧
    is_valid_number m ∧
    has_both_digits m ∧
    (∀ k : ℕ, k > 0 ∧ k % 5 = 0 ∧ k % 7 = 0 ∧ is_valid_number k ∧ has_both_digits k → m ≤ k) ∧
    last_four_digits m = 2772 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l1451_145134


namespace NUMINAMATH_CALUDE_shared_triangle_angle_measure_l1451_145147

-- Define the angle measures
def angle1 : Real := 58
def angle2 : Real := 35
def angle3 : Real := 42

-- Define the theorem
theorem shared_triangle_angle_measure :
  ∃ (angle4 angle5 angle6 : Real),
    -- The sum of angles in the first triangle is 180°
    angle1 + angle2 + angle5 = 180 ∧
    -- The sum of angles in the second triangle is 180°
    angle3 + angle5 + angle6 = 180 ∧
    -- The sum of angles in the third triangle (with the unknown angle) is 180°
    angle4 + angle5 + angle6 = 180 ∧
    -- The measure of the unknown angle (angle4) is 135°
    angle4 = 135 := by
  sorry

end NUMINAMATH_CALUDE_shared_triangle_angle_measure_l1451_145147


namespace NUMINAMATH_CALUDE_picture_hanging_l1451_145192

theorem picture_hanging (board_width : ℕ) (picture_width : ℕ) (num_pictures : ℕ) :
  board_width = 320 ∧ picture_width = 30 ∧ num_pictures = 6 →
  (board_width - num_pictures * picture_width) / (num_pictures + 1) = 20 :=
by sorry

end NUMINAMATH_CALUDE_picture_hanging_l1451_145192


namespace NUMINAMATH_CALUDE_handshake_count_l1451_145128

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (2 * n) * ((2 * n) - 2) / 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l1451_145128


namespace NUMINAMATH_CALUDE_max_value_of_function_l1451_145123

theorem max_value_of_function (x : ℝ) : 
  (3 * Real.sin x + 2 * Real.sqrt (2 + 2 * Real.cos (2 * x))) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1451_145123


namespace NUMINAMATH_CALUDE_train_crossing_time_l1451_145154

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 750 ∧ 
  train_speed_kmh = 180 →
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1451_145154


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l1451_145196

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x-2) + (x-1) + x + (x+1) + (x+2)}

theorem gcd_of_B_is_five :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l1451_145196


namespace NUMINAMATH_CALUDE_nested_radical_equality_l1451_145169

theorem nested_radical_equality : ∃! (a b : ℕ), 
  0 < a ∧ a < b ∧ 
  (Real.sqrt (1 + Real.sqrt (24 + 15 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) ∧
  a = 2 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_nested_radical_equality_l1451_145169


namespace NUMINAMATH_CALUDE_division_equation_l1451_145199

theorem division_equation : (786 * 74) / 30 = 1938.8 := by
  sorry

end NUMINAMATH_CALUDE_division_equation_l1451_145199


namespace NUMINAMATH_CALUDE_swim_team_capacity_difference_l1451_145109

/-- Represents the number of each type of vehicle --/
structure Vehicles where
  cars : Nat
  vans : Nat
  minibuses : Nat

/-- Represents the maximum capacity of each type of vehicle --/
structure VehicleCapacities where
  car : Nat
  van : Nat
  minibus : Nat

/-- Represents the actual number of people in each vehicle --/
structure ActualOccupancy where
  car1 : Nat
  car2 : Nat
  van1 : Nat
  van2 : Nat
  van3 : Nat
  minibus : Nat

def vehicles : Vehicles := {
  cars := 2,
  vans := 3,
  minibuses := 1
}

def capacities : VehicleCapacities := {
  car := 6,
  van := 8,
  minibus := 15
}

def occupancy : ActualOccupancy := {
  car1 := 5,
  car2 := 4,
  van1 := 3,
  van2 := 3,
  van3 := 5,
  minibus := 10
}

def totalMaxCapacity (v : Vehicles) (c : VehicleCapacities) : Nat :=
  v.cars * c.car + v.vans * c.van + v.minibuses * c.minibus

def actualTotalOccupancy (o : ActualOccupancy) : Nat :=
  o.car1 + o.car2 + o.van1 + o.van2 + o.van3 + o.minibus

theorem swim_team_capacity_difference :
  totalMaxCapacity vehicles capacities - actualTotalOccupancy occupancy = 21 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_capacity_difference_l1451_145109


namespace NUMINAMATH_CALUDE_octal_addition_521_146_l1451_145159

/-- Represents an octal number as a list of digits (0-7) in reverse order --/
def OctalNumber := List Nat

/-- Converts an octal number to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- Adds two octal numbers and returns the result in octal --/
def add_octal (a b : OctalNumber) : OctalNumber :=
  sorry

theorem octal_addition_521_146 :
  let a : OctalNumber := [1, 2, 5]  -- 521₈ in reverse order
  let b : OctalNumber := [6, 4, 1]  -- 146₈ in reverse order
  let result : OctalNumber := [7, 6, 6]  -- 667₈ in reverse order
  add_octal a b = result :=
by sorry

end NUMINAMATH_CALUDE_octal_addition_521_146_l1451_145159


namespace NUMINAMATH_CALUDE_book_pages_from_digits_l1451_145177

theorem book_pages_from_digits (total_digits : ℕ) : total_digits = 792 → ∃ (pages : ℕ), pages = 300 ∧ 
  (pages ≤ 9 → total_digits = pages) ∧
  (9 < pages ∧ pages ≤ 99 → total_digits = 9 + 2 * (pages - 9)) ∧
  (99 < pages → total_digits = 189 + 3 * (pages - 99)) :=
by
  sorry

end NUMINAMATH_CALUDE_book_pages_from_digits_l1451_145177


namespace NUMINAMATH_CALUDE_condition_relationship_l1451_145198

theorem condition_relationship :
  (∀ x : ℝ, x^2 - x - 2 < 0 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ x^2 - x - 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1451_145198


namespace NUMINAMATH_CALUDE_gcd_lcm_45_150_l1451_145181

theorem gcd_lcm_45_150 : 
  (Nat.gcd 45 150 = 15) ∧ (Nat.lcm 45 150 = 450) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_45_150_l1451_145181


namespace NUMINAMATH_CALUDE_company_employees_l1451_145114

theorem company_employees (female_managers : ℕ) (male_female_ratio : ℚ) 
  (total_manager_ratio : ℚ) (male_manager_ratio : ℚ) :
  female_managers = 200 →
  male_female_ratio = 3 / 2 →
  total_manager_ratio = 2 / 5 →
  male_manager_ratio = 2 / 5 →
  ∃ (female_employees : ℕ) (total_employees : ℕ),
    female_employees = 500 ∧
    total_employees = 1250 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l1451_145114


namespace NUMINAMATH_CALUDE_whisky_replacement_fraction_l1451_145153

/-- Proves the fraction of whisky replaced given initial and final alcohol percentages -/
theorem whisky_replacement_fraction (initial_percent : ℝ) (replacement_percent : ℝ) (final_percent : ℝ) :
  initial_percent = 0.40 →
  replacement_percent = 0.19 →
  final_percent = 0.24 →
  ∃ (fraction : ℝ), fraction = 0.16 / 0.21 ∧
    initial_percent * (1 - fraction) + replacement_percent * fraction = final_percent :=
by sorry

end NUMINAMATH_CALUDE_whisky_replacement_fraction_l1451_145153


namespace NUMINAMATH_CALUDE_probability_three_blue_marbles_l1451_145122

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 4

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := 5

/-- The number of white marbles in the jar -/
def white_marbles : ℕ := 8

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + blue_marbles + white_marbles + green_marbles

/-- The number of marbles drawn -/
def marbles_drawn : ℕ := 3

theorem probability_three_blue_marbles :
  (blue_marbles : ℚ) / total_marbles *
  ((blue_marbles - 1) : ℚ) / (total_marbles - 1) *
  ((blue_marbles - 2) : ℚ) / (total_marbles - 2) = 1 / 114 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_blue_marbles_l1451_145122


namespace NUMINAMATH_CALUDE_largest_integer_below_sqrt_sum_power_l1451_145113

theorem largest_integer_below_sqrt_sum_power : 
  ∃ n : ℕ, n = 7168 ∧ n < (Real.sqrt 5 + Real.sqrt (3/2))^8 ∧ 
  ∀ m : ℕ, m < (Real.sqrt 5 + Real.sqrt (3/2))^8 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_integer_below_sqrt_sum_power_l1451_145113


namespace NUMINAMATH_CALUDE_simplify_expression_l1451_145118

theorem simplify_expression (a : ℝ) : 3*a - 5*a + a = -a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1451_145118


namespace NUMINAMATH_CALUDE_motorcyclist_distance_l1451_145140

/-- Represents the motion of a motorcyclist --/
structure Motion where
  initial_speed : ℝ
  acceleration : ℝ
  time_to_b : ℝ
  time_b_to_c : ℝ
  speed_at_c : ℝ

/-- Calculates the distance between points A and C --/
def distance_a_to_c (m : Motion) : ℝ :=
  let speed_at_b := m.initial_speed + m.acceleration * m.time_to_b
  let distance_a_to_b := m.initial_speed * m.time_to_b + 0.5 * m.acceleration * m.time_to_b^2
  let distance_b_to_c := speed_at_b * m.time_b_to_c - 0.5 * m.acceleration * m.time_b_to_c^2
  distance_a_to_b - distance_b_to_c

/-- The main theorem to prove --/
theorem motorcyclist_distance (m : Motion) 
  (h1 : m.initial_speed = 90)
  (h2 : m.time_to_b = 3)
  (h3 : m.time_b_to_c = 2)
  (h4 : m.speed_at_c = 110)
  (h5 : m.acceleration = (m.speed_at_c - m.initial_speed) / (m.time_to_b + m.time_b_to_c)) :
  distance_a_to_c m = 92 := by
  sorry


end NUMINAMATH_CALUDE_motorcyclist_distance_l1451_145140


namespace NUMINAMATH_CALUDE_ice_cream_volume_l1451_145116

/-- The volume of ice cream in a cone and hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (1/2) * (4/3) * π * r^3
  h = 8 ∧ r = 2 → cone_volume + hemisphere_volume = 16 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l1451_145116


namespace NUMINAMATH_CALUDE_ribbons_given_in_afternoon_l1451_145172

/-- Given the initial number of ribbons, the number given away in the morning,
    and the number left at the end, prove that the number of ribbons given away
    in the afternoon is 16. -/
theorem ribbons_given_in_afternoon
  (initial : ℕ)
  (morning : ℕ)
  (left : ℕ)
  (h1 : initial = 38)
  (h2 : morning = 14)
  (h3 : left = 8) :
  initial - morning - left = 16 := by
  sorry

end NUMINAMATH_CALUDE_ribbons_given_in_afternoon_l1451_145172


namespace NUMINAMATH_CALUDE_dog_training_weeks_l1451_145164

/-- The number of weeks of training for a seeing-eye dog -/
def training_weeks : ℕ := 12

/-- The adoption fee for an untrained dog in dollars -/
def adoption_fee : ℕ := 150

/-- The cost of training per week in dollars -/
def training_cost_per_week : ℕ := 250

/-- The total certification cost in dollars -/
def certification_cost : ℕ := 3000

/-- The percentage of certification cost covered by insurance -/
def insurance_coverage : ℕ := 90

/-- The total out-of-pocket cost in dollars -/
def total_out_of_pocket : ℕ := 3450

theorem dog_training_weeks :
  adoption_fee +
  training_cost_per_week * training_weeks +
  certification_cost * (100 - insurance_coverage) / 100 =
  total_out_of_pocket :=
by sorry

end NUMINAMATH_CALUDE_dog_training_weeks_l1451_145164


namespace NUMINAMATH_CALUDE_euston_carriages_l1451_145146

/-- The number of carriages in different towns --/
structure Carriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions of the carriage problem --/
def carriage_conditions (c : Carriages) : Prop :=
  c.euston = c.norfolk + 20 ∧
  c.norwich = 100 ∧
  c.flying_scotsman = c.norwich + 20 ∧
  c.euston + c.norfolk + c.norwich + c.flying_scotsman = 460

/-- Theorem stating that under the given conditions, Euston had 130 carriages --/
theorem euston_carriages (c : Carriages) (h : carriage_conditions c) : c.euston = 130 := by
  sorry

end NUMINAMATH_CALUDE_euston_carriages_l1451_145146


namespace NUMINAMATH_CALUDE_tournament_participants_perfect_square_l1451_145124

-- Define the tournament structure
structure ChessTournament where
  masters : ℕ
  grandmasters : ℕ

-- Define the property that each participant scored half their points against masters
def halfPointsAgainstMasters (t : ChessTournament) : Prop :=
  let totalParticipants := t.masters + t.grandmasters
  (t.masters * (t.masters - 1) + t.grandmasters * (t.grandmasters - 1)) / 2 = t.masters * t.grandmasters

-- Theorem statement
theorem tournament_participants_perfect_square (t : ChessTournament) 
  (h : halfPointsAgainstMasters t) : 
  ∃ n : ℕ, (t.masters + t.grandmasters) = n^2 :=
sorry

end NUMINAMATH_CALUDE_tournament_participants_perfect_square_l1451_145124


namespace NUMINAMATH_CALUDE_missing_donuts_percentage_l1451_145106

def initial_donuts : ℕ := 30
def remaining_donuts : ℕ := 9

theorem missing_donuts_percentage :
  (initial_donuts - remaining_donuts) / initial_donuts * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_missing_donuts_percentage_l1451_145106


namespace NUMINAMATH_CALUDE_canoe_oar_probability_l1451_145189

theorem canoe_oar_probability (p_row : ℝ) (h_p_row : p_row = 0.84) : 
  ∃ (p_right : ℝ), 
    (p_right = 1 - Real.sqrt (1 - p_row)) ∧ 
    (p_right = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_canoe_oar_probability_l1451_145189


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1451_145187

theorem quadratic_factorization (c d : ℕ) (h1 : c > d) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - c)*(x - d)) : c - 2*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1451_145187


namespace NUMINAMATH_CALUDE_family_money_sum_l1451_145171

/-- Given Madeline has $48, her brother has half as much as her, and their sister has twice as much as Madeline, the total amount of money all three of them have together is $168. -/
theorem family_money_sum (madeline_money : ℕ) (brother_money : ℕ) (sister_money : ℕ) 
  (h1 : madeline_money = 48)
  (h2 : brother_money = madeline_money / 2)
  (h3 : sister_money = madeline_money * 2) : 
  madeline_money + brother_money + sister_money = 168 := by
  sorry

end NUMINAMATH_CALUDE_family_money_sum_l1451_145171


namespace NUMINAMATH_CALUDE_second_shirt_price_l1451_145121

/-- Proves that the price of the second shirt must be $100 given the conditions --/
theorem second_shirt_price (total_shirts : Nat) (first_shirt_price third_shirt_price : ℝ)
  (remaining_shirts_min_avg : ℝ) (overall_avg : ℝ) :
  total_shirts = 10 →
  first_shirt_price = 82 →
  third_shirt_price = 90 →
  remaining_shirts_min_avg = 104 →
  overall_avg = 100 →
  ∃ (second_shirt_price : ℝ),
    second_shirt_price = 100 ∧
    (first_shirt_price + second_shirt_price + third_shirt_price +
      (total_shirts - 3) * remaining_shirts_min_avg) / total_shirts ≥ overall_avg :=
by sorry

end NUMINAMATH_CALUDE_second_shirt_price_l1451_145121


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1451_145115

theorem lcm_from_product_and_hcf (a b : ℕ+) (h1 : a * b = 987153000) (h2 : Nat.gcd a b = 440) :
  Nat.lcm a b = 2243525 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1451_145115


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1451_145120

theorem shaded_area_calculation (R : ℝ) (h : R = 9) :
  let r : ℝ := R / 2
  let larger_circle_area : ℝ := π * R^2
  let smaller_circle_area : ℝ := π * r^2
  let shaded_area : ℝ := larger_circle_area - 3 * smaller_circle_area
  shaded_area = 20.25 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1451_145120


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1451_145194

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2

-- Define the properties of function f
def is_quadratic (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

def is_odd_sum (f : ℝ → ℝ) : Prop := ∀ x, f x + g x = -(f (-x) + g (-x))

def has_equal_roots (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f x = 3 * x + 2 ∧ 
  ∀ y : ℝ, f y = 3 * y + 2 → y = x

-- Main theorem
theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h1 : is_quadratic f)
  (h2 : is_odd_sum f)
  (h3 : has_equal_roots f) :
  (∀ x, f x = -x^2 + 3*x + 2) ∧ 
  (∀ x, (3 - Real.sqrt 41) / 4 < x ∧ x < (3 + Real.sqrt 41) / 4 → f x > g x) ∧
  (∃ m n : ℝ, m = -1 ∧ n = 17/8 ∧ 
    (∀ x, f x ∈ Set.Icc (-2) (247/64) ↔ x ∈ Set.Icc m n)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1451_145194


namespace NUMINAMATH_CALUDE_rectangular_field_area_decrease_l1451_145151

theorem rectangular_field_area_decrease :
  ∀ (L W : ℝ),
  L > 0 → W > 0 →
  let original_area := L * W
  let new_length := L * (1 - 0.4)
  let new_width := W * (1 - 0.4)
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_decrease_l1451_145151


namespace NUMINAMATH_CALUDE_line_chart_best_for_temperature_l1451_145165

/-- Represents different types of charts --/
inductive ChartType
| BarChart
| LineChart
| PieChart

/-- Represents the characteristics a chart can show --/
structure ChartCharacteristics where
  showsAmount : Bool
  showsChangeOverTime : Bool
  showsPartToWhole : Bool

/-- Defines the characteristics of different chart types --/
def chartTypeCharacteristics : ChartType → ChartCharacteristics
| ChartType.BarChart => ⟨true, false, false⟩
| ChartType.LineChart => ⟨true, true, false⟩
| ChartType.PieChart => ⟨false, false, true⟩

/-- Defines what characteristics are needed for temperature representation --/
def temperatureRepresentationNeeds : ChartCharacteristics :=
  ⟨true, true, false⟩

/-- Theorem: Line chart is the most appropriate for representing temperature changes --/
theorem line_chart_best_for_temperature : 
  ∀ (ct : ChartType), 
    (chartTypeCharacteristics ct = temperatureRepresentationNeeds) → 
    (ct = ChartType.LineChart) :=
by sorry

end NUMINAMATH_CALUDE_line_chart_best_for_temperature_l1451_145165


namespace NUMINAMATH_CALUDE_prob_not_adjacent_l1451_145163

/-- The number of desks in the classroom -/
def num_desks : ℕ := 9

/-- The number of students choosing desks -/
def num_students : ℕ := 2

/-- The number of ways two adjacent desks can be chosen -/
def adjacent_choices : ℕ := num_desks - 1

/-- The probability that two students do not sit next to each other when randomly choosing from a row of desks -/
theorem prob_not_adjacent (n : ℕ) (k : ℕ) (h : n ≥ 2 ∧ k = 2) : 
  (1 : ℚ) - (adjacent_choices : ℚ) / (n.choose k) = 7/9 :=
sorry

end NUMINAMATH_CALUDE_prob_not_adjacent_l1451_145163


namespace NUMINAMATH_CALUDE_probability_diamond_spade_standard_deck_l1451_145149

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (diamonds : ℕ)
  (spades : ℕ)

/-- The probability of drawing a diamond first and then a spade from a standard deck -/
def probability_diamond_then_spade (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.total_cards * d.spades / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a diamond first and then a spade from a standard deck -/
theorem probability_diamond_spade_standard_deck :
  ∃ d : Deck, d.total_cards = 52 ∧ d.diamonds = 13 ∧ d.spades = 13 ∧
  probability_diamond_then_spade d = 13 / 204 := by
  sorry

#check probability_diamond_spade_standard_deck

end NUMINAMATH_CALUDE_probability_diamond_spade_standard_deck_l1451_145149


namespace NUMINAMATH_CALUDE_three_squares_divisible_to_not_divisible_l1451_145197

theorem three_squares_divisible_to_not_divisible (N : ℕ) :
  (∃ (n : ℕ) (a b c : ℤ), N = 9^n * (a^2 + b^2 + c^2) ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ ¬(3 ∣ k) ∧ ¬(3 ∣ m) ∧ ¬(3 ∣ n)) :=
by sorry

end NUMINAMATH_CALUDE_three_squares_divisible_to_not_divisible_l1451_145197


namespace NUMINAMATH_CALUDE_existence_of_five_numbers_l1451_145168

theorem existence_of_five_numbers : ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ), 
  (a₁ + a₂ < 0) ∧ (a₂ + a₃ < 0) ∧ (a₃ + a₄ < 0) ∧ (a₄ + a₅ < 0) ∧ 
  (a₁ + a₂ + a₃ + a₄ + a₅ > 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_five_numbers_l1451_145168


namespace NUMINAMATH_CALUDE_max_b_value_l1451_145101

theorem max_b_value (x b : ℤ) : 
  x^2 + b*x = -21 → 
  b > 0 → 
  (∃ (max_b : ℤ), max_b = 22 ∧ ∀ (b' : ℤ), b' > 0 → (∃ (x' : ℤ), x'^2 + b'*x' = -21) → b' ≤ max_b) :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1451_145101


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l1451_145152

theorem volleyball_team_starters (n m k : ℕ) (h1 : n = 14) (h2 : m = 6) (h3 : k = 3) :
  Nat.choose (n - k) (m - k) = 165 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l1451_145152


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1451_145131

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) : (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l1451_145131


namespace NUMINAMATH_CALUDE_late_students_total_time_l1451_145193

theorem late_students_total_time (charlize_late : ℕ) 
  (h1 : charlize_late = 20)
  (ana_late : ℕ) 
  (h2 : ana_late = charlize_late + 5)
  (ben_late : ℕ) 
  (h3 : ben_late = charlize_late - 15)
  (clara_late : ℕ) 
  (h4 : clara_late = 2 * charlize_late)
  (daniel_late : ℕ) 
  (h5 : daniel_late = clara_late - 10) :
  charlize_late + ana_late + ben_late + clara_late + daniel_late = 120 := by
  sorry

end NUMINAMATH_CALUDE_late_students_total_time_l1451_145193


namespace NUMINAMATH_CALUDE_min_value_of_x2_plus_2y2_l1451_145107

theorem min_value_of_x2_plus_2y2 (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) :
  ∃ (m : ℝ), m = 4 - 2*Real.sqrt 2 ∧ ∀ (a b : ℝ), a^2 - 2*a*b + 2*b^2 = 2 → x^2 + 2*y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_x2_plus_2y2_l1451_145107


namespace NUMINAMATH_CALUDE_participation_schemes_l1451_145166

/-- The number of people to choose from -/
def total_people : ℕ := 5

/-- The number of people to be selected -/
def selected_people : ℕ := 3

/-- The number of projects -/
def num_projects : ℕ := 3

/-- The number of special people (A and B) -/
def special_people : ℕ := 2

/-- Calculates the number of permutations of r items from n -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem participation_schemes : 
  permutations total_people selected_people - 
  permutations (total_people - special_people) selected_people = 54 := by
sorry

end NUMINAMATH_CALUDE_participation_schemes_l1451_145166


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2013_l1451_145133

def last_four_digits (n : ℕ) : ℕ := n % 10000

def cycle_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2013 :
  last_four_digits (5^2013) = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2013_l1451_145133


namespace NUMINAMATH_CALUDE_tunnel_length_l1451_145130

/-- The length of a tunnel given a train passing through it -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) : 
  train_length = 2 →
  exit_time = 4 →
  train_speed = 90 →
  (train_speed / 60 * exit_time) - train_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l1451_145130


namespace NUMINAMATH_CALUDE_marbles_lost_ratio_l1451_145170

/-- Represents the number of marbles Beth has initially -/
def total_marbles : ℕ := 72

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the number of red marbles lost -/
def red_lost : ℕ := 5

/-- Represents the number of marbles Beth has left after losing some -/
def marbles_left : ℕ := 42

/-- Represents the ratio of yellow marbles lost to red marbles lost -/
def yellow_to_red_ratio : ℕ := 3

theorem marbles_lost_ratio :
  ∃ (blue_lost : ℕ),
    (total_marbles / num_colors = total_marbles / num_colors) ∧
    (total_marbles - red_lost - blue_lost - (yellow_to_red_ratio * red_lost) = marbles_left) ∧
    (blue_lost : ℚ) / red_lost = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_ratio_l1451_145170


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1451_145191

open Real

theorem function_inequality_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → x₁ ≠ x₂ →
    |x₁ + a * log x₁ - (x₂ + a * log x₂)| < |1 / x₁ - 1 / x₂|) →
  a < 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l1451_145191


namespace NUMINAMATH_CALUDE_product_of_roots_l1451_145190

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → 
  ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x * y = -34 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1451_145190


namespace NUMINAMATH_CALUDE_range_of_a_l1451_145156

-- Define a decreasing function on (-1, 1)
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : IsDecreasingOn f)
  (h2 : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1451_145156


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1451_145108

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 5x^2 - 9x + 2 has discriminant 41 -/
theorem quadratic_discriminant : discriminant 5 (-9) 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1451_145108


namespace NUMINAMATH_CALUDE_exists_three_adjacent_sum_exceeds_17_l1451_145155

-- Define a type for jersey numbers
def JerseyNumber := Fin 10

-- Define a type for the circular arrangement of players
def CircularArrangement := Fin 10 → JerseyNumber

-- Define a function to check if three consecutive numbers sum to more than 17
def SumExceeds17 (arrangement : CircularArrangement) (i : Fin 10) : Prop :=
  (arrangement i).val + (arrangement (i + 1)).val + (arrangement (i + 2)).val > 17

-- Theorem statement
theorem exists_three_adjacent_sum_exceeds_17 (arrangement : CircularArrangement) :
  (∀ i j : Fin 10, i ≠ j → arrangement i ≠ arrangement j) →
  ∃ i : Fin 10, SumExceeds17 arrangement i := by
  sorry

end NUMINAMATH_CALUDE_exists_three_adjacent_sum_exceeds_17_l1451_145155


namespace NUMINAMATH_CALUDE_orlies_age_l1451_145150

/-- Proves Orlie's age given the conditions about Ruffy and Orlie's ages -/
theorem orlies_age (ruffy_age orlie_age : ℕ) : 
  ruffy_age = 9 →
  ruffy_age = (3 / 4) * orlie_age →
  ruffy_age - 4 = (1 / 2) * (orlie_age - 4) + 1 →
  orlie_age = 12 := by
  sorry

#check orlies_age

end NUMINAMATH_CALUDE_orlies_age_l1451_145150


namespace NUMINAMATH_CALUDE_tax_revenue_decrease_l1451_145104

theorem tax_revenue_decrease (T C : ℝ) (T_positive : T > 0) (C_positive : C > 0) :
  let new_tax := 0.8 * T
  let new_consumption := 1.05 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_tax_revenue_decrease_l1451_145104


namespace NUMINAMATH_CALUDE_fraction_denominator_l1451_145105

theorem fraction_denominator (x : ℕ) : 
  (4128 : ℚ) / x = 0.9411764705882353 → x = 4387 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l1451_145105


namespace NUMINAMATH_CALUDE_inequality_proof_l1451_145182

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) :
  2 ≤ (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ∧ 
  (1 - x^2)^2 + (1 - y^2)^2 + (1 - z^2)^2 ≤ (1 + x) * (1 + y) * (1 + z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1451_145182


namespace NUMINAMATH_CALUDE_exists_circuit_with_rational_resistance_l1451_145111

/-- Represents an electrical circuit composed of unit resistances -/
inductive Circuit
  | unit : Circuit
  | series : Circuit → Circuit → Circuit
  | parallel : Circuit → Circuit → Circuit

/-- Calculates the resistance of a circuit -/
def resistance : Circuit → ℚ
  | Circuit.unit => 1
  | Circuit.series c1 c2 => resistance c1 + resistance c2
  | Circuit.parallel c1 c2 => 1 / (1 / resistance c1 + 1 / resistance c2)

/-- Theorem: For any rational number a/b (where a and b are positive integers),
    there exists an electrical circuit composed of unit resistances
    whose total resistance is equal to a/b -/
theorem exists_circuit_with_rational_resistance (a b : ℕ) (h : b > 0) :
  ∃ c : Circuit, resistance c = a / b := by sorry

end NUMINAMATH_CALUDE_exists_circuit_with_rational_resistance_l1451_145111


namespace NUMINAMATH_CALUDE_max_candy_eaten_l1451_145117

def board_operation (board : List Nat) : Nat → Nat → List Nat :=
  fun i j => (board.removeNth i).removeNth j ++ [board[i]! + board[j]!]

def candy_eaten (board : List Nat) : Nat → Nat → Nat :=
  fun i j => board[i]! * board[j]!

theorem max_candy_eaten :
  ∃ (operations : List (Nat × Nat)),
    operations.length = 33 ∧
    (operations.foldl
      (fun (acc : List Nat × Nat) (op : Nat × Nat) =>
        (board_operation acc.1 op.1 op.2, acc.2 + candy_eaten acc.1 op.1 op.2))
      (List.replicate 34 1, 0)).2 = 561 :=
sorry

end NUMINAMATH_CALUDE_max_candy_eaten_l1451_145117


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1451_145103

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l1451_145103


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1451_145132

theorem power_fraction_simplification :
  (3^2014 + 3^2012) / (3^2014 - 3^2012) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1451_145132


namespace NUMINAMATH_CALUDE_angus_has_55_tokens_l1451_145178

/-- The number of tokens Angus has -/
def angus_tokens (elsa_tokens : ℕ) (token_value : ℕ) (value_difference : ℕ) : ℕ :=
  elsa_tokens - (value_difference / token_value)

/-- Theorem stating that Angus has 55 tokens -/
theorem angus_has_55_tokens (elsa_tokens : ℕ) (token_value : ℕ) (value_difference : ℕ)
  (h1 : elsa_tokens = 60)
  (h2 : token_value = 4)
  (h3 : value_difference = 20) :
  angus_tokens elsa_tokens token_value value_difference = 55 := by
  sorry

end NUMINAMATH_CALUDE_angus_has_55_tokens_l1451_145178


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1451_145102

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 9 = 20) :
  4 * a 5 - a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1451_145102


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l1451_145145

/-- The number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of available colors -/
def available_colors : ℕ := 7

/-- The number of distinguishable ways to paint a cube -/
def distinguishable_cubes : ℕ := 210

theorem cube_painting_theorem :
  (Nat.choose available_colors cube_faces * Nat.factorial cube_faces) / cube_symmetries = distinguishable_cubes :=
sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l1451_145145


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1451_145195

/-- Proves that given a car's speed of 95 km/h in the first hour and an average speed of 77.5 km/h over two hours, the speed in the second hour is 60 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 95)
  (h2 : average_speed = 77.5) : 
  ∃ (speed_second_hour : ℝ), 
    speed_second_hour = 60 ∧ 
    average_speed = (speed_first_hour + speed_second_hour) / 2 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_second_hour_l1451_145195


namespace NUMINAMATH_CALUDE_no_leftover_eggs_l1451_145175

/-- The number of eggs Abigail has -/
def abigail_eggs : ℕ := 58

/-- The number of eggs Beatrice has -/
def beatrice_eggs : ℕ := 35

/-- The number of eggs Carson has -/
def carson_eggs : ℕ := 27

/-- The size of each egg carton -/
def carton_size : ℕ := 10

/-- The theorem stating that there are no leftover eggs -/
theorem no_leftover_eggs : (abigail_eggs + beatrice_eggs + carson_eggs) % carton_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_leftover_eggs_l1451_145175


namespace NUMINAMATH_CALUDE_set_size_comparison_l1451_145144

/-- The size of set A for a given n -/
def size_A (n : ℕ) : ℕ := n^3 + n^5 + n^7 + n^9

/-- The size of set B for a given m -/
def size_B (m : ℕ) : ℕ := m^2 + m^4 + m^6 + m^8

/-- Theorem stating the condition for |B| ≥ |A| when n = 6 -/
theorem set_size_comparison (m : ℕ) :
  size_B m ≥ size_A 6 ↔ m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_set_size_comparison_l1451_145144


namespace NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l1451_145137

/-- Long jump competition with four competitors -/
structure LongJumpCompetition where
  first_jump : ℕ
  second_jump : ℕ
  third_jump : ℕ
  fourth_jump : ℕ

/-- The long jump competition satisfying the given conditions -/
def competition : LongJumpCompetition where
  first_jump := 22
  second_jump := 23
  third_jump := 21
  fourth_jump := 24

/-- Theorem stating the conditions and the result to be proved -/
theorem fourth_competitor_jump_distance :
  let c := competition
  c.first_jump = 22 ∧
  c.second_jump = c.first_jump + 1 ∧
  c.third_jump = c.second_jump - 2 ∧
  c.fourth_jump = c.third_jump + 3 →
  c.fourth_jump = 24 := by
  sorry


end NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l1451_145137


namespace NUMINAMATH_CALUDE_circle_area_l1451_145119

theorem circle_area (r : ℝ) (h : r = 11) : π * r^2 = π * 11^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l1451_145119
