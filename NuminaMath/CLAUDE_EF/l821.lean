import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_times_one_plus_two_i_l821_82170

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (x * i) = Complex.cos x + i * Complex.sin x

-- Define z
noncomputable def z : ℂ := Complex.exp ((Real.pi / 2) * i)

-- Theorem to prove
theorem z_times_one_plus_two_i :
  z * (1 + 2 * i) = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_times_one_plus_two_i_l821_82170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_integers_in_special_set_l821_82156

theorem equal_integers_in_special_set (n : ℕ) (a : Fin (2*n+1) → ℤ) :
  (∀ k : Fin (2*n+1), ∃ (S : Finset (Fin (2*n+1))),
    S.card = n ∧ 
    (Finset.sum S (λ i ↦ a i) = Finset.sum (Finset.univ.erase k \ S) (λ i ↦ a i))) →
  ∀ i j : Fin (2*n+1), a i = a j :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_integers_in_special_set_l821_82156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l821_82184

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the triangle OAB
structure Triangle where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

-- Define the orthocenter (this was missing in the original code)
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.O = (0, 0) ∧
  parabola t.A.1 t.A.2 ∧
  parabola t.B.1 t.B.2 ∧
  (orthocenter t = focus)

-- Define the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ := sorry

-- The theorem to prove
theorem area_of_triangle (t : Triangle) (h : satisfies_conditions t) :
  triangle_area t = 10 * Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l821_82184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_and_sum_l821_82110

theorem log_product_and_sum (x y : ℝ) : 
  x > 0 → y > 0 → (Real.log y / Real.log x) * (Real.log x / Real.log y) = 4 → x * y = 64 → (x + y) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_and_sum_l821_82110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_a_range_l821_82104

/-- The lower bound of the range of a -/
noncomputable def lower_bound : ℝ := 2 / (2 - Real.log 2)

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x

/-- The theorem stating the lower bound of a -/
theorem a_lower_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 4, f a x ≤ g a x) → a ≥ lower_bound :=
by
  sorry

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 4, f a x ≤ g a x) ↔ a ≥ lower_bound :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_a_range_l821_82104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_production_for_given_scenario_l821_82131

/-- Represents the production capacity of a worker in one shift -/
structure ProductionCapacity where
  blanks_per_shift : ℕ
  parts_per_shift : ℕ

/-- Calculates the maximum number of blanks (and parts) that can be produced in one shift -/
def max_production (capacity : ProductionCapacity) : ℕ :=
  (((capacity.blanks_per_shift * capacity.parts_per_shift : ℚ) / (capacity.blanks_per_shift + capacity.parts_per_shift)).floor).toNat

/-- Theorem stating the maximum production for the given scenario -/
theorem max_production_for_given_scenario :
  let capacity := ProductionCapacity.mk 16 10
  max_production capacity = 6 := by
  sorry

#eval max_production (ProductionCapacity.mk 16 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_production_for_given_scenario_l821_82131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_X_n_l821_82192

/-- Markov chain with two boxes, each containing 2 red balls and 1 black ball -/
structure MarkovChain where
  n : ℕ+  -- Number of operations
  X_n : ℕ → Fin 3  -- Number of black balls in Box A after n operations
  a_n : ℕ+ → ℝ  -- Probability of exactly 1 black ball in Box A
  b_n : ℕ+ → ℝ  -- Probability of exactly 2 black balls in Box A

/-- The formula for a_n -/
noncomputable def a_n_formula (n : ℕ+) : ℝ := 3/5 + (2/5) * (-1/9)^(n : ℕ)

/-- The theorem to be proved -/
theorem expectation_X_n (mc : MarkovChain) :
  ∀ n : ℕ+, mc.a_n n = a_n_formula n →
  ∃ E : ℝ, E = 1 ∧ E = mc.a_n n + 2 * mc.b_n n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_X_n_l821_82192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_l821_82146

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  r₁ : ℝ  -- Lower base radius
  r₂ : ℝ  -- Upper base radius
  h : ℝ   -- Height
  r₁_pos : 0 < r₁
  r₂_pos : 0 < r₂
  r₁_ge_r₂ : r₁ ≥ r₂

/-- Calculates the total surface area of a frustum -/
noncomputable def totalSurfaceArea (f : Frustum) : ℝ :=
  let l := Real.sqrt (f.h^2 + (f.r₁ - f.r₂)^2)
  Real.pi * (f.r₁ + f.r₂) * l + Real.pi * f.r₁^2 + Real.pi * f.r₂^2

/-- Theorem stating the total surface area of the specific frustum -/
theorem frustum_surface_area : 
  let f : Frustum := ⟨8, 4, 5, by norm_num, by norm_num, by norm_num⟩
  totalSurfaceArea f = 80 * Real.pi + 12 * Real.pi * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_l821_82146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gpa_probability_l821_82134

noncomputable def gradePoints : Char → ℚ
  | 'A' => 4
  | 'B' => 3
  | 'C' => 2
  | 'D' => 1
  | _   => 0

noncomputable def calculateGPA (g1 g2 g3 g4 : Char) : ℚ :=
  (gradePoints g1 + gradePoints g2 + gradePoints g3 + gradePoints g4) / 4

noncomputable def englishProb : Char → ℚ
  | 'A' => 1/7
  | 'B' => 1/5
  | 'C' => 1/3
  | _   => 0

noncomputable def historyProb : Char → ℚ
  | 'A' => 1/5
  | 'B' => 1/4
  | 'C' => 1/2
  | _   => 0

theorem gpa_probability :
  let validGrades := ['A', 'B', 'C']
  let validCombos := List.filter (fun (g1, g2) => calculateGPA 'A' 'A' g1 g2 ≥ 3.5) (List.product validGrades validGrades)
  (validCombos.map (fun (g1, g2) => englishProb g1 * historyProb g2)).sum = 27/175 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gpa_probability_l821_82134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_similar_numbers_l821_82140

def is_1995_digit (n : ℕ) : Prop := n ≥ 10^1994 ∧ n < 10^1995

def has_no_zero_digit (n : ℕ) : Prop := ∀ d, d ∈ n.digits 10 → d ≠ 0

def are_permutations (a b c : ℕ) : Prop := 
  ∀ d, (a.digits 10).count d = (b.digits 10).count d ∧ 
       (b.digits 10).count d = (c.digits 10).count d

theorem exist_three_similar_numbers : 
  ∃ a b c : ℕ, 
    is_1995_digit a ∧ 
    is_1995_digit b ∧ 
    is_1995_digit c ∧
    has_no_zero_digit a ∧ 
    has_no_zero_digit b ∧ 
    has_no_zero_digit c ∧
    a + b = c ∧
    are_permutations a b c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_three_similar_numbers_l821_82140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_equals_one_l821_82143

-- Define the operation *
noncomputable def star (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

-- Define a function to represent the nested operation
noncomputable def nestedStar : ℕ → ℝ
| 0 => 1000
| n + 1 => star (1000 - n) (nestedStar n)

-- State the theorem
theorem nested_star_equals_one : star 1 (nestedStar 998) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_star_equals_one_l821_82143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_equals_t_l821_82103

/-- Represents a chessboard of size n × n -/
structure Chessboard (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents an L-shaped iron piece -/
inductive LPiece
| mk : LPiece

/-- Represents a T-shaped iron piece -/
inductive TPiece
| mk : TPiece

/-- The total number of iron pieces surrounding the chessboard -/
def totalPieces (n : ℕ) : ℕ := n * (n + 1)

/-- A function that takes a chessboard and returns the number of L-shaped pieces -/
def numLPieces {n : ℕ} (board : Chessboard n) : ℕ := 
  sorry

/-- A function that takes a chessboard and returns the number of T-shaped pieces -/
def numTPieces {n : ℕ} (board : Chessboard n) : ℕ := 
  sorry

/-- Theorem stating that the number of L-shaped pieces equals the number of T-shaped pieces -/
theorem l_equals_t {n : ℕ} (board : Chessboard n) : 
  numLPieces board = numTPieces board := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_equals_t_l821_82103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_square_inequality_l821_82118

theorem inverse_proposition_square_inequality :
  (∀ (m n a : ℝ), m * a^2 > n * a^2 → m > n) ↔ 
  (∀ (m n a : ℝ), m > n → m * a^2 > n * a^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_square_inequality_l821_82118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_20_l821_82160

/-- The side length of a square given the total perimeter of rectangles formed by cutting it -/
noncomputable def square_side_length (num_squares : ℕ) (num_cuts : ℕ) (total_perimeter : ℝ) : ℝ :=
  let num_rectangles := num_squares * (num_cuts + 1)
  let perimeter_per_rectangle := total_perimeter / num_rectangles
  perimeter_per_rectangle * 3 / 8

/-- Theorem stating that under the given conditions, the side length of the squares is 20 cm -/
theorem square_side_length_is_20 :
  square_side_length 5 2 800 = 20 := by
  -- Unfold the definition of square_side_length
  unfold square_side_length
  -- Simplify the arithmetic expressions
  simp [Nat.cast_mul, Nat.cast_add, Nat.cast_one]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_is_20_l821_82160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l821_82138

/-- The time it takes for B to complete the work given A's completion time and B's efficiency relative to A -/
noncomputable def time_for_B (time_A : ℝ) (efficiency_B : ℝ) : ℝ :=
  time_A / (1 + efficiency_B)

/-- Theorem stating that if A can do a piece of work in 12 days and B is 20% more efficient than A,
    then B can do the same piece of work in 10 days -/
theorem work_completion_time 
  (time_A : ℝ) 
  (efficiency_B : ℝ) 
  (h1 : time_A = 12) 
  (h2 : efficiency_B = 0.2) : 
  time_for_B time_A efficiency_B = 10 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l821_82138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l821_82133

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^2 - abs (b * x) - 3

theorem function_properties :
  (∀ b : ℝ, (∀ x : ℝ, f b x ≥ -3) ∧ (∃ x : ℝ, f b x = -3) → b = 0) ∧
  (∀ x : ℝ, x > -2 ∧ x < 2 → -4 ≤ f (-2) x ∧ f (-2) x ≤ -3) ∧
  (∀ b m : ℝ, b ≠ 0 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f b x₁ = m ∧ f b x₂ = m) →
    m > -3 ∨ b^2 = -4*m - 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l821_82133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l821_82162

/-- An arithmetic sequence with a₂ = 1 and a₄ = 5 -/
noncomputable def arithmetic_seq (n : ℕ) : ℝ :=
  let d := (5 - 1) / 2  -- Common difference
  let a₁ := 1 - d       -- First term
  a₁ + (n - 1) * d      -- General term formula

/-- Sum of first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ :=
  let a₁ := arithmetic_seq 1
  n * (2 * a₁ + (n - 1) * ((arithmetic_seq 4 - arithmetic_seq 2) / 2)) / 2

/-- Theorem: The sum of the first 5 terms of the arithmetic sequence is 15 -/
theorem sum_of_first_five_terms :
  S 5 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l821_82162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_meeting_time_l821_82150

/-- Represents a cyclist with a given speed in km/h -/
structure Cyclist where
  speed : ℚ
  deriving Repr

/-- Represents the problem setup -/
structure CyclistProblem where
  cyclists : List Cyclist
  track_length : ℚ
  start_time : ℕ  -- in minutes past midnight

/-- Calculates the time (in minutes) for all cyclists to meet at the center for the nth time -/
noncomputable def meeting_time (problem : CyclistProblem) (n : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem fourth_meeting_time (problem : CyclistProblem) :
  problem.cyclists = [⟨6⟩, ⟨9⟩, ⟨12⟩, ⟨15⟩] →
  problem.track_length = 1/3 →
  problem.start_time = 12 * 60 →
  meeting_time problem 4 = 26 + 40/60 := by
  sorry

#eval Cyclist.mk 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_meeting_time_l821_82150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_pie_consumption_l821_82139

/-- Calculates the average number of apples eaten per guest in Jessica's apple pie scenario -/
theorem apple_pie_consumption (servings_per_pie : ℕ) (num_pies : ℕ) (num_guests : ℕ)
  (cups_per_serving : ℚ) (red_delicious_ratio : ℚ) (granny_smith_ratio : ℚ)
  (red_delicious_cups_per_apple : ℚ) (granny_smith_cups_per_apple : ℚ)
  (red_delicious_conversion : ℚ) (granny_smith_conversion : ℚ)
  (h1 : servings_per_pie = 8)
  (h2 : num_pies = 3)
  (h3 : num_guests = 12)
  (h4 : cups_per_serving = 3/2)
  (h5 : red_delicious_ratio = 2)
  (h6 : granny_smith_ratio = 1)
  (h7 : red_delicious_cups_per_apple = 1)
  (h8 : granny_smith_cups_per_apple = 5/4)
  (h9 : red_delicious_conversion = 7/10)
  (h10 : granny_smith_conversion = 4/5) :
  ∃ (avg_apples_per_guest : ℚ), abs (avg_apples_per_guest - 9/4) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_pie_consumption_l821_82139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_log_expression_l821_82109

theorem max_value_of_log_expression :
  ∃ (m : ℝ), ∀ (a b : ℝ), a ≥ b → b > Real.exp 1 →
  Real.log (a / b) + Real.log (b / (a^2)) ≤ m ∧
  ∃ (a₀ b₀ : ℝ), a₀ ≥ b₀ ∧ b₀ > Real.exp 1 ∧
  Real.log (a₀ / b₀) + Real.log (b₀ / (a₀^2)) = m :=
by
  -- The maximum value is -1, achieved when a = b = e
  use -1
  intro a b ha hb
  have h1 : Real.log (a / b) + Real.log (b / (a^2)) = -Real.log a := by
    -- Proof of logarithm simplification
    sorry
  have h2 : -Real.log a ≤ -1 := by
    -- Proof that -log(a) ≤ -1 when a > e
    sorry
  -- Combining the above results
  constructor
  · -- Proof that the expression is always ≤ -1
    sorry
  · -- Proof that the maximum is achievable
    use (Real.exp 1)
    use (Real.exp 1)
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_log_expression_l821_82109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_units_digit_l821_82113

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

theorem greatest_difference_units_digit :
  ∀ n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (n / 10) % 10 = 4 ∧ (n / 100) = 7 ∧
  is_multiple_of_five n →
  ∃ a b : ℕ,
  is_multiple_of_five (740 + a) ∧
  is_multiple_of_five (740 + b) ∧
  a < 10 ∧ b < 10 ∧
  (∀ c : ℕ, c < 10 ∧ is_multiple_of_five (740 + c) → c ≤ max a b) ∧
  (max a b - min a b) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_units_digit_l821_82113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_50_l821_82149

/-- The angle (in degrees) that the hour hand moves per hour -/
def hour_hand_speed : ℚ := 30

/-- The angle (in degrees) that the minute hand moves per minute -/
def minute_hand_speed : ℚ := 6

/-- The position of the hour hand at 5:50 -/
def hour_hand_position : ℚ := 5 * hour_hand_speed + (50 / 60) * hour_hand_speed

/-- The position of the minute hand at 5:50 -/
def minute_hand_position : ℚ := 50 * minute_hand_speed

/-- The smaller angle between two points on a circle -/
def smaller_angle (a b : ℚ) : ℚ :=
  min (abs (a - b)) (360 - abs (a - b))

/-- The theorem stating that the smaller angle between clock hands at 5:50 is 125° -/
theorem clock_angle_at_5_50 :
  smaller_angle hour_hand_position minute_hand_position = 125 := by
  sorry

#eval smaller_angle hour_hand_position minute_hand_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_5_50_l821_82149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_m_l821_82176

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [0, 2, 4, 5, 7, 9]

theorem sum_of_digits_m (M : ℕ) 
  (h_even : is_even M)
  (h_valid : valid_digits M)
  (h_double : sum_of_digits (2 * M) = 43)
  (h_half : sum_of_digits (M / 2) = 31) :
  sum_of_digits M = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_m_l821_82176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_less_than_one_l821_82159

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(|x - 3|) - 1

-- State the theorem
theorem solution_set_of_f_less_than_one :
  {x : ℝ | f x < 1} = Set.Ioo 2 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_less_than_one_l821_82159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_per_horseshoe_l821_82182

/-- Represents the problem of calculating iron needed for a horseshoe --/
structure HorseshoeIronProblem where
  total_iron : ℕ
  farm_count : ℕ
  horses_per_farm : ℕ
  stable_count : ℕ
  horses_per_stable : ℕ
  riding_school_horses : ℕ
  shoes_per_horse : ℕ

/-- The specific problem instance --/
def problem : HorseshoeIronProblem :=
  { total_iron := 400
  , farm_count := 2
  , horses_per_farm := 2
  , stable_count := 2
  , horses_per_stable := 5
  , riding_school_horses := 36
  , shoes_per_horse := 4 }

/-- Calculates the total number of horseshoes made --/
def total_horseshoes (p : HorseshoeIronProblem) : ℕ :=
  (p.farm_count * p.horses_per_farm + p.stable_count * p.horses_per_stable + p.riding_school_horses) * p.shoes_per_horse

/-- Theorem stating that 2 kg of iron is needed per horseshoe --/
theorem iron_per_horseshoe (p : HorseshoeIronProblem) :
  p.total_iron / total_horseshoes p = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_per_horseshoe_l821_82182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bigNumber_factorization_l821_82141

/-- Represents a number with 100 ones followed by 100 twos -/
def bigNumber : ℕ := 10^200 + 2 * (10^100 - 1) / 9

/-- Represents a number with 100 threes -/
def hundredThrees : ℕ := (10^100 - 1) / 3

theorem bigNumber_factorization : 
  bigNumber = hundredThrees * (hundredThrees + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bigNumber_factorization_l821_82141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cosine_transformation_l821_82128

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 3 + Real.pi / 4)

theorem period_of_cosine_transformation :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cosine_transformation_l821_82128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_inverse_proportion_l821_82165

/-- Represents a rhombus with a fixed area -/
structure Rhombus where
  area : ℝ
  diag1 : ℝ
  diag2 : ℝ
  area_eq : area = (1/2) * diag1 * diag2

/-- Defines an inverse proportion function -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- Theorem: In a rhombus with area 20, the relationship between its diagonals is an inverse proportion -/
theorem rhombus_diagonals_inverse_proportion :
  ∀ (r : Rhombus), r.area = 20 →
  is_inverse_proportion (λ x ↦ 2 * r.area / x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonals_inverse_proportion_l821_82165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_is_circle_l821_82190

noncomputable section

-- Define the parametric equations of lines C₁ and C₂
def C₁ (α t : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, t * Real.sin α)
def C₂ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the perpendicular line OA
def OA (α : ℝ) : ℝ → ℝ × ℝ := λ t ↦ (t * Real.cos α, t * Real.sin α)

-- Define point A as the intersection of C₁ and OA
def A (α : ℝ) : ℝ × ℝ := (Real.sin α ^ 2, -Real.cos α * Real.sin α)

-- Define point P as the midpoint of OA
def P (α : ℝ) : ℝ × ℝ := ((Real.sin α ^ 2) / 2, -(Real.cos α * Real.sin α) / 2)

-- Theorem statement
theorem trajectory_of_P_is_circle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1/4, 0) ∧ radius = 1/4 ∧
    ∀ α, (P α).1 - center.1 ^ 2 + (P α).2 - center.2 ^ 2 = radius ^ 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_is_circle_l821_82190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l821_82108

-- Define the constants
noncomputable def a : ℝ := 4^(1/2)
noncomputable def b : ℝ := (1/2)^4
noncomputable def c : ℝ := Real.log 4 / Real.log (1/2)

-- State the theorem
theorem ascending_order : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l821_82108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_ab_range_l821_82106

theorem circle_symmetry_ab_range (a b : ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*x - 4*y + 1 = 0}
  let L := {(x, y) : ℝ × ℝ | 2*a*x - b*y + 2 = 0}
  (∀ (p : ℝ × ℝ), p ∈ C → (∃ (q : ℝ × ℝ), q ∈ C ∧ q ≠ p ∧ Set.Icc p q ∩ L ≠ ∅)) →
  a * b ≤ 1/4 ∧ ∀ (k : ℝ), k < 1/4 → ∃ (a' b' : ℝ), a' * b' = k ∧
    (∀ (p : ℝ × ℝ), p ∈ C → (∃ (q : ℝ × ℝ), q ∈ C ∧ q ≠ p ∧ Set.Icc p q ∩ {(x, y) : ℝ × ℝ | 2*a'*x - b'*y + 2 = 0} ≠ ∅)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_ab_range_l821_82106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l821_82142

-- Define the function f
noncomputable def f (x : ℝ) := Real.log ((x - 1) / (x + 1)) / Real.log 2

-- Define the set A (domain of f)
def A : Set ℝ := {x | x < -1 ∨ x > 1}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 2) < 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (a ≤ -3 ∨ a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l821_82142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_special_ratio_l821_82154

/-- A cyclic quadrilateral with angles in degrees -/
structure CyclicQuadrilateral :=
  (a b c d : ℝ)
  (sum_360 : a + b + c + d = 360)
  (opposite_180 : a + c = 180 ∧ b + d = 180)

/-- Theorem: In a cyclic quadrilateral where three consecutive angles are in the ratio 1:2:3,
    the angles are 45°, 90°, 135°, and 90°. -/
theorem cyclic_quadrilateral_special_ratio 
  (q : CyclicQuadrilateral) 
  (ratio : q.a = q.b / 2 ∧ q.b = q.c / 3 ∧ q.a = q.c / 6) :
  q.a = 45 ∧ q.b = 90 ∧ q.c = 135 ∧ q.d = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_special_ratio_l821_82154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_12_houses_l821_82172

def paperboy_delivery_sequences : ℕ → ℕ
  | 0 => 1  -- base case for 0 houses
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | n + 5 => paperboy_delivery_sequences (n + 4) + paperboy_delivery_sequences (n + 3) + 
             paperboy_delivery_sequences (n + 2) + paperboy_delivery_sequences (n + 1)

theorem paperboy_delivery_12_houses : paperboy_delivery_sequences 12 = 2872 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperboy_delivery_12_houses_l821_82172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_a_range_l821_82145

-- Define the function f(x) as noncomputable due to its dependency on Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt x / (x^3 - 3*x + a)

-- State the theorem
theorem f_domain_implies_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x ≠ 0) ↔ a > 2 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_a_range_l821_82145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_mpg_increase_is_7_5_l821_82166

/-- Represents the fuel consumption of a car -/
structure CarFuelConsumption where
  distance : ℚ  -- Distance in miles (using rationals instead of reals)
  fuel : ℚ      -- Fuel used in gallons
  targetFuel : ℚ -- Target fuel consumption in gallons

/-- Calculates the average increase in miles per gallon needed for a set of cars -/
def averageMpgIncrease (cars : List CarFuelConsumption) : ℚ :=
  let currentMpgs := cars.map (λ c => c.distance / c.fuel)
  let requiredMpgs := cars.map (λ c => c.distance / c.targetFuel)
  let increases := List.zip currentMpgs requiredMpgs |>.map (λ (current, required) => required - current)
  increases.sum / cars.length

/-- The main theorem stating the average MPG increase for the given cars -/
theorem average_mpg_increase_is_7_5 :
  let cars := [
    ⟨180, 12, 10⟩,  -- Car A
    ⟨225, 15, 10⟩,  -- Car B
    ⟨270, 18, 10⟩   -- Car C
  ]
  averageMpgIncrease cars = 15/2 := by sorry

#eval averageMpgIncrease [
  ⟨180, 12, 10⟩,  -- Car A
  ⟨225, 15, 10⟩,  -- Car B
  ⟨270, 18, 10⟩   -- Car C
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_mpg_increase_is_7_5_l821_82166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_and_sin_values_l821_82199

theorem tan_and_sin_values (α : ℝ) 
  (h1 : Real.tan (α + π/4) = -3) 
  (h2 : α ∈ Set.Ioo 0 (π/2)) : 
  Real.tan α = 2 ∧ 
  Real.sin (2*α - π/3) = (4 + 3*Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_and_sin_values_l821_82199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_factorization_l821_82186

/-- A polynomial over a field -/
def MyPolynomial (F : Type*) [Field F] := F → F

/-- Evaluation of a polynomial at a point -/
def eval {F : Type*} [Field F] (f : MyPolynomial F) (x : F) : F := f x

/-- The degree of a polynomial -/
noncomputable def degree {F : Type*} [Field F] (f : MyPolynomial F) : ℕ := sorry

theorem polynomial_root_factorization {F : Type*} [Field F] (f : MyPolynomial F) (x₁ : F) :
  eval f x₁ = 0 →
  ∃ (g : MyPolynomial F),
    (∀ x, f x = (x - x₁) * g x) ∧
    (degree g = degree f - 1) ∧
    (∀ x, x ≠ x₁ → eval f x = 0 → eval g x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_factorization_l821_82186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_terminates_l821_82122

def replace_pair (seq : List Nat) : List (List Nat) :=
  seq.zip seq.tail
    |> List.filterMap (fun (x, y) =>
      if x > y then
        some [(x - 1) :: x :: seq.drop 2, (y + 1) :: x :: seq.drop 2]
      else
        none)
    |> List.join

def iterate_replace_pair : List Nat → Nat → List Nat
  | seq, 0 => seq
  | seq, n + 1 =>
    match replace_pair seq with
    | [] => seq
    | (new_seq :: _) => iterate_replace_pair new_seq n

theorem process_terminates (initial_seq : List Nat) :
  ∃ n : Nat, iterate_replace_pair initial_seq n = iterate_replace_pair initial_seq (n + 1) := by
  sorry

#eval iterate_replace_pair [3, 1, 4, 2] 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_terminates_l821_82122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l821_82148

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 ∧ 
  (∃ (c : ℝ), c > 0 ∧ ∀ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 → y^2 ≤ c^2)

def q (m : ℝ) : Prop := ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ 
  ∀ (x y : ℝ), y^2 / 5 - x^2 / m = 1 → 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ e = (1 + b^2/a^2)^(1/2))

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (¬(p m) ∧ ¬(q m) ∧ (p m ∨ q m)) → (1/3 ≤ m ∧ m < 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l821_82148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_proof_l821_82144

/-- Calculates the final concentration of HNO3 after adding pure HNO3 to an existing solution -/
noncomputable def final_concentration (initial_volume : ℝ) (initial_concentration : ℝ) (added_volume : ℝ) : ℝ :=
  let initial_hno3 := initial_volume * initial_concentration
  let total_hno3 := initial_hno3 + added_volume
  let final_volume := initial_volume + added_volume
  (total_hno3 / final_volume) * 100

/-- Proves that adding 24 liters of pure HNO3 to 60 liters of 30% HNO3 solution results in a 50% concentration -/
theorem hno3_concentration_proof :
  final_concentration 60 0.3 24 = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hno3_concentration_proof_l821_82144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trisection_ratio_l821_82115

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the distance between two points. -/
def dist (P Q : Point) : ℝ := sorry

/-- Represents a triangle with vertices X, Y, and Z. -/
def Triangle (X Y Z : Point) : Prop := sorry

/-- Represents that lines ZF and ZG trisect angle Z in triangle XYZ,
    meeting side XY at points F and G respectively. -/
def TrisectsAngle (Z F G X Y : Point) : Prop := sorry

/-- Given a triangle XYZ with angle Z trisected by lines ZF and ZG meeting side XY at points F and G respectively,
    prove that AF/GB = (AZ)(ZF) / (ZG)(ZY). -/
theorem triangle_trisection_ratio (X Y Z F G A B : Point) (h1 : Triangle X Y Z)
    (h2 : TrisectsAngle Z F G X Y) : 
    (dist A F) / (dist G B) = ((dist A Z) * (dist Z F)) / ((dist Z G) * (dist Z Y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trisection_ratio_l821_82115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l821_82168

/-- The time it takes for b and c to complete the work -/
noncomputable def x : ℝ := 24

/-- The work rate of person a -/
noncomputable def A : ℝ := 1 / 8 - 1 / 24

/-- The work rate of person b -/
noncomputable def B : ℝ := 1 / x - 1 / 24

/-- The work rate of person c -/
noncomputable def C : ℝ := 1 / 24

theorem work_completion_time :
  (A + B = 1 / 8) ∧
  (B + C = 1 / x) ∧
  (A + B + C = 1 / 6) ∧
  (A + C = 1 / 8) →
  x = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l821_82168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l821_82114

/-- Time for pipe to fill tank without leak -/
def T : ℝ := sorry

/-- Time to fill tank with leak -/
def fill_time_with_leak : ℝ := 8

/-- Time for leak to empty full tank -/
def leak_empty_time : ℝ := 8

/-- Theorem stating the time for pipe to fill tank without leak -/
theorem pipe_fill_time : T = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_fill_time_l821_82114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l821_82151

-- Define the ellipse C
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the focus of the ellipse
noncomputable def focus (a : ℝ) : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define point P
noncomputable def P : ℝ × ℝ := (1/2, 1/2)

-- Define the line l
def line_l (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the midpoint condition
def is_midpoint (p q r : ℝ × ℝ) : Prop :=
  p.1 = (q.1 + r.1) / 2 ∧ p.2 = (q.2 + r.2) / 2

theorem ellipse_and_line_theorem (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), ellipse a x y ∧ (x, y) = focus a) →
  (∃ (m b : ℝ) (A B : ℝ × ℝ),
    ellipse a A.1 A.2 ∧
    ellipse a B.1 B.2 ∧
    line_l m b A.1 A.2 ∧
    line_l m b B.1 B.2 ∧
    line_l m b P.1 P.2 ∧
    is_midpoint P A B) →
  a = 2 ∧ ∃ (m b : ℝ), m = -1/4 ∧ b = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l821_82151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l821_82185

-- Define the ellipse Γ
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
variable (a b c : ℝ)

axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom right_focus : c = 2 * Real.sqrt 2
axiom sum_of_distances : ∀ (x y : ℝ), ellipse a b x y → 
  Real.sqrt ((x - c)^2 + y^2) + Real.sqrt ((x + c)^2 + y^2) = 4 * Real.sqrt 3

-- Define the line l
def line (m : ℝ) (x : ℝ) : ℝ := x + m

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- State the theorem
theorem ellipse_properties :
  -- Part 1: Standard equation of the ellipse
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  -- Part 2: Properties of intersection points and equidistant point
  (∀ (m : ℝ) (xA yA xB yB x0 : ℝ),
    -- A and B are distinct intersection points of line l and ellipse Γ
    ellipse a b xA yA ∧ 
    ellipse a b xB yB ∧ 
    yA = line m xA ∧
    yB = line m xB ∧
    (xA, yA) ≠ (xB, yB) ∧
    -- |AB| = 3√2
    distance xA yA xB yB = 3 * Real.sqrt 2 ∧
    -- P(x0, 2) is equidistant from A and B
    distance x0 2 xA yA = distance x0 2 xB yB →
    -- Then x0 = -3 or x0 = -1
    x0 = -3 ∨ x0 = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l821_82185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_is_five_l821_82105

/-- An arithmetic sequence with 11 terms where the sum of odd-numbered terms is 30 -/
structure ArithmeticSequence where
  terms : Fin 11 → ℝ
  is_arithmetic : ∀ i j k, i.val + 1 = j.val ∧ j.val + 1 = k.val →
    terms j - terms i = terms k - terms j
  sum_odd_terms : (Finset.filter (λ i : Fin 11 => i.val % 2 = 0) (Finset.univ)).sum (λ i => terms i) = 30

/-- The middle term of the sequence is the 6th term -/
def middle_term (seq : ArithmeticSequence) : ℝ := seq.terms ⟨5, by norm_num⟩

theorem middle_term_is_five (seq : ArithmeticSequence) : middle_term seq = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_is_five_l821_82105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_four_l821_82126

/-- A function f with a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / (x + 1) + x

/-- The derivative of f with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := -a / ((x + 1) ^ 2) + 1

theorem extremum_implies_a_equals_four :
  ∀ a : ℝ, (∃ x : ℝ, f_derivative a x = 0 ∧ x = 1) → a = 4 := by
  intro a h
  cases' h with x hx
  have h1 : f_derivative a 1 = 0 := by
    rw [hx.right] at hx
    exact hx.left
  simp [f_derivative] at h1
  field_simp at h1
  linarith

#check extremum_implies_a_equals_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_four_l821_82126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directional_derivative_z_at_M_towards_M1_l821_82130

noncomputable def z (x y : ℝ) : ℝ := x^2 + y^2

def M : ℝ × ℝ := (3, 1)
def M1 : ℝ × ℝ := (0, 5)

def direction_vector : ℝ × ℝ := (M1.1 - M.1, M1.2 - M.2)

noncomputable def magnitude : ℝ := Real.sqrt ((direction_vector.1)^2 + (direction_vector.2)^2)

noncomputable def unit_vector : ℝ × ℝ := (direction_vector.1 / magnitude, direction_vector.2 / magnitude)

theorem directional_derivative_z_at_M_towards_M1 :
  let grad_z := (2 * M.1, 2 * M.2)
  (grad_z.1 * unit_vector.1 + grad_z.2 * unit_vector.2) = -2 := by sorry

#check directional_derivative_z_at_M_towards_M1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directional_derivative_z_at_M_towards_M1_l821_82130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_reasoning_l821_82107

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Tangent line to a circle -/
def TangentLineToCircle (c : Circle) (l : Line) : Prop :=
  sorry

/-- Tangent plane to a sphere -/
def TangentPlaneToSphere (s : Sphere) (p : Plane) : Prop :=
  sorry

/-- Perpendicular line from circle center to tangent point -/
def PerpendicularLineToTangent (c : Circle) (l : Line) : Prop :=
  sorry

/-- Perpendicular line from sphere center to tangent point -/
def PerpendicularLineToTangentPlane (s : Sphere) (p : Plane) : Prop :=
  sorry

/-- Types of reasoning -/
inductive ReasoningType where
  | Inductive
  | Deductive
  | Analogical
  | Transitive

/-- Theorem stating the type of reasoning used -/
theorem tangent_reasoning :
  (∀ (c : Circle) (l : Line), TangentLineToCircle c l → PerpendicularLineToTangent c l) →
  (∀ (s : Sphere) (p : Plane), TangentPlaneToSphere s p → PerpendicularLineToTangentPlane s p) →
  ReasoningType.Analogical = ReasoningType.Analogical :=
by
  intros _ _
  rfl

#check tangent_reasoning

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_reasoning_l821_82107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_three_one_third_l821_82125

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The probability mass function for a binomial distribution -/
def binomialProbability (ξ : BinomialDistribution) (k : ℕ) : ℝ :=
  (Nat.choose ξ.n k) * (ξ.p ^ k) * ((1 - ξ.p) ^ (ξ.n - k))

/-- Theorem: For a binomial distribution B(3, 1/3), P(ξ = 1) = 4/9 -/
theorem binomial_probability_three_one_third (ξ : BinomialDistribution) 
  (h2 : ξ.n = 3) (h3 : ξ.p = 1/3) : 
  binomialProbability ξ 1 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_three_one_third_l821_82125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l821_82111

/-- The equation of a hyperbola in the form (ax + b)^2/c^2 - (dy + e)^2/f^2 = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a hyperbola -/
structure HyperbolaCenter where
  x : ℝ
  y : ℝ

/-- Function to calculate the center of a hyperbola -/
noncomputable def calculateCenter (h : Hyperbola) : HyperbolaCenter :=
  { x := -h.b / h.a,
    y := -h.e / h.d }

theorem hyperbola_center_correct (h : Hyperbola) 
    (hₐ : h.a = 4) (hₑ : h.b = -8) (hₘ : h.c = 9)
    (hₓ : h.d = 5) (hₒ : h.e = -15) (hᵣ : h.f = 7) :
  calculateCenter h = { x := 2, y := 3 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l821_82111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l821_82158

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def contains_digit (n : ℕ) (d : ℕ) : Bool :=
  let digits := n.digits 10
  d ∈ digits

theorem sum_of_numbers (A B C : ℕ) :
  is_three_digit A →
  is_two_digit B →
  is_two_digit C →
  (contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7) →
  (if contains_digit A 7 then A else 0) + 
  (if contains_digit B 7 then B else 0) + 
  (if contains_digit C 7 then C else 0) = 208 →
  contains_digit B 3 →
  contains_digit C 3 →
  B + C = 76 →
  A + B + C = 247 := by
  sorry

#eval contains_digit 123 2
#eval contains_digit 456 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numbers_l821_82158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l821_82120

-- Define the circle C
def circleC (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 4

-- Define the curve
def curve (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * abs (x - 1) - 2

-- Define point M
def point_M : ℝ × ℝ := (1, -4)

-- Define the area of a triangle given three points
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))

-- Theorem statement
theorem triangle_area :
  ∃ (A B : ℝ × ℝ),
    circleC A.1 A.2 ∧
    circleC B.1 B.2 ∧
    curve A.1 A.2 ∧
    curve B.1 B.2 ∧
    (area_triangle A B point_M = 2 + Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l821_82120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equality_for_inverse_sum_equation_l821_82101

/-- For any integer n ≥ 2 and two n×n matrices with real entries A and B
    that satisfy A⁻¹ + B⁻¹ = (A + B)⁻¹, prove that det(A) = det(B). -/
theorem det_equality_for_inverse_sum_equation {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ)
    (h_n : n ≥ 2)
    (h_inv : A⁻¹ + B⁻¹ = (A + B)⁻¹) :
  Matrix.det A = Matrix.det B := by
  sorry

#check det_equality_for_inverse_sum_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equality_for_inverse_sum_equation_l821_82101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_basketball_cards_l821_82197

/-- Represents the cost of items and financial operations in a shopping scenario --/
structure ShoppingScenario where
  -- Mary's purchases in euros
  sunglasses_cost : ℝ
  jeans_cost : ℝ
  mary_discount_rate : ℝ
  -- Jack's purchases in pounds
  sweater_cost : ℝ
  football_cost : ℝ
  watch_cost : ℝ
  jack_tax_rate : ℝ
  -- Rose's purchases in dollars
  shoes_cost : ℝ
  shoe_discount_rate : ℝ
  rose_tax_rate : ℝ
  -- Exchange rates
  euro_to_dollar : ℝ
  pound_to_dollar : ℝ
  -- Number of card decks
  num_card_decks : ℕ

/-- Theorem stating the cost of one deck of basketball cards --/
theorem cost_of_basketball_cards (s : ShoppingScenario) :
  s.sunglasses_cost = 50 ∧
  s.jeans_cost = 100 ∧
  s.mary_discount_rate = 0.1 ∧
  s.sweater_cost = 80 ∧
  s.football_cost = 40 ∧
  s.watch_cost = 65 ∧
  s.jack_tax_rate = 0.08 ∧
  s.shoes_cost = 150 ∧
  s.shoe_discount_rate = 0.05 ∧
  s.rose_tax_rate = 0.07 ∧
  s.euro_to_dollar = 1.2 ∧
  s.pound_to_dollar = 1.3 ∧
  s.num_card_decks = 3 →
  ∃ (card_deck_cost : ℝ), abs (card_deck_cost - 35.76) < 0.01 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_basketball_cards_l821_82197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_simplification_and_evaluation_l821_82116

-- Define the original expression
noncomputable def original_expr (x : ℝ) : ℝ := 
  (x^2 - 2*x + 1) / (x^2 - 1) / (1 - 3/(x + 1))

-- Define the simplified expression
noncomputable def simplified_expr (x : ℝ) : ℝ := (x - 1) / (x - 2)

-- State the theorem
theorem expr_simplification_and_evaluation :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 2 → original_expr x = simplified_expr x) ∧
  simplified_expr 3 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_simplification_and_evaluation_l821_82116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_piece_estimation_l821_82132

/-- Represents the composition of chess pieces in a box -/
structure ChessBox where
  black : ℕ
  white : ℕ

/-- Represents the result of multiple random draws from the box -/
structure DrawResult where
  total : ℕ
  black : ℕ

/-- Calculates the expected number of white pieces given a draw result -/
def expectedWhitePieces (box : ChessBox) (result : DrawResult) : ℚ :=
  (box.white : ℚ) / (box.black + box.white : ℚ) * result.total

theorem chess_piece_estimation (box : ChessBox) (result : DrawResult) :
  box.black = 10 ∧ result.total = 300 ∧ result.black = 100 →
  expectedWhitePieces box result = 200 →
  box.white = 20 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_piece_estimation_l821_82132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fran_baked_40_green_macaroons_l821_82179

/-- The number of green macaroons Fran baked -/
def green_macaroons : ℕ := sorry

/-- The number of red macaroons Fran baked -/
def red_macaroons : ℕ := 50

/-- The number of green macaroons Fran ate -/
def eaten_green : ℕ := 15

/-- The number of red macaroons Fran ate -/
def eaten_red : ℕ := 2 * eaten_green

/-- The total number of remaining macaroons -/
def remaining_macaroons : ℕ := 45

theorem fran_baked_40_green_macaroons :
  green_macaroons = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fran_baked_40_green_macaroons_l821_82179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theodore_earnings_l821_82117

/-- Represents Theodore's monthly statue production and sales --/
structure StatueProduction where
  stone_count : ℕ
  wooden_count : ℕ
  stone_price : ℕ
  wooden_price : ℕ
  tax_rate : ℚ

/-- Calculates the total monthly earnings after taxes --/
def total_earnings_after_taxes (p : StatueProduction) : ℚ :=
  let total_before_tax := (p.stone_count * p.stone_price + p.wooden_count * p.wooden_price : ℚ)
  let tax_amount := p.tax_rate * total_before_tax
  total_before_tax - tax_amount

/-- Theodore's actual production values --/
def theodore_production : StatueProduction :=
  { stone_count := 10
  , wooden_count := 20
  , stone_price := 20
  , wooden_price := 5
  , tax_rate := 1/10 }

/-- Theorem stating Theodore's monthly earnings after taxes --/
theorem theodore_earnings :
  total_earnings_after_taxes theodore_production = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theodore_earnings_l821_82117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_and_intersection_l821_82129

/-- Hyperbola properties and intersection length -/
theorem hyperbola_properties_and_intersection :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (b / a = Real.sqrt 3 / 3) ∧
  (2 * b^2 / a = 6) ∧
  (∃ (x y : ℝ → ℝ), ∀ t, y t^2 - (x t)^2 / 3 = 1) ∧
  (∃ (M N : ℝ × ℝ),
    (M.2 - N.2)^2 + (M.1 - N.1)^2 = 36 ∧
    (M.2 = M.1 - 2 ∧ N.2 = N.1 - 2) ∧
    (M.2^2 - M.1^2 / 3 = 1) ∧
    (N.2^2 - N.1^2 / 3 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_and_intersection_l821_82129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_value_l821_82112

theorem multiple_value (q : ℚ) (m : ℚ) : 
  (q = 1 ∧ (5 - q) = 3 * (m * q - 1)) → m = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_value_l821_82112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pants_retail_price_l821_82177

/-- Given a wholesale price and a markup percentage, calculate the retail price -/
noncomputable def retail_price (wholesale : ℝ) (markup_percent : ℝ) : ℝ :=
  wholesale * (1 + markup_percent / 100)

/-- Theorem: The retail price of a pair of pants with a wholesale price of $20 
    and an 80% markup is $36 -/
theorem pants_retail_price : 
  retail_price 20 80 = 36 := by
  -- Unfold the definition of retail_price
  unfold retail_price
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that 20 * (1 + 80 / 100) = 36
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pants_retail_price_l821_82177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samia_journey_time_l821_82163

/-- Represents Samia's journey --/
structure Journey where
  bike_speed : ℝ
  bike_time : ℝ
  bus_speed : ℝ
  bus_time : ℝ
  walk_distance : ℝ
  walk_speed : ℝ

/-- Calculates the total journey time --/
noncomputable def total_journey_time (j : Journey) : ℝ :=
  j.bike_time + j.bus_time + j.walk_distance / j.walk_speed

/-- Theorem stating that Samia's journey takes 4.5 hours --/
theorem samia_journey_time :
  let j : Journey := {
    bike_speed := 20,
    bike_time := 3,
    bus_speed := 60,
    bus_time := 0.5,
    walk_distance := 4,
    walk_speed := 4
  }
  total_journey_time j = 4.5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_samia_journey_time_l821_82163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l821_82175

-- Define the function f on the real numbers
def f : ℝ → ℝ := sorry

-- Define the property of f being odd on [-5, 5]
def is_odd_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (-5) 5 → f (-x) = -f x

-- State the theorem
theorem odd_function_property (h_odd : is_odd_on_interval f) (h_ineq : f 3 < f 1) :
  f (-1) < f (-3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l821_82175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_iff_m_in_range_l821_82180

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + 6*m*x + m + 8)

-- State the theorem
theorem domain_is_real_iff_m_in_range :
  ∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ -8/9 ≤ m ∧ m ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_iff_m_in_range_l821_82180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l821_82124

-- Define the circle quadrant
def circle_quadrant (x y : ℝ) : Prop := x^2 + y^2 = 2 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2015
noncomputable def g (x : ℝ) : ℝ := 2015^x

-- State that f and g are inverse functions
axiom f_g_inverse : ∀ x, f (g x) = x ∧ g (f x) = x

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  f x₁ = y₁ ∧ g x₁ = y₁ ∧ f x₂ = y₂ ∧ g x₂ = y₂

-- Theorem statement
theorem intersection_points_on_circle (x₁ y₁ x₂ y₂ : ℝ) :
  circle_quadrant x₁ x₂ → intersection_points x₁ y₁ x₂ y₂ → x₁^2 + x₂^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_on_circle_l821_82124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_3_504_closest_to_6_l821_82123

noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 * a * b / (a + b)

theorem harmonic_mean_3_504_closest_to_6 :
  let h := harmonic_mean 3 504
  ∀ n : ℤ, n ≠ 6 → |h - 6| < |h - (n : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_3_504_closest_to_6_l821_82123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_polynomials_l821_82102

/-- Given polynomial functions f, g, and h, prove their sum equals a specific polynomial. -/
theorem sum_of_polynomials (x : ℝ) : 
  (-4 * x^3 - 3 * x^2 + 2 * x - 5) + 
  (-6 * x^2 + 4 * x - 9) + 
  (7 * x^3 + 4 * x^2 + 6 * x + 3) = 
  3 * x^3 - 5 * x^2 + 12 * x - 11 := by
  ring  -- This tactic should solve the equation automatically
  -- If 'ring' doesn't work, you can use 'sorry' instead:
  -- sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_polynomials_l821_82102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_A_proposition_B_both_propositions_true_l821_82121

-- Define the curve Ck
def Ck (k : ℚ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.rpow p.1 k + Real.rpow p.2 k = 4 ∧ k > 0}

-- Define the area enclosed by the curve and coordinate axes
noncomputable def area_enclosed (k : ℚ) : ℝ :=
  sorry -- Definition of area calculation

theorem proposition_A : area_enclosed (1/2) < 128 := by
  sorry

theorem proposition_B : ∀ n : ℕ, area_enclosed (2 * ↑n) > 4 := by
  sorry

-- Proof that both propositions are true
theorem both_propositions_true : 
  (area_enclosed (1/2) < 128) ∧ (∀ n : ℕ, area_enclosed (2 * ↑n) > 4) := by
  constructor
  · exact proposition_A
  · exact proposition_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_A_proposition_B_both_propositions_true_l821_82121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_value_decrease_l821_82195

theorem property_value_decrease (initial_value : ℝ) (h : initial_value > 0) :
  let increased_value := initial_value * 1.3
  let decrease_factor := 1 - (10 : ℝ) / 13
  initial_value = increased_value * decrease_factor :=
by
  -- Introduce the local definitions
  have increased_value := initial_value * 1.3
  have decrease_factor := 1 - (10 : ℝ) / 13
  
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_value_decrease_l821_82195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_equation_l821_82100

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Triangle DEF -/
def triangle_DEF : (Point2D × Point2D × Point2D) :=
  ({ x := 1, y := 2 }, { x := 6, y := 7 }, { x := -5, y := 5 })

/-- Image of triangle DEF after reflection -/
def triangle_DEF_image : (Point2D × Point2D × Point2D) :=
  ({ x := 1, y := -4 }, { x := 6, y := -9 }, { x := -5, y := -7 })

/-- Line M is defined by its y-intercept -/
def line_M (y_intercept : ℝ) : Set Point2D :=
  { p : Point2D | p.y = y_intercept }

/-- Reflection of a point about a horizontal line -/
def reflect_point (p : Point2D) (y_intercept : ℝ) : Point2D :=
  { x := p.x, y := 2 * y_intercept - p.y }

/-- Theorem: The line M about which the triangle is reflected has equation y = -1 -/
theorem reflection_line_equation :
  ∃ (y_intercept : ℝ),
    y_intercept = -1 ∧
    (∀ (p : Point2D),
      (p = triangle_DEF.fst ∨ p = (triangle_DEF.snd).fst ∨ p = (triangle_DEF.snd).snd) →
      (reflect_point p y_intercept = triangle_DEF_image.fst ∨
       reflect_point p y_intercept = (triangle_DEF_image.snd).fst ∨
       reflect_point p y_intercept = (triangle_DEF_image.snd).snd)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_line_equation_l821_82100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l821_82173

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ - 2*x

theorem g_range :
  Set.range g = Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l821_82173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6_problem_l821_82178

def average (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / xs.length

def median (xs : List ℕ) : ℚ :=
  if xs.length % 2 = 0 then
    (xs[xs.length / 2 - 1]! + xs[xs.length / 2]!) / 2
  else
    xs[xs.length / 2]!

def first_k_multiples (k : ℕ) (n : ℕ) : List ℕ :=
  List.range k |>.map (fun i => (i + 1) * n)

theorem multiples_of_6_problem (k : ℕ) :
  let a : ℚ := average (first_k_multiples k 6)
  let b : ℚ := median (first_k_multiples 3 12)
  a ^ 2 - b ^ 2 = 0 → k = 7 := by
  sorry

#eval median (first_k_multiples 3 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6_problem_l821_82178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l821_82136

def number_of_students : ℕ := 5
def number_of_adjacent_students : ℕ := 2

theorem photo_arrangements (n : ℕ) (k : ℕ) (h1 : n = number_of_students) (h2 : k = number_of_adjacent_students) :
  (Nat.factorial (n - k + 1)) * (Nat.factorial k) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l821_82136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_sum_lower_bound_l821_82171

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

-- Part 1: Minimum integer value of a
theorem min_a_value (a : ℤ) : (∀ x > 0, f a x ≤ a * x - 1) → a ≥ 2 := by
  sorry

-- Part 2: Lower bound for x₁ + x₂
theorem sum_lower_bound (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h : f (-2) x₁ + f (-2) x₂ + x₁ * x₂ = 0) : 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_sum_lower_bound_l821_82171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_and_triangle_area_l821_82147

-- Define the direction vector of l1
noncomputable def direction_vector_l1 : ℝ × ℝ := (1, 3)

-- Define point A
noncomputable def point_A : ℝ × ℝ := (-2, 3)

-- Define the rotation angle
noncomputable def tan_alpha : ℝ := 1 / 3

-- Define the general equation of l3
def l3_equation (k : ℝ) (x y : ℝ) : Prop :=
  (1 - 3*k)*x + (k + 1)*y - 3*k - 1 = 0

-- State the theorem
theorem line_equations_and_triangle_area :
  -- Part 1: Equation of l1
  ∃ (l1_equation : ℝ → ℝ → Prop),
    (∀ x y, l1_equation x y ↔ 3*x - y + 9 = 0) ∧
  -- Part 2: Equation of l2
  ∃ (l2_equation : ℝ → ℝ → Prop),
    (∀ x y, l2_equation x y ↔ x - y - 1 = 0) ∧
  -- Part 3: Equations of l3 when triangle area is 3
  ∃ (l3_equation1 l3_equation2 : ℝ → ℝ → Prop),
    (∀ x y, l3_equation1 x y ↔ 4*x - y + 11 = 0) ∧
    (∀ x y, l3_equation2 x y ↔ 5*x - 2*y + 16 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_and_triangle_area_l821_82147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_number_equals_sum_of_others_l821_82183

/-- Represents a number composed of a single digit repeated -/
def RepeatedDigitNumber (digit : Nat) (length : Nat) : Nat :=
  digit * (10^length - 1) / 9

/-- The sum of all nine repeated digit numbers -/
def SumOfRepeatedDigitNumbers (length : Nat) : Nat :=
  Finset.sum (Finset.range 9) (fun i => RepeatedDigitNumber (i + 1) length)

/-- Theorem: No single repeated digit number equals the sum of the others -/
theorem no_single_number_equals_sum_of_others (length : Nat) :
  ∀ k : Fin 9, 2 * RepeatedDigitNumber (k.val + 1) length ≠ SumOfRepeatedDigitNumbers length := by
  sorry

#check no_single_number_equals_sum_of_others

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_number_equals_sum_of_others_l821_82183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l821_82189

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ :=
  (6, Real.pi / 3, Real.sqrt 3)

/-- The expected rectangular coordinates -/
noncomputable def expected_rectangular : ℝ × ℝ × ℝ :=
  (3, 3 * Real.sqrt 3, Real.sqrt 3)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = expected_rectangular := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l821_82189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l821_82174

-- Define the ⊙ operation
noncomputable def circle_op (a b : ℝ) : ℝ := Real.sqrt (a * b) + a + b

-- State the theorem
theorem k_range (k : ℝ) :
  (∀ (a b : ℝ), a > 0 → b > 0 → circle_op a b = Real.sqrt (a * b) + a + b) →
  circle_op 1 (k^2) < 3 →
  -1 < k ∧ k < 1 :=
by
  sorry

#check k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l821_82174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_symmetry_l821_82152

/-- The inverse proportion function f(x) = -6/x -/
noncomputable def f (x : ℝ) : ℝ := -6 / x

/-- Theorem: For the inverse proportion function f(x) = -6/x,
    if (-a, b) is on the graph, then (a, -b) is also on the graph -/
theorem inverse_proportion_symmetry (a b : ℝ) (h : a ≠ 0) :
  f (-a) = b → f a = -b := by
  intro h1
  have h2 : f (-a) = -6 / (-a) := rfl
  have h3 : f a = -6 / a := rfl
  rw [h2] at h1
  rw [h3]
  field_simp [h] at *
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_symmetry_l821_82152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_71_l821_82127

/-- Represents the 5x5 grid of numbers -/
def grid : Matrix (Fin 5) (Fin 5) ℕ := λ i j =>
  match i, j with
  | 0, j => j + 1
  | 1, j => 10 - j
  | 2, j => 11 + j
  | 3, j => 20 - j
  | 4, j => 21 + j

/-- Checks if a list of pairs represents valid selections (no repeated rows or columns) -/
def validSelection (selection : List (Fin 5 × Fin 5)) : Prop :=
  selection.length = 5 ∧
  (selection.map Prod.fst).toFinset.card = 5 ∧
  (selection.map Prod.snd).toFinset.card = 5

/-- Calculates the sum of selected numbers -/
def sumSelection (selection : List (Fin 5 × Fin 5)) : ℕ :=
  selection.map (λ (i, j) => grid i j) |>.sum

/-- The main theorem stating that the largest possible sum is 71 -/
theorem largest_sum_is_71 :
  (∃ selection, validSelection selection ∧ sumSelection selection = 71) ∧
  (∀ selection, validSelection selection → sumSelection selection ≤ 71) := by
  sorry

#check largest_sum_is_71

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_71_l821_82127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_catch_theorem_l821_82119

/-- Represents the number of flies a spider catches per hour -/
def spider_efficiency (flies_caught : ℕ) (hours : ℕ) : ℚ :=
  ↑flies_caught / ↑hours

/-- Calculates the number of flies caught given efficiency and time -/
def flies_caught (efficiency : ℚ) (hours : ℕ) : ℕ :=
  (efficiency * ↑hours).floor.toNat

theorem spider_catch_theorem (initial_flies : ℕ) (initial_hours : ℕ) (target_hours : ℕ) :
  initial_flies = 9 →
  initial_hours = 5 →
  target_hours = 30 →
  flies_caught (spider_efficiency initial_flies initial_hours) target_hours = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_catch_theorem_l821_82119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_l821_82187

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The area of a circle -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem cone_base_area (v h : ℝ) (hv : v = 18 * Real.pi) (hh : h = 3) :
  ∃ r : ℝ, cone_volume r h = v ∧ circle_area r = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_area_l821_82187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l821_82169

theorem complex_magnitude (i : ℂ) (h : i * i = -1) : 
  Complex.abs (i * (1 + 3 * i)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_l821_82169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_not_quadratic_l821_82194

-- Define the four functions
def f_A (x : ℝ) := (x - 1)^2
noncomputable def f_B (x : ℝ) := Real.sqrt 2 * x^2 - 1
def f_C (x : ℝ) := 3*x^2 + 2*x - 1
def f_D (x : ℝ) := (x + 1)^2 - x^2

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a*x^2 + b*x + c

-- State the theorem
theorem only_D_not_quadratic :
  is_quadratic f_A ∧ is_quadratic f_B ∧ is_quadratic f_C ∧ ¬is_quadratic f_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_not_quadratic_l821_82194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_make_all_pairwise_relative_prime_l821_82135

/-- A circular arrangement of natural numbers -/
def CircularArrangement := Fin 100 → Nat

/-- The operation of adding the GCD of neighbors to a number at a given index -/
def addGcdOfNeighbors (arr : CircularArrangement) (i : Fin 100) : CircularArrangement :=
  fun j => if j = i
           then arr j + Nat.gcd (arr (j - 1)) (arr (j + 1))
           else arr j

/-- Predicate to check if all numbers in the arrangement are pairwise relatively prime -/
def allPairwiseRelativePrime (arr : CircularArrangement) : Prop :=
  ∀ i j : Fin 100, i ≠ j → Nat.gcd (arr i) (arr j) = 1

/-- The main theorem -/
theorem can_make_all_pairwise_relative_prime
  (initial : CircularArrangement)
  (h : allPairwiseRelativePrime initial) :
  ∃ (sequence : Nat → CircularArrangement),
    (sequence 0 = initial) ∧
    (∀ n, ∃ i, sequence (n + 1) = addGcdOfNeighbors (sequence n) i) ∧
    (∃ N, allPairwiseRelativePrime (sequence N)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_make_all_pairwise_relative_prime_l821_82135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_point_theorem_l821_82153

/-- The complex number that divides the line segment in the ratio 2:1 -/
noncomputable def dividing_point (z₁ z₂ : ℂ) : ℂ :=
  (2 * z₂ + z₁) / 3

theorem dividing_point_theorem :
  let z₁ : ℂ := -7 + 5*Complex.I
  let z₂ : ℂ := 5 - 3*Complex.I
  dividing_point z₁ z₂ = 1 - (1/3)*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_point_theorem_l821_82153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_astronaut_crew_selection_l821_82137

theorem astronaut_crew_selection (n : ℕ) (h : n ≥ 11000) :
  ∀ (S : Finset (Finset ℕ)),
    (∀ (A : Finset ℕ), A.card = 4 → A ⊆ Finset.range n →
      ∃ (B : Finset ℕ), B ⊆ A ∧ B.card = 3 ∧ B ∈ S) →
    ∃ (C : Finset ℕ), C.card = 5 ∧ C ⊆ Finset.range n ∧
      ∀ (D : Finset ℕ), D ⊆ C → D.card = 3 → D ∈ S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_astronaut_crew_selection_l821_82137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l821_82155

variable (x : ℚ)

def f1 (x : ℚ) : ℚ := 1 / (5 * x)
def f2 (x : ℚ) : ℚ := 1 / (10 * x)
def f3 (x : ℚ) : ℚ := 1 / (15 * x)

def lcm_result (x : ℚ) : ℚ := 1 / (30 * x)

theorem lcm_of_fractions (x : ℚ) (hx : x ≠ 0) :
  (∀ y : ℚ, (∃ k : ℚ, y = k * f1 x) ∧ (∃ k : ℚ, y = k * f2 x) ∧ (∃ k : ℚ, y = k * f3 x) 
    → (∃ k : ℚ, y = k * lcm_result x)) ∧
  (∃ k : ℚ, f1 x = k * lcm_result x) ∧ 
  (∃ k : ℚ, f2 x = k * lcm_result x) ∧ 
  (∃ k : ℚ, f3 x = k * lcm_result x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l821_82155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circles_area_ratio_l821_82198

/-- Represents a circle in the construction -/
structure Circle where
  radius : ℝ

/-- Represents the construction of nested circles in an equilateral triangle -/
structure NestedCircles where
  largestCircle : Circle
  smallerCircles : List Circle

/-- The ratio of radii between consecutive circles -/
def radiusRatio : ℝ := 3

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Calculates the total area of a list of circles -/
noncomputable def totalArea (circles : List Circle) : ℝ :=
  circles.map circleArea |>.sum

/-- The main theorem to be proved -/
theorem nested_circles_area_ratio (nc : NestedCircles) :
  circleArea nc.largestCircle / totalArea nc.smallerCircles = 8 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circles_area_ratio_l821_82198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_continuous_iff_b_eq_5_l821_82188

-- Define the function g(x)
noncomputable def g (b : ℝ) (x : ℝ) : ℝ :=
  if x > 4 then x^2 + 1 else 3*x + b

-- State the theorem
theorem g_continuous_iff_b_eq_5 :
  ∀ b : ℝ, Continuous (g b) ↔ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_continuous_iff_b_eq_5_l821_82188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l821_82164

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < -1 ∨ x ≥ 3}

-- Define set B
def B : Set ℝ := {x | 2 * x - 1 ≤ 3}

theorem set_operations :
  (A ∪ B = {x : ℝ | x ≤ 2 ∨ x ≥ 3}) ∧
  (A ∩ (U \ B) = {x : ℝ | x ≥ 3}) ∧
  ((U \ A) ∪ (U \ B) = {x : ℝ | x ≥ -1}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l821_82164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l821_82181

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest with half-yearly compounding -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * ((1 + rate / 200) ^ 2 - 1)

/-- Theorem stating that if the difference between compound and simple interest
    is 3.50 for a principal of 1400 and time of 1 year, then the annual interest rate is 10% -/
theorem interest_rate_is_ten_percent
  (h : compoundInterest 1400 R - simpleInterest 1400 R 1 = 3.50) :
  R = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l821_82181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_inequality_l821_82161

def isNonIncreasing (s : List ℝ) : Prop :=
  ∀ i j, i < j → i < s.length → j < s.length → s[i]! ≥ s[j]!

def sumSquaredDiff (x y : List ℝ) : ℝ :=
  (List.zip x y).foldl (λ acc (a, b) => acc + (a - b)^2) 0

theorem rearrangement_inequality 
  (x y : List ℝ) (hx : isNonIncreasing x) (hy : isNonIncreasing y) 
  (hlen : x.length = y.length) :
  ∀ (z : List ℝ), z.isPerm y → sumSquaredDiff x y ≤ sumSquaredDiff x z := by
  sorry

#check rearrangement_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrangement_inequality_l821_82161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l821_82193

/-- Represents the dimensions of the rectangular shed -/
structure Shed :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the tethering arrangements for the goat -/
inductive Arrangement
  | Middle
  | Corner

/-- Calculates the roaming area for a given arrangement -/
noncomputable def roamingArea (s : Shed) (rope : ℝ) (a : Arrangement) : ℝ :=
  match a with
  | Arrangement.Middle => (1/2) * Real.pi * rope^2
  | Arrangement.Corner => (3/4) * Real.pi * rope^2 + (1/4) * Real.pi * (s.width/2)^2

theorem roaming_area_difference (s : Shed) (rope : ℝ) :
  s.length = 20 →
  s.width = 10 →
  rope = 10 →
  roamingArea s rope Arrangement.Corner - roamingArea s rope Arrangement.Middle = 31.25 * Real.pi := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l821_82193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_max_overtakes_max_overtakes_achievable_l821_82191

/-- Represents a team in the relay race -/
structure Team where
  runners : Fin 20 → ℝ → ℝ  -- Each runner's position as a function of time

/-- Represents the relay race -/
structure RelayRace where
  team1 : Team
  team2 : Team

/-- Counts the number of overtakes in a relay race -/
noncomputable def countOvertakes (race : RelayRace) : ℕ :=
  sorry  -- Implementation details omitted

/-- The maximum number of overtakes possible in the relay race -/
def maxOvertakes : ℕ := 38

/-- Theorem stating the maximum number of overtakes in the relay race -/
theorem relay_race_max_overtakes :
  ∀ (race : RelayRace),
    (∀ (team : Team) (i : Fin 20), Continuous (team.runners i)) →
    (∀ (team : Team) (i : Fin 19), 
      ∃ (t : ℝ), team.runners i t = team.runners (i.succ) t) →
    countOvertakes race ≤ maxOvertakes := by
  sorry

/-- Theorem stating that the maximum number of overtakes is achievable -/
theorem max_overtakes_achievable :
  ∃ (race : RelayRace),
    (∀ (team : Team) (i : Fin 20), Continuous (team.runners i)) ∧
    (∀ (team : Team) (i : Fin 19), 
      ∃ (t : ℝ), team.runners i t = team.runners (i.succ) t) ∧
    countOvertakes race = maxOvertakes := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_max_overtakes_max_overtakes_achievable_l821_82191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_third_l821_82196

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

-- State the theorem
theorem derivative_f_at_one_third :
  deriv f (1/3) = -3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_third_l821_82196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_tournament_l821_82167

theorem table_tennis_tournament (n : ℕ) : 
  (n ≥ 3) →
  (∃ r : ℕ, r ≤ 3 ∧ (n - 3) * (n - 4) / 2 + (6 - r) = 50) →
  (∃! r : ℕ, r ≤ 3 ∧ (n - 3) * (n - 4) / 2 + (6 - r) = 50 ∧ r = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_tournament_l821_82167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l821_82157

theorem problem_solution :
  (((3 - Real.pi) ^ 0) - (2 ^ 2) + ((1 / 2) ^ (-2 : ℤ))) = 1 ∧
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → (((a * b^2)^2 - 2 * a * b^4) / (a * b^4)) = a - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l821_82157
