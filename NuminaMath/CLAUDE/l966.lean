import Mathlib

namespace NUMINAMATH_CALUDE_common_chord_of_circles_l966_96616

/-- Given two circles C₁ and C₂, prove that their common chord lies on the line 3x - 4y + 6 = 0 -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 + 2*x - 6*y + 1 = 0) ∧ (x^2 + y^2 - 4*x + 2*y - 11 = 0) →
  (3*x - 4*y + 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l966_96616


namespace NUMINAMATH_CALUDE_wine_purchase_additional_cost_l966_96668

/-- Represents the price changes and conditions for wine purchases over three months --/
structure WinePrices where
  initial_price : ℝ
  tariff_increase1 : ℝ
  tariff_increase2 : ℝ
  exchange_rate_change1 : ℝ
  exchange_rate_change2 : ℝ
  bulk_discount : ℝ
  bottles_per_month : ℕ

/-- Calculates the total additional cost of wine purchases over three months --/
def calculate_additional_cost (prices : WinePrices) : ℝ :=
  let month1_price := prices.initial_price * (1 + prices.exchange_rate_change1)
  let month2_price := prices.initial_price * (1 + prices.tariff_increase1) * (1 - prices.bulk_discount)
  let month3_price := prices.initial_price * (1 + prices.tariff_increase1 + prices.tariff_increase2) * (1 - prices.exchange_rate_change2)
  let total_cost := (month1_price + month2_price + month3_price) * prices.bottles_per_month
  let initial_total := prices.initial_price * prices.bottles_per_month * 3
  total_cost - initial_total

/-- Theorem stating that the additional cost of wine purchases over three months is $42.20 --/
theorem wine_purchase_additional_cost :
  let prices : WinePrices := {
    initial_price := 20,
    tariff_increase1 := 0.25,
    tariff_increase2 := 0.10,
    exchange_rate_change1 := 0.05,
    exchange_rate_change2 := 0.03,
    bulk_discount := 0.15,
    bottles_per_month := 5
  }
  calculate_additional_cost prices = 42.20 := by
  sorry


end NUMINAMATH_CALUDE_wine_purchase_additional_cost_l966_96668


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l966_96676

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (2 - Complex.I) * (a + 2 * Complex.I) = b * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l966_96676


namespace NUMINAMATH_CALUDE_toy_poodle_height_l966_96655

/-- Proves that the height of a toy poodle is 14 inches given the heights of standard and miniature poodles -/
theorem toy_poodle_height 
  (standard_height : ℕ) 
  (standard_miniature_diff : ℕ) 
  (miniature_toy_diff : ℕ) 
  (h1 : standard_height = 28)
  (h2 : standard_miniature_diff = 8)
  (h3 : miniature_toy_diff = 6) : 
  standard_height - standard_miniature_diff - miniature_toy_diff = 14 := by
  sorry

#check toy_poodle_height

end NUMINAMATH_CALUDE_toy_poodle_height_l966_96655


namespace NUMINAMATH_CALUDE_orange_cost_l966_96648

theorem orange_cost (cost_three_dozen : ℝ) (h : cost_three_dozen = 28.20) :
  let cost_per_dozen : ℝ := cost_three_dozen / 3
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 47.00 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_l966_96648


namespace NUMINAMATH_CALUDE_school_student_count_miyoung_school_students_l966_96691

theorem school_student_count (grades classes_per_grade : ℕ) 
  (rank_from_front rank_from_back : ℕ) : ℕ :=
  let students_per_class := rank_from_front + rank_from_back - 1
  let students_per_grade := classes_per_grade * students_per_class
  let total_students := grades * students_per_grade
  total_students

theorem miyoung_school_students : 
  school_student_count 3 12 12 12 = 828 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_miyoung_school_students_l966_96691


namespace NUMINAMATH_CALUDE_fraction_simplification_l966_96654

theorem fraction_simplification :
  (3 / 7 - 2 / 9) / (5 / 12 + 1 / 4) = 13 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l966_96654


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l966_96682

theorem problem_1 (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := by
sorry

theorem problem_2 (α β : Real)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = Real.sqrt 5 / 5)
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  β = π / 8 := by
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l966_96682


namespace NUMINAMATH_CALUDE_sqrt_225_range_l966_96652

theorem sqrt_225_range : 15 < Real.sqrt 225 ∧ Real.sqrt 225 < 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_range_l966_96652


namespace NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l966_96653

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 →
  x₂^3 - 10*x₂ + 11 = 0 →
  x₃^3 - 10*x₃ + 11 = 0 →
  -5 < x₁ ∧ x₁ < 5 →
  -5 < x₂ ∧ x₂ < 5 →
  -5 < x₃ ∧ x₃ < 5 →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l966_96653


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l966_96693

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![-2, 3], ![1, -5]]) : 
  (A^2)⁻¹ = ![![7, -21], ![-7, 28]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l966_96693


namespace NUMINAMATH_CALUDE_num_ways_to_form_triangles_l966_96636

/-- The number of distinguishable balls -/
def num_balls : ℕ := 6

/-- The number of distinguishable sticks -/
def num_sticks : ℕ := 6

/-- The number of balls required to form a triangle -/
def balls_per_triangle : ℕ := 3

/-- The number of sticks required to form a triangle -/
def sticks_per_triangle : ℕ := 3

/-- The number of triangles to be formed -/
def num_triangles : ℕ := 2

/-- The number of symmetries for each triangle (rotations and reflections) -/
def symmetries_per_triangle : ℕ := 6

/-- Theorem stating the number of ways to form two disjoint non-interlocking triangles -/
theorem num_ways_to_form_triangles : 
  (Nat.choose num_balls balls_per_triangle * Nat.factorial num_sticks) / 
  (Nat.factorial num_triangles * symmetries_per_triangle ^ num_triangles) = 200 :=
sorry

end NUMINAMATH_CALUDE_num_ways_to_form_triangles_l966_96636


namespace NUMINAMATH_CALUDE_cube_labeling_theorem_l966_96628

/-- A labeling of a cube's edges -/
def CubeLabeling := Fin 12 → Fin 13

/-- The sum of labels at a vertex given a labeling -/
def vertexSum (l : CubeLabeling) (v : Fin 8) : ℕ := sorry

/-- Predicate for a valid labeling using numbers 1 to 12 -/
def validLabeling12 (l : CubeLabeling) : Prop :=
  (∀ i : Fin 12, l i < 13) ∧ (∀ i j : Fin 12, i ≠ j → l i ≠ l j)

/-- Predicate for a valid labeling using numbers 1 to 13 with one unused -/
def validLabeling13 (l : CubeLabeling) : Prop :=
  (∀ i : Fin 12, l i > 0) ∧ (∀ i j : Fin 12, i ≠ j → l i ≠ l j)

theorem cube_labeling_theorem :
  (∀ l : CubeLabeling, validLabeling12 l →
    ∃ v1 v2 : Fin 8, v1 ≠ v2 ∧ vertexSum l v1 ≠ vertexSum l v2) ∧
  (∃ l : CubeLabeling, validLabeling13 l ∧
    ∀ v1 v2 : Fin 8, vertexSum l v1 = vertexSum l v2) :=
by sorry

end NUMINAMATH_CALUDE_cube_labeling_theorem_l966_96628


namespace NUMINAMATH_CALUDE_base_7_2534_equals_956_l966_96600

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_2534_equals_956 :
  base_7_to_10 [4, 3, 5, 2] = 956 := by
  sorry

end NUMINAMATH_CALUDE_base_7_2534_equals_956_l966_96600


namespace NUMINAMATH_CALUDE_functional_equation_solution_l966_96694

open Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the property that f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((floor x : ℝ) * y) = f x * (floor (f y) : ℝ)

-- Theorem statement
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_equation f →
    (∀ x : ℝ, f x = 0) ∨ 
    (∃ C : ℝ, 1 ≤ C ∧ C < 2 ∧ ∀ x : ℝ, f x = C) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l966_96694


namespace NUMINAMATH_CALUDE_inequality_solutions_count_l966_96601

theorem inequality_solutions_count : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 5*x^2 + 19*x + 12 ≤ 20) ∧ Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_count_l966_96601


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l966_96632

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 8 ∧ x * y = 12) →
  p + q = 60 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l966_96632


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l966_96685

/-- Given a triangle ABC with side lengths a and b, and angles A, B, and C,
    prove that C > B > A when a = 1, b = √3, A = 30°, and B is acute. -/
theorem triangle_angle_inequality (a b : ℝ) (A B C : ℝ) : 
  a = 1 → 
  b = Real.sqrt 3 → 
  A = π / 6 → 
  0 < B ∧ B < π / 2 → 
  a * Real.sin B = b * Real.sin A →
  A + B + C = π →
  C > B ∧ B > A := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l966_96685


namespace NUMINAMATH_CALUDE_linear_functions_product_sign_l966_96641

theorem linear_functions_product_sign (a b c d : ℝ) :
  b < 0 →
  d < 0 →
  ((a > 0 ∧ c < 0) ∨ (a < 0 ∧ c > 0)) →
  a * b * c * d < 0 := by
sorry

end NUMINAMATH_CALUDE_linear_functions_product_sign_l966_96641


namespace NUMINAMATH_CALUDE_polynomial_simplification_l966_96621

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l966_96621


namespace NUMINAMATH_CALUDE_min_students_satisfying_conditions_l966_96604

/-- Represents the number of students in a classroom. -/
structure Classroom where
  boys : ℕ
  girls : ℕ

/-- Checks if the given classroom satisfies all conditions. -/
def satisfiesConditions (c : Classroom) : Prop :=
  ∃ (passed_boys passed_girls : ℕ),
    passed_boys = passed_girls ∧
    passed_boys = (3 * c.boys) / 5 ∧
    passed_girls = (2 * c.girls) / 3 ∧
    (c.boys + c.girls) % 10 = 0

/-- The theorem stating the minimum number of students satisfying all conditions. -/
theorem min_students_satisfying_conditions :
  ∀ c : Classroom, satisfiesConditions c →
    ∀ c' : Classroom, satisfiesConditions c' →
      c.boys + c.girls ≤ c'.boys + c'.girls →
        c.boys + c.girls = 38 := by
  sorry

#check min_students_satisfying_conditions

end NUMINAMATH_CALUDE_min_students_satisfying_conditions_l966_96604


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l966_96640

theorem p_sufficient_not_necessary_for_q :
  (∃ x : ℝ, x = 2 ∧ x^2 ≠ 4) ∨
  (∃ x : ℝ, x^2 = 4 ∧ x ≠ 2) ∨
  (∀ x : ℝ, x = 2 → x^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l966_96640


namespace NUMINAMATH_CALUDE_table_formula_proof_l966_96657

def f (x : ℤ) : ℤ := x^2 - 4*x + 1

theorem table_formula_proof :
  (f 1 = -2) ∧ 
  (f 2 = 0) ∧ 
  (f 3 = 4) ∧ 
  (f 4 = 10) ∧ 
  (f 5 = 18) := by
  sorry

end NUMINAMATH_CALUDE_table_formula_proof_l966_96657


namespace NUMINAMATH_CALUDE_max_label_outcomes_l966_96673

/-- The number of balls in the box -/
def num_balls : ℕ := 3

/-- The number of times a ball is drawn -/
def num_draws : ℕ := 3

/-- The total number of possible outcomes when drawing num_draws times from num_balls balls -/
def total_outcomes : ℕ := num_balls ^ num_draws

/-- The number of outcomes that don't include the maximum label -/
def outcomes_without_max : ℕ := 8

/-- Theorem: The number of ways to draw a maximum label of 3 when drawing 3 balls 
    (with replacement) from a box containing balls labeled 1, 2, and 3 is equal to 19 -/
theorem max_label_outcomes : 
  total_outcomes - outcomes_without_max = 19 := by sorry

end NUMINAMATH_CALUDE_max_label_outcomes_l966_96673


namespace NUMINAMATH_CALUDE_max_non_zero_numbers_eq_sum_binary_digits_l966_96688

/-- The sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- The game state -/
structure GameState where
  numbers : List ℕ

/-- The game move -/
inductive Move
  | Sum : ℕ → ℕ → Move
  | Diff : ℕ → ℕ → Move

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState := sorry

/-- Check if the game is over -/
def isGameOver (state : GameState) : Bool := sorry

/-- The maximum number of non-zero numbers at the end of the game -/
def maxNonZeroNumbers (initialOnes : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem max_non_zero_numbers_eq_sum_binary_digits :
  maxNonZeroNumbers 2020 = sumBinaryDigits 2020 := by sorry

end NUMINAMATH_CALUDE_max_non_zero_numbers_eq_sum_binary_digits_l966_96688


namespace NUMINAMATH_CALUDE_find_B_l966_96667

theorem find_B (A B : ℚ) : (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / (4 * A) ∧ 1 / (4 * A) = 1 / B → B = 32 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l966_96667


namespace NUMINAMATH_CALUDE_certain_number_problem_l966_96633

theorem certain_number_problem (x : ℝ) : 300 + (x * 8) = 340 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l966_96633


namespace NUMINAMATH_CALUDE_lcm_5_6_8_9_l966_96610

theorem lcm_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_8_9_l966_96610


namespace NUMINAMATH_CALUDE_inequality_proof_l966_96617

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x*y + y*z + z*x = 1) : 
  x*y*z*(x + y)*(y + z)*(z + x) ≥ (1 - x^2)*(1 - y^2)*(1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l966_96617


namespace NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l966_96661

def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem x_zero_necessary_not_sufficient :
  ∃ (x : ℝ), x ≠ 0 ∧ (a + b x) • (b x) = 0 ∧
  ∀ (y : ℝ), (a + b y) • (b y) = 0 → y = 0 ∨ y = -1 :=
by sorry

end NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l966_96661


namespace NUMINAMATH_CALUDE_dubblefud_red_balls_l966_96696

/-- The game of dubblefud with red, blue, and green balls -/
def dubblefud (r b g : ℕ) : Prop :=
  2^r * 4^b * 5^g = 16000 ∧ b = g

theorem dubblefud_red_balls :
  ∃ (r b g : ℕ), dubblefud r b g ∧ r = 6 :=
sorry

end NUMINAMATH_CALUDE_dubblefud_red_balls_l966_96696


namespace NUMINAMATH_CALUDE_square_of_binomial_l966_96674

theorem square_of_binomial (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 9*x^2 - 18*x + a = (b*x + c)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l966_96674


namespace NUMINAMATH_CALUDE_log_inequality_l966_96684

theorem log_inequality (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.log x / Real.log 3) ∧ f a > f 2) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l966_96684


namespace NUMINAMATH_CALUDE_paula_candy_distribution_l966_96606

theorem paula_candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (num_friends : ℕ) :
  initial_candies = 20 →
  additional_candies = 4 →
  num_friends = 6 →
  (initial_candies + additional_candies) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_paula_candy_distribution_l966_96606


namespace NUMINAMATH_CALUDE_seeds_sown_l966_96627

/-- Given a farmer who started with 8.75 buckets of seeds and ended with 6 buckets,
    prove that the number of buckets sown is 2.75. -/
theorem seeds_sown (initial : ℝ) (remaining : ℝ) (h1 : initial = 8.75) (h2 : remaining = 6) :
  initial - remaining = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_seeds_sown_l966_96627


namespace NUMINAMATH_CALUDE_no_additional_painters_needed_l966_96650

/-- Represents the painting job scenario -/
structure PaintingJob where
  initialPainters : ℕ
  initialDays : ℚ
  initialRate : ℚ
  newDays : ℕ
  newRate : ℚ

/-- Calculates the total work required for the job -/
def totalWork (job : PaintingJob) : ℚ :=
  job.initialPainters * job.initialDays * job.initialRate

/-- Calculates the number of painters needed for the new conditions -/
def paintersNeeded (job : PaintingJob) : ℚ :=
  (totalWork job) / (job.newDays * job.newRate)

/-- Theorem stating that no additional painters are needed -/
theorem no_additional_painters_needed (job : PaintingJob) 
  (h1 : job.initialPainters = 6)
  (h2 : job.initialDays = 5/2)
  (h3 : job.initialRate = 2)
  (h4 : job.newDays = 2)
  (h5 : job.newRate = 5/2) :
  paintersNeeded job = job.initialPainters :=
by sorry

#check no_additional_painters_needed

end NUMINAMATH_CALUDE_no_additional_painters_needed_l966_96650


namespace NUMINAMATH_CALUDE_cab_base_price_l966_96644

/-- Represents the base price of a cab ride -/
def base_price : ℝ := sorry

/-- Represents the per-mile charge of a cab ride -/
def per_mile_charge : ℝ := 4

/-- Represents the total distance traveled in miles -/
def distance : ℝ := 5

/-- Represents the total cost of the cab ride -/
def total_cost : ℝ := 23

/-- Theorem stating that the base price of the cab ride is $3 -/
theorem cab_base_price : base_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_cab_base_price_l966_96644


namespace NUMINAMATH_CALUDE_quadratic_range_iff_a_values_l966_96680

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2*a + 4

/-- The theorem stating the relationship between the range of f and the values of a -/
theorem quadratic_range_iff_a_values (a : ℝ) :
  (∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 1) ↔ a = -1 ∨ a = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_iff_a_values_l966_96680


namespace NUMINAMATH_CALUDE_sum_1984_consecutive_not_square_l966_96642

theorem sum_1984_consecutive_not_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (992 * (2 * n + 1985) : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_1984_consecutive_not_square_l966_96642


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_lines_l966_96670

/-- Represents a line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle --/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Check if three lines are coplanar and equidistant --/
def are_coplanar_equidistant (l1 l2 l3 : Line) : Prop :=
  l1.slope = l2.slope ∧ l2.slope = l3.slope ∧
  |l2.intercept - l1.intercept| = |l3.intercept - l2.intercept|

/-- Check if a triangle is equilateral --/
def is_equilateral (t : Triangle) : Prop :=
  let d1 := ((t.b.x - t.a.x)^2 + (t.b.y - t.a.y)^2).sqrt
  let d2 := ((t.c.x - t.b.x)^2 + (t.c.y - t.b.y)^2).sqrt
  let d3 := ((t.a.x - t.c.x)^2 + (t.a.y - t.c.y)^2).sqrt
  d1 = d2 ∧ d2 = d3

/-- Check if a point lies on a line --/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem: Given three coplanar and equidistant lines, 
    it is possible to construct an equilateral triangle 
    with its vertices lying on these lines --/
theorem equilateral_triangle_on_lines 
  (l1 l2 l3 : Line) 
  (h : are_coplanar_equidistant l1 l2 l3) :
  ∃ (t : Triangle), 
    is_equilateral t ∧ 
    point_on_line t.a l1 ∧ 
    point_on_line t.b l2 ∧ 
    point_on_line t.c l3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_lines_l966_96670


namespace NUMINAMATH_CALUDE_divisible_by_6_up_to_88_characterization_l966_96638

def divisible_by_6_up_to_88 : Set ℕ :=
  {n : ℕ | 1 < n ∧ n ≤ 88 ∧ n % 6 = 0}

theorem divisible_by_6_up_to_88_characterization :
  divisible_by_6_up_to_88 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84} := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_6_up_to_88_characterization_l966_96638


namespace NUMINAMATH_CALUDE_power_of_product_l966_96681

theorem power_of_product (a b : ℝ) (m : ℕ+) : (a * b) ^ (m : ℕ) = a ^ (m : ℕ) * b ^ (m : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l966_96681


namespace NUMINAMATH_CALUDE_school_committee_formation_l966_96603

theorem school_committee_formation (n_children : ℕ) (n_teachers : ℕ) (committee_size : ℕ) :
  n_children = 12 →
  n_teachers = 3 →
  committee_size = 9 →
  (Nat.choose (n_children + n_teachers) committee_size) - (Nat.choose n_children committee_size) = 4785 :=
by sorry

end NUMINAMATH_CALUDE_school_committee_formation_l966_96603


namespace NUMINAMATH_CALUDE_book_difference_l966_96683

/- Define the number of books for each category -/
def total_books : ℕ := 220
def hardcover_nonfiction : ℕ := 40

/- Define the properties of the book categories -/
def book_categories (paperback_fiction paperback_nonfiction : ℕ) : Prop :=
  paperback_fiction + paperback_nonfiction + hardcover_nonfiction = total_books ∧
  paperback_nonfiction > hardcover_nonfiction ∧
  paperback_fiction = 2 * paperback_nonfiction

/- Theorem statement -/
theorem book_difference :
  ∃ (paperback_fiction paperback_nonfiction : ℕ),
    book_categories paperback_fiction paperback_nonfiction ∧
    paperback_nonfiction - hardcover_nonfiction = 20 :=
by sorry

end NUMINAMATH_CALUDE_book_difference_l966_96683


namespace NUMINAMATH_CALUDE_congested_sections_probability_l966_96629

/-- The probability of selecting exactly 4 congested sections out of 10 randomly selected sections,
    given that there are 7 congested sections out of 16 total sections. -/
theorem congested_sections_probability :
  let total_sections : ℕ := 16
  let congested_sections : ℕ := 7
  let selected_sections : ℕ := 10
  let target_congested : ℕ := 4
  
  (Nat.choose congested_sections target_congested *
   Nat.choose (total_sections - congested_sections) (selected_sections - target_congested)) /
  Nat.choose total_sections selected_sections =
  (Nat.choose congested_sections target_congested *
   Nat.choose (total_sections - congested_sections) (selected_sections - target_congested)) /
  Nat.choose total_sections selected_sections :=
by
  sorry

end NUMINAMATH_CALUDE_congested_sections_probability_l966_96629


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l966_96649

def xray_cost : ℝ := 250
def mri_cost : ℝ := 3 * xray_cost
def total_cost : ℝ := xray_cost + mri_cost
def mike_payment : ℝ := 200
def insurance_coverage : ℝ := total_cost - mike_payment

theorem insurance_coverage_percentage : (insurance_coverage / total_cost) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l966_96649


namespace NUMINAMATH_CALUDE_geometric_progression_condition_l966_96620

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_progression_condition
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a (n + 2) = a n * a (n + 1) - c)
  (c : ℝ) :
  is_geometric_progression a ↔ (a 1 = a 2 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_condition_l966_96620


namespace NUMINAMATH_CALUDE_remainder_proof_l966_96631

theorem remainder_proof (x y r : ℤ) : 
  x > 0 →
  x = 7 * y + r →
  0 ≤ r →
  r < 7 →
  2 * x = 18 * y + 2 →
  11 * y - x = 1 →
  r = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_proof_l966_96631


namespace NUMINAMATH_CALUDE_multiple_births_quintuplets_l966_96669

theorem multiple_births_quintuplets (total_babies : ℕ) 
  (triplets_to_quintuplets : ℕ → ℕ) 
  (twins_to_triplets : ℕ → ℕ) 
  (quadruplets_to_quintuplets : ℕ → ℕ) 
  (h1 : total_babies = 1540)
  (h2 : ∀ q, triplets_to_quintuplets q = 6 * q)
  (h3 : ∀ t, twins_to_triplets t = 2 * t)
  (h4 : ∀ q, quadruplets_to_quintuplets q = 3 * q)
  (h5 : ∀ q, 2 * (twins_to_triplets (triplets_to_quintuplets q)) + 
             3 * (triplets_to_quintuplets q) + 
             4 * (quadruplets_to_quintuplets q) + 
             5 * q = total_babies) : 
  ∃ q : ℚ, q = 7700 / 59 ∧ 5 * q = (quintuplets_babies : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_multiple_births_quintuplets_l966_96669


namespace NUMINAMATH_CALUDE_square_difference_equality_l966_96625

theorem square_difference_equality : 1005^2 - 995^2 - 1007^2 + 993^2 = -8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l966_96625


namespace NUMINAMATH_CALUDE_cube_tower_surface_area_l966_96609

/-- Represents a cube with its side length -/
structure Cube where
  side : ℕ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.side ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℕ := 6 * c.side ^ 2

/-- Represents the tower of cubes -/
def CubeTower : List Cube := [
  { side := 8 },
  { side := 7 },
  { side := 6 },
  { side := 5 },
  { side := 4 },
  { side := 3 },
  { side := 2 },
  { side := 1 }
]

/-- Calculates the visible surface area of a cube in the tower -/
def visibleSurfaceArea (c : Cube) (isBottom : Bool) : ℕ :=
  if isBottom then
    5 * c.side ^ 2  -- 5 visible faces for bottom cube
  else if c.side = 1 then
    5 * c.side ^ 2  -- 5 visible faces for top cube
  else
    4 * c.side ^ 2  -- 4 visible faces for other cubes (3 full + 2 partial = 4)

/-- Calculates the total visible surface area of the cube tower -/
def totalVisibleSurfaceArea (tower : List Cube) : ℕ :=
  let rec aux (cubes : List Cube) (acc : ℕ) (isFirst : Bool) : ℕ :=
    match cubes with
    | [] => acc
    | c :: rest => aux rest (acc + visibleSurfaceArea c isFirst) false
  aux tower 0 true

/-- The main theorem stating that the total visible surface area of the cube tower is 945 -/
theorem cube_tower_surface_area :
  totalVisibleSurfaceArea CubeTower = 945 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_cube_tower_surface_area_l966_96609


namespace NUMINAMATH_CALUDE_exists_x0_abs_fx0_plus_a_nonneg_l966_96643

theorem exists_x0_abs_fx0_plus_a_nonneg (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |((x₀^2 : ℝ) + a * x₀ + b) + a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x0_abs_fx0_plus_a_nonneg_l966_96643


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l966_96692

/-- Given 6 numbers with specified averages, prove the average of the remaining 2 numbers -/
theorem average_of_remaining_numbers
  (total_average : Real)
  (first_pair_average : Real)
  (second_pair_average : Real)
  (h1 : total_average = 3.95)
  (h2 : first_pair_average = 3.8)
  (h3 : second_pair_average = 3.85) :
  (6 * total_average - 2 * first_pair_average - 2 * second_pair_average) / 2 = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l966_96692


namespace NUMINAMATH_CALUDE_siblings_comparison_l966_96663

/-- Given information about siblings of Masud, Janet, Carlos, and Stella, prove that Janet has 16 fewer siblings than Carlos and Stella combined. -/
theorem siblings_comparison (masud janet carlos stella : ℕ) : 
  masud = 40 →
  janet = 4 * masud - 60 →
  carlos = 3 * masud / 4 + 12 →
  stella = 2 * (carlos - 12) - 8 →
  janet = 100 →
  carlos = 64 →
  stella = 52 →
  janet = carlos + stella - 16 := by
  sorry


end NUMINAMATH_CALUDE_siblings_comparison_l966_96663


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l966_96689

/-- Given vectors a and b, prove that |a + 2b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (Real.cos (5 * π / 180), Real.sin (5 * π / 180)) →
  b = (Real.cos (65 * π / 180), Real.sin (65 * π / 180)) →
  ‖a + 2 • b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l966_96689


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l966_96613

theorem geometric_sequence_sum (a b c : ℝ) : 
  (1 < a ∧ a < b ∧ b < c ∧ c < 16) →
  (∃ q : ℝ, q ≠ 0 ∧ a = 1 * q ∧ b = a * q ∧ c = b * q ∧ 16 = c * q) →
  (a + c = 10 ∨ a + c = -10) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l966_96613


namespace NUMINAMATH_CALUDE_extra_money_spent_theorem_l966_96665

/-- Represents the price of radishes and pork ribs last month and this month --/
structure PriceData where
  radish_last : ℝ
  pork_last : ℝ
  radish_this : ℝ
  pork_this : ℝ

/-- Calculates the extra money spent given the price data and quantities --/
def extra_money_spent (p : PriceData) (radish_qty : ℝ) (pork_qty : ℝ) : ℝ :=
  radish_qty * (p.radish_this - p.radish_last) + pork_qty * (p.pork_this - p.pork_last)

/-- Theorem stating the extra money spent on radishes and pork ribs --/
theorem extra_money_spent_theorem (a : ℝ) :
  let p : PriceData := {
    radish_last := a,
    pork_last := 7 * a + 2,
    radish_this := 1.25 * a,
    pork_this := 1.2 * (7 * a + 2)
  }
  extra_money_spent p 3 2 = 3.55 * a + 0.8 := by
  sorry

#check extra_money_spent_theorem

end NUMINAMATH_CALUDE_extra_money_spent_theorem_l966_96665


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l966_96666

/-- Given four squares with areas 256, 64, 225, and 49, prove that the area of the triangle formed by three of these squares is 60 -/
theorem triangle_area_from_squares (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ = 256) (h₂ : s₂ = 64) (h₃ : s₃ = 225) (h₄ : s₄ = 49) : 
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a^2 = s₃ ∧ b^2 = s₂ ∧ c^2 = s₁ ∧ (1/2 * a * b = 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l966_96666


namespace NUMINAMATH_CALUDE_geometric_progression_values_l966_96698

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (3*p + 1) = |p - 3| * r ∧ (9*p + 10) = (3*p + 1) * r) ↔ 
  (p = -1 ∨ p = 29/18) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l966_96698


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l966_96699

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (0 < a ∧ a < b → (1 / a > 1 / b)) ∧
  ¬(∀ a b : ℝ, (1 / a > 1 / b) → (0 < a ∧ a < b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l966_96699


namespace NUMINAMATH_CALUDE_cookie_difference_l966_96602

/-- Proves that Cristian had 50 more black cookies than white cookies initially -/
theorem cookie_difference (black_cookies white_cookies : ℕ) : 
  white_cookies = 80 →
  black_cookies > white_cookies →
  black_cookies / 2 + white_cookies / 4 = 85 →
  black_cookies - white_cookies = 50 :=
by
  sorry

#check cookie_difference

end NUMINAMATH_CALUDE_cookie_difference_l966_96602


namespace NUMINAMATH_CALUDE_expression_simplification_l966_96635

theorem expression_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 3*x^3 =
  -x^3 - x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l966_96635


namespace NUMINAMATH_CALUDE_uphill_speed_calculation_l966_96678

theorem uphill_speed_calculation (uphill_distance : ℝ) (downhill_distance : ℝ) 
  (downhill_speed : ℝ) (average_speed : ℝ) :
  uphill_distance = 100 →
  downhill_distance = 50 →
  downhill_speed = 40 →
  average_speed = 32.73 →
  ∃ uphill_speed : ℝ,
    uphill_speed = 30 ∧
    average_speed = (uphill_distance + downhill_distance) / 
      (uphill_distance / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_uphill_speed_calculation_l966_96678


namespace NUMINAMATH_CALUDE_two_identical_objects_five_recipients_l966_96656

theorem two_identical_objects_five_recipients : ∀ n : ℕ, n = 5 →
  (Nat.choose n 2) + (Nat.choose n 1) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_two_identical_objects_five_recipients_l966_96656


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l966_96619

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 20 * x^4 - 18 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 15 / 5 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l966_96619


namespace NUMINAMATH_CALUDE_solution_concentration_l966_96687

/-- Theorem: Concentration of solution to be added to achieve target concentration --/
theorem solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (drain_volume : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 50)
  (h2 : initial_concentration = 0.6)
  (h3 : drain_volume = 35)
  (h4 : final_concentration = 0.46)
  : ∃ (x : ℝ), 
    (initial_volume - drain_volume) * initial_concentration + drain_volume * x = 
    initial_volume * final_concentration ∧ 
    x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_solution_concentration_l966_96687


namespace NUMINAMATH_CALUDE_local_minimum_implies_m_equals_two_l966_96645

/-- The function f(x) = x(x-m)² -/
def f (x m : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (x m : ℝ) : ℝ := (x - m)^2 + 2*x*(x - m)

theorem local_minimum_implies_m_equals_two :
  ∀ m : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x m ≥ f 2 m) →
  f_derivative 2 m = 0 →
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_m_equals_two_l966_96645


namespace NUMINAMATH_CALUDE_original_equals_scientific_l966_96671

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1030000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.03
  , exponent := 9
  , is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l966_96671


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l966_96690

theorem imaginary_part_of_z (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin (2 * θ) - 1) (Real.sqrt 2 * Real.cos θ - 1)
  (z.re = 0) → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l966_96690


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l966_96675

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (999 : ℚ)

/-- The main theorem: proving that 0.714714... divided by 2.857857... equals 119/476 -/
theorem repeating_decimal_division :
  let x : RepeatingDecimal := ⟨0, 714⟩
  let y : RepeatingDecimal := ⟨2, 857⟩
  (repeatingDecimalToRational x) / (repeatingDecimalToRational y) = 119 / 476 := by
  sorry


end NUMINAMATH_CALUDE_repeating_decimal_division_l966_96675


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l966_96639

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l966_96639


namespace NUMINAMATH_CALUDE_roper_lawn_cut_area_l966_96672

/-- Calculates the average area of grass cut per month for a rectangular lawn --/
def average_area_cut_per_month (length width : ℝ) (cuts_per_month_high cuts_per_month_low : ℕ) (months_high months_low : ℕ) : ℝ :=
  let lawn_area := length * width
  let total_cuts_per_year := cuts_per_month_high * months_high + cuts_per_month_low * months_low
  let average_cuts_per_month := total_cuts_per_year / 12
  lawn_area * average_cuts_per_month

/-- Theorem stating that the average area of grass cut per month for Mr. Roper's lawn is 14175 square meters --/
theorem roper_lawn_cut_area :
  average_area_cut_per_month 45 35 15 3 6 6 = 14175 := by sorry

end NUMINAMATH_CALUDE_roper_lawn_cut_area_l966_96672


namespace NUMINAMATH_CALUDE_complex_square_l966_96686

theorem complex_square : (1 - Complex.I) ^ 2 = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l966_96686


namespace NUMINAMATH_CALUDE_probability_mixed_selection_l966_96677

/- Define the number of male and female students -/
def num_male : ℕ := 3
def num_female : ℕ := 4

/- Define the total number of students -/
def total_students : ℕ := num_male + num_female

/- Define the number of volunteers to be selected -/
def num_volunteers : ℕ := 3

/- Theorem stating the probability of selecting both male and female students -/
theorem probability_mixed_selection :
  (1 : ℚ) - (Nat.choose num_male num_volunteers + Nat.choose num_female num_volunteers : ℚ) / 
  (Nat.choose total_students num_volunteers : ℚ) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_mixed_selection_l966_96677


namespace NUMINAMATH_CALUDE_possible_S_n_plus_1_l966_96662

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Property: n ≡ S(n) (mod 9) for all natural numbers n -/
axiom S_mod_9 (n : ℕ) : n % 9 = S n % 9

theorem possible_S_n_plus_1 (n : ℕ) (h : S n = 3096) : 
  ∃ m : ℕ, m = n + 1 ∧ S m = 3097 := by sorry

end NUMINAMATH_CALUDE_possible_S_n_plus_1_l966_96662


namespace NUMINAMATH_CALUDE_square_side_length_l966_96651

/-- Right triangle PQR with legs PQ and PR, and a square inside --/
structure RightTriangleWithSquare where
  /-- Length of leg PQ --/
  pq : ℝ
  /-- Length of leg PR --/
  pr : ℝ
  /-- Side length of the square --/
  s : ℝ
  /-- PQ is 9 cm --/
  pq_length : pq = 9
  /-- PR is 12 cm --/
  pr_length : pr = 12
  /-- The square has one side on hypotenuse QR and one vertex on each leg --/
  square_position : s > 0 ∧ s < pq ∧ s < pr

/-- The side length of the square is 15/2 cm --/
theorem square_side_length (t : RightTriangleWithSquare) : t.s = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l966_96651


namespace NUMINAMATH_CALUDE_option_c_not_algorithm_l966_96679

-- Define what constitutes an algorithm
def is_algorithm (process : String) : Prop :=
  ∃ (steps : List String), steps.length > 0 ∧ steps.all (λ step => step.length > 0)

-- Define the options
def option_a : String := "The process of solving the equation 2x-6=0 involves moving terms and making the coefficient 1"
def option_b : String := "To get from Jinan to Vancouver, one must first take a train to Beijing, then transfer to a plane"
def option_c : String := "Solving the equation 2x^2+x-1=0"
def option_d : String := "Using the formula S=πr^2 to calculate the area of a circle with radius 3 involves computing π×3^2"

-- Theorem stating that option C is not an algorithm while others are
theorem option_c_not_algorithm :
  is_algorithm option_a ∧
  is_algorithm option_b ∧
  ¬is_algorithm option_c ∧
  is_algorithm option_d :=
sorry

end NUMINAMATH_CALUDE_option_c_not_algorithm_l966_96679


namespace NUMINAMATH_CALUDE_polynomial_root_ratio_l966_96605

theorem polynomial_root_ratio (a b c d e : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ↔ x = 5 ∨ x = -3 ∨ x = 2 ∨ x = (-(b+d)/a - 5 - (-3) - 2)) →
  (b + d) / a = -12496 / 3173 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_ratio_l966_96605


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_64_l966_96622

theorem x_times_one_minus_f_equals_64 :
  let x : ℝ := (2 + Real.sqrt 2) ^ 6
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 64 := by
  sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_64_l966_96622


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l966_96612

/-- Given a geometric sequence with sum of first m terms Sm, prove S3m = 70 -/
theorem geometric_sequence_sum (m : ℕ) (Sm S2m S3m : ℝ) : 
  Sm = 10 → 
  S2m = 30 → 
  (∃ r : ℝ, r ≠ 0 ∧ S2m - Sm = r * Sm ∧ S3m - S2m = r * (S2m - Sm)) →
  S3m = 70 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l966_96612


namespace NUMINAMATH_CALUDE_prob_three_diff_suits_l966_96614

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing 3 cards with different suits from a standard 52-card deck -/
theorem prob_three_diff_suits : 
  (Nat.choose num_suits cards_drawn * cards_per_suit ^ cards_drawn) / 
  (Nat.choose deck_size cards_drawn) = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_diff_suits_l966_96614


namespace NUMINAMATH_CALUDE_square_area_error_l966_96646

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l966_96646


namespace NUMINAMATH_CALUDE_hyperbola_dimensions_l966_96634

/-- Proves that for a hyperbola with given conditions, a = 3 and b = 4 -/
theorem hyperbola_dimensions (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_conjugate : 2 * b = 8)
  (h_distance : a * b / Real.sqrt (a^2 + b^2) = 12/5) :
  a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_dimensions_l966_96634


namespace NUMINAMATH_CALUDE_magical_stack_size_with_157_fixed_l966_96637

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (pile_a : Finset ℕ := Finset.range n)
  (pile_b : Finset ℕ := Finset.range n)
  (card_157_position : ℕ)
  (card_157_retains_position : card_157_position = 157)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem: The number of cards in a magical stack where card 157 retains its position is 470 --/
theorem magical_stack_size_with_157_fixed (stack : MagicalStack) :
  magical_stack_size stack = 470 := by sorry

end NUMINAMATH_CALUDE_magical_stack_size_with_157_fixed_l966_96637


namespace NUMINAMATH_CALUDE_correct_ratio_achieved_l966_96630

/-- Represents the ratio of diesel to water in the final mixture -/
def diesel_water_ratio : ℚ := 3 / 5

/-- The initial amount of diesel in quarts -/
def initial_diesel : ℚ := 4

/-- The initial amount of petrol in quarts -/
def initial_petrol : ℚ := 4

/-- The amount of water to be added in quarts -/
def water_to_add : ℚ := 20 / 3

/-- Theorem stating that adding the calculated amount of water results in the desired ratio -/
theorem correct_ratio_achieved :
  diesel_water_ratio = initial_diesel / water_to_add := by
  sorry

#check correct_ratio_achieved

end NUMINAMATH_CALUDE_correct_ratio_achieved_l966_96630


namespace NUMINAMATH_CALUDE_bill_drew_eight_squares_l966_96611

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := 4

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 88

/-- Theorem: Bill drew 8 squares -/
theorem bill_drew_eight_squares :
  ∃ (num_squares : ℕ),
    num_squares * square_sides + 
    num_triangles * triangle_sides + 
    num_pentagons * pentagon_sides = total_lines ∧
    num_squares = 8 := by
  sorry

end NUMINAMATH_CALUDE_bill_drew_eight_squares_l966_96611


namespace NUMINAMATH_CALUDE_cafeteria_cottage_pies_l966_96664

theorem cafeteria_cottage_pies :
  ∀ (lasagna_count : ℕ) (lasagna_mince : ℕ) (cottage_pie_mince : ℕ) (total_mince : ℕ),
    lasagna_count = 100 →
    lasagna_mince = 2 →
    cottage_pie_mince = 3 →
    total_mince = 500 →
    ∃ (cottage_pie_count : ℕ),
      cottage_pie_count * cottage_pie_mince + lasagna_count * lasagna_mince = total_mince ∧
      cottage_pie_count = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_cottage_pies_l966_96664


namespace NUMINAMATH_CALUDE_doll_collection_increase_l966_96607

/-- Proves that if adding 2 dolls to a collection increases it by 25%, then the final number of dolls in the collection is 10. -/
theorem doll_collection_increase (original : ℕ) : 
  (original + 2 : ℚ) = original * (1 + 1/4) → original + 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l966_96607


namespace NUMINAMATH_CALUDE_rectangle_diagonal_shorter_percentage_rectangle_diagonal_shorter_approx_25_percent_l966_96660

/-- The percentage difference between the sum of two sides of a 2x1 rectangle
    and its diagonal, relative to the sum of the sides. -/
theorem rectangle_diagonal_shorter_percentage : ℝ :=
  let side_sum := 2 + 1
  let diagonal := Real.sqrt (2^2 + 1^2)
  (side_sum - diagonal) / side_sum * 100

/-- The percentage difference is approximately 25%. -/
theorem rectangle_diagonal_shorter_approx_25_percent :
  ∃ ε > 0, abs (rectangle_diagonal_shorter_percentage - 25) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_shorter_percentage_rectangle_diagonal_shorter_approx_25_percent_l966_96660


namespace NUMINAMATH_CALUDE_first_discount_percentage_l966_96608

/-- Proves that the first discount percentage is 15% given the original price,
    final price after two discounts, and the second discount percentage. -/
theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 495)
  (h2 : final_price = 378.675)
  (h3 : second_discount = 10) :
  ∃ (first_discount : ℝ),
    first_discount = 15 ∧
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l966_96608


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l966_96623

theorem quadratic_roots_range (m : ℝ) : 
  m > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ x < 1 ∧ y < 1 ∧ 
    m * x^2 + (2*m - 1) * x - m + 2 = 0 ∧
    m * y^2 + (2*m - 1) * y - m + 2 = 0) →
  m > (3 + Real.sqrt 7) / 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l966_96623


namespace NUMINAMATH_CALUDE_chess_tournament_games_l966_96647

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 6 → total_games = 12 → (n * (n - 1)) / 2 = total_games → n - 1 = 5 :=
by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l966_96647


namespace NUMINAMATH_CALUDE_integral_proofs_l966_96697

theorem integral_proofs :
  ∀ x : ℝ,
    (deriv (λ y => Real.arctan (Real.log y)) x = 1 / (x * (1 + Real.log x ^ 2))) ∧
    (deriv (λ y => Real.arctan (Real.exp y)) x = Real.exp x / (1 + Real.exp (2 * x))) ∧
    (deriv (λ y => Real.arctan (Real.sin y)) x = Real.cos x / (1 + Real.sin x ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proofs_l966_96697


namespace NUMINAMATH_CALUDE_cows_eating_grass_l966_96618

/-- Amount of grass one cow eats in one week -/
def cow_week_consumption : ℝ := 2

/-- Amount of grass that regrows on one hectare in one week -/
def grass_growth_rate : ℝ := 1

/-- Initial amount of grass on one hectare -/
def initial_grass : ℝ := 4

/-- Theorem stating that 5 cows will eat all the grass on 6 hectares in 6 weeks -/
theorem cows_eating_grass 
  (h1 : 3 * cow_week_consumption * 2 = 2 * initial_grass + 4 * grass_growth_rate)
  (h2 : 2 * cow_week_consumption * 4 = 2 * initial_grass + 8 * grass_growth_rate)
  (h3 : initial_grass = 4 * grass_growth_rate)
  (h4 : cow_week_consumption = 2 * grass_growth_rate) :
  5 * cow_week_consumption * 6 = 6 * initial_grass + 6 * 6 * grass_growth_rate :=
by sorry

end NUMINAMATH_CALUDE_cows_eating_grass_l966_96618


namespace NUMINAMATH_CALUDE_election_result_l966_96615

/-- Represents the result of an election with three candidates. -/
structure ElectionResult where
  totalVotes : ℕ
  votesA : ℕ
  votesB : ℕ
  votesC : ℕ

/-- Theorem stating the correct election results given the conditions. -/
theorem election_result : ∃ (result : ElectionResult),
  result.totalVotes = 10000 ∧
  result.votesA = 3400 ∧
  result.votesB = 4800 ∧
  result.votesC = 2900 ∧
  result.votesA = (34 * result.totalVotes) / 100 ∧
  result.votesB = (48 * result.totalVotes) / 100 ∧
  result.votesB = result.votesA + 1400 ∧
  result.votesA = result.votesC + 500 ∧
  result.totalVotes = result.votesA + result.votesB + result.votesC :=
by
  sorry

#check election_result

end NUMINAMATH_CALUDE_election_result_l966_96615


namespace NUMINAMATH_CALUDE_class_mean_calculation_l966_96624

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_mean = 68 / 100 →
  group2_mean = 74 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 692 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l966_96624


namespace NUMINAMATH_CALUDE_system_solution_l966_96659

/-- The system of equations has only two solutions -/
theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 13 →
  x^2 + y^2 + z^2 = 61 →
  x*y + x*z = 2*y*z →
  ((x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l966_96659


namespace NUMINAMATH_CALUDE_inscribed_square_area_l966_96626

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A square inscribed in the region bounded by the parabola and the x-axis -/
structure InscribedSquare where
  side : ℝ
  center_x : ℝ
  lower_left : ℝ × ℝ
  upper_right : ℝ × ℝ
  on_x_axis : lower_left.2 = 0 ∧ upper_right.2 = side
  on_parabola : parabola upper_right.1 = upper_right.2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.side^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l966_96626


namespace NUMINAMATH_CALUDE_cube_volume_l966_96658

/-- Given a cube where the sum of all edge lengths is 48 cm, prove its volume is 64 cm³ -/
theorem cube_volume (total_edge_length : ℝ) (h : total_edge_length = 48) : 
  (total_edge_length / 12)^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l966_96658


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l966_96695

open Real

theorem sufficient_not_necessary (α : ℝ) :
  (∀ α, α = π/4 → sin α = cos α) ∧
  (∃ α, α ≠ π/4 ∧ sin α = cos α) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l966_96695
