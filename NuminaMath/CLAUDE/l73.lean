import Mathlib

namespace NUMINAMATH_CALUDE_proportional_increase_l73_7310

theorem proportional_increase (x y : ℝ) (c : ℝ) (h1 : y = c * x) :
  let x' := 1.3 * x
  let y' := c * x'
  y' = 2.6 * y →
  (y' - y) / y = 1.6 := by
sorry

end NUMINAMATH_CALUDE_proportional_increase_l73_7310


namespace NUMINAMATH_CALUDE_fraction_equivalence_l73_7390

theorem fraction_equivalence : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 7 / 25 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 10 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 14 / 70 : ℚ) ≠ 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l73_7390


namespace NUMINAMATH_CALUDE_maria_flour_calculation_l73_7305

/-- The amount of flour needed for a given number of cookies -/
def flour_needed (cookies : ℕ) : ℚ :=
  (3 : ℚ) * cookies / 40

theorem maria_flour_calculation :
  flour_needed 120 = 9 := by sorry

end NUMINAMATH_CALUDE_maria_flour_calculation_l73_7305


namespace NUMINAMATH_CALUDE_roots_eccentricity_l73_7319

theorem roots_eccentricity (x₁ x₂ : ℝ) : 
  x₁ * x₂ = 1 → x₁ + x₂ = 79 → (x₁ > 1 ∧ x₂ < 1) ∨ (x₁ < 1 ∧ x₂ > 1) := by
  sorry

end NUMINAMATH_CALUDE_roots_eccentricity_l73_7319


namespace NUMINAMATH_CALUDE_factorization_72_P_72_l73_7342

/-- P(n) represents the number of ways to write a positive integer n 
    as a product of integers greater than 1, where order matters. -/
def P (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 72 is 2^3 * 3^2 -/
theorem factorization_72 : 72 = 2^3 * 3^2 := sorry

/-- The main theorem: P(72) = 17 -/
theorem P_72 : P 72 = 17 := sorry

end NUMINAMATH_CALUDE_factorization_72_P_72_l73_7342


namespace NUMINAMATH_CALUDE_proposition_problem_l73_7381

theorem proposition_problem (a : ℝ) :
  ((∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + a < 0) ∨
   (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + a*x + 1 = 0 ∧ y^2 + a*y + 1 = 0)) →
  (a > 2 ∨ a < 1) :=
sorry

end NUMINAMATH_CALUDE_proposition_problem_l73_7381


namespace NUMINAMATH_CALUDE_line_l_properties_l73_7340

/-- Given a line l: (m-1)x + 2my + 2 = 0, where m is a real number -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x + 2 * m * y + 2 = 0

theorem line_l_properties (m : ℝ) :
  /- 1. Line l passes through the point (2, -1) -/
  line_l m 2 (-1) ∧
  /- 2. If the slope of l is non-positive and y-intercept is non-negative, then m ≤ 0 -/
  ((∀ x y : ℝ, line_l m x y → (1 - m) / (2 * m) ≤ 0 ∧ -1 / m ≥ 0) → m ≤ 0) ∧
  /- 3. When the x-intercept equals the y-intercept, m = -1 -/
  (∃ a : ℝ, a ≠ 0 ∧ line_l m a 0 ∧ line_l m 0 (-a)) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_l_properties_l73_7340


namespace NUMINAMATH_CALUDE_max_exterior_sum_is_34_l73_7360

/-- Represents a rectangular prism with a pyramid added to one face -/
structure PrismWithPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertices : Nat

/-- Calculates the total number of exterior elements (faces, edges, vertices) -/
def totalExteriorElements (shape : PrismWithPyramid) : Nat :=
  shape.prism_faces - 1 + shape.pyramid_new_faces +
  shape.prism_edges + shape.pyramid_new_edges +
  shape.prism_vertices + shape.pyramid_new_vertices

/-- The maximum sum of exterior faces, vertices, and edges -/
def maxExteriorSum : Nat := 34

/-- Theorem stating that the maximum sum of exterior elements is 34 -/
theorem max_exterior_sum_is_34 :
  ∀ shape : PrismWithPyramid,
    shape.prism_faces = 6 ∧
    shape.prism_edges = 12 ∧
    shape.prism_vertices = 8 ∧
    shape.pyramid_new_faces ≤ 4 ∧
    shape.pyramid_new_edges ≤ 4 ∧
    shape.pyramid_new_vertices = 1 →
    totalExteriorElements shape ≤ maxExteriorSum :=
by
  sorry


end NUMINAMATH_CALUDE_max_exterior_sum_is_34_l73_7360


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l73_7364

theorem sum_of_two_numbers (larger smaller : ℕ) : 
  larger = 22 → larger - smaller = 10 → larger + smaller = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l73_7364


namespace NUMINAMATH_CALUDE_three_folds_halved_cut_segments_l73_7383

/-- A rope folded into equal parts, then folded in half, and cut in the middle -/
structure FoldedRope where
  initial_folds : ℕ  -- number of initial equal folds
  halved : Bool      -- whether the rope is folded in half after initial folding
  cut : Bool         -- whether the rope is cut in the middle

/-- Calculate the number of segments after folding and cutting -/
def num_segments (rope : FoldedRope) : ℕ :=
  if rope.halved ∧ rope.cut then
    rope.initial_folds * 2 + 1
  else
    rope.initial_folds

/-- Theorem: A rope folded into 3 equal parts, then folded in half, and cut in the middle results in 7 segments -/
theorem three_folds_halved_cut_segments :
  ∀ (rope : FoldedRope), rope.initial_folds = 3 → rope.halved → rope.cut →
  num_segments rope = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_three_folds_halved_cut_segments_l73_7383


namespace NUMINAMATH_CALUDE_red_white_flowers_l73_7376

/-- Represents the number of flowers of each color combination --/
structure FlowerCounts where
  total : ℕ
  yellowWhite : ℕ
  redYellow : ℕ
  redWhite : ℕ

/-- The difference between flowers containing red and white --/
def redWhiteDifference (f : FlowerCounts) : ℤ :=
  (f.redYellow + f.redWhite : ℤ) - (f.yellowWhite + f.redWhite : ℤ)

/-- Theorem stating the number of red and white flowers --/
theorem red_white_flowers (f : FlowerCounts) 
  (h_total : f.total = 44)
  (h_yellowWhite : f.yellowWhite = 13)
  (h_redYellow : f.redYellow = 17)
  (h_redWhiteDiff : redWhiteDifference f = 4) :
  f.redWhite = 14 := by
  sorry

end NUMINAMATH_CALUDE_red_white_flowers_l73_7376


namespace NUMINAMATH_CALUDE_max_friends_theorem_l73_7348

/-- Represents the configuration of gnomes in towers --/
structure GnomeCity (n : ℕ) where
  (n_even : Even n)
  (n_pos : 0 < n)

/-- The maximal number of pairs of gnomes which are friends --/
def max_friends (city : GnomeCity n) : ℕ := n^3 / 4

/-- Theorem stating the maximal number of pairs of gnomes which are friends --/
theorem max_friends_theorem (n : ℕ) (city : GnomeCity n) :
  max_friends city = n^3 / 4 := by sorry

end NUMINAMATH_CALUDE_max_friends_theorem_l73_7348


namespace NUMINAMATH_CALUDE_area_not_above_x_axis_is_half_l73_7389

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four points -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculates the area of a parallelogram -/
def parallelogramArea (pg : Parallelogram) : ℝ :=
  sorry

/-- Calculates the area of the portion of a parallelogram below or on the x-axis -/
def areaNotAboveXAxis (pg : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem area_not_above_x_axis_is_half (pg : Parallelogram) :
  pg.p = ⟨4, 2⟩ ∧ pg.q = ⟨-2, -2⟩ ∧ pg.r = ⟨-6, -6⟩ ∧ pg.s = ⟨0, -2⟩ →
  areaNotAboveXAxis pg = (parallelogramArea pg) / 2 :=
sorry

end NUMINAMATH_CALUDE_area_not_above_x_axis_is_half_l73_7389


namespace NUMINAMATH_CALUDE_thirteenth_term_is_30_l73_7349

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a5_eq_6 : a 5 = 6
  a8_eq_15 : a 8 = 15

/-- The 13th term of the arithmetic sequence is 30 -/
theorem thirteenth_term_is_30 (seq : ArithmeticSequence) : seq.a 13 = 30 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_is_30_l73_7349


namespace NUMINAMATH_CALUDE_min_value_product_l73_7325

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 8) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 22 * Real.sqrt 11 - 57 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l73_7325


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_three_l73_7353

theorem abs_m_minus_n_equals_three (m n : ℝ) 
  (h1 : m * n = 4) 
  (h2 : m + n = 5) : 
  |m - n| = 3 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_three_l73_7353


namespace NUMINAMATH_CALUDE_cyclic_inequality_l73_7330

theorem cyclic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  18 * (1 / ((3 - a) * (4 - a)) + 1 / ((3 - b) * (4 - b)) + 1 / ((3 - c) * (4 - c))) + 
  2 * (a * b + b * c + c * a) ≥ 15 := by
sorry


end NUMINAMATH_CALUDE_cyclic_inequality_l73_7330


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l73_7328

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} :=
sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x < 2 - 7/2 * t) ↔ (t < 3/2 ∨ t > 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l73_7328


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l73_7304

theorem greatest_integer_with_gcf_five : ∃ n : ℕ, n < 200 ∧ Nat.gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 → Nat.gcd m 30 = 5 → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l73_7304


namespace NUMINAMATH_CALUDE_bo_number_l73_7375

theorem bo_number (a b : ℂ) : 
  a * b = 52 - 28 * I ∧ a = 7 + 4 * I → b = 476 / 65 - 404 / 65 * I :=
by sorry

end NUMINAMATH_CALUDE_bo_number_l73_7375


namespace NUMINAMATH_CALUDE_halfway_fraction_l73_7393

theorem halfway_fraction (a b : ℚ) (ha : a = 1/4) (hb : b = 1/2) :
  (a + b) / 2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l73_7393


namespace NUMINAMATH_CALUDE_best_shooter_D_l73_7336

structure Shooter where
  name : String
  average_score : ℝ
  variance : ℝ

def is_best_shooter (s : Shooter) (shooters : List Shooter) : Prop :=
  (∀ t ∈ shooters, s.average_score ≥ t.average_score) ∧
  (∀ t ∈ shooters, s.average_score = t.average_score → s.variance ≤ t.variance)

theorem best_shooter_D :
  let shooters := [
    ⟨"A", 9, 1.2⟩,
    ⟨"B", 8, 0.4⟩,
    ⟨"C", 9, 1.8⟩,
    ⟨"D", 9, 0.4⟩
  ]
  let D := ⟨"D", 9, 0.4⟩
  is_best_shooter D shooters := by
  sorry

#check best_shooter_D

end NUMINAMATH_CALUDE_best_shooter_D_l73_7336


namespace NUMINAMATH_CALUDE_integer_product_condition_l73_7312

theorem integer_product_condition (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 3) := by
sorry

end NUMINAMATH_CALUDE_integer_product_condition_l73_7312


namespace NUMINAMATH_CALUDE_quadratic_function_and_area_l73_7337

-- Define the quadratic function f
def f : ℝ → ℝ := fun x ↦ x^2 + 2*x + 1

-- Theorem statement
theorem quadratic_function_and_area :
  (∀ x, (deriv f) x = 2*x + 2) ∧ 
  (∃! x, f x = 0) ∧
  (∫ x in (-3)..0, ((-x^2 - 4*x + 1) - f x)) = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_and_area_l73_7337


namespace NUMINAMATH_CALUDE_quadratic_set_theorem_l73_7300

theorem quadratic_set_theorem (a : ℝ) : 
  ({x : ℝ | x^2 + a*x = 0} = {0, 1}) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_set_theorem_l73_7300


namespace NUMINAMATH_CALUDE_odd_function_property_l73_7385

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : ∀ x, f (x + 2) = -f x) : 
  f 6 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l73_7385


namespace NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_incorrect_statement_d_main_theorem_l73_7354

-- Statement A
theorem statement_a (x y : ℝ) (hx : x > 0) (hy : y > 0) : x / y + y / x ≥ 2 := by sorry

-- Statement B
theorem statement_b (x : ℝ) : (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by sorry

-- Statement C (incorrect)
theorem statement_c_incorrect : ∃ x : ℝ, x > 0 ∧ x < 1 ∧ Real.log x / Real.log 10 + Real.log 10 / Real.log x < 2 := by sorry

-- Statement D
theorem statement_d (a : ℝ) (ha : a > 0) : (1 + a) * (1 + 1 / a) ≥ 4 := by sorry

-- Main theorem
theorem main_theorem : 
  (∀ x y : ℝ, x > 0 → y > 0 → x / y + y / x ≥ 2) ∧
  (∀ x : ℝ, (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2) ∧
  (∃ x : ℝ, x > 0 ∧ x < 1 ∧ Real.log x / Real.log 10 + Real.log 10 / Real.log x < 2) ∧
  (∀ a : ℝ, a > 0 → (1 + a) * (1 + 1 / a) ≥ 4) := by sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_incorrect_statement_d_main_theorem_l73_7354


namespace NUMINAMATH_CALUDE_sequence_problem_l73_7323

/-- Given a sequence {a_n} and a geometric sequence {b_n}, prove a_10 = 64 -/
theorem sequence_problem (a b : ℕ → ℚ) : 
  a 1 = 1/8 →                           -- First term of sequence {a_n}
  b 5 = 2 →                             -- b_5 = 2 in geometric sequence {b_n}
  (∀ n, b n = a (n+1) / a n) →          -- Relation between a_n and b_n
  (∃ q, ∀ n, b n = 2 * q^(n-5)) →       -- b_n is a geometric sequence
  a 10 = 64 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l73_7323


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_for_quadratic_l73_7368

theorem no_positive_integer_solutions_for_quadratic :
  ∀ A : ℕ, 1 ≤ A → A ≤ 9 →
    ¬∃ x : ℕ, x > 0 ∧ x^2 - (A + 1) * x + A * 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_for_quadratic_l73_7368


namespace NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l73_7333

/-- A function is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The sum of two functions --/
def FunctionSum (f g : ℝ → ℝ) : ℝ → ℝ := fun x ↦ f x + g x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (FunctionSum f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (FunctionSum f g) ∧ (¬IsEven f ∨ ¬IsEven g)) := by
  sorry

#check even_sum_sufficient_not_necessary

end NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l73_7333


namespace NUMINAMATH_CALUDE_oliver_candy_theorem_l73_7324

/-- Oliver's Halloween candy problem -/
theorem oliver_candy_theorem (initial_candy : ℕ) (candy_given : ℕ) (remaining_candy : ℕ) :
  initial_candy = 78 →
  candy_given = 10 →
  remaining_candy = initial_candy - candy_given →
  remaining_candy = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_theorem_l73_7324


namespace NUMINAMATH_CALUDE_soda_bottle_duration_l73_7306

/-- Calculates the number of days a bottle of soda will last -/
def soda_duration (bottle_volume : ℚ) (daily_consumption : ℚ) : ℚ :=
  (bottle_volume * 1000) / daily_consumption

theorem soda_bottle_duration :
  let bottle_volume : ℚ := 2
  let daily_consumption : ℚ := 500
  soda_duration bottle_volume daily_consumption = 4 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_duration_l73_7306


namespace NUMINAMATH_CALUDE_maoming_population_scientific_notation_l73_7387

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The population of Maoming city in millions -/
def maoming_population : ℝ := 6.8

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem maoming_population_scientific_notation :
  to_scientific_notation maoming_population = ScientificNotation.mk 6.8 6 sorry := by
  sorry

end NUMINAMATH_CALUDE_maoming_population_scientific_notation_l73_7387


namespace NUMINAMATH_CALUDE_range_of_a_l73_7302

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) ↔ -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l73_7302


namespace NUMINAMATH_CALUDE_sequence_property_l73_7363

def strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def gcd_property (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, gcd (a m) (a n) = a (gcd m n)

def least_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
  (∃ r s : ℕ, r < k ∧ k < s ∧ a k ^ 2 = a r * a s) ∧
  (∀ k' : ℕ, k' < k → ¬∃ r s : ℕ, r < k' ∧ k' < s ∧ a k' ^ 2 = a r * a s)

theorem sequence_property (a : ℕ → ℕ) (k r s : ℕ) :
  strictly_increasing a →
  gcd_property a →
  least_k a k →
  r < k →
  k < s →
  a k ^ 2 = a r * a s →
  r ∣ k ∧ k ∣ s :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l73_7363


namespace NUMINAMATH_CALUDE_min_ticket_cost_is_800_l73_7394

/-- Represents the ticket pricing structure and group composition --/
structure TicketPricing where
  adultPrice : ℕ
  childPrice : ℕ
  groupPrice : ℕ
  groupMinSize : ℕ
  numAdults : ℕ
  numChildren : ℕ

/-- Calculates the minimum cost for tickets given the pricing structure --/
def minTicketCost (pricing : TicketPricing) : ℕ :=
  sorry

/-- Theorem stating that the minimum cost for the given scenario is 800 yuan --/
theorem min_ticket_cost_is_800 :
  let pricing : TicketPricing := {
    adultPrice := 100,
    childPrice := 50,
    groupPrice := 70,
    groupMinSize := 10,
    numAdults := 8,
    numChildren := 4
  }
  minTicketCost pricing = 800 := by sorry

end NUMINAMATH_CALUDE_min_ticket_cost_is_800_l73_7394


namespace NUMINAMATH_CALUDE_other_class_size_l73_7399

theorem other_class_size (avg_zits_other : ℝ) (avg_zits_jones : ℝ) 
  (zits_diff : ℕ) (jones_kids : ℕ) :
  avg_zits_other = 5 →
  avg_zits_jones = 6 →
  zits_diff = 67 →
  jones_kids = 32 →
  ∃ other_kids : ℕ, 
    (jones_kids : ℝ) * avg_zits_jones = 
    (other_kids : ℝ) * avg_zits_other + zits_diff ∧
    other_kids = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_class_size_l73_7399


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l73_7317

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 5 = 3 ∧ 
  a 6 = -2

/-- The first term of the sequence -/
def FirstTerm (a : ℕ → ℤ) : ℤ := a 1

/-- The common difference of the sequence -/
def CommonDifference (a : ℕ → ℤ) : ℤ := a 2 - a 1

/-- The general term formula of the sequence -/
def GeneralTerm (a : ℕ → ℤ) (n : ℕ) : ℤ := 28 - 5 * n

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : ArithmeticSequence a) : 
  FirstTerm a = 23 ∧ 
  CommonDifference a = -5 ∧ 
  (∀ n : ℕ, a n = GeneralTerm a n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l73_7317


namespace NUMINAMATH_CALUDE_power_zero_eq_one_three_power_zero_l73_7398

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem three_power_zero : (3 : ℝ)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_three_power_zero_l73_7398


namespace NUMINAMATH_CALUDE_divisibility_check_l73_7338

theorem divisibility_check (n : ℕ) : 
  n = 1493826 → 
  n % 3 = 0 ∧ 
  ¬(n % 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_check_l73_7338


namespace NUMINAMATH_CALUDE_rectangle_y_value_l73_7346

/-- A rectangle with vertices at (-3, y), (5, y), (-3, -2), and (5, -2) has an area of 96 square units. -/
def rectangle_area (y : ℝ) : Prop :=
  (5 - (-3)) * (y - (-2)) = 96

/-- The theorem states that if y is negative and satisfies the rectangle_area condition, then y = -14. -/
theorem rectangle_y_value (y : ℝ) (h1 : y < 0) (h2 : rectangle_area y) : y = -14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l73_7346


namespace NUMINAMATH_CALUDE_equation_solutions_l73_7377

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 14*x - 10) + 1 / (x^2 + 3*x - 10) + 1 / (x^2 - 16*x - 10)
  {x : ℝ | f x = 0} = {5, -2, 2, -5} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l73_7377


namespace NUMINAMATH_CALUDE_relative_error_comparison_l73_7343

theorem relative_error_comparison :
  let line1_length : ℝ := 25
  let line1_error : ℝ := 0.05
  let line2_length : ℝ := 125
  let line2_error : ℝ := 0.25
  let relative_error1 : ℝ := line1_error / line1_length
  let relative_error2 : ℝ := line2_error / line2_length
  relative_error1 = relative_error2 :=
by sorry

end NUMINAMATH_CALUDE_relative_error_comparison_l73_7343


namespace NUMINAMATH_CALUDE_divides_power_sum_l73_7397

theorem divides_power_sum (a b c : ℤ) (h : (a + b + c) ∣ (a^2 + b^2 + c^2)) :
  ∀ k : ℕ, (a + b + c) ∣ (a^(2^k) + b^(2^k) + c^(2^k)) :=
sorry

end NUMINAMATH_CALUDE_divides_power_sum_l73_7397


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l73_7361

theorem logarithmic_equation_solution (a : ℝ) (ha : a > 0) :
  ∃ x : ℝ, x > 1 ∧ Real.log (a * x) = 2 * Real.log (x - 1) ↔
  ∃ x : ℝ, x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l73_7361


namespace NUMINAMATH_CALUDE_only_one_valid_assignment_l73_7329

/-- Represents an assignment statement --/
inductive AssignmentStatement
  | Assign (lhs : String) (rhs : String)

/-- Checks if an assignment statement is valid --/
def isValidAssignment (stmt : AssignmentStatement) : Bool :=
  match stmt with
  | AssignmentStatement.Assign lhs rhs => 
    (lhs.all Char.isAlpha) && (rhs ≠ "")

/-- The list of given statements --/
def givenStatements : List AssignmentStatement := [
  AssignmentStatement.Assign "2" "A",
  AssignmentStatement.Assign "x+y" "2",
  AssignmentStatement.Assign "A-B" "-2",
  AssignmentStatement.Assign "A" "A*A"
]

/-- Theorem: Only one of the given statements is a valid assignment --/
theorem only_one_valid_assignment :
  (givenStatements.filter isValidAssignment).length = 1 :=
sorry

end NUMINAMATH_CALUDE_only_one_valid_assignment_l73_7329


namespace NUMINAMATH_CALUDE_gcd_of_175_100_75_base_conversion_l73_7307

-- Part 1: GCD of 175, 100, and 75
theorem gcd_of_175_100_75 : Nat.gcd 175 (Nat.gcd 100 75) = 25 := by sorry

-- Part 2: Base conversion
def base_6_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

def decimal_to_base_8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem base_conversion :
  (base_6_to_decimal [5, 1, 0, 1] = 227) ∧
  (decimal_to_base_8 227 = [3, 4, 3]) := by sorry

end NUMINAMATH_CALUDE_gcd_of_175_100_75_base_conversion_l73_7307


namespace NUMINAMATH_CALUDE_product_purchase_l73_7341

theorem product_purchase (misunderstood_total : ℕ) (actual_total : ℕ) 
  (h1 : misunderstood_total = 189)
  (h2 : actual_total = 147) :
  ∃ (price : ℕ) (quantity : ℕ),
    price * quantity = actual_total ∧
    (price + 6) * quantity = misunderstood_total ∧
    price = 21 ∧
    quantity = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_purchase_l73_7341


namespace NUMINAMATH_CALUDE_orange_eating_contest_l73_7326

theorem orange_eating_contest (num_students : ℕ) (max_oranges min_oranges : ℕ) :
  num_students = 8 →
  max_oranges = 8 →
  min_oranges = 1 →
  max_oranges - min_oranges = 7 := by
sorry

end NUMINAMATH_CALUDE_orange_eating_contest_l73_7326


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l73_7380

theorem x_squared_plus_reciprocal (x : ℝ) (h : 20 = x^6 + 1/x^6) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l73_7380


namespace NUMINAMATH_CALUDE_clock_hands_separation_l73_7379

/-- Represents the angle between clock hands at a given time -/
def clockHandAngle (m : ℕ) : ℝ :=
  |6 * m - 0.5 * m|

/-- Checks if the angle between clock hands is 1° (or equivalent) -/
def isOneDegreeSeparation (m : ℕ) : Prop :=
  ∃ k : ℤ, clockHandAngle m = 1 + 360 * k ∨ clockHandAngle m = 1 - 360 * k

theorem clock_hands_separation :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 720 →
    (isOneDegreeSeparation m ↔ m = 262 ∨ m = 458) :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_separation_l73_7379


namespace NUMINAMATH_CALUDE_conic_section_focus_l73_7356

/-- The conic section defined by parametric equations x = t^2 and y = 2t -/
def conic_section (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

/-- The focus of the conic section -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the conic section defined by parametric equations x = t^2 and y = 2t is (1, 0) -/
theorem conic_section_focus :
  ∀ t : ℝ, ∃ a : ℝ, a > 0 ∧ (conic_section t).2^2 = 4 * a * (conic_section t).1 ∧ focus = (a, 0) :=
sorry

end NUMINAMATH_CALUDE_conic_section_focus_l73_7356


namespace NUMINAMATH_CALUDE_adjacent_diff_one_l73_7373

/-- Represents a 9x9 table filled with integers from 1 to 81 --/
def Table := Fin 9 → Fin 9 → Fin 81

/-- Two cells are adjacent if they are horizontally or vertically neighboring --/
def adjacent (i j i' j' : Fin 9) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j'.val + 1 = j.val)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i'.val + 1 = i.val))

/-- The table contains all integers from 1 to 81 exactly once --/
def validTable (t : Table) : Prop :=
  ∀ n : Fin 81, ∃! (i j : Fin 9), t i j = n

/-- Main theorem: In a 9x9 table filled with integers from 1 to 81,
    there exist two adjacent cells whose values differ by exactly 1 --/
theorem adjacent_diff_one (t : Table) (h : validTable t) :
  ∃ (i j i' j' : Fin 9), adjacent i j i' j' ∧ 
    (t i j).val = (t i' j').val + 1 ∨ (t i j).val + 1 = (t i' j').val :=
sorry

end NUMINAMATH_CALUDE_adjacent_diff_one_l73_7373


namespace NUMINAMATH_CALUDE_uncovered_area_square_circle_l73_7365

/-- The area of a square that cannot be covered by a moving circle -/
theorem uncovered_area_square_circle (square_side : ℝ) (circle_diameter : ℝ) 
  (h_square : square_side = 4)
  (h_circle : circle_diameter = 1) :
  (square_side - circle_diameter) ^ 2 + π * (circle_diameter / 2) ^ 2 = 4 + π / 4 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_square_circle_l73_7365


namespace NUMINAMATH_CALUDE_output_for_five_l73_7392

def program_output (x : ℤ) : ℤ :=
  if x < 3 then 2 * x
  else if x > 3 then x * x - 1
  else 2

theorem output_for_five :
  program_output 5 = 24 :=
by sorry

end NUMINAMATH_CALUDE_output_for_five_l73_7392


namespace NUMINAMATH_CALUDE_unique_solution_values_l73_7366

/-- The function representing the quadratic expression inside the absolute value -/
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 3*a

/-- The inequality condition -/
def inequality_condition (a x : ℝ) : Prop := |f a x| ≤ 2

/-- The property of having exactly one solution -/
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x, inequality_condition a x

/-- The main theorem stating that a = 1 and a = 2 are the only values satisfying the condition -/
theorem unique_solution_values :
  ∀ a : ℝ, has_exactly_one_solution a ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_values_l73_7366


namespace NUMINAMATH_CALUDE_set_union_implies_a_value_l73_7347

theorem set_union_implies_a_value (a : ℝ) :
  let A : Set ℝ := {2^a, 3}
  let B : Set ℝ := {2, 3}
  A ∪ B = {2, 3, 4} →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_set_union_implies_a_value_l73_7347


namespace NUMINAMATH_CALUDE_museum_travel_distance_l73_7352

/-- Calculates the total distance traveled to visit two museums on separate days -/
def totalDistanceTraveled (distance1 : ℕ) (distance2 : ℕ) : ℕ :=
  2 * distance1 + 2 * distance2

/-- Proves that visiting museums at 5 and 15 miles results in a total travel of 40 miles -/
theorem museum_travel_distance :
  totalDistanceTraveled 5 15 = 40 := by
  sorry

#eval totalDistanceTraveled 5 15

end NUMINAMATH_CALUDE_museum_travel_distance_l73_7352


namespace NUMINAMATH_CALUDE_total_students_in_line_l73_7320

theorem total_students_in_line 
  (students_in_front : ℕ) 
  (students_behind : ℕ) 
  (h1 : students_in_front = 15)
  (h2 : students_behind = 12) :
  students_in_front + 1 + students_behind = 28 :=
by sorry

end NUMINAMATH_CALUDE_total_students_in_line_l73_7320


namespace NUMINAMATH_CALUDE_max_value_x_4_minus_3x_l73_7351

theorem max_value_x_4_minus_3x :
  ∃ (max : ℝ), max = 4/3 ∧
  (∀ x : ℝ, 0 < x → x < 4/3 → x * (4 - 3 * x) ≤ max) ∧
  (∃ x : ℝ, 0 < x ∧ x < 4/3 ∧ x * (4 - 3 * x) = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_4_minus_3x_l73_7351


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l73_7321

theorem pure_imaginary_condition (k : ℝ) : 
  (2 * k^2 - 3 * k - 2 : ℂ) + (k^2 - 2 * k : ℂ) * Complex.I = Complex.I * (k^2 - 2 * k : ℂ) ↔ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l73_7321


namespace NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degrees_l73_7378

theorem scalene_triangle_with_double_angle_and_36_degrees :
  ∀ (x y z : ℝ),
  0 < x ∧ 0 < y ∧ 0 < z →  -- angles are positive
  x < y ∧ y < z →  -- scalene triangle condition
  x + y + z = 180 →  -- sum of angles in a triangle
  (x = 36 ∨ y = 36 ∨ z = 36) →  -- one angle is 36°
  (x = 2*y ∨ y = 2*x ∨ y = 2*z ∨ z = 2*x ∨ z = 2*y) →  -- one angle is double another
  ((x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126)) := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degrees_l73_7378


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l73_7308

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l73_7308


namespace NUMINAMATH_CALUDE_inequality_chain_l73_7313

theorem inequality_chain (x : ℝ) (h : 1 < x ∧ x < 2) :
  ((Real.log x) / x) ^ 2 < (Real.log x) / x ∧ (Real.log x) / x < (Real.log (x^2)) / (x^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l73_7313


namespace NUMINAMATH_CALUDE_flagstaff_height_l73_7359

/-- Given a flagstaff and a building casting shadows under similar conditions, 
    this theorem proves the height of the flagstaff. -/
theorem flagstaff_height 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (flagstaff_shadow : ℝ) 
  (h_building : building_height = 12.5)
  (h_building_shadow : building_shadow = 28.75)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25) :
  (building_height * flagstaff_shadow) / building_shadow = 17.5 := by
  sorry

#check flagstaff_height

end NUMINAMATH_CALUDE_flagstaff_height_l73_7359


namespace NUMINAMATH_CALUDE_square_difference_l73_7357

theorem square_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l73_7357


namespace NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l73_7318

/-- A positive three-digit number in the form 2C4, where C is a single digit. -/
def number (c : ℕ) : ℕ := 200 + 10 * c + 4

/-- Predicate to check if a number is a multiple of 5. -/
def is_multiple_of_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

/-- Theorem stating that there are no digits C such that 2C4 is a multiple of 5. -/
theorem no_multiple_of_five_2c4 :
  ¬ ∃ c : ℕ, c < 10 ∧ is_multiple_of_five (number c) := by
  sorry


end NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l73_7318


namespace NUMINAMATH_CALUDE_probability_no_shaded_square_l73_7367

/-- Represents a rectangle in the grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The grid configuration --/
def grid_width : Nat := 201
def grid_height : Nat := 3
def shaded_column : Nat := grid_width / 2 + 1

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Bool :=
  r.left ≤ shaded_column && shaded_column ≤ r.right

/-- Counts the total number of possible rectangles --/
def total_rectangles : Nat :=
  (grid_width.choose 2) * (grid_height.choose 2)

/-- Counts the number of rectangles that contain a shaded square --/
def shaded_rectangles : Nat :=
  grid_height * (shaded_column - 1) * (grid_width - shaded_column)

/-- The main theorem --/
theorem probability_no_shaded_square :
  (total_rectangles - shaded_rectangles) / total_rectangles = 100 / 201 := by
  sorry


end NUMINAMATH_CALUDE_probability_no_shaded_square_l73_7367


namespace NUMINAMATH_CALUDE_average_minutes_run_is_16_l73_7372

/-- Represents the average number of minutes run per day for each grade --/
structure GradeRunningAverage where
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ

/-- Represents the ratio of students in each grade --/
structure GradeRatio where
  sixth_to_eighth : ℝ
  sixth_to_seventh : ℝ

/-- Calculates the average number of minutes run per day by all students --/
def average_minutes_run (avg : GradeRunningAverage) (ratio : GradeRatio) : ℝ :=
  sorry

/-- Theorem stating that the average number of minutes run per day is 16 --/
theorem average_minutes_run_is_16 (avg : GradeRunningAverage) (ratio : GradeRatio) 
  (h1 : avg.sixth = 16)
  (h2 : avg.seventh = 18)
  (h3 : avg.eighth = 12)
  (h4 : ratio.sixth_to_eighth = 3)
  (h5 : ratio.sixth_to_seventh = 1.5) :
  average_minutes_run avg ratio = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_minutes_run_is_16_l73_7372


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l73_7303

-- System of equations (1)
theorem system_one_solution (x y : ℝ) : 
  3*x - 2*y = 6 ∧ 2*x + 3*y = 17 → x = 4 ∧ y = 3 := by
sorry

-- System of equations (2)
theorem system_two_solution (x y : ℝ) :
  x + 4*y = 14 ∧ (x-3)/4 - (y-3)/3 = 1/12 → x = 3 ∧ y = 11/4 := by
sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l73_7303


namespace NUMINAMATH_CALUDE_harrison_croissant_expenditure_l73_7327

/-- The cost of a regular croissant in dollars -/
def regular_croissant_cost : ℚ := 7/2

/-- The cost of an almond croissant in dollars -/
def almond_croissant_cost : ℚ := 11/2

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The total amount Harrison spends on croissants in a year -/
def total_spent_on_croissants : ℚ := 
  (regular_croissant_cost * weeks_in_year) + (almond_croissant_cost * weeks_in_year)

theorem harrison_croissant_expenditure : 
  total_spent_on_croissants = 468 := by sorry

end NUMINAMATH_CALUDE_harrison_croissant_expenditure_l73_7327


namespace NUMINAMATH_CALUDE_half_work_days_l73_7358

/-- Represents the number of days it takes for the larger group to complete half the work -/
def days_for_half_work (original_days : ℕ) (efficiency_ratio : ℚ) : ℚ :=
  original_days / (2 * (1 + 2 * efficiency_ratio))

/-- Theorem stating that under the given conditions, it takes 4 days for the larger group to complete half the work -/
theorem half_work_days :
  days_for_half_work 20 (3/4) = 4 := by sorry

end NUMINAMATH_CALUDE_half_work_days_l73_7358


namespace NUMINAMATH_CALUDE_finite_seq_nat_countable_l73_7388

-- Define the type for finite sequences of natural numbers
def FiniteSeqNat := List Nat

-- Statement of the theorem
theorem finite_seq_nat_countable : 
  ∃ f : FiniteSeqNat → Nat, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_finite_seq_nat_countable_l73_7388


namespace NUMINAMATH_CALUDE_total_fencing_cost_l73_7371

/-- Calculates the total fencing cost for an irregular shaped plot -/
theorem total_fencing_cost (square_area : ℝ) (rect_length rect_height : ℝ) (triangle_side : ℝ)
  (square_cost rect_cost triangle_cost : ℝ) (gate_cost : ℝ)
  (h_square_area : square_area = 289)
  (h_rect_length : rect_length = 45)
  (h_rect_height : rect_height = 15)
  (h_triangle_side : triangle_side = 20)
  (h_square_cost : square_cost = 55)
  (h_rect_cost : rect_cost = 65)
  (h_triangle_cost : triangle_cost = 70)
  (h_gate_cost : gate_cost = 750) :
  4 * Real.sqrt square_area * square_cost +
  (2 * rect_height + rect_length) * rect_cost +
  3 * triangle_side * triangle_cost +
  gate_cost = 13565 := by
  sorry


end NUMINAMATH_CALUDE_total_fencing_cost_l73_7371


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l73_7332

def initial_amount : ℚ := 150

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_remaining_money :
  remaining_amount = 20 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l73_7332


namespace NUMINAMATH_CALUDE_prob_only_one_selected_l73_7301

/-- The probability of only one person being selected given individual and joint selection probabilities -/
theorem prob_only_one_selected
  (pH : ℚ) (pW : ℚ) (pHW : ℚ)
  (hpH : pH = 2 / 5)
  (hpW : pW = 3 / 7)
  (hpHW : pHW = 1 / 3) :
  pH * (1 - pW) + (1 - pH) * pW = 17 / 35 := by
sorry


end NUMINAMATH_CALUDE_prob_only_one_selected_l73_7301


namespace NUMINAMATH_CALUDE_new_year_money_distribution_l73_7314

/-- Represents the distribution of money to three grandsons --/
structure MoneyDistribution :=
  (grandson1 : ℕ)
  (grandson2 : ℕ)
  (grandson3 : ℕ)

/-- Checks if a distribution is valid according to the problem conditions --/
def is_valid_distribution (d : MoneyDistribution) : Prop :=
  -- Total sum is 300
  d.grandson1 + d.grandson2 + d.grandson3 = 300 ∧
  -- Each amount is divisible by 10 (smallest denomination)
  d.grandson1 % 10 = 0 ∧ d.grandson2 % 10 = 0 ∧ d.grandson3 % 10 = 0 ∧
  -- Each amount is one of the allowed denominations (50, 20, or 10)
  (d.grandson1 % 50 = 0 ∨ d.grandson1 % 20 = 0 ∨ d.grandson1 % 10 = 0) ∧
  (d.grandson2 % 50 = 0 ∨ d.grandson2 % 20 = 0 ∨ d.grandson2 % 10 = 0) ∧
  (d.grandson3 % 50 = 0 ∨ d.grandson3 % 20 = 0 ∨ d.grandson3 % 10 = 0) ∧
  -- Number of bills condition
  (d.grandson1 / 10 = (d.grandson2 / 20) * (d.grandson3 / 50) ∨
   d.grandson2 / 20 = (d.grandson1 / 10) * (d.grandson3 / 50) ∨
   d.grandson3 / 50 = (d.grandson1 / 10) * (d.grandson2 / 20))

/-- The theorem to be proved --/
theorem new_year_money_distribution :
  ∀ d : MoneyDistribution,
    is_valid_distribution d →
    (d = ⟨100, 100, 100⟩ ∨ d = ⟨90, 60, 150⟩ ∨ d = ⟨90, 150, 60⟩ ∨
     d = ⟨60, 90, 150⟩ ∨ d = ⟨60, 150, 90⟩ ∨ d = ⟨150, 60, 90⟩ ∨
     d = ⟨150, 90, 60⟩) :=
by sorry


end NUMINAMATH_CALUDE_new_year_money_distribution_l73_7314


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l73_7384

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 40 = 16) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l73_7384


namespace NUMINAMATH_CALUDE_train_speed_calculation_l73_7344

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 290)
  (h3 : crossing_time = 23.998080153587715) :
  (((train_length + bridge_length) / crossing_time) * 3.6) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l73_7344


namespace NUMINAMATH_CALUDE_acetone_nine_moles_weight_l73_7370

/-- The molecular weight of a single molecule of Acetone in g/mol -/
def acetone_molecular_weight : ℝ :=
  3 * 12.01 + 6 * 1.008 + 1 * 16.00

/-- The molecular weight of n moles of Acetone in grams -/
def acetone_weight (n : ℝ) : ℝ :=
  n * acetone_molecular_weight

/-- Theorem: The molecular weight of 9 moles of Acetone is 522.702 grams -/
theorem acetone_nine_moles_weight :
  acetone_weight 9 = 522.702 := by
  sorry

end NUMINAMATH_CALUDE_acetone_nine_moles_weight_l73_7370


namespace NUMINAMATH_CALUDE_polynomial_equality_l73_7339

theorem polynomial_equality : 11^5 - 5 * 11^4 + 10 * 11^3 - 10 * 11^2 + 5 * 11 - 1 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l73_7339


namespace NUMINAMATH_CALUDE_solution_set_f_less_g_range_of_a_l73_7369

-- Define the functions f and g
def f (x : ℝ) := abs (x - 4)
def g (x : ℝ) := abs (2 * x + 1)

-- Statement 1
theorem solution_set_f_less_g :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * f x + g x > a * x) ↔ a ∈ Set.Icc (-4) (9/4) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_g_range_of_a_l73_7369


namespace NUMINAMATH_CALUDE_congruence_mod_210_l73_7374

theorem congruence_mod_210 (x : ℤ) : x^5 ≡ x [ZMOD 210] ↔ x ≡ 0 [ZMOD 7] ∨ x ≡ 1 [ZMOD 7] ∨ x ≡ -1 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_congruence_mod_210_l73_7374


namespace NUMINAMATH_CALUDE_ratio_problem_l73_7311

theorem ratio_problem (first_number second_number : ℝ) : 
  first_number / second_number = 20 → first_number = 200 → second_number = 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l73_7311


namespace NUMINAMATH_CALUDE_closest_angles_to_2013_l73_7316

theorem closest_angles_to_2013 (x : ℝ) :
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2) →
  (x = 1935 * π / 180 ∨ x = 2025 * π / 180) ∧
  ∀ y : ℝ, (2^(Real.sin y)^2 + 2^(Real.cos y)^2 = 2 * Real.sqrt 2) →
    (1935 * π / 180 < y ∧ y < 2025 * π / 180) →
    (y ≠ 1935 * π / 180 ∧ y ≠ 2025 * π / 180) →
    ¬(∃ n : ℤ, y = n * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_closest_angles_to_2013_l73_7316


namespace NUMINAMATH_CALUDE_expression_simplification_l73_7386

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/3) (hy : y = -2) : 
  2 * (x^2 - 2*x^2*y) - (3*(x^2 - x*y^2) - (x^2*y - 2*x*y^2 + x^2)) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l73_7386


namespace NUMINAMATH_CALUDE_publishing_break_even_l73_7362

/-- A publishing company's break-even point calculation -/
theorem publishing_break_even 
  (fixed_cost : ℝ) 
  (variable_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : fixed_cost = 50000)
  (h2 : variable_cost = 4)
  (h3 : selling_price = 9) :
  ∃ x : ℝ, x = 10000 ∧ selling_price * x = fixed_cost + variable_cost * x :=
sorry

end NUMINAMATH_CALUDE_publishing_break_even_l73_7362


namespace NUMINAMATH_CALUDE_apple_division_problem_l73_7309

/-- Calculates the minimal number of pieces needed to evenly divide apples among students -/
def minimalPieces (apples : ℕ) (students : ℕ) : ℕ :=
  let components := apples.gcd students
  let applesPerComponent := apples / components
  let studentsPerComponent := students / components
  components * (applesPerComponent + studentsPerComponent - 1)

/-- Proves that the minimal number of pieces to evenly divide 221 apples among 403 students is 611 -/
theorem apple_division_problem :
  minimalPieces 221 403 = 611 := by
  sorry

#eval minimalPieces 221 403

end NUMINAMATH_CALUDE_apple_division_problem_l73_7309


namespace NUMINAMATH_CALUDE_range_of_m_l73_7382

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1
def C₂ (m : ℝ) (x y : ℝ) : Prop := y^2 = 2*(x + m)

-- Define the condition for a single common point above x-axis
def single_common_point (a m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, C₁ a p.1 p.2 ∧ C₂ m p.1 p.2 ∧ p.2 > 0

-- State the theorem
theorem range_of_m (a : ℝ) (h : a > 0) :
  (∀ m : ℝ, single_common_point a m →
    ((0 < a ∧ a < 1 → m = (a^2 + 1)/2 ∨ (-a < m ∧ m ≤ a)) ∧
     (a ≥ 1 → -a < m ∧ m < a))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l73_7382


namespace NUMINAMATH_CALUDE_angela_deliveries_l73_7331

/-- Calculates the total number of meals and packages delivered -/
def total_deliveries (meals : ℕ) (package_multiplier : ℕ) : ℕ :=
  meals + meals * package_multiplier

/-- Proves that given 3 meals and 8 times as many packages, the total deliveries is 27 -/
theorem angela_deliveries : total_deliveries 3 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_angela_deliveries_l73_7331


namespace NUMINAMATH_CALUDE_triangle_properties_l73_7322

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating properties of a specific triangle -/
theorem triangle_properties (t : Triangle) (m : ℝ) : 
  (Real.sqrt 2 * Real.sin t.A = Real.sqrt (3 * Real.cos t.A)) →
  (t.a^2 - t.c^2 = t.b^2 - m * t.b * t.c) →
  (t.a = Real.sqrt 3) →
  (m = 1 ∧ 
   (∀ S : ℝ, S ≤ (3 * Real.sqrt 3) / 4 ∨ 
    ¬(∃ t' : Triangle, S = (t'.b * t'.c * Real.sin t'.A) / 2))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l73_7322


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l73_7396

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l73_7396


namespace NUMINAMATH_CALUDE_total_balls_count_l73_7345

/-- The number of balls owned by Jungkook -/
def jungkook_balls : ℕ := 3

/-- The number of balls owned by Yoongi -/
def yoongi_balls : ℕ := 2

/-- The total number of balls owned by Jungkook and Yoongi -/
def total_balls : ℕ := jungkook_balls + yoongi_balls

theorem total_balls_count : total_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_count_l73_7345


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l73_7350

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l73_7350


namespace NUMINAMATH_CALUDE_average_salary_is_8000_l73_7355

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 11000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def num_people : ℕ := 5

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e

theorem average_salary_is_8000 : (total_salary : ℚ) / num_people = 8000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_is_8000_l73_7355


namespace NUMINAMATH_CALUDE_triangle_inequality_two_points_l73_7335

/-- Triangle inequality for two points in the plane of a triangle -/
theorem triangle_inequality_two_points (A B C P₁ P₂ : ℝ × ℝ) 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) 
  (a₂ : ℝ) (b₂ : ℝ) (c₂ : ℝ) 
  (ha : a = dist B C) 
  (hb : b = dist A C) 
  (hc : c = dist A B) 
  (ha₁ : a₁ = dist P₁ A) 
  (hb₁ : b₁ = dist P₁ B) 
  (hc₁ : c₁ = dist P₁ C) 
  (ha₂ : a₂ = dist P₂ A) 
  (hb₂ : b₂ = dist P₂ B) 
  (hc₂ : c₂ = dist P₂ C) : 
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
sorry

#check triangle_inequality_two_points

end NUMINAMATH_CALUDE_triangle_inequality_two_points_l73_7335


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l73_7395

/-- 
Given a triangle with two sides of 36 units and 45 units, and the third side being an integer,
the least possible perimeter is 91 units.
-/
theorem least_perimeter_triangle : 
  ∀ (x : ℕ), 
  x > 0 → 
  x + 36 > 45 → 
  x + 45 > 36 → 
  36 + 45 > x → 
  (∀ y : ℕ, y > 0 → y + 36 > 45 → y + 45 > 36 → 36 + 45 > y → x + 36 + 45 ≤ y + 36 + 45) →
  x + 36 + 45 = 91 := by
sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l73_7395


namespace NUMINAMATH_CALUDE_same_color_probability_5_8_l73_7391

/-- The probability of drawing two balls of the same color from a bag containing
    5 green balls and 8 white balls. -/
def same_color_probability (green : ℕ) (white : ℕ) : ℚ :=
  let total := green + white
  let prob_both_green := (green / total) * ((green - 1) / (total - 1))
  let prob_both_white := (white / total) * ((white - 1) / (total - 1))
  prob_both_green + prob_both_white

/-- Theorem stating that the probability of drawing two balls of the same color
    from a bag with 5 green balls and 8 white balls is 19/39. -/
theorem same_color_probability_5_8 :
  same_color_probability 5 8 = 19 / 39 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_5_8_l73_7391


namespace NUMINAMATH_CALUDE_cyclic_sum_minimum_l73_7334

theorem cyclic_sum_minimum (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_eq_four : a + b + c + d = 4) : 
  ((b + 3) / (a^2 + 4) + 
   (c + 3) / (b^2 + 4) + 
   (d + 3) / (c^2 + 4) + 
   (a + 3) / (d^2 + 4)) ≥ 3 ∧ 
  ∃ a b c d, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a + b + c + d = 4 ∧ 
    ((b + 3) / (a^2 + 4) + 
     (c + 3) / (b^2 + 4) + 
     (d + 3) / (c^2 + 4) + 
     (a + 3) / (d^2 + 4)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_minimum_l73_7334


namespace NUMINAMATH_CALUDE_min_side_triangle_l73_7315

theorem min_side_triangle (S γ : ℝ) (hS : S > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (1/2 * a * b * Real.sin γ = S) ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    1/2 * a' * b' * Real.sin γ = S →
    c' ≥ 2 * Real.sqrt (S * Real.tan (γ/2))) :=
sorry

end NUMINAMATH_CALUDE_min_side_triangle_l73_7315
