import Mathlib

namespace NUMINAMATH_CALUDE_average_increase_l648_64811

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (player : CricketPlayer) (newRuns : ℕ) : ℚ :=
  (player.totalRuns + newRuns) / (player.innings + 1)

/-- Theorem: A player with 10 innings and 33 run average increases average by 4 after scoring 77 runs -/
theorem average_increase (player : CricketPlayer) 
  (h1 : player.innings = 10)
  (h2 : player.average = 33)
  (h3 : player.totalRuns = player.innings * player.average) :
  newAverage player 77 - player.average = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_l648_64811


namespace NUMINAMATH_CALUDE_rhombus_not_necessarily_planar_l648_64808

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A shape in 3D space -/
class Shape where
  vertices : List Point3D

/-- A triangle is always planar -/
def Triangle (a b c : Point3D) : Shape :=
  { vertices := [a, b, c] }

/-- A trapezoid is always planar -/
def Trapezoid (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- A parallelogram is always planar -/
def Parallelogram (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- A rhombus (quadrilateral with equal sides) -/
def Rhombus (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- Predicate to check if a shape is planar -/
def isPlanar (s : Shape) : Prop :=
  sorry

/-- Theorem stating that a rhombus is not necessarily planar -/
theorem rhombus_not_necessarily_planar :
  ∃ (a b c d : Point3D), ¬(isPlanar (Rhombus a b c d)) ∧
    (∀ (x y z : Point3D), isPlanar (Triangle x y z)) ∧
    (∀ (w x y z : Point3D), isPlanar (Trapezoid w x y z)) ∧
    (∀ (w x y z : Point3D), isPlanar (Parallelogram w x y z)) :=
  sorry

end NUMINAMATH_CALUDE_rhombus_not_necessarily_planar_l648_64808


namespace NUMINAMATH_CALUDE_exists_rational_rearrangement_l648_64893

/-- Represents an infinite decimal fraction as a sequence of digits. -/
def InfiniteDecimal := ℕ → Fin 10

/-- Represents a rearrangement of digits. -/
def Rearrangement := ℕ → ℕ

/-- A number is rational if it can be expressed as a ratio of two integers. -/
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

/-- Converts an InfiniteDecimal to a real number. -/
noncomputable def toReal (d : InfiniteDecimal) : ℝ := sorry

/-- Applies a rearrangement to an InfiniteDecimal. -/
def applyRearrangement (d : InfiniteDecimal) (r : Rearrangement) : InfiniteDecimal :=
  fun n => d (r n)

/-- Theorem: For any infinite decimal, there exists a rearrangement that results in a rational number. -/
theorem exists_rational_rearrangement (d : InfiniteDecimal) :
  ∃ (r : Rearrangement), IsRational (toReal (applyRearrangement d r)) := by sorry

end NUMINAMATH_CALUDE_exists_rational_rearrangement_l648_64893


namespace NUMINAMATH_CALUDE_billys_age_l648_64859

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l648_64859


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l648_64899

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    prove the cost of one dozen pens. -/
theorem cost_of_dozen_pens (cost_3pens_5pencils : ℕ) (cost_ratio_pen_pencil : ℚ) :
  cost_3pens_5pencils = 200 →
  cost_ratio_pen_pencil = 5 / 1 →
  ∃ (cost_pen : ℚ), cost_pen * 12 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l648_64899


namespace NUMINAMATH_CALUDE_problem_statement_l648_64849

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -2) :
  (1 - x) * (1 - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l648_64849


namespace NUMINAMATH_CALUDE_absolute_value_equals_negation_implies_nonpositive_l648_64862

theorem absolute_value_equals_negation_implies_nonpositive (a : ℝ) :
  |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negation_implies_nonpositive_l648_64862


namespace NUMINAMATH_CALUDE_clarinet_rate_is_40_l648_64866

/-- The hourly rate for clarinet lessons --/
def clarinet_rate : ℝ := 40

/-- The number of hours of clarinet lessons per week --/
def clarinet_hours_per_week : ℝ := 3

/-- The number of hours of piano lessons per week --/
def piano_hours_per_week : ℝ := 5

/-- The hourly rate for piano lessons --/
def piano_rate : ℝ := 28

/-- The difference in annual cost between piano and clarinet lessons --/
def annual_cost_difference : ℝ := 1040

/-- The number of weeks in a year --/
def weeks_per_year : ℝ := 52

theorem clarinet_rate_is_40 : 
  piano_hours_per_week * piano_rate * weeks_per_year = 
  clarinet_hours_per_week * clarinet_rate * weeks_per_year + annual_cost_difference :=
by sorry

end NUMINAMATH_CALUDE_clarinet_rate_is_40_l648_64866


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l648_64843

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 6)
  (h_sum : a 3 + a 5 = a 10) :
  a 12 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l648_64843


namespace NUMINAMATH_CALUDE_parabola_directrix_l648_64894

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 2*x + 1) / 8

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -2

/-- Theorem: The directrix of the given parabola is y = -2 -/
theorem parabola_directrix : ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.1 - x)^2 + (p.2 - d)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l648_64894


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l648_64819

-- Factorization of 4x^2 - 64
theorem factorization_1 (x : ℝ) : 4 * x^2 - 64 = 4 * (x + 4) * (x - 4) := by
  sorry

-- Factorization of 4ab^2 - 4a^2b - b^3
theorem factorization_2 (a b : ℝ) : 4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l648_64819


namespace NUMINAMATH_CALUDE_shaded_area_is_74_l648_64877

/-- Represents a square with shaded and unshaded areas -/
structure ShadedSquare where
  side_length : ℝ
  unshaded_rectangles : ℕ
  unshaded_area : ℝ

/-- Calculates the area of the shaded part of the square -/
def shaded_area (s : ShadedSquare) : ℝ :=
  s.side_length ^ 2 - s.unshaded_area

/-- Theorem stating the area of the shaded part for the given conditions -/
theorem shaded_area_is_74 (s : ShadedSquare) 
    (h1 : s.side_length = 10)
    (h2 : s.unshaded_rectangles = 4)
    (h3 : s.unshaded_area = 26) : 
  shaded_area s = 74 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_74_l648_64877


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l648_64895

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 4) (h2 : x ≠ 2) (h3 : x ≠ 5) (h4 : x ≠ 3) (h5 : x ≠ 1) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 1 / ((x - 4) * (x - 2)) :=
by
  sorry

#check simplify_fraction_division

end NUMINAMATH_CALUDE_simplify_fraction_division_l648_64895


namespace NUMINAMATH_CALUDE_increasing_geometric_sequence_exists_l648_64807

theorem increasing_geometric_sequence_exists : ∃ (a : ℕ → ℝ), 
  (∀ n : ℕ, a (n + 1) > a n) ∧  -- increasing
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) ∧  -- geometric
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧  -- first three terms
  a 2 + a 3 = 6 * a 1  -- given condition
:= by sorry

end NUMINAMATH_CALUDE_increasing_geometric_sequence_exists_l648_64807


namespace NUMINAMATH_CALUDE_sequence_formula_l648_64869

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := 2 * n^2 + n

-- Theorem statement
theorem sequence_formula (n : ℕ) : a n = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l648_64869


namespace NUMINAMATH_CALUDE_largest_n_for_integer_factors_l648_64857

def polynomial (n : ℤ) (x : ℤ) : ℤ := 3 * x^2 + n * x + 72

def has_integer_linear_factors (n : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ x, polynomial n x = (3*x + a) * (x + b)

theorem largest_n_for_integer_factors :
  (∃ n : ℤ, has_integer_linear_factors n) ∧
  (∀ m : ℤ, has_integer_linear_factors m → m ≤ 217) ∧
  has_integer_linear_factors 217 :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_integer_factors_l648_64857


namespace NUMINAMATH_CALUDE_negative_three_plus_nine_equals_six_l648_64823

theorem negative_three_plus_nine_equals_six : (-3) + 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_plus_nine_equals_six_l648_64823


namespace NUMINAMATH_CALUDE_unique_n_with_prime_divisor_property_l648_64801

theorem unique_n_with_prime_divisor_property : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (p : ℕ), Prime p ∧
    (∀ (q : ℕ), Prime q → q ∣ (n^2 + 3) → q ≤ p) ∧
    (∀ (q : ℕ), Prime q → q ∣ (n^4 + 6) → p ≤ q) ∧
    p ∣ (n^2 + 3) ∧ p ∣ (n^4 + 6)) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_n_with_prime_divisor_property_l648_64801


namespace NUMINAMATH_CALUDE_system_no_solution_l648_64812

def has_no_solution (a b c : ℤ) : Prop :=
  2 / a = -b / 5 ∧ -b / 5 = 1 / -c ∧ 2 / a ≠ 2 * b / a

theorem system_no_solution : 
  {(a, b, c) : ℤ × ℤ × ℤ | has_no_solution a b c} = 
  {(-2, 5, 1), (2, -5, -1), (10, -1, -5)} := by sorry

end NUMINAMATH_CALUDE_system_no_solution_l648_64812


namespace NUMINAMATH_CALUDE_night_crew_ratio_l648_64856

theorem night_crew_ratio (D N : ℚ) (h1 : D > 0) (h2 : N > 0) : 
  (N * (3/4)) / (D + N * (3/4)) = 1/3 → N/D = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_night_crew_ratio_l648_64856


namespace NUMINAMATH_CALUDE_paths_count_l648_64816

/-- The number of distinct paths from (0, n) to (m, m) on a plane,
    where only moves of 1 unit up or 1 unit left are allowed. -/
def numPaths (n m : ℕ) : ℕ :=
  Nat.choose n m

/-- Theorem stating that the number of distinct paths from (0, n) to (m, m)
    is equal to (n choose m) -/
theorem paths_count (n m : ℕ) (h : m ≤ n) :
  numPaths n m = Nat.choose n m := by
  sorry

end NUMINAMATH_CALUDE_paths_count_l648_64816


namespace NUMINAMATH_CALUDE_f_is_even_g_is_not_odd_even_function_symmetry_odd_function_symmetry_l648_64888

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the functions
def f (x : ℝ) : ℝ := x^4 + x^2
def g (x : ℝ) : ℝ := x^3 + x^2

-- Theorem statements
theorem f_is_even : IsEven f := by sorry

theorem g_is_not_odd : ¬ IsOdd g := by sorry

theorem even_function_symmetry (f : ℝ → ℝ) (h : IsEven f) :
  ∀ x y, f x = y ↔ f (-x) = y := by sorry

theorem odd_function_symmetry (f : ℝ → ℝ) (h : IsOdd f) :
  ∀ x y, f x = y ↔ f (-x) = -y := by sorry

end NUMINAMATH_CALUDE_f_is_even_g_is_not_odd_even_function_symmetry_odd_function_symmetry_l648_64888


namespace NUMINAMATH_CALUDE_pure_imaginary_quotient_l648_64836

/-- Given a real number a and i as the imaginary unit, if (a-i)/(1+i) is a pure imaginary number, then a = 1 -/
theorem pure_imaginary_quotient (a : ℝ) : 
  (∃ (b : ℝ), (a - Complex.I) / (1 + Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_quotient_l648_64836


namespace NUMINAMATH_CALUDE_smallest_w_l648_64851

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 10^2) → 
  w ≥ 300 := by
sorry

end NUMINAMATH_CALUDE_smallest_w_l648_64851


namespace NUMINAMATH_CALUDE_max_cables_theorem_l648_64892

/-- Represents the number of employees using each brand of computer -/
structure EmployeeCount where
  total : Nat
  brandX : Nat
  brandY : Nat

/-- Represents the constraints on connections -/
structure ConnectionConstraints where
  maxConnectionPercentage : Real

/-- Calculates the maximum number of cables that can be installed -/
def maxCables (employees : EmployeeCount) (constraints : ConnectionConstraints) : Nat :=
  sorry

/-- Theorem stating the maximum number of cables that can be installed -/
theorem max_cables_theorem (employees : EmployeeCount) (constraints : ConnectionConstraints) :
  employees.total = 50 →
  employees.brandX = 30 →
  employees.brandY = 20 →
  constraints.maxConnectionPercentage = 0.95 →
  maxCables employees constraints = 300 :=
  sorry

end NUMINAMATH_CALUDE_max_cables_theorem_l648_64892


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l648_64852

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 + 3 * x = 0

/-- Definition of a hyperbola based on its general form -/
def is_hyperbola (A B C D E F : ℝ) : Prop :=
  B^2 - 4*A*C > 0 ∧ A ≠ 0 ∧ C ≠ 0 ∧ A ≠ C

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : 
  ∃ A B C D E F : ℝ, 
    (∀ x y : ℝ, conic_equation x y ↔ A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0) ∧
    is_hyperbola A B C D E F :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l648_64852


namespace NUMINAMATH_CALUDE_prob_spade_heart_king_l648_64891

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a spade, then a heart, then a King from a standard 52-card deck -/
theorem prob_spade_heart_king :
  (NumSpades * NumHearts * NumKings) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 17 / 3683 := by
  sorry


end NUMINAMATH_CALUDE_prob_spade_heart_king_l648_64891


namespace NUMINAMATH_CALUDE_baseball_glove_price_l648_64845

theorem baseball_glove_price :
  let cards_price : ℝ := 25
  let bat_price : ℝ := 10
  let cleats_price : ℝ := 10
  let total_sales : ℝ := 79
  let discount_rate : ℝ := 0.2
  let other_items_total : ℝ := cards_price + bat_price + 2 * cleats_price
  let glove_discounted_price : ℝ := total_sales - other_items_total
  let glove_original_price : ℝ := glove_discounted_price / (1 - discount_rate)
  glove_original_price = 42.5 := by
sorry

end NUMINAMATH_CALUDE_baseball_glove_price_l648_64845


namespace NUMINAMATH_CALUDE_aluminium_count_l648_64880

/-- The number of Aluminium atoms in the compound -/
def n : ℕ := sorry

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 78

/-- The number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- The number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- Theorem stating that the number of Aluminium atoms in the compound is 1 -/
theorem aluminium_count : n = 1 := by sorry

end NUMINAMATH_CALUDE_aluminium_count_l648_64880


namespace NUMINAMATH_CALUDE_emily_spent_234_l648_64840

/-- The cost of Charlie's purchase of 4 burgers and 3 sodas -/
def charlie_cost : ℝ := 4.40

/-- The cost of Dana's purchase of 3 burgers and 4 sodas -/
def dana_cost : ℝ := 3.80

/-- The number of burgers in Charlie's purchase -/
def charlie_burgers : ℕ := 4

/-- The number of sodas in Charlie's purchase -/
def charlie_sodas : ℕ := 3

/-- The number of burgers in Dana's purchase -/
def dana_burgers : ℕ := 3

/-- The number of sodas in Dana's purchase -/
def dana_sodas : ℕ := 4

/-- The number of burgers in Emily's purchase -/
def emily_burgers : ℕ := 2

/-- The number of sodas in Emily's purchase -/
def emily_sodas : ℕ := 1

/-- The cost of a single burger -/
noncomputable def burger_cost : ℝ := 
  (charlie_cost * dana_sodas - dana_cost * charlie_sodas) / 
  (charlie_burgers * dana_sodas - dana_burgers * charlie_sodas)

/-- The cost of a single soda -/
noncomputable def soda_cost : ℝ := 
  (charlie_cost * dana_burgers - dana_cost * charlie_burgers) / 
  (charlie_sodas * dana_burgers - dana_sodas * charlie_burgers)

/-- Emily's total cost -/
noncomputable def emily_cost : ℝ := emily_burgers * burger_cost + emily_sodas * soda_cost

theorem emily_spent_234 : ∃ ε > 0, |emily_cost - 2.34| < ε :=
sorry

end NUMINAMATH_CALUDE_emily_spent_234_l648_64840


namespace NUMINAMATH_CALUDE_tea_party_waiting_time_l648_64820

/-- Mad Hatter's clock speed relative to real time -/
def mad_hatter_clock_speed : ℚ := 5/4

/-- March Hare's clock speed relative to real time -/
def march_hare_clock_speed : ℚ := 5/6

/-- Time shown on both clocks when they meet (in hours after noon) -/
def meeting_time : ℚ := 5

theorem tea_party_waiting_time :
  let mad_hatter_arrival_time := meeting_time / mad_hatter_clock_speed
  let march_hare_arrival_time := meeting_time / march_hare_clock_speed
  march_hare_arrival_time - mad_hatter_arrival_time = 2 := by sorry

end NUMINAMATH_CALUDE_tea_party_waiting_time_l648_64820


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l648_64881

theorem cyclic_sum_inequality (x y z : ℝ) : 
  let a := x + y + z
  ((a - x)^4 + (a - y)^4 + (a - z)^4) + 
  2 * (x^3*y + x^3*z + y^3*x + y^3*z + z^3*x + z^3*y) + 
  4 * (x^2*y^2 + y^2*z^2 + z^2*x^2) + 
  8 * x*y*z*a ≥ 
  ((a - x)^2*(a^2 - x^2) + (a - y)^2*(a^2 - y^2) + (a - z)^2*(a^2 - z^2)) := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l648_64881


namespace NUMINAMATH_CALUDE_store_acquired_twenty_books_l648_64810

/-- The number of additional coloring books acquired by a store -/
def additional_books (initial_stock : ℝ) (books_per_shelf : ℝ) (total_shelves : ℝ) : ℝ :=
  books_per_shelf * total_shelves - initial_stock

/-- Theorem stating that the store acquired 20.0 additional coloring books -/
theorem store_acquired_twenty_books :
  additional_books 40.0 4.0 15 = 20.0 := by
  sorry

end NUMINAMATH_CALUDE_store_acquired_twenty_books_l648_64810


namespace NUMINAMATH_CALUDE_thabos_book_collection_difference_l648_64847

/-- Theorem: Thabo's Book Collection Difference --/
theorem thabos_book_collection_difference :
  ∀ (paperback_fiction paperback_nonfiction hardcover_nonfiction : ℕ),
  -- Total number of books is 180
  paperback_fiction + paperback_nonfiction + hardcover_nonfiction = 180 →
  -- More paperback nonfiction than hardcover nonfiction
  paperback_nonfiction > hardcover_nonfiction →
  -- Twice as many paperback fiction as paperback nonfiction
  paperback_fiction = 2 * paperback_nonfiction →
  -- 30 hardcover nonfiction books
  hardcover_nonfiction = 30 →
  -- Prove: Difference between paperback nonfiction and hardcover nonfiction is 20
  paperback_nonfiction - hardcover_nonfiction = 20 := by
  sorry

end NUMINAMATH_CALUDE_thabos_book_collection_difference_l648_64847


namespace NUMINAMATH_CALUDE_special_day_price_l648_64846

theorem special_day_price (original_price : ℝ) (first_discount_percent : ℝ) (second_discount_percent : ℝ) : 
  original_price = 240 →
  first_discount_percent = 40 →
  second_discount_percent = 25 →
  let first_discounted_price := original_price * (1 - first_discount_percent / 100)
  let special_day_price := first_discounted_price * (1 - second_discount_percent / 100)
  special_day_price = 108 := by
sorry

end NUMINAMATH_CALUDE_special_day_price_l648_64846


namespace NUMINAMATH_CALUDE_min_value_expression_l648_64824

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / (3 * a) + 3 / b ≥ 5 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ b / (3 * a) + 3 / b = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l648_64824


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l648_64826

/-- Given two rectangles with equal areas, where one rectangle measures 15 inches by 20 inches
    and the other has a width of 50 inches, prove that the length of the second rectangle is 6 inches. -/
theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℝ)
    (carol_area jordan_area : ℝ) (h1 : carol_length = 15)
    (h2 : carol_width = 20) (h3 : jordan_width = 50)
    (h4 : carol_area = carol_length * carol_width)
    (h5 : jordan_area = jordan_width * 6)
    (h6 : carol_area = jordan_area) : 6 = jordan_area / jordan_width := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_length_l648_64826


namespace NUMINAMATH_CALUDE_triangle_side_length_l648_64863

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = 2 * Real.pi / 3 →
  b = Real.sqrt 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l648_64863


namespace NUMINAMATH_CALUDE_initial_value_proof_l648_64871

theorem initial_value_proof (final_number : ℕ) (divisor : ℕ) (h1 : final_number = 859560) (h2 : divisor = 456) :
  ∃ (initial_value : ℕ) (added_number : ℕ),
    initial_value + added_number = final_number ∧
    final_number % divisor = 0 ∧
    initial_value = 859376 := by
  sorry

end NUMINAMATH_CALUDE_initial_value_proof_l648_64871


namespace NUMINAMATH_CALUDE_ball_bounces_to_vertex_l648_64865

/-- The height of the rectangle --/
def rectangle_height : ℕ := 10

/-- The vertical distance covered in one bounce --/
def vertical_distance_per_bounce : ℕ := 2

/-- The number of bounces required to reach the top of the rectangle --/
def number_of_bounces : ℕ := rectangle_height / vertical_distance_per_bounce

theorem ball_bounces_to_vertex :
  number_of_bounces = 5 :=
sorry

end NUMINAMATH_CALUDE_ball_bounces_to_vertex_l648_64865


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l648_64821

def johnsons_share : ℕ := 2500
def mikes_shirt_cost : ℕ := 200
def mikes_remaining : ℕ := 800

def mikes_share : ℕ := mikes_remaining + mikes_shirt_cost

def ratio_numerator : ℕ := 2
def ratio_denominator : ℕ := 5

theorem profit_sharing_ratio :
  (mikes_share : ℚ) / johnsons_share = ratio_numerator / ratio_denominator :=
by sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l648_64821


namespace NUMINAMATH_CALUDE_intersection_M_N_l648_64835

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l648_64835


namespace NUMINAMATH_CALUDE_angle_with_complement_one_third_of_supplement_l648_64814

theorem angle_with_complement_one_third_of_supplement (x : Real) : 
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_one_third_of_supplement_l648_64814


namespace NUMINAMATH_CALUDE_oxide_other_element_weight_l648_64815

/-- The atomic weight of the other element in a calcium oxide -/
def atomic_weight_other_element (molecular_weight : ℝ) (calcium_weight : ℝ) : ℝ :=
  molecular_weight - calcium_weight

/-- Theorem stating that the atomic weight of the other element in the oxide is 16 -/
theorem oxide_other_element_weight :
  let molecular_weight : ℝ := 56
  let calcium_weight : ℝ := 40
  atomic_weight_other_element molecular_weight calcium_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_oxide_other_element_weight_l648_64815


namespace NUMINAMATH_CALUDE_passengers_proportion_ge_cars_proportion_passengers_proportion_not_lt_cars_proportion_l648_64853

/-- Represents the distribution of passenger cars -/
structure CarDistribution where
  total : ℕ
  overcrowded : ℕ
  passengers : ℕ
  passengers_overcrowded : ℕ

/-- Definition of an overcrowded car (60 or more passengers) -/
def is_overcrowded (passengers : ℕ) : Prop := passengers ≥ 60

/-- The proportion of overcrowded cars -/
def proportion_overcrowded (d : CarDistribution) : ℚ :=
  d.overcrowded / d.total

/-- The proportion of passengers in overcrowded cars -/
def proportion_passengers_overcrowded (d : CarDistribution) : ℚ :=
  d.passengers_overcrowded / d.passengers

/-- Theorem: The proportion of passengers in overcrowded cars is always
    greater than or equal to the proportion of overcrowded cars -/
theorem passengers_proportion_ge_cars_proportion (d : CarDistribution) :
  proportion_passengers_overcrowded d ≥ proportion_overcrowded d := by
  sorry

/-- Corollary: The proportion of passengers in overcrowded cars cannot be
    less than the proportion of overcrowded cars -/
theorem passengers_proportion_not_lt_cars_proportion (d : CarDistribution) :
  ¬(proportion_passengers_overcrowded d < proportion_overcrowded d) := by
  sorry

end NUMINAMATH_CALUDE_passengers_proportion_ge_cars_proportion_passengers_proportion_not_lt_cars_proportion_l648_64853


namespace NUMINAMATH_CALUDE_kevin_food_expenditure_l648_64831

theorem kevin_food_expenditure (total_budget : ℕ) (samuel_ticket : ℕ) (samuel_food_drinks : ℕ) (kevin_drinks : ℕ) :
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_food_drinks = 6 →
  kevin_drinks = 2 →
  total_budget = samuel_ticket + samuel_food_drinks →
  ∃ (kevin_food : ℕ), total_budget = samuel_ticket + kevin_drinks + kevin_food ∧ kevin_food = 4 :=
by sorry

end NUMINAMATH_CALUDE_kevin_food_expenditure_l648_64831


namespace NUMINAMATH_CALUDE_nantucket_meeting_l648_64885

/-- The number of females attending the meeting in Nantucket --/
def females_attending : ℕ := sorry

/-- The total population of Nantucket --/
def total_population : ℕ := 300

/-- The total number of people attending the meeting --/
def total_attending : ℕ := total_population / 2

/-- The number of males attending the meeting --/
def males_attending : ℕ := 2 * females_attending

theorem nantucket_meeting :
  females_attending = 50 ∧
  females_attending + males_attending = total_attending :=
by sorry

end NUMINAMATH_CALUDE_nantucket_meeting_l648_64885


namespace NUMINAMATH_CALUDE_dog_to_rabbit_age_ratio_l648_64844

/- Define the ages of the animals -/
def cat_age : ℕ := 8
def dog_age : ℕ := 12

/- Define the rabbit's age as half of the cat's age -/
def rabbit_age : ℕ := cat_age / 2

/- Define the ratio of the dog's age to the rabbit's age -/
def age_ratio : ℚ := dog_age / rabbit_age

/- Theorem statement -/
theorem dog_to_rabbit_age_ratio :
  age_ratio = 3 :=
sorry

end NUMINAMATH_CALUDE_dog_to_rabbit_age_ratio_l648_64844


namespace NUMINAMATH_CALUDE_unique_injective_function_l648_64803

/-- Iterate a function n times -/
def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The property that f must satisfy -/
def satisfies_equation (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, iterate f (f a) b * iterate f (f b) a = (f (a + b))^2

/-- The main theorem statement -/
theorem unique_injective_function :
  ∀ f : ℕ → ℕ, Function.Injective f → satisfies_equation f → ∀ x : ℕ, f x = x + 1 := by
  sorry


end NUMINAMATH_CALUDE_unique_injective_function_l648_64803


namespace NUMINAMATH_CALUDE_power_negative_two_a_squared_cubed_l648_64874

theorem power_negative_two_a_squared_cubed (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_two_a_squared_cubed_l648_64874


namespace NUMINAMATH_CALUDE_expression_simplification_l648_64878

theorem expression_simplification (α : ℝ) : 
  (2 * Real.tan (π/4 - α)) / (1 - Real.tan (π/4 - α)^2) * 
  (Real.sin α * Real.cos α) / (Real.cos α^2 - Real.sin α^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l648_64878


namespace NUMINAMATH_CALUDE_total_cost_theorem_l648_64842

def sandwich_cost : ℕ := 3
def soda_cost : ℕ := 2
def num_sandwiches : ℕ := 5
def num_sodas : ℕ := 8

theorem total_cost_theorem : 
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l648_64842


namespace NUMINAMATH_CALUDE_line_slope_l648_64897

theorem line_slope (x y : ℝ) (h : x / 4 + y / 3 = 1) : 
  ∃ m b : ℝ, y = m * x + b ∧ m = -3/4 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l648_64897


namespace NUMINAMATH_CALUDE_not_in_first_quadrant_l648_64822

/-- Proves that the complex number z = (m-2i)/(1+2i) cannot be in the first quadrant for any real m -/
theorem not_in_first_quadrant (m : ℝ) : 
  let z : ℂ := (m - 2*Complex.I) / (1 + 2*Complex.I)
  ¬ (z.re > 0 ∧ z.im > 0) := by
  sorry


end NUMINAMATH_CALUDE_not_in_first_quadrant_l648_64822


namespace NUMINAMATH_CALUDE_students_guinea_pigs_difference_l648_64883

theorem students_guinea_pigs_difference : 
  let students_per_class : ℕ := 25
  let guinea_pigs_per_class : ℕ := 3
  let num_classes : ℕ := 6
  let total_students : ℕ := students_per_class * num_classes
  let total_guinea_pigs : ℕ := guinea_pigs_per_class * num_classes
  total_students - total_guinea_pigs = 132 :=
by
  sorry


end NUMINAMATH_CALUDE_students_guinea_pigs_difference_l648_64883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_10_l648_64860

def arithmetic_sequence (a₁ d n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_mod (l : List ℕ) (m : ℕ) : ℕ :=
  (l.foldl (· + ·) 0) % m

theorem arithmetic_sequence_sum_mod_10 :
  let seq := arithmetic_sequence 7 7 13
  sum_mod seq 10 = 7 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_10_l648_64860


namespace NUMINAMATH_CALUDE_cartesian_points_proof_l648_64848

/-- Given points P and Q in the Cartesian coordinate system, prove cos 2θ and sin(α + β) -/
theorem cartesian_points_proof (θ α β : Real) : 
  let P : Real × Real := (1/2, Real.cos θ ^ 2)
  let Q : Real × Real := (Real.sin θ ^ 2, -1)
  (P.1 * Q.1 + P.2 * Q.2 = -1/2) → 
  (Real.cos (2 * θ) = 1/3 ∧ 
   Real.sin (α + β) = -Real.sqrt 10 / 10) := by
sorry

end NUMINAMATH_CALUDE_cartesian_points_proof_l648_64848


namespace NUMINAMATH_CALUDE_parallelogram_vector_subtraction_l648_64800

-- Define a parallelogram ABCD
structure Parallelogram (V : Type*) [AddCommGroup V] :=
  (A B C D : V)
  (parallelogram_condition : A - B = D - C)

-- Define the theorem
theorem parallelogram_vector_subtraction 
  {V : Type*} [AddCommGroup V] (ABCD : Parallelogram V) :
  ABCD.A - ABCD.C - (ABCD.B - ABCD.C) = ABCD.D - ABCD.C :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_vector_subtraction_l648_64800


namespace NUMINAMATH_CALUDE_linear_system_solution_l648_64806

theorem linear_system_solution (m : ℕ) (x y : ℝ) : 
  (2 * x - y = 4 * m - 5) →
  (x + 4 * y = -7 * m + 2) →
  (x + y > -3) →
  (m = 0 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l648_64806


namespace NUMINAMATH_CALUDE_find_m_l648_64898

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem find_m (z m : ℂ) (h1 : det z i m i = 1 - 2*I) (h2 : z.re = 0) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l648_64898


namespace NUMINAMATH_CALUDE_tank_length_proof_l648_64827

/-- Proves that a rectangular tank with given dimensions and plastering cost has a specific length -/
theorem tank_length_proof (width depth cost_per_sqm total_cost : ℝ) 
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost_per_sqm : cost_per_sqm = 0.70)
  (h_total_cost : total_cost = 520.8)
  : ∃ length : ℝ, 
    length = 25 ∧ 
    total_cost = (2 * width * depth + 2 * length * depth + width * length) * cost_per_sqm :=
by sorry

end NUMINAMATH_CALUDE_tank_length_proof_l648_64827


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l648_64828

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 - 12*x + 32

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define the conditions of the problem
def inscribedRectangle (r : Rectangle) : Prop :=
  ∃ t : ℝ,
    r.base = 2*t ∧
    r.height = (2*t)/3 ∧
    parabola (6 - t) = r.height ∧
    t > 0

-- The theorem to prove
theorem inscribed_rectangle_area :
  ∀ r : Rectangle, inscribedRectangle r →
    r.base * r.height = 91 + 25 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l648_64828


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l648_64864

/-- A parabola with equation y = (x - h)^2 + k, where (h, k) is the vertex. -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The focus of a parabola. -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem: The focus of the parabola y = (x - 3)^2 is at (3, 1/8). -/
theorem focus_of_specific_parabola :
  let p : Parabola := { h := 3, k := 0 }
  focus p = (3, 1/8) := by sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l648_64864


namespace NUMINAMATH_CALUDE_other_x_intercept_is_seven_l648_64875

/-- A quadratic function with vertex (4, -3) and one x-intercept at (1, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := -3
  intercept_x : ℝ := 1

/-- The x-coordinate of the other x-intercept of the quadratic function -/
def other_x_intercept (f : QuadraticFunction) : ℝ := 7

/-- Theorem stating that the x-coordinate of the other x-intercept is 7 -/
theorem other_x_intercept_is_seven (f : QuadraticFunction) :
  other_x_intercept f = 7 := by sorry

end NUMINAMATH_CALUDE_other_x_intercept_is_seven_l648_64875


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l648_64876

theorem fitness_center_membership_ratio 
  (f m : ℕ) -- f: number of female members, m: number of male members
  (hf : f > 0) -- assume there's at least one female member
  (hm : m > 0) -- assume there's at least one male member
  (h_avg_female : (50 * f) / (f + m) = 50) -- average age of female members is 50
  (h_avg_male : (30 * m) / (f + m) = 30)   -- average age of male members is 30
  (h_avg_total : (50 * f + 30 * m) / (f + m) = 35) -- average age of all members is 35
  : f / m = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l648_64876


namespace NUMINAMATH_CALUDE_no_valid_numbers_l648_64896

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  first : Nat
  middle : Nat
  last : Nat
  first_digit : first < 10
  middle_digit : middle < 10
  last_digit : last < 10
  three_digits : first ≠ 0

/-- Checks if a number is not divisible by 3 -/
def notDivisibleByThree (n : ThreeDigitNumber) : Prop :=
  (100 * n.first + 10 * n.middle + n.last) % 3 ≠ 0

/-- Checks if the sum of digits is less than 22 -/
def sumLessThan22 (n : ThreeDigitNumber) : Prop :=
  n.first + n.middle + n.last < 22

/-- Checks if the middle digit is twice the first digit -/
def middleTwiceFirst (n : ThreeDigitNumber) : Prop :=
  n.middle = 2 * n.first

theorem no_valid_numbers :
  ¬ ∃ (n : ThreeDigitNumber),
    notDivisibleByThree n ∧
    sumLessThan22 n ∧
    middleTwiceFirst n :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l648_64896


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l648_64833

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a > 0) → (0 ≤ a ∧ a ≤ 4) ∧
  ¬(0 ≤ a ∧ a ≤ 4 → ∀ x : ℝ, x^2 + a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l648_64833


namespace NUMINAMATH_CALUDE_intersection_segment_length_l648_64867

/-- The length of the line segment AB, where A and B are the intersection points
    of the line y = √3 x and the circle (x + √3)² + (y + 2)² = 1, is equal to √3. -/
theorem intersection_segment_length :
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  let circle := {p : ℝ × ℝ | (p.1 + Real.sqrt 3)^2 + (p.2 + 2)^2 = 1}
  let intersection := {p : ℝ × ℝ | p ∈ line ∩ circle}
  ∃ A B : ℝ × ℝ, A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l648_64867


namespace NUMINAMATH_CALUDE_least_multiple_24_above_450_l648_64873

theorem least_multiple_24_above_450 : 
  ∀ n : ℕ, n > 0 ∧ 24 ∣ n ∧ n > 450 → n ≥ 456 := by sorry

end NUMINAMATH_CALUDE_least_multiple_24_above_450_l648_64873


namespace NUMINAMATH_CALUDE_divisibility_condition_l648_64850

theorem divisibility_condition (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2*y - 1) ↔ 
  ((x = 1 ∧ y > 0) ∨ 
   (∃ t : ℕ, t > 0 ∧ x = 2*t - 1 ∧ y = t) ∨ 
   (x = 3 ∧ y = 8) ∨ 
   (x = 5 ∧ y = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l648_64850


namespace NUMINAMATH_CALUDE_function_composition_equality_l648_64890

/-- Given a function f and a condition on f[g(x)], prove the form of g(x) -/
theorem function_composition_equality 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f : ∀ x, f x = 3 * x - 1) 
  (h_fg : ∀ x, f (g x) = 2 * x + 3) : 
  ∀ x, g x = (2/3) * x + (4/3) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l648_64890


namespace NUMINAMATH_CALUDE_car_distance_theorem_l648_64802

/-- Given a car traveling at 160 km/h for 5 hours, the distance covered is 800 km. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 160 ∧ time = 5 → distance = speed * time → distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l648_64802


namespace NUMINAMATH_CALUDE_lcm_48_180_l648_64855

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l648_64855


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l648_64879

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- The theorem stating the area of the inscribed rectangle R -/
theorem area_of_inscribed_rectangle (largerSquare : Square) 
  (smallerSquare : Square) 
  (rect1 rect2 : Rectangle) :
  smallerSquare.side = 2 ∧ 
  rect1.width = 2 ∧ rect1.height = 4 ∧
  rect2.width = 1 ∧ rect2.height = 2 ∧
  largerSquare.side = 6 →
  largerSquare.area - (smallerSquare.area + rect1.area + rect2.area) = 22 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l648_64879


namespace NUMINAMATH_CALUDE_max_remainder_problem_l648_64809

theorem max_remainder_problem :
  ∃ (n : ℕ) (r : ℕ),
    2013 ≤ n ∧ n ≤ 2156 ∧
    n % 5 = r ∧ n % 11 = r ∧ n % 13 = r ∧
    r ≤ 4 ∧
    ∀ (m : ℕ) (s : ℕ),
      2013 ≤ m ∧ m ≤ 2156 ∧
      m % 5 = s ∧ m % 11 = s ∧ m % 13 = s →
      s ≤ r :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_problem_l648_64809


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l648_64825

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 16 →
    pond_side = 4 →
    (pond_side^2) / (field_length * field_width) = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l648_64825


namespace NUMINAMATH_CALUDE_zero_is_root_of_polynomial_l648_64818

theorem zero_is_root_of_polynomial : ∃ (x : ℝ), 12 * x^4 + 38 * x^3 - 51 * x^2 + 40 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_root_of_polynomial_l648_64818


namespace NUMINAMATH_CALUDE_power_equality_l648_64804

theorem power_equality : 32^4 * 4^5 = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l648_64804


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l648_64817

/-- Proves the time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 165) 
  (h2 : train_speed_kmph = 72) 
  (h3 : bridge_length = 660) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 41.25 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l648_64817


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l648_64870

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_increasing_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l648_64870


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l648_64868

theorem smallest_number_of_students (n : ℕ) : 
  n > 0 ∧
  (n : ℚ) * (75 : ℚ) / 100 = ↑(n - (n / 4 : ℕ)) ∧
  (n / 40 : ℕ) = (n / 4 : ℕ) * 10 / 100 ∧
  (33 * n / 200 : ℕ) = ((11 * n / 100 : ℕ) * 3 / 2 : ℕ) ∧
  ∀ m : ℕ, m > 0 ∧ 
    (m : ℚ) * (75 : ℚ) / 100 = ↑(m - (m / 4 : ℕ)) ∧
    (m / 40 : ℕ) = (m / 4 : ℕ) * 10 / 100 ∧
    (33 * m / 200 : ℕ) = ((11 * m / 100 : ℕ) * 3 / 2 : ℕ) →
    m ≥ n →
  n = 200 := by sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l648_64868


namespace NUMINAMATH_CALUDE_parametric_eq_represents_line_l648_64829

/-- Prove that the given parametric equations represent the line x + y - 2 = 0 --/
theorem parametric_eq_represents_line :
  ∀ (t : ℝ), (3 + t) + (1 - t) - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parametric_eq_represents_line_l648_64829


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base5_max_digit_sum_attainable_l648_64861

/-- Given a positive integer n, returns the sum of its digits in base 5 representation -/
def sumOfDigitsBase5 (n : ℕ) : ℕ := sorry

/-- The greatest possible sum of digits in base 5 for integers less than 3139 -/
def maxDigitSum : ℕ := 16

theorem greatest_digit_sum_base5 :
  ∀ n : ℕ, n > 0 ∧ n < 3139 → sumOfDigitsBase5 n ≤ maxDigitSum :=
by sorry

theorem max_digit_sum_attainable :
  ∃ n : ℕ, n > 0 ∧ n < 3139 ∧ sumOfDigitsBase5 n = maxDigitSum :=
by sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base5_max_digit_sum_attainable_l648_64861


namespace NUMINAMATH_CALUDE_cost_effective_purchase_anton_offer_is_best_l648_64887

/-- Represents a shareholder in the company -/
structure Shareholder where
  name : String
  shares : Nat
  sellPrice : Rat

/-- Represents the company and its shareholders -/
structure Company where
  totalShares : Nat
  sharePrice : Nat
  shareholders : List Shareholder

/-- Calculates the cost of buying shares from a shareholder -/
def buyCost (shareholder : Shareholder) : Rat :=
  shareholder.shares * shareholder.sellPrice

/-- Checks if a shareholder has enough shares to be the largest -/
def isLargestShareholder (company : Company) (shares : Nat) : Prop :=
  ∀ s : Shareholder, s ∈ company.shareholders → shares > s.shares

/-- The main theorem to prove -/
theorem cost_effective_purchase (company : Company) : Prop :=
  let arina : Shareholder := { name := "Arina", shares := 90001, sellPrice := 10 }
  let anton : Shareholder := { name := "Anton", shares := 15000, sellPrice := 14 }
  let arinaNewShares := arina.shares + anton.shares
  isLargestShareholder company arinaNewShares ∧
  ∀ s : Shareholder, s ∈ company.shareholders → s.name ≠ "Arina" →
    buyCost anton ≤ buyCost s ∨ ¬(isLargestShareholder company (arina.shares + s.shares))

/-- The company instance with given conditions -/
def jscCompany : Company := {
  totalShares := 300000,
  sharePrice := 10,
  shareholders := [
    { name := "Arina", shares := 90001, sellPrice := 10 },
    { name := "Maxim", shares := 104999, sellPrice := 11 },
    { name := "Inga", shares := 30000, sellPrice := 12.5 },
    { name := "Yuri", shares := 30000, sellPrice := 11.5 },
    { name := "Yulia", shares := 30000, sellPrice := 13 },
    { name := "Anton", shares := 15000, sellPrice := 14 }
  ]
}

/-- The main theorem applied to our specific company -/
theorem anton_offer_is_best : cost_effective_purchase jscCompany := by
  sorry


end NUMINAMATH_CALUDE_cost_effective_purchase_anton_offer_is_best_l648_64887


namespace NUMINAMATH_CALUDE_trapezoid_sides_l648_64884

/-- A rectangular trapezoid with an inscribed circle -/
structure RectangularTrapezoid (r : ℝ) where
  /-- The radius of the inscribed circle -/
  radius : r > 0
  /-- The shorter base of the trapezoid -/
  short_base : ℝ
  /-- The longer base of the trapezoid -/
  long_base : ℝ
  /-- One of the non-parallel sides of the trapezoid -/
  side1 : ℝ
  /-- The other non-parallel side of the trapezoid -/
  side2 : ℝ
  /-- The shorter base is equal to 4r/3 -/
  short_base_eq : short_base = 4*r/3
  /-- The circle is inscribed, so one non-parallel side equals the diameter -/
  side1_eq_diameter : side1 = 2*r
  /-- Property of trapezoids with an inscribed circle -/
  inscribed_circle_property : side1 + long_base = short_base + side2

/-- Theorem: The sides of the rectangular trapezoid with an inscribed circle of radius r 
    and shorter base 4r/3 are 4r, 10r/3, and 2r -/
theorem trapezoid_sides (r : ℝ) (t : RectangularTrapezoid r) : 
  t.short_base = 4*r/3 ∧ t.long_base = 10*r/3 ∧ t.side1 = 2*r ∧ t.side2 = 8*r/3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l648_64884


namespace NUMINAMATH_CALUDE_correct_operation_result_l648_64889

theorem correct_operation_result (x : ℝ) : (x - 9) / 3 = 43 → (x - 3) / 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_result_l648_64889


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l648_64830

/-- A sequence of 8 positive real numbers -/
def Sequence := Fin 8 → ℝ

/-- Predicate to check if a sequence is positive -/
def is_positive (s : Sequence) : Prop :=
  ∀ i, s i > 0

/-- Predicate to check if a sequence is geometric -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 7, s (i + 1) = q * s i

theorem sufficient_but_not_necessary (s : Sequence) 
  (h_pos : is_positive s) :
  (s 0 + s 7 < s 3 + s 4 → ¬is_geometric s) ∧
  ∃ s' : Sequence, is_positive s' ∧ ¬is_geometric s' ∧ s' 0 + s' 7 ≥ s' 3 + s' 4 :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l648_64830


namespace NUMINAMATH_CALUDE_fair_walking_distance_l648_64886

theorem fair_walking_distance (total_distance : ℝ) (short_segment : ℝ) 
  (h1 : total_distance = 0.75)
  (h2 : short_segment = 0.08)
  (h3 : ∃ x : ℝ, total_distance = 2 * x + short_segment) :
  ∃ x : ℝ, x = 0.335 ∧ total_distance = 2 * x + short_segment :=
sorry

end NUMINAMATH_CALUDE_fair_walking_distance_l648_64886


namespace NUMINAMATH_CALUDE_mary_saw_36_snakes_l648_64854

/-- The total number of snakes Mary saw -/
def total_snakes (breeding_balls : ℕ) (snakes_per_ball : ℕ) (additional_pairs : ℕ) : ℕ :=
  breeding_balls * snakes_per_ball + additional_pairs * 2

/-- Theorem stating that Mary saw 36 snakes in total -/
theorem mary_saw_36_snakes :
  total_snakes 3 8 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_mary_saw_36_snakes_l648_64854


namespace NUMINAMATH_CALUDE_equation_solutions_range_l648_64832

-- Define the equation
def equation (x a : ℝ) : Prop := |2^x - a| = 1

-- Define the condition of having two unequal real solutions
def has_two_unequal_solutions (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ equation x a ∧ equation y a

-- State the theorem
theorem equation_solutions_range :
  ∀ a : ℝ, has_two_unequal_solutions a ↔ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_range_l648_64832


namespace NUMINAMATH_CALUDE_chef_initial_potatoes_l648_64882

/-- Represents the number of fries that can be made from one potato -/
def fries_per_potato : ℕ := 25

/-- Represents the total number of fries needed -/
def total_fries_needed : ℕ := 200

/-- Represents the number of potatoes leftover after making the required fries -/
def leftover_potatoes : ℕ := 7

/-- Calculates the initial number of potatoes the chef had -/
def initial_potatoes : ℕ := (total_fries_needed / fries_per_potato) + leftover_potatoes

/-- Proves that the initial number of potatoes is 15 -/
theorem chef_initial_potatoes :
  initial_potatoes = 15 :=
by sorry

end NUMINAMATH_CALUDE_chef_initial_potatoes_l648_64882


namespace NUMINAMATH_CALUDE_pyramid_display_sum_l648_64813

/-- Proves that the sum of an arithmetic sequence with given parameters is 255 -/
theorem pyramid_display_sum : 
  ∀ (a₁ aₙ d n : ℕ),
  a₁ = 12 →
  aₙ = 39 →
  d = 3 →
  aₙ = a₁ + (n - 1) * d →
  (n : ℕ) * (a₁ + aₙ) / 2 = 255 :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_display_sum_l648_64813


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l648_64841

/-- Given two parallel vectors a and b in R², prove that their linear combination results in (14, 7) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (3 • a + 2 • b : Fin 2 → ℝ) = ![14, 7] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l648_64841


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l648_64872

theorem max_value_cos_sin (a b : ℝ) : 
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ ≤ Real.sqrt (a^2 + b^2)) ∧ 
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ = Real.sqrt (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l648_64872


namespace NUMINAMATH_CALUDE_tina_money_left_is_40_l648_64858

/-- Calculates the amount of money Tina has left after savings and expenses -/
def tina_money_left (june_savings july_savings august_savings book_expense shoe_expense : ℕ) : ℕ :=
  (june_savings + july_savings + august_savings) - (book_expense + shoe_expense)

/-- Theorem stating that Tina has $40 left given her savings and expenses -/
theorem tina_money_left_is_40 :
  tina_money_left 27 14 21 5 17 = 40 := by
  sorry

#eval tina_money_left 27 14 21 5 17

end NUMINAMATH_CALUDE_tina_money_left_is_40_l648_64858


namespace NUMINAMATH_CALUDE_f_monotonic_increase_interval_l648_64837

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + cos (2 * x - π / 3)

theorem f_monotonic_increase_interval :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo (k * π - π / 3) (k * π + π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonic_increase_interval_l648_64837


namespace NUMINAMATH_CALUDE_largest_base5_five_digit_in_base10_l648_64834

def largest_base5_five_digit : ℕ := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_five_digit_in_base10 : 
  largest_base5_five_digit = 3124 := by sorry

end NUMINAMATH_CALUDE_largest_base5_five_digit_in_base10_l648_64834


namespace NUMINAMATH_CALUDE_min_participants_correct_l648_64838

/-- Represents a participant in the race -/
inductive Participant
| Andrei
| Dima
| Lenya
| Other

/-- Represents the race results -/
def RaceResult := List Participant

/-- Checks if the race result satisfies the given conditions -/
def satisfiesConditions (result : RaceResult) : Prop :=
  let n := result.length
  ∃ (a d l : Nat),
    a + 1 + 2 * a = n ∧
    d + 1 + 3 * d = n ∧
    l + 1 + 4 * l = n ∧
    a ≠ d ∧ a ≠ l ∧ d ≠ l

/-- The minimum number of participants in the race -/
def minParticipants : Nat := 61

theorem min_participants_correct :
  ∃ (result : RaceResult),
    result.length = minParticipants ∧
    satisfiesConditions result ∧
    ∀ (result' : RaceResult),
      satisfiesConditions result' →
      result'.length ≥ minParticipants :=
sorry

end NUMINAMATH_CALUDE_min_participants_correct_l648_64838


namespace NUMINAMATH_CALUDE_complex_modulus_l648_64805

theorem complex_modulus (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l648_64805


namespace NUMINAMATH_CALUDE_wrench_sales_profit_l648_64839

theorem wrench_sales_profit (selling_price : ℝ) : 
  selling_price > 0 →
  let profit_percent : ℝ := 0.25
  let loss_percent : ℝ := 0.15
  let cost_price1 : ℝ := selling_price / (1 + profit_percent)
  let cost_price2 : ℝ := selling_price / (1 - loss_percent)
  let total_cost : ℝ := cost_price1 + cost_price2
  let total_revenue : ℝ := 2 * selling_price
  let net_gain : ℝ := total_revenue - total_cost
  net_gain / selling_price = 0.028 :=
by sorry

end NUMINAMATH_CALUDE_wrench_sales_profit_l648_64839
