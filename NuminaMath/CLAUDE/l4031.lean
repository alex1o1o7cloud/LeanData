import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l4031_403111

/-- Three numbers forming an arithmetic sequence -/
structure ArithmeticSequence :=
  (a : ℝ)
  (d : ℝ)

/-- The sum of three numbers in an arithmetic sequence -/
def sum (seq : ArithmeticSequence) : ℝ :=
  (seq.a - seq.d) + seq.a + (seq.a + seq.d)

/-- The sum of squares of three numbers in an arithmetic sequence -/
def sumOfSquares (seq : ArithmeticSequence) : ℝ :=
  (seq.a - seq.d)^2 + seq.a^2 + (seq.a + seq.d)^2

/-- Theorem: If three numbers form an arithmetic sequence with a sum of 15 and a sum of squares of 83,
    then these numbers are either 3, 5, 7 or 7, 5, 3 -/
theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) :
  sum seq = 15 ∧ sumOfSquares seq = 83 →
  (seq.a = 5 ∧ (seq.d = 2 ∨ seq.d = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l4031_403111


namespace NUMINAMATH_CALUDE_quadratic_properties_l4031_403164

def f (x : ℝ) := -x^2 + 3*x + 1

theorem quadratic_properties :
  (∀ x y, x < y → f y < f x) ∧ 
  (3/2 = -(-3)/(2*(-1))) ∧
  (∀ x y, x < y → y < 3/2 → f x < f y) ∧
  (∀ x, f x = 0 → x < 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l4031_403164


namespace NUMINAMATH_CALUDE_sqrt_less_implies_less_l4031_403113

theorem sqrt_less_implies_less (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a < Real.sqrt b → a < b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_implies_less_l4031_403113


namespace NUMINAMATH_CALUDE_gcd_79625_51575_l4031_403170

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_79625_51575_l4031_403170


namespace NUMINAMATH_CALUDE_empty_set_proof_l4031_403166

theorem empty_set_proof : {x : ℝ | x^2 + 1 = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l4031_403166


namespace NUMINAMATH_CALUDE_company_manager_fraction_l4031_403106

/-- Fraction of employees who are managers -/
def manager_fraction (total_employees : ℕ) (total_managers : ℕ) : ℚ :=
  total_managers / total_employees

theorem company_manager_fraction :
  ∀ (total_employees : ℕ) (total_managers : ℕ) (male_employees : ℕ) (male_managers : ℕ),
    total_employees > 0 →
    male_employees > 0 →
    total_employees = 625 + male_employees →
    total_managers = 250 + male_managers →
    manager_fraction total_employees total_managers = manager_fraction 625 250 →
    manager_fraction total_employees total_managers = manager_fraction male_employees male_managers →
    manager_fraction total_employees total_managers = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_company_manager_fraction_l4031_403106


namespace NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l4031_403114

/-- The largest five-digit number in base 5 -/
def largest_base5_5digit : ℕ := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_5digit_in_base10 : largest_base5_5digit = 3124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_5digit_in_base10_l4031_403114


namespace NUMINAMATH_CALUDE_unique_quadruple_l4031_403142

theorem unique_quadruple :
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a + b + c + d = 2 ∧
    a^2 + b^2 + c^2 + d^2 = 3 ∧
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 :=
by sorry

end NUMINAMATH_CALUDE_unique_quadruple_l4031_403142


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_parallel_implies_perpendicular_planes_l4031_403167

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_implies_parallel 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel_planes α β :=
sorry

-- Theorem 2
theorem perpendicular_parallel_implies_perpendicular_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_parallel_implies_perpendicular_planes_l4031_403167


namespace NUMINAMATH_CALUDE_cindy_envelopes_l4031_403153

theorem cindy_envelopes (friends : ℕ) (envelopes_per_friend : ℕ) (envelopes_left : ℕ) :
  friends = 5 →
  envelopes_per_friend = 3 →
  envelopes_left = 22 →
  friends * envelopes_per_friend + envelopes_left = 37 :=
by sorry

end NUMINAMATH_CALUDE_cindy_envelopes_l4031_403153


namespace NUMINAMATH_CALUDE_water_bottles_sold_l4031_403181

/-- The number of water bottles sold in a store, given the prices and quantities of other drinks --/
theorem water_bottles_sold : ℕ := by
  -- Define the prices of drinks
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1

  -- Define the quantities of cola and juice sold
  let cola_quantity : ℕ := 15
  let juice_quantity : ℕ := 12

  -- Define the total earnings
  let total_earnings : ℚ := 88

  -- Define the function to calculate the number of water bottles
  let water_bottles (x : ℕ) : Prop :=
    cola_price * cola_quantity + juice_price * juice_quantity + water_price * x = total_earnings

  -- Prove that the number of water bottles sold is 25
  have h : water_bottles 25 := by sorry

  exact 25

end NUMINAMATH_CALUDE_water_bottles_sold_l4031_403181


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l4031_403158

theorem certain_number_subtraction (x : ℝ) (y : ℝ) : 
  (3 * x = (y - x) + 4) → (x = 5) → (y = 16) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l4031_403158


namespace NUMINAMATH_CALUDE_probability_is_half_l4031_403122

/-- The probability of drawing either a red or blue marble from a bag -/
def probability_red_or_blue (red : ℕ) (blue : ℕ) (yellow : ℕ) : ℚ :=
  (red + blue : ℚ) / (red + blue + yellow)

/-- Theorem: The probability of drawing either a red or blue marble
    from a bag containing 3 red, 2 blue, and 5 yellow marbles is 1/2 -/
theorem probability_is_half :
  probability_red_or_blue 3 2 5 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l4031_403122


namespace NUMINAMATH_CALUDE_investment_strategy_l4031_403129

/-- Represents the investment and profit parameters for a manufacturing company's production lines. -/
structure ProductionParameters where
  initialInvestmentA : ℝ  -- Initial investment in production line A (in million yuan)
  profitRateA : ℝ         -- Profit rate for production line A (profit per 10,000 yuan invested)
  investmentReduction : ℝ -- Reduction in investment for A (in million yuan)
  profitIncreaseRate : ℝ  -- Rate of profit increase for A (as a percentage)
  profitRateB : ℝ → ℝ     -- Profit rate function for production line B
  a : ℝ                   -- Parameter a for production line B's profit rate

/-- The main theorem about the manufacturing company's investment strategy. -/
theorem investment_strategy 
  (params : ProductionParameters) 
  (h_initialInvestmentA : params.initialInvestmentA = 50)
  (h_profitRateA : params.profitRateA = 1.5)
  (h_profitIncreaseRate : params.profitIncreaseRate = 0.005)
  (h_profitRateB : params.profitRateB = fun x => 1.5 * (params.a - 0.013 * x))
  (h_a_positive : params.a > 0) :
  (∃ x_range : Set ℝ, x_range = {x | 0 < x ∧ x ≤ 300} ∧ 
    ∀ x ∈ x_range, 
      (params.initialInvestmentA - x) * params.profitRateA * (1 + x * params.profitIncreaseRate) ≥ 
      params.initialInvestmentA * params.profitRateA) ∧
  (∃ a_max : ℝ, a_max = 5.5 ∧
    ∀ x > 0, x * params.profitRateB x ≤ 
      (params.initialInvestmentA - x) * params.profitRateA * (1 + x * params.profitIncreaseRate) ∧
    params.a ≤ a_max) := by
  sorry

end NUMINAMATH_CALUDE_investment_strategy_l4031_403129


namespace NUMINAMATH_CALUDE_opposite_of_neg_2023_l4031_403128

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem: The opposite of -2023 is 2023. -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_2023_l4031_403128


namespace NUMINAMATH_CALUDE_combination_problem_l4031_403124

theorem combination_problem (n : ℕ) : 
  n * (n - 1) = 42 → n.choose 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_combination_problem_l4031_403124


namespace NUMINAMATH_CALUDE_counterexample_exists_l4031_403139

theorem counterexample_exists : ∃ (a b c : ℝ), a > b ∧ ¬(a * c > b * c) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4031_403139


namespace NUMINAMATH_CALUDE_equation_solutions_l4031_403195

theorem equation_solutions : 
  (∃ (x : ℝ), (x + 8) * (x + 1) = -12 ↔ (x = -4 ∨ x = -5)) ∧
  (∃ (x : ℝ), (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ (x = 3/2 ∨ x = 4)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4031_403195


namespace NUMINAMATH_CALUDE_distance_on_number_line_l4031_403159

theorem distance_on_number_line (A B C : ℝ) : 
  (|B - A| = 5) → (|C - B| = 3) → (|C - A| = 2 ∨ |C - A| = 8) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l4031_403159


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l4031_403126

theorem shaded_area_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h1 : total_squares = 5) 
  (h2 : shaded_squares = 2) : 
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l4031_403126


namespace NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l4031_403199

/-- Calculates the number of seashells in Jar A after n weeks -/
def shellsInJarA (n : ℕ) : ℕ := 50 + 20 * n

/-- Calculates the number of seashells in Jar B after n weeks -/
def shellsInJarB (n : ℕ) : ℕ := 30 * (2 ^ n)

/-- The total number of seashells in both jars after n weeks -/
def totalShells (n : ℕ) : ℕ := shellsInJarA n + shellsInJarB n

theorem seashell_count_after_six_weeks :
  totalShells 6 = 1110 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_after_six_weeks_l4031_403199


namespace NUMINAMATH_CALUDE_small_gardens_and_pepper_seeds_l4031_403168

/-- Represents the number of small gardens for each vegetable type -/
structure SmallGardens where
  tomatoes : ℕ
  lettuce : ℕ
  peppers : ℕ

/-- Represents the seed requirements for each vegetable type -/
structure SeedRequirements where
  tomatoes : ℕ
  lettuce : ℕ
  peppers : ℕ

def total_seeds : ℕ := 42
def big_garden_seeds : ℕ := 36

def small_gardens : SmallGardens :=
  { tomatoes := 3
  , lettuce := 2
  , peppers := 0 }

def seed_requirements : SeedRequirements :=
  { tomatoes := 4
  , lettuce := 3
  , peppers := 2 }

def remaining_seeds : ℕ := total_seeds - big_garden_seeds

theorem small_gardens_and_pepper_seeds :
  (small_gardens.tomatoes + small_gardens.lettuce + small_gardens.peppers = 5) ∧
  (small_gardens.peppers * seed_requirements.peppers = 0) :=
by sorry

end NUMINAMATH_CALUDE_small_gardens_and_pepper_seeds_l4031_403168


namespace NUMINAMATH_CALUDE_equation_roots_and_solution_l4031_403188

-- Define the equation
def equation (x p : ℝ) : Prop :=
  Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x

-- Theorem statement
theorem equation_roots_and_solution :
  ∀ p : ℝ, (∃ x : ℝ, equation x p) ↔ (0 ≤ p ∧ p ≤ 4/3) ∧
  ∀ p : ℝ, 0 ≤ p → p ≤ 4/3 → equation 1 p :=
sorry

end NUMINAMATH_CALUDE_equation_roots_and_solution_l4031_403188


namespace NUMINAMATH_CALUDE_repeating_decimal_568_l4031_403184

/-- The repeating decimal 0.568568568... is equal to the fraction 568/999 -/
theorem repeating_decimal_568 : 
  (∑' n : ℕ, (568 : ℚ) / 1000^(n+1)) = 568 / 999 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_568_l4031_403184


namespace NUMINAMATH_CALUDE_chess_tournament_l4031_403144

/-- Represents the number of participants from each city --/
structure Participants where
  moscow : ℕ
  saintPetersburg : ℕ
  kazan : ℕ

/-- Represents the number of games played between cities --/
structure Games where
  moscowSaintPetersburg : ℕ
  moscowKazan : ℕ
  saintPetersburgKazan : ℕ

/-- The theorem stating the conditions and the result to be proved --/
theorem chess_tournament (p : Participants) (g : Games) : 
  p.moscow * 9 = p.saintPetersburg * 6 ∧ 
  p.moscow * g.moscowKazan = p.kazan * 8 ∧ 
  p.saintPetersburg * 2 = p.kazan * 6 →
  g.moscowKazan = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_l4031_403144


namespace NUMINAMATH_CALUDE_exists_removable_piece_l4031_403109

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  pieces : Finset (Fin 8 × Fin 8)
  piece_count : pieces.card = 15
  row_coverage : ∀ r : Fin 8, ∃ c : Fin 8, (r, c) ∈ pieces
  col_coverage : ∀ c : Fin 8, ∃ r : Fin 8, (r, c) ∈ pieces

/-- Theorem stating that there always exists a removable piece -/
theorem exists_removable_piece (config : ChessboardConfig) :
  ∃ p ∈ config.pieces, 
    let remaining := config.pieces.erase p
    (∀ r : Fin 8, ∃ c : Fin 8, (r, c) ∈ remaining) ∧
    (∀ c : Fin 8, ∃ r : Fin 8, (r, c) ∈ remaining) :=
  sorry

end NUMINAMATH_CALUDE_exists_removable_piece_l4031_403109


namespace NUMINAMATH_CALUDE_complement_of_union_in_U_l4031_403121

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union_in_U : (U \ (A ∪ B)) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_in_U_l4031_403121


namespace NUMINAMATH_CALUDE_all_statements_imply_not_all_true_l4031_403101

theorem all_statements_imply_not_all_true (p q r : Prop) :
  -- Statement 1
  ((p ∧ q ∧ ¬r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 2
  ((p ∧ ¬q ∧ r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 3
  ((¬p ∧ q ∧ ¬r) → ¬(p ∧ q ∧ r)) ∧
  -- Statement 4
  ((¬p ∧ ¬q ∧ ¬r) → ¬(p ∧ q ∧ r)) :=
by sorry


end NUMINAMATH_CALUDE_all_statements_imply_not_all_true_l4031_403101


namespace NUMINAMATH_CALUDE_sum_of_b_and_c_l4031_403179

theorem sum_of_b_and_c (a b c d : ℝ) 
  (h1 : a + b = 14)
  (h2 : c + d = 3)
  (h3 : a + d = 8) :
  b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_b_and_c_l4031_403179


namespace NUMINAMATH_CALUDE_cupcakes_sold_l4031_403116

/-- Proves that Carol sold 9 cupcakes given the initial and final conditions -/
theorem cupcakes_sold (initial : ℕ) (made_after : ℕ) (final : ℕ) : 
  initial = 30 → made_after = 28 → final = 49 → 
  ∃ (sold : ℕ), sold = 9 ∧ initial - sold + made_after = final := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_sold_l4031_403116


namespace NUMINAMATH_CALUDE_OPRQ_shapes_l4031_403152

-- Define the points
def O : ℝ × ℝ := (0, 0)
def P (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, y₁)
def Q (x₂ y₂ : ℝ) : ℝ × ℝ := (x₂, y₂)
def R (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := (x₁ - x₂, y₁ - y₂)

-- Define the quadrilateral OPRQ
def OPRQ (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {O, P x₁ y₁, Q x₂ y₂, R x₁ y₁ x₂ y₂}

-- Define conditions for parallelogram, straight line, and trapezoid
def isParallelogram (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  P x₁ y₁ + Q x₂ y₂ = R x₁ y₁ x₂ y₂

def isStraightLine (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * y₂ = x₂ * y₁

def isTrapezoid (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (k : ℝ), x₂ = k * (x₁ - x₂) ∧ y₂ = k * (y₁ - y₂)

-- Theorem statement
theorem OPRQ_shapes (x₁ y₁ x₂ y₂ : ℝ) (h : P x₁ y₁ ≠ Q x₂ y₂) :
  (isParallelogram x₁ y₁ x₂ y₂) ∧
  (isStraightLine x₁ y₁ x₂ y₂ → OPRQ x₁ y₁ x₂ y₂ = {O, P x₁ y₁, Q x₂ y₂, R x₁ y₁ x₂ y₂}) ∧
  (∃ x₁' y₁' x₂' y₂', isTrapezoid x₁' y₁' x₂' y₂') :=
sorry

end NUMINAMATH_CALUDE_OPRQ_shapes_l4031_403152


namespace NUMINAMATH_CALUDE_total_seashells_l4031_403118

def sam_seashells : ℕ := 35
def joan_seashells : ℕ := 18

theorem total_seashells : sam_seashells + joan_seashells = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l4031_403118


namespace NUMINAMATH_CALUDE_least_marbles_nine_marbles_marbles_solution_l4031_403120

theorem least_marbles (n : ℕ) : n > 0 ∧ n % 6 = 3 ∧ n % 4 = 1 → n ≥ 9 :=
by sorry

theorem nine_marbles : 9 % 6 = 3 ∧ 9 % 4 = 1 :=
by sorry

theorem marbles_solution : ∃ (n : ℕ), n > 0 ∧ n % 6 = 3 ∧ n % 4 = 1 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 6 = 3 ∧ m % 4 = 1 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_nine_marbles_marbles_solution_l4031_403120


namespace NUMINAMATH_CALUDE_integer_equation_solution_l4031_403105

theorem integer_equation_solution : 
  {p : ℤ | ∃ (b c : ℤ), ∀ (x : ℤ), (x - p) * (x - 15) + 1 = (x + b) * (x + c)} = {13, 17} := by
sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l4031_403105


namespace NUMINAMATH_CALUDE_homework_ratio_l4031_403148

theorem homework_ratio (total : ℕ) (algebra_percent : ℚ) (linear_eq : ℕ) : 
  total = 140 →
  algebra_percent = 40/100 →
  linear_eq = 28 →
  (linear_eq : ℚ) / (algebra_percent * total) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_homework_ratio_l4031_403148


namespace NUMINAMATH_CALUDE_christmas_tree_ornaments_l4031_403115

/-- The number of ornaments Pilyulkin hung on the tree -/
def pilyulkin_ornaments : ℕ := 3

/-- The number of ornaments Guslya hung on the tree -/
def guslya_ornaments : ℕ := 2 * pilyulkin_ornaments

/-- The number of ornaments Toropyzhka hung on the tree -/
def toropyzhka_ornaments : ℕ := pilyulkin_ornaments + 15

theorem christmas_tree_ornaments :
  guslya_ornaments = 2 * pilyulkin_ornaments ∧
  toropyzhka_ornaments = pilyulkin_ornaments + 15 ∧
  toropyzhka_ornaments = 2 * (guslya_ornaments + pilyulkin_ornaments) ∧
  pilyulkin_ornaments + guslya_ornaments + toropyzhka_ornaments = 27 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_ornaments_l4031_403115


namespace NUMINAMATH_CALUDE_exclusive_math_enrollment_is_29_l4031_403146

/-- Represents the number of students in each class or combination of classes --/
structure ClassEnrollment where
  total : ℕ
  math : ℕ
  foreign : ℕ
  musicOnly : ℕ

/-- Calculates the number of students enrolled exclusively in math classes --/
def exclusiveMathEnrollment (e : ClassEnrollment) : ℕ :=
  e.math - (e.math + e.foreign - (e.total - e.musicOnly))

/-- Theorem stating that given the conditions, 29 students are enrolled exclusively in math --/
theorem exclusive_math_enrollment_is_29 (e : ClassEnrollment)
  (h1 : e.total = 120)
  (h2 : e.math = 82)
  (h3 : e.foreign = 71)
  (h4 : e.musicOnly = 20) :
  exclusiveMathEnrollment e = 29 := by
  sorry

#eval exclusiveMathEnrollment ⟨120, 82, 71, 20⟩

end NUMINAMATH_CALUDE_exclusive_math_enrollment_is_29_l4031_403146


namespace NUMINAMATH_CALUDE_minimum_speed_to_arrive_first_l4031_403104

/-- Proves the minimum speed required to arrive first given a specific scenario -/
theorem minimum_speed_to_arrive_first 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (delay : ℝ) 
  (h1 : distance = 180) 
  (h2 : speed_first = 30) 
  (h3 : delay = 2) : 
  ∃ (min_speed : ℝ), min_speed > 45 ∧ 
    ∀ (speed_second : ℝ), speed_second > min_speed → 
      distance / speed_second < distance / speed_first - delay :=
by sorry

end NUMINAMATH_CALUDE_minimum_speed_to_arrive_first_l4031_403104


namespace NUMINAMATH_CALUDE_pizza_feeding_capacity_l4031_403140

theorem pizza_feeding_capacity 
  (total_people : ℕ) 
  (pizza_cost : ℕ) 
  (earnings_per_night : ℕ) 
  (babysitting_nights : ℕ) : 
  total_people / (babysitting_nights * earnings_per_night / pizza_cost) = 3 :=
by
  -- Assuming:
  -- total_people = 15
  -- pizza_cost = 12
  -- earnings_per_night = 4
  -- babysitting_nights = 15
  sorry

#check pizza_feeding_capacity

end NUMINAMATH_CALUDE_pizza_feeding_capacity_l4031_403140


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l4031_403176

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l4031_403176


namespace NUMINAMATH_CALUDE_mika_stickers_problem_l4031_403123

/-- The number of stickers Mika gave to her sister -/
def stickers_given_to_sister (initial bought birthday used left : ℕ) : ℕ :=
  initial + bought + birthday - used - left

theorem mika_stickers_problem (initial bought birthday used left : ℕ) 
  (h1 : initial = 20)
  (h2 : bought = 26)
  (h3 : birthday = 20)
  (h4 : used = 58)
  (h5 : left = 2) :
  stickers_given_to_sister initial bought birthday used left = 6 := by
sorry

end NUMINAMATH_CALUDE_mika_stickers_problem_l4031_403123


namespace NUMINAMATH_CALUDE_train_length_l4031_403161

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 14 → speed * time * (1000 / 3600) = 280 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l4031_403161


namespace NUMINAMATH_CALUDE_complex_sum_parts_zero_l4031_403135

theorem complex_sum_parts_zero (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 1 / (i * (1 - i))
  a + b = 0 ∧ z = Complex.mk a b :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_parts_zero_l4031_403135


namespace NUMINAMATH_CALUDE_frog_arrangements_eq_25200_l4031_403137

/-- The number of ways to arrange 8 frogs (3 green, 4 red, 1 blue) in a row,
    where green frogs cannot sit next to the blue frog. -/
def frog_arrangements : ℕ :=
  let total_frogs : ℕ := 8
  let green_frogs : ℕ := 3
  let red_frogs : ℕ := 4
  let blue_frogs : ℕ := 1
  let red_arrangements : ℕ := Nat.factorial red_frogs
  let blue_positions : ℕ := red_frogs + 1
  let green_positions : ℕ := total_frogs - 1
  let green_arrangements : ℕ := Nat.choose green_positions green_frogs * Nat.factorial green_frogs
  red_arrangements * blue_positions * green_arrangements

theorem frog_arrangements_eq_25200 : frog_arrangements = 25200 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangements_eq_25200_l4031_403137


namespace NUMINAMATH_CALUDE_binomial_18_6_l4031_403190

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l4031_403190


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l4031_403119

/-- The trajectory equation of the center of a moving circle -/
theorem moving_circle_trajectory 
  (x y : ℝ) 
  (h1 : 0 < y ∧ y ≤ 1) 
  (h2 : x^2 + y^2 = (2 - y)^2) : 
  x^2 = -4*(y - 1) := by
  sorry

#check moving_circle_trajectory

end NUMINAMATH_CALUDE_moving_circle_trajectory_l4031_403119


namespace NUMINAMATH_CALUDE_adams_school_schedule_l4031_403180

/-- Represents the number of lessons Adam had on Tuesday -/
def tuesday_lessons : ℕ := 3

theorem adams_school_schedule :
  let monday_hours : ℝ := 3
  let tuesday_hours : ℝ := tuesday_lessons
  let wednesday_hours : ℝ := 2 * tuesday_hours
  let total_hours : ℝ := 12
  monday_hours + tuesday_hours + wednesday_hours = total_hours :=
by sorry


end NUMINAMATH_CALUDE_adams_school_schedule_l4031_403180


namespace NUMINAMATH_CALUDE_at_least_one_trinomial_has_two_roots_l4031_403150

theorem at_least_one_trinomial_has_two_roots 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ * a₂ * a₃ = b₁ * b₂ * b₃) 
  (h2 : b₁ * b₂ * b₃ > 1) : 
  ∃ (i : Fin 3), 
    let f := fun x => x^2 + 2 * ([a₁, a₂, a₃].get i) * x + ([b₁, b₂, b₃].get i)
    (∃ (x y : ℝ), x ≠ y ∧ f x = 0 ∧ f y = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_trinomial_has_two_roots_l4031_403150


namespace NUMINAMATH_CALUDE_farm_leg_count_l4031_403154

/-- The number of legs for animals on a farm --/
def farm_legs (total_animals : ℕ) (num_ducks : ℕ) (duck_legs : ℕ) (dog_legs : ℕ) : ℕ :=
  let num_dogs := total_animals - num_ducks
  num_ducks * duck_legs + num_dogs * dog_legs

/-- Theorem stating the total number of legs on the farm --/
theorem farm_leg_count : farm_legs 11 6 2 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_farm_leg_count_l4031_403154


namespace NUMINAMATH_CALUDE_remainder_problem_l4031_403157

theorem remainder_problem (N : ℕ) : 
  N % 68 = 0 ∧ N / 68 = 269 → N % 67 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l4031_403157


namespace NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l4031_403196

theorem percentage_of_boys_from_school_A (total_boys : ℕ) 
  (boys_A_not_science : ℕ) (science_percentage : ℚ) :
  total_boys = 400 →
  boys_A_not_science = 56 →
  science_percentage = 30 / 100 →
  (boys_A_not_science : ℚ) / ((1 - science_percentage) * total_boys) = 20 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l4031_403196


namespace NUMINAMATH_CALUDE_homework_assignments_for_28_points_l4031_403193

/-- Calculates the number of assignments required for a given point in the homework system -/
def assignmentsForPoint (n : ℕ) : ℕ := (n - 1) / 7 + 1

/-- Calculates the total number of assignments required for a given number of points -/
def totalAssignments (points : ℕ) : ℕ := 
  Finset.sum (Finset.range points) (λ i => assignmentsForPoint (i + 1))

/-- Proves that 28 homework points require 70 assignments -/
theorem homework_assignments_for_28_points : totalAssignments 28 = 70 := by
  sorry

end NUMINAMATH_CALUDE_homework_assignments_for_28_points_l4031_403193


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l4031_403131

theorem factorization_of_difference_of_squares (a b : ℝ) : 
  -a^2 + 4*b^2 = (2*b + a) * (2*b - a) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l4031_403131


namespace NUMINAMATH_CALUDE_square_of_85_l4031_403162

theorem square_of_85 : 85^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l4031_403162


namespace NUMINAMATH_CALUDE_bobby_shoe_cost_l4031_403185

theorem bobby_shoe_cost (mold_cost labor_rate hours discount_rate : ℝ) : 
  mold_cost = 250 →
  labor_rate = 75 →
  hours = 8 →
  discount_rate = 0.8 →
  mold_cost + (labor_rate * hours * discount_rate) = 730 := by
sorry

end NUMINAMATH_CALUDE_bobby_shoe_cost_l4031_403185


namespace NUMINAMATH_CALUDE_brooke_earnings_l4031_403169

/-- Represents Brooke's milk and butter business -/
structure MilkBusiness where
  milk_price : ℝ
  butter_cost : ℝ
  milk_to_butter : ℝ
  butter_price : ℝ
  num_cows : ℕ
  milk_per_cow : ℝ
  num_customers : ℕ
  min_demand : ℝ
  max_demand : ℝ

/-- Calculates the total milk produced -/
def total_milk (b : MilkBusiness) : ℝ :=
  b.num_cows * b.milk_per_cow

/-- Calculates Brooke's earnings -/
def earnings (b : MilkBusiness) : ℝ :=
  total_milk b * b.milk_price

/-- Theorem stating that Brooke's earnings are $144 -/
theorem brooke_earnings :
  ∀ b : MilkBusiness,
    b.milk_price = 3 ∧
    b.butter_cost = 0.5 ∧
    b.milk_to_butter = 2 ∧
    b.butter_price = 1.5 ∧
    b.num_cows = 12 ∧
    b.milk_per_cow = 4 ∧
    b.num_customers = 6 ∧
    b.min_demand = 4 ∧
    b.max_demand = 8 →
    earnings b = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_brooke_earnings_l4031_403169


namespace NUMINAMATH_CALUDE_phone_profit_maximization_l4031_403163

theorem phone_profit_maximization
  (profit_A_B : ℕ → ℕ → ℕ)
  (h1 : profit_A_B 1 1 = 600)
  (h2 : profit_A_B 3 2 = 1400)
  (total_phones : ℕ)
  (h3 : total_phones = 20)
  (h4 : ∀ x y : ℕ, x + y = total_phones → 3 * y ≤ 2 * x) :
  ∃ (x y : ℕ),
    x + y = total_phones ∧
    3 * y ≤ 2 * x ∧
    ∀ (a b : ℕ), a + b = total_phones → 3 * b ≤ 2 * a →
      profit_A_B x y ≥ profit_A_B a b ∧
      profit_A_B x y = 5600 ∧
      x = 12 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_phone_profit_maximization_l4031_403163


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l4031_403165

/-- The number of jelly beans initially in the barrel -/
def initial_jelly_beans : ℕ := 8000

/-- The number of people who took jelly beans -/
def total_people : ℕ := 10

/-- The number of people who took twice as many jelly beans -/
def first_group : ℕ := 6

/-- The number of people who took fewer jelly beans -/
def second_group : ℕ := 4

/-- The number of jelly beans taken by each person in the second group -/
def jelly_beans_per_second_group : ℕ := 400

/-- The number of jelly beans remaining in the barrel after everyone took their share -/
def remaining_jelly_beans : ℕ := 1600

theorem jelly_bean_problem :
  initial_jelly_beans = 
    (first_group * 2 * jelly_beans_per_second_group) + 
    (second_group * jelly_beans_per_second_group) + 
    remaining_jelly_beans :=
by
  sorry

#check jelly_bean_problem

end NUMINAMATH_CALUDE_jelly_bean_problem_l4031_403165


namespace NUMINAMATH_CALUDE_right_triangle_legs_l4031_403171

theorem right_triangle_legs (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a + b + c = 60 →   -- Perimeter condition
  h = 12 →           -- Altitude condition
  h = (a * b) / c →  -- Altitude formula
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l4031_403171


namespace NUMINAMATH_CALUDE_triangle_acute_angles_l4031_403194

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180
  positive : ∀ i, 0 < angles i

-- Define an acute angle
def is_acute (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- Define an exterior angle
def exterior_angle (t : Triangle) (i : Fin 3) : ℝ := 180 - t.angles i

-- Theorem statement
theorem triangle_acute_angles (t : Triangle) : 
  (∃ i j : Fin 3, i ≠ j ∧ is_acute (t.angles i) ∧ is_acute (t.angles j)) ∧ 
  (∀ i j k : Fin 3, i ≠ j → j ≠ k → i ≠ k → 
    ¬(is_acute (exterior_angle t i) ∧ is_acute (exterior_angle t j))) :=
sorry

end NUMINAMATH_CALUDE_triangle_acute_angles_l4031_403194


namespace NUMINAMATH_CALUDE_halfway_fraction_l4031_403112

theorem halfway_fraction (a b c : ℚ) (ha : a = 1/4) (hb : b = 1/6) (hc : c = 1/3) :
  (a + b + c) / 3 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l4031_403112


namespace NUMINAMATH_CALUDE_population_change_approx_19_58_percent_l4031_403136

/-- Represents the population change over three years given specific growth and decrease rates -/
def population_change (natural_growth : ℝ) (migration_year1 : ℝ) (migration_year2 : ℝ) (migration_year3 : ℝ) (disaster_decrease : ℝ) : ℝ :=
  let year1 := (1 + natural_growth) * (1 + migration_year1)
  let year2 := (1 + natural_growth) * (1 + migration_year2)
  let year3 := (1 + natural_growth) * (1 + migration_year3)
  let three_year_change := year1 * year2 * year3
  three_year_change * (1 - disaster_decrease)

/-- Theorem stating that the population change over three years is approximately 19.58% -/
theorem population_change_approx_19_58_percent :
  let natural_growth := 0.09
  let migration_year1 := -0.01
  let migration_year2 := -0.015
  let migration_year3 := -0.02
  let disaster_decrease := 0.03
  abs (population_change natural_growth migration_year1 migration_year2 migration_year3 disaster_decrease - 1.1958) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_population_change_approx_19_58_percent_l4031_403136


namespace NUMINAMATH_CALUDE_bird_watching_percentage_difference_l4031_403100

-- Define the number of birds seen by each person
def gabrielle_robins : ℕ := 5
def gabrielle_cardinals : ℕ := 4
def gabrielle_blue_jays : ℕ := 3

def chase_robins : ℕ := 2
def chase_cardinals : ℕ := 5
def chase_blue_jays : ℕ := 3

-- Calculate total birds seen by each person
def gabrielle_total : ℕ := gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
def chase_total : ℕ := chase_robins + chase_cardinals + chase_blue_jays

-- Define the percentage difference
def percentage_difference : ℚ := (gabrielle_total - chase_total : ℚ) / chase_total * 100

-- Theorem statement
theorem bird_watching_percentage_difference :
  percentage_difference = 20 :=
sorry

end NUMINAMATH_CALUDE_bird_watching_percentage_difference_l4031_403100


namespace NUMINAMATH_CALUDE_inequality_proof_l4031_403175

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a * Real.sqrt (c^2 + 1))) + (1 / (b * Real.sqrt (a^2 + 1))) + (1 / (c * Real.sqrt (b^2 + 1))) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4031_403175


namespace NUMINAMATH_CALUDE_squares_in_figure_100_l4031_403134

/-- The number of nonoverlapping unit squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^2 + 2 * n + 1

/-- The sequence of nonoverlapping unit squares follows the pattern -/
axiom sequence_pattern :
  f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25

/-- The number of nonoverlapping unit squares in figure 100 is 20201 -/
theorem squares_in_figure_100 : f 100 = 20201 := by sorry

end NUMINAMATH_CALUDE_squares_in_figure_100_l4031_403134


namespace NUMINAMATH_CALUDE_quadratic_condition_l4031_403141

def is_quadratic (m : ℝ) : Prop :=
  (|m| = 2) ∧ (m - 2 ≠ 0)

theorem quadratic_condition (m : ℝ) :
  is_quadratic m ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l4031_403141


namespace NUMINAMATH_CALUDE_radius_of_circle_B_l4031_403107

/-- A configuration of four circles A, B, C, and D with specific properties -/
structure CircleConfiguration where
  /-- Radius of circle A -/
  radiusA : ℝ
  /-- Radius of circle B -/
  radiusB : ℝ
  /-- Radius of circle D -/
  radiusD : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externallyTangent : Bool
  /-- Circles A, B, and C are internally tangent to circle D -/
  internallyTangentD : Bool
  /-- Circles B and C are congruent -/
  bCongruentC : Bool
  /-- Circle A passes through the center of D -/
  aPassesThroughCenterD : Bool

/-- The theorem stating that given the specific configuration, the radius of circle B is 7/3 -/
theorem radius_of_circle_B (config : CircleConfiguration) 
  (h1 : config.radiusA = 2)
  (h2 : config.externallyTangent = true)
  (h3 : config.internallyTangentD = true)
  (h4 : config.bCongruentC = true)
  (h5 : config.aPassesThroughCenterD = true) :
  config.radiusB = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_B_l4031_403107


namespace NUMINAMATH_CALUDE_george_turning_25_l4031_403130

/-- Represents George's age and bill exchange scenario --/
def GeorgeBirthdayProblem (n : ℕ) : Prop :=
  let billsReceived : ℕ := n
  let billsRemaining : ℚ := 0.8 * n
  let exchangeRate : ℚ := 1.5
  let totalExchange : ℚ := 12
  (exchangeRate * billsRemaining = totalExchange) ∧ (n + 15 = 25)

/-- Theorem stating that George is turning 25 years old --/
theorem george_turning_25 : ∃ n : ℕ, GeorgeBirthdayProblem n := by
  sorry

end NUMINAMATH_CALUDE_george_turning_25_l4031_403130


namespace NUMINAMATH_CALUDE_amit_left_after_three_days_l4031_403143

/-- The number of days Amit can complete the work alone -/
def amit_days : ℝ := 15

/-- The number of days Ananthu can complete the work alone -/
def ananthu_days : ℝ := 30

/-- The total number of days taken to complete the work -/
def total_days : ℝ := 27

/-- The number of days Amit worked before leaving -/
def amit_worked_days : ℝ := 3

theorem amit_left_after_three_days :
  amit_worked_days * (1 / amit_days) + (total_days - amit_worked_days) * (1 / ananthu_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_amit_left_after_three_days_l4031_403143


namespace NUMINAMATH_CALUDE_perfect_square_count_l4031_403192

theorem perfect_square_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ n ≤ 2000 ∧ ∃ k : ℕ, 10 * n = k^2) ∧ 
  (∀ n : ℕ, n > 0 ∧ n ≤ 2000 ∧ (∃ k : ℕ, 10 * n = k^2) → n ∈ S) ∧
  Finset.card S = 14 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_count_l4031_403192


namespace NUMINAMATH_CALUDE_percentage_and_subtraction_l4031_403149

theorem percentage_and_subtraction (y : ℝ) : 
  (20 : ℝ) / y = (80 : ℝ) / 100 → y = 25 ∧ y - 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_and_subtraction_l4031_403149


namespace NUMINAMATH_CALUDE_diamond_four_three_l4031_403172

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_four_three_l4031_403172


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l4031_403110

theorem power_of_power_at_three : (3^(3^2))^(3^3) = 3^243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l4031_403110


namespace NUMINAMATH_CALUDE_sum_of_cube_difference_l4031_403145

theorem sum_of_cube_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 150 →
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_difference_l4031_403145


namespace NUMINAMATH_CALUDE_min_students_theorem_l4031_403183

/-- Given a class of students, returns the minimum number of students
    who have brown eyes, a lunch box, and do not wear glasses. -/
def min_students_with_characteristics (total : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) (glasses : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of students with the given characteristics -/
theorem min_students_theorem :
  min_students_with_characteristics 40 18 25 16 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_students_theorem_l4031_403183


namespace NUMINAMATH_CALUDE_minimum_dinner_cost_l4031_403108

/-- Represents an ingredient with its cost, quantity, and number of servings -/
structure Ingredient where
  name : String
  cost : ℚ
  quantity : ℚ
  servings : ℕ

/-- Calculates the minimum number of units needed to serve a given number of people -/
def minUnitsNeeded (servingsPerUnit : ℕ) (people : ℕ) : ℕ :=
  (people + servingsPerUnit - 1) / servingsPerUnit

/-- Calculates the total cost for an ingredient given the number of people to serve -/
def ingredientCost (i : Ingredient) (people : ℕ) : ℚ :=
  i.cost * (minUnitsNeeded i.servings people : ℚ)

/-- The list of ingredients for the dinner -/
def ingredients : List Ingredient := [
  ⟨"Pasta", 112/100, 500, 5⟩,
  ⟨"Meatballs", 524/100, 500, 4⟩,
  ⟨"Tomato sauce", 231/100, 400, 5⟩,
  ⟨"Tomatoes", 147/100, 400, 4⟩,
  ⟨"Lettuce", 97/100, 1, 6⟩,
  ⟨"Olives", 210/100, 1, 8⟩,
  ⟨"Cheese", 270/100, 1, 7⟩
]

/-- The number of people to serve -/
def numPeople : ℕ := 8

/-- The theorem stating the minimum total cost and cost per serving -/
theorem minimum_dinner_cost :
  let totalCost := (ingredients.map (ingredientCost · numPeople)).sum
  totalCost = 2972/100 ∧ totalCost / (numPeople : ℚ) = 3715/1000 := by
  sorry


end NUMINAMATH_CALUDE_minimum_dinner_cost_l4031_403108


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l4031_403132

def LinearFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

theorem linear_function_decreasing (a b : ℝ) :
  (∀ x y, x < y → LinearFunction a b x > LinearFunction a b y) ↔ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l4031_403132


namespace NUMINAMATH_CALUDE_johns_allowance_l4031_403117

theorem johns_allowance (A : ℚ) : 
  (7/15 + 3/10 + 1/6 : ℚ) * A +  -- Spent on arcade, books, and clothes
  2/5 * (1 - (7/15 + 3/10 + 1/6 : ℚ)) * A +  -- Spent at toy store
  (6/5 : ℚ) = A  -- Last $1.20 spent at candy store (represented as 6/5)
  → A = 30 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l4031_403117


namespace NUMINAMATH_CALUDE_dinosaur_egg_theft_l4031_403186

theorem dinosaur_egg_theft (total_eggs : ℕ) (claimed_max : ℕ) : 
  total_eggs = 20 → 
  claimed_max = 7 → 
  ¬(∃ (a b : ℕ), 
    a + b + claimed_max = total_eggs ∧ 
    a ≠ b ∧ 
    a ≠ claimed_max ∧ 
    b ≠ claimed_max ∧
    a < claimed_max ∧ 
    b < claimed_max) := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_egg_theft_l4031_403186


namespace NUMINAMATH_CALUDE_chord_intersection_theorem_l4031_403182

theorem chord_intersection_theorem (r : ℝ) (PT OT : ℝ) : 
  r = 7 → OT = 3 → PT = 8 → 
  ∃ (RS : ℝ), RS = 16 ∧ 
  ∃ (x : ℝ), x * (RS - x) = PT * PT ∧ 
  ∃ (n : ℕ), x * (RS - x) = n^2 :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_theorem_l4031_403182


namespace NUMINAMATH_CALUDE_area_ratio_lateral_angle_relation_area_ratio_bounds_l4031_403125

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  -- Add necessary fields here
  mk ::

/-- The ratio of cross-section area to lateral surface area -/
def area_ratio (p : RegularQuadPyramid) : ℝ := sorry

/-- The angle between two adjacent lateral faces -/
def lateral_face_angle (p : RegularQuadPyramid) : ℝ := sorry

/-- Theorem about the relationship between area ratio and lateral face angle -/
theorem area_ratio_lateral_angle_relation (p : RegularQuadPyramid) :
  lateral_face_angle p = Real.arccos (8 * (area_ratio p)^2 - 1) :=
sorry

/-- Theorem about the permissible values of the area ratio -/
theorem area_ratio_bounds (p : RegularQuadPyramid) :
  0 < area_ratio p ∧ area_ratio p < Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_lateral_angle_relation_area_ratio_bounds_l4031_403125


namespace NUMINAMATH_CALUDE_abc_inequality_l4031_403156

/-- Given a + 2b + 3c = 4, prove two statements about a, b, and c -/
theorem abc_inequality (a b c : ℝ) (h : a + 2*b + 3*c = 4) :
  (∀ (ha : a > 0) (hb : b > 0) (hc : c > 0), 1/a + 2/b + 3/c ≥ 9) ∧
  (∃ (m : ℝ), m = 4/3 ∧ ∀ (x y z : ℝ), x + 2*y + 3*z = 4 → |1/2*x + y| + |z| ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l4031_403156


namespace NUMINAMATH_CALUDE_system_solution_l4031_403102

theorem system_solution :
  ∀ (x y z : ℝ),
    (x + 1) * y * z = 12 ∧
    (y + 1) * z * x = 4 ∧
    (z + 1) * x * y = 4 →
    ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4031_403102


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l4031_403127

/-- The probability of picking two red balls from a bag containing red, blue, and green balls -/
theorem prob_two_red_balls (red blue green : ℕ) (h : red = 5 ∧ blue = 6 ∧ green = 4) :
  let total := red + blue + green
  (red : ℚ) / total * ((red - 1) : ℚ) / (total - 1) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l4031_403127


namespace NUMINAMATH_CALUDE_equal_area_rectangles_length_l4031_403160

/-- Given two rectangles of equal area, where one rectangle has dimensions 2 inches by 60 inches,
    and the other has a width of 24 inches, prove that the length of the second rectangle is 5 inches. -/
theorem equal_area_rectangles_length (l : ℝ) :
  (2 : ℝ) * 60 = l * 24 → l = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_length_l4031_403160


namespace NUMINAMATH_CALUDE_division_problem_l4031_403173

theorem division_problem (number : ℕ) : 
  number / 179 = 89 ∧ number % 179 = 37 → number = 15968 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l4031_403173


namespace NUMINAMATH_CALUDE_prob_one_of_A_or_B_is_two_thirds_l4031_403198

/-- The number of study groups -/
def num_groups : ℕ := 4

/-- The number of groups to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one of group A and group B -/
def prob_one_of_A_or_B : ℚ := 2/3

/-- Theorem stating that the probability of selecting exactly one of group A and group B
    when randomly selecting two groups out of four groups is 2/3 -/
theorem prob_one_of_A_or_B_is_two_thirds :
  prob_one_of_A_or_B = (num_groups - 2) / (Nat.choose num_groups num_selected) := by
  sorry

end NUMINAMATH_CALUDE_prob_one_of_A_or_B_is_two_thirds_l4031_403198


namespace NUMINAMATH_CALUDE_parents_contribution_half_l4031_403174

/-- Represents the financial details for Nancy's university tuition --/
structure TuitionFinances where
  tuition : ℕ
  scholarship : ℕ
  workHours : ℕ
  hourlyWage : ℕ

/-- Calculates the ratio of parents' contribution to total tuition --/
def parentsContributionRatio (finances : TuitionFinances) : Rat :=
  let studentLoan := 2 * finances.scholarship
  let totalAid := finances.scholarship + studentLoan
  let workEarnings := finances.workHours * finances.hourlyWage
  let nancyContribution := totalAid + workEarnings
  let parentsContribution := finances.tuition - nancyContribution
  parentsContribution / finances.tuition

/-- Theorem stating that the parents' contribution ratio is 1/2 --/
theorem parents_contribution_half (finances : TuitionFinances) 
  (h1 : finances.tuition = 22000)
  (h2 : finances.scholarship = 3000)
  (h3 : finances.workHours = 200)
  (h4 : finances.hourlyWage = 10) :
  parentsContributionRatio finances = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parents_contribution_half_l4031_403174


namespace NUMINAMATH_CALUDE_modular_sum_of_inverses_l4031_403189

theorem modular_sum_of_inverses (p : ℕ) (h_prime : Nat.Prime p) (h_p : p = 31) :
  ∃ (a b : ℕ), a < p ∧ b < p ∧
  (5 * a) % p = 1 ∧
  (25 * b) % p = 1 ∧
  (a + b) % p = 26 := by
sorry

end NUMINAMATH_CALUDE_modular_sum_of_inverses_l4031_403189


namespace NUMINAMATH_CALUDE_negation_absolute_value_inequality_l4031_403151

theorem negation_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_absolute_value_inequality_l4031_403151


namespace NUMINAMATH_CALUDE_tangent_two_implies_expression_equals_negative_two_l4031_403187

-- Define the theorem
theorem tangent_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_implies_expression_equals_negative_two_l4031_403187


namespace NUMINAMATH_CALUDE_machine_productivity_problem_l4031_403103

theorem machine_productivity_problem 
  (productivity_second : ℝ) 
  (productivity_first : ℝ := 1.4 * productivity_second) 
  (hours_first : ℝ := 6) 
  (hours_second : ℝ := 8) 
  (total_parts : ℕ := 820) :
  productivity_first * hours_first + productivity_second * hours_second = total_parts → 
  (productivity_first * hours_first = 420 ∧ productivity_second * hours_second = 400) :=
by
  sorry

end NUMINAMATH_CALUDE_machine_productivity_problem_l4031_403103


namespace NUMINAMATH_CALUDE_dagger_example_l4031_403147

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * ((n + q) / n)

-- Theorem statement
theorem dagger_example : dagger (9/5) (7/2) = 441/5 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l4031_403147


namespace NUMINAMATH_CALUDE_fifth_toss_probability_l4031_403178

def coin_flip_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ (n - 1) * (1 / 2)

theorem fifth_toss_probability :
  coin_flip_probability 5 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_toss_probability_l4031_403178


namespace NUMINAMATH_CALUDE_intersection_M_N_l4031_403177

def M : Set ℝ := {x | ∃ t : ℝ, x = 2^(-t)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4031_403177


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l4031_403138

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Given side lengths
    (b = c) →                  -- Isosceles condition
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (a + b + c = 22)           -- Perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l4031_403138


namespace NUMINAMATH_CALUDE_complex_number_problem_l4031_403133

/-- Given a complex number z = 3 + bi where b is a positive real number,
    and (z - 2)² is a pure imaginary number, prove that:
    1. z = 3 + i
    2. |z / (2 + i)| = √2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) 
    (h1 : b > 0)
    (h2 : z = 3 + b * I)
    (h3 : ∃ (y : ℝ), (z - 2)^2 = y * I) :
  z = 3 + I ∧ Complex.abs (z / (2 + I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l4031_403133


namespace NUMINAMATH_CALUDE_johns_shower_duration_johns_shower_theorem_l4031_403155

theorem johns_shower_duration (shower_duration : ℕ) (shower_frequency : ℕ) 
  (water_usage_rate : ℕ) (total_water_usage : ℕ) : ℕ :=
  let water_per_shower := shower_duration * water_usage_rate
  let num_showers := total_water_usage / water_per_shower
  let num_days := num_showers * shower_frequency
  let num_weeks := num_days / 7
  num_weeks

theorem johns_shower_theorem : 
  johns_shower_duration 10 2 2 280 = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_shower_duration_johns_shower_theorem_l4031_403155


namespace NUMINAMATH_CALUDE_train_crossing_time_l4031_403191

/-- Proves that a train with given length and speed takes a specific time to cross a fixed point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 31.5 →
  crossing_time = 16 →
  train_length / (train_speed_kmh * 1000 / 3600) = crossing_time :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l4031_403191


namespace NUMINAMATH_CALUDE_unique_solution_non_unique_solution_l4031_403197

-- Define the equation
def equation (x a b : ℝ) : Prop :=
  (x - a) / (x - 2) + (x - b) / (x - 3) = 2

-- Theorem for unique solution
theorem unique_solution (a b : ℝ) :
  (∃! x, equation x a b) ↔ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
sorry

-- Theorem for non-unique solution
theorem non_unique_solution (a b : ℝ) :
  (∃ x y, x ≠ y ∧ equation x a b ∧ equation y a b) ↔ (a = 2 ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_non_unique_solution_l4031_403197
