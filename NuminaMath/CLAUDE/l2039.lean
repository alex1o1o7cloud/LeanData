import Mathlib

namespace min_value_x_plus_reciprocal_l2039_203945

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end min_value_x_plus_reciprocal_l2039_203945


namespace problem_1_problem_2_l2039_203922

theorem problem_1 : (1) - 2^2 + (-1/2)^4 + (3 - Real.pi)^0 = -47/16 := by sorry

theorem problem_2 : 5^2022 * (-1/5)^2023 = -1/5 := by sorry

end problem_1_problem_2_l2039_203922


namespace revenue_in_scientific_notation_l2039_203991

/-- Represents 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The tourism revenue in billions -/
def revenue : ℝ := 2.93

theorem revenue_in_scientific_notation : 
  revenue * billion = 2.93 * (10 : ℝ)^9 := by sorry

end revenue_in_scientific_notation_l2039_203991


namespace max_true_statements_l2039_203940

theorem max_true_statements (a b : ℝ) : ∃ a b : ℝ,
  (a^2 + b^2 < (a + b)^2) ∧
  (a * b > 0) ∧
  (a > b) ∧
  (a > 0) ∧
  (b > 0) := by
sorry

end max_true_statements_l2039_203940


namespace valid_triangulations_are_4_7_19_l2039_203992

/-- Represents a triangulation of a triangle. -/
structure Triangulation where
  num_triangles : ℕ
  sides_per_vertex : ℕ

/-- Predicate to check if a triangulation is valid according to the problem conditions. -/
def is_valid_triangulation (t : Triangulation) : Prop :=
  t.num_triangles ≤ 19 ∧
  t.num_triangles > 0 ∧
  t.sides_per_vertex > 2

/-- The set of all valid triangulations. -/
def valid_triangulations : Set Triangulation :=
  {t : Triangulation | is_valid_triangulation t}

/-- Theorem stating that the only valid triangulations have 4, 7, or 19 triangles. -/
theorem valid_triangulations_are_4_7_19 :
  ∀ t ∈ valid_triangulations, t.num_triangles = 4 ∨ t.num_triangles = 7 ∨ t.num_triangles = 19 := by
  sorry

end valid_triangulations_are_4_7_19_l2039_203992


namespace money_division_l2039_203941

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 4400 →
  r - q = 5500 := by
sorry

end money_division_l2039_203941


namespace heart_equal_set_is_four_lines_l2039_203988

-- Define the ♥ operation
def heart (a b : ℝ) : ℝ := a^3 * b - a^2 * b^2 + a * b^3

-- Define the set of points satisfying x ♥ y = y ♥ x
def heart_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | heart p.1 p.2 = heart p.2 p.1}

-- Theorem statement
theorem heart_equal_set_is_four_lines :
  heart_equal_set = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2} :=
by sorry

end heart_equal_set_is_four_lines_l2039_203988


namespace alice_overall_score_approx_80_percent_l2039_203939

/-- Represents a test with a number of questions and a score percentage -/
structure Test where
  questions : ℕ
  score : ℚ
  scoreInRange : 0 ≤ score ∧ score ≤ 1

/-- Alice's test results -/
def aliceTests : List Test := [
  ⟨20, 3/4, by norm_num⟩,
  ⟨50, 17/20, by norm_num⟩,
  ⟨30, 3/5, by norm_num⟩,
  ⟨40, 9/10, by norm_num⟩
]

/-- The total number of questions Alice answered correctly -/
def totalCorrect : ℚ :=
  aliceTests.foldl (fun acc test => acc + test.questions * test.score) 0

/-- The total number of questions across all tests -/
def totalQuestions : ℕ :=
  aliceTests.foldl (fun acc test => acc + test.questions) 0

/-- Alice's overall score as a percentage -/
def overallScore : ℚ := totalCorrect / totalQuestions

theorem alice_overall_score_approx_80_percent :
  abs (overallScore - 4/5) < 1/100 := by
  sorry

end alice_overall_score_approx_80_percent_l2039_203939


namespace probability_all_red_or_all_white_l2039_203906

/-- The probability of drawing either all red marbles or all white marbles when drawing 3 marbles
    without replacement from a bag containing 5 red, 4 white, and 6 blue marbles -/
theorem probability_all_red_or_all_white (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) 
    (blue_marbles : ℕ) (drawn_marbles : ℕ) :
  total_marbles = red_marbles + white_marbles + blue_marbles →
  total_marbles = 15 →
  red_marbles = 5 →
  white_marbles = 4 →
  blue_marbles = 6 →
  drawn_marbles = 3 →
  (red_marbles.choose drawn_marbles * (total_marbles - drawn_marbles).factorial / total_marbles.factorial +
   white_marbles.choose drawn_marbles * (total_marbles - drawn_marbles).factorial / total_marbles.factorial : ℚ) = 14 / 455 := by
  sorry

#check probability_all_red_or_all_white

end probability_all_red_or_all_white_l2039_203906


namespace polynomial_divisible_by_nine_l2039_203971

theorem polynomial_divisible_by_nine (n : ℤ) : ∃ k : ℤ, n^6 - 3*n^5 + 4*n^4 - 3*n^3 + 4*n^2 - 3*n = 9*k := by
  sorry

end polynomial_divisible_by_nine_l2039_203971


namespace children_catered_count_l2039_203934

/-- Represents the number of children that can be catered with remaining food --/
def children_catered (total_adults : ℕ) (total_children : ℕ) (adults_meal_capacity : ℕ) (children_meal_capacity : ℕ) (adults_eaten : ℕ) (adult_child_consumption_ratio : ℚ) (adult_diet_restriction_percent : ℚ) (child_diet_restriction_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of children that can be catered under given conditions --/
theorem children_catered_count : 
  children_catered 55 70 70 90 21 (3/2) (1/5) (3/20) = 63 :=
sorry

end children_catered_count_l2039_203934


namespace parallel_condition_l2039_203994

-- Define the structure for a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

-- Define the two lines from the problem
def line1 (a : ℝ) : Line := ⟨2, a, -1⟩
def line2 (b : ℝ) : Line := ⟨b, 2, 1⟩

theorem parallel_condition (a b : ℝ) :
  (parallel (line1 a) (line2 b) → a * b = 4) ∧
  ∃ a b, a * b = 4 ∧ ¬parallel (line1 a) (line2 b) := by sorry

end parallel_condition_l2039_203994


namespace foundation_cost_theorem_l2039_203927

/-- Represents the dimensions of a concrete slab -/
structure SlabDimensions where
  length : Float
  width : Float
  height : Float

/-- Calculates the volume of a concrete slab -/
def slabVolume (d : SlabDimensions) : Float :=
  d.length * d.width * d.height

/-- Calculates the weight of concrete given its volume and density -/
def concreteWeight (volume density : Float) : Float :=
  volume * density

/-- Calculates the cost of concrete given its weight and price per pound -/
def concreteCost (weight pricePerPound : Float) : Float :=
  weight * pricePerPound

theorem foundation_cost_theorem 
  (slabDim : SlabDimensions)
  (concreteDensity : Float)
  (concretePricePerPound : Float)
  (numHomes : Nat) :
  slabDim.length = 100 →
  slabDim.width = 100 →
  slabDim.height = 0.5 →
  concreteDensity = 150 →
  concretePricePerPound = 0.02 →
  numHomes = 3 →
  concreteCost 
    (concreteWeight 
      (slabVolume slabDim * numHomes.toFloat) 
      concreteDensity) 
    concretePricePerPound = 45000 := by
  sorry

end foundation_cost_theorem_l2039_203927


namespace hyperbola_equation_l2039_203955

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1,
if one of its asymptotes is y = (√7/3)x and 
the distance from one of its vertices to the nearer focus is 1,
then a = 3 and b = √7.
-/
theorem hyperbola_equation (a b : ℝ) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (b/a = Real.sqrt 7 / 3) →               -- Asymptote condition
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c - a = 1) →  -- Vertex-focus distance condition
  (a = 3 ∧ b = Real.sqrt 7) :=
by sorry

end hyperbola_equation_l2039_203955


namespace tangent_slope_at_one_l2039_203925

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State that f is differentiable
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
  |((f 1 - f (1 + 2 * Δx)) / Δx) - 2| < ε)

-- Theorem statement
theorem tangent_slope_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f 1 - f (1 + 2 * Δx)) / Δx) - 2| < ε) : 
  deriv f 1 = -1 := by sorry

end tangent_slope_at_one_l2039_203925


namespace probability_consecutive_numbers_l2039_203935

/-- The total number of lottery numbers --/
def total_numbers : ℕ := 90

/-- The number of drawn lottery numbers --/
def drawn_numbers : ℕ := 5

/-- The set of all possible combinations of drawn numbers --/
def all_combinations : ℕ := Nat.choose total_numbers drawn_numbers

/-- The set of combinations with at least one pair of consecutive numbers --/
def consecutive_combinations : ℕ := 9122966

/-- The probability of drawing at least one pair of consecutive numbers --/
theorem probability_consecutive_numbers :
  (consecutive_combinations : ℚ) / all_combinations = 9122966 / 43949268 := by
  sorry

end probability_consecutive_numbers_l2039_203935


namespace fraction_to_decimal_l2039_203965

theorem fraction_to_decimal : (17 : ℚ) / 50 = 0.34 := by
  sorry

end fraction_to_decimal_l2039_203965


namespace quadratic_one_root_l2039_203947

theorem quadratic_one_root (n : ℝ) : 
  (∃! x : ℝ, x^2 - 6*n*x - 9*n = 0) ∧ n ≥ 0 → n = 0 := by
sorry

end quadratic_one_root_l2039_203947


namespace min_turns_to_win_l2039_203959

/-- Represents the state of the game -/
structure GameState :=
  (a₁ a₂ a₃ a₄ a₅ : ℕ)

/-- Defines a valid move in the game -/
def validMove (i : ℕ) : Prop :=
  2 ≤ i ∧ i ≤ 5

/-- Applies a move to the game state -/
def applyMove (state : GameState) (i : ℕ) : GameState :=
  match i with
  | 2 => GameState.mk state.a₁ (state.a₁ + state.a₂) state.a₃ state.a₄ state.a₅
  | 3 => GameState.mk state.a₁ state.a₂ (state.a₂ + state.a₃) state.a₄ state.a₅
  | 4 => GameState.mk state.a₁ state.a₂ state.a₃ (state.a₃ + state.a₄) state.a₅
  | 5 => GameState.mk state.a₁ state.a₂ state.a₃ state.a₄ (state.a₄ + state.a₅)
  | _ => state

/-- The initial state of the game -/
def initialState : GameState :=
  GameState.mk 1 0 0 0 0

/-- Predicate to check if the game is won -/
def isWinningState (state : GameState) : Prop :=
  state.a₅ > 1000000

/-- Theorem: The minimum number of turns to win the game is 127 -/
theorem min_turns_to_win :
  ∃ (moves : List ℕ),
    (∀ m ∈ moves, validMove m) ∧
    isWinningState (moves.foldl applyMove initialState) ∧
    moves.length = 127 ∧
    (∀ (other_moves : List ℕ),
      (∀ m ∈ other_moves, validMove m) →
      isWinningState (other_moves.foldl applyMove initialState) →
      other_moves.length ≥ 127) :=
by sorry


end min_turns_to_win_l2039_203959


namespace girls_in_college_l2039_203912

theorem girls_in_college (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : boys + girls = 520) : girls = 200 := by
  sorry

end girls_in_college_l2039_203912


namespace wrench_force_calculation_l2039_203948

/-- Represents the force required to loosen a bolt with a wrench of a given length -/
structure WrenchForce where
  length : ℝ
  force : ℝ

/-- The inverse relationship between force and wrench length -/
def inverseProportion (w1 w2 : WrenchForce) : Prop :=
  w1.force * w1.length = w2.force * w2.length

theorem wrench_force_calculation 
  (w1 w2 : WrenchForce)
  (h1 : w1.length = 12)
  (h2 : w1.force = 300)
  (h3 : w2.length = 18)
  (h4 : inverseProportion w1 w2) :
  w2.force = 200 := by
  sorry

end wrench_force_calculation_l2039_203948


namespace expand_product_l2039_203979

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := by
  sorry

end expand_product_l2039_203979


namespace chocolates_for_charlie_l2039_203951

/-- Represents the number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def chocolates_for_self : ℕ := 2

/-- Represents the number of chocolates Kantana buys for her sister each Saturday -/
def chocolates_for_sister : ℕ := 1

/-- Represents the total number of chocolates Kantana bought for the month -/
def total_chocolates : ℕ := 22

/-- Theorem stating that Kantana bought 10 chocolates for Charlie's birthday gift -/
theorem chocolates_for_charlie : 
  total_chocolates - (saturdays_in_month * (chocolates_for_self + chocolates_for_sister)) = 10 := by
  sorry

end chocolates_for_charlie_l2039_203951


namespace infinitely_many_triangular_squares_l2039_203967

/-- Definition of triangular numbers -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for a number being square -/
def is_square (k : ℕ) : Prop := ∃ m : ℕ, k = m * m

/-- The recurrence relation for generating triangular square numbers -/
axiom recurrence_relation (n : ℕ) : T (4 * n * (n + 1)) = 4 * T n * (2 * n + 1)^2

/-- Theorem: There are infinitely many numbers that are both triangular and square -/
theorem infinitely_many_triangular_squares :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_square (T n) :=
sorry

end infinitely_many_triangular_squares_l2039_203967


namespace child_b_share_l2039_203963

theorem child_b_share (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 1800 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b * total_amount) / (ratio_a + ratio_b + ratio_c) = 600 := by
  sorry

end child_b_share_l2039_203963


namespace binomial_coefficient_1502_1_l2039_203911

theorem binomial_coefficient_1502_1 : Nat.choose 1502 1 = 1502 := by
  sorry

end binomial_coefficient_1502_1_l2039_203911


namespace telescope_payment_difference_l2039_203964

theorem telescope_payment_difference (joan_payment karl_payment : ℕ) : 
  joan_payment = 158 →
  joan_payment + karl_payment = 400 →
  2 * joan_payment - karl_payment = 74 := by
sorry

end telescope_payment_difference_l2039_203964


namespace square_root_division_l2039_203983

theorem square_root_division : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end square_root_division_l2039_203983


namespace spade_equation_solution_l2039_203907

/-- The spade operation -/
def spade_op (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

/-- Theorem stating that if X spade 5 = 23, then X = 7.75 -/
theorem spade_equation_solution :
  ∀ X : ℝ, spade_op X 5 = 23 → X = 7.75 := by
  sorry

end spade_equation_solution_l2039_203907


namespace sports_field_dimensions_l2039_203957

/-- The dimensions of a rectangular sports field with a surrounding path -/
theorem sports_field_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x : ℝ,
    x > 0 ∧
    x * (x + b) = (x + 2*a) * (x + b + 2*a) - x * (x + b) ∧
    x = (Real.sqrt (b^2 + 32*a^2) - b + 4*a) / 2 ∧
    x + b = (Real.sqrt (b^2 + 32*a^2) + b + 4*a) / 2 :=
by sorry

end sports_field_dimensions_l2039_203957


namespace ellipse_property_l2039_203968

/-- Definition of an ellipse with foci F₁ and F₂ -/
def Ellipse (F₁ F₂ : ℝ × ℝ) (a b : ℝ) :=
  {P : ℝ × ℝ | (P.1^2 / a^2) + (P.2^2 / b^2) = 1 ∧ a > b ∧ b > 0}

/-- The angle between two vectors -/
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_property (F₁ F₂ : ℝ × ℝ) (a b : ℝ) (P : ℝ × ℝ) :
  P ∈ Ellipse F₁ F₂ a b →
  angle (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) = π / 3 →
  triangle_area P F₁ F₂ = 3 * Real.sqrt 3 →
  b = 3 := by sorry

end ellipse_property_l2039_203968


namespace m_range_l2039_203932

theorem m_range (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ)^2 * m - 5*m + 4*(Real.sin θ)^2 ≥ 0) → 
  (m ≥ 4 ∨ m ≤ 0) := by
sorry

end m_range_l2039_203932


namespace sum_of_roots_equals_864_l2039_203903

theorem sum_of_roots_equals_864 
  (p q r s : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h_eq1 : ∀ x, x^2 - 8*p*x - 12*q = 0 ↔ x = r ∨ x = s)
  (h_eq2 : ∀ x, x^2 - 8*r*x - 12*s = 0 ↔ x = p ∨ x = q) :
  p + q + r + s = 864 := by
sorry

end sum_of_roots_equals_864_l2039_203903


namespace isosceles_triangle_base_l2039_203984

/-- An isosceles triangle with perimeter 16 and one side 4 has a base of 4 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a + b + c = 16 →  -- perimeter is 16
  a = b →           -- isosceles triangle condition
  a = 4 →           -- one side is 4
  c = 4 :=          -- prove that the base is 4
by sorry

end isosceles_triangle_base_l2039_203984


namespace probability_defect_free_l2039_203928

/-- Represents the proportion of components from Company A in the warehouse -/
def proportion_A : ℝ := 0.60

/-- Represents the proportion of components from Company B in the warehouse -/
def proportion_B : ℝ := 0.40

/-- Represents the defect rate of components from Company A -/
def defect_rate_A : ℝ := 0.98

/-- Represents the defect rate of components from Company B -/
def defect_rate_B : ℝ := 0.95

/-- Theorem stating that the probability of a randomly selected component being defect-free is 0.032 -/
theorem probability_defect_free : 
  proportion_A * (1 - defect_rate_A) + proportion_B * (1 - defect_rate_B) = 0.032 := by
  sorry


end probability_defect_free_l2039_203928


namespace cos_160_sin_10_minus_sin_20_cos_10_l2039_203924

theorem cos_160_sin_10_minus_sin_20_cos_10 :
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) -
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) = -1/2 := by
  sorry

end cos_160_sin_10_minus_sin_20_cos_10_l2039_203924


namespace unique_fraction_sum_l2039_203944

theorem unique_fraction_sum : ∃! (a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
  (5 : ℚ) / 7 = a₂ / 2 + a₃ / 6 + a₄ / 24 + a₅ / 120 + a₆ / 720 + a₇ / 5040 ∧
  a₂ < 2 ∧ a₃ < 3 ∧ a₄ < 4 ∧ a₅ < 5 ∧ a₆ < 6 ∧ a₇ < 7 →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 9 := by
sorry

end unique_fraction_sum_l2039_203944


namespace sum_of_max_pairs_nonnegative_l2039_203958

theorem sum_of_max_pairs_nonnegative 
  (a b c d : ℝ) 
  (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := by
  sorry

end sum_of_max_pairs_nonnegative_l2039_203958


namespace y_axis_symmetry_sum_l2039_203961

/-- Given a point M(a, b+3, 2c+1) with y-axis symmetric point M'(-4, -2, 15), prove a+b+c = -9 -/
theorem y_axis_symmetry_sum (a b c : ℝ) : 
  (a = 4) ∧ (b + 3 = -2) ∧ (2 * c + 1 = 15) → a + b + c = -9 := by
  sorry

end y_axis_symmetry_sum_l2039_203961


namespace snow_probability_l2039_203904

/-- The probability of no snow on each of the first five days -/
def no_snow_prob (n : ℕ) : ℚ :=
  if n ≤ 5 then (n + 1) / (n + 2) else 7/8

/-- The probability of snow on at least one day out of seven -/
def snow_prob : ℚ :=
  1 - (no_snow_prob 1 * no_snow_prob 2 * no_snow_prob 3 * no_snow_prob 4 * no_snow_prob 5 * no_snow_prob 6 * no_snow_prob 7)

theorem snow_probability : snow_prob = 139/384 := by
  sorry

end snow_probability_l2039_203904


namespace marcus_walking_speed_l2039_203929

/-- Calculates Marcus's walking speed given the conditions of his dog care routine -/
theorem marcus_walking_speed (bath_time : ℝ) (total_time : ℝ) (walk_distance : ℝ) : 
  bath_time = 20 →
  total_time = 60 →
  walk_distance = 3 →
  (walk_distance / (total_time - bath_time - bath_time / 2)) * 60 = 6 := by
sorry

end marcus_walking_speed_l2039_203929


namespace smallest_with_twelve_factors_l2039_203930

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The set of positive factors of a positive integer -/
def factors (n : ℕ+) : Set ℕ+ := sorry

theorem smallest_with_twelve_factors :
  ∃ (n : ℕ+), (num_factors n = 12) ∧
    (∀ m : ℕ+, m < n → num_factors m ≠ 12) ∧
    (n = 60) := by sorry

end smallest_with_twelve_factors_l2039_203930


namespace club_selection_count_l2039_203908

theorem club_selection_count (n : ℕ) (h : n = 18) : 
  n * (Nat.choose (n - 1) 2) = 2448 := by
  sorry

end club_selection_count_l2039_203908


namespace min_cost_is_84_l2039_203943

/-- Represents a salon with prices for haircut, facial cleaning, and nails --/
structure Salon where
  haircut : ℕ
  facial : ℕ
  nails : ℕ

/-- Calculates the total cost for a salon --/
def totalCost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

/-- The three salons with their respective prices --/
def gustranSalon : Salon := ⟨45, 22, 30⟩
def barbarasShop : Salon := ⟨30, 28, 40⟩
def fancySalon : Salon := ⟨34, 30, 20⟩

/-- Theorem stating that the minimum total cost among the three salons is 84 --/
theorem min_cost_is_84 : 
  min (totalCost gustranSalon) (min (totalCost barbarasShop) (totalCost fancySalon)) = 84 := by
  sorry


end min_cost_is_84_l2039_203943


namespace inequality_proof_l2039_203999

theorem inequality_proof (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3*(x-y)^2/4) (max (3*(y-z)^2/4) (3*(z-x)^2/4)) := by
  sorry

end inequality_proof_l2039_203999


namespace estimate_fish_population_l2039_203905

/-- Represents the fish pond scenario --/
structure FishPond where
  totalFish : ℕ  -- Total number of fish in the pond
  markedFish : ℕ  -- Number of fish initially marked
  secondSampleSize : ℕ  -- Size of the second sample
  markedInSecondSample : ℕ  -- Number of marked fish in the second sample

/-- Theorem stating the estimated number of fish in the pond --/
theorem estimate_fish_population (pond : FishPond) 
  (h1 : pond.markedFish = 100)
  (h2 : pond.secondSampleSize = 120)
  (h3 : pond.markedInSecondSample = 15) :
  pond.totalFish = 800 := by
  sorry

#check estimate_fish_population

end estimate_fish_population_l2039_203905


namespace all_days_equal_availability_l2039_203938

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the team members
inductive Member
| Alice
| Bob
| Charlie
| Diana

-- Define a function to represent availability
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Charlie, Day.Wednesday => false
  | Member.Charlie, Day.Thursday => false
  | Member.Charlie, Day.Friday => false
  | Member.Diana, Day.Monday => false
  | Member.Diana, Day.Tuesday => false
  | _, _ => true

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Charlie, Member.Diana]).length

-- Theorem: All days have equal availability
theorem all_days_equal_availability :
  ∀ d1 d2 : Day, availableCount d1 = availableCount d2 :=
sorry

end all_days_equal_availability_l2039_203938


namespace cost_of_dozen_pens_l2039_203901

/-- The cost of a pen in rupees -/
def pen_cost : ℝ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℝ := sorry

/-- The cost ratio of a pen to a pencil -/
def cost_ratio : ℝ := 5

/-- The total cost of 3 pens and 5 pencils in rupees -/
def total_cost : ℝ := 240

/-- The number of pens in a dozen -/
def dozen : ℕ := 12

theorem cost_of_dozen_pens :
  pen_cost = pencil_cost * cost_ratio ∧
  3 * pen_cost + 5 * pencil_cost = total_cost →
  dozen * pen_cost = 720 := by
  sorry

end cost_of_dozen_pens_l2039_203901


namespace cubic_function_constraint_l2039_203950

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- f has both a maximum and a minimum value -/
def has_max_and_min (a : ℝ) : Prop := ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

theorem cubic_function_constraint (a : ℝ) : 
  has_max_and_min a → a < -1 ∨ a > 2 := by sorry

end cubic_function_constraint_l2039_203950


namespace sum_of_integers_ending_in_3_l2039_203933

theorem sum_of_integers_ending_in_3 :
  let first_term : ℕ := 103
  let last_term : ℕ := 493
  let common_difference : ℕ := 10
  let n : ℕ := (last_term - first_term) / common_difference + 1
  (n : ℤ) * (first_term + last_term) / 2 = 11920 :=
by sorry

end sum_of_integers_ending_in_3_l2039_203933


namespace infinite_series_sum_l2039_203902

/-- The sum of the infinite series ∑(k=1 to ∞) k^3 / 2^k is equal to 26. -/
theorem infinite_series_sum : ∑' k : ℕ, (k^3 : ℝ) / 2^k = 26 := by
  sorry

end infinite_series_sum_l2039_203902


namespace total_fish_count_l2039_203969

-- Define the number of fish for each person
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish
def jenny_fish : ℕ := bobby_fish - 4

-- Theorem to prove
theorem total_fish_count : billy_fish + tony_fish + sarah_fish + bobby_fish + jenny_fish = 211 := by
  sorry

end total_fish_count_l2039_203969


namespace mans_swimming_speed_l2039_203973

/-- 
Given a man who swims against a current, this theorem proves his swimming speed in still water.
-/
theorem mans_swimming_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (h1 : distance = 40) 
  (h2 : time = 5) 
  (h3 : current_speed = 12) : 
  ∃ (speed : ℝ), speed = 20 ∧ distance = time * (speed - current_speed) :=
by sorry

end mans_swimming_speed_l2039_203973


namespace expressions_are_integers_l2039_203987

-- Define the expressions as functions
def expr1 (m n : ℕ) : ℚ := (m + n).factorial / (m.factorial * n.factorial)

def expr2 (m n : ℕ) : ℚ := ((2*m).factorial * (2*n).factorial) / 
  (m.factorial * n.factorial * (m + n).factorial)

def expr3 (m n : ℕ) : ℚ := ((5*m).factorial * (5*n).factorial) / 
  (m.factorial * n.factorial * (3*m + n).factorial * (3*n + m).factorial)

def expr4 (m n : ℕ) : ℚ := ((3*m + 3*n).factorial * (3*n).factorial * (2*m).factorial * (2*n).factorial) / 
  ((2*m + 3*n).factorial * (m + 2*n).factorial * m.factorial * (n.factorial^2) * (m + n).factorial)

-- Theorem statement
theorem expressions_are_integers (m n : ℕ) : 
  (∃ k : ℤ, expr1 m n = k) ∧ 
  (∃ k : ℤ, expr2 m n = k) ∧ 
  (∃ k : ℤ, expr3 m n = k) ∧ 
  (∃ k : ℤ, expr4 m n = k) := by
  sorry

end expressions_are_integers_l2039_203987


namespace seven_items_ten_people_distribution_l2039_203986

/-- The number of ways to distribute n unique items among m people,
    where no more than k people receive at least one item. -/
def distribution_ways (n m k : ℕ) : ℕ :=
  (Nat.choose m k) * (k^n)

/-- Theorem stating the correct number of ways to distribute
    7 unique items among 10 people, where no more than 3 people
    receive at least one item. -/
theorem seven_items_ten_people_distribution :
  distribution_ways 7 10 3 = 262440 := by
  sorry

end seven_items_ten_people_distribution_l2039_203986


namespace composition_ratio_theorem_l2039_203926

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_ratio_theorem :
  (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := by
  sorry

end composition_ratio_theorem_l2039_203926


namespace special_function_form_l2039_203962

/-- A positive continuous function satisfying the given inequality. -/
structure SpecialFunction where
  f : ℝ → ℝ
  continuous : Continuous f
  positive : ∀ x, f x > 0
  inequality : ∀ x y, f x - f y ≥ (x - y) * f ((x + y) / 2) * a
  a : ℝ

/-- The theorem stating that any function satisfying the SpecialFunction properties
    must be of the form c * exp(a * x) for some positive c. -/
theorem special_function_form (sf : SpecialFunction) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, sf.f x = c * Real.exp (sf.a * x) := by
  sorry

end special_function_form_l2039_203962


namespace one_root_l2039_203982

/-- A quadratic function f(x) = x^2 + bx + c with discriminant 2020 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The discriminant of f(x) = 0 is 2020 -/
axiom discriminant_is_2020 (b c : ℝ) : b^2 - 4*c = 2020

/-- The equation f(x - 2020) + f(x) = 0 has exactly one root -/
theorem one_root (b c : ℝ) : ∃! x, f b c (x - 2020) + f b c x = 0 :=
sorry

end one_root_l2039_203982


namespace class_factory_arrangements_l2039_203970

/-- The number of classes -/
def num_classes : ℕ := 5

/-- The number of factories -/
def num_factories : ℕ := 4

/-- The number of ways to arrange classes into factories -/
def arrangements : ℕ := 240

/-- Theorem stating the number of arrangements -/
theorem class_factory_arrangements :
  (∀ (arrangement : Fin num_classes → Fin num_factories),
    (∀ f : Fin num_factories, ∃ c : Fin num_classes, arrangement c = f) →
    (∀ c : Fin num_classes, arrangement c < num_factories)) →
  arrangements = 240 :=
sorry

end class_factory_arrangements_l2039_203970


namespace mike_camera_purchase_l2039_203975

/-- Given:
  - The new camera model costs 30% more than the current model
  - The old camera costs $4000
  - Mike gets $200 off a $400 lens

  Prove that Mike paid $5400 for the camera and lens. -/
theorem mike_camera_purchase (old_camera_cost : ℝ) (lens_cost : ℝ) (lens_discount : ℝ) :
  old_camera_cost = 4000 →
  lens_cost = 400 →
  lens_discount = 200 →
  let new_camera_cost := old_camera_cost * 1.3
  let discounted_lens_cost := lens_cost - lens_discount
  new_camera_cost + discounted_lens_cost = 5400 := by
  sorry

end mike_camera_purchase_l2039_203975


namespace max_k_value_l2039_203990

theorem max_k_value (a b k : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → a^2 + b^2 ≥ k → k ≤ 1/2 := by sorry

end max_k_value_l2039_203990


namespace paint_joined_cubes_paint_divided_cube_cube_division_l2039_203989

-- Constants
def paint_coverage : ℝ := 100 -- 1 mL covers 100 cm²

-- Theorem 1
theorem paint_joined_cubes (small_edge large_edge : ℝ) (h1 : small_edge = 10) (h2 : large_edge = 20) :
  (6 * small_edge^2 + 6 * large_edge^2 - 2 * small_edge^2) / paint_coverage = 28 :=
sorry

-- Theorem 2
theorem paint_divided_cube (original_paint : ℝ) (h : original_paint = 54) :
  2 * (original_paint / 6) = 18 :=
sorry

-- Theorem 3
theorem cube_division (original_paint additional_paint : ℝ) (n : ℕ)
  (h1 : original_paint = 54) (h2 : additional_paint = 216) :
  6 * (original_paint / 6) * n = original_paint + additional_paint →
  n = 5 :=
sorry

end paint_joined_cubes_paint_divided_cube_cube_division_l2039_203989


namespace perimeter_ratio_square_to_rectangle_l2039_203974

/-- The ratio of the perimeter of a square with side length 700 to the perimeter of a rectangle with length 400 and width 300 is 2:1 -/
theorem perimeter_ratio_square_to_rectangle : 
  let square_side : ℕ := 700
  let rect_length : ℕ := 400
  let rect_width : ℕ := 300
  let square_perimeter : ℕ := 4 * square_side
  let rect_perimeter : ℕ := 2 * (rect_length + rect_width)
  (square_perimeter : ℚ) / rect_perimeter = 2 / 1 := by
  sorry

end perimeter_ratio_square_to_rectangle_l2039_203974


namespace distribute_10_balls_3_boxes_l2039_203921

/-- The number of ways to distribute n identical balls into k boxes, where each box i must contain at least i balls. -/
def distributeWithMinimum (n : ℕ) (k : ℕ) : ℕ :=
  let remainingBalls := n - (k * (k + 1) / 2)
  Nat.choose (remainingBalls + k - 1) (k - 1)

/-- Theorem stating that there are 15 ways to distribute 10 identical balls into 3 boxes with the given conditions. -/
theorem distribute_10_balls_3_boxes : distributeWithMinimum 10 3 = 15 := by
  sorry

#eval distributeWithMinimum 10 3

end distribute_10_balls_3_boxes_l2039_203921


namespace work_completion_time_l2039_203996

/-- The number of days it takes for worker A to complete the work alone -/
def days_A : ℝ := 6

/-- The number of days it takes for worker B to complete the work alone -/
def days_B : ℝ := 5

/-- The number of days it takes for workers A, B, and C to complete the work together -/
def days_ABC : ℝ := 2

/-- The number of days it takes for worker C to complete the work alone -/
def days_C : ℝ := 7.5

theorem work_completion_time :
  (1 / days_A) + (1 / days_B) + (1 / days_C) = (1 / days_ABC) := by
  sorry

end work_completion_time_l2039_203996


namespace equal_intercept_line_equation_l2039_203980

/-- A line passing through (1,2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1,2) -/
  point_condition : k + b = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : k ≠ -1 → b = k * b

/-- The equation of the line is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.k = 2 ∧ l.b = 0) ∨ (l.k = 1 ∧ l.b = 1) :=
sorry

end equal_intercept_line_equation_l2039_203980


namespace simplify_trig_expression_l2039_203920

theorem simplify_trig_expression (α β : ℝ) :
  1 - Real.sin α ^ 2 - Real.sin β ^ 2 + 2 * Real.sin α * Real.sin β * Real.cos (α - β) = Real.cos (α - β) ^ 2 := by
  sorry

end simplify_trig_expression_l2039_203920


namespace unfair_die_expected_value_l2039_203977

/-- Represents an unfair eight-sided die -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℚ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℚ
  /-- The sum of all probabilities is 1 -/
  prob_sum : prob_eight + 7 * prob_others = 1
  /-- The probability of rolling an 8 is 3/8 -/
  eight_is_three_eighths : prob_eight = 3 / 8

/-- Calculates the expected value of a roll of the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  (d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7)) + (d.prob_eight * 8)

/-- Theorem stating that the expected value of the unfair die is 77/14 -/
theorem unfair_die_expected_value (d : UnfairDie) : expected_value d = 77 / 14 := by
  sorry

end unfair_die_expected_value_l2039_203977


namespace tan_function_property_l2039_203976

/-- Given a function f(x) = a tan(bx) where a and b are positive constants,
    if f has roots at ±π/4 and passes through (π/8, 1), then a · b = 2 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = 0 ↔ x = π/4 ∨ x = -π/4) →
  a * Real.tan (b * π/8) = 1 →
  a * b = 2 := by sorry

end tan_function_property_l2039_203976


namespace sixth_power_sum_l2039_203937

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 511 := by
  sorry

end sixth_power_sum_l2039_203937


namespace min_lines_to_cover_plane_l2039_203981

-- Define the circle on a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a line on a plane
structure Line :=
  (a b c : ℝ)

-- Define a reflection of a point with respect to a line
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

-- Define a function to check if a point is covered by a circle
def is_covered (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define a function to perform a finite sequence of reflections
def reflect_sequence (c : Circle) (lines : List Line) : Circle := sorry

-- Theorem statement
theorem min_lines_to_cover_plane (c : Circle) :
  ∃ (lines : List Line),
    (lines.length = 3) ∧
    (∀ (p : ℝ × ℝ), ∃ (seq : List Line),
      (∀ (l : Line), l ∈ seq → l ∈ lines) ∧
      is_covered p (reflect_sequence c seq)) ∧
    (∀ (lines' : List Line),
      lines'.length < 3 →
      ∃ (p : ℝ × ℝ), ∀ (seq : List Line),
        (∀ (l : Line), l ∈ seq → l ∈ lines') →
        ¬is_covered p (reflect_sequence c seq)) :=
sorry

end min_lines_to_cover_plane_l2039_203981


namespace order_of_magnitudes_l2039_203985

theorem order_of_magnitudes (x a : ℝ) (hx : x < 0) (ha : a = 2 * x) :
  x^2 < a * x ∧ a * x < a^2 := by
  sorry

end order_of_magnitudes_l2039_203985


namespace exists_a_for_f_with_real_domain_and_range_l2039_203931

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

-- State the theorem
theorem exists_a_for_f_with_real_domain_and_range :
  ∃ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a y = x) ∧ (∀ y : ℝ, ∃ x : ℝ, f a x = y) := by
  sorry

end exists_a_for_f_with_real_domain_and_range_l2039_203931


namespace largest_solution_and_ratio_l2039_203916

theorem largest_solution_and_ratio : ∃ (a b c d : ℤ),
  let x : ℝ := (a + b * Real.sqrt c) / d
  (7 * x) / 9 + 2 = 4 / x ∧
  (∀ (a' b' c' d' : ℤ), 
    let x' : ℝ := (a' + b' * Real.sqrt c') / d'
    (7 * x') / 9 + 2 = 4 / x' → x' ≤ x) ∧
  x = (-9 + 3 * Real.sqrt 111) / 7 ∧
  a * c * d / b = -2313 :=
by sorry

end largest_solution_and_ratio_l2039_203916


namespace apple_selling_price_l2039_203956

/-- The selling price of an apple given its cost price and loss ratio -/
def selling_price (cost_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  cost_price * (1 - loss_ratio)

/-- Theorem stating the selling price of an apple with given conditions -/
theorem apple_selling_price :
  let cost_price : ℚ := 20
  let loss_ratio : ℚ := 1/6
  selling_price cost_price loss_ratio = 50/3 := by
sorry

end apple_selling_price_l2039_203956


namespace restaurant_order_combinations_l2039_203954

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 3

/-- The number of specialty dishes -/
def specialty_dishes : ℕ := 3

/-- The number of different meal combinations -/
def meal_combinations : ℕ := 1611

theorem restaurant_order_combinations :
  (menu_items ^ num_people) - (num_people * (specialty_dishes * (menu_items - specialty_dishes) ^ (num_people - 1))) = meal_combinations := by
  sorry

end restaurant_order_combinations_l2039_203954


namespace tan_period_l2039_203919

/-- The smallest positive period of tan((a + b)x/2) given conditions -/
theorem tan_period (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = 1) : 
  let f := fun x => Real.tan ((a + b) * x / 2)
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = 2 * Real.pi := by
  sorry

end tan_period_l2039_203919


namespace alex_remaining_money_l2039_203978

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def water_bill : ℝ := 55

theorem alex_remaining_money :
  weekly_income - (weekly_income * tax_rate + weekly_income * tithe_rate + water_bill) = 345 :=
by sorry

end alex_remaining_money_l2039_203978


namespace bezdikovPopulationTheorem_l2039_203923

/-- Represents the population of Bezdíkov -/
structure BezdikovPopulation where
  women1966 : ℕ
  men1966 : ℕ
  womenNow : ℕ
  menNow : ℕ

/-- Conditions for the Bezdíkov population problem -/
def bezdikovConditions (p : BezdikovPopulation) : Prop :=
  p.women1966 = p.men1966 + 30 ∧
  p.womenNow = p.women1966 / 4 ∧
  p.menNow = p.men1966 - 196 ∧
  p.womenNow = p.menNow + 10

/-- The theorem stating that the current total population of Bezdíkov is 134 -/
theorem bezdikovPopulationTheorem (p : BezdikovPopulation) 
  (h : bezdikovConditions p) : p.womenNow + p.menNow = 134 :=
by
  sorry

#check bezdikovPopulationTheorem

end bezdikovPopulationTheorem_l2039_203923


namespace ivy_cupcakes_l2039_203966

theorem ivy_cupcakes (morning_cupcakes : ℕ) (afternoon_difference : ℕ) : 
  morning_cupcakes = 20 →
  afternoon_difference = 15 →
  morning_cupcakes + (morning_cupcakes + afternoon_difference) = 55 :=
by
  sorry

end ivy_cupcakes_l2039_203966


namespace largest_prime_factor_of_3913_l2039_203960

theorem largest_prime_factor_of_3913 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3913 → q ≤ p :=
by sorry

end largest_prime_factor_of_3913_l2039_203960


namespace triangle_inequality_sum_largest_coefficient_l2039_203998

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b * c / (b + c - a)) + (a * c / (a + c - b)) + (a * b / (a + b - c)) ≥ (a + b + c) :=
sorry

theorem largest_coefficient (k : ℝ) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → c + a > b →
    (b * c / (b + c - a)) + (a * c / (a + c - b)) + (a * b / (a + b - c)) ≥ k * (a + b + c)) →
  k ≤ 1 :=
sorry

end triangle_inequality_sum_largest_coefficient_l2039_203998


namespace trig_expression_equality_l2039_203917

theorem trig_expression_equality : 
  2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2 := by
  sorry

end trig_expression_equality_l2039_203917


namespace quadratic_prime_values_l2039_203953

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

/-- Predicate to check if a number is prime -/
def IsPrime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem quadratic_prime_values 
  (a b c : ℤ) (n : ℤ) :
  (IsPrime (QuadraticPolynomial a b c (n - 1))) →
  (IsPrime (QuadraticPolynomial a b c n)) →
  (IsPrime (QuadraticPolynomial a b c (n + 1))) →
  ∃ m : ℤ, m ≠ n - 1 ∧ m ≠ n ∧ m ≠ n + 1 ∧ IsPrime (QuadraticPolynomial a b c m) :=
by sorry

end quadratic_prime_values_l2039_203953


namespace leila_cake_count_l2039_203913

/-- The number of cakes Leila ate on Monday -/
def monday_cakes : ℕ := 6

/-- The number of cakes Leila ate on Friday -/
def friday_cakes : ℕ := 9

/-- The number of cakes Leila ate on Saturday -/
def saturday_cakes : ℕ := 3 * monday_cakes

/-- The total number of cakes Leila ate -/
def total_cakes : ℕ := monday_cakes + friday_cakes + saturday_cakes

theorem leila_cake_count : total_cakes = 33 := by
  sorry

end leila_cake_count_l2039_203913


namespace inequality_proof_l2039_203909

theorem inequality_proof (a b c d : ℝ) : 
  (a + c)^2 * (b + d)^2 - 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) ≥ 0 ∧ 
  (a + c)^2 * (b + d)^2 - 4 * b * c * (c * d + d * a + a * b) ≥ 0 := by
sorry

end inequality_proof_l2039_203909


namespace min_cost_for_zoo_visit_l2039_203997

/-- Represents the ticket pricing structure for the zoo --/
structure TicketPrices where
  adult : ℕ
  child : ℕ
  group : ℕ
  group_min : ℕ

/-- Calculates the total cost for a group given the pricing and number of adults and children --/
def calculate_cost (prices : TicketPrices) (adults children : ℕ) : ℕ :=
  min (prices.adult * adults + prices.child * children)
      (min (prices.group * (adults + children))
           (prices.group * prices.group_min + prices.child * (adults + children - prices.group_min)))

/-- Theorem stating the minimum cost for the given group --/
theorem min_cost_for_zoo_visit (prices : TicketPrices) 
    (h1 : prices.adult = 150)
    (h2 : prices.child = 60)
    (h3 : prices.group = 100)
    (h4 : prices.group_min = 5) :
  calculate_cost prices 4 7 = 860 := by
  sorry

end min_cost_for_zoo_visit_l2039_203997


namespace sqrt_equation_solution_l2039_203949

theorem sqrt_equation_solution :
  ∃ x : ℝ, x = 196 ∧ Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4) :=
by sorry

end sqrt_equation_solution_l2039_203949


namespace josh_film_cost_l2039_203914

/-- The cost of each film Josh bought -/
def film_cost : ℚ := 5

/-- The number of films Josh bought -/
def num_films : ℕ := 9

/-- The number of books Josh bought -/
def num_books : ℕ := 4

/-- The cost of each book -/
def book_cost : ℚ := 4

/-- The number of CDs Josh bought -/
def num_cds : ℕ := 6

/-- The cost of each CD -/
def cd_cost : ℚ := 3

/-- The total amount Josh spent -/
def total_spent : ℚ := 79

theorem josh_film_cost :
  film_cost * num_films + book_cost * num_books + cd_cost * num_cds = total_spent :=
by sorry

end josh_film_cost_l2039_203914


namespace eggs_sold_equals_540_l2039_203952

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 36

/-- The initial number of trays collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped accidentally -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added -/
def additional_trays : ℕ := 7

/-- The total number of eggs sold -/
def total_eggs_sold : ℕ := eggs_per_tray * (initial_trays - dropped_trays + additional_trays)

theorem eggs_sold_equals_540 : total_eggs_sold = 540 := by
  sorry

end eggs_sold_equals_540_l2039_203952


namespace rex_saved_100_nickels_l2039_203972

/-- Represents the number of coins of each type saved by the children -/
structure Savings where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Converts a number of coins to their value in cents -/
def coinsToCents (s : Savings) : ℕ :=
  s.pennies + 5 * s.nickels + 10 * s.dimes

/-- The main theorem: Given the conditions, Rex saved 100 nickels -/
theorem rex_saved_100_nickels (s : Savings) 
    (h1 : s.pennies = 200)
    (h2 : s.dimes = 330)
    (h3 : coinsToCents s = 4000) : 
  s.nickels = 100 := by
  sorry

end rex_saved_100_nickels_l2039_203972


namespace regular_polygon_with_150_degree_angles_has_12_sides_l2039_203995

/-- A regular polygon with interior angles of 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
  sorry

end regular_polygon_with_150_degree_angles_has_12_sides_l2039_203995


namespace poly_arrangement_l2039_203918

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := -2 * x^3 * y + 4 * x * y^3 + 1 - 3 * x^2 * y^2

/-- The polynomial arranged in descending order of y -/
def arranged_poly (x y : ℝ) : ℝ := 4 * x * y^3 - 3 * x^2 * y^2 - 2 * x^3 * y + 1

/-- Theorem stating that the original polynomial is equal to the arranged polynomial -/
theorem poly_arrangement (x y : ℝ) : original_poly x y = arranged_poly x y := by
  sorry

end poly_arrangement_l2039_203918


namespace range_of_2x_minus_y_l2039_203936

theorem range_of_2x_minus_y (x y : ℝ) 
  (hx : 0 < x ∧ x < 4) 
  (hy : 0 < y ∧ y < 6) : 
  -6 < 2*x - y ∧ 2*x - y < 8 := by
  sorry

end range_of_2x_minus_y_l2039_203936


namespace max_spheres_in_cube_l2039_203910

/-- Represents a three-dimensional cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents a sphere -/
structure Sphere where
  diameter : ℝ

/-- Calculates the maximum number of spheres that can fit in a cube -/
def maxSpheres (c : Cube) (s : Sphere) : ℕ :=
  sorry

/-- Theorem stating the maximum number of spheres in the given cube -/
theorem max_spheres_in_cube :
  ∃ (c : Cube) (s : Sphere),
    c.edgeLength = 4 ∧ s.diameter = 1 ∧ maxSpheres c s = 66 :=
by
  sorry

end max_spheres_in_cube_l2039_203910


namespace tetrahedron_non_coplanar_selections_l2039_203915

/-- The number of ways to select 4 non-coplanar points from a tetrahedron -/
def nonCoplanarSelections : ℕ := 141

/-- Total number of points on the tetrahedron -/
def totalPoints : ℕ := 10

/-- Number of vertices of the tetrahedron -/
def vertices : ℕ := 4

/-- Number of midpoints of edges -/
def midpoints : ℕ := 6

/-- Number of points to be selected -/
def selectPoints : ℕ := 4

/-- Theorem stating that the number of ways to select 4 non-coplanar points
    from 10 points on a tetrahedron (4 vertices and 6 midpoints of edges) is 141 -/
theorem tetrahedron_non_coplanar_selections :
  totalPoints = vertices + midpoints ∧
  nonCoplanarSelections = 141 :=
sorry

end tetrahedron_non_coplanar_selections_l2039_203915


namespace impossible_transformation_l2039_203900

/-- Represents a natural number and its digits -/
structure DigitNumber where
  value : ℕ
  digits : List ℕ
  digits_valid : digits.all (· < 10)
  value_eq_digits : value = digits.foldl (fun acc d => acc * 10 + d) 0

/-- Defines the allowed operations on a DigitNumber -/
inductive Operation
  | multiply_by_two : Operation
  | rearrange_digits : Operation

/-- Applies an operation to a DigitNumber -/
def apply_operation (n : DigitNumber) (op : Operation) : DigitNumber :=
  match op with
  | Operation.multiply_by_two => sorry
  | Operation.rearrange_digits => sorry

/-- Checks if a DigitNumber is valid (non-zero first digit) -/
def is_valid (n : DigitNumber) : Prop :=
  n.digits.head? ≠ some 0

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a DigitNumber -/
def apply_sequence (n : DigitNumber) (seq : OperationSequence) : DigitNumber :=
  seq.foldl apply_operation n

theorem impossible_transformation :
  ¬∃ (seq : OperationSequence),
    let start : DigitNumber := ⟨1, [1], sorry, sorry⟩
    let result := apply_sequence start seq
    result.value = 811 ∧ is_valid result :=
  sorry

end impossible_transformation_l2039_203900


namespace statement_equivalence_l2039_203993

theorem statement_equivalence (x y : ℝ) :
  ((x > 1 ∧ y < -3) → x - y > 4) ↔ (x - y ≤ 4 → x ≤ 1 ∨ y ≥ -3) :=
by sorry

end statement_equivalence_l2039_203993


namespace max_candy_pieces_l2039_203946

theorem max_candy_pieces (n : ℕ) (mean : ℚ) (min_pieces : ℕ) 
  (h1 : n = 40)
  (h2 : mean = 4)
  (h3 : min_pieces = 2) :
  ∃ (max_pieces : ℕ), max_pieces = 82 ∧ 
  (∀ (student_pieces : List ℕ), 
    student_pieces.length = n ∧ 
    (∀ x ∈ student_pieces, x ≥ min_pieces) ∧
    (student_pieces.sum / n : ℚ) = mean →
    ∀ x ∈ student_pieces, x ≤ max_pieces) :=
by sorry

end max_candy_pieces_l2039_203946


namespace quilt_cost_theorem_l2039_203942

def quilt_width : ℕ := 16
def quilt_length : ℕ := 20
def patch_area : ℕ := 4
def initial_patch_cost : ℕ := 10
def initial_patch_count : ℕ := 10

def total_quilt_area : ℕ := quilt_width * quilt_length
def total_patches : ℕ := total_quilt_area / patch_area
def discounted_patch_cost : ℕ := initial_patch_cost / 2
def discounted_patches : ℕ := total_patches - initial_patch_count

def total_cost : ℕ := initial_patch_count * initial_patch_cost + discounted_patches * discounted_patch_cost

theorem quilt_cost_theorem : total_cost = 450 := by
  sorry

end quilt_cost_theorem_l2039_203942
