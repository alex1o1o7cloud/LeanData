import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l498_49831

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (1, -2) to the line 3x + 4y + 10 = 0 is 1 -/
theorem distance_point_to_line_example : distance_point_to_line 1 (-2) 3 4 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l498_49831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_equals_sqrt32_plus_sqrt48_l498_49841

/-- The distance between the outer vertices of two equilateral triangles
    mounted on adjacent sides of a square --/
noncomputable def distance_AB (square_side : ℝ) : ℝ :=
  let triangle_height := square_side * Real.sqrt 3 / 2
  let x_diff := square_side + triangle_height - square_side / 2
  let y_diff := square_side / 2 + triangle_height - square_side / 2
  Real.sqrt (x_diff ^ 2 + y_diff ^ 2)

/-- Theorem stating that for a square with side length 4, the distance between
    the outer vertices of two equilateral triangles mounted on adjacent sides
    can be written as √32 + √48 --/
theorem distance_AB_equals_sqrt32_plus_sqrt48 :
  distance_AB 4 = Real.sqrt 32 + Real.sqrt 48 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_AB 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_equals_sqrt32_plus_sqrt48_l498_49841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_esther_distance_proof_l498_49837

/-- Proves that Esther walked 975 yards given the distances walked by Lionel and Niklaus and the total distance walked by all three friends. -/
def esther_distance (lionel_miles : ℕ) (niklaus_feet : ℕ) (total_feet : ℕ) : ℕ :=
  let feet_per_mile : ℕ := 5280
  let feet_per_yard : ℕ := 3
  let lionel_feet : ℕ := lionel_miles * feet_per_mile
  let esther_feet : ℕ := total_feet - lionel_feet - niklaus_feet
  esther_feet / feet_per_yard

theorem esther_distance_proof (lionel_miles niklaus_feet total_feet : ℕ) 
  (h1 : lionel_miles = 4) 
  (h2 : niklaus_feet = 1287) 
  (h3 : total_feet = 25332) : 
  esther_distance lionel_miles niklaus_feet total_feet = 975 := by
  sorry

#eval esther_distance 4 1287 25332

end NUMINAMATH_CALUDE_ERRORFEEDBACK_esther_distance_proof_l498_49837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l498_49807

/-- The length of a bridge that a train can cross -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem stating the length of the bridge -/
theorem bridge_length_calculation :
  bridge_length 120 45 30 = 255 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l498_49807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_points_range_l498_49808

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

def is_mean_value_point (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  a < m ∧ m < b ∧ f b - f a = (deriv f m) * (b - a)

theorem mean_value_points_range :
  ∀ b : ℝ, (∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ 
    is_mean_value_point f 0 b m₁ ∧ 
    is_mean_value_point f 0 b m₂ ∧
    (∀ m : ℝ, is_mean_value_point f 0 b m → m = m₁ ∨ m = m₂)) ↔
  (3/2 < b ∧ b < 3) := by
  sorry

#check mean_value_points_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_points_range_l498_49808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kan_subtraction_l498_49858

/-- Represents a digit (0-9) -/
def Digit := Fin 10

theorem kan_subtraction
  (K A N G R O : Digit)
  (distinct : K ≠ A ∧ K ≠ N ∧ K ≠ G ∧ K ≠ R ∧ K ≠ O ∧
              A ≠ N ∧ A ≠ G ∧ A ≠ R ∧ A ≠ O ∧
              N ≠ G ∧ N ≠ R ∧ N ≠ O ∧
              G ≠ R ∧ G ≠ O ∧
              R ≠ O)
  (sum_equation : (100 * K.val + 10 * A.val + N.val) + (10 * G.val + A.val) = 100 * R.val + 10 * O.val + O.val)
  (r_relation : R.val = K.val + 1)
  (n_relation : N.val = G.val + 1)
  : (10 * R.val + N.val) - (10 * K.val + G.val) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kan_subtraction_l498_49858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_sided_figure_area_l498_49862

/-- A twelve-sided figure on a 1 cm × 1 cm grid -/
structure TwelveSidedFigure where
  full_squares : ℕ
  right_triangles : ℕ
  hfs : full_squares = 9
  hrt : right_triangles = 8

/-- The area of a right-angled triangle with legs of length 1 cm -/
noncomputable def triangle_area : ℚ := 1 / 2

/-- The area of the twelve-sided figure -/
noncomputable def figure_area (f : TwelveSidedFigure) : ℚ :=
  f.full_squares + (f.right_triangles / 2 : ℚ)

theorem twelve_sided_figure_area (f : TwelveSidedFigure) : 
  figure_area f = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_sided_figure_area_l498_49862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brownie_cost_is_correct_l498_49880

/-- The cost of a "Build Your Own Hot Brownie" dessert --/
structure BrownieDessert where
  brownieCost : ℚ
  iceCreamScoops : ℕ
  syrups : ℕ
  hasNuts : Bool

/-- The total cost of a brownie dessert --/
def totalCost (dessert : BrownieDessert) : ℚ :=
  dessert.brownieCost +
  1 * dessert.iceCreamScoops +
  (1/2) * dessert.syrups +
  (if dessert.hasNuts then (3/2) else 0)

/-- Juanita's dessert order --/
def juanitaOrder : BrownieDessert :=
  { brownieCost := 5/2,  -- This is what we want to prove
    iceCreamScoops := 2,
    syrups := 2,
    hasNuts := true }

theorem brownie_cost_is_correct :
  totalCost juanitaOrder = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brownie_cost_is_correct_l498_49880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_inequality_l498_49806

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, (2 : ℝ)^x > 0) ↔ (∀ x : ℝ, (2 : ℝ)^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_inequality_l498_49806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_property_star_property_2_star_one_one_fifteen_star_twenty_five_l498_49891

noncomputable section

def star (x y : ℝ) : ℝ := Real.log x / Real.log y

theorem star_property (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  star (x^2) y = star x y + star x y := by
  sorry

theorem star_property_2 (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  star x (star y y) = star (star x y) (star x 1) := by
  sorry

theorem star_one_one : star 1 1 = 0 := by
  sorry

theorem fifteen_star_twenty_five :
  star 15 25 = (Real.log 3) / (2 * Real.log 5) + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_property_star_property_2_star_one_one_fifteen_star_twenty_five_l498_49891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nandan_earning_is_2000_l498_49839

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℚ
  nandan_time : ℚ
  krishan_investment : ℚ
  krishan_time : ℚ
  total_gain : ℚ

/-- Calculates Nandan's earning given the business investment scenario -/
def nandan_earning (b : BusinessInvestment) : ℚ :=
  b.total_gain / (1 + (b.krishan_investment * b.krishan_time) / (b.nandan_investment * b.nandan_time))

theorem nandan_earning_is_2000 (b : BusinessInvestment) : 
  b.krishan_investment = 4 * b.nandan_investment →
  b.krishan_time = 3 * b.nandan_time →
  b.total_gain = 26000 →
  nandan_earning b = 2000 := by
  sorry

#eval nandan_earning { nandan_investment := 1, nandan_time := 1, krishan_investment := 4, krishan_time := 3, total_gain := 26000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nandan_earning_is_2000_l498_49839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_l498_49886

noncomputable def a : Fin 2 → ℝ := ![1, -Real.sqrt 3]

theorem vector_b (b : Fin 2 → ℝ) 
  (h1 : ∃ k : ℝ, b = k • a) 
  (h2 : ‖b‖ = 1) : 
  b = ![1/2, -Real.sqrt 3 / 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_b_l498_49886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_difference_l498_49860

theorem max_value_of_exponential_difference :
  ∃ (x : ℝ), ∀ (y : ℝ), (2 : ℝ)^x - (4 : ℝ)^x ≥ (2 : ℝ)^y - (4 : ℝ)^y ∧ (2 : ℝ)^x - (4 : ℝ)^x = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_exponential_difference_l498_49860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_1_evaluate_expression_2_l498_49894

-- Part 1
theorem evaluate_expression_1 : 2 * Real.sqrt 3 * (1.5 ^ (1/3)) * (12 ^ (1/6)) = 6 := by sorry

-- Part 2
theorem evaluate_expression_2 : (Real.log 27 / Real.log 8) * (Real.log 4 / Real.log 3) + 3 ^ (Real.log 2 / Real.log 3) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_1_evaluate_expression_2_l498_49894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tile_problem_l498_49873

/-- Represents the number of tiles a big horse can pull -/
def big_horse_capacity : ℚ := 3

/-- Represents the number of small horses needed to pull one tile -/
def small_horses_per_tile : ℚ := 3

/-- Represents the total number of horses -/
def total_horses : ℕ := 100

/-- Represents the total number of tiles -/
def total_tiles : ℕ := 100

/-- Proves that the system of equations correctly represents the horse-tile problem -/
theorem horse_tile_problem (x y : ℚ) :
  (x + y = total_horses ∧ 
   big_horse_capacity * x + (1 / small_horses_per_tile) * y = total_tiles) ↔
  (x ≥ 0 ∧ y ≥ 0 ∧
   x + y = total_horses ∧
   3 * x + (1 / 3) * y = total_tiles) := by
  sorry

#check horse_tile_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_tile_problem_l498_49873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l498_49830

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def focal_length (a b : ℝ) : ℝ := 4 * Real.sqrt 2

def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

def point_relation (P P₁ P₂ : ℝ × ℝ) : Prop :=
  3 * P.1 = P₁.1 + 2 * P₂.1 ∧ 3 * P.2 = P₁.2 + 2 * P₂.2

theorem hyperbola_properties
  (a b : ℝ) (P P₁ P₂ : ℝ × ℝ)
  (ha : a > 0) (hb : b > 0)
  (hP : hyperbola a b P.1 P.2)
  (hfl : focal_length a b = 4 * Real.sqrt 2)
  (hasy₁ : asymptote a b P₁.1 P₁.2)
  (hasy₂ : asymptote a b P₂.1 P₂.2)
  (hrel : point_relation P P₁ P₂) :
  (P₁.1 * P₂.1 - P₁.2 * P₂.2 = 9) ∧
  (∃ (S : ℝ), S = (9 / 2) ∧ ∀ (a' b' : ℝ), hyperbola a' b' P.1 P.2 →
    (focal_length a' b' = 4 * Real.sqrt 2) →
    (a' = b' → a' = 2) ∧ (a' = 2 → b' = 2)) := by
  sorry

#check hyperbola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l498_49830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l498_49884

/-- Predicate for arithmetic sequence -/
def is_arithmetic_seq (s : List ℝ) : Prop :=
  ∃ d, ∀ i, i + 1 < s.length → s[i+1]! - s[i]! = d

/-- Predicate for geometric sequence -/
def is_geometric_seq (s : List ℝ) : Prop :=
  ∃ r, ∀ i, i + 1 < s.length → s[i+1]! / s[i]! = r

/-- Given that (1, a₁, a₂, 9) is an arithmetic sequence and (1, b₁, b₂, b₃, 9) is a geometric sequence,
    prove that b₂ / (a₁ + a₂) = 3/10 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h_arith : is_arithmetic_seq [1, a₁, a₂, 9])
  (h_geom : is_geometric_seq [1, b₁, b₂, b₃, 9]) :
  b₂ / (a₁ + a₂) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l498_49884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_tangent_line_equation_l498_49863

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem for the definite integral
theorem integral_value :
  ∫ x in (-3)..3, (f x + x^2) = 18 :=
sorry

-- Theorem for the tangent line
theorem tangent_line_equation :
  ∃ (m : ℝ), 
    (f m - (-2) = (3 * m^2 + 1) * (m - 0)) ∧ 
    (∀ x, f m + (3 * m^2 + 1) * (x - m) = 4 * x - 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_tangent_line_equation_l498_49863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_log_base_2_l498_49888

theorem derivative_log_base_2 (x : ℝ) (h : x > 0) :
  deriv (λ y => Real.log y / Real.log 2) x = 1 / (x * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_log_base_2_l498_49888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_rectangle_l498_49810

theorem min_diagonal_rectangle (l w : ℝ) : 
  (l + w = 10) →  -- Perimeter constraint (half of 20)
  l > 0 → w > 0 → -- Positive dimensions
  (l^2 + w^2) ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_rectangle_l498_49810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_is_nine_l498_49874

def is_valid_number (n : ℕ) : Prop :=
  (Nat.digits 10 n).length = 1998 ∧
  (∀ i : ℕ, i < 1997 → 
    let pair := (Nat.digits 10 n)[i]! * 10 + (Nat.digits 10 n)[i+1]!
    pair % 17 = 0 ∨ pair % 23 = 0) ∧
  (Nat.digits 10 n).getLast? = some 1

theorem first_digit_is_nine (n : ℕ) (h : is_valid_number n) : 
  (Nat.digits 10 n).head? = some 9 := by
  sorry

#check first_digit_is_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_is_nine_l498_49874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_infinity_l498_49827

theorem limit_fraction_infinity : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((n : ℝ) + 5) * (1 - 3*(n : ℝ)) / ((2*(n : ℝ) + 1)^2) - (-3/4)| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_fraction_infinity_l498_49827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_40_has_non_power_sum_l498_49838

/-- An arithmetic sequence of 40 distinct positive integers -/
structure ArithmeticSequence40 where
  terms : Fin 40 → ℕ
  distinct : ∀ i j, i ≠ j → terms i ≠ terms j
  arithmetic : ∀ i j k, i.val + k = j.val → terms j = terms i + k * (terms 1 - terms 0)
  positive : ∀ i, terms i > 0

/-- Expression of a number as 2^k + 3^l -/
def isPowerSum (n : ℕ) : Prop :=
  ∃ k l : ℕ, n = 2^k + 3^l

theorem arithmetic_sequence_40_has_non_power_sum (seq : ArithmeticSequence40) :
  ∃ i : Fin 40, ¬isPowerSum (seq.terms i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_40_has_non_power_sum_l498_49838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_solution_set_l498_49870

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (11 - x)

-- Define the maximum value M
noncomputable def M : ℝ := 3 * Real.sqrt 2

-- Theorem for the maximum value of f
theorem f_max_value : 
  ∀ x : ℝ, 2 < x ∧ x < 11 → f x ≤ M :=
by sorry

-- Theorem for the solution set of the inequality
theorem solution_set :
  ∀ x : ℝ, |x - Real.sqrt 2| + |x + 2 * Real.sqrt 2| ≤ M ↔ 
  -2 * Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_solution_set_l498_49870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_four_or_sixteen_l498_49899

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The set of possible outcomes when rolling two standard dice -/
def twoStandardDiceOutcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range standardDieSides) (Finset.range standardDieSides)

/-- The sum of two numbers -/
def sum (p : ℕ × ℕ) : ℕ := p.1 + p.2

/-- The set of outcomes that result in a sum of 4 or 16 -/
def sumFourOrSixteen : Finset (ℕ × ℕ) :=
  twoStandardDiceOutcomes.filter (λ p => sum p = 4 ∨ sum p = 16)

/-- The probability of an event occurring when rolling two standard dice -/
def probability (event : Finset (ℕ × ℕ)) : ℚ :=
  (event.card : ℚ) / (twoStandardDiceOutcomes.card : ℚ)

theorem probability_sum_four_or_sixteen :
  probability sumFourOrSixteen = 1/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_four_or_sixteen_l498_49899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_intersection_l498_49811

-- Define a type for line segments
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

-- Define a type for lines
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a function to check if a line intersects a segment
def intersects (l : Line) (s : LineSegment) : Prop :=
  ∃ x y, l.slope * x + l.intercept = y ∧ 
         s.start.1 ≤ x ∧ x ≤ s.endpoint.1 ∧
         s.start.2 ≤ y ∧ y ≤ s.endpoint.2

-- State the theorem
theorem parallel_segments_intersection 
  (segments : Set LineSegment) 
  (parallel : ∀ s₁ s₂, s₁ ∈ segments → s₂ ∈ segments → s₁.start.1 = s₂.start.1 ∧ s₁.endpoint.1 = s₂.endpoint.1) 
  (three_intersection : ∀ s₁ s₂ s₃, s₁ ∈ segments → s₂ ∈ segments → s₃ ∈ segments → 
    ∃ l : Line, intersects l s₁ ∧ intersects l s₂ ∧ intersects l s₃) :
  ∃ l : Line, ∀ s ∈ segments, intersects l s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_intersection_l498_49811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_cookie_purchase_l498_49820

/-- Represents the grocery purchase scenario -/
structure GroceryPurchase where
  total_cost : ℚ
  milk_cost : ℚ
  cereal_cost : ℚ
  cereal_boxes : ℕ
  banana_cost : ℚ
  banana_count : ℕ
  apple_cost : ℚ
  apple_count : ℕ
  cookie_box_cost : ℚ

/-- Calculates the number of cookie boxes bought given a grocery purchase -/
def cookie_boxes (purchase : GroceryPurchase) : ℚ :=
  (purchase.total_cost - 
   (purchase.milk_cost + 
    purchase.cereal_cost * purchase.cereal_boxes + 
    purchase.banana_cost * purchase.banana_count + 
    purchase.apple_cost * purchase.apple_count)) / 
  purchase.cookie_box_cost

/-- Theorem stating that Steve bought 2 boxes of cookies -/
theorem steve_cookie_purchase : 
  let purchase := GroceryPurchase.mk 25 3 (7/2) 2 (1/4) 4 (1/2) 4 6
  cookie_boxes purchase = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_cookie_purchase_l498_49820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l498_49840

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), ∀ y ∈ Set.Ioo 0 (Real.pi / 2), x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l498_49840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_reciprocal_sum_l498_49822

/-- Given a quadratic polynomial x² - sx + p with roots r₁ and r₂ satisfying
    r₁ + r₂ = r₁² + r₂² = r₁³ + r₂³ = ... = r₁²⁰⁰⁷ + r₂²⁰⁰⁷,
    the maximum value of 1/r₁²⁰⁰⁸ + 1/r₂²⁰⁰⁸ is 2. -/
theorem max_value_of_reciprocal_sum (s p r₁ r₂ : ℝ) : 
  (r₁^2 - s*r₁ + p = 0) → 
  (r₂^2 - s*r₂ + p = 0) →
  (∀ n : ℕ, 1 ≤ n → n ≤ 2007 → r₁^n + r₂^n = r₁ + r₂) →
  (∃ (x : ℝ), (1/r₁^2008 + 1/r₂^2008 ≤ x) ∧ 
   (∀ (y : ℝ), (1/r₁^2008 + 1/r₂^2008 ≤ y) → x ≤ y)) →
  (1/r₁^2008 + 1/r₂^2008 ≤ 2) ∧ 
  (∃ (s p r₁ r₂ : ℝ), (r₁^2 - s*r₁ + p = 0) ∧ 
                      (r₂^2 - s*r₂ + p = 0) ∧
                      (∀ n : ℕ, 1 ≤ n → n ≤ 2007 → r₁^n + r₂^n = r₁ + r₂) ∧
                      (1/r₁^2008 + 1/r₂^2008 = 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_reciprocal_sum_l498_49822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_l498_49872

-- Define the plane
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points A and B
variable (A B : V)

-- Define the set of points P satisfying the condition
def trajectory (A B : V) : Set V :=
  {P : V | ‖P - A‖ = 2 * ‖P - B‖}

-- Theorem statement
theorem trajectory_is_circle (A B : V) :
  ∃ (center : V) (radius : ℝ), trajectory A B = {P : V | ‖P - center‖ = radius} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_circle_l498_49872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_even_function_extension_l498_49882

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f x = f (2 - x))
  (h_unit_interval : ∀ x, x ∈ Set.Icc 0 1 → f x = x^2) :
  ∀ (k : ℤ) (x : ℝ), x ∈ Set.Icc (2 * (k : ℝ) - 1) (2 * (k : ℝ) + 1) → f x = (x - 2 * (k : ℝ))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_even_function_extension_l498_49882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_plane_max_points_in_space_l498_49893

-- Define a type for points in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Function to check if three points form an obtuse triangle
def isObtuseTriangle (p1 p2 p3 : Point2D) : Prop := sorry

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point2D) : Prop := sorry

-- Function to check if four points in 3D space form an obtuse triangle
def isObtuseTriangle3D (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Function to check if four points in 3D space are coplanar
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Theorem for maximum points in a plane
theorem max_points_in_plane : 
  ∀ (points : List Point2D), 
    (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
      ¬(isObtuseTriangle p1 p2 p3) ∧ ¬(areCollinear p1 p2 p3)) →
    points.length ≤ 4 :=
by sorry

-- Theorem for maximum points in space
theorem max_points_in_space :
  ∀ (points : List Point3D),
    (∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
      ¬(isObtuseTriangle3D p1 p2 p3 p4) ∧ ¬(areCoplanar p1 p2 p3 p4)) →
    points.length ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_in_plane_max_points_in_space_l498_49893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_difference_l498_49875

/-- Represents the vacation cost-sharing scenario -/
structure VacationCosts where
  tom_paid : ℚ
  dorothy_paid : ℚ
  sammy_paid : ℚ
  dorothy_extra_percent : ℚ

/-- Calculates the fair share amounts and transfers -/
noncomputable def calculate_transfers (costs : VacationCosts) : ℚ × ℚ :=
  let total := costs.tom_paid + costs.dorothy_paid + costs.sammy_paid
  let even_share := total / 3
  let dorothy_share := even_share * (1 + costs.dorothy_extra_percent)
  let others_share := (total - dorothy_share) / 2
  let tom_transfer := others_share - costs.tom_paid
  let dorothy_transfer := dorothy_share - costs.dorothy_paid
  (tom_transfer, dorothy_transfer)

/-- The main theorem stating the difference between Tom's and Dorothy's transfers -/
theorem transfer_difference (costs : VacationCosts) 
    (h1 : costs.tom_paid = 150)
    (h2 : costs.dorothy_paid = 180)
    (h3 : costs.sammy_paid = 210)
    (h4 : costs.dorothy_extra_percent = 1/10) :
    let (t, d) := calculate_transfers costs
    t - d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_difference_l498_49875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_equals_87_for_n_ge_3_l498_49801

def a : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 1 => 3^(a n)

def b (n : ℕ) : ℕ := a n % 100

theorem b_equals_87_for_n_ge_3 (n : ℕ) (h : n ≥ 3) : b n = 87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_equals_87_for_n_ge_3_l498_49801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_neg_half_implies_expression_equals_three_l498_49869

theorem tan_alpha_neg_half_implies_expression_equals_three (α : ℝ) :
  Real.tan α = -1/2 → (Real.cos α - Real.sin α)^2 / Real.cos (2*α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_neg_half_implies_expression_equals_three_l498_49869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_equals_reciprocal_l498_49861

open BigOperators

def fraction_product (n : ℕ) : ℚ :=
  ∏ k in Finset.range n, (4 * k : ℚ) / (4 * k + 4)

theorem fraction_product_equals_reciprocal :
  fraction_product 501 = 1 / 502 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_equals_reciprocal_l498_49861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_sum_partition_l498_49842

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def isValidPartition (partition : List (Nat × Nat)) : Prop :=
  partition.length = 5 ∧
  (partition.map (λ p => p.fst + p.snd)).Nodup ∧
  (∀ p ∈ partition, isPrime (p.fst + p.snd)) ∧
  (∀ n ∈ List.range 10, ∃ p ∈ partition, n + 1 = p.fst ∨ n + 1 = p.snd)

theorem exists_prime_sum_partition :
  ∃ partition : List (Nat × Nat), isValidPartition partition := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_prime_sum_partition_l498_49842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l498_49823

noncomputable def sequenceY (m : ℕ) : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (k + 2) => ((m - 2) * sequenceY m (k + 1) - (m - k) * sequenceY m k) / (k + 1)

theorem sequence_sum (m : ℕ) (h : m > 0) : 
  (∑' k, sequenceY m k) = 2^(m - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l498_49823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l498_49834

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) / (x + 1) - a * Real.log x

-- State the theorem
theorem function_properties (a : ℝ) (h_a : 0 < a ∧ a < 1/2) :
  -- Part 1: When a = 1, f is monotonically decreasing on (0, +∞)
  (∀ x y, 0 < x ∧ 0 < y ∧ x < y → f 1 y < f 1 x) ∧
  -- Part 2: f has exactly three zeros
  (∃ x₁ x₂ x₃, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x, 0 < x → f a x = 0 → (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  -- Part 3: For the three zeros x₁ < x₂ < x₃, x₁²(1-x₃) > a(x₁²-1)
  (∀ x₁ x₂ x₃, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 →
    x₁^2 * (1 - x₃) > a * (x₁^2 - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l498_49834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_of_distances_l498_49832

theorem min_value_sum_of_distances :
  ∃ (min_x : ℝ), (λ x : ℝ ↦ Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2)) min_x = 2 * Real.sqrt 5 ∧
  ∀ y : ℝ, (λ x : ℝ ↦ Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2)) y ≥ 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_of_distances_l498_49832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_savings_theorem_l498_49843

/-- Calculates the percentage of water saved per flush by a new toilet compared to an old one. -/
noncomputable def water_savings_percentage (old_gallons_per_flush : ℝ) (flushes_per_day : ℝ) (days : ℝ) (total_savings : ℝ) : ℝ :=
  let old_total_usage := old_gallons_per_flush * flushes_per_day * days
  let new_total_usage := old_total_usage - total_savings
  let new_gallons_per_flush := new_total_usage / (flushes_per_day * days)
  let savings_per_flush := old_gallons_per_flush - new_gallons_per_flush
  (savings_per_flush / old_gallons_per_flush) * 100

/-- The percentage of water saved per flush by the new toilet is 80%. -/
theorem water_savings_theorem :
  water_savings_percentage 5 15 30 1800 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_savings_theorem_l498_49843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_vector_relation_l498_49846

structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

def extended_points (P : Pentagon) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

theorem pentagon_vector_relation (P : Pentagon) :
  let (A', B', C', D', E') := extended_points P
  (1/31 : ℝ) • A' + (2/31 : ℝ) • B' + (4/31 : ℝ) • C' + (8/31 : ℝ) • D' + (16/31 : ℝ) • E' = P.A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_vector_relation_l498_49846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l498_49848

theorem cos_difference (α β : ℝ) 
  (h1 : Real.sin α - Real.sin β = 1 - Real.sqrt 3 / 2) 
  (h2 : Real.cos α - Real.cos β = 1 / 2) : 
  Real.cos (α - β) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l498_49848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y2_in_expansion_l498_49895

/-- The coefficient of x^3y^2 in the expansion of (x^2-x+y)^5 -/
def coefficient_x3y2 : ℤ := -10

/-- The expansion of (x^2-x+y)^5 -/
def expansion (x y : ℝ) : ℝ := (x^2 - x + y)^5

/-- Terms without x^3y^2 in the expansion -/
noncomputable def terms_without_x3y2 (x y : ℝ) : ℝ :=
  expansion x y - coefficient_x3y2 * x^3 * y^2

theorem coefficient_x3y2_in_expansion :
  ∃ (c : ℝ), ∀ (x y : ℝ),
    expansion x y = c * x^3 * y^2 + terms_without_x3y2 x y
    ∧ c = coefficient_x3y2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y2_in_expansion_l498_49895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cashew_price_is_six_l498_49814

/-- The price of cashews per pound that satisfies the merchant's mixture requirements -/
noncomputable def cashew_price : ℝ :=
  let peanut_price : ℝ := 2.40
  let mixture_price : ℝ := 3.00
  let total_weight : ℝ := 60
  let cashew_weight : ℝ := 10
  let peanut_weight : ℝ := total_weight - cashew_weight
  (mixture_price * total_weight - peanut_price * peanut_weight) / cashew_weight

/-- Theorem stating that the cashew price is 6 dollars per pound -/
theorem cashew_price_is_six : cashew_price = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cashew_price_is_six_l498_49814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_l498_49835

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the configuration of circular arcs as described in the problem -/
structure CircularArcs where
  r : ℝ
  C : Point
  D : Point
  A : Point
  B : Point
  arc_AB_is_eighth : ℝ  -- Placeholder for the property that AB is 1/8 of a circle
  arc_AC_is_quarter : ℝ -- Placeholder for the property that AC is 1/4 of a circle

/-- Calculates the area of the shaded triangle T -/
noncomputable def triangle_area (p : CircularArcs) : ℝ := sorry

/-- Calculates the area of the shaded region R -/
noncomputable def region_area (p : CircularArcs) : ℝ := sorry

/-- Theorem stating that the areas of T and R are equal -/
theorem areas_equal (p : CircularArcs) : 
  triangle_area p = region_area p := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_l498_49835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_sum_product_three_digit_coprime_l498_49847

/-- Three-digit natural number -/
def ThreeDigitNat : Type := {n : ℕ // 100 ≤ n ∧ n ≤ 999}

/-- Pairwise coprime property for three numbers -/
def PairwiseCoprime (a b c : ℕ) : Prop :=
  Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime b c

/-- The maximum GCD of the sum and product of three pairwise coprime three-digit numbers -/
theorem max_gcd_sum_product_three_digit_coprime :
  ∃ (x y z : ThreeDigitNat),
    PairwiseCoprime x.val y.val z.val ∧
    ∀ (a b c : ThreeDigitNat),
      PairwiseCoprime a.val b.val c.val →
      Nat.gcd (a.val + b.val + c.val) (a.val * b.val * c.val) ≤
      Nat.gcd (x.val + y.val + z.val) (x.val * y.val * z.val) ∧
      Nat.gcd (x.val + y.val + z.val) (x.val * y.val * z.val) = 2994 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_sum_product_three_digit_coprime_l498_49847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_leg_length_l498_49833

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields and properties
  sideA : ℝ
  sideB : ℝ
  sideC : ℝ

/-- Indicates if a triangle is isosceles and right-angled -/
def Triangle.isIsoscelesRight (T : Triangle) : Prop :=
  T.sideA = T.sideB ∧ T.sideA^2 + T.sideB^2 = T.sideC^2

/-- Returns the length of a leg in an isosceles right triangle -/
def Triangle.legLength (T : Triangle) : ℝ :=
  T.sideA

/-- An isosceles right triangle with a median to the hypotenuse of length 8 units has legs of length 8√2 units. -/
theorem isosceles_right_triangle_leg_length (T : Triangle) (h : T.isIsoscelesRight) 
  (median : ℝ) (h_median : median = 8) : T.legLength = 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_leg_length_l498_49833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawrence_county_kids_count_l498_49821

theorem lawrence_county_kids_count : ℕ := by
  let kids_at_camp : ℕ := 564237
  let kids_at_home : ℕ := 495718
  let total_kids : ℕ := kids_at_camp + kids_at_home
  have : total_kids = 1059955 := by
    -- Proof goes here
    sorry
  exact total_kids

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawrence_county_kids_count_l498_49821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l498_49817

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 3 / 5) : 
  Real.tan (α + π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l498_49817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l498_49803

/-- A point in the 2D Cartesian plane -/
structure MyPoint where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure MyVector where
  x : ℝ
  y : ℝ

/-- Apply a translation vector to a point -/
def translate (p : MyPoint) (v : MyVector) : MyPoint :=
  { x := p.x + v.x, y := p.y + v.y }

theorem translation_problem :
  let A : MyPoint := { x := 1, y := -3 }
  let B : MyPoint := { x := 4, y := -1 }
  let A₁ : MyPoint := { x := -2, y := 0 }
  let v : MyVector := { x := A₁.x - A.x, y := A₁.y - A.y }
  let B₁ : MyPoint := translate B v
  B₁ = { x := 1, y := 2 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l498_49803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_angle_bisector_and_altitude_l498_49887

/-- A triangle with vertices A(4, 6), B(-3, 0), and C(2, -3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 6)
  B : ℝ × ℝ := (-3, 0)
  C : ℝ × ℝ := (2, -3)

/-- The angle bisector of angle A in the triangle -/
def angle_bisector (t : Triangle) : ℝ → ℝ → Prop :=
  λ x y ↦ 5 * x - 3 * y - 2 = 0

/-- The altitude from point C in the triangle -/
def altitude (t : Triangle) : ℝ → ℝ → Prop :=
  λ x y ↦ 7 * x + 6 * y + 4 = 0

/-- The intersection point of the angle bisector and altitude -/
noncomputable def intersection_point : ℝ × ℝ := (0, -2/3)

/-- Theorem: The intersection of the angle bisector and altitude is (0, -2/3) -/
theorem intersection_of_angle_bisector_and_altitude (t : Triangle) :
  angle_bisector t (intersection_point.1) (intersection_point.2) ∧
  altitude t (intersection_point.1) (intersection_point.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_angle_bisector_and_altitude_l498_49887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l498_49857

noncomputable def grasshopper_jumps (n : ℕ) : ℚ :=
  (3 : ℚ) / 4 * (3 : ℚ) ^ n + (1 : ℚ) / 4 * (-1 : ℚ) ^ n

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 2 * f (n + 1) + 3 * f n

theorem grasshopper_theorem (n : ℕ) :
  (f n : ℚ) = grasshopper_jumps n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l498_49857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l498_49849

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Theorem for the four parts of the problem
theorem set_operations :
  (U \ A = {x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2}) ∧
  (A ∩ B = {x | -2 < x ∧ x < 3}) ∧
  (U \ (A ∩ B) = {x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2}) ∧
  ((U \ A) ∩ B = {x | -3 < x ∧ x ≤ -2 ∨ x = 3}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l498_49849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_all_labels_l498_49881

/-- A triangle in a 2D plane. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A partition of a triangle is a set of smaller triangles that cover the original triangle without overlapping. -/
structure TrianglePartition where
  triangles : Set Triangle
  -- Add other necessary properties

/-- A labeling assigns a number (0, 1, or 2) to each vertex in the partition. -/
def Labeling (p : TrianglePartition) := (t : Triangle) → (i : Fin 3) → Fin 3

/-- A triangle in the partition has vertices labeled with 0, 1, and 2. -/
def HasAllLabels (t : Triangle) (l : Labeling p) : Prop :=
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    l t i = 0 ∧ l t j = 1 ∧ l t k = 2

theorem exists_triangle_with_all_labels (p : TrianglePartition) (l : Labeling p) :
  ∃ (t : Triangle), t ∈ p.triangles ∧ HasAllLabels t l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_with_all_labels_l498_49881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_floor_shaded_area_l498_49889

/-- Represents the dimensions of a rectangular floor -/
structure FloorDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a square tile -/
structure TileDimensions where
  side : ℝ

/-- Calculates the shaded area of a floor with given dimensions and tile properties -/
noncomputable def shaded_area (floor : FloorDimensions) (tile : TileDimensions) : ℝ :=
  let total_area := floor.length * floor.width
  let tiles_count := (floor.length / tile.side) * (floor.width / tile.side)
  let shaded_per_tile := tile.side^2 - Real.pi
  tiles_count * shaded_per_tile

/-- Theorem stating the shaded area of the specific floor described in the problem -/
theorem specific_floor_shaded_area :
  let floor := FloorDimensions.mk 12 15
  let tile := TileDimensions.mk 2
  shaded_area floor tile = 180 - 45 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_floor_shaded_area_l498_49889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l498_49815

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := -2 * Real.cos x - x + (x + 1) * Real.log (x + 1)

/-- The function g(x) -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := k * (x^2 + 2/x)

/-- The theorem statement -/
theorem range_of_k (k : ℝ) :
  (k ≠ 0) →
  (∃ x₁ ∈ Set.Icc (-1) 1, ∀ x₂ ∈ Set.Ioo (1/2) 2, f x₁ - g k x₂ < k - 6) →
  k ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l498_49815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_and_none_painted_l498_49825

/-- Represents a unit cube within a larger cube --/
structure UnitCube where
  x : Fin 5
  y : Fin 5
  z : Fin 5
deriving Fintype

/-- Counts the number of painted faces for a unit cube --/
def paintedFaces (c : UnitCube) : Nat :=
  (if c.x = 0 then 1 else 0) +
  (if c.y = 0 then 1 else 0) +
  (if c.z = 0 then 1 else 0)

/-- The set of all unit cubes in the 5x5x5 cube --/
def allCubes : Finset UnitCube :=
  Finset.univ

/-- The number of ways to choose 2 cubes from 125 cubes --/
def totalChoices : Nat :=
  Nat.choose 125 2

/-- The set of unit cubes with exactly three painted faces --/
def threePaintedFaces : Finset UnitCube :=
  Finset.filter (fun c => paintedFaces c = 3) allCubes

/-- The set of unit cubes with no painted faces --/
def noPaintedFaces : Finset UnitCube :=
  Finset.filter (fun c => paintedFaces c = 0) allCubes

/-- The theorem to be proved --/
theorem probability_three_and_none_painted :
  (threePaintedFaces.card * noPaintedFaces.card : ℚ) / totalChoices = 53 / 3875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_and_none_painted_l498_49825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l498_49819

def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {2, 3}

def star (X Y : Finset Nat) : Finset Nat :=
  Finset.image (λ (p : Nat × Nat) => p.1 + p.2) (X.product Y)

theorem star_properties :
  (∃ (m : Nat), m ∈ star A B ∧ ∀ x ∈ star A B, x ≤ m ∧ m = 6) ∧
  (Finset.card (Finset.powerset (star A B)) - 1 = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_l498_49819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_negative_when_a_lt_b_lt_c_expression_positive_when_a_gt_b_gt_c_l498_49865

noncomputable def expression (a b c x : ℝ) : ℝ :=
  (a - b) / (Real.sqrt (abs (c - x))) +
  (b - c) / (Real.sqrt (abs (a - x))) +
  (c - x) / (Real.sqrt (abs (b - x)))

-- Theorem for a < b < c
theorem expression_negative_when_a_lt_b_lt_c
  (a b c x : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : x < a ∨ c < x) :
  expression a b c x < 0 :=
by sorry

-- Theorem for a > b > c
theorem expression_positive_when_a_gt_b_gt_c
  (a b c x : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : x < c ∨ a < x) :
  expression a b c x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_negative_when_a_lt_b_lt_c_expression_positive_when_a_gt_b_gt_c_l498_49865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_theorem_l498_49824

/-- Represents a marathon runner with their completion time -/
structure Runner where
  time : ℝ
  time_positive : time > 0

/-- Calculate the average speed of a runner given the distance -/
noncomputable def averageSpeed (runner : Runner) (distance : ℝ) : ℝ :=
  distance / runner.time

/-- Theorem stating the ratio of average speeds for three runners -/
theorem speed_ratio_theorem (jack jill jamie : Runner)
  (h1 : jack.time = 5)
  (h2 : jill.time = 4.2)
  (h3 : jamie.time = 3.5)
  (distance : ℝ)
  (h4 : distance = 42) :
  ∃ (k : ℝ), k > 0 ∧
    (averageSpeed jack distance) * k = 100 ∧
    (averageSpeed jill distance) * k = 119 ∧
    (averageSpeed jamie distance) * k = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_theorem_l498_49824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_shirt_purchase_cost_l498_49813

theorem tom_shirt_purchase_cost : 
  let normal_price : ℝ := 15
  let discount_percent : ℝ := 20
  let tax_percent : ℝ := 10
  let fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let discounted_price : ℝ := normal_price * (1 - discount_percent / 100)
  let total_shirts : ℕ := fandoms * shirts_per_fandom
  let pre_tax_total : ℝ := (total_shirts : ℝ) * discounted_price
  let tax_amount : ℝ := pre_tax_total * (tax_percent / 100)
  let final_cost : ℝ := pre_tax_total + tax_amount
  final_cost = 264 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_shirt_purchase_cost_l498_49813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_proof_l498_49800

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (1/2 * x - Real.pi/3)

noncomputable def translated_function (x : ℝ) : ℝ := original_function (x + Real.pi/3)

theorem translation_proof :
  ∀ x : ℝ, translated_function x = Real.sin (1/2 * x - Real.pi/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_proof_l498_49800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biathlon_distance_l498_49892

/-- Represents a biathlon race with running and bicycling portions -/
structure Biathlon where
  totalTime : ℝ
  runVelocity : ℝ
  bikeVelocity : ℝ
  runDistance : ℝ

/-- Calculates the total distance of a biathlon -/
noncomputable def totalDistance (b : Biathlon) : ℝ :=
  b.runDistance + b.bikeVelocity * (b.totalTime - b.runDistance / b.runVelocity)

/-- Theorem stating the total distance of a specific biathlon -/
theorem biathlon_distance :
  let b : Biathlon := {
    totalTime := 6,
    runVelocity := 10,
    bikeVelocity := 29,
    runDistance := 10
  }
  totalDistance b = 155 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biathlon_distance_l498_49892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_part_i_value_difference_bound_part_ii_l498_49876

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := a * Real.log x + (1/2) * b * x^2 - (b + a) * x

-- Helper function for the derivative of f
def f' (a b x : ℝ) : ℝ := a / x + b * x - (b + a)

-- Part I
theorem max_value_part_i :
  ∃ (x_max : ℝ), x_max > 0 ∧ ∀ (x : ℝ), x > 0 → f 1 0 x ≤ f 1 0 x_max ∧ f 1 0 x_max = -1 := by
  sorry

-- Part II
theorem value_difference_bound_part_ii (a : ℝ) (h_a : 1 < a ∧ a ≤ Real.exp 1) :
  ∃ (α β : ℝ), α = 1 ∧ β = a ∧
  (∀ (x : ℝ), x > 0 → (f' a 1 x = 0 ↔ x = α ∨ x = β)) ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc α β → x₂ ∈ Set.Icc α β → |f a 1 x₁ - f a 1 x₂| < 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_part_i_value_difference_bound_part_ii_l498_49876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_parallel_lines_from_intersecting_planes_parallel_line_and_plane_parallel_lines_from_three_planes_l498_49898

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Theorem 1
theorem line_perpendicular_to_plane 
  (l m : Line) (α : Plane) 
  (h1 : parallel m l) 
  (h2 : perpendicular m α) : 
  perpendicular l α := by sorry

-- Theorem 2
theorem parallel_lines_from_intersecting_planes 
  (l m n : Line) (α β γ : Plane)
  (h1 : intersect α β l)
  (h2 : intersect β γ m)
  (h3 : intersect γ α n)
  (h4 : line_parallel_plane n β) :
  parallel l m := by sorry

-- Additional theorems to represent other propositions

-- Proposition 2 (false, but included for completeness)
theorem parallel_line_and_plane 
  (l m : Line) (α : Plane)
  (h1 : parallel m l)
  (h2 : line_parallel_plane m α) :
  line_parallel_plane l α := by sorry

-- Proposition 3 (false, but included for completeness)
theorem parallel_lines_from_three_planes 
  (l m n : Line) (α β γ : Plane)
  (h1 : intersect α β l)
  (h2 : intersect β γ m)
  (h3 : intersect γ α n) :
  parallel l m ∧ parallel m n ∧ parallel l n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_plane_parallel_lines_from_intersecting_planes_parallel_line_and_plane_parallel_lines_from_three_planes_l498_49898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_52_l498_49879

/-- A rectangular sheet of paper ABCD with a fold --/
structure FoldedRectangle where
  /-- Length of AG --/
  ag : ℝ
  /-- Length of BG --/
  bg : ℝ
  /-- Length of CH --/
  ch : ℝ
  /-- AG = 5 --/
  ag_eq : ag = 5
  /-- BG = 12 --/
  bg_eq : bg = 12
  /-- CH = 7 --/
  ch_eq : ch = 7

/-- The perimeter of the rectangle ABCD is 52 --/
theorem perimeter_is_52 (rect : FoldedRectangle) : 
  let ab := Real.sqrt (rect.ag ^ 2 + rect.bg ^ 2)
  let ac := Real.sqrt (ab ^ 2 + rect.ch ^ 2)
  2 * (ab + ac) = 52 := by
  sorry

#check perimeter_is_52

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_52_l498_49879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l498_49855

/-- A structure representing a regular triangular pyramid -/
structure RegularTriangularPyramid where
  lateralSurfaceArea : ℝ
  baseArea : ℝ
  inscribedCircleRadius : ℝ
  volume : ℝ

/-- Theorem about the volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (pyramid : RegularTriangularPyramid)
  (h1 : pyramid.lateralSurfaceArea = 3 * pyramid.baseArea)
  (h2 : Real.pi * pyramid.inscribedCircleRadius^2 = pyramid.inscribedCircleRadius) :
  pyramid.volume = 2 * Real.sqrt 6 / Real.pi^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l498_49855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l498_49818

/-- The function representing the sum of distances from (x, y) to four fixed points -/
noncomputable def f (x y : ℝ) : ℝ :=
  ((x + 1)^2 + (y - 1)^2).sqrt +
  ((x + 1)^2 + (y + 1)^2).sqrt +
  ((x - 1)^2 + (y + 1)^2).sqrt +
  ((x - 1)^2 + (y - 1)^2).sqrt

/-- The theorem stating that the minimum value of f(x,y) for x,y ∈ (-1,1) is 4√2 -/
theorem min_value_f :
  ∀ x y : ℝ, -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 → f x y ≥ 4 * Real.sqrt 2 := by
  sorry

#check min_value_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l498_49818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_special_value_l498_49844

noncomputable def f (ω φ : ℝ) (x : ℝ) := Real.sin (ω * x + φ)

theorem sine_special_value (ω φ : ℝ) :
  ω > 0 →
  |φ| < π / 2 →
  (∀ x ∈ Set.Ioo (π / 6) (2 * π / 3), ∀ y ∈ Set.Ioo (π / 6) (2 * π / 3), x < y → f ω φ x > f ω φ y) →
  f ω φ (π / 6) = 1 →
  f ω φ (2 * π / 3) = -1 →
  f ω φ (π / 4) = Real.sqrt 3 / 2 := by
  sorry

#check sine_special_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_special_value_l498_49844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l498_49828

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def octagon_area : ℝ := 18 * Real.sqrt 3 * (2 - Real.sqrt 2)

/-- Theorem: The area of a regular octagon inscribed in a circle with radius 3 units is 18√3(2 - √2) square units -/
theorem regular_octagon_area_in_circle (r : ℝ) (h : r = 3) : 
  octagon_area = 18 * Real.sqrt 3 * (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l498_49828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l498_49878

/-- Represents a complex number -/
structure MyComplex where
  re : ℝ
  im : ℝ

/-- The voltage in an AC circuit -/
def V : MyComplex := { re := 2, im := -2 }

/-- The impedance in an AC circuit -/
def Z : MyComplex := { re := 2, im := 4 }

/-- Calculates the current in an AC circuit -/
noncomputable def current (v z : MyComplex) : MyComplex :=
  { re := (v.re * z.re + v.im * z.im) / (z.re^2 + z.im^2),
    im := (v.im * z.re - v.re * z.im) / (z.re^2 + z.im^2) }

/-- Theorem stating that the current I is equal to 0.6 - 0.6i -/
theorem current_calculation : 
  let I := current V Z
  I.re = 0.6 ∧ I.im = -0.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l498_49878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l498_49885

noncomputable def home : ℝ × ℝ := (0, 0)
noncomputable def milk : ℝ × ℝ := (3, 0)
noncomputable def cookies : ℝ × ℝ := (0, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shortest_path :
  distance home milk + distance milk cookies + distance cookies home = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l498_49885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l498_49802

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The point A on the x-axis -/
def A : ℝ × ℝ := (2, 0)

/-- The first given point -/
def P1 : ℝ × ℝ := (-3, 2)

/-- The second given point -/
def P2 : ℝ × ℝ := (4, -5)

theorem equidistant_point :
  distance A.1 A.2 P1.1 P1.2 = distance A.1 A.2 P2.1 P2.2 :=
by sorry

#eval A
#eval P1
#eval P2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l498_49802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_train_length_l498_49804

/-- Calculates the length of the longer train given the speeds of two trains, 
    the length of the shorter train, and the time they take to clear each other. -/
theorem longer_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (shorter_train_length : ℝ) 
  (clear_time : ℝ) 
  (h1 : speed1 = 42) 
  (h2 : speed2 = 30) 
  (h3 : shorter_train_length = 60) 
  (h4 : clear_time = 16.998640108791296) : 
  ∃ (longer_train_length : ℝ), 
    abs (longer_train_length - 279.9728021758259) < 0.0001 := by
  sorry

#check longer_train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longer_train_length_l498_49804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_configuration_l498_49868

/-- The number of radars -/
def n : ℕ := 9

/-- The radius of each radar's coverage circle in km -/
noncomputable def r : ℝ := 61

/-- The width of the coverage ring in km -/
noncomputable def w : ℝ := 22

/-- The angle between two adjacent radars in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The maximum distance from the center of the platform to the radars -/
noncomputable def max_distance : ℝ := r * Real.cos (θ / 2) / Real.sin (θ / 2)

/-- The area of the coverage ring -/
noncomputable def coverage_area : ℝ := 2 * r * w * Real.pi / Real.tan (θ / 2)

/-- Theorem stating the maximum distance and coverage area for the given configuration -/
theorem radar_configuration :
  (max_distance = 60 / Real.sin (20 * Real.pi / 180)) ∧
  (coverage_area = 2640 * Real.pi / Real.tan (20 * Real.pi / 180)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_configuration_l498_49868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_net_profit_loss_percent_l498_49816

/-- Calculates the net profit or loss percent when two articles are bought for the same price,
    one is sold with a given profit percent, and the other is sold with the same loss percent. -/
noncomputable def netProfitLossPercent (buyPrice : ℝ) (profitLossPercent : ℝ) : ℝ :=
  let sellingPrice1 := buyPrice * (1 + profitLossPercent / 100)
  let sellingPrice2 := buyPrice * (1 - profitLossPercent / 100)
  let totalBuyPrice := 2 * buyPrice
  let totalSellingPrice := sellingPrice1 + sellingPrice2
  (totalSellingPrice - totalBuyPrice) / totalBuyPrice * 100

/-- Theorem stating that when two articles are bought for the same price,
    and one is sold with a 10% profit while the other is sold with a 10% loss,
    the net profit or loss percent is 0%. -/
theorem zero_net_profit_loss_percent (buyPrice : ℝ) (h : buyPrice > 0) :
  netProfitLossPercent buyPrice 10 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_net_profit_loss_percent_l498_49816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magazine_discount_per_issue_l498_49897

/-- Calculates the discount per issue for a magazine subscription promotion -/
theorem magazine_discount_per_issue 
  (normal_cost : ℚ) 
  (subscription_months : ℕ) 
  (issues_per_month : ℕ) 
  (total_discount : ℚ) : 
  normal_cost = 34 →
  subscription_months = 18 →
  issues_per_month = 2 →
  total_discount = 9 →
  (total_discount / (subscription_months * issues_per_month : ℚ)) = 1/4 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magazine_discount_per_issue_l498_49897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l498_49845

/-- Given a geometric sequence {a_n} with common ratio q ≠ 0, 
    S_n represents the sum of the first n terms of the sequence. -/
noncomputable def S (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence {a_n} with a₅ + 2a₁₀ = 0,
    the ratio of S₂₀ to S₁₀ is 5/4. -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (h₁ : q ≠ 0) (h₂ : q ≠ 1) 
  (h₃ : a₁ * q^4 + 2 * a₁ * q^9 = 0) : 
  S a₁ q 20 / S a₁ q 10 = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l498_49845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_is_25_percent_l498_49859

/-- Represents the fluid consumption of a person named Jon --/
structure FluidConsumption where
  smallBottleSize : ℚ
  smallBottlesPerDay : ℚ
  daysPerWeek : ℚ
  largeBottlesPerDay : ℚ
  totalFluidPerWeek : ℚ

/-- Calculates the percentage increase in bottle size --/
def percentageIncrease (fc : FluidConsumption) : ℚ :=
  let smallBottleFluidPerWeek := fc.smallBottleSize * fc.smallBottlesPerDay * fc.daysPerWeek
  let largeBottleFluidPerWeek := fc.totalFluidPerWeek - smallBottleFluidPerWeek
  let largeBottleSize := largeBottleFluidPerWeek / (fc.largeBottlesPerDay * fc.daysPerWeek)
  ((largeBottleSize - fc.smallBottleSize) / fc.smallBottleSize) * 100

/-- Theorem stating that the percentage increase in bottle size is 25% --/
theorem percentage_increase_is_25_percent (fc : FluidConsumption) 
    (h1 : fc.smallBottleSize = 16)
    (h2 : fc.smallBottlesPerDay = 4)
    (h3 : fc.daysPerWeek = 7)
    (h4 : fc.largeBottlesPerDay = 2)
    (h5 : fc.totalFluidPerWeek = 728) :
    percentageIncrease fc = 25 := by
  sorry

#eval percentageIncrease {
  smallBottleSize := 16,
  smallBottlesPerDay := 4,
  daysPerWeek := 7,
  largeBottlesPerDay := 2,
  totalFluidPerWeek := 728
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_increase_is_25_percent_l498_49859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_f_101_l498_49890

def f (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_f_100_f_101 : Int.gcd (f 100) (f 101) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_f_101_l498_49890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_for_inequality_satisfying_cubic_l498_49883

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The value of the polynomial at a given point -/
def CubicPolynomial.eval (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The condition that Q(x^4 + x^2) ≥ Q(x^3 + 1) for all real x -/
def satisfiesInequality (p : CubicPolynomial) : Prop :=
  ∀ x : ℝ, p.eval (x^4 + x^2) ≥ p.eval (x^3 + 1)

/-- The sum of the roots of the cubic polynomial -/
noncomputable def sumOfRoots (p : CubicPolynomial) : ℝ := -p.b / p.a

theorem sum_of_roots_for_inequality_satisfying_cubic 
  (p : CubicPolynomial) (h : satisfiesInequality p) : 
  sumOfRoots p = -p.b / p.a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_for_inequality_satisfying_cubic_l498_49883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l498_49829

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.sqrt (x + 1)

-- Define the function h
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := x^4 + (f a x - Real.sqrt (x + 1)) * (x^2 + 1) + b * x^2 + 1

theorem problem_solution :
  (∀ x ≥ -1, f 1 x ≥ -1) ∧
  (∀ a : ℝ, (∀ x ≥ -1, x + 1 ≥ 0 ∧ x - f a x - 1 ≤ 0) → 1 ≤ a ∧ a ≤ 2) ∧
  (∀ a b : ℝ, (∃ x > 0, h a b x = 0) → a^2 + b^2 ≥ 4/5) ∧
  (∃ a b : ℝ, (∃ x > 0, h a b x = 0) ∧ a^2 + b^2 = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l498_49829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_to_total_ratio_l498_49852

/-- The ratio of white marbles to total marbles is 1:2 -/
theorem white_to_total_ratio (total : ℕ) (yellow : ℕ) (red : ℕ) :
  total = 50 →
  yellow = 12 →
  red = 7 →
  (total - (yellow + yellow / 2 + red) : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_to_total_ratio_l498_49852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_increase_l498_49856

/-- Calculates the percentage increase between two values -/
noncomputable def percentageIncrease (lower upper : ℝ) : ℝ :=
  (upper - lower) / lower * 100

theorem gasoline_price_increase :
  let month3Price : ℝ := 15  -- price in dollars
  let month1PriceEuros : ℝ := 20
  let exchangeRate : ℝ := 1.2  -- 1 euro = 1.2 dollars
  let month1PriceDollars : ℝ := month1PriceEuros * exchangeRate
  percentageIncrease month3Price month1PriceDollars = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_increase_l498_49856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_perpendicular_lines_a_l498_49809

-- Define the slope of a line given its coefficients
noncomputable def my_slope (a b : ℝ) : ℝ := -a / b

-- Define parallel lines
def my_parallel (a1 b1 a2 b2 : ℝ) : Prop := my_slope a1 b1 = my_slope a2 b2

-- Define perpendicular lines
def my_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Problem 1
theorem parallel_lines_m (m : ℝ) : 
  my_parallel 2 (m+1) m 3 → m = 2 ∨ m = -3 :=
by
  sorry

-- Problem 2
theorem perpendicular_lines_a (a : ℝ) : 
  my_perpendicular (a+2) (1-a) (a-1) (2*a+3) → a = 1 ∨ a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_perpendicular_lines_a_l498_49809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l498_49871

-- Define the concept of an increasing function on an interval
def IsIncreasing (f : ℝ → ℝ) (I : Set ℝ) :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x
def f₂ (x : ℝ) : ℝ := (x - 1)^2
noncomputable def f₃ (x : ℝ) : ℝ := -1 / x

-- Define the intervals
def I₁ : Set ℝ := {x | x ≠ 0}
def I₂ : Set ℝ := {x | x > 0}
def I₃ : Set ℝ := {x | x < 0}

-- State the theorem
theorem function_properties :
  (∃ f : ℝ → ℝ, ∃ I : Set ℝ, ∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ ∧ f x₁ < f x₂ ∧ ¬IsIncreasing f I) ∧
  (¬∀ x, x ∈ I₁ → x > 0 → f₁ x < f₁ (x + 1)) ∧
  (¬IsIncreasing f₂ I₂) ∧
  (IsIncreasing f₃ I₃) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l498_49871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_statement_about_perpendicularity_and_parallelism_l498_49826

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relationships between lines and planes
def perpendicular_lines : Line → Line → Prop := sorry
def perpendicular_line_plane : Line → Plane → Prop := sorry
def parallel_line_plane : Line → Plane → Prop := sorry
def perpendicular_planes : Plane → Plane → Prop := sorry
def line_in_plane : Line → Plane → Prop := sorry

-- Theorem statement
theorem false_statement_about_perpendicularity_and_parallelism
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β) :
  ¬(parallel_line_plane m α → perpendicular_planes α β → perpendicular_line_plane m β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_statement_about_perpendicularity_and_parallelism_l498_49826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l498_49805

def geometric_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ m n, a (m + n) = a m * a n

theorem sequence_property (a : ℕ → ℕ) (k : ℕ) 
  (h_seq : geometric_sequence a) (h_ak : a (k + 1) = 1024) : k = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l498_49805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l498_49896

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / |x - 1|
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the property of no intersection
def no_intersection (k : ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ g k x

-- State the theorem
theorem intersection_range :
  ∀ k : ℝ, no_intersection k ↔ k ∈ Set.Icc (-4 : ℝ) (-1) ∧ k ≠ -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l498_49896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l498_49866

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (3 * x^2 - 6 * x + 9)

-- Define the antiderivative function
noncomputable def F (x : ℝ) : ℝ := (1 / Real.sqrt 3) * Real.log (abs (x - 1 + Real.sqrt (x^2 - 2*x + 3)))

-- Theorem statement
theorem integral_equality (x : ℝ) : deriv F x = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l498_49866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l498_49867

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 6)

-- State the theorem
theorem f_range :
  ∀ x ∈ Set.Icc 0 (π / 2),
  ∃ y ∈ Set.Icc (-3/2) 3,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-3/2) 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l498_49867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_max_and_min_l498_49864

/-- The sequence a_n = (n+1) * (-10/11)^n has both a maximum and a minimum value -/
theorem sequence_has_max_and_min :
  ∃ (max min : ℝ) (i j : ℕ), 
    (∀ n : ℕ, ((n : ℝ) + 1) * (-10/11)^n ≤ max) ∧
    (∀ n : ℕ, min ≤ ((n : ℝ) + 1) * (-10/11)^n) ∧
    max = ((i : ℝ) + 1) * (-10/11)^i ∧
    min = ((j : ℝ) + 1) * (-10/11)^j :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_has_max_and_min_l498_49864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_formula_l498_49851

def sequence_sum (n : ℕ) : ℚ :=
  sorry  -- We'll define this later

axiom a_1 : sequence_sum 1 = -2/3

axiom sum_relation : ∀ n : ℕ, n ≥ 2 → 
  sequence_sum n + (sequence_sum n)⁻¹ + 2 = sequence_sum n - sequence_sum (n-1)

theorem sum_formula : ∀ n : ℕ, n > 0 → 
  sequence_sum n = -(n + 1 : ℚ) / (n + 2 : ℚ) := by
  sorry  -- Proof to be added later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_formula_l498_49851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_in_cone_l498_49854

/-- Represents a conical container -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

theorem volume_ratio_in_cone (c : Cone) :
  let upperHalf := Cone.mk (c.radius/2) (c.height/2)
  let lowerHalf := Cone.mk (c.radius/2) (c.height/2)
  coneVolume c - coneVolume lowerHalf = 7 * coneVolume lowerHalf := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_in_cone_l498_49854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_balloon_purchase_l498_49812

/-- Represents the store's balloon pricing strategy -/
structure BalloonStore where
  fullPrice : ℚ
  groupSize : ℕ
  discountFactor : ℚ

/-- Calculates the cost of a group of balloons -/
def groupCost (store : BalloonStore) : ℚ :=
  store.fullPrice * (store.groupSize - 1 + store.discountFactor)

/-- Calculates the maximum number of balloons that can be bought with a given budget -/
def maxBalloons (store : BalloonStore) (budget : ℚ) : ℕ :=
  let fullGroups := (budget / groupCost store).floor
  let remainingMoney := budget - fullGroups * groupCost store
  let extraBalloons := (remainingMoney / store.fullPrice).floor
  (fullGroups * store.groupSize + extraBalloons).toNat

/-- The theorem to be proved -/
theorem orvin_balloon_purchase (orvinBudget : ℚ) :
  orvinBudget = 40 →
  maxBalloons { fullPrice := 1, groupSize := 5, discountFactor := 1/2 } orvinBudget = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_balloon_purchase_l498_49812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_ratio_l498_49836

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Define the midpoint of two points
def midpoint_of (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_line_intersection_ratio
  (x₁ y₁ x₂ y₂ x₀ y₀ m : ℝ)
  (h1 : hyperbola_C x₁ y₁)
  (h2 : hyperbola_C x₂ y₂)
  (h3 : line x₁ y₁ m)
  (h4 : line x₂ y₂ m)
  (h5 : midpoint_of x₁ y₁ x₂ y₂ x₀ y₀)
  (h6 : x₀ ≠ 0) :
  y₀ / x₀ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_ratio_l498_49836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_l498_49850

/-- Represents the time it takes for both leaks to empty a full cistern -/
noncomputable def empty_time (fill_time_no_leak : ℝ) (fill_time_with_leaks : ℝ) (leak1_empty_time : ℝ) (leak2_empty_time : ℝ) : ℝ :=
  1 / (1 / fill_time_no_leak - 1 / fill_time_with_leaks)

/-- Theorem stating that under given conditions, the time to empty the cistern is 24 hours -/
theorem cistern_empty_time :
  let fill_time_no_leak : ℝ := 8
  let fill_time_with_leaks : ℝ := 12
  let leak1_empty_time : ℝ := x
  let leak2_empty_time : ℝ := y
  empty_time fill_time_no_leak fill_time_with_leaks leak1_empty_time leak2_empty_time = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_l498_49850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_squares_l498_49853

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)
def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define points M₁ and M₂
def M₁ : ℝ × ℝ := (0, 1)
def M₂ : ℝ × ℝ := (2, 0)

-- Define the line M₁M₂
def line_M₁M₂ (x y : ℝ) : Prop := x + 2*y = 2

-- Define the intersection points P and Q
axiom P_on_C₂ : ∃ θ : ℝ, C₂ θ = Real.sqrt ((M₁.1 - M₂.1)^2 + (M₁.2 - M₂.2)^2) / 2
axiom Q_on_C₂ : ∃ θ : ℝ, C₂ θ = Real.sqrt ((M₁.1 - M₂.1)^2 + (M₁.2 - M₂.2)^2) / 2

-- Define points A and B
axiom A_on_C₁ : ∃ θ : ℝ, C₁ θ = (Real.cos θ * C₂ θ, Real.sin θ * C₂ θ)
axiom B_on_C₁ : ∃ θ : ℝ, C₁ θ = (-Real.sin θ * C₂ θ, Real.cos θ * C₂ θ)

-- State the theorem
theorem sum_reciprocal_squares :
  ∃ (θA θB : ℝ),
    let (xA, yA) := C₁ θA
    let (xB, yB) := C₁ θB
    1 / (xA^2 + yA^2) + 1 / (xB^2 + yB^2) = 5/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_squares_l498_49853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angular_velocity_at_ground_l498_49877

/-- A thin, homogeneous rod falling from vertical position --/
structure FallingRod where
  l : ℝ  -- length of the rod
  g : ℝ  -- acceleration due to gravity
  α : ℝ  -- angle between the rod and the horizontal plane

/-- The angular velocity of the falling rod as a function of α --/
noncomputable def angular_velocity (rod : FallingRod) : ℝ :=
  Real.sqrt ((6 * rod.g / rod.l) * ((1 - Real.sin rod.α) / (1 + 3 * (Real.cos rod.α)^2)))

/-- Theorem: The angular velocity when the rod hits the ground is √(3g/l) --/
theorem angular_velocity_at_ground (rod : FallingRod) :
  angular_velocity { l := rod.l, g := rod.g, α := 0 } = Real.sqrt (3 * rod.g / rod.l) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angular_velocity_at_ground_l498_49877
