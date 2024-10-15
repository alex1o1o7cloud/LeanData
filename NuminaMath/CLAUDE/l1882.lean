import Mathlib

namespace NUMINAMATH_CALUDE_jerome_toy_cars_l1882_188265

/-- The number of toy cars Jerome has now -/
def total_cars (original : ℕ) (last_month : ℕ) (this_month : ℕ) : ℕ :=
  original + last_month + this_month

/-- Theorem: Jerome has 40 toy cars now -/
theorem jerome_toy_cars :
  let original := 25
  let last_month := 5
  let this_month := 2 * last_month
  total_cars original last_month this_month = 40 := by
sorry

end NUMINAMATH_CALUDE_jerome_toy_cars_l1882_188265


namespace NUMINAMATH_CALUDE_book_order_theorem_l1882_188287

/-- Represents the book "Journey to the West" -/
inductive JourneyToWest

/-- Represents the book "Morning Blossoms and Evening Blossoms" -/
inductive MorningBlossoms

/-- Represents the initial order of books -/
structure InitialOrder where
  mb_cost : ℕ
  jw_cost : ℕ
  mb_price_ratio : ℚ
  mb_quantity_diff : ℕ

/-- Represents the additional order constraints -/
structure AdditionalOrderConstraints where
  total_books : ℕ
  min_mb : ℕ
  max_cost : ℕ

/-- Calculates the unit prices based on the initial order -/
def calculate_unit_prices (order : InitialOrder) : ℚ × ℚ :=
  sorry

/-- Calculates the number of possible ordering schemes and the lowest total cost -/
def calculate_additional_order (constraints : AdditionalOrderConstraints) (mb_price jw_price : ℚ) : ℕ × ℕ :=
  sorry

theorem book_order_theorem (initial_order : InitialOrder) (constraints : AdditionalOrderConstraints) :
  initial_order.mb_cost = 14000 ∧
  initial_order.jw_cost = 7000 ∧
  initial_order.mb_price_ratio = 1.4 ∧
  initial_order.mb_quantity_diff = 300 ∧
  constraints.total_books = 10 ∧
  constraints.min_mb = 3 ∧
  constraints.max_cost = 124 →
  let (jw_price, mb_price) := calculate_unit_prices initial_order
  let (schemes, lowest_cost) := calculate_additional_order constraints mb_price jw_price
  jw_price = 10 ∧ mb_price = 14 ∧ schemes = 4 ∧ lowest_cost = 112 :=
sorry

end NUMINAMATH_CALUDE_book_order_theorem_l1882_188287


namespace NUMINAMATH_CALUDE_female_muscovy_percentage_problem_l1882_188256

/-- The percentage of female Muscovy ducks -/
def female_muscovy_percentage (total_ducks : ℕ) (muscovy_percentage : ℚ) (female_muscovy : ℕ) : ℚ :=
  (female_muscovy : ℚ) / (muscovy_percentage * total_ducks) * 100

theorem female_muscovy_percentage_problem :
  female_muscovy_percentage 40 (1/2) 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_female_muscovy_percentage_problem_l1882_188256


namespace NUMINAMATH_CALUDE_unique_pair_exists_l1882_188217

theorem unique_pair_exists (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_exists_l1882_188217


namespace NUMINAMATH_CALUDE_coin_array_problem_l1882_188220

/-- The number of coins in a triangular array -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem stating the problem -/
theorem coin_array_problem :
  ∃ (n : ℕ), triangular_sum n = 2211 ∧ sum_of_digits n = 12 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l1882_188220


namespace NUMINAMATH_CALUDE_extended_triangle_similarity_l1882_188223

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the extension of a line segment
def extend (p1 p2 : ℝ × ℝ) (length : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem extended_triangle_similarity (ABC : Triangle) (P : ℝ × ℝ) :
  distance ABC.A ABC.B = 8 →
  distance ABC.B ABC.C = 7 →
  distance ABC.C ABC.A = 6 →
  P = extend ABC.B ABC.C (distance ABC.B P - 7) →
  similar (Triangle.mk P ABC.A ABC.B) (Triangle.mk P ABC.C ABC.A) →
  distance P ABC.C = 9 := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_similarity_l1882_188223


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1882_188266

theorem inheritance_calculation (inheritance : ℝ) : 
  inheritance * 0.25 + (inheritance - inheritance * 0.25) * 0.15 = 15000 →
  inheritance = 41379 := by
sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1882_188266


namespace NUMINAMATH_CALUDE_michael_work_days_l1882_188243

-- Define the work rates for Michael, Adam, and Lisa
def M : ℚ := 1 / 40
def A : ℚ := 1 / 60
def L : ℚ := 1 / 60

-- Define the total work as 1 unit
def total_work : ℚ := 1

-- Theorem stating the conditions and the result to be proved
theorem michael_work_days :
  -- Condition 1: Michael, Adam, and Lisa can do the work together in 15 days
  M + A + L = 1 / 15 →
  -- Condition 2: After 10 days of working together, 2/3 of the work is completed
  (M + A + L) * 10 = 2 / 3 →
  -- Condition 3: Adam and Lisa complete the remaining 1/3 of the work in 8 days
  (A + L) * 8 = 1 / 3 →
  -- Conclusion: Michael takes 40 days to complete the work separately
  total_work / M = 40 :=
by sorry


end NUMINAMATH_CALUDE_michael_work_days_l1882_188243


namespace NUMINAMATH_CALUDE_average_of_combined_results_l1882_188210

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 45) (h₂ : n₂ = 25) (h₃ : avg₁ = 25) (h₄ : avg₂ = 45) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = 2250 / 70 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l1882_188210


namespace NUMINAMATH_CALUDE_cake_distribution_l1882_188253

theorem cake_distribution (total_pieces : ℕ) (num_friends : ℕ) (pieces_per_friend : ℕ) :
  total_pieces = 150 →
  num_friends = 50 →
  pieces_per_friend * num_friends = total_pieces →
  pieces_per_friend = 3 := by
sorry

end NUMINAMATH_CALUDE_cake_distribution_l1882_188253


namespace NUMINAMATH_CALUDE_cyclic_equation_system_solution_l1882_188241

theorem cyclic_equation_system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : (x₃ + x₄ + x₅)^5 = 3*x₁)
  (h₂ : (x₄ + x₅ + x₁)^5 = 3*x₂)
  (h₃ : (x₅ + x₁ + x₂)^5 = 3*x₃)
  (h₄ : (x₁ + x₂ + x₃)^5 = 3*x₄)
  (h₅ : (x₂ + x₃ + x₄)^5 = 3*x₅)
  (pos₁ : x₁ > 0) (pos₂ : x₂ > 0) (pos₃ : x₃ > 0) (pos₄ : x₄ > 0) (pos₅ : x₅ > 0) :
  x₁ = 1/3 ∧ x₂ = 1/3 ∧ x₃ = 1/3 ∧ x₄ = 1/3 ∧ x₅ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_equation_system_solution_l1882_188241


namespace NUMINAMATH_CALUDE_equation_transformation_l1882_188280

theorem equation_transformation (x y : ℝ) : x = y → x - 2 = y - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1882_188280


namespace NUMINAMATH_CALUDE_even_product_probability_l1882_188262

def ten_sided_die := Finset.range 10

theorem even_product_probability :
  let outcomes := ten_sided_die.product ten_sided_die
  (outcomes.filter (fun (x, y) => (x + 1) * (y + 1) % 2 = 0)).card / outcomes.card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_even_product_probability_l1882_188262


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_equation_l1882_188291

theorem smallest_n_for_candy_equation : ∃ (n : ℕ), n > 0 ∧
  (∀ (r g b y : ℕ), r > 0 ∧ g > 0 ∧ b > 0 ∧ y > 0 →
    (10 * r = 8 * g ∧ 8 * g = 9 * b ∧ 9 * b = 12 * y ∧ 12 * y = 18 * n) →
    (∀ (m : ℕ), m > 0 ∧ m < n →
      ¬(∃ (r' g' b' y' : ℕ), r' > 0 ∧ g' > 0 ∧ b' > 0 ∧ y' > 0 ∧
        10 * r' = 8 * g' ∧ 8 * g' = 9 * b' ∧ 9 * b' = 12 * y' ∧ 12 * y' = 18 * m))) ∧
  n = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_equation_l1882_188291


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1882_188294

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Icc 0 (Real.pi / 2)) 
  (h2 : Real.sin (α - Real.pi / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1882_188294


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l1882_188216

/-- Given a line passing through points (3, -8) and (k, 20) that is parallel to the line 3x + 4y = 12, 
    the value of k is -103/3. -/
theorem parallel_line_k_value :
  ∀ (k : ℚ),
  (∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + b → (x = 3 ∧ y = -8) ∨ (x = k ∧ y = 20)) ∧
                (m = -3/4)) →
  k = -103/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l1882_188216


namespace NUMINAMATH_CALUDE_f_19_equals_zero_l1882_188276

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period_two_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

-- Theorem statement
theorem f_19_equals_zero 
  (h1 : is_even f) 
  (h2 : has_period_two_negation f) : 
  f 19 = 0 := by sorry

end NUMINAMATH_CALUDE_f_19_equals_zero_l1882_188276


namespace NUMINAMATH_CALUDE_possible_values_of_d_over_a_l1882_188272

theorem possible_values_of_d_over_a (a d : ℝ) (h1 : a^2 - 6*a*d + 8*d^2 = 0) (h2 : a ≠ 0) :
  d/a = 1/2 ∨ d/a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_d_over_a_l1882_188272


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1882_188260

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l1882_188260


namespace NUMINAMATH_CALUDE_smallest_angle_SQR_l1882_188200

-- Define the angles
def angle_PQR : ℝ := 40
def angle_PQS : ℝ := 28

-- Define the theorem
theorem smallest_angle_SQR : 
  let angle_SQR := angle_PQR - angle_PQS
  angle_SQR = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_SQR_l1882_188200


namespace NUMINAMATH_CALUDE_average_marks_of_combined_classes_l1882_188219

theorem average_marks_of_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) :
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 40 →
  avg₂ = 60 →
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℝ) = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_of_combined_classes_l1882_188219


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1882_188268

def repeating_decimal (a b c : ℕ) : ℚ := (a * 100 + b * 10 + c : ℚ) / 999

theorem repeating_decimal_sum :
  repeating_decimal 2 3 4 - repeating_decimal 5 6 7 + repeating_decimal 8 9 1 = 186 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1882_188268


namespace NUMINAMATH_CALUDE_product_expansion_l1882_188213

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1882_188213


namespace NUMINAMATH_CALUDE_tessa_apples_left_l1882_188273

/-- The number of apples Tessa has left after making a pie -/
def apples_left (initial : ℝ) (gift : ℝ) (pie_requirement : ℝ) : ℝ :=
  initial + gift - pie_requirement

/-- Theorem: Given the initial conditions, Tessa will have 11.25 apples left -/
theorem tessa_apples_left :
  apples_left 10.0 5.5 4.25 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_tessa_apples_left_l1882_188273


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1882_188249

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The range of a function -/
def Range (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y, y ∈ S ↔ ∃ x, f x = y

/-- The theorem statement -/
theorem quadratic_function_range
  (f g : ℝ → ℝ)
  (h1 : f = g)
  (h2 : QuadraticFunction f)
  (h3 : Range (f ∘ g) (Set.Ici 0)) :
  Range (fun x ↦ g x) (Set.Ici 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1882_188249


namespace NUMINAMATH_CALUDE_new_average_and_variance_l1882_188263

/-- Given three numbers with average 5 and variance 2, prove that adding 1 results in four numbers with average 4 and variance 4.5 -/
theorem new_average_and_variance 
  (x y z : ℝ) 
  (h_avg : (x + y + z) / 3 = 5)
  (h_var : ((x - 5)^2 + (y - 5)^2 + (z - 5)^2) / 3 = 2) :
  let new_numbers := [x, y, z, 1]
  ((x + y + z + 1) / 4 = 4) ∧ 
  (((x - 4)^2 + (y - 4)^2 + (z - 4)^2 + (1 - 4)^2) / 4 = 4.5) := by
sorry


end NUMINAMATH_CALUDE_new_average_and_variance_l1882_188263


namespace NUMINAMATH_CALUDE_square_state_after_2010_transforms_l1882_188269

/-- Represents the four possible states of the square labeling -/
inductive SquareState
  | BADC
  | DCBA
  | ABCD
  | CDAB

/-- Applies one transformation (reflection then rotation) to the square -/
def transform (s : SquareState) : SquareState :=
  match s with
  | SquareState.BADC => SquareState.DCBA
  | SquareState.DCBA => SquareState.ABCD
  | SquareState.ABCD => SquareState.DCBA
  | SquareState.CDAB => SquareState.BADC

/-- Applies n transformations to the initial square state -/
def applyNTransforms (n : Nat) : SquareState :=
  match n with
  | 0 => SquareState.BADC
  | n + 1 => transform (applyNTransforms n)

theorem square_state_after_2010_transforms :
  applyNTransforms 2010 = SquareState.DCBA := by
  sorry

end NUMINAMATH_CALUDE_square_state_after_2010_transforms_l1882_188269


namespace NUMINAMATH_CALUDE_x_value_l1882_188224

theorem x_value : ∃ x : ℝ, (3 * x = (16 - x) + 4) ∧ (x = 5) := by sorry

end NUMINAMATH_CALUDE_x_value_l1882_188224


namespace NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l1882_188240

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorterLeg : ℝ
  longerLeg : ℝ
  hypotenuse_eq : hypotenuse = 2 * shorterLeg
  longerLeg_eq : longerLeg = shorterLeg * Real.sqrt 3

/-- Represents a sequence of four 30-60-90 triangles -/
def TriangleSequence (t1 t2 t3 t4 : Triangle30_60_90) : Prop :=
  t1.hypotenuse = 16 ∧
  t2.hypotenuse = t1.longerLeg ∧
  t3.hypotenuse = t2.longerLeg ∧
  t4.hypotenuse = t3.longerLeg ∧
  t2.hypotenuse = t1.hypotenuse / 2 ∧
  t3.hypotenuse = t2.hypotenuse / 2 ∧
  t4.hypotenuse = t3.hypotenuse / 2

theorem smallest_triangle_longer_leg
  (t1 t2 t3 t4 : Triangle30_60_90)
  (h : TriangleSequence t1 t2 t3 t4) :
  t4.longerLeg = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_longer_leg_l1882_188240


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1882_188215

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) 
  (h₁ : 6 * x₁^2 - 13 * x₁ + 5 = 0)
  (h₂ : 6 * x₂^2 - 13 * x₂ + 5 = 0)
  (h₃ : x₁ ≠ x₂) : 
  x₁^2 + x₂^2 = 109 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1882_188215


namespace NUMINAMATH_CALUDE_sum_of_base8_digits_878_l1882_188231

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits in the base 8 representation of 878 is 17 -/
theorem sum_of_base8_digits_878 :
  sumDigits (toBase8 878) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base8_digits_878_l1882_188231


namespace NUMINAMATH_CALUDE_banana_cost_is_three_l1882_188248

/-- The cost of a single fruit item -/
structure FruitCost where
  apple : ℕ
  orange : ℕ
  banana : ℕ

/-- The quantity of fruits bought -/
structure FruitQuantity where
  apple : ℕ
  orange : ℕ
  banana : ℕ

/-- Calculate the discount based on the total number of fruits -/
def calculateDiscount (totalFruits : ℕ) : ℕ :=
  totalFruits / 5

/-- Calculate the total cost of fruits before discount -/
def calculateTotalCost (cost : FruitCost) (quantity : FruitQuantity) : ℕ :=
  cost.apple * quantity.apple + cost.orange * quantity.orange + cost.banana * quantity.banana

/-- The main theorem to prove -/
theorem banana_cost_is_three
  (cost : FruitCost)
  (quantity : FruitQuantity)
  (h1 : cost.apple = 1)
  (h2 : cost.orange = 2)
  (h3 : quantity.apple = 5)
  (h4 : quantity.orange = 3)
  (h5 : quantity.banana = 2)
  (h6 : calculateTotalCost cost quantity - calculateDiscount (quantity.apple + quantity.orange + quantity.banana) = 15) :
  cost.banana = 3 := by
  sorry

#check banana_cost_is_three

end NUMINAMATH_CALUDE_banana_cost_is_three_l1882_188248


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1882_188218

def team_size : ℕ := 12
def offensive_linemen : ℕ := 5

theorem starting_lineup_combinations : 
  (offensive_linemen) *
  (team_size - 1) *
  (team_size - 2) *
  ((team_size - 3) * (team_size - 4) / 2) = 19800 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1882_188218


namespace NUMINAMATH_CALUDE_inequality_properties_l1882_188247

theorem inequality_properties (a b c : ℝ) :
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c : ℝ), a > b → a * (2^c) > b * (2^c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l1882_188247


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1882_188211

theorem equilateral_triangle_area (h : ℝ) (A : ℝ) : 
  h = 3 * Real.sqrt 3 → A = (Real.sqrt 3 / 4) * (2 * h / Real.sqrt 3)^2 → A = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1882_188211


namespace NUMINAMATH_CALUDE_percentage_women_red_hair_men_dark_hair_l1882_188246

theorem percentage_women_red_hair_men_dark_hair (
  women_fair_hair : Real) (women_dark_hair : Real) (women_red_hair : Real)
  (men_fair_hair : Real) (men_dark_hair : Real) (men_red_hair : Real)
  (h1 : women_fair_hair = 30)
  (h2 : women_dark_hair = 28)
  (h3 : women_red_hair = 12)
  (h4 : men_fair_hair = 20)
  (h5 : men_dark_hair = 35)
  (h6 : men_red_hair = 5)
  : women_red_hair + men_dark_hair = 47 := by
  sorry

end NUMINAMATH_CALUDE_percentage_women_red_hair_men_dark_hair_l1882_188246


namespace NUMINAMATH_CALUDE_correct_multiplication_l1882_188222

theorem correct_multiplication (x : ℕ) : 
  987 * x = 559981 → 987 * x = 559989 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l1882_188222


namespace NUMINAMATH_CALUDE_locus_of_Q_l1882_188221

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/24 + y^2/16 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x/12 + y/8 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the intersection point R of OP and ellipse C
def point_R (x y : ℝ) : Prop := ellipse_C x y ∧ ∃ t : ℝ, x = t * (x - 0) ∧ y = t * (y - 0)

-- Define point Q on OP satisfying |OQ| * |OP| = |OR|^2
def point_Q (x y : ℝ) (xp yp xr yr : ℝ) : Prop :=
  ∃ t : ℝ, x = t * xp ∧ y = t * yp ∧ 
  (x^2 + y^2) * (xp^2 + yp^2) = (xr^2 + yr^2)^2

-- Theorem statement
theorem locus_of_Q (x y : ℝ) : 
  (∃ xp yp xr yr : ℝ, 
    point_P xp yp ∧ 
    point_R xr yr ∧ 
    point_Q x y xp yp xr yr) → 
  (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_Q_l1882_188221


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l1882_188204

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  /-- The altitude to the base -/
  altitude : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The triangle is isosceles -/
  is_isosceles : True

/-- Calculate the area of an isosceles triangle -/
def triangle_area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area_theorem :
  ∀ (t : IsoscelesTriangle), t.altitude = 10 ∧ t.perimeter = 40 → triangle_area t = 75 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_theorem_l1882_188204


namespace NUMINAMATH_CALUDE_mosquito_shadow_speed_l1882_188284

/-- The speed of a mosquito's shadow on the bottom of a water body. -/
def shadow_speed (v : ℝ) (cos_beta : ℝ) : Set ℝ :=
  {0, 2 * v * cos_beta}

/-- Theorem: Given the conditions of the mosquito problem, the speed of the shadow is either 0 m/s or 0.8 m/s. -/
theorem mosquito_shadow_speed 
  (v : ℝ) 
  (t : ℝ) 
  (h : ℝ) 
  (cos_theta : ℝ) 
  (cos_beta : ℝ) 
  (hv : v = 0.5)
  (ht : t = 20)
  (hh : h = 6)
  (hcos_theta : cos_theta = 0.6)
  (hcos_beta : cos_beta = 0.8)
  : shadow_speed v cos_beta = {0, 0.8} := by
  sorry

#check mosquito_shadow_speed

end NUMINAMATH_CALUDE_mosquito_shadow_speed_l1882_188284


namespace NUMINAMATH_CALUDE_jed_cards_40_after_4_weeks_l1882_188298

/-- The number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + 6 * weeks - 2 * (weeks / 2)

/-- The theorem stating that Jed will have 40 cards after 4 weeks -/
theorem jed_cards_40_after_4_weeks :
  cards_after_weeks 20 4 = 40 := by sorry

end NUMINAMATH_CALUDE_jed_cards_40_after_4_weeks_l1882_188298


namespace NUMINAMATH_CALUDE_power_multiplication_l1882_188233

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1882_188233


namespace NUMINAMATH_CALUDE_smallest_multiple_exceeding_100_l1882_188293

theorem smallest_multiple_exceeding_100 : ∃ (n : ℕ), 
  n > 0 ∧ 
  n % 45 = 0 ∧ 
  (n - 100) % 7 = 0 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 45 = 0 ∧ (m - 100) % 7 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_exceeding_100_l1882_188293


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1882_188267

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 24*x + 125

-- Define the lower and upper bounds of the solution interval
def lower_bound : ℝ := 6.71
def upper_bound : ℝ := 17.29

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x ≤ 9 ↔ lower_bound ≤ x ∧ x ≤ upper_bound := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1882_188267


namespace NUMINAMATH_CALUDE_parabola_intersection_l1882_188238

theorem parabola_intersection :
  let f (x : ℝ) := 2 * x^2 + 5 * x - 3
  let g (x : ℝ) := x^2 + 8
  let x₁ := (-5 - Real.sqrt 69) / 2
  let x₂ := (-5 + Real.sqrt 69) / 2
  let y₁ := f x₁
  let y₂ := f x₂
  (∀ x, f x = g x ↔ x = x₁ ∨ x = x₂) ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1882_188238


namespace NUMINAMATH_CALUDE_max_value_cosine_sine_fraction_l1882_188230

theorem max_value_cosine_sine_fraction :
  ∀ x : ℝ, (1 + Real.cos x) / (Real.sin x + Real.cos x + 2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cosine_sine_fraction_l1882_188230


namespace NUMINAMATH_CALUDE_salary_proof_l1882_188277

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := 6500

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.2

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.2

/-- Represents the new savings amount in Rupees after expense increase -/
def new_savings : ℝ := 260

theorem salary_proof :
  let original_expenses := monthly_salary * (1 - savings_rate)
  let new_expenses := original_expenses * (1 + expense_increase_rate)
  monthly_salary - new_expenses = new_savings := by
  sorry

#check salary_proof

end NUMINAMATH_CALUDE_salary_proof_l1882_188277


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1882_188283

theorem root_equation_implies_expression_value (m : ℝ) : 
  2 * m^2 + 3 * m - 1 = 0 → 4 * m^2 + 6 * m - 2019 = -2017 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1882_188283


namespace NUMINAMATH_CALUDE_total_bottles_is_255_l1882_188206

-- Define the number of bottles in each box
def boxA_water : ℕ := 24
def boxA_orange : ℕ := 21
def boxA_apple : ℕ := boxA_water + 6

def boxB_water : ℕ := boxA_water + boxA_water / 4
def boxB_orange : ℕ := boxA_orange - boxA_orange * 3 / 10
def boxB_apple : ℕ := boxA_apple

def boxC_water : ℕ := 2 * boxB_water
def boxC_apple : ℕ := (3 * boxB_apple) / 2
def boxC_orange : ℕ := 0

-- Define the total number of bottles
def total_bottles : ℕ := 
  boxA_water + boxA_orange + boxA_apple + 
  boxB_water + boxB_orange + boxB_apple + 
  boxC_water + boxC_orange + boxC_apple

-- Theorem to prove
theorem total_bottles_is_255 : total_bottles = 255 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_is_255_l1882_188206


namespace NUMINAMATH_CALUDE_james_total_spent_l1882_188288

def entry_fee : ℕ := 20
def num_rounds : ℕ := 2
def num_friends : ℕ := 5
def james_drinks : ℕ := 6
def drink_cost : ℕ := 6
def food_cost : ℕ := 14
def tip_percentage : ℚ := 30 / 100

def total_spent : ℕ := 163

theorem james_total_spent : 
  entry_fee + 
  (num_rounds * num_friends * drink_cost) + 
  (james_drinks * drink_cost) + 
  food_cost + 
  (((num_rounds * num_friends * drink_cost) + (james_drinks * drink_cost) + food_cost : ℚ) * tip_percentage).floor = 
  total_spent := by
  sorry

end NUMINAMATH_CALUDE_james_total_spent_l1882_188288


namespace NUMINAMATH_CALUDE_katie_packages_l1882_188212

def cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

theorem katie_packages :
  cupcake_packages 18 8 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_katie_packages_l1882_188212


namespace NUMINAMATH_CALUDE_technicians_count_l1882_188278

/-- Represents the workshop scenario with given salary information --/
structure Workshop where
  totalWorkers : ℕ
  avgSalaryAll : ℚ
  avgSalaryTech : ℚ
  avgSalaryNonTech : ℚ

/-- Calculates the number of technicians in the workshop --/
def numTechnicians (w : Workshop) : ℚ :=
  (w.avgSalaryAll * w.totalWorkers - w.avgSalaryNonTech * w.totalWorkers) /
  (w.avgSalaryTech - w.avgSalaryNonTech)

/-- Theorem stating that the number of technicians is 7 --/
theorem technicians_count (w : Workshop) 
  (h1 : w.totalWorkers = 22)
  (h2 : w.avgSalaryAll = 850)
  (h3 : w.avgSalaryTech = 1000)
  (h4 : w.avgSalaryNonTech = 780) :
  numTechnicians w = 7 := by
  sorry

#eval numTechnicians ⟨22, 850, 1000, 780⟩

end NUMINAMATH_CALUDE_technicians_count_l1882_188278


namespace NUMINAMATH_CALUDE_isi_club_member_count_l1882_188237

/-- Represents a club with committees and members -/
structure Club where
  committee_count : ℕ
  member_count : ℕ
  committees_per_member : ℕ
  common_members : ℕ

/-- The ISI club satisfies the given conditions -/
def isi_club : Club :=
  { committee_count := 5,
    member_count := 10,
    committees_per_member := 2,
    common_members := 1 }

/-- Theorem: The ISI club has 10 members -/
theorem isi_club_member_count :
  isi_club.member_count = (isi_club.committee_count.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_isi_club_member_count_l1882_188237


namespace NUMINAMATH_CALUDE_john_change_theorem_l1882_188282

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total payment in cents given the number of each coin type -/
def total_payment (quarters dimes nickels : ℕ) : ℕ :=
  quarters * coin_value "quarter" + dimes * coin_value "dime" + nickels * coin_value "nickel"

/-- Calculates the change received given the total payment and the cost of the item -/
def change_received (payment cost : ℕ) : ℕ :=
  payment - cost

theorem john_change_theorem (candy_cost : ℕ) (h1 : candy_cost = 131) :
  change_received (total_payment 4 3 1) candy_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_change_theorem_l1882_188282


namespace NUMINAMATH_CALUDE_complex_magnitude_l1882_188271

theorem complex_magnitude (z : ℂ) (h : z^2 = 4 - 3*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1882_188271


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1882_188244

theorem arithmetic_expression_equality : 1874 + 230 / 46 - 874 * 2 = 131 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1882_188244


namespace NUMINAMATH_CALUDE_octagon_diagonal_intersections_l1882_188236

/-- The number of vertices in a regular octagon -/
def n : ℕ := 8

/-- The number of diagonals in a regular octagon -/
def num_diagonals : ℕ := n * (n - 3) / 2

/-- The number of distinct intersection points of diagonals in the interior of a regular octagon -/
def num_intersection_points : ℕ := Nat.choose n 4

theorem octagon_diagonal_intersections :
  num_intersection_points = 70 :=
sorry

end NUMINAMATH_CALUDE_octagon_diagonal_intersections_l1882_188236


namespace NUMINAMATH_CALUDE_exists_sum_of_five_squares_l1882_188258

theorem exists_sum_of_five_squares : 
  ∃ (n : ℕ) (a b c d e : ℤ), 
    (n : ℤ)^2 = a^2 + b^2 + c^2 + d^2 + e^2 ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e) ∧
    (a = 7 ∨ b = 7 ∨ c = 7 ∨ d = 7 ∨ e = 7) :=
by sorry

end NUMINAMATH_CALUDE_exists_sum_of_five_squares_l1882_188258


namespace NUMINAMATH_CALUDE_stating_last_passenger_seat_probability_l1882_188297

/-- 
Represents the probability that the last passenger sits in their assigned seat
given n seats and n passengers, where the first passenger (Absent-Minded Scientist)
takes a random seat, and subsequent passengers follow the described seating rules.
-/
def last_passenger_correct_seat_prob (n : ℕ) : ℚ :=
  if n ≥ 2 then 1/2 else 0

/-- 
Theorem stating that for any number of seats n ≥ 2, the probability that 
the last passenger sits in their assigned seat is 1/2.
-/
theorem last_passenger_seat_probability (n : ℕ) (h : n ≥ 2) : 
  last_passenger_correct_seat_prob n = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_stating_last_passenger_seat_probability_l1882_188297


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1882_188264

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstGroupSelection : ℕ) (groupNumber : ℕ) : ℕ :=
  firstGroupSelection + (groupNumber - 1) * (totalStudents / sampleSize)

/-- Theorem: In a systematic sampling of 400 students with a sample size of 20,
    if the selected number from the first group is 12,
    then the selected number from the 14th group is 272. -/
theorem systematic_sampling_theorem :
  systematicSample 400 20 12 14 = 272 := by
  sorry

#eval systematicSample 400 20 12 14

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1882_188264


namespace NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l1882_188275

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem for the first line
theorem line_through_P_and_origin :
  ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (x + y = 0 ↔ (∃ t : ℝ, x = t * P.1 ∧ y = t * P.2)) :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ :
  ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (2 * x + y + 2 = 0 ↔ 
    (∃ t : ℝ, x = P.1 + t * 1 ∧ y = P.2 + t * (-2) ∧ 
    (1 * (-2) + 2 * 1 = 0))) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l1882_188275


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1882_188242

-- Define the hyperbola equation
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 0)

-- Main theorem
theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_asymptote : ∃ x y, asymptote_equation x y ∧ hyperbola_equation a b x y)
  (h_focus : ∃ x y, hyperbola_equation a b x y ∧ (x, y) = parabola_focus) :
  a^2 = 4 ∧ b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1882_188242


namespace NUMINAMATH_CALUDE_larger_number_proof_l1882_188203

theorem larger_number_proof (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A - B = 1660) (h4 : 0.075 * A = 0.125 * B) : A = 4150 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1882_188203


namespace NUMINAMATH_CALUDE_min_value_sum_l1882_188234

theorem min_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 1) : 
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 1 → x + 4*y + 9*z ≤ a + 4*b + 9*c :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l1882_188234


namespace NUMINAMATH_CALUDE_abs_inequality_implies_a_greater_than_two_l1882_188285

theorem abs_inequality_implies_a_greater_than_two :
  (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_abs_inequality_implies_a_greater_than_two_l1882_188285


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_slope_product_constant_parallelogram_condition_l1882_188214

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := 9 * x^2 + y^2 = m^2

-- Define the line
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem ellipse_dot_product_range :
  ∀ (x y : ℝ), ellipse 3 x y →
  ∃ (f1x f1y f2x f2y : ℝ),
    f1x = 0 ∧ f1y = 2 * Real.sqrt 2 ∧
    f2x = 0 ∧ f2y = -2 * Real.sqrt 2 ∧
    -7 ≤ dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) ∧
    dot_product (x - f1x) (y - f1y) (x - f2x) (y - f2y) ≤ 1 :=
sorry

theorem slope_product_constant (m k b : ℝ) :
  k ≠ 0 → b ≠ 0 →
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse m x1 y1 ∧ ellipse m x2 y2 ∧
    line k b x1 y1 ∧ line k b x2 y2 ∧
    let x0 := (x1 + x2) / 2
    let y0 := (y1 + y2) / 2
    (y0 / x0) * k = -9 :=
sorry

theorem parallelogram_condition (m k : ℝ) :
  ellipse m (m/3) m →
  line k ((3-k)*m/3) (m/3) m →
  (∃ (x y : ℝ),
    ellipse m x y ∧
    line k ((3-k)*m/3) x y ∧
    x ≠ m/3 ∧ y ≠ m ∧
    (∃ (xp yp : ℝ),
      ellipse m xp yp ∧
      yp / xp = -9 / k ∧
      2 * (-(m - k*m/3)*k / (k^2 + 9)) = xp)) ↔
  (k = 4 + Real.sqrt 7 ∨ k = 4 - Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_slope_product_constant_parallelogram_condition_l1882_188214


namespace NUMINAMATH_CALUDE_angle_through_point_l1882_188292

/-- 
Given an angle α whose terminal side passes through point P(1, 2) in a plane coordinate system,
prove that:
1. tan α = 2
2. (sin α + 2 cos α) / (2 sin α - cos α) = 4/3
-/
theorem angle_through_point (α : Real) : 
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.tan α = 2 ∧ (Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_through_point_l1882_188292


namespace NUMINAMATH_CALUDE_sequence_problem_l1882_188207

theorem sequence_problem (a : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧
  (∀ k, a (k + 1) = a k + 3) ∧
  a n = 2014 →
  n = 672 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1882_188207


namespace NUMINAMATH_CALUDE_extreme_values_and_range_of_a_l1882_188226

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x^2 - x

theorem extreme_values_and_range_of_a :
  (∀ x : ℝ, f (1/4) x ≤ f (1/4) 0) ∧
  (∀ x : ℝ, f (1/4) x ≥ f (1/4) 1) ∧
  (f (1/4) 0 = 0) ∧
  (f (1/4) 1 = Real.log 2 - 3/4) ∧
  (∀ a : ℝ, (∀ b : ℝ, 1 < b → b < 2 → 
    (∀ x : ℝ, -1 < x → x ≤ b → f a x ≤ f a b)) →
    a ≥ 1 - Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_range_of_a_l1882_188226


namespace NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l1882_188289

theorem a_fourth_plus_b_fourth (a b : ℝ) 
  (h1 : a^2 - b^2 = 8) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 56 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l1882_188289


namespace NUMINAMATH_CALUDE_venkis_trip_speed_l1882_188229

/-- Venki's trip between towns X, Y, and Z -/
def venkis_trip (speed_xz speed_zy : ℝ) (time_xz time_zy : ℝ) : Prop :=
  let distance_xz := speed_xz * time_xz
  let distance_zy := distance_xz / 2
  speed_zy = distance_zy / time_zy

/-- The theorem statement for Venki's trip -/
theorem venkis_trip_speed :
  ∃ (speed_zy : ℝ),
    venkis_trip 80 speed_zy 5 (4 + 4/9) ∧
    abs (speed_zy - 42.86) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_venkis_trip_speed_l1882_188229


namespace NUMINAMATH_CALUDE_lemonade_cost_calculation_l1882_188201

/-- The cost of lemonade purchased by Coach Mike -/
def lemonade_cost : ℕ := sorry

/-- The amount Coach Mike gave to the girls -/
def amount_given : ℕ := 75

/-- The change Coach Mike received -/
def change_received : ℕ := 17

/-- Theorem stating that the lemonade cost is equal to the amount given minus the change received -/
theorem lemonade_cost_calculation : 
  lemonade_cost = amount_given - change_received := by sorry

end NUMINAMATH_CALUDE_lemonade_cost_calculation_l1882_188201


namespace NUMINAMATH_CALUDE_seventh_term_is_eight_l1882_188209

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, sum n = (n : ℝ) * (a 1 + a n) / 2
  third_eighth_sum : a 3 + a 8 = 13
  seventh_sum : sum 7 = 35

/-- The main theorem stating that the 7th term of the sequence is 8 -/
theorem seventh_term_is_eight (seq : ArithmeticSequence) : seq.a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_eight_l1882_188209


namespace NUMINAMATH_CALUDE_wheel_rotation_l1882_188202

/-- Proves that a wheel with given radius and arc length rotates by the calculated number of radians -/
theorem wheel_rotation (radius : ℝ) (arc_length : ℝ) (rotation : ℝ) 
  (h1 : radius = 20)
  (h2 : arc_length = 40)
  (h3 : rotation = arc_length / radius)
  (h4 : rotation > 0) : -- represents counterclockwise rotation
  rotation = 2 := by
  sorry

end NUMINAMATH_CALUDE_wheel_rotation_l1882_188202


namespace NUMINAMATH_CALUDE_initial_mushroom_amount_l1882_188254

-- Define the initial amount of mushrooms
def initial_amount : ℕ := sorry

-- Define the amount of mushrooms eaten
def eaten_amount : ℕ := 8

-- Define the amount of mushrooms left
def left_amount : ℕ := 7

-- Theorem stating that the initial amount is 15 pounds
theorem initial_mushroom_amount :
  initial_amount = eaten_amount + left_amount ∧ initial_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_mushroom_amount_l1882_188254


namespace NUMINAMATH_CALUDE_nine_nines_squared_zeros_l1882_188208

/-- The number of nines in 9,999,999 -/
def n : ℕ := 7

/-- The number 9,999,999 -/
def x : ℕ := 10^n - 1

/-- The number of zeros at the end of x^2 -/
def num_zeros (x : ℕ) : ℕ := n - 1

theorem nine_nines_squared_zeros :
  num_zeros x = 6 :=
sorry

end NUMINAMATH_CALUDE_nine_nines_squared_zeros_l1882_188208


namespace NUMINAMATH_CALUDE_intersection_distance_squared_l1882_188295

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the intersection points
def intersection_points (A B M : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  line_l M.1 M.2 ∧ M.1 = 0

-- State the theorem
theorem intersection_distance_squared 
  (A B M : ℝ × ℝ) 
  (h : intersection_points A B M) : 
  (Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2) + 
   Real.sqrt ((B.1 - M.1)^2 + (B.2 - M.2)^2))^2 = 16 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_l1882_188295


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l1882_188227

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l1882_188227


namespace NUMINAMATH_CALUDE_max_successful_throws_l1882_188251

/-- Represents the number of free throws attempted by Andrew -/
def andrew_throws : ℕ → ℕ := λ a => a

/-- Represents the number of free throws attempted by Beatrice -/
def beatrice_throws : ℕ → ℕ := λ b => b

/-- Represents the total number of free throws -/
def total_throws : ℕ := 105

/-- Represents the success rate of Andrew's free throws -/
def andrew_success_rate : ℚ := 1/3

/-- Represents the success rate of Beatrice's free throws -/
def beatrice_success_rate : ℚ := 3/5

/-- Calculates the total number of successful free throws -/
def total_successful_throws (a b : ℕ) : ℚ :=
  andrew_success_rate * a + beatrice_success_rate * b

/-- Theorem stating the maximum number of successful free throws -/
theorem max_successful_throws :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    a + b = total_throws ∧
    ∀ (x y : ℕ), 
      x > 0 → 
      y > 0 → 
      x + y = total_throws → 
      total_successful_throws a b ≥ total_successful_throws x y ∧
      total_successful_throws a b = 59 :=
sorry

end NUMINAMATH_CALUDE_max_successful_throws_l1882_188251


namespace NUMINAMATH_CALUDE_drews_age_l1882_188296

theorem drews_age (sam_current_age : ℕ) (h1 : sam_current_age = 46) :
  ∃ drew_current_age : ℕ,
    drew_current_age = 12 ∧
    sam_current_age + 5 = 3 * (drew_current_age + 5) :=
by sorry

end NUMINAMATH_CALUDE_drews_age_l1882_188296


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l1882_188228

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l1882_188228


namespace NUMINAMATH_CALUDE_original_number_proof_l1882_188281

theorem original_number_proof (x : ℝ) : 
  (1.25 * x - 0.70 * x = 22) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1882_188281


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1882_188290

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := x > 2

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1882_188290


namespace NUMINAMATH_CALUDE_double_inequality_solution_l1882_188232

theorem double_inequality_solution (x : ℝ) : 
  (0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2) ↔ 
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l1882_188232


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_complex_expression_l1882_188250

-- Part 1
theorem simplify_expression (a b : ℝ) :
  2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 := by sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (h : x^2 + 2*y = 4) :
  -3*x^2 - 6*y + 17 = 5 := by sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 3*b = 3) (h2 : 2*b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2*b - d) - (2*b - c) = 7 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_complex_expression_l1882_188250


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1882_188299

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1882_188299


namespace NUMINAMATH_CALUDE_fraction_equality_l1882_188257

theorem fraction_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4) :
  (a + b + c) / (2*a + b - c) = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1882_188257


namespace NUMINAMATH_CALUDE_obtuseTrianglesIn120Gon_l1882_188286

/-- The number of vertices in the regular polygon -/
def n : ℕ := 120

/-- A function that calculates the number of ways to choose three vertices 
    forming an obtuse triangle in a regular n-gon -/
def obtuseTrianglesCount (n : ℕ) : ℕ :=
  n * (n / 2 - 1) * (n / 2 - 2) / 2

/-- Theorem stating that the number of ways to choose three vertices 
    forming an obtuse triangle in a regular 120-gon is 205320 -/
theorem obtuseTrianglesIn120Gon : obtuseTrianglesCount n = 205320 := by
  sorry

end NUMINAMATH_CALUDE_obtuseTrianglesIn120Gon_l1882_188286


namespace NUMINAMATH_CALUDE_beef_cabbage_cost_comparison_l1882_188235

/-- Represents the cost calculation for beef and spicy cabbage orders --/
theorem beef_cabbage_cost_comparison (a : ℝ) (h : a > 50) :
  (4500 + 27 * a) ≤ (4400 + 30 * a) := by
  sorry

#check beef_cabbage_cost_comparison

end NUMINAMATH_CALUDE_beef_cabbage_cost_comparison_l1882_188235


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l1882_188274

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 4500 →
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 1350 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l1882_188274


namespace NUMINAMATH_CALUDE_a_perp_b_l1882_188259

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def are_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The given vectors -/
def a : ℝ × ℝ := (-5, 6)
def b : ℝ × ℝ := (6, 5)

/-- Theorem: Vectors a and b are perpendicular -/
theorem a_perp_b : are_perpendicular a b := by sorry

end NUMINAMATH_CALUDE_a_perp_b_l1882_188259


namespace NUMINAMATH_CALUDE_garrison_provisions_duration_l1882_188279

/-- The number of days provisions last for a garrison with reinforcements --/
def provisions_duration (initial_men : ℕ) (reinforcement_men : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_men * days_before_reinforcement + (initial_men + reinforcement_men) * days_after_reinforcement) / initial_men

/-- Theorem stating that given the problem conditions, the provisions were supposed to last 54 days initially --/
theorem garrison_provisions_duration :
  provisions_duration 2000 1600 18 20 = 54 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provisions_duration_l1882_188279


namespace NUMINAMATH_CALUDE_product_sum_quotient_l1882_188252

theorem product_sum_quotient (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x * y = 9375 ∧ x + y = 400 → max x y / min x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_quotient_l1882_188252


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1882_188255

/-- Three concentric circles with radii 1, 2, and 3 units -/
def circles := {r : ℝ | r = 1 ∨ r = 2 ∨ r = 3}

/-- A point on one of the circles -/
structure CirclePoint where
  x : ℝ
  y : ℝ
  r : ℝ
  on_circle : x^2 + y^2 = r^2
  radius_valid : r ∈ circles

/-- An equilateral triangle formed by points on the circles -/
structure EquilateralTriangle where
  A : CirclePoint
  B : CirclePoint
  C : CirclePoint
  equilateral : (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2 ∧
                (B.x - C.x)^2 + (B.y - C.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2
  on_different_circles : A.r ≠ B.r ∧ B.r ≠ C.r ∧ C.r ≠ A.r

/-- The theorem stating that the side length of the equilateral triangle is √7 -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangle) : 
  (triangle.A.x - triangle.B.x)^2 + (triangle.A.y - triangle.B.y)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1882_188255


namespace NUMINAMATH_CALUDE_least_number_of_cans_l1882_188239

theorem least_number_of_cans (maaza pepsi sprite : ℕ) 
  (h_maaza : maaza = 80)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368) :
  let can_size := Nat.gcd maaza (Nat.gcd pepsi sprite)
  let total_cans := maaza / can_size + pepsi / can_size + sprite / can_size
  total_cans = 37 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l1882_188239


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1882_188225

def p (x y : ℝ) : Prop := x > 1 ∨ y > 2

def q (x y : ℝ) : Prop := x + y > 3

theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧ 
  (∃ x y : ℝ, p x y ∧ ¬(q x y)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1882_188225


namespace NUMINAMATH_CALUDE_outlets_per_room_l1882_188261

theorem outlets_per_room (total_rooms : ℕ) (total_outlets : ℕ) (h1 : total_rooms = 7) (h2 : total_outlets = 42) :
  total_outlets / total_rooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_outlets_per_room_l1882_188261


namespace NUMINAMATH_CALUDE_variance_of_specific_set_l1882_188245

theorem variance_of_specific_set (a : ℝ) : 
  (5 + 8 + a + 7 + 4) / 5 = a → 
  ((5 - a)^2 + (8 - a)^2 + (a - a)^2 + (7 - a)^2 + (4 - a)^2) / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_specific_set_l1882_188245


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1882_188205

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of members per school -/
def members_per_school : ℕ := 6

/-- The number of representatives from the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of ways to arrange the presidency meeting -/
def total_arrangements : ℕ := num_schools * (Nat.choose members_per_school host_representatives) * (Nat.choose members_per_school non_host_representatives)^2

theorem presidency_meeting_arrangements :
  total_arrangements = 2160 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1882_188205


namespace NUMINAMATH_CALUDE_square_numbers_existence_l1882_188270

theorem square_numbers_existence (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ (q1 q2 : ℕ), q1.Prime ∧ q2.Prime ∧ q1 ≠ q2 ∧
    ¬(p^2 ∣ (q1^(p-1) - 1)) ∧ ¬(p^2 ∣ (q2^(p-1) - 1)) := by
  sorry

end NUMINAMATH_CALUDE_square_numbers_existence_l1882_188270
