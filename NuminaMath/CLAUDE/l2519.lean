import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l2519_251964

-- Define the function f(x) = x³ - 3x²
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧  -- f is increasing on (-∞, 0)
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧  -- f is decreasing on (0, 2)
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧  -- f is increasing on (2, +∞)
  (∀ x, x ≠ 0 → f x ≤ f 0) ∧  -- f(0) is a local maximum
  (∀ x, x ≠ 2 → f x ≥ f 2) ∧  -- f(2) is a local minimum
  f 0 = 0 ∧  -- value at x = 0
  f 2 = -4  -- value at x = 2
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l2519_251964


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2519_251970

/-- An arithmetic sequence. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 16)
  (h_a4 : a 4 = 6) :
  a 6 = 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2519_251970


namespace NUMINAMATH_CALUDE_hurdle_distance_l2519_251914

theorem hurdle_distance (total_distance : ℕ) (num_hurdles : ℕ) (start_distance : ℕ) (end_distance : ℕ) 
  (h1 : total_distance = 600)
  (h2 : num_hurdles = 12)
  (h3 : start_distance = 50)
  (h4 : end_distance = 55) :
  ∃ d : ℕ, d = 45 ∧ total_distance = start_distance + (num_hurdles - 1) * d + end_distance :=
by sorry

end NUMINAMATH_CALUDE_hurdle_distance_l2519_251914


namespace NUMINAMATH_CALUDE_remaining_wire_length_l2519_251925

-- Define the initial wire length in centimeters
def initial_length_cm : ℝ := 23.3

-- Define the first cut in millimeters
def first_cut_mm : ℝ := 105

-- Define the second cut in centimeters
def second_cut_cm : ℝ := 4.6

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Theorem statement
theorem remaining_wire_length :
  (initial_length_cm * cm_to_mm - first_cut_mm - second_cut_cm * cm_to_mm) = 82 := by
  sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l2519_251925


namespace NUMINAMATH_CALUDE_negation_of_existence_l2519_251961

theorem negation_of_existence (p : Prop) :
  (¬∃ x₀ : ℝ, x₀ ≥ 1 ∧ x₀^2 - x₀ < 0) ↔ (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2519_251961


namespace NUMINAMATH_CALUDE_aluminum_weight_in_compound_l2519_251950

/-- The molecular weight of the aluminum part in Al2(CO3)3 -/
def aluminum_weight : ℝ := 2 * 26.98

/-- Proof that the molecular weight of the aluminum part in Al2(CO3)3 is 53.96 g/mol -/
theorem aluminum_weight_in_compound : aluminum_weight = 53.96 := by
  sorry

end NUMINAMATH_CALUDE_aluminum_weight_in_compound_l2519_251950


namespace NUMINAMATH_CALUDE_smallest_n_value_l2519_251983

/-- The number of ordered quadruplets (a, b, c, d) satisfying the given conditions -/
def quadruplet_count : ℕ := 84000

/-- The given GCD value -/
def gcd_value : ℕ := 84

/-- The function that counts the number of ordered quadruplets (a, b, c, d) 
    satisfying gcd(a, b, c, d) = gcd_value and lcm(a, b, c, d) = n -/
def count_quadruplets (n : ℕ) : ℕ := sorry

/-- The theorem stating the smallest n that satisfies the conditions -/
theorem smallest_n_value : 
  (∀ m < 1555848, count_quadruplets m ≠ quadruplet_count) ∧ 
  count_quadruplets 1555848 = quadruplet_count := by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2519_251983


namespace NUMINAMATH_CALUDE_parabola_translation_l2519_251986

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 3x² -/
def original_parabola : Parabola := ⟨3, 0, 0⟩

/-- The vertical translation amount -/
def translation : ℝ := -2

/-- Translates a parabola vertically by a given amount -/
def translate_vertically (p : Parabola) (t : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + t⟩

/-- The resulting parabola after translation -/
def resulting_parabola : Parabola :=
  translate_vertically original_parabola translation

theorem parabola_translation :
  resulting_parabola = ⟨3, 0, -2⟩ := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2519_251986


namespace NUMINAMATH_CALUDE_no_three_distinct_squares_sum_to_100_l2519_251920

/-- A function that checks if a natural number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The proposition that there are no three distinct positive perfect squares that sum to 100 --/
theorem no_three_distinct_squares_sum_to_100 : 
  ¬ ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectSquare c ∧
    a + b + c = 100 :=
sorry

end NUMINAMATH_CALUDE_no_three_distinct_squares_sum_to_100_l2519_251920


namespace NUMINAMATH_CALUDE_snow_probability_value_l2519_251923

/-- The probability of snow occurring at least once in a week, where the first 4 days 
    have a 1/4 chance of snow each day and the next 3 days have a 1/3 chance of snow each day. -/
def snow_probability : ℚ := 1 - (3/4)^4 * (2/3)^3

/-- Theorem stating that the probability of snow occurring at least once in the described week
    is equal to 125/128. -/
theorem snow_probability_value : snow_probability = 125/128 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_value_l2519_251923


namespace NUMINAMATH_CALUDE_product_equality_l2519_251934

theorem product_equality (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2519_251934


namespace NUMINAMATH_CALUDE_alex_fourth_test_score_l2519_251951

theorem alex_fourth_test_score :
  ∀ (s1 s2 s3 s4 s5 : ℕ),
  (85 ≤ s1 ∧ s1 ≤ 95) ∧
  (85 ≤ s2 ∧ s2 ≤ 95) ∧
  (85 ≤ s3 ∧ s3 ≤ 95) ∧
  (85 ≤ s4 ∧ s4 ≤ 95) ∧
  (s5 = 90) ∧
  (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧
   s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧
   s3 ≠ s4 ∧ s3 ≠ s5 ∧
   s4 ≠ s5) ∧
  (∃ (k : ℕ), (s1 + s2) = 2 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3) = 3 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3 + s4) = 4 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3 + s4 + s5) = 5 * k) →
  s4 = 95 :=
by sorry

end NUMINAMATH_CALUDE_alex_fourth_test_score_l2519_251951


namespace NUMINAMATH_CALUDE_min_value_abc_l2519_251922

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + 2*a*b + 2*a*c + 4*b*c = 12) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + 2*x*y + 2*x*z + 4*y*z = 12 → 
  a + b + c ≤ x + y + z ∧ a + b + c ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l2519_251922


namespace NUMINAMATH_CALUDE_dichromate_molecular_weight_l2519_251903

/-- Given that the molecular weight of 9 moles of Dichromate is 2664 g/mol,
    prove that the molecular weight of one mole of Dichromate is 296 g/mol. -/
theorem dichromate_molecular_weight :
  let mw_9_moles : ℝ := 2664 -- molecular weight of 9 moles in g/mol
  let num_moles : ℝ := 9 -- number of moles
  mw_9_moles / num_moles = 296 := by sorry

end NUMINAMATH_CALUDE_dichromate_molecular_weight_l2519_251903


namespace NUMINAMATH_CALUDE_cake_sale_theorem_l2519_251953

/-- Represents the pricing and sales model for small cakes in a charity sale event -/
structure CakeSaleModel where
  initial_price : ℝ
  initial_sales : ℕ
  price_increase : ℝ
  sales_decrease : ℕ
  max_price : ℝ

/-- Calculates the new price after two equal percentage increases -/
def price_after_two_increases (model : CakeSaleModel) (percent : ℝ) : ℝ :=
  model.initial_price * (1 + percent) ^ 2

/-- Calculates the total sales per hour given a price increase -/
def total_sales (model : CakeSaleModel) (price_increase : ℝ) : ℝ :=
  (model.initial_price + price_increase) * 
  (model.initial_sales - model.sales_decrease * price_increase)

/-- The main theorem stating the correct percentage increase and optimal selling price -/
theorem cake_sale_theorem (model : CakeSaleModel) 
  (h1 : model.initial_price = 6)
  (h2 : model.initial_sales = 30)
  (h3 : model.price_increase = 1)
  (h4 : model.sales_decrease = 2)
  (h5 : model.max_price = 10) :
  ∃ (percent : ℝ) (optimal_price : ℝ),
    price_after_two_increases model percent = 8.64 ∧
    percent = 0.2 ∧
    total_sales model (optimal_price - model.initial_price) = 216 ∧
    optimal_price = 9 ∧
    optimal_price ≤ model.max_price :=
by sorry

end NUMINAMATH_CALUDE_cake_sale_theorem_l2519_251953


namespace NUMINAMATH_CALUDE_actual_sleep_time_l2519_251995

/-- The required sleep time for middle school students -/
def requiredSleepTime : ℝ := 9

/-- The recorded excess sleep time for Xiao Ming -/
def recordedExcessTime : ℝ := 0.4

/-- Theorem: Actual sleep time is the sum of required sleep time and recorded excess time -/
theorem actual_sleep_time : 
  requiredSleepTime + recordedExcessTime = 9.4 := by
  sorry

end NUMINAMATH_CALUDE_actual_sleep_time_l2519_251995


namespace NUMINAMATH_CALUDE_calculation_difference_l2519_251948

theorem calculation_difference : 
  let correct_calculation := 10 - (3 * 4)
  let incorrect_calculation := 10 - 3 + 4
  correct_calculation - incorrect_calculation = -13 := by
sorry

end NUMINAMATH_CALUDE_calculation_difference_l2519_251948


namespace NUMINAMATH_CALUDE_arrangement_problem_l2519_251938

theorem arrangement_problem (n : ℕ) (h1 : n ≥ 2) : 
  ((n - 1) * (n - 1) = 25) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_problem_l2519_251938


namespace NUMINAMATH_CALUDE_expression_value_l2519_251959

theorem expression_value (x : ℤ) (h : x = -3) : x^2 - 4*(x - 5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2519_251959


namespace NUMINAMATH_CALUDE_house_height_calculation_l2519_251989

/-- Given two trees with consistent shadow ratios and a house with a known shadow length,
    prove that the height of the house can be determined. -/
theorem house_height_calculation (tree1_height tree1_shadow tree2_height tree2_shadow house_shadow : ℝ)
    (h1 : tree1_height > 0)
    (h2 : tree2_height > 0)
    (h3 : tree1_shadow > 0)
    (h4 : tree2_shadow > 0)
    (h5 : house_shadow > 0)
    (h6 : tree1_shadow / tree1_height = tree2_shadow / tree2_height) :
    ∃ (house_height : ℝ), house_height = tree1_height * (house_shadow / tree1_shadow) :=
  sorry

end NUMINAMATH_CALUDE_house_height_calculation_l2519_251989


namespace NUMINAMATH_CALUDE_remaining_cherries_l2519_251943

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem remaining_cherries : initial_cherries - cherries_used = 17 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cherries_l2519_251943


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l2519_251976

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l2519_251976


namespace NUMINAMATH_CALUDE_subtracted_number_l2519_251996

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem subtracted_number (x : Nat) : 
  (sum_of_digits (10^38 - x) = 330) → 
  (x = 10^37 + 3 * 10^36) :=
by sorry

end NUMINAMATH_CALUDE_subtracted_number_l2519_251996


namespace NUMINAMATH_CALUDE_negation_of_square_positivity_l2519_251982

theorem negation_of_square_positivity :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_square_positivity_l2519_251982


namespace NUMINAMATH_CALUDE_tims_kittens_l2519_251978

theorem tims_kittens (initial_kittens : ℕ) : 
  (initial_kittens > 0) →
  (initial_kittens * 2 / 3 * 3 / 5 = 12) →
  initial_kittens = 30 := by
sorry

end NUMINAMATH_CALUDE_tims_kittens_l2519_251978


namespace NUMINAMATH_CALUDE_speeding_fine_lawyer_hours_mark_speeding_fine_l2519_251907

theorem speeding_fine_lawyer_hours 
  (base_fine : ℕ) 
  (fine_increase_per_mph : ℕ) 
  (actual_speed : ℕ) 
  (speed_limit : ℕ) 
  (court_costs : ℕ) 
  (lawyer_hourly_rate : ℕ) 
  (total_owed : ℕ) : ℕ :=
  let speed_over_limit := actual_speed - speed_limit
  let speed_penalty := speed_over_limit * fine_increase_per_mph
  let initial_fine := base_fine + speed_penalty
  let doubled_fine := initial_fine * 2
  let fine_with_court_costs := doubled_fine + court_costs
  let lawyer_fees := total_owed - fine_with_court_costs
  lawyer_fees / lawyer_hourly_rate

theorem mark_speeding_fine 
  (h1 : speeding_fine_lawyer_hours 50 2 75 30 300 80 820 = 3) : 
  speeding_fine_lawyer_hours 50 2 75 30 300 80 820 = 3 := by
  sorry

end NUMINAMATH_CALUDE_speeding_fine_lawyer_hours_mark_speeding_fine_l2519_251907


namespace NUMINAMATH_CALUDE_number_difference_l2519_251949

theorem number_difference (a b c d : ℝ) : 
  a = 2 * b ∧ 
  a = 3 * c ∧ 
  (a + b + c + d) / 4 = 110 ∧ 
  d = a + b + c 
  → a - c = 80 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2519_251949


namespace NUMINAMATH_CALUDE_three_times_relation_l2519_251954

/-- Given four numbers M₁, M₂, M₃, and M₄, prove that M₄ = 3M₂ -/
theorem three_times_relation (M₁ M₂ M₃ M₄ : ℝ) 
  (hM₁ : M₁ = 2.02e-6)
  (hM₂ : M₂ = 0.0000202)
  (hM₃ : M₃ = 0.00000202)
  (hM₄ : M₄ = 6.06e-5) :
  M₄ = 3 * M₂ := by
  sorry

end NUMINAMATH_CALUDE_three_times_relation_l2519_251954


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2519_251960

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 ≥ 121 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ θ φ : ℝ, (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2519_251960


namespace NUMINAMATH_CALUDE_houses_not_yellow_l2519_251967

theorem houses_not_yellow (green yellow red : ℕ) : 
  green = 3 * yellow →
  yellow + 40 = red →
  green = 90 →
  green + red = 160 :=
by sorry

end NUMINAMATH_CALUDE_houses_not_yellow_l2519_251967


namespace NUMINAMATH_CALUDE_third_chapter_pages_l2519_251918

theorem third_chapter_pages (total_pages first_chapter second_chapter : ℕ) 
  (h1 : total_pages = 125)
  (h2 : first_chapter = 66)
  (h3 : second_chapter = 35) :
  total_pages - (first_chapter + second_chapter) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_third_chapter_pages_l2519_251918


namespace NUMINAMATH_CALUDE_determinant_property_l2519_251985

theorem determinant_property (p q r s : ℝ) 
  (h : Matrix.det !![p, q; r, s] = 3) : 
  Matrix.det !![2*p, 2*p + 5*q; 2*r, 2*r + 5*s] = 30 := by
  sorry

end NUMINAMATH_CALUDE_determinant_property_l2519_251985


namespace NUMINAMATH_CALUDE_f_range_f_period_one_l2519_251958

-- Define the nearest integer function
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if x - ⌊x⌋ ≤ 1/2 then ⌊x⌋ else ⌈x⌉

-- Define the function f(x) = x - {x}
noncomputable def f (x : ℝ) : ℝ := x - nearest_integer x

-- Theorem stating the range of f(x)
theorem f_range : Set.range f = Set.Ioc (-1/2) (1/2) := by sorry

-- Theorem stating that f(x) has a period of 1
theorem f_period_one (x : ℝ) : f (x + 1) = f x := by sorry

end NUMINAMATH_CALUDE_f_range_f_period_one_l2519_251958


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l2519_251952

theorem ten_thousandths_place_of_5_32 : 
  (5 : ℚ) / 32 * 10000 - ((5 : ℚ) / 32 * 10000).floor = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l2519_251952


namespace NUMINAMATH_CALUDE_probability_of_specific_outcome_l2519_251963

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure SixCoins :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (halfDollar : CoinFlip)
  (oneDollar : CoinFlip)

/-- The total number of possible outcomes when flipping six coins -/
def totalOutcomes : Nat := 64

/-- A specific outcome we're interested in -/
def specificOutcome : SixCoins :=
  { penny := CoinFlip.Heads,
    nickel := CoinFlip.Heads,
    dime := CoinFlip.Heads,
    quarter := CoinFlip.Tails,
    halfDollar := CoinFlip.Tails,
    oneDollar := CoinFlip.Tails }

/-- The probability of getting the specific outcome when flipping six coins -/
theorem probability_of_specific_outcome :
  (1 : ℚ) / totalOutcomes = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_outcome_l2519_251963


namespace NUMINAMATH_CALUDE_arcsin_one_half_l2519_251926

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l2519_251926


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l2519_251947

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^15732 + i^15733 + i^15734 + i^15735 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l2519_251947


namespace NUMINAMATH_CALUDE_min_cars_in_group_l2519_251957

/-- Represents the properties of a group of cars -/
structure CarGroup where
  total : ℕ
  withAC : ℕ
  withStripes : ℕ
  withACNoStripes : ℕ

/-- The conditions of the car group problem -/
def validCarGroup (g : CarGroup) : Prop :=
  g.total - g.withAC = 47 ∧
  g.withStripes ≥ 55 ∧
  g.withACNoStripes ≤ 45

/-- The theorem stating the minimum number of cars in the group -/
theorem min_cars_in_group (g : CarGroup) (h : validCarGroup g) : g.total ≥ 102 := by
  sorry

#check min_cars_in_group

end NUMINAMATH_CALUDE_min_cars_in_group_l2519_251957


namespace NUMINAMATH_CALUDE_parabola_single_intersection_parabola_y_decreases_l2519_251988

def parabola (x m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_single_intersection (m : ℝ) :
  (∃! x, parabola x m = 0) ↔ m = -2 := sorry

theorem parabola_y_decreases (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  parabola x₁ m = y₁ →
  parabola x₂ m = y₂ →
  x₁ > x₂ →
  x₂ > 2 →
  y₁ < y₂ := sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_parabola_y_decreases_l2519_251988


namespace NUMINAMATH_CALUDE_parallel_lines_a_values_l2519_251999

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 10 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 3 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ a x y → l₂ a x y

-- Theorem statement
theorem parallel_lines_a_values :
  ∀ a : ℝ, parallel a → (a = -1 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_values_l2519_251999


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2519_251901

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * Real.sqrt 2 * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2519_251901


namespace NUMINAMATH_CALUDE_gcd_3570_4840_l2519_251927

theorem gcd_3570_4840 : Nat.gcd 3570 4840 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3570_4840_l2519_251927


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_difference_l2519_251987

theorem crazy_silly_school_series_difference : 
  let num_books : ℕ := 15
  let num_movies : ℕ := 14
  num_books - num_movies = 1 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_difference_l2519_251987


namespace NUMINAMATH_CALUDE_hyperbola_intersection_range_l2519_251990

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 2

-- Main theorem
theorem hyperbola_intersection_range :
  ∀ k : ℝ, 
    (intersects_at_two_points k ∧ 
     ∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧ 
                        line_l k x₁ y₁ ∧ line_l k x₂ y₂ → 
                        dot_product_condition x₁ y₁ x₂ y₂) →
    (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_range_l2519_251990


namespace NUMINAMATH_CALUDE_no_intersection_empty_union_equality_iff_l2519_251919

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Theorem 1: There is no value of a such that A ∩ B = ∅
theorem no_intersection_empty (a : ℝ) : (A a) ∩ B ≠ ∅ := by
  sorry

-- Theorem 2: A ∪ B = B if and only if a ∈ (-∞, -4) ∪ (5, ∞)
theorem union_equality_iff (a : ℝ) : (A a) ∪ B = B ↔ a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_empty_union_equality_iff_l2519_251919


namespace NUMINAMATH_CALUDE_library_visitors_on_sunday_l2519_251921

/-- Proves that the average number of visitors on Sundays is 660 given the specified conditions --/
theorem library_visitors_on_sunday (total_days : Nat) (non_sunday_avg : Nat) (overall_avg : Nat) : 
  total_days = 30 →
  non_sunday_avg = 240 →
  overall_avg = 310 →
  (5 * (total_days * overall_avg - 25 * non_sunday_avg)) / 5 = 660 := by
  sorry

#check library_visitors_on_sunday

end NUMINAMATH_CALUDE_library_visitors_on_sunday_l2519_251921


namespace NUMINAMATH_CALUDE_rectangle_area_equation_l2519_251905

theorem rectangle_area_equation : ∃! x : ℝ, x > 3 ∧ (x - 3) * (3 * x + 4) = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_equation_l2519_251905


namespace NUMINAMATH_CALUDE_salary_increase_20_percent_l2519_251932

-- Define Sharon's original weekly salary
variable (S : ℝ)

-- Define the condition that a 16% increase results in $406
axiom increase_16_percent : S * 1.16 = 406

-- Define the target salary of $420
def target_salary : ℝ := 420

-- Theorem to prove
theorem salary_increase_20_percent : 
  S * 1.20 = target_salary := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_20_percent_l2519_251932


namespace NUMINAMATH_CALUDE_rectangular_yard_area_l2519_251910

theorem rectangular_yard_area (w : ℝ) (l : ℝ) : 
  l = 2 * w + 30 →
  2 * w + 2 * l = 700 →
  w * l = 233600 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_yard_area_l2519_251910


namespace NUMINAMATH_CALUDE_expression_evaluation_l2519_251916

theorem expression_evaluation :
  let a : ℚ := -3/2
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2519_251916


namespace NUMINAMATH_CALUDE_a_2018_mod_49_l2519_251902

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : a 2018 % 49 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_2018_mod_49_l2519_251902


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l2519_251935

/-- Given a cube with side length 6, the volume of the tetrahedron formed by any vertex
    and the three vertices connected to that vertex by edges of the cube is 36. -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (tetrahedron_volume : ℝ) :
  cube_side_length = 6 →
  tetrahedron_volume = (1 / 3) * (1 / 2 * cube_side_length * cube_side_length) * cube_side_length →
  tetrahedron_volume = 36 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l2519_251935


namespace NUMINAMATH_CALUDE_percentage_failed_both_l2519_251994

/-- Percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 35

/-- Percentage of students who failed in English -/
def failed_english : ℝ := 45

/-- Percentage of students who passed in both subjects -/
def passed_both : ℝ := 40

/-- Percentage of students who failed in both subjects -/
def failed_both : ℝ := 20

theorem percentage_failed_both :
  failed_both = failed_hindi + failed_english - (100 - passed_both) := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_both_l2519_251994


namespace NUMINAMATH_CALUDE_chess_players_per_game_l2519_251913

/-- The number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ := n.choose k

/-- The total number of games played when each player plays each other once -/
def totalGames (n : ℕ) (k : ℕ) : ℕ := combinations n k

theorem chess_players_per_game (n k : ℕ) (h1 : n = 30) (h2 : totalGames n k = 435) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_players_per_game_l2519_251913


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2519_251945

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2519_251945


namespace NUMINAMATH_CALUDE_linear_function_properties_l2519_251911

def f (x : ℝ) : ℝ := x + 2

theorem linear_function_properties :
  (f 1 = 3) ∧
  (f (-2) = 0) ∧
  (∀ x y, f x = y → x ≥ 0 ∧ y ≤ 0 → x = 0 ∧ y = 2) ∧
  (∃ x, x > 2 ∧ f x ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2519_251911


namespace NUMINAMATH_CALUDE_same_total_price_implies_sams_price_l2519_251998

/-- The price per sheet charged by Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := sorry

/-- The price per sheet charged by John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The sitting fee charged by John's Photo World -/
def johns_sitting_fee : ℝ := 125

/-- The sitting fee charged by Sam's Picture Emporium -/
def sams_sitting_fee : ℝ := 140

/-- The number of sheets in the package -/
def num_sheets : ℕ := 12

theorem same_total_price_implies_sams_price (h : johns_price_per_sheet * num_sheets + johns_sitting_fee = sams_price_per_sheet * num_sheets + sams_sitting_fee) : 
  sams_price_per_sheet = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_same_total_price_implies_sams_price_l2519_251998


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2519_251917

/-- For a regular polygon with exterior angles of 45 degrees, the sum of interior angles is 1080 degrees. -/
theorem regular_polygon_interior_angle_sum : 
  ∀ (n : ℕ), n > 2 → (360 / n = 45) → (n - 2) * 180 = 1080 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2519_251917


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l2519_251941

theorem chess_game_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.9) : 
  prob_A_not_lose - prob_A_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l2519_251941


namespace NUMINAMATH_CALUDE_first_square_length_is_correct_l2519_251933

/-- The length of the first square of fabric -/
def first_square_length : ℝ := 8

/-- The height of the first square of fabric -/
def first_square_height : ℝ := 5

/-- The length of the second square of fabric -/
def second_square_length : ℝ := 10

/-- The height of the second square of fabric -/
def second_square_height : ℝ := 7

/-- The length of the third square of fabric -/
def third_square_length : ℝ := 5

/-- The height of the third square of fabric -/
def third_square_height : ℝ := 5

/-- The desired length of the flag -/
def flag_length : ℝ := 15

/-- The desired height of the flag -/
def flag_height : ℝ := 9

theorem first_square_length_is_correct : 
  first_square_length * first_square_height + 
  second_square_length * second_square_height + 
  third_square_length * third_square_height = 
  flag_length * flag_height := by
  sorry

end NUMINAMATH_CALUDE_first_square_length_is_correct_l2519_251933


namespace NUMINAMATH_CALUDE_compound_bar_chart_must_have_legend_l2519_251937

/-- Represents a compound bar chart -/
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_different_colors_or_patterns : Bool

/-- Theorem: A compound bar chart must have a clearly indicated legend -/
theorem compound_bar_chart_must_have_legend (chart : CompoundBarChart) 
  (h1 : chart.distinguishes_two_quantities = true)
  (h2 : chart.uses_different_colors_or_patterns = true) : 
  ∃ legend : Bool, legend = true :=
sorry

end NUMINAMATH_CALUDE_compound_bar_chart_must_have_legend_l2519_251937


namespace NUMINAMATH_CALUDE_unrepaired_road_not_thirty_percent_l2519_251977

/-- Represents the percentage of road repaired in the first phase -/
def first_phase_repair : ℝ := 0.4

/-- Represents the percentage of remaining road repaired in the second phase -/
def second_phase_repair : ℝ := 0.3

/-- Represents the total length of the road in meters -/
def total_road_length : ℝ := 200

/-- Theorem stating that the unrepaired portion of the road is not 30% -/
theorem unrepaired_road_not_thirty_percent :
  let remaining_after_first := 1 - first_phase_repair
  let repaired_in_second := remaining_after_first * second_phase_repair
  let total_repaired := first_phase_repair + repaired_in_second
  total_repaired ≠ 0.7 := by sorry

end NUMINAMATH_CALUDE_unrepaired_road_not_thirty_percent_l2519_251977


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2519_251939

theorem sum_product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x*y + y*z + z*x) * (1/(x+y)^2 + 1/(y+z)^2 + 1/(z+x)^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2519_251939


namespace NUMINAMATH_CALUDE_four_X_three_l2519_251940

/-- The operation X defined for any two real numbers -/
def X (a b : ℝ) : ℝ := b + 7*a - a^3 + 2*b

/-- Theorem stating that 4 X 3 = -27 -/
theorem four_X_three : X 4 3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_four_X_three_l2519_251940


namespace NUMINAMATH_CALUDE_triangle_max_area_l2519_251969

/-- The maximum area of a triangle with medians satisfying certain conditions -/
theorem triangle_max_area (m_a m_b m_c : ℝ) 
  (h_a : m_a ≤ 2) (h_b : m_b ≤ 3) (h_c : m_c ≤ 4) : 
  (∃ (E : ℝ), E = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) ∧
  (∀ (E' : ℝ), E' = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) → E' ≤ E)) →
  (∃ (E_max : ℝ), E_max = 4 ∧
  (∀ (E : ℝ), E = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) → E ≤ E_max)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2519_251969


namespace NUMINAMATH_CALUDE_horner_v2_equals_16_l2519_251944

/-- Horner's method for evaluating a polynomial -/
def horner_v2 (x : ℝ) : ℝ :=
  let v0 : ℝ := x
  let v1 : ℝ := 3 * v0 + 1
  v1 * v0 + 2

/-- The polynomial f(x) = 3x^4 + x^3 + 2x^2 + x + 4 -/
def f (x : ℝ) : ℝ := 3*x^4 + x^3 + 2*x^2 + x + 4

theorem horner_v2_equals_16 : horner_v2 2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_horner_v2_equals_16_l2519_251944


namespace NUMINAMATH_CALUDE_digit_move_correctness_l2519_251946

theorem digit_move_correctness : 
  let original_number := 102
  let moved_digit := 2
  let base := 10
  let new_left_term := original_number - moved_digit
  let new_right_term := base ^ moved_digit
  (new_left_term - new_right_term = 1) = True
  := by sorry

end NUMINAMATH_CALUDE_digit_move_correctness_l2519_251946


namespace NUMINAMATH_CALUDE_decision_box_is_diamond_l2519_251991

/-- A type representing different shapes that can be used in a flowchart --/
inductive FlowchartShape
  | Rectangle
  | Diamond
  | Oval
  | Parallelogram

/-- A function that returns the shape used for decision boxes in a flowchart --/
def decisionBoxShape : FlowchartShape := FlowchartShape.Diamond

/-- Theorem stating that the decision box in a flowchart is represented by a diamond shape --/
theorem decision_box_is_diamond : decisionBoxShape = FlowchartShape.Diamond := by
  sorry

end NUMINAMATH_CALUDE_decision_box_is_diamond_l2519_251991


namespace NUMINAMATH_CALUDE_truncated_tetrahedron_volume_squared_l2519_251912

/-- A truncated tetrahedron is a solid with 4 triangular faces and 4 hexagonal faces. --/
structure TruncatedTetrahedron where
  side_length : ℝ
  triangular_faces : Fin 4
  hexagonal_faces : Fin 4

/-- The volume of a truncated tetrahedron. --/
noncomputable def volume (t : TruncatedTetrahedron) : ℝ := sorry

/-- Theorem: The square of the volume of a truncated tetrahedron with side length 1 is 529/72. --/
theorem truncated_tetrahedron_volume_squared :
  ∀ (t : TruncatedTetrahedron), t.side_length = 1 → (volume t)^2 = 529/72 := by sorry

end NUMINAMATH_CALUDE_truncated_tetrahedron_volume_squared_l2519_251912


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l2519_251966

theorem rachel_milk_consumption (don_milk : ℚ) (rachel_fraction : ℚ) :
  don_milk = 3 / 7 →
  rachel_fraction = 4 / 5 →
  rachel_fraction * don_milk = 12 / 35 :=
by sorry

end NUMINAMATH_CALUDE_rachel_milk_consumption_l2519_251966


namespace NUMINAMATH_CALUDE_towel_folding_theorem_l2519_251908

-- Define the folding rates for each person
def jane_rate : ℚ := 5 / 5
def kyla_rate : ℚ := 9 / 10
def anthony_rate : ℚ := 14 / 20
def david_rate : ℚ := 6 / 15

-- Define the total number of towels folded in one hour
def total_towels : ℕ := 180

-- Theorem statement
theorem towel_folding_theorem :
  (jane_rate + kyla_rate + anthony_rate + david_rate) * 60 = total_towels := by
  sorry

end NUMINAMATH_CALUDE_towel_folding_theorem_l2519_251908


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2519_251900

/-- Given a boat traveling downstream with a stream rate of 5 km/hr and covering 84 km in 4 hours,
    the speed of the boat in still water is 16 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (stream_rate : ℝ) (distance : ℝ) (time : ℝ) (boat_speed : ℝ),
    stream_rate = 5 →
    distance = 84 →
    time = 4 →
    distance = (boat_speed + stream_rate) * time →
    boat_speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2519_251900


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2519_251924

theorem perfect_square_condition (x y k : ℝ) : 
  (∃ a : ℝ, x^2 + k*x*y + 81*y^2 = a^2) ↔ k = 18 ∨ k = -18 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2519_251924


namespace NUMINAMATH_CALUDE_tennis_ball_order_l2519_251971

theorem tennis_ball_order (white yellow : ℕ) : 
  white = yellow →                        -- Initially equal number of white and yellow balls
  (white : ℚ) / (yellow + 70 : ℚ) = 8/13 →  -- Ratio after error
  white + yellow = 224 :=                 -- Total number of balls ordered
by sorry

end NUMINAMATH_CALUDE_tennis_ball_order_l2519_251971


namespace NUMINAMATH_CALUDE_centerIsSeven_l2519_251936

-- Define the type for our 3x3 array
def Array3x3 := Fin 3 → Fin 3 → Fin 9

-- Define what it means for two positions to be adjacent
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

-- Define the property of consecutive numbers
def consecutive (n m : Fin 9) : Prop :=
  n.val + 1 = m.val ∨ m.val + 1 = n.val

-- Define the property that consecutive numbers are in adjacent squares
def consecutiveAdjacent (arr : Array3x3) : Prop :=
  ∀ i j k l, consecutive (arr i j) (arr k l) → adjacent (i, j) (k, l)

-- Define the property that corner numbers sum to 20
def cornerSum20 (arr : Array3x3) : Prop :=
  (arr 0 0).val + (arr 0 2).val + (arr 2 0).val + (arr 2 2).val = 20

-- Define the property that all numbers from 1 to 9 are used
def allNumbersUsed (arr : Array3x3) : Prop :=
  ∀ n : Fin 9, ∃ i j, arr i j = n

-- The main theorem
theorem centerIsSeven (arr : Array3x3) 
  (h1 : consecutiveAdjacent arr) 
  (h2 : cornerSum20 arr) 
  (h3 : allNumbersUsed arr) : 
  arr 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_centerIsSeven_l2519_251936


namespace NUMINAMATH_CALUDE_triple_overlap_is_six_l2519_251956

/-- Represents a rectangular carpet with width and height -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the arrangement of carpets -/
structure CarpetArrangement where
  hallWidth : ℝ
  hallHeight : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap in the carpet arrangement -/
def tripleOverlapArea (arrangement : CarpetArrangement) : ℝ :=
  sorry

/-- Theorem stating that the triple overlap area is 6 square meters -/
theorem triple_overlap_is_six (arrangement : CarpetArrangement) 
  (h1 : arrangement.hallWidth = 10 ∧ arrangement.hallHeight = 10)
  (h2 : arrangement.carpet1 = ⟨6, 8⟩)
  (h3 : arrangement.carpet2 = ⟨6, 6⟩)
  (h4 : arrangement.carpet3 = ⟨5, 7⟩) :
  tripleOverlapArea arrangement = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_is_six_l2519_251956


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2519_251975

/-- Given that x and y are inversely proportional, x + y = 30, and x - y = 10,
    prove that y = 200/7 when x = 7 -/
theorem inverse_proportion_problem (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k)
    (h2 : x + y = 30) (h3 : x - y = 10) : 
    x = 7 → y = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2519_251975


namespace NUMINAMATH_CALUDE_equation_solution_l2519_251992

theorem equation_solution : ∃ x : ℝ, 61 + 5 * x / (180 / 3) = 62 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2519_251992


namespace NUMINAMATH_CALUDE_equation_solution_l2519_251930

theorem equation_solution : 
  {x : ℝ | (x + 2)^4 + (x - 4)^4 = 272} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2519_251930


namespace NUMINAMATH_CALUDE_married_couples_with_2_to_4_children_l2519_251997

/-- The fraction of married couples with 2 to 4 children in a population with given characteristics -/
theorem married_couples_with_2_to_4_children (total_population : ℕ) 
  (married_couple_percentage : ℚ) (one_child : ℚ) (two_children : ℚ) 
  (three_children : ℚ) (four_children : ℚ) (five_children : ℚ) :
  total_population = 10000 →
  married_couple_percentage = 1/5 →
  one_child = 1/5 →
  two_children = 1/4 →
  three_children = 3/20 →
  four_children = 1/6 →
  five_children = 1/10 →
  two_children + three_children + four_children = 17/30 := by
  sorry


end NUMINAMATH_CALUDE_married_couples_with_2_to_4_children_l2519_251997


namespace NUMINAMATH_CALUDE_function_inequality_l2519_251980

/-- Given functions f and g, prove that if 2f(x) ≥ g(x) for all x > 0, then a ≤ 4 -/
theorem function_inequality (a : ℝ) : 
  (∀ x > 0, 2 * (x * Real.log x) ≥ -x^2 + a*x - 3) → a ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_function_inequality_l2519_251980


namespace NUMINAMATH_CALUDE_sum_50_to_75_l2519_251909

/-- Sum of integers from a to b, inclusive -/
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

/-- Theorem: The sum of all integers from 50 through 75, inclusive, is 1625 -/
theorem sum_50_to_75 : sum_integers 50 75 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_sum_50_to_75_l2519_251909


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2519_251972

/-- A regular polygon with interior angles of 160 degrees and side length of 4 units has 18 sides and a perimeter of 72 units. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (side_length : ℝ),
    n > 2 →
    side_length = 4 →
    (180 * (n - 2) : ℝ) / n = 160 →
    n = 18 ∧ n * side_length = 72 := by
  sorry

#check regular_polygon_properties

end NUMINAMATH_CALUDE_regular_polygon_properties_l2519_251972


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2519_251962

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1011011110₂ -/
def binary_input : List Bool := [true, false, true, true, false, true, true, true, true, false]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_input) = [2, 3, 1, 3, 2] :=
sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2519_251962


namespace NUMINAMATH_CALUDE_cricketer_matches_l2519_251993

theorem cricketer_matches (score1 score2 overall_avg : ℚ) (matches1 matches2 : ℕ) :
  score1 = 40 →
  score2 = 10 →
  matches1 = 2 →
  matches2 = 3 →
  overall_avg = 22 →
  (score1 * matches1 + score2 * matches2) / (matches1 + matches2) = overall_avg →
  matches1 + matches2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_matches_l2519_251993


namespace NUMINAMATH_CALUDE_first_part_value_l2519_251915

theorem first_part_value (x y : ℝ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_part_value_l2519_251915


namespace NUMINAMATH_CALUDE_additional_miles_for_average_speed_l2519_251974

theorem additional_miles_for_average_speed 
  (initial_distance : ℝ) 
  (initial_speed : ℝ) 
  (desired_average_speed : ℝ) 
  (additional_speed : ℝ) : 
  initial_distance = 20 ∧ 
  initial_speed = 40 ∧ 
  desired_average_speed = 55 ∧ 
  additional_speed = 60 → 
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / 
    (initial_distance / initial_speed + additional_distance / additional_speed) = 
    desired_average_speed ∧ 
    additional_distance = 90 := by
sorry

end NUMINAMATH_CALUDE_additional_miles_for_average_speed_l2519_251974


namespace NUMINAMATH_CALUDE_letter_150_is_Z_l2519_251965

/-- Represents the letters in the repeating pattern -/
inductive Letter
| X
| Y
| Z

/-- The length of the repeating pattern -/
def pattern_length : Nat := 3

/-- Function to determine the nth letter in the repeating pattern -/
def nth_letter (n : Nat) : Letter :=
  match n % pattern_length with
  | 0 => Letter.Z
  | 1 => Letter.X
  | _ => Letter.Y

/-- Theorem stating that the 150th letter in the pattern is Z -/
theorem letter_150_is_Z : nth_letter 150 = Letter.Z := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_Z_l2519_251965


namespace NUMINAMATH_CALUDE_right_triangle_area_l2519_251928

/-- The area of a right triangle with one leg of length 3 and hypotenuse of length 5 is 6. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : c = 5) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2519_251928


namespace NUMINAMATH_CALUDE_independence_test_incorrect_judgment_l2519_251929

/-- The chi-squared test statistic -/
def K_squared : ℝ := 4.05

/-- The significance level (α) for the test -/
def significance_level : ℝ := 0.05

/-- The critical value for the chi-squared distribution with 1 degree of freedom at 0.05 significance level -/
def critical_value : ℝ := 3.841

/-- The probability of incorrect judgment in an independence test -/
def probability_incorrect_judgment : ℝ := significance_level

theorem independence_test_incorrect_judgment :
  K_squared > critical_value →
  probability_incorrect_judgment = significance_level :=
sorry

end NUMINAMATH_CALUDE_independence_test_incorrect_judgment_l2519_251929


namespace NUMINAMATH_CALUDE_multiply_18396_9999_l2519_251979

theorem multiply_18396_9999 : 18396 * 9999 = 183941604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_18396_9999_l2519_251979


namespace NUMINAMATH_CALUDE_three_roots_iff_b_in_range_l2519_251955

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The statement that f(x) + b = 0 has three distinct real roots iff -1 < b < 0 -/
theorem three_roots_iff_b_in_range (b : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x + b = 0 ∧ f y + b = 0 ∧ f z + b = 0) ↔ 
  -1 < b ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_three_roots_iff_b_in_range_l2519_251955


namespace NUMINAMATH_CALUDE_paul_buys_two_toys_l2519_251981

/-- The number of toys Paul can buy given his savings, allowance, and toy price -/
def toys_paul_can_buy (savings : ℕ) (allowance : ℕ) (toy_price : ℕ) : ℕ :=
  (savings + allowance) / toy_price

/-- Theorem: Paul can buy 2 toys with his savings and allowance -/
theorem paul_buys_two_toys :
  toys_paul_can_buy 3 7 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_paul_buys_two_toys_l2519_251981


namespace NUMINAMATH_CALUDE_tank_bucket_ratio_l2519_251906

theorem tank_bucket_ratio : 
  ∀ (tank_capacity bucket_capacity : ℚ),
  tank_capacity > 0 → bucket_capacity > 0 →
  ∃ (water_transferred : ℚ),
  (3/5 * tank_capacity - water_transferred = 2/3 * tank_capacity) ∧
  (1/4 * bucket_capacity + water_transferred = 1/2 * bucket_capacity) →
  tank_capacity / bucket_capacity = 15/4 := by
sorry

end NUMINAMATH_CALUDE_tank_bucket_ratio_l2519_251906


namespace NUMINAMATH_CALUDE_platform_length_l2519_251942

/-- Given a train of length 750 m that crosses a platform in 65 seconds
    and a signal pole in 30 seconds, the length of the platform is 875 m. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 750)
  (h2 : platform_crossing_time = 65)
  (h3 : pole_crossing_time = 30) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 875 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2519_251942


namespace NUMINAMATH_CALUDE_nonzero_digits_count_l2519_251904

def original_fraction : ℚ := 120 / (2^5 * 5^10)

def decimal_result : ℝ := (original_fraction : ℝ) - 0.000001

def count_nonzero_digits (x : ℝ) : ℕ :=
  sorry -- Implementation of counting non-zero digits after decimal point

theorem nonzero_digits_count :
  count_nonzero_digits decimal_result = 3 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_digits_count_l2519_251904


namespace NUMINAMATH_CALUDE_expression_simplification_l2519_251973

theorem expression_simplification :
  let x := (1 : ℝ) / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 7 - 2)))
  let y := (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) / ((9 * Real.sqrt 5 + 4 * Real.sqrt 7)^2 - 100)
  x = y := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2519_251973


namespace NUMINAMATH_CALUDE_min_y_intercept_l2519_251968

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 12*x + 11

-- Define the y-intercept of the tangent line as a function of x
def r (x : ℝ) : ℝ := -2*x^3 + 6*x^2 - 6

-- Theorem statement
theorem min_y_intercept :
  ∀ x ∈ Set.Icc 0 2, r 0 ≤ r x :=
sorry

end NUMINAMATH_CALUDE_min_y_intercept_l2519_251968


namespace NUMINAMATH_CALUDE_definite_integral_abs_x_squared_minus_two_l2519_251984

theorem definite_integral_abs_x_squared_minus_two :
  ∫ x in (-2)..1, |x^2 - 2| = 1/3 + 8*Real.sqrt 2/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_abs_x_squared_minus_two_l2519_251984


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2519_251931

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2519_251931
