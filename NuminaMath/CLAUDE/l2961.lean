import Mathlib

namespace NUMINAMATH_CALUDE_book_selection_probability_l2961_296177

theorem book_selection_probability :
  let n : ℕ := 12  -- Total number of books
  let k : ℕ := 6   -- Number of books each student selects
  let m : ℕ := 3   -- Number of books in common

  -- Probability of selecting exactly m books in common
  (Nat.choose n m * Nat.choose (n - m) (k - m) * Nat.choose (n - k) (k - m) : ℚ) /
  (Nat.choose n k * Nat.choose n k : ℚ) = 100 / 231 := by
sorry

end NUMINAMATH_CALUDE_book_selection_probability_l2961_296177


namespace NUMINAMATH_CALUDE_min_calls_proof_l2961_296169

/-- Represents the minimum number of calls per month -/
def min_calls : ℕ := 66

/-- Represents the monthly rental fee in yuan -/
def rental_fee : ℚ := 12

/-- Represents the cost per call in yuan -/
def cost_per_call : ℚ := (1/5 : ℚ)

/-- Represents the minimum monthly phone bill in yuan -/
def min_monthly_bill : ℚ := 25

theorem min_calls_proof :
  (min_calls : ℚ) * cost_per_call + rental_fee > min_monthly_bill ∧
  ∀ n : ℕ, n < min_calls → (n : ℚ) * cost_per_call + rental_fee ≤ min_monthly_bill :=
by sorry

end NUMINAMATH_CALUDE_min_calls_proof_l2961_296169


namespace NUMINAMATH_CALUDE_wrong_observation_value_l2961_296173

theorem wrong_observation_value (n : ℕ) (original_mean corrected_mean correct_value : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : corrected_mean = 41.5)
  (h4 : correct_value = 48) :
  let wrong_value := n * original_mean - (n * corrected_mean - correct_value)
  wrong_value = 23 := by sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l2961_296173


namespace NUMINAMATH_CALUDE_prime_roots_sum_reciprocals_l2961_296150

theorem prime_roots_sum_reciprocals (p q m : ℕ) : 
  Prime p → Prime q → 
  (p : ℝ)^2 - 99*p + m = 0 → 
  (q : ℝ)^2 - 99*q + m = 0 → 
  (p : ℝ)/q + (q : ℝ)/p = 9413/194 := by
  sorry

end NUMINAMATH_CALUDE_prime_roots_sum_reciprocals_l2961_296150


namespace NUMINAMATH_CALUDE_pyramid_sphere_radii_relation_main_theorem_l2961_296171

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  h : ℝ  -- height of the pyramid
  a : ℝ  -- half of the base edge length

/-- The theorem stating the relationship between R and r for a regular quadrilateral pyramid -/
theorem pyramid_sphere_radii_relation (p : RegularQuadrilateralPyramid) :
  p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem :
  ∀ p : RegularQuadrilateralPyramid, p.R ≥ (Real.sqrt 2 + 1) * p.r := by
  intro p
  exact pyramid_sphere_radii_relation p

end NUMINAMATH_CALUDE_pyramid_sphere_radii_relation_main_theorem_l2961_296171


namespace NUMINAMATH_CALUDE_unique_triple_existence_l2961_296195

theorem unique_triple_existence (p : ℕ) (hp : Prime p) 
  (h_prime : ∀ n : ℕ, 0 < n → n < p → Prime (n^2 - n + p)) :
  ∃! (a b c : ℤ), 
    b^2 - 4*a*c = 1 - 4*p ∧ 
    0 < a ∧ a ≤ c ∧ 
    -a ≤ b ∧ b < a ∧
    a = 1 ∧ b = -1 ∧ c = p := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_existence_l2961_296195


namespace NUMINAMATH_CALUDE_sticker_count_l2961_296122

theorem sticker_count (stickers_per_page : ℕ) (number_of_pages : ℕ) : 
  stickers_per_page = 10 → number_of_pages = 22 → stickers_per_page * number_of_pages = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l2961_296122


namespace NUMINAMATH_CALUDE_tablet_cash_price_l2961_296145

/-- Represents the installment plan for a tablet purchase -/
structure InstallmentPlan where
  downPayment : ℕ
  firstFourMonths : ℕ
  middleFourMonths : ℕ
  lastFourMonths : ℕ
  savings : ℕ

/-- Calculates the cash price of the tablet given the installment plan -/
def cashPrice (plan : InstallmentPlan) : ℕ :=
  plan.downPayment +
  4 * plan.firstFourMonths +
  4 * plan.middleFourMonths +
  4 * plan.lastFourMonths -
  plan.savings

/-- Theorem stating that the cash price of the tablet is 450 -/
theorem tablet_cash_price :
  let plan := InstallmentPlan.mk 100 40 35 30 70
  cashPrice plan = 450 := by
  sorry

end NUMINAMATH_CALUDE_tablet_cash_price_l2961_296145


namespace NUMINAMATH_CALUDE_largest_number_with_property_l2961_296103

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def property (n : Nat) : Prop :=
  n % sum_of_digits n = 0

theorem largest_number_with_property :
  ∃ (n : Nat), n < 900 ∧ property n ∧ ∀ (m : Nat), m < 900 → property m → m ≤ n :=
by
  use 888
  sorry

#eval sum_of_digits 888  -- Should output 24
#eval 888 % 24           -- Should output 0

end NUMINAMATH_CALUDE_largest_number_with_property_l2961_296103


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2961_296124

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 4) : x^3 - 1/x^3 = 76 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2961_296124


namespace NUMINAMATH_CALUDE_assembly_line_increased_rate_l2961_296199

/-- Represents the production rate of an assembly line -/
structure AssemblyLine where
  initial_rate : ℝ
  increased_rate : ℝ
  initial_order : ℝ
  second_order : ℝ
  average_output : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem assembly_line_increased_rate (a : AssemblyLine) 
  (h1 : a.initial_rate = 30)
  (h2 : a.initial_order = 60)
  (h3 : a.second_order = 60)
  (h4 : a.average_output = 40)
  (h5 : (a.initial_order + a.second_order) / 
        (a.initial_order / a.initial_rate + a.second_order / a.increased_rate) = a.average_output) :
  a.increased_rate = 60 := by
  sorry


end NUMINAMATH_CALUDE_assembly_line_increased_rate_l2961_296199


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l2961_296137

/-- Recursive definition of the sequence F_i -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ F 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 2 → F n = 3 * F (n - 1) - F (n - 2)) ∧
  F 12 % 23 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l2961_296137


namespace NUMINAMATH_CALUDE_cube_surface_area_l2961_296198

/-- Given a cube where the sum of all edge lengths is 180 cm, 
    prove that its surface area is 1350 cm². -/
theorem cube_surface_area (edge_sum : ℝ) (h_edge_sum : edge_sum = 180) :
  let edge_length := edge_sum / 12
  6 * edge_length^2 = 1350 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2961_296198


namespace NUMINAMATH_CALUDE_rain_hours_calculation_l2961_296147

theorem rain_hours_calculation (total_hours rain_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : rain_hours = 4) : 
  total_hours - rain_hours = 5 := by
sorry

end NUMINAMATH_CALUDE_rain_hours_calculation_l2961_296147


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2961_296139

/-- An arithmetic sequence and its partial sums with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ+, S n = (n : ℝ) * (a 1 + a n) / 2
  S5_lt_S6 : S 5 < S 6
  S6_eq_S7 : S 6 = S 7
  S7_gt_S8 : S 7 > S 8

/-- The common difference of the arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 6 > 0 ∧ seq.a 7 = 0 ∧ seq.a 8 < 0 ∧ common_difference seq < 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2961_296139


namespace NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_sqrt_three_l2961_296136

/-- The function f as defined in the problem -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 3 * x * y + 1

/-- The theorem statement -/
theorem abs_a_plus_b_equals_three_sqrt_three
  (a b : ℝ)
  (h1 : f a b + 1 = 42)
  (h2 : f b a = 42) :
  |a + b| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_sqrt_three_l2961_296136


namespace NUMINAMATH_CALUDE_car_speed_conversion_l2961_296154

/-- Conversion factor from m/s to km/h -/
def conversion_factor : ℝ := 3.6

/-- Speed of the car in m/s -/
def speed_ms : ℝ := 10

/-- Speed of the car in km/h -/
def speed_kmh : ℝ := speed_ms * conversion_factor

theorem car_speed_conversion :
  speed_kmh = 36 := by sorry

end NUMINAMATH_CALUDE_car_speed_conversion_l2961_296154


namespace NUMINAMATH_CALUDE_complement_union_eq_result_l2961_296197

-- Define the universal set U
def U : Set ℕ := {x | 0 ≤ x ∧ x < 5}

-- Define sets P and Q
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {2, 4}

-- Theorem statement
theorem complement_union_eq_result : (U \ P) ∪ Q = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_eq_result_l2961_296197


namespace NUMINAMATH_CALUDE_arithmetic_seq_property_l2961_296119

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ + a₆ = 2, prove that a₄ = 1 -/
theorem arithmetic_seq_property (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_seq a) (h_sum : a 2 + a 6 = 2) : 
  a 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_property_l2961_296119


namespace NUMINAMATH_CALUDE_max_m_value_l2961_296144

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) →
  m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l2961_296144


namespace NUMINAMATH_CALUDE_probability_of_pair_l2961_296109

/-- Represents a deck of cards with their counts -/
def Deck := List (Nat × Nat)

/-- The initial deck configuration -/
def initial_deck : Deck := List.replicate 10 (5, 5)

/-- Remove a matching pair from the deck -/
def remove_pair (d : Deck) : Deck :=
  match d with
  | (n, count) :: rest => if count ≥ 2 then (n, count - 2) :: rest else d
  | [] => []

/-- Calculate the total number of cards in the deck -/
def total_cards (d : Deck) : Nat :=
  d.foldr (fun (_, count) acc => acc + count) 0

/-- Calculate the number of ways to choose 2 cards from n cards -/
def choose_2 (n : Nat) : Nat := n * (n - 1) / 2

/-- Calculate the number of possible pairs in the deck -/
def count_pairs (d : Deck) : Nat :=
  d.foldr (fun (_, count) acc => acc + choose_2 count) 0

theorem probability_of_pair (d : Deck) :
  let remaining_deck := remove_pair initial_deck
  let total := total_cards remaining_deck
  let pairs := count_pairs remaining_deck
  (pairs : Rat) / (choose_2 total) = 31 / 376 := by sorry

end NUMINAMATH_CALUDE_probability_of_pair_l2961_296109


namespace NUMINAMATH_CALUDE_largest_remainder_269_l2961_296143

theorem largest_remainder_269 (n : ℕ) (h : n < 150) :
  ∃ (q r : ℕ), 269 = n * q + r ∧ r < n ∧ r ≤ 133 ∧
  (∀ (q' r' : ℕ), 269 = n * q' + r' ∧ r' < n → r' ≤ r) :=
sorry

end NUMINAMATH_CALUDE_largest_remainder_269_l2961_296143


namespace NUMINAMATH_CALUDE_similar_triangles_leg_l2961_296189

/-- Two similar right triangles with legs 10 and 8 in the first triangle,
    and x and 5 in the second triangle. -/
structure SimilarRightTriangles where
  x : ℝ
  similarity : (10 : ℝ) / x = 8 / 5

theorem similar_triangles_leg (t : SimilarRightTriangles) : t.x = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_l2961_296189


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2961_296183

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Predicate to check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If P(m, 1) is in the second quadrant, then B(-m+1, -1) is in the fourth quadrant -/
theorem point_in_fourth_quadrant (m : ℝ) :
  isInSecondQuadrant (Point.mk m 1) → isInFourthQuadrant (Point.mk (-m + 1) (-1)) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2961_296183


namespace NUMINAMATH_CALUDE_total_bill_correct_l2961_296156

/-- Represents the group composition and meal prices -/
structure GroupInfo where
  adults : Nat
  teenagers : Nat
  children : Nat
  adultMealPrice : ℚ
  teenagerMealPrice : ℚ
  childrenMealPrice : ℚ
  adultSodaPrice : ℚ
  childrenSodaPrice : ℚ
  appetizerPrice : ℚ
  dessertPrice : ℚ

/-- Represents the number of additional items ordered -/
structure AdditionalItems where
  appetizers : Nat
  desserts : Nat

/-- Represents the discount conditions -/
structure DiscountConditions where
  adultMealDiscount : ℚ
  childrenMealSodaDiscount : ℚ
  totalBillDiscount : ℚ
  minChildrenForDiscount : Nat
  teenagersPerFreeDessert : Nat
  minTotalForExtraDiscount : ℚ

/-- Calculates the total bill after all applicable discounts and special offers -/
def calculateTotalBill (group : GroupInfo) (items : AdditionalItems) (discounts : DiscountConditions) : ℚ :=
  sorry

/-- Theorem stating that the calculated total bill matches the expected result -/
theorem total_bill_correct (group : GroupInfo) (items : AdditionalItems) (discounts : DiscountConditions) :
  let expectedBill : ℚ := 230.70
  calculateTotalBill group items discounts = expectedBill :=
by
  sorry

end NUMINAMATH_CALUDE_total_bill_correct_l2961_296156


namespace NUMINAMATH_CALUDE_larger_complementary_angle_measure_l2961_296162

def complementary_angles (a b : ℝ) : Prop := a + b = 90

theorem larger_complementary_angle_measure :
  ∀ (x y : ℝ),
    complementary_angles x y →
    x / y = 4 / 3 →
    x > y →
    x = 51 + 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_larger_complementary_angle_measure_l2961_296162


namespace NUMINAMATH_CALUDE_unoccupied_area_l2961_296176

/-- The area of the region not occupied by a smaller square inside a larger square -/
theorem unoccupied_area (large_side small_side : ℝ) (h1 : large_side = 10) (h2 : small_side = 4) :
  large_side ^ 2 - small_side ^ 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_area_l2961_296176


namespace NUMINAMATH_CALUDE_third_day_breath_holding_l2961_296160

def breath_holding_sequence (n : ℕ) : ℕ :=
  10 * n

theorem third_day_breath_holding :
  let seq := breath_holding_sequence
  seq 1 = 10 ∧ 
  seq 2 = 20 ∧ 
  seq 6 = 90 →
  seq 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_third_day_breath_holding_l2961_296160


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2961_296159

/-- The circumcircle of a triangle ABC with vertices A(-√3, 0), B(√3, 0), and C(0, 3) 
    has the equation x² + (y - 1)² = 4 -/
theorem circumcircle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-Real.sqrt 3, 0)
  let B : ℝ × ℝ := (Real.sqrt 3, 0)
  let C : ℝ × ℝ := (0, 3)
  x^2 + (y - 1)^2 = 4 ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2 :=
by sorry


end NUMINAMATH_CALUDE_circumcircle_equation_l2961_296159


namespace NUMINAMATH_CALUDE_sons_age_l2961_296184

/-- Given a man and his son, where the man is 18 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 16 years. -/
theorem sons_age (man_age son_age : ℕ) : 
  man_age = son_age + 18 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l2961_296184


namespace NUMINAMATH_CALUDE_function_extrema_l2961_296178

/-- Given constants a and b, if the function f(x) = ax^3 + b*ln(x + sqrt(1+x^2)) + 3
    has a maximum value of 10 on the interval (-∞, 0),
    then the minimum value of f(x) on the interval (0, +∞) is -4. -/
theorem function_extrema (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^3 + b * Real.log (x + Real.sqrt (1 + x^2)) + 3
  (∃ (M : ℝ), M = 10 ∧ ∀ (x : ℝ), x < 0 → f x ≤ M) →
  ∃ (m : ℝ), m = -4 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
by sorry


end NUMINAMATH_CALUDE_function_extrema_l2961_296178


namespace NUMINAMATH_CALUDE_target_sectors_degrees_l2961_296153

def circle_degrees : ℝ := 360

def microphotonics_percent : ℝ := 12
def home_electronics_percent : ℝ := 17
def food_additives_percent : ℝ := 9
def genetically_modified_microorganisms_percent : ℝ := 22
def industrial_lubricants_percent : ℝ := 6
def artificial_intelligence_percent : ℝ := 4
def nanotechnology_percent : ℝ := 5

def basic_astrophysics_percent : ℝ :=
  100 - (microphotonics_percent + home_electronics_percent + food_additives_percent +
         genetically_modified_microorganisms_percent + industrial_lubricants_percent +
         artificial_intelligence_percent + nanotechnology_percent)

def target_sectors_percent : ℝ :=
  basic_astrophysics_percent + artificial_intelligence_percent + nanotechnology_percent

theorem target_sectors_degrees :
  target_sectors_percent * (circle_degrees / 100) = 122.4 := by
  sorry

end NUMINAMATH_CALUDE_target_sectors_degrees_l2961_296153


namespace NUMINAMATH_CALUDE_dog_weight_multiple_l2961_296164

/-- Given the weights of three dogs (chihuahua, pitbull, and great dane), 
    prove that the great dane's weight is 3 times the pitbull's weight plus 10 pounds. -/
theorem dog_weight_multiple (c p g : ℝ) 
  (h1 : c + p + g = 439)  -- Total weight
  (h2 : p = 3 * c)        -- Pitbull's weight relation to chihuahua
  (h3 : g = 307)          -- Great dane's weight
  : ∃ m : ℝ, g = m * p + 10 ∧ m = 3 := by
  sorry

#check dog_weight_multiple

end NUMINAMATH_CALUDE_dog_weight_multiple_l2961_296164


namespace NUMINAMATH_CALUDE_seed_mixture_percentage_l2961_296112

/-- Given two seed mixtures X and Y, and their combination, 
    this theorem proves that the percentage of mixture X in the final mixture is 1/3. -/
theorem seed_mixture_percentage (x y : ℝ) :
  x ≥ 0 → y ≥ 0 →  -- Ensure non-negative weights
  0.40 * x + 0.25 * y = 0.30 * (x + y) →  -- Ryegrass balance equation
  x / (x + y) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_seed_mixture_percentage_l2961_296112


namespace NUMINAMATH_CALUDE_log_base_three_seven_l2961_296127

theorem log_base_three_seven (a b : ℝ) (h1 : Real.log 2 / Real.log 3 = a) (h2 : Real.log 7 / Real.log 2 = b) :
  Real.log 7 / Real.log 3 = a * b := by
  sorry

end NUMINAMATH_CALUDE_log_base_three_seven_l2961_296127


namespace NUMINAMATH_CALUDE_max_fraction_value_l2961_296193

theorem max_fraction_value : 
  ∃ (a b c d e f : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    f ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a / b + c / d) / (e / f) = 14 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      y ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      z ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      w ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      u ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      v ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧
      w ≠ u ∧ w ≠ v ∧
      u ≠ v →
      (x / y + z / w) / (u / v) ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_value_l2961_296193


namespace NUMINAMATH_CALUDE_min_reciprocal_distances_min_value_achieved_l2961_296108

/-- Given a right triangle ABC with AC = 4 and BC = 1, and a point P on the hypotenuse AB
    (excluding endpoints) with distances d1 and d2 to the legs, the minimum value of (1/d1 + 1/d2) is 9/4 -/
theorem min_reciprocal_distances (d1 d2 : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d1 + 4 * d2 = 4) :
  (1 / d1 + 1 / d2) ≥ 9 / 4 := by
  sorry

/-- The minimum value 9/4 is achieved when d1 = 4/3 and d2 = 2/3 -/
theorem min_value_achieved (d1 d2 : ℝ) (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d1 + 4 * d2 = 4) :
  (1 / d1 + 1 / d2 = 9 / 4) ↔ (d1 = 4 / 3 ∧ d2 = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_distances_min_value_achieved_l2961_296108


namespace NUMINAMATH_CALUDE_max_true_statements_l2961_296115

theorem max_true_statements (x : ℝ) : ∃ (n : ℕ), n ≤ 3 ∧
  n = (Bool.toNat (0 < x^2 ∧ x^2 < 4) +
       Bool.toNat (x^2 > 4) +
       Bool.toNat (-2 < x ∧ x < 0) +
       Bool.toNat (0 < x ∧ x < 2) +
       Bool.toNat (0 < x - x^2 ∧ x - x^2 < 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l2961_296115


namespace NUMINAMATH_CALUDE_range_of_a_for_equation_solution_l2961_296102

theorem range_of_a_for_equation_solution (a : ℝ) : 
  (∃ x : ℝ, (a + Real.cos x) * (a - Real.sin x) = 1) ↔ 
  (a ∈ Set.Icc (-1 - Real.sqrt 2 / 2) (-1 + Real.sqrt 2 / 2) ∪ 
   Set.Icc (1 - Real.sqrt 2 / 2) (1 + Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_equation_solution_l2961_296102


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l2961_296190

/-- The number of routes on a grid with more right moves than down moves -/
def num_routes (right down : ℕ) : ℕ :=
  Nat.choose (right + down) down

/-- The grid dimensions -/
def grid_width : ℕ := 3
def grid_height : ℕ := 2

theorem routes_on_3x2_grid :
  num_routes grid_width grid_height = 21 :=
sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l2961_296190


namespace NUMINAMATH_CALUDE_set_theory_propositions_l2961_296106

theorem set_theory_propositions (A B : Set α) : 
  (∀ a, a ∈ A → a ∈ A ∪ B) ∧
  (A ⊆ B → A ∪ B = B) ∧
  (A ∪ B = B → A ∩ B = A) ∧
  ¬(∀ a, a ∈ B → a ∈ A ∩ B) ∧
  ¬(∀ C, A ∪ B = B ∪ C → A = C) :=
by sorry

end NUMINAMATH_CALUDE_set_theory_propositions_l2961_296106


namespace NUMINAMATH_CALUDE_chord_intersection_segments_l2961_296140

theorem chord_intersection_segments (r : ℝ) (chord_length : ℝ) 
  (hr : r = 7) (hchord : chord_length = 10) : 
  ∃ (ak kb : ℝ), 
    ak = r - 2 * Real.sqrt 6 ∧ 
    kb = r + 2 * Real.sqrt 6 ∧ 
    ak + kb = 2 * r ∧
    ak * kb = (chord_length / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_segments_l2961_296140


namespace NUMINAMATH_CALUDE_prime_between_50_60_mod_7_l2961_296131

theorem prime_between_50_60_mod_7 :
  ∀ n : ℕ,
  (Prime n) →
  (50 < n) →
  (n < 60) →
  (n % 7 = 4) →
  n = 53 :=
by sorry

end NUMINAMATH_CALUDE_prime_between_50_60_mod_7_l2961_296131


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2961_296166

/-- An isosceles triangle with sides a, b, and c, where at least two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : (a = b ∧ a > 0) ∨ (b = c ∧ b > 0) ∨ (a = c ∧ a > 0)

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: An isosceles triangle with one side of length 6 and another of length 5 
    has a perimeter of either 16 or 17 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 6 ∧ (t.b = 5 ∨ t.c = 5)) ∨ (t.b = 6 ∧ (t.a = 5 ∨ t.c = 5)) ∨ (t.c = 6 ∧ (t.a = 5 ∨ t.b = 5))) →
  (perimeter t = 16 ∨ perimeter t = 17) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2961_296166


namespace NUMINAMATH_CALUDE_x_gt_4_sufficient_not_necessary_for_inequality_l2961_296142

theorem x_gt_4_sufficient_not_necessary_for_inequality :
  (∀ x : ℝ, x > 4 → x^2 - 4*x > 0) ∧
  (∃ x : ℝ, x^2 - 4*x > 0 ∧ ¬(x > 4)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_4_sufficient_not_necessary_for_inequality_l2961_296142


namespace NUMINAMATH_CALUDE_nested_square_root_fourth_power_l2961_296134

theorem nested_square_root_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1))))^4 = 2 + 2 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_fourth_power_l2961_296134


namespace NUMINAMATH_CALUDE_smallest_added_number_l2961_296181

theorem smallest_added_number (n : ℤ) (x : ℕ) 
  (h1 : n % 25 = 4)
  (h2 : (n + x) % 5 = 4)
  (h3 : x > 0) :
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_added_number_l2961_296181


namespace NUMINAMATH_CALUDE_cards_remaining_l2961_296132

def initial_cards : Nat := 13
def cards_given_away : Nat := 9

theorem cards_remaining (initial : Nat) (given_away : Nat) : 
  initial = initial_cards → given_away = cards_given_away → 
  initial - given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_remaining_l2961_296132


namespace NUMINAMATH_CALUDE_crayons_per_child_l2961_296186

/-- Given that there are 6 children and a total of 18 crayons,
    prove that each child has 3 crayons. -/
theorem crayons_per_child :
  ∀ (total_crayons : ℕ) (num_children : ℕ),
    total_crayons = 18 →
    num_children = 6 →
    total_crayons / num_children = 3 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_child_l2961_296186


namespace NUMINAMATH_CALUDE_value_of_x_l2961_296180

theorem value_of_x : (2011^3 - 2011^2) / 2011 = 2011 * 2010 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2961_296180


namespace NUMINAMATH_CALUDE_phone_number_probability_l2961_296116

/-- The set of possible area codes -/
def areaCodes : Finset ℕ := {407, 410, 415}

/-- The set of digits for the remaining part of the phone number -/
def remainingDigits : Finset ℕ := {0, 1, 2, 3, 4}

/-- The total number of digits in the phone number -/
def totalDigits : ℕ := 8

/-- The number of digits after the area code -/
def remainingDigitsCount : ℕ := 5

theorem phone_number_probability :
  (1 : ℚ) / (areaCodes.card * Nat.factorial remainingDigitsCount) =
  (1 : ℚ) / 360 := by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l2961_296116


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2961_296167

/-- Approximate number of fish in a pond given tagging and recapture data -/
theorem fish_population_estimate (tagged_initial : ℕ) (second_catch : ℕ) (tagged_second : ℕ) :
  tagged_initial = 70 →
  second_catch = 50 →
  tagged_second = 2 →
  (tagged_second : ℚ) / second_catch = tagged_initial / (tagged_initial + 1680) :=
by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l2961_296167


namespace NUMINAMATH_CALUDE_tan_2alpha_values_l2961_296191

theorem tan_2alpha_values (α : ℝ) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4/3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_values_l2961_296191


namespace NUMINAMATH_CALUDE_sale_price_determination_l2961_296114

/-- Proves that the sale price of each machine is $10,000 given the commission structure and total commission --/
theorem sale_price_determination (commission_rate_first_100 : ℝ) (commission_rate_after_100 : ℝ) 
  (total_machines : ℕ) (machines_at_first_rate : ℕ) (total_commission : ℝ) :
  commission_rate_first_100 = 0.03 →
  commission_rate_after_100 = 0.04 →
  total_machines = 130 →
  machines_at_first_rate = 100 →
  total_commission = 42000 →
  ∃ (sale_price : ℝ), 
    sale_price = 10000 ∧
    (machines_at_first_rate : ℝ) * commission_rate_first_100 * sale_price + 
    ((total_machines - machines_at_first_rate) : ℝ) * commission_rate_after_100 * sale_price = 
    total_commission :=
by
  sorry

#check sale_price_determination

end NUMINAMATH_CALUDE_sale_price_determination_l2961_296114


namespace NUMINAMATH_CALUDE_canoe_oar_probability_l2961_296161

theorem canoe_oar_probability (p : ℝ) :
  p ≥ 0 ∧ p ≤ 1 →
  2 * p - p^2 = 0.84 →
  p = 0.6 := by
sorry

end NUMINAMATH_CALUDE_canoe_oar_probability_l2961_296161


namespace NUMINAMATH_CALUDE_not_A_implies_not_all_right_l2961_296188

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (got_all_right : Student → Prop)
variable (received_A : Student → Prop)

-- State Ms. Carroll's promise
variable (carroll_promise : ∀ s : Student, got_all_right s → received_A s)

-- Theorem to prove
theorem not_A_implies_not_all_right :
  ∀ s : Student, ¬(received_A s) → ¬(got_all_right s) :=
sorry

end NUMINAMATH_CALUDE_not_A_implies_not_all_right_l2961_296188


namespace NUMINAMATH_CALUDE_divisor_of_99_l2961_296151

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisor_of_99 (k : ℕ) 
  (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : 
  k ∣ 99 := by sorry

end NUMINAMATH_CALUDE_divisor_of_99_l2961_296151


namespace NUMINAMATH_CALUDE_function_composition_equality_l2961_296141

/-- Given functions f, g, and h, proves that A = 3B / (1 + C) -/
theorem function_composition_equality (A B C : ℝ) : 
  let f := fun x => A * x - 3 * B^2
  let g := fun x => B * x
  let h := fun x => x + C
  f (g (h 1)) = 0 → A = 3 * B / (1 + C) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2961_296141


namespace NUMINAMATH_CALUDE_parabola_ellipse_tangent_property_l2961_296165

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola E
def parabola (x y : ℝ) : Prop := y^2 = 6*x + 15

-- Define the focus F
def F : ℝ × ℝ := (-1, 0)

-- Define a point on the parabola
def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

-- Define tangent points on the ellipse
def tangent_points (M N : ℝ × ℝ) : Prop := ellipse M.1 M.2 ∧ ellipse N.1 N.2

-- Theorem statement
theorem parabola_ellipse_tangent_property
  (A M N : ℝ × ℝ)
  (h_A : on_parabola A)
  (h_MN : tangent_points M N) :
  (∃ (t : ℝ), F.1 + t * (A.1 - F.1) = (M.1 + N.1) / 2 ∧
              F.2 + t * (A.2 - F.2) = (M.2 + N.2) / 2) ∧
  (∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧
    Real.cos θ = ((M.1 + 1) * (N.1 + 1) + M.2 * N.2) /
                 (Real.sqrt ((M.1 + 1)^2 + M.2^2) * Real.sqrt ((N.1 + 1)^2 + N.2^2))) :=
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_tangent_property_l2961_296165


namespace NUMINAMATH_CALUDE_initial_dogs_count_l2961_296158

/-- Proves that the initial number of dogs in a pet center is 36, given the conditions of the problem. -/
theorem initial_dogs_count (initial_cats : ℕ) (adopted_dogs : ℕ) (added_cats : ℕ) (final_total : ℕ) 
  (h1 : initial_cats = 29)
  (h2 : adopted_dogs = 20)
  (h3 : added_cats = 12)
  (h4 : final_total = 57)
  (h5 : final_total = initial_cats + added_cats + (initial_dogs - adopted_dogs)) :
  initial_dogs = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_dogs_count_l2961_296158


namespace NUMINAMATH_CALUDE_intersection_minimizes_sum_of_distances_l2961_296123

/-- Given a triangle ABC, construct equilateral triangles ABC₁, ACB₁, and BCA₁ externally --/
def constructExternalTriangles (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Compute the intersection point of lines AA₁, BB₁, and CC₁ --/
def intersectionPoint (A B C : ℝ × ℝ) (A₁ B₁ C₁ : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Compute the sum of distances from a point to the vertices of a triangle --/
def sumOfDistances (P A B C : ℝ × ℝ) : ℝ := sorry

/-- The main theorem: The intersection point minimizes the sum of distances --/
theorem intersection_minimizes_sum_of_distances (A B C : ℝ × ℝ) :
  let (A₁, B₁, C₁) := constructExternalTriangles A B C
  let O := intersectionPoint A B C A₁ B₁ C₁
  ∀ P : ℝ × ℝ, sumOfDistances O A B C ≤ sumOfDistances P A B C :=
sorry

end NUMINAMATH_CALUDE_intersection_minimizes_sum_of_distances_l2961_296123


namespace NUMINAMATH_CALUDE_toast_costs_one_pound_l2961_296182

/-- The cost of a slice of toast -/
def toast_cost : ℝ := sorry

/-- The cost of an egg -/
def egg_cost : ℝ := 3

/-- Dale's breakfast cost -/
def dale_breakfast : ℝ := 2 * toast_cost + 2 * egg_cost

/-- Andrew's breakfast cost -/
def andrew_breakfast : ℝ := toast_cost + 2 * egg_cost

/-- The total cost of both breakfasts -/
def total_cost : ℝ := 15

theorem toast_costs_one_pound :
  dale_breakfast + andrew_breakfast = total_cost →
  toast_cost = 1 := by sorry

end NUMINAMATH_CALUDE_toast_costs_one_pound_l2961_296182


namespace NUMINAMATH_CALUDE_no_rin_is_bin_l2961_296174

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Bin, Fin, and Rin
variable (Bin Fin Rin : U → Prop)

-- Premise I: All Bins are Fins
axiom all_bins_are_fins : ∀ x, Bin x → Fin x

-- Premise II: Some Rins are not Fins
axiom some_rins_not_fins : ∃ x, Rin x ∧ ¬Fin x

-- Theorem to prove
theorem no_rin_is_bin : (∀ x, Bin x → Fin x) → (∃ x, Rin x ∧ ¬Fin x) → (∀ x, Rin x → ¬Bin x) :=
sorry

end NUMINAMATH_CALUDE_no_rin_is_bin_l2961_296174


namespace NUMINAMATH_CALUDE_number_equality_l2961_296104

theorem number_equality (x : ℝ) : 9^6 = x^12 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2961_296104


namespace NUMINAMATH_CALUDE_fraction_equality_l2961_296157

theorem fraction_equality (a b : ℚ) (h : a / 5 = b / 3) : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2961_296157


namespace NUMINAMATH_CALUDE_lucy_fish_count_l2961_296149

/-- The number of fish Lucy needs to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := 280

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := total_fish - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l2961_296149


namespace NUMINAMATH_CALUDE_rose_jasmine_distance_l2961_296129

/-- Represents the positions of trees and flowers on a straight line -/
structure ForestLine where
  -- Positions of trees
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  -- Ensure trees are in order
  ab_pos : a < b
  bc_pos : b < c
  cd_pos : c < d
  de_pos : d < e
  -- Total distance between A and E is 28
  ae_dist : e - a = 28
  -- Positions of flowers
  daisy : ℝ
  rose : ℝ
  jasmine : ℝ
  carnation : ℝ
  -- Flowers at midpoints
  daisy_mid : daisy = (a + b) / 2
  rose_mid : rose = (b + c) / 2
  jasmine_mid : jasmine = (c + d) / 2
  carnation_mid : carnation = (d + e) / 2
  -- Distance between daisy and carnation is 20
  daisy_carnation_dist : carnation - daisy = 20

/-- The distance between the rose bush and the jasmine is 6 meters -/
theorem rose_jasmine_distance (f : ForestLine) : f.jasmine - f.rose = 6 := by
  sorry

end NUMINAMATH_CALUDE_rose_jasmine_distance_l2961_296129


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2961_296170

theorem trigonometric_inequality (a b : ℝ) : 
  (∀ x : ℝ, |a * Real.sin x + b * Real.sin (2 * x)| ≤ 1) →
  ((a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨
   (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3))) :=
by sorry


end NUMINAMATH_CALUDE_trigonometric_inequality_l2961_296170


namespace NUMINAMATH_CALUDE_arrangement_count_l2961_296192

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of teachers in each group -/
def teachers_per_group : ℕ := 1

/-- The number of students in each group -/
def students_per_group : ℕ := 2

/-- The number of groups -/
def num_groups : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := 12

theorem arrangement_count :
  (Nat.choose num_teachers teachers_per_group) *
  (Nat.choose num_students students_per_group) = total_arrangements :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l2961_296192


namespace NUMINAMATH_CALUDE_product_of_large_numbers_l2961_296185

theorem product_of_large_numbers : (4 * 10^6) * (8 * 10^6) = 3.2 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_product_of_large_numbers_l2961_296185


namespace NUMINAMATH_CALUDE_polynomial_property_implies_P0_values_l2961_296172

/-- A polynomial P with real coefficients satisfying the given property -/
def SatisfiesProperty (P : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (|y^2 - P x| ≤ 2 * |x|) ↔ (|x^2 - P y| ≤ 2 * |y|)

/-- The theorem stating the possible values of P(0) -/
theorem polynomial_property_implies_P0_values (P : ℝ → ℝ) (h : SatisfiesProperty P) :
  P 0 < 0 ∨ P 0 = 1 :=
sorry

end NUMINAMATH_CALUDE_polynomial_property_implies_P0_values_l2961_296172


namespace NUMINAMATH_CALUDE_power_two_divides_power_odd_minus_one_l2961_296138

theorem power_two_divides_power_odd_minus_one (k n : ℕ) (h_k_odd : Odd k) (h_n_ge_one : n ≥ 1) :
  ∃ m : ℤ, k^(2^n) - 1 = 2^(n+2) * m :=
sorry

end NUMINAMATH_CALUDE_power_two_divides_power_odd_minus_one_l2961_296138


namespace NUMINAMATH_CALUDE_square_root_equals_arithmetic_square_root_l2961_296101

theorem square_root_equals_arithmetic_square_root (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x ∧ y = Real.sqrt x) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_square_root_equals_arithmetic_square_root_l2961_296101


namespace NUMINAMATH_CALUDE_boxes_per_hand_for_ten_people_l2961_296175

/-- Given a group of people and the total number of boxes they can hold,
    calculate the number of boxes a single person can hold in each hand. -/
def boxes_per_hand (group_size : ℕ) (total_boxes : ℕ) : ℕ :=
  (total_boxes / group_size) / 2

/-- Theorem stating that for a group of 10 people holding 20 boxes in total,
    each person can hold 1 box in each hand. -/
theorem boxes_per_hand_for_ten_people :
  boxes_per_hand 10 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_hand_for_ten_people_l2961_296175


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2961_296100

theorem sin_2alpha_value (α : ℝ) (h : Real.sin (α - π/4) = -Real.cos (2*α)) :
  Real.sin (2*α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2961_296100


namespace NUMINAMATH_CALUDE_fraction_defined_iff_not_five_l2961_296107

theorem fraction_defined_iff_not_five (x : ℝ) : IsRegular (x - 5)⁻¹ ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_fraction_defined_iff_not_five_l2961_296107


namespace NUMINAMATH_CALUDE_triangle_problem_l2961_296155

theorem triangle_problem (a b c A B C : ℝ) (h1 : 2 * a * Real.cos B + b = 2 * c)
  (h2 : a = 2 * Real.sqrt 3) (h3 : (1 / 2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π / 3 ∧ Real.sin B + Real.sin C = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2961_296155


namespace NUMINAMATH_CALUDE_balloons_in_park_l2961_296105

/-- The number of balloons Allan and Jake have in the park -/
def total_balloons (allan_initial : ℕ) (jake : ℕ) (allan_bought : ℕ) : ℕ :=
  (allan_initial + allan_bought) + jake

/-- Theorem: Allan and Jake have 10 balloons in total -/
theorem balloons_in_park :
  total_balloons 3 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l2961_296105


namespace NUMINAMATH_CALUDE_remi_water_spill_l2961_296187

/-- Represents the amount of water Remi spilled the first time -/
def first_spill : ℕ := sorry

/-- The capacity of Remi's water bottle in ounces -/
def bottle_capacity : ℕ := 20

/-- The number of times Remi refills his bottle per day -/
def refills_per_day : ℕ := 3

/-- The number of days Remi drinks water -/
def days : ℕ := 7

/-- The amount of water Remi spilled the second time -/
def second_spill : ℕ := 8

/-- The total amount of water Remi actually drank in ounces -/
def total_drunk : ℕ := 407

theorem remi_water_spill :
  first_spill = 5 ∧
  bottle_capacity * refills_per_day * days - first_spill - second_spill = total_drunk :=
by sorry

end NUMINAMATH_CALUDE_remi_water_spill_l2961_296187


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l2961_296194

/-- Calculates the total cost of a shopping trip, including discounts and sales tax -/
theorem shopping_cost_calculation 
  (tshirt_price sweater_price jacket_price : ℚ)
  (jacket_discount sales_tax : ℚ)
  (tshirt_quantity sweater_quantity jacket_quantity : ℕ)
  (h1 : tshirt_price = 8)
  (h2 : sweater_price = 18)
  (h3 : jacket_price = 80)
  (h4 : jacket_discount = 1/10)
  (h5 : sales_tax = 1/20)
  (h6 : tshirt_quantity = 6)
  (h7 : sweater_quantity = 4)
  (h8 : jacket_quantity = 5) :
  let subtotal := tshirt_price * tshirt_quantity + 
                  sweater_price * sweater_quantity + 
                  jacket_price * jacket_quantity * (1 - jacket_discount)
  let total_with_tax := subtotal * (1 + sales_tax)
  total_with_tax = 504 := by sorry


end NUMINAMATH_CALUDE_shopping_cost_calculation_l2961_296194


namespace NUMINAMATH_CALUDE_tuna_salmon_ratio_l2961_296110

/-- Proves that the ratio of tuna weight to salmon weight is 2:1 given specific conditions --/
theorem tuna_salmon_ratio (trout_weight salmon_weight tuna_weight : ℝ) : 
  trout_weight = 200 →
  salmon_weight = trout_weight * 1.5 →
  trout_weight + salmon_weight + tuna_weight = 1100 →
  tuna_weight / salmon_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuna_salmon_ratio_l2961_296110


namespace NUMINAMATH_CALUDE_roots_equation_l2961_296179

theorem roots_equation (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → 
  β^2 - 3*β + 1 = 0 → 
  7 * α^5 + 8 * β^4 = 1448 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l2961_296179


namespace NUMINAMATH_CALUDE_product_of_sums_l2961_296163

theorem product_of_sums (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b + a + b = 35) (hbc : b * c + b + c = 35) (hca : c * a + c + a = 35) :
  (a + 1) * (b + 1) * (c + 1) = 216 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l2961_296163


namespace NUMINAMATH_CALUDE_sum_of_roots_l2961_296128

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = -13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2961_296128


namespace NUMINAMATH_CALUDE_lines_parallel_if_one_in_plane_one_parallel_to_plane_l2961_296125

-- Define the plane and lines
variable (α : Plane) (m n : Line)

-- Define the property of lines being coplanar
def coplanar (l₁ l₂ : Line) : Prop := sorry

-- Define the property of a line being contained in a plane
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- Define the property of a line being parallel to a plane
def parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the property of two lines being parallel
def parallel (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem lines_parallel_if_one_in_plane_one_parallel_to_plane
  (h_coplanar : coplanar m n)
  (h_m_in_α : contained_in m α)
  (h_n_parallel_α : parallel_to_plane n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_if_one_in_plane_one_parallel_to_plane_l2961_296125


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l2961_296196

/-- Proves that adding 20 liters of chemical x to 80 liters of a mixture that is 30% chemical x
    results in a new mixture that is 44% chemical x. -/
theorem chemical_mixture_problem :
  let initial_volume : ℝ := 80
  let initial_concentration : ℝ := 0.30
  let added_volume : ℝ := 20
  let final_concentration : ℝ := 0.44
  (initial_volume * initial_concentration + added_volume) / (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l2961_296196


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l2961_296152

/-- The range of a, given the conditions in the problem -/
def range_of_a : Set ℝ :=
  {a | a ≤ -2 ∨ a = 1}

/-- Proposition p: For all x in [1,2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

/-- Proposition q: There exists x₀ ∈ ℝ such that x₀^2 + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The main theorem stating that given the conditions, the range of a is as defined -/
theorem range_of_a_theorem (a : ℝ) :
  (prop_p a ∧ prop_q a) → a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l2961_296152


namespace NUMINAMATH_CALUDE_derivative_at_three_l2961_296121

/-- Given a function f(x) = -x^2 + 10, prove that its derivative at x = 3 is -3. -/
theorem derivative_at_three (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + 10) :
  deriv f 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_three_l2961_296121


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_transformation_l2961_296130

/-- Given a point with rectangular coordinates (3, -4, 2) and spherical coordinates (ρ, θ, φ),
    the point with spherical coordinates (ρ, θ + π, φ) has rectangular coordinates (-3, 4, 2). -/
theorem spherical_to_rectangular_transformation (ρ θ φ : Real) :
  (ρ * Real.sin φ * Real.cos θ = 3 ∧
   ρ * Real.sin φ * Real.sin θ = -4 ∧
   ρ * Real.cos φ = 2) →
  (ρ * Real.sin φ * Real.cos (θ + Real.pi) = -3 ∧
   ρ * Real.sin φ * Real.sin (θ + Real.pi) = 4 ∧
   ρ * Real.cos φ = 2) :=
by sorry


end NUMINAMATH_CALUDE_spherical_to_rectangular_transformation_l2961_296130


namespace NUMINAMATH_CALUDE_neighborhood_total_l2961_296168

/-- Represents the number of households in different categories -/
structure Neighborhood where
  neither : ℕ
  both : ℕ
  car : ℕ
  bikeOnly : ℕ

/-- Calculates the total number of households in the neighborhood -/
def totalHouseholds (n : Neighborhood) : ℕ :=
  n.neither + (n.car - n.both) + n.bikeOnly + n.both

/-- Theorem stating that the total number of households is 90 -/
theorem neighborhood_total (n : Neighborhood) 
  (h1 : n.neither = 11)
  (h2 : n.both = 16)
  (h3 : n.car = 44)
  (h4 : n.bikeOnly = 35) :
  totalHouseholds n = 90 := by
  sorry

#eval totalHouseholds { neither := 11, both := 16, car := 44, bikeOnly := 35 }

end NUMINAMATH_CALUDE_neighborhood_total_l2961_296168


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l2961_296111

/-- The expected number of boy-girl adjacencies in a circular arrangement -/
theorem expected_boy_girl_adjacencies (n_boys n_girls : ℕ) (h : n_boys = 10 ∧ n_girls = 8) :
  let total := n_boys + n_girls
  let prob_boy_girl := (n_boys : ℚ) * n_girls / (total * (total - 1))
  total * (2 * prob_boy_girl) = 480 / 51 := by
  sorry

#check expected_boy_girl_adjacencies

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l2961_296111


namespace NUMINAMATH_CALUDE_arithmetic_progression_condition_l2961_296146

theorem arithmetic_progression_condition 
  (a b c : ℝ) (p n k : ℕ+) : 
  (∃ (d : ℝ) (a₁ : ℝ), a = a₁ + (p - 1) * d ∧ b = a₁ + (n - 1) * d ∧ c = a₁ + (k - 1) * d) ↔ 
  (a * (n - k) + b * (k - p) + c * (p - n) = 0) ∧ 
  ((b - a) / (c - b) = (n - p : ℝ) / (k - n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_condition_l2961_296146


namespace NUMINAMATH_CALUDE_min_value_theorem_l2961_296148

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3) :
  (∀ x y z, x > 0 → y > 0 → z > 0 → x * (x + y + z) + y * z = 4 + 2 * Real.sqrt 3 →
    2 * x + y + z ≥ 2 * Real.sqrt 3 + 2) ∧
  (∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (x + y + z) + y * z = 4 + 2 * Real.sqrt 3 ∧
    2 * x + y + z = 2 * Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2961_296148


namespace NUMINAMATH_CALUDE_amy_jeremy_age_ratio_l2961_296135

/-- Proves that the ratio of Amy's age to Jeremy's age is 1:3 given the specified conditions -/
theorem amy_jeremy_age_ratio :
  ∀ (amy_age jeremy_age chris_age : ℕ),
    jeremy_age = 66 →
    amy_age + jeremy_age + chris_age = 132 →
    chris_age = 2 * amy_age →
    (amy_age : ℚ) / jeremy_age = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_jeremy_age_ratio_l2961_296135


namespace NUMINAMATH_CALUDE_find_n_l2961_296117

def A (i : ℕ) : ℕ := 2 * i - 1

def B (n i : ℕ) : ℕ := n - 2 * (i - 1)

theorem find_n : ∃ n : ℕ, 
  (∃ k : ℕ, A k = 19 ∧ B n k = 89) → n = 107 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2961_296117


namespace NUMINAMATH_CALUDE_symmetry_coincidence_l2961_296118

-- Define the type for points in the plane
def Point : Type := ℝ × ℝ

-- Define the symmetry operation
def symmetric (A B O : Point) : Prop := 
  ∃ (x y : ℝ), A = (x, y) ∧ B = (2 * O.1 - x, 2 * O.2 - y)

-- Define the given points
variable (A A₁ A₂ A₃ A₄ A₅ A₆ O₁ O₂ O₃ : Point)

-- State the theorem
theorem symmetry_coincidence 
  (h1 : symmetric A A₁ O₁)
  (h2 : symmetric A₁ A₂ O₂)
  (h3 : symmetric A₂ A₃ O₃)
  (h4 : symmetric A₃ A₄ O₁)
  (h5 : symmetric A₄ A₅ O₂)
  (h6 : symmetric A₅ A₆ O₃) :
  A = A₆ := by sorry

end NUMINAMATH_CALUDE_symmetry_coincidence_l2961_296118


namespace NUMINAMATH_CALUDE_min_box_value_l2961_296133

theorem min_box_value (a b box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 45 * x^2 + box * x + 45) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  (∃ box_min : ℤ, box_min = 106 ∧ box ≥ box_min ∧
    (∀ a' b' box' : ℤ, 
      (∀ x, (a' * x + b') * (b' * x + a') = 45 * x^2 + box' * x + 45) →
      a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' →
      box' ≥ box_min)) :=
sorry

end NUMINAMATH_CALUDE_min_box_value_l2961_296133


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_achievable_l2961_296120

theorem max_value_of_function (x : ℝ) (hx : x > 0) : 
  (-2 * x^2 + x - 3) / x ≤ 1 - 2 * Real.sqrt 6 := by sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x > 0 ∧ (-2 * x^2 + x - 3) / x = 1 - 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_achievable_l2961_296120


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2961_296126

/-- Simple interest rate calculation -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 800)
  (h2 : final_amount = 950)
  (h3 : time = 5)
  : (final_amount - principal) * 100 / (principal * time) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2961_296126


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_right_triangle_l2961_296113

theorem quadratic_roots_imply_right_triangle 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (hroots : ∃ x : ℝ, x^2 - (a + b + c)*x + (a*b + b*c + c*a) = 0 ∧ 
    ∀ y : ℝ, y^2 - (a + b + c)*y + (a*b + b*c + c*a) = 0 → y = x) :
  ∃ p q r : ℝ, p^4 = a ∧ q^4 = b ∧ r^4 = c ∧ p^2 = q^2 + r^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_right_triangle_l2961_296113
