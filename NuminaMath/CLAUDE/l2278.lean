import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l2278_227893

def f (x : ℝ) : ℝ := 4*x^5 - 9*x^4 + 3*x^3 + 5*x^2 - x - 15

theorem remainder_theorem :
  f 4 = 2045 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2278_227893


namespace NUMINAMATH_CALUDE_min_framing_for_picture_l2278_227828

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- Theorem stating the minimum framing needed for the given picture specifications. -/
theorem min_framing_for_picture :
  min_framing_feet 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_picture_l2278_227828


namespace NUMINAMATH_CALUDE_percent_relation_l2278_227873

theorem percent_relation (x y : ℝ) (h : 0.6 * (x - y) = 0.3 * (x + y)) : y = (1/3) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2278_227873


namespace NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l2278_227838

/-- Alice's schedule cycle length -/
def alice_cycle : ℕ := 6

/-- Bob's schedule cycle length -/
def bob_cycle : ℕ := 6

/-- Number of days Alice works in her cycle -/
def alice_work_days : ℕ := 4

/-- Number of days Bob works in his cycle -/
def bob_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 800

/-- Function to calculate the number of coinciding rest days -/
def coinciding_rest_days : ℕ := 
  (total_days / alice_cycle) * (alice_cycle - alice_work_days - bob_work_days + 1)

theorem coinciding_rest_days_theorem : 
  coinciding_rest_days = 133 := by sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l2278_227838


namespace NUMINAMATH_CALUDE_not_perfect_square_8p_plus_1_l2278_227889

theorem not_perfect_square_8p_plus_1 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ¬ ∃ n : ℕ, 8 * p + 1 = (2 * n + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_8p_plus_1_l2278_227889


namespace NUMINAMATH_CALUDE_sum_squared_l2278_227882

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y)^2 = 90 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l2278_227882


namespace NUMINAMATH_CALUDE_inventory_depletion_l2278_227874

def inventory_model (x : ℝ) : ℝ := -3 * x^3 + 12 * x + 8

theorem inventory_depletion :
  ∃ (ε : ℝ), ε > 0 ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → inventory_model x ≥ 0) ∧
  (∀ (x : ℝ), 2 < x ∧ x < 2 + ε → inventory_model x < 0) :=
sorry

end NUMINAMATH_CALUDE_inventory_depletion_l2278_227874


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_optimal_price_satisfies_conditions_l2278_227801

/-- Represents the daily profit function for a merchant's goods -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- Represents the optimal selling price that maximizes daily profit -/
def optimal_price : ℝ := 14

theorem optimal_price_maximizes_profit :
  ∀ (x : ℝ), x ≠ optimal_price → profit_function x < profit_function optimal_price :=
by sorry

/-- Verifies that the optimal price satisfies the given conditions -/
theorem optimal_price_satisfies_conditions :
  let initial_price : ℝ := 10
  let initial_sales : ℝ := 100
  let cost_per_item : ℝ := 8
  let price_increase : ℝ := optimal_price - initial_price
  let sales_decrease : ℝ := 10 * price_increase
  (initial_sales - sales_decrease) * (optimal_price - cost_per_item) = profit_function optimal_price :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_optimal_price_satisfies_conditions_l2278_227801


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2278_227840

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2278_227840


namespace NUMINAMATH_CALUDE_expression_evaluation_l2278_227845

theorem expression_evaluation : (16 : ℝ) * 0.5 - (4.5 - 0.125 * 8) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2278_227845


namespace NUMINAMATH_CALUDE_new_student_weight_l2278_227860

/-- Given a group of 10 students, proves that replacing a 120 kg student with a new student
    that causes the average weight to decrease by 6 kg results in the new student weighing 60 kg. -/
theorem new_student_weight
  (n : ℕ) -- number of students
  (old_avg : ℝ) -- original average weight
  (replaced_weight : ℝ) -- weight of the replaced student
  (new_avg : ℝ) -- new average weight after replacement
  (h1 : n = 10) -- there are 10 students
  (h2 : new_avg = old_avg - 6) -- average weight decreases by 6 kg
  (h3 : replaced_weight = 120) -- replaced student weighs 120 kg
  : n * new_avg + 60 = n * old_avg - replaced_weight := by
  sorry

#check new_student_weight

end NUMINAMATH_CALUDE_new_student_weight_l2278_227860


namespace NUMINAMATH_CALUDE_max_profit_at_85_optimal_selling_price_l2278_227850

/-- Represents the profit function for the item sales --/
def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x) - 500

/-- Theorem stating that the maximum profit is achieved at a selling price of 85 yuan --/
theorem max_profit_at_85 :
  ∃ (x : ℝ), x > 0 ∧ x < 20 ∧
  ∀ (y : ℝ), y > 0 → y < 20 → profit x ≥ profit y ∧
  x + 80 = 85 := by
  sorry

/-- Corollary: The selling price that maximizes profit is 85 yuan --/
theorem optimal_selling_price : 
  ∃ (x : ℝ), x > 0 ∧ x < 20 ∧
  ∀ (y : ℝ), y > 0 → y < 20 → profit x ≥ profit y ∧
  x + 80 = 85 := by
  exact max_profit_at_85

end NUMINAMATH_CALUDE_max_profit_at_85_optimal_selling_price_l2278_227850


namespace NUMINAMATH_CALUDE_most_stable_athlete_l2278_227883

def athlete_variance (a b c d : ℝ) : Prop :=
  a = 0.5 ∧ b = 0.5 ∧ c = 0.6 ∧ d = 0.4

theorem most_stable_athlete (a b c d : ℝ) 
  (h : athlete_variance a b c d) : 
  d < a ∧ d < b ∧ d < c :=
by
  sorry

#check most_stable_athlete

end NUMINAMATH_CALUDE_most_stable_athlete_l2278_227883


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2278_227865

theorem hyperbola_equation (x y : ℝ) :
  (∃ (f : ℝ × ℝ), (f.1^2 / 16 - f.2^2 / 4 = 1) ∧
   ((x^2 / 15 - y^2 / 5 = 1) → (f = (x, y) ∨ f = (-x, y)))) →
  (x^2 / 15 - y^2 / 5 = 1) →
  ((3 * Real.sqrt 2)^2 / 15 - 2^2 / 5 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2278_227865


namespace NUMINAMATH_CALUDE_inverse_proposition_l2278_227869

theorem inverse_proposition : 
  (∀ a b : ℝ, a^2 + b^2 ≠ 0 → a = 0 ∧ b = 0) ↔ 
  (∀ a b : ℝ, a = 0 ∧ b = 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2278_227869


namespace NUMINAMATH_CALUDE_f_of_f_3_l2278_227829

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem f_of_f_3 : f (f 3) = 1429 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_3_l2278_227829


namespace NUMINAMATH_CALUDE_sequence_minimum_l2278_227859

theorem sequence_minimum (n : ℤ) : ∃ (m : ℤ), ∀ (n : ℤ), n^2 - 8*n + 5 ≥ m ∧ ∃ (k : ℤ), k^2 - 8*k + 5 = m :=
sorry

end NUMINAMATH_CALUDE_sequence_minimum_l2278_227859


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2278_227805

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2278_227805


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2278_227852

def alphabet_size : ℕ := 25  -- Excluding 'A'
def letter_positions : ℕ := 4
def digit_positions : ℕ := 2
def total_digits : ℕ := 10

-- Define the function to calculate the number of license plate combinations
def license_plate_combinations : ℕ :=
  (alphabet_size.choose 2) *  -- Choose 2 letters from 25
  (letter_positions.choose 2) *  -- Choose 2 positions for one letter
  (total_digits) *  -- Choose first digit
  (total_digits - 1)  -- Choose second digit

-- Theorem statement
theorem license_plate_theorem :
  license_plate_combinations = 162000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2278_227852


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l2278_227896

theorem lcm_of_ratio_and_sum (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 → a + b = 30 → Nat.lcm a b = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l2278_227896


namespace NUMINAMATH_CALUDE_monroe_collection_legs_l2278_227847

/-- Represents the number of legs for each type of creature -/
structure CreatureLegs where
  ant : Nat
  spider : Nat
  beetle : Nat
  centipede : Nat

/-- Represents the count of each type of creature in the collection -/
structure CreatureCount where
  ants : Nat
  spiders : Nat
  beetles : Nat
  centipedes : Nat

/-- Calculates the total number of legs in the collection -/
def totalLegs (legs : CreatureLegs) (count : CreatureCount) : Nat :=
  legs.ant * count.ants + 
  legs.spider * count.spiders + 
  legs.beetle * count.beetles + 
  legs.centipede * count.centipedes

/-- Theorem: The total number of legs in Monroe's collection is 726 -/
theorem monroe_collection_legs : 
  let legs : CreatureLegs := { ant := 6, spider := 8, beetle := 6, centipede := 100 }
  let count : CreatureCount := { ants := 12, spiders := 8, beetles := 15, centipedes := 5 }
  totalLegs legs count = 726 := by
  sorry

end NUMINAMATH_CALUDE_monroe_collection_legs_l2278_227847


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2278_227887

/-- A quadratic function f(x) = x^2 - 3x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + m + 2

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (-3)^2 - 4*(1)*(m+2)

theorem quadratic_one_root (m : ℝ) : 
  (∃! x, f m x = 0) → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2278_227887


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2278_227858

theorem smallest_solution_of_equation (x : ℝ) : 
  (x = (7 - Real.sqrt 33) / 2) ↔ 
  (x < (7 + Real.sqrt 33) / 2 ∧ 1 / (x - 1) + 1 / (x - 5) = 4 / (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2278_227858


namespace NUMINAMATH_CALUDE_ceiling_square_fraction_plus_eighth_l2278_227856

theorem ceiling_square_fraction_plus_eighth : ⌈(-7/4)^2 + 1/8⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_square_fraction_plus_eighth_l2278_227856


namespace NUMINAMATH_CALUDE_hyperbola_and_line_l2278_227807

/-- Hyperbola with center at origin, right focus at (2,0), and distance 1 from focus to asymptote -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  right_focus : ℝ × ℝ := (2, 0)
  focus_to_asymptote : ℝ := 1

/-- Line that intersects the hyperbola at two distinct points -/
structure IntersectingLine where
  k : ℝ
  b : ℝ := 2

/-- Theorem about the hyperbola and its intersecting line -/
theorem hyperbola_and_line (C : Hyperbola) (l : IntersectingLine) :
  (∀ A B : ℝ × ℝ, A ≠ B → (A.1^2/3 - A.2^2 = 1 ∧ A.2 = l.k * A.1 + l.b) →
                        (B.1^2/3 - B.2^2 = 1 ∧ B.2 = l.k * B.1 + l.b) →
                        A.1 * B.1 + A.2 * B.2 > 2) →
  (∀ x y : ℝ, x^2/3 - y^2 = 1 ↔ C.center = (0, 0) ∧ C.right_focus = (2, 0) ∧ C.focus_to_asymptote = 1) ∧
  (l.k ∈ Set.Ioo (-Real.sqrt 15 / 3) (-Real.sqrt 3 / 3) ∪ Set.Ioo (Real.sqrt 3 / 3) (Real.sqrt 15 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_l2278_227807


namespace NUMINAMATH_CALUDE_hours_to_seconds_l2278_227895

-- Define the conversion factors
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the problem
def hours : ℚ := 3.5

-- Theorem to prove
theorem hours_to_seconds : 
  (hours * minutes_per_hour * seconds_per_minute : ℚ) = 12600 := by
  sorry

end NUMINAMATH_CALUDE_hours_to_seconds_l2278_227895


namespace NUMINAMATH_CALUDE_blue_marbles_count_l2278_227880

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The total number of blue marbles Jason and Tom have together -/
def total_blue_marbles : ℕ := jason_blue_marbles + tom_blue_marbles

theorem blue_marbles_count : total_blue_marbles = 68 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l2278_227880


namespace NUMINAMATH_CALUDE_wrong_value_correction_l2278_227844

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 20)
  (h2 : initial_mean = 150)
  (h3 : correct_mean = 151.25)
  (h4 : correct_value = 160) :
  ∃ x : ℝ, n * initial_mean - x + correct_value = n * correct_mean ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_wrong_value_correction_l2278_227844


namespace NUMINAMATH_CALUDE_phone_production_ratio_l2278_227832

/-- Proves that the ratio of this year's production to last year's production is 2:1 --/
theorem phone_production_ratio :
  ∀ (this_year last_year : ℕ),
  last_year = 5000 →
  (3 * this_year) / 4 = 7500 →
  (this_year : ℚ) / last_year = 2 := by
sorry

end NUMINAMATH_CALUDE_phone_production_ratio_l2278_227832


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2278_227854

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a sampling method -/
def selectionProbability (method : SamplingMethod) (N : ℕ) (n : ℕ) : ℝ :=
  sorry

theorem equal_selection_probability (N : ℕ) (n : ℕ) :
  ∀ (m₁ m₂ : SamplingMethod), selectionProbability m₁ N n = selectionProbability m₂ N n :=
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2278_227854


namespace NUMINAMATH_CALUDE_books_written_proof_l2278_227877

/-- The number of books Zig wrote -/
def zig_books : ℕ := 60

/-- The number of books Flo wrote -/
def flo_books : ℕ := zig_books / 4

/-- The total number of books written by Zig and Flo -/
def total_books : ℕ := zig_books + flo_books

theorem books_written_proof :
  (zig_books = 4 * flo_books) → total_books = 75 := by
  sorry

end NUMINAMATH_CALUDE_books_written_proof_l2278_227877


namespace NUMINAMATH_CALUDE_marathon_distance_l2278_227820

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) 
  (yards_per_mile : ℕ) (h1 : marathons = 15) (h2 : miles_per_marathon = 26) 
  (h3 : yards_per_marathon = 395) (h4 : yards_per_mile = 1760) : 
  ∃ (m : ℕ) (y : ℕ), 
    (marathons * miles_per_marathon * yards_per_mile + marathons * yards_per_marathon = 
      m * yards_per_mile + y) ∧ 
    y < yards_per_mile ∧ 
    y = 645 := by
  sorry

end NUMINAMATH_CALUDE_marathon_distance_l2278_227820


namespace NUMINAMATH_CALUDE_special_rectangle_dimensions_l2278_227846

/-- A rectangle with the given properties -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  perimeter_area_relation : 2 * (width + length) = 3 * (width * length)
  length_width_relation : length = 2 * width

/-- The dimensions of the special rectangle are 1 inch width and 2 inches length -/
theorem special_rectangle_dimensions (rect : SpecialRectangle) : rect.width = 1 ∧ rect.length = 2 := by
  sorry

#check special_rectangle_dimensions

end NUMINAMATH_CALUDE_special_rectangle_dimensions_l2278_227846


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2278_227810

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2278_227810


namespace NUMINAMATH_CALUDE_horner_operations_count_l2278_227853

/-- Represents a polynomial of degree 6 with a constant term -/
structure Polynomial6 where
  coeffs : Fin 7 → ℝ
  constant_term : coeffs 0 ≠ 0

/-- Counts the number of operations in Horner's method for a polynomial of degree 6 -/
def horner_operations (p : Polynomial6) : ℕ :=
  6 + 6

theorem horner_operations_count (p : Polynomial6) :
  horner_operations p = 12 := by
  sorry

#check horner_operations_count

end NUMINAMATH_CALUDE_horner_operations_count_l2278_227853


namespace NUMINAMATH_CALUDE_number_categorization_l2278_227898

def given_numbers : List ℚ := [-10, 2/3, 0, -0.6, 4, -4 - 2/7]

def positive_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x > 0}

def negative_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x < 0}

def integer_numbers (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ ∃ n : ℤ, x = n}

def negative_fractions (numbers : List ℚ) : Set ℚ :=
  {x | x ∈ numbers ∧ x < 0 ∧ ¬∃ n : ℤ, x = n}

theorem number_categorization :
  positive_numbers given_numbers = {2/3, 4} ∧
  negative_numbers given_numbers = {-10, -0.6, -4 - 2/7} ∧
  integer_numbers given_numbers = {-10, 0, 4} ∧
  negative_fractions given_numbers = {-0.6, -4 - 2/7} := by
  sorry

end NUMINAMATH_CALUDE_number_categorization_l2278_227898


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2278_227870

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 72) - (Real.sqrt 294 / Real.sqrt 98) = Real.sqrt 10 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2278_227870


namespace NUMINAMATH_CALUDE_inequality_solution_l2278_227897

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 24) ↔
  (x < 1 ∨ (4 < x ∧ x < 5) ∨ 6 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2278_227897


namespace NUMINAMATH_CALUDE_intersection_circles_sum_l2278_227821

/-- Given two circles intersecting at points A(1,3) and B(m,-1), with their centers lying on the line x-y+c=0, prove that m + c = 3 -/
theorem intersection_circles_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (1, 3) ∈ circle1 ∧ (1, 3) ∈ circle2 ∧
    (m, -1) ∈ circle1 ∧ (m, -1) ∈ circle2 ∧
    (∃ (center1 center2 : ℝ × ℝ), 
      center1 ∈ circle1 ∧ center2 ∈ circle2 ∧
      center1.1 - center1.2 + c = 0 ∧
      center2.1 - center2.2 + c = 0)) →
  m + c = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_circles_sum_l2278_227821


namespace NUMINAMATH_CALUDE_square_area_error_l2278_227833

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := 1.25 * x
  let actual_area := x ^ 2
  let calculated_area := measured_side ^ 2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 56.25 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l2278_227833


namespace NUMINAMATH_CALUDE_shoe_selection_outcomes_l2278_227843

/-- The number of distinct pairs of shoes -/
def num_pairs : ℕ := 10

/-- The number of shoes drawn -/
def num_drawn : ℕ := 4

/-- The number of ways to select 4 shoes such that none form a pair -/
def no_pairs : ℕ := (Nat.choose num_pairs num_drawn) * (2^num_drawn)

/-- The number of ways to select 4 shoes such that two form a pair and the other two do not form pairs -/
def one_pair : ℕ := (Nat.choose num_pairs 2) * (2^2) * (Nat.choose (num_pairs - 2) 1)

/-- The number of ways to select 4 shoes such that they form two complete pairs -/
def two_pairs : ℕ := Nat.choose num_pairs 2

theorem shoe_selection_outcomes :
  no_pairs = 3360 ∧ one_pair = 1440 ∧ two_pairs = 45 := by
  sorry

end NUMINAMATH_CALUDE_shoe_selection_outcomes_l2278_227843


namespace NUMINAMATH_CALUDE_prob_not_same_group_three_groups_l2278_227811

/-- The probability that two students are not in the same interest group -/
def prob_not_same_group (num_groups : ℕ) : ℚ :=
  if num_groups = 0 then 0
  else (num_groups - 1 : ℚ) / num_groups

theorem prob_not_same_group_three_groups :
  prob_not_same_group 3 = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_not_same_group_three_groups_l2278_227811


namespace NUMINAMATH_CALUDE_basketball_team_score_l2278_227842

theorem basketball_team_score :
  ∀ (chandra akiko michiko bailey damien ella : ℕ),
    chandra = 2 * akiko →
    akiko = michiko + 4 →
    michiko * 2 = bailey →
    bailey = 14 →
    damien = 3 * akiko →
    ella = chandra + (chandra / 5) →
    chandra + akiko + michiko + bailey + damien + ella = 113 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_score_l2278_227842


namespace NUMINAMATH_CALUDE_solve_star_equation_l2278_227885

-- Define the ★ operation
def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

-- State the theorem
theorem solve_star_equation : 
  ∃ (a : ℝ), star a 3 = 15 ∧ a = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l2278_227885


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2278_227808

theorem sum_of_cubes (x y z : ℕ+) : 
  (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 378 → x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2278_227808


namespace NUMINAMATH_CALUDE_count_numbers_correct_l2278_227823

/-- The count of n-digit numbers composed of digits 1, 2, and 3, where each digit appears at least once -/
def count_numbers (n : ℕ) : ℕ :=
  3^n - 3 * 2^n + 3

/-- Theorem stating that count_numbers gives the correct count -/
theorem count_numbers_correct (n : ℕ) :
  count_numbers n = (3^n : ℕ) - 3 * (2^n : ℕ) + 3 :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_correct_l2278_227823


namespace NUMINAMATH_CALUDE_eugene_initial_pencils_l2278_227890

/-- The number of pencils Eugene initially had -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Eugene gave to Joyce -/
def pencils_given : ℕ := 6

/-- The number of pencils Eugene has left -/
def pencils_left : ℕ := 45

/-- Theorem: Eugene initially had 51 pencils -/
theorem eugene_initial_pencils :
  initial_pencils = pencils_given + pencils_left ∧ initial_pencils = 51 := by
  sorry

end NUMINAMATH_CALUDE_eugene_initial_pencils_l2278_227890


namespace NUMINAMATH_CALUDE_unique_multiplication_property_l2278_227841

theorem unique_multiplication_property : ∃! n : ℕ, 
  (n ≥ 10000000 ∧ n < 100000000) ∧  -- 8-digit number
  (n % 10 = 9) ∧                    -- ends in 9
  (∃ k : ℕ, n * 9 = k * 111111111)  -- when multiplied by 9, equals k * 111111111
    := by sorry

end NUMINAMATH_CALUDE_unique_multiplication_property_l2278_227841


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l2278_227879

theorem no_real_solution_for_log_equation :
  ∀ x : ℝ, ¬(Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 15)) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l2278_227879


namespace NUMINAMATH_CALUDE_payment_calculation_l2278_227819

/-- Represents the pricing and discount options for suits and ties -/
structure StorePolicy where
  suit_price : ℕ
  tie_price : ℕ
  option1_free_ties : ℕ
  option2_discount : ℚ

/-- Calculates the payment for Option 1 -/
def option1_payment (policy : StorePolicy) (suits : ℕ) (ties : ℕ) : ℕ :=
  policy.suit_price * suits + policy.tie_price * (ties - suits)

/-- Calculates the payment for Option 2 -/
def option2_payment (policy : StorePolicy) (suits : ℕ) (ties : ℕ) : ℚ :=
  (1 - policy.option2_discount) * (policy.suit_price * suits + policy.tie_price * ties)

/-- Theorem statement for the payment calculations -/
theorem payment_calculation (x : ℕ) (h : x > 10) :
  let policy : StorePolicy := {
    suit_price := 1000,
    tie_price := 200,
    option1_free_ties := 1,
    option2_discount := 1/10
  }
  option1_payment policy 10 x = 200 * x + 8000 ∧
  option2_payment policy 10 x = 180 * x + 9000 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l2278_227819


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l2278_227861

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l2278_227861


namespace NUMINAMATH_CALUDE_probability_above_parabola_l2278_227835

/-- The type of single-digit positive integers -/
def SingleDigitPos := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- The condition for a point (a,b) to be above the parabola y = ax^2 + bx for all x -/
def IsAboveParabola (a b : SingleDigitPos) : Prop :=
  ∀ x : ℝ, (b : ℝ) > (a : ℝ) * x^2 + (b : ℝ) * x

/-- The number of valid (a,b) pairs -/
def NumValidPairs : ℕ := 72

/-- The total number of possible (a,b) pairs -/
def TotalPairs : ℕ := 81

/-- The main theorem: the probability of (a,b) being above the parabola is 8/9 -/
theorem probability_above_parabola :
  (NumValidPairs : ℚ) / (TotalPairs : ℚ) = 8 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_above_parabola_l2278_227835


namespace NUMINAMATH_CALUDE_area_of_triangle_NOI_l2278_227815

/-- Triangle PQR with given side lengths -/
structure TrianglePQR where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  side_lengths : PQ = 15 ∧ PR = 8 ∧ QR = 17

/-- Point O is the circumcenter of triangle PQR -/
def is_circumcenter (O : ℝ × ℝ) (t : TrianglePQR) : Prop :=
  sorry

/-- Point I is the incenter of triangle PQR -/
def is_incenter (I : ℝ × ℝ) (t : TrianglePQR) : Prop :=
  sorry

/-- Point N is the center of a circle tangent to sides PQ, PR, and the circumcircle -/
def is_tangent_circle_center (N : ℝ × ℝ) (t : TrianglePQR) (O : ℝ × ℝ) : Prop :=
  sorry

/-- Calculate the area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem: The area of triangle NOI is 5 -/
theorem area_of_triangle_NOI (t : TrianglePQR) (O I N : ℝ × ℝ) 
  (hO : is_circumcenter O t) 
  (hI : is_incenter I t)
  (hN : is_tangent_circle_center N t O) : 
  triangle_area N O I = 5 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_NOI_l2278_227815


namespace NUMINAMATH_CALUDE_cake_sharing_percentage_l2278_227816

theorem cake_sharing_percentage (total : ℝ) (rich_portion : ℝ) (ben_portion : ℝ) : 
  total > 0 →
  rich_portion > 0 →
  ben_portion > 0 →
  rich_portion + ben_portion = total →
  rich_portion / ben_portion = 3 →
  ben_portion / total = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_cake_sharing_percentage_l2278_227816


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l2278_227830

/-- Given a cube root of unity ω, prove the equality for any complex numbers a, b, c -/
theorem cube_root_unity_product (ω : ℂ) (a b c : ℂ) 
  (h1 : ω^3 = 1) 
  (h2 : 1 + ω + ω^2 = 0) : 
  (a + b*ω + c*ω^2) * (a + b*ω^2 + c*ω) = a^2 + b^2 + c^2 - a*b - b*c - c*a := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l2278_227830


namespace NUMINAMATH_CALUDE_white_rhino_weight_is_5100_l2278_227831

/-- The weight of one white rhino in pounds -/
def white_rhino_weight : ℝ := 5100

/-- The weight of one black rhino in pounds -/
def black_rhino_weight : ℝ := 2000

/-- The total weight of 7 white rhinos and 8 black rhinos in pounds -/
def total_weight : ℝ := 51700

/-- Theorem: The weight of one white rhino is 5100 pounds -/
theorem white_rhino_weight_is_5100 :
  7 * white_rhino_weight + 8 * black_rhino_weight = total_weight :=
by sorry

end NUMINAMATH_CALUDE_white_rhino_weight_is_5100_l2278_227831


namespace NUMINAMATH_CALUDE_four_integer_sum_l2278_227878

theorem four_integer_sum (a b c d : ℕ+) 
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_sums : a + b = 6 ∧ a + c = 8 ∧ b + c = 12 ∧ a + d = 21)
  (h_distinct : a + b ≠ a + c ∧ a + b ≠ a + d ∧ a + b ≠ b + c ∧ a + b ≠ b + d ∧ a + b ≠ c + d ∧
                a + c ≠ a + d ∧ a + c ≠ b + c ∧ a + c ≠ b + d ∧ a + c ≠ c + d ∧
                a + d ≠ b + c ∧ a + d ≠ b + d ∧ a + d ≠ c + d ∧
                b + c ≠ b + d ∧ b + c ≠ c + d ∧
                b + d ≠ c + d) :
  d = 20 := by
  sorry

end NUMINAMATH_CALUDE_four_integer_sum_l2278_227878


namespace NUMINAMATH_CALUDE_circle_line_intersection_sum_l2278_227867

/-- Given a circle with radius 4 centered at the origin and a line y = √3x - 4
    intersecting the circle at points A and B, the sum of the length of segment AB
    and the length of the larger arc AB is (16π/3) + 4√3. -/
theorem circle_line_intersection_sum (A B : ℝ × ℝ) : 
  let r : ℝ := 4
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x - 4}
  A ∈ circle ∧ A ∈ line ∧ B ∈ circle ∧ B ∈ line ∧ A ≠ B →
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := (2 * π - angle) * r
  segment_length + arc_length = (16 * π / 3) + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_sum_l2278_227867


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2278_227836

theorem sum_of_squares_of_roots (a b : ℝ) 
  (ha : a^2 - 6*a + 4 = 0) 
  (hb : b^2 - 6*b + 4 = 0) 
  (hab : a ≠ b) : 
  a^2 + b^2 = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2278_227836


namespace NUMINAMATH_CALUDE_sequence_problem_l2278_227899

theorem sequence_problem (a : ℕ → ℕ) (n : ℕ) : 
  a 1 = 2 ∧ 
  (∀ k ≥ 1, a (k + 1) = a k + 3) ∧ 
  a n = 2009 →
  n = 670 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l2278_227899


namespace NUMINAMATH_CALUDE_geometry_problem_l2278_227875

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ m : ℝ, (∀ x y : ℝ, f m x y ↔ g m x y) → 
    (m = -3 ∨ m = 0)

-- Define a point P
def P (m : ℝ) : ℝ × ℝ := (1, 2 * m)

-- Define a line l passing through a point with specific intercept property
def l (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ (k : ℝ), (y - p.2 = k * (x - p.1)) ∧ 
  ((k * p.1 - p.2) / k = -(k - p.2))

-- Main theorem
theorem geometry_problem :
  (perpendicular l₁ l₂) ∧
  (∀ m : ℝ, l₂ m (P m).1 (P m).2 → 
    (∀ x y : ℝ, l (P m) x y ↔ (2 * x - y = 0 ∨ x - y + 1 = 0))) :=
sorry

end NUMINAMATH_CALUDE_geometry_problem_l2278_227875


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l2278_227803

theorem cos_pi_4_plus_alpha (α : Real) 
  (h : Real.sin (α - π/4) = 1/3) : 
  Real.cos (π/4 + α) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l2278_227803


namespace NUMINAMATH_CALUDE_nell_card_count_l2278_227814

/-- The number of cards Nell has after receiving cards from Jeff -/
def total_cards (initial : Float) (received : Float) : Float :=
  initial + received

/-- Theorem stating that Nell's total cards is the sum of her initial cards and received cards -/
theorem nell_card_count (initial : Float) (received : Float) :
  total_cards initial received = initial + received := by sorry

end NUMINAMATH_CALUDE_nell_card_count_l2278_227814


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2278_227886

theorem fraction_to_decimal : (47 : ℚ) / 160 = 0.29375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2278_227886


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2278_227800

def A (m : ℝ) : Set ℝ := {3, m^2}
def B (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}

theorem subset_implies_m_equals_one (m : ℝ) : A m ⊆ B m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2278_227800


namespace NUMINAMATH_CALUDE_equation_solution_l2278_227851

theorem equation_solution : ∃! x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2278_227851


namespace NUMINAMATH_CALUDE_line_equation_theorem_l2278_227891

-- Define the line l
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ
  xIntercept : ℝ
  yIntercept : ℝ

-- Define the conditions
def lineConditions (l : Line) : Prop :=
  l.passesThrough = (2, 3) ∧
  l.slope = Real.tan (2 * Real.pi / 3) ∧
  l.xIntercept + l.yIntercept = 0

-- Define the possible equations of the line
def lineEquation (l : Line) (x y : ℝ) : Prop :=
  (3 * x - 2 * y = 0) ∨ (x - y + 1 = 0)

-- The theorem to prove
theorem line_equation_theorem (l : Line) :
  lineConditions l → ∀ x y, lineEquation l x y :=
sorry

end NUMINAMATH_CALUDE_line_equation_theorem_l2278_227891


namespace NUMINAMATH_CALUDE_iodine_atom_radius_scientific_notation_l2278_227884

theorem iodine_atom_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000000133 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -8 :=
by sorry

end NUMINAMATH_CALUDE_iodine_atom_radius_scientific_notation_l2278_227884


namespace NUMINAMATH_CALUDE_R_is_top_right_l2278_227817

/-- Represents a rectangle with integer labels at its corners -/
structure Rectangle where
  a : Int  -- left-top
  b : Int  -- right-top
  c : Int  -- right-bottom
  d : Int  -- left-bottom

/-- The set of four rectangles -/
def rectangles : Finset Rectangle := sorry

/-- P is one of the rectangles -/
def P : Rectangle := ⟨5, 1, 8, 2⟩

/-- Q is one of the rectangles -/
def Q : Rectangle := ⟨2, 8, 10, 4⟩

/-- R is one of the rectangles -/
def R : Rectangle := ⟨4, 5, 1, 7⟩

/-- S is one of the rectangles -/
def S : Rectangle := ⟨8, 3, 7, 5⟩

/-- The rectangles are arranged in a 2x2 matrix -/
def isArranged2x2 (rects : Finset Rectangle) : Prop := sorry

/-- A rectangle is at the top-right position -/
def isTopRight (rect : Rectangle) (rects : Finset Rectangle) : Prop := sorry

/-- Main theorem: R is at the top-right position -/
theorem R_is_top_right : isTopRight R rectangles := by sorry

end NUMINAMATH_CALUDE_R_is_top_right_l2278_227817


namespace NUMINAMATH_CALUDE_total_elixir_ways_l2278_227892

/-- The number of ways to prepare magical dust -/
def total_magical_dust_ways : ℕ := 4

/-- The number of elixirs made from fairy dust -/
def fairy_dust_elixirs : ℕ := 3

/-- The number of elixirs made from elf dust -/
def elf_dust_elixirs : ℕ := 4

/-- The number of ways to prepare fairy dust -/
def fairy_dust_ways : ℕ := 2

/-- The number of ways to prepare elf dust -/
def elf_dust_ways : ℕ := 2

/-- Theorem: The total number of ways to prepare all the elixirs is 14 -/
theorem total_elixir_ways : 
  fairy_dust_ways * fairy_dust_elixirs + elf_dust_ways * elf_dust_elixirs = 14 :=
by sorry

end NUMINAMATH_CALUDE_total_elixir_ways_l2278_227892


namespace NUMINAMATH_CALUDE_weekly_average_rainfall_l2278_227818

/-- Calculates the daily average rainfall for a week given specific conditions. -/
theorem weekly_average_rainfall : 
  let monday_rain : ℝ := 2 + 1
  let tuesday_rain : ℝ := 2 * monday_rain
  let wednesday_rain : ℝ := 0
  let thursday_rain : ℝ := 1
  let friday_rain : ℝ := monday_rain + tuesday_rain + wednesday_rain + thursday_rain
  let total_rainfall : ℝ := monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain
  let days_in_week : ℕ := 7
  total_rainfall / days_in_week = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_weekly_average_rainfall_l2278_227818


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l2278_227863

/-- The number of dice being rolled -/
def numDice : ℕ := 6

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def probOdd : ℚ := 1/2

/-- The number of dice that need to show even (and odd) numbers for the desired outcome -/
def numEven : ℕ := numDice / 2

theorem equal_even_odd_probability :
  (Nat.choose numDice numEven : ℚ) * probEven ^ numDice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l2278_227863


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2278_227822

theorem abs_neg_three_eq_three : |(-3 : ℚ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2278_227822


namespace NUMINAMATH_CALUDE_supermarket_spending_l2278_227866

theorem supermarket_spending (total_spent : ℚ) 
  (h1 : total_spent = 150)
  (h2 : ∃ (fruits_veg meat bakery candy : ℚ),
    fruits_veg = 1/2 * total_spent ∧
    meat = 1/3 * total_spent ∧
    candy = 10 ∧
    fruits_veg + meat + bakery + candy = total_spent) :
  ∃ (bakery : ℚ), bakery = 1/10 * total_spent := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2278_227866


namespace NUMINAMATH_CALUDE_min_value_theorem_l2278_227802

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (heq : a + 2*b = 2) :
  ∃ (min_val : ℝ), min_val = 4*(1 + Real.sqrt 2) ∧
  ∀ (x : ℝ), x = 2/(a - 1) + a/b → x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2278_227802


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2278_227804

theorem larger_integer_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 198) : 
  x.val = 18 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2278_227804


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2278_227871

theorem sphere_surface_area (C : ℝ) (h : C = 4 * Real.pi) :
  ∃ (S : ℝ), S = 16 * Real.pi ∧ S = 4 * Real.pi * (C / (2 * Real.pi))^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2278_227871


namespace NUMINAMATH_CALUDE_product_327_3_base9_l2278_227806

/-- Represents a number in base 9 --/
def Base9 := ℕ

/-- Converts a base 9 number to a natural number --/
def to_nat (x : Base9) : ℕ := sorry

/-- Converts a natural number to a base 9 number --/
def from_nat (x : ℕ) : Base9 := sorry

/-- Multiplies two base 9 numbers --/
def mul_base9 (x y : Base9) : Base9 := sorry

theorem product_327_3_base9 : 
  mul_base9 (from_nat 327) (from_nat 3) = from_nat 1083 := by sorry

end NUMINAMATH_CALUDE_product_327_3_base9_l2278_227806


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2278_227868

theorem complex_arithmetic_equality : (469157 * 9999)^2 / 53264 + 3758491 = 413303758491 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2278_227868


namespace NUMINAMATH_CALUDE_triangle_area_l2278_227848

/-- The area of a triangle with perimeter 28 cm and inradius 2.0 cm is 28 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2 → area = inradius * (perimeter / 2) → area = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2278_227848


namespace NUMINAMATH_CALUDE_shampoo_duration_l2278_227876

theorem shampoo_duration (rose_shampoo : Rat) (jasmine_shampoo : Rat) (daily_usage : Rat) : 
  rose_shampoo = 1/3 → jasmine_shampoo = 1/4 → daily_usage = 1/12 →
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_duration_l2278_227876


namespace NUMINAMATH_CALUDE_smallest_three_types_sixty_nine_includes_three_types_l2278_227864

/-- Represents a type of tree in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset ℕ)
  (type : ℕ → TreeType)
  (total_count : trees.card = 100)
  (four_types_in_85 : ∀ s : Finset ℕ, s ⊆ trees → s.card = 85 → 
    (∃ i ∈ s, type i = TreeType.Birch) ∧
    (∃ i ∈ s, type i = TreeType.Spruce) ∧
    (∃ i ∈ s, type i = TreeType.Pine) ∧
    (∃ i ∈ s, type i = TreeType.Aspen))

/-- The main theorem stating the smallest number of trees that must include at least three types -/
theorem smallest_three_types (g : Grove) : 
  ∀ n < 69, ∃ s : Finset ℕ, s ⊆ g.trees ∧ s.card = n ∧ 
    (∃ t1 t2 : TreeType, ∀ i ∈ s, g.type i = t1 ∨ g.type i = t2) :=
by sorry

/-- The theorem stating that 69 trees always include at least three types -/
theorem sixty_nine_includes_three_types (g : Grove) :
  ∀ s : Finset ℕ, s ⊆ g.trees → s.card = 69 → 
    ∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    (∃ i ∈ s, g.type i = t1) ∧ (∃ i ∈ s, g.type i = t2) ∧ (∃ i ∈ s, g.type i = t3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_types_sixty_nine_includes_three_types_l2278_227864


namespace NUMINAMATH_CALUDE_cube_sum_plus_three_l2278_227837

theorem cube_sum_plus_three (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_plus_three_l2278_227837


namespace NUMINAMATH_CALUDE_barbara_candies_l2278_227894

/-- The number of candies Barbara has in total is 27, given her initial candies and additional purchase. -/
theorem barbara_candies : 
  ∀ (initial_candies additional_candies : ℕ),
    initial_candies = 9 →
    additional_candies = 18 →
    initial_candies + additional_candies = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_l2278_227894


namespace NUMINAMATH_CALUDE_square_diff_product_plus_square_equals_five_l2278_227857

theorem square_diff_product_plus_square_equals_five 
  (a b : ℝ) (ha : a = Real.sqrt 2 + 1) (hb : b = Real.sqrt 2 - 1) : 
  a^2 - a*b + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_square_diff_product_plus_square_equals_five_l2278_227857


namespace NUMINAMATH_CALUDE_initial_overs_is_ten_l2278_227809

/-- Represents a cricket game scenario --/
structure CricketGame where
  target : ℕ
  initialRunRate : ℚ
  remainingOvers : ℕ
  requiredRunRate : ℚ

/-- Calculates the number of overs played initially in a cricket game --/
def initialOvers (game : CricketGame) : ℚ :=
  (game.target - game.remainingOvers * game.requiredRunRate) / game.initialRunRate

/-- Theorem stating that the number of overs played initially is 10 --/
theorem initial_overs_is_ten (game : CricketGame) 
  (h1 : game.target = 282)
  (h2 : game.initialRunRate = 16/5)
  (h3 : game.remainingOvers = 50)
  (h4 : game.requiredRunRate = 5)
  : initialOvers game = 10 := by
  sorry

#eval initialOvers { target := 282, initialRunRate := 16/5, remainingOvers := 50, requiredRunRate := 5 }

end NUMINAMATH_CALUDE_initial_overs_is_ten_l2278_227809


namespace NUMINAMATH_CALUDE_fruit_selection_problem_l2278_227855

/-- The number of ways to choose n items from k groups with at least m items from each group -/
def choose_with_minimum (n k m : ℕ) : ℕ :=
  (n - k * m + k - 1).choose (k - 1)

/-- The problem statement -/
theorem fruit_selection_problem :
  choose_with_minimum 15 4 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_selection_problem_l2278_227855


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2278_227872

/-- Given a train of length 120 m crossing a bridge of length 255 m in 30 seconds,
    prove that the speed of the train is 45 km/hr. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2278_227872


namespace NUMINAMATH_CALUDE_salary_after_changes_l2278_227862

/-- Given an original salary, calculate the final salary after a raise and a reduction -/
def finalSalary (originalSalary : ℚ) (raisePercentage : ℚ) (reductionPercentage : ℚ) : ℚ :=
  let salaryAfterRaise := originalSalary * (1 + raisePercentage / 100)
  salaryAfterRaise * (1 - reductionPercentage / 100)

theorem salary_after_changes : 
  finalSalary 5000 10 5 = 5225 := by sorry

end NUMINAMATH_CALUDE_salary_after_changes_l2278_227862


namespace NUMINAMATH_CALUDE_total_money_collected_is_960_l2278_227813

/-- Calculates the total money collected from admission receipts for a play. -/
def totalMoneyCollected (totalPeople : Nat) (adultPrice : Nat) (childPrice : Nat) (numAdults : Nat) : Nat :=
  let numChildren := totalPeople - numAdults
  adultPrice * numAdults + childPrice * numChildren

/-- Theorem stating that the total money collected is 960 dollars given the specified conditions. -/
theorem total_money_collected_is_960 :
  totalMoneyCollected 610 2 1 350 = 960 := by
  sorry

end NUMINAMATH_CALUDE_total_money_collected_is_960_l2278_227813


namespace NUMINAMATH_CALUDE_intersection_A_B_min_value_fraction_l2278_227825

-- Define the parameters b and c based on the given inequality
def b : ℝ := 3
def c : ℝ := 6

-- Define the solution set of the original inequality
def original_solution_set : Set ℝ := {x | 2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -2 ≤ x ∧ x < 2}

-- Define the solution set of bx^2 - (c+1)x - c > 0
def A : Set ℝ := {x | b * x^2 - (c + 1) * x - c > 0}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 ≤ x ∧ x < -2/3} := by sorry

-- Theorem 2: Minimum value of the fraction
theorem min_value_fraction :
  ∀ x > 1, (x^2 - b*x + c) / (x - 1) ≥ 3 ∧
  ∃ x > 1, (x^2 - b*x + c) / (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_min_value_fraction_l2278_227825


namespace NUMINAMATH_CALUDE_distance_product_l2278_227824

noncomputable def f (x : ℝ) : ℝ := 2 * x + 5 / x

theorem distance_product (x : ℝ) (hx : x ≠ 0) :
  let P : ℝ × ℝ := (x, f x)
  let d₁ : ℝ := |f x - 2 * x| / Real.sqrt 5
  let d₂ : ℝ := |x|
  d₁ * d₂ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_product_l2278_227824


namespace NUMINAMATH_CALUDE_max_value_interval_m_range_l2278_227849

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem max_value_interval_m_range 
  (m : ℝ) 
  (h1 : ∃ (x : ℝ), m < x ∧ x < 8 - m^2 ∧ ∀ (y : ℝ), m < y ∧ y < 8 - m^2 → f y ≤ f x) :
  m ∈ Set.Ioc (-3) (-Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_max_value_interval_m_range_l2278_227849


namespace NUMINAMATH_CALUDE_lindsey_owns_four_more_cars_l2278_227881

/-- The number of cars owned by each person --/
structure CarOwnership where
  cathy : ℕ
  carol : ℕ
  susan : ℕ
  lindsey : ℕ

/-- The conditions of the car ownership problem --/
def carProblemConditions (co : CarOwnership) : Prop :=
  co.cathy = 5 ∧
  co.carol = 2 * co.cathy ∧
  co.susan = co.carol - 2 ∧
  co.lindsey > co.cathy ∧
  co.cathy + co.carol + co.susan + co.lindsey = 32

/-- The theorem stating that Lindsey owns 4 more cars than Cathy --/
theorem lindsey_owns_four_more_cars (co : CarOwnership) 
  (h : carProblemConditions co) : co.lindsey - co.cathy = 4 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_owns_four_more_cars_l2278_227881


namespace NUMINAMATH_CALUDE_power_equation_solution_l2278_227834

theorem power_equation_solution : ∃ y : ℕ, (12 ^ 3 * 6 ^ y) / 432 = 5184 :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2278_227834


namespace NUMINAMATH_CALUDE_largest_divisor_of_even_square_diff_l2278_227888

theorem largest_divisor_of_even_square_diff (m n : ℤ) : 
  Even m → Even n → n < m → 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) → 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) ∧ 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) → k = 4 :=
by sorry


end NUMINAMATH_CALUDE_largest_divisor_of_even_square_diff_l2278_227888


namespace NUMINAMATH_CALUDE_race_finish_order_l2278_227812

-- Define the athletes
inductive Athlete : Type
| Grisha : Athlete
| Sasha : Athlete
| Lena : Athlete

-- Define the race
structure Race where
  start_order : List Athlete
  finish_order : List Athlete
  overtakes : Athlete → Nat
  no_triple_overtake : Bool

-- Define the specific race conditions
def race_conditions (r : Race) : Prop :=
  r.start_order = [Athlete.Grisha, Athlete.Sasha, Athlete.Lena] ∧
  r.overtakes Athlete.Grisha = 10 ∧
  r.overtakes Athlete.Lena = 6 ∧
  r.overtakes Athlete.Sasha = 4 ∧
  r.no_triple_overtake = true ∧
  r.finish_order.length = 3 ∧
  r.finish_order.Nodup

-- Theorem statement
theorem race_finish_order (r : Race) :
  race_conditions r →
  r.finish_order = [Athlete.Grisha, Athlete.Sasha, Athlete.Lena] :=
by sorry

end NUMINAMATH_CALUDE_race_finish_order_l2278_227812


namespace NUMINAMATH_CALUDE_remainder_theorem_l2278_227839

/-- Given a polynomial p(x) satisfying p(0) = 2 and p(2) = 6,
    prove that the remainder when p(x) is divided by x(x-2) is 2x + 2 -/
theorem remainder_theorem (p : ℝ → ℝ) (h1 : p 0 = 2) (h2 : p 2 = 6) :
  ∃ (q : ℝ → ℝ), ∀ x, p x = q x * (x * (x - 2)) + (2 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2278_227839


namespace NUMINAMATH_CALUDE_expression_evaluation_l2278_227827

/-- Proves that (c^c - c(c-d)^c)^c = 136048896 when c = 4 and d = 2 -/
theorem expression_evaluation (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c*(c-d)^c)^c = 136048896 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2278_227827


namespace NUMINAMATH_CALUDE_rectangle_puzzle_l2278_227826

theorem rectangle_puzzle (top1 top2 top3 top4 bottom1 bottom2 bottom3 bottom4 X : ℝ) :
  top1 = 1 ∧ top2 = 3 ∧ top3 = 1 ∧ top4 = 1 ∧
  bottom1 = 3 ∧ bottom2 = 1 ∧ bottom3 = 3 ∧ bottom4 = 3 ∧
  top1 + top2 + top3 + top4 + X = bottom1 + bottom2 + bottom3 + bottom4 →
  X = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_puzzle_l2278_227826
