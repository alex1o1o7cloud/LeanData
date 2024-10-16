import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l1229_122953

-- Define the total number of purchases and samples for the first category
def purchases_category1 : ℕ := 116000
def samples_category1 : ℕ := 116

-- Define the number of purchases for the second category
def purchases_category2 : ℕ := 94000

-- Define the function to calculate the number of samples for the second category
def samples_category2 : ℚ := (samples_category1 : ℚ) * (purchases_category2 : ℚ) / (purchases_category1 : ℚ)

-- Theorem statement
theorem stratified_sampling_proportion :
  samples_category2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l1229_122953


namespace NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l1229_122952

/-- Given two concentric equilateral triangles with areas 25 and 4 square units respectively,
    prove that the area of one of the four congruent trapezoids formed between them is 5.25 square units. -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ) (inner_area : ℝ) (num_trapezoids : ℕ)
  (h_outer : outer_area = 25)
  (h_inner : inner_area = 4)
  (h_num : num_trapezoids = 4) :
  (outer_area - inner_area) / num_trapezoids = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l1229_122952


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1229_122910

theorem complex_fraction_equality : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1229_122910


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1229_122979

/-- Given a square A with perimeter 24 cm and a square B with area equal to one-fourth the area of square A, prove that the perimeter of square B is 12 cm. -/
theorem square_perimeter_relation (A B : ℝ → ℝ → Prop) : 
  (∃ a, ∀ x y, A x y ↔ (x = 0 ∨ x = a) ∧ (y = 0 ∨ y = a) ∧ 4 * a = 24) →
  (∃ b, ∀ x y, B x y ↔ (x = 0 ∨ x = b) ∧ (y = 0 ∨ y = b) ∧ b^2 = (a^2 / 4)) →
  (∃ p, p = 4 * b ∧ p = 12) :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1229_122979


namespace NUMINAMATH_CALUDE_binomial_8_5_l1229_122951

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_5_l1229_122951


namespace NUMINAMATH_CALUDE_intersection_line_l1229_122937

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line
def line (x y : ℝ) : Prop := 3*x - 3*y - 10 = 0

-- Theorem statement
theorem intersection_line :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle1 B.1 B.2) →
  (circle2 A.1 A.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  (line A.1 A.2 ∧ line B.1 B.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_l1229_122937


namespace NUMINAMATH_CALUDE_g_of_2_equals_5_l1229_122987

/-- Given a function g(x) = x^3 - 2x + 1, prove that g(2) = 5 -/
theorem g_of_2_equals_5 :
  let g : ℝ → ℝ := fun x ↦ x^3 - 2*x + 1
  g 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_equals_5_l1229_122987


namespace NUMINAMATH_CALUDE_temperature_altitude_relationship_l1229_122985

/-- Given that the ground temperature is 20°C and the temperature decreases by 6°C
    for every 1000m increase in altitude, prove that the functional relationship
    between temperature t(°C) and altitude h(m) is t = -0.006h + 20. -/
theorem temperature_altitude_relationship (h : ℝ) :
  let ground_temp : ℝ := 20
  let temp_decrease_per_km : ℝ := 6
  let altitude_increase : ℝ := 1000
  let t : ℝ → ℝ := fun h => -((temp_decrease_per_km / altitude_increase) * h) + ground_temp
  t h = -0.006 * h + 20 := by
  sorry

end NUMINAMATH_CALUDE_temperature_altitude_relationship_l1229_122985


namespace NUMINAMATH_CALUDE_cow_count_is_sixteen_l1229_122976

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem: If the total number of legs is 32 more than twice the number of heads,
    then the number of cows is 16 -/
theorem cow_count_is_sixteen (count : AnimalCount) :
  totalLegs count = 2 * totalHeads count + 32 → count.cows = 16 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_sixteen_l1229_122976


namespace NUMINAMATH_CALUDE_cookie_price_is_three_l1229_122934

/-- The price of each cookie in Zane's purchase --/
def cookie_price : ℚ := 3

/-- The total number of items (Oreos and cookies) --/
def total_items : ℕ := 65

/-- The ratio of Oreos to cookies --/
def oreo_cookie_ratio : ℚ := 4 / 9

/-- The price of each Oreo --/
def oreo_price : ℚ := 2

/-- The difference in total spent on cookies vs Oreos --/
def cookie_oreo_diff : ℚ := 95

theorem cookie_price_is_three :
  let num_cookies : ℚ := total_items / (1 + oreo_cookie_ratio)
  let num_oreos : ℚ := total_items - num_cookies
  let total_oreo_cost : ℚ := num_oreos * oreo_price
  let total_cookie_cost : ℚ := total_oreo_cost + cookie_oreo_diff
  cookie_price = total_cookie_cost / num_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_price_is_three_l1229_122934


namespace NUMINAMATH_CALUDE_distance_between_points_l1229_122950

/-- The distance between points (0,12) and (9,0) is 15 -/
theorem distance_between_points : Real.sqrt ((9 - 0)^2 + (0 - 12)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1229_122950


namespace NUMINAMATH_CALUDE_triangle_area_l1229_122932

theorem triangle_area (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  C = π / 4 →
  c = 2 →
  -- Area formula
  (1 / 2) * a * c * Real.sin B = (3 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1229_122932


namespace NUMINAMATH_CALUDE_second_number_irrelevant_l1229_122981

/-- A custom addition operation where the result is twice the first number -/
def customAdd (a b : ℝ) : ℝ := 2 * a

/-- Theorem stating that the second number in customAdd doesn't affect the result -/
theorem second_number_irrelevant (a b c : ℝ) : 
  customAdd a b = customAdd a c := by sorry

end NUMINAMATH_CALUDE_second_number_irrelevant_l1229_122981


namespace NUMINAMATH_CALUDE_pencils_per_box_l1229_122984

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 27)
  (h2 : num_boxes = 3)
  (h3 : total_pencils = num_boxes * pencils_per_box) :
  pencils_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_box_l1229_122984


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1229_122912

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ↔ 
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1229_122912


namespace NUMINAMATH_CALUDE_number_of_large_boats_proof_number_of_large_boats_l1229_122919

theorem number_of_large_boats (total_students : ℕ) (total_boats : ℕ) 
  (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) : ℕ :=
  let number_of_large_boats := 
    total_boats - (total_students - large_boat_capacity * total_boats) / 
      (large_boat_capacity - small_boat_capacity)
  number_of_large_boats

#check number_of_large_boats 50 10 6 4 = 5

theorem proof_number_of_large_boats :
  number_of_large_boats 50 10 6 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_large_boats_proof_number_of_large_boats_l1229_122919


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l1229_122916

/-- The cost of plastering a rectangular tank's walls and bottom -/
theorem tank_plastering_cost
  (length width depth : ℝ)
  (cost_per_sq_m_paise : ℝ)
  (h_length : length = 40)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost : cost_per_sq_m_paise = 125) :
  let bottom_area := length * width
  let perimeter := 2 * (length + width)
  let wall_area := perimeter * depth
  let total_area := bottom_area + wall_area
  let cost_per_sq_m_rupees := cost_per_sq_m_paise / 100
  total_area * cost_per_sq_m_rupees = 2350 :=
by
  sorry


end NUMINAMATH_CALUDE_tank_plastering_cost_l1229_122916


namespace NUMINAMATH_CALUDE_expression_equality_l1229_122925

/-- The base-10 logarithm -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Prove that 27^(1/3) + lg 4 + 2 * lg 5 - e^(ln 3) = 2 -/
theorem expression_equality : 27^(1/3) + lg 4 + 2 * lg 5 - Real.exp (Real.log 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1229_122925


namespace NUMINAMATH_CALUDE_count_pears_l1229_122954

/-- Given a box of fruits with apples and pears, prove the number of pears. -/
theorem count_pears (total_fruits : ℕ) (apples : ℕ) (pears : ℕ) : 
  total_fruits = 51 → apples = 12 → total_fruits = pears + apples → pears = 39 := by
  sorry

end NUMINAMATH_CALUDE_count_pears_l1229_122954


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l1229_122957

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) - b * cos(A) = 3/5 * c, then tan(A) / tan(B) = 4 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π → B > 0 → B < π → C > 0 → C < π →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = 3/5 * c →
  Real.tan A / Real.tan B = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l1229_122957


namespace NUMINAMATH_CALUDE_special_equation_result_l1229_122935

theorem special_equation_result (x : ℝ) (h : x + 1/x = Real.sqrt 7) :
  x^12 - 5*x^8 + 2*x^6 = 1944 * Real.sqrt 7 * x - 2494 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_result_l1229_122935


namespace NUMINAMATH_CALUDE_square_area_doubled_l1229_122906

theorem square_area_doubled (a : ℝ) (ha : a > 0) :
  (Real.sqrt 2 * a)^2 = 2 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_area_doubled_l1229_122906


namespace NUMINAMATH_CALUDE_marble_fraction_after_change_l1229_122971

theorem marble_fraction_after_change (total : ℚ) (h : total > 0) :
  let initial_blue := (2 / 3 : ℚ) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_fraction_after_change_l1229_122971


namespace NUMINAMATH_CALUDE_unique_three_digit_numbers_l1229_122963

/-- The number of available digits -/
def n : ℕ := 5

/-- The number of digits to be used in each number -/
def r : ℕ := 3

/-- The number of unique three-digit numbers that can be formed without repetition -/
def uniqueNumbers : ℕ := n.choose r * r.factorial

theorem unique_three_digit_numbers :
  uniqueNumbers = 60 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_numbers_l1229_122963


namespace NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l1229_122980

theorem smallest_angle_in_right_triangle (α β γ : ℝ) : 
  α = 90 → β = 55 → α + β + γ = 180 → min α (min β γ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l1229_122980


namespace NUMINAMATH_CALUDE_range_of_a_l1229_122944

-- Define the set of real numbers where the expression is meaningful
def MeaningfulSet : Set ℝ :=
  {a : ℝ | a - 2 ≥ 0 ∧ a ≠ 4}

-- Theorem stating the range of values for a
theorem range_of_a : MeaningfulSet = Set.Icc 2 4 ∪ Set.Ioi 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1229_122944


namespace NUMINAMATH_CALUDE_fair_hair_percentage_l1229_122942

/-- Proves that if 32% of employees are women with fair hair and 40% of fair-haired employees are women, then 80% of employees have fair hair. -/
theorem fair_hair_percentage (total_employees : ℝ) (women_fair_hair : ℝ) (fair_hair : ℝ)
  (h1 : women_fair_hair = 0.32 * total_employees)
  (h2 : women_fair_hair = 0.40 * fair_hair) :
  fair_hair / total_employees = 0.80 := by
sorry

end NUMINAMATH_CALUDE_fair_hair_percentage_l1229_122942


namespace NUMINAMATH_CALUDE_exponential_always_positive_l1229_122927

theorem exponential_always_positive : ¬∃ (x : ℝ), Real.exp x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_exponential_always_positive_l1229_122927


namespace NUMINAMATH_CALUDE_floor_x_floor_x_eq_42_l1229_122926

theorem floor_x_floor_x_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 := by
sorry

end NUMINAMATH_CALUDE_floor_x_floor_x_eq_42_l1229_122926


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1229_122969

theorem min_reciprocal_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) (heq : 30 - a = 4 * b) :
  (1 : ℚ) / a + 1 / b ≥ 3 / 10 ∧
  ((1 : ℚ) / a + 1 / b = 3 / 10 ↔ a = 10 ∧ b = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1229_122969


namespace NUMINAMATH_CALUDE_system_solutions_l1229_122929

theorem system_solutions : 
  ∀ (x y z : ℝ), 
    (x + y - z = -1) ∧ 
    (x^2 - y^2 + z^2 = 1) ∧ 
    (-x^3 + y^3 + z^3 = -1) → 
    ((x = 1 ∧ y = -1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1229_122929


namespace NUMINAMATH_CALUDE_probability_of_spades_formula_l1229_122991

def standardDeckSize : ℕ := 52
def spadesInDeck : ℕ := 13
def cardsDrawn : ℕ := 13

def probabilityOfSpades (n : ℕ) : ℚ :=
  (Nat.choose spadesInDeck n * Nat.choose (standardDeckSize - spadesInDeck) (cardsDrawn - n)) /
  Nat.choose standardDeckSize cardsDrawn

theorem probability_of_spades_formula (n : ℕ) (h1 : n ≤ spadesInDeck) (h2 : n ≤ cardsDrawn) :
  probabilityOfSpades n = (Nat.choose spadesInDeck n * Nat.choose (standardDeckSize - spadesInDeck) (cardsDrawn - n)) /
                          Nat.choose standardDeckSize cardsDrawn := by
  sorry

end NUMINAMATH_CALUDE_probability_of_spades_formula_l1229_122991


namespace NUMINAMATH_CALUDE_expand_product_l1229_122995

theorem expand_product (x : ℝ) : (x + 3) * (x - 1) * (x + 4) = x^3 + 6*x^2 + 5*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1229_122995


namespace NUMINAMATH_CALUDE_scaled_tile_height_l1229_122905

/-- Calculates the new height of a proportionally scaled tile -/
def new_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem: The new height of the scaled tile is 16 inches -/
theorem scaled_tile_height :
  let original_width : ℚ := 3
  let original_height : ℚ := 4
  let new_width : ℚ := 12
  new_height original_width original_height new_width = 16 := by
sorry

end NUMINAMATH_CALUDE_scaled_tile_height_l1229_122905


namespace NUMINAMATH_CALUDE_division_remainder_l1229_122902

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = divisor * quotient + remainder →
  dividend = 167 →
  divisor = 18 →
  quotient = 9 →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1229_122902


namespace NUMINAMATH_CALUDE_jake_weight_is_152_l1229_122904

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := sorry

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := sorry

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℝ := 212

theorem jake_weight_is_152 :
  (jake_weight - 32 = 2 * sister_weight) →
  (jake_weight + sister_weight = combined_weight) →
  jake_weight = 152 := by sorry

end NUMINAMATH_CALUDE_jake_weight_is_152_l1229_122904


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1229_122994

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : x + y = 28)
  (h3 : x - y = 8) :
  (∃ z : ℝ, z = 7 → y = 180 / 7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1229_122994


namespace NUMINAMATH_CALUDE_cloud_counting_proof_l1229_122920

def carson_clouds : ℕ := 6

def brother_clouds : ℕ := 3 * carson_clouds

def total_clouds : ℕ := carson_clouds + brother_clouds

theorem cloud_counting_proof : total_clouds = 24 := by
  sorry

end NUMINAMATH_CALUDE_cloud_counting_proof_l1229_122920


namespace NUMINAMATH_CALUDE_unique_row_contains_101_l1229_122974

/-- The number of rows in Pascal's Triangle that contain the number 101 -/
def rows_containing_101 : ℕ := 1

/-- 101 is a prime number -/
axiom prime_101 : Nat.Prime 101

/-- A number appears in Pascal's Triangle if it's a binomial coefficient -/
def appears_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (row k : ℕ), Nat.choose row k = n

theorem unique_row_contains_101 :
  (∃! row : ℕ, appears_in_pascals_triangle 101 ∧ row > 0) ∧
  rows_containing_101 = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_row_contains_101_l1229_122974


namespace NUMINAMATH_CALUDE_common_ratio_values_l1229_122968

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q
  third_term : a 3 = 2
  sum_second_fourth : a 2 + a 4 = 20 / 3
  q : ℝ

/-- The common ratio of the geometric sequence is either 3 or 1/3 -/
theorem common_ratio_values (seq : GeometricSequence) : seq.q = 3 ∨ seq.q = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_values_l1229_122968


namespace NUMINAMATH_CALUDE_f_increasing_l1229_122958

def f (x : ℝ) := 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l1229_122958


namespace NUMINAMATH_CALUDE_consecutive_zeros_in_power_of_five_l1229_122999

theorem consecutive_zeros_in_power_of_five : 
  ∃ n : ℕ, n < 1000000 ∧ (5^n : ℕ) % 1000000 = 0 := by sorry

end NUMINAMATH_CALUDE_consecutive_zeros_in_power_of_five_l1229_122999


namespace NUMINAMATH_CALUDE_book_page_ratio_l1229_122914

theorem book_page_ratio (total_pages : ℕ) (intro_pages : ℕ) (text_pages : ℕ) 
  (h1 : total_pages = 98)
  (h2 : intro_pages = 11)
  (h3 : text_pages = 19)
  (h4 : text_pages = (total_pages - intro_pages - text_pages * 2) / 2) :
  (total_pages - intro_pages - text_pages * 2) / total_pages = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_page_ratio_l1229_122914


namespace NUMINAMATH_CALUDE_joe_ball_choices_l1229_122911

/-- The number of balls in the bin -/
def num_balls : ℕ := 18

/-- The number of times a ball is chosen -/
def num_choices : ℕ := 4

/-- The number of different possible lists -/
def num_lists : ℕ := num_balls ^ num_choices

theorem joe_ball_choices :
  num_lists = 104976 := by
  sorry

end NUMINAMATH_CALUDE_joe_ball_choices_l1229_122911


namespace NUMINAMATH_CALUDE_train_speed_l1229_122986

/-- Proves that a train with given length and time to cross a stationary object has the specified speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed : ℝ) : 
  train_length = 250 →
  crossing_time = 12.857142857142858 →
  speed = (train_length / 1000) / (crossing_time / 3600) →
  speed = 70 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1229_122986


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1229_122983

theorem rectangle_dimensions : ∃ (a b : ℝ), 
  b = a + 3 ∧ 
  2*a + 2*b + a = a*b ∧ 
  a = 3 ∧ 
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1229_122983


namespace NUMINAMATH_CALUDE_females_advanced_degrees_count_l1229_122998

/-- Represents the employee distribution in a company -/
structure EmployeeDistribution where
  total : Nat
  females : Nat
  advanced_degrees : Nat
  males_college_only : Nat

/-- Calculates the number of females with advanced degrees -/
def females_with_advanced_degrees (e : EmployeeDistribution) : Nat :=
  e.advanced_degrees - (e.total - e.females - e.males_college_only)

/-- Theorem stating the number of females with advanced degrees -/
theorem females_advanced_degrees_count 
  (e : EmployeeDistribution)
  (h1 : e.total = 200)
  (h2 : e.females = 120)
  (h3 : e.advanced_degrees = 100)
  (h4 : e.males_college_only = 40) :
  females_with_advanced_degrees e = 60 := by
  sorry

#eval females_with_advanced_degrees { 
  total := 200, 
  females := 120, 
  advanced_degrees := 100, 
  males_college_only := 40 
}

end NUMINAMATH_CALUDE_females_advanced_degrees_count_l1229_122998


namespace NUMINAMATH_CALUDE_solution_set_fraction_inequality_l1229_122964

theorem solution_set_fraction_inequality :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_fraction_inequality_l1229_122964


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_is_625_div_3_l1229_122988

/-- A right triangle XYZ in the xy-plane with specific properties -/
structure RightTriangle where
  /-- The length of the hypotenuse XY -/
  hypotenuse_length : ℝ
  /-- The y-intercept of the line containing the median through X -/
  median_x_intercept : ℝ
  /-- The slope of the line containing the median through Y -/
  median_y_slope : ℝ
  /-- The y-intercept of the line containing the median through Y -/
  median_y_intercept : ℝ
  /-- Condition: The hypotenuse length is 50 -/
  hypotenuse_cond : hypotenuse_length = 50
  /-- Condition: The median through X lies on y = x + 5 -/
  median_x_cond : median_x_intercept = 5
  /-- Condition: The median through Y lies on y = 3x + 6 -/
  median_y_cond : median_y_slope = 3 ∧ median_y_intercept = 6

/-- The theorem stating that the area of the specific right triangle is 625/3 -/
theorem right_triangle_area (t : RightTriangle) : ℝ :=
  625 / 3

/-- The main theorem to be proved -/
theorem right_triangle_area_is_625_div_3 (t : RightTriangle) :
  right_triangle_area t = 625 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_is_625_div_3_l1229_122988


namespace NUMINAMATH_CALUDE_no_solution_condition_l1229_122907

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → x ≠ -1 → (1 / (x + 1) ≠ 3 * k / x)) ↔ (k = 0 ∨ k = 1/3) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1229_122907


namespace NUMINAMATH_CALUDE_inequality_proof_l1229_122922

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 ∧
  ((a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3 / 2 ↔
   a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1229_122922


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1229_122931

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 3 - x^2 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = x * Real.sqrt 3 ∨ y = -x * Real.sqrt 3

/-- Theorem: The asymptotes of the given hyperbola are y = ±√3x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1229_122931


namespace NUMINAMATH_CALUDE_power_division_result_l1229_122970

theorem power_division_result : (6 : ℕ)^12 / (36 : ℕ)^5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_division_result_l1229_122970


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1229_122923

theorem complex_equation_solution (z : ℂ) : 2 + z = (2 - z) * I → z = 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1229_122923


namespace NUMINAMATH_CALUDE_second_number_value_l1229_122949

theorem second_number_value (A B C : ℝ) 
  (sum_eq : A + B + C = 157.5)
  (ratio_AB : A / B = 3.5 / 4.25)
  (ratio_BC : B / C = 7.5 / 11.25)
  (diff_AC : A - C = 12.75) :
  B = 18.75 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1229_122949


namespace NUMINAMATH_CALUDE_f_symmetry_l1229_122982

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the theorem
theorem f_symmetry (a b : ℝ) :
  f a b 2017 = 7 → f a b (-2017) = -11 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l1229_122982


namespace NUMINAMATH_CALUDE_minimal_fraction_difference_l1229_122940

theorem minimal_fraction_difference (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (2 : ℚ) / 3 → q' ≥ q) →
  q - p = 11 := by
  sorry

end NUMINAMATH_CALUDE_minimal_fraction_difference_l1229_122940


namespace NUMINAMATH_CALUDE_max_value_sqrt_inequality_l1229_122941

theorem max_value_sqrt_inequality (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 6) :
  ∃ (k : ℝ), (∀ y : ℝ, y ≥ k → ∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ y) ∧
  (∀ z : ℝ, z > k → ¬∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ z) ∧
  k = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_inequality_l1229_122941


namespace NUMINAMATH_CALUDE_planes_distance_l1229_122978

/-- The total distance traveled by two planes moving towards each other -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem: The total distance traveled by two planes moving towards each other
    at 283 miles per hour for 2 hours is 1132 miles. -/
theorem planes_distance :
  total_distance 283 2 = 1132 :=
by sorry

end NUMINAMATH_CALUDE_planes_distance_l1229_122978


namespace NUMINAMATH_CALUDE_smallest_x_for_prime_abs_f_l1229_122930

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def f (x : ℤ) : ℤ := 4 * x^2 - 34 * x + 21

theorem smallest_x_for_prime_abs_f :
  ∃ (x : ℤ), (∀ (y : ℤ), y < x → ¬(is_prime (Int.natAbs (f y)))) ∧
             (is_prime (Int.natAbs (f x))) ∧
             x = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_prime_abs_f_l1229_122930


namespace NUMINAMATH_CALUDE_trapezium_side_length_l1229_122913

/-- Proves that given a trapezium with specified dimensions, the length of the unknown parallel side is 28 cm. -/
theorem trapezium_side_length 
  (known_side : ℝ)
  (height : ℝ)
  (area : ℝ)
  (h1 : known_side = 20)
  (h2 : height = 21)
  (h3 : area = 504)
  (h4 : area = (1/2) * (known_side + unknown_side) * height) :
  unknown_side = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l1229_122913


namespace NUMINAMATH_CALUDE_min_abs_diff_sum_l1229_122901

theorem min_abs_diff_sum (x a b : ℚ) : 
  x ≠ a ∧ x ≠ b ∧ a ≠ b → 
  a > b → 
  (∀ y : ℚ, |y - a| + |y - b| ≥ 2) ∧ (∃ z : ℚ, |z - a| + |z - b| = 2) →
  2022 + a - b = 2024 := by
sorry

end NUMINAMATH_CALUDE_min_abs_diff_sum_l1229_122901


namespace NUMINAMATH_CALUDE_system_solution_l1229_122973

theorem system_solution :
  ∃! (X Y Z : ℝ),
    0.15 * 40 = 0.25 * X + 2 ∧
    0.30 * 60 = 0.20 * Y + 3 ∧
    0.10 * Z = X - Y ∧
    X = 16 ∧ Y = 75 ∧ Z = -590 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1229_122973


namespace NUMINAMATH_CALUDE_gcd_of_container_volumes_l1229_122960

theorem gcd_of_container_volumes : Nat.gcd 496 (Nat.gcd 403 (Nat.gcd 713 (Nat.gcd 824 1171))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_container_volumes_l1229_122960


namespace NUMINAMATH_CALUDE_child_ticket_cost_l1229_122993

/-- Proves that the cost of each child's ticket is $7 -/
theorem child_ticket_cost (num_adults num_children : ℕ) (concession_cost total_cost adult_ticket_cost : ℚ) :
  num_adults = 5 →
  num_children = 2 →
  concession_cost = 12 →
  total_cost = 76 →
  adult_ticket_cost = 10 →
  (total_cost - concession_cost - num_adults * adult_ticket_cost) / num_children = 7 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l1229_122993


namespace NUMINAMATH_CALUDE_tangent_points_distance_circle_fixed_point_l1229_122948

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define a point on the directrix
structure PointOnDirectrix where
  x : ℝ
  y : ℝ
  on_directrix : directrix x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the tangent line from a point on the directrix to the parabola
def tangent_line (P : PointOnDirectrix) (Q : PointOnParabola) : Prop :=
  ∃ k : ℝ, Q.y = k * (Q.x - P.x)

-- Theorem 1: Distance between tangent points when P is on x-axis
theorem tangent_points_distance :
  ∀ (P : PointOnDirectrix) (Q R : PointOnParabola),
  P.y = 0 →
  tangent_line P Q →
  tangent_line P R →
  Q ≠ R →
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 16 :=
sorry

-- Theorem 2: Circle with diameter PQ passes through (1, 0)
theorem circle_fixed_point :
  ∀ (P : PointOnDirectrix) (Q : PointOnParabola),
  tangent_line P Q →
  ∃ (r : ℝ),
    (1 - ((P.x + Q.x) / 2))^2 + (0 - ((P.y + Q.y) / 2))^2 = r^2 ∧
    (P.x - ((P.x + Q.x) / 2))^2 + (P.y - ((P.y + Q.y) / 2))^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_points_distance_circle_fixed_point_l1229_122948


namespace NUMINAMATH_CALUDE_smallest_odd_with_same_divisors_as_360_l1229_122975

/-- Count the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := sorry

theorem smallest_odd_with_same_divisors_as_360 :
  ∃ (n : ℕ), isOdd n ∧ countDivisors n = countDivisors 360 ∧
  ∀ (m : ℕ), isOdd m ∧ countDivisors m = countDivisors 360 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_with_same_divisors_as_360_l1229_122975


namespace NUMINAMATH_CALUDE_job_pay_difference_l1229_122908

/-- Proves that the difference between two job pays is $375 given the total pay and the pay of the first job. -/
theorem job_pay_difference (total_pay first_job_pay : ℕ) 
  (h1 : total_pay = 3875)
  (h2 : first_job_pay = 2125) :
  first_job_pay - (total_pay - first_job_pay) = 375 := by
  sorry

end NUMINAMATH_CALUDE_job_pay_difference_l1229_122908


namespace NUMINAMATH_CALUDE_tim_total_score_l1229_122939

/-- The score for a single line in the game -/
def single_line_score : ℕ := 1000

/-- The score multiplier for a tetris -/
def tetris_multiplier : ℕ := 8

/-- The number of single lines Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Theorem: Tim's total score is 38000 points -/
theorem tim_total_score : 
  tim_singles * single_line_score + tim_tetrises * (tetris_multiplier * single_line_score) = 38000 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_score_l1229_122939


namespace NUMINAMATH_CALUDE_eulerian_circuit_iff_even_degree_l1229_122938

/-- A graph is a pair of a type of vertices and an edge relation -/
structure Graph (V : Type) :=
  (adj : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- An Eulerian circuit in a graph is a path that traverses every edge exactly once and returns to the starting vertex -/
def has_eulerian_circuit {V : Type} (G : Graph V) : Prop := sorry

/-- Theorem: A graph has an Eulerian circuit if and only if every vertex has even degree -/
theorem eulerian_circuit_iff_even_degree {V : Type} (G : Graph V) :
  has_eulerian_circuit G ↔ ∀ v : V, Even (degree G v) := by sorry

end NUMINAMATH_CALUDE_eulerian_circuit_iff_even_degree_l1229_122938


namespace NUMINAMATH_CALUDE_johns_score_less_than_winning_score_l1229_122903

/-- In a blackjack game, given the scores of three players and the winning score,
    prove that the score of the player who didn't win is less than the winning score. -/
theorem johns_score_less_than_winning_score 
  (theodore_score : ℕ) 
  (zoey_score : ℕ) 
  (john_score : ℕ) 
  (winning_score : ℕ) 
  (h1 : theodore_score = 13)
  (h2 : zoey_score = 19)
  (h3 : winning_score = 19)
  (h4 : zoey_score = winning_score)
  (h5 : john_score ≠ zoey_score) : 
  john_score < winning_score :=
sorry

end NUMINAMATH_CALUDE_johns_score_less_than_winning_score_l1229_122903


namespace NUMINAMATH_CALUDE_table_runner_coverage_l1229_122921

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (coverage_percentage : ℝ) (two_layer_area : ℝ) : 
  total_runner_area = 208 →
  table_area = 175 →
  coverage_percentage = 0.8 →
  two_layer_area = 24 →
  ∃ (three_layer_area : ℝ),
    three_layer_area = 22 ∧
    total_runner_area = (coverage_percentage * table_area - two_layer_area - three_layer_area) +
                        2 * two_layer_area +
                        3 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l1229_122921


namespace NUMINAMATH_CALUDE_recurrence_sequence_has_composite_l1229_122992

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) = 2 * a n + 1 ∨ a (n + 1) = 2 * a n - 1)

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- The main theorem stating that any sequence satisfying the recurrence relation contains a composite number -/
theorem recurrence_sequence_has_composite
  (a : ℕ → ℕ)
  (h_seq : RecurrenceSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_nonconstant : ∃ m n, m ≠ n ∧ a m ≠ a n) :
  ∃ k, IsComposite (a k) :=
sorry

end NUMINAMATH_CALUDE_recurrence_sequence_has_composite_l1229_122992


namespace NUMINAMATH_CALUDE_olgas_fish_colors_l1229_122928

theorem olgas_fish_colors (total : ℕ) (yellow : ℕ) (blue : ℕ) (green : ℕ)
  (h_total : total = 42)
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_green : green = yellow * 2)
  (h_sum : total = yellow + blue + green) :
  ∃ (num_colors : ℕ), num_colors = 3 ∧ num_colors > 0 := by
sorry

end NUMINAMATH_CALUDE_olgas_fish_colors_l1229_122928


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1229_122917

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 * b2 = a2 * b1) ∧ (a1 ≠ 0 ∨ a2 ≠ 0)

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  parallel_lines 1 (2*m) (-1) (m-2) (-m) 2 → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l1229_122917


namespace NUMINAMATH_CALUDE_student_count_l1229_122989

theorem student_count (initial_avg : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) (final_avg : ℚ) :
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  final_avg = 98 →
  ∃ n : ℕ, n > 0 ∧ n * final_avg = n * initial_avg - (wrong_mark - correct_mark) :=
by
  sorry

end NUMINAMATH_CALUDE_student_count_l1229_122989


namespace NUMINAMATH_CALUDE_half_area_to_longest_side_l1229_122943

/-- Represents a parallelogram field with given dimensions and angles -/
structure ParallelogramField where
  side1 : Real
  side2 : Real
  angle1 : Real
  angle2 : Real

/-- Calculates the fraction of the area closer to the longest side of the parallelogram field -/
def fraction_to_longest_side (field : ParallelogramField) : Real :=
  sorry

/-- Theorem stating that for a parallelogram field with specific dimensions,
    the fraction of the area closer to the longest side is 1/2 -/
theorem half_area_to_longest_side :
  let field : ParallelogramField := {
    side1 := 120,
    side2 := 80,
    angle1 := π / 3,  -- 60 degrees in radians
    angle2 := 2 * π / 3  -- 120 degrees in radians
  }
  fraction_to_longest_side field = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_area_to_longest_side_l1229_122943


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1229_122966

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x = 4 ∧ 
  (∀ (y : ℕ), (1100 + y) % 23 = 0 → y ≥ x) ∧ 
  (1100 + x) % 23 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1229_122966


namespace NUMINAMATH_CALUDE_tangent_line_implies_sum_l1229_122924

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that the tangent line at (1, f(1)) has equation x - 2y + 1 = 0
def has_tangent_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), (∀ x, m * x + b = f x) ∧ (m * 1 + b = f 1) ∧ (m = 1 / 2) ∧ (b = 1 / 2)

-- Theorem statement
theorem tangent_line_implies_sum (f : ℝ → ℝ) (h : has_tangent_line f) :
  f 1 + 2 * (deriv f 1) = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_sum_l1229_122924


namespace NUMINAMATH_CALUDE_divisible_by_99_l1229_122997

theorem divisible_by_99 (A B : ℕ) : 
  A < 10 → B < 10 → 
  99 ∣ (A * 100000 + 15000 + B * 100 + 94) → 
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_99_l1229_122997


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l1229_122956

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 7 →
  points_per_member = 4 →
  total_points = 20 →
  total_members - (total_points / points_per_member) = 2 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_absentees_l1229_122956


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1229_122933

theorem sum_of_two_numbers (x y : ℕ) : y = x + 4 → y = 30 → x + y = 56 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1229_122933


namespace NUMINAMATH_CALUDE_expression_equals_40_times_10_to_2003_l1229_122946

theorem expression_equals_40_times_10_to_2003 :
  (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_40_times_10_to_2003_l1229_122946


namespace NUMINAMATH_CALUDE_cubic_equation_value_l1229_122936

theorem cubic_equation_value (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 + 2006 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l1229_122936


namespace NUMINAMATH_CALUDE_investment_growth_l1229_122959

/-- The present value of an investment -/
def present_value : ℝ := 217474.41

/-- The future value of the investment -/
def future_value : ℝ := 600000

/-- The annual interest rate -/
def interest_rate : ℝ := 0.07

/-- The number of years for the investment -/
def years : ℕ := 15

/-- Theorem stating that the present value invested at the given interest rate
    for the specified number of years will result in the future value -/
theorem investment_growth (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  |future_value - present_value * (1 + interest_rate) ^ years| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_growth_l1229_122959


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1229_122955

-- Define the equation of motion
def s (t : ℝ) : ℝ := -t + t^2

-- Define the velocity function
def v (t : ℝ) : ℝ := (-1) + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1229_122955


namespace NUMINAMATH_CALUDE_miss_darlington_blueberries_l1229_122967

def blueberries_problem (initial_basket : ℕ) (additional_baskets : ℕ) : Prop :=
  let total_blueberries := initial_basket + additional_baskets * initial_basket
  total_blueberries = 200

theorem miss_darlington_blueberries : blueberries_problem 20 9 := by
  sorry

end NUMINAMATH_CALUDE_miss_darlington_blueberries_l1229_122967


namespace NUMINAMATH_CALUDE_problem_solution_l1229_122977

theorem problem_solution (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1229_122977


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l1229_122909

/-- A circle with center C(r,r) is tangent to the positive x-axis and y-axis,
    and externally tangent to a circle centered at (4,0) with radius 2.
    The sum of all possible radii of the circle with center C is 12. -/
theorem circle_tangent_sum_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r₁ > 0 ∧ r₂ > 0) ∧ 
    ((r₁ - 4)^2 + r₁^2 = (r₁ + 2)^2) ∧
    ((r₂ - 4)^2 + r₂^2 = (r₂ + 2)^2) ∧
    r₁ + r₂ = 12) :=
by
  sorry

#check circle_tangent_sum_radii

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l1229_122909


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l1229_122918

-- Define the angles
variable (A B C D E F : ℝ)

-- Define the theorem
theorem angle_sum_theorem (h : A + B + C + D + E + F = 90 * n) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l1229_122918


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1229_122990

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 9 meters, width 8 meters, and depth 5 meters is 314 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 9 8 5 = 314 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1229_122990


namespace NUMINAMATH_CALUDE_circle_area_equality_l1229_122962

theorem circle_area_equality (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  let r : Real := 1  -- Assuming unit circle for simplicity
  let sector_area := θ * r^2
  let triangle_area := (r^2 * Real.tan θ * Real.tan (2 * θ)) / 2
  let circle_area := π * r^2
  triangle_area = circle_area - sector_area ↔ 2 * θ = Real.tan θ * Real.tan (2 * θ) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1229_122962


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1229_122947

-- Define the sets A and B
def A : Set ℝ := {x | Real.log x ≥ 0}
def B : Set ℝ := {x | x^2 < 9}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1229_122947


namespace NUMINAMATH_CALUDE_triangle_inequality_l1229_122915

theorem triangle_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum : a + b + c ≤ 2) : 
  -3 < (a^3/b + b^3/c + c^3/a - a^3/c - b^3/a - c^3/b) ∧
  (a^3/b + b^3/c + c^3/a - a^3/c - b^3/a - c^3/b) < 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1229_122915


namespace NUMINAMATH_CALUDE_expression_simplification_l1229_122961

theorem expression_simplification (a b c : ℝ) :
  3 / 4 * (6 * a^2 - 12 * a) - 8 / 5 * (3 * b^2 + 15 * b) + (2 * c^2 - 6 * c) / 6 =
  (9/2) * a^2 - 9 * a - (24/5) * b^2 - 24 * b + (1/3) * c^2 - c :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1229_122961


namespace NUMINAMATH_CALUDE_line_translation_slope_l1229_122996

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a translation function
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy - l.slope * dx }

-- State the theorem
theorem line_translation_slope (l : Line) :
  translate l 3 2 = l → l.slope = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_slope_l1229_122996


namespace NUMINAMATH_CALUDE_paintings_distribution_l1229_122900

theorem paintings_distribution (total_paintings : ℕ) (num_rooms : ℕ) (paintings_per_room : ℕ) :
  total_paintings = 32 →
  num_rooms = 4 →
  total_paintings = num_rooms * paintings_per_room →
  paintings_per_room = 8 := by
  sorry

end NUMINAMATH_CALUDE_paintings_distribution_l1229_122900


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1229_122965

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1229_122965


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1229_122972

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1229_122972


namespace NUMINAMATH_CALUDE_sphere_volume_from_intersection_l1229_122945

/-- The volume of a sphere, given specific intersection properties -/
theorem sphere_volume_from_intersection (r : ℝ) : 
  (∃ (d : ℝ), d = 1 ∧ π = π * (r^2 - d^2)) →
  (4/3) * π * r^3 = (8 * Real.sqrt 2 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_intersection_l1229_122945
