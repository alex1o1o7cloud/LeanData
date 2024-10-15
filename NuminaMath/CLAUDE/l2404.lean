import Mathlib

namespace NUMINAMATH_CALUDE_pens_bought_theorem_l2404_240439

/-- The number of pens bought at the cost price -/
def num_pens_bought : ℕ := 17

/-- The number of pens sold to equal the cost price of the bought pens -/
def num_pens_sold : ℕ := 12

/-- The gain percentage -/
def gain_percentage : ℚ := 40/100

theorem pens_bought_theorem :
  ∀ (cost_price selling_price : ℚ),
  cost_price > 0 →
  selling_price > 0 →
  (num_pens_bought : ℚ) * cost_price = (num_pens_sold : ℚ) * selling_price →
  (selling_price - cost_price) / cost_price = gain_percentage →
  num_pens_bought = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_bought_theorem_l2404_240439


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_l2404_240489

/-- The cost of paint per kg for a cube with given conditions -/
theorem paint_cost_per_kg (coverage : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  coverage = 16 →
  total_cost = 876 →
  side_length = 8 →
  (total_cost / (6 * side_length^2 / coverage)) = 36.5 :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_l2404_240489


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2404_240435

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2404_240435


namespace NUMINAMATH_CALUDE_water_fountain_problem_l2404_240446

/-- The number of men needed to build a water fountain of a given length in a given number of days -/
def men_needed (length : ℝ) (days : ℝ) : ℝ :=
  sorry

theorem water_fountain_problem :
  let first_length : ℝ := 56
  let first_days : ℝ := 21
  let second_length : ℝ := 14
  let second_days : ℝ := 3
  let second_men : ℝ := 35

  (men_needed first_length first_days) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_water_fountain_problem_l2404_240446


namespace NUMINAMATH_CALUDE_complex_magnitude_two_thirds_plus_three_i_l2404_240402

theorem complex_magnitude_two_thirds_plus_three_i :
  Complex.abs (2/3 + 3*Complex.I) = Real.sqrt 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_two_thirds_plus_three_i_l2404_240402


namespace NUMINAMATH_CALUDE_book_words_per_page_l2404_240467

theorem book_words_per_page 
  (total_pages : ℕ) 
  (words_per_page : ℕ) 
  (max_words_per_page : ℕ) 
  (total_words_mod : ℕ) :
  total_pages = 150 →
  words_per_page ≤ max_words_per_page →
  max_words_per_page = 120 →
  (total_pages * words_per_page) % 221 = total_words_mod →
  total_words_mod = 200 →
  words_per_page = 118 := by
sorry

end NUMINAMATH_CALUDE_book_words_per_page_l2404_240467


namespace NUMINAMATH_CALUDE_two_point_distribution_max_value_l2404_240456

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  p : ℝ
  hp : 0 < p ∧ p < 1

/-- The expected value of a two-point distribution -/
def expectedValue (ξ : TwoPointDistribution) : ℝ := ξ.p

/-- The variance of a two-point distribution -/
def variance (ξ : TwoPointDistribution) : ℝ := ξ.p * (1 - ξ.p)

/-- The theorem stating the maximum value of (2D(ξ)-1)/E(ξ) for a two-point distribution -/
theorem two_point_distribution_max_value (ξ : TwoPointDistribution) :
  (∃ (c : ℝ), ∀ (η : TwoPointDistribution), (2 * variance η - 1) / expectedValue η ≤ c) ∧
  (∃ (ξ_max : TwoPointDistribution), (2 * variance ξ_max - 1) / expectedValue ξ_max = 2 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_two_point_distribution_max_value_l2404_240456


namespace NUMINAMATH_CALUDE_sphere_radius_calculation_l2404_240419

/-- Given a sphere on a horizontal plane, if a vertical stick casts a shadow and the sphere's shadow extends from its base, then we can calculate the radius of the sphere. -/
theorem sphere_radius_calculation (stick_height stick_shadow sphere_shadow : ℝ) 
  (stick_height_pos : stick_height > 0)
  (stick_shadow_pos : stick_shadow > 0)
  (sphere_shadow_pos : sphere_shadow > 0)
  (h_stick : stick_height = 1.5)
  (h_stick_shadow : stick_shadow = 1)
  (h_sphere_shadow : sphere_shadow = 8) :
  ∃ r : ℝ, r > 0 ∧ r / (sphere_shadow - r) = stick_height / stick_shadow ∧ r = 4.8 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_calculation_l2404_240419


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l2404_240437

/-- Represents the cost structure and group composition at a restaurant --/
structure RestaurantBill where
  adult_meal_costs : Fin 3 → ℕ
  adult_beverage_cost : ℕ
  kid_beverage_cost : ℕ
  total_people : ℕ
  kids_count : ℕ
  adult_meal_counts : Fin 3 → ℕ
  total_beverages : ℕ

/-- Calculates the total bill for a group at the restaurant --/
def calculate_total_bill (bill : RestaurantBill) : ℕ :=
  let adult_meals_cost := (bill.adult_meal_costs 0 * bill.adult_meal_counts 0) +
                          (bill.adult_meal_costs 1 * bill.adult_meal_counts 1) +
                          (bill.adult_meal_costs 2 * bill.adult_meal_counts 2)
  let adult_beverages_cost := min (bill.total_people - bill.kids_count) bill.total_beverages * bill.adult_beverage_cost
  let kid_beverages_cost := (bill.total_beverages - min (bill.total_people - bill.kids_count) bill.total_beverages) * bill.kid_beverage_cost
  adult_meals_cost + adult_beverages_cost + kid_beverages_cost

/-- Theorem stating that the total bill for the given group is $59 --/
theorem restaurant_bill_theorem (bill : RestaurantBill)
  (h1 : bill.adult_meal_costs = ![5, 7, 9])
  (h2 : bill.adult_beverage_cost = 2)
  (h3 : bill.kid_beverage_cost = 1)
  (h4 : bill.total_people = 14)
  (h5 : bill.kids_count = 7)
  (h6 : bill.adult_meal_counts = ![4, 2, 1])
  (h7 : bill.total_beverages = 9) :
  calculate_total_bill bill = 59 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l2404_240437


namespace NUMINAMATH_CALUDE_max_value_inequality_l2404_240418

theorem max_value_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : |1/a| + |1/b| + |1/c| ≤ 3) : 
  (a^2 + 4*(b^2 + c^2)) * (b^2 + 4*(a^2 + c^2)) * (c^2 + 4*(a^2 + b^2)) ≥ 729 ∧ 
  ∀ m > 729, ∃ a' b' c' : ℝ, a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ 
    |1/a'| + |1/b'| + |1/c'| ≤ 3 ∧
    (a'^2 + 4*(b'^2 + c'^2)) * (b'^2 + 4*(a'^2 + c'^2)) * (c'^2 + 4*(a'^2 + b'^2)) < m :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2404_240418


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2404_240466

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (|2*x - 3| < 1 → x*(x - 3) < 0)) ∧
  (∃ x : ℝ, x*(x - 3) < 0 ∧ ¬(|2*x - 3| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2404_240466


namespace NUMINAMATH_CALUDE_at_most_four_greater_than_one_l2404_240485

theorem at_most_four_greater_than_one 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : (Real.sqrt (a * b) - 1) * (Real.sqrt (b * c) - 1) * (Real.sqrt (c * a) - 1) = 1) : 
  ∃ (S : Finset ℝ), S ⊆ {a - b/c, a - c/b, b - a/c, b - c/a, c - a/b, c - b/a} ∧ 
    S.card ≤ 4 ∧ 
    (∀ x ∈ S, x > 1) ∧
    (∀ y ∈ {a - b/c, a - c/b, b - a/c, b - c/a, c - a/b, c - b/a} \ S, y ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_at_most_four_greater_than_one_l2404_240485


namespace NUMINAMATH_CALUDE_cos_103pi_4_l2404_240436

theorem cos_103pi_4 : Real.cos (103 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_103pi_4_l2404_240436


namespace NUMINAMATH_CALUDE_total_legs_bees_and_spiders_l2404_240465

theorem total_legs_bees_and_spiders :
  let bee_legs : ℕ := 6
  let spider_legs : ℕ := 8
  let num_bees : ℕ := 5
  let num_spiders : ℕ := 2
  (num_bees * bee_legs + num_spiders * spider_legs) = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_total_legs_bees_and_spiders_l2404_240465


namespace NUMINAMATH_CALUDE_abs_plus_square_zero_implies_sum_l2404_240428

theorem abs_plus_square_zero_implies_sum (x y : ℝ) :
  |x + 3| + (2*y - 5)^2 = 0 → x + 2*y = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_square_zero_implies_sum_l2404_240428


namespace NUMINAMATH_CALUDE_second_feeding_maggots_l2404_240444

/-- Given the total number of maggots served and the number of maggots in the first feeding,
    calculate the number of maggots in the second feeding. -/
def maggots_in_second_feeding (total_maggots : ℕ) (first_feeding : ℕ) : ℕ :=
  total_maggots - first_feeding

/-- Theorem stating that given 20 total maggots and 10 maggots in the first feeding,
    the number of maggots in the second feeding is 10. -/
theorem second_feeding_maggots :
  maggots_in_second_feeding 20 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_feeding_maggots_l2404_240444


namespace NUMINAMATH_CALUDE_max_profit_is_200_l2404_240453

/-- Represents a neighborhood with its characteristics --/
structure Neighborhood where
  homes : ℕ
  boxes_per_home : ℕ
  price_per_box : ℚ

/-- Calculates the total sales for a neighborhood --/
def total_sales (n : Neighborhood) : ℚ :=
  n.homes * n.boxes_per_home * n.price_per_box

/-- The four neighborhoods with their respective characteristics --/
def neighborhood_A : Neighborhood := ⟨12, 3, 3⟩
def neighborhood_B : Neighborhood := ⟨8, 6, 4⟩
def neighborhood_C : Neighborhood := ⟨15, 2, 5/2⟩
def neighborhood_D : Neighborhood := ⟨5, 8, 5⟩

/-- List of all neighborhoods --/
def neighborhoods : List Neighborhood := [neighborhood_A, neighborhood_B, neighborhood_C, neighborhood_D]

/-- Theorem stating that the maximum profit among the neighborhoods is $200 --/
theorem max_profit_is_200 : 
  (neighborhoods.map total_sales).maximum? = some 200 := by sorry

end NUMINAMATH_CALUDE_max_profit_is_200_l2404_240453


namespace NUMINAMATH_CALUDE_socks_in_washing_machine_l2404_240455

/-- The number of players in a soccer match -/
def num_players : ℕ := 11

/-- The number of socks each player wears -/
def socks_per_player : ℕ := 2

/-- The total number of socks in the washing machine -/
def total_socks : ℕ := num_players * socks_per_player

theorem socks_in_washing_machine : total_socks = 22 := by
  sorry

end NUMINAMATH_CALUDE_socks_in_washing_machine_l2404_240455


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l2404_240426

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17 / 4 ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l2404_240426


namespace NUMINAMATH_CALUDE_line_equation_proof_l2404_240464

-- Define the line l
def line_l : Set (ℝ × ℝ) := {(x, y) | 4*x - 3*y - 1 = 0}

-- Define the given line
def given_line : Set (ℝ × ℝ) := {(x, y) | 3*x + 4*y - 3 = 0}

-- Define the point A
def point_A : ℝ × ℝ := (-2, -3)

theorem line_equation_proof :
  -- Line l passes through point A
  point_A ∈ line_l ∧
  -- Line l is perpendicular to the given line
  (∀ (p q : ℝ × ℝ), p ∈ line_l → q ∈ line_l → p ≠ q →
    ∀ (r s : ℝ × ℝ), r ∈ given_line → s ∈ given_line → r ≠ s →
      ((p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2404_240464


namespace NUMINAMATH_CALUDE_shelves_per_closet_l2404_240454

/-- Given the following constraints for stacking cans in a closet:
  * 12 cans fit in one row
  * 4 rows fit on one shelf
  * 480 cans can be stored in one closet
  Prove that 10 shelves can fit in one closet -/
theorem shelves_per_closet (cans_per_row : ℕ) (rows_per_shelf : ℕ) (cans_per_closet : ℕ)
  (h1 : cans_per_row = 12)
  (h2 : rows_per_shelf = 4)
  (h3 : cans_per_closet = 480) :
  cans_per_closet / (cans_per_row * rows_per_shelf) = 10 := by
  sorry

end NUMINAMATH_CALUDE_shelves_per_closet_l2404_240454


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2404_240483

theorem unique_prime_solution : 
  ∃! (p m : ℕ), 
    Prime p ∧ 
    m > 0 ∧ 
    p^3 + m*(p + 2) = m^2 + p + 1 ∧ 
    p = 2 ∧ 
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2404_240483


namespace NUMINAMATH_CALUDE_prime_divisor_implies_equal_l2404_240422

theorem prime_divisor_implies_equal (m n : ℕ) : 
  Prime (m + n + 1) → 
  (m + n + 1) ∣ (2 * (m^2 + n^2) - 1) → 
  m = n :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_implies_equal_l2404_240422


namespace NUMINAMATH_CALUDE_min_value_at_three_l2404_240470

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The statement that x = 3 minimizes the function f -/
theorem min_value_at_three :
  ∀ x : ℝ, f 3 ≤ f x :=
by
  sorry

#check min_value_at_three

end NUMINAMATH_CALUDE_min_value_at_three_l2404_240470


namespace NUMINAMATH_CALUDE_unique_quadratic_pair_l2404_240477

/-- A function that checks if a quadratic equation has exactly one real solution -/
def hasExactlyOneRealSolution (a b c : ℤ) : Prop :=
  b * b = 4 * a * c

/-- The theorem stating that there exists exactly one ordered pair (b,c) satisfying the conditions -/
theorem unique_quadratic_pair :
  ∃! (b c : ℕ), 
    0 < b ∧ b ≤ 6 ∧
    0 < c ∧ c ≤ 6 ∧
    hasExactlyOneRealSolution 1 b c ∧
    hasExactlyOneRealSolution 1 c b :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_pair_l2404_240477


namespace NUMINAMATH_CALUDE_largest_argument_l2404_240434

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z - 10i| = 5√2
def satisfies_condition (z : ℂ) : Prop :=
  Complex.abs (z - Complex.I * 10) = 5 * Real.sqrt 2

-- Define the theorem
theorem largest_argument :
  ∃ (z : ℂ), satisfies_condition z ∧
  ∀ (w : ℂ), satisfies_condition w → Complex.arg w ≤ Complex.arg z ∧
  z = -5 + 5 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_largest_argument_l2404_240434


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2404_240472

/-- Proves that the weight of the replaced person is 45 kg given the conditions -/
theorem weight_of_replaced_person
  (n : ℕ)
  (original_average : ℝ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 65)
  : ∃ (replaced_weight : ℝ),
    n * (original_average + weight_increase) - n * original_average
    = new_person_weight - replaced_weight
    ∧ replaced_weight = 45 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2404_240472


namespace NUMINAMATH_CALUDE_range_of_a_l2404_240425

-- Define the propositions p and q
def p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4
def q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, q x → p x a) →  -- q is a sufficient condition for p
  -1 ≤ a ∧ a ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2404_240425


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2404_240438

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₁ + a₉ = 10, then a₅ = 5 -/
theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2404_240438


namespace NUMINAMATH_CALUDE_jons_textbooks_weight_l2404_240482

theorem jons_textbooks_weight (brandon_weight : ℝ) (jon_weight : ℝ) : 
  brandon_weight = 8 → jon_weight = 3 * brandon_weight → jon_weight = 24 := by
  sorry

end NUMINAMATH_CALUDE_jons_textbooks_weight_l2404_240482


namespace NUMINAMATH_CALUDE_work_completion_time_l2404_240461

/-- The number of days it takes for a group to complete a work -/
def days_to_complete (women : ℕ) (children : ℕ) : ℚ :=
  1 / ((women / 50 : ℚ) + (children / 100 : ℚ))

/-- The theorem stating that 5 women and 10 children working together will complete the work in 5 days -/
theorem work_completion_time :
  days_to_complete 5 10 = 5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2404_240461


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2404_240494

theorem ratio_x_to_y (x y : ℚ) (h : (7 * x - 4 * y) / (20 * x - 3 * y) = 4 / 9) :
  x / y = -24 / 17 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2404_240494


namespace NUMINAMATH_CALUDE_jessica_test_score_l2404_240451

-- Define the given conditions
def initial_students : ℕ := 20
def initial_average : ℚ := 75
def new_students : ℕ := 21
def new_average : ℚ := 76

-- Define Jessica's score as a variable
def jessica_score : ℚ := sorry

-- Theorem to prove
theorem jessica_test_score : 
  (initial_students * initial_average + jessica_score) / new_students = new_average := by
  sorry

end NUMINAMATH_CALUDE_jessica_test_score_l2404_240451


namespace NUMINAMATH_CALUDE_farthest_line_from_origin_l2404_240499

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The point A(1,2) -/
def pointA : Point := ⟨1, 2⟩

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The line x + 2y - 5 = 0 -/
def targetLine : Line := ⟨1, 2, -5⟩

theorem farthest_line_from_origin : 
  (pointOnLine pointA targetLine) ∧ 
  (∀ l : Line, pointOnLine pointA l → distancePointToLine origin targetLine ≥ distancePointToLine origin l) :=
sorry

end NUMINAMATH_CALUDE_farthest_line_from_origin_l2404_240499


namespace NUMINAMATH_CALUDE_difference_divisible_by_19_l2404_240492

theorem difference_divisible_by_19 (n : ℕ) : 26^n ≡ 7^n [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_19_l2404_240492


namespace NUMINAMATH_CALUDE_square_perimeter_difference_l2404_240429

theorem square_perimeter_difference (a b : ℝ) 
  (h1 : a^2 + b^2 = 85)
  (h2 : a^2 - b^2 = 45) :
  4*a - 4*b = 4*(Real.sqrt 65 - 2*Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_difference_l2404_240429


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2404_240452

/-- Calculates the total cost of production given fixed cost, marginal cost, and number of products. -/
def totalCost (fixedCost marginalCost : ℕ) (numProducts : ℕ) : ℕ :=
  fixedCost + marginalCost * numProducts

/-- Proves that the total cost of producing 20 products is $16,000, given a fixed cost of $12,000 and a marginal cost of $200 per product. -/
theorem total_cost_calculation :
  totalCost 12000 200 20 = 16000 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2404_240452


namespace NUMINAMATH_CALUDE_quadratic_function_a_range_l2404_240486

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_a_range 
  (a b c : ℝ) 
  (h1 : f a b c (-2) = 1) 
  (h2 : f a b c 2 = 3) 
  (h3 : 0 < c) 
  (h4 : c < 1) : 
  1/4 < a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_a_range_l2404_240486


namespace NUMINAMATH_CALUDE_no_possible_values_for_a_l2404_240447

def M (a : ℝ) : Set ℝ := {1, 9, a}
def P (a : ℝ) : Set ℝ := {1, a, 2}

theorem no_possible_values_for_a :
  ∀ a : ℝ, (P a) ⊆ (M a) → False :=
sorry

end NUMINAMATH_CALUDE_no_possible_values_for_a_l2404_240447


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2404_240404

def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_minimum (a : ℕ → ℝ) (h1 : isGeometric a)
  (h2 : ∀ n : ℕ, a n > 0)
  (h3 : ∃ m n : ℕ, Real.sqrt (a m * a n) = 8 * a 1)
  (h4 : a 9 = a 8 + 2 * a 7) :
  (∃ m n : ℕ, 1 / m + 4 / n = 17 / 15) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 17 / 15) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2404_240404


namespace NUMINAMATH_CALUDE_series_sum_minus_eight_l2404_240407

theorem series_sum_minus_eight : 
  (5/3 + 13/9 + 41/27 + 125/81 + 379/243 + 1145/729) - 8 = 950/729 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_minus_eight_l2404_240407


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2404_240496

theorem complete_square_quadratic (a b c : ℝ) (h : a = 1 ∧ b = -6 ∧ c = -16) :
  ∃ (k m : ℝ), ∀ x, (a * x^2 + b * x + c = 0) ↔ ((x + k)^2 = m) ∧ m = 25 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2404_240496


namespace NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l2404_240445

/-- Calculate the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The number of trailing zeroes in 500! + 200! is 124 -/
theorem trailing_zeroes_sum_factorials :
  max (trailingZeroes 500) (trailingZeroes 200) = 124 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l2404_240445


namespace NUMINAMATH_CALUDE_reading_pages_solution_l2404_240441

/-- The number of pages Xiao Ming's father reads per day -/
def father_pages : ℕ := sorry

/-- The number of pages Xiao Ming reads per day -/
def xiao_ming_pages : ℕ := sorry

/-- Xiao Ming reads 5 pages more than his father every day -/
axiom pages_difference : xiao_ming_pages = father_pages + 5

/-- The time it takes for Xiao Ming to read 100 pages is equal to the time it takes for his father to read 80 pages -/
axiom reading_time_equality : (100 : ℚ) / xiao_ming_pages = (80 : ℚ) / father_pages

theorem reading_pages_solution :
  father_pages = 20 ∧ xiao_ming_pages = 25 :=
sorry

end NUMINAMATH_CALUDE_reading_pages_solution_l2404_240441


namespace NUMINAMATH_CALUDE_infinitely_many_close_fractions_l2404_240423

theorem infinitely_many_close_fractions (x : ℝ) (hx_pos : x > 0) (hx_irrational : ¬ ∃ (a b : ℤ), x = a / b) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| ≤ 1 / q^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_close_fractions_l2404_240423


namespace NUMINAMATH_CALUDE_probability_of_flush_l2404_240469

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- Size of a poker hand -/
def HandSize : ℕ := 5

/-- Probability of drawing a flush in a 5-card poker hand -/
theorem probability_of_flush (deck : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (hand_size : ℕ) :
  deck = StandardDeck →
  suits = NumSuits →
  cards_per_suit = CardsPerSuit →
  hand_size = HandSize →
  (suits * (Nat.choose cards_per_suit hand_size) : ℚ) / (Nat.choose deck hand_size) = 33 / 16660 :=
sorry

end NUMINAMATH_CALUDE_probability_of_flush_l2404_240469


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l2404_240475

theorem not_all_perfect_squares (k : ℕ) : ¬(∃ a b c : ℤ, (2 * k - 1 = a^2) ∧ (5 * k - 1 = b^2) ∧ (13 * k - 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l2404_240475


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2404_240417

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2404_240417


namespace NUMINAMATH_CALUDE_tim_total_score_l2404_240401

/-- The score for a single line in Tetris -/
def single_line_score : ℕ := 1000

/-- The score for a Tetris (four lines cleared at once) -/
def tetris_score : ℕ := 8 * single_line_score

/-- Tim's number of single lines cleared -/
def tim_singles : ℕ := 6

/-- Tim's number of Tetrises -/
def tim_tetrises : ℕ := 4

/-- Theorem stating Tim's total score -/
theorem tim_total_score : tim_singles * single_line_score + tim_tetrises * tetris_score = 38000 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_score_l2404_240401


namespace NUMINAMATH_CALUDE_sample_size_correct_l2404_240460

/-- The sample size that satisfies the given conditions -/
def sample_size : ℕ := 6

/-- The total population size -/
def total_population : ℕ := 36

/-- Theorem stating that the sample size satisfies all conditions -/
theorem sample_size_correct : 
  (sample_size ∣ total_population) ∧ 
  (6 ∣ sample_size) ∧
  (∃ k : ℕ, 35 = k * (sample_size + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sample_size_correct_l2404_240460


namespace NUMINAMATH_CALUDE_walking_students_speed_l2404_240484

/-- Two students walking towards each other -/
structure WalkingStudents where
  distance : ℝ
  time : ℝ
  speed1 : ℝ
  speed2 : ℝ

/-- The conditions of the problem -/
def problem : WalkingStudents where
  distance := 350
  time := 100
  speed1 := 1.9
  speed2 := 1.6  -- The speed we want to prove

theorem walking_students_speed (w : WalkingStudents) 
  (h1 : w.distance = 350)
  (h2 : w.time = 100)
  (h3 : w.speed1 = 1.9)
  (h4 : w.speed2 * w.time + w.speed1 * w.time = w.distance) :
  w.speed2 = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_walking_students_speed_l2404_240484


namespace NUMINAMATH_CALUDE_P_not_in_second_quadrant_l2404_240468

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The coordinates of point P as a function of m -/
def P (m : ℝ) : ℝ × ℝ := (m^2 + m, m - 1)

/-- Theorem stating that P(m) cannot be in the second quadrant for any real m -/
theorem P_not_in_second_quadrant (m : ℝ) : ¬ second_quadrant (P m).1 (P m).2 := by
  sorry

end NUMINAMATH_CALUDE_P_not_in_second_quadrant_l2404_240468


namespace NUMINAMATH_CALUDE_carlas_classroom_desks_full_l2404_240432

/-- Represents the classroom setup and attendance for Carla's sixth-grade class -/
structure Classroom where
  total_students : ℕ
  restroom_students : ℕ
  rows : ℕ
  desks_per_row : ℕ

/-- Calculates the fraction of desks that are full in the classroom -/
def fraction_of_desks_full (c : Classroom) : ℚ :=
  let absent_students := 3 * c.restroom_students - 1
  let students_in_classroom := c.total_students - absent_students - c.restroom_students
  let total_desks := c.rows * c.desks_per_row
  (students_in_classroom : ℚ) / (total_desks : ℚ)

/-- Theorem stating that the fraction of desks full in Carla's classroom is 2/3 -/
theorem carlas_classroom_desks_full :
  ∃ (c : Classroom), c.total_students = 23 ∧ c.restroom_students = 2 ∧ c.rows = 4 ∧ c.desks_per_row = 6 ∧
  fraction_of_desks_full c = 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_carlas_classroom_desks_full_l2404_240432


namespace NUMINAMATH_CALUDE_proposition_implication_l2404_240459

theorem proposition_implication (P : ℕ → Prop) :
  (∀ k : ℕ, k ≥ 1 → (P k → P (k + 1))) →
  (¬ P 10) →
  (¬ P 9) := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l2404_240459


namespace NUMINAMATH_CALUDE_fractional_square_gt_floor_square_l2404_240497

theorem fractional_square_gt_floor_square (x : ℝ) (hx : x > 0) :
  (x ^ 2 - ⌊x ^ 2⌋) > (⌊x⌋ ^ 2) ↔ ∃ n : ℤ, Real.sqrt (n ^ 2 + 1) ≤ x ∧ x < n + 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_square_gt_floor_square_l2404_240497


namespace NUMINAMATH_CALUDE_equation_solution_range_l2404_240424

theorem equation_solution_range (x m : ℝ) : 9^x + 4 * 3^x - m = 0 → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2404_240424


namespace NUMINAMATH_CALUDE_seashell_collection_count_l2404_240480

theorem seashell_collection_count (initial_count additional_count : ℕ) :
  initial_count = 19 → additional_count = 6 →
  initial_count + additional_count = 25 :=
by sorry

end NUMINAMATH_CALUDE_seashell_collection_count_l2404_240480


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2404_240476

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ y, f y = (y + 2 + y⁻¹)^3) ∧ 
  (∃ c : ℝ, ∀ z ≠ 0, f z = c + z * (z⁻¹ * (f z - c))) ∧ 
  (∃ c : ℝ, ∀ z ≠ 0, f z = c + z * (z⁻¹ * (f z - c)) ∧ c = 20) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2404_240476


namespace NUMINAMATH_CALUDE_curve_symmetric_line_k_l2404_240457

/-- The curve equation --/
def curve (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line equation --/
def line (k x y : ℝ) : Prop :=
  k*x + 2*y - 4 = 0

/-- Two points are symmetric with respect to a line --/
def symmetric (P Q : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧ 
    (P.1 + Q.1 = 2*x) ∧ (P.2 + Q.2 = 2*y)

theorem curve_symmetric_line_k (P Q : ℝ × ℝ) (k : ℝ) :
  P ≠ Q →
  curve P.1 P.2 →
  curve Q.1 Q.2 →
  symmetric P Q k →
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetric_line_k_l2404_240457


namespace NUMINAMATH_CALUDE_meshed_gears_angular_velocity_ratio_l2404_240463

structure Gear where
  teeth : ℕ
  angularVelocity : ℝ

/-- The ratio of angular velocities for three meshed gears is proportional to the product of the other two gears' teeth counts. -/
theorem meshed_gears_angular_velocity_ratio 
  (A B C : Gear) 
  (h_mesh : A.angularVelocity * A.teeth = B.angularVelocity * B.teeth ∧ 
            B.angularVelocity * B.teeth = C.angularVelocity * C.teeth) :
  A.angularVelocity / (B.teeth * C.teeth) = 
  B.angularVelocity / (A.teeth * C.teeth) ∧
  B.angularVelocity / (A.teeth * C.teeth) = 
  C.angularVelocity / (A.teeth * B.teeth) :=
by sorry

end NUMINAMATH_CALUDE_meshed_gears_angular_velocity_ratio_l2404_240463


namespace NUMINAMATH_CALUDE_robin_photos_count_l2404_240430

/-- Given that each page holds six photos and Robin can fill 122 full pages,
    prove that Robin has 732 photos in total. -/
theorem robin_photos_count :
  let photos_per_page : ℕ := 6
  let full_pages : ℕ := 122
  photos_per_page * full_pages = 732 :=
by sorry

end NUMINAMATH_CALUDE_robin_photos_count_l2404_240430


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2404_240449

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2404_240449


namespace NUMINAMATH_CALUDE_instrumental_measurements_insufficient_l2404_240493

-- Define the concept of instrumental measurements
def InstrumentalMeasurement : Type := Unit

-- Define the concept of general geometric statements
def GeneralGeometricStatement : Type := Unit

-- Define the property of being approximate
def is_approximate (m : InstrumentalMeasurement) : Prop := sorry

-- Define the property of applying to infinite configurations
def applies_to_infinite_configurations (s : GeneralGeometricStatement) : Prop := sorry

-- Define the property of being performed on a finite number of instances
def performed_on_finite_instances (m : InstrumentalMeasurement) : Prop := sorry

-- Theorem stating that instrumental measurements are insufficient to justify general geometric statements
theorem instrumental_measurements_insufficient 
  (m : InstrumentalMeasurement) 
  (s : GeneralGeometricStatement) : 
  is_approximate m → 
  applies_to_infinite_configurations s → 
  performed_on_finite_instances m → 
  ¬(∃ (justification : Unit), True) := by sorry

end NUMINAMATH_CALUDE_instrumental_measurements_insufficient_l2404_240493


namespace NUMINAMATH_CALUDE_percentage_spent_l2404_240414

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 1200)
  (h2 : remaining_amount = 840) :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_l2404_240414


namespace NUMINAMATH_CALUDE_pizza_cost_per_slice_l2404_240411

-- Define the pizza and topping costs
def large_pizza_cost : ℚ := 10
def first_topping_cost : ℚ := 2
def next_two_toppings_cost : ℚ := 1
def remaining_toppings_cost : ℚ := 0.5

-- Define the number of slices and toppings
def num_slices : ℕ := 8
def num_toppings : ℕ := 7

-- Calculate the total cost of toppings
def total_toppings_cost : ℚ :=
  first_topping_cost +
  2 * next_two_toppings_cost +
  (num_toppings - 3) * remaining_toppings_cost

-- Calculate the total cost of the pizza
def total_pizza_cost : ℚ := large_pizza_cost + total_toppings_cost

-- Theorem to prove
theorem pizza_cost_per_slice :
  total_pizza_cost / num_slices = 2 := by sorry

end NUMINAMATH_CALUDE_pizza_cost_per_slice_l2404_240411


namespace NUMINAMATH_CALUDE_solve_system_l2404_240474

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2404_240474


namespace NUMINAMATH_CALUDE_at_least_100_odd_population_days_l2404_240412

/-- Represents the state of the Martian population on a given day -/
structure PopulationState :=
  (day : ℕ)
  (births : ℕ)
  (population : ℕ)

/-- A function that calculates the population state for each day -/
def populationEvolution : ℕ → PopulationState → PopulationState :=
  sorry

/-- The total number of Martians born throughout history -/
def totalBirths : ℕ := sorry

/-- Theorem stating that there are at least 100 days with odd population -/
theorem at_least_100_odd_population_days
  (h_odd_births : Odd totalBirths)
  (h_lifespan : ∀ (m : ℕ), m < totalBirths → ∃ (b d : ℕ), d - b = 100 ∧ PopulationState.population (populationEvolution d (PopulationState.mk b 1 1)) = PopulationState.population (populationEvolution (d + 1) (PopulationState.mk b 1 1)) - 1) :
  ∃ (S : Finset ℕ), S.card ≥ 100 ∧ ∀ (d : ℕ), d ∈ S → Odd (PopulationState.population (populationEvolution d (PopulationState.mk 0 0 0))) :=
sorry

end NUMINAMATH_CALUDE_at_least_100_odd_population_days_l2404_240412


namespace NUMINAMATH_CALUDE_intersection_complement_l2404_240487

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem intersection_complement : S ∩ (U \ T) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_l2404_240487


namespace NUMINAMATH_CALUDE_draw_probability_value_l2404_240409

/-- The number of green chips in the bag -/
def green_chips : ℕ := 4

/-- The number of blue chips in the bag -/
def blue_chips : ℕ := 3

/-- The number of yellow chips in the bag -/
def yellow_chips : ℕ := 5

/-- The total number of chips in the bag -/
def total_chips : ℕ := green_chips + blue_chips + yellow_chips

/-- The number of ways to arrange the color groups (green-blue-yellow or yellow-green-blue) -/
def color_group_arrangements : ℕ := 2

/-- The probability of drawing the chips in the specified order -/
def draw_probability : ℚ :=
  (Nat.factorial green_chips * Nat.factorial blue_chips * Nat.factorial yellow_chips * color_group_arrangements : ℚ) /
  Nat.factorial total_chips

theorem draw_probability_value : draw_probability = 1 / 13860 := by
  sorry

end NUMINAMATH_CALUDE_draw_probability_value_l2404_240409


namespace NUMINAMATH_CALUDE_roses_in_garden_l2404_240440

/-- Proves that the number of roses in the garden before cutting is equal to
    the final number of roses in the vase minus the initial number of roses in the vase. -/
theorem roses_in_garden (initial_vase : ℕ) (cut_from_garden : ℕ) (final_vase : ℕ)
  (h1 : initial_vase = 7)
  (h2 : cut_from_garden = 13)
  (h3 : final_vase = 20)
  (h4 : final_vase = initial_vase + cut_from_garden) :
  cut_from_garden = final_vase - initial_vase :=
by sorry

end NUMINAMATH_CALUDE_roses_in_garden_l2404_240440


namespace NUMINAMATH_CALUDE_functional_equation_l2404_240403

-- Define the function f
def f (x : ℝ) : ℝ := 1 - x^2

-- State the theorem
theorem functional_equation (x : ℝ) : x^2 * f x + f (1 - x) = 2*x - x^4 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_l2404_240403


namespace NUMINAMATH_CALUDE_expression_value_at_eight_l2404_240448

theorem expression_value_at_eight :
  let x : ℝ := 8
  (x^6 - 64*x^3 + 1024) / (x^3 - 16) = 480 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_eight_l2404_240448


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2404_240481

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (stream_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  stream_speed = 5 →
  downstream_distance = 70 →
  downstream_time = 2 →
  ∃ (boat_speed : ℝ), boat_speed = 30 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2404_240481


namespace NUMINAMATH_CALUDE_tan_beta_value_l2404_240490

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -2) 
  (h2 : Real.tan (α + β) = 1) : 
  Real.tan β = -3 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2404_240490


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2404_240471

/-- The radius of a circle tangent to four semicircles in a square -/
theorem tangent_circle_radius (s : ℝ) (h : s = 4) : 
  let r := 2 * (Real.sqrt 2 - 1)
  let semicircle_radius := s / 2
  let square_diagonal := s * Real.sqrt 2
  r = square_diagonal / 2 - semicircle_radius :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2404_240471


namespace NUMINAMATH_CALUDE_function_max_min_sum_l2404_240498

theorem function_max_min_sum (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f := fun x : ℝ => (5 * a^x + 1) / (a^x - 1) + Real.log (Real.sqrt (1 + x^2) - x)
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, N ≤ f x) ∧ M + N = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_sum_l2404_240498


namespace NUMINAMATH_CALUDE_solve_equation_l2404_240478

theorem solve_equation : ∃ y : ℚ, (2 / 7) * (1 / 8) * y = 12 ∧ y = 336 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2404_240478


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l2404_240431

theorem divisibility_by_eleven (n : ℕ) (a b c d e : ℕ) 
  (h1 : n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)
  (h2 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) : 
  n ≡ (a + c + e) - (b + d) [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l2404_240431


namespace NUMINAMATH_CALUDE_sum_of_coordinates_B_l2404_240495

/-- Given points A(0, 0) and B(x, 3) where the slope of AB is 3/4, 
    prove that the sum of B's coordinates is 7. -/
theorem sum_of_coordinates_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 →
  x + 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_B_l2404_240495


namespace NUMINAMATH_CALUDE_ninth_grade_class_distribution_l2404_240400

theorem ninth_grade_class_distribution (total students_science students_programming : ℕ) 
  (h_total : total = 120)
  (h_science : students_science = 80)
  (h_programming : students_programming = 75) :
  students_science - (total - (students_science + students_programming - total)) = 45 :=
sorry

end NUMINAMATH_CALUDE_ninth_grade_class_distribution_l2404_240400


namespace NUMINAMATH_CALUDE_gcd_210_378_l2404_240413

theorem gcd_210_378 : Nat.gcd 210 378 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_210_378_l2404_240413


namespace NUMINAMATH_CALUDE_initial_girls_count_l2404_240442

theorem initial_girls_count (initial_boys : ℕ) (boys_left : ℕ) (girls_entered : ℕ) (final_total : ℕ) :
  initial_boys = 5 →
  boys_left = 3 →
  girls_entered = 2 →
  final_total = 8 →
  ∃ initial_girls : ℕ, 
    initial_girls = 4 ∧
    final_total = (initial_boys - boys_left) + (initial_girls + girls_entered) :=
by sorry

end NUMINAMATH_CALUDE_initial_girls_count_l2404_240442


namespace NUMINAMATH_CALUDE_room_population_problem_l2404_240420

theorem room_population_problem (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →  -- Initial ratio of men to women is 4:5
  (initial_men + 2) = 14 →  -- After 2 men entered, there are now 14 men
  (2 * (initial_women - 3)) = 24 :=  -- Number of women after changes
by
  sorry

end NUMINAMATH_CALUDE_room_population_problem_l2404_240420


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2404_240421

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c (x + 5) = 4 * x^2 + 9 * x + 2) →
  a + b + c = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2404_240421


namespace NUMINAMATH_CALUDE_skew_lines_cannot_both_project_to_points_l2404_240427

/-- Two lines in 3D space are skew -/
def are_skew (l1 l2 : Line3) : Prop := sorry

/-- A line in 3D space -/
def Line3 : Type := sorry

/-- A plane in 3D space -/
def Plane3 : Type := sorry

/-- The projection of a line onto a plane -/
def project_line_to_plane (l : Line3) (p : Plane3) : Set Point := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3) (p : Plane3) : Prop := sorry

theorem skew_lines_cannot_both_project_to_points (a b : Line3) (α : Plane3) 
  (h_skew : are_skew a b) :
  ¬(∃ (pa pb : Point), project_line_to_plane a α = {pa} ∧ project_line_to_plane b α = {pb}) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_cannot_both_project_to_points_l2404_240427


namespace NUMINAMATH_CALUDE_ed_length_l2404_240462

/-- Given five points in a plane with specific distances between them, prove that ED = 74 -/
theorem ed_length (A B C D E : EuclideanSpace ℝ (Fin 2)) 
  (h_AB : dist A B = 12)
  (h_BC : dist B C = 50)
  (h_CD : dist C D = 38)
  (h_AD : dist A D = 100)
  (h_BE : dist B E = 30)
  (h_CE : dist C E = 40) :
  dist E D = 74 := by
  sorry

end NUMINAMATH_CALUDE_ed_length_l2404_240462


namespace NUMINAMATH_CALUDE_chen_pushups_l2404_240406

/-- The number of push-ups done by Chen -/
def chen : ℕ := sorry

/-- The number of push-ups done by Ruan -/
def ruan : ℕ := sorry

/-- The number of push-ups done by Lu -/
def lu : ℕ := sorry

/-- The number of push-ups done by Tao -/
def tao : ℕ := sorry

/-- The number of push-ups done by Yang -/
def yang : ℕ := sorry

/-- Chen, Lu, and Yang together averaged 40 push-ups per person -/
axiom condition1 : chen + lu + yang = 40 * 3

/-- Ruan, Tao, and Chen together averaged 28 push-ups per person -/
axiom condition2 : ruan + tao + chen = 28 * 3

/-- Ruan, Lu, Tao, and Yang together averaged 33 push-ups per person -/
axiom condition3 : ruan + lu + tao + yang = 33 * 4

theorem chen_pushups : chen = 36 := by
  sorry

end NUMINAMATH_CALUDE_chen_pushups_l2404_240406


namespace NUMINAMATH_CALUDE_exp_ge_x_plus_one_l2404_240450

theorem exp_ge_x_plus_one : ∀ x : ℝ, Real.exp x ≥ x + 1 := by
  sorry

end NUMINAMATH_CALUDE_exp_ge_x_plus_one_l2404_240450


namespace NUMINAMATH_CALUDE_fenced_area_with_cutouts_l2404_240405

/-- The area of a fenced region with cutouts -/
theorem fenced_area_with_cutouts :
  let rectangle_length : ℝ := 20
  let rectangle_width : ℝ := 18
  let square_side : ℝ := 4
  let triangle_leg : ℝ := 3
  let rectangle_area := rectangle_length * rectangle_width
  let square_cutout_area := square_side * square_side
  let triangle_cutout_area := (1 / 2) * triangle_leg * triangle_leg
  rectangle_area - square_cutout_area - triangle_cutout_area = 339.5 := by
sorry

end NUMINAMATH_CALUDE_fenced_area_with_cutouts_l2404_240405


namespace NUMINAMATH_CALUDE_first_ring_hexagons_fiftieth_ring_hexagons_nth_ring_hexagons_l2404_240433

/-- The number of hexagons in the nth ring around a central hexagon in a hexagonal tiling -/
def hexagons_in_nth_ring (n : ℕ) : ℕ := 6 * n

/-- The first ring contains 6 hexagons -/
theorem first_ring_hexagons : hexagons_in_nth_ring 1 = 6 := by sorry

/-- The 50th ring contains 300 hexagons -/
theorem fiftieth_ring_hexagons : hexagons_in_nth_ring 50 = 300 := by sorry

/-- For any natural number n, the nth ring contains 6n hexagons -/
theorem nth_ring_hexagons (n : ℕ) : hexagons_in_nth_ring n = 6 * n := by sorry

end NUMINAMATH_CALUDE_first_ring_hexagons_fiftieth_ring_hexagons_nth_ring_hexagons_l2404_240433


namespace NUMINAMATH_CALUDE_largest_angle_in_isosceles_triangle_l2404_240416

-- Define an isosceles triangle with one angle of 50°
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a = b ∧ a = 50

-- Theorem statement
theorem largest_angle_in_isosceles_triangle 
  {a b c : ℝ} (h : IsoscelesTriangle a b c) : 
  max a (max b c) = 80 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_isosceles_triangle_l2404_240416


namespace NUMINAMATH_CALUDE_nine_chapters_problem_l2404_240415

theorem nine_chapters_problem (x y : ℕ) :
  y = 2*x + 9 ∧ y = 3*(x - 2) ↔ 
  (∃ (filled_cars : ℕ), 
    x = filled_cars + 2 ∧ 
    y = 3 * filled_cars) :=
sorry

end NUMINAMATH_CALUDE_nine_chapters_problem_l2404_240415


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2404_240479

theorem inclination_angle_range (θ : ℝ) :
  let k := Real.cos θ
  let α := Real.arctan k
  α ∈ Set.Icc 0 (π / 4) ∪ Set.Ico (3 * π / 4) π :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2404_240479


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2404_240458

/-- The focal length of a hyperbola with equation x²/a² - y² = 1,
    where one of its asymptotes is perpendicular to the line 3x + y + 1 = 0,
    is equal to 2√10. -/
theorem hyperbola_focal_length (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 = 1) →
  (∃ (m : ℝ), m * (-1/3) = -1 ∧ y = m * x) →
  2 * Real.sqrt (1 + a^2) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2404_240458


namespace NUMINAMATH_CALUDE_alarm_system_probability_l2404_240443

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) :
  let prob_at_least_one := 1 - (1 - p) * (1 - p)
  prob_at_least_one = 0.64 := by
sorry

end NUMINAMATH_CALUDE_alarm_system_probability_l2404_240443


namespace NUMINAMATH_CALUDE_plot_length_l2404_240491

/-- Proves that the length of a rectangular plot is 55 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  (breadth + 10 = breadth + 10) →  -- Length is 10 more than breadth
  (cost_per_meter = 26.5) →        -- Cost per meter is 26.50 rupees
  (total_cost = 5300) →            -- Total cost is 5300 rupees
  (4 * breadth + 20) * cost_per_meter = total_cost →  -- Perimeter calculation
  (breadth + 10 = 55) :=            -- Length of the plot is 55 meters
by sorry

end NUMINAMATH_CALUDE_plot_length_l2404_240491


namespace NUMINAMATH_CALUDE_no_valid_pop_of_223_l2404_240408

/-- Represents the population of Minerva -/
structure MinervaPop where
  people : ℕ
  horses : ℕ
  sheep : ℕ
  cows : ℕ
  ducks : ℕ

/-- Checks if a given population satisfies the Minerva conditions -/
def isValidMinervaPop (pop : MinervaPop) : Prop :=
  pop.people = 4 * pop.horses ∧
  pop.sheep = 3 * pop.cows ∧
  pop.ducks = 2 * pop.people - 2

/-- The total population of Minerva -/
def totalPop (pop : MinervaPop) : ℕ :=
  pop.people + pop.horses + pop.sheep + pop.cows + pop.ducks

/-- Theorem stating that 223 cannot be the total population of Minerva -/
theorem no_valid_pop_of_223 :
  ¬ ∃ (pop : MinervaPop), isValidMinervaPop pop ∧ totalPop pop = 223 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_pop_of_223_l2404_240408


namespace NUMINAMATH_CALUDE_salamander_population_decline_l2404_240488

def decrease_rate : ℝ := 0.3
def target_percentage : ℝ := 0.05
def start_year : ℕ := 2007

def population_percentage (n : ℕ) : ℝ := (1 - decrease_rate) ^ n

theorem salamander_population_decline :
  ∃ n : ℕ, 
    population_percentage n ≤ target_percentage ∧
    population_percentage (n - 1) > target_percentage ∧
    start_year + n = 2016 :=
  sorry

end NUMINAMATH_CALUDE_salamander_population_decline_l2404_240488


namespace NUMINAMATH_CALUDE_natural_number_pairs_l2404_240473

theorem natural_number_pairs : 
  ∀ a b : ℕ, 
    (90 < a + b ∧ a + b < 100) ∧ 
    (0.9 < (a : ℝ) / (b : ℝ) ∧ (a : ℝ) / (b : ℝ) < 0.91) → 
    ((a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l2404_240473


namespace NUMINAMATH_CALUDE_palm_meadows_beds_l2404_240410

theorem palm_meadows_beds (total_rooms : ℕ) (two_bed_rooms : ℕ) (total_beds : ℕ) 
  (h1 : total_rooms = 13)
  (h2 : two_bed_rooms = 8)
  (h3 : total_beds = 31)
  (h4 : two_bed_rooms ≤ total_rooms) :
  (total_beds - 2 * two_bed_rooms) / (total_rooms - two_bed_rooms) = 3 := by
  sorry

end NUMINAMATH_CALUDE_palm_meadows_beds_l2404_240410
