import Mathlib

namespace NUMINAMATH_CALUDE_number_problem_l2829_282916

/-- Given a number N, if N/p = 8, N/q = 18, and p - q = 0.2777777777777778, then N = 4 -/
theorem number_problem (N p q : ℝ) 
  (h1 : N / p = 8)
  (h2 : N / q = 18)
  (h3 : p - q = 0.2777777777777778) : 
  N = 4 := by sorry

end NUMINAMATH_CALUDE_number_problem_l2829_282916


namespace NUMINAMATH_CALUDE_inequality_proof_l2829_282985

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a - b / a > b - a / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2829_282985


namespace NUMINAMATH_CALUDE_bake_sale_earnings_l2829_282928

theorem bake_sale_earnings (total : ℝ) (ingredients_cost shelter_donation : ℝ) 
  (h1 : ingredients_cost = 100)
  (h2 : shelter_donation = (total - ingredients_cost) / 2 + 10)
  (h3 : shelter_donation = 160) : 
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_earnings_l2829_282928


namespace NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l2829_282917

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of the first 60 terms is 500
  sum_first_60 : (60 : ℚ) / 2 * (2 * a + 59 * d) = 500
  -- Sum of the next 60 terms (61 to 120) is 2900
  sum_next_60 : (60 : ℚ) / 2 * (2 * (a + 60 * d) + 59 * d) = 2900

/-- The first term of the arithmetic sequence with given properties is -34/3 -/
theorem first_term_of_special_arithmetic_sequence (seq : ArithmeticSequence) : seq.a = -34/3 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l2829_282917


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2829_282963

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def standard_circle_equation (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Given a circle with center (1, -2) and radius 6, its standard equation is (x-1)^2 + (y+2)^2 = 36 -/
theorem circle_equation_proof :
  ∀ x y : ℝ, standard_circle_equation 1 (-2) 6 x y ↔ (x - 1)^2 + (y + 2)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2829_282963


namespace NUMINAMATH_CALUDE_solution_set_a_1_no_a_for_all_reals_l2829_282912

-- Define the inequality function
def inequality (a : ℝ) (x : ℝ) : Prop :=
  |a*x - 1| + |a*x - a| ≥ 2

-- Part 1: Solution set when a = 1
theorem solution_set_a_1 :
  ∀ x : ℝ, inequality 1 x ↔ (x ≤ 0 ∨ x ≥ 2) :=
sorry

-- Part 2: No a > 0 makes the solution set ℝ
theorem no_a_for_all_reals :
  ¬ ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, inequality a x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_1_no_a_for_all_reals_l2829_282912


namespace NUMINAMATH_CALUDE_inequality_solution_l2829_282900

theorem inequality_solution (x : ℝ) : 
  (3 * x - 2 ≥ 0) → 
  (|Real.sqrt (3 * x - 2) - 3| > 1 ↔ (x > 6 ∨ (2/3 ≤ x ∧ x < 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2829_282900


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l2829_282924

theorem number_satisfying_condition : ∃ x : ℤ, (x - 29) / 13 = 15 ∧ x = 224 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l2829_282924


namespace NUMINAMATH_CALUDE_total_rent_is_245_l2829_282948

-- Define the oxen-months for each person
def a_oxen_months : ℕ := 10 * 7
def b_oxen_months : ℕ := 12 * 5
def c_oxen_months : ℕ := 15 * 3

-- Define the total oxen-months
def total_oxen_months : ℕ := a_oxen_months + b_oxen_months + c_oxen_months

-- Define c's payment
def c_payment : ℚ := 62.99999999999999

-- Define the cost per oxen-month
def cost_per_oxen_month : ℚ := c_payment / c_oxen_months

-- Theorem to prove
theorem total_rent_is_245 : 
  ∃ (total_rent : ℚ), total_rent = cost_per_oxen_month * total_oxen_months ∧ 
                       total_rent = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_is_245_l2829_282948


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2829_282903

/-- Given a circle D with equation x^2 + 14y + 63 = -y^2 - 12x, 
    where (a, b) is the center and r is the radius, 
    prove that a + b + r = -13 + √22 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (∀ x y, x^2 + 14*y + 63 = -y^2 - 12*x) →
  ((x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = -13 + Real.sqrt 22 := by
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2829_282903


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2829_282927

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P > 0 →
  r > 0 →
  P * (1 + r)^2 - P = 492 →
  P * (1 + r)^2 = 5292 →
  r = 0.05 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2829_282927


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2829_282964

theorem smaller_number_in_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / b = 3 / 8 → (a - 24) / (b - 24) = 4 / 9 → a = 72 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2829_282964


namespace NUMINAMATH_CALUDE_library_capacity_is_400_l2829_282983

/-- The capacity of Karson's home library -/
def library_capacity : ℕ := sorry

/-- The number of books Karson currently has -/
def current_books : ℕ := 120

/-- The number of additional books Karson needs to buy -/
def additional_books : ℕ := 240

/-- The percentage of the library that will be full after buying additional books -/
def full_percentage : ℚ := 9/10

theorem library_capacity_is_400 : 
  library_capacity = 400 :=
by
  have h1 : current_books + additional_books = (library_capacity : ℚ) * full_percentage :=
    sorry
  sorry

end NUMINAMATH_CALUDE_library_capacity_is_400_l2829_282983


namespace NUMINAMATH_CALUDE_complex_power_difference_l2829_282960

theorem complex_power_difference (x : ℂ) : 
  x - 1 / x = 2 * Complex.I → x^729 - 1 / x^729 = 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_difference_l2829_282960


namespace NUMINAMATH_CALUDE_initial_candies_equals_sum_of_given_and_left_l2829_282978

/-- Given the number of candies given away and the number of candies left,
    prove that the initial number of candies is their sum. -/
theorem initial_candies_equals_sum_of_given_and_left (given away : ℕ) (left : ℕ) :
  given + left = given + left := by sorry

end NUMINAMATH_CALUDE_initial_candies_equals_sum_of_given_and_left_l2829_282978


namespace NUMINAMATH_CALUDE_elysses_carrying_capacity_l2829_282969

/-- The number of bags Elysse can carry in one trip -/
def elysses_bags : ℕ := 3

/-- The number of trips Elysse and her brother take -/
def num_trips : ℕ := 5

/-- The total number of bags they carry -/
def total_bags : ℕ := 30

theorem elysses_carrying_capacity :
  (elysses_bags * 2 * num_trips = total_bags) ∧ 
  (elysses_bags > 0) ∧ 
  (num_trips > 0) ∧ 
  (total_bags > 0) := by
  sorry

end NUMINAMATH_CALUDE_elysses_carrying_capacity_l2829_282969


namespace NUMINAMATH_CALUDE_age_equality_l2829_282990

theorem age_equality (joe_current_age : ℕ) (james_current_age : ℕ) (years_until_equality : ℕ) : 
  joe_current_age = 22 →
  james_current_age = joe_current_age - 10 →
  2 * (joe_current_age + years_until_equality) = 3 * (james_current_age + years_until_equality) →
  years_until_equality = 8 := by
sorry

end NUMINAMATH_CALUDE_age_equality_l2829_282990


namespace NUMINAMATH_CALUDE_calculation_proof_l2829_282956

theorem calculation_proof : (((20^10 / 20^9)^3 * 10^6) / 2^12) = 1953125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2829_282956


namespace NUMINAMATH_CALUDE_equation_linear_implies_a_equals_one_l2829_282943

theorem equation_linear_implies_a_equals_one (a : ℝ) :
  (∀ x, (a^2 - 1) * x^2 - a*x - x + 2 = 0 → ∃ m b, (a^2 - 1) * x^2 - a*x - x + 2 = m*x + b) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_linear_implies_a_equals_one_l2829_282943


namespace NUMINAMATH_CALUDE_square_of_negative_triple_l2829_282996

theorem square_of_negative_triple (a : ℝ) : (-3 * a)^2 = 9 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_triple_l2829_282996


namespace NUMINAMATH_CALUDE_cubic_equation_complex_root_l2829_282953

theorem cubic_equation_complex_root (k : ℝ) : 
  (∃ z : ℂ, z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  k = 2 ∨ k = -2/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_complex_root_l2829_282953


namespace NUMINAMATH_CALUDE_pancake_max_pieces_l2829_282984

/-- The maximum number of pieces a circle can be divided into with n straight cuts -/
def maxPieces (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- A round pancake can be divided into at most 7 pieces with three straight cuts -/
theorem pancake_max_pieces :
  maxPieces 3 = 7 :=
sorry

end NUMINAMATH_CALUDE_pancake_max_pieces_l2829_282984


namespace NUMINAMATH_CALUDE_jenny_reads_three_books_l2829_282936

/-- Represents the number of books Jenny can read given the conditions --/
def books_jenny_can_read (days : ℕ) (reading_speed : ℕ) (reading_time : ℚ) 
  (book1_words : ℕ) (book2_words : ℕ) (book3_words : ℕ) : ℕ :=
  let total_words := book1_words + book2_words + book3_words
  let total_reading_hours := (days : ℚ) * reading_time
  let words_read := (reading_speed : ℚ) * total_reading_hours
  if words_read ≥ total_words then 3 else 
    if words_read ≥ book1_words + book2_words then 2 else
      if words_read ≥ book1_words then 1 else 0

/-- Theorem stating that Jenny can read exactly 3 books in 10 days --/
theorem jenny_reads_three_books : 
  books_jenny_can_read 10 100 (54/60) 200 400 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jenny_reads_three_books_l2829_282936


namespace NUMINAMATH_CALUDE_sum_of_ages_l2829_282925

/-- Represents the ages of two people P and Q -/
structure Ages where
  p : ℝ
  q : ℝ

/-- The condition that P's age is thrice Q's age when P was as old as Q is now -/
def age_relation (ages : Ages) : Prop :=
  ages.p = 3 * (ages.q - (ages.p - ages.q))

/-- Theorem stating the sum of P and Q's ages given the conditions -/
theorem sum_of_ages :
  ∀ (ages : Ages),
    ages.q = 37.5 →
    age_relation ages →
    ages.p + ages.q = 93.75 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_l2829_282925


namespace NUMINAMATH_CALUDE_optimal_sale_l2829_282958

/-- Represents the selling price and number of items that maximize profit while meeting constraints --/
def OptimalSale : Type := ℕ × ℕ

/-- Calculates the number of items sold given a selling price --/
def itemsSold (initialItems : ℕ) (initialPrice : ℕ) (newPrice : ℕ) : ℕ :=
  initialItems - 10 * (newPrice - initialPrice)

/-- Calculates the total cost given a number of items and cost per item --/
def totalCost (items : ℕ) (costPerItem : ℕ) : ℕ := items * costPerItem

/-- Calculates the profit given selling price, cost per item, and number of items sold --/
def profit (sellingPrice : ℕ) (costPerItem : ℕ) (itemsSold : ℕ) : ℕ :=
  (sellingPrice - costPerItem) * itemsSold

/-- Theorem stating the optimal selling price and number of items to purchase --/
theorem optimal_sale (initialItems : ℕ) (initialPrice : ℕ) (costPerItem : ℕ) (targetProfit : ℕ) (maxCost : ℕ)
    (h_initialItems : initialItems = 500)
    (h_initialPrice : initialPrice = 50)
    (h_costPerItem : costPerItem = 40)
    (h_targetProfit : targetProfit = 8000)
    (h_maxCost : maxCost = 10000) :
    ∃ (sale : OptimalSale),
      let (sellingPrice, itemsToBuy) := sale
      profit sellingPrice costPerItem (itemsSold initialItems initialPrice sellingPrice) = targetProfit ∧
      totalCost itemsToBuy costPerItem < maxCost ∧
      sellingPrice = 80 ∧
      itemsToBuy = 200 := by
  sorry

end NUMINAMATH_CALUDE_optimal_sale_l2829_282958


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2829_282994

theorem linear_equation_solution (x y : ℝ) :
  2 * x + y - 5 = 0 → x = (5 - y) / 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2829_282994


namespace NUMINAMATH_CALUDE_problem_statement_l2829_282947

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_statement (m : ℝ) :
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) →
  (m = 1 ∧
   ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
     1/a + 1/(2*b) + 1/(3*c) = m →
     a + 2*b + 3*c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2829_282947


namespace NUMINAMATH_CALUDE_max_n_given_average_l2829_282993

theorem max_n_given_average (m n : ℕ+) : 
  (m + n : ℚ) / 2 = 5 → n ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_n_given_average_l2829_282993


namespace NUMINAMATH_CALUDE_smallest_k_value_l2829_282966

theorem smallest_k_value (x y : ℤ) (h1 : x = -2) (h2 : y = 5) : 
  ∃ k : ℤ, (∀ m : ℤ, k * x + 2 * y ≤ 4 → m * x + 2 * y ≤ 4 → k ≤ m) ∧ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_value_l2829_282966


namespace NUMINAMATH_CALUDE_joan_book_sale_l2829_282918

/-- Given that Joan initially gathered 33 books and found 26 more,
    prove that the total number of books she has for sale is 59. -/
theorem joan_book_sale (initial_books : ℕ) (additional_books : ℕ) 
  (h1 : initial_books = 33) (h2 : additional_books = 26) : 
  initial_books + additional_books = 59 := by
  sorry

end NUMINAMATH_CALUDE_joan_book_sale_l2829_282918


namespace NUMINAMATH_CALUDE_decimal_to_binary_89_l2829_282957

theorem decimal_to_binary_89 : 
  (89 : ℕ).digits 2 = [1, 0, 0, 1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_89_l2829_282957


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l2829_282906

/-- Sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if a year is after 2020 and has sum of digits equal to 4 -/
def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

/-- 2022 is the first year after 2020 with sum of digits equal to 4 -/
theorem first_year_after_2020_with_sum_4 :
  (∀ y : ℕ, y < 2022 → ¬(isValidYear y)) ∧ isValidYear 2022 := by
  sorry

#eval sumOfDigits 2020  -- Should output 4
#eval sumOfDigits 2022  -- Should output 4

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l2829_282906


namespace NUMINAMATH_CALUDE_conclusion_1_conclusion_2_conclusion_3_l2829_282977

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

-- Theorem 1
theorem conclusion_1 (b : ℝ) :
  (∀ m : ℝ, m*(m - 2*b) ≥ 1 - 2*b) → b = 1 := by sorry

-- Theorem 2
theorem conclusion_2 (b : ℝ) :
  ∃ h k : ℝ, (∀ x : ℝ, f b x ≥ f b h) ∧ k = f b h ∧ k = -h^2 + 3 := by sorry

-- Theorem 3
theorem conclusion_3 (b : ℝ) :
  (∀ x : ℝ, -1 ≤ x → x ≤ 5 → f b x ≤ f b (-1)) →
  (∃ m₁ m₂ p : ℝ, m₁ ≠ m₂ ∧ f b m₁ = p ∧ f b m₂ = p) →
  ∃ m₁ m₂ : ℝ, m₁ + m₂ > 4 := by sorry

end NUMINAMATH_CALUDE_conclusion_1_conclusion_2_conclusion_3_l2829_282977


namespace NUMINAMATH_CALUDE_lake_radius_l2829_282932

/-- Given a circular lake with a diameter of 26 meters, its radius is 13 meters. -/
theorem lake_radius (lake_diameter : ℝ) (h : lake_diameter = 26) : 
  lake_diameter / 2 = 13 := by sorry

end NUMINAMATH_CALUDE_lake_radius_l2829_282932


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2829_282942

theorem quadratic_equation_roots (k : ℝ) :
  let f := fun x : ℝ => x^2 - 2*(k-1)*x + k^2
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k ≤ 1/2) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁*x₂ + x₁ + x₂ - 1 = 0 → k = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2829_282942


namespace NUMINAMATH_CALUDE_parabola_properties_l2829_282954

/-- Parabola intersecting x-axis -/
def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (m^2 + 4) * x - 2 * m^2 - 12

/-- Discriminant of the parabola -/
def discriminant (m : ℝ) : ℝ := (m^2 + 4)^2 + 4 * (2 * m^2 + 12)

/-- Chord length of the parabola intersecting x-axis -/
def chord_length (m : ℝ) : ℝ := m^2 + 8

theorem parabola_properties (m : ℝ) :
  (∀ m, discriminant m > 0) ∧
  (chord_length m = m^2 + 8) ∧
  (∀ m, chord_length m ≥ 8) ∧
  (chord_length 0 = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2829_282954


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2829_282967

theorem right_triangle_sides : 
  (7^2 + 24^2 = 25^2) ∧ 
  (1.5^2 + 2^2 = 2.5^2) ∧ 
  (8^2 + 15^2 = 17^2) ∧ 
  (Real.sqrt 3)^2 + (Real.sqrt 4)^2 ≠ (Real.sqrt 5)^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2829_282967


namespace NUMINAMATH_CALUDE_solution_count_l2829_282920

/-- The number of distinct ordered pairs of non-negative integers (a, b) that sum to 50 -/
def count_solutions : ℕ := 51

/-- Predicate for valid solutions -/
def is_valid_solution (a b : ℕ) : Prop := a + b = 50

theorem solution_count :
  (∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ is_valid_solution p.1 p.2) ∧ 
    s.card = count_solutions) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l2829_282920


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2829_282946

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → 
  -2 ≤ a ∧ a < 6/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2829_282946


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l2829_282913

/-- Given a cylinder with original volume of 15 cubic feet, 
    prove that tripling its radius and quadrupling its height 
    results in a new volume of 540 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 15 → π * (3*r)^2 * (4*h) = 540 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l2829_282913


namespace NUMINAMATH_CALUDE_mike_ride_distance_l2829_282979

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  start_fee : ℝ
  per_mile_fee : ℝ
  toll_fee : ℝ
  distance : ℝ

/-- Calculates the total fare for a taxi ride -/
def total_fare (ride : TaxiRide) : ℝ :=
  ride.start_fee + ride.toll_fee + ride.per_mile_fee * ride.distance

/-- Proves that Mike's ride was 34 miles long given the conditions -/
theorem mike_ride_distance :
  let mike : TaxiRide := { start_fee := 2.5, per_mile_fee := 0.25, toll_fee := 0, distance := 34 }
  let annie : TaxiRide := { start_fee := 2.5, per_mile_fee := 0.25, toll_fee := 5, distance := 14 }
  total_fare mike = total_fare annie := by
  sorry

#check mike_ride_distance

end NUMINAMATH_CALUDE_mike_ride_distance_l2829_282979


namespace NUMINAMATH_CALUDE_rationalize_and_product_l2829_282929

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2 : ℝ) + Real.sqrt 5) / ((3 : ℝ) - Real.sqrt 5) = (A : ℝ) / 4 + (B : ℝ) / 4 * Real.sqrt (C : ℝ)) ∧
  A * B * C = 275 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l2829_282929


namespace NUMINAMATH_CALUDE_temperature_increase_l2829_282973

theorem temperature_increase (morning_temp afternoon_temp : ℤ) : 
  morning_temp = -3 → afternoon_temp = 5 → afternoon_temp - morning_temp = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_temperature_increase_l2829_282973


namespace NUMINAMATH_CALUDE_blue_fish_ratio_l2829_282950

/-- Given a fish tank with the following properties:
  - The total number of fish is 60.
  - Half of the blue fish have spots.
  - There are 10 blue, spotted fish.
  Prove that the ratio of blue fish to the total number of fish is 1/3. -/
theorem blue_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) 
  (h1 : total_fish = 60)
  (h2 : blue_spotted_fish = 10)
  (h3 : blue_spotted_fish * 2 = blue_spotted_fish + (total_fish - blue_spotted_fish * 2)) :
  (blue_spotted_fish * 2 : ℚ) / total_fish = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_ratio_l2829_282950


namespace NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l2829_282915

-- Part 1: Calculation proof
theorem calculation_proof :
  -2^2 - |2 - Real.sqrt 5| + (8 : ℝ)^(1/3) = -Real.sqrt 5 := by sorry

-- Part 2: System of equations proof
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x + y = 5 ∧ x - 3*y = 6 ∧ x = 3 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l2829_282915


namespace NUMINAMATH_CALUDE_spinner_probability_l2829_282955

/-- Represents a game board based on an equilateral triangle -/
structure GameBoard where
  /-- The number of regions formed by altitudes and one median -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Ensure the number of shaded regions is less than or equal to the total regions -/
  h_valid : shaded_regions ≤ total_regions

/-- Calculate the probability of landing in a shaded region -/
def probability (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- The main theorem to be proved -/
theorem spinner_probability (board : GameBoard) 
  (h_total : board.total_regions = 12)
  (h_shaded : board.shaded_regions = 3) : 
  probability board = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2829_282955


namespace NUMINAMATH_CALUDE_max_value_theorem_l2829_282959

theorem max_value_theorem (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 3) :
  a^3 * b + b^3 * a ≤ 81/16 ∧ ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ a₀ + b₀ = 3 ∧ a₀^3 * b₀ + b₀^3 * a₀ = 81/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2829_282959


namespace NUMINAMATH_CALUDE_trig_simplification_l2829_282939

theorem trig_simplification (α : Real) :
  Real.sin (-α) * Real.cos (π + α) * Real.tan (2 * π + α) = Real.sin α ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2829_282939


namespace NUMINAMATH_CALUDE_proposition_truth_count_l2829_282945

theorem proposition_truth_count (a b c : ℝ) : 
  (∃ (x y z : ℝ), x * z^2 > y * z^2 ∧ x ≤ y) ∨ 
  (∀ (x y z : ℝ), x > y → x * z^2 > y * z^2) ∨
  (∀ (x y z : ℝ), x ≤ y → x * z^2 ≤ y * z^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_count_l2829_282945


namespace NUMINAMATH_CALUDE_value_of_a_l2829_282968

theorem value_of_a (A B : Set ℕ) (a : ℕ) :
  A = {a, 2} →
  B = {1, 2} →
  A ∪ B = {1, 2, 3} →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2829_282968


namespace NUMINAMATH_CALUDE_ellipse_min_value_l2829_282970

/-- For an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    prove that the minimum value of (a² + 1) / b is 4√3 / 3 when e = 1/2. -/
theorem ellipse_min_value (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a^2 - b^2) / a^2 = 1/4) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (a^2 + 1) / b ≥ 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_value_l2829_282970


namespace NUMINAMATH_CALUDE_intersection_is_empty_l2829_282971

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1 - 3}

theorem intersection_is_empty : set_A ∩ set_B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l2829_282971


namespace NUMINAMATH_CALUDE_a_value_l2829_282941

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

-- Theorem statement
theorem a_value (a : ℝ) : f_prime a 1 = 4 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l2829_282941


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l2829_282923

theorem consecutive_product_not_power (x a n : ℕ) : 
  a ≥ 2 → n ≥ 2 → (x - 1) * x * (x + 1) ≠ a^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l2829_282923


namespace NUMINAMATH_CALUDE_last_equal_sum_date_l2829_282980

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2008 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

def sum_first_four (year month day : ℕ) : ℕ :=
  (day / 10) + (day % 10) + (month / 10) + (month % 10)

def sum_last_four (year : ℕ) : ℕ :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

def has_equal_sum (year month day : ℕ) : Prop :=
  sum_first_four year month day = sum_last_four year

def is_after (year1 month1 day1 year2 month2 day2 : ℕ) : Prop :=
  year1 > year2 ∨ (year1 = year2 ∧ (month1 > month2 ∨ (month1 = month2 ∧ day1 > day2)))

theorem last_equal_sum_date :
  ∀ (year month day : ℕ),
    is_valid_date year month day →
    has_equal_sum year month day →
    ¬(is_after year month day 2008 12 25) →
    year = 2008 ∧ month = 12 ∧ day = 25 :=
sorry

end NUMINAMATH_CALUDE_last_equal_sum_date_l2829_282980


namespace NUMINAMATH_CALUDE_tan_greater_than_cubic_l2829_282989

theorem tan_greater_than_cubic (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  Real.tan x > x + (1 / 3) * x^3 := by
  sorry

end NUMINAMATH_CALUDE_tan_greater_than_cubic_l2829_282989


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l2829_282981

/-- Given the regular price per can and the discounted price for 72 cans,
    calculate the discount percentage. -/
theorem soda_discount_percentage
  (regular_price : ℝ)
  (discounted_price : ℝ)
  (h_regular_price : regular_price = 0.60)
  (h_discounted_price : discounted_price = 34.56) :
  (1 - discounted_price / (72 * regular_price)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l2829_282981


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_surface_area_l2829_282988

/-- The total surface area of an oblique triangular prism -/
theorem oblique_triangular_prism_surface_area
  (a l : ℝ)
  (h_a_pos : 0 < a)
  (h_l_pos : 0 < l) :
  let lateral_surface_area := 3 * a * l
  let base_area := a^2 * Real.sqrt 3 / 2
  let total_surface_area := lateral_surface_area + 2 * base_area
  total_surface_area = 3 * a * l + a^2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_oblique_triangular_prism_surface_area_l2829_282988


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2829_282905

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l2829_282905


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2829_282961

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel
  (m n : Line) (α : Plane) 
  (h_diff : m ≠ n)
  (h_m_perp : perp m α)
  (h_n_perp : perp n α) :
  para m n :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2829_282961


namespace NUMINAMATH_CALUDE_f_max_at_neg_four_l2829_282910

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 8*x + 16

-- State the theorem
theorem f_max_at_neg_four :
  ∀ x : ℝ, f x ≤ f (-4) :=
by sorry

end NUMINAMATH_CALUDE_f_max_at_neg_four_l2829_282910


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l2829_282991

theorem half_abs_diff_squares : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l2829_282991


namespace NUMINAMATH_CALUDE_equation_solutions_l2829_282995

theorem equation_solutions :
  (∃ x : ℝ, 3 * x^3 - 15 = 9 ∧ x = 2) ∧
  (∃ x₁ x₂ : ℝ, 2 * (x₁ - 1)^2 = 72 ∧ 2 * (x₂ - 1)^2 = 72 ∧ x₁ = 7 ∧ x₂ = -5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2829_282995


namespace NUMINAMATH_CALUDE_difference_of_squares_l2829_282940

theorem difference_of_squares : (635 : ℕ)^2 - (365 : ℕ)^2 = 270000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2829_282940


namespace NUMINAMATH_CALUDE_colin_skipping_speed_l2829_282976

theorem colin_skipping_speed (bruce_speed tony_speed brandon_speed colin_speed : ℝ) :
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = (1/3) * tony_speed →
  colin_speed = 6 * brandon_speed →
  colin_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_colin_skipping_speed_l2829_282976


namespace NUMINAMATH_CALUDE_sin_negative_150_degrees_l2829_282998

theorem sin_negative_150_degrees : Real.sin (-(150 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_150_degrees_l2829_282998


namespace NUMINAMATH_CALUDE_inequality_properties_l2829_282926

theorem inequality_properties (x y : ℝ) (h : x > y) : x^3 > y^3 ∧ Real.log x > Real.log y := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l2829_282926


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2829_282997

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2/x₀ + 1/y₀ = 1 ∧ x₀ + 2*y₀ = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2829_282997


namespace NUMINAMATH_CALUDE_base6_addition_sum_l2829_282986

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- Converts a Base6Digit to its natural number representation -/
def to_nat (d : Base6Digit) : Nat := d.val

/-- Represents the base-6 addition problem 5CD₆ + 32₆ = 61C₆ -/
def base6_addition_problem (C D : Base6Digit) : Prop :=
  (5 * 6 * 6 + to_nat C * 6 + to_nat D) + (3 * 6 + 2) = 
  (6 * 6 + 1 * 6 + to_nat C)

theorem base6_addition_sum (C D : Base6Digit) :
  base6_addition_problem C D → to_nat C + to_nat D = 6 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_sum_l2829_282986


namespace NUMINAMATH_CALUDE_yardwork_earnings_contribution_l2829_282909

def earnings : List ℕ := [18, 22, 30, 35, 45]
def max_contribution : ℕ := 40
def num_friends : ℕ := 5

theorem yardwork_earnings_contribution :
  let total := (earnings.sum - 45 + max_contribution)
  let equal_share := total / num_friends
  35 - equal_share = 6 := by sorry

end NUMINAMATH_CALUDE_yardwork_earnings_contribution_l2829_282909


namespace NUMINAMATH_CALUDE_binomial_probability_l2829_282951

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- The probability mass function for a binomial distribution -/
def pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem binomial_probability (X : BinomialDistribution 6 (1/2)) :
  pmf 6 (1/2) 3 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l2829_282951


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l2829_282919

-- First equation
theorem solve_equation_1 : 
  ∃ x : ℚ, 15 - (7 - 5 * x) = 2 * x + (5 - 3 * x) ↔ x = -1/2 := by sorry

-- Second equation
theorem solve_equation_2 : 
  ∃ x : ℚ, (x - 3) / 2 - (2 * x - 3) / 5 = 1 ↔ x = 19 := by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l2829_282919


namespace NUMINAMATH_CALUDE_train_journey_time_l2829_282987

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 →
  usual_time > 0 →
  (6/7 * usual_speed) * (usual_time + 15/60) = usual_speed * usual_time →
  usual_time = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l2829_282987


namespace NUMINAMATH_CALUDE_truck_distance_proof_l2829_282938

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The distance traveled by the truck -/
def truck_distance : ℕ := arithmetic_sum 5 7 30

theorem truck_distance_proof : truck_distance = 3195 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_proof_l2829_282938


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2829_282992

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 4) :
  w / y = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2829_282992


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2829_282974

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℝ := {y | ∃ x : ℤ, y = 2 * x}

theorem intersection_of_P_and_Q : P ∩ Q = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2829_282974


namespace NUMINAMATH_CALUDE_all_transformed_points_in_S_l2829_282901

def S : Set ℂ := {z | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_points_in_S :
  ∀ z ∈ S, (1/2 + 1/2*I) * z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_all_transformed_points_in_S_l2829_282901


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l2829_282934

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l2829_282934


namespace NUMINAMATH_CALUDE_x_seventh_plus_27x_squared_l2829_282930

theorem x_seventh_plus_27x_squared (x : ℝ) (h : x^3 - 3*x = 7) :
  x^7 + 27*x^2 = 76*x^2 + 270*x + 483 := by
  sorry

end NUMINAMATH_CALUDE_x_seventh_plus_27x_squared_l2829_282930


namespace NUMINAMATH_CALUDE_maddie_purchase_cost_l2829_282933

/-- Calculates the total cost of Maddie's beauty product purchase --/
def calculate_total_cost (
  palette_price : ℝ)
  (palette_count : ℕ)
  (palette_discount : ℝ)
  (lipstick_price : ℝ)
  (lipstick_count : ℕ)
  (hair_color_price : ℝ)
  (hair_color_count : ℕ)
  (hair_color_discount : ℝ)
  (sales_tax_rate : ℝ) : ℝ :=
  let palette_cost := palette_price * palette_count * (1 - palette_discount)
  let lipstick_cost := lipstick_price * (lipstick_count - 1)
  let hair_color_cost := hair_color_price * hair_color_count * (1 - hair_color_discount)
  let subtotal := palette_cost + lipstick_cost + hair_color_cost
  let total := subtotal * (1 + sales_tax_rate)
  total

/-- Theorem stating that the total cost of Maddie's purchase is $58.64 --/
theorem maddie_purchase_cost :
  calculate_total_cost 15 3 0.2 2.5 4 4 3 0.1 0.08 = 58.64 := by
  sorry

end NUMINAMATH_CALUDE_maddie_purchase_cost_l2829_282933


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2829_282914

/-- Given two parallel vectors a and b, prove that m = 1/2 --/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2829_282914


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2829_282952

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℚ, x - 2*y = 0 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℚ, 3*x - 5*y = 9 ∧ 2*x + 3*y = -6 ∧ x = -3/19 ∧ y = -36/19) := by
  sorry


end NUMINAMATH_CALUDE_system_of_equations_solutions_l2829_282952


namespace NUMINAMATH_CALUDE_min_omega_for_symmetry_axis_l2829_282911

/-- The minimum positive value of ω for which f(x) = sin(ωx + π/6) has a symmetry axis at x = π/12 -/
theorem min_omega_for_symmetry_axis : ∃ (ω_min : ℝ), 
  (∀ (ω : ℝ), ω > 0 → (∃ (k : ℤ), ω = 12 * k + 4)) → 
  (∀ (ω : ℝ), ω > 0 → ω ≥ ω_min) → 
  ω_min = 4 := by
sorry

end NUMINAMATH_CALUDE_min_omega_for_symmetry_axis_l2829_282911


namespace NUMINAMATH_CALUDE_least_integer_b_for_quadratic_range_l2829_282935

theorem least_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -10) ↔ b ≤ -10 :=
sorry

end NUMINAMATH_CALUDE_least_integer_b_for_quadratic_range_l2829_282935


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2829_282949

theorem consecutive_integers_sum_of_squares (n : ℤ) : 
  n * (n + 1) * (n + 2) = 12 * (3 * n + 3) → 
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2829_282949


namespace NUMINAMATH_CALUDE_store_earnings_calculation_l2829_282999

/-- Represents the earnings calculation for a store selling bottled drinks -/
theorem store_earnings_calculation (cola_price juice_price water_price sports_price : ℚ)
                                   (cola_sold juice_sold water_sold sports_sold : ℕ) :
  cola_price = 3 →
  juice_price = 3/2 →
  water_price = 1 →
  sports_price = 5/2 →
  cola_sold = 18 →
  juice_sold = 15 →
  water_sold = 30 →
  sports_sold = 22 →
  cola_price * cola_sold + juice_price * juice_sold + 
  water_price * water_sold + sports_price * sports_sold = 161.5 := by
sorry

end NUMINAMATH_CALUDE_store_earnings_calculation_l2829_282999


namespace NUMINAMATH_CALUDE_square_difference_equals_product_l2829_282965

theorem square_difference_equals_product : (51 + 15)^2 - (51^2 + 15^2) = 1530 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_product_l2829_282965


namespace NUMINAMATH_CALUDE_solution_product_l2829_282972

theorem solution_product (a b : ℝ) : 
  (a - 3) * (2 * a + 7) = a^2 - 11 * a + 28 →
  (b - 3) * (2 * b + 7) = b^2 - 11 * b + 28 →
  a ≠ b →
  (a + 2) * (b + 2) = -66 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2829_282972


namespace NUMINAMATH_CALUDE_M_value_l2829_282922

def M : ℕ → ℕ
  | 0 => 0
  | 1 => 4
  | (n + 2) => (2*n + 2)^2 + (2*n + 4)^2 - M n

theorem M_value : M 75 = 22800 := by
  sorry

end NUMINAMATH_CALUDE_M_value_l2829_282922


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2829_282908

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |3*x - 15|

theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 3 10, g x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = max) ∧
    (∀ x ∈ Set.Icc 3 10, min ≤ g x) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = min) ∧
    max + min = -21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2829_282908


namespace NUMINAMATH_CALUDE_line_through_two_points_l2829_282937

/-- 
Given two distinct points P₁(x₁, y₁) and P₂(x₂, y₂) in the plane,
the equation (x-x₁)(y₂-y₁) = (y-y₁)(x₂-x₁) represents the line passing through these points.
-/
theorem line_through_two_points (x₁ y₁ x₂ y₂ : ℝ) (h : (x₁, y₁) ≠ (x₂, y₂)) :
  ∀ x y : ℝ, (x - x₁) * (y₂ - y₁) = (y - y₁) * (x₂ - x₁) ↔ 
  ∃ t : ℝ, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁) :=
by sorry

end NUMINAMATH_CALUDE_line_through_two_points_l2829_282937


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l2829_282962

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -7/4 ≤ x ∧ x ≤ 3/4} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) → (m > 6 ∨ m < -2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l2829_282962


namespace NUMINAMATH_CALUDE_dandelion_picking_l2829_282944

theorem dandelion_picking (billy_initial : ℕ) (george_initial : ℕ) (average : ℕ) : 
  billy_initial = 36 →
  george_initial = billy_initial / 3 →
  average = 34 →
  (billy_initial + george_initial + 2 * (average - (billy_initial + george_initial) / 2)) / 2 = average →
  average - (billy_initial + george_initial) / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_dandelion_picking_l2829_282944


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2829_282975

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → 1/x + 4/y ≥ 1/a + 4/b) ∧ 
  1/a + 4/b = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2829_282975


namespace NUMINAMATH_CALUDE_toluene_moles_formed_l2829_282904

-- Define the molar mass of benzene
def benzene_molar_mass : ℝ := 78.11

-- Define the chemical reaction
def chemical_reaction (benzene methane toluene hydrogen : ℝ) : Prop :=
  benzene = methane ∧ benzene = toluene ∧ benzene = hydrogen

-- Define the given conditions
def given_conditions (benzene_mass methane_moles : ℝ) : Prop :=
  benzene_mass = 156 ∧ methane_moles = 2

-- Theorem statement
theorem toluene_moles_formed 
  (benzene_mass methane_moles toluene_moles : ℝ)
  (h1 : given_conditions benzene_mass methane_moles)
  (h2 : chemical_reaction (benzene_mass / benzene_molar_mass) methane_moles toluene_moles 2) :
  toluene_moles = 2 := by
  sorry

end NUMINAMATH_CALUDE_toluene_moles_formed_l2829_282904


namespace NUMINAMATH_CALUDE_trajectory_of_G_l2829_282907

/-- The ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point P on the ellipse C -/
def point_P (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀

/-- Relation between vectors PG and GO -/
def vector_relation (x₀ y₀ x y : ℝ) : Prop :=
  (x - x₀, y - y₀) = (2 * (-x), 2 * (-y))

/-- Trajectory of point G -/
def trajectory_G (x y : ℝ) : Prop := 9*x^2/4 + 3*y^2 = 1

theorem trajectory_of_G (x₀ y₀ x y : ℝ) :
  point_P x₀ y₀ → vector_relation x₀ y₀ x y → trajectory_G x y := by sorry

end NUMINAMATH_CALUDE_trajectory_of_G_l2829_282907


namespace NUMINAMATH_CALUDE_calcium_iodide_weight_l2829_282921

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of calcium iodide -/
def moles_CaI2 : ℝ := 5

/-- The molecular weight of calcium iodide (CaI2) in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The total weight of calcium iodide in grams -/
def total_weight_CaI2 : ℝ := moles_CaI2 * molecular_weight_CaI2

theorem calcium_iodide_weight : total_weight_CaI2 = 1469.4 := by
  sorry

end NUMINAMATH_CALUDE_calcium_iodide_weight_l2829_282921


namespace NUMINAMATH_CALUDE_one_third_of_recipe_l2829_282931

theorem one_third_of_recipe (original_amount : ℚ) (reduced_amount : ℚ) : 
  original_amount = 27/4 → reduced_amount = original_amount / 3 → reduced_amount = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_one_third_of_recipe_l2829_282931


namespace NUMINAMATH_CALUDE_sock_drawer_theorem_l2829_282902

/-- The minimum number of socks needed to guarantee at least n pairs when selecting from m colors -/
def min_socks_for_pairs (m n : ℕ) : ℕ := m + 1 + 2 * (n - 1)

/-- The number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- The number of pairs we want to guarantee -/
def required_pairs : ℕ := 15

theorem sock_drawer_theorem :
  min_socks_for_pairs num_colors required_pairs = 33 :=
sorry

end NUMINAMATH_CALUDE_sock_drawer_theorem_l2829_282902


namespace NUMINAMATH_CALUDE_distance_traveled_is_9_miles_l2829_282982

/-- The total distance traveled when biking and jogging for a given time and rate -/
def total_distance (bike_time : ℚ) (bike_rate : ℚ) (jog_time : ℚ) (jog_rate : ℚ) : ℚ :=
  (bike_time * bike_rate) + (jog_time * jog_rate)

/-- Theorem stating that the total distance traveled is 9 miles -/
theorem distance_traveled_is_9_miles :
  let bike_time : ℚ := 1/2  -- 30 minutes in hours
  let bike_rate : ℚ := 6
  let jog_time : ℚ := 3/4   -- 45 minutes in hours
  let jog_rate : ℚ := 8
  total_distance bike_time bike_rate jog_time jog_rate = 9 := by
  sorry

#eval total_distance (1/2) 6 (3/4) 8

end NUMINAMATH_CALUDE_distance_traveled_is_9_miles_l2829_282982
