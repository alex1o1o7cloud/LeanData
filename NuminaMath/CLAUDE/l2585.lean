import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2585_258526

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2585_258526


namespace NUMINAMATH_CALUDE_handshakes_count_l2585_258514

/-- The number of people in the gathering -/
def total_people : ℕ := 30

/-- The number of people who know each other (Group A) -/
def group_a : ℕ := 20

/-- The number of people who know no one (Group B) -/
def group_b : ℕ := 10

/-- The number of handshakes between Group A and Group B -/
def handshakes_between : ℕ := group_a * group_b

/-- The number of handshakes within Group B -/
def handshakes_within : ℕ := group_b * (group_b - 1) / 2

/-- The total number of handshakes -/
def total_handshakes : ℕ := handshakes_between + handshakes_within

theorem handshakes_count : total_handshakes = 245 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l2585_258514


namespace NUMINAMATH_CALUDE_tetrahedron_octahedron_volume_ratio_l2585_258578

/-- The volume of a regular tetrahedron -/
def tetrahedronVolume (edgeLength : ℝ) : ℝ := sorry

/-- The volume of a regular octahedron -/
def octahedronVolume (edgeLength : ℝ) : ℝ := sorry

/-- Theorem: The ratio of the volume of a regular tetrahedron to the volume of a regular octahedron 
    with the same edge length is 1/2 -/
theorem tetrahedron_octahedron_volume_ratio (edgeLength : ℝ) (h : edgeLength > 0) : 
  tetrahedronVolume edgeLength / octahedronVolume edgeLength = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_octahedron_volume_ratio_l2585_258578


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2585_258563

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 30 * r = b ∧ b * r = 3/8) → b = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2585_258563


namespace NUMINAMATH_CALUDE_largest_square_tile_l2585_258537

theorem largest_square_tile (board_width board_length tile_size : ℕ) : 
  board_width = 19 → 
  board_length = 29 → 
  tile_size > 0 →
  (∀ n : ℕ, n > 1 → (board_width % n = 0 ∧ board_length % n = 0) → False) →
  tile_size = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_l2585_258537


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2585_258560

theorem inequality_system_solution :
  {x : ℝ | 2 + x > 7 - 4*x ∧ x < (4 + x) / 2} = {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2585_258560


namespace NUMINAMATH_CALUDE_cost_sharing_ratio_l2585_258551

def monthly_cost : ℚ := 14
def your_payment : ℚ := 84
def total_months : ℕ := 12

theorem cost_sharing_ratio :
  let yearly_cost := monthly_cost * total_months
  let friend_payment := yearly_cost - your_payment
  your_payment = friend_payment :=
by sorry

end NUMINAMATH_CALUDE_cost_sharing_ratio_l2585_258551


namespace NUMINAMATH_CALUDE_symmetry_axes_count_other_rotation_axes_count_l2585_258587

/-- Enumeration of regular polyhedra -/
inductive RegularPolyhedron
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Function to calculate the number of symmetry axes for a regular polyhedron -/
def symmetryAxes (p : RegularPolyhedron) : Nat :=
  match p with
  | RegularPolyhedron.Tetrahedron => 3
  | RegularPolyhedron.Cube => 9
  | RegularPolyhedron.Octahedron => 9
  | RegularPolyhedron.Dodecahedron => 16
  | RegularPolyhedron.Icosahedron => 16

/-- Function to calculate the number of other rotation axes for a regular polyhedron -/
def otherRotationAxes (p : RegularPolyhedron) : Nat :=
  match p with
  | RegularPolyhedron.Tetrahedron => 4
  | RegularPolyhedron.Cube => 10
  | RegularPolyhedron.Octahedron => 10
  | RegularPolyhedron.Dodecahedron => 16
  | RegularPolyhedron.Icosahedron => 16

/-- Theorem stating the number of symmetry axes for each regular polyhedron -/
theorem symmetry_axes_count :
  (∀ p : RegularPolyhedron, symmetryAxes p = 
    match p with
    | RegularPolyhedron.Tetrahedron => 3
    | RegularPolyhedron.Cube => 9
    | RegularPolyhedron.Octahedron => 9
    | RegularPolyhedron.Dodecahedron => 16
    | RegularPolyhedron.Icosahedron => 16) :=
by sorry

/-- Theorem stating the number of other rotation axes for each regular polyhedron -/
theorem other_rotation_axes_count :
  (∀ p : RegularPolyhedron, otherRotationAxes p = 
    match p with
    | RegularPolyhedron.Tetrahedron => 4
    | RegularPolyhedron.Cube => 10
    | RegularPolyhedron.Octahedron => 10
    | RegularPolyhedron.Dodecahedron => 16
    | RegularPolyhedron.Icosahedron => 16) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axes_count_other_rotation_axes_count_l2585_258587


namespace NUMINAMATH_CALUDE_billy_reads_three_books_l2585_258577

theorem billy_reads_three_books 
  (free_time_per_day : ℕ) 
  (weekend_days : ℕ) 
  (video_game_percentage : ℚ) 
  (pages_per_hour : ℕ) 
  (pages_per_book : ℕ) 
  (h1 : free_time_per_day = 8)
  (h2 : weekend_days = 2)
  (h3 : video_game_percentage = 3/4)
  (h4 : pages_per_hour = 60)
  (h5 : pages_per_book = 80) :
  (free_time_per_day * weekend_days * (1 - video_game_percentage) * pages_per_hour) / pages_per_book = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_reads_three_books_l2585_258577


namespace NUMINAMATH_CALUDE_jaymee_is_22_l2585_258505

/-- The age of Shara -/
def shara_age : ℕ := 10

/-- The age of Jaymee -/
def jaymee_age : ℕ := 2 * shara_age + 2

/-- Theorem stating Jaymee's age is 22 -/
theorem jaymee_is_22 : jaymee_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_jaymee_is_22_l2585_258505


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2585_258527

/-- A straight line passing through (-3, -2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-3, -2) -/
  passes_through_point : slope * (-3) + y_intercept = -2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : ∃ (a : ℝ), a ≠ 0 ∧ slope * a + y_intercept = 0 ∧ a + y_intercept = 0

/-- The equation of the line is either 2x - 3y = 0 or x + y + 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 2/3 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = -5) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2585_258527


namespace NUMINAMATH_CALUDE_sum_of_first_100_inverse_terms_l2585_258564

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + n + 1

def sequence_inverse_a (n : ℕ) : ℚ := 1 / sequence_a n

theorem sum_of_first_100_inverse_terms :
  (Finset.range 100).sum sequence_inverse_a = 200 / 101 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_inverse_terms_l2585_258564


namespace NUMINAMATH_CALUDE_min_draw_same_number_and_suit_min_draw_consecutive_numbers_l2585_258547

/-- Represents a card in the deck -/
structure Card where
  suit : Fin 4
  number : Fin 13

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- The number of cards in the deck -/
def deck_size : Nat := 52

/-- The number of suits in the deck -/
def num_suits : Nat := 4

/-- The number of cards per suit -/
def cards_per_suit : Nat := 13

theorem min_draw_same_number_and_suit :
  ∀ (S : Finset Card), S ⊆ Deck → S.card = 27 →
    ∃ (c1 c2 : Card), c1 ∈ S ∧ c2 ∈ S ∧ c1 ≠ c2 ∧ c1.number = c2.number ∧ c1.suit = c2.suit :=
sorry

theorem min_draw_consecutive_numbers :
  ∀ (S : Finset Card), S ⊆ Deck → S.card = 37 →
    ∃ (c1 c2 c3 : Card), c1 ∈ S ∧ c2 ∈ S ∧ c3 ∈ S ∧
      c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      (c1.number + 1 = c2.number ∧ c2.number + 1 = c3.number ∨
       c2.number + 1 = c1.number ∧ c1.number + 1 = c3.number ∨
       c1.number + 1 = c3.number ∧ c3.number + 1 = c2.number) :=
sorry

end NUMINAMATH_CALUDE_min_draw_same_number_and_suit_min_draw_consecutive_numbers_l2585_258547


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2585_258512

open Real

theorem quadratic_equation_solution (A : ℝ) (h1 : 0 < A) (h2 : A < π) :
  (∃ x y : ℝ, x^2 * cos A - 2*x + cos A = 0 ∧
              y^2 * cos A - 2*y + cos A = 0 ∧
              x^2 - y^2 = 3/8) →
  sin A = (sqrt 265 - 16) / 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2585_258512


namespace NUMINAMATH_CALUDE_term2017_is_one_sixty_fifth_l2585_258506

/-- A proper fraction is a pair of natural numbers (n, d) where 0 < n < d -/
def ProperFraction := { p : ℕ × ℕ // 0 < p.1 ∧ p.1 < p.2 }

/-- The sequence of proper fractions arranged by increasing denominators and numerators -/
def properFractionSequence : ℕ → ProperFraction :=
  sorry

/-- The 2017th term of the proper fraction sequence -/
def term2017 : ProperFraction :=
  properFractionSequence 2017

theorem term2017_is_one_sixty_fifth :
  term2017 = ⟨(1, 65), sorry⟩ := by sorry

end NUMINAMATH_CALUDE_term2017_is_one_sixty_fifth_l2585_258506


namespace NUMINAMATH_CALUDE_delivery_driver_stops_l2585_258556

theorem delivery_driver_stops (initial_stops additional_stops : ℕ) 
  (h1 : initial_stops = 3) 
  (h2 : additional_stops = 4) : 
  initial_stops + additional_stops = 7 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_stops_l2585_258556


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l2585_258531

/-- The probability of exactly k successes in n independent trials with success probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The main theorem: probability of 3 successes in 7 trials with p = 2/3 is 280/2187 -/
theorem chocolate_milk_probability :
  binomial_probability 7 3 (2/3) = 280/2187 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l2585_258531


namespace NUMINAMATH_CALUDE_molecular_weight_CaO_l2585_258543

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- A compound with 1 Calcium atom and 1 Oxygen atom -/
structure CaO where
  ca : ℕ := 1
  o : ℕ := 1

/-- The molecular weight of a compound is the sum of the atomic weights of its constituent atoms -/
def molecular_weight (c : CaO) : ℝ := c.ca * atomic_weight_Ca + c.o * atomic_weight_O

theorem molecular_weight_CaO :
  molecular_weight { ca := 1, o := 1 : CaO } = 56.08 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_CaO_l2585_258543


namespace NUMINAMATH_CALUDE_new_average_production_l2585_258510

theorem new_average_production (n : ℕ) (old_average : ℝ) (today_production : ℝ) 
  (h1 : n = 5)
  (h2 : old_average = 60)
  (h3 : today_production = 90) :
  let total_production := n * old_average
  let new_total_production := total_production + today_production
  let new_days := n + 1
  (new_total_production / new_days : ℝ) = 65 := by sorry

end NUMINAMATH_CALUDE_new_average_production_l2585_258510


namespace NUMINAMATH_CALUDE_zoo_animals_count_l2585_258580

theorem zoo_animals_count (female_count : ℕ) (male_excess : ℕ) : 
  female_count = 35 → male_excess = 7 → female_count + (female_count + male_excess) = 77 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l2585_258580


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l2585_258558

/-- Represents the number of rooms that can be painted with the initial amount of paint -/
def initial_rooms : ℕ := 50

/-- Represents the number of rooms that can be painted after losing two cans -/
def remaining_rooms : ℕ := 42

/-- Represents the number of cans lost -/
def lost_cans : ℕ := 2

/-- Calculates the number of cans used to paint the remaining rooms -/
def cans_used : ℕ := 
  let rooms_per_can := (initial_rooms - remaining_rooms) / lost_cans
  (remaining_rooms + rooms_per_can - 1) / rooms_per_can

theorem paint_cans_theorem : cans_used = 11 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l2585_258558


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l2585_258588

theorem cos_sin_sum_equals_sqrt3_over_2 :
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l2585_258588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2585_258500

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7)
  (h_arith : arithmetic_sequence a) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2585_258500


namespace NUMINAMATH_CALUDE_sample_size_from_model_a_l2585_258557

/-- Represents the ratio of quantities for models A, B, and C -/
structure ProductRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents a stratified sample -/
structure StratifiedSample :=
  (size : ℕ) (model_a_count : ℕ)

/-- Theorem: Given the product ratio and model A count in a stratified sample, 
    prove the total sample size -/
theorem sample_size_from_model_a
  (ratio : ProductRatio)
  (sample : StratifiedSample)
  (h_ratio : ratio = ⟨3, 4, 7⟩)
  (h_model_a : sample.model_a_count = 15) :
  sample.size = 70 :=
sorry

end NUMINAMATH_CALUDE_sample_size_from_model_a_l2585_258557


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2585_258594

theorem quadratic_root_value (m : ℝ) : 
  (1 : ℝ)^2 + (1 : ℝ) - m = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2585_258594


namespace NUMINAMATH_CALUDE_equal_savings_l2585_258532

/-- Represents the financial situation of Uma and Bala -/
structure FinancialSituation where
  uma_income : ℝ
  bala_income : ℝ
  uma_expenditure : ℝ
  bala_expenditure : ℝ

/-- The conditions given in the problem -/
def problem_conditions (fs : FinancialSituation) : Prop :=
  fs.uma_income / fs.bala_income = 8 / 7 ∧
  fs.uma_expenditure / fs.bala_expenditure = 7 / 6 ∧
  fs.uma_income = 16000

/-- The savings of Uma and Bala -/
def savings (fs : FinancialSituation) : ℝ × ℝ :=
  (fs.uma_income - fs.uma_expenditure, fs.bala_income - fs.bala_expenditure)

/-- The theorem to be proved -/
theorem equal_savings (fs : FinancialSituation) :
  problem_conditions fs → savings fs = (2000, 2000) := by
  sorry


end NUMINAMATH_CALUDE_equal_savings_l2585_258532


namespace NUMINAMATH_CALUDE_find_divisor_l2585_258509

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 12401)
  (h2 : quotient = 76)
  (h3 : remainder = 13)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) : 
  dividend / quotient = 163 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2585_258509


namespace NUMINAMATH_CALUDE_intersection_point_of_linear_function_and_inverse_l2585_258539

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

-- Define the theorem
theorem intersection_point_of_linear_function_and_inverse
  (b : ℤ) (a : ℤ) :
  (f b (-4) = a ∧ f b a = -4) → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_linear_function_and_inverse_l2585_258539


namespace NUMINAMATH_CALUDE_unique_pair_satisfies_conditions_l2585_258566

/-- Represents the property that a product can be factored into more than one pair of integers (a, b) where b > a > 1 -/
def HasMultipleFactorizations (n : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂ : ℕ, 1 < a₁ ∧ a₁ < b₁ ∧ 1 < a₂ ∧ a₂ < b₂ ∧ a₁ ≠ a₂ ∧ n = a₁ * b₁ ∧ n = a₂ * b₂

/-- Represents the property that for a given sum, all possible pairs (a, b) that sum to it have products with multiple factorizations -/
def AllProductsHaveMultipleFactorizations (s : ℕ) : Prop :=
  ∀ a b : ℕ, 1 < a → a < b → a + b = s → HasMultipleFactorizations (a * b)

theorem unique_pair_satisfies_conditions (a b : ℕ) :
  1 < a →
  a < b →
  a ≤ 20 →
  b ≤ 20 →
  HasMultipleFactorizations (a * b) →
  AllProductsHaveMultipleFactorizations (a + b) →
  ¬(HasMultipleFactorizations (a * b) ∧ AllProductsHaveMultipleFactorizations (a + b)) →
  a = 4 ∧ b = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_satisfies_conditions_l2585_258566


namespace NUMINAMATH_CALUDE_find_a_range_of_m_l2585_258570

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 2
theorem find_a (a : ℝ) : 
  (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 := by sorry

-- Theorem 2: Prove the range of m
theorem range_of_m : 
  ∀ x, f 2 x + f 2 (x + 5) ≥ 5 ∧ 
  ∀ ε > 0, ∃ x, f 2 x + f 2 (x + 5) < 5 + ε := by sorry

end NUMINAMATH_CALUDE_find_a_range_of_m_l2585_258570


namespace NUMINAMATH_CALUDE_unique_number_with_digit_sum_product_l2585_258544

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that 251 is the unique positive integer whose product
    with the sum of its digits equals 2008 -/
theorem unique_number_with_digit_sum_product : ∃! n : ℕ+, (n : ℕ) * sum_of_digits n = 2008 :=
  sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_sum_product_l2585_258544


namespace NUMINAMATH_CALUDE_sum_lent_calculation_l2585_258565

/-- Proves that the sum lent is 300 given the specified conditions -/
theorem sum_lent_calculation (interest_rate : ℚ) (time_period : ℕ) (interest_difference : ℕ) :
  interest_rate = 4/100 →
  time_period = 8 →
  interest_difference = 204 →
  (interest_rate * time_period * 300 : ℚ) = 300 - interest_difference :=
by sorry

end NUMINAMATH_CALUDE_sum_lent_calculation_l2585_258565


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l2585_258582

theorem geometric_progression_proof (b q : ℝ) : 
  b + b*q + b*q^2 + b*q^3 = -40 ∧ 
  b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280 → 
  b = 2 ∧ q = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l2585_258582


namespace NUMINAMATH_CALUDE_gold_cube_value_scaling_l2585_258523

-- Define the properties of the 4-inch gold cube
def gold_cube_4inch_value : ℝ := 500
def gold_cube_4inch_side : ℝ := 4

-- Define the side length of the 5-inch gold cube
def gold_cube_5inch_side : ℝ := 5

-- Function to calculate the volume of a cube
def cube_volume (side : ℝ) : ℝ := side ^ 3

-- Theorem statement
theorem gold_cube_value_scaling :
  let v4 := cube_volume gold_cube_4inch_side
  let v5 := cube_volume gold_cube_5inch_side
  let scale_factor := v5 / v4
  let scaled_value := gold_cube_4inch_value * scale_factor
  ⌊scaled_value + 0.5⌋ = 977 := by sorry

end NUMINAMATH_CALUDE_gold_cube_value_scaling_l2585_258523


namespace NUMINAMATH_CALUDE_quadratic_solution_set_implies_coefficients_l2585_258533

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 + 5 * x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x : ℝ, f a c x > 0 ↔ 1/3 < x ∧ x < 1/2

-- Theorem statement
theorem quadratic_solution_set_implies_coefficients :
  ∀ a c : ℝ, solution_set a c → a = -6 ∧ c = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_implies_coefficients_l2585_258533


namespace NUMINAMATH_CALUDE_total_wood_planks_l2585_258520

def initial_planks : ℕ := 15
def charlie_planks : ℕ := 10
def father_planks : ℕ := 10

theorem total_wood_planks : 
  initial_planks + charlie_planks + father_planks = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_wood_planks_l2585_258520


namespace NUMINAMATH_CALUDE_fourth_person_height_l2585_258579

/-- Represents a person with height, weight, and age -/
structure Person where
  height : ℝ
  weight : ℝ
  age : ℕ

/-- Given conditions for the problem -/
def fourPeople (p1 p2 p3 p4 : Person) : Prop :=
  p1.height < p2.height ∧ p2.height < p3.height ∧ p3.height < p4.height ∧
  p2.height - p1.height = 2 ∧
  p3.height - p2.height = 3 ∧
  p4.height - p3.height = 6 ∧
  p1.weight + p2.weight + p3.weight + p4.weight = 600 ∧
  p1.age = 25 ∧ p2.age = 32 ∧ p3.age = 37 ∧ p4.age = 46 ∧
  (p1.height + p2.height + p3.height + p4.height) / 4 = 72 ∧
  ∀ (i j : Fin 4), (i.val < j.val) → 
    (p1.height * p1.age = p2.height * p2.age) ∧
    (p1.height * p2.weight = p2.height * p1.weight)

/-- Theorem: The fourth person's height is 78.5 inches -/
theorem fourth_person_height (p1 p2 p3 p4 : Person) 
  (h : fourPeople p1 p2 p3 p4) : p4.height = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l2585_258579


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l2585_258561

theorem cricket_team_captain_age (team_size : Nat) (captain_age wicket_keeper_age : Nat) 
  (remaining_players_avg_age team_avg_age : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 5 →
  remaining_players_avg_age = team_avg_age - 1 →
  team_avg_age = 24 →
  (team_size - 2 : ℚ) * remaining_players_avg_age + captain_age + wicket_keeper_age = 
    team_size * team_avg_age →
  captain_age = 26 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l2585_258561


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l2585_258528

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ, 
    is_prime p ∧ 
    is_prime q ∧ 
    p > 30 ∧ 
    q > 30 ∧ 
    p ≠ q ∧ 
    p * q = 1147 ∧ 
    (∀ p' q' : ℕ, is_prime p' → is_prime q' → p' > 30 → q' > 30 → p' ≠ q' → p' * q' ≥ 1147) :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l2585_258528


namespace NUMINAMATH_CALUDE_number_is_composite_l2585_258554

theorem number_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^1962 + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_number_is_composite_l2585_258554


namespace NUMINAMATH_CALUDE_monomial_sum_implies_mn_twelve_l2585_258502

/-- If the sum of 2x³yⁿ and -½xᵐy⁴ is a monomial, then mn = 12 -/
theorem monomial_sum_implies_mn_twelve (x y : ℝ) (m n : ℕ) :
  (∃ (c : ℝ), ∀ x y, 2 * x^3 * y^n - 1/2 * x^m * y^4 = c * x^3 * y^4) →
  m * n = 12 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_mn_twelve_l2585_258502


namespace NUMINAMATH_CALUDE_b_nonempty_implies_a_geq_two_thirds_a_intersect_b_eq_b_implies_a_geq_two_l2585_258525

-- Define sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | (1/2) * a ≤ x ∧ x ≤ 2*a - 1}

-- Theorem 1: If B is non-empty, then a ≥ 2/3
theorem b_nonempty_implies_a_geq_two_thirds (a : ℝ) :
  (B a).Nonempty → a ≥ 2/3 := by sorry

-- Theorem 2: If A ∩ B = B, then a ≥ 2
theorem a_intersect_b_eq_b_implies_a_geq_two (a : ℝ) :
  A ∩ (B a) = B a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_b_nonempty_implies_a_geq_two_thirds_a_intersect_b_eq_b_implies_a_geq_two_l2585_258525


namespace NUMINAMATH_CALUDE_red_pencils_count_l2585_258599

theorem red_pencils_count (total_packs : ℕ) (normal_red_per_pack : ℕ) (special_packs : ℕ) (extra_red_per_special : ℕ) : 
  total_packs = 15 → 
  normal_red_per_pack = 1 → 
  special_packs = 3 → 
  extra_red_per_special = 2 → 
  total_packs * normal_red_per_pack + special_packs * extra_red_per_special = 21 := by
sorry

end NUMINAMATH_CALUDE_red_pencils_count_l2585_258599


namespace NUMINAMATH_CALUDE_scientific_notation_320000_l2585_258503

theorem scientific_notation_320000 : 
  320000 = 3.2 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_320000_l2585_258503


namespace NUMINAMATH_CALUDE_marly_bills_denomination_l2585_258575

theorem marly_bills_denomination (x : ℕ) : 
  (10 * 20 + 8 * x + 4 * 5 = 3 * 100) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_marly_bills_denomination_l2585_258575


namespace NUMINAMATH_CALUDE_exists_question_with_different_answers_l2585_258569

/-- Represents a person who always tells the truth -/
structure TruthTeller where
  name : String
  always_truthful : Bool

/-- Represents a question that can be asked -/
inductive Question where
  | count_questions : Question

/-- Represents the state of the conversation -/
structure ConversationState where
  questions_asked : Nat

/-- The answer given by a TruthTeller to a Question in a given ConversationState -/
def answer (person : TruthTeller) (q : Question) (state : ConversationState) : Nat :=
  match q with
  | Question.count_questions => state.questions_asked

/-- Theorem stating that there exists a question that can have different truthful answers when asked twice -/
theorem exists_question_with_different_answers (ilya : TruthTeller) 
    (h_truthful : ilya.always_truthful = true) :
    ∃ (q : Question) (s1 s2 : ConversationState), 
      s1 ≠ s2 ∧ answer ilya q s1 ≠ answer ilya q s2 := by
  sorry


end NUMINAMATH_CALUDE_exists_question_with_different_answers_l2585_258569


namespace NUMINAMATH_CALUDE_concentric_polygons_inequality_l2585_258504

theorem concentric_polygons_inequality (n : ℕ) (R r : ℝ) (h : Fin n → ℝ) :
  n ≥ 3 →
  R > 0 →
  r > 0 →
  r < R →
  (∀ i, h i > 0) →
  (∀ i, h i ≤ R) →
  R * Real.cos (π / n) ≥ (Finset.sum Finset.univ h) / n ∧ (Finset.sum Finset.univ h) / n ≥ r :=
by sorry

end NUMINAMATH_CALUDE_concentric_polygons_inequality_l2585_258504


namespace NUMINAMATH_CALUDE_max_abs_z_value_l2585_258598

/-- Given complex numbers a, b, c, z satisfying the conditions, 
    the maximum value of |z| is 1 + √2 -/
theorem max_abs_z_value (a b c z : ℂ) (r : ℝ) 
  (hr : r > 0)
  (ha : Complex.abs a = r)
  (hb : Complex.abs b = 2*r)
  (hc : Complex.abs c = r)
  (heq : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l2585_258598


namespace NUMINAMATH_CALUDE_gym_class_students_l2585_258595

theorem gym_class_students (n : ℕ) : 
  150 ≤ n ∧ n ≤ 300 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 2 → 
  n = 165 ∨ n = 237 := by
sorry

end NUMINAMATH_CALUDE_gym_class_students_l2585_258595


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2585_258590

/-- Simple interest calculation -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 10000) 
  (h2 : rate = 0.05) 
  (h3 : time = 1) :
  principal * rate * time = 500 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2585_258590


namespace NUMINAMATH_CALUDE_f_properties_l2585_258581

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 - x

theorem f_properties :
  -- Part 1: Monotonicity
  (∀ x ≥ 1, ∀ y ≥ x, f y ≤ f x) ∧
  -- Part 2: Inequality for a ≥ 2
  (∀ a ≥ 2, ∀ x > 0, f x < (a/2 - 1) * x^2 + a * x - 1) ∧
  -- Part 3: Inequality for x1 and x2
  (∀ x1 > 0, ∀ x2 > 0,
    f x1 + f x2 + 2 * (x1^2 + x2^2) + x1 * x2 = 0 →
    x1 + x2 ≥ (Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2585_258581


namespace NUMINAMATH_CALUDE_rent_during_harvest_l2585_258524

/-- The total rent paid during the harvest season -/
def total_rent (weekly_rent : ℕ) (weeks : ℕ) : ℕ :=
  weekly_rent * weeks

/-- Proof that the total rent paid during the harvest season is $526,692 -/
theorem rent_during_harvest : total_rent 388 1359 = 526692 := by
  sorry

end NUMINAMATH_CALUDE_rent_during_harvest_l2585_258524


namespace NUMINAMATH_CALUDE_inequality_proof_l2585_258567

theorem inequality_proof (a b : ℝ) 
  (h : ∀ x : ℝ, Real.cos (a * Real.sin x) > Real.sin (b * Real.cos x)) : 
  a^2 + b^2 < (Real.pi^2) / 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2585_258567


namespace NUMINAMATH_CALUDE_proportionality_check_l2585_258530

-- Define the type of proportionality
inductive Proportionality
  | Direct
  | Inverse
  | Neither

-- Define a function to check proportionality
def check_proportionality (eq : ℝ → ℝ → Prop) : Proportionality :=
  sorry

-- Theorem statement
theorem proportionality_check :
  (check_proportionality (fun x y => 2*x + y = 5) = Proportionality.Neither) ∧
  (check_proportionality (fun x y => 4*x*y = 15) = Proportionality.Inverse) ∧
  (check_proportionality (fun x y => x = 7*y) = Proportionality.Direct) ∧
  (check_proportionality (fun x y => 2*x + 3*y = 12) = Proportionality.Neither) ∧
  (check_proportionality (fun x y => x/y = 4) = Proportionality.Direct) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_check_l2585_258530


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l2585_258535

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle given its dimensions -/
def perimeter (d : Dimensions) : ℕ := 2 * (d.length + d.width)

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the tiling pattern of the large rectangle -/
structure TilingPattern where
  inner : Dimensions
  redTiles : ℕ

theorem large_rectangle_perimeter 
  (pattern : TilingPattern)
  (h1 : pattern.redTiles = 2900) :
  ∃ (large : Dimensions), 
    area large = area pattern.inner + 2900 + 2 * area { length := pattern.inner.length + 20, width := pattern.inner.width + 20 } ∧ 
    perimeter large = 350 := by
  sorry


end NUMINAMATH_CALUDE_large_rectangle_perimeter_l2585_258535


namespace NUMINAMATH_CALUDE_golden_ratio_pentagon_l2585_258562

theorem golden_ratio_pentagon (θ : Real) : 
  θ = 108 * Real.pi / 180 →  -- Interior angle of a regular pentagon
  2 * Real.sin (18 * Real.pi / 180) = (Real.sqrt 5 - 1) / 2 →
  Real.sin θ / Real.sin (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_pentagon_l2585_258562


namespace NUMINAMATH_CALUDE_spring_earnings_calculation_l2585_258545

def spring_earnings (summer_earnings total_spent final_amount : ℕ) : ℕ :=
  total_spent + final_amount - summer_earnings

theorem spring_earnings_calculation 
  (summer_earnings total_spent final_amount : ℕ) :
  spring_earnings summer_earnings total_spent final_amount = 4 :=
by
  sorry

#eval spring_earnings 50 4 50

end NUMINAMATH_CALUDE_spring_earnings_calculation_l2585_258545


namespace NUMINAMATH_CALUDE_sequence_recurrence_problem_l2585_258591

/-- Given a sequence of positive real numbers {a_n} (n ≥ 0) satisfying the recurrence relation
    a_n = a_{n-1} / (m * a_{n-2}) for n ≥ 2, where m is a real parameter,
    prove that if a_2009 = a_0 / a_1, then m = 1. -/
theorem sequence_recurrence_problem (a : ℕ → ℝ) (m : ℝ) 
    (h_positive : ∀ n, a n > 0)
    (h_recurrence : ∀ n ≥ 2, a n = a (n-1) / (m * a (n-2)))
    (h_equality : a 2009 = a 0 / a 1) :
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_recurrence_problem_l2585_258591


namespace NUMINAMATH_CALUDE_bacteria_growth_l2585_258517

/-- The number of times a bacteria culture doubles in 4 minutes -/
def doublings : ℕ := 240 / 30

/-- The final number of bacteria after 4 minutes -/
def final_count : ℕ := 524288

theorem bacteria_growth (n : ℕ) : n * 2^doublings = final_count ↔ n = 2048 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2585_258517


namespace NUMINAMATH_CALUDE_triangle_problem_l2585_258507

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  (2 * t.b * (2 * t.b - t.c) * Real.cos t.A = t.a^2 + t.b^2 - t.c^2) →
  ((1/2) * t.b * t.c * Real.sin t.A = 25 * Real.sqrt 3 / 4) →
  (t.a = 5) →
  -- Conclusions
  (t.A = π/3 ∧ t.b + t.c = 10) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l2585_258507


namespace NUMINAMATH_CALUDE_alice_quarters_l2585_258529

/-- Represents the number of quarters Alice had initially -/
def initial_quarters : ℕ := 20

/-- Represents the number of nickels Alice received after exchange -/
def total_nickels : ℕ := 100

/-- Represents the value of a regular nickel in dollars -/
def regular_nickel_value : ℚ := 1/20

/-- Represents the value of an iron nickel in dollars -/
def iron_nickel_value : ℚ := 3

/-- Represents the proportion of iron nickels -/
def iron_nickel_proportion : ℚ := 1/5

/-- Represents the proportion of regular nickels -/
def regular_nickel_proportion : ℚ := 4/5

/-- Represents the total value of all nickels in dollars -/
def total_value : ℚ := 64

theorem alice_quarters :
  (iron_nickel_proportion * total_nickels * iron_nickel_value + 
   regular_nickel_proportion * total_nickels * regular_nickel_value = total_value) ∧
  (initial_quarters * 5 = total_nickels) := by
  sorry

end NUMINAMATH_CALUDE_alice_quarters_l2585_258529


namespace NUMINAMATH_CALUDE_correct_observation_value_l2585_258501

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 40) 
  (h2 : initial_mean = 100) 
  (h3 : wrong_value = 75) 
  (h4 : corrected_mean = 99.075) : 
  (n : ℝ) * corrected_mean - ((n : ℝ) * initial_mean - wrong_value) = 38 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2585_258501


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2585_258542

theorem equal_roots_quadratic (a : ℝ) : 
  (∃! x : ℝ, x^2 - a*x + 3*a = 0) ↔ (a = 0 ∨ a = 12) :=
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2585_258542


namespace NUMINAMATH_CALUDE_point_relationships_l2585_258536

def A : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2*x + y - 5 ≤ 0}

theorem point_relationships :
  (¬ ((0 : ℝ), (0 : ℝ)) ∈ A) ∧ ((1 : ℝ), (1 : ℝ)) ∈ A := by sorry

end NUMINAMATH_CALUDE_point_relationships_l2585_258536


namespace NUMINAMATH_CALUDE_sugar_added_indeterminate_l2585_258576

-- Define the recipe requirements
def total_flour : ℕ := 9
def total_sugar : ℕ := 5

-- Define Mary's current actions
def flour_added : ℕ := 3
def flour_to_add : ℕ := 6

-- Define a variable for the unknown amount of sugar added
variable (sugar_added : ℕ)

-- Theorem stating that sugar_added cannot be uniquely determined
theorem sugar_added_indeterminate : 
  ∀ (x y : ℕ), x ≠ y → 
  (x ≤ total_sugar ∧ y ≤ total_sugar) → 
  (∃ (state₁ state₂ : ℕ × ℕ), 
    state₁.1 = flour_added ∧ 
    state₁.2 = x ∧ 
    state₂.1 = flour_added ∧ 
    state₂.2 = y) :=
by sorry

end NUMINAMATH_CALUDE_sugar_added_indeterminate_l2585_258576


namespace NUMINAMATH_CALUDE_max_value_of_f_l2585_258515

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8

-- State the theorem
theorem max_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2585_258515


namespace NUMINAMATH_CALUDE_inequality_proof_l2585_258584

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (1/2) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2585_258584


namespace NUMINAMATH_CALUDE_yoga_studio_women_count_l2585_258574

theorem yoga_studio_women_count :
  let num_men : ℕ := 8
  let avg_weight_men : ℚ := 190
  let avg_weight_women : ℚ := 120
  let total_people : ℕ := 14
  let avg_weight_all : ℚ := 160
  let num_women : ℕ := total_people - num_men
  (num_men : ℚ) * avg_weight_men + (num_women : ℚ) * avg_weight_women = (total_people : ℚ) * avg_weight_all →
  num_women = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_women_count_l2585_258574


namespace NUMINAMATH_CALUDE_max_prob_second_game_l2585_258521

variable (p₁ p₂ p₃ : ℝ)

def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

theorem max_prob_second_game (h1 : 0 < p₁) (h2 : p₁ < p₂) (h3 : p₂ < p₃) :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
by sorry

end NUMINAMATH_CALUDE_max_prob_second_game_l2585_258521


namespace NUMINAMATH_CALUDE_coefficient_is_three_l2585_258508

/-- The derivative function for our equation -/
noncomputable def derivative (q : ℝ) : ℝ := 3 * q - 3

/-- The second derivative of 6 -/
def second_derivative_of_six : ℝ := 210

/-- The coefficient of q in the equation q' = 3q - 3 -/
def coefficient : ℝ := 3

theorem coefficient_is_three : coefficient = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_is_three_l2585_258508


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2585_258552

theorem cubic_root_sum (p q r : ℝ) : 
  (3 * p^3 - 5 * p^2 + 50 * p - 7 = 0) →
  (3 * q^3 - 5 * q^2 + 50 * q - 7 = 0) →
  (3 * r^3 - 5 * r^2 + 50 * r - 7 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 249/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2585_258552


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2585_258589

def is_valid_number (a b : Nat) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ (100000 * a + 19880 + b) % 12 = 0

theorem count_valid_numbers :
  ∃ (S : Finset (Nat × Nat)),
    (∀ (p : Nat × Nat), p ∈ S ↔ is_valid_number p.1 p.2) ∧
    S.card = 9 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2585_258589


namespace NUMINAMATH_CALUDE_sample_size_is_200_l2585_258516

/-- The expected sample size for a school with given student counts and selection probability -/
def expected_sample_size (freshmen sophomores juniors : ℕ) (prob : ℝ) : ℝ :=
  (freshmen + sophomores + juniors : ℝ) * prob

/-- Theorem stating that the expected sample size is 200 for the given school population and selection probability -/
theorem sample_size_is_200 :
  expected_sample_size 280 320 400 0.2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_200_l2585_258516


namespace NUMINAMATH_CALUDE_corner_sum_implies_bottom_right_l2585_258559

/-- Represents a 24 by 24 grid containing numbers 1 to 576 -/
def Grid := Fin 24 → Fin 24 → Nat

/-- Checks if a given number is in the grid -/
def in_grid (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 576

/-- Defines a valid 24 by 24 grid -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ i j, in_grid (g i j) ∧ g i j = i * 24 + j + 1

/-- Represents an 8 by 8 square within the grid -/
structure Square (g : Grid) where
  top_left : Fin 24 × Fin 24
  h_valid : top_left.1 + 7 < 24 ∧ top_left.2 + 7 < 24

/-- Gets the corner values of an 8 by 8 square -/
def corner_values (g : Grid) (s : Square g) : Fin 4 → Nat
| 0 => g s.top_left.1 s.top_left.2
| 1 => g s.top_left.1 (s.top_left.2 + 7)
| 2 => g (s.top_left.1 + 7) s.top_left.2
| 3 => g (s.top_left.1 + 7) (s.top_left.2 + 7)
| _ => 0

/-- The main theorem -/
theorem corner_sum_implies_bottom_right (g : Grid) (s : Square g) :
  is_valid_grid g →
  (corner_values g s 0 + corner_values g s 1 + corner_values g s 2 + corner_values g s 3 = 1646) →
  corner_values g s 3 = 499 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_implies_bottom_right_l2585_258559


namespace NUMINAMATH_CALUDE_smallest_possible_campers_l2585_258538

/-- Represents the number of campers participating in different combinations of activities -/
structure CampActivities where
  only_canoeing : ℕ
  canoeing_swimming : ℕ
  only_swimming : ℕ
  canoeing_fishing : ℕ
  swimming_fishing : ℕ
  only_fishing : ℕ

/-- Represents the camp with its activities and camper counts -/
structure Camp where
  activities : CampActivities
  no_activity : ℕ

/-- Calculates the total number of campers in the camp -/
def total_campers (camp : Camp) : ℕ :=
  camp.activities.only_canoeing +
  camp.activities.canoeing_swimming +
  camp.activities.only_swimming +
  camp.activities.canoeing_fishing +
  camp.activities.swimming_fishing +
  camp.activities.only_fishing +
  camp.no_activity

/-- Checks if the camp satisfies the given conditions -/
def satisfies_conditions (camp : Camp) : Prop :=
  (camp.activities.only_canoeing + camp.activities.canoeing_swimming + camp.activities.canoeing_fishing = 15) ∧
  (camp.activities.canoeing_swimming + camp.activities.only_swimming + camp.activities.swimming_fishing = 22) ∧
  (camp.activities.canoeing_fishing + camp.activities.swimming_fishing + camp.activities.only_fishing = 12) ∧
  (camp.no_activity = 9)

theorem smallest_possible_campers :
  ∀ camp : Camp,
    satisfies_conditions camp →
    total_campers camp ≥ 34 :=
by sorry

#check smallest_possible_campers

end NUMINAMATH_CALUDE_smallest_possible_campers_l2585_258538


namespace NUMINAMATH_CALUDE_farm_animal_difference_l2585_258540

/-- Proves that the difference between the number of goats and pigs is 33 -/
theorem farm_animal_difference : 
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let ducks : ℕ := (goats + chickens) / 2
  let pigs : ℕ := ducks / 3
  goats - pigs = 33 := by sorry

end NUMINAMATH_CALUDE_farm_animal_difference_l2585_258540


namespace NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l2585_258585

/-- Given a point C with coordinates (3, y) and its reflection D over the x-axis,
    the sum of all four coordinates of C and D is 6. -/
theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)
  C.1 + C.2 + D.1 + D.2 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l2585_258585


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2585_258518

-- Define the hyperbola and its properties
def Hyperbola (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2

-- Define the point P on the right branch of the hyperbola
def PointOnHyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (x, y) := P
  x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0

-- Define the right focus F₂
def RightFocus (F₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  F₂ = (c, 0)

-- Define the midpoint M of PF₂
def Midpoint (M P F₂ : ℝ × ℝ) : Prop :=
  M = ((P.1 + F₂.1) / 2, (P.2 + F₂.2) / 2)

-- Define the property |OF₂| = |F₂M|
def EqualDistances (O F₂ M : ℝ × ℝ) : Prop :=
  (F₂.1 - O.1)^2 + (F₂.2 - O.2)^2 = (M.1 - F₂.1)^2 + (M.2 - F₂.2)^2

-- Define the dot product property
def DotProductProperty (O F₂ M : ℝ × ℝ) (c : ℝ) : Prop :=
  (F₂.1 - O.1) * (M.1 - F₂.1) + (F₂.2 - O.2) * (M.2 - F₂.2) = c^2 / 2

-- The main theorem
theorem hyperbola_eccentricity 
  (a b c : ℝ) (O P F₂ M : ℝ × ℝ) 
  (h1 : Hyperbola a b c)
  (h2 : PointOnHyperbola P a b)
  (h3 : RightFocus F₂ c)
  (h4 : Midpoint M P F₂)
  (h5 : EqualDistances O F₂ M)
  (h6 : DotProductProperty O F₂ M c)
  (h7 : O = (0, 0)) :
  c / a = (Real.sqrt 3 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2585_258518


namespace NUMINAMATH_CALUDE_x_completion_time_l2585_258572

/-- The time taken by x to complete the work -/
def time_x : ℝ := 40

/-- The time x works on the project -/
def x_work_time : ℝ := 8

/-- The time y takes to finish the remaining work after x -/
def y_finish_time : ℝ := 20

/-- The time y takes to complete the entire work -/
def y_total_time : ℝ := 25

/-- The total work to be done -/
def total_work : ℝ := 1

theorem x_completion_time :
  (x_work_time / time_x + y_finish_time / y_total_time = 1) →
  time_x = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_x_completion_time_l2585_258572


namespace NUMINAMATH_CALUDE_function_composition_result_l2585_258511

theorem function_composition_result (a b : ℝ) :
  (∀ x, (3 * ((a * x) + b) - 4) = 4 * x + 3) →
  a + b = 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_result_l2585_258511


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2585_258546

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2585_258546


namespace NUMINAMATH_CALUDE_two_tailed_coin_probability_l2585_258513

/-- The probability of drawing the 2-tailed coin given that the flip resulted in tails -/
def prob_two_tailed_given_tails (total_coins : ℕ) (fair_coins : ℕ) (p_tails_fair : ℚ) : ℚ :=
  let two_tailed_coins := total_coins - fair_coins
  let p_two_tailed := two_tailed_coins / total_coins
  let p_tails_two_tailed := 1
  let p_tails := p_two_tailed * p_tails_two_tailed + (fair_coins / total_coins) * p_tails_fair
  (p_tails_two_tailed * p_two_tailed) / p_tails

theorem two_tailed_coin_probability :
  prob_two_tailed_given_tails 10 9 (1/2) = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_two_tailed_coin_probability_l2585_258513


namespace NUMINAMATH_CALUDE_woman_birth_year_l2585_258568

/-- A woman born in the first half of the twentieth century was x years old in the year x^2. This theorem proves her birth year was 1892. -/
theorem woman_birth_year :
  ∃ (x : ℕ),
    (x^2 : ℕ) < 2000 ∧  -- Born in the first half of the 20th century
    (x^2 : ℕ) ≥ 1900 ∧  -- Born in the 20th century
    (x^2 - x : ℕ) = 1892  -- Birth year calculation
  := by sorry

#check woman_birth_year

end NUMINAMATH_CALUDE_woman_birth_year_l2585_258568


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2585_258571

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h1 : a 1 = 1024)  -- First term is 1024
  (h2 : a 8 = 125)   -- 8th term is 125
  (h3 : ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a n = a 1 * r^(n-1))  -- Definition of geometric sequence
  : a 6 = 5^(5/7) * 32 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2585_258571


namespace NUMINAMATH_CALUDE_factorial_divisibility_implies_inequality_l2585_258596

theorem factorial_divisibility_implies_inequality (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) 
  (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : 
  a ≥ 2 * b + 1 := by
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_implies_inequality_l2585_258596


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_not_all_lines_perpendicular_to_line_are_parallel_not_all_planes_perpendicular_to_plane_are_parallel_or_intersect_l2585_258548

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (perpendicular_line_line : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect_plane : Plane → Plane → Prop)

-- Theorem statements
theorem planes_perpendicular_to_line_are_parallel 
  (p1 p2 : Plane) (l : Line) 
  (h1 : perpendicular_plane_line p1 l) 
  (h2 : perpendicular_plane_line p2 l) : 
  parallel_plane p1 p2 :=
sorry

theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular_line_plane l1 p) 
  (h2 : perpendicular_line_plane l2 p) : 
  parallel_line l1 l2 :=
sorry

theorem not_all_lines_perpendicular_to_line_are_parallel : 
  ∃ (l1 l2 l3 : Line), 
    perpendicular_line_line l1 l3 ∧ 
    perpendicular_line_line l2 l3 ∧ 
    ¬(parallel_line l1 l2) :=
sorry

theorem not_all_planes_perpendicular_to_plane_are_parallel_or_intersect : 
  ∃ (p1 p2 p3 : Plane), 
    perpendicular_plane_plane p1 p3 ∧ 
    perpendicular_plane_plane p2 p3 ∧ 
    ¬(parallel_plane p1 p2 ∨ intersect_plane p1 p2) :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_not_all_lines_perpendicular_to_line_are_parallel_not_all_planes_perpendicular_to_plane_are_parallel_or_intersect_l2585_258548


namespace NUMINAMATH_CALUDE_bounds_on_y_l2585_258583

-- Define the equations
def eq1 (x y : ℝ) : Prop := x^2 - 6*x + 2*y = 0
def eq2 (x y : ℝ) : Prop := 3*x^2 + 12*x - 2*y - 4 = 0
def eq3 (x y : ℝ) : Prop := y = 2*x / (1 + x^2)
def eq4 (x y : ℝ) : Prop := y = (2*x - 1) / (x^2 + 2*x + 1)

-- Define the theorem
theorem bounds_on_y :
  ∀ x y : ℝ,
  eq1 x y ∧ eq2 x y ∧ eq3 x y ∧ eq4 x y →
  y ≤ 4.5 ∧ y ≥ -8 ∧ -1 ≤ y ∧ y ≤ 1 ∧ y ≤ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_bounds_on_y_l2585_258583


namespace NUMINAMATH_CALUDE_investment_problem_l2585_258553

def first_investment_value (x : ℝ) : Prop :=
  let second_investment : ℝ := 1500
  let combined_return_rate : ℝ := 0.085
  let first_return_rate : ℝ := 0.07
  let second_return_rate : ℝ := 0.09
  (first_return_rate * x + second_return_rate * second_investment = 
   combined_return_rate * (x + second_investment)) ∧
  x = 500

theorem investment_problem : ∃ x : ℝ, first_investment_value x := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2585_258553


namespace NUMINAMATH_CALUDE_board_cut_theorem_l2585_258534

theorem board_cut_theorem (total_length : ℝ) (x : ℝ) 
  (h1 : total_length = 120)
  (h2 : x = 1.5) : 
  let shorter_piece := total_length / (1 + (2 * x + 1/3))
  let longer_piece := shorter_piece * (2 * x + 1/3)
  longer_piece = 92 + 4/13 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l2585_258534


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2585_258522

theorem power_of_two_equality (x : ℕ) : (1 / 4 : ℝ) * (2 ^ 30) = 2 ^ x → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2585_258522


namespace NUMINAMATH_CALUDE_max_sets_with_even_intersection_l2585_258573

theorem max_sets_with_even_intersection (v : ℕ) (h : v = 2016) :
  (∃ (n : ℕ) (A : Fin n → Finset (Fin v)),
    (∀ i : Fin n, (A i).card = 4) ∧
    (∀ i j : Fin n, i < j → (A i ∩ A j).card % 2 = 0)) ∧
  (∀ m : ℕ, m > 33860 →
    ¬∃ (A : Fin m → Finset (Fin v)),
      (∀ i : Fin m, (A i).card = 4) ∧
      (∀ i j : Fin m, i < j → (A i ∩ A j).card % 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_sets_with_even_intersection_l2585_258573


namespace NUMINAMATH_CALUDE_monotonicity_condition_even_function_condition_minimum_value_l2585_258550

-- Define the function f(x) = x^2 + 2ax
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x

-- Define the domain [-5, 5]
def domain : Set ℝ := Set.Icc (-5) 5

-- Statement 1: Monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x < f a y) ∨
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f a x > f a y) ↔
  a ≤ -5 ∨ a ≥ 5 :=
sorry

-- Statement 2: Even function condition and extrema
theorem even_function_condition (a : ℝ) :
  (∀ x ∈ domain, f a x - 2*x = f a (-x) - 2*(-x)) →
  a = 1 ∧ 
  (∀ x ∈ domain, f a x ≤ 35) ∧
  (∀ x ∈ domain, f a x ≥ -1) ∧
  (∃ x ∈ domain, f a x = 35) ∧
  (∃ x ∈ domain, f a x = -1) :=
sorry

-- Statement 3: Minimum value
theorem minimum_value (a : ℝ) :
  (a ≥ 5 → ∀ x ∈ domain, f a x ≥ 25 - 10*a) ∧
  (a ≤ -5 → ∀ x ∈ domain, f a x ≥ 25 + 10*a) ∧
  (-5 < a ∧ a < 5 → ∀ x ∈ domain, f a x ≥ -a^2) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_even_function_condition_minimum_value_l2585_258550


namespace NUMINAMATH_CALUDE_complex_root_coefficients_l2585_258519

theorem complex_root_coefficients :
  ∀ (b c : ℝ),
  (∃ (z : ℂ), z = 1 + Complex.I * Real.sqrt 2 ∧ z^2 + b*z + c = 0) →
  b = -2 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_root_coefficients_l2585_258519


namespace NUMINAMATH_CALUDE_quadratic_roots_imaginary_l2585_258541

theorem quadratic_roots_imaginary (k : ℝ) : 
  (∃ x y : ℂ, x ≠ y ∧ 3 * x^2 - 5*k*x + 4*k^2 - 2 = 0 ∧ 3 * y^2 - 5*k*x + 4*k^2 - 2 = 0 ∧ x * y = 9) →
  (∀ x : ℝ, 3 * x^2 - 5*k*x + 4*k^2 - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imaginary_l2585_258541


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2585_258586

/-- The lateral surface area of a cone with base radius 2 and slant height 4 is 8π -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 4 → π * r * l = 8 * π :=
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2585_258586


namespace NUMINAMATH_CALUDE_win_sector_area_l2585_258592

/-- Given a circular spinner with radius 8 cm and probability of winning 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_prob : ℝ) (win_area : ℝ) : 
  radius = 8 →
  win_prob = 3 / 8 →
  win_area = win_prob * π * radius^2 →
  win_area = 24 * π := by
  sorry

#check win_sector_area

end NUMINAMATH_CALUDE_win_sector_area_l2585_258592


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2585_258549

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The equation of the first circle: x^2 + y^2 = 1 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The equation of the second circle: x^2 + y^2 - 6x - 8y + 9 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

theorem circles_externally_tangent :
  externally_tangent (0, 0) (3, 4) 1 4 := by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2585_258549


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l2585_258555

/-- The inclination angle of a line is the angle between the positive x-axis and the line, 
    measured counterclockwise. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- The equation of the line is ax + by + c = 0 -/
def is_line_equation (a b c : ℝ) : Prop := sorry

theorem inclination_angle_of_line :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := -5
  is_line_equation a b c →
  inclination_angle a b c = 135 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l2585_258555


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2585_258597

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2585_258597


namespace NUMINAMATH_CALUDE_square_side_length_l2585_258593

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 9) (h2 : rectangle_width = 16) :
  ∃ (square_side : ℝ), square_side ^ 2 = rectangle_length * rectangle_width ∧ square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2585_258593
