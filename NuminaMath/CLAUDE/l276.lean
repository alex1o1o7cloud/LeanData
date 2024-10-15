import Mathlib

namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l276_27607

-- Define the universal set I
def I : Set ℕ := {0, 1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2, 3}

-- Define set N
def N : Set ℕ := {0, 3, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ M) ∩ N = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l276_27607


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l276_27622

/-- The simple interest calculation problem --/
theorem simple_interest_calculation (P : ℝ) : 
  (∀ (r : ℝ) (A : ℝ), 
    r = 0.04 → 
    A = 36.4 → 
    A = P + P * r) → 
  P = 35 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l276_27622


namespace NUMINAMATH_CALUDE_amelia_win_probability_l276_27653

/-- The probability of Amelia winning the coin toss game -/
def ameliaWinProbability (ameliaHeadProb blainHeadProb : ℚ) : ℚ :=
  ameliaHeadProb / (1 - (1 - ameliaHeadProb) * (1 - blainHeadProb))

/-- The coin toss game where Amelia goes first -/
theorem amelia_win_probability :
  let ameliaHeadProb : ℚ := 1/3
  let blaineHeadProb : ℚ := 2/5
  ameliaWinProbability ameliaHeadProb blaineHeadProb = 5/9 := by
  sorry

#eval ameliaWinProbability (1/3) (2/5)

end NUMINAMATH_CALUDE_amelia_win_probability_l276_27653


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l276_27661

theorem equidistant_point_on_y_axis :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (2, 5)
  ∃! y : ℝ, 
    ((-3 - 0)^2 + (0 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧
    y = 2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l276_27661


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l276_27603

/-- Calculates the rate of paving per square meter given room dimensions and total cost -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 4 ∧ total_cost = 17600 →
  total_cost / (length * width) = 800 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l276_27603


namespace NUMINAMATH_CALUDE_smallest_number_l276_27672

theorem smallest_number (a b c d : ℝ) 
  (ha : a = -Real.sqrt 2) 
  (hb : b = 0) 
  (hc : c = 3.14) 
  (hd : d = 2021) : 
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l276_27672


namespace NUMINAMATH_CALUDE_james_sales_theorem_l276_27601

theorem james_sales_theorem (houses_day1 : ℕ) (houses_day2 : ℕ) (sale_rate_day2 : ℚ) (items_per_house : ℕ) :
  houses_day1 = 20 →
  houses_day2 = 2 * houses_day1 →
  sale_rate_day2 = 4/5 →
  items_per_house = 2 →
  houses_day1 * items_per_house + (houses_day2 : ℚ) * sale_rate_day2 * items_per_house = 104 :=
by sorry

end NUMINAMATH_CALUDE_james_sales_theorem_l276_27601


namespace NUMINAMATH_CALUDE_quadratic_properties_l276_27623

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 + 6

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  -- The parabola opens upwards
  (∀ x y : ℝ, x < y → f ((x + y) / 2) < (f x + f y) / 2) ∧
  -- When x < 2, f(x) decreases as x increases
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) ∧
  -- The axis of symmetry is x = 2
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧
  -- f(0) = 10
  f 0 = 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l276_27623


namespace NUMINAMATH_CALUDE_equation_solution_l276_27611

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((2 * x - 5) / (x - 2) = (3 * x - 3) / (x - 2) - 3) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l276_27611


namespace NUMINAMATH_CALUDE_donny_gas_change_l276_27678

/-- Calculates the change Donny will receive after filling up his gas tank. -/
theorem donny_gas_change (tank_capacity : ℝ) (current_fuel : ℝ) (cost_per_liter : ℝ) (amount_paid : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : current_fuel = 38)
  (h3 : cost_per_liter = 3)
  (h4 : amount_paid = 350) :
  amount_paid - (tank_capacity - current_fuel) * cost_per_liter = 14 := by
  sorry

end NUMINAMATH_CALUDE_donny_gas_change_l276_27678


namespace NUMINAMATH_CALUDE_sandy_age_l276_27687

theorem sandy_age :
  ∀ (S M J : ℕ),
    S = M - 14 →
    J = S + 6 →
    9 * S = 7 * M →
    6 * S = 5 * J →
    S = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_age_l276_27687


namespace NUMINAMATH_CALUDE_laptop_price_l276_27637

theorem laptop_price (sticker_price : ℝ) : 
  (0.8 * sticker_price - 120 = 0.7 * sticker_price - 50 - 30) →
  sticker_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l276_27637


namespace NUMINAMATH_CALUDE_find_a_l276_27642

-- Define the sets A and B
def A (a : ℤ) : Set ℤ := {1, 3, a}
def B (a : ℤ) : Set ℤ := {1, a^2}

-- State the theorem
theorem find_a : 
  ∀ a : ℤ, (A a ∪ B a = {1, 3, a}) → (a = 0 ∨ a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_find_a_l276_27642


namespace NUMINAMATH_CALUDE_bob_profit_is_1600_l276_27650

/-- Calculates the total profit from selling puppies given the number of show dogs bought,
    cost per show dog, number of puppies, and selling price per puppy. -/
def calculate_profit (num_dogs : ℕ) (cost_per_dog : ℚ) (num_puppies : ℕ) (price_per_puppy : ℚ) : ℚ :=
  num_puppies * price_per_puppy - num_dogs * cost_per_dog

/-- Proves that Bob's total profit from selling puppies is $1,600.00 -/
theorem bob_profit_is_1600 :
  calculate_profit 2 250 6 350 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_bob_profit_is_1600_l276_27650


namespace NUMINAMATH_CALUDE_craig_final_apples_l276_27648

def craig_initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem craig_final_apples :
  craig_initial_apples - shared_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_craig_final_apples_l276_27648


namespace NUMINAMATH_CALUDE_max_x_minus_y_l276_27668

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a - b ∧ a^2 + b^2 - 4*a - 2*b - 4 = 0) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l276_27668


namespace NUMINAMATH_CALUDE_simplify_fraction_l276_27632

/-- Given x = 3 and y = 4, prove that (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 -/
theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l276_27632


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l276_27657

-- Define proposition p
def p (x y : ℝ) : Prop := x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)

-- Define proposition q
def q (m : ℝ) : Prop := m > -2 → ∃ x : ℝ, x^2 + 2*x - m = 0

-- Theorem statement
theorem p_or_q_is_true :
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) →
  (∃ m : ℝ, m > -2 ∧ ¬(∃ x : ℝ, x^2 + 2*x - m = 0)) →
  ∀ x y m : ℝ, p x y ∨ q m :=
sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l276_27657


namespace NUMINAMATH_CALUDE_intersection_points_count_l276_27680

/-- Represents a line with a specific number of points -/
structure Line where
  numPoints : ℕ

/-- Represents a configuration of two parallel lines -/
structure ParallelLines where
  line1 : Line
  line2 : Line

/-- Calculates the number of intersection points for a given configuration of parallel lines -/
def intersectionPoints (pl : ParallelLines) : ℕ :=
  (pl.line1.numPoints.choose 2) * (pl.line2.numPoints.choose 2)

/-- The specific configuration of parallel lines in our problem -/
def problemConfig : ParallelLines :=
  { line1 := { numPoints := 10 }
    line2 := { numPoints := 11 } }

theorem intersection_points_count :
  intersectionPoints problemConfig = 2475 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l276_27680


namespace NUMINAMATH_CALUDE_angelas_height_l276_27628

/-- Given the heights of Amy, Helen, and Angela, prove Angela's height. -/
theorem angelas_height (amy_height helen_height angela_height : ℕ) 
  (helen_taller : helen_height = amy_height + 3)
  (angela_taller : angela_height = helen_height + 4)
  (amy_is_150 : amy_height = 150) :
  angela_height = 157 := by
  sorry

end NUMINAMATH_CALUDE_angelas_height_l276_27628


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l276_27630

theorem player_a_not_losing_probability 
  (prob_draw : ℝ) 
  (prob_a_win : ℝ) 
  (h1 : prob_draw = 0.4) 
  (h2 : prob_a_win = 0.4) : 
  prob_draw + prob_a_win = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l276_27630


namespace NUMINAMATH_CALUDE_optimal_distance_optimal_distance_with_discount_l276_27669

/-- Represents the optimal store distance problem --/
structure OptimalStoreDistance where
  s₀ : ℝ  -- Distance from home to city center
  v : ℝ   -- Base utility value

/-- Calculates the price at a given distance --/
def price (s_m : ℝ) : ℝ :=
  1000 * (1 - 0.02 * s_m)

/-- Calculates the transportation cost --/
def transportCost (s₀ s_m : ℝ) : ℝ :=
  0.5 * (s_m - s₀)^2

/-- Calculates the utility without discount --/
def utility (osd : OptimalStoreDistance) (s_m : ℝ) : ℝ :=
  osd.v - price s_m - transportCost osd.s₀ s_m

/-- Calculates the utility with discount --/
def utilityWithDiscount (osd : OptimalStoreDistance) (s_m : ℝ) : ℝ :=
  osd.v - 0.9 * price s_m - transportCost osd.s₀ s_m

/-- Theorem: Optimal store distance without discount --/
theorem optimal_distance (osd : OptimalStoreDistance) :
  ∃ s_m : ℝ, s_m = min 60 (osd.s₀ + 20) ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 60 → utility osd s_m ≥ utility osd s :=
sorry

/-- Theorem: Optimal store distance with discount --/
theorem optimal_distance_with_discount (osd : OptimalStoreDistance) :
  ∃ s_m : ℝ, s_m = min 60 (osd.s₀ + 9) ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 60 → utilityWithDiscount osd s_m ≥ utilityWithDiscount osd s :=
sorry

end NUMINAMATH_CALUDE_optimal_distance_optimal_distance_with_discount_l276_27669


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l276_27664

/-- The axis of symmetry of a shifted sine function -/
theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.sin (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 + π / 6
  ∀ x : ℝ, f (axis + x) = f (axis - x) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l276_27664


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l276_27612

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y + 100 = 0

/-- The center of a hyperbola -/
def hyperbola_center (h k : ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    hyperbola_equation x y ↔ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (3, 4) -/
theorem hyperbola_center_is_3_4 : hyperbola_center 3 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l276_27612


namespace NUMINAMATH_CALUDE_set_relations_l276_27606

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem set_relations (a : ℝ) :
  (B a ⊆ A ↔ a ∈ Set.Ici 1) ∧
  (Set.Nonempty (A ∩ B a) ↔ a ∈ Set.Ioi 0) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l276_27606


namespace NUMINAMATH_CALUDE_jerry_lawsuit_years_l276_27615

def salary_per_year : ℕ := 50000
def medical_bills : ℕ := 200000
def punitive_multiplier : ℕ := 3
def settlement_percentage : ℚ := 4/5
def total_received : ℕ := 5440000

def total_damages (Y : ℕ) : ℕ :=
  Y * salary_per_year + medical_bills + punitive_multiplier * (Y * salary_per_year + medical_bills)

theorem jerry_lawsuit_years :
  ∃ Y : ℕ, (↑total_received : ℚ) = settlement_percentage * (↑(total_damages Y) : ℚ) ∧ Y = 30 :=
by sorry

end NUMINAMATH_CALUDE_jerry_lawsuit_years_l276_27615


namespace NUMINAMATH_CALUDE_tin_content_in_new_alloy_l276_27638

theorem tin_content_in_new_alloy 
  (tin_percent_first : Real) 
  (copper_percent_second : Real)
  (zinc_percent_new : Real)
  (weight_first : Real)
  (weight_second : Real)
  (h1 : tin_percent_first = 40)
  (h2 : copper_percent_second = 26)
  (h3 : zinc_percent_new = 30)
  (h4 : weight_first = 150)
  (h5 : weight_second = 250)
  : Real :=
by
  sorry

#check tin_content_in_new_alloy

end NUMINAMATH_CALUDE_tin_content_in_new_alloy_l276_27638


namespace NUMINAMATH_CALUDE_electricity_fee_properties_l276_27634

-- Define the relationship between electricity usage and fee
def electricity_fee (x : ℝ) : ℝ := 0.55 * x

-- Theorem stating the properties of the electricity fee function
theorem electricity_fee_properties :
  -- 1. x is independent, y is dependent (implicit in the function definition)
  -- 2. For every increase of 1 in x, y increases by 0.55
  (∀ x : ℝ, electricity_fee (x + 1) = electricity_fee x + 0.55) ∧
  -- 3. When x = 8, y = 4.4
  (electricity_fee 8 = 4.4) ∧
  -- 4. When y = 3.75, x ≠ 7
  (∀ x : ℝ, electricity_fee x = 3.75 → x ≠ 7) := by
  sorry


end NUMINAMATH_CALUDE_electricity_fee_properties_l276_27634


namespace NUMINAMATH_CALUDE_product_of_areas_is_perfect_square_l276_27677

/-- A convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- The areas of the four triangles formed by the diagonals -/
  area₁ : ℤ
  area₂ : ℤ
  area₃ : ℤ
  area₄ : ℤ

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem product_of_areas_is_perfect_square (q : ConvexQuadrilateral) :
  ∃ (n : ℤ), q.area₁ * q.area₂ * q.area₃ * q.area₄ = n ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_product_of_areas_is_perfect_square_l276_27677


namespace NUMINAMATH_CALUDE_peanut_butter_duration_l276_27698

/-- The number of days that peanut butter lasts for Phoebe and her dog -/
def peanut_butter_days (servings_per_jar : ℕ) (num_jars : ℕ) (daily_consumption : ℕ) : ℕ :=
  (servings_per_jar * num_jars) / daily_consumption

/-- Theorem stating how long 4 jars of peanut butter will last for Phoebe and her dog -/
theorem peanut_butter_duration :
  let servings_per_jar : ℕ := 15
  let num_jars : ℕ := 4
  let phoebe_consumption : ℕ := 1
  let dog_consumption : ℕ := 1
  let daily_consumption : ℕ := phoebe_consumption + dog_consumption
  peanut_butter_days servings_per_jar num_jars daily_consumption = 30 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_duration_l276_27698


namespace NUMINAMATH_CALUDE_performances_distribution_l276_27685

/-- The number of ways to distribute performances among classes -/
def distribute_performances (total : ℕ) (num_classes : ℕ) (min_per_class : ℕ) : ℕ :=
  Nat.choose (total - num_classes * min_per_class + num_classes - 1) (num_classes - 1)

/-- Theorem stating the number of ways to distribute 14 performances among 3 classes -/
theorem performances_distribution :
  distribute_performances 14 3 3 = 21 := by sorry

end NUMINAMATH_CALUDE_performances_distribution_l276_27685


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l276_27613

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : r - p = 28) :
  (q + r) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l276_27613


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l276_27684

theorem complex_number_in_first_quadrant (z : ℂ) : z = 2 + I → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l276_27684


namespace NUMINAMATH_CALUDE_stratified_sample_problem_l276_27652

/-- Given a stratified sample from a high school where:
  * The sample size is 55 students
  * 10 students are from the first year
  * 25 students are from the second year
  * There are 400 students in the third year
Prove that the total number of students in the first and second years combined is 700. -/
theorem stratified_sample_problem (sample_size : ℕ) (first_year_sample : ℕ) (second_year_sample : ℕ) (third_year_total : ℕ) :
  sample_size = 55 →
  first_year_sample = 10 →
  second_year_sample = 25 →
  third_year_total = 400 →
  ∃ (first_and_second_total : ℕ),
    first_and_second_total = 700 ∧
    (first_year_sample + second_year_sample : ℚ) / sample_size = first_and_second_total / (first_and_second_total + third_year_total) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_problem_l276_27652


namespace NUMINAMATH_CALUDE_probability_more_than_third_correct_l276_27614

-- Define the number of questions
def n : ℕ := 12

-- Define the probability of guessing correctly
def p : ℚ := 1/2

-- Define the minimum number of correct answers needed
def k : ℕ := 5

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of getting at least k correct answers
def prob_at_least_k (n k : ℕ) (p : ℚ) : ℚ := sorry

-- State the theorem
theorem probability_more_than_third_correct :
  prob_at_least_k n k p = 825/1024 := by sorry

end NUMINAMATH_CALUDE_probability_more_than_third_correct_l276_27614


namespace NUMINAMATH_CALUDE_max_abs_diff_f_g_l276_27693

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the absolute difference function
def absDiff (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem max_abs_diff_f_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  (∀ x, x ∈ Set.Icc 0 1 → absDiff x ≤ absDiff c) ∧
  absDiff c = 4/27 :=
sorry

end NUMINAMATH_CALUDE_max_abs_diff_f_g_l276_27693


namespace NUMINAMATH_CALUDE_inequality_solution_set_l276_27673

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x - 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l276_27673


namespace NUMINAMATH_CALUDE_paperboy_delivery_count_l276_27688

def delivery_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | m + 3 => delivery_ways (m + 2) + delivery_ways (m + 1) + delivery_ways m

theorem paperboy_delivery_count :
  delivery_ways 12 = 504 :=
by sorry

end NUMINAMATH_CALUDE_paperboy_delivery_count_l276_27688


namespace NUMINAMATH_CALUDE_power_multiplication_l276_27658

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l276_27658


namespace NUMINAMATH_CALUDE_sequence_problem_l276_27699

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d ∧ d ≠ 0

-- Define a geometric sequence
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, b (n + 1) = r * b n

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  (2 * a 3 - (a 7)^2 + 2 * a 11 = 0) →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l276_27699


namespace NUMINAMATH_CALUDE_dana_total_earnings_l276_27691

/-- Dana's hourly wage in dollars -/
def hourly_wage : ℕ := 13

/-- Hours worked on Friday -/
def friday_hours : ℕ := 9

/-- Hours worked on Saturday -/
def saturday_hours : ℕ := 10

/-- Hours worked on Sunday -/
def sunday_hours : ℕ := 3

/-- Calculate total earnings given hourly wage and hours worked -/
def total_earnings (wage : ℕ) (hours_fri hours_sat hours_sun : ℕ) : ℕ :=
  wage * (hours_fri + hours_sat + hours_sun)

/-- Theorem stating Dana's total earnings -/
theorem dana_total_earnings :
  total_earnings hourly_wage friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end NUMINAMATH_CALUDE_dana_total_earnings_l276_27691


namespace NUMINAMATH_CALUDE_fraction_addition_l276_27621

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l276_27621


namespace NUMINAMATH_CALUDE_pam_has_ten_bags_l276_27604

/-- The number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- The number of Gerald's bags equivalent to one of Pam's bags -/
def bags_ratio : ℕ := 3

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- The number of bags Pam has -/
def pams_bag_count : ℕ := pams_total_apples / (bags_ratio * geralds_bag_count)

theorem pam_has_ten_bags : pams_bag_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_ten_bags_l276_27604


namespace NUMINAMATH_CALUDE_league_teams_l276_27605

theorem league_teams (n : ℕ) : n * (n - 1) / 2 = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_l276_27605


namespace NUMINAMATH_CALUDE_original_triangle_area_l276_27608

/-- Given a triangle with area A, if its dimensions are quadrupled to form a new triangle
    with an area of 144 square feet, then the area A of the original triangle is 9 square feet. -/
theorem original_triangle_area (A : ℝ) : 
  (∃ (new_triangle : ℝ), new_triangle = 144 ∧ new_triangle = 16 * A) → A = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l276_27608


namespace NUMINAMATH_CALUDE_extreme_values_range_l276_27629

/-- Given a function f(x) = 2x³ - (1/2)ax² + ax + 1, where a is a real number,
    this theorem states that the range of values for a such that f(x) has two
    extreme values in the interval (0, +∞) is (0, +∞). -/
theorem extreme_values_range (a : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧
    (∀ z : ℝ, 0 < z → (6 * z^2 - a * z + a = 0) ↔ (z = x ∨ z = y))) ↔
  (0 < a) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_range_l276_27629


namespace NUMINAMATH_CALUDE_PA_square_property_l276_27679

theorem PA_square_property : 
  {PA : ℕ | 
    10 ≤ PA ∧ PA < 100 ∧ 
    1000 ≤ PA^2 ∧ PA^2 < 10000 ∧
    (PA^2 / 1000 = PA / 10) ∧ 
    (PA^2 % 10 = PA % 10)} = {95, 96} := by
  sorry

end NUMINAMATH_CALUDE_PA_square_property_l276_27679


namespace NUMINAMATH_CALUDE_carly_dogs_worked_on_l276_27671

/-- The number of dogs Carly worked on given the number of nails trimmed,
    nails per paw, and number of three-legged dogs. -/
def dogs_worked_on (total_nails : ℕ) (nails_per_paw : ℕ) (three_legged_dogs : ℕ) : ℕ :=
  let total_paws := total_nails / nails_per_paw
  let three_legged_paws := three_legged_dogs * 3
  let four_legged_paws := total_paws - three_legged_paws
  let four_legged_dogs := four_legged_paws / 4
  four_legged_dogs + three_legged_dogs

theorem carly_dogs_worked_on :
  dogs_worked_on 164 4 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_carly_dogs_worked_on_l276_27671


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l276_27683

theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 - 4 * X^2 + 3 * X + 2
  let p₂ : Polynomial ℤ := -X^2 + 3 * X - 5
  (p₁ * p₂).coeff 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_product_l276_27683


namespace NUMINAMATH_CALUDE_sum_of_digits_product_35_42_base8_l276_27618

def base8_to_base10 (n : Nat) : Nat :=
  (n / 10) * 8 + n % 10

def base10_to_base8 (n : Nat) : Nat :=
  if n < 8 then n
  else (base10_to_base8 (n / 8)) * 10 + n % 8

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n
  else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_product_35_42_base8 :
  sum_of_digits (base10_to_base8 (base8_to_base10 35 * base8_to_base10 42)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_35_42_base8_l276_27618


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l276_27616

/-- The charge difference between two print shops for a given number of copies. -/
def chargeDifference (priceX priceY : ℚ) (copies : ℕ) : ℚ :=
  copies * (priceY - priceX)

/-- The theorem stating the charge difference for 70 color copies between shop Y and shop X. -/
theorem print_shop_charge_difference :
  chargeDifference (1.20 : ℚ) (1.70 : ℚ) 70 = 35 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l276_27616


namespace NUMINAMATH_CALUDE_circle_radius_is_zero_l276_27659

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * x + 2 * y^2 + 4 * y + 10 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 0

/-- Theorem: The radius of the circle defined by the equation is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y → ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_zero_l276_27659


namespace NUMINAMATH_CALUDE_max_value_of_expression_l276_27633

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → x + y^2 + z^4 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l276_27633


namespace NUMINAMATH_CALUDE_min_value_trig_sum_l276_27640

theorem min_value_trig_sum (θ : ℝ) : 
  1 / (2 - Real.cos θ ^ 2) + 1 / (2 - Real.sin θ ^ 2) ≥ 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_sum_l276_27640


namespace NUMINAMATH_CALUDE_managers_salary_l276_27674

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 1000 →
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + salary_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary = 22500 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l276_27674


namespace NUMINAMATH_CALUDE_expression_factorization_l276_27647

theorem expression_factorization (x : ℝ) :
  (3 * x^3 - 67 * x^2 - 14) - (-8 * x^3 + 3 * x^2 - 14) = x^2 * (11 * x - 70) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l276_27647


namespace NUMINAMATH_CALUDE_existence_of_solution_l276_27649

theorem existence_of_solution : ∃ (a b c d : ℕ+), 
  (a^3 + b^4 + c^5 = d^11) ∧ (a * b * c < 10^5) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l276_27649


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l276_27694

/-- The surface area of a sphere circumscribing a cube with side length 4 is 48π. -/
theorem circumscribed_sphere_surface_area (cube_side : ℝ) (h : cube_side = 4) :
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius^2 = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l276_27694


namespace NUMINAMATH_CALUDE_smallest_divisor_after_437_l276_27697

theorem smallest_divisor_after_437 (m : ℕ) (h1 : 10000 ≤ m ∧ m ≤ 99999) 
  (h2 : Odd m) (h3 : m % 437 = 0) : 
  (∃ (d : ℕ), d > 437 ∧ d ∣ m ∧ ∀ (x : ℕ), 437 < x ∧ x < d → ¬(x ∣ m)) → 
  (Nat.minFac (m / 437) = 19 ∨ Nat.minFac (m / 437) = 23) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_437_l276_27697


namespace NUMINAMATH_CALUDE_t_divides_t_2n_plus_1_l276_27645

def t : ℕ → ℕ
  | 0 => 1  -- Assuming t_0 = 1 for completeness
  | 1 => 2
  | 2 => 5
  | (n + 3) => 2 * t (n + 2) + t (n + 1)

theorem t_divides_t_2n_plus_1 (n : ℕ) : t n ∣ t (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_t_divides_t_2n_plus_1_l276_27645


namespace NUMINAMATH_CALUDE_farmer_truck_count_l276_27631

/-- Proves the number of trucks a farmer has, given tank capacity, tanks per truck, and total water capacity. -/
theorem farmer_truck_count (tank_capacity : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) 
  (h1 : tank_capacity = 150)
  (h2 : tanks_per_truck = 3)
  (h3 : total_capacity = 1350) :
  total_capacity / (tank_capacity * tanks_per_truck) = 3 :=
by sorry

end NUMINAMATH_CALUDE_farmer_truck_count_l276_27631


namespace NUMINAMATH_CALUDE_students_wanting_fruit_l276_27692

theorem students_wanting_fruit (red_apples green_apples extra_apples : ℕ) :
  let total_apples := red_apples + green_apples
  let students := total_apples - extra_apples
  students = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_students_wanting_fruit_l276_27692


namespace NUMINAMATH_CALUDE_max_value_sine_function_l276_27682

theorem max_value_sine_function (x : Real) (h : -π/2 ≤ x ∧ x ≤ 0) :
  ∃ y_max : Real, y_max = 2 ∧ ∀ y : Real, y = 3 * Real.sin x + 2 → y ≤ y_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_function_l276_27682


namespace NUMINAMATH_CALUDE_opposite_vertices_distance_l276_27625

def cube_side_length : ℝ := 2

theorem opposite_vertices_distance (cube_side : ℝ) (h : cube_side = cube_side_length) :
  let diagonal := Real.sqrt (3 * cube_side ^ 2)
  diagonal = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_opposite_vertices_distance_l276_27625


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l276_27654

/-- The polynomial function f(x) = x^11 + 5x^10 + 20x^9 + 1000x^8 - 800x^7 -/
def f (x : ℝ) : ℝ := x^11 + 5*x^10 + 20*x^9 + 1000*x^8 - 800*x^7

/-- Theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem one_positive_real_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l276_27654


namespace NUMINAMATH_CALUDE_exists_k_for_all_n_l276_27667

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Statement of the problem -/
theorem exists_k_for_all_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k > 0 ∧ sumOfDigits k = n ∧ sumOfDigits (k^2) = n^2 := by sorry

end NUMINAMATH_CALUDE_exists_k_for_all_n_l276_27667


namespace NUMINAMATH_CALUDE_log_base_range_l276_27609

-- Define the function f(x) = log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_base_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 < f a 3 → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_log_base_range_l276_27609


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l276_27646

-- Define the three lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x - y + 8 = 0
def line3 (a x y : ℝ) : Prop := a*x + 3*y - 5 = 0

-- Define the condition of at most 2 intersection points
def at_most_two_intersections (a : ℝ) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (line1 x₁ y₁ ∧ line2 x₁ y₁ ∧ line3 a x₁ y₁) →
    (line1 x₂ y₂ ∧ line2 x₂ y₂ ∧ line3 a x₂ y₂) →
    (line1 x₃ y₃ ∧ line2 x₃ y₃ ∧ line3 a x₃ y₃) →
    ((x₁ = x₂ ∧ y₁ = y₂) ∨ (x₁ = x₃ ∧ y₁ = y₃) ∨ (x₂ = x₃ ∧ y₂ = y₃))

-- The theorem statement
theorem intersection_points_theorem :
  ∀ a : ℝ, at_most_two_intersections a ↔ (a = -3 ∨ a = -6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l276_27646


namespace NUMINAMATH_CALUDE_right_triangle_with_specific_altitude_and_segment_difference_l276_27635

/-- Represents a right-angled triangle with an altitude to the hypotenuse -/
structure RightTriangleWithAltitude where
  /-- First leg of the triangle -/
  leg1 : ℝ
  /-- Second leg of the triangle -/
  leg2 : ℝ
  /-- Hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Altitude drawn to the hypotenuse -/
  altitude : ℝ
  /-- First segment of the hypotenuse -/
  segment1 : ℝ
  /-- Second segment of the hypotenuse -/
  segment2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : leg1^2 + leg2^2 = hypotenuse^2
  /-- The altitude divides the hypotenuse into two segments -/
  hypotenuse_segments : segment1 + segment2 = hypotenuse
  /-- The altitude creates similar triangles -/
  similar_triangles : altitude^2 = segment1 * segment2

/-- Theorem: Given a right-angled triangle with specific altitude and hypotenuse segment difference, prove its sides -/
theorem right_triangle_with_specific_altitude_and_segment_difference
  (t : RightTriangleWithAltitude)
  (h_altitude : t.altitude = 12)
  (h_segment_diff : t.segment1 - t.segment2 = 7) :
  t.leg1 = 15 ∧ t.leg2 = 20 ∧ t.hypotenuse = 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_specific_altitude_and_segment_difference_l276_27635


namespace NUMINAMATH_CALUDE_pet_snake_cost_l276_27696

def initial_amount : ℕ := 73
def amount_left : ℕ := 18

theorem pet_snake_cost : initial_amount - amount_left = 55 := by sorry

end NUMINAMATH_CALUDE_pet_snake_cost_l276_27696


namespace NUMINAMATH_CALUDE_marias_trip_l276_27627

theorem marias_trip (total_distance : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 360)
  (h2 : remaining_distance = 135)
  (h3 : remaining_distance = total_distance - (x * total_distance + 1/4 * (total_distance - x * total_distance)))
  : x = 1/2 :=
by
  sorry

#check marias_trip

end NUMINAMATH_CALUDE_marias_trip_l276_27627


namespace NUMINAMATH_CALUDE_firework_explosion_velocity_firework_explosion_velocity_is_correct_l276_27681

/-- The magnitude of the second fragment's velocity after a firework explosion -/
theorem firework_explosion_velocity : ℝ :=
  let initial_velocity : ℝ := 20
  let gravity : ℝ := 10
  let explosion_time : ℝ := 1
  let mass_ratio : ℝ := 2
  let small_fragment_horizontal_velocity : ℝ := 16

  let velocity_at_explosion : ℝ := initial_velocity - gravity * explosion_time
  let small_fragment_mass : ℝ := 1
  let large_fragment_mass : ℝ := mass_ratio * small_fragment_mass

  let small_fragment_vertical_velocity : ℝ := velocity_at_explosion
  let large_fragment_horizontal_velocity : ℝ := 
    -(small_fragment_mass * small_fragment_horizontal_velocity) / large_fragment_mass
  let large_fragment_vertical_velocity : ℝ := velocity_at_explosion

  let large_fragment_velocity_magnitude : ℝ := 
    Real.sqrt (large_fragment_horizontal_velocity^2 + large_fragment_vertical_velocity^2)

  2 * Real.sqrt 41

theorem firework_explosion_velocity_is_correct : 
  firework_explosion_velocity = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_firework_explosion_velocity_firework_explosion_velocity_is_correct_l276_27681


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equality_l276_27690

theorem cosine_sine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equality_l276_27690


namespace NUMINAMATH_CALUDE_major_premise_wrong_l276_27660

theorem major_premise_wrong : ¬ ∀ a b : ℝ, a > b → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_major_premise_wrong_l276_27660


namespace NUMINAMATH_CALUDE_friends_in_group_l276_27619

def number_of_friends (initial_wings : ℕ) (additional_wings : ℕ) (wings_per_person : ℕ) : ℕ :=
  (initial_wings + additional_wings) / wings_per_person

theorem friends_in_group :
  number_of_friends 8 10 6 = 3 :=
by sorry

end NUMINAMATH_CALUDE_friends_in_group_l276_27619


namespace NUMINAMATH_CALUDE_cat_adoptions_correct_l276_27639

/-- The number of families who adopted cats at an animal shelter event -/
def num_cat_adoptions : ℕ := 3

/-- Vet fees for dogs in dollars -/
def dog_fee : ℕ := 15

/-- Vet fees for cats in dollars -/
def cat_fee : ℕ := 13

/-- Number of families who adopted dogs -/
def num_dog_adoptions : ℕ := 8

/-- The fraction of fees donated back to the shelter -/
def donation_fraction : ℚ := 1/3

/-- The amount donated back to the shelter in dollars -/
def donation_amount : ℕ := 53

theorem cat_adoptions_correct : 
  (num_dog_adoptions * dog_fee + num_cat_adoptions * cat_fee) * donation_fraction = donation_amount :=
sorry

end NUMINAMATH_CALUDE_cat_adoptions_correct_l276_27639


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l276_27655

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (k₁ k₂ b₁ b₂ : ℝ) :
  (∀ x y : ℝ, y = k₁ * x + b₁ → y = k₂ * x + b₂ → (k₁ * k₂ = -1 ↔ ∀ x₁ y₁ x₂ y₂ : ℝ,
    (y₁ = k₁ * x₁ + b₁ ∧ y₂ = k₁ * x₂ + b₁) →
    (y₁ = k₂ * x₁ + b₂ ∧ y₂ = k₂ * x₂ + b₂) →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (y₂ - y₁) = 0))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l276_27655


namespace NUMINAMATH_CALUDE_no_divisibility_by_4_or_8_l276_27620

/-- The set of all numbers which are the sum of the squares of three consecutive odd integers -/
def T : Set ℤ :=
  {x | ∃ n : ℤ, Odd n ∧ x = 3 * n^2 + 8}

/-- Theorem stating that no member of T is divisible by 4 or 8 -/
theorem no_divisibility_by_4_or_8 (x : ℤ) (hx : x ∈ T) :
  ¬(4 ∣ x) ∧ ¬(8 ∣ x) := by
  sorry

#check no_divisibility_by_4_or_8

end NUMINAMATH_CALUDE_no_divisibility_by_4_or_8_l276_27620


namespace NUMINAMATH_CALUDE_sweets_distribution_l276_27626

/-- The number of children initially supposed to receive sweets -/
def initial_children : ℕ := 190

/-- The number of absent children -/
def absent_children : ℕ := 70

/-- The number of extra sweets each child received due to absences -/
def extra_sweets : ℕ := 14

/-- The total number of sweets each child received -/
def total_sweets : ℕ := 38

theorem sweets_distribution :
  initial_children * (total_sweets - extra_sweets) = 
  (initial_children - absent_children) * total_sweets :=
sorry

end NUMINAMATH_CALUDE_sweets_distribution_l276_27626


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l276_27641

theorem geometric_mean_minimum (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h_geom_mean : z^2 = x*y) : 
  (Real.log z) / (4 * Real.log x) + (Real.log z) / (Real.log y) ≥ 9/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l276_27641


namespace NUMINAMATH_CALUDE_square_rectangle_intersection_l276_27617

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def Square (q : Quadrilateral) : Prop :=
  ∃ s : ℝ, s > 0 ∧
    q.B.1 - q.A.1 = s ∧ q.B.2 - q.A.2 = 0 ∧
    q.C.1 - q.B.1 = 0 ∧ q.C.2 - q.B.2 = s ∧
    q.D.1 - q.C.1 = -s ∧ q.D.2 - q.C.2 = 0 ∧
    q.A.1 - q.D.1 = 0 ∧ q.A.2 - q.D.2 = -s

def Rectangle (q : Quadrilateral) : Prop :=
  ∃ w h : ℝ, w > 0 ∧ h > 0 ∧
    q.B.1 - q.A.1 = w ∧ q.B.2 - q.A.2 = 0 ∧
    q.C.1 - q.B.1 = 0 ∧ q.C.2 - q.B.2 = h ∧
    q.D.1 - q.C.1 = -w ∧ q.D.2 - q.C.2 = 0 ∧
    q.A.1 - q.D.1 = 0 ∧ q.A.2 - q.D.2 = -h

def Perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem square_rectangle_intersection (EFGH IJKL : Quadrilateral) (E Q : ℝ × ℝ) :
  Square EFGH ∧ 
  Rectangle IJKL ∧
  EFGH.A = E ∧
  IJKL.B.1 - IJKL.A.1 = 12 ∧
  IJKL.C.2 - IJKL.B.2 = 8 ∧
  Perpendicular (EFGH.D.1 - EFGH.A.1, EFGH.D.2 - EFGH.A.2) (IJKL.B.1 - IJKL.A.1, IJKL.B.2 - IJKL.A.2) ∧
  (Q.2 - EFGH.D.2) * (IJKL.B.1 - IJKL.A.1) = 1/3 * (IJKL.B.1 - IJKL.A.1) * (IJKL.C.2 - IJKL.B.2) →
  E.2 - Q.2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_intersection_l276_27617


namespace NUMINAMATH_CALUDE_fast_pulsar_period_scientific_notation_l276_27602

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fast_pulsar_period_scientific_notation :
  toScientificNotation 0.00519 = ScientificNotation.mk 5.19 (-3) sorry := by
  sorry

end NUMINAMATH_CALUDE_fast_pulsar_period_scientific_notation_l276_27602


namespace NUMINAMATH_CALUDE_green_sequin_rows_jane_green_sequin_rows_l276_27695

/-- Calculates the number of rows of green sequins in Jane's costume. -/
theorem green_sequin_rows (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_rows : Nat) (purple_per_row : Nat) (green_per_row : Nat) 
  (total_sequins : Nat) : Nat :=
  let blue_sequins := blue_rows * blue_per_row
  let purple_sequins := purple_rows * purple_per_row
  let blue_and_purple := blue_sequins + purple_sequins
  let green_sequins := total_sequins - blue_and_purple
  let green_rows := green_sequins / green_per_row
  green_rows

/-- Proves that Jane sews 9 rows of green sequins. -/
theorem jane_green_sequin_rows : 
  green_sequin_rows 6 8 5 12 6 162 = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_sequin_rows_jane_green_sequin_rows_l276_27695


namespace NUMINAMATH_CALUDE_max_m_value_l276_27644

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m / (3 * a + b) - 3 / a - 1 / b ≤ 0) →
  (∃ m : ℝ, m = 16 ∧ ∀ n : ℝ, (n / (3 * a + b) - 3 / a - 1 / b ≤ 0) → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l276_27644


namespace NUMINAMATH_CALUDE_log_equation_l276_27656

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l276_27656


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l276_27670

/-- Given a quadratic function f(x) = 5x^2 + 20x + 45, 
    prove that the y-coordinate of its vertex is 25. -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ 5 * x^2 + 20 * x + 45
  ∃ h k : ℝ, (∀ x, f x = 5 * (x - h)^2 + k) ∧ k = 25 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l276_27670


namespace NUMINAMATH_CALUDE_large_square_area_l276_27651

/-- The area of a square formed by four congruent rectangles and a smaller square -/
theorem large_square_area (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 20) : 
  (x + y)^2 = 400 := by
  sorry

#check large_square_area

end NUMINAMATH_CALUDE_large_square_area_l276_27651


namespace NUMINAMATH_CALUDE_error_percentage_l276_27676

theorem error_percentage (x : ℝ) (h : x > 0) : 
  (|4*x - x/4|) / (4*x) = 15/16 := by
sorry

end NUMINAMATH_CALUDE_error_percentage_l276_27676


namespace NUMINAMATH_CALUDE_initial_men_count_is_eight_l276_27665

/-- The number of men in the initial group -/
def initial_men_count : ℕ := sorry

/-- The increase in average age when two men are replaced -/
def average_age_increase : ℕ := 2

/-- The age of the first man being replaced -/
def first_replaced_man_age : ℕ := 21

/-- The age of the second man being replaced -/
def second_replaced_man_age : ℕ := 23

/-- The average age of the two new men -/
def new_men_average_age : ℕ := 30

theorem initial_men_count_is_eight :
  initial_men_count = 8 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_is_eight_l276_27665


namespace NUMINAMATH_CALUDE_four_digit_sum_l276_27624

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit numbers that are multiples of 7 -/
def D : ℕ := 1285

/-- Theorem stating that the sum of four-digit even numbers and four-digit multiples of 7 is 5785 -/
theorem four_digit_sum : C + D = 5785 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_l276_27624


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l276_27600

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 4) (hb : b = 5) :
  ∃ (c : ℝ), c > 0 ∧ a^2 + c^2 = b^2 ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  ((x = a ∧ y = b) ∨ (x = b ∧ y = a) ∨ (y = a ∧ z = b) ∨ (y = b ∧ z = a)) →
  x^2 + y^2 = z^2 →
  (1/2) * x * y ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l276_27600


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l276_27666

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 + i) / i
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l276_27666


namespace NUMINAMATH_CALUDE_cow_inheritance_problem_l276_27675

theorem cow_inheritance_problem (x y : ℕ) (z : ℝ) 
  (h1 : x^2 = 10*y + z)
  (h2 : z < 10)
  (h3 : Odd y)
  (h4 : x^2 % 10 = 6) :
  (10 - z) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cow_inheritance_problem_l276_27675


namespace NUMINAMATH_CALUDE_smallest_integer_square_equation_l276_27663

theorem smallest_integer_square_equation : ∃ x : ℤ, 
  (∀ y : ℤ, y^2 = 3*y + 72 → x ≤ y) ∧ x^2 = 3*x + 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_square_equation_l276_27663


namespace NUMINAMATH_CALUDE_marble_probability_l276_27689

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  valid : black + white = total

/-- Represents two boxes of marbles -/
structure TwoBoxes where
  box1 : MarbleBox
  box2 : MarbleBox
  total36 : box1.total + box2.total = 36

/-- The probability of drawing a black marble from a box -/
def probBlack (box : MarbleBox) : ℚ :=
  box.black / box.total

/-- The probability of drawing a white marble from a box -/
def probWhite (box : MarbleBox) : ℚ :=
  box.white / box.total

theorem marble_probability (boxes : TwoBoxes)
    (h : probBlack boxes.box1 * probBlack boxes.box2 = 13/18) :
    probWhite boxes.box1 * probWhite boxes.box2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l276_27689


namespace NUMINAMATH_CALUDE_min_x_for_sqrt_2x_minus_1_l276_27662

theorem min_x_for_sqrt_2x_minus_1 :
  ∀ x : ℝ, (∃ y : ℝ, y^2 = 2*x - 1) → x ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_x_for_sqrt_2x_minus_1_l276_27662


namespace NUMINAMATH_CALUDE_constant_value_l276_27636

theorem constant_value (x : ℝ) (constant : ℝ) 
  (eq : 5 * x + 3 = 10 * x - constant) 
  (h : x = 5) : 
  constant = 22 := by
sorry

end NUMINAMATH_CALUDE_constant_value_l276_27636


namespace NUMINAMATH_CALUDE_range_of_odd_function_l276_27610

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_positive (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x = 3

theorem range_of_odd_function (f : ℝ → ℝ) (h1 : is_odd f) (h2 : f_positive f) :
  Set.range f = {-3, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_odd_function_l276_27610


namespace NUMINAMATH_CALUDE_shopping_expenditure_l276_27643

theorem shopping_expenditure (x : ℝ) 
  (h1 : x + 10 + 40 = 100) 
  (h2 : 0.04 * x + 0.08 * 40 = 5.2) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l276_27643


namespace NUMINAMATH_CALUDE_number_of_sets_l276_27686

/-- The number of flowers in each set -/
def flowers_per_set : ℕ := 90

/-- The total number of flowers bought -/
def total_flowers : ℕ := 270

/-- Theorem: The number of sets of flowers bought is 3 -/
theorem number_of_sets : total_flowers / flowers_per_set = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sets_l276_27686
