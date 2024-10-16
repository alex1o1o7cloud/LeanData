import Mathlib

namespace NUMINAMATH_CALUDE_largest_n_satisfying_equation_l476_47613

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 397 is the largest positive integer satisfying the equation -/
theorem largest_n_satisfying_equation :
  ∀ n : ℕ, n > 0 → n = (sum_of_digits n)^2 + 2*(sum_of_digits n) - 2 → n ≤ 397 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_equation_l476_47613


namespace NUMINAMATH_CALUDE_david_average_speed_l476_47652

/-- Calculates the average speed given distance and time -/
def average_speed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

/-- Converts hours and minutes to hours -/
def hours_and_minutes_to_hours (hours : ℕ) (minutes : ℕ) : ℚ :=
  hours + (minutes : ℚ) / 60

theorem david_average_speed :
  let distance : ℚ := 49/3  -- 16 1/3 miles as an improper fraction
  let time : ℚ := hours_and_minutes_to_hours 2 20
  average_speed distance time = 7 := by sorry

end NUMINAMATH_CALUDE_david_average_speed_l476_47652


namespace NUMINAMATH_CALUDE_percentage_difference_l476_47656

theorem percentage_difference (x y : ℝ) (h : y = x + 0.6 * x) :
  (y - x) / y = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l476_47656


namespace NUMINAMATH_CALUDE_alice_ball_drawing_l476_47697

/-- The number of balls in the bin -/
def n : ℕ := 20

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k balls from n balls with replacement -/
def num_possible_lists (n k : ℕ) : ℕ := n ^ k

theorem alice_ball_drawing :
  num_possible_lists n k = 160000 := by
  sorry

end NUMINAMATH_CALUDE_alice_ball_drawing_l476_47697


namespace NUMINAMATH_CALUDE_ratio_equality_l476_47638

theorem ratio_equality : (1722^2 - 1715^2) / (1729^2 - 1708^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l476_47638


namespace NUMINAMATH_CALUDE_simplify_expression_l476_47615

theorem simplify_expression (x : ℝ) : (3 * x + 20) - (7 * x - 5) = -4 * x + 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l476_47615


namespace NUMINAMATH_CALUDE_strictly_increasing_function_l476_47676

theorem strictly_increasing_function
  (a b c d : ℝ)
  (h1 : a > c)
  (h2 : c > d)
  (h3 : d > b)
  (h4 : b > 1)
  (h5 : a * b > c * d) :
  let f : ℝ → ℝ := λ x ↦ a^x + b^x - c^x - d^x
  ∀ x ≥ 0, (deriv f) x > 0 :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_l476_47676


namespace NUMINAMATH_CALUDE_inequality_solution_set_l476_47670

theorem inequality_solution_set (x : ℝ) : (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l476_47670


namespace NUMINAMATH_CALUDE_no_unique_solution_implies_a_equals_four_l476_47643

/-- Given two linear equations in two variables, this function determines if they have a unique solution. -/
def hasUniqueSolution (a k : ℝ) : Prop :=
  ∃! (x y : ℝ), a * (3 * x + 4 * y) = 36 ∧ k * x + 12 * y = 30

/-- The theorem states that when k = 9 and the equations don't have a unique solution, a must equal 4. -/
theorem no_unique_solution_implies_a_equals_four :
  ∀ (a : ℝ), (¬ hasUniqueSolution a 9) → a = 4 := by
  sorry

#check no_unique_solution_implies_a_equals_four

end NUMINAMATH_CALUDE_no_unique_solution_implies_a_equals_four_l476_47643


namespace NUMINAMATH_CALUDE_circular_pool_volume_l476_47632

/-- The volume of a circular pool with given dimensions -/
theorem circular_pool_volume (diameter : ℝ) (depth1 : ℝ) (depth2 : ℝ) :
  diameter = 20 →
  depth1 = 3 →
  depth2 = 5 →
  (π * (diameter / 2)^2 * depth1 + π * (diameter / 2)^2 * depth2) = 800 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_pool_volume_l476_47632


namespace NUMINAMATH_CALUDE_at_op_difference_l476_47680

-- Define the new operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x - 4 * y

-- State the theorem
theorem at_op_difference : at_op 7 2 - at_op 2 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l476_47680


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l476_47655

theorem unequal_gender_probability (n : ℕ) (p : ℚ) : 
  n = 12 → p = 1/2 → 
  (1 - (Nat.choose n (n/2) : ℚ) * p^(n/2) * (1-p)^(n/2)) = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l476_47655


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l476_47684

theorem least_four_digit_solution (x : ℕ) : x = 1011 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 10 ≡ 19 [ZMOD 7] ∧
     -3 * y + 4 ≡ 2 * y [ZMOD 16]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20]) ∧
  (3 * x + 10 ≡ 19 [ZMOD 7]) ∧
  (-3 * x + 4 ≡ 2 * x [ZMOD 16]) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l476_47684


namespace NUMINAMATH_CALUDE_transformed_dataset_properties_l476_47668

/-- Represents a dataset with its average and variance -/
structure Dataset where
  average : ℝ
  variance : ℝ

/-- Represents a linear transformation of a dataset -/
structure LinearTransform where
  scale : ℝ
  shift : ℝ

/-- Theorem stating the properties of a transformed dataset -/
theorem transformed_dataset_properties (original : Dataset) (transform : LinearTransform) :
  original.average = 3 ∧ 
  original.variance = 4 ∧ 
  transform.scale = 3 ∧ 
  transform.shift = -1 →
  ∃ (transformed : Dataset),
    transformed.average = 8 ∧
    transformed.variance = 36 := by
  sorry

end NUMINAMATH_CALUDE_transformed_dataset_properties_l476_47668


namespace NUMINAMATH_CALUDE_A_intersect_B_l476_47648

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l476_47648


namespace NUMINAMATH_CALUDE_right_triangle_sides_from_medians_l476_47606

/-- Given a right-angled triangle with medians ka and kb, prove the lengths of its sides. -/
theorem right_triangle_sides_from_medians (ka kb : ℝ) 
  (h_ka : ka = 30) (h_kb : kb = 40) : ∃ (a b c : ℝ),
  -- Definition of medians
  ka^2 = (1/4) * (2*b^2 + 2*c^2 - a^2) ∧ 
  kb^2 = (1/4) * (2*a^2 + 2*c^2 - b^2) ∧
  -- Pythagorean theorem
  a^2 + b^2 = c^2 ∧
  -- Side lengths
  a = 20 * Real.sqrt (11/3) ∧
  b = 40 / Real.sqrt 3 ∧
  c = 20 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_from_medians_l476_47606


namespace NUMINAMATH_CALUDE_w_squared_value_l476_47671

theorem w_squared_value (w : ℚ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l476_47671


namespace NUMINAMATH_CALUDE_mushroom_collection_l476_47677

theorem mushroom_collection : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    (n / 100 + (n / 10) % 10 + n % 10 = 14) ∧  -- sum of digits is 14
    n % 50 = 0 ∧  -- divisible by 50
    n = 950 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l476_47677


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l476_47618

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 + 2 > 3*x) ↔ (∀ x : ℝ, x^2 + 2 ≤ 3*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l476_47618


namespace NUMINAMATH_CALUDE_fraction_modification_l476_47642

theorem fraction_modification (x : ℚ) : 
  x = 437 → (537 - x) / (463 + x) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l476_47642


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l476_47640

/-- A quadratic equation of the form ax^2 + bx + c = 0 has no real roots if and only if its discriminant is negative. -/
axiom no_real_roots_iff_neg_discriminant {a b c : ℝ} (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≠ 0) ↔ b^2 - 4*a*c < 0

/-- For the quadratic equation 6x^2 - 5x + a = 0 to have no real roots, a must be greater than 25/24. -/
theorem no_real_roots_condition (a : ℝ) :
  (∀ x, 6 * x^2 - 5 * x + a ≠ 0) ↔ a > 25/24 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l476_47640


namespace NUMINAMATH_CALUDE_function_composition_equality_l476_47662

/-- Given two functions f and g, where f(x) = Ax - 3B^2 and g(x) = Bx + C,
    with B ≠ 0 and f(g(1)) = 0, prove that A = 3B^2 / (B + C) -/
theorem function_composition_equality (A B C : ℝ) (hB : B ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x + C
  f (g 1) = 0 → A = 3 * B^2 / (B + C) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l476_47662


namespace NUMINAMATH_CALUDE_bob_corn_rows_l476_47611

/-- Represents the number of corn stalks in each row -/
def stalks_per_row : ℕ := 80

/-- Represents the number of corn stalks needed to produce one bushel -/
def stalks_per_bushel : ℕ := 8

/-- Represents the total number of bushels Bob will harvest -/
def total_bushels : ℕ := 50

/-- Calculates the number of rows of corn Bob has -/
def number_of_rows : ℕ := (total_bushels * stalks_per_bushel) / stalks_per_row

theorem bob_corn_rows :
  number_of_rows = 5 :=
sorry

end NUMINAMATH_CALUDE_bob_corn_rows_l476_47611


namespace NUMINAMATH_CALUDE_egg_plant_theorem_l476_47687

/-- Represents the egg processing plant scenario --/
structure EggPlant where
  accepted : ℕ
  rejected : ℕ
  total : ℕ
  accepted_to_rejected_ratio : ℚ

/-- The initial state of the egg plant --/
def initial_state : EggPlant := {
  accepted := 0,
  rejected := 0,
  total := 400,
  accepted_to_rejected_ratio := 0
}

/-- The state after additional eggs are accepted --/
def modified_state (initial : EggPlant) : EggPlant := {
  accepted := initial.accepted + 12,
  rejected := initial.rejected - 4,
  total := initial.total,
  accepted_to_rejected_ratio := 99/1
}

/-- The theorem to prove --/
theorem egg_plant_theorem (initial : EggPlant) : 
  initial.accepted = 392 ∧ 
  initial.rejected = 8 ∧
  initial.total = 400 ∧
  (initial.accepted : ℚ) / initial.rejected = (initial.accepted + 12 : ℚ) / (initial.rejected - 4) ∧
  (initial.accepted + 12 : ℚ) / (initial.rejected - 4) = 99/1 := by
  sorry

end NUMINAMATH_CALUDE_egg_plant_theorem_l476_47687


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l476_47639

theorem simple_interest_rate_calculation (P : ℝ) (P_positive : P > 0) :
  let final_amount := (7 / 6) * P
  let time := 6
  let interest := final_amount - P
  let R := (interest / P / time) * 100
  R = 100 / 36 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l476_47639


namespace NUMINAMATH_CALUDE_exam_failure_count_l476_47675

/-- Given an examination where 740 students appeared and 35% passed,
    prove that 481 students failed the examination. -/
theorem exam_failure_count : ∀ (total_students : ℕ) (pass_percentage : ℚ),
  total_students = 740 →
  pass_percentage = 35 / 100 →
  (total_students : ℚ) * (1 - pass_percentage) = 481 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_count_l476_47675


namespace NUMINAMATH_CALUDE_linear_function_characterization_l476_47637

/-- A function satisfying the Cauchy functional equation -/
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A monotonic function -/
def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- A function bounded between 0 and 1 -/
def is_bounded_01 (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ f x ∧ f x ≤ 1

/-- Main theorem: If f satisfies the given conditions, then it is linear -/
theorem linear_function_characterization (f : ℝ → ℝ)
  (h_additive : is_additive f)
  (h_monotonic : is_monotonic f)
  (h_bounded : is_bounded_01 f) :
  ∀ x, f x = x * f 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l476_47637


namespace NUMINAMATH_CALUDE_first_walk_time_l476_47608

/-- Represents a walk with its speed and distance -/
structure Walk where
  speed : ℝ
  distance : ℝ

/-- Proves that the time taken for the first walk is 2 hours given the problem conditions -/
theorem first_walk_time (first_walk : Walk) (second_walk : Walk) 
  (h1 : first_walk.speed = 3)
  (h2 : second_walk.speed = 4)
  (h3 : second_walk.distance = first_walk.distance + 2)
  (h4 : first_walk.distance / first_walk.speed + second_walk.distance / second_walk.speed = 4) :
  first_walk.distance / first_walk.speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_first_walk_time_l476_47608


namespace NUMINAMATH_CALUDE_laptop_savings_weeks_l476_47693

theorem laptop_savings_weeks (laptop_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) 
  (h1 : laptop_cost = 800)
  (h2 : birthday_money = 140)
  (h3 : weekly_earnings = 20) :
  ∃ (weeks : ℕ), birthday_money + weekly_earnings * weeks = laptop_cost ∧ weeks = 33 := by
  sorry

end NUMINAMATH_CALUDE_laptop_savings_weeks_l476_47693


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l476_47699

/-- Given a cubic polynomial x^3 - 2023x + m with three integer roots,
    prove that the sum of the absolute values of the roots is 80. -/
theorem sum_of_absolute_roots (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l476_47699


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l476_47667

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.35 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l476_47667


namespace NUMINAMATH_CALUDE_cars_rented_at_3600_optimal_rent_max_revenue_l476_47600

/-- Represents the rental company's car fleet and pricing model -/
structure RentalCompany where
  totalCars : ℕ
  baseRent : ℕ
  rentIncrement : ℕ
  maintenanceCost : ℕ
  decreaseRate : ℕ

/-- Calculates the number of cars rented at a given rent -/
def carsRented (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.baseRent) / company.rentIncrement

/-- Calculates the revenue at a given rent -/
def revenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  (carsRented company rent) * (rent - company.maintenanceCost)

/-- Theorem stating the correct number of cars rented at 3600 yuan -/
theorem cars_rented_at_3600 (company : RentalCompany) 
  (h1 : company.totalCars = 100)
  (h2 : company.baseRent = 3000)
  (h3 : company.rentIncrement = 50)
  (h4 : company.maintenanceCost = 200)
  (h5 : company.decreaseRate = 1) :
  carsRented company 3600 = 88 := by sorry

/-- Theorem stating the rent that maximizes revenue -/
theorem optimal_rent (company : RentalCompany)
  (h1 : company.totalCars = 100)
  (h2 : company.baseRent = 3000)
  (h3 : company.rentIncrement = 50)
  (h4 : company.maintenanceCost = 200)
  (h5 : company.decreaseRate = 1) :
  ∃ (r : ℕ), ∀ (rent : ℕ), revenue company rent ≤ revenue company r ∧ r = 4100 := by sorry

/-- Theorem stating the maximum revenue -/
theorem max_revenue (company : RentalCompany)
  (h1 : company.totalCars = 100)
  (h2 : company.baseRent = 3000)
  (h3 : company.rentIncrement = 50)
  (h4 : company.maintenanceCost = 200)
  (h5 : company.decreaseRate = 1) :
  ∃ (r : ℕ), ∀ (rent : ℕ), revenue company rent ≤ revenue company r ∧ revenue company r = 304200 := by sorry

end NUMINAMATH_CALUDE_cars_rented_at_3600_optimal_rent_max_revenue_l476_47600


namespace NUMINAMATH_CALUDE_product_of_fractions_squared_l476_47641

theorem product_of_fractions_squared :
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (1 / 4) ^ 2 = 4 / 729 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_squared_l476_47641


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l476_47610

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then x = 6 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l476_47610


namespace NUMINAMATH_CALUDE_sequence_value_l476_47627

theorem sequence_value (a : ℕ → ℕ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 2 * n) : a 100 = 9902 := by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l476_47627


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l476_47674

theorem smallest_non_prime_non_square_with_large_factors : ∃ n : ℕ+, 
  (n.val = 5183 ∧ 
   ¬ Nat.Prime n.val ∧ 
   ¬ ∃ m : ℕ, m ^ 2 = n.val ∧ 
   ∀ p : ℕ, Nat.Prime p → p < 70 → ¬ p ∣ n.val) ∧
  (∀ k : ℕ+, k.val < 5183 → 
    Nat.Prime k.val ∨ 
    (∃ m : ℕ, m ^ 2 = k.val) ∨ 
    (∃ p : ℕ, Nat.Prime p ∧ p < 70 ∧ p ∣ k.val)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l476_47674


namespace NUMINAMATH_CALUDE_common_chord_length_l476_47661

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem common_chord_length : 
  ∃ (a b c d : ℝ), 
    (circle1 a b ∧ circle1 c d ∧ common_chord a b ∧ common_chord c d) →
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 4 * 2^(1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l476_47661


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l476_47658

theorem negative_fraction_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l476_47658


namespace NUMINAMATH_CALUDE_actual_daily_production_l476_47623

/-- The actual daily production of TVs given the planned production and early completion. -/
theorem actual_daily_production
  (planned_production : ℕ)
  (planned_days : ℕ)
  (days_ahead : ℕ)
  (h1 : planned_production = 560)
  (h2 : planned_days = 16)
  (h3 : days_ahead = 2)
  : (planned_production : ℚ) / (planned_days - days_ahead) = 40 := by
  sorry

end NUMINAMATH_CALUDE_actual_daily_production_l476_47623


namespace NUMINAMATH_CALUDE_probability_five_or_king_l476_47645

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (unique_combinations : Bool)

/-- The probability of drawing a specific set of cards from a deck -/
def probability (d : Deck) (favorable_outcomes : Nat) : ℚ :=
  favorable_outcomes / d.cards

/-- Theorem: The probability of drawing a 5 or King from a standard deck is 2/13 -/
theorem probability_five_or_king (d : Deck) 
  (h1 : d.cards = 52)
  (h2 : d.ranks = 13)
  (h3 : d.suits = 4)
  (h4 : d.unique_combinations = true) :
  probability d 8 = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_or_king_l476_47645


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l476_47679

theorem quadratic_no_real_roots (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x - 4*a = 0) ↔ (-16 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l476_47679


namespace NUMINAMATH_CALUDE_rent_increase_group_size_l476_47685

theorem rent_increase_group_size 
  (initial_avg : ℝ) 
  (new_avg : ℝ) 
  (increased_rent : ℝ) 
  (increase_rate : ℝ) :
  initial_avg = 800 →
  new_avg = 870 →
  increased_rent = 1400 →
  increase_rate = 0.2 →
  ∃ n : ℕ, 
    n > 0 ∧
    n * new_avg = (n * initial_avg - increased_rent + increased_rent * (1 + increase_rate)) ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_rent_increase_group_size_l476_47685


namespace NUMINAMATH_CALUDE_special_function_at_two_l476_47602

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 0 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)

/-- Theorem stating that f(2) = 0 for any function satisfying the given conditions -/
theorem special_function_at_two (f : ℝ → ℝ) (h : special_function f) : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_two_l476_47602


namespace NUMINAMATH_CALUDE_deductive_reasoning_examples_l476_47673

-- Define a type for the reasoning examples
inductive ReasoningExample
  | example1
  | example2
  | example3
  | example4

-- Define a function to check if a reasoning example is deductive
def isDeductive : ReasoningExample → Bool
  | ReasoningExample.example1 => false
  | ReasoningExample.example2 => true
  | ReasoningExample.example3 => false
  | ReasoningExample.example4 => true

-- Theorem statement
theorem deductive_reasoning_examples :
  ∀ (e : ReasoningExample), isDeductive e ↔ (e = ReasoningExample.example2 ∨ e = ReasoningExample.example4) := by
  sorry


end NUMINAMATH_CALUDE_deductive_reasoning_examples_l476_47673


namespace NUMINAMATH_CALUDE_probability_b_greater_than_a_l476_47604

def A : Finset ℕ := {2, 3, 4, 5, 6}
def B : Finset ℕ := {1, 2, 3, 5}

theorem probability_b_greater_than_a : 
  (Finset.filter (λ (p : ℕ × ℕ) => p.2 > p.1) (A.product B)).card / (A.card * B.card : ℚ) = 1/5 :=
sorry

end NUMINAMATH_CALUDE_probability_b_greater_than_a_l476_47604


namespace NUMINAMATH_CALUDE_problem_solution_l476_47688

theorem problem_solution (a b : ℝ) (h : a^2 + |b+1| = 0) : (a+b)^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l476_47688


namespace NUMINAMATH_CALUDE_two_greater_than_negative_three_l476_47630

theorem two_greater_than_negative_three : 2 > -3 := by
  sorry

end NUMINAMATH_CALUDE_two_greater_than_negative_three_l476_47630


namespace NUMINAMATH_CALUDE_photo_gallery_total_l476_47633

theorem photo_gallery_total (initial_photos : ℕ) 
  (h1 : initial_photos = 1200) 
  (first_day : ℕ) 
  (h2 : first_day = initial_photos * 3 / 5) 
  (second_day : ℕ) 
  (h3 : second_day = first_day + 230) : 
  initial_photos + first_day + second_day = 2870 := by
  sorry

end NUMINAMATH_CALUDE_photo_gallery_total_l476_47633


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l476_47621

-- Define the sets A and B
def A (m : ℝ) := {x : ℝ | x^2 - 4*m*x + 2*m + 6 = 0}
def B := {x : ℝ | x < 0}

-- State the theorem
theorem intersection_nonempty_implies_m_range (m : ℝ) :
  (A m ∩ B).Nonempty → m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_range_l476_47621


namespace NUMINAMATH_CALUDE_missed_solution_l476_47636

theorem missed_solution (x : ℝ) : x * (x - 3) = x - 3 → (x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_missed_solution_l476_47636


namespace NUMINAMATH_CALUDE_mantou_distribution_theorem_l476_47698

/-- Represents the distribution of mantou among monks -/
structure MantouDistribution where
  bigMonks : ℕ
  smallMonks : ℕ
  totalMonks : ℕ
  totalMantou : ℕ

/-- The mantou distribution satisfies the problem conditions -/
def isValidDistribution (d : MantouDistribution) : Prop :=
  d.bigMonks + d.smallMonks = d.totalMonks ∧
  d.totalMonks = 100 ∧
  d.totalMantou = 100 ∧
  3 * d.bigMonks + (1/3) * d.smallMonks = d.totalMantou

/-- The system of equations correctly represents the mantou distribution -/
theorem mantou_distribution_theorem (d : MantouDistribution) :
  isValidDistribution d ↔
  d.bigMonks + d.smallMonks = 100 ∧
  3 * d.bigMonks + (1/3) * d.smallMonks = 100 :=
sorry

end NUMINAMATH_CALUDE_mantou_distribution_theorem_l476_47698


namespace NUMINAMATH_CALUDE_prob_three_diff_suits_is_169_425_l476_47624

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a standard deck -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Function to get the suit of a card -/
def card_suit : Fin 52 → Suit := sorry

/-- Probability of drawing three cards of different suits -/
def prob_three_different_suits (d : Deck) : ℚ :=
  let first_draw := d.cards.card
  let second_draw := d.cards.card - 1
  let third_draw := d.cards.card - 2
  let diff_suit_second := 39
  let diff_suit_third := 26
  (diff_suit_second / second_draw) * (diff_suit_third / third_draw)

theorem prob_three_diff_suits_is_169_425 (d : Deck) :
  prob_three_different_suits d = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_diff_suits_is_169_425_l476_47624


namespace NUMINAMATH_CALUDE_extended_volume_calculation_l476_47605

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points that are either inside or within one unit
    of a rectangular parallelepiped with the given dimensions -/
def extended_volume (d : Dimensions) : ℝ :=
  sorry

/-- The dimensions of the given rectangular parallelepiped -/
def given_dimensions : Dimensions :=
  { length := 2, width := 3, height := 7 }

theorem extended_volume_calculation :
  extended_volume given_dimensions = (372 + 112 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_extended_volume_calculation_l476_47605


namespace NUMINAMATH_CALUDE_max_value_problem_l476_47601

theorem max_value_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 1 / Real.sqrt 3) : 
  27 * a * b * c + a * Real.sqrt (a^2 + 2*b*c) + b * Real.sqrt (b^2 + 2*c*a) + c * Real.sqrt (c^2 + 2*a*b) 
  ≤ 2 / (3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_max_value_problem_l476_47601


namespace NUMINAMATH_CALUDE_two_thirds_cubed_l476_47689

theorem two_thirds_cubed : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_cubed_l476_47689


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l476_47625

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧
  n % 6 = 2 ∧
  n % 7 = 3 ∧
  n % 8 = 4 ∧
  (∀ m : ℕ, m > 0 → m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) ∧
  n = 164 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l476_47625


namespace NUMINAMATH_CALUDE_modulus_of_complex_l476_47609

theorem modulus_of_complex (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  a / (1 - i) = 1 - b * i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l476_47609


namespace NUMINAMATH_CALUDE_quadratic_property_l476_47696

/-- A quadratic function f(x) = ax² + bx + c with specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > c
  h3 : a + b + c = 0

/-- The set A of m where f(m) < 0 -/
def A (f : QuadraticFunction) : Set ℝ :=
  {m | f.a * m^2 + f.b * m + f.c < 0}

/-- Main theorem: For any m in A, f(m+3) > 0 -/
theorem quadratic_property (f : QuadraticFunction) :
  ∀ m ∈ A f, f.a * (m + 3)^2 + f.b * (m + 3) + f.c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_property_l476_47696


namespace NUMINAMATH_CALUDE_hidden_message_last_word_l476_47614

/-- Represents a color in the embroidery --/
inductive Color
| X | Dot | Ampersand | Colon | Star | GreaterThan | LessThan | S | Equals | Zh

/-- Represents a cell in the embroidery grid --/
structure Cell :=
  (number : ℕ)
  (color : Color)

/-- Represents the embroidery system --/
structure EmbroiderySystem :=
  (p : ℕ)
  (grid : List Cell)
  (letterMapping : Fin 33 → Fin 100)
  (colorMapping : Fin 10 → Color)

/-- Represents a decoded message --/
def DecodedMessage := List Char

/-- Function to decode the embroidery --/
def decodeEmbroidery (system : EmbroiderySystem) : DecodedMessage :=
  sorry

/-- The last word of the decoded message --/
def lastWord (message : DecodedMessage) : String :=
  sorry

/-- Theorem stating that the last word of the decoded message is "магистратура" --/
theorem hidden_message_last_word (system : EmbroiderySystem) :
  lastWord (decodeEmbroidery system) = "магистратура" :=
  sorry

end NUMINAMATH_CALUDE_hidden_message_last_word_l476_47614


namespace NUMINAMATH_CALUDE_chair_cost_l476_47619

/-- Given that Ellen spent $180 for 12 chairs, prove that each chair costs $15. -/
theorem chair_cost (total_spent : ℕ) (num_chairs : ℕ) (h1 : total_spent = 180) (h2 : num_chairs = 12) :
  total_spent / num_chairs = 15 := by
sorry

end NUMINAMATH_CALUDE_chair_cost_l476_47619


namespace NUMINAMATH_CALUDE_square_commutes_with_multiplication_l476_47634

theorem square_commutes_with_multiplication (m n : ℝ) : m^2 * n - n * m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_commutes_with_multiplication_l476_47634


namespace NUMINAMATH_CALUDE_interest_rate_difference_l476_47603

/-- The difference between two simple interest rates given specific conditions -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (interest_diff : ℝ) :
  principal = 2600 →
  time = 3 →
  interest_diff = 78 →
  ∃ (rate1 rate2 : ℝ), rate2 - rate1 = 0.01 ∧
    principal * time * (rate2 - rate1) / 100 = interest_diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l476_47603


namespace NUMINAMATH_CALUDE_right_triangle_circumcenter_angles_l476_47650

theorem right_triangle_circumcenter_angles (α : Real) (h1 : α = 25 * π / 180) :
  let β := π / 2 - α
  let θ₁ := 2 * α
  let θ₂ := 2 * β
  θ₁ = 50 * π / 180 ∧ θ₂ = 130 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumcenter_angles_l476_47650


namespace NUMINAMATH_CALUDE_unique_integer_property_l476_47647

theorem unique_integer_property (a : ℕ+) : 
  let b := 2 * a ^ 2
  let c := 2 * b ^ 2
  let d := 2 * c ^ 2
  (∃ n k : ℕ, a * 10^(n+k) + b * 10^k + c = d) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_property_l476_47647


namespace NUMINAMATH_CALUDE_initial_deposit_is_one_l476_47646

def initial_amount : ℕ := 100
def weeks : ℕ := 52
def final_total : ℕ := 1478

def arithmetic_sum (a₁ : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ))

theorem initial_deposit_is_one :
  ∃ (x : ℚ), 
    arithmetic_sum x weeks + initial_amount = final_total ∧ 
    x = 1 := by sorry

end NUMINAMATH_CALUDE_initial_deposit_is_one_l476_47646


namespace NUMINAMATH_CALUDE_sum_maximized_at_11_or_12_l476_47690

/-- The sequence term defined as a function of n -/
def a (n : ℕ) : ℤ := 24 - 2 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (24 - n)

/-- Theorem stating that the sum is maximized when n is 11 or 12 -/
theorem sum_maximized_at_11_or_12 :
  ∀ k : ℕ, k > 0 → S k ≤ max (S 11) (S 12) :=
sorry

end NUMINAMATH_CALUDE_sum_maximized_at_11_or_12_l476_47690


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_and_integer_intersection_l476_47678

/-- Represents a quadratic function y = mx^2 - (m+2)x + 2 --/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2

theorem parabola_intersects_x_axis_twice_and_integer_intersection (m : ℝ) 
  (hm_nonzero : m ≠ 0) (hm_not_two : m ≠ 2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_function m x1 = 0 ∧ quadratic_function m x2 = 0) ∧
  (∃! m : ℕ+, m ≠ 2 ∧ ∃ x1 x2 : ℤ, quadratic_function ↑m ↑x1 = 0 ∧ quadratic_function ↑m ↑x2 = 0 ∧ x1 ≠ x2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_and_integer_intersection_l476_47678


namespace NUMINAMATH_CALUDE_lewis_age_l476_47657

theorem lewis_age (ages : List Nat) 
  (h1 : ages = [4, 6, 8, 10, 12])
  (h2 : ∃ (a b : Nat), a ∈ ages ∧ b ∈ ages ∧ a + b = 18 ∧ a ≠ b)
  (h3 : ∃ (c d : Nat), c ∈ ages ∧ d ∈ ages ∧ c > 5 ∧ c < 11 ∧ d > 5 ∧ d < 11 ∧ c ≠ d)
  (h4 : 6 ∈ ages)
  (h5 : ∀ (x : Nat), x ∈ ages → x = 4 ∨ x = 6 ∨ x = 8 ∨ x = 10 ∨ x = 12) :
  4 ∈ ages := by
  sorry

end NUMINAMATH_CALUDE_lewis_age_l476_47657


namespace NUMINAMATH_CALUDE_evaluate_expression_l476_47626

theorem evaluate_expression : (2 : ℕ)^(3^2) + 1^(3^3) = 513 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l476_47626


namespace NUMINAMATH_CALUDE_perpendicular_planes_not_transitive_l476_47631

-- Define the type for planes
variable (Plane : Type)

-- Define the perpendicular relation between planes
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_not_transitive :
  ∃ (α β γ : Plane),
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧
    perp α β ∧ perp β γ ∧
    ¬(∀ (α β γ : Plane), perp α β → perp β γ → perp α γ) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_not_transitive_l476_47631


namespace NUMINAMATH_CALUDE_rectangle_area_error_l476_47686

theorem rectangle_area_error (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let measured_length := 1.15 * L
  let measured_width := 1.20 * W
  let true_area := L * W
  let calculated_area := measured_length * measured_width
  (calculated_area - true_area) / true_area * 100 = 38 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_l476_47686


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l476_47617

/-- Given a hyperbola y = k/x where the point (3, -2) lies on it, 
    prove that the point (-2, 3) also lies on the same hyperbola. -/
theorem point_on_hyperbola (k : ℝ) : 
  (∃ k, k = 3 * (-2) ∧ -2 = k / 3) → 
  (∃ k, k = (-2) * 3 ∧ 3 = k / (-2)) := by
  sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l476_47617


namespace NUMINAMATH_CALUDE_max_x_minus_y_l476_47644

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a - b ∧ a^2 + b^2 - 4*a - 2*b - 4 = 0) →
  w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l476_47644


namespace NUMINAMATH_CALUDE_initial_chickens_is_300_l476_47660

/-- Represents the initial state and conditions of the poultry farm problem --/
structure PoultryFarm where
  initial_turkeys : ℕ
  initial_guinea_fowls : ℕ
  daily_loss_chickens : ℕ
  daily_loss_turkeys : ℕ
  daily_loss_guinea_fowls : ℕ
  duration_days : ℕ
  total_remaining : ℕ

/-- Calculates the initial number of chickens given the farm conditions --/
def calculate_initial_chickens (farm : PoultryFarm) : ℕ :=
  let remaining_turkeys := farm.initial_turkeys - farm.daily_loss_turkeys * farm.duration_days
  let remaining_guinea_fowls := farm.initial_guinea_fowls - farm.daily_loss_guinea_fowls * farm.duration_days
  let remaining_chickens := farm.total_remaining - remaining_turkeys - remaining_guinea_fowls
  remaining_chickens + farm.daily_loss_chickens * farm.duration_days

/-- Theorem stating that the initial number of chickens is 300 --/
theorem initial_chickens_is_300 (farm : PoultryFarm)
  (h1 : farm.initial_turkeys = 200)
  (h2 : farm.initial_guinea_fowls = 80)
  (h3 : farm.daily_loss_chickens = 20)
  (h4 : farm.daily_loss_turkeys = 8)
  (h5 : farm.daily_loss_guinea_fowls = 5)
  (h6 : farm.duration_days = 7)
  (h7 : farm.total_remaining = 349) :
  calculate_initial_chickens farm = 300 := by
  sorry

end NUMINAMATH_CALUDE_initial_chickens_is_300_l476_47660


namespace NUMINAMATH_CALUDE_melissa_bonus_points_l476_47666

/-- Given a player's regular points per game, number of games played, and total score,
    calculate the bonus points per game. -/
def bonusPointsPerGame (regularPointsPerGame : ℕ) (numGames : ℕ) (totalScore : ℕ) : ℕ :=
  ((totalScore - regularPointsPerGame * numGames) / numGames : ℕ)

/-- Theorem stating that for the given conditions, the bonus points per game is 82. -/
theorem melissa_bonus_points :
  bonusPointsPerGame 109 79 15089 = 82 := by
  sorry

#eval bonusPointsPerGame 109 79 15089

end NUMINAMATH_CALUDE_melissa_bonus_points_l476_47666


namespace NUMINAMATH_CALUDE_required_machines_eq_ten_l476_47629

/-- The number of cell phones produced by 2 machines per minute -/
def phones_per_2machines : ℕ := 10

/-- The number of machines used in the given condition -/
def given_machines : ℕ := 2

/-- The desired number of cell phones to be produced per minute -/
def desired_phones : ℕ := 50

/-- Calculates the number of machines required to produce the desired number of phones per minute -/
def required_machines : ℕ := desired_phones * given_machines / phones_per_2machines

theorem required_machines_eq_ten : required_machines = 10 := by
  sorry

end NUMINAMATH_CALUDE_required_machines_eq_ten_l476_47629


namespace NUMINAMATH_CALUDE_intersection_M_N_l476_47694

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l476_47694


namespace NUMINAMATH_CALUDE_wall_brick_count_l476_47665

/-- Calculates the total number of bricks in a wall -/
def totalBricks (initialCourses : ℕ) (addedCourses : ℕ) (bricksPerCourse : ℕ) : ℕ :=
  let totalCourses := initialCourses + addedCourses
  let fullCourseBricks := totalCourses * bricksPerCourse
  let removedBricks := bricksPerCourse / 2
  fullCourseBricks - removedBricks

/-- Theorem stating that the wall has 1800 bricks -/
theorem wall_brick_count :
  totalBricks 3 2 400 = 1800 := by
  sorry

#eval totalBricks 3 2 400

end NUMINAMATH_CALUDE_wall_brick_count_l476_47665


namespace NUMINAMATH_CALUDE_simplify_expression_l476_47651

theorem simplify_expression (x : ℝ) : 
  2*x + 4*x^2 - 3 + (5 - 3*x - 9*x^2) + (x+1)^2 = -4*x^2 + x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l476_47651


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l476_47663

theorem lucky_larry_problem (a b c d e : ℝ) : 
  a = 12 ∧ b = 3 ∧ c = 15 ∧ d = 2 →
  (a / b - c - d * e = a / (b - (c - (d * e)))) →
  e = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l476_47663


namespace NUMINAMATH_CALUDE_circle_value_l476_47649

theorem circle_value (circle triangle : ℕ) 
  (eq1 : circle + circle + circle + circle = triangle + triangle + circle)
  (eq2 : triangle = 63) : 
  circle = 42 := by
  sorry

end NUMINAMATH_CALUDE_circle_value_l476_47649


namespace NUMINAMATH_CALUDE_intersection_condition_l476_47607

open Set Real

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem intersection_condition (m : ℝ) :
  A ∩ B m = B m ↔ ((-2 * sqrt 2 < m ∧ m < 2 * sqrt 2) ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l476_47607


namespace NUMINAMATH_CALUDE_annual_increase_rate_l476_47622

theorem annual_increase_rate (initial_value final_value : ℝ) 
  (h1 : initial_value = 6400)
  (h2 : final_value = 8100) :
  ∃ r : ℝ, initial_value * (1 + r)^2 = final_value ∧ r = 0.125 := by
sorry

end NUMINAMATH_CALUDE_annual_increase_rate_l476_47622


namespace NUMINAMATH_CALUDE_ceiling_equality_iff_x_in_range_l476_47653

theorem ceiling_equality_iff_x_in_range (x : ℝ) : 
  ⌈⌈3*x⌉ + 1/2⌉ = ⌈x - 2⌉ ↔ x ∈ Set.Icc (-1) (-2/3) :=
sorry

end NUMINAMATH_CALUDE_ceiling_equality_iff_x_in_range_l476_47653


namespace NUMINAMATH_CALUDE_product_equality_l476_47612

theorem product_equality (a b : ℝ) (h1 : 4 * a = 30) (h2 : 5 * b = 30) : 40 * a * b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l476_47612


namespace NUMINAMATH_CALUDE_part_one_part_two_l476_47654

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x, f x a - |x - a| ≤ 2 ↔ x ∈ Set.Icc (-5) (-1)) → a = 2 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ 2 < 4*m + m^2) → m < -5 ∨ m > 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l476_47654


namespace NUMINAMATH_CALUDE_expression_C_is_negative_l476_47692

-- Define the variables with their approximate values
def A : ℝ := -4.2
def B : ℝ := 2.3
def C : ℝ := -0.5
def D : ℝ := 3.4
def E : ℝ := -1.8

-- Theorem stating that the expression (D/B) * C is negative
theorem expression_C_is_negative : (D / B) * C < 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_C_is_negative_l476_47692


namespace NUMINAMATH_CALUDE_point_movement_to_y_axis_l476_47681

/-- 
Given a point P with coordinates (a+1, a), if P is moved 3 units to the right 
to get P₁, and P₁ lies on the y-axis, then P's coordinates are (-3, -4).
-/
theorem point_movement_to_y_axis (a : ℝ) :
  let P : ℝ × ℝ := (a + 1, a)
  let P₁ : ℝ × ℝ := (a + 4, a)
  (P₁.1 = 0) → P = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_to_y_axis_l476_47681


namespace NUMINAMATH_CALUDE_square_side_length_l476_47672

/-- Given six identical squares arranged to form a larger rectangle ABCD with an area of 3456,
    the side length of each square is 24. -/
theorem square_side_length (s : ℝ) : s > 0 → s * s * 6 = 3456 → s = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l476_47672


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l476_47664

theorem complex_fraction_simplification :
  (3 - Complex.I) / (1 - Complex.I) = 2 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l476_47664


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l476_47683

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (5 * y) % 31 = 17 % 31 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l476_47683


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l476_47616

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b / a = 4 / 3) →
  (c / a = 6 / 3) →
  a * b * c = 1944 →
  max a (max b c) = 18 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l476_47616


namespace NUMINAMATH_CALUDE_blue_balloons_l476_47691

theorem blue_balloons (total red green purple : ℕ) (h1 : total = 135) (h2 : red = 45) (h3 : green = 27) (h4 : purple = 32) :
  total - (red + green + purple) = 31 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloons_l476_47691


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_two_l476_47682

theorem sqrt_fraction_equals_two (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 7) :
  Real.sqrt ((14 * a^2) / b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_two_l476_47682


namespace NUMINAMATH_CALUDE_max_page_number_l476_47659

def count_digit_2 (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := (n / 10) % 10
  let hundreds := (n / 100) % 10
  (if ones = 2 then 1 else 0) +
  (if tens = 2 then 1 else 0) +
  (if hundreds = 2 then 1 else 0)

def total_2s_up_to (n : ℕ) : ℕ :=
  (List.range (n + 1)).map count_digit_2 |> List.sum

theorem max_page_number (available_2s : ℕ) (h : available_2s = 100) : 
  ∃ (max_page : ℕ), max_page = 244 ∧ 
    total_2s_up_to max_page ≤ available_2s ∧
    ∀ (n : ℕ), n > max_page → total_2s_up_to n > available_2s :=
by sorry

end NUMINAMATH_CALUDE_max_page_number_l476_47659


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l476_47620

/-- Calculates the average number of visitors per day in a library for a 30-day month -/
def average_visitors (sundays : Nat) (sunday_visitors : Nat) (regular_visitors : Nat) (holiday_visitors : Nat) (holidays : Nat) : Nat :=
  let regular_days := 30 - sundays - holidays
  let total_visitors := sundays * sunday_visitors + regular_days * regular_visitors + holidays * holiday_visitors
  total_visitors / 30

/-- Theorem stating the average number of visitors for different scenarios -/
theorem library_visitors_theorem (sundays : Nat) (sunday_visitors : Nat) (regular_visitors : Nat) (holiday_visitors : Nat) (holidays : Nat) :
  (sundays = 4 ∨ sundays = 5) →
  sunday_visitors = 510 →
  regular_visitors = 240 →
  holiday_visitors = 375 →
  holidays = 2 →
  (average_visitors sundays sunday_visitors regular_visitors holiday_visitors holidays = 
    if sundays = 4 then 285 else 294) :=
by
  sorry

#eval average_visitors 4 510 240 375 2
#eval average_visitors 5 510 240 375 2

end NUMINAMATH_CALUDE_library_visitors_theorem_l476_47620


namespace NUMINAMATH_CALUDE_max_m_inequality_l476_47695

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m : ℝ, ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b ≥ m/(2*a+b)) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b ≥ m/(2*a+b)) → m ≤ 9) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2/a + 1/b = 9/(2*a+b)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l476_47695


namespace NUMINAMATH_CALUDE_sin_80_sin_40_minus_cos_80_cos_40_l476_47669

theorem sin_80_sin_40_minus_cos_80_cos_40 :
  Real.sin (80 * π / 180) * Real.sin (40 * π / 180) -
  Real.cos (80 * π / 180) * Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_80_sin_40_minus_cos_80_cos_40_l476_47669


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l476_47635

/-- A point in a 2D coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis. -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting P(3, -4) across the y-axis results in P'(-3, -4). -/
theorem reflection_across_y_axis :
  let P : Point := { x := 3, y := -4 }
  let P' : Point := reflectAcrossYAxis P
  P'.x = -3 ∧ P'.y = -4 := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l476_47635


namespace NUMINAMATH_CALUDE_tan_y_plus_pi_third_l476_47628

theorem tan_y_plus_pi_third (y : Real) (h : Real.tan y = -3) :
  Real.tan (y + π / 3) = -(5 * Real.sqrt 3 - 6) / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_y_plus_pi_third_l476_47628
