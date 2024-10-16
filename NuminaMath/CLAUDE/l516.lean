import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l516_51694

theorem geometric_sequence_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ b = a * q ∧ c = a * q^2) :
  ∃ r : ℝ, r > 0 ∧
    (a + b + c) = (Real.sqrt (3 * (a * b + b * c + c * a))) * r ∧
    (Real.sqrt (3 * (a * b + b * c + c * a))) = (27 * a * b * c)^(1/3) * r :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l516_51694


namespace NUMINAMATH_CALUDE_trapezium_area_l516_51697

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (a + b) * h / 2 = 285 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l516_51697


namespace NUMINAMATH_CALUDE_motorcycle_material_cost_is_250_l516_51605

/-- Represents the factory's production and sales data -/
structure FactoryData where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycles_sold : ℕ
  motorcycle_price : ℕ
  profit_increase : ℕ

/-- Calculates the cost of materials for motorcycle production -/
def motorcycle_material_cost (data : FactoryData) : ℕ :=
  data.motorcycles_sold * data.motorcycle_price -
  (data.cars_produced * data.car_price - data.car_material_cost + data.profit_increase)

/-- Theorem stating the cost of materials for motorcycle production -/
theorem motorcycle_material_cost_is_250 (data : FactoryData)
  (h1 : data.car_material_cost = 100)
  (h2 : data.cars_produced = 4)
  (h3 : data.car_price = 50)
  (h4 : data.motorcycles_sold = 8)
  (h5 : data.motorcycle_price = 50)
  (h6 : data.profit_increase = 50) :
  motorcycle_material_cost data = 250 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_material_cost_is_250_l516_51605


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l516_51666

theorem wizard_elixir_combinations (herbs : ℕ) (gems : ℕ) (incompatible_combinations : ℕ) : 
  herbs = 4 → gems = 5 → incompatible_combinations = 3 →
  herbs * gems - incompatible_combinations = 17 := by
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l516_51666


namespace NUMINAMATH_CALUDE_max_uncovered_corridor_length_l516_51632

theorem max_uncovered_corridor_length 
  (corridor_length : ℝ) 
  (num_rugs : ℕ) 
  (total_rug_length : ℝ) 
  (h1 : corridor_length = 100)
  (h2 : num_rugs = 20)
  (h3 : total_rug_length = 1000) :
  (corridor_length - (total_rug_length - corridor_length)) ≤ 50 := by
sorry

end NUMINAMATH_CALUDE_max_uncovered_corridor_length_l516_51632


namespace NUMINAMATH_CALUDE_age_solution_l516_51696

/-- The age equation as described in the problem -/
def age_equation (x : ℝ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

/-- Theorem stating that 18 is the solution to the age equation -/
theorem age_solution : ∃ x : ℝ, age_equation x ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_solution_l516_51696


namespace NUMINAMATH_CALUDE_min_people_hat_glove_not_scarf_l516_51618

theorem min_people_hat_glove_not_scarf (n : ℕ) 
  (gloves : ℕ) (hats : ℕ) (scarves : ℕ) :
  gloves = (3 * n) / 8 ∧ 
  hats = (5 * n) / 6 ∧ 
  scarves = n / 4 →
  ∃ (x : ℕ), x = hats + gloves - (n - scarves) ∧ 
  x ≥ 11 ∧ 
  (∀ (m : ℕ), m < n → 
    (3 * m) % 8 ≠ 0 ∨ (5 * m) % 6 ≠ 0 ∨ m % 4 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_people_hat_glove_not_scarf_l516_51618


namespace NUMINAMATH_CALUDE_vector_problem_l516_51607

-- Define the vectors a and b
def a (m : ℝ) : Fin 2 → ℝ := ![m, 2]
def b : Fin 2 → ℝ := ![2, -3]

-- Define the parallel condition
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v 0 * w 1 = k * v 1 * w 0

-- State the theorem
theorem vector_problem (m : ℝ) :
  are_parallel (a m + b) (a m - b) → m = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l516_51607


namespace NUMINAMATH_CALUDE_manager_percentage_l516_51630

theorem manager_percentage (total_employees : ℕ) (initial_percentage : ℚ) 
  (managers_leaving : ℚ) (final_percentage : ℚ) : 
  total_employees = 300 →
  initial_percentage = 99/100 →
  managers_leaving = 149.99999999999986 →
  final_percentage = 49/100 →
  (↑total_employees * initial_percentage - managers_leaving) / ↑total_employees = final_percentage :=
by sorry

end NUMINAMATH_CALUDE_manager_percentage_l516_51630


namespace NUMINAMATH_CALUDE_bottle_production_l516_51626

/-- Given that 4 identical machines produce 16 bottles per minute at a constant rate,
    prove that 8 such machines will produce 96 bottles in 3 minutes. -/
theorem bottle_production (machines : ℕ) (bottles_per_minute : ℕ) (time : ℕ) : 
  machines = 4 → bottles_per_minute = 16 → time = 3 →
  (2 * machines) * (bottles_per_minute / machines) * time = 96 := by
  sorry

#check bottle_production

end NUMINAMATH_CALUDE_bottle_production_l516_51626


namespace NUMINAMATH_CALUDE_boys_camp_total_l516_51651

theorem boys_camp_total (total : ℝ) 
  (h1 : 0.2 * total = total_school_a)
  (h2 : 0.3 * total_school_a = science_school_a)
  (h3 : total_school_a - science_school_a = 28) : 
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l516_51651


namespace NUMINAMATH_CALUDE_complex_equation_sum_l516_51633

theorem complex_equation_sum (x y : ℝ) : (2*x - y : ℂ) + (x + 3)*I = 0 → x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l516_51633


namespace NUMINAMATH_CALUDE_xiao_ming_final_position_l516_51621

-- Define the stores and their positions relative to the bookstore
def stationery_store : ℤ := -200
def bookstore : ℤ := 0
def toy_store : ℤ := 100

-- Define Xiao Ming's movements
def first_movement : ℤ := 40
def second_movement : ℤ := -60

-- Theorem to prove
theorem xiao_ming_final_position :
  bookstore + first_movement + second_movement = toy_store :=
sorry

end NUMINAMATH_CALUDE_xiao_ming_final_position_l516_51621


namespace NUMINAMATH_CALUDE_average_income_is_400_l516_51606

def daily_incomes : List ℝ := [300, 150, 750, 200, 600]

theorem average_income_is_400 : 
  (daily_incomes.sum / daily_incomes.length : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_income_is_400_l516_51606


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l516_51662

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 243 ways to put 5 distinguishable balls in 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l516_51662


namespace NUMINAMATH_CALUDE_probability_of_car_Z_winning_l516_51660

/-- Given a race with 15 cars, prove that the probability of car Z winning is 1/12 -/
theorem probability_of_car_Z_winning (total_cars : ℕ) (prob_X prob_Y prob_XYZ : ℚ) :
  total_cars = 15 →
  prob_X = 1/4 →
  prob_Y = 1/8 →
  prob_XYZ = 458333333333333333/1000000000000000000 →
  prob_XYZ = prob_X + prob_Y + (1/12) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_car_Z_winning_l516_51660


namespace NUMINAMATH_CALUDE_consecutive_prime_product_l516_51672

-- Define the first four consecutive prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define the product of these primes
def product_of_primes : Nat := first_four_primes.prod

theorem consecutive_prime_product :
  (product_of_primes = 210) ∧
  (product_of_primes % 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_prime_product_l516_51672


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_attained_l516_51684

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((1 - x)^2 + (-1 + x)^2) ≥ Real.sqrt 10 :=
sorry

theorem min_value_sqrt_sum_attained : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((1 - x)^2 + (-1 + x)^2) = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_min_value_sqrt_sum_attained_l516_51684


namespace NUMINAMATH_CALUDE_money_ratio_problem_l516_51636

/-- Proves that given the ratios between Ravi, Giri, and Kiran's money, and the fact that Ravi has $36, Kiran must have $105. -/
theorem money_ratio_problem (ravi giri kiran : ℕ) : 
  (ravi : ℚ) / giri = 6 / 7 → 
  (giri : ℚ) / kiran = 6 / 15 → 
  ravi = 36 → 
  kiran = 105 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l516_51636


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l516_51612

/-- Given a quadratic equation x^2 + mx - 2 = 0 where -1 is a root,
    prove that m = -1 and the other root is 2 -/
theorem quadratic_equation_roots (m : ℝ) : 
  ((-1 : ℝ)^2 + m*(-1) - 2 = 0) → 
  (m = -1 ∧ ∃ r : ℝ, r ≠ -1 ∧ r^2 + m*r - 2 = 0 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l516_51612


namespace NUMINAMATH_CALUDE_absolute_value_sum_equality_l516_51657

theorem absolute_value_sum_equality (a b c d : ℝ) 
  (h1 : |a - b| + |c - d| = 99)
  (h2 : |a - c| + |b - d| = 1) :
  |a - d| + |b - c| = 99 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_equality_l516_51657


namespace NUMINAMATH_CALUDE_total_fruits_is_213_l516_51608

/-- Represents a fruit grower -/
structure FruitGrower where
  watermelons : ℕ
  pineapples : ℕ

/-- Calculates the total fruits grown by a single grower -/
def totalFruits (grower : FruitGrower) : ℕ :=
  grower.watermelons + grower.pineapples

/-- Represents the group of fruit growers -/
def fruitGrowers : List FruitGrower :=
  [{ watermelons := 37, pineapples := 56 },  -- Jason
   { watermelons := 68, pineapples := 27 },  -- Mark
   { watermelons := 11, pineapples := 14 }]  -- Sandy

/-- Theorem: The total number of fruits grown by the group is 213 -/
theorem total_fruits_is_213 : 
  (fruitGrowers.map totalFruits).sum = 213 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_213_l516_51608


namespace NUMINAMATH_CALUDE_subset_of_A_l516_51627

def A : Set ℝ := {x | x ≤ 10}

theorem subset_of_A : {2} ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_subset_of_A_l516_51627


namespace NUMINAMATH_CALUDE_number_of_factors_of_M_l516_51691

def M : ℕ := 57^5 + 5*57^4 + 10*57^3 + 10*57^2 + 5*57 + 1

theorem number_of_factors_of_M : 
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 36 :=
by sorry

end NUMINAMATH_CALUDE_number_of_factors_of_M_l516_51691


namespace NUMINAMATH_CALUDE_police_force_competition_l516_51671

theorem police_force_competition (x y : ℕ) : 
  (70 * x + 60 * y = 740) → 
  ((x = 8 ∧ y = 3) ∨ (x = 2 ∧ y = 10)) := by
sorry

end NUMINAMATH_CALUDE_police_force_competition_l516_51671


namespace NUMINAMATH_CALUDE_find_other_number_l516_51601

/-- Given two positive integers with known LCM, HCF, and one of the numbers, prove the value of the other number -/
theorem find_other_number (a b : ℕ+) (h1 : Nat.lcm a b = 76176) (h2 : Nat.gcd a b = 116) (h3 : a = 8128) : b = 1087 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l516_51601


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l516_51668

theorem factorization_of_4x_squared_minus_16 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l516_51668


namespace NUMINAMATH_CALUDE_polyhedron_has_triangle_l516_51669

/-- A polyhedron with edges of non-increasing lengths -/
structure Polyhedron where
  n : ℕ
  edges : Fin n → ℝ
  edges_decreasing : ∀ i j, i ≤ j → edges i ≥ edges j

/-- Three edges can form a triangle if the sum of any two is greater than the third -/
def CanFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- In any polyhedron, there exist three edges that can form a triangle -/
theorem polyhedron_has_triangle (P : Polyhedron) :
  ∃ i j k, i < j ∧ j < k ∧ CanFormTriangle (P.edges i) (P.edges j) (P.edges k) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_has_triangle_l516_51669


namespace NUMINAMATH_CALUDE_train_speed_l516_51658

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 320) (h2 : time = 16) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l516_51658


namespace NUMINAMATH_CALUDE_special_triangle_base_l516_51622

/-- Triangle with specific side lengths -/
structure SpecialTriangle where
  left : ℝ
  right : ℝ
  base : ℝ
  sum_of_sides : left + right + base = 50
  right_longer : right = left + 2
  left_length : left = 12

/-- The base of the special triangle is 24 cm -/
theorem special_triangle_base (t : SpecialTriangle) : t.base = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_base_l516_51622


namespace NUMINAMATH_CALUDE_a_is_best_l516_51698

-- Define the structure for an athlete
structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

-- Define the athletes
def athleteA : Athlete := ⟨"A", 185, 3.6⟩
def athleteB : Athlete := ⟨"B", 180, 3.6⟩
def athleteC : Athlete := ⟨"C", 185, 7.4⟩
def athleteD : Athlete := ⟨"D", 180, 8.1⟩

-- Define a function to compare athletes
def isBetterAthlete (a1 a2 : Athlete) : Prop :=
  (a1.average > a2.average) ∨ (a1.average = a2.average ∧ a1.variance < a2.variance)

-- Theorem stating that A is the best athlete
theorem a_is_best : 
  isBetterAthlete athleteA athleteB ∧ 
  isBetterAthlete athleteA athleteC ∧ 
  isBetterAthlete athleteA athleteD :=
sorry

end NUMINAMATH_CALUDE_a_is_best_l516_51698


namespace NUMINAMATH_CALUDE_probability_of_prime_sum_two_dice_l516_51649

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The set of possible sums when rolling two dice -/
def possibleSums : Finset ℕ := sorry

/-- The set of prime sums when rolling two dice -/
def primeSums : Finset ℕ := sorry

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

theorem probability_of_prime_sum_two_dice :
  (Finset.card primeSums : ℚ) / totalOutcomes = 23 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_of_prime_sum_two_dice_l516_51649


namespace NUMINAMATH_CALUDE_abcd_multiplication_l516_51635

theorem abcd_multiplication (A B C D : ℕ) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) →
  (1000 * A + 100 * B + 10 * C + D) * 9 = 1000 * D + 100 * C + 10 * B + A →
  A = 1 ∧ B = 0 ∧ C = 8 ∧ D = 9 := by
sorry

end NUMINAMATH_CALUDE_abcd_multiplication_l516_51635


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l516_51683

theorem no_real_roots_condition (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l516_51683


namespace NUMINAMATH_CALUDE_angle_measure_l516_51685

theorem angle_measure (x : ℝ) : 
  (90 - x) = 3 * (180 - x) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l516_51685


namespace NUMINAMATH_CALUDE_no_valid_house_numbers_l516_51676

def is_two_digit_prime (n : ℕ) : Prop :=
  10 < n ∧ n < 50 ∧ Nat.Prime n

def valid_house_number (w x y z : ℕ) : Prop :=
  w ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  is_two_digit_prime (w * 10 + x) ∧
  is_two_digit_prime (y * 10 + z) ∧
  (w * 10 + x) ≠ (y * 10 + z) ∧
  w + x + y + z = 19

theorem no_valid_house_numbers :
  ¬ ∃ w x y z : ℕ, valid_house_number w x y z :=
sorry

end NUMINAMATH_CALUDE_no_valid_house_numbers_l516_51676


namespace NUMINAMATH_CALUDE_min_pours_to_half_l516_51679

def water_remaining (n : ℕ) : ℝ := (0.9 : ℝ) ^ n

theorem min_pours_to_half : 
  (∀ k < 7, water_remaining k ≥ 0.5) ∧ 
  (water_remaining 7 < 0.5) := by
sorry

end NUMINAMATH_CALUDE_min_pours_to_half_l516_51679


namespace NUMINAMATH_CALUDE_exponent_equality_l516_51620

theorem exponent_equality : 
  (2^3 ≠ (-3)^2) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-3 * 2^3 ≠ (-3 * 2)^3) := by
sorry

end NUMINAMATH_CALUDE_exponent_equality_l516_51620


namespace NUMINAMATH_CALUDE_inequality_proof_l516_51644

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 1)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x^2 + y^2 + z^2) * (a^3 / (x^2 + 2*y^2) + b^3 / (y^2 + 2*z^2) + c^3 / (z^2 + 2*x^2)) ≥ 1/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l516_51644


namespace NUMINAMATH_CALUDE_expression_value_l516_51628

theorem expression_value (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  ∃ ε > 0, |x + x^3/y^2 + y^3/x^2 + y - 440| < ε :=
sorry

end NUMINAMATH_CALUDE_expression_value_l516_51628


namespace NUMINAMATH_CALUDE_a_equals_one_range_of_f_final_no_fixed_points_l516_51640

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + (a^2 - 1) * x + 1

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Theorem 1: If f is an even function, then a = 1 -/
theorem a_equals_one (a : ℝ) (h : is_even_function (f a)) : a = 1 := by
  sorry

/-- The quadratic function f(x) with a = 1 -/
def f_final (x : ℝ) : ℝ := 2 * x^2 + 1

/-- Theorem 2: If x ∈ [-1, 2], then the range of f_final(x) is [1, 9] -/
theorem range_of_f_final : 
  ∀ y ∈ Set.range f_final, y ∈ Set.Icc 1 9 ∧ 
  ∃ x ∈ Set.Icc (-1) 2, f_final x = 1 ∧
  ∃ x ∈ Set.Icc (-1) 2, f_final x = 9 := by
  sorry

/-- Theorem 3: The equation 2x^2 + 1 = x has no real solutions -/
theorem no_fixed_points : ¬ ∃ x : ℝ, f_final x = x := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_range_of_f_final_no_fixed_points_l516_51640


namespace NUMINAMATH_CALUDE_candy_distribution_proof_l516_51647

/-- Given a number of candy pieces and sisters, returns the minimum number of pieces to remove for equal distribution. -/
def minPiecesToRemove (totalPieces sisters : ℕ) : ℕ :=
  totalPieces % sisters

theorem candy_distribution_proof :
  minPiecesToRemove 20 3 = 2 := by
  sorry

#eval minPiecesToRemove 20 3

end NUMINAMATH_CALUDE_candy_distribution_proof_l516_51647


namespace NUMINAMATH_CALUDE_apple_percentage_after_orange_removal_l516_51617

/-- Calculates the percentage of apples in a bowl of fruit after removing oranges -/
theorem apple_percentage_after_orange_removal (initial_apples initial_oranges removed_oranges : ℕ) 
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 20)
  (h3 : removed_oranges = 14) :
  (initial_apples : ℚ) / (initial_apples + initial_oranges - removed_oranges) * 100 = 70 := by
  sorry


end NUMINAMATH_CALUDE_apple_percentage_after_orange_removal_l516_51617


namespace NUMINAMATH_CALUDE_tommy_nickels_count_l516_51689

/-- Proves that Tommy has 100 nickels given the relationships between his coins -/
theorem tommy_nickels_count : 
  ∀ (quarters pennies dimes nickels : ℕ),
    quarters = 4 →
    pennies = 10 * quarters →
    dimes = pennies + 10 →
    nickels = 2 * dimes →
    nickels = 100 := by sorry

end NUMINAMATH_CALUDE_tommy_nickels_count_l516_51689


namespace NUMINAMATH_CALUDE_total_jellybeans_l516_51646

/-- The number of jellybeans in a bag with black, green, and orange beans. -/
def jellybean_count (black green orange : ℕ) : ℕ :=
  black + green + orange

/-- Theorem stating the total number of jellybeans in the bag -/
theorem total_jellybeans :
  ∃ (black green orange : ℕ),
    black = 8 ∧
    green = black + 2 ∧
    orange = green - 1 ∧
    jellybean_count black green orange = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_l516_51646


namespace NUMINAMATH_CALUDE_rod_speed_l516_51648

/-- 
Given a rod moving freely between a horizontal floor and a slanted wall:
- v: speed of the end in contact with the floor
- θ: angle between the rod and the horizontal floor
- α: angle such that (α - θ) is the angle between the rod and the slanted wall

This theorem states that the speed of the end in contact with the wall 
is v * cos(θ) / cos(α - θ)
-/
theorem rod_speed (v θ α : ℝ) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_rod_speed_l516_51648


namespace NUMINAMATH_CALUDE_problem_solution_l516_51682

theorem problem_solution : (12 : ℝ)^2 * 6^3 / 432 = 72 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l516_51682


namespace NUMINAMATH_CALUDE_class_size_l516_51604

theorem class_size (S : ℚ) 
  (basketball : S / 2 = S * (1 / 2))
  (volleyball : S * (2 / 5) = S * (2 / 5))
  (both : S / 10 = S * (1 / 10))
  (neither : S * (1 / 5) = 4) : S = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l516_51604


namespace NUMINAMATH_CALUDE_initial_pups_proof_l516_51615

/-- The number of initial mice -/
def initial_mice : ℕ := 8

/-- The number of additional pups each mouse has in the second round -/
def second_round_pups : ℕ := 6

/-- The number of pups eaten by each adult mouse -/
def eaten_pups : ℕ := 2

/-- The total number of mice at the end -/
def total_mice : ℕ := 280

/-- The initial number of pups per mouse -/
def initial_pups_per_mouse : ℕ := 6

theorem initial_pups_proof :
  initial_mice +
  initial_mice * initial_pups_per_mouse +
  (initial_mice + initial_mice * initial_pups_per_mouse) * second_round_pups -
  (initial_mice + initial_mice * initial_pups_per_mouse) * eaten_pups =
  total_mice :=
by sorry

end NUMINAMATH_CALUDE_initial_pups_proof_l516_51615


namespace NUMINAMATH_CALUDE_equation_solution_l516_51693

theorem equation_solution : ∃! x : ℝ, (1 / 6 + 6 / x = 10 / x + 1 / 15) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l516_51693


namespace NUMINAMATH_CALUDE_tangency_point_unique_l516_51625

/-- The point of tangency between two parabolas -/
def point_of_tangency : ℝ × ℝ := (-9.5, -31.5)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 20*x + 72

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 64*y + 992

theorem tangency_point_unique :
  ∀ x y : ℝ, parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
sorry

end NUMINAMATH_CALUDE_tangency_point_unique_l516_51625


namespace NUMINAMATH_CALUDE_triangle_side_length_l516_51680

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (t : Triangle) (side : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (t : Triangle) :
  -- Conditions
  (median t t.B = (1/3) * length t.B t.C) →
  (length t.A t.B = 3) →
  (length t.A t.C = 2) →
  -- Conclusion
  length t.B t.C = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l516_51680


namespace NUMINAMATH_CALUDE_fraction_simplification_l516_51686

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (1 - 1 / (x + 2)) / ((x^2 - 1) / (x + 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l516_51686


namespace NUMINAMATH_CALUDE_prime_triplets_equation_l516_51643

theorem prime_triplets_equation :
  ∀ p q r : ℕ,
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
    (p : ℚ) / q = 8 / (r - 1) + 1 →
    ((p = 3 ∧ q = 2 ∧ r = 17) ∨
     (p = 7 ∧ q = 3 ∧ r = 7) ∨
     (p = 5 ∧ q = 3 ∧ r = 13)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triplets_equation_l516_51643


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l516_51681

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≠ 0 ∧ hundreds + tens + ones = 7

/-- Represents a car journey -/
structure CarJourney where
  duration : Nat
  average_speed : Nat
  initial_reading : OdometerReading
  final_reading : OdometerReading
  speed_constraint : average_speed = 60
  odometer_constraint : final_reading.hundreds = initial_reading.ones ∧
                        final_reading.tens = initial_reading.tens ∧
                        final_reading.ones = initial_reading.hundreds

theorem odometer_sum_squares (journey : CarJourney) :
  journey.initial_reading.hundreds ^ 2 +
  journey.initial_reading.tens ^ 2 +
  journey.initial_reading.ones ^ 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l516_51681


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l516_51674

/-- A triangle with sides 2a, 2a+2, and 2a+4 is a right triangle if and only if a = 3 -/
theorem right_triangle_consecutive_even_sides (a : ℕ) : 
  (2*a)^2 + (2*a+2)^2 = (2*a+4)^2 ↔ a = 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_even_sides_l516_51674


namespace NUMINAMATH_CALUDE_weight_of_new_man_l516_51623

/-- Given a group of 15 men where replacing a 75 kg man with a new man increases the average weight by 2 kg, the weight of the new man is 105 kg. -/
theorem weight_of_new_man (num_men : ℕ) (weight_replaced : ℝ) (avg_increase : ℝ) (weight_new : ℝ) : 
  num_men = 15 → 
  weight_replaced = 75 → 
  avg_increase = 2 → 
  weight_new = num_men * avg_increase + weight_replaced → 
  weight_new = 105 := by
sorry

end NUMINAMATH_CALUDE_weight_of_new_man_l516_51623


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l516_51695

theorem greatest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 36 ≤ 0 → n ≤ 9 ∧
  ∃ m : ℤ, m^2 - 13*m + 36 ≤ 0 ∧ m = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l516_51695


namespace NUMINAMATH_CALUDE_weekend_to_weekday_practice_ratio_l516_51609

/-- Given Daniel's basketball practice schedule, prove the ratio of weekend to weekday practice time -/
theorem weekend_to_weekday_practice_ratio :
  let weekday_daily_practice : ℕ := 15
  let weekday_count : ℕ := 5
  let total_weekly_practice : ℕ := 135
  let weekday_practice := weekday_daily_practice * weekday_count
  let weekend_practice := total_weekly_practice - weekday_practice
  (weekend_practice : ℚ) / weekday_practice = 4 / 5 := by
sorry


end NUMINAMATH_CALUDE_weekend_to_weekday_practice_ratio_l516_51609


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l516_51692

theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a = -12 ∧ b = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l516_51692


namespace NUMINAMATH_CALUDE_min_sum_power_mod_l516_51641

theorem min_sum_power_mod (m n : ℕ) : 
  n > m → 
  m > 1 → 
  (1978^m) % 1000 = (1978^n) % 1000 → 
  ∃ (m₀ n₀ : ℕ), m₀ + n₀ = 106 ∧ 
    ∀ (m' n' : ℕ), n' > m' → m' > 1 → 
      (1978^m') % 1000 = (1978^n') % 1000 → 
      m' + n' ≥ m₀ + n₀ :=
by sorry

end NUMINAMATH_CALUDE_min_sum_power_mod_l516_51641


namespace NUMINAMATH_CALUDE_triangle_properties_l516_51639

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

def isArithmeticSequence (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

def aEquals2c (t : Triangle) : Prop :=
  t.a = 2 * t.c

def areaIs3Sqrt15Over4 (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15 / 4

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isArithmeticSequence t)
  (h3 : aEquals2c t)
  (h4 : areaIs3Sqrt15Over4 t) :
  Real.cos t.A = -1/4 ∧ t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l516_51639


namespace NUMINAMATH_CALUDE_conditional_probability_l516_51624

open Set
open Finset

def Ω : Finset ℕ := {1,2,3,4,5,6}
def A : Finset ℕ := {2,3,5}
def B : Finset ℕ := {1,2,4,5,6}

def P (X : Finset ℕ) : ℚ := (X.card : ℚ) / (Ω.card : ℚ)

theorem conditional_probability : 
  P (A ∩ B) / P B = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_l516_51624


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l516_51610

/-- A geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)

/-- Theorem: In a geometric sequence where a₁a₈³a₁₅ = 243, the value of a₉³/a₁₁ is 9 -/
theorem geometric_sequence_property (seq : GeometricSequence) 
    (h : seq.a 1 * (seq.a 8)^3 * seq.a 15 = 243) :
    (seq.a 9)^3 / seq.a 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l516_51610


namespace NUMINAMATH_CALUDE_find_a_l516_51661

theorem find_a (x y : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : ∃ a : ℝ, a * x - y = 3) : 
  ∃ a : ℝ, a = 5 ∧ a * x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l516_51661


namespace NUMINAMATH_CALUDE_cloak_change_in_silver_l516_51699

/-- Represents the price of an invisibility cloak and the change received in different scenarios --/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the exchange rate between silver and gold coins --/
def exchange_rate (t1 t2 : CloakTransaction) : ℚ :=
  (t1.silver_paid - t2.silver_paid : ℚ) / (t1.gold_change - t2.gold_change)

/-- Calculates the price of the cloak in gold coins --/
def cloak_price_gold (t : CloakTransaction) (rate : ℚ) : ℚ :=
  t.silver_paid / rate - t.gold_change

/-- Theorem stating the change received when buying a cloak with gold coins --/
theorem cloak_change_in_silver 
  (t1 t2 : CloakTransaction)
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1)
  (gold_paid : ℕ)
  (h3 : gold_paid = 14) :
  ∃ (silver_change : ℕ), silver_change = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_change_in_silver_l516_51699


namespace NUMINAMATH_CALUDE_tangent_line_determines_function_l516_51638

noncomputable def f (a b x : ℝ) : ℝ := a * x / (x^2 + b)

theorem tangent_line_determines_function (a b : ℝ) :
  (∃ x, f a b x = 2 ∧ (deriv (f a b)) x = 0) ∧ f a b 1 = 2 →
  ∀ x, f a b x = 4 * x / (x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_determines_function_l516_51638


namespace NUMINAMATH_CALUDE_inequality_proof_l516_51659

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (x^6 / y^2) + (y^6 / x^2) ≥ x^4 + y^4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l516_51659


namespace NUMINAMATH_CALUDE_particle_position_1989_l516_51637

/-- Represents the position of a particle -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- The theorem stating the particle's position after 1989 minutes -/
theorem particle_position_1989 : particlePosition 1989 = Position.mk 44 35 := by
  sorry

end NUMINAMATH_CALUDE_particle_position_1989_l516_51637


namespace NUMINAMATH_CALUDE_clock_problem_l516_51653

/-- Represents the original cost of the clock to the shop -/
def original_cost : ℝ := 250

/-- Represents the first selling price of the clock -/
def first_sell_price : ℝ := original_cost * 1.2

/-- Represents the buy-back price of the clock -/
def buy_back_price : ℝ := first_sell_price * 0.5

/-- Represents the second selling price of the clock -/
def second_sell_price : ℝ := buy_back_price * 1.8

theorem clock_problem :
  (original_cost - buy_back_price = 100) →
  (second_sell_price = 270) := by
  sorry

end NUMINAMATH_CALUDE_clock_problem_l516_51653


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l516_51677

theorem geese_percentage_among_non_swans 
  (geese_percent : ℝ) 
  (swans_percent : ℝ) 
  (herons_percent : ℝ) 
  (ducks_percent : ℝ) 
  (h1 : geese_percent = 30)
  (h2 : swans_percent = 25)
  (h3 : herons_percent = 10)
  (h4 : ducks_percent = 35)
  (h5 : geese_percent + swans_percent + herons_percent + ducks_percent = 100) :
  (geese_percent / (100 - swans_percent)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l516_51677


namespace NUMINAMATH_CALUDE_tablecloth_width_l516_51655

/-- Given a rectangular tablecloth and napkins with specified dimensions,
    prove that the width of the tablecloth is 54 inches. -/
theorem tablecloth_width
  (tablecloth_length : ℕ)
  (napkin_length napkin_width : ℕ)
  (num_napkins : ℕ)
  (total_area : ℕ)
  (h1 : tablecloth_length = 102)
  (h2 : napkin_length = 6)
  (h3 : napkin_width = 7)
  (h4 : num_napkins = 8)
  (h5 : total_area = 5844) :
  total_area - num_napkins * napkin_length * napkin_width = 54 * tablecloth_length :=
by sorry

end NUMINAMATH_CALUDE_tablecloth_width_l516_51655


namespace NUMINAMATH_CALUDE_expression_equality_l516_51600

theorem expression_equality (x y z : ℝ) : (x + (y + z)) - ((x + z) + y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l516_51600


namespace NUMINAMATH_CALUDE_power_three_times_three_l516_51642

theorem power_three_times_three (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_three_times_three_l516_51642


namespace NUMINAMATH_CALUDE_solve_equation_l516_51603

theorem solve_equation (x : ℝ) : (x - 5) ^ 4 = 16 ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l516_51603


namespace NUMINAMATH_CALUDE_correct_average_weight_l516_51650

/-- Proves that the correct average weight of a class is 61.2 kg given the initial miscalculation and corrections. -/
theorem correct_average_weight 
  (num_students : ℕ) 
  (initial_average : ℝ)
  (student_A_misread student_A_correct : ℝ)
  (student_B_misread student_B_correct : ℝ)
  (student_C_misread student_C_correct : ℝ)
  (h1 : num_students = 30)
  (h2 : initial_average = 60.2)
  (h3 : student_A_misread = 54)
  (h4 : student_A_correct = 64)
  (h5 : student_B_misread = 58)
  (h6 : student_B_correct = 68)
  (h7 : student_C_misread = 50)
  (h8 : student_C_correct = 60) :
  (num_students : ℝ) * initial_average + 
  (student_A_correct - student_A_misread) + 
  (student_B_correct - student_B_misread) + 
  (student_C_correct - student_C_misread) / num_students = 61.2 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l516_51650


namespace NUMINAMATH_CALUDE_oil_leak_total_l516_51664

/-- The total amount of oil leaked from four pipes -/
def total_oil_leaked (pipe_a_before pipe_a_during pipe_b_before pipe_b_during pipe_c_first pipe_c_second pipe_d_first pipe_d_second pipe_d_third : ℕ) : ℕ :=
  pipe_a_before + pipe_a_during + 
  pipe_b_before + pipe_b_during + 
  pipe_c_first + pipe_c_second + 
  pipe_d_first + pipe_d_second + pipe_d_third

/-- Theorem stating the total amount of oil leaked from the four pipes -/
theorem oil_leak_total : 
  total_oil_leaked 6522 5165 4378 3250 2897 7562 1789 3574 5110 = 40247 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_total_l516_51664


namespace NUMINAMATH_CALUDE_data_set_properties_l516_51667

def data_set : List Nat := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def mode (l : List Nat) : Nat := sorry

def range (l : List Nat) : Nat := sorry

def quantile (l : List Nat) (p : Rat) : Rat := sorry

theorem data_set_properties :
  (mode data_set = 31) ∧
  (range data_set = 37) ∧
  (quantile data_set (1/10) = 30.5) := by sorry

end NUMINAMATH_CALUDE_data_set_properties_l516_51667


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l516_51663

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a < b →            -- a is the shorter leg
  a ≤ b →            -- Ensure a is not equal to b
  a = 16 :=          -- The shorter leg is 16 units
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l516_51663


namespace NUMINAMATH_CALUDE_tshirt_pricing_l516_51652

theorem tshirt_pricing (first_batch_cost second_batch_cost : ℕ)
  (quantity_ratio : ℚ) (price_difference : ℕ) (first_batch_selling_price : ℕ)
  (min_total_profit : ℕ) :
  first_batch_cost = 4000 →
  second_batch_cost = 5400 →
  quantity_ratio = 3/2 →
  price_difference = 5 →
  first_batch_selling_price = 70 →
  min_total_profit = 4060 →
  ∃ (first_batch_unit_cost : ℕ) (second_batch_min_selling_price : ℕ),
    first_batch_unit_cost = 50 ∧
    second_batch_min_selling_price = 66 ∧
    (second_batch_cost : ℚ) / (first_batch_unit_cost - price_difference) = quantity_ratio * ((first_batch_cost : ℚ) / first_batch_unit_cost) ∧
    (first_batch_selling_price - first_batch_unit_cost) * (first_batch_cost / first_batch_unit_cost) +
    (second_batch_min_selling_price - (first_batch_unit_cost - price_difference)) * (second_batch_cost / (first_batch_unit_cost - price_difference)) ≥ min_total_profit :=
by sorry

end NUMINAMATH_CALUDE_tshirt_pricing_l516_51652


namespace NUMINAMATH_CALUDE_gerbils_sold_l516_51613

theorem gerbils_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 85 → remaining = 16 → sold = initial - remaining → sold = 69 := by
  sorry

end NUMINAMATH_CALUDE_gerbils_sold_l516_51613


namespace NUMINAMATH_CALUDE_negation_of_proposition_l516_51688

theorem negation_of_proposition (a : ℝ) (h : 0 < a ∧ a < 1) :
  (¬ (∀ x : ℝ, x < 0 → a^x > 1)) ↔ (∃ x₀ : ℝ, x₀ < 0 ∧ a^x₀ ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l516_51688


namespace NUMINAMATH_CALUDE_circleplus_composition_l516_51616

-- Define the ⊕ operation
def circleplus (y : Int) : Int := 9 - y

-- Define the prefix ⊕ operation
def prefix_circleplus (y : Int) : Int := y - 9

-- Theorem to prove
theorem circleplus_composition : prefix_circleplus (circleplus 18) = -18 := by
  sorry

end NUMINAMATH_CALUDE_circleplus_composition_l516_51616


namespace NUMINAMATH_CALUDE_abc_inequality_l516_51611

theorem abc_inequality : 
  let a : ℝ := (3/4)^(2/3)
  let b : ℝ := (2/3)^(3/4)
  let c : ℝ := Real.log (4/3) / Real.log (2/3)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l516_51611


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l516_51629

/-- Given a quadratic function f(x) = ax² + bx with certain constraints on f(-1) and f(1),
    prove that f(-2) is bounded between 6 and 10. -/
theorem quadratic_function_bounds (a b : ℝ) :
  let f := fun (x : ℝ) => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) →
  (3 ≤ f 1 ∧ f 1 ≤ 4) →
  (6 ≤ f (-2) ∧ f (-2) ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l516_51629


namespace NUMINAMATH_CALUDE_tournament_prize_orders_l516_51614

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of elimination rounds in the tournament -/
def num_rounds : ℕ := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Theorem stating the number of possible prize orders in the tournament -/
theorem tournament_prize_orders :
  (outcomes_per_match ^ num_rounds : ℕ) = 32 := by sorry

end NUMINAMATH_CALUDE_tournament_prize_orders_l516_51614


namespace NUMINAMATH_CALUDE_symmetry_implies_f_3_equals_1_l516_51670

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_about_y_equals_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f (x - 1) = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_f_3_equals_1
  (h_sym : symmetric_about_y_equals_x f g)
  (h_g : g 1 = 2) :
  f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_f_3_equals_1_l516_51670


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l516_51654

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 4 = 6 → a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l516_51654


namespace NUMINAMATH_CALUDE_smallest_nonnegative_solution_congruence_l516_51602

theorem smallest_nonnegative_solution_congruence :
  ∃ (x : ℕ), x < 15 ∧ (7 * x + 3) % 15 = 6 % 15 ∧
  ∀ (y : ℕ), y < x → (7 * y + 3) % 15 ≠ 6 % 15 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_solution_congruence_l516_51602


namespace NUMINAMATH_CALUDE_least_possible_z_l516_51687

theorem least_possible_z (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  (∃ m n : ℤ, y = 2 * m + 1 ∧ z = 2 * n + 1) →  -- y and z are odd
  y - x > 5 →
  (∀ w : ℤ, w - x ≥ 9 → z ≤ w) →  -- least possible value of z - x is 9
  z ≥ 11 ∧ (∀ v : ℤ, v ≥ 11 → z ≤ v) :=  -- z is at least 11 and is the least such value
by sorry

end NUMINAMATH_CALUDE_least_possible_z_l516_51687


namespace NUMINAMATH_CALUDE_fraction_difference_simplification_l516_51673

theorem fraction_difference_simplification :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_simplification_l516_51673


namespace NUMINAMATH_CALUDE_C_share_approx_l516_51645

-- Define the total rent
def total_rent : ℚ := 225

-- Define the number of oxen and months for each person
def oxen_A : ℕ := 10
def months_A : ℕ := 7
def oxen_B : ℕ := 12
def months_B : ℕ := 5
def oxen_C : ℕ := 15
def months_C : ℕ := 3
def oxen_D : ℕ := 20
def months_D : ℕ := 6

-- Calculate oxen-months for each person
def oxen_months_A : ℕ := oxen_A * months_A
def oxen_months_B : ℕ := oxen_B * months_B
def oxen_months_C : ℕ := oxen_C * months_C
def oxen_months_D : ℕ := oxen_D * months_D

-- Calculate total oxen-months
def total_oxen_months : ℕ := oxen_months_A + oxen_months_B + oxen_months_C + oxen_months_D

-- Calculate C's share of the rent
def C_share : ℚ := total_rent * (oxen_months_C : ℚ) / (total_oxen_months : ℚ)

-- Theorem to prove
theorem C_share_approx : ∃ ε > 0, abs (C_share - 34.32) < ε :=
sorry

end NUMINAMATH_CALUDE_C_share_approx_l516_51645


namespace NUMINAMATH_CALUDE_secret_spread_exceeds_3000_l516_51631

def secret_spread (n : ℕ) : ℕ := 3^(n-1)

theorem secret_spread_exceeds_3000 :
  ∃ (n : ℕ), n = 9 ∧ secret_spread n > 3000 :=
by sorry

end NUMINAMATH_CALUDE_secret_spread_exceeds_3000_l516_51631


namespace NUMINAMATH_CALUDE_rearrangement_impossibility_l516_51678

theorem rearrangement_impossibility : ¬ ∃ (arrangement : Fin 3972 → ℕ),
  (∀ i : Fin 1986, ∃ (m n : Fin 3972), m < n ∧ 
    arrangement m = i.val + 1 ∧ 
    arrangement n = i.val + 1 ∧ 
    n.val - m.val - 1 = i.val) ∧
  (∀ k : Fin 3972, ∃ i : Fin 1986, arrangement k = i.val + 1) :=
sorry

end NUMINAMATH_CALUDE_rearrangement_impossibility_l516_51678


namespace NUMINAMATH_CALUDE_parity_of_linear_system_solution_l516_51619

theorem parity_of_linear_system_solution (n m : ℤ) 
  (h_n_odd : Odd n) (h_m_odd : Odd m) :
  ∃ (x y : ℤ), x + 2*y = n ∧ 3*x - y = m → Odd x ∧ Even y := by
  sorry

end NUMINAMATH_CALUDE_parity_of_linear_system_solution_l516_51619


namespace NUMINAMATH_CALUDE_wire_parts_used_l516_51665

theorem wire_parts_used (total_length : ℝ) (total_parts : ℕ) (unused_length : ℝ) : 
  total_length = 50 →
  total_parts = 5 →
  unused_length = 20 →
  (total_parts : ℝ) - (unused_length / (total_length / total_parts)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_wire_parts_used_l516_51665


namespace NUMINAMATH_CALUDE_closest_point_parabola_to_line_l516_51656

/-- The point on the parabola y = x^2 that is closest to the line 2x - y = 4 is (1, 1) -/
theorem closest_point_parabola_to_line :
  let parabola := λ x : ℝ => (x, x^2)
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance := λ p : ℝ × ℝ => |2 * p.1 - p.2 - 4| / Real.sqrt 5
  ∀ x : ℝ, distance (parabola x) ≥ distance (parabola 1) :=
by sorry

end NUMINAMATH_CALUDE_closest_point_parabola_to_line_l516_51656


namespace NUMINAMATH_CALUDE_fuel_purchase_calculation_l516_51675

/-- Given the cost of fuel per gallon, the fuel consumption rate per hour,
    and the total time to consume all fuel, calculate the number of gallons purchased. -/
theorem fuel_purchase_calculation 
  (cost_per_gallon : ℝ) 
  (consumption_rate_per_hour : ℝ) 
  (total_hours : ℝ) 
  (h1 : cost_per_gallon = 0.70)
  (h2 : consumption_rate_per_hour = 0.40)
  (h3 : total_hours = 175) :
  (consumption_rate_per_hour * total_hours) / cost_per_gallon = 100 := by
  sorry

end NUMINAMATH_CALUDE_fuel_purchase_calculation_l516_51675


namespace NUMINAMATH_CALUDE_intersection_equals_two_to_infinity_l516_51690

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log ((1 - x) / x)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 3}

-- Define the complement of M in ℝ
def M_complement : Set ℝ := {x | x ∉ M}

-- Define the set [2, +∞)
def two_to_infinity : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem intersection_equals_two_to_infinity : (M_complement ∩ N) = two_to_infinity := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_two_to_infinity_l516_51690


namespace NUMINAMATH_CALUDE_triangle_inequality_sign_l516_51634

/-- Given a triangle ABC with sides a, b, c (a ≤ b ≤ c), circumradius R, and inradius r,
    the sign of a + b - 2R - 2r depends on angle C as follows:
    1. If π/3 ≤ C < π/2, then a + b - 2R - 2r > 0
    2. If C = π/2, then a + b - 2R - 2r = 0
    3. If π/2 < C < π, then a + b - 2R - 2r < 0 -/
theorem triangle_inequality_sign (a b c R r : ℝ) (C : ℝ) :
  a ≤ b ∧ b ≤ c ∧ 0 < a ∧ 0 < R ∧ 0 < r ∧ 0 < C ∧ C < π →
  (π/3 ≤ C ∧ C < π/2 → a + b - 2*R - 2*r > 0) ∧
  (C = π/2 → a + b - 2*R - 2*r = 0) ∧
  (π/2 < C ∧ C < π → a + b - 2*R - 2*r < 0) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_sign_l516_51634
