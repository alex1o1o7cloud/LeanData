import Mathlib

namespace NUMINAMATH_CALUDE_water_in_bucket_l2826_282607

theorem water_in_bucket (initial_water : ℝ) (poured_out : ℝ) (remaining_water : ℝ) :
  initial_water = 0.8 →
  poured_out = 0.2 →
  remaining_water = initial_water - poured_out →
  remaining_water = 0.6 := by
sorry

end NUMINAMATH_CALUDE_water_in_bucket_l2826_282607


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2826_282677

theorem matrix_equation_solution :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  N ^ 2 - 3 • N + 2 • N = !![6, 12; 3, 6] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2826_282677


namespace NUMINAMATH_CALUDE_john_piggy_bank_balance_l2826_282619

/-- The amount John saves monthly in dollars -/
def monthly_savings : ℕ := 25

/-- The number of months John saves -/
def saving_period : ℕ := 2 * 12

/-- The amount John spends on car repairs in dollars -/
def car_repair_cost : ℕ := 400

/-- The amount left in John's piggy bank after savings and car repair -/
def piggy_bank_balance : ℕ := monthly_savings * saving_period - car_repair_cost

theorem john_piggy_bank_balance : piggy_bank_balance = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_piggy_bank_balance_l2826_282619


namespace NUMINAMATH_CALUDE_at_most_one_root_l2826_282609

theorem at_most_one_root {f : ℝ → ℝ} (h : ∀ a b, a < b → f a < f b) :
  ∃! x, f x = 0 ∨ ∀ x, f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_root_l2826_282609


namespace NUMINAMATH_CALUDE_jakes_weight_l2826_282605

theorem jakes_weight (j k : ℝ) 
  (h1 : j - 8 = 2 * k)  -- If Jake loses 8 pounds, he will weigh twice as much as Kendra
  (h2 : j + k = 290)    -- Together they now weigh 290 pounds
  : j = 196 :=          -- Jake's present weight is 196 pounds
by sorry

end NUMINAMATH_CALUDE_jakes_weight_l2826_282605


namespace NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l2826_282691

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 2 x ≥ 1 - 2*x} = {x : ℝ | x ≥ -1} := by sorry

-- Part 2
theorem min_value_part2 (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) 
  (h4 : m^2 * n = a) (h5 : ∀ x, f a x + |x - 1| ≥ 3) :
  ∃ (x : ℝ), m + n ≥ x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l2826_282691


namespace NUMINAMATH_CALUDE_seating_arrangement_one_between_AB_seating_arrangement_no_adjacent_empty_l2826_282682

/-- Number of students -/
def num_students : ℕ := 4

/-- Number of seats in the row -/
def num_seats : ℕ := 6

/-- Number of seating arrangements with exactly one person between A and B and no empty seats between them -/
def arrangements_with_one_between_AB : ℕ := 48

/-- Number of seating arrangements where all empty seats are not adjacent -/
def arrangements_no_adjacent_empty : ℕ := 240

/-- Theorem for the first question -/
theorem seating_arrangement_one_between_AB :
  (num_students = 4) → (num_seats = 6) →
  (arrangements_with_one_between_AB = 48) := by sorry

/-- Theorem for the second question -/
theorem seating_arrangement_no_adjacent_empty :
  (num_students = 4) → (num_seats = 6) →
  (arrangements_no_adjacent_empty = 240) := by sorry

end NUMINAMATH_CALUDE_seating_arrangement_one_between_AB_seating_arrangement_no_adjacent_empty_l2826_282682


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l2826_282668

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l2826_282668


namespace NUMINAMATH_CALUDE_min_faces_prism_min_vertices_pyramid_l2826_282647

/-- A prism is a three-dimensional shape with two identical ends and flat sides. -/
structure Prism where
  base : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  height : ℝ

/-- A pyramid is a three-dimensional shape with a polygonal base and triangular faces meeting at a point. -/
structure Pyramid where
  base : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  apex : ℝ × ℝ × ℝ    -- The apex point in 3D

/-- The number of faces in a prism. -/
def num_faces_prism (p : Prism) : ℕ := sorry

/-- The number of vertices in a pyramid. -/
def num_vertices_pyramid (p : Pyramid) : ℕ := sorry

/-- The minimum number of faces in any prism is 5. -/
theorem min_faces_prism : ∀ p : Prism, num_faces_prism p ≥ 5 := sorry

/-- The number of vertices in a pyramid with the minimum number of faces is 4. -/
theorem min_vertices_pyramid : ∃ p : Pyramid, num_vertices_pyramid p = 4 ∧ 
  (∀ q : Pyramid, num_vertices_pyramid q ≥ num_vertices_pyramid p) := sorry

end NUMINAMATH_CALUDE_min_faces_prism_min_vertices_pyramid_l2826_282647


namespace NUMINAMATH_CALUDE_race_car_time_problem_l2826_282633

theorem race_car_time_problem (time_A time_sync : ℕ) (time_B : ℕ) : 
  time_A = 28 →
  time_sync = 168 →
  time_sync % time_A = 0 →
  time_sync % time_B = 0 →
  time_B > time_A →
  time_B < time_sync →
  (time_sync / time_A) % (time_sync / time_B) = 0 →
  time_B = 42 :=
by sorry

end NUMINAMATH_CALUDE_race_car_time_problem_l2826_282633


namespace NUMINAMATH_CALUDE_min_real_part_x_l2826_282630

theorem min_real_part_x (x y : ℂ) 
  (eq1 : x + 2 * y^2 = x^4)
  (eq2 : y + 2 * x^2 = y^4) :
  Real.sqrt (Real.sqrt ((1 - Real.sqrt 33) / 2)) ≤ x.re :=
sorry

end NUMINAMATH_CALUDE_min_real_part_x_l2826_282630


namespace NUMINAMATH_CALUDE_one_real_root_condition_l2826_282697

/-- Given the equation lg(kx) = 2lg(x+1), this theorem states the condition for k
    such that the equation has only one real root. -/
theorem one_real_root_condition (k : ℝ) : 
  (∃! x : ℝ, Real.log (k * x) = 2 * Real.log (x + 1)) ↔ (k < 0 ∨ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_one_real_root_condition_l2826_282697


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2826_282611

theorem arithmetic_geometric_sequence_product (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, a n ≠ 0) →  -- a_n is non-zero for all n
  (a 3 - (a 7)^2 / 2 + a 11 = 0) →  -- given condition
  (∃ r, ∀ n, b (n + 1) = r * b n) →  -- b is a geometric sequence
  (b 7 = a 7) →  -- given condition
  (b 1 * b 13 = 16) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2826_282611


namespace NUMINAMATH_CALUDE_probability_even_and_prime_on_two_dice_l2826_282670

/-- A die is a finite set of natural numbers from 1 to 6 -/
def Die : Finset ℕ := Finset.range 6

/-- Even numbers on a die -/
def EvenNumbers : Finset ℕ := {2, 4, 6}

/-- Prime numbers on a die -/
def PrimeNumbers : Finset ℕ := {2, 3, 5}

/-- The probability of an event occurring in a finite sample space -/
def probability (event : Finset ℕ) (sampleSpace : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_even_and_prime_on_two_dice : 
  (probability EvenNumbers Die) * (probability PrimeNumbers Die) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_and_prime_on_two_dice_l2826_282670


namespace NUMINAMATH_CALUDE_vector_problem_l2826_282610

def a : Fin 2 → ℝ := ![2, 4]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

theorem vector_problem (x : ℝ) :
  (∀ (i : Fin 2), (a i * b x i) > 0) →
  (x > -2 ∧ x ≠ 1/2) ∧
  ((∀ (i : Fin 2), ((2 * a i - b x i) * a i) = 0) →
   Real.sqrt ((a 0 + b x 0)^2 + (a 1 + b x 1)^2) = 5 * Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l2826_282610


namespace NUMINAMATH_CALUDE_increasing_sequence_range_l2826_282699

def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 7 then (4 - a) * n - 10 else a^(n - 6)

theorem increasing_sequence_range (a : ℝ) :
  (∀ n m : ℕ, n < m → a_n a n < a_n a m) →
  2 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_range_l2826_282699


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l2826_282644

/-- Represents a collection of stamps -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreign_and_old : ℕ
  neither : ℕ

/-- Calculates the number of foreign stamps in a collection -/
def foreign_stamps (c : StampCollection) : ℕ :=
  c.total - c.old - c.neither + c.foreign_and_old

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (c : StampCollection) 
  (h1 : c.total = 200)
  (h2 : c.old = 50)
  (h3 : c.foreign_and_old = 20)
  (h4 : c.neither = 80) :
  foreign_stamps c = 90 := by
  sorry

#eval foreign_stamps { total := 200, old := 50, foreign_and_old := 20, neither := 80 }

end NUMINAMATH_CALUDE_foreign_stamps_count_l2826_282644


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2826_282604

/-- If x^2 - kx + 25 is a perfect square polynomial, then k = ±10 -/
theorem perfect_square_polynomial (k : ℝ) : 
  (∃ (a : ℝ), ∀ x, x^2 - k*x + 25 = (x - a)^2) → (k = 10 ∨ k = -10) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2826_282604


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l2826_282646

theorem probability_four_ones_in_five_rolls :
  let n_rolls : ℕ := 5
  let n_desired : ℕ := 4
  let die_sides : ℕ := 6
  let p_success : ℚ := 1 / die_sides
  let p_failure : ℚ := 1 - p_success
  let combinations : ℕ := Nat.choose n_rolls n_desired
  combinations * p_success ^ n_desired * p_failure ^ (n_rolls - n_desired) = 25 / 7776 :=
by sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l2826_282646


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l2826_282683

/-- Calculates the toll for a truck based on the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels -/
def axles_count (wheels : ℕ) : ℕ :=
  wheels / 2

theorem eighteen_wheel_truck_toll :
  let wheels : ℕ := 18
  let axles : ℕ := axles_count wheels
  toll axles = 5 := by sorry

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l2826_282683


namespace NUMINAMATH_CALUDE_petrol_consumption_reduction_l2826_282635

/-- Theorem: Calculation of required reduction in petrol consumption to maintain constant expenditure --/
theorem petrol_consumption_reduction
  (price_increase_A : ℝ) (price_increase_B : ℝ)
  (maintenance_cost_ratio : ℝ) (maintenance_cost_increase : ℝ)
  (h1 : price_increase_A = 0.20)
  (h2 : price_increase_B = 0.15)
  (h3 : maintenance_cost_ratio = 0.30)
  (h4 : maintenance_cost_increase = 0.10) :
  let avg_price_increase := (1 + price_increase_A + 1 + price_increase_B) / 2 - 1
  let total_maintenance_increase := maintenance_cost_ratio * maintenance_cost_increase
  let total_increase := avg_price_increase + total_maintenance_increase
  total_increase = 0.205 := by sorry

end NUMINAMATH_CALUDE_petrol_consumption_reduction_l2826_282635


namespace NUMINAMATH_CALUDE_max_elements_in_S_l2826_282659

theorem max_elements_in_S (A : Finset ℝ) (h_card : A.card = 100) (h_pos : ∀ a ∈ A, a > 0) :
  let S := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 ∈ A}
  (Finset.filter (fun p => p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 ∈ A) (A.product A)).card ≤ 4950 :=
by sorry

end NUMINAMATH_CALUDE_max_elements_in_S_l2826_282659


namespace NUMINAMATH_CALUDE_total_brownies_l2826_282678

/-- The number of brownies Tina ate per day -/
def tina_daily : ℕ := 2

/-- The number of days Tina ate brownies -/
def days : ℕ := 5

/-- The number of brownies Tina's husband ate per day -/
def husband_daily : ℕ := 1

/-- The number of brownies shared with guests -/
def shared : ℕ := 4

/-- The number of brownies left -/
def left : ℕ := 5

/-- Theorem stating the total number of brownie pieces -/
theorem total_brownies : 
  tina_daily * days + husband_daily * days + shared + left = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_brownies_l2826_282678


namespace NUMINAMATH_CALUDE_seven_abba_divisible_by_eleven_l2826_282645

theorem seven_abba_divisible_by_eleven (A : Nat) :
  A < 10 →
  (∃ B : Nat, B < 10 ∧ (70000 + A * 1000 + B * 100 + B * 10 + A) % 11 = 0) ↔
  A = 7 := by
sorry

end NUMINAMATH_CALUDE_seven_abba_divisible_by_eleven_l2826_282645


namespace NUMINAMATH_CALUDE_pizza_varieties_theorem_four_topping_combinations_l2826_282687

/-- Represents the number of base pizza flavors -/
def num_flavors : Nat := 4

/-- Represents the number of topping combinations -/
def num_topping_combinations : Nat := 4

/-- Represents the total number of pizza varieties -/
def total_varieties : Nat := 16

/-- Theorem stating that the number of pizza varieties is the product of 
    the number of flavors and the number of topping combinations -/
theorem pizza_varieties_theorem :
  num_flavors * num_topping_combinations = total_varieties := by
  sorry

/-- Definition of the possible topping combinations -/
inductive ToppingCombination
  | None
  | ExtraCheese
  | Mushrooms
  | ExtraCheeseAndMushrooms

/-- Theorem stating that there are exactly 4 topping combinations -/
theorem four_topping_combinations :
  (ToppingCombination.None :: ToppingCombination.ExtraCheese :: 
   ToppingCombination.Mushrooms :: ToppingCombination.ExtraCheeseAndMushrooms :: []).length = 
  num_topping_combinations := by
  sorry

end NUMINAMATH_CALUDE_pizza_varieties_theorem_four_topping_combinations_l2826_282687


namespace NUMINAMATH_CALUDE_fraction_equality_l2826_282650

theorem fraction_equality : (1/4 - 1/5) / (1/3 - 1/6 + 1/12) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2826_282650


namespace NUMINAMATH_CALUDE_pants_purchase_l2826_282660

theorem pants_purchase (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (total_paid : ℝ) :
  original_price = 45 →
  discount_rate = 0.20 →
  tax_rate = 0.10 →
  total_paid = 396 →
  ∃ (num_pairs : ℕ), 
    (num_pairs : ℝ) * (original_price * (1 - discount_rate) * (1 + tax_rate)) = total_paid ∧
    num_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_pants_purchase_l2826_282660


namespace NUMINAMATH_CALUDE_f_properties_l2826_282618

-- Define the function f(x) = lg|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem f_properties :
  -- f is defined for all real numbers except 0
  (∀ x : ℝ, x ≠ 0 → f x = Real.log (abs x)) →
  -- f is an even function
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  -- f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2826_282618


namespace NUMINAMATH_CALUDE_department_age_analysis_l2826_282616

/-- Represents the age data for a department -/
def DepartmentData := List Nat

/-- Calculate the mode of a list of numbers -/
def mode (data : DepartmentData) : Nat :=
  sorry

/-- Calculate the median of a list of numbers -/
def median (data : DepartmentData) : Nat :=
  sorry

/-- Calculate the average of a list of numbers -/
def average (data : DepartmentData) : Rat :=
  sorry

theorem department_age_analysis 
  (dept_A dept_B : DepartmentData)
  (h1 : dept_A.length = 10)
  (h2 : dept_B.length = 10)
  (h3 : dept_A = [21, 23, 25, 26, 27, 28, 30, 32, 32, 32])
  (h4 : dept_B = [20, 22, 24, 24, 26, 28, 28, 30, 34, 40]) :
  (mode dept_A = 32) ∧ 
  (median dept_B = 26) ∧ 
  (average dept_A < average dept_B) :=
sorry

end NUMINAMATH_CALUDE_department_age_analysis_l2826_282616


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2826_282639

theorem absolute_value_inequality (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 8) ↔ ((-10 ≤ x ∧ x ≤ -5) ∨ (1 ≤ x ∧ x ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2826_282639


namespace NUMINAMATH_CALUDE_profit_at_55_profit_price_relationship_optimal_price_l2826_282685

-- Define the constants and variables
def sales_cost : ℝ := 40
def initial_price : ℝ := 50
def initial_volume : ℝ := 500
def volume_decrease_rate : ℝ := 10

-- Define the sales volume function
def sales_volume (price : ℝ) : ℝ :=
  initial_volume - volume_decrease_rate * (price - initial_price)

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - sales_cost) * sales_volume price

-- Theorem 1: Monthly sales profit at $55 per kilogram
theorem profit_at_55 :
  profit 55 = 6750 := by sorry

-- Theorem 2: Relationship between profit and price
theorem profit_price_relationship (price : ℝ) :
  profit price = -10 * price^2 + 1400 * price - 40000 := by sorry

-- Theorem 3: Optimal price for $8000 profit without exceeding $10000 cost
theorem optimal_price :
  ∃ (price : ℝ),
    profit price = 8000 ∧
    sales_volume price * sales_cost ≤ 10000 ∧
    price = 80 := by sorry

end NUMINAMATH_CALUDE_profit_at_55_profit_price_relationship_optimal_price_l2826_282685


namespace NUMINAMATH_CALUDE_waiter_problem_l2826_282664

/-- The number of customers who left a waiter's table. -/
def customers_left (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem waiter_problem :
  let initial : ℕ := 14
  let remaining : ℕ := 9
  customers_left initial remaining = 5 := by
sorry

end NUMINAMATH_CALUDE_waiter_problem_l2826_282664


namespace NUMINAMATH_CALUDE_probability_triangle_from_random_chords_probability_is_favorable_over_total_total_pairings_calculation_favorable_pairings_is_one_probability_triangle_from_random_chords_value_l2826_282638

/-- The probability of forming a triangle when choosing three random chords on a circle -/
theorem probability_triangle_from_random_chords : ℚ :=
  1 / 15

/-- The number of ways to pair 6 points into three pairs -/
def total_pairings : ℕ := 15

/-- The number of pairings that result in all chords intersecting and forming a triangle -/
def favorable_pairings : ℕ := 1

theorem probability_is_favorable_over_total :
  probability_triangle_from_random_chords = favorable_pairings / total_pairings :=
sorry

theorem total_pairings_calculation :
  total_pairings = (1 / 6 : ℚ) * (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 2 2) :=
sorry

theorem favorable_pairings_is_one :
  favorable_pairings = 1 :=
sorry

theorem probability_triangle_from_random_chords_value :
  probability_triangle_from_random_chords = 1 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_triangle_from_random_chords_probability_is_favorable_over_total_total_pairings_calculation_favorable_pairings_is_one_probability_triangle_from_random_chords_value_l2826_282638


namespace NUMINAMATH_CALUDE_money_left_l2826_282658

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Hugo spent on an online game -/
def online_game_cost : ℕ := 75

/-- The theorem stating how much money Norris has left -/
theorem money_left : 
  september_savings + october_savings + november_savings - online_game_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l2826_282658


namespace NUMINAMATH_CALUDE_min_expression_proof_l2826_282612

theorem min_expression_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) ≥ 3 ∧
  (((x^2 * y * z) / 324 = (144 * y) / (x * z) ∧ (144 * y) / (x * z) = 9 / (4 * x * y^2)) →
    z / (16 * y) + x / 9 ≥ 2) ∧
  ((x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) = 3 ∧
   z / (16 * y) + x / 9 = 2) ↔ (x = 9 ∧ y = 1/2 ∧ z = 16) := by
  sorry

#check min_expression_proof

end NUMINAMATH_CALUDE_min_expression_proof_l2826_282612


namespace NUMINAMATH_CALUDE_star_is_addition_l2826_282608

/-- A binary operation on real numbers satisfying (a ★ b) ★ c = a + b + c -/
def star_op (star : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, star (star a b) c = a + b + c

theorem star_is_addition (star : ℝ → ℝ → ℝ) (h : star_op star) :
  ∀ a b : ℝ, star a b = a + b :=
sorry

end NUMINAMATH_CALUDE_star_is_addition_l2826_282608


namespace NUMINAMATH_CALUDE_permutations_with_non_adjacent_yellow_eq_11760_l2826_282684

/-- The number of permutations of 3 green, 2 red, 2 white, and 3 yellow balls
    where no two yellow balls are adjacent -/
def permutations_with_non_adjacent_yellow : ℕ :=
  let green : ℕ := 3
  let red : ℕ := 2
  let white : ℕ := 2
  let yellow : ℕ := 3
  let non_yellow : ℕ := green + red + white
  let gaps : ℕ := non_yellow + 1
  (Nat.factorial non_yellow / (Nat.factorial green * Nat.factorial red * Nat.factorial white)) *
  (Nat.choose gaps yellow)

theorem permutations_with_non_adjacent_yellow_eq_11760 :
  permutations_with_non_adjacent_yellow = 11760 := by
  sorry

end NUMINAMATH_CALUDE_permutations_with_non_adjacent_yellow_eq_11760_l2826_282684


namespace NUMINAMATH_CALUDE_total_investment_sum_l2826_282636

/-- Proves that the total sum of investments is 6358 given the specified conditions --/
theorem total_investment_sum (raghu_investment : ℝ) 
  (h1 : raghu_investment = 2200)
  (h2 : ∃ trishul_investment : ℝ, trishul_investment = raghu_investment * 0.9)
  (h3 : ∃ vishal_investment : ℝ, vishal_investment = trishul_investment * 1.1) :
  ∃ total_investment : ℝ, total_investment = raghu_investment + trishul_investment + vishal_investment ∧ 
  total_investment = 6358 :=
by sorry

end NUMINAMATH_CALUDE_total_investment_sum_l2826_282636


namespace NUMINAMATH_CALUDE_fries_sold_total_l2826_282603

/-- Represents the number of fries sold -/
structure FriesSold where
  small : ℕ
  large : ℕ

/-- Calculates the total number of fries sold -/
def total_fries (f : FriesSold) : ℕ := f.small + f.large

/-- Theorem: If 4 small fries were sold and the ratio of large to small fries is 5:1, 
    then the total number of fries sold is 24 -/
theorem fries_sold_total (f : FriesSold) 
    (h1 : f.small = 4) 
    (h2 : f.large = 5 * f.small) : 
  total_fries f = 24 := by
  sorry


end NUMINAMATH_CALUDE_fries_sold_total_l2826_282603


namespace NUMINAMATH_CALUDE_unique_integer_product_of_digits_l2826_282688

/-- Given a positive integer n, returns the product of its digits -/
def productOfDigits (n : ℕ+) : ℕ := sorry

/-- Theorem: The only positive integer n whose product of digits equals n^2 - 15n - 27 is 17 -/
theorem unique_integer_product_of_digits : 
  ∃! (n : ℕ+), productOfDigits n = n^2 - 15*n - 27 ∧ n = 17 := by sorry

end NUMINAMATH_CALUDE_unique_integer_product_of_digits_l2826_282688


namespace NUMINAMATH_CALUDE_sum_remainder_eleven_l2826_282627

theorem sum_remainder_eleven (n : ℤ) : ((11 - n) + (n + 5)) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_eleven_l2826_282627


namespace NUMINAMATH_CALUDE_danny_larry_score_difference_l2826_282673

theorem danny_larry_score_difference :
  ∀ (keith larry danny : ℕ),
    keith = 3 →
    larry = 3 * keith →
    danny > larry →
    keith + larry + danny = 26 →
    danny - larry = 5 := by
  sorry

end NUMINAMATH_CALUDE_danny_larry_score_difference_l2826_282673


namespace NUMINAMATH_CALUDE_haley_trees_after_typhoon_l2826_282625

/-- The number of trees Haley has left after a typhoon -/
def trees_left (initial_trees dead_trees : ℕ) : ℕ :=
  initial_trees - dead_trees

/-- Theorem: Haley has 10 trees left after the typhoon -/
theorem haley_trees_after_typhoon :
  trees_left 12 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_after_typhoon_l2826_282625


namespace NUMINAMATH_CALUDE_apples_left_is_ten_l2826_282686

/-- The number of apples left in the cafeteria -/
def apples_left : ℕ := sorry

/-- The initial number of apples -/
def initial_apples : ℕ := 50

/-- The initial number of oranges -/
def initial_oranges : ℕ := 40

/-- The cost of an apple in cents -/
def apple_cost : ℕ := 80

/-- The cost of an orange in cents -/
def orange_cost : ℕ := 50

/-- The total earnings from apples and oranges in cents -/
def total_earnings : ℕ := 4900

/-- The number of oranges left -/
def oranges_left : ℕ := 6

/-- Theorem stating that the number of apples left is 10 -/
theorem apples_left_is_ten :
  apples_left = 10 ∧
  initial_apples * apple_cost - apples_left * apple_cost +
  (initial_oranges - oranges_left) * orange_cost = total_earnings :=
sorry

end NUMINAMATH_CALUDE_apples_left_is_ten_l2826_282686


namespace NUMINAMATH_CALUDE_moles_CH₄_required_l2826_282657

/-- Represents a chemical species in a reaction --/
inductive Species
| CH₄ : Species
| Cl₂ : Species
| CHCl₃ : Species
| HCl : Species

/-- Represents the stoichiometric coefficients in a chemical reaction --/
def reaction_coefficients : Species → ℚ
| Species.CH₄ => -1
| Species.Cl₂ => -3
| Species.CHCl₃ => 1
| Species.HCl => 3

/-- The number of moles of CHCl₃ formed --/
def moles_CHCl₃_formed : ℚ := 3

/-- Theorem stating that the number of moles of CH₄ required to form 3 moles of CHCl₃ is 3 moles --/
theorem moles_CH₄_required :
  -reaction_coefficients Species.CH₄ * moles_CHCl₃_formed = 3 := by sorry

end NUMINAMATH_CALUDE_moles_CH₄_required_l2826_282657


namespace NUMINAMATH_CALUDE_large_birdhouses_sold_l2826_282648

/-- Represents the number of large birdhouses sold -/
def large_birdhouses : ℕ := sorry

/-- The price of a large birdhouse in dollars -/
def large_price : ℕ := 22

/-- The price of a medium birdhouse in dollars -/
def medium_price : ℕ := 16

/-- The price of a small birdhouse in dollars -/
def small_price : ℕ := 7

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total sales in dollars -/
def total_sales : ℕ := 97

/-- Theorem stating that the number of large birdhouses sold is 2 -/
theorem large_birdhouses_sold : large_birdhouses = 2 := by
  sorry

end NUMINAMATH_CALUDE_large_birdhouses_sold_l2826_282648


namespace NUMINAMATH_CALUDE_train_crossing_lamppost_l2826_282663

/-- Calculates the time for a train to cross a lamp post given its length, bridge length, and time to cross the bridge -/
theorem train_crossing_lamppost 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (bridge_crossing_time : ℝ) 
  (h1 : train_length = 400)
  (h2 : bridge_length = 800)
  (h3 : bridge_crossing_time = 45)
  : (train_length * bridge_crossing_time) / bridge_length = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_lamppost_l2826_282663


namespace NUMINAMATH_CALUDE_gnome_distribution_ways_l2826_282653

/-- The number of ways to distribute n identical objects among k recipients,
    with each recipient receiving at least m objects. -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * (m - 1) + k - 1) (k - 1)

/-- The number of gnomes -/
def num_gnomes : ℕ := 3

/-- The total number of stones -/
def total_stones : ℕ := 70

/-- The minimum number of stones each gnome must receive -/
def min_stones : ℕ := 10

theorem gnome_distribution_ways : 
  distribution_ways total_stones num_gnomes min_stones = 946 := by
  sorry

end NUMINAMATH_CALUDE_gnome_distribution_ways_l2826_282653


namespace NUMINAMATH_CALUDE_unique_ages_l2826_282621

def is_valid_ages (f : ℤ → ℤ) (b c : ℤ) : Prop :=
  (∀ x y : ℤ, x - y ∣ f x - f y) ∧
  f 7 = 77 ∧
  f b = 85 ∧
  f c = 0 ∧
  7 < b ∧
  b < c

theorem unique_ages :
  ∀ f b c, is_valid_ages f b c → b = 9 ∧ c = 14 := by
sorry

end NUMINAMATH_CALUDE_unique_ages_l2826_282621


namespace NUMINAMATH_CALUDE_amandas_family_painting_l2826_282672

/-- The number of walls each person should paint in Amanda's family house painting problem -/
theorem amandas_family_painting (
  total_rooms : ℕ)
  (rooms_with_four_walls : ℕ)
  (rooms_with_five_walls : ℕ)
  (family_size : ℕ)
  (h1 : total_rooms = rooms_with_four_walls + rooms_with_five_walls)
  (h2 : total_rooms = 9)
  (h3 : rooms_with_four_walls = 5)
  (h4 : rooms_with_five_walls = 4)
  (h5 : family_size = 5)
  : (4 * rooms_with_four_walls + 5 * rooms_with_five_walls) / family_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_amandas_family_painting_l2826_282672


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l2826_282642

/-- The surface area of a sphere that circumscribes a rectangular solid -/
theorem sphere_surface_area_from_rectangular_solid 
  (length width height : ℝ) 
  (h_length : length = 4) 
  (h_width : width = 3) 
  (h_height : height = 2) : 
  ∃ (radius : ℝ), 4 * Real.pi * radius^2 = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l2826_282642


namespace NUMINAMATH_CALUDE_bracket_mult_equation_solution_l2826_282651

-- Define the operation
def bracket_mult (a b c d : ℝ) : ℝ := a * c - b * d

-- State the theorem
theorem bracket_mult_equation_solution :
  ∃ (x : ℝ), (bracket_mult (-x) 3 (x - 2) (-6) = 10) ∧ (x = 4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_bracket_mult_equation_solution_l2826_282651


namespace NUMINAMATH_CALUDE_matching_times_correct_l2826_282676

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Calculates the total minutes elapsed since 00:00 -/
def totalMinutes (t : Time) : Nat :=
  t.hours * 60 + t.minutes

/-- Calculates the charge of the mortar at a given time -/
def charge (t : Time) : Nat :=
  100 - (totalMinutes t) / 6

/-- The list of times when the charge equals the number of minutes -/
def matchingTimes : List Time := [
  ⟨4, 52, by sorry, by sorry⟩,
  ⟨5, 43, by sorry, by sorry⟩,
  ⟨6, 35, by sorry, by sorry⟩,
  ⟨7, 26, by sorry, by sorry⟩,
  ⟨9, 9, by sorry, by sorry⟩
]

/-- Theorem stating that the matching times are correct -/
theorem matching_times_correct :
  ∀ t ∈ matchingTimes, charge t = t.minutes :=
by sorry

end NUMINAMATH_CALUDE_matching_times_correct_l2826_282676


namespace NUMINAMATH_CALUDE_change_in_average_weight_l2826_282623

/-- The change in average weight when replacing a person in a group -/
theorem change_in_average_weight 
  (n : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : n = 6) 
  (h2 : old_weight = 69) 
  (h3 : new_weight = 79.8) : 
  (new_weight - old_weight) / n = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_change_in_average_weight_l2826_282623


namespace NUMINAMATH_CALUDE_bookshop_inventory_l2826_282661

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 900

/-- The number of books sold on Monday -/
def monday_sales : ℕ := 75

/-- The number of books sold on Tuesday -/
def tuesday_sales : ℕ := 50

/-- The number of books sold on Wednesday -/
def wednesday_sales : ℕ := 64

/-- The number of books sold on Thursday -/
def thursday_sales : ℕ := 78

/-- The number of books sold on Friday -/
def friday_sales : ℕ := 135

/-- The percentage of books that were not sold -/
def unsold_percentage : ℚ := 55333333333333336 / 100000000000000000

theorem bookshop_inventory :
  initial_books = 900 ∧
  (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales : ℚ) / initial_books = 1 - unsold_percentage :=
by sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l2826_282661


namespace NUMINAMATH_CALUDE_score_for_175_enemies_l2826_282680

def points_per_enemy : ℕ := 10

def bonus_percentage (enemies_killed : ℕ) : ℚ :=
  if enemies_killed ≥ 200 then 1
  else if enemies_killed ≥ 150 then 3/4
  else if enemies_killed ≥ 100 then 1/2
  else 0

def calculate_score (enemies_killed : ℕ) : ℕ :=
  let base_score := enemies_killed * points_per_enemy
  let bonus := (base_score : ℚ) * bonus_percentage enemies_killed
  ⌊(base_score : ℚ) + bonus⌋₊

theorem score_for_175_enemies :
  calculate_score 175 = 3063 := by sorry

end NUMINAMATH_CALUDE_score_for_175_enemies_l2826_282680


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l2826_282624

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 5x + 2 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 2 = 0

theorem discriminant_of_specific_quadratic :
  discriminant 1 (-5) 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l2826_282624


namespace NUMINAMATH_CALUDE_fourth_side_distance_l2826_282693

/-- A square with a point inside it -/
structure SquareWithPoint where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  d3 : ℝ
  d4 : ℝ
  h_positive : 0 < side_length
  h_inside : d1 + d2 + d3 + d4 = side_length
  h_d1 : 0 < d1
  h_d2 : 0 < d2
  h_d3 : 0 < d3
  h_d4 : 0 < d4

/-- The theorem stating the possible distances to the fourth side -/
theorem fourth_side_distance (s : SquareWithPoint) 
  (h1 : s.d1 = 4)
  (h2 : s.d2 = 7)
  (h3 : s.d3 = 13) :
  s.d4 = 10 ∨ s.d4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_distance_l2826_282693


namespace NUMINAMATH_CALUDE_BaCl2_mass_produced_l2826_282690

-- Define the molar masses
def molar_mass_BaCl2 : ℝ := 208.23

-- Define the initial amounts
def initial_BaCl2_moles : ℝ := 8
def initial_NaOH_moles : ℝ := 12

-- Define the stoichiometric ratios
def ratio_NaOH_to_BaCl2 : ℝ := 2
def ratio_BaOH2_to_BaCl2 : ℝ := 1

-- Define the theorem
theorem BaCl2_mass_produced : 
  let BaCl2_produced := min initial_BaCl2_moles (initial_NaOH_moles / ratio_NaOH_to_BaCl2)
  BaCl2_produced * molar_mass_BaCl2 = 1665.84 :=
by sorry

end NUMINAMATH_CALUDE_BaCl2_mass_produced_l2826_282690


namespace NUMINAMATH_CALUDE_root_in_interval_l2826_282662

def f (x : ℝ) := x^3 + x - 4

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 1 2 ∧ f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2826_282662


namespace NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l2826_282669

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_fourth_fifth_sixth (seq : ArithmeticSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l2826_282669


namespace NUMINAMATH_CALUDE_subtracted_number_l2826_282666

theorem subtracted_number (x : ℚ) : x = 40 → ∃ y : ℚ, ((x / 4) * 5 + 10) - y = 48 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2826_282666


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l2826_282667

/-- Given an examination where 260 students failed out of 400 total students,
    prove that 35% of students passed the examination. -/
theorem exam_pass_percentage :
  let total_students : ℕ := 400
  let failed_students : ℕ := 260
  let passed_students : ℕ := total_students - failed_students
  let pass_percentage : ℚ := (passed_students : ℚ) / (total_students : ℚ) * 100
  pass_percentage = 35 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l2826_282667


namespace NUMINAMATH_CALUDE_max_value_constraint_l2826_282695

theorem max_value_constraint (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) :
  ∃ (M : ℝ), M = (2 * Real.sqrt 21) / 7 ∧ 3*x + y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2826_282695


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_l2826_282654

/-- Given an ellipse ax² + by² = 1 intersecting the line y = 1 - x, 
    if a line through the origin and the midpoint of the intersection points 
    has slope √3/2, then a/b = √3/2 -/
theorem ellipse_intersection_slope (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  (∃ x₁ x₂ : ℝ, 
    a * x₁^2 + b * (1 - x₁)^2 = 1 ∧ 
    a * x₂^2 + b * (1 - x₂)^2 = 1 ∧
    x₁ ≠ x₂ ∧
    (a / (a + b)) / (b / (a + b)) = Real.sqrt 3 / 2) →
  a / b = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_l2826_282654


namespace NUMINAMATH_CALUDE_total_fish_in_lake_l2826_282601

/-- The number of fish per white duck -/
def fishPerWhiteDuck : ℕ := 5

/-- The number of fish per black duck -/
def fishPerBlackDuck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fishPerMulticolorDuck : ℕ := 12

/-- The number of white ducks -/
def numWhiteDucks : ℕ := 3

/-- The number of black ducks -/
def numBlackDucks : ℕ := 7

/-- The number of multicolor ducks -/
def numMulticolorDucks : ℕ := 6

/-- The total number of fish in the lake -/
def totalFish : ℕ := fishPerWhiteDuck * numWhiteDucks + 
                     fishPerBlackDuck * numBlackDucks + 
                     fishPerMulticolorDuck * numMulticolorDucks

theorem total_fish_in_lake : totalFish = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_in_lake_l2826_282601


namespace NUMINAMATH_CALUDE_congruence_problem_l2826_282602

theorem congruence_problem (x : ℤ) :
  (4 * x + 9) % 17 = 3 → (3 * x + 12) % 17 = 16 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2826_282602


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2826_282620

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2826_282620


namespace NUMINAMATH_CALUDE_students_taking_both_courses_l2826_282606

theorem students_taking_both_courses 
  (total : ℕ) 
  (french : ℕ) 
  (german : ℕ) 
  (neither : ℕ) 
  (h1 : total = 69) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : neither = 15) :
  french + german - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_courses_l2826_282606


namespace NUMINAMATH_CALUDE_goose_eggs_laid_l2826_282696

theorem goose_eggs_laid (
  hatch_rate : ℚ)
  (first_month_survival : ℚ)
  (first_six_months_death : ℚ)
  (first_year_death : ℚ)
  (survived_first_year : ℕ)
  (h1 : hatch_rate = 3 / 7)
  (h2 : first_month_survival = 5 / 9)
  (h3 : first_six_months_death = 11 / 16)
  (h4 : first_year_death = 7 / 12)
  (h5 : survived_first_year = 84) :
  ∃ (eggs : ℕ), eggs ≥ 678 ∧
    (eggs : ℚ) * hatch_rate * first_month_survival * (1 - first_six_months_death) * (1 - first_year_death) = survived_first_year :=
by sorry

end NUMINAMATH_CALUDE_goose_eggs_laid_l2826_282696


namespace NUMINAMATH_CALUDE_cindy_envelopes_left_l2826_282617

def envelopes_left (initial_envelopes : ℕ) (num_friends : ℕ) (envelopes_per_friend : ℕ) : ℕ :=
  initial_envelopes - (num_friends * envelopes_per_friend)

theorem cindy_envelopes_left :
  let initial_envelopes : ℕ := 74
  let num_friends : ℕ := 11
  let envelopes_per_friend : ℕ := 6
  envelopes_left initial_envelopes num_friends envelopes_per_friend = 8 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_left_l2826_282617


namespace NUMINAMATH_CALUDE_number_division_l2826_282681

theorem number_division (x : ℚ) : x / 3 = 27 → x / 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l2826_282681


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l2826_282665

theorem company_picnic_attendance
  (total_employees : ℕ)
  (men_percentage : ℚ)
  (women_percentage : ℚ)
  (men_attendance_rate : ℚ)
  (women_attendance_rate : ℚ)
  (h1 : men_percentage = 1/2)
  (h2 : women_percentage = 1 - men_percentage)
  (h3 : men_attendance_rate = 1/5)
  (h4 : women_attendance_rate = 2/5) :
  (men_percentage * men_attendance_rate + women_percentage * women_attendance_rate) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l2826_282665


namespace NUMINAMATH_CALUDE_solution_range_l2826_282615

theorem solution_range (b : ℝ) : 
  (∀ x : ℝ, x = -2 → x^2 - b*x - 5 = 5) ∧
  (∀ x : ℝ, x = -1 → x^2 - b*x - 5 = -1) ∧
  (∀ x : ℝ, x = 4 → x^2 - b*x - 5 = -1) ∧
  (∀ x : ℝ, x = 5 → x^2 - b*x - 5 = 5) →
  ∀ x : ℝ, x^2 - b*x - 5 = 0 ↔ (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l2826_282615


namespace NUMINAMATH_CALUDE_A_and_B_complementary_l2826_282637

-- Define the sample space for a die toss
def DieOutcome := Fin 6

-- Define events A, B, and C
def eventA (outcome : DieOutcome) : Prop := outcome.val ≤ 3
def eventB (outcome : DieOutcome) : Prop := outcome.val ≥ 4
def eventC (outcome : DieOutcome) : Prop := outcome.val % 2 = 1

-- Theorem stating that A and B are complementary events
theorem A_and_B_complementary :
  ∀ (outcome : DieOutcome), eventA outcome ↔ ¬ eventB outcome :=
by sorry

end NUMINAMATH_CALUDE_A_and_B_complementary_l2826_282637


namespace NUMINAMATH_CALUDE_cube_split_theorem_l2826_282694

/-- Given a natural number m > 1, returns the first odd number in the split of m³ -/
def firstSplitNumber (m : ℕ) : ℕ := m * (m - 1) + 1

/-- Given a natural number m > 1, returns the list of odd numbers in the split of m³ -/
def splitNumbers (m : ℕ) : List ℕ :=
  List.range m |>.map (λ i => firstSplitNumber m + 2 * i)

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) (h2 : 333 ∈ splitNumbers m) : m = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_theorem_l2826_282694


namespace NUMINAMATH_CALUDE_four_students_two_groups_l2826_282655

/-- The number of different ways to assign n students to 2 groups -/
def signUpMethods (n : ℕ) : ℕ := 2^n

/-- The problem statement -/
theorem four_students_two_groups : 
  signUpMethods 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_students_two_groups_l2826_282655


namespace NUMINAMATH_CALUDE_nathan_ate_100_gumballs_l2826_282631

/-- The number of gumballs in each package -/
def gumballs_per_package : ℝ := 5.0

/-- The number of packages Nathan ate -/
def packages_eaten : ℝ := 20.0

/-- The total number of gumballs Nathan ate -/
def total_gumballs : ℝ := gumballs_per_package * packages_eaten

theorem nathan_ate_100_gumballs : total_gumballs = 100.0 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_100_gumballs_l2826_282631


namespace NUMINAMATH_CALUDE_beadshop_profit_l2826_282679

theorem beadshop_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ)
  (h_total : total_profit = 1200)
  (h_monday : monday_fraction = 1/3)
  (h_tuesday : tuesday_fraction = 1/4) :
  total_profit - (monday_fraction * total_profit + tuesday_fraction * total_profit) = 500 := by
  sorry

end NUMINAMATH_CALUDE_beadshop_profit_l2826_282679


namespace NUMINAMATH_CALUDE_total_numbers_correct_l2826_282649

/-- Represents a student in the talent show -/
inductive Student : Type
| Sarah : Student
| Ben : Student
| Jake : Student
| Lily : Student

/-- The total number of musical numbers in the show -/
def total_numbers : ℕ := 7

/-- The number of songs Sarah performed -/
def sarah_songs : ℕ := 6

/-- The number of songs Ben performed -/
def ben_songs : ℕ := sarah_songs - 3

/-- The number of songs Jake performed -/
def jake_songs : ℕ := 6

/-- The number of songs Lily performed -/
def lily_songs : ℕ := 6

/-- The number of duo shows Jake and Lily performed together -/
def jake_lily_duo : ℕ := 1

/-- The number of shows Jake and Lily performed together -/
def jake_lily_together : ℕ := 6

/-- Theorem stating that the total number of musical numbers is correct -/
theorem total_numbers_correct : 
  (sarah_songs = total_numbers - 2) ∧ 
  (ben_songs = sarah_songs - 3) ∧
  (jake_songs = lily_songs) ∧
  (jake_lily_together ≤ 7) ∧
  (jake_lily_together > jake_songs - jake_lily_duo) ∧
  (total_numbers = jake_songs + 1) := by
  sorry

#check total_numbers_correct

end NUMINAMATH_CALUDE_total_numbers_correct_l2826_282649


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2826_282613

/-- The function f(x) = x^2 - px + q reaches its minimum when x = p/2, given p > 0 and q > 0 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := fun x ↦ x^2 - p*x + q
  ∃ (x_min : ℝ), x_min = p/2 ∧ ∀ (x : ℝ), f x_min ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2826_282613


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2826_282698

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a + 1 ≥ (2 * Real.sqrt 2 / 3) * (Real.sqrt ((a + b) / c) + Real.sqrt ((b + c) / a) + Real.sqrt ((c + a) / b))) ∧
  (a / b + b / c + c / a + 1 = (2 * Real.sqrt 2 / 3) * (Real.sqrt ((a + b) / c) + Real.sqrt ((b + c) / a) + Real.sqrt ((c + a) / b)) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2826_282698


namespace NUMINAMATH_CALUDE_rope_division_l2826_282640

def rope_length : ℝ := 3
def num_segments : ℕ := 7

theorem rope_division (segment_fraction : ℝ) (segment_length : ℝ) :
  (segment_fraction = 1 / num_segments) ∧
  (segment_length = rope_length / num_segments) ∧
  (segment_fraction = 1 / 7) ∧
  (segment_length = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l2826_282640


namespace NUMINAMATH_CALUDE_equation_roots_l2826_282632

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => -x * (x + 3) - x * (x + 3)
  (f 0 = 0 ∧ f (-3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 0 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2826_282632


namespace NUMINAMATH_CALUDE_cone_volume_l2826_282656

/-- Given a cone with base radius 1 and slant height equal to the diameter of the base,
    prove that its volume is (√3 * π) / 3 -/
theorem cone_volume (r : ℝ) (l : ℝ) (h : ℝ) :
  r = 1 →
  l = 2 * r →
  h ^ 2 + r ^ 2 = l ^ 2 →
  (1 / 3) * π * r ^ 2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2826_282656


namespace NUMINAMATH_CALUDE_original_to_circle_l2826_282671

/-- The original curve in polar coordinates -/
def original_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

/-- The transformation applied to the curve -/
def transformation (x y x'' y'' : ℝ) : Prop :=
  x'' = (1/2) * x ∧ y'' = (Real.sqrt 3 / 3) * y

/-- The resulting curve after transformation -/
def resulting_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

/-- Theorem stating that the original curve transforms into a circle -/
theorem original_to_circle :
  ∀ (ρ θ x y x'' y'' : ℝ),
    original_curve ρ θ →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    transformation x y x'' y'' →
    resulting_curve x'' y'' :=
sorry

end NUMINAMATH_CALUDE_original_to_circle_l2826_282671


namespace NUMINAMATH_CALUDE_f_is_odd_l2826_282629

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + Real.sin x) / Real.cos x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_is_odd : is_odd f := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l2826_282629


namespace NUMINAMATH_CALUDE_exponential_graph_quadrants_l2826_282600

theorem exponential_graph_quadrants (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_exponential_graph_quadrants_l2826_282600


namespace NUMINAMATH_CALUDE_team_division_probabilities_l2826_282652

/-- The total number of teams -/
def total_teams : ℕ := 8

/-- The number of weak teams -/
def weak_teams : ℕ := 3

/-- The number of teams in each group -/
def group_size : ℕ := 4

/-- The probability that one group has exactly two weak teams -/
def prob_two_weak : ℚ := 6/7

/-- The probability that group A has at least two weak teams -/
def prob_A_at_least_two : ℚ := 1/2

/-- Theorem stating the probabilities for the team division problem -/
theorem team_division_probabilities :
  (prob_two_weak = 6/7) ∧ (prob_A_at_least_two = 1/2) := by sorry

end NUMINAMATH_CALUDE_team_division_probabilities_l2826_282652


namespace NUMINAMATH_CALUDE_exit_times_theorem_l2826_282634

/-- Represents the time in minutes it takes to exit through a door -/
structure ExitTime where
  time : ℝ
  time_positive : time > 0

/-- Represents the cinema with two doors -/
structure Cinema where
  wide_door : ExitTime
  narrow_door : ExitTime
  combined_exit_time : ℝ
  combined_exit_time_value : combined_exit_time = 3.75
  door_time_difference : narrow_door.time = wide_door.time + 4

theorem exit_times_theorem (c : Cinema) :
  c.wide_door.time = 6 ∧ c.narrow_door.time = 10 := by
  sorry

#check exit_times_theorem

end NUMINAMATH_CALUDE_exit_times_theorem_l2826_282634


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l2826_282674

/-- Given two points M and N in a 2D plane, this theorem proves that the midpoint P of the line segment MN has specific coordinates. -/
theorem midpoint_coordinates (M N : ℝ × ℝ) (hM : M = (3, -2)) (hN : N = (-1, 0)) :
  let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l2826_282674


namespace NUMINAMATH_CALUDE_circle_condition_l2826_282675

/-- The equation x^2 + y^2 - 2x + 6y + m = 0 represents a circle if and only if m < 10 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + 6*y + m = 0 ∧ 
   ∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x + 6*y + m = 0) 
  ↔ m < 10 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2826_282675


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2826_282626

theorem geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- a is a geometric sequence with common ratio q
  a 1 + a 2 + a 3 + a 4 = 15/8 →    -- sum of first four terms
  a 2 * a 3 = -9/8 →                -- product of second and third terms
  1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 = -5/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2826_282626


namespace NUMINAMATH_CALUDE_david_weighted_average_l2826_282643

def david_marks : List ℕ := [76, 65, 82, 67, 85, 93, 71]

def english_weight : ℕ := 2
def math_weight : ℕ := 3
def science_weight : ℕ := 1

def weighted_sum : ℕ := 
  david_marks[0] * english_weight + 
  david_marks[1] * math_weight + 
  david_marks[2] * science_weight + 
  david_marks[3] * science_weight + 
  david_marks[4] * science_weight

def total_weight : ℕ := english_weight + math_weight + 3 * science_weight

theorem david_weighted_average :
  (weighted_sum : ℚ) / total_weight = 581 / 8 := by sorry

end NUMINAMATH_CALUDE_david_weighted_average_l2826_282643


namespace NUMINAMATH_CALUDE_star_op_equivalence_l2826_282628

-- Define the ※ operation
def star_op (m n : ℝ) : ℝ := m * n - m - n + 3

-- State the theorem
theorem star_op_equivalence (x : ℝ) :
  6 < star_op 2 x ∧ star_op 2 x < 7 ↔ 5 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_star_op_equivalence_l2826_282628


namespace NUMINAMATH_CALUDE_inequality_proof_l2826_282622

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 3/4) ∧
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2826_282622


namespace NUMINAMATH_CALUDE_two_pythagorean_triples_l2826_282614

-- Define a Pythagorean triple
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- State the theorem
theorem two_pythagorean_triples :
  isPythagoreanTriple 3 4 5 ∧ isPythagoreanTriple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_two_pythagorean_triples_l2826_282614


namespace NUMINAMATH_CALUDE_triangle_mass_l2826_282641

-- Define the shapes
variable (Square Circle Triangle : ℝ)

-- Define the scale equations
axiom scale1 : Square + Circle = 8
axiom scale2 : Square + 2 * Circle = 11
axiom scale3 : Circle + 2 * Triangle = 15

-- Theorem to prove
theorem triangle_mass : Triangle = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_mass_l2826_282641


namespace NUMINAMATH_CALUDE_number_problem_l2826_282689

theorem number_problem (n : ℝ) : (0.6 * (3/5) * n = 36) → n = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2826_282689


namespace NUMINAMATH_CALUDE_student_in_all_clubs_l2826_282692

theorem student_in_all_clubs (n : ℕ) (F G C : Finset (Fin n)) :
  n = 30 →
  F.card = 22 →
  G.card = 21 →
  C.card = 18 →
  ∃ s, s ∈ F ∩ G ∩ C :=
by
  sorry

end NUMINAMATH_CALUDE_student_in_all_clubs_l2826_282692
