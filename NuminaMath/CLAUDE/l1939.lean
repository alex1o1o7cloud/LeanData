import Mathlib

namespace NUMINAMATH_CALUDE_expected_malfunctioning_computers_correct_l1939_193912

/-- The expected number of malfunctioning computers -/
def expected_malfunctioning_computers (a b : ℝ) : ℝ := a + b

/-- Theorem: The expected number of malfunctioning computers is a + b -/
theorem expected_malfunctioning_computers_correct (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  expected_malfunctioning_computers a b = a + b := by
  sorry

end NUMINAMATH_CALUDE_expected_malfunctioning_computers_correct_l1939_193912


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1939_193983

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- The common ratio of a geometric sequence is the constant factor between successive terms. -/
def CommonRatio (a : ℕ → ℚ) : ℚ :=
  a 1 / a 0

theorem geometric_sequence_ratio :
  ∀ a : ℕ → ℚ,
  IsGeometricSequence a →
  a 0 = 25 →
  a 1 = -50 →
  a 2 = 100 →
  a 3 = -200 →
  CommonRatio a = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1939_193983


namespace NUMINAMATH_CALUDE_least_distance_between_ticks_l1939_193942

theorem least_distance_between_ticks (n m : ℕ) (hn : n = 11) (hm : m = 13) :
  let lcm := Nat.lcm n m
  1 / lcm = (1 : ℚ) / 143 := by sorry

end NUMINAMATH_CALUDE_least_distance_between_ticks_l1939_193942


namespace NUMINAMATH_CALUDE_protective_clothing_equation_l1939_193908

/-- Represents the equation for the protective clothing production problem -/
theorem protective_clothing_equation (x : ℝ) (h : x > 0) :
  let total_sets := 1000
  let increase_rate := 0.2
  let days_ahead := 2
  let original_days := total_sets / x
  let actual_days := total_sets / (x * (1 + increase_rate))
  original_days - actual_days = days_ahead :=
by sorry

end NUMINAMATH_CALUDE_protective_clothing_equation_l1939_193908


namespace NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l1939_193945

theorem sinusoidal_vertical_shift 
  (A B C D : ℝ) 
  (h_max : ∀ x, A * Real.sin (B * x + C) + D ≤ 5)
  (h_min : ∀ x, A * Real.sin (B * x + C) + D ≥ -3)
  (h_max_achieved : ∃ x, A * Real.sin (B * x + C) + D = 5)
  (h_min_achieved : ∃ x, A * Real.sin (B * x + C) + D = -3) :
  D = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l1939_193945


namespace NUMINAMATH_CALUDE_expected_throws_in_leap_year_l1939_193956

/-- The expected number of throws for a single day -/
def expected_throws_per_day : ℚ := 8/7

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The expected number of throws in a leap year -/
def expected_throws_leap_year : ℚ := expected_throws_per_day * leap_year_days

theorem expected_throws_in_leap_year :
  expected_throws_leap_year = 2928/7 := by sorry

end NUMINAMATH_CALUDE_expected_throws_in_leap_year_l1939_193956


namespace NUMINAMATH_CALUDE_equation_solutions_l1939_193903

/-- The equation has solutions when the parameter a is greater than 1 -/
def has_solution (a : ℝ) : Prop :=
  a > 1

/-- The solutions of the equation for a given parameter a -/
def solutions (a : ℝ) : Set ℝ :=
  if a > 2 then { (1 - a) / a, -1, 1 - a }
  else if a = 2 then { -1, -1/2 }
  else if 1 < a ∧ a < 2 then { (1 - a) / a, -1, 1 - a }
  else ∅

/-- The main theorem stating that the equation has solutions for a > 1 
    and providing these solutions -/
theorem equation_solutions (a : ℝ) :
  has_solution a →
  ∃ x : ℝ, x ∈ solutions a ∧
    (2 - 2 * a * (x + 1)) / (|x| - x) = Real.sqrt (1 - a - a * x) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l1939_193903


namespace NUMINAMATH_CALUDE_inequality_max_m_l1939_193979

theorem inequality_max_m : 
  ∀ (m : ℝ), 
  (∀ (a b : ℝ), a > 0 → b > 0 → (2/a + 1/b ≥ m/(2*a+b))) ↔ m ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_max_m_l1939_193979


namespace NUMINAMATH_CALUDE_even_function_inequality_l1939_193980

/-- An even function from ℝ to ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf_even : EvenFunction f)
  (hf_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x, 2 * f x + x * f' x < 2) :
  ∀ x, x^2 * f x - f 1 < x^2 - 1 ↔ |x| > 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_inequality_l1939_193980


namespace NUMINAMATH_CALUDE_ones_digit_of_complex_expression_l1939_193918

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Define the expression
def complex_expression : ℕ := 
  ones_digit ((73^567 % 10) * (47^123 % 10) + (86^784 % 10) - (32^259 % 10))

-- Theorem statement
theorem ones_digit_of_complex_expression :
  complex_expression = 9 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_complex_expression_l1939_193918


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_perpendicular_line_plane_from_intersection_l1939_193914

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (contained_in : Line → Plane → Prop)

-- Theorem for statement ②
theorem parallel_planes_from_perpendicular_lines 
  (l m : Line) (α β : Plane) :
  parallel l m →
  perpendicular_line_plane m α →
  perpendicular_line_plane l β →
  parallel_plane α β :=
sorry

-- Theorem for statement ④
theorem perpendicular_line_plane_from_intersection 
  (l : Line) (α β : Plane) (m : Line) :
  perpendicular_plane α β →
  intersection α β = m →
  contained_in l β →
  perpendicular l m →
  perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_perpendicular_line_plane_from_intersection_l1939_193914


namespace NUMINAMATH_CALUDE_sam_memorized_digits_l1939_193984

/-- Given information about the number of digits of pi memorized by Sam, Carlos, and Mina,
    prove that Sam memorized 10 digits. -/
theorem sam_memorized_digits (sam carlos mina : ℕ) 
  (h1 : sam = carlos + 6)
  (h2 : mina = 6 * carlos)
  (h3 : mina = 24) : 
  sam = 10 := by sorry

end NUMINAMATH_CALUDE_sam_memorized_digits_l1939_193984


namespace NUMINAMATH_CALUDE_sharons_journey_l1939_193919

/-- The distance between Sharon's house and her mother's house -/
def total_distance : ℝ := 200

/-- The time Sharon usually takes to complete the journey -/
def usual_time : ℝ := 200

/-- The time taken on the day with traffic -/
def traffic_time : ℝ := 300

/-- The speed reduction due to traffic in miles per hour -/
def speed_reduction : ℝ := 30

theorem sharons_journey :
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    initial_speed * usual_time = total_distance ∧
    (total_distance / 2) / initial_speed +
    (total_distance / 2) / (initial_speed - speed_reduction / 60) = traffic_time :=
  sorry

end NUMINAMATH_CALUDE_sharons_journey_l1939_193919


namespace NUMINAMATH_CALUDE_sum_of_symmetric_points_coords_l1939_193967

/-- Two points P₁ and P₂ are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are the same. -/
def symmetric_wrt_y_axis (P₁ P₂ : ℝ × ℝ) : Prop :=
  P₁.1 = -P₂.1 ∧ P₁.2 = P₂.2

/-- Given two points P₁(a,-5) and P₂(3,b) that are symmetric with respect to the y-axis,
    prove that a + b = -8. -/
theorem sum_of_symmetric_points_coords (a b : ℝ) 
    (h : symmetric_wrt_y_axis (a, -5) (3, b)) : 
  a + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_points_coords_l1939_193967


namespace NUMINAMATH_CALUDE_distance_range_m_l1939_193927

-- Define the distance function
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + 2 * |y₁ - y₂|

-- Define the theorem
theorem distance_range_m :
  ∀ m : ℝ,
  (distance 2 1 (-1) m ≤ 5) ↔ (0 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_range_m_l1939_193927


namespace NUMINAMATH_CALUDE_infinitely_many_n_divisible_by_prime_count_l1939_193958

-- Define π(n) as the number of prime numbers not greater than n
def prime_count (n : ℕ) : ℕ := (Finset.filter Nat.Prime (Finset.range (n + 1))).card

-- Statement of the theorem
theorem infinitely_many_n_divisible_by_prime_count :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, prime_count n ∣ n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_divisible_by_prime_count_l1939_193958


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l1939_193960

theorem diet_soda_bottles (total_regular_and_diet : ℕ) (regular_bottles : ℕ) 
  (h1 : total_regular_and_diet = 89)
  (h2 : regular_bottles = 49) :
  total_regular_and_diet - regular_bottles = 40 := by
sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l1939_193960


namespace NUMINAMATH_CALUDE_aquarium_trainers_l1939_193991

/-- The number of trainers required to equally split the total training hours for all dolphins -/
def number_of_trainers (num_dolphins : ℕ) (hours_per_dolphin : ℕ) (hours_per_trainer : ℕ) : ℕ :=
  (num_dolphins * hours_per_dolphin) / hours_per_trainer

theorem aquarium_trainers :
  number_of_trainers 4 3 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_trainers_l1939_193991


namespace NUMINAMATH_CALUDE_modified_short_bingo_first_column_l1939_193963

/-- The number of elements in the set from which we select numbers -/
def n : ℕ := 15

/-- The number of elements we select -/
def k : ℕ := 5

/-- The number of ways to select k distinct numbers from a set of n numbers, where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

theorem modified_short_bingo_first_column : permutations n k = 360360 := by
  sorry

end NUMINAMATH_CALUDE_modified_short_bingo_first_column_l1939_193963


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l1939_193955

theorem smallest_number_in_sequence (x : ℝ) : 
  let second := 4 * x
  let third := 2 * second
  (x + second + third) / 3 = 78 →
  x = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l1939_193955


namespace NUMINAMATH_CALUDE_expansion_simplification_l1939_193944

theorem expansion_simplification (y : ℝ) : (2*y - 3)*(2*y + 3) - (4*y - 1)*(y + 5) = -19*y - 4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l1939_193944


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1939_193910

theorem minimum_value_theorem (m n t : ℝ) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m + n = 1) 
  (ht : t > 0) 
  (hmin : ∀ s > 0, s / m + 1 / n ≥ t / m + 1 / n) 
  (heq : t / m + 1 / n = 9) : 
  t = 4 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1939_193910


namespace NUMINAMATH_CALUDE_complex_power_absolute_value_l1939_193959

theorem complex_power_absolute_value : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_absolute_value_l1939_193959


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l1939_193968

theorem estimate_sqrt_expression :
  5 < Real.sqrt (1/3) * Real.sqrt 27 + Real.sqrt 7 ∧
  Real.sqrt (1/3) * Real.sqrt 27 + Real.sqrt 7 < 6 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l1939_193968


namespace NUMINAMATH_CALUDE_five_dice_not_same_l1939_193990

theorem five_dice_not_same (n : ℕ) (h : n = 8) : 
  (1 - (n : ℚ)/(n^5 : ℚ)) = 4095/4096 := by
  sorry

end NUMINAMATH_CALUDE_five_dice_not_same_l1939_193990


namespace NUMINAMATH_CALUDE_george_dimes_count_l1939_193973

/-- Prove the number of dimes in George's collection --/
theorem george_dimes_count :
  let total_coins : ℕ := 28
  let total_value : ℚ := 260/100
  let nickel_count : ℕ := 4
  let nickel_value : ℚ := 5/100
  let dime_value : ℚ := 10/100
  ∃ dime_count : ℕ,
    dime_count = 24 ∧
    dime_count + nickel_count = total_coins ∧
    dime_count * dime_value + nickel_count * nickel_value = total_value :=
by sorry

end NUMINAMATH_CALUDE_george_dimes_count_l1939_193973


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1939_193962

theorem point_in_second_quadrant (x y : ℝ) : 
  x < 0 ∧ y > 0 →  -- point is in the second quadrant
  |y| = 4 →        -- 4 units away from x-axis
  |x| = 7 →        -- 7 units away from y-axis
  x = -7 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1939_193962


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l1939_193964

theorem cube_root_of_eight (x : ℝ) : x^3 = 8 → x = 2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l1939_193964


namespace NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1939_193972

/-- A system of linear equations with three variables -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ → Prop
  eq2 : ℝ → ℝ → ℝ → Prop
  eq3 : ℝ → ℝ → ℝ → Prop

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem where
  eq1 := fun x y z => x + y + z = 15
  eq2 := fun x y z => x - y + z = 5
  eq3 := fun x y z => x + y - z = 10

/-- The solution to the system of equations -/
def solution : ℝ × ℝ × ℝ := (7.5, 5, 2.5)

/-- Theorem stating that the solution satisfies the system of equations -/
theorem solution_satisfies_system :
  let (x, y, z) := solution
  problemSystem.eq1 x y z ∧
  problemSystem.eq2 x y z ∧
  problemSystem.eq3 x y z :=
by sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique :
  ∀ x y z, 
    problemSystem.eq1 x y z →
    problemSystem.eq2 x y z →
    problemSystem.eq3 x y z →
    (x, y, z) = solution :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1939_193972


namespace NUMINAMATH_CALUDE_parent_payment_amount_l1939_193997

-- Define the given parameters
def original_salary : ℕ := 60000
def raise_percentage : ℚ := 25 / 100
def num_kids : ℕ := 15

-- Define the theorem
theorem parent_payment_amount :
  (original_salary + original_salary * raise_percentage) / num_kids = 5000 := by
  sorry

end NUMINAMATH_CALUDE_parent_payment_amount_l1939_193997


namespace NUMINAMATH_CALUDE_clothing_retailer_optimal_strategy_l1939_193935

/-- Represents the clothing retailer's purchase and sales data --/
structure ClothingRetailer where
  first_purchase_cost : ℝ
  second_purchase_cost : ℝ
  cost_increase_per_item : ℝ
  base_price : ℝ
  base_sales : ℝ
  price_decrease : ℝ
  sales_increase : ℝ
  daily_profit : ℝ

/-- Theorem stating the initial purchase quantity and price, and the optimal selling price --/
theorem clothing_retailer_optimal_strategy (r : ClothingRetailer)
  (h1 : r.first_purchase_cost = 48000)
  (h2 : r.second_purchase_cost = 100000)
  (h3 : r.cost_increase_per_item = 10)
  (h4 : r.base_price = 300)
  (h5 : r.base_sales = 80)
  (h6 : r.price_decrease = 10)
  (h7 : r.sales_increase = 20)
  (h8 : r.daily_profit = 3600) :
  ∃ (initial_quantity : ℝ) (initial_price : ℝ) (selling_price : ℝ),
    initial_quantity = 200 ∧
    initial_price = 240 ∧
    selling_price = 280 ∧
    (selling_price - (initial_price + r.cost_increase_per_item)) *
      (r.base_sales + (r.base_price - selling_price) / r.price_decrease * r.sales_increase) = r.daily_profit :=
by sorry

end NUMINAMATH_CALUDE_clothing_retailer_optimal_strategy_l1939_193935


namespace NUMINAMATH_CALUDE_petes_flag_has_128_shapes_l1939_193998

/-- Calculates the total number of shapes on Pete's flag given the number of stars and stripes on the US flag. -/
def petes_flag_shapes (us_stars : ℕ) (us_stripes : ℕ) : ℕ :=
  let circles := us_stars / 2 - 3
  let squares := 2 * us_stripes + 6
  let triangles := 2 * (us_stars - us_stripes)
  circles + squares + triangles

/-- Theorem stating that Pete's flag has 128 shapes given the US flag has 50 stars and 13 stripes. -/
theorem petes_flag_has_128_shapes :
  petes_flag_shapes 50 13 = 128 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_has_128_shapes_l1939_193998


namespace NUMINAMATH_CALUDE_ratio_problem_l1939_193913

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1939_193913


namespace NUMINAMATH_CALUDE_discount_equation_l1939_193916

/-- Represents the discount rate as a real number between 0 and 1 -/
def discount_rate : ℝ := sorry

/-- The original price in yuan -/
def original_price : ℝ := 200

/-- The final selling price in yuan -/
def final_price : ℝ := 148

/-- Theorem stating the relationship between original price, discount rate, and final price -/
theorem discount_equation : 
  original_price * (1 - discount_rate)^2 = final_price := by sorry

end NUMINAMATH_CALUDE_discount_equation_l1939_193916


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_n_equals_three_l1939_193989

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (3, 2) and b = (2, n), if a is perpendicular to b, then n = 3 -/
theorem perpendicular_vectors_imply_n_equals_three :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ → ℝ × ℝ := λ n => (2, n)
  ∀ n : ℝ, perpendicular a (b n) → n = 3 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_vectors_imply_n_equals_three_l1939_193989


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1939_193909

theorem complex_fraction_simplification :
  1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1939_193909


namespace NUMINAMATH_CALUDE_merchant_discount_l1939_193934

theorem merchant_discount (markup : ℝ) (profit : ℝ) (discount : ℝ) : 
  markup = 0.75 → 
  profit = 0.225 → 
  discount = (markup + 1 - (profit + 1)) / (markup + 1) * 100 →
  discount = 30 := by
  sorry

end NUMINAMATH_CALUDE_merchant_discount_l1939_193934


namespace NUMINAMATH_CALUDE_tens_digit_of_9_power_2023_l1939_193952

theorem tens_digit_of_9_power_2023 (h1 : 9^10 % 50 = 1) (h2 : 9^3 % 50 = 29) :
  (9^2023 / 10) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_power_2023_l1939_193952


namespace NUMINAMATH_CALUDE_ned_good_games_l1939_193931

/-- Calculates the number of good games Ned ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - non_working_games

/-- Theorem: Ned ended up with 14 good games -/
theorem ned_good_games : good_games 11 22 19 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ned_good_games_l1939_193931


namespace NUMINAMATH_CALUDE_no_real_solutions_l1939_193938

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 4) ∧ (x * y - z^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1939_193938


namespace NUMINAMATH_CALUDE_vector_to_line_l1939_193988

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if a vector is on a parametric line -/
def vector_on_line (v : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = v.1 ∧ l.y t = v.2

/-- Check if two vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_to_line (l : ParametricLine) (v : ℝ × ℝ) :
  l.x t = 3 * t + 5 →
  l.y t = 2 * t + 1 →
  vector_on_line v l →
  parallel v (3, 2) →
  v = (21/2, 7) := by sorry

end NUMINAMATH_CALUDE_vector_to_line_l1939_193988


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l1939_193920

theorem right_triangle_arithmetic_progression :
  ∃ (a d c : ℕ), 
    a > 0 ∧ d > 0 ∧ c > 0 ∧
    a * a + (a + d) * (a + d) = c * c ∧
    c = a + 2 * d ∧
    (a = 120 ∨ a + d = 120 ∨ c = 120) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_l1939_193920


namespace NUMINAMATH_CALUDE_farm_spiders_l1939_193966

/-- Represents the number of animals on a farm -/
structure FarmAnimals where
  ducks : ℕ
  cows : ℕ
  spiders : ℕ

/-- Calculates the total number of legs for the given farm animals -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  2 * animals.ducks + 4 * animals.cows + 8 * animals.spiders

/-- Calculates the total number of wings for the given farm animals -/
def totalWings (animals : FarmAnimals) : ℕ :=
  2 * animals.ducks

/-- Calculates the total number of heads (animals) for the given farm animals -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.ducks + animals.cows + animals.spiders

/-- Theorem: Given the conditions, there are 20 spiders on the farm -/
theorem farm_spiders (animals : FarmAnimals) :
  (3 * animals.cows = 2 * animals.ducks) →
  (totalLegs animals = 270) →
  (totalHeads animals = 70) →
  (totalWings animals = 60) →
  animals.spiders = 20 := by
  sorry


end NUMINAMATH_CALUDE_farm_spiders_l1939_193966


namespace NUMINAMATH_CALUDE_x_minus_y_equals_negative_three_l1939_193969

theorem x_minus_y_equals_negative_three
  (eq1 : 2020 * x + 2024 * y = 2028)
  (eq2 : 2022 * x + 2026 * y = 2030)
  : x - y = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_negative_three_l1939_193969


namespace NUMINAMATH_CALUDE_triangle_side_length_l1939_193987

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (area : Real) :
  area = Real.sqrt 3 →
  B = 60 * π / 180 →
  a^2 + c^2 = 3 * a * c →
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1939_193987


namespace NUMINAMATH_CALUDE_middle_income_sample_size_l1939_193930

/-- Calculates the number of households to be drawn from a specific income group in a stratified sample. -/
def stratifiedSampleSize (totalHouseholds : ℕ) (groupHouseholds : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupHouseholds * sampleSize) / totalHouseholds

/-- Proves that the number of middle-income households in the stratified sample is 60. -/
theorem middle_income_sample_size :
  let totalHouseholds : ℕ := 600
  let middleIncomeHouseholds : ℕ := 360
  let sampleSize : ℕ := 100
  stratifiedSampleSize totalHouseholds middleIncomeHouseholds sampleSize = 60 := by
  sorry


end NUMINAMATH_CALUDE_middle_income_sample_size_l1939_193930


namespace NUMINAMATH_CALUDE_invisible_dots_count_l1939_193957

/-- The sum of numbers on a standard six-sided die -/
def standard_die_sum : Nat := 21

/-- The number of dice rolled -/
def num_dice : Nat := 4

/-- The sum of visible numbers on the dice -/
def visible_sum : Nat := 6 + 6 + 4 + 4 + 3 + 2 + 1

/-- The total number of dots on all dice -/
def total_dots : Nat := num_dice * standard_die_sum

theorem invisible_dots_count : total_dots - visible_sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l1939_193957


namespace NUMINAMATH_CALUDE_division_of_decimals_l1939_193940

theorem division_of_decimals : (0.045 : ℚ) / (0.009 : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1939_193940


namespace NUMINAMATH_CALUDE_even_function_monotonicity_l1939_193929

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

theorem even_function_monotonicity (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  ∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x ∈ Set.Ioo (-3) 1, f m x = -x^2 + 3) ∧
    (∀ x y, -3 < x ∧ x < y ∧ y < a → f m x < f m y) ∧
    (∀ x y, a < x ∧ x < y ∧ y < 1 → f m x > f m y) :=
by sorry

end NUMINAMATH_CALUDE_even_function_monotonicity_l1939_193929


namespace NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_length_proof_l1939_193946

/-- Represents a right triangle with one leg of 10 inches and the angle opposite that leg being 60° --/
structure RightTriangle where
  leg : ℝ
  angle : ℝ
  leg_eq : leg = 10
  angle_eq : angle = 60

/-- Theorem stating that the hypotenuse of the described right triangle is (20√3)/3 inches --/
theorem hypotenuse_length (t : RightTriangle) : ℝ :=
  (20 * Real.sqrt 3) / 3

/-- Proof of the theorem --/
theorem hypotenuse_length_proof (t : RightTriangle) : 
  hypotenuse_length t = (20 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_length_proof_l1939_193946


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1939_193907

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_one :
  (f' 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1939_193907


namespace NUMINAMATH_CALUDE_percentage_of_english_books_l1939_193970

theorem percentage_of_english_books (total_books : ℕ) 
  (english_books_outside : ℕ) (percentage_published_in_country : ℚ) :
  total_books = 2300 →
  english_books_outside = 736 →
  percentage_published_in_country = 60 / 100 →
  (english_books_outside / (1 - percentage_published_in_country)) / total_books = 80 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_english_books_l1939_193970


namespace NUMINAMATH_CALUDE_landscape_length_l1939_193971

theorem landscape_length (breadth : ℝ) 
  (length_eq : length = 8 * breadth)
  (playground_area : ℝ)
  (playground_eq : playground_area = 1200)
  (playground_ratio : playground_area = (1/6) * (length * breadth)) : 
  length = 240 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_l1939_193971


namespace NUMINAMATH_CALUDE_binary_to_decimal_example_l1939_193923

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number -/
def binary_number : List Nat := [1, 1, 0, 1, 1, 1, 1, 0, 1]

/-- Theorem stating that the given binary number is equal to 379 in decimal -/
theorem binary_to_decimal_example : binary_to_decimal binary_number = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_example_l1939_193923


namespace NUMINAMATH_CALUDE_inverse_proportion_intersection_l1939_193917

theorem inverse_proportion_intersection (b : ℝ) :
  ∃ k : ℝ, 1 < k ∧ k < 2 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (6 - 3 * k) / x₁ = -7 * x₁ + b ∧
    (6 - 3 * k) / x₂ = -7 * x₂ + b ∧
    x₁ * x₂ > 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_intersection_l1939_193917


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l1939_193996

theorem solution_satisfies_equation (C D : ℝ) :
  D = 2 → C = 5.25 → 4 * C + 2 * D + 5 = 30 := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l1939_193996


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l1939_193986

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x * y ≤ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x + 2*y ≤ a + 2*b) ∧
  x * y = 36 ∧
  x + 2*y = 19 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l1939_193986


namespace NUMINAMATH_CALUDE_C_div_D_eq_17_l1939_193950

noncomputable def C : ℝ := ∑' n, if n % 4 ≠ 0 ∧ n % 2 = 0 then (-1)^((n/2) % 2 + 1) / n^2 else 0

noncomputable def D : ℝ := ∑' n, if n % 4 = 0 then (-1)^(n/4 + 1) / n^2 else 0

theorem C_div_D_eq_17 : C / D = 17 := by sorry

end NUMINAMATH_CALUDE_C_div_D_eq_17_l1939_193950


namespace NUMINAMATH_CALUDE_box_length_proof_l1939_193905

/-- Proves that the length of a box with given dimensions and cube requirements is 10 cm -/
theorem box_length_proof (width : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h_width : width = 13)
  (h_height : height = 5)
  (h_cube_volume : cube_volume = 5)
  (h_min_cubes : min_cubes = 130) :
  (min_cubes : ℝ) * cube_volume / (width * height) = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_length_proof_l1939_193905


namespace NUMINAMATH_CALUDE_expression_equals_49_l1939_193939

theorem expression_equals_49 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_49_l1939_193939


namespace NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l1939_193948

def is_perpendicular (m : ℝ) : Prop :=
  let line1_slope := -m
  let line2_slope := 1 / m
  line1_slope * line2_slope = -1

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 1 ∧ is_perpendicular m) ∧
  (is_perpendicular 1) :=
sorry

end NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l1939_193948


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1939_193941

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem unique_function_satisfying_conditions :
  (∀ (x : ℝ), x ≠ 0 → f x = x * f (1 / x)) ∧
  (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x ≠ -y → f x + f y = 1 + f (x + y)) ∧
  (∀ (g : ℝ → ℝ), 
    ((∀ (x : ℝ), x ≠ 0 → g x = x * g (1 / x)) ∧
     (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x ≠ -y → g x + g y = 1 + g (x + y)))
    → (∀ (x : ℝ), x ≠ 0 → g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1939_193941


namespace NUMINAMATH_CALUDE_triangle_problem_l1939_193928

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of Sines holds
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  a / Real.sin A = 2 * c / Real.sqrt 3 →
  -- c = √7
  c = Real.sqrt 7 →
  -- Area of triangle ABC is 3√3/2
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  -- Prove:
  C = π/3 ∧ a^2 + b^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1939_193928


namespace NUMINAMATH_CALUDE_cary_calorie_deficit_l1939_193965

/-- Calculates the net calorie deficit given the total distance walked, calories burned per mile, and calories consumed. -/
def net_calorie_deficit (distance : ℝ) (calories_per_mile : ℝ) (calories_consumed : ℝ) : ℝ :=
  distance * calories_per_mile - calories_consumed

/-- Proves that given the specified conditions, the net calorie deficit is 250 calories. -/
theorem cary_calorie_deficit :
  let distance : ℝ := 3
  let calories_per_mile : ℝ := 150
  let calories_consumed : ℝ := 200
  net_calorie_deficit distance calories_per_mile calories_consumed = 250 := by
  sorry

end NUMINAMATH_CALUDE_cary_calorie_deficit_l1939_193965


namespace NUMINAMATH_CALUDE_sum_of_ratios_theorem_l1939_193943

theorem sum_of_ratios_theorem (a b c : ℚ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a * b * c ≠ 0) (h5 : a + b + c = 0) : 
  a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ratios_theorem_l1939_193943


namespace NUMINAMATH_CALUDE_cistern_leak_emptying_time_l1939_193993

/-- Given a cistern that normally fills in 8 hours, but takes 10 hours to fill with a leak,
    prove that it takes 40 hours for a full cistern to empty due to the leak. -/
theorem cistern_leak_emptying_time (normal_fill_time leak_fill_time : ℝ) 
    (h1 : normal_fill_time = 8)
    (h2 : leak_fill_time = 10) : 
  let normal_fill_rate := 1 / normal_fill_time
  let leak_rate := normal_fill_rate - (1 / leak_fill_time)
  (1 / leak_rate) = 40 := by sorry

end NUMINAMATH_CALUDE_cistern_leak_emptying_time_l1939_193993


namespace NUMINAMATH_CALUDE_total_cellphones_sold_l1939_193947

/-- Calculates the number of cell phones sold given initial and final inventories and damaged/defective phones. -/
def cellphonesSold (initialSamsung : ℕ) (finalSamsung : ℕ) (initialIphone : ℕ) (finalIphone : ℕ) (damagedSamsung : ℕ) (defectiveIphone : ℕ) : ℕ :=
  (initialSamsung - damagedSamsung - finalSamsung) + (initialIphone - defectiveIphone - finalIphone)

/-- Proves that the total number of cell phones sold is 4 given the inventory information. -/
theorem total_cellphones_sold :
  cellphonesSold 14 10 8 5 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_cellphones_sold_l1939_193947


namespace NUMINAMATH_CALUDE_nuts_in_third_box_l1939_193985

theorem nuts_in_third_box (x y z : ℕ) : 
  x + 6 = y + z → y + 10 = x + z → z = 8 := by sorry

end NUMINAMATH_CALUDE_nuts_in_third_box_l1939_193985


namespace NUMINAMATH_CALUDE_airport_distance_proof_l1939_193999

/-- The distance from David's home to the airport in miles -/
def airport_distance : ℝ := 155

/-- David's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- The increase in speed for the remaining journey in miles per hour -/
def speed_increase : ℝ := 20

/-- The time David would be late if he continued at the initial speed, in hours -/
def late_time : ℝ := 0.75

theorem airport_distance_proof :
  ∃ (t : ℝ),
    -- t is the actual time needed to arrive on time
    t > 0 ∧
    -- The total distance equals the distance covered at the initial speed
    airport_distance = initial_speed * (t + late_time) ∧
    -- The remaining distance equals the distance covered at the increased speed
    airport_distance - initial_speed = (initial_speed + speed_increase) * (t - 1) :=
by sorry

#check airport_distance_proof

end NUMINAMATH_CALUDE_airport_distance_proof_l1939_193999


namespace NUMINAMATH_CALUDE_pythagoras_students_l1939_193992

theorem pythagoras_students : ∃ n : ℕ, 
  n > 0 ∧ 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 7 : ℚ) + 3 = n ∧ 
  n = 28 := by
  sorry

end NUMINAMATH_CALUDE_pythagoras_students_l1939_193992


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1939_193975

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1939_193975


namespace NUMINAMATH_CALUDE_intercepts_correct_l1939_193925

/-- The line equation is 5x - 2y - 10 = 0 -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Proof that the x-intercept and y-intercept are correct for the given line equation -/
theorem intercepts_correct : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by sorry

end NUMINAMATH_CALUDE_intercepts_correct_l1939_193925


namespace NUMINAMATH_CALUDE_mother_age_twice_alex_age_l1939_193976

/-- The year when Alex's mother's age will be twice his age -/
def target_year : ℕ := 2025

/-- Alex's birth year -/
def alex_birth_year : ℕ := 1997

/-- The year of Alex's 7th birthday -/
def seventh_birthday_year : ℕ := 2004

/-- Alex's age on his 7th birthday -/
def alex_age_seventh_birthday : ℕ := 7

/-- Alex's mother's age on Alex's 7th birthday -/
def mother_age_seventh_birthday : ℕ := 35

theorem mother_age_twice_alex_age :
  (target_year - seventh_birthday_year) + alex_age_seventh_birthday = 
  (target_year - seventh_birthday_year + mother_age_seventh_birthday) / 2 ∧
  mother_age_seventh_birthday = 5 * alex_age_seventh_birthday ∧
  seventh_birthday_year - alex_birth_year = alex_age_seventh_birthday :=
by sorry

end NUMINAMATH_CALUDE_mother_age_twice_alex_age_l1939_193976


namespace NUMINAMATH_CALUDE_complex_number_location_l1939_193926

/-- Given a complex number z satisfying (1-i)z = (1+i)^2, 
    prove that z has a negative real part and a positive imaginary part. -/
theorem complex_number_location (z : ℂ) (h : (1 - I) * z = (1 + I)^2) : 
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1939_193926


namespace NUMINAMATH_CALUDE_sum_first_three_terms_l1939_193994

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The first term of the sequence -/
  a : ℝ
  /-- The eighth term of the sequence is 20 -/
  eighth_term : a + 7 * d = 20
  /-- The common difference is 2 -/
  diff_is_two : d = 2

/-- The sum of the first three terms of the arithmetic sequence is 24 -/
theorem sum_first_three_terms (seq : ArithmeticSequence) :
  seq.a + (seq.a + seq.d) + (seq.a + 2 * seq.d) = 24 := by
  sorry


end NUMINAMATH_CALUDE_sum_first_three_terms_l1939_193994


namespace NUMINAMATH_CALUDE_work_completion_time_l1939_193953

/-- The time taken to complete a work given the rates of two workers and their working pattern -/
theorem work_completion_time
  (rate_A rate_B : ℝ)  -- Rates at which A and B can complete the work alone
  (days_together : ℝ)  -- Number of days A and B work together
  (h_rate_A : rate_A = 1 / 15)  -- A can complete the work in 15 days
  (h_rate_B : rate_B = 1 / 10)  -- B can complete the work in 10 days
  (h_days_together : days_together = 2)  -- A and B work together for 2 days
  : ∃ (total_days : ℝ), total_days = 12 ∧ 
    rate_A * (total_days - days_together) + (rate_A + rate_B) * days_together = 1 :=
by sorry


end NUMINAMATH_CALUDE_work_completion_time_l1939_193953


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1939_193904

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1939_193904


namespace NUMINAMATH_CALUDE_cos_equality_proof_l1939_193977

theorem cos_equality_proof (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * π / 180) = Real.cos (317 * π / 180)) : n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l1939_193977


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l1939_193901

theorem largest_prime_divisor_test (n : ℕ) : 
  1000 ≤ n → n ≤ 1100 → 
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l1939_193901


namespace NUMINAMATH_CALUDE_smallest_positive_m_for_symmetry_l1939_193932

open Real

/-- The smallest positive value of m for which the function 
    y = sin(2(x-m) + π/6) is symmetric about the y-axis -/
theorem smallest_positive_m_for_symmetry : 
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x : ℝ), sin (2*(x-m) + π/6) = sin (2*(-x-m) + π/6)) ∧
  (∀ (m' : ℝ), 0 < m' ∧ m' < m → 
    ∃ (x : ℝ), sin (2*(x-m') + π/6) ≠ sin (2*(-x-m') + π/6)) ∧
  m = π/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_m_for_symmetry_l1939_193932


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1939_193900

/-- Given two vectors a and b in ℝ², where a = (-√3, 1) and b = (1, x),
    if a and b are perpendicular, then x = √3. -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-Real.sqrt 3, 1)
  let b : ℝ × ℝ := (1, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1939_193900


namespace NUMINAMATH_CALUDE_inequality_proof_l1939_193921

theorem inequality_proof (a b c d x y u v : ℝ) (h : a * b * c * d > 0) :
  (a * x + b * u) * (a * v + b * y) * (c * x + d * v) * (c * u + d * y) ≥ 
  (a * c * u * v * x + b * c * u * x * y + a * d * v * x * y + b * d * u * v * y) * 
  (a * c * x + b * c * u + a * d * v + b * d * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1939_193921


namespace NUMINAMATH_CALUDE_constant_zero_unique_solution_l1939_193936

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the derivative of a function
noncomputable def derivative (f : RealFunction) : RealFunction :=
  λ x => deriv f x

-- State the theorem
theorem constant_zero_unique_solution :
  ∃! f : RealFunction, ∀ x : ℝ, f x = derivative f x ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_constant_zero_unique_solution_l1939_193936


namespace NUMINAMATH_CALUDE_min_value_of_some_expression_l1939_193906

-- Define the expression
def expression (x : ℝ) (some_expression : ℝ) : ℝ :=
  |x - 4| + |x + 5| + |some_expression|

-- State the theorem
theorem min_value_of_some_expression :
  ∃ (some_expression : ℝ), 
    (∀ x : ℝ, expression x some_expression ≥ 10) ∧ 
    (∃ x : ℝ, expression x some_expression = 10) ∧
    |some_expression| = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_some_expression_l1939_193906


namespace NUMINAMATH_CALUDE_a_2n_is_perfect_square_l1939_193961

/-- Define a function that counts the number of natural numbers with a given digit sum,
    where each digit can only be 1, 3, or 4 -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, a(2n) is a perfect square -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_a_2n_is_perfect_square_l1939_193961


namespace NUMINAMATH_CALUDE_existence_of_coprime_sum_l1939_193922

theorem existence_of_coprime_sum (n k : ℕ) (hn : n > 0) (hk : Even (k * (n - 1))) :
  ∃ x y : ℤ, (Nat.gcd x.natAbs n = 1) ∧ (Nat.gcd y.natAbs n = 1) ∧ ((x + y) % n = k % n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_coprime_sum_l1939_193922


namespace NUMINAMATH_CALUDE_basketball_weight_is_20_l1939_193974

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 20

/-- The weight of one bicycle in pounds -/
def bicycle_weight : ℝ := 40

/-- Theorem stating that one basketball weighs 20 pounds given the conditions -/
theorem basketball_weight_is_20 :
  (8 * basketball_weight = 4 * bicycle_weight) →
  (3 * bicycle_weight = 120) →
  basketball_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_basketball_weight_is_20_l1939_193974


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_zero_l1939_193982

theorem sum_of_powers_equals_zero :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 + (-1)^2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_zero_l1939_193982


namespace NUMINAMATH_CALUDE_no_integer_solution_exists_l1939_193915

theorem no_integer_solution_exists (a b : ℤ) : 
  ∃ c : ℤ, ∀ m n : ℤ, m^2 + a*m + b ≠ 2*n^2 + 2*n + c :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_exists_l1939_193915


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l1939_193902

/-- Given plane vectors a and b with specified properties, prove that |2a-b| = √91 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ) :
  (Real.cos (5 * Real.pi / 6) = a.1 * b.1 + a.2 * b.2) →  -- angle between a and b is 5π/6
  (a.1^2 + a.2^2 = 16) →  -- |a| = 4
  (b.1^2 + b.2^2 = 3) →  -- |b| = √3
  ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 = 91) :=  -- |2a-b| = √91
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l1939_193902


namespace NUMINAMATH_CALUDE_mr_a_loss_l1939_193911

/-- Represents the house transaction between Mr. A and Mr. B -/
def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (rent : ℝ) (gain_percent : ℝ) : ℝ :=
  let sale_price := initial_value * (1 - loss_percent)
  let repurchase_price := sale_price * (1 + gain_percent)
  repurchase_price - initial_value

/-- Theorem stating that Mr. A loses $144 in the transaction -/
theorem mr_a_loss :
  house_transaction 12000 0.12 1000 0.15 = 144 := by
  sorry

end NUMINAMATH_CALUDE_mr_a_loss_l1939_193911


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l1939_193995

theorem polynomial_product_sum (a b c d e : ℝ) : 
  (∀ x : ℝ, (3 * x^3 - 5 * x^2 + 4 * x - 6) * (7 - 2 * x) = a * x^4 + b * x^3 + c * x^2 + d * x + e) →
  16 * a + 8 * b + 4 * c + 2 * d + e = 42 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l1939_193995


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1939_193937

theorem cousins_ages_sum : ∃ (a b c d : ℕ), 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧  -- single-digit
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧      -- positive
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  ((a * b = 24 ∧ c * d = 35) ∨ (a * c = 24 ∧ b * d = 35) ∨ 
   (a * d = 24 ∧ b * c = 35) ∨ (b * c = 24 ∧ a * d = 35) ∨ 
   (b * d = 24 ∧ a * c = 35) ∨ (c * d = 24 ∧ a * b = 35)) →
  a + b + c + d = 23 := by
sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l1939_193937


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l1939_193978

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 16 * x + 16 = (r * x + s)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l1939_193978


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1939_193949

theorem max_sum_of_squares (p q r s : ℝ) : 
  p + q = 18 →
  p * q + r + s = 85 →
  p * r + q * s = 190 →
  r * s = 120 →
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1939_193949


namespace NUMINAMATH_CALUDE_common_ratio_satisfies_cubic_cubic_solution_approx_l1939_193951

/-- A geometric progression with positive terms where each term is the sum of the next three terms -/
structure GeometricProgressionWithSumProperty where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a geometric progression with the sum property satisfies a cubic equation -/
theorem common_ratio_satisfies_cubic (gp : GeometricProgressionWithSumProperty) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 :=
sorry

/-- The positive real solution to the cubic equation x³ + x² + x - 1 = 0 is approximately 0.5437 -/
theorem cubic_solution_approx :
  ∃ x : ℝ, x > 0 ∧ x^3 + x^2 + x - 1 = 0 ∧ abs (x - 0.5437) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_satisfies_cubic_cubic_solution_approx_l1939_193951


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1939_193954

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, ax - b > 0 ↔ x > 1) →
  (∀ x, (a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1939_193954


namespace NUMINAMATH_CALUDE_surface_integral_I_value_surface_integral_J_value_surface_integral_K_value_l1939_193981

-- Define the surface integrals
def surface_integral_I (a : ℝ) : ℝ := sorry

def surface_integral_J : ℝ := sorry

def surface_integral_K : ℝ := sorry

-- State the theorems to be proved
theorem surface_integral_I_value (a : ℝ) (h : a > 0) : 
  surface_integral_I a = (4 * Real.pi / 5) * a^(5/2) := by sorry

theorem surface_integral_J_value : 
  surface_integral_J = (8 * Real.pi / 3) - (4 / 15) := by sorry

theorem surface_integral_K_value : 
  surface_integral_K = -1 / 6 := by sorry

end NUMINAMATH_CALUDE_surface_integral_I_value_surface_integral_J_value_surface_integral_K_value_l1939_193981


namespace NUMINAMATH_CALUDE_eliza_walking_distance_l1939_193933

/-- Proves that Eliza walked 4.5 kilometers given the conditions of the problem -/
theorem eliza_walking_distance :
  ∀ (total_time : ℝ) (rollerblade_speed : ℝ) (walk_speed : ℝ) (distance : ℝ),
    total_time = 1.5 →  -- 90 minutes converted to hours
    rollerblade_speed = 12 →
    walk_speed = 4 →
    (distance / rollerblade_speed) + (distance / walk_speed) = total_time →
    distance = 4.5 := by
  sorry

#check eliza_walking_distance

end NUMINAMATH_CALUDE_eliza_walking_distance_l1939_193933


namespace NUMINAMATH_CALUDE_equation_consequences_l1939_193924

theorem equation_consequences (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (-2 : ℝ) ≤ x + y ∧ x + y ≤ 2 ∧ 2/3 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_consequences_l1939_193924
