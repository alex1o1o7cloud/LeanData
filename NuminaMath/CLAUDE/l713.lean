import Mathlib

namespace NUMINAMATH_CALUDE_largest_base7_to_base3_l713_71337

/-- Converts a number from base 7 to base 10 -/
def base7ToDecimal (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- Converts a number from base 10 to base 3 -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

/-- The largest three-digit number in base 7 -/
def largestBase7 : Nat := 666

theorem largest_base7_to_base3 :
  decimalToBase3 (base7ToDecimal largestBase7) = [1, 1, 0, 2, 0, 0] := by
  sorry

#eval decimalToBase3 (base7ToDecimal largestBase7)

end NUMINAMATH_CALUDE_largest_base7_to_base3_l713_71337


namespace NUMINAMATH_CALUDE_cans_given_away_equals_2500_l713_71319

/-- Represents the food bank's inventory and distribution --/
structure FoodBank where
  initialStock : Nat
  day1People : Nat
  day1CansPerPerson : Nat
  day1Restock : Nat
  day2People : Nat
  day2CansPerPerson : Nat
  day2Restock : Nat

/-- Calculates the total number of cans given away --/
def totalCansGivenAway (fb : FoodBank) : Nat :=
  fb.day1People * fb.day1CansPerPerson + fb.day2People * fb.day2CansPerPerson

/-- Theorem stating that given the specific conditions, 2500 cans were given away --/
theorem cans_given_away_equals_2500 (fb : FoodBank) 
  (h1 : fb.initialStock = 2000)
  (h2 : fb.day1People = 500)
  (h3 : fb.day1CansPerPerson = 1)
  (h4 : fb.day1Restock = 1500)
  (h5 : fb.day2People = 1000)
  (h6 : fb.day2CansPerPerson = 2)
  (h7 : fb.day2Restock = 3000) :
  totalCansGivenAway fb = 2500 := by
  sorry

end NUMINAMATH_CALUDE_cans_given_away_equals_2500_l713_71319


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l713_71344

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l713_71344


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l713_71327

/-- Calculates the average speed of a round trip given the specified conditions -/
theorem round_trip_average_speed
  (outbound_distance : ℝ)
  (outbound_time : ℝ)
  (return_distance : ℝ)
  (return_speed : ℝ)
  (h1 : outbound_distance = 5)
  (h2 : outbound_time = 1)
  (h3 : return_distance = outbound_distance)
  (h4 : return_speed = 20)
  : (outbound_distance + return_distance) / (outbound_time + return_distance / return_speed) = 8 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l713_71327


namespace NUMINAMATH_CALUDE_range_of_f_l713_71354

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 1 ≤ y ∧ y ≤ 5} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l713_71354


namespace NUMINAMATH_CALUDE_john_double_sam_age_l713_71346

/-- The number of years until John is twice as old as Sam -/
def years_until_double : ℕ := 9

/-- Sam's current age -/
def sam_age : ℕ := 9

/-- John's current age -/
def john_age : ℕ := 3 * sam_age

theorem john_double_sam_age :
  john_age + years_until_double = 2 * (sam_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_john_double_sam_age_l713_71346


namespace NUMINAMATH_CALUDE_complex_magnitude_l713_71349

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = 1 - 7 * Complex.I) : 
  Complex.abs z = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l713_71349


namespace NUMINAMATH_CALUDE_james_savings_proof_l713_71320

def weekly_allowance : ℕ := 10
def savings_weeks : ℕ := 4
def video_game_fraction : ℚ := 1 / 2
def book_fraction : ℚ := 1 / 4

theorem james_savings_proof :
  let total_savings := weekly_allowance * savings_weeks
  let after_video_game := total_savings - (video_game_fraction * total_savings)
  let final_amount := after_video_game - (book_fraction * after_video_game)
  final_amount = 15 := by sorry

end NUMINAMATH_CALUDE_james_savings_proof_l713_71320


namespace NUMINAMATH_CALUDE_fraction_arithmetic_l713_71372

theorem fraction_arithmetic : (1/2 - 1/6) / (1/6009 : ℚ) = 2003 := by
  sorry

end NUMINAMATH_CALUDE_fraction_arithmetic_l713_71372


namespace NUMINAMATH_CALUDE_book_arrangement_l713_71376

theorem book_arrangement (k m n : ℕ) :
  (∃ (f : ℕ → ℕ), f 0 = 3 * k.factorial * m.factorial * n.factorial) ∧
  (∃ (g : ℕ → ℕ), g 0 = (m + n).factorial * (m + n + 1) * k.factorial) := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l713_71376


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l713_71316

/-- The parabola y = 2x^2 intersects the line y = x - 4 at exactly one point when
    shifted right by p units or down by q units, where p = q = 31/8 -/
theorem parabola_line_intersection (p q : ℝ) : 
  (∀ x y : ℝ, y = 2*(x - p)^2 ∧ y = x - 4 → (∃! z : ℝ, z = x)) ∧
  (∀ x y : ℝ, y = 2*x^2 - q ∧ y = x - 4 → (∃! z : ℝ, z = x)) →
  p = 31/8 ∧ q = 31/8 := by
sorry


end NUMINAMATH_CALUDE_parabola_line_intersection_l713_71316


namespace NUMINAMATH_CALUDE_trolley_passengers_l713_71325

/-- The number of people on a trolley after three stops -/
def people_on_trolley (initial_pickup : ℕ) (second_stop_off : ℕ) (second_stop_on : ℕ) 
  (third_stop_off : ℕ) (third_stop_on : ℕ) : ℕ :=
  initial_pickup - second_stop_off + second_stop_on - third_stop_off + third_stop_on

/-- Theorem stating the number of people on the trolley after three stops -/
theorem trolley_passengers : 
  people_on_trolley 10 3 20 18 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_trolley_passengers_l713_71325


namespace NUMINAMATH_CALUDE_new_average_income_l713_71303

/-- Given a family with 3 earning members and an average monthly income,
    calculate the new average income after one member passes away. -/
theorem new_average_income
  (initial_members : ℕ)
  (initial_average : ℚ)
  (deceased_income : ℚ)
  (h1 : initial_members = 3)
  (h2 : initial_average = 735)
  (h3 : deceased_income = 905) :
  let total_income := initial_members * initial_average
  let remaining_income := total_income - deceased_income
  let remaining_members := initial_members - 1
  remaining_income / remaining_members = 650 := by
sorry

end NUMINAMATH_CALUDE_new_average_income_l713_71303


namespace NUMINAMATH_CALUDE_solve_equation_l713_71393

theorem solve_equation (Z : ℝ) (h : (19 + 43 / Z) * Z = 2912) : Z = 151 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l713_71393


namespace NUMINAMATH_CALUDE_max_profit_at_upper_bound_l713_71326

/-- Represents the profit function for a product given its cost price, initial selling price,
    initial daily sales, and the rate of sales decrease per yuan increase in price. -/
def profit_function (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (sales_decrease_rate : ℝ) (x : ℝ) : ℝ :=
  (x - cost_price) * (initial_sales - (x - initial_price) * sales_decrease_rate)

/-- Theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_at_upper_bound (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) 
    (sales_decrease_rate : ℝ) (lower_bound : ℝ) (upper_bound : ℝ) :
    cost_price = 30 →
    initial_price = 40 →
    initial_sales = 600 →
    sales_decrease_rate = 10 →
    lower_bound = 40 →
    upper_bound = 60 →
    (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound →
      profit_function cost_price initial_price initial_sales sales_decrease_rate x ≤
      profit_function cost_price initial_price initial_sales sales_decrease_rate upper_bound) ∧
    profit_function cost_price initial_price initial_sales sales_decrease_rate upper_bound = 12000 :=
  sorry

#check max_profit_at_upper_bound

end NUMINAMATH_CALUDE_max_profit_at_upper_bound_l713_71326


namespace NUMINAMATH_CALUDE_sequence_limit_is_two_l713_71391

/-- The limit of the sequence √(n(n+2)) - √(n^2 - 2n + 3) as n approaches infinity is 2 -/
theorem sequence_limit_is_two :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |Real.sqrt (n * (n + 2)) - Real.sqrt (n^2 - 2*n + 3) - 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_sequence_limit_is_two_l713_71391


namespace NUMINAMATH_CALUDE_height_of_smaller_cone_is_9_l713_71310

/-- The height of the smaller cone removed from a right circular cone to create a frustum -/
def height_of_smaller_cone (frustum_height : ℝ) (larger_base_area : ℝ) (smaller_base_area : ℝ) : ℝ :=
  sorry

theorem height_of_smaller_cone_is_9 :
  height_of_smaller_cone 18 (324 * Real.pi) (36 * Real.pi) = 9 := by
  sorry

end NUMINAMATH_CALUDE_height_of_smaller_cone_is_9_l713_71310


namespace NUMINAMATH_CALUDE_pencil_profit_l713_71379

def pencil_problem (pencils : ℕ) (buy_price : ℚ) (sell_price : ℚ) : Prop :=
  let cost := (pencils : ℚ) * buy_price / 4
  let revenue := (pencils : ℚ) * sell_price / 5
  let profit := revenue - cost
  profit = 60

theorem pencil_profit : 
  pencil_problem 1200 3 4 :=
sorry

end NUMINAMATH_CALUDE_pencil_profit_l713_71379


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l713_71369

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents the number of painted faces on a unit cube -/
def painted_faces (c : Cube 4) (unit_cube : Fin 64) : ℕ := sorry

theorem unpainted_cubes_count (c : Cube 4) :
  (Finset.univ.filter (fun unit_cube => painted_faces c unit_cube = 0)).card = 58 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l713_71369


namespace NUMINAMATH_CALUDE_shirt_cost_l713_71380

/-- Given the cost equations for jeans, shirts, and hats, prove the cost of a shirt. -/
theorem shirt_cost (j s h : ℚ) 
  (eq1 : 3 * j + 2 * s + h = 89)
  (eq2 : 2 * j + 3 * s + 2 * h = 102)
  (eq3 : 4 * j + s + 3 * h = 125) :
  s = 12.53 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l713_71380


namespace NUMINAMATH_CALUDE_square_root_and_cube_root_l713_71355

theorem square_root_and_cube_root : 
  (∃ x : ℝ, x^2 = 16 ∧ (x = 4 ∨ x = -4)) ∧ 
  (∃ y : ℝ, y^3 = -2 ∧ y = -Real.rpow 2 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_cube_root_l713_71355


namespace NUMINAMATH_CALUDE_at_least_two_unusual_numbers_l713_71332

/-- A hundred-digit number is unusual if its cube ends with itself but its square does not. -/
def IsUnusual (n : ℕ) : Prop :=
  n ^ 3 % 10^100 = n % 10^100 ∧ n ^ 2 % 10^100 ≠ n % 10^100

/-- There are at least two hundred-digit unusual numbers. -/
theorem at_least_two_unusual_numbers : ∃ n₁ n₂ : ℕ,
  n₁ ≠ n₂ ∧
  10^99 ≤ n₁ ∧ n₁ < 10^100 ∧
  10^99 ≤ n₂ ∧ n₂ < 10^100 ∧
  IsUnusual n₁ ∧ IsUnusual n₂ := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_unusual_numbers_l713_71332


namespace NUMINAMATH_CALUDE_range_of_a_l713_71317

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l713_71317


namespace NUMINAMATH_CALUDE_woodworker_tables_l713_71370

/-- Proves the number of tables made by a woodworker given the total number of furniture legs and chairs made -/
theorem woodworker_tables (total_legs : ℕ) (chairs : ℕ) : 
  total_legs = 40 → 
  chairs = 6 → 
  ∃ (tables : ℕ), 
    tables * 4 + chairs * 4 = total_legs ∧ 
    tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_tables_l713_71370


namespace NUMINAMATH_CALUDE_kalebs_savings_l713_71313

/-- Kaleb's initial savings problem -/
theorem kalebs_savings : ∀ (x : ℕ), 
  (x + 25 = 8 * 8) → x = 39 := by sorry

end NUMINAMATH_CALUDE_kalebs_savings_l713_71313


namespace NUMINAMATH_CALUDE_max_k_for_inequality_l713_71323

theorem max_k_for_inequality : 
  (∃ k : ℤ, ∀ x y : ℝ, x > 0 → y > 0 → 4 * x^2 + 9 * y^2 ≥ 2^k * x * y) ∧ 
  (∀ k : ℤ, k > 3 → ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x^2 + 9 * y^2 < 2^k * x * y) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_inequality_l713_71323


namespace NUMINAMATH_CALUDE_S_excludes_A_and_B_only_l713_71348

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 - 1)^2 + (p.2 - 1)^2) * ((p.1 - 2)^2 + (p.2 + 2)^2) ≠ 0}

theorem S_excludes_A_and_B_only :
  ∀ p : ℝ × ℝ, p ∉ S ↔ p = A ∨ p = B := by sorry

end NUMINAMATH_CALUDE_S_excludes_A_and_B_only_l713_71348


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l713_71318

theorem reciprocal_roots_quadratic (a b : ℂ) : 
  (a^2 + 4*a + 8 = 0) ∧ (b^2 + 4*b + 8 = 0) → 
  (8*(1/a)^2 + 4*(1/a) + 1 = 0) ∧ (8*(1/b)^2 + 4*(1/b) + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l713_71318


namespace NUMINAMATH_CALUDE_total_collection_is_32_49_l713_71334

/-- Represents the number of members in the group -/
def group_size : ℕ := 57

/-- Represents the contribution of each member in paise -/
def contribution_per_member : ℕ := group_size

/-- Converts paise to rupees -/
def paise_to_rupees (paise : ℕ) : ℚ :=
  (paise : ℚ) / 100

/-- Calculates the total collection amount in rupees -/
def total_collection : ℚ :=
  paise_to_rupees (group_size * contribution_per_member)

/-- Theorem stating that the total collection amount is 32.49 rupees -/
theorem total_collection_is_32_49 :
  total_collection = 32.49 := by sorry

end NUMINAMATH_CALUDE_total_collection_is_32_49_l713_71334


namespace NUMINAMATH_CALUDE_time_interval_is_two_seconds_l713_71361

/-- The time interval for birth and death rates in a city --/
def time_interval (birth_rate death_rate net_increase_per_day seconds_per_day : ℕ) : ℚ :=
  seconds_per_day / (net_increase_per_day / (birth_rate - death_rate))

/-- Theorem: The time interval for birth and death rates is 2 seconds --/
theorem time_interval_is_two_seconds :
  time_interval 4 2 86400 86400 = 2 := by
  sorry

#eval time_interval 4 2 86400 86400

end NUMINAMATH_CALUDE_time_interval_is_two_seconds_l713_71361


namespace NUMINAMATH_CALUDE_total_earnings_l713_71392

/-- Proves that the total amount earned by 5 men, W women, and 8 boys is 210 rupees -/
theorem total_earnings (W : ℕ) (mens_wage : ℕ) 
  (h1 : 5 = W)  -- 5 men are equal to W women
  (h2 : W = 8)  -- W women are equal to 8 boys
  (h3 : mens_wage = 14)  -- Men's wages are Rs. 14 each
  : 5 * mens_wage + W * mens_wage + 8 * mens_wage = 210 := by
  sorry

#eval 5 * 14 + 5 * 14 + 8 * 14  -- Evaluates to 210

end NUMINAMATH_CALUDE_total_earnings_l713_71392


namespace NUMINAMATH_CALUDE_certain_number_proof_l713_71381

theorem certain_number_proof (p q x : ℝ) 
  (h1 : 3 / p = x)
  (h2 : 3 / q = 15)
  (h3 : p - q = 0.3) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l713_71381


namespace NUMINAMATH_CALUDE_cat_sale_theorem_l713_71378

/-- Represents the count of cats for each breed -/
structure CatCounts where
  siamese : Nat
  persian : Nat
  house : Nat
  maineCoon : Nat

/-- Represents the number of pairs sold for each breed -/
structure SoldPairs where
  siamese : Nat
  persian : Nat
  maineCoon : Nat

/-- Calculates the remaining cats after the sale -/
def remainingCats (initial : CatCounts) (sold : SoldPairs) : CatCounts :=
  { siamese := initial.siamese - sold.siamese,
    persian := initial.persian - sold.persian,
    house := initial.house,
    maineCoon := initial.maineCoon - sold.maineCoon }

theorem cat_sale_theorem (initial : CatCounts) (sold : SoldPairs) :
  initial.siamese = 25 →
  initial.persian = 18 →
  initial.house = 12 →
  initial.maineCoon = 10 →
  sold.siamese = 6 →
  sold.persian = 4 →
  sold.maineCoon = 3 →
  let remaining := remainingCats initial sold
  remaining.siamese = 19 ∧
  remaining.persian = 14 ∧
  remaining.house = 12 ∧
  remaining.maineCoon = 7 :=
by sorry

end NUMINAMATH_CALUDE_cat_sale_theorem_l713_71378


namespace NUMINAMATH_CALUDE_people_per_column_second_arrangement_l713_71367

/-- 
Given a group of people that can be arranged in two ways:
1. 16 columns with 30 people per column
2. 8 columns with an unknown number of people per column

This theorem proves that the number of people per column in the second arrangement is 60.
-/
theorem people_per_column_second_arrangement 
  (total_people : ℕ) 
  (columns_first : ℕ) 
  (people_per_column_first : ℕ) 
  (columns_second : ℕ) : 
  columns_first = 16 → 
  people_per_column_first = 30 → 
  columns_second = 8 → 
  total_people = columns_first * people_per_column_first → 
  total_people / columns_second = 60 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_second_arrangement_l713_71367


namespace NUMINAMATH_CALUDE_positive_integer_solution_is_perfect_square_l713_71339

theorem positive_integer_solution_is_perfect_square (t : ℤ) (n : ℕ+) 
  (h : n^2 + (4*t - 1)*n + 4*t^2 = 0) : 
  ∃ (k : ℕ), n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_is_perfect_square_l713_71339


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_one_l713_71364

theorem no_solution_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (3 - 2*x)/(x - 3) + (2 + m*x)/(3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_one_l713_71364


namespace NUMINAMATH_CALUDE_hyperbola_equation_l713_71321

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) →
  (∃ x₀ : ℝ, x₀ = 5 ∧ (∀ y : ℝ, y^2 = 20*x₀ → (x₀^2 / a^2 - y^2 / b^2 = 1))) →
  a^2 = 5 ∧ b^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l713_71321


namespace NUMINAMATH_CALUDE_quadratic_condition_l713_71307

theorem quadratic_condition (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) →
  (a > 0 ∧ b^2 - 4*a*c < 0) ∧
  ¬(a > 0 ∧ b^2 - 4*a*c < 0 → ∀ x : ℝ, a * x^2 + b * x + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l713_71307


namespace NUMINAMATH_CALUDE_book_selection_probability_book_selection_proof_l713_71338

theorem book_selection_probability : ℕ → ℝ
  | 12 => 55 / 209
  | _ => 0

theorem book_selection_proof (n : ℕ) :
  n = 12 →
  (book_selection_probability n) = 
    (Nat.choose n 3 * Nat.choose (n - 3) 2 * Nat.choose (n - 5) 2 : ℝ) / 
    ((Nat.choose n 5 : ℝ) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_book_selection_probability_book_selection_proof_l713_71338


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l713_71365

-- Define the function f
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x, f x = f (-x)) ∧                   -- f is an even function
  (∀ x y, 0 ≤ x ∧ x ≤ y → f y ≤ f x) ∧    -- f is monotonically decreasing on [0,+∞)
  (∀ y, ∃ x, f x = y ↔ y ≤ 0) :=          -- Range of f is (-∞,0]
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l713_71365


namespace NUMINAMATH_CALUDE_added_value_theorem_l713_71382

theorem added_value_theorem (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 8) :
  x + y = 128 * (1/x) → y = 8 := by
sorry

end NUMINAMATH_CALUDE_added_value_theorem_l713_71382


namespace NUMINAMATH_CALUDE_outfits_count_l713_71328

/-- Represents the number of shirts available -/
def num_shirts : Nat := 7

/-- Represents the number of pants available -/
def num_pants : Nat := 5

/-- Represents the number of ties available -/
def num_ties : Nat := 4

/-- Represents the number of jackets available -/
def num_jackets : Nat := 2

/-- Represents the number of tie options (wearing a tie or not) -/
def tie_options : Nat := num_ties + 1

/-- Represents the number of jacket options (wearing a jacket or not) -/
def jacket_options : Nat := num_jackets + 1

/-- Calculates the total number of possible outfits -/
def total_outfits : Nat := num_shirts * num_pants * tie_options * jacket_options

/-- Proves that the total number of possible outfits is 525 -/
theorem outfits_count : total_outfits = 525 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l713_71328


namespace NUMINAMATH_CALUDE_banquet_guests_l713_71329

theorem banquet_guests (x : ℚ) : 
  (1/3 * x + 3/5 * (2/3 * x) + 4 = x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_banquet_guests_l713_71329


namespace NUMINAMATH_CALUDE_cookies_per_tray_l713_71340

/-- Given that Marian baked 276 oatmeal cookies and used 23 trays,
    prove that she can place 12 cookies on a tray at a time. -/
theorem cookies_per_tray (total_cookies : ℕ) (num_trays : ℕ) 
  (h1 : total_cookies = 276) (h2 : num_trays = 23) :
  total_cookies / num_trays = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_tray_l713_71340


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l713_71331

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (2*k + 1)*x + k^2 + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k > 3/4) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ + x₂ = -x₁ * x₂ → k = 2) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equation_roots_l713_71331


namespace NUMINAMATH_CALUDE_fraction_simplification_l713_71311

theorem fraction_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + b^2 + x^2) - (x^2 - a^2 - b^2) / Real.sqrt (a^2 + b^2 + x^2)) / (a^2 + b^2 + x^2) = 
  2 * (a^2 + b^2) / (a^2 + b^2 + x^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l713_71311


namespace NUMINAMATH_CALUDE_line_points_k_value_l713_71341

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 4 = 2 * (n + k) + 5) → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l713_71341


namespace NUMINAMATH_CALUDE_retail_price_calculation_l713_71351

theorem retail_price_calculation (total_cost : ℕ) (price_difference : ℕ) (additional_books : ℕ) :
  total_cost = 48 ∧ price_difference = 2 ∧ additional_books = 4 →
  ∃ (n : ℕ), n > 0 ∧ total_cost / n = 6 ∧ 
  (total_cost / n - price_difference) * (n + additional_books) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l713_71351


namespace NUMINAMATH_CALUDE_new_average_age_l713_71350

def initial_people : ℕ := 8
def initial_average_age : ℚ := 35
def leaving_person_age : ℕ := 25
def remaining_people : ℕ := 7

theorem new_average_age :
  let total_age : ℚ := initial_people * initial_average_age
  let remaining_age : ℚ := total_age - leaving_person_age
  remaining_age / remaining_people = 36.42857 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l713_71350


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l713_71360

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 4*x - 15) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l713_71360


namespace NUMINAMATH_CALUDE_circle_radius_l713_71353

theorem circle_radius (P Q : ℝ) (h : P / Q = 15) : 
  ∃ r : ℝ, r > 0 ∧ P = π * r^2 ∧ Q = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l713_71353


namespace NUMINAMATH_CALUDE_possible_teams_count_l713_71302

/-- Represents the number of players in each position in the squad --/
structure SquadComposition :=
  (goalkeepers : Nat)
  (defenders : Nat)
  (midfielders : Nat)
  (strikers : Nat)

/-- Represents the required composition of a team --/
structure TeamComposition :=
  (goalkeepers : Nat)
  (defenders : Nat)
  (midfielders : Nat)
  (strikers : Nat)

/-- Function to calculate the number of possible teams --/
def calculatePossibleTeams (squad : SquadComposition) (team : TeamComposition) : Nat :=
  let goalkeeperChoices := Nat.choose squad.goalkeepers team.goalkeepers
  let strikerChoices := Nat.choose squad.strikers team.strikers
  let midfielderChoices := Nat.choose squad.midfielders team.midfielders
  let defenderChoices := Nat.choose (squad.defenders + (squad.midfielders - team.midfielders)) team.defenders
  goalkeeperChoices * strikerChoices * midfielderChoices * defenderChoices

/-- Theorem stating the number of possible teams --/
theorem possible_teams_count (squad : SquadComposition) (team : TeamComposition) :
  squad.goalkeepers = 3 →
  squad.defenders = 5 →
  squad.midfielders = 5 →
  squad.strikers = 5 →
  team.goalkeepers = 1 →
  team.defenders = 4 →
  team.midfielders = 4 →
  team.strikers = 2 →
  calculatePossibleTeams squad team = 2250 := by
  sorry

end NUMINAMATH_CALUDE_possible_teams_count_l713_71302


namespace NUMINAMATH_CALUDE_angle_rotation_l713_71357

def first_quadrant (α : Real) : Prop :=
  0 < α ∧ α < Real.pi / 2

def third_quadrant (α : Real) : Prop :=
  Real.pi < α ∧ α < 3 * Real.pi / 2

theorem angle_rotation (α : Real) :
  first_quadrant α → third_quadrant (α + Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_angle_rotation_l713_71357


namespace NUMINAMATH_CALUDE_new_person_weight_l713_71345

theorem new_person_weight
  (initial_count : ℕ)
  (average_increase : ℝ)
  (replaced_weight : ℝ)
  (hcount : initial_count = 7)
  (hincrease : average_increase = 3.5)
  (hreplaced : replaced_weight = 75)
  : ℝ :=
by
  -- The weight of the new person
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_new_person_weight_l713_71345


namespace NUMINAMATH_CALUDE_product_of_numbers_l713_71390

theorem product_of_numbers (x y : ℝ) : 
  x - y = 12 → x^2 + y^2 = 250 → x * y = 52.7364 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l713_71390


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l713_71397

def x : ℕ := 5 * 18 * 36

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  y > 0 ∧ 
  ∃ (n : ℕ), x * y = n^3 ∧
  ∀ (z : ℕ), z > 0 → (∃ (m : ℕ), x * z = m^3) → y ≤ z
  ↔ y = 225 := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l713_71397


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l713_71330

/-- A quadratic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + b + 2

/-- The theorem statement -/
theorem quadratic_function_properties (a b : ℝ) :
  a > 0 →
  (∀ x ∈ Set.Icc 0 1, f a b x ≤ f a b 0) →
  (∀ x ∈ Set.Icc 0 1, f a b x ≥ f a b 1) →
  f a b 0 - f a b 1 = 3 →
  f a b 1 = 0 →
  a = 3 ∧ b = 1 ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc (1/3) 2, f a b x < m * x^2 + 1) ↔ m > 3) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l713_71330


namespace NUMINAMATH_CALUDE_water_percentage_fresh_is_75_percent_l713_71375

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 75

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 25

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 200

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 66.67

/-- Theorem stating that the percentage of water in fresh grapes is 75% -/
theorem water_percentage_fresh_is_75_percent :
  water_percentage_fresh = 75 := by sorry

end NUMINAMATH_CALUDE_water_percentage_fresh_is_75_percent_l713_71375


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l713_71387

/-- Given a complex number z satisfying z⋅2i = |z|² + 1, prove that |z| = 1 -/
theorem complex_magnitude_one (z : ℂ) (h : z * (2 * Complex.I) = Complex.abs z ^ 2 + 1) :
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l713_71387


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l713_71371

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 21.5

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 2

/-- The number of days for which pills are taken -/
def days : ℕ := 18

/-- The total cost of all pills over the given period -/
def total_cost : ℝ := 738

theorem green_pill_cost_proof :
  (green_pill_cost + pink_pill_cost) * days = total_cost ∧
  green_pill_cost = pink_pill_cost + 2 ∧
  green_pill_cost = 21.5 :=
sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l713_71371


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l713_71308

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 + 17*x - 72 = 0) → (x = -24 ∨ x = 3) → x = min (-24) 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l713_71308


namespace NUMINAMATH_CALUDE_walking_speed_l713_71366

theorem walking_speed (distance : Real) (time_minutes : Real) (speed : Real) : 
  distance = 500 ∧ time_minutes = 6 → speed = 5000 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_l713_71366


namespace NUMINAMATH_CALUDE_contrapositive_of_p_is_true_l713_71305

theorem contrapositive_of_p_is_true :
  (∀ x : ℝ, x^2 - 2*x - 8 ≤ 0 → x ≥ -3) := by sorry

end NUMINAMATH_CALUDE_contrapositive_of_p_is_true_l713_71305


namespace NUMINAMATH_CALUDE_power_of_256_l713_71342

theorem power_of_256 : (256 : ℝ) ^ (5/4 : ℝ) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_of_256_l713_71342


namespace NUMINAMATH_CALUDE_imaginary_power_sum_zero_l713_71352

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum_zero : 
  i^14762 + i^14763 + i^14764 + i^14765 = 0 :=
by
  sorry

-- Define the property of i
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_imaginary_power_sum_zero_l713_71352


namespace NUMINAMATH_CALUDE_set_equivalence_l713_71362

theorem set_equivalence : 
  {x : ℕ | 8 < x ∧ x < 12} = {9, 10, 11} := by
sorry

end NUMINAMATH_CALUDE_set_equivalence_l713_71362


namespace NUMINAMATH_CALUDE_range_of_f_l713_71389

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem
theorem range_of_f :
  let S := {y : ℝ | ∃ x : ℝ, x ≠ -8 ∧ f x = y}
  S = {y : ℝ | y < -36 ∨ y > -36} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l713_71389


namespace NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l713_71322

theorem second_smallest_hot_dog_packs : 
  (∃ n : ℕ, n > 0 ∧ (12 * n) % 8 = (8 - 7) % 8 ∧ 
   (∀ m : ℕ, m > 0 ∧ m < n → (12 * m) % 8 ≠ (8 - 7) % 8)) → 
  (∃ n : ℕ, n > 0 ∧ (12 * n) % 8 = (8 - 7) % 8 ∧ 
   (∃! m : ℕ, m > 0 ∧ m < n ∧ (12 * m) % 8 = (8 - 7) % 8) ∧ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l713_71322


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l713_71368

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 + 5*x₁ + 1 = 0 →
  x₂^2 + 5*x₂ + 1 = 0 →
  (x₁*Real.sqrt 6 / (1 + x₂))^2 + (x₂*Real.sqrt 6 / (1 + x₁))^2 = 220 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l713_71368


namespace NUMINAMATH_CALUDE_lcm_5_6_10_12_l713_71347

theorem lcm_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_10_12_l713_71347


namespace NUMINAMATH_CALUDE_solution_system_l713_71324

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_solution_system_l713_71324


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l713_71312

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  ∃ k : ℕ, n + 1 = 4 * k ∧
  ∃ l : ℕ, n + 1 = 5 * l ∧
  ∃ m : ℕ, n + 1 = 6 * m ∧
  ∃ p : ℕ, n + 1 = 8 * p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l713_71312


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l713_71383

-- Define the expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^7 * z^9) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^3 * (5 * x^2 * y) ^ (1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 1 + 3

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  original_expression x y z = simplified_expression x y z ∧
  sum_of_exponents = 5 := by sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l713_71383


namespace NUMINAMATH_CALUDE_line_inclination_range_l713_71396

-- Define the line equation
def line_equation (x y α : ℝ) : Prop := x * Real.cos α + Real.sqrt 3 * y + 2 = 0

-- Define the range of α
def α_range (α : ℝ) : Prop := 0 ≤ α ∧ α < Real.pi

-- Define the range of the inclination angle θ
def inclination_range (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

-- Theorem statement
theorem line_inclination_range :
  ∀ α : ℝ, α_range α →
  ∃ θ : ℝ, inclination_range θ ∧
  ∀ x y : ℝ, line_equation x y α ↔ line_equation x y θ :=
sorry

end NUMINAMATH_CALUDE_line_inclination_range_l713_71396


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l713_71315

theorem quadratic_inequality_range (a b c : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 3 → -1 < a * x^2 + b * x + c ∧ a * x^2 + b * x + c < 1) ∧
  (∀ x, x ∉ Set.Ioo (-1 : ℝ) 3 → a * x^2 + b * x + c ≤ -1 ∨ a * x^2 + b * x + c ≥ 1) →
  -1/2 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l713_71315


namespace NUMINAMATH_CALUDE_solve_for_x_l713_71399

theorem solve_for_x (x : ℤ) : x + 1315 + 9211 - 1569 = 11901 → x = 2944 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l713_71399


namespace NUMINAMATH_CALUDE_log_inequality_implies_x_geq_125_l713_71306

theorem log_inequality_implies_x_geq_125 (x : ℝ) (h1 : x > 0) 
  (h2 : Real.log x / Real.log 3 ≥ Real.log 5 / Real.log 3 + (2/3) * (Real.log x / Real.log 3)) :
  x ≥ 125 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_x_geq_125_l713_71306


namespace NUMINAMATH_CALUDE_chris_donuts_l713_71300

theorem chris_donuts (initial_donuts : ℕ) : 
  (initial_donuts : ℝ) * 0.9 - 4 = 23 → initial_donuts = 30 := by
  sorry

end NUMINAMATH_CALUDE_chris_donuts_l713_71300


namespace NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l713_71336

/-- Two points A(a, 2022) and A'(-2023, b) are symmetric with respect to the origin if and only if
    their coordinates satisfy the given conditions. -/
def symmetric_points (a b : ℝ) : Prop :=
  a = 2023 ∧ b = -2022

/-- The sum of a and b is 1 when A(a, 2022) and A'(-2023, b) are symmetric with respect to the origin. -/
theorem sum_of_coordinates_symmetric_points (a b : ℝ) 
    (h : symmetric_points a b) : a + b = 1 := by
  sorry

#check sum_of_coordinates_symmetric_points

end NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l713_71336


namespace NUMINAMATH_CALUDE_odd_digits_base4_157_l713_71309

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of digits -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 157 is 3 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_157_l713_71309


namespace NUMINAMATH_CALUDE_min_abs_z_is_zero_l713_71388

theorem min_abs_z_is_zero (z : ℂ) (h : Complex.abs (z + 2 - 3*I) + Complex.abs (z - 2*I) = 7) :
  ∃ (w : ℂ), ∀ (z : ℂ), Complex.abs (z + 2 - 3*I) + Complex.abs (z - 2*I) = 7 → Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 0 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_is_zero_l713_71388


namespace NUMINAMATH_CALUDE_solution_implies_c_value_l713_71301

-- Define the function f
def f (x b : ℝ) : ℝ := x^2 + x + b

-- State the theorem
theorem solution_implies_c_value
  (b : ℝ)  -- b is a real number
  (h1 : ∀ x, f x b ≥ 0)  -- Value range of f is [0, +∞)
  (h2 : ∃ m, ∀ x, f x b < 16 ↔ x < m + 8)  -- Solution to f(x) < c is m + 8
  : 16 = 16 :=  -- We want to prove c = 16
by
  sorry

end NUMINAMATH_CALUDE_solution_implies_c_value_l713_71301


namespace NUMINAMATH_CALUDE_dog_eaten_cost_l713_71386

/-- Represents the cost of ingredients for a cake -/
structure CakeIngredients where
  flour : Float
  sugar : Float
  eggs : Float
  butter : Float

/-- Represents the cake and its consumption -/
structure Cake where
  ingredients : CakeIngredients
  totalSlices : Nat
  slicesEatenByMother : Nat

def totalCost (c : CakeIngredients) : Float :=
  c.flour + c.sugar + c.eggs + c.butter

def costPerSlice (cake : Cake) : Float :=
  totalCost cake.ingredients / cake.totalSlices.toFloat

def slicesEatenByDog (cake : Cake) : Nat :=
  cake.totalSlices - cake.slicesEatenByMother

theorem dog_eaten_cost (cake : Cake) 
  (h1 : cake.ingredients = { flour := 4, sugar := 2, eggs := 0.5, butter := 2.5 })
  (h2 : cake.totalSlices = 6)
  (h3 : cake.slicesEatenByMother = 2) :
  costPerSlice cake * (slicesEatenByDog cake).toFloat = 6 := by
  sorry

#check dog_eaten_cost

end NUMINAMATH_CALUDE_dog_eaten_cost_l713_71386


namespace NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l713_71343

/-- The number of days after which the reinforcement arrived -/
def reinforcement_arrival_day : ℕ := 15

/-- The initial number of men in the garrison -/
def initial_garrison : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_provision_days : ℕ := 54

/-- The number of men in the reinforcement -/
def reinforcement : ℕ := 1900

/-- The number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem reinforcement_arrival_theorem :
  initial_garrison * (initial_provision_days - reinforcement_arrival_day) =
  (initial_garrison + reinforcement) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_theorem_l713_71343


namespace NUMINAMATH_CALUDE_area_between_curves_l713_71314

-- Define the two functions
def f (x : ℝ) := 3 * x
def g (x : ℝ) := x^2

-- Define the intersection points
def x₁ : ℝ := 0
def x₂ : ℝ := 3

-- State the theorem
theorem area_between_curves :
  ∫ x in x₁..x₂, (f x - g x) = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l713_71314


namespace NUMINAMATH_CALUDE_stock_price_increase_l713_71384

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (percent_increase : ℝ) (h1 : closing_price = 9) 
  (h2 : percent_increase = 12.5) :
  closing_price = opening_price * (1 + percent_increase / 100) →
  opening_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l713_71384


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l713_71335

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(2 : ℝ) / (x + 3)⌉
  else if x < -3 then
    ⌊(2 : ℝ) / (x + 3)⌋
  else
    0  -- Arbitrary value for x = -3, as g is not defined there

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l713_71335


namespace NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l713_71333

def jungkook_erasers : ℕ := 6

def jimin_erasers (j : ℕ) : ℕ := j + 4

def seokjin_erasers (j : ℕ) : ℕ := j - 3

theorem jungkook_has_fewest_erasers :
  ∀ (j s : ℕ), 
    j = jimin_erasers jungkook_erasers →
    s = seokjin_erasers j →
    jungkook_erasers ≤ j ∧ jungkook_erasers ≤ s :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l713_71333


namespace NUMINAMATH_CALUDE_duck_cow_problem_l713_71377

theorem duck_cow_problem (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 28 → C = 14 := by
sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l713_71377


namespace NUMINAMATH_CALUDE_line_problem_l713_71304

theorem line_problem (front_position back_position total : ℕ) 
  (h1 : front_position = 8)
  (h2 : back_position = 6)
  (h3 : total = front_position + back_position - 1) :
  total = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l713_71304


namespace NUMINAMATH_CALUDE_data_set_median_and_variance_l713_71394

def data_set : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_set_median_and_variance :
  median data_set = 8 ∧ variance data_set = 2.8 := by sorry

end NUMINAMATH_CALUDE_data_set_median_and_variance_l713_71394


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l713_71363

def a (n : ℕ) : ℚ := (1 - 2 * n^2) / (2 + 4 * n^2)

theorem limit_of_sequence_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l713_71363


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l713_71385

/-- The sum of divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number n -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l713_71385


namespace NUMINAMATH_CALUDE_rectangle_C_in_position_I_l713_71358

-- Define the Rectangle type
structure Rectangle where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the five rectangles
def A : Rectangle := ⟨1, 4, 3, 1⟩
def B : Rectangle := ⟨4, 3, 1, 2⟩
def C : Rectangle := ⟨1, 1, 2, 4⟩
def D : Rectangle := ⟨3, 2, 2, 3⟩
def E : Rectangle := ⟨4, 4, 1, 1⟩

-- Define a function to check if two rectangles can be placed side by side
def canPlaceSideBySide (r1 r2 : Rectangle) : Bool :=
  r1.right = r2.left

-- Define a function to check if two rectangles can be placed top to bottom
def canPlaceTopToBottom (r1 r2 : Rectangle) : Bool :=
  r1.bottom = r2.top

-- Theorem: Rectangle C is the only one that can be placed in position I
theorem rectangle_C_in_position_I :
  ∃! r : Rectangle, r = C ∧ 
  (∃ r2 r3 : Rectangle, r2 ≠ r ∧ r3 ≠ r ∧ r2 ≠ r3 ∧
   canPlaceSideBySide r r2 ∧ canPlaceSideBySide r2 r3 ∧
   (∃ r4 : Rectangle, r4 ≠ r ∧ r4 ≠ r2 ∧ r4 ≠ r3 ∧
    canPlaceTopToBottom r r4 ∧
    (∃ r5 : Rectangle, r5 ≠ r ∧ r5 ≠ r2 ∧ r5 ≠ r3 ∧ r5 ≠ r4 ∧
     canPlaceTopToBottom r2 r5 ∧ canPlaceSideBySide r4 r5))) :=
sorry

end NUMINAMATH_CALUDE_rectangle_C_in_position_I_l713_71358


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l713_71356

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l713_71356


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l713_71374

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point2D
  sideLength : ℝ

/-- Represents a division of an equilateral triangle -/
def TriangleDivision (t : EquilateralTriangle) (n : ℕ) :=
  { subdivisions : List EquilateralTriangle // 
    subdivisions.length = n * n ∧
    ∀ sub ∈ subdivisions, sub.sideLength = t.sideLength / n }

/-- Theorem: An equilateral triangle can be divided into 9 smaller congruent equilateral triangles -/
theorem equilateral_triangle_division (t : EquilateralTriangle) : 
  ∃ (div : TriangleDivision t 3), True := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l713_71374


namespace NUMINAMATH_CALUDE_intersection_range_l713_71373

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The theorem stating that if 2x³ - 3x² + (1 + b) = 0 has three distinct real roots, then -1 < b < 0 -/
theorem intersection_range (b : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = -b ∧ f y = -b ∧ f z = -b) →
  -1 < b ∧ b < 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l713_71373


namespace NUMINAMATH_CALUDE_fort_blocks_count_l713_71398

/-- Represents the dimensions of a rectangular fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build a fort with given dimensions and wall thickness --/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  d.length * d.width * d.height - (d.length - 2 * wallThickness) * (d.width - 2 * wallThickness) * (d.height - 2 * wallThickness)

/-- Theorem stating that a fort with dimensions 15x12x6 and wall thickness 1 requires 560 blocks --/
theorem fort_blocks_count : 
  let fortDims : FortDimensions := { length := 15, width := 12, height := 6 }
  blocksNeeded fortDims 1 = 560 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l713_71398


namespace NUMINAMATH_CALUDE_pattern_D_cannot_form_cube_l713_71395

/-- Represents a pattern of squares -/
structure SquarePattern where
  num_squares : ℕ
  is_plus_shape : Bool
  has_extra_square : Bool

/-- Represents the requirements for forming a cube -/
def can_form_cube (pattern : SquarePattern) : Prop :=
  pattern.num_squares = 6 ∧ 
  (pattern.is_plus_shape → ¬pattern.has_extra_square)

/-- Pattern D definition -/
def pattern_D : SquarePattern :=
  { num_squares := 7
  , is_plus_shape := true
  , has_extra_square := true }

/-- Theorem stating that Pattern D cannot form a cube -/
theorem pattern_D_cannot_form_cube : ¬(can_form_cube pattern_D) := by
  sorry


end NUMINAMATH_CALUDE_pattern_D_cannot_form_cube_l713_71395


namespace NUMINAMATH_CALUDE_base_of_power_l713_71359

theorem base_of_power (b : ℝ) (x y : ℤ) 
  (h1 : b^x * 4^y = 531441)
  (h2 : x - y = 12)
  (h3 : x = 12) : 
  b = 3 := by sorry

end NUMINAMATH_CALUDE_base_of_power_l713_71359
