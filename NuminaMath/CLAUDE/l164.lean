import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l164_16408

theorem problem_solution (a : ℝ) (x y : ℝ) 
  (h1 : a ^ (x - y) = 343)
  (h2 : a ^ (x + y) = 16807) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l164_16408


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l164_16467

/-- The range of k for which the quadratic equation (k-1)x^2 + 3x - 1 = 0 has real roots -/
theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -5/4 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l164_16467


namespace NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l164_16429

/-- Sum of digits of a three-digit number -/
def sumOfDigits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- Theorem: Among any 18 consecutive three-digit numbers, there exists one divisible by its sum of digits -/
theorem exists_divisible_by_sum_of_digits (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sumOfDigits k = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_l164_16429


namespace NUMINAMATH_CALUDE_speaker_cost_correct_l164_16421

/-- The amount Keith spent on speakers -/
def speaker_cost : ℚ := 136.01

/-- The amount Keith spent on a CD player -/
def cd_player_cost : ℚ := 139.38

/-- The amount Keith spent on new tires -/
def tire_cost : ℚ := 112.46

/-- The total amount Keith spent -/
def total_cost : ℚ := 387.85

/-- Theorem stating that the speaker cost is correct given the other expenses -/
theorem speaker_cost_correct : 
  speaker_cost = total_cost - (cd_player_cost + tire_cost) := by
  sorry

end NUMINAMATH_CALUDE_speaker_cost_correct_l164_16421


namespace NUMINAMATH_CALUDE_farmer_apples_l164_16452

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l164_16452


namespace NUMINAMATH_CALUDE_width_to_perimeter_ratio_l164_16495

/-- Represents a rectangular classroom -/
structure Classroom where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a classroom -/
def perimeter (c : Classroom) : ℝ := 2 * (c.length + c.width)

/-- Theorem: The ratio of width to perimeter for a 15x10 classroom is 1:5 -/
theorem width_to_perimeter_ratio (c : Classroom) 
  (h1 : c.length = 15) 
  (h2 : c.width = 10) : 
  c.width / perimeter c = 1 / 5 := by
  sorry

#check width_to_perimeter_ratio

end NUMINAMATH_CALUDE_width_to_perimeter_ratio_l164_16495


namespace NUMINAMATH_CALUDE_smallest_angle_is_27_l164_16463

/-- Represents the properties of a circle divided into sectors --/
structure CircleSectors where
  num_sectors : ℕ
  angle_sum : ℕ
  is_arithmetic_sequence : Bool
  all_angles_integer : Bool

/-- Finds the smallest possible sector angle given the circle properties --/
def smallest_sector_angle (circle : CircleSectors) : ℕ :=
  sorry

/-- Theorem stating that for a circle divided into 10 sectors with the given properties,
    the smallest possible sector angle is 27 degrees --/
theorem smallest_angle_is_27 :
  ∀ (circle : CircleSectors),
    circle.num_sectors = 10 ∧
    circle.angle_sum = 360 ∧
    circle.is_arithmetic_sequence = true ∧
    circle.all_angles_integer = true →
    smallest_sector_angle circle = 27 :=
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_27_l164_16463


namespace NUMINAMATH_CALUDE_triangle_side_length_l164_16427

theorem triangle_side_length (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 3 →
  Real.sin A = Real.sqrt 3 / 2 →
  B = π / 6 →
  b = a * Real.sin B / Real.sin A →
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l164_16427


namespace NUMINAMATH_CALUDE_village_burn_time_l164_16487

/-- Represents the number of cottages remaining after n intervals -/
def A : ℕ → ℕ
| 0 => 90
| n + 1 => 2 * A n - 96

/-- The time it takes Trodgor to burn down the village -/
def burnTime : ℕ := 1920

theorem village_burn_time : 
  ∀ n : ℕ, A n = 0 → n * 480 = burnTime := by
  sorry

#check village_burn_time

end NUMINAMATH_CALUDE_village_burn_time_l164_16487


namespace NUMINAMATH_CALUDE_sin_2x_value_l164_16412

theorem sin_2x_value (x : Real) (h : (1 + Real.tan x) / (1 - Real.tan x) = 2) : 
  Real.sin (2 * x) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l164_16412


namespace NUMINAMATH_CALUDE_total_bananas_used_l164_16425

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of loaves made on both days -/
def total_loaves : ℕ := monday_loaves + tuesday_loaves

/-- Theorem: The total number of bananas used is 36 -/
theorem total_bananas_used : bananas_per_loaf * total_loaves = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_bananas_used_l164_16425


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l164_16471

/-- Given that 3/4 of 12 bananas are worth as much as 9 oranges,
    prove that 2/5 of 15 bananas are worth as much as 6 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 12 * banana_value = 9 * orange_value →
  (2/5 : ℚ) * 15 * banana_value = 6 * orange_value := by
sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l164_16471


namespace NUMINAMATH_CALUDE_scientific_notation_of_2590000_l164_16490

theorem scientific_notation_of_2590000 :
  2590000 = 2.59 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2590000_l164_16490


namespace NUMINAMATH_CALUDE_identity_proof_l164_16492

theorem identity_proof (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c)) / ((a - b) * (a - c)) +
  (b^2 * (x - a) * (x - c)) / ((b - a) * (b - c)) +
  (c^2 * (x - a) * (x - b)) / ((c - a) * (c - b)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l164_16492


namespace NUMINAMATH_CALUDE_exists_complete_list_l164_16426

/-- Represents a tournament where each competitor meets every other competitor exactly once with no draws -/
structure Tournament (α : Type*) :=
  (competitors : Set α)
  (beats : α → α → Prop)
  (all_play_once : ∀ x y : α, x ≠ y → (beats x y ∨ beats y x) ∧ ¬(beats x y ∧ beats y x))

/-- The list of players beaten by a given player and those beaten by the players they've beaten -/
def extended_wins (T : Tournament α) (x : α) : Set α :=
  {y | T.beats x y ∨ ∃ z, T.beats x z ∧ T.beats z y}

/-- There exists a player whose extended wins list includes all other players -/
theorem exists_complete_list (T : Tournament α) :
  ∃ x : α, ∀ y : α, y ≠ x → y ∈ extended_wins T x := by
  sorry


end NUMINAMATH_CALUDE_exists_complete_list_l164_16426


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l164_16497

theorem cricket_team_captain_age 
  (team_size : ℕ) 
  (captain_age wicket_keeper_age : ℕ) 
  (team_average_age : ℚ) 
  (remaining_players_average_age : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 7 →
  team_average_age = 23 →
  remaining_players_average_age = team_average_age - 1 →
  (team_size : ℚ) * team_average_age = 
    ((team_size - 2) : ℚ) * remaining_players_average_age + captain_age + wicket_keeper_age →
  captain_age = 24 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l164_16497


namespace NUMINAMATH_CALUDE_folded_paper_length_l164_16450

def paper_length : ℝ := 12

theorem folded_paper_length : 
  paper_length / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_folded_paper_length_l164_16450


namespace NUMINAMATH_CALUDE_remainder_of_2543_base12_div_9_l164_16475

/-- Converts a base-12 digit to its decimal value -/
def base12ToDecimal (digit : Nat) : Nat :=
  if digit < 12 then digit else 0

/-- Converts a base-12 number to its decimal equivalent -/
def convertBase12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun digit acc => acc * 12 + base12ToDecimal digit) 0

/-- The base-12 representation of 2543 -/
def base12Number : List Nat := [2, 5, 4, 3]

theorem remainder_of_2543_base12_div_9 :
  (convertBase12ToDecimal base12Number) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2543_base12_div_9_l164_16475


namespace NUMINAMATH_CALUDE_ratio_equality_l164_16420

theorem ratio_equality (x : ℚ) : (x / (2 / 6)) = ((3 / 4) / (1 / 2)) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l164_16420


namespace NUMINAMATH_CALUDE_solution_set_x_abs_x_less_x_l164_16409

theorem solution_set_x_abs_x_less_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_abs_x_less_x_l164_16409


namespace NUMINAMATH_CALUDE_baby_shower_parking_lot_wheels_l164_16478

/-- Calculates the total number of car wheels in a parking lot --/
def total_wheels (guest_cars : ℕ) (parent_cars : ℕ) (wheels_per_car : ℕ) : ℕ :=
  (guest_cars + parent_cars) * wheels_per_car

/-- Theorem statement for the baby shower parking lot problem --/
theorem baby_shower_parking_lot_wheels : 
  total_wheels 10 2 4 = 48 := by
sorry

end NUMINAMATH_CALUDE_baby_shower_parking_lot_wheels_l164_16478


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l164_16476

-- Problem 1
theorem simplify_expression_1 (a : ℝ) (h : a ≠ 0) :
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l164_16476


namespace NUMINAMATH_CALUDE_birthday_candles_distribution_l164_16440

/-- The number of people sharing the candles -/
def num_people : ℕ := 4

/-- The number of candles Ambika has -/
def ambika_candles : ℕ := 4

/-- The ratio of Aniyah's candles to Ambika's candles -/
def aniyah_ratio : ℕ := 6

/-- The total number of candles -/
def total_candles : ℕ := aniyah_ratio * ambika_candles + ambika_candles

/-- The number of candles each person gets when shared equally -/
def candles_per_person : ℕ := total_candles / num_people

theorem birthday_candles_distribution :
  candles_per_person = 7 :=
sorry

end NUMINAMATH_CALUDE_birthday_candles_distribution_l164_16440


namespace NUMINAMATH_CALUDE_jazmin_dolls_count_l164_16422

/-- The number of dolls Geraldine has -/
def geraldine_dolls : ℕ := 2186

/-- The total number of dolls Jazmin and Geraldine have together -/
def total_dolls : ℕ := 3395

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℕ := total_dolls - geraldine_dolls

theorem jazmin_dolls_count : jazmin_dolls = 1209 := by
  sorry

end NUMINAMATH_CALUDE_jazmin_dolls_count_l164_16422


namespace NUMINAMATH_CALUDE_matrix_multiplication_proof_l164_16464

theorem matrix_multiplication_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![2, -6; -1, 3]
  A * B = !![8, -24; 3, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_proof_l164_16464


namespace NUMINAMATH_CALUDE_y_axis_reflection_of_P_l164_16491

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem y_axis_reflection_of_P :
  let P : ℝ × ℝ := (-1, 2)
  reflect_y_axis P = (1, 2) := by sorry

end NUMINAMATH_CALUDE_y_axis_reflection_of_P_l164_16491


namespace NUMINAMATH_CALUDE_abc_range_l164_16442

theorem abc_range (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 1) (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  -1 < a * b * c ∧ a * b * c < 5/27 := by
  sorry

end NUMINAMATH_CALUDE_abc_range_l164_16442


namespace NUMINAMATH_CALUDE_equation_solutions_l164_16413

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, (3*y₁^2 + 2*y₁ - 1 = 0 ∧ 3*y₂^2 + 2*y₂ - 1 = 0) ∧ y₁ = 1/3 ∧ y₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l164_16413


namespace NUMINAMATH_CALUDE_jamies_coins_l164_16485

/-- Represents the number of coins of each type -/
structure CoinCounts where
  quarters : ℚ
  nickels : ℚ
  dimes : ℚ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℚ :=
  25 * coins.quarters + 5 * coins.nickels + 10 * coins.dimes

/-- Theorem stating the solution to Jamie's coin problem -/
theorem jamies_coins :
  ∃ (coins : CoinCounts),
    coins.nickels = 2 * coins.quarters ∧
    coins.dimes = coins.quarters ∧
    totalValue coins = 1520 ∧
    coins.quarters = 304/9 ∧
    coins.nickels = 608/9 ∧
    coins.dimes = 304/9 := by
  sorry

end NUMINAMATH_CALUDE_jamies_coins_l164_16485


namespace NUMINAMATH_CALUDE_spells_conversion_l164_16438

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The number of spells in base 9 -/
def spellsBase9 : List Nat := [7, 4, 5]

theorem spells_conversion :
  base9ToBase10 spellsBase9 = 448 := by
  sorry

end NUMINAMATH_CALUDE_spells_conversion_l164_16438


namespace NUMINAMATH_CALUDE_petya_spent_less_than_5000_l164_16459

/-- Represents the purchase of a book -/
inductive Purchase
  | Expensive (cost : ℕ)
  | Cheap (cost : ℕ)

/-- Represents Petya's shopping process -/
structure ShoppingProcess where
  initial_money : ℕ
  purchases : List Purchase
  final_coins : ℕ

/-- Checks if a shopping process is valid according to the problem conditions -/
def is_valid_process (p : ShoppingProcess) : Prop :=
  p.initial_money % 100 = 0 ∧
  (∀ purchase ∈ p.purchases, match purchase with
    | Purchase.Expensive cost => cost ≥ 100
    | Purchase.Cheap cost => cost < 100
  ) ∧
  p.final_coins < 100 ∧
  2 * (p.initial_money - p.final_coins) = p.initial_money

/-- Calculates the total amount spent on books -/
def total_spent (p : ShoppingProcess) : ℕ :=
  p.initial_money - p.final_coins

/-- Theorem stating that Petya could not have spent at least 5000 rubles on books -/
theorem petya_spent_less_than_5000 (p : ShoppingProcess) :
  is_valid_process p → total_spent p < 5000 := by
  sorry

end NUMINAMATH_CALUDE_petya_spent_less_than_5000_l164_16459


namespace NUMINAMATH_CALUDE_rent_percentage_is_seven_percent_l164_16403

/-- Proves that the percentage of monthly earnings spent on rent is 7% -/
theorem rent_percentage_is_seven_percent (monthly_earnings : ℝ) 
  (rent_amount : ℝ) (savings_amount : ℝ) :
  rent_amount = 133 →
  savings_amount = 817 →
  monthly_earnings = rent_amount + savings_amount + (monthly_earnings / 2) →
  (rent_amount / monthly_earnings) * 100 = 7 := by
sorry

end NUMINAMATH_CALUDE_rent_percentage_is_seven_percent_l164_16403


namespace NUMINAMATH_CALUDE_inscribed_square_area_l164_16406

/-- The area of a square inscribed in a circle, which is itself inscribed in an equilateral triangle -/
theorem inscribed_square_area (s : ℝ) (h : s = 6) :
  let r := s / (2 * Real.sqrt 3)
  let d := 2 * r
  let side := d / Real.sqrt 2
  side ^ 2 = (s / (2 * Real.sqrt 3))^2 * 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l164_16406


namespace NUMINAMATH_CALUDE_runner_speed_increase_l164_16481

theorem runner_speed_increase (v : ℝ) (h : v > 0) : 
  (v + 2) / v = 2.5 → (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_increase_l164_16481


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l164_16462

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- State the theorem
theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : min (fib n / fib (n-1)) (fib (n+1) / fib n) < a / b ∧ 
            a / b < max (fib n / fib (n-1)) (fib (n+1) / fib n)) : 
  b ≥ fib (n+1) := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_inequality_l164_16462


namespace NUMINAMATH_CALUDE_expand_product_l164_16451

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l164_16451


namespace NUMINAMATH_CALUDE_perpendicular_polygon_area_l164_16472

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  area : ℝ
  sides_congruent : sides > 0
  perimeter_eq : perimeter = sides * side_length
  area_calc : area = 16 * side_length^2

/-- Theorem: The area of a specific perpendicular polygon -/
theorem perpendicular_polygon_area :
  ∀ (p : PerpendicularPolygon),
    p.sides = 20 ∧ p.perimeter = 60 → p.area = 144 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_polygon_area_l164_16472


namespace NUMINAMATH_CALUDE_no_charming_seven_digit_number_l164_16474

/-- A function that checks if a list of digits forms a charming number -/
def is_charming (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = Finset.range 7 ∧
  (∀ k : Nat, k ∈ Finset.range 7 → 
    (digits.take k).foldl (fun acc d => acc * 10 + d) 0 % k = 0) ∧
  digits.getLast? = some 7

/-- Theorem stating that no charming 7-digit number exists -/
theorem no_charming_seven_digit_number : 
  ¬ ∃ (digits : List Nat), is_charming digits := by
  sorry

end NUMINAMATH_CALUDE_no_charming_seven_digit_number_l164_16474


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l164_16468

-- Define the vectors
def AB : Fin 2 → ℝ := ![2, 5]
def BC (m : ℝ) : Fin 2 → ℝ := ![m, -2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_m_value :
  (∀ m : ℝ, dot_product AB (BC m) = 0) → (∃ m : ℝ, m = 5) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l164_16468


namespace NUMINAMATH_CALUDE_jake_brought_four_balloons_l164_16433

/-- The number of balloons Allan brought -/
def allan_balloons : ℕ := 2

/-- The total number of balloons Allan and Jake had -/
def total_balloons : ℕ := 6

/-- The number of balloons Jake brought -/
def jake_balloons : ℕ := total_balloons - allan_balloons

theorem jake_brought_four_balloons : jake_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_jake_brought_four_balloons_l164_16433


namespace NUMINAMATH_CALUDE_cost_price_calculation_l164_16496

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 400)
  (h2 : profit_percentage = 25) : 
  ∃ (cost_price : ℝ), 
    cost_price = 320 ∧ 
    selling_price = cost_price * (1 + profit_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l164_16496


namespace NUMINAMATH_CALUDE_calculation_proof_l164_16444

theorem calculation_proof : (1/2)⁻¹ + 4 * Real.cos (60 * π / 180) - (5 - Real.pi)^0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l164_16444


namespace NUMINAMATH_CALUDE_subset_condition_l164_16461

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - a^2 < 0}

theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ a < -1 - sqrt 5 ∨ a > (1 + sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l164_16461


namespace NUMINAMATH_CALUDE_approx_cylinder_volume_increase_l164_16477

/-- Approximate increase in cylinder volume -/
theorem approx_cylinder_volume_increase (H R : ℝ) (h : H = 40) (r : R = 30) :
  let dR := 0.5
  let dV := 2 * π * H * R * dR
  dV = 1200 * π := by sorry

end NUMINAMATH_CALUDE_approx_cylinder_volume_increase_l164_16477


namespace NUMINAMATH_CALUDE_divisible_by_120_l164_16473

theorem divisible_by_120 (n : ℤ) : ∃ k : ℤ, n^6 + 2*n^5 - n^2 - 2*n = 120*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l164_16473


namespace NUMINAMATH_CALUDE_m_range_l164_16424

/-- Proposition p: For all x ∈ ℝ, |x| + x ≥ 0 -/
def prop_p : Prop := ∀ x : ℝ, |x| + x ≥ 0

/-- Proposition q: The equation x² + mx + 1 = 0 has real roots -/
def prop_q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m*x + 1 = 0

/-- The composite proposition "p ∧ q" is false -/
axiom p_and_q_false : ∀ m : ℝ, ¬(prop_p ∧ prop_q m)

/-- The main theorem: Given the conditions above, prove that -2 < m < 2 -/
theorem m_range : ∀ m : ℝ, (¬(prop_p ∧ prop_q m)) → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l164_16424


namespace NUMINAMATH_CALUDE_range_of_negative_values_l164_16453

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (-∞, 0] if f(x) ≥ f(y) for all x, y ∈ (-∞, 0] with x ≤ y -/
def IsDecreasingOnNegative (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → x ≤ 0 → y ≤ 0 → f x ≥ f y

/-- The theorem stating the range of x for which f(x) < 0 -/
theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_decreasing : IsDecreasingOnNegative f) 
  (h_zero : f 2 = 0) : 
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l164_16453


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l164_16488

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 1 ∨ d = 2 ∨ d = 3

theorem largest_number_with_digit_sum_13 :
  ∀ n : ℕ, 
    valid_digits n → 
    digit_sum n = 13 → 
    n ≤ 222211111 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l164_16488


namespace NUMINAMATH_CALUDE_largest_product_of_digits_l164_16414

/-- A function that returns the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is equal to the product of its digits -/
def isProductOfDigits (n : ℕ) : Prop :=
  n = productOfDigits n

/-- Theorem stating that 9 is the largest natural number equal to the product of its digits -/
theorem largest_product_of_digits : 
  ∀ n : ℕ, isProductOfDigits n → n ≤ 9 := by sorry

end NUMINAMATH_CALUDE_largest_product_of_digits_l164_16414


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l164_16431

theorem arithmetic_sequence_sum (a₁ aₙ d : ℕ) (n : ℕ+) :
  (a₁ ≤ aₙ) →
  (aₙ = a₁ + (n - 1) * d) →
  3 * (n : ℕ) * (a₁ + aₙ) / 2 = 3774 →
  3 * (Finset.sum (Finset.range n) (λ i => a₁ + i * d)) = 3774 := by
  sorry

#check arithmetic_sequence_sum 50 98 3 17

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l164_16431


namespace NUMINAMATH_CALUDE_fifth_month_sales_l164_16449

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_3 : ℕ := 6855
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 4991
def average_sale : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    sales_5 = average_sale * num_months - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6562 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l164_16449


namespace NUMINAMATH_CALUDE_martha_lasagna_cheese_amount_l164_16432

/-- The amount of cheese Martha needs for her lasagna -/
def cheese_amount : ℝ :=
  1.5

/-- The cost of cheese per kilogram in dollars -/
def cheese_cost_per_kg : ℝ :=
  6

/-- The cost of meat per kilogram in dollars -/
def meat_cost_per_kg : ℝ :=
  8

/-- The amount of meat Martha needs in grams -/
def meat_amount_grams : ℝ :=
  500

/-- The total cost of ingredients in dollars -/
def total_cost : ℝ :=
  13

theorem martha_lasagna_cheese_amount :
  cheese_amount * cheese_cost_per_kg +
  (meat_amount_grams / 1000) * meat_cost_per_kg =
  total_cost :=
by sorry

end NUMINAMATH_CALUDE_martha_lasagna_cheese_amount_l164_16432


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_b_greater_than_neg_one_l164_16458

def A : Set ℝ := {x | Real.log (x + 2) / Real.log (1/2) < 0}
def B (a b : ℝ) : Set ℝ := {x | (x - a) * (x - b) < 0}

theorem intersection_nonempty_implies_b_greater_than_neg_one :
  (∀ b : ℝ, (A ∩ B (-3) b).Nonempty) → ∀ b : ℝ, b > -1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_b_greater_than_neg_one_l164_16458


namespace NUMINAMATH_CALUDE_angle_c_measure_l164_16486

theorem angle_c_measure (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A + B = 80 →       -- Given condition
  C = 100            -- Conclusion to prove
:= by sorry

end NUMINAMATH_CALUDE_angle_c_measure_l164_16486


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l164_16466

def standard_deck_size : ℕ := 52

def num_fives : ℕ := 4
def num_diamonds : ℕ := 13
def num_threes : ℕ := 4

def probability_specific_draw : ℚ :=
  (num_fives * num_diamonds * num_threes) / (standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2))

theorem probability_of_specific_draw :
  probability_specific_draw = 17 / 11050 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l164_16466


namespace NUMINAMATH_CALUDE_bike_price_proof_l164_16401

theorem bike_price_proof (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percentage = 0.20 →
  upfront_payment = 240 →
  upfront_percentage * total_price = upfront_payment →
  total_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_proof_l164_16401


namespace NUMINAMATH_CALUDE_hexagon_midpoint_area_l164_16400

-- Define the hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the set of line segments
def line_segments (h : Set (ℝ × ℝ)) : Set (ℝ × ℝ × ℝ × ℝ) := sorry

-- Define the midpoints of the line segments
def midpoints (segments : Set (ℝ × ℝ × ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Define the area enclosed by the midpoints
def enclosed_area (points : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem hexagon_midpoint_area :
  let h := regular_hexagon 3
  let s := line_segments h
  let m := midpoints s
  let a := enclosed_area m
  ∃ ε > 0, abs (a - 1.85) < ε := by sorry

end NUMINAMATH_CALUDE_hexagon_midpoint_area_l164_16400


namespace NUMINAMATH_CALUDE_function_composition_equality_l164_16419

/-- Given a function f(x) = a x^2 - √2, where a is a constant,
    prove that if f(f(√2)) = -√2, then a = √2 / 2 -/
theorem function_composition_equality (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - Real.sqrt 2
  f (f (Real.sqrt 2)) = -Real.sqrt 2 → a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l164_16419


namespace NUMINAMATH_CALUDE_total_books_calculation_l164_16493

/-- The number of boxes containing children's books. -/
def num_boxes : ℕ := 5

/-- The number of children's books in each box. -/
def books_per_box : ℕ := 20

/-- The total number of children's books in all boxes. -/
def total_books : ℕ := num_boxes * books_per_box

theorem total_books_calculation : total_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_books_calculation_l164_16493


namespace NUMINAMATH_CALUDE_raul_remaining_money_l164_16415

def initial_amount : ℕ := 87
def number_of_comics : ℕ := 8
def cost_per_comic : ℕ := 4

theorem raul_remaining_money : 
  initial_amount - (number_of_comics * cost_per_comic) = 55 := by
  sorry

end NUMINAMATH_CALUDE_raul_remaining_money_l164_16415


namespace NUMINAMATH_CALUDE_max_blocks_fit_l164_16439

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- The dimensions of the small block -/
def smallBlock : Dimensions := ⟨3, 2, 1⟩

/-- The dimensions of the box -/
def box : Dimensions := ⟨4, 6, 2⟩

/-- The maximum number of small blocks that can fit in the box -/
def maxBlocks : ℕ := 8

theorem max_blocks_fit :
  volume box / volume smallBlock = maxBlocks ∧
  maxBlocks * volume smallBlock ≤ volume box ∧
  (maxBlocks + 1) * volume smallBlock > volume box :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l164_16439


namespace NUMINAMATH_CALUDE_circle_and_tangents_l164_16460

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line y = 2x
def Line (m : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1}

-- Define the point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem statement
theorem circle_and_tangents 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C ∈ Line 2) -- Center lies on y = 2x
  (h2 : (0, 0) ∈ Circle C (Real.sqrt 5)) -- Circle passes through (0,0)
  (h3 : (2, 0) ∈ Circle C (Real.sqrt 5)) -- Circle passes through (2,0)
  : 
  -- 1. The circle equation
  Circle C (Real.sqrt 5) = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 5} ∧
  -- 2. The tangent line equations
  ∃ (k₁ k₂ : ℝ), 
    k₁ = Real.sqrt 5 / 2 ∧ 
    k₂ = -Real.sqrt 5 / 2 ∧
    (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.2 - 2 = k₁ * (p.1 + 2)} → 
      ((x, y) ∈ Circle C (Real.sqrt 5) → (x, y) = P)) ∧
    (∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.2 - 2 = k₂ * (p.1 + 2)} → 
      ((x, y) ∈ Circle C (Real.sqrt 5) → (x, y) = P)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l164_16460


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_three_l164_16435

theorem sqrt_difference_equals_three : Real.sqrt (81 + 49) - Real.sqrt (36 + 25) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_three_l164_16435


namespace NUMINAMATH_CALUDE_measure_one_kg_possible_l164_16482

/-- Represents a balance scale with two pans -/
structure BalanceScale :=
  (left_pan : ℝ)
  (right_pan : ℝ)

/-- Represents the state of the weighing process -/
structure WeighingState :=
  (scale : BalanceScale)
  (remaining_grain : ℝ)
  (weighings_left : ℕ)

/-- Performs a single weighing operation -/
def perform_weighing (state : WeighingState) : WeighingState :=
  sorry

/-- Checks if the current state has isolated 1 kg of grain -/
def has_isolated_one_kg (state : WeighingState) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 1 kg of grain under the given conditions -/
theorem measure_one_kg_possible :
  ∃ (initial_state : WeighingState),
    initial_state.scale.left_pan = 0 ∧
    initial_state.scale.right_pan = 0 ∧
    initial_state.remaining_grain = 19 ∧
    initial_state.weighings_left = 3 ∧
    ∃ (final_state : WeighingState),
      final_state = (perform_weighing ∘ perform_weighing ∘ perform_weighing) initial_state ∧
      has_isolated_one_kg final_state :=
by
  sorry

end NUMINAMATH_CALUDE_measure_one_kg_possible_l164_16482


namespace NUMINAMATH_CALUDE_range_of_f_l164_16407

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -2 ≤ y ∧ y ≤ 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l164_16407


namespace NUMINAMATH_CALUDE_gcd_459_357_l164_16454

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_l164_16454


namespace NUMINAMATH_CALUDE_square_sum_zero_l164_16430

theorem square_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) (h_cubic_heptic : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_l164_16430


namespace NUMINAMATH_CALUDE_triangle_angles_sum_l164_16470

theorem triangle_angles_sum (x y : ℕ+) : 
  (5 * x + 3 * y : ℕ) + (3 * x + 20 : ℕ) + (10 * y + 30 : ℕ) = 180 → x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_sum_l164_16470


namespace NUMINAMATH_CALUDE_reporters_coverage_l164_16455

theorem reporters_coverage (total : ℕ) (h_total : total > 0) : 
  let local_politics := (28 : ℕ) * total / 100
  let not_politics := (60 : ℕ) * total / 100
  let politics := total - not_politics
  (politics - local_politics) * 100 / politics = 30 :=
by sorry

end NUMINAMATH_CALUDE_reporters_coverage_l164_16455


namespace NUMINAMATH_CALUDE_largest_multiple_of_60_with_7_and_0_l164_16428

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def consists_of_7_and_0 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7 ∨ d = 0

theorem largest_multiple_of_60_with_7_and_0 :
  ∃ n : ℕ,
    is_multiple_of n 60 ∧
    consists_of_7_and_0 n ∧
    (∀ m : ℕ, m > n → ¬(is_multiple_of m 60 ∧ consists_of_7_and_0 m)) ∧
    n / 15 = 518 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_60_with_7_and_0_l164_16428


namespace NUMINAMATH_CALUDE_ellipse_properties_l164_16445

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (n : ℝ) : ℝ × ℝ := (2, n)

-- Define the perpendicular condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the isosceles condition
def isosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Theorem statement
theorem ellipse_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x y, ellipse_C a x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  ∃ m n, perpendicular (A m) (B n) F₁ ∧
         isosceles (A m) (B n) F₁ ∧
         abs ((A m).1 - (B n).1) * abs ((A m).2 - F₁.2) / 2 = 6 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l164_16445


namespace NUMINAMATH_CALUDE_cube_inequality_l164_16479

theorem cube_inequality (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l164_16479


namespace NUMINAMATH_CALUDE_divisibility_property_l164_16416

theorem divisibility_property (a b d : ℕ+) 
  (h1 : (a + b : ℕ) % d = 0)
  (h2 : (a * b : ℕ) % (d * d) = 0) :
  (a : ℕ) % d = 0 ∧ (b : ℕ) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l164_16416


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l164_16402

theorem prime_sum_theorem (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h_eq : 2 * p + 3 * q = 6 * r) : p + q + r = 7 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l164_16402


namespace NUMINAMATH_CALUDE_roots_equation_s_value_l164_16404

theorem roots_equation_s_value (n r : ℝ) (c d : ℝ) :
  c^2 - n*c + 3 = 0 →
  d^2 - n*d + 3 = 0 →
  (c + 1/d)^2 - r*(c + 1/d) + s = 0 →
  (d + 1/c)^2 - r*(d + 1/c) + s = 0 →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_s_value_l164_16404


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l164_16443

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a parabola -/
def lies_on (p : Point) (par : Parabola) : Prop :=
  p.y = par.a * p.x^2 + par.b * p.x + par.c

/-- The axis of symmetry of a parabola -/
def axis_of_symmetry (par : Parabola) : ℝ := 3

/-- Theorem: The axis of symmetry of a parabola y = ax^2 + bx + c is x = 3, 
    given that the points (2,5) and (4,5) lie on the parabola -/
theorem parabola_axis_of_symmetry (par : Parabola) 
  (h1 : lies_on ⟨2, 5⟩ par) 
  (h2 : lies_on ⟨4, 5⟩ par) : 
  axis_of_symmetry par = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l164_16443


namespace NUMINAMATH_CALUDE_nth_equation_specific_case_l164_16499

theorem nth_equation (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 - (2 * n - 1) / (n * n)) = (n - 1) / n :=
by sorry

theorem specific_case : Real.sqrt (1 - 199 / 10000) = 99 / 100 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_specific_case_l164_16499


namespace NUMINAMATH_CALUDE_replaced_person_weight_l164_16494

/-- Given a group of 8 people, if replacing one person with a new person weighing 89 kg
    increases the average weight by 3 kg, then the weight of the replaced person is 65 kg. -/
theorem replaced_person_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_increase : ℝ)
  (h1 : n = 8)
  (h2 : new_weight = 89)
  (h3 : avg_increase = 3)
  : ∃ (old_weight : ℝ), old_weight = new_weight - n * avg_increase :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l164_16494


namespace NUMINAMATH_CALUDE_older_sibling_age_l164_16417

/-- Given two siblings with a two-year age gap and their combined age, 
    prove the older sibling's age -/
theorem older_sibling_age 
  (h : ℕ) -- Hyeongjun's age
  (s : ℕ) -- Older sister's age
  (age_gap : s = h + 2) -- Two-year age gap condition
  (total_age : h + s = 26) -- Sum of ages condition
  : s = 14 := by
  sorry

end NUMINAMATH_CALUDE_older_sibling_age_l164_16417


namespace NUMINAMATH_CALUDE_zero_points_count_midpoint_derivative_negative_l164_16456

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

-- Theorem for the number of zero points
theorem zero_points_count (a : ℝ) (h : a > 0) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, f a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃! x, f a x = 0) ∨
  (∀ x, f a x ≠ 0) :=
sorry

-- Theorem for f'(x₀) < 0
theorem midpoint_derivative_negative (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 0) (h₂ : x₁ < x₂) (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) :
  let x₀ := (x₁ + x₂) / 2
  f_deriv a x₀ < 0 :=
sorry

end NUMINAMATH_CALUDE_zero_points_count_midpoint_derivative_negative_l164_16456


namespace NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l164_16498

theorem cos_15_cos_45_minus_cos_75_sin_45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) - 
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l164_16498


namespace NUMINAMATH_CALUDE_benny_missed_games_l164_16436

theorem benny_missed_games (total_games attended_games : ℕ) 
  (h1 : total_games = 39)
  (h2 : attended_games = 14) :
  total_games - attended_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_benny_missed_games_l164_16436


namespace NUMINAMATH_CALUDE_bubble_pass_probability_specific_l164_16465

/-- The probability that in a sequence of n distinct terms,
    the kth term ends up in the mth position after one bubble pass. -/
def bubble_pass_probability (n k m : ℕ) : ℚ :=
  if k ≤ m ∧ m < n then 1 / ((m - k + 2) * (m - k + 1))
  else 0

theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 25 40 = 1 / 272 := by
  sorry

end NUMINAMATH_CALUDE_bubble_pass_probability_specific_l164_16465


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l164_16480

theorem largest_stamps_per_page (stamps_book1 stamps_book2 : ℕ) 
  (h1 : stamps_book1 = 960) 
  (h2 : stamps_book2 = 1200) 
  (h3 : stamps_book1 > 0) 
  (h4 : stamps_book2 > 0) : 
  ∃ (stamps_per_page : ℕ), 
    stamps_per_page > 0 ∧ 
    stamps_book1 % stamps_per_page = 0 ∧ 
    stamps_book2 % stamps_per_page = 0 ∧ 
    stamps_book1 / stamps_per_page ≥ 2 ∧ 
    stamps_book2 / stamps_per_page ≥ 2 ∧ 
    ∀ (n : ℕ), n > stamps_per_page → 
      (stamps_book1 % n ≠ 0 ∨ 
       stamps_book2 % n ≠ 0 ∨ 
       stamps_book1 / n < 2 ∨ 
       stamps_book2 / n < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l164_16480


namespace NUMINAMATH_CALUDE_original_number_is_84_l164_16484

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := n % 10 + n / 10

def swap_digits (n : ℕ) : ℕ := (n % 10) * 10 + n / 10

theorem original_number_is_84 (n : ℕ) 
  (h1 : is_two_digit n)
  (h2 : digit_sum n = 12)
  (h3 : n = swap_digits n + 36) :
  n = 84 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_84_l164_16484


namespace NUMINAMATH_CALUDE_least_k_divisible_by_1260_two_ten_divisible_by_1260_least_k_is_210_l164_16437

theorem least_k_divisible_by_1260 (k : ℕ) : k > 0 ∧ k^4 % 1260 = 0 → k ≥ 210 := by
  sorry

theorem two_ten_divisible_by_1260 : (210 : ℕ)^4 % 1260 = 0 := by
  sorry

theorem least_k_is_210 : ∃ k : ℕ, k > 0 ∧ k^4 % 1260 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m^4 % 1260 = 0) → m ≥ k :=
  ⟨210, by
    sorry⟩

end NUMINAMATH_CALUDE_least_k_divisible_by_1260_two_ten_divisible_by_1260_least_k_is_210_l164_16437


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l164_16441

/-- Diamond operation -/
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating the unique solution to A ◊ 7 = 76 -/
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 7 = 76 ∧ A = 12 := by
sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l164_16441


namespace NUMINAMATH_CALUDE_inequality_proof_l164_16418

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1 / x + 1 / y ≤ 1 / x^2 + 1 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l164_16418


namespace NUMINAMATH_CALUDE_tree_height_after_three_years_l164_16457

/-- The height of a tree that doubles every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (2 ^ years)

theorem tree_height_after_three_years :
  ∃ (initial_height : ℝ),
    tree_height initial_height 6 = 32 ∧
    tree_height initial_height 3 = 4 := by
  sorry

#check tree_height_after_three_years

end NUMINAMATH_CALUDE_tree_height_after_three_years_l164_16457


namespace NUMINAMATH_CALUDE_sum_equals_140_l164_16447

theorem sum_equals_140 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (h1 : x^2 + y^2 = 2500) (h2 : z^2 + w^2 = 2500)
  (h3 : x * z = 1200) (h4 : y * w = 1200) : 
  x + y + z + w = 140 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_140_l164_16447


namespace NUMINAMATH_CALUDE_power_function_through_point_l164_16405

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 4 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l164_16405


namespace NUMINAMATH_CALUDE_slope_of_sine_at_pi_fourth_l164_16434

theorem slope_of_sine_at_pi_fourth (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  deriv f (π/4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_sine_at_pi_fourth_l164_16434


namespace NUMINAMATH_CALUDE_power_division_rule_l164_16446

theorem power_division_rule (a : ℝ) : a^3 / a^2 = a := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l164_16446


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l164_16489

def baseball_cards_problem (initial_cards : ℝ) (promised_cards : ℝ) (bought_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - promised_cards

theorem mary_baseball_cards :
  baseball_cards_problem 18.0 26.0 40.0 = 32.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l164_16489


namespace NUMINAMATH_CALUDE_exists_m_with_all_digits_l164_16469

/-- For any positive integer n, there exists a positive integer m such that
    the decimal representation of m * n contains all digits from 0 to 9. -/
theorem exists_m_with_all_digits (n : ℕ+) : ∃ m : ℕ+, ∀ d : Fin 10, ∃ k : ℕ,
  (m * n : ℕ) / 10^k % 10 = d.val :=
sorry

end NUMINAMATH_CALUDE_exists_m_with_all_digits_l164_16469


namespace NUMINAMATH_CALUDE_linear_function_parallel_and_point_l164_16410

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x ↦ k * x + b

-- Define parallel lines
def parallel (f g : ℝ → ℝ) : Prop := ∃ c : ℝ, ∀ x : ℝ, f x = g x + c

theorem linear_function_parallel_and_point :
  ∀ k b : ℝ,
  parallel (linear_function k b) (linear_function 2 1) →
  linear_function k b (-3) = 4 →
  linear_function k b = linear_function 2 10 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_parallel_and_point_l164_16410


namespace NUMINAMATH_CALUDE_cup_volume_ratio_l164_16483

/-- Given a bottle that can be filled with 10 pours of cup a or 5 pours of cup b,
    prove that the volume of cup b is twice the volume of cup a. -/
theorem cup_volume_ratio (V A B : ℝ) (hA : 10 * A = V) (hB : 5 * B = V) :
  B = 2 * A := by sorry

end NUMINAMATH_CALUDE_cup_volume_ratio_l164_16483


namespace NUMINAMATH_CALUDE_complex_number_modulus_l164_16448

theorem complex_number_modulus (z : ℂ) (h : 1 + z = (1 - z) * Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l164_16448


namespace NUMINAMATH_CALUDE_new_shipment_bears_l164_16423

/-- Calculates the number of bears in a new shipment given the initial stock,
    bears per shelf, and number of shelves used. -/
theorem new_shipment_bears (initial_stock : ℕ) (bears_per_shelf : ℕ) (shelves_used : ℕ) :
  (bears_per_shelf * shelves_used) - initial_stock =
  (bears_per_shelf * shelves_used) - initial_stock :=
by sorry

end NUMINAMATH_CALUDE_new_shipment_bears_l164_16423


namespace NUMINAMATH_CALUDE_valid_pictures_invalid_pictures_l164_16411

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Option ℕ

-- Define adjacency in the grid
def adjacent (x₁ y₁ x₂ y₂ : Fin 4) : Prop :=
  (x₁ = x₂ ∧ y₁.val + 1 = y₂.val) ∨
  (x₁ = x₂ ∧ y₂.val + 1 = y₁.val) ∨
  (y₁ = y₂ ∧ x₁.val + 1 = x₂.val) ∨
  (y₁ = y₂ ∧ x₂.val + 1 = x₁.val)

-- Define a valid grid configuration
def valid_grid (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 15 →
    ∃ x₁ y₁ x₂ y₂ : Fin 4,
      g x₁ y₁ = some n ∧
      g x₂ y₂ = some (n + 1) ∧
      adjacent x₁ y₁ x₂ y₂

-- Define the specific configurations for Pictures 3 and 5
def picture3 : Grid := fun x y =>
  match x, y with
  | 0, 0 => some 1 | 0, 1 => some 2 | 0, 2 => some 7 | 0, 3 => some 8
  | 1, 0 => some 14 | 1, 1 => some 3 | 1, 2 => some 6 | 1, 3 => some 9
  | 2, 0 => some 15 | 2, 1 => some 4 | 2, 2 => some 5 | 2, 3 => some 10
  | 3, 0 => some 16 | 3, 1 => none | 3, 2 => none | 3, 3 => some 11
  
def picture5 : Grid := fun x y =>
  match x, y with
  | 0, 0 => none | 0, 1 => some 4 | 0, 2 => some 5 | 0, 3 => some 6
  | 1, 0 => none | 1, 1 => some 3 | 1, 2 => none | 1, 3 => some 7
  | 2, 0 => some 14 | 2, 1 => some 2 | 2, 2 => some 9 | 2, 3 => some 8
  | 3, 0 => some 15 | 3, 1 => some 1 | 3, 2 => some 10 | 3, 3 => none

-- Theorem stating that Pictures 3 and 5 are valid configurations
theorem valid_pictures :
  valid_grid picture3 ∧ valid_grid picture5 := by sorry

-- Theorem stating that Pictures 1, 2, 4, and 6 are not valid configurations
theorem invalid_pictures :
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃,
      g x₁ y₁ = some 3 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 1 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧
      (∃ x₄ y₄ x₅ y₅, g x₄ y₄ = some 11 ∧ g x₅ y₅ = some 10 ∧ ¬adjacent x₄ y₄ x₅ y₅))) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃,
      g x₁ y₁ = some 1 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 3 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ ¬adjacent x₂ y₂ x₃ y₃)) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      g x₁ y₁ = some 1 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 3 ∧ g x₄ y₄ = some 4 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧ ¬adjacent x₃ y₃ x₄ y₄)) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      g x₁ y₁ = some 4 ∧ g x₂ y₂ = some 5 ∧ g x₃ y₃ = some 6 ∧ g x₄ y₄ = some 7 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧ ¬adjacent x₃ y₃ x₄ y₄)) := by sorry

end NUMINAMATH_CALUDE_valid_pictures_invalid_pictures_l164_16411
