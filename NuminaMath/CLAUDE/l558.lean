import Mathlib

namespace NUMINAMATH_CALUDE_nalani_puppy_price_l558_55861

/-- The price per puppy in Nalani's sale --/
def price_per_puppy (num_dogs : ℕ) (puppies_per_dog : ℕ) (fraction_sold : ℚ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (fraction_sold * (num_dogs * puppies_per_dog))

/-- Theorem stating the price per puppy in Nalani's specific case --/
theorem nalani_puppy_price :
  price_per_puppy 2 10 (3/4) 3000 = 200 := by
  sorry

#eval price_per_puppy 2 10 (3/4) 3000

end NUMINAMATH_CALUDE_nalani_puppy_price_l558_55861


namespace NUMINAMATH_CALUDE_minimize_y_l558_55803

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

/-- The theorem stating that (2a + 3b) / 5 minimizes y -/
theorem minimize_y (a b : ℝ) :
  let x_min := (2 * a + 3 * b) / 5
  ∀ x, y x_min a b ≤ y x a b :=
by sorry

end NUMINAMATH_CALUDE_minimize_y_l558_55803


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l558_55817

theorem sum_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (x + y + z = -b / a) := by sorry

theorem sum_of_roots_specific_cubic :
  let f : ℝ → ℝ := λ x => 3 * x^3 + 7 * x^2 - 12 * x - 4
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (x + y + z = -7 / 3) := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l558_55817


namespace NUMINAMATH_CALUDE_pump_calculations_l558_55818

/-- Ultraflow pump rate in gallons per hour -/
def ultraflow_rate : ℚ := 560

/-- MiniFlow pump rate in gallons per hour -/
def miniflow_rate : ℚ := 220

/-- Convert minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ := minutes / 60

/-- Calculate gallons pumped given rate and time -/
def gallons_pumped (rate : ℚ) (time : ℚ) : ℚ := rate * time

theorem pump_calculations :
  (gallons_pumped ultraflow_rate (minutes_to_hours 75) = 700) ∧
  (gallons_pumped ultraflow_rate (minutes_to_hours 50) + gallons_pumped miniflow_rate (minutes_to_hours 50) = 883) := by
  sorry

end NUMINAMATH_CALUDE_pump_calculations_l558_55818


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_contained_implication_l558_55857

structure GeometrySpace where
  Line : Type
  Plane : Type
  parallel_lines : Line → Line → Prop
  parallel_plane_line : Plane → Line → Prop
  parallel_planes : Plane → Plane → Prop
  perpendicular_plane_line : Plane → Line → Prop
  line_in_plane : Line → Plane → Prop
  line_not_in_plane : Line → Plane → Prop

variable (G : GeometrySpace)

theorem parallel_perpendicular_implication
  (m n : G.Line) (α β : G.Plane)
  (h1 : G.parallel_lines m n)
  (h2 : G.perpendicular_plane_line α m)
  (h3 : G.perpendicular_plane_line β n) :
  G.parallel_planes α β :=
sorry

theorem parallel_contained_implication
  (m n : G.Line) (α β : G.Plane)
  (h1 : G.parallel_lines m n)
  (h2 : G.line_in_plane n α)
  (h3 : G.parallel_planes α β)
  (h4 : G.line_not_in_plane m β) :
  G.parallel_plane_line β m :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_contained_implication_l558_55857


namespace NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l558_55869

theorem right_triangle_and_modular_inverse :
  -- Define the sides of the triangle
  let a : ℕ := 15
  let b : ℕ := 112
  let c : ℕ := 113
  -- Define the modulus
  let m : ℕ := 2799
  -- Define the number we're finding the inverse for
  let x : ℕ := 225
  -- Condition: a, b, c form a right triangle
  (a^2 + b^2 = c^2) →
  -- Conclusion: 1 is the multiplicative inverse of x modulo m
  (1 * x) % m = 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l558_55869


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l558_55856

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def water_filter_cost : ℝ := 65
def water_filter_discount : ℝ := 0.25
def camping_mat_cost : ℝ := 45
def camping_mat_discount : ℝ := 0.15
def backpack_cost : ℝ := 105

def clothing_tax_rate : ℝ := 0.05
def electronics_tax_rate : ℝ := 0.1
def other_equipment_tax_rate : ℝ := 0.08

def total_cost : ℝ :=
  (hoodie_cost * (1 + clothing_tax_rate)) +
  (flashlight_cost * (1 + electronics_tax_rate)) +
  (boots_original_cost * (1 - boots_discount) * (1 + clothing_tax_rate)) +
  (water_filter_cost * (1 - water_filter_discount) * (1 + other_equipment_tax_rate)) +
  (camping_mat_cost * (1 - camping_mat_discount) * (1 + other_equipment_tax_rate)) +
  (backpack_cost * (1 + other_equipment_tax_rate))

theorem hiking_equipment_cost : total_cost = 413.91 := by
  sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l558_55856


namespace NUMINAMATH_CALUDE_min_games_for_prediction_l558_55841

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  total_games : ℕ
  games_per_white_student : ℕ

/-- Defines the specific chess tournament in the problem -/
def problem_tournament : ChessTournament :=
  { white_rook_students := 15,
    black_elephant_students := 20,
    total_games := 300,
    games_per_white_student := 20 }

/-- Theorem stating the minimum number of games for Sasha's prediction -/
theorem min_games_for_prediction (t : ChessTournament) 
  (h1 : t.white_rook_students * t.black_elephant_students = t.total_games)
  (h2 : t.games_per_white_student = t.black_elephant_students) :
  t.total_games - (t.white_rook_students - 1) * t.games_per_white_student = 280 :=
sorry

end NUMINAMATH_CALUDE_min_games_for_prediction_l558_55841


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_for_non_intersection_l558_55826

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define lines in the space
def Line (V : Type*) [NormedAddCommGroup V] := V → Set V

-- Define the property of being skew
def are_skew (l₁ l₂ : Line V) : Prop := sorry

-- Define the property of not intersecting
def do_not_intersect (l₁ l₂ : Line V) : Prop := sorry

-- Theorem statement
theorem skew_lines_sufficient_not_necessary_for_non_intersection :
  (∀ l₁ l₂ : Line V, are_skew l₁ l₂ → do_not_intersect l₁ l₂) ∧
  (∃ l₁ l₂ : Line V, do_not_intersect l₁ l₂ ∧ ¬are_skew l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_for_non_intersection_l558_55826


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l558_55899

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The value of silver medallions in base 7 --/
def silverValue : List Nat := [6, 2, 3, 5]

/-- The value of precious gemstones in base 7 --/
def gemstonesValue : List Nat := [1, 6, 4, 3]

/-- The value of spices in base 7 --/
def spicesValue : List Nat := [6, 5, 6]

theorem pirate_treasure_sum :
  base7ToBase10 silverValue + base7ToBase10 gemstonesValue + base7ToBase10 spicesValue = 3485 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_sum_l558_55899


namespace NUMINAMATH_CALUDE_twenty_four_point_game_l558_55812

theorem twenty_four_point_game (Q : ℕ) (h : Q = 12) : 
  (Q * 9) - (Q * 7) = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_point_game_l558_55812


namespace NUMINAMATH_CALUDE_largest_number_l558_55883

theorem largest_number (S : Set ℝ) (hS : S = {-1, 0, 1, 1/3}) : 
  ∃ m ∈ S, ∀ x ∈ S, x ≤ m ∧ m = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_number_l558_55883


namespace NUMINAMATH_CALUDE_flour_to_add_l558_55880

/-- Given the total amount of flour required for a recipe and the amount already added,
    this theorem proves that the amount of flour needed to be added is the difference
    between the total required and the amount already added. -/
theorem flour_to_add (total_flour : ℕ) (flour_added : ℕ) :
  total_flour ≥ flour_added →
  total_flour - flour_added = total_flour - flour_added := by
  sorry

#check flour_to_add

end NUMINAMATH_CALUDE_flour_to_add_l558_55880


namespace NUMINAMATH_CALUDE_saline_solution_concentration_l558_55878

/-- Proves that mixing 100 kg of 30% saline solution with 200 kg of pure water
    results in a final saline solution with a concentration of 10%. -/
theorem saline_solution_concentration
  (initial_solution_weight : ℝ)
  (initial_concentration : ℝ)
  (pure_water_weight : ℝ)
  (h1 : initial_solution_weight = 100)
  (h2 : initial_concentration = 0.3)
  (h3 : pure_water_weight = 200) :
  let salt_weight := initial_solution_weight * initial_concentration
  let total_weight := initial_solution_weight + pure_water_weight
  let final_concentration := salt_weight / total_weight
  final_concentration = 0.1 := by
sorry

end NUMINAMATH_CALUDE_saline_solution_concentration_l558_55878


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l558_55827

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l558_55827


namespace NUMINAMATH_CALUDE_factor_expression_l558_55896

theorem factor_expression (x : ℝ) : 100 * x^23 + 225 * x^46 = 25 * x^23 * (4 + 9 * x^23) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l558_55896


namespace NUMINAMATH_CALUDE_duodecimal_reversal_difference_divisibility_l558_55828

/-- Represents a duodecimal digit (0 to 11) -/
def DuodecimalDigit := {n : ℕ // n ≤ 11}

/-- Converts a two-digit duodecimal number to its decimal representation -/
def toDecimal (a b : DuodecimalDigit) : ℤ :=
  12 * a.val + b.val

theorem duodecimal_reversal_difference_divisibility
  (a b : DuodecimalDigit)
  (h : a ≠ b) :
  ∃ k : ℤ, toDecimal a b - toDecimal b a = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_duodecimal_reversal_difference_divisibility_l558_55828


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l558_55890

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- The number in base 7 --/
def base7Number : List Nat := [1, 2, 0, 2, 1, 0, 1, 2]

/-- The decimal representation of the base 7 number --/
def decimalNumber : Nat := toDecimal base7Number

/-- Predicate to check if a number is prime --/
def isPrime (n : Nat) : Prop := sorry

theorem largest_prime_divisor :
  ∃ (p : Nat), isPrime p ∧ p ∣ decimalNumber ∧ 
  ∀ (q : Nat), isPrime q → q ∣ decimalNumber → q ≤ p ∧ p = 397 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l558_55890


namespace NUMINAMATH_CALUDE_largest_c_for_function_range_l558_55867

theorem largest_c_for_function_range (f : ℝ → ℝ) (c : ℝ) :
  (∀ x, f x = x^2 - 6*x + c) →
  (∃ x, f x = 4) →
  c ≤ 13 ∧ 
  (∀ d > 13, ¬∃ x, x^2 - 6*x + d = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_function_range_l558_55867


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l558_55843

/-- The function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) if and only if k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l558_55843


namespace NUMINAMATH_CALUDE_sum_f_negative_l558_55855

def f (x : ℝ) : ℝ := x + x^3

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l558_55855


namespace NUMINAMATH_CALUDE_centrally_symmetric_multiple_symmetry_axes_l558_55898

/-- A polygon in a 2D plane. -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- Represents a line in a 2D plane. -/
structure Line where
  -- Add necessary fields for a line

/-- Predicate to check if a polygon is centrally symmetric. -/
def is_centrally_symmetric (p : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line is a symmetry axis of a polygon. -/
def is_symmetry_axis (l : Line) (p : Polygon) : Prop :=
  sorry

/-- The number of symmetry axes a polygon has. -/
def num_symmetry_axes (p : Polygon) : Nat :=
  sorry

/-- Theorem: A centrally symmetric polygon with at least one symmetry axis must have more than one symmetry axis. -/
theorem centrally_symmetric_multiple_symmetry_axes (p : Polygon) :
  is_centrally_symmetric p → (∃ l : Line, is_symmetry_axis l p) → num_symmetry_axes p > 1 :=
by sorry

end NUMINAMATH_CALUDE_centrally_symmetric_multiple_symmetry_axes_l558_55898


namespace NUMINAMATH_CALUDE_equation_has_three_real_solutions_l558_55831

-- Define the equation
def equation (x : ℝ) : Prop :=
  (18 * x - 2) ^ (1/3) + (14 * x - 4) ^ (1/3) = 5 * (2 * x + 4) ^ (1/3)

-- State the theorem
theorem equation_has_three_real_solutions :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation x₁ ∧ equation x₂ ∧ equation x₃ :=
sorry

end NUMINAMATH_CALUDE_equation_has_three_real_solutions_l558_55831


namespace NUMINAMATH_CALUDE_john_necklaces_l558_55858

/-- Given the number of wire spools, length of each spool, and wire required per necklace,
    calculate the number of necklaces that can be made. -/
def necklaces_from_wire (num_spools : ℕ) (spool_length : ℕ) (wire_per_necklace : ℕ) : ℕ :=
  (num_spools * spool_length) / wire_per_necklace

/-- Prove that John can make 15 necklaces with the given conditions. -/
theorem john_necklaces : necklaces_from_wire 3 20 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_necklaces_l558_55858


namespace NUMINAMATH_CALUDE_cheryl_pesto_production_l558_55825

/-- Represents the pesto production scenario -/
structure PestoProduction where
  basil_per_pesto : ℕ  -- cups of basil needed for 1 cup of pesto
  basil_per_week : ℕ   -- cups of basil harvested per week
  harvest_weeks : ℕ    -- number of weeks of harvest

/-- Calculates the total cups of pesto that can be produced -/
def total_pesto (p : PestoProduction) : ℕ :=
  (p.basil_per_week * p.harvest_weeks) / p.basil_per_pesto

/-- Theorem: Given the conditions, Cheryl can make 32 cups of pesto -/
theorem cheryl_pesto_production :
  let p := PestoProduction.mk 4 16 8
  total_pesto p = 32 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_pesto_production_l558_55825


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l558_55804

theorem arithmetic_calculations :
  (7 + (-14) - (-9) - 12 = -10) ∧
  (25 / (-5) * (1 / 5) / (3 / 4) = -4 / 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l558_55804


namespace NUMINAMATH_CALUDE_janinas_daily_rent_l558_55864

/-- Janina's pancake stand financial model -/
def pancake_stand_model (daily_supply_cost : ℝ) (pancake_price : ℝ) (breakeven_pancakes : ℕ) : ℝ :=
  pancake_price * (breakeven_pancakes : ℝ) - daily_supply_cost

/-- Theorem: Janina's daily rent is $30 -/
theorem janinas_daily_rent :
  pancake_stand_model 12 2 21 = 30 := by
  sorry

end NUMINAMATH_CALUDE_janinas_daily_rent_l558_55864


namespace NUMINAMATH_CALUDE_peter_wants_17_dogs_l558_55815

/-- The number of dogs Peter wants to have -/
def PetersDogs (samGS : ℕ) (samFB : ℕ) (peterGSFactor : ℕ) (peterFBFactor : ℕ) : ℕ :=
  peterGSFactor * samGS + peterFBFactor * samFB

/-- Theorem stating the number of dogs Peter wants to have -/
theorem peter_wants_17_dogs :
  PetersDogs 3 4 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_peter_wants_17_dogs_l558_55815


namespace NUMINAMATH_CALUDE_pool_area_is_30_l558_55866

/-- The surface area of a rectangular pool -/
def pool_surface_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The surface area of a rectangular pool with width 3 meters and length 10 meters is 30 square meters -/
theorem pool_area_is_30 : pool_surface_area 3 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pool_area_is_30_l558_55866


namespace NUMINAMATH_CALUDE_july_capsule_intake_l558_55800

/-- Represents the number of capsules taken in a month -/
def capsulesTaken (totalDays : ℕ) (forgottenDays : ℕ) : ℕ :=
  totalDays - forgottenDays

/-- Theorem: Given a 31-day month where a person forgets to take capsules on 3 days,
    the total number of capsules taken is 28 -/
theorem july_capsule_intake :
  capsulesTaken 31 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_july_capsule_intake_l558_55800


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_solution_set_for_any_a_l558_55854

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for the second part of the problem
theorem solution_set_for_any_a (a : ℝ) :
  {x : ℝ | f a x > 0} = 
    if a > -1 then
      {x : ℝ | x < -1 ∨ x > a}
    else if a = -1 then
      {x : ℝ | x < -1 ∨ x > -1}
    else
      {x : ℝ | x < a ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_solution_set_for_any_a_l558_55854


namespace NUMINAMATH_CALUDE_ABABABA_probability_l558_55884

/-- The number of tiles marked A -/
def num_A : ℕ := 4

/-- The number of tiles marked B -/
def num_B : ℕ := 3

/-- The total number of tiles -/
def total_tiles : ℕ := num_A + num_B

/-- The number of favorable arrangements (ABABABA) -/
def favorable_arrangements : ℕ := 1

/-- The probability of the specific arrangement ABABABA -/
def probability_ABABABA : ℚ := favorable_arrangements / (total_tiles.choose num_A)

theorem ABABABA_probability : probability_ABABABA = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ABABABA_probability_l558_55884


namespace NUMINAMATH_CALUDE_binomial_200_200_l558_55882

theorem binomial_200_200 : Nat.choose 200 200 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_200_200_l558_55882


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l558_55851

/-- Given a curve y = x^2 + a ln(x) where a > 0, if the minimum value of the slope
    of the tangent line at any point on the curve is 4, then the coordinates of the
    point of tangency at this minimum slope are (1, 1). -/
theorem tangent_point_coordinates (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, 2 * x + a / x ≥ 4) ∧ (∃ x > 0, 2 * x + a / x = 4) →
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = x^2 + a * Real.log x ∧ 2 * x + a / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l558_55851


namespace NUMINAMATH_CALUDE_incorrect_statement_l558_55886

def A (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 4*n + k}

theorem incorrect_statement :
  ¬ (∀ a b : ℤ, (a + b) ∈ A 3 → (a ∈ A 1 ∧ b ∈ A 2)) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l558_55886


namespace NUMINAMATH_CALUDE_new_tax_rate_calculation_l558_55871

theorem new_tax_rate_calculation (original_rate : ℝ) (income : ℝ) (savings : ℝ) : 
  original_rate = 0.46 → 
  income = 36000 → 
  savings = 5040 → 
  (income * original_rate - savings) / income = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_new_tax_rate_calculation_l558_55871


namespace NUMINAMATH_CALUDE_find_m_l558_55865

def A (m : ℕ) : Set ℕ := {1, 2, m}
def B : Set ℕ := {4, 7, 13}

def f (x : ℕ) : ℕ := 3 * x + 1

theorem find_m : ∃ m : ℕ, 
  (∀ x ∈ A m, f x ∈ B) ∧ 
  m = 4 := by sorry

end NUMINAMATH_CALUDE_find_m_l558_55865


namespace NUMINAMATH_CALUDE_range_of_a_l558_55852

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l558_55852


namespace NUMINAMATH_CALUDE_max_distance_to_origin_l558_55872

theorem max_distance_to_origin : 
  let curve := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = Real.sqrt 3 + Real.cos θ ∧ y = 1 + Real.sin θ}
  ∀ p ∈ curve, ∃ q ∈ curve, ∀ r ∈ curve, Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ≥ Real.sqrt ((r.1 - 0)^2 + (r.2 - 0)^2) ∧
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = 3 := by
sorry


end NUMINAMATH_CALUDE_max_distance_to_origin_l558_55872


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l558_55814

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_roots : a 3 * a 7 = 4 ∧ a 3 + a 7 = 5) :
  a 5 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l558_55814


namespace NUMINAMATH_CALUDE_fraction_ordering_l558_55837

theorem fraction_ordering : (8 : ℚ) / 25 < 1 / 3 ∧ 1 / 3 < 10 / 31 ∧ 10 / 31 < 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l558_55837


namespace NUMINAMATH_CALUDE_missing_digit_is_seven_l558_55840

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def insert_digit (d : ℕ) : ℕ := 351000 + d * 100 + 92

theorem missing_digit_is_seven :
  ∃! d : ℕ, d < 10 ∧ is_divisible_by_9 (insert_digit d) :=
by
  sorry

#check missing_digit_is_seven

end NUMINAMATH_CALUDE_missing_digit_is_seven_l558_55840


namespace NUMINAMATH_CALUDE_inequality_solution_set_l558_55810

theorem inequality_solution_set :
  ∀ x : ℝ, (6 - x - 2 * x^2 < 0) ↔ (x > 3/2 ∨ x < -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l558_55810


namespace NUMINAMATH_CALUDE_inequality_proof_l558_55848

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l558_55848


namespace NUMINAMATH_CALUDE_equation_solution_l558_55833

theorem equation_solution (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 9) 
  (eq2 : x + 3 * y = 8) : 
  3 * x^2 + 7 * x * y + 3 * y^2 = 145 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l558_55833


namespace NUMINAMATH_CALUDE_three_numbers_sum_square_counterexample_l558_55853

theorem three_numbers_sum_square_counterexample :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b^2 + c^2 = b + a^2 + c^2) ∧
    (b + a^2 + c^2 = c + a^2 + b^2) ∧
    (a ≠ b ∨ b ≠ c ∨ a ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_square_counterexample_l558_55853


namespace NUMINAMATH_CALUDE_different_terminal_sides_not_equal_l558_55847

-- Define an angle
def Angle : Type := ℝ

-- Define the initial side of an angle
def initial_side (a : Angle) : ℝ × ℝ := sorry

-- Define the terminal side of an angle
def terminal_side (a : Angle) : ℝ × ℝ := sorry

-- Define equality of angles
def angle_eq (a b : Angle) : Prop := 
  initial_side a = initial_side b ∧ terminal_side a = terminal_side b

-- Theorem statement
theorem different_terminal_sides_not_equal (a b : Angle) :
  initial_side a = initial_side b → 
  terminal_side a ≠ terminal_side b → 
  ¬(angle_eq a b) := by sorry

end NUMINAMATH_CALUDE_different_terminal_sides_not_equal_l558_55847


namespace NUMINAMATH_CALUDE_mijeong_box_volume_l558_55894

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of Mijeong's cuboid box -/
theorem mijeong_box_volume :
  cuboid_volume 14 13 = 182 := by
  sorry

end NUMINAMATH_CALUDE_mijeong_box_volume_l558_55894


namespace NUMINAMATH_CALUDE_divisor_greater_than_remainder_l558_55849

theorem divisor_greater_than_remainder (a b q r : ℕ) : 
  a = b * q + r → r = 8 → b > 8 := by sorry

end NUMINAMATH_CALUDE_divisor_greater_than_remainder_l558_55849


namespace NUMINAMATH_CALUDE_least_possible_FG_l558_55889

-- Define the triangle EFG
structure TriangleEFG where
  EF : ℝ
  EG : ℝ
  FG : ℝ

-- Define the triangle HFG
structure TriangleHFG where
  HF : ℝ
  HG : ℝ
  FG : ℝ

-- Define the shared triangle configuration
def SharedTriangles (t1 : TriangleEFG) (t2 : TriangleHFG) : Prop :=
  t1.FG = t2.FG ∧
  t1.EF = 7 ∧
  t1.EG = 15 ∧
  t2.HG = 10 ∧
  t2.HF = 25

-- Theorem statement
theorem least_possible_FG (t1 : TriangleEFG) (t2 : TriangleHFG) 
  (h : SharedTriangles t1 t2) : 
  ∃ (n : ℕ), n = 15 ∧ t1.FG = n ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (t1' : TriangleEFG) (t2' : TriangleHFG), 
    SharedTriangles t1' t2' ∧ t1'.FG = m)) :=
  sorry

end NUMINAMATH_CALUDE_least_possible_FG_l558_55889


namespace NUMINAMATH_CALUDE_geometric_arithmetic_interleaving_l558_55823

theorem geometric_arithmetic_interleaving (n : ℕ) (h : n > 3) :
  ∃ (x y : ℕ → ℕ),
    (∀ i, i < n → x i > 0) ∧
    (∀ i, i < n → y i > 0) ∧
    (∃ r : ℚ, r > 1 ∧ ∀ i, i < n - 1 → x (i + 1) = (x i : ℚ) * r) ∧
    (∃ d : ℚ, d > 0 ∧ ∀ i, i < n - 1 → y (i + 1) = y i + d) ∧
    (∀ i, i < n - 1 → x i < y i ∧ y i < x (i + 1)) ∧
    x (n - 1) < y (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_interleaving_l558_55823


namespace NUMINAMATH_CALUDE_jerrys_age_l558_55877

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 20 → 
  mickey_age = 2 * jerry_age - 8 → 
  jerry_age = 14 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l558_55877


namespace NUMINAMATH_CALUDE_g_of_3_equals_3_over_17_l558_55836

-- Define the function g
def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

-- State the theorem
theorem g_of_3_equals_3_over_17 : g 3 = 3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_3_over_17_l558_55836


namespace NUMINAMATH_CALUDE_carson_clawed_39_times_l558_55887

/-- The number of times Carson gets clawed in the zoo enclosure. -/
def total_claws (num_wombats : ℕ) (num_rheas : ℕ) (wombat_claws : ℕ) (rhea_claws : ℕ) : ℕ :=
  num_wombats * wombat_claws + num_rheas * rhea_claws

/-- Theorem stating that Carson gets clawed 39 times given the specific conditions. -/
theorem carson_clawed_39_times :
  total_claws 9 3 4 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_carson_clawed_39_times_l558_55887


namespace NUMINAMATH_CALUDE_exists_x_y_for_a_l558_55813

def a : ℕ → ℤ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * a (n + 1) - a n

def b : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => 2 * b (n + 1) + b n

theorem exists_x_y_for_a : ∃ (x y : ℕ → ℕ), ∀ n, 
  (y n)^2 + 7 = (x n - y n) * a n :=
sorry

end NUMINAMATH_CALUDE_exists_x_y_for_a_l558_55813


namespace NUMINAMATH_CALUDE_variance_of_specific_random_variable_l558_55845

/-- A random variable that takes values 0, 1, and 2 -/
structure RandomVariable where
  prob0 : ℝ
  prob1 : ℝ
  prob2 : ℝ
  sum_to_one : prob0 + prob1 + prob2 = 1
  nonnegative : prob0 ≥ 0 ∧ prob1 ≥ 0 ∧ prob2 ≥ 0

/-- The expectation of a random variable -/
def expectation (ξ : RandomVariable) : ℝ :=
  0 * ξ.prob0 + 1 * ξ.prob1 + 2 * ξ.prob2

/-- The variance of a random variable -/
def variance (ξ : RandomVariable) : ℝ :=
  (0 - expectation ξ)^2 * ξ.prob0 +
  (1 - expectation ξ)^2 * ξ.prob1 +
  (2 - expectation ξ)^2 * ξ.prob2

/-- Theorem: If P(ξ=0) = 1/5 and E(ξ) = 1, then D(ξ) = 2/5 -/
theorem variance_of_specific_random_variable :
  ∀ (ξ : RandomVariable),
    ξ.prob0 = 1/5 →
    expectation ξ = 1 →
    variance ξ = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_specific_random_variable_l558_55845


namespace NUMINAMATH_CALUDE_sum_and_product_positive_iff_both_positive_l558_55846

theorem sum_and_product_positive_iff_both_positive (a b : ℝ) :
  (a + b > 0 ∧ a * b > 0) ↔ (a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_positive_iff_both_positive_l558_55846


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_measure_l558_55868

theorem isosceles_triangle_smallest_angle_measure :
  ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Isosceles triangle condition
  c = 90 + 0.4 * 90  -- One angle is 40% larger than a right angle
  →
  a = 27 :=          -- One of the two smallest angles is 27°
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_measure_l558_55868


namespace NUMINAMATH_CALUDE_remaining_amount_correct_l558_55821

/-- Calculates the remaining amount in Will's original currency after shopping --/
def remaining_amount (initial_amount conversion_fee exchange_rate sweater_price tshirt_price
                      shoes_price hat_price socks_price shoe_refund_rate discount_rate
                      sales_tax_rate : ℚ) : ℚ :=
  let amount_after_fee := initial_amount - conversion_fee
  let local_currency_amount := amount_after_fee * exchange_rate
  let total_cost := sweater_price + tshirt_price + shoes_price + hat_price + socks_price
  let refund := shoes_price * shoe_refund_rate
  let cost_after_refund := total_cost - refund
  let discountable_items := sweater_price + tshirt_price + hat_price + socks_price
  let discount := discountable_items * discount_rate
  let cost_after_discount := cost_after_refund - discount
  let sales_tax := cost_after_discount * sales_tax_rate
  let final_cost := cost_after_discount + sales_tax
  let remaining_local := local_currency_amount - final_cost
  remaining_local / exchange_rate

/-- Theorem stating that the remaining amount is correct --/
theorem remaining_amount_correct :
  remaining_amount 74 2 (3/2) (27/2) (33/2) 45 (15/2) 6 (17/20) (1/10) (1/20) = (3987/100) := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_correct_l558_55821


namespace NUMINAMATH_CALUDE_nineteen_to_binary_l558_55892

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- State the theorem
theorem nineteen_to_binary :
  decimalToBinary 19 = [1, 0, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_nineteen_to_binary_l558_55892


namespace NUMINAMATH_CALUDE_smallest_multiple_l558_55874

theorem smallest_multiple (n : ℕ) : n = 204 ↔ 
  n > 0 ∧ 
  17 ∣ n ∧ 
  n % 43 = 11 ∧ 
  ∀ m : ℕ, m > 0 → 17 ∣ m → m % 43 = 11 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l558_55874


namespace NUMINAMATH_CALUDE_hyperbola_equation_l558_55838

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if point P(3, 5/2) lies on the hyperbola and the radius of the incircle of
    triangle PF₁F₂ (where F₁ and F₂ are the left and right foci) is 1,
    then a = 2 and b = √5. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (F₁ F₂ : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the hyperbola
    (F₁.1 < 0 ∧ F₂.1 > 0) ∧
    -- P(3, 5/2) lies on the hyperbola
    3^2 / a^2 - (5/2)^2 / b^2 = 1 ∧
    -- The radius of the incircle of triangle PF₁F₂ is 1
    (∃ (r : ℝ), r = 1 ∧
      r = (dist F₁ (3, 5/2) + dist F₂ (3, 5/2) + dist F₁ F₂) /
          (dist F₁ (3, 5/2) / r + dist F₂ (3, 5/2) / r + dist F₁ F₂ / r))) →
  a = 2 ∧ b = Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l558_55838


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l558_55808

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 < 4}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l558_55808


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l558_55834

theorem fractional_equation_solution :
  ∃ x : ℚ, (x + 1) / (4 * (x - 1)) = 2 / (3 * x - 3) - 1 ↔ x = 17 / 15 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l558_55834


namespace NUMINAMATH_CALUDE_trig_expression_equality_l558_55850

theorem trig_expression_equality : 
  2 * Real.sin (30 * π / 180) - Real.tan (45 * π / 180) - Real.sqrt ((1 - Real.tan (60 * π / 180))^2) = Real.sqrt 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l558_55850


namespace NUMINAMATH_CALUDE_y_increases_as_x_decreases_l558_55809

theorem y_increases_as_x_decreases (α : Real) (h_acute : 0 < α ∧ α < π / 2) :
  let f : Real → Real := λ x ↦ (Real.sin α - 1) * x - 6
  ∀ x₁ x₂ : Real, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_y_increases_as_x_decreases_l558_55809


namespace NUMINAMATH_CALUDE_root_equation_problem_l558_55829

theorem root_equation_problem (b c x₁ x₂ : ℝ) (y : ℝ) : 
  x₁ ≠ x₂ →
  (x₁^2 + 5*b*x₁ + c = 0) →
  (x₂^2 + 5*b*x₂ + c = 0) →
  (y^2 + 2*x₁*y + 2*x₂ = 0) →
  (y^2 + 2*x₂*y + 2*x₁ = 0) →
  b = 1/10 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l558_55829


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l558_55875

/-- A quadratic radical is considered simpler if it cannot be further simplified by factoring out perfect squares or simplifying fractions. -/
def is_simplest_quadratic_radical (x : ℝ) (options : List ℝ) : Prop :=
  x ∈ options ∧ 
  (∀ y ∈ options, x ≠ y → ∃ (n : ℕ) (m : ℚ), n > 1 ∧ y = n • (Real.sqrt m) ∨ ∃ (a b : ℚ), b ≠ 1 ∧ y = (Real.sqrt a) / b)

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 7) [Real.sqrt 12, Real.sqrt 7, Real.sqrt (2/3), Real.sqrt 0.2] :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l558_55875


namespace NUMINAMATH_CALUDE_sandwich_fraction_l558_55822

theorem sandwich_fraction (total : ℚ) (ticket : ℚ) (book : ℚ) (leftover : ℚ) 
  (h1 : total = 180)
  (h2 : ticket = 1/6)
  (h3 : book = 1/2)
  (h4 : leftover = 24) :
  ∃ (sandwich : ℚ), 
    sandwich * total + ticket * total + book * total = total - leftover ∧ 
    sandwich = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_fraction_l558_55822


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_14_l558_55844

/-- Given two points A and B in a 2D plane, where:
  - A is at (0, 0)
  - B is on the line y = 6
  - The slope of segment AB is 3/4
  This theorem proves that the sum of the x- and y-coordinates of point B is 14. -/
theorem sum_of_coordinates_is_14 (B : ℝ × ℝ) : 
  B.2 = 6 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 4 →
  B.1 + B.2 = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_14_l558_55844


namespace NUMINAMATH_CALUDE_range_of_a_l558_55839

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := a * x^2 - (3 - a) * x + 1

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x

/-- The theorem stating the range of a -/
theorem range_of_a : 
  {a : ℝ | ∀ x, max (f a x) (g x) > 0} = Set.Icc 0 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l558_55839


namespace NUMINAMATH_CALUDE_test_score_problem_l558_55870

theorem test_score_problem (total_questions : ℕ) (correct_points : ℚ) (incorrect_penalty : ℚ) 
  (final_score : ℚ) (h1 : total_questions = 120) (h2 : correct_points = 1) 
  (h3 : incorrect_penalty = 1/4) (h4 : final_score = 100) : 
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_penalty = final_score ∧
    correct_answers = 104 := by
  sorry

end NUMINAMATH_CALUDE_test_score_problem_l558_55870


namespace NUMINAMATH_CALUDE_class_average_l558_55893

theorem class_average (total_students : Nat) (perfect_scores : Nat) (zero_scores : Nat) (rest_average : Nat) : 
  total_students = 20 →
  perfect_scores = 2 →
  zero_scores = 3 →
  rest_average = 40 →
  (perfect_scores * 100 + zero_scores * 0 + (total_students - perfect_scores - zero_scores) * rest_average) / total_students = 40 := by
sorry

end NUMINAMATH_CALUDE_class_average_l558_55893


namespace NUMINAMATH_CALUDE_sticker_trade_result_l558_55805

/-- Calculates the final number of stickers after a given number of trades -/
def final_sticker_count (initial_count : ℕ) (num_trades : ℕ) : ℕ :=
  initial_count + num_trades * 4

/-- Theorem stating that after 50 trades, starting with 1 sticker, 
    the final count is 201 stickers -/
theorem sticker_trade_result : final_sticker_count 1 50 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sticker_trade_result_l558_55805


namespace NUMINAMATH_CALUDE_bus_interval_theorem_l558_55819

/-- Given a circular bus route with two buses operating at the same speed with an interval of 21 minutes,
    the interval between three buses operating on the same route at the same speed is 14 minutes. -/
theorem bus_interval_theorem (interval_two_buses : ℕ) (interval_three_buses : ℕ) : 
  interval_two_buses = 21 → interval_three_buses = (2 * interval_two_buses) / 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_interval_theorem_l558_55819


namespace NUMINAMATH_CALUDE_product_of_first_three_terms_l558_55816

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem product_of_first_three_terms 
  (a₁ : ℝ) -- first term
  (d : ℝ) -- common difference
  (h1 : arithmetic_sequence a₁ d 7 = 20) -- seventh term is 20
  (h2 : d = 2) -- common difference is 2
  : a₁ * (a₁ + d) * (a₁ + 2 * d) = 960 := by
  sorry

end NUMINAMATH_CALUDE_product_of_first_three_terms_l558_55816


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_999_l558_55881

/-- The sum of all digits of integers from 0 to 999 inclusive -/
def sumOfDigits : ℕ := sorry

/-- The range of integers we're considering -/
def integerRange : Set ℕ := { n | 0 ≤ n ∧ n ≤ 999 }

theorem sum_of_digits_0_to_999 : sumOfDigits = 13500 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_0_to_999_l558_55881


namespace NUMINAMATH_CALUDE_inequality_solution_set_l558_55873

theorem inequality_solution_set (x : ℝ) : 
  (3/16 : ℝ) + |x - 5/32| < 7/32 ↔ x ∈ Set.Ioo (1/8 : ℝ) (3/16 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l558_55873


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l558_55888

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (-2/3 : ℂ) + (1/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (-2/3 : ℂ) - (1/9 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l558_55888


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l558_55876

/-- Given a cylinder whose lateral surface unfolds into a rectangle with sides of length 6π and 4π,
    its total surface area is either 24π² + 18π or 24π² + 8π. -/
theorem cylinder_surface_area (h : ℝ) (r : ℝ) :
  (h = 6 * Real.pi ∧ 2 * Real.pi * r = 4 * Real.pi) ∨ 
  (h = 4 * Real.pi ∧ 2 * Real.pi * r = 6 * Real.pi) →
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l558_55876


namespace NUMINAMATH_CALUDE_number_of_factors_l558_55832

theorem number_of_factors (n : ℕ+) : 
  (Finset.range n).card = n :=
by sorry

#check number_of_factors

end NUMINAMATH_CALUDE_number_of_factors_l558_55832


namespace NUMINAMATH_CALUDE_basketball_court_fits_l558_55842

theorem basketball_court_fits (total_area : ℝ) (court_area : ℝ) (length_width_ratio : ℝ) (space_width : ℝ) :
  total_area = 1100 ∧ 
  court_area = 540 ∧ 
  length_width_ratio = 5/3 ∧
  space_width = 1 →
  ∃ (width : ℝ), 
    width > 0 ∧
    length_width_ratio * width * width = court_area ∧
    (length_width_ratio * width + 2 * space_width) * (width + 2 * space_width) ≤ total_area :=
by sorry

#check basketball_court_fits

end NUMINAMATH_CALUDE_basketball_court_fits_l558_55842


namespace NUMINAMATH_CALUDE_time_addition_theorem_l558_55811

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Adds a duration to a given time, wrapping around a 12-hour clock -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : ℕ) : Time :=
  sorry

/-- Converts 24-hour time to 12-hour time -/
def to12HourFormat (time : Time) : Time :=
  sorry

/-- Calculates the sum of hours, minutes, and seconds digits -/
def sumTimeDigits (time : Time) : ℕ :=
  sorry

theorem time_addition_theorem :
  let startTime := Time.mk 15 15 20
  let finalTime := addTime startTime 198 47 36
  let finalTime12Hour := to12HourFormat finalTime
  finalTime12Hour = Time.mk 10 2 56 ∧ sumTimeDigits finalTime12Hour = 68 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l558_55811


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l558_55885

theorem jelly_bean_probability (p_red p_orange : ℝ) 
  (h_red : p_red = 0.25)
  (h_orange : p_orange = 0.35)
  (h_sum : p_red + p_orange + (p_yellow + p_green) = 1) :
  p_yellow + p_green = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l558_55885


namespace NUMINAMATH_CALUDE_hcf_problem_l558_55824

/-- Given two positive integers with specific properties, prove their HCF is 24 -/
theorem hcf_problem (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a ≤ b) → 
  (b = 744) → 
  (Nat.lcm a b = Nat.gcd a b * 11 * 12) → 
  Nat.gcd a b = 24 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l558_55824


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l558_55860

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l558_55860


namespace NUMINAMATH_CALUDE_radar_arrangements_l558_55807

def word_length : ℕ := 5
def r_count : ℕ := 2
def a_count : ℕ := 2

theorem radar_arrangements : 
  (word_length.factorial) / (r_count.factorial * a_count.factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_radar_arrangements_l558_55807


namespace NUMINAMATH_CALUDE_julio_twice_james_age_l558_55862

/-- The age difference between Julio and James -/
def age_difference : ℕ := 36 - 11

/-- The number of years until Julio's age is twice James' age -/
def years_until_double : ℕ := 14

theorem julio_twice_james_age :
  36 + years_until_double = 2 * (11 + years_until_double) :=
sorry

end NUMINAMATH_CALUDE_julio_twice_james_age_l558_55862


namespace NUMINAMATH_CALUDE_trig_identity_l558_55801

theorem trig_identity (x y : ℝ) : 
  Real.sin (x - y) * Real.sin x + Real.cos (x - y) * Real.cos x = Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l558_55801


namespace NUMINAMATH_CALUDE_remainder_problem_l558_55835

theorem remainder_problem (N : ℕ) : 
  (∃ R : ℕ, N = 44 * 432 + R ∧ R < 44) → 
  (∃ Q : ℕ, N = 31 * Q + 5) → 
  (∃ R : ℕ, N = 44 * 432 + R ∧ R = 2) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l558_55835


namespace NUMINAMATH_CALUDE_apartment_utilities_cost_l558_55891

/-- Proves that the utilities cost for an apartment is $114 given specific conditions -/
theorem apartment_utilities_cost 
  (rent : ℕ) 
  (groceries : ℕ) 
  (one_roommate_payment : ℕ) 
  (h_rent : rent = 1100)
  (h_groceries : groceries = 300)
  (h_one_roommate : one_roommate_payment = 757)
  (h_equal_split : ∀ total_cost, one_roommate_payment * 2 = total_cost) :
  ∃ utilities : ℕ, utilities = 114 ∧ rent + utilities + groceries = one_roommate_payment * 2 :=
by sorry

end NUMINAMATH_CALUDE_apartment_utilities_cost_l558_55891


namespace NUMINAMATH_CALUDE_sticker_problem_solution_l558_55820

def sticker_problem (initial_stickers : ℕ) (front_page_stickers : ℕ) (stickers_per_page : ℕ) (remaining_stickers : ℕ) : ℕ :=
  (initial_stickers - remaining_stickers - front_page_stickers) / stickers_per_page

theorem sticker_problem_solution :
  sticker_problem 89 3 7 44 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sticker_problem_solution_l558_55820


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l558_55830

theorem cosine_sine_identity (θ : Real) (h : Real.tan θ = 1/3) :
  Real.cos θ ^ 2 + (1/2) * Real.sin (2 * θ) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l558_55830


namespace NUMINAMATH_CALUDE_max_product_sum_300_l558_55802

theorem max_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l558_55802


namespace NUMINAMATH_CALUDE_sin_sum_equality_l558_55895

theorem sin_sum_equality : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l558_55895


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l558_55879

theorem sin_negative_1740_degrees : Real.sin ((-1740 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l558_55879


namespace NUMINAMATH_CALUDE_class_size_l558_55859

theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 10) :
  football + tennis - both + neither = 39 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l558_55859


namespace NUMINAMATH_CALUDE_money_distribution_l558_55806

theorem money_distribution (a b c : ℤ) : 
  a + b + c = 900 → a + c = 400 → b + c = 750 → c = 250 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l558_55806


namespace NUMINAMATH_CALUDE_hyperbola_intersection_range_l558_55897

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def distinct_intersection (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂, x₁ ≠ x₂ →
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the dot product condition
def dot_product_condition (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂,
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ →
    x₁ * x₂ + y₁ * y₂ > 0

-- Main theorem
theorem hyperbola_intersection_range :
  ∀ k : ℝ, distinct_intersection k ∧ dot_product_condition k ↔
    (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_range_l558_55897


namespace NUMINAMATH_CALUDE_square_area_increase_l558_55863

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := 1.15 * s
  let new_area := new_side^2
  (new_area - original_area) / original_area * 100 = 32.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l558_55863
