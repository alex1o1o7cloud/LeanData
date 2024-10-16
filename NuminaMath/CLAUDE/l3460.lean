import Mathlib

namespace NUMINAMATH_CALUDE_bacon_suggestion_l3460_346063

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 408

/-- The difference between the number of students who suggested mashed potatoes and bacon -/
def difference : ℕ := 366

/-- The number of students who suggested adding bacon -/
def bacon : ℕ := mashed_potatoes - difference

theorem bacon_suggestion :
  bacon = 42 :=
by sorry

end NUMINAMATH_CALUDE_bacon_suggestion_l3460_346063


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3460_346078

theorem remainder_444_power_444_mod_13 : 444^444 ≡ 1 [MOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3460_346078


namespace NUMINAMATH_CALUDE_no_valid_domino_placement_without_2x2_square_l3460_346009

/-- Represents a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a domino placement on the chessboard -/
def DominoPlacement := List (Fin 8 × Fin 8 × Bool)

/-- Checks if a domino placement is valid (covers the entire board without overlaps) -/
def isValidPlacement (board : Chessboard) (placement : DominoPlacement) : Prop :=
  sorry

/-- Checks if a domino placement forms a 2x2 square -/
def forms2x2Square (placement : DominoPlacement) : Prop :=
  sorry

/-- The main theorem: it's impossible to cover an 8x8 chessboard with 2x1 dominoes
    without forming a 2x2 square -/
theorem no_valid_domino_placement_without_2x2_square :
  ¬ ∃ (board : Chessboard) (placement : DominoPlacement),
    isValidPlacement board placement ∧ ¬ forms2x2Square placement :=
  sorry

end NUMINAMATH_CALUDE_no_valid_domino_placement_without_2x2_square_l3460_346009


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l3460_346004

theorem factorial_sum_equality : 
  ∃! (w x y z : ℕ), w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  Nat.factorial w = Nat.factorial x + Nat.factorial y + Nat.factorial z ∧
  w = 3 ∧ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l3460_346004


namespace NUMINAMATH_CALUDE_sum_of_integers_l3460_346067

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 6) (h2 : a * b = 272) : a + b = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3460_346067


namespace NUMINAMATH_CALUDE_sin_15_degrees_l3460_346084

theorem sin_15_degrees : Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_degrees_l3460_346084


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3460_346098

theorem inequality_equivalence (x y : ℝ) : (y - x)^2 < x^2 ↔ y > 0 ∧ y < 2*x := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3460_346098


namespace NUMINAMATH_CALUDE_systematic_sampling_60_6_l3460_346095

def systematic_sampling (total : Nat) (sample_size : Nat) : List Nat :=
  sorry

theorem systematic_sampling_60_6 :
  systematic_sampling 60 6 = [7, 17, 27, 37, 47, 57] := by sorry

end NUMINAMATH_CALUDE_systematic_sampling_60_6_l3460_346095


namespace NUMINAMATH_CALUDE_sector_area_l3460_346036

theorem sector_area (centralAngle : Real) (radius : Real) : 
  centralAngle = π / 6 → radius = 2 → (1 / 2) * centralAngle * radius^2 = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3460_346036


namespace NUMINAMATH_CALUDE_dimitri_burger_calories_l3460_346072

/-- Calculates the number of calories per burger given the daily burger consumption and total calories over two days. -/
def calories_per_burger (burgers_per_day : ℕ) (total_calories : ℕ) : ℕ :=
  total_calories / (burgers_per_day * 2)

/-- Theorem stating that given Dimitri's burger consumption and calorie intake, each burger contains 20 calories. -/
theorem dimitri_burger_calories :
  calories_per_burger 3 120 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dimitri_burger_calories_l3460_346072


namespace NUMINAMATH_CALUDE_three_fifths_of_difference_l3460_346038

theorem three_fifths_of_difference : (3 : ℚ) / 5 * ((7 * 9) - (4 * 3)) = 153 / 5 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_of_difference_l3460_346038


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3460_346008

/-- A function f: ℝ⁺ → ℝ⁺ satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (y * f x) * (x + y) = x^2 * (f x + f y)

/-- The theorem stating that the only function satisfying the equation is f(x) = 1/x -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f → ∀ x, x > 0 → f x = 1 / x := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l3460_346008


namespace NUMINAMATH_CALUDE_total_egg_weight_in_pounds_l3460_346045

-- Define the weight of a single egg in pounds
def egg_weight : ℚ := 1 / 16

-- Define the number of dozens of eggs needed
def dozens_needed : ℕ := 8

-- Define the number of eggs in a dozen
def eggs_per_dozen : ℕ := 12

-- Theorem to prove
theorem total_egg_weight_in_pounds : 
  (dozens_needed * eggs_per_dozen : ℚ) * egg_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_egg_weight_in_pounds_l3460_346045


namespace NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l3460_346058

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1 / x - 1

theorem f_monotonicity_and_m_range :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧ 
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ m : ℝ, (∀ a : ℝ, -1 < a ∧ a < 1 → ∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ m * a - f x₀ < 0) ↔ 
    -1 / Real.exp 1 ≤ m ∧ m ≤ 1 / Real.exp 1) :=
sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l3460_346058


namespace NUMINAMATH_CALUDE_difference_of_largest_and_smallest_l3460_346093

def digits : List ℕ := [2, 7, 4, 9]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

def largest_number : ℕ := 974
def smallest_number : ℕ := 247

theorem difference_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_number) ∧
  largest_number - smallest_number = 727 :=
sorry

end NUMINAMATH_CALUDE_difference_of_largest_and_smallest_l3460_346093


namespace NUMINAMATH_CALUDE_pig_farmer_scenario_profit_l3460_346092

/-- Represents the profit calculation for a pig farmer --/
def pig_farmer_profit (num_piglets : ℕ) (sale_price : ℕ) (feed_cost : ℕ) 
  (months_group1 : ℕ) (months_group2 : ℕ) : ℕ :=
  let revenue := num_piglets * sale_price
  let cost_group1 := (num_piglets / 2) * feed_cost * months_group1
  let cost_group2 := (num_piglets / 2) * feed_cost * months_group2
  let total_cost := cost_group1 + cost_group2
  revenue - total_cost

/-- Theorem stating the profit for the given scenario --/
theorem pig_farmer_scenario_profit :
  pig_farmer_profit 6 300 10 12 16 = 960 :=
sorry

end NUMINAMATH_CALUDE_pig_farmer_scenario_profit_l3460_346092


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3460_346077

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 3, 4}

theorem complement_M_intersect_N : (U \ M) ∩ N = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3460_346077


namespace NUMINAMATH_CALUDE_greg_pages_per_day_l3460_346080

/-- 
Given that Brad reads 26 pages per day and 8 more pages than Greg each day,
prove that Greg reads 18 pages per day.
-/
theorem greg_pages_per_day 
  (brad_pages : ℕ) 
  (difference : ℕ) 
  (h1 : brad_pages = 26)
  (h2 : difference = 8)
  : brad_pages - difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_greg_pages_per_day_l3460_346080


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3460_346041

theorem rectangle_dimensions (length width : ℝ) : 
  (2 * length + 2 * width = 16) →  -- Perimeter is 16 cm
  (length - width = 1) →           -- Difference between length and width is 1 cm
  (length = 4.5 ∧ width = 3.5) :=  -- Length is 4.5 cm and width is 3.5 cm
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3460_346041


namespace NUMINAMATH_CALUDE_jolene_washed_five_cars_l3460_346012

/-- The number of cars Jolene washed to raise money for a bicycle -/
def cars_washed (families : ℕ) (babysitting_rate : ℕ) (car_wash_rate : ℕ) (total_raised : ℕ) : ℕ :=
  (total_raised - families * babysitting_rate) / car_wash_rate

/-- Theorem: Jolene washed 5 cars given the problem conditions -/
theorem jolene_washed_five_cars :
  cars_washed 4 30 12 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jolene_washed_five_cars_l3460_346012


namespace NUMINAMATH_CALUDE_solve_a_and_m_solve_inequality_l3460_346073

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1
theorem solve_a_and_m (a m : ℝ) : 
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
sorry

-- Theorem 2
theorem solve_inequality (t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) :=
sorry

end NUMINAMATH_CALUDE_solve_a_and_m_solve_inequality_l3460_346073


namespace NUMINAMATH_CALUDE_function_value_at_sqrt_two_l3460_346087

/-- Given a function f : ℝ → ℝ satisfying the equation 2 * f x + f (x^2 - 1) = 1 for all real x,
    prove that f(√2) = 1/3 -/
theorem function_value_at_sqrt_two (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, 2 * f x + f (x^2 - 1) = 1) : 
    f (Real.sqrt 2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_sqrt_two_l3460_346087


namespace NUMINAMATH_CALUDE_f_properties_l3460_346037

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/3) * x^3 + ((1-a)/2) * x^2 - a^2 * Real.log x + a^2 * Real.log a

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x > 0, f 1 x ≥ 1/3 ∧ f 1 1 = 1/3) ∧
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a > 3 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3460_346037


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l3460_346090

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 41 :=
by
  sorry

/-- The focal length of the hyperbola x²/16 - y²/25 = 1 is 2√41 -/
theorem specific_hyperbola_focal_length :
  let focal_length := 2 * Real.sqrt (16 + 25)
  focal_length = 2 * Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l3460_346090


namespace NUMINAMATH_CALUDE_sara_red_balloons_l3460_346099

def total_red_balloons : ℕ := 55
def sandy_red_balloons : ℕ := 24

theorem sara_red_balloons : ∃ (sara_balloons : ℕ), 
  sara_balloons + sandy_red_balloons = total_red_balloons ∧ sara_balloons = 31 := by
  sorry

end NUMINAMATH_CALUDE_sara_red_balloons_l3460_346099


namespace NUMINAMATH_CALUDE_range_of_a_l3460_346055

def p (a : ℝ) : Prop := a * (1 - a) > 0

def q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*a - 3)*x₁ + 1 = 0 ∧ 
  x₂^2 + (2*a - 3)*x₂ + 1 = 0

def S : Set ℝ := {a | a ≤ 0 ∨ (1/2 ≤ a ∧ a < 1) ∨ a > 5/2}

theorem range_of_a : {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} = S := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3460_346055


namespace NUMINAMATH_CALUDE_original_raw_silk_amount_l3460_346020

/-- Given information about silk drying process, prove the original amount of raw silk. -/
theorem original_raw_silk_amount 
  (initial_wet : ℚ) 
  (water_loss : ℚ) 
  (final_dry : ℚ) 
  (h1 : initial_wet = 30) 
  (h2 : water_loss = 3) 
  (h3 : final_dry = 12) : 
  (initial_wet * final_dry) / (initial_wet - water_loss) = 40 / 3 := by
  sorry

#check original_raw_silk_amount

end NUMINAMATH_CALUDE_original_raw_silk_amount_l3460_346020


namespace NUMINAMATH_CALUDE_val_money_value_l3460_346060

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of nickels Val initially has -/
def initial_nickels : ℕ := 20

/-- The number of dimes Val has -/
def dimes : ℕ := 3 * initial_nickels

/-- The number of additional nickels Val finds -/
def additional_nickels : ℕ := 2 * initial_nickels

/-- The total value of money Val has after taking the additional nickels -/
def total_value : ℚ := 
  (initial_nickels : ℚ) * nickel_value + 
  (dimes : ℚ) * dime_value + 
  (additional_nickels : ℚ) * nickel_value

theorem val_money_value : total_value = 9 := by
  sorry

end NUMINAMATH_CALUDE_val_money_value_l3460_346060


namespace NUMINAMATH_CALUDE_calculator_presses_to_exceed_250_l3460_346053

def calculator_function (x : ℕ) : ℕ := x^2 + 3

def iterate_calculator (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => calculator_function (iterate_calculator n m)

theorem calculator_presses_to_exceed_250 :
  ∃ n : ℕ, n > 0 ∧ iterate_calculator n 3 > 250 ∧ ∀ m : ℕ, m < n → iterate_calculator n m ≤ 250 :=
by sorry

end NUMINAMATH_CALUDE_calculator_presses_to_exceed_250_l3460_346053


namespace NUMINAMATH_CALUDE_total_earnings_proof_l3460_346030

def lauryn_earnings : ℝ := 2000
def aurelia_percentage : ℝ := 0.7

theorem total_earnings_proof :
  let aurelia_earnings := lauryn_earnings * aurelia_percentage
  lauryn_earnings + aurelia_earnings = 3400 :=
by sorry

end NUMINAMATH_CALUDE_total_earnings_proof_l3460_346030


namespace NUMINAMATH_CALUDE_valid_pairs_l3460_346015

def is_valid_pair (a b : ℕ) : Prop :=
  (a^2 + b) % (b^2 - a) = 0 ∧ (b^2 + a) % (a^2 - b) = 0

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ 
    ((a = 1 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 1) ∨ 
     (a = 2 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (a = 3 ∧ b = 2) ∨ 
     (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l3460_346015


namespace NUMINAMATH_CALUDE_complex_power_500_l3460_346083

theorem complex_power_500 : ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_500_l3460_346083


namespace NUMINAMATH_CALUDE_ratio_problem_l3460_346088

theorem ratio_problem (N : ℝ) (h1 : (1/1) * (1/3) * (2/5) * N = 25) (h2 : (40/100) * N = 300) :
  (25 : ℝ) / ((1/3) * (2/5) * N) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3460_346088


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3460_346066

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3460_346066


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3460_346097

theorem least_number_for_divisibility : ∃ (n : ℕ), n = 11 ∧
  (∀ (m : ℕ), m < n → ¬((1789 + m) % 5 = 0 ∧ (1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
  ((1789 + n) % 5 = 0 ∧ (1789 + n) % 6 = 0 ∧ (1789 + n) % 4 = 0 ∧ (1789 + n) % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3460_346097


namespace NUMINAMATH_CALUDE_birds_on_fence_proof_l3460_346003

/-- Given an initial number of birds and the number of birds remaining,
    calculate the number of birds that flew away. -/
def birds_flew_away (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem birds_on_fence_proof :
  let initial_birds : ℝ := 12.0
  let remaining_birds : ℕ := 4
  birds_flew_away initial_birds remaining_birds = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_proof_l3460_346003


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3460_346025

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (-2 * t^2 + 3 * t + 6) = 
  -6 * t^5 + 5 * t^4 + 4 * t^3 + 22 * t^2 + 27 * t - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3460_346025


namespace NUMINAMATH_CALUDE_max_checkers_on_6x6_board_l3460_346057

/-- A checker placement on a 6x6 board is represented as a list of 36 booleans -/
def CheckerPlacement := List Bool

/-- A function to check if three points are collinear on a 6x6 board -/
def areCollinear (p1 p2 p3 : Nat × Nat) : Bool :=
  sorry

/-- A function to check if a placement is valid (no three checkers are collinear) -/
def isValidPlacement (placement : CheckerPlacement) : Bool :=
  sorry

/-- The maximum number of checkers that can be placed on a 6x6 board
    such that no three checkers are collinear -/
def maxCheckers : Nat := 12

/-- Theorem stating that 12 is the maximum number of checkers
    that can be placed on a 6x6 board with no three collinear -/
theorem max_checkers_on_6x6_board :
  (∀ placement : CheckerPlacement,
    isValidPlacement placement → placement.length ≤ maxCheckers) ∧
  (∃ placement : CheckerPlacement,
    isValidPlacement placement ∧ placement.length = maxCheckers) :=
sorry

end NUMINAMATH_CALUDE_max_checkers_on_6x6_board_l3460_346057


namespace NUMINAMATH_CALUDE_f_value_l3460_346006

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (x + 1) = -f (-x + 1)
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, -1 ≤ x → x ≤ 0 → f x = -2 * x * (x + 1)

-- State the theorem to be proved
theorem f_value : f (-3/2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_l3460_346006


namespace NUMINAMATH_CALUDE_revenue_division_l3460_346085

theorem revenue_division (total_revenue : ℝ) (ratio_sum : ℕ) (salary_ratio rent_ratio marketing_ratio : ℕ) :
  total_revenue = 10000 →
  ratio_sum = 3 + 5 + 2 + 7 →
  salary_ratio = 3 →
  rent_ratio = 2 →
  marketing_ratio = 7 →
  (salary_ratio + rent_ratio + marketing_ratio) * (total_revenue / ratio_sum) = 7058.88 := by
  sorry

end NUMINAMATH_CALUDE_revenue_division_l3460_346085


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l3460_346022

theorem triangle_angle_relation (A B C C₁ C₂ : ℝ) : 
  B = 2 * A →
  C + A + B = Real.pi →
  C₁ + A = Real.pi / 2 →
  C₂ + B = Real.pi / 2 →
  C = C₁ + C₂ →
  C₁ - C₂ = A :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l3460_346022


namespace NUMINAMATH_CALUDE_mayor_cocoa_powder_l3460_346040

/-- The amount of cocoa powder given by the mayor for a chocolate cake -/
theorem mayor_cocoa_powder (total : ℕ) (additional : ℕ) (h : total > additional) :
  total - additional = (total - additional : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_mayor_cocoa_powder_l3460_346040


namespace NUMINAMATH_CALUDE_banana_survey_l3460_346023

theorem banana_survey (total_students : ℕ) (banana_percentage : ℚ) : 
  total_students = 100 →
  banana_percentage = 1/5 →
  (banana_percentage * total_students : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_banana_survey_l3460_346023


namespace NUMINAMATH_CALUDE_beam_cost_calculation_l3460_346075

/-- Represents the dimensions of a beam -/
structure BeamDimensions where
  thickness : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a beam given its dimensions -/
def beamVolume (d : BeamDimensions) : ℕ :=
  d.thickness * d.width * d.length

/-- Calculates the total volume of multiple beams with the same dimensions -/
def totalVolume (count : ℕ) (d : BeamDimensions) : ℕ :=
  count * beamVolume d

/-- Theorem: Given the cost of 30 beams with dimensions 12x16x14,
    the cost of 14 beams with dimensions 8x12x10 is 16 2/3 coins -/
theorem beam_cost_calculation (cost_30_beams : ℚ) :
  let d1 : BeamDimensions := ⟨12, 16, 14⟩
  let d2 : BeamDimensions := ⟨8, 12, 10⟩
  cost_30_beams = 100 →
  (14 : ℚ) * cost_30_beams * (totalVolume 14 d2 : ℚ) / (totalVolume 30 d1 : ℚ) = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_beam_cost_calculation_l3460_346075


namespace NUMINAMATH_CALUDE_triangle_obtuse_l3460_346054

theorem triangle_obtuse (A B C : Real) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  π / 2 < C ∧ C < π :=
sorry

end NUMINAMATH_CALUDE_triangle_obtuse_l3460_346054


namespace NUMINAMATH_CALUDE_equidistant_point_distance_l3460_346081

-- Define the equilateral triangle DEF
def triangle_DEF : Set (ℝ × ℝ × ℝ) := sorry

-- Define the side length of triangle DEF
def side_length : ℝ := 300

-- Define points X and Y
def X : ℝ × ℝ × ℝ := sorry
def Y : ℝ × ℝ × ℝ := sorry

-- Define the property that X and Y are equidistant from vertices of DEF
def equidistant_X (X : ℝ × ℝ × ℝ) : Prop := sorry
def equidistant_Y (Y : ℝ × ℝ × ℝ) : Prop := sorry

-- Define the 90° dihedral angle between planes XDE and YDE
def dihedral_angle_90 (X Y : ℝ × ℝ × ℝ) : Prop := sorry

-- Define point R
def R : ℝ × ℝ × ℝ := sorry

-- Define the distance r
def r : ℝ := sorry

-- Define the property that R is equidistant from D, E, F, X, and Y
def equidistant_R (R : ℝ × ℝ × ℝ) : Prop := sorry

theorem equidistant_point_distance :
  equidistant_X X →
  equidistant_Y Y →
  dihedral_angle_90 X Y →
  equidistant_R R →
  r = 50 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_equidistant_point_distance_l3460_346081


namespace NUMINAMATH_CALUDE_school_sections_l3460_346052

/-- The number of sections formed when dividing boys and girls into equal groups -/
def total_sections (num_boys num_girls : ℕ) : ℕ :=
  (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls))

/-- Theorem stating that for 408 boys and 192 girls, the total number of sections is 25 -/
theorem school_sections : total_sections 408 192 = 25 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l3460_346052


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3460_346046

theorem sin_2alpha_value (α : Real) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3460_346046


namespace NUMINAMATH_CALUDE_insect_count_l3460_346033

/-- Given a number of leaves, ladybugs per leaf, and ants per leaf, 
    calculate the total number of ladybugs and ants combined. -/
def total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) : ℕ :=
  leaves * ladybugs_per_leaf + leaves * ants_per_leaf

/-- Theorem stating that given 84 leaves, 139 ladybugs per leaf, and 97 ants per leaf,
    the total number of ladybugs and ants combined is 19,824. -/
theorem insect_count : total_insects 84 139 97 = 19824 := by
  sorry

end NUMINAMATH_CALUDE_insect_count_l3460_346033


namespace NUMINAMATH_CALUDE_marks_change_factor_l3460_346013

theorem marks_change_factor (n : ℕ) (initial_avg final_avg : ℝ) (h_n : n = 12) (h_initial : initial_avg = 50) (h_final : final_avg = 100) :
  (final_avg * n) / (initial_avg * n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_change_factor_l3460_346013


namespace NUMINAMATH_CALUDE_airport_exchange_calculation_l3460_346069

/-- Calculates the amount of dollars received when exchanging euros at an airport with a reduced exchange rate. -/
theorem airport_exchange_calculation (euros : ℝ) (normal_rate : ℝ) (airport_rate_fraction : ℝ) : 
  euros / normal_rate * airport_rate_fraction = 10 :=
by
  -- Assuming euros = 70, normal_rate = 5, and airport_rate_fraction = 5/7
  sorry

#check airport_exchange_calculation

end NUMINAMATH_CALUDE_airport_exchange_calculation_l3460_346069


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3460_346018

theorem sum_of_fractions : 
  7/8 + 11/12 = 43/24 ∧ 
  (∀ n d : ℤ, (n ≠ 0 ∨ d ≠ 1) → 43 * d ≠ 24 * n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3460_346018


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3460_346000

theorem absolute_value_inequality (x a : ℝ) 
  (h1 : |x - 4| + |x - 3| < a) 
  (h2 : a > 0) : 
  a > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3460_346000


namespace NUMINAMATH_CALUDE_miranda_savings_l3460_346031

theorem miranda_savings (total_cost sister_contribution saving_period : ℕ) 
  (h1 : total_cost = 260)
  (h2 : sister_contribution = 50)
  (h3 : saving_period = 3) :
  (total_cost - sister_contribution) / saving_period = 70 := by
  sorry

end NUMINAMATH_CALUDE_miranda_savings_l3460_346031


namespace NUMINAMATH_CALUDE_ln_plus_x_eq_three_solution_exists_in_two_three_l3460_346070

open Real

theorem ln_plus_x_eq_three_solution_exists_in_two_three :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ln_plus_x_eq_three_solution_exists_in_two_three_l3460_346070


namespace NUMINAMATH_CALUDE_min_droppers_required_l3460_346076

theorem min_droppers_required (container_volume : ℕ) (dropper_volume : ℕ) : container_volume = 265 → dropper_volume = 19 → (14 : ℕ) = (container_volume + dropper_volume - 1) / dropper_volume := by
  sorry

end NUMINAMATH_CALUDE_min_droppers_required_l3460_346076


namespace NUMINAMATH_CALUDE_quadratic_from_means_l3460_346019

theorem quadratic_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 8) 
  (h_geometric : Real.sqrt (a * b) = 12) : 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) := by
sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l3460_346019


namespace NUMINAMATH_CALUDE_feline_sanctuary_count_l3460_346035

theorem feline_sanctuary_count :
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let cougars : ℕ := (lions + tigers) / 2
  lions + tigers + cougars = 39 :=
by sorry

end NUMINAMATH_CALUDE_feline_sanctuary_count_l3460_346035


namespace NUMINAMATH_CALUDE_overlapping_rectangle_area_l3460_346051

theorem overlapping_rectangle_area (Y : ℝ) (X : ℝ) (h1 : Y > 0) (h2 : X > 0) 
  (h3 : X = (1/8) * (2*Y - X)) : X = (2/9) * Y := by
  sorry

end NUMINAMATH_CALUDE_overlapping_rectangle_area_l3460_346051


namespace NUMINAMATH_CALUDE_probability_heart_then_king_l3460_346029

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function to determine if a card is a heart -/
def is_heart (card : Fin 52) : Prop := sorry

/-- A function to determine if a card is a king -/
def is_king (card : Fin 52) : Prop := sorry

/-- The number of hearts in a standard deck -/
def num_hearts : Nat := 13

/-- The number of kings in a standard deck -/
def num_kings : Nat := 4

/-- The probability of drawing a heart as the first card and a king as the second card -/
theorem probability_heart_then_king (d : Deck) :
  (num_hearts / d.cards.val) * (num_kings / (d.cards.val - 1)) = 1 / d.cards.val :=
sorry

end NUMINAMATH_CALUDE_probability_heart_then_king_l3460_346029


namespace NUMINAMATH_CALUDE_work_completion_time_l3460_346047

/-- The time it takes for A to complete the entire work -/
def a_complete_time : ℝ := 21

/-- The time it takes for B to complete the entire work -/
def b_complete_time : ℝ := 15

/-- The number of days B worked before leaving -/
def b_worked_days : ℝ := 10

/-- The time it takes for A to complete the remaining work after B leaves -/
def a_remaining_time : ℝ := 7

theorem work_completion_time :
  a_complete_time = 21 →
  b_complete_time = 15 →
  b_worked_days = 10 →
  a_remaining_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3460_346047


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l3460_346082

theorem subtraction_of_decimals : 5.18 - 3.45 = 1.73 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l3460_346082


namespace NUMINAMATH_CALUDE_reciprocal_equation_l3460_346010

theorem reciprocal_equation (x : ℝ) : 
  (((5 * x - 1) / 6 - 2)⁻¹ = 3) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l3460_346010


namespace NUMINAMATH_CALUDE_definite_integrals_l3460_346056

theorem definite_integrals :
  (∫ (x : ℝ) in (-1)..(1), x^3) = 0 ∧
  (∫ (x : ℝ) in (2)..(ℯ + 1), 1 / (x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integrals_l3460_346056


namespace NUMINAMATH_CALUDE_five_objects_three_boxes_l3460_346042

/-- Number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object -/
def distributionCount (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 5 distinct objects into 3 distinct boxes,
    with each box containing at least one object, is equal to 150 -/
theorem five_objects_three_boxes : distributionCount 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_objects_three_boxes_l3460_346042


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l3460_346049

/-- The area of a square with perimeter equal to a triangle with sides 6.1, 8.2, and 9.7 -/
theorem square_area_equal_perimeter (s : Real) (h1 : s > 0) 
  (h2 : 4 * s = 6.1 + 8.2 + 9.7) : s^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l3460_346049


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l3460_346096

theorem mixed_number_calculation : 
  23 * ((1 + 2/3) + (2 + 1/4)) / ((1 + 1/2) + (1 + 1/5)) = 3 + 43/108 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l3460_346096


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_l3460_346065

-- Define the function f(x) = x^2 - 1
def f (x : ℝ) : ℝ := x^2 - 1

-- Theorem statement
theorem average_rate_of_change_f :
  (f 1.1 - f 1) / (1.1 - 1) = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_l3460_346065


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3460_346034

theorem smallest_positive_integer_with_remainders : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (∀ y : ℕ, y > 0 → (y % 3 = 2) → (y % 4 = 3) → (y % 5 = 4) → y ≥ x) ∧
  x = 59 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3460_346034


namespace NUMINAMATH_CALUDE_triangle_area_from_altitudes_l3460_346050

/-- Given a triangle ABC with altitudes h_a, h_b, and h_c, 
    its area S is equal to 
    1 / sqrt((1/h_a + 1/h_b + 1/h_c) * (1/h_a + 1/h_b - 1/h_c) * 
             (1/h_a + 1/h_c - 1/h_b) * (1/h_b + 1/h_c - 1/h_a)) -/
theorem triangle_area_from_altitudes (h_a h_b h_c : ℝ) (h_pos_a : h_a > 0) (h_pos_b : h_b > 0) (h_pos_c : h_c > 0) :
  let S := 1 / Real.sqrt ((1/h_a + 1/h_b + 1/h_c) * (1/h_a + 1/h_b - 1/h_c) * 
                          (1/h_a + 1/h_c - 1/h_b) * (1/h_b + 1/h_c - 1/h_a))
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    S = (a * h_a) / 2 ∧ S = (b * h_b) / 2 ∧ S = (c * h_c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_altitudes_l3460_346050


namespace NUMINAMATH_CALUDE_unique_a10a_divisible_by_12_l3460_346002

def is_form_a10a (n : ℕ) : Prop :=
  ∃ a : ℕ, a < 10 ∧ n = 1000 * a + 100 + 10 + a

theorem unique_a10a_divisible_by_12 :
  ∃! n : ℕ, is_form_a10a n ∧ n % 12 = 0 ∧ n = 4104 := by sorry

end NUMINAMATH_CALUDE_unique_a10a_divisible_by_12_l3460_346002


namespace NUMINAMATH_CALUDE_mouse_ratio_l3460_346094

/-- Represents the mouse distribution problem --/
def mouse_distribution (total_mice : ℕ) (robbie_fraction : ℚ) (store_multiple : ℕ) (feeder_fraction : ℚ) (remaining : ℕ) : Prop :=
  let robbie_mice := (total_mice : ℚ) * robbie_fraction
  let store_mice := (robbie_mice * store_multiple : ℚ)
  let before_feeder := (total_mice : ℚ) - robbie_mice - store_mice
  (before_feeder * feeder_fraction = (remaining : ℚ)) ∧
  (store_mice / robbie_mice = 3)

/-- Theorem stating the ratio of mice sold to pet store vs given to Robbie --/
theorem mouse_ratio :
  ∃ (store_multiple : ℕ),
    mouse_distribution 24 (1/6 : ℚ) store_multiple (1/2 : ℚ) 4 :=
sorry

end NUMINAMATH_CALUDE_mouse_ratio_l3460_346094


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l3460_346043

theorem contractor_daily_wage
  (total_days : ℕ)
  (absence_fine : ℚ)
  (total_payment : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : absence_fine = 7.5)
  (h3 : total_payment = 425)
  (h4 : absent_days = 10)
  : ∃ (daily_wage : ℚ), daily_wage = 25 ∧
    (total_days - absent_days) * daily_wage - absent_days * absence_fine = total_payment :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l3460_346043


namespace NUMINAMATH_CALUDE_group_ratio_l3460_346005

theorem group_ratio (x : ℝ) (h1 : x > 0) (h2 : 1 - x > 0) : 
  15 * x + 21 * (1 - x) = 20 → x / (1 - x) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_group_ratio_l3460_346005


namespace NUMINAMATH_CALUDE_pen_ratio_l3460_346074

/-- Represents the number of pens bought by each person -/
structure PenPurchase where
  dorothy : ℕ
  julia : ℕ
  robert : ℕ

/-- The cost of one pen in cents -/
def pen_cost : ℕ := 150

/-- The total amount spent by the three friends in cents -/
def total_spent : ℕ := 3300

/-- Conditions of the pen purchase -/
def pen_purchase_conditions (p : PenPurchase) : Prop :=
  p.julia = 3 * p.robert ∧
  p.robert = 4 ∧
  p.dorothy + p.julia + p.robert = total_spent / pen_cost

theorem pen_ratio (p : PenPurchase) 
  (h : pen_purchase_conditions p) : 
  p.dorothy * 2 = p.julia := by
  sorry

end NUMINAMATH_CALUDE_pen_ratio_l3460_346074


namespace NUMINAMATH_CALUDE_viktor_tally_theorem_l3460_346068

/-- Represents Viktor's tally system -/
structure TallySystem where
  x_value : ℕ  -- number of rallies represented by an X
  o_value : ℕ  -- number of rallies represented by an O

/-- Represents the final tally -/
structure FinalTally where
  o_count : ℕ  -- number of O's in the tally
  x_count : ℕ  -- number of X's in the tally

/-- Calculates the range of possible rallies given a tally system and final tally -/
def rally_range (system : TallySystem) (tally : FinalTally) : 
  {min_rallies : ℕ // ∃ max_rallies : ℕ, min_rallies ≤ max_rallies ∧ max_rallies ≤ min_rallies + system.x_value - 1} :=
sorry

theorem viktor_tally_theorem (system : TallySystem) (tally : FinalTally) :
  system.x_value = 10 ∧ 
  system.o_value = 100 ∧ 
  tally.o_count = 3 ∧ 
  tally.x_count = 7 →
  ∃ (range : {min_rallies : ℕ // ∃ max_rallies : ℕ, min_rallies ≤ max_rallies ∧ max_rallies ≤ min_rallies + system.x_value - 1}),
    range = rally_range system tally ∧
    range.val = 370 ∧
    (∃ max_rallies : ℕ, range.property.choose = 379) :=
sorry

end NUMINAMATH_CALUDE_viktor_tally_theorem_l3460_346068


namespace NUMINAMATH_CALUDE_dans_remaining_money_l3460_346014

theorem dans_remaining_money (initial_amount : ℚ) (candy_price : ℚ) (gum_price : ℚ) :
  initial_amount = 3.75 →
  candy_price = 1.25 →
  gum_price = 0.80 →
  initial_amount - (candy_price + gum_price) = 1.70 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l3460_346014


namespace NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l3460_346079

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) 
  (h_total : total = 100)
  (h_writers : writers = 45)
  (h_editors : editors > 36)
  (x : ℕ)
  (h_both : x = writers + editors - total + (total - writers - editors) / 2) :
  x ≤ 18 ∧ ∃ (e : ℕ), e > 36 ∧ x = writers + e - total + (total - writers - e) / 2 ∧ x = 18 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_overlap_l3460_346079


namespace NUMINAMATH_CALUDE_floor_sum_existence_l3460_346032

theorem floor_sum_existence : ∃ (a b c : ℝ), 
  (⌊a⌋ + ⌊b⌋ = ⌊a + b⌋) ∧ (⌊a⌋ + ⌊c⌋ < ⌊a + c⌋) := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_existence_l3460_346032


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l3460_346062

theorem no_solution_to_equation : ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3*x^2 - 15*x) / (x^2 - 5*x) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l3460_346062


namespace NUMINAMATH_CALUDE_existence_of_graph_with_chromatic_number_without_clique_l3460_346016

/-- A graph is a structure with vertices and an edge relation -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The chromatic number of a graph is the minimum number of colors needed to color its vertices 
    such that no two adjacent vertices have the same color -/
def chromaticNumber (G : Graph V) : ℕ := sorry

/-- An n-clique in a graph is a complete subgraph with n vertices -/
def hasClique (G : Graph V) (n : ℕ) : Prop := sorry

theorem existence_of_graph_with_chromatic_number_without_clique :
  ∀ n : ℕ, n > 3 → ∃ (V : Type) (G : Graph V), chromaticNumber G = n ∧ ¬hasClique G n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_graph_with_chromatic_number_without_clique_l3460_346016


namespace NUMINAMATH_CALUDE_prime_sum_1998_l3460_346061

theorem prime_sum_1998 (p q r : ℕ) (s t u : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h_eq : 1998 = p^s * q^t * r^u) : p + q + r = 42 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_1998_l3460_346061


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3460_346011

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 4 (-2 * k) 1 → k = 2 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3460_346011


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l3460_346028

theorem recurring_decimal_sum : 
  (∃ (x y : ℚ), x = 123 / 999 ∧ y = 123 / 999999 ∧ x + y = 154 / 1001) :=
by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l3460_346028


namespace NUMINAMATH_CALUDE_birdhouse_wood_pieces_l3460_346086

/-- The number of pieces of wood used for each birdhouse -/
def wood_pieces : ℕ := sorry

/-- The cost of wood per piece in dollars -/
def cost_per_piece : ℚ := 3/2

/-- The profit made on each birdhouse in dollars -/
def profit_per_birdhouse : ℚ := 11/2

/-- The total price for two birdhouses in dollars -/
def price_for_two : ℚ := 32

theorem birdhouse_wood_pieces :
  (2 * ((wood_pieces : ℚ) * cost_per_piece + profit_per_birdhouse) = price_for_two) →
  wood_pieces = 7 := by sorry

end NUMINAMATH_CALUDE_birdhouse_wood_pieces_l3460_346086


namespace NUMINAMATH_CALUDE_marias_age_half_anns_l3460_346026

/-- Proves that Maria's age was half of Ann's age 4 years ago -/
theorem marias_age_half_anns (maria_current_age ann_current_age years_ago : ℕ) : 
  maria_current_age = 7 →
  ann_current_age = maria_current_age + 3 →
  maria_current_age - years_ago = (ann_current_age - years_ago) / 2 →
  years_ago = 4 := by
sorry

end NUMINAMATH_CALUDE_marias_age_half_anns_l3460_346026


namespace NUMINAMATH_CALUDE_maggies_work_hours_l3460_346027

/-- Maggie's work hours problem -/
theorem maggies_work_hours 
  (office_rate : ℝ) 
  (tractor_rate : ℝ) 
  (total_income : ℝ) 
  (h1 : office_rate = 10)
  (h2 : tractor_rate = 12)
  (h3 : total_income = 416) : 
  ∃ (tractor_hours : ℝ),
    tractor_hours = 13 ∧ 
    office_rate * (2 * tractor_hours) + tractor_rate * tractor_hours = total_income :=
by sorry

end NUMINAMATH_CALUDE_maggies_work_hours_l3460_346027


namespace NUMINAMATH_CALUDE_exists_factorial_starting_with_2005_l3460_346059

theorem exists_factorial_starting_with_2005 : 
  ∃ (n : ℕ+), ∃ (k : ℕ), 2005 * 10^k ≤ n.val.factorial ∧ n.val.factorial < 2006 * 10^k :=
sorry

end NUMINAMATH_CALUDE_exists_factorial_starting_with_2005_l3460_346059


namespace NUMINAMATH_CALUDE_small_apple_cost_is_correct_l3460_346039

/-- The cost of a small apple -/
def small_apple_cost : ℚ := 1.5

/-- The cost of a medium apple -/
def medium_apple_cost : ℚ := 2

/-- The cost of a big apple -/
def big_apple_cost : ℚ := 3

/-- The number of small and medium apples bought -/
def small_medium_apples : ℕ := 6

/-- The number of big apples bought -/
def big_apples : ℕ := 8

/-- The total cost of all apples bought -/
def total_cost : ℚ := 45

/-- Theorem stating that the cost of each small apple is $1.50 -/
theorem small_apple_cost_is_correct : 
  small_apple_cost * small_medium_apples + 
  medium_apple_cost * small_medium_apples + 
  big_apple_cost * big_apples = total_cost := by
sorry

end NUMINAMATH_CALUDE_small_apple_cost_is_correct_l3460_346039


namespace NUMINAMATH_CALUDE_parallel_linear_function_through_point_l3460_346089

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- State the theorem
theorem parallel_linear_function_through_point :
  ∀ (k b : ℝ),
  -- The linear function is parallel to y = 2x + 1
  k = 2 →
  -- The linear function passes through the point (-1, 1)
  linear_function k b (-1) = 1 →
  -- The linear function is equal to y = 2x + 3
  linear_function k b = linear_function 2 3 := by
sorry


end NUMINAMATH_CALUDE_parallel_linear_function_through_point_l3460_346089


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3460_346048

theorem two_numbers_problem : ∃ (x y : ℕ), 
  (x + y = 1244) ∧ 
  (10 * x + 3 = (y - 2) / 10) ∧
  (x = 12) ∧ 
  (y = 1232) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3460_346048


namespace NUMINAMATH_CALUDE_marta_apples_weight_l3460_346064

/-- The weight of one apple in ounces -/
def apple_weight : ℕ := 4

/-- The weight of one orange in ounces -/
def orange_weight : ℕ := 3

/-- The maximum weight a bag can hold in ounces -/
def bag_capacity : ℕ := 49

/-- The number of bags Marta wants to buy -/
def num_bags : ℕ := 3

/-- The number of apples in one bag -/
def apples_per_bag : ℕ := 7

theorem marta_apples_weight : 
  apple_weight * apples_per_bag * num_bags = 84 ∧
  orange_weight * apples_per_bag * num_bags = 63 ∧
  apple_weight * apples_per_bag + orange_weight * apples_per_bag ≤ bag_capacity :=
by sorry

end NUMINAMATH_CALUDE_marta_apples_weight_l3460_346064


namespace NUMINAMATH_CALUDE_cone_volume_from_inscribed_cylinder_and_frustum_l3460_346024

/-- Given a cone with an inscribed cylinder and a truncated cone (frustum), 
    this theorem proves the volume of the original cone. -/
theorem cone_volume_from_inscribed_cylinder_and_frustum 
  (V_cylinder : ℝ) 
  (V_frustum : ℝ) 
  (h_cylinder : V_cylinder = 21) 
  (h_frustum : V_frustum = 91) : 
  ∃ (V_cone : ℝ), V_cone = 94.5 ∧ 
  ∃ (R r H h : ℝ), 
    R > 0 ∧ r > 0 ∧ H > 0 ∧ h > 0 ∧
    V_cylinder = π * r^2 * h ∧
    V_frustum = (1/3) * π * (R^2 + R*r + r^2) * (H - h) ∧
    R / r = 3 ∧
    h / H = 1/3 ∧
    V_cone = (1/3) * π * R^2 * H := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_inscribed_cylinder_and_frustum_l3460_346024


namespace NUMINAMATH_CALUDE_arrangement_problem_l3460_346091

def A (n m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

theorem arrangement_problem (n_boys n_girls : ℕ) (h_boys : n_boys = 6) (h_girls : n_girls = 4) :
  -- (I) Girls standing together
  (A n_girls n_girls * A (n_boys + 1) (n_boys + 1) = A 4 4 * A 7 7) ∧
  -- (II) No two girls adjacent
  (A n_boys n_boys * A (n_boys + 1) n_girls = A 6 6 * A 7 4) ∧
  -- (III) Boys A, B, C in alphabetical order
  (A (n_boys + n_girls) (n_boys + n_girls - 3) = A 10 7) :=
sorry

end NUMINAMATH_CALUDE_arrangement_problem_l3460_346091


namespace NUMINAMATH_CALUDE_unique_solution_system_l3460_346017

theorem unique_solution_system (x y z : ℝ) : 
  (x + y = 3 * x + 4) ∧ 
  (2 * y + 3 + z = 6 * y + 6) ∧ 
  (3 * z + 3 + x = 9 * z + 8) ↔ 
  (x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3460_346017


namespace NUMINAMATH_CALUDE_emilys_dogs_l3460_346021

theorem emilys_dogs (food_per_dog_per_day : ℕ) (vacation_days : ℕ) (total_food_kg : ℕ) :
  food_per_dog_per_day = 250 →
  vacation_days = 14 →
  total_food_kg = 14 →
  (total_food_kg * 1000) / (food_per_dog_per_day * vacation_days) = 4 :=
by sorry

end NUMINAMATH_CALUDE_emilys_dogs_l3460_346021


namespace NUMINAMATH_CALUDE_circle_area_above_line_l3460_346044

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 16*y + 48 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop := y = 4

-- Theorem statement
theorem circle_area_above_line :
  ∃ (A : ℝ), 
    (∀ x y : ℝ, circle_equation x y → 
      (y > 4 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ A})) ∧
    A = 24 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_above_line_l3460_346044


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3460_346007

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3460_346007


namespace NUMINAMATH_CALUDE_num_common_tangents_for_given_circles_l3460_346071

/-- Circle represented by its equation in the form (x - h)² + (y - k)² = r² --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Number of common tangents between two circles --/
def num_common_tangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Convert from x² + y² - 2ax = 0 form to Circle structure --/
def circle_from_equation (a : ℝ) : Circle :=
  { h := a, k := 0, r := a }

theorem num_common_tangents_for_given_circles :
  let c1 := circle_from_equation 1
  let c2 := circle_from_equation 2
  num_common_tangents c1 c2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_common_tangents_for_given_circles_l3460_346071


namespace NUMINAMATH_CALUDE_perpendicular_bisector_eq_l3460_346001

/-- The perpendicular bisector of a line segment MN is the set of all points
    equidistant from M and N. This theorem proves that for M(2, 4) and N(6, 2),
    the equation of the perpendicular bisector is 2x - y - 5 = 0. -/
theorem perpendicular_bisector_eq (x y : ℝ) :
  let M : ℝ × ℝ := (2, 4)
  let N : ℝ × ℝ := (6, 2)
  (x - M.1)^2 + (y - M.2)^2 = (x - N.1)^2 + (y - N.2)^2 ↔ 2*x - y - 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_eq_l3460_346001
