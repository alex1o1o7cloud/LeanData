import Mathlib

namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l6_600

noncomputable def f (k : ℝ) (c : ℝ) (x : ℝ) : ℝ := x^k + c

theorem range_of_f_on_interval (k : ℝ) (c : ℝ) (h : k > 0) :
  Set.range (fun x => f k c x) ∩ Set.Ici 1 = Set.Ici (1 + c) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l6_600


namespace NUMINAMATH_CALUDE_intersection_area_formula_l6_676

/-- Regular octahedron with side length s -/
structure RegularOctahedron where
  s : ℝ
  s_pos : 0 < s

/-- Plane parallel to two opposite faces of the octahedron -/
structure ParallelPlane where
  distance_ratio : ℝ
  is_one_third : distance_ratio = 1/3

/-- The intersection of the plane and the octahedron forms a polygon -/
def intersection_polygon (o : RegularOctahedron) (p : ParallelPlane) : Set (ℝ × ℝ) := sorry

/-- The area of the intersection polygon -/
def intersection_area (o : RegularOctahedron) (p : ParallelPlane) : ℝ := sorry

/-- Theorem: The area of the intersection polygon is √3 * s^2 / 6 -/
theorem intersection_area_formula (o : RegularOctahedron) (p : ParallelPlane) :
  intersection_area o p = (Real.sqrt 3 * o.s^2) / 6 := by sorry

end NUMINAMATH_CALUDE_intersection_area_formula_l6_676


namespace NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l6_672

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleC : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 - t.c^2 + t.b^2 = t.a * t.b

-- Theorem for part 1
theorem angle_C_measure (t : Triangle) (h : triangle_condition t) : 
  t.angleC = π / 3 := by sorry

-- Theorem for part 2
theorem triangle_area (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = 3) (h3 : t.b = 3) :
  (1/2) * t.a * t.b * Real.sin t.angleC = 9 * Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l6_672


namespace NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_is_884_l6_696

theorem max_sum_of_factors (a b : ℕ+) : 
  a * b = 1764 → ∀ x y : ℕ+, x * y = 1764 → a + b ≥ x + y :=
by sorry

theorem max_sum_is_884 : 
  ∃ a b : ℕ+, a * b = 1764 ∧ a + b = 884 ∧ 
  (∀ x y : ℕ+, x * y = 1764 → x + y ≤ 884) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_is_884_l6_696


namespace NUMINAMATH_CALUDE_dollar_operation_result_l6_697

/-- Custom dollar operation -/
def dollar (a b c : ℝ) : ℝ := (a - b + c)^2

/-- Theorem statement -/
theorem dollar_operation_result (x z : ℝ) :
  dollar ((x + z)^2) ((z - x)^2) ((x - z)^2) = (x + z)^4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_operation_result_l6_697


namespace NUMINAMATH_CALUDE_sequence_decreasing_l6_621

def x (a : ℝ) (n : ℕ) : ℝ := 2^n * (a^(1/(2*n)) - 1)

theorem sequence_decreasing (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∀ n : ℕ, x a n > x a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_decreasing_l6_621


namespace NUMINAMATH_CALUDE_sector_max_area_angle_l6_630

/-- Given a sector with circumference 36, the radian measure of the central angle
    that maximizes the area of the sector is 2. -/
theorem sector_max_area_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  2 * r + l = 36 →
  α = l / r →
  (∀ r' l' α', 2 * r' + l' = 36 → α' = l' / r' →
    r * l ≥ r' * l') →
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_angle_l6_630


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l6_650

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equation_sum_of_squares 
  (a b : ℝ) 
  (h : (a - 2 * i) * i = b - i) : 
  a^2 + b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l6_650


namespace NUMINAMATH_CALUDE_last_day_same_as_fifteenth_day_l6_623

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- A function to determine the day of the week for any day in the year,
    given the day of the week for the 15th day -/
def dayOfWeekFor (fifteenthDay : DayOfWeek) (dayNumber : Nat) : DayOfWeek :=
  sorry

theorem last_day_same_as_fifteenth_day 
  (year : Nat) 
  (h1 : year = 2005) 
  (h2 : (dayOfWeekFor DayOfWeek.Tuesday 15) = DayOfWeek.Tuesday) 
  (h3 : (dayOfWeekFor DayOfWeek.Tuesday 365) = (dayOfWeekFor DayOfWeek.Tuesday 15)) :
  (dayOfWeekFor DayOfWeek.Tuesday 365) = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_last_day_same_as_fifteenth_day_l6_623


namespace NUMINAMATH_CALUDE_randys_brother_biscuits_l6_660

/-- The number of biscuits Randy's brother ate -/
def biscuits_eaten (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (remaining : ℕ) : ℕ :=
  initial + from_father + from_mother - remaining

/-- Theorem stating the number of biscuits Randy's brother ate -/
theorem randys_brother_biscuits :
  biscuits_eaten 32 13 15 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_randys_brother_biscuits_l6_660


namespace NUMINAMATH_CALUDE_profit_difference_l6_617

def business_problem (capital_A capital_B capital_C capital_D capital_E profit_B : ℕ) : Prop :=
  let total_capital := capital_A + capital_B + capital_C + capital_D + capital_E
  let total_profit := profit_B * total_capital / capital_B
  let profit_C := total_profit * capital_C / total_capital
  let profit_E := total_profit * capital_E / total_capital
  profit_E - profit_C = 900

theorem profit_difference :
  business_problem 8000 10000 12000 15000 18000 1500 := by sorry

end NUMINAMATH_CALUDE_profit_difference_l6_617


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l6_680

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π/3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) : 
  Real.sqrt (((a.1 + 2*b.1) ^ 2) + ((a.2 + 2*b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l6_680


namespace NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l6_628

theorem cube_root_over_sixth_root_of_eight (x : ℝ) :
  (8 : ℝ) ^ (1/3) / (8 : ℝ) ^ (1/6) = (8 : ℝ) ^ (1/6) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_over_sixth_root_of_eight_l6_628


namespace NUMINAMATH_CALUDE_quadratic_coefficients_divisible_by_three_l6_638

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ := λ x ↦ a * x^2 + b * x + c

/-- The property that a polynomial is divisible by 3 for all integer inputs -/
def DivisibleByThreeForAllIntegers (P : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, P x = 3 * k

theorem quadratic_coefficients_divisible_by_three
  (a b c : ℤ)
  (h : DivisibleByThreeForAllIntegers (QuadraticPolynomial a b c)) :
  (∃ k₁ k₂ k₃ : ℤ, a = 3 * k₁ ∧ b = 3 * k₂ ∧ c = 3 * k₃) :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_divisible_by_three_l6_638


namespace NUMINAMATH_CALUDE_expression_not_constant_l6_681

theorem expression_not_constant : 
  ∀ x y : ℝ, x ≠ 3 → x ≠ -2 → y ≠ 3 → y ≠ -2 → x ≠ y → 
  (3*x^2 + 2*x - 5) / ((x-3)*(x+2)) - (5*x - 7) / ((x-3)*(x+2)) ≠ 
  (3*y^2 + 2*y - 5) / ((y-3)*(y+2)) - (5*y - 7) / ((y-3)*(y+2)) := by
  sorry

end NUMINAMATH_CALUDE_expression_not_constant_l6_681


namespace NUMINAMATH_CALUDE_michaels_brother_final_money_l6_632

/-- Given the initial conditions of Michael and his brother's money, and their subsequent actions,
    this theorem proves the final amount of money Michael's brother has. -/
theorem michaels_brother_final_money (michael_initial : ℕ) (brother_initial : ℕ) 
    (candy_cost : ℕ) (h1 : michael_initial = 42) (h2 : brother_initial = 17) 
    (h3 : candy_cost = 3) : 
    brother_initial + michael_initial / 2 - candy_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_michaels_brother_final_money_l6_632


namespace NUMINAMATH_CALUDE_comb_cost_is_one_l6_637

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℝ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℝ := 1

/-- Kristine's total purchase cost in dollars -/
def kristine_cost : ℝ := barrette_cost + comb_cost

/-- Crystal's total purchase cost in dollars -/
def crystal_cost : ℝ := 3 * barrette_cost + comb_cost

/-- The total amount spent by both girls in dollars -/
def total_spent : ℝ := 14

theorem comb_cost_is_one :
  kristine_cost + crystal_cost = total_spent → comb_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_comb_cost_is_one_l6_637


namespace NUMINAMATH_CALUDE_triangle_properties_l6_675

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem triangle_properties (t : Triangle)
  (h1 : t.A + t.B + t.C = π)
  (h2 : t.S > 0)
  (h3 : Real.tan (t.A / 2) * Real.tan (t.B / 2) + Real.sqrt 3 * (Real.tan (t.A / 2) + Real.tan (t.B / 2)) = 1) :
  t.C = 2 * π / 3 ∧ t.c^2 ≥ 4 * Real.sqrt 3 * t.S := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l6_675


namespace NUMINAMATH_CALUDE_factorization_equality_l6_648

theorem factorization_equality (a b : ℝ) : b^2 - a*b + a - b = (b - 1) * (b - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l6_648


namespace NUMINAMATH_CALUDE_modified_counting_game_45th_number_l6_643

/-- Represents the modified counting game sequence -/
def modifiedSequence (n : ℕ) : ℕ :=
  n + (n - 1) / 10

/-- The 45th number in the modified counting game is 54 -/
theorem modified_counting_game_45th_number : modifiedSequence 45 = 54 := by
  sorry

end NUMINAMATH_CALUDE_modified_counting_game_45th_number_l6_643


namespace NUMINAMATH_CALUDE_cookie_radius_is_8_l6_654

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 21 = 4*x + 18*y

/-- The radius of the cookie -/
def cookie_radius : ℝ := 8

/-- Theorem stating that the radius of the cookie defined by the equation is 8 -/
theorem cookie_radius_is_8 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = cookie_radius^2 :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_is_8_l6_654


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l6_665

/-- Represents Elaine's financial situation over two years -/
structure ElaineFinances where
  last_year_earnings : ℝ
  last_year_rent_percentage : ℝ
  this_year_earnings_increase : ℝ
  this_year_rent_percentage : ℝ
  rent_increase_percentage : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem elaine_rent_percentage
  (e : ElaineFinances)
  (h1 : e.this_year_earnings_increase = 0.15)
  (h2 : e.this_year_rent_percentage = 0.30)
  (h3 : e.rent_increase_percentage = 3.45)
  : e.last_year_rent_percentage = 0.10 := by
  sorry

#check elaine_rent_percentage

end NUMINAMATH_CALUDE_elaine_rent_percentage_l6_665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l6_639

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) → a 3 + a 7 = 2 * a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l6_639


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l6_614

theorem fraction_sum_squared (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l6_614


namespace NUMINAMATH_CALUDE_tv_price_changes_l6_604

theorem tv_price_changes (P : ℝ) (P_positive : P > 0) :
  let price_after_changes := P * 1.30 * 1.20 * 0.90 * 1.15
  let single_increase := 1.6146
  price_after_changes = P * single_increase :=
by sorry

end NUMINAMATH_CALUDE_tv_price_changes_l6_604


namespace NUMINAMATH_CALUDE_distance_between_points_l6_692

/-- The distance between two points given their net movements -/
theorem distance_between_points (south west : ℝ) (h : south = 30 ∧ west = 40) :
  Real.sqrt (south^2 + west^2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l6_692


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l6_609

/-- A positive integer n is a perfect square if there exists an integer k such that n = k^2 -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A positive integer n is a perfect fourth power if there exists an integer k such that n = k^4 -/
def IsPerfectFourthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^4

/-- The main theorem stating that 54 is the smallest positive integer satisfying the conditions -/
theorem smallest_n_satisfying_conditions : 
  (∀ m : ℕ, m > 0 ∧ m < 54 → ¬(IsPerfectSquare (2 * m) ∧ IsPerfectFourthPower (3 * m))) ∧ 
  (IsPerfectSquare (2 * 54) ∧ IsPerfectFourthPower (3 * 54)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l6_609


namespace NUMINAMATH_CALUDE_digits_1498_to_1500_form_229_l6_647

/-- A function that generates the list of positive integers starting with 2 -/
def integerListStartingWith2 : ℕ → ℕ
| 0 => 2
| n + 1 => 
  let prev := integerListStartingWith2 n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- A function that returns the nth digit in the concatenated list -/
def nthDigitInList (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 1498th, 1499th, and 1500th digits form 229 -/
theorem digits_1498_to_1500_form_229 : 
  (nthDigitInList 1498) * 100 + (nthDigitInList 1499) * 10 + nthDigitInList 1500 = 229 := by sorry

end NUMINAMATH_CALUDE_digits_1498_to_1500_form_229_l6_647


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l6_664

/-- A trinomial of the form ax^2 + bx + c is a perfect square if and only if
    there exist real numbers p and q such that ax^2 + bx + c = (px + q)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 9 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l6_664


namespace NUMINAMATH_CALUDE_max_money_is_zero_l6_657

/-- Represents the state of the stone piles and A's money --/
structure GameState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  money : ℤ

/-- Represents a move from one pile to another --/
inductive Move
  | one_to_two
  | one_to_three
  | two_to_one
  | two_to_three
  | three_to_one
  | three_to_two

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.one_to_two => 
      { pile1 := state.pile1 - 1, 
        pile2 := state.pile2 + 1, 
        pile3 := state.pile3,
        money := state.money + (state.pile2 - state.pile1 + 1) }
  | Move.one_to_three => 
      { pile1 := state.pile1 - 1, 
        pile2 := state.pile2, 
        pile3 := state.pile3 + 1,
        money := state.money + (state.pile3 - state.pile1 + 1) }
  | Move.two_to_one => 
      { pile1 := state.pile1 + 1, 
        pile2 := state.pile2 - 1, 
        pile3 := state.pile3,
        money := state.money + (state.pile1 - state.pile2 + 1) }
  | Move.two_to_three => 
      { pile1 := state.pile1, 
        pile2 := state.pile2 - 1, 
        pile3 := state.pile3 + 1,
        money := state.money + (state.pile3 - state.pile2 + 1) }
  | Move.three_to_one => 
      { pile1 := state.pile1 + 1, 
        pile2 := state.pile2, 
        pile3 := state.pile3 - 1,
        money := state.money + (state.pile1 - state.pile3 + 1) }
  | Move.three_to_two => 
      { pile1 := state.pile1, 
        pile2 := state.pile2 + 1, 
        pile3 := state.pile3 - 1,
        money := state.money + (state.pile2 - state.pile3 + 1) }

/-- Theorem: The maximum amount of money A can have when all stones return to their initial positions is 0 --/
theorem max_money_is_zero (initial : GameState) (moves : List Move) :
  (moves.foldl applyMove initial).pile1 = initial.pile1 ∧
  (moves.foldl applyMove initial).pile2 = initial.pile2 ∧
  (moves.foldl applyMove initial).pile3 = initial.pile3 →
  (moves.foldl applyMove initial).money ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_money_is_zero_l6_657


namespace NUMINAMATH_CALUDE_furniture_store_optimal_profit_l6_691

/-- Represents the furniture store's purchase and sales plan -/
structure FurnitureStore where
  a : ℝ  -- Original purchase price of dining table
  tableRetailPrice : ℝ := 270
  chairRetailPrice : ℝ := 70
  setPrice : ℝ := 500
  numTables : ℕ
  numChairs : ℕ

/-- Calculates the profit for the furniture store -/
def profit (store : FurnitureStore) : ℝ :=
  let numSets := store.numTables / 2
  let remainingTables := store.numTables - numSets
  let chairsInSets := numSets * 4
  let remainingChairs := store.numChairs - chairsInSets
  (store.setPrice - store.a - 4 * (store.a - 110)) * numSets +
  (store.tableRetailPrice - store.a) * remainingTables +
  (store.chairRetailPrice - (store.a - 110)) * remainingChairs

/-- The main theorem to be proved -/
theorem furniture_store_optimal_profit (store : FurnitureStore) :
  (600 / store.a = 160 / (store.a - 110)) →
  (store.numChairs = 5 * store.numTables + 20) →
  (store.numTables + store.numChairs ≤ 200) →
  (∃ (maxProfit : ℝ), 
    maxProfit = 7950 ∧ 
    store.a = 150 ∧ 
    store.numTables = 30 ∧ 
    store.numChairs = 170 ∧
    profit store = maxProfit ∧
    ∀ (otherStore : FurnitureStore), 
      (600 / otherStore.a = 160 / (otherStore.a - 110)) →
      (otherStore.numChairs = 5 * otherStore.numTables + 20) →
      (otherStore.numTables + otherStore.numChairs ≤ 200) →
      profit otherStore ≤ maxProfit) := by
  sorry

end NUMINAMATH_CALUDE_furniture_store_optimal_profit_l6_691


namespace NUMINAMATH_CALUDE_oil_production_scientific_notation_l6_649

theorem oil_production_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 45000000 = a * (10 : ℝ) ^ n ∧ a = 4.5 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_oil_production_scientific_notation_l6_649


namespace NUMINAMATH_CALUDE_two_solutions_l6_608

-- Define the equation
def equation (x a : ℝ) : Prop := abs (x - 3) = a * x - 1

-- Define the condition for two solutions
theorem two_solutions (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ a ∧ equation x₂ a) ↔ a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_l6_608


namespace NUMINAMATH_CALUDE_soda_price_ratio_l6_646

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio 
  (v : ℝ) -- Volume of Brand Y soda
  (p : ℝ) -- Price of Brand Y soda
  (h_v_pos : v > 0) -- Assumption that volume is positive
  (h_p_pos : p > 0) -- Assumption that price is positive
  : (0.85 * p) / (1.35 * v) / (p / v) = 17 / 27 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l6_646


namespace NUMINAMATH_CALUDE_journey_time_calculation_l6_652

/-- Proves that given a journey of 240 km completed in 5 hours, 
    where the first part is traveled at 40 kmph and the second part at 60 kmph, 
    the time spent on the first part of the journey is 3 hours. -/
theorem journey_time_calculation (total_distance : ℝ) (total_time : ℝ) 
    (speed_first_part : ℝ) (speed_second_part : ℝ) 
    (h1 : total_distance = 240)
    (h2 : total_time = 5)
    (h3 : speed_first_part = 40)
    (h4 : speed_second_part = 60) :
    ∃ (first_part_time : ℝ), 
      first_part_time * speed_first_part + 
      (total_time - first_part_time) * speed_second_part = total_distance ∧
      first_part_time = 3 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l6_652


namespace NUMINAMATH_CALUDE_russian_football_championship_l6_627

/-- Represents a football championship. -/
structure Championship where
  teams : Nat
  matches_per_pair : Nat

/-- Calculate the number of matches a single team plays in a season. -/
def matches_per_team (c : Championship) : Nat :=
  (c.teams - 1) * c.matches_per_pair

/-- Calculate the total number of matches in a season. -/
def total_matches (c : Championship) : Nat :=
  (c.teams * (c.teams - 1) * c.matches_per_pair) / 2

/-- Theorem stating the number of matches for a single team and total matches in the championship. -/
theorem russian_football_championship 
  (c : Championship) 
  (h1 : c.teams = 16) 
  (h2 : c.matches_per_pair = 2) : 
  matches_per_team c = 30 ∧ total_matches c = 240 := by
  sorry

#eval matches_per_team ⟨16, 2⟩
#eval total_matches ⟨16, 2⟩

end NUMINAMATH_CALUDE_russian_football_championship_l6_627


namespace NUMINAMATH_CALUDE_drivers_days_off_l6_674

/-- Proves that drivers get 5 days off per month given the specified conditions -/
theorem drivers_days_off 
  (num_drivers : ℕ) 
  (days_in_month : ℕ) 
  (total_cars : ℕ) 
  (maintenance_percentage : ℚ) 
  (h1 : num_drivers = 54)
  (h2 : days_in_month = 30)
  (h3 : total_cars = 60)
  (h4 : maintenance_percentage = 1/4) : 
  (days_in_month : ℚ) - (total_cars * (1 - maintenance_percentage) * days_in_month) / num_drivers = 5 := by
  sorry

end NUMINAMATH_CALUDE_drivers_days_off_l6_674


namespace NUMINAMATH_CALUDE_combined_area_square_triangle_l6_656

/-- The combined area of a square with diagonal 30 m and an equilateral triangle sharing that diagonal as its side is 450 m² + 225√3 m². -/
theorem combined_area_square_triangle (diagonal : ℝ) (h_diagonal : diagonal = 30) :
  let square_side := diagonal / Real.sqrt 2
  let square_area := square_side ^ 2
  let triangle_area := (Real.sqrt 3 / 4) * diagonal ^ 2
  square_area + triangle_area = 450 + 225 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_combined_area_square_triangle_l6_656


namespace NUMINAMATH_CALUDE_password_probability_l6_615

def positive_single_digit_numbers : ℕ := 9
def alphabet_size : ℕ := 26
def vowels : ℕ := 5

def even_single_digit_numbers : ℕ := 4
def numbers_greater_than_five : ℕ := 4

theorem password_probability : 
  (even_single_digit_numbers : ℚ) / positive_single_digit_numbers *
  (vowels : ℚ) / alphabet_size *
  (numbers_greater_than_five : ℚ) / positive_single_digit_numbers = 40 / 1053 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l6_615


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l6_610

theorem largest_stamps_per_page (book1 book2 book3 : Nat) 
  (h1 : book1 = 1050) 
  (h2 : book2 = 1260) 
  (h3 : book3 = 1470) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 210 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l6_610


namespace NUMINAMATH_CALUDE_plant_pricing_theorem_l6_618

/-- Represents the selling price per plant as a function of the number of plants per pot -/
def selling_price_per_plant (x : ℝ) : ℝ := -0.3 * x + 4.5

/-- Represents the price per pot as a function of the number of plants per pot -/
def price_per_pot (x : ℝ) : ℝ := -0.3 * x^2 + 4.5 * x

/-- Represents the cultivation cost per pot as a function of the number of plants -/
def cultivation_cost (x : ℝ) : ℝ := 2 + 0.3 * x

theorem plant_pricing_theorem :
  ∀ x : ℝ,
  5 ≤ x → x ≤ 12 →
  (selling_price_per_plant x = -0.3 * x + 4.5) ∧
  (price_per_pot x = -0.3 * x^2 + 4.5 * x) ∧
  ((price_per_pot x = 16.2) → (x = 6 ∨ x = 9)) ∧
  (∃ x : ℝ, (x = 12 ∨ x = 15) ∧
    30 * (price_per_pot x) - 40 * (cultivation_cost x) = 100) :=
by sorry


end NUMINAMATH_CALUDE_plant_pricing_theorem_l6_618


namespace NUMINAMATH_CALUDE_no_factorial_with_2021_zeros_l6_607

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- There is no natural number n such that n! ends with exactly 2021 zeros -/
theorem no_factorial_with_2021_zeros : ∀ n : ℕ, trailingZeros n ≠ 2021 := by
  sorry

end NUMINAMATH_CALUDE_no_factorial_with_2021_zeros_l6_607


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l6_644

theorem min_value_sum_squares (a b s : ℝ) (h : 2 * a + 2 * b = s) :
  ∃ (min : ℝ), min = s^2 / 2 ∧ ∀ (x y : ℝ), 2 * x + 2 * y = s → 2 * x^2 + 2 * y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l6_644


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l6_699

theorem unique_quadratic_solution (a c : ℤ) : 
  (∃! x : ℝ, a * x^2 + 36 * x + c = 0) →
  a + c = 37 →
  a < c →
  (a = 12 ∧ c = 25) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l6_699


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l6_694

theorem max_value_of_exponential_difference : 
  ∃ (M : ℝ), M = 1/4 ∧ ∀ (x : ℝ), 2^x - 16^x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l6_694


namespace NUMINAMATH_CALUDE_daisy_shop_total_sales_l6_640

def daisy_shop_sales (day1 : ℕ) (day2_increase : ℕ) (day3_decrease : ℕ) (day4 : ℕ) : ℕ :=
  let day2 := day1 + day2_increase
  let day3 := 2 * day2 - day3_decrease
  day1 + day2 + day3 + day4

theorem daisy_shop_total_sales :
  daisy_shop_sales 45 20 10 120 = 350 := by
  sorry

end NUMINAMATH_CALUDE_daisy_shop_total_sales_l6_640


namespace NUMINAMATH_CALUDE_book_arrangement_count_l6_631

/-- The number of ways to arrange books of different languages on a shelf --/
def arrange_books (total : ℕ) (italian : ℕ) (german : ℕ) (french : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial italian * Nat.factorial german * Nat.factorial french

/-- Theorem stating the number of arrangements for the given book problem --/
theorem book_arrangement_count :
  arrange_books 11 3 3 5 = 25920 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l6_631


namespace NUMINAMATH_CALUDE_polynomial_factorization_l6_613

theorem polynomial_factorization (x : ℤ) : 
  x^15 + x^8 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^8 - x^7 + x^6 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l6_613


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l6_682

structure PencilCase where
  pencils : ℕ
  pens : ℕ

def case : PencilCase := { pencils := 2, pens := 2 }

def select_two (pc : PencilCase) : ℕ := 2

def exactly_one_pen (pc : PencilCase) : Prop :=
  ∃ (x : ℕ), x = 1 ∧ x ≤ pc.pens

def exactly_two_pencils (pc : PencilCase) : Prop :=
  ∃ (x : ℕ), x = 2 ∧ x ≤ pc.pencils

theorem mutually_exclusive_not_opposite :
  (exactly_one_pen case ∧ exactly_two_pencils case → False) ∧
  ¬(exactly_one_pen case ↔ ¬exactly_two_pencils case) :=
by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_l6_682


namespace NUMINAMATH_CALUDE_S_equals_T_l6_622

def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

theorem S_equals_T : S = T := by sorry

end NUMINAMATH_CALUDE_S_equals_T_l6_622


namespace NUMINAMATH_CALUDE_sallys_nickels_l6_683

theorem sallys_nickels (x : ℕ) : x + 9 + 2 = 18 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_l6_683


namespace NUMINAMATH_CALUDE_negative_a_to_zero_power_l6_661

theorem negative_a_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_to_zero_power_l6_661


namespace NUMINAMATH_CALUDE_chair_distribution_count_l6_653

/-- The number of ways to distribute n identical objects into two groups,
    where one group must have at least a objects and the other group
    must have at least b objects. -/
def distribution_count (n a b : ℕ) : ℕ :=
  (n - a - b + 1).max 0

/-- Theorem: There are 5 ways to distribute 8 identical chairs into two groups,
    where one group (circle) must have at least 2 chairs and the other group (stack)
    must have at least 1 chair. -/
theorem chair_distribution_count : distribution_count 8 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chair_distribution_count_l6_653


namespace NUMINAMATH_CALUDE_proposition_B_is_false_l6_670

-- Define propositions as boolean variables
variable (p q : Prop)

-- Define the proposition B
def proposition_B (p q : Prop) : Prop :=
  (¬p ∧ ¬q) → (¬p ∧ ¬q)

-- Theorem stating that proposition B is false
theorem proposition_B_is_false :
  ∃ p q : Prop, ¬(proposition_B p q) :=
sorry

end NUMINAMATH_CALUDE_proposition_B_is_false_l6_670


namespace NUMINAMATH_CALUDE_johns_current_income_l6_629

/-- Calculates John's current yearly income based on tax rates and tax increase --/
theorem johns_current_income
  (initial_tax_rate : ℝ)
  (new_tax_rate : ℝ)
  (initial_income : ℝ)
  (tax_increase : ℝ)
  (h1 : initial_tax_rate = 0.20)
  (h2 : new_tax_rate = 0.30)
  (h3 : initial_income = 1000000)
  (h4 : tax_increase = 250000) :
  ∃ current_income : ℝ,
    current_income = 1500000 ∧
    new_tax_rate * current_income - initial_tax_rate * initial_income = tax_increase :=
by
  sorry


end NUMINAMATH_CALUDE_johns_current_income_l6_629


namespace NUMINAMATH_CALUDE_range_of_positives_in_K_l6_671

/-- Definition of the list K -/
def list_K : List ℤ := List.range 40 |>.map (fun i => -25 + 3 * i)

/-- The range of positive integers in list K -/
def positive_range (L : List ℤ) : ℤ :=
  let positives := L.filter (· > 0)
  positives.maximum.getD 0 - positives.minimum.getD 0

/-- Theorem: The range of positive integers in list K is 90 -/
theorem range_of_positives_in_K : positive_range list_K = 90 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positives_in_K_l6_671


namespace NUMINAMATH_CALUDE_translation_problem_l6_685

-- Define a translation of the complex plane
def translation (w : ℂ) : ℂ → ℂ := λ z ↦ z + w

-- Theorem statement
theorem translation_problem (w : ℂ) 
  (h : translation w (1 + 2*I) = 3 + 6*I) : 
  translation w (2 + 3*I) = 4 + 7*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l6_685


namespace NUMINAMATH_CALUDE_exists_real_for_special_sequence_l6_666

/-- A sequence of non-negative integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, n ≤ 1999 → a n ≥ 0) ∧
  (∀ i j, i + j ≤ 1999 → a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1)

/-- The main theorem -/
theorem exists_real_for_special_sequence (a : ℕ → ℕ) (h : SpecialSequence a) :
  ∃ x : ℝ, ∀ n : ℕ, n ≤ 1999 → a n = ⌊n * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_exists_real_for_special_sequence_l6_666


namespace NUMINAMATH_CALUDE_inequality_proof_l6_612

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l6_612


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_l6_606

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem
theorem parallel_to_y_axis (m n : ℝ) :
  let A : Point2D := ⟨-3, m⟩
  let B : Point2D := ⟨n, -4⟩
  (A.x = B.x) → -- Condition for line AB to be parallel to y-axis
  (n = -3 ∧ m ≠ -4) := by
  sorry


end NUMINAMATH_CALUDE_parallel_to_y_axis_l6_606


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l6_636

/-- The ellipse in the first quadrant -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1 ∧ x > 0 ∧ y > 0

/-- The dot product of OP and PF -/
def dot_product (x y : ℝ) : ℝ := x*(3-x) - y^2

theorem ellipse_dot_product_range :
  ∀ x y : ℝ, ellipse x y → -16 < dot_product x y ∧ dot_product x y ≤ -39/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l6_636


namespace NUMINAMATH_CALUDE_four_Y_three_l6_669

def Y (a b : ℝ) : ℝ := (a - b)^3 + 5

theorem four_Y_three : Y 4 3 = 6 := by sorry

end NUMINAMATH_CALUDE_four_Y_three_l6_669


namespace NUMINAMATH_CALUDE_average_monthly_balance_l6_695

def initial_balance : ℝ := 120
def february_change : ℝ := 80
def march_change : ℝ := -50
def april_change : ℝ := 70
def may_change : ℝ := 0
def june_change : ℝ := 100
def num_months : ℕ := 6

def monthly_balances : List ℝ := [
  initial_balance,
  initial_balance + february_change,
  initial_balance + february_change + march_change,
  initial_balance + february_change + march_change + april_change,
  initial_balance + february_change + march_change + april_change + may_change,
  initial_balance + february_change + march_change + april_change + may_change + june_change
]

theorem average_monthly_balance :
  (monthly_balances.sum / num_months) = 205 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l6_695


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l6_651

theorem dvd_rental_cost (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) 
  (h1 : total_cost = 4.8)
  (h2 : num_dvds = 4)
  (h3 : cost_per_dvd = total_cost / num_dvds) :
  cost_per_dvd = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l6_651


namespace NUMINAMATH_CALUDE_triangle_area_is_four_l6_655

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The area of a triangle given two sides and the sine of the included angle. -/
def triangleArea (s1 s2 sinAngle : ℝ) : ℝ :=
  0.5 * s1 * s2 * sinAngle

/-- The theorem stating that the area of the given triangle is 4. -/
theorem triangle_area_is_four (t : Triangle) 
    (ha : t.a = 2)
    (hc : t.c = 5)
    (hcosB : Real.cos t.angleB = 3/5) : 
    triangleArea t.a t.c (Real.sin t.angleB) = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_four_l6_655


namespace NUMINAMATH_CALUDE_theater_sales_proof_l6_688

/-- Calculates the total ticket sales for a theater performance. -/
def theater_sales (adult_price child_price total_attendance children_attendance : ℕ) : ℕ :=
  let adults := total_attendance - children_attendance
  let adult_sales := adults * adult_price
  let child_sales := children_attendance * child_price
  adult_sales + child_sales

/-- Theorem stating that given the specific conditions, the theater collects $50 from ticket sales. -/
theorem theater_sales_proof :
  theater_sales 8 1 22 18 = 50 := by
  sorry

end NUMINAMATH_CALUDE_theater_sales_proof_l6_688


namespace NUMINAMATH_CALUDE_range_of_a_l6_677

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l6_677


namespace NUMINAMATH_CALUDE_statements_B_and_C_are_correct_l6_620

theorem statements_B_and_C_are_correct (a b c d : ℝ) :
  (((a * b > 0 ∧ b * c - a * d > 0) → (c / a - d / b > 0)) ∧
   ((a > b ∧ c > d) → (a - d > b - c))) := by
  sorry

end NUMINAMATH_CALUDE_statements_B_and_C_are_correct_l6_620


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_range_l6_626

/-- Given f(x) = |x+a| + |x-1/a| where a ≠ 0, if for all x ∈ ℝ, f(x) ≥ |m-1|, then m ∈ [-1, 3] -/
theorem function_inequality_implies_m_range (a m : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, |x + a| + |x - 1/a| ≥ |m - 1|) → m ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_range_l6_626


namespace NUMINAMATH_CALUDE_alice_pens_count_l6_601

/-- Proves that Alice has 60 pens given the conditions of the problem -/
theorem alice_pens_count :
  ∀ (alice_pens clara_pens alice_age clara_age : ℕ),
    clara_pens = (2 * alice_pens) / 5 →
    alice_pens - clara_pens = clara_age - alice_age →
    alice_age = 20 →
    clara_age > alice_age →
    clara_age + 5 = 61 →
    alice_pens = 60 := by
  sorry

end NUMINAMATH_CALUDE_alice_pens_count_l6_601


namespace NUMINAMATH_CALUDE_circle_center_sum_l6_684

/-- Given a circle with equation x^2 + y^2 - 6x + 8y + 9 = 0, 
    prove that the sum of the coordinates of its center is -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 6*x + 8*y + 9 = 0 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9)) →
  h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l6_684


namespace NUMINAMATH_CALUDE_train_crossing_time_l6_667

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 200 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l6_667


namespace NUMINAMATH_CALUDE_jacket_price_proof_l6_634

theorem jacket_price_proof (S P : ℝ) (h1 : S = P + 0.4 * S) 
  (h2 : 0.8 * S - P = 18) : P = 54 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_proof_l6_634


namespace NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l6_687

/-- Represents a gathering of people with specific knowledge relationships -/
structure Gathering where
  total : Nat
  group1 : Nat
  group2 : Nat
  group2_with_connections : Nat
  group2_without_connections : Nat

/-- Calculates the number of handshakes in the gathering -/
def count_handshakes (g : Gathering) : Nat :=
  let group2_no_connections_handshakes := g.group2_without_connections * (g.total - 1)
  let group2_with_connections_handshakes := g.group2_with_connections * (g.total - 11)
  (group2_no_connections_handshakes + group2_with_connections_handshakes) / 2

/-- Theorem stating the number of handshakes in the specific gathering -/
theorem handshakes_in_specific_gathering :
  let g : Gathering := {
    total := 40,
    group1 := 25,
    group2 := 15,
    group2_with_connections := 5,
    group2_without_connections := 10
  }
  count_handshakes g = 305 := by
  sorry

#eval count_handshakes {
  total := 40,
  group1 := 25,
  group2 := 15,
  group2_with_connections := 5,
  group2_without_connections := 10
}

end NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l6_687


namespace NUMINAMATH_CALUDE_gcd_problem_l6_605

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2700 * k) :
  Int.gcd (b^2 + 27*b + 75) (b + 25) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l6_605


namespace NUMINAMATH_CALUDE_special_function_is_x_plus_one_l6_633

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

/-- Theorem stating that the special function is x + 1 -/
theorem special_function_is_x_plus_one (f : ℝ → ℝ) (hf : special_function f) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_x_plus_one_l6_633


namespace NUMINAMATH_CALUDE_f_composition_value_l6_616

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem f_composition_value : f (f (f (-2))) = 4 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l6_616


namespace NUMINAMATH_CALUDE_geometric_series_sum_l6_693

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1/4
  let n : ℕ := 5
  geometricSum a r n = 341/256 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l6_693


namespace NUMINAMATH_CALUDE_single_room_cost_l6_619

/-- Proves that the cost of each single room is $35 given the hotel booking information -/
theorem single_room_cost (total_rooms : ℕ) (double_room_cost : ℕ) (total_revenue : ℕ) (double_rooms : ℕ)
  (h1 : total_rooms = 260)
  (h2 : double_room_cost = 60)
  (h3 : total_revenue = 14000)
  (h4 : double_rooms = 196) :
  (total_revenue - double_rooms * double_room_cost) / (total_rooms - double_rooms) = 35 := by
  sorry

#check single_room_cost

end NUMINAMATH_CALUDE_single_room_cost_l6_619


namespace NUMINAMATH_CALUDE_sum_of_solutions_equation_l6_690

theorem sum_of_solutions_equation : ∃ (x₁ x₂ : ℝ), 
  (4 * x₁ + 3) * (3 * x₁ - 7) = 0 ∧
  (4 * x₂ + 3) * (3 * x₂ - 7) = 0 ∧
  x₁ + x₂ = 19 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equation_l6_690


namespace NUMINAMATH_CALUDE_binary_equals_base_4_l6_642

-- Define the binary number
def binary_num : List Bool := [true, false, true, false, true, true, true, false, true]

-- Define the base 4 number
def base_4_num : List Nat := [1, 1, 3, 1]

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : Nat :=
  bin.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert base 4 to decimal
def base_4_to_decimal (b4 : List Nat) : Nat :=
  b4.reverse.enum.foldl (fun acc (i, d) => acc + d * 4^i) 0

-- Theorem statement
theorem binary_equals_base_4 :
  binary_to_decimal binary_num = base_4_to_decimal base_4_num := by
  sorry

end NUMINAMATH_CALUDE_binary_equals_base_4_l6_642


namespace NUMINAMATH_CALUDE_D_72_eq_45_l6_611

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where order matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- Theorem stating that D(72) is equal to 45 -/
theorem D_72_eq_45 : D 72 = 45 := by sorry

end NUMINAMATH_CALUDE_D_72_eq_45_l6_611


namespace NUMINAMATH_CALUDE_rectangular_field_area_l6_668

/-- A rectangular field with width one-third of its length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  w = l / 3 → 2 * (w + l) = 72 → w * l = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l6_668


namespace NUMINAMATH_CALUDE_family_road_trip_l6_658

/-- A theorem about a family's road trip with constant speed -/
theorem family_road_trip 
  (total_time : ℝ) 
  (first_part_distance : ℝ) 
  (first_part_time : ℝ) 
  (h1 : total_time = 4) 
  (h2 : first_part_distance = 100) 
  (h3 : first_part_time = 1) :
  let speed := first_part_distance / first_part_time
  let remaining_time := total_time - first_part_time
  remaining_time * speed = 300 := by
  sorry

#check family_road_trip

end NUMINAMATH_CALUDE_family_road_trip_l6_658


namespace NUMINAMATH_CALUDE_marilyn_final_bottle_caps_l6_698

/-- Calculates the final number of bottle caps Marilyn has after a series of exchanges --/
def final_bottle_caps (initial : ℕ) (shared : ℕ) (received : ℕ) : ℕ :=
  let remaining := initial - shared + received
  remaining - remaining / 2

/-- Theorem stating that Marilyn ends up with 55 bottle caps --/
theorem marilyn_final_bottle_caps : 
  final_bottle_caps 165 78 23 = 55 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_final_bottle_caps_l6_698


namespace NUMINAMATH_CALUDE_helpers_count_l6_625

theorem helpers_count (pouches_per_pack : ℕ) (team_members : ℕ) (coaches : ℕ) (packs_bought : ℕ) :
  pouches_per_pack = 6 →
  team_members = 13 →
  coaches = 3 →
  packs_bought = 3 →
  (pouches_per_pack * packs_bought) - (team_members + coaches) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_helpers_count_l6_625


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l6_645

theorem max_gcd_of_sequence (n : ℕ+) : 
  Nat.gcd (99 + n^2) (99 + (n + 1)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l6_645


namespace NUMINAMATH_CALUDE_f_mapping_result_l6_663

def A : Set (ℝ × ℝ) := Set.univ

def B : Set (ℝ × ℝ) := Set.univ

def f : (ℝ × ℝ) → (ℝ × ℝ) := λ (x, y) ↦ (x - y, x + y)

theorem f_mapping_result : f (-1, 2) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_f_mapping_result_l6_663


namespace NUMINAMATH_CALUDE_circle_area_with_radius_four_l6_662

theorem circle_area_with_radius_four (π : ℝ) : 
  let r : ℝ := 4
  let area := π * r^2
  area = 16 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_radius_four_l6_662


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l6_686

theorem complex_modulus_problem (x y : ℝ) (h : (x + Complex.I) * x = 4 + 2 * y * Complex.I) :
  Complex.abs ((x + 4 * y * Complex.I) / (1 + Complex.I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l6_686


namespace NUMINAMATH_CALUDE_laundry_time_calculation_l6_602

theorem laundry_time_calculation (loads : ℕ) (wash_time dry_time : ℕ) : 
  loads = 8 → 
  wash_time = 45 → 
  dry_time = 60 → 
  (loads * (wash_time + dry_time)) / 60 = 14 := by
sorry

end NUMINAMATH_CALUDE_laundry_time_calculation_l6_602


namespace NUMINAMATH_CALUDE_no_real_roots_l6_678

theorem no_real_roots (A B : ℝ) : 
  (∀ x y : ℝ, x^2 + x*y + y = A ∧ y / (y - x) = B → False) ↔ A = 2 ∧ B = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l6_678


namespace NUMINAMATH_CALUDE_fraction_mediant_l6_673

theorem fraction_mediant (r s u v : ℚ) (l m : ℕ+) 
  (h1 : 0 < r) (h2 : 0 < s) (h3 : 0 < u) (h4 : 0 < v) 
  (h5 : s * u - r * v = 1) : 
  (∀ x, r / u < x ∧ x < s / v → 
    ∃ l m : ℕ+, x = (l * r + m * s) / (l * u + m * v)) ∧
  (r / u < (l * r + m * s) / (l * u + m * v) ∧ 
   (l * r + m * s) / (l * u + m * v) < s / v) :=
sorry

end NUMINAMATH_CALUDE_fraction_mediant_l6_673


namespace NUMINAMATH_CALUDE_smallest_divisible_by_five_million_l6_689

def geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem smallest_divisible_by_five_million :
  let a₁ := 2
  let a₂ := 70
  let r := a₂ / a₁
  ∀ n : ℕ, n > 0 →
    (is_divisible_by (geometric_sequence a₁ r n) 5000000 ∧
     ∀ m : ℕ, 0 < m → m < n →
       ¬ is_divisible_by (geometric_sequence a₁ r m) 5000000) →
    n = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_five_million_l6_689


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l6_624

theorem polynomial_coefficient_b (a b c d : ℝ) : 
  (∃ (z w : ℂ), z * w = 9 - 3*I ∧ z + w = -2 - 6*I) →
  (∀ (r : ℂ), r^4 + a*r^3 + b*r^2 + c*r + d = 0 → r.im ≠ 0) →
  b = 58 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l6_624


namespace NUMINAMATH_CALUDE_tangent_line_at_y_axis_l6_659

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1)

theorem tangent_line_at_y_axis (x : ℝ) :
  let y_intercept := f 0
  let slope := (deriv f) 0
  (fun x => slope * x + y_intercept) = (fun x => 2 * Real.exp 1 * x + Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_y_axis_l6_659


namespace NUMINAMATH_CALUDE_tan_product_30_15_l6_641

theorem tan_product_30_15 :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_30_15_l6_641


namespace NUMINAMATH_CALUDE_john_finish_time_l6_635

-- Define the start time of the first task
def start_time : Nat := 14 * 60 + 30  -- 2:30 PM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 16 * 60 + 20  -- 4:20 PM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem statement
theorem john_finish_time :
  let task_duration := (end_second_task - start_time) / 2
  let finish_time := end_second_task + 2 * task_duration
  finish_time = 18 * 60 + 10  -- 6:10 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_john_finish_time_l6_635


namespace NUMINAMATH_CALUDE_new_apples_grown_l6_679

/-- Given a tree with apples, calculate the number of new apples grown -/
theorem new_apples_grown
  (initial_apples : ℕ)
  (picked_apples : ℕ)
  (current_apples : ℕ)
  (h1 : initial_apples = 4)
  (h2 : picked_apples = 2)
  (h3 : current_apples = 5)
  (h4 : picked_apples ≤ initial_apples) :
  current_apples - (initial_apples - picked_apples) = 3 :=
by sorry

end NUMINAMATH_CALUDE_new_apples_grown_l6_679


namespace NUMINAMATH_CALUDE_function_inequality_l6_603

/-- Given a function f(x) = axe^x where a ≠ 0 and a ≥ 4/e^2, 
    prove that f(x)/(x+1) - (x+1)ln(x) > 0 for x > 0 -/
theorem function_inequality (a : ℝ) (h1 : a ≠ 0) (h2 : a ≥ 4 / Real.exp 2) :
  ∀ x > 0, (a * x * Real.exp x) / (x + 1) - (x + 1) * Real.log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l6_603
