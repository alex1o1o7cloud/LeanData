import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_B_l2295_229581

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, n = 6 * x + 6}

theorem gcd_of_B : ∃ d : ℕ, d > 0 ∧ ∀ n ∈ B, d ∣ n ∧ ∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d :=
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_l2295_229581


namespace NUMINAMATH_CALUDE_condition_relationship_l2295_229575

theorem condition_relationship (x : ℝ) : 
  (∀ x, x^2 - 2*x + 1 ≤ 0 → x > 0) ∧ 
  (∃ x, x > 0 ∧ x^2 - 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2295_229575


namespace NUMINAMATH_CALUDE_ab_greater_than_b_squared_l2295_229538

theorem ab_greater_than_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_b_squared_l2295_229538


namespace NUMINAMATH_CALUDE_coffee_cost_for_three_dozen_l2295_229589

/-- Calculates the cost of coffee for a given number of dozens of donuts -/
def coffee_cost (dozens : ℕ) : ℕ :=
  let donuts_per_dozen : ℕ := 12
  let coffee_per_donut : ℕ := 2
  let coffee_per_pot : ℕ := 12
  let cost_per_pot : ℕ := 3
  let total_donuts : ℕ := dozens * donuts_per_dozen
  let total_coffee : ℕ := total_donuts * coffee_per_donut
  let pots_needed : ℕ := (total_coffee + coffee_per_pot - 1) / coffee_per_pot
  pots_needed * cost_per_pot

theorem coffee_cost_for_three_dozen : coffee_cost 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_for_three_dozen_l2295_229589


namespace NUMINAMATH_CALUDE_right_triangles_count_l2295_229514

/-- Represents a geometric solid with front, top, and side views -/
structure GeometricSolid where
  front_view : Set (Point × Point)
  top_view : Set (Point × Point)
  side_view : Set (Point × Point)

/-- Counts the number of unique right-angled triangles in a geometric solid -/
def count_right_triangles (solid : GeometricSolid) : ℕ :=
  sorry

/-- Theorem stating that the number of right-angled triangles is 3 -/
theorem right_triangles_count (solid : GeometricSolid) :
  count_right_triangles solid = 3 :=
sorry

end NUMINAMATH_CALUDE_right_triangles_count_l2295_229514


namespace NUMINAMATH_CALUDE_inverse_proportion_l2295_229507

/-- Given that α is inversely proportional to β, prove that if α = -3 when β = -6, 
    then α = 9/4 when β = 8. -/
theorem inverse_proportion (α β : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ x, α x * β x = k)  -- α is inversely proportional to β
    (h2 : α (-6) = -3)         -- α = -3 when β = -6
    (h3 : β (-6) = -6)         -- β = -6 when β = -6 (implicit in the problem)
    : α 8 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2295_229507


namespace NUMINAMATH_CALUDE_range_of_g_l2295_229504

-- Define the functions
def f (x : ℝ) := x^2 - 7*x + 12
def g (x : ℝ) := x^2 - 7*x + 14

-- State the theorem
theorem range_of_g (x : ℝ) : 
  f x < 0 → ∃ y ∈ Set.Icc (1.75 : ℝ) 2, y = g x :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l2295_229504


namespace NUMINAMATH_CALUDE_min_sum_squares_l2295_229501

theorem min_sum_squares (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 2)
  let B : ℝ × ℝ := (3, b)
  let C : ℝ × ℝ := (2, 3)
  let O : ℝ × ℝ := (0, 0)
  let OB : ℝ × ℝ := (3 - 0, b - 0)
  let AC : ℝ × ℝ := (2 - a, 3 - 2)
  (OB.1 * AC.1 + OB.2 * AC.2 = 0) →
  (∃ (x : ℝ), ∀ (a b : ℝ), a^2 + b^2 ≥ x ∧ (∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = x)) ∧
  (∀ (x : ℝ), (∃ (a b : ℝ), a^2 + b^2 = x) → x ≥ 18/5) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2295_229501


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2295_229535

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (623 * n) % 32 = (1319 * n) % 32 ∧ ∀ (m : ℕ), m > 0 → m < n → (623 * m) % 32 ≠ (1319 * m) % 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2295_229535


namespace NUMINAMATH_CALUDE_characterize_satisfying_function_l2295_229585

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y u : ℝ, f (x + u) (y + u) = f x y + u) ∧
  (∀ x y v : ℝ, f (x * v) (y * v) = f x y * v)

/-- The theorem statement -/
theorem characterize_satisfying_function :
  ∀ f : ℝ → ℝ → ℝ, SatisfyingFunction f →
  ∃ p q : ℝ, (∀ x y : ℝ, f x y = p * x + q * y) ∧ p + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_characterize_satisfying_function_l2295_229585


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2295_229534

theorem rectangle_perimeter (area : ℝ) (side_ratio : ℝ) (perimeter : ℝ) : 
  area = 500 →
  side_ratio = 2 →
  let shorter_side := Real.sqrt (area / side_ratio)
  let longer_side := side_ratio * shorter_side
  perimeter = 2 * (shorter_side + longer_side) →
  perimeter = 30 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2295_229534


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l2295_229566

theorem sum_of_extreme_prime_factors_of_1365 : ∃ (min max : ℕ), 
  (min.Prime ∧ max.Prime ∧ 
   min ∣ 1365 ∧ max ∣ 1365 ∧
   (∀ p : ℕ, p.Prime → p ∣ 1365 → min ≤ p) ∧
   (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≤ max)) ∧
  min + max = 16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l2295_229566


namespace NUMINAMATH_CALUDE_class_average_mark_l2295_229591

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (remaining_students : ℕ)
  (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 25 →
  excluded_students = 5 →
  remaining_students = 20 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (total_students : ℚ) * ((excluded_students : ℚ) * excluded_avg + 
    (remaining_students : ℚ) * remaining_avg) / total_students = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2295_229591


namespace NUMINAMATH_CALUDE_square_side_length_l2295_229540

theorem square_side_length (d : ℝ) (h : d = 2) :
  ∃ s : ℝ, s * s = 2 ∧ s * Real.sqrt 2 = d :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l2295_229540


namespace NUMINAMATH_CALUDE_volume_of_T_l2295_229508

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p | let (x, y, z) := p
       (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The volume of T is 64/9 -/
theorem volume_of_T : volume T = 64/9 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l2295_229508


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l2295_229531

/-- The surface area of a cuboid given its length, breadth, and height. -/
def surfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

/-- Theorem: For a cuboid with surface area 720, length 12, and breadth 6, the height is 16. -/
theorem cuboid_height_calculation (SA l b h : ℝ) 
  (h_SA : SA = 720) 
  (h_l : l = 12) 
  (h_b : b = 6) 
  (h_surface_area : surfaceArea l b h = SA) : h = 16 := by
  sorry

#check cuboid_height_calculation

end NUMINAMATH_CALUDE_cuboid_height_calculation_l2295_229531


namespace NUMINAMATH_CALUDE_inequality_solution_l2295_229528

theorem inequality_solution (x : ℝ) : 
  (2*x - 1)/(x^2 + 2) > 5/x + 21/10 ↔ -5 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2295_229528


namespace NUMINAMATH_CALUDE_cube_root_plus_abs_plus_power_equals_six_linear_function_through_two_points_l2295_229582

-- Problem 1
theorem cube_root_plus_abs_plus_power_equals_six :
  (8 : ℝ) ^ (1/3) + |(-5)| + (-1)^2023 = 6 := by sorry

-- Problem 2
theorem linear_function_through_two_points :
  ∀ (k b : ℝ), (∀ x y : ℝ, y = k * x + b) →
  (1 = k * 0 + b) →
  (5 = k * 2 + b) →
  (∀ x : ℝ, k * x + b = 2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_cube_root_plus_abs_plus_power_equals_six_linear_function_through_two_points_l2295_229582


namespace NUMINAMATH_CALUDE_third_number_proof_l2295_229547

theorem third_number_proof (x : ℕ) : 
  let second := 3 * x - 7
  let third := 2 * x + 2
  x + second + third = 168 →
  third = 60 := by
sorry

end NUMINAMATH_CALUDE_third_number_proof_l2295_229547


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l2295_229593

/-- Represents the ticket sales for a theater performance --/
structure TicketSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ

/-- Calculates the difference between balcony and orchestra ticket sales --/
def ticket_difference (ts : TicketSales) : ℕ :=
  let orchestra_tickets := (ts.total_revenue - ts.balcony_price * ts.total_tickets) / 
    (ts.orchestra_price - ts.balcony_price)
  let balcony_tickets := ts.total_tickets - orchestra_tickets
  balcony_tickets - orchestra_tickets

/-- Theorem stating the difference in ticket sales for the given scenario --/
theorem theater_ticket_difference :
  ∃ (ts : TicketSales), 
    ts.orchestra_price = 12 ∧
    ts.balcony_price = 8 ∧
    ts.total_tickets = 370 ∧
    ts.total_revenue = 3320 ∧
    ticket_difference ts = 190 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l2295_229593


namespace NUMINAMATH_CALUDE_tax_discount_commute_ana_equals_bob_miltonville_market_problem_l2295_229548

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) (discount_rate_pos : 0 < discount_rate) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

/-- Calculates Ana's total (tax then discount) --/
def ana_total (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Bob's total (discount then tax) --/
def bob_total (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that Ana's total equals Bob's total --/
theorem ana_equals_bob (price : ℝ) (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) (discount_rate_pos : 0 < discount_rate) :
  ana_total price tax_rate discount_rate = bob_total price tax_rate discount_rate := by
  sorry

/-- Specific case for the problem --/
theorem miltonville_market_problem :
  ana_total 120 0.08 0.25 = bob_total 120 0.08 0.25 := by
  sorry

end NUMINAMATH_CALUDE_tax_discount_commute_ana_equals_bob_miltonville_market_problem_l2295_229548


namespace NUMINAMATH_CALUDE_min_value_of_P_l2295_229587

/-- The polynomial P as a function of a real number a -/
def P (a : ℝ) : ℝ := a^2 + 4*a + 2014

/-- Theorem stating that the minimum value of P is 2010 -/
theorem min_value_of_P :
  ∃ (min : ℝ), min = 2010 ∧ ∀ (a : ℝ), P a ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_P_l2295_229587


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2295_229555

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l2295_229555


namespace NUMINAMATH_CALUDE_near_square_quotient_l2295_229512

/-- A natural number is a near-square if it is the product of two consecutive natural numbers. -/
def is_near_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

/-- Theorem: Every near-square can be represented as the quotient of two near-squares. -/
theorem near_square_quotient (n : ℕ) :
  is_near_square (n * (n + 1)) →
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n * (n + 1) = a / b :=
by sorry

end NUMINAMATH_CALUDE_near_square_quotient_l2295_229512


namespace NUMINAMATH_CALUDE_f_negative_implies_a_range_l2295_229596

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * (x - 1)

theorem f_negative_implies_a_range (a : ℝ) :
  (∀ x > 1, f a x < 0) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_f_negative_implies_a_range_l2295_229596


namespace NUMINAMATH_CALUDE_transportation_theorem_l2295_229515

/-- Represents the capacity and cost of a truck type -/
structure TruckType where
  capacity : ℕ
  cost : ℕ

/-- Represents a transportation plan -/
structure TransportPlan where
  typeA : ℕ
  typeB : ℕ

/-- Solves the transportation problem -/
def solve_transportation_problem (typeA typeB : TruckType) (total_goods : ℕ) : 
  (TruckType × TruckType × TransportPlan) := sorry

theorem transportation_theorem 
  (typeA typeB : TruckType) (total_goods : ℕ) 
  (h1 : 3 * typeA.capacity + 2 * typeB.capacity = 90)
  (h2 : 5 * typeA.capacity + 4 * typeB.capacity = 160)
  (h3 : typeA.cost = 500)
  (h4 : typeB.cost = 400)
  (h5 : total_goods = 190) :
  let (solvedA, solvedB, optimal_plan) := solve_transportation_problem typeA typeB total_goods
  solvedA.capacity = 20 ∧ 
  solvedB.capacity = 15 ∧ 
  optimal_plan.typeA = 8 ∧ 
  optimal_plan.typeB = 2 := by sorry

end NUMINAMATH_CALUDE_transportation_theorem_l2295_229515


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2295_229509

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (1/4)) →  -- Each term is 1/4 of the previous term
  a 3 = 256 →                       -- The fourth term is 256
  a 5 = 4 →                         -- The sixth term is 4
  a 6 = 1 →                         -- The seventh term is 1
  a 3 + a 4 = 80 :=                 -- The sum of the fourth and fifth terms is 80
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2295_229509


namespace NUMINAMATH_CALUDE_simplify_fraction_l2295_229530

theorem simplify_fraction : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2295_229530


namespace NUMINAMATH_CALUDE_total_instruments_is_19_l2295_229521

-- Define the number of instruments for Charlie
def charlie_flutes : ℕ := 1
def charlie_horns : ℕ := 2
def charlie_harps : ℕ := 1
def charlie_drums : ℕ := 1

-- Define the number of instruments for Carli
def carli_flutes : ℕ := 2 * charlie_flutes
def carli_horns : ℕ := charlie_horns / 2
def carli_harps : ℕ := 0
def carli_drums : ℕ := 3

-- Define the number of instruments for Nick
def nick_flutes : ℕ := charlie_flutes + carli_flutes
def nick_horns : ℕ := charlie_horns - carli_horns
def nick_harps : ℕ := 0
def nick_drums : ℕ := 4

-- Define the total number of instruments
def total_instruments : ℕ := 
  charlie_flutes + charlie_horns + charlie_harps + charlie_drums +
  carli_flutes + carli_horns + carli_harps + carli_drums +
  nick_flutes + nick_horns + nick_harps + nick_drums

-- Theorem statement
theorem total_instruments_is_19 : total_instruments = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_instruments_is_19_l2295_229521


namespace NUMINAMATH_CALUDE_lunch_break_duration_l2295_229511

structure PaintingScenario where
  paula_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

def monday_work (s : PaintingScenario) : ℝ :=
  (9 - s.lunch_break) * (s.paula_rate + s.assistants_rate)

def tuesday_work (s : PaintingScenario) : ℝ :=
  (7 - s.lunch_break) * s.assistants_rate

def wednesday_work (s : PaintingScenario) : ℝ :=
  (10 - s.lunch_break) * s.paula_rate

theorem lunch_break_duration (s : PaintingScenario) :
  monday_work s = 0.6 →
  tuesday_work s = 0.3 →
  wednesday_work s = 0.1 →
  s.lunch_break = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l2295_229511


namespace NUMINAMATH_CALUDE_triangle_theorem_l2295_229563

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.b - 2 * t.a) * Real.cos t.C + t.c * Real.cos t.B = 0)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.b = 3 * t.a) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2295_229563


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2295_229573

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2295_229573


namespace NUMINAMATH_CALUDE_books_per_shelf_l2295_229503

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) (h1 : total_books = 12) (h2 : num_shelves = 3) :
  total_books / num_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2295_229503


namespace NUMINAMATH_CALUDE_coin_sum_theorem_l2295_229532

/-- Represents the possible coin values in cents -/
inductive Coin : Type
  | Penny : Coin
  | Nickel : Coin
  | Dime : Coin
  | Quarter : Coin
  | HalfDollar : Coin

/-- Returns the value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Checks if a given amount can be achieved using exactly six coins -/
def canAchieveWithSixCoins (amount : Nat) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : Coin), 
    coinValue c1 + coinValue c2 + coinValue c3 + coinValue c4 + coinValue c5 + coinValue c6 = amount

theorem coin_sum_theorem : 
  ¬ canAchieveWithSixCoins 62 ∧ 
  canAchieveWithSixCoins 80 ∧ 
  canAchieveWithSixCoins 90 ∧ 
  canAchieveWithSixCoins 96 := by
  sorry

end NUMINAMATH_CALUDE_coin_sum_theorem_l2295_229532


namespace NUMINAMATH_CALUDE_library_sunday_visitors_l2295_229588

/-- Calculates the average number of visitors on Sundays in a library -/
theorem library_sunday_visitors
  (total_days : Nat)
  (sunday_count : Nat)
  (non_sunday_visitors : Nat)
  (total_average : Nat)
  (h1 : total_days = 30)
  (h2 : sunday_count = 5)
  (h3 : non_sunday_visitors = 240)
  (h4 : total_average = 295) :
  (total_average * total_days - non_sunday_visitors * (total_days - sunday_count)) / sunday_count = 570 := by
  sorry

end NUMINAMATH_CALUDE_library_sunday_visitors_l2295_229588


namespace NUMINAMATH_CALUDE_lineup_combinations_l2295_229505

/-- The number of ways to choose a starting lineup for a basketball team -/
def choose_lineup (total_players : ℕ) (center_players : ℕ) (point_guard_players : ℕ) : ℕ :=
  center_players * point_guard_players * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating the number of ways to choose a starting lineup -/
theorem lineup_combinations :
  choose_lineup 12 3 2 = 4320 :=
by sorry

end NUMINAMATH_CALUDE_lineup_combinations_l2295_229505


namespace NUMINAMATH_CALUDE_subtracted_amount_l2295_229545

theorem subtracted_amount (number : ℝ) (result : ℝ) (amount : ℝ) : 
  number = 85 → 
  result = 23 → 
  0.4 * number - amount = result →
  amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2295_229545


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2295_229500

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, n > 0 → a n = a 1 * q^(n-1) ∧ S n = (a 1 * (1 - q^n)) / (1 - q)

/-- The theorem stating that for a geometric sequence with q = 2 and S_4 = 60, a_2 = 8 -/
theorem geometric_sequence_second_term 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : geometric_sequence a 2 S) 
  (h_sum : S 4 = 60) : 
  a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2295_229500


namespace NUMINAMATH_CALUDE_circumscribed_circle_condition_l2295_229584

/-- Two lines forming a quadrilateral with coordinate axes that has a circumscribed circle -/
def has_circumscribed_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((a + 2) * x + (1 - a) * y - 3 = 0) ∧
    ((a - 1) * x + (2 * a + 3) * y + 2 = 0) ∧
    (x ≥ 0 ∧ y ≥ 0)

/-- Theorem stating the condition for the quadrilateral to have a circumscribed circle -/
theorem circumscribed_circle_condition (a : ℝ) :
  has_circumscribed_circle a → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_condition_l2295_229584


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_max_consecutive_integers_sum_500_thirty_one_is_max_l2295_229574

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem max_consecutive_integers_sum_500 : 
  ∀ k > 31, k * (k + 1) / 2 > 500 := by sorry

theorem thirty_one_is_max : 
  31 * 32 / 2 ≤ 500 ∧ ∀ n > 31, n * (n + 1) / 2 > 500 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_max_consecutive_integers_sum_500_thirty_one_is_max_l2295_229574


namespace NUMINAMATH_CALUDE_perpendicular_to_horizontal_is_vertical_l2295_229524

/-- The angle of inclination of a line -/
def angle_of_inclination (l : Line2D) : ℝ := sorry

/-- A line is horizontal if its angle of inclination is 0 -/
def is_horizontal (l : Line2D) : Prop := angle_of_inclination l = 0

/-- Two lines are perpendicular if their angles of inclination sum to 90° -/
def are_perpendicular (l1 l2 : Line2D) : Prop :=
  angle_of_inclination l1 + angle_of_inclination l2 = 90

theorem perpendicular_to_horizontal_is_vertical (l1 l2 : Line2D) :
  is_horizontal l1 → are_perpendicular l1 l2 → angle_of_inclination l2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_to_horizontal_is_vertical_l2295_229524


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2295_229558

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_7 = 12, the sum of a_3 and a_11 is 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_a7 : a 7 = 12) : 
  a 3 + a 11 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2295_229558


namespace NUMINAMATH_CALUDE_randy_blocks_total_l2295_229586

theorem randy_blocks_total (house_blocks tower_blocks : ℕ) 
  (house_tower_diff : ℕ) (total_blocks : ℕ) : 
  house_blocks = 20 →
  tower_blocks = 50 →
  tower_blocks = house_blocks + house_tower_diff →
  house_tower_diff = 30 →
  total_blocks = house_blocks + tower_blocks →
  total_blocks = 70 := by
sorry

end NUMINAMATH_CALUDE_randy_blocks_total_l2295_229586


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l2295_229578

/-- Given a two-digit number, return its tens digit -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Given a two-digit number, return its ones digit -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- The product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ := (tens_digit n) * (ones_digit n)

/-- The sum of digits of a two-digit number -/
def digit_sum (n : ℕ) : ℕ := (tens_digit n) + (ones_digit n)

theorem two_digit_number_theorem (x : ℕ) 
  (h1 : is_two_digit x)
  (h2 : digit_product (x + 46) = 6)
  (h3 : digit_sum x = 14) :
  x = 77 ∨ x = 86 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l2295_229578


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l2295_229541

theorem max_value_of_sum_products (a b c : ℝ) (h : a + b + 3 * c = 6) :
  ∃ (max : ℝ), max = 516 / 49 ∧ ∀ (x y z : ℝ), x + y + 3 * z = 6 → x * y + x * z + y * z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l2295_229541


namespace NUMINAMATH_CALUDE_a_greater_than_b_l2295_229580

theorem a_greater_than_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eq1 : a^3 = a + 1) (eq2 : b^6 = b + 3*a) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l2295_229580


namespace NUMINAMATH_CALUDE_alligators_hiding_l2295_229518

/-- Given a zoo cage with alligators, prove the number of hiding alligators -/
theorem alligators_hiding (total_alligators : ℕ) (not_hiding : ℕ) 
  (h1 : total_alligators = 75)
  (h2 : not_hiding = 56) :
  total_alligators - not_hiding = 19 := by
  sorry

#check alligators_hiding

end NUMINAMATH_CALUDE_alligators_hiding_l2295_229518


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2295_229568

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ (∀ x > 1, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 2*x - 3 = 0) ↔ (∀ x > 1, x^2 - 2*x - 3 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2295_229568


namespace NUMINAMATH_CALUDE_inequalities_from_sum_of_reciprocal_squares_l2295_229543

theorem inequalities_from_sum_of_reciprocal_squares
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : 1 / a^2 + 1 / b^2 + 1 / c^2 = 1) :
  (1 / a + 1 / b + 1 / c ≤ Real.sqrt 3) ∧
  (a^2 / b^4 + b^2 / c^4 + c^2 / a^4 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sum_of_reciprocal_squares_l2295_229543


namespace NUMINAMATH_CALUDE_divisors_of_60_and_90_l2295_229533

theorem divisors_of_60_and_90 : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ 60 % n = 0 ∧ 90 % n = 0) ∧ 
  (∀ n : ℕ, n > 0 → 60 % n = 0 → 90 % n = 0 → n ∈ S) ∧
  Finset.card S = 8 := by
sorry

end NUMINAMATH_CALUDE_divisors_of_60_and_90_l2295_229533


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l2295_229551

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  (1 / 3) * x^2 = 2 ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : 
  8 * (x - 1)^3 = -(27 / 8) ↔ x = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l2295_229551


namespace NUMINAMATH_CALUDE_workshop_workers_count_l2295_229557

/-- Proves that the total number of workers in a workshop is 14, given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  (W : ℚ) * 8000 = 70000 + (N : ℚ) * 6000 →
  W = 7 + N →
  W = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l2295_229557


namespace NUMINAMATH_CALUDE_expression_value_l2295_229536

theorem expression_value (x y m n a : ℝ) 
  (h1 : x = -y) 
  (h2 : m * n = 1) 
  (h3 : |a| = 3) : 
  (a / (m * n)) + 2018 * (x + y) = 3 ∨ (a / (m * n)) + 2018 * (x + y) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2295_229536


namespace NUMINAMATH_CALUDE_hexagon_minus_rhombus_area_l2295_229554

-- Define the regular hexagon
def regular_hexagon (area : ℝ) : Prop :=
  area > 0 ∧ ∃ (side : ℝ), area = (3 * Real.sqrt 3 / 2) * side^2

-- Define the rhombus inside the hexagon
def rhombus_in_hexagon (hexagon_area : ℝ) (rhombus_area : ℝ) : Prop :=
  ∃ (side : ℝ), 
    rhombus_area = 2 * (Real.sqrt 3 / 4) * (4 / 3 * 30 * Real.sqrt 3)

-- The theorem to be proved
theorem hexagon_minus_rhombus_area 
  (hexagon_area : ℝ) (rhombus_area : ℝ) (remaining_area : ℝ) :
  regular_hexagon hexagon_area →
  rhombus_in_hexagon hexagon_area rhombus_area →
  hexagon_area = 135 →
  remaining_area = hexagon_area - rhombus_area →
  remaining_area = 75 := by
sorry

end NUMINAMATH_CALUDE_hexagon_minus_rhombus_area_l2295_229554


namespace NUMINAMATH_CALUDE_circle_radius_is_17_4_l2295_229569

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is tangent to the y-axis at a given point -/
def isTangentToYAxis (c : Circle) (p : ℝ × ℝ) : Prop :=
  c.center.1 = c.radius ∧ p.1 = 0 ∧ p.2 = c.center.2

/-- Predicate to check if a given x-coordinate is an x-intercept of the circle -/
def isXIntercept (c : Circle) (x : ℝ) : Prop :=
  ∃ y, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ∧ y = 0

theorem circle_radius_is_17_4 (c : Circle) :
  isTangentToYAxis c (0, 2) →
  isXIntercept c 8 →
  c.radius = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_17_4_l2295_229569


namespace NUMINAMATH_CALUDE_remainder_problem_l2295_229550

theorem remainder_problem (x : ℤ) : x % 9 = 2 → x % 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2295_229550


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2295_229567

theorem quadratic_inequality_equivalence (m : ℝ) : 
  (∀ x > 1, x^2 + (m - 2) * x + 3 - m ≥ 0) ↔ m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2295_229567


namespace NUMINAMATH_CALUDE_paintbrush_cost_l2295_229546

theorem paintbrush_cost (paint_cost easel_cost albert_has albert_needs : ℚ) :
  paint_cost = 4.35 →
  easel_cost = 12.65 →
  albert_has = 6.50 →
  albert_needs = 12 →
  paint_cost + easel_cost + (albert_has + albert_needs - (paint_cost + easel_cost)) = 1.50 := by
sorry

end NUMINAMATH_CALUDE_paintbrush_cost_l2295_229546


namespace NUMINAMATH_CALUDE_line_slope_45_degrees_l2295_229517

/-- Given a line passing through points P(-2, m) and Q(m, 4) with a slope angle of 45°, the value of m is 1. -/
theorem line_slope_45_degrees (m : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    P = (-2, m) ∧ 
    Q = (m, 4) ∧ 
    (Q.2 - P.2) / (Q.1 - P.1) = Real.tan (π / 4)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_45_degrees_l2295_229517


namespace NUMINAMATH_CALUDE_question_1_question_2_l2295_229513

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Define the range of x for question 1
def range_x : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Define the range of a for question 2
def range_a : Set ℝ := {a | a > 3}

-- Theorem for question 1
theorem question_1 (a : ℝ) (h : a = 2) :
  {x : ℝ | p x a ∧ q x} = range_x := by sorry

-- Theorem for question 2
theorem question_2 :
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) →
  a ∈ range_a := by sorry

end NUMINAMATH_CALUDE_question_1_question_2_l2295_229513


namespace NUMINAMATH_CALUDE_lcm_14_21_35_l2295_229549

theorem lcm_14_21_35 : Nat.lcm 14 (Nat.lcm 21 35) = 210 := by sorry

end NUMINAMATH_CALUDE_lcm_14_21_35_l2295_229549


namespace NUMINAMATH_CALUDE_two_tangent_lines_exist_l2295_229537

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  (l.a * c.center.x + l.b * c.center.y + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

/-- Check if a line passes through a point -/
def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem two_tangent_lines_exist (p : Point) (c : Circle) : 
  p.x = 2 ∧ p.y = 3 ∧ c.center.x = 0 ∧ c.center.y = 0 ∧ c.radius = 2 →
  ∃! (l1 l2 : Line), l1 ≠ l2 ∧ 
    isTangent l1 c ∧ isTangent l2 c ∧ 
    passesThrough l1 p ∧ passesThrough l2 p ∧
    ∀ (l : Line), isTangent l c ∧ passesThrough l p → l = l1 ∨ l = l2 :=
by sorry

end NUMINAMATH_CALUDE_two_tangent_lines_exist_l2295_229537


namespace NUMINAMATH_CALUDE_mixed_sample_more_suitable_l2295_229519

-- Define the probability of having the disease
def disease_probability : ℝ := 0.1

-- Define the number of animals in each group
def group_size : ℕ := 2

-- Define the total number of animals
def total_animals : ℕ := 2 * group_size

-- Define the expected number of tests for individual testing
def expected_tests_individual : ℝ := total_animals

-- Define the probability of a negative mixed sample
def prob_negative_mixed : ℝ := (1 - disease_probability) ^ total_animals

-- Define the expected number of tests for mixed sample testing
def expected_tests_mixed : ℝ :=
  1 * prob_negative_mixed + (1 + total_animals) * (1 - prob_negative_mixed)

-- Theorem statement
theorem mixed_sample_more_suitable :
  expected_tests_mixed < expected_tests_individual :=
sorry

end NUMINAMATH_CALUDE_mixed_sample_more_suitable_l2295_229519


namespace NUMINAMATH_CALUDE_ratio_RN_NS_l2295_229560

/-- Square ABCD with side length 10, F is on DC 3 units from D, N is midpoint of AF,
    perpendicular bisector of AF intersects AD at R and BC at S -/
structure SquareConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ
  N : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_square : A = (0, 10) ∧ B = (10, 10) ∧ C = (10, 0) ∧ D = (0, 0)
  h_F : F = (3, 0)
  h_N : N = (3/2, 5)
  h_R : R.1 = 57/3 ∧ R.2 = 10
  h_S : S.1 = -43/3 ∧ S.2 = 0

/-- The ratio of RN to NS is 1:1 -/
theorem ratio_RN_NS (cfg : SquareConfiguration) : 
  dist cfg.R cfg.N = dist cfg.N cfg.S :=
by sorry


end NUMINAMATH_CALUDE_ratio_RN_NS_l2295_229560


namespace NUMINAMATH_CALUDE_triangle_inequality_l2295_229526

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  htr : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define an equilateral triangle
def isEquilateral (t : Triangle) : Prop := t.a = t.b ∧ t.b = t.c

-- Theorem statement
theorem triangle_inequality (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * Real.sqrt 3 * area t ∧
  (t.a^2 + t.b^2 + t.c^2 = 4 * Real.sqrt 3 * area t ↔ isEquilateral t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2295_229526


namespace NUMINAMATH_CALUDE_rice_price_reduction_l2295_229565

theorem rice_price_reduction (P : ℝ) (h : P > 0) :
  49 * P = 50 * (P * (1 - 2/100)) :=
by sorry

end NUMINAMATH_CALUDE_rice_price_reduction_l2295_229565


namespace NUMINAMATH_CALUDE_triangle_area_is_integer_l2295_229552

theorem triangle_area_is_integer (a b c : ℝ) (h1 : a^2 = 377) (h2 : b^2 = 153) (h3 : c^2 = 80)
  (h4 : ∃ (w h : ℤ), ∃ (x y z : ℝ),
    (x^2 + y^2 = w^2) ∧ (x^2 + z^2 = h^2) ∧
    (y + z = a ∨ y + z = b ∨ y + z = c) ∧
    (∃ (d1 d2 : ℤ), d1 ≥ 0 ∧ d2 ≥ 0 ∧ d1 + d2 + x = w ∧ d1 + d2 + y = h)) :
  ∃ (A : ℤ), A = 42 ∧ 16 * A^2 = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_integer_l2295_229552


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2295_229527

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2295_229527


namespace NUMINAMATH_CALUDE_eighth_finger_number_l2295_229502

-- Define the function f
def f (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 4
  | 1 => 3
  | 2 => 6
  | 3 => 5
  | _ => 0  -- This case should never occur

-- Define a function to apply f n times
def apply_f_n_times (n : ℕ) : ℕ :=
  match n with
  | 0 => 4  -- Start with 4
  | n + 1 => f (apply_f_n_times n)

-- Theorem statement
theorem eighth_finger_number : apply_f_n_times 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eighth_finger_number_l2295_229502


namespace NUMINAMATH_CALUDE_infinite_non_prime_polynomials_l2295_229520

theorem infinite_non_prime_polynomials :
  ∃ f : ℕ → ℕ, ∀ k n : ℕ, ¬ Prime (n^4 + f k * n) := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_prime_polynomials_l2295_229520


namespace NUMINAMATH_CALUDE_output_for_15_l2295_229577

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 25 then step1 + 10 else step1 - 7

theorem output_for_15 : function_machine 15 = 38 := by sorry

end NUMINAMATH_CALUDE_output_for_15_l2295_229577


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2295_229597

theorem simplify_trig_expression (x : ℝ) (h : 5 * Real.pi / 2 < x ∧ x < 3 * Real.pi) :
  Real.sqrt ((1 - Real.sin (3 * Real.pi / 2 - x)) / 2) = -Real.cos (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2295_229597


namespace NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l2295_229556

/-- The probability of selecting 3 non-defective pencils from a box of 7 pencils with 2 defective pencils -/
theorem prob_three_non_defective_pencils :
  let total_pencils : ℕ := 7
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let ways_to_select_all := Nat.choose total_pencils selected_pencils
  let ways_to_select_non_defective := Nat.choose non_defective_pencils selected_pencils
  (ways_to_select_non_defective : ℚ) / ways_to_select_all = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_prob_three_non_defective_pencils_l2295_229556


namespace NUMINAMATH_CALUDE_triangle_transformation_indefinite_l2295_229522

/-- A triangle can undergo the given transformation indefinitely iff it's equilateral -/
theorem triangle_transformation_indefinite (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  (∀ n : ℕ, ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    a' = (-a + b + c) / 2 ∧ 
    b' = (a - b + c) / 2 ∧ 
    c' = (a + b - c) / 2) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_transformation_indefinite_l2295_229522


namespace NUMINAMATH_CALUDE_flagpole_height_l2295_229572

/-- Given a lamppost height and shadow length, calculate the height of another object with a known shadow length -/
theorem flagpole_height
  (lamppost_height : ℝ) 
  (lamppost_shadow : ℝ) 
  (flagpole_shadow : ℝ) 
  (h1 : lamppost_height = 50)
  (h2 : lamppost_shadow = 12)
  (h3 : flagpole_shadow = 18 / 12)  -- Convert 18 inches to feet
  : ∃ (flagpole_height : ℝ), 
    flagpole_height * lamppost_shadow = lamppost_height * flagpole_shadow ∧ 
    flagpole_height * 12 = 75 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l2295_229572


namespace NUMINAMATH_CALUDE_unique_determination_by_gcds_l2295_229525

theorem unique_determination_by_gcds :
  ∀ X : ℕ, X ≤ 100 →
  ∃ (M N : Fin 7 → ℕ), (∀ i, M i < 100 ∧ N i < 100) ∧
    ∀ Y : ℕ, Y ≤ 100 →
      (∀ i : Fin 7, Nat.gcd (X + M i) (N i) = Nat.gcd (Y + M i) (N i)) →
      X = Y :=
by sorry

end NUMINAMATH_CALUDE_unique_determination_by_gcds_l2295_229525


namespace NUMINAMATH_CALUDE_range_of_a_l2295_229598

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(q x) → ¬(p x a)) ∧ 
  (∃ x, ¬(q x) ∧ p x a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  condition a → (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2295_229598


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2295_229594

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a circle -/
def Point.liesOn (p : Point) (c : Circle) : Prop :=
  c.equation p.x p.y

/-- The circle we want to prove about -/
def ourCircle : Circle :=
  { center := { x := 2, y := -1 }
  , equation := fun x y => (x - 2)^2 + (y + 1)^2 = 2 }

/-- The theorem to prove -/
theorem circle_equation_correct :
  ourCircle.center = { x := 2, y := -1 } ∧
  Point.liesOn { x := 3, y := 0 } ourCircle :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2295_229594


namespace NUMINAMATH_CALUDE_S_inter_T_eq_T_l2295_229592

/-- The set of odd integers -/
def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}

/-- The set of integers of the form 4n + 1 -/
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

/-- Theorem stating that the intersection of S and T is equal to T -/
theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_inter_T_eq_T_l2295_229592


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2295_229542

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a > b ∧ b > 0 → 1/a < 1/b) ∧
  (∃ a b, 1/a < 1/b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2295_229542


namespace NUMINAMATH_CALUDE_lemonade_glasses_per_gallon_l2295_229576

theorem lemonade_glasses_per_gallon 
  (total_gallons : ℕ) 
  (cost_per_gallon : ℚ) 
  (price_per_glass : ℚ) 
  (glasses_drunk : ℕ) 
  (glasses_unsold : ℕ) 
  (net_profit : ℚ) :
  total_gallons = 2 ∧ 
  cost_per_gallon = 7/2 ∧ 
  price_per_glass = 1 ∧ 
  glasses_drunk = 5 ∧ 
  glasses_unsold = 6 ∧ 
  net_profit = 14 →
  ∃ (glasses_per_gallon : ℕ),
    glasses_per_gallon = 16 ∧
    (total_gallons * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass
    = total_gallons * cost_per_gallon + net_profit :=
by sorry

end NUMINAMATH_CALUDE_lemonade_glasses_per_gallon_l2295_229576


namespace NUMINAMATH_CALUDE_phil_coin_collection_l2295_229553

def coin_collection (initial : ℕ) (year1 : ℕ → ℕ) (year2 : ℕ) (year3 : ℕ) (year4 : ℕ) 
                    (year5 : ℕ → ℕ) (year6 : ℕ) (year7 : ℕ) (year8 : ℕ) (year9 : ℕ → ℕ) : ℕ :=
  let after_year1 := year1 initial
  let after_year2 := after_year1 + year2
  let after_year3 := after_year2 + year3
  let after_year4 := after_year3 + year4
  let after_year5 := year5 after_year4
  let after_year6 := after_year5 + year6
  let after_year7 := after_year6 + year7
  let after_year8 := after_year7 + year8
  year9 after_year8

theorem phil_coin_collection :
  coin_collection 1000 (λ x => x * 4) (7 * 52) (3 * 182) (2 * 52) 
                  (λ x => x - (x * 2 / 5)) (5 * 91) (20 * 12) (10 * 52)
                  (λ x => x - (x / 3)) = 2816 := by
  sorry

end NUMINAMATH_CALUDE_phil_coin_collection_l2295_229553


namespace NUMINAMATH_CALUDE_handshakes_count_l2295_229599

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group1_size : ℕ  -- People who know each other
  group2_size : ℕ  -- People who don't know anyone
  h_total : total_people = group1_size + group2_size

/-- Calculates the number of handshakes in a social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  (event.group2_size * (event.total_people - 1)) / 2

/-- Theorem stating the number of handshakes in the specific social event -/
theorem handshakes_count :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group1_size = 25 ∧
    event.group2_size = 15 ∧
    count_handshakes event = 292 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l2295_229599


namespace NUMINAMATH_CALUDE_gamma_delta_sum_l2295_229559

theorem gamma_delta_sum : 
  ∃ (γ δ : ℝ), ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90*x + 1980) / (x^2 + 60*x - 3240) → 
  γ + δ = 140 := by
  sorry

end NUMINAMATH_CALUDE_gamma_delta_sum_l2295_229559


namespace NUMINAMATH_CALUDE_coin_distribution_l2295_229510

theorem coin_distribution (n k : ℕ) (h1 : 2 * k^2 - 2 * k < n) (h2 : n < 2 * k^2 + 2 * k) :
  (2 * k^2 - 2 * k < n ∧ n < 2 * k^2 → 
    (k - 1)^2 + (n - (2 * k^2 - 2 * k)) > k^2 - k) ∧
  (2 * k^2 < n ∧ n < 2 * k^2 + 2 * k → 
    k^2 < k^2 - k + (n - (2 * k^2 - k))) :=
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l2295_229510


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2295_229539

/-- Given income and savings, calculate the ratio of income to expenditure --/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (expenditure : ℕ) 
  (h1 : income = 16000) 
  (h2 : savings = 3200) 
  (h3 : savings = income - expenditure) : 
  (income : ℚ) / expenditure = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2295_229539


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l2295_229579

theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : son_age = 26 → age_difference = 28 → 
  ∃ y : ℕ, (son_age + y + age_difference) = 2 * (son_age + y) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l2295_229579


namespace NUMINAMATH_CALUDE_table_runners_area_l2295_229529

theorem table_runners_area (table_area : ℝ) (covered_percentage : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) :
  table_area = 175 →
  covered_percentage = 0.8 →
  two_layer_area = 24 →
  three_layer_area = 24 →
  ∃ (total_area : ℝ), total_area = 188 ∧ 
    total_area = (covered_percentage * table_area - 2 * three_layer_area - two_layer_area) + 
                 2 * two_layer_area + 3 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_table_runners_area_l2295_229529


namespace NUMINAMATH_CALUDE_integer_and_mod_three_remainder_l2295_229583

theorem integer_and_mod_three_remainder (n : ℕ+) :
  ∃ k : ℤ, (n.val : ℝ)^3 + (3/2) * (n.val : ℝ)^2 + (1/2) * (n.val : ℝ) - 1 = (k : ℝ) ∧ k ≡ 2 [ZMOD 3] :=
sorry

end NUMINAMATH_CALUDE_integer_and_mod_three_remainder_l2295_229583


namespace NUMINAMATH_CALUDE_opposite_values_l2295_229571

theorem opposite_values (a b c m : ℚ) 
  (eq1 : a + 2*b + 3*c = m) 
  (eq2 : a + b + 2*c = m) : 
  b = -c := by
sorry

end NUMINAMATH_CALUDE_opposite_values_l2295_229571


namespace NUMINAMATH_CALUDE_helga_wrote_250_articles_l2295_229561

/-- Represents Helga's work schedule and article production --/
structure HelgaWork where
  articles_per_30min : ℕ := 5
  usual_hours_per_day : ℕ := 4
  usual_days_per_week : ℕ := 5
  extra_hours_thursday : ℕ := 2
  extra_hours_friday : ℕ := 3

/-- Calculates the total number of articles Helga wrote in a week --/
def total_articles_in_week (h : HelgaWork) : ℕ :=
  let articles_per_hour := h.articles_per_30min * 2
  let usual_articles_per_day := articles_per_hour * h.usual_hours_per_day
  let usual_articles_per_week := usual_articles_per_day * h.usual_days_per_week
  let extra_articles_thursday := articles_per_hour * h.extra_hours_thursday
  let extra_articles_friday := articles_per_hour * h.extra_hours_friday
  usual_articles_per_week + extra_articles_thursday + extra_articles_friday

/-- Theorem stating that Helga wrote 250 articles in the given week --/
theorem helga_wrote_250_articles : 
  ∀ (h : HelgaWork), total_articles_in_week h = 250 := by
  sorry

end NUMINAMATH_CALUDE_helga_wrote_250_articles_l2295_229561


namespace NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l2295_229562

theorem sum_reciprocal_lower_bound (a₁ a₂ a₃ : ℝ) 
  (h_pos₁ : a₁ > 0) (h_pos₂ : a₂ > 0) (h_pos₃ : a₃ > 0)
  (h_sum : a₁ + a₂ + a₃ = 1) : 
  1/a₁ + 1/a₂ + 1/a₃ ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l2295_229562


namespace NUMINAMATH_CALUDE_stadium_entrance_count_l2295_229595

/-- The number of placards initially in the basket -/
def initial_placards : ℕ := 5682

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The number of people who entered the stadium -/
def people_entered : ℕ := initial_placards / placards_per_person

theorem stadium_entrance_count :
  people_entered = 2841 :=
sorry

end NUMINAMATH_CALUDE_stadium_entrance_count_l2295_229595


namespace NUMINAMATH_CALUDE_officer_selection_l2295_229544

theorem officer_selection (n m k l : ℕ) (hn : n = 20) (hm : m = 8) (hk : k = 10) (hl : l = 3) :
  Nat.choose n m - (Nat.choose k m + Nat.choose k 1 * Nat.choose (n - k) (m - 1) + Nat.choose k 2 * Nat.choose (n - k) (m - 2)) = 115275 :=
by sorry

end NUMINAMATH_CALUDE_officer_selection_l2295_229544


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2295_229570

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2295_229570


namespace NUMINAMATH_CALUDE_total_cuts_after_six_operations_l2295_229523

def cuts_in_operation (n : ℕ) : ℕ :=
  3 * 4^(n - 1)

def total_cuts (n : ℕ) : ℕ :=
  (List.range n).map (cuts_in_operation ∘ (· + 1)) |> List.sum

theorem total_cuts_after_six_operations :
  total_cuts 6 = 4095 := by
  sorry

end NUMINAMATH_CALUDE_total_cuts_after_six_operations_l2295_229523


namespace NUMINAMATH_CALUDE_oranges_per_box_l2295_229506

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 56) (h2 : num_boxes = 8) :
  total_oranges / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2295_229506


namespace NUMINAMATH_CALUDE_brittany_brooke_money_ratio_l2295_229590

/-- Given the following conditions about money possession:
  - Alison has half as much money as Brittany
  - Brooke has twice as much money as Kent
  - Kent has $1,000
  - Alison has $4,000
Prove that Brittany has 4 times as much money as Brooke -/
theorem brittany_brooke_money_ratio :
  ∀ (alison brittany brooke kent : ℝ),
  alison = brittany / 2 →
  brooke = 2 * kent →
  kent = 1000 →
  alison = 4000 →
  brittany = 4 * brooke :=
by sorry

end NUMINAMATH_CALUDE_brittany_brooke_money_ratio_l2295_229590


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2295_229564

theorem adult_ticket_cost (student_price : ℕ) (num_students : ℕ) (num_adults : ℕ) (total_amount : ℕ) : 
  student_price = 6 →
  num_students = 20 →
  num_adults = 12 →
  total_amount = 216 →
  ∃ (adult_price : ℕ), 
    student_price * num_students + adult_price * num_adults = total_amount ∧ 
    adult_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2295_229564


namespace NUMINAMATH_CALUDE_simplify_expression_l2295_229516

theorem simplify_expression (a : ℝ) : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2295_229516
