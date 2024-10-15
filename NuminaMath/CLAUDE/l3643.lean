import Mathlib

namespace NUMINAMATH_CALUDE_kramer_packing_theorem_l3643_364319

/-- Kramer's packing rate in cases per hour -/
def packing_rate : ℝ := 120

/-- Number of boxes Kramer packs per minute -/
def boxes_per_minute : ℕ := 10

/-- Number of boxes in one case -/
def boxes_per_case : ℕ := 5

/-- Number of cases Kramer packs in 2 hours -/
def cases_in_two_hours : ℕ := 240

/-- The number of cases Kramer can pack in x hours -/
def cases_packed (x : ℝ) : ℝ := packing_rate * x

theorem kramer_packing_theorem (x : ℝ) : 
  cases_packed x = packing_rate * x ∧
  (boxes_per_minute : ℝ) * 60 / boxes_per_case = packing_rate ∧
  cases_in_two_hours = packing_rate * 2 :=
sorry

end NUMINAMATH_CALUDE_kramer_packing_theorem_l3643_364319


namespace NUMINAMATH_CALUDE_apricot_tea_calories_l3643_364339

/-- Represents the composition of the apricot tea -/
structure ApricotTea where
  apricot_juice : ℝ
  honey : ℝ
  water : ℝ

/-- Calculates the total calories in the apricot tea mixture -/
def total_calories (tea : ApricotTea) : ℝ :=
  tea.apricot_juice * 0.3 + tea.honey * 3.04

/-- Calculates the total weight of the apricot tea mixture -/
def total_weight (tea : ApricotTea) : ℝ :=
  tea.apricot_juice + tea.honey + tea.water

/-- Theorem: 250g of Nathan's apricot tea contains 98.5 calories -/
theorem apricot_tea_calories :
  let tea : ApricotTea := { apricot_juice := 150, honey := 50, water := 300 }
  let caloric_density : ℝ := total_calories tea / total_weight tea
  250 * caloric_density = 98.5 := by
  sorry

#check apricot_tea_calories

end NUMINAMATH_CALUDE_apricot_tea_calories_l3643_364339


namespace NUMINAMATH_CALUDE_orange_cells_theorem_l3643_364386

/-- Represents the possible outcomes of orange cells on the board -/
inductive OrangeCellsOutcome
  | lower  : OrangeCellsOutcome  -- represents 2021 * 2020
  | higher : OrangeCellsOutcome  -- represents 2022 * 2020

/-- The size of one side of the square board -/
def boardSize : Nat := 2022

/-- The size of one side of the paintable square -/
def squareSize : Nat := 2

/-- Represents the game rules and outcomes -/
structure GameBoard where
  size : Nat
  squareSize : Nat
  possibleOutcomes : List OrangeCellsOutcome

/-- The main theorem to prove -/
theorem orange_cells_theorem (board : GameBoard) 
  (h1 : board.size = boardSize) 
  (h2 : board.squareSize = squareSize) 
  (h3 : board.possibleOutcomes = [OrangeCellsOutcome.lower, OrangeCellsOutcome.higher]) : 
  ∃ (n : Nat), (n = 2021 * 2020 ∨ n = 2022 * 2020) ∧ 
  (∀ (m : Nat), m = 2021 * 2020 ∨ m = 2022 * 2020 → 
    ∃ (outcome : OrangeCellsOutcome), outcome ∈ board.possibleOutcomes ∧
    (outcome = OrangeCellsOutcome.lower → m = 2021 * 2020) ∧
    (outcome = OrangeCellsOutcome.higher → m = 2022 * 2020)) :=
  sorry


end NUMINAMATH_CALUDE_orange_cells_theorem_l3643_364386


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3643_364371

/-- A quadratic function of the form y = x^2 - bx + 2c with axis of symmetry x = 3 has b = 6 -/
theorem quadratic_symmetry_axis (b c : ℝ) : 
  (∀ x y : ℝ, y = x^2 - b*x + 2*c → (∀ y1 y2 : ℝ, (3 - x)^2 = (3 + x)^2 → y1 = y2)) → 
  b = 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3643_364371


namespace NUMINAMATH_CALUDE_line_through_points_l3643_364395

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a line passing through distinct vectors a and b, 
    if k*a + (5/6)*b lies on the same line, then k = 5/6 -/
theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ (t : ℝ), k • a + (5/6) • b = a + t • (b - a) → k = 5/6 :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l3643_364395


namespace NUMINAMATH_CALUDE_horner_method_example_l3643_364352

def f (x : ℝ) : ℝ := 3 * x^3 + x - 3

theorem horner_method_example : f 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l3643_364352


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3643_364367

-- Define the number of flowers
def num_flowers : ℕ := 4

-- Define the number of gems
def num_gems : ℕ := 6

-- Define the number of invalid combinations
def num_invalid : ℕ := 3

-- Theorem statement
theorem wizard_elixir_combinations :
  (num_flowers * num_gems) - num_invalid = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3643_364367


namespace NUMINAMATH_CALUDE_two_primes_not_congruent_to_one_l3643_364313

theorem two_primes_not_congruent_to_one (p : Nat) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ (q r : Nat), q ≠ r ∧ q.Prime ∧ r.Prime ∧ 2 ≤ q ∧ q ≤ p - 2 ∧ 2 ≤ r ∧ r ≤ p - 2 ∧
  ¬(q^(p-1) ≡ 1 [MOD p^2]) ∧ ¬(r^(p-1) ≡ 1 [MOD p^2]) := by
  sorry

end NUMINAMATH_CALUDE_two_primes_not_congruent_to_one_l3643_364313


namespace NUMINAMATH_CALUDE_simplify_expression_l3643_364353

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2*x + 1)) = 1 / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3643_364353


namespace NUMINAMATH_CALUDE_bond_face_value_l3643_364301

/-- Proves that the face value of a bond is 5000 given specific conditions --/
theorem bond_face_value (F : ℝ) : 
  (0.10 * F = 0.065 * 7692.307692307692) → F = 5000 := by
  sorry

end NUMINAMATH_CALUDE_bond_face_value_l3643_364301


namespace NUMINAMATH_CALUDE_max_floor_length_l3643_364336

/-- Represents a rectangular tile with length and width in centimeters. -/
structure Tile where
  length : ℕ
  width : ℕ

/-- Represents a rectangular floor with length and width in centimeters. -/
structure Floor where
  length : ℕ
  width : ℕ

/-- Checks if a given number of tiles can fit on the floor without overlap or overshooting. -/
def canFitTiles (t : Tile) (f : Floor) (n : ℕ) : Prop :=
  (f.length % t.length = 0 ∧ f.width ≥ t.width ∧ (f.length / t.length) * (f.width / t.width) ≥ n) ∨
  (f.length % t.width = 0 ∧ f.width ≥ t.length ∧ (f.length / t.width) * (f.width / t.length) ≥ n)

theorem max_floor_length (t : Tile) (maxTiles : ℕ) :
  t.length = 50 →
  t.width = 40 →
  maxTiles = 9 →
  ∃ (f : Floor), canFitTiles t f maxTiles ∧
    ∀ (f' : Floor), canFitTiles t f' maxTiles → f'.length ≤ f.length ∧ f.length = 450 := by
  sorry

end NUMINAMATH_CALUDE_max_floor_length_l3643_364336


namespace NUMINAMATH_CALUDE_range_of_t_squared_minus_one_l3643_364318

theorem range_of_t_squared_minus_one :
  ∀ z : ℝ, ∃ x y : ℝ, x ≠ 0 ∧ (y / x)^2 - 1 = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_squared_minus_one_l3643_364318


namespace NUMINAMATH_CALUDE_speaking_brother_is_tryalya_l3643_364346

-- Define the brothers
inductive Brother
| Tralyalya
| Tryalya

-- Define the card suits
inductive Suit
| Black
| Red

-- Define the statement about having a black suit card
def claims_black_suit (b : Brother) : Prop :=
  match b with
  | Brother.Tralyalya => true
  | Brother.Tryalya => true

-- Define the rule that the brother with the black suit card cannot tell the truth
axiom black_suit_rule : ∀ (b : Brother), 
  (∃ (s : Suit), s = Suit.Black ∧ claims_black_suit b) → ¬(claims_black_suit b)

-- Theorem: The speaking brother must be Tryalya and he must have the black suit card
theorem speaking_brother_is_tryalya : 
  ∃ (b : Brother) (s : Suit), 
    b = Brother.Tryalya ∧ 
    s = Suit.Black ∧ 
    claims_black_suit b :=
sorry

end NUMINAMATH_CALUDE_speaking_brother_is_tryalya_l3643_364346


namespace NUMINAMATH_CALUDE_solve_m_n_l3643_364324

def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m*x + n = 0}

theorem solve_m_n :
  ∃ (m n : ℝ),
    (A ∪ B m n = A) ∧
    (A ∩ B m n = {5}) ∧
    m = -10 ∧
    n = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_m_n_l3643_364324


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3643_364351

theorem fraction_multiplication :
  (3 : ℚ) / 4 * 4 / 5 * 5 / 6 * 6 / 7 = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3643_364351


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3643_364359

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3643_364359


namespace NUMINAMATH_CALUDE_quadrilateral_exists_l3643_364329

/-- A quadrilateral with side lengths and a diagonal -/
structure Quadrilateral :=
  (AB BC CD DA AC : ℝ)

/-- The triangle inequality theorem -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

/-- Theorem: There exists a quadrilateral ABCD with diagonal AC, where AB = 10, BC = 9, CD = 19, DA = 5, and AC = 15 -/
theorem quadrilateral_exists : ∃ (q : Quadrilateral), 
  q.AB = 10 ∧ 
  q.BC = 9 ∧ 
  q.CD = 19 ∧ 
  q.DA = 5 ∧ 
  q.AC = 15 ∧
  triangle_inequality q.AB q.BC q.AC ∧
  triangle_inequality q.AC q.CD q.DA :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_exists_l3643_364329


namespace NUMINAMATH_CALUDE_average_equation_solution_l3643_364321

theorem average_equation_solution (x : ℝ) : 
  ((x + 8) + (5 * x + 4) + (2 * x + 7)) / 3 = 3 * x - 10 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3643_364321


namespace NUMINAMATH_CALUDE_three_real_roots_l3643_364341

/-- The polynomial f(x) = x^3 - 6x^2 + 9x - 2 -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 2

/-- Theorem: The equation f(x) = 0 has exactly three real roots -/
theorem three_real_roots : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f x = 0 := by sorry

end NUMINAMATH_CALUDE_three_real_roots_l3643_364341


namespace NUMINAMATH_CALUDE_calculation_proof_l3643_364302

theorem calculation_proof : 211 * 555 + 445 * 789 + 555 * 789 + 211 * 445 = 10^6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3643_364302


namespace NUMINAMATH_CALUDE_odd_m_triple_g_35_l3643_364388

def g (n : Int) : Int :=
  if n % 2 = 1 then n + 5 else n / 2

theorem odd_m_triple_g_35 (m : Int) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_35_l3643_364388


namespace NUMINAMATH_CALUDE_product_base5_digit_sum_l3643_364374

/-- Converts a base-5 number represented as a list of digits to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5, returning a list of digits --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec toDigits (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else toDigits (m / 5) ((m % 5) :: acc)
    toDigits n []

/-- Sums the digits of a number represented as a list --/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

theorem product_base5_digit_sum (n1 n2 : List Nat) :
  sumDigits (base10ToBase5 (base5ToBase10 n1 * base5ToBase10 n2)) = 8 :=
sorry

end NUMINAMATH_CALUDE_product_base5_digit_sum_l3643_364374


namespace NUMINAMATH_CALUDE_value_added_to_fraction_l3643_364340

theorem value_added_to_fraction (x y : ℝ) : 
  x = 8 → 0.75 * x + y = 8 → y = 2 := by sorry

end NUMINAMATH_CALUDE_value_added_to_fraction_l3643_364340


namespace NUMINAMATH_CALUDE_survey_preferences_l3643_364396

theorem survey_preferences (total : ℕ) (mac_pref : ℕ) (windows_pref : ℕ) : 
  total = 210 →
  mac_pref = 60 →
  windows_pref = 40 →
  ∃ (no_pref : ℕ),
    no_pref = total - (mac_pref + windows_pref + (mac_pref / 3)) ∧
    no_pref = 90 :=
by sorry

end NUMINAMATH_CALUDE_survey_preferences_l3643_364396


namespace NUMINAMATH_CALUDE_green_blue_difference_after_borders_l3643_364312

/-- Represents the number of tiles in a hexagonal figure -/
structure HexagonalFigure where
  blue : ℕ
  green : ℕ

/-- Calculates the number of green tiles added by one border -/
def greenTilesPerBorder : ℕ := 6 * 3

/-- Theorem: The difference between green and blue tiles after adding two borders -/
theorem green_blue_difference_after_borders (initial : HexagonalFigure) :
  let newFigure := HexagonalFigure.mk
    initial.blue
    (initial.green + 2 * greenTilesPerBorder)
  newFigure.green - newFigure.blue = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_borders_l3643_364312


namespace NUMINAMATH_CALUDE_root_product_theorem_l3643_364326

theorem root_product_theorem (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) →
  (d^2 - n*d + 3 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3643_364326


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3643_364307

def f (x : ℝ) : ℝ := 3 - 2*x

theorem solution_set_of_inequality (x : ℝ) :
  (|f (x + 1) + 2| ≤ 3) ↔ (0 ≤ x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3643_364307


namespace NUMINAMATH_CALUDE_triangle_problem_l3643_364372

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > t.c ∧
  t.a * t.c * (1/3) = 2 ∧  -- Vector BA · Vector BC = 2 and cos B = 1/3
  t.b = 3

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.a = 3 ∧ t.c = 2 ∧ Real.cos (t.B - t.C) = 23/27 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3643_364372


namespace NUMINAMATH_CALUDE_marble_distribution_l3643_364373

theorem marble_distribution (a : ℕ) : 
  let angela := a
  let brian := 2 * a
  let caden := angela + brian
  let daryl := 2 * caden
  angela + brian + caden + daryl = 144 → a = 12 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l3643_364373


namespace NUMINAMATH_CALUDE_circumscribed_quadrilateral_sides_l3643_364397

/-- A circumscribed quadrilateral with perimeter 24 and three consecutive sides in ratio 1:2:3 has sides 3, 6, 9, and 6. -/
theorem circumscribed_quadrilateral_sides (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- sides are positive
  a + b + c + d = 24 →  -- perimeter is 24
  a + c = b + d →  -- circumscribed property
  ∃ (x : ℝ), a = x ∧ b = 2*x ∧ c = 3*x →  -- consecutive sides in ratio 1:2:3
  a = 3 ∧ b = 6 ∧ c = 9 ∧ d = 6 := by
  sorry

#check circumscribed_quadrilateral_sides

end NUMINAMATH_CALUDE_circumscribed_quadrilateral_sides_l3643_364397


namespace NUMINAMATH_CALUDE_walters_hourly_wage_l3643_364355

/-- Walter's work schedule and earnings allocation --/
structure WorkSchedule where
  days_per_week : ℕ
  hours_per_day : ℕ
  school_allocation_ratio : ℚ
  school_allocation_amount : ℚ

/-- Calculate Walter's hourly wage --/
def hourly_wage (w : WorkSchedule) : ℚ :=
  w.school_allocation_amount / w.school_allocation_ratio / (w.days_per_week * w.hours_per_day)

/-- Theorem: Walter's hourly wage is $5 --/
theorem walters_hourly_wage (w : WorkSchedule)
  (h1 : w.days_per_week = 5)
  (h2 : w.hours_per_day = 4)
  (h3 : w.school_allocation_ratio = 3/4)
  (h4 : w.school_allocation_amount = 75) :
  hourly_wage w = 5 := by
  sorry

end NUMINAMATH_CALUDE_walters_hourly_wage_l3643_364355


namespace NUMINAMATH_CALUDE_snack_cost_l3643_364384

/-- Given the following conditions:
    - There are 4 people
    - Each ticket costs $18
    - The total cost for tickets and snacks for all 4 people is $92
    Prove that the cost of a set of snacks is $5. -/
theorem snack_cost (num_people : ℕ) (ticket_price : ℕ) (total_cost : ℕ) :
  num_people = 4 →
  ticket_price = 18 →
  total_cost = 92 →
  (total_cost - num_people * ticket_price) / num_people = 5 := by
  sorry

end NUMINAMATH_CALUDE_snack_cost_l3643_364384


namespace NUMINAMATH_CALUDE_price_restoration_l3643_364348

theorem price_restoration (original_price : ℝ) (original_price_positive : original_price > 0) :
  let reduced_price := original_price * (1 - 0.15)
  let restoration_factor := (1 + 0.1765)
  (reduced_price * restoration_factor - original_price) / original_price < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_price_restoration_l3643_364348


namespace NUMINAMATH_CALUDE_a_representation_l3643_364361

theorem a_representation (a : ℤ) (x y : ℤ) (h : 3 * a = x^2 + 2 * y^2) :
  ∃ (u v : ℤ), a = u^2 + 2 * v^2 := by
sorry

end NUMINAMATH_CALUDE_a_representation_l3643_364361


namespace NUMINAMATH_CALUDE_simplify_fraction_l3643_364390

theorem simplify_fraction (x y : ℝ) (hxy : x ≠ y) (hxy_neg : x ≠ -y) (hx : x ≠ 0) :
  (1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3643_364390


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_l3643_364377

theorem binomial_coefficient_n_plus_one_choose_n (n : ℕ+) : 
  Nat.choose (n + 1) n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_l3643_364377


namespace NUMINAMATH_CALUDE_work_completion_proof_l3643_364391

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 20

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 5

/-- The number of days y needs to finish the work alone -/
def y_days : ℕ := 16

theorem work_completion_proof :
  (1 : ℚ) / x_days * x_remaining + (1 : ℚ) / y_days * y_worked = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3643_364391


namespace NUMINAMATH_CALUDE_simplified_quadratic_radical_example_l3643_364362

def is_simplified_quadratic_radical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ ¬∃ m : ℕ, m > 1 ∧ ∃ k : ℕ, n = m^2 * k

theorem simplified_quadratic_radical_example :
  is_simplified_quadratic_radical (Real.sqrt 6) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 12) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 20) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 32) :=
by sorry

end NUMINAMATH_CALUDE_simplified_quadratic_radical_example_l3643_364362


namespace NUMINAMATH_CALUDE_dice_faces_theorem_l3643_364350

theorem dice_faces_theorem (n m : ℕ) : 
  (n ≥ 1) → 
  (m ≥ 1) → 
  (∀ i ∈ Finset.range n, ∀ j ∈ Finset.range m, 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 8) (Finset.product (Finset.range n) (Finset.range m))).card = 
    (1/2 : ℚ) * (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 11) (Finset.product (Finset.range n) (Finset.range m))).card) →
  ((Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 13) (Finset.product (Finset.range n) (Finset.range m))).card : ℚ) / (n * m) = 1/15 →
  (∃ k : ℕ, n + m = 5 * k) →
  (∀ n' m' : ℕ, n' + m' < n + m → 
    ¬((n' ≥ 1) ∧ 
      (m' ≥ 1) ∧ 
      (∀ i ∈ Finset.range n', ∀ j ∈ Finset.range m', 
        (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 8) (Finset.product (Finset.range n') (Finset.range m'))).card = 
        (1/2 : ℚ) * (Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 11) (Finset.product (Finset.range n') (Finset.range m'))).card) ∧
      ((Finset.filter (λ (p : ℕ × ℕ) => p.1 + p.2 = 13) (Finset.product (Finset.range n') (Finset.range m'))).card : ℚ) / (n' * m') = 1/15 ∧
      (∃ k : ℕ, n' + m' = 5 * k))) →
  n + m = 25 := by
sorry

end NUMINAMATH_CALUDE_dice_faces_theorem_l3643_364350


namespace NUMINAMATH_CALUDE_bakery_stop_difference_l3643_364375

/-- Represents the distances between locations in Kona's trip -/
structure TripDistances where
  apartment_to_bakery : ℕ
  bakery_to_grandma : ℕ
  grandma_to_apartment : ℕ

/-- Calculates the additional miles driven with a bakery stop -/
def additional_miles (d : TripDistances) : ℕ :=
  (d.apartment_to_bakery + d.bakery_to_grandma + d.grandma_to_apartment) -
  (2 * d.grandma_to_apartment)

/-- Theorem stating that the additional miles driven with a bakery stop is 6 -/
theorem bakery_stop_difference (d : TripDistances)
  (h1 : d.apartment_to_bakery = 9)
  (h2 : d.bakery_to_grandma = 24)
  (h3 : d.grandma_to_apartment = 27) :
  additional_miles d = 6 := by
  sorry


end NUMINAMATH_CALUDE_bakery_stop_difference_l3643_364375


namespace NUMINAMATH_CALUDE_ratio_equality_l3643_364316

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) :
  (a / 3) / (b / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3643_364316


namespace NUMINAMATH_CALUDE_largest_constant_inequality_largest_constant_is_three_equality_condition_l3643_364381

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
sorry

theorem largest_constant_is_three :
  ∀ C > 3, ∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℝ,
    (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 < C * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 = 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ↔
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_largest_constant_is_three_equality_condition_l3643_364381


namespace NUMINAMATH_CALUDE_polygon_division_theorem_l3643_364344

/-- A polygon that can be divided into a specific number of rectangles -/
structure DivisiblePolygon where
  vertices : ℕ
  can_divide : ℕ → Prop
  h_100 : can_divide 100
  h_not_99 : ¬ can_divide 99

/-- The main theorem stating that a polygon divisible into 100 rectangles but not 99
    has more than 200 vertices and cannot be divided into 100 triangles -/
theorem polygon_division_theorem (P : DivisiblePolygon) :
  P.vertices > 200 ∧ ¬ ∃ (triangles : ℕ), triangles = 100 ∧ P.can_divide triangles := by
  sorry


end NUMINAMATH_CALUDE_polygon_division_theorem_l3643_364344


namespace NUMINAMATH_CALUDE_base_10_to_6_conversion_l3643_364365

/-- Converts a base-10 number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Converts a list of digits in base-6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 6 * acc + d) 0

theorem base_10_to_6_conversion :
  fromBase6 (toBase6 110) = 110 :=
sorry

end NUMINAMATH_CALUDE_base_10_to_6_conversion_l3643_364365


namespace NUMINAMATH_CALUDE_mp_eq_nq_l3643_364314

/-- Two circles in a plane -/
structure TwoCircles where
  c1 : Set (ℝ × ℝ)
  c2 : Set (ℝ × ℝ)

/-- Points on the circles -/
structure CirclePoints (tc : TwoCircles) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_A_intersect : A ∈ tc.c1 ∩ tc.c2
  h_B_intersect : B ∈ tc.c1 ∩ tc.c2
  h_M_on_c1 : M ∈ tc.c1
  h_N_on_c2 : N ∈ tc.c2
  h_P_on_c1 : P ∈ tc.c1
  h_Q_on_c2 : Q ∈ tc.c2

/-- AM is tangent to c2 at A -/
def is_tangent_AM (tc : TwoCircles) (cp : CirclePoints tc) : Prop := sorry

/-- AN is tangent to c1 at A -/
def is_tangent_AN (tc : TwoCircles) (cp : CirclePoints tc) : Prop := sorry

/-- B, M, and P are collinear -/
def collinear_BMP (cp : CirclePoints tc) : Prop := sorry

/-- B, N, and Q are collinear -/
def collinear_BNQ (cp : CirclePoints tc) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem mp_eq_nq (tc : TwoCircles) (cp : CirclePoints tc)
    (h_AM_tangent : is_tangent_AM tc cp)
    (h_AN_tangent : is_tangent_AN tc cp)
    (h_BMP_collinear : collinear_BMP cp)
    (h_BNQ_collinear : collinear_BNQ cp) :
    distance cp.M cp.P = distance cp.N cp.Q := by sorry

end NUMINAMATH_CALUDE_mp_eq_nq_l3643_364314


namespace NUMINAMATH_CALUDE_min_sections_problem_l3643_364311

theorem min_sections_problem (num_boys num_girls max_per_section : ℕ) 
  (h_boys : num_boys = 408)
  (h_girls : num_girls = 240)
  (h_max : max_per_section = 24)
  (h_ratio : ∃ (x : ℕ), x > 0 ∧ num_boys ≤ 3 * x * max_per_section ∧ num_girls ≤ 2 * x * max_per_section) :
  ∃ (total_sections : ℕ), 
    total_sections = 30 ∧
    ∃ (boys_sections girls_sections : ℕ),
      boys_sections + girls_sections = total_sections ∧
      3 * girls_sections = 2 * boys_sections ∧
      num_boys ≤ boys_sections * max_per_section ∧
      num_girls ≤ girls_sections * max_per_section ∧
      ∀ (other_total : ℕ),
        (∃ (other_boys other_girls : ℕ),
          other_boys + other_girls = other_total ∧
          3 * other_girls = 2 * other_boys ∧
          num_boys ≤ other_boys * max_per_section ∧
          num_girls ≤ other_girls * max_per_section) →
        other_total ≥ total_sections :=
by sorry

end NUMINAMATH_CALUDE_min_sections_problem_l3643_364311


namespace NUMINAMATH_CALUDE_fraction_sum_from_hcf_lcm_and_sum_l3643_364310

theorem fraction_sum_from_hcf_lcm_and_sum (m n : ℕ+) 
  (hcf : Nat.gcd m n = 6)
  (lcm : Nat.lcm m n = 210)
  (sum : m + n = 80) :
  (1 : ℚ) / m + (1 : ℚ) / n = 2 / 31.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_from_hcf_lcm_and_sum_l3643_364310


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l3643_364331

theorem solution_set_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x < 0 ↔ 0 < x ∧ x < 1) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l3643_364331


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l3643_364378

def m : ℕ := 2010^2 + 2^2010

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ) : (m^2 + 3^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l3643_364378


namespace NUMINAMATH_CALUDE_five_vents_per_zone_l3643_364382

/-- Represents an HVAC system -/
structure HVACSystem where
  totalCost : ℕ
  numZones : ℕ
  costPerVent : ℕ

/-- Calculates the number of vents in each zone of an HVAC system -/
def ventsPerZone (system : HVACSystem) : ℕ :=
  (system.totalCost / system.costPerVent) / system.numZones

/-- Theorem: For the given HVAC system, there are 5 vents in each zone -/
theorem five_vents_per_zone (system : HVACSystem)
    (h1 : system.totalCost = 20000)
    (h2 : system.numZones = 2)
    (h3 : system.costPerVent = 2000) :
    ventsPerZone system = 5 := by
  sorry

#eval ventsPerZone { totalCost := 20000, numZones := 2, costPerVent := 2000 }

end NUMINAMATH_CALUDE_five_vents_per_zone_l3643_364382


namespace NUMINAMATH_CALUDE_product_of_roots_l3643_364343

theorem product_of_roots (x : ℝ) : 
  (∃ α β : ℝ, α * β = -10 ∧ -20 = -2 * x^2 - 6 * x ↔ (x = α ∨ x = β)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l3643_364343


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l3643_364368

theorem complex_arithmetic_expression : 
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l3643_364368


namespace NUMINAMATH_CALUDE_linear_function_theorem_l3643_364370

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ := sorry

theorem linear_function_theorem :
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) → -- f is linear
  (∀ x : ℝ, f x = 3 * f_inv x + 9) → -- f(x) = 3f^(-1)(x) + 9
  f 0 = 3 → -- f(0) = 3
  f_inv 3 = 0 → -- f^(-1)(3) = 0
  f 3 = 6 * Real.sqrt 3 := by -- f(3) = 6√3
sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l3643_364370


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3643_364323

theorem largest_n_satisfying_conditions : 
  ∃ (m : ℤ), (313 : ℤ)^2 = (m + 1)^3 - m^3 ∧ 
  ∃ (k : ℤ), (2 * 313 + 103 : ℤ) = k^2 ∧
  ∀ (n : ℤ), n > 313 → 
    (∃ (m : ℤ), n^2 = (m + 1)^3 - m^3 ∧ 
    ∃ (k : ℤ), (2 * n + 103 : ℤ) = k^2) → False :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3643_364323


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l3643_364383

/-- Family of lines parameterized by t -/
def C (t : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x * Real.cos t + (y + 1) * Real.sin t = 2

/-- Predicate for three lines from C forming an equilateral triangle -/
def forms_equilateral_triangle (t₁ t₂ t₃ : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ x₃ y₃,
    C t₁ x₁ y₁ ∧ C t₂ x₂ y₂ ∧ C t₃ x₃ y₃ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
    (x₂ - x₃)^2 + (y₂ - y₃)^2 = (x₃ - x₁)^2 + (y₃ - y₁)^2

/-- The area of the equilateral triangle formed by three lines from C -/
def triangle_area (t₁ t₂ t₃ : ℝ) : ℝ := sorry

theorem equilateral_triangle_area :
  ∀ t₁ t₂ t₃, forms_equilateral_triangle t₁ t₂ t₃ →
  triangle_area t₁ t₂ t₃ = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l3643_364383


namespace NUMINAMATH_CALUDE_square_difference_65_35_l3643_364357

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l3643_364357


namespace NUMINAMATH_CALUDE_no_prime_polynomial_l3643_364345

-- Define a polynomial with integer coefficients
def IntPolynomial := ℕ → ℤ

-- Define what it means for a polynomial to be constant
def IsConstant (P : IntPolynomial) : Prop :=
  ∀ n m : ℕ, P n = P m

-- Define primality
def IsPrime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

-- The main theorem
theorem no_prime_polynomial :
  ¬∃ (P : IntPolynomial),
    (¬IsConstant P) ∧
    (∀ n : ℕ, n > 0 → IsPrime (P n)) :=
sorry

end NUMINAMATH_CALUDE_no_prime_polynomial_l3643_364345


namespace NUMINAMATH_CALUDE_outfit_count_l3643_364315

/-- The number of different outfits that can be made with shirts, pants, and hats of different colors. -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
                (pants : ℕ) 
                (red_hats green_hats blue_hats : ℕ) : ℕ :=
  (red_shirts * pants * (green_hats + blue_hats)) +
  (green_shirts * pants * (red_hats + blue_hats)) +
  (blue_shirts * pants * (red_hats + green_hats))

/-- Theorem stating the number of outfits under given conditions. -/
theorem outfit_count : 
  num_outfits 4 4 4 10 6 6 4 = 1280 :=
by sorry

end NUMINAMATH_CALUDE_outfit_count_l3643_364315


namespace NUMINAMATH_CALUDE_area_of_triangle_AFK_l3643_364387

/-- Parabola with equation y² = 8x, focus F(2, 0), and directrix intersecting x-axis at K(-2, 0) -/
structure Parabola where
  F : ℝ × ℝ := (2, 0)
  K : ℝ × ℝ := (-2, 0)

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  A : ℝ × ℝ
  on_parabola : A.2^2 = 8 * A.1
  distance_condition : (A.1 + 2)^2 + A.2^2 = 2 * ((A.1 - 2)^2 + A.2^2)

/-- The area of triangle AFK is 8 -/
theorem area_of_triangle_AFK (p : Parabola) (point : PointOnParabola p) :
  (1 / 2 : ℝ) * 4 * |point.A.2| = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AFK_l3643_364387


namespace NUMINAMATH_CALUDE_modular_inverse_37_mod_39_l3643_364333

theorem modular_inverse_37_mod_39 : 
  ∃ x : ℕ, x < 39 ∧ (37 * x) % 39 = 1 ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_37_mod_39_l3643_364333


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l3643_364308

open Real

-- Define the triangle
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem acute_triangle_properties
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (h_relation : c - b = 2 * b * cos A) :
  A = 2 * B ∧
  π/6 < B ∧ B < π/4 ∧
  sqrt 2 < a/b ∧ a/b < sqrt 3 ∧
  5 * sqrt 3 / 3 < 1 / tan B - 1 / tan A + 2 * sin A ∧
  1 / tan B - 1 / tan A + 2 * sin A < 3 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l3643_364308


namespace NUMINAMATH_CALUDE_train_length_l3643_364320

/-- The length of a train given specific passing times -/
theorem train_length : ∃ (L : ℝ), 
  (L / 24 = (L + 650) / 89) ∧ L = 240 := by sorry

end NUMINAMATH_CALUDE_train_length_l3643_364320


namespace NUMINAMATH_CALUDE_gustran_nails_cost_l3643_364399

structure Salon where
  name : String
  haircut : ℕ
  facial : ℕ
  nails : ℕ

def gustran_salon : Salon := {
  name := "Gustran Salon"
  haircut := 45
  facial := 22
  nails := 0  -- Unknown, to be proved
}

def barbaras_shop : Salon := {
  name := "Barbara's Shop"
  haircut := 30
  facial := 28
  nails := 40
}

def fancy_salon : Salon := {
  name := "The Fancy Salon"
  haircut := 34
  facial := 30
  nails := 20
}

def total_cost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

theorem gustran_nails_cost :
  ∃ (x : ℕ), 
    gustran_salon.nails = x ∧ 
    total_cost gustran_salon = 84 ∧
    total_cost barbaras_shop ≥ 84 ∧
    total_cost fancy_salon = 84 ∧
    x = 17 := by sorry

end NUMINAMATH_CALUDE_gustran_nails_cost_l3643_364399


namespace NUMINAMATH_CALUDE_coefficient_not_fifty_l3643_364328

theorem coefficient_not_fifty :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 →
  (Nat.choose 5 k) * (2^(5-k)) ≠ 50 := by
sorry

end NUMINAMATH_CALUDE_coefficient_not_fifty_l3643_364328


namespace NUMINAMATH_CALUDE_paint_cans_for_house_l3643_364392

/-- Represents the number of paint cans needed for a house painting job -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_rooms := num_bedrooms + num_other_rooms
  let total_paint_needed := total_rooms * paint_per_room
  let color_cans := num_bedrooms * paint_per_room
  let white_paint_needed := num_other_rooms * paint_per_room
  let white_cans := (white_paint_needed + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10 -/
theorem paint_cans_for_house : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_for_house_l3643_364392


namespace NUMINAMATH_CALUDE_distance_between_points_l3643_364363

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (4, -6)
  let p2 : ℝ × ℝ := (-8, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 265 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3643_364363


namespace NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3643_364389

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_exactly_five_prime_factors (n : ℕ) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧ is_prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅ ∧
    ∀ (q : ℕ), is_prime q → q ∣ n → (q = p₁ ∨ q = p₂ ∨ q = p₃ ∨ q = p₄ ∨ q = p₅)

theorem smallest_odd_with_five_prime_factors :
  (∀ n : ℕ, n < 15015 → ¬(n % 2 = 1 ∧ has_exactly_five_prime_factors n)) ∧
  15015 % 2 = 1 ∧ has_exactly_five_prime_factors 15015 := by
  sorry

end NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3643_364389


namespace NUMINAMATH_CALUDE_atBatsAgainstLeft_is_180_l3643_364354

/-- Represents the batting statistics of a baseball player -/
structure BattingStats where
  totalAtBats : ℕ
  totalHits : ℕ
  avgAgainstLeft : ℚ
  avgAgainstRight : ℚ

/-- Calculates the number of at-bats against left-handed pitchers -/
def atBatsAgainstLeft (stats : BattingStats) : ℕ :=
  sorry

/-- Theorem stating that the number of at-bats against left-handed pitchers is 180 -/
theorem atBatsAgainstLeft_is_180 (stats : BattingStats) 
  (h1 : stats.totalAtBats = 600)
  (h2 : stats.totalHits = 192)
  (h3 : stats.avgAgainstLeft = 1/4)
  (h4 : stats.avgAgainstRight = 7/20)
  (h5 : (stats.totalHits : ℚ) / stats.totalAtBats = 8/25) :
  atBatsAgainstLeft stats = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_atBatsAgainstLeft_is_180_l3643_364354


namespace NUMINAMATH_CALUDE_binary_111011_equals_59_l3643_364303

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111011_equals_59 :
  binary_to_decimal [true, true, false, true, true, true] = 59 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011_equals_59_l3643_364303


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3643_364342

theorem greatest_whole_number_satisfying_inequality :
  ∀ n : ℤ, (∀ x : ℤ, x ≤ n → 4 * x - 3 < 2 - x) → n ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l3643_364342


namespace NUMINAMATH_CALUDE_number_equation_solution_l3643_364300

theorem number_equation_solution :
  ∀ B : ℝ, (4 * B + 4 = 33) → B = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3643_364300


namespace NUMINAMATH_CALUDE_gabrielle_cardinals_count_l3643_364366

/-- Represents the number of birds of each type seen by a person -/
structure BirdCount where
  robins : ℕ
  cardinals : ℕ
  bluejays : ℕ

/-- Calculates the total number of birds seen -/
def total_birds (count : BirdCount) : ℕ :=
  count.robins + count.cardinals + count.bluejays

/-- The bird counts for Chase and Gabrielle -/
def chase : BirdCount := { robins := 2, cardinals := 5, bluejays := 3 }
def gabrielle : BirdCount := { robins := 5, cardinals := 0, bluejays := 3 }

theorem gabrielle_cardinals_count : gabrielle.cardinals = 4 := by
  have h1 : total_birds chase = 10 := by sorry
  have h2 : total_birds gabrielle = (120 * total_birds chase) / 100 := by sorry
  have h3 : gabrielle.robins + gabrielle.bluejays = 8 := by sorry
  sorry

end NUMINAMATH_CALUDE_gabrielle_cardinals_count_l3643_364366


namespace NUMINAMATH_CALUDE_fraction_numerator_l3643_364335

theorem fraction_numerator (y : ℝ) (x : ℤ) (h1 : y > 0) :
  (x : ℝ) / y + 3 * y / 10 = (35 : ℝ) / 100 * y → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l3643_364335


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l3643_364337

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x - f y)^2 - (4 * x * y) * f y

/-- Theorem stating that the only function satisfying the condition is the zero function -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l3643_364337


namespace NUMINAMATH_CALUDE_percent_of_x_is_y_l3643_364394

theorem percent_of_x_is_y (x y : ℝ) (h : 0.7 * (x - y) = 0.3 * (x + y)) : y = 0.4 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_y_l3643_364394


namespace NUMINAMATH_CALUDE_largest_g_is_correct_l3643_364322

/-- The largest positive integer g for which there exists exactly one pair of positive integers (a, b) satisfying 5a + gb = 70 -/
def largest_g : ℕ := 65

/-- The unique pair of positive integers (a, b) satisfying 5a + (largest_g)b = 70 -/
def unique_pair : ℕ × ℕ := (1, 1)

theorem largest_g_is_correct :
  (∀ g : ℕ, g > largest_g →
    ¬(∃! p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ 5 * p.1 + g * p.2 = 70)) ∧
  (∃! p : ℕ × ℕ, p.1 > 0 ∧ p.2 > 0 ∧ 5 * p.1 + largest_g * p.2 = 70) ∧
  (unique_pair.1 > 0 ∧ unique_pair.2 > 0 ∧ 5 * unique_pair.1 + largest_g * unique_pair.2 = 70) :=
by sorry

#check largest_g_is_correct

end NUMINAMATH_CALUDE_largest_g_is_correct_l3643_364322


namespace NUMINAMATH_CALUDE_perpendicular_bisector_complex_l3643_364356

/-- The set of points equidistant from two distinct complex numbers forms a perpendicular bisector -/
theorem perpendicular_bisector_complex (z₁ z₂ : ℂ) (hz : z₁ ≠ z₂) :
  {z : ℂ | Complex.abs (z - z₁) = Complex.abs (z - z₂)} =
  {z : ℂ | (z - (z₁ + z₂) / 2) • (z₁ - z₂) = 0} :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_complex_l3643_364356


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3643_364364

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- The value that is exactly n standard deviations less than the mean -/
def valueNStdDevBelow (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.stdDev

/-- Theorem: For a normal distribution with mean 17.5 and standard deviation 2.5,
    the value that is exactly 2 standard deviations less than the mean is 12.5 -/
theorem two_std_dev_below_mean :
  let d : NormalDistribution := { mean := 17.5, stdDev := 2.5 }
  valueNStdDevBelow d 2 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3643_364364


namespace NUMINAMATH_CALUDE_average_of_first_45_results_l3643_364325

theorem average_of_first_45_results
  (n₁ : ℕ)
  (n₂ : ℕ)
  (a₂ : ℝ)
  (total_avg : ℝ)
  (h₁ : n₁ = 45)
  (h₂ : n₂ = 25)
  (h₃ : a₂ = 45)
  (h₄ : total_avg = 32.142857142857146)
  (h₅ : (n₁ : ℝ) * a₁ + (n₂ : ℝ) * a₂ = (n₁ + n₂ : ℝ) * total_avg) :
  a₁ = 25 :=
by sorry

end NUMINAMATH_CALUDE_average_of_first_45_results_l3643_364325


namespace NUMINAMATH_CALUDE_group_size_l3643_364306

theorem group_size (total : ℕ) (over_30 : ℕ) (under_20 : ℕ) 
  (h1 : over_30 = 90)
  (h2 : total = over_30 + under_20)
  (h3 : (under_20 : ℚ) / total = 1 / 10) : 
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_group_size_l3643_364306


namespace NUMINAMATH_CALUDE_notebook_difference_l3643_364309

theorem notebook_difference (tara_spent lea_spent : ℚ) 
  (h1 : tara_spent = 5.20)
  (h2 : lea_spent = 7.80)
  (h3 : ∃ (price : ℚ), price > 1 ∧ 
    ∃ (tara_count lea_count : ℕ), 
      tara_count * price = tara_spent ∧ 
      lea_count * price = lea_spent) :
  ∃ (price : ℚ) (tara_count lea_count : ℕ), 
    price > 1 ∧
    tara_count * price = tara_spent ∧
    lea_count * price = lea_spent ∧
    lea_count = tara_count + 2 :=
by sorry

end NUMINAMATH_CALUDE_notebook_difference_l3643_364309


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3643_364327

theorem simplify_complex_fraction (m : ℝ) (h1 : m ≠ 2) (h2 : m ≠ -3) :
  (m - (4*m - 9) / (m - 2)) / ((m^2 - 9) / (m - 2)) = (m - 3) / (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3643_364327


namespace NUMINAMATH_CALUDE_bart_earnings_l3643_364376

/-- The amount of money Bart earns per question answered -/
def money_per_question : ℚ := 0.2

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday -/
def surveys_monday : ℕ := 3

/-- The number of surveys Bart completed on Tuesday -/
def surveys_tuesday : ℕ := 4

/-- Theorem stating the total money Bart earned over two days -/
theorem bart_earnings : 
  (surveys_monday + surveys_tuesday) * questions_per_survey * money_per_question = 14 := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l3643_364376


namespace NUMINAMATH_CALUDE_no_solutions_power_equation_l3643_364385

theorem no_solutions_power_equation (x n r : ℕ) (hx : x > 1) :
  x^(2*n + 1) ≠ 2^r + 1 ∧ x^(2*n + 1) ≠ 2^r - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_power_equation_l3643_364385


namespace NUMINAMATH_CALUDE_subtracted_number_l3643_364338

theorem subtracted_number (x : ℝ) : 3889 + 12.808 - x = 3854.002 → x = 47.806 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3643_364338


namespace NUMINAMATH_CALUDE_no_square_prime_ratio_in_triangular_sequence_l3643_364330

theorem no_square_prime_ratio_in_triangular_sequence (p : ℕ) (hp : Prime p) :
  ∀ (x y l : ℕ), l ≥ 1 →
    (x * (x + 1)) / (y * (y + 1)) ≠ p^(2 * l) := by
  sorry

end NUMINAMATH_CALUDE_no_square_prime_ratio_in_triangular_sequence_l3643_364330


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_certain_event_l3643_364358

/-- Definition of a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- The triangle inequality theorem -/
theorem triangle_inequality (t : Triangle) : 
  (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b) :=
sorry

/-- Proof that the triangle inequality is a certain event -/
theorem triangle_inequality_certain_event : 
  ∀ (t : Triangle), (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_certain_event_l3643_364358


namespace NUMINAMATH_CALUDE_trapezium_area_l3643_364380

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) :
  (1/2 : ℝ) * (a + b) * h = 27 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_l3643_364380


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l3643_364398

theorem purely_imaginary_complex_equation (a : ℝ) (z : ℂ) :
  z + 3 * Complex.I = a + a * Complex.I →
  z.re = 0 →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l3643_364398


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3643_364334

theorem simplify_polynomial (x : ℝ) : 
  x * (4 * x^3 - 3) - 6 * (x^2 - 3*x + 9) = 4 * x^4 - 6 * x^2 + 15 * x - 54 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3643_364334


namespace NUMINAMATH_CALUDE_shark_sightings_l3643_364347

theorem shark_sightings (cape_may daytona_beach : ℕ) : 
  cape_may + daytona_beach = 40 →
  cape_may = 2 * daytona_beach - 8 →
  cape_may = 24 :=
by sorry

end NUMINAMATH_CALUDE_shark_sightings_l3643_364347


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l3643_364379

theorem solution_set_reciprocal_gt_one :
  {x : ℝ | (1 : ℝ) / x > 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l3643_364379


namespace NUMINAMATH_CALUDE_probability_of_six_or_less_l3643_364369

def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls
def num_drawn : ℕ := 4
def red_points : ℕ := 1
def black_points : ℕ := 3

def score (red_drawn : ℕ) : ℕ :=
  red_drawn * red_points + (num_drawn - red_drawn) * black_points

def probability_of_score (s : ℕ) : ℚ :=
  (Nat.choose num_red_balls s * Nat.choose num_black_balls (num_drawn - s)) /
  Nat.choose total_balls num_drawn

theorem probability_of_six_or_less :
  probability_of_score 4 + probability_of_score 3 = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_six_or_less_l3643_364369


namespace NUMINAMATH_CALUDE_expression_evaluation_l3643_364332

theorem expression_evaluation : 
  let f (x : ℝ) := (x - 1) / (x + 1)
  let expr (x : ℝ) := (f x + 1) / (f x - 1)
  expr 2 = -2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3643_364332


namespace NUMINAMATH_CALUDE_exponent_division_l3643_364305

theorem exponent_division (a : ℝ) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3643_364305


namespace NUMINAMATH_CALUDE_tv_cost_l3643_364393

theorem tv_cost (total_budget : ℕ) (computer_cost : ℕ) (fridge_extra_cost : ℕ) :
  total_budget = 1600 →
  computer_cost = 250 →
  fridge_extra_cost = 500 →
  ∃ tv_cost : ℕ, tv_cost = 600 ∧ 
    tv_cost + (computer_cost + fridge_extra_cost) + computer_cost = total_budget :=
by sorry

end NUMINAMATH_CALUDE_tv_cost_l3643_364393


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l3643_364304

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → Complex.abs (4 + n * Complex.I) = 4 * Real.sqrt 13 → n = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l3643_364304


namespace NUMINAMATH_CALUDE_usual_time_is_36_l3643_364360

-- Define the usual time T as a positive real number
variable (T : ℝ) (hT : T > 0)

-- Define the relationship between normal speed and reduced speed
def reduced_speed_time : ℝ := T + 12

-- Theorem stating that the usual time T is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_is_36_l3643_364360


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3643_364317

-- Define the sets A and B
def A : Set ℝ := { x | 2 < x ∧ x ≤ 4 }
def B : Set ℝ := { x | x^2 - 2*x < 3 }

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3643_364317


namespace NUMINAMATH_CALUDE_hexagonal_diagram_impossible_l3643_364349

/-- Represents a hexagonal diagram filled with numbers -/
structure HexagonalDiagram :=
  (first_row : Fin 6 → ℕ)
  (is_valid : ∀ i : Fin 6, first_row i ∈ Finset.range 22)

/-- Calculates the sum of all numbers in the hexagonal diagram -/
def hexagon_sum (h : HexagonalDiagram) : ℕ :=
  6 * h.first_row 0 + 20 * h.first_row 1 + 34 * h.first_row 2 +
  34 * h.first_row 3 + 20 * h.first_row 4 + 6 * h.first_row 5

/-- The sum of numbers from 1 to 21 -/
def sum_1_to_21 : ℕ := (21 * 22) / 2

/-- Theorem stating the impossibility of filling the hexagonal diagram -/
theorem hexagonal_diagram_impossible :
  ¬ ∃ (h : HexagonalDiagram), hexagon_sum h = sum_1_to_21 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_diagram_impossible_l3643_364349
