import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_equation_properties_l2698_269880

theorem polynomial_equation_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) →
  (a₀ = 1 ∧ a₃ = -32 ∧ a₄ = 16 ∧ a₁ + a₂ + a₃ + a₄ = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_properties_l2698_269880


namespace NUMINAMATH_CALUDE_ratio_value_l2698_269855

theorem ratio_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 5 * d) 
  (h4 : b ≠ 0) 
  (h5 : d ≠ 0) : 
  (a * c) / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_value_l2698_269855


namespace NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l2698_269877

theorem greatest_difference_of_units_digit (x : ℕ) : 
  (x < 10) →
  (637 * 10 + x) % 3 = 0 →
  ∃ y z, y < 10 ∧ z < 10 ∧ 
         (637 * 10 + y) % 3 = 0 ∧ 
         (637 * 10 + z) % 3 = 0 ∧ 
         y - z ≤ 6 ∧
         ∀ w, w < 10 → (637 * 10 + w) % 3 = 0 → y - w ≤ 6 ∧ w - z ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l2698_269877


namespace NUMINAMATH_CALUDE_only_solutions_l2698_269833

/-- A function from nonnegative integers to nonnegative integers -/
def NonNegIntFunction := ℕ → ℕ

/-- The property that f(f(f(n))) = f(n+1) + 1 for all n -/
def SatisfiesEquation (f : NonNegIntFunction) : Prop :=
  ∀ n, f (f (f n)) = f (n + 1) + 1

/-- The first solution function: f(n) = n + 1 -/
def Solution1 : NonNegIntFunction :=
  λ n => n + 1

/-- The second solution function: 
    f(n) = n + 1 if n ≡ 0 (mod 4) or n ≡ 2 (mod 4),
    f(n) = n + 5 if n ≡ 1 (mod 4),
    f(n) = n - 3 if n ≡ 3 (mod 4) -/
def Solution2 : NonNegIntFunction :=
  λ n => match n % 4 with
    | 0 | 2 => n + 1
    | 1 => n + 5
    | 3 => n - 3
    | _ => n  -- This case is unreachable, but needed for exhaustiveness

/-- The main theorem: Solution1 and Solution2 are the only functions satisfying the equation -/
theorem only_solutions (f : NonNegIntFunction) :
  SatisfiesEquation f ↔ (f = Solution1 ∨ f = Solution2) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_l2698_269833


namespace NUMINAMATH_CALUDE_g_100_eq_520_l2698_269869

/-- The sum of greatest common divisors function -/
def g (n : ℕ+) : ℕ := (Finset.range n.val.succ).sum (fun k => Nat.gcd k n.val)

/-- The theorem stating that g(100) = 520 -/
theorem g_100_eq_520 : g 100 = 520 := by sorry

end NUMINAMATH_CALUDE_g_100_eq_520_l2698_269869


namespace NUMINAMATH_CALUDE_total_pages_in_collection_l2698_269822

/-- Represents a book in the reader's collection -/
structure Book where
  chapterPages : List Nat
  additionalPages : Nat

/-- The reader's book collection -/
def bookCollection : List Book := [
  { chapterPages := [22, 34, 18, 46, 30, 38], additionalPages := 14 },  -- Science
  { chapterPages := [24, 32, 40, 20], additionalPages := 13 },          -- History
  { chapterPages := [12, 28, 16, 22, 18, 26, 20], additionalPages := 8 }, -- Literature
  { chapterPages := [48, 52, 36, 62, 24], additionalPages := 18 },      -- Art
  { chapterPages := [16, 28, 44], additionalPages := 28 }               -- Mathematics
]

/-- Calculate the total pages in a book -/
def totalPagesInBook (book : Book) : Nat :=
  (book.chapterPages.sum) + book.additionalPages

/-- Calculate the total pages in the collection -/
def totalPagesInCollection (collection : List Book) : Nat :=
  collection.map totalPagesInBook |>.sum

/-- Theorem: The total number of pages in the reader's collection is 837 -/
theorem total_pages_in_collection :
  totalPagesInCollection bookCollection = 837 := by
  sorry


end NUMINAMATH_CALUDE_total_pages_in_collection_l2698_269822


namespace NUMINAMATH_CALUDE_deane_gas_cost_l2698_269859

/-- Calculates the total cost of gas for Mr. Deane --/
def total_gas_cost (rollback : ℝ) (current_price : ℝ) (liters_today : ℝ) (liters_friday : ℝ) : ℝ :=
  let price_friday := current_price - rollback
  let cost_today := current_price * liters_today
  let cost_friday := price_friday * liters_friday
  cost_today + cost_friday

/-- Proves that Mr. Deane's total gas cost is $39 --/
theorem deane_gas_cost :
  let rollback : ℝ := 0.4
  let current_price : ℝ := 1.4
  let liters_today : ℝ := 10
  let liters_friday : ℝ := 25
  total_gas_cost rollback current_price liters_today liters_friday = 39 := by
  sorry

#eval total_gas_cost 0.4 1.4 10 25

end NUMINAMATH_CALUDE_deane_gas_cost_l2698_269859


namespace NUMINAMATH_CALUDE_max_sum_of_abs_on_unit_sphere_l2698_269851

theorem max_sum_of_abs_on_unit_sphere :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_abs_on_unit_sphere_l2698_269851


namespace NUMINAMATH_CALUDE_miles_owns_seventeen_instruments_l2698_269821

/-- Represents the number of musical instruments Miles owns --/
structure MilesInstruments where
  fingers : ℕ
  hands : ℕ
  heads : ℕ
  trumpets : ℕ
  guitars : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- The total number of musical instruments Miles owns --/
def totalInstruments (m : MilesInstruments) : ℕ :=
  m.trumpets + m.guitars + m.trombones + m.frenchHorns

/-- Theorem stating that Miles owns 17 musical instruments --/
theorem miles_owns_seventeen_instruments (m : MilesInstruments)
  (h1 : m.fingers = 10)
  (h2 : m.hands = 2)
  (h3 : m.heads = 1)
  (h4 : m.trumpets = m.fingers - 3)
  (h5 : m.guitars = m.hands + 2)
  (h6 : m.trombones = m.heads + 2)
  (h7 : m.frenchHorns = m.guitars - 1) :
  totalInstruments m = 17 := by
  sorry

#check miles_owns_seventeen_instruments

end NUMINAMATH_CALUDE_miles_owns_seventeen_instruments_l2698_269821


namespace NUMINAMATH_CALUDE_marble_redistribution_l2698_269816

/-- Represents the number of marbles each person has -/
structure MarbleDistribution :=
  (person1 : ℕ)
  (person2 : ℕ)
  (person3 : ℕ)
  (person4 : ℕ)

/-- The theorem statement -/
theorem marble_redistribution 
  (initial : MarbleDistribution)
  (h1 : initial.person1 = 14)
  (h2 : initial.person2 = 19)
  (h3 : initial.person3 = 7)
  (h4 : ∀ (final : MarbleDistribution), 
    final.person1 + final.person2 + final.person3 + final.person4 = 
    initial.person1 + initial.person2 + initial.person3 + initial.person4)
  (h5 : ∀ (final : MarbleDistribution), 
    final.person1 = final.person2 ∧ 
    final.person2 = final.person3 ∧ 
    final.person3 = final.person4 ∧
    final.person4 = 15) :
  initial.person4 = 20 := by
sorry


end NUMINAMATH_CALUDE_marble_redistribution_l2698_269816


namespace NUMINAMATH_CALUDE_gcd_72_108_150_l2698_269875

theorem gcd_72_108_150 : Nat.gcd 72 (Nat.gcd 108 150) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_72_108_150_l2698_269875


namespace NUMINAMATH_CALUDE_proportional_function_two_quadrants_l2698_269888

/-- A proportional function passing through two quadrants -/
theorem proportional_function_two_quadrants (m : ℝ) : 
  let f : ℝ → ℝ := λ x => (m + 3) * x^(m^2 + m - 5)
  m = 2 → (∃ x y, f x = y ∧ x > 0 ∧ y > 0) ∧ 
          (∃ x y, f x = y ∧ x < 0 ∧ y < 0) ∧
          (∀ x y, f x = y → (x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≤ 0)) :=
by sorry


end NUMINAMATH_CALUDE_proportional_function_two_quadrants_l2698_269888


namespace NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l2698_269896

theorem smallest_multiple_of_one_to_five : ∃ (n : ℕ), n > 0 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → i ∣ m) → n ≤ m) ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_one_to_five_l2698_269896


namespace NUMINAMATH_CALUDE_quotient_of_composites_l2698_269837

def first_five_even_composites : List Nat := [4, 6, 8, 10, 12]
def next_five_odd_composites : List Nat := [9, 15, 21, 25, 27]

def product_even : Nat := first_five_even_composites.prod
def product_odd : Nat := next_five_odd_composites.prod

theorem quotient_of_composites :
  (product_even : ℚ) / (product_odd : ℚ) = 512 / 28525 := by
  sorry

end NUMINAMATH_CALUDE_quotient_of_composites_l2698_269837


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2698_269871

/-- Given a line y = mx + b passing through (0, 3) and reflecting (2, -4) to (-4, 8), prove that m + b = 3.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (3 = m * 0 + b) →  -- Line passes through (0, 3)
  (let midpoint_x := (2 + (-4)) / 2
   let midpoint_y := (-4 + 8) / 2
   (midpoint_y - 3) = m * (midpoint_x - 0)) →  -- Midpoint lies on the line
  (8 - (-4)) / (-4 - 2) = -1 / m →  -- Perpendicular slopes
  m + b = 3.5 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2698_269871


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l2698_269828

theorem quadrilateral_inequality (a b c : ℝ) : 
  (a > 0) →  -- EF has positive length
  (b > 0) →  -- EG has positive length
  (c > 0) →  -- EH has positive length
  (a < b) →  -- F is between E and G
  (b < c) →  -- G is between E and H
  (2 * b > c) →  -- Condition for positive area after rotation
  (a < c / 3) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l2698_269828


namespace NUMINAMATH_CALUDE_sally_rum_amount_l2698_269849

theorem sally_rum_amount (x : ℝ) : 
  (∀ (max_rum : ℝ), max_rum = 3 * x) →   -- Maximum amount is 3 times what Sally gave
  (∀ (earlier_rum : ℝ), earlier_rum = 12) →  -- Don already had 12 oz
  (∀ (remaining_rum : ℝ), remaining_rum = 8) →  -- Don can still have 8 oz
  (x + 12 + 8 = 3 * x) →  -- Total amount equals maximum healthy amount
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_sally_rum_amount_l2698_269849


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l2698_269843

-- Define the variables and conditions
variable (x y : ℝ)
variable (h1 : 2 * x + 3 * y = 18)
variable (h2 : x * y = 8)

-- Define the quadratic polynomial
def f (t : ℝ) := t^2 - 18*t + 8

-- State the theorem
theorem roots_of_quadratic :
  f x = 0 ∧ f y = 0 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l2698_269843


namespace NUMINAMATH_CALUDE_min_a_value_l2698_269814

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem min_a_value (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, g (-x) = g x) →   -- g is even
  (∀ x, f x + g x = 2^x) →  -- f(x) + g(x) = 2^x
  (∀ x ∈ Set.Icc 1 2, a * f x + g (2*x) ≥ 0) →  -- inequality holds for x ∈ [1, 2]
  a ≥ -17/6 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l2698_269814


namespace NUMINAMATH_CALUDE_jesselton_orchestra_max_size_l2698_269800

theorem jesselton_orchestra_max_size :
  ∀ n m : ℕ,
  n = 30 * m →
  n % 32 = 7 →
  n < 1200 →
  (∀ k : ℕ, k = 30 * m ∧ k % 32 = 7 ∧ k < 1200 → k ≤ n) →
  n = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_jesselton_orchestra_max_size_l2698_269800


namespace NUMINAMATH_CALUDE_alice_painted_cuboids_l2698_269852

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := 6

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces painted -/
def total_faces_painted : ℕ := 36

theorem alice_painted_cuboids :
  num_cuboids * faces_per_cuboid = total_faces_painted :=
by sorry

end NUMINAMATH_CALUDE_alice_painted_cuboids_l2698_269852


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2698_269862

theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2698_269862


namespace NUMINAMATH_CALUDE_largest_interesting_is_max_l2698_269838

/-- A natural number is interesting if all its digits, except for the first and last,
    are less than the arithmetic mean of their two neighboring digits. -/
def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

/-- The largest interesting number -/
def largest_interesting : ℕ := 96433469

theorem largest_interesting_is_max :
  is_interesting largest_interesting ∧
  ∀ n : ℕ, is_interesting n → n ≤ largest_interesting :=
sorry

end NUMINAMATH_CALUDE_largest_interesting_is_max_l2698_269838


namespace NUMINAMATH_CALUDE_circle_through_three_points_l2698_269895

/-- The equation of a circle passing through three given points -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y - 12 = 0

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (5, 1)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (6, 0)

/-- Point C coordinates -/
def point_C : ℝ × ℝ := (-1, 1)

/-- Theorem stating that the given equation represents the unique circle passing through the three points -/
theorem circle_through_three_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 ∧
  (∀ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 ↔ circle_equation x y) →
    D = -4 ∧ E = 6 ∧ F = -12) :=
by sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l2698_269895


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l2698_269891

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.6)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l2698_269891


namespace NUMINAMATH_CALUDE_final_amoeba_is_blue_l2698_269858

/-- Represents the color of an amoeba -/
inductive AmoebaCop
  | Red
  | Blue
  | Yellow

/-- Represents the state of the puddle -/
structure PuddleState where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Determines if a number is odd -/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- The initial state of the puddle -/
def initialState : PuddleState :=
  { red := 26, blue := 31, yellow := 16 }

/-- Determines the color of the final amoeba based on the initial state -/
def finalAmoeba (state : PuddleState) : AmoebaCop :=
  if isOdd (state.red - state.blue) ∧ 
     isOdd (state.blue - state.yellow) ∧ 
     ¬isOdd (state.red - state.yellow)
  then AmoebaCop.Blue
  else if isOdd (state.red - state.blue) ∧ 
          isOdd (state.red - state.yellow) ∧ 
          ¬isOdd (state.blue - state.yellow)
  then AmoebaCop.Red
  else AmoebaCop.Yellow

theorem final_amoeba_is_blue :
  finalAmoeba initialState = AmoebaCop.Blue :=
by
  sorry


end NUMINAMATH_CALUDE_final_amoeba_is_blue_l2698_269858


namespace NUMINAMATH_CALUDE_brick_width_calculation_l2698_269878

theorem brick_width_calculation (courtyard_length courtyard_width : ℝ)
                                (brick_length : ℝ)
                                (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ (brick_width : ℝ),
    brick_width = 0.1 ∧
    courtyard_length * courtyard_width * 10000 = total_bricks * brick_length * brick_width :=
by
  sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l2698_269878


namespace NUMINAMATH_CALUDE_temperature_function_properties_l2698_269882

-- Define the temperature function
def T (t : ℝ) : ℝ := t^3 - 3*t + 60

-- Define the theorem
theorem temperature_function_properties :
  -- Conditions
  (T (-4) = 8) ∧
  (T 0 = 60) ∧
  (T 1 = 58) ∧
  (deriv T (-4) = deriv T 4) ∧
  -- Conclusions
  (∀ t ∈ Set.Icc (-2) 2, T t ≤ 62) ∧
  (T (-1) = 62) ∧
  (T 2 = 62) :=
by sorry

end NUMINAMATH_CALUDE_temperature_function_properties_l2698_269882


namespace NUMINAMATH_CALUDE_regular_price_calculation_l2698_269811

/-- Represents the promotional offer and total paid for tires -/
structure TireOffer where
  regularPrice : ℝ  -- Regular price of one tire
  totalPaid : ℝ     -- Total amount paid for four tires
  fourthTirePrice : ℝ -- Price of the fourth tire in the offer

/-- The promotional offer satisfies the given conditions -/
def validOffer (offer : TireOffer) : Prop :=
  offer.totalPaid = 3 * offer.regularPrice + offer.fourthTirePrice

/-- The theorem to prove -/
theorem regular_price_calculation (offer : TireOffer) 
  (h1 : offer.totalPaid = 310)
  (h2 : offer.fourthTirePrice = 5)
  (h3 : validOffer offer) :
  offer.regularPrice = 101.67 := by
  sorry


end NUMINAMATH_CALUDE_regular_price_calculation_l2698_269811


namespace NUMINAMATH_CALUDE_chandra_akiko_ratio_l2698_269868

/-- Represents the points scored by each player in the basketball game -/
structure GameScores where
  chandra : ℕ
  akiko : ℕ
  michiko : ℕ
  bailey : ℕ

/-- The conditions of the basketball game -/
def gameConditions (s : GameScores) : Prop :=
  s.akiko = s.michiko + 4 ∧
  s.michiko * 2 = s.bailey ∧
  s.bailey = 14 ∧
  s.chandra + s.akiko + s.michiko + s.bailey = 54

/-- The theorem stating the ratio of Chandra's points to Akiko's points -/
theorem chandra_akiko_ratio (s : GameScores) : 
  gameConditions s → s.chandra * 1 = s.akiko * 2 := by
  sorry

#check chandra_akiko_ratio

end NUMINAMATH_CALUDE_chandra_akiko_ratio_l2698_269868


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2698_269874

/-- The polynomial p(x) -/
def p (x : ℝ) : ℝ := -4*x^2 + 2*x - 5

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := -6*x^2 + 4*x - 9

/-- The polynomial r(x) -/
def r (x : ℝ) : ℝ := 6*x^2 + 6*x + 2

/-- The polynomial s(x) -/
def s (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

/-- The sum of polynomials p(x), q(x), r(x), and s(x) is equal to -x^2 + 10x - 11 -/
theorem sum_of_polynomials (x : ℝ) : p x + q x + r x + s x = -x^2 + 10*x - 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2698_269874


namespace NUMINAMATH_CALUDE_brothers_money_l2698_269884

theorem brothers_money (a₁ a₂ a₃ a₄ : ℚ) :
  a₁ + a₂ + a₃ + a₄ = 48 ∧
  a₁ + 3 = a₂ - 3 ∧
  a₁ + 3 = 3 * a₃ ∧
  a₁ + 3 = a₄ / 3 →
  a₁ = 6 ∧ a₂ = 12 ∧ a₃ = 3 ∧ a₄ = 27 := by
sorry

end NUMINAMATH_CALUDE_brothers_money_l2698_269884


namespace NUMINAMATH_CALUDE_initial_medium_size_shoes_l2698_269856

/-- Given a shoe shop's inventory and sales data, prove the initial number of medium-size shoes. -/
theorem initial_medium_size_shoes
  (large_size : Nat) -- Initial number of large-size shoes
  (small_size : Nat) -- Initial number of small-size shoes
  (sold : Nat) -- Number of shoes sold
  (remaining : Nat) -- Number of shoes remaining after sale
  (h1 : large_size = 22)
  (h2 : small_size = 24)
  (h3 : sold = 83)
  (h4 : remaining = 13)
  (h5 : ∃ M : Nat, large_size + M + small_size = sold + remaining) :
  ∃ M : Nat, M = 26 ∧ large_size + M + small_size = sold + remaining :=
by sorry


end NUMINAMATH_CALUDE_initial_medium_size_shoes_l2698_269856


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_10080_l2698_269889

/-- Given that 10080 = 2^4 * 3^2 * 5 * 7, this function counts the number of positive integer factors of 10080 that are perfect squares. -/
def count_perfect_square_factors : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1), (7, 1)]
  -- Function implementation
  sorry

/-- The number of positive integer factors of 10080 that are perfect squares is 6. -/
theorem perfect_square_factors_of_10080 : count_perfect_square_factors = 6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_10080_l2698_269889


namespace NUMINAMATH_CALUDE_sin_150_degrees_l2698_269842

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l2698_269842


namespace NUMINAMATH_CALUDE_power_digit_cycle_l2698_269863

theorem power_digit_cycle (n : ℤ) (k : ℕ) : n^(k+4) ≡ n^k [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_power_digit_cycle_l2698_269863


namespace NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l2698_269819

/-- Proves that the speed of Car A is 58 miles per hour given the initial conditions -/
theorem car_a_speed (initial_distance : ℝ) (time : ℝ) (speed_b : ℝ) : ℝ :=
  let distance_b := speed_b * time
  let total_distance := distance_b + initial_distance + 8
  total_distance / time

#check car_a_speed 24 4 50 = 58

/-- Theorem stating that the speed of Car A is indeed 58 miles per hour -/
theorem car_a_speed_is_58 :
  car_a_speed 24 4 50 = 58 := by sorry

end NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l2698_269819


namespace NUMINAMATH_CALUDE_sequence_general_term_l2698_269826

/-- The sequence a_n defined by a_1 = 2 and a_{n+1} = 2a_n for n ≥ 1 has the general term a_n = 2^n -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2698_269826


namespace NUMINAMATH_CALUDE_kids_played_monday_l2698_269857

theorem kids_played_monday (total : ℕ) (tuesday : ℕ) (h1 : total = 16) (h2 : tuesday = 14) :
  total - tuesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kids_played_monday_l2698_269857


namespace NUMINAMATH_CALUDE_remainder_seven_pow_2023_mod_5_l2698_269813

theorem remainder_seven_pow_2023_mod_5 : 7^2023 % 5 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_seven_pow_2023_mod_5_l2698_269813


namespace NUMINAMATH_CALUDE_greatest_b_value_l2698_269873

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2698_269873


namespace NUMINAMATH_CALUDE_inequality_solution_l2698_269827

theorem inequality_solution (m : ℝ) (x : ℝ) :
  (m * x - 2 ≥ 3 * x - 4 * m) ↔
  (m > 3 ∧ x ≥ (2 - 4*m) / (m - 3)) ∨
  (m < 3 ∧ x ≤ (2 - 4*m) / (m - 3)) ∨
  (m = 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2698_269827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2698_269844

/-- Given an arithmetic sequence {a_n} with a₁ = 1, d = 2, and aₙ = 19, prove that n = 10 -/
theorem arithmetic_sequence_nth_term (a : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) - a k = 2) →  -- common difference is 2
  a 1 = 1 →                     -- first term is 1
  a n = 19 →                    -- n-th term is 19
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l2698_269844


namespace NUMINAMATH_CALUDE_exam_survey_analysis_l2698_269848

structure SurveyData where
  total_candidates : Nat
  sample_size : Nat

def sampling_survey_method (data : SurveyData) : Prop :=
  data.sample_size < data.total_candidates

def is_population (data : SurveyData) (n : Nat) : Prop :=
  n = data.total_candidates

def is_sample (data : SurveyData) (n : Nat) : Prop :=
  n = data.sample_size

theorem exam_survey_analysis (data : SurveyData)
  (h1 : data.total_candidates = 60000)
  (h2 : data.sample_size = 1000) :
  ∃ (correct_statements : Finset (Fin 4)),
    correct_statements.card = 2 ∧
    (1 ∈ correct_statements ↔ sampling_survey_method data) ∧
    (2 ∈ correct_statements ↔ is_population data data.total_candidates) ∧
    (3 ∈ correct_statements ↔ is_sample data data.sample_size) ∧
    (4 ∈ correct_statements ↔ data.sample_size = 1000) :=
sorry

end NUMINAMATH_CALUDE_exam_survey_analysis_l2698_269848


namespace NUMINAMATH_CALUDE_f_properties_l2698_269839

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x + a) - (1/2) * Real.log x

theorem f_properties :
  (∀ x > 0, ∀ a : ℝ,
    (a = 0 → (∀ y > (1/2), f a y > f a x) ∧ (∀ z ∈ Set.Ioo 0 (1/2), f a z < f a x)) ∧
    (a < 0 →
      (a < -2 → ∃ x₁ x₂, x₁ = (-a - Real.sqrt (a^2 - 4)) / 4 ∧
                         x₂ = (-a + Real.sqrt (a^2 - 4)) / 4 ∧
                         (∀ y ≠ x₁, f a y ≥ f a x₁) ∧
                         (∀ y ≠ x₂, f a y ≤ f a x₂)) ∧
      (-2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2 → ∀ y > 0, f a y ≠ f a x) ∧
      (-Real.sqrt 2 / 2 < a ∧ a < 0 →
        ∃ x₃, x₃ = (-a + Real.sqrt (a^2 + 4)) / 4 ∧
               (∀ y ≠ x₃, f a y ≥ f a x₃)))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2698_269839


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2698_269865

open Set Real

theorem condition_necessary_not_sufficient :
  let A : Set ℝ := {x | 0 < x ∧ x < 3}
  let B : Set ℝ := {x | log (x - 2) < 0}
  B ⊂ A ∧ B ≠ A := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2698_269865


namespace NUMINAMATH_CALUDE_trig_identity_l2698_269808

theorem trig_identity (α : Real) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2698_269808


namespace NUMINAMATH_CALUDE_footprint_calculation_l2698_269831

/-- Calculates the total number of footprints left by three creatures on their respective planets -/
theorem footprint_calculation (pogo_rate : ℕ) (grimzi_rate : ℕ) (zeb_rate : ℕ)
  (pogo_distance : ℕ) (grimzi_distance : ℕ) (zeb_distance : ℕ)
  (total_distance : ℕ) :
  pogo_rate = 4 ∧ 
  grimzi_rate = 3 ∧ 
  zeb_rate = 5 ∧
  pogo_distance = 1 ∧ 
  grimzi_distance = 6 ∧ 
  zeb_distance = 8 ∧
  total_distance = 6000 →
  pogo_rate * total_distance + 
  (total_distance / grimzi_distance) * grimzi_rate + 
  (total_distance / zeb_distance) * zeb_rate = 30750 := by
sorry


end NUMINAMATH_CALUDE_footprint_calculation_l2698_269831


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l2698_269818

theorem sum_of_integers_ending_in_3 :
  let first_term : ℕ := 103
  let last_term : ℕ := 493
  let common_difference : ℕ := 10
  let n : ℕ := (last_term - first_term) / common_difference + 1
  let sum : ℕ := n * (first_term + last_term) / 2
  sum = 11920 := by sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l2698_269818


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_train_platform_crossing_time_specific_l2698_269815

/-- Calculates the time taken for a train to cross a platform -/
theorem train_platform_crossing_time 
  (train_length platform_length : ℝ) 
  (time_to_cross_pole : ℝ) : ℝ :=
  let train_speed := train_length / time_to_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Proves that the time taken for a 300m train to cross a 250m platform 
    is approximately 33 seconds, given that it takes 18 seconds to cross a signal pole -/
theorem train_platform_crossing_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_platform_crossing_time 300 250 18 - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_train_platform_crossing_time_specific_l2698_269815


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2698_269802

-- Define the property of being a quadratic equation
def is_quadratic (m : ℤ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, x^(m+1) - (m+1)*x - 2 = a*x^2 + b*x + c

-- State the theorem
theorem quadratic_equation_m_value :
  is_quadratic m → m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2698_269802


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2698_269830

theorem inequality_not_always_true (x y : ℝ) (h : x > 1 ∧ 1 > y) :
  ¬ (∀ x y : ℝ, x > 1 ∧ 1 > y → x - 1 > 1 - y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2698_269830


namespace NUMINAMATH_CALUDE_problem_statement_l2698_269853

theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 4) :
  a^4 + 1/a^4 = -158/81 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2698_269853


namespace NUMINAMATH_CALUDE_ring_weight_sum_l2698_269885

/-- The weight of the orange ring in ounces -/
def orange_ring : ℚ := 0.08

/-- The weight of the purple ring in ounces -/
def purple_ring : ℚ := 0.33

/-- The weight of the white ring in ounces -/
def white_ring : ℚ := 0.42

/-- The total weight of all rings in ounces -/
def total_weight : ℚ := orange_ring + purple_ring + white_ring

theorem ring_weight_sum :
  total_weight = 0.83 := by sorry

end NUMINAMATH_CALUDE_ring_weight_sum_l2698_269885


namespace NUMINAMATH_CALUDE_max_distance_between_points_l2698_269825

/-- Given vector OA = (1, -1) and |OA| = |OB|, the maximum value of |AB| is 2√2. -/
theorem max_distance_between_points (OA OB : ℝ × ℝ) : 
  OA = (1, -1) → 
  Real.sqrt ((OA.1 ^ 2) + (OA.2 ^ 2)) = Real.sqrt ((OB.1 ^ 2) + (OB.2 ^ 2)) →
  (∃ (AB : ℝ × ℝ), AB = OB - OA ∧ 
    Real.sqrt ((AB.1 ^ 2) + (AB.2 ^ 2)) ≤ 2 * Real.sqrt 2 ∧
    ∃ (OB' : ℝ × ℝ), Real.sqrt ((OB'.1 ^ 2) + (OB'.2 ^ 2)) = Real.sqrt ((OA.1 ^ 2) + (OA.2 ^ 2)) ∧
      let AB' := OB' - OA
      Real.sqrt ((AB'.1 ^ 2) + (AB'.2 ^ 2)) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_points_l2698_269825


namespace NUMINAMATH_CALUDE_tan_roots_sum_l2698_269864

theorem tan_roots_sum (α β : Real) : 
  (∃ x y : Real, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  α > -π/2 ∧ α < π/2 ∧ β > -π/2 ∧ β < π/2 →
  α + β = π/3 ∨ α + β = -2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_roots_sum_l2698_269864


namespace NUMINAMATH_CALUDE_flower_pattern_perimeter_l2698_269809

/-- The perimeter of a "flower" pattern formed by removing a 45° sector from a circle --/
theorem flower_pattern_perimeter (r : ℝ) (h : r = 3) : 
  let circumference := 2 * π * r
  let arc_length := (315 / 360) * circumference
  let straight_edges := 2 * r
  arc_length + straight_edges = (21 / 4) * π + 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_pattern_perimeter_l2698_269809


namespace NUMINAMATH_CALUDE_vector_linear_combination_l2698_269804

/-- Given vectors a, b, and c in R², prove that c is a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, -2)) : 
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l2698_269804


namespace NUMINAMATH_CALUDE_residue_mod_12_l2698_269845

theorem residue_mod_12 : (172 * 15 - 13 * 8 + 6) % 12 = 10 := by sorry

end NUMINAMATH_CALUDE_residue_mod_12_l2698_269845


namespace NUMINAMATH_CALUDE_inequality_proof_l2698_269861

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2698_269861


namespace NUMINAMATH_CALUDE_elena_earnings_l2698_269898

def charging_sequence : List Nat := [3, 4, 5, 6, 7]

def calculate_earnings (hours : Nat) : Nat :=
  let complete_cycles := hours / 5
  let remaining_hours := hours % 5
  let cycle_earnings := charging_sequence.sum * complete_cycles
  let remaining_earnings := (charging_sequence.take remaining_hours).sum
  cycle_earnings + remaining_earnings

theorem elena_earnings :
  calculate_earnings 47 = 232 := by
  sorry

end NUMINAMATH_CALUDE_elena_earnings_l2698_269898


namespace NUMINAMATH_CALUDE_sum_of_digits_l2698_269824

/-- 
Given a three-digit number ABC, where:
- ABC is an integer between 100 and 999 (inclusive)
- ABC = 17 * 28 + 9

Prove that the sum of its digits A, B, and C is 17.
-/
theorem sum_of_digits (ABC : ℕ) (h1 : 100 ≤ ABC) (h2 : ABC ≤ 999) (h3 : ABC = 17 * 28 + 9) :
  (ABC / 100) + ((ABC / 10) % 10) + (ABC % 10) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2698_269824


namespace NUMINAMATH_CALUDE_percentage_calculation_l2698_269890

theorem percentage_calculation (x : ℝ) (h : 0.2 * x = 300) : 1.2 * x = 1800 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2698_269890


namespace NUMINAMATH_CALUDE_system_equation_solution_range_l2698_269829

theorem system_equation_solution_range (x y m : ℝ) : 
  (3 * x + y = m - 1) → 
  (x - 3 * y = 2 * m) → 
  (x + 2 * y ≥ 0) → 
  (m ≤ -1) := by
sorry

end NUMINAMATH_CALUDE_system_equation_solution_range_l2698_269829


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l2698_269867

/-- The area of a regular hexagon with vertices A(0,0) and C(4,6) is 78√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (4, 6)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let hexagon_area : ℝ := 6 * (Real.sqrt 3 / 4 * AC^2)
  hexagon_area = 78 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_regular_hexagon_area_l2698_269867


namespace NUMINAMATH_CALUDE_C_power_50_l2698_269832

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 1;
    -4, -1]

theorem C_power_50 :
  C^50 = !![101, 50;
            -200, -99] := by
  sorry

end NUMINAMATH_CALUDE_C_power_50_l2698_269832


namespace NUMINAMATH_CALUDE_solution_set_contains_two_and_zero_l2698_269879

/-- The solution set of the inequality (1+k²)x ≤ k⁴+4 with respect to x -/
def M (k : ℝ) : Set ℝ :=
  {x : ℝ | (1 + k^2) * x ≤ k^4 + 4}

/-- For any real constant k, both 2 and 0 are in the solution set M -/
theorem solution_set_contains_two_and_zero :
  ∀ k : ℝ, (2 ∈ M k) ∧ (0 ∈ M k) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_contains_two_and_zero_l2698_269879


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l2698_269886

theorem min_value_of_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a - 2*a*b + b = 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x - 2*x*y + y = 0 → 2*x + y ≥ 2*a + b) →
  2*a + b = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l2698_269886


namespace NUMINAMATH_CALUDE_fraction_equality_l2698_269834

theorem fraction_equality (x : ℚ) : (1/5)^35 * x^18 = 1/(2*(10)^35) → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2698_269834


namespace NUMINAMATH_CALUDE_equation_solution_l2698_269897

theorem equation_solution :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2698_269897


namespace NUMINAMATH_CALUDE_smallest_prime_after_five_nonprimes_l2698_269841

/-- A function that returns true if a natural number is prime, false otherwise -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nth_prime (n : ℕ) : ℕ := sorry

/-- A function that returns true if there are at least five consecutive nonprime numbers before n, false otherwise -/
def five_consecutive_nonprimes_before (n : ℕ) : Prop := sorry

theorem smallest_prime_after_five_nonprimes : 
  ∃ (n : ℕ), is_prime n ∧ five_consecutive_nonprimes_before n ∧ 
  ∀ (m : ℕ), m < n → ¬(is_prime m ∧ five_consecutive_nonprimes_before m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_five_nonprimes_l2698_269841


namespace NUMINAMATH_CALUDE_determine_a_l2698_269810

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 1-a}

-- Define set A
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Theorem statement
theorem determine_a : 
  ∀ a : ℝ, (U a \ A a = {-1}) → a = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_determine_a_l2698_269810


namespace NUMINAMATH_CALUDE_triangle_problem_l2698_269812

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π/4 ∧
  b = Real.sqrt 6 ∧
  (1/2) * b * c * Real.sin A = (3 + Real.sqrt 3)/2 →
  c = 1 + Real.sqrt 3 ∧ B = π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2698_269812


namespace NUMINAMATH_CALUDE_fruit_basket_composition_l2698_269803

/-- Represents a fruit basket -/
structure FruitBasket where
  apples : ℕ
  pears : ℕ
  others : ℕ

/-- The total number of fruits in the basket -/
def FruitBasket.total (b : FruitBasket) : ℕ := b.apples + b.pears + b.others

/-- Predicate to check if any 3 fruits contain an apple -/
def hasAppleIn3 (b : FruitBasket) : Prop :=
  b.pears + b.others ≤ 2

/-- Predicate to check if any 4 fruits contain a pear -/
def hasPearIn4 (b : FruitBasket) : Prop :=
  b.apples + b.others ≤ 3

/-- The main theorem -/
theorem fruit_basket_composition (b : FruitBasket) :
  b.total ≥ 5 →
  hasAppleIn3 b →
  hasPearIn4 b →
  b.apples = 3 ∧ b.pears = 2 ∧ b.others = 0 :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_composition_l2698_269803


namespace NUMINAMATH_CALUDE_cricketer_wickets_after_match_l2698_269876

/-- Represents a cricketer's bowling statistics -/
structure CricketerStats where
  wickets : ℕ
  runs : ℕ
  average : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : CricketerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

/-- Theorem: A cricketer with given stats takes 5 wickets for 26 runs, decreasing average by 0.4 -/
theorem cricketer_wickets_after_match 
  (stats : CricketerStats)
  (h1 : stats.average = 12.4)
  (h2 : newAverage stats 5 26 = stats.average - 0.4) :
  stats.wickets + 5 = 90 := by
sorry

end NUMINAMATH_CALUDE_cricketer_wickets_after_match_l2698_269876


namespace NUMINAMATH_CALUDE_binomial_coefficient_17_8_l2698_269892

theorem binomial_coefficient_17_8 (h1 : Nat.choose 15 6 = 5005) 
                                  (h2 : Nat.choose 15 7 = 6435) 
                                  (h3 : Nat.choose 15 8 = 6435) : 
  Nat.choose 17 8 = 24310 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_17_8_l2698_269892


namespace NUMINAMATH_CALUDE_parentheses_value_l2698_269805

theorem parentheses_value (x : ℤ) (h : x - (-2) = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_value_l2698_269805


namespace NUMINAMATH_CALUDE_tire_circumference_l2698_269846

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference (revolutions_per_minute : ℝ) (car_speed_kmh : ℝ) :
  revolutions_per_minute = 400 →
  car_speed_kmh = 48 →
  (car_speed_kmh * 1000 / 60) / revolutions_per_minute = 2 :=
by sorry

end NUMINAMATH_CALUDE_tire_circumference_l2698_269846


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l2698_269854

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l2698_269854


namespace NUMINAMATH_CALUDE_cake_distribution_l2698_269801

theorem cake_distribution (total_pieces : ℕ) (pieces_per_friend : ℕ) (num_friends : ℕ) :
  total_pieces = 150 →
  pieces_per_friend = 3 →
  total_pieces = pieces_per_friend * num_friends →
  num_friends = 50 := by
sorry

end NUMINAMATH_CALUDE_cake_distribution_l2698_269801


namespace NUMINAMATH_CALUDE_jack_morning_emails_l2698_269887

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- The total number of emails Jack received in the morning and evening -/
def total_morning_evening : ℕ := 11

/-- Theorem stating that Jack received 3 emails in the morning -/
theorem jack_morning_emails :
  morning_emails = 3 :=
by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l2698_269887


namespace NUMINAMATH_CALUDE_carol_trivia_game_points_l2698_269806

theorem carol_trivia_game_points (first_round : ℕ) (last_round : ℤ) (final_score : ℕ) 
  (h1 : first_round = 17)
  (h2 : last_round = -16)
  (h3 : final_score = 7) :
  ∃ second_round : ℕ, (first_round : ℤ) + second_round + last_round = final_score ∧ second_round = 6 := by
  sorry

end NUMINAMATH_CALUDE_carol_trivia_game_points_l2698_269806


namespace NUMINAMATH_CALUDE_pencil_cost_l2698_269893

/-- If 120 pencils cost $40, then 3600 pencils will cost $1200. -/
theorem pencil_cost (cost_120 : ℕ) (pencils : ℕ) :
  cost_120 = 40 ∧ pencils = 3600 → pencils * cost_120 / 120 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2698_269893


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2698_269894

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2698_269894


namespace NUMINAMATH_CALUDE_carpet_dimensions_l2698_269850

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  width : ℝ

/-- Represents a rectangular carpet --/
structure Carpet where
  length : ℝ
  width : ℝ

/-- Checks if a carpet fits in a room such that each corner touches a different wall --/
def fits_in_room (c : Carpet) (r : Room) : Prop :=
  ∃ (α b : ℝ),
    α + (c.length / c.width) * b = r.length ∧
    (c.length / c.width) * α + b = r.width ∧
    c.width^2 = α^2 + b^2

/-- The main theorem to prove --/
theorem carpet_dimensions :
  ∃ (c : Carpet),
    fits_in_room c { length := 38, width := 55 } ∧
    fits_in_room c { length := 50, width := 55 } ∧
    c.length = 50 ∧
    c.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_carpet_dimensions_l2698_269850


namespace NUMINAMATH_CALUDE_min_xy_value_l2698_269817

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 8) :
  (x : ℚ) * y ≥ 128 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l2698_269817


namespace NUMINAMATH_CALUDE_subtraction_problem_l2698_269807

theorem subtraction_problem (x : ℤ) : 821 - x = 267 → x - 267 = 287 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2698_269807


namespace NUMINAMATH_CALUDE_smallest_white_buttons_l2698_269860

theorem smallest_white_buttons (n : ℕ) (h1 : n % 10 = 0) : 
  (n / 2 : ℚ) + (n / 5 : ℚ) + 8 ≤ n → 
  (∃ m : ℕ, m ≥ 1 ∧ (n : ℚ) - ((n / 2 : ℚ) + (n / 5 : ℚ) + 8) = m) →
  (∃ k : ℕ, k ≥ 1 ∧ (30 : ℚ) - ((30 / 2 : ℚ) + (30 / 5 : ℚ) + 8) = k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_white_buttons_l2698_269860


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2698_269820

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
def CircleEquation (h k r : ℝ) : ℝ × ℝ → Prop :=
  λ p => (p.1 - h)^2 + (p.2 - k)^2 = r^2

/-- The center of a circle given by its equation --/
def CircleCenter (eq : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem circle_center_coordinates :
  CircleCenter (CircleEquation 1 2 1) = (1, 2) := by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2698_269820


namespace NUMINAMATH_CALUDE_student_pairs_l2698_269840

theorem student_pairs (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l2698_269840


namespace NUMINAMATH_CALUDE_largest_changeable_digit_is_nine_l2698_269835

/-- The original incorrect sum --/
def original_sum : ℕ := 2436

/-- The correct sum of the addends --/
def correct_sum : ℕ := 731 + 962 + 843

/-- The difference between the correct sum and the original sum --/
def difference : ℕ := correct_sum - original_sum

/-- The largest digit in the hundreds place of the addends --/
def largest_hundreds_digit : ℕ := max (731 / 100) (max (962 / 100) (843 / 100))

theorem largest_changeable_digit_is_nine :
  largest_hundreds_digit = 9 ∧ difference = 100 :=
sorry

end NUMINAMATH_CALUDE_largest_changeable_digit_is_nine_l2698_269835


namespace NUMINAMATH_CALUDE_positive_solution_form_l2698_269866

theorem positive_solution_form (x : ℝ) (a b : ℕ+) :
  x^2 + 14*x = 82 →
  x > 0 →
  x = Real.sqrt a - b →
  a + b = 138 :=
by
  sorry

end NUMINAMATH_CALUDE_positive_solution_form_l2698_269866


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l2698_269870

/-- The value of a for a hyperbola with given properties -/
def hyperbola_a : ℝ → Prop := λ a =>
  a > 0 ∧
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / 4 = 1 ∧
  (y = 2 * x / a ∨ y = -2 * x / a) ∧
  x = 2 ∧ y = 1

/-- Theorem: The value of a for the given hyperbola is 4 -/
theorem hyperbola_a_value : ∃ (a : ℝ), hyperbola_a a ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l2698_269870


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l2698_269823

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the points E and F
def E : ℝ × ℝ := (1, -3)
def F : ℝ × ℝ := (0, 4)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  ∃ (A B : ℝ × ℝ) (C2 : ℝ → ℝ → Prop),
    (∀ x y, C1 x y ∧ l x y ↔ (x, y) = A ∨ (x, y) = B) ∧
    (C2 E.1 E.2 ∧ C2 F.1 F.2) ∧
    (∃ D E F, ∀ x y, C2 x y ↔ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (∃ k, ∀ x y, (C1 x y ∧ C2 x y) → (∃ c, x + k*y = c ∧ ∀ x' y', parallel_line x' y' → ∃ c', x' + k*y' = c')) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 ∧
    (∀ x y, C2 x y ↔ x^2 + y^2 + 6*x - 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l2698_269823


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l2698_269872

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem 1: When a = 0, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_zero :
  A 0 ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ⊆ B if and only if 1 ≤ a ≤ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ 1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l2698_269872


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l2698_269847

theorem exponential_equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^x * 9^x = 27^(x - 20) ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l2698_269847


namespace NUMINAMATH_CALUDE_shrink_ray_effect_l2698_269881

/-- Represents the shrink ray's effect on volume -/
def shrink_factor : ℝ := 0.5

/-- The number of coffee cups -/
def num_cups : ℕ := 5

/-- The initial volume of coffee in each cup (in ounces) -/
def initial_volume : ℝ := 8

/-- Calculates the total volume of coffee after shrinking -/
def total_volume_after_shrink : ℝ := num_cups * (initial_volume * shrink_factor)

theorem shrink_ray_effect :
  total_volume_after_shrink = 20 := by sorry

end NUMINAMATH_CALUDE_shrink_ray_effect_l2698_269881


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l2698_269836

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_8th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 23)
  (h_6th : a 6 = 47) :
  a 8 = 71 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l2698_269836


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2698_269899

/-- Given a function f(x) = (x+a)e^x that satisfies f(x) ≥ (1/6)x^3 - x - 2 for all x ∈ ℝ,
    prove that a ≥ -2. -/
theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x : ℝ, (x + a) * Real.exp x ≥ (1/6) * x^3 - x - 2) →
  a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2698_269899


namespace NUMINAMATH_CALUDE_log_positive_iff_greater_than_one_l2698_269883

theorem log_positive_iff_greater_than_one (x : ℝ) : x > 1 ↔ Real.log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_log_positive_iff_greater_than_one_l2698_269883
