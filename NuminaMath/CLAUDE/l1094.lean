import Mathlib

namespace set_operations_with_empty_l1094_109492

theorem set_operations_with_empty (A : Set α) : 
  (A ∩ ∅ = ∅) ∧ 
  (A ∪ ∅ = A) ∧ 
  ((A ∩ ∅ = ∅) ∧ (A ∪ ∅ = A)) ∧ 
  ((A ∩ ∅ = ∅) ∨ (A ∪ ∅ = A)) ∧ 
  ¬¬(A ∩ ∅ = ∅) ∧ 
  ¬¬(A ∪ ∅ = A) := by
  sorry

end set_operations_with_empty_l1094_109492


namespace father_current_age_l1094_109413

/-- The age of the daughter now -/
def daughter_age : ℕ := 10

/-- The age of the father now -/
def father_age : ℕ := 4 * daughter_age

/-- In 20 years, the father will be twice as old as the daughter -/
axiom future_relation : father_age + 20 = 2 * (daughter_age + 20)

theorem father_current_age : father_age = 40 := by
  sorry

end father_current_age_l1094_109413


namespace digit_reversal_difference_exists_198_difference_l1094_109462

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the hundreds and units digits of a three-digit number -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.units
    tens := n.tens
    units := n.hundreds
    h_hundreds := n.h_units
    h_tens := n.h_tens
    h_units := n.h_hundreds }

theorem digit_reversal_difference (n : ThreeDigitNumber) :
  ∃ k : ℕ, n.value - n.reverse.value = 99 * k ∨ n.reverse.value - n.value = 99 * k :=
sorry

theorem exists_198_difference :
  ∃ n : ThreeDigitNumber, n.value - n.reverse.value = 198 ∨ n.reverse.value - n.value = 198 :=
sorry

end digit_reversal_difference_exists_198_difference_l1094_109462


namespace ratio_w_to_y_l1094_109449

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 5 / 3)
  (hz_x : z / x = 1 / 6) :
  w / y = 24 / 5 := by
  sorry

end ratio_w_to_y_l1094_109449


namespace juanita_daily_cost_l1094_109434

/-- The amount Juanita spends on a newspaper from Monday through Saturday -/
def daily_cost : ℝ := sorry

/-- Grant's yearly newspaper cost -/
def grant_yearly_cost : ℝ := 200

/-- Juanita's Sunday newspaper cost -/
def sunday_cost : ℝ := 2

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Difference between Juanita's and Grant's yearly newspaper costs -/
def cost_difference : ℝ := 60

theorem juanita_daily_cost :
  daily_cost * 6 * weeks_per_year + sunday_cost * weeks_per_year = 
  grant_yearly_cost + cost_difference :=
by sorry

end juanita_daily_cost_l1094_109434


namespace fathers_remaining_chocolates_fathers_remaining_chocolates_eq_five_l1094_109410

theorem fathers_remaining_chocolates 
  (initial_chocolates : ℕ) 
  (num_sisters : ℕ) 
  (chocolates_to_mother : ℕ) 
  (chocolates_eaten : ℕ) : ℕ :=
  let total_people := num_sisters + 1
  let chocolates_per_person := initial_chocolates / total_people
  let chocolates_given_to_father := total_people * (chocolates_per_person / 2)
  let remaining_chocolates := chocolates_given_to_father - chocolates_to_mother - chocolates_eaten
  remaining_chocolates

theorem fathers_remaining_chocolates_eq_five :
  fathers_remaining_chocolates 20 4 3 2 = 5 := by
  sorry

end fathers_remaining_chocolates_fathers_remaining_chocolates_eq_five_l1094_109410


namespace orange_picking_theorem_l1094_109481

/-- The total number of oranges picked over three days -/
def totalOranges (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  day1 + day2 + day3

/-- Theorem stating the total number of oranges picked -/
theorem orange_picking_theorem :
  let day1 := 100
  let day2 := 3 * day1
  let day3 := 70
  totalOranges day1 day2 day3 = 470 := by
  sorry

end orange_picking_theorem_l1094_109481


namespace inequality_proof_l1094_109498

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1/(2*a) + 1/(2*b) + 1/(2*c) ≥ 1/(b+c) + 1/(c+a) + 1/(a+b) := by
  sorry

end inequality_proof_l1094_109498


namespace pin_pierces_all_sheets_l1094_109499

/-- Represents a rectangular sheet of paper -/
structure Sheet :=
  (width : ℝ)
  (height : ℝ)
  (center : ℝ × ℝ)

/-- Represents a collection of sheets on a table -/
structure TableSetup :=
  (sheets : List Sheet)
  (top_sheet : Sheet)

/-- Predicate to check if a point is on a sheet -/
def point_on_sheet (p : ℝ × ℝ) (s : Sheet) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  |x - cx| ≤ s.width / 2 ∧ |y - cy| ≤ s.height / 2

/-- The main theorem -/
theorem pin_pierces_all_sheets (setup : TableSetup) 
  (h_identical : ∀ s ∈ setup.sheets, s = setup.top_sheet)
  (h_cover : ∀ s ∈ setup.sheets, s ≠ setup.top_sheet → 
    (Set.inter (Set.range (point_on_sheet · setup.top_sheet)) 
               (Set.range (point_on_sheet · s))).ncard > 
    (Set.range (point_on_sheet · s)).ncard / 2) :
  ∃ p : ℝ × ℝ, ∀ s ∈ setup.sheets, point_on_sheet p s :=
sorry

end pin_pierces_all_sheets_l1094_109499


namespace down_payment_calculation_l1094_109459

theorem down_payment_calculation (purchase_price : ℝ) (monthly_payment : ℝ) 
  (num_payments : ℕ) (interest_rate : ℝ) (down_payment : ℝ) : 
  purchase_price = 130 ∧ 
  monthly_payment = 10 ∧ 
  num_payments = 12 ∧ 
  interest_rate = 0.23076923076923077 →
  down_payment = purchase_price + interest_rate * purchase_price - num_payments * monthly_payment :=
by
  sorry

end down_payment_calculation_l1094_109459


namespace arithmetic_sequence_second_term_l1094_109495

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 10th term is 20 and the 11th term is 24,
    the 2nd term of the sequence is -12. -/
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_10th : a 10 = 20)
  (h_11th : a 11 = 24) :
  a 2 = -12 :=
by sorry

end arithmetic_sequence_second_term_l1094_109495


namespace abc_over_def_value_l1094_109452

theorem abc_over_def_value (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 10)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0) : 
  a * b * c / (d * e * f) = 1 / 10 := by
  sorry

end abc_over_def_value_l1094_109452


namespace rope_percentage_theorem_l1094_109421

theorem rope_percentage_theorem (total_length used_length : ℝ) 
  (h1 : total_length = 20)
  (h2 : used_length = 15) :
  used_length / total_length = 0.75 ∧ (1 - used_length / total_length) = 0.25 := by
  sorry

end rope_percentage_theorem_l1094_109421


namespace first_discount_is_twenty_percent_l1094_109405

/-- Proves that the first discount is 20% given the conditions of the problem -/
theorem first_discount_is_twenty_percent
  (list_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5)
  : ∃ (first_discount : ℝ),
    first_discount = 20 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end first_discount_is_twenty_percent_l1094_109405


namespace tan_22_5_deg_representation_l1094_109469

theorem tan_22_5_deg_representation :
  ∃ (a b c d : ℕ+), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) ∧
    (a + b + c + d = 10) := by
  sorry

end tan_22_5_deg_representation_l1094_109469


namespace percentage_relationship_l1094_109426

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.375)) :
  y = x * 1.6 :=
sorry

end percentage_relationship_l1094_109426


namespace prime_divisor_count_l1094_109407

theorem prime_divisor_count (p : ℕ) (hp : Prime p) 
  (h : ∃ k : ℤ, (28^p - 1 : ℤ) = k * (2*p^2 + 2*p + 1)) : 
  Prime (2*p^2 + 2*p + 1) := by
sorry

end prime_divisor_count_l1094_109407


namespace father_son_age_difference_l1094_109403

theorem father_son_age_difference :
  ∀ (f s : ℕ+),
  f * s = 2015 →
  f > s →
  f - s = 34 :=
by
  sorry

end father_son_age_difference_l1094_109403


namespace barbecue_sauce_ketchup_amount_l1094_109483

theorem barbecue_sauce_ketchup_amount :
  let total_sauce := k + 1 + 1
  let burger_sauce := (1 : ℚ) / 4
  let sandwich_sauce := (1 : ℚ) / 6
  let num_burgers := 8
  let num_sandwiches := 18
  ∀ k : ℚ,
  (num_burgers * burger_sauce + num_sandwiches * sandwich_sauce = total_sauce) →
  k = 3 := by
sorry

end barbecue_sauce_ketchup_amount_l1094_109483


namespace negation_equivalence_l1094_109468

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨ (Odd a ∧ Even b ∧ Odd c) ∨ (Odd a ∧ Odd b ∧ Even c)

def all_odd_or_at_least_two_even (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∨ (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ all_odd_or_at_least_two_even a b c := by sorry

end negation_equivalence_l1094_109468


namespace running_preference_related_to_gender_l1094_109442

/-- Represents the contingency table for students liking running --/
structure RunningPreference where
  total_students : Nat
  boys : Nat
  girls : Nat
  girls_like_running : Nat
  boys_dont_like_running : Nat

/-- Calculates the K^2 value for the contingency table --/
def calculate_k_squared (pref : RunningPreference) : Rat :=
  let boys_like_running := pref.boys - pref.boys_dont_like_running
  let girls_dont_like_running := pref.girls - pref.girls_like_running
  let N := pref.total_students
  let a := boys_like_running
  let b := pref.boys_dont_like_running
  let c := pref.girls_like_running
  let d := girls_dont_like_running
  (N * (a * d - b * c)^2 : Rat) / ((a + c) * (b + d) * (a + b) * (c + d))

/-- Theorem stating that the K^2 value exceeds the critical value --/
theorem running_preference_related_to_gender (pref : RunningPreference) 
  (h1 : pref.total_students = 200)
  (h2 : pref.boys = 120)
  (h3 : pref.girls = 80)
  (h4 : pref.girls_like_running = 30)
  (h5 : pref.boys_dont_like_running = 50)
  (critical_value : Rat := 6635 / 1000) :
  calculate_k_squared pref > critical_value := by
  sorry

end running_preference_related_to_gender_l1094_109442


namespace namjoon_lowest_l1094_109430

def board_A : ℝ := 2.4
def board_B : ℝ := 3.2
def board_C : ℝ := 2.8

def eunji_height : ℝ := 8 * board_A
def namjoon_height : ℝ := 4 * board_B
def hoseok_height : ℝ := 5 * board_C

theorem namjoon_lowest : 
  namjoon_height < eunji_height ∧ namjoon_height < hoseok_height :=
by sorry

end namjoon_lowest_l1094_109430


namespace average_difference_l1094_109400

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (20 + 60 + x) / 3 + 5 → x = 25 := by
  sorry

end average_difference_l1094_109400


namespace quadratic_inequality_l1094_109431

/-- A quadratic polynomial with nonnegative coefficients -/
structure NonnegQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of a quadratic polynomial at a given point -/
def evalQuadratic (P : NonnegQuadratic) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- Theorem: For any quadratic polynomial with nonnegative coefficients and any real numbers x and y,
    the inequality P(xy)^2 ≤ P(x^2)P(y^2) holds -/
theorem quadratic_inequality (P : NonnegQuadratic) (x y : ℝ) :
    (evalQuadratic P (x * y))^2 ≤ (evalQuadratic P (x^2)) * (evalQuadratic P (y^2)) := by
  sorry

end quadratic_inequality_l1094_109431


namespace not_in_range_iff_b_in_interval_l1094_109487

theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + 5 ≠ -2) ↔ b ∈ Set.Ioo (-Real.sqrt 28) (Real.sqrt 28) := by
  sorry

end not_in_range_iff_b_in_interval_l1094_109487


namespace average_squares_first_11_even_numbers_l1094_109456

theorem average_squares_first_11_even_numbers :
  let first_11_even : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  let squares := first_11_even.map (λ x => x * x)
  let sum_squares := squares.sum
  let average := sum_squares / first_11_even.length
  average = 184 := by
sorry

end average_squares_first_11_even_numbers_l1094_109456


namespace hex_addition_l1094_109435

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C | D | E | F

/-- Converts a HexDigit to its decimal value --/
def hexToDecimal (h : HexDigit) : Nat :=
  match h with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Converts a list of HexDigits to its decimal value --/
def hexListToDecimal (l : List HexDigit) : Nat :=
  l.foldr (fun d acc => 16 * acc + hexToDecimal d) 0

/-- Theorem: The sum of 7A3₁₆ and 1F4₁₆ is equal to 997₁₆ --/
theorem hex_addition : 
  let a := [HexDigit.D7, HexDigit.A, HexDigit.D3]
  let b := [HexDigit.D1, HexDigit.F, HexDigit.D4]
  let result := [HexDigit.D9, HexDigit.D9, HexDigit.D7]
  hexListToDecimal a + hexListToDecimal b = hexListToDecimal result := by
  sorry


end hex_addition_l1094_109435


namespace complex_equation_solution_l1094_109455

theorem complex_equation_solution (z : ℂ) : 
  Complex.abs z + z = 2 + 4 * Complex.I → z = -3 + 4 * Complex.I := by
  sorry

end complex_equation_solution_l1094_109455


namespace equation_rewrite_and_product_l1094_109428

theorem equation_rewrite_and_product (a b x y : ℝ) (m n p : ℤ) :
  a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1) →
  ∃ m n p : ℤ, (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5 ∧ m*n*p = 8 :=
by sorry

end equation_rewrite_and_product_l1094_109428


namespace plane_q_satisfies_conditions_l1094_109458

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if a plane contains a line defined by the intersection of two other planes -/
def containsIntersectionLine (p : Plane) (p1 p2 : Plane) : Prop := sorry

/-- Calculates the distance from a plane to a point -/
def distanceToPoint (p : Plane) (pt : Point) : ℝ := sorry

/-- Checks if two planes are different -/
def areDifferentPlanes (p1 p2 : Plane) : Prop := sorry

/-- Calculates the greatest common divisor of four integers -/
def gcd4 (a b c d : ℤ) : ℕ := sorry

theorem plane_q_satisfies_conditions : 
  let p1 : Plane := { a := 2, b := -1, c := 3, d := -4 }
  let p2 : Plane := { a := 3, b := 2, c := -1, d := -6 }
  let q : Plane := { a := 0, b := -7, c := 11, d := -6 }
  let pt : Point := { x := 2, y := -2, z := 1 }
  containsIntersectionLine q p1 p2 ∧ 
  areDifferentPlanes q p1 ∧
  areDifferentPlanes q p2 ∧
  distanceToPoint q pt = 3 / Real.sqrt 5 ∧
  q.a > 0 ∧
  gcd4 (Int.natAbs q.a) (Int.natAbs q.b) (Int.natAbs q.c) (Int.natAbs q.d) = 1 := by
  sorry

end plane_q_satisfies_conditions_l1094_109458


namespace dihedral_angle_cosine_l1094_109474

/-- Given two spheres inscribed in a dihedral angle, this theorem proves that
    the cosine of the measure of the dihedral angle is 5/9 under specific conditions. -/
theorem dihedral_angle_cosine (α : Real) (R r : Real) (β : Real) :
  -- Two spheres are inscribed in a dihedral angle
  -- The spheres touch each other
  -- R is the radius of the larger sphere, r is the radius of the smaller sphere
  (R = 2 * r) →
  -- The line connecting the centers of the spheres forms a 45° angle with the edge of the dihedral angle
  (β = Real.pi / 4) →
  -- α is the measure of the dihedral angle
  -- The cosine of the measure of the dihedral angle is 5/9
  (Real.cos α = 5 / 9) :=
by sorry

end dihedral_angle_cosine_l1094_109474


namespace minimal_cost_proof_l1094_109439

/-- Represents an entity that can clean -/
inductive Cleaner
| Janitor
| Student
| Company

/-- Represents a location to be cleaned -/
inductive Location
| Classes
| Gym

/-- Time (in hours) it takes for a cleaner to clean a location -/
def cleaning_time (c : Cleaner) (l : Location) : ℕ :=
  match c, l with
  | Cleaner.Janitor, Location.Classes => 8
  | Cleaner.Janitor, Location.Gym => 6
  | Cleaner.Student, Location.Classes => 20
  | Cleaner.Student, Location.Gym => 0  -- Student cannot clean the gym
  | Cleaner.Company, Location.Classes => 10
  | Cleaner.Company, Location.Gym => 5

/-- Hourly rate (in dollars) for each cleaner -/
def hourly_rate (c : Cleaner) : ℕ :=
  match c with
  | Cleaner.Janitor => 21
  | Cleaner.Student => 7
  | Cleaner.Company => 60

/-- Cost for a cleaner to clean a location -/
def cleaning_cost (c : Cleaner) (l : Location) : ℕ :=
  (cleaning_time c l) * (hourly_rate c)

/-- The minimal cost to clean both the classes and the gym -/
def minimal_cleaning_cost : ℕ := 266

theorem minimal_cost_proof :
  ∀ (c1 c2 : Cleaner) (l1 l2 : Location),
    l1 ≠ l2 →
    cleaning_cost c1 l1 + cleaning_cost c2 l2 ≥ minimal_cleaning_cost :=
by sorry

end minimal_cost_proof_l1094_109439


namespace quarters_in_jar_l1094_109438

/-- Represents the number of coins of each type in the jar -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ
  two_dollar_coins : ℕ

/-- Represents the cost of a sundae and its modifications -/
structure SundaeCost where
  base : ℚ
  special_topping : ℚ
  featured_flavor : ℚ

/-- Represents the family's ice cream trip details -/
structure IceCreamTrip where
  family_size : ℕ
  special_toppings : ℕ
  featured_flavors : ℕ
  leftover : ℚ

def count_quarters (coins : CoinCounts) (sundae : SundaeCost) (trip : IceCreamTrip) : ℕ :=
  sorry

theorem quarters_in_jar 
  (coins : CoinCounts)
  (sundae : SundaeCost)
  (trip : IceCreamTrip) :
  coins.pennies = 123 ∧ 
  coins.nickels = 85 ∧ 
  coins.dimes = 35 ∧ 
  coins.half_dollars = 15 ∧ 
  coins.dollar_coins = 5 ∧ 
  coins.two_dollar_coins = 4 ∧
  sundae.base = 5.25 ∧
  sundae.special_topping = 0.5 ∧
  sundae.featured_flavor = 0.25 ∧
  trip.family_size = 8 ∧
  trip.special_toppings = 3 ∧
  trip.featured_flavors = 5 ∧
  trip.leftover = 0.97 →
  count_quarters coins sundae trip = 54 :=
by sorry

end quarters_in_jar_l1094_109438


namespace quadratic_real_roots_imply_a_equals_negative_one_l1094_109423

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℂ :=
  a * (1 + i) * x^2 + (1 + a^2 * i) * x + a^2 + i

-- Theorem statement
theorem quadratic_real_roots_imply_a_equals_negative_one :
  ∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0) → a = -1 := by
  sorry

end quadratic_real_roots_imply_a_equals_negative_one_l1094_109423


namespace simplify_expression_l1094_109453

theorem simplify_expression (y : ℝ) : (3/2 - 5*y) - (5/2 + 7*y) = -1 - 12*y := by
  sorry

end simplify_expression_l1094_109453


namespace loan_problem_l1094_109488

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- Theorem stating the conditions and conclusion of the loan problem -/
theorem loan_problem (loan : SimpleLoan) 
  (h1 : loan.time = loan.rate)
  (h2 : loan.interest = 108)
  (h3 : loan.rate = 0.03)
  (h4 : loan.interest = loan.principal * loan.rate * loan.time) :
  loan.principal = 1200 := by
  sorry

#check loan_problem

end loan_problem_l1094_109488


namespace algebraic_expression_equality_l1094_109443

theorem algebraic_expression_equality (x : ℝ) : 
  3 * x^2 - 4 * x = 6 → 6 * x^2 - 8 * x - 9 = 3 := by
  sorry

end algebraic_expression_equality_l1094_109443


namespace A_necessary_not_sufficient_l1094_109485

-- Define proposition A
def proposition_A (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 2 * a * x + 1 > 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem stating that A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∀ a, proposition_B a → proposition_A a) ∧
  ¬(∀ a, proposition_A a → proposition_B a) :=
sorry

end A_necessary_not_sufficient_l1094_109485


namespace units_digit_sum_product_l1094_109497

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_product : units_digit ((13 * 41) + (27 * 34)) = 1 := by
  sorry

end units_digit_sum_product_l1094_109497


namespace arithmetic_sequence_75th_term_l1094_109482

/-- Arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1 : ℤ) * d

/-- The 75th term of the arithmetic sequence starting with 3 and common difference 5 is 373 -/
theorem arithmetic_sequence_75th_term :
  arithmetic_sequence 3 5 75 = 373 := by
  sorry

end arithmetic_sequence_75th_term_l1094_109482


namespace stating_tour_days_correct_l1094_109486

/-- Represents the number of days Mr. Bhaskar is on tour -/
def tour_days : ℕ := 20

/-- Total budget for the tour -/
def total_budget : ℕ := 360

/-- Number of days the tour could be extended -/
def extension_days : ℕ := 4

/-- Amount by which daily expenses must be reduced if tour is extended -/
def expense_reduction : ℕ := 3

/-- 
Theorem stating that tour_days satisfies the given conditions:
1. The total budget divided by tour_days gives the daily expense
2. If the tour is extended by extension_days, the new daily expense is 
   reduced by expense_reduction
3. The total expenditure remains the same in both scenarios
-/
theorem tour_days_correct : 
  (total_budget / tour_days) * tour_days = 
  ((total_budget / tour_days) - expense_reduction) * (tour_days + extension_days) := by
  sorry

#check tour_days_correct

end stating_tour_days_correct_l1094_109486


namespace journey_speed_calculation_l1094_109461

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 540 →
  total_time = 15 →
  second_half_speed = 30 →
  (total_distance / 2) / ((total_time - (total_distance / 2) / second_half_speed)) = 45 :=
by sorry

end journey_speed_calculation_l1094_109461


namespace democrat_ratio_l1094_109480

/-- Prove that the ratio of democrats to total participants is 1:3 -/
theorem democrat_ratio (total : ℕ) (female_democrats : ℕ) :
  total = 780 →
  female_democrats = 130 →
  (∃ (female male : ℕ),
    female + male = total ∧
    female = 2 * female_democrats ∧
    4 * (female_democrats + male / 4) = total / 3) :=
by sorry

end democrat_ratio_l1094_109480


namespace min_value_and_inequality_range_l1094_109491

theorem min_value_and_inequality_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 3) :
  (∃ (min : ℝ), min = 6 ∧ ∀ x y, x > 0 → y > 0 → x * y = 3 → x + 3 * y ≥ min) ∧
  (∀ m : ℝ, (∀ x y, x > 0 → y > 0 → x * y = 3 → m^2 - (x + 3 * y) * m + 5 ≤ 0) → 1 ≤ m ∧ m ≤ 5) :=
by sorry

end min_value_and_inequality_range_l1094_109491


namespace smallest_lucky_number_l1094_109411

theorem smallest_lucky_number : 
  ∃ (a b c d : ℕ+), 
    (545 = a^2 + b^2 ∧ 545 = c^2 + d^2) ∧
    (a - c = 7 ∧ d - b = 13) ∧
    (∀ (N : ℕ) (a' b' c' d' : ℕ+), 
      (N < 545 → ¬(N = a'^2 + b'^2 ∧ N = c'^2 + d'^2 ∧ a' - c' = 7 ∧ d' - b' = 13))) := by
  sorry

#check smallest_lucky_number

end smallest_lucky_number_l1094_109411


namespace binomial_10_3_l1094_109479

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l1094_109479


namespace erroneous_multiplication_l1094_109422

/-- Given two positive integers where one is a two-digit number,
    if the product of the digit-reversed two-digit number and the other integer is 161,
    then the product of the original numbers is 224. -/
theorem erroneous_multiplication (a b : ℕ) : 
  a ≥ 10 ∧ a ≤ 99 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (10 * (a % 10) + (a / 10)) * b = 161 →  -- reversed a multiplied by b is 161
  a * b = 224 :=
by sorry

end erroneous_multiplication_l1094_109422


namespace expression_simplification_l1094_109414

theorem expression_simplification (x : ℝ) (h : x = 2 + Real.sqrt 3) :
  (x + 1) / (x^2 - 4) * ((1 / (x + 1)) + 1) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1094_109414


namespace count_negative_numbers_l1094_109444

def numbers : List ℝ := [-2.5, 7, -3, 2, 0, 4, 5, -1]

theorem count_negative_numbers : 
  (numbers.filter (· < 0)).length = 3 := by sorry

end count_negative_numbers_l1094_109444


namespace final_orange_count_l1094_109476

def initial_oranges : ℕ := 150

def sold_to_peter (n : ℕ) : ℕ := n - n * 20 / 100

def sold_to_paula (n : ℕ) : ℕ := n - n * 30 / 100

def give_to_neighbor (n : ℕ) : ℕ := n - 10

def give_to_teacher (n : ℕ) : ℕ := n - 1

theorem final_orange_count :
  give_to_teacher (give_to_neighbor (sold_to_paula (sold_to_peter initial_oranges))) = 73 := by
  sorry

end final_orange_count_l1094_109476


namespace initial_volumes_l1094_109460

/-- Represents the initial state and operations on three cubic containers --/
structure ContainerSystem where
  -- Capacities of the containers
  c₁ : ℝ
  c₂ : ℝ
  c₃ : ℝ
  -- Initial volumes of liquid
  v₁ : ℝ
  v₂ : ℝ
  v₃ : ℝ
  -- Ratio of capacities
  hCapRatio : c₂ = 8 * c₁ ∧ c₃ = 27 * c₁
  -- Ratio of initial volumes
  hVolRatio : v₂ = 2 * v₁ ∧ v₃ = 3 * v₁
  -- Total volume remains constant
  hTotalVol : ℝ
  hTotalVolDef : hTotalVol = v₁ + v₂ + v₃
  -- Volume transferred in final operation
  transferVol : ℝ
  hTransferVol : transferVol = 128 + 4/7
  -- Final state properties
  hFinalState : ∃ (h₁ h₂ : ℝ),
    h₁ > 0 ∧ h₂ > 0 ∧
    h₁ * c₁ + transferVol = v₁ - 100 ∧
    h₂ * c₂ - transferVol = v₂ ∧
    h₁ = 2 * h₂

/-- Theorem stating the initial volumes of liquid in the three containers --/
theorem initial_volumes (s : ContainerSystem) : 
  s.v₁ = 350 ∧ s.v₂ = 700 ∧ s.v₃ = 1050 := by
  sorry


end initial_volumes_l1094_109460


namespace solution_set_inequality_l1094_109457

theorem solution_set_inequality (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 := by
sorry

end solution_set_inequality_l1094_109457


namespace triangle_table_height_l1094_109473

theorem triangle_table_height (DE EF FD : ℝ) (hDE : DE = 20) (hEF : EF = 21) (hFD : FD = 29) :
  let s := (DE + EF + FD) / 2
  let A := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_d := 2 * A / FD
  let h_f := 2 * A / EF
  let k := (h_f * h_d) / (h_f + h_d)
  k = 7 * Real.sqrt 210 / 5 := by sorry

end triangle_table_height_l1094_109473


namespace cuboid_volume_l1094_109429

/-- The volume of a cuboid with given edge lengths -/
theorem cuboid_volume (l w h : ℝ) (hl : l = 3/2 + Real.sqrt (5/3)) 
  (hw : w = 2 + Real.sqrt (3/5)) (hh : h = π / 2) : 
  l * w * h = (3/2 + Real.sqrt (5/3)) * (2 + Real.sqrt (3/5)) * (π / 2) := by
  sorry

#check cuboid_volume

end cuboid_volume_l1094_109429


namespace max_value_of_function_l1094_109454

theorem max_value_of_function (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 ∧ (2 - x - 4 / x = -2 ↔ x = 2) := by
  sorry

end max_value_of_function_l1094_109454


namespace inequality_always_holds_l1094_109489

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) :
  a + c > b + c := by sorry

end inequality_always_holds_l1094_109489


namespace f_inequality_l1094_109424

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end f_inequality_l1094_109424


namespace A_intersection_Z_l1094_109450

def A : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersection_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {0, 1, 2} := by
  sorry

end A_intersection_Z_l1094_109450


namespace unique_plants_count_l1094_109447

/-- Represents a flower bed -/
structure FlowerBed where
  plants : ℕ

/-- Represents the overlap between two flower beds -/
structure Overlap where
  plants : ℕ

/-- Represents the overlap among three flower beds -/
structure TripleOverlap where
  plants : ℕ

/-- Calculates the total number of unique plants in three overlapping flower beds -/
def totalUniquePlants (x y z : FlowerBed) (xy yz xz : Overlap) (xyz : TripleOverlap) : ℕ :=
  x.plants + y.plants + z.plants - xy.plants - yz.plants - xz.plants + xyz.plants

/-- Theorem stating that the total number of unique plants is 1320 -/
theorem unique_plants_count :
  let x : FlowerBed := ⟨600⟩
  let y : FlowerBed := ⟨480⟩
  let z : FlowerBed := ⟨420⟩
  let xy : Overlap := ⟨60⟩
  let yz : Overlap := ⟨70⟩
  let xz : Overlap := ⟨80⟩
  let xyz : TripleOverlap := ⟨30⟩
  totalUniquePlants x y z xy yz xz xyz = 1320 := by
  sorry

end unique_plants_count_l1094_109447


namespace equal_values_l1094_109475

theorem equal_values (p q a b : ℝ) 
  (h1 : p + q = 1)
  (h2 : p * q ≠ 0)
  (h3 : (p / a) + (q / b) = 1 / (p * a + q * b)) :
  a = b := by sorry

end equal_values_l1094_109475


namespace simplify_expression_l1094_109493

theorem simplify_expression (a b : ℝ) :
  (33 * a + 75 * b + 12) + (15 * a + 44 * b + 7) - (12 * a + 65 * b + 5) = 36 * a + 54 * b + 14 := by
  sorry

end simplify_expression_l1094_109493


namespace simon_candy_count_l1094_109467

def candy_problem (initial_candies : ℕ) : Prop :=
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_remaining := day1_remaining - (day1_remaining / 2) - 5
  let day3_remaining := day2_remaining - ((3 * day2_remaining) / 4) - 6
  day3_remaining = 4

theorem simon_candy_count : 
  ∃ (x : ℕ), candy_problem x ∧ x = 124 :=
sorry

end simon_candy_count_l1094_109467


namespace quadratic_equations_solution_l1094_109451

theorem quadratic_equations_solution :
  -- Part I
  let eq1 : ℝ → Prop := λ x ↦ x^2 + 6*x + 5 = 0
  ∃ x1 x2 : ℝ, eq1 x1 ∧ eq1 x2 ∧ x1 = -5 ∧ x2 = -1 ∧
  -- Part II
  ∀ k : ℝ,
    let eq2 : ℝ → Prop := λ x ↦ x^2 - 3*x + k = 0
    (∃ x1 x2 : ℝ, eq2 x1 ∧ eq2 x2 ∧ (x1 - 1) * (x2 - 1) = -6) →
    k = -4 ∧ ∃ x1 x2 : ℝ, eq2 x1 ∧ eq2 x2 ∧ x1 = 4 ∧ x2 = -1 :=
by sorry

end quadratic_equations_solution_l1094_109451


namespace shifted_parabola_equation_l1094_109446

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the vertical shift amount
def shift_amount : ℝ := 4

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := original_parabola x - shift_amount

-- Theorem stating that the shifted parabola has the correct form
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = 2 * x^2 - 4 := by
  sorry

end shifted_parabola_equation_l1094_109446


namespace power_function_above_identity_l1094_109494

theorem power_function_above_identity {α : ℝ} :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^α > x) ↔ α < 1 :=
sorry

end power_function_above_identity_l1094_109494


namespace greatest_constant_inequality_l1094_109415

theorem greatest_constant_inequality (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

#check greatest_constant_inequality

end greatest_constant_inequality_l1094_109415


namespace shaded_area_between_circles_l1094_109417

theorem shaded_area_between_circles (r₁ r₂ chord_length : ℝ) 
  (h₁ : r₁ = 40)
  (h₂ : r₂ = 60)
  (h₃ : chord_length = 100)
  (h₄ : r₁ < r₂)
  (h₅ : chord_length^2 = 4 * (r₂^2 - r₁^2)) : -- Condition for tangency
  (π * r₂^2 - π * r₁^2) = 2000 * π :=
sorry

end shaded_area_between_circles_l1094_109417


namespace vehicle_original_value_l1094_109436

/-- The original value of a vehicle given its insurance details -/
def original_value (insured_fraction : ℚ) (premium : ℚ) (premium_rate : ℚ) : ℚ :=
  premium / (premium_rate / 100) / insured_fraction

/-- Theorem stating the original value of the vehicle -/
theorem vehicle_original_value :
  original_value (4/5) 910 1.3 = 87500 := by
  sorry

end vehicle_original_value_l1094_109436


namespace triangle_inequality_sum_l1094_109432

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (a + c) + c / (a + b) < 5 / 2 := by
  sorry

end triangle_inequality_sum_l1094_109432


namespace roots_sum_of_squares_l1094_109418

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 4*x + 4 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 8 := by
  sorry

end roots_sum_of_squares_l1094_109418


namespace inequality_proof_l1094_109409

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (a + 2*b + c)^2 / (2*b^2 + (c + a)^2) +
  (a + b + 2*c)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end inequality_proof_l1094_109409


namespace cosine_function_theorem_l1094_109412

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem cosine_function_theorem (f : ℝ → ℝ) (T : ℝ) :
  is_periodic f T →
  (∀ x, Real.cos x = f x - 2 * f (x - π)) →
  (∀ x, Real.cos x = f (x - T) - 2 * f (x - T - π)) →
  (∀ x, Real.cos x = Real.cos (x - T)) →
  (∀ x, f x = (1/3) * Real.cos x) :=
by sorry

end cosine_function_theorem_l1094_109412


namespace h_zero_iff_b_eq_two_l1094_109448

def h (x : ℝ) : ℝ := 5 * x - 10

theorem h_zero_iff_b_eq_two : ∀ b : ℝ, h b = 0 ↔ b = 2 := by sorry

end h_zero_iff_b_eq_two_l1094_109448


namespace gcd_1343_816_l1094_109490

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end gcd_1343_816_l1094_109490


namespace system_solution_l1094_109406

theorem system_solution :
  let x₁ : ℝ := (35 + Real.sqrt 1321) / 24
  let y₁ : ℝ := (-125 - 7 * Real.sqrt 1321) / 72
  let x₂ : ℝ := (35 - Real.sqrt 1321) / 24
  let y₂ : ℝ := (-125 + 7 * Real.sqrt 1321) / 72
  (7 * x₁ + 3 * y₁ = 5 ∧ 4 * x₁^2 + 5 * y₁ = 9) ∧
  (7 * x₂ + 3 * y₂ = 5 ∧ 4 * x₂^2 + 5 * y₂ = 9) :=
by sorry

end system_solution_l1094_109406


namespace similar_triangles_side_proportional_l1094_109419

/-- Two triangles are similar if their corresponding angles are equal -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_side_proportional 
  (G H I X Y Z : ℝ × ℝ) 
  (h_similar : similar_triangles {G, H, I} {X, Y, Z}) 
  (h_GH : dist G H = 8)
  (h_HI : dist H I = 20)
  (h_YZ : dist Y Z = 25) : 
  dist X Y = 80 := by sorry

end similar_triangles_side_proportional_l1094_109419


namespace reciprocal_of_negative_three_l1094_109465

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end reciprocal_of_negative_three_l1094_109465


namespace chef_cooked_seven_potatoes_l1094_109416

/-- Represents the cooking scenario of a chef with potatoes -/
structure PotatoCookingScenario where
  total_potatoes : ℕ
  cooking_time_per_potato : ℕ
  remaining_cooking_time : ℕ

/-- Calculates the number of potatoes already cooked -/
def potatoes_already_cooked (scenario : PotatoCookingScenario) : ℕ :=
  scenario.total_potatoes - (scenario.remaining_cooking_time / scenario.cooking_time_per_potato)

/-- Theorem stating that the chef has already cooked 7 potatoes -/
theorem chef_cooked_seven_potatoes (scenario : PotatoCookingScenario)
  (h1 : scenario.total_potatoes = 16)
  (h2 : scenario.cooking_time_per_potato = 5)
  (h3 : scenario.remaining_cooking_time = 45) :
  potatoes_already_cooked scenario = 7 := by
  sorry

#eval potatoes_already_cooked { total_potatoes := 16, cooking_time_per_potato := 5, remaining_cooking_time := 45 }

end chef_cooked_seven_potatoes_l1094_109416


namespace abs_p_minus_q_equals_five_l1094_109425

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 := by
  sorry

end abs_p_minus_q_equals_five_l1094_109425


namespace miss_evans_class_contribution_l1094_109472

/-- Calculates the original contribution amount for a class given the number of students,
    individual contribution after using class funds, and the amount of class funds used. -/
def originalContribution (numStudents : ℕ) (individualContribution : ℕ) (classFunds : ℕ) : ℕ :=
  numStudents * individualContribution + classFunds

/-- Proves that for Miss Evans' class, the original contribution amount was $90. -/
theorem miss_evans_class_contribution :
  originalContribution 19 4 14 = 90 := by
  sorry

end miss_evans_class_contribution_l1094_109472


namespace multiply_mixed_number_l1094_109427

theorem multiply_mixed_number : 8 * (11 + 1/4) = 90 := by
  sorry

end multiply_mixed_number_l1094_109427


namespace sine_angle_equality_l1094_109433

theorem sine_angle_equality (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (604 * π / 180) → n = -64 := by
  sorry

end sine_angle_equality_l1094_109433


namespace base7_3652_equals_base10_1360_l1094_109464

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- The theorem stating that 3652 in base 7 is equal to 1360 in base 10 --/
theorem base7_3652_equals_base10_1360 : base7ToBase10 3 6 5 2 = 1360 := by
  sorry

end base7_3652_equals_base10_1360_l1094_109464


namespace derivative_at_one_l1094_109477

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 12 := by sorry

end derivative_at_one_l1094_109477


namespace area_of_triangle_WRX_l1094_109437

-- Define the points
variable (W X Y Z P Q R : ℝ × ℝ)

-- Define the conditions
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry
def on_line (P A B : ℝ × ℝ) : Prop := sorry
def distance (A B : ℝ × ℝ) : ℝ := sorry
def intersect (A B C D E : ℝ × ℝ) : Prop := sorry
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_WRX 
  (h1 : is_rectangle W X Y Z)
  (h2 : distance W Z = 7)
  (h3 : distance X Y = 4)
  (h4 : on_line P Y Z)
  (h5 : on_line Q Y Z)
  (h6 : distance Y P = 2)
  (h7 : distance Q Z = 3)
  (h8 : intersect W P X Q R) :
  area_triangle W R X = 98/5 := by
  sorry

end area_of_triangle_WRX_l1094_109437


namespace line_and_circle_tangent_l1094_109404

-- Define the lines and circle
def l₁ (x y : ℝ) : Prop := 2 * x - y = 1
def l₂ (x y : ℝ) : Prop := x + 2 * y = 3
def l₃ (x y : ℝ) : Prop := x - y + 1 = 0
def C (x y a : ℝ) : Prop := (x - a)^2 + y^2 = 8

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Main theorem
theorem line_and_circle_tangent :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = P) →  -- P is the intersection of l₁ and l₂
  (∀ x y : ℝ, l x y → l₃ (x + 1) (y + 1)) →  -- l is perpendicular to l₃
  ∃ a : ℝ, a > 0 ∧
    (∀ x y : ℝ, l x y → 
      (∃ t : ℝ, C x y a ∧ 
        (∀ x' y', C x' y' a → (x - x')^2 + (y - y')^2 ≥ t^2) ∧
        (∃ x' y', C x' y' a ∧ (x - x')^2 + (y - y')^2 = t^2))) →
  (∀ x y : ℝ, l x y ↔ x + y - 2 = 0) ∧ a = 6 :=
sorry

end line_and_circle_tangent_l1094_109404


namespace abs_S_equals_512_l1094_109440

-- Define the complex number i
def i : ℂ := Complex.I

-- Define S
def S : ℂ := (1 + i)^17 - (1 - i)^17

-- Theorem statement
theorem abs_S_equals_512 : Complex.abs S = 512 := by sorry

end abs_S_equals_512_l1094_109440


namespace average_speed_two_hours_l1094_109441

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 95 → speed2 = 60 → (speed1 + speed2) / 2 = 77.5 := by
  sorry

end average_speed_two_hours_l1094_109441


namespace original_population_multiple_of_three_l1094_109420

theorem original_population_multiple_of_three (x y z : ℕ) 
  (h1 : y * y = x * x + 121)
  (h2 : z * z = y * y + 121) : 
  ∃ k : ℕ, x * x = 3 * k :=
sorry

end original_population_multiple_of_three_l1094_109420


namespace work_to_pump_liquid_l1094_109466

/-- Work required to pump liquid from a paraboloid cauldron -/
theorem work_to_pump_liquid (R H γ : ℝ) (h_R : R > 0) (h_H : H > 0) (h_γ : γ > 0) :
  ∃ (W : ℝ), W = 240 * π * H^3 * γ / 9810 ∧ W > 0 := by
  sorry

end work_to_pump_liquid_l1094_109466


namespace biased_coin_probability_l1094_109484

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1)
  (h3 : 5 * p * (1 - p)^4 = 10 * p^2 * (1 - p)^3)
  (h4 : 5 * p * (1 - p)^4 ≠ 0) :
  10 * p^3 * (1 - p)^2 = 40 / 243 := by
  sorry

end biased_coin_probability_l1094_109484


namespace probability_three_men_l1094_109471

/-- The probability of selecting 3 men out of 3 selections from a workshop with 7 men and 3 women -/
theorem probability_three_men (total : ℕ) (men : ℕ) (women : ℕ) (selections : ℕ) :
  total = men + women →
  total = 10 →
  men = 7 →
  women = 3 →
  selections = 3 →
  (men.choose selections : ℚ) / (total.choose selections) = 7 / 24 := by
  sorry

end probability_three_men_l1094_109471


namespace beshmi_investment_l1094_109408

theorem beshmi_investment (savings : ℝ) : 
  (1 / 5 : ℝ) * savings + 0.42 * savings + (savings - (1 / 5 : ℝ) * savings - 0.42 * savings) = savings
    → 0.42 * savings = 10500
    → savings - (1 / 5 : ℝ) * savings - 0.42 * savings = 9500 :=
by
  sorry

end beshmi_investment_l1094_109408


namespace S_tiles_integers_not_naturals_l1094_109478

def S : Set ℤ := {1, 3, 4, 6}

def tiles_integers (S : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃ s ∈ S, ∃ k : ℤ, n = s + 4 * k

def tiles_naturals (S : Set ℤ) : Prop :=
  ∀ n : ℕ, ∃ s ∈ S, ∃ k : ℤ, (n : ℤ) = s + 4 * k

theorem S_tiles_integers_not_naturals :
  tiles_integers S ∧ ¬tiles_naturals S := by sorry

end S_tiles_integers_not_naturals_l1094_109478


namespace rental_income_calculation_l1094_109463

theorem rental_income_calculation (total_units : ℕ) (occupancy_rate : ℚ) (monthly_rent : ℕ) :
  total_units = 100 →
  occupancy_rate = 3/4 →
  monthly_rent = 400 →
  (total_units : ℚ) * occupancy_rate * (monthly_rent : ℚ) * 12 = 360000 := by
  sorry

end rental_income_calculation_l1094_109463


namespace quadratic_inequality_solution_l1094_109445

/-- The solution set of the quadratic inequality ax^2 + 3x - 2 < 0 --/
def SolutionSet (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

/-- The quadratic function ax^2 + 3x - 2 --/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 3*x - 2

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, QuadraticFunction a x < 0 ↔ x ∈ SolutionSet a b) →
  a = -1 ∧ b = 2 :=
by sorry

end quadratic_inequality_solution_l1094_109445


namespace range_of_m_l1094_109470

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∀ y ∈ Set.Icc (-10) (-6), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
by sorry

end range_of_m_l1094_109470


namespace cricket_players_count_l1094_109401

/-- The number of cricket players in a games hour -/
def cricket_players (total_players hockey_players football_players softball_players : ℕ) : ℕ :=
  total_players - (hockey_players + football_players + softball_players)

/-- Theorem: There are 16 cricket players given the conditions -/
theorem cricket_players_count : cricket_players 59 12 18 13 = 16 := by
  sorry

end cricket_players_count_l1094_109401


namespace inequality_preservation_l1094_109402

theorem inequality_preservation (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end inequality_preservation_l1094_109402


namespace divisibility_implies_k_value_l1094_109496

/-- 
If x^2 + kx - 3 is divisible by (x - 1), then k = 2.
-/
theorem divisibility_implies_k_value (k : ℤ) : 
  (∀ x : ℤ, (x - 1) ∣ (x^2 + k*x - 3)) → k = 2 := by
  sorry

end divisibility_implies_k_value_l1094_109496
