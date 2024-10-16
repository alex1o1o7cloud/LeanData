import Mathlib

namespace NUMINAMATH_CALUDE_probability_four_of_each_color_l612_61212

-- Define the number of balls
def n : ℕ := 8

-- Define the probability of painting a ball black or white
def p : ℚ := 1/2

-- Define the number of ways to choose 4 balls out of 8
def ways_to_choose : ℕ := Nat.choose n (n/2)

-- Define the probability of one specific arrangement
def prob_one_arrangement : ℚ := p^n

-- Statement to prove
theorem probability_four_of_each_color :
  ways_to_choose * prob_one_arrangement = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_of_each_color_l612_61212


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l612_61205

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧ 
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l612_61205


namespace NUMINAMATH_CALUDE_ellipse_equation_l612_61295

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The condition that an ellipse passes through a point -/
def passes_through (e : Ellipse) (p : Point) : Prop :=
  e.equation p.x p.y

/-- The condition that two foci and one endpoint of the minor axis form an isosceles right triangle -/
def isosceles_right_triangle (e : Ellipse) : Prop :=
  e.a = Real.sqrt 2 * e.b

theorem ellipse_equation (e : Ellipse) (p : Point) 
    (h1 : passes_through e p)
    (h2 : p.x = 1 ∧ p.y = Real.sqrt 2 / 2)
    (h3 : isosceles_right_triangle e) :
    ∀ x y : ℝ, e.equation x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l612_61295


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l612_61203

/-- Given a glucose solution where 500 cubic centimeters contain 10 grams of glucose,
    this theorem proves that the volume of solution containing 20 grams of glucose
    is 1000 cubic centimeters. -/
theorem glucose_solution_volume :
  let volume_500cc : ℝ := 500
  let glucose_500cc : ℝ := 10
  let glucose_target : ℝ := 20
  let volume_target : ℝ := (glucose_target * volume_500cc) / glucose_500cc
  volume_target = 1000 := by
  sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l612_61203


namespace NUMINAMATH_CALUDE_tv_cost_l612_61289

theorem tv_cost (original_savings : ℝ) (furniture_fraction : ℚ) (tv_cost : ℝ) : 
  original_savings = 3000.0000000000005 →
  furniture_fraction = 5/6 →
  tv_cost = original_savings * (1 - furniture_fraction) →
  tv_cost = 500.0000000000001 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l612_61289


namespace NUMINAMATH_CALUDE_preimage_of_20_l612_61291

def f (n : ℕ) : ℕ := 2^n + n

theorem preimage_of_20 : ∃! n : ℕ, f n = 20 ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_preimage_of_20_l612_61291


namespace NUMINAMATH_CALUDE_five_wednesdays_theorem_l612_61240

/-- The year of the Gregorian calendar reform -/
def gregorian_reform_year : ℕ := 1752

/-- The cycle length for years with 5 Wednesdays in February -/
def cycle_length : ℕ := 28

/-- The reference year with 5 Wednesdays in February -/
def reference_year : ℕ := 1928

/-- Predicate to check if a year has 5 Wednesdays in February -/
def has_five_wednesdays (year : ℕ) : Prop :=
  (year ≥ gregorian_reform_year) ∧ 
  (year = reference_year ∨ (year - reference_year) % cycle_length = 0)

/-- The nearest year before the reference year with 5 Wednesdays in February -/
def nearest_before : ℕ := 1888

/-- The nearest year after the reference year with 5 Wednesdays in February -/
def nearest_after : ℕ := 1956

theorem five_wednesdays_theorem :
  (has_five_wednesdays nearest_before) ∧
  (has_five_wednesdays nearest_after) ∧
  (∀ y : ℕ, nearest_before < y ∧ y < reference_year → ¬(has_five_wednesdays y)) ∧
  (∀ y : ℕ, reference_year < y ∧ y < nearest_after → ¬(has_five_wednesdays y)) :=
sorry

end NUMINAMATH_CALUDE_five_wednesdays_theorem_l612_61240


namespace NUMINAMATH_CALUDE_unique_g_two_num_solutions_sum_solutions_final_result_l612_61245

/-- A function satisfying the given property for all real x and y -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y

/-- The main theorem stating that g(2) = 3 is the only solution -/
theorem unique_g_two (g : ℝ → ℝ) (h : SatisfiesProperty g) : g 2 = 3 := by
  sorry

/-- The number of possible values for g(2) is 1 -/
theorem num_solutions (g : ℝ → ℝ) (h : SatisfiesProperty g) : 
  ∃! x : ℝ, g 2 = x := by
  sorry

/-- The sum of all possible values of g(2) is 3 -/
theorem sum_solutions (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  ∃ x : ℝ, g 2 = x ∧ x = 3 := by
  sorry

/-- The product of the number of solutions and their sum is 3 -/
theorem final_result (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  (∃! x : ℝ, g 2 = x) ∧ (∃ x : ℝ, g 2 = x ∧ x = 3) → 1 * 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_g_two_num_solutions_sum_solutions_final_result_l612_61245


namespace NUMINAMATH_CALUDE_initial_blue_balls_l612_61225

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 15 → 
  removed = 3 → 
  prob = 1/3 → 
  (total - removed : ℚ) * prob = (total - removed - (total - removed - prob * (total - removed))) → 
  total - removed - (total - removed - prob * (total - removed)) + removed = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l612_61225


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l612_61202

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def power_of_two (n : ℕ) : ℕ := 2^n

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def sum_powers_of_two (n : ℕ) : ℕ := (List.range n).map power_of_two |>.sum

theorem units_digit_of_sum (n : ℕ) : 
  (sum_factorials n + sum_powers_of_two n) % 10 = 9 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l612_61202


namespace NUMINAMATH_CALUDE_product_divisible_by_49_l612_61259

theorem product_divisible_by_49 (a b : ℕ) (h : 7 ∣ (a^2 + b^2)) : 49 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_49_l612_61259


namespace NUMINAMATH_CALUDE_triangle_inequality_l612_61214

/-- Given a triangle with side lengths a, b, and c, the expression
    a^2 b(a-b) + b^2 c(b-c) + c^2 a(c-a) is non-negative,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l612_61214


namespace NUMINAMATH_CALUDE_valid_triples_are_solutions_l612_61287

def is_valid_triple (x y z : ℕ+) : Prop :=
  ∃ (n : ℤ), (Real.sqrt (2005 / (x + y : ℝ)) + 
              Real.sqrt (2005 / (y + z : ℝ)) + 
              Real.sqrt (2005 / (z + x : ℝ))) = n

def is_solution_triple (x y z : ℕ+) : Prop :=
  (x = 2005 * 2 ∧ y = 2005 * 2 ∧ z = 2005 * 14) ∨
  (x = 2005 * 2 ∧ y = 2005 * 14 ∧ z = 2005 * 2) ∨
  (x = 2005 * 14 ∧ y = 2005 * 2 ∧ z = 2005 * 2)

theorem valid_triples_are_solutions (x y z : ℕ+) :
  is_valid_triple x y z ↔ is_solution_triple x y z := by
  sorry

end NUMINAMATH_CALUDE_valid_triples_are_solutions_l612_61287


namespace NUMINAMATH_CALUDE_no_cracked_seashells_l612_61254

theorem no_cracked_seashells 
  (tom_initial : ℕ) 
  (fred_initial : ℕ) 
  (fred_more_than_tom : ℕ) 
  (h1 : tom_initial = 15)
  (h2 : fred_initial = 43)
  (h3 : fred_more_than_tom = 28)
  : ∃ (tom_final fred_final : ℕ),
    tom_initial + fred_initial = tom_final + fred_final ∧
    fred_final = tom_final + fred_more_than_tom ∧
    tom_initial - tom_final = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_cracked_seashells_l612_61254


namespace NUMINAMATH_CALUDE_min_participants_l612_61255

/-- Represents a single-round robin tournament --/
structure Tournament where
  participants : ℕ
  matches_per_player : ℕ
  winner_wins : ℕ

/-- Conditions for the tournament --/
def valid_tournament (t : Tournament) : Prop :=
  t.participants > 1 ∧
  t.matches_per_player = t.participants - 1 ∧
  (t.winner_wins : ℝ) / t.matches_per_player > 0.68 ∧
  (t.winner_wins : ℝ) / t.matches_per_player < 0.69

/-- The theorem to be proved --/
theorem min_participants (t : Tournament) (h : valid_tournament t) :
  t.participants ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_min_participants_l612_61255


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l612_61294

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 2| > a) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l612_61294


namespace NUMINAMATH_CALUDE_other_metal_price_l612_61231

/-- Given the price of Metal A, the ratio of Metal A to another metal, and the cost of their alloy,
    this theorem proves the price of the other metal. -/
theorem other_metal_price
  (price_a : ℝ)
  (ratio : ℝ)
  (alloy_cost : ℝ)
  (h1 : price_a = 68)
  (h2 : ratio = 3)
  (h3 : alloy_cost = 75) :
  (4 * alloy_cost - 3 * price_a) = 96 := by
  sorry

end NUMINAMATH_CALUDE_other_metal_price_l612_61231


namespace NUMINAMATH_CALUDE_heather_biking_days_l612_61226

/-- Given that Heather bicycled 40.0 kilometers per day for some days and 320 kilometers in total,
    prove that the number of days she biked is 8. -/
theorem heather_biking_days (daily_distance : ℝ) (total_distance : ℝ) 
    (h1 : daily_distance = 40.0)
    (h2 : total_distance = 320) :
    total_distance / daily_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_heather_biking_days_l612_61226


namespace NUMINAMATH_CALUDE_total_shells_l612_61223

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def shell_problem (counts : ShellCounts) : Prop :=
  counts.david = 15 ∧
  counts.mia = 4 * counts.david ∧
  counts.ava = counts.mia + 20 ∧
  counts.alice = counts.ava / 2

/-- The theorem to prove -/
theorem total_shells (counts : ShellCounts) : 
  shell_problem counts → counts.david + counts.mia + counts.ava + counts.alice = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l612_61223


namespace NUMINAMATH_CALUDE_prob_select_boy_is_correct_l612_61278

/-- Represents the number of boys in the calligraphy group -/
def calligraphy_boys : ℕ := 6

/-- Represents the number of girls in the calligraphy group -/
def calligraphy_girls : ℕ := 4

/-- Represents the number of boys in the original art group -/
def art_boys : ℕ := 5

/-- Represents the number of girls in the original art group -/
def art_girls : ℕ := 5

/-- Represents the number of people selected from the calligraphy group -/
def selected_from_calligraphy : ℕ := 2

/-- Calculates the probability of selecting a boy from the new art group -/
def prob_select_boy : ℚ := 31/60

theorem prob_select_boy_is_correct :
  prob_select_boy = 31/60 := by sorry

end NUMINAMATH_CALUDE_prob_select_boy_is_correct_l612_61278


namespace NUMINAMATH_CALUDE_equality_of_products_l612_61249

theorem equality_of_products (a b c d x y z q : ℝ) 
  (h1 : a ^ x = c ^ q) (h2 : c ^ q = b) 
  (h3 : c ^ y = a ^ z) (h4 : a ^ z = d) : 
  x * y = q * z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_products_l612_61249


namespace NUMINAMATH_CALUDE_triangle_area_l612_61213

/-- Given a triangle ABC where BC = 10 cm and the height from A to BC is 12 cm,
    prove that the area of triangle ABC is 60 square centimeters. -/
theorem triangle_area (BC height : ℝ) (h1 : BC = 10) (h2 : height = 12) :
  (1 / 2) * BC * height = 60 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l612_61213


namespace NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l612_61234

/-- The number of permits Andrew stamps in a day given his schedule and stamping rate -/
def permits_stamped (appointments : ℕ) (appointment_duration : ℕ) (workday_hours : ℕ) (stamping_rate : ℕ) : ℕ :=
  let total_appointment_hours := appointments * appointment_duration
  let stamping_hours := workday_hours - total_appointment_hours
  stamping_hours * stamping_rate

/-- Theorem stating that Andrew stamps 100 permits given his specific schedule and stamping rate -/
theorem andrew_stamps_hundred_permits :
  permits_stamped 2 3 8 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l612_61234


namespace NUMINAMATH_CALUDE_change_in_quadratic_expression_l612_61268

theorem change_in_quadratic_expression (x b : ℝ) (h : b > 0) :
  let f := fun x => 2 * x^2 + 5
  let change_plus := f (x + b) - f x
  let change_minus := f (x - b) - f x
  change_plus = 4 * x * b + 2 * b^2 ∧ change_minus = -4 * x * b + 2 * b^2 :=
by sorry

end NUMINAMATH_CALUDE_change_in_quadratic_expression_l612_61268


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l612_61277

/-- Given an angle α whose terminal side passes through the point P(-4m, 3m) where m < 0,
    prove that 2sin(α) + cos(α) = -2/5 -/
theorem angle_terminal_side_point (m : ℝ) (α : ℝ) (h1 : m < 0) 
  (h2 : Real.cos α = 4 * m / (5 * abs m)) (h3 : Real.sin α = 3 * m / (5 * abs m)) :
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l612_61277


namespace NUMINAMATH_CALUDE_transaction_period_is_one_year_l612_61216

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  principal : ℝ
  borrow_rate : ℝ
  lend_rate : ℝ
  gain_per_year : ℝ

/-- Calculates the number of years for the transaction -/
def transaction_years (t : Transaction) : ℝ :=
  1

/-- Theorem stating that the transaction period is 1 year -/
theorem transaction_period_is_one_year (t : Transaction) 
  (h1 : t.principal = 5000)
  (h2 : t.borrow_rate = 0.04)
  (h3 : t.lend_rate = 0.08)
  (h4 : t.gain_per_year = 200) :
  transaction_years t = 1 := by
  sorry

end NUMINAMATH_CALUDE_transaction_period_is_one_year_l612_61216


namespace NUMINAMATH_CALUDE_good_sets_exist_l612_61279

-- Define a "good" subset of natural numbers
def is_good (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (∃! p : ℕ, Prime p ∧ p ∣ n ∧ (n - p) ∈ A)

-- Define the set of perfect squares
def perfect_squares : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k^2}

-- Define the set of prime numbers
def prime_set : Set ℕ := {p : ℕ | Prime p}

theorem good_sets_exist :
  (is_good perfect_squares) ∧ 
  (is_good prime_set) ∧ 
  (Set.Infinite prime_set) ∧ 
  (perfect_squares ∩ prime_set = ∅) := by
  sorry

end NUMINAMATH_CALUDE_good_sets_exist_l612_61279


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_implies_a_eq_neg_two_l612_61242

/-- A system of two linear equations in two variables -/
structure LinearSystem (a : ℝ) where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  h1 : ∀ x y, eq1 x y = 2*x + 2*y + 1
  h2 : ∀ x y, eq2 x y = 4*x + a^2*y - a

/-- The system has infinitely many solutions -/
def HasInfinitelySolutions (sys : LinearSystem a) : Prop :=
  ∃ x₀ y₀, ∀ t : ℝ, sys.eq1 (x₀ + t) (y₀ - t) = 0 ∧ sys.eq2 (x₀ + t) (y₀ - t) = 0

/-- When the system has infinitely many solutions, a = -2 -/
theorem infinitely_many_solutions_implies_a_eq_neg_two (a : ℝ) (sys : LinearSystem a) :
  HasInfinitelySolutions sys → a = -2 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_implies_a_eq_neg_two_l612_61242


namespace NUMINAMATH_CALUDE_thanksgiving_to_christmas_l612_61290

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by a given number of days
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDays d m)

theorem thanksgiving_to_christmas (thanksgiving : DayOfWeek) :
  thanksgiving = DayOfWeek.Thursday →
  advanceDays thanksgiving 29 = DayOfWeek.Friday :=
by sorry

#check thanksgiving_to_christmas

end NUMINAMATH_CALUDE_thanksgiving_to_christmas_l612_61290


namespace NUMINAMATH_CALUDE_find_B_l612_61275

-- Define the polynomial g(x)
def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- State the theorem
theorem find_B :
  ∀ (A B C D : ℝ),
  (∀ x : ℝ, g A B C D x = 0 ↔ x = -2 ∨ x = 1 ∨ x = 2) →
  g A B C D 0 = -8 →
  B = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_find_B_l612_61275


namespace NUMINAMATH_CALUDE_domain_shift_l612_61208

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_shift (h : Set.Icc (-1 : ℝ) 1 = {x | ∃ y, f (y + 1) = x}) :
  {x | ∃ y, f y = x} = Set.Icc (-2 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_domain_shift_l612_61208


namespace NUMINAMATH_CALUDE_rhombus_side_length_l612_61257

-- Define the rhombus ABCD
def Rhombus (A B C D : Point) : Prop := sorry

-- Define the pyramid SABCD
def Pyramid (S A B C D : Point) : Prop := sorry

-- Define the inclination of lateral faces
def LateralFacesInclined (S A B C D : Point) (angle : ℝ) : Prop := sorry

-- Define midpoints
def Midpoint (M A B : Point) : Prop := sorry

-- Define the rectangular parallelepiped
def RectangularParallelepiped (M N K L F P R Q : Point) : Prop := sorry

-- Define the intersection points
def IntersectionPoints (S A B C D M N K L F P R Q : Point) : Prop := sorry

-- Define the volume of a polyhedron
def PolyhedronVolume (M N K L F P R Q : Point) : ℝ := sorry

-- Define the radius of an inscribed circle
def InscribedCircleRadius (A B C D : Point) : ℝ := sorry

-- Define the side length of a rhombus
def RhombusSideLength (A B C D : Point) : ℝ := sorry

theorem rhombus_side_length 
  (A B C D S M N K L F P R Q : Point) :
  Rhombus A B C D →
  Pyramid S A B C D →
  LateralFacesInclined S A B C D (60 * π / 180) →
  Midpoint M A B ∧ Midpoint N B C ∧ Midpoint K C D ∧ Midpoint L D A →
  RectangularParallelepiped M N K L F P R Q →
  IntersectionPoints S A B C D M N K L F P R Q →
  PolyhedronVolume M N K L F P R Q = 12 * Real.sqrt 3 →
  InscribedCircleRadius A B C D = 2.4 →
  RhombusSideLength A B C D = 5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l612_61257


namespace NUMINAMATH_CALUDE_reachability_l612_61233

/-- Number of positive integer divisors of n -/
def τ (n : ℕ) : ℕ := sorry

/-- Sum of positive integer divisors of n -/
def σ (n : ℕ) : ℕ := sorry

/-- Number of positive integers less than or equal to n that are relatively prime to n -/
def φ (n : ℕ) : ℕ := sorry

/-- Represents the operation of applying τ, σ, or φ -/
inductive Operation
| tau : Operation
| sigma : Operation
| phi : Operation

/-- Applies an operation to a natural number -/
def applyOperation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.tau => τ n
  | Operation.sigma => σ n
  | Operation.phi => φ n

/-- Theorem: For any two integers a and b greater than 1, 
    there exists a finite sequence of operations that transforms a into b -/
theorem reachability (a b : ℕ) (ha : a > 1) (hb : b > 1) : 
  ∃ (ops : List Operation), 
    (ops.foldl (fun n op => applyOperation op n) a) = b :=
sorry

end NUMINAMATH_CALUDE_reachability_l612_61233


namespace NUMINAMATH_CALUDE_trigonometric_equations_l612_61209

theorem trigonometric_equations (n m : ℤ) 
  (hn : -120 ≤ n ∧ n ≤ 120) (hm : -120 ≤ m ∧ m ≤ 120) :
  (Real.sin (n * π / 180) = Real.sin (580 * π / 180) → n = -40) ∧
  (Real.cos (m * π / 180) = Real.cos (300 * π / 180) → m = -60) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equations_l612_61209


namespace NUMINAMATH_CALUDE_problem_solution_l612_61292

-- Define the equation
def equation (m x : ℝ) : ℝ := x^2 + m*x + 2*m + 5

-- Define the set A
def set_A : Set ℝ := {m : ℝ | ∀ x : ℝ, equation m x ≠ 0 ∨ ∃ y : ℝ, y ≠ x ∧ equation m x = 0 ∧ equation m y = 0}

-- Define the set B
def set_B (a : ℝ) : Set ℝ := {x : ℝ | 1 - 2*a ≤ x ∧ x ≤ a - 1}

theorem problem_solution :
  (∀ m : ℝ, m ∈ set_A ↔ -2 ≤ m ∧ m ≤ 10) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ set_A → x ∈ set_B a) ∧ (∃ x : ℝ, x ∈ set_B a ∧ x ∉ set_A) ↔ 11 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l612_61292


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l612_61251

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a₁ : ℝ  -- First term
  d : ℝ   -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℝ := sorry

/-- Condition for symmetry of intersection points -/
def symmetricIntersectionPoints (seq : ArithmeticSequence) : Prop := sorry

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  symmetricIntersectionPoints seq →
  sumFirstNTerms seq n = -n^2 + 2*n := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l612_61251


namespace NUMINAMATH_CALUDE_product_of_specific_integers_l612_61273

theorem product_of_specific_integers : 
  ∃ (a b : ℤ), 
    a = 32 ∧ 
    b = 3125 ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    a * b = 100000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_integers_l612_61273


namespace NUMINAMATH_CALUDE_x_value_proof_l612_61270

theorem x_value_proof (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : 
  x = 134 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l612_61270


namespace NUMINAMATH_CALUDE_line_l_prime_equation_l612_61262

-- Define the fixed point P
def P : ℝ × ℝ := (-1, 1)

-- Define the direction vector of line l'
def direction_vector : ℝ × ℝ := (3, 2)

-- Define the equation of line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y + m = 0

-- State the theorem
theorem line_l_prime_equation :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), line_l m x y ∧ (x, y) = P) →
  (∃ (k : ℝ), 2 * P.1 - 3 * P.2 + 5 = 0 ∧
              ∀ (t : ℝ), 2 * (P.1 + t * direction_vector.1) - 3 * (P.2 + t * direction_vector.2) + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_l_prime_equation_l612_61262


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l612_61229

theorem min_distance_to_origin (x y : ℝ) : 
  (3 * x + y = 10) → (x^2 + y^2 ≥ 10) := by sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l612_61229


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l612_61228

theorem sum_of_three_numbers (a b c : ℤ) (N : ℚ) : 
  a + b + c = 80 ∧ 
  2 * a = N ∧ 
  b - 10 = N ∧ 
  3 * c = N → 
  N = 38 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l612_61228


namespace NUMINAMATH_CALUDE_min_value_theorem_l612_61200

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 3 * m + n = 1) :
  (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l612_61200


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l612_61266

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l612_61266


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_equation_l612_61237

/-- A quadrilateral inscribed in a semicircle -/
structure InscribedQuadrilateral where
  /-- The diameter of the semicircle -/
  x : ℝ
  /-- The length of side AM -/
  a : ℝ
  /-- The length of side MN -/
  b : ℝ
  /-- The length of side NB -/
  c : ℝ
  /-- x is positive (diameter) -/
  x_pos : 0 < x
  /-- a is positive (side length) -/
  a_pos : 0 < a
  /-- b is positive (side length) -/
  b_pos : 0 < b
  /-- c is positive (side length) -/
  c_pos : 0 < c
  /-- The sum of a, b, and c is less than or equal to x (semicircle property) -/
  sum_abc_le_x : a + b + c ≤ x

/-- The theorem stating the relationship between the sides of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_equation (q : InscribedQuadrilateral) :
  q.x^3 - (q.a^2 + q.b^2 + q.c^2) * q.x - 2 * q.a * q.b * q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_equation_l612_61237


namespace NUMINAMATH_CALUDE_inequality_proof_l612_61211

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) - 6 * a * b * c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l612_61211


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l612_61246

theorem largest_prime_factor_of_expression : 
  let expr := 15^4 + 2*15^2 + 1 - 14^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expr ∧ p = 211 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expr → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l612_61246


namespace NUMINAMATH_CALUDE_chocolate_candy_difference_l612_61282

-- Define the cost of chocolate and candy bar
def chocolate_cost : ℕ := 3
def candy_bar_cost : ℕ := 2

-- Theorem statement
theorem chocolate_candy_difference :
  chocolate_cost - candy_bar_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_candy_difference_l612_61282


namespace NUMINAMATH_CALUDE_election_votes_calculation_l612_61221

theorem election_votes_calculation (total_votes : ℕ) : 
  let valid_votes := (85 : ℚ) / 100 * total_votes
  let candidate_a_votes := (70 : ℚ) / 100 * valid_votes
  candidate_a_votes = 333200 →
  total_votes = 560000 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l612_61221


namespace NUMINAMATH_CALUDE_no_rabbits_perished_l612_61263

/-- Represents the farm with animals before and after the disease outbreak -/
structure Farm where
  initial_count : ℕ  -- Initial count of each animal type
  surviving_cows : ℕ
  surviving_pigs : ℕ
  surviving_horses : ℕ
  surviving_rabbits : ℕ

/-- The conditions of the farm after the disease outbreak -/
def farm_conditions (f : Farm) : Prop :=
  -- Initially equal number of each animal
  f.initial_count > 0 ∧
  -- One out of every five cows died
  f.surviving_cows = (4 * f.initial_count) / 5 ∧
  -- Number of horses that died equals number of pigs that survived
  f.surviving_horses = f.initial_count - f.surviving_pigs ∧
  -- Proportion of rabbits among survivors is 5/14
  14 * f.surviving_rabbits = 5 * (f.surviving_cows + f.surviving_pigs + f.surviving_horses + f.surviving_rabbits)

/-- The theorem to prove -/
theorem no_rabbits_perished (f : Farm) (h : farm_conditions f) : 
  f.surviving_rabbits = f.initial_count := by
  sorry

end NUMINAMATH_CALUDE_no_rabbits_perished_l612_61263


namespace NUMINAMATH_CALUDE_negation_of_exp_inequality_l612_61280

theorem negation_of_exp_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, x > 0 → Real.exp x ≥ 1) → 
  (¬p ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x < 1) :=
sorry

end NUMINAMATH_CALUDE_negation_of_exp_inequality_l612_61280


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l612_61244

theorem candy_box_price_increase (current_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) : 
  current_price = 10 ∧ 
  increase_percentage = 25 ∧ 
  current_price = original_price * (1 + increase_percentage / 100) →
  original_price = 8 := by
sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l612_61244


namespace NUMINAMATH_CALUDE_expression_value_l612_61236

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 2 * y) / (x - 2 * y) = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l612_61236


namespace NUMINAMATH_CALUDE_factor_expression_l612_61222

theorem factor_expression (m n x y : ℝ) : m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l612_61222


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_condition_l612_61288

/-- A triangle with sides a, b, and c is isosceles if it satisfies a^2 - bc = a(b - c) -/
theorem triangle_isosceles_from_condition (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_condition : a^2 - b*c = a*(b - c)) :
  a = b ∨ b = c ∨ c = a :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_from_condition_l612_61288


namespace NUMINAMATH_CALUDE_eleventh_term_value_l612_61283

/-- An arithmetic progression with specified properties -/
structure ArithmeticProgression where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 15 terms is 56.25
  sum_15_terms : (15 / 2 : ℝ) * (2 * a + 14 * d) = 56.25
  -- 7th term is 3.25
  term_7 : a + 6 * d = 3.25

/-- Theorem: The 11th term of the specified arithmetic progression is 5.25 -/
theorem eleventh_term_value (ap : ArithmeticProgression) : ap.a + 10 * ap.d = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_value_l612_61283


namespace NUMINAMATH_CALUDE_function_passes_through_point_l612_61260

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l612_61260


namespace NUMINAMATH_CALUDE_complement_union_A_B_l612_61201

open Set

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l612_61201


namespace NUMINAMATH_CALUDE_num_lines_eq_60_l612_61296

def coefficients : Finset ℕ := {1, 3, 5, 7, 9}

/-- The number of different lines formed by the equation Ax + By + C = 0,
    where A, B, and C are distinct elements from the set {1, 3, 5, 7, 9} -/
def num_lines : ℕ :=
  (coefficients.card) * (coefficients.card - 1) * (coefficients.card - 2)

theorem num_lines_eq_60 : num_lines = 60 := by
  sorry

end NUMINAMATH_CALUDE_num_lines_eq_60_l612_61296


namespace NUMINAMATH_CALUDE_least_integer_with_8_factors_l612_61218

/-- A function that counts the number of positive factors of a natural number -/
def count_factors (n : ℕ) : ℕ := sorry

/-- The property of being the least positive integer with exactly 8 factors -/
def is_least_with_8_factors (n : ℕ) : Prop :=
  count_factors n = 8 ∧ ∀ m : ℕ, m > 0 ∧ m < n → count_factors m ≠ 8

theorem least_integer_with_8_factors :
  is_least_with_8_factors 24 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_8_factors_l612_61218


namespace NUMINAMATH_CALUDE_point_b_satisfies_inequality_l612_61227

def satisfies_inequality (x y : ℝ) : Prop := x + 2 * y - 1 > 0

theorem point_b_satisfies_inequality :
  satisfies_inequality 0 1 ∧
  ¬ satisfies_inequality 1 (-1) ∧
  ¬ satisfies_inequality 1 0 ∧
  ¬ satisfies_inequality (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_point_b_satisfies_inequality_l612_61227


namespace NUMINAMATH_CALUDE_mixture_carbonated_water_percentage_l612_61252

theorem mixture_carbonated_water_percentage
  (solution1_lemonade : Real)
  (solution1_carbonated : Real)
  (solution2_lemonade : Real)
  (solution2_carbonated : Real)
  (mixture_ratio : Real)
  (h1 : solution1_lemonade = 0.2)
  (h2 : solution1_carbonated = 0.8)
  (h3 : solution2_lemonade = 0.45)
  (h4 : solution2_carbonated = 0.55)
  (h5 : mixture_ratio = 0.6799999999999997)
  (h6 : solution1_lemonade + solution1_carbonated = 1)
  (h7 : solution2_lemonade + solution2_carbonated = 1) :
  let total_carbonated := mixture_ratio * solution1_carbonated + (1 - mixture_ratio) * solution2_carbonated
  total_carbonated = 0.7199999999999999 := by sorry

end NUMINAMATH_CALUDE_mixture_carbonated_water_percentage_l612_61252


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l612_61298

def digits : Finset Nat := {1, 4, 7, 9}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

def largest_number : Nat := 97
def smallest_number : Nat := 14

theorem two_digit_number_difference :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n, is_valid_number n → n ≤ largest_number) ∧
  (∀ n, is_valid_number n → n ≥ smallest_number) ∧
  largest_number - smallest_number = 83 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l612_61298


namespace NUMINAMATH_CALUDE_no_uphill_integers_divisible_by_45_l612_61265

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < (Nat.digits 10 n).length →
    (Nat.digits 10 n).get ⟨i, by sorry⟩ < (Nat.digits 10 n).get ⟨j, by sorry⟩

/-- A number is divisible by 45 if and only if it is divisible by both 9 and 5. -/
def divisible_by_45 (n : ℕ) : Prop :=
  n % 45 = 0

theorem no_uphill_integers_divisible_by_45 :
  ¬ ∃ n : ℕ, is_uphill n ∧ divisible_by_45 n :=
sorry

end NUMINAMATH_CALUDE_no_uphill_integers_divisible_by_45_l612_61265


namespace NUMINAMATH_CALUDE_product_of_differences_l612_61232

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2023) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2022)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2023) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2022)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2023) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2022)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/2023 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l612_61232


namespace NUMINAMATH_CALUDE_curve_is_ellipse_iff_l612_61264

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k

/-- The condition for the curve to be a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -19

/-- Theorem stating that the curve is a non-degenerate ellipse iff k > -19 -/
theorem curve_is_ellipse_iff (x y k : ℝ) :
  (∀ x y, curve_equation x y k) ↔ is_non_degenerate_ellipse k :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_iff_l612_61264


namespace NUMINAMATH_CALUDE_new_members_weight_combined_weight_proof_l612_61220

/-- Calculates the combined weight of new members in a group replacement scenario. -/
theorem new_members_weight (group_size : ℕ) (original_avg : ℝ) (new_avg : ℝ)
  (replaced_weights : List ℝ) : ℝ :=
  let total_original := group_size * original_avg
  let total_replaced := replaced_weights.sum
  let remaining_weight := total_original - total_replaced
  let new_total := group_size * new_avg
  new_total - remaining_weight

/-- Proves that the combined weight of new members is 238 kg in the given scenario. -/
theorem combined_weight_proof :
  new_members_weight 8 70 76 [50, 65, 75] = 238 := by
  sorry

end NUMINAMATH_CALUDE_new_members_weight_combined_weight_proof_l612_61220


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l612_61267

/-- Given a rectangle DRAK with area 44, rectangle DUPE with area 64,
    and polygon DUPLAK with area 92, this theorem proves that there are
    only three possible sets of integer side lengths for the polygon. -/
theorem rectangle_side_lengths :
  ∀ (dr de du dk pl la : ℕ),
    dr * de = 16 →
    dr * dk = 44 →
    du * de = 64 →
    dk - de = la →
    du - dr = pl →
    (dr, de, du, dk, pl, la) ∈ ({(1, 16, 4, 44, 3, 28), (2, 8, 8, 22, 6, 14), (4, 4, 16, 11, 12, 7)} : Set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l612_61267


namespace NUMINAMATH_CALUDE_marie_message_clearing_l612_61281

/-- Calculate the number of days required to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  if read_per_day > new_per_day then
    (initial_messages + (read_per_day - new_per_day - 1)) / (read_per_day - new_per_day)
  else
    0

theorem marie_message_clearing :
  days_to_clear_messages 98 20 6 = 7 := by
sorry

end NUMINAMATH_CALUDE_marie_message_clearing_l612_61281


namespace NUMINAMATH_CALUDE_naoh_equals_agoh_l612_61276

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- The reaction between AgNO3 and NaOH to form AgOH -/
structure Reaction where
  agno3_initial : Moles
  agoh_formed : Moles
  naoh_combined : Moles

/-- The conditions of the reaction -/
class ReactionConditions (r : Reaction) where
  agno3_agoh_equal : r.agno3_initial = r.agoh_formed
  one_to_one_ratio : r.agoh_formed = r.naoh_combined

/-- Theorem stating that the number of moles of NaOH combined equals the number of moles of AgOH formed -/
theorem naoh_equals_agoh (r : Reaction) [ReactionConditions r] : r.naoh_combined = r.agoh_formed := by
  sorry

end NUMINAMATH_CALUDE_naoh_equals_agoh_l612_61276


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l612_61297

theorem simplify_and_evaluate (x y : ℤ) (A B : ℤ) (h1 : A = 2*x + y) (h2 : B = 2*x - y) (h3 : x = -1) (h4 : y = 2) :
  (A^2 - B^2) * (x - 2*y) = 80 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l612_61297


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l612_61239

/-- The distance between the center of a circle and a point in polar coordinates -/
theorem distance_circle_center_to_point 
  (ρ : ℝ → ℝ) -- Radius function for the circle
  (θ : ℝ) -- Angle parameter
  (r : ℝ) -- Radius of point D
  (φ : ℝ) -- Angle of point D
  (h1 : ∀ θ, ρ θ = 2 * Real.sin θ) -- Circle equation
  (h2 : r = 1) -- Radius of point D
  (h3 : φ = Real.pi) -- Angle of point D
  : Real.sqrt 2 = Real.sqrt ((0 - r * Real.cos φ)^2 + (1 - r * Real.sin φ)^2) :=
sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l612_61239


namespace NUMINAMATH_CALUDE_shaded_area_in_divided_square_l612_61271

/-- The area of shaded regions in a square with specific divisions -/
theorem shaded_area_in_divided_square (side_length : ℝ) (h_side : side_length = 4) :
  let square_area := side_length ^ 2
  let num_rectangles := 4
  let num_triangles_per_rectangle := 2
  let num_shaded_triangles := num_rectangles
  let rectangle_area := square_area / num_rectangles
  let triangle_area := rectangle_area / num_triangles_per_rectangle
  let total_shaded_area := num_shaded_triangles * triangle_area
  total_shaded_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_divided_square_l612_61271


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l612_61272

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 1456 [ZMOD 11]) → n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l612_61272


namespace NUMINAMATH_CALUDE_equivalence_condition_l612_61210

theorem equivalence_condition (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔
  (∃ a b c d : ℕ, x = a + 2*b + 3*c + 7*d ∧ y = b + 2*c + 5*d) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l612_61210


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l612_61274

theorem inequality_system_solution_set :
  let S := {x : ℝ | -3 * (x - 2) ≥ 4 - x ∧ (1 + 2 * x) / 3 > x - 1}
  S = {x : ℝ | x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l612_61274


namespace NUMINAMATH_CALUDE_expression_value_for_x_2_l612_61241

theorem expression_value_for_x_2 : 
  let x : ℝ := 2
  (x + x * (x * x)) = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_value_for_x_2_l612_61241


namespace NUMINAMATH_CALUDE_alpha_plus_beta_value_l612_61204

theorem alpha_plus_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sqrt 3 * (Real.cos (α/2))^2 + Real.sqrt 2 * (Real.sin (β/2))^2 = Real.sqrt 2 / 2 + Real.sqrt 3 / 2)
  (h4 : Real.sin (2017 * π - α) = Real.sqrt 2 * Real.cos (5 * π / 2 - β)) :
  α + β = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_value_l612_61204


namespace NUMINAMATH_CALUDE_cosine_product_equals_one_eighth_two_minus_sqrt_two_l612_61207

theorem cosine_product_equals_one_eighth_two_minus_sqrt_two :
  (1 + Real.cos (π / 9)) * (1 + Real.cos (4 * π / 9)) *
  (1 + Real.cos (5 * π / 9)) * (1 + Real.cos (8 * π / 9)) =
  1 / 8 * (2 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_cosine_product_equals_one_eighth_two_minus_sqrt_two_l612_61207


namespace NUMINAMATH_CALUDE_winning_game_score_is_3_0_l612_61299

structure FootballTeam where
  games_played : ℕ
  total_goals_scored : ℕ
  total_goals_conceded : ℕ
  wins : ℕ
  draws : ℕ
  losses : ℕ

def winning_game_score (team : FootballTeam) : ℕ × ℕ := sorry

theorem winning_game_score_is_3_0 (team : FootballTeam) 
  (h1 : team.games_played = 3)
  (h2 : team.total_goals_scored = 3)
  (h3 : team.total_goals_conceded = 1)
  (h4 : team.wins = 1)
  (h5 : team.draws = 1)
  (h6 : team.losses = 1) :
  winning_game_score team = (3, 0) := by sorry

end NUMINAMATH_CALUDE_winning_game_score_is_3_0_l612_61299


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l612_61293

open Real

noncomputable def y (x C₁ C₂ : ℝ) : ℝ :=
  C₁ * cos (2 * x) + C₂ * sin (2 * x) + (2 * cos (2 * x) + 8 * sin (2 * x)) * x + (1/2) * exp (2 * x)

theorem solution_satisfies_equation (x C₁ C₂ : ℝ) :
  (deriv^[2] (y C₁ C₂)) x + 4 * y C₁ C₂ x = -8 * sin (2 * x) + 32 * cos (2 * x) + 4 * exp (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l612_61293


namespace NUMINAMATH_CALUDE_shop_rent_per_square_foot_l612_61253

/-- Given a shop with dimensions 10 feet × 8 feet and a monthly rent of Rs. 2400,
    the annual rent per square foot is Rs. 360. -/
theorem shop_rent_per_square_foot :
  let length : ℕ := 10
  let width : ℕ := 8
  let monthly_rent : ℕ := 2400
  let area := length * width
  let annual_rent := monthly_rent * 12
  (annual_rent / area : ℚ) = 360 := by sorry

end NUMINAMATH_CALUDE_shop_rent_per_square_foot_l612_61253


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l612_61248

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ k : ℕ, k < 9 → ¬(11 ∣ (11002 + k))) ∧ (11 ∣ (11002 + 9)) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l612_61248


namespace NUMINAMATH_CALUDE_concert_attendance_difference_l612_61219

theorem concert_attendance_difference (first_concert : Nat) (second_concert : Nat)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_difference_l612_61219


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l612_61256

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x y : List ℝ) : ℝ := sorry

def r₁ : ℝ := linear_correlation_coefficient X Y
def r₂ : ℝ := linear_correlation_coefficient U V

theorem correlation_coefficient_relationship : r₂ < 0 ∧ 0 < r₁ := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l612_61256


namespace NUMINAMATH_CALUDE_product_of_radicals_l612_61269

theorem product_of_radicals (p : ℝ) (hp : p > 0) :
  Real.sqrt (42 * p) * Real.sqrt (14 * p) * Real.sqrt (7 * p) = 14 * p * Real.sqrt (21 * p) := by
  sorry

end NUMINAMATH_CALUDE_product_of_radicals_l612_61269


namespace NUMINAMATH_CALUDE_complex_equation_solution_l612_61235

theorem complex_equation_solution :
  ∃ (z : ℂ), 3 - 3 * Complex.I * z = -2 + 5 * Complex.I * z + (1 - 2 * Complex.I) ∧
             z = (1 / 4 : ℂ) - (3 / 8 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l612_61235


namespace NUMINAMATH_CALUDE_triple_q_2000_power_l612_61247

/-- Sum of digits function -/
def q (n : ℕ) : ℕ :=
  if n < 10 then n else q (n / 10) + n % 10

/-- Theorem: The triple application of q to 2000^2000 results in 4 -/
theorem triple_q_2000_power : q (q (q (2000^2000))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_triple_q_2000_power_l612_61247


namespace NUMINAMATH_CALUDE_fantasia_license_plates_l612_61250

/-- Represents the number of available letters in the alphabet. -/
def num_letters : ℕ := 26

/-- Represents the number of available digits. -/
def num_digits : ℕ := 10

/-- Calculates the number of valid license plates in Fantasia. -/
def count_license_plates : ℕ :=
  num_letters * num_letters * num_letters * num_digits * (num_digits - 1) * (num_digits - 2)

/-- Theorem stating that the number of valid license plates in Fantasia is 15,818,400. -/
theorem fantasia_license_plates :
  count_license_plates = 15818400 :=
by sorry

end NUMINAMATH_CALUDE_fantasia_license_plates_l612_61250


namespace NUMINAMATH_CALUDE_point_outside_circle_l612_61258

theorem point_outside_circle (m : ℝ) : 
  (1 : ℝ)^2 + (1 : ℝ)^2 + 4*m*1 - 2*1 + 5*m > 0 ∧ 
  ∃ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ 
  m > 1 ∨ (0 < m ∧ m < 1/4) := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l612_61258


namespace NUMINAMATH_CALUDE_smallest_natural_power_l612_61285

theorem smallest_natural_power (n : ℕ) : n^(Nat.zero) = 1 := by
  sorry

#check smallest_natural_power 2009

end NUMINAMATH_CALUDE_smallest_natural_power_l612_61285


namespace NUMINAMATH_CALUDE_simplify_fraction_l612_61238

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * y) / (-(x^2 * y)) = -2 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l612_61238


namespace NUMINAMATH_CALUDE_geometric_sequence_nth_term_l612_61261

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_nth_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_sum : a 2 + a 5 = 18)
  (h_prod : a 3 * a 4 = 32)
  (h_nth : ∃ (n : ℕ), a n = 128) :
  ∃ (n : ℕ), a n = 128 ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_nth_term_l612_61261


namespace NUMINAMATH_CALUDE_fill_time_calculation_l612_61284

/-- Represents the time to fill a leaky tank -/
def fill_time_with_leak : ℝ := 8

/-- Represents the time for the tank to empty due to the leak -/
def empty_time : ℝ := 56

/-- Represents the time to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 7

/-- Theorem stating that given the fill time with leak and empty time,
    the fill time without leak is 7 hours -/
theorem fill_time_calculation :
  (fill_time_with_leak * empty_time) / (empty_time - fill_time_with_leak) = fill_time_without_leak :=
sorry

end NUMINAMATH_CALUDE_fill_time_calculation_l612_61284


namespace NUMINAMATH_CALUDE_negation_of_proposition_l612_61217

theorem negation_of_proposition :
  (¬ ∀ (x y : ℝ), xy = 0 → x = 0) ↔ (∃ (x y : ℝ), xy = 0 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l612_61217


namespace NUMINAMATH_CALUDE_figure_can_form_square_l612_61230

/-- Represents a figure drawn on squared paper -/
structure Figure where
  -- Add necessary fields to represent the figure

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to cut a figure into triangles -/
def cut_into_triangles (f : Figure) : List Triangle :=
  sorry

/-- Function to check if a list of triangles can form a square -/
def can_form_square (triangles : List Triangle) : Bool :=
  sorry

/-- Theorem stating that the figure can be cut into 5 triangles that form a square -/
theorem figure_can_form_square (f : Figure) :
  ∃ (triangles : List Triangle), 
    cut_into_triangles f = triangles ∧ 
    triangles.length = 5 ∧ 
    can_form_square triangles = true :=
  sorry

end NUMINAMATH_CALUDE_figure_can_form_square_l612_61230


namespace NUMINAMATH_CALUDE_water_balloon_ratio_l612_61215

theorem water_balloon_ratio : ∀ (anthony_balloons luke_balloons tom_balloons : ℕ),
  anthony_balloons = 44 →
  luke_balloons = anthony_balloons / 4 →
  tom_balloons = 33 →
  (tom_balloons : ℚ) / luke_balloons = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_ratio_l612_61215


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l612_61286

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -3 ∧ x₂ = 5 ∧ 
  (x₁^2 - 2*x₁ - 15 = 0) ∧ 
  (x₂^2 - 2*x₂ - 15 = 0) ∧
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 → x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l612_61286


namespace NUMINAMATH_CALUDE_y_squared_value_l612_61206

theorem y_squared_value (y : ℝ) (h : Real.sqrt (y + 16) - Real.sqrt (y - 16) = 2) : 
  y^2 = 9216 := by
sorry

end NUMINAMATH_CALUDE_y_squared_value_l612_61206


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l612_61224

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 3/5) : 
  x / y = 16/15 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l612_61224


namespace NUMINAMATH_CALUDE_expected_additional_cases_l612_61243

/-- Proves the expected number of additional individuals with a condition in a sample -/
theorem expected_additional_cases
  (population_ratio : ℚ) -- Ratio of population with the condition
  (sample_size : ℕ) -- Size of the sample
  (known_cases : ℕ) -- Number of known cases in the sample
  (h1 : population_ratio = 1 / 4) -- Condition: 1/4 of population has the condition
  (h2 : sample_size = 300) -- Condition: Sample size is 300
  (h3 : known_cases = 20) -- Condition: 20 known cases in the sample
  : ℕ := by
  sorry

#check expected_additional_cases

end NUMINAMATH_CALUDE_expected_additional_cases_l612_61243
