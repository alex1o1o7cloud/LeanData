import Mathlib

namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2596_259649

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, n < 1000 → n ≥ 100 → n % 17 = 0 → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2596_259649


namespace NUMINAMATH_CALUDE_porter_monthly_earnings_l2596_259642

/-- Calculates the monthly earnings of a worker with overtime -/
def monthlyEarningsWithOvertime (dailyRate : ℕ) (regularDaysPerWeek : ℕ) (overtimeRatePercent : ℕ) (weeksInMonth : ℕ) : ℕ :=
  let regularWeeklyEarnings := dailyRate * regularDaysPerWeek
  let overtimeDailyRate := dailyRate * overtimeRatePercent / 100
  let overtimeWeeklyEarnings := dailyRate + overtimeDailyRate
  (regularWeeklyEarnings + overtimeWeeklyEarnings) * weeksInMonth

/-- Theorem stating that under given conditions, monthly earnings with overtime equal $208 -/
theorem porter_monthly_earnings :
  monthlyEarningsWithOvertime 8 5 150 4 = 208 := by
  sorry

#eval monthlyEarningsWithOvertime 8 5 150 4

end NUMINAMATH_CALUDE_porter_monthly_earnings_l2596_259642


namespace NUMINAMATH_CALUDE_sequence_property_l2596_259687

-- Define the sequence a_n
def a (n : ℕ) : ℚ := n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n)

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := (n^2 + n) / 2

-- Theorem statement
theorem sequence_property (n k : ℕ) (h1 : k > 2) :
  (∀ m : ℕ, S m = (m^2 + m) / 2) →
  (2 * b (n + 2) = b n + b (n + k)) →
  (k ≠ 4 ∧ k ≠ 10) ∧ (k = 6 ∨ k = 8) :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2596_259687


namespace NUMINAMATH_CALUDE_equation_solution_l2596_259695

theorem equation_solution : ∃ x : ℝ, (45 / 75 = Real.sqrt (x / 75) + 1 / 5) ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2596_259695


namespace NUMINAMATH_CALUDE_max_distance_is_1375_l2596_259694

/-- Represents the boat trip scenario -/
structure BoatTrip where
  totalTime : Real
  rowingTime : Real
  restTime : Real
  boatSpeed : Real
  currentSpeed : Real

/-- Calculates the maximum distance the boat can travel from the starting point -/
def maxDistance (trip : BoatTrip) : Real :=
  sorry

/-- Theorem stating that the maximum distance is 1.375 km for the given conditions -/
theorem max_distance_is_1375 :
  let trip : BoatTrip := {
    totalTime := 120,
    rowingTime := 30,
    restTime := 10,
    boatSpeed := 3,
    currentSpeed := 1.5
  }
  maxDistance trip = 1.375 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_is_1375_l2596_259694


namespace NUMINAMATH_CALUDE_fabric_difference_total_fabric_l2596_259609

/-- The amount of fabric used to make a coat, in meters -/
def coat_fabric : ℝ := 1.55

/-- The amount of fabric used to make a pair of pants, in meters -/
def pants_fabric : ℝ := 1.05

/-- The difference in fabric usage between a coat and pants is 0.5 meters -/
theorem fabric_difference : coat_fabric - pants_fabric = 0.5 := by sorry

/-- The total fabric needed for a coat and pants is 2.6 meters -/
theorem total_fabric : coat_fabric + pants_fabric = 2.6 := by sorry

end NUMINAMATH_CALUDE_fabric_difference_total_fabric_l2596_259609


namespace NUMINAMATH_CALUDE_min_transportation_cost_l2596_259630

/-- Represents the transportation problem with two production locations and two delivery venues -/
structure TransportationProblem where
  unitsJ : ℕ  -- Units produced in location J
  unitsY : ℕ  -- Units produced in location Y
  unitsA : ℕ  -- Units delivered to venue A
  unitsB : ℕ  -- Units delivered to venue B
  costJB : ℕ  -- Transportation cost from J to B per unit
  fixedCost : ℕ  -- Fixed overhead cost

/-- Calculates the total transportation cost given the number of units transported from J to A -/
def totalCost (p : TransportationProblem) (x : ℕ) : ℕ :=
  p.costJB * (p.unitsJ - x) + p.fixedCost

/-- Theorem stating the minimum transportation cost -/
theorem min_transportation_cost (p : TransportationProblem) 
    (h1 : p.unitsJ = 17) (h2 : p.unitsY = 15) (h3 : p.unitsA = 18) (h4 : p.unitsB = 14)
    (h5 : p.costJB = 200) (h6 : p.fixedCost = 19300) :
    ∃ (x : ℕ), x ≥ 3 ∧ 
    (∀ (y : ℕ), y ≥ 3 → totalCost p x ≤ totalCost p y) ∧
    totalCost p x = 19900 := by
  sorry

end NUMINAMATH_CALUDE_min_transportation_cost_l2596_259630


namespace NUMINAMATH_CALUDE_number_calculation_l2596_259669

theorem number_calculation (n : ℝ) : 0.125 * 0.20 * 0.40 * 0.75 * n = 148.5 → n = 23760 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2596_259669


namespace NUMINAMATH_CALUDE_half_triangles_isosceles_l2596_259617

/-- A function that returns the number of pairwise non-congruent triangles
    that can be formed from N points on a circle. -/
def totalTriangles (N : ℕ) : ℕ := N * (N - 1) * (N - 2) / 6

/-- A function that returns the number of isosceles triangles
    that can be formed from N points on a circle. -/
def isoscelesTriangles (N : ℕ) : ℕ := N * (N - 2) / 3

/-- The theorem stating that exactly half of the triangles are isosceles
    if and only if N is 10 or 11, for N > 2. -/
theorem half_triangles_isosceles (N : ℕ) (h : N > 2) :
  2 * isoscelesTriangles N = totalTriangles N ↔ N = 10 ∨ N = 11 :=
sorry

end NUMINAMATH_CALUDE_half_triangles_isosceles_l2596_259617


namespace NUMINAMATH_CALUDE_unit_complex_rational_power_minus_one_is_rational_l2596_259624

/-- A complex number with rational real and imaginary parts and modulus 1 -/
structure UnitComplexRational where
  re : ℚ
  im : ℚ
  norm_sq : re^2 + im^2 = 1

/-- The main theorem: z^(2n) - 1 is rational for any integer n -/
theorem unit_complex_rational_power_minus_one_is_rational
  (z : UnitComplexRational) (n : ℤ) :
  ∃ (q : ℚ), (z.re + z.im * Complex.I) ^ (2 * n) - 1 = q := by
  sorry

end NUMINAMATH_CALUDE_unit_complex_rational_power_minus_one_is_rational_l2596_259624


namespace NUMINAMATH_CALUDE_dasha_flag_count_l2596_259666

/-- Represents the number of flags held by each first-grader -/
structure FlagCount where
  tata : ℕ
  yasha : ℕ
  vera : ℕ
  maxim : ℕ
  dasha : ℕ

/-- The problem statement -/
def flag_problem (fc : FlagCount) : Prop :=
  fc.tata + fc.yasha + fc.vera + fc.maxim + fc.dasha = 37 ∧
  fc.yasha + fc.vera + fc.maxim + fc.dasha = 32 ∧
  fc.vera + fc.maxim + fc.dasha = 20 ∧
  fc.maxim + fc.dasha = 14 ∧
  fc.dasha = 8

/-- The theorem to prove -/
theorem dasha_flag_count :
  ∀ fc : FlagCount, flag_problem fc → fc.dasha = 8 := by
  sorry

end NUMINAMATH_CALUDE_dasha_flag_count_l2596_259666


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l2596_259664

/-- Given two cylinders A and B, where A's radius is r and height is h,
    and B's radius is h and height is r, prove that if A's volume is
    three times B's volume, then A's volume is 9πh^3. -/
theorem cylinder_volume_relation (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 3 * (π * h^2 * r) →
  π * r^2 * h = 9 * π * h^3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l2596_259664


namespace NUMINAMATH_CALUDE_inequality_preservation_l2596_259644

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2596_259644


namespace NUMINAMATH_CALUDE_some_number_value_l2596_259661

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = some_number * 35 * 45 * 35) : 
  some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2596_259661


namespace NUMINAMATH_CALUDE_quadrilateral_similarity_l2596_259612

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Add necessary fields and properties to define a convex quadrilateral
  -- This is a placeholder and should be expanded based on specific requirements

/-- Construct a new quadrilateral from the given one using perpendicular bisectors -/
def constructNextQuadrilateral (Q : ConvexQuadrilateral) : ConvexQuadrilateral :=
  sorry  -- Definition of the construction process

/-- Two quadrilaterals are similar -/
def isSimilar (Q1 Q2 : ConvexQuadrilateral) : Prop :=
  sorry  -- Definition of similarity for quadrilaterals

theorem quadrilateral_similarity (Q1 : ConvexQuadrilateral) :
  let Q2 := constructNextQuadrilateral Q1
  let Q3 := constructNextQuadrilateral Q2
  isSimilar Q3 Q1 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_similarity_l2596_259612


namespace NUMINAMATH_CALUDE_hurricane_damage_conversion_l2596_259651

def damage_in_euros : ℝ := 45000000
def exchange_rate : ℝ := 0.9

theorem hurricane_damage_conversion :
  damage_in_euros * (1 / exchange_rate) = 49995000 := by
  sorry

end NUMINAMATH_CALUDE_hurricane_damage_conversion_l2596_259651


namespace NUMINAMATH_CALUDE_robin_bobbin_chickens_l2596_259653

def chickens_eaten_sept_1 (chickens_sept_2 chickens_total_sept_15 : ℕ) : ℕ :=
  let avg_daily_consumption := chickens_total_sept_15 / 15
  let chickens_sept_1_and_2 := 2 * avg_daily_consumption
  chickens_sept_1_and_2 - chickens_sept_2

theorem robin_bobbin_chickens :
  chickens_eaten_sept_1 12 32 = 52 :=
sorry

end NUMINAMATH_CALUDE_robin_bobbin_chickens_l2596_259653


namespace NUMINAMATH_CALUDE_xy_squared_l2596_259658

theorem xy_squared (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + x + y = 7) : 
  x^2 * y^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_l2596_259658


namespace NUMINAMATH_CALUDE_course_selection_count_l2596_259605

def num_courses_A : ℕ := 3
def num_courses_B : ℕ := 4
def total_courses_selected : ℕ := 3

theorem course_selection_count : 
  (Nat.choose num_courses_A 2 * Nat.choose num_courses_B 1) + 
  (Nat.choose num_courses_A 1 * Nat.choose num_courses_B 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_count_l2596_259605


namespace NUMINAMATH_CALUDE_prime_divides_or_coprime_l2596_259650

theorem prime_divides_or_coprime (p n : ℕ) (hp : Prime p) :
  p ∣ n ∨ Nat.gcd p n = 1 := by sorry

end NUMINAMATH_CALUDE_prime_divides_or_coprime_l2596_259650


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2596_259616

theorem no_solution_for_equation :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2596_259616


namespace NUMINAMATH_CALUDE_percent_product_theorem_l2596_259675

theorem percent_product_theorem :
  let p1 : ℝ := 15
  let p2 : ℝ := 20
  let p3 : ℝ := 25
  (p1 / 100) * (p2 / 100) * (p3 / 100) * 100 = 0.75
  := by sorry

end NUMINAMATH_CALUDE_percent_product_theorem_l2596_259675


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2596_259613

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 66 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2596_259613


namespace NUMINAMATH_CALUDE_constant_function_proof_l2596_259619

theorem constant_function_proof (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x * y) + f (x * z) ≥ 1 + f x * f (y * z)) : 
  ∀ x : ℝ, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l2596_259619


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2596_259610

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := -x^2 - 4*x + 5

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 1}

-- Theorem stating that the solution set of the inequality is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2596_259610


namespace NUMINAMATH_CALUDE_angle_inequality_l2596_259692

theorem angle_inequality (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0) ↔
  x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l2596_259692


namespace NUMINAMATH_CALUDE_sqrt_square_12321_l2596_259623

theorem sqrt_square_12321 : (Real.sqrt 12321)^2 = 12321 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_12321_l2596_259623


namespace NUMINAMATH_CALUDE_circle_non_intersect_l2596_259618

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line
def line (k : ℤ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for non-intersection
def non_intersect (k : ℤ) (l : ℝ) : Prop :=
  ∀ x y : ℝ, line k x y →
    (x - 1)^2 + y^2 > (1 + l)^2

-- Main theorem
theorem circle_non_intersect :
  ∃ k : ℤ, ∀ l : ℝ, l > 0 → non_intersect k l ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_non_intersect_l2596_259618


namespace NUMINAMATH_CALUDE_vector_magnitude_l2596_259635

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_magnitude (e₁ e₂ : ℝ × ℝ) (h₁ : unit_vector e₁) (h₂ : unit_vector e₂)
  (h₃ : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) : 
  let a := (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2)
  (a.1^2 + a.2^2) = 7 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2596_259635


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2596_259685

theorem simplify_and_rationalize (x : ℝ) (hx : x > 0) :
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt x / Real.sqrt 12) * (Real.sqrt 6 / Real.sqrt 8) = 
  Real.sqrt (1260 * x) / 168 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2596_259685


namespace NUMINAMATH_CALUDE_base_8_properties_l2596_259645

-- Define the base 10 number
def base_10_num : ℕ := 9257

-- Define the base 8 representation as a list of digits
def base_8_rep : List ℕ := [2, 2, 0, 5, 1]

-- Theorem stating the properties we want to prove
theorem base_8_properties :
  -- The base 8 representation is correct
  (List.foldl (λ acc d => acc * 8 + d) 0 base_8_rep = base_10_num) ∧
  -- The product of the digits is 0
  (List.foldl (· * ·) 1 base_8_rep = 0) ∧
  -- The sum of the digits is 10
  (List.sum base_8_rep = 10) := by
  sorry

end NUMINAMATH_CALUDE_base_8_properties_l2596_259645


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l2596_259622

theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : (1003 : ℝ) ^ a + (1004 : ℝ) ^ b = (2006 : ℝ) ^ b)
  (h2 : (997 : ℝ) ^ a + (1009 : ℝ) ^ b = (2007 : ℝ) ^ a) : 
  a < b := by
sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l2596_259622


namespace NUMINAMATH_CALUDE_chocolate_packs_l2596_259638

theorem chocolate_packs (total packs_cookies packs_cake : ℕ) 
  (h_total : total = 42)
  (h_cookies : packs_cookies = 4)
  (h_cake : packs_cake = 22) :
  total - packs_cookies - packs_cake = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_packs_l2596_259638


namespace NUMINAMATH_CALUDE_number_of_factors_34650_l2596_259654

def number_to_factor := 34650

theorem number_of_factors_34650 :
  (Finset.filter (· ∣ number_to_factor) (Finset.range (number_to_factor + 1))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_34650_l2596_259654


namespace NUMINAMATH_CALUDE_henrys_action_figures_l2596_259648

def action_figure_problem (total_needed : ℕ) (cost_per_figure : ℕ) (money_needed : ℕ) : Prop :=
  let figures_to_buy : ℕ := money_needed / cost_per_figure
  let initial_figures : ℕ := total_needed - figures_to_buy
  initial_figures = 3

theorem henrys_action_figures :
  action_figure_problem 8 6 30 := by
  sorry

end NUMINAMATH_CALUDE_henrys_action_figures_l2596_259648


namespace NUMINAMATH_CALUDE_total_meal_cost_l2596_259606

def meal_cost (num_people : ℕ) (cost_per_person : ℚ) (tax_rate : ℚ) (tip_percentages : List ℚ) : ℚ :=
  let base_cost := num_people * cost_per_person
  let tax := base_cost * tax_rate
  let cost_with_tax := base_cost + tax
  let avg_tip_percentage := (tip_percentages.sum + 1) / tip_percentages.length
  let tip := cost_with_tax * avg_tip_percentage
  cost_with_tax + tip

theorem total_meal_cost :
  let num_people : ℕ := 5
  let cost_per_person : ℚ := 90
  let tax_rate : ℚ := 825 / 10000
  let tip_percentages : List ℚ := [15/100, 18/100, 20/100, 22/100, 25/100]
  meal_cost num_people cost_per_person tax_rate tip_percentages = 97426 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_meal_cost_l2596_259606


namespace NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_proof_l2596_259673

theorem cauchy_schwarz_and_inequality_proof :
  (∀ a b c x y z : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a*x + b*y + c*z)^2 ∧
    ((a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = (a*x + b*y + c*z)^2 ↔ a/x = b/y ∧ b/y = c/z)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    Real.sqrt a + Real.sqrt (2*b) + Real.sqrt (3*c) ≤ Real.sqrt 6 ∧
    (Real.sqrt a + Real.sqrt (2*b) + Real.sqrt (3*c) = Real.sqrt 6 ↔ a = 1/6 ∧ b = 1/3 ∧ c = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_proof_l2596_259673


namespace NUMINAMATH_CALUDE_some_number_equation_l2596_259621

theorem some_number_equation : ∃ n : ℤ, (69842^2 - n^2) / (69842 - n) = 100000 ∧ n = 30158 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l2596_259621


namespace NUMINAMATH_CALUDE_cereal_spending_l2596_259696

/-- The amount spent by Pop on cereal -/
def pop_spend : ℝ := 15

/-- The amount spent by Crackle on cereal -/
def crackle_spend : ℝ := 3 * pop_spend

/-- The amount spent by Snap on cereal -/
def snap_spend : ℝ := 2 * crackle_spend

/-- The total amount spent by Snap, Crackle, and Pop on cereal -/
def total_spend : ℝ := snap_spend + crackle_spend + pop_spend

theorem cereal_spending :
  total_spend = 150 := by sorry

end NUMINAMATH_CALUDE_cereal_spending_l2596_259696


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l2596_259682

theorem modular_inverse_17_mod_800 : ∃ x : ℕ, x < 800 ∧ (17 * x) % 800 = 1 :=
by
  use 47
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l2596_259682


namespace NUMINAMATH_CALUDE_no_solution_for_gcd_equation_l2596_259614

theorem no_solution_for_gcd_equation :
  ¬ ∃ (a b c : ℕ+), 
    Nat.gcd (a.val^2) (b.val^2) + 
    Nat.gcd a.val (Nat.gcd b.val c.val) + 
    Nat.gcd b.val (Nat.gcd a.val c.val) + 
    Nat.gcd c.val (Nat.gcd a.val b.val) = 199 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_gcd_equation_l2596_259614


namespace NUMINAMATH_CALUDE_union_complement_equality_l2596_259647

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 3, 4}

theorem union_complement_equality : A ∪ (U \ B) = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l2596_259647


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l2596_259663

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l2596_259663


namespace NUMINAMATH_CALUDE_circle_symmetry_minimum_l2596_259627

theorem circle_symmetry_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 1 = 0 ↔ x^2 + y^2 + 4*x - 2*y + 1 = 0 ∧ a*x - b*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (a' + 2*b') / (a' * b') ≥ (a + 2*b) / (a * b)) →
  (a + 2*b) / (a * b) = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_minimum_l2596_259627


namespace NUMINAMATH_CALUDE_inequality_proof_l2596_259626

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) :
  x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2596_259626


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l2596_259629

/-- Represents the amount of pure water added in liters -/
def W : ℝ := 1

/-- The initial volume of salt solution in liters -/
def initial_volume : ℝ := 1

/-- The initial concentration of salt in the solution -/
def initial_concentration : ℝ := 0.40

/-- The final concentration of salt in the mixture -/
def final_concentration : ℝ := 0.20

theorem salt_solution_mixture :
  initial_volume * initial_concentration = 
  (initial_volume + W) * final_concentration := by sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l2596_259629


namespace NUMINAMATH_CALUDE_lcm_gcd_1365_910_l2596_259677

theorem lcm_gcd_1365_910 :
  (Nat.lcm 1365 910 = 2730) ∧ (Nat.gcd 1365 910 = 455) := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_1365_910_l2596_259677


namespace NUMINAMATH_CALUDE_expansion_properties_l2596_259667

/-- Given that for some natural number n, the expansion of (x^(1/6) + x^(-1/6))^n has
    binomial coefficients of the 2nd, 3rd, and 4th terms forming an arithmetic sequence,
    prove that n = 7 and there is no constant term in the expansion. -/
theorem expansion_properties (n : ℕ) 
  (h : 2 * (n.choose 2) = n.choose 1 + n.choose 3) : 
  (n = 7) ∧ (∀ k : ℕ, (7 : ℚ) - 2 * k ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l2596_259667


namespace NUMINAMATH_CALUDE_parabola_sum_zero_l2596_259620

/-- A parabola passing through two specific points has a + b + c = 0 --/
theorem parabola_sum_zero (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (a * (-2)^2 + b * (-2) + c = -3) →
  (a * 2^2 + b * 2 + c = 5) →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_zero_l2596_259620


namespace NUMINAMATH_CALUDE_point_on_parabola_l2596_259659

theorem point_on_parabola (a : ℝ) : (a, -9) ∈ {(x, y) | y = -x^2} → (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_parabola_l2596_259659


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l2596_259603

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l2596_259603


namespace NUMINAMATH_CALUDE_opposites_imply_x_equals_one_l2596_259607

theorem opposites_imply_x_equals_one : 
  ∀ x : ℝ, (-2 * x) = -(3 * x - 1) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposites_imply_x_equals_one_l2596_259607


namespace NUMINAMATH_CALUDE_vector_operation_l2596_259608

/-- Given two vectors AB and AC in R², prove that 2AB - AC equals (5,7) -/
theorem vector_operation (AB AC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : AC = (-1, -1)) : 
  (2 : ℝ) • AB - AC = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l2596_259608


namespace NUMINAMATH_CALUDE_smallest_positive_omega_l2596_259600

theorem smallest_positive_omega : ∃ (ω : ℝ), 
  (ω > 0) ∧ 
  (∀ x, Real.sin (ω * (x - Real.pi / 6)) = Real.cos (ω * x)) ∧
  (∀ ω' > 0, (∀ x, Real.sin (ω' * (x - Real.pi / 6)) = Real.cos (ω' * x)) → ω ≤ ω') ∧
  ω = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_omega_l2596_259600


namespace NUMINAMATH_CALUDE_binomial_8_3_l2596_259678

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_3_l2596_259678


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l2596_259683

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 64 ways to distribute 6 distinguishable balls into 2 distinguishable boxes -/
theorem six_balls_two_boxes :
  distribute_balls 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l2596_259683


namespace NUMINAMATH_CALUDE_total_highlighters_l2596_259639

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 15

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 12

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 9

/-- The number of green highlighters in the teacher's desk -/
def green_highlighters : ℕ := 7

/-- The number of purple highlighters in the teacher's desk -/
def purple_highlighters : ℕ := 6

/-- Theorem stating that the total number of highlighters is 49 -/
theorem total_highlighters : 
  pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l2596_259639


namespace NUMINAMATH_CALUDE_sum_integers_50_to_70_l2596_259686

theorem sum_integers_50_to_70 (x y : ℕ) : 
  (x = (50 + 70) * (70 - 50 + 1) / 2) →  -- Sum of integers from 50 to 70
  (y = ((70 - 50) / 2 + 1)) →            -- Number of even integers from 50 to 70
  (x + y = 1271) → 
  (x = 1260) := by sorry

end NUMINAMATH_CALUDE_sum_integers_50_to_70_l2596_259686


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2596_259631

def z : ℂ := Complex.I * (1 + Complex.I)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2596_259631


namespace NUMINAMATH_CALUDE_area_of_triangle_LGH_l2596_259637

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def P : ℝ × ℝ := (0, 0)
def L : ℝ × ℝ := (-24, 0)
def M : ℝ × ℝ := (-10, 0)
def N : ℝ × ℝ := (10, 0)

-- Define the chords
def EF : Set (ℝ × ℝ) := {p | p.2 = 6}
def GH : Set (ℝ × ℝ) := {p | p.2 = 8}

-- State the theorem
theorem area_of_triangle_LGH : 
  ∀ (G H : ℝ × ℝ),
  G ∈ GH → H ∈ GH →
  G.1 < H.1 →
  H.1 - G.1 = 16 →
  (∀ (E F : ℝ × ℝ), E ∈ EF → F ∈ EF → F.1 - E.1 = 12) →
  (∀ p, p ∈ Circle P 10) →
  (∀ x, (x, 0) ∈ Set.Icc L N → (x, 0) ∈ Circle P 10) →
  let triangle_area := (1 / 2) * 16 * 6
  triangle_area = 48 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_LGH_l2596_259637


namespace NUMINAMATH_CALUDE_timothy_land_cost_l2596_259646

/-- Represents the cost breakdown of Timothy's farm --/
structure FarmCosts where
  land_acres : ℕ
  house_cost : ℕ
  cow_count : ℕ
  cow_cost : ℕ
  chicken_count : ℕ
  chicken_cost : ℕ
  solar_install_hours : ℕ
  solar_install_rate : ℕ
  solar_equipment_cost : ℕ
  total_cost : ℕ

/-- Calculates the cost per acre of land given the farm costs --/
def land_cost_per_acre (costs : FarmCosts) : ℕ :=
  (costs.total_cost - 
   (costs.house_cost + 
    costs.cow_count * costs.cow_cost + 
    costs.chicken_count * costs.chicken_cost + 
    costs.solar_install_hours * costs.solar_install_rate + 
    costs.solar_equipment_cost)) / costs.land_acres

/-- Theorem stating that the cost per acre of Timothy's land is $20 --/
theorem timothy_land_cost (costs : FarmCosts) 
  (h1 : costs.land_acres = 30)
  (h2 : costs.house_cost = 120000)
  (h3 : costs.cow_count = 20)
  (h4 : costs.cow_cost = 1000)
  (h5 : costs.chicken_count = 100)
  (h6 : costs.chicken_cost = 5)
  (h7 : costs.solar_install_hours = 6)
  (h8 : costs.solar_install_rate = 100)
  (h9 : costs.solar_equipment_cost = 6000)
  (h10 : costs.total_cost = 147700) :
  land_cost_per_acre costs = 20 := by
  sorry


end NUMINAMATH_CALUDE_timothy_land_cost_l2596_259646


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2596_259628

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + 
    a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2596_259628


namespace NUMINAMATH_CALUDE_thirty_percent_of_hundred_l2596_259670

theorem thirty_percent_of_hundred : ∃ x : ℝ, 30 = 0.30 * x ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_hundred_l2596_259670


namespace NUMINAMATH_CALUDE_dillar_dallar_never_equal_l2596_259662

/-- Represents the state of the financier's money -/
structure MoneyState :=
  (dillars : ℕ)
  (dallars : ℕ)

/-- Represents a currency exchange operation -/
inductive ExchangeOp
  | DillarToDallar : ExchangeOp
  | DallarToDillar : ExchangeOp

/-- Applies an exchange operation to a money state -/
def applyExchange (state : MoneyState) (op : ExchangeOp) : MoneyState :=
  match op with
  | ExchangeOp.DillarToDallar => 
      ⟨state.dillars - 1, state.dallars + 10⟩
  | ExchangeOp.DallarToDillar => 
      ⟨state.dillars + 10, state.dallars - 1⟩

/-- Applies a sequence of exchange operations to an initial state -/
def applyExchanges (initial : MoneyState) (ops : List ExchangeOp) : MoneyState :=
  ops.foldl applyExchange initial

theorem dillar_dallar_never_equal :
  ∀ (ops : List ExchangeOp),
    let finalState := applyExchanges ⟨1, 0⟩ ops
    finalState.dillars ≠ finalState.dallars :=
by sorry

end NUMINAMATH_CALUDE_dillar_dallar_never_equal_l2596_259662


namespace NUMINAMATH_CALUDE_inequality_range_l2596_259672

theorem inequality_range (a : ℝ) : 
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → (y / 4) - Real.cos x ^ 2 ≥ a * Real.sin x - (9 / y)) →
  -3 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2596_259672


namespace NUMINAMATH_CALUDE_twelve_mile_ride_cost_l2596_259698

/-- Calculates the cost of a taxi ride given the specified conditions -/
def taxiRideCost (baseFare mileRate discountThreshold discountRate miles : ℚ) : ℚ :=
  let totalBeforeDiscount := baseFare + mileRate * miles
  if miles > discountThreshold then
    totalBeforeDiscount * (1 - discountRate)
  else
    totalBeforeDiscount

theorem twelve_mile_ride_cost :
  taxiRideCost 2 (30/100) 10 (10/100) 12 = 504/100 := by
  sorry

#eval taxiRideCost 2 (30/100) 10 (10/100) 12

end NUMINAMATH_CALUDE_twelve_mile_ride_cost_l2596_259698


namespace NUMINAMATH_CALUDE_minimum_students_l2596_259657

theorem minimum_students (b g : ℕ) : 
  (3 * b = 4 * g) →  -- Equal number of boys and girls passed
  (∃ k : ℕ, b = 4 * k ∧ g = 3 * k) →  -- b and g are integers
  (b + g ≥ 7) ∧ (∀ m n : ℕ, (3 * m = 4 * n) → (m + n < 7 → m = 0 ∨ n = 0)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_students_l2596_259657


namespace NUMINAMATH_CALUDE_probability_not_pulling_prize_l2596_259679

/-- Given odds of 5:8 for pulling a prize, the probability of not pulling the prize is 8/13 -/
theorem probability_not_pulling_prize (favorable_outcomes unfavorable_outcomes : ℕ) 
  (h_odds : favorable_outcomes = 5 ∧ unfavorable_outcomes = 8) :
  (unfavorable_outcomes : ℚ) / (favorable_outcomes + unfavorable_outcomes) = 8 / 13 := by
  sorry


end NUMINAMATH_CALUDE_probability_not_pulling_prize_l2596_259679


namespace NUMINAMATH_CALUDE_ways_without_first_grade_ways_with_all_grades_l2596_259611

/-- Represents the number of products of each grade -/
structure ProductCounts where
  total : Nat
  firstGrade : Nat
  secondGrade : Nat
  thirdGrade : Nat

/-- The given product counts in the problem -/
def givenCounts : ProductCounts :=
  { total := 8
  , firstGrade := 3
  , secondGrade := 3
  , thirdGrade := 2 }

/-- Number of products to draw -/
def drawCount : Nat := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem for the first question -/
theorem ways_without_first_grade (counts : ProductCounts) :
  choose (counts.secondGrade + counts.thirdGrade) drawCount = 5 :=
sorry

/-- Theorem for the second question -/
theorem ways_with_all_grades (counts : ProductCounts) :
  choose counts.firstGrade 2 * choose counts.secondGrade 1 * choose counts.thirdGrade 1 +
  choose counts.firstGrade 1 * choose counts.secondGrade 2 * choose counts.thirdGrade 1 +
  choose counts.firstGrade 1 * choose counts.secondGrade 1 * choose counts.thirdGrade 2 = 45 :=
sorry

end NUMINAMATH_CALUDE_ways_without_first_grade_ways_with_all_grades_l2596_259611


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2596_259680

theorem complex_arithmetic_equality : (5 - 5*Complex.I) + (-2 - Complex.I) - (3 + 4*Complex.I) = -10*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2596_259680


namespace NUMINAMATH_CALUDE_emberly_walks_l2596_259634

theorem emberly_walks (total_days : Nat) (miles_per_walk : Nat) (total_miles : Nat) :
  total_days = 31 →
  miles_per_walk = 4 →
  total_miles = 108 →
  total_days - (total_miles / miles_per_walk) = 4 :=
by sorry

end NUMINAMATH_CALUDE_emberly_walks_l2596_259634


namespace NUMINAMATH_CALUDE_max_volume_smaller_pyramid_l2596_259633

/-- Regular square pyramid with base side length 2 and height 3 -/
structure SquarePyramid where
  base_side : ℝ
  height : ℝ
  base_side_eq : base_side = 2
  height_eq : height = 3

/-- Smaller pyramid formed by intersecting the main pyramid with a parallel plane -/
structure SmallerPyramid (p : SquarePyramid) where
  intersection_height : ℝ
  volume : ℝ
  height_bounds : 0 < intersection_height ∧ intersection_height < p.height
  volume_eq : volume = (4 / 27) * intersection_height^3 - (8 / 9) * intersection_height^2 + (4 / 3) * intersection_height

/-- The maximum volume of the smaller pyramid is 16/27 -/
theorem max_volume_smaller_pyramid (p : SquarePyramid) : 
  ∃ (sp : SmallerPyramid p), ∀ (other : SmallerPyramid p), sp.volume ≥ other.volume ∧ sp.volume = 16/27 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_smaller_pyramid_l2596_259633


namespace NUMINAMATH_CALUDE_no_integer_solution_l2596_259625

theorem no_integer_solution : ∀ x y : ℤ, 5 * x^2 - 4 * y^2 ≠ 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2596_259625


namespace NUMINAMATH_CALUDE_transport_cost_proof_l2596_259681

def cost_per_kg : ℝ := 18000
def instrument_mass_g : ℝ := 300

theorem transport_cost_proof :
  let instrument_mass_kg : ℝ := instrument_mass_g / 1000
  instrument_mass_kg * cost_per_kg = 5400 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_proof_l2596_259681


namespace NUMINAMATH_CALUDE_sphere_packing_ratio_l2596_259656

/-- Configuration of four spheres with two radii -/
structure SpherePacking where
  r : ℝ  -- radius of smaller spheres
  R : ℝ  -- radius of larger spheres
  r_positive : r > 0
  R_positive : R > 0
  touch_plane : True  -- represents that all spheres touch the plane
  touch_others : True  -- represents that each sphere touches three others

/-- Theorem stating the ratio of radii in the sphere packing configuration -/
theorem sphere_packing_ratio (config : SpherePacking) : config.R / config.r = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_packing_ratio_l2596_259656


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_15_l2596_259636

def f (n : ℕ+) : ℕ := sorry

theorem smallest_n_exceeding_15 :
  (∀ k : ℕ+, k < 3 → f k ≤ 15) ∧ f 3 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_15_l2596_259636


namespace NUMINAMATH_CALUDE_johns_dog_walking_earnings_l2596_259693

/-- Proves that John earns $10 per day for walking the dog -/
theorem johns_dog_walking_earnings :
  ∀ (days_in_april : ℕ) (sundays : ℕ) (total_spent : ℕ) (money_left : ℕ),
    days_in_april = 30 →
    sundays = 4 →
    total_spent = 100 →
    money_left = 160 →
    (days_in_april - sundays) * 10 = total_spent + money_left :=
by
  sorry

end NUMINAMATH_CALUDE_johns_dog_walking_earnings_l2596_259693


namespace NUMINAMATH_CALUDE_sum_of_gcd_values_l2596_259660

theorem sum_of_gcd_values (n : ℕ+) : 
  (Finset.sum (Finset.range 4) (λ i => (Nat.gcd (5 * n + 6) n).succ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_values_l2596_259660


namespace NUMINAMATH_CALUDE_veg_eaters_count_l2596_259699

/-- Represents the number of people in a family with different eating habits -/
structure FamilyEatingHabits where
  only_veg : ℕ
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ

/-- Theorem stating that the number of people who eat veg in the family is 20 -/
theorem veg_eaters_count (family : FamilyEatingHabits)
  (h1 : family.only_veg = 11)
  (h2 : family.only_non_veg = 6)
  (h3 : family.both_veg_and_non_veg = 9) :
  family.only_veg + family.both_veg_and_non_veg = 20 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l2596_259699


namespace NUMINAMATH_CALUDE_stating_lunch_potatoes_count_l2596_259641

/-- Represents the number of potatoes used for different purposes -/
structure PotatoUsage where
  total : ℕ
  dinner : ℕ
  lunch : ℕ

/-- 
Theorem stating that given a total of 7 potatoes and 2 used for dinner,
the number of potatoes used for lunch must be 5.
-/
theorem lunch_potatoes_count (usage : PotatoUsage) 
    (h1 : usage.total = 7)
    (h2 : usage.dinner = 2)
    (h3 : usage.total = usage.lunch + usage.dinner) : 
  usage.lunch = 5 := by
  sorry

end NUMINAMATH_CALUDE_stating_lunch_potatoes_count_l2596_259641


namespace NUMINAMATH_CALUDE_remaining_volleyballs_l2596_259668

/-- Given an initial number of volleyballs and a number of volleyballs lent out,
    calculate the number of volleyballs remaining. -/
def volleyballs_remaining (initial : ℕ) (lent_out : ℕ) : ℕ :=
  initial - lent_out

/-- Theorem stating that given 9 initial volleyballs and 5 lent out,
    the number of volleyballs remaining is 4. -/
theorem remaining_volleyballs :
  volleyballs_remaining 9 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_volleyballs_l2596_259668


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l2596_259690

theorem opposite_of_negative_five : 
  -((-5 : ℤ)) = (5 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l2596_259690


namespace NUMINAMATH_CALUDE_max_value_of_f_l2596_259652

def f (x : ℝ) : ℝ := -2 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2596_259652


namespace NUMINAMATH_CALUDE_trigonometric_properties_l2596_259615

theorem trigonometric_properties :
  (¬ ∃ α : ℝ, Real.sin α + Real.cos α = 3/2) ∧
  (∀ x : ℝ, Real.cos (7 * Real.pi / 2 - 3 * x) = -Real.cos (7 * Real.pi / 2 + 3 * x)) ∧
  (∀ x : ℝ, 4 * Real.sin (2 * (-9 * Real.pi / 8 + x) + 5 * Real.pi / 4) = 
            4 * Real.sin (2 * (-9 * Real.pi / 8 - x) + 5 * Real.pi / 4)) ∧
  (∃ x : ℝ, Real.sin (2 * x - Real.pi / 4) ≠ Real.sin (2 * (x - Real.pi / 8))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l2596_259615


namespace NUMINAMATH_CALUDE_participants_in_both_competitions_l2596_259604

theorem participants_in_both_competitions
  (total : ℕ)
  (chinese : ℕ)
  (math : ℕ)
  (neither : ℕ)
  (h1 : total = 50)
  (h2 : chinese = 30)
  (h3 : math = 38)
  (h4 : neither = 2) :
  chinese + math - (total - neither) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_participants_in_both_competitions_l2596_259604


namespace NUMINAMATH_CALUDE_equations_have_same_solutions_l2596_259691

def daniels_equation (x : ℝ) : Prop := |x - 8| = 3

def emmas_equation (x : ℝ) : Prop := x^2 - 16*x + 55 = 0

theorem equations_have_same_solutions :
  (∀ x : ℝ, daniels_equation x ↔ emmas_equation x) :=
sorry

end NUMINAMATH_CALUDE_equations_have_same_solutions_l2596_259691


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2596_259689

theorem container_volume_ratio (container1 container2 : ℝ) : 
  container1 > 0 → container2 > 0 →
  (2/3 : ℝ) * container1 + (1/6 : ℝ) * container1 = (5/6 : ℝ) * container2 →
  container1 = container2 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2596_259689


namespace NUMINAMATH_CALUDE_rain_free_paths_l2596_259602

/-- The function f representing the amount of rain at point (x,y) -/
def f (x y : ℝ) : ℝ := |x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3|

/-- The theorem stating that the set of m values for which f(x,mx) = 0 for all x
    is exactly {-1, 1/2, -1/3} -/
theorem rain_free_paths (x : ℝ) :
  {m : ℝ | ∀ x, f x (m*x) = 0} = {-1, 1/2, -1/3} := by
  sorry

end NUMINAMATH_CALUDE_rain_free_paths_l2596_259602


namespace NUMINAMATH_CALUDE_complex_multiplication_l2596_259640

theorem complex_multiplication (P F G : ℂ) : 
  P = 3 + 4*Complex.I ∧ 
  F = 2*Complex.I ∧ 
  G = 3 - 4*Complex.I → 
  (P + F) * G = 21 + 6*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2596_259640


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2596_259671

theorem inequality_solution_set (x : ℝ) : 
  (5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10) ↔ (8 / 3 < x ∧ x ≤ 20 / 7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2596_259671


namespace NUMINAMATH_CALUDE_quadratic_roots_squared_difference_l2596_259684

theorem quadratic_roots_squared_difference (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (x₁^2 - x₂^2 = c^2 / a^2) ↔ (b^4 - c^4 = 4*a*b^2*c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_squared_difference_l2596_259684


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l2596_259655

def repeating_decimal : ℚ := 36 / 99

theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l2596_259655


namespace NUMINAMATH_CALUDE_adult_meals_sold_l2596_259632

theorem adult_meals_sold (kids_meals : ℕ) (adult_meals : ℕ) : 
  (10 : ℚ) / 7 = kids_meals / adult_meals →
  kids_meals = 70 →
  adult_meals = 49 := by
sorry

end NUMINAMATH_CALUDE_adult_meals_sold_l2596_259632


namespace NUMINAMATH_CALUDE_brick_length_is_20_l2596_259697

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 26

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks required to build the wall -/
def num_bricks : ℕ := 26000

/-- Theorem stating that the length of the brick is 20 cm given the conditions -/
theorem brick_length_is_20 :
  brick_length = 20 ∧
  brick_width * brick_height * brick_length * num_bricks = 
  wall_length * wall_width * wall_height * 1000000 :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_20_l2596_259697


namespace NUMINAMATH_CALUDE_complete_square_sum_l2596_259601

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + b)^2 = c
    where b and c are integers, prove that b + c = 1 -/
theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + b)^2 = c) → b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2596_259601


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2596_259665

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  (a * b) / (c * d) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2596_259665


namespace NUMINAMATH_CALUDE_triangle_property_l2596_259643

open Real

theorem triangle_property (A B C a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  1 / tan A + 1 / tan C = 1 / sin B →
  b^2 = a * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2596_259643


namespace NUMINAMATH_CALUDE_proportional_value_l2596_259688

-- Define the given ratio
def given_ratio : ℚ := 12 / 6

-- Define the conversion factor from minutes to seconds
def minutes_to_seconds : ℕ := 60

-- Define the target time in minutes
def target_time_minutes : ℕ := 8

-- Define the target time in seconds
def target_time_seconds : ℕ := target_time_minutes * minutes_to_seconds

-- State the theorem
theorem proportional_value :
  (given_ratio * target_time_seconds : ℚ) = 960 := by sorry

end NUMINAMATH_CALUDE_proportional_value_l2596_259688


namespace NUMINAMATH_CALUDE_floor_sqrt_150_l2596_259674

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_150_l2596_259674


namespace NUMINAMATH_CALUDE_bus_ride_cost_l2596_259676

theorem bus_ride_cost (train_cost bus_cost : ℝ) : 
  train_cost = bus_cost + 6.35 →
  train_cost + bus_cost = 9.85 →
  bus_cost = 1.75 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l2596_259676
