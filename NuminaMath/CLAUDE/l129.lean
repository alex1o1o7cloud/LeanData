import Mathlib

namespace NUMINAMATH_CALUDE_selene_purchase_cost_l129_12970

/-- Calculates the total cost of items after applying a discount -/
def total_cost_after_discount (camera_price : ℚ) (frame_price : ℚ) (camera_count : ℕ) (frame_count : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_before_discount := camera_price * camera_count + frame_price * frame_count
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

/-- Proves that Selene pays $551 for her purchase -/
theorem selene_purchase_cost :
  let camera_price : ℚ := 110
  let frame_price : ℚ := 120
  let camera_count : ℕ := 2
  let frame_count : ℕ := 3
  let discount_rate : ℚ := 5 / 100
  total_cost_after_discount camera_price frame_price camera_count frame_count discount_rate = 551 := by
  sorry


end NUMINAMATH_CALUDE_selene_purchase_cost_l129_12970


namespace NUMINAMATH_CALUDE_marbles_cost_calculation_l129_12936

/-- The amount spent on marbles when the total spent on toys is known, along with the costs of a football and baseball. -/
def marbles_cost (total_spent football_cost baseball_cost : ℚ) : ℚ :=
  total_spent - (football_cost + baseball_cost)

/-- Theorem stating that the cost of marbles is the difference between the total spent and the sum of football and baseball costs. -/
theorem marbles_cost_calculation (total_spent football_cost baseball_cost : ℚ) 
  (h1 : total_spent = 20.52)
  (h2 : football_cost = 4.95)
  (h3 : baseball_cost = 6.52) : 
  marbles_cost total_spent football_cost baseball_cost = 9.05 := by
sorry

end NUMINAMATH_CALUDE_marbles_cost_calculation_l129_12936


namespace NUMINAMATH_CALUDE_expression_value_l129_12930

theorem expression_value (x y : ℝ) 
  (hx : (x - 15)^2 = 169) 
  (hy : (y - 1)^3 = -0.125) : 
  Real.sqrt x - Real.sqrt (2 * x * y) - (2 * y - x)^(1/3) = 3 ∨
  Real.sqrt x - Real.sqrt (2 * x * y) - (2 * y - x)^(1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l129_12930


namespace NUMINAMATH_CALUDE_magic_square_sum_l129_12963

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : Fin 3 → Fin 3 → ℕ
  sum_row : ∀ i, (Finset.univ.sum (λ j => a i j)) = (Finset.univ.sum (λ j => a 0 j))
  sum_col : ∀ j, (Finset.univ.sum (λ i => a i j)) = (Finset.univ.sum (λ j => a 0 j))
  sum_diag1 : (Finset.univ.sum (λ i => a i i)) = (Finset.univ.sum (λ j => a 0 j))
  sum_diag2 : (Finset.univ.sum (λ i => a i (2 - i))) = (Finset.univ.sum (λ j => a 0 j))

/-- The theorem to be proved -/
theorem magic_square_sum (s : MagicSquare) 
  (h1 : s.a 0 0 = 25)
  (h2 : s.a 0 2 = 23)
  (h3 : s.a 1 0 = 18)
  (h4 : s.a 2 1 = 22) :
  s.a 1 2 + s.a 0 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l129_12963


namespace NUMINAMATH_CALUDE_three_dice_not_one_or_six_l129_12981

/-- The probability of a single die not showing 1 or 6 -/
def single_die_prob : ℚ := 4 / 6

/-- The number of dice tossed -/
def num_dice : ℕ := 3

/-- The probability that none of the three dice show 1 or 6 -/
def three_dice_prob : ℚ := single_die_prob ^ num_dice

theorem three_dice_not_one_or_six :
  three_dice_prob = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_three_dice_not_one_or_six_l129_12981


namespace NUMINAMATH_CALUDE_wang_hua_practice_days_l129_12904

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in the Gregorian calendar -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInMonth (year : Nat) (month : Nat) : Nat :=
  match month with
  | 2 => if isLeapYear year then 29 else 28
  | 4 | 6 | 9 | 11 => 30
  | _ => 31

def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def countPracticeDays (year : Nat) (month : Nat) : Nat :=
  sorry

theorem wang_hua_practice_days :
  let newYearsDay2016 := Date.mk 2016 1 1
  let augustFirst2016 := Date.mk 2016 8 1
  dayOfWeek newYearsDay2016 = DayOfWeek.Friday →
  isLeapYear 2016 = true →
  countPracticeDays 2016 8 = 9 :=
by sorry

end NUMINAMATH_CALUDE_wang_hua_practice_days_l129_12904


namespace NUMINAMATH_CALUDE_total_cost_european_stamps_50s_60s_l129_12925

-- Define the cost of stamps
def italy_stamp_cost : ℚ := 0.07
def germany_stamp_cost : ℚ := 0.03

-- Define the number of stamps collected
def italy_stamps_50s_60s : ℕ := 9
def germany_stamps_50s_60s : ℕ := 15

-- Theorem statement
theorem total_cost_european_stamps_50s_60s : 
  (italy_stamp_cost * italy_stamps_50s_60s + germany_stamp_cost * germany_stamps_50s_60s : ℚ) = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_european_stamps_50s_60s_l129_12925


namespace NUMINAMATH_CALUDE_domain_of_f_l129_12945

def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (9 - x) ^ (1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 9 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l129_12945


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l129_12942

/-- Represents the sum of elements in the nth set of a sequence of sets,
    where each set contains consecutive integers and has one more element than the previous set. -/
def S (n : ℕ) : ℕ :=
  let first := (n * (n - 1)) / 2 + 1
  let last := first + n - 1
  n * (first + last) / 2

/-- The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l129_12942


namespace NUMINAMATH_CALUDE_min_tangent_length_l129_12966

/-- The minimum length of a tangent from a point on the line y = x + 2 to the circle (x - 4)² + (y + 2)² = 1 is √31. -/
theorem min_tangent_length (x y : ℝ) : 
  let line := {(x, y) | y = x + 2}
  let circle := {(x, y) | (x - 4)^2 + (y + 2)^2 = 1}
  let center := (4, -2)
  let dist_to_line (p : ℝ × ℝ) := |p.1 + p.2 + 2| / Real.sqrt 2
  let min_dist := dist_to_line center
  let tangent_length := Real.sqrt (min_dist^2 - 1)
  tangent_length = Real.sqrt 31 := by sorry

end NUMINAMATH_CALUDE_min_tangent_length_l129_12966


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l129_12948

def C : Set Nat := {70, 72, 75, 76, 78}

theorem smallest_prime_factor_in_C :
  ∃ n ∈ C, (∃ p : Nat, Nat.Prime p ∧ p ∣ n ∧ p = 2) ∧
  ∀ m ∈ C, ∀ q : Nat, Nat.Prime q → q ∣ m → q ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l129_12948


namespace NUMINAMATH_CALUDE_kayak_production_sum_l129_12952

theorem kayak_production_sum (a : ℕ) (r : ℕ) (n : ℕ) : 
  a = 9 → r = 3 → n = 5 → a * (r^n - 1) / (r - 1) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l129_12952


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l129_12956

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_of_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l129_12956


namespace NUMINAMATH_CALUDE_g_fixed_points_l129_12935

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 5 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_fixed_points_l129_12935


namespace NUMINAMATH_CALUDE_peters_candies_l129_12977

theorem peters_candies : ∃ (initial : ℚ), 
  initial > 0 ∧ 
  (1/4 * initial - 13/2 : ℚ) = 6 ∧
  initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_peters_candies_l129_12977


namespace NUMINAMATH_CALUDE_train_length_l129_12906

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length 
  (speed : ℝ) 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (h1 : speed = 45) -- km/hr
  (h2 : time_to_cross = 30 / 3600) -- 30 seconds converted to hours
  (h3 : bridge_length = 265) -- meters
  : ∃ (train_length : ℝ), train_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l129_12906


namespace NUMINAMATH_CALUDE_exists_non_unique_f_l129_12954

theorem exists_non_unique_f : ∃ (f : ℕ → ℕ), 
  (∀ n : ℕ, f (f n) = 4 * n + 9) ∧ 
  (∀ k : ℕ, f (2^(k-1)) = 2^k + 3) ∧ 
  (∃ n : ℕ, f n ≠ 2 * n + 3) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_unique_f_l129_12954


namespace NUMINAMATH_CALUDE_cost_increase_l129_12926

/-- Cost function -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

theorem cost_increase (t : ℝ) (b₀ b₁ : ℝ) (h : t > 0) :
  cost t b₁ = 16 * cost t b₀ → b₁ = 2 * b₀ := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_l129_12926


namespace NUMINAMATH_CALUDE_total_ways_eq_2501_l129_12903

/-- The number of different types of cookies --/
def num_cookie_types : ℕ := 6

/-- The number of different types of milk --/
def num_milk_types : ℕ := 4

/-- The total number of item types --/
def total_item_types : ℕ := num_cookie_types + num_milk_types

/-- The number of items they purchase collectively --/
def total_items : ℕ := 4

/-- Represents a purchase combination for Charlie and Delta --/
structure Purchase where
  charlie_items : ℕ
  delta_items : ℕ
  charlie_items_le_total_types : charlie_items ≤ total_item_types
  delta_items_le_cookies : delta_items ≤ num_cookie_types
  sum_eq_total : charlie_items + delta_items = total_items

/-- The number of ways to choose items for a given purchase combination --/
def ways_to_choose (p : Purchase) : ℕ := sorry

/-- The total number of ways Charlie and Delta can purchase items --/
def total_ways : ℕ := sorry

/-- The main theorem: proving the total number of ways is 2501 --/
theorem total_ways_eq_2501 : total_ways = 2501 := sorry

end NUMINAMATH_CALUDE_total_ways_eq_2501_l129_12903


namespace NUMINAMATH_CALUDE_circle_condition_implies_m_range_necessary_but_not_sufficient_condition_implies_a_range_l129_12987

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*m*x + 5*m^2 + m - 2 = 0

-- Define the condition for being a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y m

-- Define the inequality condition
def inequality_condition (m a : ℝ) : Prop :=
  (m - a) * (m - a - 4) < 0

theorem circle_condition_implies_m_range :
  (∀ m : ℝ, is_circle m → m > -2 ∧ m < 1) :=
sorry

theorem necessary_but_not_sufficient_condition_implies_a_range :
  (∀ a : ℝ, (∀ m : ℝ, inequality_condition m a → is_circle m) ∧
            (∃ m : ℝ, is_circle m ∧ ¬inequality_condition m a) →
   a ≥ -3 ∧ a ≤ -2) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_implies_m_range_necessary_but_not_sufficient_condition_implies_a_range_l129_12987


namespace NUMINAMATH_CALUDE_product_of_fractions_l129_12939

theorem product_of_fractions (p : ℝ) (hp : p ≠ 0) :
  (p^3 + 4*p^2 + 10*p + 12) / (p^3 - p^2 + 2*p + 16) *
  (p^3 - 3*p^2 + 8*p) / (p^2 + 2*p + 6) =
  ((p^3 + 4*p^2 + 10*p + 12) * (p^3 - 3*p^2 + 8*p)) /
  ((p^3 - p^2 + 2*p + 16) * (p^2 + 2*p + 6)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_fractions_l129_12939


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l129_12915

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) : ℕ :=
  let cans_for_children := (children_fed + can_capacity.children - 1) / can_capacity.children
  let remaining_cans := total_cans - cans_for_children
  remaining_cans * can_capacity.adults

/-- The main theorem to be proved -/
theorem soup_feeding_theorem (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) :
  total_cans = 8 →
  can_capacity.adults = 5 →
  can_capacity.children = 10 →
  children_fed = 20 →
  remaining_adults_fed total_cans can_capacity children_fed = 30 := by
  sorry

#check soup_feeding_theorem

end NUMINAMATH_CALUDE_soup_feeding_theorem_l129_12915


namespace NUMINAMATH_CALUDE_sin_2023_closest_to_neg_sqrt2_over_2_l129_12994

-- Define the set of options
def options : Set ℝ := {1/2, Real.sqrt 2 / 2, -1/2, -Real.sqrt 2 / 2}

-- Define the sine function with period 360°
noncomputable def periodic_sin (x : ℝ) : ℝ := Real.sin (2 * Real.pi * (x / 360))

-- State the theorem
theorem sin_2023_closest_to_neg_sqrt2_over_2 :
  ∃ (y : ℝ), y ∈ options ∧ 
  ∀ (z : ℝ), z ∈ options → |periodic_sin 2023 - y| ≤ |periodic_sin 2023 - z| :=
sorry

end NUMINAMATH_CALUDE_sin_2023_closest_to_neg_sqrt2_over_2_l129_12994


namespace NUMINAMATH_CALUDE_product_zero_l129_12984

theorem product_zero (a b c : ℝ) : 
  (a^2 + b^2 = 1 ∧ a + b = 1 → a * b = 0) ∧
  (a^3 + b^3 + c^3 = 1 ∧ a^2 + b^2 + c^2 = 1 ∧ a + b + c = 1 → a * b * c = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l129_12984


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l129_12959

/-- The percentage of students who passed an examination, given the total number of students and the number of students who failed. -/
def percentage_passed (total : ℕ) (failed : ℕ) : ℚ :=
  (total - failed : ℚ) / total * 100

/-- Theorem stating that the percentage of students who passed is 35% -/
theorem exam_pass_percentage :
  percentage_passed 540 351 = 35 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l129_12959


namespace NUMINAMATH_CALUDE_part_one_part_two_l129_12934

-- Part I
theorem part_one (a : ℝ) (h : ∀ x, x ∈ Set.Icc (-6 : ℝ) 2 ↔ |a * x - 1| ≤ 2) : 
  a = -1/2 := by sorry

-- Part II
theorem part_two (m : ℝ) (h : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : 
  m ∈ Set.Iic (7/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l129_12934


namespace NUMINAMATH_CALUDE_trapezoid_existence_l129_12968

/-- Represents a trapezoid ABCD with AB parallel to CD -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Checks if the given points form a valid trapezoid -/
def is_valid_trapezoid (t : Trapezoid) : Prop := sorry

/-- Calculates the perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Calculates the length of diagonal AC -/
def diagonal_ac (t : Trapezoid) : ℝ := sorry

/-- Calculates the angle DAB -/
def angle_dab (t : Trapezoid) : ℝ := sorry

/-- Calculates the angle ABC -/
def angle_abc (t : Trapezoid) : ℝ := sorry

/-- Theorem: Given a perimeter k, diagonal e, and angles α and β,
    there exists 0, 1, or 2 trapezoids satisfying these conditions -/
theorem trapezoid_existence (k e α β : ℝ) :
  ∃ n : Fin 3, ∃ ts : Finset Trapezoid,
    ts.card = n ∧
    ∀ t ∈ ts, is_valid_trapezoid t ∧
               perimeter t = k ∧
               diagonal_ac t = e ∧
               angle_dab t = α ∧
               angle_abc t = β := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_existence_l129_12968


namespace NUMINAMATH_CALUDE_correct_tax_distribution_l129_12985

-- Define the types of taxes
inductive TaxType
  | PropertyTax
  | FederalTax
  | ProfitTax
  | RegionalTax
  | TransportTax

-- Define the budget levels
inductive BudgetLevel
  | Federal
  | Regional

-- Function to map tax types to budget levels
def taxDistribution : TaxType → BudgetLevel
  | TaxType.PropertyTax => BudgetLevel.Regional
  | TaxType.FederalTax => BudgetLevel.Federal
  | TaxType.ProfitTax => BudgetLevel.Regional
  | TaxType.RegionalTax => BudgetLevel.Regional
  | TaxType.TransportTax => BudgetLevel.Regional

-- Theorem stating the correct distribution of taxes
theorem correct_tax_distribution :
  (taxDistribution TaxType.PropertyTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.FederalTax = BudgetLevel.Federal) ∧
  (taxDistribution TaxType.ProfitTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.RegionalTax = BudgetLevel.Regional) ∧
  (taxDistribution TaxType.TransportTax = BudgetLevel.Regional) :=
by sorry

end NUMINAMATH_CALUDE_correct_tax_distribution_l129_12985


namespace NUMINAMATH_CALUDE_officer_assignment_count_l129_12988

def group_size : ℕ := 4
def roles : ℕ := 3

theorem officer_assignment_count : (group_size.choose roles) * (Nat.factorial roles) = 24 := by
  sorry

end NUMINAMATH_CALUDE_officer_assignment_count_l129_12988


namespace NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l129_12951

/-- The number of days in a leap year -/
def leapYearDays : ℕ := 366

/-- The number of full weeks in a leap year -/
def fullWeeks : ℕ := leapYearDays / 7

/-- The number of extra days in a leap year after full weeks -/
def extraDays : ℕ := leapYearDays % 7

/-- The number of possible combinations for the extra days -/
def possibleCombinations : ℕ := 7

/-- The number of combinations that include a Monday -/
def combinationsWithMonday : ℕ := 2

/-- The probability of a leap year having 53 Mondays -/
def probabilityOf53Mondays : ℚ := combinationsWithMonday / possibleCombinations

theorem leap_year_53_mondays_probability :
  probabilityOf53Mondays = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l129_12951


namespace NUMINAMATH_CALUDE_equation_solutions_l129_12917

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x - 1)^2 - 4*x = 0
def equation2 (x : ℝ) : Prop := (2*x - 3)^2 = x^2

-- State the theorem
theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 3 / 2 ∧ x2 = 1 - Real.sqrt 3 / 2 ∧ equation1 x1 ∧ equation1 x2) ∧
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 1 ∧ equation2 x1 ∧ equation2 x2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l129_12917


namespace NUMINAMATH_CALUDE_trigonometric_identity_l129_12916

theorem trigonometric_identity (x : ℝ) :
  Real.sin (x + 2 * Real.pi) * Real.cos (2 * x - 7 * Real.pi / 2) +
  Real.sin (3 * Real.pi / 2 - x) * Real.sin (2 * x - 5 * Real.pi / 2) =
  Real.cos (3 * x) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l129_12916


namespace NUMINAMATH_CALUDE_solve_plug_problem_l129_12921

def plug_problem (mittens_pairs : ℕ) (original_plug_pairs_diff : ℕ) (final_plugs : ℕ) : Prop :=
  let original_plug_pairs : ℕ := mittens_pairs + original_plug_pairs_diff
  let original_plugs : ℕ := original_plug_pairs * 2
  let added_plugs : ℕ := final_plugs - original_plugs
  let added_pairs : ℕ := added_plugs / 2
  added_pairs = 70

theorem solve_plug_problem :
  plug_problem 150 20 400 :=
by sorry

end NUMINAMATH_CALUDE_solve_plug_problem_l129_12921


namespace NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l129_12908

/-- Set A is defined as {x | x^2 + x - 6 = 0} -/
def A : Set ℝ := {x | x^2 + x - 6 = 0}

/-- Set B is defined as {x | x * m + 1 = 0} -/
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

/-- Theorem stating a sufficient condition for B to be a proper subset of A -/
theorem sufficient_condition_B_proper_subset_A :
  ∀ m : ℝ, m ∈ ({0, 1/3} : Set ℝ) → B m ⊂ A ∧ B m ≠ A :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l129_12908


namespace NUMINAMATH_CALUDE_specific_flowerbed_area_l129_12971

/-- Represents a circular flowerbed with a straight path through its center -/
structure Flowerbed where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the plantable area of a flowerbed -/
def plantableArea (f : Flowerbed) : ℝ := sorry

/-- Theorem stating the plantable area of a specific flowerbed configuration -/
theorem specific_flowerbed_area :
  let f : Flowerbed := { diameter := 20, pathWidth := 4 }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |plantableArea f - 58.66 * Real.pi| < ε :=
sorry

end NUMINAMATH_CALUDE_specific_flowerbed_area_l129_12971


namespace NUMINAMATH_CALUDE_probability_larry_first_specific_l129_12962

/-- The probability of erasing Larry's whales first --/
def probability_larry_first (evan larry alex : ℕ) : ℚ :=
  let total := evan + larry + alex
  let p_evan_last := (evan : ℚ) / total
  let p_alex_last := (alex : ℚ) / total
  let p_larry_before_alex := (larry : ℚ) / (evan + larry)
  let p_larry_before_evan := (larry : ℚ) / (alex + larry)
  p_evan_last * p_larry_before_alex + p_alex_last * p_larry_before_evan

theorem probability_larry_first_specific :
  probability_larry_first 10 15 20 = 38 / 105 := by
  sorry

#eval probability_larry_first 10 15 20

end NUMINAMATH_CALUDE_probability_larry_first_specific_l129_12962


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l129_12950

theorem complex_arithmetic_equality : (481 * 7 + 426 * 5)^3 - 4 * (481 * 7) * (426 * 5) = 166021128033 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l129_12950


namespace NUMINAMATH_CALUDE_expression_equality_l129_12914

theorem expression_equality : 
  (1/2)⁻¹ + 4 * Real.cos (60 * π / 180) - |-3| + Real.sqrt 9 - (-2023)^0 + (-1)^(2023-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l129_12914


namespace NUMINAMATH_CALUDE_average_price_is_45_cents_l129_12907

/-- Represents the fruit selection and pricing problem -/
structure FruitProblem where
  apple_price : ℕ
  orange_price : ℕ
  total_fruits : ℕ
  initial_avg_price : ℕ
  oranges_removed : ℕ

/-- Calculates the average price of remaining fruits -/
def average_price_after_removal (fp : FruitProblem) : ℚ :=
  sorry

/-- Theorem stating that the average price of remaining fruits is 45 cents -/
theorem average_price_is_45_cents (fp : FruitProblem) 
  (h1 : fp.apple_price = 40)
  (h2 : fp.orange_price = 60)
  (h3 : fp.total_fruits = 10)
  (h4 : fp.initial_avg_price = 54)
  (h5 : fp.oranges_removed = 6) :
  average_price_after_removal fp = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_price_is_45_cents_l129_12907


namespace NUMINAMATH_CALUDE_solution_set_inequality_l129_12989

-- Define the determinant function
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the logarithm with base √2
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- Theorem statement
theorem solution_set_inequality (x : ℝ) :
  (log_sqrt2 (det 1 1 1 x) < 0) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l129_12989


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l129_12955

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (lamp_post_time : ℝ) 
  (h1 : train_length = 75) 
  (h2 : bridge_length = 150) 
  (h3 : lamp_post_time = 2.5) : 
  (train_length + bridge_length) / (train_length / lamp_post_time) = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l129_12955


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l129_12932

/-- Given that nine identical bowling balls weigh the same as four identical canoes,
    and one canoe weighs 36 pounds, prove that one bowling ball weighs 16 pounds. -/
theorem bowling_ball_weight (canoe_weight : ℕ) (ball_weight : ℚ)
  (h1 : canoe_weight = 36)
  (h2 : 9 * ball_weight = 4 * canoe_weight) :
  ball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l129_12932


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l129_12975

theorem sum_of_squares_problem (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → 
  a^2 + b^2 + c^2 = 64 → 
  a*b + b*c + c*a = 30 → 
  a + b + c = 2 * Real.sqrt 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l129_12975


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l129_12996

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, -1)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y - 16 = 0

-- Theorem statement
theorem parallel_line_through_point :
  (parallel_line point_P.1 point_P.2) ∧ 
  (∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), given_line (x + k) (y + (3/4) * k)) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l129_12996


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l129_12943

theorem pure_imaginary_complex_number (a : ℝ) : 
  (a^3 - a = 0 ∧ a/(1-a) ≠ 0) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l129_12943


namespace NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_is_12_l129_12953

/-- A right triangle with legs of length 6 containing an inscribed rectangle -/
structure RightTriangleWithInscribedRectangle where
  /-- The length of each leg of the right triangle -/
  leg_length : ℝ
  /-- The inscribed rectangle shares an angle with the triangle -/
  shares_angle : Bool
  /-- The inscribed rectangle is contained within the triangle -/
  is_inscribed : Bool

/-- The perimeter of the inscribed rectangle -/
def inscribed_rectangle_perimeter (t : RightTriangleWithInscribedRectangle) : ℝ := 12

/-- Theorem: The perimeter of the inscribed rectangle is 12 -/
theorem inscribed_rectangle_perimeter_is_12 (t : RightTriangleWithInscribedRectangle)
  (h1 : t.leg_length = 6)
  (h2 : t.shares_angle = true)
  (h3 : t.is_inscribed = true) :
  inscribed_rectangle_perimeter t = 12 := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_is_12_l129_12953


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_squared_l129_12990

theorem decimal_equivalent_of_one_fourth_squared :
  (1 / 4 : ℚ) ^ 2 = (0.0625 : ℚ) := by sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fourth_squared_l129_12990


namespace NUMINAMATH_CALUDE_max_value_of_function_l129_12937

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  ∃ (max_y : ℝ), max_y = 1 ∧ ∀ y, y = 4*x - 2 + 1/(4*x - 5) → y ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l129_12937


namespace NUMINAMATH_CALUDE_houses_before_boom_count_l129_12973

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before_boom : ℕ := 2000 - 574

/-- The current number of houses in Lawrence County -/
def current_houses : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

theorem houses_before_boom_count : houses_before_boom = 1426 := by
  sorry

end NUMINAMATH_CALUDE_houses_before_boom_count_l129_12973


namespace NUMINAMATH_CALUDE_sample_size_equals_surveyed_parents_l129_12961

/-- Represents a school survey about students' daily activities -/
structure SchoolSurvey where
  total_students : ℕ
  surveyed_parents : ℕ
  sleep_6_to_7_hours_percentage : ℚ
  homework_3_to_4_hours_percentage : ℚ

/-- The sample size of a school survey is equal to the number of surveyed parents -/
theorem sample_size_equals_surveyed_parents (survey : SchoolSurvey) 
  (h1 : survey.total_students = 1800)
  (h2 : survey.surveyed_parents = 1000)
  (h3 : survey.sleep_6_to_7_hours_percentage = 70/100)
  (h4 : survey.homework_3_to_4_hours_percentage = 28/100) :
  survey.surveyed_parents = 1000 := by
  sorry

#check sample_size_equals_surveyed_parents

end NUMINAMATH_CALUDE_sample_size_equals_surveyed_parents_l129_12961


namespace NUMINAMATH_CALUDE_expression_eval_zero_l129_12938

theorem expression_eval_zero (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_eval_zero_l129_12938


namespace NUMINAMATH_CALUDE_train_speed_equation_l129_12986

theorem train_speed_equation (x : ℝ) (h : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 ↔ 
  (∃ (t_express t_highspeed : ℝ),
    t_express = 700 / x ∧
    t_highspeed = 700 / (2.8 * x) ∧
    t_express - t_highspeed = 3.6) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_equation_l129_12986


namespace NUMINAMATH_CALUDE_factor_sum_l129_12958

theorem factor_sum (R S : ℝ) : 
  (∃ d e : ℝ, (X ^ 2 + 3 * X + 7) * (X ^ 2 + d * X + e) = X ^ 4 + R * X ^ 2 + S) →
  R + S = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l129_12958


namespace NUMINAMATH_CALUDE_shoe_price_increase_l129_12960

theorem shoe_price_increase (regular_price : ℝ) (h : regular_price > 0) :
  let sale_price := regular_price * (1 - 0.2)
  (regular_price - sale_price) / sale_price * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_increase_l129_12960


namespace NUMINAMATH_CALUDE_square_area_ratio_l129_12992

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := (s₂ * Real.sqrt 2) / 2
  (s₁^2) / (s₂^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l129_12992


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l129_12900

theorem merchant_markup_percentage
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (profit_percentage : ℝ)
  (h1 : discount_percentage = 10)
  (h2 : profit_percentage = 35)
  (h3 : cost_price > 0) :
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + profit_percentage / 100) →
  markup_percentage = 50 :=
by sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l129_12900


namespace NUMINAMATH_CALUDE_cubic_system_solution_l129_12901

theorem cubic_system_solution (x y z : ℝ) : 
  ((x + y)^3 = z ∧ (y + z)^3 = x ∧ (z + x)^3 = y) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = Real.sqrt 2 / 4 ∧ y = Real.sqrt 2 / 4 ∧ z = Real.sqrt 2 / 4) ∨ 
   (x = -Real.sqrt 2 / 4 ∧ y = -Real.sqrt 2 / 4 ∧ z = -Real.sqrt 2 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_system_solution_l129_12901


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_function_l129_12993

theorem max_value_of_quadratic_function :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x : ℝ, x - x^2 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_function_l129_12993


namespace NUMINAMATH_CALUDE_cistern_filling_time_l129_12995

/-- The time it takes to fill a cistern when two taps (one filling, one emptying) are opened simultaneously -/
theorem cistern_filling_time 
  (fill_time : ℝ) 
  (empty_time : ℝ) 
  (fill_time_pos : 0 < fill_time)
  (empty_time_pos : 0 < empty_time) : 
  (fill_time * empty_time) / (empty_time - fill_time) = 
    1 / (1 / fill_time - 1 / empty_time) :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l129_12995


namespace NUMINAMATH_CALUDE_existence_of_n_with_s_prime_divisors_l129_12913

theorem existence_of_n_with_s_prime_divisors (s : ℕ) (hs : s > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ (P : Finset Nat), P.card ≥ s ∧ 
    (∀ p ∈ P, Nat.Prime p ∧ p ∣ (2^n - 1))) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_n_with_s_prime_divisors_l129_12913


namespace NUMINAMATH_CALUDE_parabola_and_intersection_l129_12976

/-- Parabola with vertex at origin and focus on x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on the parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : par.equation x y

/-- Line intersecting the parabola -/
structure IntersectingLine (par : Parabola) where
  k : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = k * x + b
  intersects_twice : ∃ (p1 p2 : PointOnParabola par), p1 ≠ p2 ∧ 
    equation p1.x p1.y ∧ equation p2.x p2.y

theorem parabola_and_intersection 
    (par : Parabola) 
    (A : PointOnParabola par)
    (h1 : A.x = 4)
    (h2 : (A.x + par.p / 2)^2 + A.y^2 = 6^2)
    (line : IntersectingLine par)
    (h3 : line.b = -2)
    (h4 : ∃ (B : PointOnParabola par), 
      line.equation B.x B.y ∧ (A.x + B.x) / 2 = 2) :
  par.p = 4 ∧ line.k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_and_intersection_l129_12976


namespace NUMINAMATH_CALUDE_sin_theta_value_l129_12982

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) 
  (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : Real.sin θ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l129_12982


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_2718_l129_12967

/-- Calculate the cost of whitewashing a room's walls --/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ) 
                     (doorHeight doorWidth : ℝ)
                     (windowHeight windowWidth : ℝ)
                     (numWindows : ℕ)
                     (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorHeight * doorWidth
  let windowArea := windowHeight * windowWidth * numWindows
  (wallArea - doorArea - windowArea) * costPerSquareFoot

/-- Theorem stating the cost of whitewashing for the given room --/
theorem whitewashing_cost_is_2718 :
  whitewashingCost 25 15 12 6 3 4 3 3 3 = 2718 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_2718_l129_12967


namespace NUMINAMATH_CALUDE_parabola_point_order_l129_12922

/-- A parabola with equation y = -x^2 + 2x + c -/
structure Parabola where
  c : ℝ

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = -p.x^2 + 2*p.x + para.c

theorem parabola_point_order (para : Parabola) (p1 p2 p3 : Point)
  (h1 : p1.x = 0) (h2 : p2.x = 1) (h3 : p3.x = 3)
  (h4 : lies_on p1 para) (h5 : lies_on p2 para) (h6 : lies_on p3 para) :
  p2.y > p1.y ∧ p1.y > p3.y := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l129_12922


namespace NUMINAMATH_CALUDE_product_of_roots_l129_12929

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (x + 3) * (x - 4) = 22 ∧ x * y = -34 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l129_12929


namespace NUMINAMATH_CALUDE_decimal_difference_l129_12949

/-- The value of the repeating decimal 0.717171... -/
def repeating_decimal : ℚ := 71 / 99

/-- The value of the terminating decimal 0.71 -/
def terminating_decimal : ℚ := 71 / 100

/-- The theorem stating that the difference between 0.717171... and 0.71 is 71/9900 -/
theorem decimal_difference : 
  repeating_decimal - terminating_decimal = 71 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l129_12949


namespace NUMINAMATH_CALUDE_relationship_abcd_l129_12909

theorem relationship_abcd (a b c d : ℝ) 
  (hab : a < b) 
  (hdc : d < c) 
  (hcab : (c - a) * (c - b) < 0) 
  (hdab : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_relationship_abcd_l129_12909


namespace NUMINAMATH_CALUDE_s_squared_minus_c_squared_range_l129_12919

/-- The theorem states that for any point (x, y, z) in 3D space,
    where r is the distance from the origin to the point,
    s = y/r, and c = x/r, the value of s^2 - c^2 is always
    between -1 and 1, inclusive. -/
theorem s_squared_minus_c_squared_range
  (x y z : ℝ) 
  (r : ℝ) 
  (hr : r = Real.sqrt (x^2 + y^2 + z^2)) 
  (s : ℝ) 
  (hs : s = y / r) 
  (c : ℝ) 
  (hc : c = x / r) : 
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_s_squared_minus_c_squared_range_l129_12919


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l129_12928

theorem imaginary_part_of_complex_expression : 
  let z : ℂ := 1 - Complex.I
  let expression : ℂ := z^2 + 2/z
  Complex.im expression = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l129_12928


namespace NUMINAMATH_CALUDE_janice_starting_sentences_janice_started_with_258_sentences_l129_12918

/-- Calculates the number of sentences Janice started with today -/
theorem janice_starting_sentences 
  (typing_speed : ℕ) 
  (typing_duration1 typing_duration2 typing_duration3 : ℕ)
  (erased_sentences : ℕ)
  (total_sentences : ℕ) : ℕ :=
  let total_duration := typing_duration1 + typing_duration2 + typing_duration3
  let typed_sentences := total_duration * typing_speed
  let net_typed_sentences := typed_sentences - erased_sentences
  total_sentences - net_typed_sentences

/-- Proves that Janice started with 258 sentences today -/
theorem janice_started_with_258_sentences : 
  janice_starting_sentences 6 20 15 18 40 536 = 258 := by
  sorry

end NUMINAMATH_CALUDE_janice_starting_sentences_janice_started_with_258_sentences_l129_12918


namespace NUMINAMATH_CALUDE_negation_equivalence_l129_12999

def original_statement (a b : ℤ) : Prop :=
  (¬(Odd a ∧ Odd b)) → Even (a + b)

def proposed_negation (a b : ℤ) : Prop :=
  (¬(Odd a ∧ Odd b)) → ¬Even (a + b)

def correct_negation (a b : ℤ) : Prop :=
  ¬(Odd a ∧ Odd b) ∧ ¬Even (a + b)

theorem negation_equivalence :
  ∀ a b : ℤ, ¬(original_statement a b) ↔ correct_negation a b :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l129_12999


namespace NUMINAMATH_CALUDE_product_evaluation_l129_12946

theorem product_evaluation :
  (3 - 4) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 + 4) = -5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l129_12946


namespace NUMINAMATH_CALUDE_ghee_mixture_problem_l129_12998

theorem ghee_mixture_problem (Q : ℝ) : 
  (0.6 * Q = Q - 0.4 * Q) →  -- 60% is pure ghee, 40% is vanaspati
  (0.4 * Q = 0.2 * (Q + 10)) →  -- After adding 10 kg, vanaspati is 20%
  Q = 10 := by
sorry

end NUMINAMATH_CALUDE_ghee_mixture_problem_l129_12998


namespace NUMINAMATH_CALUDE_eighth_fibonacci_term_l129_12947

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_fibonacci_term :
  fibonacci 7 = 21 :=
by sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_term_l129_12947


namespace NUMINAMATH_CALUDE_interest_problem_l129_12920

/-- Proves that given the conditions of the interest problem, the sum is 700 --/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 7.5) * 12 / 100 - P * R * 12 / 100 = 630) → P = 700 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l129_12920


namespace NUMINAMATH_CALUDE_prime_sum_91_l129_12923

theorem prime_sum_91 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_sum : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_91_l129_12923


namespace NUMINAMATH_CALUDE_drawBalls_18_4_l129_12905

/-- The number of balls in the bin -/
def n : ℕ := 18

/-- The number of balls to be drawn -/
def k : ℕ := 4

/-- The number of ways to draw k balls from n balls, 
    where the first ball is returned and the rest are not -/
def drawBalls (n k : ℕ) : ℕ := n * n * (n - 1) * (n - 2)

/-- Theorem stating that drawing 4 balls from 18 balls, 
    where the first ball is returned and the rest are not, 
    can be done in 87984 ways -/
theorem drawBalls_18_4 : drawBalls n k = 87984 := by sorry

end NUMINAMATH_CALUDE_drawBalls_18_4_l129_12905


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l129_12974

theorem condition_necessary_not_sufficient : 
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → (x - 1) * (x + 2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l129_12974


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l129_12924

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 90 → speed2 = 55 → (speed1 + speed2) / 2 = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l129_12924


namespace NUMINAMATH_CALUDE_x0_value_l129_12972

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 3) → x₀ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l129_12972


namespace NUMINAMATH_CALUDE_tan_276_equals_96_l129_12978

theorem tan_276_equals_96 : 
  ∃! (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (276 * π / 180) :=
by
  sorry

end NUMINAMATH_CALUDE_tan_276_equals_96_l129_12978


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l129_12941

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling 8 six-sided dice and getting an equal number of even and odd results -/
def prob_equal_even_odd : ℚ := 35/128

theorem equal_even_odd_probability :
  (Nat.choose num_dice (num_dice / 2)) * (prob_even ^ num_dice) = prob_equal_even_odd := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l129_12941


namespace NUMINAMATH_CALUDE_cos_range_theorem_l129_12980

theorem cos_range_theorem (ω : ℝ) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc 0 (π / 3), 3 * Real.sin (ω * x) + 4 * Real.cos (ω * x) ∈ Set.Icc 4 5) →
  (∃ y ∈ Set.Icc (7 / 25) (4 / 5), y = Real.cos (π * ω / 3)) ∧
  (∀ y, y = Real.cos (π * ω / 3) → y ∈ Set.Icc (7 / 25) (4 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_cos_range_theorem_l129_12980


namespace NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l129_12902

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: The value of b that satisfies h(b) = 0 is 7/5 -/
theorem h_zero_at_seven_fifths : ∃ b : ℝ, h b = 0 ∧ b = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l129_12902


namespace NUMINAMATH_CALUDE_unique_covering_100x100_l129_12944

/-- A frame is the border of a square in a grid. -/
structure Frame where
  side_length : ℕ

/-- A covering is a list of non-overlapping frames that completely cover a square grid. -/
structure Covering where
  frames : List Frame
  non_overlapping : ∀ (f1 f2 : Frame), f1 ∈ frames → f2 ∈ frames → f1 ≠ f2 → 
    f1.side_length ≠ f2.side_length
  complete : ∀ (n : ℕ), n ∈ List.range 50 → 
    ∃ (f : Frame), f ∈ frames ∧ f.side_length = 100 - 2 * n

/-- The theorem states that there is a unique covering of a 100×100 grid with 50 frames. -/
theorem unique_covering_100x100 : 
  ∃! (c : Covering), c.frames.length = 50 ∧ 
    (∀ (f : Frame), f ∈ c.frames → f.side_length ≤ 100 ∧ f.side_length % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_covering_100x100_l129_12944


namespace NUMINAMATH_CALUDE_ant_path_distance_l129_12969

/-- Represents the rectangle in which the ant walks --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the ant's path --/
structure AntPath where
  start : ℝ  -- Distance from the nearest corner to the starting point X
  angle : ℝ  -- Angle of the path with respect to the sides of the rectangle

/-- Theorem stating the conditions and the result to be proved --/
theorem ant_path_distance (rect : Rectangle) (path : AntPath) :
  rect.width = 18 ∧ 
  rect.height = 150 ∧ 
  path.angle = 45 ∧ 
  path.start ≥ 0 ∧ 
  path.start ≤ rect.width ∧
  (∃ n : ℕ, n * rect.width = rect.height / 2) →
  path.start = 3 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_distance_l129_12969


namespace NUMINAMATH_CALUDE_orthogonal_projection_area_range_l129_12979

/-- Regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  base_side : ℝ
  lateral_edge : ℝ

/-- Orthogonal projection area of a regular quadrangular pyramid -/
def orthogonal_projection_area (p : RegularQuadrangularPyramid) (angle : ℝ) : ℝ :=
  sorry

/-- Theorem: Range of orthogonal projection area -/
theorem orthogonal_projection_area_range 
  (p : RegularQuadrangularPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_edge = Real.sqrt 6) : 
  ∀ angle, 2 ≤ orthogonal_projection_area p angle ∧ 
           orthogonal_projection_area p angle ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_orthogonal_projection_area_range_l129_12979


namespace NUMINAMATH_CALUDE_six_objects_three_parts_l129_12964

/-- The number of ways to partition n indistinguishable objects into at most k non-empty parts -/
def partition_count (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to partition 6 indistinguishable objects into at most 3 non-empty parts -/
theorem six_objects_three_parts : partition_count 6 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_objects_three_parts_l129_12964


namespace NUMINAMATH_CALUDE_binary_111111_equals_63_l129_12931

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111111_equals_63 :
  binary_to_decimal [true, true, true, true, true, true] = 63 := by
  sorry

end NUMINAMATH_CALUDE_binary_111111_equals_63_l129_12931


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l129_12933

theorem complex_number_magnitude (a : ℝ) :
  let i : ℂ := Complex.I
  let z : ℂ := (a - i)^2
  (∃ b : ℝ, z = b * i) → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l129_12933


namespace NUMINAMATH_CALUDE_percent_gain_is_588_l129_12965

-- Define the number of sheep bought and sold
def total_sheep : ℕ := 900
def sold_first : ℕ := 850
def sold_second : ℕ := 50

-- Define the cost and revenue functions
def cost (price_per_sheep : ℚ) : ℚ := price_per_sheep * total_sheep
def revenue_first (price_per_sheep : ℚ) : ℚ := cost price_per_sheep
def revenue_second (price_per_sheep : ℚ) : ℚ := 
  (revenue_first price_per_sheep / sold_first) * sold_second

-- Define the total revenue and profit
def total_revenue (price_per_sheep : ℚ) : ℚ := 
  revenue_first price_per_sheep + revenue_second price_per_sheep
def profit (price_per_sheep : ℚ) : ℚ := 
  total_revenue price_per_sheep - cost price_per_sheep

-- Define the percent gain
def percent_gain (price_per_sheep : ℚ) : ℚ := 
  (profit price_per_sheep / cost price_per_sheep) * 100

-- Theorem statement
theorem percent_gain_is_588 (price_per_sheep : ℚ) :
  percent_gain price_per_sheep = 52.94 / 9 :=
by sorry

end NUMINAMATH_CALUDE_percent_gain_is_588_l129_12965


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l129_12911

/-- The number of boys in a class with given height information -/
theorem number_of_boys_in_class :
  ∀ (n : ℕ) (initial_avg real_avg wrong_height actual_height : ℝ),
  initial_avg = 180 →
  wrong_height = 166 →
  actual_height = 106 →
  real_avg = 178 →
  initial_avg * n - (wrong_height - actual_height) = real_avg * n →
  n = 30 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l129_12911


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l129_12927

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l129_12927


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l129_12997

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  (1/x + 1/y + 1/z) ≥ 3 := by
  sorry

theorem min_reciprocal_sum_attained (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (1/a + 1/b + 1/c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l129_12997


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l129_12912

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | -x^2 + 6*x - 8 > 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioc 2 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l129_12912


namespace NUMINAMATH_CALUDE_smallest_divisor_for_perfect_square_l129_12983

theorem smallest_divisor_for_perfect_square (n : ℕ) (h : n = 2880) :
  ∃ (d : ℕ), d > 0 ∧ d.min = 5 ∧ (∃ (k : ℕ), n / d = k * k) ∧
  ∀ (x : ℕ), 0 < x ∧ x < d → ¬∃ (m : ℕ), n / x = m * m :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_for_perfect_square_l129_12983


namespace NUMINAMATH_CALUDE_fib_recurrence_l129_12957

/-- Fibonacci sequence defined as the number of ways to represent n as an ordered sum of ones and twos -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The Fibonacci sequence satisfies the recurrence relation F_n = F_{n-1} + F_{n-2} for n ≥ 2 -/
theorem fib_recurrence (n : ℕ) (h : n ≥ 2) : fib n = fib (n - 1) + fib (n - 2) := by
  sorry

#check fib_recurrence

end NUMINAMATH_CALUDE_fib_recurrence_l129_12957


namespace NUMINAMATH_CALUDE_test_subjects_count_l129_12910

def number_of_colors : ℕ := 8
def colors_per_code : ℕ := 4
def unidentified_subjects : ℕ := 19

theorem test_subjects_count :
  (number_of_colors.choose colors_per_code) + unidentified_subjects = 299 :=
by sorry

end NUMINAMATH_CALUDE_test_subjects_count_l129_12910


namespace NUMINAMATH_CALUDE_clothing_profit_l129_12991

theorem clothing_profit (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  price = 180 ∧ 
  profit_percent = 20 ∧ 
  loss_percent = 10 → 
  (2 * price) - (price / (1 + profit_percent / 100) + price / (1 - loss_percent / 100)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_clothing_profit_l129_12991


namespace NUMINAMATH_CALUDE_base_representation_digit_difference_l129_12940

theorem base_representation_digit_difference : 
  let n : ℕ := 1234
  let base_5_digits := (Nat.log n 5).succ
  let base_9_digits := (Nat.log n 9).succ
  base_5_digits - base_9_digits = 1 := by
sorry

end NUMINAMATH_CALUDE_base_representation_digit_difference_l129_12940
