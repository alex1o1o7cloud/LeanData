import Mathlib

namespace NUMINAMATH_CALUDE_square_angles_equal_l1857_185737

-- Define a rectangle
structure Rectangle where
  angles : Fin 4 → ℝ

-- Define a square as a special case of rectangle
structure Square extends Rectangle

-- State that all angles in a rectangle are equal
axiom rectangle_angles_equal (r : Rectangle) : ∀ i j : Fin 4, r.angles i = r.angles j

-- State that a square is a rectangle
axiom square_is_rectangle (s : Square) : Rectangle

-- Theorem to prove
theorem square_angles_equal (s : Square) : ∀ i j : Fin 4, s.angles i = s.angles j := by
  sorry

end NUMINAMATH_CALUDE_square_angles_equal_l1857_185737


namespace NUMINAMATH_CALUDE_sequence_properties_l1857_185774

def sequence_a (n : ℕ+) : ℚ := sorry

def S (n : ℕ+) : ℚ := sorry

def T (n : ℕ+) : ℚ := sorry

theorem sequence_properties :
  (∀ n : ℕ+, 3 * S n = (n + 2) * sequence_a n) ∧
  sequence_a 1 = 2 →
  (∀ n : ℕ+, sequence_a n = n + 1) ∧
  ∃ M : Set ℕ+, Set.Infinite M ∧ ∀ n ∈ M, |T n - 1| < (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1857_185774


namespace NUMINAMATH_CALUDE_third_day_is_tuesday_or_wednesday_l1857_185738

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  days : ℕ
  startDay : DayOfWeek
  mondayCount : ℕ
  tuesdayCount : ℕ
  wednesdayCount : ℕ
  sundayCount : ℕ

/-- Given the properties of a month, determine the day of the week for the 3rd day -/
def thirdDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Theorem stating that the 3rd day of the month is either Tuesday or Wednesday -/
theorem third_day_is_tuesday_or_wednesday (m : Month) 
  (h1 : m.mondayCount = m.wednesdayCount + 1)
  (h2 : m.tuesdayCount = m.sundayCount) :
  (thirdDayOfMonth m = DayOfWeek.Tuesday) ∨ (thirdDayOfMonth m = DayOfWeek.Wednesday) :=
  sorry

end NUMINAMATH_CALUDE_third_day_is_tuesday_or_wednesday_l1857_185738


namespace NUMINAMATH_CALUDE_compound_ratio_proof_l1857_185766

theorem compound_ratio_proof (x y : ℝ) (y_nonzero : y ≠ 0) :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) * (4 / 5) * (x / y) = x / (17.5 * y) :=
by sorry

end NUMINAMATH_CALUDE_compound_ratio_proof_l1857_185766


namespace NUMINAMATH_CALUDE_equation_solution_l1857_185739

theorem equation_solution : 
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1857_185739


namespace NUMINAMATH_CALUDE_sarah_trip_distance_l1857_185707

/-- Represents Sarah's trip to the airport -/
structure AirportTrip where
  initial_speed : ℝ
  initial_time : ℝ
  final_speed : ℝ
  early_arrival : ℝ
  total_distance : ℝ

/-- The theorem stating the total distance of Sarah's trip -/
theorem sarah_trip_distance (trip : AirportTrip) : 
  trip.initial_speed = 15 ∧ 
  trip.initial_time = 1 ∧ 
  trip.final_speed = 60 ∧ 
  trip.early_arrival = 0.5 →
  trip.total_distance = 45 := by
  sorry

#check sarah_trip_distance

end NUMINAMATH_CALUDE_sarah_trip_distance_l1857_185707


namespace NUMINAMATH_CALUDE_doris_babysitting_earnings_l1857_185716

/-- Represents the problem of calculating how many weeks Doris needs to earn enough for her monthly expenses --/
theorem doris_babysitting_earnings :
  let hourly_rate : ℚ := 20
  let weekday_hours : ℚ := 3
  let saturday_hours : ℚ := 5
  let monthly_expense : ℚ := 1200
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  let weeks_needed := monthly_expense / weekly_earnings
  weeks_needed = 3 := by sorry

end NUMINAMATH_CALUDE_doris_babysitting_earnings_l1857_185716


namespace NUMINAMATH_CALUDE_trigonometric_identity_angle_relation_l1857_185701

-- Part 1
theorem trigonometric_identity :
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) -
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2 := by sorry

-- Part 2
theorem angle_relation (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.tan (α - β) = 1 / 2) (h6 : Real.tan β = -1 / 7) :
  2 * α - β = -3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_angle_relation_l1857_185701


namespace NUMINAMATH_CALUDE_expression_evaluation_l1857_185782

theorem expression_evaluation : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1857_185782


namespace NUMINAMATH_CALUDE_daniels_noodles_l1857_185713

def noodles_problem (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : Prop :=
  initial = given_away + remaining

theorem daniels_noodles : ∃ initial : ℕ, noodles_problem initial 12 54 ∧ initial = 66 := by
  sorry

end NUMINAMATH_CALUDE_daniels_noodles_l1857_185713


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l1857_185709

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def P : Set ℕ := {3,4,5}
def Q : Set ℕ := {1,3,6}

theorem complement_intersection_equals_set : (U \ P) ∩ (U \ Q) = {2,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l1857_185709


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_l1857_185749

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let v_sphere := (4 / 3) * π * r_sphere^3
  let h_cylinder := Real.sqrt (r_sphere^2 - r_cylinder^2)
  let v_cylinder := π * r_cylinder^2 * h_cylinder
  v_sphere - v_cylinder = ((1372 - 48 * Real.sqrt 33) / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_l1857_185749


namespace NUMINAMATH_CALUDE_card_game_combinations_l1857_185700

theorem card_game_combinations : Nat.choose 52 10 = 158200242220 := by sorry

end NUMINAMATH_CALUDE_card_game_combinations_l1857_185700


namespace NUMINAMATH_CALUDE_mary_savings_problem_l1857_185755

theorem mary_savings_problem (S : ℝ) (x : ℝ) (h1 : S > 0) (h2 : 0 ≤ x ∧ x ≤ 1) : 
  12 * x * S = 7 * (1 - x) * S → (1 - x) = 12 / 19 := by
  sorry

end NUMINAMATH_CALUDE_mary_savings_problem_l1857_185755


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1857_185714

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ ∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1857_185714


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1857_185760

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1857_185760


namespace NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l1857_185710

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_with_55_divisors :
  ∀ n : ℕ, number_of_divisors n = 55 → n ≥ 3^4 * 2^10 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l1857_185710


namespace NUMINAMATH_CALUDE_find_x_value_l1857_185756

theorem find_x_value (x : ℝ) : (15 : ℝ)^x * 8^3 / 256 = 450 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l1857_185756


namespace NUMINAMATH_CALUDE_C2H6_C3H8_impossible_l1857_185759

-- Define the heat released by combustion of 1 mol of each hydrocarbon
def heat_CH4 : ℝ := 889.5
def heat_C2H6 : ℝ := 1558.35
def heat_C2H4 : ℝ := 1409.6
def heat_C2H2 : ℝ := 1298.35
def heat_C3H8 : ℝ := 2217.8

-- Define the total heat released by the mixture
def total_heat : ℝ := 3037.6

-- Define the number of moles in the mixture
def total_moles : ℝ := 2

-- Theorem to prove that C₂H₆ and C₃H₈ combination is impossible
theorem C2H6_C3H8_impossible : 
  ¬(∃ (x y : ℝ), x + y = total_moles ∧ 
                  x * heat_C2H6 + y * heat_C3H8 = total_heat ∧
                  x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_C2H6_C3H8_impossible_l1857_185759


namespace NUMINAMATH_CALUDE_expand_product_l1857_185720

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1857_185720


namespace NUMINAMATH_CALUDE_inequality_proof_l1857_185795

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) < Real.sqrt (3*a) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1857_185795


namespace NUMINAMATH_CALUDE_max_quarters_proof_l1857_185771

/-- Represents the number of coins of each type --/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.nickels * 5 + coins.dimes * 10

/-- Checks if the coin count satisfies the problem conditions --/
def isValidCount (coins : CoinCount) : Prop :=
  coins.quarters = coins.nickels ∧ coins.dimes * 2 = coins.quarters

/-- The maximum number of quarters possible given the conditions --/
def maxQuarters : ℕ := 11

theorem max_quarters_proof :
  ∀ coins : CoinCount,
    isValidCount coins →
    totalValue coins = 400 →
    coins.quarters ≤ maxQuarters :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_proof_l1857_185771


namespace NUMINAMATH_CALUDE_clock_rings_count_l1857_185717

def clock_rings (hour : ℕ) : Bool :=
  if hour ≤ 12 then
    hour % 2 = 1
  else
    hour % 4 = 1

def total_rings : ℕ :=
  (List.range 24).filter (λ h => clock_rings (h + 1)) |>.length

theorem clock_rings_count : total_rings = 10 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_count_l1857_185717


namespace NUMINAMATH_CALUDE_diamond_computation_l1857_185719

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.five
  | Element.one, Element.five => Element.four
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.five
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.three
  | Element.two, Element.five => Element.two
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.two
  | Element.three, Element.four => Element.one
  | Element.three, Element.five => Element.five
  | Element.four, Element.one => Element.five
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.four
  | Element.five, Element.two => Element.three
  | Element.five, Element.three => Element.five
  | Element.five, Element.four => Element.two
  | Element.five, Element.five => Element.one

theorem diamond_computation :
  diamond (diamond Element.four Element.five) (diamond Element.one Element.three) = Element.two :=
by sorry

end NUMINAMATH_CALUDE_diamond_computation_l1857_185719


namespace NUMINAMATH_CALUDE_expression_simplification_l1857_185748

theorem expression_simplification (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ),
    (15 : ℝ) * d + 16 + 17 * d^2 + (3 : ℝ) * d + 2 = (a : ℝ) * d + b + (c : ℝ) * d^2 ∧
    a + b + c = 53 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1857_185748


namespace NUMINAMATH_CALUDE_derivative_reciprocal_sum_sqrt_derivative_reciprocal_sum_sqrt_value_l1857_185787

theorem derivative_reciprocal_sum_sqrt (x : ℝ) (h : x ≠ 1) :
  (fun x => 2 / (1 - x)) = (fun x => 1 / (1 - Real.sqrt x) + 1 / (1 + Real.sqrt x)) :=
by sorry

theorem derivative_reciprocal_sum_sqrt_value (x : ℝ) (h : x ≠ 1) :
  deriv (fun x => 2 / (1 - x)) x = 2 / (1 - x)^2 :=
by sorry

end NUMINAMATH_CALUDE_derivative_reciprocal_sum_sqrt_derivative_reciprocal_sum_sqrt_value_l1857_185787


namespace NUMINAMATH_CALUDE_package_weight_problem_l1857_185752

theorem package_weight_problem (a b c : ℝ) 
  (hab : a + b = 108)
  (hbc : b + c = 132)
  (hca : c + a = 138) :
  a + b + c = 189 ∧ a ≥ 40 ∧ b ≥ 40 ∧ c ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_problem_l1857_185752


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_neg_three_l1857_185711

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem decreasing_function_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 4 → f a x₁ > f a x₂) → a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_neg_three_l1857_185711


namespace NUMINAMATH_CALUDE_log_equation_solution_l1857_185768

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9 →
  x = 2^(54/11) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1857_185768


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1857_185702

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- The point P with coordinates (-3, a^2 + 1) lies in the second quadrant for any real number a. -/
theorem point_in_second_quadrant (a : ℝ) : second_quadrant (-3, a^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1857_185702


namespace NUMINAMATH_CALUDE_fruit_basket_cost_l1857_185777

/-- Represents the contents of a fruit basket --/
structure FruitBasket where
  bananas : ℕ
  apples : ℕ
  oranges : ℕ
  kiwis : ℕ
  strawberries : ℕ
  avocados : ℕ
  grapes : ℕ
  melons : ℕ

/-- Represents the prices of individual fruits --/
structure FruitPrices where
  banana : ℚ
  apple : ℚ
  orange : ℚ
  kiwi : ℚ
  strawberry_dozen : ℚ
  avocado : ℚ
  grapes_half_bunch : ℚ
  melon : ℚ

/-- Calculates the total cost of the fruit basket after all discounts --/
def calculateTotalCost (basket : FruitBasket) (prices : FruitPrices) : ℚ :=
  sorry

/-- Theorem stating that the total cost of the given fruit basket is $35.43 --/
theorem fruit_basket_cost :
  let basket : FruitBasket := {
    bananas := 4,
    apples := 3,
    oranges := 4,
    kiwis := 2,
    strawberries := 24,
    avocados := 2,
    grapes := 1,
    melons := 1
  }
  let prices : FruitPrices := {
    banana := 1,
    apple := 2,
    orange := 3/2,
    kiwi := 5/4,
    strawberry_dozen := 4,
    avocado := 3,
    grapes_half_bunch := 2,
    melon := 7/2
  }
  calculateTotalCost basket prices = 3543/100 :=
sorry

end NUMINAMATH_CALUDE_fruit_basket_cost_l1857_185777


namespace NUMINAMATH_CALUDE_ellipse_foci_l1857_185740

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 100 = 1

/-- The coordinates of a focus of the ellipse -/
def focus_coordinate : ℝ × ℝ := (0, 6)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  (ellipse_equation (focus_coordinate.1) (focus_coordinate.2) ∧
   ellipse_equation (focus_coordinate.1) (-focus_coordinate.2)) ∧
  (∀ x y : ℝ, ellipse_equation x y →
    (x^2 + y^2 < focus_coordinate.1^2 + focus_coordinate.2^2 ∨
     x^2 + y^2 = focus_coordinate.1^2 + focus_coordinate.2^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l1857_185740


namespace NUMINAMATH_CALUDE_xyz_sum_l1857_185781

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = 47)
  (h2 : y.val * z.val + x.val = 47)
  (h3 : x.val * z.val + y.val = 47) :
  x.val + y.val + z.val = 48 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sum_l1857_185781


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l1857_185791

/-- Given a triangle ABC with points E on BC and G on AB, and Q the intersection of AE and CG,
    if AQ:QE = 3:2 and GQ:QC = 2:3, then AG:GB = 1:2 -/
theorem triangle_ratio_theorem (A B C E G Q : ℝ × ℝ) : 
  (E.1 - B.1) / (C.1 - B.1) = (E.2 - B.2) / (C.2 - B.2) →  -- E is on BC
  (G.1 - A.1) / (B.1 - A.1) = (G.2 - A.2) / (B.2 - A.2) →  -- G is on AB
  ∃ (t : ℝ), Q = (1 - t) • A + t • E ∧                     -- Q is on AE
             Q = (1 - t) • C + t • G →                     -- Q is on CG
  (Q.1 - A.1) / (E.1 - Q.1) = 3 / 2 →                      -- AQ:QE = 3:2
  (G.1 - Q.1) / (Q.1 - C.1) = 2 / 3 →                      -- GQ:QC = 2:3
  (G.1 - A.1) / (B.1 - G.1) = 1 / 2 :=                     -- AG:GB = 1:2
by sorry


end NUMINAMATH_CALUDE_triangle_ratio_theorem_l1857_185791


namespace NUMINAMATH_CALUDE_f_is_monotonic_and_odd_l1857_185784

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_is_monotonic_and_odd :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_f_is_monotonic_and_odd_l1857_185784


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1857_185792

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y, x < y ∧ y < 0 → x^2 > y^2) ∧ 
  (∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1857_185792


namespace NUMINAMATH_CALUDE_quadratic_properties_l1857_185730

/-- Quadratic function definition -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + b - 1

/-- Point definition -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem quadratic_properties (b : ℝ) :
  (∀ x, f b x = 0 ↔ x = -1 ∨ x = 1 - b) ∧
  (b < 2 → ∀ m, ∃ xp, xp = m - b + 1 ∧ 
    ∃ yp, Point.mk xp yp ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y > 0}) ∧
  (b = -3 → ∃ c, ∀ m n, 
    (∃ xp yp xq yq, 
      Point.mk xp yp ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y > 0} ∧
      Point.mk xq yq ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y < 0} ∧
      (yp - 0) / (xp - (-1)) = m ∧
      (yq - 0) / (xq - (-1)) = n) →
    m * n = c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1857_185730


namespace NUMINAMATH_CALUDE_married_fraction_l1857_185723

theorem married_fraction (total : ℕ) (women_fraction : ℚ) (max_unmarried_women : ℕ) :
  total = 80 →
  women_fraction = 1/4 →
  max_unmarried_women = 20 →
  (total - max_unmarried_women : ℚ) / total = 3/4 := by
sorry

end NUMINAMATH_CALUDE_married_fraction_l1857_185723


namespace NUMINAMATH_CALUDE_base_ten_to_base_five_158_l1857_185790

/-- Converts a natural number to its base 5 representation -/
def toBaseFive (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Checks if a list of digits is a valid base 5 representation -/
def isValidBaseFive (digits : List ℕ) : Prop :=
  digits.all (· < 5) ∧ digits ≠ []

theorem base_ten_to_base_five_158 :
  let base_five_repr := toBaseFive 158
  isValidBaseFive base_five_repr ∧ base_five_repr = [1, 1, 3, 3] := by sorry

end NUMINAMATH_CALUDE_base_ten_to_base_five_158_l1857_185790


namespace NUMINAMATH_CALUDE_probability_at_least_one_chooses_23_l1857_185772

def num_students : ℕ := 4
def num_questions : ℕ := 2

theorem probability_at_least_one_chooses_23 :
  (1 : ℚ) - (1 / num_questions) ^ num_students = 15 / 16 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_chooses_23_l1857_185772


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1857_185722

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1857_185722


namespace NUMINAMATH_CALUDE_other_birds_percentage_l1857_185785

/-- Represents the composition of birds in the Goshawk-Eurasian Nature Reserve -/
structure BirdReserve where
  total : ℝ
  hawk_percent : ℝ
  paddyfield_warbler_percent_of_nonhawk : ℝ
  kingfisher_to_paddyfield_warbler_ratio : ℝ

/-- Theorem stating the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
theorem other_birds_percentage (reserve : BirdReserve) 
  (h1 : reserve.hawk_percent = 0.3)
  (h2 : reserve.paddyfield_warbler_percent_of_nonhawk = 0.4)
  (h3 : reserve.kingfisher_to_paddyfield_warbler_ratio = 0.25)
  (h4 : reserve.total > 0) :
  let hawk_count := reserve.hawk_percent * reserve.total
  let nonhawk_count := reserve.total - hawk_count
  let paddyfield_warbler_count := reserve.paddyfield_warbler_percent_of_nonhawk * nonhawk_count
  let kingfisher_count := reserve.kingfisher_to_paddyfield_warbler_ratio * paddyfield_warbler_count
  let other_count := reserve.total - (hawk_count + paddyfield_warbler_count + kingfisher_count)
  (other_count / reserve.total) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_other_birds_percentage_l1857_185785


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l1857_185733

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l1857_185733


namespace NUMINAMATH_CALUDE_total_frogs_caught_l1857_185786

def initial_frogs : ℕ := 5
def additional_frogs : ℕ := 2

theorem total_frogs_caught :
  initial_frogs + additional_frogs = 7 := by sorry

end NUMINAMATH_CALUDE_total_frogs_caught_l1857_185786


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1857_185734

theorem distinct_prime_factors_count : ∃ (p : ℕ → Prop), 
  (∀ n, p n ↔ Nat.Prime n) ∧ 
  (∃ (S : Finset ℕ), 
    (∀ x ∈ S, p x) ∧ 
    Finset.card S = 7 ∧
    (∀ q, p q → q ∣ ((87 * 89 * 91 + 1) * 93) ↔ q ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1857_185734


namespace NUMINAMATH_CALUDE_wire_service_reporters_l1857_185731

theorem wire_service_reporters (total_reporters : ℝ) 
  (local_politics_reporters : ℝ) (politics_reporters : ℝ) :
  local_politics_reporters = 0.12 * total_reporters →
  local_politics_reporters = 0.6 * politics_reporters →
  total_reporters - politics_reporters = 0.8 * total_reporters :=
by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l1857_185731


namespace NUMINAMATH_CALUDE_add_particular_number_to_34_l1857_185724

theorem add_particular_number_to_34 (x : ℝ) (h : 96 / x = 6) : 34 + x = 50 := by
  sorry

end NUMINAMATH_CALUDE_add_particular_number_to_34_l1857_185724


namespace NUMINAMATH_CALUDE_max_integer_value_of_function_l1857_185793

theorem max_integer_value_of_function (x : ℝ) : 
  (4*x^2 + 8*x + 5 ≠ 0) → 
  ∃ (y : ℤ), y = 17 ∧ ∀ (z : ℤ), z ≤ (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 5) → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_max_integer_value_of_function_l1857_185793


namespace NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l1857_185750

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem: The 100th odd positive integer is 199 -/
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_hundredthOddPositiveInteger_l1857_185750


namespace NUMINAMATH_CALUDE_closest_to_fraction_l1857_185783

def options : List ℝ := [0.3, 3, 30, 300, 3000]

theorem closest_to_fraction (x : ℝ) (h : x = 613 / 0.307) :
  ∃ y ∈ options, ∀ z ∈ options, |x - y| ≤ |x - z| :=
sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l1857_185783


namespace NUMINAMATH_CALUDE_two_red_cards_selection_count_l1857_185796

/-- Represents a deck of cards with a specific structure -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Calculates the number of ways to select two different cards from red suits -/
def select_two_red_cards (d : Deck) : ℕ :=
  let red_cards := d.red_suits * d.cards_per_suit
  red_cards * (red_cards - 1)

/-- The main theorem to be proved -/
theorem two_red_cards_selection_count (d : Deck) 
  (h1 : d.total_cards = 36)
  (h2 : d.suits = 3)
  (h3 : d.cards_per_suit = 12)
  (h4 : d.red_suits = 2)
  (h5 : d.black_suits = 1)
  (h6 : d.red_suits + d.black_suits = d.suits) :
  select_two_red_cards d = 552 := by
  sorry


end NUMINAMATH_CALUDE_two_red_cards_selection_count_l1857_185796


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1857_185769

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1857_185769


namespace NUMINAMATH_CALUDE_min_value_theorem_l1857_185741

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 9*x + 81/x^4 ≥ 19 ∧
  (x^2 + 9*x + 81/x^4 = 19 ↔ x = 3) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1857_185741


namespace NUMINAMATH_CALUDE_increase_and_subtract_l1857_185704

theorem increase_and_subtract (initial : ℝ) (increase_percent : ℝ) (subtract : ℝ) : 
  initial = 75 → increase_percent = 150 → subtract = 40 →
  initial * (1 + increase_percent / 100) - subtract = 147.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_and_subtract_l1857_185704


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1857_185779

def is_first_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (Real.pi / 2) + 2 * k * Real.pi

def is_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < (Real.pi / 2) + 2 * k * Real.pi) ∨
           ((Real.pi + 2 * k * Real.pi < α) ∧ (α < (3 * Real.pi / 2) + 2 * k * Real.pi))

theorem half_angle_quadrant (α : Real) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1857_185779


namespace NUMINAMATH_CALUDE_condition_A_sufficient_not_necessary_l1857_185727

/-- Condition A: a > 1 and b > 1 -/
def condition_A (a b : ℝ) : Prop := a > 1 ∧ b > 1

/-- Condition B: a + b > 2 and ab > 1 -/
def condition_B (a b : ℝ) : Prop := a + b > 2 ∧ a * b > 1

theorem condition_A_sufficient_not_necessary :
  (∀ a b : ℝ, condition_A a b → condition_B a b) ∧
  (∃ a b : ℝ, condition_B a b ∧ ¬condition_A a b) :=
by sorry

end NUMINAMATH_CALUDE_condition_A_sufficient_not_necessary_l1857_185727


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l1857_185715

theorem point_movement_on_number_line (m : ℝ) : 
  (|m - 3 + 5| = 6) → (m = -8 ∨ m = 4) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l1857_185715


namespace NUMINAMATH_CALUDE_factorial_sum_quotient_l1857_185735

theorem factorial_sum_quotient : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_quotient_l1857_185735


namespace NUMINAMATH_CALUDE_symmetric_f_inequality_solution_l1857_185706

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - a|

-- State the theorem
theorem symmetric_f_inequality_solution (a : ℝ) :
  (∀ x : ℝ, f a x = f a (2 - x)) →
  {x : ℝ | f a (x^2 - 3) < f a (x - 1)} = {x : ℝ | -3 < x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_f_inequality_solution_l1857_185706


namespace NUMINAMATH_CALUDE_log_equality_condition_l1857_185762

theorem log_equality_condition (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q ≠ 2) :
  Real.log p + Real.log q = Real.log (2 * p + 3 * q) ↔ p = (3 * q) / (q - 2) :=
sorry

end NUMINAMATH_CALUDE_log_equality_condition_l1857_185762


namespace NUMINAMATH_CALUDE_set_equality_l1857_185736

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem set_equality : M = {1} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1857_185736


namespace NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l1857_185773

def sum_even_integers (a b : ℕ) : ℕ :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  let n := (last_even - first_even) / 2 + 1
  n * (first_even + last_even) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  (last_even - first_even) / 2 + 1

theorem gcd_sum_and_count_even_integers :
  Nat.gcd (sum_even_integers 13 63) (count_even_integers 13 63) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l1857_185773


namespace NUMINAMATH_CALUDE_expression_evaluation_l1857_185728

/-- Given x = -2 and y = 1/2, prove that 2(x^2y + xy^2) - 2(x^2y - 1) - 3xy^2 - 2 evaluates to 1/2 -/
theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1/2) :
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1857_185728


namespace NUMINAMATH_CALUDE_student_D_most_stable_l1857_185753

/-- Represents a student in the long jump training --/
inductive Student
| A
| B
| C
| D

/-- Returns the variance of a student's performance --/
def variance (s : Student) : ℝ :=
  match s with
  | Student.A => 2.1
  | Student.B => 3.5
  | Student.C => 9
  | Student.D => 0.7

/-- Determines if a student has the most stable performance --/
def has_most_stable_performance (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

/-- Theorem stating that student D has the most stable performance --/
theorem student_D_most_stable :
  has_most_stable_performance Student.D :=
sorry

end NUMINAMATH_CALUDE_student_D_most_stable_l1857_185753


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l1857_185725

/-- Given a quadratic equation x^2 + (m - 3)x + m = 0 where m is a real number,
    if one root is greater than 1 and the other root is less than 1,
    then m < 1 -/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 1 ∧ r₂ < 1 ∧ 
    r₁^2 + (m - 3) * r₁ + m = 0 ∧ 
    r₂^2 + (m - 3) * r₂ + m = 0) → 
  m < 1 := by
sorry


end NUMINAMATH_CALUDE_quadratic_root_condition_l1857_185725


namespace NUMINAMATH_CALUDE_cube_root_of_product_l1857_185754

theorem cube_root_of_product (a : ℕ) : a^3 = 21 * 35 * 45 * 35 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l1857_185754


namespace NUMINAMATH_CALUDE_probability_not_red_card_l1857_185794

theorem probability_not_red_card (odds_red : ℚ) (h : odds_red = 5/7) :
  1 - odds_red / (1 + odds_red) = 7/12 := by sorry

end NUMINAMATH_CALUDE_probability_not_red_card_l1857_185794


namespace NUMINAMATH_CALUDE_target_hit_probability_l1857_185765

theorem target_hit_probability (p_a p_b : ℝ) (h_a : p_a = 0.6) (h_b : p_b = 0.5) :
  let p_hit := 1 - (1 - p_a) * (1 - p_b)
  (p_a / p_hit) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1857_185765


namespace NUMINAMATH_CALUDE_extremum_at_negative_three_l1857_185763

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

-- Theorem statement
theorem extremum_at_negative_three (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extremum_at_negative_three_l1857_185763


namespace NUMINAMATH_CALUDE_count_is_nine_l1857_185775

/-- A function that returns the count of valid 4-digit numbers greater than 1000 
    that can be formed using the digits of 2012 -/
def count_valid_numbers : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating that the count of valid numbers is 9 -/
theorem count_is_nine : count_valid_numbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_is_nine_l1857_185775


namespace NUMINAMATH_CALUDE_dream_car_gas_consumption_l1857_185789

/-- Calculates the total gas consumption for a car over two days -/
def total_gas_consumption (consumption_rate : ℝ) (miles_today : ℝ) (miles_tomorrow : ℝ) : ℝ :=
  consumption_rate * (miles_today + miles_tomorrow)

theorem dream_car_gas_consumption :
  let consumption_rate : ℝ := 4
  let miles_today : ℝ := 400
  let miles_tomorrow : ℝ := miles_today + 200
  total_gas_consumption consumption_rate miles_today miles_tomorrow = 4000 := by
sorry

end NUMINAMATH_CALUDE_dream_car_gas_consumption_l1857_185789


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1857_185767

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c p : ℝ) (x₀ y₀ : ℝ) : 
  a > 0 → b > 0 → p > 0 → x₀ > 0 → y₀ > 0 →
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →  -- hyperbola equation
  y₀ = (b / a) * x₀ →  -- point on asymptote
  x₀^2 + y₀^2 = c^2 →  -- MF₁ ⊥ MF₂
  y₀^2 = 2 * p * x₀ →  -- parabola equation
  c / a = 2 + Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1857_185767


namespace NUMINAMATH_CALUDE_parallelogram_height_l1857_185721

theorem parallelogram_height (area base height : ℝ) : 
  area = 96 ∧ base = 12 ∧ area = base * height → height = 8 := by sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1857_185721


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1857_185729

def U : Set Int := {-4, -2, -1, 0, 2, 4, 5, 6, 7}
def A : Set Int := {-2, 0, 4, 6}
def B : Set Int := {-1, 2, 4, 6, 7}

theorem intersection_complement_equality : A ∩ (U \ B) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1857_185729


namespace NUMINAMATH_CALUDE_cosine_difference_simplification_l1857_185761

theorem cosine_difference_simplification (α β γ : ℝ) :
  Real.cos (α - β) * Real.cos (β - γ) - Real.sin (α - β) * Real.sin (β - γ) = Real.cos (α - γ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_simplification_l1857_185761


namespace NUMINAMATH_CALUDE_problem_statement_l1857_185712

open Set

def p (m : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (m * x^2 - m * x + 1)

def q (m : ℝ) : Prop := ∃ x₀ ∈ Icc 0 3, x₀^2 - 2*x₀ - m ≥ 0

theorem problem_statement (m : ℝ) :
  (q m ↔ m ∈ Iic 3) ∧
  ((p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Iio 0 ∪ Ioo 3 4) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1857_185712


namespace NUMINAMATH_CALUDE_pizza_slices_sold_l1857_185788

/-- Proves that the number of small slices sold is 2000 -/
theorem pizza_slices_sold (small_price large_price : ℕ) 
  (total_slices total_revenue : ℕ) (h1 : small_price = 150) 
  (h2 : large_price = 250) (h3 : total_slices = 5000) 
  (h4 : total_revenue = 1050000) : 
  ∃ (small_slices large_slices : ℕ),
    small_slices + large_slices = total_slices ∧
    small_price * small_slices + large_price * large_slices = total_revenue ∧
    small_slices = 2000 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_sold_l1857_185788


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1857_185745

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle :=
  (a b c : ℝ)

/-- Checks if two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

theorem similar_triangles_side_length 
  (FGH IJK : Triangle)
  (h_similar : are_similar FGH IJK)
  (h_GH : FGH.c = 30)
  (h_FG : FGH.a = 24)
  (h_IJ : IJK.a = 20) :
  IJK.c = 25 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1857_185745


namespace NUMINAMATH_CALUDE_arithmetic_sequence_diff_l1857_185780

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_diff (a₁ d : ℤ) :
  |arithmetic_sequence a₁ d 105 - arithmetic_sequence a₁ d 100| = 40 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_diff_l1857_185780


namespace NUMINAMATH_CALUDE_percent_of_y_l1857_185747

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 / y) / 20 + (3 / y) / 10) / y = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l1857_185747


namespace NUMINAMATH_CALUDE_unique_solution_l1857_185757

def pizza_problem (boys girls : ℕ) : Prop :=
  let day1_consumption := 7 * boys + 3 * girls
  let day2_consumption := 6 * boys + 2 * girls
  (49 ≤ day1_consumption) ∧ (day1_consumption ≤ 59) ∧
  (49 ≤ day2_consumption) ∧ (day2_consumption ≤ 59)

theorem unique_solution : ∃! (b g : ℕ), pizza_problem b g ∧ b = 8 ∧ g = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1857_185757


namespace NUMINAMATH_CALUDE_min_sum_at_6_l1857_185798

/-- Arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -11
  sum_5_6 : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * n - 24)

/-- Theorem stating that S_n reaches its minimum value when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → S seq 6 ≤ S seq n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l1857_185798


namespace NUMINAMATH_CALUDE_circle_intersection_existence_l1857_185746

theorem circle_intersection_existence :
  ∃ n : ℝ, 0 < n ∧ n < 2 ∧
  (∃ x y : ℝ, x^2 + y^2 - 2*n*x + 2*n*y + 2*n^2 - 8 = 0 ∧
              (x+1)^2 + (y-1)^2 = 2) ∧
  (∃ x' y' : ℝ, x' ≠ x ∨ y' ≠ y ∧
              x'^2 + y'^2 - 2*n*x' + 2*n*y' + 2*n^2 - 8 = 0 ∧
              (x'+1)^2 + (y'-1)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_existence_l1857_185746


namespace NUMINAMATH_CALUDE_quadratic_function_increasing_on_positive_x_l1857_185751

theorem quadratic_function_increasing_on_positive_x 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = x₁^2 - 1) 
  (h2 : y₂ = x₂^2 - 1) 
  (h3 : 0 < x₁) 
  (h4 : x₁ < x₂) : 
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_increasing_on_positive_x_l1857_185751


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1857_185742

/-- The minimum distance from the origin to a point on the line 3x + 4y - 20 = 0 is 4 -/
theorem min_distance_to_line : 
  ∀ a b : ℝ, (3 * a + 4 * b = 20) → (∀ x y : ℝ, (3 * x + 4 * y = 20) → (a^2 + b^2 ≤ x^2 + y^2)) → 
  Real.sqrt (a^2 + b^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1857_185742


namespace NUMINAMATH_CALUDE_solve_system_l1857_185718

theorem solve_system (A B C D : ℤ) 
  (eq1 : A + C = 15)
  (eq2 : A - B = 1)
  (eq3 : C + C = A)
  (eq4 : B - D = 2)
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  D = 7 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1857_185718


namespace NUMINAMATH_CALUDE_cross_area_is_two_l1857_185732

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- Represents a triangle in the grid --/
structure Triangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- The center point of the 4x4 grid --/
def gridCenter : GridPoint := { x := 2, y := 2 }

/-- A function to create a midpoint on the grid edge --/
def gridEdgeMidpoint (x y : ℚ) : GridPoint := { x := x, y := y }

/-- The four triangles forming the cross shape --/
def crossTriangles : List Triangle := [
  { v1 := gridCenter, v2 := gridEdgeMidpoint 0 2, v3 := gridEdgeMidpoint 2 0 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 2 4, v3 := gridEdgeMidpoint 4 2 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 0 2, v3 := gridEdgeMidpoint 2 4 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 2 0, v3 := gridEdgeMidpoint 4 2 }
]

/-- Calculate the area of a single triangle --/
def triangleArea (t : Triangle) : ℚ := 0.5

/-- Calculate the total area of the cross shape --/
def crossArea : ℚ := (crossTriangles.map triangleArea).sum

/-- The theorem stating that the area of the cross shape is 2 --/
theorem cross_area_is_two : crossArea = 2 := by sorry

end NUMINAMATH_CALUDE_cross_area_is_two_l1857_185732


namespace NUMINAMATH_CALUDE_b_cubed_is_zero_l1857_185703

theorem b_cubed_is_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_cubed_is_zero_l1857_185703


namespace NUMINAMATH_CALUDE_sum_properties_l1857_185797

theorem sum_properties (a b : ℤ) (ha : 4 ∣ a) (hb : 8 ∣ b) : 
  Even (a + b) ∧ (4 ∣ (a + b)) ∧ ¬(∀ (a b : ℤ), 4 ∣ a → 8 ∣ b → 8 ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_sum_properties_l1857_185797


namespace NUMINAMATH_CALUDE_intersection_M_N_l1857_185743

def M : Set ℝ := {x | x ≥ -2}
def N : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1857_185743


namespace NUMINAMATH_CALUDE_daily_rate_proof_l1857_185799

/-- The daily rental rate for Jason's carriage house -/
def daily_rate : ℝ := 40

/-- The total cost for Eric's rental -/
def total_cost : ℝ := 800

/-- The number of days Eric is renting -/
def rental_days : ℕ := 20

/-- Theorem stating that the daily rate multiplied by the number of rental days equals the total cost -/
theorem daily_rate_proof : daily_rate * (rental_days : ℝ) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_daily_rate_proof_l1857_185799


namespace NUMINAMATH_CALUDE_farmer_corn_rows_l1857_185778

/-- Given a farmer's crop scenario, prove the number of corn stalk rows. -/
theorem farmer_corn_rows (C : ℕ) : 
  (C * 9 + 5 * 30 = 240) → C = 10 := by
  sorry

end NUMINAMATH_CALUDE_farmer_corn_rows_l1857_185778


namespace NUMINAMATH_CALUDE_cd_length_l1857_185764

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (N : Point)
  (M : Point)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (ABNM : Quadrilateral)
  (C : Point)
  (A' : Point)
  (D : Point)
  (x : ℝ)
  (AB : ℝ)
  (AM : ℝ)
  (AC : ℝ)

/-- Main theorem: CD = AC * cos(x) -/
theorem cd_length
  (setup : ProblemSetup)
  (h1 : setup.ABNM.A.y = setup.ABNM.B.y) -- AB is initially horizontal
  (h2 : setup.ABNM.N.y = setup.ABNM.M.y) -- MN is horizontal
  (h3 : setup.C.y = setup.ABNM.M.y) -- C is on line MN
  (h4 : setup.A'.x - setup.ABNM.B.x = setup.AB * Real.cos setup.x) -- A' position after rotation
  (h5 : setup.A'.y - setup.ABNM.B.y = setup.AB * Real.sin setup.x)
  (h6 : Real.sqrt ((setup.A'.x - setup.ABNM.B.x)^2 + (setup.A'.y - setup.ABNM.B.y)^2) = setup.AB) -- A'B = AB
  : Real.sqrt ((setup.D.x - setup.C.x)^2 + (setup.D.y - setup.C.y)^2) = setup.AC * Real.cos setup.x :=
sorry

end NUMINAMATH_CALUDE_cd_length_l1857_185764


namespace NUMINAMATH_CALUDE_power_2020_l1857_185758

theorem power_2020 (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^(m-4*n) = 4/81) : 
  2020^n = 2020 := by
  sorry

end NUMINAMATH_CALUDE_power_2020_l1857_185758


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1857_185705

/-- Hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) -/
structure Hyperbola (a b : ℝ) : Prop where
  a_pos : a > 0
  b_pos : b > 0

/-- Line l with equation y = 2x - 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1 - 2}

/-- The asymptote of the hyperbola C -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (b / a) * p.1 ∨ p.2 = -(b / a) * p.1}

/-- Predicate to check if a line is parallel to an asymptote -/
def is_parallel_to_asymptote (h : Hyperbola a b) : Prop :=
  b / a = 2

/-- Predicate to check if a line passes through a vertex of the hyperbola -/
def passes_through_vertex (h : Hyperbola a b) : Prop :=
  (1, 0) ∈ Line

/-- The distance from the focus of the hyperbola to its asymptote -/
def focus_asymptote_distance (h : Hyperbola a b) : ℝ := b

/-- Main theorem -/
theorem hyperbola_focus_asymptote_distance 
  (h : Hyperbola a b) 
  (parallel : is_parallel_to_asymptote h) 
  (through_vertex : passes_through_vertex h) : 
  focus_asymptote_distance h = 2 := by
    sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1857_185705


namespace NUMINAMATH_CALUDE_frog_jump_theorem_l1857_185776

/-- A regular polygon with 2n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) :=
  (n_ge_two : n ≥ 2)

/-- A configuration of frogs on the vertices of a regular polygon -/
structure FrogConfiguration (n : ℕ) :=
  (polygon : RegularPolygon n)
  (frogs : Fin (2*n) → Bool)

/-- A jumping method for the frogs -/
def JumpingMethod (n : ℕ) := Fin (2*n) → Bool

/-- Check if a line segment passes through the center of the circle -/
def passes_through_center (n : ℕ) (v1 v2 : Fin (2*n)) : Prop :=
  ∃ k : ℕ, v2 = v1 + n ∨ v1 = v2 + n

/-- The main theorem -/
theorem frog_jump_theorem (n : ℕ) :
  (∃ (config : FrogConfiguration n) (jump : JumpingMethod n),
    ∀ v1 v2 : Fin (2*n),
      v1 ≠ v2 →
      config.frogs v1 = true →
      config.frogs v2 = true →
      ¬passes_through_center n v1 v2) ↔
  n % 4 = 2 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_theorem_l1857_185776


namespace NUMINAMATH_CALUDE_right_prism_surface_area_l1857_185744

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  /-- Length of parallel sides AB and CD -/
  ab_cd : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Area of the diagonal cross-section -/
  diagonal_area : ℝ
  /-- Condition: AB and CD are equal -/
  ab_eq_cd : ab_cd > 0
  /-- Condition: BC is positive -/
  bc_pos : bc > 0
  /-- Condition: AD is positive -/
  ad_pos : ad > 0
  /-- Condition: AD > BC (trapezoid property) -/
  ad_gt_bc : ad > bc
  /-- Condition: Diagonal area is positive -/
  diagonal_area_pos : diagonal_area > 0

/-- Total surface area of the right prism -/
def totalSurfaceArea (p : RightPrism) : ℝ :=
  sorry

/-- Theorem: The total surface area of the specified right prism is 906 -/
theorem right_prism_surface_area :
  ∀ (p : RightPrism),
    p.ab_cd = 13 ∧ p.bc = 11 ∧ p.ad = 21 ∧ p.diagonal_area = 180 →
    totalSurfaceArea p = 906 := by
  sorry

end NUMINAMATH_CALUDE_right_prism_surface_area_l1857_185744


namespace NUMINAMATH_CALUDE_set_relationship_l1857_185708

def P : Set ℝ := {y | ∃ x, y = -x^2 + 1}
def Q : Set ℝ := {y | ∃ x, y = 2^x}

theorem set_relationship : ∀ y : ℝ, y > 1 → y ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_set_relationship_l1857_185708


namespace NUMINAMATH_CALUDE_gravitational_force_in_orbit_l1857_185770

/-- Gravitational force calculation -/
theorem gravitational_force_in_orbit 
  (surface_distance : ℝ) 
  (orbit_distance : ℝ) 
  (surface_force : ℝ) 
  (h1 : surface_distance = 6000)
  (h2 : orbit_distance = 36000)
  (h3 : surface_force = 800)
  (h4 : ∀ (d f : ℝ), f * d^2 = surface_force * surface_distance^2) :
  ∃ (orbit_force : ℝ), 
    orbit_force * orbit_distance^2 = surface_force * surface_distance^2 ∧ 
    orbit_force = 1 / 45 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_in_orbit_l1857_185770


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_two_alpha_l1857_185726

theorem cos_two_pi_thirds_minus_two_alpha (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.cos ((2 * π) / 3 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_two_alpha_l1857_185726
