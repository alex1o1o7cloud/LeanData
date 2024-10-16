import Mathlib

namespace NUMINAMATH_CALUDE_water_added_to_reach_new_ratio_l3448_344895

-- Define the initial mixture volume
def initial_volume : ℝ := 80

-- Define the initial ratio of milk to water
def initial_milk_ratio : ℝ := 7
def initial_water_ratio : ℝ := 3

-- Define the amount of water evaporated
def evaporated_water : ℝ := 8

-- Define the new ratio of milk to water
def new_milk_ratio : ℝ := 5
def new_water_ratio : ℝ := 4

-- Theorem to prove
theorem water_added_to_reach_new_ratio :
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let water_after_evaporation := initial_water - evaporated_water
  let x := (((new_water_ratio / new_milk_ratio) * initial_milk) - water_after_evaporation)
  x = 28.8 := by sorry

end NUMINAMATH_CALUDE_water_added_to_reach_new_ratio_l3448_344895


namespace NUMINAMATH_CALUDE_cinema_lineup_ways_l3448_344817

def number_of_people : ℕ := 8
def number_of_windows : ℕ := 2

theorem cinema_lineup_ways :
  (2 ^ number_of_people) * (Nat.factorial number_of_people) = 10321920 := by
  sorry

end NUMINAMATH_CALUDE_cinema_lineup_ways_l3448_344817


namespace NUMINAMATH_CALUDE_paper_towel_savings_l3448_344857

/-- Calculates the percent of savings per roll when buying a package of paper towels
    compared to buying individual rolls. -/
def percent_savings (package_price : ℚ) (package_size : ℕ) (individual_price : ℚ) : ℚ :=
  let package_price_per_roll := package_price / package_size
  let savings_per_roll := individual_price - package_price_per_roll
  (savings_per_roll / individual_price) * 100

/-- Theorem stating that the percent of savings for a 12-roll package priced at $9
    compared to buying 12 rolls individually at $1 each is 25%. -/
theorem paper_towel_savings :
  percent_savings 9 12 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l3448_344857


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3448_344815

/-- Given that the solution set of ax^2 - 1999x + b > 0 is {x | -3 < x < -1},
    prove that the solution set of ax^2 + 1999x + b > 0 is {x | 1 < x < 3} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, (a * x^2 - 1999 * x + b > 0) ↔ (-3 < x ∧ x < -1)) :
  ∀ x : ℝ, (a * x^2 + 1999 * x + b > 0) ↔ (1 < x ∧ x < 3) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3448_344815


namespace NUMINAMATH_CALUDE_solution_implies_m_equals_one_l3448_344821

theorem solution_implies_m_equals_one (x y m : ℝ) : 
  x = 2 → y = -1 → m * x - y = 3 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_equals_one_l3448_344821


namespace NUMINAMATH_CALUDE_equation_root_l3448_344827

theorem equation_root : ∃ x : ℝ, (18 / (x^3 - 8) - 2 / (x - 2) = 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_l3448_344827


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3448_344823

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0
  h_asymptote : b / a = Real.sqrt 3
  h_vertex : a = 1

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hyperbola equation -/
def hyperbola_eq (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The line equation -/
def line_eq (m : ℝ) (p : Point) : Prop :=
  p.y = p.x + m

/-- Theorem stating the properties of the hyperbola and its intersection with a line -/
theorem hyperbola_properties (h : Hyperbola) (m : ℝ) (A B M : Point) 
    (h_distinct : A ≠ B)
    (h_intersect_A : hyperbola_eq h A ∧ line_eq m A)
    (h_intersect_B : hyperbola_eq h B ∧ line_eq m B)
    (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2)
    (h_nonzero : M.x ≠ 0) :
  (h.a = 1 ∧ h.b = Real.sqrt 3) ∧ M.y / M.x = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3448_344823


namespace NUMINAMATH_CALUDE_athlete_B_most_stable_l3448_344842

-- Define the athletes
inductive Athlete : Type
  | A : Athlete
  | B : Athlete
  | C : Athlete

-- Define the variance for each athlete
def variance (a : Athlete) : ℝ :=
  match a with
  | Athlete.A => 0.78
  | Athlete.B => 0.2
  | Athlete.C => 1.28

-- Define the concept of most stable performance
def most_stable (a : Athlete) : Prop :=
  ∀ b : Athlete, variance a ≤ variance b

-- Theorem statement
theorem athlete_B_most_stable :
  most_stable Athlete.B :=
sorry

end NUMINAMATH_CALUDE_athlete_B_most_stable_l3448_344842


namespace NUMINAMATH_CALUDE_problem_statement_l3448_344873

theorem problem_statement : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3448_344873


namespace NUMINAMATH_CALUDE_spot_fraction_l3448_344806

theorem spot_fraction (rover_spots cisco_spots granger_spots total_spots : ℕ) 
  (f : ℚ) : 
  rover_spots = 46 →
  granger_spots = 5 * cisco_spots →
  granger_spots + cisco_spots = total_spots →
  total_spots = 108 →
  cisco_spots = f * 46 - 5 →
  f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spot_fraction_l3448_344806


namespace NUMINAMATH_CALUDE_interval_bound_l3448_344818

-- Define the functions
def f (x : ℝ) := x^4 - 2*x^2
def g (x : ℝ) := 4*x^2 - 8
def h (t x : ℝ) := 4*(t^3 - t)*x - 3*t^4 + 2*t^2

-- State the theorem
theorem interval_bound 
  (t : ℝ) 
  (ht : 0 < |t| ∧ |t| ≤ Real.sqrt 2) 
  (m n : ℝ) 
  (hmn : m ≤ n ∧ Set.Icc m n ⊆ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)) 
  (h_inequality : ∀ x ∈ Set.Icc m n, f x ≥ h t x ∧ h t x ≥ g x) : 
  n - m ≤ Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_interval_bound_l3448_344818


namespace NUMINAMATH_CALUDE_total_percent_decrease_l3448_344802

def year1_decrease : ℝ := 0.20
def year2_decrease : ℝ := 0.10
def year3_decrease : ℝ := 0.15

def compound_decrease (initial_value : ℝ) : ℝ :=
  initial_value * (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease)

theorem total_percent_decrease (initial_value : ℝ) (h : initial_value > 0) :
  (initial_value - compound_decrease initial_value) / initial_value = 0.388 := by
  sorry

end NUMINAMATH_CALUDE_total_percent_decrease_l3448_344802


namespace NUMINAMATH_CALUDE_smallest_share_is_five_l3448_344864

/-- Represents the distribution of coins among three children --/
structure CoinDistribution where
  one_franc : ℕ
  five_franc : ℕ
  fifty_cent : ℕ

/-- Checks if the distribution satisfies the problem conditions --/
def valid_distribution (d : CoinDistribution) : Prop :=
  d.one_franc + 5 * d.five_franc + (d.fifty_cent : ℚ) / 2 = 100 ∧
  d.fifty_cent = d.one_franc / 9

/-- Calculates the smallest share among the three children --/
def smallest_share (d : CoinDistribution) : ℚ :=
  min (min (d.one_franc : ℚ) (5 * d.five_franc : ℚ)) ((d.fifty_cent : ℚ) / 2)

/-- Theorem stating the smallest possible share is 5 francs --/
theorem smallest_share_is_five :
  ∀ d : CoinDistribution, valid_distribution d → smallest_share d = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_share_is_five_l3448_344864


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l3448_344851

theorem rectangle_side_lengths :
  ∀ x y : ℝ,
  x > 0 →
  y > 0 →
  y = 2 * x →
  x * y = 2 * (x + y) →
  (x = 3 ∧ y = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l3448_344851


namespace NUMINAMATH_CALUDE_xy_value_l3448_344836

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56/9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3448_344836


namespace NUMINAMATH_CALUDE_last_three_average_l3448_344879

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / 6 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l3448_344879


namespace NUMINAMATH_CALUDE_revolver_problem_l3448_344807

/-- Probability of the gun firing on any given shot -/
def p : ℚ := 1 / 6

/-- Probability of the gun not firing on any given shot -/
def q : ℚ := 1 - p

/-- The probability that the gun will fire while A is holding it -/
noncomputable def prob_A_fires : ℚ := sorry

theorem revolver_problem : prob_A_fires = 6 / 11 := by sorry

end NUMINAMATH_CALUDE_revolver_problem_l3448_344807


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3448_344813

theorem distinct_prime_factors_count : ∃ (p : ℕ → Prop), 
  (∀ n, p n ↔ Nat.Prime n) ∧ 
  (∃ (S : Finset ℕ), 
    (∀ x ∈ S, p x) ∧ 
    Finset.card S = 7 ∧
    (∀ q, p q → q ∣ ((87 * 89 * 91 + 1) * 93) ↔ q ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3448_344813


namespace NUMINAMATH_CALUDE_third_day_is_tuesday_or_wednesday_l3448_344811

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

end NUMINAMATH_CALUDE_third_day_is_tuesday_or_wednesday_l3448_344811


namespace NUMINAMATH_CALUDE_third_number_from_lcm_hcf_l3448_344863

/-- Given three positive integers with known LCM and HCF, prove the third number -/
theorem third_number_from_lcm_hcf (A B C : ℕ+) : 
  A = 36 → B = 44 → Nat.lcm A (Nat.lcm B C) = 792 → Nat.gcd A (Nat.gcd B C) = 12 → C = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_number_from_lcm_hcf_l3448_344863


namespace NUMINAMATH_CALUDE_system_solutions_l3448_344866

/-- The first equation of the system -/
def equation1 (x y z : ℝ) : Prop :=
  5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0

/-- The second equation of the system -/
def equation2 (x y z : ℝ) : Prop :=
  49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 = 0

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  equation1 x y z ∧ equation2 x y z

/-- The theorem stating that the given points are the only solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3448_344866


namespace NUMINAMATH_CALUDE_logarithmic_inequality_l3448_344898

theorem logarithmic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (Real.log a)/((a-b)*(a-c)) + (Real.log b)/((b-c)*(b-a)) + (Real.log c)/((c-a)*(c-b)) < 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_inequality_l3448_344898


namespace NUMINAMATH_CALUDE_large_pepperoni_has_14_slices_l3448_344872

/-- The number of slices in a large pepperoni pizza -/
def large_pepperoni_slices (total_eaten : ℕ) (total_left : ℕ) (small_cheese_slices : ℕ) : ℕ :=
  total_eaten + total_left - small_cheese_slices

/-- Theorem stating that the large pepperoni pizza has 14 slices -/
theorem large_pepperoni_has_14_slices : 
  large_pepperoni_slices 18 4 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_large_pepperoni_has_14_slices_l3448_344872


namespace NUMINAMATH_CALUDE_power_multiplication_l3448_344825

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3448_344825


namespace NUMINAMATH_CALUDE_point_c_coordinates_l3448_344867

/-- Point with x and y coordinates -/
structure Point where
  x : ℚ
  y : ℚ

/-- Distance between two points -/
def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Check if a point is on a line segment -/
def isOnSegment (p q r : Point) : Prop :=
  distance p r + distance r q = distance p q

theorem point_c_coordinates :
  let a : Point := ⟨-3, 2⟩
  let b : Point := ⟨5, 10⟩
  ∀ c : Point,
    isOnSegment a c b →
    distance a c = 2 * distance c b →
    c = ⟨7/3, 22/3⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l3448_344867


namespace NUMINAMATH_CALUDE_equation_solution_l3448_344830

theorem equation_solution :
  ∃ x : ℚ, (4 * x^2 + 3 * x + 2) / (x + 2) = 4 * x + 3 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3448_344830


namespace NUMINAMATH_CALUDE_additional_amount_needed_for_free_shipping_l3448_344854

def free_shipping_threshold : ℝ := 50.00

def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

def first_two_discount : ℝ := 0.25
def total_discount : ℝ := 0.10

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price : ℝ :=
  discounted_price book1_price first_two_discount +
  discounted_price book2_price first_two_discount +
  book3_price + book4_price

def final_price : ℝ :=
  discounted_price total_price total_discount

theorem additional_amount_needed_for_free_shipping :
  free_shipping_threshold - final_price = 13.10 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_for_free_shipping_l3448_344854


namespace NUMINAMATH_CALUDE_office_distance_l3448_344862

/-- The distance to the office in kilometers -/
def distance : ℝ := sorry

/-- The time it takes to reach the office on time in hours -/
def on_time : ℝ := sorry

/-- Condition 1: At 10 kmph, the person arrives 10 minutes late -/
axiom condition_1 : distance = 10 * (on_time + 1/6)

/-- Condition 2: At 15 kmph, the person arrives 10 minutes early -/
axiom condition_2 : distance = 15 * (on_time - 1/6)

/-- Theorem: The distance to the office is 10 kilometers -/
theorem office_distance : distance = 10 := by sorry

end NUMINAMATH_CALUDE_office_distance_l3448_344862


namespace NUMINAMATH_CALUDE_floor_sqrt_24_squared_l3448_344819

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_24_squared_l3448_344819


namespace NUMINAMATH_CALUDE_abs_equation_solution_l3448_344832

theorem abs_equation_solution :
  ∀ x : ℝ, |2*x + 6| = 3*x - 1 ↔ x = 7 := by sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l3448_344832


namespace NUMINAMATH_CALUDE_no_real_roots_iff_m_zero_l3448_344850

theorem no_real_roots_iff_m_zero (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_m_zero_l3448_344850


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_expression_l3448_344868

/-- Represents a cubic polynomial of the form ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluates the cubic polynomial at a given x -/
def CubicPolynomial.evaluate (p : CubicPolynomial) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The specific cubic polynomial f(x) = 2x^3 - 3x^2 + 5x - 7 -/
def f : CubicPolynomial :=
  { a := 2, b := -3, c := 5, d := -7 }

theorem cubic_polynomial_coefficient_expression :
  16 * f.a - 9 * f.b + 3 * f.c - 2 * f.d = 88 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_expression_l3448_344868


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3448_344805

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ k ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3448_344805


namespace NUMINAMATH_CALUDE_animals_left_in_barn_l3448_344847

theorem animals_left_in_barn (pigs cows sold : ℕ) 
  (h1 : pigs = 156)
  (h2 : cows = 267)
  (h3 : sold = 115) :
  pigs + cows - sold = 308 :=
by sorry

end NUMINAMATH_CALUDE_animals_left_in_barn_l3448_344847


namespace NUMINAMATH_CALUDE_smallest_b_value_l3448_344858

theorem smallest_b_value (a b : ℕ+) 
  (h1 : a.val - b.val = 10)
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ x : ℕ+, 2 ≤ x.val → x.val < b.val → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3448_344858


namespace NUMINAMATH_CALUDE_kombucha_refund_bottles_l3448_344809

/-- Prove that Henry can buy 6 additional bottles of kombucha after one year using his cash refund from recycling. -/
theorem kombucha_refund_bottles (bottles_per_month : ℕ) (bottle_cost : ℚ) (refund_per_bottle : ℚ) (months_per_year : ℕ) :
  bottles_per_month = 15 →
  bottle_cost = 3 →
  refund_per_bottle = 1/10 →
  months_per_year = 12 →
  (bottles_per_month * months_per_year * refund_per_bottle) / bottle_cost = 6 := by
  sorry

#eval (15 * 12 * (1/10 : ℚ)) / 3

end NUMINAMATH_CALUDE_kombucha_refund_bottles_l3448_344809


namespace NUMINAMATH_CALUDE_prove_callys_colored_shirts_l3448_344893

/-- The number of colored shirts Cally washed -/
def callys_colored_shirts : ℕ := 5

theorem prove_callys_colored_shirts :
  let callys_other_clothes : ℕ := 10 + 7 + 6 -- white shirts + shorts + pants
  let dannys_clothes : ℕ := 6 + 8 + 10 + 6 -- white shirts + colored shirts + shorts + pants
  let total_clothes : ℕ := 58
  callys_colored_shirts = total_clothes - (callys_other_clothes + dannys_clothes) :=
by sorry

end NUMINAMATH_CALUDE_prove_callys_colored_shirts_l3448_344893


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3448_344848

/-- Given a circle described by the equation x^2 + y^2 - 2x + 4y = 0,
    prove that its center coordinates are (1, -2) and its radius is √5. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧
    radius = Real.sqrt 5 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 2*x + 4*y = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3448_344848


namespace NUMINAMATH_CALUDE_logarithm_simplification_l3448_344841

theorem logarithm_simplification
  (p q r s t u : ℝ)
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0) :
  Real.log (p / q) - Real.log (q / r) + Real.log (r / s) + Real.log ((s * t) / (p * u)) = Real.log (t / u) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l3448_344841


namespace NUMINAMATH_CALUDE_triangle_side_length_l3448_344890

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a = Real.sqrt 3 →
  Real.sin B = 1 / 2 →
  C = π / 6 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3448_344890


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3448_344896

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 400 ∧ 
  total_time = 30 ∧ 
  second_half_speed = 10 →
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l3448_344896


namespace NUMINAMATH_CALUDE_probability_two_nondefective_pens_l3448_344875

/-- Given a box of 12 pens with 3 defective pens, prove that the probability
    of selecting 2 non-defective pens at random without replacement is 6/11. -/
theorem probability_two_nondefective_pens (total_pens : Nat) (defective_pens : Nat)
    (h1 : total_pens = 12)
    (h2 : defective_pens = 3) :
    (total_pens - defective_pens : ℚ) / total_pens *
    ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_nondefective_pens_l3448_344875


namespace NUMINAMATH_CALUDE_noah_class_size_l3448_344878

theorem noah_class_size (n : ℕ) (noah_rank_best : ℕ) (noah_rank_worst : ℕ) 
  (h1 : noah_rank_best = 40)
  (h2 : noah_rank_worst = 40)
  (h3 : n = noah_rank_best + noah_rank_worst - 1) :
  n = 79 := by
  sorry

end NUMINAMATH_CALUDE_noah_class_size_l3448_344878


namespace NUMINAMATH_CALUDE_min_tests_for_16_people_l3448_344880

/-- Represents the number of people in the group -/
def total_people : ℕ := 16

/-- Represents the number of infected people -/
def infected_people : ℕ := 1

/-- The function that calculates the minimum number of tests required -/
def min_tests (n : ℕ) : ℕ := Nat.log2 n + 1

/-- Theorem stating that the minimum number of tests for 16 people is 4 -/
theorem min_tests_for_16_people :
  min_tests total_people = 4 :=
sorry

end NUMINAMATH_CALUDE_min_tests_for_16_people_l3448_344880


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_equal_intercept_line_standard_form_l3448_344849

/-- A line passing through point (-3, 4) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-3, 4) -/
  passes_through_point : slope * (-3) + y_intercept = 4
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = slope * y_intercept

/-- The equation of the line is either 4x + 3y = 0 or x + y = 1 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = -4/3 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 1) := by
  sorry

/-- The line equation in standard form is either 4x + 3y = 0 or x + y = 1 -/
theorem equal_intercept_line_standard_form (l : EqualInterceptLine) :
  (∃ (k : ℝ), k ≠ 0 ∧ 4*k*l.slope + 3*k = 0 ∧ k*l.y_intercept = 0) ∨
  (∃ (k : ℝ), k ≠ 0 ∧ k*l.slope + k = 0 ∧ k*l.y_intercept = k) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_equal_intercept_line_standard_form_l3448_344849


namespace NUMINAMATH_CALUDE_parallelogram_height_l3448_344824

/-- Given a parallelogram with area 576 square cm and base 12 cm, its height is 48 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 ∧ base = 12 ∧ area = base * height → height = 48 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3448_344824


namespace NUMINAMATH_CALUDE_product_equality_l3448_344828

theorem product_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a * b + b * c + a * c) * ((a * b)⁻¹ + (b * c)⁻¹ + (a * c)⁻¹) = 
  (a * b + b * c + a * c)^2 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3448_344828


namespace NUMINAMATH_CALUDE_curve_is_two_lines_l3448_344846

-- Define the equation of the curve
def curve_equation (x y : ℝ) : Prop := x^2 + x*y = x

-- Theorem stating that the curve equation represents two lines
theorem curve_is_two_lines :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, curve_equation x y ↔ (y = m₁ * x + b₁ ∨ y = m₂ * x + b₂)) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_lines_l3448_344846


namespace NUMINAMATH_CALUDE_mistake_percentage_l3448_344803

theorem mistake_percentage (n : ℕ) (x : ℕ) : 
  n > 0 ∧ x > 0 ∧ x ≤ n ∧
  (x - 1 : ℚ) / n = 24 / 100 ∧
  (x - 1 : ℚ) / (n - 1) = 25 / 100 →
  (x : ℚ) / n = 28 / 100 :=
by sorry

end NUMINAMATH_CALUDE_mistake_percentage_l3448_344803


namespace NUMINAMATH_CALUDE_division_problem_l3448_344889

theorem division_problem (total : ℚ) (a b c : ℚ) 
  (h1 : total = 544)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : c = 384 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3448_344889


namespace NUMINAMATH_CALUDE_product_equality_l3448_344844

theorem product_equality (h : 213 * 16 = 3408) : 0.16 * 2.13 = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3448_344844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3448_344808

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₄ = 5 and a₅ + a₆ = 11, then a₇ = 6 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 5) 
    (h_a5_a6 : a 5 + a 6 = 11) : 
  a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3448_344808


namespace NUMINAMATH_CALUDE_condition_relationship_l3448_344877

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧ 
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l3448_344877


namespace NUMINAMATH_CALUDE_monday_distance_l3448_344820

/-- Debby's jogging distances over three days -/
structure JoggingDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  total : ℝ

/-- The jogging distances satisfy the given conditions -/
def satisfies_conditions (d : JoggingDistances) : Prop :=
  d.tuesday = 5 ∧ d.wednesday = 9 ∧ d.total = 16 ∧ d.monday + d.tuesday + d.wednesday = d.total

/-- Theorem: Debby jogged 2 kilometers on Monday -/
theorem monday_distance (d : JoggingDistances) (h : satisfies_conditions d) : d.monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_monday_distance_l3448_344820


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3448_344885

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hR : R = 0.6 * P)
  (hN : N = 0.5 * R) :
  M / N = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3448_344885


namespace NUMINAMATH_CALUDE_leftover_apples_for_ivan_l3448_344840

/-- Given a number of initial apples and mini pies, calculate the number of leftover apples -/
def leftover_apples (initial_apples : ℕ) (mini_pies : ℕ) : ℕ :=
  initial_apples - (mini_pies / 2)

/-- Theorem: Given 48 initial apples and 24 mini pies, each requiring 1/2 an apple, 
    the number of leftover apples is 36 -/
theorem leftover_apples_for_ivan : leftover_apples 48 24 = 36 := by
  sorry

end NUMINAMATH_CALUDE_leftover_apples_for_ivan_l3448_344840


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3448_344814

/-- Properties of a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_is_thrice_breadth : length = 3 * breadth
  area_formula : area = length * breadth

/-- Theorem: The breadth of a rectangular plot with given properties is 11 meters -/
theorem rectangular_plot_breadth (plot : RectangularPlot) 
  (h : plot.area = 363) : plot.breadth = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3448_344814


namespace NUMINAMATH_CALUDE_average_income_calculation_l3448_344886

theorem average_income_calculation (total_customers : ℕ) 
  (wealthy_customers : ℕ) (other_customers : ℕ) 
  (wealthy_avg_income : ℝ) (other_avg_income : ℝ) :
  total_customers = wealthy_customers + other_customers →
  wealthy_customers = 10 →
  other_customers = 40 →
  wealthy_avg_income = 55000 →
  other_avg_income = 42500 →
  (wealthy_customers * wealthy_avg_income + other_customers * other_avg_income) / total_customers = 45000 :=
by sorry

end NUMINAMATH_CALUDE_average_income_calculation_l3448_344886


namespace NUMINAMATH_CALUDE_sarah_reads_40_wpm_l3448_344843

/-- Calculates Sarah's reading speed in words per minute -/
def sarah_reading_speed (words_per_page : ℕ) (pages_per_book : ℕ) (reading_hours : ℕ) (num_books : ℕ) : ℕ :=
  let total_words := words_per_page * pages_per_book * num_books
  let total_minutes := reading_hours * 60
  total_words / total_minutes

/-- Proves that Sarah's reading speed is 40 words per minute given the problem conditions -/
theorem sarah_reads_40_wpm : sarah_reading_speed 100 80 20 6 = 40 := by
  sorry

#eval sarah_reading_speed 100 80 20 6

end NUMINAMATH_CALUDE_sarah_reads_40_wpm_l3448_344843


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3448_344860

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  ∃ (k : ℕ), k = 287 ∧ 
  (∀ (m : ℕ), 1729^m ∣ factorial 1729 → m ≤ k) ∧
  (1729^k ∣ factorial 1729) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l3448_344860


namespace NUMINAMATH_CALUDE_rotation_transformation_l3448_344826

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := ⟨(0, 0), (0, 10), (20, 0)⟩
def DEF : Triangle := ⟨(20, 10), (30, 10), (20, 2)⟩

def rotatePoint (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

def rotateTriangle (center : ℝ × ℝ) (angle : ℝ) (t : Triangle) : Triangle := sorry

theorem rotation_transformation (n x y : ℝ) 
  (h1 : 0 < n ∧ n < 180) 
  (h2 : rotateTriangle (x, y) n ABC = DEF) : 
  n + x + y = 92 := by sorry

end NUMINAMATH_CALUDE_rotation_transformation_l3448_344826


namespace NUMINAMATH_CALUDE_bugs_meet_on_bc_l3448_344853

/-- Triangle with side lengths -/
structure Triangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ

/-- Bug with starting position and speed -/
structure Bug where
  start : ℕ  -- 0 for A, 1 for B, 2 for C
  speed : ℝ
  clockwise : Bool

/-- The point where bugs meet -/
def MeetingPoint (t : Triangle) (bugA bugC : Bug) : ℝ := sorry

theorem bugs_meet_on_bc (t : Triangle) (bugA bugC : Bug) :
  t.ab = 5 ∧ t.bc = 6 ∧ t.ac = 7 ∧
  bugA.start = 0 ∧ bugA.speed = 1 ∧ bugA.clockwise = true ∧
  bugC.start = 2 ∧ bugC.speed = 2 ∧ bugC.clockwise = false →
  MeetingPoint t bugA bugC = 1 := by sorry

end NUMINAMATH_CALUDE_bugs_meet_on_bc_l3448_344853


namespace NUMINAMATH_CALUDE_cucumber_count_l3448_344870

theorem cucumber_count (total : ℕ) (ratio : ℕ) (h1 : total = 420) (h2 : ratio = 4) :
  ∃ (cucumbers : ℕ), cucumbers * (ratio + 1) = total ∧ cucumbers = 84 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_count_l3448_344870


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3448_344835

theorem arithmetic_calculation : 2 * (-5 + 3) + 2^3 / (-4) = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3448_344835


namespace NUMINAMATH_CALUDE_square_angles_equal_l3448_344810

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

end NUMINAMATH_CALUDE_square_angles_equal_l3448_344810


namespace NUMINAMATH_CALUDE_vector_problem_l3448_344861

/-- Given two vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, -2]

/-- Define vector c as a linear combination of a and b -/
def c : Fin 2 → ℝ := λ i ↦ 4 * a i + b i

/-- The dot product of two vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

theorem vector_problem :
  (dot_product b c • a = 0) ∧
  (dot_product a (a + (5/2 • b)) = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3448_344861


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3448_344874

/-- The number of ways to place 4 different balls into 4 numbered boxes with exactly one empty box -/
def ball_placement_count : ℕ := 144

/-- The number of different balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

theorem ball_placement_theorem :
  (num_balls = 4) →
  (num_boxes = 4) →
  (ball_placement_count = 144) :=
by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3448_344874


namespace NUMINAMATH_CALUDE_smallest_angle_sine_cosine_equality_l3448_344876

theorem smallest_angle_sine_cosine_equality : 
  ∃ x : ℝ, x > 0 ∧ x < (2 * Real.pi / 360) * 11 ∧
    Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) ∧
    ∀ y : ℝ, 0 < y ∧ y < x → 
      Real.sin (4 * y) * Real.sin (5 * y) ≠ Real.cos (4 * y) * Real.cos (5 * y) ∧
    x = (Real.pi / 18) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_sine_cosine_equality_l3448_344876


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l3448_344891

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, x < y → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l3448_344891


namespace NUMINAMATH_CALUDE_brians_age_in_eight_years_l3448_344834

/-- Given that Christian is twice as old as Brian and Christian will be 72 years old in eight years,
    prove that Brian will be 40 years old in eight years. -/
theorem brians_age_in_eight_years (christian_age : ℕ) (brian_age : ℕ) : 
  christian_age = 2 * brian_age →
  christian_age + 8 = 72 →
  brian_age + 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_brians_age_in_eight_years_l3448_344834


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3448_344838

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁ > x₂) ∧ 
  (|2 * x₁ - 3| = 14) ∧ 
  (|2 * x₂ - 3| = 14) ∧ 
  (x₁ - x₂ = 14) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3448_344838


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3448_344831

theorem geometric_progression_ratio (x y z w r : ℂ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
  x * (y - w) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (w - y) ≠ 0 ∧ w * (x - z) ≠ 0 ∧
  x * (y - w) ≠ y * (z - x) ∧ y * (z - x) ≠ z * (w - y) ∧ z * (w - y) ≠ w * (x - z) ∧
  ∃ (a : ℂ), a ≠ 0 ∧
    y * (z - x) = r * (x * (y - w)) ∧
    z * (w - y) = r * (y * (z - x)) ∧
    w * (x - z) = r * (z * (w - y)) →
  r^3 + r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3448_344831


namespace NUMINAMATH_CALUDE_jesses_room_difference_l3448_344856

theorem jesses_room_difference (width : ℝ) (length : ℝ) 
  (h1 : width = 19.7) (h2 : length = 20.25) : length - width = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_jesses_room_difference_l3448_344856


namespace NUMINAMATH_CALUDE_harry_owns_three_geckos_l3448_344833

/-- Represents the number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- Represents the number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- Represents the number of snakes Harry owns -/
def num_snakes : ℕ := 4

/-- Represents the monthly feeding cost per snake in dollars -/
def snake_cost : ℕ := 10

/-- Represents the monthly feeding cost per iguana in dollars -/
def iguana_cost : ℕ := 5

/-- Represents the monthly feeding cost per gecko in dollars -/
def gecko_cost : ℕ := 15

/-- Represents the total annual feeding cost for all pets in dollars -/
def total_annual_cost : ℕ := 1140

/-- Theorem stating that the number of geckos Harry owns is 3 -/
theorem harry_owns_three_geckos :
  num_geckos = 3 ∧
  num_geckos * gecko_cost * 12 + num_iguanas * iguana_cost * 12 + num_snakes * snake_cost * 12 = total_annual_cost :=
by sorry

end NUMINAMATH_CALUDE_harry_owns_three_geckos_l3448_344833


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3448_344892

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) * a m = a n * a (m + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  a 5 + a 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3448_344892


namespace NUMINAMATH_CALUDE_isosceles_equilateral_conditions_l3448_344884

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Represents the feet of perpendiculars from a point to the sides of a triangle -/
structure Perpendiculars where
  D : Point  -- foot on AB
  E : Point  -- foot on BC
  F : Point  -- foot on CA

/-- Calculates the feet of perpendiculars from a point to the sides of a triangle -/
def calculatePerpendiculars (p : Point) (t : Triangle) : Perpendiculars := sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Represents an Apollonius circle of a triangle -/
structure ApolloniusCircle where
  center : Point
  radius : ℝ

/-- Calculates the Apollonius circles of a triangle -/
def calculateApolloniusCircles (t : Triangle) : List ApolloniusCircle := sorry

/-- Checks if a point lies on an Apollonius circle -/
def liesOnApolloniusCircle (p : Point) (c : ApolloniusCircle) : Prop := sorry

/-- Calculates the Fermat point of a triangle -/
def calculateFermatPoint (t : Triangle) : Point := sorry

/-- The main theorem -/
theorem isosceles_equilateral_conditions 
  (t : Triangle) 
  (h1 : isAcuteAngled t) 
  (p : Point) 
  (h2 : isInside p t) 
  (perps : Perpendiculars) 
  (h3 : perps = calculatePerpendiculars p t) :
  (isIsosceles (Triangle.mk perps.D perps.E perps.F) ↔ 
    ∃ c ∈ calculateApolloniusCircles t, liesOnApolloniusCircle p c) ∧
  (isEquilateral (Triangle.mk perps.D perps.E perps.F) ↔ 
    p = calculateFermatPoint t) := by sorry

end NUMINAMATH_CALUDE_isosceles_equilateral_conditions_l3448_344884


namespace NUMINAMATH_CALUDE_rectangle_side_length_l3448_344800

theorem rectangle_side_length (area : ℚ) (side1 : ℚ) (side2 : ℚ) : 
  area = 1/8 → side1 = 1/2 → area = side1 * side2 → side2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l3448_344800


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3448_344894

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + a

-- Define the solution set condition
def is_solution_set (a m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ m < x ∧ x < 1

-- Theorem statement
theorem quadratic_inequality_solution (a m : ℝ) 
  (h : is_solution_set a m) : m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3448_344894


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3448_344801

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3448_344801


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l3448_344887

theorem easter_egg_hunt (total_eggs : ℕ) (club_house_eggs : ℕ) (town_hall_eggs : ℕ) 
  (h1 : total_eggs = 80)
  (h2 : club_house_eggs = 40)
  (h3 : town_hall_eggs = 15) :
  ∃ park_eggs : ℕ, 
    park_eggs = total_eggs - club_house_eggs - town_hall_eggs ∧ 
    park_eggs = 25 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l3448_344887


namespace NUMINAMATH_CALUDE_gcd_problem_l3448_344897

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) :
  Int.gcd (4 * b ^ 2 + 35 * b + 72) (3 * b + 8) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3448_344897


namespace NUMINAMATH_CALUDE_oil_division_l3448_344883

/-- Proves that given 12.4 liters of oil divided into two bottles, where the large bottle can hold 2.6 liters more than the small bottle, the large bottle will hold 7.5 liters. -/
theorem oil_division (total_oil : ℝ) (difference : ℝ) (large_bottle : ℝ) : 
  total_oil = 12.4 →
  difference = 2.6 →
  large_bottle = (total_oil + difference) / 2 →
  large_bottle = 7.5 :=
by
  sorry

end NUMINAMATH_CALUDE_oil_division_l3448_344883


namespace NUMINAMATH_CALUDE_problem_statement_l3448_344812

theorem problem_statement 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_a_order : a₁ < a₂ ∧ a₂ < a₃)
  (h_b_order : b₁ < b₂ ∧ b₂ < b₃)
  (h_sum : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
  (h_product_sum : a₁*a₂ + a₁*a₃ + a₂*a₃ = b₁*b₂ + b₁*b₃ + b₂*b₃)
  (h_a₁_b₁ : a₁ < b₁) : 
  (b₂ < a₂) ∧ 
  (a₃ < b₃) ∧ 
  (a₁*a₂*a₃ < b₁*b₂*b₃) ∧ 
  ((1-a₁)*(1-a₂)*(1-a₃) > (1-b₁)*(1-b₂)*(1-b₃)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3448_344812


namespace NUMINAMATH_CALUDE_problem_statement_l3448_344845

theorem problem_statement (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : Real.exp a + a = Real.log (b * Real.exp b) ∧ Real.log (b * Real.exp b) = 2) :
  (b * Real.exp b = Real.exp 2) ∧
  (a + b = 2) ∧
  (Real.exp a + Real.log b = 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3448_344845


namespace NUMINAMATH_CALUDE_candidate_fail_marks_l3448_344899

theorem candidate_fail_marks (max_marks : ℝ) (passing_percentage : ℝ) (candidate_score : ℝ) :
  max_marks = 153.84615384615384 →
  passing_percentage = 52 →
  candidate_score = 45 →
  ⌈passing_percentage / 100 * max_marks⌉ - candidate_score = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_fail_marks_l3448_344899


namespace NUMINAMATH_CALUDE_partnership_profit_l3448_344822

/-- A partnership business problem -/
theorem partnership_profit (investment_ratio : ℕ) (time_ratio : ℕ) (profit_B : ℕ) : 
  investment_ratio = 5 →
  time_ratio = 3 →
  profit_B = 4000 →
  investment_ratio * time_ratio * profit_B + profit_B = 64000 :=
by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l3448_344822


namespace NUMINAMATH_CALUDE_balloon_difference_l3448_344804

theorem balloon_difference (your_balloons friend_balloons : ℝ) 
  (h1 : your_balloons = -7)
  (h2 : friend_balloons = 4.5) :
  friend_balloons - your_balloons = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3448_344804


namespace NUMINAMATH_CALUDE_brother_age_proof_l3448_344888

def brother_age_in_5_years (nick_age : ℕ) : ℕ :=
  let sister_age := nick_age + 6
  let brother_age := (nick_age + sister_age) / 2
  brother_age + 5

theorem brother_age_proof (nick_age : ℕ) (h : nick_age = 13) :
  brother_age_in_5_years nick_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_brother_age_proof_l3448_344888


namespace NUMINAMATH_CALUDE_math_club_team_selection_l3448_344829

def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8
def boys_in_team : ℕ := 5
def girls_in_team : ℕ := 3

theorem math_club_team_selection :
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 55440 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l3448_344829


namespace NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l3448_344865

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- Theorem statement
theorem line_perp_plane_sufficient_condition 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_sufficient_condition_l3448_344865


namespace NUMINAMATH_CALUDE_luncheon_table_capacity_l3448_344855

theorem luncheon_table_capacity (invited : Nat) (no_shows : Nat) (tables : Nat) : Nat :=
  if invited = 18 ∧ no_shows = 12 ∧ tables = 2 then
    3
  else
    0

#check luncheon_table_capacity

end NUMINAMATH_CALUDE_luncheon_table_capacity_l3448_344855


namespace NUMINAMATH_CALUDE_inscribed_pentagon_external_angles_sum_inscribed_pentagon_external_angles_sum_is_720_l3448_344839

/-- Represents a pentagon inscribed in a circle -/
structure InscribedPentagon where
  -- We don't need to define the specific properties of the pentagon,
  -- as the problem doesn't require detailed information about its structure

/-- 
Theorem: For a pentagon inscribed in a circle, the sum of the angles
inscribed in the five segments outside the pentagon but inside the circle
is equal to 720°.
-/
theorem inscribed_pentagon_external_angles_sum
  (p : InscribedPentagon) : Real :=
  720

/-- 
Main theorem: The sum of the angles inscribed in the five segments
outside an inscribed pentagon but inside the circle is 720°.
-/
theorem inscribed_pentagon_external_angles_sum_is_720
  (p : InscribedPentagon) :
  inscribed_pentagon_external_angles_sum p = 720 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_pentagon_external_angles_sum_inscribed_pentagon_external_angles_sum_is_720_l3448_344839


namespace NUMINAMATH_CALUDE_digit_counting_theorem_l3448_344871

/-- The set of available digits -/
def availableDigits : Finset ℕ := {0, 1, 2, 3, 5, 9}

/-- Count of four-digit numbers -/
def countFourDigit : ℕ := 300

/-- Count of four-digit odd numbers -/
def countFourDigitOdd : ℕ := 192

/-- Count of four-digit even numbers -/
def countFourDigitEven : ℕ := 108

/-- Total count of natural numbers -/
def countNaturalNumbers : ℕ := 1631

/-- Main theorem stating the counting results -/
theorem digit_counting_theorem :
  (∀ d ∈ availableDigits, d < 10) ∧
  (countFourDigit = 300) ∧
  (countFourDigitOdd = 192) ∧
  (countFourDigitEven = 108) ∧
  (countNaturalNumbers = 1631) := by
  sorry

end NUMINAMATH_CALUDE_digit_counting_theorem_l3448_344871


namespace NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l3448_344816

/-- The decimal representation of a repeating decimal with a single digit. -/
def repeating_decimal (d : ℕ) : ℚ :=
  (d : ℚ) / 9

/-- Theorem stating that 0.666... (repeating) is equal to 2/3 -/
theorem repeating_six_equals_two_thirds : repeating_decimal 6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_six_equals_two_thirds_l3448_344816


namespace NUMINAMATH_CALUDE_solution_characterization_l3448_344869

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0),
   (Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
   (-Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
   (-Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2),
   (Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2)}

def satisfies_equations (p : ℝ × ℝ) : Prop :=
  let x := p.1
  let y := p.2
  x = 3 * x^2 * y - y^3 ∧ y = x^3 - 3 * x * y^2

theorem solution_characterization :
  ∀ p : ℝ × ℝ, satisfies_equations p ↔ p ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l3448_344869


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3448_344859

/-- A line with slope 1 passing through (0, a) is tangent to the circle x^2 + y^2 = 2 if and only if a = ±2 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∃ (x y : ℝ), y = x + a ∧ x^2 + y^2 = 2 ∧ 
  ∀ (x' y' : ℝ), y' = x' + a → x'^2 + y'^2 ≥ 2) ↔ 
  (a = 2 ∨ a = -2) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3448_344859


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l3448_344837

/-- Proposition p: A real number x satisfies the given inequalities -/
def p (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- Proposition q: A real number x satisfies the given inequality -/
def q (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

/-- Theorem stating that if p is a sufficient condition for q, then 7 ≤ a ≤ 8 -/
theorem sufficient_condition_implies_a_range (a : ℝ) : 
  (∀ x, p x → q x a) → 7 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l3448_344837


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l3448_344882

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ m > 0 ∧ n > 0 ∧ (2 : ℚ) / p = 1 / n + 1 / m ∧
  n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l3448_344882


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3448_344852

theorem least_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 6 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 5 = 4 ∧ m % 7 = 6 → n ≤ m :=
by
  use 69
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3448_344852


namespace NUMINAMATH_CALUDE_division_problem_l3448_344881

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℕ) : 
  dividend = 55053 → 
  quotient = 120 → 
  remainder = 333 → 
  dividend = divisor * quotient + remainder → 
  divisor = 456 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3448_344881
