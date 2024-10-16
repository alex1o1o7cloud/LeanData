import Mathlib

namespace NUMINAMATH_CALUDE_x_greater_than_sin_x_negation_of_implication_and_sufficient_not_necessary_for_or_negation_of_forall_x_minus_ln_x_positive_l1819_181971

-- Statement 1
theorem x_greater_than_sin_x (x : ℝ) (h : x > 0) : x > Real.sin x := by sorry

-- Statement 2
theorem negation_of_implication :
  (¬ (∀ x : ℝ, x - Real.sin x = 0 → x = 0)) ↔
  (∃ x : ℝ, x - Real.sin x ≠ 0 ∧ x ≠ 0) := by sorry

-- Statement 3
theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by sorry

-- Statement 4
theorem negation_of_forall_x_minus_ln_x_positive :
  (¬ (∀ x : ℝ, x - Real.log x > 0)) ↔
  (∃ x : ℝ, x - Real.log x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_x_greater_than_sin_x_negation_of_implication_and_sufficient_not_necessary_for_or_negation_of_forall_x_minus_ln_x_positive_l1819_181971


namespace NUMINAMATH_CALUDE_function_properties_l1819_181968

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem function_properties (a : ℝ) (h_min : ∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f a x ≤ f a y) 
  (h_min_value : ∃ (x : ℝ), x ∈ interval ∧ f a x = -37) :
  a = 3 ∧ ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), x ∈ interval → f a x ≤ m := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1819_181968


namespace NUMINAMATH_CALUDE_challenging_polynomial_theorem_l1819_181913

/-- Defines a quadratic polynomial q(x) = x^2 + bx + c -/
def q (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- Defines the composition q(q(x)) -/
def q_comp (b c : ℝ) (x : ℝ) : ℝ := q b c (q b c x)

/-- States that q(q(x)) = 1 has exactly four distinct real solutions -/
def has_four_solutions (b c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    q_comp b c x₁ = 1 ∧ q_comp b c x₂ = 1 ∧ q_comp b c x₃ = 1 ∧ q_comp b c x₄ = 1 ∧
    ∀ (y : ℝ), q_comp b c y = 1 → y = x₁ ∨ y = x₂ ∨ y = x₃ ∨ y = x₄

/-- The product of roots for a quadratic polynomial -/
def root_product (b c : ℝ) : ℝ := c

theorem challenging_polynomial_theorem :
  has_four_solutions (3/4) 1 ∧
  (∀ b c : ℝ, has_four_solutions b c → root_product b c ≤ root_product (3/4) 1) ∧
  q (3/4) 1 (-3) = 31/4 := by sorry

end NUMINAMATH_CALUDE_challenging_polynomial_theorem_l1819_181913


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l1819_181966

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 = 1 is a quadratic equation in one variable -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l1819_181966


namespace NUMINAMATH_CALUDE_initial_saline_concentration_l1819_181988

theorem initial_saline_concentration 
  (initial_weight : ℝ) 
  (water_added : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_weight = 100)
  (h2 : water_added = 200)
  (h3 : final_concentration = 10)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 30 ∧ 
    (initial_concentration / 100) * initial_weight = 
    (final_concentration / 100) * (initial_weight + water_added) :=
sorry

end NUMINAMATH_CALUDE_initial_saline_concentration_l1819_181988


namespace NUMINAMATH_CALUDE_line_problem_l1819_181920

theorem line_problem (front_position back_position total : ℕ) 
  (h1 : front_position = 8)
  (h2 : back_position = 6)
  (h3 : total = front_position + back_position - 1) :
  total = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l1819_181920


namespace NUMINAMATH_CALUDE_triangle_side_less_than_semiperimeter_l1819_181952

theorem triangle_side_less_than_semiperimeter (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_less_than_semiperimeter_l1819_181952


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1819_181902

/-- A curve represented by the equation mx^2 + ny^2 = 1 -/
structure Curve (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate to check if a curve is an ellipse -/
def IsEllipse (c : Curve m n) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is a necessary but not sufficient condition for the curve to be an ellipse -/
theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (∀ (c : Curve m n), IsEllipse c → m * n > 0) ∧
  ¬(∀ (c : Curve m n), m * n > 0 → IsEllipse c) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1819_181902


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l1819_181925

/-- The equation represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0) ∧
    (∀ x y : ℝ, x^2 - 36*y^2 - 12*x + y + 64 = 0 ↔ 
      a*(x - c)^2 + b*(y - d)^2 + e*(x - c) + f*(y - d) = 1) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l1819_181925


namespace NUMINAMATH_CALUDE_douyin_sales_and_profit_l1819_181903

/-- Represents an e-commerce platform selling a small commodity. -/
structure ECommercePlatform where
  cost_price : ℕ
  initial_price : ℕ
  initial_volume : ℕ
  price_decrease : ℕ
  volume_increase : ℕ

/-- Calculates the daily sales volume for a given selling price. -/
def daily_sales_volume (platform : ECommercePlatform) (selling_price : ℕ) : ℕ :=
  platform.initial_volume + 
    (platform.initial_price - selling_price) / platform.price_decrease * platform.volume_increase

/-- Calculates the daily profit for a given selling price. -/
def daily_profit (platform : ECommercePlatform) (selling_price : ℕ) : ℕ :=
  (selling_price - platform.cost_price) * daily_sales_volume platform selling_price

/-- The e-commerce platform with given conditions. -/
def douyin_platform : ECommercePlatform := {
  cost_price := 40
  initial_price := 60
  initial_volume := 20
  price_decrease := 5
  volume_increase := 10
}

theorem douyin_sales_and_profit :
  (daily_sales_volume douyin_platform 50 = 40) ∧
  (∃ (price : ℕ), daily_profit douyin_platform price = 448 ∧
    ∀ (p : ℕ), daily_profit douyin_platform p = 448 → p ≥ price) :=
by sorry

end NUMINAMATH_CALUDE_douyin_sales_and_profit_l1819_181903


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1819_181991

theorem largest_n_for_factorization : 
  let P (n : ℤ) := ∃ (A B : ℤ), 5 * X^2 + n * X + 120 = (5 * X + A) * (X + B)
  ∀ (m : ℤ), P m → m ≤ 601 ∧ P 601 := by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1819_181991


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l1819_181985

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l1819_181985


namespace NUMINAMATH_CALUDE_initial_cost_of_article_l1819_181956

/-- 
Proves that the initial cost of an article is 3000, given the conditions of two successive discounts.
-/
theorem initial_cost_of_article (price_after_first_discount : ℕ) 
  (final_price : ℕ) (h1 : price_after_first_discount = 2100) 
  (h2 : final_price = 1050) : ℕ :=
  by
    sorry

#check initial_cost_of_article

end NUMINAMATH_CALUDE_initial_cost_of_article_l1819_181956


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1819_181980

theorem complex_magnitude_product : Complex.abs ((7 - 4*I) * (3 + 11*I)) = Real.sqrt 8450 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1819_181980


namespace NUMINAMATH_CALUDE_inequality_solution_part1_inequality_solution_part2_l1819_181995

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the inequality function
def inequality (x a : ℝ) : Prop := lg (|x + 3| + |x - 7|) > a

theorem inequality_solution_part1 :
  ∀ x : ℝ, inequality x 1 ↔ (x < -3 ∨ x > 7) := by sorry

theorem inequality_solution_part2 :
  ∀ a : ℝ, (∀ x : ℝ, inequality x a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_part1_inequality_solution_part2_l1819_181995


namespace NUMINAMATH_CALUDE_point_coordinates_l1819_181909

/-- If point A(a, a-2) lies on the x-axis, then the coordinates of point B(a+2, a-1) are (4, 1) -/
theorem point_coordinates (a : ℝ) :
  (a = 2) → (a + 2 = 4 ∧ a - 1 = 1) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_l1819_181909


namespace NUMINAMATH_CALUDE_acidic_mixture_concentration_l1819_181948

/-- Proves that mixing liquids from two containers with given concentrations
    results in a mixture with the desired concentration. -/
theorem acidic_mixture_concentration
  (volume1 : ℝ) (volume2 : ℝ) (conc1 : ℝ) (conc2 : ℝ) (target_conc : ℝ)
  (x : ℝ) (y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0)
  (h_vol1 : volume1 = 54) (h_vol2 : volume2 = 48)
  (h_conc1 : conc1 = 0.35) (h_conc2 : conc2 = 0.25)
  (h_target : target_conc = 0.75)
  (h_mixture : conc1 * x + conc2 * y = target_conc * (x + y)) :
  (conc1 * x + conc2 * y) / (x + y) = target_conc :=
sorry

end NUMINAMATH_CALUDE_acidic_mixture_concentration_l1819_181948


namespace NUMINAMATH_CALUDE_cans_given_away_equals_2500_l1819_181947

/-- Represents the food bank's inventory and distribution --/
structure FoodBank where
  initialStock : Nat
  day1People : Nat
  day1CansPerPerson : Nat
  day1Restock : Nat
  day2People : Nat
  day2CansPerPerson : Nat
  day2Restock : Nat

/-- Calculates the total number of cans given away --/
def totalCansGivenAway (fb : FoodBank) : Nat :=
  fb.day1People * fb.day1CansPerPerson + fb.day2People * fb.day2CansPerPerson

/-- Theorem stating that given the specific conditions, 2500 cans were given away --/
theorem cans_given_away_equals_2500 (fb : FoodBank) 
  (h1 : fb.initialStock = 2000)
  (h2 : fb.day1People = 500)
  (h3 : fb.day1CansPerPerson = 1)
  (h4 : fb.day1Restock = 1500)
  (h5 : fb.day2People = 1000)
  (h6 : fb.day2CansPerPerson = 2)
  (h7 : fb.day2Restock = 3000) :
  totalCansGivenAway fb = 2500 := by
  sorry

end NUMINAMATH_CALUDE_cans_given_away_equals_2500_l1819_181947


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1819_181983

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a-2, 2*a-2}
  A ⊆ B → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1819_181983


namespace NUMINAMATH_CALUDE_function_analysis_l1819_181926

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x + a / x^2

theorem function_analysis (a : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a) ((2 / x) - (2 * a / x^3)) x) →
  (HasDerivAt (f a) 0 1 → a = 1) ∧
  (a > 0 → IsLocalMin (f a) (Real.sqrt a)) ∧
  ((∃ x y, 1 ≤ x ∧ x < y ∧ f a x = 2 ∧ f a y = 2) → 2 ≤ a ∧ a < Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_function_analysis_l1819_181926


namespace NUMINAMATH_CALUDE_f_monotone_and_inequality_l1819_181986

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - Real.log (x + 1) + Real.log (x - 1)

theorem f_monotone_and_inequality (k : ℝ) (h₁ : -1 ≤ k) (h₂ : k ≤ 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧
  (∀ x > 1, ∀ y > 1, x < y → f x < f y) ∧
  (∀ x > 1, x * (f x₁ + f x₂) ≥ (x + 1) * (f x + 2 - 2*x)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_inequality_l1819_181986


namespace NUMINAMATH_CALUDE_isosceles_triangle_a_values_l1819_181936

/-- An isosceles triangle with sides 10-a, 7, and 6 -/
structure IsoscelesTriangle (a : ℝ) :=
  (side1 : ℝ := 10 - a)
  (side2 : ℝ := 7)
  (side3 : ℝ := 6)
  (isIsosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ 
                 (side1 = side3 ∧ side2 ≠ side1) ∨ 
                 (side2 = side3 ∧ side1 ≠ side2))

/-- The theorem stating that a is either 3 or 4 for an isosceles triangle with sides 10-a, 7, and 6 -/
theorem isosceles_triangle_a_values :
  ∀ a : ℝ, IsoscelesTriangle a → a = 3 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_a_values_l1819_181936


namespace NUMINAMATH_CALUDE_max_inverse_sum_14th_power_l1819_181921

/-- A quadratic polynomial x^2 - tx + q with roots r1 and r2 -/
structure QuadraticPolynomial where
  t : ℝ
  q : ℝ
  r1 : ℝ
  r2 : ℝ
  is_root : r1^2 - t*r1 + q = 0 ∧ r2^2 - t*r2 + q = 0

/-- The condition that the sum of powers of roots are equal up to 13th power -/
def equal_sum_powers (p : QuadraticPolynomial) : Prop :=
  ∀ n : ℕ, n ≤ 13 → p.r1^n + p.r2^n = p.r1 + p.r2

/-- The theorem statement -/
theorem max_inverse_sum_14th_power (p : QuadraticPolynomial) 
  (h : equal_sum_powers p) : 
  (∀ p' : QuadraticPolynomial, equal_sum_powers p' → 
    1 / p'.r1^14 + 1 / p'.r2^14 ≤ 1 / p.r1^14 + 1 / p.r2^14) →
  1 / p.r1^14 + 1 / p.r2^14 = 2 :=
sorry

end NUMINAMATH_CALUDE_max_inverse_sum_14th_power_l1819_181921


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l1819_181908

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  let c := Real.sqrt (x^2 + y^2)
  let c' := Real.sqrt ((x+1)^2 + (y+1)^2)
  c' - c ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l1819_181908


namespace NUMINAMATH_CALUDE_point_on_line_l1819_181907

/-- Given two points (m, n) and (m + p, n + 18) on the line x = (y / 6) - (2 / 5), prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 6 - 2 / 5) → 
  (m + p = (n + 18) / 6 - 2 / 5) → 
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1819_181907


namespace NUMINAMATH_CALUDE_three_common_tangents_l1819_181963

/-- Represents a circle in the 2D plane --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Calculates the number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem: The number of common tangents between the given circles is 3 --/
theorem three_common_tangents :
  let c1 : Circle := { a := 1, b := 1, c := 2, d := 2, e := -2 }
  let c2 : Circle := { a := 1, b := 1, c := -6, d := 2, e := 6 }
  commonTangents c1 c2 = 3 := by sorry

end NUMINAMATH_CALUDE_three_common_tangents_l1819_181963


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_17_5_minus_13_5_l1819_181915

/-- The highest power of 2 that divides 17^5 - 13^5 is 4 -/
theorem highest_power_of_two_dividing_17_5_minus_13_5 :
  (∃ k : ℕ, 17^5 - 13^5 = 2^2 * (2*k + 1)) ∧
  ¬(∃ k : ℕ, 17^5 - 13^5 = 2^3 * (2*k + 1)) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_17_5_minus_13_5_l1819_181915


namespace NUMINAMATH_CALUDE_correct_calculation_l1819_181979

theorem correct_calculation (a : ℝ) : -3*a - 2*a = -5*a := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1819_181979


namespace NUMINAMATH_CALUDE_quadratic_equation_property_l1819_181943

theorem quadratic_equation_property (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   3 * x^2 + 5 * x + k = 0 ∧ 
   3 * y^2 + 5 * y + k = 0 ∧
   |x - y| = x^2 + y^2) ↔ 
  (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_property_l1819_181943


namespace NUMINAMATH_CALUDE_work_day_ends_at_430pm_l1819_181962

-- Define the structure for time
structure Time where
  hours : Nat
  minutes : Nat

-- Define the work schedule
def workStartTime : Time := { hours := 8, minutes := 0 }
def lunchStartTime : Time := { hours := 13, minutes := 0 }
def lunchDuration : Nat := 30
def totalWorkHours : Nat := 8

-- Function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Function to calculate time difference in hours
def timeDifferenceInHours (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes - (t1.hours * 60 + t1.minutes)) / 60

-- Theorem stating that Maria's work day ends at 4:30 P.M.
theorem work_day_ends_at_430pm :
  let lunchEndTime := addMinutes lunchStartTime lunchDuration
  let workBeforeLunch := timeDifferenceInHours workStartTime lunchStartTime
  let remainingWorkHours := totalWorkHours - workBeforeLunch
  let endTime := addMinutes lunchEndTime (remainingWorkHours * 60)
  endTime = { hours := 16, minutes := 30 } :=
by sorry

end NUMINAMATH_CALUDE_work_day_ends_at_430pm_l1819_181962


namespace NUMINAMATH_CALUDE_completing_square_equiv_l1819_181969

/-- Proves that y = -x^2 + 2x + 3 can be rewritten as y = -(x-1)^2 + 4 -/
theorem completing_square_equiv :
  ∀ x y : ℝ, y = -x^2 + 2*x + 3 ↔ y = -(x-1)^2 + 4 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equiv_l1819_181969


namespace NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l1819_181950

/-- Yan's position between home and stadium -/
structure Position where
  distanceToHome : ℝ
  distanceToStadium : ℝ
  isBetween : 0 < distanceToHome ∧ 0 < distanceToStadium

/-- Yan's travel speeds -/
structure Speeds where
  walkingSpeed : ℝ
  bicycleSpeed : ℝ
  bicycleFaster : bicycleSpeed = 5 * walkingSpeed

/-- The theorem stating the ratio of distances -/
theorem distance_ratio_is_two_thirds 
  (pos : Position) (speeds : Speeds) 
  (equalTime : pos.distanceToStadium / speeds.walkingSpeed = 
               pos.distanceToHome / speeds.walkingSpeed + 
               (pos.distanceToHome + pos.distanceToStadium) / speeds.bicycleSpeed) :
  pos.distanceToHome / pos.distanceToStadium = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_ratio_is_two_thirds_l1819_181950


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1819_181953

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 3 ∧ lg (x - 3) + lg x = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1819_181953


namespace NUMINAMATH_CALUDE_prob_4_largest_l1819_181975

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def draw_size : ℕ := 3

def prob_not_select_5 : ℚ := 2 / 5

def prob_not_select_4_and_5 : ℚ := 1 / 10

theorem prob_4_largest (s : Finset ℕ) (n : ℕ) 
  (h1 : s = card_set) 
  (h2 : n = draw_size) 
  (h3 : prob_not_select_5 = 2 / 5) 
  (h4 : prob_not_select_4_and_5 = 1 / 10) : 
  (prob_not_select_5 - prob_not_select_4_and_5 : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_4_largest_l1819_181975


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l1819_181964

def initial_stock : ℝ := 800
def type_a_percent : ℝ := 0.40
def type_b_percent : ℝ := 0.35
def type_c_percent : ℝ := 0.25
def type_a_decaf : ℝ := 0.20
def type_b_decaf : ℝ := 0.50
def type_c_decaf : ℝ := 0

def additional_purchase : ℝ := 300
def additional_type_a_percent : ℝ := 0.50
def additional_type_b_percent : ℝ := 0.30
def additional_type_c_percent : ℝ := 0.20

theorem coffee_decaf_percentage :
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := 
    initial_stock * (type_a_percent * type_a_decaf + 
                     type_b_percent * type_b_decaf + 
                     type_c_percent * type_c_decaf)
  let additional_decaf := 
    additional_purchase * (additional_type_a_percent * type_a_decaf + 
                           additional_type_b_percent * type_b_decaf + 
                           additional_type_c_percent * type_c_decaf)
  let total_decaf := initial_decaf + additional_decaf
  (total_decaf / total_stock) * 100 = (279 / 1100) * 100 := by
sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l1819_181964


namespace NUMINAMATH_CALUDE_cd_cost_l1819_181958

theorem cd_cost (two_cd_cost : ℝ) (h : two_cd_cost = 36) :
  8 * (two_cd_cost / 2) = 144 := by
sorry

end NUMINAMATH_CALUDE_cd_cost_l1819_181958


namespace NUMINAMATH_CALUDE_team_selection_combinations_l1819_181924

theorem team_selection_combinations (n m k : ℕ) (hn : n = 5) (hm : m = 5) (hk : k = 3) :
  (Nat.choose (n + m) k) - (Nat.choose n k) = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_combinations_l1819_181924


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1819_181990

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 27 * Real.sqrt 8 = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1819_181990


namespace NUMINAMATH_CALUDE_only_square_relationship_functional_l1819_181974

/-- Represents a relationship between two variables -/
structure Relationship where
  is_functional : Bool

/-- The relationship between the side length and the area of a square -/
def square_relationship : Relationship := sorry

/-- The relationship between rice yield and the amount of fertilizer applied -/
def rice_fertilizer_relationship : Relationship := sorry

/-- The relationship between snowfall and the rate of traffic accidents -/
def snowfall_accidents_relationship : Relationship := sorry

/-- The relationship between a person's height and weight -/
def height_weight_relationship : Relationship := sorry

/-- Theorem stating that only the square relationship is functional -/
theorem only_square_relationship_functional :
  square_relationship.is_functional ∧
  ¬rice_fertilizer_relationship.is_functional ∧
  ¬snowfall_accidents_relationship.is_functional ∧
  ¬height_weight_relationship.is_functional :=
by sorry

end NUMINAMATH_CALUDE_only_square_relationship_functional_l1819_181974


namespace NUMINAMATH_CALUDE_circle_slope_bounds_l1819_181987

theorem circle_slope_bounds (x y : ℝ) (h : x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∃ (k : ℝ), y = k*(x-4) ∧ -20/21 ≤ k ∧ k ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_slope_bounds_l1819_181987


namespace NUMINAMATH_CALUDE_special_hexagon_area_l1819_181905

/-- A hexagon with specific properties -/
structure SpecialHexagon where
  -- Each angle measures 120°
  angle_measure : ℝ
  angle_measure_eq : angle_measure = 120
  -- Sides alternately measure 1 cm and √3 cm
  side_length1 : ℝ
  side_length2 : ℝ
  side_length1_eq : side_length1 = 1
  side_length2_eq : side_length2 = Real.sqrt 3

/-- The area of the special hexagon -/
noncomputable def area (h : SpecialHexagon) : ℝ := 3 + Real.sqrt 3

/-- Theorem stating that the area of the special hexagon is 3 + √3 cm² -/
theorem special_hexagon_area (h : SpecialHexagon) : area h = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_area_l1819_181905


namespace NUMINAMATH_CALUDE_same_color_probability_l1819_181922

def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def drawn_marbles : ℕ := 4

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

theorem same_color_probability : 
  (Nat.choose red_marbles drawn_marbles + 
   Nat.choose white_marbles drawn_marbles + 
   Nat.choose blue_marbles drawn_marbles : ℚ) / 
  (Nat.choose total_marbles drawn_marbles : ℚ) = 11 / 612 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1819_181922


namespace NUMINAMATH_CALUDE_congruence_implies_prime_and_n_equals_m_minus_one_l1819_181934

theorem congruence_implies_prime_and_n_equals_m_minus_one 
  (n m : ℕ) 
  (h_n : n ≥ 2) 
  (h_m : m ≥ 2) 
  (h_cong : ∀ k : ℕ, 1 ≤ k → k ≤ n → k^n % m = 1) : 
  Nat.Prime m ∧ n = m - 1 := by
sorry

end NUMINAMATH_CALUDE_congruence_implies_prime_and_n_equals_m_minus_one_l1819_181934


namespace NUMINAMATH_CALUDE_school_dinner_theatre_attendance_l1819_181912

theorem school_dinner_theatre_attendance
  (child_ticket_price : ℕ)
  (adult_ticket_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (h1 : child_ticket_price = 6)
  (h2 : adult_ticket_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_revenue = 1875) :
  ∃ (child_tickets adult_tickets : ℕ),
    child_tickets + adult_tickets = total_tickets ∧
    child_tickets * child_ticket_price + adult_tickets * adult_ticket_price = total_revenue ∧
    adult_tickets = 175 :=
sorry

end NUMINAMATH_CALUDE_school_dinner_theatre_attendance_l1819_181912


namespace NUMINAMATH_CALUDE_problem_statement_l1819_181955

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 6)
  (h2 : x * y = 128)
  (h3 : x = 2 * y^2) :
  (x + y) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1819_181955


namespace NUMINAMATH_CALUDE_expression_eval_zero_l1819_181931

theorem expression_eval_zero (a : ℚ) (h : a = 3/2) : 
  (5 * a^2 - 13 * a + 4) * (2 * a - 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_eval_zero_l1819_181931


namespace NUMINAMATH_CALUDE_james_remaining_milk_l1819_181919

/-- Calculates the remaining amount of milk in ounces after drinking some -/
def remaining_milk (initial_gallons : ℕ) (ounces_per_gallon : ℕ) (ounces_drunk : ℕ) : ℕ :=
  initial_gallons * ounces_per_gallon - ounces_drunk

/-- Proves that given the initial conditions, James has 371 ounces of milk left -/
theorem james_remaining_milk :
  remaining_milk 3 128 13 = 371 := by
  sorry

end NUMINAMATH_CALUDE_james_remaining_milk_l1819_181919


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1819_181945

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Generates the seat numbers in a systematic sample -/
def generate_sample (s : SystematicSample) : List ℕ :=
  List.range s.sample_size |>.map (λ i => s.first_element + i * s.interval)

theorem systematic_sample_theorem (sample : SystematicSample)
  (h1 : sample.population_size = 48)
  (h2 : sample.sample_size = 4)
  (h3 : sample.interval = sample.population_size / sample.sample_size)
  (h4 : sample.first_element = 6)
  (h5 : 30 ∈ generate_sample sample)
  (h6 : 42 ∈ generate_sample sample) :
  18 ∈ generate_sample sample :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1819_181945


namespace NUMINAMATH_CALUDE_flour_scoops_to_remove_l1819_181940

-- Define the constants
def total_flour : ℚ := 8
def needed_flour : ℚ := 6
def scoop_size : ℚ := 1/4

-- Theorem statement
theorem flour_scoops_to_remove : 
  (total_flour - needed_flour) / scoop_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_flour_scoops_to_remove_l1819_181940


namespace NUMINAMATH_CALUDE_final_fish_count_l1819_181965

/- Define the initial number of fish -/
variable (F : ℚ)

/- Define the number of fish after each day's operations -/
def fish_count (day : ℕ) : ℚ :=
  match day with
  | 0 => F
  | 1 => 2 * F
  | 2 => 4 * F * (2/3)
  | 3 => 8 * F * (2/3)
  | 4 => 16 * F * (2/3) * (3/4)
  | 5 => 32 * F * (2/3) * (3/4)
  | 6 => 64 * F * (2/3) * (3/4)
  | _ => 128 * F * (2/3) * (3/4) + 15

/- Theorem stating that the final count is 207 if and only if F = 6 -/
theorem final_fish_count (F : ℚ) : fish_count F 7 = 207 ↔ F = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_fish_count_l1819_181965


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1819_181916

theorem polynomial_simplification (x : ℝ) : 
  5 - 5*x - 10*x^2 + 10 + 15*x - 20*x^2 - 10 + 20*x + 30*x^2 = 5 + 30*x := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1819_181916


namespace NUMINAMATH_CALUDE_move_point_on_number_line_l1819_181917

theorem move_point_on_number_line (start : ℤ) (movement : ℤ) (result : ℤ) :
  start = -2 →
  movement = 3 →
  result = start + movement →
  result = 1 := by
  sorry

end NUMINAMATH_CALUDE_move_point_on_number_line_l1819_181917


namespace NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_neg_two_l1819_181998

/-- A polynomial in x and y with a parameter m -/
def polynomial (x y m : ℝ) : ℝ := 8 * x^2 + (m + 1) * x * y - 5 * y + x * y - 8

theorem no_xy_term_implies_m_eq_neg_two (m : ℝ) :
  (∀ x y : ℝ, polynomial x y m = 8 * x^2 - 5 * y - 8) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_neg_two_l1819_181998


namespace NUMINAMATH_CALUDE_arrangements_not_adjacent_l1819_181989

theorem arrangements_not_adjacent (n : ℕ) (h : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_not_adjacent_l1819_181989


namespace NUMINAMATH_CALUDE_fraction_to_sofia_is_one_twelfth_l1819_181970

/-- Represents the initial egg distribution and sharing problem --/
structure EggDistribution where
  mia_eggs : ℕ
  sofia_eggs : ℕ
  pablo_eggs : ℕ
  lucas_eggs : ℕ
  (sofia_eggs_def : sofia_eggs = 3 * mia_eggs)
  (pablo_eggs_def : pablo_eggs = 4 * sofia_eggs)
  (lucas_eggs_def : lucas_eggs = 0)

/-- Calculates the fraction of Pablo's eggs given to Sofia --/
def fraction_to_sofia (d : EggDistribution) : ℚ :=
  let total_eggs := d.mia_eggs + d.sofia_eggs + d.pablo_eggs + d.lucas_eggs
  let equal_share := total_eggs / 4
  let sofia_needs := equal_share - d.sofia_eggs
  sofia_needs / d.pablo_eggs

/-- Theorem stating that the fraction of Pablo's eggs given to Sofia is 1/12 --/
theorem fraction_to_sofia_is_one_twelfth (d : EggDistribution) :
  fraction_to_sofia d = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_sofia_is_one_twelfth_l1819_181970


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1819_181959

theorem expression_equals_zero : 
  |Real.sqrt 3 - 1| + (Real.pi - 3)^0 - Real.tan (Real.pi / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1819_181959


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1819_181994

/-- Represents the outcome of selecting an item -/
inductive ItemSelection
  | Qualified
  | Defective

/-- Represents a batch of products -/
structure Batch where
  qualified : ℕ
  defective : ℕ
  qualified_exceeds_two : qualified > 2
  defective_exceeds_two : defective > 2

/-- Represents the selection of two items from a batch -/
def TwoItemSelection := Prod ItemSelection ItemSelection

/-- Event: At least one defective item -/
def AtLeastOneDefective (selection : TwoItemSelection) : Prop :=
  selection.1 = ItemSelection.Defective ∨ selection.2 = ItemSelection.Defective

/-- Event: All qualified items -/
def AllQualified (selection : TwoItemSelection) : Prop :=
  selection.1 = ItemSelection.Qualified ∧ selection.2 = ItemSelection.Qualified

/-- Theorem: AtLeastOneDefective and AllQualified are mutually exclusive -/
theorem mutually_exclusive_events (batch : Batch) (selection : TwoItemSelection) :
  ¬(AtLeastOneDefective selection ∧ AllQualified selection) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1819_181994


namespace NUMINAMATH_CALUDE_triangle_area_circumradius_l1819_181984

theorem triangle_area_circumradius (a b c R : ℝ) (α β γ : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a / Real.sin α = b / Real.sin β →
  b / Real.sin β = c / Real.sin γ →
  c / Real.sin γ = 2 * R →
  S = 1/2 * a * b * Real.sin γ →
  S = a * b * c / (4 * R) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_circumradius_l1819_181984


namespace NUMINAMATH_CALUDE_fourteen_sided_figure_area_l1819_181973

/-- A fourteen-sided figure on a 1 cm × 1 cm graph paper -/
structure FourteenSidedFigure where
  /-- The number of full unit squares in the figure -/
  full_squares : ℕ
  /-- The number of small right-angled triangles in the figure -/
  small_triangles : ℕ

/-- Calculate the area of a fourteen-sided figure -/
def calculate_area (figure : FourteenSidedFigure) : ℝ :=
  figure.full_squares + (figure.small_triangles * 0.5)

theorem fourteen_sided_figure_area :
  ∀ (figure : FourteenSidedFigure),
    figure.full_squares = 10 →
    figure.small_triangles = 8 →
    calculate_area figure = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_sided_figure_area_l1819_181973


namespace NUMINAMATH_CALUDE_speeding_fine_calculation_l1819_181938

/-- The speeding fine calculation --/
theorem speeding_fine_calculation 
  (base_fine : ℝ)
  (speed_limit : ℝ)
  (actual_speed : ℝ)
  (court_costs : ℝ)
  (lawyer_hourly_rate : ℝ)
  (lawyer_hours : ℝ)
  (total_owed : ℝ)
  (h1 : base_fine = 50)
  (h2 : speed_limit = 30)
  (h3 : actual_speed = 75)
  (h4 : court_costs = 300)
  (h5 : lawyer_hourly_rate = 80)
  (h6 : lawyer_hours = 3)
  (h7 : total_owed = 820) :
  ∃ (fine_increase_per_mph : ℝ), 
    fine_increase_per_mph = 2.56 ∧
    total_owed = base_fine + court_costs + (lawyer_hourly_rate * lawyer_hours) + 
      (2 * fine_increase_per_mph * (actual_speed - speed_limit)) :=
by sorry

end NUMINAMATH_CALUDE_speeding_fine_calculation_l1819_181938


namespace NUMINAMATH_CALUDE_short_walk_probability_l1819_181939

/-- The number of gates in the airport --/
def numGates : ℕ := 20

/-- The distance between adjacent gates in feet --/
def gateDistance : ℕ := 50

/-- The maximum distance Dave can walk in feet --/
def maxWalkDistance : ℕ := 200

/-- The probability of selecting two different gates that are at most maxWalkDistance apart --/
def probabilityShortWalk : ℚ := 67 / 190

theorem short_walk_probability :
  (numGates : ℚ) * (numGates - 1) * probabilityShortWalk =
    (2 * 4) +  -- Gates at extreme ends
    (6 * 5) +  -- Gates 2 to 4 and 17 to 19
    (12 * 8)   -- Gates 5 to 16
  ∧ maxWalkDistance / gateDistance = 4 := by sorry

end NUMINAMATH_CALUDE_short_walk_probability_l1819_181939


namespace NUMINAMATH_CALUDE_carriage_problem_representation_l1819_181997

/-- Represents the problem of people sharing carriages -/
structure CarriageProblem where
  x : ℕ  -- Total number of people
  y : ℕ  -- Total number of carriages

/-- The conditions of the carriage problem are satisfied -/
def satisfies_conditions (p : CarriageProblem) : Prop :=
  (p.x / 3 = p.y + 2) ∧ ((p.x - 9) / 2 = p.y)

/-- The system of equations correctly represents the carriage problem -/
theorem carriage_problem_representation (p : CarriageProblem) :
  satisfies_conditions p ↔ 
    (p.x / 3 = p.y + 2) ∧ ((p.x - 9) / 2 = p.y) :=
by sorry


end NUMINAMATH_CALUDE_carriage_problem_representation_l1819_181997


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1819_181906

/-- For a quadratic equation x^2 + mx + k = 0 with real coefficients,
    this function determines if it has two distinct real roots. -/
def has_two_distinct_real_roots (m k : ℝ) : Prop :=
  k > 0 ∧ (m < -2 * Real.sqrt k ∨ m > 2 * Real.sqrt k)

/-- Theorem stating the conditions for a quadratic equation to have two distinct real roots. -/
theorem quadratic_two_distinct_roots (m k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + k = 0 ∧ y^2 + m*y + k = 0) ↔ has_two_distinct_real_roots m k :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1819_181906


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l1819_181978

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- The bridge length is 195 meters given specific train parameters -/
theorem bridge_length_specific : bridge_length 180 45 30 = 195 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_specific_l1819_181978


namespace NUMINAMATH_CALUDE_second_class_average_l1819_181914

theorem second_class_average (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (combined_avg : ℝ) :
  n1 = 25 →
  n2 = 30 →
  avg1 = 40 →
  combined_avg = 50.90909090909091 →
  (n1 * avg1 + n2 * ((n1 + n2) * combined_avg - n1 * avg1) / n2) / (n1 + n2) = combined_avg →
  ((n1 + n2) * combined_avg - n1 * avg1) / n2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_class_average_l1819_181914


namespace NUMINAMATH_CALUDE_price_restoration_l1819_181929

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (restored_price : ℝ) : 
  reduced_price = original_price * (1 - 0.2) →
  restored_price = reduced_price * (1 + 0.25) →
  restored_price = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_l1819_181929


namespace NUMINAMATH_CALUDE_acid_percentage_proof_l1819_181941

/-- Given a solution with 1.4 litres of pure acid in 4 litres total volume,
    prove that the percentage of pure acid is 35%. -/
theorem acid_percentage_proof : 
  let pure_acid_volume : ℝ := 1.4
  let total_solution_volume : ℝ := 4
  let percentage_pure_acid : ℝ := (pure_acid_volume / total_solution_volume) * 100
  percentage_pure_acid = 35 := by
  sorry

end NUMINAMATH_CALUDE_acid_percentage_proof_l1819_181941


namespace NUMINAMATH_CALUDE_daves_initial_files_l1819_181942

theorem daves_initial_files (initial_apps : ℕ) (final_apps : ℕ) (final_files : ℕ) (deleted_files : ℕ) : 
  initial_apps = 17 → 
  final_apps = 3 → 
  final_files = 7 → 
  deleted_files = 14 → 
  ∃ initial_files : ℕ, initial_files = 21 ∧ initial_files = final_files + deleted_files :=
by sorry

end NUMINAMATH_CALUDE_daves_initial_files_l1819_181942


namespace NUMINAMATH_CALUDE_permutation_absolute_difference_equality_l1819_181961

theorem permutation_absolute_difference_equality :
  ∀ (a : Fin 2011 → Fin 2011), Function.Bijective a →
  ∃ j k : Fin 2011, j < k ∧ |a j - j| = |a k - k| :=
by
  sorry

end NUMINAMATH_CALUDE_permutation_absolute_difference_equality_l1819_181961


namespace NUMINAMATH_CALUDE_equation_one_solutions_l1819_181901

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l1819_181901


namespace NUMINAMATH_CALUDE_walts_investment_rate_l1819_181949

/-- Proves that given the conditions of Walt's investment, the unknown interest rate is 8% -/
theorem walts_investment_rate : ∀ (total_money : ℝ) (known_rate : ℝ) (unknown_amount : ℝ) (total_interest : ℝ),
  total_money = 9000 →
  known_rate = 0.09 →
  unknown_amount = 4000 →
  total_interest = 770 →
  ∃ (unknown_rate : ℝ),
    unknown_rate * unknown_amount + known_rate * (total_money - unknown_amount) = total_interest ∧
    unknown_rate = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_walts_investment_rate_l1819_181949


namespace NUMINAMATH_CALUDE_correct_num_license_plates_l1819_181923

/-- The number of distinct license plates with 5 digits and two consecutive letters -/
def num_license_plates : ℕ :=
  let num_digits : ℕ := 10  -- 0 to 9
  let num_uppercase : ℕ := 26
  let num_lowercase : ℕ := 26
  let num_digit_positions : ℕ := 5
  let num_letter_pair_positions : ℕ := 6  -- The letter pair can start in any of the first 6 positions
  let num_letter_pair_arrangements : ℕ := 2  -- uppercase-lowercase or lowercase-uppercase

  num_uppercase * num_lowercase *
  num_letter_pair_arrangements *
  num_letter_pair_positions *
  num_digits ^ num_digit_positions

theorem correct_num_license_plates : num_license_plates = 809280000 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_license_plates_l1819_181923


namespace NUMINAMATH_CALUDE_total_cans_is_twelve_l1819_181982

/-- Represents the ratio of chili beans to tomato soup -/
def chili_to_tomato_ratio : ℚ := 2

/-- Represents the number of chili bean cans ordered -/
def chili_beans_ordered : ℕ := 8

/-- Calculates the total number of cans ordered -/
def total_cans_ordered : ℕ :=
  chili_beans_ordered + (chili_beans_ordered / chili_to_tomato_ratio.num).toNat

/-- Proves that the total number of cans ordered is 12 -/
theorem total_cans_is_twelve : total_cans_ordered = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_is_twelve_l1819_181982


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_difference_of_squares_l1819_181900

theorem largest_common_divisor_of_difference_of_squares (m n : ℤ) 
  (h_m_even : Even m) (h_n_odd : Odd n) (h_n_lt_m : n < m) :
  (∀ k : ℤ, k ∣ (m^2 - n^2) → k ≤ 2) ∧ 2 ∣ (m^2 - n^2) := by
  sorry

#check largest_common_divisor_of_difference_of_squares

end NUMINAMATH_CALUDE_largest_common_divisor_of_difference_of_squares_l1819_181900


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1819_181996

/-- An infinite geometric series with first term a and common ratio r has sum S if and only if |r| < 1 and S = a / (1 - r) -/
def is_infinite_geometric_series_sum (a : ℝ) (r : ℝ) (S : ℝ) : Prop :=
  |r| < 1 ∧ S = a / (1 - r)

/-- The positive common ratio of an infinite geometric series with first term 500 and sum 4000 is 7/8 -/
theorem infinite_geometric_series_ratio : 
  ∃ (r : ℝ), r > 0 ∧ is_infinite_geometric_series_sum 500 r 4000 ∧ r = 7/8 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1819_181996


namespace NUMINAMATH_CALUDE_crowdfunding_highest_level_l1819_181927

theorem crowdfunding_highest_level (x : ℝ) : 
  x > 0 ∧ 
  7298 * x = 200000 → 
  ⌊1296 * x⌋ = 35534 :=
by
  sorry

end NUMINAMATH_CALUDE_crowdfunding_highest_level_l1819_181927


namespace NUMINAMATH_CALUDE_triangle_properties_l1819_181928

-- Define the triangle
def Triangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let DC := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB = 25 ∧
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0 ∧ -- Right angle at D
  AD / AB = 4 / 5 ∧ -- sin A = 4/5
  BD / BC = 1 / 5   -- sin C = 1/5

-- Theorem statement
theorem triangle_properties (A B C D : ℝ × ℝ) (h : Triangle A B C D) :
  let DC := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  DC = 40 * Real.sqrt 6 ∧
  1/2 * AD * BD = 150 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1819_181928


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1819_181918

theorem divisibility_theorem (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) :
  ¬(d ∣ a^(2^n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1819_181918


namespace NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l1819_181944

/-- Represents a convex polygon with diagonals drawn to create triangles -/
structure ConvexPolygonWithDiagonals where
  sides : ℕ
  triangles : ℕ
  diagonalTriangles : ℕ

/-- The property that needs to be proven -/
def impossibleHalfDiagonalTriangles (p : ConvexPolygonWithDiagonals) : Prop :=
  p.sides = 2016 ∧ p.triangles = 2014 → p.diagonalTriangles ≠ 1007

theorem convex_polygon_diagonal_triangles :
  ∀ p : ConvexPolygonWithDiagonals, impossibleHalfDiagonalTriangles p :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l1819_181944


namespace NUMINAMATH_CALUDE_z1_over_z2_value_l1819_181954

def complex_symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem z1_over_z2_value (z₁ z₂ : ℂ) :
  complex_symmetric_about_imaginary_axis z₁ z₂ →
  z₁ = 3 - I →
  z₁ / z₂ = -4/5 + 3/5 * I :=
by sorry

end NUMINAMATH_CALUDE_z1_over_z2_value_l1819_181954


namespace NUMINAMATH_CALUDE_min_value_expression_l1819_181930

theorem min_value_expression (α β : ℝ) (h1 : α ≠ 0) (h2 : |β| = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), x = |((β + α) / (1 + α * β))| → x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1819_181930


namespace NUMINAMATH_CALUDE_science_club_membership_l1819_181937

theorem science_club_membership (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : biology = 50)
  (h3 : chemistry = 40)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_science_club_membership_l1819_181937


namespace NUMINAMATH_CALUDE_soda_preference_result_l1819_181972

/-- The number of people who chose "Soda" in a survey about carbonated beverages -/
def soda_preference (total_surveyed : ℕ) (soda_angle : ℕ) : ℕ :=
  (total_surveyed * soda_angle) / 360

/-- Theorem stating that 243 people chose "Soda" in the survey -/
theorem soda_preference_result : soda_preference 540 162 = 243 := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_result_l1819_181972


namespace NUMINAMATH_CALUDE_range_of_m_l1819_181981

-- Define the proposition
def P (m : ℝ) : Prop := ∀ x : ℝ, 5^x + 3 > m

-- Theorem statement
theorem range_of_m :
  (∃ m : ℝ, P m) → (∀ m : ℝ, P m → m ≤ 3) ∧ (∀ y : ℝ, y < 3 → P y) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1819_181981


namespace NUMINAMATH_CALUDE_right_triangle_sin_identity_l1819_181960

theorem right_triangle_sin_identity (A B C : Real) (h1 : C = Real.pi / 2) (h2 : A + B = Real.pi / 2) :
  Real.sin A * Real.sin B * Real.sin (A - B) + 
  Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + 
  Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_identity_l1819_181960


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l1819_181946

def parallel_side1_length : ℕ := 25
def parallel_side2_length : ℕ := 37
def non_parallel_side1_length : ℕ := 20
def non_parallel_side2_length : ℕ := 24
def parallel_side_price : ℕ := 48
def non_parallel_side_price : ℕ := 60

theorem fence_cost_calculation :
  (parallel_side1_length * parallel_side_price) +
  (parallel_side2_length * parallel_side_price) +
  (non_parallel_side1_length * non_parallel_side_price) +
  (non_parallel_side2_length * non_parallel_side_price) = 5616 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_calculation_l1819_181946


namespace NUMINAMATH_CALUDE_negation_of_forall_not_prime_l1819_181911

theorem negation_of_forall_not_prime :
  (¬ (∀ n : ℕ, ¬ (Nat.Prime (2^n - 2)))) ↔ (∃ n : ℕ, Nat.Prime (2^n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_not_prime_l1819_181911


namespace NUMINAMATH_CALUDE_capital_ratio_specific_case_l1819_181933

/-- Given a total loss and Pyarelal's loss, calculate the ratio of Ashok's capital to Pyarelal's capital -/
def capital_ratio (total_loss : ℕ) (pyarelal_loss : ℕ) : ℕ × ℕ :=
  let ashok_loss := total_loss - pyarelal_loss
  (ashok_loss, pyarelal_loss)

/-- Theorem stating that given the specific losses, the capital ratio is 67:603 -/
theorem capital_ratio_specific_case :
  capital_ratio 670 603 = (67, 603) := by
  sorry

end NUMINAMATH_CALUDE_capital_ratio_specific_case_l1819_181933


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1819_181999

theorem arithmetic_calculation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1819_181999


namespace NUMINAMATH_CALUDE_point_on_line_l1819_181967

/-- If (m, n) and (m + 2, n + k) are two points on the line with equation x = 2y + 3, then k = 1 -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2*n + 3) → 
  (m + 2 = 2*(n + k) + 3) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1819_181967


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l1819_181976

theorem unequal_gender_probability :
  let n : ℕ := 12  -- Total number of children
  let p : ℚ := 1/2  -- Probability of each child being a boy (or girl)
  let total_outcomes : ℕ := 2^n  -- Total number of possible outcomes
  let equal_outcomes : ℕ := n.choose (n/2)  -- Number of outcomes with equal boys and girls
  (1 : ℚ) - (equal_outcomes : ℚ) / total_outcomes = 793/1024 :=
by sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l1819_181976


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1819_181951

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = (180 - x) / 2) → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1819_181951


namespace NUMINAMATH_CALUDE_common_area_rectangle_circle_l1819_181957

/-- The area of the region common to a 10 by 4 rectangle and a circle with radius 3, sharing the same center, is equal to 9π. -/
theorem common_area_rectangle_circle :
  let rectangle_width : ℝ := 10
  let rectangle_height : ℝ := 4
  let circle_radius : ℝ := 3
  let circle_area : ℝ := π * circle_radius^2
  (∀ x y, x^2 / (rectangle_width/2)^2 + y^2 / (rectangle_height/2)^2 ≤ 1 → x^2 + y^2 ≤ circle_radius^2) →
  circle_area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_common_area_rectangle_circle_l1819_181957


namespace NUMINAMATH_CALUDE_ketchup_tomatoes_ratio_l1819_181932

/-- Given that 3 liters of ketchup require 69 kg of tomatoes, 
    prove that 5 liters of ketchup require 115 kg of tomatoes. -/
theorem ketchup_tomatoes_ratio (tomatoes_for_three : ℝ) (ketchup_liters : ℝ) : 
  tomatoes_for_three = 69 → ketchup_liters = 5 → 
  (tomatoes_for_three / 3) * ketchup_liters = 115 := by
sorry

end NUMINAMATH_CALUDE_ketchup_tomatoes_ratio_l1819_181932


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l1819_181904

def cricket_team_problem (team_size : ℕ) (avg_age : ℚ) (wicket_keeper_age_diff : ℚ) : Prop :=
  let total_age : ℚ := team_size * avg_age
  let wicket_keeper_age : ℚ := avg_age + wicket_keeper_age_diff
  let remaining_total_age : ℚ := total_age - wicket_keeper_age - avg_age
  let remaining_team_size : ℕ := team_size - 2
  let remaining_avg_age : ℚ := remaining_total_age / remaining_team_size
  (avg_age - remaining_avg_age) = 0.3

theorem cricket_team_age_difference :
  cricket_team_problem 11 24 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l1819_181904


namespace NUMINAMATH_CALUDE_volume_specific_pyramid_l1819_181910

/-- A triangular pyramid with specific edge lengths -/
structure TriangularPyramid where
  edge_opposite1 : ℝ
  edge_opposite2 : ℝ
  edge_other : ℝ

/-- Volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific triangular pyramid is 24 cm³ -/
theorem volume_specific_pyramid :
  let p : TriangularPyramid := {
    edge_opposite1 := 4,
    edge_opposite2 := 12,
    edge_other := 7
  }
  volume p = 24 := by sorry

end NUMINAMATH_CALUDE_volume_specific_pyramid_l1819_181910


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l1819_181993

/-- A quadratic function f(x) = kx^2 - 7x - 7 intersects the x-axis if and only if
    k ≥ -7/4 and k ≠ 0 -/
theorem quadratic_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l1819_181993


namespace NUMINAMATH_CALUDE_n_sided_polygon_exterior_angle_l1819_181992

theorem n_sided_polygon_exterior_angle (n : ℕ) : 
  (n ≠ 0) → (40 * n = 360) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_n_sided_polygon_exterior_angle_l1819_181992


namespace NUMINAMATH_CALUDE_rex_cards_left_l1819_181935

def nicole_cards : ℕ := 500

def cindy_cards : ℕ := (2 * nicole_cards) + (2 * nicole_cards * 25 / 100)

def total_cards : ℕ := nicole_cards + cindy_cards

def rex_cards : ℕ := (2 * total_cards) / 3

def num_siblings : ℕ := 5

def cards_per_person : ℕ := rex_cards / (num_siblings + 1)

def cards_given_away : ℕ := cards_per_person * num_siblings

theorem rex_cards_left : rex_cards - cards_given_away = 196 := by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_l1819_181935


namespace NUMINAMATH_CALUDE_karlson_candies_theorem_l1819_181977

/-- The number of ones initially on the board -/
def initial_ones : ℕ := 33

/-- The number of minutes Karlson has -/
def total_minutes : ℕ := 33

/-- The maximum number of candies Karlson can eat -/
def max_candies : ℕ := initial_ones.choose 2

/-- Theorem stating that the maximum number of candies Karlson can eat
    is equal to the number of unique pairs from the initial ones -/
theorem karlson_candies_theorem :
  max_candies = (initial_ones * (initial_ones - 1)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_karlson_candies_theorem_l1819_181977
