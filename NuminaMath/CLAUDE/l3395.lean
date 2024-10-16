import Mathlib

namespace NUMINAMATH_CALUDE_dans_egg_purchase_l3395_339538

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Dan bought -/
def total_eggs : ℕ := 108

/-- The number of dozens of eggs Dan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem dans_egg_purchase : dozens_bought = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_egg_purchase_l3395_339538


namespace NUMINAMATH_CALUDE_min_value_theorem_l3395_339563

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 5) :
  (9 / a) + (16 / b) + (25 / c) ≥ 144 / 5 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 5 ∧
    (9 / a') + (16 / b') + (25 / c') = 144 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3395_339563


namespace NUMINAMATH_CALUDE_percentage_problem_l3395_339564

theorem percentage_problem (P : ℝ) : 
  (0.15 * 0.30 * (P / 100) * 4400 = 99) → P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3395_339564


namespace NUMINAMATH_CALUDE_greyson_payment_l3395_339530

/-- The number of dimes in a dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 5

/-- The number of dimes Greyson paid for the watch -/
def dimes_paid : ℕ := watch_cost * dimes_per_dollar

theorem greyson_payment : dimes_paid = 50 := by
  sorry

end NUMINAMATH_CALUDE_greyson_payment_l3395_339530


namespace NUMINAMATH_CALUDE_smallest_power_complex_equality_l3395_339545

theorem smallest_power_complex_equality (n : ℕ) (c d : ℝ) :
  (n > 0) →
  (c > 0) →
  (d > 0) →
  (∀ k < n, ∃ a b : ℝ, (a > 0 ∧ b > 0 ∧ (a + b * I) ^ (2 * k) ≠ (a - b * I) ^ (2 * k))) →
  ((c + d * I) ^ (2 * n) = (c - d * I) ^ (2 * n)) →
  (d / c = 1) := by
sorry

end NUMINAMATH_CALUDE_smallest_power_complex_equality_l3395_339545


namespace NUMINAMATH_CALUDE_catch_turtle_certain_l3395_339586

-- Define the type for idioms
inductive Idiom
| CatchTurtle
| CarveBoat
| WaitRabbit
| FishMoon

-- Define a function to determine if an idiom represents a certain event
def isCertainEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.CatchTurtle => True
  | _ => False

-- Theorem statement
theorem catch_turtle_certain :
  ∀ i : Idiom, isCertainEvent i ↔ i = Idiom.CatchTurtle :=
by
  sorry


end NUMINAMATH_CALUDE_catch_turtle_certain_l3395_339586


namespace NUMINAMATH_CALUDE_restaurant_bill_fraction_l3395_339506

theorem restaurant_bill_fraction (akshitha veena lasya total : ℚ) : 
  akshitha = (3 / 4) * veena →
  veena = (1 / 2) * lasya →
  total = akshitha + veena + lasya →
  veena / total = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_fraction_l3395_339506


namespace NUMINAMATH_CALUDE_toy_cost_price_l3395_339532

/-- Given a man sold 18 toys for Rs. 25200 and gained the cost price of 3 toys,
    prove that the cost price of a single toy is Rs. 1200. -/
theorem toy_cost_price (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) 
    (h1 : total_selling_price = 25200)
    (h2 : num_toys_sold = 18)
    (h3 : num_toys_gain = 3) :
  ∃ (cost_price : ℕ), cost_price = 1200 ∧ 
    total_selling_price = num_toys_sold * (cost_price + (num_toys_gain * cost_price) / num_toys_sold) :=
by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l3395_339532


namespace NUMINAMATH_CALUDE_faucet_turning_is_rotational_motion_l3395_339512

/-- A motion that involves revolving around a center and changing direction -/
structure FaucetTurning where
  revolves_around_center : Bool
  direction_changes : Bool

/-- Definition of rotational motion -/
def is_rotational_motion (motion : FaucetTurning) : Prop :=
  motion.revolves_around_center ∧ motion.direction_changes

/-- Theorem: Turning a faucet by hand is a rotational motion -/
theorem faucet_turning_is_rotational_motion :
  ∀ (faucet_turning : FaucetTurning),
  faucet_turning.revolves_around_center = true →
  faucet_turning.direction_changes = true →
  is_rotational_motion faucet_turning :=
by
  sorry

end NUMINAMATH_CALUDE_faucet_turning_is_rotational_motion_l3395_339512


namespace NUMINAMATH_CALUDE_find_D_l3395_339555

theorem find_D (A B C D : ℤ) 
  (h1 : A + C = 15)
  (h2 : A - B = 1)
  (h3 : C + C = A)
  (h4 : B - D = 2)
  (h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  D = 7 := by
sorry

end NUMINAMATH_CALUDE_find_D_l3395_339555


namespace NUMINAMATH_CALUDE_angle_C_measure_l3395_339578

theorem angle_C_measure (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Given conditions
  (a = 2 * Real.sqrt 6) →
  (b = 6) →
  (Real.cos B = -1/2) →
  -- Conclusion
  C = π/12 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l3395_339578


namespace NUMINAMATH_CALUDE_biff_ticket_cost_l3395_339547

/-- Represents the cost of Biff's bus ticket in dollars -/
def ticket_cost : ℝ := 11

/-- Represents the cost of drinks and snacks in dollars -/
def snacks_cost : ℝ := 3

/-- Represents the cost of headphones in dollars -/
def headphones_cost : ℝ := 16

/-- Represents Biff's hourly rate for online work in dollars per hour -/
def online_rate : ℝ := 12

/-- Represents the hourly cost of WiFi access in dollars per hour -/
def wifi_cost : ℝ := 2

/-- Represents the duration of the bus trip in hours -/
def trip_duration : ℝ := 3

theorem biff_ticket_cost :
  ticket_cost + snacks_cost + headphones_cost + wifi_cost * trip_duration =
  online_rate * trip_duration :=
by sorry

end NUMINAMATH_CALUDE_biff_ticket_cost_l3395_339547


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3395_339588

theorem ceiling_sum_sqrt : ⌈Real.sqrt 19⌉ + ⌈Real.sqrt 57⌉ + ⌈Real.sqrt 119⌉ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3395_339588


namespace NUMINAMATH_CALUDE_secant_min_value_l3395_339505

/-- The secant function -/
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

/-- The function y = a sec(bx) -/
noncomputable def f (a b x : ℝ) : ℝ := a * sec (b * x)

theorem secant_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≥ a) ∧ (∃ x, f a b x = a) →
  (∀ x, f a b x ≥ 3) ∧ (∃ x, f a b x = 3) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_secant_min_value_l3395_339505


namespace NUMINAMATH_CALUDE_hillary_activities_lcm_l3395_339536

theorem hillary_activities_lcm : Nat.lcm 6 (Nat.lcm 4 (Nat.lcm 16 (Nat.lcm 12 8))) = 48 := by
  sorry

end NUMINAMATH_CALUDE_hillary_activities_lcm_l3395_339536


namespace NUMINAMATH_CALUDE_plane_division_l3395_339508

/-- The maximum number of parts a plane can be divided into by n lines -/
def f (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of parts a plane can be divided into by n lines is (n^2 + n + 2) / 2 -/
theorem plane_division (n : ℕ) : f n = (n^2 + n + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l3395_339508


namespace NUMINAMATH_CALUDE_total_cost_is_54_44_l3395_339566

-- Define the quantities and prices
def book_quantity : ℕ := 1
def book_price : ℚ := 16
def binder_quantity : ℕ := 3
def binder_price : ℚ := 2
def notebook_quantity : ℕ := 6
def notebook_price : ℚ := 1
def pen_quantity : ℕ := 4
def pen_price : ℚ := 1/2
def calculator_quantity : ℕ := 2
def calculator_price : ℚ := 12

-- Define discount and tax rates
def discount_rate : ℚ := 1/10
def tax_rate : ℚ := 7/100

-- Define the total cost function
def total_cost : ℚ :=
  let book_cost := book_quantity * book_price
  let binder_cost := binder_quantity * binder_price
  let notebook_cost := notebook_quantity * notebook_price
  let pen_cost := pen_quantity * pen_price
  let calculator_cost := calculator_quantity * calculator_price
  
  let discounted_book_cost := book_cost * (1 - discount_rate)
  let discounted_binder_cost := binder_cost * (1 - discount_rate)
  
  let subtotal := discounted_book_cost + discounted_binder_cost + notebook_cost + pen_cost + calculator_cost
  let tax := (notebook_cost + pen_cost + calculator_cost) * tax_rate
  
  subtotal + tax

-- Theorem statement
theorem total_cost_is_54_44 : total_cost = 5444 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_54_44_l3395_339566


namespace NUMINAMATH_CALUDE_sugar_for_partial_recipe_result_as_mixed_number_l3395_339529

-- Define the original amount of sugar in the recipe
def original_sugar : ℚ := 19/3

-- Define the fraction of the recipe we want to make
def recipe_fraction : ℚ := 1/3

-- Theorem statement
theorem sugar_for_partial_recipe :
  recipe_fraction * original_sugar = 19/9 :=
by sorry

-- Convert the result to a mixed number
theorem result_as_mixed_number :
  ∃ (whole : ℕ) (num denom : ℕ), 
    recipe_fraction * original_sugar = whole + num / denom ∧
    whole = 2 ∧ num = 1 ∧ denom = 9 :=
by sorry

end NUMINAMATH_CALUDE_sugar_for_partial_recipe_result_as_mixed_number_l3395_339529


namespace NUMINAMATH_CALUDE_max_sine_cosine_function_l3395_339525

theorem max_sine_cosine_function (a b : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x ≤ 4) ∧
  (a * Real.sin (π/3) + b * Real.cos (π/3) = 4) →
  a / b = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_sine_cosine_function_l3395_339525


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3395_339599

/-- Given a parabola y = -x^2 + (k+1)x - k where (4,0) lies on the parabola,
    prove that the intersection point of the parabola with the y-axis is (0, -4). -/
theorem parabola_y_axis_intersection
  (k : ℝ)
  (h : 0 = -(4^2) + (k+1)*4 - k) :
  ∃ y, y = -4 ∧ 0 = -(0^2) + (k+1)*0 - k ∧ y = -(0^2) + (k+1)*0 - k :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3395_339599


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3395_339579

-- Define the number of pages read per night
def pages_per_night : ℝ := 120.0

-- Define the number of days of reading
def days_of_reading : ℝ := 10.0

-- Define the total number of pages in the book
def total_pages : ℝ := pages_per_night * days_of_reading

-- Theorem statement
theorem book_pages_calculation : total_pages = 1200.0 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3395_339579


namespace NUMINAMATH_CALUDE_inequality_solution_l3395_339554

-- Define the inequality and its solution set
def inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 6 > 4
def solution_set (x : ℝ) : Prop := x < 1 ∨ x > 2

-- State the theorem
theorem inequality_solution :
  (∀ x, inequality 1 x ↔ solution_set x) →
  (∀ c, 
    (c = -2 → ∀ x, ¬((c - x) * (x + 2) > 0)) ∧
    (c > -2 → ∀ x, (c - x) * (x + 2) > 0 ↔ -2 < x ∧ x < c) ∧
    (c < -2 → ∀ x, (c - x) * (x + 2) > 0 ↔ c < x ∧ x < -2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3395_339554


namespace NUMINAMATH_CALUDE_person_peach_count_l3395_339540

theorem person_peach_count (jake_peaches jake_apples person_apples person_peaches : ℕ) : 
  jake_peaches + 6 = person_peaches →
  jake_apples = person_apples + 8 →
  person_apples = 16 →
  person_peaches = person_apples + 1 →
  person_peaches = 17 := by
sorry

end NUMINAMATH_CALUDE_person_peach_count_l3395_339540


namespace NUMINAMATH_CALUDE_slope_angle_expression_l3395_339527

theorem slope_angle_expression (x y : ℝ) (α : ℝ) : 
  (6 * x - 2 * y - 5 = 0) →
  (Real.tan α = 3) →
  ((Real.sin (π - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (π + α)) = -2) := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_expression_l3395_339527


namespace NUMINAMATH_CALUDE_triangle_area_bounds_l3395_339531

/-- Given a triangle with sides a, b, c, this theorem proves bounds on its area S. -/
theorem triangle_area_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := S / p
  3 * Real.sqrt 3 * r^2 ≤ S ∧ S ≤ p^2 / (3 * Real.sqrt 3) ∧
  S ≤ (a^2 + b^2 + c^2) / (4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_bounds_l3395_339531


namespace NUMINAMATH_CALUDE_complex_power_quadrant_l3395_339500

theorem complex_power_quadrant : ∃ (z : ℂ), z = (Complex.I + 1) / Real.sqrt 2 ∧ 
  (z^2015).re > 0 ∧ (z^2015).im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_quadrant_l3395_339500


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l3395_339539

-- Define the total amount spent excluding taxes
variable (T : ℝ)

-- Define the tax rate on clothing as a percentage
variable (x : ℝ)

-- Define the spending percentages
def clothing_percent : ℝ := 0.45
def food_percent : ℝ := 0.45
def other_percent : ℝ := 0.10

-- Define the tax rates
def other_tax_rate : ℝ := 0.10
def total_tax_rate : ℝ := 0.0325

-- Theorem statement
theorem clothing_tax_rate :
  clothing_percent * T * (x / 100) + other_percent * T * other_tax_rate = total_tax_rate * T →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l3395_339539


namespace NUMINAMATH_CALUDE_positive_sum_inequality_l3395_339596

theorem positive_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_inequality_l3395_339596


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l3395_339560

theorem five_dice_not_same_probability :
  let n := 8  -- number of sides on each die
  let k := 5  -- number of dice rolled
  (1 - (n : ℚ) / n^k) = 4095 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l3395_339560


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l3395_339522

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part I
theorem solution_set_part_I :
  ∀ x : ℝ, f (-2) x + f (-2) (2*x) > 2 ↔ x < -2 ∨ x > -2/3 :=
sorry

-- Theorem for part II
theorem range_of_a_part_II :
  ∀ a : ℝ, a < 0 → (∃ x : ℝ, f a x + f a (2*x) < 1/2) → -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l3395_339522


namespace NUMINAMATH_CALUDE_family_ages_l3395_339507

theorem family_ages (oleg_age : ℕ) (father_age : ℕ) (grandfather_age : ℕ) :
  father_age = oleg_age + 32 →
  grandfather_age = father_age + 32 →
  (oleg_age - 3) + (father_age - 3) + (grandfather_age - 3) < 100 →
  oleg_age > 0 →
  oleg_age = 4 ∧ father_age = 36 ∧ grandfather_age = 68 := by
  sorry

#check family_ages

end NUMINAMATH_CALUDE_family_ages_l3395_339507


namespace NUMINAMATH_CALUDE_f_minimum_value_l3395_339577

def f (x : ℝ) : ℝ := |x - 1| + |x - 2| - |x - 3|

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ -1) ∧ (∃ x : ℝ, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l3395_339577


namespace NUMINAMATH_CALUDE_quadrilateral_division_theorem_l3395_339552

/-- Represents a convex quadrilateral with areas of its four parts --/
structure ConvexQuadrilateral :=
  (area1 : ℝ)
  (area2 : ℝ)
  (area3 : ℝ)
  (area4 : ℝ)

/-- The theorem stating the relationship between the areas of the four parts --/
theorem quadrilateral_division_theorem (q : ConvexQuadrilateral) 
  (h1 : q.area1 = 360)
  (h2 : q.area2 = 720)
  (h3 : q.area3 = 900) :
  q.area4 = 540 := by
  sorry

#check quadrilateral_division_theorem

end NUMINAMATH_CALUDE_quadrilateral_division_theorem_l3395_339552


namespace NUMINAMATH_CALUDE_problem_statement_l3395_339575

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -24)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3395_339575


namespace NUMINAMATH_CALUDE_extremum_at_two_min_value_of_sum_l3395_339571

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem extremum_at_two (a : ℝ) : f_deriv a 2 = 0 ↔ a = 3 := by sorry

theorem min_value_of_sum (m n : ℝ) (hm : m ∈ Set.Icc (-1 : ℝ) 1) (hn : n ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ (a : ℝ), f_deriv a 2 = 0 ∧ f a m + f_deriv a n ≥ -13 ∧
  ∃ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 ∧ n' ∈ Set.Icc (-1 : ℝ) 1 ∧ f a m' + f_deriv a n' = -13 := by sorry

end NUMINAMATH_CALUDE_extremum_at_two_min_value_of_sum_l3395_339571


namespace NUMINAMATH_CALUDE_arithmetic_sequence_slope_l3395_339524

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2

/-- The slope of the line passing through P(n, a_n) and Q(n+2, a_{n+2}) is 4 -/
theorem arithmetic_sequence_slope (seq : ArithmeticSequence) 
    (h1 : seq.S 2 = 10) (h2 : seq.S 5 = 55) :
    ∀ n : ℕ+, (seq.a (n + 2) - seq.a n) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_slope_l3395_339524


namespace NUMINAMATH_CALUDE_intersection_point_sum_l3395_339518

/-- Two lines in a plane -/
structure TwoLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ

/-- Points P, Q, and T for the given lines -/
structure LinePoints (l : TwoLines) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  T : ℝ × ℝ
  h_P : l.line1 P.1 = P.2 ∧ P.2 = 0
  h_Q : l.line1 Q.1 = Q.2 ∧ Q.1 = 0
  h_T : l.line1 T.1 = T.2 ∧ l.line2 T.1 = T.2

/-- The theorem statement -/
theorem intersection_point_sum (l : TwoLines) (pts : LinePoints l) 
  (h_line1 : ∀ x, l.line1 x = -2/3 * x + 8)
  (h_line2 : ∀ x, l.line2 x = 3/2 * x - 9)
  (h_area : (pts.P.1 * pts.Q.2) / 2 = 2 * ((pts.P.1 - pts.T.1) * pts.T.2) / 2) :
  pts.T.1 + pts.T.2 = 138/13 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l3395_339518


namespace NUMINAMATH_CALUDE_line_l_equation_line_l₃_equation_l3395_339558

-- Define the point M
def M : ℝ × ℝ := (3, 0)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the line l
def l (x y : ℝ) : Prop := 8 * x - y - 24 = 0

-- Define the line l₃
def l₃ (x y : ℝ) : Prop := x - 2 * y - 5 = 0

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∃ (P Q : ℝ × ℝ), 
    l₁ P.1 P.2 ∧ 
    l₂ Q.1 Q.2 ∧ 
    l M.1 M.2 ∧ 
    l P.1 P.2 ∧ 
    l Q.1 Q.2 ∧ 
    M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) :=
sorry

-- Theorem for the equation of line l₃
theorem line_l₃_equation :
  ∃ (I : ℝ × ℝ), 
    l₁ I.1 I.2 ∧ 
    l₂ I.1 I.2 ∧ 
    l₃ I.1 I.2 ∧
    ∀ (x y : ℝ), l₃ x y ↔ x - 2 * y - 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_line_l₃_equation_l3395_339558


namespace NUMINAMATH_CALUDE_min_value_theorem_l3395_339559

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3395_339559


namespace NUMINAMATH_CALUDE_fourth_root_of_390625_l3395_339589

theorem fourth_root_of_390625 (x : ℝ) (h1 : x > 0) (h2 : x^4 = 390625) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_390625_l3395_339589


namespace NUMINAMATH_CALUDE_third_number_proof_l3395_339501

def mean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem third_number_proof (x : ℕ) (y : ℕ) :
  mean [28, x, y, 78, 104] = 90 →
  mean [128, 255, 511, 1023, x] = 423 →
  y = 42 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l3395_339501


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_ratio_quadratic_function_minimum_ratio_exact_l3395_339561

/-- A quadratic function f(x) = ax^2 + bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  second_derivative_positive : 2 * a > 0
  non_negative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for the quadratic function -/
theorem quadratic_function_minimum_ratio (f : QuadraticFunction) :
  (f.a + f.b + f.c) / (2 * f.a) ≥ 2 := by
  sorry

/-- The theorem stating that the minimum value of f(1) / f''(0) is exactly 2 -/
theorem quadratic_function_minimum_ratio_exact :
  ∃ f : QuadraticFunction, (f.a + f.b + f.c) / (2 * f.a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_ratio_quadratic_function_minimum_ratio_exact_l3395_339561


namespace NUMINAMATH_CALUDE_pascals_identity_l3395_339553

theorem pascals_identity (n k : ℕ) : 
  Nat.choose n k + Nat.choose n (k + 1) = Nat.choose (n + 1) (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_pascals_identity_l3395_339553


namespace NUMINAMATH_CALUDE_surface_a_properties_surface_b_properties_surface_c_properties_l3395_339587

-- Part (a)
def surface_a1 (x y z : ℝ) : Prop := 2 * y = x^2 + z^2
def surface_a2 (x y z : ℝ) : Prop := x^2 + z^2 = 1

theorem surface_a_properties (x y z : ℝ) :
  surface_a1 x y z ∧ surface_a2 x y z → y ≥ 0 :=
sorry

-- Part (b)
def surface_b1 (x y z : ℝ) : Prop := z = 0
def surface_b2 (x y z : ℝ) : Prop := y + z = 2
def surface_b3 (x y z : ℝ) : Prop := y = x^2

theorem surface_b_properties (x y z : ℝ) :
  surface_b1 x y z ∧ surface_b2 x y z ∧ surface_b3 x y z → 
  y ≤ 2 ∧ y ≥ 0 ∧ z ≤ 2 ∧ z ≥ 0 :=
sorry

-- Part (c)
def surface_c1 (x y z : ℝ) : Prop := z = 6 - x^2 - y^2
def surface_c2 (x y z : ℝ) : Prop := x^2 + y^2 - z^2 = 0

theorem surface_c_properties (x y z : ℝ) :
  surface_c1 x y z ∧ surface_c2 x y z → 
  z ≤ 3 ∧ z ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_surface_a_properties_surface_b_properties_surface_c_properties_l3395_339587


namespace NUMINAMATH_CALUDE_inequality_proof_l3395_339573

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  (1 + x + x^2) * (1 + x + x^2 + x^3 + x^4) ≤ (1 + x + x^2 + x^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3395_339573


namespace NUMINAMATH_CALUDE_painting_time_is_18_17_l3395_339568

/-- The time required for three painters to complete a room, given their individual rates and break times -/
def total_painting_time (linda_rate tom_rate jerry_rate : ℚ) (tom_break jerry_break : ℚ) : ℚ :=
  let combined_rate := linda_rate + tom_rate + jerry_rate
  18 / 17

/-- Theorem stating that the total painting time for Linda, Tom, and Jerry is 18/17 hours -/
theorem painting_time_is_18_17 :
  let linda_rate : ℚ := 1 / 3
  let tom_rate : ℚ := 1 / 4
  let jerry_rate : ℚ := 1 / 6
  let tom_break : ℚ := 2
  let jerry_break : ℚ := 1
  total_painting_time linda_rate tom_rate jerry_rate tom_break jerry_break = 18 / 17 := by
  sorry

#eval total_painting_time (1/3) (1/4) (1/6) 2 1

end NUMINAMATH_CALUDE_painting_time_is_18_17_l3395_339568


namespace NUMINAMATH_CALUDE_periodic_odd_quadratic_function_properties_l3395_339535

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f (-x) = -f x

def is_quadratic_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ A B C : ℝ, ∀ x, a ≤ x ∧ x ≤ b → f x = A * x^2 + B * x + C

theorem periodic_odd_quadratic_function_properties
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 5)
  (h_odd : is_odd_on_interval f (-1) 1)
  (h_quadratic : is_quadratic_on_interval f 1 4)
  (h_min : f 2 = -5 ∧ ∀ x, f x ≥ -5) :
  (f 1 + f 4 = 0) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 4 → f x = 2 * (x - 2)^2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_periodic_odd_quadratic_function_properties_l3395_339535


namespace NUMINAMATH_CALUDE_horner_method_v3_equals_20_l3395_339542

def f (x : ℝ) : ℝ := 2*x^5 + 3*x^3 - 2*x^2 + x - 1

def horner_v3 (a : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x
  let v2 := (v1 + 3) * x - 2
  (v2 * x + 1) * x - 1

theorem horner_method_v3_equals_20 :
  horner_v3 f 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_equals_20_l3395_339542


namespace NUMINAMATH_CALUDE_winning_strategy_correct_l3395_339519

/-- A stone-picking game with two players. -/
structure StoneGame where
  total_stones : ℕ
  min_take : ℕ
  max_take : ℕ

/-- A winning strategy for the first player in the stone game. -/
def winning_first_move (game : StoneGame) : ℕ := 3

/-- Theorem stating that the winning strategy for the first player is correct. -/
theorem winning_strategy_correct (game : StoneGame) 
  (h1 : game.total_stones = 18)
  (h2 : game.min_take = 1)
  (h3 : game.max_take = 4) :
  ∃ (n : ℕ), n ≥ game.min_take ∧ n ≤ game.max_take ∧
  (winning_first_move game = n → 
   ∀ (m : ℕ), m ≥ game.min_take → m ≤ game.max_take → 
   ∃ (k : ℕ), k ≥ game.min_take ∧ k ≤ game.max_take ∧
   (game.total_stones - n - m - k) % 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_correct_l3395_339519


namespace NUMINAMATH_CALUDE_mary_seashells_l3395_339504

theorem mary_seashells (sam_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : sam_seashells = 18) 
  (h2 : total_seashells = 65) : 
  total_seashells - sam_seashells = 47 := by
  sorry

end NUMINAMATH_CALUDE_mary_seashells_l3395_339504


namespace NUMINAMATH_CALUDE_convex_quadrilateral_from_circles_in_square_l3395_339570

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ

/-- Predicate to check if a point is inside a circle -/
def isInsideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Theorem statement -/
theorem convex_quadrilateral_from_circles_in_square 
  (s : Square) 
  (c1 c2 c3 c4 : Circle) 
  (p1 p2 p3 p4 : Point) : 
  -- The circles are centered at the vertices of the square
  (c1.center.x = 0 ∧ c1.center.y = 0) →
  (c2.center.x = s.sideLength ∧ c2.center.y = 0) →
  (c3.center.x = s.sideLength ∧ c3.center.y = s.sideLength) →
  (c4.center.x = 0 ∧ c4.center.y = s.sideLength) →
  -- The sum of the areas of the circles equals the area of the square
  (π * (c1.radius^2 + c2.radius^2 + c3.radius^2 + c4.radius^2) = s.sideLength^2) →
  -- The points are inside their respective circles
  isInsideCircle p1 c1 →
  isInsideCircle p2 c2 →
  isInsideCircle p3 c3 →
  isInsideCircle p4 c4 →
  -- The four points form a convex quadrilateral
  ∃ (a b c : ℝ), a * p1.x + b * p1.y + c < 0 ∧
                 a * p2.x + b * p2.y + c < 0 ∧
                 a * p3.x + b * p3.y + c > 0 ∧
                 a * p4.x + b * p4.y + c > 0 :=
by sorry


end NUMINAMATH_CALUDE_convex_quadrilateral_from_circles_in_square_l3395_339570


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l3395_339582

def B_current_age : ℕ := 39
def A_current_age : ℕ := B_current_age + 9

def A_future_age : ℕ := A_current_age + 10
def B_past_age : ℕ := B_current_age - 10

theorem age_ratio_is_two_to_one :
  A_future_age / B_past_age = 2 ∧ B_past_age ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l3395_339582


namespace NUMINAMATH_CALUDE_escalator_walking_speed_l3395_339562

/-- Proves that a person walks at 5 ft/sec on an escalator given specific conditions -/
theorem escalator_walking_speed 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15) 
  (h2 : escalator_length = 200) 
  (h3 : time_taken = 10) : 
  ∃ (walking_speed : ℝ), 
    walking_speed = 5 ∧ 
    escalator_length = (walking_speed + escalator_speed) * time_taken :=
by
  sorry

end NUMINAMATH_CALUDE_escalator_walking_speed_l3395_339562


namespace NUMINAMATH_CALUDE_min_ratio_four_digit_number_l3395_339537

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest value of n/s_n for four-digit numbers is 1099/19 -/
theorem min_ratio_four_digit_number :
  ∀ n : ℕ, 1000 ≤ n → n ≤ 9999 → (n : ℚ) / (sum_of_digits n) ≥ 1099 / 19 := by
  sorry

end NUMINAMATH_CALUDE_min_ratio_four_digit_number_l3395_339537


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l3395_339590

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l3395_339590


namespace NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l3395_339521

theorem percentage_of_boys_studying_science 
  (total_boys : ℕ) 
  (school_A_percentage : ℚ) 
  (non_science_boys : ℕ) 
  (h1 : total_boys = 300)
  (h2 : school_A_percentage = 1/5)
  (h3 : non_science_boys = 42) :
  (↑((school_A_percentage * ↑total_boys - ↑non_science_boys) / (school_A_percentage * ↑total_boys)) : ℚ) = 3/10 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l3395_339521


namespace NUMINAMATH_CALUDE_function_properties_l3395_339520

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_one_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period_neg : has_period_one_negation f) : 
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ (x : ℝ) (k : ℤ), f (2 * ↑k - x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3395_339520


namespace NUMINAMATH_CALUDE_three_zeros_condition_l3395_339557

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Theorem stating the condition for f to have exactly 3 zeros -/
theorem three_zeros_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ 
  a < -3 ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l3395_339557


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_7_l3395_339569

theorem cos_2alpha_minus_3pi_over_7 (α : ℝ) 
  (h : Real.sin (α + 2 * Real.pi / 7) = Real.sqrt 6 / 3) : 
  Real.cos (2 * α - 3 * Real.pi / 7) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_7_l3395_339569


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l3395_339509

/-- Parabola type representing y = ax² --/
structure Parabola where
  a : ℝ

/-- Point type representing (x, y) --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing y = mx + b --/
structure Line where
  m : ℝ
  b : ℝ

/-- Circle type representing (x - h)² + (y - k)² = r² --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Given conditions of the problem --/
def given (C : Parabola) (M A B : Point) : Prop :=
  M.y = C.a * M.x^2 ∧
  M.x = 2 ∧ M.y = 1 ∧
  A.y = C.a * A.x^2 ∧
  B.y = C.a * B.x^2 ∧
  A ≠ M ∧ B ≠ M ∧
  ∃ (circ : Circle), (A.x - circ.h)^2 + (A.y - circ.k)^2 = circ.r^2 ∧
                     (B.x - circ.h)^2 + (B.y - circ.k)^2 = circ.r^2 ∧
                     (M.x - circ.h)^2 + (M.y - circ.k)^2 = circ.r^2 ∧
                     circ.r = (A.x - B.x)^2 + (A.y - B.y)^2

/-- The main theorem to be proved --/
theorem parabola_circle_intersection 
  (C : Parabola) (M A B : Point) (h : given C M A B) :
  (∃ (l : Line), l.m * (-2) + l.b = 5 ∧ l.m * A.x + l.b = A.y ∧ l.m * B.x + l.b = B.y) ∧
  (∃ (N : Point), N.x^2 + (N.y - 3)^2 = 8 ∧ N.y ≠ 1 ∧
    (N.x - M.x) * (B.x - A.x) + (N.y - M.y) * (B.y - A.y) = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l3395_339509


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3395_339502

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.5)) 
  (h2 : P = G * (1 - 0.2)) : 
  R = G * (1 + 0.6) := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3395_339502


namespace NUMINAMATH_CALUDE_exists_set_equal_partitions_l3395_339583

/-- The type of positive integers -/
def PositiveInt : Type := { n : ℕ // n > 0 }

/-- Count of partitions where each number appears at most twice -/
def countLimitedPartitions (n : ℕ) : ℕ :=
  sorry

/-- Count of partitions using elements from a set -/
def countSetPartitions (n : ℕ) (S : Set PositiveInt) : ℕ :=
  sorry

/-- The existence of a set S satisfying the partition property -/
theorem exists_set_equal_partitions :
  ∃ (S : Set PositiveInt), ∀ (n : ℕ), n > 0 →
    countLimitedPartitions n = countSetPartitions n S :=
  sorry

end NUMINAMATH_CALUDE_exists_set_equal_partitions_l3395_339583


namespace NUMINAMATH_CALUDE_smallest_ending_nine_div_thirteen_l3395_339597

/-- A function that checks if a number ends with 9 -/
def endsWithNine (n : ℕ) : Prop := n % 10 = 9

/-- The theorem stating that 169 is the smallest positive integer ending in 9 and divisible by 13 -/
theorem smallest_ending_nine_div_thirteen :
  ∀ n : ℕ, n > 0 → endsWithNine n → n % 13 = 0 → n ≥ 169 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ending_nine_div_thirteen_l3395_339597


namespace NUMINAMATH_CALUDE_cookie_count_l3395_339523

theorem cookie_count (x y : ℕ) (hx : x = 137) (hy : y = 251) : x * y = 34387 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l3395_339523


namespace NUMINAMATH_CALUDE_election_total_votes_l3395_339513

-- Define the set of candidates
inductive Candidate : Type
  | Alicia : Candidate
  | Brenda : Candidate
  | Colby : Candidate
  | David : Candidate

-- Define the election
structure Election where
  totalVotes : ℕ
  brendaVotes : ℕ
  brendaPercentage : ℚ

-- Theorem statement
theorem election_total_votes (e : Election) 
  (h1 : e.brendaVotes = 40)
  (h2 : e.brendaPercentage = 1/4) :
  e.totalVotes = 160 := by
  sorry


end NUMINAMATH_CALUDE_election_total_votes_l3395_339513


namespace NUMINAMATH_CALUDE_line_segment_lattice_points_l3395_339515

/-- The number of lattice points on a line segment with given integer coordinates --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5, 23) to (53, 311) is 49 --/
theorem line_segment_lattice_points :
  latticePointCount 5 23 53 311 = 49 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_lattice_points_l3395_339515


namespace NUMINAMATH_CALUDE_sum_of_special_series_l3395_339533

def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_special_series :
  let a₁ := 1
  let d := 2
  let secondToLast := 99
  let last := 100
  let n := (secondToLast - a₁) / d + 1
  arithmeticSequenceSum a₁ d n + last = 2600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_series_l3395_339533


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_l3395_339556

-- Define the triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector
def angleBisector (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the intersection point
def intersectionPoint (p q r s : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

theorem angle_bisector_ratio (t : Triangle) :
  let D : ℝ × ℝ := ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)
  let E : ℝ × ℝ := ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2)
  let T : ℝ × ℝ := angleBisector t
  let F : ℝ × ℝ := intersectionPoint t.A T D E
  distance t.A D = distance D t.B ∧ 
  distance t.A E = distance E t.C ∧
  distance t.A D = 2 ∧
  distance t.A E = 3 →
  distance t.A F / distance t.A T = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_l3395_339556


namespace NUMINAMATH_CALUDE_exists_divisible_with_at_most_1988_ones_l3395_339510

/-- A natural number is representable with at most 1988 ones if its binary representation
    has at most 1988 ones. -/
def representable_with_at_most_1988_ones (n : ℕ) : Prop :=
  (n.digits 2).count 1 ≤ 1988

/-- For any natural number M, there exists a natural number N that is
    representable with at most 1988 ones and is divisible by M. -/
theorem exists_divisible_with_at_most_1988_ones (M : ℕ) :
  ∃ N : ℕ, representable_with_at_most_1988_ones N ∧ M ∣ N :=
by sorry


end NUMINAMATH_CALUDE_exists_divisible_with_at_most_1988_ones_l3395_339510


namespace NUMINAMATH_CALUDE_complement_union_A_B_range_of_m_when_B_subset_A_l3395_339549

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem complement_union_A_B : 
  (A ∪ B (-2))ᶜ = {x | x < -2 ∨ x > 2} := by sorry

-- Theorem 2
theorem range_of_m_when_B_subset_A : 
  ∀ m : ℝ, B m ⊆ A ↔ -1 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_range_of_m_when_B_subset_A_l3395_339549


namespace NUMINAMATH_CALUDE_tom_remaining_candy_l3395_339584

/-- The number of candy pieces Tom still has after giving some away to his brother -/
def remaining_candy_pieces : ℕ :=
  let initial_chocolate_boxes : ℕ := 14
  let initial_fruit_boxes : ℕ := 10
  let initial_caramel_boxes : ℕ := 8
  let given_chocolate_boxes : ℕ := 8
  let given_fruit_boxes : ℕ := 5
  let pieces_per_chocolate_box : ℕ := 3
  let pieces_per_fruit_box : ℕ := 4
  let pieces_per_caramel_box : ℕ := 5

  let initial_total_pieces : ℕ := 
    initial_chocolate_boxes * pieces_per_chocolate_box +
    initial_fruit_boxes * pieces_per_fruit_box +
    initial_caramel_boxes * pieces_per_caramel_box

  let given_away_pieces : ℕ := 
    given_chocolate_boxes * pieces_per_chocolate_box +
    given_fruit_boxes * pieces_per_fruit_box

  initial_total_pieces - given_away_pieces

theorem tom_remaining_candy : remaining_candy_pieces = 78 := by
  sorry

end NUMINAMATH_CALUDE_tom_remaining_candy_l3395_339584


namespace NUMINAMATH_CALUDE_four_circle_plus_two_l3395_339593

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem four_circle_plus_two : circle_plus 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_four_circle_plus_two_l3395_339593


namespace NUMINAMATH_CALUDE_expression_simplification_l3395_339514

theorem expression_simplification :
  let a := 16 / 2015
  let b := 17 / 2016
  (6 + a) * (9 + b) - (3 - a) * (18 - b) - 27 * a = 17 / 224 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3395_339514


namespace NUMINAMATH_CALUDE_charlottes_phone_usage_l3395_339594

/-- Charlotte's daily phone usage problem -/
theorem charlottes_phone_usage 
  (social_media_time : ℝ) 
  (weekly_social_media : ℝ) 
  (h1 : social_media_time = weekly_social_media / 7)
  (h2 : weekly_social_media = 56)
  (h3 : social_media_time = daily_phone_time / 2) : 
  daily_phone_time = 16 :=
sorry

end NUMINAMATH_CALUDE_charlottes_phone_usage_l3395_339594


namespace NUMINAMATH_CALUDE_car_trading_theorem_l3395_339511

/-- Represents the profit and purchase constraints for a car trading company. -/
structure CarTrading where
  profit_A_2_B_5 : ℕ  -- Profit from selling 2 A and 5 B
  profit_A_1_B_2 : ℕ  -- Profit from selling 1 A and 2 B
  price_A : ℕ         -- Purchase price of model A
  price_B : ℕ         -- Purchase price of model B
  total_budget : ℕ    -- Total budget
  total_units : ℕ     -- Total number of cars to purchase

/-- Theorem stating the profit per unit and minimum purchase of model A -/
theorem car_trading_theorem (ct : CarTrading) 
  (h1 : ct.profit_A_2_B_5 = 31000)
  (h2 : ct.profit_A_1_B_2 = 13000)
  (h3 : ct.price_A = 120000)
  (h4 : ct.price_B = 150000)
  (h5 : ct.total_budget = 3000000)
  (h6 : ct.total_units = 22) :
  ∃ (profit_A profit_B min_A : ℕ),
    profit_A = 3000 ∧
    profit_B = 5000 ∧
    min_A = 10 ∧
    2 * profit_A + 5 * profit_B = ct.profit_A_2_B_5 ∧
    profit_A + 2 * profit_B = ct.profit_A_1_B_2 ∧
    min_A * ct.price_A + (ct.total_units - min_A) * ct.price_B ≤ ct.total_budget :=
by sorry

end NUMINAMATH_CALUDE_car_trading_theorem_l3395_339511


namespace NUMINAMATH_CALUDE_cousin_distribution_count_l3395_339581

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribution_count (n k : ℕ) : ℕ :=
  sorry

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The number of rooms -/
def num_rooms : ℕ := 5

/-- Theorem stating that the number of ways to distribute 5 cousins into 5 rooms is 37 -/
theorem cousin_distribution_count :
  distribution_count num_cousins num_rooms = 37 := by sorry

end NUMINAMATH_CALUDE_cousin_distribution_count_l3395_339581


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3395_339591

theorem decimal_multiplication (a b : ℚ) (ha : a = 0.4) (hb : b = 0.75) : a * b = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l3395_339591


namespace NUMINAMATH_CALUDE_shaded_area_in_square_l3395_339585

/-- Given a square with side length a, the area bounded by a semicircle on one side
    and two quarter-circle arcs on the adjacent sides is equal to a²/2 -/
theorem shaded_area_in_square (a : ℝ) (h : a > 0) :
  (π * a^2 / 8) + (π * a^2 / 8) = a^2 / 2 := by
  sorry

#check shaded_area_in_square

end NUMINAMATH_CALUDE_shaded_area_in_square_l3395_339585


namespace NUMINAMATH_CALUDE_pen_count_l3395_339543

theorem pen_count (initial : ℕ) (received : ℕ) (given_away : ℕ) : 
  initial = 20 → received = 22 → given_away = 19 → 
  ((initial + received) * 2 - given_away) = 65 := by
sorry

end NUMINAMATH_CALUDE_pen_count_l3395_339543


namespace NUMINAMATH_CALUDE_magic_king_episodes_l3395_339503

/-- Calculates the total number of episodes for a TV show with the given parameters -/
def total_episodes (total_seasons : ℕ) (episodes_first_half : ℕ) (episodes_second_half : ℕ) : ℕ :=
  let half_seasons := total_seasons / 2
  half_seasons * episodes_first_half + half_seasons * episodes_second_half

/-- Theorem stating that a show with 10 seasons, 20 episodes per season in the first half,
    and 25 episodes per season in the second half, has a total of 225 episodes -/
theorem magic_king_episodes :
  total_episodes 10 20 25 = 225 := by
  sorry

#eval total_episodes 10 20 25

end NUMINAMATH_CALUDE_magic_king_episodes_l3395_339503


namespace NUMINAMATH_CALUDE_resistance_of_single_rod_l3395_339580

/-- The resistance of the entire construction between points A and B -/
def R : ℝ := 8

/-- The number of identical metallic rods in the network -/
def num_rods : ℕ := 13

/-- The resistance of one rod -/
def R₀ : ℝ := 20

/-- The relation between the total resistance and the resistance of one rod -/
axiom resistance_relation : R = (4/10) * R₀

theorem resistance_of_single_rod : R₀ = 20 :=
  sorry

end NUMINAMATH_CALUDE_resistance_of_single_rod_l3395_339580


namespace NUMINAMATH_CALUDE_contact_lenses_sold_l3395_339528

/-- Represents the number of pairs of hard contact lenses sold -/
def hard_lenses : ℕ := sorry

/-- Represents the number of pairs of soft contact lenses sold -/
def soft_lenses : ℕ := sorry

/-- The price of a pair of hard contact lenses in cents -/
def hard_price : ℕ := 8500

/-- The price of a pair of soft contact lenses in cents -/
def soft_price : ℕ := 15000

/-- The total sales in cents -/
def total_sales : ℕ := 145500

theorem contact_lenses_sold :
  (soft_lenses = hard_lenses + 5) →
  (hard_price * hard_lenses + soft_price * soft_lenses = total_sales) →
  (hard_lenses + soft_lenses = 11) := by
  sorry

end NUMINAMATH_CALUDE_contact_lenses_sold_l3395_339528


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l3395_339565

theorem smallest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  (a + b + c = 180) →
  (b = 6/5 * a) →
  (c = 7/5 * a) →
  a = 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_triangle_l3395_339565


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l3395_339574

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 2

theorem arithmetic_seq_sum 
  (a : ℕ → ℤ) 
  (h1 : arithmetic_seq a) 
  (h2 : a 1 + a 4 + a 7 = -50) : 
  a 3 + a 6 + a 9 = -38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l3395_339574


namespace NUMINAMATH_CALUDE_fencing_cost_is_2210_l3395_339572

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (width : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) : ℝ :=
  perimeter * cost_per_meter

/-- Theorem: The total cost of fencing the rectangular plot is 2210 -/
theorem fencing_cost_is_2210 :
  ∃ (width : ℝ),
    let length := width + 10
    let perimeter := 2 * (length + width)
    perimeter = 340 ∧
    total_fencing_cost width perimeter 6.5 = 2210 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_2210_l3395_339572


namespace NUMINAMATH_CALUDE_b_in_terms_of_a_l3395_339534

theorem b_in_terms_of_a (k : ℝ) (a b : ℝ) 
  (ha : a = 3 + 3^k) 
  (hb : b = 3 + 3^(-k)) : 
  b = (3*a - 8) / (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_b_in_terms_of_a_l3395_339534


namespace NUMINAMATH_CALUDE_gcd_of_specific_squares_l3395_339551

theorem gcd_of_specific_squares : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_squares_l3395_339551


namespace NUMINAMATH_CALUDE_farmers_field_planted_fraction_l3395_339516

theorem farmers_field_planted_fraction :
  ∀ (a b c : ℝ) (s : ℝ),
    a = 5 →
    b = 12 →
    c^2 = a^2 + b^2 →
    (s / a) = (4 / c) →
    (a * b / 2 - s^2) / (a * b / 2) = 470 / 507 :=
by sorry

end NUMINAMATH_CALUDE_farmers_field_planted_fraction_l3395_339516


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3395_339550

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_product : a 1 * a 7 = 3/4) :
  a 4 = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3395_339550


namespace NUMINAMATH_CALUDE_categorical_variables_correct_l3395_339548

-- Define the type for variables
inductive Variable
  | Smoking
  | Gender
  | ReligiousBelief
  | Nationality

-- Define a function to check if a variable is categorical
def isCategorical (v : Variable) : Prop :=
  match v with
  | Variable.Smoking => False
  | _ => True

-- Define the set of all variables
def allVariables : Set Variable :=
  {Variable.Smoking, Variable.Gender, Variable.ReligiousBelief, Variable.Nationality}

-- Define the set of categorical variables
def categoricalVariables : Set Variable :=
  {v ∈ allVariables | isCategorical v}

-- Theorem statement
theorem categorical_variables_correct :
  categoricalVariables = {Variable.Gender, Variable.ReligiousBelief, Variable.Nationality} :=
by sorry

end NUMINAMATH_CALUDE_categorical_variables_correct_l3395_339548


namespace NUMINAMATH_CALUDE_garden_area_increase_l3395_339517

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter := 2 * (rect_length + rect_width)
  let square_side := rect_perimeter / 4
  let rect_area := rect_length * rect_width
  let square_area := square_side * square_side
  square_area - rect_area = 400 := by sorry

end NUMINAMATH_CALUDE_garden_area_increase_l3395_339517


namespace NUMINAMATH_CALUDE_jasmine_concentration_proof_l3395_339546

/-- Proves that adding 5 liters of jasmine and 15 liters of water to an 80-liter solution
    with 10% jasmine results in a new solution with 13% jasmine concentration. -/
theorem jasmine_concentration_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_jasmine : ℝ) (added_water : ℝ) (final_concentration : ℝ) : 
  initial_volume = 80 →
  initial_concentration = 0.10 →
  added_jasmine = 5 →
  added_water = 15 →
  final_concentration = 0.13 →
  (initial_volume * initial_concentration + added_jasmine) / 
  (initial_volume + added_jasmine + added_water) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_jasmine_concentration_proof_l3395_339546


namespace NUMINAMATH_CALUDE_shooting_target_proof_l3395_339576

theorem shooting_target_proof (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (¬(p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_shooting_target_proof_l3395_339576


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3395_339598

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length :
  bridge_length 150 45 30 = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l3395_339598


namespace NUMINAMATH_CALUDE_arcsin_equation_solutions_l3395_339595

theorem arcsin_equation_solutions :
  let f (x : ℝ) := Real.arcsin (2 * x / Real.sqrt 15) + Real.arcsin (3 * x / Real.sqrt 15) = Real.arcsin (4 * x / Real.sqrt 15)
  let valid (x : ℝ) := abs (2 * x / Real.sqrt 15) ≤ 1 ∧ abs (3 * x / Real.sqrt 15) ≤ 1 ∧ abs (4 * x / Real.sqrt 15) ≤ 1
  ∀ x : ℝ, valid x → (f x ↔ x = 0 ∨ x = 15 / 16 ∨ x = -15 / 16) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solutions_l3395_339595


namespace NUMINAMATH_CALUDE_apples_bought_correct_l3395_339541

/-- Represents the number of apples Mary bought -/
def apples_bought : ℕ := 6

/-- Represents the number of apples Mary ate -/
def apples_eaten : ℕ := 2

/-- Represents the number of trees planted per apple eaten -/
def trees_per_apple : ℕ := 2

/-- Theorem stating that the number of apples Mary bought is correct -/
theorem apples_bought_correct : 
  apples_bought = apples_eaten + apples_eaten * trees_per_apple :=
by sorry

end NUMINAMATH_CALUDE_apples_bought_correct_l3395_339541


namespace NUMINAMATH_CALUDE_lisa_additional_marbles_l3395_339567

def minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_additional_marbles :
  minimum_additional_marbles 11 45 = 21 := by
  sorry

end NUMINAMATH_CALUDE_lisa_additional_marbles_l3395_339567


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3395_339526

theorem quadratic_polynomial_with_complex_root : 
  ∃ (a b c : ℝ), 
    (a = 3 ∧ 
     (Complex.I : ℂ)^2 = -1 ∧
     (3 : ℂ) * ((4 + 2 * Complex.I) ^ 2 - 8 * (4 + 2 * Complex.I) + 16 + 4) = 3 * (Complex.I : ℂ)^2 + b * (Complex.I : ℂ) + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l3395_339526


namespace NUMINAMATH_CALUDE_problem_statement_l3395_339544

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4)
  (h2 : x * y = 64) :
  (x + y) / 2 = (64^(1/(3+Real.sqrt 3)) + 64^((2+Real.sqrt 3)/(3+Real.sqrt 3))) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3395_339544


namespace NUMINAMATH_CALUDE_graduation_day_after_85_days_l3395_339592

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| monday : DayOfWeek
| tuesday : DayOfWeek
| wednesday : DayOfWeek
| thursday : DayOfWeek
| friday : DayOfWeek
| saturday : DayOfWeek
| sunday : DayOfWeek

/-- Function to add days to a given day of the week -/
def addDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (days % 7) with
  | 0 => start
  | 1 => match start with
    | DayOfWeek.monday => DayOfWeek.tuesday
    | DayOfWeek.tuesday => DayOfWeek.wednesday
    | DayOfWeek.wednesday => DayOfWeek.thursday
    | DayOfWeek.thursday => DayOfWeek.friday
    | DayOfWeek.friday => DayOfWeek.saturday
    | DayOfWeek.saturday => DayOfWeek.sunday
    | DayOfWeek.sunday => DayOfWeek.monday
  | _ => sorry -- Other cases omitted for brevity

theorem graduation_day_after_85_days : 
  addDays DayOfWeek.monday 85 = DayOfWeek.tuesday :=
by sorry


end NUMINAMATH_CALUDE_graduation_day_after_85_days_l3395_339592
