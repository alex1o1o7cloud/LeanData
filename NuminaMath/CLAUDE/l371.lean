import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l371_37131

theorem geometric_sequence_terms (x : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ (2*x + 2) = x * r ∧ (3*x + 3) = (2*x + 2) * r) → 
  x = 1 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l371_37131


namespace NUMINAMATH_CALUDE_pencil_pen_multiple_l371_37194

theorem pencil_pen_multiple (total : ℕ) (pens : ℕ) (M : ℕ) : 
  total = 108 →
  pens = 16 →
  total = pens + (M * pens + 12) →
  M = 5 := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_multiple_l371_37194


namespace NUMINAMATH_CALUDE_tangent_curve_sum_l371_37169

/-- The curve y = -2x^2 + bx + c is tangent to the line y = x - 3 at the point (2, -1).
    This theorem proves that b + c = -2. -/
theorem tangent_curve_sum (b c : ℝ) : 
  (∀ x, -2*x^2 + b*x + c = x - 3 → x = 2) →  -- Tangent condition
  -2*2^2 + b*2 + c = -1 →                    -- Point (2, -1) lies on the curve
  2 - 3 = -1 →                               -- Point (2, -1) lies on the line
  (-4*2 + b = 1) →                           -- Derivative equality at x = 2
  b + c = -2 := by
sorry

end NUMINAMATH_CALUDE_tangent_curve_sum_l371_37169


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l371_37123

theorem matrix_equation_solution :
  ∀ (a b c d : ℝ),
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, a; b, 1]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![c, 2; 0, d]
  M * N = !![2, 4; -2, 0] →
  a = 1 ∧ b = -1 ∧ c = 2 ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l371_37123


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l371_37178

/-- Given a tetrahedron with volume V, surface areas S₁, S₂, S₃, S₄, and an inscribed sphere of radius R,
    prove that R = 3V / (S₁ + S₂ + S₃ + S₄) -/
theorem inscribed_sphere_radius_tetrahedron (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (R : ℝ) 
    (h_positive : V > 0 ∧ S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ R > 0) :
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_tetrahedron_l371_37178


namespace NUMINAMATH_CALUDE_letter_puzzle_solutions_l371_37132

/-- A function that checks if a number is a single digit (1 to 9) -/
def isSingleDigit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- A function that checks if two numbers are distinct -/
def areDistinct (a b : ℕ) : Prop := a ≠ b

/-- A function that checks if a number is a two-digit number -/
def isTwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that constructs a two-digit number from two single digits -/
def twoDigitConstruct (b a : ℕ) : ℕ := 10 * b + a

/-- The main theorem stating the only solutions to A^B = BA -/
theorem letter_puzzle_solutions :
  ∀ A B : ℕ,
  isSingleDigit A →
  isSingleDigit B →
  areDistinct A B →
  isTwoDigitNumber (twoDigitConstruct B A) →
  twoDigitConstruct B A ≠ B * A →
  A^B = twoDigitConstruct B A →
  ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by sorry

end NUMINAMATH_CALUDE_letter_puzzle_solutions_l371_37132


namespace NUMINAMATH_CALUDE_shoes_cost_calculation_l371_37119

def shopping_problem (initial_amount sweater_cost tshirt_cost amount_left : ℕ) : Prop :=
  let total_spent := initial_amount - amount_left
  let other_items_cost := sweater_cost + tshirt_cost
  let shoes_cost := total_spent - other_items_cost
  shoes_cost = 11

theorem shoes_cost_calculation :
  shopping_problem 91 24 6 50 := by sorry

end NUMINAMATH_CALUDE_shoes_cost_calculation_l371_37119


namespace NUMINAMATH_CALUDE_computer_table_price_l371_37100

/-- The selling price of an item given its cost price and markup percentage -/
def sellingPrice (costPrice : ℚ) (markupPercentage : ℚ) : ℚ :=
  costPrice * (1 + markupPercentage / 100)

/-- Theorem: The selling price of a computer table with cost price 3840 and markup 25% is 4800 -/
theorem computer_table_price : sellingPrice 3840 25 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l371_37100


namespace NUMINAMATH_CALUDE_x_minus_y_value_l371_37140

theorem x_minus_y_value (x y : ℝ) (h : |x + y + 1| + Real.sqrt (2 * x - y) = 0) : 
  x - y = 1/3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l371_37140


namespace NUMINAMATH_CALUDE_sum_cubes_quartics_bounds_l371_37175

theorem sum_cubes_quartics_bounds (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) :
  let expr := 3 * (p^3 + q^3 + r^3 + s^3) - 5 * (p^4 + q^4 + r^4 + s^4)
  ∃ (min max : ℝ), min = 132 ∧ max = -20 ∧ min ≤ expr ∧ expr ≤ max := by
  sorry

end NUMINAMATH_CALUDE_sum_cubes_quartics_bounds_l371_37175


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l371_37139

theorem same_terminal_side (θ₁ θ₂ : Real) : 
  ∃ k : Int, θ₂ = θ₁ + 2 * π * k → 
  θ₁.cos = θ₂.cos ∧ θ₁.sin = θ₂.sin :=
by sorry

theorem angle_with_same_terminal_side : 
  ∃ k : Int, (11 * π / 8 : Real) = (-5 * π / 8 : Real) + 2 * π * k :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l371_37139


namespace NUMINAMATH_CALUDE_base4_division_l371_37111

-- Define a function to convert from base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (4^i)) 0

-- Define the numbers in base 4
def num2013Base4 : List Nat := [3, 1, 0, 2]
def num13Base4 : List Nat := [3, 1]
def result13Base4 : List Nat := [3, 1]

-- State the theorem
theorem base4_division :
  (base4ToDecimal num2013Base4) / (base4ToDecimal num13Base4) = base4ToDecimal result13Base4 :=
sorry

end NUMINAMATH_CALUDE_base4_division_l371_37111


namespace NUMINAMATH_CALUDE_f_decreasing_range_f_less_than_g_range_l371_37149

open Real

noncomputable def f (a x : ℝ) : ℝ := log x - a^2 * x^2 + a * x

noncomputable def g (a x : ℝ) : ℝ := (3*a + 1) * x - (a^2 + a) * x^2

theorem f_decreasing_range (a : ℝ) (h : a ≠ 0) :
  (∀ x ≥ 1, ∀ y ≥ x, f a x ≥ f a y) ↔ a ≥ 1 :=
sorry

theorem f_less_than_g_range (a : ℝ) (h : a ≠ 0) :
  (∀ x > 1, f a x < g a x) ↔ -1 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_range_f_less_than_g_range_l371_37149


namespace NUMINAMATH_CALUDE_triangle_town_intersections_l371_37126

/-- The number of intersections for n non-parallel lines in a plane where no three lines meet at a single point -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a configuration of 10 non-parallel lines in a plane, 
    where no three lines intersect at a single point, 
    the maximum number of intersection points is 45 -/
theorem triangle_town_intersections :
  max_intersections 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_town_intersections_l371_37126


namespace NUMINAMATH_CALUDE_maria_quiz_goal_l371_37156

theorem maria_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (quizzes_taken : ℕ) (as_earned : ℕ) (remaining_lower_a : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 70 / 100 →
  quizzes_taken = 35 →
  as_earned = 28 →
  remaining_lower_a = 11 →
  (as_earned + (total_quizzes - quizzes_taken - remaining_lower_a) : ℚ) / total_quizzes ≥ goal_percentage := by
  sorry

#check maria_quiz_goal

end NUMINAMATH_CALUDE_maria_quiz_goal_l371_37156


namespace NUMINAMATH_CALUDE_part_one_part_two_l371_37185

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a / Real.cos t.A = (3 * t.c - 2 * t.b) / Real.cos t.B

-- Theorem for part (1)
theorem part_one (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.b = Real.sqrt 5 * Real.sin t.B) : 
  t.a = 5/3 := by
sorry

-- Theorem for part (2)
theorem part_two (t : Triangle) 
  (h1 : triangle_condition t)
  (h2 : t.a = Real.sqrt 6)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 5 / 2) :
  t.b + t.c = 4 := by
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l371_37185


namespace NUMINAMATH_CALUDE_andrea_skating_schedule_l371_37174

/-- Represents Andrea's skating schedule and target average -/
structure SkatingSchedule where
  days_schedule1 : ℕ  -- Number of days for schedule 1
  minutes_per_day1 : ℕ  -- Minutes skated per day in schedule 1
  days_schedule2 : ℕ  -- Number of days for schedule 2
  minutes_per_day2 : ℕ  -- Minutes skated per day in schedule 2
  total_days : ℕ  -- Total number of days
  target_average : ℕ  -- Target average minutes per day

/-- Calculates the required skating time for the last day to achieve the target average -/
def required_last_day_minutes (schedule : SkatingSchedule) : ℕ :=
  schedule.target_average * schedule.total_days -
  (schedule.days_schedule1 * schedule.minutes_per_day1 +
   schedule.days_schedule2 * schedule.minutes_per_day2)

/-- Theorem stating that given Andrea's skating schedule, 
    she needs to skate 175 minutes on the ninth day to achieve the target average -/
theorem andrea_skating_schedule :
  let schedule : SkatingSchedule := {
    days_schedule1 := 6,
    minutes_per_day1 := 80,
    days_schedule2 := 2,
    minutes_per_day2 := 100,
    total_days := 9,
    target_average := 95
  }
  required_last_day_minutes schedule = 175 := by
  sorry

end NUMINAMATH_CALUDE_andrea_skating_schedule_l371_37174


namespace NUMINAMATH_CALUDE_chichikov_guarantee_l371_37144

/-- Represents a distribution of nuts into three boxes -/
def Distribution := (ℕ × ℕ × ℕ)

/-- Checks if a distribution is valid (sum is 1001) -/
def valid_distribution (d : Distribution) : Prop :=
  d.1 + d.2.1 + d.2.2 = 1001

/-- Represents the number of nuts that need to be moved for a given N -/
def nuts_to_move (d : Distribution) (N : ℕ) : ℕ :=
  sorry

/-- The maximum number of nuts that need to be moved for any N -/
def max_nuts_to_move (d : Distribution) : ℕ :=
  sorry

theorem chichikov_guarantee :
  ∀ d : Distribution, valid_distribution d →
  ∃ N : ℕ, 1 ≤ N ∧ N ≤ 1001 ∧ nuts_to_move d N ≥ 71 ∧
  ∀ M : ℕ, M > 71 → ∃ d' : Distribution, valid_distribution d' ∧
  ∀ N' : ℕ, 1 ≤ N' ∧ N' ≤ 1001 → nuts_to_move d' N' < M :=
sorry

end NUMINAMATH_CALUDE_chichikov_guarantee_l371_37144


namespace NUMINAMATH_CALUDE_abcd_imag_zero_l371_37125

open Complex

-- Define the condition for angles being equal and oppositely oriented
def anglesEqualOpposite (a b c d : ℂ) : Prop :=
  ∃ θ : ℝ, b / a = exp (θ * I) ∧ d / c = exp (-θ * I)

theorem abcd_imag_zero (a b c d : ℂ) 
  (h : anglesEqualOpposite a b c d) : 
  (a * b * c * d).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_abcd_imag_zero_l371_37125


namespace NUMINAMATH_CALUDE_rectangle_z_value_l371_37152

/-- A rectangle with given vertices and area -/
structure Rectangle where
  z : ℝ
  area : ℝ
  h_vertices : z > 5
  h_area : area = 64

/-- The value of z for the given rectangle is 13 -/
theorem rectangle_z_value (rect : Rectangle) : rect.z = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_z_value_l371_37152


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l371_37116

/-- The function f(x) = x^2 - 2mx + 3 is monotonic on the interval [1, 3] if and only if m ≤ 1 or m ≥ 3 -/
theorem monotonic_quadratic_function (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, Monotone (fun x => x^2 - 2*m*x + 3)) ↔ (m ≤ 1 ∨ m ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l371_37116


namespace NUMINAMATH_CALUDE_pool_filling_time_pool_filling_time_is_50_hours_l371_37186

/-- The time required to fill a swimming pool given the hose flow rate, water cost, and total cost to fill the pool. -/
theorem pool_filling_time 
  (hose_flow_rate : ℝ) 
  (water_cost_per_ten_gallons : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let cost_per_gallon := water_cost_per_ten_gallons / 10
  let total_gallons := total_cost / cost_per_gallon
  total_gallons / hose_flow_rate

/-- The time to fill the pool is 50 hours. -/
theorem pool_filling_time_is_50_hours 
  (hose_flow_rate : ℝ) 
  (water_cost_per_ten_gallons : ℝ) 
  (total_cost : ℝ) 
  (h1 : hose_flow_rate = 100)
  (h2 : water_cost_per_ten_gallons = 1)
  (h3 : total_cost = 5) : 
  pool_filling_time hose_flow_rate water_cost_per_ten_gallons total_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_pool_filling_time_is_50_hours_l371_37186


namespace NUMINAMATH_CALUDE_teacher_selection_theorem_l371_37129

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 6 teachers out of 10, where two specific teachers cannot be selected together -/
def selectTeachers (totalTeachers invitedTeachers : ℕ) : ℕ :=
  binomial totalTeachers invitedTeachers - binomial (totalTeachers - 2) (invitedTeachers - 2)

theorem teacher_selection_theorem :
  selectTeachers 10 6 = 140 := by sorry

end NUMINAMATH_CALUDE_teacher_selection_theorem_l371_37129


namespace NUMINAMATH_CALUDE_continuous_function_image_interval_l371_37188

open Set

theorem continuous_function_image_interval 
  (f : ℝ → ℝ) (hf : Continuous f) (a b : ℝ) (hab : a < b)
  (ha : a ∈ Set.range f) (hb : b ∈ Set.range f) :
  ∃ (I : Set ℝ), ∃ (s t : ℝ), I = Icc s t ∧ f '' I = Icc a b := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_image_interval_l371_37188


namespace NUMINAMATH_CALUDE_hyperbola_min_eccentricity_l371_37176

/-- Given an ellipse and a hyperbola with coinciding foci, and a line intersecting
    the right branch of the hyperbola, when the eccentricity of the hyperbola is minimized,
    the equation of the hyperbola is x^2/5 - y^2/4 = 1 -/
theorem hyperbola_min_eccentricity 
  (ellipse : ℝ → ℝ → Prop)
  (hyperbola : ℝ → ℝ → ℝ → ℝ → Prop)
  (line : ℝ → ℝ → Prop)
  (h_ellipse : ∀ x y, ellipse x y ↔ x^2/16 + y^2/7 = 1)
  (h_hyperbola : ∀ a b x y, a > b ∧ b > 0 → (hyperbola a b x y ↔ x^2/a^2 - y^2/b^2 = 1))
  (h_foci : ∀ a b, hyperbola a b (-3) 0 ∧ hyperbola a b 3 0)
  (h_line : ∀ x y, line x y ↔ x - y = 1)
  (h_intersect : ∃ x y, hyperbola a b x y ∧ line x y ∧ x > 0)
  (h_min_eccentricity : ∀ a' b', (∃ x y, hyperbola a' b' x y ∧ line x y) → 
    (a^2 - b^2)/(a^2) ≤ (a'^2 - b'^2)/(a'^2)) :
  hyperbola 5 4 x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_min_eccentricity_l371_37176


namespace NUMINAMATH_CALUDE_max_consecutive_semiprimes_l371_37198

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semiprimes :
  (∃ start : ℕ, ∀ i : ℕ, i < 5 → isSemiPrime (start + i)) ∧
  (¬∃ start : ℕ, ∀ i : ℕ, i < 6 → isSemiPrime (start + i)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semiprimes_l371_37198


namespace NUMINAMATH_CALUDE_unique_solution_pqr_l371_37171

theorem unique_solution_pqr : 
  ∀ p q r : ℕ,
  Prime p → Prime q → Even r → r > 0 →
  p^3 + q^2 = 4*r^2 + 45*r + 103 →
  p = 7 ∧ q = 2 ∧ r = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_pqr_l371_37171


namespace NUMINAMATH_CALUDE_infinite_occurrence_in_sequence_l371_37130

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Computes the sum of digits of a natural number in decimal system -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

/-- Computes a_n for a given polynomial and natural number n -/
def computeA (P : IntPolynomial) (n : Nat) : Nat :=
  sumOfDigits (sorry)  -- Evaluate P(n) and compute sum of digits

/-- The main theorem -/
theorem infinite_occurrence_in_sequence (P : IntPolynomial) :
  ∃ (k : Nat), Set.Infinite {n : Nat | computeA P n = k} :=
sorry

end NUMINAMATH_CALUDE_infinite_occurrence_in_sequence_l371_37130


namespace NUMINAMATH_CALUDE_sum_squares_and_inverses_bound_l371_37102

theorem sum_squares_and_inverses_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + 1/a^2 + b^2 + 1/b^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_and_inverses_bound_l371_37102


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l371_37199

theorem fractional_equation_solution_range (x m : ℝ) : 
  ((2 * x - m) / (x + 1) = 3) → 
  (x < 0) → 
  (m > -3 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l371_37199


namespace NUMINAMATH_CALUDE_cubic_factorization_l371_37135

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l371_37135


namespace NUMINAMATH_CALUDE_impossible_average_l371_37151

theorem impossible_average (test1 test2 test3 test4 test5 test6 : ℕ) 
  (h1 : test1 = 85)
  (h2 : test2 = 79)
  (h3 : test3 = 92)
  (h4 : test4 = 84)
  (h5 : test5 = 88)
  (h6 : test6 = 7)
  : ¬ ∃ (test7 test8 : ℕ), (test1 + test2 + test3 + test4 + test5 + test6 + test7 + test8) / 8 = 87 :=
sorry

end NUMINAMATH_CALUDE_impossible_average_l371_37151


namespace NUMINAMATH_CALUDE_jeff_new_cabinet_counters_l371_37189

/-- Calculates the number of counters over which new cabinets were installed --/
def counters_with_new_cabinets (initial_cabinets : ℕ) (cabinets_per_new_counter : ℕ) (additional_cabinets : ℕ) (total_cabinets : ℕ) : ℕ :=
  (total_cabinets - initial_cabinets - additional_cabinets) / cabinets_per_new_counter

/-- Proves that Jeff installed new cabinets over 9 counters --/
theorem jeff_new_cabinet_counters :
  let initial_cabinets := 3
  let cabinets_per_new_counter := 2
  let additional_cabinets := 5
  let total_cabinets := 26
  counters_with_new_cabinets initial_cabinets cabinets_per_new_counter additional_cabinets total_cabinets = 9 := by
  sorry

end NUMINAMATH_CALUDE_jeff_new_cabinet_counters_l371_37189


namespace NUMINAMATH_CALUDE_quadratic_intersection_l371_37184

/-- A quadratic function f(x) = x^2 - 6x + c intersects the x-axis at only one point
    if and only if its discriminant is zero. -/
def intersects_once (c : ℝ) : Prop :=
  ((-6)^2 - 4*1*c) = 0

/-- The theorem states that if a quadratic function f(x) = x^2 - 6x + c
    intersects the x-axis at only one point, then c = 9. -/
theorem quadratic_intersection (c : ℝ) :
  intersects_once c → c = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l371_37184


namespace NUMINAMATH_CALUDE_books_given_difference_l371_37147

theorem books_given_difference (mike_books_tuesday : ℕ) (mike_gave : ℕ) (lily_total : ℕ)
  (h1 : mike_books_tuesday = 45)
  (h2 : mike_gave = 10)
  (h3 : lily_total = 35) :
  lily_total - mike_gave - (mike_gave) = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_given_difference_l371_37147


namespace NUMINAMATH_CALUDE_equation_system_solution_l371_37115

theorem equation_system_solution (a b : ℝ) : 
  (∃ (a' : ℝ), a' * (-1) + 5 * (-1) = 15 ∧ 4 * (-1) - b * (-1) = -2) →
  (∃ (b' : ℝ), a * 5 + 5 * 2 = 15 ∧ 4 * 5 - b' * 2 = -2) →
  (a + 4 * b)^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l371_37115


namespace NUMINAMATH_CALUDE_compute_d_l371_37168

-- Define the polynomial
def f (c d : ℚ) (x : ℝ) : ℝ := x^3 + c*x^2 + d*x - 36

-- State the theorem
theorem compute_d (c : ℚ) :
  ∃ d : ℚ, f c d (3 + Real.sqrt 2) = 0 → d = -23 - 6/7 :=
by sorry

end NUMINAMATH_CALUDE_compute_d_l371_37168


namespace NUMINAMATH_CALUDE_tshirt_cost_l371_37167

theorem tshirt_cost (original_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  original_price = 240 ∧ 
  discount_rate = 0.2 ∧ 
  profit_rate = 0.2 →
  ∃ (cost : ℝ), cost = 160 ∧ 
    cost * (1 + profit_rate) = original_price * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_tshirt_cost_l371_37167


namespace NUMINAMATH_CALUDE_kelly_apples_l371_37142

theorem kelly_apples (initial : ℕ) (second_day : ℕ) (third_day : ℕ) (eaten : ℕ) : 
  initial = 56 → second_day = 105 → third_day = 84 → eaten = 23 →
  initial + second_day + third_day - eaten = 222 :=
by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l371_37142


namespace NUMINAMATH_CALUDE_expression_evaluation_l371_37105

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l371_37105


namespace NUMINAMATH_CALUDE_system_solution_l371_37153

theorem system_solution (x y : ℝ) (h1 : 3 * x + y = 21) (h2 : x + 3 * y = 1) : 2 * x + 2 * y = 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l371_37153


namespace NUMINAMATH_CALUDE_ledi_age_in_future_l371_37193

/-- The number of years ago when the sum of Duoduo and Ledi's ages was 12 years -/
def years_ago : ℝ := 12.3

/-- Duoduo's current age -/
def duoduo_current_age : ℝ := 10

/-- The sum of Duoduo and Ledi's ages 12.3 years ago -/
def sum_ages_past : ℝ := 12

/-- The number of years until Ledi will be 10 years old -/
def years_until_ledi_ten : ℝ := 6.3

theorem ledi_age_in_future :
  ∃ (ledi_current_age : ℝ),
    ledi_current_age + duoduo_current_age = sum_ages_past + 2 * years_ago ∧
    ledi_current_age + years_until_ledi_ten = 10 :=
by sorry

end NUMINAMATH_CALUDE_ledi_age_in_future_l371_37193


namespace NUMINAMATH_CALUDE_average_weight_increase_l371_37158

def number_of_oarsmen : ℕ := 10
def old_weight : ℝ := 53
def new_weight : ℝ := 71

theorem average_weight_increase :
  (new_weight - old_weight) / number_of_oarsmen = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l371_37158


namespace NUMINAMATH_CALUDE_gcd_1729_1323_l371_37138

theorem gcd_1729_1323 : Nat.gcd 1729 1323 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1323_l371_37138


namespace NUMINAMATH_CALUDE_product_102_105_l371_37181

theorem product_102_105 : 102 * 105 = 10710 := by
  sorry

end NUMINAMATH_CALUDE_product_102_105_l371_37181


namespace NUMINAMATH_CALUDE_blocks_per_box_l371_37172

theorem blocks_per_box (total_blocks : ℕ) (num_boxes : ℕ) (h1 : total_blocks = 16) (h2 : num_boxes = 8) :
  total_blocks / num_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_box_l371_37172


namespace NUMINAMATH_CALUDE_nobel_laureates_count_l371_37146

/-- Represents the number of scientists at a workshop with various prize combinations -/
structure WorkshopScientists where
  total : Nat
  wolf : Nat
  wolfAndNobel : Nat
  nonWolfNobel : Nat
  nonWolfNonNobel : Nat

/-- The conditions of the workshop -/
def workshop : WorkshopScientists where
  total := 50
  wolf := 31
  wolfAndNobel := 12
  nonWolfNobel := (50 - 31 + 3) / 2
  nonWolfNonNobel := (50 - 31 - 3) / 2

/-- Theorem stating the total number of Nobel prize laureates -/
theorem nobel_laureates_count (w : WorkshopScientists) (h1 : w = workshop) :
  w.wolfAndNobel + w.nonWolfNobel = 23 := by
  sorry

#check nobel_laureates_count

end NUMINAMATH_CALUDE_nobel_laureates_count_l371_37146


namespace NUMINAMATH_CALUDE_old_record_calculation_old_record_proof_l371_37191

/-- Calculates the old record given James' performance and the points he beat the record by -/
theorem old_record_calculation (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (points_beat_record_by : ℕ) : ℕ :=
  let james_total_points := touchdowns_per_game * points_per_touchdown * games_in_season + 
                            two_point_conversions * 2
  james_total_points - points_beat_record_by

/-- Proves that the old record was 300 points given James' performance -/
theorem old_record_proof :
  old_record_calculation 4 6 15 6 72 = 300 := by
  sorry

end NUMINAMATH_CALUDE_old_record_calculation_old_record_proof_l371_37191


namespace NUMINAMATH_CALUDE_total_dolls_l371_37136

def sister_dolls : ℕ := 8
def hannah_multiplier : ℕ := 5

theorem total_dolls : sister_dolls + hannah_multiplier * sister_dolls = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l371_37136


namespace NUMINAMATH_CALUDE_final_class_size_l371_37128

def fourth_grade_class_size (initial_students : ℕ) 
  (first_semester_left : ℕ) (first_semester_joined : ℕ)
  (second_semester_joined : ℕ) (second_semester_transferred : ℕ) (second_semester_switched : ℕ) : ℕ :=
  initial_students - first_semester_left + first_semester_joined + 
  second_semester_joined - second_semester_transferred - second_semester_switched

theorem final_class_size : 
  fourth_grade_class_size 11 6 25 15 3 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_final_class_size_l371_37128


namespace NUMINAMATH_CALUDE_max_perimeter_of_rectangle_from_triangles_l371_37154

theorem max_perimeter_of_rectangle_from_triangles :
  ∀ (L W : ℝ),
  L > 0 → W > 0 →
  L * W = 60 * (1/2 * 2 * 3) →
  2 * (L + W) ≤ 184 :=
by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_rectangle_from_triangles_l371_37154


namespace NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l371_37177

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : abs a + abs b < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l371_37177


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l371_37110

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 + 8 * i) / (5 - 4 * i) = (-2 : ℚ) / 41 + (64 : ℚ) / 41 * i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l371_37110


namespace NUMINAMATH_CALUDE_sea_glass_problem_l371_37159

theorem sea_glass_problem (blanche_green : ℕ) (rose_red rose_blue : ℕ) (dorothy_total : ℕ)
  (h1 : blanche_green = 12)
  (h2 : rose_red = 9)
  (h3 : rose_blue = 11)
  (h4 : dorothy_total = 57) :
  ∃ (blanche_red : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧
    blanche_red = 3 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_problem_l371_37159


namespace NUMINAMATH_CALUDE_largest_fraction_l371_37162

theorem largest_fraction :
  let a := 35 / 69
  let b := 7 / 15
  let c := 9 / 19
  let d := 399 / 799
  let e := 150 / 299
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l371_37162


namespace NUMINAMATH_CALUDE_cos_pi_twelve_squared_identity_l371_37106

theorem cos_pi_twelve_squared_identity : 2 * (Real.cos (π / 12))^2 - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_twelve_squared_identity_l371_37106


namespace NUMINAMATH_CALUDE_A_inter_B_eq_A_l371_37166

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

-- Theorem statement
theorem A_inter_B_eq_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_A_l371_37166


namespace NUMINAMATH_CALUDE_standard_form_of_negative_r_l371_37161

/-- Converts a polar coordinate point to its standard form where r > 0 and 0 ≤ θ < 2π -/
def standardPolarForm (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  sorry

theorem standard_form_of_negative_r :
  let original : ℝ × ℝ := (-3, π/6)
  let standard : ℝ × ℝ := standardPolarForm original.1 original.2
  standard = (3, 7*π/6) ∧ standard.1 > 0 ∧ 0 ≤ standard.2 ∧ standard.2 < 2*π :=
by sorry

end NUMINAMATH_CALUDE_standard_form_of_negative_r_l371_37161


namespace NUMINAMATH_CALUDE_amelias_apples_l371_37145

theorem amelias_apples (george_oranges : ℕ) (george_apples_diff : ℕ) (amelia_oranges_diff : ℕ) (total_fruits : ℕ) :
  george_oranges = 45 →
  george_apples_diff = 5 →
  amelia_oranges_diff = 18 →
  total_fruits = 107 →
  ∃ (amelia_apples : ℕ),
    total_fruits = george_oranges + (george_oranges - amelia_oranges_diff) + (amelia_apples + george_apples_diff) + amelia_apples ∧
    amelia_apples = 15 :=
by sorry

end NUMINAMATH_CALUDE_amelias_apples_l371_37145


namespace NUMINAMATH_CALUDE_total_blood_cells_l371_37180

/-- The total number of blood cells in two samples is 7341, given the number of cells in each sample. -/
theorem total_blood_cells (sample1 : Nat) (sample2 : Nat)
  (h1 : sample1 = 4221) (h2 : sample2 = 3120) :
  sample1 + sample2 = 7341 := by
  sorry

end NUMINAMATH_CALUDE_total_blood_cells_l371_37180


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l371_37163

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 12 = 43200 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l371_37163


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l371_37134

theorem least_three_digit_multiple_of_eight : ∀ n : ℕ, 
  n ≥ 100 ∧ n < 1000 ∧ n % 8 = 0 → n ≥ 104 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l371_37134


namespace NUMINAMATH_CALUDE_jose_investment_is_4500_l371_37122

/-- Represents the investment and profit scenario of Tom and Jose's shop -/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit_share : ℕ

/-- Calculates Jose's investment based on the given shop investment scenario -/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  -- The actual calculation will be implemented in the proof
  sorry

/-- Theorem stating that Jose's investment is 4500 given the specific scenario -/
theorem jose_investment_is_4500 :
  let s : ShopInvestment := {
    tom_investment := 3000,
    jose_join_delay := 2,
    total_profit := 6300,
    jose_profit_share := 3500
  }
  calculate_jose_investment s = 4500 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_4500_l371_37122


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_25_l371_37157

theorem three_digit_perfect_cube_divisible_by_25 : 
  ∃! (n : ℕ), 100 ≤ 125 * n^3 ∧ 125 * n^3 ≤ 999 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_25_l371_37157


namespace NUMINAMATH_CALUDE_marks_reading_time_l371_37183

/-- Calculates Mark's new weekly reading time given his daily reading time,
    the number of days in a week, and his planned increase in weekly reading time. -/
def new_weekly_reading_time (daily_reading_time : ℕ) (days_in_week : ℕ) (weekly_increase : ℕ) : ℕ :=
  daily_reading_time * days_in_week + weekly_increase

/-- Proves that Mark's new weekly reading time is 18 hours -/
theorem marks_reading_time :
  new_weekly_reading_time 2 7 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_marks_reading_time_l371_37183


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l371_37104

theorem algebra_test_female_students 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 83)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l371_37104


namespace NUMINAMATH_CALUDE_function_intersection_l371_37141

/-- Two functions f and g have exactly one common point if and only if (a-c)(b-d) = 2,
    given that they are centrally symmetric with respect to the point ((b+d)/2, a+c) -/
theorem function_intersection (a b c d : ℝ) :
  let f (x : ℝ) := 2*a + 1/(x-b)
  let g (x : ℝ) := 2*c + 1/(x-d)
  let center : ℝ × ℝ := ((b+d)/2, a+c)
  (∃! p, f p = g p ∧ 
   ∀ x y, f x = y ↔ g (b+d-x) = 2*(a+c)-y) ↔ 
  (a-c)*(b-d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_intersection_l371_37141


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l371_37103

-- Problem 1
theorem problem_one : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_two : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1)^2 = 14 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l371_37103


namespace NUMINAMATH_CALUDE_work_days_and_wages_l371_37179

/-- Represents the number of days worked and wages for two workers --/
structure WorkData where
  days_A : ℕ
  days_B : ℕ
  wage_A : ℚ
  wage_B : ℚ

/-- Verifies if the given work data satisfies the problem conditions --/
def satisfies_conditions (data : WorkData) : Prop :=
  data.days_B = data.days_A - 3 ∧
  data.wage_A * data.days_A = 30 ∧
  data.wage_B * data.days_B = 14 ∧
  data.wage_A * (data.days_A - 2) = data.wage_B * (data.days_B + 5)

/-- The theorem to be proved --/
theorem work_days_and_wages : 
  ∃ (data : WorkData), satisfies_conditions data ∧ 
    data.days_A = 10 ∧ data.days_B = 7 ∧ 
    data.wage_A = 3 ∧ data.wage_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_days_and_wages_l371_37179


namespace NUMINAMATH_CALUDE_crude_oil_mixture_theorem_l371_37108

/-- Represents the percentage of hydrocarbons in crude oil from the first source -/
def first_source_percentage : ℝ := 25

/-- Represents the total amount of crude oil needed in gallons -/
def total_crude_oil : ℝ := 50

/-- Represents the desired percentage of hydrocarbons in the final mixture -/
def final_mixture_percentage : ℝ := 55

/-- Represents the amount of crude oil from the second source in gallons -/
def second_source_amount : ℝ := 30

/-- Represents the percentage of hydrocarbons in crude oil from the second source -/
def second_source_percentage : ℝ := 75

/-- Theorem stating that given the conditions, the percentage of hydrocarbons
    in the first source is 25% -/
theorem crude_oil_mixture_theorem :
  (first_source_percentage / 100 * (total_crude_oil - second_source_amount) +
   second_source_percentage / 100 * second_source_amount) / total_crude_oil * 100 =
  final_mixture_percentage := by
  sorry

end NUMINAMATH_CALUDE_crude_oil_mixture_theorem_l371_37108


namespace NUMINAMATH_CALUDE_oranges_for_three_rubles_l371_37182

/-- Given that 25 oranges cost as many rubles as can be bought for 1 ruble,
    prove that 15 oranges can be bought for 3 rubles -/
theorem oranges_for_three_rubles : ∀ x : ℝ,
  (25 : ℝ) / x = x →  -- 25 oranges cost x rubles, and x oranges can be bought for 1 ruble
  (3 : ℝ) * x = 15 :=  -- 15 oranges can be bought for 3 rubles
by
  sorry

end NUMINAMATH_CALUDE_oranges_for_three_rubles_l371_37182


namespace NUMINAMATH_CALUDE_computation_proof_l371_37127

theorem computation_proof : 24 * ((150 / 3) - (36 / 6) + (7.2 / 0.4) + 2) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l371_37127


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l371_37101

/-- Given two vectors a and b in ℝ², where b = (-1, 2) and a + b = (1, 3),
    prove that the magnitude of a - 2b is 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  b = (-1, 2) → a + b = (1, 3) → ‖a - 2 • b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l371_37101


namespace NUMINAMATH_CALUDE_derivative_sin_2x_minus_1_l371_37164

theorem derivative_sin_2x_minus_1 (x : ℝ) :
  deriv (λ x => Real.sin (2 * x - 1)) x = 2 * Real.cos (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_2x_minus_1_l371_37164


namespace NUMINAMATH_CALUDE_stratified_sampling_young_representatives_l371_37187

/-- Represents the number of young representatives to be selected in a stratified sampling scenario. -/
def young_representatives (total_population : ℕ) (young_population : ℕ) (total_representatives : ℕ) : ℕ :=
  (young_population * total_representatives) / total_population

/-- Theorem stating that for the given population numbers and sampling size, 
    the number of young representatives to be selected is 7. -/
theorem stratified_sampling_young_representatives :
  young_representatives 1000 350 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_young_representatives_l371_37187


namespace NUMINAMATH_CALUDE_doctor_nurse_ratio_l371_37107

theorem doctor_nurse_ratio (total : ℕ) (nurses : ℕ) (h1 : total = 200) (h2 : nurses = 120) :
  (total - nurses) / nurses = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_doctor_nurse_ratio_l371_37107


namespace NUMINAMATH_CALUDE_graph_shift_up_by_two_l371_37117

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the transformed function g
def g (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem graph_shift_up_by_two :
  ∀ x : ℝ, g x = f x + 2 := by sorry

end NUMINAMATH_CALUDE_graph_shift_up_by_two_l371_37117


namespace NUMINAMATH_CALUDE_cos_135_degrees_l371_37137

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l371_37137


namespace NUMINAMATH_CALUDE_village_population_l371_37148

theorem village_population (P : ℝ) : 0.85 * (0.95 * P) = 3294 → P = 4080 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l371_37148


namespace NUMINAMATH_CALUDE_nth_root_power_comparison_l371_37197

theorem nth_root_power_comparison (a : ℝ) (n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  (a > 1 → a^(m/n) > 1) ∧ (a < 1 → a^(m/n) < 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_power_comparison_l371_37197


namespace NUMINAMATH_CALUDE_cube_surface_area_l371_37192

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p q : Point3D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2

/-- The vertices of the cube -/
def A : Point3D := ⟨5, 7, 15⟩
def B : Point3D := ⟨6, 3, 6⟩
def C : Point3D := ⟨9, -2, 14⟩

/-- Theorem: The surface area of the cube with vertices A, B, and C is 294 -/
theorem cube_surface_area : ∃ (side : ℝ), 
  squaredDistance A B = 2 * side^2 ∧ 
  squaredDistance A C = 2 * side^2 ∧ 
  squaredDistance B C = 2 * side^2 ∧ 
  6 * side^2 = 294 := by
  sorry


end NUMINAMATH_CALUDE_cube_surface_area_l371_37192


namespace NUMINAMATH_CALUDE_officers_on_duty_l371_37170

theorem officers_on_duty (total_female_officers : ℕ) 
  (female_on_duty_percentage : ℚ) (female_ratio_on_duty : ℚ) :
  total_female_officers = 300 →
  female_on_duty_percentage = 2/5 →
  female_ratio_on_duty = 1/2 →
  (female_on_duty_percentage * total_female_officers : ℚ) / female_ratio_on_duty = 240 := by
  sorry

end NUMINAMATH_CALUDE_officers_on_duty_l371_37170


namespace NUMINAMATH_CALUDE_ghee_mixture_volume_l371_37155

/-- Prove that the volume of a mixture of two brands of vegetable ghee is 4 liters -/
theorem ghee_mixture_volume :
  ∀ (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ),
    weight_a = 900 →
    weight_b = 700 →
    ratio_a = 3 →
    ratio_b = 2 →
    total_weight = 3280 →
    ∃ (volume_a volume_b : ℝ),
      volume_a / volume_b = ratio_a / ratio_b ∧
      weight_a * volume_a + weight_b * volume_b = total_weight ∧
      volume_a + volume_b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ghee_mixture_volume_l371_37155


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l371_37160

theorem polynomial_identity_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l371_37160


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l371_37124

/-- Two parabolas with mutually perpendicular axes -/
structure PerpendicularParabolas where
  -- First parabola: x = ay² + b
  a : ℝ
  b : ℝ
  -- Second parabola: y = cx² + d
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  c_pos : 0 < c

/-- The four intersection points of two perpendicular parabolas -/
def intersectionPoints (p : PerpendicularParabolas) : Set (ℝ × ℝ) :=
  {point | point.1 = p.a * point.2^2 + p.b ∧ point.2 = p.c * point.1^2 + p.d}

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The theorem stating that the intersection points lie on a circle -/
theorem intersection_points_on_circle (p : PerpendicularParabolas) :
  ∃ (circle : Circle), ∀ point ∈ intersectionPoints p,
    (point.1 - circle.center.1)^2 + (point.2 - circle.center.2)^2 = circle.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l371_37124


namespace NUMINAMATH_CALUDE_inequality_solution_set_l371_37143

theorem inequality_solution_set (x : ℝ) : 
  1 - 7 / (2 * x - 1) < 0 ↔ 1/2 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l371_37143


namespace NUMINAMATH_CALUDE_fraction_order_l371_37190

theorem fraction_order (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hab : a > b) : 
  (b / a < (b + c) / (a + c)) ∧ 
  ((b + c) / (a + c) < (a + d) / (b + d)) ∧ 
  ((a + d) / (b + d) < a / b) := by
sorry

end NUMINAMATH_CALUDE_fraction_order_l371_37190


namespace NUMINAMATH_CALUDE_problem_l371_37165

def p : Prop := ∀ x : ℝ, (x > 3 ↔ x^2 > 9)
def q : Prop := ∀ a b : ℝ, (a^2 > b^2 ↔ a > b)

theorem problem : ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_l371_37165


namespace NUMINAMATH_CALUDE_quadratic_prime_square_solution_l371_37173

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

/-- The theorem states that the only integer solution to the equation 2x^2 - x - 36 = p^2,
    where p is a prime number, is x = 13. -/
theorem quadratic_prime_square_solution :
  ∀ x : ℤ, (∃ p : ℕ, is_prime p ∧ (2 * x^2 - x - 36 : ℤ) = p^2) ↔ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_prime_square_solution_l371_37173


namespace NUMINAMATH_CALUDE_seventh_term_is_84_l371_37113

/-- A sequence where the differences between consecutive terms form a quadratic sequence -/
def CookieSequence (a : ℕ → ℕ) : Prop :=
  ∃ p q r : ℕ,
    (∀ n, a (n + 1) - a n = p * n * n + q * n + r) ∧
    a 1 = 5 ∧ a 2 = 9 ∧ a 3 = 14 ∧ a 4 = 22 ∧ a 5 = 35

theorem seventh_term_is_84 (a : ℕ → ℕ) (h : CookieSequence a) : a 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_84_l371_37113


namespace NUMINAMATH_CALUDE_solutions_bounded_above_not_below_l371_37133

/-- The differential equation y'' = (x^3 + kx)y with initial conditions y(0) = 1 and y'(0) = 0 -/
noncomputable def DiffEq (k : ℝ) (y : ℝ → ℝ) : Prop :=
  (∀ x, (deriv^[2] y) x = (x^3 + k*x) * y x) ∧ y 0 = 1 ∧ (deriv y) 0 = 0

/-- The theorem stating that solutions of y = 0 for the given differential equation
    are bounded above but not below -/
theorem solutions_bounded_above_not_below (k : ℝ) (y : ℝ → ℝ) 
  (h : DiffEq k y) : 
  (∃ M : ℝ, ∀ x : ℝ, y x = 0 → x ≤ M) ∧ 
  (∀ M : ℝ, ∃ x : ℝ, x < M ∧ y x = 0) :=
sorry

end NUMINAMATH_CALUDE_solutions_bounded_above_not_below_l371_37133


namespace NUMINAMATH_CALUDE_gumball_probability_l371_37109

theorem gumball_probability (blue_prob : ℝ) (pink_prob : ℝ) : 
  (blue_prob ^ 2 = 25 / 49) → 
  (blue_prob + pink_prob = 1) → 
  (pink_prob = 2 / 7) := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l371_37109


namespace NUMINAMATH_CALUDE_dance_group_equality_l371_37112

def dance_group_total (initial_boys initial_girls weekly_boys_increase weekly_girls_increase : ℕ) : ℕ :=
  let weeks := (initial_boys - initial_girls) / (weekly_girls_increase - weekly_boys_increase)
  let final_boys := initial_boys + weeks * weekly_boys_increase
  let final_girls := initial_girls + weeks * weekly_girls_increase
  final_boys + final_girls

theorem dance_group_equality (initial_boys initial_girls weekly_boys_increase weekly_girls_increase : ℕ) 
  (h1 : initial_boys = 39)
  (h2 : initial_girls = 23)
  (h3 : weekly_boys_increase = 6)
  (h4 : weekly_girls_increase = 8) :
  dance_group_total initial_boys initial_girls weekly_boys_increase weekly_girls_increase = 174 := by
  sorry

end NUMINAMATH_CALUDE_dance_group_equality_l371_37112


namespace NUMINAMATH_CALUDE_fourth_quadrant_trig_simplification_l371_37121

/-- For an angle α in the fourth quadrant, 
    cos α √((1 - sin α) / (1 + sin α)) + sin α √((1 - cos α) / (1 + cos α)) = cos α - sin α -/
theorem fourth_quadrant_trig_simplification (α : Real) 
  (h_fourth_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  Real.cos α - Real.sin α := by
sorry

end NUMINAMATH_CALUDE_fourth_quadrant_trig_simplification_l371_37121


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l371_37150

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- The common ratio
  h : ∀ n, a (n + 1) = q * a n

/-- 
Given a geometric sequence where the third term is equal to twice 
the sum of the first two terms plus 1, and the fourth term is equal 
to twice the sum of the first three terms plus 1, prove that the 
common ratio is 3.
-/
theorem geometric_sequence_common_ratio 
  (seq : GeometricSequence) 
  (h₁ : seq.a 3 = 2 * (seq.a 1 + seq.a 2) + 1)
  (h₂ : seq.a 4 = 2 * (seq.a 1 + seq.a 2 + seq.a 3) + 1) : 
  seq.q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l371_37150


namespace NUMINAMATH_CALUDE_willy_lucy_crayon_difference_l371_37196

theorem willy_lucy_crayon_difference :
  let willy_crayons : ℕ := 1400
  let lucy_crayons : ℕ := 290
  willy_crayons - lucy_crayons = 1110 :=
by sorry

end NUMINAMATH_CALUDE_willy_lucy_crayon_difference_l371_37196


namespace NUMINAMATH_CALUDE_walking_speed_ratio_l371_37114

/-- The ratio of a slower walking speed to a usual walking speed, given the times taken for the same distance. -/
theorem walking_speed_ratio (usual_time slower_time : ℝ) 
  (h1 : usual_time = 32)
  (h2 : slower_time = 40) :
  (usual_time / slower_time) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_ratio_l371_37114


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l371_37118

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 6) = 10 → x = 106 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l371_37118


namespace NUMINAMATH_CALUDE_binary_addition_multiplication_l371_37195

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_addition_multiplication : 
  let b1 := [true, true, true, true, true]
  let b2 := [true, true, true, true, true, true, true, true]
  let b3 := [false, true]
  (binary_to_decimal b1 + binary_to_decimal b2) * binary_to_decimal b3 = 572 := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_multiplication_l371_37195


namespace NUMINAMATH_CALUDE_proposition_is_false_l371_37120

theorem proposition_is_false : ∃ m n : ℤ, m > n ∧ m^2 ≤ n^2 := by sorry

end NUMINAMATH_CALUDE_proposition_is_false_l371_37120
