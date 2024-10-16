import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_vertices_l924_92447

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the vertices of the hyperbola
def vertices : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Theorem: The distance between the vertices of the hyperbola is 8
theorem distance_between_vertices :
  ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l924_92447


namespace NUMINAMATH_CALUDE_union_of_sets_l924_92470

theorem union_of_sets (A B : Set ℕ) (h1 : A = {0, 1}) (h2 : B = {2}) :
  A ∪ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l924_92470


namespace NUMINAMATH_CALUDE_tea_bags_in_box_l924_92490

theorem tea_bags_in_box : ∀ n : ℕ,
  (2 * n ≤ 41 ∧ 41 ≤ 3 * n) ∧
  (2 * n ≤ 58 ∧ 58 ≤ 3 * n) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_tea_bags_in_box_l924_92490


namespace NUMINAMATH_CALUDE_curve_slope_implies_a_range_l924_92474

/-- The curve y = ln x + ax^2 has no tangent lines with negative slopes for all x > 0 -/
def no_negative_slopes (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1 / x + 2 * a * x) ≥ 0

/-- The theorem states that if the curve has no tangent lines with negative slopes,
    then a is in the range [0, +∞) -/
theorem curve_slope_implies_a_range (a : ℝ) :
  no_negative_slopes a → a ∈ Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_curve_slope_implies_a_range_l924_92474


namespace NUMINAMATH_CALUDE_inequality_problem_l924_92442

theorem inequality_problem (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 2*x^2 - 9*x + m < 0) → 
  m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l924_92442


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l924_92402

theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 9*k = 0 ∧ x₂^2 - 6*x₂ + 9*k = 0) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l924_92402


namespace NUMINAMATH_CALUDE_right_triangle_legs_l924_92498

theorem right_triangle_legs (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  c = 10 → -- hypotenuse length
  r = 2 → -- inscribed circle radius
  a + b - c = 2 * r → -- formula for inscribed circle radius
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  ((a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_legs_l924_92498


namespace NUMINAMATH_CALUDE_bus_passengers_count_l924_92458

theorem bus_passengers_count :
  let men_count : ℕ := 18
  let women_count : ℕ := 26
  let children_count : ℕ := 10
  let total_passengers : ℕ := men_count + women_count + children_count
  total_passengers = 54 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_count_l924_92458


namespace NUMINAMATH_CALUDE_max_3k_value_l924_92471

theorem max_3k_value (k : ℝ) : 
  (∃ x : ℝ, Real.sqrt (x^2 - k) + 2 * Real.sqrt (x^3 - 1) = x) →
  k ≥ 0 →
  k < 2 →
  ∃ m : ℝ, m = 4 ∧ ∀ k' : ℝ, 
    (∃ x : ℝ, Real.sqrt (x'^2 - k') + 2 * Real.sqrt (x'^3 - 1) = x') →
    k' ≥ 0 →
    k' < 2 →
    3 * k' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_3k_value_l924_92471


namespace NUMINAMATH_CALUDE_ceiling_of_negative_decimal_l924_92428

theorem ceiling_of_negative_decimal : ⌈(-3.87 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_decimal_l924_92428


namespace NUMINAMATH_CALUDE_ratio_bounds_in_acute_triangle_l924_92432

theorem ratio_bounds_in_acute_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  -- Sum of angles in a triangle
  A + B + C = π ∧
  -- A = 2B
  A = 2 * B ∧
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c / (Real.sin C) = a / (Real.sin A) →
  -- Conclusion: a/b is bounded by √2 and √3
  Real.sqrt 2 < a / b ∧ a / b < Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ratio_bounds_in_acute_triangle_l924_92432


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l924_92461

theorem quadratic_square_of_binomial (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 4 * x^2 + 18 * x + a = (2 * x + b)^2) → a = 81 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l924_92461


namespace NUMINAMATH_CALUDE_train_speeds_l924_92415

/-- Represents the speeds and lengths of two trains meeting on parallel tracks -/
structure TrainMeeting where
  speed1 : ℝ  -- Speed of first train in m/s
  speed2 : ℝ  -- Speed of second train in m/s
  length1 : ℝ  -- Length of first train in meters
  length2 : ℝ  -- Length of second train in meters
  initialDistance : ℝ  -- Initial distance between trains in meters
  meetingTime : ℝ  -- Time taken for trains to meet in seconds
  timeDifference : ℝ  -- Difference in time taken to pass a signal

/-- The theorem stating the speeds of the trains given the conditions -/
theorem train_speeds (tm : TrainMeeting) 
  (h1 : tm.length1 = 490)
  (h2 : tm.length2 = 210)
  (h3 : tm.initialDistance = 700)
  (h4 : tm.meetingTime = 28)
  (h5 : tm.timeDifference = 35)
  (h6 : tm.initialDistance = tm.meetingTime * (tm.speed1 + tm.speed2))
  (h7 : tm.length1 / tm.speed1 - tm.length2 / tm.speed2 = tm.timeDifference) :
  tm.speed1 = 10 ∧ tm.speed2 = 15 := by
  sorry


end NUMINAMATH_CALUDE_train_speeds_l924_92415


namespace NUMINAMATH_CALUDE_equation_solutions_l924_92440

theorem equation_solutions :
  (∀ x : ℝ, x^2 + 2*x - 8 = 0 ↔ x = -4 ∨ x = 2) ∧
  (∀ x : ℝ, 2*(x+3)^2 = x*(x+3) ↔ x = -3 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l924_92440


namespace NUMINAMATH_CALUDE_club_members_count_l924_92464

theorem club_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l924_92464


namespace NUMINAMATH_CALUDE_packing_peanuts_theorem_l924_92475

/-- Calculates the amount of packing peanuts needed for each small order -/
def packing_peanuts_per_small_order (total_peanuts : ℕ) (large_orders : ℕ) (small_orders : ℕ) (peanuts_per_large_order : ℕ) : ℕ :=
  (total_peanuts - large_orders * peanuts_per_large_order) / small_orders

/-- Theorem: Given the conditions, the amount of packing peanuts needed for each small order is 50g -/
theorem packing_peanuts_theorem :
  packing_peanuts_per_small_order 800 3 4 200 = 50 := by
  sorry

end NUMINAMATH_CALUDE_packing_peanuts_theorem_l924_92475


namespace NUMINAMATH_CALUDE_banana_profit_calculation_grocer_profit_is_eight_dollars_l924_92493

/-- Calculates the profit for a grocer selling bananas -/
theorem banana_profit_calculation (purchase_price : ℚ) (purchase_weight : ℚ) 
  (sale_price : ℚ) (sale_weight : ℚ) (total_weight : ℚ) : ℚ :=
  let cost_per_pound := purchase_price / purchase_weight
  let revenue_per_pound := sale_price / sale_weight
  let total_cost := cost_per_pound * total_weight
  let total_revenue := revenue_per_pound * total_weight
  let profit := total_revenue - total_cost
  profit

/-- Proves that the grocer's profit is $8.00 given the specified conditions -/
theorem grocer_profit_is_eight_dollars : 
  banana_profit_calculation (1/2) 3 1 4 96 = 8 := by
  sorry

end NUMINAMATH_CALUDE_banana_profit_calculation_grocer_profit_is_eight_dollars_l924_92493


namespace NUMINAMATH_CALUDE_term_without_x_in_special_expansion_l924_92457

/-- Given a binomial expansion of (x³ + 1/x²)^n where n is such that only
    the coefficient of the sixth term is maximum, the term without x is 210 -/
theorem term_without_x_in_special_expansion :
  ∃ n : ℕ,
    (∀ k : ℕ, k ≠ 5 → Nat.choose n k ≤ Nat.choose n 5) ∧
    (∃ r : ℕ, Nat.choose n r = 210 ∧ 3 * n = 5 * r) :=
sorry

end NUMINAMATH_CALUDE_term_without_x_in_special_expansion_l924_92457


namespace NUMINAMATH_CALUDE_yurts_are_xarps_and_zarqs_l924_92433

-- Define the sets
variable (U : Type) -- Universe set
variable (Xarp Zarq Yurt Wint : Set U)

-- Define the conditions
variable (h1 : Xarp ⊆ Zarq)
variable (h2 : Yurt ⊆ Zarq)
variable (h3 : Xarp ⊆ Wint)
variable (h4 : Yurt ⊆ Xarp)

-- Theorem to prove
theorem yurts_are_xarps_and_zarqs : Yurt ⊆ Xarp ∩ Zarq :=
sorry

end NUMINAMATH_CALUDE_yurts_are_xarps_and_zarqs_l924_92433


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l924_92411

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l924_92411


namespace NUMINAMATH_CALUDE_nina_taller_than_lena_probability_l924_92477

-- Define the set of friends
inductive Friend
| Masha
| Nina
| Lena
| Olya

-- Define the height relation
def taller_than (a b : Friend) : Prop := sorry

-- Define the conditions
axiom different_heights :
  ∀ (a b : Friend), a ≠ b → (taller_than a b ∨ taller_than b a)

axiom nina_shorter_than_masha :
  taller_than Friend.Masha Friend.Nina

axiom lena_taller_than_olya :
  taller_than Friend.Lena Friend.Olya

-- Define the probability function
noncomputable def probability (event : Prop) : ℝ := sorry

-- Theorem to prove
theorem nina_taller_than_lena_probability :
  probability (taller_than Friend.Nina Friend.Lena) = 0 := by sorry

end NUMINAMATH_CALUDE_nina_taller_than_lena_probability_l924_92477


namespace NUMINAMATH_CALUDE_sum_of_positive_reals_l924_92400

theorem sum_of_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_reals_l924_92400


namespace NUMINAMATH_CALUDE_constant_ratio_theorem_l924_92487

theorem constant_ratio_theorem (x₁ x₂ : ℚ) (y₁ y₂ : ℚ) (k : ℚ) :
  (2 * x₁ - 5) / (y₁ + 10) = k →
  (2 * x₂ - 5) / (y₂ + 10) = k →
  x₁ = 5 →
  y₁ = 4 →
  y₂ = 8 →
  x₂ = 40 / 7 := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_theorem_l924_92487


namespace NUMINAMATH_CALUDE_x_value_l924_92414

theorem x_value : ∃ x : ℤ, 9823 + x = 13200 ∧ x = 3377 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l924_92414


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l924_92405

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Digit
  b : Digit
  c : Digit
  d : Digit

def FourDigitNumber.value (n : FourDigitNumber) : ℕ :=
  1000 * n.a.val + 100 * n.b.val + 10 * n.c.val + n.d.val

def adjacent (x y : Digit) : Prop :=
  x.val + 1 = y.val ∨ y.val + 1 = x.val

theorem cryptarithm_solution :
  ∃! (n : FourDigitNumber),
    adjacent n.a n.c
    ∧ (n.b.val + 2 = n.d.val ∨ n.d.val + 2 = n.b.val)
    ∧ (∃ (e f g h i j : Digit),
        g.val * 10 + h.val = 19
        ∧ f.val + j.val = 14
        ∧ e.val + i.val = 10
        ∧ n.value = 5240) := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l924_92405


namespace NUMINAMATH_CALUDE_drink_conversion_l924_92453

theorem drink_conversion (x : ℚ) : 
  (4 / (4 + x) * 63 = 3 / 7 * (63 + 21)) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_drink_conversion_l924_92453


namespace NUMINAMATH_CALUDE_boys_on_slide_l924_92412

theorem boys_on_slide (initial_boys additional_boys : ℕ) 
  (h1 : initial_boys = 22)
  (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 := by
sorry

end NUMINAMATH_CALUDE_boys_on_slide_l924_92412


namespace NUMINAMATH_CALUDE_ice_cream_survey_l924_92434

theorem ice_cream_survey (total_people : ℕ) (ice_cream_angle : ℕ) :
  total_people = 620 →
  ice_cream_angle = 198 →
  ⌊(total_people : ℝ) * (ice_cream_angle : ℝ) / 360⌋ = 341 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_survey_l924_92434


namespace NUMINAMATH_CALUDE_distribute_5_3_l924_92417

/-- The number of ways to distribute n college graduates to k employers,
    with each employer receiving at least 1 graduate -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 college graduates to 3 employers,
    with each employer receiving at least 1 graduate, is 150 -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l924_92417


namespace NUMINAMATH_CALUDE_absolute_value_expression_l924_92456

theorem absolute_value_expression (x : ℝ) (E : ℝ) :
  x = 10 ∧ 30 - |E| = 26 → E = 4 ∨ E = -4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l924_92456


namespace NUMINAMATH_CALUDE_garden_area_l924_92455

/-- The area of a rectangular garden given specific walking conditions -/
theorem garden_area (length width : ℝ) : 
  length * 30 = 1500 →
  2 * (length + width) * 12 = 1500 →
  length * width = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l924_92455


namespace NUMINAMATH_CALUDE_work_completion_time_l924_92404

theorem work_completion_time (x_time y_time : ℝ) (hx : x_time = 30) (hy : y_time = 45) :
  (1 / x_time + 1 / y_time)⁻¹ = 18 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l924_92404


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l924_92476

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem a_equals_one_sufficient_not_necessary :
  ∃ (a : ℝ),
    (a = 1 → is_pure_imaginary ((a - 1) * (a + 2) + (a + 3) * Complex.I)) ∧
    (∃ (b : ℝ), b ≠ 1 ∧ is_pure_imaginary ((b - 1) * (b + 2) + (b + 3) * Complex.I)) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l924_92476


namespace NUMINAMATH_CALUDE_f_properties_l924_92406

def f (a x : ℝ) : ℝ := x * |x - a|

theorem f_properties (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x + x < f a y + y ↔ -1 ≤ a ∧ a ≤ 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < 1 ↔ 3/2 < a ∧ a < 2) ∧
  (a ≥ 2 →
    (∀ x : ℝ, x ∈ Set.Icc 2 4 → 
      (a > 8 → f a x ∈ Set.Icc (2*a-4) (4*a-16)) ∧
      (4 ≤ a ∧ a < 6 → f a x ∈ Set.Icc (4*a-16) (a^2/4)) ∧
      (6 ≤ a ∧ a ≤ 8 → f a x ∈ Set.Icc (2*a-4) (a^2/4)) ∧
      (2 ≤ a ∧ a < 10/3 → f a x ∈ Set.Icc 0 (16-4*a)) ∧
      (10/3 ≤ a ∧ a < 4 → f a x ∈ Set.Icc 0 (2*a-4)))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l924_92406


namespace NUMINAMATH_CALUDE_maria_cookies_distribution_l924_92491

/-- Calculates the number of cookies per bag given the total number of cookies and the number of bags. -/
def cookiesPerBag (totalCookies : ℕ) (numBags : ℕ) : ℕ :=
  totalCookies / numBags

theorem maria_cookies_distribution (chocolateChipCookies oatmealCookies numBags : ℕ) 
  (h1 : chocolateChipCookies = 33)
  (h2 : oatmealCookies = 2)
  (h3 : numBags = 7) :
  cookiesPerBag (chocolateChipCookies + oatmealCookies) numBags = 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_distribution_l924_92491


namespace NUMINAMATH_CALUDE_charles_pictures_l924_92419

theorem charles_pictures (total_papers : ℕ) (pictures_before_work : ℕ) (pictures_after_work : ℕ) (papers_left : ℕ) :
  let total_yesterday := pictures_before_work + pictures_after_work
  let used_papers := total_papers - papers_left
  used_papers - total_yesterday = total_papers - papers_left - (pictures_before_work + pictures_after_work) :=
by sorry

end NUMINAMATH_CALUDE_charles_pictures_l924_92419


namespace NUMINAMATH_CALUDE_remainder_problem_l924_92426

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 31 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l924_92426


namespace NUMINAMATH_CALUDE_abs_negative_eight_l924_92483

theorem abs_negative_eight : |(-8 : ℤ)| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_eight_l924_92483


namespace NUMINAMATH_CALUDE_james_coins_value_l924_92427

/-- Represents the value of James's coins in cents -/
def coin_value : ℕ := 38

/-- Represents the total number of coins James has -/
def total_coins : ℕ := 15

/-- Represents the number of nickels James has -/
def num_nickels : ℕ := 6

/-- Represents the number of pennies James has -/
def num_pennies : ℕ := 9

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

theorem james_coins_value :
  (num_nickels * nickel_value + num_pennies * penny_value = coin_value) ∧
  (num_nickels + num_pennies = total_coins) ∧
  (num_pennies = num_nickels + 2) := by
  sorry

end NUMINAMATH_CALUDE_james_coins_value_l924_92427


namespace NUMINAMATH_CALUDE_ellipse_properties_l924_92424

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- Given ellipse properties, prove eccentricity and equation -/
theorem ellipse_properties (e : Ellipse) 
  (h1 : e.a = (3/2) * e.b)  -- Ratio of major to minor axis
  (h2 : e.c = 2)            -- Focus at (0, -2)
  : e.c / e.a = Real.sqrt 5 / 3 ∧   -- Eccentricity
    ∀ x y : ℝ, (y^2 / (36/5) + x^2 / (16/5) = 1) ↔ 
    (y^2 / e.b^2 + x^2 / e.a^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l924_92424


namespace NUMINAMATH_CALUDE_cell_growth_10_days_l924_92465

/-- Calculates the number of cells after a given number of days, 
    starting with an initial population that doubles every two days. -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 2)

/-- Theorem stating that given 4 initial cells that double every two days for 10 days, 
    the final number of cells is 64. -/
theorem cell_growth_10_days : cell_population 4 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cell_growth_10_days_l924_92465


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l924_92450

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l924_92450


namespace NUMINAMATH_CALUDE_gaeun_wins_l924_92459

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Nana's flight distance in meters -/
def nana_distance_m : ℝ := 1.618

/-- Gaeun's flight distance in centimeters -/
def gaeun_distance_cm : ℝ := 162.3

/-- Theorem stating that Gaeun's flight distance is greater than Nana's by 0.5 cm -/
theorem gaeun_wins :
  gaeun_distance_cm - (nana_distance_m * meters_to_cm) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_gaeun_wins_l924_92459


namespace NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l924_92488

theorem positive_integer_pairs_satisfying_equation :
  ∀ a b : ℕ+, 
    (a.val * b.val - a.val - b.val = 12) ↔ ((a = 2 ∧ b = 14) ∨ (a = 14 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_satisfying_equation_l924_92488


namespace NUMINAMATH_CALUDE_multiply_by_15_subtract_1_l924_92421

theorem multiply_by_15_subtract_1 (x : ℝ) : 15 * x = 45 → x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_15_subtract_1_l924_92421


namespace NUMINAMATH_CALUDE_triangle_angles_l924_92443

theorem triangle_angles (a b c : ℝ) (h1 : a = 2) (h2 : b = 2) (h3 : c = Real.sqrt 6 - Real.sqrt 2) :
  ∃ (α β γ : ℝ),
    α = 30 * π / 180 ∧
    β = 75 * π / 180 ∧
    γ = 75 * π / 180 ∧
    (Real.cos α = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    (Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (Real.cos γ = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
    α + β + γ = π := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l924_92443


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l924_92435

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f (x : ℝ) : 
  f x ≤ 5 ↔ -4/3 ≤ x ∧ x ≤ 0 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |a^2 - 3*a| ≤ f x) ↔ -1 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l924_92435


namespace NUMINAMATH_CALUDE_right_triangle_cos_z_l924_92407

theorem right_triangle_cos_z (X Y Z : ℝ) : 
  -- Triangle XYZ is right-angled at X
  X + Y + Z = π →
  X = π / 2 →
  -- sin Y = 3/5
  Real.sin Y = 3 / 5 →
  -- Prove: cos Z = 3/5
  Real.cos Z = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_cos_z_l924_92407


namespace NUMINAMATH_CALUDE_female_worker_wage_l924_92436

theorem female_worker_wage (male_count : ℕ) (female_count : ℕ) (child_count : ℕ)
  (male_wage : ℕ) (child_wage : ℕ) (average_wage : ℕ) :
  male_count = 20 →
  female_count = 15 →
  child_count = 5 →
  male_wage = 35 →
  child_wage = 8 →
  average_wage = 26 →
  ∃ (female_wage : ℕ),
    female_wage * female_count = 
      (male_count + female_count + child_count) * average_wage - 
      (male_count * male_wage + child_count * child_wage) ∧
    female_wage = 20 :=
by
  sorry

#check female_worker_wage

end NUMINAMATH_CALUDE_female_worker_wage_l924_92436


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l924_92489

/-- Two vectors in ℝ² are orthogonal if their dot product is zero -/
def orthogonal (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The vector a -/
def a : ℝ × ℝ := (4, 2)

/-- The vector b -/
def b (y : ℝ) : ℝ × ℝ := (6, y)

/-- Theorem: If a and b are orthogonal, then y = -12 -/
theorem orthogonal_vectors (y : ℝ) :
  orthogonal a (b y) → y = -12 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l924_92489


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l924_92448

/-- A triangle is acute-angled if all its angles are less than 90 degrees -/
def IsAcuteAngled (triangle : Set Point) : Prop :=
  sorry

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side -/
def IsAltitude (segment : Set Point) (triangle : Set Point) : Prop :=
  sorry

/-- The area of a triangle -/
noncomputable def TriangleArea (triangle : Set Point) : ℝ :=
  sorry

/-- A triangle is isosceles if it has at least two equal sides -/
def IsIsosceles (triangle : Set Point) : Prop :=
  sorry

theorem isosceles_triangle_proof (A B C D E : Point) :
  let triangle := {A, B, C}
  IsAcuteAngled triangle →
  IsAltitude {A, D} triangle →
  IsAltitude {B, E} triangle →
  TriangleArea {B, D, E} ≤ TriangleArea {D, E, A} ∧
  TriangleArea {D, E, A} ≤ TriangleArea {E, A, B} ∧
  TriangleArea {E, A, B} ≤ TriangleArea {A, B, D} →
  IsIsosceles triangle :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l924_92448


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l924_92416

theorem greatest_divisor_with_remainders : 
  Nat.gcd (450 - 60) (Nat.gcd (330 - 15) (Nat.gcd (675 - 45) (725 - 25))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l924_92416


namespace NUMINAMATH_CALUDE_pizza_sales_l924_92468

theorem pizza_sales (pepperoni cheese total : ℕ) (h1 : pepperoni = 2) (h2 : cheese = 6) (h3 : total = 14) :
  total - (pepperoni + cheese) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sales_l924_92468


namespace NUMINAMATH_CALUDE_function_value_at_two_l924_92408

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l924_92408


namespace NUMINAMATH_CALUDE_scientific_notation_410000_l924_92438

theorem scientific_notation_410000 : 410000 = 4.1 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_410000_l924_92438


namespace NUMINAMATH_CALUDE_isosceles_base_angle_l924_92452

-- Define an isosceles triangle with one interior angle of 110°
def IsoscelesTriangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ α = β ∧ γ = 110

-- Theorem: In an isosceles triangle with one interior angle of 110°, each base angle measures 35°
theorem isosceles_base_angle (α β γ : ℝ) (h : IsoscelesTriangle α β γ) : α = 35 ∧ β = 35 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_base_angle_l924_92452


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l924_92496

-- Define the hyperbola
def hyperbola (m : ℤ) (x y : ℝ) : Prop :=
  x^2 / m^2 + y^2 / (m^2 - 4) = 1

-- Define the eccentricity
def eccentricity (m : ℤ) : ℝ :=
  2

-- Theorem statement
theorem hyperbola_eccentricity (m : ℤ) :
  ∃ (e : ℝ), e = eccentricity m ∧ 
  ∀ (x y : ℝ), hyperbola m x y → e = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l924_92496


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l924_92410

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l924_92410


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l924_92437

theorem triangle_angle_measure (P Q R : ℝ) (h1 : P = 90) (h2 : Q = 4 * R - 10) (h3 : P + Q + R = 180) : R = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l924_92437


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_eq_one_l924_92494

theorem sqrt_abs_sum_eq_one (a : ℝ) (h : 1 < a ∧ a < 2) :
  Real.sqrt ((a - 2)^2) + |a - 1| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_eq_one_l924_92494


namespace NUMINAMATH_CALUDE_days_worked_together_is_two_l924_92449

-- Define the efficiencies and time ratios
def efficiency_ratio_A_C : ℚ := 5 / 3
def time_ratio_B_C : ℚ := 2 / 3

-- Define the difference in days between A and C
def days_difference_A_C : ℕ := 6

-- Define the time A took to finish the remaining work
def remaining_work_days_A : ℕ := 6

-- Function to calculate the number of days B and C worked together
def days_worked_together (efficiency_ratio_A_C : ℚ) (time_ratio_B_C : ℚ) 
                         (days_difference_A_C : ℕ) (remaining_work_days_A : ℕ) : ℚ := 
  sorry

-- Theorem statement
theorem days_worked_together_is_two :
  days_worked_together efficiency_ratio_A_C time_ratio_B_C days_difference_A_C remaining_work_days_A = 2 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_together_is_two_l924_92449


namespace NUMINAMATH_CALUDE_triangle_perimeter_l924_92497

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.cos B = 3 ∧
  b * Real.sin A = 4 ∧
  (1/2) * a * c * Real.sin B = 10 →
  a + b + c = 10 + 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l924_92497


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l924_92439

theorem simplify_trigonometric_expression (x : ℝ) : 
  2 * Real.sin (2 * x) * Real.sin x + Real.cos (3 * x) = Real.cos x := by
sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l924_92439


namespace NUMINAMATH_CALUDE_extremum_at_zero_l924_92495

/-- Given a function f(x) = e^x - ax with an extremum at x = 0, prove that a = 1 -/
theorem extremum_at_zero (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x - a * x) ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x ≤ f 0 ∨ f x ≥ f 0)) → 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_at_zero_l924_92495


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l924_92445

theorem polygon_interior_angles (n : ℕ) (d : ℝ) (largest_angle : ℝ) : 
  n ≥ 3 →
  d = 3 →
  largest_angle = 150 →
  (n : ℝ) * (2 * largest_angle - d * (n - 1)) / 2 = 180 * (n - 2) →
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l924_92445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_max_sum_at_25_l924_92480

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  second_eighth_sum : a 2 + a 8 = 82
  sum_equality : S 41 = S 9

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∃ d : ℝ, ∀ n : ℕ, seq.a n = 51 - 2 * n) ∧
  (∃ n : ℕ, seq.S n = 625 ∧ ∀ m : ℕ, seq.S m ≤ seq.S n) := by
  sorry

/-- The maximum value of S_n occurs when n = 25 -/
theorem max_sum_at_25 (seq : ArithmeticSequence) :
  seq.S 25 = 625 ∧ ∀ n : ℕ, seq.S n ≤ seq.S 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_max_sum_at_25_l924_92480


namespace NUMINAMATH_CALUDE_train_speed_problem_l924_92451

/-- Given two trains starting from the same station, traveling along parallel tracks in the same direction,
    with one train traveling at 31 mph, and the distance between them after 8 hours being 160 miles,
    prove that the speed of the first train is 51 mph. -/
theorem train_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed for the first train
  (v - 31) * 8 = 160 → 
  v = 51 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l924_92451


namespace NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l924_92444

theorem sum_of_squares_with_given_means (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l924_92444


namespace NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l924_92479

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ ((3*y)^2 + 3*41*(3*y) + 41^2))) ∧
  (53 ∣ ((3*x)^2 + 3*41*(3*x) + 41^2)) ∧
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_multiple_of_53_l924_92479


namespace NUMINAMATH_CALUDE_g_is_even_l924_92446

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g as f(x) + f(-x)
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem: g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by sorry

end NUMINAMATH_CALUDE_g_is_even_l924_92446


namespace NUMINAMATH_CALUDE_mowers_for_three_hours_l924_92478

/-- The number of mowers required to drink a barrel of kvass in a given time -/
def mowers_required (initial_mowers : ℕ) (initial_hours : ℕ) (target_hours : ℕ) : ℕ :=
  (initial_mowers * initial_hours) / target_hours

/-- Theorem stating that 16 mowers are required to drink a barrel of kvass in 3 hours,
    given that 6 mowers can drink it in 8 hours -/
theorem mowers_for_three_hours :
  mowers_required 6 8 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_mowers_for_three_hours_l924_92478


namespace NUMINAMATH_CALUDE_unknown_number_proof_l924_92485

theorem unknown_number_proof (x : ℝ) : 3034 - (x / 200.4) = 3029 → x = 1002 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l924_92485


namespace NUMINAMATH_CALUDE_banana_arrangements_l924_92403

def banana_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements :
  (banana_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l924_92403


namespace NUMINAMATH_CALUDE_oliver_shelves_l924_92401

/-- Given the total number of books, books taken by librarian, and books per shelf,
    calculate the number of shelves needed. -/
def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - books_taken) / books_per_shelf

/-- Proof that 9 shelves are needed for Oliver's book arrangement. -/
theorem oliver_shelves :
  shelves_needed 46 10 4 = 9 := by
  sorry

#eval shelves_needed 46 10 4

end NUMINAMATH_CALUDE_oliver_shelves_l924_92401


namespace NUMINAMATH_CALUDE_twice_square_sum_l924_92492

theorem twice_square_sum (x y : ℤ) : x^4 + y^4 + (x+y)^4 = 2 * (x^2 + x*y + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_twice_square_sum_l924_92492


namespace NUMINAMATH_CALUDE_female_red_ants_percentage_l924_92486

/-- Given an ant colony where 85% of the population is red and 46.75% of the total population
    are male red ants, prove that 45% of the red ants are females. -/
theorem female_red_ants_percentage
  (total_population : ℝ)
  (red_ants_percentage : ℝ)
  (male_red_ants_percentage : ℝ)
  (h1 : red_ants_percentage = 85)
  (h2 : male_red_ants_percentage = 46.75)
  (h3 : total_population > 0) :
  let total_red_ants := red_ants_percentage * total_population / 100
  let male_red_ants := male_red_ants_percentage * total_population / 100
  let female_red_ants := total_red_ants - male_red_ants
  female_red_ants / total_red_ants * 100 = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_female_red_ants_percentage_l924_92486


namespace NUMINAMATH_CALUDE_basketball_wins_l924_92499

theorem basketball_wins (x : ℚ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : x + (5/8)*x + (x + (5/8)*x) = 130) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_basketball_wins_l924_92499


namespace NUMINAMATH_CALUDE_smallest_divisible_by_14_15_18_l924_92482

theorem smallest_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ 14 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n ∧ ∀ m : ℕ, m > 0 → 14 ∣ m → 15 ∣ m → 18 ∣ m → n ≤ m :=
by
  use 630
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_14_15_18_l924_92482


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_l924_92430

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_l924_92430


namespace NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_pqr_l924_92481

theorem log_ten_seven_in_terms_of_pqr (p q r : ℝ) 
  (hp : Real.log 3 / Real.log 8 = p)
  (hq : Real.log 5 / Real.log 3 = q)
  (hr : Real.log 7 / Real.log 4 = r) :
  Real.log 7 / Real.log 10 = 2 * r / (1 + 4 * q * p) := by
  sorry

end NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_pqr_l924_92481


namespace NUMINAMATH_CALUDE_max_candies_consumed_max_candies_is_1225_l924_92418

/-- The number of initial ones on the board -/
def initial_ones : ℕ := 50

/-- The number of minutes the process continues -/
def total_minutes : ℕ := 50

/-- The number of candies consumed is equal to the number of edges in a complete graph -/
theorem max_candies_consumed (n : ℕ) (h : n = initial_ones) :
  (n * (n - 1)) / 2 = total_minutes * (total_minutes - 1) / 2 := by sorry

/-- The maximum number of candies consumed after the process -/
def max_candies : ℕ := (initial_ones * (initial_ones - 1)) / 2

/-- Proof that the maximum number of candies consumed is 1225 -/
theorem max_candies_is_1225 : max_candies = 1225 := by sorry

end NUMINAMATH_CALUDE_max_candies_consumed_max_candies_is_1225_l924_92418


namespace NUMINAMATH_CALUDE_special_heptagon_perimeter_l924_92463

/-- A heptagon with six sides of length 3 and one side of length 5 -/
structure SpecialHeptagon where
  side_length_six : ℝ
  side_length_one : ℝ
  is_heptagon : side_length_six = 3 ∧ side_length_one = 5

/-- The perimeter of a SpecialHeptagon -/
def perimeter (h : SpecialHeptagon) : ℝ :=
  6 * h.side_length_six + h.side_length_one

theorem special_heptagon_perimeter (h : SpecialHeptagon) :
  perimeter h = 23 := by
  sorry

end NUMINAMATH_CALUDE_special_heptagon_perimeter_l924_92463


namespace NUMINAMATH_CALUDE_range_of_a_l924_92441

def p (a : ℝ) : Prop := ∀ x : ℝ, (a - 3/2) ^ x > 0 ∧ (a - 3/2) ^ x < 1

def q (a : ℝ) : Prop := ∃ f : ℝ → ℝ, (∀ x ∈ [0, a], f x = x^2 - 4*x + 3) ∧
  (∀ y ∈ Set.range f, -1 ≤ y ∧ y ≤ 3)

theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) →
  ((3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l924_92441


namespace NUMINAMATH_CALUDE_odd_function_with_property_M_even_function_with_property_M_l924_92431

def has_property_M (f : ℝ → ℝ) (A : Set ℝ) :=
  ∃ c : ℝ, ∀ x ∈ A, Real.exp x * (f x - Real.exp x) = c

def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

theorem odd_function_with_property_M (f : ℝ → ℝ) (h1 : is_odd f) (h2 : has_property_M f Set.univ) :
  ∀ x, f x = Real.exp x - 1 / Real.exp x := by sorry

theorem even_function_with_property_M (g : ℝ → ℝ) (h1 : is_even g)
    (h2 : has_property_M g (Set.Icc (-1) 1))
    (h3 : ∀ x ∈ Set.Icc (-1) 1, g (2 * x) - 2 * Real.exp 1 * g x + n > 0) :
  n > Real.exp 2 + 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_with_property_M_even_function_with_property_M_l924_92431


namespace NUMINAMATH_CALUDE_toys_produced_daily_l924_92413

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 8000

/-- The number of working days per week -/
def working_days : ℕ := 4

/-- The number of toys produced each day -/
def toys_per_day : ℕ := toys_per_week / working_days

/-- Theorem stating that the number of toys produced each day is 2000 -/
theorem toys_produced_daily :
  toys_per_day = 2000 :=
by sorry

end NUMINAMATH_CALUDE_toys_produced_daily_l924_92413


namespace NUMINAMATH_CALUDE_rectangular_map_area_l924_92454

/-- The area of a rectangular map with given length and width. -/
def map_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular map with length 5 meters and width 2 meters is 10 square meters. -/
theorem rectangular_map_area :
  map_area 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_map_area_l924_92454


namespace NUMINAMATH_CALUDE_sequence_sum_l924_92484

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_sum (a b : ℕ → ℝ) :
  is_geometric a →
  is_arithmetic b →
  a 3 * a 11 = 4 * a 7 →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l924_92484


namespace NUMINAMATH_CALUDE_price_per_dozen_calculation_l924_92462

/-- The price per dozen of additional doughnuts -/
def price_per_dozen (first_doughnut_price : ℚ) (total_cost : ℚ) (total_doughnuts : ℕ) : ℚ :=
  (total_cost - first_doughnut_price) / ((total_doughnuts - 1 : ℚ) / 12)

/-- Theorem stating the price per dozen of additional doughnuts -/
theorem price_per_dozen_calculation (first_doughnut_price : ℚ) (total_cost : ℚ) (total_doughnuts : ℕ) 
  (h1 : first_doughnut_price = 1)
  (h2 : total_cost = 24)
  (h3 : total_doughnuts = 48) :
  price_per_dozen first_doughnut_price total_cost total_doughnuts = 276 / 47 :=
by sorry

end NUMINAMATH_CALUDE_price_per_dozen_calculation_l924_92462


namespace NUMINAMATH_CALUDE_massachusetts_avenue_pairings_l924_92425

/-- Represents the number of possible pairings for n blocks -/
def pairings : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => pairings (n + 1) + pairings n

/-- The 10th Fibonacci number -/
def fib10 : ℕ := pairings 10

theorem massachusetts_avenue_pairings :
  fib10 = 89 :=
by sorry

end NUMINAMATH_CALUDE_massachusetts_avenue_pairings_l924_92425


namespace NUMINAMATH_CALUDE_brick_height_proof_l924_92473

/-- Proves that given a wall of specific dimensions and bricks of specific dimensions,
    if a certain number of bricks are used, then the height of each brick is 6 cm. -/
theorem brick_height_proof (wall_length wall_height wall_thickness : ℝ)
                           (brick_length brick_width brick_height : ℝ)
                           (num_bricks : ℝ) :
  wall_length = 8 →
  wall_height = 6 →
  wall_thickness = 0.02 →
  brick_length = 0.05 →
  brick_width = 0.11 →
  brick_height = 0.06 →
  num_bricks = 2909.090909090909 →
  brick_height * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_proof_l924_92473


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l924_92460

theorem sum_reciprocal_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l924_92460


namespace NUMINAMATH_CALUDE_relay_race_selections_l924_92469

def number_of_athletes : ℕ := 6
def number_of_legs : ℕ := 4
def athletes_cant_run_first : ℕ := 2

theorem relay_race_selections :
  let total_athletes := number_of_athletes
  let race_legs := number_of_legs
  let excluded_first_leg := athletes_cant_run_first
  let first_leg_choices := total_athletes - excluded_first_leg
  let remaining_athletes := total_athletes - 1
  let remaining_legs := race_legs - 1
  (first_leg_choices : ℕ) * (remaining_athletes.choose remaining_legs) = 240 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_selections_l924_92469


namespace NUMINAMATH_CALUDE_inverse_function_problem_l924_92420

theorem inverse_function_problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) 
  (h1 : Function.LeftInverse g_inv g) 
  (h2 : Function.RightInverse g_inv g)
  (h3 : g 4 = 6)
  (h4 : g 6 = 2)
  (h5 : g 3 = 7) :
  g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l924_92420


namespace NUMINAMATH_CALUDE_points_difference_l924_92422

theorem points_difference (zach_points ben_points : ℕ) 
  (h1 : zach_points = 42) 
  (h2 : ben_points = 21) : 
  zach_points - ben_points = 21 := by
sorry

end NUMINAMATH_CALUDE_points_difference_l924_92422


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_29_l924_92409

theorem modular_inverse_of_5_mod_29 :
  ∃ a : ℕ, a ≤ 28 ∧ (5 * a) % 29 = 1 ∧ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_29_l924_92409


namespace NUMINAMATH_CALUDE_class_size_quotient_l924_92472

theorem class_size_quotient (N H J : ℝ) 
  (h1 : N / H = 1.2) 
  (h2 : H / J = 5/6) : 
  N / J = 1 := by
  sorry

end NUMINAMATH_CALUDE_class_size_quotient_l924_92472


namespace NUMINAMATH_CALUDE_tank_water_level_l924_92466

theorem tank_water_level (tank_capacity : ℝ) (initial_level : ℝ) 
  (empty_percentage : ℝ) (fill_percentage : ℝ) (final_volume : ℝ) :
  tank_capacity = 8000 →
  empty_percentage = 0.4 →
  fill_percentage = 0.3 →
  final_volume = 4680 →
  final_volume = initial_level * (1 - empty_percentage) * (1 + fill_percentage) →
  initial_level / tank_capacity = 0.75 := by
sorry

end NUMINAMATH_CALUDE_tank_water_level_l924_92466


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l924_92467

/-- Given two 2D vectors a and b, if a is perpendicular to (a + m*b), then m = 2/5 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (ha : a = (1, -1)) 
  (hb : b = (-2, 3)) 
  (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) : 
  m = 2/5 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l924_92467


namespace NUMINAMATH_CALUDE_square_diagonal_length_l924_92429

/-- The diagonal length of a square with area 72 square meters is 12 meters. -/
theorem square_diagonal_length (area : ℝ) (side : ℝ) (diagonal : ℝ) : 
  area = 72 → 
  area = side ^ 2 → 
  diagonal ^ 2 = 2 * side ^ 2 → 
  diagonal = 12 := by
  sorry


end NUMINAMATH_CALUDE_square_diagonal_length_l924_92429


namespace NUMINAMATH_CALUDE_periodic_function_value_l924_92423

/-- Given a function f(x) = a * sin(π * x + α) + b * cos(π * x + β) + 4,
    where a, b, α, β are non-zero real numbers, and f(2012) = 6,
    prove that f(2013) = 2 -/
theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 2012 = 6) → (f 2013 = 2) := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l924_92423
