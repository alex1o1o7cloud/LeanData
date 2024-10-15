import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_f_l2843_284390

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem max_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (∀ x ∈ Set.Icc 1 4, f a x ≥ -16/3) →
  (∃ x ∈ Set.Icc 1 4, f a x = -16/3) →
  (∃ x ∈ Set.Icc 1 4, f a x = 10/3) ∧
  (∀ x ∈ Set.Icc 1 4, f a x ≤ 10/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2843_284390


namespace NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l2843_284352

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c = 0 has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) (h_geom : b^2 = a*c ∧ a*c > 0) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l2843_284352


namespace NUMINAMATH_CALUDE_integers_between_negative_two_and_three_l2843_284341

theorem integers_between_negative_two_and_three :
  {x : ℤ | x > -2 ∧ x ≤ 3} = {-1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_integers_between_negative_two_and_three_l2843_284341


namespace NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l2843_284322

/-- Represents the student population at Gateway Academy -/
structure StudentPopulation where
  total : ℝ
  like_skating : ℝ
  dislike_skating : ℝ
  say_like_actually_like : ℝ
  say_dislike_actually_like : ℝ
  say_like_actually_dislike : ℝ
  say_dislike_actually_dislike : ℝ

/-- The conditions of the problem -/
def gateway_academy (pop : StudentPopulation) : Prop :=
  pop.total > 0 ∧
  pop.like_skating = 0.4 * pop.total ∧
  pop.dislike_skating = 0.6 * pop.total ∧
  pop.say_like_actually_like = 0.7 * pop.like_skating ∧
  pop.say_dislike_actually_like = 0.3 * pop.like_skating ∧
  pop.say_like_actually_dislike = 0.2 * pop.dislike_skating ∧
  pop.say_dislike_actually_dislike = 0.8 * pop.dislike_skating

/-- The theorem to be proved -/
theorem fraction_who_say_dislike_but_like (pop : StudentPopulation) 
  (h : gateway_academy pop) : 
  pop.say_dislike_actually_like / (pop.say_dislike_actually_like + pop.say_dislike_actually_dislike) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l2843_284322


namespace NUMINAMATH_CALUDE_f_min_at_x_min_l2843_284314

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

/-- The point where the minimum of f occurs -/
def x_min : ℝ := 6

theorem f_min_at_x_min :
  ∀ x : ℝ, f x ≥ f x_min :=
sorry

end NUMINAMATH_CALUDE_f_min_at_x_min_l2843_284314


namespace NUMINAMATH_CALUDE_equation_satisfied_when_m_is_34_l2843_284315

theorem equation_satisfied_when_m_is_34 :
  let m : ℕ := 34
  (((1 : ℚ) ^ (m + 1)) / ((5 : ℚ) ^ (m + 1))) * (((1 : ℚ) ^ 18) / ((4 : ℚ) ^ 18)) = 1 / (2 * ((10 : ℚ) ^ 35)) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_when_m_is_34_l2843_284315


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2843_284385

theorem complex_fraction_simplification :
  (5 - 3*I) / (2 - 3*I) = -19/5 - 9/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2843_284385


namespace NUMINAMATH_CALUDE_simplify_expression_l2843_284379

theorem simplify_expression (n : ℕ) : (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2843_284379


namespace NUMINAMATH_CALUDE_hexagon_area_l2843_284343

/-- Right triangle with legs 3 and 4, hypotenuse 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5

/-- Square with side length 3 -/
def square1_area : ℝ := 9

/-- Square with side length 4 -/
def square2_area : ℝ := 16

/-- Rectangle with sides 5 and 6 -/
def rectangle_area : ℝ := 30

/-- Area of the triangle formed by extending one side of the first square -/
def extended_triangle_area : ℝ := 4.5

/-- Theorem: The area of the hexagon DEFGHI is 52.5 -/
theorem hexagon_area (t : RightTriangle) : 
  square1_area + square2_area + rectangle_area + extended_triangle_area = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l2843_284343


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2843_284361

theorem arithmetic_geometric_progression (b c : ℝ) 
  (not_both_one : ¬(b = 1 ∧ c = 1))
  (arithmetic_prog : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2*n)
  (geometric_prog : c / 1 = b / c) :
  100 * (b - c) = 75 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2843_284361


namespace NUMINAMATH_CALUDE_tangent_two_implies_expression_equals_negative_two_l2843_284337

-- Define the theorem
theorem tangent_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / 
  (Real.sin (π / 2 - θ) - Real.sin (π - θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_implies_expression_equals_negative_two_l2843_284337


namespace NUMINAMATH_CALUDE_vector_dot_product_result_l2843_284323

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

theorem vector_dot_product_result :
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_result_l2843_284323


namespace NUMINAMATH_CALUDE_equation_classification_l2843_284394

def equation (m : ℝ) (x : ℝ) : ℝ := (m^2 - 1) * x^2 + (m + 1) * x + (m - 2)

theorem equation_classification (m : ℝ) :
  (∀ x, equation m x = 0 → (m^2 - 1 ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1)) ∧
  (∀ x, equation m x = 0 → (m^2 - 1 = 0 ∧ m + 1 ≠ 0 ↔ m = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_classification_l2843_284394


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_five_l2843_284339

theorem largest_five_digit_divisible_by_five : 
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 5 = 0 → n ≤ 99995 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_five_l2843_284339


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_4_l2843_284310

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Proof that the discriminant of 5x^2 - 9x + 4 is 1 -/
theorem discriminant_of_5x2_minus_9x_plus_4 :
  discriminant 5 (-9) 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_4_l2843_284310


namespace NUMINAMATH_CALUDE_bakery_calculations_l2843_284391

-- Define the bakery's parameters
def cost_price : ℝ := 4
def selling_price : ℝ := 10
def clearance_price : ℝ := 2
def min_loaves : ℕ := 15
def max_loaves : ℕ := 30
def baked_loaves : ℕ := 21

-- Define the demand frequencies
def demand_freq : List (ℕ × ℕ) := [(15, 10), (18, 8), (21, 7), (24, 3), (27, 2)]

-- Calculate the probability of demand being at least 21 loaves
def prob_demand_ge_21 : ℚ := 2/5

-- Calculate the daily profit when demand is 15 loaves
def profit_demand_15 : ℝ := 78

-- Calculate the average daily profit over 30 days
def avg_daily_profit : ℝ := 103.6

theorem bakery_calculations :
  (prob_demand_ge_21 = 2/5) ∧
  (profit_demand_15 = 78) ∧
  (avg_daily_profit = 103.6) := by
  sorry

end NUMINAMATH_CALUDE_bakery_calculations_l2843_284391


namespace NUMINAMATH_CALUDE_david_pushups_count_l2843_284365

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 44

/-- The difference between David's and Zachary's push-ups -/
def pushup_difference : ℕ := 19

/-- The difference between Zachary's and David's crunches -/
def crunch_difference : ℕ := 27

/-- David's push-ups -/
def david_pushups : ℕ := zachary_pushups + pushup_difference

theorem david_pushups_count : david_pushups = 78 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l2843_284365


namespace NUMINAMATH_CALUDE_somu_age_problem_l2843_284370

/-- Proves that Somu was one-fifth of his father's age 8 years ago -/
theorem somu_age_problem (somu_age father_age years_ago : ℕ) : 
  somu_age = 16 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 8 := by
  sorry


end NUMINAMATH_CALUDE_somu_age_problem_l2843_284370


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2843_284348

-- Equation 1
theorem solve_equation_one : 
  ∃ x : ℝ, (3 * x - 4 = -2 * (x - 1)) ∧ (x = 1.2) := by sorry

-- Equation 2
theorem solve_equation_two :
  ∃ x : ℝ, (1 + (2 * x + 1) / 3 = (3 * x - 2) / 2) ∧ (x = 14 / 5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l2843_284348


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l2843_284388

theorem absolute_value_fraction (x y : ℝ) 
  (h : y < Real.sqrt (x - 1) + Real.sqrt (1 - x) + 1) : 
  |y - 1| / (y - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l2843_284388


namespace NUMINAMATH_CALUDE_third_quadrant_angle_sum_l2843_284357

theorem third_quadrant_angle_sum (θ : Real) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.tan (θ - π/4) = 1/3) → 
  (Real.sin θ + Real.cos θ = -3/5 * Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_third_quadrant_angle_sum_l2843_284357


namespace NUMINAMATH_CALUDE_power_inequality_l2843_284329

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  a^a < b^a := by sorry

end NUMINAMATH_CALUDE_power_inequality_l2843_284329


namespace NUMINAMATH_CALUDE_sandwich_cost_is_181_l2843_284397

/-- The cost in cents for Joe to make a deluxe ham and cheese sandwich -/
def sandwich_cost : ℕ :=
  let bread_cost : ℕ := 15 -- Cost of one slice of bread in cents
  let ham_cost : ℕ := 25 -- Cost of one slice of ham in cents
  let cheese_cost : ℕ := 35 -- Cost of one slice of cheese in cents
  let mayo_cost : ℕ := 10 -- Cost of one tablespoon of mayonnaise in cents
  let lettuce_cost : ℕ := 5 -- Cost of one lettuce leaf in cents
  let tomato_cost : ℕ := 8 -- Cost of one tomato slice in cents
  
  let bread_slices : ℕ := 2 -- Number of bread slices used
  let ham_slices : ℕ := 2 -- Number of ham slices used
  let cheese_slices : ℕ := 2 -- Number of cheese slices used
  let mayo_tbsp : ℕ := 1 -- Number of tablespoons of mayonnaise used
  let lettuce_leaves : ℕ := 1 -- Number of lettuce leaves used
  let tomato_slices : ℕ := 2 -- Number of tomato slices used
  
  bread_cost * bread_slices +
  ham_cost * ham_slices +
  cheese_cost * cheese_slices +
  mayo_cost * mayo_tbsp +
  lettuce_cost * lettuce_leaves +
  tomato_cost * tomato_slices

theorem sandwich_cost_is_181 : sandwich_cost = 181 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_181_l2843_284397


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2843_284380

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_increasing : IsIncreasingGeometricSequence a)
    (h_a3 : a 3 = 4)
    (h_sum : 1 / a 1 + 1 / a 5 = 5 / 8) :
  a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2843_284380


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2843_284327

theorem least_addition_for_divisibility :
  ∃! x : ℕ, x < 103 ∧ (3457 + x) % 103 = 0 ∧ ∀ y : ℕ, y < x → (3457 + y) % 103 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2843_284327


namespace NUMINAMATH_CALUDE_tuesday_appointment_duration_l2843_284319

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℚ := 20

/-- Duration of Monday appointments in hours -/
def monday_hours : ℚ := 5 * (3/2)

/-- Duration of Thursday appointments in hours -/
def thursday_hours : ℚ := 2 * 2

/-- Duration of Saturday appointment in hours -/
def saturday_hours : ℚ := 6

/-- Total earnings for the week in dollars -/
def total_earnings : ℚ := 410

/-- Duration of Tuesday appointment in hours -/
def tuesday_hours : ℚ := 3

theorem tuesday_appointment_duration :
  hourly_rate * (monday_hours + thursday_hours + saturday_hours + tuesday_hours) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_tuesday_appointment_duration_l2843_284319


namespace NUMINAMATH_CALUDE_initial_ducks_l2843_284347

theorem initial_ducks (initial final additional : ℕ) 
  (h1 : final = initial + additional)
  (h2 : final = 33)
  (h3 : additional = 20) : 
  initial = 13 := by
sorry

end NUMINAMATH_CALUDE_initial_ducks_l2843_284347


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2843_284309

/-- The range of k values for which the line y = kx + 2 intersects the ellipse 2x^2 + 3y^2 = 6 at two distinct points -/
theorem line_ellipse_intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + 2 ∧ y₂ = k * x₂ + 2 ∧
    2 * x₁^2 + 3 * y₁^2 = 6 ∧ 
    2 * x₂^2 + 3 * y₂^2 = 6) ↔ 
  k < -Real.sqrt (2/3) ∨ k > Real.sqrt (2/3) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2843_284309


namespace NUMINAMATH_CALUDE_sea_turtle_count_sea_turtle_count_proof_l2843_284386

theorem sea_turtle_count : ℕ → Prop :=
  fun total_turtles =>
    (total_turtles : ℚ) * (1 : ℚ) / (3 : ℚ) + (28 : ℚ) = total_turtles ∧
    total_turtles = 42

-- Proof
theorem sea_turtle_count_proof : sea_turtle_count 42 := by
  sorry

end NUMINAMATH_CALUDE_sea_turtle_count_sea_turtle_count_proof_l2843_284386


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l2843_284392

/-- A nonzero digit is a natural number between 1 and 9. -/
def NonzeroDigit : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- Convert a three-digit number represented by three digits to a natural number. -/
def threeDigitToNat (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Convert a two-digit number represented by two digits to a natural number. -/
def twoDigitToNat (a b : ℕ) : ℕ := 10 * a + b

/-- The main theorem: the sum of the four numbers always has 5 digits. -/
theorem sum_has_five_digits (A B C : NonzeroDigit) :
  ∃ (n : ℕ), 10000 ≤ 21478 + threeDigitToNat A.val 5 9 + twoDigitToNat B.val 4 + twoDigitToNat C.val 6 ∧
             21478 + threeDigitToNat A.val 5 9 + twoDigitToNat B.val 4 + twoDigitToNat C.val 6 < 100000 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l2843_284392


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2843_284368

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- General form of hyperbola
  (∃ x y : ℝ, y^2 = -4*x) →  -- Parabola equation
  ((-1 : ℝ) = a) →  -- Real axis endpoint coincides with parabola focus
  ((a + b) / a = 2) →  -- Eccentricity is 2
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1) :=  -- Resulting hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2843_284368


namespace NUMINAMATH_CALUDE_jackie_apples_l2843_284334

-- Define the number of apples Adam has
def adam_apples : ℕ := 8

-- Define the difference between Jackie's and Adam's apples
def difference : ℕ := 2

-- Theorem: Jackie has 10 apples
theorem jackie_apples : adam_apples + difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_jackie_apples_l2843_284334


namespace NUMINAMATH_CALUDE_complex_subtraction_multiplication_l2843_284396

theorem complex_subtraction_multiplication (i : ℂ) : 
  (7 - 3*i) - 3*(2 - 5*i) = 1 + 12*i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_multiplication_l2843_284396


namespace NUMINAMATH_CALUDE_imoCandidate1988_l2843_284328

theorem imoCandidate1988 (d r : ℤ) : 
  d > 1 ∧ 
  (∃ k m n : ℤ, 1059 = k * d + r ∧ 
               1417 = m * d + r ∧ 
               2312 = n * d + r) →
  d - r = 15 := by sorry

end NUMINAMATH_CALUDE_imoCandidate1988_l2843_284328


namespace NUMINAMATH_CALUDE_melanie_yard_sale_books_l2843_284307

/-- The number of books Melanie bought at a yard sale -/
def books_bought (initial_books final_books : ℝ) : ℝ :=
  final_books - initial_books

/-- Proof that Melanie bought 87 books at the yard sale -/
theorem melanie_yard_sale_books : books_bought 41.0 128 = 87 := by
  sorry

end NUMINAMATH_CALUDE_melanie_yard_sale_books_l2843_284307


namespace NUMINAMATH_CALUDE_max_xy_value_l2843_284302

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 6) :
  x*y ≤ 3/2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 6 ∧ x₀*y₀ = 3/2 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2843_284302


namespace NUMINAMATH_CALUDE_police_force_female_officers_l2843_284399

theorem police_force_female_officers :
  ∀ (total_female : ℕ) (first_shift_total : ℕ) (first_shift_female_percent : ℚ),
    first_shift_total = 204 →
    first_shift_female_percent = 17 / 100 →
    (first_shift_total / 2 : ℚ) = first_shift_female_percent * total_female →
    total_female = 600 := by
  sorry

end NUMINAMATH_CALUDE_police_force_female_officers_l2843_284399


namespace NUMINAMATH_CALUDE_base_8_first_digit_350_l2843_284305

def base_8_first_digit (n : ℕ) : ℕ :=
  (n / 64) % 8

theorem base_8_first_digit_350 :
  base_8_first_digit 350 = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_8_first_digit_350_l2843_284305


namespace NUMINAMATH_CALUDE_min_value_on_line_min_value_achieved_l2843_284325

/-- The minimum value of 2/a + 3/b for points (a, b) in the first quadrant on the line 2x + 3y = 1 -/
theorem min_value_on_line (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + 3*b = 1) :
  2/a + 3/b ≥ 25 := by
  sorry

/-- The minimum value 25 is achieved for some point on the line -/
theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + 3*b = 1 ∧ |2/a + 3/b - 25| < ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_line_min_value_achieved_l2843_284325


namespace NUMINAMATH_CALUDE_min_both_brown_eyes_and_lunch_box_example_l2843_284353

/-- Given a class of students, calculates the minimum number of students
    who have both brown eyes and a lunch box. -/
def min_both_brown_eyes_and_lunch_box (total : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) : ℕ :=
  max 0 (brown_eyes + lunch_box - total)

/-- Theorem stating that in a class of 35 students, where 18 have brown eyes
    and 25 have a lunch box, at least 8 students have both brown eyes and a lunch box. -/
theorem min_both_brown_eyes_and_lunch_box_example :
  min_both_brown_eyes_and_lunch_box 35 18 25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_both_brown_eyes_and_lunch_box_example_l2843_284353


namespace NUMINAMATH_CALUDE_trigonometric_roots_theorem_l2843_284389

-- Define the equation and its roots
def equation (m : ℝ) (x : ℝ) : Prop := 8 * x^2 + 6 * m * x + 2 * m + 1 = 0

-- Define the interval for α
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < Real.pi

-- Theorem statement
theorem trigonometric_roots_theorem (α : ℝ) (m : ℝ) 
  (h1 : alpha_in_interval α)
  (h2 : equation m (Real.sin α))
  (h3 : equation m (Real.cos α)) :
  m = -10/9 ∧ 
  (Real.cos α + Real.sin α) * Real.tan α / (1 - Real.tan α^2) = 11 * Real.sqrt 47 / 564 :=
sorry

end NUMINAMATH_CALUDE_trigonometric_roots_theorem_l2843_284389


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l2843_284364

theorem existence_of_special_sequence :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j, i < j → a i < a j) ∧
    (∀ i : Fin 98, Nat.gcd (a i) (a (i + 1)) > Nat.gcd (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l2843_284364


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2843_284359

theorem fraction_sum_equality : (3 : ℚ) / 8 - 5 / 6 + 9 / 4 = 43 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2843_284359


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2843_284332

theorem toms_age_ratio (T N : ℝ) : 
  (T - N = 3 * (T - 4 * N)) → T / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2843_284332


namespace NUMINAMATH_CALUDE_inequality_one_min_value_min_point_l2843_284304

-- Define the variables and conditions
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 4)

-- Theorem 1
theorem inequality_one : 1/a + 1/(b+1) ≥ 4/5 := by sorry

-- Theorem 2
theorem min_value : ∃ (min_val : ℝ), min_val = (1 + Real.sqrt 5) / 2 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 4 → 4/(x*y) + x/y ≥ min_val := by sorry

-- Theorem for the values of a and b at the minimum point
theorem min_point : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧
  4/(a*b) + a/b = (1 + Real.sqrt 5) / 2 ∧
  a = Real.sqrt 5 - 1 ∧ b = 5 - Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_inequality_one_min_value_min_point_l2843_284304


namespace NUMINAMATH_CALUDE_garden_length_l2843_284356

/-- Given a rectangular garden with perimeter 1200 meters and breadth 240 meters, 
    prove that its length is 360 meters. -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ) : 
  perimeter = 1200 ∧ 
  breadth = 240 ∧ 
  perimeter = 2 * (length + breadth) →
  length = 360 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l2843_284356


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l2843_284384

/-- Given two cubes with side length b joined to form a cuboid, 
    the surface area of the resulting cuboid is 10b^2 -/
theorem cuboid_surface_area (b : ℝ) (h : b > 0) : 
  2 * (2*b*b + b*b + b*(2*b)) = 10 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l2843_284384


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l2843_284320

/-- The length of a rectangular metallic sheet that forms an open box with given dimensions and volume -/
theorem metallic_sheet_length : ∃ (L : ℝ),
  (L > 0) ∧ 
  (L - 2 * 8) * (36 - 2 * 8) * 8 = 5120 ∧ 
  L = 48 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l2843_284320


namespace NUMINAMATH_CALUDE_combined_collection_size_l2843_284393

/-- The number of books in Tim's collection -/
def tim_books : ℕ := 44

/-- The number of books in Sam's collection -/
def sam_books : ℕ := 52

/-- The number of books in Alex's collection -/
def alex_books : ℕ := 65

/-- The number of books in Katie's collection -/
def katie_books : ℕ := 37

/-- The total number of books in the combined collections -/
def total_books : ℕ := tim_books + sam_books + alex_books + katie_books

theorem combined_collection_size : total_books = 198 := by
  sorry

end NUMINAMATH_CALUDE_combined_collection_size_l2843_284393


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2843_284318

theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (x y : ℝ),
      (x + y = 2005 ∧
       x / 2005 + y / 2006 = 1 ∧
       x / 2006 + y / 2005 = 1) →
      area = 2005^2 / (2 * 4011)

-- The proof is omitted
theorem triangle_area_proof : triangle_area (2005^2 / (2 * 4011)) :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2843_284318


namespace NUMINAMATH_CALUDE_equation_roots_and_solution_l2843_284338

-- Define the equation
def equation (x p : ℝ) : Prop :=
  Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x

-- Theorem statement
theorem equation_roots_and_solution :
  ∀ p : ℝ, (∃ x : ℝ, equation x p) ↔ (0 ≤ p ∧ p ≤ 4/3) ∧
  ∀ p : ℝ, 0 ≤ p → p ≤ 4/3 → equation 1 p :=
sorry

end NUMINAMATH_CALUDE_equation_roots_and_solution_l2843_284338


namespace NUMINAMATH_CALUDE_square_area_11cm_l2843_284340

/-- The area of a square with side length 11 cm is 121 cm². -/
theorem square_area_11cm (side_length : ℝ) (h : side_length = 11) :
  side_length * side_length = 121 := by
  sorry

end NUMINAMATH_CALUDE_square_area_11cm_l2843_284340


namespace NUMINAMATH_CALUDE_expression_square_l2843_284342

theorem expression_square (x y : ℕ) 
  (h : (1 : ℚ) / x + 1 / y + 1 / (x * y) = 1 / (x + 4) + 1 / (y - 4) + 1 / ((x + 4) * (y - 4))) : 
  ∃ n : ℕ, x * y + 4 = n^2 := by
sorry

end NUMINAMATH_CALUDE_expression_square_l2843_284342


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2843_284317

/-- 
Given an isosceles triangle ABC where:
- Angle A is congruent to angle C
- The measure of angle B is 40 degrees less than twice the measure of angle A
Prove that the measure of angle B is 70 degrees
-/
theorem isosceles_triangle_angle_measure (A B C : ℝ) : 
  A = C →  -- Angle A is congruent to angle C
  B = 2 * A - 40 →  -- Measure of angle B is 40 degrees less than twice the measure of angle A
  A + B + C = 180 →  -- Sum of angles in a triangle is 180 degrees
  B = 70 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2843_284317


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2843_284366

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  (a * Real.cos B + b * Real.cos A) * Real.cos (2 * C) = c * Real.cos C →
  b = 2 * a →
  S = (Real.sqrt 3 / 2) * Real.sin A * Real.sin B →
  C = 2 * Real.pi / 3 ∧
  Real.sin A = Real.sqrt 21 / 14 ∧
  c = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2843_284366


namespace NUMINAMATH_CALUDE_marble_count_l2843_284308

-- Define the number of marbles for each person
def allison_marbles : ℕ := 28
def angela_marbles : ℕ := allison_marbles + 8
def albert_marbles : ℕ := 3 * angela_marbles
def addison_marbles : ℕ := 2 * albert_marbles

-- Define the total number of marbles
def total_marbles : ℕ := allison_marbles + angela_marbles + albert_marbles + addison_marbles

-- Theorem to prove
theorem marble_count : total_marbles = 388 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2843_284308


namespace NUMINAMATH_CALUDE_helicopter_rental_theorem_l2843_284378

/-- Calculates the total cost of renting a helicopter given the daily rental hours, number of days, and hourly rate. -/
def helicopter_rental_cost (hours_per_day : ℕ) (days : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_day * days * rate_per_hour

/-- Proves that renting a helicopter for 2 hours a day for 3 days at $75 per hour costs $450 in total. -/
theorem helicopter_rental_theorem : helicopter_rental_cost 2 3 75 = 450 := by
  sorry

end NUMINAMATH_CALUDE_helicopter_rental_theorem_l2843_284378


namespace NUMINAMATH_CALUDE_complement_of_M_l2843_284316

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

-- State the theorem
theorem complement_of_M (x : ℝ) : x ∈ (Set.univ \ M) ↔ x < 0 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l2843_284316


namespace NUMINAMATH_CALUDE_bing_position_guimao_in_cycle_l2843_284355

-- Define the cyclic arrangement
def heavenly_stems := 10
def earthly_branches := 12
def cycle_length := 60

-- Define the position of 丙 (bǐng)
def bing_first_appearance := 3

-- Define the function for the nth appearance of 丙 (bǐng)
def bing_column (n : ℕ) : ℕ := 10 * n - 7

-- Define the position of 癸卯 (guǐ mǎo) in the cycle
def guimao_position := 40

-- Theorem for the position of 丙 (bǐng)
theorem bing_position (n : ℕ) : 
  bing_column n ≡ bing_first_appearance [MOD cycle_length] :=
sorry

-- Theorem for the position of 癸卯 (guǐ mǎo)
theorem guimao_in_cycle : 
  guimao_position > 0 ∧ guimao_position ≤ cycle_length :=
sorry

end NUMINAMATH_CALUDE_bing_position_guimao_in_cycle_l2843_284355


namespace NUMINAMATH_CALUDE_point_trajectory_l2843_284313

/-- The trajectory of a point M(x,y) satisfying a specific distance condition -/
theorem point_trajectory (x y : ℝ) (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) (hx : x > 0) :
  x^2 / 16 - y^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_trajectory_l2843_284313


namespace NUMINAMATH_CALUDE_pyramid_volume_no_conditional_l2843_284375

/-- Algorithm to calculate triangle area from three side lengths -/
def triangle_area (a b c : ℝ) : ℝ := sorry

/-- Algorithm to calculate line slope from two points' coordinates -/
def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- Algorithm to calculate common logarithm of a number -/
noncomputable def common_log (x : ℝ) : ℝ := sorry

/-- Algorithm to calculate pyramid volume from base area and height -/
def pyramid_volume (base_area height : ℝ) : ℝ := sorry

/-- Predicate to check if an algorithm contains conditional statements -/
def has_conditional {α β : Type} (f : α → β) : Prop := sorry

theorem pyramid_volume_no_conditional :
  ¬ has_conditional pyramid_volume ∧
  has_conditional triangle_area ∧
  has_conditional line_slope ∧
  has_conditional common_log :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_no_conditional_l2843_284375


namespace NUMINAMATH_CALUDE_limit_f_derivative_at_one_l2843_284335

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x) * Real.exp x

theorem limit_f_derivative_at_one :
  (deriv f) 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_limit_f_derivative_at_one_l2843_284335


namespace NUMINAMATH_CALUDE_chloe_earnings_l2843_284383

/-- Chloe's earnings over two weeks -/
theorem chloe_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 26 →
  extra_earnings = 65.45 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1 : ℚ) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2 : ℚ) = 360 :=
by sorry

end NUMINAMATH_CALUDE_chloe_earnings_l2843_284383


namespace NUMINAMATH_CALUDE_lisa_photos_last_weekend_l2843_284333

/-- Calculates the number of photos Lisa took last weekend given the conditions --/
def photos_last_weekend (animal_photos : ℕ) (flower_multiplier : ℕ) (scenery_difference : ℕ) (weekend_difference : ℕ) : ℕ :=
  let flower_photos := animal_photos * flower_multiplier
  let scenery_photos := flower_photos - scenery_difference
  let total_photos := animal_photos + flower_photos + scenery_photos
  total_photos - weekend_difference

/-- Theorem stating that Lisa took 45 photos last weekend --/
theorem lisa_photos_last_weekend :
  photos_last_weekend 10 3 10 15 = 45 := by
  sorry

#eval photos_last_weekend 10 3 10 15

end NUMINAMATH_CALUDE_lisa_photos_last_weekend_l2843_284333


namespace NUMINAMATH_CALUDE_even_sum_necessary_not_sufficient_l2843_284395

theorem even_sum_necessary_not_sufficient :
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) ∧
  (∃ a b : ℤ, Even (a + b) ∧ ¬(Even a ∧ Even b)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_necessary_not_sufficient_l2843_284395


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l2843_284301

theorem square_difference_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 26) 
  (product_eq : x * y = 168) : 
  x^2 - y^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l2843_284301


namespace NUMINAMATH_CALUDE_elena_garden_petals_l2843_284300

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end NUMINAMATH_CALUDE_elena_garden_petals_l2843_284300


namespace NUMINAMATH_CALUDE_lowest_fraction_job_l2843_284377

/-- Given three people who can individually complete a job in 4, 6, and 8 hours respectively,
    the lowest fraction of the job that can be done in 1 hour by 2 of the people working together is 7/24. -/
theorem lowest_fraction_job (person_a person_b person_c : ℝ) 
    (ha : person_a = 1 / 4) (hb : person_b = 1 / 6) (hc : person_c = 1 / 8) : 
    min (person_a + person_b) (min (person_a + person_c) (person_b + person_c)) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_lowest_fraction_job_l2843_284377


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_sums_of_squares_l2843_284387

/-- A function that checks if a number is the sum of two squares --/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- The main theorem stating that there are infinitely many n satisfying the condition --/
theorem infinitely_many_consecutive_sums_of_squares :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ 
    isSumOfTwoSquares n ∧
    isSumOfTwoSquares (n + 1) ∧
    isSumOfTwoSquares (n + 2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_sums_of_squares_l2843_284387


namespace NUMINAMATH_CALUDE_intersection_M_N_l2843_284371

def M : Set ℝ := { x | -3 < x ∧ x < 1 }
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2843_284371


namespace NUMINAMATH_CALUDE_weight_sum_abby_damon_l2843_284345

/-- Given the weights of four people in pairs, prove that the sum of the weights of the first and fourth person is 300 pounds. -/
theorem weight_sum_abby_damon (a b c d : ℕ) : 
  a + b = 270 → 
  b + c = 250 → 
  c + d = 280 → 
  a + c = 300 → 
  a + d = 300 := by
sorry

end NUMINAMATH_CALUDE_weight_sum_abby_damon_l2843_284345


namespace NUMINAMATH_CALUDE_factorization_equality_l2843_284354

theorem factorization_equality (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2843_284354


namespace NUMINAMATH_CALUDE_extreme_values_and_roots_l2843_284349

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_roots (a b c : ℝ) :
  (∀ x : ℝ, f' a b x = 0 ↔ x = 1 ∨ x = 3) →
  (a = -6 ∧ b = 9) ∧
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f (-6) 9 c x = 0 ∧ f (-6) 9 c y = 0 ∧ f (-6) 9 c z = 0) →
  -4 < c ∧ c < 0 :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_roots_l2843_284349


namespace NUMINAMATH_CALUDE_factorization_x4_minus_64_l2843_284321

theorem factorization_x4_minus_64 (x : ℝ) : 
  x^4 - 64 = (x^2 + 8) * (x + 2 * Real.sqrt 2) * (x - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_64_l2843_284321


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2843_284346

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (0 : ℤ) ≤ x → x^2 < 2*x + 1 → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2843_284346


namespace NUMINAMATH_CALUDE_remainder_of_nested_star_l2843_284326

-- Define the star operation
def star (a b : ℕ) : ℕ := a * b - 2

-- Define a function to represent the nested star operations
def nested_star : ℕ → ℕ
| 0 => 9
| n + 1 => star (579 - 10 * n) (nested_star n)

-- Theorem statement
theorem remainder_of_nested_star :
  nested_star 57 % 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_nested_star_l2843_284326


namespace NUMINAMATH_CALUDE_roast_cost_is_17_l2843_284350

/-- Calculates the cost of a roast given initial money, vegetable cost, and remaining money --/
def roast_cost (initial_money : ℤ) (vegetable_cost : ℤ) (remaining_money : ℤ) : ℤ :=
  initial_money - vegetable_cost - remaining_money

/-- Proves that the roast cost €17 given the problem conditions --/
theorem roast_cost_is_17 :
  roast_cost 100 11 72 = 17 := by
  sorry

end NUMINAMATH_CALUDE_roast_cost_is_17_l2843_284350


namespace NUMINAMATH_CALUDE_nina_running_distance_l2843_284331

theorem nina_running_distance : 0.08 + 0.08 + 0.67 = 0.83 := by sorry

end NUMINAMATH_CALUDE_nina_running_distance_l2843_284331


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2843_284372

def club_size (initial_members : ℕ) (years : ℕ) : ℕ :=
  let active_members := initial_members - 3
  let growth_factor := 4
  (growth_factor ^ years) * active_members + 3

theorem club_size_after_four_years :
  club_size 21 4 = 4611 := by sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l2843_284372


namespace NUMINAMATH_CALUDE_max_dot_product_l2843_284306

/-- The ellipse with equation x^2/4 + y^2/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The dot product of vectors OP and FP -/
def dotProduct (P : ℝ × ℝ) : ℝ :=
  (P.1 * (P.1 + 1)) + (P.2 * P.2)

theorem max_dot_product :
  ∃ (M : ℝ), M = 6 ∧ ∀ P ∈ Ellipse, dotProduct P ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l2843_284306


namespace NUMINAMATH_CALUDE_baron_munchausen_crowd_size_l2843_284324

theorem baron_munchausen_crowd_size :
  ∃ (n : ℕ), n > 0 ∧
  (n / 2 + n / 3 + n / 5 ≤ n + 1) ∧
  (∀ m : ℕ, m > n → m / 2 + m / 3 + m / 5 > m + 1) ∧
  n = 37 := by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_crowd_size_l2843_284324


namespace NUMINAMATH_CALUDE_composite_polynomial_l2843_284363

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 6*n^2 + 12*n + 16 = a * b :=
by sorry

end NUMINAMATH_CALUDE_composite_polynomial_l2843_284363


namespace NUMINAMATH_CALUDE_lulu_piggy_bank_l2843_284336

theorem lulu_piggy_bank (initial_amount : ℝ) : 
  (4/5 * (1/2 * (initial_amount - 5))) = 24 → initial_amount = 65 := by
  sorry

end NUMINAMATH_CALUDE_lulu_piggy_bank_l2843_284336


namespace NUMINAMATH_CALUDE_base_76_minus_b_multiple_of_17_l2843_284303

/-- The value of 528376415 in base 76 -/
def base_76_number : ℕ := 5 + 1*76 + 4*(76^2) + 6*(76^3) + 7*(76^4) + 3*(76^5) + 8*(76^6) + 2*(76^7) + 5*(76^8)

theorem base_76_minus_b_multiple_of_17 (b : ℤ) 
  (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : ∃ k : ℤ, base_76_number - b = 17 * k) :
  b = 0 ∨ b = 17 := by
sorry

end NUMINAMATH_CALUDE_base_76_minus_b_multiple_of_17_l2843_284303


namespace NUMINAMATH_CALUDE_water_missing_calculation_l2843_284382

/-- Calculates the amount of water missing from a tank's maximum capacity after a series of leaks and refilling. -/
def water_missing (initial_capacity : ℕ) (leak_rate1 leak_duration1 : ℕ) (leak_rate2 leak_duration2 : ℕ) (fill_rate fill_duration : ℕ) : ℕ :=
  let total_leak := leak_rate1 * leak_duration1 + leak_rate2 * leak_duration2
  let remaining_water := initial_capacity - total_leak
  let filled_water := fill_rate * fill_duration
  let final_water := remaining_water + filled_water
  initial_capacity - final_water

/-- Theorem stating that the amount of water missing from the tank's maximum capacity is 140,000 gallons. -/
theorem water_missing_calculation :
  water_missing 350000 32000 5 10000 10 40000 3 = 140000 := by
  sorry

end NUMINAMATH_CALUDE_water_missing_calculation_l2843_284382


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_l2843_284344

/-- Calculate the total cost of items after applying discounts --/
theorem total_cost_after_discounts :
  let board_game_cost : ℚ := 2
  let action_figure_cost : ℚ := 7
  let action_figure_count : ℕ := 4
  let puzzle_cost : ℚ := 6
  let deck_cost : ℚ := 3.5
  let toy_car_cost : ℚ := 4
  let toy_car_count : ℕ := 2
  let action_figure_discount : ℚ := 0.15
  let puzzle_toy_car_discount : ℚ := 0.10
  let deck_discount : ℚ := 0.05

  let total_cost : ℚ := 
    board_game_cost +
    (action_figure_cost * action_figure_count) * (1 - action_figure_discount) +
    puzzle_cost * (1 - puzzle_toy_car_discount) +
    deck_cost * (1 - deck_discount) +
    (toy_car_cost * toy_car_count) * (1 - puzzle_toy_car_discount)

  total_cost = 41.73 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_l2843_284344


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2843_284369

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = -16 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l2843_284369


namespace NUMINAMATH_CALUDE_room_length_is_19_l2843_284381

/-- Represents the dimensions of a rectangular room with a surrounding veranda. -/
structure RoomWithVeranda where
  roomLength : ℝ
  roomWidth : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda given the room dimensions. -/
def verandaArea (room : RoomWithVeranda) : ℝ :=
  (room.roomLength + 2 * room.verandaWidth) * (room.roomWidth + 2 * room.verandaWidth) -
  room.roomLength * room.roomWidth

/-- Theorem: The length of the room is 19 meters given the specified conditions. -/
theorem room_length_is_19 (room : RoomWithVeranda)
  (h1 : room.roomWidth = 12)
  (h2 : room.verandaWidth = 2)
  (h3 : verandaArea room = 140) :
  room.roomLength = 19 := by
  sorry

end NUMINAMATH_CALUDE_room_length_is_19_l2843_284381


namespace NUMINAMATH_CALUDE_fraction_addition_l2843_284367

theorem fraction_addition : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2843_284367


namespace NUMINAMATH_CALUDE_turtle_race_times_l2843_284362

/-- The time it took for Greta's turtle to finish the race -/
def greta_time : ℕ := sorry

/-- The time it took for George's turtle to finish the race -/
def george_time : ℕ := sorry

/-- The time it took for Gloria's turtle to finish the race -/
def gloria_time : ℕ := 8

theorem turtle_race_times :
  (george_time = greta_time - 2) ∧
  (gloria_time = 2 * george_time) ∧
  (greta_time = 6) := by sorry

end NUMINAMATH_CALUDE_turtle_race_times_l2843_284362


namespace NUMINAMATH_CALUDE_hiker_distance_at_blast_l2843_284398

/-- The time in seconds for which the timer is set -/
def timer_duration : ℝ := 45

/-- The speed of the hiker in yards per second -/
def hiker_speed : ℝ := 6

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1200

/-- The distance the hiker has traveled at time t -/
def hiker_distance (t : ℝ) : ℝ := hiker_speed * t * 3

/-- The distance the sound has traveled at time t (t ≥ timer_duration) -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - timer_duration)

/-- The time at which the hiker hears the blast -/
noncomputable def blast_time : ℝ := 
  (sound_speed * timer_duration) / (sound_speed - hiker_speed * 3)

/-- The theorem stating that the hiker's distance when they hear the blast is approximately 275 yards -/
theorem hiker_distance_at_blast : 
  ∃ ε > 0, abs (hiker_distance blast_time / 3 - 275) < ε :=
sorry

end NUMINAMATH_CALUDE_hiker_distance_at_blast_l2843_284398


namespace NUMINAMATH_CALUDE_flight_cost_X_to_Y_l2843_284376

/-- The cost to fly between two cities given the distance and cost parameters. -/
def flight_cost (distance : ℝ) (cost_per_km : ℝ) (booking_fee : ℝ) : ℝ :=
  distance * cost_per_km + booking_fee

/-- Theorem stating that the flight cost from X to Y is $660. -/
theorem flight_cost_X_to_Y :
  flight_cost 4500 0.12 120 = 660 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_X_to_Y_l2843_284376


namespace NUMINAMATH_CALUDE_polynomial_identity_l2843_284330

theorem polynomial_identity (a b c d : ℝ) :
  (∀ x y : ℝ, (10*x + 6*y)^3 = a*x^3 + b*x^2*y + c*x*y^2 + d*y^3) →
  -a + 2*b - 4*c + 8*d = 8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2843_284330


namespace NUMINAMATH_CALUDE_some_number_solution_l2843_284312

theorem some_number_solution : 
  ∃ x : ℚ, (1 / 2 : ℚ) + ((2 / 3 : ℚ) * (3 / 8 : ℚ) + 4) - x = (17 / 4 : ℚ) ∧ x = (1 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_some_number_solution_l2843_284312


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l2843_284373

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = Set.Iio (-1/3) ∪ Set.Ioi 3 := by sorry

-- Theorem for part II
theorem range_of_m (h : ∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) :
  m ∈ Set.Ioo (-1/2) (5/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l2843_284373


namespace NUMINAMATH_CALUDE_sam_sandwich_count_l2843_284358

/-- The number of sandwiches Sam eats per day -/
def sandwiches_per_day : ℕ := sorry

/-- The ratio of apples to sandwiches -/
def apples_per_sandwich : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of apples eaten in a week -/
def total_apples : ℕ := 280

theorem sam_sandwich_count :
  sandwiches_per_day = 10 ∧
  sandwiches_per_day * apples_per_sandwich * days_in_week = total_apples :=
sorry

end NUMINAMATH_CALUDE_sam_sandwich_count_l2843_284358


namespace NUMINAMATH_CALUDE_sam_watermelons_l2843_284351

def total_watermelons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

theorem sam_watermelons : 
  let initial := 4
  let additional := 3
  total_watermelons initial additional = 7 := by
  sorry

end NUMINAMATH_CALUDE_sam_watermelons_l2843_284351


namespace NUMINAMATH_CALUDE_line_direction_vector_l2843_284360

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := (2, -1) + t • d
  let d : ℝ × ℝ := direction_vector line
  let y (x : ℝ) : ℝ := (2 * x + 3) / 5
  (∀ x ≥ 2, (x - 2) ^ 2 + (y x + 1) ^ 2 = t ^ 2 → 
    line t = (x, y x)) →
  d = (5 / Real.sqrt 29, 2 / Real.sqrt 29) :=
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2843_284360


namespace NUMINAMATH_CALUDE_book_price_theorem_l2843_284374

/-- The price of a book on Monday when prices are 10% more than normal -/
def monday_price : ℚ := 5.50

/-- The normal price increase factor on Monday -/
def monday_factor : ℚ := 1.10

/-- The normal price decrease factor on Friday -/
def friday_factor : ℚ := 0.90

/-- The price of the book on Friday -/
def friday_price : ℚ := monday_price / monday_factor * friday_factor

theorem book_price_theorem :
  friday_price = 4.50 := by sorry

end NUMINAMATH_CALUDE_book_price_theorem_l2843_284374


namespace NUMINAMATH_CALUDE_train_platform_lengths_l2843_284311

/-- Two trains with different constant velocities passing platforms -/
theorem train_platform_lengths 
  (V1 V2 L1 L2 T1 T2 : ℝ) 
  (h_diff_vel : V1 ≠ V2) 
  (h_pos_V1 : V1 > 0) 
  (h_pos_V2 : V2 > 0)
  (h_pos_T1 : T1 > 0)
  (h_pos_T2 : T2 > 0)
  (h_L1 : L1 = V1 * T1)
  (h_L2 : L2 = V2 * T2) :
  ∃ (P1 P2 : ℝ), 
    P1 = 3 * V1 * T1 ∧ 
    P2 = 2 * V2 * T2 ∧
    V1 * (4 * T1) = L1 + P1 ∧
    V2 * (3 * T2) = L2 + P2 := by
  sorry


end NUMINAMATH_CALUDE_train_platform_lengths_l2843_284311
