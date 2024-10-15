import Mathlib

namespace NUMINAMATH_CALUDE_chess_team_photo_arrangements_l3438_343815

def chess_team_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  2 * (Nat.factorial num_boys) * (Nat.factorial num_girls)

theorem chess_team_photo_arrangements :
  chess_team_arrangements 3 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_photo_arrangements_l3438_343815


namespace NUMINAMATH_CALUDE_comet_watch_percentage_l3438_343875

-- Define the total time spent on activities in minutes
def total_time : ℕ := 655

-- Define the time spent watching the comet in minutes
def comet_watch_time : ℕ := 20

-- Function to calculate percentage
def calculate_percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Function to round to nearest integer
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Theorem statement
theorem comet_watch_percentage :
  round_to_nearest (calculate_percentage comet_watch_time total_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_comet_watch_percentage_l3438_343875


namespace NUMINAMATH_CALUDE_johnsonville_marching_band_max_members_l3438_343845

theorem johnsonville_marching_band_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 15 * n = 30 * k + 6) →
  15 * n < 900 →
  (∀ m : ℕ, (∃ j : ℕ, 15 * m = 30 * j + 6) → 15 * m < 900 → 15 * m ≤ 15 * n) →
  15 * n = 810 :=
by sorry

end NUMINAMATH_CALUDE_johnsonville_marching_band_max_members_l3438_343845


namespace NUMINAMATH_CALUDE_function_expression_l3438_343844

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = x^2 - x + 1) :
  ∀ x, f x = x^2 - 5*x + 7 := by
sorry

end NUMINAMATH_CALUDE_function_expression_l3438_343844


namespace NUMINAMATH_CALUDE_watermelons_last_two_weeks_l3438_343872

/-- Represents the number of watermelons Jeremy eats in a given week -/
def jeremyEats (week : ℕ) : ℕ :=
  match week % 3 with
  | 0 => 3
  | 1 => 4
  | _ => 5

/-- Represents the number of watermelons Jeremy gives to his dad in a given week -/
def dadReceives (week : ℕ) : ℕ := week + 1

/-- Represents the number of watermelons Jeremy gives to his sister in a given week -/
def sisterReceives (week : ℕ) : ℕ := 2 * week - 1

/-- Represents the number of watermelons Jeremy gives to his neighbor in a given week -/
def neighborReceives (week : ℕ) : ℕ := max (2 - week) 0

/-- Represents the total number of watermelons consumed in a given week -/
def totalConsumed (week : ℕ) : ℕ :=
  jeremyEats week + dadReceives week + sisterReceives week + neighborReceives week

/-- The initial number of watermelons -/
def initialWatermelons : ℕ := 30

/-- Theorem stating that the watermelons will last for 2 complete weeks -/
theorem watermelons_last_two_weeks :
  initialWatermelons ≥ totalConsumed 1 + totalConsumed 2 ∧
  initialWatermelons < totalConsumed 1 + totalConsumed 2 + totalConsumed 3 :=
sorry

end NUMINAMATH_CALUDE_watermelons_last_two_weeks_l3438_343872


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3438_343883

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (sum_prod : x + y = 6 * x * y) (double : y = 2 * x) :
  1 / x + 1 / y = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3438_343883


namespace NUMINAMATH_CALUDE_inequality_proof_l3438_343838

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3438_343838


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3438_343874

theorem no_function_satisfies_inequality :
  ¬∃ (f : ℝ → ℝ), (∀ x > 0, f x > 0) ∧
    (∀ x y, x > 0 → y > 0 → f x ^ 2 ≥ f (x + y) * (f x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3438_343874


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_two_l3438_343894

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_equals_negative_two :
  log10 (1/4) - log10 25 = -2 := by sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_two_l3438_343894


namespace NUMINAMATH_CALUDE_zeros_in_fraction_l3438_343800

-- Define the fraction
def fraction : ℚ := 18 / 50000

-- Define the function to count zeros after the decimal point
def count_zeros_after_decimal (q : ℚ) : ℕ := sorry

-- Theorem statement
theorem zeros_in_fraction :
  count_zeros_after_decimal fraction = 3 := by sorry

end NUMINAMATH_CALUDE_zeros_in_fraction_l3438_343800


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3438_343892

theorem complex_equation_solution (a b : ℝ) (z : ℂ) 
  (hz : z = Complex.mk a b) 
  (heq : Complex.I / z = Complex.mk 2 (-1)) : 
  a - b = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3438_343892


namespace NUMINAMATH_CALUDE_division_problem_l3438_343888

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 690 →
  divisor = 36 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3438_343888


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3438_343849

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80)
  : a = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3438_343849


namespace NUMINAMATH_CALUDE_triangle_ef_length_l3438_343873

/-- Given a triangle DEF with the specified conditions, prove that EF = 3 -/
theorem triangle_ef_length (D E F : ℝ) (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2) (h2 : DE = 6) : EF = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ef_length_l3438_343873


namespace NUMINAMATH_CALUDE_hurleys_age_l3438_343833

theorem hurleys_age (hurley_age richard_age : ℕ) : 
  richard_age - hurley_age = 20 →
  (richard_age + 40) + (hurley_age + 40) = 128 →
  hurley_age = 14 := by
sorry

end NUMINAMATH_CALUDE_hurleys_age_l3438_343833


namespace NUMINAMATH_CALUDE_polynomial_sum_l3438_343832

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ - a₃ + a₂ - a₁ + a₀ = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3438_343832


namespace NUMINAMATH_CALUDE_range_of_a_l3438_343826

def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∨ q a) ↔ a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3438_343826


namespace NUMINAMATH_CALUDE_division_problem_l3438_343813

theorem division_problem : ∃ (a b c d : Nat), 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  19858 / 102 = 1000 * a + 100 * b + 10 * c + d ∧
  19858 % 102 = 0 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l3438_343813


namespace NUMINAMATH_CALUDE_income_calculation_l3438_343850

/-- Represents a person's financial situation -/
structure FinancialSituation where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Theorem: Given a person's financial situation where the income to expenditure ratio
    is 3:2 and savings are 7000, the income is 21000 -/
theorem income_calculation (fs : FinancialSituation) 
  (h1 : fs.income = 3 * (fs.expenditure / 2))
  (h2 : fs.savings = 7000)
  (h3 : fs.income = fs.expenditure + fs.savings) : 
  fs.income = 21000 := by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l3438_343850


namespace NUMINAMATH_CALUDE_difference_of_fractions_l3438_343827

theorem difference_of_fractions : (7 / 8 : ℚ) * 320 - (11 / 16 : ℚ) * 144 = 181 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l3438_343827


namespace NUMINAMATH_CALUDE_perfect_squares_product_sum_l3438_343803

theorem perfect_squares_product_sum (a b : ℕ) : 
  (∃ x : ℕ, a = x^2) → 
  (∃ y : ℕ, b = y^2) → 
  a * b = a + b + 4844 →
  (Real.sqrt a + 1) * (Real.sqrt b + 1) * (Real.sqrt a - 1) * (Real.sqrt b - 1) - 
  (Real.sqrt 68 + 1) * (Real.sqrt 63 + 1) * (Real.sqrt 68 - 1) * (Real.sqrt 63 - 1) = 691 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_product_sum_l3438_343803


namespace NUMINAMATH_CALUDE_pencil_distribution_l3438_343889

theorem pencil_distribution (total_pencils : ℕ) (num_students : ℕ) (pencils_per_student : ℕ) :
  total_pencils = 125 →
  num_students = 25 →
  pencils_per_student * num_students = total_pencils →
  pencils_per_student = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3438_343889


namespace NUMINAMATH_CALUDE_product_of_roots_l3438_343847

theorem product_of_roots (x₁ x₂ k m : ℝ) : 
  x₁ ≠ x₂ →
  5 * x₁^2 - k * x₁ = m →
  5 * x₂^2 - k * x₂ = m →
  x₁ * x₂ = -m / 5 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3438_343847


namespace NUMINAMATH_CALUDE_product_expansion_l3438_343862

theorem product_expansion {R : Type*} [CommRing R] (x : R) :
  (3 * x + 4) * (2 * x^2 + x + 6) = 6 * x^3 + 11 * x^2 + 22 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3438_343862


namespace NUMINAMATH_CALUDE_function_equality_l3438_343871

theorem function_equality (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 14 * x + 7) →
  (∀ x : ℝ, f x = x^2 + 5 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3438_343871


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3438_343802

theorem complex_equation_solution (z : ℂ) :
  (z - 2*I) * (2 - I) = 5 → z = 2 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3438_343802


namespace NUMINAMATH_CALUDE_profit_margin_increase_l3438_343835

theorem profit_margin_increase (P S : ℝ) (r : ℝ) : 
  P > 0 → S > P →
  (S - P) / P * 100 = r →
  (S - 0.92 * P) / (0.92 * P) * 100 = r + 10 →
  r = 15 := by
sorry

end NUMINAMATH_CALUDE_profit_margin_increase_l3438_343835


namespace NUMINAMATH_CALUDE_meal_cost_calculation_l3438_343809

/-- Proves that given a meal with specific tax and tip rates, and a total cost,
    the original meal cost can be determined. -/
theorem meal_cost_calculation (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
    (h_total : total_cost = 36.90)
    (h_tax : tax_rate = 0.09)
    (h_tip : tip_rate = 0.18) :
    ∃ (original_cost : ℝ), 
      original_cost * (1 + tax_rate + tip_rate) = total_cost ∧ 
      original_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_calculation_l3438_343809


namespace NUMINAMATH_CALUDE_remainder_sum_l3438_343821

theorem remainder_sum (a b : ℤ) (ha : a % 70 = 64) (hb : b % 105 = 99) :
  (a + b) % 35 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3438_343821


namespace NUMINAMATH_CALUDE_count_valid_triangles_l3438_343899

/-- A triangle with integral side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of all valid IntTriangles --/
def validTriangles : Finset IntTriangle := sorry

theorem count_valid_triangles : Finset.card validTriangles = 6 := by sorry

end NUMINAMATH_CALUDE_count_valid_triangles_l3438_343899


namespace NUMINAMATH_CALUDE_tom_height_l3438_343814

theorem tom_height (t m : ℝ) : 
  t = 0.75 * m →                     -- Tom was 25% shorter than Mary two years ago
  m + 4 = 1.2 * (1.2 * t) →          -- Mary is now 20% taller than Tom after both have grown
  1.2 * t = 45 :=                    -- Tom's current height is 45 inches
by sorry

end NUMINAMATH_CALUDE_tom_height_l3438_343814


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3438_343818

theorem arithmetic_sequence_ninth_term
  (a : ℝ) (d : ℝ) -- first term and common difference
  (h1 : a + 2 * d = 23) -- third term is 23
  (h2 : a + 5 * d = 29) -- sixth term is 29
  : a + 8 * d = 35 := -- ninth term is 35
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3438_343818


namespace NUMINAMATH_CALUDE_pencil_cost_l3438_343884

theorem pencil_cost (pen_price pencil_price : ℚ) 
  (eq1 : 5 * pen_price + 4 * pencil_price = 310)
  (eq2 : 3 * pen_price + 6 * pencil_price = 238) :
  pencil_price = 130 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3438_343884


namespace NUMINAMATH_CALUDE_max_similar_triangle_lines_l3438_343886

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A line in a 2D plane --/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is outside a triangle --/
def IsOutside (P : Point) (T : Triangle) : Prop := sorry

/-- Predicate to check if a line passes through a point --/
def PassesThrough (L : Line) (P : Point) : Prop := sorry

/-- Predicate to check if a line cuts off a similar triangle --/
def CutsSimilarTriangle (L : Line) (T : Triangle) : Prop := sorry

/-- The main theorem --/
theorem max_similar_triangle_lines 
  (T : Triangle) (P : Point) (h : IsOutside P T) :
  ∃ (S : Finset Line), 
    (∀ L ∈ S, PassesThrough L P ∧ CutsSimilarTriangle L T) ∧ 
    S.card = 6 ∧
    (∀ S' : Finset Line, 
      (∀ L ∈ S', PassesThrough L P ∧ CutsSimilarTriangle L T) → 
      S'.card ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_max_similar_triangle_lines_l3438_343886


namespace NUMINAMATH_CALUDE_dans_song_book_cost_l3438_343876

/-- The cost of Dan's song book is equal to the total amount spent at the music store
    minus the cost of the clarinet. -/
theorem dans_song_book_cost (clarinet_cost total_spent : ℚ) 
  (h1 : clarinet_cost = 130.30)
  (h2 : total_spent = 141.54) :
  total_spent - clarinet_cost = 11.24 := by
  sorry

end NUMINAMATH_CALUDE_dans_song_book_cost_l3438_343876


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_f_max_on_interval_f_min_on_interval_l3438_343816

-- Define the function f(x) = -x^2 + 2x
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem for monotonicity
theorem f_increasing_on_interval : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ < f x₂ := by sorry

-- Theorem for maximum value
theorem f_max_on_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ f x = 1 ∧ ∀ y ∈ Set.Icc 0 5, f y ≤ f x := by sorry

-- Theorem for minimum value
theorem f_min_on_interval : 
  ∃ x : ℝ, x ∈ Set.Icc 0 5 ∧ f x = -15 ∧ ∀ y ∈ Set.Icc 0 5, f y ≥ f x := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_f_max_on_interval_f_min_on_interval_l3438_343816


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3438_343859

theorem floor_ceiling_sum : ⌊(-3.87 : ℝ)⌋ + ⌈(30.75 : ℝ)⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3438_343859


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l3438_343806

theorem largest_of_seven_consecutive_integers (n : ℕ) :
  (n > 0) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) →
  (n + 6 = 433) :=
by sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l3438_343806


namespace NUMINAMATH_CALUDE_complex_squared_norm_l3438_343810

theorem complex_squared_norm (w : ℂ) (h : w^2 + Complex.abs w^2 = 7 + 2*I) : 
  Complex.abs w^2 = 53/14 := by
  sorry

end NUMINAMATH_CALUDE_complex_squared_norm_l3438_343810


namespace NUMINAMATH_CALUDE_product_greater_than_sum_implies_sum_greater_than_four_l3438_343843

theorem product_greater_than_sum_implies_sum_greater_than_four (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_product : x * y > x + y) : x + y > 4 := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_implies_sum_greater_than_four_l3438_343843


namespace NUMINAMATH_CALUDE_number_of_students_l3438_343890

/-- Given an initial average of 100, and a correction of one student's mark from 60 to 10
    resulting in a new average of 98, prove that the number of students in the class is 25. -/
theorem number_of_students (initial_average : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (new_average : ℝ)
  (h1 : initial_average = 100)
  (h2 : wrong_mark = 60)
  (h3 : correct_mark = 10)
  (h4 : new_average = 98) :
  ∃ n : ℕ, n * new_average = n * initial_average - (wrong_mark - correct_mark) ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l3438_343890


namespace NUMINAMATH_CALUDE_arithmetic_progression_formula_recursive_formula_initial_condition_l3438_343841

/-- Arithmetic progression with first term 13.5 and common difference 4.2 -/
def arithmetic_progression (n : ℕ) : ℝ :=
  13.5 + (n - 1 : ℝ) * 4.2

/-- The nth term of the arithmetic progression -/
def nth_term (n : ℕ) : ℝ :=
  4.2 * n + 9.3

theorem arithmetic_progression_formula (n : ℕ) :
  arithmetic_progression n = nth_term n := by sorry

theorem recursive_formula (n : ℕ) (h : n > 0) :
  arithmetic_progression (n + 1) = arithmetic_progression n + 4.2 := by sorry

theorem initial_condition :
  arithmetic_progression 1 = 13.5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_formula_recursive_formula_initial_condition_l3438_343841


namespace NUMINAMATH_CALUDE_vermont_ads_l3438_343834

def ads_problem (first_page_ads : ℕ) (total_pages : ℕ) (click_fraction : ℚ) : Prop :=
  let second_page_ads := 2 * first_page_ads
  let third_page_ads := second_page_ads + 24
  let fourth_page_ads := (3 : ℚ) / 4 * second_page_ads
  let total_ads := first_page_ads + second_page_ads + third_page_ads + fourth_page_ads
  let clicked_ads := click_fraction * total_ads
  
  first_page_ads = 12 ∧
  total_pages = 4 ∧
  click_fraction = 2 / 3 ∧
  clicked_ads = 68

theorem vermont_ads : ads_problem 12 4 (2/3) := by sorry

end NUMINAMATH_CALUDE_vermont_ads_l3438_343834


namespace NUMINAMATH_CALUDE_cargo_realization_time_l3438_343880

/-- Represents the speed of a boat in still water -/
structure BoatSpeed where
  speed : ℝ
  positive : speed > 0

/-- Represents the current speed of the river -/
structure RiverCurrent where
  speed : ℝ

/-- Represents a boat on the river -/
structure Boat where
  speed : BoatSpeed
  position : ℝ
  direction : Bool  -- True for downstream, False for upstream

/-- The time it takes for Boat 1 to realize its cargo is missing -/
def timeToCargo (boat1 : Boat) (boat2 : Boat) (river : RiverCurrent) : ℝ :=
  sorry

/-- Theorem stating that the time taken for Boat 1 to realize its cargo is missing is 40 minutes -/
theorem cargo_realization_time
  (boat1 : Boat)
  (boat2 : Boat)
  (river : RiverCurrent)
  (h1 : boat1.speed.speed = 2 * boat2.speed.speed)
  (h2 : boat1.direction = false)  -- Boat 1 starts upstream
  (h3 : boat2.direction = true)   -- Boat 2 starts downstream
  (h4 : ∃ (t : ℝ), t > 0 ∧ t < timeToCargo boat1 boat2 river ∧ 
        boat1.position + t * (boat1.speed.speed + river.speed) = 
        boat2.position + t * (boat2.speed.speed - river.speed))  -- Boats meet before cargo realization
  (h5 : ∃ (t : ℝ), t = 20 ∧ 
        boat1.position + t * (boat1.speed.speed + river.speed) = 
        boat2.position + t * (boat2.speed.speed - river.speed))  -- Boats meet at 20 minutes
  : timeToCargo boat1 boat2 river = 40 := by
  sorry

end NUMINAMATH_CALUDE_cargo_realization_time_l3438_343880


namespace NUMINAMATH_CALUDE_gcd_bound_for_special_lcm_l3438_343819

theorem gcd_bound_for_special_lcm (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) → 
  (10^6 ≤ b ∧ b < 10^7) → 
  (10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) → 
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_bound_for_special_lcm_l3438_343819


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l3438_343823

/-- A parabola with directrix y = -4 has the standard equation x² = 16y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y = -4 → (x^2 = 2*p*y ↔ x^2 = 16*y)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l3438_343823


namespace NUMINAMATH_CALUDE_solution_set_l3438_343848

theorem solution_set (y : ℝ) : 2 ≤ y / (3 * y - 4) ∧ y / (3 * y - 4) < 5 ↔ 10 / 7 < y ∧ y ≤ 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3438_343848


namespace NUMINAMATH_CALUDE_least_pennies_count_eleven_satisfies_conditions_least_pennies_is_eleven_l3438_343840

theorem least_pennies_count (n : ℕ) : n > 0 ∧ n % 5 = 1 ∧ n % 3 = 2 → n ≥ 11 :=
by sorry

theorem eleven_satisfies_conditions : 11 % 5 = 1 ∧ 11 % 3 = 2 :=
by sorry

theorem least_pennies_is_eleven : ∃ (n : ℕ), n > 0 ∧ n % 5 = 1 ∧ n % 3 = 2 ∧ ∀ m : ℕ, (m > 0 ∧ m % 5 = 1 ∧ m % 3 = 2) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_pennies_count_eleven_satisfies_conditions_least_pennies_is_eleven_l3438_343840


namespace NUMINAMATH_CALUDE_second_divisor_exists_l3438_343895

theorem second_divisor_exists : ∃ (x y : ℕ), 0 < y ∧ y < 61 ∧ x % 61 = 24 ∧ x % y = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_exists_l3438_343895


namespace NUMINAMATH_CALUDE_coefficient_of_expression_l3438_343830

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (expression : ℚ) : ℚ := sorry

/-- The expression -2ab/3 -/
def expression : ℚ := -2 / 3

theorem coefficient_of_expression :
  coefficient expression = -2 / 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_expression_l3438_343830


namespace NUMINAMATH_CALUDE_inequality_proof_l3438_343846

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a * b * c ≤ (a + b) * (b + c) * (c + a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3438_343846


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3438_343837

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k - 3 = 0 ∧ (∀ y : ℝ, y^2 + k - 3 = 0 → y = x)) ↔ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3438_343837


namespace NUMINAMATH_CALUDE_four_numbers_theorem_l3438_343866

theorem four_numbers_theorem (a b c d : ℕ) : 
  a + b + c = 17 ∧ 
  a + b + d = 21 ∧ 
  a + c + d = 25 ∧ 
  b + c + d = 30 → 
  (a = 14 ∧ b = 10 ∧ c = 6 ∧ d = 1) ∨
  (a = 14 ∧ b = 10 ∧ c = 1 ∧ d = 6) ∨
  (a = 14 ∧ b = 6 ∧ c = 10 ∧ d = 1) ∨
  (a = 14 ∧ b = 6 ∧ c = 1 ∧ d = 10) ∨
  (a = 14 ∧ b = 1 ∧ c = 10 ∧ d = 6) ∨
  (a = 14 ∧ b = 1 ∧ c = 6 ∧ d = 10) ∨
  (a = 10 ∧ b = 14 ∧ c = 6 ∧ d = 1) ∨
  (a = 10 ∧ b = 14 ∧ c = 1 ∧ d = 6) ∨
  (a = 10 ∧ b = 6 ∧ c = 14 ∧ d = 1) ∨
  (a = 10 ∧ b = 6 ∧ c = 1 ∧ d = 14) ∨
  (a = 10 ∧ b = 1 ∧ c = 14 ∧ d = 6) ∨
  (a = 10 ∧ b = 1 ∧ c = 6 ∧ d = 14) ∨
  (a = 6 ∧ b = 14 ∧ c = 10 ∧ d = 1) ∨
  (a = 6 ∧ b = 14 ∧ c = 1 ∧ d = 10) ∨
  (a = 6 ∧ b = 10 ∧ c = 14 ∧ d = 1) ∨
  (a = 6 ∧ b = 10 ∧ c = 1 ∧ d = 14) ∨
  (a = 6 ∧ b = 1 ∧ c = 14 ∧ d = 10) ∨
  (a = 6 ∧ b = 1 ∧ c = 10 ∧ d = 14) ∨
  (a = 1 ∧ b = 14 ∧ c = 10 ∧ d = 6) ∨
  (a = 1 ∧ b = 14 ∧ c = 6 ∧ d = 10) ∨
  (a = 1 ∧ b = 10 ∧ c = 14 ∧ d = 6) ∨
  (a = 1 ∧ b = 10 ∧ c = 6 ∧ d = 14) ∨
  (a = 1 ∧ b = 6 ∧ c = 14 ∧ d = 10) ∨
  (a = 1 ∧ b = 6 ∧ c = 10 ∧ d = 14) :=
by sorry


end NUMINAMATH_CALUDE_four_numbers_theorem_l3438_343866


namespace NUMINAMATH_CALUDE_carol_initial_amount_l3438_343861

/-- Carol's initial amount of money -/
def carol_initial : ℕ := sorry

/-- Carol's weekly savings -/
def carol_weekly_savings : ℕ := 9

/-- Mike's initial amount of money -/
def mike_initial : ℕ := 90

/-- Mike's weekly savings -/
def mike_weekly_savings : ℕ := 3

/-- Number of weeks -/
def weeks : ℕ := 5

theorem carol_initial_amount :
  carol_initial = 60 :=
by
  have h1 : carol_initial + weeks * carol_weekly_savings = mike_initial + weeks * mike_weekly_savings :=
    sorry
  sorry

end NUMINAMATH_CALUDE_carol_initial_amount_l3438_343861


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3438_343807

-- Define the total number of votes
def total_votes : ℕ := 560000

-- Define the percentage of valid votes received by candidate A
def candidate_A_percentage : ℚ := 55 / 100

-- Define the number of valid votes received by candidate A
def candidate_A_votes : ℕ := 261800

-- Define the percentage of invalid votes
def invalid_vote_percentage : ℚ := 15 / 100

-- Theorem statement
theorem invalid_votes_percentage :
  (1 - (candidate_A_votes : ℚ) / (candidate_A_percentage * total_votes)) = invalid_vote_percentage := by
  sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3438_343807


namespace NUMINAMATH_CALUDE_altitude_length_of_triangle_on_rectangle_diagonal_l3438_343828

/-- Given a rectangle with sides a and b, and a triangle constructed on its diagonal
    with an area equal to the rectangle's area, the length of the altitude drawn to
    the base (diagonal) of the triangle is (2ab) / √(a² + b²). -/
theorem altitude_length_of_triangle_on_rectangle_diagonal
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let rectangle_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := (1/2) * diagonal * (2 * rectangle_area / diagonal)
  triangle_area = rectangle_area →
  (2 * rectangle_area / diagonal) = (2 * a * b) / Real.sqrt (a^2 + b^2) := by
sorry


end NUMINAMATH_CALUDE_altitude_length_of_triangle_on_rectangle_diagonal_l3438_343828


namespace NUMINAMATH_CALUDE_share_decrease_proof_l3438_343831

theorem share_decrease_proof (total : ℕ) (c_share : ℕ) (b_decrease : ℕ) (c_decrease : ℕ) 
  (h_total : total = 1010)
  (h_c_share : c_share = 495)
  (h_b_decrease : b_decrease = 10)
  (h_c_decrease : c_decrease = 15) :
  ∃ (a_share b_share : ℕ) (x : ℕ),
    a_share + b_share + c_share = total ∧
    (a_share - x) / 3 = (b_share - b_decrease) / 2 ∧
    (a_share - x) / 3 = (c_share - c_decrease) / 5 ∧
    x = 25 := by
  sorry

end NUMINAMATH_CALUDE_share_decrease_proof_l3438_343831


namespace NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l3438_343898

theorem max_sum_given_sum_squares_and_product (x y : ℝ) : 
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l3438_343898


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3438_343808

theorem divisible_by_nine : ∃ k : ℤ, 2^10 - 2^8 + 2^6 - 2^4 + 2^2 - 1 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3438_343808


namespace NUMINAMATH_CALUDE_passing_percentage_l3438_343870

theorem passing_percentage (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ) 
  (h1 : student_marks = 175)
  (h2 : failed_by = 56)
  (h3 : max_marks = 700) :
  (((student_marks + failed_by : ℚ) / max_marks) * 100).floor = 33 := by
sorry

end NUMINAMATH_CALUDE_passing_percentage_l3438_343870


namespace NUMINAMATH_CALUDE_main_theorem_l3438_343882

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the tangent length
def tangent_length (p : Point_P) : ℝ := sorry

-- Define the circumcircle N
def circle_N (p : Point_P) (x y : ℝ) : Prop := sorry

-- Define the chord length AB
def chord_length (p : Point_P) : ℝ := sorry

theorem main_theorem :
  (∃ p1 p2 : Point_P, tangent_length p1 = 2*Real.sqrt 3 ∧ tangent_length p2 = 2*Real.sqrt 3 ∧
    ((p1.x = 0 ∧ p1.y = 0) ∨ (p1.x = 16/5 ∧ p1.y = 8/5)) ∧
    ((p2.x = 0 ∧ p2.y = 0) ∨ (p2.x = 16/5 ∧ p2.y = 8/5))) ∧
  (∀ p : Point_P, circle_N p 0 4 ∧ circle_N p (8/5) (4/5)) ∧
  (∃ p_min : Point_P, ∀ p : Point_P, chord_length p_min ≤ chord_length p ∧ chord_length p_min = Real.sqrt 11) :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l3438_343882


namespace NUMINAMATH_CALUDE_five_letter_words_count_l3438_343839

def alphabet_size : Nat := 26
def excluded_letter : Nat := 1

theorem five_letter_words_count :
  let available_letters := alphabet_size - excluded_letter
  (available_letters ^ 4 : Nat) = 390625 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l3438_343839


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3438_343811

theorem ceiling_floor_difference : ⌈((15 / 8 : ℚ) ^ 2 * (-34 / 4 : ℚ))⌉ - ⌊(15 / 8 : ℚ) * ⌊-34 / 4⌋⌋ = -12 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3438_343811


namespace NUMINAMATH_CALUDE_tricia_age_l3438_343860

theorem tricia_age (tricia amilia yorick eugene khloe rupert vincent : ℕ) : 
  tricia = amilia / 3 →
  amilia = yorick / 4 →
  ∃ k : ℕ, yorick = k * eugene →
  khloe = eugene / 3 →
  rupert = khloe + 10 →
  rupert = vincent - 2 →
  vincent = 22 →
  tricia = 5 →
  tricia = 5 := by sorry

end NUMINAMATH_CALUDE_tricia_age_l3438_343860


namespace NUMINAMATH_CALUDE_correspondence_proof_l3438_343836

/-- Given sets A and B, and a mapping f from A to B defined as
    f(x, y) = (x + 2y, 2x - y), prove that (1, 1) in A
    corresponds to (3, 1) in B under this mapping. -/
theorem correspondence_proof (A B : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ × ℝ)
    (hf : ∀ (x y : ℝ), f (x, y) = (x + 2*y, 2*x - y))
    (hA : (1, 1) ∈ A) (hB : (3, 1) ∈ B) :
    f (1, 1) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_correspondence_proof_l3438_343836


namespace NUMINAMATH_CALUDE_time_after_duration_l3438_343825

/-- Represents time in a 12-hour format -/
structure Time12 where
  hour : Nat
  minute : Nat
  second : Nat
  isPM : Bool

/-- Adds a duration to a given time -/
def addDuration (t : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Converts the hour component to 12-hour format -/
def to12Hour (h : Nat) : Nat :=
  sorry

theorem time_after_duration (initial : Time12) (final : Time12) :
  initial = Time12.mk 3 15 15 true →
  final = addDuration initial 196 58 16 →
  final.hour = 8 ∧ 
  final.minute = 13 ∧ 
  final.second = 31 ∧ 
  final.isPM = true ∧
  final.hour + final.minute + final.second = 52 :=
sorry

end NUMINAMATH_CALUDE_time_after_duration_l3438_343825


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l3438_343865

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation x^2 - x + 3 = 0 -/
def givenEquation : QuadraticEquation :=
  { a := 1, b := -1, c := 3 }

theorem coefficients_of_given_equation :
  givenEquation.a = 1 ∧ givenEquation.b = -1 ∧ givenEquation.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l3438_343865


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3438_343887

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as natural numbers
  base_angle₁ : ℕ
  base_angle₂ : ℕ
  vertex_angle : ℕ
  is_isosceles : base_angle₁ = base_angle₂
  angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180

theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : t.base_angle₁ = 50 ∨ t.vertex_angle = 50) :
  t.base_angle₁ = 50 ∨ t.base_angle₁ = 65 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3438_343887


namespace NUMINAMATH_CALUDE_infinite_gcd_condition_l3438_343867

open Set Function Nat

/-- A permutation of positive integers -/
def PositiveIntegerPermutation := ℕ+ → ℕ+

/-- The set of indices satisfying the GCD condition -/
def GcdConditionSet (a : PositiveIntegerPermutation) : Set ℕ+ :=
  {i | Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4}

/-- The main theorem -/
theorem infinite_gcd_condition (a : PositiveIntegerPermutation) 
  (h : Bijective a) : Infinite (GcdConditionSet a) := by
  sorry


end NUMINAMATH_CALUDE_infinite_gcd_condition_l3438_343867


namespace NUMINAMATH_CALUDE_min_n_is_correct_l3438_343864

/-- The minimum positive integer n such that the expansion of (x^2 - 1/x^3)^n contains a constant term -/
def min_n : ℕ := 5

/-- The expansion of (x^2 - 1/x^3)^n contains a constant term if and only if
    there exists an r such that 2n - 5r = 0 -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 2 * n = 5 * r

theorem min_n_is_correct :
  (∀ k < min_n, ¬ has_constant_term k) ∧ has_constant_term min_n :=
sorry

end NUMINAMATH_CALUDE_min_n_is_correct_l3438_343864


namespace NUMINAMATH_CALUDE_smallest_factor_perfect_square_l3438_343881

theorem smallest_factor_perfect_square : 
  (∀ k : ℕ, k < 14 → ¬ ∃ m : ℕ, 3150 * k = m * m) ∧ 
  ∃ n : ℕ, 3150 * 14 = n * n := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_perfect_square_l3438_343881


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3438_343801

/-- The sum of the infinite series ∑(k=1 to ∞) [12^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1)))] is equal to 3. -/
theorem infinite_series_sum : 
  (∑' k, (12 : ℝ)^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))) = 3 :=
sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3438_343801


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l3438_343879

theorem households_without_car_or_bike 
  (total : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (bike_only : ℕ) 
  (h1 : total = 90) 
  (h2 : both = 22) 
  (h3 : with_car = 44) 
  (h4 : bike_only = 35) : 
  total - (with_car + bike_only + both - both) = 11 := by
  sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l3438_343879


namespace NUMINAMATH_CALUDE_solve_for_m_l3438_343856

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 8

-- State the theorem
theorem solve_for_m :
  ∃ m : ℝ, (f 8 - g m 8 = 20) ∧ (m = -25.5) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3438_343856


namespace NUMINAMATH_CALUDE_set_intersection_union_theorem_l3438_343854

def A : Set ℝ := {x | 2*x - x^2 ≤ x}
def B : Set ℝ := {x | x/(1-x) ≤ x/(1-x)}
def C (a b : ℝ) : Set ℝ := {x | a*x^2 + x + b < 0}

theorem set_intersection_union_theorem (a b : ℝ) :
  (A ∪ B) ∩ (C a b) = ∅ ∧ (A ∪ B) ∪ (C a b) = Set.univ →
  a = -1/3 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_theorem_l3438_343854


namespace NUMINAMATH_CALUDE_constant_is_two_l3438_343863

theorem constant_is_two (p c : ℕ) (n : ℕ) (hp : Prime p) (hp_gt_two : p > 2)
  (hn : n = c * p) (h_one_even_divisor : ∃! d : ℕ, d ∣ n ∧ Even d) : c = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_is_two_l3438_343863


namespace NUMINAMATH_CALUDE_infinitely_many_n_squared_divides_b_power_n_plus_one_l3438_343868

theorem infinitely_many_n_squared_divides_b_power_n_plus_one
  (b : ℕ) (hb : b > 2) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n^2 ∣ b^n + 1) ↔ ¬∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_squared_divides_b_power_n_plus_one_l3438_343868


namespace NUMINAMATH_CALUDE_equation_solution_l3438_343852

theorem equation_solution : ∃! y : ℚ, (4 * y + 2) / (5 * y - 5) = 3 / 4 ∧ 5 * y - 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3438_343852


namespace NUMINAMATH_CALUDE_negation_of_implication_l3438_343896

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3438_343896


namespace NUMINAMATH_CALUDE_tvs_on_auction_site_l3438_343878

def tvs_in_person : ℕ := 8
def tvs_online_multiplier : ℕ := 3
def total_tvs : ℕ := 42

theorem tvs_on_auction_site :
  let tvs_online := tvs_online_multiplier * tvs_in_person
  let tvs_before_auction := tvs_in_person + tvs_online
  total_tvs - tvs_before_auction = 10 := by
sorry

end NUMINAMATH_CALUDE_tvs_on_auction_site_l3438_343878


namespace NUMINAMATH_CALUDE_intersection_A_B_l3438_343885

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | Real.log x < 1}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3438_343885


namespace NUMINAMATH_CALUDE_average_score_calculation_l3438_343897

theorem average_score_calculation (T : ℝ) (h : T > 0) :
  let male_ratio : ℝ := 0.4
  let female_ratio : ℝ := 1 - male_ratio
  let male_avg : ℝ := 75
  let female_avg : ℝ := 80
  let total_score : ℝ := male_ratio * T * male_avg + female_ratio * T * female_avg
  total_score / T = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_calculation_l3438_343897


namespace NUMINAMATH_CALUDE_solution_difference_l3438_343851

theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (6 * r - 18) / (r^2 + 2*r - 15) = r + 3 →
  (6 * s - 18) / (s^2 + 2*s - 15) = s + 3 →
  r > s →
  r - s = 8 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3438_343851


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l3438_343857

theorem mrs_hilt_reading (books : ℝ) (chapters_per_book : ℝ) 
  (h1 : books = 4.0) (h2 : chapters_per_book = 4.25) : 
  books * chapters_per_book = 17 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l3438_343857


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l3438_343822

theorem fraction_zero_implies_x_zero (x : ℚ) : 
  x / (2 * x - 1) = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l3438_343822


namespace NUMINAMATH_CALUDE_dividend_calculation_l3438_343829

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 158 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3438_343829


namespace NUMINAMATH_CALUDE_total_players_l3438_343824

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabadi = 10) 
  (h2 : kho_kho_only = 20) 
  (h3 : both = 5) : 
  kabadi + kho_kho_only - both = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l3438_343824


namespace NUMINAMATH_CALUDE_tangent_slope_of_circle_l3438_343893

/-- Given a circle with center (2,3) and a point (7,4) on the circle,
    the slope of the tangent line at (7,4) is -5. -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (2, 3) →
  point = (7, 4) →
  (point.2 - center.2) / (point.1 - center.1) = 1/5 →
  -(point.1 - center.1) / (point.2 - center.2) = -5 :=
by sorry


end NUMINAMATH_CALUDE_tangent_slope_of_circle_l3438_343893


namespace NUMINAMATH_CALUDE_linear_dependence_condition_l3438_343805

def v1 : ℝ × ℝ × ℝ := (1, 2, 3)
def v2 (k : ℝ) : ℝ × ℝ × ℝ := (4, k, 6)

def is_linearly_dependent (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), (a, b) ≠ (0, 0) ∧ a • v1 + b • v2 = (0, 0, 0)

theorem linear_dependence_condition (k : ℝ) :
  is_linearly_dependent v1 (v2 k) ↔ k = 8 :=
sorry

end NUMINAMATH_CALUDE_linear_dependence_condition_l3438_343805


namespace NUMINAMATH_CALUDE_chips_sales_third_fourth_week_l3438_343855

/-- Proves that the number of bags of chips sold in each of the third and fourth week is 20 --/
theorem chips_sales_third_fourth_week :
  let total_sales : ℕ := 100
  let first_week_sales : ℕ := 15
  let second_week_sales : ℕ := 3 * first_week_sales
  let remaining_sales : ℕ := total_sales - (first_week_sales + second_week_sales)
  let third_fourth_week_sales : ℕ := remaining_sales / 2
  third_fourth_week_sales = 20 := by
  sorry

end NUMINAMATH_CALUDE_chips_sales_third_fourth_week_l3438_343855


namespace NUMINAMATH_CALUDE_two_solutions_system_l3438_343817

theorem two_solutions_system (x y : ℝ) : 
  (x = 3 * x^2 + y^2 ∧ y = 3 * x * y) → 
  (∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ p.1 = 3 * p.1^2 + p.2^2 ∧ p.2 = 3 * p.1 * p.2) ∧
  (∃! q : ℝ × ℝ, q ≠ p ∧ q.1 = x ∧ q.2 = y ∧ q.1 = 3 * q.1^2 + q.2^2 ∧ q.2 = 3 * q.1 * q.2) ∧
  (∀ r : ℝ × ℝ, r ≠ p ∧ r ≠ q → ¬(r.1 = 3 * r.1^2 + r.2^2 ∧ r.2 = 3 * r.1 * r.2)) :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_system_l3438_343817


namespace NUMINAMATH_CALUDE_identify_burned_bulb_l3438_343858

/-- Represents the time in seconds for screwing or unscrewing a bulb -/
def operation_time : ℕ := 10

/-- Represents the number of bulbs in the series -/
def num_bulbs : ℕ := 4

/-- Represents the minimum time to identify the burned-out bulb -/
def min_identification_time : ℕ := 60

/-- Theorem stating that the minimum time to identify the burned-out bulb is 60 seconds -/
theorem identify_burned_bulb :
  ∀ (burned_bulb_position : Fin num_bulbs),
  min_identification_time = operation_time * (2 * (num_bulbs - 1)) :=
by sorry

end NUMINAMATH_CALUDE_identify_burned_bulb_l3438_343858


namespace NUMINAMATH_CALUDE_spherical_segment_volume_l3438_343891

/-- Given a sphere of radius 10 cm, prove that a spherical segment with a ratio of 10:7 for its curved surface area to base area has a volume of 288π cm³ -/
theorem spherical_segment_volume (r : ℝ) (m : ℝ) (h_r : r = 10) 
  (h_ratio : (2 * r * m) / (m * (2 * r - m)) = 10 / 7) : 
  (m^2 * π / 3) * (3 * r - m) = 288 * π := by
  sorry

#check spherical_segment_volume

end NUMINAMATH_CALUDE_spherical_segment_volume_l3438_343891


namespace NUMINAMATH_CALUDE_triangle_ratio_l3438_343877

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  2 * (Real.cos (A / 2))^2 = (Real.sqrt 3 / 3) * Real.sin A →
  Real.sin (B - C) = 4 * Real.cos B * Real.sin C →
  b / c = 1 + Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3438_343877


namespace NUMINAMATH_CALUDE_number_remainder_l3438_343869

theorem number_remainder (N : ℤ) 
  (h1 : N % 195 = 79)
  (h2 : N % 273 = 109) : 
  N % 39 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_remainder_l3438_343869


namespace NUMINAMATH_CALUDE_certain_number_proof_l3438_343853

theorem certain_number_proof (x : ℝ) : 
  (x / 3 = 248.14814814814815 / 100 * 162) → x = 1206 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3438_343853


namespace NUMINAMATH_CALUDE_jane_age_problem_l3438_343842

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Define what it means for a number to be a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem jane_age_problem :
  ∃! x : ℕ, x > 0 ∧ is_perfect_square (x - 1) ∧ is_perfect_cube (x + 1) ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_jane_age_problem_l3438_343842


namespace NUMINAMATH_CALUDE_simplify_expression_l3438_343820

theorem simplify_expression (a b : ℝ) : (22*a + 60*b) + (10*a + 29*b) - (9*a + 50*b) = 23*a + 39*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3438_343820


namespace NUMINAMATH_CALUDE_remaining_tanning_time_l3438_343812

/-- Calculates the remaining tanning time for the last two weeks of the month. -/
theorem remaining_tanning_time 
  (max_monthly_time : ℕ) 
  (daily_time : ℕ) 
  (days_per_week : ℕ) 
  (first_half_weeks : ℕ) 
  (h1 : max_monthly_time = 200)
  (h2 : daily_time = 30)
  (h3 : days_per_week = 2)
  (h4 : first_half_weeks = 2) :
  max_monthly_time - (daily_time * days_per_week * first_half_weeks) = 80 :=
by
  sorry

#check remaining_tanning_time

end NUMINAMATH_CALUDE_remaining_tanning_time_l3438_343812


namespace NUMINAMATH_CALUDE_correct_division_result_l3438_343804

theorem correct_division_result (student_divisor student_quotient correct_divisor : ℕ) 
  (h1 : student_divisor = 63)
  (h2 : student_quotient = 24)
  (h3 : correct_divisor = 36) :
  (student_divisor * student_quotient) / correct_divisor = 42 :=
by sorry

end NUMINAMATH_CALUDE_correct_division_result_l3438_343804
