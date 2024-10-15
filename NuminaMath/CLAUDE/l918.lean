import Mathlib

namespace NUMINAMATH_CALUDE_li_ming_father_age_l918_91871

theorem li_ming_father_age :
  ∃! age : ℕ, 
    18 ≤ age ∧ age ≤ 70 ∧
    ∃ (month day : ℕ), 
      1 ≤ month ∧ month ≤ 12 ∧
      1 ≤ day ∧ day ≤ 31 ∧
      age * month * day = 2975 ∧
      age = 35 := by
sorry

end NUMINAMATH_CALUDE_li_ming_father_age_l918_91871


namespace NUMINAMATH_CALUDE_total_art_pieces_l918_91892

theorem total_art_pieces (asian_art : ℕ) (egyptian_art : ℕ) 
  (h1 : asian_art = 465) (h2 : egyptian_art = 527) :
  asian_art + egyptian_art = 992 := by
  sorry

end NUMINAMATH_CALUDE_total_art_pieces_l918_91892


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l918_91837

-- Define the curve
def curve (x a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x a : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) : 
  (curve (-1) a = a + 2) →  -- Point (-1, a+2) is on the curve
  (curve_derivative (-1) a = 8) →  -- Slope at (-1, a+2) is 8
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l918_91837


namespace NUMINAMATH_CALUDE_right_triangle_area_l918_91862

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b = 24 →
  c = 24 →
  a^2 + c^2 = (a + b + c)^2 →
  (1/2) * a * c = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l918_91862


namespace NUMINAMATH_CALUDE_expression_evaluation_l918_91824

theorem expression_evaluation : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = 
  (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l918_91824


namespace NUMINAMATH_CALUDE_special_function_is_zero_l918_91813

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z

/-- Theorem stating that any function satisfying the special property must be the constant zero function -/
theorem special_function_is_zero (f : ℝ → ℝ) (h : special_function f) : 
  ∀ x : ℝ, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_special_function_is_zero_l918_91813


namespace NUMINAMATH_CALUDE_division_and_addition_l918_91851

theorem division_and_addition : (12 / (1/4)) + 5 = 53 := by sorry

end NUMINAMATH_CALUDE_division_and_addition_l918_91851


namespace NUMINAMATH_CALUDE_jungs_youngest_sibling_age_l918_91804

/-- Represents the ages of the people in the problem -/
structure Ages where
  li : ℕ
  zhang : ℕ
  jung : ℕ
  mei : ℕ
  youngest : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.li = 12 ∧
  ages.zhang = 2 * ages.li ∧
  ages.jung = ages.zhang + 2 ∧
  ages.mei = ages.jung / 2 ∧
  ages.zhang + ages.jung + ages.mei + ages.youngest = 66

/-- The theorem stating that Jung's youngest sibling is 3 years old -/
theorem jungs_youngest_sibling_age (ages : Ages) 
  (h : problem_conditions ages) : ages.youngest = 3 := by
  sorry


end NUMINAMATH_CALUDE_jungs_youngest_sibling_age_l918_91804


namespace NUMINAMATH_CALUDE_xy_minus_x_equals_nine_l918_91849

theorem xy_minus_x_equals_nine (x y : ℝ) (hx : x = 3) (hy : y = 4) : x * y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_minus_x_equals_nine_l918_91849


namespace NUMINAMATH_CALUDE_range_of_m_l918_91877

theorem range_of_m (m x : ℝ) : 
  (((m + 3) / (x - 1) = 1) ∧ (x > 0)) → (m > -4 ∧ m ≠ -3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l918_91877


namespace NUMINAMATH_CALUDE_meaningful_range_of_fraction_l918_91883

/-- The meaningful range of a fraction is the set of values for which the denominator is non-zero. -/
def meaningful_range (f : ℝ → ℝ) : Set ℝ :=
  {x | f x ≠ 0}

/-- The function representing the denominator of the fraction x / (x - 3). -/
def denominator (x : ℝ) : ℝ := x - 3

theorem meaningful_range_of_fraction :
    meaningful_range denominator = {x | x ≠ 3} := by
  sorry

end NUMINAMATH_CALUDE_meaningful_range_of_fraction_l918_91883


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l918_91806

-- Define a geometric sequence
def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ q : ℝ, b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem geometric_sequence_properties :
  -- There exist real numbers b₁, b₂, b₃ forming a geometric sequence such that b₁ < b₂ and b₂ > b₃
  (∃ b₁ b₂ b₃ : ℝ, is_geometric_sequence b₁ b₂ b₃ ∧ b₁ < b₂ ∧ b₂ > b₃) ∧
  -- If b₁ * b₂ < 0, then b₂ * b₃ < 0 for any geometric sequence b₁, b₂, b₃
  (∀ b₁ b₂ b₃ : ℝ, is_geometric_sequence b₁ b₂ b₃ → b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l918_91806


namespace NUMINAMATH_CALUDE_subtracted_number_l918_91803

theorem subtracted_number (t k x : ℝ) : 
  t = 5 / 9 * (k - x) → 
  t = 50 → 
  k = 122 → 
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_l918_91803


namespace NUMINAMATH_CALUDE_consecutive_product_divisible_by_two_l918_91879

theorem consecutive_product_divisible_by_two (n : ℕ) : 
  2 ∣ (n * (n + 1)) := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_divisible_by_two_l918_91879


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l918_91800

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 2, 3]
  Matrix.det A = 2 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l918_91800


namespace NUMINAMATH_CALUDE_fraction_simplification_l918_91872

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l918_91872


namespace NUMINAMATH_CALUDE_a_closed_form_l918_91869

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => a n - (n + 1 : ℚ) / (Nat.factorial (n + 2))

theorem a_closed_form (n : ℕ) :
  a n = (Nat.factorial (n + 1) + 1 : ℚ) / Nat.factorial (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_a_closed_form_l918_91869


namespace NUMINAMATH_CALUDE_min_sum_of_integers_l918_91811

theorem min_sum_of_integers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (5 * a) > (20 * b)) : 
  ∃ (min : ℕ), min = 6 ∧ ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ (5 * x) > (20 * y) → x + y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_integers_l918_91811


namespace NUMINAMATH_CALUDE_three_coin_outcomes_l918_91809

/-- The number of possible outcomes when throwing a single coin -/
def coin_outcomes : Nat := 2

/-- The number of coins being thrown -/
def num_coins : Nat := 3

/-- Calculates the total number of outcomes when throwing multiple coins -/
def total_outcomes (n : Nat) : Nat := coin_outcomes ^ n

/-- Theorem: The number of possible outcomes when throwing three distinguishable coins is 8 -/
theorem three_coin_outcomes : total_outcomes num_coins = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_coin_outcomes_l918_91809


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l918_91891

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + 2*I
  is_in_second_quadrant z :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l918_91891


namespace NUMINAMATH_CALUDE_girls_in_ritas_class_l918_91846

/-- Calculates the number of girls in a class given the total number of students and the ratio of girls to boys -/
def girls_in_class (total_students : ℕ) (girl_ratio : ℕ) (boy_ratio : ℕ) : ℕ :=
  (total_students * girl_ratio) / (girl_ratio + boy_ratio)

/-- Theorem stating that in a class with 35 students and a 3:4 ratio of girls to boys, there are 15 girls -/
theorem girls_in_ritas_class :
  girls_in_class 35 3 4 = 15 := by
  sorry

#eval girls_in_class 35 3 4

end NUMINAMATH_CALUDE_girls_in_ritas_class_l918_91846


namespace NUMINAMATH_CALUDE_james_soda_packs_l918_91878

/-- The number of packs of sodas James bought -/
def packs_bought : ℕ := 5

/-- The number of sodas in each pack -/
def sodas_per_pack : ℕ := 12

/-- The number of sodas James already had -/
def initial_sodas : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of sodas James drinks per day -/
def sodas_per_day : ℕ := 10

theorem james_soda_packs :
  packs_bought * sodas_per_pack + initial_sodas = sodas_per_day * days_in_week := by
  sorry

end NUMINAMATH_CALUDE_james_soda_packs_l918_91878


namespace NUMINAMATH_CALUDE_cone_division_ratio_l918_91886

/-- Given a right circular cone with height 6 inches and base radius 4 inches,
    if a plane parallel to the base divides the cone into two solids C and F
    such that the ratio of their surface areas and volumes is k = 3/7,
    then the radius of the smaller cone C is (4 * (3/10)^(1/3)) / 3 times the original radius. -/
theorem cone_division_ratio (h : ℝ) (r : ℝ) (k : ℝ) :
  h = 6 →
  r = 4 →
  k = 3 / 7 →
  ∃ x : ℝ,
    x = (4 * (3 / 10) ^ (1 / 3)) / 3 * r ∧
    (π * x^2 + π * x * (Real.sqrt (h^2 + r^2) * x / r)) / 
    (π * r^2 + π * r * Real.sqrt (h^2 + r^2) - 
     (π * x^2 + π * x * (Real.sqrt (h^2 + r^2) * x / r))) = k ∧
    ((1 / 3) * π * x^2 * (h * x / r)) / 
    ((1 / 3) * π * r^2 * h - (1 / 3) * π * x^2 * (h * x / r)) = k :=
by
  sorry


end NUMINAMATH_CALUDE_cone_division_ratio_l918_91886


namespace NUMINAMATH_CALUDE_problem_solution_l918_91897

theorem problem_solution (x : ℝ) 
  (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (6 * x) * Real.sqrt (5 * x) * Real.sqrt (10 * x) = 20) : 
  x = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l918_91897


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l918_91874

theorem quadratic_equation_solution (x m n : ℝ) : 
  (x^2 + x + m = (x - n)^2) → (m = 1/4 ∧ n = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l918_91874


namespace NUMINAMATH_CALUDE_total_black_dots_l918_91832

/-- The number of black dots on a Type A butterfly -/
def dotsA : ℝ := 12

/-- The number of black dots on a Type B butterfly -/
def dotsB : ℝ := 8.5

/-- The number of black dots on a Type C butterfly -/
def dotsC : ℝ := 19

/-- The number of Type A butterflies -/
def numA : ℕ := 145

/-- The number of Type B butterflies -/
def numB : ℕ := 112

/-- The number of Type C butterflies -/
def numC : ℕ := 140

/-- The total number of butterflies -/
def totalButterflies : ℕ := 397

/-- Theorem: The total number of black dots among all butterflies is 5352 -/
theorem total_black_dots :
  dotsA * numA + dotsB * numB + dotsC * numC = 5352 := by
  sorry

end NUMINAMATH_CALUDE_total_black_dots_l918_91832


namespace NUMINAMATH_CALUDE_quadratic_root_value_l918_91868

theorem quadratic_root_value (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 : ℂ) * (4 + Complex.I) ^ 2 + p * (4 + Complex.I) + q = 0 →
  q = -51 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l918_91868


namespace NUMINAMATH_CALUDE_computer_price_increase_l918_91884

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 520) : 
  d * 1.3 = 338 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l918_91884


namespace NUMINAMATH_CALUDE_orange_pounds_value_l918_91828

/-- The price of a dozen eggs -/
def egg_price : ℝ := sorry

/-- The price per pound of oranges -/
def orange_price : ℝ := sorry

/-- The number of pounds of oranges -/
def orange_pounds : ℝ := sorry

/-- The current prices of eggs and oranges are equal -/
axiom price_equality : egg_price = orange_price * orange_pounds

/-- The price increase equation -/
axiom price_increase : 0.09 * egg_price + 0.06 * orange_price * orange_pounds = 15

theorem orange_pounds_value : orange_pounds = 100 := by sorry

end NUMINAMATH_CALUDE_orange_pounds_value_l918_91828


namespace NUMINAMATH_CALUDE_concert_ticket_price_l918_91845

theorem concert_ticket_price (total_people : Nat) (discount_group1 : Nat) (discount_group2 : Nat)
  (discount1 : Real) (discount2 : Real) (total_revenue : Real) :
  total_people = 56 →
  discount_group1 = 10 →
  discount_group2 = 20 →
  discount1 = 0.4 →
  discount2 = 0.15 →
  total_revenue = 980 →
  ∃ (original_price : Real),
    original_price = 20 ∧
    total_revenue = (discount_group1 * (1 - discount1) * original_price) +
                    (discount_group2 * (1 - discount2) * original_price) +
                    ((total_people - discount_group1 - discount_group2) * original_price) :=
by sorry


end NUMINAMATH_CALUDE_concert_ticket_price_l918_91845


namespace NUMINAMATH_CALUDE_division_problem_l918_91815

theorem division_problem (n : ℕ) : 
  n / 3 = 7 ∧ n % 3 = 1 → n = 22 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l918_91815


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l918_91890

def n : ℕ := 2014

theorem rearrangement_theorem (x y : Fin n → ℤ) 
  (hx : ∀ i j : Fin n, i ≠ j → x i % n ≠ x j % n)
  (hy : ∀ i j : Fin n, i ≠ j → y i % n ≠ y j % n) :
  ∃ σ : Equiv.Perm (Fin n), 
    ∀ i j : Fin n, i ≠ j → (x i + y (σ i)) % (2 * n) ≠ (x j + y (σ j)) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l918_91890


namespace NUMINAMATH_CALUDE_max_min_difference_l918_91870

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
               (∀ x ∈ I, m ≤ f x) ∧ 
               (∃ x₁ ∈ I, f x₁ = M) ∧ 
               (∃ x₂ ∈ I, f x₂ = m) ∧ 
               M - m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l918_91870


namespace NUMINAMATH_CALUDE_last_digit_389_quaternary_l918_91887

def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem last_digit_389_quaternary :
  (decimal_to_quaternary 389).getLast? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_389_quaternary_l918_91887


namespace NUMINAMATH_CALUDE_regression_slope_l918_91838

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 2 - x

theorem regression_slope :
  ∀ x : ℝ, linear_regression (x + 1) = linear_regression x - 1 := by
  sorry

end NUMINAMATH_CALUDE_regression_slope_l918_91838


namespace NUMINAMATH_CALUDE_total_project_hours_l918_91822

/-- Represents the time spent on an activity --/
structure ActivityTime where
  hoursPerDay : ℕ
  days : ℕ

/-- Calculates the total hours for an activity --/
def totalHours (a : ActivityTime) : ℕ := a.hoursPerDay * a.days

/-- Represents the time spent on various activities for a song --/
structure SongTime where
  vocals : ActivityTime
  instrument : ActivityTime
  mixing : ActivityTime

/-- Represents the overall project time --/
structure ProjectTime where
  song1 : SongTime
  song2 : SongTime
  song3 : SongTime
  videoProduction : ActivityTime
  marketing : ActivityTime

/-- The given project time data --/
def givenProjectTime : ProjectTime :=
  { song1 := { vocals := { hoursPerDay := 8, days := 12 },
               instrument := { hoursPerDay := 2, days := 6 },
               mixing := { hoursPerDay := 4, days := 3 } },
    song2 := { vocals := { hoursPerDay := 10, days := 9 },
               instrument := { hoursPerDay := 3, days := 4 },
               mixing := { hoursPerDay := 5, days := 2 } },
    song3 := { vocals := { hoursPerDay := 6, days := 15 },
               instrument := { hoursPerDay := 1, days := 5 },
               mixing := { hoursPerDay := 3, days := 4 } },
    videoProduction := { hoursPerDay := 5, days := 7 },
    marketing := { hoursPerDay := 4, days := 10 } }

/-- Calculates the total hours spent on the project --/
def calculateTotalHours (p : ProjectTime) : ℕ :=
  totalHours p.song1.vocals + totalHours p.song1.instrument + totalHours p.song1.mixing +
  totalHours p.song2.vocals + totalHours p.song2.instrument + totalHours p.song2.mixing +
  totalHours p.song3.vocals + totalHours p.song3.instrument + totalHours p.song3.mixing +
  totalHours p.videoProduction + totalHours p.marketing

/-- Theorem: The total hours spent on the project is 414 --/
theorem total_project_hours : calculateTotalHours givenProjectTime = 414 := by
  sorry

end NUMINAMATH_CALUDE_total_project_hours_l918_91822


namespace NUMINAMATH_CALUDE_johns_star_wars_toys_cost_l918_91857

theorem johns_star_wars_toys_cost (lightsaber_cost other_toys_cost total_spent : ℕ) : 
  lightsaber_cost = 2 * other_toys_cost →
  total_spent = lightsaber_cost + other_toys_cost →
  total_spent = 3000 →
  other_toys_cost = 1000 := by
sorry

end NUMINAMATH_CALUDE_johns_star_wars_toys_cost_l918_91857


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l918_91812

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : Complex.im ((1 - i)^3) = -2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l918_91812


namespace NUMINAMATH_CALUDE_find_divisor_l918_91847

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 176) (h2 : quotient = 9) (h3 : remainder = 5) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 19 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l918_91847


namespace NUMINAMATH_CALUDE_quadratic_function_value_l918_91807

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_value (a b c : ℝ) :
  f a b c (Real.sqrt 2) = 3 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b c x| ≤ 1) →
  f a b c (Real.sqrt 2013) = 1343.67 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l918_91807


namespace NUMINAMATH_CALUDE_sum_floor_is_179_l918_91814

theorem sum_floor_is_179 
  (p q r s : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0)
  (h1 : p^2 + q^2 = 4016) (h2 : r^2 + s^2 = 4016)
  (h3 : p * r = 2000) (h4 : q * s = 2000) : 
  ⌊p + q + r + s⌋ = 179 := by
sorry

end NUMINAMATH_CALUDE_sum_floor_is_179_l918_91814


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l918_91817

/-- Given an arithmetic sequence {aₙ}, prove that 3a₅ + a₇ = 20 when a₃ + a₈ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 3 + a 8 = 10) →                               -- given condition
  3 * a 5 + a 7 = 20 :=                            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l918_91817


namespace NUMINAMATH_CALUDE_fraction_equality_l918_91896

theorem fraction_equality (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l918_91896


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l918_91895

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) : 
  total_length = 68 →
  difference = 12 →
  shorter_piece + (shorter_piece + difference) = total_length →
  shorter_piece = 28 := by
  sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l918_91895


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l918_91802

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 2 * x - y + 2 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -1

/-- The y-intercept of the line -/
def y_intercept : ℝ := 2

/-- The area of the triangle -/
def triangle_area : ℝ := 1

/-- Theorem: The area of the triangle formed by the line 2x - y + 2 = 0 and the coordinate axes is 1 -/
theorem triangle_area_is_one : triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_one_l918_91802


namespace NUMINAMATH_CALUDE_min_Q_value_l918_91805

/-- The integer closest to a rational number -/
def closest_integer (m : ℤ) (k : ℤ) : ℤ := sorry

/-- The probability Q(k) as defined in the problem -/
def Q (k : ℤ) : ℚ := sorry

theorem min_Q_value :
  ∀ k : ℤ, k % 2 = 1 → 1 ≤ k → k ≤ 150 → Q k ≥ 37/75 := by sorry

end NUMINAMATH_CALUDE_min_Q_value_l918_91805


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l918_91855

/-- The number of arrangements for a team of 3 players selected from 5 players (2 veterans and 3 new players) -/
def num_arrangements : ℕ :=
  let total_players : ℕ := 5
  let veteran_players : ℕ := 2
  let new_players : ℕ := 3
  let team_size : ℕ := 3
  -- Calculate the number of arrangements
  sorry

/-- Theorem stating that the number of valid arrangements is 48 -/
theorem valid_arrangements_count :
  num_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l918_91855


namespace NUMINAMATH_CALUDE_percentage_calculation_l918_91808

theorem percentage_calculation (x y : ℝ) : 
  (0.003 = (x/100) * 0.09) → 
  (0.008 = (y/100) * 0.15) → 
  (x = (0.003 / 0.09) * 100) ∧ 
  (y = (0.008 / 0.15) * 100) := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l918_91808


namespace NUMINAMATH_CALUDE_range_of_k_l918_91840

/-- Given a function f(x) = √(x+1) + k and an interval [a, b] where the range of f(x) on [a, b] is [a+1, b+1], prove that the range of k is (-1/4, 0]. -/
theorem range_of_k (a b : ℝ) (k : ℝ) (h_le : a ≤ b) :
  (∀ y ∈ Set.Icc (a + 1) (b + 1), ∃ x ∈ Set.Icc a b, Real.sqrt (x + 1) + k = y) →
  k ∈ Set.Ioo (-1/4) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l918_91840


namespace NUMINAMATH_CALUDE_digits_sum_after_erasures_l918_91833

/-- Represents the initial sequence of digits -/
def initial_sequence : List Nat := [1, 2, 3, 4, 5, 6]

/-- Applies the erasure steps to a given sequence -/
def apply_erasures (seq : List Nat) : List Nat :=
  sorry

/-- Gets the digit at a specific position in the final sequence -/
def get_digit_at_position (pos : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem digits_sum_after_erasures :
  get_digit_at_position 3000 + get_digit_at_position 3001 + get_digit_at_position 3002 = 8 :=
sorry

end NUMINAMATH_CALUDE_digits_sum_after_erasures_l918_91833


namespace NUMINAMATH_CALUDE_equation_solution_l918_91885

theorem equation_solution : ∃ x : ℝ, 35 * 2 - 10 = 5 * x + 20 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l918_91885


namespace NUMINAMATH_CALUDE_age_difference_l918_91827

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 20 → 
  man_age + 2 = 2 * (son_age + 2) → 
  man_age - son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l918_91827


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l918_91889

/-- Represents a seven-digit number in the form 3b5n678 -/
def sevenDigitNumber (b n : ℕ) : ℕ := 3000000 + 100000 * b + 50000 + 10000 * n + 678

/-- Checks if a number is divisible by 11 -/
def isDivisibleBy11 (num : ℕ) : Prop := ∃ k : ℕ, num = 11 * k

/-- b and n are single digits -/
def isSingleDigit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

theorem seven_digit_divisible_by_11 :
  ∃ b n : ℕ, isSingleDigit b ∧ isSingleDigit n ∧ 
  isDivisibleBy11 (sevenDigitNumber b n) ∧ 
  b = 4 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_11_l918_91889


namespace NUMINAMATH_CALUDE_equation_solution_l918_91852

theorem equation_solution : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l918_91852


namespace NUMINAMATH_CALUDE_min_years_plan_b_exceeds_plan_a_l918_91859

-- Define the investment amount for Plan A
def plan_a_investment : ℕ := 1000000

-- Define the initial investment and yearly increase for Plan B
def plan_b_initial : ℕ := 100000
def plan_b_increase : ℕ := 100000

-- Function to calculate the total investment of Plan B after n years
def plan_b_total (n : ℕ) : ℕ :=
  n * (2 * plan_b_initial + (n - 1) * plan_b_increase) / 2

-- Theorem stating the minimum number of years for Plan B to match or exceed Plan A
theorem min_years_plan_b_exceeds_plan_a :
  ∃ n : ℕ, (∀ k : ℕ, k < n → plan_b_total k < plan_a_investment) ∧
           plan_b_total n ≥ plan_a_investment ∧
           n = 5 :=
sorry

end NUMINAMATH_CALUDE_min_years_plan_b_exceeds_plan_a_l918_91859


namespace NUMINAMATH_CALUDE_garden_perimeter_l918_91875

/-- The perimeter of a rectangle with length l and breadth b is 2 * (l + b) -/
def rectanglePerimeter (l b : ℝ) : ℝ := 2 * (l + b)

/-- The perimeter of a rectangular garden with length 500 m and breadth 400 m is 1800 m -/
theorem garden_perimeter :
  rectanglePerimeter 500 400 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l918_91875


namespace NUMINAMATH_CALUDE_gumball_probability_l918_91863

/-- Given a jar with blue and pink gumballs, if the probability of drawing two blue
    gumballs in succession (with replacement) is 9/49, then the probability of
    drawing a pink gumball is 4/7. -/
theorem gumball_probability (B P : ℕ) (h : B > 0 ∧ P > 0) :
  (B : ℚ)^2 / (B + P : ℚ)^2 = 9/49 → (P : ℚ) / (B + P : ℚ) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_gumball_probability_l918_91863


namespace NUMINAMATH_CALUDE_first_zero_position_l918_91854

open Real

-- Define the decimal expansion of √2
def sqrt2_expansion : ℕ → ℕ
  | 0 => 1  -- integer part
  | (n+1) => sorry  -- n-th decimal digit, implementation details omitted

-- Define a function to check if there's a sequence of k zeroes starting at position n
def has_k_zeroes (k n : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range k → sqrt2_expansion (n + i) = 0

-- Main theorem
theorem first_zero_position (k : ℕ) (h : k > 0) :
  ∀ n, has_k_zeroes k n → n ≥ k :=
sorry

end NUMINAMATH_CALUDE_first_zero_position_l918_91854


namespace NUMINAMATH_CALUDE_modern_model_leads_to_older_structure_l918_91894

/-- Represents the population growth model -/
structure PopulationGrowthModel where
  -- Add necessary fields here

/-- Represents the age structure of a population -/
inductive AgeStructure
  | Younger
  | Older

/-- The modern population growth model -/
def modernPopulationGrowthModel : PopulationGrowthModel :=
  sorry

/-- The consequence of a population growth model on age structure -/
def consequenceOnAgeStructure (model : PopulationGrowthModel) : AgeStructure :=
  sorry

/-- Theorem stating that the modern population growth model leads to an older age structure -/
theorem modern_model_leads_to_older_structure :
  consequenceOnAgeStructure modernPopulationGrowthModel = AgeStructure.Older :=
sorry

end NUMINAMATH_CALUDE_modern_model_leads_to_older_structure_l918_91894


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l918_91835

theorem repeating_decimal_difference : 
  let repeating_decimal := (4 : ℚ) / 11
  let non_repeating_decimal := (36 : ℚ) / 100
  repeating_decimal - non_repeating_decimal = (4 : ℚ) / 1100 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l918_91835


namespace NUMINAMATH_CALUDE_remainder_division_l918_91836

theorem remainder_division (y : ℤ) (h : y % 288 = 45) : y % 24 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_division_l918_91836


namespace NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l918_91880

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def zero_at (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

def count_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ → Prop
  | 0 => True
  | n + 1 => ∃ x, a ≤ x ∧ x ≤ b ∧ zero_at f x ∧ count_zeros f a b n

theorem min_zeros_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 3) 
  (h_zero : zero_at f 2) : 
  count_zeros f (-3) 3 9 :=
sorry

end NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l918_91880


namespace NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l918_91843

/-- Given three consecutive digits p, q, and r, the sum of the three-digit 
    numbers pqr and rqp is always divisible by 212. -/
theorem consecutive_digits_sum_divisibility (p : ℕ) (hp : p < 8) : ∃ (k : ℕ),
  (100 * p + 10 * (p + 1) + (p + 2)) + (100 * (p + 2) + 10 * (p + 1) + p) = 212 * k :=
by
  sorry

#check consecutive_digits_sum_divisibility

end NUMINAMATH_CALUDE_consecutive_digits_sum_divisibility_l918_91843


namespace NUMINAMATH_CALUDE_min_quotient_value_l918_91818

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0) ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem min_quotient_value :
  ∀ n : ℕ, is_valid_number n → (n : ℚ) / (digit_sum n : ℚ) ≥ 105 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_value_l918_91818


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l918_91830

/-- The function f(x) = x^2(x-2) + 1 -/
def f (x : ℝ) : ℝ := x^2 * (x - 2) + 1

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

theorem tangent_line_at_one :
  let p : ℝ × ℝ := (1, f 1)
  let m : ℝ := f' 1
  ∀ x y : ℝ, (y - p.2 = m * (x - p.1)) ↔ (x + y - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l918_91830


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l918_91850

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l918_91850


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l918_91876

theorem inequality_and_equality_conditions (a b : ℝ) (h : a * b > 0) :
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) ≤ (a^2 + 10*a*b + b^2) / 12) ∧
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) = (a^2 + 10*a*b + b^2) / 12 ↔ a = b) ∧
  (((a^2 * b^2 * (a + b)^2) / 4)^(1/3) ≤ (a^2 + a*b + b^2) / 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l918_91876


namespace NUMINAMATH_CALUDE_root_product_l918_91819

def f (x : ℂ) : ℂ := x^5 - x^3 + x + 1

def g (x : ℂ) : ℂ := x^2 - 3

theorem root_product (x₁ x₂ x₃ x₄ x₅ : ℂ) 
  (hf : f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = 146 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l918_91819


namespace NUMINAMATH_CALUDE_two_roots_condition_l918_91899

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/4) * x + 1 else Real.log x

-- State the theorem
theorem two_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x = a * x ∧ f y = a * y ∧
   ∀ z : ℝ, z ≠ x ∧ z ≠ y → f z ≠ a * z) ↔
  a > 1/4 ∧ a < 1/Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l918_91899


namespace NUMINAMATH_CALUDE_son_age_l918_91860

/-- Represents the age of the father when the son was born -/
def N : ℕ := sorry

/-- Represents the current age of the son -/
def k : ℕ := sorry

/-- The father's current age is no more than 75 -/
axiom father_age_bound : N + k ≤ 75

/-- The son is exactly half the age of the father -/
axiom son_half_father_age : 2 * k = N + k

/-- There are exactly 8 distinct values of k where N is divisible by k -/
axiom eight_divisors : ∃ (S : Finset ℕ), S.card = 8 ∧ ∀ x ∈ S, N % x = 0

/-- The son's age is either 24 or 30 -/
theorem son_age : k = 24 ∨ k = 30 := by sorry

end NUMINAMATH_CALUDE_son_age_l918_91860


namespace NUMINAMATH_CALUDE_sector_angle_l918_91856

theorem sector_angle (r : ℝ) (θ : ℝ) (h1 : r * θ = 5) (h2 : r^2 * θ / 2 = 5) : θ = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l918_91856


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l918_91873

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -3; -2, -5]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l918_91873


namespace NUMINAMATH_CALUDE_triangular_array_digit_sum_l918_91848

def triangular_array_sum (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_digit_sum :
  ∃ n : ℕ, triangular_array_sum n = 1575 ∧ sum_of_digits n = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_digit_sum_l918_91848


namespace NUMINAMATH_CALUDE_athlete_difference_ultimate_fitness_camp_problem_l918_91888

/-- The difference in the number of athletes at Ultimate Fitness Camp over two nights -/
theorem athlete_difference (initial_athletes : ℕ) 
                           (leaving_rate : ℕ) (leaving_hours : ℕ)
                           (arriving_rate : ℕ) (arriving_hours : ℕ) : ℕ :=
  let athletes_leaving := leaving_rate * leaving_hours
  let athletes_remaining := initial_athletes - athletes_leaving
  let athletes_arriving := arriving_rate * arriving_hours
  let final_athletes := athletes_remaining + athletes_arriving
  initial_athletes - final_athletes

/-- The specific case of the Ultimate Fitness Camp problem -/
theorem ultimate_fitness_camp_problem : 
  athlete_difference 300 28 4 15 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_athlete_difference_ultimate_fitness_camp_problem_l918_91888


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l918_91867

-- Define the points
def C : ℝ × ℝ := (-6, 1)
def M : ℝ × ℝ := (-2, 3)

-- Define the midpoint formula
def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  m.1 = (x.1 + y.1) / 2 ∧ m.2 = (x.2 + y.2) / 2

-- Theorem statement
theorem sum_of_coordinates_D :
  ∃ D : ℝ × ℝ, is_midpoint M C D ∧ D.1 + D.2 = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l918_91867


namespace NUMINAMATH_CALUDE_probability_integer_log_l918_91861

/-- The set S of powers of 3 from 1 to 18 -/
def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 18 ∧ n = 3^k}

/-- The condition for log_a b to be an integer -/
def is_integer_log (a b : ℕ) : Prop :=
  ∃ k : ℕ, a^k = b

/-- The number of valid pairs (a,b) where log_a b is an integer -/
def count_valid_pairs : ℕ := 40

/-- The total number of distinct pairs from S -/
def total_pairs : ℕ := 153

/-- The main theorem stating the probability -/
theorem probability_integer_log :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 40 / 153 := by sorry

end NUMINAMATH_CALUDE_probability_integer_log_l918_91861


namespace NUMINAMATH_CALUDE_dan_marbles_remaining_l918_91853

/-- The number of marbles Dan has after giving some to Mary -/
def marbles_remaining (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem stating that Dan has 50 marbles after giving 14 to Mary -/
theorem dan_marbles_remaining : marbles_remaining 64 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_remaining_l918_91853


namespace NUMINAMATH_CALUDE_line_circle_intersection_equilateral_l918_91823

/-- Given a line and a circle in a Cartesian coordinate system, 
    if they intersect to form an equilateral triangle with the circle's center,
    then the parameter 'a' in the line equation must be 0. -/
theorem line_circle_intersection_equilateral (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + A.2 - 2 = 0) ∧ 
    (a * B.1 + B.2 - 2 = 0) ∧
    ((A.1 - 1)^2 + (A.2 - a)^2 = 16/3) ∧
    ((B.1 - 1)^2 + (B.2 - a)^2 = 16/3) ∧
    (let C : ℝ × ℝ := (1, a);
     (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
     (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
     (B.1 - C.1)^2 + (B.2 - C.2)^2 = 
     (C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_equilateral_l918_91823


namespace NUMINAMATH_CALUDE_beanie_babies_total_l918_91801

/-- The number of beanie babies Lori has -/
def lori_beanie_babies : ℕ := 300

/-- The ratio of Lori's beanie babies to Sydney's -/
def ratio : ℕ := 15

/-- The total number of beanie babies Lori and Sydney have together -/
def total_beanie_babies : ℕ := lori_beanie_babies + (lori_beanie_babies / ratio)

theorem beanie_babies_total : total_beanie_babies = 320 := by
  sorry

end NUMINAMATH_CALUDE_beanie_babies_total_l918_91801


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l918_91841

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)) →  -- arithmetic sequence condition
  a 3 + a 6 + a 9 = 3 * Real.pi / 4 →               -- given condition
  Real.cos (a 2 + a 10 + Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l918_91841


namespace NUMINAMATH_CALUDE_stratified_sample_probability_l918_91898

theorem stratified_sample_probability (grade10 grade11 grade12 : ℕ) (sample_size : ℕ) :
  grade10 = 300 →
  grade11 = 300 →
  grade12 = 400 →
  sample_size = 40 →
  (grade12 : ℚ) / (grade10 + grade11 + grade12 : ℚ) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_probability_l918_91898


namespace NUMINAMATH_CALUDE_ladder_cost_theorem_l918_91810

/-- Calculates the total cost of ladders given the number of ladders, rungs per ladder, and cost per rung for three different types of ladders. -/
def total_ladder_cost (ladders1 rungs1 cost1 ladders2 rungs2 cost2 ladders3 rungs3 cost3 : ℕ) : ℕ :=
  ladders1 * rungs1 * cost1 + ladders2 * rungs2 * cost2 + ladders3 * rungs3 * cost3

/-- Proves that the total cost of ladders for the given specifications is $14200. -/
theorem ladder_cost_theorem :
  total_ladder_cost 10 50 2 20 60 3 30 80 4 = 14200 := by
  sorry

end NUMINAMATH_CALUDE_ladder_cost_theorem_l918_91810


namespace NUMINAMATH_CALUDE_probability_of_C_l918_91858

def spinner_game (pA pB pC pD : ℚ) : Prop :=
  pA + pB + pC + pD = 1 ∧ pA ≥ 0 ∧ pB ≥ 0 ∧ pC ≥ 0 ∧ pD ≥ 0

theorem probability_of_C (pA pB pC pD : ℚ) :
  spinner_game pA pB pC pD → pA = 1/4 → pB = 1/3 → pC = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_C_l918_91858


namespace NUMINAMATH_CALUDE_angle_C_range_l918_91839

theorem angle_C_range (A B C : Real) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : AB = 1) (h6 : BC = 2) : 
  0 < C ∧ C ≤ π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_range_l918_91839


namespace NUMINAMATH_CALUDE_min_n_value_l918_91831

/-- The set A containing numbers from 1 to 6 -/
def A : Finset ℕ := Finset.range 6

/-- The set B containing numbers from 7 to n -/
def B (n : ℕ) : Finset ℕ := Finset.Icc 7 n

/-- A function that generates a set A_i -/
def generate_A_i (i : ℕ) (n : ℕ) : Finset ℕ := sorry

/-- The proposition that all A_i sets satisfy the given conditions -/
def valid_A_i_sets (n : ℕ) : Prop :=
  ∃ (f : ℕ → Finset ℕ), 
    (∀ i, i ≤ 20 → (f i).card = 8) ∧
    (∀ i, i ≤ 20 → (f i ∩ A).card = 3) ∧
    (∀ i, i ≤ 20 → (f i ∩ B n).card = 5) ∧
    (∀ i j, i < j ∧ j ≤ 20 → (f i ∩ f j).card ≤ 2)

theorem min_n_value : 
  (∀ n < 41, ¬ valid_A_i_sets n) ∧ valid_A_i_sets 41 := by sorry

end NUMINAMATH_CALUDE_min_n_value_l918_91831


namespace NUMINAMATH_CALUDE_girls_in_class_l918_91825

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 35) (h2 : ratio_girls = 3) (h3 : ratio_boys = 4) : 
  (total * ratio_girls) / (ratio_girls + ratio_boys) = 15 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_class_l918_91825


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l918_91864

theorem multiplication_table_odd_fraction :
  let factors := Finset.range 16
  let table := factors.product factors
  let odd_product (a b : ℕ) := Odd (a * b)
  (table.filter (fun (a, b) => odd_product a b)).card / table.card = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l918_91864


namespace NUMINAMATH_CALUDE_inverse_11_mod_1021_l918_91834

theorem inverse_11_mod_1021 : ∃ x : ℕ, x < 1021 ∧ (11 * x) % 1021 = 1 := by
  use 557
  sorry

end NUMINAMATH_CALUDE_inverse_11_mod_1021_l918_91834


namespace NUMINAMATH_CALUDE_min_upper_base_perimeter_is_12_l918_91816

/-- Represents a frustum with rectangular bases -/
structure Frustum where
  upperBaseLength : ℝ
  upperBaseWidth : ℝ
  height : ℝ
  volume : ℝ

/-- The minimum perimeter of the upper base of a frustum with given properties -/
def minUpperBasePerimeter (f : Frustum) : ℝ :=
  2 * (f.upperBaseLength + f.upperBaseWidth)

/-- Theorem stating the minimum perimeter of the upper base for a specific frustum -/
theorem min_upper_base_perimeter_is_12 (f : Frustum) 
  (h1 : f.height = 3)
  (h2 : f.volume = 63)
  (h3 : f.upperBaseLength * f.upperBaseWidth * 7 = 63) :
  minUpperBasePerimeter f ≥ 12 ∧ 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    f.upperBaseLength = a ∧ f.upperBaseWidth = b ∧ 
    minUpperBasePerimeter f = 12 :=
  sorry


end NUMINAMATH_CALUDE_min_upper_base_perimeter_is_12_l918_91816


namespace NUMINAMATH_CALUDE_simplify_expression_l918_91829

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l918_91829


namespace NUMINAMATH_CALUDE_cubic_roots_property_l918_91820

theorem cubic_roots_property (a b c : ℂ) : 
  (a^3 - a^2 - a - 1 = 0) → 
  (b^3 - b^2 - b - 1 = 0) → 
  (c^3 - c^2 - c - 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
  (∃ n : ℤ, (a^1982 - b^1982) / (a - b) + (b^1982 - c^1982) / (b - c) + (c^1982 - a^1982) / (c - a) = n) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_property_l918_91820


namespace NUMINAMATH_CALUDE_square_difference_plus_square_l918_91821

theorem square_difference_plus_square : 5^2 - 4^2 + 3^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_square_l918_91821


namespace NUMINAMATH_CALUDE_larger_number_problem_l918_91844

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1325)
  (h2 : L = 5 * S + 5) :
  L = 1655 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l918_91844


namespace NUMINAMATH_CALUDE_evaluate_expression_l918_91882

theorem evaluate_expression (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d)^d = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l918_91882


namespace NUMINAMATH_CALUDE_unique_solutions_l918_91842

def system_solution (x y : ℝ) : Prop :=
  x > 0 ∧ x ≠ 1 ∧ y > 0 ∧ y ≠ 1 ∧
  x + y = 12 ∧
  2 * (2 * (Real.log x / Real.log (y^2)) - Real.log y / Real.log (1/x)) = 5

theorem unique_solutions :
  ∀ x y : ℝ, system_solution x y ↔ ((x = 9 ∧ y = 3) ∨ (x = 3 ∧ y = 9)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l918_91842


namespace NUMINAMATH_CALUDE_faster_train_speed_l918_91865

/-- The speed of the faster train given the conditions of the problem -/
theorem faster_train_speed 
  (slower_speed : ℝ) 
  (slower_length : ℝ) 
  (faster_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : slower_speed = 90) 
  (h2 : slower_length = 1.10) 
  (h3 : faster_length = 0.90) 
  (h4 : crossing_time = 24 / 3600) : 
  ∃ faster_speed : ℝ, 
    faster_speed = 210 ∧ 
    (slower_length + faster_length) / crossing_time = faster_speed + slower_speed :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l918_91865


namespace NUMINAMATH_CALUDE_triangular_trip_distance_l918_91881

theorem triangular_trip_distance 
  (XY XZ YZ : ℝ) 
  (h1 : XY = 5000) 
  (h2 : XZ = 4000) 
  (h3 : YZ * YZ = XY * XY - XZ * XZ) : 
  XY + YZ + XZ = 12000 := by
sorry

end NUMINAMATH_CALUDE_triangular_trip_distance_l918_91881


namespace NUMINAMATH_CALUDE_swimmers_speed_l918_91866

/-- The speed of a swimmer in still water, given:
  1. The speed of the water current is 2 km/h.
  2. The swimmer takes 1.5 hours to swim 3 km against the current. -/
theorem swimmers_speed (current_speed : ℝ) (swim_time : ℝ) (swim_distance : ℝ) 
  (h1 : current_speed = 2)
  (h2 : swim_time = 1.5)
  (h3 : swim_distance = 3)
  (h4 : swim_distance = (swimmer_speed - current_speed) * swim_time) :
  swimmer_speed = 4 :=
by
  sorry

#check swimmers_speed

end NUMINAMATH_CALUDE_swimmers_speed_l918_91866


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_and_validity_l918_91826

/-- Given a triangle with sides a, b, c, and a similar triangle with sides k*a, k*b, k*c and perimeter p,
    prove that the longest side of the similar triangle is 60 and that it forms a valid triangle. -/
theorem similar_triangle_longest_side_and_validity 
  (a b c : ℝ) 
  (h_original : a = 8 ∧ b = 10 ∧ c = 12) 
  (k : ℝ) 
  (h_perimeter : k*a + k*b + k*c = 150) :
  (max (k*a) (max (k*b) (k*c)) = 60) ∧ 
  (k*a + k*b > k*c ∧ k*a + k*c > k*b ∧ k*b + k*c > k*a) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_and_validity_l918_91826


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l918_91893

theorem shaded_area_between_circles (d_small : ℝ) (r_large : ℝ) :
  d_small = 6 →
  r_large = 5 * (d_small / 2) →
  let r_small := d_small / 2
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  area_large - area_small = 216 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l918_91893
