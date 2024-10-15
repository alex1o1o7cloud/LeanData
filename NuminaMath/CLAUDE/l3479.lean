import Mathlib

namespace NUMINAMATH_CALUDE_inverse_mod_89_l3479_347953

theorem inverse_mod_89 (h : (9⁻¹ : ZMod 89) = 79) : (81⁻¹ : ZMod 89) = 11 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_89_l3479_347953


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3479_347952

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 3 (1/3) + 2 →
  x^3 - 6*x^2 + 12*x - 11 = 0 := by sorry

#check cubic_polynomial_root

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3479_347952


namespace NUMINAMATH_CALUDE_miles_on_wednesday_l3479_347930

/-- Represents the miles run by Mrs. Hilt on different days of the week -/
structure RunningWeek where
  monday : ℕ
  wednesday : ℕ
  friday : ℕ
  total : ℕ

/-- Theorem stating that Mrs. Hilt ran 2 miles on Wednesday -/
theorem miles_on_wednesday (week : RunningWeek) 
  (h1 : week.monday = 3)
  (h2 : week.friday = 7)
  (h3 : week.total = 12)
  : week.wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_miles_on_wednesday_l3479_347930


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3479_347928

theorem quadratic_root_transformation (p q r u v : ℝ) : 
  (p * u^2 + q * u + r = 0) → 
  (p * v^2 + q * v + r = 0) → 
  ((2*p*u + q)^2 - q^2 + 4*p*r = 0) ∧ 
  ((2*p*v + q)^2 - q^2 + 4*p*r = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3479_347928


namespace NUMINAMATH_CALUDE_shop_weekly_earnings_value_l3479_347908

/-- Represents the shop's weekly earnings calculation -/
def shop_weekly_earnings : ℝ :=
  let open_minutes : ℕ := 12 * 60
  let womens_tshirts_sold : ℕ := open_minutes / 30
  let mens_tshirts_sold : ℕ := open_minutes / 40
  let womens_jeans_sold : ℕ := open_minutes / 45
  let mens_jeans_sold : ℕ := open_minutes / 60
  let unisex_hoodies_sold : ℕ := open_minutes / 70

  let daily_earnings : ℝ :=
    womens_tshirts_sold * 18 +
    mens_tshirts_sold * 15 +
    womens_jeans_sold * 40 +
    mens_jeans_sold * 45 +
    unisex_hoodies_sold * 35

  let wednesday_earnings : ℝ := daily_earnings * 0.9
  let saturday_earnings : ℝ := daily_earnings * 1.05
  let other_days_earnings : ℝ := daily_earnings * 5

  wednesday_earnings + saturday_earnings + other_days_earnings

theorem shop_weekly_earnings_value :
  shop_weekly_earnings = 15512.40 := by sorry

end NUMINAMATH_CALUDE_shop_weekly_earnings_value_l3479_347908


namespace NUMINAMATH_CALUDE_product_digits_sum_l3479_347992

/-- Converts a base-9 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a base-10 number to base-9 --/
def toBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Sums the digits of a number represented as a list of digits --/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

theorem product_digits_sum :
  let a := [1, 2, 5]  -- 125 in base 9
  let b := [3, 3]     -- 33 in base 9
  let product := toBase10 a * toBase10 b
  sumDigits (toBase9 product) = 16 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_l3479_347992


namespace NUMINAMATH_CALUDE_max_value_of_f_l3479_347920

-- Define the function
def f (x : ℝ) := abs (x^2 - 4) - 6*x

-- State the theorem
theorem max_value_of_f :
  ∃ (b : ℝ), b = 12 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → f x ≤ b) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 5 ∧ f x = b) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3479_347920


namespace NUMINAMATH_CALUDE_max_min_difference_z_l3479_347978

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 15) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w, (∃ u v, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 15) → w ≤ z_max) ∧
    (∀ w, (∃ u v, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 15) → w ≥ z_min) ∧
    z_max - z_min = 8 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l3479_347978


namespace NUMINAMATH_CALUDE_statement_1_statement_2_false_statement_3_statement_4_l3479_347954

-- Statement ①
theorem statement_1 (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc (2*a - 1) (a + 4), f x = a*x^2 + (2*a + b)*x + 2) :
  (∀ x ∈ Set.Icc (2*a - 1) (a + 4), f x = f (-x)) → b = 2 := by sorry

-- Statement ②
theorem statement_2_false : ∃ f : ℝ → ℝ, 
  (∀ x, f x = min (-2*x + 2) (-2*x^2 + 4*x + 2)) ∧ 
  (∃ x, f x > 1) := by sorry

-- Statement ③
theorem statement_3 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |2*x + a|) :
  (∀ x y, x ≥ 3 ∧ y > x → f x < f y) → a = -6 := by sorry

-- Statement ④
theorem statement_4 (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0) 
  (h2 : ∀ x y, f (x * y) = x * f y + y * f x) :
  ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_2_false_statement_3_statement_4_l3479_347954


namespace NUMINAMATH_CALUDE_binomial_expansion_equality_l3479_347981

theorem binomial_expansion_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  (45 : ℝ) * p^8 * q^2 = (120 : ℝ) * p^7 * q^3 → 
  p = 8/11 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_equality_l3479_347981


namespace NUMINAMATH_CALUDE_meaningful_range_l3479_347910

def is_meaningful (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -1 ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l3479_347910


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_given_pyramid_l3479_347937

/-- A right truncated quadrilateral pyramid -/
structure TruncatedPyramid where
  height : ℝ
  volume : ℝ
  base_ratio : ℝ × ℝ

/-- The lateral surface area of a truncated pyramid -/
def lateral_surface_area (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The given truncated pyramid -/
def given_pyramid : TruncatedPyramid :=
  { height := 3,
    volume := 38,
    base_ratio := (4, 9) }

/-- Theorem: The lateral surface area of the given truncated pyramid is 10√19 -/
theorem lateral_surface_area_of_given_pyramid :
  lateral_surface_area given_pyramid = 10 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_given_pyramid_l3479_347937


namespace NUMINAMATH_CALUDE_cobbler_weekly_shoes_l3479_347990

/-- The number of pairs of shoes a cobbler can mend per hour -/
def shoes_per_hour : ℕ := 3

/-- The number of hours the cobbler works from Monday to Thursday each day -/
def hours_per_day : ℕ := 8

/-- The number of days the cobbler works full hours (Monday to Thursday) -/
def full_days : ℕ := 4

/-- The number of hours the cobbler works on Friday -/
def friday_hours : ℕ := 3

/-- The total number of pairs of shoes the cobbler can mend in a week -/
def total_shoes : ℕ := shoes_per_hour * (hours_per_day * full_days + friday_hours)

theorem cobbler_weekly_shoes : total_shoes = 105 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_weekly_shoes_l3479_347990


namespace NUMINAMATH_CALUDE_limit_f_difference_quotient_l3479_347971

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem limit_f_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, 0 < |t| ∧ |t| < δ →
    |(f 2 - f (2 - 3*t)) / t + 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_f_difference_quotient_l3479_347971


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_is_nine_l3479_347980

/-- Calculates the number of flowers per bouquet given the initial number of seeds,
    the number of flowers killed, and the number of bouquets to be made. -/
def flowersPerBouquet (seedsPerColor : ℕ) (redKilled yellowKilled orangeKilled purpleKilled : ℕ)
    (numBouquets : ℕ) : ℕ :=
  let redSurvived := seedsPerColor - redKilled
  let yellowSurvived := seedsPerColor - yellowKilled
  let orangeSurvived := seedsPerColor - orangeKilled
  let purpleSurvived := seedsPerColor - purpleKilled
  let totalSurvived := redSurvived + yellowSurvived + orangeSurvived + purpleSurvived
  totalSurvived / numBouquets

/-- Theorem stating that the number of flowers per bouquet is 9 under the given conditions. -/
theorem flowers_per_bouquet_is_nine :
  flowersPerBouquet 125 45 61 30 40 36 = 9 := by
  sorry

#eval flowersPerBouquet 125 45 61 30 40 36

end NUMINAMATH_CALUDE_flowers_per_bouquet_is_nine_l3479_347980


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3479_347904

/-- Given a circle and a line with a specific chord length, prove the possible values of 'a' -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y + a)^2 = 4 ∧ x - y - 2 = 0) →  -- Circle and line equations
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + (y₁ + a)^2 = 4 ∧ 
    x₁ - y₁ - 2 = 0 ∧
    x₂^2 + (y₂ + a)^2 = 4 ∧ 
    x₂ - y₂ - 2 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →  -- Chord length condition
  a = 0 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3479_347904


namespace NUMINAMATH_CALUDE_train_passing_time_l3479_347924

/-- Two trains passing problem -/
theorem train_passing_time (length1 length2 : ℝ) (speed1 speed2 : ℝ) (h1 : length1 = 280)
    (h2 : length2 = 350) (h3 : speed1 = 72) (h4 : speed2 = 54) :
    (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 126 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3479_347924


namespace NUMINAMATH_CALUDE_product_of_A_and_B_l3479_347967

theorem product_of_A_and_B (A B : ℝ) (h1 : 3/9 = 6/A) (h2 : 6/A = B/63) : A * B = 378 := by
  sorry

end NUMINAMATH_CALUDE_product_of_A_and_B_l3479_347967


namespace NUMINAMATH_CALUDE_power_function_m_squared_minus_three_l3479_347983

/-- A function f(x) is a power function if it can be written as f(x) = ax^n, where a and n are constants and n ≠ 0. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), n ≠ 0 ∧ ∀ x, f x = a * x^n

/-- Given that y = (m^2 - 3)x^(2m) is a power function with respect to x, prove that m = ±2. -/
theorem power_function_m_squared_minus_three (m : ℝ) :
  is_power_function (λ x => (m^2 - 3) * x^(2*m)) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_squared_minus_three_l3479_347983


namespace NUMINAMATH_CALUDE_fraction_sum_cubes_l3479_347931

theorem fraction_sum_cubes : (5 / 6 : ℚ)^3 + (3 / 5 : ℚ)^3 = 21457 / 27000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_cubes_l3479_347931


namespace NUMINAMATH_CALUDE_chef_potato_problem_l3479_347921

/-- The number of potatoes a chef needs to cook -/
def total_potatoes (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  already_cooked + (remaining_cooking_time / cooking_time_per_potato)

/-- Proof that the chef needs to cook 12 potatoes in total -/
theorem chef_potato_problem : 
  total_potatoes 6 6 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_problem_l3479_347921


namespace NUMINAMATH_CALUDE_points_form_circle_l3479_347987

theorem points_form_circle :
  ∀ (t : ℝ), (∃ (x y : ℝ), x = Real.cos t ∧ y = Real.sin t) →
  ∃ (r : ℝ), x^2 + y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_points_form_circle_l3479_347987


namespace NUMINAMATH_CALUDE_train_length_problem_l3479_347959

/-- The length of a train given its speed and time to cross a fixed point. -/
def trainLength (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train traveling at 30 m/s that takes 12 seconds to cross a fixed point has a length of 360 meters. -/
theorem train_length_problem : trainLength 30 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l3479_347959


namespace NUMINAMATH_CALUDE_cookie_distribution_l3479_347950

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3479_347950


namespace NUMINAMATH_CALUDE_difference_local_face_value_65793_l3479_347917

/-- The difference between the local value and face value of a digit in a numeral -/
def local_face_value_difference (numeral : ℕ) (digit : ℕ) (place : ℕ) : ℕ :=
  digit * (10 ^ place) - digit

/-- The hundreds place in a decimal number system -/
def hundreds_place : ℕ := 2

theorem difference_local_face_value_65793 :
  local_face_value_difference 65793 7 hundreds_place = 693 := by
  sorry

end NUMINAMATH_CALUDE_difference_local_face_value_65793_l3479_347917


namespace NUMINAMATH_CALUDE_minimum_peanuts_l3479_347996

theorem minimum_peanuts (N A₁ A₂ A₃ A₄ A₅ : ℕ) : 
  N = 5 * A₁ + 1 ∧
  4 * A₁ = 5 * A₂ + 1 ∧
  4 * A₂ = 5 * A₃ + 1 ∧
  4 * A₃ = 5 * A₄ + 1 ∧
  4 * A₄ = 5 * A₅ + 1 →
  N ≥ 3121 ∧ (N = 3121 → 
    A₁ = 624 ∧ A₂ = 499 ∧ A₃ = 399 ∧ A₄ = 319 ∧ A₅ = 255) :=
by sorry

#check minimum_peanuts

end NUMINAMATH_CALUDE_minimum_peanuts_l3479_347996


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_four_l3479_347933

theorem sqrt_plus_square_zero_implies_diff_four (m n : ℝ) : 
  Real.sqrt (m - 3) + (n + 1)^2 = 0 → m - n = 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_four_l3479_347933


namespace NUMINAMATH_CALUDE_full_price_tickets_l3479_347995

theorem full_price_tickets (total : ℕ) (reduced : ℕ) (h1 : total = 25200) (h2 : reduced = 5400) :
  total - reduced = 19800 := by
  sorry

end NUMINAMATH_CALUDE_full_price_tickets_l3479_347995


namespace NUMINAMATH_CALUDE_distinct_roots_find_m_l3479_347991

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-2*m)^2 - 4*(m^2 - 9)

-- Theorem 1: The quadratic equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : discriminant m > 0 := by sorry

-- Define the roots of the quadratic equation
noncomputable def x₁ (m : ℝ) : ℝ := sorry
noncomputable def x₂ (m : ℝ) : ℝ := sorry

-- Theorem 2: When x₂ = 3x₁, m = ±6
theorem find_m : 
  ∃ m : ℝ, (x₂ m = 3 * x₁ m) ∧ (m = 6 ∨ m = -6) := by sorry

end NUMINAMATH_CALUDE_distinct_roots_find_m_l3479_347991


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3479_347962

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = min) ∧
    max = 18 ∧ min = -2 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3479_347962


namespace NUMINAMATH_CALUDE_parabola_and_line_theorem_l3479_347927

/-- A parabola with focus F and point A on it -/
structure Parabola where
  p : ℝ
  m : ℝ
  h : p > 0

/-- The distance from point A to the focus F is 5 -/
def distance_condition (par : Parabola) : Prop :=
  4 + par.p / 2 = 5

/-- Point A lies on the parabola -/
def point_on_parabola (par : Parabola) : Prop :=
  par.m^2 = 2 * par.p * 4

/-- m is positive -/
def m_positive (par : Parabola) : Prop :=
  par.m > 0

/-- A line that passes through point A -/
structure Line where
  k : ℝ
  b : ℝ

/-- The line intersects the parabola at exactly one point -/
def line_intersects_once (par : Parabola) (l : Line) : Prop :=
  (∀ x y, y = l.k * x + l.b → y^2 = 4 * x) →
  (∃! x, (l.k * x + l.b)^2 = 4 * x)

theorem parabola_and_line_theorem (par : Parabola) 
  (h1 : distance_condition par)
  (h2 : point_on_parabola par)
  (h3 : m_positive par) :
  (par.p = 2 ∧ par.m = 4) ∧
  (∃ l1 l2 : Line, 
    (l1.k = -2 ∧ l1.b = 4 ∧ line_intersects_once par l1) ∧
    (l2.k = 0 ∧ l2.b = 4 ∧ line_intersects_once par l2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_theorem_l3479_347927


namespace NUMINAMATH_CALUDE_fundraiser_total_l3479_347922

/-- Calculates the total money raised from a class fundraiser --/
def totalMoneyRaised (numStudentsBrownies : ℕ) (numBrowniesPerStudent : ℕ) 
                     (numStudentsCookies : ℕ) (numCookiesPerStudent : ℕ)
                     (numStudentsDonuts : ℕ) (numDonutsPerStudent : ℕ)
                     (priceBrownie : ℚ) (priceCookie : ℚ) (priceDonut : ℚ) : ℚ :=
  (numStudentsBrownies * numBrowniesPerStudent : ℚ) * priceBrownie +
  (numStudentsCookies * numCookiesPerStudent : ℚ) * priceCookie +
  (numStudentsDonuts * numDonutsPerStudent : ℚ) * priceDonut

theorem fundraiser_total : 
  totalMoneyRaised 50 20 30 36 25 18 (3/2) (9/4) 3 = 5280 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l3479_347922


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3479_347901

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions of the swimmer's journey, his speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed) 
  (h1 : effectiveSpeed s true * 5 = 45)  -- Downstream condition
  (h2 : effectiveSpeed s false * 5 = 25) -- Upstream condition
  : s.swimmer = 7 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3479_347901


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3479_347944

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 16 * Real.sqrt 13 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3479_347944


namespace NUMINAMATH_CALUDE_min_value_of_S_l3479_347926

theorem min_value_of_S (x y : ℝ) : 2 * x^2 - x*y + y^2 + 2*x + 3*y ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_S_l3479_347926


namespace NUMINAMATH_CALUDE_factor_implies_absolute_value_l3479_347947

theorem factor_implies_absolute_value (m n : ℤ) :
  (∀ x : ℝ, (x - 3) * (x + 4) ∣ (3 * x^3 - m * x + n)) →
  |3 * m - 2 * n| = 45 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_absolute_value_l3479_347947


namespace NUMINAMATH_CALUDE_unique_four_digit_products_l3479_347946

def digit_product (n : ℕ) : ℕ :=
  if n < 1000 ∨ n > 9999 then 0
  else (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def is_unique_product (n : ℕ) : Prop :=
  ∃! x : ℕ, x ≥ 1000 ∧ x ≤ 9999 ∧ digit_product x = n

theorem unique_four_digit_products :
  {n : ℕ | is_unique_product n} = {1, 625, 2401, 4096, 6561} :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_products_l3479_347946


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l3479_347902

/-- Triangle inequality condition -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proposition P: segments can form a triangle -/
def P (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Proposition Q: sum of squares inequality -/
def Q (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a)

/-- P is sufficient but not necessary for Q -/
theorem P_sufficient_not_necessary_Q :
  (∀ a b c : ℝ, P a b c → Q a b c) ∧
  (∃ a b c : ℝ, Q a b c ∧ ¬P a b c) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l3479_347902


namespace NUMINAMATH_CALUDE_markers_problem_l3479_347906

/-- Given the initial number of markers, the number of markers in each new box,
    and the final number of markers, prove that the number of new boxes bought is 6. -/
theorem markers_problem (initial_markers final_markers markers_per_box : ℕ)
  (h1 : initial_markers = 32)
  (h2 : final_markers = 86)
  (h3 : markers_per_box = 9) :
  (final_markers - initial_markers) / markers_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_markers_problem_l3479_347906


namespace NUMINAMATH_CALUDE_club_diamond_heart_probability_l3479_347900

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total : Nat)
  (clubs : Nat)
  (diamonds : Nat)
  (hearts : Nat)

/-- The probability of drawing the sequence: club, diamond, heart -/
def sequence_probability (d : Deck) : ℚ :=
  (d.clubs : ℚ) / d.total *
  (d.diamonds : ℚ) / (d.total - 1) *
  (d.hearts : ℚ) / (d.total - 2)

theorem club_diamond_heart_probability :
  let standard_deck : Deck := ⟨52, 13, 13, 13⟩
  sequence_probability standard_deck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_club_diamond_heart_probability_l3479_347900


namespace NUMINAMATH_CALUDE_fraction_product_l3479_347976

theorem fraction_product : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3479_347976


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3479_347948

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3479_347948


namespace NUMINAMATH_CALUDE_min_value_fraction_l3479_347911

theorem min_value_fraction (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  1 / x + 4 / (1 - x) ≥ 9 ∧
  (1 / x + 4 / (1 - x) = 9 ↔ x = 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3479_347911


namespace NUMINAMATH_CALUDE_original_room_population_l3479_347963

theorem original_room_population (x : ℚ) : 
  (x / 4 : ℚ) - (x / 12 : ℚ) = 15 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_original_room_population_l3479_347963


namespace NUMINAMATH_CALUDE_local_max_at_one_f_one_eq_zero_l3479_347956

/-- The function f(x) = x³ - 3x² + 2 has a local maximum value of 0 at x = 1 -/
theorem local_max_at_one (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 3*x^2 + 2) :
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≤ f 1 := by
  sorry

/-- The value of f(1) is 0 -/
theorem f_one_eq_zero (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 3*x^2 + 2) :
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_at_one_f_one_eq_zero_l3479_347956


namespace NUMINAMATH_CALUDE_equation_solution_l3479_347969

theorem equation_solution (a b : ℂ) (h1 : (2 : ℂ) * a ≠ 0) (h2 : (2 : ℂ) * a + (3 : ℂ) * b ≠ 0) 
  (h3 : ((2 : ℂ) * a + (3 : ℂ) * b) / ((2 : ℂ) * a) = ((3 : ℂ) * b) / ((2 : ℂ) * a + (3 : ℂ) * b)) :
  (a.im ≠ 0 ∧ b.im = 0) ∨ (a.im = 0 ∧ b.im ≠ 0) ∨ (a.im ≠ 0 ∧ b.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3479_347969


namespace NUMINAMATH_CALUDE_forgetful_scientist_rain_probability_l3479_347935

/-- The probability of taking an umbrella -/
def umbrella_probability : ℝ := 0.2

/-- The Forgetful Scientist scenario -/
structure ForgetfulScientist where
  /-- The probability of rain -/
  rain_prob : ℝ
  /-- The probability of having no umbrella at the destination -/
  no_umbrella_prob : ℝ
  /-- The condition that the Scientist takes an umbrella if it's raining or there's no umbrella -/
  umbrella_condition : umbrella_probability = rain_prob + no_umbrella_prob - rain_prob * no_umbrella_prob
  /-- The condition that the probabilities are between 0 and 1 -/
  prob_bounds : 0 ≤ rain_prob ∧ rain_prob ≤ 1 ∧ 0 ≤ no_umbrella_prob ∧ no_umbrella_prob ≤ 1

/-- The theorem stating that the probability of rain is 1/9 -/
theorem forgetful_scientist_rain_probability (fs : ForgetfulScientist) : fs.rain_prob = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_forgetful_scientist_rain_probability_l3479_347935


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l3479_347909

theorem coffee_shop_sales (coffee_customers : ℕ) (coffee_price : ℕ) 
  (tea_customers : ℕ) (tea_price : ℕ) : 
  coffee_customers = 7 → 
  coffee_price = 5 → 
  tea_customers = 8 → 
  tea_price = 4 → 
  coffee_customers * coffee_price + tea_customers * tea_price = 67 := by
sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l3479_347909


namespace NUMINAMATH_CALUDE_coefficient_x3y0_l3479_347964

/-- The coefficient of x^m * y^n in the expansion of (1+x)^6 * (1+y)^4 -/
def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem coefficient_x3y0 : f 3 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y0_l3479_347964


namespace NUMINAMATH_CALUDE_double_magic_result_l3479_347914

/-- Magic box function that takes two rational numbers and produces a new rational number -/
def magic_box (a b : ℚ) : ℚ := a^2 + b + 1

/-- The result of applying the magic box function twice -/
def double_magic (a b c : ℚ) : ℚ :=
  let m := magic_box a b
  magic_box m c

/-- Theorem stating that the double application of the magic box function
    with inputs (-2, 3) and then (m, 1) results in 66 -/
theorem double_magic_result : double_magic (-2) 3 1 = 66 := by
  sorry

end NUMINAMATH_CALUDE_double_magic_result_l3479_347914


namespace NUMINAMATH_CALUDE_pen_price_proof_l3479_347949

theorem pen_price_proof (total_cost : ℝ) (notebook_ratio : ℝ) :
  total_cost = 36.45 →
  notebook_ratio = 15 / 4 →
  ∃ (pen_price : ℝ),
    pen_price + 3 * (notebook_ratio * pen_price) = total_cost ∧
    pen_price = 5.4 :=
by sorry

end NUMINAMATH_CALUDE_pen_price_proof_l3479_347949


namespace NUMINAMATH_CALUDE_arithmetic_progression_cubes_l3479_347938

theorem arithmetic_progression_cubes (x y z : ℤ) : 
  x < y ∧ y < z ∧ y = (x + z) / 2 → ¬(y^3 = (x^3 + z^3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cubes_l3479_347938


namespace NUMINAMATH_CALUDE_chord_equation_l3479_347913

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 24

-- Define point P
def P : ℝ × ℝ := (1, -2)

-- Define a chord AB that passes through P and is bisected by P
structure Chord :=
  (A B : ℝ × ℝ)
  (passes_through_P : (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2)
  (on_ellipse : ellipse A.1 A.2 ∧ ellipse B.1 B.2)

-- Theorem statement
theorem chord_equation (AB : Chord) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ (x y : ℝ), 
    ((x, y) = AB.A ∨ (x, y) = AB.B) → a * x + b * y + c = 0) ∧
    a = 3 ∧ b = -2 ∧ c = -7 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l3479_347913


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3479_347940

theorem circle_center_and_radius :
  let equation := (fun (x y : ℝ) => x^2 + y^2 - 2*x - 5 = 0)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) ∧ 
    radius = Real.sqrt 6 ∧
    ∀ (x y : ℝ), equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3479_347940


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_l3479_347982

/-- The area of the shaded region in a rectangle with specific dimensions and unshaded triangles --/
theorem shaded_area_rectangle (rectangle_length : ℝ) (rectangle_width : ℝ)
  (triangle_base : ℝ) (triangle_height : ℝ) :
  rectangle_length = 12 →
  rectangle_width = 5 →
  triangle_base = 2 →
  triangle_height = 5 →
  rectangle_length * rectangle_width - 2 * (1/2 * triangle_base * triangle_height) = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_l3479_347982


namespace NUMINAMATH_CALUDE_max_lambda_inequality_l3479_347977

theorem max_lambda_inequality (a b x y : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0)
  (h_sum : a + b = 27) :
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 2916 * (a * x^2 * y + b * x * y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_max_lambda_inequality_l3479_347977


namespace NUMINAMATH_CALUDE_right_triangle_segment_relation_l3479_347988

/-- Given a right-angled triangle with legs of lengths a and b, and a segment of length d
    connecting the right angle vertex to the hypotenuse forming an angle δ with leg a,
    prove that 1/d = (cos δ)/a + (sin δ)/b. -/
theorem right_triangle_segment_relation (a b d : ℝ) (δ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hd : d > 0) (hδ : 0 < δ ∧ δ < π / 2) :
    1 / d = (Real.cos δ) / a + (Real.sin δ) / b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_relation_l3479_347988


namespace NUMINAMATH_CALUDE_largest_coin_distribution_l3479_347985

theorem largest_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 12 * k + 3) ∧ 
  n < 100 ∧ 
  (∀ m : ℕ, (∃ j : ℕ, m = 12 * j + 3) → m < 100 → m ≤ n) → 
  n = 99 := by
sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_l3479_347985


namespace NUMINAMATH_CALUDE_distribute_four_items_three_bags_l3479_347975

/-- The number of ways to distribute n distinct items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 14 ways to distribute 4 distinct items into 3 identical bags, allowing empty bags. -/
theorem distribute_four_items_three_bags : distribute 4 3 = 14 := by sorry

end NUMINAMATH_CALUDE_distribute_four_items_three_bags_l3479_347975


namespace NUMINAMATH_CALUDE_train_passenger_ratio_l3479_347923

theorem train_passenger_ratio :
  let initial_passengers : ℕ := 288
  let first_drop : ℕ := initial_passengers / 3
  let first_take : ℕ := 280
  let second_take : ℕ := 12
  let third_station_passengers : ℕ := 248
  
  let after_first_station : ℕ := initial_passengers - first_drop + first_take
  let dropped_second_station : ℕ := after_first_station - (third_station_passengers - second_take)
  let ratio : ℚ := dropped_second_station / after_first_station
  
  ratio = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_train_passenger_ratio_l3479_347923


namespace NUMINAMATH_CALUDE_adult_meal_cost_l3479_347934

/-- Calculates the cost of an adult meal given the total number of people,
    number of kids, and total cost for a group at a restaurant where kids eat free. -/
theorem adult_meal_cost (total_people : ℕ) (num_kids : ℕ) (total_cost : ℚ) :
  total_people = 9 →
  num_kids = 2 →
  total_cost = 14 →
  (total_cost / (total_people - num_kids : ℚ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l3479_347934


namespace NUMINAMATH_CALUDE_prop_1_prop_4_l3479_347903

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

variable (m n : Line)
variable (α β : Plane)

-- Proposition 1
theorem prop_1 (h1 : parallel_planes α β) (h2 : subset m α) :
  parallel_line_plane m β := by sorry

-- Proposition 4
theorem prop_4 (h1 : parallel_line_plane m β) (h2 : subset m α) (h3 : intersect α β n) :
  parallel_lines m n := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_4_l3479_347903


namespace NUMINAMATH_CALUDE_calculation_difference_l3479_347968

theorem calculation_difference : 
  (0.70 * 120 - ((6/9) * 150 / (0.80 * 250))) - (0.18 * 180 * (5/7) * 210) = -4776.5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l3479_347968


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l3479_347984

theorem clock_equivalent_hours : ∃ (n : ℕ), n > 6 ∧ n ≡ n^2 [ZMOD 24] ∧
  ∀ (m : ℕ), m > 6 ∧ m < n → ¬(m ≡ m^2 [ZMOD 24]) :=
by sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l3479_347984


namespace NUMINAMATH_CALUDE_distinct_values_in_sequence_l3479_347974

def is_valid_f (f : ℕ → ℕ) : Prop :=
  f 1 = 1 ∧
  (∀ a b : ℕ, 0 < a → 0 < b → a ≤ b → f a ≤ f b) ∧
  (∀ a : ℕ, 0 < a → f (2 * a) = f a + 1)

theorem distinct_values_in_sequence (f : ℕ → ℕ) (hf : is_valid_f f) :
  Finset.card (Finset.image f (Finset.range 2015)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_in_sequence_l3479_347974


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3479_347918

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 11))) →
  11 ∣ x →
  x = 59048 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l3479_347918


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3479_347958

theorem polynomial_factorization (x : ℝ) : x^3 + 2*x^2 - 3*x = x*(x+3)*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3479_347958


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3479_347994

theorem complex_equation_solution (z : ℂ) : 
  Complex.abs z - 2 * z = -1 + 8 * Complex.I → 
  z = 3 - 4 * Complex.I ∨ z = -5/3 - 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3479_347994


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3479_347966

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3479_347966


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3479_347989

/-- Theorem: When each edge of a cube is increased by p%, 
    the surface area of the cube is increased by 2p + (p^2/100)%. -/
theorem cube_surface_area_increase (p : ℝ) :
  let original_edge : ℝ → ℝ := λ s => s
  let increased_edge : ℝ → ℝ := λ s => s * (1 + p / 100)
  let original_surface_area : ℝ → ℝ := λ s => 6 * s^2
  let increased_surface_area : ℝ → ℝ := λ s => 6 * (increased_edge s)^2
  let percent_increase : ℝ → ℝ := λ s => 
    (increased_surface_area s - original_surface_area s) / original_surface_area s * 100
  ∀ s > 0, percent_increase s = 2 * p + p^2 / 100 :=
by sorry


end NUMINAMATH_CALUDE_cube_surface_area_increase_l3479_347989


namespace NUMINAMATH_CALUDE_vector_relationships_l3479_347929

/-- Given vector a and unit vector b, prove their parallel and perpendicular relationships -/
theorem vector_relationships (a b : ℝ × ℝ) :
  a = (3, 4) →
  norm b = 1 →
  (b.1 * a.2 = b.2 * a.1 → b = (3/5, 4/5) ∨ b = (-3/5, -4/5)) ∧
  (b.1 * a.1 + b.2 * a.2 = 0 → b = (-4/5, 3/5) ∨ b = (4/5, -3/5)) :=
by sorry

end NUMINAMATH_CALUDE_vector_relationships_l3479_347929


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3479_347972

/-- Quadratic function y = ax² - 4ax + 3a -/
def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 * a

theorem quadratic_function_properties :
  (∀ x, quadratic_function 1 x ≥ -1) ∧
  (∃ x, quadratic_function 1 x = -1) ∧
  (∀ x ∈ Set.Icc 1 4, quadratic_function (4/3) x ≤ 4) ∧
  (∃ x ∈ Set.Icc 1 4, quadratic_function (4/3) x = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3479_347972


namespace NUMINAMATH_CALUDE_remainder_13_pow_2031_mod_100_l3479_347999

theorem remainder_13_pow_2031_mod_100 : 13^2031 % 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2031_mod_100_l3479_347999


namespace NUMINAMATH_CALUDE_isosceles_tetrahedron_ratio_bounds_l3479_347943

/-- An isosceles tetrahedron with edge lengths a, b, and c. -/
structure IsoscelesTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The circumradius of the tetrahedron. -/
noncomputable def R (t : IsoscelesTetrahedron) : ℝ :=
  sorry

/-- The circumradius of the base triangle. -/
noncomputable def r (t : IsoscelesTetrahedron) : ℝ :=
  sorry

/-- The theorem stating the bounds of the ratio r/R. -/
theorem isosceles_tetrahedron_ratio_bounds (t : IsoscelesTetrahedron) :
    2 * Real.sqrt 2 / 3 ≤ r t / R t ∧ r t / R t < 1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_tetrahedron_ratio_bounds_l3479_347943


namespace NUMINAMATH_CALUDE_triangle_median_equality_l3479_347907

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the length function
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_median_equality (t : Triangle) :
  length t.P t.Q = 2 →
  length t.P t.R = 3 →
  length t.Q t.R = median t t.P →
  length t.Q t.R = Real.sqrt (26 * 0.2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_equality_l3479_347907


namespace NUMINAMATH_CALUDE_expression_equivalence_l3479_347942

theorem expression_equivalence (a : ℝ) : 
  (a^2 + a - 2) / (a^2 + 3*a + 2) * (5 * (a + 1)^2) = 5*a^2 - 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3479_347942


namespace NUMINAMATH_CALUDE_stable_painted_area_l3479_347932

/-- Calculates the total area to be painted for a rectangular stable with a chimney -/
def total_painted_area (width length height chim_width chim_length chim_height : ℝ) : ℝ :=
  let wall_area_1 := 2 * 2 * (width * height)
  let wall_area_2 := 2 * 2 * (length * height)
  let roof_area := width * length
  let ceiling_area := width * length
  let chimney_area := 4 * (chim_width * chim_height) + (chim_width * chim_length)
  wall_area_1 + wall_area_2 + roof_area + ceiling_area + chimney_area

/-- Theorem stating that the total area to be painted for the given stable is 1060 sq yd -/
theorem stable_painted_area :
  total_painted_area 12 15 6 2 2 2 = 1060 := by
  sorry

end NUMINAMATH_CALUDE_stable_painted_area_l3479_347932


namespace NUMINAMATH_CALUDE_die_throw_outcomes_l3479_347951

/-- Represents the number of sides on a fair cubic die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 4

/-- Represents the number of different outcomes required to stop -/
def differentOutcomes : ℕ := 3

/-- Calculates the total number of different outcomes for the die throws -/
def totalOutcomes : ℕ := numSides * (numSides - 1) * (numSides - 2) * differentOutcomes

theorem die_throw_outcomes :
  totalOutcomes = 270 :=
sorry

end NUMINAMATH_CALUDE_die_throw_outcomes_l3479_347951


namespace NUMINAMATH_CALUDE_all_can_be_top_l3479_347955

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- Defines the "not inferior" relation between two people -/
def notInferior (a b : Person) : Prop :=
  a.height ≥ b.height ∨ a.weight ≥ b.weight

/-- Defines a top person as someone who is not inferior to all others -/
def isTop (p : Person) (group : Finset Person) : Prop :=
  ∀ q ∈ group, p ≠ q → notInferior p q

/-- Theorem: It's possible to have 100 top people in a group of 100 -/
theorem all_can_be_top :
  ∃ (group : Finset Person), Finset.card group = 100 ∧
    ∀ p ∈ group, isTop p group := by
  sorry


end NUMINAMATH_CALUDE_all_can_be_top_l3479_347955


namespace NUMINAMATH_CALUDE_sum_of_three_squares_divisibility_l3479_347979

theorem sum_of_three_squares_divisibility (N : ℕ) :
  (∃ a b c : ℤ, (N : ℤ) = a^2 + b^2 + c^2 ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ x y z : ℤ, (N : ℤ) = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_divisibility_l3479_347979


namespace NUMINAMATH_CALUDE_gcf_of_18_and_10_l3479_347961

theorem gcf_of_18_and_10 (h : Nat.lcm 18 10 = 36) : Nat.gcd 18 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_18_and_10_l3479_347961


namespace NUMINAMATH_CALUDE_movies_to_watch_l3479_347912

theorem movies_to_watch (total_movies : ℕ) (watched_movies : ℕ) 
  (h1 : total_movies = 35) (h2 : watched_movies = 18) :
  total_movies - watched_movies = 17 := by
  sorry

end NUMINAMATH_CALUDE_movies_to_watch_l3479_347912


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3479_347997

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^((a + b)/2) = Real.sqrt 3 → 
  (1/a + 1/b) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3479_347997


namespace NUMINAMATH_CALUDE_max_value_of_f_l3479_347998

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 12*x + 16

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 3 → f y ≤ f x) ∧
  f x = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3479_347998


namespace NUMINAMATH_CALUDE_radical_simplification_l3479_347919

-- Define the statement
theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (11 * q) * Real.sqrt (8 * q^3) * Real.sqrt (9 * q^5) = 28 * q^4 * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l3479_347919


namespace NUMINAMATH_CALUDE_sum_abc_values_l3479_347939

theorem sum_abc_values (a b c : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1)
  (hab : b = a^2 / (2 - a^2))
  (hbc : c = b^2 / (2 - b^2))
  (hca : a = c^2 / (2 - c^2)) :
  (a + b + c = 6) ∨ (a + b + c = -4) ∨ (a + b + c = -6) := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_values_l3479_347939


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l3479_347957

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l3479_347957


namespace NUMINAMATH_CALUDE_k_range_theorem_l3479_347973

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - x * log x + 2

theorem k_range_theorem (a b : ℝ) (h1 : 1/2 ≤ a) (h2 : a < b) 
  (h3 : ∀ x ∈ Set.Icc a b, ∃ k : ℝ, f x = k * (x + 2)) :
  ∃ k : ℝ, 1 < k ∧ k ≤ (9 + 2 * log 2) / 10 :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l3479_347973


namespace NUMINAMATH_CALUDE_apples_per_pie_is_seven_l3479_347960

/-- Calculates the number of apples used per pie given the initial conditions -/
def apples_per_pie (
  total_apples : ℕ)
  (num_children : ℕ)
  (apples_per_child : ℕ)
  (num_pies : ℕ)
  (remaining_apples : ℕ) : ℕ :=
  let apples_for_teachers := num_children * apples_per_child
  let apples_for_pies := total_apples - apples_for_teachers - remaining_apples
  apples_for_pies / num_pies

/-- Proves that the number of apples used per pie is 7 under the given conditions -/
theorem apples_per_pie_is_seven :
  apples_per_pie 50 2 6 2 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_is_seven_l3479_347960


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3479_347970

theorem root_sum_theorem (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → 
  (β^3 - β - 1 = 0) → 
  (γ^3 - γ - 1 = 0) → 
  ((1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3479_347970


namespace NUMINAMATH_CALUDE_nolan_saving_months_l3479_347945

def monthly_savings : ℕ := 3000
def total_saved : ℕ := 36000

theorem nolan_saving_months :
  total_saved / monthly_savings = 12 :=
by sorry

end NUMINAMATH_CALUDE_nolan_saving_months_l3479_347945


namespace NUMINAMATH_CALUDE_solve_for_P_l3479_347925

theorem solve_for_P : ∃ P : ℝ, (P^3).sqrt = 81 * Real.rpow 81 (1/3) → P = Real.rpow 3 (32/9) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_P_l3479_347925


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3479_347916

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |5 - 2*x| < 3} = {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3479_347916


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l3479_347936

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) :
  selling_price = 720 →
  num_balls_sold = 20 →
  num_balls_loss = 5 →
  ∃ (cost_price : ℕ),
    cost_price * num_balls_sold - selling_price = cost_price * num_balls_loss ∧
    cost_price = 48 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l3479_347936


namespace NUMINAMATH_CALUDE_min_value_trig_fraction_min_value_is_one_l3479_347986

theorem min_value_trig_fraction (x : ℝ) :
  (Real.sin x)^5 + (Real.cos x)^5 + 1 ≥ (Real.sin x)^3 + (Real.cos x)^3 + 1 := by
  sorry

theorem min_value_is_one :
  ∀ x : ℝ, ((Real.sin x)^5 + (Real.cos x)^5 + 1) / ((Real.sin x)^3 + (Real.cos x)^3 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_fraction_min_value_is_one_l3479_347986


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3479_347965

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3479_347965


namespace NUMINAMATH_CALUDE_probability_of_divisor_of_12_l3479_347993

/-- An 8-sided die numbered from 1 to 8 -/
def Die := Finset.range 8

/-- The set of divisors of 12 that are less than or equal to 8 -/
def DivisorsOf12 : Finset ℕ := {1, 2, 3, 4, 6}

/-- The probability of rolling a divisor of 12 on an 8-sided die -/
def probability : ℚ := (DivisorsOf12.card : ℚ) / (Die.card : ℚ)

theorem probability_of_divisor_of_12 : probability = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_divisor_of_12_l3479_347993


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l3479_347941

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l3479_347941


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l3479_347905

theorem smallest_divisible_by_10_and_24 : ∃ n : ℕ, n > 0 ∧ n % 10 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 0 → m % 24 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_and_24_l3479_347905


namespace NUMINAMATH_CALUDE_log_fifty_equals_one_plus_log_five_l3479_347915

theorem log_fifty_equals_one_plus_log_five : Real.log 50 / Real.log 10 = 1 + Real.log 5 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_fifty_equals_one_plus_log_five_l3479_347915
