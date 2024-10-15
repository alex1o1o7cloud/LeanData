import Mathlib

namespace NUMINAMATH_CALUDE_min_value_xy_min_value_xy_achieved_l2580_258051

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x*y ≥ 18 := by
  sorry

theorem min_value_xy_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + y + 6 = x*y ∧ x*y < 18 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_min_value_xy_achieved_l2580_258051


namespace NUMINAMATH_CALUDE_no_solution_equation_l2580_258080

theorem no_solution_equation (x : ℝ) : 
  (4 * x - 1) / 6 - (5 * x - 2/3) / 10 + (9 - x/2) / 3 ≠ 101/20 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2580_258080


namespace NUMINAMATH_CALUDE_buccaneer_loot_sum_l2580_258013

def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

theorem buccaneer_loot_sum : 
  let pearls := base5ToBase10 [1, 2, 3, 4]
  let silk := base5ToBase10 [1, 1, 1, 1]
  let spices := base5ToBase10 [1, 2, 2]
  let maps := base5ToBase10 [0, 1]
  pearls + silk + spices + maps = 808 := by sorry

end NUMINAMATH_CALUDE_buccaneer_loot_sum_l2580_258013


namespace NUMINAMATH_CALUDE_roxanne_sandwiches_l2580_258094

theorem roxanne_sandwiches (lemonade_price : ℚ) (sandwich_price : ℚ) 
  (lemonade_count : ℕ) (paid : ℚ) (change : ℚ) :
  lemonade_price = 2 →
  sandwich_price = 5/2 →
  lemonade_count = 2 →
  paid = 20 →
  change = 11 →
  (paid - change - lemonade_price * lemonade_count) / sandwich_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_roxanne_sandwiches_l2580_258094


namespace NUMINAMATH_CALUDE_profit_share_ratio_l2580_258084

theorem profit_share_ratio (total_profit : ℚ) (difference : ℚ) 
  (h1 : total_profit = 1000)
  (h2 : difference = 200) :
  ∃ (x y : ℚ), x + y = total_profit ∧ x - y = difference ∧ y / total_profit = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l2580_258084


namespace NUMINAMATH_CALUDE_expression_values_l2580_258060

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -1 ∨ e = -5 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l2580_258060


namespace NUMINAMATH_CALUDE_custom_op_eight_twelve_l2580_258007

/-- The custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b : ℚ) / (a + b + 1 : ℚ)

/-- Theorem stating that 8 @ 12 = 96/21 -/
theorem custom_op_eight_twelve : custom_op 8 12 = 96 / 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_eight_twelve_l2580_258007


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_implies_b_bound_l2580_258089

theorem quadratic_always_real_roots_implies_b_bound (b : ℝ) :
  (∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) →
  b ≤ -1/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_implies_b_bound_l2580_258089


namespace NUMINAMATH_CALUDE_triangle_properties_main_theorem_l2580_258032

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  cosQ : ℝ

-- Define our specific triangle
def trianglePQR : RightTriangle where
  PQ := 15
  QR := 30  -- We'll prove this
  cosQ := 0.5

-- Theorem to prove QR = 30 and area = 225
theorem triangle_properties (t : RightTriangle) 
  (h1 : t.PQ = 15) 
  (h2 : t.cosQ = 0.5) : 
  t.QR = 30 ∧ (1/2 * t.PQ * t.QR) = 225 := by
  sorry

-- Main theorem combining all properties
theorem main_theorem : 
  trianglePQR.QR = 30 ∧ 
  (1/2 * trianglePQR.PQ * trianglePQR.QR) = 225 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_main_theorem_l2580_258032


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2580_258015

-- Define the function f(x) = 2x^2 + 4
def f (x : ℝ) : ℝ := 2 * x^2 + 4

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2580_258015


namespace NUMINAMATH_CALUDE_intersection_theorem_l2580_258036

/-- The line y = x + m intersects the circle x^2 + y^2 - 2x + 4y - 4 = 0 at two distinct points A and B. -/
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.1^2 + A.2^2 - 2*A.1 + 4*A.2 - 4 = 0) ∧
    (B.1^2 + B.2^2 - 2*B.1 + 4*B.2 - 4 = 0) ∧
    (A.2 = A.1 + m) ∧ (B.2 = B.1 + m)

/-- The circle with diameter AB passes through the origin. -/
def circle_passes_origin (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2 = 0)

/-- Main theorem about the intersection of the line and circle. -/
theorem intersection_theorem :
  (∀ m : ℝ, intersects_at_two_points m ↔ -3-3*Real.sqrt 2 < m ∧ m < -3+3*Real.sqrt 2) ∧
  (∀ m : ℝ, intersects_at_two_points m →
    (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_passes_origin A B) →
    (m = -4 ∨ m = 1)) :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2580_258036


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2580_258023

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon -/
def num_vertices : ℕ := 10

/-- The number of diagonals in a regular decagon -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs : ℕ := 595

/-- The number of ways to choose 4 vertices from the decagon that form a convex quadrilateral -/
def num_convex_quadrilaterals : ℕ := 210

/-- The probability that two randomly chosen diagonals in a regular decagon
    intersect inside the decagon and form a convex quadrilateral -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  (num_convex_quadrilaterals : ℚ) / num_diagonal_pairs = 210 / 595 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l2580_258023


namespace NUMINAMATH_CALUDE_negation_of_forall_square_geq_one_l2580_258038

theorem negation_of_forall_square_geq_one :
  ¬(∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ↔ ∃ x : ℝ, x ≥ 1 ∧ x^2 < 1 := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_square_geq_one_l2580_258038


namespace NUMINAMATH_CALUDE_painted_portion_is_five_eighths_additional_painting_needed_l2580_258004

-- Define the bridge as having a total length of 1
def bridge_length : ℝ := 1

-- Define the painted portion of the bridge
def painted_portion : ℝ → Prop := λ x => 
  -- The painted and unpainted portions sum to the total length
  x + (bridge_length - x) = bridge_length ∧
  -- If the painted portion increases by 30%, the unpainted portion decreases by 50%
  1.3 * x + 0.5 * (bridge_length - x) = bridge_length

-- Theorem: The painted portion is 5/8 of the bridge length
theorem painted_portion_is_five_eighths : 
  ∃ x : ℝ, painted_portion x ∧ x = 5/8 * bridge_length :=
sorry

-- Theorem: An additional 1/8 of the bridge length needs to be painted to have half the bridge painted
theorem additional_painting_needed : 
  ∃ x : ℝ, painted_portion x ∧ x + 1/8 * bridge_length = 1/2 * bridge_length :=
sorry

end NUMINAMATH_CALUDE_painted_portion_is_five_eighths_additional_painting_needed_l2580_258004


namespace NUMINAMATH_CALUDE_triangle_side_altitude_sum_l2580_258066

theorem triangle_side_altitude_sum (x y : ℝ) : 
  x < 75 →
  y < 28 →
  x * 60 = 75 * 28 →
  100 * y = 75 * 28 →
  x + y = 56 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_altitude_sum_l2580_258066


namespace NUMINAMATH_CALUDE_library_visitors_average_l2580_258071

/-- Calculates the average number of visitors per day for a month in a library --/
def averageVisitorsPerDay (
  daysInMonth : ℕ)
  (sundayVisitors : ℕ)
  (regularDayVisitors : ℕ)
  (publicHolidays : ℕ)
  (specialEvents : ℕ) : ℚ :=
  let sundayCount := (daysInMonth + 6) / 7
  let regularDays := daysInMonth - sundayCount - publicHolidays - specialEvents
  let totalVisitors := 
    sundayCount * sundayVisitors +
    regularDays * regularDayVisitors +
    publicHolidays * (2 * regularDayVisitors) +
    specialEvents * (3 * regularDayVisitors)
  (totalVisitors : ℚ) / daysInMonth

theorem library_visitors_average :
  averageVisitorsPerDay 30 510 240 2 1 = 308 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l2580_258071


namespace NUMINAMATH_CALUDE_haley_marbles_l2580_258018

/-- The number of boys in Haley's class who love to play marbles -/
def num_boys : ℕ := 11

/-- The number of marbles Haley gives to each boy -/
def marbles_per_boy : ℕ := 9

/-- Theorem stating the total number of marbles Haley had -/
theorem haley_marbles : num_boys * marbles_per_boy = 99 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_l2580_258018


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l2580_258009

/-- Given a hyperbola with equation x²/64 - y²/36 = 1 and foci F₁ and F₂,
    if P is a point on the hyperbola and |PF₁| = 17, then |PF₂| = 33 -/
theorem hyperbola_focal_distance (P F₁ F₂ : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2/64 - y^2/36 = 1) →  -- P is on the hyperbola
  (∃ c : ℝ, c > 0 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)) →  -- F₁ and F₂ are foci
  abs (P.1 - F₁.1) + abs (P.1 - F₁.2) = 17 →       -- |PF₁| = 17
  abs (P.1 - F₂.1) + abs (P.1 - F₂.2) = 33 :=      -- |PF₂| = 33
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l2580_258009


namespace NUMINAMATH_CALUDE_unique_solution_for_quadratic_equation_l2580_258037

/-- Given an equation (x+m)^2 - (x^2+n^2) = (m-n)^2 where m and n are unequal non-zero constants,
    prove that the unique solution for x in the form x = am + bn has a = 0 and b = -m + n. -/
theorem unique_solution_for_quadratic_equation 
  (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b : ℝ), ∀ (x : ℝ), 
    (x + m)^2 - (x^2 + n^2) = (m - n)^2 ↔ x = a*m + b*n ∧ a = 0 ∧ b = -m + n := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_quadratic_equation_l2580_258037


namespace NUMINAMATH_CALUDE_second_quadrant_characterization_l2580_258092

/-- The set of points in the second quadrant of the Cartesian coordinate system -/
def second_quadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

/-- Theorem stating that the second quadrant is equivalent to the set of points (x, y) where x < 0 and y > 0 -/
theorem second_quadrant_characterization :
  second_quadrant = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0} := by
sorry

end NUMINAMATH_CALUDE_second_quadrant_characterization_l2580_258092


namespace NUMINAMATH_CALUDE_baker_usual_bread_sales_l2580_258063

/-- Represents the baker's sales and pricing information -/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculates the difference between usual sales and today's sales -/
def sales_difference (s : BakerSales) : ℤ :=
  (s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price) -
  (s.today_pastries * s.pastry_price + s.today_bread * s.bread_price)

/-- Theorem stating that given the conditions, the baker usually sells 34 loaves of bread -/
theorem baker_usual_bread_sales :
  ∀ (s : BakerSales),
    s.usual_pastries = 20 ∧
    s.today_pastries = 14 ∧
    s.today_bread = 25 ∧
    s.pastry_price = 2 ∧
    s.bread_price = 4 ∧
    sales_difference s = 48 →
    s.usual_bread = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_baker_usual_bread_sales_l2580_258063


namespace NUMINAMATH_CALUDE_remaining_dimes_l2580_258045

def initial_dimes : ℕ := 5
def spent_dimes : ℕ := 2

theorem remaining_dimes : initial_dimes - spent_dimes = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_dimes_l2580_258045


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l2580_258000

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_and_perpendicular_properties 
  (a b c : Line) (y : Plane) :
  (∀ a b c, parallel a b → parallel b c → parallel a c) ∧
  (∀ a b, perpendicular a y → perpendicular b y → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l2580_258000


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2580_258001

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 15 →
  a * b + c + d = 78 →
  a * d + b * c = 160 →
  c * d = 96 →
  ∃ (max : ℝ), (∀ (a' b' c' d' : ℝ), 
    a' + b' = 15 →
    a' * b' + c' + d' = 78 →
    a' * d' + b' * c' = 160 →
    c' * d' = 96 →
    a'^2 + b'^2 + c'^2 + d'^2 ≤ max) ∧
  max = 717 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2580_258001


namespace NUMINAMATH_CALUDE_brad_books_this_month_l2580_258085

theorem brad_books_this_month (william_last_month : ℕ) (brad_last_month : ℕ) (william_total : ℕ) (brad_total : ℕ) :
  william_last_month = 6 →
  brad_last_month = 3 * william_last_month →
  william_total = brad_total + 4 →
  william_total = william_last_month + 2 * (brad_total - brad_last_month) →
  brad_total - brad_last_month = 16 := by
  sorry

end NUMINAMATH_CALUDE_brad_books_this_month_l2580_258085


namespace NUMINAMATH_CALUDE_integral_rational_function_l2580_258011

open Real

theorem integral_rational_function (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) (h3 : x ≠ -1) :
  (deriv fun x => 2*x + 4*log (abs x) + log (abs (x - 3)) - 2*log (abs (x + 1))) x =
  (2*x^3 - x^2 - 7*x - 12) / (x*(x-3)*(x+1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_rational_function_l2580_258011


namespace NUMINAMATH_CALUDE_F_is_integer_exists_valid_s_and_t_l2580_258059

/-- Given a four-digit number, swap the thousands and tens digits, and the hundreds and units digits -/
def swap_digits (n : Nat) : Nat :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * c + 100 * d + 10 * a + b

/-- The "wholehearted number" function -/
def F (n : Nat) : Nat :=
  (n + swap_digits n) / 101

/-- Theorem: F(n) is an integer for any four-digit number n -/
theorem F_is_integer (n : Nat) (h : 1000 ≤ n ∧ n < 10000) : ∃ k : Nat, F n = k := by
  sorry

/-- Helper function to check if a number is divisible by 8 -/
def is_divisible_by_8 (n : Int) : Prop :=
  ∃ k : Int, n = 8 * k

/-- Function to generate s given a and b -/
def s (a b : Nat) : Nat :=
  3800 + 10 * a + b

/-- Function to generate t given a and b -/
def t (a b : Nat) : Nat :=
  1000 * b + 100 * a + 13

/-- Theorem: There exist values of a and b such that 3F(t) - F(s) is divisible by 8 -/
theorem exists_valid_s_and_t :
  ∃ (a b : Nat), 1 ≤ a ∧ a ≤ 5 ∧ 5 ≤ b ∧ b ≤ 9 ∧ is_divisible_by_8 (3 * (F (t a b)) - (F (s a b))) := by
  sorry

end NUMINAMATH_CALUDE_F_is_integer_exists_valid_s_and_t_l2580_258059


namespace NUMINAMATH_CALUDE_train_length_l2580_258048

/-- The length of a train given its speed, time to cross a platform, and the platform's length -/
theorem train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : 
  speed = 90 * (1000 / 3600) → 
  time = 25 → 
  platform_length = 400.05 → 
  speed * time - platform_length = 224.95 := by sorry

end NUMINAMATH_CALUDE_train_length_l2580_258048


namespace NUMINAMATH_CALUDE_peanut_seed_germination_l2580_258052

/-- The probability of at least k successes in n independent Bernoulli trials -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

/-- The probability of exactly k successes in n independent Bernoulli trials -/
def prob_exactly (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem peanut_seed_germination :
  let n : ℕ := 4
  let k : ℕ := 2
  let p : ℝ := 4/5
  prob_at_least n k p = 608/625 := by sorry

end NUMINAMATH_CALUDE_peanut_seed_germination_l2580_258052


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2580_258096

theorem sine_cosine_inequality (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2580_258096


namespace NUMINAMATH_CALUDE_pats_calculation_l2580_258091

theorem pats_calculation (x : ℝ) : (x / 8 - 20 = 12) → (x * 8 + 20 = 2068) := by
  sorry

end NUMINAMATH_CALUDE_pats_calculation_l2580_258091


namespace NUMINAMATH_CALUDE_geometric_progression_sum_equality_l2580_258043

/-- Proves the equality for sums of geometric progression terms -/
theorem geometric_progression_sum_equality 
  (a q : ℝ) (n : ℕ) (h : q ≠ 1) :
  let S : ℕ → ℝ := λ k => a * (q^k - 1) / (q - 1)
  S n * (S (3*n) - S (2*n)) = (S (2*n) - S n)^2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_equality_l2580_258043


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2580_258067

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : ℕ, isThreeDigitBase8 n → n ≤ 774 ∨ ¬(7 ∣ base8ToDecimal n) :=
by sorry

#check greatest_3digit_base8_divisible_by_7

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2580_258067


namespace NUMINAMATH_CALUDE_smallest_number_l2580_258006

theorem smallest_number (a b c d : ℝ) : 
  a = -2024 → b = -2022 → c = -2022.5 → d = 0 →
  (a < -2023 ∧ b > -2023 ∧ c > -2023 ∧ d > -2023) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2580_258006


namespace NUMINAMATH_CALUDE_trailing_zeros_count_l2580_258081

/-- The number of trailing zeros in (10¹² - 25)² is 12 -/
theorem trailing_zeros_count : ∃ n : ℕ, n > 0 ∧ (10^12 - 25)^2 = n * 10^12 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_count_l2580_258081


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2580_258042

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 27)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 2) :
  badminton + tennis - total + neither = 11 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2580_258042


namespace NUMINAMATH_CALUDE_ratio_equality_l2580_258050

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - y) = (x - y) / z ∧ (x - y) / z = z / (x + y)) :
  x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2580_258050


namespace NUMINAMATH_CALUDE_aiden_sleep_fraction_l2580_258027

/-- Proves that 15 minutes is equal to 1/4 of an hour, given that an hour has 60 minutes. -/
theorem aiden_sleep_fraction (minutes_in_hour : ℕ) (aiden_sleep_minutes : ℕ) : 
  minutes_in_hour = 60 → aiden_sleep_minutes = 15 → 
  (aiden_sleep_minutes : ℚ) / minutes_in_hour = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_aiden_sleep_fraction_l2580_258027


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2580_258046

theorem arithmetic_sequence_terms (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 10 → aₙ = 150 → d = 5 → aₙ = a₁ + (n - 1) * d → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2580_258046


namespace NUMINAMATH_CALUDE_existence_of_irrational_term_l2580_258054

theorem existence_of_irrational_term (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n, (a (n + 1))^2 = a n + 1) :
  ∃ n, Irrational (a n) :=
sorry

end NUMINAMATH_CALUDE_existence_of_irrational_term_l2580_258054


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2580_258044

theorem system_of_equations_solution :
  let x : ℚ := -133 / 57
  let y : ℚ := 64 / 19
  (3 * x - 4 * y = -7) ∧ (7 * x - 3 * y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2580_258044


namespace NUMINAMATH_CALUDE_calligraphy_supplies_problem_l2580_258062

/-- Represents the unit price of a brush in yuan -/
def brush_price : ℝ := 6

/-- Represents the unit price of rice paper in yuan -/
def paper_price : ℝ := 0.4

/-- Represents the maximum number of brushes that can be purchased -/
def max_brushes : ℕ := 50

/-- Theorem stating the solution to the calligraphy supplies problem -/
theorem calligraphy_supplies_problem :
  /- Given conditions -/
  (40 * brush_price + 100 * paper_price = 280) ∧
  (30 * brush_price + 200 * paper_price = 260) ∧
  (∀ m : ℕ, m ≤ 200 → 
    m * brush_price + (200 - m) * paper_price ≤ 360 → 
    m ≤ max_brushes) →
  /- Conclusion -/
  brush_price = 6 ∧ paper_price = 0.4 ∧ max_brushes = 50 :=
by sorry

end NUMINAMATH_CALUDE_calligraphy_supplies_problem_l2580_258062


namespace NUMINAMATH_CALUDE_sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2_l2580_258008

theorem sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2 :
  (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - 2 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2_l2580_258008


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l2580_258099

theorem collinear_vectors_y_value (y : ℝ) : 
  let a : Fin 2 → ℝ := ![(-3), 1]
  let b : Fin 2 → ℝ := ![6, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l2580_258099


namespace NUMINAMATH_CALUDE_range_of_f_l2580_258055

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the domain
def domain : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem range_of_f : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | 1 ≤ y ∧ y ≤ 10} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2580_258055


namespace NUMINAMATH_CALUDE_courtney_marble_weight_l2580_258056

/-- The weight of Courtney's marble collection --/
def marbleCollectionWeight (firstJarCount : ℕ) (firstJarWeight : ℚ) 
  (secondJarWeight : ℚ) (thirdJarWeight : ℚ) : ℚ :=
  firstJarCount * firstJarWeight + 
  (2 * firstJarCount) * secondJarWeight + 
  (firstJarCount / 4) * thirdJarWeight

/-- Theorem stating the total weight of Courtney's marble collection --/
theorem courtney_marble_weight : 
  marbleCollectionWeight 80 (35/100) (45/100) (25/100) = 105 := by
  sorry

end NUMINAMATH_CALUDE_courtney_marble_weight_l2580_258056


namespace NUMINAMATH_CALUDE_count_acute_triangles_l2580_258020

/-- A triangle classification based on its angles -/
inductive TriangleType
  | Acute   : TriangleType
  | Right   : TriangleType
  | Obtuse  : TriangleType

/-- Represents a set of triangles -/
structure TriangleSet where
  total : Nat
  right : Nat
  obtuse : Nat

/-- Theorem: Given 7 triangles with 2 right angles and 3 obtuse angles, there are 2 acute triangles -/
theorem count_acute_triangles (ts : TriangleSet) :
  ts.total = 7 ∧ ts.right = 2 ∧ ts.obtuse = 3 →
  ts.total - ts.right - ts.obtuse = 2 := by
  sorry

#check count_acute_triangles

end NUMINAMATH_CALUDE_count_acute_triangles_l2580_258020


namespace NUMINAMATH_CALUDE_car_speed_problem_l2580_258097

/-- Given a car traveling for two hours with speeds x and 60 km/h, 
    prove that if the average speed is 102.5 km/h, then x must be 145 km/h. -/
theorem car_speed_problem (x : ℝ) :
  (x + 60) / 2 = 102.5 → x = 145 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2580_258097


namespace NUMINAMATH_CALUDE_exist_permutation_sum_all_nines_l2580_258017

/-- A function that checks if two natural numbers have the same digits (permutation) -/
def is_permutation (m n : ℕ) : Prop := sorry

/-- A function that checks if a natural number consists of all 9s -/
def all_nines (n : ℕ) : Prop := sorry

/-- Theorem stating the existence of two natural numbers satisfying the given conditions -/
theorem exist_permutation_sum_all_nines : 
  ∃ (m n : ℕ), is_permutation m n ∧ all_nines (m + n) := by sorry

end NUMINAMATH_CALUDE_exist_permutation_sum_all_nines_l2580_258017


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l2580_258021

theorem no_function_satisfies_condition : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l2580_258021


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2580_258034

/-- Given a point M with polar coordinates (6, 11π/6), 
    the Cartesian coordinates of the point symmetric to M 
    with respect to the y-axis are (-3√3, -3) -/
theorem symmetric_point_coordinates : 
  let r : ℝ := 6
  let θ : ℝ := 11 * π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (- x, y) = (-3 * Real.sqrt 3, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2580_258034


namespace NUMINAMATH_CALUDE_arithmetic_progression_perfect_squares_l2580_258028

theorem arithmetic_progression_perfect_squares :
  ∃ (a b c : ℤ),
    b - a = c - b ∧
    ∃ (x y z : ℤ),
      a + b = x^2 ∧
      a + c = y^2 ∧
      b + c = z^2 ∧
      a = 482 ∧
      b = 3362 ∧
      c = 6242 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_perfect_squares_l2580_258028


namespace NUMINAMATH_CALUDE_equation_solution_l2580_258053

theorem equation_solution : ∃! x : ℤ, 27474 + x + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2580_258053


namespace NUMINAMATH_CALUDE_circle_probability_l2580_258075

def total_figures : ℕ := 10
def triangle_count : ℕ := 4
def circle_count : ℕ := 3
def square_count : ℕ := 3

theorem circle_probability : 
  (circle_count : ℚ) / total_figures = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_circle_probability_l2580_258075


namespace NUMINAMATH_CALUDE_final_balance_calculation_l2580_258031

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def withdrawal : ℕ := 4

theorem final_balance_calculation : 
  initial_balance + deposit - withdrawal = 76 := by sorry

end NUMINAMATH_CALUDE_final_balance_calculation_l2580_258031


namespace NUMINAMATH_CALUDE_complex_power_sum_l2580_258022

theorem complex_power_sum (w : ℂ) (h : w + 1 / w = 2 * Real.cos (5 * π / 180)) :
  w^1000 + 1 / w^1000 = -(Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2580_258022


namespace NUMINAMATH_CALUDE_red_chips_probability_l2580_258069

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
| AllRed
| AllGreen

/-- Represents a hat with red and green chips -/
structure Hat :=
  (redChips : ℕ)
  (greenChips : ℕ)

/-- Represents the probability of an outcome -/
def probability (outcome : DrawOutcome) (hat : Hat) : ℚ :=
  sorry

theorem red_chips_probability (hat : Hat) :
  hat.redChips = 3 ∧ hat.greenChips = 3 →
  probability DrawOutcome.AllRed hat = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_red_chips_probability_l2580_258069


namespace NUMINAMATH_CALUDE_cobbler_charge_percentage_l2580_258047

theorem cobbler_charge_percentage (mold_cost : ℝ) (hourly_rate : ℝ) (hours_worked : ℝ) (total_paid : ℝ)
  (h1 : mold_cost = 250)
  (h2 : hourly_rate = 75)
  (h3 : hours_worked = 8)
  (h4 : total_paid = 730) :
  (1 - total_paid / (mold_cost + hourly_rate * hours_worked)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_charge_percentage_l2580_258047


namespace NUMINAMATH_CALUDE_solve_equation_l2580_258057

theorem solve_equation (a : ℚ) (h : a + a/4 - 1/2 = 10/5) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2580_258057


namespace NUMINAMATH_CALUDE_trivia_game_points_per_round_l2580_258039

theorem trivia_game_points_per_round 
  (total_points : ℕ) 
  (num_rounds : ℕ) 
  (h1 : total_points = 78) 
  (h2 : num_rounds = 26) : 
  total_points / num_rounds = 3 := by
sorry

end NUMINAMATH_CALUDE_trivia_game_points_per_round_l2580_258039


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_fraction_l2580_258088

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio_equals_fraction : 
  (factorial 6)^2 / (factorial 5 * factorial 7) = 100 / 101 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_fraction_l2580_258088


namespace NUMINAMATH_CALUDE_spherical_coordinates_negated_z_l2580_258016

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates
    (5, 3π/4, π/3), prove that the spherical coordinates of (x, y, -z) are (5, 3π/4, 2π/3) -/
theorem spherical_coordinates_negated_z 
  (x y z : ℝ) 
  (h1 : x = 5 * Real.sin (π/3) * Real.cos (3*π/4))
  (h2 : y = 5 * Real.sin (π/3) * Real.sin (3*π/4))
  (h3 : z = 5 * Real.cos (π/3)) :
  ∃ (ρ θ φ : ℝ), 
    ρ = 5 ∧ 
    θ = 3*π/4 ∧ 
    φ = 2*π/3 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    -z = ρ * Real.cos φ ∧
    ρ > 0 ∧ 
    0 ≤ θ ∧ θ < 2*π ∧
    0 ≤ φ ∧ φ ≤ π := by
  sorry

end NUMINAMATH_CALUDE_spherical_coordinates_negated_z_l2580_258016


namespace NUMINAMATH_CALUDE_constant_c_value_l2580_258061

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 4) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l2580_258061


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l2580_258083

theorem least_addition_for_divisibility (n : ℕ) (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  ∃! x : ℕ, x < a * b ∧ (n + x) % a = 0 ∧ (n + x) % b = 0 ∧
  ∀ y : ℕ, y < x → ((n + y) % a ≠ 0 ∨ (n + y) % b ≠ 0) :=
by sorry

theorem problem_solution : 
  let n := 1056
  let a := 27
  let b := 31
  ∃! x : ℕ, x < a * b ∧ (n + x) % a = 0 ∧ (n + x) % b = 0 ∧
  ∀ y : ℕ, y < x → ((n + y) % a ≠ 0 ∨ (n + y) % b ≠ 0) ∧
  x = 618 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_problem_solution_l2580_258083


namespace NUMINAMATH_CALUDE_find_number_l2580_258033

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_find_number_l2580_258033


namespace NUMINAMATH_CALUDE_specific_can_stack_total_l2580_258072

/-- Represents a stack of cans forming an arithmetic sequence -/
structure CanStack where
  bottom_layer : ℕ
  difference : ℕ
  top_layer : ℕ

/-- Calculates the number of layers in the stack -/
def num_layers (stack : CanStack) : ℕ :=
  (stack.bottom_layer - stack.top_layer) / stack.difference + 1

/-- Calculates the total number of cans in the stack -/
def total_cans (stack : CanStack) : ℕ :=
  let n := num_layers stack
  (n * (stack.bottom_layer + stack.top_layer)) / 2

/-- Theorem stating that a specific can stack contains 172 cans -/
theorem specific_can_stack_total :
  let stack : CanStack := { bottom_layer := 35, difference := 4, top_layer := 1 }
  total_cans stack = 172 := by
  sorry

end NUMINAMATH_CALUDE_specific_can_stack_total_l2580_258072


namespace NUMINAMATH_CALUDE_f_properties_l2580_258082

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

theorem f_properties :
  (∀ x ≠ 0, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2580_258082


namespace NUMINAMATH_CALUDE_least_three_digit_product_12_l2580_258029

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_product_12_l2580_258029


namespace NUMINAMATH_CALUDE_sum_of_exponents_2023_l2580_258090

/-- Represents 2023 as a sum of distinct powers of 2 -/
def representation_2023 : List ℕ :=
  [10, 9, 8, 7, 6, 5, 2, 1, 0]

/-- The sum of the exponents in the representation of 2023 -/
def sum_of_exponents : ℕ :=
  representation_2023.sum

/-- Checks if the representation is valid -/
def is_valid_representation (n : ℕ) (rep : List ℕ) : Prop :=
  n = (rep.map (fun x => 2^x)).sum ∧ rep.Nodup

theorem sum_of_exponents_2023 :
  is_valid_representation 2023 representation_2023 ∧
  sum_of_exponents = 48 := by
  sorry

#eval sum_of_exponents -- Should output 48

end NUMINAMATH_CALUDE_sum_of_exponents_2023_l2580_258090


namespace NUMINAMATH_CALUDE_li_cake_purchase_l2580_258064

theorem li_cake_purchase 
  (fruit_price : ℝ) 
  (chocolate_price : ℝ) 
  (total_spent : ℝ) 
  (average_price : ℝ)
  (h1 : fruit_price = 4.8)
  (h2 : chocolate_price = 6.6)
  (h3 : total_spent = 167.4)
  (h4 : average_price = 6.2) :
  ∃ (fruit_count chocolate_count : ℕ),
    fruit_count = 6 ∧ 
    chocolate_count = 21 ∧
    fruit_count * fruit_price + chocolate_count * chocolate_price = total_spent ∧
    (fruit_count + chocolate_count : ℝ) * average_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_li_cake_purchase_l2580_258064


namespace NUMINAMATH_CALUDE_sum_inequality_l2580_258014

theorem sum_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h6 : a ≥ b ∧ a ≥ c ∧ a ≥ d)
  (h7 : d ≤ b ∧ d ≤ c)
  (h8 : a * d = b * c) :
  a + d > b + c := by
sorry

end NUMINAMATH_CALUDE_sum_inequality_l2580_258014


namespace NUMINAMATH_CALUDE_mean_equality_problem_l2580_258087

theorem mean_equality_problem (y : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l2580_258087


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l2580_258024

theorem smallest_root_of_quadratic (x : ℝ) :
  9 * x^2 - 45 * x + 50 = 0 → x ≥ 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l2580_258024


namespace NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l2580_258019

theorem smallest_k_for_no_real_roots : ∃ k : ℤ, k = 3 ∧ 
  (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ m : ℤ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l2580_258019


namespace NUMINAMATH_CALUDE_lettuce_salads_per_plant_l2580_258065

theorem lettuce_salads_per_plant (total_salads : ℕ) (plants : ℕ) (loss_fraction : ℚ) : 
  total_salads = 12 →
  loss_fraction = 1/2 →
  plants = 8 →
  (total_salads / (1 - loss_fraction)) / plants = 3 := by
  sorry

end NUMINAMATH_CALUDE_lettuce_salads_per_plant_l2580_258065


namespace NUMINAMATH_CALUDE_initial_workers_count_l2580_258035

/-- The time it takes one person to complete the task -/
def total_time : ℕ := 40

/-- The time the initial group works -/
def initial_work_time : ℕ := 4

/-- The number of additional people joining -/
def additional_workers : ℕ := 2

/-- The time the expanded group works -/
def expanded_work_time : ℕ := 8

/-- Proves that the initial number of workers is 2 -/
theorem initial_workers_count : 
  ∃ (x : ℕ), 
    (initial_work_time * x + expanded_work_time * (x + additional_workers)) / total_time = 1 ∧ 
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l2580_258035


namespace NUMINAMATH_CALUDE_square_root_equation_l2580_258095

theorem square_root_equation (x : ℝ) : Real.sqrt (x - 3) = 10 → x = 103 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2580_258095


namespace NUMINAMATH_CALUDE_ellipse_and_tangents_l2580_258003

/-- An ellipse with given properties -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a^2 * (9/4) + b^2 = a^2 * b^2)  -- Passes through (1, 3/2)
  (h4 : a^2 - b^2 = (1/4) * a^2)  -- Eccentricity = 1/2

/-- The main theorem about the ellipse and tangent lines -/
theorem ellipse_and_tangents (C : Ellipse) :
  C.a = 2 ∧ C.b = Real.sqrt 3 ∧
  ∀ (r : ℝ) (hr : 0 < r ∧ r < 3/2),
    ∃ (k : ℝ), ∀ (M N : ℝ × ℝ),
      (M.1^2 / 4 + M.2^2 / 3 = 1) →
      (N.1^2 / 4 + N.2^2 / 3 = 1) →
      (∃ (k1 k2 : ℝ),
        (M.2 - 3/2 = k1 * (M.1 - 1)) ∧
        (N.2 - 3/2 = k2 * (N.1 - 1)) ∧
        ((k1 * (1 - M.1))^2 + (3/2 - M.2)^2 = r^2) ∧
        ((k2 * (1 - N.1))^2 + (3/2 - N.2)^2 = r^2) ∧
        k1 = -k2) →
      (N.2 - M.2) / (N.1 - M.1) = k ∧ k = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_tangents_l2580_258003


namespace NUMINAMATH_CALUDE_sum_of_roots_l2580_258086

theorem sum_of_roots (p q r : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 72 = 0)
  (hq : 27*q^3 - 243*q^2 + 729*q - 972 = 0)
  (hr : 3*r = 9) : 
  p + q + r = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2580_258086


namespace NUMINAMATH_CALUDE_percentage_problem_l2580_258073

theorem percentage_problem (x : ℝ) (h : 45 = 25 / 100 * x) : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2580_258073


namespace NUMINAMATH_CALUDE_inequality_proof_l2580_258079

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2580_258079


namespace NUMINAMATH_CALUDE_elective_schemes_count_l2580_258026

/-- The number of courses offered -/
def total_courses : ℕ := 9

/-- The number of mutually exclusive courses -/
def exclusive_courses : ℕ := 3

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 4

/-- The number of different elective schemes -/
def elective_schemes : ℕ := 75

theorem elective_schemes_count :
  (Nat.choose exclusive_courses 1 * Nat.choose (total_courses - exclusive_courses) (courses_to_choose - 1)) +
  (Nat.choose (total_courses - exclusive_courses) courses_to_choose) = elective_schemes :=
by sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l2580_258026


namespace NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l2580_258041

-- Define the cost of one piece of gum in cents
def cost_per_gum : ℕ := 5

-- Define the number of pieces of gum
def num_gums : ℕ := 2000

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_gums_in_dollars : 
  (cost_per_gum * num_gums) / cents_per_dollar = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l2580_258041


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l2580_258070

/-- The function f(x) = ax² + 2x - 3 is monotonically increasing on (-∞, 4) iff -1/4 ≤ a ≤ 0 -/
theorem monotonic_quadratic (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → a * x^2 + 2 * x - 3 < a * y^2 + 2 * y - 3) ↔
  -1/4 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l2580_258070


namespace NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l2580_258093

def purchase1 : ℚ := 1.98
def purchase2 : ℚ := 5.04
def purchase3 : ℚ := 9.89

def roundToNearestInteger (x : ℚ) : ℤ :=
  if x - ↑(Int.floor x) < 1/2 then Int.floor x else Int.ceil x

theorem total_rounded_to_nearest_dollar :
  roundToNearestInteger (purchase1 + purchase2 + purchase3) = 17 := by sorry

end NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l2580_258093


namespace NUMINAMATH_CALUDE_all_propositions_true_l2580_258049

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop :=
  x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Define the converse proposition
def converse_proposition (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Define the inverse proposition
def inverse_proposition (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0

-- Define the contrapositive proposition
def contrapositive_proposition (x y : ℝ) : Prop :=
  x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0

-- Theorem stating that all propositions are true
theorem all_propositions_true :
  ∀ x y : ℝ,
    original_proposition x y ∧
    converse_proposition x y ∧
    inverse_proposition x y ∧
    contrapositive_proposition x y :=
by
  sorry


end NUMINAMATH_CALUDE_all_propositions_true_l2580_258049


namespace NUMINAMATH_CALUDE_only_cylinder_has_rectangular_front_view_l2580_258010

-- Define the solid figures
inductive SolidFigure
  | Cylinder
  | TriangularPyramid
  | Sphere
  | Cone

-- Define the front view shapes
inductive FrontViewShape
  | Rectangle
  | Triangle
  | Circle

-- Function to determine the front view shape of a solid figure
def frontViewShape (figure : SolidFigure) : FrontViewShape :=
  match figure with
  | SolidFigure.Cylinder => FrontViewShape.Rectangle
  | SolidFigure.TriangularPyramid => FrontViewShape.Triangle
  | SolidFigure.Sphere => FrontViewShape.Circle
  | SolidFigure.Cone => FrontViewShape.Triangle

-- Theorem stating that only the cylinder has a rectangular front view
theorem only_cylinder_has_rectangular_front_view :
  ∀ (figure : SolidFigure),
    frontViewShape figure = FrontViewShape.Rectangle ↔ figure = SolidFigure.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_has_rectangular_front_view_l2580_258010


namespace NUMINAMATH_CALUDE_john_payment_first_year_l2580_258077

/- Define the family members -/
inductive FamilyMember
| John
| Wife
| Son
| Daughter

/- Define whether a family member is extended or not -/
def isExtended : FamilyMember → Bool
  | FamilyMember.Wife => true
  | _ => false

/- Define the initial membership fee -/
def initialMembershipFee : ℕ := 4000

/- Define the monthly cost for each family member -/
def monthlyCost : FamilyMember → ℕ
  | FamilyMember.John => 1000
  | FamilyMember.Wife => 1200
  | FamilyMember.Son => 800
  | FamilyMember.Daughter => 900

/- Define the membership fee discount rate for extended family members -/
def membershipDiscountRate : ℚ := 1/5

/- Define the monthly fee discount rate for extended family members -/
def monthlyDiscountRate : ℚ := 1/10

/- Define the number of months in a year -/
def monthsInYear : ℕ := 12

/- Define John's payment fraction -/
def johnPaymentFraction : ℚ := 1/2

/- Theorem statement -/
theorem john_payment_first_year :
  let totalCost := (FamilyMember.John :: FamilyMember.Wife :: FamilyMember.Son :: FamilyMember.Daughter :: []).foldl
    (fun acc member =>
      let membershipFee := if isExtended member then initialMembershipFee * (1 - membershipDiscountRate) else initialMembershipFee
      let monthlyFee := if isExtended member then monthlyCost member * (1 - monthlyDiscountRate) else monthlyCost member
      acc + membershipFee + monthlyFee * monthsInYear)
    0
  johnPaymentFraction * totalCost = 30280 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_first_year_l2580_258077


namespace NUMINAMATH_CALUDE_toy_cost_price_l2580_258074

/-- The cost price of a toy -/
def cost_price : ℕ := sorry

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The total selling price of all toys -/
def total_selling_price : ℕ := 23100

/-- The number of toys whose cost price equals the gain -/
def gain_equivalent_toys : ℕ := 3

theorem toy_cost_price : 
  (toys_sold + gain_equivalent_toys) * cost_price = total_selling_price ∧ 
  cost_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2580_258074


namespace NUMINAMATH_CALUDE_no_consecutive_sum_for_2004_l2580_258068

theorem no_consecutive_sum_for_2004 :
  ¬ ∃ (n : ℕ) (a : ℕ), n > 1 ∧ n * (2 * a + n - 1) = 4008 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_sum_for_2004_l2580_258068


namespace NUMINAMATH_CALUDE_black_squares_in_58th_row_l2580_258078

/-- Represents a square color in the stair-step figure -/
inductive SquareColor
| White
| Black
| Red

/-- Represents a row in the stair-step figure -/
def StairRow := List SquareColor

/-- Generates a row of the stair-step figure -/
def generateRow (n : ℕ) : StairRow :=
  sorry

/-- Counts the number of black squares in a row -/
def countBlackSquares (row : StairRow) : ℕ :=
  sorry

/-- Main theorem: The number of black squares in the 58th row is 38 -/
theorem black_squares_in_58th_row :
  countBlackSquares (generateRow 58) = 38 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_in_58th_row_l2580_258078


namespace NUMINAMATH_CALUDE_square_plot_area_l2580_258005

/-- Given a square plot with fencing cost of 58 Rs per foot and total fencing cost of 1160 Rs,
    the area of the plot is 25 square feet. -/
theorem square_plot_area (side_length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  perimeter = 4 * side_length →
  58 * perimeter = 1160 →
  area = side_length ^ 2 →
  area = 25 := by
sorry

end NUMINAMATH_CALUDE_square_plot_area_l2580_258005


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2580_258040

theorem min_reciprocal_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2580_258040


namespace NUMINAMATH_CALUDE_triangle_bisector_inequality_l2580_258098

/-- Given a triangle ABC with side lengths a, b, c, semiperimeter p, circumradius R,
    inradius r, and angle bisector lengths l_a, l_b, l_c, prove that
    l_a * l_b + l_b * l_c + l_c * l_a ≤ p * √(3r² + 12Rr) -/
theorem triangle_bisector_inequality
  (a b c : ℝ)
  (p : ℝ)
  (R r : ℝ)
  (l_a l_b l_c : ℝ)
  (h_p : p = (a + b + c) / 2)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_R : R > 0)
  (h_r : r > 0)
  (h_l_a : l_a > 0)
  (h_l_b : l_b > 0)
  (h_l_c : l_c > 0) :
  l_a * l_b + l_b * l_c + l_c * l_a ≤ p * Real.sqrt (3 * r^2 + 12 * R * r) :=
by sorry

end NUMINAMATH_CALUDE_triangle_bisector_inequality_l2580_258098


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2580_258002

/-- Given a geometric sequence {a_n} with common ratio q, prove that if 16a_1, 4a_2, and a_3 form an arithmetic sequence, then q = 4 -/
theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (16 * a 1 + a 3 = 2 * (4 * a 2)) →  -- 16a_1, 4a_2, a_3 form an arithmetic sequence
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2580_258002


namespace NUMINAMATH_CALUDE_correct_head_start_for_dead_heat_l2580_258012

/-- The fraction of the race length that runner a should give as a head start to runner b -/
def head_start_fraction (speed_ratio : ℚ) : ℚ :=
  1 - (1 / speed_ratio)

/-- Theorem stating the correct head start fraction for the given speed ratio -/
theorem correct_head_start_for_dead_heat (race_length : ℚ) (speed_a speed_b : ℚ) 
  (h_speed : speed_a = 16/15 * speed_b) (h_positive : speed_b > 0) :
  head_start_fraction (speed_a / speed_b) * race_length = 1/16 * race_length :=
by sorry

end NUMINAMATH_CALUDE_correct_head_start_for_dead_heat_l2580_258012


namespace NUMINAMATH_CALUDE_wheel_probability_l2580_258030

theorem wheel_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_G = 1/6 → 
  p_D + p_E + p_F + p_G = 1 →
  p_F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l2580_258030


namespace NUMINAMATH_CALUDE_store_holiday_customers_l2580_258058

/-- The number of customers entering a store during holiday season -/
def holiday_customers (normal_rate : ℕ) (hours : ℕ) : ℕ :=
  2 * normal_rate * hours

/-- Theorem: Given the conditions, the store will see 2800 customers in 8 hours during the holiday season -/
theorem store_holiday_customers :
  holiday_customers 175 8 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_store_holiday_customers_l2580_258058


namespace NUMINAMATH_CALUDE_total_employees_calculation_l2580_258025

/-- Represents the number of employees in different categories and calculates the total full-time equivalents -/
def calculate_total_employees (part_time : ℕ) (full_time : ℕ) (remote : ℕ) (temporary : ℕ) : ℕ :=
  let hours_per_fte : ℕ := 40
  let total_hours : ℕ := part_time + full_time * hours_per_fte + remote * hours_per_fte + temporary * hours_per_fte
  (total_hours + hours_per_fte / 2) / hours_per_fte

/-- Theorem stating that given the specified number of employees in each category, 
    the total number of full-time equivalent employees is 76,971 -/
theorem total_employees_calculation :
  calculate_total_employees 2041 63093 5230 8597 = 76971 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_calculation_l2580_258025


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2580_258076

/-- Given an arithmetic sequence with first term 6, last term 154, and common difference 4,
    prove that the number of terms is 38. -/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ),
    a = 6 →
    d = 4 →
    last = 154 →
    last = a + (n - 1) * d →
    n = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2580_258076
