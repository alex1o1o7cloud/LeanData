import Mathlib

namespace bounce_count_is_seven_l1510_151065

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The number of bounces a ball makes before returning to a vertex -/
def num_bounces (t : Triangle) (Y : ℝ × ℝ) : ℕ := sorry

/-- The theorem stating that the number of bounces is 7 for the given triangle and point -/
theorem bounce_count_is_seven :
  let t : Triangle := { A := (0, 0), B := (7, 0), C := (7/2, 3*Real.sqrt 3/2) }
  let Y : ℝ × ℝ := (7/2, 3*Real.sqrt 3/2)
  num_bounces t Y = 7 := by sorry

end bounce_count_is_seven_l1510_151065


namespace office_absenteeism_l1510_151017

theorem office_absenteeism (p : ℕ) (x : ℚ) (h : 0 < p) :
  (1 / ((1 - x) * p) - 1 / p = 1 / (3 * p)) → x = 1 / 4 := by
  sorry

end office_absenteeism_l1510_151017


namespace razorback_shop_profit_l1510_151026

theorem razorback_shop_profit : 
  let tshirt_profit : ℕ := 67
  let jersey_profit : ℕ := 165
  let hat_profit : ℕ := 32
  let jacket_profit : ℕ := 245
  let tshirts_sold : ℕ := 74
  let jerseys_sold : ℕ := 156
  let hats_sold : ℕ := 215
  let jackets_sold : ℕ := 45
  (tshirt_profit * tshirts_sold + 
   jersey_profit * jerseys_sold + 
   hat_profit * hats_sold + 
   jacket_profit * jackets_sold) = 48603 :=
by sorry

end razorback_shop_profit_l1510_151026


namespace total_monthly_time_is_200_l1510_151033

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Returns true if the given day is a weekday -/
def is_weekday (d : Day) : Bool :=
  match d with
  | Day.Saturday | Day.Sunday => false
  | _ => true

/-- Returns the amount of TV time for a given day -/
def tv_time (d : Day) : Nat :=
  match d with
  | Day.Monday | Day.Wednesday | Day.Friday => 4
  | Day.Tuesday | Day.Thursday => 3
  | Day.Saturday | Day.Sunday => 5

/-- Returns the amount of piano practice time for a given day -/
def piano_time (d : Day) : Nat :=
  if is_weekday d then 2 else 3

/-- Calculates the total weekly TV time -/
def total_weekly_tv_time : Nat :=
  (tv_time Day.Monday) + (tv_time Day.Tuesday) + (tv_time Day.Wednesday) +
  (tv_time Day.Thursday) + (tv_time Day.Friday) + (tv_time Day.Saturday) +
  (tv_time Day.Sunday)

/-- Calculates the average daily TV time -/
def avg_daily_tv_time : Nat :=
  total_weekly_tv_time / 7

/-- Calculates the total weekly video game time -/
def total_weekly_video_game_time : Nat :=
  (avg_daily_tv_time / 2) * 3

/-- Calculates the total weekly piano time -/
def total_weekly_piano_time : Nat :=
  (piano_time Day.Monday) + (piano_time Day.Tuesday) + (piano_time Day.Wednesday) +
  (piano_time Day.Thursday) + (piano_time Day.Friday) + (piano_time Day.Saturday) +
  (piano_time Day.Sunday)

/-- Calculates the total weekly time for all activities -/
def total_weekly_time : Nat :=
  total_weekly_tv_time + total_weekly_video_game_time + total_weekly_piano_time

/-- The main theorem stating that the total monthly time is 200 hours -/
theorem total_monthly_time_is_200 :
  total_weekly_time * 4 = 200 := by
  sorry

end total_monthly_time_is_200_l1510_151033


namespace purely_imaginary_complex_number_l1510_151014

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)).im ≠ 0 ∧ 
  (Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)).re = 0 → 
  m = 5 := by sorry

end purely_imaginary_complex_number_l1510_151014


namespace inequality_equivalence_l1510_151044

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0

def solution_set (x : ℝ) : Prop :=
  x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ 7 < x

theorem inequality_equivalence :
  ∀ x : ℝ, inequality_solution x ↔ solution_set x := by sorry

end inequality_equivalence_l1510_151044


namespace constant_e_value_l1510_151002

theorem constant_e_value (x y e : ℝ) 
  (h1 : x / (2 * y) = 3 / e) 
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 25) : 
  e = 2 := by sorry

end constant_e_value_l1510_151002


namespace divisible_by_five_l1510_151078

theorem divisible_by_five (n : ℕ) : 5 ∣ (2^(4*n+1) + 3^(4*n+1)) := by
  sorry

end divisible_by_five_l1510_151078


namespace consecutive_cube_product_divisible_l1510_151049

theorem consecutive_cube_product_divisible (a : ℤ) : 
  504 ∣ ((a^3 - 1) * a^3 * (a^3 + 1)) := by
  sorry

end consecutive_cube_product_divisible_l1510_151049


namespace number_divided_by_16_equals_16_times_8_l1510_151082

theorem number_divided_by_16_equals_16_times_8 : 
  2048 / 16 = 16 * 8 := by sorry

end number_divided_by_16_equals_16_times_8_l1510_151082


namespace sine_equality_solution_l1510_151019

theorem sine_equality_solution (m : ℤ) : 
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (780 * π / 180) → 
  m = 60 ∨ m = 120 := by
  sorry

end sine_equality_solution_l1510_151019


namespace pure_imaginary_complex_number_l1510_151025

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ) = (a^2 + 2*a - 3 : ℝ) + Complex.I * (a - 1) → a = -3 := by
  sorry

end pure_imaginary_complex_number_l1510_151025


namespace paco_cookies_eaten_l1510_151037

/-- Represents the number of cookies Paco ate -/
structure CookiesEaten where
  sweet : ℕ
  salty : ℕ

/-- Proves that if Paco ate 20 sweet cookies and 14 more salty cookies than sweet cookies,
    then he ate 34 salty cookies. -/
theorem paco_cookies_eaten (cookies : CookiesEaten) 
  (h1 : cookies.sweet = 20) 
  (h2 : cookies.salty = cookies.sweet + 14) : 
  cookies.salty = 34 := by
  sorry

#check paco_cookies_eaten

end paco_cookies_eaten_l1510_151037


namespace gcd_bound_for_special_numbers_l1510_151080

/-- Given two 2019-digit numbers a and b with specific non-zero digit patterns,
    prove that their greatest common divisor has at most 14 digits. -/
theorem gcd_bound_for_special_numbers (a b : ℕ) : 
  (∃ A B C D : ℕ,
    a = A * 10^2014 + B ∧ 
    b = C * 10^2014 + D ∧
    10^4 < A ∧ A < 10^5 ∧
    10^6 < B ∧ B < 10^7 ∧
    10^4 < C ∧ C < 10^5 ∧
    10^8 < D ∧ D < 10^9) →
  Nat.gcd a b < 10^14 :=
sorry

end gcd_bound_for_special_numbers_l1510_151080


namespace geometric_sequence_sum_l1510_151085

/-- Given a geometric sequence {a_n} with first term a₁ = 1 and positive common ratio q,
    if S₄ - 5S₂ = 0, then S₅ = 31. -/
theorem geometric_sequence_sum (q : ℝ) (hq : q > 0) : 
  let a : ℕ → ℝ := λ n => q^(n-1)
  let S : ℕ → ℝ := λ n => (1 - q^n) / (1 - q)
  (S 4 - 5 * S 2 = 0) → S 5 = 31 := by
sorry

end geometric_sequence_sum_l1510_151085


namespace matrix_determinant_zero_l1510_151083

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det !![1, a+b, b+c; 1, a+2*b, b+2*c; 1, a+3*b, b+3*c] = 0 := by
  sorry

end matrix_determinant_zero_l1510_151083


namespace student_in_first_vehicle_probability_l1510_151091

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of vehicles -/
def num_vehicles : ℕ := 2

/-- The number of seats in each vehicle -/
def seats_per_vehicle : ℕ := 2

/-- The probability that a specific student is in the first vehicle -/
def prob_student_in_first_vehicle : ℚ := 1/2

theorem student_in_first_vehicle_probability :
  prob_student_in_first_vehicle = 1/2 := by sorry

end student_in_first_vehicle_probability_l1510_151091


namespace functional_equation_solution_l1510_151028

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1) :=
by sorry

end functional_equation_solution_l1510_151028


namespace max_sum_theorem_l1510_151043

theorem max_sum_theorem (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_sq : x^2 + y^2 + z^2 + v^2 + w^2 = 2025) : 
  ∃ (N x_N y_N z_N v_N w_N : ℝ),
    (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2025 → 
      a*c + 3*b*c + 5*c*d + 2*c*e ≤ N) ∧
    x_N > 0 ∧ y_N > 0 ∧ z_N > 0 ∧ v_N > 0 ∧ w_N > 0 ∧
    x_N^2 + y_N^2 + z_N^2 + v_N^2 + w_N^2 = 2025 ∧
    x_N*z_N + 3*y_N*z_N + 5*z_N*v_N + 2*z_N*w_N = N ∧
    N + x_N + y_N + z_N + v_N + w_N = 55 + 3037.5 * Real.sqrt 13 + 5 * Real.sqrt 202.5 :=
by sorry

end max_sum_theorem_l1510_151043


namespace two_digit_numbers_from_123_l1510_151095

def Digits : Set Nat := {1, 2, 3}

def TwoDigitNumber (n : Nat) : Prop :=
  n ≥ 10 ∧ n ≤ 99

def FormedFromDigits (n : Nat) : Prop :=
  ∃ (tens units : Nat), tens ∈ Digits ∧ units ∈ Digits ∧ n = 10 * tens + units

theorem two_digit_numbers_from_123 :
  {n : Nat | TwoDigitNumber n ∧ FormedFromDigits n} =
  {11, 12, 13, 21, 22, 23, 31, 32, 33} := by sorry

end two_digit_numbers_from_123_l1510_151095


namespace tim_stored_26_bales_l1510_151079

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Tim stored 26 bales in the barn -/
theorem tim_stored_26_bales : bales_stored 28 54 = 26 := by
  sorry

end tim_stored_26_bales_l1510_151079


namespace quadratic_polynomial_functional_equation_l1510_151030

theorem quadratic_polynomial_functional_equation 
  (P : ℝ → ℝ) 
  (h_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) 
  (f : ℝ → ℝ) 
  (h_add : ∀ x y, f (x + y) = f x + f y) 
  (h_poly : ∀ x, f (P x) = f x) : 
  ∀ x, f x = 0 := by sorry

end quadratic_polynomial_functional_equation_l1510_151030


namespace leap_year_classification_l1510_151066

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

theorem leap_year_classification :
  let leap_years : Set ℕ := {1992, 2040}
  let common_years : Set ℕ := {1800, 1994}
  (∀ y ∈ leap_years, is_leap_year y) ∧
  (∀ y ∈ common_years, ¬is_leap_year y) ∧
  (leap_years ∪ common_years = {1800, 1992, 1994, 2040}) :=
by sorry

end leap_year_classification_l1510_151066


namespace number_problem_l1510_151055

theorem number_problem (x : ℝ) : (0.7 * x - 40 = 30) → x = 100 := by
  sorry

end number_problem_l1510_151055


namespace sqrt_36_div_6_l1510_151011

theorem sqrt_36_div_6 : Real.sqrt 36 / 6 = 1 := by sorry

end sqrt_36_div_6_l1510_151011


namespace x_positive_sufficient_not_necessary_for_x_nonzero_l1510_151090

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) ∧
  (∀ x : ℝ, x > 0 → x ≠ 0) :=
by sorry

end x_positive_sufficient_not_necessary_for_x_nonzero_l1510_151090


namespace right_triangle_area_l1510_151021

/-- The area of a right triangle given the sum of its legs and the altitude from the right angle. -/
theorem right_triangle_area (l h : ℝ) (hl : l > 0) (hh : h > 0) :
  ∃ S : ℝ, S = (1/2) * h * (Real.sqrt (l^2 + h^2) - h) ∧ 
  S > 0 ∧ 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = l ∧ 
  S = (1/2) * x * h ∧ S = (1/2) * y * h :=
by sorry


end right_triangle_area_l1510_151021


namespace tangent_implies_t_equals_4e_l1510_151056

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) (x y : ℝ) : Prop := y^2 = t*x ∧ y > 0 ∧ t > 0

def C₂ (x y : ℝ) : Prop := y = Real.exp (x + 1) - 1

-- Define the tangent line condition
def tangent_condition (t : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), C₁ t (4/t) 2 ∧ C₁ t x₀ y₀ ∧ C₂ x₀ y₀ ∧
  (∀ (x y : ℝ), y - 2 = (t/4)*(x - 4/t) → (C₁ t x y ∨ C₂ x y))

-- State the theorem
theorem tangent_implies_t_equals_4e :
  ∀ t : ℝ, tangent_condition t → t = 4 * Real.exp 1 :=
sorry

end tangent_implies_t_equals_4e_l1510_151056


namespace divisor_property_implies_prime_l1510_151035

theorem divisor_property_implies_prime (n : ℕ) 
  (h1 : n > 1)
  (h2 : ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) : 
  Nat.Prime n := by
  sorry

end divisor_property_implies_prime_l1510_151035


namespace amp_five_two_squared_l1510_151063

/-- The & operation defined for real numbers -/
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that (5 & 2)^2 = 441 -/
theorem amp_five_two_squared : (amp 5 2)^2 = 441 := by
  sorry

end amp_five_two_squared_l1510_151063


namespace green_balls_count_l1510_151096

theorem green_balls_count (total : ℕ) (p : ℚ) (h1 : total = 12) (h2 : p = 1 / 22) : 
  ∃ (green : ℕ), green ≤ total ∧ (green * (green - 1) : ℚ) / (total * (total - 1)) = p :=
by
  sorry

#check green_balls_count

end green_balls_count_l1510_151096


namespace similar_triangles_area_l1510_151015

/-- Given two similar triangles with corresponding sides of 1 cm and 2 cm, 
    and a total area of 25 cm², the area of the larger triangle is 20 cm². -/
theorem similar_triangles_area (A B : ℝ) : 
  A > 0 → B > 0 →  -- Areas are positive
  A + B = 25 →     -- Sum of areas is 25 cm²
  B / A = 4 →      -- Ratio of areas is 4 (square of the ratio of sides)
  B = 20 := by 
sorry

end similar_triangles_area_l1510_151015


namespace sqrt_equation_solution_l1510_151024

theorem sqrt_equation_solution (y : ℚ) :
  (Real.sqrt (4 * y + 3) / Real.sqrt (8 * y + 10) = Real.sqrt 3 / 2) →
  y = -9/4 := by
  sorry

end sqrt_equation_solution_l1510_151024


namespace floor_abs_negative_real_l1510_151006

theorem floor_abs_negative_real : ⌊|(-57.6 : ℝ)|⌋ = 57 := by sorry

end floor_abs_negative_real_l1510_151006


namespace power_equation_solution_l1510_151004

theorem power_equation_solution :
  ∃ x : ℤ, (3 : ℝ)^7 * (3 : ℝ)^x = 81 ∧ x = -3 := by
  sorry

end power_equation_solution_l1510_151004


namespace trapezoid_shorter_base_l1510_151098

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_line : ℝ
  longer_base_length : longer_base = 97
  midpoint_line_length : midpoint_line = 3
  midpoint_property : midpoint_line = (longer_base - shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) : t.shorter_base = 91 := by
  sorry

end trapezoid_shorter_base_l1510_151098


namespace ellipse_chord_theorem_l1510_151053

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines if a point lies on an ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines if a point is the midpoint of two other points -/
def isMidpoint (m p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Defines if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem ellipse_chord_theorem (e : Ellipse) (p1 p2 m : Point) :
  e.a = 6 ∧ e.b = 3 →
  p1.onEllipse e ∧ p2.onEllipse e →
  isMidpoint m p1 p2 →
  m = Point.mk 4 2 →
  areCollinear p1 p2 (Point.mk 0 4) :=
sorry

end ellipse_chord_theorem_l1510_151053


namespace tartar_arrangements_l1510_151061

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : ℕ) (duplicateSets : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (duplicateSets.map Nat.factorial).prod

/-- The word TARTAR has 6 letters with T, A, and R each appearing twice -/
theorem tartar_arrangements :
  uniqueArrangements 6 [2, 2, 2] = 90 := by
  sorry

end tartar_arrangements_l1510_151061


namespace equation_satisfied_l1510_151022

theorem equation_satisfied (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end equation_satisfied_l1510_151022


namespace complex_multiplication_l1510_151077

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 + 3 * i) = -3 + 2 * i := by
  sorry

end complex_multiplication_l1510_151077


namespace smallest_number_with_55_divisors_l1510_151060

/-- The number of divisors of n = p₁^k₁ * p₂^k₂ * ... * pₘ^kₘ is (k₁+1)(k₂+1)...(kₘ+1) -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n has exactly 55 divisors -/
def has_55_divisors (n : ℕ) : Prop := num_divisors n = 55

theorem smallest_number_with_55_divisors :
  ∃ (n : ℕ), has_55_divisors n ∧ ∀ (m : ℕ), has_55_divisors m → n ≤ m :=
by sorry

end smallest_number_with_55_divisors_l1510_151060


namespace unique_ticket_number_l1510_151097

def is_valid_ticket (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  n = (22 * x + 22 * y + 22 * z) / 2

theorem unique_ticket_number : ∃! n : ℕ, is_valid_ticket n ∧ n = 198 := by
  sorry

end unique_ticket_number_l1510_151097


namespace express_x_in_terms_of_y_l1510_151042

theorem express_x_in_terms_of_y (x y : ℝ) (h : 2 * x - 3 * y = 7) : 
  x = 7 / 2 + 3 / 2 * y := by
  sorry

end express_x_in_terms_of_y_l1510_151042


namespace sum_even_odd_is_odd_l1510_151068

theorem sum_even_odd_is_odd (a b : ℤ) (h1 : Even a) (h2 : Odd b) : Odd (a + b) := by
  sorry

end sum_even_odd_is_odd_l1510_151068


namespace greening_problem_l1510_151071

/-- The greening problem -/
theorem greening_problem 
  (total_area : ℝ) 
  (team_a_speed : ℝ) 
  (team_b_speed : ℝ) 
  (team_a_cost : ℝ) 
  (team_b_cost : ℝ) 
  (max_cost : ℝ) 
  (h1 : total_area = 1800) 
  (h2 : team_a_speed = 2 * team_b_speed) 
  (h3 : 400 / team_a_speed + 4 = 400 / team_b_speed) 
  (h4 : team_a_cost = 0.4) 
  (h5 : team_b_cost = 0.25) 
  (h6 : max_cost = 8) :
  ∃ (team_a_area team_b_area min_days : ℝ),
    team_a_area = 100 ∧ 
    team_b_area = 50 ∧ 
    min_days = 10 ∧
    (∀ y : ℝ, y ≥ min_days → 
      team_a_cost * y + team_b_cost * ((total_area - team_a_area * y) / team_b_area) ≤ max_cost) := by
  sorry

end greening_problem_l1510_151071


namespace dark_tiles_three_fourths_l1510_151039

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor :=
  (pattern_size : Nat)
  (corner_dark_tiles : Nat)
  (corner_size : Nat)

/-- The fraction of dark tiles in the entire floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  floor.corner_dark_tiles / (floor.corner_size * floor.corner_size)

/-- Theorem stating that for a floor with a 4x4 repeating pattern and 3 dark tiles
    in a 2x2 corner section, 3/4 of the entire floor is made of darker tiles -/
theorem dark_tiles_three_fourths (floor : TiledFloor)
  (h1 : floor.pattern_size = 4)
  (h2 : floor.corner_size = 2)
  (h3 : floor.corner_dark_tiles = 3) :
  dark_tile_fraction floor = 3/4 := by
  sorry

end dark_tiles_three_fourths_l1510_151039


namespace divisor_expression_l1510_151052

theorem divisor_expression (N D Y : ℕ) : 
  N = 45 * D + 13 → N = 6 * Y + 4 → D = (2 * Y - 3) / 15 := by
  sorry

end divisor_expression_l1510_151052


namespace linear_independence_preservation_l1510_151018

variable {n : ℕ}
variable (v : Fin (n - 1) → (Fin n → ℝ))

/-- P_{i,k} sets the i-th component of a vector to zero -/
def P (i k : ℕ) (x : Fin k → ℝ) : Fin k → ℝ :=
  λ j => if j = i then 0 else x j

theorem linear_independence_preservation (hn : n ≥ 2) 
  (hv : LinearIndependent ℝ v) :
  ∃ k : Fin n, LinearIndependent ℝ (λ i => P k n (v i)) := by
  sorry

end linear_independence_preservation_l1510_151018


namespace cubic_minus_linear_factorization_l1510_151045

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l1510_151045


namespace power_division_l1510_151003

theorem power_division (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end power_division_l1510_151003


namespace sequence_periodicity_l1510_151057

theorem sequence_periodicity (u : ℕ → ℝ) 
  (h : ∀ n : ℕ, u (n + 2) = |u (n + 1)| - u n) : 
  ∃ p : ℕ+, ∀ n : ℕ, u n = u (n + p) := by sorry

end sequence_periodicity_l1510_151057


namespace square_of_97_l1510_151016

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end square_of_97_l1510_151016


namespace tomatoes_rotted_l1510_151031

def initial_shipment : ℕ := 1000
def saturday_sales : ℕ := 300
def monday_shipment : ℕ := 2 * initial_shipment
def tuesday_ready : ℕ := 2500

theorem tomatoes_rotted (rotted : ℕ) : 
  rotted = initial_shipment - saturday_sales + monday_shipment - tuesday_ready := by sorry

end tomatoes_rotted_l1510_151031


namespace joe_remaining_money_l1510_151046

theorem joe_remaining_money (pocket_money : ℚ) (chocolate_fraction : ℚ) (fruit_fraction : ℚ) :
  pocket_money = 450 ∧
  chocolate_fraction = 1/9 ∧
  fruit_fraction = 2/5 →
  pocket_money - (chocolate_fraction * pocket_money + fruit_fraction * pocket_money) = 220 :=
by sorry

end joe_remaining_money_l1510_151046


namespace max_profit_is_270000_l1510_151041

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Represents the constraints and profit calculation for the company -/
def Company :=
  { p : Production //
    p.a ≥ 0 ∧
    p.b ≥ 0 ∧
    3 * p.a + p.b ≤ 13 ∧
    2 * p.a + 3 * p.b ≤ 18 }

/-- Calculates the profit for a given production -/
def profit (p : Production) : ℝ := 50000 * p.a + 30000 * p.b

/-- Theorem stating that the maximum profit is 270,000 yuan -/
theorem max_profit_is_270000 :
  ∃ (p : Company), ∀ (q : Company), profit p.val ≥ profit q.val ∧ profit p.val = 270000 := by
  sorry


end max_profit_is_270000_l1510_151041


namespace equation_solution_l1510_151081

theorem equation_solution : ∃! x : ℚ, 2 * (x - 1) = 2 - (5 * x - 2) := by
  sorry

end equation_solution_l1510_151081


namespace purely_imaginary_condition_l1510_151092

theorem purely_imaginary_condition (a : ℝ) : 
  (∃ (y : ℝ), Complex.mk (a^2 - 4) (a + 2) = Complex.I * y) ↔ a = 2 := by
  sorry

end purely_imaginary_condition_l1510_151092


namespace ascending_order_l1510_151058

theorem ascending_order (x y : ℝ) (hx : x > 1) (hy : -1 < y ∧ y < 0) :
  y < -y ∧ -y < -x*y ∧ -x*y < x := by sorry

end ascending_order_l1510_151058


namespace last_digit_alternating_factorial_sum_2014_l1510_151088

def alternatingFactorialSum (n : ℕ) : ℤ :=
  (List.range n).foldl (fun acc i => acc + (if i % 2 = 0 then 1 else -1) * (i + 1).factorial) 0

theorem last_digit_alternating_factorial_sum_2014 :
  (alternatingFactorialSum 2014) % 10 = 1 := by sorry

end last_digit_alternating_factorial_sum_2014_l1510_151088


namespace equation_holds_iff_specific_values_l1510_151047

/-- The equation holds for all real x if and only if a, b, p, and q have specific values -/
theorem equation_holds_iff_specific_values :
  ∀ (a b p q : ℝ),
    (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
    (a = (2^20 - 1)^(1/20) ∧
     b = -(2^20 - 1)^(1/20) / 2 ∧
     p = -1 ∧
     q = 1/4) :=
by sorry

end equation_holds_iff_specific_values_l1510_151047


namespace max_a_value_l1510_151051

theorem max_a_value (x a : ℤ) : 
  (x^2 + a*x = -28) → 
  (a > 0) → 
  ∃ (max_a : ℤ), max_a = 29 ∧ 
    ∀ (b : ℤ), (∃ (y : ℤ), y^2 + b*y = -28) → b ≤ max_a :=
by sorry

end max_a_value_l1510_151051


namespace square_difference_equals_two_l1510_151008

theorem square_difference_equals_two (x y : ℝ) 
  (h1 : 1/x + 1/y = 2) 
  (h2 : x*y + x - y = 6) : 
  x^2 - y^2 = 2 := by
  sorry

end square_difference_equals_two_l1510_151008


namespace solution_to_modular_equation_l1510_151010

theorem solution_to_modular_equation :
  ∃ x : ℤ, (7 * x + 2) % 15 = 11 % 15 ∧ x % 15 = 12 % 15 := by
  sorry

end solution_to_modular_equation_l1510_151010


namespace cost_of_one_each_l1510_151072

theorem cost_of_one_each (x y z : ℝ) 
  (eq1 : 3 * x + 7 * y + z = 325)
  (eq2 : 4 * x + 10 * y + z = 410) :
  x + y + z = 155 := by
  sorry

end cost_of_one_each_l1510_151072


namespace cos_equality_problem_l1510_151034

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → (Real.cos (n * π / 180) = Real.cos (832 * π / 180) ↔ n = 112) := by
  sorry

end cos_equality_problem_l1510_151034


namespace inequality_solution_l1510_151086

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_increasing : ∀ x y, x < y → f x < f y
axiom f_point1 : f 0 = -2
axiom f_point2 : f 3 = 2

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x ≥ 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | |f (x + 1)| ≥ 2} = solution_set :=
sorry

end inequality_solution_l1510_151086


namespace perpendicular_vectors_l1510_151074

/-- Given vectors a and b in ℝ², if a is perpendicular to (a + 2b), then the second component of b is -3/4 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (-1, 2)) (h' : b.1 = 1) 
    (h'' : a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) : 
    b.2 = -3/4 := by
  sorry

end perpendicular_vectors_l1510_151074


namespace trapezoidal_channel_bottom_width_l1510_151064

theorem trapezoidal_channel_bottom_width
  (top_width : ℝ)
  (area : ℝ)
  (depth : ℝ)
  (h_top_width : top_width = 12)
  (h_area : area = 700)
  (h_depth : depth = 70) :
  ∃ bottom_width : ℝ,
    bottom_width = 8 ∧
    area = (1 / 2) * (top_width + bottom_width) * depth :=
by sorry

end trapezoidal_channel_bottom_width_l1510_151064


namespace farmer_apples_l1510_151050

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end farmer_apples_l1510_151050


namespace cubic_equation_roots_relation_l1510_151012

theorem cubic_equation_roots_relation (a b c : ℝ) (s₁ s₂ s₃ : ℂ) :
  (s₁^3 + a*s₁^2 + b*s₁ + c = 0) →
  (s₂^3 + a*s₂^2 + b*s₂ + c = 0) →
  (s₃^3 + a*s₃^2 + b*s₃ + c = 0) →
  (∃ p q r : ℝ, (s₁^2)^3 + p*(s₁^2)^2 + q*(s₁^2) + r = 0 ∧
               (s₂^2)^3 + p*(s₂^2)^2 + q*(s₂^2) + r = 0 ∧
               (s₃^2)^3 + p*(s₃^2)^2 + q*(s₃^2) + r = 0) →
  (∃ p q r : ℝ, p = a^2 - 2*b ∧ q = b^2 + 2*a*c ∧ r = c^2) :=
by sorry

end cubic_equation_roots_relation_l1510_151012


namespace silver_medals_count_l1510_151087

theorem silver_medals_count (total_medals gold_medals bronze_medals : ℕ) 
  (h1 : total_medals = 67)
  (h2 : gold_medals = 19)
  (h3 : bronze_medals = 16) :
  total_medals - gold_medals - bronze_medals = 32 := by
sorry

end silver_medals_count_l1510_151087


namespace complex_equation_solution_l1510_151062

theorem complex_equation_solution (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : m / (1 + i) = 1 - n * i) : 
  m - n = 1 := by sorry

end complex_equation_solution_l1510_151062


namespace dividend_divisor_quotient_ratio_l1510_151013

theorem dividend_divisor_quotient_ratio 
  (dividend : ℚ) (divisor : ℚ) (quotient : ℚ) 
  (h : dividend / divisor = 9 / 2) : 
  divisor / quotient = 2 / 9 := by
sorry

end dividend_divisor_quotient_ratio_l1510_151013


namespace sum_congruence_modulo_nine_l1510_151070

theorem sum_congruence_modulo_nine : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end sum_congruence_modulo_nine_l1510_151070


namespace fraction_unchanged_when_multiplied_by_two_l1510_151000

theorem fraction_unchanged_when_multiplied_by_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / (x + y) = (2 * x) / (2 * (x + y)) := by
  sorry

end fraction_unchanged_when_multiplied_by_two_l1510_151000


namespace power_zero_minus_one_equals_zero_l1510_151067

theorem power_zero_minus_one_equals_zero : 2^0 - 1 = 0 := by
  sorry

end power_zero_minus_one_equals_zero_l1510_151067


namespace jonathan_weekly_caloric_deficit_l1510_151036

/-- Jonathan's daily caloric intake on regular days -/
def regular_daily_intake : ℕ := 2500

/-- Jonathan's extra caloric intake on Saturday -/
def saturday_extra_intake : ℕ := 1000

/-- Jonathan's daily caloric burn -/
def daily_burn : ℕ := 3000

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Number of regular eating days in a week -/
def regular_days : ℕ := 6

/-- Calculate Jonathan's weekly caloric deficit -/
theorem jonathan_weekly_caloric_deficit :
  daily_burn * days_in_week - (regular_daily_intake * regular_days + (regular_daily_intake + saturday_extra_intake)) = 2500 := by
  sorry

end jonathan_weekly_caloric_deficit_l1510_151036


namespace f_not_monotonic_iff_a_in_range_l1510_151093

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧ 
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem f_not_monotonic_iff_a_in_range (a : ℝ) :
  is_not_monotonic (f a) (-1) 1 ↔ 
  (a > -5 ∧ a < -1/2) ∨ (a > -1/2 ∧ a < 1) :=
sorry

end f_not_monotonic_iff_a_in_range_l1510_151093


namespace parabola_translation_l1510_151032

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -x^2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := -(x + 2)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 3 :=
by
  sorry

end parabola_translation_l1510_151032


namespace absolute_value_inequality_l1510_151075

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x + 1))| > 3 ↔ x < -1 ∨ (-5/6 < x ∧ x < -1) :=
sorry

end absolute_value_inequality_l1510_151075


namespace plane_line_perpendicular_l1510_151099

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem plane_line_perpendicular 
  (m : Line) (α β γ : Plane) :
  parallel α β → parallel β γ → perpendicular m α → perpendicular m γ :=
sorry

end plane_line_perpendicular_l1510_151099


namespace vector_sum_zero_parallel_sufficient_not_necessary_l1510_151089

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_sum_zero_parallel_sufficient_not_necessary :
  ∀ (a b : V), a ≠ 0 → b ≠ 0 →
  (a + b = 0 → parallel a b) ∧
  ¬(parallel a b → a + b = 0) :=
by sorry

end vector_sum_zero_parallel_sufficient_not_necessary_l1510_151089


namespace parabola_constant_term_l1510_151040

theorem parabola_constant_term (p q : ℝ) : 
  (∀ x y : ℝ, y = x^2 + p*x + q → 
    ((x = 3 ∧ y = 4) ∨ (x = 5 ∧ y = 4))) → 
  q = 19 := by
sorry

end parabola_constant_term_l1510_151040


namespace age_ratio_solution_l1510_151059

/-- Represents the age ratio problem of Mandy and her siblings -/
def age_ratio_problem (mandy_age brother_age sister_age : ℚ) : Prop :=
  mandy_age = 3 ∧
  sister_age = brother_age - 5 ∧
  mandy_age - sister_age = 4 ∧
  brother_age / mandy_age = 4 / 3

/-- Theorem stating that there exists a unique solution to the age ratio problem -/
theorem age_ratio_solution :
  ∃! (mandy_age brother_age sister_age : ℚ),
    age_ratio_problem mandy_age brother_age sister_age :=
by
  sorry

#check age_ratio_solution

end age_ratio_solution_l1510_151059


namespace absolute_value_inequality_l1510_151094

theorem absolute_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 2| > k) ↔ k < 3 := by sorry

end absolute_value_inequality_l1510_151094


namespace family_income_proof_l1510_151038

/-- Proves that the initial average monthly income of a family is 840 given the conditions --/
theorem family_income_proof (initial_members : ℕ) (deceased_income new_average : ℚ) :
  initial_members = 4 →
  deceased_income = 1410 →
  new_average = 650 →
  (initial_members : ℚ) * (initial_members * new_average + deceased_income) / initial_members = 840 :=
by sorry

end family_income_proof_l1510_151038


namespace quadrilateral_inequality_l1510_151076

/-- Given four points on a plane, the distance between any two points
    is less than or equal to the sum of the distances along a path
    through the other two points. -/
theorem quadrilateral_inequality (A B C D : EuclideanSpace ℝ (Fin 2)) :
  dist A D ≤ dist A B + dist B C + dist C D := by
  sorry

end quadrilateral_inequality_l1510_151076


namespace max_abs_z_given_distance_from_2i_l1510_151005

theorem max_abs_z_given_distance_from_2i (z : ℂ) : 
  Complex.abs (z - 2 * Complex.I) = 1 → Complex.abs z ≤ 3 ∧ ∃ w : ℂ, Complex.abs (w - 2 * Complex.I) = 1 ∧ Complex.abs w = 3 := by
  sorry

end max_abs_z_given_distance_from_2i_l1510_151005


namespace ellipse_intersection_length_l1510_151023

-- Define the ellipse (C)
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line passing through (0, 2) with slope 1
def line_l (x y : ℝ) : Prop :=
  y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem ellipse_intersection_length :
  -- Given conditions
  let F₁ : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let F₂ : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let major_axis_length : ℝ := 6
  
  -- Prove the following
  ∀ A B : ℝ × ℝ, intersection_points A B →
    -- 1. The standard equation of the ellipse is correct
    (∀ x y : ℝ, (x^2 / 9 + y^2 = 1) ↔ ellipse_C x y) ∧
    -- 2. The length of AB is 6√3/5
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * Real.sqrt 3 / 5 :=
by
  sorry


end ellipse_intersection_length_l1510_151023


namespace dog_kennel_problem_l1510_151084

theorem dog_kennel_problem (total long_fur brown neither : ℕ) 
  (h_total : total = 45)
  (h_long_fur : long_fur = 26)
  (h_brown : brown = 30)
  (h_neither : neither = 8)
  : long_fur + brown - (total - neither) = 19 := by
  sorry

end dog_kennel_problem_l1510_151084


namespace max_third_term_l1510_151020

/-- An arithmetic sequence of four positive integers with sum 50 -/
structure ArithSequence :=
  (a : ℕ+) -- First term
  (d : ℕ+) -- Common difference
  (sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50)

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value of the third term is 16 -/
theorem max_third_term :
  ∀ seq : ArithSequence, third_term seq ≤ 16 ∧ ∃ seq : ArithSequence, third_term seq = 16 :=
sorry

end max_third_term_l1510_151020


namespace series_solution_l1510_151009

/-- The sum of the infinite series 1 + 3x + 6x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := 1 / (1 - x)^3

/-- Theorem: If S(x) = 4, then x = 1 - 1/∛4 -/
theorem series_solution (x : ℝ) (h : S x = 4) : x = 1 - 1 / Real.rpow 4 (1/3) := by
  sorry

end series_solution_l1510_151009


namespace nth_equation_pattern_l1510_151007

theorem nth_equation_pattern (n : ℕ) :
  (-n : ℚ) * (n / (n + 1)) = -n + (n / (n + 1)) := by
  sorry

end nth_equation_pattern_l1510_151007


namespace travel_methods_count_l1510_151048

/-- The number of transportation options from Shijiazhuang to Qingdao -/
def shijiazhuang_to_qingdao : Nat := 3

/-- The number of transportation options from Qingdao to Guangzhou -/
def qingdao_to_guangzhou : Nat := 4

/-- The total number of travel methods for the entire journey -/
def total_travel_methods : Nat := shijiazhuang_to_qingdao * qingdao_to_guangzhou

theorem travel_methods_count : total_travel_methods = 12 := by
  sorry

end travel_methods_count_l1510_151048


namespace prime_pairs_square_sum_l1510_151069

theorem prime_pairs_square_sum (p q : ℕ) : 
  Prime p → Prime q → (∃ n : ℕ, p^2 + 5*p*q + 4*q^2 = n^2) → 
  ((p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11)) :=
by sorry

end prime_pairs_square_sum_l1510_151069


namespace number_of_paths_equals_combinations_l1510_151029

-- Define the grid size
def gridSize : Nat := 6

-- Define the total number of moves
def totalMoves : Nat := gridSize * 2

-- Define the number of rightward (or downward) moves
def directionMoves : Nat := gridSize

-- Theorem statement
theorem number_of_paths_equals_combinations :
  (Nat.choose totalMoves directionMoves) = 924 := by
  sorry

end number_of_paths_equals_combinations_l1510_151029


namespace negative_expression_l1510_151001

theorem negative_expression : 
  let expr1 := -(-1)
  let expr2 := (-1)^2
  let expr3 := |-1|
  let expr4 := -|-1|
  (expr1 ≥ 0 ∧ expr2 ≥ 0 ∧ expr3 ≥ 0 ∧ expr4 < 0) := by sorry

end negative_expression_l1510_151001


namespace isabella_trip_l1510_151054

def exchange_rate : ℚ := 8 / 5

def spent_aud : ℕ := 80

def remaining_aud (e : ℕ) : ℕ := e + 20

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (fun c => c.toNat - '0'.toNat)
  digits.sum

theorem isabella_trip (e : ℕ) : 
  (exchange_rate * e : ℚ) - spent_aud = remaining_aud e →
  e = 167 ∧ sum_of_digits e = 14 := by
  sorry

end isabella_trip_l1510_151054


namespace c_share_is_75_l1510_151027

-- Define the total payment
def total_payment : ℚ := 600

-- Define the time taken by each worker individually
def a_time : ℚ := 6
def b_time : ℚ := 8

-- Define the time taken by all three workers together
def abc_time : ℚ := 3

-- Define the shares of A and B
def a_share : ℚ := 300
def b_share : ℚ := 225

-- Define C's share as a function of the given parameters
def c_share (total : ℚ) (a_t b_t abc_t : ℚ) (a_s b_s : ℚ) : ℚ :=
  total - (a_s + b_s)

-- Theorem statement
theorem c_share_is_75 :
  c_share total_payment a_time b_time abc_time a_share b_share = 75 := by
  sorry


end c_share_is_75_l1510_151027


namespace pizza_special_pricing_l1510_151073

/-- Represents the cost calculation for pizzas with special pricing --/
def pizza_cost (standard_price : ℕ) (triple_cheese_count : ℕ) (meat_lovers_count : ℕ) : ℕ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * standard_price
  let meat_lovers_cost := ((meat_lovers_count + 2) / 3 * 2) * standard_price
  triple_cheese_cost + meat_lovers_cost

/-- Theorem stating the total cost of pizzas under special pricing --/
theorem pizza_special_pricing :
  pizza_cost 5 10 9 = 55 := by
  sorry


end pizza_special_pricing_l1510_151073
