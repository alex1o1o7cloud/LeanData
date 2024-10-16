import Mathlib

namespace NUMINAMATH_CALUDE_find_divisor_l2727_272746

theorem find_divisor : ∃ d : ℕ, d > 1 ∧ (1077 + 4) % d = 0 ∧ d = 1081 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2727_272746


namespace NUMINAMATH_CALUDE_factorization_sum_l2727_272763

theorem factorization_sum (A B C D E F G H J K : ℤ) :
  (∀ x y : ℝ, 27 * x^6 - 512 * y^6 = (A * x + B * y) * (C * x^2 + D * x * y + E * y^2) * 
                                     (F * x + G * y) * (H * x^2 + J * x * y + K * y^2)) →
  A + B + C + D + E + F + G + H + J + K = 32 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l2727_272763


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l2727_272748

theorem half_radius_circle_y (x y : Real) :
  (∃ (r : Real), x = π * r^2 ∧ y = π * r^2) →  -- circles x and y have the same area
  (∃ (r : Real), 18 * π = 2 * π * r) →         -- circle x has circumference 18π
  (∃ (r : Real), y = π * r^2 ∧ r / 2 = 4.5) := by
sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l2727_272748


namespace NUMINAMATH_CALUDE_money_problem_l2727_272774

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a - b > 32)
  (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l2727_272774


namespace NUMINAMATH_CALUDE_inequality_solution_l2727_272759

theorem inequality_solution (a b c : ℝ) (h1 : a < b)
  (h2 : ∀ x : ℝ, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -6 ∨ |x - 31| ≤ 1) :
  a + 2 * b + 3 * c = 76 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2727_272759


namespace NUMINAMATH_CALUDE_value_of_P_l2727_272762

theorem value_of_P : ∃ P : ℚ, (3/4 : ℚ) * (1/9 : ℚ) * P = (1/4 : ℚ) * (1/8 : ℚ) * 160 ∧ P = 60 := by
  sorry

end NUMINAMATH_CALUDE_value_of_P_l2727_272762


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2727_272796

theorem stock_price_calculation (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : 
  initial_price = 100 ∧ 
  first_year_increase = 1.5 ∧ 
  second_year_decrease = 0.4 → 
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 150 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2727_272796


namespace NUMINAMATH_CALUDE_box_sales_ratio_l2727_272721

/-- Proof of the ratio of boxes sold on Saturday to Friday -/
theorem box_sales_ratio :
  ∀ (friday saturday sunday : ℕ),
  friday = 30 →
  sunday = saturday - 15 →
  friday + saturday + sunday = 135 →
  saturday / friday = 2 := by
sorry

end NUMINAMATH_CALUDE_box_sales_ratio_l2727_272721


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2727_272707

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  asymptote_slope : ℝ
  real_axis_length : ℝ
  foci_on_x_axis : Bool

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 9 = 1

/-- Theorem: Given a hyperbola with asymptote slope 3, real axis length 2, and foci on x-axis,
    its standard equation is x² - y²/9 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 3)
    (h_real_axis : h.real_axis_length = 2)
    (h_foci : h.foci_on_x_axis = true) :
    standard_equation h :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2727_272707


namespace NUMINAMATH_CALUDE_donny_money_theorem_l2727_272727

/-- The amount of money Donny had initially in his piggy bank. -/
def initial_money : ℕ := 78

/-- The cost of the kite Donny bought. -/
def kite_cost : ℕ := 8

/-- The cost of the frisbee Donny bought. -/
def frisbee_cost : ℕ := 9

/-- The amount of money Donny has left after purchases. -/
def money_left : ℕ := 61

/-- Theorem stating that Donny's initial money equals the sum of his purchases and remaining money. -/
theorem donny_money_theorem : 
  initial_money = kite_cost + frisbee_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_donny_money_theorem_l2727_272727


namespace NUMINAMATH_CALUDE_cone_sphere_volume_l2727_272705

/-- Given a cone with lateral surface forming a semicircle of radius 2√3 when unrolled,
    and with vertex and base circle on the surface of a sphere O,
    prove that the volume of sphere O is 32π/3 -/
theorem cone_sphere_volume (l : ℝ) (r : ℝ) (h : ℝ) (R : ℝ) :
  l = 2 * Real.sqrt 3 →
  r = l / 2 →
  h^2 = l^2 - r^2 →
  2 * R = l^2 / h →
  (4 / 3) * Real.pi * R^3 = (32 / 3) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_l2727_272705


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2727_272735

theorem isosceles_triangle_base_angle (α : ℝ) (h1 : α = 42) :
  let β := (180 - α) / 2
  (β = 42 ∨ β = 69) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2727_272735


namespace NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l2727_272792

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (2/3 - 4/5 * Complex.I) = Real.sqrt 244 / 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l2727_272792


namespace NUMINAMATH_CALUDE_cauchy_schwarz_2d_l2727_272791

theorem cauchy_schwarz_2d {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 ∧
  ((a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 ↔ a*d = b*c) :=
sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_2d_l2727_272791


namespace NUMINAMATH_CALUDE_sophies_spend_is_72_80_l2727_272781

/-- The total amount Sophie spends on her purchases -/
def sophies_total_spend : ℚ :=
  let cupcakes := 5 * 2
  let doughnuts := 6 * 1
  let apple_pie := 4 * 2
  let cookies := 15 * 0.6
  let chocolate_bars := 8 * 1.5
  let soda := 12 * 1.2
  let gum := 3 * 0.8
  let chips := 10 * 1.1
  cupcakes + doughnuts + apple_pie + cookies + chocolate_bars + soda + gum + chips

/-- Theorem stating that Sophie's total spend is $72.80 -/
theorem sophies_spend_is_72_80 : sophies_total_spend = 72.8 := by
  sorry

end NUMINAMATH_CALUDE_sophies_spend_is_72_80_l2727_272781


namespace NUMINAMATH_CALUDE_abc_sum_mod_7_l2727_272752

theorem abc_sum_mod_7 (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 2 ∧ 
  (4 * c) % 7 = 3 ∧ 
  (7 * b) % 7 = (4 + b) % 7 → 
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_7_l2727_272752


namespace NUMINAMATH_CALUDE_all_acute_triangle_count_l2727_272744

/-- A function that checks if a triangle with sides a, b, c has all acute angles -/
def isAllAcuteTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧
  a * a + b * b > c * c ∧
  a * a + c * c > b * b ∧
  b * b + c * c > a * a

/-- The theorem stating that there are exactly 5 integer values of y that form an all-acute triangle with sides 15 and 8 -/
theorem all_acute_triangle_count :
  ∃! (s : Finset ℕ), s.card = 5 ∧ ∀ y ∈ s, isAllAcuteTriangle 15 8 y :=
sorry

end NUMINAMATH_CALUDE_all_acute_triangle_count_l2727_272744


namespace NUMINAMATH_CALUDE_min_value_of_f_l2727_272754

/-- The function f(x) = e^x - e^(2x) has a minimum value of -e^2 -/
theorem min_value_of_f (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.exp x - Real.exp (2 * x)
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2727_272754


namespace NUMINAMATH_CALUDE_sector_angle_l2727_272710

theorem sector_angle (r : ℝ) (α : ℝ) (h1 : α * r = 5) (h2 : (1/2) * α * r^2 = 5) : α = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2727_272710


namespace NUMINAMATH_CALUDE_pats_picnic_dessert_l2727_272775

/-- Pat's picnic dessert problem -/
theorem pats_picnic_dessert (cookies : ℕ) (candy : ℕ) (family_size : ℕ) (dessert_per_person : ℕ) 
  (h1 : cookies = 42)
  (h2 : candy = 63)
  (h3 : family_size = 7)
  (h4 : dessert_per_person = 18) :
  family_size * dessert_per_person - (cookies + candy) = 21 := by
  sorry

end NUMINAMATH_CALUDE_pats_picnic_dessert_l2727_272775


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2727_272758

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event "at most one shot is successful"
def at_most_one_successful (ω : Ω) : Prop :=
  ¬(ω.1 ∧ ω.2)

-- Define the event "both shots are successful"
def both_successful (ω : Ω) : Prop :=
  ω.1 ∧ ω.2

-- Theorem: "both shots are successful" is mutually exclusive to "at most one shot is successful"
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(at_most_one_successful ω ∧ both_successful ω) :=
by
  sorry


end NUMINAMATH_CALUDE_mutually_exclusive_events_l2727_272758


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l2727_272788

/-- Given a sequence of positive real numbers {aₙ} where the sum of the first n terms
    Sₙ satisfies Sₙ = (1/2)(aₙ + 1/aₙ), prove that aₙ = √n - √(n-1) for all positive integers n. -/
theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (h_pos : ∀ k, k > 0 → a k > 0)
  (h_sum : ∀ k, k > 0 → S k = (1/2) * (a k + 1 / a k)) :
  a n = Real.sqrt n - Real.sqrt (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l2727_272788


namespace NUMINAMATH_CALUDE_example_quadratic_function_l2727_272718

/-- A function f: ℝ → ℝ is quadratic if there exist constants a, b, c where a ≠ 0 such that
    f(x) = ax^2 + bx + c for all x ∈ ℝ -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 3x^2 + x - 1 is quadratic -/
theorem example_quadratic_function :
  IsQuadratic (fun x => 3 * x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_example_quadratic_function_l2727_272718


namespace NUMINAMATH_CALUDE_at_least_one_correct_guess_l2727_272701

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Blue
| Green

/-- Converts HatColor to its corresponding integer representation -/
def hatColorToInt (color : HatColor) : Fin 3 :=
  match color with
  | HatColor.Red => 0
  | HatColor.Blue => 1
  | HatColor.Green => 2

/-- Represents the configuration of hats on the four sages -/
structure HatConfiguration where
  a : HatColor
  b : HatColor
  c : HatColor
  d : HatColor

/-- Represents a sage's guess -/
def SageGuess := Fin 3

/-- The strategy for Sage A -/
def guessA (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.b + hatColorToInt config.d) % 3

/-- The strategy for Sage B -/
def guessB (config : HatConfiguration) : SageGuess :=
  (-(hatColorToInt config.a + hatColorToInt config.c)) % 3

/-- The strategy for Sage C -/
def guessC (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.b - hatColorToInt config.d) % 3

/-- The strategy for Sage D -/
def guessD (config : HatConfiguration) : SageGuess :=
  (hatColorToInt config.c - hatColorToInt config.a) % 3

/-- Theorem stating that the strategy guarantees at least one correct guess -/
theorem at_least_one_correct_guess (config : HatConfiguration) :
  (guessA config = hatColorToInt config.a) ∨
  (guessB config = hatColorToInt config.b) ∨
  (guessC config = hatColorToInt config.c) ∨
  (guessD config = hatColorToInt config.d) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_correct_guess_l2727_272701


namespace NUMINAMATH_CALUDE_harry_hike_water_remaining_l2727_272709

/-- Calculates the remaining water in Harry's canteen after a hike -/
def remaining_water (initial_water : ℝ) (hike_distance : ℝ) (hike_duration : ℝ) 
  (leak_rate : ℝ) (last_mile_consumption : ℝ) (first_miles_consumption_rate : ℝ) : ℝ :=
  initial_water - 
  (leak_rate * hike_duration) - 
  (first_miles_consumption_rate * (hike_distance - 1)) - 
  last_mile_consumption

/-- Theorem stating that the remaining water in Harry's canteen is 2 cups -/
theorem harry_hike_water_remaining :
  remaining_water 11 7 3 1 3 0.5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_harry_hike_water_remaining_l2727_272709


namespace NUMINAMATH_CALUDE_sum_of_m_and_n_l2727_272737

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6*m + 10*n + 34 = 0) : m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_and_n_l2727_272737


namespace NUMINAMATH_CALUDE_right_triangle_area_l2727_272785

/-- 
  Given a right-angled triangle with legs x and y, and hypotenuse z,
  where x:y = 3:4 and x^2 + y^2 = z^2, prove that the area A of the triangle
  is equal to (2/3)x^2 or (6/25)z^2.
-/
theorem right_triangle_area (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x^2 + y^2 = z^2) (h5 : 3 * y = 4 * x) :
  ∃ A : ℝ, A = (2/3) * x^2 ∧ A = (6/25) * z^2 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l2727_272785


namespace NUMINAMATH_CALUDE_abs_inequality_iff_gt_l2727_272769

theorem abs_inequality_iff_gt (a b : ℝ) (h : a * b > 0) :
  a * |a| > b * |b| ↔ a > b :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_iff_gt_l2727_272769


namespace NUMINAMATH_CALUDE_peggy_doll_ratio_l2727_272789

/-- Represents the number of dolls in various situations --/
structure DollCount where
  initial : Nat
  fromGrandmother : Nat
  final : Nat

/-- Calculates the ratio of birthday/Christmas dolls to grandmother's dolls --/
def dollRatio (d : DollCount) : Rat :=
  let birthdayChristmas := d.final - d.initial - d.fromGrandmother
  birthdayChristmas / d.fromGrandmother

/-- Theorem stating the ratio of dolls Peggy received --/
theorem peggy_doll_ratio (d : DollCount) 
  (h1 : d.initial = 6)
  (h2 : d.fromGrandmother = 30)
  (h3 : d.final = 51) :
  dollRatio d = 1/2 := by
  sorry

#eval dollRatio ⟨6, 30, 51⟩

end NUMINAMATH_CALUDE_peggy_doll_ratio_l2727_272789


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_two_fifths_l2727_272765

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2 + y, 6; 4 - y, 9]

theorem matrix_not_invertible_iff_y_eq_two_fifths :
  ∀ y : ℝ, ¬(Matrix.det (matrix y) ≠ 0) ↔ y = 2/5 := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_two_fifths_l2727_272765


namespace NUMINAMATH_CALUDE_annie_original_seat_l2727_272787

-- Define the type for seats
inductive Seat
| one
| two
| three
| four
| five

-- Define the type for friends
inductive Friend
| Annie
| Beth
| Cass
| Dana
| Ella

-- Define the function type for seating arrangement
def SeatingArrangement := Seat → Friend

-- Define the movement function type
def Movement := SeatingArrangement → SeatingArrangement

-- Define the specific movements
def bethMove : Movement := sorry
def cassDanaSwap : Movement := sorry
def ellaMove : Movement := sorry

-- Define the property of Ella ending in an end seat
def ellaInEndSeat (arrangement : SeatingArrangement) : Prop := sorry

-- Define the theorem
theorem annie_original_seat (initial : SeatingArrangement) :
  (∃ (final : SeatingArrangement),
    final = ellaMove (cassDanaSwap (bethMove initial)) ∧
    ellaInEndSeat final) →
  initial Seat.one = Friend.Annie := by sorry

end NUMINAMATH_CALUDE_annie_original_seat_l2727_272787


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l2727_272700

theorem sum_of_cyclic_equations (p q r : ℕ+) 
  (eq1 : p * q + r = 47)
  (eq2 : q * r + p = 47)
  (eq3 : r * p + q = 47) :
  p + q + r = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l2727_272700


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l2727_272795

-- Define a circle
variable (circle : Type) [MetricSpace circle]

-- Define the chords and intersection point
variable (chord1 chord2 : Set circle)
variable (P : circle)

-- Define the segments of the first chord
variable (PA PB : ℝ)

-- Define the ratio of the segments of the second chord
variable (r : ℚ)

-- State the theorem
theorem intersecting_chords_theorem 
  (h1 : P ∈ chord1 ∩ chord2)
  (h2 : PA = 12)
  (h3 : PB = 18)
  (h4 : r = 3 / 8)
  : ∃ (PC PD : ℝ), PC + PD = 33 ∧ PC / PD = r := by
  sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l2727_272795


namespace NUMINAMATH_CALUDE_place_value_ratio_l2727_272723

def number : ℚ := 86743.2951

def place_value_6 : ℚ := 10000
def place_value_5 : ℚ := 0.1

theorem place_value_ratio :
  place_value_6 / place_value_5 = 100000 := by
  sorry

#check place_value_ratio

end NUMINAMATH_CALUDE_place_value_ratio_l2727_272723


namespace NUMINAMATH_CALUDE_rectangle_width_equal_square_side_l2727_272713

theorem rectangle_width_equal_square_side 
  (square_side : ℝ) 
  (rect_length : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_length = 3)
  (h3 : square_side * square_side = rect_length * (square_side)) :
  square_side = rect_length :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_equal_square_side_l2727_272713


namespace NUMINAMATH_CALUDE_total_checks_is_30_l2727_272797

/-- The number of $50 checks -/
def F : ℕ := sorry

/-- The number of $100 checks -/
def H : ℕ := sorry

/-- The total worth of all checks is $1800 -/
axiom total_worth : 50 * F + 100 * H = 1800

/-- The average of remaining checks after removing 18 $50 checks is $75 -/
axiom remaining_average : (1800 - 18 * 50) / (F + H - 18) = 75

/-- The total number of travelers checks -/
def total_checks : ℕ := F + H

/-- Theorem: The total number of travelers checks is 30 -/
theorem total_checks_is_30 : total_checks = 30 := by sorry

end NUMINAMATH_CALUDE_total_checks_is_30_l2727_272797


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2727_272708

/-- Given a principal sum and a time period of 8 years, if the simple interest
    is one-fifth of the principal sum, then the rate of interest per annum is 2.5%. -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  (P * 2.5 * 8) / 100 = P / 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2727_272708


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2727_272751

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ d : ℕ+, d ∣ n → d ≤ 12) : 144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2727_272751


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2727_272743

-- Define the polynomial
def polynomial (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem root_sum_theorem (a b c d : ℝ) (h_a : a ≠ 0) :
  polynomial a b c d 4 = 0 ∧
  polynomial a b c d (-1) = 0 ∧
  polynomial a b c d (-3) = 0 →
  (b + c) / a = -1441 / 37 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2727_272743


namespace NUMINAMATH_CALUDE_max_moves_21x21_max_moves_20x21_l2727_272783

/-- Represents a rectangular grid with lights -/
structure Grid where
  rows : ℕ
  cols : ℕ
  lights : Set (ℕ × ℕ)

/-- Represents a move in the light-turning game -/
structure Move where
  line : (ℝ × ℝ) → Prop
  affected_lights : Set (ℕ × ℕ)

/-- The maximum number of moves possible for a given grid -/
def max_moves (g : Grid) : ℕ := sorry

/-- Theorem stating the maximum number of moves for a 21×21 square grid -/
theorem max_moves_21x21 :
  ∀ (g : Grid), g.rows = 21 ∧ g.cols = 21 → max_moves g = 3 := by sorry

/-- Theorem stating the maximum number of moves for a 20×21 rectangular grid -/
theorem max_moves_20x21 :
  ∀ (g : Grid), g.rows = 20 ∧ g.cols = 21 → max_moves g = 4 := by sorry

end NUMINAMATH_CALUDE_max_moves_21x21_max_moves_20x21_l2727_272783


namespace NUMINAMATH_CALUDE_square_sum_ge_third_square_sum_l2727_272772

theorem square_sum_ge_third_square_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_third_square_sum_l2727_272772


namespace NUMINAMATH_CALUDE_triangle_height_inradius_inequality_l2727_272745

theorem triangle_height_inradius_inequality 
  (h₁ h₂ h₃ r : ℝ) (α : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ r > 0)
  (h_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = 2 * (r * (a + b + c) / 2) / a ∧
    h₂ = 2 * (r * (a + b + c) / 2) / b ∧
    h₃ = 2 * (r * (a + b + c) / 2) / c)
  (h_alpha : α ≥ 1) :
  h₁^α + h₂^α + h₃^α ≥ 3 * (3 * r)^α := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_inradius_inequality_l2727_272745


namespace NUMINAMATH_CALUDE_andrews_age_l2727_272731

theorem andrews_age :
  ∀ (a g : ℚ),
  g = 15 * a →
  g - a = 55 →
  a = 55 / 14 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l2727_272731


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2727_272773

variable (f g : ℝ → ℝ)

def IsConcave (f : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) ≥ t * f x + (1 - t) * f y

theorem polynomial_characterization
  (hf_concave : IsConcave f)
  (hg_continuous : Continuous g)
  (h_equality : ∀ x y : ℝ, f (x + y) + f (x - y) - 2 * f x = g x * y^2) :
  ∃ A B C : ℝ, ∀ x : ℝ, f x = A * x + B * x^2 + C :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2727_272773


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2727_272730

theorem quadratic_factorization (x : ℝ) :
  16 * x^2 + 8 * x - 24 = 8 * (2 * x + 3) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2727_272730


namespace NUMINAMATH_CALUDE_pizza_sales_distribution_l2727_272726

/-- The total number of pizzas sold in a year -/
def total_pizzas : ℝ := 12.5

/-- The percentage of pizzas sold in summer -/
def summer_percent : ℝ := 0.4

/-- The number of pizzas sold in summer (in millions) -/
def summer_pizzas : ℝ := 5

/-- The percentage of pizzas sold in fall -/
def fall_percent : ℝ := 0.1

/-- The percentage of pizzas sold in winter -/
def winter_percent : ℝ := 0.2

/-- The number of pizzas sold in spring (in millions) -/
def spring_pizzas : ℝ := total_pizzas - (summer_pizzas + fall_percent * total_pizzas + winter_percent * total_pizzas)

theorem pizza_sales_distribution :
  spring_pizzas = 3.75 ∧
  summer_percent * total_pizzas = summer_pizzas ∧
  total_pizzas = summer_pizzas / summer_percent :=
by sorry

end NUMINAMATH_CALUDE_pizza_sales_distribution_l2727_272726


namespace NUMINAMATH_CALUDE_field_ratio_l2727_272771

/-- Given a rectangular field with perimeter 240 meters and width 50 meters,
    prove that the ratio of length to width is 7:5 -/
theorem field_ratio (perimeter width length : ℝ) : 
  perimeter = 240 ∧ width = 50 ∧ perimeter = 2 * (length + width) →
  length / width = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l2727_272771


namespace NUMINAMATH_CALUDE_rearrange_three_of_eight_count_l2727_272784

/-- The number of ways to select and rearrange 3 people out of 8 -/
def rearrange_three_of_eight : ℕ :=
  Nat.choose 8 3 * (3 * 2)

/-- Theorem stating that rearranging 3 people out of 8 has C₈₃ * A³₂ ways -/
theorem rearrange_three_of_eight_count :
  rearrange_three_of_eight = Nat.choose 8 3 * (3 * 2) := by
  sorry

end NUMINAMATH_CALUDE_rearrange_three_of_eight_count_l2727_272784


namespace NUMINAMATH_CALUDE_box_volume_increase_l2727_272761

theorem box_volume_increase (l w h : ℝ) : 
  l * w * h = 5000 →
  2 * (l * w + w * h + l * h) = 1850 →
  4 * (l + w + h) = 240 →
  (l + 3) * (w + 3) * (h + 3) = 8342 := by
sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2727_272761


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2727_272767

-- Define a function to convert a number from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  -- Implementation details are omitted
  sorry

-- Define a function to convert a number from base 9 to base 10
def base9ToBase10 (n : Nat) : Nat :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem base_conversion_subtraction :
  base8ToBase10 76432 - base9ToBase10 2541 = 30126 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2727_272767


namespace NUMINAMATH_CALUDE_room_width_calculation_l2727_272778

/-- Given a rectangular room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 300)
    (h3 : total_cost = 6187.5) :
    total_cost / cost_per_sqm / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2727_272778


namespace NUMINAMATH_CALUDE_range_of_a_l2727_272756

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then (5 - a) * n - 11 else a^(n - 4)

theorem range_of_a (a : ℝ) :
  (∀ n m : ℕ, n < m → sequence_a a n < sequence_a a m) →
  2 < a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2727_272756


namespace NUMINAMATH_CALUDE_joeys_reading_assignment_l2727_272703

/-- The number of pages Joey must read after his break -/
def pages_after_break : ℕ := 9

/-- The percentage of pages Joey reads before taking a break -/
def percentage_before_break : ℚ := 70 / 100

theorem joeys_reading_assignment :
  ∃ (total_pages : ℕ),
    (1 - percentage_before_break) * total_pages = pages_after_break ∧
    total_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_joeys_reading_assignment_l2727_272703


namespace NUMINAMATH_CALUDE_ball_bounce_problem_l2727_272782

def bounce_height (k : ℕ) : ℝ :=
  1500 * (0.40 ^ k) * (0.95 ^ (k * (k - 1) / 2))

def is_first_bounce_below_two (k : ℕ) : Prop :=
  bounce_height k < 2 ∧ ∀ j : ℕ, j < k → bounce_height j ≥ 2

theorem ball_bounce_problem :
  ∃ k : ℕ, is_first_bounce_below_two k ∧ k = 6 :=
sorry

end NUMINAMATH_CALUDE_ball_bounce_problem_l2727_272782


namespace NUMINAMATH_CALUDE_temperature_difference_l2727_272764

theorem temperature_difference (M L N : ℝ) : 
  M = L + N →  -- Minneapolis is N degrees warmer than St. Louis at noon
  (∃ (M_4 L_4 : ℝ), 
    M_4 = M - 5 ∧  -- Minneapolis temperature falls by 5 degrees at 4:00
    L_4 = L + 3 ∧  -- St. Louis temperature rises by 3 degrees at 4:00
    abs (M_4 - L_4) = 2) →  -- Temperatures differ by 2 degrees at 4:00
  (N = 10 ∨ N = 6) ∧ N * (16 - N) = 60 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l2727_272764


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l2727_272724

theorem chicken_wings_distribution (num_friends : ℕ) (total_wings : ℕ) :
  num_friends = 9 →
  total_wings = 27 →
  ∃ (wings_per_person : ℕ), 
    wings_per_person * num_friends = total_wings ∧
    wings_per_person = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l2727_272724


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2727_272760

theorem sum_of_reciprocal_relations (x y : ℝ) 
  (h1 : 1/x + 1/y = 1) 
  (h2 : 1/x - 1/y = 9) : 
  x + y = -1/20 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2727_272760


namespace NUMINAMATH_CALUDE_last_k_digits_power_l2727_272736

theorem last_k_digits_power (A B : ℤ) (k n : ℕ) (h : A ≡ B [ZMOD 10^k]) :
  A^n ≡ B^n [ZMOD 10^k] := by sorry

end NUMINAMATH_CALUDE_last_k_digits_power_l2727_272736


namespace NUMINAMATH_CALUDE_equation_solutions_l2727_272702

theorem equation_solutions : 
  {x : ℝ | (x^3 - 3*x^2)/(x^2 - 4*x + 4) + x = -3} = {-2, 3/2} :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2727_272702


namespace NUMINAMATH_CALUDE_trapezoid_area_l2727_272768

/-- The area of a trapezoid bounded by y = 2x, y = 10, y = 5, and the y-axis -/
theorem trapezoid_area : ∃ (A : ℝ), A = 18.75 ∧ 
  A = ((5 - 0) + (10 - 5)) / 2 * 5 ∧
  (∀ x y : ℝ, (y = 2*x ∨ y = 10 ∨ y = 5 ∨ x = 0) → 
    0 ≤ x ∧ x ≤ 5 ∧ 5 ≤ y ∧ y ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2727_272768


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2727_272776

theorem wire_service_reporters (total : ℕ) (h_total : total > 0) :
  let local_politics := (18 : ℚ) / 100 * total
  let no_politics := (70 : ℚ) / 100 * total
  let cover_politics := total - no_politics
  let cover_not_local := cover_politics - local_politics
  (cover_not_local / cover_politics) = (2 : ℚ) / 5 := by
sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2727_272776


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l2727_272747

/-- Given an ellipse with equation x^2/25 + y^2/9 = 1, 
    this theorem states that for any point P on the ellipse 
    (distinct from the endpoints of the major axis), 
    the product of the slopes of the lines connecting P 
    to the endpoints of the major axis is -9/25. -/
theorem ellipse_slope_product : 
  ∀ (x y : ℝ), 
  x^2/25 + y^2/9 = 1 →  -- P is on the ellipse
  x ≠ 5 →              -- P is not the right endpoint
  x ≠ -5 →             -- P is not the left endpoint
  ∃ (m₁ m₂ : ℝ),       -- slopes exist
  (m₁ = y / (x - 5) ∧ m₂ = y / (x + 5)) ∧  -- definition of slopes
  m₁ * m₂ = -9/25 :=   -- product of slopes
by sorry


end NUMINAMATH_CALUDE_ellipse_slope_product_l2727_272747


namespace NUMINAMATH_CALUDE_perfect_square_solutions_l2727_272738

theorem perfect_square_solutions : 
  {n : ℤ | ∃ k : ℤ, n^2 + 8*n + 44 = k^2} = {2, -10} := by sorry

end NUMINAMATH_CALUDE_perfect_square_solutions_l2727_272738


namespace NUMINAMATH_CALUDE_intersection_M_P_l2727_272725

def M : Set ℝ := {x | x^2 = x}
def P : Set ℝ := {x | |x - 1| = 1}

theorem intersection_M_P : M ∩ P = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_P_l2727_272725


namespace NUMINAMATH_CALUDE_shooting_competition_expected_score_l2727_272780

/-- Represents the scoring system for the shooting competition -/
structure ScoringSystem where
  miss_both : ℕ
  hit_one : ℕ
  hit_both : ℕ

/-- Calculates the expected score for a shooting competition -/
def expected_score (scoring : ScoringSystem) (hit_rate : ℚ) : ℚ :=
  let miss_prob : ℚ := 1 - hit_rate
  let p_miss_both : ℚ := miss_prob * miss_prob
  let p_hit_one : ℚ := 2 * hit_rate * miss_prob
  let p_hit_both : ℚ := hit_rate * hit_rate
  scoring.miss_both * p_miss_both + scoring.hit_one * p_hit_one + scoring.hit_both * p_hit_both

/-- Theorem stating the expected score for the given shooting competition -/
theorem shooting_competition_expected_score :
  let scoring : ScoringSystem := ⟨0, 10, 15⟩
  let hit_rate : ℚ := 4/5
  expected_score scoring hit_rate = 64/5 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_expected_score_l2727_272780


namespace NUMINAMATH_CALUDE_percent_decrease_l2727_272794

theorem percent_decrease (original_price sale_price : ℝ) (h : original_price > 0) :
  let decrease := original_price - sale_price
  let percent_decrease := (decrease / original_price) * 100
  original_price = 100 ∧ sale_price = 75 → percent_decrease = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_l2727_272794


namespace NUMINAMATH_CALUDE_least_repeating_digits_of_seven_thirteenths_l2727_272706

theorem least_repeating_digits_of_seven_thirteenths : 
  (∀ n : ℕ, 0 < n → n < 6 → (10^n : ℤ) % 13 ≠ 1) ∧ (10^6 : ℤ) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_repeating_digits_of_seven_thirteenths_l2727_272706


namespace NUMINAMATH_CALUDE_second_car_distance_rate_l2727_272712

/-- Represents the race scenario with two cars and a motorcycle --/
structure RaceScenario where
  l : ℝ  -- Length of the race distance
  v1 : ℝ  -- Speed of the first car
  v2 : ℝ  -- Speed of the second car
  vM : ℝ  -- Speed of the motorcycle

/-- Conditions of the race --/
def race_conditions (r : RaceScenario) : Prop :=
  r.l > 0 ∧  -- The race distance is positive
  r.v1 > 0 ∧ r.v2 > 0 ∧ r.vM > 0 ∧  -- All speeds are positive
  r.l / r.v2 - r.l / r.v1 = 1/60 ∧  -- Second car takes 1 minute longer than the first car
  r.v1 = 4 * r.vM ∧  -- First car is 4 times faster than the motorcycle
  r.v2 / 60 - r.vM / 60 = r.l / 6 ∧  -- Second car covers 1/6 more distance per minute than the motorcycle
  r.l / r.vM < 10  -- Motorcycle covers the distance in less than 10 minutes

/-- The theorem to be proved --/
theorem second_car_distance_rate (r : RaceScenario) :
  race_conditions r → r.v2 / 60 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_second_car_distance_rate_l2727_272712


namespace NUMINAMATH_CALUDE_power_equality_l2727_272716

theorem power_equality (k : ℕ) : 9^4 = 3^k → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2727_272716


namespace NUMINAMATH_CALUDE_prop_truth_values_l2727_272777

theorem prop_truth_values (p q : Prop) :
  ¬(p ∨ (¬q)) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_prop_truth_values_l2727_272777


namespace NUMINAMATH_CALUDE_company_fund_problem_l2727_272793

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) :
  (80 * n = initial_fund + 8) →
  (70 * n + 160 = initial_fund) →
  initial_fund = 1352 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_problem_l2727_272793


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2727_272741

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | ∃ y, y = x^2 - 2*x + 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2727_272741


namespace NUMINAMATH_CALUDE_regular_octahedron_parallel_edges_l2727_272733

structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  faces : Finset (Fin 3 → Fin 6)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 12
  face_count : faces.card = 8

def parallel_edges (o : RegularOctahedron) : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6) :=
  sorry

theorem regular_octahedron_parallel_edges (o : RegularOctahedron) :
  (parallel_edges o).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_octahedron_parallel_edges_l2727_272733


namespace NUMINAMATH_CALUDE_compute_expression_l2727_272799

theorem compute_expression : 9 * (2 / 7 : ℚ)^4 = 144 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2727_272799


namespace NUMINAMATH_CALUDE_triangle_area_with_arithmetic_sides_l2727_272739

/-- Given a triangle ABC with one angle of 120° and sides in arithmetic progression with common difference 2, its area is 15√3/4 -/
theorem triangle_area_with_arithmetic_sides : ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  ∃ (θ : ℝ), θ = 2 * π / 3 →
  ∃ (d : ℝ), d = 2 →
  b = a + d ∧ c = b + d →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos θ →
  (1/2) * a * b * Real.sin θ = 15 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_with_arithmetic_sides_l2727_272739


namespace NUMINAMATH_CALUDE_percentage_increase_l2727_272790

theorem percentage_increase (original : ℝ) (final : ℝ) (percentage : ℝ) : 
  original = 900 →
  final = 1080 →
  percentage = ((final - original) / original) * 100 →
  percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2727_272790


namespace NUMINAMATH_CALUDE_no_solution_iff_m_less_than_neg_two_l2727_272711

theorem no_solution_iff_m_less_than_neg_two (m : ℝ) :
  (∀ x : ℝ, ¬(x - m ≤ 2*m + 3 ∧ (x - 1)/2 ≥ m)) ↔ m < -2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_less_than_neg_two_l2727_272711


namespace NUMINAMATH_CALUDE_square_sum_value_l2727_272722

theorem square_sum_value (x y : ℝ) (h1 : x * y = 16) (h2 : x^2 + y^2 = 34) : 
  (x + y)^2 = 66 := by sorry

end NUMINAMATH_CALUDE_square_sum_value_l2727_272722


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_l2727_272720

/-- The problem of Tom's fruit purchase -/
theorem toms_fruit_purchase 
  (apple_kg : ℕ) 
  (apple_rate : ℕ) 
  (mango_rate : ℕ) 
  (total_paid : ℕ) 
  (h1 : apple_kg = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_rate = 75)
  (h4 : total_paid = 1235)
  : ∃ (mango_kg : ℕ), 
    apple_kg * apple_rate + mango_kg * mango_rate = total_paid ∧ 
    mango_kg = 9 := by
  sorry

end NUMINAMATH_CALUDE_toms_fruit_purchase_l2727_272720


namespace NUMINAMATH_CALUDE_triangle_equivalence_l2727_272704

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angles of a triangle -/
def Triangle.angles (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The Nine-point circle of a triangle -/
def Triangle.ninePointCircle (t : Triangle) : Circle := sorry

/-- The Incircle of a triangle -/
def Triangle.incircle (t : Triangle) : Circle := sorry

/-- The Euler Line of a triangle -/
def Triangle.eulerLine (t : Triangle) : Line := sorry

/-- Check if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

/-- Check if one of the angles is 60° -/
def Triangle.hasAngle60 (t : Triangle) : Prop := sorry

/-- Check if the angles are in arithmetic progression -/
def Triangle.anglesInArithmeticProgression (t : Triangle) : Prop := sorry

/-- Check if the common tangent to the Nine-point circle and Incircle is parallel to the Euler Line -/
def Triangle.commonTangentParallelToEulerLine (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_equivalence (t : Triangle) (h : ¬ t.isEquilateral) :
  t.hasAngle60 ↔ t.anglesInArithmeticProgression ∧ t.commonTangentParallelToEulerLine :=
sorry

end NUMINAMATH_CALUDE_triangle_equivalence_l2727_272704


namespace NUMINAMATH_CALUDE_factorization_equality_l2727_272714

theorem factorization_equality (x : ℝ) : (x^2 - 1)^2 - 6*(x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2727_272714


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l2727_272728

/-- Definition of triangular numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l2727_272728


namespace NUMINAMATH_CALUDE_derivative_log2_l2727_272742

-- Define the base-2 logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem derivative_log2 (x : ℝ) (h : x > 0) :
  deriv log2 x = 1 / (x * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_log2_l2727_272742


namespace NUMINAMATH_CALUDE_expression_simplification_l2727_272729

theorem expression_simplification (a b : ℝ) (h : (a + 2)^2 + |b - 1| = 0) :
  (3 * a^2 * b - a * b^2) - (1/2) * (a^2 * b - (2 * a * b^2 - 4)) + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2727_272729


namespace NUMINAMATH_CALUDE_number_thought_of_l2727_272715

theorem number_thought_of : ∃ x : ℝ, (x / 6) + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2727_272715


namespace NUMINAMATH_CALUDE_star_properties_l2727_272749

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- Theorem statement
theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧ 
  (∀ x : ℝ, star x (-1) = x ∧ star (-1) x = x) ∧
  (∀ x : ℝ, star x x = x^2 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l2727_272749


namespace NUMINAMATH_CALUDE_prob_blue_face_four_blue_two_red_l2727_272753

/-- A cube with blue and red faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a blue face on a colored cube -/
def prob_blue_face (cube : ColoredCube) : ℚ :=
  cube.blue_faces / (cube.blue_faces + cube.red_faces)

/-- Theorem: The probability of rolling a blue face on a cube with 4 blue faces and 2 red faces is 2/3 -/
theorem prob_blue_face_four_blue_two_red :
  prob_blue_face ⟨4, 2⟩ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_blue_face_four_blue_two_red_l2727_272753


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2727_272766

/-- Given two natural numbers a and b, their LCM is 2310 and a is 462, prove that their HCF is 1 -/
theorem lcm_hcf_problem (a b : ℕ) (h1 : a = 462) (h2 : Nat.lcm a b = 2310) : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2727_272766


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l2727_272786

theorem quadratic_equation_real_root (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l2727_272786


namespace NUMINAMATH_CALUDE_oven_usage_calculation_l2727_272757

/-- Represents the problem of calculating oven usage time given electricity price, consumption rate, and total cost. -/
def OvenUsage (price : ℝ) (consumption : ℝ) (total_cost : ℝ) : Prop :=
  let hours := total_cost / (price * consumption)
  hours = 25

/-- Theorem stating that given the specific values in the problem, the oven usage time is 25 hours. -/
theorem oven_usage_calculation :
  OvenUsage 0.10 2.4 6 := by
  sorry

end NUMINAMATH_CALUDE_oven_usage_calculation_l2727_272757


namespace NUMINAMATH_CALUDE_turtle_speed_specific_turtle_speed_l2727_272755

/-- Given a race with a hare and a turtle, calculate the turtle's speed -/
theorem turtle_speed (race_distance : ℝ) (hare_speed : ℝ) (head_start : ℝ) : ℝ :=
  let turtle_speed := race_distance / (race_distance / hare_speed + head_start)
  turtle_speed

/-- The turtle's speed in the specific race scenario -/
theorem specific_turtle_speed : 
  turtle_speed 20 10 18 = 1 := by sorry

end NUMINAMATH_CALUDE_turtle_speed_specific_turtle_speed_l2727_272755


namespace NUMINAMATH_CALUDE_cone_areas_l2727_272734

/-- Represents a cone with given slant height and height -/
structure Cone where
  slantHeight : ℝ
  height : ℝ

/-- Calculates the lateral area of a cone -/
def lateralArea (c : Cone) : ℝ := sorry

/-- Calculates the area of the sector when the cone's lateral surface is unfolded -/
def sectorArea (c : Cone) : ℝ := sorry

theorem cone_areas (c : Cone) (h1 : c.slantHeight = 1) (h2 : c.height = 0.8) : 
  lateralArea c = 3/5 * Real.pi ∧ sectorArea c = 3/5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_areas_l2727_272734


namespace NUMINAMATH_CALUDE_isaiah_typing_speed_l2727_272732

theorem isaiah_typing_speed 
  (micah_speed : ℕ) 
  (isaiah_hourly_diff : ℕ) 
  (h1 : micah_speed = 20)
  (h2 : isaiah_hourly_diff = 1200) : 
  (micah_speed * 60 + isaiah_hourly_diff) / 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_isaiah_typing_speed_l2727_272732


namespace NUMINAMATH_CALUDE_cow_spots_l2727_272719

/-- Calculates the total number of spots on a cow given the number of spots on its left side. -/
def totalSpots (leftSpots : ℕ) : ℕ :=
  let rightSpots := 3 * leftSpots + 7
  leftSpots + rightSpots

/-- Theorem stating that a cow with 16 spots on its left side has 71 spots in total. -/
theorem cow_spots : totalSpots 16 = 71 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_l2727_272719


namespace NUMINAMATH_CALUDE_point_on_line_l2727_272798

/-- Given a line passing through points (0,2) and (-4,-1), prove that if (t,7) lies on this line, then t = 20/3 -/
theorem point_on_line (t : ℝ) : 
  (∀ x y : ℝ, (y - 2) / x = (-1 - 2) / (-4 - 0)) → -- Line through (0,2) and (-4,-1)
  ((7 - 2) / t = (-1 - 2) / (-4 - 0)) →             -- (t,7) lies on the line
  t = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l2727_272798


namespace NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_for_lt_neg_two_l2727_272770

theorem abs_gt_one_necessary_not_sufficient_for_lt_neg_two (x : ℝ) :
  (∀ x, x < -2 → |x| > 1) ∧ 
  (∃ x, |x| > 1 ∧ ¬(x < -2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_for_lt_neg_two_l2727_272770


namespace NUMINAMATH_CALUDE_final_apple_count_l2727_272740

def apples_on_tree (initial : ℕ) (picked : ℕ) (new_growth : ℕ) : ℕ :=
  initial - picked + new_growth

theorem final_apple_count :
  apples_on_tree 11 7 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_apple_count_l2727_272740


namespace NUMINAMATH_CALUDE_rachel_reading_homework_l2727_272717

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := by sorry

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 4

/-- The relationship between math and reading homework pages -/
axiom math_reading_relation : math_pages = reading_pages + 2

theorem rachel_reading_homework : reading_pages = 2 := by sorry

end NUMINAMATH_CALUDE_rachel_reading_homework_l2727_272717


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2727_272750

theorem quadratic_inequality (d : ℝ) : 
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ -5/2 < x ∧ x < 1) ↔ d = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2727_272750


namespace NUMINAMATH_CALUDE_regular_polygon_with_120_degree_angles_has_6_sides_l2727_272779

theorem regular_polygon_with_120_degree_angles_has_6_sides :
  ∀ n : ℕ, n > 2 →
  (∀ θ : ℝ, θ = 120 → θ * n = 180 * (n - 2)) →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_120_degree_angles_has_6_sides_l2727_272779
