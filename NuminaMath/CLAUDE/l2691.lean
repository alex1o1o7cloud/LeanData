import Mathlib

namespace NUMINAMATH_CALUDE_thursday_to_wednesday_ratio_l2691_269194

/-- Represents the number of laundry loads washed on each day of the week --/
structure LaundryWeek where
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Defines the conditions for Vincent's laundry week --/
def vincentLaundryWeek (w : LaundryWeek) : Prop :=
  w.wednesday = 6 ∧
  w.friday = w.thursday / 2 ∧
  w.saturday = w.wednesday / 3 ∧
  w.wednesday + w.thursday + w.friday + w.saturday = 26

/-- Theorem stating that the ratio of loads washed on Thursday to Wednesday is 2:1 --/
theorem thursday_to_wednesday_ratio (w : LaundryWeek) 
  (h : vincentLaundryWeek w) : w.thursday = 2 * w.wednesday := by
  sorry

end NUMINAMATH_CALUDE_thursday_to_wednesday_ratio_l2691_269194


namespace NUMINAMATH_CALUDE_line_slope_is_two_l2691_269171

/-- Given a line ax + y - 4 = 0 passing through the point (-1, 2), prove that its slope is 2 -/
theorem line_slope_is_two (a : ℝ) : 
  (a * (-1) + 2 - 4 = 0) → -- Line passes through (-1, 2)
  (∃ m b : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = m * x + b) → -- Line can be written in slope-intercept form
  (∃ m : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = m * x + 4) → -- Specific y-intercept is 4
  (∃ m : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = 2 * x + 4) -- Slope is 2
  := by sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l2691_269171


namespace NUMINAMATH_CALUDE_equilateral_triangle_from_polynomial_roots_l2691_269120

theorem equilateral_triangle_from_polynomial_roots (a b c : ℂ) :
  (∀ z : ℂ, z^3 + 5*z + 7 = 0 ↔ z = a ∨ z = b ∨ z = c) →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  Complex.abs (a - b) = Complex.abs (b - c) ∧ 
  Complex.abs (b - c) = Complex.abs (c - a) →
  (Complex.abs (a - b))^2 = 225 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_from_polynomial_roots_l2691_269120


namespace NUMINAMATH_CALUDE_number_times_one_fourth_squared_equals_four_cubed_l2691_269127

theorem number_times_one_fourth_squared_equals_four_cubed (x : ℝ) : 
  x * (1/4)^2 = 4^3 ↔ x = 1024 := by
  sorry

end NUMINAMATH_CALUDE_number_times_one_fourth_squared_equals_four_cubed_l2691_269127


namespace NUMINAMATH_CALUDE_transformation_sequence_l2691_269147

def rotate_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_y (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (z, y, -x)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

theorem transformation_sequence :
  (reflect_xz ∘ rotate_y ∘ reflect_yz ∘ reflect_xy ∘ rotate_x) initial_point = (2, 2, -2) := by
  sorry

end NUMINAMATH_CALUDE_transformation_sequence_l2691_269147


namespace NUMINAMATH_CALUDE_cube_vertical_faces_same_color_prob_l2691_269128

/-- Represents the probability of painting a face blue -/
def blue_prob : ℚ := 1/3

/-- Represents the probability of painting a face red -/
def red_prob : ℚ := 2/3

/-- Represents the number of faces on a cube -/
def num_faces : ℕ := 6

/-- Represents the number of vertical faces when a cube is placed on a horizontal surface -/
def num_vertical_faces : ℕ := 4

/-- Calculates the probability of all faces being the same color -/
def all_same_color_prob : ℚ := red_prob^num_faces + blue_prob^num_faces

/-- Calculates the probability of vertical faces being one color and top/bottom being another -/
def mixed_color_prob : ℚ := 3 * (red_prob^num_vertical_faces * blue_prob^(num_faces - num_vertical_faces) +
                                 blue_prob^num_vertical_faces * red_prob^(num_faces - num_vertical_faces))

/-- The main theorem stating the probability of the cube having all four vertical faces
    the same color when placed on a horizontal surface -/
theorem cube_vertical_faces_same_color_prob :
  all_same_color_prob + mixed_color_prob = 789/6561 := by sorry

end NUMINAMATH_CALUDE_cube_vertical_faces_same_color_prob_l2691_269128


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_fraction_l2691_269169

theorem ceiling_neg_sqrt_fraction : ⌈-Real.sqrt (36 / 9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_fraction_l2691_269169


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l2691_269111

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (a b : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : parallel a α) 
  (h4 : perpendicular a β) : 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l2691_269111


namespace NUMINAMATH_CALUDE_transmission_time_approx_seven_minutes_l2691_269197

/-- Represents the data transmission scenario -/
structure DataTransmission where
  num_blocks : ℕ
  chunks_per_block : ℕ
  transmission_rate : ℕ
  delay_per_block : ℕ

/-- Calculates the total transmission time in minutes -/
def total_transmission_time (dt : DataTransmission) : ℚ :=
  let total_chunks := dt.num_blocks * dt.chunks_per_block
  let transmission_time := total_chunks / dt.transmission_rate
  let total_delay := dt.num_blocks * dt.delay_per_block
  (transmission_time + total_delay) / 60

/-- Theorem stating that the transmission time is approximately 7 minutes -/
theorem transmission_time_approx_seven_minutes (dt : DataTransmission) 
  (h1 : dt.num_blocks = 80)
  (h2 : dt.chunks_per_block = 600)
  (h3 : dt.transmission_rate = 150)
  (h4 : dt.delay_per_block = 1) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |total_transmission_time dt - 7| < ε :=
sorry

end NUMINAMATH_CALUDE_transmission_time_approx_seven_minutes_l2691_269197


namespace NUMINAMATH_CALUDE_circle_symmetry_l2691_269164

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x + 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), 
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
   symmetry_line ((x + x₀) / 2) ((y + y₀) / 2) ∧
   (y - y₀) = -(x - x₀)) →
  symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2691_269164


namespace NUMINAMATH_CALUDE_tom_apple_purchase_l2691_269186

/-- The price of apples per kg -/
def apple_price : ℕ := 70

/-- The amount of mangoes Tom bought in kg -/
def mango_amount : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The total amount Tom paid -/
def total_paid : ℕ := 1055

/-- Theorem stating that Tom purchased 8 kg of apples -/
theorem tom_apple_purchase :
  ∃ (x : ℕ), x * apple_price + mango_amount * mango_price = total_paid ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_apple_purchase_l2691_269186


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2691_269110

theorem unique_solution_for_equation : ∃! (x y z : ℕ),
  (x < 10 ∧ y < 10 ∧ z < 10) ∧
  (10 * x + 5 < 100) ∧
  (300 ≤ 300 + 10 * y + z) ∧
  (300 + 10 * y + z < 400) ∧
  ((10 * x + 5) * (300 + 10 * y + z) = 7850) ∧
  x = 2 ∧ y = 1 ∧ z = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2691_269110


namespace NUMINAMATH_CALUDE_unique_prime_pair_l2691_269166

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime (p^2 + 2*p*q^2 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l2691_269166


namespace NUMINAMATH_CALUDE_factorization_of_expression_l2691_269108

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

axiom natural_prime_factorization : ∀ n : ℕ, n > 1 → ∃ (primes : List ℕ), (∀ p ∈ primes, is_prime p) ∧ n = primes.prod

theorem factorization_of_expression : 2^4 * 3^2 - 1 = 11 * 13 := by sorry

end NUMINAMATH_CALUDE_factorization_of_expression_l2691_269108


namespace NUMINAMATH_CALUDE_fraction_difference_equals_eight_sqrt_three_l2691_269176

theorem fraction_difference_equals_eight_sqrt_three :
  let a : ℝ := 2 + Real.sqrt 3
  let b : ℝ := 2 - Real.sqrt 3
  (a / b) - (b / a) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_eight_sqrt_three_l2691_269176


namespace NUMINAMATH_CALUDE_sara_letters_problem_l2691_269158

theorem sara_letters_problem (january february march total : ℕ) :
  february = 9 →
  march = 3 * january →
  total = january + february + march →
  total = 33 →
  january = 6 :=
by sorry

end NUMINAMATH_CALUDE_sara_letters_problem_l2691_269158


namespace NUMINAMATH_CALUDE_left_square_side_length_l2691_269126

/-- Given three squares with specific side length relationships, prove the left square's side length --/
theorem left_square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 → 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l2691_269126


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2691_269195

def apples_handed_out (initial_apples : ℕ) (pies_made : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - pies_made * apples_per_pie

theorem cafeteria_apples 
  (initial_apples : ℕ) 
  (pies_made : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 50) 
  (h2 : pies_made = 9) 
  (h3 : apples_per_pie = 5) :
  apples_handed_out initial_apples pies_made apples_per_pie = 5 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2691_269195


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2691_269116

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 8) * π * r^2 → r = 16 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2691_269116


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2691_269145

def complex_number_quadrant (z : ℂ) : Prop :=
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 + 2 * Complex.I) / Complex.I
  complex_number_quadrant z :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2691_269145


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2691_269123

/-- Calculates the final price of a bicycle after two consecutive discounts. -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that a $200 bicycle, after a 40% discount followed by a 25% discount, costs $90. -/
theorem bicycle_price_after_discounts :
  final_price 200 0.4 0.25 = 90 := by
  sorry

#eval final_price 200 0.4 0.25

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2691_269123


namespace NUMINAMATH_CALUDE_platform_completion_time_l2691_269177

/-- Represents the number of days required to complete a portion of a project given a number of workers -/
def days_to_complete (workers : ℕ) (portion : ℚ) : ℚ :=
  sorry

theorem platform_completion_time :
  let initial_workers : ℕ := 90
  let initial_days : ℕ := 6
  let initial_portion : ℚ := 1/2
  let remaining_workers : ℕ := 60
  let remaining_portion : ℚ := 1/2
  days_to_complete initial_workers initial_portion = initial_days →
  days_to_complete remaining_workers remaining_portion = 9 :=
by sorry

end NUMINAMATH_CALUDE_platform_completion_time_l2691_269177


namespace NUMINAMATH_CALUDE_expression_evaluation_l2691_269156

theorem expression_evaluation (x : ℚ) (h : x = -4) : 
  (1 - 4 / (x + 3)) / ((x^2 - 1) / (x^2 + 6*x + 9)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2691_269156


namespace NUMINAMATH_CALUDE_jackie_has_ten_apples_l2691_269191

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := adam_apples + 1

/-- Theorem: Jackie has 10 apples -/
theorem jackie_has_ten_apples : jackie_apples = 10 := by
  sorry

end NUMINAMATH_CALUDE_jackie_has_ten_apples_l2691_269191


namespace NUMINAMATH_CALUDE_square_perimeter_9cm_l2691_269198

/-- Calculates the perimeter of a square given its side length -/
def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The perimeter of a square with side length 9 cm is 36 cm -/
theorem square_perimeter_9cm : square_perimeter 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_9cm_l2691_269198


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2691_269185

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
def ω : ℂ := sorry

/-- The polynomial x^11 + Ax^2 + B -/
def P (A B : ℝ) (x : ℂ) : ℂ := x^11 + A * x^2 + B

/-- The polynomial x^2 + x + 1 -/
def Q (x : ℂ) : ℂ := x^2 + x + 1

theorem polynomial_divisibility (A B : ℝ) :
  (∀ x, Q x = 0 → P A B x = 0) → A = -1 ∧ B = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2691_269185


namespace NUMINAMATH_CALUDE_circle_tangency_radius_l2691_269199

theorem circle_tangency_radius 
  (d1 d2 r1 r2 r y : ℝ) 
  (h1 : d1 < d2) 
  (h2 : r1 = d1 / 2) 
  (h3 : r2 = d2 / 2) 
  (h4 : (r + r1)^2 = (r - 2*r2 - r1)^2 + y^2) 
  (h5 : (r + r2)^2 = (r - r2)^2 + y^2) : 
  r = ((d1 + d2) * d2) / (2 * d1) := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_radius_l2691_269199


namespace NUMINAMATH_CALUDE_remainder_r_15_minus_1_l2691_269150

theorem remainder_r_15_minus_1 (r : ℝ) : (r^15 - 1) % (r - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_r_15_minus_1_l2691_269150


namespace NUMINAMATH_CALUDE_linear_function_theorem_l2691_269174

/-- A linear function f(x) = ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) : ℝ := a

theorem linear_function_theorem (a b : ℝ) :
  f a b 1 = 2 ∧ f_derivative a = 2 → f a b 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l2691_269174


namespace NUMINAMATH_CALUDE_triangle_abc_area_l2691_269117

/-- Triangle ABC with vertices A(0,0), B(1,7), and C(0,8) has an area of 28 square units -/
theorem triangle_abc_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 7)
  let C : ℝ × ℝ := (0, 8)
  let triangle_area := (1/2) * |(C.2 - A.2)| * |(B.1 - A.1)|
  triangle_area = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l2691_269117


namespace NUMINAMATH_CALUDE_project_completion_proof_l2691_269130

/-- Represents the number of days to complete the project -/
def project_completion_time : ℕ := 11

/-- Represents A's completion rate per day -/
def rate_A : ℚ := 1 / 20

/-- Represents B's initial completion rate per day -/
def rate_B : ℚ := 1 / 30

/-- Represents C's completion rate per day -/
def rate_C : ℚ := 1 / 40

/-- Represents B's doubled completion rate -/
def rate_B_doubled : ℚ := 2 * rate_B

/-- Represents the time A quits before project completion -/
def time_A_quits_before : ℕ := 10

/-- Theorem stating that the project will be completed in 11 days -/
theorem project_completion_proof :
  let total_work : ℚ := 1
  let combined_rate : ℚ := rate_A + rate_B + rate_C
  let final_rate : ℚ := rate_B_doubled + rate_C
  (project_completion_time - time_A_quits_before) * combined_rate +
  time_A_quits_before * final_rate = total_work :=
by sorry


end NUMINAMATH_CALUDE_project_completion_proof_l2691_269130


namespace NUMINAMATH_CALUDE_jeremy_speed_l2691_269187

/-- Given a distance of 20 kilometers and a time of 10 hours, prove that the speed is 2 kilometers per hour. -/
theorem jeremy_speed (distance : ℝ) (time : ℝ) (h1 : distance = 20) (h2 : time = 10) :
  distance / time = 2 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_speed_l2691_269187


namespace NUMINAMATH_CALUDE_estate_division_percentage_l2691_269122

/-- Represents the estate division problem --/
structure EstateDivision where
  amount₁ : ℝ  -- Amount received by the first person
  range : ℝ    -- Smallest possible range between highest and lowest amounts
  percentage : ℝ -- Percentage stipulation

/-- The estate division problem satisfies the given conditions --/
def valid_division (e : EstateDivision) : Prop :=
  e.amount₁ = 20000 ∧ 
  e.range = 10000 ∧ 
  0 < e.percentage ∧ 
  e.percentage < 100

/-- The theorem stating that the percentage stipulation is 25% --/
theorem estate_division_percentage (e : EstateDivision) 
  (h : valid_division e) : e.percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_estate_division_percentage_l2691_269122


namespace NUMINAMATH_CALUDE_tim_extra_running_days_l2691_269190

def extra_running_days (original_days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours / hours_per_day) - original_days

theorem tim_extra_running_days :
  extra_running_days 3 2 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tim_extra_running_days_l2691_269190


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_twelve_l2691_269143

/-- The number of ways to arrange 4 people in a row of 4 seats, 
    where 2 specific people must sit next to each other. -/
def seating_arrangements : ℕ := 12

/-- Theorem stating that the number of seating arrangements is 12. -/
theorem seating_arrangements_eq_twelve : seating_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_twelve_l2691_269143


namespace NUMINAMATH_CALUDE_remainder_2503_div_28_l2691_269107

theorem remainder_2503_div_28 : 2503 % 28 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2503_div_28_l2691_269107


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l2691_269184

theorem perfect_cube_units_digits : ∀ d : Fin 10, ∃ n : ℤ, (n ^ 3 : ℤ) % 10 = d.val :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l2691_269184


namespace NUMINAMATH_CALUDE_smallest_two_digit_k_for_45k_perfect_square_l2691_269144

/-- A number is a perfect square if it has an integer square root -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A number is a two-digit positive integer if it's between 10 and 99 inclusive -/
def is_two_digit_positive (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_two_digit_k_for_45k_perfect_square :
  ∀ k : ℕ, is_two_digit_positive k → is_perfect_square (45 * k) → k ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_k_for_45k_perfect_square_l2691_269144


namespace NUMINAMATH_CALUDE_wife_cookie_percentage_l2691_269105

theorem wife_cookie_percentage (total_cookies : ℕ) (daughter_cookies : ℕ) (uneaten_cookies : ℕ) :
  total_cookies = 200 →
  daughter_cookies = 40 →
  uneaten_cookies = 50 →
  ∃ (wife_percentage : ℚ),
    wife_percentage = 30 ∧
    (total_cookies - (wife_percentage / 100) * total_cookies - daughter_cookies) / 2 = uneaten_cookies :=
by sorry

end NUMINAMATH_CALUDE_wife_cookie_percentage_l2691_269105


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2691_269196

theorem arithmetic_square_root_of_16 : 
  ∃ x : ℝ, x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2691_269196


namespace NUMINAMATH_CALUDE_sum_a_3000_l2691_269172

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 17 = 0 then 18
  else if n % 15 = 0 ∧ n % 17 = 0 then 21
  else 0

theorem sum_a_3000 :
  (Finset.range 3000).sum (fun n => a (n + 1)) = 888 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_3000_l2691_269172


namespace NUMINAMATH_CALUDE_min_value_product_l2691_269137

theorem min_value_product (a b c x y z : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)
  (sum_abc : a + b + c = 1)
  (sum_xyz : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≥ -1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l2691_269137


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2691_269102

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) :
  (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2691_269102


namespace NUMINAMATH_CALUDE_units_digit_of_19_times_37_l2691_269170

theorem units_digit_of_19_times_37 : (19 * 37) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_19_times_37_l2691_269170


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2691_269192

theorem ratio_sum_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 4) (h3 : a * b + b * c + c * a = 13) : b * c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2691_269192


namespace NUMINAMATH_CALUDE_favorite_subject_count_l2691_269179

theorem favorite_subject_count (total : ℕ) (math_fraction : ℚ) (english_fraction : ℚ)
  (science_fraction : ℚ) (h_total : total = 30) (h_math : math_fraction = 1/5)
  (h_english : english_fraction = 1/3) (h_science : science_fraction = 1/7) :
  total - (total * math_fraction).floor - (total * english_fraction).floor -
  ((total - (total * math_fraction).floor - (total * english_fraction).floor) * science_fraction).floor = 12 :=
by sorry

end NUMINAMATH_CALUDE_favorite_subject_count_l2691_269179


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2691_269159

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2691_269159


namespace NUMINAMATH_CALUDE_unique_intersection_and_geometric_progression_l2691_269100

noncomputable section

def f (x : ℝ) : ℝ := x / Real.exp x
def g (x : ℝ) : ℝ := Real.log x / x

theorem unique_intersection_and_geometric_progression :
  (∃! x : ℝ, f x = g x) ∧
  (∀ a : ℝ, 0 < a → a < Real.exp (-1) →
    (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
      f x₁ = a ∧ g x₂ = a ∧ f x₃ = a →
      ∃ r : ℝ, x₂ = x₁ * r ∧ x₃ = x₂ * r)) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_and_geometric_progression_l2691_269100


namespace NUMINAMATH_CALUDE_toothpaste_cost_is_three_l2691_269157

/-- Represents the shopping scenario with given conditions -/
structure Shopping where
  budget : ℕ
  showerGelPrice : ℕ
  showerGelCount : ℕ
  laundryDetergentPrice : ℕ
  remaining : ℕ

/-- Calculates the cost of toothpaste given the shopping conditions -/
def toothpasteCost (s : Shopping) : ℕ :=
  s.budget - s.remaining - (s.showerGelPrice * s.showerGelCount) - s.laundryDetergentPrice

/-- Theorem stating that the toothpaste costs $3 under the given conditions -/
theorem toothpaste_cost_is_three :
  let s : Shopping := {
    budget := 60,
    showerGelPrice := 4,
    showerGelCount := 4,
    laundryDetergentPrice := 11,
    remaining := 30
  }
  toothpasteCost s = 3 := by sorry

end NUMINAMATH_CALUDE_toothpaste_cost_is_three_l2691_269157


namespace NUMINAMATH_CALUDE_farm_pets_after_changes_l2691_269168

/-- Calculates the total number of pets after changes to a farm's pet population -/
theorem farm_pets_after_changes 
  (initial_dogs : ℕ) 
  (initial_fish : ℕ) 
  (initial_cats : ℕ) 
  (dogs_left : ℕ) 
  (rabbits_added : ℕ) 
  (h_initial_dogs : initial_dogs = 43)
  (h_initial_fish : initial_fish = 72)
  (h_initial_cats : initial_cats = 34)
  (h_dogs_left : dogs_left = 5)
  (h_rabbits_added : rabbits_added = 10) :
  initial_dogs - dogs_left + 2 * initial_fish + initial_cats + rabbits_added = 226 := by
  sorry

end NUMINAMATH_CALUDE_farm_pets_after_changes_l2691_269168


namespace NUMINAMATH_CALUDE_roger_step_goal_time_l2691_269132

/-- Represents the number of steps Roger can walk in 30 minutes -/
def steps_per_30_min : ℕ := 2000

/-- Represents Roger's daily step goal -/
def daily_goal : ℕ := 10000

/-- Represents the time in minutes it takes Roger to reach his daily goal -/
def time_to_reach_goal : ℕ := 150

/-- Theorem stating that the time required for Roger to reach his daily goal is 150 minutes -/
theorem roger_step_goal_time : 
  (daily_goal / steps_per_30_min) * 30 = time_to_reach_goal :=
by sorry

end NUMINAMATH_CALUDE_roger_step_goal_time_l2691_269132


namespace NUMINAMATH_CALUDE_first_person_work_days_l2691_269103

-- Define the work rates
def work_rate_prakash : ℚ := 1 / 40
def work_rate_together : ℚ := 1 / 15

-- Define the theorem
theorem first_person_work_days :
  ∃ (x : ℚ), 
    x > 0 ∧ 
    (1 / x) + work_rate_prakash = work_rate_together ∧ 
    x = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_person_work_days_l2691_269103


namespace NUMINAMATH_CALUDE_adams_trivia_score_l2691_269162

/-- Adam's trivia game score calculation -/
theorem adams_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
  first_half = 8 →
  second_half = 2 →
  points_per_question = 8 →
  (first_half + second_half) * points_per_question = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_adams_trivia_score_l2691_269162


namespace NUMINAMATH_CALUDE_gcd_360_504_l2691_269136

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l2691_269136


namespace NUMINAMATH_CALUDE_blue_sequins_count_l2691_269114

/-- The number of blue sequins in each row of Jane's costume. -/
def blue_sequins_per_row : ℕ := 
  let total_sequins : ℕ := 162
  let blue_rows : ℕ := 6
  let purple_rows : ℕ := 5
  let purple_per_row : ℕ := 12
  let green_rows : ℕ := 9
  let green_per_row : ℕ := 6
  (total_sequins - purple_rows * purple_per_row - green_rows * green_per_row) / blue_rows

theorem blue_sequins_count : blue_sequins_per_row = 8 := by
  sorry

end NUMINAMATH_CALUDE_blue_sequins_count_l2691_269114


namespace NUMINAMATH_CALUDE_matrix_zero_product_implies_zero_multiplier_l2691_269181

theorem matrix_zero_product_implies_zero_multiplier 
  (A B : Matrix (Fin 3) (Fin 3) ℂ) 
  (hB : B ≠ 0) 
  (hAB : A * B = 0) : 
  ∃ D : Matrix (Fin 3) (Fin 3) ℂ, D ≠ 0 ∧ A * D = 0 ∧ D * A = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_zero_product_implies_zero_multiplier_l2691_269181


namespace NUMINAMATH_CALUDE_budgets_equal_after_6_years_l2691_269175

def initial_budget_Q : ℕ := 540000
def initial_budget_V : ℕ := 780000
def annual_increase_Q : ℕ := 30000
def annual_decrease_V : ℕ := 10000

def budget_Q (years : ℕ) : ℕ := initial_budget_Q + annual_increase_Q * years
def budget_V (years : ℕ) : ℕ := initial_budget_V - annual_decrease_V * years

theorem budgets_equal_after_6_years :
  ∃ (years : ℕ), years = 6 ∧ budget_Q years = budget_V years :=
sorry

end NUMINAMATH_CALUDE_budgets_equal_after_6_years_l2691_269175


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l2691_269142

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 72) 
  (h_hcf : Nat.gcd a b = 6) : 
  a * b = 432 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l2691_269142


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2691_269173

/-- Given real numbers x and y satisfying |x-4| + √(y-10) = 0,
    prove that the perimeter of an isosceles triangle with side lengths x, y, and y is 24. -/
theorem isosceles_triangle_perimeter (x y : ℝ) 
  (h : |x - 4| + Real.sqrt (y - 10) = 0) : 
  x + y + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2691_269173


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_l2691_269180

-- Define the regular polygons we're considering
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | Pentagon
  | Hexagon

-- Function to calculate the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℚ :=
  match p with
  | RegularPolygon.EquilateralTriangle => 60
  | RegularPolygon.Square => 90
  | RegularPolygon.Pentagon => 108
  | RegularPolygon.Hexagon => 120

-- Define what it means for a shape to be able to tile a plane
def canTilePlane (p : RegularPolygon) : Prop :=
  ∃ (n : ℕ), n * interiorAngle p = 360

-- Theorem stating that only the pentagon cannot tile the plane
theorem pentagon_cannot_tile :
  ∀ p : RegularPolygon,
    ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
by sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_l2691_269180


namespace NUMINAMATH_CALUDE_last_two_digits_squared_l2691_269193

theorem last_two_digits_squared (n : ℤ) : 
  (n * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2 % 100 = 76 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_squared_l2691_269193


namespace NUMINAMATH_CALUDE_only_valid_solutions_l2691_269124

/-- A pair of natural numbers (m, n) is a valid solution if both n^2 + 4m and m^2 + 5n are perfect squares. -/
def is_valid_solution (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), n^2 + 4*m = a^2 ∧ m^2 + 5*n = b^2

/-- The set of all valid solutions. -/
def valid_solutions : Set (ℕ × ℕ) :=
  {p | is_valid_solution p.1 p.2}

/-- The theorem stating that the only valid solutions are (2,1), (22,9), and (9,8). -/
theorem only_valid_solutions :
  valid_solutions = {(2, 1), (22, 9), (9, 8)} :=
by sorry

end NUMINAMATH_CALUDE_only_valid_solutions_l2691_269124


namespace NUMINAMATH_CALUDE_intersection_A_B_l2691_269183

def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2691_269183


namespace NUMINAMATH_CALUDE_a_2017_equals_2_l2691_269149

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (sequence_a n - 1) / (sequence_a n + 1)

theorem a_2017_equals_2 : sequence_a 2016 = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_2017_equals_2_l2691_269149


namespace NUMINAMATH_CALUDE_buratino_malvina_equation_l2691_269139

theorem buratino_malvina_equation (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_buratino_malvina_equation_l2691_269139


namespace NUMINAMATH_CALUDE_candy_distribution_l2691_269165

theorem candy_distribution (total_candies : ℕ) (total_children : ℕ) (lollipops_per_boy : ℕ) :
  total_candies = 90 →
  total_children = 40 →
  lollipops_per_boy = 3 →
  ∃ (num_boys num_girls : ℕ) (candy_canes_per_girl : ℕ),
    num_boys + num_girls = total_children ∧
    num_boys * lollipops_per_boy = total_candies / 3 ∧
    num_girls * candy_canes_per_girl = total_candies * 2 / 3 ∧
    candy_canes_per_girl = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2691_269165


namespace NUMINAMATH_CALUDE_percentage_problem_l2691_269131

theorem percentage_problem (x : ℝ) : (0.15 * 0.30 * 0.50 * x = 99) → x = 4400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2691_269131


namespace NUMINAMATH_CALUDE_modular_inverse_87_mod_88_l2691_269154

theorem modular_inverse_87_mod_88 : ∃ x : ℤ, 0 ≤ x ∧ x < 88 ∧ (87 * x) % 88 = 1 :=
by
  use 87
  sorry

end NUMINAMATH_CALUDE_modular_inverse_87_mod_88_l2691_269154


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2691_269119

-- Define the set A
def A : Set ℝ := {-1, 0, 2}

-- Define the set B as a function of a
def B (a : ℝ) : Set ℝ := {2^a}

-- Theorem statement
theorem subset_implies_a_equals_one (a : ℝ) (h : B a ⊆ A) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2691_269119


namespace NUMINAMATH_CALUDE_equation_solution_l2691_269161

theorem equation_solution (m n : ℚ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-2) = 6) → 
  m = 4.5 ∧ n = 1.5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2691_269161


namespace NUMINAMATH_CALUDE_linda_purchase_theorem_l2691_269155

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars2 : ℕ
  dollars4 : ℕ

/-- Calculates the total cost in cents given the item counts -/
def totalCost (items : ItemCounts) : ℕ :=
  50 * items.cents50 + 200 * items.dollars2 + 400 * items.dollars4

/-- Theorem stating that given the conditions, Linda bought 40 50-cent items -/
theorem linda_purchase_theorem (items : ItemCounts) : 
  (items.cents50 + items.dollars2 + items.dollars4 = 50) →
  (totalCost items = 5000) →
  (items.cents50 = 40) := by
  sorry

#eval totalCost { cents50 := 40, dollars2 := 4, dollars4 := 6 }

end NUMINAMATH_CALUDE_linda_purchase_theorem_l2691_269155


namespace NUMINAMATH_CALUDE_a_plus_b_eq_neg_one_l2691_269133

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {1, a, b}
def B (a : ℝ) : Set ℝ := {a, a^2, a*a}

-- State the theorem
theorem a_plus_b_eq_neg_one (a b : ℝ) : A a b = B a → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_eq_neg_one_l2691_269133


namespace NUMINAMATH_CALUDE_min_value_of_y_l2691_269115

theorem min_value_of_y (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  let y := |x - a| + |x - 15| + |x - (a + 15)|
  ∃ (min_y : ℝ), min_y = 15 ∧ ∀ z, a ≤ z ∧ z ≤ 15 → y ≤ |z - a| + |z - 15| + |z - (a + 15)| :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l2691_269115


namespace NUMINAMATH_CALUDE_fraction_simplification_l2691_269121

-- Define the statement
theorem fraction_simplification : (36 ^ 40) / (72 ^ 20) = 18 ^ 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2691_269121


namespace NUMINAMATH_CALUDE_average_other_color_marbles_l2691_269101

/-- Given a collection of marbles where 40% are clear, 20% are black, and the remainder are other colors,
    prove that when taking 5 marbles, the average number of marbles of other colors is 2. -/
theorem average_other_color_marbles
  (total : ℕ) -- Total number of marbles
  (clear : ℕ) -- Number of clear marbles
  (black : ℕ) -- Number of black marbles
  (other : ℕ) -- Number of other color marbles
  (h1 : clear = (40 * total) / 100) -- 40% are clear
  (h2 : black = (20 * total) / 100) -- 20% are black
  (h3 : other = total - clear - black) -- Remainder are other colors
  : (40 : ℚ) / 100 * 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_other_color_marbles_l2691_269101


namespace NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_f_leq_abs_3m_minus_2_l2691_269146

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 3|

-- Theorem for part (Ⅰ)
theorem solution_set_f_gt_7_minus_x :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m_for_f_leq_abs_3m_minus_2 :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_f_leq_abs_3m_minus_2_l2691_269146


namespace NUMINAMATH_CALUDE_negation_equivalence_l2691_269138

theorem negation_equivalence (x : ℝ) :
  ¬(2 < x ∧ x < 5 → x^2 - 7*x + 10 < 0) ↔
  (x ≤ 2 ∨ x ≥ 5 → x^2 - 7*x + 10 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2691_269138


namespace NUMINAMATH_CALUDE_discriminant_less_than_negative_one_l2691_269182

/-- A quadratic function that doesn't intersect with y = x and y = -x -/
structure NonIntersectingQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : ∀ x : ℝ, a * x^2 + b * x + c ≠ x
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ -x

/-- The discriminant of a quadratic function is less than -1 -/
theorem discriminant_less_than_negative_one (f : NonIntersectingQuadratic) :
  |f.b^2 - 4 * f.a * f.c| > 1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_less_than_negative_one_l2691_269182


namespace NUMINAMATH_CALUDE_series_convergence_l2691_269140

theorem series_convergence 
  (u v : ℕ → ℝ) 
  (hu : Summable (fun i => (u i)^2))
  (hv : Summable (fun i => (v i)^2))
  (p : ℕ) 
  (hp : p ≥ 2) : 
  Summable (fun i => (u i - v i)^p) :=
by
  sorry

end NUMINAMATH_CALUDE_series_convergence_l2691_269140


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2691_269113

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2691_269113


namespace NUMINAMATH_CALUDE_four_thirds_of_number_is_36_l2691_269178

theorem four_thirds_of_number_is_36 (x : ℚ) : (4 : ℚ) / 3 * x = 36 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_number_is_36_l2691_269178


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2691_269163

def expression (n : ℕ) : ℤ :=
  9 * (n - 3)^7 - 2 * n^3 + 15 * n - 33

theorem largest_n_divisible_by_seven :
  ∃ (n : ℕ), n = 149998 ∧
  n < 150000 ∧
  expression n % 7 = 0 ∧
  ∀ (m : ℕ), m < 150000 → m > n → expression m % 7 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2691_269163


namespace NUMINAMATH_CALUDE_no_two_distinct_roots_ellipse_slope_product_constant_l2691_269151

-- Statement for ①
theorem no_two_distinct_roots (f : ℝ → ℝ) (h : Monotone f) :
  ¬∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ f x + k = 0 ∧ f y + k = 0 :=
sorry

-- Statement for ④
theorem ellipse_slope_product_constant (a b : ℝ) (h : a > b) (h' : b > 0) :
  ∃ c : ℝ, ∀ m n : ℝ, 
    (b^2 * m^2 + a^2 * n^2 = a^2 * b^2) →
    (n / (m + a)) * (n / (m - a)) = c :=
sorry

end NUMINAMATH_CALUDE_no_two_distinct_roots_ellipse_slope_product_constant_l2691_269151


namespace NUMINAMATH_CALUDE_linear_equation_implies_mn_zero_l2691_269160

/-- If x^(m+n) + 5y^(m-n+2) = 8 is a linear equation in x and y, then mn = 0 -/
theorem linear_equation_implies_mn_zero (m n : ℤ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, x^(m+n) + 5*y^(m-n+2) = a*x + b*y + c) → 
  m * n = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_implies_mn_zero_l2691_269160


namespace NUMINAMATH_CALUDE_difference_of_squares_l2691_269153

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2691_269153


namespace NUMINAMATH_CALUDE_prob_double_is_one_seventh_l2691_269152

/-- Represents a domino set with integers from 0 to 12 -/
def DominoSet : Type := Unit

/-- The number of integers in the domino set -/
def num_integers : ℕ := 13

/-- The total number of domino tiles in the set -/
def total_tiles (ds : DominoSet) : ℕ := (num_integers * (num_integers + 1)) / 2

/-- The number of double tiles in the set -/
def num_doubles (ds : DominoSet) : ℕ := num_integers

/-- The probability of randomly selecting a double from the domino set -/
def prob_double (ds : DominoSet) : ℚ := (num_doubles ds : ℚ) / (total_tiles ds : ℚ)

theorem prob_double_is_one_seventh (ds : DominoSet) :
  prob_double ds = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_double_is_one_seventh_l2691_269152


namespace NUMINAMATH_CALUDE_marco_trading_cards_l2691_269125

theorem marco_trading_cards (x : ℚ) : 
  (2 / 15 : ℚ) * x = 850 → x = 6375 := by
  sorry

end NUMINAMATH_CALUDE_marco_trading_cards_l2691_269125


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2691_269188

/-- Given two 2D vectors a and b, where a = (x-1, 2) and b = (2, 1),
    if a is perpendicular to b, then x = 0. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2691_269188


namespace NUMINAMATH_CALUDE_three_couples_arrangement_l2691_269134

/-- The number of arrangements for three couples standing in a row -/
def couple_arrangements : ℕ := 48

/-- The number of ways to arrange three distinct units in a row -/
def unit_arrangements : ℕ := 6

/-- The number of internal arrangements for each couple -/
def internal_arrangements : ℕ := 2

/-- Theorem: The number of different arrangements for three couples standing in a row,
    where each couple must stand next to each other, is equal to 48. -/
theorem three_couples_arrangement :
  couple_arrangements = unit_arrangements * internal_arrangements^3 :=
by sorry

end NUMINAMATH_CALUDE_three_couples_arrangement_l2691_269134


namespace NUMINAMATH_CALUDE_lcm_18_30_l2691_269104

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l2691_269104


namespace NUMINAMATH_CALUDE_simplify_expression_l2691_269148

theorem simplify_expression (w x : ℝ) : 
  3*w + 5*w + 7*w + 9*w + 11*w + 13*x + 15 = 35*w + 13*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2691_269148


namespace NUMINAMATH_CALUDE_complement_of_A_l2691_269141

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}

theorem complement_of_A : (I \ A) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2691_269141


namespace NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l2691_269106

/-- Given an angle α = 7π/5, prove that its terminal side is located in the third quadrant. -/
theorem terminal_side_in_third_quadrant (α : Real) (h : α = 7 * Real.pi / 5) :
  ∃ (x y : Real), x < 0 ∧ y < 0 ∧ (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l2691_269106


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l2691_269135

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) :
  1 + n * x ≤ (1 + x)^n := by sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l2691_269135


namespace NUMINAMATH_CALUDE_regular_survey_rate_l2691_269189

/-- Proves that the regular rate for completing a survey is 10 given the specified conditions. -/
theorem regular_survey_rate (total_surveys : ℕ) (cellphone_surveys : ℕ) (total_earnings : ℚ) :
  total_surveys = 100 →
  cellphone_surveys = 60 →
  total_earnings = 1180 →
  ∃ (regular_rate : ℚ),
    regular_rate * (total_surveys - cellphone_surveys) +
    (regular_rate * 1.3) * cellphone_surveys = total_earnings ∧
    regular_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_survey_rate_l2691_269189


namespace NUMINAMATH_CALUDE_sum_of_seven_angles_l2691_269112

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 angle7 angle8 angle9 angle10 : ℝ)

-- State the theorem
theorem sum_of_seven_angles :
  (angle5 + angle6 + angle7 + angle8 = 360) →
  (angle2 + angle3 + angle4 + (180 - angle9) = 360) →
  (angle9 = angle10) →
  (angle8 = angle10 + angle1) →
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_seven_angles_l2691_269112


namespace NUMINAMATH_CALUDE_total_fishes_is_32_l2691_269129

/-- The total number of fishes caught by Melanie and Tom -/
def total_fishes (melanie_trout : ℕ) (tom_salmon_multiplier : ℕ) : ℕ :=
  melanie_trout + tom_salmon_multiplier * melanie_trout

/-- Proof that the total number of fishes caught is 32 -/
theorem total_fishes_is_32 : total_fishes 8 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_fishes_is_32_l2691_269129


namespace NUMINAMATH_CALUDE_line_through_points_l2691_269118

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) is
    (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

theorem line_through_points :
  ∀ x y : ℝ, line_equation 2 (-2) (-2) 6 x y ↔ 2 * x + y - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l2691_269118


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l2691_269109

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define a predicate for geometric sequence
def is_geometric (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ fib b = r * fib a ∧ fib c = r * fib b

theorem fibonacci_geometric_sequence :
  ∀ a b c : ℕ,
    is_geometric a b c →
    a + b + c = 3000 →
    a = 999 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l2691_269109


namespace NUMINAMATH_CALUDE_square_of_product_plus_one_l2691_269167

theorem square_of_product_plus_one :
  24 * 25 * 26 * 27 + 1 = (24^2 + 3 * 24 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_plus_one_l2691_269167
