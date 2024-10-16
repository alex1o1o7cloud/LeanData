import Mathlib

namespace NUMINAMATH_CALUDE_select_shoes_result_l2444_244444

/-- The number of ways to select 4 shoes from 5 pairs, with exactly one pair included -/
def select_shoes (n : ℕ) : ℕ :=
  let total_pairs := 5
  let pairs_to_choose := 1
  let single_shoes := n - 2 * pairs_to_choose
  let remaining_pairs := total_pairs - pairs_to_choose
  (total_pairs.choose pairs_to_choose) *
  (remaining_pairs.choose single_shoes) *
  (2^pairs_to_choose * 2^single_shoes)

theorem select_shoes_result : select_shoes 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_shoes_result_l2444_244444


namespace NUMINAMATH_CALUDE_exponent_square_negative_product_l2444_244463

theorem exponent_square_negative_product (a b : ℝ) : (-a^3 * b)^2 = a^6 * b^2 := by sorry

end NUMINAMATH_CALUDE_exponent_square_negative_product_l2444_244463


namespace NUMINAMATH_CALUDE_range_of_m_l2444_244429

def p (x : ℝ) : Prop := 12 / (x + 2) ≥ 1

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, ¬(p x) → ¬(q x m)) →
  (∃ x : ℝ, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2444_244429


namespace NUMINAMATH_CALUDE_greatest_power_of_two_congruence_l2444_244496

theorem greatest_power_of_two_congruence (m : ℕ) : 
  (∀ n : ℤ, Odd n → (n^2 * (1 + n^2 - n^4)) ≡ 1 [ZMOD 2^m]) ↔ m ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_congruence_l2444_244496


namespace NUMINAMATH_CALUDE_sum_difference_equality_l2444_244407

theorem sum_difference_equality : 291 + 503 - 91 + 492 - 103 - 392 = 700 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equality_l2444_244407


namespace NUMINAMATH_CALUDE_at_least_four_same_probability_l2444_244459

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling a specific value on a single fair die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability that all five dice show the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def fourSameProbability : ℚ :=
  (numDice : ℚ) * (singleDieProbability ^ (numDice - 2)) * ((numSides - 1 : ℚ) / numSides)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_probability :
  allSameProbability + fourSameProbability = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_at_least_four_same_probability_l2444_244459


namespace NUMINAMATH_CALUDE_ball_contact_height_l2444_244479

theorem ball_contact_height (horizontal_distance : ℝ) (hypotenuse : ℝ) (height : ℝ) : 
  horizontal_distance = 7 → hypotenuse = 53 → height ^ 2 + horizontal_distance ^ 2 = hypotenuse ^ 2 → height = 2 := by
  sorry

end NUMINAMATH_CALUDE_ball_contact_height_l2444_244479


namespace NUMINAMATH_CALUDE_range_of_f_l2444_244405

noncomputable def f (x : ℝ) : ℝ := 2 * (x + 7) * (x - 5) / (x + 7)

theorem range_of_f :
  Set.range f = {y | y < -24 ∨ y > -24} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2444_244405


namespace NUMINAMATH_CALUDE_sams_dimes_l2444_244423

/-- Given that Sam had 9 dimes initially and received 7 more dimes from his dad,
    prove that the total number of dimes Sam has now is 16. -/
theorem sams_dimes (initial_dimes : ℕ) (received_dimes : ℕ) (total_dimes : ℕ) : 
  initial_dimes = 9 → received_dimes = 7 → total_dimes = initial_dimes + received_dimes → total_dimes = 16 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_l2444_244423


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2444_244411

theorem absolute_value_expression : |(|-|-2 + 3| - 2| + 2)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2444_244411


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2444_244427

/-- A police emergency number is a positive integer that ends with 133 in decimal representation -/
def IsPoliceEmergencyNumber (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7 -/
theorem police_emergency_number_prime_divisor (n : ℕ) (h : IsPoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2444_244427


namespace NUMINAMATH_CALUDE_lower_interest_rate_l2444_244408

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem lower_interest_rate 
  (principal : ℚ) 
  (time : ℚ) 
  (higher_rate : ℚ) 
  (interest_difference : ℚ) : 
  principal = 5000 → 
  time = 2 → 
  higher_rate = 18 → 
  interest_difference = 600 → 
  ∃ (lower_rate : ℚ), 
    simpleInterest principal higher_rate time - simpleInterest principal lower_rate time = interest_difference ∧ 
    lower_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_lower_interest_rate_l2444_244408


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l2444_244453

/-- Given a cone whose lateral surface development has a central angle of α radians,
    the vertex angle of its axial section is equal to 2 * arcsin(α / (2π)). -/
theorem cone_vertex_angle (α : ℝ) (h : 0 < α ∧ α < 2 * Real.pi) :
  let vertex_angle := 2 * Real.arcsin (α / (2 * Real.pi))
  vertex_angle = 2 * Real.arcsin (α / (2 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l2444_244453


namespace NUMINAMATH_CALUDE_initial_quarters_count_l2444_244422

-- Define the problem parameters
def cents_left : ℕ := 300
def cents_spent : ℕ := 50
def cents_per_quarter : ℕ := 25

-- Theorem statement
theorem initial_quarters_count : 
  (cents_left + cents_spent) / cents_per_quarter = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_quarters_count_l2444_244422


namespace NUMINAMATH_CALUDE_square_side_length_l2444_244495

/-- Given a rectangle with length 400 feet and width 300 feet, prove that a square with perimeter
    twice that of the rectangle has a side length of 700 feet. -/
theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ)
    (h1 : rectangle_length = 400)
    (h2 : rectangle_width = 300)
    (square_perimeter : ℝ)
    (h3 : square_perimeter = 2 * (2 * (rectangle_length + rectangle_width)))
    (square_side : ℝ)
    (h4 : square_perimeter = 4 * square_side) :
  square_side = 700 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l2444_244495


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l2444_244439

theorem no_solution_for_inequality :
  ¬∃ (x : ℝ), x > 0 ∧ x * Real.sqrt (10 - x) + Real.sqrt (10 * x - x^3) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l2444_244439


namespace NUMINAMATH_CALUDE_frame_width_is_five_l2444_244448

/-- A rectangular frame containing three square photograph openings with uniform width. -/
structure PhotoFrame where
  /-- The side length of each square opening -/
  opening_side : ℝ
  /-- The width of the frame -/
  frame_width : ℝ

/-- The perimeter of one square opening -/
def opening_perimeter (frame : PhotoFrame) : ℝ :=
  4 * frame.opening_side

/-- The perimeter of the entire rectangular frame -/
def frame_perimeter (frame : PhotoFrame) : ℝ :=
  2 * (frame.opening_side + 2 * frame.frame_width) + 2 * (3 * frame.opening_side + 2 * frame.frame_width)

/-- Theorem stating that if the perimeter of one opening is 60 cm and the perimeter of the entire frame is 180 cm, 
    then the width of the frame is 5 cm -/
theorem frame_width_is_five (frame : PhotoFrame) 
  (h1 : opening_perimeter frame = 60) 
  (h2 : frame_perimeter frame = 180) : 
  frame.frame_width = 5 := by
  sorry

end NUMINAMATH_CALUDE_frame_width_is_five_l2444_244448


namespace NUMINAMATH_CALUDE_min_value_f_range_of_t_l2444_244421

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + |x - 4|

-- Theorem for the minimum value of f(x)
theorem min_value_f : ∀ x : ℝ, f x ≥ 6 := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t) ↔ (t ≤ -2 ∨ t ≥ 3) := by sorry

end NUMINAMATH_CALUDE_min_value_f_range_of_t_l2444_244421


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2444_244441

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 23 * n ≡ 789 [ZMOD 11] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(23 * m ≡ 789 [ZMOD 11])) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2444_244441


namespace NUMINAMATH_CALUDE_red_chips_probability_l2444_244446

def total_chips : ℕ := 6
def red_chips : ℕ := 3
def green_chips : ℕ := 3

def favorable_arrangements : ℕ := Nat.choose 5 2
def total_arrangements : ℕ := Nat.choose total_chips red_chips

theorem red_chips_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_red_chips_probability_l2444_244446


namespace NUMINAMATH_CALUDE_minimal_extensive_h_21_l2444_244471

/-- An extensive function is a function from positive integers to integers
    such that f(x) + f(y) ≥ x² + y² for all positive integers x and y. -/
def Extensive (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y ≥ (x.val ^ 2 : ℤ) + (y.val ^ 2 : ℤ)

/-- The sum of the first 30 values of an extensive function -/
def SumFirst30 (f : ℕ+ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => f ⟨i + 1, Nat.succ_pos i⟩)

/-- An extensive function with minimal sum of first 30 values -/
def MinimalExtensive (h : ℕ+ → ℤ) : Prop :=
  Extensive h ∧ ∀ g : ℕ+ → ℤ, Extensive g → SumFirst30 h ≤ SumFirst30 g

theorem minimal_extensive_h_21 (h : ℕ+ → ℤ) (hmin : MinimalExtensive h) :
    h ⟨21, by norm_num⟩ ≥ 301 := by
  sorry

end NUMINAMATH_CALUDE_minimal_extensive_h_21_l2444_244471


namespace NUMINAMATH_CALUDE_expand_expression_l2444_244478

theorem expand_expression (x : ℝ) : (17 * x - 12) * (3 * x) = 51 * x^2 - 36 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2444_244478


namespace NUMINAMATH_CALUDE_inequality_proof_l2444_244480

theorem inequality_proof (n : ℕ) : 
  2 * Real.sqrt (n + 1 : ℝ) - 2 * Real.sqrt (n : ℝ) < 1 / Real.sqrt (n : ℝ) ∧ 
  1 / Real.sqrt (n : ℝ) < 2 * Real.sqrt (n : ℝ) - 2 * Real.sqrt ((n - 1) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2444_244480


namespace NUMINAMATH_CALUDE_tangent_line_intercept_product_minimum_l2444_244445

/-- The minimum product of x and y intercepts of a tangent line to the unit circle -/
theorem tangent_line_intercept_product_minimum : ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), x^2 + y^2 = 1 → (x / a + y / b = 1 → False) ∨ (x / a + y / b ≠ 1)) →
  a * b ≥ 2 ∧ (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (∀ (x y : ℝ), x^2 + y^2 = 1 → (x / a₀ + y / b₀ = 1 → False) ∨ (x / a₀ + y / b₀ ≠ 1)) ∧
    a₀ * b₀ = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_product_minimum_l2444_244445


namespace NUMINAMATH_CALUDE_min_value_of_function_l2444_244428

theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  (4 / (x - 2) + x) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2444_244428


namespace NUMINAMATH_CALUDE_even_function_quadratic_behavior_l2444_244435

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- State the theorem
theorem even_function_quadratic_behavior :
  ∀ m : ℝ, (∀ x : ℝ, f m x = f m (-x)) →
  (m = 0 ∧
   ∃ c : ℝ, c ∈ (Set.Ioo (-4) 2) ∧
   (∀ x y : ℝ, x ∈ (Set.Ioo (-4) c) → y ∈ (Set.Ioo (-4) c) → x < y → f m x < f m y) ∧
   (∀ x y : ℝ, x ∈ (Set.Ioo c 2) → y ∈ (Set.Ioo c 2) → x < y → f m x > f m y)) :=
by sorry

end NUMINAMATH_CALUDE_even_function_quadratic_behavior_l2444_244435


namespace NUMINAMATH_CALUDE_randy_picture_count_randy_drew_five_pictures_l2444_244450

theorem randy_picture_count : ℕ → ℕ → ℕ → Prop :=
  fun randy peter quincy =>
    (peter = randy + 3) →
    (quincy = peter + 20) →
    (randy + peter + quincy = 41) →
    (randy = 5)

-- The proof of the theorem
theorem randy_drew_five_pictures : ∃ (randy peter quincy : ℕ), randy_picture_count randy peter quincy :=
  sorry

end NUMINAMATH_CALUDE_randy_picture_count_randy_drew_five_pictures_l2444_244450


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2444_244420

def is_valid_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def satisfies_bounds (f : ℝ → ℝ) : Prop :=
  ∀ x, |x| ≤ 1 → |f x| ≤ 1

def derivative_max (f : ℝ → ℝ) (K : ℝ) : Prop :=
  ∀ x, |x| ≤ 1 → |(deriv f) x| ≤ K

def max_attained (f : ℝ → ℝ) (K : ℝ) : Prop :=
  ∃ x₀, x₀ ∈ Set.Icc (-1) 1 ∧ |(deriv f) x₀| = K

theorem quadratic_function_theorem (f : ℝ → ℝ) (K : ℝ) :
  is_valid_quadratic f →
  satisfies_bounds f →
  derivative_max f K →
  max_attained f K →
  (∃ (ε : ℝ), ε = 1 ∨ ε = -1) ∧ (∀ x, f x = ε * (2 * x^2 - 1)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2444_244420


namespace NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l2444_244472

theorem sqrt_twelve_equals_two_sqrt_three :
  Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_twelve_equals_two_sqrt_three_l2444_244472


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2444_244412

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2444_244412


namespace NUMINAMATH_CALUDE_delores_money_theorem_l2444_244466

/-- The amount of money Delores had at first, given her purchases and remaining money. -/
def delores_initial_money (computer_price printer_price headphones_price discount_rate remaining_money : ℚ) : ℚ :=
  let discounted_computer_price := computer_price * (1 - discount_rate)
  let total_spent := discounted_computer_price + printer_price + headphones_price
  total_spent + remaining_money

/-- Theorem stating that Delores had $470 at first. -/
theorem delores_money_theorem :
  delores_initial_money 400 40 60 (1/10) 10 = 470 := by
  sorry

end NUMINAMATH_CALUDE_delores_money_theorem_l2444_244466


namespace NUMINAMATH_CALUDE_log_inequality_l2444_244415

theorem log_inequality (x : ℝ) (h : 0 < x) (h' : x < 1) : Real.log (1 + x) > x^3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2444_244415


namespace NUMINAMATH_CALUDE_linear_function_intersection_l2444_244440

/-- The linear function that intersects with two given lines -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_intersection : 
  ∃ k b : ℝ, 
    (linear_function k b 4 = -4 + 6) ∧ 
    (linear_function k b (1 + 1) = 1) ∧
    (∀ x : ℝ, linear_function k b x = (1/2) * x) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l2444_244440


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l2444_244401

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l2444_244401


namespace NUMINAMATH_CALUDE_goods_train_length_l2444_244436

/-- The length of a goods train passing a man in an opposite moving train -/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) (passing_time : ℝ) : 
  man_train_speed = 20 →
  goods_train_speed = 92 →
  passing_time = 9 →
  ∃ (length : ℝ), abs (length - 279.99) < 0.01 ∧ 
    length = (man_train_speed + goods_train_speed) * (5/18) * passing_time :=
by
  sorry

#check goods_train_length

end NUMINAMATH_CALUDE_goods_train_length_l2444_244436


namespace NUMINAMATH_CALUDE_fraction_equality_l2444_244437

theorem fraction_equality : 
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2444_244437


namespace NUMINAMATH_CALUDE_break_even_books_l2444_244489

/-- Represents the fixed cost of making books -/
def fixed_cost : ℝ := 50000

/-- Represents the marketing cost per book -/
def marketing_cost_per_book : ℝ := 4

/-- Represents the selling price per book -/
def selling_price_per_book : ℝ := 9

/-- Calculates the total cost for a given number of books -/
def total_cost (num_books : ℝ) : ℝ :=
  fixed_cost + marketing_cost_per_book * num_books

/-- Calculates the revenue for a given number of books -/
def revenue (num_books : ℝ) : ℝ :=
  selling_price_per_book * num_books

/-- Theorem: The number of books needed to break even is 10000 -/
theorem break_even_books : 
  ∃ (x : ℝ), x = 10000 ∧ total_cost x = revenue x :=
by sorry

end NUMINAMATH_CALUDE_break_even_books_l2444_244489


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2444_244424

/-- For the quadratic equation x^2 + 4x√2 + k = 0, prove that k = 8 makes the discriminant zero and the roots real and equal. -/
theorem quadratic_equal_roots (k : ℝ) : 
  (∀ x, x^2 + 4*x*Real.sqrt 2 + k = 0) →
  (k = 8 ↔ (∃! r, r^2 + 4*r*Real.sqrt 2 + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2444_244424


namespace NUMINAMATH_CALUDE_square_nonnegative_l2444_244469

theorem square_nonnegative (x : ℝ) : x ^ 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_nonnegative_l2444_244469


namespace NUMINAMATH_CALUDE_complex_circle_l2444_244494

-- Define a complex number
def z : ℂ := sorry

-- Define the condition |z - (-1 + i)| = 4
def condition (z : ℂ) : Prop := Complex.abs (z - (-1 + Complex.I)) = 4

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 16

-- Theorem statement
theorem complex_circle (z : ℂ) (h : condition z) :
  circle_equation z.re z.im := by sorry

end NUMINAMATH_CALUDE_complex_circle_l2444_244494


namespace NUMINAMATH_CALUDE_transformation_identity_l2444_244454

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotation 180° about y-axis -/
def rotateY180 (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- Reflection through yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Rotation 90° about z-axis -/
def rotateZ90 (p : Point3D) : Point3D :=
  { x := p.y, y := -p.x, z := p.z }

/-- Reflection through xz-plane -/
def reflectXZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- Reflection through xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- The sequence of transformations -/
def transformSequence (p : Point3D) : Point3D :=
  reflectXY (reflectXZ (rotateZ90 (reflectYZ (rotateY180 p))))

theorem transformation_identity :
  transformSequence { x := 2, y := 2, z := 2 } = { x := 2, y := 2, z := 2 } := by
  sorry

end NUMINAMATH_CALUDE_transformation_identity_l2444_244454


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l2444_244402

theorem quadratic_equation_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) → k = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l2444_244402


namespace NUMINAMATH_CALUDE_carmen_pets_difference_l2444_244443

def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_given_up : ℕ := 3

theorem carmen_pets_difference :
  initial_cats - cats_given_up - initial_dogs = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_carmen_pets_difference_l2444_244443


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2444_244456

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | x = 1/2 ∨ x = -3}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y : Set ℝ :=
  {y | y = 3/4 ∨ y = 20}

/-- First parabola function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- Second parabola function -/
def g (x : ℝ) : ℝ := 3*x^2 + 2*x - 1

theorem parabolas_intersection :
  ∀ x y : ℝ, (f x = g x ∧ y = f x) ↔ (x ∈ intersection_x ∧ y ∈ intersection_y) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2444_244456


namespace NUMINAMATH_CALUDE_factorization_coefficient_sum_l2444_244498

theorem factorization_coefficient_sum : 
  ∃ (a b c d e f g h i j k l m n o p : ℤ),
  (∀ x y : ℝ, 
    81 * x^8 - 256 * y^8 = 
    (a*x + b*y) * 
    (c*x^2 + d*x*y + e*y^2) * 
    (f*x^3 + g*x*y^2 + h*y^3) * 
    (i*x + j*y) * 
    (k*x^2 + l*x*y + m*y^2) * 
    (n*x^3 + o*x*y^2 + p*y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
sorry

end NUMINAMATH_CALUDE_factorization_coefficient_sum_l2444_244498


namespace NUMINAMATH_CALUDE_palmer_photos_l2444_244425

theorem palmer_photos (initial_photos : ℕ) (final_photos : ℕ) (third_fourth_week_photos : ℕ) :
  initial_photos = 100 →
  final_photos = 380 →
  third_fourth_week_photos = 80 →
  ∃ (first_week_photos : ℕ),
    first_week_photos = 67 ∧
    final_photos = initial_photos + first_week_photos + 2 * first_week_photos + third_fourth_week_photos :=
by sorry

end NUMINAMATH_CALUDE_palmer_photos_l2444_244425


namespace NUMINAMATH_CALUDE_sum_of_specific_primes_l2444_244431

theorem sum_of_specific_primes : ∃ (S : Finset Nat),
  (∀ p ∈ S, p.Prime ∧ 1 < p ∧ p ≤ 100 ∧ p % 6 = 1 ∧ p % 7 = 6) ∧
  (∀ p, p.Prime → 1 < p → p ≤ 100 → p % 6 = 1 → p % 7 = 6 → p ∈ S) ∧
  S.sum id = 104 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_primes_l2444_244431


namespace NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l2444_244462

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*y - 4 = -y^2 + 12*x - 12

-- Define the center and radius of the circle
def circle_center_radius (a' b' r' : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a')^2 + (y - b')^2 = r'^2

-- Theorem statement
theorem circle_sum_of_center_and_radius :
  ∃ a' b' r', circle_center_radius a' b' r' ∧ a' + b' + r' = 3 + Real.sqrt 37 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_of_center_and_radius_l2444_244462


namespace NUMINAMATH_CALUDE_gcd_91_49_l2444_244452

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_49_l2444_244452


namespace NUMINAMATH_CALUDE_two_number_problem_l2444_244470

theorem two_number_problem :
  ∃ (x y : ℝ), 38 + 2 * x = 124 ∧ x + 3 * y = 47 ∧ x = 43 ∧ y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l2444_244470


namespace NUMINAMATH_CALUDE_son_age_l2444_244442

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2444_244442


namespace NUMINAMATH_CALUDE_equation_solution_l2444_244473

theorem equation_solution : ∃! x : ℚ, (x - 60) / 3 = (4 - 3 * x) / 6 :=
  by
    use 124 / 5
    sorry

end NUMINAMATH_CALUDE_equation_solution_l2444_244473


namespace NUMINAMATH_CALUDE_second_number_equation_l2444_244474

/-- Given three real numbers x, y, and z satisfying the equation 3x + 3y + 3z + 11 = 143,
    prove that y = 44 - x - z. -/
theorem second_number_equation (x y z : ℝ) (h : 3*x + 3*y + 3*z + 11 = 143) :
  y = 44 - x - z := by
  sorry

end NUMINAMATH_CALUDE_second_number_equation_l2444_244474


namespace NUMINAMATH_CALUDE_complex_square_real_implies_zero_l2444_244449

theorem complex_square_real_implies_zero (x : ℝ) :
  (Complex.I + x)^2 ∈ Set.range Complex.ofReal → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_real_implies_zero_l2444_244449


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2444_244406

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 600) : 
  (0.8 * L) * (1.15 * W) = 552 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2444_244406


namespace NUMINAMATH_CALUDE_coin_toss_problem_l2444_244493

theorem coin_toss_problem (p : ℝ) (n : ℕ) : 
  p = 1 / 2 →  -- Condition 1: Fair coin
  (1 / 2 : ℝ) ^ n = (1 / 8 : ℝ) →  -- Condition 2: Probability of same side is 0.125 (1/8)
  n = 3 := by  -- Question: Prove n = 3
sorry

end NUMINAMATH_CALUDE_coin_toss_problem_l2444_244493


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l2444_244417

theorem quilt_shaded_fraction (total_squares : ℕ) (divided_squares : ℕ) 
  (h1 : total_squares = 16) 
  (h2 : divided_squares = 8) : 
  (divided_squares : ℚ) / (2 : ℚ) / total_squares = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l2444_244417


namespace NUMINAMATH_CALUDE_fraction_value_l2444_244416

theorem fraction_value (x y : ℚ) (hx : x = 7/9) (hy : y = 3/5) :
  (7*x + 5*y) / (63*x*y) = 20/69 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l2444_244416


namespace NUMINAMATH_CALUDE_sequence_value_l2444_244475

theorem sequence_value (n : ℕ) (a : ℕ → ℕ) : 
  (∀ k, a k = 3 * k + 4) → a n = 13 → n = 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_value_l2444_244475


namespace NUMINAMATH_CALUDE_average_reduction_percentage_option1_more_favorable_l2444_244432

-- Define the original and final prices
def original_price : ℝ := 5
def final_price : ℝ := 3.2

-- Define the quantity to purchase in kilograms
def quantity : ℝ := 5000

-- Define the discount percentage and cash discount
def discount_percentage : ℝ := 0.1
def cash_discount_per_ton : ℝ := 200

-- Theorem for the average percentage reduction
theorem average_reduction_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ original_price * (1 - x)^2 = final_price ∧ x = 0.2 :=
sorry

-- Theorem for the more favorable option
theorem option1_more_favorable :
  final_price * (1 - discount_percentage) * quantity <
  final_price * quantity - (cash_discount_per_ton * (quantity / 1000)) :=
sorry

end NUMINAMATH_CALUDE_average_reduction_percentage_option1_more_favorable_l2444_244432


namespace NUMINAMATH_CALUDE_cyclic_matrix_determinant_zero_l2444_244418

theorem cyclic_matrix_determinant_zero (p q r : ℝ) (a b c d : ℝ) : 
  (a^4 + p*a^2 + q*a + r = 0) → 
  (b^4 + p*b^2 + q*b + r = 0) → 
  (c^4 + p*c^2 + q*c + r = 0) → 
  (d^4 + p*d^2 + q*d + r = 0) → 
  Matrix.det 
    ![![a, b, c, d],
      ![b, c, d, a],
      ![c, d, a, b],
      ![d, a, b, c]] = 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_matrix_determinant_zero_l2444_244418


namespace NUMINAMATH_CALUDE_find_a_and_b_l2444_244486

-- Define the system of inequalities
def inequality_system (a b x : ℝ) : Prop :=
  (3 * x - 2 < a + 1) ∧ (6 - 2 * x < b + 2)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

-- Theorem statement
theorem find_a_and_b :
  ∀ a b : ℝ,
  (∀ x : ℝ, inequality_system a b x ↔ solution_set x) →
  a = 3 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2444_244486


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l2444_244468

/-- Set A defined by the equation x^2 + mx - y + 2 = 0 -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + m * p.1 - p.2 + 2 = 0}

/-- Set B defined by the equation x - y + 1 = 0 with 0 ≤ x ≤ 2 -/
def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0 ∧ 0 ≤ p.1 ∧ p.1 ≤ 2}

/-- The main theorem stating that A ∩ B is nonempty if and only if m ≤ -1 -/
theorem intersection_nonempty_iff_m_leq_neg_one (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_m_leq_neg_one_l2444_244468


namespace NUMINAMATH_CALUDE_quadratic_root_l2444_244460

/-- Given a quadratic equation 2x^2 + 3x - k = 0 where k = 44, 
    prove that 4 is one of its roots. -/
theorem quadratic_root : ∃ x : ℝ, 2 * x^2 + 3 * x - 44 = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l2444_244460


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l2444_244483

theorem candy_box_price_increase : 
  ∀ (original_candy_price original_soda_price : ℝ),
  original_candy_price + original_soda_price = 20 →
  original_soda_price * 1.5 = 6 →
  original_candy_price * 1.25 = 20 →
  (20 - original_candy_price) / original_candy_price = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l2444_244483


namespace NUMINAMATH_CALUDE_nosuch_junction_population_l2444_244455

theorem nosuch_junction_population : ∃ (a b c : ℕ+), 
  (a.val^2 + 100 = b.val^2 + 1) ∧ 
  (b.val^2 + 101 = c.val^2) ∧ 
  (7 ∣ a.val^2) := by
  sorry

end NUMINAMATH_CALUDE_nosuch_junction_population_l2444_244455


namespace NUMINAMATH_CALUDE_median_in_60_64_interval_l2444_244426

/-- Represents the score intervals --/
inductive ScoreInterval
| interval_45_49
| interval_50_54
| interval_55_59
| interval_60_64
| interval_65_69

/-- Represents the frequency of each score interval --/
def frequency : ScoreInterval → Nat
| ScoreInterval.interval_45_49 => 10
| ScoreInterval.interval_50_54 => 15
| ScoreInterval.interval_55_59 => 20
| ScoreInterval.interval_60_64 => 25
| ScoreInterval.interval_65_69 => 30

/-- The total number of students --/
def totalStudents : Nat := 100

/-- Function to calculate the cumulative frequency up to a given interval --/
def cumulativeFrequency (interval : ScoreInterval) : Nat :=
  match interval with
  | ScoreInterval.interval_45_49 => frequency ScoreInterval.interval_45_49
  | ScoreInterval.interval_50_54 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54
  | ScoreInterval.interval_55_59 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54 + frequency ScoreInterval.interval_55_59
  | ScoreInterval.interval_60_64 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54 + frequency ScoreInterval.interval_55_59 + frequency ScoreInterval.interval_60_64
  | ScoreInterval.interval_65_69 => totalStudents

/-- The median position --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score falls in the interval 60-64 --/
theorem median_in_60_64_interval :
  cumulativeFrequency ScoreInterval.interval_55_59 < medianPosition ∧
  medianPosition ≤ cumulativeFrequency ScoreInterval.interval_60_64 :=
sorry

end NUMINAMATH_CALUDE_median_in_60_64_interval_l2444_244426


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2444_244467

theorem solve_linear_equation (x : ℝ) (h : x + 1 = 4) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2444_244467


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2444_244413

theorem cubic_equation_roots (x : ℝ) : ∃ (a b : ℝ),
  (x^3 - x^2 - 2*x + 1 = 0) ∧ (a^3 - a^2 - 2*a + 1 = 0) ∧ (b^3 - b^2 - 2*b + 1 = 0) ∧ (a - a*b = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2444_244413


namespace NUMINAMATH_CALUDE_yanna_payment_l2444_244461

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def num_sandals : ℕ := 3
def change : ℕ := 41

theorem yanna_payment :
  shirt_price * num_shirts + sandal_price * num_sandals + change = 100 := by
  sorry

end NUMINAMATH_CALUDE_yanna_payment_l2444_244461


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l2444_244488

theorem polygon_interior_exterior_angles_equality (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l2444_244488


namespace NUMINAMATH_CALUDE_ratio_problem_l2444_244491

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : (a + b) / (b + c) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2444_244491


namespace NUMINAMATH_CALUDE_intersection_M_N_l2444_244485

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2444_244485


namespace NUMINAMATH_CALUDE_dolls_count_l2444_244434

/-- The number of dolls Jane has -/
def jane_dolls : ℕ := 13

/-- The difference between Jill's and Jane's dolls -/
def doll_difference : ℕ := 6

/-- The total number of dolls Jane and Jill have together -/
def total_dolls : ℕ := jane_dolls + (jane_dolls + doll_difference)

theorem dolls_count : total_dolls = 32 := by sorry

end NUMINAMATH_CALUDE_dolls_count_l2444_244434


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2444_244419

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2444_244419


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2444_244447

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2444_244447


namespace NUMINAMATH_CALUDE_clown_count_l2444_244404

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 5

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 140 := by
  sorry

end NUMINAMATH_CALUDE_clown_count_l2444_244404


namespace NUMINAMATH_CALUDE_rook_configuration_exists_iff_even_l2444_244457

/-- A configuration of rooks on an n×n board. -/
def RookConfiguration (n : ℕ) := Fin n → Fin n

/-- Predicate to check if a rook configuration is valid (no two rooks attack each other). -/
def is_valid_configuration (n : ℕ) (config : RookConfiguration n) : Prop :=
  ∀ i j : Fin n, i ≠ j → config i ≠ config j ∧ i ≠ config j

/-- Predicate to check if two positions on the board are adjacent. -/
def are_adjacent (n : ℕ) (p1 p2 : Fin n × Fin n) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Theorem stating that a valid rook configuration with a valid move exists if and only if n is even. -/
theorem rook_configuration_exists_iff_even (n : ℕ) (h : n ≥ 2) :
  (∃ (initial final : RookConfiguration n),
    is_valid_configuration n initial ∧
    is_valid_configuration n final ∧
    (∀ i : Fin n, are_adjacent n (i, initial i) (i, final i))) ↔
  Even n :=
sorry

end NUMINAMATH_CALUDE_rook_configuration_exists_iff_even_l2444_244457


namespace NUMINAMATH_CALUDE_songbook_cost_is_seven_l2444_244403

/-- The cost of Jason's music purchases -/
structure MusicPurchase where
  flute : ℝ
  stand : ℝ
  total : ℝ

/-- The cost of the song book given Jason's other music purchases -/
def songbook_cost (p : MusicPurchase) : ℝ :=
  p.total - (p.flute + p.stand)

/-- Theorem: The cost of the song book is $7.00 -/
theorem songbook_cost_is_seven (p : MusicPurchase)
  (h1 : p.flute = 142.46)
  (h2 : p.stand = 8.89)
  (h3 : p.total = 158.35) :
  songbook_cost p = 7.00 := by
  sorry

#eval songbook_cost { flute := 142.46, stand := 8.89, total := 158.35 }

end NUMINAMATH_CALUDE_songbook_cost_is_seven_l2444_244403


namespace NUMINAMATH_CALUDE_seq_is_bounded_l2444_244497

-- Define P(n) as the product of all digits of n
def P (n : ℕ) : ℕ := sorry

-- Define the sequence (n_k)
def seq (k : ℕ) : ℕ → ℕ
  | n₁ => match k with
    | 0 => n₁
    | k + 1 => seq k n₁ + P (seq k n₁)

-- Theorem statement
theorem seq_is_bounded (n₁ : ℕ) : ∃ M : ℕ, ∀ k : ℕ, seq k n₁ ≤ M := by sorry

end NUMINAMATH_CALUDE_seq_is_bounded_l2444_244497


namespace NUMINAMATH_CALUDE_complex_product_real_l2444_244477

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 + a * I
  let z₂ : ℂ := a - 3 * I
  (z₁ * z₂).im = 0 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l2444_244477


namespace NUMINAMATH_CALUDE_toy_store_fraction_l2444_244458

-- Define John's weekly allowance
def weekly_allowance : ℚ := 4.80

-- Define the fraction spent at the arcade
def arcade_fraction : ℚ := 3/5

-- Define the amount spent at the candy store
def candy_store_spending : ℚ := 1.28

-- Theorem statement
theorem toy_store_fraction :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_spending := remaining_after_arcade - candy_store_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by
sorry

end NUMINAMATH_CALUDE_toy_store_fraction_l2444_244458


namespace NUMINAMATH_CALUDE_vector_at_t_4_l2444_244482

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfies the conditions -/
def given_line : ParameterizedLine :=
  { point := sorry }

theorem vector_at_t_4 (line : ParameterizedLine) 
  (h1 : line.point (-2) = (2, 6, 16)) 
  (h2 : line.point 1 = (-1, -5, -10)) :
  line.point 4 = (0, -4, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_4_l2444_244482


namespace NUMINAMATH_CALUDE_symmetric_points_range_l2444_244464

open Real

theorem symmetric_points_range (g h : ℝ → ℝ) (a : ℝ) :
  (∀ x, 1/ℯ ≤ x → x ≤ ℯ → g x = a - x^2) →
  (∀ x, h x = 2 * log x) →
  (∃ x, 1/ℯ ≤ x ∧ x ≤ ℯ ∧ g x = -h x) →
  1 ≤ a ∧ a ≤ ℯ^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l2444_244464


namespace NUMINAMATH_CALUDE_prairie_size_and_untouched_percentage_l2444_244433

/-- Represents the prairie and its natural events -/
structure Prairie where
  dust_storm1 : ℕ
  dust_storm2 : ℕ
  flood : ℕ
  wildfire : ℕ
  untouched : ℕ
  affected : ℕ

/-- The prairie with given conditions -/
def our_prairie : Prairie :=
  { dust_storm1 := 75000
  , dust_storm2 := 120000
  , flood := 30000
  , wildfire := 80000
  , untouched := 5000
  , affected := 290000
  }

/-- The theorem stating the total size and untouched percentage of the prairie -/
theorem prairie_size_and_untouched_percentage (p : Prairie) 
  (h : p = our_prairie) : 
  (p.affected + p.untouched = 295000) ∧ 
  (p.untouched : ℚ) / (p.affected + p.untouched : ℚ) * 100 = 5000 / 295000 * 100 := by
  sorry

#eval (our_prairie.untouched : ℚ) / (our_prairie.affected + our_prairie.untouched : ℚ) * 100

end NUMINAMATH_CALUDE_prairie_size_and_untouched_percentage_l2444_244433


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l2444_244481

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 21) (LCM 10 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l2444_244481


namespace NUMINAMATH_CALUDE_discount_difference_l2444_244476

def original_bill : ℝ := 15000

def single_discount_rate : ℝ := 0.3
def first_successive_discount_rate : ℝ := 0.25
def second_successive_discount_rate : ℝ := 0.06

def single_discount_amount : ℝ := original_bill * (1 - single_discount_rate)
def successive_discount_amount : ℝ := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)

theorem discount_difference :
  successive_discount_amount - single_discount_amount = 75 := by sorry

end NUMINAMATH_CALUDE_discount_difference_l2444_244476


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l2444_244410

/-- Given a sequence a_n with sum of first n terms S_n = n^2 + 1, prove it's arithmetic -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_def : ∀ n : ℕ, S n = n^2 + 1) 
  (sum_relation : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l2444_244410


namespace NUMINAMATH_CALUDE_local_min_iff_a_lt_one_l2444_244499

/-- The function f(x) defined as (x-1)^2 * (x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1)^2 * (x - a)

/-- x = 1 is a local minimum point of f(x) if and only if a < 1 -/
theorem local_min_iff_a_lt_one (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_local_min_iff_a_lt_one_l2444_244499


namespace NUMINAMATH_CALUDE_team_c_score_l2444_244400

/-- Given a trivia game with three teams, prove that Team C's score is 4 points. -/
theorem team_c_score (team_a team_b team_c total : ℕ) : 
  team_a = 2 → team_b = 9 → total = 15 → team_a + team_b + team_c = total → team_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_team_c_score_l2444_244400


namespace NUMINAMATH_CALUDE_midpoint_implies_xy_24_l2444_244484

-- Define the points
def A : ℝ × ℝ := (2, 10)
def C : ℝ × ℝ := (4, 7)

-- Define B as a function of x and y
def B (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

-- Theorem statement
theorem midpoint_implies_xy_24 (x y : ℝ) :
  is_midpoint C A (B x y) → x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_implies_xy_24_l2444_244484


namespace NUMINAMATH_CALUDE_jimmy_needs_four_packs_l2444_244487

/-- The number of packs of bread Jimmy needs to buy for his sandwiches -/
def breadPacksNeeded (sandwiches : ℕ) (slicesPerSandwich : ℕ) (slicesPerPack : ℕ) : ℕ :=
  (sandwiches * slicesPerSandwich + slicesPerPack - 1) / slicesPerPack

/-- Proof that Jimmy needs to buy 4 packs of bread -/
theorem jimmy_needs_four_packs :
  breadPacksNeeded 8 2 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_needs_four_packs_l2444_244487


namespace NUMINAMATH_CALUDE_point_N_coordinates_l2444_244492

def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)

theorem point_N_coordinates : 
  ∀ N : ℝ × ℝ, 
  (N.1 - M.1, N.2 - M.2) = (-3 * a.1, -3 * a.2) → 
  N = (2, 0) := by sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l2444_244492


namespace NUMINAMATH_CALUDE_negation_of_nonnegative_squares_l2444_244438

theorem negation_of_nonnegative_squares :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_nonnegative_squares_l2444_244438


namespace NUMINAMATH_CALUDE_bug_meeting_point_l2444_244490

/-- Triangle with sides a, b, c and point S on perimeter --/
structure Triangle (a b c : ℝ) where
  S : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : a + b > c ∧ b + c > a ∧ c + a > b
  h3 : 0 ≤ S ∧ S ≤ a + b + c

/-- The length of QS in the triangle --/
def qsLength (t : Triangle 7 8 9) : ℝ :=
  5

theorem bug_meeting_point (t : Triangle 7 8 9) : qsLength t = 5 := by
  sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l2444_244490


namespace NUMINAMATH_CALUDE_remainder_squared_pred_l2444_244465

theorem remainder_squared_pred (n : ℤ) (h : n % 5 = 3) : (n - 1)^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_pred_l2444_244465


namespace NUMINAMATH_CALUDE_car_distance_in_30_minutes_l2444_244409

theorem car_distance_in_30_minutes 
  (train_speed : ℝ) 
  (car_speed_ratio : ℝ) 
  (time : ℝ) 
  (h1 : train_speed = 90) 
  (h2 : car_speed_ratio = 2/3) 
  (h3 : time = 1/2) : 
  car_speed_ratio * train_speed * time = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_in_30_minutes_l2444_244409


namespace NUMINAMATH_CALUDE_book_difference_l2444_244451

def jungkook_initial : ℕ := 28
def seokjin_initial : ℕ := 28
def jungkook_bought : ℕ := 18
def seokjin_bought : ℕ := 11

theorem book_difference : 
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_difference_l2444_244451


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2444_244414

theorem complex_fraction_evaluation :
  2 + (3 / (2 + (1 / (2 + (1/2))))) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2444_244414


namespace NUMINAMATH_CALUDE_union_of_sets_l2444_244430

-- Define the sets A and B
def A (a : ℕ) : Set ℕ := {3, 2^a}
def B (a b : ℕ) : Set ℕ := {a, b}

-- Theorem statement
theorem union_of_sets (a b : ℕ) :
  (A a ∩ B a b = {2}) → (A a ∪ B a b = {1, 2, 3}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2444_244430
