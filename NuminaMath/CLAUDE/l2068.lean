import Mathlib

namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l2068_206878

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l2068_206878


namespace NUMINAMATH_CALUDE_sqrt_sum_abs_equal_six_l2068_206871

theorem sqrt_sum_abs_equal_six :
  Real.sqrt 2 + Real.sqrt 16 + |Real.sqrt 2 - 2| = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_abs_equal_six_l2068_206871


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2068_206846

theorem quadratic_coefficient (x : ℝ) : 
  (3 * x^2 = 8 * x + 10) → 
  ∃ a b c : ℝ, (a * x^2 + b * x + c = 0 ∧ b = -8) := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2068_206846


namespace NUMINAMATH_CALUDE_crease_length_l2068_206803

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  a_eq_5 : a = 5
  b_eq_12 : b = 12
  c_eq_13 : c = 13

/-- The length of the perpendicular bisector from the right angle to the hypotenuse -/
def perp_bisector_length (t : RightTriangle) : ℝ := t.b

theorem crease_length (t : RightTriangle) :
  perp_bisector_length t = 12 := by sorry

end NUMINAMATH_CALUDE_crease_length_l2068_206803


namespace NUMINAMATH_CALUDE_find_number_l2068_206837

theorem find_number : ∃ N : ℚ, (5/6 : ℚ) * N = (5/16 : ℚ) * N + 50 → N = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2068_206837


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2068_206887

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a →
  Real.cos B = 3 / 5 →
  A = π / 3 ∧ Real.sin (B - C) = (7 * Real.sqrt 3 - 24) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2068_206887


namespace NUMINAMATH_CALUDE_car_catch_up_time_l2068_206815

/-- The time it takes for a car to catch up to a truck on a highway -/
theorem car_catch_up_time (truck_speed car_speed : ℝ) (head_start : ℝ) : 
  truck_speed = 45 →
  car_speed = 60 →
  head_start = 1 →
  ∃ t : ℝ, t = 6 ∧ car_speed * t = truck_speed * (t + head_start) + truck_speed * head_start :=
by
  sorry


end NUMINAMATH_CALUDE_car_catch_up_time_l2068_206815


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2068_206812

theorem division_remainder_problem (D : ℕ) : 
  (D / 12 = 70) → (D / 21 = 40) → (D % 21 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2068_206812


namespace NUMINAMATH_CALUDE_guitar_ratio_l2068_206807

/-- The ratio of Davey's guitars to Barbeck's guitars is 1:1 -/
theorem guitar_ratio (davey barbeck : ℕ) : 
  davey = 18 → davey = barbeck → davey / barbeck = 1 := by
  sorry

end NUMINAMATH_CALUDE_guitar_ratio_l2068_206807


namespace NUMINAMATH_CALUDE_larger_number_problem_l2068_206801

theorem larger_number_problem (a b : ℝ) (h1 : a > b) (h2 : (a + b) + (a - b) = 68) : a = 34 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2068_206801


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l2068_206885

theorem sum_of_squares_bound (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l2068_206885


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2068_206810

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- The point P -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- A chord AB passing through F -/
def chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ∃ (t : ℝ), (1 - t) • A + t • B = F

/-- The angle equality condition -/
def angle_equality (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  let P := (p, 0)
  let AP := (A.1 - p, A.2)
  let BP := (B.1 - p, B.2)
  let PF := (2 - p, 0)
  AP.1 * PF.1 + AP.2 * PF.2 = BP.1 * PF.1 + BP.2 * PF.2

/-- The main theorem -/
theorem ellipse_focal_property :
  ∃! (p : ℝ), p > 0 ∧
    (∀ A B : ℝ × ℝ, chord A B → angle_equality p A B) ∧
    p = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2068_206810


namespace NUMINAMATH_CALUDE_lecture_schedules_count_l2068_206851

/-- Represents the number of lecturers --/
def num_lecturers : ℕ := 8

/-- Represents the number of lecturer pairs with order requirements --/
def num_ordered_pairs : ℕ := 2

/-- Calculates the number of valid lecture schedules --/
def num_valid_schedules : ℕ := (Nat.factorial num_lecturers) / (2^num_ordered_pairs)

/-- Theorem stating the number of valid lecture schedules --/
theorem lecture_schedules_count : num_valid_schedules = 10080 := by
  sorry

end NUMINAMATH_CALUDE_lecture_schedules_count_l2068_206851


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l2068_206820

/-- Two points symmetric about the y-axis in a Cartesian coordinate system -/
structure SymmetricPoints where
  m : ℝ
  n : ℝ
  symmetric : m + 4 = 0 ∧ n = 3

/-- The theorem stating that for symmetric points A(m,3) and B(4,n), (m+n)^2023 = -1 -/
theorem symmetric_points_sum_power (p : SymmetricPoints) : (p.m + p.n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l2068_206820


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2068_206865

theorem hemisphere_surface_area (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 225 * π) : 
  3 * π * r^2 + π * r^2 = 900 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2068_206865


namespace NUMINAMATH_CALUDE_total_combinations_eq_nine_l2068_206847

/-- The number of characters available to choose from. -/
def num_characters : ℕ := 3

/-- The number of cars available to choose from. -/
def num_cars : ℕ := 3

/-- The total number of possible combinations when choosing one character and one car. -/
def total_combinations : ℕ := num_characters * num_cars

/-- Theorem stating that the total number of combinations is 9. -/
theorem total_combinations_eq_nine : total_combinations = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_eq_nine_l2068_206847


namespace NUMINAMATH_CALUDE_sin_cos_difference_sin_negative_main_theorem_l2068_206850

theorem sin_cos_difference (x y : Real) : 
  Real.sin (x * π / 180) * Real.cos (y * π / 180) - Real.cos (x * π / 180) * Real.sin (y * π / 180) = 
  Real.sin ((x - y) * π / 180) :=
sorry

theorem sin_negative (x : Real) : Real.sin (-x) = -Real.sin x :=
sorry

theorem main_theorem : 
  Real.sin (24 * π / 180) * Real.cos (54 * π / 180) - Real.cos (24 * π / 180) * Real.sin (54 * π / 180) = -1/2 :=
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_sin_negative_main_theorem_l2068_206850


namespace NUMINAMATH_CALUDE_unique_six_digit_square_l2068_206870

/-- Checks if a number has all digits different --/
def has_different_digits (n : Nat) : Bool :=
  sorry

/-- Checks if digits in a number are in ascending order --/
def digits_ascending (n : Nat) : Bool :=
  sorry

/-- The unique six-digit perfect square with ascending, different digits --/
theorem unique_six_digit_square : 
  ∃! n : Nat, 
    100000 ≤ n ∧ n < 1000000 ∧  -- six-digit number
    has_different_digits n ∧ 
    digits_ascending n ∧ 
    ∃ m : Nat, n = m^2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_square_l2068_206870


namespace NUMINAMATH_CALUDE_f_derivative_l2068_206843

noncomputable def f (x : ℝ) : ℝ :=
  (5^x * (2 * Real.sin (2*x) + Real.cos (2*x) * Real.log 5)) / (4 + (Real.log 5)^2)

theorem f_derivative (x : ℝ) :
  deriv f x = 5^x * Real.cos (2*x) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l2068_206843


namespace NUMINAMATH_CALUDE_option1_cheapest_l2068_206856

/-- Regular ticket price -/
def regular_price : ℕ → ℕ := λ x => 40 * x

/-- Platinum card (Option 1) price -/
def platinum_price : ℕ → ℕ := λ x => 200 + 20 * x

/-- Diamond card (Option 2) price -/
def diamond_price : ℕ → ℕ := λ _ => 1000

/-- Theorem: For 8 < x < 40, Option 1 is the cheapest -/
theorem option1_cheapest (x : ℕ) (h1 : 8 < x) (h2 : x < 40) :
  platinum_price x < regular_price x ∧ platinum_price x < diamond_price x :=
by sorry

end NUMINAMATH_CALUDE_option1_cheapest_l2068_206856


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2068_206879

theorem quadratic_equation_solution : 
  ∀ a b : ℝ, 
  (∀ x : ℝ, x^2 - 6*x + 13 = 25 ↔ x = a ∨ x = b) → 
  a ≥ b → 
  3*a + 2*b = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2068_206879


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2068_206808

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I : ℂ)⁻¹ * (a + Complex.I) = 1 + b * Complex.I → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2068_206808


namespace NUMINAMATH_CALUDE_certain_number_proof_l2068_206861

theorem certain_number_proof (X : ℝ) : 
  X / 3 = (169.4915254237288 / 100) * 236 → X = 1200 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2068_206861


namespace NUMINAMATH_CALUDE_additional_grazing_area_l2068_206806

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 18^2 - π * 12^2 = 180 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l2068_206806


namespace NUMINAMATH_CALUDE_nonzero_real_solution_cube_equation_l2068_206886

theorem nonzero_real_solution_cube_equation (y : ℝ) (h1 : y ≠ 0) (h2 : (3 * y)^5 = (9 * y)^4) : y = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_real_solution_cube_equation_l2068_206886


namespace NUMINAMATH_CALUDE_touching_balls_theorem_l2068_206869

/-- Represents a spherical ball with a given radius -/
structure Ball where
  radius : ℝ

/-- Represents two touching balls on the ground -/
structure TouchingBalls where
  ball1 : Ball
  ball2 : Ball
  contactHeight : ℝ

/-- The radius of the other ball given the conditions -/
def otherBallRadius (balls : TouchingBalls) : ℝ := 6

theorem touching_balls_theorem (balls : TouchingBalls) 
  (h1 : balls.ball1.radius = 4)
  (h2 : balls.contactHeight = 6) :
  otherBallRadius balls = balls.ball2.radius :=
sorry

end NUMINAMATH_CALUDE_touching_balls_theorem_l2068_206869


namespace NUMINAMATH_CALUDE_bigger_part_of_52_l2068_206830

theorem bigger_part_of_52 (x y : ℕ) (h1 : x + y = 52) (h2 : 10 * x + 22 * y = 780) :
  max x y = 30 := by sorry

end NUMINAMATH_CALUDE_bigger_part_of_52_l2068_206830


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2068_206855

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2068_206855


namespace NUMINAMATH_CALUDE_problem_statement_l2068_206873

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2068_206873


namespace NUMINAMATH_CALUDE_trapezium_height_l2068_206811

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 18) (harea : area = 475) :
  (2 * area) / (a + b) = 25 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l2068_206811


namespace NUMINAMATH_CALUDE_system_solution_proof_l2068_206891

theorem system_solution_proof (x y z : ℝ) : 
  (2 * x + y = 3) ∧ 
  (3 * x - z = 7) ∧ 
  (x - y + 3 * z = 0) → 
  (x = 2 ∧ y = -1 ∧ z = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2068_206891


namespace NUMINAMATH_CALUDE_brick_length_is_25_cm_l2068_206813

/-- Proves that the length of each brick is 25 cm, given the wall dimensions,
    brick dimensions (except length), and the number of bricks needed. -/
theorem brick_length_is_25_cm
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (num_bricks : ℝ)
  (h_wall_length : wall_length = 2)
  (h_wall_height : wall_height = 3)
  (h_wall_thickness : wall_thickness = 0.02)
  (h_brick_width : brick_width = 0.11)
  (h_brick_height : brick_height = 0.06)
  (h_num_bricks : num_bricks = 72.72727272727273)
  : ∃ (brick_length : ℝ), brick_length = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_25_cm_l2068_206813


namespace NUMINAMATH_CALUDE_elderly_arrangements_proof_l2068_206835

def number_of_arrangements (n_volunteers : ℕ) (n_elderly : ℕ) : ℕ :=
  (n_volunteers.factorial) * 
  (n_volunteers - 1) * 
  (n_elderly.factorial)

theorem elderly_arrangements_proof :
  number_of_arrangements 4 2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_elderly_arrangements_proof_l2068_206835


namespace NUMINAMATH_CALUDE_jacket_discount_percentage_l2068_206889

/-- Calculates the discount percentage on a jacket sale --/
theorem jacket_discount_percentage
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 48)
  (h2 : markup_percentage = 0.4)
  (h3 : gross_profit = 16) :
  let selling_price := purchase_price / (1 - markup_percentage)
  let sale_price := purchase_price + gross_profit
  (selling_price - sale_price) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_jacket_discount_percentage_l2068_206889


namespace NUMINAMATH_CALUDE_total_miles_walked_l2068_206863

def monday_miles : ℕ := 9
def tuesday_miles : ℕ := 9

theorem total_miles_walked : monday_miles + tuesday_miles = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_walked_l2068_206863


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2068_206827

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rates for the two booths --/
structure ExchangeRates where
  redToSilver : TokenCount → TokenCount
  blueToSilver : TokenCount → TokenCount

/-- The initial token count --/
def initialTokens : TokenCount :=
  { red := 60, blue := 90, silver := 0 }

/-- The exchange rates for the two booths --/
def boothRates : ExchangeRates :=
  { redToSilver := λ tc => { red := tc.red - 3, blue := tc.blue + 2, silver := tc.silver + 1 },
    blueToSilver := λ tc => { red := tc.red + 1, blue := tc.blue - 4, silver := tc.silver + 2 } }

/-- Determines if further exchanges are possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens :
  ∃ (finalTokens : TokenCount),
    (¬canExchange finalTokens) ∧
    (finalTokens.silver = 101) ∧
    (∃ (exchanges : List (TokenCount → TokenCount)),
      exchanges.foldl (λ acc f => f acc) initialTokens = finalTokens) :=
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2068_206827


namespace NUMINAMATH_CALUDE_triangle_area_l2068_206828

/-- Given a triangle ABC where:
    - The side opposite to angle B has length 2
    - The side opposite to angle C has length 2√3
    - Angle C measures 2π/3 radians
    Prove that the area of the triangle is 3 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 2 * Real.sqrt 3 →
  C = 2 * π / 3 →
  (1/2) * b * c * Real.sin A = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2068_206828


namespace NUMINAMATH_CALUDE_trig_identity_l2068_206874

theorem trig_identity (α β : Real) : 
  (1 / Real.tan α)^2 + (1 / Real.tan β)^2 - 2 * Real.cos (β - α) / (Real.sin α * Real.sin β) + 2 = 
  Real.sin (α - β)^2 / (Real.sin α^2 * Real.sin β^2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2068_206874


namespace NUMINAMATH_CALUDE_appended_number_cube_sum_l2068_206845

theorem appended_number_cube_sum (a b c : ℕ) : 
  b ≥ 10 ∧ b < 100 ∧ c ≥ 10 ∧ c < 100 →
  10000 * a + 100 * b + c = (a + b + c)^3 →
  a = 9 ∧ b = 11 ∧ c = 25 :=
by sorry

end NUMINAMATH_CALUDE_appended_number_cube_sum_l2068_206845


namespace NUMINAMATH_CALUDE_cobys_speed_l2068_206834

/-- Coby's road trip problem -/
theorem cobys_speed (d_WI d_IN : ℝ) (v_WI : ℝ) (t_total : ℝ) (h1 : d_WI = 640) (h2 : d_IN = 550) (h3 : v_WI = 80) (h4 : t_total = 19) :
  (d_IN / (t_total - d_WI / v_WI)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_cobys_speed_l2068_206834


namespace NUMINAMATH_CALUDE_fraction_repeating_block_length_l2068_206857

/-- The number of digits in the smallest repeating block of the decimal expansion of 5/7 -/
def repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 5 / 7

theorem fraction_repeating_block_length :
  repeating_block_length = 6 ∧ 
  ∀ n : ℕ, n < repeating_block_length → 
    ∃ k : ℕ, fraction * 10^repeating_block_length - fraction * 10^n = k :=
sorry

end NUMINAMATH_CALUDE_fraction_repeating_block_length_l2068_206857


namespace NUMINAMATH_CALUDE_person_a_higher_probability_l2068_206840

/-- Represents the space station simulation programming challenge. -/
structure Challenge where
  total_questions : Nat
  questions_per_participant : Nat
  passing_threshold : Nat
  person_a_correct_questions : Nat
  person_b_success_probability : Real

/-- Calculates the probability of passing the challenge given the number of correct programs. -/
def probability_of_passing (c : Challenge) (correct_programs : Nat) : Real :=
  if correct_programs ≥ c.passing_threshold then 1 else 0

/-- Calculates the probability of person B passing the challenge. -/
def person_b_passing_probability (c : Challenge) : Real :=
  sorry

/-- Calculates the probability of person A passing the challenge. -/
def person_a_passing_probability (c : Challenge) : Real :=
  sorry

/-- The main theorem stating that person A has a higher probability of passing the challenge. -/
theorem person_a_higher_probability (c : Challenge) 
  (h1 : c.total_questions = 10)
  (h2 : c.questions_per_participant = 3)
  (h3 : c.passing_threshold = 2)
  (h4 : c.person_a_correct_questions = 6)
  (h5 : c.person_b_success_probability = 0.6) :
  person_a_passing_probability c > person_b_passing_probability c :=
sorry

end NUMINAMATH_CALUDE_person_a_higher_probability_l2068_206840


namespace NUMINAMATH_CALUDE_ab_value_l2068_206826

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + c = 2*b) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l2068_206826


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2068_206852

theorem polynomial_division_theorem (x : ℝ) :
  ∃ (q r : ℝ), 5*x^4 - 3*x^3 + 7*x^2 - 9*x + 12 = (x - 3) * (5*x^3 + 12*x^2 + 43*x + 120) + 372 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2068_206852


namespace NUMINAMATH_CALUDE_f_of_10_l2068_206849

theorem f_of_10 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 5*x*y = f (3*x - y) + 2*x^2 + 1) :
  f 10 = -49 := by
sorry

end NUMINAMATH_CALUDE_f_of_10_l2068_206849


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l2068_206800

theorem inequality_holds_iff (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, a * x^2 - (a + 2) * x + 2 < 0) ↔ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l2068_206800


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l2068_206880

/-- The area of a rhombus formed by the intersection of two equilateral triangles in a square -/
theorem rhombus_area_in_square (s : ℝ) (h : s = 4) : 
  let triangle_height := s * Real.sqrt 3 / 2
  let rhombus_diagonal1 := 2 * triangle_height - s
  let rhombus_diagonal2 := s
  rhombus_diagonal1 * rhombus_diagonal2 / 2 = 8 * Real.sqrt 3 - 8 := by
  sorry

#check rhombus_area_in_square

end NUMINAMATH_CALUDE_rhombus_area_in_square_l2068_206880


namespace NUMINAMATH_CALUDE_all_are_siblings_l2068_206890

-- Define a finite type with 7 elements to represent the boys
inductive Boy : Type
  | B1 | B2 | B3 | B4 | B5 | B6 | B7

-- Define the sibling relation
def is_sibling : Boy → Boy → Prop := sorry

-- State the theorem
theorem all_are_siblings :
  (∀ b : Boy, ∃ (s : Finset Boy), s.card ≥ 3 ∧ ∀ s' ∈ s, is_sibling b s') →
  (∀ b1 b2 : Boy, is_sibling b1 b2) :=
sorry

end NUMINAMATH_CALUDE_all_are_siblings_l2068_206890


namespace NUMINAMATH_CALUDE_max_lateral_area_inscribed_prism_l2068_206866

noncomputable section

-- Define the sphere's surface area
def sphere_surface_area : ℝ := 8 * Real.pi

-- Define the sphere's radius
def sphere_radius : ℝ := Real.sqrt (sphere_surface_area / (4 * Real.pi))

-- Define the base edge length of the prism
def base_edge_length : ℝ := Real.sqrt 3

-- Define the lateral area of the prism as a function of the base edge length
def lateral_area (x : ℝ) : ℝ := 
  6 * Real.sqrt (-(1/3) * (x^2 - 3)^2 + 3)

-- Theorem statement
theorem max_lateral_area_inscribed_prism :
  (lateral_area base_edge_length = 6 * Real.sqrt 3) ∧
  (∀ x : ℝ, 0 < x → x < Real.sqrt 6 → lateral_area x ≤ lateral_area base_edge_length) := by
  sorry

end

end NUMINAMATH_CALUDE_max_lateral_area_inscribed_prism_l2068_206866


namespace NUMINAMATH_CALUDE_susanas_chocolate_chips_l2068_206893

theorem susanas_chocolate_chips 
  (viviana_chocolate : ℕ) 
  (susana_chocolate : ℕ) 
  (viviana_vanilla : ℕ) 
  (susana_vanilla : ℕ) 
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : viviana_chocolate + susana_chocolate + viviana_vanilla + susana_vanilla = 90) :
  susana_chocolate = 25 := by
sorry

end NUMINAMATH_CALUDE_susanas_chocolate_chips_l2068_206893


namespace NUMINAMATH_CALUDE_no_rational_solution_l2068_206862

theorem no_rational_solution :
  ¬∃ (a b c d : ℚ) (n : ℕ), (a + b * Real.sqrt 2)^(2*n) + (c + d * Real.sqrt 2)^(2*n) = 5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l2068_206862


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_50n_two_satisfies_condition_two_is_smallest_l2068_206859

theorem smallest_n_for_sqrt_50n (n : ℕ) : (∃ k : ℕ, k * k = 50 * n) → n ≥ 2 := by
  sorry

theorem two_satisfies_condition : ∃ k : ℕ, k * k = 50 * 2 := by
  sorry

theorem two_is_smallest : ∀ n : ℕ, n > 0 → n < 2 → ¬(∃ k : ℕ, k * k = 50 * n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_50n_two_satisfies_condition_two_is_smallest_l2068_206859


namespace NUMINAMATH_CALUDE_constant_grid_function_l2068_206839

theorem constant_grid_function 
  (f : ℤ × ℤ → ℕ) 
  (h : ∀ x y : ℤ, 4 * f (x, y) = f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) : 
  ∀ p q : ℤ × ℤ, f p = f q :=
sorry

end NUMINAMATH_CALUDE_constant_grid_function_l2068_206839


namespace NUMINAMATH_CALUDE_max_trio_sum_l2068_206868

/-- A trio is a set of three distinct integers where two are divisors or multiples of the third -/
def is_trio (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ((a ∣ c ∧ b ∣ c) ∨ (a ∣ b ∧ c ∣ b) ∨ (b ∣ a ∧ c ∣ a))

/-- The set of integers from 1 to 2002 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2002}

theorem max_trio_sum :
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → is_trio a b c →
    a + b + c ≤ 4004 ∧
    (a + b + c = 4004 ↔ c = 2002 ∧ a ∣ 2002 ∧ b = 2002 - a) :=
sorry

end NUMINAMATH_CALUDE_max_trio_sum_l2068_206868


namespace NUMINAMATH_CALUDE_gcd_digit_bound_l2068_206892

theorem gcd_digit_bound (a b : ℕ) : 
  (10^6 ≤ a ∧ a < 10^7) →
  (10^6 ≤ b ∧ b < 10^7) →
  (10^12 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^13) →
  Nat.gcd a b < 10^2 :=
sorry

end NUMINAMATH_CALUDE_gcd_digit_bound_l2068_206892


namespace NUMINAMATH_CALUDE_range_of_k_l2068_206829

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

-- Define the complement of A in ℝ
def C_ℝA : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem range_of_k (k : ℝ) :
  (C_ℝA ∩ B k = ∅) → (k ≤ 0 ∨ k ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l2068_206829


namespace NUMINAMATH_CALUDE_complex_simplification_l2068_206897

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  3 * (4 - 2*i) + 2*i*(3 + 2*i) - (1 + i)*(2 - i) = 5 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2068_206897


namespace NUMINAMATH_CALUDE_nuts_per_student_l2068_206816

theorem nuts_per_student (bags : ℕ) (students : ℕ) (nuts_per_bag : ℕ) 
  (h1 : bags = 65) 
  (h2 : students = 13) 
  (h3 : nuts_per_bag = 15) : 
  (bags * nuts_per_bag) / students = 75 := by
sorry

end NUMINAMATH_CALUDE_nuts_per_student_l2068_206816


namespace NUMINAMATH_CALUDE_quadratic_bound_l2068_206877

theorem quadratic_bound (a b c : ℝ) 
  (h : ∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) : 
  ∀ x : ℝ, |x| ≤ 1 → |2*a*x + b| ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_bound_l2068_206877


namespace NUMINAMATH_CALUDE_odd_function_implies_a_zero_l2068_206821

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

-- State the theorem
theorem odd_function_implies_a_zero :
  (∀ x, f a x = -f a (-x)) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_zero_l2068_206821


namespace NUMINAMATH_CALUDE_power_mod_seven_l2068_206838

theorem power_mod_seven : 3^1995 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l2068_206838


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_diff_sum_of_roots_eq_fourteen_l2068_206841

theorem sum_of_roots_squared_diff (a b : ℝ) :
  (∃ x y : ℝ, (x - a)^2 = b^2 ∧ (y - a)^2 = b^2 ∧ x + y = 2 * a) :=
by
  sorry

theorem sum_of_roots_eq_fourteen :
  let roots := {x : ℝ | (x - 7)^2 = 16}
  (∃ x y : ℝ, x ∈ roots ∧ y ∈ roots ∧ x + y = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_diff_sum_of_roots_eq_fourteen_l2068_206841


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l2068_206883

theorem negation_of_forall_exp_gt_x :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_gt_x_l2068_206883


namespace NUMINAMATH_CALUDE_ken_snow_days_l2068_206802

/-- Represents the cycling scenario for Ken in a week -/
structure CyclingWeek where
  rain_speed : ℕ  -- miles per 20 minutes when raining
  snow_speed : ℕ  -- miles per 20 minutes when snowing
  rain_days : ℕ   -- number of rainy days
  total_miles : ℕ -- total miles cycled in the week
  hours_per_day : ℕ -- hours cycled per day

/-- Calculates the number of snowy days in a week -/
def snow_days (w : CyclingWeek) : ℕ :=
  ((w.total_miles - w.rain_days * w.rain_speed * 3) / (w.snow_speed * 3))

/-- Theorem stating the number of snowy days in Ken's cycling week -/
theorem ken_snow_days :
  let w : CyclingWeek := {
    rain_speed := 30,
    snow_speed := 10,
    rain_days := 3,
    total_miles := 390,
    hours_per_day := 1
  }
  snow_days w = 4 := by sorry

end NUMINAMATH_CALUDE_ken_snow_days_l2068_206802


namespace NUMINAMATH_CALUDE_exists_student_with_sqrt_k_classes_l2068_206818

/-- Represents a school with students and classes -/
structure School where
  n : ℕ  -- number of classes
  k : ℕ  -- number of students
  shared_class : Fin k → Fin k → Fin n
  class_size : Fin n → ℕ
  h1 : ∀ i j, i ≠ j → shared_class i j = shared_class j i
  h2 : ∀ i, class_size i < k
  h3 : ¬ ∃ m, k - 1 = m * m

/-- The number of classes a student attends -/
def num_classes_attended (s : School) (student : Fin s.k) : ℕ :=
  (Finset.univ.filter (λ c : Fin s.n => ∃ other, s.shared_class student other = c)).card

/-- Main theorem: There exists a student who has attended at least √k classes -/
theorem exists_student_with_sqrt_k_classes (s : School) :
  ∃ student : Fin s.k, s.k.sqrt ≤ num_classes_attended s student := by
  sorry

end NUMINAMATH_CALUDE_exists_student_with_sqrt_k_classes_l2068_206818


namespace NUMINAMATH_CALUDE_sqrt_diff_inequality_l2068_206848

theorem sqrt_diff_inequality (k : ℕ) (h : k ≥ 2) :
  Real.sqrt k - Real.sqrt (k - 1) > Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_sqrt_diff_inequality_l2068_206848


namespace NUMINAMATH_CALUDE_first_year_after_2100_digit_sum_15_l2068_206894

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Check if a year is after 2100 -/
def is_after_2100 (year : ℕ) : Prop :=
  year > 2100

/-- First year after 2100 with digit sum 15 -/
def first_year_after_2100_with_digit_sum_15 : ℕ := 2139

theorem first_year_after_2100_digit_sum_15 :
  (is_after_2100 first_year_after_2100_with_digit_sum_15) ∧
  (sum_of_digits first_year_after_2100_with_digit_sum_15 = 15) ∧
  (∀ y : ℕ, is_after_2100 y ∧ y < first_year_after_2100_with_digit_sum_15 →
    sum_of_digits y ≠ 15) :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2100_digit_sum_15_l2068_206894


namespace NUMINAMATH_CALUDE_five_letter_word_count_l2068_206895

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of five-letter words that begin and end with the same letter
    and have a vowel as the third letter -/
def word_count : ℕ := alphabet_size * alphabet_size * vowel_count * alphabet_size

theorem five_letter_word_count : word_count = 87880 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_word_count_l2068_206895


namespace NUMINAMATH_CALUDE_julia_tag_game_l2068_206875

theorem julia_tag_game (monday tuesday wednesday : ℕ) 
  (h1 : monday = 17) 
  (h2 : tuesday = 15) 
  (h3 : wednesday = 2) : 
  monday + tuesday + wednesday = 34 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l2068_206875


namespace NUMINAMATH_CALUDE_pie_slices_yesterday_l2068_206819

theorem pie_slices_yesterday (total : ℕ) (today : ℕ) (yesterday : ℕ) : 
  total = 7 → today = 2 → yesterday = total - today → yesterday = 5 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_yesterday_l2068_206819


namespace NUMINAMATH_CALUDE_simplify_expression_l2068_206833

theorem simplify_expression (x t : ℝ) : (x^2 * t^3) * (x^3 * t^4) = x^5 * t^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2068_206833


namespace NUMINAMATH_CALUDE_walter_age_theorem_l2068_206853

def walter_age_1999 (walter_age_1994 grandmother_age_1994 birth_year_sum : ℕ) : Prop :=
  walter_age_1994 * 2 = grandmother_age_1994 ∧
  (1994 - walter_age_1994) + (1994 - grandmother_age_1994) = birth_year_sum ∧
  walter_age_1994 + (1999 - 1994) = 55

theorem walter_age_theorem : 
  ∃ (walter_age_1994 grandmother_age_1994 : ℕ), 
    walter_age_1999 walter_age_1994 grandmother_age_1994 3838 :=
by
  sorry

end NUMINAMATH_CALUDE_walter_age_theorem_l2068_206853


namespace NUMINAMATH_CALUDE_f_properties_l2068_206823

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x - (1 / π) * x^2 + cos x

theorem f_properties :
  (∀ x ∈ Set.Icc 0 (π / 2), Monotone f) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < π → f x1 = f x2 → 
    deriv f ((x1 + x2) / 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2068_206823


namespace NUMINAMATH_CALUDE_area_of_triangle_BXC_l2068_206817

/-- Represents a trapezoid with bases and area -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Represents the triangle formed by the intersection of diagonals -/
structure DiagonalTriangle where
  area : ℝ

/-- Theorem stating the area of triangle BXC in the given trapezoid -/
theorem area_of_triangle_BXC (trapezoid : Trapezoid) (triangle : DiagonalTriangle) :
  trapezoid.base1 = 15 ∧ 
  trapezoid.base2 = 35 ∧ 
  trapezoid.area = 375 →
  triangle.area = 78.75 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_BXC_l2068_206817


namespace NUMINAMATH_CALUDE_car_max_acceleration_l2068_206825

theorem car_max_acceleration
  (g : ℝ) -- acceleration due to gravity
  (θ : ℝ) -- angle of the hill
  (μ : ℝ) -- coefficient of static friction
  (h1 : 0 < g)
  (h2 : 0 ≤ θ)
  (h3 : θ < π / 2)
  (h4 : μ > Real.tan θ) :
  ∃ a : ℝ,
    a = g * (μ * Real.cos θ - Real.sin θ) ∧
    ∀ a' : ℝ,
      (∃ m : ℝ, 0 < m ∧
        m * a' ≤ μ * (m * g * Real.cos θ) - m * g * Real.sin θ) →
      a' ≤ a :=
by sorry

end NUMINAMATH_CALUDE_car_max_acceleration_l2068_206825


namespace NUMINAMATH_CALUDE_problem_statement_l2068_206836

theorem problem_statement (a b k : ℝ) 
  (h1 : 2^a = k) 
  (h2 : 3^b = k) 
  (h3 : k ≠ 1) 
  (h4 : 1/a + 2/b = 1) : 
  k = 18 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2068_206836


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2068_206842

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 265 / 15 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2068_206842


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l2068_206881

theorem smallest_solution_quadratic (y : ℝ) : 
  (3 * y^2 + 33 * y - 90 = y * (y + 16)) → y ≥ -10 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l2068_206881


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_nine_l2068_206872

theorem product_of_fractions_equals_nine (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (a_nonzero : a ≠ 0)
  (b_nonzero : b ≠ 0)
  (c_nonzero : c ≠ 0)
  (a_neq_b : a ≠ b)
  (a_neq_c : a ≠ c)
  (b_neq_c : b ≠ c) :
  ((a - b) / c + (b - c) / a + (c - a) / b) * (c / (a - b) + a / (b - c) + b / (c - a)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_nine_l2068_206872


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2068_206831

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l2068_206831


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2068_206814

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0, 
    this function returns true if they are perpendicular. -/
def are_perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

/-- Theorem stating that for lines l₁: (m+2)x-(m-2)y+2=0 and l₂: 3x+my-1=0,
    if they are perpendicular, then m = 6 or m = -1 -/
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, are_perpendicular (m + 2) (-(m - 2)) 3 m → m = 6 ∨ m = -1 := by
  sorry

#check perpendicular_lines_m_values

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2068_206814


namespace NUMINAMATH_CALUDE_multiple_exists_l2068_206884

theorem multiple_exists (n : ℕ) (S : Finset ℕ) : 
  S ⊆ Finset.range (2 * n + 1) →
  S.card = n + 1 →
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
sorry

end NUMINAMATH_CALUDE_multiple_exists_l2068_206884


namespace NUMINAMATH_CALUDE_cube_side_length_l2068_206822

theorem cube_side_length (v : Real) (s : Real) :
  v = 8 →
  v = s^3 →
  ∃ (x : Real), 
    6 * x^2 = 3 * (6 * s^2) ∧
    x = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_l2068_206822


namespace NUMINAMATH_CALUDE_birds_joined_l2068_206899

theorem birds_joined (initial_birds : ℕ) (final_birds : ℕ) (initial_storks : ℕ) :
  let birds_joined := final_birds - initial_birds
  birds_joined = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_joined_l2068_206899


namespace NUMINAMATH_CALUDE_log_equation_solution_l2068_206864

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.log x + Real.log (x + 1) = 2 ∧ x = (-1 + Real.sqrt 401) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2068_206864


namespace NUMINAMATH_CALUDE_f_geq_6_iff_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_l2068_206888

-- Part 1
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 4|

theorem f_geq_6_iff (x : ℝ) : f x ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

-- Part 2
theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 4*c = 8) : 
  (1/a + 1/b + 1/c) ≥ (11 + 6*Real.sqrt 2) / 8 := by sorry

theorem min_value_sum_reciprocals_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + 4*c = 8) : 
  (1/a + 1/b + 1/c) = (11 + 6*Real.sqrt 2) / 8 ↔ a = Real.sqrt 2 * b ∧ b = 2 * c := by sorry

end NUMINAMATH_CALUDE_f_geq_6_iff_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_l2068_206888


namespace NUMINAMATH_CALUDE_gray_eyed_brunettes_l2068_206876

theorem gray_eyed_brunettes (total : ℕ) (green_eyed_blondes : ℕ) (brunettes : ℕ) (gray_eyed : ℕ)
  (h1 : total = 60)
  (h2 : green_eyed_blondes = 20)
  (h3 : brunettes = 35)
  (h4 : gray_eyed = 25) :
  total - brunettes - green_eyed_blondes = gray_eyed - (total - brunettes) + green_eyed_blondes :=
by
  sorry

#check gray_eyed_brunettes

end NUMINAMATH_CALUDE_gray_eyed_brunettes_l2068_206876


namespace NUMINAMATH_CALUDE_pencil_boxes_count_l2068_206867

theorem pencil_boxes_count (pencils_per_box : ℝ) (total_pencils : ℕ) 
  (h1 : pencils_per_box = 648.0) 
  (h2 : total_pencils = 2592) : 
  ↑total_pencils / pencils_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_boxes_count_l2068_206867


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2068_206804

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * (Real.rpow 2 (1/3))^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2068_206804


namespace NUMINAMATH_CALUDE_tom_apples_l2068_206882

/-- The number of apples Phillip has -/
def phillip_apples : ℕ := 40

/-- The number of apples Ben has more than Phillip -/
def ben_extra_apples : ℕ := 8

/-- The fraction of Ben's apples that Tom has -/
def tom_fraction : ℚ := 3 / 8

/-- Theorem stating that Tom has 18 apples -/
theorem tom_apples : ℕ := by sorry

end NUMINAMATH_CALUDE_tom_apples_l2068_206882


namespace NUMINAMATH_CALUDE_inscribed_circle_in_tangent_quadrilateral_l2068_206854

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Represents a quadrilateral formed by tangent lines -/
structure TangentQuadrilateral where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- Function to check if two circles intersect -/
def circlesIntersect (c1 c2 : Circle) : Prop := sorry

/-- Function to check if a quadrilateral is tangential (can have an inscribed circle) -/
def isTangentialQuadrilateral (quad : TangentQuadrilateral) : Prop := sorry

/-- Main theorem statement -/
theorem inscribed_circle_in_tangent_quadrilateral 
  (rect : Rectangle) 
  (circleA circleB circleC circleD : Circle)
  (quad : TangentQuadrilateral) :
  circleA.center = rect.A ∧
  circleB.center = rect.B ∧
  circleC.center = rect.C ∧
  circleD.center = rect.D ∧
  ¬(circlesIntersect circleA circleB) ∧
  ¬(circlesIntersect circleA circleC) ∧
  ¬(circlesIntersect circleA circleD) ∧
  ¬(circlesIntersect circleB circleC) ∧
  ¬(circlesIntersect circleB circleD) ∧
  ¬(circlesIntersect circleC circleD) ∧
  circleA.radius + circleC.radius = circleB.radius + circleD.radius →
  isTangentialQuadrilateral quad :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_in_tangent_quadrilateral_l2068_206854


namespace NUMINAMATH_CALUDE_negative_angle_quadrant_l2068_206860

/-- If an angle α is in the third quadrant, then -α is in the second quadrant -/
theorem negative_angle_quadrant (α : Real) : 
  (∃ k : ℤ, k * 2 * π + π < α ∧ α < k * 2 * π + 3 * π / 2) → 
  (∃ m : ℤ, m * 2 * π + π / 2 < -α ∧ -α < m * 2 * π + π) :=
by sorry

end NUMINAMATH_CALUDE_negative_angle_quadrant_l2068_206860


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l2068_206809

/-- The number of ways to arrange 4 boys and 3 girls in a line such that no two boys are adjacent -/
def boys_not_adjacent : ℕ := 144

/-- The number of ways to arrange 4 boys and 3 girls in a line such that no two girls are adjacent -/
def girls_not_adjacent : ℕ := 1440

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

theorem photo_lineup_arrangements :
  (boys_not_adjacent = 144) ∧ (girls_not_adjacent = 1440) := by
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l2068_206809


namespace NUMINAMATH_CALUDE_train_distance_problem_l2068_206832

theorem train_distance_problem (speed1 speed2 distance_difference : ℝ) 
  (h1 : speed1 = 16)
  (h2 : speed2 = 21)
  (h3 : distance_difference = 60)
  (time : ℝ)
  (h4 : time > 0)
  (distance1 : ℝ)
  (h5 : distance1 = speed1 * time)
  (distance2 : ℝ)
  (h6 : distance2 = speed2 * time)
  (h7 : distance2 = distance1 + distance_difference) :
  distance1 + distance2 = 444 := by
sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2068_206832


namespace NUMINAMATH_CALUDE_age_problem_l2068_206844

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2068_206844


namespace NUMINAMATH_CALUDE_math_majors_consecutive_seats_probability_l2068_206858

/-- The number of people sitting at the round table. -/
def totalPeople : ℕ := 12

/-- The number of math majors. -/
def mathMajors : ℕ := 5

/-- The number of physics majors. -/
def physicsMajors : ℕ := 4

/-- The number of biology majors. -/
def biologyMajors : ℕ := 3

/-- The probability of all math majors sitting in consecutive seats. -/
def probabilityConsecutiveSeats : ℚ := 1 / 66

theorem math_majors_consecutive_seats_probability :
  probabilityConsecutiveSeats = (totalPeople : ℚ) / (totalPeople.choose mathMajors) := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_seats_probability_l2068_206858


namespace NUMINAMATH_CALUDE_berkeley_class_as_l2068_206805

theorem berkeley_class_as (abraham_total : ℕ) (abraham_as : ℕ) (berkeley_total : ℕ) :
  abraham_total = 20 →
  abraham_as = 12 →
  berkeley_total = 30 →
  (berkeley_total : ℚ) * (abraham_as : ℚ) / (abraham_total : ℚ) = 18 :=
by sorry

end NUMINAMATH_CALUDE_berkeley_class_as_l2068_206805


namespace NUMINAMATH_CALUDE_maruti_car_price_increase_l2068_206896

theorem maruti_car_price_increase (P S : ℝ) (x : ℝ) 
  (h1 : P > 0) (h2 : S > 0) : 
  (P * (1 + x / 100) * (S * 0.8) = P * S * 1.04) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_maruti_car_price_increase_l2068_206896


namespace NUMINAMATH_CALUDE_prob_different_colors_l2068_206898

/-- The probability of drawing two balls of different colors from a box containing 3 red balls and 2 yellow balls. -/
theorem prob_different_colors (total : ℕ) (red : ℕ) (yellow : ℕ) : 
  total = 5 → red = 3 → yellow = 2 → 
  (red.choose 1 * yellow.choose 1 : ℚ) / total.choose 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_l2068_206898


namespace NUMINAMATH_CALUDE_convex_polygon_coverage_l2068_206824

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define a function to check if a polygon can cover a triangle of given area
def can_cover (M : ConvexPolygon) (area : ℝ) : Prop := sorry

-- Define a function to check if a polygon can be covered by a triangle of given area
def can_be_covered (M : ConvexPolygon) (area : ℝ) : Prop := sorry

-- Theorem statement
theorem convex_polygon_coverage (M : ConvexPolygon) :
  (¬ can_cover M 1) → can_be_covered M 4 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_coverage_l2068_206824
