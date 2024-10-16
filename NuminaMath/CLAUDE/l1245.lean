import Mathlib

namespace NUMINAMATH_CALUDE_area_max_opposite_angles_sum_pi_l1245_124510

/-- A quadrilateral with sides a, b, c, d and angles α, β, γ, δ. -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  angle_sum : α + β + γ + δ = 2 * Real.pi

/-- The area of a quadrilateral. -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The area of a quadrilateral is maximized when the sum of its opposite angles is π (180°). -/
theorem area_max_opposite_angles_sum_pi (q : Quadrilateral) :
  ∀ q' : Quadrilateral, q'.a = q.a ∧ q'.b = q.b ∧ q'.c = q.c ∧ q'.d = q.d →
  area q' ≤ area q ↔ q.α + q.γ = Real.pi ∧ q.β + q.δ = Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_max_opposite_angles_sum_pi_l1245_124510


namespace NUMINAMATH_CALUDE_platinum_sphere_weight_in_mercury_l1245_124533

/-- The weight of a platinum sphere in mercury at elevated temperature -/
theorem platinum_sphere_weight_in_mercury
  (p : ℝ)
  (d₁ : ℝ)
  (d₂ : ℝ)
  (a₁ : ℝ)
  (a₂ : ℝ)
  (h_p : p = 30)
  (h_d₁ : d₁ = 21.5)
  (h_d₂ : d₂ = 13.60)
  (h_a₁ : a₁ = 0.0000264)
  (h_a₂ : a₂ = 0.0001815)
  : ∃ w : ℝ, abs (w - 11.310) < 0.001 :=
by
  sorry


end NUMINAMATH_CALUDE_platinum_sphere_weight_in_mercury_l1245_124533


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1245_124575

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1245_124575


namespace NUMINAMATH_CALUDE_factorization_x4_minus_81_complete_factorization_l1245_124537

theorem factorization_x4_minus_81 (x : ℝ) :
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by sorry

theorem complete_factorization (p q r : ℝ → ℝ) :
  (∀ x, x^4 - 81 = p x * q x * r x) →
  (∀ x, p x = x - 3 ∨ p x = x + 3 ∨ p x = x^2 + 9) →
  (∀ x, q x = x - 3 ∨ q x = x + 3 ∨ q x = x^2 + 9) →
  (∀ x, r x = x - 3 ∨ r x = x + 3 ∨ r x = x^2 + 9) →
  (p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  (∀ x, p x * q x * r x = (x - 3) * (x + 3) * (x^2 + 9)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_81_complete_factorization_l1245_124537


namespace NUMINAMATH_CALUDE_chord_intersections_count_l1245_124518

/-- The number of intersection points of chords on a circle -/
def chord_intersections (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of intersection points of chords drawn between n vertices 
    on a circle, excluding the vertices themselves, is equal to binom(n, 4), 
    given that no three chords are concurrent except at a vertex. -/
theorem chord_intersections_count (n : ℕ) (h : n ≥ 4) :
  chord_intersections n = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersections_count_l1245_124518


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1245_124551

theorem sum_of_two_numbers (x y : ℝ) : 
  (x + y) + (x - y) = 8 → x^2 - y^2 = 160 → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1245_124551


namespace NUMINAMATH_CALUDE_next_challenge_digits_estimate_l1245_124530

/-- The number of decimal digits in RSA-640 -/
def rsa640_digits : ℕ := 193

/-- The prize amount for RSA-640 in dollars -/
def rsa640_prize : ℕ := 20000

/-- The prize amount for the next challenge in dollars -/
def next_challenge_prize : ℕ := 30000

/-- A reasonable upper bound for the number of digits in the next challenge -/
def reasonable_upper_bound : ℕ := 220

/-- Theorem stating that a reasonable estimate for the number of digits
    in the next challenge is greater than RSA-640's digits and at most 220 -/
theorem next_challenge_digits_estimate :
  ∃ (N : ℕ), N > rsa640_digits ∧ N ≤ reasonable_upper_bound ∧
  next_challenge_prize > rsa640_prize :=
sorry

end NUMINAMATH_CALUDE_next_challenge_digits_estimate_l1245_124530


namespace NUMINAMATH_CALUDE_rolling_cone_surface_area_l1245_124501

/-- The surface area described by the height of a rolling cone -/
theorem rolling_cone_surface_area (h l : ℝ) (h_pos : 0 < h) (l_pos : 0 < l) :
  let surface_area := π * h^3 / l
  surface_area = π * h^3 / l :=
by sorry

end NUMINAMATH_CALUDE_rolling_cone_surface_area_l1245_124501


namespace NUMINAMATH_CALUDE_estimate_black_pieces_is_twelve_l1245_124528

/-- Represents the result of drawing chess pieces -/
structure DrawResult where
  total_pieces : ℕ
  total_draws : ℕ
  black_draws : ℕ

/-- Estimates the number of black chess pieces in the bag -/
def estimate_black_pieces (result : DrawResult) : ℚ :=
  result.total_pieces * (result.black_draws : ℚ) / result.total_draws

/-- Theorem: The estimated number of black chess pieces is 12 -/
theorem estimate_black_pieces_is_twelve (result : DrawResult) 
  (h1 : result.total_pieces = 20)
  (h2 : result.total_draws = 100)
  (h3 : result.black_draws = 60) : 
  estimate_black_pieces result = 12 := by
  sorry

#eval estimate_black_pieces ⟨20, 100, 60⟩

end NUMINAMATH_CALUDE_estimate_black_pieces_is_twelve_l1245_124528


namespace NUMINAMATH_CALUDE_log_sum_minus_exp_equality_l1245_124557

theorem log_sum_minus_exp_equality : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 - 42 * 8^(1/4) - 2017^0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_minus_exp_equality_l1245_124557


namespace NUMINAMATH_CALUDE_A_inter_B_l1245_124569

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_inter_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_l1245_124569


namespace NUMINAMATH_CALUDE_abs_negative_2035_l1245_124578

theorem abs_negative_2035 : |(-2035 : ℤ)| = 2035 := by sorry

end NUMINAMATH_CALUDE_abs_negative_2035_l1245_124578


namespace NUMINAMATH_CALUDE_berry_difference_l1245_124539

theorem berry_difference (stacy_initial : ℕ) (steve_initial : ℕ) (taken : ℕ) : 
  stacy_initial = 32 → 
  steve_initial = 21 → 
  taken = 4 → 
  stacy_initial - (steve_initial + taken) = 7 := by
sorry

end NUMINAMATH_CALUDE_berry_difference_l1245_124539


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1245_124546

-- Define p and q as predicates on real numbers x and y
def p (x y : ℝ) : Prop := x + y ≠ 4
def q (x y : ℝ) : Prop := x ≠ 1 ∨ y ≠ 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1245_124546


namespace NUMINAMATH_CALUDE_cos_double_angle_special_l1245_124549

theorem cos_double_angle_special (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = 1/5) : Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_l1245_124549


namespace NUMINAMATH_CALUDE_min_tiles_to_cover_l1245_124503

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Theorem stating the minimum number of tiles needed to cover the specified area -/
theorem min_tiles_to_cover (tileSize : Dimensions) (regionSize : Dimensions) (coveredSize : Dimensions) : 
  tileSize.length = 3 →
  tileSize.width = 4 →
  regionSize.length = feetToInches 3 →
  regionSize.width = feetToInches 6 →
  coveredSize.length = feetToInches 1 →
  coveredSize.width = feetToInches 1 →
  (area regionSize - area coveredSize) / area tileSize = 204 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_to_cover_l1245_124503


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1245_124593

theorem polynomial_coefficient_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 - x^5 = a + a₁*(x + 4)^4*x + a₂*(x + 1)^3*x^2 + a₃*(x + 1)^2*x^3 + a₄*(x + 1)*x^4) →
  a₁ + a₃ = 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1245_124593


namespace NUMINAMATH_CALUDE_additional_batches_is_seven_l1245_124597

/-- Represents the number of cups of flour needed for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the number of batches of cookies Gigi baked -/
def batches_baked : ℕ := 3

/-- Represents the initial amount of flour in cups -/
def initial_flour : ℕ := 20

/-- Calculates the number of additional batches that can be made with the remaining flour -/
def additional_batches : ℕ :=
  (initial_flour - flour_per_batch * batches_baked) / flour_per_batch

/-- Theorem stating that the number of additional batches is 7 -/
theorem additional_batches_is_seven :
  additional_batches = 7 := by sorry

end NUMINAMATH_CALUDE_additional_batches_is_seven_l1245_124597


namespace NUMINAMATH_CALUDE_g_composed_has_two_distinct_roots_l1245_124588

/-- The function g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_composed (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 2 distinct real roots when d = 8 -/
theorem g_composed_has_two_distinct_roots :
  ∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
  (∀ x : ℝ, g_composed 8 x = 0 ↔ x = r₁ ∨ x = r₂) :=
sorry

end NUMINAMATH_CALUDE_g_composed_has_two_distinct_roots_l1245_124588


namespace NUMINAMATH_CALUDE_adjacent_teacher_performances_probability_l1245_124587

-- Define the number of student performances
def num_student_performances : ℕ := 5

-- Define the number of teacher performances
def num_teacher_performances : ℕ := 2

-- Define the total number of performances
def total_performances : ℕ := num_student_performances + num_teacher_performances

-- Define the function to calculate the probability
def probability_adjacent_teacher_performances : ℚ :=
  (num_student_performances + 1 : ℚ) * 2 / ((total_performances : ℚ) * (total_performances - 1 : ℚ) / 2)

-- Theorem statement
theorem adjacent_teacher_performances_probability :
  probability_adjacent_teacher_performances = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_teacher_performances_probability_l1245_124587


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1245_124511

theorem interest_rate_calculation (P t D : ℝ) (h1 : P = 500) (h2 : t = 2) (h3 : D = 20) : 
  ∃ r : ℝ, r = 20 ∧ 
    P * ((1 + r / 100) ^ t - 1) - P * r * t / 100 = D :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1245_124511


namespace NUMINAMATH_CALUDE_range_of_q_l1245_124531

def q (x : ℝ) : ℝ := x^4 - 4*x^2 + 4

theorem range_of_q :
  Set.range q = Set.Icc 0 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_q_l1245_124531


namespace NUMINAMATH_CALUDE_sprite_to_coke_ratio_l1245_124599

/-- Represents a drink mixture with three components -/
structure Drink where
  total : ℝ
  coke : ℝ
  sprite : ℝ
  mountainDew : ℝ
  cokeParts : ℝ
  mountainDewParts : ℝ

/-- Theorem stating the ratio of Sprite to Coke in the drink -/
theorem sprite_to_coke_ratio (d : Drink) 
  (h1 : d.total = 18)
  (h2 : d.coke = 6)
  (h3 : d.cokeParts = 2)
  (h4 : d.mountainDewParts = 3)
  (h5 : d.total = d.coke + d.sprite + d.mountainDew)
  (h6 : d.coke / d.cokeParts = d.mountainDew / d.mountainDewParts) : 
  d.sprite / d.coke = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sprite_to_coke_ratio_l1245_124599


namespace NUMINAMATH_CALUDE_sandy_molly_age_difference_l1245_124584

theorem sandy_molly_age_difference :
  ∀ (sandy_age molly_age : ℕ),
    sandy_age = 70 →
    sandy_age * 9 = molly_age * 7 →
    molly_age - sandy_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_difference_l1245_124584


namespace NUMINAMATH_CALUDE_octal_perfect_square_b_is_one_l1245_124577

/-- Represents a digit in base 8 -/
def OctalDigit := { n : Nat // n < 8 }

/-- Converts a number from base 8 to decimal -/
def octalToDecimal (a b c : OctalDigit) : Nat :=
  512 * a.val + 192 + 8 * b.val + c.val

/-- Represents a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

theorem octal_perfect_square_b_is_one
  (a : OctalDigit)
  (h_a : a.val ≠ 0)
  (b : OctalDigit)
  (c : OctalDigit) :
  isPerfectSquare (octalToDecimal a b c) → b.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_octal_perfect_square_b_is_one_l1245_124577


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1245_124554

theorem complex_absolute_value (z : ℂ) (h : (z + 1) * Complex.I = 3 + 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1245_124554


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1188_l1245_124547

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 13 -/
def C : ℕ := 12

theorem sum_of_bases_equals_1188 :
  base8ToBase10 537 + base13ToBase10 (4 * 13^2 + C * 13 + 5) = 1188 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1188_l1245_124547


namespace NUMINAMATH_CALUDE_A_infinite_l1245_124553

/-- τ(n) denotes the number of positive divisors of the positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The set of positive integers a for which τ(an) = n has no positive integer solutions n -/
def A : Set ℕ+ := {a | ∀ n : ℕ+, tau (a * n) ≠ n}

/-- Theorem: The set A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

end NUMINAMATH_CALUDE_A_infinite_l1245_124553


namespace NUMINAMATH_CALUDE_negation_equivalence_l1245_124556

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1245_124556


namespace NUMINAMATH_CALUDE_A_equals_Z_l1245_124548

-- Define the set A
def A : Set Int :=
  {n | ∃ (a b : Nat), a ≥ 1 ∧ b ≥ 1 ∧ n = 2^a - 2^b}

-- Define the closure property of A
axiom A_closure (a b : Int) : a ∈ A → b ∈ A → (a + b) ∈ A

-- Axiom stating that A contains at least one odd number
axiom A_contains_odd : ∃ (n : Int), n ∈ A ∧ n % 2 ≠ 0

-- Theorem to prove
theorem A_equals_Z : A = Set.univ := by sorry

end NUMINAMATH_CALUDE_A_equals_Z_l1245_124548


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l1245_124558

theorem quadratic_is_perfect_square (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, 9*x^2 + 24*x + a = (b*x + c)^2) → a = 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l1245_124558


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l1245_124555

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 775)
  (h2 : new_price = 620) :
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l1245_124555


namespace NUMINAMATH_CALUDE_bus_wheel_radius_l1245_124579

/-- The radius of a bus wheel given its speed and revolutions per minute -/
theorem bus_wheel_radius 
  (speed_kmh : ℝ) 
  (rpm : ℝ) 
  (h1 : speed_kmh = 66) 
  (h2 : rpm = 175.15923566878982) : 
  ∃ (r : ℝ), abs (r - 99.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bus_wheel_radius_l1245_124579


namespace NUMINAMATH_CALUDE_xyz_equals_ten_l1245_124506

theorem xyz_equals_ten (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (sum_prod : x*y + x*z + y*z = 10)
  (sum : x + y + z = 6) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_ten_l1245_124506


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l1245_124574

/-- Represents the number of wheels on a vehicle -/
inductive VehicleType
  | twoWheeler
  | fourWheeler

/-- Calculates the number of wheels for a given vehicle type -/
def wheelCount (v : VehicleType) : Nat :=
  match v with
  | .twoWheeler => 2
  | .fourWheeler => 4

/-- Represents a parking configuration -/
structure ParkingConfig where
  twoWheelers : Nat
  fourWheelers : Nat

/-- Calculates the total number of wheels for a given parking configuration -/
def totalWheels (config : ParkingConfig) : Nat :=
  config.twoWheelers * wheelCount VehicleType.twoWheeler +
  config.fourWheelers * wheelCount VehicleType.fourWheeler

/-- Theorem stating that multiple solutions exist for the parking problem -/
theorem multiple_solutions_exist :
  ∃ (config1 config2 : ParkingConfig),
    totalWheels config1 = 70 ∧
    totalWheels config2 = 70 ∧
    config1.fourWheelers ≠ config2.fourWheelers :=
by
  sorry

#check multiple_solutions_exist

end NUMINAMATH_CALUDE_multiple_solutions_exist_l1245_124574


namespace NUMINAMATH_CALUDE_fish_value_in_rice_fish_value_in_rice_mixed_l1245_124505

-- Define the trading rates
def fish_to_bread_rate : ℚ := 3 / 5
def bread_to_rice_rate : ℕ := 7

-- Theorem statement
theorem fish_value_in_rice : 
  fish_to_bread_rate * bread_to_rice_rate = 21 / 5 := by
  sorry

-- Converting the result to a mixed number
theorem fish_value_in_rice_mixed : 
  ∃ (whole : ℕ) (frac : ℚ), 
    fish_to_bread_rate * bread_to_rice_rate = whole + frac ∧ 
    whole = 4 ∧ 
    frac = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fish_value_in_rice_fish_value_in_rice_mixed_l1245_124505


namespace NUMINAMATH_CALUDE_largest_square_advertisement_l1245_124512

theorem largest_square_advertisement (rectangle_width rectangle_length min_border : Real)
  (h1 : rectangle_width = 9)
  (h2 : rectangle_length = 16)
  (h3 : min_border = 1.5)
  (h4 : rectangle_width ≤ rectangle_length) :
  let max_side := min (rectangle_width - 2 * min_border) (rectangle_length - 2 * min_border)
  (max_side * max_side) = 36 := by
  sorry

#check largest_square_advertisement

end NUMINAMATH_CALUDE_largest_square_advertisement_l1245_124512


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1245_124527

/-- An equilateral triangle with height 9 and perimeter 36 has area 54 -/
theorem equilateral_triangle_area (h : ℝ) (p : ℝ) :
  h = 9 → p = 36 → (1/2) * (p/3) * h = 54 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1245_124527


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1245_124525

/-- Given a cylinder with height 15 cm and radius 5 cm, and a cone with the same radius
    and height one-third of the cylinder's, prove that the ratio of their volumes is 1/9. -/
theorem cone_cylinder_volume_ratio :
  let cylinder_height : ℝ := 15
  let cylinder_radius : ℝ := 5
  let cone_radius : ℝ := cylinder_radius
  let cone_height : ℝ := cylinder_height / 3
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  let cone_volume := (1/3) * π * cone_radius^2 * cone_height
  cone_volume / cylinder_volume = 1/9 := by
sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l1245_124525


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1245_124538

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1245_124538


namespace NUMINAMATH_CALUDE_specificGrid_toothpicks_l1245_124514

/-- Represents a rectangular grid with diagonal supports -/
structure ToothpickGrid where
  length : ℕ
  width : ℕ
  diagonalInterval : ℕ

/-- Calculates the total number of toothpicks used in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let verticalToothpicks := (grid.length + 1) * grid.width
  let horizontalToothpicks := (grid.width + 1) * grid.length
  let diagonalLines := (grid.length + 1) / grid.diagonalInterval + (grid.width + 1) / grid.diagonalInterval
  let diagonalToothpicks := diagonalLines * 7  -- Approximation of √50
  verticalToothpicks + horizontalToothpicks + diagonalToothpicks

/-- The specific grid described in the problem -/
def specificGrid : ToothpickGrid :=
  { length := 45
    width := 25
    diagonalInterval := 5 }

theorem specificGrid_toothpicks :
  totalToothpicks specificGrid = 2446 := by
  sorry

end NUMINAMATH_CALUDE_specificGrid_toothpicks_l1245_124514


namespace NUMINAMATH_CALUDE_x_plus_y_values_l1245_124517

theorem x_plus_y_values (x y : ℝ) (hx : x = y * (3 - y)^2) (hy : y = x * (3 - x)^2) :
  x + y ∈ ({0, 3, 4, 5, 8} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l1245_124517


namespace NUMINAMATH_CALUDE_three_painters_three_rooms_l1245_124583

/-- Represents the time taken for painters to complete rooms -/
def time_to_complete (painters : ℕ) (rooms : ℕ) : ℝ := sorry

/-- The work rate is proportional to the number of painters -/
axiom work_rate_proportional (p1 p2 r1 r2 : ℕ) (t : ℝ) :
  time_to_complete p1 r1 = t → time_to_complete p2 r2 = t * (r2 * p1 : ℝ) / (r1 * p2 : ℝ)

theorem three_painters_three_rooms : 
  time_to_complete 9 27 = 9 → time_to_complete 3 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_painters_three_rooms_l1245_124583


namespace NUMINAMATH_CALUDE_square_equals_four_digit_l1245_124572

theorem square_equals_four_digit : ∃ (M N : ℕ), 
  10 ≤ M ∧ M < 100 ∧ 
  1000 ≤ N ∧ N < 10000 ∧ 
  M^2 = N :=
sorry

end NUMINAMATH_CALUDE_square_equals_four_digit_l1245_124572


namespace NUMINAMATH_CALUDE_b₁_value_l1245_124526

/-- The polynomial f(x) with 4 distinct real roots -/
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

/-- The set of roots of f(x) -/
def roots_f : Set ℝ := {x | f x = 0}

/-- The polynomial g(x) with roots being squares of roots of f(x) -/
def g (b₀ b₁ b₂ b₃ : ℝ) (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

/-- The set of roots of g(x) -/
def roots_g (b₀ b₁ b₂ b₃ : ℝ) : Set ℝ := {x | g b₀ b₁ b₂ b₃ x = 0}

theorem b₁_value (b₀ b₁ b₂ b₃ : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
    roots_f = {x₁, x₂, x₃, x₄} ∧ 
    roots_g b₀ b₁ b₂ b₃ = {x₁^2, x₂^2, x₃^2, x₄^2}) →
  b₁ = -1216 := by
sorry

end NUMINAMATH_CALUDE_b₁_value_l1245_124526


namespace NUMINAMATH_CALUDE_sum_two_smallest_angles_l1245_124586

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the angles of the quadrilateral
def angle (P Q R : Point) : ℝ := sorry

-- Define the conditions
axiom quad_angles_arithmetic : ∃ (a d : ℝ), 
  angle B A D = a ∧
  angle A B C = a + d ∧
  angle B C D = a + 2*d ∧
  angle C D A = a + 3*d

axiom angle_equality1 : angle A B D = angle D B C
axiom angle_equality2 : angle A D B = angle B D C

axiom triangle_ABD_arithmetic : ∃ (x y : ℝ),
  angle B A D = x ∧
  angle A B D = x + y ∧
  angle A D B = x + 2*y

axiom triangle_DCB_arithmetic : ∃ (x y : ℝ),
  angle D C B = x ∧
  angle C D B = x + y ∧
  angle C B D = x + 2*y

axiom smallest_angle : angle B A D = 10
axiom second_angle : angle A B C = 70

-- Theorem to prove
theorem sum_two_smallest_angles :
  angle B A D + angle A B C = 80 := by sorry

end NUMINAMATH_CALUDE_sum_two_smallest_angles_l1245_124586


namespace NUMINAMATH_CALUDE_iains_pennies_l1245_124598

theorem iains_pennies (initial_pennies : ℕ) (older_pennies : ℕ) (discard_percentage : ℚ) : 
  initial_pennies = 200 →
  older_pennies = 30 →
  discard_percentage = 1/5 →
  initial_pennies - older_pennies - (initial_pennies - older_pennies) * discard_percentage = 136 := by
  sorry

end NUMINAMATH_CALUDE_iains_pennies_l1245_124598


namespace NUMINAMATH_CALUDE_alternate_interior_angles_relationship_l1245_124507

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to create alternate interior angles
def alternate_interior_angles (l1 l2 l3 : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem alternate_interior_angles_relationship 
  (l1 l2 l3 : Line) : 
  ¬ (∀ (a1 a2 : Angle), 
    (a1, a2) = alternate_interior_angles l1 l2 l3 → 
    a1.measure = a2.measure ∨ 
    a1.measure ≠ a2.measure) :=
sorry

end NUMINAMATH_CALUDE_alternate_interior_angles_relationship_l1245_124507


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1245_124519

/-- Given a geometric sequence with first term 2 and fifth term 18, the third term is 6 -/
theorem geometric_sequence_third_term : ∀ (x y z : ℝ),
  (∃ (q : ℝ), q ≠ 0 ∧ x = 2 * q ∧ y = 2 * q^2 ∧ z = 2 * q^3 ∧ 18 = 2 * q^4) →
  y = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1245_124519


namespace NUMINAMATH_CALUDE_last_digit_tower3_5_l1245_124563

/-- The last digit of a number n -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo m -/
def powMod (base exp m : ℕ) : ℕ :=
  (base ^ exp) % m

/-- The tower of powers of 3 with height 5 -/
def tower3_5 : ℕ := 3^(3^(3^(3^3)))

/-- The last digit of the tower of powers of 3 with height 5 is 7 -/
theorem last_digit_tower3_5 : lastDigit tower3_5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_tower3_5_l1245_124563


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1245_124542

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) 
  (h1 : total_questions = 60)
  (h2 : correct_answers = 40)
  (h3 : total_marks = 140)
  (h4 : correct_answers ≤ total_questions) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1245_124542


namespace NUMINAMATH_CALUDE_box_volume_increase_l1245_124581

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + l * h + w * h) = 1950)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7198 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1245_124581


namespace NUMINAMATH_CALUDE_rotate_rectangle_is_cylinder_l1245_124560

/-- A rectangle is a 2D shape with four sides and four right angles. -/
structure Rectangle where
  width : ℝ
  height : ℝ
  (positive_dimensions : width > 0 ∧ height > 0)

/-- A cylinder is a 3D shape with two circular bases connected by a curved surface. -/
structure Cylinder where
  radius : ℝ
  height : ℝ
  (positive_dimensions : radius > 0 ∧ height > 0)

/-- The result of rotating a rectangle around one of its sides. -/
def rotateRectangle (r : Rectangle) : Cylinder :=
  sorry

/-- Theorem stating that rotating a rectangle around one of its sides results in a cylinder. -/
theorem rotate_rectangle_is_cylinder (r : Rectangle) :
  ∃ (c : Cylinder), c = rotateRectangle r :=
sorry

end NUMINAMATH_CALUDE_rotate_rectangle_is_cylinder_l1245_124560


namespace NUMINAMATH_CALUDE_mystical_village_population_l1245_124592

theorem mystical_village_population : ∃ (x y z : ℕ+), 
  (y.val : ℤ)^2 = (x.val : ℤ)^2 + 200 ∧ 
  (z.val : ℤ)^2 = (y.val : ℤ)^2 + 180 ∧ 
  ∃ (k : ℕ), x.val^2 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_mystical_village_population_l1245_124592


namespace NUMINAMATH_CALUDE_work_days_calculation_l1245_124534

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the total earnings of all workers -/
def totalEarnings (days : WorkDays) (wages : DailyWages) : ℕ :=
  days.a * wages.a + days.b * wages.b + days.c * wages.c

/-- The main theorem stating the problem conditions and the result to be proved -/
theorem work_days_calculation (days : WorkDays) (wages : DailyWages) :
  days.a = 6 ∧
  days.c = 4 ∧
  wages.a * 4 = wages.b * 3 ∧
  wages.b * 5 = wages.c * 4 ∧
  wages.c = 125 ∧
  totalEarnings days wages = 1850 →
  days.b = 9 := by
  sorry


end NUMINAMATH_CALUDE_work_days_calculation_l1245_124534


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1245_124509

def proposition (x : Real) : Prop := x ∈ Set.Icc 0 (2 * Real.pi) → |Real.sin x| ≤ 1

theorem negation_of_proposition :
  (¬ ∀ x, proposition x) ↔ (∃ x, x ∈ Set.Icc 0 (2 * Real.pi) ∧ |Real.sin x| > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1245_124509


namespace NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l1245_124545

/-- A positive integer n is lovely if there exists a positive integer k and 
    positive integers d₁, d₂, ..., dₖ such that n = d₁d₂...dₖ and d_i² | n+d_i for all i ∈ {1, ..., k}. -/
def IsLovely (n : ℕ+) : Prop :=
  ∃ k : ℕ+, ∃ d : Fin k → ℕ+, 
    (n = (Finset.univ.prod (λ i => d i))) ∧ 
    (∀ i : Fin k, (d i)^2 ∣ (n + d i))

/-- There are infinitely many lovely numbers. -/
theorem infinitely_many_lovely_numbers : ∀ N : ℕ, ∃ n : ℕ+, n > N ∧ IsLovely n :=
sorry

/-- There does not exist a lovely number greater than 1 which is a square of an integer. -/
theorem no_lovely_square_greater_than_one : ¬∃ n : ℕ+, n > 1 ∧ ∃ m : ℕ+, n = m^2 ∧ IsLovely n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l1245_124545


namespace NUMINAMATH_CALUDE_cat_mouse_position_after_299_moves_l1245_124550

/-- Represents the four rooms for the cat --/
inductive CatRoom
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the eight segments for the mouse --/
inductive MouseSegment
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- Function to determine cat's position after n moves --/
def catPosition (n : ℕ) : CatRoom :=
  match (n - n / 100) % 4 with
  | 0 => CatRoom.TopLeft
  | 1 => CatRoom.TopRight
  | 2 => CatRoom.BottomRight
  | _ => CatRoom.BottomLeft

/-- Function to determine mouse's position after n moves --/
def mousePosition (n : ℕ) : MouseSegment :=
  match n % 8 with
  | 0 => MouseSegment.TopLeft
  | 1 => MouseSegment.TopMiddle
  | 2 => MouseSegment.TopRight
  | 3 => MouseSegment.RightMiddle
  | 4 => MouseSegment.BottomRight
  | 5 => MouseSegment.BottomMiddle
  | 6 => MouseSegment.BottomLeft
  | _ => MouseSegment.LeftMiddle

theorem cat_mouse_position_after_299_moves :
  catPosition 299 = CatRoom.TopLeft ∧
  mousePosition 299 = MouseSegment.RightMiddle :=
by sorry

end NUMINAMATH_CALUDE_cat_mouse_position_after_299_moves_l1245_124550


namespace NUMINAMATH_CALUDE_matches_per_box_l1245_124580

/-- Given 5 dozen boxes containing a total of 1200 matches, prove that each box contains 20 matches. -/
theorem matches_per_box (dozen_boxes : ℕ) (total_matches : ℕ) : 
  dozen_boxes = 5 → total_matches = 1200 → (dozen_boxes * 12) * 20 = total_matches := by
  sorry

end NUMINAMATH_CALUDE_matches_per_box_l1245_124580


namespace NUMINAMATH_CALUDE_correct_prediction_probability_l1245_124596

theorem correct_prediction_probability :
  let n_monday : ℕ := 5
  let n_tuesday : ℕ := 6
  let n_total : ℕ := n_monday + n_tuesday
  let n_correct : ℕ := 7
  let n_correct_monday : ℕ := 3
  let n_correct_tuesday : ℕ := n_correct - n_correct_monday
  let p : ℝ := 1 / 2

  (Nat.choose n_monday n_correct_monday * p^n_monday * (1-p)^(n_monday - n_correct_monday)) *
  (Nat.choose n_tuesday n_correct_tuesday * p^n_tuesday * (1-p)^(n_tuesday - n_correct_tuesday)) /
  (Nat.choose n_total n_correct * p^n_correct * (1-p)^(n_total - n_correct)) = 5 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_prediction_probability_l1245_124596


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1245_124571

/-- Given a triangle ABC where a + b = 10 and cos C is a root of 2x^2 - 3x - 2 = 0,
    prove that the minimum perimeter of the triangle is 10 + 5√3 -/
theorem min_perimeter_triangle (a b c : ℝ) (C : ℝ) :
  a + b = 10 →
  2 * (Real.cos C)^2 - 3 * (Real.cos C) - 2 = 0 →
  ∃ (p : ℝ), p = a + b + c ∧ p ≥ 10 + 5 * Real.sqrt 3 ∧
  ∀ (a' b' c' : ℝ), a' + b' = 10 →
    2 * (Real.cos C)^2 - 3 * (Real.cos C) - 2 = 0 →
    a' + b' + c' ≥ p :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1245_124571


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1245_124521

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1245_124521


namespace NUMINAMATH_CALUDE_pint_cost_is_eight_l1245_124541

/-- The cost of a pint of paint given the number of doors, cost of a gallon, and savings -/
def pint_cost (num_doors : ℕ) (gallon_cost : ℚ) (savings : ℚ) : ℚ :=
  (gallon_cost + savings) / num_doors

theorem pint_cost_is_eight :
  pint_cost 8 55 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_pint_cost_is_eight_l1245_124541


namespace NUMINAMATH_CALUDE_trapezoid_area_l1245_124508

theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 36) (h2 : inner_area = 4) :
  (outer_area - inner_area) / 3 = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1245_124508


namespace NUMINAMATH_CALUDE_camp_cedar_counselor_ratio_l1245_124515

theorem camp_cedar_counselor_ratio : 
  ∀ (num_boys : ℕ) (num_girls : ℕ) (num_counselors : ℕ),
    num_boys = 40 →
    num_girls = 3 * num_boys →
    num_counselors = 20 →
    (num_boys + num_girls) / num_counselors = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_camp_cedar_counselor_ratio_l1245_124515


namespace NUMINAMATH_CALUDE_brand_x_pen_price_l1245_124536

/-- The price of a brand X pen given the total number of pens, total cost, number of brand X pens, and price of brand Y pens. -/
theorem brand_x_pen_price
  (total_pens : ℕ)
  (total_cost : ℚ)
  (brand_x_count : ℕ)
  (brand_y_price : ℚ)
  (h1 : total_pens = 12)
  (h2 : total_cost = 42)
  (h3 : brand_x_count = 6)
  (h4 : brand_y_price = 2.2)
  : (total_cost - (total_pens - brand_x_count) * brand_y_price) / brand_x_count = 4.8 := by
  sorry

#check brand_x_pen_price

end NUMINAMATH_CALUDE_brand_x_pen_price_l1245_124536


namespace NUMINAMATH_CALUDE_tape_length_calculation_l1245_124520

/-- Calculates the length of tape wrapped around a cylindrical core -/
theorem tape_length_calculation 
  (initial_diameter : ℝ) 
  (tape_width : ℝ) 
  (num_wraps : ℕ) 
  (final_diameter : ℝ) 
  (h1 : initial_diameter = 4)
  (h2 : tape_width = 4)
  (h3 : num_wraps = 800)
  (h4 : final_diameter = 16) :
  (π / 2) * (initial_diameter + final_diameter) * num_wraps = 80 * π := by
  sorry

#check tape_length_calculation

end NUMINAMATH_CALUDE_tape_length_calculation_l1245_124520


namespace NUMINAMATH_CALUDE_waldo_total_time_l1245_124565

/-- The number of "Where's Waldo?" books -/
def num_books : ℕ := 15

/-- The number of puzzles per book -/
def puzzles_per_book : ℕ := 30

/-- The average time (in minutes) to find Waldo in a puzzle -/
def time_per_puzzle : ℕ := 3

/-- The total time (in minutes) to find Waldo in all puzzles across all books -/
def total_time : ℕ := num_books * puzzles_per_book * time_per_puzzle

theorem waldo_total_time : total_time = 1350 := by
  sorry

end NUMINAMATH_CALUDE_waldo_total_time_l1245_124565


namespace NUMINAMATH_CALUDE_intersection_line_polar_equation_l1245_124523

/-- Given two circles in polar coordinates, find the polar equation of the line
    passing through their intersection points. -/
theorem intersection_line_polar_equation
  (O₁ : ℝ → ℝ → Prop) -- Circle O₁ in polar coordinates
  (O₂ : ℝ → ℝ → Prop) -- Circle O₂ in polar coordinates
  (h₁ : ∀ ρ θ, O₁ ρ θ ↔ ρ = 2)
  (h₂ : ∀ ρ θ, O₂ ρ θ ↔ ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ - π/4) = 2) :
  ∀ ρ θ, (∃ θ₁ θ₂, O₁ ρ θ₁ ∧ O₂ ρ θ₂) →
    ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_polar_equation_l1245_124523


namespace NUMINAMATH_CALUDE_initial_persons_count_l1245_124595

/-- The number of persons initially present -/
def N : ℕ := sorry

/-- The initial average weight -/
def initial_average : ℝ := sorry

/-- The weight increase when the new person replaces one person -/
def weight_increase : ℝ := 4

/-- The weight of the person being replaced -/
def replaced_weight : ℝ := 65

/-- The weight of the new person -/
def new_weight : ℝ := 97

theorem initial_persons_count : N = 8 := by sorry

end NUMINAMATH_CALUDE_initial_persons_count_l1245_124595


namespace NUMINAMATH_CALUDE_ellipse_condition_l1245_124566

/-- The equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 represents an ellipse if and only if m ∈ (5, +∞) -/
theorem ellipse_condition (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) ↔ m > 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1245_124566


namespace NUMINAMATH_CALUDE_variance_linear_transform_l1245_124573

variable {α : Type*} [LinearOrderedField α]
variable (x : Finset ℕ → α)
variable (n : ℕ)

def variance (x : Finset ℕ → α) (n : ℕ) : α := sorry

theorem variance_linear_transform 
  (h : variance x n = 2) : 
  variance (fun i => 3 * x i + 2) n = 18 := by
  sorry

end NUMINAMATH_CALUDE_variance_linear_transform_l1245_124573


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_is_29_l1245_124529

/-- The polynomial p(x) = x^4 + 2x^2 + 5 -/
def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

/-- The remainder theorem: For a polynomial p(x) and a real number a,
    the remainder when p(x) is divided by (x - a) is equal to p(a) -/
theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a :=
sorry

theorem remainder_is_29 :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x + 29 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_is_29_l1245_124529


namespace NUMINAMATH_CALUDE_congruence_existence_no_solution_for_6_8_solution_exists_for_7_9_l1245_124532

theorem congruence_existence (A B : ℕ) : Prop :=
  ∃ C : ℕ, C % A = 1 ∧ C % B = 2

theorem no_solution_for_6_8 : ¬(congruence_existence 6 8) := by sorry

theorem solution_exists_for_7_9 : congruence_existence 7 9 := by sorry

end NUMINAMATH_CALUDE_congruence_existence_no_solution_for_6_8_solution_exists_for_7_9_l1245_124532


namespace NUMINAMATH_CALUDE_parallel_to_x_axis_symmetric_in_third_quadrant_equal_distance_to_axes_l1245_124524

-- Define point P
def P (x : ℝ) : ℝ × ℝ := (2*x - 3, 3 - x)

-- Define point A
def A : ℝ × ℝ := (-3, 4)

-- Theorem 1: If AP is parallel to x-axis, then P(-5, 4)
theorem parallel_to_x_axis :
  (∃ x : ℝ, P x = (-5, 4)) ↔ (∃ x : ℝ, (P x).2 = A.2) :=
sorry

-- Theorem 2: If symmetric point is in third quadrant, then x < 3/2
theorem symmetric_in_third_quadrant :
  (∃ x : ℝ, (2*x - 3 < 0 ∧ x - 3 < 0)) ↔ (∃ x : ℝ, x < 3/2) :=
sorry

-- Theorem 3: If distances to axes are equal, then P(1,1) or P(-3,3)
theorem equal_distance_to_axes :
  (∃ x : ℝ, |2*x - 3| = |3 - x|) ↔ (P 2 = (1, 1) ∨ P 0 = (-3, 3)) :=
sorry

end NUMINAMATH_CALUDE_parallel_to_x_axis_symmetric_in_third_quadrant_equal_distance_to_axes_l1245_124524


namespace NUMINAMATH_CALUDE_radio_sale_profit_percentage_l1245_124589

/-- Represents the problem of calculating profit percentage for a radio sale --/
theorem radio_sale_profit_percentage 
  (original_cost_usd : ℝ) 
  (exchange_rate : ℝ) 
  (discount_rate : ℝ) 
  (tax_rate : ℝ) 
  (final_price : ℝ) 
  (h1 : original_cost_usd = 110)
  (h2 : exchange_rate = 30)
  (h3 : discount_rate = 0.15)
  (h4 : tax_rate = 0.12)
  (h5 : final_price = 4830) :
  let original_cost_inr : ℝ := original_cost_usd * exchange_rate
  let selling_price_before_tax : ℝ := final_price / (1 + tax_rate)
  let profit : ℝ := selling_price_before_tax - original_cost_inr
  let profit_percentage : ℝ := (profit / original_cost_inr) * 100
  ∃ (ε : ℝ), abs (profit_percentage - 30.68) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_radio_sale_profit_percentage_l1245_124589


namespace NUMINAMATH_CALUDE_intersecting_line_equation_l1245_124562

/-- A line passing through a point and intersecting both axes -/
structure IntersectingLine where
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  passes_through_P : true  -- Placeholder for the condition that the line passes through P
  intersects_x_axis : A.2 = 0
  intersects_y_axis : B.1 = 0
  P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of the line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

theorem intersecting_line_equation (l : IntersectingLine) (eq : LineEquation) :
  l.P = (-4, 6) →
  eq.a = 3 ∧ eq.b = -2 ∧ eq.c = 24 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_line_equation_l1245_124562


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1245_124504

/-- Given a geometric sequence {a_n} with a₂ = 8 and a₅ = 64, prove that the common ratio q = 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 2 = 8 →                    -- Given condition
  a 5 = 64 →                   -- Given condition
  q = 2 := by                  -- Conclusion to prove
sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1245_124504


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_cost_of_dozen_pens_is_720_l1245_124564

/-- The cost of one dozen pens given the cost of 3 pens and 5 pencils and the ratio of pen to pencil cost -/
theorem cost_of_dozen_pens (total_cost : ℕ) (ratio_pen_pencil : ℕ) : ℕ :=
  let pen_cost := ratio_pen_pencil * (total_cost / (3 * ratio_pen_pencil + 5))
  12 * pen_cost

/-- Proof that the cost of one dozen pens is 720 given the conditions -/
theorem cost_of_dozen_pens_is_720 :
  cost_of_dozen_pens 240 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_cost_of_dozen_pens_is_720_l1245_124564


namespace NUMINAMATH_CALUDE_inequality_proof_l1245_124552

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1245_124552


namespace NUMINAMATH_CALUDE_equation_solution_l1245_124590

theorem equation_solution : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1245_124590


namespace NUMINAMATH_CALUDE_sequence_prime_divisor_l1245_124543

/-- Given a positive integer n > 1, prove that for all k ≥ 1, the k-th term of the sequence
    a_k = n^(n^(k-1)) - 1 has a prime divisor that does not divide any of the previous terms. -/
theorem sequence_prime_divisor (n : ℕ) (hn : n > 1) :
  ∀ k : ℕ, k ≥ 1 →
    ∃ p : ℕ, Nat.Prime p ∧ p ∣ (n^(n^(k-1)) - 1) ∧
      ∀ i : ℕ, 1 ≤ i ∧ i < k → ¬(p ∣ (n^(n^(i-1)) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_prime_divisor_l1245_124543


namespace NUMINAMATH_CALUDE_least_number_to_add_l1245_124591

theorem least_number_to_add (n : ℕ) : 
  (∀ m : ℕ, m < 7 → ¬((1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
  ((1789 + 7) % 6 = 0 ∧ (1789 + 7) % 4 = 0 ∧ (1789 + 7) % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_to_add_l1245_124591


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l1245_124559

theorem circle_line_distance_range (b : ℝ) :
  (∃! (p q : ℝ × ℝ), 
    ((p.1 - 1)^2 + (p.2 - 1)^2 = 4) ∧
    ((q.1 - 1)^2 + (q.2 - 1)^2 = 4) ∧
    (p ≠ q) ∧
    (|p.2 - (p.1 + b)| / Real.sqrt 2 = 1) ∧
    (|q.2 - (q.1 + b)| / Real.sqrt 2 = 1)) →
  b ∈ Set.Ioo (-3 * Real.sqrt 2) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) (3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l1245_124559


namespace NUMINAMATH_CALUDE_probability_theorem_l1245_124502

/-- Represents a club with members -/
structure Club where
  total : ℕ
  girls : ℕ
  boys : ℕ
  girls_under_18 : ℕ

/-- Calculates the probability of choosing two girls with at least one under 18 -/
def probability_two_girls_one_under_18 (club : Club) : ℚ :=
  let total_combinations := club.total.choose 2
  let girls_combinations := club.girls.choose 2
  let underaged_combinations := 
    club.girls_under_18 * (club.girls - club.girls_under_18) + club.girls_under_18.choose 2
  (underaged_combinations : ℚ) / total_combinations

/-- The main theorem to prove -/
theorem probability_theorem (club : Club) 
    (h1 : club.total = 15)
    (h2 : club.girls = 8)
    (h3 : club.boys = 7)
    (h4 : club.girls_under_18 = 3)
    (h5 : club.total = club.girls + club.boys) :
  probability_two_girls_one_under_18 club = 6/35 := by
  sorry

#eval probability_two_girls_one_under_18 ⟨15, 8, 7, 3⟩

end NUMINAMATH_CALUDE_probability_theorem_l1245_124502


namespace NUMINAMATH_CALUDE_largest_m_for_quadratic_function_l1245_124561

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem largest_m_for_quadratic_function (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x, ∀ y, f a b c y ≥ f a b c x) →
  (∃ x, f a b c x = 0) →
  (∃ m > 1, ∃ t, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) →
  (∀ m > 9, ¬∃ t, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_quadratic_function_l1245_124561


namespace NUMINAMATH_CALUDE_equation_solution_l1245_124500

theorem equation_solution : 
  ∃ (x : ℚ), (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1245_124500


namespace NUMINAMATH_CALUDE_negation_equivalence_l1245_124570

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 ∧ (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1245_124570


namespace NUMINAMATH_CALUDE_ribbon_distribution_l1245_124594

theorem ribbon_distribution (total_ribbon : ℚ) (num_boxes : ℕ) :
  total_ribbon = 2 / 5 →
  num_boxes = 5 →
  (total_ribbon / num_boxes : ℚ) = 2 / 25 := by
sorry

end NUMINAMATH_CALUDE_ribbon_distribution_l1245_124594


namespace NUMINAMATH_CALUDE_coprime_power_sum_not_divisible_by_11_l1245_124522

theorem coprime_power_sum_not_divisible_by_11 (a b : ℤ) (h : Int.gcd a b = 1) :
  ¬(11 ∣ (a^5 + 2*b^5)) ∧ ¬(11 ∣ (a^5 - 2*b^5)) := by
  sorry

end NUMINAMATH_CALUDE_coprime_power_sum_not_divisible_by_11_l1245_124522


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l1245_124576

/-- Quadratic function f(x) = -(x-2)² + h -/
def f (x h : ℝ) : ℝ := -(x - 2)^2 + h

theorem quadratic_points_ordering (h : ℝ) :
  let y₁ := f (-1/2) h
  let y₂ := f 1 h
  let y₃ := f 2 h
  y₁ < y₂ ∧ y₂ < y₃ := by sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l1245_124576


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1245_124513

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 6)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = Real.sqrt 132.525 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1245_124513


namespace NUMINAMATH_CALUDE_equal_roots_condition_l1245_124516

/-- 
For a quadratic equation ax^2 + bx + c = 0, 
the discriminant is defined as b^2 - 4ac
-/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- 
A quadratic equation has two equal real roots 
if and only if its discriminant is zero
-/
axiom equal_roots_iff_zero_discriminant (a b c : ℝ) : 
  a ≠ 0 → (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ (∀ y : ℝ, a*y^2 + b*y + c = 0 → y = x)) ↔ 
    discriminant a b c = 0

/-- 
For the quadratic equation x^2 + 6x + m = 0 to have two equal real roots, 
m must equal 9
-/
theorem equal_roots_condition : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ (∀ y : ℝ, y^2 + 6*y + m = 0 → y = x)) → m = 9 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l1245_124516


namespace NUMINAMATH_CALUDE_min_cuboids_for_cube_l1245_124582

def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

theorem min_cuboids_for_cube : 
  let cube_side := Nat.lcm (Nat.lcm cuboid_length cuboid_width) cuboid_height
  let cube_volume := cube_side ^ 3
  let cuboid_volume := cuboid_length * cuboid_width * cuboid_height
  cube_volume / cuboid_volume = 3600 := by
  sorry

end NUMINAMATH_CALUDE_min_cuboids_for_cube_l1245_124582


namespace NUMINAMATH_CALUDE_tournament_dominance_chain_l1245_124544

/-- Represents a round-robin tournament with 8 players -/
structure Tournament :=
  (players : Finset (Fin 8))
  (defeated : Fin 8 → Fin 8 → Prop)
  (round_robin : ∀ i j, i ≠ j → (defeated i j ∨ defeated j i))
  (asymmetric : ∀ i j, defeated i j → ¬ defeated j i)

/-- The main theorem to be proved -/
theorem tournament_dominance_chain (t : Tournament) :
  ∃ (a b c d : Fin 8),
    a ∈ t.players ∧ b ∈ t.players ∧ c ∈ t.players ∧ d ∈ t.players ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    t.defeated a b ∧ t.defeated a c ∧ t.defeated a d ∧
    t.defeated b c ∧ t.defeated b d ∧
    t.defeated c d :=
sorry

end NUMINAMATH_CALUDE_tournament_dominance_chain_l1245_124544


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1245_124568

theorem arithmetic_geometric_inequality 
  (a b c d h k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_h : 0 < h) (pos_k : 0 < k)
  (arith_prog : ∃ t : ℝ, t > 0 ∧ a = d + 3*t ∧ b = d + 2*t ∧ c = d + t)
  (geom_prog : ∃ r : ℝ, r > 1 ∧ a = d * r^3 ∧ h = d * r^2 ∧ k = d * r) :
  b * c > h * k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1245_124568


namespace NUMINAMATH_CALUDE_original_cost_equals_new_cost_l1245_124535

/-- Proves that the original manufacturing cost was equal to the new manufacturing cost
    when the profit percentage remains constant at 50% of the selling price. -/
theorem original_cost_equals_new_cost
  (selling_price : ℝ)
  (new_cost : ℝ)
  (h_profit_percentage : selling_price / 2 = selling_price - new_cost)
  (h_new_cost : new_cost = 50)
  : selling_price - (selling_price / 2) = new_cost :=
by sorry

end NUMINAMATH_CALUDE_original_cost_equals_new_cost_l1245_124535


namespace NUMINAMATH_CALUDE_square_of_modified_41_l1245_124567

theorem square_of_modified_41 (n : ℕ) :
  let modified_num := (5 * 10^n - 1) * 10^(n+1) + 1
  modified_num^2 = (10^(n+1) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_modified_41_l1245_124567


namespace NUMINAMATH_CALUDE_gcd_lcm_450_possibilities_l1245_124585

theorem gcd_lcm_450_possibilities (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 450) :
  ∃! s : Finset ℕ+, s.card = 8 ∧ ∀ x : ℕ+, x ∈ s ↔ ∃ a' b' : ℕ+, Nat.gcd a' b' * Nat.lcm a' b' = 450 ∧ Nat.gcd a' b' = x :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_450_possibilities_l1245_124585


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1245_124540

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1245_124540
