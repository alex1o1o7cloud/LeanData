import Mathlib

namespace NUMINAMATH_CALUDE_highest_throw_l1644_164460

def christine_first : ℕ := 20

def janice_first (christine_first : ℕ) : ℕ := christine_first - 4

def christine_second (christine_first : ℕ) : ℕ := christine_first + 10

def janice_second (janice_first : ℕ) : ℕ := janice_first * 2

def christine_third (christine_second : ℕ) : ℕ := christine_second + 4

def janice_third (christine_first : ℕ) : ℕ := christine_first + 17

theorem highest_throw :
  let c1 := christine_first
  let j1 := janice_first c1
  let c2 := christine_second c1
  let j2 := janice_second j1
  let c3 := christine_third c2
  let j3 := janice_third c1
  max c1 (max j1 (max c2 (max j2 (max c3 j3)))) = 37 := by sorry

end NUMINAMATH_CALUDE_highest_throw_l1644_164460


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1644_164494

theorem quadratic_expression_value : (3^2 : ℝ) - 3*3 + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1644_164494


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1644_164462

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as given in the problem
def z : ℂ := (1 + i) * i

-- Theorem statement
theorem imaginary_part_of_z :
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1644_164462


namespace NUMINAMATH_CALUDE_cosine_function_property_l1644_164450

/-- Given a cosine function with specific properties, prove that its angular frequency is 2. -/
theorem cosine_function_property (f : ℝ → ℝ) (ω φ : ℝ) (h_ω_pos : ω > 0) (h_φ_bound : |φ| ≤ π/2) 
  (h_f_def : ∀ x, f x = Real.sqrt 2 * Real.cos (ω * x + φ)) 
  (h_product : ∃ x₁ x₂ : ℝ, f x₁ * f x₂ = -2)
  (h_min_diff : ∃ x₁ x₂ : ℝ, f x₁ * f x₂ = -2 ∧ |x₁ - x₂| = π/2) : ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_function_property_l1644_164450


namespace NUMINAMATH_CALUDE_circle_existence_l1644_164444

-- Define the lines and the given circle
def line1 (x y : ℝ) : Prop := x + y = 7
def line2 (x y : ℝ) : Prop := x - 7*y = -33
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 28*x + 6*y + 165 = 0

-- Define the distance ratio condition
def distance_ratio (x y u v : ℝ) : Prop :=
  |x + y - 7| / Real.sqrt 2 = 5 * |x - 7*y + 33| / Real.sqrt 50

-- Define the intersection point of the two lines
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the orthogonality condition
def orthogonal_intersection (x y u v r : ℝ) : Prop :=
  (u - 14)^2 + (v + 3)^2 = r^2 + 40

-- Define the two resulting circles
def circle1 (x y : ℝ) : Prop := (x - 11)^2 + (y - 8)^2 = 87
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 168

theorem circle_existence :
  ∃ (x y u₁ v₁ u₂ v₂ : ℝ),
    (∀ (a b : ℝ), intersection_point a b → (circle1 a b ∨ circle2 a b)) ∧
    distance_ratio u₁ v₁ u₁ v₁ ∧
    distance_ratio u₂ v₂ u₂ v₂ ∧
    orthogonal_intersection u₁ v₁ u₁ v₁ (Real.sqrt 87) ∧
    orthogonal_intersection u₂ v₂ u₂ v₂ (Real.sqrt 168) :=
  sorry


end NUMINAMATH_CALUDE_circle_existence_l1644_164444


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1644_164408

/-- Given two positive integers with LCM 36 and ratio 2:3, prove their sum is 30 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 36)
  (h_ratio : a * 3 = b * 2) : 
  a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1644_164408


namespace NUMINAMATH_CALUDE_function_inequality_l1644_164449

theorem function_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, 2 * a * x^2 - a * x > 3 - a) →
  a > 24/7 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1644_164449


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1644_164438

theorem intersection_point_of_lines (x y : ℚ) : 
  (3 * y = -2 * x + 6 ∧ -2 * y = 7 * x + 4) ↔ (x = -24/17 ∧ y = 50/17) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1644_164438


namespace NUMINAMATH_CALUDE_inequality_range_l1644_164482

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x + 1| > a) → a < 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1644_164482


namespace NUMINAMATH_CALUDE_binomial_10_3_l1644_164454

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l1644_164454


namespace NUMINAMATH_CALUDE_crushing_load_square_pillars_l1644_164404

theorem crushing_load_square_pillars (T H : ℝ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / H^3 = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_square_pillars_l1644_164404


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1644_164419

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (3 * X ^ 3 - 2 * X ^ 2 - 23 * X + 60) = (X - 6) * q + (-378) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1644_164419


namespace NUMINAMATH_CALUDE_brendas_blisters_l1644_164477

theorem brendas_blisters (blisters_per_arm : ℕ) : 
  (2 * blisters_per_arm + 80 = 200) → blisters_per_arm = 60 := by
  sorry

end NUMINAMATH_CALUDE_brendas_blisters_l1644_164477


namespace NUMINAMATH_CALUDE_power_division_rule_l1644_164406

theorem power_division_rule (a : ℝ) (ha : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1644_164406


namespace NUMINAMATH_CALUDE_initial_saree_purchase_l1644_164489

/-- The number of sarees in the initial purchase -/
def num_sarees : ℕ := 2

/-- The price of one saree -/
def saree_price : ℕ := 400

/-- The price of one shirt -/
def shirt_price : ℕ := 200

/-- Theorem stating that the number of sarees in the initial purchase is 2 -/
theorem initial_saree_purchase : 
  (∃ (X : ℕ), X * saree_price + 4 * shirt_price = 1600) ∧ 
  (saree_price + 6 * shirt_price = 1600) ∧
  (12 * shirt_price = 2400) →
  num_sarees = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_saree_purchase_l1644_164489


namespace NUMINAMATH_CALUDE_negation_equivalence_l1644_164490

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1644_164490


namespace NUMINAMATH_CALUDE_prob_A_not_lose_l1644_164443

-- Define the probabilities
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Define the property of mutually exclusive events
def mutually_exclusive (p q : ℝ) : Prop := p + q ≤ 1

-- State the theorem
theorem prob_A_not_lose : 
  mutually_exclusive prob_A_win prob_draw →
  prob_A_win + prob_draw = 0.8 :=
by
  sorry

end NUMINAMATH_CALUDE_prob_A_not_lose_l1644_164443


namespace NUMINAMATH_CALUDE_scientific_notation_439000_l1644_164409

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_439000 :
  toScientificNotation 439000 = ScientificNotation.mk 4.39 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_439000_l1644_164409


namespace NUMINAMATH_CALUDE_even_count_pascal_triangle_l1644_164470

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool := sorry

/-- Count even binomial coefficients in a single row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ := sorry

/-- Count even binomial coefficients in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ := sorry

/-- The number of even integers in the top 15 rows of Pascal's Triangle is 84 -/
theorem even_count_pascal_triangle : countEvenInTriangle 15 = 84 := by sorry

end NUMINAMATH_CALUDE_even_count_pascal_triangle_l1644_164470


namespace NUMINAMATH_CALUDE_june_bike_ride_l1644_164448

/-- June's bike ride problem -/
theorem june_bike_ride (june_distance : ℝ) (june_time : ℝ) (bernard_distance : ℝ) (bernard_time : ℝ) (june_to_bernard : ℝ) :
  june_distance = 2 →
  june_time = 6 →
  bernard_distance = 5 →
  bernard_time = 15 →
  june_to_bernard = 7 →
  (june_to_bernard / (june_distance / june_time)) = 21 := by
sorry

end NUMINAMATH_CALUDE_june_bike_ride_l1644_164448


namespace NUMINAMATH_CALUDE_vector_magnitude_l1644_164488

/-- Given two vectors a and b in ℝ², prove that |b| = √3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (let angle := Real.pi / 3
   let a_x := 1
   let a_y := Real.sqrt 2
   a = (a_x, a_y) ∧ 
   Real.cos angle = a.1 * b.1 + a.2 * b.2 / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) ∧
   (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0)) →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1644_164488


namespace NUMINAMATH_CALUDE_smallest_b_is_correct_l1644_164435

/-- N(b) is the number of natural numbers a for which x^2 + ax + b = 0 has integer roots -/
def N (b : ℕ) : ℕ := sorry

/-- The smallest value of b for which N(b) = 20 -/
def smallest_b : ℕ := 240

theorem smallest_b_is_correct :
  (N smallest_b = 20) ∧ (∀ b : ℕ, b < smallest_b → N b ≠ 20) := by sorry

end NUMINAMATH_CALUDE_smallest_b_is_correct_l1644_164435


namespace NUMINAMATH_CALUDE_cube_circumscribed_sphere_volume_l1644_164491

theorem cube_circumscribed_sphere_volume (surface_area : ℝ) (h : surface_area = 24) :
  let edge_length := Real.sqrt (surface_area / 6)
  let sphere_radius := edge_length * Real.sqrt 3 / 2
  (4 / 3) * Real.pi * sphere_radius ^ 3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_circumscribed_sphere_volume_l1644_164491


namespace NUMINAMATH_CALUDE_rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power_l1644_164458

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.choose n k)

def rational_coefficient_sum (n : ℕ) : ℕ :=
  2 * (binomial_coefficient n 2) + (binomial_coefficient n n)

theorem rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power :
  rational_coefficient_sum 5 = 21 := by sorry

end NUMINAMATH_CALUDE_rational_coefficient_sum_for_cube_root_two_plus_x_fifth_power_l1644_164458


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l1644_164485

theorem opposite_of_negative_half : -(-(1/2 : ℚ)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l1644_164485


namespace NUMINAMATH_CALUDE_subtracted_value_problem_solution_l1644_164412

theorem subtracted_value (chosen_number : ℕ) (final_answer : ℕ) : ℕ :=
  let divided_result := chosen_number / 8
  divided_result - final_answer

theorem problem_solution :
  subtracted_value 1376 12 = 160 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_problem_solution_l1644_164412


namespace NUMINAMATH_CALUDE_quadratic_root_values_l1644_164483

theorem quadratic_root_values : 
  (Real.sqrt (9 - 8 * 0) = 3) ∧ 
  (Real.sqrt (9 - 8 * (1/2)) = Real.sqrt 5) ∧ 
  (Real.sqrt (9 - 8 * (-2)) = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_values_l1644_164483


namespace NUMINAMATH_CALUDE_student_sums_proof_l1644_164400

def total_sums (right_sums wrong_sums : ℕ) : ℕ :=
  right_sums + wrong_sums

theorem student_sums_proof (right_sums : ℕ) 
  (h1 : right_sums = 18) 
  (h2 : ∃ wrong_sums : ℕ, wrong_sums = 2 * right_sums) : 
  total_sums right_sums (2 * right_sums) = 54 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_proof_l1644_164400


namespace NUMINAMATH_CALUDE_f_composition_equals_9184_l1644_164441

/-- The function f(x) = 3x^2 + 2x - 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

/-- Theorem: f(f(f(1))) = 9184 -/
theorem f_composition_equals_9184 : f (f (f 1)) = 9184 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_9184_l1644_164441


namespace NUMINAMATH_CALUDE_factorial_ratio_l1644_164446

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 2 * Nat.factorial 1) = 360 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1644_164446


namespace NUMINAMATH_CALUDE_phi_product_of_primes_exists_primes_satisfying_equation_verify_p_q_solution_l1644_164471

-- Define φ(n) as the number of natural numbers less than n that are coprime to n
def phi (n : ℕ) : ℕ := Nat.totient n

-- Define the property of being prime
def isPrime (p : ℕ) : Prop := Nat.Prime p

-- Theorem 1: For distinct primes p and q, φ(pq) = (p-1)(q-1)
theorem phi_product_of_primes (p q : ℕ) (hp : isPrime p) (hq : isPrime q) (hpq : p ≠ q) :
  phi (p * q) = (p - 1) * (q - 1) := by sorry

-- Theorem 2: There exist prime numbers p and q such that φ(pq) = 3p + q
theorem exists_primes_satisfying_equation :
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ phi (p * q) = 3 * p + q := by sorry

-- Find the values of p and q
def find_p_q : ℕ × ℕ := (3, 11)

-- Verify that the found values satisfy the equation
theorem verify_p_q_solution :
  let (p, q) := find_p_q
  isPrime p ∧ isPrime q ∧ p ≠ q ∧ phi (p * q) = 3 * p + q := by sorry

end NUMINAMATH_CALUDE_phi_product_of_primes_exists_primes_satisfying_equation_verify_p_q_solution_l1644_164471


namespace NUMINAMATH_CALUDE_dog_barking_problem_l1644_164466

/-- Given the barking patterns of two dogs and the owner's hushing behavior, 
    calculate the number of times the owner said "hush". -/
theorem dog_barking_problem (poodle_barks terrier_barks owner_hushes : ℕ) : 
  poodle_barks = 24 →
  poodle_barks = 2 * terrier_barks →
  owner_hushes * 2 = terrier_barks →
  owner_hushes = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_barking_problem_l1644_164466


namespace NUMINAMATH_CALUDE_inequality_statements_l1644_164411

theorem inequality_statements (a b c : ℝ) :
  (a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2)) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_statements_l1644_164411


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1644_164434

theorem quadratic_transformation (a b c : ℝ) :
  (∃ (m q : ℝ), ∀ x, ax^2 + bx + c = 5*(x - 3)^2 + 15) →
  (∃ (m p q : ℝ), ∀ x, 4*ax^2 + 4*bx + 4*c = m*(x - p)^2 + q ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1644_164434


namespace NUMINAMATH_CALUDE_mountain_climbing_speed_l1644_164475

theorem mountain_climbing_speed 
  (total_time : ℝ) 
  (break_day1 : ℝ) 
  (break_day2 : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 14) 
  (h2 : break_day1 = 0.5) 
  (h3 : break_day2 = 0.75) 
  (h4 : speed_difference = 0.5) 
  (h5 : time_difference = 2) 
  (h6 : total_distance = 52) : 
  ∃ (speed_day1 : ℝ), 
    speed_day1 + speed_difference = 4.375 ∧ 
    (∃ (time_day1 : ℝ), 
      time_day1 + (time_day1 - time_difference) = total_time ∧
      speed_day1 * (time_day1 - break_day1) + 
      (speed_day1 + speed_difference) * (time_day1 - time_difference - break_day2) = total_distance) :=
by sorry

end NUMINAMATH_CALUDE_mountain_climbing_speed_l1644_164475


namespace NUMINAMATH_CALUDE_no_natural_solution_l1644_164432

theorem no_natural_solution : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1644_164432


namespace NUMINAMATH_CALUDE_circle_equation_l1644_164427

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (h : ℝ), (p.1 - h)^2 + p.2^2 = (h - 1)^2 + 1^2}

-- Define points A and B
def point_A : ℝ × ℝ := (5, 2)
def point_B : ℝ × ℝ := (-1, 4)

-- Theorem statement
theorem circle_equation :
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + p.2^2 = 20) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  (∃ h : ℝ, ∀ p : ℝ × ℝ, p ∈ circle_C → p.2 = 0 → p.1 = h) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1644_164427


namespace NUMINAMATH_CALUDE_consecutive_product_prime_factors_l1644_164480

theorem consecutive_product_prime_factors (n : ℕ) (hn : n ≥ 1) :
  ∃ x : ℕ+, ∃ p : Fin n → ℕ, 
    (∀ i : Fin n, Prime (p i)) ∧ 
    (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
    (∀ i : Fin n, (p i) ∣ (x * (x + 1) + 1)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_product_prime_factors_l1644_164480


namespace NUMINAMATH_CALUDE_coefficient_of_degree_10_l1644_164499

-- Define the degree of the term nxy^n
def degree (n : ℕ) : ℕ := 1 + n

-- State the theorem
theorem coefficient_of_degree_10 (n : ℕ) : degree n = 10 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_degree_10_l1644_164499


namespace NUMINAMATH_CALUDE_compound_interest_proof_l1644_164401

/-- The compound interest rate that turns $1200 into $1348.32 in 2 years with annual compounding -/
def compound_interest_rate : ℝ :=
  0.06

theorem compound_interest_proof (initial_sum final_sum : ℝ) (years : ℕ) :
  initial_sum = 1200 →
  final_sum = 1348.32 →
  years = 2 →
  final_sum = initial_sum * (1 + compound_interest_rate) ^ years :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l1644_164401


namespace NUMINAMATH_CALUDE_art_book_cost_l1644_164437

theorem art_book_cost (total_cost : ℕ) (num_math num_art num_science : ℕ) (cost_math cost_science : ℕ) :
  total_cost = 30 ∧
  num_math = 2 ∧
  num_art = 3 ∧
  num_science = 6 ∧
  cost_math = 3 ∧
  cost_science = 3 →
  (total_cost - num_math * cost_math - num_science * cost_science) / num_art = 2 :=
by sorry

end NUMINAMATH_CALUDE_art_book_cost_l1644_164437


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l1644_164484

/-- A figure composed of unit squares arranged in a specific pattern -/
structure UnitSquareFigure where
  horizontalSegments : ℕ
  verticalSegments : ℕ

/-- The perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : ℕ :=
  figure.horizontalSegments + figure.verticalSegments

/-- The specific figure from the problem -/
def specificFigure : UnitSquareFigure :=
  { horizontalSegments := 16, verticalSegments := 10 }

/-- Theorem stating that the perimeter of the specific figure is 26 -/
theorem specific_figure_perimeter :
  perimeter specificFigure = 26 := by sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l1644_164484


namespace NUMINAMATH_CALUDE_tan_five_pi_four_l1644_164403

theorem tan_five_pi_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_four_l1644_164403


namespace NUMINAMATH_CALUDE_point_line_distance_l1644_164445

/-- A type representing points on a line -/
structure Point where
  x : ℝ

/-- Distance between two points -/
def dist (p q : Point) : ℝ := |p.x - q.x|

theorem point_line_distance (A : Fin 11 → Point) :
  (dist (A 0) (A 10) = 56) →
  (∀ i, i < 9 → dist (A i) (A (i + 2)) ≤ 12) →
  (∀ j, j < 8 → dist (A j) (A (j + 3)) ≥ 17) →
  dist (A 1) (A 6) = 29 := by
sorry

end NUMINAMATH_CALUDE_point_line_distance_l1644_164445


namespace NUMINAMATH_CALUDE_investment_interest_theorem_l1644_164497

/-- Calculates the total interest earned from two investments -/
def totalInterest (totalAmount : ℝ) (amount1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount2 := totalAmount - amount1
  let interest1 := amount1 * rate1
  let interest2 := amount2 * rate2
  interest1 + interest2

/-- Proves that investing $9000 with $4000 at 8% and the rest at 9% yields $770 in interest -/
theorem investment_interest_theorem :
  totalInterest 9000 4000 0.08 0.09 = 770 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_theorem_l1644_164497


namespace NUMINAMATH_CALUDE_dans_candy_bar_cost_l1644_164492

/-- The cost of each candy bar given Dan's purchase scenario -/
def candy_bar_cost (initial_amount : ℚ) (num_candy_bars : ℕ) (amount_left : ℚ) : ℚ :=
  (initial_amount - amount_left) / num_candy_bars

/-- Theorem stating that the cost of each candy bar in Dan's scenario is $3 ÷ 99 -/
theorem dans_candy_bar_cost :
  candy_bar_cost 4 99 1 = 3 / 99 := by
  sorry

end NUMINAMATH_CALUDE_dans_candy_bar_cost_l1644_164492


namespace NUMINAMATH_CALUDE_inequality_proof_l1644_164407

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (1 - 2*x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2*y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2*z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1644_164407


namespace NUMINAMATH_CALUDE_hamburger_meat_price_per_pound_l1644_164426

/-- Given the following grocery items and their prices:
    - 2 pounds of hamburger meat (price unknown)
    - 1 pack of hamburger buns for $1.50
    - A head of lettuce for $1.00
    - A 1.5-pound tomato priced at $2.00 per pound
    - A jar of pickles that cost $2.50 with a $1.00 off coupon
    And given that Lauren paid with a $20 bill and got $6 change back,
    prove that the price per pound of hamburger meat is $3.50. -/
theorem hamburger_meat_price_per_pound
  (hamburger_meat_weight : ℝ)
  (buns_price : ℝ)
  (lettuce_price : ℝ)
  (tomato_weight : ℝ)
  (tomato_price_per_pound : ℝ)
  (pickles_price : ℝ)
  (pickles_discount : ℝ)
  (paid_amount : ℝ)
  (change_amount : ℝ)
  (h1 : hamburger_meat_weight = 2)
  (h2 : buns_price = 1.5)
  (h3 : lettuce_price = 1)
  (h4 : tomato_weight = 1.5)
  (h5 : tomato_price_per_pound = 2)
  (h6 : pickles_price = 2.5)
  (h7 : pickles_discount = 1)
  (h8 : paid_amount = 20)
  (h9 : change_amount = 6) :
  (paid_amount - change_amount - (buns_price + lettuce_price + tomato_weight * tomato_price_per_pound + pickles_price - pickles_discount)) / hamburger_meat_weight = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_meat_price_per_pound_l1644_164426


namespace NUMINAMATH_CALUDE_car_braking_distance_l1644_164431

/-- Represents the distance traveled by a car during braking -/
def distance_traveled (initial_speed : ℕ) (deceleration : ℕ) : ℕ :=
  let stopping_time := initial_speed / deceleration
  (initial_speed * stopping_time) - (deceleration * stopping_time * (stopping_time - 1) / 2)

/-- Theorem: A car with initial speed 40 ft/s and deceleration 10 ft/s² travels 100 ft before stopping -/
theorem car_braking_distance :
  distance_traveled 40 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_car_braking_distance_l1644_164431


namespace NUMINAMATH_CALUDE_cards_given_to_miguel_miguel_received_13_cards_l1644_164463

theorem cards_given_to_miguel (initial_cards : ℕ) (kept_cards : ℕ) (friends : ℕ) (cards_per_friend : ℕ) (sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  by
  -- Define the conditions
  have h1 : initial_cards = 130 := by sorry
  have h2 : kept_cards = 15 := by sorry
  have h3 : friends = 8 := by sorry
  have h4 : cards_per_friend = 12 := by sorry
  have h5 : sisters = 2 := by sorry
  have h6 : cards_per_sister = 3 := by sorry

  -- Calculate the number of cards given to Miguel
  let cards_to_give := initial_cards - kept_cards
  let cards_to_friends := friends * cards_per_friend
  let cards_left_after_friends := cards_to_give - cards_to_friends
  let cards_to_sisters := sisters * cards_per_sister
  let cards_to_miguel := cards_left_after_friends - cards_to_sisters

  -- Prove that cards_to_miguel = 13
  sorry

-- State the theorem
theorem miguel_received_13_cards : cards_given_to_miguel 130 15 8 12 2 3 = 13 := by sorry

end NUMINAMATH_CALUDE_cards_given_to_miguel_miguel_received_13_cards_l1644_164463


namespace NUMINAMATH_CALUDE_digit_40000_is_1_l1644_164472

/-- The sequence of digits formed by concatenating natural numbers -/
def digit_sequence : ℕ → ℕ := sorry

/-- The 40,000th digit in the sequence -/
def digit_40000 : ℕ := digit_sequence 40000

/-- Theorem: The 40,000th digit in the sequence is 1 -/
theorem digit_40000_is_1 : digit_40000 = 1 := by sorry

end NUMINAMATH_CALUDE_digit_40000_is_1_l1644_164472


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l1644_164478

theorem perpendicular_vectors_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x - 1, -x]
  (a 0 * b 0 + a 1 * b 1 = 0) → 
  Real.sqrt ((a 0 + b 0)^2 + (a 1 + b 1)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l1644_164478


namespace NUMINAMATH_CALUDE_sum_of_integers_l1644_164486

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c - e = 7)
  (eq2 : b - c + d + e = 9)
  (eq3 : c - d + a - e = 5)
  (eq4 : d - a + b + e = 1) :
  a + b + c + d + e = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1644_164486


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1644_164455

theorem two_digit_number_property (a b k : ℕ) : 
  (a ≥ 1 ∧ a ≤ 9) →  -- a is a single digit (tens place)
  (b ≥ 0 ∧ b ≤ 9) →  -- b is a single digit (ones place)
  (10 * a + b = k * (a + b)) →  -- original number condition
  (10 * b + a = (13 - k) * (a + b)) →  -- interchanged digits condition
  k = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1644_164455


namespace NUMINAMATH_CALUDE_average_weight_solution_l1644_164442

def average_weight_problem (a b c : ℝ) : Prop :=
  let avg_abc := (a + b + c) / 3
  let avg_ab := (a + b) / 2
  (avg_abc = 45) ∧ (avg_ab = 40) ∧ (b = 31) → ((b + c) / 2 = 43)

theorem average_weight_solution :
  ∀ a b c : ℝ, average_weight_problem a b c :=
by
  sorry

end NUMINAMATH_CALUDE_average_weight_solution_l1644_164442


namespace NUMINAMATH_CALUDE_count_sets_with_seven_l1644_164459

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem count_sets_with_seven :
  (Finset.filter (fun s : Finset ℕ => 
    s.card = 3 ∧ 
    (∀ x ∈ s, x ∈ S) ∧ 
    (s.sum id = 21) ∧ 
    (7 ∈ s))
  (Finset.powerset S)).card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_sets_with_seven_l1644_164459


namespace NUMINAMATH_CALUDE_custom_op_problem_l1644_164469

/-- The custom operation @ defined as a @ b = a × (a + 1) × ... × (a + b - 1) -/
def custom_op (a b : ℕ) : ℕ := 
  (List.range b).foldl (fun acc i => acc * (a + i)) a

/-- Theorem stating that if x @ y @ 2 = 420, then y @ x = 20 -/
theorem custom_op_problem (x y : ℕ) : 
  custom_op x (custom_op y 2) = 420 → custom_op y x = 20 := by
  sorry

#check custom_op_problem

end NUMINAMATH_CALUDE_custom_op_problem_l1644_164469


namespace NUMINAMATH_CALUDE_smallest_w_l1644_164413

theorem smallest_w (w : ℕ+) (h1 : (2^5 : ℕ) ∣ (936 * w))
                            (h2 : (3^3 : ℕ) ∣ (936 * w))
                            (h3 : (12^2 : ℕ) ∣ (936 * w)) :
  w ≥ 36 ∧ (∃ (v : ℕ+), v ≥ 36 → 
    (2^5 : ℕ) ∣ (936 * v) ∧ 
    (3^3 : ℕ) ∣ (936 * v) ∧ 
    (12^2 : ℕ) ∣ (936 * v)) :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l1644_164413


namespace NUMINAMATH_CALUDE_equal_candies_after_sharing_l1644_164423

/-- The number of candies Minyoung and Taehyung should have to be equal -/
def target_candies (total_candies : ℕ) : ℕ :=
  total_candies / 2

/-- The number of candies Taehyung should take from Minyoung -/
def candies_to_take (minyoung_candies taehyung_candies : ℕ) : ℕ :=
  (minyoung_candies + taehyung_candies) / 2 - taehyung_candies

theorem equal_candies_after_sharing 
  (minyoung_initial : ℕ) 
  (taehyung_initial : ℕ) 
  (h1 : minyoung_initial = 9) 
  (h2 : taehyung_initial = 3) :
  let candies_taken := candies_to_take minyoung_initial taehyung_initial
  minyoung_initial - candies_taken = taehyung_initial + candies_taken ∧
  candies_taken = 3 :=
by sorry

#eval candies_to_take 9 3

end NUMINAMATH_CALUDE_equal_candies_after_sharing_l1644_164423


namespace NUMINAMATH_CALUDE_inverse_function_problem_l1644_164440

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, g x = (Function.invFun (f c d) x) - 5) →
  (Function.invFun (f c d) = Function.invFun (f c d)) →
  7 * c + 3 * d = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l1644_164440


namespace NUMINAMATH_CALUDE_elena_book_purchase_l1644_164428

theorem elena_book_purchase (M : ℝ) (h : M > 0) : 
  let B := 2 * (1/3 * M)
  M - B = 1/3 * M :=
by
  sorry

end NUMINAMATH_CALUDE_elena_book_purchase_l1644_164428


namespace NUMINAMATH_CALUDE_sticker_collection_value_l1644_164402

theorem sticker_collection_value (total_stickers : ℕ) (sample_size : ℕ) (sample_value : ℕ) 
  (h1 : total_stickers = 18)
  (h2 : sample_size = 6)
  (h3 : sample_value = 24) :
  (total_stickers : ℚ) * (sample_value : ℚ) / (sample_size : ℚ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_sticker_collection_value_l1644_164402


namespace NUMINAMATH_CALUDE_inequality_chain_l1644_164473

theorem inequality_chain (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l1644_164473


namespace NUMINAMATH_CALUDE_range_of_m_l1644_164498

open Set

/-- Proposition p: There exists a real x such that x^2 + m < 0 -/
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m < 0

/-- Proposition q: For all real x, x^2 + mx + 1 > 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

/-- The set of real numbers m that satisfy the given conditions -/
def M : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

/-- The theorem stating the range of m -/
theorem range_of_m : M = Iic (-2) ∪ Ico 0 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1644_164498


namespace NUMINAMATH_CALUDE_fraction_inequality_l1644_164418

theorem fraction_inequality (a b c x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) : 
  min (x / (a*b + a*c)) (y / (a*c + b*c)) < (x + y) / (a*b + b*c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1644_164418


namespace NUMINAMATH_CALUDE_hotdog_cost_l1644_164425

/-- Represents the cost of Sara's lunch items in dollars -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Theorem stating that given the total lunch cost and salad cost, the hotdog cost can be determined -/
theorem hotdog_cost (lunch : LunchCost) 
  (h1 : lunch.total = 10.46)
  (h2 : lunch.salad = 5.1)
  (h3 : lunch.total = lunch.salad + lunch.hotdog) : 
  lunch.hotdog = 5.36 := by
  sorry

#check hotdog_cost

end NUMINAMATH_CALUDE_hotdog_cost_l1644_164425


namespace NUMINAMATH_CALUDE_mask_assignment_unique_l1644_164481

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

structure MaskAssignment where
  elephant : ℕ
  mouse : ℕ
  pig : ℕ
  panda : ℕ

def valid_assignment (a : MaskAssignment) : Prop :=
  is_single_digit a.elephant ∧
  is_single_digit a.mouse ∧
  is_single_digit a.pig ∧
  is_single_digit a.panda ∧
  a.elephant ≠ a.mouse ∧
  a.elephant ≠ a.pig ∧
  a.elephant ≠ a.panda ∧
  a.mouse ≠ a.pig ∧
  a.mouse ≠ a.panda ∧
  a.pig ≠ a.panda ∧
  (a.mouse * a.mouse) % 10 = a.elephant ∧
  (a.elephant * a.elephant) ≥ 10 ∧ (a.elephant * a.elephant) ≤ 99 ∧
  (a.mouse * a.mouse) ≥ 10 ∧ (a.mouse * a.mouse) ≤ 99 ∧
  (a.pig * a.pig) ≥ 10 ∧ (a.pig * a.pig) ≤ 99 ∧
  (a.panda * a.panda) ≥ 10 ∧ (a.panda * a.panda) ≤ 99 ∧
  (a.elephant * a.elephant) % 10 ≠ (a.mouse * a.mouse) % 10 ∧
  (a.elephant * a.elephant) % 10 ≠ (a.pig * a.pig) % 10 ∧
  (a.elephant * a.elephant) % 10 ≠ (a.panda * a.panda) % 10 ∧
  (a.mouse * a.mouse) % 10 ≠ (a.pig * a.pig) % 10 ∧
  (a.mouse * a.mouse) % 10 ≠ (a.panda * a.panda) % 10 ∧
  (a.pig * a.pig) % 10 ≠ (a.panda * a.panda) % 10

theorem mask_assignment_unique :
  ∃! a : MaskAssignment, valid_assignment a ∧ 
    a.elephant = 6 ∧ a.mouse = 4 ∧ a.pig = 8 ∧ a.panda = 1 :=
sorry

end NUMINAMATH_CALUDE_mask_assignment_unique_l1644_164481


namespace NUMINAMATH_CALUDE_square_area_l1644_164436

/-- A square with a circle tangent to three sides and passing through the diagonal midpoint -/
structure SquareWithCircle where
  s : ℝ  -- side length of the square
  r : ℝ  -- radius of the circle
  s_pos : 0 < s  -- side length is positive
  r_pos : 0 < r  -- radius is positive
  tangent_condition : s = 4 * r  -- derived from the tangent and midpoint conditions

/-- The area of the square is 16r^2 -/
theorem square_area (config : SquareWithCircle) : config.s^2 = 16 * config.r^2 := by
  sorry

#check square_area

end NUMINAMATH_CALUDE_square_area_l1644_164436


namespace NUMINAMATH_CALUDE_shortest_halving_segment_345_triangle_l1644_164493

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The shortest segment that halves the area of a triangle -/
def shortestHalvingSegment (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The shortest segment that halves the area of a 3-4-5 triangle has length 2 -/
theorem shortest_halving_segment_345_triangle :
  let t : Triangle := { a := 3, b := 4, c := 5 }
  shortestHalvingSegment t = 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_halving_segment_345_triangle_l1644_164493


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1644_164474

theorem trig_equation_solution (x : ℝ) : 
  2 * Real.cos x + Real.cos (3 * x) + Real.cos (5 * x) = 0 →
  ∃ (n : ℤ), x = n * (π / 4) ∧ n % 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1644_164474


namespace NUMINAMATH_CALUDE_intern_distribution_theorem_l1644_164447

def distribute_interns (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem intern_distribution_theorem :
  distribute_interns 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_intern_distribution_theorem_l1644_164447


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1644_164456

open Set

def A : Set ℝ := {x | 1 < x^2 ∧ x^2 < 4}
def B : Set ℝ := {x | x - 1 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1644_164456


namespace NUMINAMATH_CALUDE_fraction_simplification_l1644_164420

theorem fraction_simplification {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a^(2*b) * b^(3*a)) / (b^(2*b) * a^(3*a)) = (a/b)^(2*b - 3*a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1644_164420


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1644_164439

theorem quadratic_form_equivalence (x : ℝ) : 
  (2*x - 1) * (x + 2) + 1 = 2*(x + 3/4)^2 - 17/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1644_164439


namespace NUMINAMATH_CALUDE_order_of_magnitudes_l1644_164421

-- Define the function f(x) = ln(x) - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- Define a, b, and c
noncomputable def a : ℝ := f (3/2)
noncomputable def b : ℝ := f Real.pi
noncomputable def c : ℝ := f 3

-- State the theorem
theorem order_of_magnitudes (h1 : 3/2 < 3) (h2 : 3 < Real.pi) : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitudes_l1644_164421


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1644_164405

theorem algebraic_expression_value (x y : ℝ) : 
  x - 2*y + 1 = 3 → 2*x - 4*y + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1644_164405


namespace NUMINAMATH_CALUDE_abie_initial_bags_l1644_164468

/-- The number of bags of chips Abie initially had -/
def initial_bags : ℕ := sorry

/-- The number of bags Abie gave away -/
def bags_given_away : ℕ := 4

/-- The number of bags Abie bought -/
def bags_bought : ℕ := 6

/-- The final number of bags Abie has -/
def final_bags : ℕ := 22

/-- Theorem stating that Abie initially had 20 bags of chips -/
theorem abie_initial_bags : 
  initial_bags = 20 ∧ 
  initial_bags - bags_given_away + bags_bought = final_bags :=
sorry

end NUMINAMATH_CALUDE_abie_initial_bags_l1644_164468


namespace NUMINAMATH_CALUDE_no_winning_strategy_l1644_164457

-- Define the game board
def GameBoard := Fin 99

-- Define a piece
structure Piece where
  number : Fin 99
  position : GameBoard

-- Define a player
inductive Player
| Jia
| Yi

-- Define the game state
structure GameState where
  board : List Piece
  currentPlayer : Player

-- Define a winning condition
def isWinningState (state : GameState) : Prop :=
  ∃ (i j k : GameBoard),
    (i.val + 1) % 99 = j.val ∧
    (j.val + 1) % 99 = k.val ∧
    ∃ (pi pj pk : Piece),
      pi ∈ state.board ∧ pj ∈ state.board ∧ pk ∈ state.board ∧
      pi.position = i ∧ pj.position = j ∧ pk.position = k ∧
      pj.number.val - pi.number.val = pk.number.val - pj.number.val

-- Define a strategy
def Strategy := GameState → Option Piece

-- Define the theorem
theorem no_winning_strategy :
  ¬∃ (s : Strategy), ∀ (opponent_strategy : Strategy),
    (∃ (n : ℕ) (final_state : GameState),
      final_state.currentPlayer = Player.Yi ∧
      isWinningState final_state) ∨
    (∃ (n : ℕ) (final_state : GameState),
      final_state.currentPlayer = Player.Jia ∧
      isWinningState final_state) :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l1644_164457


namespace NUMINAMATH_CALUDE_solution_difference_l1644_164461

theorem solution_difference (m : ℚ) : 
  (∃ x₁ x₂ : ℚ, 5*m + 3*x₁ = 1 + x₁ ∧ 2*x₂ + m = 3*m ∧ x₁ = x₂ + 2) ↔ m = -3/7 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l1644_164461


namespace NUMINAMATH_CALUDE_bathing_suit_sets_proof_l1644_164416

-- Define the constants from the problem
def total_time : ℕ := 60
def runway_time : ℕ := 2
def num_models : ℕ := 6
def evening_wear_sets : ℕ := 3

-- Define the function to calculate bathing suit sets per model
def bathing_suit_sets_per_model : ℕ :=
  let total_evening_wear_time := num_models * evening_wear_sets * runway_time
  let remaining_time := total_time - total_evening_wear_time
  let total_bathing_suit_trips := remaining_time / runway_time
  total_bathing_suit_trips / num_models

-- Theorem statement
theorem bathing_suit_sets_proof :
  bathing_suit_sets_per_model = 2 :=
by sorry

end NUMINAMATH_CALUDE_bathing_suit_sets_proof_l1644_164416


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1644_164465

theorem complex_equation_solution (z : ℂ) : (2 * Complex.I) / z = 1 - Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1644_164465


namespace NUMINAMATH_CALUDE_nonzero_even_from_second_step_l1644_164424

/-- Represents a bi-infinite sequence of integers -/
def BiInfiniteSequence := ℤ → ℤ

/-- The initial sequence with one 1 and all other elements 0 -/
def initial_sequence : BiInfiniteSequence :=
  fun i => if i = 0 then 1 else 0

/-- The next sequence after one step of evolution -/
def next_sequence (s : BiInfiniteSequence) : BiInfiniteSequence :=
  fun i => s (i - 1) + s i + s (i + 1)

/-- The sequence after n steps of evolution -/
def evolved_sequence (n : ℕ) : BiInfiniteSequence :=
  match n with
  | 0 => initial_sequence
  | m + 1 => next_sequence (evolved_sequence m)

/-- Predicate to check if a sequence contains a non-zero even number -/
def contains_nonzero_even (s : BiInfiniteSequence) : Prop :=
  ∃ i : ℤ, s i ≠ 0 ∧ s i % 2 = 0

/-- The main theorem to be proved -/
theorem nonzero_even_from_second_step :
  ∀ n : ℕ, n ≥ 2 → contains_nonzero_even (evolved_sequence n) :=
sorry

end NUMINAMATH_CALUDE_nonzero_even_from_second_step_l1644_164424


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l1644_164452

/-- Given two identical cylinders with initial radius 5 inches and height 4 inches,
    prove that when the radius of one cylinder is increased by x inches and
    the height of the second cylinder is increased by 2 inches, resulting in
    equal volumes, x = (5(√6 - 2)) / 2. -/
theorem cylinder_volume_equality (x : ℝ) : 
  (π * (5 + x)^2 * 4 = π * 5^2 * (4 + 2)) → x = (5 * (Real.sqrt 6 - 2)) / 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l1644_164452


namespace NUMINAMATH_CALUDE_ratio_of_P_and_Q_l1644_164453

theorem ratio_of_P_and_Q (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4*x) = (x^2 + x + 15) / (x^3 + x^2 - 20*x)) →
  (Q : ℚ) / P = -45 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_P_and_Q_l1644_164453


namespace NUMINAMATH_CALUDE_katie_baked_18_cupcakes_l1644_164451

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 8

/-- The number of packages Katie could make after Todd ate some cupcakes -/
def packages : ℕ := 5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 2

/-- The initial number of cupcakes Katie baked -/
def initial_cupcakes : ℕ := todd_ate + packages * cupcakes_per_package

theorem katie_baked_18_cupcakes : initial_cupcakes = 18 := by
  sorry

end NUMINAMATH_CALUDE_katie_baked_18_cupcakes_l1644_164451


namespace NUMINAMATH_CALUDE_range_of_x_l1644_164410

theorem range_of_x (x y : ℝ) (h1 : 2*x - y = 4) (h2 : -2 < y) (h3 : y ≤ 3) :
  1 < x ∧ x ≤ 7/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1644_164410


namespace NUMINAMATH_CALUDE_solve_for_x_l1644_164422

theorem solve_for_x (x y : ℝ) : 3 * x - 4 * y = 6 → x = (6 + 4 * y) / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1644_164422


namespace NUMINAMATH_CALUDE_root_existence_condition_l1644_164476

theorem root_existence_condition (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, a * x + 3 = 0) ↔ (a ≤ -3/2 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_root_existence_condition_l1644_164476


namespace NUMINAMATH_CALUDE_mosaic_length_l1644_164433

theorem mosaic_length 
  (height_feet : ℝ) 
  (tile_size_inch : ℝ) 
  (total_tiles : ℕ) : ℝ :=
  let height_inch : ℝ := height_feet * 12
  let area_inch_sq : ℝ := total_tiles * tile_size_inch ^ 2
  let length_inch : ℝ := area_inch_sq / height_inch
  let length_feet : ℝ := length_inch / 12
  by
    have h1 : height_feet = 10 := by sorry
    have h2 : tile_size_inch = 1 := by sorry
    have h3 : total_tiles = 21600 := by sorry
    sorry

#check mosaic_length

end NUMINAMATH_CALUDE_mosaic_length_l1644_164433


namespace NUMINAMATH_CALUDE_set_union_problem_l1644_164415

theorem set_union_problem (a b : ℝ) :
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {2^a, b}
  A ∩ B = {1} → A ∪ B = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1644_164415


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1644_164414

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 4 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1644_164414


namespace NUMINAMATH_CALUDE_cashier_error_l1644_164496

theorem cashier_error (x y : ℕ) : 9 * x + 15 * y ≠ 485 := by
  sorry

end NUMINAMATH_CALUDE_cashier_error_l1644_164496


namespace NUMINAMATH_CALUDE_union_covers_reals_l1644_164479

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem union_covers_reals (a : ℝ) : 
  (A a ∪ B a = Set.univ) ↔ a ∈ Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_union_covers_reals_l1644_164479


namespace NUMINAMATH_CALUDE_cell_division_problem_l1644_164417

/-- The number of cells after a given time, starting with one cell -/
def num_cells (division_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  2^(elapsed_time / division_time)

/-- The time between cell divisions in minutes -/
def division_time : ℕ := 30

/-- The total elapsed time in minutes -/
def total_time : ℕ := 4 * 60 + 30

theorem cell_division_problem :
  num_cells division_time total_time = 512 := by
  sorry

end NUMINAMATH_CALUDE_cell_division_problem_l1644_164417


namespace NUMINAMATH_CALUDE_gcd_of_72_120_180_l1644_164430

theorem gcd_of_72_120_180 : Nat.gcd 72 (Nat.gcd 120 180) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_180_l1644_164430


namespace NUMINAMATH_CALUDE_new_light_wattage_l1644_164429

theorem new_light_wattage (original_wattage : ℝ) (percentage_increase : ℝ) :
  original_wattage = 80 →
  percentage_increase = 25 →
  original_wattage * (1 + percentage_increase / 100) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_new_light_wattage_l1644_164429


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1644_164495

theorem smallest_solution_of_equation (x : ℝ) :
  (3 * x^2 + 33 * x - 90 = x * (x + 18)) →
  x ≥ -10.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1644_164495


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l1644_164467

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 13*x + 30 = 0 ∧ x ≠ 3 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l1644_164467


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1644_164487

def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

theorem contrapositive_equivalence :
  (∀ m : ℝ, m ≤ 0 → has_real_roots m) ↔
  (∀ m : ℝ, ¬(has_real_roots m) → m > 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1644_164487


namespace NUMINAMATH_CALUDE_goods_train_length_l1644_164464

/-- The length of a goods train passing a man on another train --/
theorem goods_train_length
  (man_train_speed : ℝ)
  (goods_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : man_train_speed = 40)
  (h2 : goods_train_speed = 72)
  (h3 : passing_time = 9) :
  (man_train_speed + goods_train_speed) * passing_time * 1000 / 3600 = 280 := by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l1644_164464
