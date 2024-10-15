import Mathlib

namespace NUMINAMATH_CALUDE_equiangular_rational_sides_prime_is_regular_l4055_405526

/-- An equiangular polygon with p sides -/
structure EquiangularPolygon (p : ℕ) where
  sides : Fin p → ℚ
  is_equiangular : True  -- We assume this property is satisfied

/-- A regular polygon is an equiangular polygon with all sides equal -/
def is_regular (poly : EquiangularPolygon p) : Prop :=
  ∀ i j : Fin p, poly.sides i = poly.sides j

theorem equiangular_rational_sides_prime_is_regular
  (p : ℕ) (hp : p.Prime) (hp2 : p > 2) (poly : EquiangularPolygon p) :
  is_regular poly :=
sorry

end NUMINAMATH_CALUDE_equiangular_rational_sides_prime_is_regular_l4055_405526


namespace NUMINAMATH_CALUDE_platform_length_l4055_405592

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, the length of the platform is 350 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 39 →
  pole_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length / pole_time) * platform_time = train_length + platform_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l4055_405592


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l4055_405568

/-- Given points A, B, C on the inverse proportion function y = -2/x, 
    prove the relationship between their y-coordinates. -/
theorem inverse_proportion_y_relationship (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-2) → y₂ = -2 / 2 → y₃ = -2 / 3 → y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

#check inverse_proportion_y_relationship

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l4055_405568


namespace NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l4055_405543

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ+) : ℕ := (Finset.range n.val.succ).prod (fun i => i + 1)

/-- Theorem stating that the greatest common divisor of 8! and 10! is equal to 8! -/
theorem gcd_8_factorial_10_factorial :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l4055_405543


namespace NUMINAMATH_CALUDE_jenny_recycling_problem_l4055_405557

/-- The weight of each can in ounces -/
def can_weight : ℚ := 2

theorem jenny_recycling_problem :
  let total_weight : ℚ := 100
  let bottle_weight : ℚ := 6
  let num_cans : ℚ := 20
  let cents_per_bottle : ℚ := 10
  let cents_per_can : ℚ := 3
  let total_cents : ℚ := 160
  (total_weight - num_cans * can_weight) / bottle_weight * cents_per_bottle + num_cans * cents_per_can = total_cents :=
by sorry

end NUMINAMATH_CALUDE_jenny_recycling_problem_l4055_405557


namespace NUMINAMATH_CALUDE_emily_furniture_time_l4055_405529

def chairs : ℕ := 4
def tables : ℕ := 2
def time_per_piece : ℕ := 8

theorem emily_furniture_time : chairs + tables * time_per_piece = 48 := by
  sorry

end NUMINAMATH_CALUDE_emily_furniture_time_l4055_405529


namespace NUMINAMATH_CALUDE_photo_gallery_problem_l4055_405587

/-- The total number of photos in a gallery after a two-day trip -/
def total_photos (initial : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  initial + first_day + second_day

/-- Theorem: Given the conditions of the photo gallery problem, the total number of photos is 920 -/
theorem photo_gallery_problem :
  let initial := 400
  let first_day := initial / 2
  let second_day := first_day + 120
  total_photos initial first_day second_day = 920 := by
  sorry

end NUMINAMATH_CALUDE_photo_gallery_problem_l4055_405587


namespace NUMINAMATH_CALUDE_exponential_decreasing_for_base_less_than_one_l4055_405505

theorem exponential_decreasing_for_base_less_than_one 
  (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  a^((-0.1) : ℝ) > a^(0.1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_exponential_decreasing_for_base_less_than_one_l4055_405505


namespace NUMINAMATH_CALUDE_certain_number_problem_l4055_405553

theorem certain_number_problem (x : ℝ) : 300 + (x * 8) = 340 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4055_405553


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_l4055_405573

/-- Triangle XYZ -/
structure TriangleXYZ where
  X : ℝ × ℝ
  Y : ℝ × ℝ := (0, 0)
  Z : ℝ × ℝ := (150, 0)
  area : ℝ := 1200

/-- Triangle XWV -/
structure TriangleXWV where
  X : ℝ × ℝ
  W : ℝ × ℝ := (500, 300)
  V : ℝ × ℝ := (510, 290)
  area : ℝ := 3600

/-- The theorem stating that the sum of all possible x-coordinates of X is 3200 -/
theorem sum_of_x_coordinates (triangle_xyz : TriangleXYZ) (triangle_xwv : TriangleXWV) 
  (h : triangle_xyz.X = triangle_xwv.X) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ + x₂ + x₃ + x₄ = 3200 ∧ 
    (triangle_xyz.X.1 = x₁ ∨ triangle_xyz.X.1 = x₂ ∨ triangle_xyz.X.1 = x₃ ∨ triangle_xyz.X.1 = x₄)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_l4055_405573


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_octagon_l4055_405528

/-- An octagon is a polygon with 8 vertices -/
def Octagon : Type := Unit

/-- The number of vertices in an octagon -/
def num_vertices (o : Octagon) : ℕ := 8

/-- The number of adjacent vertices for any vertex in an octagon -/
def num_adjacent (o : Octagon) : ℕ := 2

/-- The probability of choosing two distinct adjacent vertices in an octagon -/
def prob_adjacent_vertices (o : Octagon) : ℚ :=
  (num_adjacent o : ℚ) / ((num_vertices o - 1) : ℚ)

theorem prob_adjacent_vertices_octagon :
  ∀ o : Octagon, prob_adjacent_vertices o = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_octagon_l4055_405528


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4055_405566

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₅ = -1 and a₈ = 2,
    prove that the common difference is 1 and the first term is -5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
    (h_arith : is_arithmetic_sequence a)
    (h_a5 : a 5 = -1)
    (h_a8 : a 8 = 2) :
    (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 1 ∧ a 1 = -5 :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4055_405566


namespace NUMINAMATH_CALUDE_age_difference_proof_l4055_405583

theorem age_difference_proof (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 245 →
  monica_age - patrick_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l4055_405583


namespace NUMINAMATH_CALUDE_problem_solution_l4055_405534

theorem problem_solution (x y : ℕ+) :
  (x : ℚ) / (Nat.gcd x.val y.val : ℚ) + (y : ℚ) / (Nat.gcd x.val y.val : ℚ) = 18 ∧
  Nat.lcm x.val y.val = 975 →
  x = 75 ∧ y = 195 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4055_405534


namespace NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l4055_405546

def number1 : ℕ := 1305
def number2 : ℕ := 4665
def number3 : ℕ := 6905

def difference1 : ℕ := number2 - number1
def difference2 : ℕ := number3 - number2
def difference3 : ℕ := number3 - number1

def n : ℕ := Nat.gcd difference1 (Nat.gcd difference2 difference3)

def sum_of_digits (num : ℕ) : ℕ :=
  if num < 10 then num
  else (num % 10) + sum_of_digits (num / 10)

theorem greatest_common_divisor_and_digit_sum :
  n = 1120 ∧ sum_of_digits n = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l4055_405546


namespace NUMINAMATH_CALUDE_intersection_point_polar_curves_l4055_405516

theorem intersection_point_polar_curves (θ : Real) (ρ : Real) 
  (h1 : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h2 : ρ = 2 * Real.sin θ)
  (h3 : ρ * Real.cos θ = -1) :
  ∃ (ρ_intersect θ_intersect : Real),
    ρ_intersect = Real.sqrt (8 + 4 * Real.sqrt 3) ∧
    θ_intersect = 3 * Real.pi / 4 ∧
    ρ_intersect = 2 * Real.sin θ_intersect ∧
    ρ_intersect * Real.cos θ_intersect = -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_polar_curves_l4055_405516


namespace NUMINAMATH_CALUDE_mass_calculation_l4055_405572

/-- Given a concentration and volume, calculate the mass -/
def calculate_mass (C : ℝ) (V : ℝ) : ℝ := C * V

/-- Theorem stating that for given concentration and volume, the mass is 32 mg -/
theorem mass_calculation (C V : ℝ) (hC : C = 4) (hV : V = 8) :
  calculate_mass C V = 32 := by
  sorry

end NUMINAMATH_CALUDE_mass_calculation_l4055_405572


namespace NUMINAMATH_CALUDE_solve_for_d_l4055_405581

theorem solve_for_d (n k c d : ℝ) (h : n = (2 * k * c * d) / (c + d)) (h_nonzero : 2 * k * c ≠ n) :
  d = (n * c) / (2 * k * c - n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l4055_405581


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l4055_405588

theorem at_least_one_equation_has_two_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  ∃ (x y : ℝ),
    (x ≠ y ∧
      ((a * x^2 + 2 * b * x + c = 0 ∧ a * y^2 + 2 * b * y + c = 0) ∨
       (b * x^2 + 2 * c * x + a = 0 ∧ b * y^2 + 2 * c * y + a = 0) ∨
       (c * x^2 + 2 * a * x + b = 0 ∧ c * y^2 + 2 * a * y + b = 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l4055_405588


namespace NUMINAMATH_CALUDE_petes_number_l4055_405514

theorem petes_number : ∃ x : ℝ, 3 * (2 * x + 15) = 141 ∧ x = 16 := by sorry

end NUMINAMATH_CALUDE_petes_number_l4055_405514


namespace NUMINAMATH_CALUDE_smallest_n_squares_average_is_square_l4055_405589

theorem smallest_n_squares_average_is_square : 
  (∀ k : ℕ, k > 1 ∧ k < 337 → ¬ (∃ m : ℕ, (k + 1) * (2 * k + 1) / 6 = m^2)) ∧ 
  (∃ m : ℕ, (337 + 1) * (2 * 337 + 1) / 6 = m^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_squares_average_is_square_l4055_405589


namespace NUMINAMATH_CALUDE_rectangular_field_area_l4055_405547

/-- Proves that a rectangular field with width one-third of length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  2 * (width + length) = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l4055_405547


namespace NUMINAMATH_CALUDE_fraction_multiplication_l4055_405536

theorem fraction_multiplication : (2 : ℚ) / 3 * 3 / 5 * 4 / 7 * 5 / 8 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l4055_405536


namespace NUMINAMATH_CALUDE_valid_k_values_l4055_405598

def A (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

def ends_with_k_zeros (m k : ℕ) : Prop :=
  ∃ r : ℕ, m = r * 10^k ∧ r % 10 ≠ 0

theorem valid_k_values :
  {k : ℕ | ∃ n : ℕ, ends_with_k_zeros (A n) k} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_valid_k_values_l4055_405598


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l4055_405511

theorem largest_n_for_equation : 
  ∃ (x y z : ℕ+), 8^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 10 ∧ 
  ∀ (n : ℕ+), n > 8 → ¬∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l4055_405511


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4055_405521

theorem complex_fraction_equality : 
  let i : ℂ := Complex.I
  (7 + i) / (3 + 4*i) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4055_405521


namespace NUMINAMATH_CALUDE_store_optimal_pricing_l4055_405556

/-- Represents the store's product information and pricing strategy. -/
structure Store where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  retail_price_A : ℝ
  retail_price_B : ℝ
  daily_sales : ℝ
  price_decrease : ℝ

/-- Conditions for the store's pricing and sales. -/
def store_conditions (s : Store) : Prop :=
  s.purchase_price_A + s.purchase_price_B = 3 ∧
  s.retail_price_A = s.purchase_price_A + 1 ∧
  s.retail_price_B = 2 * s.purchase_price_B - 1 ∧
  3 * s.retail_price_A + 2 * s.retail_price_B = 12 ∧
  s.daily_sales = 500 ∧
  s.price_decrease > 0

/-- The profit function for the store. -/
def profit (s : Store) : ℝ :=
  (s.retail_price_A - s.price_decrease) * (s.daily_sales + 1000 * s.price_decrease) + s.retail_price_B * s.daily_sales - (s.purchase_price_A + s.purchase_price_B) * s.daily_sales

/-- Theorem stating the correct retail prices and optimal price decrease for maximum profit. -/
theorem store_optimal_pricing (s : Store) (h : store_conditions s) :
  s.retail_price_A = 2 ∧ s.retail_price_B = 3 ∧ profit s = 1000 ↔ s.price_decrease = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_store_optimal_pricing_l4055_405556


namespace NUMINAMATH_CALUDE_power_of_power_l4055_405559

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l4055_405559


namespace NUMINAMATH_CALUDE_average_of_wxz_l4055_405564

variable (w x y z t : ℝ)

theorem average_of_wxz (h1 : 3/w + 3/x + 3/z = 3/(y + t))
                       (h2 : w*x*z = y + t)
                       (h3 : w*z + x*t + y*z = 3*w + 3*x + 3*z) :
  (w + x + z) / 3 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_wxz_l4055_405564


namespace NUMINAMATH_CALUDE_p_plus_q_equals_31_l4055_405518

theorem p_plus_q_equals_31 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 18 * x + 27) / (x - 3)) →
  P + Q = 31 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_31_l4055_405518


namespace NUMINAMATH_CALUDE_sum_equals_zero_l4055_405552

def f (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_equals_zero :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l4055_405552


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l4055_405578

def distance : ℝ := 120
def gasoline : ℝ := 6

theorem car_fuel_efficiency : distance / gasoline = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l4055_405578


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l4055_405597

def is_prime (n : ℕ) : Prop := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def has_no_prime_factor_less_than (n m : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square_with_large_factors : 
  ∃ (n : ℕ), n > 0 ∧ 
  ¬ is_prime n ∧ 
  ¬ is_perfect_square n ∧ 
  has_no_prime_factor_less_than n 50 ∧ 
  ∀ (m : ℕ), m > 0 ∧ 
    m < n ∧ 
    ¬ is_prime m ∧ 
    ¬ is_perfect_square m → 
    ¬ has_no_prime_factor_less_than m 50 :=
by
  use 3127
  sorry

#eval 3127

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l4055_405597


namespace NUMINAMATH_CALUDE_lucy_cookie_packs_l4055_405504

/-- The number of cookie packs Lucy bought at the grocery store. -/
def cookie_packs : ℕ := 28 - 16

/-- The total number of grocery packs Lucy bought. -/
def total_packs : ℕ := 28

/-- The number of noodle packs Lucy bought. -/
def noodle_packs : ℕ := 16

theorem lucy_cookie_packs : 
  cookie_packs = 12 ∧ 
  total_packs = cookie_packs + noodle_packs :=
by sorry

end NUMINAMATH_CALUDE_lucy_cookie_packs_l4055_405504


namespace NUMINAMATH_CALUDE_inequality_proof_l4055_405586

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2)/c + (c^2 - b^2)/a + (a^2 - c^2)/b ≥ 3*a - 4*b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4055_405586


namespace NUMINAMATH_CALUDE_only_solutions_all_negative_one_or_all_one_l4055_405508

/-- A sequence of 2016 real numbers satisfying the given equation -/
def SequenceSatisfyingEquation (x : Fin 2016 → ℝ) : Prop :=
  ∀ i : Fin 2016, x i ^ 2 + x i - 1 = x (i.succ)

/-- The theorem stating the only solutions are all -1 or all 1 -/
theorem only_solutions_all_negative_one_or_all_one
  (x : Fin 2016 → ℝ) (h : SequenceSatisfyingEquation x) :
  (∀ i, x i = -1) ∨ (∀ i, x i = 1) := by
  sorry

#check only_solutions_all_negative_one_or_all_one

end NUMINAMATH_CALUDE_only_solutions_all_negative_one_or_all_one_l4055_405508


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l4055_405525

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l4055_405525


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4055_405510

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.I * (x + y) = x - 1) → (x = 1 ∧ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4055_405510


namespace NUMINAMATH_CALUDE_decode_sequence_is_palindrome_l4055_405582

/-- Represents the mapping from indices to letters -/
def letter_mapping : Nat → Char
| 1 => 'A'
| 2 => 'E'
| 3 => 'B'
| 4 => 'Γ'
| 5 => 'Δ'
| 6 => 'E'
| 7 => 'E'
| 8 => 'E'
| 9 => '3'
| 10 => 'V'
| 11 => 'U'
| 12 => 'K'
| 13 => 'J'
| 14 => 'M'
| 15 => 'H'
| 16 => 'O'
| 17 => '4'
| 18 => 'P'
| 19 => 'C'
| 20 => 'T'
| 21 => 'y'
| 22 => 'Φ'
| 23 => 'X'
| 24 => '4'
| 25 => '4'
| 26 => 'W'
| 27 => 'M'
| 28 => 'b'
| 29 => 'b'
| 30 => 'b'
| 31 => '3'
| 32 => 'O'
| 33 => '夕'
| _ => ' '  -- Default case

/-- The sequence of numbers to be decoded -/
def encoded_sequence : List Nat := [1, 1, 3, 0, 1, 1, 1, 7, 1, 5, 3, 1, 5, 1, 3, 2, 3, 2, 1, 5, 3, 1, 1, 2, 3, 2, 6, 2, 6, 1, 4, 1, 1, 2, 7, 3, 1, 4, 1, 1, 9, 1, 5, 0, 4, 1, 4, 9]

/-- Function to decode the sequence -/
def decode (seq : List Nat) : String := sorry

/-- The expected decoded palindrome -/
def expected_palindrome : String := "голоден носитель лет и сон не долг"

/-- Theorem stating that decoding the sequence results in the expected palindrome -/
theorem decode_sequence_is_palindrome : decode encoded_sequence = expected_palindrome := by sorry

end NUMINAMATH_CALUDE_decode_sequence_is_palindrome_l4055_405582


namespace NUMINAMATH_CALUDE_brendan_afternoon_catch_brendan_fishing_proof_l4055_405591

theorem brendan_afternoon_catch (morning_catch : ℕ) (thrown_back : ℕ) (dad_catch : ℕ) (total_catch : ℕ) : ℕ :=
  let kept_morning := morning_catch - thrown_back
  let afternoon_catch := total_catch - kept_morning - dad_catch
  afternoon_catch

theorem brendan_fishing_proof :
  let morning_catch := 8
  let thrown_back := 3
  let dad_catch := 13
  let total_catch := 23
  brendan_afternoon_catch morning_catch thrown_back dad_catch total_catch = 5 := by
  sorry

end NUMINAMATH_CALUDE_brendan_afternoon_catch_brendan_fishing_proof_l4055_405591


namespace NUMINAMATH_CALUDE_wario_field_goals_l4055_405571

theorem wario_field_goals 
  (missed_fraction : ℚ)
  (wide_right_fraction : ℚ)
  (wide_right_misses : ℕ)
  (h1 : missed_fraction = 1 / 4)
  (h2 : wide_right_fraction = 1 / 5)
  (h3 : wide_right_misses = 3) :
  ∃ (total_attempts : ℕ), 
    (↑wide_right_misses : ℚ) / wide_right_fraction / missed_fraction = total_attempts ∧ 
    total_attempts = 60 := by
sorry


end NUMINAMATH_CALUDE_wario_field_goals_l4055_405571


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l4055_405558

/-- The total percentage allocated to categories other than basic astrophysics -/
def other_categories_percentage : ℝ := 98

/-- The total degrees in a circle -/
def circle_degrees : ℝ := 360

/-- The percentage allocated to basic astrophysics -/
def basic_astrophysics_percentage : ℝ := 100 - other_categories_percentage

theorem basic_astrophysics_degrees :
  (basic_astrophysics_percentage / 100) * circle_degrees = 7.2 := by sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l4055_405558


namespace NUMINAMATH_CALUDE_set_operations_theorem_l4055_405517

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem set_operations_theorem :
  (Set.compl A ∪ B = {x | x < 5}) ∧
  (A ∩ Set.compl B = {x | x ≥ 5}) := by sorry

end NUMINAMATH_CALUDE_set_operations_theorem_l4055_405517


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_l4055_405549

theorem sum_of_imaginary_parts (a c d e f : ℂ) : 
  (a + 2*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 4*Complex.I →
  e = -2*a - c →
  d + f = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_l4055_405549


namespace NUMINAMATH_CALUDE_square_area_measurement_error_l4055_405569

theorem square_area_measurement_error :
  let actual_length : ℝ := L
  let measured_side1 : ℝ := L * (1 + 0.02)
  let measured_side2 : ℝ := L * (1 - 0.03)
  let calculated_area : ℝ := measured_side1 * measured_side2
  let actual_area : ℝ := L * L
  let error : ℝ := actual_area - calculated_area
  let percentage_error : ℝ := (error / actual_area) * 100
  percentage_error = 1.06 := by
sorry

end NUMINAMATH_CALUDE_square_area_measurement_error_l4055_405569


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4055_405523

def complex_equation (z : ℂ) : Prop :=
  z * ((1 + Complex.I) ^ 2) / 2 = 1 + 2 * Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : complex_equation z) :
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4055_405523


namespace NUMINAMATH_CALUDE_tree_planting_solution_l4055_405532

/-- Represents the configuration of trees in a circle. -/
structure TreeCircle where
  total : ℕ
  birches : ℕ
  lindens : ℕ
  all_lindens_between_birches : Bool
  one_birch_same_neighbors : Bool

/-- The theorem stating the unique solution for the tree planting problem. -/
theorem tree_planting_solution (circle : TreeCircle) : 
  circle.total = 130 ∧ 
  circle.total = circle.birches + circle.lindens ∧ 
  circle.birches > 0 ∧ 
  circle.lindens > 0 ∧
  circle.all_lindens_between_birches = true ∧
  circle.one_birch_same_neighbors = true →
  circle.birches = 87 := by
  sorry

#check tree_planting_solution

end NUMINAMATH_CALUDE_tree_planting_solution_l4055_405532


namespace NUMINAMATH_CALUDE_distribute_five_items_four_bags_l4055_405542

/-- The number of ways to distribute n different items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 36 ways to distribute 5 different items into 4 identical bags, allowing empty bags. -/
theorem distribute_five_items_four_bags : distribute 5 4 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_five_items_four_bags_l4055_405542


namespace NUMINAMATH_CALUDE_sqrt3_expression_equals_zero_l4055_405590

theorem sqrt3_expression_equals_zero :
  Real.sqrt 3 * (1 - Real.sqrt 3) - |-(Real.sqrt 3)| + (27 : ℝ) ^ (1/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_expression_equals_zero_l4055_405590


namespace NUMINAMATH_CALUDE_sequence_difference_l4055_405551

def sequence1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence2 (n : ℕ) : ℤ := sequence1 n - 1
def sequence3 (n : ℕ) : ℤ := (-2)^n - sequence2 n

theorem sequence_difference : sequence1 7 - sequence2 7 + sequence3 7 = -254 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l4055_405551


namespace NUMINAMATH_CALUDE_first_valid_year_is_1980_l4055_405538

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 1950 ∧ sum_of_digits year = 18

theorem first_valid_year_is_1980 :
  (∀ y : ℕ, y < 1980 → ¬(is_valid_year y)) ∧ is_valid_year 1980 := by
  sorry

end NUMINAMATH_CALUDE_first_valid_year_is_1980_l4055_405538


namespace NUMINAMATH_CALUDE_inequality_solution_l4055_405577

theorem inequality_solution (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x * (x + 1) + 1) :
  {x : ℝ | f x < 0} = {x : ℝ | x < 1/a ∨ x > 1} ∩ {x : ℝ | a ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4055_405577


namespace NUMINAMATH_CALUDE_part_one_part_two_l4055_405544

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - (a - 1)) * (x - (a + 1)) < 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Part 1: Prove that when a = 2, A ∪ B = B
theorem part_one : A 2 ∪ B = B := by sorry

-- Part 2: Prove that x ∈ A ⇔ x ∈ B holds if and only if 0 ≤ a ≤ 2
theorem part_two : (∀ x, x ∈ A a ↔ x ∈ B) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4055_405544


namespace NUMINAMATH_CALUDE_sum_equals_12x_l4055_405565

theorem sum_equals_12x (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y - x) : 
  x + y + z = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_12x_l4055_405565


namespace NUMINAMATH_CALUDE_fiftieth_term_is_247_l4055_405599

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 50th term of the arithmetic sequence starting with 2 and common difference 5 is 247 -/
theorem fiftieth_term_is_247 : arithmetic_sequence 2 5 50 = 247 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_247_l4055_405599


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l4055_405563

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a2 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l4055_405563


namespace NUMINAMATH_CALUDE_non_tipping_customers_l4055_405500

/-- Calculates the number of non-tipping customers given the total number of customers,
    the tip amount per tipping customer, and the total tips earned. -/
theorem non_tipping_customers
  (total_customers : ℕ)
  (tip_amount : ℕ)
  (total_tips : ℕ)
  (h1 : total_customers > 0)
  (h2 : tip_amount > 0)
  (h3 : total_tips % tip_amount = 0)
  (h4 : total_tips / tip_amount ≤ total_customers) :
  total_customers - (total_tips / tip_amount) =
    total_customers - (total_tips / tip_amount) :=
by sorry

end NUMINAMATH_CALUDE_non_tipping_customers_l4055_405500


namespace NUMINAMATH_CALUDE_work_completion_time_l4055_405512

theorem work_completion_time (P : ℕ) (D : ℕ) : 
  (P * D = 2 * (2 * P * 3)) → D = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4055_405512


namespace NUMINAMATH_CALUDE_minimum_parts_to_exceed_plan_l4055_405585

def plan : ℕ := 40
def excess_percentage : ℚ := 47/100

theorem minimum_parts_to_exceed_plan : 
  ∀ n : ℕ, (n : ℚ) ≥ plan * (1 + excess_percentage) → n ≥ 59 :=
sorry

end NUMINAMATH_CALUDE_minimum_parts_to_exceed_plan_l4055_405585


namespace NUMINAMATH_CALUDE_min_distance_at_median_l4055_405513

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- Represents the distance between two points -/
def distance (p q : Point) : ℝ :=
  |p.x - q.x|

/-- Given 9 points on a line, the sum of distances from an arbitrary point to all 9 points
    is minimized when the arbitrary point coincides with the 5th point -/
theorem min_distance_at_median (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ p₉ : Point) 
    (h : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ 
         p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x ∧ p₈.x < p₉.x) :
  ∀ p : Point, 
    distance p p₁ + distance p p₂ + distance p p₃ + distance p p₄ + 
    distance p p₅ + distance p p₆ + distance p p₇ + distance p p₈ + distance p p₉ ≥
    distance p₅ p₁ + distance p₅ p₂ + distance p₅ p₃ + distance p₅ p₄ + 
    distance p₅ p₅ + distance p₅ p₆ + distance p₅ p₇ + distance p₅ p₈ + distance p₅ p₉ :=
by sorry

end NUMINAMATH_CALUDE_min_distance_at_median_l4055_405513


namespace NUMINAMATH_CALUDE_robin_gum_packages_l4055_405584

/-- Given that Robin has some packages of gum, with 7 pieces in each package,
    6 extra pieces, and 41 pieces in total, prove that Robin has 5 packages. -/
theorem robin_gum_packages : ∀ (p : ℕ), 
  (7 * p + 6 = 41) → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l4055_405584


namespace NUMINAMATH_CALUDE_first_half_speed_l4055_405561

def total_distance : ℝ := 112
def total_time : ℝ := 5
def second_half_speed : ℝ := 24

theorem first_half_speed : 
  ∃ (v : ℝ), 
    v * (total_time - (total_distance / 2) / second_half_speed) = total_distance / 2 ∧ 
    v = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_half_speed_l4055_405561


namespace NUMINAMATH_CALUDE_sin_600_degrees_l4055_405596

theorem sin_600_degrees : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l4055_405596


namespace NUMINAMATH_CALUDE_part_one_part_two_l4055_405545

-- Define the function f
def f (a x : ℝ) : ℝ := (a + 1) * x^2 + 4 * a * x - 3

-- Part I
theorem part_one (a : ℝ) :
  a > 0 →
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ x₁ > 1 ∧ x₂ < 1) ↔
  (0 < a ∧ a < 2/5) :=
sorry

-- Part II
theorem part_two (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ f a 2) ↔
  a ≥ -1/3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4055_405545


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l4055_405535

theorem rectangular_prism_diagonal : 
  let a : ℝ := 12
  let b : ℝ := 24
  let c : ℝ := 15
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  diagonal = 3 * Real.sqrt 105 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l4055_405535


namespace NUMINAMATH_CALUDE_some_number_proof_l4055_405593

theorem some_number_proof (x y : ℝ) (h1 : 5 * x + 3 = 10 * x - y) (h2 : x = 4) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_some_number_proof_l4055_405593


namespace NUMINAMATH_CALUDE_fraction_difference_l4055_405554

theorem fraction_difference (a b : ℝ) : 
  a / (a + 1) - b / (b + 1) = (a - b) / ((a + 1) * (b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l4055_405554


namespace NUMINAMATH_CALUDE_distance_to_school_prove_distance_to_school_l4055_405550

theorem distance_to_school (time_with_traffic time_without_traffic : ℝ)
  (speed_difference : ℝ) (distance : ℝ) : Prop :=
  time_with_traffic = 20 / 60 →
  time_without_traffic = 15 / 60 →
  speed_difference = 15 →
  ∃ (speed_with_traffic : ℝ),
    distance = speed_with_traffic * time_with_traffic ∧
    distance = (speed_with_traffic + speed_difference) * time_without_traffic →
  distance = 15

-- The proof of the theorem
theorem prove_distance_to_school :
  ∀ (time_with_traffic time_without_traffic speed_difference distance : ℝ),
  distance_to_school time_with_traffic time_without_traffic speed_difference distance :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_prove_distance_to_school_l4055_405550


namespace NUMINAMATH_CALUDE_intersection_A_B_l4055_405537

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4055_405537


namespace NUMINAMATH_CALUDE_bingo_paths_l4055_405519

/-- Represents the number of paths to spell BINGO on a grid --/
def num_bingo_paths (b_to_i : Nat) (i_to_n : Nat) (n_to_g : Nat) (g_to_o : Nat) : Nat :=
  b_to_i * i_to_n * n_to_g * g_to_o

/-- Theorem stating the number of paths to spell BINGO --/
theorem bingo_paths :
  ∀ (b_to_i i_to_n n_to_g g_to_o : Nat),
    b_to_i = 3 →
    i_to_n = 3 →
    n_to_g = 2 →
    g_to_o = 2 →
    num_bingo_paths b_to_i i_to_n n_to_g g_to_o = 36 :=
by
  sorry

#eval num_bingo_paths 3 3 2 2

end NUMINAMATH_CALUDE_bingo_paths_l4055_405519


namespace NUMINAMATH_CALUDE_solve_equation_l4055_405562

theorem solve_equation (m x : ℝ) : 
  (m * x + 1 = 2 * (m - x)) ∧ (|x + 2| = 0) → m = -|3/4| :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l4055_405562


namespace NUMINAMATH_CALUDE_smallest_valid_number_l4055_405560

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 5 * k ∧ n % 3 = 1

theorem smallest_valid_number : (∀ m : ℕ, m > 0 ∧ m < 10 → ¬(is_valid m)) ∧ is_valid 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l4055_405560


namespace NUMINAMATH_CALUDE_suv_coupe_price_ratio_l4055_405540

theorem suv_coupe_price_ratio 
  (coupe_price : ℝ) 
  (commission_rate : ℝ) 
  (total_commission : ℝ) 
  (h1 : coupe_price = 30000)
  (h2 : commission_rate = 0.02)
  (h3 : total_commission = 1800)
  (h4 : ∃ x : ℝ, commission_rate * (coupe_price + x * coupe_price) = total_commission) :
  ∃ x : ℝ, x * coupe_price = 2 * coupe_price := by
sorry

end NUMINAMATH_CALUDE_suv_coupe_price_ratio_l4055_405540


namespace NUMINAMATH_CALUDE_divisors_of_1728_power_1728_l4055_405539

theorem divisors_of_1728_power_1728 :
  ∃! n : ℕ, n = (Finset.filter
    (fun d => (Finset.filter (fun x => x ∣ d) (Finset.range (d + 1))).card = 1728)
    (Finset.filter (fun x => x ∣ 1728^1728) (Finset.range (1728^1728 + 1)))).card :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_1728_power_1728_l4055_405539


namespace NUMINAMATH_CALUDE_expression_equality_l4055_405509

theorem expression_equality : 
  500 * 987 * 0.0987 * 50 = 2.5 * 987^2 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l4055_405509


namespace NUMINAMATH_CALUDE_ab_value_l4055_405575

theorem ab_value (a b : ℕ+) (h1 : a + b = 24) (h2 : 2 * a * b + 10 * a = 3 * b + 222) : a * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l4055_405575


namespace NUMINAMATH_CALUDE_room_occupancy_l4055_405548

theorem room_occupancy (people stools chairs : ℕ) : 
  people > stools ∧ 
  people > chairs ∧ 
  people < stools + chairs ∧ 
  2 * people + 3 * stools + 4 * chairs = 32 →
  people = 5 ∧ stools = 2 ∧ chairs = 4 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l4055_405548


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fourth_powers_equality_condition_l4055_405530

theorem min_value_of_sum_of_fourth_powers (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b) / c) ^ 4 + ((b + c) / d) ^ 4 + ((c + d) / a) ^ 4 + ((d + a) / b) ^ 4 ≥ 64 :=
by sorry

theorem equality_condition (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ((a + b) / c) ^ 4 + ((b + c) / d) ^ 4 + ((c + d) / a) ^ 4 + ((d + a) / b) ^ 4 = 64 ↔ 
  a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fourth_powers_equality_condition_l4055_405530


namespace NUMINAMATH_CALUDE_only_optionA_is_valid_l4055_405533

-- Define a type for programming statements
inductive ProgramStatement
  | Print (expr : String)
  | Input
  | InputAssign (var : String) (value : Nat)
  | PrintAssign (var : String) (expr : String)

-- Define a function to check if a statement is valid
def isValidStatement (stmt : ProgramStatement) : Prop :=
  match stmt with
  | ProgramStatement.Print expr => True
  | ProgramStatement.Input => False
  | ProgramStatement.InputAssign _ _ => False
  | ProgramStatement.PrintAssign _ _ => False

-- Define the given options
def optionA := ProgramStatement.Print "4*x"
def optionB := ProgramStatement.Input
def optionC := ProgramStatement.InputAssign "B" 3
def optionD := ProgramStatement.PrintAssign "y" "2*x+1"

-- Theorem to prove
theorem only_optionA_is_valid :
  isValidStatement optionA ∧
  ¬isValidStatement optionB ∧
  ¬isValidStatement optionC ∧
  ¬isValidStatement optionD :=
sorry

end NUMINAMATH_CALUDE_only_optionA_is_valid_l4055_405533


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4055_405520

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4055_405520


namespace NUMINAMATH_CALUDE_min_value_in_region_l4055_405574

-- Define the region
def in_region (x y : ℝ) : Prop :=
  y ≥ |x - 1| ∧ y ≤ 2

-- Define the function to be minimized
def f (x y : ℝ) : ℝ := 2*x - y

-- Theorem statement
theorem min_value_in_region :
  ∃ (min : ℝ), min = -4 ∧
  (∀ x y : ℝ, in_region x y → f x y ≥ min) ∧
  (∃ x y : ℝ, in_region x y ∧ f x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_in_region_l4055_405574


namespace NUMINAMATH_CALUDE_inequality_proof_root_inequality_l4055_405576

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hne : ¬(a = b ∧ b = c)) : 
  a*(b^2 + c^2) + b*(c^2 + a^2) + c*(a^2 + b^2) > 6*a*b*c :=
sorry

theorem root_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_root_inequality_l4055_405576


namespace NUMINAMATH_CALUDE_percentage_comparison_l4055_405555

theorem percentage_comparison (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : 0.02 * A > 0.03 * B) :
  0.05 * A > 0.07 * B := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l4055_405555


namespace NUMINAMATH_CALUDE_tim_income_percentage_forty_percent_less_l4055_405541

/-- Proves that Tim's income is 40% less than Juan's income given the conditions -/
theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.4 * tim) 
  (h2 : mary = 0.84 * juan) : 
  tim = 0.6 * juan := by
  sorry

/-- Proves that 40% less is equivalent to 60% of the original amount -/
theorem forty_percent_less (x y : ℝ) (h : x = 0.6 * y) : 
  x = y - 0.4 * y := by
  sorry

end NUMINAMATH_CALUDE_tim_income_percentage_forty_percent_less_l4055_405541


namespace NUMINAMATH_CALUDE_max_ab_value_l4055_405502

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 4) :
  ab ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 4 ∧ a₀*b₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l4055_405502


namespace NUMINAMATH_CALUDE_max_term_at_k_max_l4055_405506

/-- The value of k that maximizes the term C_{209}^k (√5)^k in the expansion of (1+√5)^209 -/
def k_max : ℕ := 145

/-- The binomial coefficient C(n,k) -/
def binomial_coeff (n k : ℕ) : ℕ := sorry

/-- The term C_{209}^k (√5)^k in the expansion of (1+√5)^209 -/
noncomputable def term (k : ℕ) : ℝ := (binomial_coeff 209 k) * (Real.sqrt 5) ^ k

theorem max_term_at_k_max :
  ∀ k : ℕ, k ≠ k_max → term k ≤ term k_max :=
sorry

end NUMINAMATH_CALUDE_max_term_at_k_max_l4055_405506


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nine_sqrt_three_l4055_405579

theorem sqrt_sum_equals_nine_sqrt_three : 
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nine_sqrt_three_l4055_405579


namespace NUMINAMATH_CALUDE_sqrt2_irrational_sqrt2_approximation_no_exact_rational_sqrt2_l4055_405503

-- Define √2 as an irrational number
noncomputable def sqrt2 : ℝ := Real.sqrt 2

-- Statement that √2 is irrational
theorem sqrt2_irrational : Irrational sqrt2 := sorry

-- Statement that √2 can be approximated by rationals
theorem sqrt2_approximation :
  ∀ ε > 0, ∃ p q : ℤ, q ≠ 0 ∧ |((p : ℝ) / q)^2 - 2| < ε := sorry

-- Statement that no rational number exactly equals √2
theorem no_exact_rational_sqrt2 :
  ¬∃ p q : ℤ, q ≠ 0 ∧ ((p : ℝ) / q)^2 = 2 := sorry

end NUMINAMATH_CALUDE_sqrt2_irrational_sqrt2_approximation_no_exact_rational_sqrt2_l4055_405503


namespace NUMINAMATH_CALUDE_curve_C_properties_l4055_405522

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 6*ρ*(Real.sin θ) + 12 = 0

-- Define the rectangular coordinates of a point on C
def point_on_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the distance |PM| + |PN|
def PM_PN (x y : ℝ) : ℝ :=
  y + (x + 1)

-- Theorem statement
theorem curve_C_properties :
  (∀ ρ θ : ℝ, C ρ θ ↔ point_on_C (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∃ max : ℝ, max = 6 + Real.sqrt 2 ∧
    ∀ x y : ℝ, point_on_C x y → PM_PN x y ≤ max) :=
sorry

end NUMINAMATH_CALUDE_curve_C_properties_l4055_405522


namespace NUMINAMATH_CALUDE_prob_two_threes_correct_l4055_405567

/-- The probability of rolling exactly two 3s when rolling eight standard 6-sided dice -/
def prob_two_threes : ℚ :=
  (28 : ℚ) * 15625 / 559872

/-- The probability calculated using binomial distribution -/
def prob_two_threes_calc : ℚ :=
  (Nat.choose 8 2 : ℚ) * (1/6)^2 * (5/6)^6

theorem prob_two_threes_correct : prob_two_threes = prob_two_threes_calc := by
  sorry

end NUMINAMATH_CALUDE_prob_two_threes_correct_l4055_405567


namespace NUMINAMATH_CALUDE_binomial_expected_value_l4055_405594

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expected_value (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Theorem: The expected value of X ~ B(6, 1/4) is 3/2 -/
theorem binomial_expected_value :
  let X : BinomialDistribution := { n := 6, p := 1/4, h_p := by norm_num }
  expected_value X = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expected_value_l4055_405594


namespace NUMINAMATH_CALUDE_multiples_of_four_median_l4055_405570

def first_seven_multiples_of_four : List ℕ := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem multiples_of_four_median (n : ℕ) :
  a ^ 2 - (b n) ^ 2 = 0 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_median_l4055_405570


namespace NUMINAMATH_CALUDE_salary_comparison_l4055_405527

/-- Represents the salary ratios of employees A, B, C, D, and E -/
def salary_ratios : Fin 5 → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

/-- The combined salary of employees B, C, and D in rupees -/
def combined_salary_bcd : ℚ := 15000

/-- Calculates the base salary unit given the combined salary of B, C, and D -/
def base_salary : ℚ := combined_salary_bcd / (salary_ratios 1 + salary_ratios 2 + salary_ratios 3)

theorem salary_comparison :
  /- The salary of C is 200% more than that of A -/
  (salary_ratios 2 * base_salary - salary_ratios 0 * base_salary) / (salary_ratios 0 * base_salary) * 100 = 200 ∧
  /- The ratio of the salary of E to the combined salary of A and B is 5:3 -/
  (salary_ratios 4 * base_salary) / ((salary_ratios 0 + salary_ratios 1) * base_salary) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l4055_405527


namespace NUMINAMATH_CALUDE_solve_system_l4055_405524

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7)
  (eq2 : x + 3 * y = 6) : 
  x = 2.7 := by sorry

end NUMINAMATH_CALUDE_solve_system_l4055_405524


namespace NUMINAMATH_CALUDE_product_difference_l4055_405507

theorem product_difference (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : ∃ k, b = 10 * k) :
  let correct_product := a * b
  let incorrect_product := (a * b) / 10
  correct_product = 10 * incorrect_product :=
by
  sorry

end NUMINAMATH_CALUDE_product_difference_l4055_405507


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l4055_405595

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l4055_405595


namespace NUMINAMATH_CALUDE_crayon_count_is_44_l4055_405515

/-- The number of crayons in the drawer after a series of additions and removals. -/
def final_crayon_count (initial : ℝ) (benny_add : ℝ) (lucy_remove : ℝ) (sam_add : ℝ) : ℝ :=
  initial + benny_add - lucy_remove + sam_add

/-- Theorem stating that the final number of crayons is 44 given the initial count and actions. -/
theorem crayon_count_is_44 :
  final_crayon_count 25 15.5 8.75 12.25 = 44 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_is_44_l4055_405515


namespace NUMINAMATH_CALUDE_four_step_staircase_l4055_405501

/-- The number of ways to climb a staircase with n steps -/
def climbStairs (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 8 ways to climb a staircase with 4 steps -/
theorem four_step_staircase : climbStairs 4 = 8 := by sorry

end NUMINAMATH_CALUDE_four_step_staircase_l4055_405501


namespace NUMINAMATH_CALUDE_partnership_share_calculation_l4055_405580

/-- Given a partnership where three partners invest different amounts and one partner's share is known, 
    calculate the share of another partner. -/
theorem partnership_share_calculation 
  (investment_a investment_b investment_c : ℕ)
  (duration : ℕ)
  (share_b : ℕ) 
  (h1 : investment_a = 11000)
  (h2 : investment_b = 15000)
  (h3 : investment_c = 23000)
  (h4 : duration = 8)
  (h5 : share_b = 3315) :
  (investment_a : ℚ) / (investment_a + investment_b + investment_c) * 
  (share_b : ℚ) * ((investment_a + investment_b + investment_c) : ℚ) / investment_b = 2421 :=
by sorry

end NUMINAMATH_CALUDE_partnership_share_calculation_l4055_405580


namespace NUMINAMATH_CALUDE_base8_plus_15_l4055_405531

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (x : ℕ) : ℕ :=
  (x / 100) * 64 + ((x / 10) % 10) * 8 + (x % 10)

/-- The problem statement --/
theorem base8_plus_15 : base8_to_base10 123 + 15 = 98 := by
  sorry

end NUMINAMATH_CALUDE_base8_plus_15_l4055_405531
