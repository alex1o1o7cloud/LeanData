import Mathlib

namespace NUMINAMATH_CALUDE_mrs_heine_treats_l540_54012

/-- The number of treats Mrs. Heine needs to buy for her pets -/
def total_treats (num_dogs : ℕ) (num_cats : ℕ) (num_parrots : ℕ) 
                 (biscuits_per_dog : ℕ) (treats_per_cat : ℕ) (sticks_per_parrot : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog + num_cats * treats_per_cat + num_parrots * sticks_per_parrot

/-- Theorem stating that Mrs. Heine needs to buy 11 treats in total -/
theorem mrs_heine_treats : total_treats 2 1 3 3 2 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_treats_l540_54012


namespace NUMINAMATH_CALUDE_three_function_properties_l540_54015

theorem three_function_properties :
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x - (deriv f) x = f (-x) - (deriv f) (-x)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, (deriv f) x ≠ 0) ∧ (∀ x : ℝ, f x = (deriv f) x)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, (deriv f) x ≠ 0) ∧ (∀ x : ℝ, f x = -(deriv f) x)) :=
by sorry

end NUMINAMATH_CALUDE_three_function_properties_l540_54015


namespace NUMINAMATH_CALUDE_least_clock_equivalent_l540_54097

def clock_equivalent (n : ℕ) : Prop :=
  n > 5 ∧ (n^2 - n) % 12 = 0

theorem least_clock_equivalent : ∃ (n : ℕ), clock_equivalent n ∧ ∀ m, m < n → ¬ clock_equivalent m :=
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_l540_54097


namespace NUMINAMATH_CALUDE_probability_two_ones_l540_54005

def num_dice : ℕ := 15
def num_sides : ℕ := 6
def target_num : ℕ := 1
def target_count : ℕ := 2

theorem probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) :
  n = num_dice →
  k = target_count →
  p = 1 / num_sides →
  (n.choose k * p^k * (1 - p)^(n - k) : ℚ) = (105 * 5^13 : ℚ) / 6^14 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_ones_l540_54005


namespace NUMINAMATH_CALUDE_circle_equation_l540_54091

/-- The equation of a circle with center (1, 2) passing through the origin (0, 0) -/
theorem circle_equation : ∀ x y : ℝ, 
  (x - 1)^2 + (y - 2)^2 = 5 ↔ 
  (x - 1)^2 + (y - 2)^2 = (0 - 1)^2 + (0 - 2)^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_l540_54091


namespace NUMINAMATH_CALUDE_equal_play_time_l540_54000

theorem equal_play_time (team_size : ℕ) (field_players : ℕ) (match_duration : ℕ) 
  (h1 : team_size = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45)
  (h4 : field_players < team_size) :
  (field_players * match_duration) / team_size = 36 := by
  sorry

end NUMINAMATH_CALUDE_equal_play_time_l540_54000


namespace NUMINAMATH_CALUDE_smaller_number_problem_l540_54016

theorem smaller_number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 14) (h4 : y = 3 * x) : x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l540_54016


namespace NUMINAMATH_CALUDE_sin_10_cos_20_cos_40_l540_54067

theorem sin_10_cos_20_cos_40 :
  Real.sin (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (40 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_cos_20_cos_40_l540_54067


namespace NUMINAMATH_CALUDE_average_b_c_l540_54074

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115)
  (h2 : a - c = 90) : 
  (b + c) / 2 = 70 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l540_54074


namespace NUMINAMATH_CALUDE_total_subscription_amount_l540_54061

/-- Prove that the total subscription amount is 50000 given the conditions of the problem -/
theorem total_subscription_amount (c b a : ℕ) 
  (h1 : b = c + 5000)  -- B subscribes 5000 more than C
  (h2 : a = b + 4000)  -- A subscribes 4000 more than B
  (h3 : 14700 * (a + b + c) = 35000 * a)  -- A's profit proportion
  : a + b + c = 50000 := by
  sorry

end NUMINAMATH_CALUDE_total_subscription_amount_l540_54061


namespace NUMINAMATH_CALUDE_sequence_next_term_l540_54079

theorem sequence_next_term (a₁ a₂ a₃ a₄ a₅ x : ℕ) : 
  a₁ = 2 ∧ a₂ = 5 ∧ a₃ = 11 ∧ a₄ = 20 ∧ a₅ = 32 ∧
  (a₂ - a₁) = 3 ∧ (a₃ - a₂) = 6 ∧ (a₄ - a₃) = 9 ∧ (a₅ - a₄) = 12 ∧
  (x - a₅) = (a₅ - a₄) + 3 →
  x = 47 := by
sorry

end NUMINAMATH_CALUDE_sequence_next_term_l540_54079


namespace NUMINAMATH_CALUDE_smallest_possible_a_l540_54064

theorem smallest_possible_a (a b c : ℝ) : 
  a > 0 → 
  (∃ n : ℤ, a + 2*b + 3*c = n) →
  (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y = a*(x - 1/2)^2 - 1/2) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ n : ℤ, a' + 2*b' + 3*c' = n) ∧
    (∀ x y : ℝ, y = a'*x^2 + b'*x + c' ↔ y = a'*(x - 1/2)^2 - 1/2)) →
    a ≤ a') →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l540_54064


namespace NUMINAMATH_CALUDE_uncovered_area_three_circles_l540_54023

theorem uncovered_area_three_circles (R : ℝ) (h : R = 10) :
  let r := R / 2
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let total_smaller_circles_area := 3 * smaller_circle_area
  larger_circle_area - total_smaller_circles_area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_uncovered_area_three_circles_l540_54023


namespace NUMINAMATH_CALUDE_four_digit_number_count_l540_54090

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := sorry

/-- A function that returns true if all digits in a natural number are different -/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- A function that returns the leftmost digit of a natural number -/
def leftmostDigit (n : ℕ) : ℕ := sorry

/-- A function that returns the rightmost digit of a natural number -/
def rightmostDigit (n : ℕ) : ℕ := sorry

theorem four_digit_number_count :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 1000 ≤ n ∧ n < 10000) ∧ 
    (∀ n ∈ S, isPrime (leftmostDigit n)) ∧
    (∀ n ∈ S, isPerfectSquare (rightmostDigit n)) ∧
    (∀ n ∈ S, allDigitsDifferent n) ∧
    Finset.card S ≥ 288 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_count_l540_54090


namespace NUMINAMATH_CALUDE_quadratic_polynomial_half_coefficient_integer_values_l540_54093

theorem quadratic_polynomial_half_coefficient_integer_values :
  ∃ (b c : ℚ), ∀ (x : ℤ), ∃ (y : ℤ), ((1/2 : ℚ) * x^2 + b * x + c : ℚ) = y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_half_coefficient_integer_values_l540_54093


namespace NUMINAMATH_CALUDE_square_eq_neg_two_i_implies_a_eq_one_coordinates_of_z_over_one_plus_i_l540_54077

-- Define the complex number z
def z (a : ℝ) : ℂ := a - Complex.I

-- Theorem 1
theorem square_eq_neg_two_i_implies_a_eq_one :
  ∀ a : ℝ, (z a)^2 = -2 * Complex.I → a = 1 := by sorry

-- Theorem 2
theorem coordinates_of_z_over_one_plus_i :
  let z : ℂ := z 2
  (z / (1 + Complex.I)).re = 1/2 ∧ (z / (1 + Complex.I)).im = -3/2 := by sorry

end NUMINAMATH_CALUDE_square_eq_neg_two_i_implies_a_eq_one_coordinates_of_z_over_one_plus_i_l540_54077


namespace NUMINAMATH_CALUDE_cattle_truck_capacity_l540_54025

/-- Calculates the capacity of a cattle transport truck given the total number of cattle,
    distance to safety, truck speed, and total transport time. -/
theorem cattle_truck_capacity
  (total_cattle : ℕ)
  (distance : ℝ)
  (speed : ℝ)
  (total_time : ℝ)
  (h_total_cattle : total_cattle = 400)
  (h_distance : distance = 60)
  (h_speed : speed = 60)
  (h_total_time : total_time = 40)
  : ℕ :=
by
  sorry

#check cattle_truck_capacity

end NUMINAMATH_CALUDE_cattle_truck_capacity_l540_54025


namespace NUMINAMATH_CALUDE_total_paid_equals_143_l540_54065

def manicure_cost : ℝ := 30
def pedicure_cost : ℝ := 40
def hair_treatment_cost : ℝ := 50

def manicure_tip_rate : ℝ := 0.25
def pedicure_tip_rate : ℝ := 0.20
def hair_treatment_tip_rate : ℝ := 0.15

def total_cost (service_cost : ℝ) (tip_rate : ℝ) : ℝ :=
  service_cost * (1 + tip_rate)

theorem total_paid_equals_143 :
  total_cost manicure_cost manicure_tip_rate +
  total_cost pedicure_cost pedicure_tip_rate +
  total_cost hair_treatment_cost hair_treatment_tip_rate = 143 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_equals_143_l540_54065


namespace NUMINAMATH_CALUDE_sum_of_xy_l540_54053

theorem sum_of_xy (x y : ℕ) : 
  x > 0 → y > 0 → x < 25 → y < 25 → x + y + x * y = 118 → x + y = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l540_54053


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l540_54039

/-- Given an angle α whose terminal side passes through the point (-3a, 4a) where a > 0,
    prove that sin α + 2cos α = -2/5 -/
theorem sine_cosine_sum (a : ℝ) (α : ℝ) (h1 : a > 0) 
    (h2 : Real.cos α = -3 * a / (5 * a)) (h3 : Real.sin α = 4 * a / (5 * a)) : 
    Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l540_54039


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l540_54017

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Theorem stating that under given conditions, the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola) (p : Parabola)
  (h_focus : h.a * h.a - h.b * h.b = p.p * h.a) -- Right focus of hyperbola coincides with focus of parabola
  (h_intersection : ∃ A B : ℝ × ℝ, 
    A.1^2 / h.a^2 - A.2^2 / h.b^2 = 1 ∧
    B.1^2 / h.a^2 - B.2^2 / h.b^2 = 1 ∧
    A.1 = -p.p/2 ∧ B.1 = -p.p/2) -- Directrix of parabola intersects hyperbola at A and B
  (h_asymptotes : ∃ C D : ℝ × ℝ,
    C.2 = h.b / h.a * C.1 ∧
    D.2 = -h.b / h.a * D.1) -- Asymptotes of hyperbola intersect at C and D
  (h_distance : ∃ A B C D : ℝ × ℝ,
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)) -- |CD| = √2|AB|
  : Real.sqrt ((h.a^2 - h.b^2) / h.a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l540_54017


namespace NUMINAMATH_CALUDE_opposite_and_reciprocal_sum_l540_54098

theorem opposite_and_reciprocal_sum (a b x y : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : x * y = 1)  -- x and y are reciprocals
  : 2 * (a + b) + (7 / 4) * x * y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_and_reciprocal_sum_l540_54098


namespace NUMINAMATH_CALUDE_solution_verification_l540_54081

theorem solution_verification (x : ℚ) : 
  x = 22 / 5 ↔ 10 * (5 * x + 4) - 4 = -4 * (2 - 15 * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l540_54081


namespace NUMINAMATH_CALUDE_xyz_sum_l540_54069

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 53)
  (h2 : y * z + x = 53)
  (h3 : z * x + y = 53) : 
  x + y + z = 54 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l540_54069


namespace NUMINAMATH_CALUDE_max_list_length_l540_54010

def is_valid_list (D : List Nat) : Prop :=
  ∀ x ∈ D, 1 ≤ x ∧ x ≤ 10

def count_occurrences (x : Nat) (L : List Nat) : Nat :=
  L.filter (· = x) |>.length

def generate_M (D : List Nat) : List Nat :=
  D.map (λ x => count_occurrences x D)

theorem max_list_length :
  ∃ (D : List Nat),
    is_valid_list D ∧
    D.length = 10 ∧
    generate_M D = D.reverse ∧
    ∀ (D' : List Nat),
      is_valid_list D' →
      generate_M D' = D'.reverse →
      D'.length ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_list_length_l540_54010


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l540_54060

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  geometric_sequence a q →
  a 1 = 1 →
  q ≠ 1 →
  q ≠ -1 →
  (∃ m : ℕ, a m = a 1 * a 2 * a 3 * a 4 * a 5) →
  m = 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l540_54060


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l540_54052

theorem polar_to_cartesian (p θ x y : ℝ) :
  (p = 8 * Real.cos θ) ∧ (x = p * Real.cos θ) ∧ (y = p * Real.sin θ) →
  x^2 + y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l540_54052


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l540_54054

theorem complex_number_real_condition (m : ℝ) :
  let z : ℂ := m - 3 + (m^2 - 9) * Complex.I
  z.im = 0 → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l540_54054


namespace NUMINAMATH_CALUDE_x_plus_y_equals_eight_l540_54046

theorem x_plus_y_equals_eight (x y : ℝ) 
  (h1 : |x| - x + y = 8) 
  (h2 : x + |y| + y = 16) : 
  x + y = 8 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_eight_l540_54046


namespace NUMINAMATH_CALUDE_skew_lines_planes_perpendicularity_l540_54031

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (are_skew : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem skew_lines_planes_perpendicularity 
  (m n l : Line) (α β : Plane) :
  are_skew m n →
  parallel_plane_line α m →
  parallel_plane_line α n →
  perpendicular_line_line l m →
  perpendicular_line_line l n →
  parallel_line_plane l β →
  perpendicular_plane_plane α β ∧ perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_skew_lines_planes_perpendicularity_l540_54031


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l540_54019

theorem sock_selection_theorem (n : Nat) (k : Nat) (red : Nat) :
  n = 8 → k = 4 → red = 1 →
  (Nat.choose n k) - (Nat.choose (n - red) k) = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l540_54019


namespace NUMINAMATH_CALUDE_population_growth_proof_l540_54014

theorem population_growth_proof (growth_rate_1 : ℝ) (growth_rate_2 : ℝ) : 
  growth_rate_1 = 0.2 →
  growth_rate_2 = growth_rate_1 + 0.3 * growth_rate_1 →
  (1 + growth_rate_1) * (1 + growth_rate_2) - 1 = 0.512 :=
by
  sorry

#check population_growth_proof

end NUMINAMATH_CALUDE_population_growth_proof_l540_54014


namespace NUMINAMATH_CALUDE_smallest_number_l540_54066

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [5, 8]  -- 85 in base 9
def num2 : List Nat := [0, 1, 2]  -- 210 in base 6
def num3 : List Nat := [0, 0, 0, 1]  -- 1000 in base 4
def num4 : List Nat := [1, 1, 1, 1, 1, 1]  -- 111111 in base 2

-- Theorem statement
theorem smallest_number :
  to_base_10 num4 2 < to_base_10 num1 9 ∧
  to_base_10 num4 2 < to_base_10 num2 6 ∧
  to_base_10 num4 2 < to_base_10 num3 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l540_54066


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l540_54062

/-- The ellipse equation -/
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line equation -/
def on_line (x y : ℝ) : Prop := 4*x - 2*y - 3 = 0

/-- Symmetry about the line -/
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), on_line x₀ y₀ ∧ x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem vector_sum_magnitude (x₁ y₁ x₂ y₂ : ℝ) :
  on_ellipse x₁ y₁ → on_ellipse x₂ y₂ → symmetric_about_line x₁ y₁ x₂ y₂ →
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l540_54062


namespace NUMINAMATH_CALUDE_toy_phone_price_l540_54056

theorem toy_phone_price (bert_phones : ℕ) (tory_guns : ℕ) (gun_price : ℕ) (extra_earnings : ℕ) :
  bert_phones = 8 →
  tory_guns = 7 →
  gun_price = 20 →
  extra_earnings = 4 →
  (tory_guns * gun_price + extra_earnings) / bert_phones = 18 :=
by sorry

end NUMINAMATH_CALUDE_toy_phone_price_l540_54056


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l540_54063

theorem average_of_a_and_b (a b : ℝ) : 
  (3 + 5 + 7 + a + b) / 5 = 15 → (a + b) / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l540_54063


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l540_54058

theorem complex_magnitude_product : 
  Complex.abs ((3 - 4*Complex.I) * (2 + 6*Complex.I) * 5) = 50 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l540_54058


namespace NUMINAMATH_CALUDE_books_left_to_read_l540_54084

theorem books_left_to_read 
  (total_books : ℕ) 
  (books_read : ℕ) 
  (h1 : total_books = 19) 
  (h2 : books_read = 4) : 
  total_books - books_read = 15 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l540_54084


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l540_54009

-- Define the first equation
def equation1 (x : ℝ) : Prop := (x + 1) * (x + 3) = 15

-- Define the second equation
def equation2 (y : ℝ) : Prop := (y - 3)^2 + 3*(y - 3) + 2 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = -6 ∨ x = 2)) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ y : ℝ, equation2 y) ∧ 
  (∀ y : ℝ, equation2 y ↔ (y = 1 ∨ y = 2)) :=
sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l540_54009


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l540_54086

theorem point_in_first_quadrant (α : Real) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin α - Real.cos α > 0 ∧ Real.tan α > 0) ↔ 
  (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) ∪ Set.Ioo Real.pi (5 * Real.pi / 4)) := by
sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l540_54086


namespace NUMINAMATH_CALUDE_square_b_minus_d_l540_54076

theorem square_b_minus_d (a b c d : ℝ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 9) : 
  (b - d)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_square_b_minus_d_l540_54076


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l540_54071

/-- Given an ellipse with equation 4(x+2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ), 
    (∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 ↔ (x + 2)^2 / 16 + y^2 / 4 = 1) ∧
    (C.1 = -2 ∧ C.2 = 4 ∨ C.1 = -2 ∧ C.2 = -4) ∧  -- C is an endpoint of the major axis
    (D.1 = 0 ∧ D.2 = 0 ∨ D.1 = -4 ∧ D.2 = 0) ∧    -- D is an endpoint of the minor axis
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l540_54071


namespace NUMINAMATH_CALUDE_hackathon_ends_at_noon_l540_54013

-- Define the start time of the hackathon
def hackathon_start : Nat := 12 * 60  -- noon in minutes since midnight

-- Define the duration of the hackathon
def hackathon_duration : Nat := 1440  -- duration in minutes

-- Define a function to calculate the end time of the hackathon
def hackathon_end (start : Nat) (duration : Nat) : Nat :=
  (start + duration) % (24 * 60)

-- Theorem to prove
theorem hackathon_ends_at_noon :
  hackathon_end hackathon_start hackathon_duration = hackathon_start :=
by sorry

end NUMINAMATH_CALUDE_hackathon_ends_at_noon_l540_54013


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l540_54021

theorem power_fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l540_54021


namespace NUMINAMATH_CALUDE_unique_card_arrangement_l540_54033

def CardPair := (Nat × Nat)

def is_valid_pair (p : CardPair) : Prop :=
  (p.1 ∣ p.2) ∨ (p.2 ∣ p.1)

def is_unique_arrangement (arr : List CardPair) : Prop :=
  arr.length = 5 ∧
  (∀ p ∈ arr, 1 ≤ p.1 ∧ p.1 ≤ 10 ∧ 1 ≤ p.2 ∧ p.2 ≤ 10) ∧
  (∀ p ∈ arr, is_valid_pair p) ∧
  (∀ n : Nat, 1 ≤ n ∧ n ≤ 10 → (arr.map Prod.fst ++ arr.map Prod.snd).count n = 1)

theorem unique_card_arrangement :
  ∃! arr : List CardPair, is_unique_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_unique_card_arrangement_l540_54033


namespace NUMINAMATH_CALUDE_odd_sum_games_exists_l540_54004

theorem odd_sum_games_exists (n : ℕ) (h : n = 15) : 
  ∃ (i j : ℕ) (games_played : ℕ → ℕ), 
    i < n ∧ j < n ∧ i ≠ j ∧ 
    (games_played i + games_played j) % 2 = 1 ∧
    ∀ k, k < n → games_played k ≤ n - 2 :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_games_exists_l540_54004


namespace NUMINAMATH_CALUDE_ball_game_probabilities_l540_54022

theorem ball_game_probabilities (total : ℕ) (p_white p_red p_yellow : ℚ) 
  (h_total : total = 6)
  (h_white : p_white = 1/2)
  (h_red : p_red = 1/3)
  (h_yellow : p_yellow = 1/6)
  (h_sum : p_white + p_red + p_yellow = 1) :
  ∃ (white red yellow : ℕ),
    white + red + yellow = total ∧
    (white : ℚ) / total = p_white ∧
    (red : ℚ) / total = p_red ∧
    (yellow : ℚ) / total = p_yellow ∧
    white = 3 ∧ red = 2 ∧ yellow = 1 := by
  sorry

end NUMINAMATH_CALUDE_ball_game_probabilities_l540_54022


namespace NUMINAMATH_CALUDE_marble_count_l540_54099

theorem marble_count (allison_marbles angela_marbles albert_marbles : ℕ) : 
  allison_marbles = 28 →
  angela_marbles = allison_marbles + 8 →
  albert_marbles = 3 * angela_marbles →
  albert_marbles + allison_marbles = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_count_l540_54099


namespace NUMINAMATH_CALUDE_parallel_condition_l540_54037

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem parallel_condition (l m : Line) (α : Plane)
  (h1 : ¬ subset l α)
  (h2 : subset m α) :
  (∀ l m, parallel_lines l m → parallel_line_plane l α) ∧
  (∃ l m, parallel_line_plane l α ∧ ¬ parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l540_54037


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l540_54027

theorem four_digit_number_with_specific_remainders :
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧
    n % 7 = 3 ∧
    n % 10 = 6 ∧
    n % 12 = 8 ∧
    n % 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l540_54027


namespace NUMINAMATH_CALUDE_fraction_product_equivalence_l540_54083

theorem fraction_product_equivalence :
  ∀ x : ℝ, x ≠ 1 → ((x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_equivalence_l540_54083


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l540_54001

theorem factorization_difference_of_squares (y : ℝ) : 1 - 4 * y^2 = (1 - 2*y) * (1 + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l540_54001


namespace NUMINAMATH_CALUDE_not_perfect_square_600_sixes_and_zeros_l540_54044

/-- Represents a number with 600 digits of 6 followed by some zeros -/
def number_with_600_sixes_and_zeros (n : ℕ) : ℕ :=
  6 * 10^600 + n

/-- Theorem stating that a number with 600 digits of 6 followed by any number of zeros cannot be a perfect square -/
theorem not_perfect_square_600_sixes_and_zeros (n : ℕ) :
  ∃ (m : ℕ), (number_with_600_sixes_and_zeros n) = m^2 → False :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_600_sixes_and_zeros_l540_54044


namespace NUMINAMATH_CALUDE_base6_addition_subtraction_l540_54080

-- Define a function to convert from base 6 to decimal
def base6ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

-- Define a function to convert from decimal to base 6
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem base6_addition_subtraction :
  let a := [2, 4, 5, 3, 1]  -- 13542₆ in reverse order
  let b := [5, 3, 4, 3, 2]  -- 23435₆ in reverse order
  let c := [2, 1, 3, 4]     -- 4312₆ in reverse order
  let result := [5, 0, 4, 1, 3]  -- 31405₆ in reverse order
  decimalToBase6 ((base6ToDecimal a + base6ToDecimal b) - base6ToDecimal c) = result := by
  sorry


end NUMINAMATH_CALUDE_base6_addition_subtraction_l540_54080


namespace NUMINAMATH_CALUDE_defective_pens_probability_l540_54038

theorem defective_pens_probability (total_pens : Nat) (defective_pens : Nat) (bought_pens : Nat) :
  total_pens = 10 →
  defective_pens = 2 →
  bought_pens = 2 →
  (((total_pens - defective_pens : ℚ) / total_pens) * 
   ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1))) = 0.6222222222222222 := by
  sorry

end NUMINAMATH_CALUDE_defective_pens_probability_l540_54038


namespace NUMINAMATH_CALUDE_seventh_house_number_l540_54095

theorem seventh_house_number (k : ℕ) (p : ℕ) : 
  p = 5 →
  k * (p + k - 1) = 2021 →
  p + 2 * (7 - 1) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_house_number_l540_54095


namespace NUMINAMATH_CALUDE_tangent_lines_through_origin_l540_54043

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

-- Theorem statement
theorem tangent_lines_through_origin :
  ∃ (x₀ : ℝ), 
    (f x₀ = x₀ * (f' x₀)) ∧ 
    ((f' 0 = 2 ∧ f 0 = 0) ∨ 
     (f' x₀ = -1/4 ∧ f x₀ = -1/4 * x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_origin_l540_54043


namespace NUMINAMATH_CALUDE_unique_positive_solution_l540_54049

theorem unique_positive_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∃! x : ℝ, x > 0 ∧ 
    (2 * (a + b) * x + 2 * a * b) / (4 * x + a + b) = ((a^(1/3) + b^(1/3)) / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l540_54049


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_is_four_fifths_l540_54011

/-- The number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball when two balls are randomly drawn -/
def prob_at_least_one_black : ℚ := 4/5

theorem prob_at_least_one_black_is_four_fifths :
  prob_at_least_one_black = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_is_four_fifths_l540_54011


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l540_54047

theorem rectangle_longer_side (r : ℝ) (h1 : r = 6) : ∃ L : ℝ,
  (L * (2 * r) = 3 * (π * r^2)) ∧ L = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l540_54047


namespace NUMINAMATH_CALUDE_library_books_count_l540_54057

/-- The number of books in a library after two years of purchases -/
def total_books (initial : ℕ) (last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial + last_year + multiplier * last_year

/-- Theorem stating the total number of books in the library -/
theorem library_books_count : total_books 100 50 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l540_54057


namespace NUMINAMATH_CALUDE_volume_increase_when_radius_doubled_l540_54073

/-- The volume increase of a right circular cylinder when its radius is doubled -/
theorem volume_increase_when_radius_doubled (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 6 → 
  π * (2*r)^2 * h - π * r^2 * h = 18 := by
  sorry

end NUMINAMATH_CALUDE_volume_increase_when_radius_doubled_l540_54073


namespace NUMINAMATH_CALUDE_eddies_sister_pies_per_day_l540_54041

theorem eddies_sister_pies_per_day :
  let eddie_pies_per_day : ℕ := 3
  let mother_pies_per_day : ℕ := 8
  let total_days : ℕ := 7
  let total_pies : ℕ := 119
  ∃ (sister_pies_per_day : ℕ),
    sister_pies_per_day * total_days + eddie_pies_per_day * total_days + mother_pies_per_day * total_days = total_pies ∧
    sister_pies_per_day = 6 :=
by sorry

end NUMINAMATH_CALUDE_eddies_sister_pies_per_day_l540_54041


namespace NUMINAMATH_CALUDE_square_root_of_36_l540_54034

theorem square_root_of_36 : ∃ x : ℝ, x ^ 2 = 36 ∧ (x = 6 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_36_l540_54034


namespace NUMINAMATH_CALUDE_polynomial_factor_theorem_l540_54003

theorem polynomial_factor_theorem (a b c : ℤ) :
  (∃ d e : ℤ, (X^3 - X^2 - X - 1) * (d * X + e) = a * X^4 + b * X^3 + c * X^2 + 1) →
  c = 1 - a :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_theorem_l540_54003


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l540_54059

theorem complex_fraction_equality (i : ℂ) (h : i^2 = -1) :
  (1 - 2*i) / (2 + i) = -3*i / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l540_54059


namespace NUMINAMATH_CALUDE_strawberry_juice_problem_l540_54051

theorem strawberry_juice_problem (T : ℚ) 
  (h1 : T > 0)
  (h2 : (5/6 * T - 2/5 * (5/6 * T) - 2/3 * (5/6 * T - 2/5 * (5/6 * T))) = 120) : 
  T = 720 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_juice_problem_l540_54051


namespace NUMINAMATH_CALUDE_no_integer_solutions_l540_54020

theorem no_integer_solutions : 
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l540_54020


namespace NUMINAMATH_CALUDE_larger_number_problem_l540_54002

theorem larger_number_problem (x y : ℝ) : 
  x > y → x + y = 40 → x * y = 96 → x = 37.435 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l540_54002


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l540_54082

/-- Given a circle divided into 15 equal arcs, this theorem proves that
    the sum of two inscribed angles, one subtended by 3 arcs and the other by 5 arcs,
    is equal to 96 degrees. -/
theorem inscribed_angles_sum (circle : Real) (x y : Real) : 
  (circle = 360) →
  (x = 3 * (circle / 15) / 2) →
  (y = 5 * (circle / 15) / 2) →
  x + y = 96 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l540_54082


namespace NUMINAMATH_CALUDE_g_sum_neg_one_l540_54042

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the functional equation
axiom func_eq : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y

-- State the condition f(-2) = f(1) ≠ 0
axiom f_cond : f (-2) = f 1 ∧ f 1 ≠ 0

-- Theorem to prove
theorem g_sum_neg_one : g 1 + g (-1) = -1 :=
sorry

end NUMINAMATH_CALUDE_g_sum_neg_one_l540_54042


namespace NUMINAMATH_CALUDE_solve_shooting_stars_l540_54036

def shooting_stars_problem (bridget_count reginald_count sam_count : ℕ) : Prop :=
  bridget_count = 14 ∧
  reginald_count = bridget_count - 2 ∧
  sam_count = reginald_count + 4 ∧
  sam_count - ((bridget_count + reginald_count + sam_count) / 3) = 2

theorem solve_shooting_stars :
  ∃ (bridget_count reginald_count sam_count : ℕ),
    shooting_stars_problem bridget_count reginald_count sam_count :=
by
  sorry

end NUMINAMATH_CALUDE_solve_shooting_stars_l540_54036


namespace NUMINAMATH_CALUDE_central_angle_for_given_sector_l540_54089

/-- A circular sector with given area and perimeter -/
structure CircularSector where
  area : ℝ
  perimeter : ℝ

/-- The central angle of a circular sector in radians -/
def central_angle (s : CircularSector) : ℝ := 
  2 -- We define this as 2, which is what we want to prove

/-- Theorem: For a circular sector with area 1 and perimeter 4, the central angle is 2 radians -/
theorem central_angle_for_given_sector :
  ∀ (s : CircularSector), s.area = 1 ∧ s.perimeter = 4 → central_angle s = 2 := by
  sorry

#check central_angle_for_given_sector

end NUMINAMATH_CALUDE_central_angle_for_given_sector_l540_54089


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l540_54072

theorem complex_fraction_equality : (5 / 2 / (1 / 2) * (5 / 2)) / (5 / 2 * (1 / 2) / (5 / 2)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l540_54072


namespace NUMINAMATH_CALUDE_baker_duration_l540_54035

/-- Represents the number of weeks Steve bakes pies -/
def duration : ℕ := sorry

/-- Number of days per week Steve bakes apple pies -/
def apple_days : ℕ := 3

/-- Number of days per week Steve bakes cherry pies -/
def cherry_days : ℕ := 2

/-- Number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The difference in the number of apple pies and cherry pies -/
def pie_difference : ℕ := 12

theorem baker_duration :
  apple_days * pies_per_day * duration = cherry_days * pies_per_day * duration + pie_difference ∧
  duration = 1 := by sorry

end NUMINAMATH_CALUDE_baker_duration_l540_54035


namespace NUMINAMATH_CALUDE_slope_implies_y_coordinate_l540_54045

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q
    is equal to 5/3, then the y-coordinate of Q is 56/3. -/
theorem slope_implies_y_coordinate :
  ∀ (y : ℚ),
  let P : ℚ × ℚ := (-2, 7)
  let Q : ℚ × ℚ := (5, y)
  let slope : ℚ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 5/3 → y = 56/3 :=
by
  sorry

end NUMINAMATH_CALUDE_slope_implies_y_coordinate_l540_54045


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l540_54070

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_equation :
  ∃ (l : Line),
    perpendicular l (Line.mk 2 (-3) 4) ∧
    point_on_line (-1) 2 l ∧
    l = Line.mk 3 2 (-1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l540_54070


namespace NUMINAMATH_CALUDE_smallest_double_square_triple_cube_l540_54024

theorem smallest_double_square_triple_cube : ∃! k : ℕ, 
  (∃ m : ℕ, k = 2 * m^2) ∧ 
  (∃ n : ℕ, k = 3 * n^3) ∧ 
  (∀ j : ℕ, j < k → ¬(∃ x : ℕ, j = 2 * x^2) ∨ ¬(∃ y : ℕ, j = 3 * y^3)) ∧
  k = 648 := by
sorry

end NUMINAMATH_CALUDE_smallest_double_square_triple_cube_l540_54024


namespace NUMINAMATH_CALUDE_fourth_customer_new_item_probability_l540_54085

/-- The number of menu items --/
def menu_items : ℕ := 5

/-- The number of customers --/
def customers : ℕ := 4

/-- The probability that the 4th customer orders a previously unordered item --/
def probability : ℚ := 32 / 125

theorem fourth_customer_new_item_probability :
  (menu_items ^ (customers - 1) * (menu_items - (customers - 1))) /
  (menu_items ^ customers) = probability := by
  sorry

end NUMINAMATH_CALUDE_fourth_customer_new_item_probability_l540_54085


namespace NUMINAMATH_CALUDE_integer_root_values_l540_54030

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 2*x^2 + b*x + 8 = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-81, -26, -19, -12, -11, 4, 9, 47} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l540_54030


namespace NUMINAMATH_CALUDE_cherry_olive_discount_l540_54096

theorem cherry_olive_discount (cherry_price olives_price bags_count total_cost : ℝ) :
  cherry_price = 5 →
  olives_price = 7 →
  bags_count = 50 →
  total_cost = 540 →
  let original_cost := cherry_price * bags_count + olives_price * bags_count
  let discount_amount := original_cost - total_cost
  let discount_percentage := (discount_amount / original_cost) * 100
  discount_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_cherry_olive_discount_l540_54096


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l540_54006

theorem sum_of_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_sum_product : a + b = 6 * (a * b)) : 1 / a + 1 / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l540_54006


namespace NUMINAMATH_CALUDE_negation_equivalence_l540_54050

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l540_54050


namespace NUMINAMATH_CALUDE_max_divisors_1_to_20_l540_54094

def divisorCount (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem max_divisors_1_to_20 :
  ∃ (max : ℕ) (S : Finset ℕ),
    (∀ n ∈ Finset.range 21, divisorCount n ≤ max) ∧
    (∀ m ∈ S, divisorCount m = max) ∧
    S = {12, 18, 20} ∧
    max = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_divisors_1_to_20_l540_54094


namespace NUMINAMATH_CALUDE_unique_solution_l540_54032

-- Define the digits as natural numbers
def A : ℕ := sorry
def B : ℕ := sorry
def d : ℕ := sorry
def I : ℕ := sorry

-- Define the conditions
axiom digit_constraint : A < 10 ∧ B < 10 ∧ d < 10 ∧ I < 10
axiom equation : 58 * (100 * A + 10 * B + A) = 1000 * I + 100 * d + 10 * B + A

-- State the theorem
theorem unique_solution : d = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l540_54032


namespace NUMINAMATH_CALUDE_expand_equality_l540_54075

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := (binomial 10 2 : ℝ) * x^8 * y^2

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) : ℝ := (binomial 10 3 : ℝ) * x^7 * y^3

-- Main theorem
theorem expand_equality (p q : ℝ) 
  (h_pos_p : p > 0) 
  (h_pos_q : q > 0) 
  (h_sum : p + q = 2) 
  (h_equal : third_term p q = fourth_term p q) : 
  p = 16/11 := by
sorry

end NUMINAMATH_CALUDE_expand_equality_l540_54075


namespace NUMINAMATH_CALUDE_ellipse_m_range_l540_54055

-- Define the equation
def equation (m x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- Define what it means for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), equation m x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

-- State the theorem
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m > 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l540_54055


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l540_54026

theorem arithmetic_calculation : 4 * 6 * 8 - 24 / 6 = 188 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l540_54026


namespace NUMINAMATH_CALUDE_points_coplanar_iff_b_eq_neg_one_l540_54018

/-- Given four points in 3D space, prove they are coplanar iff b = -1 --/
theorem points_coplanar_iff_b_eq_neg_one (b : ℝ) :
  let p1 := (0 : ℝ × ℝ × ℝ)
  let p2 := (1, b, 0)
  let p3 := (0, 1, b^2)
  let p4 := (b^2, 0, 1)
  (∃ (a b c d : ℝ), a • p1 + b • p2 + c • p3 + d • p4 = 0 ∧ (a, b, c, d) ≠ 0) ↔ b = -1 := by
  sorry

#check points_coplanar_iff_b_eq_neg_one

end NUMINAMATH_CALUDE_points_coplanar_iff_b_eq_neg_one_l540_54018


namespace NUMINAMATH_CALUDE_probability_ratio_l540_54007

def total_slips : ℕ := 50
def numbers_per_slip : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 4

def probability_same_number (total : ℕ) (per_number : ℕ) (numbers : ℕ) (drawn : ℕ) : ℚ :=
  (numbers * Nat.choose per_number drawn) / Nat.choose total drawn

def probability_three_same_one_different (total : ℕ) (per_number : ℕ) (numbers : ℕ) (drawn : ℕ) : ℚ :=
  (numbers * Nat.choose per_number (drawn - 1) * (numbers - 1) * per_number) / Nat.choose total drawn

theorem probability_ratio :
  probability_three_same_one_different total_slips slips_per_number numbers_per_slip drawn_slips /
  probability_same_number total_slips slips_per_number numbers_per_slip drawn_slips = 90 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l540_54007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l540_54068

theorem arithmetic_sequence_common_difference :
  let a : ℕ → ℤ := λ n => 2 - 3 * n
  ∀ n : ℕ, a (n + 1) - a n = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l540_54068


namespace NUMINAMATH_CALUDE_lara_flowers_theorem_l540_54040

/-- The number of flowers Lara bought -/
def total_flowers : ℕ := sorry

/-- The number of flowers Lara gave to her mom -/
def flowers_to_mom : ℕ := 15

/-- The number of flowers Lara gave to her grandma -/
def flowers_to_grandma : ℕ := flowers_to_mom + 6

/-- The number of flowers Lara put in the vase -/
def flowers_in_vase : ℕ := 16

/-- Theorem stating the total number of flowers Lara bought -/
theorem lara_flowers_theorem : 
  total_flowers = flowers_to_mom + flowers_to_grandma + flowers_in_vase ∧ 
  total_flowers = 52 := by sorry

end NUMINAMATH_CALUDE_lara_flowers_theorem_l540_54040


namespace NUMINAMATH_CALUDE_min_value_xy_l540_54092

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + 4 * y + 5) :
  x * y ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l540_54092


namespace NUMINAMATH_CALUDE_max_value_theorem_l540_54087

theorem max_value_theorem (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → (k * a + b)^2 / (a^2 + b^2) ≤ (k * x + y)^2 / (x^2 + y^2)) →
  (k * x + y)^2 / (x^2 + y^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l540_54087


namespace NUMINAMATH_CALUDE_unique_paths_in_grid_l540_54048

/-- The number of rows in the grid -/
def rows : ℕ := 6

/-- The number of columns in the grid -/
def cols : ℕ := 6

/-- The number of moves required to reach the bottom-right corner -/
def moves : ℕ := 5

/-- A function that calculates the number of unique paths in the grid -/
def uniquePaths (r : ℕ) (c : ℕ) (m : ℕ) : ℕ := 2^m

/-- Theorem stating that the number of unique paths in the given grid is 32 -/
theorem unique_paths_in_grid : uniquePaths rows cols moves = 32 := by
  sorry

end NUMINAMATH_CALUDE_unique_paths_in_grid_l540_54048


namespace NUMINAMATH_CALUDE_work_together_proof_l540_54088

/-- The number of days after which Alice, Bob, Carol, and Dave work together again -/
def days_until_work_together_again : ℕ := 360

/-- Alice's work schedule (every 5th day) -/
def alice_schedule : ℕ := 5

/-- Bob's work schedule (every 6th day) -/
def bob_schedule : ℕ := 6

/-- Carol's work schedule (every 8th day) -/
def carol_schedule : ℕ := 8

/-- Dave's work schedule (every 9th day) -/
def dave_schedule : ℕ := 9

theorem work_together_proof :
  days_until_work_together_again = Nat.lcm alice_schedule (Nat.lcm bob_schedule (Nat.lcm carol_schedule dave_schedule)) :=
by
  sorry

#eval days_until_work_together_again

end NUMINAMATH_CALUDE_work_together_proof_l540_54088


namespace NUMINAMATH_CALUDE_distance_polynomial_l540_54008

theorem distance_polynomial (m n : ℝ) : 
  ∃ (x y : ℝ), x + y = m ∧ x * y = n^2 ∧ 
  (∀ z : ℝ, z^2 - m*z + n^2 = 0 ↔ (z = x ∨ z = y)) := by
sorry

end NUMINAMATH_CALUDE_distance_polynomial_l540_54008


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l540_54078

def total_amount : ℕ := 5000
def x_amount : ℕ := 1000

theorem ratio_x_to_y :
  (x_amount : ℚ) / (total_amount - x_amount : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l540_54078


namespace NUMINAMATH_CALUDE_circle_properties_l540_54028

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = 3 ∧ center_y = 0 ∧ radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l540_54028


namespace NUMINAMATH_CALUDE_probability_inequalities_l540_54029

open ProbabilityTheory MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] [Fintype Ω]
variable (P : Measure Ω) [IsProbabilityMeasure P]
variable (A B : Set Ω)

theorem probability_inequalities
  (h1 : P A = P (Aᶜ))
  (h2 : P (Bᶜ ∩ A) / P A > P (B ∩ Aᶜ) / P Aᶜ) :
  (P (A ∩ Bᶜ) > P (Aᶜ ∩ B)) ∧ (P (A ∩ B) < P (Aᶜ ∩ Bᶜ)) := by
  sorry

end NUMINAMATH_CALUDE_probability_inequalities_l540_54029
