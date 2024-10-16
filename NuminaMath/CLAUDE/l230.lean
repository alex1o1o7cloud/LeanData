import Mathlib

namespace NUMINAMATH_CALUDE_dot_product_of_complex_vectors_l230_23042

def complex_to_vector (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem dot_product_of_complex_vectors :
  let Z₁ : ℂ := (1 - 2*I)*I
  let Z₂ : ℂ := (1 - 3*I) / (1 - I)
  let a : ℝ × ℝ := complex_to_vector Z₁
  let b : ℝ × ℝ := complex_to_vector Z₂
  (a.1 * b.1 + a.2 * b.2) = 3 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_complex_vectors_l230_23042


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l230_23081

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0

theorem quadratic_inequality_range :
  {a : ℝ | quadratic_inequality a} = Set.Ici 3 ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l230_23081


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_16_l230_23096

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the vertex of the original parabola
def original_vertex : ℝ × ℝ := (3, 4)

-- Define the transformed parabola after all operations
def transformed_parabola (x : ℝ) : ℝ := (x - 8)^2

-- Define the new vertex after transformations
def new_vertex : ℝ × ℝ := (8, 8)

-- Define the zeros of the transformed parabola
def zeros : Set ℝ := {x : ℝ | transformed_parabola x = 0}

-- Theorem statement
theorem sum_of_zeros_is_16 : ∀ p q : ℝ, p ∈ zeros ∧ q ∈ zeros → p + q = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_16_l230_23096


namespace NUMINAMATH_CALUDE_bug_safe_probability_l230_23031

theorem bug_safe_probability (r : ℝ) (h : r = 3) :
  let safe_radius := r - 1
  let total_volume := (4 / 3) * Real.pi * r^3
  let safe_volume := (4 / 3) * Real.pi * safe_radius^3
  safe_volume / total_volume = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_bug_safe_probability_l230_23031


namespace NUMINAMATH_CALUDE_range_of_x_l230_23045

theorem range_of_x (x : ℝ) : 
  (∀ m : ℝ, m ≠ 0 → |5*m - 3| + |3 - 4*m| ≥ |m| * (x - 2/x)) →
  x ∈ Set.Ici (-1) ∪ Set.Ioc 0 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l230_23045


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1365_l230_23057

theorem sum_of_largest_and_smallest_prime_factors_of_1365 :
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ 1365 ∧ 
    q ∣ 1365 ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 1365 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1365_l230_23057


namespace NUMINAMATH_CALUDE_miami_hurricane_damage_l230_23038

/-- Calculates the damage amount in Euros given the damage in US dollars and the exchange rate. -/
def damage_in_euros (damage_usd : ℝ) (exchange_rate : ℝ) : ℝ :=
  damage_usd * exchange_rate

/-- Theorem stating that the damage caused by the hurricane in Miami is 40,500,000 Euros. -/
theorem miami_hurricane_damage :
  let damage_usd : ℝ := 45000000
  let exchange_rate : ℝ := 0.9
  damage_in_euros damage_usd exchange_rate = 40500000 := by
  sorry

end NUMINAMATH_CALUDE_miami_hurricane_damage_l230_23038


namespace NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l230_23029

theorem probability_one_ball_in_last_box (n : ℕ) (h : n = 100) :
  let p := 1 / n
  (n : ℝ) * p * (1 - p)^(n - 1) = ((n - 1 : ℝ) / n)^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_probability_one_ball_in_last_box_l230_23029


namespace NUMINAMATH_CALUDE_nancy_balloons_l230_23066

theorem nancy_balloons (nancy_balloons : ℕ) (mary_balloons : ℕ) : 
  mary_balloons = 28 → mary_balloons = 4 * nancy_balloons → nancy_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_balloons_l230_23066


namespace NUMINAMATH_CALUDE_polynomial_sum_l230_23039

variable (x y : ℝ)
variable (P : ℝ → ℝ → ℝ)

theorem polynomial_sum (h : ∀ x y, P x y + (x^2 - y^2) = x^2 + y^2) :
  ∀ x y, P x y = 2 * y^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l230_23039


namespace NUMINAMATH_CALUDE_equation_solution_l230_23021

theorem equation_solution : ∃ x : ℚ, (x - 75) / 3 = (8 - 3*x) / 4 ∧ x = 324 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l230_23021


namespace NUMINAMATH_CALUDE_green_marble_probability_l230_23095

/-- The probability of selecting a green marble from a basket -/
theorem green_marble_probability :
  let total_marbles : ℕ := 4 + 9 + 5 + 10
  let green_marbles : ℕ := 9
  (green_marbles : ℚ) / total_marbles = 9 / 28 :=
by
  sorry

end NUMINAMATH_CALUDE_green_marble_probability_l230_23095


namespace NUMINAMATH_CALUDE_fibonacci_rectangle_division_l230_23028

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- A rectangle that can be divided into squares -/
structure DivisibleRectangle where
  width : ℕ
  height : ℕ
  num_squares : ℕ
  max_identical_squares : ℕ

/-- Proposition: For every natural number n, there exists a rectangle with 
    dimensions Fn × Fn+1 that can be divided into exactly n squares, 
    with no more than two squares of the same size -/
theorem fibonacci_rectangle_division (n : ℕ) : 
  ∃ (rect : DivisibleRectangle), 
    rect.width = fib n ∧ 
    rect.height = fib (n + 1) ∧ 
    rect.num_squares = n ∧ 
    rect.max_identical_squares ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_rectangle_division_l230_23028


namespace NUMINAMATH_CALUDE_janes_trip_distance_l230_23034

theorem janes_trip_distance :
  ∀ (total_distance : ℝ),
  (1/4 : ℝ) * total_distance +     -- First part (highway)
  30 +                             -- Second part (city streets)
  (1/6 : ℝ) * total_distance       -- Third part (country roads)
  = total_distance                 -- Sum of all parts equals total distance
  →
  total_distance = 360/7 := by
sorry

end NUMINAMATH_CALUDE_janes_trip_distance_l230_23034


namespace NUMINAMATH_CALUDE_simplify_radical_product_l230_23099

theorem simplify_radical_product (q : ℝ) : 
  Real.sqrt (80 * q) * Real.sqrt (45 * q^2) * Real.sqrt (20 * q^3) = 120 * q^3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l230_23099


namespace NUMINAMATH_CALUDE_smallest_d_inequality_l230_23051

theorem smallest_d_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x * y) + (x^2 - y^2)^2 ≥ x + y ∧
  ∀ d : ℝ, d > 0 → d < 1 →
    ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + d * (x^2 - y^2)^2 < x + y :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_inequality_l230_23051


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l230_23027

/-- 
Given a circular arrangement of students, if the 6th position 
is exactly opposite to the 16th position, then there are 22 students in total.
-/
theorem circular_seating_arrangement (n : ℕ) : 
  (6 + n / 2 ≡ 16 [MOD n]) → n = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l230_23027


namespace NUMINAMATH_CALUDE_banana_group_size_l230_23058

def total_bananas : ℕ := 290
def banana_groups : ℕ := 2
def total_oranges : ℕ := 87
def orange_groups : ℕ := 93

theorem banana_group_size : total_bananas / banana_groups = 145 := by
  sorry

end NUMINAMATH_CALUDE_banana_group_size_l230_23058


namespace NUMINAMATH_CALUDE_fourth_person_height_l230_23076

/-- Proves that the height of the fourth person is 82 inches given the conditions -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧  -- Heights in increasing order
  h₂ = h₁ + 2 ∧                  -- Difference between 1st and 2nd
  h₃ = h₂ + 2 ∧                  -- Difference between 2nd and 3rd
  h₄ = h₃ + 6 ∧                  -- Difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76   -- Average height
  → h₄ = 82 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l230_23076


namespace NUMINAMATH_CALUDE_remainder_theorem_l230_23071

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l230_23071


namespace NUMINAMATH_CALUDE_rotating_ngon_path_area_theorem_l230_23075

/-- Represents a regular n-gon -/
structure RegularNGon where
  n : ℕ
  sideLength : ℝ

/-- The area enclosed by the path of a rotating n-gon vertex -/
def rotatingNGonPathArea (g : RegularNGon) : ℝ := sorry

/-- The area of a regular n-gon -/
def regularNGonArea (g : RegularNGon) : ℝ := sorry

/-- Theorem: The area enclosed by the rotating n-gon vertex path
    equals four times the area of the original n-gon -/
theorem rotating_ngon_path_area_theorem (g : RegularNGon) 
    (h1 : g.sideLength = 1) :
  rotatingNGonPathArea g = 4 * regularNGonArea g := by
  sorry

end NUMINAMATH_CALUDE_rotating_ngon_path_area_theorem_l230_23075


namespace NUMINAMATH_CALUDE_max_b_in_box_l230_23048

theorem max_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) → (c < b) → (b < a) →
  b ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_in_box_l230_23048


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_cube_l230_23037

theorem power_of_two_greater_than_cube (n : ℕ) (h : n ≥ 10) : 2^n > n^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_cube_l230_23037


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l230_23059

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/2))^(1/4) = x^(5/8) := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l230_23059


namespace NUMINAMATH_CALUDE_f_properties_l230_23050

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + k / x

theorem f_properties (k : ℝ) (h_k : k ≠ 0) (h_f3 : f k 3 = 6) :
  (∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ -3 → f k x₁ < f k x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l230_23050


namespace NUMINAMATH_CALUDE_triangle_inequality_l230_23026

theorem triangle_inequality (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l230_23026


namespace NUMINAMATH_CALUDE_enlarged_poster_height_l230_23060

/-- Calculates the new height of a proportionally enlarged poster -/
def new_poster_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem: The new height of the enlarged poster is 10 inches -/
theorem enlarged_poster_height :
  new_poster_height 3 2 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_enlarged_poster_height_l230_23060


namespace NUMINAMATH_CALUDE_computer_upgrade_cost_l230_23068

/-- Calculates the total money spent on a computer after replacing a video card -/
def totalSpent (initialCost oldCardSale newCardPrice : ℕ) : ℕ :=
  initialCost + (newCardPrice - oldCardSale)

theorem computer_upgrade_cost :
  totalSpent 1200 300 500 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_computer_upgrade_cost_l230_23068


namespace NUMINAMATH_CALUDE_highway_length_l230_23000

theorem highway_length (speed1 speed2 time : ℝ) 
  (h1 : speed1 = 14)
  (h2 : speed2 = 16)
  (h3 : time = 1.5)
  : speed1 * time + speed2 * time = 45 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l230_23000


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l230_23015

theorem quiz_score_theorem :
  ∀ (correct : ℕ),
  correct ≤ 15 →
  6 * correct - 2 * (15 - correct) ≥ 75 →
  correct ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_score_theorem_l230_23015


namespace NUMINAMATH_CALUDE_max_value_abc_l230_23056

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  ∃ (max : ℝ), max = 8 ∧ ∀ (x : ℝ), x = a + b^2 + c^3 → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l230_23056


namespace NUMINAMATH_CALUDE_absolute_value_counterexample_l230_23090

theorem absolute_value_counterexample : ∃ a : ℝ, |a| ≤ -a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_counterexample_l230_23090


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l230_23020

theorem complex_magnitude_proof : Complex.abs (7/4 - 3*I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l230_23020


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l230_23044

theorem opposite_sides_line_range (a : ℝ) : 
  (0 - a) * (2 - a) < 0 ↔ 0 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l230_23044


namespace NUMINAMATH_CALUDE_units_digit_of_power_difference_l230_23041

theorem units_digit_of_power_difference : (5^35 - 6^21) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_difference_l230_23041


namespace NUMINAMATH_CALUDE_bananas_arrangements_eq_240_l230_23004

/-- The number of letters in the word BANANAS -/
def total_letters : ℕ := 7

/-- The number of 'B's in BANANAS -/
def count_B : ℕ := 1

/-- The number of 'A's in BANANAS -/
def count_A : ℕ := 3

/-- The number of 'N's in BANANAS -/
def count_N : ℕ := 1

/-- The number of 'S's in BANANAS -/
def count_S : ℕ := 2

/-- The function to calculate the number of arrangements of BANANAS with no 'A' at the first position -/
def bananas_arrangements : ℕ := sorry

/-- Theorem stating that the number of arrangements of BANANAS with no 'A' at the first position is 240 -/
theorem bananas_arrangements_eq_240 : bananas_arrangements = 240 := by sorry

end NUMINAMATH_CALUDE_bananas_arrangements_eq_240_l230_23004


namespace NUMINAMATH_CALUDE_solve_equation_l230_23025

theorem solve_equation (y : ℝ) : 7 - y = 10 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l230_23025


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l230_23061

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ ^ 2 - Real.sin θ ^ 2) = 0

/-- The Cartesian equation of two intersecting straight lines -/
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 = y^2

/-- Theorem stating that the polar equation represents two intersecting straight lines -/
theorem polar_to_cartesian :
  ∀ x y : ℝ, ∃ ρ θ : ℝ, polar_equation ρ θ ↔ cartesian_equation x y :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l230_23061


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_range_l230_23080

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The statement that f has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- The main theorem: if f has three distinct zeros, then -2 < a < 2 -/
theorem three_zeros_implies_a_range (a : ℝ) :
  has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_implies_a_range_l230_23080


namespace NUMINAMATH_CALUDE_water_percentage_in_mixture_l230_23046

/-- Given two liquids with different water percentages, prove the water percentage in their mixture -/
theorem water_percentage_in_mixture 
  (water_percent_1 water_percent_2 : ℝ) 
  (parts_1 parts_2 : ℝ) 
  (h1 : water_percent_1 = 20)
  (h2 : water_percent_2 = 35)
  (h3 : parts_1 = 10)
  (h4 : parts_2 = 4) :
  (water_percent_1 / 100 * parts_1 + water_percent_2 / 100 * parts_2) / (parts_1 + parts_2) * 100 =
  (0.2 * 10 + 0.35 * 4) / (10 + 4) * 100 := by
  sorry

#eval (0.2 * 10 + 0.35 * 4) / (10 + 4) * 100

end NUMINAMATH_CALUDE_water_percentage_in_mixture_l230_23046


namespace NUMINAMATH_CALUDE_remainder_3_pow_19_mod_10_l230_23036

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_19_mod_10_l230_23036


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l230_23072

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  (n : ℕ) + n.choose 2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l230_23072


namespace NUMINAMATH_CALUDE_last_box_weight_l230_23062

theorem last_box_weight (box1_weight box2_weight total_weight : ℕ) : 
  box1_weight = 2 → 
  box2_weight = 11 → 
  total_weight = 18 → 
  ∃ last_box_weight : ℕ, last_box_weight = total_weight - (box1_weight + box2_weight) ∧ 
                           last_box_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_box_weight_l230_23062


namespace NUMINAMATH_CALUDE_right_triangle_area_l230_23063

theorem right_triangle_area (base hypotenuse : ℝ) (h_right_angle : base > 0 ∧ hypotenuse > 0 ∧ hypotenuse > base) :
  base = 12 → hypotenuse = 13 → 
  ∃ height : ℝ, height > 0 ∧ height ^ 2 + base ^ 2 = hypotenuse ^ 2 ∧ 
  (1 / 2) * base * height = 30 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l230_23063


namespace NUMINAMATH_CALUDE_sum_divisors_450_prime_factors_l230_23006

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive divisors of 450 has exactly 3 distinct prime factors -/
theorem sum_divisors_450_prime_factors :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end NUMINAMATH_CALUDE_sum_divisors_450_prime_factors_l230_23006


namespace NUMINAMATH_CALUDE_expand_product_l230_23077

theorem expand_product (x : ℝ) : (x + 4) * (x^2 - 9) = x^3 + 4*x^2 - 9*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l230_23077


namespace NUMINAMATH_CALUDE_winter_uniform_count_l230_23067

/-- The number of packages of winter uniforms delivered -/
def num_packages : ℕ := 10

/-- The number of dozens per package -/
def dozens_per_package : ℕ := 10

/-- The number of sets per dozen -/
def sets_per_dozen : ℕ := 12

/-- The total number of winter uniform sets -/
def total_sets : ℕ := num_packages * dozens_per_package * sets_per_dozen

theorem winter_uniform_count : total_sets = 1200 := by
  sorry

end NUMINAMATH_CALUDE_winter_uniform_count_l230_23067


namespace NUMINAMATH_CALUDE_segment_length_is_70_l230_23009

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  position : ℝ
  h1 : A ≤ position
  h2 : position ≤ B

/-- The line segment AB -/
def lineSegment (A B : ℝ) := {x : ℝ | A ≤ x ∧ x ≤ B}

theorem segment_length_is_70 
  (A B : ℝ) 
  (P Q : PointOnSegment A B) 
  (h_order : P.position < Q.position) 
  (h_same_side : (P.position - A) / (B - A) < 1/2 ∧ (Q.position - A) / (B - A) < 1/2) 
  (h_P_ratio : (P.position - A) / (B - P.position) = 2/3) 
  (h_Q_ratio : (Q.position - A) / (B - Q.position) = 3/4) 
  (h_PQ_length : Q.position - P.position = 2) :
  B - A = 70 := by
  sorry

#check segment_length_is_70

end NUMINAMATH_CALUDE_segment_length_is_70_l230_23009


namespace NUMINAMATH_CALUDE_equiangular_and_equilateral_implies_regular_polygon_l230_23089

/-- A figure is equiangular if all its angles are equal. -/
def IsEquiangular (figure : Type) : Prop := sorry

/-- A figure is equilateral if all its sides are equal. -/
def IsEquilateral (figure : Type) : Prop := sorry

/-- A figure is a regular polygon if it is both equiangular and equilateral. -/
def IsRegularPolygon (figure : Type) : Prop := 
  IsEquiangular figure ∧ IsEquilateral figure

/-- Theorem: If a figure is both equiangular and equilateral, then it is a regular polygon. -/
theorem equiangular_and_equilateral_implies_regular_polygon 
  (figure : Type) 
  (h1 : IsEquiangular figure) 
  (h2 : IsEquilateral figure) : 
  IsRegularPolygon figure := by
  sorry


end NUMINAMATH_CALUDE_equiangular_and_equilateral_implies_regular_polygon_l230_23089


namespace NUMINAMATH_CALUDE_completing_square_min_value_compare_expressions_l230_23070

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Theorem 1: Completing the square
theorem completing_square : ∀ x : ℝ, f x = (x - 2)^2 + 2 := by sorry

-- Theorem 2: Minimum value and corresponding x
theorem min_value : 
  (∃ x_min : ℝ, ∀ x : ℝ, f x ≥ f x_min) ∧
  (∃ x_min : ℝ, f x_min = 2) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f x = 2 → x = x_min) ∧
  (∃ x_min : ℝ, x_min = 2) := by sorry

-- Theorem 3: Comparison of two expressions
theorem compare_expressions : ∀ x : ℝ, x^2 - 1 > 2*x - 3 := by sorry

end NUMINAMATH_CALUDE_completing_square_min_value_compare_expressions_l230_23070


namespace NUMINAMATH_CALUDE_dog_cord_length_l230_23012

/-- The maximum radius of the semi-circular path -/
def max_radius : ℝ := 5

/-- The approximate arc length of the semi-circular path -/
def arc_length : ℝ := 30

/-- The length of the nylon cord -/
def cord_length : ℝ := max_radius

theorem dog_cord_length :
  cord_length = max_radius := by sorry

end NUMINAMATH_CALUDE_dog_cord_length_l230_23012


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l230_23073

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -14 ∧ 
  (∀ (y : ℤ), y < 0 → 26 * y + 8 ≡ 4 [ZMOD 18] → y ≤ x) ∧
  (26 * x + 8 ≡ 4 [ZMOD 18]) := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l230_23073


namespace NUMINAMATH_CALUDE_impossible_to_get_50_51_l230_23013

/-- Represents the operation of replacing consecutive numbers with their count -/
def replace_with_count (s : List ℕ) (start : ℕ) (len : ℕ) : List ℕ := sorry

/-- Checks if a list contains only the numbers 50 and 51 -/
def contains_only_50_51 (s : List ℕ) : Prop := sorry

/-- The initial sequence of numbers from 1 to 100 -/
def initial_sequence : List ℕ := List.range 100

/-- Represents the result of applying the operation multiple times -/
def apply_operations (s : List ℕ) : List ℕ := sorry

theorem impossible_to_get_50_51 :
  ¬∃ (result : List ℕ), (apply_operations initial_sequence = result) ∧ (contains_only_50_51 result) := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_get_50_51_l230_23013


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l230_23055

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + b = 0 → x₂^2 - 2*x₂ + b = 0 → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l230_23055


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l230_23084

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = 5 ∨ x = 6) :
  c / d = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l230_23084


namespace NUMINAMATH_CALUDE_equivalent_transitive_l230_23085

def IsGreat (f : ℕ → ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1) (n + 1) * f m n - f (m + 1) n * f m (n + 1) = 1

def Equivalent (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ → ℕ → ℤ, IsGreat f ∧ (∀ n : ℕ, f n 0 = A n ∧ f 0 n = B n)

theorem equivalent_transitive :
  ∀ A B C D : ℕ → ℤ,
    Equivalent A B → Equivalent B C → Equivalent C D → Equivalent D A :=
by sorry

end NUMINAMATH_CALUDE_equivalent_transitive_l230_23085


namespace NUMINAMATH_CALUDE_system_solution_l230_23092

theorem system_solution : 
  ∀ (x y : ℝ), (x^2 + y^3 = x + 1 ∧ x^3 + y^2 = y + 1) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l230_23092


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l230_23007

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -99 + 64*I ∧ (5 + 8*I)^2 = -99 + 64*I → z = 5 + 8*I ∨ z = -5 - 8*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l230_23007


namespace NUMINAMATH_CALUDE_function_properties_l230_23086

/-- The function f(x) defined on the real line -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x + b

theorem function_properties (a b : ℝ) (h_a : a > 0) 
  (h_max : ∀ x ∈ Set.Icc 0 1, f a b x ≤ 1) 
  (h_max_exists : ∃ x ∈ Set.Icc 0 1, f a b x = 1)
  (h_min : ∀ x ∈ Set.Icc 0 1, f a b x ≥ -2) 
  (h_min_exists : ∃ x ∈ Set.Icc 0 1, f a b x = -2) :
  (a = 1 ∧ b = 1) ∧ 
  (∀ m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a b x > -x + m) ↔ m < -1) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l230_23086


namespace NUMINAMATH_CALUDE_partition_seven_students_l230_23074

/-- The number of ways to partition 7 students into groups of 2 or 3 -/
def partition_ways : ℕ := 105

/-- The number of students -/
def num_students : ℕ := 7

/-- The possible group sizes -/
def group_sizes : List ℕ := [2, 3]

/-- Theorem stating that the number of ways to partition 7 students into groups of 2 or 3 is 105 -/
theorem partition_seven_students :
  (∀ g ∈ group_sizes, g ≤ num_students) →
  (∃ f : List ℕ, (∀ x ∈ f, x ∈ group_sizes) ∧ f.sum = num_students) →
  partition_ways = 105 := by
  sorry

end NUMINAMATH_CALUDE_partition_seven_students_l230_23074


namespace NUMINAMATH_CALUDE_rohan_entertainment_expenses_l230_23097

/-- Proves that Rohan's entertainment expenses are 10% of his salary -/
theorem rohan_entertainment_expenses :
  let salary : ℝ := 7500
  let food_percent : ℝ := 40
  let rent_percent : ℝ := 20
  let conveyance_percent : ℝ := 10
  let savings : ℝ := 1500
  let entertainment_percent : ℝ := 100 - (food_percent + rent_percent + conveyance_percent + (savings / salary * 100))
  entertainment_percent = 10 := by
  sorry

end NUMINAMATH_CALUDE_rohan_entertainment_expenses_l230_23097


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l230_23011

theorem sum_remainder_mod_nine : 
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l230_23011


namespace NUMINAMATH_CALUDE_workshop_workers_l230_23064

theorem workshop_workers (total_average : ℕ) (technician_average : ℕ) (other_average : ℕ) 
  (technician_count : ℕ) :
  total_average = 8000 →
  technician_average = 16000 →
  other_average = 6000 →
  technician_count = 7 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = 
      technician_count * technician_average + (total_workers - technician_count) * other_average ∧
    total_workers = 35 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l230_23064


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l230_23016

/-- Given a circle with equation x² + y² - 4x + 2y + 2 = 0, 
    its center is at (2, -1) and its radius is √3. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧ 
    radius = Real.sqrt 3 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l230_23016


namespace NUMINAMATH_CALUDE_quadratic_solution_l230_23088

theorem quadratic_solution (b : ℚ) : 
  ((-4 : ℚ)^2 + b * (-4) - 45 = 0) → b = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l230_23088


namespace NUMINAMATH_CALUDE_number_classification_l230_23091

def numbers : List ℝ := [4, -2.5, -12, 3.14159, 0, 0.4]

def is_negative (x : ℝ) : Prop := x < 0
def is_non_negative (x : ℝ) : Prop := x ≥ 0
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℝ) : Prop := ¬(is_integer x)

theorem number_classification :
  (∀ x ∈ numbers, is_negative x ↔ (x = -2.5 ∨ x = -12)) ∧
  (∀ x ∈ numbers, is_non_negative x ↔ (x = 4 ∨ x = 3.14159 ∨ x = 0 ∨ x = 0.4)) ∧
  (∀ x ∈ numbers, is_integer x ↔ (x = 4 ∨ x = -12 ∨ x = 0)) ∧
  (∀ x ∈ numbers, is_fraction x ↔ (x = -2.5 ∨ x = 3.14159 ∨ x = 0.4)) :=
by sorry


end NUMINAMATH_CALUDE_number_classification_l230_23091


namespace NUMINAMATH_CALUDE_determine_m_l230_23083

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + m
def g (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*m

-- State the theorem
theorem determine_m : ∃ m : ℝ, 2 * (f m 3) = 3 * (g m 3) ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_determine_m_l230_23083


namespace NUMINAMATH_CALUDE_goldfish_count_l230_23014

/-- Given that 25% of goldfish are at the surface and 75% are below the surface,
    with 45 goldfish below the surface, prove that there are 15 goldfish at the surface. -/
theorem goldfish_count (surface_percent : ℝ) (below_percent : ℝ) (below_count : ℕ) :
  surface_percent = 25 →
  below_percent = 75 →
  below_count = 45 →
  ↑below_count / below_percent * surface_percent = 15 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_count_l230_23014


namespace NUMINAMATH_CALUDE_arithmetic_progression_formula_geometric_progression_formula_l230_23079

-- Arithmetic Progression
def arithmeticProgression (u₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := u₁ + (n - 1 : ℝ) * d

-- Geometric Progression
def geometricProgression (u₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := u₁ * q ^ (n - 1)

theorem arithmetic_progression_formula (u₁ d : ℝ) (n : ℕ) :
  ∀ k : ℕ, k ≤ n → arithmeticProgression u₁ d k = u₁ + (k - 1 : ℝ) * d :=
by sorry

theorem geometric_progression_formula (u₁ q : ℝ) (n : ℕ) :
  ∀ k : ℕ, k ≤ n → geometricProgression u₁ q k = u₁ * q ^ (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_formula_geometric_progression_formula_l230_23079


namespace NUMINAMATH_CALUDE_game_points_total_l230_23030

theorem game_points_total (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  samanta_points + mark_points + eric_points = 32 := by
sorry

end NUMINAMATH_CALUDE_game_points_total_l230_23030


namespace NUMINAMATH_CALUDE_ship_age_conversion_l230_23087

/-- Converts an octal number represented as (a, b, c) to its decimal equivalent -/
def octal_to_decimal (a b c : ℕ) : ℕ := c * 8^2 + b * 8^1 + a * 8^0

/-- The age of the sunken pirate ship in octal -/
def ship_age_octal : ℕ × ℕ × ℕ := (7, 4, 2)

theorem ship_age_conversion :
  octal_to_decimal ship_age_octal.1 ship_age_octal.2.1 ship_age_octal.2.2 = 482 := by
  sorry

end NUMINAMATH_CALUDE_ship_age_conversion_l230_23087


namespace NUMINAMATH_CALUDE_distance_eq_speed_times_time_l230_23010

/-- The distance between Martin's house and Lawrence's house -/
def distance : ℝ := 12

/-- The time Martin spent walking -/
def time : ℝ := 6

/-- Martin's walking speed -/
def speed : ℝ := 2

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_eq_speed_times_time : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_distance_eq_speed_times_time_l230_23010


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l230_23035

theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 →
  x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l230_23035


namespace NUMINAMATH_CALUDE_cubic_inverse_exists_l230_23094

noncomputable def k : ℝ := Real.sqrt 3

theorem cubic_inverse_exists (x y z : ℚ) (h : x + y * k + z * k^2 ≠ 0) :
  ∃ u v w : ℚ, (x + y * k + z * k^2) * (u + v * k + w * k^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inverse_exists_l230_23094


namespace NUMINAMATH_CALUDE_min_value_3x_4y_l230_23053

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = x * y) :
  ∃ (m : ℝ), m = 28 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 3 * y' = x' * y' → 3 * x' + 4 * y' ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_4y_l230_23053


namespace NUMINAMATH_CALUDE_triangle_abc_exists_l230_23098

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  /-- Side length opposite to angle A -/
  a : ℝ
  /-- Side length opposite to angle B -/
  b : ℝ
  /-- Length of angle bisector from angle C -/
  l_c : ℝ
  /-- Measure of angle A in radians -/
  angle_A : ℝ
  /-- Height to side a -/
  h_a : ℝ
  /-- Perimeter of the triangle -/
  p : ℝ

/-- Theorem stating the existence of a triangle with given properties -/
theorem triangle_abc_exists (a b l_c angle_A h_a p : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : l_c > 0) 
  (h4 : 0 < angle_A ∧ angle_A < π) 
  (h5 : h_a > 0) (h6 : p > 0) :
  ∃ (t : TriangleABC), t.a = a ∧ t.b = b ∧ t.l_c = l_c ∧ 
    t.angle_A = angle_A ∧ t.h_a = h_a ∧ t.p = p :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_exists_l230_23098


namespace NUMINAMATH_CALUDE_twentieth_term_is_400_l230_23019

/-- A second-order arithmetic sequence -/
def SecondOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) - (a (n-1) - a (n-2)) = 2

/-- The sequence starts with 1, 4, 9, 16 -/
def SequenceStart (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 4 ∧ a 3 = 9 ∧ a 4 = 16

theorem twentieth_term_is_400 (a : ℕ → ℕ) 
  (h1 : SecondOrderArithmeticSequence a) 
  (h2 : SequenceStart a) : 
  a 20 = 400 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_400_l230_23019


namespace NUMINAMATH_CALUDE_y_derivative_l230_23002

noncomputable def y (x : ℝ) : ℝ :=
  x - Real.log (1 + Real.exp x) - 2 * Real.exp (-x/2) * Real.arctan (Real.exp (x/2)) - (Real.arctan (Real.exp (x/2)))^2

theorem y_derivative (x : ℝ) :
  deriv y x = Real.arctan (Real.exp (x/2)) / (Real.exp (x/2) * (1 + Real.exp x)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l230_23002


namespace NUMINAMATH_CALUDE_circle_passes_through_intersections_l230_23024

/-- Line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 2*y = 0

/-- Line l₂ -/
def l₂ (x y : ℝ) : Prop := y + 1 = 0

/-- Line l₃ -/
def l₃ (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + x + 2*y - 1 = 0

/-- Theorem stating that the circle passes through the intersection points of the lines -/
theorem circle_passes_through_intersections :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
  (l₁ x₁ y₁ ∧ l₂ x₁ y₁) →
  (l₁ x₂ y₂ ∧ l₃ x₂ y₂) →
  (l₂ x₃ y₃ ∧ l₃ x₃ y₃) →
  circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧ circle_equation x₃ y₃ :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_intersections_l230_23024


namespace NUMINAMATH_CALUDE_polynomial_composition_positive_l230_23043

/-- A quadratic polynomial with real coefficients -/
def P (r s x : ℝ) : ℝ := x^2 + r*x + s

theorem polynomial_composition_positive
  (r s : ℝ)
  (h1 : ∃ a b : ℝ, a < b ∧ b < -1 ∧ P r s a = 0 ∧ P r s b = 0)
  (h2 : ∃ a b : ℝ, P r s a = 0 ∧ P r s b = 0 ∧ 0 < b - a ∧ b - a < 2) :
  ∀ x : ℝ, P r s (P r s x) > 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_composition_positive_l230_23043


namespace NUMINAMATH_CALUDE_triangle_side_length_l230_23023

theorem triangle_side_length (a b c : ℝ) (area : ℝ) : 
  a = 1 → b = Real.sqrt 7 → area = Real.sqrt 3 / 2 → 
  c = 2 ∨ c = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l230_23023


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l230_23003

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (0 < x ∧ x < 5) → (-5 < x - 2 ∧ x - 2 < 5)) ∧
  (∃ x : ℝ, (-5 < x - 2 ∧ x - 2 < 5) ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l230_23003


namespace NUMINAMATH_CALUDE_elderly_selected_in_scenario_l230_23069

/-- Represents the number of elderly people selected in a stratified sampling -/
def elderly_selected (total_population : ℕ) (elderly_population : ℕ) (sample_size : ℕ) : ℚ :=
  (sample_size : ℚ) * (elderly_population : ℚ) / (total_population : ℚ)

/-- Theorem stating the number of elderly people selected in the given scenario -/
theorem elderly_selected_in_scenario : 
  elderly_selected 100 60 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_elderly_selected_in_scenario_l230_23069


namespace NUMINAMATH_CALUDE_polynomial_factorization_l230_23093

theorem polynomial_factorization (x : ℝ) : 
  x^12 + x^6 + 1 = (x^2 + 1) * (x^4 - x^2 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l230_23093


namespace NUMINAMATH_CALUDE_expected_cells_theorem_l230_23018

/-- A grid of height 2 stretching infinitely in one direction -/
def Grid := ℕ × Fin 2

/-- Probability of a door being locked -/
def p_locked : ℚ := 1/2

/-- Philip's starting position -/
def start : Grid := (0, 0)

/-- Expected number of reachable cells -/
noncomputable def expected_reachable_cells : ℚ := 32/7

/-- Main theorem: The expected number of cells Philip can reach is 32/7 -/
theorem expected_cells_theorem :
  let grid : Type := Grid
  let p_lock : ℚ := p_locked
  let start_pos : grid := start
  expected_reachable_cells = 32/7 := by
  sorry

end NUMINAMATH_CALUDE_expected_cells_theorem_l230_23018


namespace NUMINAMATH_CALUDE_haley_magazines_l230_23040

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) 
  (h1 : boxes = 7) (h2 : magazines_per_box = 9) : 
  boxes * magazines_per_box = 63 := by
  sorry

end NUMINAMATH_CALUDE_haley_magazines_l230_23040


namespace NUMINAMATH_CALUDE_flour_already_added_is_three_l230_23017

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 9

/-- The number of cups of flour Mary still needs to add -/
def flour_to_add : ℕ := 6

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := total_flour - flour_to_add

theorem flour_already_added_is_three : flour_already_added = 3 := by
  sorry

end NUMINAMATH_CALUDE_flour_already_added_is_three_l230_23017


namespace NUMINAMATH_CALUDE_max_value_of_expression_l230_23008

theorem max_value_of_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 64) + Real.sqrt (20 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 285.72 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l230_23008


namespace NUMINAMATH_CALUDE_square_difference_equality_l230_23005

theorem square_difference_equality : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l230_23005


namespace NUMINAMATH_CALUDE_range_of_a_l230_23052

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) → 
  a ∈ Set.Icc (-5/4 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l230_23052


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l230_23032

theorem simplify_sqrt_expression :
  Real.sqrt 8 - Real.sqrt 50 + Real.sqrt 72 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l230_23032


namespace NUMINAMATH_CALUDE_compound_interest_years_l230_23049

/-- Compound interest calculation --/
theorem compound_interest_years (P : ℝ) (r : ℝ) (CI : ℝ) (n : ℕ) : 
  P > 0 → r > 0 → CI > 0 → n > 0 →
  let A := P + CI
  let t := Real.log (A / P) / Real.log (1 + r / n)
  P = 1200 → r = 0.20 → n = 1 → CI = 873.60 →
  ⌈t⌉ = 3 := by sorry

#check compound_interest_years

end NUMINAMATH_CALUDE_compound_interest_years_l230_23049


namespace NUMINAMATH_CALUDE_courtyard_paving_l230_23001

/-- Given a rectangular courtyard and rectangular bricks, calculate the number of bricks needed to pave the courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℕ) 
  (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16)
  (h3 : brick_length = 20)
  (h4 : brick_width = 10) :
  (courtyard_length * 100) * (courtyard_width * 100) / (brick_length * brick_width) = 20000 := by
  sorry

#check courtyard_paving

end NUMINAMATH_CALUDE_courtyard_paving_l230_23001


namespace NUMINAMATH_CALUDE_larger_number_proof_l230_23078

theorem larger_number_proof (x y : ℝ) 
  (h1 : x - y = 1860)
  (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l230_23078


namespace NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l230_23082

theorem greatest_difference_of_units_digit (x : ℕ) : 
  x < 10 →
  (720 + x) % 4 = 0 →
  ∃ y z, y < 10 ∧ z < 10 ∧ 
         (720 + y) % 4 = 0 ∧ 
         (720 + z) % 4 = 0 ∧ 
         y - z ≤ 8 ∧
         ∀ w, w < 10 → (720 + w) % 4 = 0 → y - w ≤ 8 ∧ w - z ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_of_units_digit_l230_23082


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l230_23065

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (sides : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 2 ∧
  sides = n ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  sides * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l230_23065


namespace NUMINAMATH_CALUDE_ap_sum_100_l230_23033

/-- Given an arithmetic progression where:
    - The sum of the first 15 terms is 45
    - The sum of the first 85 terms is 255
    This theorem proves that the sum of the first 100 terms is 300. -/
theorem ap_sum_100 (a d : ℝ) 
  (sum_15 : (15 : ℝ) / 2 * (2 * a + (15 - 1) * d) = 45)
  (sum_85 : (85 : ℝ) / 2 * (2 * a + (85 - 1) * d) = 255) :
  (100 : ℝ) / 2 * (2 * a + (100 - 1) * d) = 300 :=
by sorry

end NUMINAMATH_CALUDE_ap_sum_100_l230_23033


namespace NUMINAMATH_CALUDE_vector_sum_equality_l230_23022

/-- Given two 2D vectors a and b, prove that 2a + 3b equals the specified result. -/
theorem vector_sum_equality (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-1, 2) →
  2 • a + 3 • b = (1, 8) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l230_23022


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_minus_one_l230_23047

theorem roots_sum_reciprocal_minus_one (b c : ℝ) : 
  b^2 - b - 1 = 0 → c^2 - c - 1 = 0 → b ≠ c → 1 / (1 - b) + 1 / (1 - c) = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_minus_one_l230_23047


namespace NUMINAMATH_CALUDE_factor_expression_l230_23054

theorem factor_expression (m n : ℝ) : 3 * m^2 - 6 * m * n + 3 * n^2 = 3 * (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l230_23054
