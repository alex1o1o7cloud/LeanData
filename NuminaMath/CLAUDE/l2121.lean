import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2121_212194

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (4 * a^3 + 2023 * a + 4012 = 0) ∧ 
  (4 * b^3 + 2023 * b + 4012 = 0) ∧ 
  (4 * c^3 + 2023 * c + 4012 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 3009 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2121_212194


namespace NUMINAMATH_CALUDE_min_dials_for_lighting_l2121_212100

/-- A regular 12-sided polygon dial with numbers from 1 to 12 -/
structure Dial :=
  (numbers : Fin 12 → Fin 12)

/-- A stack of dials -/
def DialStack := List Dial

/-- The sum of numbers in a column of the dial stack -/
def columnSum (stack : DialStack) (column : Fin 12) : ℕ :=
  stack.foldr (λ dial acc => acc + dial.numbers column) 0

/-- Predicate for when the Christmas tree lights up -/
def lightsUp (stack : DialStack) : Prop :=
  ∀ i j : Fin 12, columnSum stack i % 12 = columnSum stack j % 12

/-- The theorem stating the minimum number of dials required -/
theorem min_dials_for_lighting : 
  ∀ n : ℕ, (∃ stack : DialStack, stack.length = n ∧ lightsUp stack) → n ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_dials_for_lighting_l2121_212100


namespace NUMINAMATH_CALUDE_equal_digit_probability_l2121_212129

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of one-digit outcomes on a die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on a die -/
def two_digit_outcomes : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers with 5 20-sided dice -/
theorem equal_digit_probability : 
  (Nat.choose num_dice (num_dice / 2) *
   (one_digit_outcomes ^ (num_dice / 2) * two_digit_outcomes ^ (num_dice - num_dice / 2))) /
  (num_sides ^ num_dice) = 539055 / 1600000 := by
sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l2121_212129


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l2121_212142

theorem dvd_pack_cost (total_amount : ℕ) (num_packs : ℕ) (h1 : total_amount = 132) (h2 : num_packs = 11) :
  total_amount / num_packs = 12 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l2121_212142


namespace NUMINAMATH_CALUDE_unique_function_l2121_212116

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x + 1/x else 0

theorem unique_function (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x : ℝ, f (2*x) = a * f x + b * x) ∧
  (∀ x y : ℝ, y ≠ 0 → f x * f y = f (x*y) + f (x/y)) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x : ℝ, g (2*x) = a * g x + b * x) ∧
     (∀ x y : ℝ, y ≠ 0 → g x * g y = g (x*y) + g (x/y)))
    → g = f) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_l2121_212116


namespace NUMINAMATH_CALUDE_integer_distance_implies_horizontal_segment_l2121_212119

/-- A polynomial function with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The squared Euclidean distance between two points -/
def squaredDistance (x₁ y₁ x₂ y₂ : ℤ) : ℤ :=
  (x₂ - x₁)^2 + (y₂ - y₁)^2

theorem integer_distance_implies_horizontal_segment
  (f : IntPolynomial) (a b : ℤ) :
  (∃ d : ℤ, d^2 = squaredDistance a (f a) b (f b)) →
  f a = f b :=
sorry

end NUMINAMATH_CALUDE_integer_distance_implies_horizontal_segment_l2121_212119


namespace NUMINAMATH_CALUDE_daisies_per_bouquet_l2121_212184

/-- Represents a flower shop with rose and daisy bouquets -/
structure FlowerShop where
  roses_per_bouquet : ℕ
  total_bouquets : ℕ
  rose_bouquets : ℕ
  daisy_bouquets : ℕ
  total_flowers : ℕ

/-- Theorem stating the number of daisies in each daisy bouquet -/
theorem daisies_per_bouquet (shop : FlowerShop) 
  (h1 : shop.roses_per_bouquet = 12)
  (h2 : shop.total_bouquets = 20)
  (h3 : shop.rose_bouquets = 10)
  (h4 : shop.daisy_bouquets = 10)
  (h5 : shop.total_flowers = 190)
  (h6 : shop.rose_bouquets + shop.daisy_bouquets = shop.total_bouquets) :
  (shop.total_flowers - shop.roses_per_bouquet * shop.rose_bouquets) / shop.daisy_bouquets = 7 := by
  sorry

end NUMINAMATH_CALUDE_daisies_per_bouquet_l2121_212184


namespace NUMINAMATH_CALUDE_parallelogram_count_l2121_212152

/-- Given two sets of parallel lines in a plane, prove the number of parallelograms formed -/
theorem parallelogram_count (m n : ℕ) : ℕ := by
  /- m is the number of lines in the first set -/
  /- n is the number of lines in the second set -/
  /- The two sets of lines are parallel and intersect -/
  /- The number of parallelograms formed is Combination(m,2) * Combination(n,2) -/
  sorry

#check parallelogram_count

end NUMINAMATH_CALUDE_parallelogram_count_l2121_212152


namespace NUMINAMATH_CALUDE_triangle_area_l2121_212169

/-- Given a triangle ABC where:
  * b is the length of the side opposite to angle B
  * c is the length of the side opposite to angle C
  * C is twice the measure of angle B
prove that the area of the triangle is 15√7/16 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b = 2 → 
  c = 3 → 
  C = 2 * B → 
  (1/2) * b * c * Real.sin A = 15 * Real.sqrt 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2121_212169


namespace NUMINAMATH_CALUDE_carla_water_consumption_l2121_212157

/-- Given the conditions of Carla's liquid consumption, prove that she drank 15 ounces of water. -/
theorem carla_water_consumption (water soda : ℝ) 
  (h1 : soda = 3 * water - 6)
  (h2 : water + soda = 54) :
  water = 15 := by
  sorry

end NUMINAMATH_CALUDE_carla_water_consumption_l2121_212157


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2121_212137

/-- A parabola with vertex (h, k) passing through point (x₀, y₀) has equation y = a(x - h)² + k -/
def is_parabola (a h k x₀ y₀ : ℝ) : Prop :=
  y₀ = a * (x₀ - h)^2 + k

/-- The general form of a parabola y = ax² + bx + c can be derived from the vertex form -/
def general_form (a h k : ℝ) : ℝ × ℝ × ℝ :=
  (a, -2*a*h, a*h^2 + k)

theorem parabola_coefficients :
  ∀ (a : ℝ), is_parabola a 4 (-1) 2 3 →
  general_form a 4 (-1) = (1, -8, 15) := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2121_212137


namespace NUMINAMATH_CALUDE_some_number_value_l2121_212195

theorem some_number_value (X : ℝ) :
  2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002 →
  X = 1.25 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l2121_212195


namespace NUMINAMATH_CALUDE_sum_of_powers_and_mersenne_is_sum_of_squares_l2121_212124

/-- A Mersenne prime is a prime number of the form 2^k - 1 for some positive integer k. -/
def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, k > 0 ∧ p = 2^k - 1

/-- An integer n that is both the sum of two different powers of 2 
    and the sum of two different Mersenne primes is the sum of two different square numbers. -/
theorem sum_of_powers_and_mersenne_is_sum_of_squares (n : ℕ) 
  (h1 : ∃ a b : ℕ, a ≠ b ∧ n = 2^a + 2^b)
  (h2 : ∃ p q : ℕ, p ≠ q ∧ is_mersenne_prime p ∧ is_mersenne_prime q ∧ n = p + q) :
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_powers_and_mersenne_is_sum_of_squares_l2121_212124


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_l2121_212192

theorem celsius_to_fahrenheit (C F : ℝ) : 
  C = (4/7) * (F - 40) → C = 35 → F = 101.25 := by
sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_l2121_212192


namespace NUMINAMATH_CALUDE_usual_time_to_school_l2121_212150

/-- The usual time for a boy to reach school, given that when he walks 7/6 of his usual rate,
    he reaches school 2 minutes early. -/
theorem usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) (h2 : usual_time > 0)
    (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 2)) : 
  usual_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_school_l2121_212150


namespace NUMINAMATH_CALUDE_sara_quarters_remaining_l2121_212181

/-- Given that Sara initially had 783 quarters and her dad borrowed 271 quarters,
    prove that Sara now has 512 quarters. -/
theorem sara_quarters_remaining (initial_quarters borrowed_quarters : ℕ) 
    (h1 : initial_quarters = 783)
    (h2 : borrowed_quarters = 271) :
    initial_quarters - borrowed_quarters = 512 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_remaining_l2121_212181


namespace NUMINAMATH_CALUDE_xyz_value_l2121_212187

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 10)
  (eq5 : x + y + z = 6) :
  x * y * z = 15 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2121_212187


namespace NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l2121_212128

/-- Calculates the expected number of dogwood trees in the park after a given number of years -/
def expected_trees (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) 
                   (growth_rate_today : ℕ) (growth_rate_tomorrow : ℕ) (years : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow + 
  (planted_today * growth_rate_today * years) + 
  (planted_tomorrow * growth_rate_tomorrow * years)

/-- Theorem stating the expected number of dogwood trees after 5 years -/
theorem dogwood_trees_after_five_years :
  expected_trees 39 41 20 2 4 5 = 130 := by
  sorry

#eval expected_trees 39 41 20 2 4 5

end NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l2121_212128


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2121_212136

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 5 = 6) :
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2121_212136


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l2121_212101

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l2121_212101


namespace NUMINAMATH_CALUDE_pool_capacity_l2121_212174

theorem pool_capacity : 
  ∀ (initial_fraction final_fraction added_volume total_capacity : ℚ),
  initial_fraction = 1 / 8 →
  final_fraction = 2 / 3 →
  added_volume = 210 →
  (final_fraction - initial_fraction) * total_capacity = added_volume →
  total_capacity = 5040 / 13 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_l2121_212174


namespace NUMINAMATH_CALUDE_triangle_problem_l2121_212114

theorem triangle_problem (A B C : Real) (a b c S : Real) :
  -- Given conditions
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 3 →
  S = Real.sqrt 3 / 2 →
  -- Triangle properties
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  -- Conclusion
  A = π/3 ∧ b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2121_212114


namespace NUMINAMATH_CALUDE_mikes_ride_length_l2121_212144

/-- Proves that Mike's ride was 36 miles long given the taxi fare conditions -/
theorem mikes_ride_length :
  let mike_base_fare : ℚ := 2.5
  let mike_per_mile : ℚ := 0.25
  let annie_base_fare : ℚ := 2.5
  let annie_toll : ℚ := 5
  let annie_per_mile : ℚ := 0.25
  let annie_miles : ℚ := 16
  ∀ m : ℚ,
    mike_base_fare + mike_per_mile * m = 
    annie_base_fare + annie_toll + annie_per_mile * annie_miles →
    m = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_mikes_ride_length_l2121_212144


namespace NUMINAMATH_CALUDE_cafeteria_shirts_l2121_212172

theorem cafeteria_shirts (total : ℕ) (checkered : ℕ) (horizontal : ℕ) (vertical : ℕ) : 
  total = 40 →
  checkered = 7 →
  horizontal = 4 * checkered →
  vertical = total - (checkered + horizontal) →
  vertical = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_shirts_l2121_212172


namespace NUMINAMATH_CALUDE_short_track_speed_skating_selection_l2121_212177

theorem short_track_speed_skating_selection
  (p : Prop) -- A gets first place
  (q : Prop) -- B gets second place
  (r : Prop) -- C gets third place
  (h1 : p ∨ q) -- p ∨ q is true
  (h2 : ¬(p ∧ q)) -- p ∧ q is false
  (h3 : (¬q) ∧ r) -- (¬q) ∧ r is true
  : p ∧ ¬q ∧ r := by sorry

end NUMINAMATH_CALUDE_short_track_speed_skating_selection_l2121_212177


namespace NUMINAMATH_CALUDE_intersection_of_specific_sets_l2121_212186

theorem intersection_of_specific_sets :
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_sets_l2121_212186


namespace NUMINAMATH_CALUDE_locus_is_circle_l2121_212135

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Defines a right triangle with sides s, s, and s√2 -/
def rightTriangle (s : ℝ) : Triangle :=
  { A := { x := 0, y := 0 },
    B := { x := s, y := 0 },
    C := { x := 0, y := s } }

/-- Calculates the sum of squared distances from a point to the vertices of a triangle -/
def sumOfSquaredDistances (t : Triangle) (p : Point) : ℝ :=
  distanceSquared p t.A + distanceSquared p t.B + distanceSquared p t.C

/-- Theorem: The locus of points P such that the sum of squares of distances from P
    to the vertices of a right triangle with sides s, s, and s√2 equals a
    is a circle if and only if a > K, where K is a constant dependent on s -/
theorem locus_is_circle (s a : ℝ) (h1 : s > 0) :
  ∃ K, ∀ p : Point,
    sumOfSquaredDistances (rightTriangle s) p = a ↔ 
    ∃ r, r > 0 ∧ distanceSquared p { x := s/4, y := s/4 } = r^2 ∧ a > K := by
  sorry


end NUMINAMATH_CALUDE_locus_is_circle_l2121_212135


namespace NUMINAMATH_CALUDE_jack_second_half_time_l2121_212123

/-- Jack and Jill's race up the hill -/
def hill_race (jack_first_half jack_total jill_total : ℕ) : Prop :=
  jack_first_half = 19 ∧
  jill_total = 32 ∧
  jack_total + 7 = jill_total

theorem jack_second_half_time 
  (jack_first_half jack_total jill_total : ℕ) 
  (h : hill_race jack_first_half jack_total jill_total) : 
  jack_total - jack_first_half = 6 := by
  sorry

end NUMINAMATH_CALUDE_jack_second_half_time_l2121_212123


namespace NUMINAMATH_CALUDE_total_cost_is_21_16_l2121_212139

def sandwich_price : ℝ := 2.49
def soda_price : ℝ := 1.87
def chips_price : ℝ := 1.25
def chocolate_price : ℝ := 0.99

def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4
def chips_quantity : ℕ := 3
def chocolate_quantity : ℕ := 5

def total_cost : ℝ :=
  sandwich_price * sandwich_quantity +
  soda_price * soda_quantity +
  chips_price * chips_quantity +
  chocolate_price * chocolate_quantity

theorem total_cost_is_21_16 : total_cost = 21.16 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_21_16_l2121_212139


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_3_l2121_212166

theorem smallest_perfect_square_divisible_by_2_and_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m^2) ∧ 2 ∣ n ∧ 3 ∣ n ∧
  ∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → 2 ∣ k → 3 ∣ k → n ≤ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_3_l2121_212166


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2121_212146

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^3 - y^3 = 2*x*y + 8 ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2121_212146


namespace NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_k_l2121_212163

theorem sin_2alpha_in_terms_of_k (k α : ℝ) (h : Real.cos (π / 4 - α) = k) :
  Real.sin (2 * α) = 2 * k^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_in_terms_of_k_l2121_212163


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2121_212155

-- Part 1
theorem simplify_expression_1 (a : ℝ) : a - 2*a + 3*a = 2*a := by
  sorry

-- Part 2
theorem simplify_expression_2 (x y : ℝ) : 3*(2*x - 7*y) - (4*x - 10*y) = 2*x - 11*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2121_212155


namespace NUMINAMATH_CALUDE_intersection_height_l2121_212170

/-- Represents a line in 2D space --/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

/-- Creates a line from two points --/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  { m := m, b := b }

/-- Calculates the y-coordinate for a given x on the line --/
def Line.yAt (l : Line) (x : ℚ) : ℚ :=
  l.m * x + l.b

theorem intersection_height : 
  let line1 := lineFromPoints 0 30 120 0
  let line2 := lineFromPoints 0 0 120 50
  let x_intersect := (line2.b - line1.b) / (line1.m - line2.m)
  line1.yAt x_intersect = 75/4 := by sorry

end NUMINAMATH_CALUDE_intersection_height_l2121_212170


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2121_212111

/-- Given an isosceles triangle with inscribed circle radius ρ and circumscribed circle radius r,
    prove the side lengths of the triangle. -/
theorem isosceles_triangle_side_lengths
  (r ρ : ℝ)
  (h_positive_r : r > 0)
  (h_positive_ρ : ρ > 0)
  (h_r_geq_2ρ : r ≥ 2 * ρ) :
  ∃ (AC AB : ℝ),
    AC = Real.sqrt (2 * r * (r + ρ + Real.sqrt (r * (r - 2 * ρ)))) ∧
    AB = 2 * Real.sqrt (ρ * (2 * r - ρ - 2 * Real.sqrt (r * (r - 2 * ρ)))) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2121_212111


namespace NUMINAMATH_CALUDE_z_properties_l2121_212102

def z : ℂ := -(2 * Complex.I + 6) * Complex.I

theorem z_properties : 
  (z.re > 0 ∧ z.im < 0) ∧ 
  ∃ (y : ℝ), z - 2 = y * Complex.I :=
sorry

end NUMINAMATH_CALUDE_z_properties_l2121_212102


namespace NUMINAMATH_CALUDE_combination_three_choose_two_l2121_212133

theorem combination_three_choose_two : Finset.card (Finset.powerset {0, 1, 2} |>.filter (fun s => Finset.card s = 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_three_choose_two_l2121_212133


namespace NUMINAMATH_CALUDE_triangular_pyramid_inequality_l2121_212160

-- Define a structure for a triangular pyramid
structure TriangularPyramid where
  -- We don't need to explicitly define vertices A, B, C, D
  -- as they are implicit in the following measurements
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  a : ℝ  -- length of longest edge
  h : ℝ  -- length of shortest altitude

-- State the theorem
theorem triangular_pyramid_inequality (pyramid : TriangularPyramid) :
  pyramid.R / pyramid.r > pyramid.a / pyramid.h := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_inequality_l2121_212160


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2121_212185

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 6*x + c < 0 ↔ x < 2 ∨ x > 4) → c = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2121_212185


namespace NUMINAMATH_CALUDE_pencils_per_student_l2121_212145

/-- Represents the distribution of pencils to students -/
def pencil_distribution (total_pencils : ℕ) (max_students : ℕ) : ℕ :=
  total_pencils / max_students

/-- Theorem stating that given 910 pencils and 91 students, each student receives 10 pencils -/
theorem pencils_per_student :
  pencil_distribution 910 91 = 10 := by
  sorry

#check pencils_per_student

end NUMINAMATH_CALUDE_pencils_per_student_l2121_212145


namespace NUMINAMATH_CALUDE_combination_lock_code_l2121_212162

theorem combination_lock_code (x y : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ x ≠ 0 → 
  (x + y + x * y = 10 * x + y) ↔ 
  (y = 9 ∧ x ∈ Finset.range 10 \ {0}) :=
sorry

end NUMINAMATH_CALUDE_combination_lock_code_l2121_212162


namespace NUMINAMATH_CALUDE_equation_satisfies_condition_l2121_212168

theorem equation_satisfies_condition (x y z : ℤ) : 
  x = z - 2 ∧ y = x + 1 → x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfies_condition_l2121_212168


namespace NUMINAMATH_CALUDE_james_tshirt_purchase_l2121_212107

/-- The total cost for a discounted purchase of t-shirts -/
def discounted_total_cost (num_shirts : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_shirts * original_price * (1 - discount_percent)

/-- Theorem: James pays $60 for 6 t-shirts at 50% off, originally priced at $20 each -/
theorem james_tshirt_purchase : 
  discounted_total_cost 6 20 (1/2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_tshirt_purchase_l2121_212107


namespace NUMINAMATH_CALUDE_fraction_of_y_l2121_212183

theorem fraction_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_y_l2121_212183


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2121_212106

theorem reciprocal_problem :
  (∀ x : ℚ, x ≠ 0 → x * (1 / x) = 1) →
  (1 / 0.125 = 8) ∧ (1 / 1 = 1) := by sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2121_212106


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l2121_212193

/-- The number of ways to arrange 7 people in a row with 2 people between A and B -/
def arrangement_count : ℕ := 960

/-- The total number of people -/
def total_people : ℕ := 7

/-- The number of people between A and B -/
def people_between : ℕ := 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct : 
  arrangement_count = 
    (Nat.factorial 2) *  -- Ways to arrange A and B
    (Nat.choose (total_people - 2) people_between) *  -- Ways to choose people between A and B
    (Nat.factorial people_between) *  -- Ways to arrange people between A and B
    (Nat.factorial (total_people - people_between - 2))  -- Ways to arrange remaining people
  := by sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l2121_212193


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2121_212175

/-- Proves that a boat's speed in still water is 2.5 km/hr given its downstream and upstream travel times -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h1 : distance = 10) 
  (h2 : downstream_time = 3) 
  (h3 : upstream_time = 6) : 
  ∃ (boat_speed : ℝ), boat_speed = 2.5 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2121_212175


namespace NUMINAMATH_CALUDE_optimal_mask_purchase_l2121_212199

/-- Represents the profit function for mask sales -/
def profit_function (x : ℝ) : ℝ := -0.05 * x + 400

/-- Represents the constraints on the number of masks -/
def mask_constraints (x : ℝ) : Prop := 500 ≤ x ∧ x ≤ 1000

/-- Theorem stating the optimal purchase for maximum profit -/
theorem optimal_mask_purchase :
  ∀ x : ℝ, mask_constraints x →
  profit_function 500 ≥ profit_function x :=
sorry

end NUMINAMATH_CALUDE_optimal_mask_purchase_l2121_212199


namespace NUMINAMATH_CALUDE_sales_volume_decrease_and_may_prediction_l2121_212122

/-- Represents the monthly sales volume decrease rate -/
def monthly_decrease_rate : ℝ := 0.05

/-- Calculates the sales volume after n months given an initial volume and monthly decrease rate -/
def sales_volume (initial_volume : ℝ) (n : ℕ) : ℝ :=
  initial_volume * (1 - monthly_decrease_rate) ^ n

theorem sales_volume_decrease_and_may_prediction
  (january_volume : ℝ)
  (march_volume : ℝ)
  (h1 : january_volume = 6000)
  (h2 : march_volume = 5400)
  (h3 : sales_volume january_volume 2 = march_volume)
  : monthly_decrease_rate = 0.05 ∧ sales_volume january_volume 4 > 4500 := by
  sorry

#eval sales_volume 6000 4

end NUMINAMATH_CALUDE_sales_volume_decrease_and_may_prediction_l2121_212122


namespace NUMINAMATH_CALUDE_ellipse_product_l2121_212197

/-- Represents an ellipse with center O, major axis AB, minor axis CD, and focus F. -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  f : ℝ  -- Distance from center to focus

/-- Conditions for the ellipse problem -/
def EllipseProblem (e : Ellipse) : Prop :=
  e.f = 6 ∧ e.a - e.b = 4 ∧ e.a^2 - e.b^2 = e.f^2

theorem ellipse_product (e : Ellipse) (h : EllipseProblem e) : (2 * e.a) * (2 * e.b) = 65 := by
  sorry

#check ellipse_product

end NUMINAMATH_CALUDE_ellipse_product_l2121_212197


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l2121_212198

-- Part 1
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0) → m ∈ Set.Ioc (-1) 0 :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → m * x^2 - 2 * m * x - 1 < 0) → 
  x ∈ Set.Ioo (1 - Real.sqrt 2) 1 ∪ Set.Ioo 1 (1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l2121_212198


namespace NUMINAMATH_CALUDE_unique_prime_product_power_l2121_212121

/-- Given a natural number k, returns the product of the first k prime numbers -/
def primeProd (k : ℕ) : ℕ := sorry

/-- The only natural number k for which the product of the first k prime numbers 
    minus 1 is an exact power (greater than 1) of a natural number is 1 -/
theorem unique_prime_product_power : 
  ∀ k : ℕ, k > 0 → 
  (∃ (a n : ℕ), n > 1 ∧ primeProd k - 1 = a^n) → 
  k = 1 := by sorry

end NUMINAMATH_CALUDE_unique_prime_product_power_l2121_212121


namespace NUMINAMATH_CALUDE_angle_negative_1445_quadrant_l2121_212149

theorem angle_negative_1445_quadrant : 
  ∃ (k : ℤ) (θ : ℝ), -1445 = 360 * k + θ ∧ 270 < θ ∧ θ ≤ 360 :=
sorry

end NUMINAMATH_CALUDE_angle_negative_1445_quadrant_l2121_212149


namespace NUMINAMATH_CALUDE_min_sum_of_positive_integers_l2121_212179

theorem min_sum_of_positive_integers (a b : ℕ+) (h : a.val * b.val - 7 * a.val - 11 * b.val + 13 = 0) :
  ∃ (a₀ b₀ : ℕ+), a₀.val * b₀.val - 7 * a₀.val - 11 * b₀.val + 13 = 0 ∧
    ∀ (x y : ℕ+), x.val * y.val - 7 * x.val - 11 * y.val + 13 = 0 → a₀.val + b₀.val ≤ x.val + y.val ∧
    a₀.val + b₀.val = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_positive_integers_l2121_212179


namespace NUMINAMATH_CALUDE_simplify_expression_l2121_212132

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) = 16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2121_212132


namespace NUMINAMATH_CALUDE_plot_perimeter_l2121_212126

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ
  length_width_relation : length = width + 10
  cost_relation : fencing_cost = (2 * (length + width)) * fencing_rate

/-- The perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem stating the perimeter of the specific plot -/
theorem plot_perimeter (plot : RectangularPlot) 
  (h1 : plot.fencing_rate = 6.5)
  (h2 : plot.fencing_cost = 910) : 
  perimeter plot = 140 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_l2121_212126


namespace NUMINAMATH_CALUDE_master_title_possibilities_l2121_212138

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  num_players : Nat
  points_for_win : Rat
  points_for_draw : Rat
  points_for_loss : Rat
  master_threshold : Rat

/-- Determines if it's possible for a given number of players to earn the Master of Sports title --/
def can_earn_master_title (t : ChessTournament) (num_masters : Nat) : Prop :=
  num_masters ≤ t.num_players ∧
  ∃ (point_distribution : Fin t.num_players → Rat),
    (∀ i, point_distribution i ≥ (t.num_players - 1 : Rat) * t.points_for_win * t.master_threshold) ∧
    (∀ i j, i ≠ j → point_distribution i + point_distribution j ≤ t.points_for_win)

/-- The specific tournament described in the problem --/
def tournament : ChessTournament :=
  { num_players := 12
  , points_for_win := 1
  , points_for_draw := 1/2
  , points_for_loss := 0
  , master_threshold := 7/10 }

theorem master_title_possibilities :
  (can_earn_master_title tournament 7) ∧
  ¬(can_earn_master_title tournament 8) := by sorry

end NUMINAMATH_CALUDE_master_title_possibilities_l2121_212138


namespace NUMINAMATH_CALUDE_n_has_five_digits_l2121_212156

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 15 -/
axiom n_div_15 : 15 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^4 is a perfect square -/
axiom n_fourth_square : ∃ m : ℕ, n^4 = m^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ k : ℕ, k > 0 → (15 ∣ k) → (∃ a : ℕ, k^2 = a^4) → (∃ b : ℕ, k^4 = b^2) → n ≤ k

/-- The number of digits in n -/
def digits (m : ℕ) : ℕ := sorry

/-- Theorem stating that n has 5 digits -/
theorem n_has_five_digits : digits n = 5 := sorry

end NUMINAMATH_CALUDE_n_has_five_digits_l2121_212156


namespace NUMINAMATH_CALUDE_total_peaches_is_450_l2121_212165

/-- Represents the number of baskets in the fruit shop -/
def num_baskets : ℕ := 15

/-- Represents the initial number of red peaches in each basket -/
def initial_red : ℕ := 19

/-- Represents the initial number of green peaches in each basket -/
def initial_green : ℕ := 4

/-- Represents the number of moldy peaches in each basket -/
def moldy : ℕ := 6

/-- Represents the number of red peaches removed from each basket -/
def removed_red : ℕ := 3

/-- Represents the number of green peaches removed from each basket -/
def removed_green : ℕ := 1

/-- Represents the number of freshly harvested peaches added to each basket -/
def added_fresh : ℕ := 5

/-- Calculates the total number of peaches in all baskets after adjustments -/
def total_peaches_after_adjustment : ℕ :=
  num_baskets * ((initial_red - removed_red) + (initial_green - removed_green) + moldy + added_fresh)

/-- Theorem stating that the total number of peaches after adjustments is 450 -/
theorem total_peaches_is_450 : total_peaches_after_adjustment = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_is_450_l2121_212165


namespace NUMINAMATH_CALUDE_f_inequality_l2121_212141

open Real

-- Define the function f and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem f_inequality (hf : ∀ x, x ∈ Set.Ici 0 → (x + 1) * f x + x * f' x ≥ 0)
  (hf_not_const : ¬∀ x y, f x = f y) :
  f 1 < 2 * ℯ * f 2 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2121_212141


namespace NUMINAMATH_CALUDE_select_three_from_four_l2121_212196

theorem select_three_from_four : Nat.choose 4 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_four_l2121_212196


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2121_212159

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5,
    and two sides with distinct integer lengths has an area of 12. -/
theorem quadrilateral_area (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (B.1 - A.1) * (B.2 - D.2) = (B.2 - A.2) * (B.1 - D.1) →  -- right angle at B
  (D.1 - C.1) * (D.2 - B.2) = (D.2 - C.2) * (D.1 - B.1) →  -- right angle at D
  AC = 5 →
  (∃ (x y : ℕ), (AB = x ∨ BC = x ∨ CD = x ∨ DA = x) ∧ 
                (AB = y ∨ BC = y ∨ CD = y ∨ DA = y) ∧ x ≠ y) →
  (1/2 * AB * BC) + (1/2 * CD * DA) = 12 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2121_212159


namespace NUMINAMATH_CALUDE_complex_number_problem_l2121_212134

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * Complex.I = 1 + Complex.I) →
  (z₂.im = 2) →
  ((z₁ * z₂).im = 0) →
  (z₁ = 3 - Complex.I ∧ z₂ = 6 + 2 * Complex.I) := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2121_212134


namespace NUMINAMATH_CALUDE_sin_plus_2cos_equals_two_fifths_l2121_212117

theorem sin_plus_2cos_equals_two_fifths (a : ℝ) (α : ℝ) :
  a < 0 →
  (∃ (x y : ℝ), x = -3*a ∧ y = 4*a ∧ Real.sin α = y / Real.sqrt (x^2 + y^2) ∧ Real.cos α = x / Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = 2/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_2cos_equals_two_fifths_l2121_212117


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2121_212189

theorem polynomial_expansion_equality (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2121_212189


namespace NUMINAMATH_CALUDE_john_metal_purchase_cost_l2121_212188

/-- Calculates the total cost of John's metal purchases in USD -/
def total_cost (silver_oz gold_oz platinum_oz palladium_oz : ℝ)
               (silver_price_usd gold_price_multiplier : ℝ)
               (platinum_price_gbp palladium_price_eur : ℝ)
               (usd_gbp_rate1 usd_gbp_rate2 : ℝ)
               (usd_eur_rate1 usd_eur_rate2 : ℝ)
               (silver_gold_discount platinum_tax : ℝ) : ℝ :=
  sorry

theorem john_metal_purchase_cost :
  total_cost 2.5 3.5 4.5 5.5 25 60 80 100 1.3 1.4 1.15 1.2 0.05 0.08 = 6184.815 := by
  sorry

end NUMINAMATH_CALUDE_john_metal_purchase_cost_l2121_212188


namespace NUMINAMATH_CALUDE_original_number_is_sixty_l2121_212190

theorem original_number_is_sixty : 
  ∀ x : ℝ, (0.5 * x = 30) → x = 60 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_sixty_l2121_212190


namespace NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l2121_212104

theorem perimeter_ratio_not_integer (a k l : ℕ+) (h : a ^ 2 = k * l) :
  ¬ ∃ (n : ℕ), (2 * (k + l) : ℚ) / (4 * a) = n := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l2121_212104


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_11_l2121_212112

def is_even_digit (d : ℕ) : Prop := d % 2 = 0 ∧ d < 10

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

theorem largest_even_digit_multiple_of_11 :
  ∀ n : ℕ,
    n < 10000 →
    has_only_even_digits n →
    n % 11 = 0 →
    n ≤ 8800 :=
sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_11_l2121_212112


namespace NUMINAMATH_CALUDE_cube_root_equation_l2121_212105

theorem cube_root_equation (x : ℝ) : 
  (x * (x^3)^(1/2))^(1/3) = 3 → x = 3^(6/5) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_l2121_212105


namespace NUMINAMATH_CALUDE_intersection_M_N_l2121_212130

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2121_212130


namespace NUMINAMATH_CALUDE_profitable_iff_price_ge_132_l2121_212108

/-- The transaction fee rate for stock trading in China -/
def fee_rate : ℚ := 75 / 10000

/-- The number of shares traded -/
def num_shares : ℕ := 1000

/-- The price increase per share -/
def price_increase : ℚ := 2

/-- Determines if a stock transaction is profitable given the initial price -/
def is_profitable (x : ℚ) : Prop :=
  (x + price_increase) * (1 - fee_rate) * num_shares ≥ (1 + fee_rate) * num_shares * x

/-- Theorem: The transaction is profitable if and only if the initial share price is at least 132 yuan -/
theorem profitable_iff_price_ge_132 (x : ℚ) : is_profitable x ↔ x ≥ 132 := by
  sorry

end NUMINAMATH_CALUDE_profitable_iff_price_ge_132_l2121_212108


namespace NUMINAMATH_CALUDE_train_crossing_time_l2121_212154

/-- Time taken for a faster train to cross a man in a slower train -/
theorem train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : 
  faster_speed = 54 →
  slower_speed = 36 →
  train_length = 135 →
  (train_length / (faster_speed - slower_speed)) * (3600 / 1000) = 27 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2121_212154


namespace NUMINAMATH_CALUDE_work_completion_time_l2121_212180

theorem work_completion_time 
  (total_work : ℝ) 
  (a b c : ℝ) 
  (h1 : a + b + c = total_work / 4)  -- a, b, and c together finish in 4 days
  (h2 : b = total_work / 9)          -- b alone finishes in 9 days
  (h3 : c = total_work / 18)         -- c alone finishes in 18 days
  : a = total_work / 12 :=           -- a alone finishes in 12 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2121_212180


namespace NUMINAMATH_CALUDE_divisibility_relation_l2121_212178

theorem divisibility_relation (x y z : ℤ) (h : (11 : ℤ) ∣ (7 * x + 2 * y - 5 * z)) :
  (11 : ℤ) ∣ (3 * x - 7 * y + 12 * z) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2121_212178


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2121_212120

theorem smallest_number_divisibility (x : ℕ) : x = 257 ↔ 
  (x > 0) ∧ 
  (∀ z : ℕ, z > 0 → z < x → ¬((z + 7) % 8 = 0 ∧ (z + 7) % 11 = 0 ∧ (z + 7) % 24 = 0)) ∧ 
  ((x + 7) % 8 = 0) ∧ 
  ((x + 7) % 11 = 0) ∧ 
  ((x + 7) % 24 = 0) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2121_212120


namespace NUMINAMATH_CALUDE_max_value_and_inequality_l2121_212191

def f (x m : ℝ) : ℝ := |x - m| - |x + 2*m|

theorem max_value_and_inequality (m : ℝ) (hm : m > 0) 
  (hmax : ∀ x, f x m ≤ 3) :
  m = 1 ∧ 
  ∀ a b : ℝ, a * b > 0 → a^2 + b^2 = m^2 → a^3 / b + b^3 / a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_max_value_and_inequality_l2121_212191


namespace NUMINAMATH_CALUDE_linear_equation_implies_mn_zero_l2121_212110

/-- If x^(m+n) + 5y^(m-n+2) = 8 is a linear equation in x and y, then mn = 0 -/
theorem linear_equation_implies_mn_zero (m n : ℤ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, a * x + b * y = c ↔ x^(m+n) + 5 * y^(m-n+2) = 8) →
  m * n = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_implies_mn_zero_l2121_212110


namespace NUMINAMATH_CALUDE_willow_playing_time_l2121_212171

/-- Calculates the total playing time in hours given the time spent on football and basketball in minutes -/
def total_playing_time (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Proves that given Willow played football for 60 minutes and basketball for 60 minutes, 
    the total time he played is 2 hours -/
theorem willow_playing_time :
  total_playing_time 60 60 = 2 := by sorry

end NUMINAMATH_CALUDE_willow_playing_time_l2121_212171


namespace NUMINAMATH_CALUDE_circle_placement_possible_l2121_212173

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square -/
structure Square where
  center : Point

/-- The main theorem -/
theorem circle_placement_possible
  (rect : Rectangle)
  (squares : Finset Square)
  (h_rect_dim : rect.width = 20 ∧ rect.height = 25)
  (h_squares_count : squares.card = 120) :
  ∃ (p : Point),
    (0.5 ≤ p.x ∧ p.x ≤ rect.width - 0.5) ∧
    (0.5 ≤ p.y ∧ p.y ≤ rect.height - 0.5) ∧
    ∀ (s : Square), s ∈ squares →
      (p.x - s.center.x)^2 + (p.y - s.center.y)^2 ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_circle_placement_possible_l2121_212173


namespace NUMINAMATH_CALUDE_sculpture_cost_in_rupees_l2121_212127

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_namibian : ℝ := 5

/-- Exchange rate from US dollars to Indian rupees -/
def usd_to_rupees : ℝ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_namibian : ℝ := 200

/-- Theorem stating the cost of the sculpture in Indian rupees -/
theorem sculpture_cost_in_rupees :
  (sculpture_cost_namibian / usd_to_namibian) * usd_to_rupees = 320 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_rupees_l2121_212127


namespace NUMINAMATH_CALUDE_min_lines_for_200_intersections_l2121_212115

/-- The number of intersection points for m lines -/
def intersectionPoints (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The minimum number of lines that intersect in exactly 200 points -/
def minLines : ℕ := 21

theorem min_lines_for_200_intersections :
  (intersectionPoints minLines = 200) ∧
  (∀ k : ℕ, k < minLines → intersectionPoints k < 200) := by
  sorry

end NUMINAMATH_CALUDE_min_lines_for_200_intersections_l2121_212115


namespace NUMINAMATH_CALUDE_gcd_48_180_l2121_212161

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_48_180_l2121_212161


namespace NUMINAMATH_CALUDE_statement_a_is_correct_l2121_212103

theorem statement_a_is_correct (x y : ℝ) : x + y < 0 → x^2 - y > x := by
  sorry

end NUMINAMATH_CALUDE_statement_a_is_correct_l2121_212103


namespace NUMINAMATH_CALUDE_length_of_A_l2121_212147

def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 6)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, (1 - t) • p + t • q = r

theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ,
    on_line_y_eq_x A' ∧
    on_line_y_eq_x B' ∧
    intersect A A' C ∧
    intersect B B' C ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = (12 / 7) * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_of_A_l2121_212147


namespace NUMINAMATH_CALUDE_star_properties_l2121_212140

-- Define the * operation
def star (x y : ℝ) : ℝ := x - y

-- State the theorem
theorem star_properties :
  (∀ x : ℝ, star x x = 0) ∧
  (∀ x y z : ℝ, star x (star y z) = star x y + z) ∧
  (star 1993 1935 = 58) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l2121_212140


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2121_212109

/-- Represents a shape formed by unit cubes -/
structure CubeShape where
  cubes : ℕ
  central_cube : Bool
  surrounding_cubes : ℕ

/-- Calculates the volume of the shape -/
def volume (shape : CubeShape) : ℕ := shape.cubes

/-- Calculates the surface area of the shape -/
def surface_area (shape : CubeShape) : ℕ :=
  shape.surrounding_cubes * 5

/-- The specific shape described in the problem -/
def problem_shape : CubeShape :=
  { cubes := 8
  , central_cube := true
  , surrounding_cubes := 7 }

theorem volume_to_surface_area_ratio :
  (volume problem_shape : ℚ) / (surface_area problem_shape : ℚ) = 8 / 35 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2121_212109


namespace NUMINAMATH_CALUDE_hiker_distance_l2121_212153

theorem hiker_distance (north east south east2 : ℝ) 
  (h1 : north = 15)
  (h2 : east = 8)
  (h3 : south = 9)
  (h4 : east2 = 2) : 
  Real.sqrt ((north - south)^2 + (east + east2)^2) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l2121_212153


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2121_212151

theorem constant_term_expansion (α : Real) 
  (h : Real.sin (π - α) = 2 * Real.cos α) : 
  (Finset.range 7).sum (fun k => 
    (Nat.choose 6 k : Real) * 
    (Real.tan α)^k * 
    ((-1)^k * Nat.choose 6 (6-k))) = 160 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2121_212151


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2121_212158

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → 
  (n + (n + 5) = 35) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2121_212158


namespace NUMINAMATH_CALUDE_mike_money_total_l2121_212125

/-- Given that Mike has 9 5-dollar bills, prove that his total money is $45. -/
theorem mike_money_total : 
  let number_of_bills : ℕ := 9
  let bill_value : ℕ := 5
  number_of_bills * bill_value = 45 := by
  sorry

end NUMINAMATH_CALUDE_mike_money_total_l2121_212125


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2121_212182

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ ((m - 1) * x < Real.sqrt (4 * x - x^2))) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2121_212182


namespace NUMINAMATH_CALUDE_parabola_points_difference_l2121_212131

/-- Given a parabola x^2 = 4y with focus F, and two points A and B on it satisfying |AF| - |BF| = 2,
    prove that y₁ + x₁² - y₂ - x₂² = 10 -/
theorem parabola_points_difference (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 = 4*y₁) →  -- A is on the parabola
  (x₂^2 = 4*y₂) →  -- B is on the parabola
  (y₁ + 1 - (y₂ + 1) = 2) →  -- |AF| - |BF| = 2, where F is (0, 1)
  y₁ + x₁^2 - y₂ - x₂^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_parabola_points_difference_l2121_212131


namespace NUMINAMATH_CALUDE_pool_perimeter_l2121_212164

theorem pool_perimeter (garden_length garden_width pool_area : ℝ) 
  (h1 : garden_length = 8)
  (h2 : garden_width = 6)
  (h3 : pool_area = 24)
  (h4 : ∃ x : ℝ, (garden_length - 2*x) * (garden_width - 2*x) = pool_area ∧ 
                 x > 0 ∧ x < garden_length/2 ∧ x < garden_width/2) :
  ∃ pool_length pool_width : ℝ,
    pool_length * pool_width = pool_area ∧
    pool_length < garden_length ∧
    pool_width < garden_width ∧
    2 * pool_length + 2 * pool_width = 20 :=
by sorry

end NUMINAMATH_CALUDE_pool_perimeter_l2121_212164


namespace NUMINAMATH_CALUDE_opposite_of_neg_six_l2121_212113

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Theorem: The opposite of -6 is 6. -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_six_l2121_212113


namespace NUMINAMATH_CALUDE_vodka_alcohol_consumption_l2121_212176

/-- Calculates the amount of pure alcohol consumed by one person when splitting vodka shots. -/
theorem vodka_alcohol_consumption
  (total_shots : ℕ)
  (ounces_per_shot : ℚ)
  (alcohol_percentage : ℚ)
  (h1 : total_shots = 8)
  (h2 : ounces_per_shot = 3/2)
  (h3 : alcohol_percentage = 1/2) :
  (((total_shots : ℚ) / 2) * ounces_per_shot) * alcohol_percentage = 3 := by
  sorry

end NUMINAMATH_CALUDE_vodka_alcohol_consumption_l2121_212176


namespace NUMINAMATH_CALUDE_problem_statement_l2121_212167

theorem problem_statement 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (heq : a^2 + 2*b^2 + 3*c^2 = 4) : 
  (a = c → a*b ≤ Real.sqrt 2 / 2) ∧ 
  (a + 2*b + 3*c ≤ 2 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2121_212167


namespace NUMINAMATH_CALUDE_conditions_on_m_l2121_212143

/-- The set A defined by the quadratic equation mx² - 2x + 1 = 0 -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 - 2 * x + 1 = 0}

/-- Theorem stating the conditions on m for different properties of set A -/
theorem conditions_on_m :
  (∀ m : ℝ, A m = ∅ ↔ m > 1) ∧
  (∀ m : ℝ, (∃ x : ℝ, A m = {x}) ↔ m = 0 ∨ m = 1) ∧
  (∀ m : ℝ, (∃ x : ℝ, x ∈ A m ∧ x > 1/2 ∧ x < 2) ↔ m > 0 ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_conditions_on_m_l2121_212143


namespace NUMINAMATH_CALUDE_reading_competition_result_l2121_212118

/-- Represents the number of pages read by each girl --/
structure Pages where
  sasa : ℕ
  zuzka : ℕ
  ivana : ℕ
  majka : ℕ
  lucka : ℕ

/-- The conditions of the reading competition --/
def reading_conditions (p : Pages) : Prop :=
  p.lucka = 32 ∧
  p.lucka = (p.sasa + p.zuzka) / 2 ∧
  p.ivana = p.zuzka + 5 ∧
  p.majka = p.sasa - 8 ∧
  p.ivana = (p.majka + p.zuzka) / 2

/-- The theorem stating the correct number of pages read by each girl --/
theorem reading_competition_result :
  ∃ (p : Pages), reading_conditions p ∧
    p.sasa = 41 ∧ p.zuzka = 23 ∧ p.ivana = 28 ∧ p.majka = 33 ∧ p.lucka = 32 := by
  sorry

end NUMINAMATH_CALUDE_reading_competition_result_l2121_212118


namespace NUMINAMATH_CALUDE_second_number_proof_l2121_212148

theorem second_number_proof (x : ℝ) : 
  let set1 := [10, 60, 35]
  let set2 := [20, 60, x]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 5 →
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2121_212148
