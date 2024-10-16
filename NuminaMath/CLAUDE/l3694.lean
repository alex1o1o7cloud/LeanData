import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_l3694_369479

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 2) * (deriv^[2] f x) > 0) : 
  f 2 < f 0 ∧ f 0 < f (-3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3694_369479


namespace NUMINAMATH_CALUDE_zeros_of_odd_and_even_functions_l3694_369470

-- Define odd and even functions
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def EvenFunction (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the number of zeros for a function
def NumberOfZeros (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem zeros_of_odd_and_even_functions 
  (f g : ℝ → ℝ) 
  (hf : OddFunction f) 
  (hg : EvenFunction g) :
  (∃ k : ℕ, NumberOfZeros f = 2 * k + 1) ∧ 
  (∃ m : ℕ, NumberOfZeros g = m) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_odd_and_even_functions_l3694_369470


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_derivative_monotone_increasing_f_superadditive_l3694_369437

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (x + 1)

theorem tangent_line_at_zero (x : ℝ) :
  (deriv f) 0 = 1 :=
sorry

theorem derivative_monotone_increasing :
  StrictMonoOn (deriv f) (Set.Icc 0 2) :=
sorry

theorem f_superadditive {s t : ℝ} (hs : s > 0) (ht : t > 0) :
  f (s + t) > f s + f t :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_derivative_monotone_increasing_f_superadditive_l3694_369437


namespace NUMINAMATH_CALUDE_volume_circumscribed_sphere_folded_rectangle_l3694_369493

/-- The volume of the circumscribed sphere of a tetrahedron formed by folding a rectangle --/
theorem volume_circumscribed_sphere_folded_rectangle (a b : ℝ) (ha : a = 4) (hb : b = 3) :
  let diagonal := Real.sqrt (a^2 + b^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = (125/6) * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_circumscribed_sphere_folded_rectangle_l3694_369493


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3694_369417

/-- Given two rectangles with equal areas, where one rectangle has dimensions 12 inches by W inches,
    and the other has dimensions 9 inches by 20 inches, prove that W equals 15 inches. -/
theorem equal_area_rectangles (W : ℝ) :
  (12 * W = 9 * 20) → W = 15 := by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3694_369417


namespace NUMINAMATH_CALUDE_paint_difference_l3694_369439

theorem paint_difference (R r : ℝ) (h : R > 0) (h' : r > 0) : 
  (4 / 3 * Real.pi * R^3 - 4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * r^3) = 14.625 →
  (4 * Real.pi * R^2 - 4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 0.84 :=
by sorry

end NUMINAMATH_CALUDE_paint_difference_l3694_369439


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_l3694_369431

/-- If the terminal side of angle α passes through the point (-1, 6), then cos α = -√37/37 -/
theorem cos_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 6) →
  Real.cos α = -Real.sqrt 37 / 37 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_l3694_369431


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3694_369481

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3694_369481


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l3694_369483

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_num : Nat := 56

-- Theorem statement
theorem binary_to_octal_conversion :
  (binary_num.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0) = octal_num * 8 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l3694_369483


namespace NUMINAMATH_CALUDE_randys_house_blocks_l3694_369445

/-- Given Randy's block building scenario, prove the number of blocks used for the house. -/
theorem randys_house_blocks (total : ℕ) (tower : ℕ) (difference : ℕ) (house : ℕ) : 
  total = 90 → tower = 63 → difference = 26 → house = tower + difference → house = 89 := by
  sorry

end NUMINAMATH_CALUDE_randys_house_blocks_l3694_369445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_sum_l3694_369408

theorem arithmetic_sequence_constant_sum (a₁ d : ℝ) :
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let S : ℕ → ℝ := λ n => n * (2 * a₁ + (n - 1) * d) / 2
  (∀ a₁' d', a₁' + (1 + 7 + 10) * d' = a₁ + (1 + 7 + 10) * d) →
  (∀ a₁' d', S 13 = 13 * (2 * a₁' + 12 * d') / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_sum_l3694_369408


namespace NUMINAMATH_CALUDE_approximate_fish_population_l3694_369473

/-- Represents the fish population in a pond with tagging and recapture. -/
structure FishPopulation where
  total : ℕ  -- Total number of fish in the pond
  tagged : ℕ  -- Number of fish tagged in the first catch
  recaptured : ℕ  -- Number of fish recaptured in the second catch
  tagged_recaptured : ℕ  -- Number of tagged fish in the second catch

/-- The conditions of the problem -/
def pond_conditions : FishPopulation := {
  total := 0,  -- Unknown, to be determined
  tagged := 50,
  recaptured := 50,
  tagged_recaptured := 5
}

/-- Theorem stating the approximate number of fish in the pond -/
theorem approximate_fish_population (p : FishPopulation) 
  (h1 : p.tagged = pond_conditions.tagged)
  (h2 : p.recaptured = pond_conditions.recaptured)
  (h3 : p.tagged_recaptured = pond_conditions.tagged_recaptured)
  (h4 : p.tagged_recaptured / p.recaptured = p.tagged / p.total) :
  p.total = 500 := by
  sorry


end NUMINAMATH_CALUDE_approximate_fish_population_l3694_369473


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3694_369433

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_decreasing : ∀ n, a (n + 1) < a n)
  (h_geom : geometric_sequence a)
  (h_prod : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3694_369433


namespace NUMINAMATH_CALUDE_expression_value_l3694_369403

theorem expression_value (a b x y c : ℝ) 
  (h1 : a = -b) 
  (h2 : x * y = 1) 
  (h3 : c = 2 ∨ c = -2) : 
  (a + b) / 2 + x * y - (1 / 4) * c = 1 / 2 ∨ (a + b) / 2 + x * y - (1 / 4) * c = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3694_369403


namespace NUMINAMATH_CALUDE_time_to_run_around_field_l3694_369467

-- Define the side length of the square field
def side_length : ℝ := 50

-- Define the boy's running speed in km/hr
def running_speed : ℝ := 9

-- Theorem statement
theorem time_to_run_around_field : 
  let perimeter : ℝ := 4 * side_length
  let speed_in_mps : ℝ := running_speed * 1000 / 3600
  let time : ℝ := perimeter / speed_in_mps
  time = 80 := by sorry

end NUMINAMATH_CALUDE_time_to_run_around_field_l3694_369467


namespace NUMINAMATH_CALUDE_quarterback_passes_l3694_369434

theorem quarterback_passes (total : ℕ) (left : ℕ) (right : ℕ) (center : ℕ) : 
  total = 50 ∧ 
  right = 2 * left ∧ 
  center = left + 2 ∧ 
  total = left + right + center → 
  left = 12 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l3694_369434


namespace NUMINAMATH_CALUDE_smallest_integer_proof_l3694_369400

def smallest_integer : ℕ := 299576986419800

theorem smallest_integer_proof :
  (∀ i ∈ Finset.range 34, smallest_integer % (i + 1) = 0) ∧
  (∀ i ∈ Finset.range 3, smallest_integer % (i + 38) = 0) ∧
  (∀ i ∈ Finset.range 3, smallest_integer % (i + 35) ≠ 0) ∧
  (∀ n < smallest_integer, 
    (∃ i ∈ Finset.range 34, n % (i + 1) ≠ 0) ∨
    (∃ i ∈ Finset.range 3, n % (i + 38) ≠ 0) ∨
    (∀ i ∈ Finset.range 3, n % (i + 35) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_proof_l3694_369400


namespace NUMINAMATH_CALUDE_radius_formula_l3694_369491

/-- A square with a circumscribed circle where the sum of the lengths of all sides
    of the square equals the area of the circumscribed circle. -/
structure SquareWithCircle where
  side : ℝ
  radius : ℝ
  side_positive : 0 < side
  radius_positive : 0 < radius
  diagonal_eq_diameter : side * Real.sqrt 2 = 2 * radius
  perimeter_eq_area : 4 * side = π * radius^2

/-- The radius of the circumscribed circle is 4√2/π. -/
theorem radius_formula (s : SquareWithCircle) : s.radius = 4 * Real.sqrt 2 / π := by
  sorry

end NUMINAMATH_CALUDE_radius_formula_l3694_369491


namespace NUMINAMATH_CALUDE_age_difference_l3694_369450

/-- Proves that A was half of B's age 10 years ago given the conditions -/
theorem age_difference (a b : ℕ) : 
  (a : ℚ) / b = 3 / 4 →  -- ratio of present ages is 3:4
  a + b = 35 →          -- sum of present ages is 35
  ∃ (y : ℕ), y = 10 ∧ (a - y : ℚ) = (1 / 2) * (b - y) := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3694_369450


namespace NUMINAMATH_CALUDE_g_inequality_l3694_369468

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
axiom f_continuous : Continuous f
axiom f_property : ∀ m n : ℝ, Real.exp n * f m + Real.exp (2 * m) * f (n - m) = Real.exp m * f n
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- Define the function g
def g (x : ℝ) : ℝ :=
  if x < 1 then Real.exp (x - 1) * f (1 - x)
  else Real.exp (1 - x) * f (x - 1)

-- State the theorem to be proved
theorem g_inequality : g 2 < g (-1) := by
  sorry

end

end NUMINAMATH_CALUDE_g_inequality_l3694_369468


namespace NUMINAMATH_CALUDE_ellipse_complementary_angles_point_l3694_369495

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

-- Define the right focus of ellipse C
def right_focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the right focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - right_focus.1)

-- Define the property of complementary angles of inclination
def complementary_angles (P A B : ℝ × ℝ) : Prop :=
  (A.2 - P.2) / (A.1 - P.1) + (B.2 - P.2) / (B.1 - P.1) = 0

-- Main theorem
theorem ellipse_complementary_angles_point :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    k ≠ 0 →
    line_through_focus k A.1 A.2 →
    line_through_focus k B.1 B.2 →
    ellipse_C A.1 A.2 →
    ellipse_C B.1 B.2 →
    A ≠ B →
    complementary_angles P A B :=
sorry

end NUMINAMATH_CALUDE_ellipse_complementary_angles_point_l3694_369495


namespace NUMINAMATH_CALUDE_coin_problem_l3694_369460

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (ten_cent : ℕ) : ℕ :=
  23 - five_cent

/-- Represents the total number of coins -/
def total_coins (five_cent : ℕ) (ten_cent : ℕ) : ℕ :=
  five_cent + ten_cent

theorem coin_problem (five_cent ten_cent : ℕ) :
  total_coins five_cent ten_cent = 12 →
  different_values five_cent ten_cent = 19 →
  ten_cent = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3694_369460


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_production_l3694_369475

/-- Sales revenue as a function of production volume -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Total production cost as a function of production volume -/
def total_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit as a function of production volume -/
def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

/-- The production volume that maximizes profit -/
def optimal_production : ℝ := 6

theorem profit_maximized_at_optimal_production :
  ∀ x > 0, profit x ≤ profit optimal_production :=
by sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_production_l3694_369475


namespace NUMINAMATH_CALUDE_lanas_concert_expense_l3694_369444

def ticket_price : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lanas_concert_expense :
  (tickets_for_friends + extra_tickets) * ticket_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_lanas_concert_expense_l3694_369444


namespace NUMINAMATH_CALUDE_exactly_two_primes_probability_l3694_369405

-- Define a 12-sided die
def Die := Fin 12

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define the probability of rolling a prime number on a single die
def probPrime : ℚ := 5 / 12

-- Define the probability of not rolling a prime number on a single die
def probNotPrime : ℚ := 7 / 12

-- Define the number of dice
def numDice : ℕ := 3

-- Define the number of dice that should show prime numbers
def numPrimeDice : ℕ := 2

-- Theorem statement
theorem exactly_two_primes_probability :
  (numDice.choose numPrimeDice : ℚ) * probPrime ^ numPrimeDice * probNotPrime ^ (numDice - numPrimeDice) = 525 / 1728 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_primes_probability_l3694_369405


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3694_369480

/-- The maximum area of an equilateral triangle inscribed in a 12x13 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : 
  ∃ (A : ℝ), A = 205 * Real.sqrt 3 - 468 ∧
  ∀ (triangle_area : ℝ),
    (∃ (x y : ℝ),
      0 ≤ x ∧ x ≤ 12 ∧
      0 ≤ y ∧ y ≤ 13 ∧
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3694_369480


namespace NUMINAMATH_CALUDE_blocks_count_l3694_369477

/-- The number of blocks in Jacob's toy bin --/
def total_blocks (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ := red + yellow + blue

/-- Theorem: Given the conditions, the total number of blocks is 75 --/
theorem blocks_count :
  let red : ℕ := 18
  let yellow : ℕ := red + 7
  let blue : ℕ := red + 14
  total_blocks red yellow blue = 75 := by sorry

end NUMINAMATH_CALUDE_blocks_count_l3694_369477


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_induction_base_l3694_369497

/-- A polygon is a closed plane figure with at least 3 sides. -/
structure Polygon where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- The base case for the polygon interior angle sum theorem. -/
def polygon_interior_angle_sum_base_case : ℕ := 3

/-- Theorem: The base case for mathematical induction in the polygon interior angle sum theorem is n=3. -/
theorem polygon_interior_angle_sum_induction_base :
  polygon_interior_angle_sum_base_case = 3 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_induction_base_l3694_369497


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l3694_369476

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l3694_369476


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3694_369442

def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

def parallel (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, l1 a x1 y1 ∧ l2 a x2 y2 → (y1 - y2) * (a + 1) = (x1 - x2) * 2

theorem sufficient_not_necessary :
  (parallel (-2)) ∧ (∃ a : ℝ, a ≠ -2 ∧ parallel a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3694_369442


namespace NUMINAMATH_CALUDE_equation_solution_l3694_369496

theorem equation_solution : ∃! x : ℚ, (3 / 5) * (1 / 9) * x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3694_369496


namespace NUMINAMATH_CALUDE_triangle_side_range_l3694_369418

theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  (∃ (a₁ b₁ : ℝ) (A₁ B₁ : ℝ), a₁ ≠ a ∨ b₁ ≠ b ∨ A₁ ≠ A ∨ B₁ ≠ B) →
  Real.sqrt 2 < b ∧ b < 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3694_369418


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l3694_369464

theorem quadratic_roots_difference_squared :
  ∀ p q : ℝ, (2 * p^2 - 9 * p + 7 = 0) → (2 * q^2 - 9 * q + 7 = 0) → (p - q)^2 = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l3694_369464


namespace NUMINAMATH_CALUDE_smallest_odd_k_for_cubic_irreducibility_l3694_369456

/-- A cubic polynomial with integer coefficients -/
def CubicPolynomial := ℤ → ℤ

/-- Checks if a number is prime -/
def isPrime (n : ℤ) : Prop := sorry

/-- Checks if a polynomial is irreducible over ℤ -/
def isIrreducible (f : CubicPolynomial) : Prop := sorry

/-- The main theorem -/
theorem smallest_odd_k_for_cubic_irreducibility : 
  ∃ (k : ℕ), k % 2 = 1 ∧
  (∀ (j : ℕ), j % 2 = 1 → j < k →
    ∃ (f : CubicPolynomial),
      (∃ (S : Finset ℤ), S.card = j ∧ ∀ n ∈ S, isPrime (|f n|)) ∧
      ¬isIrreducible f) ∧
  (∀ (f : CubicPolynomial),
    (∃ (S : Finset ℤ), S.card = k ∧ ∀ n ∈ S, isPrime (|f n|)) →
    isIrreducible f) ∧
  k = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_k_for_cubic_irreducibility_l3694_369456


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3694_369486

theorem irreducible_fraction : (201920192019 : ℚ) / 191719171917 = 673 / 639 := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3694_369486


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_constrained_numbers_l3694_369463

theorem sum_reciprocals_of_constrained_numbers (m n : ℕ+) : 
  Nat.gcd m n = 6 → 
  Nat.lcm m n = 210 → 
  m + n = 72 → 
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 17.5 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_constrained_numbers_l3694_369463


namespace NUMINAMATH_CALUDE_min_value_of_4a_plus_b_l3694_369448

theorem min_value_of_4a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (1 : ℝ) / a + (1 : ℝ) / b = 1) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1 : ℝ) / a' + (1 : ℝ) / b' = 1 → 4 * a' + b' ≥ 4 * a + b) ∧ 
  4 * a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_4a_plus_b_l3694_369448


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3694_369489

theorem decimal_to_fraction (x : ℚ) (h : x = 368/100) : x = 92/25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3694_369489


namespace NUMINAMATH_CALUDE_calculation_result_l3694_369422

theorem calculation_result : (786 * 74) / 30 = 1938.8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3694_369422


namespace NUMINAMATH_CALUDE_men_added_to_group_l3694_369455

theorem men_added_to_group (original_days : ℕ) (new_days : ℕ) (new_men : ℕ) 
  (h1 : original_days = 24)
  (h2 : new_days = 16)
  (h3 : new_men = 12)
  (h4 : ∃ (original_men : ℕ), original_men * original_days = new_men * new_days) :
  new_men - (new_men * new_days / original_days) = 4 := by
  sorry

end NUMINAMATH_CALUDE_men_added_to_group_l3694_369455


namespace NUMINAMATH_CALUDE_beckett_olaf_age_difference_l3694_369435

/-- Given the ages of four people satisfying certain conditions, prove that Beckett is 8 years younger than Olaf. -/
theorem beckett_olaf_age_difference :
  ∀ (beckett_age olaf_age shannen_age jack_age : ℕ),
    beckett_age = 12 →
    ∃ (x : ℕ), beckett_age + x = olaf_age →
    shannen_age + 2 = olaf_age →
    jack_age = 2 * shannen_age + 5 →
    beckett_age + olaf_age + shannen_age + jack_age = 71 →
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_beckett_olaf_age_difference_l3694_369435


namespace NUMINAMATH_CALUDE_max_single_color_coins_l3694_369410

/-- Represents the state of coins -/
structure CoinState where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Represents a coin exchange -/
inductive Exchange
  | RedYellowToBlue
  | RedBlueToYellow
  | YellowBlueToRed

/-- Applies an exchange to a coin state -/
def applyExchange (state : CoinState) (exchange : Exchange) : CoinState :=
  match exchange with
  | Exchange.RedYellowToBlue => 
      { red := state.red - 1, yellow := state.yellow - 1, blue := state.blue + 1 }
  | Exchange.RedBlueToYellow => 
      { red := state.red - 1, yellow := state.yellow + 1, blue := state.blue - 1 }
  | Exchange.YellowBlueToRed => 
      { red := state.red + 1, yellow := state.yellow - 1, blue := state.blue - 1 }

/-- Checks if all coins are of the same color -/
def isSingleColor (state : CoinState) : Bool :=
  (state.red = 0 && state.blue = 0) || 
  (state.red = 0 && state.yellow = 0) || 
  (state.yellow = 0 && state.blue = 0)

/-- Counts the total number of coins -/
def totalCoins (state : CoinState) : Nat :=
  state.red + state.yellow + state.blue

/-- The main theorem to prove -/
theorem max_single_color_coins :
  ∃ (finalState : CoinState) (exchanges : List Exchange), 
    let initialState := { red := 3, yellow := 4, blue := 5 : CoinState }
    finalState = exchanges.foldl applyExchange initialState ∧
    isSingleColor finalState ∧
    totalCoins finalState = 7 ∧
    finalState.yellow = 7 ∧
    ∀ (otherState : CoinState) (otherExchanges : List Exchange),
      otherState = otherExchanges.foldl applyExchange initialState →
      isSingleColor otherState →
      totalCoins otherState ≤ totalCoins finalState :=
by
  sorry


end NUMINAMATH_CALUDE_max_single_color_coins_l3694_369410


namespace NUMINAMATH_CALUDE_midpoint_locus_l3694_369411

/-- Given a circle x^2 + y^2 = 1, point A(1,0), and triangle ABC inscribed in the circle
    with angle BAC = 60°, the locus of the midpoint of BC as BC moves on the circle
    is described by the equation x^2 + y^2 = 1/4 for x < 1/4 -/
theorem midpoint_locus (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ),
    x1^2 + y1^2 = 1 ∧
    x2^2 + y2^2 = 1 ∧
    x = (x1 + x2) / 2 ∧
    y = (y1 + y2) / 2 ∧
    (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + y2^2 - (x1 - x2)^2 - (y1 - y2)^2 = 1) →
  x < 1/4 →
  x^2 + y^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_midpoint_locus_l3694_369411


namespace NUMINAMATH_CALUDE_last_three_digits_of_seven_to_123_l3694_369490

theorem last_three_digits_of_seven_to_123 : 7^123 ≡ 773 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_seven_to_123_l3694_369490


namespace NUMINAMATH_CALUDE_unfenced_length_l3694_369499

theorem unfenced_length 
  (field_side : ℝ) 
  (wire_cost : ℝ) 
  (budget : ℝ) 
  (h1 : field_side = 5000)
  (h2 : wire_cost = 30)
  (h3 : budget = 120000) : 
  field_side * 4 - (budget / wire_cost) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_unfenced_length_l3694_369499


namespace NUMINAMATH_CALUDE_carrots_theorem_l3694_369498

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 8

/-- The number of carrots Mary grew -/
def mary_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := sandy_carrots + mary_carrots

theorem carrots_theorem : total_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_carrots_theorem_l3694_369498


namespace NUMINAMATH_CALUDE_product_nine_sum_zero_l3694_369421

theorem product_nine_sum_zero (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 9 →
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_nine_sum_zero_l3694_369421


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l3694_369407

theorem sqrt_expression_simplification :
  (Real.sqrt 7 - 1)^2 - (Real.sqrt 14 - Real.sqrt 2) * (Real.sqrt 14 + Real.sqrt 2) = -4 - 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l3694_369407


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l3694_369459

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

theorem sqrt_three_times_sqrt_two_equals_sqrt_six : 
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l3694_369459


namespace NUMINAMATH_CALUDE_sum_three_digit_numbers_eq_255744_l3694_369402

/-- The sum of all three-digit natural numbers with digits ranging from 1 to 8 -/
def sum_three_digit_numbers : ℕ :=
  let digit_sum : ℕ := (8 * 9) / 2  -- Sum of digits from 1 to 8
  let digit_count : ℕ := 8 * 8      -- Number of times each digit appears in each place
  let place_sum : ℕ := digit_sum * digit_count
  place_sum * 111

theorem sum_three_digit_numbers_eq_255744 :
  sum_three_digit_numbers = 255744 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_digit_numbers_eq_255744_l3694_369402


namespace NUMINAMATH_CALUDE_unique_polynomial_composition_l3694_369484

-- Define the polynomial P(x) = a x^2 + b x + c
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define a general n-degree polynomial
def NPolynomial (n : ℕ) := ℝ → ℝ

theorem unique_polynomial_composition (a b c : ℝ) (ha : a ≠ 0) (n : ℕ) :
  ∃! Q : NPolynomial n, ∀ x : ℝ, Q (P a b c x) = P a b c (Q x) :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_composition_l3694_369484


namespace NUMINAMATH_CALUDE_carla_book_count_l3694_369428

theorem carla_book_count (ceiling_tiles : ℕ) (tuesday_count : ℕ) : 
  ceiling_tiles = 38 → 
  tuesday_count = 301 → 
  ∃ (books : ℕ), 2 * ceiling_tiles + 3 * books = tuesday_count ∧ books = 75 :=
by sorry

end NUMINAMATH_CALUDE_carla_book_count_l3694_369428


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l3694_369425

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given the man is 22 years older than his son and the son's present age is 20 years. -/
theorem mans_age_twice_sons_age (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 20 → age_difference = 22 → 
  ∃ (years : ℕ), years = 2 ∧ 
    (son_age + years) * 2 = (son_age + age_difference + years) := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l3694_369425


namespace NUMINAMATH_CALUDE_organizing_teams_count_l3694_369429

theorem organizing_teams_count (total_members senior_members team_size : ℕ) 
  (h1 : total_members = 12)
  (h2 : senior_members = 5)
  (h3 : team_size = 5) :
  (Nat.choose total_members team_size) - 
  ((Nat.choose (total_members - senior_members) team_size) + 
   (Nat.choose senior_members 1 * Nat.choose (total_members - senior_members) (team_size - 1))) = 596 := by
sorry

end NUMINAMATH_CALUDE_organizing_teams_count_l3694_369429


namespace NUMINAMATH_CALUDE_inequality_solution_l3694_369415

theorem inequality_solution (x : ℝ) (hx : x ≠ 0) (hx2 : x^2 ≠ 6) :
  |4 * x^2 - 32 / x| + |x^2 + 5 / (x^2 - 6)| ≤ |3 * x^2 - 5 / (x^2 - 6) - 32 / x| ↔
  (x > -Real.sqrt 6 ∧ x ≤ -Real.sqrt 5) ∨
  (x ≥ -1 ∧ x < 0) ∨
  (x ≥ 1 ∧ x ≤ 2) ∨
  (x ≥ Real.sqrt 5 ∧ x < Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3694_369415


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3694_369474

theorem geometric_sequence_sum (a r : ℝ) (h1 : a * (1 - r^1010) / (1 - r) = 100) 
  (h2 : a * (1 - r^2020) / (1 - r) = 190) : 
  a * (1 - r^3030) / (1 - r) = 271 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3694_369474


namespace NUMINAMATH_CALUDE_correct_num_kids_l3694_369478

/-- The number of kids in a group that can wash whiteboards -/
def num_kids : ℕ := 4

/-- The number of whiteboards the group can wash in 20 minutes -/
def group_whiteboards : ℕ := 3

/-- The time in minutes it takes the group to wash their whiteboards -/
def group_time : ℕ := 20

/-- The number of whiteboards one kid can wash -/
def one_kid_whiteboards : ℕ := 6

/-- The time in minutes it takes one kid to wash their whiteboards -/
def one_kid_time : ℕ := 160

/-- Theorem stating that the number of kids in the group is correct -/
theorem correct_num_kids :
  num_kids * (group_whiteboards * one_kid_time) = (one_kid_whiteboards * group_time) :=
by sorry

end NUMINAMATH_CALUDE_correct_num_kids_l3694_369478


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_bound_l3694_369426

/-- Given a quadrilateral with sides a, b, c, and d, the surface area of a rectangular prism
    with edges a, b, and c meeting at a vertex is at most (a+b+c)^2 - d^2/3 -/
theorem rectangular_prism_surface_area_bound 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (quad : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) :
  2 * (a * b + b * c + c * a) ≤ (a + b + c)^2 - d^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_bound_l3694_369426


namespace NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3694_369420

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sum_odd_integers (start : ℕ) (n : ℕ) : ℕ :=
  let last := start + 2 * (n - 1)
  n * (start + last) / 2

/-- Theorem: The sum of the first 15 odd positive integers starting from 5 is 255 -/
theorem sum_first_15_odd_from_5 : sum_odd_integers 5 15 = 255 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3694_369420


namespace NUMINAMATH_CALUDE_cube_probability_l3694_369414

-- Define the type for cube faces
inductive CubeFace
| Face1 | Face2 | Face3 | Face4 | Face5 | Face6

-- Define the type for numbers
inductive Number
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

-- Define a function to check if two numbers are consecutive
def isConsecutive (n1 n2 : Number) : Prop := sorry

-- Define a function to check if two faces share an edge
def sharesEdge (f1 f2 : CubeFace) : Prop := sorry

-- Define the type for cube configuration
def CubeConfig := CubeFace → Option Number

-- Define a valid cube configuration
def isValidConfig (config : CubeConfig) : Prop :=
  (∀ f1 f2 : CubeFace, f1 ≠ f2 → config f1 ≠ config f2) ∧
  (∀ f1 f2 : CubeFace, sharesEdge f1 f2 →
    ∀ n1 n2 : Number, config f1 = some n1 → config f2 = some n2 →
      ¬isConsecutive n1 n2)

-- Define the total number of possible configurations
def totalConfigs : ℕ := sorry

-- Define the number of valid configurations
def validConfigs : ℕ := sorry

-- The main theorem
theorem cube_probability :
  (validConfigs : ℚ) / totalConfigs = 1 / 672 := by sorry

end NUMINAMATH_CALUDE_cube_probability_l3694_369414


namespace NUMINAMATH_CALUDE_allens_blocks_combinations_l3694_369427

/-- Given conditions for Allen's blocks problem -/
structure BlocksProblem where
  total_blocks : ℕ
  num_shapes : ℕ
  blocks_per_color : ℕ

/-- Calculate the number of color and shape combinations -/
def calculate_combinations (problem : BlocksProblem) : ℕ :=
  let num_colors := problem.total_blocks / problem.blocks_per_color
  problem.num_shapes * num_colors

/-- Theorem: The number of color and shape combinations is 80 -/
theorem allens_blocks_combinations (problem : BlocksProblem) 
  (h1 : problem.total_blocks = 100)
  (h2 : problem.num_shapes = 4)
  (h3 : problem.blocks_per_color = 5) :
  calculate_combinations problem = 80 := by
  sorry

#eval calculate_combinations ⟨100, 4, 5⟩

end NUMINAMATH_CALUDE_allens_blocks_combinations_l3694_369427


namespace NUMINAMATH_CALUDE_tailoring_cost_l3694_369409

theorem tailoring_cost (num_shirts num_pants : ℕ) (shirt_time : ℝ) (hourly_rate : ℝ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 1.5 →
  hourly_rate = 30 →
  (num_shirts * shirt_time + num_pants * (2 * shirt_time)) * hourly_rate = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tailoring_cost_l3694_369409


namespace NUMINAMATH_CALUDE_problem_statement_l3694_369469

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + y * z + z * x ≠ 1)
  (h2 : (x^2 - 1) * (y^2 - 1) / (x * y) + (y^2 - 1) * (z^2 - 1) / (y * z) + (z^2 - 1) * (x^2 - 1) / (z * x) = 4) :
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 1) ∧
  (9 * (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z * (x * y + y * z + z * x)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3694_369469


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l3694_369466

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90) 
  (h2 : male_count = 8) 
  (h3 : male_average = 84) 
  (h4 : female_average = 92) : 
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧ 
    female_count = 24 :=
by sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l3694_369466


namespace NUMINAMATH_CALUDE_problem_statement_l3694_369485

theorem problem_statement (n m : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + n) = x^2 + m*x - 15) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3694_369485


namespace NUMINAMATH_CALUDE_range_of_a_l3694_369438

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) : p a ∧ q a → a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3694_369438


namespace NUMINAMATH_CALUDE_distance_between_points_l3694_369488

def point_distance (a b : ℝ) : ℝ := |a - b|

theorem distance_between_points (A B : ℝ) :
  A = 3 →
  (B = 9 ∨ B = -9) →
  (point_distance A B = 6 ∨ point_distance A B = 12) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3694_369488


namespace NUMINAMATH_CALUDE_halfway_fraction_l3694_369472

theorem halfway_fraction (a b c : ℚ) : 
  a = 1/4 → b = 1/2 → c = (a + b) / 2 → c = 3/8 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3694_369472


namespace NUMINAMATH_CALUDE_distribute_four_into_two_l3694_369446

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 16 ways to distribute 4 distinguishable balls into 2 distinguishable boxes -/
theorem distribute_four_into_two : distribute 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_into_two_l3694_369446


namespace NUMINAMATH_CALUDE_dessert_preference_l3694_369465

theorem dessert_preference (total : Nat) (apple : Nat) (chocolate : Nat) (pumpkin : Nat) (none : Nat)
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 17)
  (h4 : pumpkin = 10)
  (h5 : none = 15)
  (h6 : total ≥ apple + chocolate + pumpkin - none) :
  ∃ x : Nat, x = 7 ∧ x ≤ apple ∧ x ≤ chocolate ∧ x ≤ pumpkin ∧
  apple + chocolate + pumpkin - 2*x = total - none :=
by sorry

end NUMINAMATH_CALUDE_dessert_preference_l3694_369465


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l3694_369423

theorem complex_in_first_quadrant (m : ℝ) (h : m < 1) :
  let z : ℂ := (1 - m) + I
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l3694_369423


namespace NUMINAMATH_CALUDE_cubic_factorization_l3694_369413

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3694_369413


namespace NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l3694_369436

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l3694_369436


namespace NUMINAMATH_CALUDE_stable_poly_characterization_l3694_369449

-- Define the set K of positive integers not containing the digit 7
def K : Set Nat := {n : Nat | n > 0 ∧ ∀ d, d ∈ n.digits 10 → d ≠ 7}

-- Define a polynomial with nonnegative coefficients
def NonNegativePoly (f : Nat → Nat) : Prop :=
  ∃ (coeffs : List Nat), ∀ x, f x = (coeffs.enum.map (λ (i, a) => a * x^i)).sum

-- Define the stable property for a polynomial
def Stable (f : Nat → Nat) : Prop :=
  ∀ x, x ∈ K → f x ∈ K

-- Theorem statement
theorem stable_poly_characterization (f : Nat → Nat) 
  (h_nonneg : NonNegativePoly f) (h_stable : Stable f) :
  (∃ e k, k ∈ K ∧ ∀ x, f x = 10^e * x + k) ∨
  (∃ e, ∀ x, f x = 10^e * x) ∨
  (∃ k, k ∈ K ∧ ∀ x, f x = k) :=
sorry

end NUMINAMATH_CALUDE_stable_poly_characterization_l3694_369449


namespace NUMINAMATH_CALUDE_annual_price_decrease_l3694_369416

def price_2001 : ℝ := 1950
def price_2009 : ℝ := 1670
def year_2001 : ℕ := 2001
def year_2009 : ℕ := 2009

theorem annual_price_decrease :
  (price_2001 - price_2009) / (year_2009 - year_2001 : ℝ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_annual_price_decrease_l3694_369416


namespace NUMINAMATH_CALUDE_select_and_arrange_l3694_369487

theorem select_and_arrange (n m : ℕ) (hn : n = 7) (hm : m = 4) :
  (Nat.choose n m) * (Nat.factorial m) = 840 := by
  sorry

end NUMINAMATH_CALUDE_select_and_arrange_l3694_369487


namespace NUMINAMATH_CALUDE_lunch_meeting_probability_l3694_369406

/-- The probability of Janet and Donald meeting for lunch -/
theorem lunch_meeting_probability :
  let arrival_interval : ℝ := 60
  let janet_wait_time : ℝ := 15
  let donald_wait_time : ℝ := 5
  let meeting_condition (x y : ℝ) : Prop := |x - y| ≤ min donald_wait_time janet_wait_time
  let total_area : ℝ := arrival_interval ^ 2
  let meeting_area : ℝ := arrival_interval * (2 * min donald_wait_time janet_wait_time)
  (meeting_area / total_area : ℝ) = 1/6 := by
sorry

end NUMINAMATH_CALUDE_lunch_meeting_probability_l3694_369406


namespace NUMINAMATH_CALUDE_function_minimum_l3694_369458

/-- The function f(x) = x³ - 3x² + 4 attains its minimum value at x = 2 -/
theorem function_minimum (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 3*x^2 + 4) :
  ∃ x₀ : ℝ, x₀ = 2 ∧ ∀ x, f x₀ ≤ f x := by sorry

end NUMINAMATH_CALUDE_function_minimum_l3694_369458


namespace NUMINAMATH_CALUDE_fathers_age_l3694_369424

theorem fathers_age (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 5 = (father_age + 5) / 2 →
  father_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l3694_369424


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_6_l3694_369461

theorem greatest_integer_with_gcd_6 :
  ∃ n : ℕ, n < 150 ∧ n.gcd 12 = 6 ∧ ∀ m : ℕ, m < 150 → m.gcd 12 = 6 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_6_l3694_369461


namespace NUMINAMATH_CALUDE_jumping_contest_l3694_369451

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump : ℕ) (frog_extra : ℕ) (mouse_extra : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra = 10)
  (h3 : mouse_extra = 20) :
  (grasshopper_jump + frog_extra + mouse_extra) - grasshopper_jump = 30 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l3694_369451


namespace NUMINAMATH_CALUDE_triangle_inequality_bound_l3694_369401

theorem triangle_inequality_bound (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : b = 2 * a) :
  (a^2 + b^2) / c^2 > 5/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_bound_l3694_369401


namespace NUMINAMATH_CALUDE_height_order_l3694_369462

-- Define the set of children
inductive Child : Type
  | A : Child
  | B : Child
  | C : Child
  | D : Child

-- Define the height relation
def taller_than (x y : Child) : Prop := sorry

-- Define the conditions
axiom A_taller_than_B : taller_than Child.A Child.B
axiom B_shorter_than_C : taller_than Child.C Child.B
axiom D_shorter_than_A : taller_than Child.A Child.D
axiom A_not_tallest : ∃ x : Child, taller_than x Child.A
axiom D_not_shortest : ∃ x : Child, taller_than Child.D x

-- Define the order relation
def in_order (w x y z : Child) : Prop :=
  taller_than w x ∧ taller_than x y ∧ taller_than y z

-- State the theorem
theorem height_order : in_order Child.C Child.A Child.D Child.B := by sorry

end NUMINAMATH_CALUDE_height_order_l3694_369462


namespace NUMINAMATH_CALUDE_smallest_prime_after_eight_nonprimes_l3694_369419

def is_first_prime_after_eight_nonprimes (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, n ≤ k ∧ k < n + 8 → ¬Nat.Prime k) ∧
    (∀ q : ℕ, Nat.Prime q → q < p → q ≤ n - 1 ∨ q ≥ n + 8)

theorem smallest_prime_after_eight_nonprimes :
  is_first_prime_after_eight_nonprimes 59 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_eight_nonprimes_l3694_369419


namespace NUMINAMATH_CALUDE_min_value_theorem_l3694_369492

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_constraint : 3 * m + n = 1) :
  1 / m + 3 / n ≥ 12 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 3 * m₀ + n₀ = 1 ∧ 1 / m₀ + 3 / n₀ = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3694_369492


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3694_369482

theorem arithmetic_mean_after_removal (s : Finset ℝ) (a b c : ℝ) :
  s.card = 80 →
  a = 50 ∧ b = 60 ∧ c = 70 →
  a ∈ s ∧ b ∈ s ∧ c ∈ s →
  (s.sum id) / s.card = 45 →
  ((s.sum id) - (a + b + c)) / (s.card - 3) = 3420 / 77 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3694_369482


namespace NUMINAMATH_CALUDE_complex_real_condition_l3694_369404

/-- If z = (2+mi)/(1+i) is a real number and m is a real number, then m = 2 -/
theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (z.im = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3694_369404


namespace NUMINAMATH_CALUDE_danny_collection_difference_l3694_369454

/-- The number of wrappers Danny has in his collection -/
def wrappers : ℕ := 67

/-- The number of soda cans Danny has in his collection -/
def soda_cans : ℕ := 22

/-- The difference between the number of wrappers and soda cans in Danny's collection -/
def wrapper_soda_difference : ℕ := wrappers - soda_cans

theorem danny_collection_difference :
  wrapper_soda_difference = 45 := by sorry

end NUMINAMATH_CALUDE_danny_collection_difference_l3694_369454


namespace NUMINAMATH_CALUDE_a_range_l3694_369447

def p (a : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - a*x + y + 1 = 0

def q (a : ℝ) : Prop := ∃ (x y : ℝ), 2*a*x + (1-a)*y + 1 = 0 ∧ (2*a)/(a-1) > 1

def range_of_a (a : ℝ) : Prop := a ∈ Set.Icc (-Real.sqrt 3) (-1) ∪ Set.Ioc 1 (Real.sqrt 3)

theorem a_range (a : ℝ) :
  (∀ a, p a ∨ q a) ∧ (∀ a, ¬(p a ∧ q a)) →
  range_of_a a :=
sorry

end NUMINAMATH_CALUDE_a_range_l3694_369447


namespace NUMINAMATH_CALUDE_negation_equivalence_l3694_369443

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3694_369443


namespace NUMINAMATH_CALUDE_add_3031_minutes_to_initial_equals_final_l3694_369432

-- Define a structure for date and time
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

-- Define the function to add minutes to a DateTime
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

-- Define the initial and final DateTimes
def initialDateTime : DateTime :=
  { year := 2020, month := 12, day := 31, hour := 17, minute := 0 }

def finalDateTime : DateTime :=
  { year := 2021, month := 1, day := 2, hour := 19, minute := 31 }

-- Theorem to prove
theorem add_3031_minutes_to_initial_equals_final :
  addMinutes initialDateTime 3031 = finalDateTime :=
sorry

end NUMINAMATH_CALUDE_add_3031_minutes_to_initial_equals_final_l3694_369432


namespace NUMINAMATH_CALUDE_daejun_marbles_l3694_369457

/-- The number of bags Daejun has -/
def num_bags : ℕ := 20

/-- The number of marbles in each bag -/
def marbles_per_bag : ℕ := 156

/-- The total number of marbles Daejun has -/
def total_marbles : ℕ := num_bags * marbles_per_bag

theorem daejun_marbles : total_marbles = 3120 := by
  sorry

end NUMINAMATH_CALUDE_daejun_marbles_l3694_369457


namespace NUMINAMATH_CALUDE_excavation_volume_scientific_notation_l3694_369441

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem excavation_volume_scientific_notation :
  toScientificNotation 632000 = ScientificNotation.mk 6.32 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_excavation_volume_scientific_notation_l3694_369441


namespace NUMINAMATH_CALUDE_six_digit_number_rotation_l3694_369453

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def rotate_last_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 100000 + r

theorem six_digit_number_rotation (n : ℕ) :
  is_six_digit n ∧ rotate_last_to_first n = n / 3 → n = 428571 ∨ n = 857142 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_rotation_l3694_369453


namespace NUMINAMATH_CALUDE_extreme_point_of_f_l3694_369430

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 1)^3 + 2

-- State the theorem
theorem extreme_point_of_f :
  ∃! x : ℝ, ∀ y : ℝ, f y ≥ f x :=
  by sorry

end NUMINAMATH_CALUDE_extreme_point_of_f_l3694_369430


namespace NUMINAMATH_CALUDE_min_max_f_l3694_369471

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = max) ∧
    min = -3 * Real.pi / 2 ∧
    max = Real.pi / 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_min_max_f_l3694_369471


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3694_369494

/-- Represents the distance a truck can travel -/
def truck_distance (miles_per_gallon : ℝ) (initial_gallons : ℝ) (added_gallons : ℝ) : ℝ :=
  miles_per_gallon * (initial_gallons + added_gallons)

/-- Theorem: A truck traveling 3 miles per gallon with 12 gallons initially and 18 gallons added can travel 90 miles -/
theorem truck_travel_distance :
  truck_distance 3 12 18 = 90 := by
  sorry

#eval truck_distance 3 12 18

end NUMINAMATH_CALUDE_truck_travel_distance_l3694_369494


namespace NUMINAMATH_CALUDE_lindsey_september_savings_l3694_369412

/-- The amount of money Lindsey saved in September -/
def september_savings : ℕ := sorry

/-- The amount of money Lindsey saved in October -/
def october_savings : ℕ := 37

/-- The amount of money Lindsey saved in November -/
def november_savings : ℕ := 11

/-- The amount of money Lindsey's mom gave her -/
def mom_gift : ℕ := 25

/-- The cost of the video game Lindsey bought -/
def video_game_cost : ℕ := 87

/-- The amount of money Lindsey had left after buying the video game -/
def money_left : ℕ := 36

/-- Theorem stating that Lindsey saved $50 in September -/
theorem lindsey_september_savings :
  september_savings = 50 ∧
  september_savings + october_savings + november_savings > 75 ∧
  september_savings + october_savings + november_savings + mom_gift = video_game_cost + money_left :=
sorry

end NUMINAMATH_CALUDE_lindsey_september_savings_l3694_369412


namespace NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l3694_369440

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (-2, 3)
theorem image_of_negative_two_three :
  f (-2, 3) = (1, -6) := by sorry

-- Theorem for the pre-image of (2, -3)
theorem preimage_of_two_negative_three :
  {p : ℝ × ℝ | f p = (2, -3)} = {(-1, 3), (3, -1)} := by sorry

end NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l3694_369440


namespace NUMINAMATH_CALUDE_work_time_ratio_l3694_369452

/-- Given two workers A and B who can complete a job together in 6 days,
    and B can complete the job alone in 36 days,
    prove that the ratio of the time A takes to complete the job alone
    to the time B takes is 1:5. -/
theorem work_time_ratio
  (time_together : ℝ)
  (time_B : ℝ)
  (h_together : time_together = 6)
  (h_B : time_B = 36)
  (time_A : ℝ)
  (h_combined_rate : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A / time_B = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l3694_369452
