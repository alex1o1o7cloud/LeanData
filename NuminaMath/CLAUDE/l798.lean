import Mathlib

namespace NUMINAMATH_CALUDE_unique_six_digit_square_split_l798_79845

def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2

def contains_no_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d ≠ 10

theorem unique_six_digit_square_split :
  ∃! n : ℕ,
    100000 ≤ n ∧ n ≤ 999999 ∧
    (∃ k : ℕ, n = k^2) ∧
    (∃ a b : ℕ, n = a * 1000 + b ∧
      is_three_digit_square a ∧
      is_three_digit_square b ∧
      contains_no_zero a ∧
      contains_no_zero b) :=
sorry

end NUMINAMATH_CALUDE_unique_six_digit_square_split_l798_79845


namespace NUMINAMATH_CALUDE_prism_volume_l798_79899

/-- Given a right rectangular prism with face areas 30 cm², 50 cm², and 75 cm², 
    its volume is 335 cm³. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 50) 
  (h3 : b * c = 75) : 
  a * b * c = 335 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l798_79899


namespace NUMINAMATH_CALUDE_sum_not_five_implies_not_two_or_not_three_l798_79836

theorem sum_not_five_implies_not_two_or_not_three (a b : ℝ) : 
  a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_sum_not_five_implies_not_two_or_not_three_l798_79836


namespace NUMINAMATH_CALUDE_max_area_rectangle_l798_79859

/-- The maximum area of a rectangle with perimeter 40 cm is 100 square centimeters. -/
theorem max_area_rectangle (x y : ℝ) (h : x + y = 20) : 
  x * y ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l798_79859


namespace NUMINAMATH_CALUDE_parabola_translation_l798_79825

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (p : Parabola) :
  p.a = -2 ∧ p.b = -4 ∧ p.c = -6 →
  translate p 1 3 = Parabola.mk (-2) 0 (-1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l798_79825


namespace NUMINAMATH_CALUDE_jinx_hak_not_flog_l798_79857

-- Define the sets
variable (U : Type) -- Universe set
variable (Flog Grep Hak Jinx : Set U)

-- Define the given conditions
variable (h1 : Flog ⊆ Grep)
variable (h2 : Hak ⊆ Grep)
variable (h3 : Hak ⊆ Jinx)
variable (h4 : Flog ∩ Jinx = ∅)

-- Theorem to prove
theorem jinx_hak_not_flog : 
  Jinx ⊆ Hak ∧ ∃ x, x ∈ Jinx ∧ x ∉ Flog :=
sorry

end NUMINAMATH_CALUDE_jinx_hak_not_flog_l798_79857


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l798_79832

theorem arithmetic_progression_of_primes (p q r d : ℕ) : 
  Prime p → Prime q → Prime r → 
  p > 3 → q > 3 → r > 3 →
  q = p + d → r = p + 2*d → 
  6 ∣ d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l798_79832


namespace NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l798_79822

theorem tangent_and_trigonometric_identity (α : Real) 
  (h : Real.tan (α + π/3) = 2 * Real.sqrt 3) : 
  (Real.tan (α - 2*π/3) = 2 * Real.sqrt 3) ∧ 
  (2 * Real.sin α ^ 2 - Real.cos α ^ 2 = -43/52) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l798_79822


namespace NUMINAMATH_CALUDE_two_fifths_of_number_l798_79856

theorem two_fifths_of_number (x : ℚ) : (2 / 9 : ℚ) * x = 10 → (2 / 5 : ℚ) * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_number_l798_79856


namespace NUMINAMATH_CALUDE_problem_statement_l798_79897

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x > 0, x > 0 → 6 - 1 / x ≤ 9 * x) ∧
  (a^2 + 9 * b^2 + 2 * a * b = a^2 * b^2 → a * b ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l798_79897


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l798_79882

-- Define the ☆ operation
def star (m n : ℝ) : ℝ := m^2 - m*n + n

-- Theorem statements
theorem problem_1 : star 3 4 = 1 := by sorry

theorem problem_2 : star (-1) (star 2 (-3)) = 15 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l798_79882


namespace NUMINAMATH_CALUDE_chord_length_l798_79887

-- Define the circle C
def Circle (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - m)^2 + (p.2 - n)^2 = 4}

-- Define the theorem
theorem chord_length
  (m n : ℝ) -- Center of the circle
  (A B : ℝ × ℝ) -- Points on the circle
  (hA : A ∈ Circle m n) -- A is on the circle
  (hB : B ∈ Circle m n) -- B is on the circle
  (hAB : A ≠ B) -- A and B are different points
  (h_sum : ‖(A.1 - m, A.2 - n) + (B.1 - m, B.2 - n)‖ = 2 * Real.sqrt 3) -- |→CA + →CB| = 2√3
  : ‖(A.1 - B.1, A.2 - B.2)‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l798_79887


namespace NUMINAMATH_CALUDE_quadratic_factorization_l798_79862

theorem quadratic_factorization (x : ℝ) : x^2 - 30*x + 225 = (x - 15)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l798_79862


namespace NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l798_79800

theorem ceiling_sum_of_square_roots : 
  ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l798_79800


namespace NUMINAMATH_CALUDE_max_programs_max_programs_achievable_max_programs_optimal_l798_79820

theorem max_programs (n : ℕ) : n ≤ 4 :=
  sorry

theorem max_programs_achievable : ∃ (P : Fin 4 → Finset (Fin 12)),
  (∀ i : Fin 4, (P i).card = 6) ∧
  (∀ i j : Fin 4, i ≠ j → (P i ∩ P j).card ≤ 2) :=
  sorry

theorem max_programs_optimal :
  ¬∃ (P : Fin 5 → Finset (Fin 12)),
    (∀ i : Fin 5, (P i).card = 6) ∧
    (∀ i j : Fin 5, i ≠ j → (P i ∩ P j).card ≤ 2) :=
  sorry

end NUMINAMATH_CALUDE_max_programs_max_programs_achievable_max_programs_optimal_l798_79820


namespace NUMINAMATH_CALUDE_smallest_marble_collection_l798_79874

theorem smallest_marble_collection (M : ℕ) : 
  M > 1 → 
  M % 5 = 2 → 
  M % 6 = 2 → 
  M % 7 = 2 → 
  (∀ n : ℕ, n > 1 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ n % 7 = 2 → n ≥ M) → 
  M = 212 :=
by sorry

end NUMINAMATH_CALUDE_smallest_marble_collection_l798_79874


namespace NUMINAMATH_CALUDE_square_5_on_top_l798_79894

/-- Represents a square on the paper grid -/
structure Square :=
  (number : Nat)
  (row : Nat)
  (col : Nat)

/-- Represents the paper grid -/
def Grid := List Square

/-- Defines the initial configuration of the grid -/
def initialGrid : Grid :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20].map
    (fun n => ⟨n, (n - 1) / 5 + 1, (n - 1) % 5 + 1⟩)

/-- Performs a folding operation on the grid -/
def fold (g : Grid) (foldType : String) : Grid := sorry

/-- Theorem stating that after all folding operations, square 5 is on top -/
theorem square_5_on_top (g : Grid) (h : g = initialGrid) :
  (fold (fold (fold (fold g "left_third") "right_third") "bottom_half") "top_half").head?.map Square.number = some 5 := by sorry

end NUMINAMATH_CALUDE_square_5_on_top_l798_79894


namespace NUMINAMATH_CALUDE_ashley_champagne_bottles_l798_79844

/-- The number of bottles of champagne needed for a wedding toast -/
def bottles_needed (glasses_per_guest : ℕ) (num_guests : ℕ) (servings_per_bottle : ℕ) : ℕ :=
  (glasses_per_guest * num_guests + servings_per_bottle - 1) / servings_per_bottle

/-- Proof that Ashley needs 40 bottles of champagne for her wedding toast -/
theorem ashley_champagne_bottles :
  bottles_needed 2 120 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashley_champagne_bottles_l798_79844


namespace NUMINAMATH_CALUDE_number_of_valid_divisors_l798_79810

def total_marbles : ℕ := 720

theorem number_of_valid_divisors :
  (Finset.filter (fun m => m > 1 ∧ m < total_marbles ∧ total_marbles % m = 0) 
    (Finset.range (total_marbles + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_divisors_l798_79810


namespace NUMINAMATH_CALUDE_jason_attended_11_games_this_month_l798_79851

/-- Represents the number of football games Jason attended or plans to attend -/
structure FootballGames where
  lastMonth : Nat
  thisMonth : Nat
  nextMonth : Nat
  total : Nat

/-- Given information about Jason's football game attendance -/
def jasonGames : FootballGames where
  lastMonth := 17
  thisMonth := 11 -- This is what we want to prove
  nextMonth := 16
  total := 44

/-- Theorem stating that Jason attended 11 games this month -/
theorem jason_attended_11_games_this_month :
  jasonGames.thisMonth = 11 ∧
  jasonGames.total = jasonGames.lastMonth + jasonGames.thisMonth + jasonGames.nextMonth :=
by sorry

end NUMINAMATH_CALUDE_jason_attended_11_games_this_month_l798_79851


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l798_79813

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b are parallel, prove that x = -6 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l798_79813


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l798_79893

/-- Given two circles with centers (1,3) and (15,8) and radii 3 and 10 respectively,
    this theorem proves that the y-intercept of their common external tangent
    with positive slope is 518/1197. -/
theorem common_external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (1, 3)
  let r1 : ℝ := 3
  let c2 : ℝ × ℝ := (15, 8)
  let r2 : ℝ := 10
  let m : ℝ := (8 - 3) / (15 - 1)  -- slope of line connecting centers
  let tan_2theta : ℝ := (2 * m) / (1 - m^2)  -- tangent of double angle
  let m_tangent : ℝ := Real.sqrt (tan_2theta / (1 + tan_2theta))  -- slope of tangent line
  let x_intercept : ℝ := -(3 - m * 1) / m  -- x-intercept of line connecting centers
  ∃ b : ℝ, b = m_tangent * (-x_intercept) ∧ b = 518 / 1197 :=
by sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l798_79893


namespace NUMINAMATH_CALUDE_cylinder_radius_equals_8_l798_79816

/-- Given a cylinder and a cone with equal volumes, prove that the cylinder's radius is 8 cm -/
theorem cylinder_radius_equals_8 (h_cyl : ℝ) (r_cyl : ℝ) (h_cone : ℝ) (r_cone : ℝ)
  (h_cyl_val : h_cyl = 2)
  (h_cone_val : h_cone = 6)
  (r_cone_val : r_cone = 8)
  (volume_equal : π * r_cyl^2 * h_cyl = (1/3) * π * r_cone^2 * h_cone) :
  r_cyl = 8 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_equals_8_l798_79816


namespace NUMINAMATH_CALUDE_school_dinner_drink_choice_l798_79829

theorem school_dinner_drink_choice (total_students : ℕ) 
  (juice_percentage : ℚ) (water_percentage : ℚ) (juice_students : ℕ) :
  juice_percentage = 3/4 →
  water_percentage = 1/4 →
  juice_students = 90 →
  ∃ water_students : ℕ, water_students = 30 ∧ 
    (juice_students : ℚ) / total_students = juice_percentage ∧
    (water_students : ℚ) / total_students = water_percentage :=
by sorry

end NUMINAMATH_CALUDE_school_dinner_drink_choice_l798_79829


namespace NUMINAMATH_CALUDE_no_integer_solutions_to_3x2_plus_7y2_eq_z4_l798_79869

theorem no_integer_solutions_to_3x2_plus_7y2_eq_z4 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 7 * y^2 = z^4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_to_3x2_plus_7y2_eq_z4_l798_79869


namespace NUMINAMATH_CALUDE_abs_equation_roots_properties_l798_79860

def abs_equation (x : ℝ) : Prop := |x|^2 + 2*|x| - 8 = 0

theorem abs_equation_roots_properties :
  ∃ (root1 root2 : ℝ),
    (abs_equation root1 ∧ abs_equation root2) ∧
    (root1 = 2 ∧ root2 = -2) ∧
    (root1 + root2 = 0) ∧
    (root1 * root2 = -4) := by sorry

end NUMINAMATH_CALUDE_abs_equation_roots_properties_l798_79860


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l798_79854

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 4 = 16 →
  a 1 * a 3 * a 5 = 64 ∨ a 1 * a 3 * a 5 = -64 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l798_79854


namespace NUMINAMATH_CALUDE_tan_alpha_2_implications_l798_79802

theorem tan_alpha_2_implications (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1/3 ∧
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implications_l798_79802


namespace NUMINAMATH_CALUDE_expression_equality_l798_79888

theorem expression_equality (y θ Q : ℝ) (h : 5 * (3 * y + 7 * Real.sin θ) = Q) :
  15 * (9 * y + 21 * Real.sin θ) = 9 * Q := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l798_79888


namespace NUMINAMATH_CALUDE_division_reduction_l798_79884

theorem division_reduction (original : ℕ) (divisor : ℕ) (reduction : ℕ) : 
  original = 72 → divisor = 3 → reduction = 48 → 
  (original : ℚ) / divisor = original - reduction :=
by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l798_79884


namespace NUMINAMATH_CALUDE_jeep_speed_calculation_l798_79818

theorem jeep_speed_calculation (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 420 ∧ original_time = 7 ∧ new_time_factor = 3/2 →
  distance / (new_time_factor * original_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_jeep_speed_calculation_l798_79818


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_six_l798_79833

theorem sum_of_solutions_is_six : 
  ∃ (x₁ x₂ : ℂ), 
    (2 : ℂ) ^ (x₁^2 - 3*x₁ - 2) = (8 : ℂ) ^ (x₁ - 5) ∧
    (2 : ℂ) ^ (x₂^2 - 3*x₂ - 2) = (8 : ℂ) ^ (x₂ - 5) ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 6 ∧
    ∀ (y : ℂ), (2 : ℂ) ^ (y^2 - 3*y - 2) = (8 : ℂ) ^ (y - 5) → y = x₁ ∨ y = x₂ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_six_l798_79833


namespace NUMINAMATH_CALUDE_adams_dog_food_packages_l798_79812

theorem adams_dog_food_packages (cat_packages : ℕ) (cat_cans_per_package : ℕ) (dog_cans_per_package : ℕ) (cat_dog_can_difference : ℕ) :
  cat_packages = 9 →
  cat_cans_per_package = 10 →
  dog_cans_per_package = 5 →
  cat_dog_can_difference = 55 →
  ∃ (dog_packages : ℕ),
    cat_packages * cat_cans_per_package = dog_packages * dog_cans_per_package + cat_dog_can_difference ∧
    dog_packages = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_adams_dog_food_packages_l798_79812


namespace NUMINAMATH_CALUDE_complex_number_value_l798_79821

theorem complex_number_value : Complex.I ^ 2 * (1 + Complex.I) = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_value_l798_79821


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l798_79878

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 30 ∧ q > 30 ∧
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ p' q' : ℕ,
      p'.Prime → q'.Prime →
      p' > 30 → q' > 30 →
      p' ≠ q' →
      p' * q' ≥ 1147 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l798_79878


namespace NUMINAMATH_CALUDE_no_perfect_square_sum_of_prime_powers_l798_79815

theorem no_perfect_square_sum_of_prime_powers (p k m : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ¬∃ x : ℕ, p^k + p^m = x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_sum_of_prime_powers_l798_79815


namespace NUMINAMATH_CALUDE_x_to_y_value_l798_79834

theorem x_to_y_value (x y : ℝ) (h : (x + 2)^2 + |y - 3| = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_value_l798_79834


namespace NUMINAMATH_CALUDE_unique_positive_solution_l798_79827

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x^8 + 8*x^7 + 28*x^6 + 2023*x^5 - 1807*x^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l798_79827


namespace NUMINAMATH_CALUDE_grandma_molly_statues_l798_79838

/-- The number of statues Grandma Molly created in the first year -/
def initial_statues : ℕ := sorry

/-- The total number of statues after four years -/
def total_statues : ℕ := 31

/-- The number of statues broken in the third year -/
def broken_statues : ℕ := 3

theorem grandma_molly_statues :
  initial_statues = 4 ∧
  (4 * initial_statues + 12 - broken_statues + 2 * broken_statues = total_statues) :=
sorry

end NUMINAMATH_CALUDE_grandma_molly_statues_l798_79838


namespace NUMINAMATH_CALUDE_doughnut_cost_calculation_l798_79895

/-- Calculates the total cost of doughnuts for a class -/
theorem doughnut_cost_calculation (total_students : ℕ) 
  (chocolate_lovers : ℕ) (glazed_lovers : ℕ) 
  (chocolate_cost : ℕ) (glazed_cost : ℕ) : 
  total_students = 25 →
  chocolate_lovers = 10 →
  glazed_lovers = 15 →
  chocolate_cost = 2 →
  glazed_cost = 1 →
  chocolate_lovers * chocolate_cost + glazed_lovers * glazed_cost = 35 :=
by
  sorry

#check doughnut_cost_calculation

end NUMINAMATH_CALUDE_doughnut_cost_calculation_l798_79895


namespace NUMINAMATH_CALUDE_complex_cube_absolute_value_l798_79847

theorem complex_cube_absolute_value : 
  Complex.abs ((1 + 2 * Complex.I + 3 - Real.sqrt 3 * Complex.I) ^ 3) = 
  (23 - 4 * Real.sqrt 3) ^ (3/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_absolute_value_l798_79847


namespace NUMINAMATH_CALUDE_vector_operation_l798_79811

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

theorem vector_operation : 
  (3 • a - 2 • b : ℝ × ℝ) = (1, 5) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l798_79811


namespace NUMINAMATH_CALUDE_fibSeriesSum_l798_79806

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the series sum
noncomputable def fibSeries : ℝ := ∑' n : ℕ, (fib (2 * n + 1) : ℝ) / (5 : ℝ) ^ n

-- Theorem statement
theorem fibSeriesSum : fibSeries = 35 / 3 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l798_79806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l798_79819

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 9 - (1/3) * a 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l798_79819


namespace NUMINAMATH_CALUDE_function_identity_l798_79877

theorem function_identity (f : ℕ → ℕ) :
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) ≤ x * (1 + f y)) →
  ∀ x : ℕ, f x = x :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l798_79877


namespace NUMINAMATH_CALUDE_fraction_calculation_l798_79852

theorem fraction_calculation : 
  (((1 / 6 : ℚ) - (1 / 8 : ℚ) + (1 / 9 : ℚ)) / ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 5 : ℚ))) * 3 = 55 / 34 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l798_79852


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_is_two_thirds_l798_79892

def coin_flip_expected_value : ℚ :=
  let p_heads : ℚ := 1/2
  let p_tails : ℚ := 1/3
  let p_edge : ℚ := 1/6
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let win_edge : ℚ := -5
  p_heads * win_heads + p_tails * win_tails + p_edge * win_edge

theorem coin_flip_expected_value_is_two_thirds :
  coin_flip_expected_value = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_is_two_thirds_l798_79892


namespace NUMINAMATH_CALUDE_popcorn_servings_for_jared_and_friends_l798_79807

/-- Calculate the number of popcorn servings needed for a group -/
def popcorn_servings (pieces_per_serving : ℕ) (jared_pieces : ℕ) (num_friends : ℕ) (friend_pieces : ℕ) : ℕ :=
  ((jared_pieces + num_friends * friend_pieces) + pieces_per_serving - 1) / pieces_per_serving

theorem popcorn_servings_for_jared_and_friends :
  popcorn_servings 30 90 3 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_servings_for_jared_and_friends_l798_79807


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l798_79876

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {0, 2, 5}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l798_79876


namespace NUMINAMATH_CALUDE_intersection_with_complement_l798_79808

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {2, 3, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l798_79808


namespace NUMINAMATH_CALUDE_only_negative_number_l798_79823

theorem only_negative_number (a b c d : ℚ) : 
  a = 0 → b = -(-3) → c = -1/2 → d = 3.2 → 
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧ 
  (a ≥ 0 ∧ b ≥ 0 ∧ d ≥ 0) ∧ 
  c < 0 := by
sorry

end NUMINAMATH_CALUDE_only_negative_number_l798_79823


namespace NUMINAMATH_CALUDE_polygon_area_is_400_l798_79814

-- Define the polygon vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (20, 0)
def C : ℝ × ℝ := (30, 10)
def D : ℝ × ℝ := (20, 20)
def E : ℝ × ℝ := (10, 10)
def F : ℝ × ℝ := (0, 20)

-- Define the polygon as a list of vertices
def polygon : List (ℝ × ℝ) := [A, B, C, D, E, F]

-- Function to calculate the area of a polygon given its vertices
def polygonArea (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry -- Implementation not required for this task

-- Theorem statement
theorem polygon_area_is_400 : polygonArea polygon = 400 := by
  sorry -- Proof not required for this task

end NUMINAMATH_CALUDE_polygon_area_is_400_l798_79814


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_negative_l798_79861

theorem triangle_inequality_sum_negative 
  (a b c x y z : ℝ) 
  (h1 : 0 < b - c) 
  (h2 : b - c < a) 
  (h3 : a < b + c) 
  (h4 : a * x + b * y + c * z = 0) : 
  a * y * z + b * z * x + c * x * y < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_negative_l798_79861


namespace NUMINAMATH_CALUDE_solutions_x_fourth_plus_81_l798_79803

theorem solutions_x_fourth_plus_81 :
  {x : ℂ | x^4 + 81 = 0} = {3 + 3*I, -3 - 3*I, -3 + 3*I, 3 - 3*I} := by
  sorry

end NUMINAMATH_CALUDE_solutions_x_fourth_plus_81_l798_79803


namespace NUMINAMATH_CALUDE_parallelogram_sides_l798_79855

/-- Represents a parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  a_positive : 0 < a
  b_positive : 0 < b

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.a + p.b)

/-- The difference between perimeters of adjacent triangles formed by diagonals -/
def triangle_perimeter_difference (p : Parallelogram) : ℝ := abs (p.b - p.a)

theorem parallelogram_sides (p : Parallelogram) 
  (h_perimeter : perimeter p = 44)
  (h_diff : triangle_perimeter_difference p = 6) :
  p.a = 8 ∧ p.b = 14 := by
  sorry

#check parallelogram_sides

end NUMINAMATH_CALUDE_parallelogram_sides_l798_79855


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l798_79837

theorem max_value_of_exponential_difference :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x : ℝ, 5^x - 25^x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l798_79837


namespace NUMINAMATH_CALUDE_smallest_rearranged_multiple_of_nine_l798_79817

/-- A function that returns the digits of a natural number as a list -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- A predicate that checks if two natural numbers have the same digits -/
def same_digits (a b : ℕ) : Prop :=
  digits a = digits b

/-- The theorem stating that 1089 is the smallest natural number
    that when multiplied by 9, results in a number with the same digits -/
theorem smallest_rearranged_multiple_of_nine :
  (∀ n : ℕ, n < 1089 → ¬(same_digits n (9 * n))) ∧
  (same_digits 1089 (9 * 1089)) :=
sorry

end NUMINAMATH_CALUDE_smallest_rearranged_multiple_of_nine_l798_79817


namespace NUMINAMATH_CALUDE_approval_ratio_rounded_l798_79881

/-- The ratio of regions needed for approval to total regions -/
def approval_ratio : ℚ := 8 / 15

/-- Rounding a rational number to the nearest tenth -/
def round_to_tenth (q : ℚ) : ℚ := 
  ⌊q * 10 + 1/2⌋ / 10

theorem approval_ratio_rounded : round_to_tenth approval_ratio = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_approval_ratio_rounded_l798_79881


namespace NUMINAMATH_CALUDE_divisor_power_equation_l798_79866

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- The statement of the problem -/
theorem divisor_power_equation :
  ∀ n k : ℕ+, ∀ p : ℕ,
  Prime p →
  (n : ℕ) ^ (d n) - 1 = p ^ (k : ℕ) →
  ((n = 2 ∧ k = 1 ∧ p = 3) ∨ (n = 3 ∧ k = 3 ∧ p = 2)) :=
by sorry

end NUMINAMATH_CALUDE_divisor_power_equation_l798_79866


namespace NUMINAMATH_CALUDE_inequality_subtraction_l798_79841

theorem inequality_subtraction (a b c d : ℝ) (h1 : a > b) (h2 : c < d) : a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_subtraction_l798_79841


namespace NUMINAMATH_CALUDE_line_equation_of_parabola_points_l798_79839

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 3*y

-- Define the quadratic equation
def quadratic_equation (x p q : ℝ) : Prop := x^2 + p*x + q = 0

theorem line_equation_of_parabola_points (p q : ℝ) (h : p^2 - 4*q > 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    quadratic_equation x₁ p q ∧ quadratic_equation x₂ p q ∧
    x₁ ≠ x₂ ∧
    ∀ (x y : ℝ), (p*x + 3*y + q = 0) ↔ (y - y₁) = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) :=
sorry

end NUMINAMATH_CALUDE_line_equation_of_parabola_points_l798_79839


namespace NUMINAMATH_CALUDE_jarry_secretary_or_treasurer_prob_l798_79826

/-- A club with 10 members, including Jarry -/
structure Club where
  members : Finset Nat
  jarry : Nat
  total_members : members.card = 10
  jarry_in_club : jarry ∈ members

/-- The probability of Jarry being either secretary or treasurer -/
def probability_jarry_secretary_or_treasurer (club : Club) : ℚ :=
  19 / 90

/-- Theorem stating the probability of Jarry being secretary or treasurer -/
theorem jarry_secretary_or_treasurer_prob (club : Club) :
  probability_jarry_secretary_or_treasurer club = 19 / 90 := by
  sorry

end NUMINAMATH_CALUDE_jarry_secretary_or_treasurer_prob_l798_79826


namespace NUMINAMATH_CALUDE_inequality_preservation_l798_79830

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l798_79830


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_three_l798_79891

theorem sqrt_sum_eq_three (a : ℝ) (h : a + 1/a = 7) : 
  Real.sqrt a + 1 / Real.sqrt a = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_three_l798_79891


namespace NUMINAMATH_CALUDE_inverse_f_l798_79896

/-- Given a function f: ℝ → ℝ satisfying f(4) = 3 and f(2x) = 2f(x) + 1 for all x,
    prove that f(128) = 127 -/
theorem inverse_f (f : ℝ → ℝ) (h1 : f 4 = 3) (h2 : ∀ x, f (2 * x) = 2 * f x + 1) :
  f 128 = 127 := by sorry

end NUMINAMATH_CALUDE_inverse_f_l798_79896


namespace NUMINAMATH_CALUDE_inequality_proof_l798_79886

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  2 / ((a + b) * (c + d)) ≤ 1 / Real.sqrt (a * b) + 1 / Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l798_79886


namespace NUMINAMATH_CALUDE_divisibility_condition_exists_divisibility_for_all_implies_equality_l798_79849

-- Part (a)
theorem divisibility_condition_exists (n : ℕ+) :
  ∃ (x y : ℕ+), x ≠ y ∧ ∀ j ∈ Finset.range n, (x + j) ∣ (y + j) := by sorry

-- Part (b)
theorem divisibility_for_all_implies_equality (x y : ℕ+) :
  (∀ j : ℕ+, (x + j) ∣ (y + j)) → x = y := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_exists_divisibility_for_all_implies_equality_l798_79849


namespace NUMINAMATH_CALUDE_max_area_inscribed_ngon_l798_79889

/-- An n-gon with given side lengths -/
structure Ngon (n : ℕ) where
  sides : Fin n → ℝ
  area : ℝ

/-- An n-gon inscribed in a circle -/
structure InscribedNgon (n : ℕ) extends Ngon n where
  isInscribed : Bool

/-- Theorem: The area of any n-gon is less than or equal to 
    the area of the inscribed n-gon with the same side lengths -/
theorem max_area_inscribed_ngon (n : ℕ) (l : Fin n → ℝ) :
  ∀ (P : Ngon n), P.sides = l →
  ∃ (Q : InscribedNgon n), Q.sides = l ∧ P.area ≤ Q.area :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_ngon_l798_79889


namespace NUMINAMATH_CALUDE_special_sequence_coprime_l798_79842

/-- A polynomial with integer coefficients that maps 0 and 1 to 1 -/
def SpecialPolynomial (p : ℤ → ℤ) : Prop :=
  (∀ x y : ℤ, p (x + y) = p x + p y - 1) ∧ p 0 = 1 ∧ p 1 = 1

/-- The sequence defined by the special polynomial -/
def SpecialSequence (p : ℤ → ℤ) (a : ℕ → ℤ) : Prop :=
  a 0 ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = p (a n)

/-- The theorem stating that any two terms in the sequence are coprime -/
theorem special_sequence_coprime (p : ℤ → ℤ) (a : ℕ → ℤ) 
  (hp : SpecialPolynomial p) (ha : SpecialSequence p a) :
  ∀ i j : ℕ, Nat.gcd (a i).natAbs (a j).natAbs = 1 :=
sorry

end NUMINAMATH_CALUDE_special_sequence_coprime_l798_79842


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l798_79871

-- Define the sets A and B
def A : Set ℝ := {x | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x | -Real.sqrt 2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l798_79871


namespace NUMINAMATH_CALUDE_stratified_sample_size_l798_79898

/-- Represents the ratio of quantities for three product models -/
structure ProductRatio :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)

/-- Calculates the total sample size given the number of items from the smallest group -/
def calculateSampleSize (ratio : ProductRatio) (smallestGroupSample : ℕ) : ℕ :=
  smallestGroupSample * (ratio.a + ratio.b + ratio.c) / ratio.a

/-- Theorem: For a stratified sample with ratio 3:4:7, if the smallest group has 9 items, the total sample size is 42 -/
theorem stratified_sample_size (ratio : ProductRatio) (h1 : ratio.a = 3) (h2 : ratio.b = 4) (h3 : ratio.c = 7) :
  calculateSampleSize ratio 9 = 42 := by
  sorry

#eval calculateSampleSize ⟨3, 4, 7⟩ 9

end NUMINAMATH_CALUDE_stratified_sample_size_l798_79898


namespace NUMINAMATH_CALUDE_extraneous_root_value_l798_79870

theorem extraneous_root_value (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) ∧
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_extraneous_root_value_l798_79870


namespace NUMINAMATH_CALUDE_tom_dance_lessons_l798_79880

theorem tom_dance_lessons 
  (cost_per_lesson : ℕ) 
  (free_lessons : ℕ) 
  (total_paid : ℕ) :
  cost_per_lesson = 10 →
  free_lessons = 2 →
  total_paid = 80 →
  (total_paid / cost_per_lesson) + free_lessons = 10 :=
by sorry

end NUMINAMATH_CALUDE_tom_dance_lessons_l798_79880


namespace NUMINAMATH_CALUDE_compute_fraction_power_and_multiply_l798_79853

theorem compute_fraction_power_and_multiply :
  8 * (1 / 7)^2 = 8 / 49 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_and_multiply_l798_79853


namespace NUMINAMATH_CALUDE_cube_edge_length_l798_79809

theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) : 
  surface_area = 54 → 
  surface_area = 6 * edge_length ^ 2 → 
  edge_length = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l798_79809


namespace NUMINAMATH_CALUDE_nabla_calculation_l798_79867

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l798_79867


namespace NUMINAMATH_CALUDE_fraction_equality_implication_l798_79864

theorem fraction_equality_implication (a b c d : ℝ) :
  (a + b) / (b + c) = (c + d) / (d + a) →
  (a = c ∨ a + b + c + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implication_l798_79864


namespace NUMINAMATH_CALUDE_alpha_value_l798_79868

/-- Given that α is an acute angle and sin(α - 10°) = √3/2, prove that α = 70°. -/
theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < 90) (h2 : Real.sin (α - 10) = Real.sqrt 3 / 2) : 
  α = 70 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l798_79868


namespace NUMINAMATH_CALUDE_reflection_curve_coefficient_product_l798_79831

/-- The reflection of the curve xy = 1 over the line y = 2x -/
def ReflectedCurve (x y : ℝ) : Prop :=
  ∃ (b c d : ℝ), 12 * x^2 + b * x * y + c * y^2 + d = 0

/-- The product of coefficients b and c in the reflected curve equation -/
def CoefficientProduct (b c : ℝ) : ℝ := b * c

theorem reflection_curve_coefficient_product :
  ∃ (b c : ℝ), ReflectedCurve x y ∧ CoefficientProduct b c = 84 := by
  sorry

end NUMINAMATH_CALUDE_reflection_curve_coefficient_product_l798_79831


namespace NUMINAMATH_CALUDE_abs_neg_one_fifth_l798_79805

theorem abs_neg_one_fifth : |(-1 : ℚ) / 5| = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_fifth_l798_79805


namespace NUMINAMATH_CALUDE_toby_monday_steps_l798_79865

/-- Represents the number of steps walked on each day of the week -/
structure WeekSteps where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total steps walked in a week -/
def totalSteps (w : WeekSteps) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

/-- Calculates the average steps per day in a week -/
def averageSteps (w : WeekSteps) : ℚ :=
  (totalSteps w : ℚ) / 7

theorem toby_monday_steps (w : WeekSteps) 
  (h1 : averageSteps w = 9000)
  (h2 : w.sunday = 9400)
  (h3 : w.tuesday = 8300)
  (h4 : w.wednesday = 9200)
  (h5 : w.thursday = 8900)
  (h6 : (w.friday + w.saturday : ℚ) / 2 = 9050) :
  w.monday = 9100 := by
  sorry


end NUMINAMATH_CALUDE_toby_monday_steps_l798_79865


namespace NUMINAMATH_CALUDE_certain_number_proof_l798_79804

theorem certain_number_proof (m : ℕ) : 9999 * m = 724827405 → m = 72483 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l798_79804


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l798_79824

theorem complex_magnitude_problem (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (1 + 2*i) / i → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l798_79824


namespace NUMINAMATH_CALUDE_no_equal_division_of_scalene_triangle_l798_79890

/-- A triangle represented by its three vertices in ℝ² -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- A triangle is scalene if all its sides have different lengths -/
def isScalene (t : Triangle) : Prop := sorry

/-- A point D that divides the triangle into two equal parts -/
def dividingPoint (t : Triangle) (D : ℝ × ℝ) : Prop :=
  triangleArea ⟨t.A, t.B, D⟩ = triangleArea ⟨t.A, t.C, D⟩

/-- Theorem: A scalene triangle cannot be divided into two equal triangles -/
theorem no_equal_division_of_scalene_triangle (t : Triangle) :
  isScalene t → ¬∃ D : ℝ × ℝ, dividingPoint t D := by
  sorry

end NUMINAMATH_CALUDE_no_equal_division_of_scalene_triangle_l798_79890


namespace NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_4_mod_9_l798_79883

theorem unique_prime_between_30_and_40_with_remainder_4_mod_9 :
  ∃! n : ℕ, 30 < n ∧ n < 40 ∧ Prime n ∧ n % 9 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_4_mod_9_l798_79883


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2_l798_79875

theorem unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 18 * k) ∧ 
    (28 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 28.2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_18_with_sqrt_between_28_and_28_2_l798_79875


namespace NUMINAMATH_CALUDE_tan_half_less_than_x_l798_79801

theorem tan_half_less_than_x (x : ℝ) (h1 : 0 < x) (h2 : x ≤ π / 2) : Real.tan (x / 2) < x := by
  sorry

end NUMINAMATH_CALUDE_tan_half_less_than_x_l798_79801


namespace NUMINAMATH_CALUDE_sum_of_four_squares_sum_of_four_squares_proof_l798_79828

theorem sum_of_four_squares : ℕ → ℕ → ℕ → Prop :=
  fun triangle circle square =>
    triangle + circle + triangle + square = 27 ∧
    circle + triangle + circle + square = 25 ∧
    square + square + square + triangle = 39 →
    4 * square = 44

-- The proof would go here, but we're skipping it as per instructions
theorem sum_of_four_squares_proof (triangle circle square : ℕ) 
  (h : sum_of_four_squares triangle circle square) : 4 * square = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_sum_of_four_squares_proof_l798_79828


namespace NUMINAMATH_CALUDE_length_AB_squared_l798_79863

/-- The parabola function y = 3x^2 + 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 2

/-- The square of the distance between two points -/
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x2 - x1)^2 + (y2 - y1)^2

theorem length_AB_squared :
  ∀ (x1 y1 x2 y2 : ℝ),
  f x1 = y1 →  -- Point A is on the parabola
  f x2 = y2 →  -- Point B is on the parabola
  (x1 + x2) / 2 = 1 →  -- x-coordinate of midpoint C
  (y1 + y2) / 2 = 1 →  -- y-coordinate of midpoint C
  distance_squared x1 y1 x2 y2 = 17 := by
    sorry

end NUMINAMATH_CALUDE_length_AB_squared_l798_79863


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l798_79873

/-- Calculates the number of females in a population given stratified sampling data -/
theorem stratified_sampling_female_count 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (females_in_sample : ℕ) 
  (total_population_pos : 0 < total_population)
  (sample_size_pos : 0 < sample_size)
  (sample_size_le_total : sample_size ≤ total_population)
  (females_in_sample_le_sample : females_in_sample ≤ sample_size) :
  let females_in_population : ℕ := (females_in_sample * total_population) / sample_size
  females_in_population = 760 ∧ females_in_population ≤ total_population :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l798_79873


namespace NUMINAMATH_CALUDE_proposition_p_and_q_true_l798_79879

theorem proposition_p_and_q_true : 
  (∃ α : ℝ, Real.cos (π - α) = Real.cos α) ∧ (∀ x : ℝ, x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_and_q_true_l798_79879


namespace NUMINAMATH_CALUDE_solve_for_q_l798_79840

theorem solve_for_q (n d p q : ℝ) (h1 : d ≠ 0) (h2 : p ≠ 0) (h3 : q ≠ 0) 
  (h4 : n = (2 * d * p * q) / (p - q)) : 
  q = (n * p) / (2 * d * p + n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l798_79840


namespace NUMINAMATH_CALUDE_exists_x0_implies_a_value_l798_79850

noncomputable def f (a x : ℝ) : ℝ := Real.exp (x + a) + x

noncomputable def g (a x : ℝ) : ℝ := Real.log (x + 3) - 4 * Real.exp (-x - a)

theorem exists_x0_implies_a_value (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 2) → a = 2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_x0_implies_a_value_l798_79850


namespace NUMINAMATH_CALUDE_fraction_equality_l798_79846

theorem fraction_equality (a b : ℝ) (h : (1 / a) + (1 / (2 * b)) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l798_79846


namespace NUMINAMATH_CALUDE_revenue_ratio_theorem_l798_79835

/-- Represents the revenue data for a product line -/
structure ProductLine where
  lastYearRevenue : ℝ
  projectedIncrease : ℝ
  actualDecrease : ℝ

/-- Calculates the projected revenue for a product line -/
def projectedRevenue (p : ProductLine) : ℝ :=
  p.lastYearRevenue * (1 + p.projectedIncrease)

/-- Calculates the actual revenue for a product line -/
def actualRevenue (p : ProductLine) : ℝ :=
  p.lastYearRevenue * (1 - p.actualDecrease)

/-- Theorem stating that the ratio of total actual revenue to total projected revenue
    is approximately 0.5276 for the given product lines -/
theorem revenue_ratio_theorem (standardGum sugarFreeGum bubbleGum : ProductLine)
    (h1 : standardGum.lastYearRevenue = 100000)
    (h2 : standardGum.projectedIncrease = 0.3)
    (h3 : standardGum.actualDecrease = 0.2)
    (h4 : sugarFreeGum.lastYearRevenue = 150000)
    (h5 : sugarFreeGum.projectedIncrease = 0.5)
    (h6 : sugarFreeGum.actualDecrease = 0.3)
    (h7 : bubbleGum.lastYearRevenue = 200000)
    (h8 : bubbleGum.projectedIncrease = 0.4)
    (h9 : bubbleGum.actualDecrease = 0.25) :
    let totalActualRevenue := actualRevenue standardGum + actualRevenue sugarFreeGum + actualRevenue bubbleGum
    let totalProjectedRevenue := projectedRevenue standardGum + projectedRevenue sugarFreeGum + projectedRevenue bubbleGum
    abs (totalActualRevenue / totalProjectedRevenue - 0.5276) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_revenue_ratio_theorem_l798_79835


namespace NUMINAMATH_CALUDE_cosine_cube_sum_l798_79858

theorem cosine_cube_sum (α : ℝ) :
  (Real.cos α)^3 + (Real.cos (α + 2 * Real.pi / 3))^3 + (Real.cos (α - 2 * Real.pi / 3))^3 = 
  3/4 * Real.cos (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_cube_sum_l798_79858


namespace NUMINAMATH_CALUDE_alpha_plus_beta_eq_128_l798_79872

theorem alpha_plus_beta_eq_128 :
  ∀ α β : ℝ, (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96*x + 2209) / (x^2 + 66*x - 3969)) →
  α + β = 128 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_eq_128_l798_79872


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l798_79843

def f (x : ℝ) := (x - 3)^2 - 4

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ↔ c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l798_79843


namespace NUMINAMATH_CALUDE_most_cars_are_blue_l798_79848

theorem most_cars_are_blue (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
  sorry

end NUMINAMATH_CALUDE_most_cars_are_blue_l798_79848


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_bound_l798_79885

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the theorem
theorem ellipse_perpendicular_bisector_bound 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (A B : ℝ × ℝ) 
  (h_A : is_on_ellipse A.1 A.2 a b) 
  (h_B : is_on_ellipse B.1 B.2 a b) 
  (x₀ : ℝ) 
  (h_perp_bisector : ∃ (k : ℝ), 
    k * (A.1 - B.1) = A.2 - B.2 ∧ 
    x₀ = (A.1 + B.1) / 2 + k * (A.2 + B.2) / 2) :
  -((a^2 - b^2) / a) < x₀ ∧ x₀ < (a^2 - b^2) / a :=
sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_bound_l798_79885
