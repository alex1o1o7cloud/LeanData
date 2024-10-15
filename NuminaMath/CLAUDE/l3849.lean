import Mathlib

namespace NUMINAMATH_CALUDE_orchid_count_l3849_384948

/-- The number of orchid bushes initially in the park -/
def initial_orchids : ℕ := 22

/-- The number of orchid bushes to be planted -/
def planted_orchids : ℕ := 13

/-- The final number of orchid bushes after planting -/
def final_orchids : ℕ := 35

/-- Theorem stating that the initial number of orchid bushes plus the planted ones equals the final number -/
theorem orchid_count : initial_orchids + planted_orchids = final_orchids := by
  sorry

end NUMINAMATH_CALUDE_orchid_count_l3849_384948


namespace NUMINAMATH_CALUDE_three_integers_sum_and_reciprocals_l3849_384995

theorem three_integers_sum_and_reciprocals (a b c : ℕ+) : 
  (a + b + c : ℕ) = 15 ∧ 
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 71 / 105) → 
  ({a, b, c} : Finset ℕ+) = {3, 5, 7} := by
sorry

end NUMINAMATH_CALUDE_three_integers_sum_and_reciprocals_l3849_384995


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_l3849_384991

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ 
    digit_product n = 9 * 8 * 7 * 6 * 5 → 
    n ≤ 98765 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_l3849_384991


namespace NUMINAMATH_CALUDE_equation_solution_l3849_384955

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 63 / 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3849_384955


namespace NUMINAMATH_CALUDE_linear_function_property_l3849_384932

/-- A function f satisfying f(x₁ + x₂) = f(x₁) + f(x₂) for all real x₁ and x₂ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), f (x₁ + x₂) = f x₁ + f x₂

/-- Theorem: A function of the form f(x) = kx, where k is a non-zero constant,
    satisfies the property f(x₁ + x₂) = f(x₁) + f(x₂) for all real x₁ and x₂ -/
theorem linear_function_property (k : ℝ) (hk : k ≠ 0) :
  LinearFunction (fun x ↦ k * x) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l3849_384932


namespace NUMINAMATH_CALUDE_cans_for_reduced_people_l3849_384916

/-- Given 600 cans feed 40 people, proves the number of cans needed for 30% fewer people is 420 -/
theorem cans_for_reduced_people (total_cans : ℕ) (original_people : ℕ) (reduction_percent : ℚ) : 
  total_cans = 600 → 
  original_people = 40 → 
  reduction_percent = 30 / 100 →
  (total_cans / original_people : ℚ) * (original_people * (1 - reduction_percent) : ℚ) = 420 := by
  sorry

end NUMINAMATH_CALUDE_cans_for_reduced_people_l3849_384916


namespace NUMINAMATH_CALUDE_lunch_breakfast_difference_l3849_384921

def muffin_cost : ℚ := 2
def coffee_cost : ℚ := 4
def soup_cost : ℚ := 3
def salad_cost : ℚ := 5.25
def lemonade_cost : ℚ := 0.75

def breakfast_cost : ℚ := muffin_cost + coffee_cost
def lunch_cost : ℚ := soup_cost + salad_cost + lemonade_cost

theorem lunch_breakfast_difference : lunch_cost - breakfast_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_lunch_breakfast_difference_l3849_384921


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3849_384934

theorem polynomial_factorization (x : ℤ) :
  3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2 =
  (3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3849_384934


namespace NUMINAMATH_CALUDE_continuous_at_5_l3849_384907

def f (x : ℝ) : ℝ := 3 * x^2 - 2

theorem continuous_at_5 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuous_at_5_l3849_384907


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l3849_384972

theorem park_trees_after_planting (current_trees new_trees : ℕ) 
  (h1 : current_trees = 25)
  (h2 : new_trees = 73) :
  current_trees + new_trees = 98 :=
by sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l3849_384972


namespace NUMINAMATH_CALUDE_problem_solution_l3849_384939

theorem problem_solution : (((Real.sqrt 25 - 1) / 2) ^ 2 + 3) ⁻¹ * 10 = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3849_384939


namespace NUMINAMATH_CALUDE_equality_of_M_and_N_l3849_384971

theorem equality_of_M_and_N (a b : ℝ) (hab : a * b = 1) : 
  (1 / (1 + a) + 1 / (1 + b)) = (a / (1 + a) + b / (1 + b)) := by
  sorry

#check equality_of_M_and_N

end NUMINAMATH_CALUDE_equality_of_M_and_N_l3849_384971


namespace NUMINAMATH_CALUDE_black_water_bottles_l3849_384968

theorem black_water_bottles (red : ℕ) (blue : ℕ) (taken_out : ℕ) (left : ℕ) :
  red = 2 →
  blue = 4 →
  taken_out = 5 →
  left = 4 →
  ∃ black : ℕ, red + black + blue = taken_out + left ∧ black = 3 :=
by sorry

end NUMINAMATH_CALUDE_black_water_bottles_l3849_384968


namespace NUMINAMATH_CALUDE_residue_mod_37_l3849_384965

theorem residue_mod_37 : ∃ k : ℤ, -927 = 37 * k + 35 ∧ (35 : ℤ) ∈ Set.range (fun i => i : Fin 37 → ℤ) := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_37_l3849_384965


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l3849_384998

/-- Given three points in a 2D plane, checks if they are collinear --/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem: If A(-1, -2), B(4, 8), and C(5, x) are collinear, then x = 10 --/
theorem collinear_points_x_value :
  collinear (-1) (-2) 4 8 5 x → x = 10 :=
by
  sorry

#check collinear_points_x_value

end NUMINAMATH_CALUDE_collinear_points_x_value_l3849_384998


namespace NUMINAMATH_CALUDE_no_prime_factor_3j_plus_2_l3849_384957

/-- A number is a cube if it's the cube of some integer -/
def IsCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

/-- The number of divisors of a natural number -/
def NumDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A number is the smallest with k divisors if it has k divisors and no smaller number has k divisors -/
def IsSmallestWithKDivisors (n k : ℕ) : Prop :=
  NumDivisors n = k ∧ ∀ m < n, NumDivisors m ≠ k

theorem no_prime_factor_3j_plus_2 (n k : ℕ) (h1 : IsSmallestWithKDivisors n k) (h2 : IsCube n) :
  ¬∃ (p : ℕ), Nat.Prime p ∧ (∃ j : ℕ, p = 3*j + 2) ∧ p ∣ k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_3j_plus_2_l3849_384957


namespace NUMINAMATH_CALUDE_railing_distance_proof_l3849_384923

/-- The distance between two railings with bicycles placed between them -/
def railing_distance (interval_distance : ℕ) (num_bicycles : ℕ) : ℕ :=
  interval_distance * (num_bicycles - 1)

/-- Theorem: The distance between two railings is 95 meters -/
theorem railing_distance_proof :
  railing_distance 5 19 = 95 := by
  sorry

end NUMINAMATH_CALUDE_railing_distance_proof_l3849_384923


namespace NUMINAMATH_CALUDE_return_trip_time_l3849_384904

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  against_wind_time : ℝ  -- time of flight against wind
  still_air_time : ℝ  -- time of flight in still air

/-- The flight conditions as given in the problem -/
def flight_conditions (s : FlightScenario) : Prop :=
  s.against_wind_time = 90 ∧
  s.d = s.against_wind_time * (s.p - s.w) ∧
  s.d / (s.p + s.w) = s.still_air_time - 15

/-- The theorem stating that the return trip takes either 30 or 45 minutes -/
theorem return_trip_time (s : FlightScenario) :
  flight_conditions s →
  (s.d / (s.p + s.w) = 30 ∨ s.d / (s.p + s.w) = 45) :=
by sorry

end NUMINAMATH_CALUDE_return_trip_time_l3849_384904


namespace NUMINAMATH_CALUDE_triangle_properties_l3849_384975

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a * Real.cos t.B + t.b * Real.cos t.A) / t.c = 2 * Real.cos t.C)
  (h2 : (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3)
  (h3 : t.a + t.b = 6) :
  t.C = π/3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3849_384975


namespace NUMINAMATH_CALUDE_function_equality_condition_l3849_384964

theorem function_equality_condition (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x
  ({x : ℝ | f x = 0} = {x : ℝ | f (f x) = 0} ∧ {x : ℝ | f x = 0}.Nonempty) ↔ 0 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_condition_l3849_384964


namespace NUMINAMATH_CALUDE_three_heads_in_a_row_probability_l3849_384919

def coin_flips : ℕ := 6

def favorable_outcomes : ℕ := 12

def total_outcomes : ℕ := 2^coin_flips

def probability : ℚ := favorable_outcomes / total_outcomes

theorem three_heads_in_a_row_probability :
  probability = 3/16 := by sorry

end NUMINAMATH_CALUDE_three_heads_in_a_row_probability_l3849_384919


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_prove_even_odd_sum_difference_l3849_384992

theorem even_odd_sum_difference : ℕ → Prop :=
  fun n =>
    let even_sum := (n + 1) * (2 + 2 * n)
    let odd_sum := n * (1 + 2 * n - 1)
    even_sum - odd_sum = 6017

theorem prove_even_odd_sum_difference :
  even_odd_sum_difference 2003 := by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_prove_even_odd_sum_difference_l3849_384992


namespace NUMINAMATH_CALUDE_complex_square_simplification_l3849_384918

theorem complex_square_simplification :
  (5 - 3 * Real.sqrt 2 * Complex.I) ^ 2 = 43 - 30 * Real.sqrt 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l3849_384918


namespace NUMINAMATH_CALUDE_solution_value_l3849_384915

theorem solution_value (a b : ℝ) (h : a - 2*b = 7) : -a + 2*b + 1 = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3849_384915


namespace NUMINAMATH_CALUDE_largest_value_is_E_l3849_384978

theorem largest_value_is_E :
  let a := 24680 + 2 / 1357
  let b := 24680 - 2 / 1357
  let c := 24680 * 2 / 1357
  let d := 24680 / (2 / 1357)
  let e := 24680 ^ 1.357
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_is_E_l3849_384978


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3849_384969

/-- A quadratic function f(x) = x^2 + 2x + a has no real roots if and only if a > 1 -/
theorem quadratic_no_real_roots (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3849_384969


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3849_384982

theorem inequality_equivalence (x y : ℝ) : 
  y - x < Real.sqrt (4 * x^2) ↔ (x ≥ 0 ∧ y < 3 * x) ∨ (x < 0 ∧ y < -x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3849_384982


namespace NUMINAMATH_CALUDE_completing_square_l3849_384963

theorem completing_square (x : ℝ) : x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l3849_384963


namespace NUMINAMATH_CALUDE_circle_circumference_l3849_384929

/-- The circumference of a circle with radius 36 is 72π -/
theorem circle_circumference (π : ℝ) (h : π > 0) : ∃ (k : ℝ), k * π = 2 * π * 36 ∧ k = 72 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l3849_384929


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_l3849_384935

theorem infinitely_many_pairs (c : ℝ) : 
  (c > 0) → 
  (∀ k : ℕ, ∃ n m : ℕ, 
    n > 0 ∧ m > 0 ∧
    (n : ℝ) ≥ (m : ℝ) + c * Real.sqrt ((m : ℝ) - 1) + 1 ∧
    ∀ i ∈ Finset.range (2 * n - m - n + 1), ¬ ∃ j : ℕ, (n + i : ℝ) = (j : ℝ) ^ 2) ↔ 
  c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_l3849_384935


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3849_384942

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 8 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -8 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3849_384942


namespace NUMINAMATH_CALUDE_jacks_burgers_l3849_384984

/-- Given Jack's barbecue sauce recipe and usage, prove how many burgers he can make. -/
theorem jacks_burgers :
  -- Total sauce
  let total_sauce : ℚ := 3 + 1 + 1

  -- Sauce per burger
  let sauce_per_burger : ℚ := 1 / 4

  -- Sauce per pulled pork sandwich
  let sauce_per_pps : ℚ := 1 / 6

  -- Number of pulled pork sandwiches
  let num_pps : ℕ := 18

  -- Sauce used for pulled pork sandwiches
  let sauce_for_pps : ℚ := sauce_per_pps * num_pps

  -- Remaining sauce for burgers
  let remaining_sauce : ℚ := total_sauce - sauce_for_pps

  -- Number of burgers Jack can make
  ↑(remaining_sauce / sauce_per_burger).floor = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_jacks_burgers_l3849_384984


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l3849_384902

theorem increasing_function_inequality (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : ∀ x₁ x₂, f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) :
  ∀ x₁ x₂, x₁ + x₂ ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l3849_384902


namespace NUMINAMATH_CALUDE_sister_brother_product_is_twelve_l3849_384999

/-- Represents a family with siblings -/
structure Family :=
  (num_sisters : ℕ)
  (num_brothers : ℕ)

/-- Calculates the product of sisters and brothers for a sister in the family -/
def sister_brother_product (f : Family) : ℕ :=
  (f.num_sisters - 1) * f.num_brothers

/-- Theorem stating that for a family where one sibling has 4 sisters and 4 brothers,
    the product of sisters and brothers for any sister is 12 -/
theorem sister_brother_product_is_twelve (f : Family) 
  (h : f.num_sisters = 4 ∧ f.num_brothers = 4) : 
  sister_brother_product f = 12 := by
  sorry

#eval sister_brother_product ⟨4, 4⟩

end NUMINAMATH_CALUDE_sister_brother_product_is_twelve_l3849_384999


namespace NUMINAMATH_CALUDE_problem_curve_is_line_segment_l3849_384922

/-- A parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ
  t_min : ℝ
  t_max : ℝ

/-- Definition of a line segment -/
def IsLineSegment (curve : ParametricCurve) : Prop :=
  ∃ (a b : ℝ × ℝ),
    (∀ t, curve.t_min ≤ t ∧ t ≤ curve.t_max →
      (curve.x t, curve.y t) = ((1 - t) • a.1 + t • b.1, (1 - t) • a.2 + t • b.2))

/-- The specific parametric curve from the problem -/
def ProblemCurve : ParametricCurve where
  x := λ t => 2 * t
  y := λ _ => 2
  t_min := -1
  t_max := 1

/-- Theorem stating that the problem curve is a line segment -/
theorem problem_curve_is_line_segment : IsLineSegment ProblemCurve := by
  sorry


end NUMINAMATH_CALUDE_problem_curve_is_line_segment_l3849_384922


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l3849_384933

-- Define a tetrahedron type
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)

-- Define the volume of a tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Define the condition that at least 5 edges are not greater than 2
def at_least_five_short_edges (t : Tetrahedron) : Prop :=
  ∃ (long_edge : Fin 6), ∀ (e : Fin 6), e ≠ long_edge → t.edges e ≤ 2

-- Theorem statement
theorem tetrahedron_volume_bound (t : Tetrahedron) :
  at_least_five_short_edges t → volume t ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l3849_384933


namespace NUMINAMATH_CALUDE_movies_needed_for_even_distribution_movie_store_problem_l3849_384906

theorem movies_needed_for_even_distribution (total_movies : Nat) (num_shelves : Nat) : Nat :=
  let movies_per_shelf := total_movies / num_shelves
  let movies_needed := (movies_per_shelf + 1) * num_shelves - total_movies
  movies_needed

theorem movie_store_problem : movies_needed_for_even_distribution 2763 17 = 155 := by
  sorry

end NUMINAMATH_CALUDE_movies_needed_for_even_distribution_movie_store_problem_l3849_384906


namespace NUMINAMATH_CALUDE_safeties_count_l3849_384901

/-- Represents the scoring of a football team -/
structure FootballScore where
  fieldGoals : ℕ      -- number of four-point field goals
  threePointGoals : ℕ -- number of three-point goals
  safeties : ℕ        -- number of two-point safeties

/-- Calculates the total score for a given FootballScore -/
def totalScore (score : FootballScore) : ℕ :=
  4 * score.fieldGoals + 3 * score.threePointGoals + 2 * score.safeties

/-- Theorem: Given the conditions, the number of safeties is 6 -/
theorem safeties_count (score : FootballScore) :
  (4 * score.fieldGoals = 2 * 3 * score.threePointGoals) →
  (score.safeties = score.threePointGoals + 2) →
  (totalScore score = 50) →
  score.safeties = 6 :=
by sorry

end NUMINAMATH_CALUDE_safeties_count_l3849_384901


namespace NUMINAMATH_CALUDE_star_computation_l3849_384976

-- Define the * operation
def star (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

-- Theorem statement
theorem star_computation : star 1 (star 2 (star 3 4)) = -18 := by
  sorry

end NUMINAMATH_CALUDE_star_computation_l3849_384976


namespace NUMINAMATH_CALUDE_negative_fraction_identification_l3849_384970

-- Define a predicate for negative fractions
def is_negative_fraction (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ) ∧ x < 0

-- Theorem statement
theorem negative_fraction_identification :
  is_negative_fraction (-0.7) ∧
  ¬is_negative_fraction (1/2) ∧
  ¬is_negative_fraction (-π) ∧
  ¬is_negative_fraction (-3/3) :=
by sorry

end NUMINAMATH_CALUDE_negative_fraction_identification_l3849_384970


namespace NUMINAMATH_CALUDE_m_range_l3849_384973

def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : A ⊆ B m ∧ A ≠ B m → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3849_384973


namespace NUMINAMATH_CALUDE_base_9_minus_b_multiple_of_7_l3849_384996

def base_9_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

def is_multiple_of (a b : Int) : Prop :=
  ∃ k : Int, a = b * k

theorem base_9_minus_b_multiple_of_7 (b : Int) :
  (0 ≤ b) →
  (b ≤ 9) →
  (is_multiple_of (base_9_to_decimal [2, 7, 6, 4, 5, 1, 3] - b) 7) →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_9_minus_b_multiple_of_7_l3849_384996


namespace NUMINAMATH_CALUDE_divisor_problem_l3849_384943

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n^2) % d = 4) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3849_384943


namespace NUMINAMATH_CALUDE_peter_reading_time_l3849_384908

/-- Given that Peter reads three times as fast as Kristin and Kristin reads half of her 20 books in 540 hours, prove that Peter takes 18 hours to read one book. -/
theorem peter_reading_time (peter_speed : ℝ) (kristin_speed : ℝ) (kristin_half_books : ℕ) (kristin_half_time : ℝ) : 
  peter_speed = 3 * kristin_speed →
  kristin_half_books = 10 →
  kristin_half_time = 540 →
  peter_speed = 1 / 18 := by
sorry

end NUMINAMATH_CALUDE_peter_reading_time_l3849_384908


namespace NUMINAMATH_CALUDE_difference_has_7_in_thousands_l3849_384946

/-- Given a number with 3 in the ten-thousands place (28943712) and its local value (30000) -/
def local_value_of_3 : ℕ := 30000

/-- The difference between an unknown number and the local value of 3 -/
def difference (x : ℕ) : ℕ := x - local_value_of_3

/-- Check if a number has 7 in the thousands place -/
def has_7_in_thousands (n : ℕ) : Prop :=
  (n / 1000) % 10 = 7

/-- The local value of 7 in the thousands place -/
def local_value_of_7_in_thousands : ℕ := 7000

/-- Theorem: If the difference has 7 in the thousands place, 
    then the local value of 7 in the difference is 7000 -/
theorem difference_has_7_in_thousands (x : ℕ) :
  has_7_in_thousands (difference x) →
  (difference x / 1000) % 10 * 1000 = local_value_of_7_in_thousands :=
by
  sorry

end NUMINAMATH_CALUDE_difference_has_7_in_thousands_l3849_384946


namespace NUMINAMATH_CALUDE_not_prime_5n_plus_3_l3849_384950

theorem not_prime_5n_plus_3 (n : ℕ) (h1 : ∃ a : ℕ, 2 * n + 1 = a ^ 2) (h2 : ∃ b : ℕ, 3 * n + 1 = b ^ 2) : 
  ¬ Nat.Prime (5 * n + 3) := by
sorry

end NUMINAMATH_CALUDE_not_prime_5n_plus_3_l3849_384950


namespace NUMINAMATH_CALUDE_quadratic_form_bounds_l3849_384993

theorem quadratic_form_bounds (x y : ℝ) (h : x^2 + x*y + y^2 = 3) :
  1 ≤ x^2 - x*y + y^2 ∧ x^2 - x*y + y^2 ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_bounds_l3849_384993


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l3849_384980

/-- An arithmetic sequence with non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l3849_384980


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3849_384945

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3849_384945


namespace NUMINAMATH_CALUDE_binary_67_l3849_384959

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- Theorem: The binary representation of 67 is [1,0,0,0,0,1,1] -/
theorem binary_67 : toBinary 67 = [1,0,0,0,0,1,1] := by
  sorry

end NUMINAMATH_CALUDE_binary_67_l3849_384959


namespace NUMINAMATH_CALUDE_calculation_proof_l3849_384953

theorem calculation_proof : (3.64 - 2.1) * 1.5 = 2.31 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3849_384953


namespace NUMINAMATH_CALUDE_quadratic_factoring_l3849_384966

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α
  a_nonzero : a ≠ 0

/-- A factored form of a quadratic equation is a product of linear factors -/
structure FactoredForm (α : Type*) [Field α] where
  factor1 : α → α
  factor2 : α → α

/-- 
Given a quadratic equation that can be factored, 
it can be expressed as a multiplication of factors.
-/
theorem quadratic_factoring 
  {α : Type*} [Field α]
  (eq : QuadraticEquation α)
  (h_factorable : ∃ (f : FactoredForm α), 
    ∀ x, eq.a * x^2 + eq.b * x + eq.c = f.factor1 x * f.factor2 x) :
  ∃ (f : FactoredForm α), 
    ∀ x, eq.a * x^2 + eq.b * x + eq.c = f.factor1 x * f.factor2 x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l3849_384966


namespace NUMINAMATH_CALUDE_bacteria_growth_l3849_384920

theorem bacteria_growth (n : ℕ) : (∀ k < n, 4 * 3^k ≤ 500) ∧ 4 * 3^n > 500 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l3849_384920


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3849_384900

/-- A geometric sequence with common ratio q < 0 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q < 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 2 = 1 - a 1) →
  (a 4 = 4 - a 3) →
  a 5 + a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3849_384900


namespace NUMINAMATH_CALUDE_reciprocal_power_2006_l3849_384960

theorem reciprocal_power_2006 (a : ℚ) : 
  (a ≠ 0 ∧ a = 1 / a) → a^2006 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_2006_l3849_384960


namespace NUMINAMATH_CALUDE_shoe_multiple_l3849_384986

theorem shoe_multiple (bonny_shoes becky_shoes bobby_shoes : ℕ) : 
  bonny_shoes = 13 →
  bonny_shoes = 2 * becky_shoes - 5 →
  bobby_shoes = 27 →
  ∃ m : ℕ, m * becky_shoes = bobby_shoes ∧ m = 3 := by
sorry

end NUMINAMATH_CALUDE_shoe_multiple_l3849_384986


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_5_mod_15_l3849_384988

theorem least_five_digit_congruent_to_5_mod_15 : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 15 = 5 ∧
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 15 = 5 → n ≤ m) ∧
  n = 10010 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_5_mod_15_l3849_384988


namespace NUMINAMATH_CALUDE_steve_commute_speed_l3849_384911

theorem steve_commute_speed (distance : ℝ) (total_time : ℝ) : 
  distance > 0 → 
  total_time > 0 → 
  ∃ (outbound_speed : ℝ), 
    outbound_speed > 0 ∧ 
    (distance / outbound_speed + distance / (2 * outbound_speed) = total_time) → 
    2 * outbound_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_steve_commute_speed_l3849_384911


namespace NUMINAMATH_CALUDE_shelter_dogs_l3849_384981

theorem shelter_dogs (x : ℕ) (dogs cats : ℕ) 
  (h1 : dogs * 7 = x * cats) 
  (h2 : dogs * 11 = 15 * (cats + 8)) : 
  dogs = 77 := by
  sorry

end NUMINAMATH_CALUDE_shelter_dogs_l3849_384981


namespace NUMINAMATH_CALUDE_final_bus_count_l3849_384914

def bus_problem (initial : ℕ) (first_stop : ℕ) (second_stop : ℕ) (third_stop : ℕ) : ℕ :=
  initial + first_stop - second_stop + third_stop

theorem final_bus_count :
  bus_problem 128 67 34 54 = 215 := by
  sorry

end NUMINAMATH_CALUDE_final_bus_count_l3849_384914


namespace NUMINAMATH_CALUDE_stating_shop_owner_cheat_percentage_l3849_384956

/-- Represents the percentage by which the shop owner cheats -/
def cheat_percentage : ℝ := 22.22222222222222

/-- Represents the profit percentage of the shop owner -/
def profit_percentage : ℝ := 22.22222222222222

/-- 
Theorem stating that if a shop owner cheats by the same percentage while buying and selling,
and their profit percentage is 22.22222222222222%, then the cheat percentage is also 22.22222222222222%.
-/
theorem shop_owner_cheat_percentage :
  cheat_percentage = profit_percentage :=
sorry

end NUMINAMATH_CALUDE_stating_shop_owner_cheat_percentage_l3849_384956


namespace NUMINAMATH_CALUDE_trajectory_and_constant_product_l3849_384926

-- Define the points and circles
def G : ℝ × ℝ := (5, 4)
def A : ℝ × ℝ := (1, 0)

def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 25

-- Define the lines
def l1 (k x y : ℝ) : Prop := k * x - y - k = 0
def l2 (x y : ℝ) : Prop := x + 2 * y + 2 = 0

-- Define the trajectory C2
def C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the theorem
theorem trajectory_and_constant_product :
  ∃ (M N : ℝ × ℝ) (k : ℝ),
    (∀ x y, C2 x y ↔ (∃ E F : ℝ × ℝ, C1 E.1 E.2 ∧ C1 F.1 F.2 ∧ 
      (x, y) = ((E.1 + F.1) / 2, (E.2 + F.2) / 2))) ∧
    l1 k M.1 M.2 ∧ 
    l1 k N.1 N.2 ∧ 
    l2 N.1 N.2 ∧
    C2 M.1 M.2 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 * ((N.1 - A.1)^2 + (N.2 - A.2)^2) = 36 :=
by sorry


end NUMINAMATH_CALUDE_trajectory_and_constant_product_l3849_384926


namespace NUMINAMATH_CALUDE_eugene_purchase_cost_l3849_384936

def tshirt_price : ℚ := 20
def pants_price : ℚ := 80
def shoes_price : ℚ := 150
def hat_price : ℚ := 25
def jacket_price : ℚ := 120

def tshirt_discount : ℚ := 0.1
def pants_discount : ℚ := 0.1
def shoes_discount : ℚ := 0.15
def hat_discount : ℚ := 0.05
def jacket_discount : ℚ := 0.2

def sales_tax : ℚ := 0.06

def tshirt_quantity : ℕ := 4
def pants_quantity : ℕ := 3
def shoes_quantity : ℕ := 2
def hat_quantity : ℕ := 3
def jacket_quantity : ℕ := 1

theorem eugene_purchase_cost :
  let discounted_tshirt := tshirt_price * (1 - tshirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let discounted_shoes := shoes_price * (1 - shoes_discount)
  let discounted_hat := hat_price * (1 - hat_discount)
  let discounted_jacket := jacket_price * (1 - jacket_discount)
  
  let total_before_tax := 
    discounted_tshirt * tshirt_quantity +
    discounted_pants * pants_quantity +
    discounted_shoes * shoes_quantity +
    discounted_hat * hat_quantity +
    discounted_jacket * jacket_quantity
  
  let total_with_tax := total_before_tax * (1 + sales_tax)
  
  total_with_tax = 752.87 := by sorry

end NUMINAMATH_CALUDE_eugene_purchase_cost_l3849_384936


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3849_384958

theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cylinder_volume := π * r^2 * h
  let cone_volume := (1/3) * π * r^2 * h
  cylinder_volume = 108 * π → cone_volume = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l3849_384958


namespace NUMINAMATH_CALUDE_wedge_volume_l3849_384983

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d h r : ℝ) (θ : ℝ) : 
  d = 18 →                           -- diameter of the log
  h = d →                            -- height of the cylindrical section
  r = d / 2 →                        -- radius of the log
  θ = 60 →                           -- angle between cuts in degrees
  (π * r^2 * h) / 2 = 729 * π := by
  sorry

#check wedge_volume

end NUMINAMATH_CALUDE_wedge_volume_l3849_384983


namespace NUMINAMATH_CALUDE_inequality_solution_l3849_384989

def solution_set : Set ℝ := {x : ℝ | x^2 - 3*x - 4 < 0}

theorem inequality_solution : solution_set = Set.Ioo (-1) 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3849_384989


namespace NUMINAMATH_CALUDE_max_value_expression_l3849_384931

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8.5 ≤ a ∧ a ≤ 8.5)
  (hb : -8.5 ≤ b ∧ b ≤ 8.5)
  (hc : -8.5 ≤ c ∧ c ≤ 8.5)
  (hd : -8.5 ≤ d ∧ d ≤ 8.5) :
  (∀ x y z w, -8.5 ≤ x ∧ x ≤ 8.5 ∧ 
              -8.5 ≤ y ∧ y ≤ 8.5 ∧ 
              -8.5 ≤ z ∧ z ≤ 8.5 ∧ 
              -8.5 ≤ w ∧ w ≤ 8.5 → 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 306) ∧
  (∃ x y z w, -8.5 ≤ x ∧ x ≤ 8.5 ∧ 
              -8.5 ≤ y ∧ y ≤ 8.5 ∧ 
              -8.5 ≤ z ∧ z ≤ 8.5 ∧ 
              -8.5 ≤ w ∧ w ≤ 8.5 ∧ 
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 306) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3849_384931


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l3849_384951

/-- Calculate the toll for a truck based on its number of axles -/
def toll (x : ℕ) : ℚ := 2.5 + 0.5 * (x - 2)

/-- Calculate the number of axles for a truck given its wheel configuration -/
def axle_count (total_wheels front_wheels other_axle_wheels : ℕ) : ℕ :=
  1 + (total_wheels - front_wheels) / other_axle_wheels

theorem truck_toll_calculation (total_wheels front_wheels other_axle_wheels : ℕ) 
  (h1 : total_wheels = 18)
  (h2 : front_wheels = 2)
  (h3 : other_axle_wheels = 4) :
  toll (axle_count total_wheels front_wheels other_axle_wheels) = 4 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_calculation_l3849_384951


namespace NUMINAMATH_CALUDE_vertex_of_f_l3849_384930

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Theorem stating that the vertex of f is at (-1, 3)
theorem vertex_of_f :
  (∃ (a : ℝ), f a = 3 ∧ ∀ x, f x ≤ 3) ∧ f (-1) = 3 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_f_l3849_384930


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3849_384941

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3849_384941


namespace NUMINAMATH_CALUDE_number_of_male_students_l3849_384937

theorem number_of_male_students 
  (total_average : ℝ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (num_female : ℕ) 
  (h1 : total_average = 90) 
  (h2 : male_average = 84) 
  (h3 : female_average = 92) 
  (h4 : num_female = 24) :
  ∃ (num_male : ℕ), 
    num_male = 8 ∧ 
    (num_male : ℝ) * male_average + (num_female : ℝ) * female_average = 
      ((num_male : ℝ) + (num_female : ℝ)) * total_average :=
by sorry

end NUMINAMATH_CALUDE_number_of_male_students_l3849_384937


namespace NUMINAMATH_CALUDE_pipe_laying_efficiency_l3849_384910

theorem pipe_laying_efficiency 
  (n : ℕ) 
  (sequential_length : ℝ) 
  (h1 : n = 7) 
  (h2 : sequential_length = 60) :
  let individual_work_time := sequential_length / (6 * n)
  let total_time := n * individual_work_time
  let simultaneous_rate := n * (sequential_length / total_time)
  simultaneous_rate * total_time = 130 := by
sorry

end NUMINAMATH_CALUDE_pipe_laying_efficiency_l3849_384910


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3849_384909

theorem tangent_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) (h₃ : r₃ = 3) :
  ∃ r : ℝ, r > 0 ∧
  (r₁ + r)^2 + (r₂ + r)^2 = (r₃ - r)^2 + (r₁ + r₂)^2 ∧
  r = 6/7 := by
sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3849_384909


namespace NUMINAMATH_CALUDE_target_hit_probability_l3849_384903

/-- The binomial probability function -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem target_hit_probability :
  let n : ℕ := 6
  let k : ℕ := 5
  let p : ℝ := 0.8
  abs (binomial_probability n k p - 0.3932) < 0.00005 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3849_384903


namespace NUMINAMATH_CALUDE_red_balls_count_l3849_384974

/-- Given a jar with white and red balls where the ratio of white to red balls is 4:3 
    and there are 12 white balls, prove that there are 9 red balls. -/
theorem red_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 4 / 3 → white_balls = 12 → red_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l3849_384974


namespace NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l3849_384949

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The condition for the line to intersect the parabola at one point -/
def intersects_at_one_point (k : ℝ) : Prop :=
  ∃! y : ℝ, parabola y = k

/-- The theorem stating the value of k for which the line intersects the parabola at one point -/
theorem line_intersects_parabola_at_one_point :
  ∃! k : ℝ, intersects_at_one_point k ∧ k = 25/3 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l3849_384949


namespace NUMINAMATH_CALUDE_f_properties_l3849_384913

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem f_properties :
  (∃ (x : ℝ), f x = 0 ∧ x = 1) ∧
  (f 0 * f 2 > 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x < 2 ∧ f x = 0) ∧
  (¬ ∀ (x y : ℝ), x < y ∧ y < 0 → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3849_384913


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l3849_384912

/-- A pentagonal prism is a three-dimensional geometric shape with pentagonal bases and rectangular lateral faces. -/
structure PentagonalPrism where
  /-- The number of faces in a pentagonal prism -/
  faces : Nat
  /-- The number of edges in a pentagonal prism -/
  edges : Nat
  /-- The number of vertices in a pentagonal prism -/
  vertices : Nat
  /-- The faces of a pentagonal prism consist of 2 pentagonal bases and 5 rectangular lateral faces -/
  faces_def : faces = 7
  /-- The edges of a pentagonal prism consist of 10 edges from the two pentagons and 5 edges connecting them -/
  edges_def : edges = 15
  /-- The vertices of a pentagonal prism are the 5 vertices from each of the two pentagonal bases -/
  vertices_def : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : p.faces + p.edges + p.vertices = 32 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l3849_384912


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3849_384928

/-- Represents the number of fish of a particular species in the pond -/
structure FishPopulation where
  speciesA : ℕ
  speciesB : ℕ
  speciesC : ℕ

/-- Represents the number of tagged fish caught in the second round -/
structure TaggedCatch where
  speciesA : ℕ
  speciesB : ℕ
  speciesC : ℕ

/-- Calculates the estimated population of a species based on the initial tagging and second catch -/
def estimatePopulation (initialTagged : ℕ) (secondCatchTotal : ℕ) (taggedInSecondCatch : ℕ) : ℕ :=
  (initialTagged * secondCatchTotal) / taggedInSecondCatch

/-- Theorem stating the estimated fish population given the initial tagging and second catch data -/
theorem fish_population_estimate 
  (initialTagged : ℕ) 
  (secondCatchTotal : ℕ) 
  (taggedCatch : TaggedCatch) : 
  initialTagged = 40 →
  secondCatchTotal = 180 →
  taggedCatch.speciesA = 3 →
  taggedCatch.speciesB = 5 →
  taggedCatch.speciesC = 2 →
  let estimatedPopulation := FishPopulation.mk
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesA)
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesB)
    (estimatePopulation initialTagged secondCatchTotal taggedCatch.speciesC)
  estimatedPopulation.speciesA = 2400 ∧ 
  estimatedPopulation.speciesB = 1440 ∧ 
  estimatedPopulation.speciesC = 3600 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l3849_384928


namespace NUMINAMATH_CALUDE_total_cost_of_pipes_l3849_384940

def copper_length : ℝ := 10
def plastic_length : ℝ := copper_length + 5
def cost_per_meter : ℝ := 4

theorem total_cost_of_pipes : copper_length * cost_per_meter + plastic_length * cost_per_meter = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_pipes_l3849_384940


namespace NUMINAMATH_CALUDE_fraction_to_percentage_l3849_384985

/-- Represents a mixed repeating decimal number -/
structure MixedRepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ

/-- Converts a rational number to a MixedRepeatingDecimal -/
def toMixedRepeatingDecimal (q : ℚ) : MixedRepeatingDecimal :=
  sorry

/-- Converts a MixedRepeatingDecimal to a percentage string -/
def toPercentageString (m : MixedRepeatingDecimal) : String :=
  sorry

theorem fraction_to_percentage (n d : ℕ) (h : d ≠ 0) :
  toPercentageString (toMixedRepeatingDecimal (n / d)) = "8.(923076)%" :=
sorry

end NUMINAMATH_CALUDE_fraction_to_percentage_l3849_384985


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3849_384967

/-- Proves that the weight of the replaced person is 55 kg given the conditions -/
theorem replaced_person_weight (initial_count : ℕ) (weight_increase : ℝ) (new_person_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4 →
  new_person_weight = 87 →
  (initial_count : ℝ) * weight_increase + new_person_weight = 
    (initial_count : ℝ) * weight_increase + 55 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3849_384967


namespace NUMINAMATH_CALUDE_infinite_chessboard_rightlines_l3849_384961

-- Define a rightline as a sequence of natural numbers
def Rightline := ℕ → ℕ

-- A rightline without multiples of 3
def NoMultiplesOfThree (r : Rightline) : Prop :=
  ∀ n : ℕ, r n % 3 ≠ 0

-- Pairwise disjoint rightlines
def PairwiseDisjoint (rs : ℕ → Rightline) : Prop :=
  ∀ i j : ℕ, i ≠ j → (∀ n : ℕ, rs i n ≠ rs j n)

theorem infinite_chessboard_rightlines :
  (∃ r : Rightline, NoMultiplesOfThree r) ∧
  (∃ rs : ℕ → Rightline, PairwiseDisjoint rs ∧ (∀ i : ℕ, NoMultiplesOfThree (rs i))) :=
sorry

end NUMINAMATH_CALUDE_infinite_chessboard_rightlines_l3849_384961


namespace NUMINAMATH_CALUDE_inequality_proof_l3849_384962

theorem inequality_proof (a b c : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0)
  (sum_one : a + b + c = 1) : 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3849_384962


namespace NUMINAMATH_CALUDE_fraction_inequality_l3849_384997

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3849_384997


namespace NUMINAMATH_CALUDE_certain_number_proof_l3849_384917

theorem certain_number_proof (x : ℝ) : 3 * (x + 8) = 36 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3849_384917


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3849_384954

theorem tan_alpha_value (α : ℝ) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) : 
  Real.tan α = -1/3 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3849_384954


namespace NUMINAMATH_CALUDE_checkerboard_exists_l3849_384977

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the boundary -/
def isAdjacentToBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic2x2 (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color, 
    board i j = c ∧ 
    board (i+1) j = c ∧ 
    board i (j+1) = c ∧ 
    board (i+1) (j+1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard2x2 (board : Board) (i j : Fin 100) : Prop :=
  (board i j = Color.Black ∧ board (i+1) (j+1) = Color.Black ∧
   board (i+1) j = Color.White ∧ board i (j+1) = Color.White) ∨
  (board i j = Color.White ∧ board (i+1) (j+1) = Color.White ∧
   board (i+1) j = Color.Black ∧ board i (j+1) = Color.Black)

theorem checkerboard_exists (board : Board) 
  (boundary_black : ∀ i j : Fin 100, isAdjacentToBoundary i j → board i j = Color.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic2x2 board i j) :
  ∃ i j : Fin 100, isCheckerboard2x2 board i j :=
sorry

end NUMINAMATH_CALUDE_checkerboard_exists_l3849_384977


namespace NUMINAMATH_CALUDE_complex_point_not_in_third_quadrant_l3849_384952

theorem complex_point_not_in_third_quadrant (m : ℝ) :
  ¬(m^2 + m - 2 < 0 ∧ 6 - m - m^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_point_not_in_third_quadrant_l3849_384952


namespace NUMINAMATH_CALUDE_quadratic_roots_l3849_384987

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
  (∃ u v : ℝ, u ≠ v ∧ a * u^2 + b * u + c = 0 ∧ a * v^2 + b * v + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3849_384987


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l3849_384990

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l3849_384990


namespace NUMINAMATH_CALUDE_bucket_capacity_problem_l3849_384947

theorem bucket_capacity_problem (tank_capacity : ℝ) (first_case_buckets : ℕ) (second_case_buckets : ℕ) (second_case_capacity : ℝ) :
  first_case_buckets = 13 →
  second_case_buckets = 39 →
  second_case_capacity = 17 →
  tank_capacity = first_case_buckets * (tank_capacity / first_case_buckets) →
  tank_capacity = second_case_buckets * second_case_capacity →
  tank_capacity / first_case_buckets = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_problem_l3849_384947


namespace NUMINAMATH_CALUDE_max_digit_sum_l3849_384944

def DigitalClock := Fin 24 × Fin 60

def digit_sum (time : DigitalClock) : Nat :=
  let (h, m) := time
  let h1 := h.val / 10
  let h2 := h.val % 10
  let m1 := m.val / 10
  let m2 := m.val % 10
  h1 + h2 + m1 + m2

theorem max_digit_sum :
  ∃ (max_time : DigitalClock), ∀ (time : DigitalClock), digit_sum time ≤ digit_sum max_time ∧ digit_sum max_time = 19 := by
  sorry

end NUMINAMATH_CALUDE_max_digit_sum_l3849_384944


namespace NUMINAMATH_CALUDE_one_million_divided_by_one_fourth_l3849_384905

theorem one_million_divided_by_one_fourth : 
  (1000000 : ℝ) / (1/4 : ℝ) = 4000000 := by sorry

end NUMINAMATH_CALUDE_one_million_divided_by_one_fourth_l3849_384905


namespace NUMINAMATH_CALUDE_must_divide_p_l3849_384994

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 40)
  (h3 : Nat.gcd r s = 60)
  (h4 : 120 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 180) : 
  7 ∣ p.val := by
  sorry

end NUMINAMATH_CALUDE_must_divide_p_l3849_384994


namespace NUMINAMATH_CALUDE_ipod_ratio_l3849_384924

-- Define the initial number of iPods Emmy has
def emmy_initial : ℕ := 14

-- Define the number of iPods Emmy loses
def emmy_lost : ℕ := 6

-- Define the total number of iPods Emmy and Rosa have together
def total_ipods : ℕ := 12

-- Define Emmy's remaining iPods
def emmy_remaining : ℕ := emmy_initial - emmy_lost

-- Define Rosa's iPods
def rosa_ipods : ℕ := total_ipods - emmy_remaining

-- Theorem statement
theorem ipod_ratio : 
  emmy_remaining * 1 = rosa_ipods * 2 := by
  sorry

end NUMINAMATH_CALUDE_ipod_ratio_l3849_384924


namespace NUMINAMATH_CALUDE_article_price_calculation_l3849_384925

/-- The original price of an article before discounts and tax -/
def original_price : ℝ := 259.20

/-- The final price of the article after discounts and tax -/
def final_price : ℝ := 144

/-- The first discount rate -/
def discount1 : ℝ := 0.12

/-- The second discount rate -/
def discount2 : ℝ := 0.22

/-- The third discount rate -/
def discount3 : ℝ := 0.15

/-- The sales tax rate -/
def tax_rate : ℝ := 0.06

theorem article_price_calculation (ε : ℝ) (hε : ε > 0) :
  ∃ (price : ℝ), 
    abs (price - original_price) < ε ∧ 
    price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 + tax_rate) = final_price :=
sorry

end NUMINAMATH_CALUDE_article_price_calculation_l3849_384925


namespace NUMINAMATH_CALUDE_decagon_partition_impossible_l3849_384938

/-- A partition of a polygon into triangles -/
structure TrianglePartition (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  is_valid : black_sides - white_sides = n

/-- Property that the number of sides in a valid triangle partition is divisible by 3 -/
def sides_divisible_by_three (partition : TrianglePartition n) : Prop :=
  partition.black_sides % 3 = 0 ∧ partition.white_sides % 3 = 0

theorem decagon_partition_impossible :
  ¬ ∃ (partition : TrianglePartition 10), sides_divisible_by_three partition :=
sorry

end NUMINAMATH_CALUDE_decagon_partition_impossible_l3849_384938


namespace NUMINAMATH_CALUDE_inequalities_not_always_hold_l3849_384979

theorem inequalities_not_always_hold :
  ∃ (a b c x y z : ℝ),
    x < a ∧ y < b ∧ z < c ∧
    ¬(x * y + y * z + z * x < a * b + b * c + c * a) ∧
    ¬(x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧
    ¬(x * y * z < a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_always_hold_l3849_384979


namespace NUMINAMATH_CALUDE_binomial_12_choose_3_l3849_384927

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_3_l3849_384927
