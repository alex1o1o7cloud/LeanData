import Mathlib

namespace NUMINAMATH_CALUDE_distributive_property_l1143_114309

theorem distributive_property (x : ℝ) : -2 * (x + 1) = -2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l1143_114309


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1143_114382

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), 2 * a * x - b * y + 2 = 0 ∧
                 x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧
                 (∃ (x1 y1 x2 y2 : ℝ),
                    2 * a * x1 - b * y1 + 2 = 0 ∧
                    x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0 ∧
                    2 * a * x2 - b * y2 + 2 = 0 ∧
                    x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0 ∧
                    (x2 - x1)^2 + (y2 - y1)^2 = 16)) →
  (∀ c d : ℝ, c > 0 → d > 0 →
    (∃ (x y : ℝ), 2 * c * x - d * y + 2 = 0 ∧
                   x^2 + y^2 + 2*x - 4*y + 1 = 0) →
    1/a + 1/b ≤ 1/c + 1/d) ∧
  (1/a + 1/b = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1143_114382


namespace NUMINAMATH_CALUDE_alloy_problem_solution_l1143_114356

/-- Represents the copper-tin alloy problem -/
structure AlloyProblem where
  mass1 : ℝ  -- Mass of the first alloy
  copper1 : ℝ  -- Copper percentage in the first alloy
  mass2 : ℝ  -- Mass of the second alloy
  copper2 : ℝ  -- Copper percentage in the second alloy
  targetMass : ℝ  -- Target mass of the resulting alloy

/-- Represents the solution to the copper-tin alloy problem -/
structure AlloySolution where
  pMin : ℝ  -- Minimum percentage of copper in the resulting alloy
  pMax : ℝ  -- Maximum percentage of copper in the resulting alloy
  mass1 : ℝ → ℝ  -- Function to calculate mass of the first alloy
  mass2 : ℝ → ℝ  -- Function to calculate mass of the second alloy

/-- Theorem stating the solution to the copper-tin alloy problem -/
theorem alloy_problem_solution (problem : AlloyProblem) 
  (h1 : problem.mass1 = 4) 
  (h2 : problem.copper1 = 40) 
  (h3 : problem.mass2 = 6) 
  (h4 : problem.copper2 = 30) 
  (h5 : problem.targetMass = 8) :
  ∃ (solution : AlloySolution),
    solution.pMin = 32.5 ∧
    solution.pMax = 35 ∧
    (∀ p, solution.mass1 p = 0.8 * p - 24) ∧
    (∀ p, solution.mass2 p = 32 - 0.8 * p) ∧
    (∀ p, 32.5 ≤ p → p ≤ 35 → 
      0 ≤ solution.mass1 p ∧ 
      solution.mass1 p ≤ problem.mass1 ∧
      0 ≤ solution.mass2 p ∧ 
      solution.mass2 p ≤ problem.mass2 ∧
      solution.mass1 p + solution.mass2 p = problem.targetMass ∧
      solution.mass1 p * (problem.copper1 / 100) + solution.mass2 p * (problem.copper2 / 100) = 
        problem.targetMass * (p / 100)) :=
by
  sorry

end NUMINAMATH_CALUDE_alloy_problem_solution_l1143_114356


namespace NUMINAMATH_CALUDE_x_equals_y_when_q_is_seven_l1143_114394

theorem x_equals_y_when_q_is_seven :
  ∀ (q : ℤ), 
  let x := 55 + 2 * q
  let y := 4 * q + 41
  q = 7 → x = y :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_when_q_is_seven_l1143_114394


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1143_114397

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < Real.exp 2}

-- Define the complement of B
def C_R_B : Set ℝ := {x | x ≤ 1 ∨ Real.exp 2 ≤ x}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ C_R_B = {x | 0 < x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1143_114397


namespace NUMINAMATH_CALUDE_factorization_equality_l1143_114313

theorem factorization_equality (a b : ℝ) : 4 * a^2 * b - b = b * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1143_114313


namespace NUMINAMATH_CALUDE_divisors_of_1442_l1143_114336

theorem divisors_of_1442 :
  let n : ℕ := 1442
  let divisors : Finset ℕ := {1, 11, 131, 1442}
  (∀ (d : ℕ), d ∣ n ↔ d ∈ divisors) ∧
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ n = p * q) := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_1442_l1143_114336


namespace NUMINAMATH_CALUDE_complex_imaginary_problem_l1143_114330

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_imaginary_problem (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isPurelyImaginary ((z + 1)^2 - 2*I)) : 
  z = -I := by sorry

end NUMINAMATH_CALUDE_complex_imaginary_problem_l1143_114330


namespace NUMINAMATH_CALUDE_no_sequence_satisfying_conditions_l1143_114384

theorem no_sequence_satisfying_conditions : ¬ ∃ (a : ℕ → ℤ), 
  (∀ i j : ℕ, i ≠ j → a i ≠ a j) ∧ 
  (∀ k : ℕ, k > 0 → a (k^2) > 0 ∧ a (k^2 + k) < 0) ∧
  (∀ n : ℕ, n > 0 → |a (n + 1) - a n| ≤ 2023 * Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_satisfying_conditions_l1143_114384


namespace NUMINAMATH_CALUDE_equation_solutions_l1143_114381

theorem equation_solutions :
  (∀ x : ℝ, (3 * x - 1)^2 = 9 ↔ x = 4/3 ∨ x = -2/3) ∧
  (∀ x : ℝ, x * (2 * x - 4) = (2 - x)^2 ↔ x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1143_114381


namespace NUMINAMATH_CALUDE_total_splash_width_l1143_114380

/-- Represents the splash width of different rock types -/
def splash_width (rock_type : String) : ℚ :=
  match rock_type with
  | "pebble" => 1/4
  | "rock" => 1/2
  | "boulder" => 2
  | "mini-boulder" => 1
  | "large_pebble" => 1/3
  | _ => 0

/-- Calculates the total splash width for a given rock type and count -/
def total_splash (rock_type : String) (count : ℕ) : ℚ :=
  (splash_width rock_type) * count

/-- Theorem: The total width of splashes is 14 meters -/
theorem total_splash_width :
  (total_splash "pebble" 8) +
  (total_splash "rock" 4) +
  (total_splash "boulder" 3) +
  (total_splash "mini-boulder" 2) +
  (total_splash "large_pebble" 6) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_splash_width_l1143_114380


namespace NUMINAMATH_CALUDE_constant_value_l1143_114366

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f x + c * f (8 - x) = x

theorem constant_value (f : ℝ → ℝ) (c : ℝ) 
    (h1 : SatisfiesCondition f c) 
    (h2 : f 2 = 2) : 
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1143_114366


namespace NUMINAMATH_CALUDE_tetrahedron_center_of_mass_l1143_114329

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D

/-- The centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- The circumcenter of a tetrahedron -/
def circumcenter (t : Tetrahedron) : Point3D := sorry

/-- The orthocenter of a tetrahedron -/
def orthocenter (t : Tetrahedron) : Point3D := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def is_midpoint (m p1 p2 : Point3D) : Prop := sorry

/-- Calculates the center of mass given masses and their positions -/
def center_of_mass (masses : List ℝ) (positions : List Point3D) : Point3D := sorry

/-- Main theorem -/
theorem tetrahedron_center_of_mass (t : Tetrahedron) :
  let s := centroid t
  let o := circumcenter t
  let m := orthocenter t
  collinear s o m ∧ is_midpoint s o m →
  center_of_mass 
    [1, 1, 1, 1, -2] 
    [t.vertices 0, t.vertices 1, t.vertices 2, t.vertices 3, m] = o := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_center_of_mass_l1143_114329


namespace NUMINAMATH_CALUDE_diamond_inequality_exists_l1143_114302

/-- Definition of the diamond operation -/
def diamond (f : ℝ → ℝ) (x y : ℝ) : ℝ := |f x - f y|

/-- The function f(x) = 3x -/
def f (x : ℝ) : ℝ := 3 * x

/-- Theorem stating that 3(x ◊ y) ≠ (3x) ◊ (3y) for some x and y -/
theorem diamond_inequality_exists : ∃ x y : ℝ, 3 * (diamond f x y) ≠ diamond f (3 * x) (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_diamond_inequality_exists_l1143_114302


namespace NUMINAMATH_CALUDE_max_area_PCD_l1143_114327

/-- Definition of the ellipse Γ -/
def Γ (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- Definition of point A (left vertex) -/
def A (a : ℝ) : ℝ × ℝ := (-a, 0)

/-- Definition of point B (top vertex) -/
def B (b : ℝ) : ℝ × ℝ := (0, b)

/-- Definition of point P on the ellipse in the fourth quadrant -/
def P (a b : ℝ) : {p : ℝ × ℝ // Γ a b p.1 p.2 ∧ p.1 > 0 ∧ p.2 < 0} := sorry

/-- Definition of point C (intersection of PA with y-axis) -/
def C (a b : ℝ) : ℝ × ℝ := sorry

/-- Definition of point D (intersection of PB with x-axis) -/
def D (a b : ℝ) : ℝ × ℝ := sorry

/-- Area of triangle PCD -/
def area_PCD (a b : ℝ) : ℝ := sorry

/-- Theorem stating the maximum area of triangle PCD -/
theorem max_area_PCD (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ (max_area : ℝ), max_area = (Real.sqrt 2 - 1) / 2 * a * b ∧
    ∀ (p : ℝ × ℝ), Γ a b p.1 p.2 → p.1 > 0 → p.2 < 0 →
      area_PCD a b ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_area_PCD_l1143_114327


namespace NUMINAMATH_CALUDE_circle_regions_l1143_114325

/-- Number of regions created by n circles -/
def P (n : ℕ) : ℕ := 2 + n * (n - 1)

/-- The problem statement -/
theorem circle_regions : P 2011 ≡ 2112 [ZMOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_circle_regions_l1143_114325


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1143_114311

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (f_prime a (-3) = 0 →
    ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (y = m*x + b ↔ x = 0 ∨ y - f a 0 = m*(x - 0))) ∧
  ((∀ x : ℝ, x ∈ Set.Icc 1 2 → f_prime a x ≤ 0) →
    a ≤ -15/4) := by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1143_114311


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1143_114305

theorem rectangle_longer_side (a : ℝ) (h1 : a > 0) : 
  (a * (0.8 * a) = 81 / 20) → a = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1143_114305


namespace NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l1143_114347

/-- Represents a 25x25 table with integers from 1 to 25 -/
def Table := Fin 25 → Fin 25 → Fin 25

/-- The table is symmetric with respect to the main diagonal -/
def isSymmetric (t : Table) : Prop :=
  ∀ i j : Fin 25, t i j = t j i

/-- Each row contains all numbers from 1 to 25 -/
def hasAllNumbersInRow (t : Table) : Prop :=
  ∀ i : Fin 25, ∀ k : Fin 25, ∃ j : Fin 25, t i j = k

/-- The main diagonal contains all numbers from 1 to 25 -/
def allNumbersOnDiagonal (t : Table) : Prop :=
  ∀ k : Fin 25, ∃ i : Fin 25, t i i = k

theorem diagonal_contains_all_numbers (t : Table) 
  (h_sym : isSymmetric t) (h_row : hasAllNumbersInRow t) : 
  allNumbersOnDiagonal t := by
  sorry

end NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l1143_114347


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1143_114308

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the increase in average for a batsman -/
def averageIncrease (prevInnings : ℕ) (prevTotalScore : ℕ) (newScore : ℕ) : ℚ :=
  let newAverage := (prevTotalScore + newScore) / (prevInnings + 1)
  let oldAverage := prevTotalScore / prevInnings
  newAverage - oldAverage

/-- Theorem stating the increase in average for the given batsman -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 19 →
    b.average = 64 →
    averageIncrease 18 (18 * (b.totalScore / 19)) 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1143_114308


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1143_114339

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 265) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1143_114339


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1143_114359

theorem consecutive_integers_cube_sum (x : ℕ) (h : x > 0) 
  (h_prod : x * (x + 1) * (x + 2) = 12 * (3 * x + 3)) :
  x^3 + (x + 1)^3 + (x + 2)^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l1143_114359


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achievable_l1143_114331

theorem min_value_quadratic (x : ℝ) : x^2 + x + 1 ≥ 3/4 :=
sorry

theorem min_value_quadratic_achievable : ∃ x : ℝ, x^2 + x + 1 = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achievable_l1143_114331


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l1143_114387

theorem min_value_cos_sin (θ : Real) (h : π/2 < θ ∧ θ < 3*π/2) :
  ∃ (min_val : Real), min_val = Real.sqrt 3 / 2 - 3 / 4 ∧
  ∀ (y : Real), y = Real.cos (θ/2) * (1 - Real.sin θ) → y ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l1143_114387


namespace NUMINAMATH_CALUDE_three_white_marbles_probability_l1143_114375

def total_marbles : ℕ := 5 + 7 + 15

def probability_three_white (red green white : ℕ) : ℚ :=
  (white / total_marbles) * 
  ((white - 1) / (total_marbles - 1)) * 
  ((white - 2) / (total_marbles - 2))

theorem three_white_marbles_probability :
  probability_three_white 5 7 15 = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_three_white_marbles_probability_l1143_114375


namespace NUMINAMATH_CALUDE_inequality_proof_l1143_114315

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1143_114315


namespace NUMINAMATH_CALUDE_equation_solution_l1143_114383

theorem equation_solution (x : ℝ) : 1 + 1 / (1 + x) = 2 / (1 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1143_114383


namespace NUMINAMATH_CALUDE_base8_subtraction_l1143_114334

-- Define a function to convert base 8 numbers to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction :
  natToBase8 ((base8ToNat 256 + base8ToNat 167) - base8ToNat 145) = 370 := by
  sorry

end NUMINAMATH_CALUDE_base8_subtraction_l1143_114334


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l1143_114319

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 1)
  (eq3 : a + c + d = 16)
  (eq4 : b + c + d = 9) :
  a * b + c * d = 734 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l1143_114319


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l1143_114322

theorem modulus_of_complex_product : Complex.abs ((3 - 4 * Complex.I) * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l1143_114322


namespace NUMINAMATH_CALUDE_sqrt_sum_div_sqrt_eq_rational_l1143_114310

theorem sqrt_sum_div_sqrt_eq_rational : (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175 = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_div_sqrt_eq_rational_l1143_114310


namespace NUMINAMATH_CALUDE_walking_sequence_intersection_l1143_114365

/-- A walking sequence is a sequence of integers where each term differs from the previous by ±1. -/
def IsWalkingSequence (a : Fin 2016 → ℤ) : Prop :=
  ∀ i : Fin 2015, a (i + 1) = a i + 1 ∨ a (i + 1) = a i - 1

/-- The sequence b as defined in the problem -/
def b : Fin 2016 → ℤ
  | ⟨i, h⟩ => if i < 1009 then i + 1 else 2018 - i

/-- The main theorem statement -/
theorem walking_sequence_intersection :
  ∃ (a : Fin 2016 → ℤ), IsWalkingSequence a ∧
  (∀ i, 1 ≤ a i ∧ a i ≤ 1010) →
  ∃ j, a j = b j :=
sorry

end NUMINAMATH_CALUDE_walking_sequence_intersection_l1143_114365


namespace NUMINAMATH_CALUDE_hancho_milk_consumption_l1143_114312

theorem hancho_milk_consumption (total_milk : Real) (yeseul_milk : Real) (gayoung_extra : Real) (remaining_milk : Real) :
  total_milk = 1 →
  yeseul_milk = 0.1 →
  gayoung_extra = 0.2 →
  remaining_milk = 0.3 →
  total_milk - yeseul_milk - (yeseul_milk + gayoung_extra) - remaining_milk = 0.3 := by
  sorry

#check hancho_milk_consumption

end NUMINAMATH_CALUDE_hancho_milk_consumption_l1143_114312


namespace NUMINAMATH_CALUDE_problem_solution_l1143_114343

/-- Predicate to check if a number is divisible by another -/
def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬is_divisible n d

/-- The four statements in the problem -/
def statement1 (a b : ℕ) : Prop := is_divisible (a^2 + 6*a + 8) b
def statement2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 15*b - 9 = 0
def statement3 (a b : ℕ) : Prop := is_divisible (a + 2*b + 2) 4
def statement4 (a b : ℕ) : Prop := is_prime (a + 6*b + 2)

/-- Predicate to check if exactly three out of four statements are true -/
def three_true (a b : ℕ) : Prop :=
  (statement1 a b ∧ statement2 a b ∧ statement3 a b ∧ ¬statement4 a b) ∨
  (statement1 a b ∧ statement2 a b ∧ ¬statement3 a b ∧ statement4 a b) ∨
  (statement1 a b ∧ ¬statement2 a b ∧ statement3 a b ∧ statement4 a b) ∨
  (¬statement1 a b ∧ statement2 a b ∧ statement3 a b ∧ statement4 a b)

theorem problem_solution :
  ∀ a b : ℕ, three_true a b ↔ ((a = 5 ∧ b = 1) ∨ (a = 17 ∧ b = 7)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1143_114343


namespace NUMINAMATH_CALUDE_square_sum_inequality_l1143_114363

theorem square_sum_inequality (x y : ℝ) :
  x^2 + y^2 ≤ 2*(x + y - 1) → x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l1143_114363


namespace NUMINAMATH_CALUDE_last_digit_101_power_100_l1143_114358

theorem last_digit_101_power_100 : 101^100 ≡ 1 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_last_digit_101_power_100_l1143_114358


namespace NUMINAMATH_CALUDE_train_length_l1143_114360

/-- Given a train that crosses two platforms of different lengths at different times, 
    this theorem proves the length of the train. -/
theorem train_length 
  (platform1_length : ℝ) 
  (platform1_time : ℝ) 
  (platform2_length : ℝ) 
  (platform2_time : ℝ) 
  (h1 : platform1_length = 120)
  (h2 : platform1_time = 15)
  (h3 : platform2_length = 250)
  (h4 : platform2_time = 20) :
  ∃ train_length : ℝ, 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧ 
    train_length = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1143_114360


namespace NUMINAMATH_CALUDE_josh_marbles_difference_l1143_114391

/-- Given Josh's marble collection scenario, prove the difference between lost and found marbles. -/
theorem josh_marbles_difference (initial : ℕ) (found : ℕ) (lost : ℕ) 
  (h1 : initial = 15) 
  (h2 : found = 9) 
  (h3 : lost = 23) : 
  lost - found = 14 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_difference_l1143_114391


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1143_114374

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 = (x^2 + 7*x + 2) * q + (-315*x - 94) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1143_114374


namespace NUMINAMATH_CALUDE_burglars_money_min_burglars_money_l1143_114364

def x (a n : ℕ) : ℚ := (a / 4 : ℚ) * (1 - (1 / 3 : ℚ) ^ n)

theorem burglars_money (a : ℕ) : 
  (∀ n : ℕ, n ≤ 2012 → (x a n).num % (x a n).den = 0 ∧ ((a : ℚ) - x a n).num % ((a : ℚ) - x a n).den = 0) →
  a ≥ 4 * 3^2012 :=
sorry

theorem min_burglars_money : 
  ∃ a : ℕ, a = 4 * 3^2012 ∧ 
  (∀ n : ℕ, n ≤ 2012 → (x a n).num % (x a n).den = 0 ∧ ((a : ℚ) - x a n).num % ((a : ℚ) - x a n).den = 0) ∧
  (∀ b : ℕ, b < a → ∃ n : ℕ, n ≤ 2012 ∧ ((x b n).num % (x b n).den ≠ 0 ∨ ((b : ℚ) - x b n).num % ((b : ℚ) - x b n).den ≠ 0)) :=
sorry

end NUMINAMATH_CALUDE_burglars_money_min_burglars_money_l1143_114364


namespace NUMINAMATH_CALUDE_cloth_sale_profit_l1143_114332

/-- The number of meters of cloth sold by a trader -/
def meters_sold : ℕ := 40

/-- The profit per meter of cloth in Rupees -/
def profit_per_meter : ℕ := 25

/-- The total profit earned by the trader in Rupees -/
def total_profit : ℕ := 1000

/-- Theorem stating that the number of meters sold multiplied by the profit per meter equals the total profit -/
theorem cloth_sale_profit : meters_sold * profit_per_meter = total_profit := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_profit_l1143_114332


namespace NUMINAMATH_CALUDE_parallel_transitivity_l1143_114399

-- Define a type for planes
variable (Plane : Type)

-- Define a relation for parallel planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity (p1 p2 p3 : Plane) :
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l1143_114399


namespace NUMINAMATH_CALUDE_island_puzzle_l1143_114303

-- Define the types of residents
inductive Resident
| TruthTeller
| Liar

-- Define the statement made by K
def kStatement (k m : Resident) : Prop :=
  k = Resident.Liar ∨ m = Resident.Liar

-- Theorem to prove
theorem island_puzzle :
  ∃ (k m : Resident),
    (k = Resident.TruthTeller ∧ 
     m = Resident.Liar ∧
     (k = Resident.TruthTeller → kStatement k m) ∧
     (k = Resident.Liar → ¬kStatement k m)) :=
sorry

end NUMINAMATH_CALUDE_island_puzzle_l1143_114303


namespace NUMINAMATH_CALUDE_tan_three_expression_zero_l1143_114355

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ - Real.sin θ / (1 - Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_zero_l1143_114355


namespace NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1143_114361

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Theorem for the complement of A ∩ B in U
theorem complement_A_inter_B :
  (A ∩ B)ᶜ = {x | x ≤ 4 ∨ x > 5} :=
sorry

-- Theorem for the range of values for a
theorem range_of_a (a : ℝ) (h : A ∪ B ⊆ C a) :
  a ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1143_114361


namespace NUMINAMATH_CALUDE_min_pool_cost_l1143_114335

def pool_volume : ℝ := 18
def pool_depth : ℝ := 2
def bottom_cost_per_sqm : ℝ := 200
def wall_cost_per_sqm : ℝ := 150

theorem min_pool_cost :
  let length : ℝ → ℝ → ℝ := λ x y => x
  let width : ℝ → ℝ → ℝ := λ x y => y
  let volume : ℝ → ℝ → ℝ := λ x y => x * y * pool_depth
  let bottom_area : ℝ → ℝ → ℝ := λ x y => x * y
  let wall_area : ℝ → ℝ → ℝ := λ x y => 2 * (x + y) * pool_depth
  let total_cost : ℝ → ℝ → ℝ := λ x y => 
    bottom_cost_per_sqm * bottom_area x y + wall_cost_per_sqm * wall_area x y
  ∃ x y : ℝ, 
    volume x y = pool_volume ∧ 
    (∀ a b : ℝ, volume a b = pool_volume → total_cost x y ≤ total_cost a b) ∧
    total_cost x y = 5400 :=
by sorry

end NUMINAMATH_CALUDE_min_pool_cost_l1143_114335


namespace NUMINAMATH_CALUDE_trevor_reed_difference_l1143_114326

/-- Represents the yearly toy spending of Trevor, Reed, and Quinn -/
structure ToySpending where
  trevor : ℕ
  reed : ℕ
  quinn : ℕ

/-- The conditions of the problem -/
def spending_conditions (s : ToySpending) : Prop :=
  s.trevor = 80 ∧
  s.reed = 2 * s.quinn ∧
  s.trevor > s.reed ∧
  4 * (s.trevor + s.reed + s.quinn) = 680

/-- The theorem to prove -/
theorem trevor_reed_difference (s : ToySpending) :
  spending_conditions s → s.trevor - s.reed = 20 := by
  sorry

end NUMINAMATH_CALUDE_trevor_reed_difference_l1143_114326


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1143_114316

/-- Given a triangle ABC with sides a, b, and c, prove that if a = b + 1, b = c + 1, 
    and the perimeter is 21, then a = 8, b = 7, and c = 6. -/
theorem triangle_side_lengths 
  (a b c : ℝ) 
  (h1 : a = b + 1) 
  (h2 : b = c + 1) 
  (h3 : a + b + c = 21) : 
  a = 8 ∧ b = 7 ∧ c = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l1143_114316


namespace NUMINAMATH_CALUDE_photo_arrangements_l1143_114338

/-- The number of different arrangements of 5 students and 2 teachers in a row,
    where exactly two students stand between the two teachers. -/
def arrangements_count : ℕ := 960

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students required to stand between teachers -/
def students_between : ℕ := 2

theorem photo_arrangements :
  arrangements_count = 960 :=
sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1143_114338


namespace NUMINAMATH_CALUDE_parabola_line_intersection_perpendicular_l1143_114393

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem parabola_line_intersection_perpendicular :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 →
  (A.1 * B.1 + A.2 * B.2 = 0) := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_perpendicular_l1143_114393


namespace NUMINAMATH_CALUDE_frog_probability_l1143_114386

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center : Position
| Interior : Position
| Edge : Position

/-- Represents the number of hops -/
def MaxHops : Nat := 5

/-- The probability of reaching an edge from a given position after n hops -/
noncomputable def probability (pos : Position) (n : Nat) : Real :=
  match pos, n with
  | Position.Edge, _ => 1
  | _, 0 => 0
  | Position.Center, n + 1 => 
      (1/4) * (probability Position.Interior n + probability Position.Interior n + 
               probability Position.Edge n + probability Position.Edge n)
  | Position.Interior, n + 1 => 
      (1/4) * (probability Position.Interior n + probability Position.Interior n + 
               probability Position.Edge n + probability Position.Edge n)

/-- The main theorem to be proved -/
theorem frog_probability : 
  probability Position.Center MaxHops = 121/128 := by
  sorry

end NUMINAMATH_CALUDE_frog_probability_l1143_114386


namespace NUMINAMATH_CALUDE_furniture_legs_problem_l1143_114370

theorem furniture_legs_problem (total_tables : ℕ) (total_legs : ℕ) (four_leg_tables : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_leg_tables = 16 →
  (total_legs - 4 * four_leg_tables) / (total_tables - four_leg_tables) = 3 :=
by sorry

end NUMINAMATH_CALUDE_furniture_legs_problem_l1143_114370


namespace NUMINAMATH_CALUDE_chess_piece_paths_l1143_114379

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem chess_piece_paths :
  let num_segments : ℕ := 15
  let steps_per_segment : ℕ := 6
  let ways_per_segment : ℕ := fibonacci (steps_per_segment + 1)
  num_segments * ways_per_segment = 195 :=
by sorry

end NUMINAMATH_CALUDE_chess_piece_paths_l1143_114379


namespace NUMINAMATH_CALUDE_team_selection_count_l1143_114341

/-- The number of ways to select a team of 6 people from a group of 7 boys and 9 girls, with at least 2 boys -/
def selectTeam (boys girls : ℕ) : ℕ := 
  (Nat.choose boys 2 * Nat.choose girls 4) +
  (Nat.choose boys 3 * Nat.choose girls 3) +
  (Nat.choose boys 4 * Nat.choose girls 2) +
  (Nat.choose boys 5 * Nat.choose girls 1) +
  (Nat.choose boys 6 * Nat.choose girls 0)

/-- Theorem stating that the number of ways to select the team is 7042 -/
theorem team_selection_count : selectTeam 7 9 = 7042 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l1143_114341


namespace NUMINAMATH_CALUDE_firewood_collection_sum_l1143_114350

/-- The amount of firewood collected by Kimberley in pounds -/
def kimberley_firewood : ℕ := 10

/-- The amount of firewood collected by Houston in pounds -/
def houston_firewood : ℕ := 12

/-- The amount of firewood collected by Ela in pounds -/
def ela_firewood : ℕ := 13

/-- The total amount of firewood collected by Kimberley, Ela, and Houston -/
def total_firewood : ℕ := kimberley_firewood + ela_firewood + houston_firewood

theorem firewood_collection_sum :
  total_firewood = 35 := by
  sorry

end NUMINAMATH_CALUDE_firewood_collection_sum_l1143_114350


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1143_114346

/-- Given a triangle with inradius 2.5 cm and area 45 cm², its perimeter is 36 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 2.5 → area = 45 → perimeter = 36 := by
  sorry

#check triangle_perimeter

end NUMINAMATH_CALUDE_triangle_perimeter_l1143_114346


namespace NUMINAMATH_CALUDE_opposite_to_gold_is_silver_l1143_114373

-- Define the colors
inductive Color
| P | M | C | S | G | V | L

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define the theorem
theorem opposite_to_gold_is_silver (cube : Cube) : 
  cube.top.color = Color.P → 
  cube.bottom.color = Color.V → 
  (cube.front.color = Color.G ∨ cube.back.color = Color.G ∨ cube.left.color = Color.G ∨ cube.right.color = Color.G) → 
  ((cube.front.color = Color.G → cube.back.color = Color.S) ∧ 
   (cube.back.color = Color.G → cube.front.color = Color.S) ∧ 
   (cube.left.color = Color.G → cube.right.color = Color.S) ∧ 
   (cube.right.color = Color.G → cube.left.color = Color.S)) := by
  sorry


end NUMINAMATH_CALUDE_opposite_to_gold_is_silver_l1143_114373


namespace NUMINAMATH_CALUDE_equation_solution_l1143_114353

theorem equation_solution : 
  {x : ℝ | (Real.sqrt (9*x - 2) + 15 / Real.sqrt (9*x - 2) = 8)} = {3, 11/9} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1143_114353


namespace NUMINAMATH_CALUDE_min_value_theorem_l1143_114345

theorem min_value_theorem (x y : ℝ) (h : x + y = 5) :
  ∃ m : ℝ, m = (6100 : ℝ) / 17 ∧ 
  ∀ z : ℝ, z ≥ m ∧ ∃ a b : ℝ, a + b = 5 ∧ 
  z = a^5*b + a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1143_114345


namespace NUMINAMATH_CALUDE_workshop_technicians_l1143_114378

theorem workshop_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℚ) 
  (avg_salary_tech : ℚ) 
  (avg_salary_others : ℚ) 
  (h1 : total_workers = 20)
  (h2 : avg_salary_all = 750)
  (h3 : avg_salary_tech = 900)
  (h4 : avg_salary_others = 700) :
  ∃ (num_technicians : ℕ), 
    num_technicians * avg_salary_tech + (total_workers - num_technicians) * avg_salary_others = 
    total_workers * avg_salary_all ∧ 
    num_technicians = 5 := by
  sorry

end NUMINAMATH_CALUDE_workshop_technicians_l1143_114378


namespace NUMINAMATH_CALUDE_length_function_is_linear_alpha_is_rate_of_change_l1143_114372

/-- Represents the length of a metal rod as a function of temperature -/
def length_function (l₀ α : ℝ) (t : ℝ) : ℝ := l₀ * (1 + α * t)

/-- States that the length function is linear in t -/
theorem length_function_is_linear (l₀ α : ℝ) : 
  ∃ m b : ℝ, ∀ t : ℝ, length_function l₀ α t = m * t + b :=
sorry

/-- Defines α as the rate of change of length with respect to temperature -/
theorem alpha_is_rate_of_change (l₀ α : ℝ) : 
  α = (length_function l₀ α 1 - length_function l₀ α 0) / l₀ :=
sorry

end NUMINAMATH_CALUDE_length_function_is_linear_alpha_is_rate_of_change_l1143_114372


namespace NUMINAMATH_CALUDE_gcd_18_30_45_l1143_114352

theorem gcd_18_30_45 : Nat.gcd 18 (Nat.gcd 30 45) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_45_l1143_114352


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_five_l1143_114389

/-- Given an arithmetic sequence with first term 105 and common difference -5,
    prove that there are 20 terms before the term with value 5. -/
theorem arithmetic_sequence_before_five (n : ℕ) : 
  (105 : ℤ) - 5 * n = 5 → n - 1 = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_five_l1143_114389


namespace NUMINAMATH_CALUDE_triangulation_has_120_triangle_l1143_114337

/-- A triangulation of a triangle -/
structure Triangulation :=
  (vertices : Set ℝ × ℝ)
  (edges : Set (ℝ × ℝ × ℝ × ℝ))
  (triangles : Set (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ))

/-- The original triangle in a triangulation -/
def originalTriangle (t : Triangulation) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Check if all angles in a triangle are not exceeding 120° -/
def allAnglesWithin120 (triangle : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem statement -/
theorem triangulation_has_120_triangle 
  (t : Triangulation) 
  (h : allAnglesWithin120 (originalTriangle t)) :
  ∃ triangle ∈ t.triangles, allAnglesWithin120 triangle :=
sorry

end NUMINAMATH_CALUDE_triangulation_has_120_triangle_l1143_114337


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1143_114344

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 → 
  ∃ m : ℕ, m = 12 ∧ 
  (∀ k : ℕ, k > m → ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3)))) ∧
  (12 ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry


end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1143_114344


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l1143_114396

theorem complex_expression_equals_negative_two :
  let z : ℂ := Complex.exp (3 * Real.pi * Complex.I / 8)
  (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l1143_114396


namespace NUMINAMATH_CALUDE_train_length_calculation_l1143_114385

theorem train_length_calculation (train_speed : Real) (platform_length : Real) (crossing_time : Real) :
  train_speed = 55 * 1000 / 3600 →
  platform_length = 300 →
  crossing_time = 35.99712023038157 →
  let total_distance := train_speed * crossing_time
  let train_length := total_distance - platform_length
  train_length = 249.9999999999999 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1143_114385


namespace NUMINAMATH_CALUDE_probability_three_standard_parts_l1143_114351

/-- Represents a box containing parts -/
structure Box where
  total : ℕ
  standard : ℕ
  h : standard ≤ total

/-- Calculates the probability of selecting a standard part from a box -/
def probabilityStandard (box : Box) : ℚ :=
  box.standard / box.total

/-- Theorem: The probability of selecting standard parts from all three boxes is 7/10 -/
theorem probability_three_standard_parts
  (box1 : Box)
  (box2 : Box)
  (box3 : Box)
  (h1 : box1.total = 30 ∧ box1.standard = 27)
  (h2 : box2.total = 30 ∧ box2.standard = 28)
  (h3 : box3.total = 30 ∧ box3.standard = 25) :
  probabilityStandard box1 * probabilityStandard box2 * probabilityStandard box3 = 7/10 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_standard_parts_l1143_114351


namespace NUMINAMATH_CALUDE_chocobites_remainder_l1143_114388

theorem chocobites_remainder (m : ℕ) : 
  m % 8 = 5 → (4 * m) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_chocobites_remainder_l1143_114388


namespace NUMINAMATH_CALUDE_number_line_problem_l1143_114323

/-- Given a number line with equally spaced markings, prove that if the starting point is 2,
    the ending point is 34, and there are 8 equal steps between them,
    then the point z reached after 6 steps from 2 is 26. -/
theorem number_line_problem (start end_ : ℝ) (total_steps : ℕ) (steps_to_z : ℕ) :
  start = 2 →
  end_ = 34 →
  total_steps = 8 →
  steps_to_z = 6 →
  let step_length := (end_ - start) / total_steps
  start + steps_to_z * step_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_line_problem_l1143_114323


namespace NUMINAMATH_CALUDE_sum_reciprocal_pairs_bound_l1143_114349

/-- 
Given non-negative real numbers x, y, and z satisfying xy + yz + zx = 1,
the sum 1/(x+y) + 1/(y+z) + 1/(z+x) is greater than or equal to 5/2.
-/
theorem sum_reciprocal_pairs_bound (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum_prod : x*y + y*z + z*x = 1) : 
  1/(x+y) + 1/(y+z) + 1/(z+x) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_pairs_bound_l1143_114349


namespace NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l1143_114368

theorem hexagon_circle_area_ratio (r : ℝ) (h : r > 0) :
  (3 * Real.sqrt 3 * r^2 / 2) / (π * r^2) = 3 * Real.sqrt 3 / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_circle_area_ratio_l1143_114368


namespace NUMINAMATH_CALUDE_sin_absolute_value_condition_l1143_114317

theorem sin_absolute_value_condition (α : ℝ) :
  (|Real.sin α| = -Real.sin α) ↔ ∃ k : ℤ, α ∈ Set.Icc ((2 * k - 1) * Real.pi) (2 * k * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sin_absolute_value_condition_l1143_114317


namespace NUMINAMATH_CALUDE_first_book_price_l1143_114300

/-- Given 41 books arranged in increasing price order with a $3 difference between adjacent books,
    if the sum of the prices of the first and last books is $246,
    then the price of the first book is $63. -/
theorem first_book_price (n : ℕ) (price_diff : ℝ) (total_sum : ℝ) :
  n = 41 →
  price_diff = 3 →
  total_sum = 246 →
  ∃ (first_price : ℝ),
    first_price + (first_price + price_diff * (n - 1)) = total_sum ∧
    first_price = 63 := by
  sorry

end NUMINAMATH_CALUDE_first_book_price_l1143_114300


namespace NUMINAMATH_CALUDE_parabola_symmetry_l1143_114377

/-- Given that M(0,5) and N(2,5) lie on the parabola y = 2(x-m)^2 + 3, prove that m = 1 -/
theorem parabola_symmetry (m : ℝ) : 
  (5 : ℝ) = 2 * (0 - m)^2 + 3 ∧ 
  (5 : ℝ) = 2 * (2 - m)^2 + 3 → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l1143_114377


namespace NUMINAMATH_CALUDE_swimming_club_girls_l1143_114318

theorem swimming_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  boys / 3 + girls = present →
  girls = 12 := by
sorry

end NUMINAMATH_CALUDE_swimming_club_girls_l1143_114318


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1143_114324

theorem quadratic_inequality_theorem (a : ℝ) :
  (∀ m > a, ∀ x : ℝ, x^2 + 2*x + m > 0) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l1143_114324


namespace NUMINAMATH_CALUDE_rachel_research_time_l1143_114369

/-- Represents the time spent on different activities while writing an essay -/
structure EssayTime where
  writing_speed : ℕ  -- pages per 30 minutes
  total_pages : ℕ
  editing_time : ℕ  -- in minutes
  total_time : ℕ    -- in minutes

/-- Calculates the time spent researching for an essay -/
def research_time (e : EssayTime) : ℕ :=
  e.total_time - (e.total_pages * 30 + e.editing_time)

/-- Theorem stating that Rachel spent 45 minutes researching -/
theorem rachel_research_time :
  let e : EssayTime := {
    writing_speed := 1,
    total_pages := 6,
    editing_time := 75,
    total_time := 300
  }
  research_time e = 45 := by sorry

end NUMINAMATH_CALUDE_rachel_research_time_l1143_114369


namespace NUMINAMATH_CALUDE_fraction_simplification_l1143_114301

theorem fraction_simplification (x : ℝ) : (2*x - 3) / 4 + (4*x + 5) / 3 = (22*x + 11) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1143_114301


namespace NUMINAMATH_CALUDE_power_function_m_value_l1143_114371

/-- A function f is a power function if it can be expressed as f(x) = x^n for some constant n. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x^n

/-- The given function parameterized by m. -/
def f (m : ℝ) (x : ℝ) : ℝ := (2*m - m^2) * x^3

theorem power_function_m_value :
  ∃! m : ℝ, IsPowerFunction (f m) ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_power_function_m_value_l1143_114371


namespace NUMINAMATH_CALUDE_forester_tree_planting_l1143_114307

theorem forester_tree_planting (initial_trees : ℕ) (monday_multiplier : ℕ) (tuesday_fraction : ℚ) : 
  initial_trees = 30 →
  monday_multiplier = 3 →
  tuesday_fraction = 1/3 →
  (monday_multiplier * initial_trees - initial_trees) + 
  (tuesday_fraction * (monday_multiplier * initial_trees - initial_trees)) = 80 := by
sorry

end NUMINAMATH_CALUDE_forester_tree_planting_l1143_114307


namespace NUMINAMATH_CALUDE_no_prime_divisor_l1143_114362

theorem no_prime_divisor : ¬ ∃ (p : ℕ), Prime p ∧ p > 1 ∧ p ∣ (1255 - 8) ∧ p ∣ (1490 - 11) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_divisor_l1143_114362


namespace NUMINAMATH_CALUDE_abc_inequalities_l1143_114376

theorem abc_inequalities (a b : Real) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b = 1) : 
  (2 * a^2 + b ≥ 7/8) ∧ 
  (a * b ≤ 1/4) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l1143_114376


namespace NUMINAMATH_CALUDE_square_perimeter_product_l1143_114321

theorem square_perimeter_product (x y : ℝ) (h1 : x^2 + y^2 = 130) (h2 : x^2 - y^2 = 58) :
  (4*x) * (4*y) = 96 * Real.sqrt 94 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_product_l1143_114321


namespace NUMINAMATH_CALUDE_sophias_age_problem_l1143_114398

/-- Sophia's age problem -/
theorem sophias_age_problem (S M : ℝ) (h1 : S > 0) (h2 : M > 0) 
  (h3 : ∃ (x : ℝ), S = 3 * x ∧ x > 0)  -- S is thrice the sum of children's ages
  (h4 : S - M = 4 * ((S / 3) - 2 * M)) :  -- Condition about age M years ago
  S / M = 21 := by
sorry

end NUMINAMATH_CALUDE_sophias_age_problem_l1143_114398


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1143_114304

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1143_114304


namespace NUMINAMATH_CALUDE_kaylin_age_l1143_114328

-- Define variables for each person's age
variable (kaylin sarah eli freyja alfred olivia : ℝ)

-- State the conditions
axiom kaylin_sarah : kaylin = sarah - 5
axiom sarah_eli : sarah = 2 * eli
axiom eli_freyja : eli = freyja + 9
axiom freyja_alfred : freyja = 2.5 * alfred
axiom alfred_olivia : alfred = 0.75 * olivia
axiom freyja_age : freyja = 9.5

-- Theorem to prove
theorem kaylin_age : kaylin = 32 := by
  sorry

end NUMINAMATH_CALUDE_kaylin_age_l1143_114328


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1143_114395

theorem gcd_of_three_numbers : Nat.gcd 18222 (Nat.gcd 24546 66364) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1143_114395


namespace NUMINAMATH_CALUDE_equation_solution_l1143_114342

theorem equation_solution : ∃! x : ℝ, (567.23 - x) * 45.7 + (64.89 / 11.5)^3 - 2.78 = 18756.120 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1143_114342


namespace NUMINAMATH_CALUDE_function_condition_implies_b_bound_l1143_114314

theorem function_condition_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, 
    Real.exp x * (x - b) + x * (Real.exp x * (x - b + 2)) > 0) →
  b < 8/3 := by
  sorry

end NUMINAMATH_CALUDE_function_condition_implies_b_bound_l1143_114314


namespace NUMINAMATH_CALUDE_remainder_theorem_l1143_114340

theorem remainder_theorem (P Q Q' R R' a b c : ℕ) 
  (h1 : P = a * Q + R) 
  (h2 : Q = (b + c) * Q' + R') : 
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1143_114340


namespace NUMINAMATH_CALUDE_excellent_students_increase_l1143_114392

theorem excellent_students_increase (total_students : ℕ) 
  (first_semester_percent : ℚ) (second_semester_percent : ℚ) :
  total_students = 650 →
  first_semester_percent = 70 / 100 →
  second_semester_percent = 80 / 100 →
  ⌈(second_semester_percent - first_semester_percent) * total_students⌉ = 65 := by
  sorry

end NUMINAMATH_CALUDE_excellent_students_increase_l1143_114392


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1143_114354

/-- Given a cubic equation x√x - 9x + 9√x - 4 = 0 with real nonnegative roots,
    the sum of the squares of its roots is 63. -/
theorem sum_of_squares_of_roots : ∃ (r s t : ℝ),
  (∀ x : ℝ, x ≥ 0 → (x * Real.sqrt x - 9 * x + 9 * Real.sqrt x - 4 = 0 ↔ x = r * r ∨ x = s * s ∨ x = t * t)) →
  r * r + s * s + t * t = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1143_114354


namespace NUMINAMATH_CALUDE_unique_solution_phi_sigma_pow_two_l1143_114357

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Sum of divisors function -/
def sigma : ℕ → ℕ := sorry

/-- The equation φ(σ(2^x)) = 2^x has only one solution in the natural numbers, and that solution is x = 1 -/
theorem unique_solution_phi_sigma_pow_two : 
  ∃! x : ℕ, phi (sigma (2^x)) = 2^x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_phi_sigma_pow_two_l1143_114357


namespace NUMINAMATH_CALUDE_borrowed_amount_proof_l1143_114367

/-- Represents the simple interest calculation for a loan -/
structure LoanInfo where
  principal : ℝ
  rate : ℝ
  time : ℝ
  total_amount : ℝ

/-- Theorem stating that given the loan conditions, the principal amount is 5400 -/
theorem borrowed_amount_proof (loan : LoanInfo) 
  (h1 : loan.rate = 0.06)
  (h2 : loan.time = 9)
  (h3 : loan.total_amount = 8310)
  : loan.principal = 5400 := by
  sorry

#check borrowed_amount_proof

end NUMINAMATH_CALUDE_borrowed_amount_proof_l1143_114367


namespace NUMINAMATH_CALUDE_stationery_costs_l1143_114320

theorem stationery_costs : ∃ (x y z : ℕ+), 
  (x : ℤ) % 2 = 0 ∧
  x + 3*y + 2*z = 98 ∧
  3*x + y = 5*z - 36 ∧
  x = 4 ∧ y = 22 ∧ z = 14 := by
sorry

end NUMINAMATH_CALUDE_stationery_costs_l1143_114320


namespace NUMINAMATH_CALUDE_find_c_l1143_114333

def f (a c x : ℝ) : ℝ := a * x^3 + c

theorem find_c (a c : ℝ) :
  (∃ x, x ∈ Set.Icc 1 2 ∧ ∀ y ∈ Set.Icc 1 2, f a c y ≤ f a c x) →
  (deriv (f a c) 1 = 6) →
  (∃ x, x ∈ Set.Icc 1 2 ∧ f a c x = 20) →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_find_c_l1143_114333


namespace NUMINAMATH_CALUDE_alexey_game_max_score_l1143_114348

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem alexey_game_max_score :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → score x ≤ score 2079 :=
by sorry

end NUMINAMATH_CALUDE_alexey_game_max_score_l1143_114348


namespace NUMINAMATH_CALUDE_street_length_calculation_l1143_114306

/-- Proves that given a speed of 5.31 km/h and a time of 8 minutes, the distance traveled is 708 meters. -/
theorem street_length_calculation (speed : ℝ) (time : ℝ) : 
  speed = 5.31 → time = 8 → speed * time * (1000 / 60) = 708 :=
by sorry

end NUMINAMATH_CALUDE_street_length_calculation_l1143_114306


namespace NUMINAMATH_CALUDE_digit_97_of_1_13_l1143_114390

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => match n % 6 with
    | 0 => 0
    | 1 => 7
    | 2 => 6
    | 3 => 9
    | 4 => 2
    | 5 => 3
    | _ => 0  -- This case should never occur due to % 6

/-- The 97th digit after the decimal point in the decimal representation of 1/13 is 0 -/
theorem digit_97_of_1_13 : decimal_rep_1_13 97 = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_97_of_1_13_l1143_114390
