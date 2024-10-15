import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3996_399612

def M : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (-1, 1) + x • (1, 2)}
def N : Set (ℝ × ℝ) := {a | ∃ x : ℝ, a = (1, -2) + x • (2, 3)}

theorem intersection_of_M_and_N :
  M ∩ N = {(-13, -23)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3996_399612


namespace NUMINAMATH_CALUDE_triangle_perimeter_proof_l3996_399614

theorem triangle_perimeter_proof (a b c : ℝ) (h1 : a = 7) (h2 : b = 10) (h3 : c = 15) :
  a + b + c = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_proof_l3996_399614


namespace NUMINAMATH_CALUDE_proposition_equivalence_l3996_399674

theorem proposition_equivalence (a b c : ℝ) :
  (a ≤ b → a * c^2 ≤ b * c^2) ↔ (a * c^2 > b * c^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3996_399674


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3996_399611

/-- Given a function f(x) = (1/2)x^4 - 2x^3 + 3m where x ∈ ℝ, 
    if f(x) + 12 ≥ 0 for all x, then m ≥ 1/2 -/
theorem function_inequality_implies_m_bound (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 12 ≥ 0) → m ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l3996_399611


namespace NUMINAMATH_CALUDE_markus_bags_l3996_399623

theorem markus_bags (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) (markus_marbles_per_bag : ℕ) (markus_extra_marbles : ℕ) :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_marbles_per_bag = 13 →
  markus_extra_marbles = 2 →
  (mara_bags * mara_marbles_per_bag + markus_extra_marbles) / markus_marbles_per_bag = 2 :=
by sorry

end NUMINAMATH_CALUDE_markus_bags_l3996_399623


namespace NUMINAMATH_CALUDE_sector_area_l3996_399646

/-- Given a circular sector with central angle 2π/3 and chord length 2√3, 
    its area is 4π/3 -/
theorem sector_area (θ : Real) (chord_length : Real) (area : Real) : 
  θ = 2 * Real.pi / 3 →
  chord_length = 2 * Real.sqrt 3 →
  area = 4 * Real.pi / 3 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l3996_399646


namespace NUMINAMATH_CALUDE_valentines_theorem_l3996_399606

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of additional Valentines Mrs. Franklin needs -/
def additional_valentines : ℕ := 16

/-- The number of students Mrs. Franklin has -/
def num_students : ℕ := 74

/-- Theorem stating that the initial number of Valentines plus the additional Valentines
    equals the total number of students -/
theorem valentines_theorem :
  initial_valentines + additional_valentines = num_students :=
by sorry

end NUMINAMATH_CALUDE_valentines_theorem_l3996_399606


namespace NUMINAMATH_CALUDE_correct_average_calculation_l3996_399669

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ incorrect_avg = 21 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n : ℚ) * incorrect_avg + (correct_num - wrong_num) = n * 22 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l3996_399669


namespace NUMINAMATH_CALUDE_fraction_equality_l3996_399642

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x - 2*y) / (2*x - 5*y) = 3) : 
  (2*x + 5*y) / (5*x - 2*y) = 31/63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3996_399642


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l3996_399699

theorem base_n_representation_of_b (n a b : ℕ) : 
  n > 9 → 
  n^2 - a*n + b = 0 → 
  a = 2*n + 1 → 
  b = n^2 + n := by
sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l3996_399699


namespace NUMINAMATH_CALUDE_log_expression_equals_three_halves_l3996_399664

theorem log_expression_equals_three_halves :
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_three_halves_l3996_399664


namespace NUMINAMATH_CALUDE_binary_111011_is_59_l3996_399627

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.zip b (List.range b.length).reverse).foldl
    (fun acc (bit, power) => acc + if bit then 2^power else 0) 0

theorem binary_111011_is_59 :
  binary_to_decimal [true, true, true, false, true, true] = 59 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011_is_59_l3996_399627


namespace NUMINAMATH_CALUDE_cyclic_n_gon_characterization_l3996_399621

/-- A convex n-gon is cyclic if and only if there exist real numbers a_i and b_i
    for each vertex P_i such that for any i < j, the distance P_i P_j = |a_i b_j - a_j b_i|. -/
theorem cyclic_n_gon_characterization {n : ℕ} (P : Fin n → ℝ × ℝ) :
  (∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin n, dist center (P i) = radius) ↔
  (∃ (a b : Fin n → ℝ), ∀ (i j : Fin n), i < j →
    dist (P i) (P j) = |a i * b j - a j * b i|) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_n_gon_characterization_l3996_399621


namespace NUMINAMATH_CALUDE_isosceles_perpendicular_division_l3996_399643

/-- An isosceles triangle with base 32 and legs 20 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_eq : base = 32
  leg_eq : leg = 20

/-- The perpendicular from the apex to the base divides the base into two segments -/
def perpendicular_segments (t : IsoscelesTriangle) : ℝ × ℝ :=
  (7, 25)

/-- Theorem: The perpendicular from the apex of the isosceles triangle
    divides the base into segments of 7 and 25 units -/
theorem isosceles_perpendicular_division (t : IsoscelesTriangle) :
  perpendicular_segments t = (7, 25) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_perpendicular_division_l3996_399643


namespace NUMINAMATH_CALUDE_cattle_milk_production_l3996_399604

theorem cattle_milk_production 
  (total_cattle : ℕ) 
  (male_percentage : ℚ) 
  (female_percentage : ℚ) 
  (num_male_cows : ℕ) 
  (avg_milk_per_cow : ℚ) : 
  male_percentage = 2/5 →
  female_percentage = 3/5 →
  num_male_cows = 50 →
  avg_milk_per_cow = 2 →
  (↑num_male_cows : ℚ) / male_percentage = ↑total_cattle →
  (↑total_cattle * female_percentage * avg_milk_per_cow : ℚ) = 150 := by
  sorry

end NUMINAMATH_CALUDE_cattle_milk_production_l3996_399604


namespace NUMINAMATH_CALUDE_arccos_range_for_sin_l3996_399625

theorem arccos_range_for_sin (a : ℝ) (x : ℝ) (h1 : x = Real.sin a) (h2 : a ∈ Set.Icc (-π/4) (3*π/4)) :
  ∃ y ∈ Set.Icc 0 (3*π/4), y = Real.arccos x :=
sorry

end NUMINAMATH_CALUDE_arccos_range_for_sin_l3996_399625


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3996_399686

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3996_399686


namespace NUMINAMATH_CALUDE_slope_and_intercept_of_3x_plus_2_l3996_399680

/-- Given a linear function y = mx + b, the slope is m and the y-intercept is b -/
def linear_function (m b : ℝ) : ℝ → ℝ := λ x ↦ m * x + b

theorem slope_and_intercept_of_3x_plus_2 :
  ∃ (f : ℝ → ℝ), f = linear_function 3 2 ∧ 
  (∀ x y : ℝ, f x - f y = 3 * (x - y)) ∧
  f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_and_intercept_of_3x_plus_2_l3996_399680


namespace NUMINAMATH_CALUDE_xy_leq_half_sum_squares_l3996_399615

theorem xy_leq_half_sum_squares (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_leq_half_sum_squares_l3996_399615


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l3996_399698

theorem min_sum_reciprocals (x y : ℕ+) (hxy : x ≠ y) (h : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (↑a + ↑b : ℕ) = 64 ∧
    ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → (↑a + ↑b : ℕ) ≤ (↑c + ↑d : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l3996_399698


namespace NUMINAMATH_CALUDE_inheritance_solution_l3996_399636

def inheritance_problem (x : ℝ) : Prop :=
  let federal_tax_rate := 0.25
  let state_tax_rate := 0.15
  let total_tax := 12000
  (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_tax

theorem inheritance_solution :
  ∃ x : ℝ, inheritance_problem x ∧ (round x = 33103) :=
sorry

end NUMINAMATH_CALUDE_inheritance_solution_l3996_399636


namespace NUMINAMATH_CALUDE_least_subtrahend_l3996_399602

theorem least_subtrahend (n : Nat) : 
  (∀ (d : Nat), d ∈ [17, 19, 23] → (997 - n) % d = 3) →
  (∀ (m : Nat), m < n → ∃ (d : Nat), d ∈ [17, 19, 23] ∧ (997 - m) % d ≠ 3) →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_least_subtrahend_l3996_399602


namespace NUMINAMATH_CALUDE_subtraction_result_l3996_399629

theorem subtraction_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3996_399629


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l3996_399609

/-- Represents a repeating decimal -/
def RepeatingDecimal (numerator denominator : ℕ) : ℚ := numerator / denominator

theorem repeating_decimal_sum_diff (a b c : ℚ) :
  a = RepeatingDecimal 6 9 →
  b = RepeatingDecimal 2 9 →
  c = RepeatingDecimal 4 9 →
  a + b - c = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_l3996_399609


namespace NUMINAMATH_CALUDE_exists_term_with_four_zero_digits_l3996_399684

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem exists_term_with_four_zero_digits : 
  ∃ n : ℕ, n < 100000001 ∧ last_four_digits (fibonacci n) = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_term_with_four_zero_digits_l3996_399684


namespace NUMINAMATH_CALUDE_f_min_value_l3996_399687

/-- The function f(x) = |2x-1| + |3x-2| + |4x-3| + |5x-4| -/
def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

/-- Theorem: The minimum value of f(x) is 1 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l3996_399687


namespace NUMINAMATH_CALUDE_same_number_of_heads_probability_p_plus_q_l3996_399689

-- Define the probability of heads for a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the probability of heads for the biased coin
def biased_coin_prob : ℚ := 2/5

-- Define the function to calculate the probability of getting k heads when flipping both coins
def prob_k_heads (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - fair_coin_prob) * (1 - biased_coin_prob)
  | 1 => fair_coin_prob * (1 - biased_coin_prob) + (1 - fair_coin_prob) * biased_coin_prob
  | 2 => fair_coin_prob * biased_coin_prob
  | _ => 0

-- State the theorem
theorem same_number_of_heads_probability :
  (prob_k_heads 0)^2 + (prob_k_heads 1)^2 + (prob_k_heads 2)^2 = 19/50 := by
  sorry

-- Define p and q
def p : ℕ := 19
def q : ℕ := 50

-- State the theorem for p + q
theorem p_plus_q : p + q = 69 := by
  sorry

end NUMINAMATH_CALUDE_same_number_of_heads_probability_p_plus_q_l3996_399689


namespace NUMINAMATH_CALUDE_inequality_proof_l3996_399618

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : 0 < b ∧ b ≤ 1) 
  (hc : 0 < c ∧ c ≤ 1) 
  (hd : 0 < d ∧ d ≤ 1) : 
  1 / (a^2 + b^2 + c^2 + d^2) ≥ 1/4 + (1-a)*(1-b)*(1-c)*(1-d) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3996_399618


namespace NUMINAMATH_CALUDE_family_size_l3996_399633

theorem family_size (planned_spending : ℝ) (orange_cost : ℝ) (savings_percentage : ℝ) :
  planned_spending = 15 →
  orange_cost = 1.5 →
  savings_percentage = 0.4 →
  (planned_spending * savings_percentage) / orange_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_family_size_l3996_399633


namespace NUMINAMATH_CALUDE_b_work_time_l3996_399697

/-- Represents the time taken by A, B, and C to complete the work individually --/
structure WorkTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem --/
def work_conditions (t : WorkTime) : Prop :=
  t.a = 2 * t.b ∧ 
  t.a = 3 * t.c ∧ 
  1 / t.a + 1 / t.b + 1 / t.c = 1 / 6

/-- The theorem stating that B takes 18 days to complete the work alone --/
theorem b_work_time (t : WorkTime) : work_conditions t → t.b = 18 := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l3996_399697


namespace NUMINAMATH_CALUDE_composite_solid_surface_area_l3996_399671

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The volume of a rectangular solid is the product of its length, width, and height. -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The surface area of a rectangular solid. -/
def surfaceAreaRectangular (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

/-- The surface area of a cube. -/
def surfaceAreaCube (s : ℕ) : ℕ := 6 * s * s

theorem composite_solid_surface_area 
  (l w h : ℕ) 
  (prime_l : isPrime l) 
  (prime_w : isPrime w) 
  (prime_h : isPrime h) 
  (vol : volume l w h = 1001) :
  surfaceAreaRectangular l w h + surfaceAreaCube 13 - 13 * 13 = 1467 := by
  sorry

end NUMINAMATH_CALUDE_composite_solid_surface_area_l3996_399671


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l3996_399610

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 5}

-- Define set N
def N : Set Nat := {4, 6}

-- Theorem statement
theorem complement_intersection_equality :
  (U \ M) ∩ N = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l3996_399610


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3996_399695

theorem sqrt_simplification : 3 * Real.sqrt 20 - 5 * Real.sqrt (1/5) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3996_399695


namespace NUMINAMATH_CALUDE_sachins_age_l3996_399691

theorem sachins_age (sachin rahul : ℝ) 
  (h1 : rahul = sachin + 7)
  (h2 : sachin / rahul = 7 / 9) : 
  sachin = 24.5 := by
sorry

end NUMINAMATH_CALUDE_sachins_age_l3996_399691


namespace NUMINAMATH_CALUDE_cube_in_pyramid_l3996_399640

/-- The edge length of a cube inscribed in a regular quadrilateral pyramid -/
theorem cube_in_pyramid (a h : ℝ) (ha : a > 0) (hh : h > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    (a * h) / (a + h * Real.sqrt 2) ≤ x ∧
    x ≤ (a * h) / (a + h) :=
by sorry

end NUMINAMATH_CALUDE_cube_in_pyramid_l3996_399640


namespace NUMINAMATH_CALUDE_triangle_area_and_minimum_ratio_l3996_399655

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition 2sin²A + sin²B = sin²C -/
def satisfiesCondition (t : Triangle) : Prop :=
  2 * (Real.sin t.A)^2 + (Real.sin t.B)^2 = (Real.sin t.C)^2

theorem triangle_area_and_minimum_ratio (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.b = 2 * t.a) 
  (h3 : t.b = 4) :
  -- Part 1: Area of triangle ABC is √15
  (1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 15) ∧
  -- Part 2: Minimum value of c²/(ab) is 2√2, and c/a = 2 at this minimum
  (∀ t' : Triangle, satisfiesCondition t' → 
    t'.c^2 / (t'.a * t'.b) ≥ 2 * Real.sqrt 2) ∧
  (∃ t' : Triangle, satisfiesCondition t' ∧ 
    t'.c^2 / (t'.a * t'.b) = 2 * Real.sqrt 2 ∧ t'.c / t'.a = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_minimum_ratio_l3996_399655


namespace NUMINAMATH_CALUDE_mary_lamb_count_l3996_399670

/-- The number of lambs Mary has after a series of events -/
def final_lamb_count (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
  (lambs_traded : ℕ) (extra_lambs_found : ℕ) : ℕ :=
  initial_lambs + lambs_with_babies * babies_per_lamb - lambs_traded + extra_lambs_found

/-- Theorem stating that Mary ends up with 14 lambs -/
theorem mary_lamb_count : 
  final_lamb_count 6 2 2 3 7 = 14 := by sorry

end NUMINAMATH_CALUDE_mary_lamb_count_l3996_399670


namespace NUMINAMATH_CALUDE_square_fraction_count_l3996_399652

theorem square_fraction_count : 
  ∃! (s : Finset Int), 
    (∀ n ∈ s, ∃ k : Int, (n : ℚ) / (20 - n) = k^2) ∧ 
    (∀ n : Int, n ∉ s → ¬∃ k : Int, (n : ℚ) / (20 - n) = k^2) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3996_399652


namespace NUMINAMATH_CALUDE_knights_in_february_l3996_399616

/-- Represents a city with knights and liars -/
structure City where
  inhabitants : ℕ
  knights_february : ℕ
  claims_february : ℕ
  claims_30th : ℕ

/-- The proposition that a city satisfies the given conditions -/
def satisfies_conditions (c : City) : Prop :=
  c.inhabitants = 366 ∧
  c.claims_february = 100 ∧
  c.claims_30th = 60 ∧
  c.knights_february ≤ 29

/-- The theorem stating that if a city satisfies the conditions, 
    then exactly 29 knights were born in February -/
theorem knights_in_february (c : City) :
  satisfies_conditions c → c.knights_february = 29 := by
  sorry

end NUMINAMATH_CALUDE_knights_in_february_l3996_399616


namespace NUMINAMATH_CALUDE_abs_equal_abs_neg_l3996_399630

theorem abs_equal_abs_neg (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_abs_equal_abs_neg_l3996_399630


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_properties_l3996_399658

noncomputable def tanHalfAngle (θ : Real) : Real := Real.tan (θ / 2)

def isAcute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

def isObtuse (θ : Real) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

theorem triangle_tangent_sum_properties
  (A B C : Real)
  (triangle_angles : A + B + C = Real.pi)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C) :
  let S := (tanHalfAngle A)^2 + (tanHalfAngle B)^2 + (tanHalfAngle C)^2
  let T := tanHalfAngle A + tanHalfAngle B + tanHalfAngle C
  -- Relationship between S and T
  (T^2 = S + 2) →
  -- 1. For acute triangles
  ((isAcute A ∧ isAcute B ∧ isAcute C) → S < 2) ∧
  -- 2. For obtuse triangles with obtuse angle ≥ 2arctan(4/3)
  ((isObtuse A ∨ isObtuse B ∨ isObtuse C) ∧
   (max A (max B C) ≥ 2 * Real.arctan (4/3)) → S ≥ 2) ∧
  -- 3. For obtuse triangles with obtuse angle < 2arctan(4/3)
  ((isObtuse A ∨ isObtuse B ∨ isObtuse C) ∧
   (max A (max B C) < 2 * Real.arctan (4/3)) →
   ∃ (A' B' C' : Real),
     A' + B' + C' = Real.pi ∧
     (isObtuse A' ∨ isObtuse B' ∨ isObtuse C') ∧
     max A' (max B' C') < 2 * Real.arctan (4/3) ∧
     (tanHalfAngle A')^2 + (tanHalfAngle B')^2 + (tanHalfAngle C')^2 > 2 ∧
     ∃ (A'' B'' C'' : Real),
       A'' + B'' + C'' = Real.pi ∧
       (isObtuse A'' ∨ isObtuse B'' ∨ isObtuse C'') ∧
       max A'' (max B'' C'') < 2 * Real.arctan (4/3) ∧
       (tanHalfAngle A'')^2 + (tanHalfAngle B'')^2 + (tanHalfAngle C'')^2 < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_properties_l3996_399658


namespace NUMINAMATH_CALUDE_fourth_root_of_2560000_l3996_399696

theorem fourth_root_of_2560000 : Real.sqrt (Real.sqrt 2560000) = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_2560000_l3996_399696


namespace NUMINAMATH_CALUDE_x_range_theorem_l3996_399666

theorem x_range_theorem (x : ℝ) :
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) →
  x < 1 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l3996_399666


namespace NUMINAMATH_CALUDE_profit_percentage_equality_l3996_399678

/-- Represents the discount percentage as a rational number -/
def discount : ℚ := 5 / 100

/-- Represents the profit percentage with discount as a rational number -/
def profit_with_discount : ℚ := 2255 / 10000

/-- Theorem stating that the profit percentage without discount is equal to the profit percentage with discount -/
theorem profit_percentage_equality :
  profit_with_discount = (profit_with_discount * (1 - discount)⁻¹) := by sorry

end NUMINAMATH_CALUDE_profit_percentage_equality_l3996_399678


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3996_399600

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 > 0) → -2 < k ∧ k < 3 ∧ 
  ∃ k₀ : ℝ, -2 < k₀ ∧ k₀ < 3 ∧ ∃ x : ℝ, x^2 + k₀*x + 1 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3996_399600


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l3996_399657

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |a * x + 1|

-- Theorem 1: Solution set for f(x) ≥ 3 when a = 1
theorem solution_set_f_geq_3 :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem 2: Range of a when solution set for f(x) ≤ 3-x contains [-1, 1]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 3 - x) → a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l3996_399657


namespace NUMINAMATH_CALUDE_line_segment_properties_l3996_399679

/-- Given a line segment with endpoints (1, 4) and (7, 18), prove properties about its midpoint and slope -/
theorem line_segment_properties :
  let x₁ : ℝ := 1
  let y₁ : ℝ := 4
  let x₂ : ℝ := 7
  let y₂ : ℝ := 18
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  let slope : ℝ := (y₂ - y₁) / (x₂ - x₁)
  (midpoint_x + midpoint_y = 15) ∧ (slope = 7 / 3) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_properties_l3996_399679


namespace NUMINAMATH_CALUDE_circumscribed_equal_sides_equal_angles_inscribed_equal_sides_not_always_equal_angles_circumscribed_equal_angles_not_always_equal_sides_inscribed_equal_angles_equal_sides_l3996_399632

/- Define the basic structures -/
structure Polygon :=
  (sides : ℕ)
  (isCircumscribed : Bool)
  (hasEqualSides : Bool)
  (hasEqualAngles : Bool)

/- Define the theorems to be proved -/
theorem circumscribed_equal_sides_equal_angles (p : Polygon) :
  p.isCircumscribed ∧ p.hasEqualSides → p.hasEqualAngles :=
sorry

theorem inscribed_equal_sides_not_always_equal_angles :
  ∃ p : Polygon, ¬p.isCircumscribed ∧ p.hasEqualSides ∧ ¬p.hasEqualAngles :=
sorry

theorem circumscribed_equal_angles_not_always_equal_sides :
  ∃ p : Polygon, p.isCircumscribed ∧ p.hasEqualAngles ∧ ¬p.hasEqualSides :=
sorry

theorem inscribed_equal_angles_equal_sides (p : Polygon) :
  ¬p.isCircumscribed ∧ p.hasEqualAngles → p.hasEqualSides :=
sorry

end NUMINAMATH_CALUDE_circumscribed_equal_sides_equal_angles_inscribed_equal_sides_not_always_equal_angles_circumscribed_equal_angles_not_always_equal_sides_inscribed_equal_angles_equal_sides_l3996_399632


namespace NUMINAMATH_CALUDE_inequality_conditions_l3996_399649

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2*(A*B + B*C + C*A)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_conditions_l3996_399649


namespace NUMINAMATH_CALUDE_total_toys_l3996_399617

theorem total_toys (mike_toys : ℕ) (annie_toys : ℕ) (tom_toys : ℕ) 
  (h1 : mike_toys = 6)
  (h2 : annie_toys = 3 * mike_toys)
  (h3 : tom_toys = annie_toys + 2) :
  mike_toys + annie_toys + tom_toys = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_l3996_399617


namespace NUMINAMATH_CALUDE_andrew_payment_l3996_399644

/-- The total amount Andrew paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1376 for his purchase -/
theorem andrew_payment :
  total_amount 14 54 10 62 = 1376 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l3996_399644


namespace NUMINAMATH_CALUDE_plates_problem_l3996_399626

theorem plates_problem (initial_plates added_plates total_plates : ℕ) 
  (h1 : added_plates = 37)
  (h2 : total_plates = 83)
  (h3 : initial_plates + added_plates = total_plates) :
  initial_plates = 46 := by
  sorry

end NUMINAMATH_CALUDE_plates_problem_l3996_399626


namespace NUMINAMATH_CALUDE_rectangle_reconfiguration_l3996_399663

/-- Given a 10 × 15 rectangle divided into two congruent polygons and reassembled into a new rectangle with length twice its width, the length of one side of the smaller rectangle formed by one of the polygons is 5√3. -/
theorem rectangle_reconfiguration (original_length original_width : ℝ)
  (new_length new_width z : ℝ) :
  original_length = 10 →
  original_width = 15 →
  original_length * original_width = new_length * new_width →
  new_length = 2 * new_width →
  z = new_length / 2 →
  z = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_reconfiguration_l3996_399663


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3996_399688

-- Define the triangle ABC and point D
variable (A B C D : EuclideanPlane)

-- Define the conditions
def on_line_segment (D A C : EuclideanPlane) : Prop := sorry

-- Angle measures in degrees
def angle_measure (p q r : EuclideanPlane) : ℝ := sorry

-- Sum of angles around a point
def angle_sum_around_point (p : EuclideanPlane) : ℝ := sorry

-- Theorem statement
theorem angle_ABC_measure
  (h1 : on_line_segment D A C)
  (h2 : angle_measure A B D = 70)
  (h3 : angle_sum_around_point B = 200)
  (h4 : angle_measure C B D = 60) :
  angle_measure A B C = 70 := by sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3996_399688


namespace NUMINAMATH_CALUDE_regular_pentagon_most_symmetric_l3996_399676

/-- Represents a geometric figure -/
inductive Figure
  | EquilateralTriangle
  | NonSquareRhombus
  | NonSquareRectangle
  | IsoscelesTrapezoid
  | RegularPentagon

/-- Returns the number of lines of symmetry for a given figure -/
def linesOfSymmetry (f : Figure) : ℕ :=
  match f with
  | Figure.EquilateralTriangle => 3
  | Figure.NonSquareRhombus => 2
  | Figure.NonSquareRectangle => 2
  | Figure.IsoscelesTrapezoid => 1
  | Figure.RegularPentagon => 5

/-- Theorem stating that the regular pentagon has the greatest number of lines of symmetry -/
theorem regular_pentagon_most_symmetric :
  ∀ f : Figure, f ≠ Figure.RegularPentagon → linesOfSymmetry Figure.RegularPentagon > linesOfSymmetry f :=
by sorry

end NUMINAMATH_CALUDE_regular_pentagon_most_symmetric_l3996_399676


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3996_399662

-- Define the discount rates and profit margin as real numbers between 0 and 1
def purchase_discount : Real := 0.3
def sale_discount : Real := 0.2
def profit_margin : Real := 0.3

-- Define the list price as an arbitrary positive real number
def list_price : Real := 100

-- Define the purchase price as a function of the list price and purchase discount
def purchase_price (lp : Real) : Real := lp * (1 - purchase_discount)

-- Define the marked price as a function of the list price
def marked_price (lp : Real) : Real := 1.25 * lp

-- Define the selling price as a function of the marked price and sale discount
def selling_price (mp : Real) : Real := mp * (1 - sale_discount)

-- Define the profit as a function of the selling price and purchase price
def profit (sp : Real) (pp : Real) : Real := sp - pp

-- Theorem statement
theorem merchant_pricing_strategy :
  profit (selling_price (marked_price list_price)) (purchase_price list_price) =
  profit_margin * selling_price (marked_price list_price) := by sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3996_399662


namespace NUMINAMATH_CALUDE_discounted_soda_price_l3996_399637

/-- Calculates the price of discounted soda cans -/
theorem discounted_soda_price (regular_price : ℝ) (discount_percent : ℝ) (num_cans : ℕ) : 
  regular_price = 0.30 →
  discount_percent = 15 →
  num_cans = 72 →
  num_cans * (regular_price * (1 - discount_percent / 100)) = 18.36 :=
by sorry

end NUMINAMATH_CALUDE_discounted_soda_price_l3996_399637


namespace NUMINAMATH_CALUDE_correct_selection_methods_l3996_399607

/-- The number of members in the class committee -/
def total_members : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members who cannot serve as the entertainment officer -/
def restricted_members : ℕ := 2

/-- The number of different selection methods -/
def selection_methods : ℕ := 36

/-- Theorem stating that the number of selection methods is correct -/
theorem correct_selection_methods :
  (total_members - restricted_members) * (total_members - 1) * (total_members - 2) = selection_methods :=
by sorry

end NUMINAMATH_CALUDE_correct_selection_methods_l3996_399607


namespace NUMINAMATH_CALUDE_sin_2x_eq_sin_x_solution_l3996_399603

open Set
open Real

def solution_set : Set ℝ := {0, π, -π/3, π/3, 5*π/3}

theorem sin_2x_eq_sin_x_solution :
  {x : ℝ | x ∈ Ioo (-π) (2*π) ∧ sin (2*x) = sin x} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_eq_sin_x_solution_l3996_399603


namespace NUMINAMATH_CALUDE_division_equality_l3996_399622

theorem division_equality : (49 : ℝ) / 0.07 = 700 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l3996_399622


namespace NUMINAMATH_CALUDE_existence_of_special_polynomial_l3996_399620

theorem existence_of_special_polynomial (n : ℕ) (hn : n > 0) :
  ∃ P : Polynomial ℕ,
    (∀ (i : ℕ), Polynomial.coeff P i ∈ ({0, 1} : Set ℕ)) ∧
    (∀ (d : ℕ), d ≥ 2 → P.eval d ≠ 0 ∧ (P.eval d) % n = 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_polynomial_l3996_399620


namespace NUMINAMATH_CALUDE_speed_difference_l3996_399645

/-- Given a truck and a car traveling the same distance, prove the difference in their average speeds -/
theorem speed_difference (distance : ℝ) (truck_time car_time : ℝ) 
  (h1 : distance = 240)
  (h2 : truck_time = 8)
  (h3 : car_time = 5) : 
  (distance / car_time) - (distance / truck_time) = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l3996_399645


namespace NUMINAMATH_CALUDE_draw_specific_sequence_l3996_399694

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the number of cards of each rank (Ace, King, Queen, Jack) -/
def rank_count : ℕ := 4

/-- Represents the number of cards in the hearts suit -/
def hearts_count : ℕ := 13

/-- Calculates the probability of drawing the specified sequence of cards -/
def draw_probability (d : Deck) : ℚ :=
  (rank_count : ℚ) / 52 *
  (rank_count : ℚ) / 51 *
  (rank_count : ℚ) / 50 *
  (rank_count : ℚ) / 49 *
  ((hearts_count - rank_count) : ℚ) / 48

/-- The theorem stating the probability of drawing the specified sequence of cards -/
theorem draw_specific_sequence (d : Deck) :
  draw_probability d = 2304 / 31187500 := by
  sorry

end NUMINAMATH_CALUDE_draw_specific_sequence_l3996_399694


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l3996_399692

theorem dot_product_specific_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-3, 1)
  (a.1 * b.1 + a.2 * b.2) = -1 := by sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l3996_399692


namespace NUMINAMATH_CALUDE_least_divisible_by_240_sixty_cube_divisible_by_240_least_positive_integer_cube_divisible_by_240_l3996_399656

theorem least_divisible_by_240 (a : ℕ) : a > 0 ∧ a^3 % 240 = 0 → a ≥ 60 := by
  sorry

theorem sixty_cube_divisible_by_240 : (60 : ℕ)^3 % 240 = 0 := by
  sorry

theorem least_positive_integer_cube_divisible_by_240 :
  ∃ (a : ℕ), a > 0 ∧ a^3 % 240 = 0 ∧ ∀ (b : ℕ), b > 0 ∧ b^3 % 240 = 0 → b ≥ a :=
by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_240_sixty_cube_divisible_by_240_least_positive_integer_cube_divisible_by_240_l3996_399656


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3996_399683

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 64)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0)
  : (total_players - throwers) * 2 / 3 + throwers = 55 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3996_399683


namespace NUMINAMATH_CALUDE_nine_pouches_sufficient_l3996_399668

-- Define the number of coins and pouches
def totalCoins : ℕ := 60
def numPouches : ℕ := 9

-- Define a type for pouch distributions
def PouchDistribution := List ℕ

-- Function to check if a distribution is valid
def isValidDistribution (d : PouchDistribution) : Prop :=
  d.length = numPouches ∧ d.sum = totalCoins

-- Function to check if a distribution can be equally split among a given number of sailors
def canSplitEqually (d : PouchDistribution) (sailors : ℕ) : Prop :=
  ∃ (groups : List (List ℕ)), 
    groups.length = sailors ∧ 
    (∀ g ∈ groups, g.sum = totalCoins / sailors) ∧
    groups.join.toFinset = d.toFinset

-- The main theorem
theorem nine_pouches_sufficient :
  ∃ (d : PouchDistribution),
    isValidDistribution d ∧
    (∀ sailors ∈ [2, 3, 4, 5], canSplitEqually d sailors) :=
sorry

end NUMINAMATH_CALUDE_nine_pouches_sufficient_l3996_399668


namespace NUMINAMATH_CALUDE_multiply_with_negative_l3996_399667

theorem multiply_with_negative (a : ℝ) : 3 * a * (-2 * a) = -6 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_with_negative_l3996_399667


namespace NUMINAMATH_CALUDE_cubic_increasing_minor_premise_l3996_399650

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the concept of a minor premise in a deduction
def IsMinorPremise (statement : Prop) (conclusion : Prop) : Prop :=
  statement → conclusion

-- Theorem statement
theorem cubic_increasing_minor_premise :
  IsMinorPremise (IsIncreasing f) (IsIncreasing f) :=
sorry

end NUMINAMATH_CALUDE_cubic_increasing_minor_premise_l3996_399650


namespace NUMINAMATH_CALUDE_shelter_dogs_count_l3996_399624

theorem shelter_dogs_count :
  ∀ (dogs cats : ℕ),
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 20) = 15 / 11 →
  dogs = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_shelter_dogs_count_l3996_399624


namespace NUMINAMATH_CALUDE_rectangles_equal_perimeter_different_shape_l3996_399654

/-- Two rectangles with equal perimeters can have different shapes -/
theorem rectangles_equal_perimeter_different_shape :
  ∃ (l₁ w₁ l₂ w₂ : ℝ), 
    l₁ > 0 ∧ w₁ > 0 ∧ l₂ > 0 ∧ w₂ > 0 ∧
    2 * (l₁ + w₁) = 2 * (l₂ + w₂) ∧
    l₁ / w₁ ≠ l₂ / w₂ :=
by sorry

end NUMINAMATH_CALUDE_rectangles_equal_perimeter_different_shape_l3996_399654


namespace NUMINAMATH_CALUDE_grandmothers_current_age_prove_grandmothers_age_l3996_399634

/-- Given Yoojung's current age and her grandmother's future age, calculate the grandmother's current age. -/
theorem grandmothers_current_age (yoojung_current_age : ℕ) (yoojung_future_age : ℕ) (grandmother_future_age : ℕ) : ℕ :=
  grandmother_future_age - (yoojung_future_age - yoojung_current_age)

/-- Prove that given the conditions, the grandmother's current age is 55. -/
theorem prove_grandmothers_age :
  let yoojung_current_age := 5
  let yoojung_future_age := 10
  let grandmother_future_age := 60
  grandmothers_current_age yoojung_current_age yoojung_future_age grandmother_future_age = 55 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_current_age_prove_grandmothers_age_l3996_399634


namespace NUMINAMATH_CALUDE_total_games_won_l3996_399660

def bulls_games : ℕ := 70
def heat_games : ℕ := bulls_games + 5

theorem total_games_won : bulls_games + heat_games = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l3996_399660


namespace NUMINAMATH_CALUDE_wendys_cookies_l3996_399693

theorem wendys_cookies (pastries_left pastries_sold num_cupcakes : ℕ) 
  (h1 : pastries_left = 24)
  (h2 : pastries_sold = 9)
  (h3 : num_cupcakes = 4) : 
  (pastries_left + pastries_sold) - num_cupcakes = 29 := by
  sorry

#check wendys_cookies

end NUMINAMATH_CALUDE_wendys_cookies_l3996_399693


namespace NUMINAMATH_CALUDE_expression_evaluation_l3996_399673

theorem expression_evaluation : 2^3 + 15 * 2 - 4 + 10 * 5 / 2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3996_399673


namespace NUMINAMATH_CALUDE_jimmy_card_distribution_l3996_399641

theorem jimmy_card_distribution (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 18)
  (h2 : remaining_cards = 9) :
  ∃ (cards_to_bob : ℕ), 
    cards_to_bob = 3 ∧ 
    initial_cards = remaining_cards + cards_to_bob + 2 * cards_to_bob :=
by
  sorry

end NUMINAMATH_CALUDE_jimmy_card_distribution_l3996_399641


namespace NUMINAMATH_CALUDE_twelfth_prime_l3996_399677

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem twelfth_prime :
  (nth_prime 7 = 17) → (nth_prime 12 = 37) := by
  sorry

end NUMINAMATH_CALUDE_twelfth_prime_l3996_399677


namespace NUMINAMATH_CALUDE_population_increase_rate_example_l3996_399631

def population_increase_rate (initial_population final_population : ℕ) : ℚ :=
  (final_population - initial_population : ℚ) / initial_population * 100

theorem population_increase_rate_example :
  population_increase_rate 300 330 = 10 := by
sorry

end NUMINAMATH_CALUDE_population_increase_rate_example_l3996_399631


namespace NUMINAMATH_CALUDE_octal_to_decimal_conversion_l3996_399665

-- Define the octal number
def octal_age : ℕ := 536

-- Define the decimal equivalent
def decimal_age : ℕ := 350

-- Theorem to prove the equivalence
theorem octal_to_decimal_conversion :
  (5 * 8^2 + 3 * 8^1 + 6 * 8^0) = decimal_age :=
by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_conversion_l3996_399665


namespace NUMINAMATH_CALUDE_arith_seq_ratio_l3996_399635

/-- Two arithmetic sequences and their properties -/
structure ArithSeqPair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  A : ℕ → ℚ  -- Sum of first n terms of a
  B : ℕ → ℚ  -- Sum of first n terms of b
  sum_ratio : ∀ n, A n / B n = (4 * n + 2 : ℚ) / (5 * n - 5 : ℚ)

/-- Main theorem -/
theorem arith_seq_ratio (seq : ArithSeqPair) : 
  (seq.a 5 + seq.a 13) / (seq.b 5 + seq.b 13) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_arith_seq_ratio_l3996_399635


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3996_399601

theorem sum_of_three_squares (s t : ℝ) : 
  (3 * s + 2 * t = 27) → 
  (2 * s + 3 * t = 23) → 
  (s + 2 * t = 13) → 
  (3 * s = 21) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3996_399601


namespace NUMINAMATH_CALUDE_solution_x_value_l3996_399628

theorem solution_x_value (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_value_l3996_399628


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l3996_399661

def total_lamps : ℕ := 6
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 2
def lamps_turned_on : ℕ := 3

def probability_specific_arrangement : ℚ := 2 / 25

theorem specific_arrangement_probability :
  (Nat.choose total_lamps blue_lamps * Nat.choose total_lamps lamps_turned_on) *
  probability_specific_arrangement =
  (Nat.choose (total_lamps - 2) (blue_lamps - 1) * Nat.choose (total_lamps - 2) (lamps_turned_on - 1)) :=
by sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l3996_399661


namespace NUMINAMATH_CALUDE_younger_person_age_l3996_399685

theorem younger_person_age (elder_age younger_age : ℕ) : 
  elder_age = younger_age + 20 →
  elder_age = 32 →
  elder_age - 7 = 5 * (younger_age - 7) →
  younger_age = 12 := by
sorry

end NUMINAMATH_CALUDE_younger_person_age_l3996_399685


namespace NUMINAMATH_CALUDE_candy_distribution_l3996_399638

theorem candy_distribution (n : ℕ) (total_candy : ℕ) : 
  total_candy = 120 →
  (∃ q : ℕ, total_candy = 2 * n + 2 * q) →
  n = 58 ∨ n = 60 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3996_399638


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l3996_399613

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def lies_on (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : ∀ (x y : ℝ), x = s ∧ y = t → lies_on l₁ x y)
  (h₂ : ∀ (x y : ℝ), x = s ∧ y = t → lies_on l₂ x y) :
  lies_on l₁ s t ∧ lies_on l₂ s t := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersection_l3996_399613


namespace NUMINAMATH_CALUDE_band_tryouts_l3996_399659

theorem band_tryouts (flutes clarinets trumpets pianists : ℕ) : 
  flutes = 20 →
  clarinets = 30 →
  pianists = 20 →
  (80 : ℚ) / 100 * flutes + 1 / 2 * clarinets + 1 / 3 * trumpets + 1 / 10 * pianists = 53 →
  trumpets = 60 :=
by sorry

end NUMINAMATH_CALUDE_band_tryouts_l3996_399659


namespace NUMINAMATH_CALUDE_silent_reading_ratio_l3996_399619

theorem silent_reading_ratio (total : ℕ) (board_games : ℕ) (homework : ℕ) 
  (h1 : total = 24)
  (h2 : board_games = total / 3)
  (h3 : homework = 4)
  : (total - board_games - homework) * 2 = total := by
  sorry

end NUMINAMATH_CALUDE_silent_reading_ratio_l3996_399619


namespace NUMINAMATH_CALUDE_min_value_theorem_l3996_399608

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 3) (hf_reaches_min : ∃ x, f x a b = 3) :
  (a^2 / b + b^2 / a) ≥ 3 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ (a^2 / b + b^2 / a) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3996_399608


namespace NUMINAMATH_CALUDE_blocks_per_box_l3996_399651

def total_blocks : ℕ := 12
def num_boxes : ℕ := 2

theorem blocks_per_box : total_blocks / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_box_l3996_399651


namespace NUMINAMATH_CALUDE_cone_volume_l3996_399639

/-- Given a cone with lateral surface area to base area ratio of 5:3 and height 4, 
    its volume is 12π. -/
theorem cone_volume (r : ℝ) (h : ℝ) (l : ℝ) : 
  h = 4 → l / r = 5 / 3 → l^2 = h^2 + r^2 → (1 / 3) * π * r^2 * h = 12 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3996_399639


namespace NUMINAMATH_CALUDE_percentage_sum_theorem_l3996_399648

theorem percentage_sum_theorem (X Y : ℝ) 
  (hX : 0.45 * X = 270) 
  (hY : 0.35 * Y = 210) : 
  0.75 * X + 0.55 * Y = 780 := by
sorry

end NUMINAMATH_CALUDE_percentage_sum_theorem_l3996_399648


namespace NUMINAMATH_CALUDE_license_plate_count_l3996_399682

def license_plate_combinations : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 5 2) * (Nat.choose 3 2) * 24 * 10 * 9 * 8

theorem license_plate_count : license_plate_combinations = 56016000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3996_399682


namespace NUMINAMATH_CALUDE_corrected_mean_l3996_399605

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 40 →
  incorrect_value = 15 →
  correct_value = 45 →
  let total_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := total_sum + difference
  corrected_sum / n = 40.6 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l3996_399605


namespace NUMINAMATH_CALUDE_triangle_segment_sum_l3996_399647

/-- Given a triangle ABC with vertices A(0,0), B(7,0), and C(3,4), and a line
    passing through (6-2√2, 3-√2) intersecting AC at P and BC at Q,
    if the area of triangle PQC is 14/3, then |CP| + |CQ| = 63. -/
theorem triangle_segment_sum (P Q : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (7, 0)
  let C : ℝ × ℝ := (3, 4)
  let line_point : ℝ × ℝ := (6 - 2 * Real.sqrt 2, 3 - Real.sqrt 2)
  (∃ (t₁ t₂ : ℝ), 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧
    P = (t₁ * C.1 + (1 - t₁) * A.1, t₁ * C.2 + (1 - t₁) * A.2) ∧
    Q = (t₂ * C.1 + (1 - t₂) * B.1, t₂ * C.2 + (1 - t₂) * B.2) ∧
    ∃ (s : ℝ), P = (line_point.1 + s * (Q.1 - line_point.1), 
                    line_point.2 + s * (Q.2 - line_point.2))) →
  (abs (P.1 * Q.2 - P.2 * Q.1 + Q.1 * C.2 - Q.2 * C.1 + C.1 * P.2 - C.2 * P.1) / 2 = 14/3) →
  Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) + Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) = 63 :=
by sorry

end NUMINAMATH_CALUDE_triangle_segment_sum_l3996_399647


namespace NUMINAMATH_CALUDE_sum_integers_minus20_to_10_l3996_399681

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus20_to_10 :
  sum_integers (-20) 10 = -155 := by sorry

end NUMINAMATH_CALUDE_sum_integers_minus20_to_10_l3996_399681


namespace NUMINAMATH_CALUDE_second_number_proof_l3996_399672

theorem second_number_proof (x : ℕ) : x > 1428 ∧ 
  x % 129 = 13 ∧ 
  1428 % 129 = 9 ∧ 
  (∀ y, y > 1428 ∧ y % 129 = 13 → y ≥ x) ∧ 
  x = 1561 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l3996_399672


namespace NUMINAMATH_CALUDE_mangoes_distribution_l3996_399675

theorem mangoes_distribution (total : ℕ) (neighbors : ℕ) 
  (h1 : total = 560) (h2 : neighbors = 8) : 
  (total / 2) / neighbors = 35 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_distribution_l3996_399675


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3996_399690

theorem arithmetic_expression_equality : 36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3996_399690


namespace NUMINAMATH_CALUDE_sum_of_squared_ratios_equals_two_thirds_l3996_399653

theorem sum_of_squared_ratios_equals_two_thirds
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)
  (hx : x₁ + x₂ + x₃ = 0)
  (hy : y₁ + y₂ + y₃ = 0)
  (hxy : x₁*y₁ + x₂*y₂ + x₃*y₃ = 0)
  (hpos : (x₁^2 + x₂^2 + x₃^2) * (y₁^2 + y₂^2 + y₃^2) > 0) :
  x₁^2 / (x₁^2 + x₂^2 + x₃^2) + y₁^2 / (y₁^2 + y₂^2 + y₃^2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_ratios_equals_two_thirds_l3996_399653
