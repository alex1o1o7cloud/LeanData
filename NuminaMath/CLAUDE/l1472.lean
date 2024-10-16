import Mathlib

namespace NUMINAMATH_CALUDE_rotary_club_eggs_l1472_147244

/-- Calculates the total number of eggs needed for the Rotary Club's Omelet Breakfast --/
def total_eggs_needed (small_children : ℕ) (older_children : ℕ) (adults : ℕ) (seniors : ℕ) 
  (waste_percent : ℚ) (extra_omelets : ℕ) (eggs_per_extra_omelet : ℚ) : ℕ :=
  let eggs_for_tickets := small_children + 2 * older_children + 3 * adults + 4 * seniors
  let waste_eggs := ⌈(eggs_for_tickets : ℚ) * waste_percent⌉
  let extra_omelet_eggs := ⌈(extra_omelets : ℚ) * eggs_per_extra_omelet⌉
  eggs_for_tickets + waste_eggs.toNat + extra_omelet_eggs.toNat

/-- Theorem stating the total number of eggs needed for the Rotary Club's Omelet Breakfast --/
theorem rotary_club_eggs : 
  total_eggs_needed 53 35 75 37 (3/100) 25 (5/2) = 574 := by
  sorry

end NUMINAMATH_CALUDE_rotary_club_eggs_l1472_147244


namespace NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_l1472_147241

theorem mashed_potatoes_vs_bacon (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 408) 
  (h2 : bacon = 42) : 
  mashed_potatoes - bacon = 366 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_vs_bacon_l1472_147241


namespace NUMINAMATH_CALUDE_min_value_function_equality_condition_l1472_147213

theorem min_value_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 12*x + y + 64/(x^2 * y) ≥ 81 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 12*x + y + 64/(x^2 * y) = 81 ↔ x = 4 ∧ y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_function_equality_condition_l1472_147213


namespace NUMINAMATH_CALUDE_bottle_t_cost_l1472_147254

/-- The cost of Bottle T given the conditions of the problem -/
theorem bottle_t_cost :
  let bottle_r_capsules : ℕ := 250
  let bottle_r_cost : ℚ := 625 / 100  -- $6.25 represented as a rational number
  let bottle_t_capsules : ℕ := 100
  let cost_per_capsule_diff : ℚ := 5 / 1000  -- $0.005 represented as a rational number
  let bottle_r_cost_per_capsule : ℚ := bottle_r_cost / bottle_r_capsules
  let bottle_t_cost_per_capsule : ℚ := bottle_r_cost_per_capsule - cost_per_capsule_diff
  bottle_t_cost_per_capsule * bottle_t_capsules = 2 := by
sorry

end NUMINAMATH_CALUDE_bottle_t_cost_l1472_147254


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l1472_147249

/-- The number of ways to arrange guests in circles with alternating hat colors -/
def ball_arrangements (N : ℕ) : ℕ := (2 * N).factorial

/-- Theorem stating that the number of valid arrangements is (2N)! -/
theorem ball_arrangements_count (N : ℕ) :
  ball_arrangements N = (2 * N).factorial :=
by sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l1472_147249


namespace NUMINAMATH_CALUDE_seventh_term_is_four_l1472_147252

/-- A geometric sequence with first term 1 and a specific condition on terms 3, 4, and 5 -/
def special_geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, ∃ r : ℝ, ∀ k : ℕ, a (k + 1) = a k * r) ∧  -- geometric sequence condition
  a 1 = 1 ∧                                           -- first term is 1
  a 3 * a 5 = 4 * (a 4 - 1)                           -- given condition

/-- The 7th term of the special geometric sequence is 4 -/
theorem seventh_term_is_four (a : ℕ → ℝ) (h : special_geometric_sequence a) : a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_four_l1472_147252


namespace NUMINAMATH_CALUDE_triangle_side_length_l1472_147262

/-- 
Given a triangle ABC where:
- a, b, c are sides opposite to angles A, B, C respectively
- A = 2π/3
- b = √2
- Area of triangle ABC is √3
Prove that a = √14
-/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * Real.pi / 3 →
  b = Real.sqrt 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a = Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1472_147262


namespace NUMINAMATH_CALUDE_red_papers_count_l1472_147205

theorem red_papers_count (papers_per_box : ℕ) (num_boxes : ℕ) : 
  papers_per_box = 2 → num_boxes = 2 → papers_per_box * num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_papers_count_l1472_147205


namespace NUMINAMATH_CALUDE_power_product_l1472_147282

theorem power_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l1472_147282


namespace NUMINAMATH_CALUDE_johnny_marble_selection_l1472_147245

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of marbles in the collection -/
def total_marbles : ℕ := 10

/-- The number of marbles chosen in the first step -/
def first_choice : ℕ := 4

/-- The number of marbles chosen in the second step -/
def second_choice : ℕ := 2

/-- The theorem stating the total number of ways Johnny can complete the selection process -/
theorem johnny_marble_selection :
  (choose total_marbles first_choice) * (choose first_choice second_choice) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_johnny_marble_selection_l1472_147245


namespace NUMINAMATH_CALUDE_unique_factors_of_2013_l1472_147227

theorem unique_factors_of_2013 (m n : ℕ) (h1 : m < n) (h2 : n < 2 * m) (h3 : m * n = 2013) :
  m = 33 ∧ n = 61 :=
sorry

end NUMINAMATH_CALUDE_unique_factors_of_2013_l1472_147227


namespace NUMINAMATH_CALUDE_find_m_l1472_147284

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

-- State the theorem
theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, f (x - 1) = |x| - |x - 2|) ∧ 
  f (f m) = f 2002 - 7/2 → 
  m = -3/8 := by sorry

end NUMINAMATH_CALUDE_find_m_l1472_147284


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l1472_147243

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 900 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((y - hyperbola_center.2)^2 / (819/36) - (x - hyperbola_center.1)^2 / (819/9) = 1) :=
sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l1472_147243


namespace NUMINAMATH_CALUDE_sine_symmetry_axis_symmetric_angles_sqrt_cos_minus_one_even_l1472_147253

open Real

-- Statement 2
theorem sine_symmetry_axis (k : ℤ) :
  ∀ x : ℝ, sin x = sin (π - x + (k * 2 * π)) := by sorry

-- Statement 3
theorem symmetric_angles (α β : ℝ) (k : ℤ) :
  (∀ x : ℝ, sin (α + x) = sin (β - x)) →
  α + β = (2 * k - 1) * π := by sorry

-- Statement 5
theorem sqrt_cos_minus_one_even :
  ∀ x : ℝ, sqrt (cos x - 1) = sqrt (cos (-x) - 1) := by sorry

end NUMINAMATH_CALUDE_sine_symmetry_axis_symmetric_angles_sqrt_cos_minus_one_even_l1472_147253


namespace NUMINAMATH_CALUDE_correct_calculation_l1472_147251

theorem correct_calculation (x : ℤ) (h : x + 26 = 61) : x + 62 = 97 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1472_147251


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l1472_147276

/-- Proves that arctan(tan 75° - 2 tan 30°) = 75° --/
theorem arctan_tan_difference : 
  Real.arctan (Real.tan (75 * π / 180) - 2 * Real.tan (30 * π / 180)) = 75 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l1472_147276


namespace NUMINAMATH_CALUDE_min_value_of_square_plus_constant_l1472_147275

theorem min_value_of_square_plus_constant (x : ℝ) :
  (x - 1)^2 + 3 ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_square_plus_constant_l1472_147275


namespace NUMINAMATH_CALUDE_wall_thickness_is_5cm_l1472_147269

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in meters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick in cubic centimeters -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the total volume of bricks in cubic centimeters -/
def totalBrickVolume (b : BrickDimensions) (n : ℝ) : ℝ :=
  brickVolume b * n

/-- Calculates the area of the wall's face in square centimeters -/
def wallFaceArea (w : WallDimensions) : ℝ :=
  w.length * w.height * 10000 -- Convert m² to cm²

/-- The main theorem stating the wall thickness -/
theorem wall_thickness_is_5cm 
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℝ) :
  brick.length = 25 ∧ 
  brick.width = 11 ∧ 
  brick.height = 6 ∧
  wall.length = 8 ∧
  wall.height = 1 ∧
  num_bricks = 242.42424242424244 →
  wall.thickness = 5 := by
sorry

end NUMINAMATH_CALUDE_wall_thickness_is_5cm_l1472_147269


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1472_147290

theorem average_marks_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : n₂ = 50) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℝ) = 3800 / 70 :=
sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1472_147290


namespace NUMINAMATH_CALUDE_homework_problem_count_l1472_147286

theorem homework_problem_count (p t : ℕ) (hp : p > 10) (ht : t > 2) : 
  p * t = (2 * p - 6) * (t - 2) → p * t = 96 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l1472_147286


namespace NUMINAMATH_CALUDE_solve_system_for_y_l1472_147214

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - 3 * y = 18) 
  (eq2 : x + 2 * y = 8) : 
  y = -2 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l1472_147214


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l1472_147273

theorem rectangular_hall_area (length width : ℝ) : 
  width = length / 2 →
  length - width = 8 →
  length * width = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l1472_147273


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l1472_147215

/-- Given a rectangular parallelepiped with face areas p, q, and r, its volume is √(pqr) -/
theorem parallelepiped_volume (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (V : ℝ), V > 0 ∧ V * V = p * q * r :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l1472_147215


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_sum_exact_l1472_147288

theorem min_value_sum (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) :
  ∀ x y : ℝ, x > 1 → y > 1 → x * y - (x + y) = 1 → a + b ≤ x + y :=
by sorry

theorem min_value_sum_exact (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) :
  a + b = 2 * (Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_sum_exact_l1472_147288


namespace NUMINAMATH_CALUDE_fraction_relation_l1472_147250

theorem fraction_relation (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c = d + 4) :
  d / a = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l1472_147250


namespace NUMINAMATH_CALUDE_orchard_tree_difference_l1472_147274

theorem orchard_tree_difference (original_trees dead_trees slightly_damaged_trees : ℕ) 
  (h1 : original_trees = 150)
  (h2 : dead_trees = 92)
  (h3 : slightly_damaged_trees = 15) :
  dead_trees - (original_trees - (dead_trees + slightly_damaged_trees)) = 49 :=
by sorry

end NUMINAMATH_CALUDE_orchard_tree_difference_l1472_147274


namespace NUMINAMATH_CALUDE_club_has_25_seniors_l1472_147267

/-- Represents a high school club with juniors and seniors -/
structure Club where
  juniors : ℕ
  seniors : ℕ
  project_juniors : ℕ
  project_seniors : ℕ

/-- The conditions of the problem -/
def club_conditions (c : Club) : Prop :=
  c.juniors + c.seniors = 50 ∧
  c.project_juniors = (40 * c.juniors) / 100 ∧
  c.project_seniors = (20 * c.seniors) / 100 ∧
  c.project_juniors = 2 * c.project_seniors

/-- The theorem stating that a club satisfying the conditions has 25 seniors -/
theorem club_has_25_seniors (c : Club) (h : club_conditions c) : c.seniors = 25 := by
  sorry


end NUMINAMATH_CALUDE_club_has_25_seniors_l1472_147267


namespace NUMINAMATH_CALUDE_bicycle_price_adjustment_l1472_147229

theorem bicycle_price_adjustment (original_price : ℝ) 
  (wednesday_discount : ℝ) (friday_increase : ℝ) (saturday_discount : ℝ) : 
  original_price = 200 →
  wednesday_discount = 0.40 →
  friday_increase = 0.20 →
  saturday_discount = 0.25 →
  original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount) = 108 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_adjustment_l1472_147229


namespace NUMINAMATH_CALUDE_equation_solutions_l1472_147217

theorem equation_solutions :
  -- Equation 1
  (∀ x : ℝ, x^2 - 5*x = 0 ↔ x = 0 ∨ x = 5) ∧
  -- Equation 2
  (∀ x : ℝ, (2*x + 1)^2 = 4 ↔ x = -3/2 ∨ x = 1/2) ∧
  -- Equation 3
  (∀ x : ℝ, x*(x - 1) + 3*(x - 1) = 0 ↔ x = 1 ∨ x = -3) ∧
  -- Equation 4
  (∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l1472_147217


namespace NUMINAMATH_CALUDE_n_must_be_even_l1472_147218

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem n_must_be_even (n : ℕ) 
  (h1 : n > 0)
  (h2 : sum_of_digits n = 2014)
  (h3 : sum_of_digits (5 * n) = 1007) :
  Even n := by
  sorry

end NUMINAMATH_CALUDE_n_must_be_even_l1472_147218


namespace NUMINAMATH_CALUDE_B_power_99_is_identity_l1472_147203

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_99_is_identity :
  B ^ 99 = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_99_is_identity_l1472_147203


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1472_147268

theorem rational_equation_solution : 
  ∀ x : ℝ, x ≠ 2 → 
  ((3 * x - 9) / (x^2 - 6*x + 8) = (x + 1) / (x - 2)) ↔ 
  (x = 1 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1472_147268


namespace NUMINAMATH_CALUDE_frequency_count_calculation_l1472_147259

/-- Given a sample of size 1000 divided into several groups,
    if the frequency of a particular group is 0.4,
    then the frequency count of that group is 400. -/
theorem frequency_count_calculation (sample_size : ℕ) (group_frequency : ℝ) :
  sample_size = 1000 →
  group_frequency = 0.4 →
  (sample_size : ℝ) * group_frequency = 400 := by
  sorry

end NUMINAMATH_CALUDE_frequency_count_calculation_l1472_147259


namespace NUMINAMATH_CALUDE_logarithm_comparison_l1472_147299

theorem logarithm_comparison : 
  (Real.log 3.4 / Real.log 3 < Real.log 8.5 / Real.log 3) ∧ 
  ¬(π^(-0.7) < π^(-0.9)) ∧ 
  ¬(Real.log 1.8 / Real.log 0.3 < Real.log 2.7 / Real.log 0.3) ∧ 
  ¬(0.99^2.7 < 0.99^3.5) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_comparison_l1472_147299


namespace NUMINAMATH_CALUDE_f_maximum_and_a_range_l1472_147231

/-- The function f(x) = |x+1| - |x-4| - a -/
def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem f_maximum_and_a_range :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x a ≤ f x_max a) ∧
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x a ≤ 5 - a) ∧
  (∃ (x : ℝ), f x a ≥ 4/a + 1 → (a = 2 ∨ a < 0)) := by
  sorry

end NUMINAMATH_CALUDE_f_maximum_and_a_range_l1472_147231


namespace NUMINAMATH_CALUDE_prime_sum_problem_l1472_147277

theorem prime_sum_problem (p q r : ℕ) : 
  Prime p → Prime q → Prime r →
  p * q + q * r + r * p = 191 →
  p + q = r - 1 →
  p + q + r = 25 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l1472_147277


namespace NUMINAMATH_CALUDE_power_relationship_l1472_147280

theorem power_relationship (y : ℝ) (h : (10 : ℝ) ^ (4 * y) = 49) : (10 : ℝ) ^ (-2 * y) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_power_relationship_l1472_147280


namespace NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_times_product_l1472_147247

def nonZeroDigits (n : ℕ) : List ℕ :=
  (n.digits 10).filter (· ≠ 0)

def productOfNonZeroDigits (n : ℕ) : ℕ :=
  (nonZeroDigits n).prod

theorem exists_consecutive_numbers_with_54_times_product : 
  ∃ n : ℕ, productOfNonZeroDigits (n + 1) = 54 * productOfNonZeroDigits n := by
  sorry

end NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_times_product_l1472_147247


namespace NUMINAMATH_CALUDE_corral_area_ratio_l1472_147209

/-- The ratio of areas between four small square corrals and one large square corral -/
theorem corral_area_ratio (s : ℝ) (h : s > 0) : 
  (4 * s^2) / ((4 * s)^2) = 1 / 4 := by
  sorry

#check corral_area_ratio

end NUMINAMATH_CALUDE_corral_area_ratio_l1472_147209


namespace NUMINAMATH_CALUDE_bill_difference_l1472_147257

theorem bill_difference (mike_tip joe_tip : ℝ) (mike_percent joe_percent : ℝ) 
  (h1 : mike_tip = 2)
  (h2 : joe_tip = 2)
  (h3 : mike_percent = 0.1)
  (h4 : joe_percent = 0.2)
  (h5 : mike_tip = mike_percent * mike_bill)
  (h6 : joe_tip = joe_percent * joe_bill)
  : mike_bill - joe_bill = 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l1472_147257


namespace NUMINAMATH_CALUDE_product_expansion_l1472_147221

theorem product_expansion (x : ℝ) : (9*x + 2) * (4*x^2 + 3) = 36*x^3 + 8*x^2 + 27*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1472_147221


namespace NUMINAMATH_CALUDE_square_of_binomial_l1472_147279

theorem square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + d = (a*x + b)^2) → d = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1472_147279


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1472_147211

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ :=
  x^3 - a*x^2 - (a+1)*x + a^2 - 2

/-- The condition for the equation to have exactly one real solution -/
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a x = 0

/-- Theorem stating the condition for unique solution -/
theorem unique_solution_condition :
  ∀ a : ℝ, has_unique_solution a ↔ a < 7/4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1472_147211


namespace NUMINAMATH_CALUDE_expected_regions_100_l1472_147255

/-- The number of points on the circle -/
def n : ℕ := 100

/-- The probability that two randomly chosen chords intersect inside the circle -/
def p_intersect : ℚ := 1/3

/-- The expected number of regions bounded by straight lines when n points are picked 
    independently and uniformly at random on a circle, and connected by line segments -/
def expected_regions (n : ℕ) : ℚ :=
  1 + p_intersect * (n.choose 2 - 3 * n)

theorem expected_regions_100 : 
  expected_regions n = 1651 := by sorry

end NUMINAMATH_CALUDE_expected_regions_100_l1472_147255


namespace NUMINAMATH_CALUDE_tangent_length_fq_l1472_147235

-- Define the triangle
structure RightTriangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  right_angle_at_e : de^2 + ef^2 = df^2

-- Define the circle
structure TangentCircle where
  center_on_de : Bool
  tangent_to_df : Bool
  tangent_to_ef : Bool

-- Theorem statement
theorem tangent_length_fq 
  (t : RightTriangle) 
  (c : TangentCircle) 
  (h1 : t.de = 7) 
  (h2 : t.df = Real.sqrt 85) 
  (h3 : c.center_on_de = true) 
  (h4 : c.tangent_to_df = true) 
  (h5 : c.tangent_to_ef = true) : 
  ∃ q : ℝ, q = 6 ∧ q = t.ef := by
  sorry

end NUMINAMATH_CALUDE_tangent_length_fq_l1472_147235


namespace NUMINAMATH_CALUDE_actual_distance_between_towns_l1472_147289

-- Define the map distance between towns
def map_distance : ℝ := 18

-- Define the scale
def scale_inches : ℝ := 0.3
def scale_miles : ℝ := 5

-- Theorem to prove
theorem actual_distance_between_towns :
  (map_distance * scale_miles) / scale_inches = 300 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_between_towns_l1472_147289


namespace NUMINAMATH_CALUDE_sets_A_and_B_solutions_l1472_147272

theorem sets_A_and_B_solutions (A B : Set ℕ) :
  (A ∩ B = {1, 2, 3}) ∧ (A ∪ B = {1, 2, 3, 4, 5}) →
  ((A = {1, 2, 3} ∧ B = {1, 2, 3, 4, 5}) ∨
   (A = {1, 2, 3, 4, 5} ∧ B = {1, 2, 3}) ∨
   (A = {1, 2, 3, 4} ∧ B = {1, 2, 3, 5}) ∨
   (A = {1, 2, 3, 5} ∧ B = {1, 2, 3, 4})) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_and_B_solutions_l1472_147272


namespace NUMINAMATH_CALUDE_taller_tree_height_l1472_147237

theorem taller_tree_height (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 20 →  -- One tree is 20 feet taller than the other
  h₁ / h₂ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₂ = 70 :=  -- The height of the taller tree is 70 feet
by sorry

end NUMINAMATH_CALUDE_taller_tree_height_l1472_147237


namespace NUMINAMATH_CALUDE_triangle_8_6_4_l1472_147248

/-- A triangle can be formed if the sum of any two sides is greater than the third side,
    and the difference between any two sides is less than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a - b < c ∧ b - c < a ∧ c - a < b

/-- Prove that line segments of lengths 8, 6, and 4 can form a triangle. -/
theorem triangle_8_6_4 : can_form_triangle 8 6 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_8_6_4_l1472_147248


namespace NUMINAMATH_CALUDE_linda_cookie_distribution_l1472_147256

/-- Calculates the number of cookies per student given the problem conditions -/
def cookies_per_student (classmates : ℕ) (cookies_per_batch : ℕ) 
  (choc_chip_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) : ℕ :=
  let total_cookies := (choc_chip_batches + oatmeal_batches + additional_batches) * cookies_per_batch
  total_cookies / classmates

/-- Proves that given the problem conditions, each student receives 10 cookies -/
theorem linda_cookie_distribution : 
  cookies_per_student 24 (4 * 12) 2 1 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_linda_cookie_distribution_l1472_147256


namespace NUMINAMATH_CALUDE_factor_polynomial_l1472_147281

theorem factor_polynomial (x : ℝ) : 80 * x^5 - 180 * x^9 = 20 * x^5 * (4 - 9 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1472_147281


namespace NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l1472_147293

theorem square_perimeter_diagonal_ratio (P₁ P₂ d₁ d₂ : ℝ) :
  P₁ > 0 ∧ P₂ > 0 ∧ d₁ > 0 ∧ d₂ > 0 ∧ 
  (P₂ / P₁ = 11) ∧
  (P₁ = 4 * (d₁ / Real.sqrt 2)) ∧
  (P₂ = 4 * (d₂ / Real.sqrt 2)) →
  d₂ / d₁ = 11 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l1472_147293


namespace NUMINAMATH_CALUDE_direct_proportionality_from_equation_l1472_147202

/-- Two real numbers are directly proportional if their ratio is constant -/
def DirectlyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

/-- Given A and B are non-zero real numbers satisfying 3A = 4B, 
    prove that A and B are directly proportional -/
theorem direct_proportionality_from_equation (A B : ℝ) 
    (h1 : 3 * A = 4 * B) (h2 : A ≠ 0) (h3 : B ≠ 0) : 
    DirectlyProportional A B := by
  sorry

end NUMINAMATH_CALUDE_direct_proportionality_from_equation_l1472_147202


namespace NUMINAMATH_CALUDE_alices_number_l1472_147242

theorem alices_number (n : ℕ) : 
  (∃ k : ℕ, n = 180 * k) → 
  (∃ m : ℕ, n = 240 * m) → 
  2000 < n → 
  n < 5000 → 
  n = 2160 ∨ n = 2880 ∨ n = 3600 ∨ n = 4320 := by
sorry

end NUMINAMATH_CALUDE_alices_number_l1472_147242


namespace NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_prime_mod_seven_l1472_147297

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the operation we're performing
def operation (n : ℕ) : ℕ := (n^2)^4

-- Theorem statement
theorem fourth_power_of_square_of_fourth_prime_mod_seven :
  operation fourth_smallest_prime % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_prime_mod_seven_l1472_147297


namespace NUMINAMATH_CALUDE_max_player_salary_460000_l1472_147210

/-- Represents a professional hockey team -/
structure Team where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player in a team -/
def maxPlayerSalary (team : Team) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a player in the given conditions -/
theorem max_player_salary_460000 :
  let team : Team := {
    players := 18,
    minSalary := 20000,
    maxTotalSalary := 800000
  }
  maxPlayerSalary team = 460000 := by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_460000_l1472_147210


namespace NUMINAMATH_CALUDE_sequence_sum_property_l1472_147230

theorem sequence_sum_property (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (k : ℕ+) :
  (∀ n : ℕ+, S n = a n / n) →
  (1 < S k ∧ S k < 9) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l1472_147230


namespace NUMINAMATH_CALUDE_half_power_inequality_l1472_147271

theorem half_power_inequality (x y : ℝ) (h : x > y) : (1/2: ℝ)^x < (1/2 : ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l1472_147271


namespace NUMINAMATH_CALUDE_number_of_pencils_l1472_147270

/-- Given that the ratio of pens to pencils is 5 to 6 and there are 8 more pencils than pens,
    prove that the number of pencils is 48. -/
theorem number_of_pencils (pens pencils : ℕ) 
    (h1 : pens * 6 = pencils * 5)  -- ratio of pens to pencils is 5 to 6
    (h2 : pencils = pens + 8)      -- 8 more pencils than pens
    : pencils = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pencils_l1472_147270


namespace NUMINAMATH_CALUDE_triangle_problem_l1472_147287

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 7 →
  C = π / 3 →
  2 * Real.sin A = 3 * Real.sin B →
  Real.cos B = 3 * Real.sqrt 10 / 10 →
  a = 3 ∧ b = 2 ∧ Real.sin (2 * A) = (3 - 4 * Real.sqrt 3) / 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1472_147287


namespace NUMINAMATH_CALUDE_final_state_theorem_l1472_147246

/-- Represents the state of the cage -/
structure CageState where
  crickets : ℕ
  katydids : ℕ

/-- Represents a magician's transformation -/
inductive Transformation
  | Red
  | Green

/-- Applies a single transformation to the cage state -/
def applyTransformation (state : CageState) (t : Transformation) : CageState :=
  match t with
  | Transformation.Red => 
      { crickets := state.crickets + 1, katydids := state.katydids - 2 }
  | Transformation.Green => 
      { crickets := state.crickets - 5, katydids := state.katydids + 2 }

/-- Applies a sequence of transformations to the cage state -/
def applyTransformations (state : CageState) (ts : List Transformation) : CageState :=
  match ts with
  | [] => state
  | t::rest => applyTransformations (applyTransformation state t) rest

theorem final_state_theorem (transformations : List Transformation) :
  transformations.length = 15 →
  (applyTransformations { crickets := 21, katydids := 30 } transformations).crickets = 0 →
  (applyTransformations { crickets := 21, katydids := 30 } transformations).katydids = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_final_state_theorem_l1472_147246


namespace NUMINAMATH_CALUDE_radio_price_reduction_l1472_147216

theorem radio_price_reduction (x : ℝ) :
  (∀ (P Q : ℝ), P > 0 ∧ Q > 0 →
    P * (1 - x / 100) * (Q * 1.8) = P * Q * 1.44) →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_radio_price_reduction_l1472_147216


namespace NUMINAMATH_CALUDE_savings_account_percentage_l1472_147219

theorem savings_account_percentage (initial_amount : ℝ) (P : ℝ) : 
  initial_amount > 0 →
  (initial_amount + initial_amount * P / 100) * 0.8 = initial_amount →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_savings_account_percentage_l1472_147219


namespace NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l1472_147204

theorem greatest_odd_integer_below_sqrt_50 :
  ∀ x : ℕ, x % 2 = 1 → x^2 < 50 → x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_odd_integer_below_sqrt_50_l1472_147204


namespace NUMINAMATH_CALUDE_div_mul_sqrt_three_reciprocal_result_equals_one_l1472_147294

theorem div_mul_sqrt_three_reciprocal (x : ℝ) (h : x > 0) : 3 / Real.sqrt x * (1 / Real.sqrt x) = 3 / x :=
by sorry

theorem result_equals_one : 3 / Real.sqrt 3 * (1 / Real.sqrt 3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_div_mul_sqrt_three_reciprocal_result_equals_one_l1472_147294


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l1472_147223

open Real

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 20° + sin θ is 40°. -/
theorem least_positive_angle_theorem : 
  (∀ θ : ℝ, 0 < θ ∧ θ < 40 → cos (10 * π / 180) ≠ sin (20 * π / 180) + sin (θ * π / 180)) ∧
  cos (10 * π / 180) = sin (20 * π / 180) + sin (40 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l1472_147223


namespace NUMINAMATH_CALUDE_water_in_first_tank_l1472_147291

theorem water_in_first_tank (capacity : ℝ) (water_second : ℝ) (fill_percentage : ℝ) (additional_water : ℝ) :
  capacity > 0 →
  water_second = 450 →
  fill_percentage = 0.45 →
  water_second = fill_percentage * capacity →
  additional_water = 1250 →
  additional_water + water_second + (capacity - water_second) = 2 * capacity →
  capacity - (additional_water + water_second) = 300 :=
by sorry

end NUMINAMATH_CALUDE_water_in_first_tank_l1472_147291


namespace NUMINAMATH_CALUDE_base_10_to_7_2023_l1472_147238

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  sorry

/-- Converts a list of digits in base 7 to a natural number --/
def fromBase7 (digits : List Nat) : Nat :=
  sorry

theorem base_10_to_7_2023 :
  toBase7 2023 = [5, 6, 2, 0] ∧ fromBase7 [5, 6, 2, 0] = 2023 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_7_2023_l1472_147238


namespace NUMINAMATH_CALUDE_shea_corn_purchase_l1472_147264

/-- The cost of corn per pound in cents -/
def corn_cost : ℕ := 110

/-- The cost of beans per pound in cents -/
def bean_cost : ℕ := 50

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℕ := 30

/-- The total cost in cents -/
def total_cost : ℕ := 2100

/-- The number of pounds of corn bought -/
def corn_pounds : ℚ := 10

theorem shea_corn_purchase :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_pounds ∧
    bean_cost * bean_pounds + corn_cost * corn_pounds = total_cost :=
by sorry

end NUMINAMATH_CALUDE_shea_corn_purchase_l1472_147264


namespace NUMINAMATH_CALUDE_largest_additional_plates_is_largest_possible_l1472_147212

/-- Represents the number of choices in each section of a license plate --/
structure LicensePlateChoices where
  section1 : Nat
  section2 : Nat
  section3 : Nat

/-- Calculates the total number of possible license plates --/
def totalPlates (choices : LicensePlateChoices) : Nat :=
  choices.section1 * choices.section2 * choices.section3

/-- The initial number of choices for each section --/
def initialChoices : LicensePlateChoices :=
  { section1 := 5, section2 := 3, section3 := 3 }

/-- The optimal distribution of new letters --/
def optimalChoices : LicensePlateChoices :=
  { section1 := 5, section2 := 5, section3 := 4 }

/-- Theorem stating that the largest number of additional plates is 55 --/
theorem largest_additional_plates :
  totalPlates optimalChoices - totalPlates initialChoices = 55 := by
  sorry

/-- Theorem stating that this is indeed the largest possible number --/
theorem is_largest_possible (newChoices : LicensePlateChoices)
  (h1 : newChoices.section1 + newChoices.section2 + newChoices.section3 = 
        initialChoices.section1 + initialChoices.section2 + initialChoices.section3 + 3) :
  totalPlates newChoices - totalPlates initialChoices ≤ 55 := by
  sorry

end NUMINAMATH_CALUDE_largest_additional_plates_is_largest_possible_l1472_147212


namespace NUMINAMATH_CALUDE_average_difference_l1472_147224

theorem average_difference (x : ℝ) : 
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1472_147224


namespace NUMINAMATH_CALUDE_square_area_given_edge_expressions_l1472_147292

theorem square_area_given_edge_expressions (x : ℚ) :
  (5 * x - 20 : ℚ) = (30 - 4 * x : ℚ) →
  (5 * x - 20 : ℚ) > 0 →
  (5 * x - 20 : ℚ)^2 = 4900 / 81 := by
  sorry

end NUMINAMATH_CALUDE_square_area_given_edge_expressions_l1472_147292


namespace NUMINAMATH_CALUDE_notebook_cost_l1472_147222

/-- The total cost of notebooks with given prices and quantities -/
def total_cost (green_price : ℕ) (green_quantity : ℕ) (black_price : ℕ) (pink_price : ℕ) : ℕ :=
  green_price * green_quantity + black_price + pink_price

/-- Theorem: The total cost of 4 notebooks (2 green at $10 each, 1 black at $15, and 1 pink at $10) is $45 -/
theorem notebook_cost : total_cost 10 2 15 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1472_147222


namespace NUMINAMATH_CALUDE_visible_black_area_ratio_l1472_147266

/-- Represents the area of a circle -/
structure CircleArea where
  area : ℝ
  area_pos : area > 0

/-- Represents the configuration of three circles -/
structure CircleConfiguration where
  black : CircleArea
  grey : CircleArea
  white : CircleArea
  initial_visible_black : ℝ
  final_visible_black : ℝ
  initial_condition : initial_visible_black = 7 * white.area
  final_condition : final_visible_black = initial_visible_black - white.area

/-- The theorem stating the ratio of visible black areas before and after rearrangement -/
theorem visible_black_area_ratio (config : CircleConfiguration) :
  config.initial_visible_black / config.final_visible_black = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_visible_black_area_ratio_l1472_147266


namespace NUMINAMATH_CALUDE_f_is_even_l1472_147278

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^2)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x : ℝ, f g (-x) = f g x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l1472_147278


namespace NUMINAMATH_CALUDE_inequality_proof_l1472_147200

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (bound_x : x ≥ (1:ℝ)/2) (bound_y : y ≥ (1:ℝ)/2) (bound_z : z ≥ (1:ℝ)/2)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  (1/x + 1/y - 1/z) * (1/x - 1/y + 1/z) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1472_147200


namespace NUMINAMATH_CALUDE_johns_children_l1472_147233

theorem johns_children (john_notebooks : ℕ → ℕ) (wife_notebooks : ℕ → ℕ) (total_notebooks : ℕ) :
  (∀ c : ℕ, john_notebooks c = 2 * c) →
  (∀ c : ℕ, wife_notebooks c = 5 * c) →
  (∃ c : ℕ, john_notebooks c + wife_notebooks c = total_notebooks) →
  total_notebooks = 21 →
  ∃ c : ℕ, c = 3 ∧ john_notebooks c + wife_notebooks c = total_notebooks :=
by sorry

end NUMINAMATH_CALUDE_johns_children_l1472_147233


namespace NUMINAMATH_CALUDE_exists_large_ratio_l1472_147239

def sequence_property (a b : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0 ∧ b n > 0) ∧
  (∀ n : ℕ+, a (n + 1) * b (n + 1) = a n ^ 2 + b n ^ 2) ∧
  (∀ n : ℕ+, a (n + 1) + b (n + 1) = a n * b n) ∧
  (∀ n : ℕ+, a n ≥ b n)

theorem exists_large_ratio (a b : ℕ+ → ℝ) (h : sequence_property a b) :
  ∃ n : ℕ+, a n / b n > 2023^2023 := by
  sorry

end NUMINAMATH_CALUDE_exists_large_ratio_l1472_147239


namespace NUMINAMATH_CALUDE_at_least_one_divisible_by_three_l1472_147228

theorem at_least_one_divisible_by_three (a b : ℤ) : 
  (3 ∣ a) ∨ (3 ∣ b) ∨ (3 ∣ (a + b)) ∨ (3 ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_divisible_by_three_l1472_147228


namespace NUMINAMATH_CALUDE_akeno_spent_more_l1472_147283

def akeno_expenditure : ℕ := 2985
def lev_expenditure : ℕ := akeno_expenditure / 3
def ambrocio_expenditure : ℕ := lev_expenditure - 177

theorem akeno_spent_more :
  akeno_expenditure - (lev_expenditure + ambrocio_expenditure) = 1172 := by
  sorry

end NUMINAMATH_CALUDE_akeno_spent_more_l1472_147283


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l1472_147265

def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2*x, 5],
    ![4*x, 9]]

theorem matrix_not_invertible_iff (x : ℝ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l1472_147265


namespace NUMINAMATH_CALUDE_modified_baseball_league_games_l1472_147236

/-- The total number of games played in a modified baseball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team,
    the total number of games played is 180 -/
theorem modified_baseball_league_games :
  total_games 10 4 = 180 := by
  sorry

#eval total_games 10 4

end NUMINAMATH_CALUDE_modified_baseball_league_games_l1472_147236


namespace NUMINAMATH_CALUDE_reciprocals_multiply_to_one_no_real_roots_when_m_greater_than_one_l1472_147208

-- Definition of reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Statement 1
theorem reciprocals_multiply_to_one (x y : ℝ) :
  are_reciprocals x y → x * y = 1 :=
sorry

-- Definition for real roots
def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c = 0

-- Statement 2
theorem no_real_roots_when_m_greater_than_one (m : ℝ) :
  m > 1 → ¬(has_real_roots 1 (-2) m) :=
sorry

end NUMINAMATH_CALUDE_reciprocals_multiply_to_one_no_real_roots_when_m_greater_than_one_l1472_147208


namespace NUMINAMATH_CALUDE_max_third_term_geometric_progression_l1472_147207

/-- 
Given an arithmetic progression of three terms starting with 5, 
where adding 5 to the second term and 30 to the third term results in a geometric progression,
the maximum possible value of the third term of the resulting geometric progression is 45.
-/
theorem max_third_term_geometric_progression (a b c : ℝ) : 
  (a = 5) ∧ 
  (∃ d : ℝ, b = a + d ∧ c = b + d) ∧ 
  (∃ r : ℝ, (5 : ℝ) * r = (b + 5) ∧ (b + 5) * r = (c + 30)) →
  (c + 30 ≤ 45) :=
by sorry

end NUMINAMATH_CALUDE_max_third_term_geometric_progression_l1472_147207


namespace NUMINAMATH_CALUDE_tangent_line_d_value_l1472_147260

-- Define the line equation
def line (x y d : ℝ) : Prop := y = 3 * x + d

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the tangency condition
def is_tangent (d : ℝ) : Prop :=
  ∃ x y : ℝ, line x y d ∧ parabola x y ∧
  ∀ x' y' : ℝ, line x' y' d → parabola x' y' → (x', y') = (x, y)

-- Theorem statement
theorem tangent_line_d_value :
  ∃ d : ℝ, is_tangent d ∧ d = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_d_value_l1472_147260


namespace NUMINAMATH_CALUDE_max_min_difference_l1472_147234

theorem max_min_difference (a b : ℝ) (h : a^2 + a*b + b^2 = 6) :
  let f := fun (x y : ℝ) => x^2 - x*y + y^2
  ∃ M m : ℝ, (∀ x y : ℝ, x^2 + x*y + y^2 = 6 → f x y ≤ M ∧ m ≤ f x y) ∧ M - m = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_l1472_147234


namespace NUMINAMATH_CALUDE_octagon_semicircles_area_l1472_147263

/-- The area of the region inside a regular octagon with side length 3 and eight inscribed semicircles --/
theorem octagon_semicircles_area : 
  let s : Real := 3  -- side length of the octagon
  let r : Real := s / 2  -- radius of each semicircle
  let octagon_area : Real := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : Real := π * r^2 / 2
  let total_semicircle_area : Real := 8 * semicircle_area
  octagon_area - total_semicircle_area = 18 * (1 + Real.sqrt 2) - 9 * π := by
sorry

end NUMINAMATH_CALUDE_octagon_semicircles_area_l1472_147263


namespace NUMINAMATH_CALUDE_health_codes_suitable_for_comprehensive_survey_other_options_not_suitable_for_comprehensive_survey_l1472_147296

/-- Represents a survey option --/
inductive SurveyOption
  | MovieViewing
  | SeedGermination
  | WaterQuality
  | HealthCodes

/-- Determines if a survey option is suitable for a comprehensive survey --/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.HealthCodes => true
  | _ => false

/-- Theorem stating that the health codes survey is suitable for a comprehensive survey --/
theorem health_codes_suitable_for_comprehensive_survey :
  isSuitableForComprehensiveSurvey SurveyOption.HealthCodes = true :=
sorry

/-- Theorem stating that other survey options are not suitable for a comprehensive survey --/
theorem other_options_not_suitable_for_comprehensive_survey (option : SurveyOption) :
  option ≠ SurveyOption.HealthCodes →
  isSuitableForComprehensiveSurvey option = false :=
sorry

end NUMINAMATH_CALUDE_health_codes_suitable_for_comprehensive_survey_other_options_not_suitable_for_comprehensive_survey_l1472_147296


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1472_147201

theorem binomial_coefficient_problem (h1 : Nat.choose 20 12 = 125970)
                                     (h2 : Nat.choose 18 12 = 18564)
                                     (h3 : Nat.choose 19 12 = 50388) :
  Nat.choose 20 13 = 125970 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1472_147201


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1472_147206

def f (x : ℝ) := x^2 + 2*x - 1

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 3 → f x ≤ f y) ∧
  f x = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1472_147206


namespace NUMINAMATH_CALUDE_smallest_x_value_l1472_147225

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (254 + x)) : 
  x ≥ 2 ∧ ∃ (y' : ℕ+), (3 : ℚ) / 4 = y' / (254 + 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1472_147225


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1472_147258

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 2 * a 6 + 2 * a 4 * a 5 + a 1 * a 9 = 25 →
  a 4 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1472_147258


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1472_147261

/-- Given a quadratic equation x² + bx + c = 0 with one non-zero real root c,
    prove that b + c = -1 -/
theorem quadratic_root_sum (b c : ℝ) (h : c ≠ 0) 
  (h_root : c^2 + b*c + c = 0) : b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1472_147261


namespace NUMINAMATH_CALUDE_only_set_b_is_right_triangle_l1472_147298

-- Define the sets of numbers
def set_a : List ℕ := [2, 3, 4]
def set_b : List ℕ := [3, 4, 5]
def set_c : List ℕ := [5, 6, 7]
def set_d : List ℕ := [7, 8, 9]

-- Define a function to check if a set of three numbers satisfies the Pythagorean theorem
def is_right_triangle (sides : List ℕ) : Prop :=
  match sides with
  | [a, b, c] => a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2
  | _ => False

-- Theorem statement
theorem only_set_b_is_right_triangle :
  ¬(is_right_triangle set_a) ∧
  (is_right_triangle set_b) ∧
  ¬(is_right_triangle set_c) ∧
  ¬(is_right_triangle set_d) :=
by sorry

end NUMINAMATH_CALUDE_only_set_b_is_right_triangle_l1472_147298


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l1472_147285

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ

/-- Represents the equation of a line in the form y = mx + b -/
structure LineEquation where
  m : ℝ
  b : ℝ

/-- The other asymptote of a hyperbola given one asymptote and the x-coordinate of the foci -/
def other_asymptote (h : Hyperbola) : LineEquation :=
  { m := -2, b := -16 }

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = -4) : 
  other_asymptote h = { m := -2, b := -16 } := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l1472_147285


namespace NUMINAMATH_CALUDE_chairs_built_in_ten_days_l1472_147226

/-- Represents the time it takes to build different types of chairs -/
structure ChairTimes where
  rocking : ℕ
  dining : ℕ
  armchair : ℕ

/-- Represents the number of chairs built -/
structure ChairsBuilt where
  rocking : ℕ
  dining : ℕ
  armchair : ℕ

/-- Calculates the maximum number of chairs that can be built in a given number of days -/
def maxChairsBuilt (shiftLength : ℕ) (times : ChairTimes) (days : ℕ) : ChairsBuilt :=
  sorry

/-- Theorem stating the maximum number of chairs that can be built in 10 days -/
theorem chairs_built_in_ten_days :
  let times : ChairTimes := ⟨5, 3, 6⟩
  let result : ChairsBuilt := maxChairsBuilt 8 times 10
  result.rocking = 10 ∧ result.dining = 10 ∧ result.armchair = 0 := by
  sorry

end NUMINAMATH_CALUDE_chairs_built_in_ten_days_l1472_147226


namespace NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l1472_147240

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 4 = 0
def circle2 (x y : ℝ) : Prop := 3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0

-- Theorem for the first circle
theorem circle1_properties :
  ∃ (h k r : ℝ), 
    (h = -1 ∧ k = -2 ∧ r = 3) ∧
    ∀ (x y : ℝ), circle1 x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  ∃ (h k r : ℝ), 
    (h = -1 ∧ k = -1/2 ∧ r = 5/2) ∧
    ∀ (x y : ℝ), circle2 x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle1_properties_circle2_properties_l1472_147240


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1472_147220

theorem trigonometric_expression_equals_one (α : Real) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1472_147220


namespace NUMINAMATH_CALUDE_function_evaluation_l1472_147295

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 for any real number a. -/
theorem function_evaluation (a : ℝ) : (fun x : ℝ => x^2 + 1) (a + 1) = a^2 + 2*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l1472_147295


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1472_147232

theorem imaginary_part_of_z (z : ℂ) : z = (2 + Complex.I) / Complex.I → Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1472_147232
