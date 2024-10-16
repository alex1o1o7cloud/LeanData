import Mathlib

namespace NUMINAMATH_CALUDE_basketball_court_width_l533_53371

theorem basketball_court_width (perimeter : ℝ) (length_diff : ℝ) : perimeter = 96 ∧ length_diff = 14 → 
  ∃ width : ℝ, width = 17 ∧ 2 * (width + length_diff) + 2 * width = perimeter := by
  sorry

end NUMINAMATH_CALUDE_basketball_court_width_l533_53371


namespace NUMINAMATH_CALUDE_least_cars_serviced_per_day_l533_53390

/-- The number of cars that can be serviced in a workday by two mechanics -/
def cars_serviced_per_day (hours_per_day : ℕ) (rate1 : ℕ) (rate2 : ℕ) : ℕ :=
  (rate1 + rate2) * hours_per_day

/-- Theorem stating the least number of cars that can be serviced by Paul and Jack in a workday -/
theorem least_cars_serviced_per_day :
  cars_serviced_per_day 8 2 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_least_cars_serviced_per_day_l533_53390


namespace NUMINAMATH_CALUDE_min_value_and_points_l533_53325

theorem min_value_and_points (x y : ℝ) :
  (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 ≥ 1/6 ∧
  (∃ x y : ℝ, (y - 1)^2 + (x + y - 3)^2 + (2*x + y - 6)^2 = 1/6 ∧ 
   x = 5/2 ∧ y = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_points_l533_53325


namespace NUMINAMATH_CALUDE_toys_per_box_l533_53356

/-- Given that Paul filled up four boxes and packed a total of 32 toys,
    prove that the number of toys in each box is 8. -/
theorem toys_per_box (total_toys : ℕ) (num_boxes : ℕ) (h1 : total_toys = 32) (h2 : num_boxes = 4) :
  total_toys / num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_box_l533_53356


namespace NUMINAMATH_CALUDE_three_digit_number_eleven_times_sum_of_digits_l533_53346

theorem three_digit_number_eleven_times_sum_of_digits :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = 11 * (n / 100 + (n / 10) % 10 + n % 10) :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_eleven_times_sum_of_digits_l533_53346


namespace NUMINAMATH_CALUDE_claires_calculation_l533_53357

theorem claires_calculation (a b c d f : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  (a + b - c + d - f = a + (b - (c * (d - f)))) →
  f = 21/5 := by sorry

end NUMINAMATH_CALUDE_claires_calculation_l533_53357


namespace NUMINAMATH_CALUDE_basketball_shot_minimum_l533_53302

theorem basketball_shot_minimum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < 1) (hbb : b < 1) 
  (h_expected : 3 * a + 2 * b = 2) : 
  (2 / a + 1 / (3 * b)) ≥ 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_basketball_shot_minimum_l533_53302


namespace NUMINAMATH_CALUDE_cubic_expansion_equality_l533_53397

theorem cubic_expansion_equality : 27^3 + 9*(27^2) + 27*(9^2) + 9^3 = (27 + 9)^3 := by sorry

end NUMINAMATH_CALUDE_cubic_expansion_equality_l533_53397


namespace NUMINAMATH_CALUDE_ln_inequality_l533_53392

theorem ln_inequality (x : ℝ) (h : x > 1) : 2 * Real.log x < x - 1 / x := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_l533_53392


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l533_53311

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l533_53311


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l533_53398

/-- A geometric sequence with common ratio q where the first, third, and second terms form an arithmetic sequence has q = 1 or q = -1 -/
theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  (a 3 - a 2 = a 2 - a 1) →    -- arithmetic sequence condition
  (q = 1 ∨ q = -1) := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l533_53398


namespace NUMINAMATH_CALUDE_vector_angle_cosine_l533_53382

theorem vector_angle_cosine (a b : ℝ × ℝ) :
  a + b = (2, -8) →
  a - b = (-8, 16) →
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = -63/65 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_cosine_l533_53382


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_x_squared_two_l533_53315

/-- Two vectors in R^2 are parallel if and only if their cross product is zero -/
axiom parallel_iff_cross_product_zero {a b : ℝ × ℝ} :
  (∃ k : ℝ, a = k • b) ↔ a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b in R^2, if they are parallel, then x^2 = 2 -/
theorem parallel_vectors_implies_x_squared_two (x : ℝ) :
  let a : ℝ × ℝ := (x + 2, 1 + x)
  let b : ℝ × ℝ := (x - 2, 1 - x)
  (∃ k : ℝ, a = k • b) → x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_x_squared_two_l533_53315


namespace NUMINAMATH_CALUDE_peppers_weight_l533_53376

theorem peppers_weight (total_weight green_weight : Float) 
  (h1 : total_weight = 0.6666666666666666)
  (h2 : green_weight = 0.3333333333333333) :
  total_weight - green_weight = 0.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_peppers_weight_l533_53376


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l533_53368

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = -1 → x^2 - 5*x - 6 = 0) ∧ 
  ¬(x^2 - 5*x - 6 = 0 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l533_53368


namespace NUMINAMATH_CALUDE_lcm_1364_884_minus_100_l533_53389

def lcm_minus_100 (a b : Nat) : Nat :=
  Nat.lcm a b - 100

theorem lcm_1364_884_minus_100 :
  lcm_minus_100 1364 884 = 1509692 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1364_884_minus_100_l533_53389


namespace NUMINAMATH_CALUDE_sperners_lemma_l533_53348

theorem sperners_lemma (n : ℕ) (A : Finset (Finset ℕ)) :
  (∀ (i j : Finset ℕ), i ∈ A → j ∈ A → i ≠ j → (¬ i ⊆ j ∧ ¬ j ⊆ i)) →
  (∀ i ∈ A, i ⊆ Finset.range n) →
  A.card ≤ Nat.choose n (n / 2) := by
  sorry

end NUMINAMATH_CALUDE_sperners_lemma_l533_53348


namespace NUMINAMATH_CALUDE_rachel_solved_sixteen_at_lunch_l533_53391

/-- Represents the number of math problems Rachel solved. -/
structure RachelsMathProblems where
  problems_per_minute : ℕ
  minutes_before_bed : ℕ
  total_problems : ℕ

/-- Calculates the number of math problems Rachel solved at lunch. -/
def problems_solved_at_lunch (r : RachelsMathProblems) : ℕ :=
  r.total_problems - (r.problems_per_minute * r.minutes_before_bed)

/-- Theorem stating that Rachel solved 16 math problems at lunch. -/
theorem rachel_solved_sixteen_at_lunch :
  let r : RachelsMathProblems := ⟨5, 12, 76⟩
  problems_solved_at_lunch r = 16 := by sorry

end NUMINAMATH_CALUDE_rachel_solved_sixteen_at_lunch_l533_53391


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l533_53358

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) :
  Real.cos (2*α + π/3) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l533_53358


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l533_53359

/-- The function g(x) -/
def g (a b x : ℝ) : ℝ := (a * x - 2) * (x + b)

/-- The theorem stating that if g(x) > 0 has solution set (-1, 2), then a + b = -4 -/
theorem sum_of_a_and_b (a b : ℝ) :
  (∀ x, g a b x > 0 ↔ -1 < x ∧ x < 2) →
  a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l533_53359


namespace NUMINAMATH_CALUDE_arrangements_theorem_l533_53386

/-- The number of arrangements of 5 people in a row with exactly 1 person between A and B -/
def arrangements_count : ℕ := 36

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

theorem arrangements_theorem :
  ∀ (n : ℕ) (k : ℕ),
  n = total_people →
  k = people_between →
  arrangements_count = 36 :=
sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l533_53386


namespace NUMINAMATH_CALUDE_cone_surface_area_l533_53310

theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 2 →
  base_circumference = 2 * Real.pi →
  π * (base_circumference / (2 * π)) * (base_circumference / (2 * π) + slant_height) = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l533_53310


namespace NUMINAMATH_CALUDE_set_union_problem_l533_53364

theorem set_union_problem (M N : Set ℕ) (a : ℕ) :
  M = {a, 0} ∧ N = {1, 2} ∧ M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l533_53364


namespace NUMINAMATH_CALUDE_equation_solution_l533_53300

theorem equation_solution : 
  let S : Set ℝ := {x | 3 * x * (x - 2) = 2 * (x - 2)}
  S = {2/3, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l533_53300


namespace NUMINAMATH_CALUDE_tenth_replacement_in_january_l533_53303

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Calculates the month after a given number of months have passed -/
def monthAfter (start : Month) (months : ℕ) : Month := sorry

/-- The number of months between battery replacements -/
def replacementInterval : ℕ := 4

/-- The ordinal number of the replacement we're interested in -/
def targetReplacement : ℕ := 10

/-- Theorem stating that the 10th replacement will occur in January -/
theorem tenth_replacement_in_january :
  monthAfter Month.January ((targetReplacement - 1) * replacementInterval) = Month.January := by
  sorry

end NUMINAMATH_CALUDE_tenth_replacement_in_january_l533_53303


namespace NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l533_53330

theorem cos_theta_plus_pi_fourth (θ : ℝ) (h : Real.sin (θ - π/4) = 1/5) : 
  Real.cos (θ + π/4) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l533_53330


namespace NUMINAMATH_CALUDE_area_ratio_preserved_under_affine_transformation_l533_53385

-- Define a polygon type
def Polygon := Set (ℝ × ℝ)

-- Define an affine transformation type
def AffineTransformation := (ℝ × ℝ) → (ℝ × ℝ)

-- Define an area function for polygons
noncomputable def area (P : Polygon) : ℝ := sorry

-- State the theorem
theorem area_ratio_preserved_under_affine_transformation
  (M N : Polygon) (f : AffineTransformation) :
  let M' := f '' M
  let N' := f '' N
  area M / area N = area M' / area N' := by sorry

end NUMINAMATH_CALUDE_area_ratio_preserved_under_affine_transformation_l533_53385


namespace NUMINAMATH_CALUDE_max_two_digit_div_sum_of_digits_l533_53316

theorem max_two_digit_div_sum_of_digits :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 →
    0 ≤ b ∧ b ≤ 9 →
    ¬(a = 0 ∧ b = 0) →
    (10 * a + b) / (a + b) ≤ 10 ∧
    ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ¬(a = 0 ∧ b = 0) ∧ (10 * a + b) / (a + b) = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_two_digit_div_sum_of_digits_l533_53316


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l533_53383

/-- Given a curve y = x^3 - 2x + 1 and a point (-1, 2) on this curve,
    if the tangent line at this point is perpendicular to the line ax + y + 1 = 0,
    then a = 1. -/
theorem tangent_line_perpendicular (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 2*x + 1
  let point : ℝ × ℝ := (-1, 2)
  let tangent_slope : ℝ := (deriv f) point.1
  let perpendicular_line : ℝ → ℝ := λ x => -a*x - 1
  f point.1 = point.2 ∧ 
  tangent_slope * (perpendicular_line point.1 - perpendicular_line (-1)) / (point.1 - (-1)) = -1 
  → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l533_53383


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_root_greater_than_three_implies_k_greater_than_one_l533_53363

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 - (2*k + 2)*x + 2*k + 1

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0 :=
sorry

-- Theorem 2: If one root is greater than 3, then k > 1
theorem root_greater_than_three_implies_k_greater_than_one (k : ℝ) :
  (∃ x : ℝ, quadratic_equation k x = 0 ∧ x > 3) → k > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_root_greater_than_three_implies_k_greater_than_one_l533_53363


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l533_53319

def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/7) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l533_53319


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l533_53362

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l533_53362


namespace NUMINAMATH_CALUDE_cookies_with_new_ingredients_l533_53378

/-- Represents the number of cookies that can be made with given amounts of flour and sugar. -/
def cookies_made (flour : ℚ) (sugar : ℚ) : ℚ :=
  18 * (flour / 2) -- or equivalently, 18 * (sugar / 1)

/-- Theorem stating that 27 cookies can be made with 3 cups of flour and 1.5 cups of sugar,
    given the initial ratio of ingredients to cookies. -/
theorem cookies_with_new_ingredients :
  cookies_made 3 1.5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cookies_with_new_ingredients_l533_53378


namespace NUMINAMATH_CALUDE_A_intersect_B_l533_53323

def A : Set ℝ := {-1, 0, 1}

def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

theorem A_intersect_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l533_53323


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l533_53305

/-- A parabola passing through two points with the same y-coordinate -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  n : ℝ

/-- The x-coordinate of the axis of symmetry of a parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := 2

/-- Theorem: The axis of symmetry of a parabola passing through (1,n) and (3,n) is x = 2 -/
theorem parabola_axis_of_symmetry (p : Parabola) : 
  p.n = p.a * 1^2 + p.b * 1 + p.c ∧ 
  p.n = p.a * 3^2 + p.b * 3 + p.c → 
  axisOfSymmetry p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l533_53305


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l533_53342

theorem sum_of_cubes_divisibility (k n : ℤ) : 
  (∃ m : ℤ, k + n = 3 * m) → (∃ l : ℤ, k^3 + n^3 = 9 * l) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l533_53342


namespace NUMINAMATH_CALUDE_project_completion_l533_53388

theorem project_completion 
  (a b c d e : ℕ) 
  (f g : ℝ) 
  (h₁ : a > 0) 
  (h₂ : c > 0) 
  (h₃ : f > 0) 
  (h₄ : g > 0) :
  (d : ℝ) * (b : ℝ) * g * (e : ℝ) / ((c : ℝ) * (a : ℝ)) = 
  (b : ℝ) * (d : ℝ) * g * (e : ℝ) / ((c : ℝ) * (a : ℝ)) :=
by sorry

#check project_completion

end NUMINAMATH_CALUDE_project_completion_l533_53388


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l533_53308

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  (2 * i / (1 - i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l533_53308


namespace NUMINAMATH_CALUDE_polygon_diagonals_l533_53373

theorem polygon_diagonals (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (n - 3 = 5) →  -- At most 5 diagonals can be drawn from any vertex
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l533_53373


namespace NUMINAMATH_CALUDE_quadratic_function_property_l533_53393

theorem quadratic_function_property (a m : ℝ) (h_a : a > 0) : 
  let f := fun (x : ℝ) ↦ x^2 + x + a
  f m < 0 → f (m + 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l533_53393


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l533_53309

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (2*m + 3)*x + m^2

-- Define the condition for distinct real roots
def has_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

-- Part 1: Range of m
theorem range_of_m (m : ℝ) (h : has_distinct_real_roots m) : m > -3/4 := by
  sorry

-- Part 2: Value of m when 1/x₁ + 1/x₂ = 1
theorem value_of_m (m : ℝ) (h1 : has_distinct_real_roots m)
  (h2 : ∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ 1/x₁ + 1/x₂ = 1) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l533_53309


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_one_max_value_implies_a_l533_53341

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

-- Part I
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x < 1} = {x : ℝ | x < (1/2 : ℝ)} := by sorry

-- Part II
theorem max_value_implies_a :
  (∃ (x : ℝ), f a x = 6) ∧ (∀ (x : ℝ), f a x ≤ 6) → a = 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_one_max_value_implies_a_l533_53341


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l533_53320

theorem baker_remaining_pastries (pastries_made pastries_sold : ℕ) 
  (h1 : pastries_made = 148)
  (h2 : pastries_sold = 103) :
  pastries_made - pastries_sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l533_53320


namespace NUMINAMATH_CALUDE_jenga_blocks_remaining_l533_53351

/-- Represents a Jenga game with the given parameters -/
structure JengaGame where
  initialBlocks : Nat
  players : Nat
  completedRounds : Nat
  blocksRemovedPerPlayer : Nat
  extraBlocksRemoved : Nat

/-- Calculates the number of blocks remaining in a Jenga game -/
def blocksRemaining (game : JengaGame) : Nat :=
  game.initialBlocks - 
  (game.players * game.completedRounds * game.blocksRemovedPerPlayer + game.extraBlocksRemoved)

/-- Theorem stating the number of blocks remaining in the specific Jenga game scenario -/
theorem jenga_blocks_remaining : 
  let game : JengaGame := {
    initialBlocks := 54,
    players := 5,
    completedRounds := 5,
    blocksRemovedPerPlayer := 1,
    extraBlocksRemoved := 1
  }
  blocksRemaining game = 28 := by sorry

end NUMINAMATH_CALUDE_jenga_blocks_remaining_l533_53351


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l533_53329

theorem sum_of_squares_and_square_of_sum : (3 + 5)^2 + (3^2 + 5^2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l533_53329


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l533_53384

/-- The distance from a point on the parabola y^2 = 4x to its focus -/
def distance_to_focus (x : ℝ) : ℝ :=
  x + 1

theorem parabola_focus_distance :
  let x : ℝ := 2
  let y : ℝ := 2 * Real.sqrt 2
  y^2 = 4*x → distance_to_focus x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l533_53384


namespace NUMINAMATH_CALUDE_factory_working_days_l533_53394

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 4560

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1140

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days :
  working_days = 4 :=
sorry

end NUMINAMATH_CALUDE_factory_working_days_l533_53394


namespace NUMINAMATH_CALUDE_yard_length_l533_53306

theorem yard_length (n : ℕ) (d : ℝ) (h1 : n = 26) (h2 : d = 14) : 
  (n - 1) * d = 350 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_l533_53306


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l533_53301

/-- Represents the maximum distance a car can travel by switching tires -/
def max_distance (front_tire_life rear_tire_life : ℕ) : ℕ :=
  let swap_point := front_tire_life / 2
  swap_point + min (rear_tire_life - swap_point) (front_tire_life - swap_point)

/-- Theorem stating the maximum distance a car can travel with given tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 21000 28000 = 24000 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l533_53301


namespace NUMINAMATH_CALUDE_negation_equivalence_l533_53387

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l533_53387


namespace NUMINAMATH_CALUDE_unique_n_with_conditions_l533_53365

theorem unique_n_with_conditions :
  ∃! n : ℕ,
    50 ≤ n ∧ n ≤ 150 ∧
    7 ∣ n ∧
    n % 9 = 3 ∧
    n % 6 = 3 ∧
    n % 11 = 5 ∧
    n = 109 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_conditions_l533_53365


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l533_53340

theorem unique_solution_absolute_value_equation :
  ∃! y : ℝ, |y - 25| + |y - 15| = |2*y - 40| :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l533_53340


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l533_53381

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let side := d / Real.sqrt 2
  side * side = 144 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l533_53381


namespace NUMINAMATH_CALUDE_remainder_2468135790_mod_99_l533_53360

theorem remainder_2468135790_mod_99 :
  2468135790 % 99 = 54 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135790_mod_99_l533_53360


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l533_53307

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l533_53307


namespace NUMINAMATH_CALUDE_distance_AB_on_parabola_l533_53336

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem distance_AB_on_parabola (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  ‖A - focus‖ = ‖point_B - focus‖ →  -- |AF| = |BF|
  ‖A - point_B‖ = 2 * Real.sqrt 2 :=  -- |AB| = 2√2
by sorry

end NUMINAMATH_CALUDE_distance_AB_on_parabola_l533_53336


namespace NUMINAMATH_CALUDE_unique_number_satisfying_means_l533_53353

theorem unique_number_satisfying_means : ∃! X : ℝ,
  (28 + X + 70 + 88 + 104) / 5 = 67 ∧
  (50 + 62 + 97 + 124 + X) / 5 = 75.6 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_means_l533_53353


namespace NUMINAMATH_CALUDE_car_distance_calculation_l533_53344

/-- Given a car's speed and how a speed increase affects travel time, calculate the distance traveled. -/
theorem car_distance_calculation (V : ℝ) (D : ℝ) (h1 : V = 40) 
  (h2 : D / V - D / (V + 20) = 0.5) : D = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l533_53344


namespace NUMINAMATH_CALUDE_f_integer_values_l533_53369

/-- The function f(a, b) as defined in the problem -/
def f (a b : ℕ+) : ℚ :=
  (a.val^2 + b.val^2 + a.val * b.val) / (a.val * b.val - 1)

/-- Main theorem stating the possible integer values of f(a, b) -/
theorem f_integer_values (a b : ℕ+) (h : a.val * b.val ≠ 1) :
  (∃ n : ℤ, f a b = n) → (f a b = 4 ∨ f a b = 7) :=
sorry

end NUMINAMATH_CALUDE_f_integer_values_l533_53369


namespace NUMINAMATH_CALUDE_ten_point_circle_chords_l533_53347

/-- The number of chords between non-adjacent points on a circle with n points -/
def non_adjacent_chords (n : ℕ) : ℕ :=
  Nat.choose n 2 - n

/-- Theorem: Given 10 points on a circle, there are 35 chords connecting non-adjacent points -/
theorem ten_point_circle_chords :
  non_adjacent_chords 10 = 35 := by
  sorry

#eval non_adjacent_chords 10  -- This should output 35

end NUMINAMATH_CALUDE_ten_point_circle_chords_l533_53347


namespace NUMINAMATH_CALUDE_smaller_number_proof_l533_53338

theorem smaller_number_proof (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : min a b = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l533_53338


namespace NUMINAMATH_CALUDE_complex_argument_range_l533_53361

theorem complex_argument_range (z : ℂ) (h : Complex.abs (2 * z + 1 / z) = 1) :
  ∃ (θ : ℝ), Complex.arg z = θ ∧
  ((θ ∈ Set.Icc (Real.pi / 2 - 1 / 2 * Real.arccos (3 / 4)) (Real.pi / 2 + 1 / 2 * Real.arccos (3 / 4))) ∨
   (θ ∈ Set.Icc (3 * Real.pi / 2 - 1 / 2 * Real.arccos (3 / 4)) (3 * Real.pi / 2 + 1 / 2 * Real.arccos (3 / 4)))) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_range_l533_53361


namespace NUMINAMATH_CALUDE_polynomial_remainder_l533_53312

def f (x : ℝ) : ℝ := 5*x^6 - 3*x^4 + 6*x^3 - 8*x + 10

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, f = λ x => (3*x - 9) * q x + 3550 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l533_53312


namespace NUMINAMATH_CALUDE_chrysler_building_floors_l533_53374

theorem chrysler_building_floors :
  ∀ (chrysler leeward : ℕ),
    chrysler = leeward + 11 →
    chrysler + leeward = 35 →
    chrysler = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_chrysler_building_floors_l533_53374


namespace NUMINAMATH_CALUDE_cube_weight_doubling_l533_53372

/-- Given a cube of metal weighing 7 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 56 pounds. -/
theorem cube_weight_doubling (ρ : ℝ) (s : ℝ) (h1 : s > 0) (h2 : ρ * s^3 = 7) :
  ρ * (2*s)^3 = 56 := by
sorry

end NUMINAMATH_CALUDE_cube_weight_doubling_l533_53372


namespace NUMINAMATH_CALUDE_smallest_value_in_ratio_l533_53313

theorem smallest_value_in_ratio (a b c d x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_ratio : ∃ k : ℝ, x = k * a ∧ y = k * b ∧ z = k * c)
  (h_sum : x + y + z = d) :
  min x (min y z) = d * a / (a + b + c) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_value_in_ratio_l533_53313


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l533_53367

-- Define the given conditions
def months_per_bar : ℕ := 2
def cost_per_bar : ℚ := 8
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 6
def months_in_year : ℕ := 12

-- Define the function to calculate the cost of soap for a year
def soap_cost_for_year : ℚ :=
  let bars_needed := months_in_year / months_per_bar
  let total_cost := bars_needed * cost_per_bar
  let discount := if bars_needed ≥ discount_threshold then discount_rate * total_cost else 0
  total_cost - discount

-- Theorem statement
theorem soap_cost_theorem : soap_cost_for_year = 43.2 := by
  sorry


end NUMINAMATH_CALUDE_soap_cost_theorem_l533_53367


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l533_53370

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = 2 * x - 3) : 
  ∀ x < 0, f x = 2 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l533_53370


namespace NUMINAMATH_CALUDE_unique_positive_solution_l533_53324

theorem unique_positive_solution : ∃! y : ℝ, y > 0 ∧ (y - 6) / 16 = 6 / (y - 16) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l533_53324


namespace NUMINAMATH_CALUDE_abs_nonnegative_rational_l533_53326

theorem abs_nonnegative_rational (x : ℚ) : |x| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_nonnegative_rational_l533_53326


namespace NUMINAMATH_CALUDE_hilt_garden_border_l533_53339

/-- The number of rocks needed to complete the border -/
def total_rocks : ℕ := 125

/-- The number of rocks Mrs. Hilt already has -/
def current_rocks : ℕ := 64

/-- The number of additional rocks Mrs. Hilt needs -/
def additional_rocks : ℕ := total_rocks - current_rocks

theorem hilt_garden_border :
  additional_rocks = 61 := by sorry

end NUMINAMATH_CALUDE_hilt_garden_border_l533_53339


namespace NUMINAMATH_CALUDE_linear_function_proof_l533_53396

/-- Given a linear function f(x) = kx passing through (2,4), prove f(-2) = -4 -/
theorem linear_function_proof (k : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = k * x) →  -- Function definition
  f 2 = 4 →             -- Point (2,4) lies on the graph
  f (-2) = -4 :=        -- Prove f(-2) = -4
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l533_53396


namespace NUMINAMATH_CALUDE_line_AB_equation_l533_53337

/-- Triangle ABC with given coordinates and line equations -/
structure Triangle where
  B : ℝ × ℝ
  C : ℝ × ℝ
  line_AC : ℝ → ℝ → ℝ
  altitude_A_AB : ℝ → ℝ → ℝ

/-- The equation of line AB in the given triangle -/
def line_AB (t : Triangle) : ℝ → ℝ → ℝ :=
  fun x y => 3 * (x - 3) - 2 * (y - 4)

/-- Theorem stating that the equation of line AB is correct -/
theorem line_AB_equation (t : Triangle) 
  (hB : t.B = (3, 4))
  (hC : t.C = (5, 2))
  (hAC : t.line_AC = fun x y => x - 4*y + 3)
  (hAlt : t.altitude_A_AB = fun x y => 2*x + 3*y - 16) :
  line_AB t = fun x y => 3 * (x - 3) - 2 * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_line_AB_equation_l533_53337


namespace NUMINAMATH_CALUDE_second_discount_percentage_l533_53350

/-- Proves that the second discount percentage is 10% given the initial price,
    first and third discount percentages, and the final price after all discounts. -/
theorem second_discount_percentage
  (initial_price : ℝ)
  (first_discount : ℝ)
  (third_discount : ℝ)
  (final_price : ℝ)
  (h1 : initial_price = 9356.725146198829)
  (h2 : first_discount = 20)
  (h3 : third_discount = 5)
  (h4 : final_price = 6400)
  : ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) ∧
    second_discount = 10 := by
  sorry


end NUMINAMATH_CALUDE_second_discount_percentage_l533_53350


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l533_53380

/-- Calculates the simple interest rate given principal, final amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  (amount - principal) * 100 / (principal * time)

theorem simple_interest_rate_calculation 
  (principal amount : ℚ) (time : ℕ) 
  (h_principal : principal = 650)
  (h_amount : amount = 950)
  (h_time : time = 5) :
  simple_interest_rate principal amount time = (950 - 650) * 100 / (650 * 5) :=
by
  sorry

#eval simple_interest_rate 650 950 5

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l533_53380


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l533_53334

-- Define the necessary types
variable {Point : Type*} [NormedAddCommGroup Point] [InnerProductSpace ℝ Point] [Finite Point]
variable {Line : Type*} [NormedAddCommGroup Line] [InnerProductSpace ℝ Line] [Finite Line]
variable {Plane : Type*} [NormedAddCommGroup Plane] [InnerProductSpace ℝ Plane] [Finite Plane]

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m n → 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  perpendicular_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l533_53334


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_is_twenty_percent_l533_53331

/-- Represents the percentage of fair-haired employees who are women -/
def fair_haired_women_ratio : ℝ := 0.4

/-- Represents the percentage of employees who have fair hair -/
def fair_haired_ratio : ℝ := 0.5

/-- Calculates the percentage of employees who are women with fair hair -/
def women_fair_hair_percentage : ℝ := fair_haired_women_ratio * fair_haired_ratio

theorem women_fair_hair_percentage_is_twenty_percent :
  women_fair_hair_percentage = 0.2 := by sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_is_twenty_percent_l533_53331


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l533_53343

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {1, 2, 3, 4}

theorem complement_intersection_equals_set : 
  (U \ (A ∩ B)) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l533_53343


namespace NUMINAMATH_CALUDE_substitution_result_l533_53399

theorem substitution_result (x y : ℝ) :
  y = x - 1 ∧ x + 2*y = 7 → x + 2*x - 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_substitution_result_l533_53399


namespace NUMINAMATH_CALUDE_profit_growth_equation_l533_53375

theorem profit_growth_equation (x : ℝ) : 
  (250000 : ℝ) * (1 + x)^2 = 360000 → 25 * (1 + x)^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_profit_growth_equation_l533_53375


namespace NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l533_53355

theorem tan_value_from_sin_cos_equation (α : ℝ) 
  (h : 3 * Real.sin ((33 * π) / 14 + α) = -5 * Real.cos ((5 * π) / 14 + α)) : 
  Real.tan ((5 * π) / 14 + α) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l533_53355


namespace NUMINAMATH_CALUDE_cos_270_degrees_l533_53379

-- Define cosine function on the unit circle
noncomputable def cosine (angle : Real) : Real :=
  (Complex.exp (Complex.I * angle)).re

-- State the theorem
theorem cos_270_degrees : cosine (3 * Real.pi / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l533_53379


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l533_53395

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 1)^2 = 4 ∧ (x₂ - 1)^2 = 4 ∧ x₁ = 3 ∧ x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l533_53395


namespace NUMINAMATH_CALUDE_honzik_triangle_solution_l533_53314

/-- Represents the side lengths of a triangle -/
structure TriangleSides where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem setup -/
def HonzikTriangleProblem (t : TriangleSides) : Prop :=
  -- Shape 1 (3 triangles): Perimeter = 43 cm
  3 * t.a + t.b + t.c = 43 ∧
  -- Shape 2 (3 triangles): Perimeter = 35 cm
  t.a + t.b + 3 * t.c = 35 ∧
  -- Shape 3 (4 triangles): Perimeter = 46 cm
  2 * (t.a + t.b + t.c) = 46

/-- The theorem to prove -/
theorem honzik_triangle_solution :
  ∃ t : TriangleSides, HonzikTriangleProblem t ∧ t.a = 10 ∧ t.b = 7 ∧ t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_honzik_triangle_solution_l533_53314


namespace NUMINAMATH_CALUDE_martian_traffic_light_signals_l533_53321

/-- Represents a Martian traffic light configuration -/
def MartianTrafficLight := Fin 6 → Bool

/-- The number of bulbs in the traffic light -/
def num_bulbs : Nat := 6

/-- Checks if two configurations are indistinguishable under the given conditions -/
def indistinguishable (c1 c2 : MartianTrafficLight) : Prop :=
  sorry

/-- Counts the number of distinguishable configurations -/
def count_distinguishable_configs : Nat :=
  sorry

/-- Theorem stating the number of distinguishable Martian traffic light signals -/
theorem martian_traffic_light_signals :
  count_distinguishable_configs = 44 :=
sorry

end NUMINAMATH_CALUDE_martian_traffic_light_signals_l533_53321


namespace NUMINAMATH_CALUDE_complex_fraction_real_l533_53318

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * Complex.I) / (2 - Complex.I)).im = 0 → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l533_53318


namespace NUMINAMATH_CALUDE_cube_volume_l533_53352

theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 96) : ∃ (volume : ℝ), volume = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l533_53352


namespace NUMINAMATH_CALUDE_horner_method_v3_l533_53332

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 - x^3 + 3x^2 + 7 -/
def f (x : ℚ) : ℚ := 2 * x^4 - x^3 + 3 * x^2 + 7

theorem horner_method_v3 :
  let coeffs := [2, -1, 3, 0, 7]
  let x := 3
  horner coeffs x = 54 ∧ f x = horner coeffs x := by sorry

#check horner_method_v3

end NUMINAMATH_CALUDE_horner_method_v3_l533_53332


namespace NUMINAMATH_CALUDE_cobys_road_trip_l533_53327

/-- Coby's road trip problem -/
theorem cobys_road_trip 
  (distance_to_idaho : ℝ) 
  (distance_from_idaho : ℝ) 
  (speed_from_idaho : ℝ) 
  (total_time : ℝ) 
  (h1 : distance_to_idaho = 640)
  (h2 : distance_from_idaho = 550)
  (h3 : speed_from_idaho = 50)
  (h4 : total_time = 19) :
  let time_from_idaho := distance_from_idaho / speed_from_idaho
  let time_to_idaho := total_time - time_from_idaho
  distance_to_idaho / time_to_idaho = 80 := by
sorry


end NUMINAMATH_CALUDE_cobys_road_trip_l533_53327


namespace NUMINAMATH_CALUDE_special_number_l533_53349

def is_consecutive (a b c d e : ℕ) : Prop :=
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (d + 1 = e)

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    is_consecutive a b c d e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (a * 10 + b) * c = d * 10 + e

theorem special_number :
  satisfies_condition 13452 :=
sorry

end NUMINAMATH_CALUDE_special_number_l533_53349


namespace NUMINAMATH_CALUDE_sum_F_equals_535501_l533_53333

/-- F(n) is the smallest positive integer greater than n whose sum of digits is equal to the sum of the digits of n -/
def F (n : ℕ) : ℕ := sorry

/-- The sum of F(n) for n from 1 to 1000 -/
def sum_F : ℕ := (List.range 1000).map F |>.sum

theorem sum_F_equals_535501 : sum_F = 535501 := by sorry

end NUMINAMATH_CALUDE_sum_F_equals_535501_l533_53333


namespace NUMINAMATH_CALUDE_angle_equality_l533_53317

theorem angle_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α + Real.cos β - Real.cos (α + β) = 3/2) :
  α = π/3 ∧ β = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l533_53317


namespace NUMINAMATH_CALUDE_point_inside_circle_l533_53328

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A point P at distance d from the center of the circle -/
structure Point where
  P : ℝ × ℝ
  d : ℝ

/-- Definition of a point being inside a circle -/
def is_inside (c : Circle) (p : Point) : Prop :=
  p.d < c.r

/-- Theorem: If the distance from a point to the center of a circle
    is less than the radius, then the point is inside the circle -/
theorem point_inside_circle (c : Circle) (p : Point) 
    (h : p.d < c.r) : is_inside c p := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l533_53328


namespace NUMINAMATH_CALUDE_average_difference_l533_53366

def average (a b c : ℕ) : ℚ :=
  (a + b + c : ℚ) / 3

theorem average_difference : average 20 40 60 - average 10 70 28 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l533_53366


namespace NUMINAMATH_CALUDE_heart_diamond_club_probability_l533_53345

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of cards of each suit
def cards_per_suit : ℕ := 13

-- Define the probability of drawing a specific sequence of cards
def draw_probability (deck_size : ℕ) (hearts diamonds clubs : ℕ) : ℚ :=
  (hearts : ℚ) / deck_size *
  (diamonds : ℚ) / (deck_size - 1) *
  (clubs : ℚ) / (deck_size - 2)

-- Theorem statement
theorem heart_diamond_club_probability :
  draw_probability standard_deck cards_per_suit cards_per_suit cards_per_suit = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_heart_diamond_club_probability_l533_53345


namespace NUMINAMATH_CALUDE_f_bound_l533_53304

/-- The function f(x) = (e^x - 1) / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem f_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, 0 < |x| ∧ |x| < Real.log (1 + a) → |f x - 1| < a :=
sorry

end NUMINAMATH_CALUDE_f_bound_l533_53304


namespace NUMINAMATH_CALUDE_pet_store_puppies_l533_53322

theorem pet_store_puppies (sold : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : sold = 24)
  (h2 : cages = 8)
  (h3 : puppies_per_cage = 4) :
  sold + cages * puppies_per_cage = 56 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l533_53322


namespace NUMINAMATH_CALUDE_parabola_intersects_line_segment_range_l533_53377

/-- Parabola equation -/
def parabola (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - 1

/-- Line segment AB -/
def line_segment_AB (x : ℝ) : ℝ := -x + 3

/-- Point A -/
def point_A : ℝ × ℝ := (3, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (0, 3)

/-- Theorem stating the range of m for which the parabola intersects line segment AB at two distinct points -/
theorem parabola_intersects_line_segment_range :
  ∃ (m_min m_max : ℝ), m_min = 3 ∧ m_max = 10/3 ∧
  ∀ (m : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
              0 ≤ x₁ ∧ x₁ ≤ 3 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3 ∧
              parabola m x₁ = line_segment_AB x₁ ∧
              parabola m x₂ = line_segment_AB x₂) ↔
             (m_min ≤ m ∧ m ≤ m_max) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersects_line_segment_range_l533_53377


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l533_53335

/-- Given a person's income and savings, prove the ratio of income to expenditure -/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (h1 : income = 14000) 
  (h2 : savings = 2000) :
  (income : ℚ) / (income - savings) = 7 / 6 :=
by sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l533_53335


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l533_53354

theorem complex_fraction_equality : (1 - I) / (2 - I) = 3/5 - (1/5) * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l533_53354
