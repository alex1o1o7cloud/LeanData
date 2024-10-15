import Mathlib

namespace NUMINAMATH_CALUDE_emily_beads_count_l503_50380

theorem emily_beads_count (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : beads_per_necklace = 5) 
  (h2 : necklaces_made = 4) : 
  beads_per_necklace * necklaces_made = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l503_50380


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l503_50320

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 16 * x - 9 * y^2 + 18 * y - 23 = 0

-- State the theorem
theorem hyperbola_vertices_distance :
  ∃ (a b c d : ℝ),
    (∀ x y, hyperbola_equation x y ↔ ((x - a)^2 / b^2 - (y - c)^2 / d^2 = 1)) ∧
    2 * Real.sqrt b^2 = Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l503_50320


namespace NUMINAMATH_CALUDE_equation_equivalence_l503_50350

theorem equation_equivalence (x y : ℝ) : 
  (2 * x + y = 1) ↔ (y = 1 - 2 * x) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l503_50350


namespace NUMINAMATH_CALUDE_range_of_m_l503_50379

def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + m*x₀ + 2*m - 3 < 0

def q (m : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m < 2 ∨ (4 ≤ m ∧ m ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l503_50379


namespace NUMINAMATH_CALUDE_range_of_a_l503_50321

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0) → 
  a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l503_50321


namespace NUMINAMATH_CALUDE_fox_initial_money_l503_50329

/-- The number of times Fox crosses the bridge -/
def crossings : ℕ := 3

/-- The toll Fox pays after each crossing -/
def toll : ℕ := 40

/-- Function to calculate Fox's money after n crossings -/
def foxMoney (initial : ℕ) (n : ℕ) : ℤ :=
  (2^n : ℤ) * initial - (2^n - 1) * toll

theorem fox_initial_money :
  ∃ x : ℕ, foxMoney x crossings = 0 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_fox_initial_money_l503_50329


namespace NUMINAMATH_CALUDE_expansion_coefficient_l503_50393

/-- Given that in the expansion of (ax+1)(x+1/x)^6, the coefficient of x^3 is 30, prove that a = 2 -/
theorem expansion_coefficient (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 6 k) * a = 30 ∧ 6 - 2*k + 1 = 3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l503_50393


namespace NUMINAMATH_CALUDE_uniform_price_is_250_l503_50387

/-- Represents the agreement between an employer and a servant --/
structure Agreement where
  full_year_salary : ℕ
  uniform_included : Bool

/-- Represents the actual outcome of the servant's employment --/
structure Outcome where
  months_worked : ℕ
  salary_received : ℕ
  uniform_received : Bool

/-- Calculates the price of the uniform given the agreement and outcome --/
def uniform_price (agreement : Agreement) (outcome : Outcome) : ℕ :=
  agreement.full_year_salary - outcome.salary_received

/-- Theorem stating that under the given conditions, the uniform price is 250 --/
theorem uniform_price_is_250 (agreement : Agreement) (outcome : Outcome) :
  agreement.full_year_salary = 500 ∧
  agreement.uniform_included = true ∧
  outcome.months_worked = 9 ∧
  outcome.salary_received = 250 ∧
  outcome.uniform_received = true →
  uniform_price agreement outcome = 250 := by
  sorry

#eval uniform_price
  { full_year_salary := 500, uniform_included := true }
  { months_worked := 9, salary_received := 250, uniform_received := true }

end NUMINAMATH_CALUDE_uniform_price_is_250_l503_50387


namespace NUMINAMATH_CALUDE_divisor_difference_greater_than_sqrt_l503_50302

theorem divisor_difference_greater_than_sqrt (A B : ℕ) 
  (h1 : A > 1) 
  (h2 : B ∣ A^2 + 1) 
  (h3 : B > A) : 
  B - A > Real.sqrt A :=
sorry

end NUMINAMATH_CALUDE_divisor_difference_greater_than_sqrt_l503_50302


namespace NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_3_l503_50385

theorem x_gt_2_necessary_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x > 2) ∧ 
  (∃ x : ℝ, x > 2 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_3_l503_50385


namespace NUMINAMATH_CALUDE_fraction_inequality_l503_50335

theorem fraction_inequality (x : ℝ) : (x + 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l503_50335


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l503_50372

-- Define arithmetic sequence a_n
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence b_n
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_geometric_sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  2 * a 4 - (a 7)^2 + 2 * a 10 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 5 * b 9 = 16 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l503_50372


namespace NUMINAMATH_CALUDE_abs_neg_half_eq_half_l503_50310

theorem abs_neg_half_eq_half : |(-1/2 : ℚ)| = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_half_eq_half_l503_50310


namespace NUMINAMATH_CALUDE_rearrange_3008_eq_6_l503_50303

/-- The number of different four-digit numbers that can be formed by rearranging the digits in 3008 -/
def rearrange_3008 : ℕ :=
  let digits : List ℕ := [3, 0, 0, 8]
  let total_permutations := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_permutations := 
    (Nat.factorial 3 / Nat.factorial 2) +  -- starting with 3
    (Nat.factorial 3 / Nat.factorial 2)    -- starting with 8
  valid_permutations

theorem rearrange_3008_eq_6 : rearrange_3008 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rearrange_3008_eq_6_l503_50303


namespace NUMINAMATH_CALUDE_base7_divisibility_l503_50349

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

/-- Checks if a number is divisible by 29 --/
def isDivisibleBy29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

theorem base7_divisibility :
  ∃! y : ℕ, y ≤ 6 ∧ isDivisibleBy29 (base7ToDecimal 2 y 6 3) :=
sorry

end NUMINAMATH_CALUDE_base7_divisibility_l503_50349


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l503_50346

/-- If the angles of a triangle are in the ratio 1:2:3, then the triangle is right-angled. -/
theorem triangle_with_angle_ratio_1_2_3_is_right_angled (A B C : ℝ) 
  (h_angle_sum : A + B + C = 180) 
  (h_angle_ratio : ∃ (x : ℝ), A = x ∧ B = 2*x ∧ C = 3*x) : 
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l503_50346


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l503_50378

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l503_50378


namespace NUMINAMATH_CALUDE_specific_sculpture_surface_area_l503_50386

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  bottomLayerCount : ℕ
  middleLayerCount : ℕ
  topLayerCount : ℕ
  cubeEdgeLength : ℝ

/-- Calculates the exposed surface area of a cube sculpture -/
def exposedSurfaceArea (sculpture : CubeSculpture) : ℝ :=
  sorry

/-- The theorem stating that the specific sculpture has 55 square meters of exposed surface area -/
theorem specific_sculpture_surface_area :
  let sculpture : CubeSculpture := {
    bottomLayerCount := 9
    middleLayerCount := 8
    topLayerCount := 3
    cubeEdgeLength := 1
  }
  exposedSurfaceArea sculpture = 55 := by
  sorry

end NUMINAMATH_CALUDE_specific_sculpture_surface_area_l503_50386


namespace NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l503_50330

theorem frog_grasshopper_jump_difference :
  let grasshopper_jump : ℕ := 25
  let frog_jump : ℕ := 40
  frog_jump - grasshopper_jump = 15 := by
  sorry

end NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l503_50330


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l503_50354

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l503_50354


namespace NUMINAMATH_CALUDE_ellipse_vertices_distance_l503_50313

/-- The distance between the vertices of the ellipse (x^2/144) + (y^2/36) = 1 is 24 -/
theorem ellipse_vertices_distance : 
  let ellipse := {p : ℝ × ℝ | (p.1^2 / 144) + (p.2^2 / 36) = 1}
  ∃ v1 v2 : ℝ × ℝ, v1 ∈ ellipse ∧ v2 ∈ ellipse ∧ 
    (∀ p ∈ ellipse, ‖p.1‖ ≤ ‖v1.1‖) ∧
    ‖v1 - v2‖ = 24 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_vertices_distance_l503_50313


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l503_50363

/-- Given a quadratic inequality x^2 + ax + b < 0 with solution set (-1, 4), prove that ab = 12 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ -1 < x ∧ x < 4) → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l503_50363


namespace NUMINAMATH_CALUDE_diamond_commutative_l503_50394

-- Define the set T of all non-zero integers
def T : Set Int := {x : Int | x ≠ 0}

-- Define the binary operation ◇
def diamond (a b : T) : Int := 3 * a * b + a + b

-- Theorem statement
theorem diamond_commutative : ∀ (a b : T), diamond a b = diamond b a := by
  sorry

end NUMINAMATH_CALUDE_diamond_commutative_l503_50394


namespace NUMINAMATH_CALUDE_jordan_novels_count_l503_50347

theorem jordan_novels_count :
  ∀ (j a : ℕ),
  a = j / 10 →
  j = a + 108 →
  j = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_jordan_novels_count_l503_50347


namespace NUMINAMATH_CALUDE_first_digit_is_two_l503_50373

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  isThreeDigit : 100 ≤ value ∧ value < 1000

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

/-- The main theorem -/
theorem first_digit_is_two
  (n : ThreeDigitNumber)
  (h1 : ∃ d : ℕ, d * 2 = n.value)
  (h2 : isDivisibleBy n.value 6)
  (h3 : ∃ d : ℕ, d * 2 = n.value ∧ d = 2)
  : n.value / 100 = 2 := by
  sorry

#check first_digit_is_two

end NUMINAMATH_CALUDE_first_digit_is_two_l503_50373


namespace NUMINAMATH_CALUDE_slow_clock_catch_up_l503_50331

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents how many minutes the clock is slow per hour -/
def slow_rate : ℕ := 4

/-- Represents the current time on the slow clock in minutes past 11:00 -/
def current_slow_time : ℕ := 46

/-- Represents the target time on the slow clock in minutes past 11:00 -/
def target_slow_time : ℕ := 60

/-- Theorem stating that it takes 15 minutes of correct time for the slow clock to reach 12:00 -/
theorem slow_clock_catch_up :
  (target_slow_time - current_slow_time) * minutes_per_hour / (minutes_per_hour - slow_rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_slow_clock_catch_up_l503_50331


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l503_50325

theorem quadratic_inequality_condition (m : ℝ) :
  (m > 1 → ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) ∧
  (∃ m : ℝ, m ≤ 1 ∧ ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l503_50325


namespace NUMINAMATH_CALUDE_equation_solution_l503_50328

theorem equation_solution : 
  ∃! x : ℚ, (3 - 2*x) / (x + 2) + (3*x - 6) / (3 - 2*x) = 2 ∧ x = -3/5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l503_50328


namespace NUMINAMATH_CALUDE_recipe_ratio_change_l503_50389

/-- Represents the ratio of ingredients in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  { flour := 7, water := 2, sugar := 1 }

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  { flour := 7, water := 1, sugar := 2 }

/-- The amount of water in the new recipe --/
def new_water_amount : ℚ := 2

/-- The amount of sugar in the new recipe --/
def new_sugar_amount : ℚ := 4

theorem recipe_ratio_change :
  (new_ratio.water - original_ratio.water) = -1 :=
sorry

end NUMINAMATH_CALUDE_recipe_ratio_change_l503_50389


namespace NUMINAMATH_CALUDE_area_triangle_ABC_l503_50322

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle ABC in the regular octagon -/
def triangle_ABC (octagon : RegularOctagon) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: Area of triangle ABC in a regular octagon with side length 3 -/
theorem area_triangle_ABC (octagon : RegularOctagon) :
  area (triangle_ABC octagon) = 9 * (2 + Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_l503_50322


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l503_50340

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 7) :
  6 * x / ((x - 7) * (x - 4)^2) = 
    14/3 / (x - 7) + 26/33 / (x - 4) + (-8) / (x - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l503_50340


namespace NUMINAMATH_CALUDE_set_B_is_empty_l503_50383

theorem set_B_is_empty : {x : ℝ | x^2 + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_set_B_is_empty_l503_50383


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l503_50341

def euler_family_ages : List ℝ := [12, 12, 12, 12, 9, 9, 15, 17]

theorem euler_family_mean_age : 
  (euler_family_ages.sum / euler_family_ages.length : ℝ) = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l503_50341


namespace NUMINAMATH_CALUDE_weight_of_ten_moles_l503_50361

/-- Represents an iron oxide compound with the number of iron and oxygen atoms -/
structure IronOxide where
  iron_atoms : ℕ
  oxygen_atoms : ℕ

/-- Calculates the molar mass of an iron oxide compound -/
def molar_mass (compound : IronOxide) : ℝ :=
  55.85 * compound.iron_atoms + 16.00 * compound.oxygen_atoms

/-- Calculates the weight of a given number of moles of an iron oxide compound -/
def weight (moles : ℝ) (compound : IronOxide) : ℝ :=
  moles * molar_mass compound

/-- Theorem: The weight of 10 moles of an iron oxide compound is 10 times its molar mass -/
theorem weight_of_ten_moles (compound : IronOxide) :
  weight 10 compound = 10 * molar_mass compound := by
  sorry

#check weight_of_ten_moles

end NUMINAMATH_CALUDE_weight_of_ten_moles_l503_50361


namespace NUMINAMATH_CALUDE_lucas_speed_equals_miguel_speed_l503_50300

/-- Given the relative speeds of Miguel, Sophie, and Lucas, prove that Lucas's speed equals Miguel's speed. -/
theorem lucas_speed_equals_miguel_speed (miguel_speed : ℝ) (sophie_speed : ℝ) (lucas_speed : ℝ)
  (h1 : miguel_speed = 6)
  (h2 : sophie_speed = 3/4 * miguel_speed)
  (h3 : lucas_speed = 4/3 * sophie_speed) :
  lucas_speed = miguel_speed :=
by sorry

end NUMINAMATH_CALUDE_lucas_speed_equals_miguel_speed_l503_50300


namespace NUMINAMATH_CALUDE_min_value_f_when_m_1_existence_of_m_l503_50353

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / (2 * x)

def g (m : ℝ) (x : ℝ) : ℝ := x - 2 * m

theorem min_value_f_when_m_1 :
  ∃ x₀ > 0, ∀ x > 0, f 1 x₀ ≤ f 1 x ∧ f 1 x₀ = 1 - Real.log 2 := by sorry

theorem existence_of_m :
  ∃ m ∈ Set.Ioo (4/5 : ℝ) 1, ∀ x ∈ Set.Icc (Real.exp (-1)) 1,
    f m x > g m x + 1 := by sorry

end NUMINAMATH_CALUDE_min_value_f_when_m_1_existence_of_m_l503_50353


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l503_50351

/-- Given an algebraic expression ax-2, if the value of the expression is 4 when x=2, then a=3 -/
theorem algebraic_expression_value (a : ℝ) : (a * 2 - 2 = 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l503_50351


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l503_50395

/-- A quadratic function with axis of symmetry at x = 6 and p(0) = -3 -/
def p (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c (6 + x) = p a b c (6 - x)) →  -- axis of symmetry at x = 6
  p a b c 0 = -3 →                           -- p(0) = -3
  p a b c 12 = -3 :=                         -- p(12) = -3
by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l503_50395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l503_50323

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a_1 + a_5 = 6,
    prove that the sum of the first five terms is 15. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 = 6) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l503_50323


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l503_50344

/-- Proves that given Terrell hiked 8.2 miles on Saturday and 9.8 miles in total,
    the distance he hiked on Sunday is 1.6 miles. -/
theorem terrell_hike_distance (saturday_distance : Real) (total_distance : Real)
    (h1 : saturday_distance = 8.2)
    (h2 : total_distance = 9.8) :
    total_distance - saturday_distance = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l503_50344


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l503_50345

-- Define the repeating decimals
def repeating_137 : ℚ := 137 / 999
def repeating_6 : ℚ := 2 / 3

-- Theorem statement
theorem product_of_repeating_decimals : 
  repeating_137 * repeating_6 = 274 / 2997 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l503_50345


namespace NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l503_50357

/-- The cost price of an article, given specific profit and loss conditions -/
theorem cost_price_from_profit_loss_equality (selling_price_profit selling_price_loss : ℕ) 
  (h : selling_price_profit = 66 ∧ selling_price_loss = 52) :
  ∃ cost_price : ℕ, 
    (selling_price_profit - cost_price = cost_price - selling_price_loss) ∧ 
    cost_price = 59 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l503_50357


namespace NUMINAMATH_CALUDE_no_zero_term_l503_50392

/-- An arithmetic progression is defined by its first term and common difference -/
structure ArithmeticProgression where
  a : ℝ  -- first term
  d : ℝ  -- common difference

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

/-- The condition given in the problem -/
def satisfiesCondition (ap : ArithmeticProgression) : Prop :=
  nthTerm ap 5 + nthTerm ap 21 = nthTerm ap 8 + nthTerm ap 15 + nthTerm ap 13

/-- The main theorem -/
theorem no_zero_term (ap : ArithmeticProgression) 
    (h : satisfiesCondition ap) : 
    ¬∃ (n : ℕ), n > 0 ∧ nthTerm ap n = 0 :=
  sorry

end NUMINAMATH_CALUDE_no_zero_term_l503_50392


namespace NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l503_50336

theorem integral_x_plus_inverse_x : ∫ x in (1 : ℝ)..2, (x + 1/x) = 3/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l503_50336


namespace NUMINAMATH_CALUDE_maci_blue_pens_l503_50368

/-- The number of blue pens Maci needs -/
def num_blue_pens : ℕ := sorry

/-- The number of red pens Maci needs -/
def num_red_pens : ℕ := 15

/-- The cost of a blue pen in cents -/
def blue_pen_cost : ℕ := 10

/-- The cost of a red pen in cents -/
def red_pen_cost : ℕ := 2 * blue_pen_cost

/-- The total cost of all pens in cents -/
def total_cost : ℕ := 400

theorem maci_blue_pens :
  num_blue_pens * blue_pen_cost + num_red_pens * red_pen_cost = total_cost ∧
  num_blue_pens = 10 :=
sorry

end NUMINAMATH_CALUDE_maci_blue_pens_l503_50368


namespace NUMINAMATH_CALUDE_fraction_equality_l503_50356

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2*b) / (a - 2*b) = (x + 2) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l503_50356


namespace NUMINAMATH_CALUDE_inverse_variation_cube_root_l503_50309

theorem inverse_variation_cube_root (y x : ℝ) (k : ℝ) (h1 : y * x^(1/3) = k) (h2 : 2 * 8^(1/3) = k) :
  8 * x^(1/3) = k → x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_root_l503_50309


namespace NUMINAMATH_CALUDE_security_system_probability_l503_50370

theorem security_system_probability (p : ℝ) : 
  (1/8 : ℝ) * (1 - p) + (7/8 : ℝ) * p = 9/40 → p = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_security_system_probability_l503_50370


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l503_50390

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisibility_by_nine (n : ℕ) : 
  (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l503_50390


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_palindromic_product_l503_50396

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem stating that 707 is the smallest three-digit palindrome
    whose product with 103 is not a five-digit palindrome -/
theorem smallest_three_digit_palindrome_non_palindromic_product :
  (∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 707 → isFiveDigitPalindrome (n * 103)) ∧
  isThreeDigitPalindrome 707 ∧
  ¬isFiveDigitPalindrome (707 * 103) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_palindromic_product_l503_50396


namespace NUMINAMATH_CALUDE_solution_set_f_intersection_condition_l503_50352

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem 1: Solution set of f(x) > 2 when m = 5
theorem solution_set_f (x : ℝ) : f 5 x > 2 ↔ -3/2 < x ∧ x < 3/2 := by sorry

-- Theorem 2: Condition for f(x) and g(x) to always intersect
theorem intersection_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f m y = g y) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_intersection_condition_l503_50352


namespace NUMINAMATH_CALUDE_park_orchid_bushes_after_planting_l503_50339

/-- The number of orchid bushes in the park after planting -/
def total_orchid_bushes (current : ℕ) (newly_planted : ℕ) : ℕ :=
  current + newly_planted

/-- Theorem: The park will have 35 orchid bushes after planting -/
theorem park_orchid_bushes_after_planting :
  total_orchid_bushes 22 13 = 35 := by
  sorry

end NUMINAMATH_CALUDE_park_orchid_bushes_after_planting_l503_50339


namespace NUMINAMATH_CALUDE_function_identity_l503_50304

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l503_50304


namespace NUMINAMATH_CALUDE_prank_combinations_l503_50332

/-- The number of choices for each day of the prank --/
def prank_choices : List Nat := [1, 2, 6, 3, 1]

/-- The total number of combinations for the prank --/
def total_combinations : Nat := prank_choices.prod

/-- Theorem stating that the total number of combinations is 36 --/
theorem prank_combinations :
  total_combinations = 36 := by sorry

end NUMINAMATH_CALUDE_prank_combinations_l503_50332


namespace NUMINAMATH_CALUDE_circle_center_proof_l503_50343

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a point is equidistant from two parallel lines -/
def equidistantFromParallelLines (p : Point) (l1 l2 : Line) : Prop :=
  abs (l1.a * p.x + l1.b * p.y - l1.c) = abs (l2.a * p.x + l2.b * p.y - l2.c)

theorem circle_center_proof (l1 l2 l3 : Line) (p : Point) :
  l1.a = 3 ∧ l1.b = -4 ∧ l1.c = 12 ∧
  l2.a = 3 ∧ l2.b = -4 ∧ l2.c = -24 ∧
  l3.a = 1 ∧ l3.b = -2 ∧ l3.c = 0 ∧
  p.x = -6 ∧ p.y = -3 →
  pointOnLine p l3 ∧ equidistantFromParallelLines p l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_proof_l503_50343


namespace NUMINAMATH_CALUDE_four_student_committees_from_six_l503_50359

theorem four_student_committees_from_six (n k : ℕ) : n = 6 ∧ k = 4 → Nat.choose n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_four_student_committees_from_six_l503_50359


namespace NUMINAMATH_CALUDE_ambulance_ride_cost_dakota_ambulance_cost_l503_50397

/-- Calculates the cost of the ambulance ride given hospital expenses and total bill -/
theorem ambulance_ride_cost 
  (days_in_hospital : ℕ) 
  (bed_cost_per_day : ℕ) 
  (specialist_cost_per_hour : ℕ) 
  (specialist_time_minutes : ℕ) 
  (num_specialists : ℕ) 
  (total_bill : ℕ) : ℕ :=
  let bed_cost := days_in_hospital * bed_cost_per_day
  let specialist_time_hours := specialist_time_minutes / 60
  let specialist_cost := num_specialists * (specialist_cost_per_hour * specialist_time_hours)
  let hospital_cost := bed_cost + specialist_cost
  total_bill - hospital_cost

/-- The cost of Dakota's ambulance ride -/
theorem dakota_ambulance_cost : 
  ambulance_ride_cost 3 900 250 15 2 4625 = 1675 := by
  sorry


end NUMINAMATH_CALUDE_ambulance_ride_cost_dakota_ambulance_cost_l503_50397


namespace NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l503_50366

-- Define variables for the prices of each item
variable (a : ℚ) -- Price of a sack of apples
variable (b : ℚ) -- Price of a bunch of bananas
variable (c : ℚ) -- Price of a cantaloupe
variable (d : ℚ) -- Price of a carton of dates
variable (h : ℚ) -- Price of a jar of honey

-- Define the conditions
axiom total_cost : a + b + c + d + h = 30
axiom dates_cost : d = 4 * a
axiom cantaloupe_cost : c = 2 * a - b

-- Theorem to prove
theorem bananas_cantaloupe_cost : b + c = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bananas_cantaloupe_cost_l503_50366


namespace NUMINAMATH_CALUDE_sqrt_x_minus_5_real_l503_50374

theorem sqrt_x_minus_5_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) ↔ x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_5_real_l503_50374


namespace NUMINAMATH_CALUDE_recipe_total_cups_l503_50315

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalCups (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given ratio and flour amount, the total cups is 30 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 5, 3⟩
  let flourCups : ℕ := 15
  totalCups ratio flourCups = 30 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l503_50315


namespace NUMINAMATH_CALUDE_expensive_gimbap_count_l503_50338

def basic_gimbap : ℕ := 2000
def tuna_gimbap : ℕ := 3500
def red_pepper_gimbap : ℕ := 3000
def beef_gimbap : ℕ := 4000
def rice_gimbap : ℕ := 3500

def gimbap_prices : List ℕ := [basic_gimbap, tuna_gimbap, red_pepper_gimbap, beef_gimbap, rice_gimbap]

def count_expensive_gimbap (prices : List ℕ) : ℕ :=
  (prices.filter (λ price => price ≥ 3500)).length

theorem expensive_gimbap_count : count_expensive_gimbap gimbap_prices = 3 := by
  sorry

end NUMINAMATH_CALUDE_expensive_gimbap_count_l503_50338


namespace NUMINAMATH_CALUDE_second_number_is_37_l503_50312

theorem second_number_is_37 (a b c d : ℕ) : 
  a + b + c + d = 260 →
  a = 2 * b →
  c = a / 3 →
  d = 2 * (b + c) →
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_37_l503_50312


namespace NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l503_50324

theorem average_waiting_time_for_first_bite 
  (rod1_bites : ℝ) 
  (rod2_bites : ℝ) 
  (total_bites : ℝ) 
  (time_interval : ℝ) 
  (h1 : rod1_bites = 3)
  (h2 : rod2_bites = 2)
  (h3 : total_bites = rod1_bites + rod2_bites)
  (h4 : time_interval = 6) :
  (time_interval / total_bites) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l503_50324


namespace NUMINAMATH_CALUDE_hyperbola_equation_l503_50308

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) :=
  fun x y => y^2 / 20 - x^2 / 16 = 1

/-- Theorem: Given a hyperbola with center at (0, 0), foci at (0, -6) and (0, 6),
    and passing through the point (2, -5), its standard equation is y^2/20 - x^2/16 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_foci : h.foci = ((0, -6), (0, 6)))
    (h_point : h.point = (2, -5)) :
    standard_equation h = fun x y => y^2 / 20 - x^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l503_50308


namespace NUMINAMATH_CALUDE_telescope_visual_range_l503_50398

/-- Given a telescope that increases visual range by 87.5% to 150 kilometers,
    prove that the initial visual range was 80 kilometers. -/
theorem telescope_visual_range (V : ℝ) : V + 0.875 * V = 150 → V = 80 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l503_50398


namespace NUMINAMATH_CALUDE_problem_solution_l503_50376

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := m * (x + n) / (x + 1)

def tangent_perpendicular (n : ℝ) : Prop :=
  let f' : ℝ → ℝ := λ x => 1 / x
  let g' : ℝ → ℝ := λ x => (1 - n) / ((x + 1) ^ 2)
  f' 1 * g' 1 = -1

def inequality_holds (m n : ℝ) : Prop :=
  ∀ x > 0, |f x| ≥ |g m n x|

theorem problem_solution :
  (∃ n : ℝ, tangent_perpendicular n ∧ n = 5) ∧
  (∃ n : ℝ, ∃ m : ℝ, m > 0 ∧ inequality_holds m n ∧ n = -1 ∧
    (∀ m' > 0, inequality_holds m' n → m' ≤ m) ∧ m = 2) := by sorry

end

end NUMINAMATH_CALUDE_problem_solution_l503_50376


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l503_50358

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ q > 0 ∧ q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) :
  a 1 + a 8 > a 4 + a 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l503_50358


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l503_50314

/-- A geometric sequence with the given first four terms -/
def geometric_sequence : Fin 5 → ℝ
  | 0 => 10
  | 1 => -15
  | 2 => 22.5
  | 3 => -33.75
  | 4 => 50.625

/-- The common ratio of the geometric sequence -/
def common_ratio : ℝ := -1.5

theorem geometric_sequence_properties :
  (∀ n : Fin 3, geometric_sequence (n + 1) = geometric_sequence n * common_ratio) ∧
  geometric_sequence 4 = geometric_sequence 3 * common_ratio :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l503_50314


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l503_50318

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 15° + sin θ is 32.5° -/
theorem least_positive_angle_theta : 
  let θ : ℝ := 32.5
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → Real.cos (10 * π / 180) ≠ Real.sin (15 * π / 180) + Real.sin (φ * π / 180) ∧
  Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (θ * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l503_50318


namespace NUMINAMATH_CALUDE_abs_m_minus_n_l503_50311

theorem abs_m_minus_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_l503_50311


namespace NUMINAMATH_CALUDE_choir_members_count_l503_50382

theorem choir_members_count : ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l503_50382


namespace NUMINAMATH_CALUDE_notebooks_last_fifty_days_l503_50369

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_days (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, used at a rate of 4 pages per day, last for 50 days. -/
theorem notebooks_last_fifty_days :
  notebook_days 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_last_fifty_days_l503_50369


namespace NUMINAMATH_CALUDE_multiples_of_5_or_7_not_35_l503_50367

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

theorem multiples_of_5_or_7_not_35 : 
  (count_multiples 3000 5) + (count_multiples 3000 7) - (count_multiples 3000 35) = 943 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_5_or_7_not_35_l503_50367


namespace NUMINAMATH_CALUDE_parabola_vertex_l503_50355

/-- Given a quadratic function f(x) = -x^2 + cx + d where c and d are real numbers,
    and the solution to f(x) ≤ 0 is (-∞, -4] ∪ [6, ∞),
    prove that the vertex of the parabola is (1, 25). -/
theorem parabola_vertex (c d : ℝ) 
  (h : ∀ x, -x^2 + c*x + d ≤ 0 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici 6) : 
  let f := fun x => -x^2 + c*x + d
  (1, f 1) = (1, 25) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l503_50355


namespace NUMINAMATH_CALUDE_six_digit_pin_probability_six_digit_pin_probability_value_l503_50364

/-- The probability of randomly selecting a 6-digit PIN with a non-zero first digit, 
    such that the first two digits are both 6 -/
theorem six_digit_pin_probability : ℝ :=
  let total_pins := 9 * 10^5  -- 9 choices for first digit, 10 choices each for other 5 digits
  let favorable_pins := 10^4  -- 4 digits can be any number from 0 to 9
  favorable_pins / total_pins

/-- The probability is equal to 1/90 -/
theorem six_digit_pin_probability_value : six_digit_pin_probability = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_pin_probability_six_digit_pin_probability_value_l503_50364


namespace NUMINAMATH_CALUDE_luke_used_eight_stickers_l503_50348

/-- The number of stickers Luke used to decorate the greeting card -/
def stickers_used_for_card (initial_stickers bought_stickers birthday_stickers : ℕ)
  (given_to_sister remaining_stickers : ℕ) : ℕ :=
  initial_stickers + bought_stickers + birthday_stickers - given_to_sister - remaining_stickers

/-- Theorem stating that Luke used 8 stickers to decorate the greeting card -/
theorem luke_used_eight_stickers :
  stickers_used_for_card 20 12 20 5 39 = 8 := by
  sorry

end NUMINAMATH_CALUDE_luke_used_eight_stickers_l503_50348


namespace NUMINAMATH_CALUDE_games_won_is_fifteen_l503_50377

/-- Represents the number of baseball games played by Dan's high school team. -/
def total_games : ℕ := 18

/-- Represents the number of games lost by Dan's high school team. -/
def games_lost : ℕ := 3

/-- Theorem stating that the number of games won is 15. -/
theorem games_won_is_fifteen : total_games - games_lost = 15 := by
  sorry

end NUMINAMATH_CALUDE_games_won_is_fifteen_l503_50377


namespace NUMINAMATH_CALUDE_candy_difference_l503_50306

theorem candy_difference (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 40)
  (h2 : yellow < 3 * red)
  (h3 : blue = yellow / 2)
  (h4 : red + blue = 90) :
  3 * red - yellow = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l503_50306


namespace NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l503_50333

-- Proposition B
theorem proposition_b (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (m + 1) / (n + 1) < m / n := by sorry

-- Proposition C
theorem proposition_c (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  a / (c - a) > b / (c - b) := by sorry

-- Proposition D
theorem proposition_d (a b : ℝ) (h1 : a ≥ b) (h2 : b > -1) :
  a / (a + 1) ≥ b / (b + 1) := by sorry

end NUMINAMATH_CALUDE_proposition_b_proposition_c_proposition_d_l503_50333


namespace NUMINAMATH_CALUDE_crude_oil_mixture_l503_50317

/-- Given two sources of crude oil, prove that the second source contains 75% hydrocarbons -/
theorem crude_oil_mixture (
  source1_percent : ℝ)
  (source2_percent : ℝ)
  (final_volume : ℝ)
  (final_percent : ℝ)
  (source2_volume : ℝ) :
  source1_percent = 25 →
  final_volume = 50 →
  final_percent = 55 →
  source2_volume = 30 →
  source2_percent = 75 :=
by
  sorry

#check crude_oil_mixture

end NUMINAMATH_CALUDE_crude_oil_mixture_l503_50317


namespace NUMINAMATH_CALUDE_shortest_diagonal_probability_l503_50316

/-- The number of sides in the regular polygon -/
def n : ℕ := 20

/-- The total number of diagonals in the polygon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in the polygon -/
def shortest_diagonals : ℕ := n

/-- The probability of selecting a shortest diagonal -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem shortest_diagonal_probability :
  probability = 2 / 17 := by sorry

end NUMINAMATH_CALUDE_shortest_diagonal_probability_l503_50316


namespace NUMINAMATH_CALUDE_expression_value_l503_50388

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l503_50388


namespace NUMINAMATH_CALUDE_inequality_proof_l503_50337

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  Real.sqrt (1/a - a) + Real.sqrt (1/b - b) + Real.sqrt (1/c - c) ≥ 
  Real.sqrt (2*a) + Real.sqrt (2*b) + Real.sqrt (2*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l503_50337


namespace NUMINAMATH_CALUDE_multiply_y_value_l503_50375

theorem multiply_y_value (x y : ℝ) (h1 : ∃ (n : ℝ), 5 * x = n * y) 
  (h2 : x * y ≠ 0) (h3 : (1/5 * x) / (1/6 * y) = 0.7200000000000001) : 
  ∃ (n : ℝ), 5 * x = n * y ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiply_y_value_l503_50375


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l503_50391

/-- Given a segment with endpoints A(-3, 5) and B(9, -1) extended through B to point C,
    where BC = 1/2 * AB, prove that the coordinates of C are (15, -4). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (-3, 5) →
  B = (9, -1) →
  C - B = (1/2 : ℝ) • (B - A) →
  C = (15, -4) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l503_50391


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_five_l503_50360

theorem unique_square_divisible_by_five : ∃! y : ℕ, 
  (∃ n : ℕ, y = n^2) ∧ 
  (∃ k : ℕ, y = 5 * k) ∧ 
  50 < y ∧ y < 120 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_five_l503_50360


namespace NUMINAMATH_CALUDE_job_application_ratio_l503_50327

theorem job_application_ratio (total_applications in_state_applications : ℕ) 
  (h1 : total_applications = 600)
  (h2 : in_state_applications = 200) :
  (total_applications - in_state_applications) / in_state_applications = 2 := by
sorry

end NUMINAMATH_CALUDE_job_application_ratio_l503_50327


namespace NUMINAMATH_CALUDE_sum_a_c_l503_50301

theorem sum_a_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) : 
  a + c = 42 / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_a_c_l503_50301


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l503_50334

/-- Given an infinitely decreasing geometric progression with sum S and terms a₁, a₂, a₃, ...,
    prove that S / (S - a₁) = a₁ / a₂ -/
theorem geometric_progression_ratio (S a₁ a₂ : ℝ) (a : ℕ → ℝ) :
  (∀ n, a n = a₁ * (a₂ / a₁) ^ (n - 1)) →  -- Geometric progression definition
  (a₂ / a₁ < 1) →                          -- Decreasing condition
  (S = ∑' n, a n) →                        -- S is the sum of the progression
  S / (S - a₁) = a₁ / a₂ := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l503_50334


namespace NUMINAMATH_CALUDE_rectangular_prism_inequality_l503_50384

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0)
  (h_diagonal : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_inequality_l503_50384


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l503_50307

theorem triangle_side_and_area 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : a = 1) 
  (h2 : b = 2) 
  (h3 : C = 60 * π / 180) : 
  c = Real.sqrt 3 ∧ (1/2 * a * b * Real.sin C) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l503_50307


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l503_50342

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + 3

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ 4) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = 4) →
  a = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l503_50342


namespace NUMINAMATH_CALUDE_seats_filled_percentage_l503_50362

/-- Given a hall with 700 seats where 175 are vacant, prove that 75% of the seats are filled. -/
theorem seats_filled_percentage (total_seats : ℕ) (vacant_seats : ℕ) 
  (h1 : total_seats = 700) 
  (h2 : vacant_seats = 175) : 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 75 := by
  sorry

#check seats_filled_percentage

end NUMINAMATH_CALUDE_seats_filled_percentage_l503_50362


namespace NUMINAMATH_CALUDE_perfect_squares_problem_l503_50399

theorem perfect_squares_problem (m n a b c d : ℕ) :
  2000 + 100 * a + 10 * b + 9 = n^2 →
  2000 + 100 * c + 10 * d + 9 = m^2 →
  m > n →
  10 ≤ 10 * a + b →
  10 * a + b ≤ 99 →
  10 ≤ 10 * c + d →
  10 * c + d ≤ 99 →
  m + n = 100 ∧ (10 * a + b) + (10 * c + d) = 100 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_problem_l503_50399


namespace NUMINAMATH_CALUDE_largest_number_hcf_lcm_l503_50319

theorem largest_number_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 52) → 
  (Nat.lcm a b = 52 * 11 * 12) → 
  (max a b = 624) := by
sorry

end NUMINAMATH_CALUDE_largest_number_hcf_lcm_l503_50319


namespace NUMINAMATH_CALUDE_event_ticket_revenue_l503_50305

theorem event_ticket_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 160)
  (h_total_revenue : total_revenue = 2400) :
  ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
    full_price + half_price = total_tickets ∧
    full_price * price + half_price * (price / 2) = total_revenue ∧
    full_price * price = 960 := by
  sorry

end NUMINAMATH_CALUDE_event_ticket_revenue_l503_50305


namespace NUMINAMATH_CALUDE_full_time_more_than_three_years_l503_50326

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  secondYear : ℝ
  thirdYear : ℝ
  notFirstYear : ℝ
  partTime : ℝ
  partTimeMoreThanTwoYears : ℝ

/-- Theorem stating the percentage of full-time associates at the firm for more than three years -/
theorem full_time_more_than_three_years 
  (percentages : AssociatePercentages)
  (h1 : percentages.secondYear = 30)
  (h2 : percentages.thirdYear = 20)
  (h3 : percentages.notFirstYear = 60)
  (h4 : percentages.partTime = 10)
  (h5 : percentages.partTimeMoreThanTwoYears = percentages.partTime / 2)
  : ℝ := by
  sorry

#check full_time_more_than_three_years

end NUMINAMATH_CALUDE_full_time_more_than_three_years_l503_50326


namespace NUMINAMATH_CALUDE_divisibility_of_integer_part_l503_50365

theorem divisibility_of_integer_part (k : ℕ+) (n : ℕ) :
  let A : ℝ := k + 1/2 + Real.sqrt (k^2 + 1/4)
  (⌊A^n⌋ : ℤ) % k = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_integer_part_l503_50365


namespace NUMINAMATH_CALUDE_tangent_circles_distance_l503_50381

/-- Given two circles of radius 5 that are externally tangent to each other
    and internally tangent to a circle of radius 13, the distance between
    their points of tangency with the larger circle is 2√39. -/
theorem tangent_circles_distance (r₁ r₂ R : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 5) (h₃ : R = 13) :
  let d := 2 * (R - r₁)  -- distance between centers of small circles and large circle
  let s := r₁ + r₂       -- distance between centers of small circles
  2 * Real.sqrt ((d ^ 2) - (s / 2) ^ 2) = 2 * Real.sqrt 39 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_distance_l503_50381


namespace NUMINAMATH_CALUDE_equation_solution_l503_50371

theorem equation_solution : 
  ∃! x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l503_50371
