import Mathlib

namespace NUMINAMATH_CALUDE_forgetful_scientist_rain_probability_l1853_185331

/-- The probability of taking an umbrella -/
def umbrella_probability : ℝ := 0.2

/-- The Forgetful Scientist scenario -/
structure ForgetfulScientist where
  /-- The probability of rain -/
  rain_prob : ℝ
  /-- The probability of having no umbrella at the destination -/
  no_umbrella_prob : ℝ
  /-- The condition that the Scientist takes an umbrella if it's raining or there's no umbrella -/
  umbrella_condition : umbrella_probability = rain_prob + no_umbrella_prob - rain_prob * no_umbrella_prob
  /-- The condition that the probabilities are between 0 and 1 -/
  prob_bounds : 0 ≤ rain_prob ∧ rain_prob ≤ 1 ∧ 0 ≤ no_umbrella_prob ∧ no_umbrella_prob ≤ 1

/-- The theorem stating that the probability of rain is 1/9 -/
theorem forgetful_scientist_rain_probability (fs : ForgetfulScientist) : fs.rain_prob = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_forgetful_scientist_rain_probability_l1853_185331


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1853_185306

/-- Given a geometric sequence where the first term is 512 and the 6th term is 8,
    the 4th term is 64. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence definition
  a 0 = 512 →                               -- First term is 512
  a 5 = 8 →                                 -- 6th term is 8
  a 3 = 64 :=                               -- 4th term is 64
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1853_185306


namespace NUMINAMATH_CALUDE_initial_storks_count_l1853_185373

theorem initial_storks_count (initial_birds : ℕ) (additional_birds : ℕ) (final_difference : ℕ) :
  initial_birds = 2 →
  additional_birds = 3 →
  final_difference = 1 →
  initial_birds + additional_birds + final_difference = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_storks_count_l1853_185373


namespace NUMINAMATH_CALUDE_thomson_savings_l1853_185300

def incentive : ℚ := 240

def food_fraction : ℚ := 1/3
def clothes_fraction : ℚ := 1/5
def savings_fraction : ℚ := 3/4

def food_expense : ℚ := food_fraction * incentive
def clothes_expense : ℚ := clothes_fraction * incentive
def total_expense : ℚ := food_expense + clothes_expense
def remaining : ℚ := incentive - total_expense
def savings : ℚ := savings_fraction * remaining

theorem thomson_savings : savings = 84 := by
  sorry

end NUMINAMATH_CALUDE_thomson_savings_l1853_185300


namespace NUMINAMATH_CALUDE_line_y_intercept_l1853_185346

/-- A line passing through the point (-2, 4) with slope 1/2 has a y-intercept of 5 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = (1/2) * x + b) → -- Line equation with slope 1/2 and y-intercept b
  f (-2) = 4 →                 -- Line passes through (-2, 4)
  b = 5 :=                     -- y-intercept is 5
by
  sorry


end NUMINAMATH_CALUDE_line_y_intercept_l1853_185346


namespace NUMINAMATH_CALUDE_function_identity_l1853_185344

theorem function_identity (f : ℕ → ℕ) (h : ∀ x y : ℕ, f (f x + f y) = x + y) :
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l1853_185344


namespace NUMINAMATH_CALUDE_ryan_chinese_learning_hours_l1853_185341

/-- The number of hours Ryan spends on learning English daily -/
def hours_english : ℕ := 6

/-- The number of hours Ryan spends on learning Chinese daily -/
def hours_chinese : ℕ := sorry

/-- The difference in hours between English and Chinese learning -/
def hour_difference : ℕ := 4

theorem ryan_chinese_learning_hours :
  hours_chinese = hours_english - hour_difference := by sorry

end NUMINAMATH_CALUDE_ryan_chinese_learning_hours_l1853_185341


namespace NUMINAMATH_CALUDE_range_of_a_l1853_185367

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x - a ≥ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ (Set.univ \ B a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1853_185367


namespace NUMINAMATH_CALUDE_cookie_recipe_total_cups_l1853_185386

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the cups of sugar used -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem stating that for the given recipe ratio and sugar amount, the total cups is 18 -/
theorem cookie_recipe_total_cups :
  let ratio := RecipeRatio.mk 1 2 3
  let sugarCups := 9
  totalCups ratio sugarCups = 18 := by
  sorry

#check cookie_recipe_total_cups

end NUMINAMATH_CALUDE_cookie_recipe_total_cups_l1853_185386


namespace NUMINAMATH_CALUDE_particular_number_multiplication_l1853_185336

theorem particular_number_multiplication (x : ℤ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_multiplication_l1853_185336


namespace NUMINAMATH_CALUDE_division_remainder_l1853_185343

theorem division_remainder (n : ℕ) : 
  (n / 8 = 8 ∧ n % 8 = 0) → n % 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1853_185343


namespace NUMINAMATH_CALUDE_r_value_when_m_is_3_l1853_185384

theorem r_value_when_m_is_3 :
  let m : ℕ := 3
  let t : ℕ := 3^m + 2
  let r : ℕ := 5^t - 2*t
  r = 5^29 - 58 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_m_is_3_l1853_185384


namespace NUMINAMATH_CALUDE_car_speed_equation_l1853_185302

/-- Given a car traveling at speed v km/h, prove that v satisfies the equation
    v = 3600 / 49, if it takes 4 seconds longer to travel 1 km at speed v
    than at 80 km/h. -/
theorem car_speed_equation (v : ℝ) : v > 0 →
  (3600 / v = 3600 / 80 + 4) → v = 3600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_equation_l1853_185302


namespace NUMINAMATH_CALUDE_expression_evaluation_l1853_185362

theorem expression_evaluation : (-7)^3 / 7^2 - 4^4 + 5^2 = -238 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1853_185362


namespace NUMINAMATH_CALUDE_correct_probability_distribution_l1853_185332

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of cookie types -/
def num_cookie_types : ℕ := 3

/-- Represents the total number of cookies -/
def total_cookies : ℕ := num_students * num_cookie_types

/-- Represents the number of cookies of each type -/
def cookies_per_type : ℕ := num_students

/-- Calculates the probability of each student receiving one cookie of each type -/
def probability_all_students_correct_distribution : ℚ :=
  144 / 3850

/-- Theorem stating that the calculated probability is correct -/
theorem correct_probability_distribution :
  probability_all_students_correct_distribution = 144 / 3850 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_distribution_l1853_185332


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1853_185361

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1853_185361


namespace NUMINAMATH_CALUDE_school_relationship_l1853_185391

/-- In a school with teachers and students, prove the relationship between the number of teachers,
    students, students per teacher, and common teachers between any two students. -/
theorem school_relationship (m n k l : ℕ) : 
  (∀ (teacher : Fin m), ∃! (students : Finset (Fin n)), students.card = k) →
  (∀ (student1 student2 : Fin n), student1 ≠ student2 → 
    ∃! (common_teachers : Finset (Fin m)), common_teachers.card = l) →
  m * k * (k - 1) = n * (n - 1) * l := by
  sorry

end NUMINAMATH_CALUDE_school_relationship_l1853_185391


namespace NUMINAMATH_CALUDE_coffee_needed_l1853_185334

/-- The amount of coffee needed for Taylor's house guests -/
theorem coffee_needed (cups_weak cups_strong : ℕ) 
  (h1 : cups_weak = cups_strong)
  (h2 : cups_weak + cups_strong = 24) : ℕ :=
by
  sorry

#check coffee_needed

end NUMINAMATH_CALUDE_coffee_needed_l1853_185334


namespace NUMINAMATH_CALUDE_train_passing_time_l1853_185333

/-- The time taken for a train to pass a stationary point -/
theorem train_passing_time (length : ℝ) (speed_kmh : ℝ) : 
  length = 280 → speed_kmh = 36 → 
  (length / (speed_kmh * 1000 / 3600)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1853_185333


namespace NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l1853_185372

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (2*m - 8) + (m - 2)*Complex.I

-- Define what it means for a complex number to be pure imaginary
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem pure_imaginary_m_equals_four :
  ∃ m : ℝ, isPureImaginary (z m) → m = 4 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l1853_185372


namespace NUMINAMATH_CALUDE_min_days_team_a_is_ten_l1853_185313

/-- Represents the greening project parameters and constraints -/
structure GreeningProject where
  totalArea : ℝ
  teamARate : ℝ
  teamBRate : ℝ
  teamADailyCost : ℝ
  teamBDailyCost : ℝ
  totalBudget : ℝ

/-- Calculates the minimum number of days Team A should work -/
def minDaysTeamA (project : GreeningProject) : ℝ :=
  sorry

/-- Theorem stating the minimum number of days Team A should work -/
theorem min_days_team_a_is_ten (project : GreeningProject) :
  project.totalArea = 1800 ∧
  project.teamARate = 2 * project.teamBRate ∧
  400 / project.teamARate + 4 = 400 / project.teamBRate ∧
  project.teamADailyCost = 0.4 ∧
  project.teamBDailyCost = 0.25 ∧
  project.totalBudget = 8 →
  minDaysTeamA project = 10 := by
  sorry

#check min_days_team_a_is_ten

end NUMINAMATH_CALUDE_min_days_team_a_is_ten_l1853_185313


namespace NUMINAMATH_CALUDE_star_equation_solution_l1853_185340

/-- Define the ⋆ operation -/
def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem: If a ⋆ 4 = 17, then a = 49/3 -/
theorem star_equation_solution (a : ℝ) (h : star a 4 = 17) : a = 49/3 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1853_185340


namespace NUMINAMATH_CALUDE_final_balance_calculation_l1853_185321

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def withdrawal : ℕ := 4

theorem final_balance_calculation : 
  initial_balance + deposit - withdrawal = 76 := by sorry

end NUMINAMATH_CALUDE_final_balance_calculation_l1853_185321


namespace NUMINAMATH_CALUDE_percentage_subtracted_from_b_l1853_185301

theorem percentage_subtracted_from_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧
  a / b = 4 / 5 ∧
  x = a + 0.75 * a ∧
  m = b - (p / 100) * b ∧
  m / x = 0.14285714285714285 →
  p = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_subtracted_from_b_l1853_185301


namespace NUMINAMATH_CALUDE_expression_value_l1853_185388

theorem expression_value :
  let a : ℝ := 10
  let b : ℝ := 4
  let c : ℝ := 3
  (a - (b - c^2)) - ((a - b) - c^2) = 18 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1853_185388


namespace NUMINAMATH_CALUDE_system_solution_l1853_185353

theorem system_solution (x y u v : ℝ) : 
  x^2 + y^2 + u^2 + v^2 = 4 →
  x * y * u + y * u * v + u * v * x + v * x * y = -2 →
  x * y * u * v = -1 →
  ((x = 1 ∧ y = 1 ∧ u = 1 ∧ v = -1) ∨
   (x = 1 ∧ y = 1 ∧ u = -1 ∧ v = 1) ∨
   (x = 1 ∧ y = -1 ∧ u = 1 ∧ v = 1) ∨
   (x = -1 ∧ y = 1 ∧ u = 1 ∧ v = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1853_185353


namespace NUMINAMATH_CALUDE_die_roll_probability_l1853_185310

/-- A fair six-sided die is rolled six times. -/
def num_rolls : ℕ := 6

/-- The probability of rolling a 5 or 6 on a fair six-sided die. -/
def prob_success : ℚ := 1/3

/-- The probability of not rolling a 5 or 6 on a fair six-sided die. -/
def prob_failure : ℚ := 1 - prob_success

/-- The number of successful outcomes we're interested in (at least 5 times). -/
def min_successes : ℕ := 5

/-- Calculates the binomial coefficient (n choose k). -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Calculates the probability of exactly k successes in n trials. -/
def prob_exactly (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1-p)^(n-k)

/-- The main theorem to prove. -/
theorem die_roll_probability : 
  prob_exactly num_rolls min_successes prob_success + 
  prob_exactly num_rolls num_rolls prob_success = 13/729 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1853_185310


namespace NUMINAMATH_CALUDE_f_properties_l1853_185379

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + 1) * Real.exp x

theorem f_properties :
  ∀ a : ℝ,
  (∃ x_min : ℝ, ∀ x : ℝ, f 0 x_min ≤ f 0 x ∧ f 0 x_min = -Real.exp (-2)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ →
    (a < 0 → (x₂ < -2 ∨ x₂ > -1/a) → f a x₁ > f a x₂) ∧
    (a < 0 → -2 < x₁ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
    (a = 0 → x₂ < -2 → f a x₁ > f a x₂) ∧
    (a = 0 → -2 < x₁ → f a x₁ < f a x₂) ∧
    (0 < a ∧ a < 1/2 → -1/a < x₁ ∧ x₂ < -2 → f a x₁ > f a x₂) ∧
    (0 < a ∧ a < 1/2 → (x₂ < -1/a ∨ -2 < x₁) → f a x₁ < f a x₂) ∧
    (a = 1/2 → f a x₁ < f a x₂) ∧
    (a > 1/2 → -2 < x₁ ∧ x₂ < -1/a → f a x₁ > f a x₂) ∧
    (a > 1/2 → (x₂ < -2 ∨ -1/a < x₁) → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1853_185379


namespace NUMINAMATH_CALUDE_total_beads_is_40_l1853_185369

-- Define the number of blue beads
def blue_beads : ℕ := 5

-- Define the number of red beads as twice the number of blue beads
def red_beads : ℕ := 2 * blue_beads

-- Define the number of white beads as the sum of blue and red beads
def white_beads : ℕ := blue_beads + red_beads

-- Define the number of silver beads
def silver_beads : ℕ := 10

-- Theorem: The total number of beads is 40
theorem total_beads_is_40 : 
  blue_beads + red_beads + white_beads + silver_beads = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_beads_is_40_l1853_185369


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l1853_185323

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_and_perpendicular_properties 
  (a b c : Line) (y : Plane) :
  (∀ a b c, parallel a b → parallel b c → parallel a c) ∧
  (∀ a b, perpendicular a y → perpendicular b y → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l1853_185323


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1853_185318

-- Define the types for lines and planes
variable (L P : Type*)

-- Define the perpendicular relationship between lines
variable (perp_line : L → L → Prop)

-- Define the perpendicular relationship between a line and a plane
variable (perp_line_plane : L → P → Prop)

-- Define the parallel relationship between a line and a plane
variable (parallel : L → P → Prop)

-- Define the subset relationship between a line and a plane
variable (subset : L → P → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : L) (α : P)
  (h1 : perp_line a b)
  (h2 : perp_line_plane b α) :
  subset a α ∨ parallel a α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1853_185318


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1853_185368

/-- The parabola function -/
def f (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- The line function -/
def g (k : ℝ) : ℝ → ℝ := λ _ ↦ k

/-- The condition for a single intersection point -/
def has_single_intersection (k : ℝ) : Prop :=
  ∃! y, f y = g k y

theorem parabola_line_intersection :
  ∀ k, has_single_intersection k ↔ k = 13/3 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1853_185368


namespace NUMINAMATH_CALUDE_triangle_median_equality_l1853_185395

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the length function
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_median_equality (t : Triangle) :
  length t.P t.Q = 2 →
  length t.P t.R = 3 →
  length t.Q t.R = median t t.P →
  length t.Q t.R = Real.sqrt (26 * 0.2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_median_equality_l1853_185395


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l1853_185338

/-- Given a line passing through points (1,3) and (3,7), 
    the sum of its slope and y-intercept is equal to 3. -/
theorem slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) →   -- Line passes through (1,3)
  (7 = m * 3 + b) →   -- Line passes through (3,7)
  m + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l1853_185338


namespace NUMINAMATH_CALUDE_sum_other_vertices_y_equals_14_l1853_185375

structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ

def Rectangle.sumOtherVerticesY (r : Rectangle) : ℝ :=
  r.vertex1.2 + r.vertex2.2

theorem sum_other_vertices_y_equals_14 (r : Rectangle) 
  (h1 : r.vertex1 = (2, 20))
  (h2 : r.vertex2 = (10, -6)) :
  r.sumOtherVerticesY = 14 := by
  sorry

#check sum_other_vertices_y_equals_14

end NUMINAMATH_CALUDE_sum_other_vertices_y_equals_14_l1853_185375


namespace NUMINAMATH_CALUDE_mean_inequalities_l1853_185360

theorem mean_inequalities (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) :
  (x + y + z) / 3 > (x * y * z) ^ (1/3) ∧ (x * y * z) ^ (1/3) > 3 * x * y * z / (x * y + y * z + z * x) :=
by sorry

end NUMINAMATH_CALUDE_mean_inequalities_l1853_185360


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l1853_185335

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.000000007 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -9 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l1853_185335


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1853_185385

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x^2020 > 0) ∧
  (∃ x, x^2020 > 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1853_185385


namespace NUMINAMATH_CALUDE_vector_sum_l1853_185309

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b (y : ℝ) : ℝ × ℝ := (1, y)
def c : ℝ × ℝ := (2, -4)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define parallelism for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

-- Theorem statement
theorem vector_sum (x y : ℝ) 
  (h1 : perpendicular (a x) c) 
  (h2 : parallel (b y) c) : 
  a x + b y = (3, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_l1853_185309


namespace NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l1853_185366

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  4 * (d / Real.sqrt 2) = 8 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l1853_185366


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1853_185347

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f x > f' x + 1)
  (h3 : ∀ x, f x - 2024 = -(f (-x) - 2024)) :
  {x : ℝ | f x - 2023 * exp x < 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1853_185347


namespace NUMINAMATH_CALUDE_pentagon_quadrilateral_angle_sum_l1853_185325

theorem pentagon_quadrilateral_angle_sum :
  ∀ (pentagon_interior_angle quadrilateral_reflex_angle : ℝ),
  pentagon_interior_angle = 108 →
  quadrilateral_reflex_angle = 360 - pentagon_interior_angle →
  360 - quadrilateral_reflex_angle = 108 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_quadrilateral_angle_sum_l1853_185325


namespace NUMINAMATH_CALUDE_cookie_difference_l1853_185329

theorem cookie_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 39 →
  initial_salty = 6 →
  eaten_sweet = 32 →
  eaten_salty = 23 →
  eaten_sweet - eaten_salty = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1853_185329


namespace NUMINAMATH_CALUDE_sound_pressure_relations_l1853_185358

noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

theorem sound_pressure_relations
  (p₀ : ℝ) (hp₀ : p₀ > 0)
  (p₁ p₂ p₃ : ℝ)
  (hp₁ : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (hp₂ : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (hp₃ : sound_pressure_level p₃ p₀ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ :=
by sorry

end NUMINAMATH_CALUDE_sound_pressure_relations_l1853_185358


namespace NUMINAMATH_CALUDE_shoe_ratio_proof_l1853_185316

theorem shoe_ratio_proof (total_shoes brown_shoes : ℕ) 
  (h1 : total_shoes = 66) 
  (h2 : brown_shoes = 22) : 
  (total_shoes - brown_shoes) / brown_shoes = 2 := by
sorry

end NUMINAMATH_CALUDE_shoe_ratio_proof_l1853_185316


namespace NUMINAMATH_CALUDE_intersection_points_form_convex_polygon_l1853_185359

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an L-shaped figure -/
structure LShape where
  A : Point
  longSegment : List Point
  shortSegment : Point

/-- Represents the problem setup -/
structure ProblemSetup where
  L1 : LShape
  L2 : LShape
  n : ℕ
  intersectionPoints : List Point

/-- Predicate to check if a list of points forms a convex polygon -/
def IsConvexPolygon (points : List Point) : Prop := sorry

/-- Main theorem statement -/
theorem intersection_points_form_convex_polygon (setup : ProblemSetup) :
  IsConvexPolygon setup.intersectionPoints :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_convex_polygon_l1853_185359


namespace NUMINAMATH_CALUDE_triangle_properties_main_theorem_l1853_185322

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  cosQ : ℝ

-- Define our specific triangle
def trianglePQR : RightTriangle where
  PQ := 15
  QR := 30  -- We'll prove this
  cosQ := 0.5

-- Theorem to prove QR = 30 and area = 225
theorem triangle_properties (t : RightTriangle) 
  (h1 : t.PQ = 15) 
  (h2 : t.cosQ = 0.5) : 
  t.QR = 30 ∧ (1/2 * t.PQ * t.QR) = 225 := by
  sorry

-- Main theorem combining all properties
theorem main_theorem : 
  trianglePQR.QR = 30 ∧ 
  (1/2 * trianglePQR.PQ * trianglePQR.QR) = 225 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_main_theorem_l1853_185322


namespace NUMINAMATH_CALUDE_hayden_ironing_days_l1853_185357

/-- Given that Hayden spends 8 minutes ironing clothes each day he does so,
    and over 4 weeks he spends 160 minutes ironing,
    prove that he irons his clothes 5 days per week. -/
theorem hayden_ironing_days (minutes_per_day : ℕ) (total_minutes : ℕ) (weeks : ℕ) :
  minutes_per_day = 8 →
  total_minutes = 160 →
  weeks = 4 →
  (total_minutes / weeks) / minutes_per_day = 5 :=
by sorry

end NUMINAMATH_CALUDE_hayden_ironing_days_l1853_185357


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1853_185352

/-- A geometric sequence with first term 1 and product of first three terms -8 has common ratio -2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence property
    a 1 = 1 →                              -- first term is 1
    a 1 * a 2 * a 3 = -8 →                 -- product of first three terms is -8
    a 2 / a 1 = -2 :=                      -- common ratio is -2
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1853_185352


namespace NUMINAMATH_CALUDE_defective_units_shipped_l1853_185330

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.05 →
  shipped_rate = 0.04 →
  (defective_rate * shipped_rate * 100) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l1853_185330


namespace NUMINAMATH_CALUDE_problem_solution_l1853_185396

theorem problem_solution (a b : ℝ) : 
  (|a| = 6 ∧ |b| = 2) →
  (((a * b > 0) → |a + b| = 8) ∧
   ((|a + b| = a + b) → (a - b = 4 ∨ a - b = 8))) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1853_185396


namespace NUMINAMATH_CALUDE_cone_sphere_equal_volume_l1853_185397

theorem cone_sphere_equal_volume (r : ℝ) (h : ℝ) :
  r = 1 →
  (1/3 * π * r^2 * h) = (4/3 * π) →
  Real.sqrt (r^2 + h^2) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_equal_volume_l1853_185397


namespace NUMINAMATH_CALUDE_shekars_math_marks_l1853_185327

def science_marks : ℝ := 65
def social_studies_marks : ℝ := 82
def english_marks : ℝ := 62
def biology_marks : ℝ := 85
def average_marks : ℝ := 74
def number_of_subjects : ℕ := 5

theorem shekars_math_marks :
  ∃ (math_marks : ℝ),
    (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / number_of_subjects = average_marks ∧
    math_marks = 76 := by
  sorry

end NUMINAMATH_CALUDE_shekars_math_marks_l1853_185327


namespace NUMINAMATH_CALUDE_tangent_lines_count_l1853_185311

theorem tangent_lines_count : ∃! (s : Finset ℝ), 
  (∀ x₀ ∈ s, x₀ * Real.exp x₀ * (x₀^2 - x₀ - 4) = 0) ∧ 
  Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l1853_185311


namespace NUMINAMATH_CALUDE_square_difference_l1853_185382

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1853_185382


namespace NUMINAMATH_CALUDE_thursday_tuesday_difference_l1853_185374

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℕ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount : ℕ := 5 * tuesday_amount

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount : ℕ := wednesday_amount + 9

/-- The theorem stating the difference between Thursday's and Tuesday's amounts -/
theorem thursday_tuesday_difference : thursday_amount - tuesday_amount = 41 := by
  sorry

end NUMINAMATH_CALUDE_thursday_tuesday_difference_l1853_185374


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1853_185324

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 15 →
  a * b + c + d = 78 →
  a * d + b * c = 160 →
  c * d = 96 →
  ∃ (max : ℝ), (∀ (a' b' c' d' : ℝ), 
    a' + b' = 15 →
    a' * b' + c' + d' = 78 →
    a' * d' + b' * c' = 160 →
    c' * d' = 96 →
    a'^2 + b'^2 + c'^2 + d'^2 ≤ max) ∧
  max = 717 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1853_185324


namespace NUMINAMATH_CALUDE_first_number_proof_l1853_185315

theorem first_number_proof (N : ℕ) : 
  (∃ k m : ℕ, N = 170 * k + 10 ∧ 875 = 170 * m + 25) →
  N = 860 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l1853_185315


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_and_gcd_l1853_185339

def sumOfDivisors (n : ℕ) : ℕ := sorry

def numberOfDistinctPrimeFactors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors_and_gcd :
  let s := sumOfDivisors 450
  numberOfDistinctPrimeFactors s = 3 ∧ Nat.gcd s 450 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_prime_factors_and_gcd_l1853_185339


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l1853_185305

def word : String := "MISSISSIPPI"

def letter_count (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem mississippi_arrangements :
  (Nat.factorial 11) / 
  (Nat.factorial (letter_count word 'S') * 
   Nat.factorial (letter_count word 'I') * 
   Nat.factorial (letter_count word 'P') * 
   Nat.factorial (letter_count word 'M')) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l1853_185305


namespace NUMINAMATH_CALUDE_fiona_finished_tenth_l1853_185390

/-- Represents a racer in the competition -/
inductive Racer
| Alice
| Ben
| Carlos
| Diana
| Emma
| Fiona

/-- The type of finishing positions -/
def Position := Fin 15

/-- The finishing order of the race -/
def FinishingOrder := Racer → Position

/-- Defines the relative positions of racers -/
def PlacesAhead (fo : FinishingOrder) (r1 r2 : Racer) (n : ℕ) : Prop :=
  (fo r1).val + n = (fo r2).val

/-- Defines the absolute position of a racer -/
def FinishedIn (fo : FinishingOrder) (r : Racer) (p : Position) : Prop :=
  fo r = p

theorem fiona_finished_tenth (fo : FinishingOrder) :
  PlacesAhead fo Racer.Emma Racer.Diana 4 →
  PlacesAhead fo Racer.Carlos Racer.Alice 2 →
  PlacesAhead fo Racer.Diana Racer.Ben 3 →
  PlacesAhead fo Racer.Carlos Racer.Fiona 3 →
  PlacesAhead fo Racer.Emma Racer.Fiona 2 →
  FinishedIn fo Racer.Ben ⟨7, by norm_num⟩ →
  FinishedIn fo Racer.Fiona ⟨10, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_fiona_finished_tenth_l1853_185390


namespace NUMINAMATH_CALUDE_cost_to_replace_movies_l1853_185354

/-- The cost to replace VHS movies with DVDs -/
theorem cost_to_replace_movies 
  (num_movies : ℕ) 
  (vhs_trade_in : ℚ) 
  (dvd_cost : ℚ) : 
  (num_movies : ℚ) * (dvd_cost - vhs_trade_in) = 800 :=
by
  sorry

#check cost_to_replace_movies 100 2 10

end NUMINAMATH_CALUDE_cost_to_replace_movies_l1853_185354


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1853_185371

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1853_185371


namespace NUMINAMATH_CALUDE_mango_purchase_l1853_185349

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The amount of grapes purchased in kg -/
def grape_amount : ℕ := 8

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 1055

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℕ := (total_paid - grape_amount * grape_price) / mango_price

theorem mango_purchase : mango_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_l1853_185349


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2047_l1853_185308

theorem tens_digit_of_13_pow_2047 : ∃ n : ℕ, 13^2047 ≡ 10 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2047_l1853_185308


namespace NUMINAMATH_CALUDE_wheel_probability_l1853_185320

theorem wheel_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 1/4 → p_E = 1/3 → p_G = 1/6 → 
  p_D + p_E + p_F + p_G = 1 →
  p_F = 1/4 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l1853_185320


namespace NUMINAMATH_CALUDE_haley_tv_time_l1853_185342

/-- The total hours Haley watched TV over the weekend -/
def total_hours (saturday_hours sunday_hours : ℕ) : ℕ :=
  saturday_hours + sunday_hours

/-- Theorem stating that Haley watched 9 hours of TV total -/
theorem haley_tv_time : total_hours 6 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_haley_tv_time_l1853_185342


namespace NUMINAMATH_CALUDE_stratified_sampling_girls_l1853_185351

theorem stratified_sampling_girls (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : girls_in_sample = 103) :
  (girls_in_sample : ℚ) / sample_size * total_students = 970 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_girls_l1853_185351


namespace NUMINAMATH_CALUDE_problem_statement_l1853_185345

theorem problem_statement (n : ℝ) : 
  (n - 2009)^2 + (2008 - n)^2 = 1 → (n - 2009) * (2008 - n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1853_185345


namespace NUMINAMATH_CALUDE_inscribed_cylinder_properties_l1853_185380

/-- An equilateral cylinder inscribed in a regular tetrahedron --/
structure InscribedCylinder where
  a : ℝ  -- Edge length of the tetrahedron
  r : ℝ  -- Radius of the cylinder
  h : ℝ  -- Height of the cylinder
  cylinder_equilateral : h = 2 * r
  cylinder_inscribed : r = (a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6

/-- Theorem about the properties of the inscribed cylinder --/
theorem inscribed_cylinder_properties (c : InscribedCylinder) :
  c.r = (c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6 ∧
  (4 * Real.pi * c.r^2 : ℝ) = 4 * Real.pi * ((c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6)^2 ∧
  (2 * Real.pi * c.r^3 : ℝ) = 2 * Real.pi * ((c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6)^3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_properties_l1853_185380


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1853_185383

theorem angle_sum_around_point (y : ℝ) : 
  3 * y + 6 * y + 2 * y + 4 * y + y = 360 → y = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1853_185383


namespace NUMINAMATH_CALUDE_solve_triangle_problem_l1853_185377

/-- Represents a right-angled isosceles triangle --/
structure RightIsoscelesTriangle where
  side : ℝ
  area : ℝ
  area_eq : area = side^2 / 2

/-- The problem setup --/
def triangle_problem (k : ℝ) : Prop :=
  let t1 := RightIsoscelesTriangle.mk k (k^2 / 2) (by rfl)
  let t2 := RightIsoscelesTriangle.mk (k * Real.sqrt 2) (k^2) (by sorry)
  let t3 := RightIsoscelesTriangle.mk (2 * k) (2 * k^2) (by sorry)
  t1.area + t2.area + t3.area = 56

/-- The theorem to prove --/
theorem solve_triangle_problem : 
  ∃ k : ℝ, triangle_problem k ∧ k = 4 := by sorry

end NUMINAMATH_CALUDE_solve_triangle_problem_l1853_185377


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1853_185314

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1853_185314


namespace NUMINAMATH_CALUDE_cube_root_seven_to_sixth_l1853_185312

theorem cube_root_seven_to_sixth (x : ℝ) : x = 7^(1/3) → x^6 = 49 := by sorry

end NUMINAMATH_CALUDE_cube_root_seven_to_sixth_l1853_185312


namespace NUMINAMATH_CALUDE_log_inequality_l1853_185319

theorem log_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.log (1 + x + y) < x + y := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1853_185319


namespace NUMINAMATH_CALUDE_collins_earnings_per_can_l1853_185393

/-- The amount of money Collin earns per aluminum can -/
def earnings_per_can (cans_home : ℕ) (cans_grandparents_multiplier : ℕ) (cans_neighbor : ℕ) (cans_dad_office : ℕ) (savings_amount : ℚ) : ℚ :=
  let total_cans := cans_home + cans_home * cans_grandparents_multiplier + cans_neighbor + cans_dad_office
  let total_earnings := 2 * savings_amount
  total_earnings / total_cans

/-- Theorem stating that Collin earns $0.25 per aluminum can -/
theorem collins_earnings_per_can :
  earnings_per_can 12 3 46 250 43 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_collins_earnings_per_can_l1853_185393


namespace NUMINAMATH_CALUDE_square_root_divided_by_six_l1853_185378

theorem square_root_divided_by_six : Real.sqrt 144 / 6 = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_six_l1853_185378


namespace NUMINAMATH_CALUDE_jay_change_calculation_l1853_185350

/-- Calculates the change Jay received after purchasing items with a discount --/
theorem jay_change_calculation (book pen ruler notebook pencil_case : ℚ)
  (h_book : book = 25)
  (h_pen : pen = 4)
  (h_ruler : ruler = 1)
  (h_notebook : notebook = 8)
  (h_pencil_case : pencil_case = 6)
  (discount_rate : ℚ)
  (h_discount : discount_rate = 0.1)
  (paid_amount : ℚ)
  (h_paid : paid_amount = 100) :
  let total_before_discount := book + pen + ruler + notebook + pencil_case
  let discount_amount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  paid_amount - total_after_discount = 60.4 := by
sorry

end NUMINAMATH_CALUDE_jay_change_calculation_l1853_185350


namespace NUMINAMATH_CALUDE_snow_probability_l1853_185389

theorem snow_probability (p1 p2 p3 : ℚ) (n1 n2 n3 : ℕ) : 
  p1 = 1/3 →
  p2 = 1/4 →
  p3 = 1/2 →
  n1 = 3 →
  n2 = 4 →
  n3 = 3 →
  1 - (1 - p1)^n1 * (1 - p2)^n2 * (1 - p3)^n3 = 2277/2304 := by
sorry

end NUMINAMATH_CALUDE_snow_probability_l1853_185389


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l1853_185307

theorem expansion_coefficient_sum (n : ℕ) : 
  (∀ a b : ℝ, (3*a + 5*b)^n = 2^15) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l1853_185307


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1853_185328

/-- Given a triangle with inradius 2.5 cm and area 30 cm², its perimeter is 24 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 30 → A = r * (p / 2) → p = 24 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1853_185328


namespace NUMINAMATH_CALUDE_election_winner_margin_l1853_185392

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) : 
  (2 : ℕ) ≤ total_votes →
  winner_votes = (75 * total_votes) / 100 →
  winner_votes = 750 →
  winner_votes - (total_votes - winner_votes) = 500 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_margin_l1853_185392


namespace NUMINAMATH_CALUDE_coat_price_calculation_l1853_185399

/-- Calculates the final price of a coat after discounts and tax -/
def finalPrice (originalPrice : ℝ) (initialDiscount : ℝ) (additionalDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterInitialDiscount := originalPrice * (1 - initialDiscount)
  let priceAfterAdditionalDiscount := priceAfterInitialDiscount - additionalDiscount
  priceAfterAdditionalDiscount * (1 + salesTax)

/-- Theorem stating that the final price of the coat is $112.75 -/
theorem coat_price_calculation :
  finalPrice 150 0.25 10 0.1 = 112.75 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_l1853_185399


namespace NUMINAMATH_CALUDE_modulus_of_z_l1853_185303

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z(1-i) = 2i
def condition : Prop := z * (1 - Complex.I) = 2 * Complex.I

-- Theorem statement
theorem modulus_of_z (h : condition z) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1853_185303


namespace NUMINAMATH_CALUDE_range_of_a_l1853_185364

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1853_185364


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1853_185363

theorem trigonometric_identities (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/3))
  (h2 : Real.sqrt 6 * Real.sin α + Real.sqrt 2 * Real.cos α = Real.sqrt 3) :
  (Real.cos (α + π/6) = Real.sqrt 10 / 4) ∧
  (Real.cos (2*α + π/12) = (Real.sqrt 30 + Real.sqrt 2) / 8) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1853_185363


namespace NUMINAMATH_CALUDE_outfit_combinations_l1853_185387

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (hats : ℕ) : 
  shirts = 4 → pants = 5 → hats = 3 → shirts * pants * hats = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1853_185387


namespace NUMINAMATH_CALUDE_product_of_powers_equals_fifty_l1853_185317

theorem product_of_powers_equals_fifty :
  (5^(2/10)) * (10^(4/10)) * (10^(1/10)) * (10^(5/10)) * (5^(8/10)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_fifty_l1853_185317


namespace NUMINAMATH_CALUDE_sum_always_positive_l1853_185304

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : is_monotone_increasing f)
  (h2 : is_odd_function f)
  (h3 : arithmetic_sequence a)
  (h4 : a 1 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l1853_185304


namespace NUMINAMATH_CALUDE_division_in_ratio_l1853_185326

theorem division_in_ratio (total : ℕ) (ratio_b ratio_c : ℕ) (amount_c : ℕ) : 
  total = 2000 →
  ratio_b = 4 →
  ratio_c = 16 →
  amount_c = total * ratio_c / (ratio_b + ratio_c) →
  amount_c = 1600 := by
sorry

end NUMINAMATH_CALUDE_division_in_ratio_l1853_185326


namespace NUMINAMATH_CALUDE_parallelogram_below_line_l1853_185365

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

def BelowOrOnLine (p : Point) (y0 : ℝ) : Prop :=
  p.y ≤ y0

theorem parallelogram_below_line :
  let A : Point := ⟨4, 2⟩
  let B : Point := ⟨-2, -4⟩
  let C : Point := ⟨-8, -4⟩
  let D : Point := ⟨0, 4⟩
  let y0 : ℝ := -2
  Parallelogram A B C D →
  ∀ p : Point, (∃ t u : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ u ∧ u ≤ 1 ∧
    p.x = A.x + t * (B.x - A.x) + u * (D.x - A.x) ∧
    p.y = A.y + t * (B.y - A.y) + u * (D.y - A.y)) →
  BelowOrOnLine p y0 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_below_line_l1853_185365


namespace NUMINAMATH_CALUDE_ellipse_symmetry_l1853_185376

-- Define the original ellipse
def original_ellipse (x y : ℝ) : Prop :=
  (x - 3)^2 / 9 + (y - 2)^2 / 4 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

-- Define the reflection transformation
def reflect (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

-- Define the resulting ellipse C
def ellipse_c (x y : ℝ) : Prop :=
  (x + 2)^2 / 9 + (y + 3)^2 / 4 = 1

-- Theorem statement
theorem ellipse_symmetry :
  ∀ (x y : ℝ),
    original_ellipse x y →
    let (x', y') := reflect x y
    ellipse_c x' y' :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetry_l1853_185376


namespace NUMINAMATH_CALUDE_ball_drawing_probability_l1853_185355

-- Define the sample space
def Ω : Type := Fin 4 × Fin 3

-- Define the events
def A : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 < 1) ∨ (ω.1 ≥ 2 ∧ ω.2 ≥ 1)}
def B : Set Ω := {ω | ω.1 < 2}
def C : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 < 1) ∨ (ω.1 ≥ 2 ∧ ω.2 = 1)}
def D : Set Ω := {ω | (ω.1 < 2 ∧ ω.2 ≥ 1) ∨ (ω.1 ≥ 2 ∧ ω.2 < 1)}

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem ball_drawing_probability :
  (P A + P D = 1) ∧
  (P (A ∩ B) = P A * P B) ∧
  (P (C ∩ D) = P C * P D) := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_probability_l1853_185355


namespace NUMINAMATH_CALUDE_expression_is_integer_l1853_185398

theorem expression_is_integer (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  ∃ k : ℤ, k = (x^n / ((x-y)*(x-z))) + (y^n / ((y-x)*(y-z))) + (z^n / ((z-x)*(z-y))) :=
by sorry

end NUMINAMATH_CALUDE_expression_is_integer_l1853_185398


namespace NUMINAMATH_CALUDE_markers_problem_l1853_185394

/-- Given the initial number of markers, the number of markers in each new box,
    and the final number of markers, prove that the number of new boxes bought is 6. -/
theorem markers_problem (initial_markers final_markers markers_per_box : ℕ)
  (h1 : initial_markers = 32)
  (h2 : final_markers = 86)
  (h3 : markers_per_box = 9) :
  (final_markers - initial_markers) / markers_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_markers_problem_l1853_185394


namespace NUMINAMATH_CALUDE_diana_hit_eight_l1853_185370

-- Define the set of players
inductive Player : Type
| Alex | Betsy | Carlos | Diana | Edward | Fiona

-- Define the function that maps players to their scores
def score : Player → ℕ
| Player.Alex => 18
| Player.Betsy => 5
| Player.Carlos => 12
| Player.Diana => 14
| Player.Edward => 19
| Player.Fiona => 11

-- Define the function that determines if a player hit a specific score
def hit_score (p : Player) (s : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = score p ∧ a ≠ b ∧ (a = s ∨ b = s) ∧ 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12

-- Define the theorem
theorem diana_hit_eight :
  (∀ p : Player, ∀ s : ℕ, hit_score p s → s ≠ 8) ∨ hit_score Player.Diana 8 :=
sorry

end NUMINAMATH_CALUDE_diana_hit_eight_l1853_185370


namespace NUMINAMATH_CALUDE_equation_roots_l1853_185337

/-- The equation has at least two distinct roots if and only if a = 20 -/
theorem equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    a^2 * (x - 2) + a * (39 - 20*x) + 20 = 0 ∧ 
    a^2 * (y - 2) + a * (39 - 20*y) + 20 = 0) ↔ 
  a = 20 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l1853_185337


namespace NUMINAMATH_CALUDE_single_plane_division_two_planes_division_l1853_185348

-- Define a type for space
structure Space :=
  (points : Set Point)

-- Define a type for plane
structure Plane :=
  (equation : Point → Prop)

-- Define a function to count the number of parts a set of planes divides space into
def countParts (space : Space) (planes : Set Plane) : ℕ :=
  sorry

-- Theorem for a single plane
theorem single_plane_division (space : Space) (plane : Plane) :
  countParts space {plane} = 2 :=
sorry

-- Theorem for two planes
theorem two_planes_division (space : Space) (plane1 plane2 : Plane) :
  countParts space {plane1, plane2} = 3 ∨ countParts space {plane1, plane2} = 4 :=
sorry

end NUMINAMATH_CALUDE_single_plane_division_two_planes_division_l1853_185348


namespace NUMINAMATH_CALUDE_lily_book_count_l1853_185356

/-- The number of books Lily read last month -/
def last_month_books : ℕ := 4

/-- The number of books Lily plans to read this month -/
def this_month_books : ℕ := 2 * last_month_books

/-- The total number of books Lily will read over two months -/
def total_books : ℕ := last_month_books + this_month_books

theorem lily_book_count : total_books = 12 := by
  sorry

end NUMINAMATH_CALUDE_lily_book_count_l1853_185356


namespace NUMINAMATH_CALUDE_rogers_first_bag_l1853_185381

/-- Represents the number of candy bags a person has -/
def num_bags : ℕ := 2

/-- Represents the number of pieces in each of Sandra's bags -/
def sandra_pieces_per_bag : ℕ := 6

/-- Represents the number of pieces in Roger's second bag -/
def roger_second_bag : ℕ := 3

/-- Represents the difference in total pieces between Roger and Sandra -/
def roger_sandra_diff : ℕ := 2

/-- Represents the number of pieces in one of Roger's bags -/
def roger_one_bag : ℕ := 11

/-- Calculates the total number of candy pieces Sandra has -/
def sandra_total : ℕ := num_bags * sandra_pieces_per_bag

/-- Calculates the total number of candy pieces Roger has -/
def roger_total : ℕ := sandra_total + roger_sandra_diff

/-- Theorem: The number of pieces in Roger's first bag is 11 -/
theorem rogers_first_bag : roger_total - roger_second_bag = roger_one_bag :=
by sorry

end NUMINAMATH_CALUDE_rogers_first_bag_l1853_185381
