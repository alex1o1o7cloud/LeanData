import Mathlib

namespace line_AB_equation_line_l_equations_l702_70267

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the midpoint of AB
def midpoint_AB (x y : ℝ) : Prop := x = 3 ∧ y = 2

-- Define the point (2, 0) on line l
def point_on_l (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the area of triangle OMN
def area_OMN : ℝ := 6

-- Theorem for the equation of line AB
theorem line_AB_equation (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → midpoint_AB ((A.1 + B.1)/2) ((A.2 + B.2)/2) →
  ∃ (k : ℝ), A.1 - A.2 - k = 0 ∧ B.1 - B.2 - k = 0 :=
sorry

-- Theorem for the equations of line l
theorem line_l_equations :
  ∃ (M N : ℝ × ℝ), parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
  point_on_l 2 0 ∧
  (∃ (m : ℝ), (M.2 = m*M.1 - 2 ∧ N.2 = m*N.1 - 2) ∨
              (M.2 = -m*M.1 - 2 ∧ N.2 = -m*N.1 - 2)) ∧
  area_OMN = 6 :=
sorry

end line_AB_equation_line_l_equations_l702_70267


namespace benny_state_tax_l702_70274

/-- Calculates the total state tax in cents per hour given an hourly wage in dollars, a tax rate percentage, and a fixed tax in cents. -/
def total_state_tax (hourly_wage : ℚ) (tax_rate_percent : ℚ) (fixed_tax_cents : ℕ) : ℕ :=
  sorry

/-- Proves that given Benny's hourly wage of $25, a 2% state tax rate, and a fixed tax of 50 cents per hour, the total amount of state taxes paid per hour is 100 cents. -/
theorem benny_state_tax :
  total_state_tax 25 2 50 = 100 := by
  sorry

end benny_state_tax_l702_70274


namespace floor_sqrt_50_l702_70265

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end floor_sqrt_50_l702_70265


namespace triangle_problem_l702_70238

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = 3 * Real.sqrt 13 / 13 ∧
  Real.sin (2 * A + π/4) = 7 * Real.sqrt 2 / 26 :=
by sorry

end triangle_problem_l702_70238


namespace simplify_fraction_product_l702_70280

theorem simplify_fraction_product : (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end simplify_fraction_product_l702_70280


namespace quadratic_equation_solution_l702_70221

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -3/2 ∧ 
  (∀ x : ℝ, 2*x^2 + 5*x + 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_solution_l702_70221


namespace sequence_properties_l702_70290

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = b n - 1

theorem sequence_properties
  (a b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : sum_of_terms b S)
  (h_a2b1 : a 2 = b 1)
  (h_a5b2 : a 5 = b 2) :
  (∀ n : ℕ, a n = 2 * n - 6) ∧
  (∀ n : ℕ, S n = (-2)^n - 1) :=
sorry

end sequence_properties_l702_70290


namespace impossible_belief_l702_70295

-- Define the characters
inductive Character : Type
| King : Character
| Queen : Character

-- Define the state of mind
inductive MindState : Type
| Sane : MindState
| NotSane : MindState

-- Define a belief
structure Belief where
  subject : Character
  object : Character
  state : MindState

-- Define a nested belief
structure NestedBelief where
  level1 : Character
  level2 : Character
  level3 : Character
  finalBelief : Belief

-- Define logical consistency
def logicallyConsistent (c : Character) : Prop :=
  ∀ (b : Belief), b.subject = c → (b.state = MindState.Sane ↔ c = b.object)

-- Define the problematic belief
def problematicBelief : NestedBelief :=
  { level1 := Character.King
  , level2 := Character.Queen
  , level3 := Character.King
  , finalBelief := { subject := Character.King
                   , object := Character.Queen
                   , state := MindState.NotSane } }

-- Theorem statement
theorem impossible_belief
  (h1 : logicallyConsistent Character.King)
  (h2 : logicallyConsistent Character.Queen) :
  ¬ (∃ (b : NestedBelief), b = problematicBelief) :=
sorry

end impossible_belief_l702_70295


namespace linear_function_m_greater_than_one_l702_70206

/-- A linear function y = (m+1)x + (m-1) whose graph passes through the first, second, and third quadrants -/
structure LinearFunction (m : ℝ) :=
  (passes_through_quadrants : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (x₂ < 0 ∧ y₂ > 0) ∧  -- Second quadrant
    (x₃ < 0 ∧ y₃ < 0) ∧  -- Third quadrant
    y₁ = (m + 1) * x₁ + (m - 1) ∧
    y₂ = (m + 1) * x₂ + (m - 1) ∧
    y₃ = (m + 1) * x₃ + (m - 1))

/-- Theorem: If a linear function y = (m+1)x + (m-1) has a graph that passes through
    the first, second, and third quadrants, then m > 1 -/
theorem linear_function_m_greater_than_one (m : ℝ) (f : LinearFunction m) : m > 1 := by
  sorry

end linear_function_m_greater_than_one_l702_70206


namespace additional_workers_needed_additional_workers_theorem_l702_70255

theorem additional_workers_needed (initial_workers : ℕ) (initial_parts : ℕ) (initial_hours : ℕ)
  (target_parts : ℕ) (target_hours : ℕ) : ℕ :=
  let production_rate := initial_parts / (initial_workers * initial_hours)
  let required_workers := (target_parts / target_hours) / production_rate
  required_workers - initial_workers

theorem additional_workers_theorem :
  additional_workers_needed 4 108 3 504 8 = 3 := by
  sorry

end additional_workers_needed_additional_workers_theorem_l702_70255


namespace bakers_purchase_cost_l702_70294

/-- Calculate the total cost in dollars after discount for a baker's purchase -/
theorem bakers_purchase_cost (flour_price : ℝ) (egg_price : ℝ) (milk_price : ℝ) (soda_price : ℝ)
  (discount_rate : ℝ) (exchange_rate : ℝ) :
  flour_price = 6 →
  egg_price = 12 →
  milk_price = 3 →
  soda_price = 1.5 →
  discount_rate = 0.15 →
  exchange_rate = 1.2 →
  let total_euro := 5 * flour_price + 6 * egg_price + 8 * milk_price + 4 * soda_price
  let discounted_euro := total_euro * (1 - discount_rate)
  let total_dollar := discounted_euro * exchange_rate
  total_dollar = 134.64 := by
sorry

end bakers_purchase_cost_l702_70294


namespace inequality_system_solution_l702_70249

theorem inequality_system_solution (x : ℝ) :
  (3 * (x - 1) < 5 * x + 11 ∧ 2 * x > (9 - x) / 4) ↔ x > 1 := by
  sorry

end inequality_system_solution_l702_70249


namespace businessmen_who_drank_nothing_l702_70288

/-- The number of businessmen who drank neither coffee, tea, nor soda -/
theorem businessmen_who_drank_nothing (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_and_tea tea_and_soda coffee_and_soda : ℕ) (all_three : ℕ) : 
  total = 40 →
  coffee = 20 →
  tea = 15 →
  soda = 10 →
  coffee_and_tea = 8 →
  tea_and_soda = 4 →
  coffee_and_soda = 3 →
  all_three = 2 →
  total - (coffee + tea + soda - coffee_and_tea - tea_and_soda - coffee_and_soda + all_three) = 8 := by
  sorry

end businessmen_who_drank_nothing_l702_70288


namespace average_of_solutions_l702_70208

variable (b : ℝ)

def quadratic_equation (x : ℝ) : Prop :=
  3 * x^2 - 6 * b * x + 2 * b = 0

def has_two_real_solutions : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation b x₁ ∧ quadratic_equation b x₂

theorem average_of_solutions :
  has_two_real_solutions b →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation b x₁ ∧ 
    quadratic_equation b x₂ ∧
    (x₁ + x₂) / 2 = b :=
by sorry

end average_of_solutions_l702_70208


namespace race_track_length_l702_70291

/-- Represents a runner in the race --/
structure Runner where
  position : ℝ
  velocity : ℝ

/-- Represents the race --/
structure Race where
  track_length : ℝ
  alberto : Runner
  bernardo : Runner
  carlos : Runner

/-- The conditions of the race --/
def race_conditions (r : Race) : Prop :=
  r.alberto.velocity > 0 ∧
  r.bernardo.velocity > 0 ∧
  r.carlos.velocity > 0 ∧
  r.alberto.position = r.track_length ∧
  r.bernardo.position = r.track_length - 36 ∧
  r.carlos.position = r.track_length - 46 ∧
  (r.track_length / r.bernardo.velocity) * r.carlos.velocity = r.track_length - 16

theorem race_track_length (r : Race) (h : race_conditions r) : r.track_length = 96 := by
  sorry

#check race_track_length

end race_track_length_l702_70291


namespace salt_solution_concentration_l702_70235

/-- Proves that the concentration of a salt solution is 75% when mixed with pure water to form a 15% solution -/
theorem salt_solution_concentration 
  (water_volume : ℝ) 
  (salt_solution_volume : ℝ) 
  (mixture_concentration : ℝ) 
  (h1 : water_volume = 1) 
  (h2 : salt_solution_volume = 0.25) 
  (h3 : mixture_concentration = 15) : 
  (mixture_concentration * (water_volume + salt_solution_volume)) / salt_solution_volume = 75 := by
  sorry

end salt_solution_concentration_l702_70235


namespace quadratic_function_unique_l702_70246

/-- A quadratic function that intersects the x-axis at (-1, 0) and (2, 0), and the y-axis at (0, -2) -/
def f (x : ℝ) : ℝ := x^2 - x - 2

/-- The theorem stating that f is the unique quadratic function satisfying the given conditions -/
theorem quadratic_function_unique :
  (f (-1) = 0) ∧ 
  (f 2 = 0) ∧ 
  (f 0 = -2) ∧ 
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ g : ℝ → ℝ, (g (-1) = 0) → (g 2 = 0) → (g 0 = -2) → 
    (∀ x : ℝ, ∃ a b c : ℝ, g x = a * x^2 + b * x + c) → 
    (∀ x : ℝ, g x = f x)) :=
by sorry

end quadratic_function_unique_l702_70246


namespace trig_product_equals_one_l702_70229

theorem trig_product_equals_one :
  let sin30 : ℝ := 1/2
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let cos60 : ℝ := 1/2
  (1 - 1/sin30) * (1 + 1/cos60) * (1 - 1/cos30) * (1 + 1/sin60) = 1 := by
  sorry

end trig_product_equals_one_l702_70229


namespace unpainted_cubes_4x4x4_l702_70259

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  total_cubes : ℕ
  painted_per_face : ℕ

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : ℕ :=
  cube.total_cubes - (6 * cube.painted_per_face - 12)

/-- Theorem stating that a 4x4x4 cube with 6 painted squares per face has 40 unpainted unit cubes -/
theorem unpainted_cubes_4x4x4 :
  let cube : PaintedCube := ⟨4, 64, 6⟩
  unpainted_cubes cube = 40 := by sorry

end unpainted_cubes_4x4x4_l702_70259


namespace ball_bounce_count_l702_70276

/-- The number of bounces required for a ball dropped from 16 feet to reach a height less than 2 feet,
    when it bounces back up two-thirds the distance it just fell. -/
theorem ball_bounce_count : ∃ k : ℕ, 
  (∀ n < k, 16 * (2/3)^n ≥ 2) ∧ 
  16 * (2/3)^k < 2 ∧
  k = 6 := by
  sorry

end ball_bounce_count_l702_70276


namespace minimum_value_of_function_l702_70252

theorem minimum_value_of_function (x : ℝ) (h : x > 3) :
  (1 / (x - 3) + x) ≥ 5 ∧ ∃ y > 3, 1 / (y - 3) + y = 5 :=
sorry

end minimum_value_of_function_l702_70252


namespace binomial_cube_plus_one_l702_70258

theorem binomial_cube_plus_one : 7^3 + 3*(7^2) + 3*7 + 2 = 513 := by
  sorry

end binomial_cube_plus_one_l702_70258


namespace no_infinite_sequence_exists_l702_70215

theorem no_infinite_sequence_exists : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n, k n ≠ 0) ∧ 
  (∀ n, k (n + 1) = k n - 1 / k n) ∧ 
  (∀ n, k n * k (n + 1) ≥ 0) := by
sorry

end no_infinite_sequence_exists_l702_70215


namespace parallelogram_bisecting_line_l702_70212

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

def parallelogram : Parallelogram := {
  v1 := ⟨12, 50⟩,
  v2 := ⟨12, 120⟩,
  v3 := ⟨30, 160⟩,
  v4 := ⟨30, 90⟩
}

/-- Function to check if a line cuts a parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (p : Parallelogram) (l : Line) : Prop := sorry

/-- Function to express a real number as a fraction of relatively prime positive integers -/
def asRelativelyPrimeFraction (x : ℝ) : (ℕ × ℕ) := sorry

theorem parallelogram_bisecting_line :
  ∃ (l : Line),
    cutsIntoCongruentPolygons parallelogram l ∧
    l.slope = 5 ∧
    let (m, n) := asRelativelyPrimeFraction l.slope
    m + n = 6 := by sorry

end parallelogram_bisecting_line_l702_70212


namespace hiking_equipment_cost_l702_70220

/-- Calculates the total cost of hiking equipment for Celina --/
theorem hiking_equipment_cost :
  let hoodie_cost : ℚ := 80
  let flashlight_cost : ℚ := 0.2 * hoodie_cost
  let boots_original : ℚ := 110
  let boots_discount : ℚ := 0.1
  let water_filter_original : ℚ := 65
  let water_filter_discount : ℚ := 0.25
  let camping_mat_original : ℚ := 45
  let camping_mat_discount : ℚ := 0.15
  let total_cost : ℚ := 
    hoodie_cost + 
    flashlight_cost + 
    (boots_original * (1 - boots_discount)) + 
    (water_filter_original * (1 - water_filter_discount)) + 
    (camping_mat_original * (1 - camping_mat_discount))
  total_cost = 282 := by
  sorry

end hiking_equipment_cost_l702_70220


namespace quadratic_polynomial_satisfies_conditions_l702_70203

-- Define the quadratic polynomial
def q (x : ℝ) : ℝ := 2.1 * x^2 - 3.1 * x - 1.2

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  q (-1) = 4 ∧ q 2 = 1 ∧ q 4 = 20 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l702_70203


namespace correct_substitution_l702_70214

theorem correct_substitution (x y : ℝ) : 
  (x = 3*y - 1 ∧ x - 2*y = 4) → (3*y - 1 - 2*y = 4) := by
  sorry

end correct_substitution_l702_70214


namespace brothers_ages_l702_70210

theorem brothers_ages (a b : ℕ) (h1 : a > b) (h2 : a / b = 3 / 2) (h3 : a - b = 24) :
  a + b = 120 := by
  sorry

end brothers_ages_l702_70210


namespace trigonometric_expression_equality_l702_70243

theorem trigonometric_expression_equality : 
  (Real.sin (10 * π / 180) * Real.sin (80 * π / 180)) / 
  (Real.cos (35 * π / 180)^2 - Real.sin (35 * π / 180)^2) = 1/2 := by
  sorry

end trigonometric_expression_equality_l702_70243


namespace product_of_powers_l702_70273

theorem product_of_powers (m : ℝ) : 2 * m^3 * (3 * m^4) = 6 * m^7 := by
  sorry

end product_of_powers_l702_70273


namespace cubic_three_distinct_roots_l702_70275

/-- The cubic equation x^3 - 3x^2 - a = 0 has three distinct real roots if and only if a is in the open interval (-4, 0) -/
theorem cubic_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x^2 - a = 0 ∧
    y^3 - 3*y^2 - a = 0 ∧
    z^3 - 3*z^2 - a = 0) ↔
  -4 < a ∧ a < 0 :=
sorry

end cubic_three_distinct_roots_l702_70275


namespace gcd_properties_l702_70282

theorem gcd_properties (a b n : ℕ) (c : ℤ) (h1 : a ≠ 0) (h2 : c > 0) :
  let d := Nat.gcd a b
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧
  (Nat.gcd (a * c.natAbs) (b * c.natAbs) = c.natAbs * Nat.gcd a b) :=
by sorry

end gcd_properties_l702_70282


namespace complex_multiplication_example_l702_70213

theorem complex_multiplication_example :
  let z₁ : ℂ := 2 + 2*I
  let z₂ : ℂ := 1 - 2*I
  z₁ * z₂ = 6 - 2*I :=
by sorry

end complex_multiplication_example_l702_70213


namespace intersection_theorem_l702_70297

/-- The line equation y = 2√2(x-1) -/
def line (x y : ℝ) : Prop := y = 2 * Real.sqrt 2 * (x - 1)

/-- The parabola equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point M with coordinates (-1, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (-1, m)

/-- Function to calculate the dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

/-- Theorem stating that m = √2/2 given the conditions -/
theorem intersection_theorem (A B : ℝ × ℝ) (m : ℝ) :
  line A.1 A.2 →
  line B.1 B.2 →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  let M := point_M m
  let MA := (A.1 - M.1, A.2 - M.2)
  let MB := (B.1 - M.1, B.2 - M.2)
  dot_product MA MB = 0 →
  m = Real.sqrt 2 / 2 := by
  sorry

end intersection_theorem_l702_70297


namespace curve_is_line_l702_70202

/-- The curve represented by the equation (x+2y-1)√(x²+y²-2x+2)=0 is a line. -/
theorem curve_is_line : 
  ∀ (x y : ℝ), (x + 2*y - 1) * Real.sqrt (x^2 + y^2 - 2*x + 2) = 0 ↔ x + 2*y - 1 = 0 :=
by sorry

end curve_is_line_l702_70202


namespace nested_radical_solution_l702_70268

theorem nested_radical_solution :
  ∃ x : ℝ, x > 0 ∧ x^2 = 6 + x ∧ x = 3 := by sorry

end nested_radical_solution_l702_70268


namespace steven_pears_count_l702_70237

/-- The number of seeds Steven needs to collect -/
def total_seeds : ℕ := 60

/-- The average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- The average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- The average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- The number of apples Steven has set aside -/
def apples_set_aside : ℕ := 4

/-- The number of grapes Steven has set aside -/
def grapes_set_aside : ℕ := 9

/-- The number of additional seeds Steven needs -/
def additional_seeds_needed : ℕ := 3

/-- The number of pears Steven has set aside -/
def pears_set_aside : ℕ := 3

theorem steven_pears_count :
  pears_set_aside * pear_seeds + 
  apples_set_aside * apple_seeds + 
  grapes_set_aside * grape_seeds = 
  total_seeds - additional_seeds_needed :=
by sorry

end steven_pears_count_l702_70237


namespace two_out_of_three_correct_probability_l702_70270

def probability_correct_forecast : ℝ := 0.8

def probability_two_out_of_three_correct : ℝ :=
  3 * probability_correct_forecast^2 * (1 - probability_correct_forecast)

theorem two_out_of_three_correct_probability :
  probability_two_out_of_three_correct = 0.384 := by
  sorry

end two_out_of_three_correct_probability_l702_70270


namespace age_sum_product_total_l702_70241

theorem age_sum_product_total (elvie_age arielle_age : ℕ) : 
  elvie_age = 10 → arielle_age = 11 → 
  (elvie_age + arielle_age) + (elvie_age * arielle_age) = 131 := by
  sorry

end age_sum_product_total_l702_70241


namespace salary_increase_proof_l702_70233

theorem salary_increase_proof (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) :
  num_employees = 20 ∧ initial_avg = 1500 ∧ manager_salary = 4650 →
  (((num_employees : ℚ) * initial_avg + manager_salary) / (num_employees + 1 : ℚ)) - initial_avg = 150 := by
  sorry

end salary_increase_proof_l702_70233


namespace smallest_linear_combination_l702_70231

theorem smallest_linear_combination (m n : ℤ) : ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 2017 * a + 48576 * b) ∧ 
  ∀ (l : ℕ), l > 0 → (∃ (c d : ℤ), l = 2017 * c + 48576 * d) → k ≤ l :=
by
  sorry

end smallest_linear_combination_l702_70231


namespace min_value_a_plus_2b_l702_70230

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b - a * b = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y - x * y = 0 → a + 2 * b ≤ x + 2 * y :=
by sorry

end min_value_a_plus_2b_l702_70230


namespace point_translation_proof_l702_70200

def translate_point (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

theorem point_translation_proof :
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := translate_point (translate_point P 0 (-3)) 2 0
  Q = (-1, 1) := by sorry

end point_translation_proof_l702_70200


namespace rate_per_axle_above_two_l702_70296

/-- The toll formula for a truck crossing a bridge -/
def toll_formula (rate : ℝ) (x : ℕ) : ℝ :=
  3.50 + rate * (x - 2 : ℝ)

/-- The number of axles on the 18-wheel truck -/
def truck_axles : ℕ := 5

/-- The toll for the 18-wheel truck -/
def truck_toll : ℝ := 5

/-- The rate per axle above 2 axles is $0.50 -/
theorem rate_per_axle_above_two (rate : ℝ) :
  toll_formula rate truck_axles = truck_toll →
  rate = 0.50 := by
  sorry

end rate_per_axle_above_two_l702_70296


namespace expected_rolls_in_year_l702_70293

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | composite
  | prime
  | reroll

/-- Represents the rules for Bob's breakfast die -/
def breakfastDie : Fin 8 → DieOutcome
  | 1 => DieOutcome.reroll
  | 2 => DieOutcome.prime
  | 3 => DieOutcome.prime
  | 4 => DieOutcome.composite
  | 5 => DieOutcome.prime
  | 6 => DieOutcome.composite
  | 7 => DieOutcome.reroll
  | 8 => DieOutcome.reroll

/-- The probability of getting each outcome -/
def outcomeProb : DieOutcome → Rat
  | DieOutcome.composite => 2/8
  | DieOutcome.prime => 3/8
  | DieOutcome.reroll => 3/8

/-- The number of days in a non-leap year -/
def daysInYear : Nat := 365

/-- Theorem stating the expected number of rolls in a non-leap year -/
theorem expected_rolls_in_year :
  let expectedRollsPerDay := 8/5
  (expectedRollsPerDay * daysInYear : Rat) = 584 := by
  sorry

end expected_rolls_in_year_l702_70293


namespace smallest_sum_consecutive_integers_l702_70224

theorem smallest_sum_consecutive_integers (n : ℕ) : 
  (∀ k < n, k * (k + 1) ≤ 420) → n * (n + 1) > 420 → n + (n + 1) = 43 := by
  sorry

end smallest_sum_consecutive_integers_l702_70224


namespace outfit_count_l702_70289

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available. -/
def num_pants : ℕ := 5

/-- The number of ties available. -/
def num_ties : ℕ := 4

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- An outfit consists of one shirt, one pair of pants, and optionally one tie and/or one belt. -/
def outfit := ℕ × ℕ × Option ℕ × Option ℕ

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the total number of possible outfits is 600. -/
theorem outfit_count : total_outfits = 600 := by sorry

end outfit_count_l702_70289


namespace system_solution_l702_70284

theorem system_solution (a b c A B C x y z : ℝ) : 
  (x + a * y + a^2 * z = A) ∧ 
  (x + b * y + b^2 * z = B) ∧ 
  (x + c * y + c^2 * z = C) →
  ((A = b + c ∧ B = c + a ∧ C = a + b) → 
    (z = 0 ∧ y = -1 ∧ x = A + b)) ∧
  ((A = b * c ∧ B = c * a ∧ C = a * b) → 
    (z = 1 ∧ y = -(a + b + c) ∧ x = a * b * c)) := by
sorry

end system_solution_l702_70284


namespace steven_needs_three_more_seeds_l702_70239

/-- Represents the number of seeds Steven needs to collect for his assignment -/
def total_seeds_required : ℕ := 60

/-- Represents the average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- Represents the average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- Represents the average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- Represents the number of apples Steven has -/
def steven_apples : ℕ := 4

/-- Represents the number of pears Steven has -/
def steven_pears : ℕ := 3

/-- Represents the number of grapes Steven has -/
def steven_grapes : ℕ := 9

/-- Theorem stating that Steven needs 3 more seeds to fulfill his assignment -/
theorem steven_needs_three_more_seeds :
  total_seeds_required - (steven_apples * apple_seeds + steven_pears * pear_seeds + steven_grapes * grape_seeds) = 3 := by
  sorry

end steven_needs_three_more_seeds_l702_70239


namespace sibling_product_l702_70256

/-- Represents a family with a specific structure -/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def sibling_count (f : Family) : ℕ × ℕ :=
  (f.sisters - 1, f.brothers)

/-- The main theorem stating the product of sisters and brothers for a sibling -/
theorem sibling_product (f : Family) (h1 : f.sisters = 6) (h2 : f.brothers = 3) :
  let (s, b) := sibling_count f
  s * b = 15 := by
  sorry

#check sibling_product

end sibling_product_l702_70256


namespace trig_expression_value_l702_70222

theorem trig_expression_value : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by
  sorry

end trig_expression_value_l702_70222


namespace bmw_sales_l702_70219

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (nissan_percent : ℚ) (chevrolet_percent : ℚ)
  (h_total : total = 300)
  (h_ford : ford_percent = 20 / 100)
  (h_nissan : nissan_percent = 25 / 100)
  (h_chevrolet : chevrolet_percent = 10 / 100)
  (h_sum : ford_percent + nissan_percent + chevrolet_percent < 1) :
  ↑total * (1 - (ford_percent + nissan_percent + chevrolet_percent)) = 135 := by
  sorry

end bmw_sales_l702_70219


namespace sandys_scooter_gain_percent_l702_70263

/-- Calculates the gain percent for a transaction given purchase price, repair cost, and selling price -/
def gainPercent (purchasePrice repairCost sellingPrice : ℚ) : ℚ :=
  let totalCost := purchasePrice + repairCost
  let gain := sellingPrice - totalCost
  (gain / totalCost) * 100

/-- Theorem: The gain percent for Sandy's scooter transaction is 10% -/
theorem sandys_scooter_gain_percent :
  gainPercent 900 300 1320 = 10 := by
  sorry

end sandys_scooter_gain_percent_l702_70263


namespace expression_value_l702_70228

theorem expression_value : 2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 := by
  sorry

end expression_value_l702_70228


namespace modulus_of_complex_number_l702_70240

theorem modulus_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 - i^3) * (1 + 2*i)
  Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_of_complex_number_l702_70240


namespace fence_painting_l702_70266

theorem fence_painting (total_length : ℝ) (percentage_difference : ℝ) : 
  total_length = 792 → percentage_difference = 0.2 → 
  ∃ (x : ℝ), x + (1 + percentage_difference) * x = total_length ∧ 
  (1 + percentage_difference) * x = 432 :=
by
  sorry

end fence_painting_l702_70266


namespace gcd_56_63_l702_70209

theorem gcd_56_63 : Nat.gcd 56 63 = 7 := by
  sorry

end gcd_56_63_l702_70209


namespace bridge_length_l702_70264

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 300 →
  train_speed_kmh = 35 →
  time_to_pass = 42.68571428571429 →
  ∃ (bridge_length : ℝ),
    bridge_length = 115 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * time_to_pass :=
by sorry

end bridge_length_l702_70264


namespace x_three_times_y_l702_70247

theorem x_three_times_y (q : ℚ) (x y : ℚ) 
  (hx : x = 5 - q) 
  (hy : y = 3 * q - 1) : 
  q = 4/5 ↔ x = 3 * y := by
sorry

end x_three_times_y_l702_70247


namespace annie_gives_mary_25_crayons_l702_70272

/-- The number of crayons Annie gives to Mary -/
def crayons_given_to_mary (pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

/-- Theorem stating that Annie gives 25 crayons to Mary -/
theorem annie_gives_mary_25_crayons :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

#eval crayons_given_to_mary 21 36

end annie_gives_mary_25_crayons_l702_70272


namespace gabriel_diabetes_capsules_l702_70225

theorem gabriel_diabetes_capsules (forgot_days took_days : ℕ) 
  (h1 : forgot_days = 3) 
  (h2 : took_days = 28) : 
  forgot_days + took_days = 31 := by
  sorry

end gabriel_diabetes_capsules_l702_70225


namespace cos_theta_equals_three_fifths_l702_70236

/-- Given that the terminal side of angle θ passes through the point (3, -4), prove that cos θ = 3/5 -/
theorem cos_theta_equals_three_fifths (θ : Real) (h : ∃ (r : Real), r > 0 ∧ r * Real.cos θ = 3 ∧ r * Real.sin θ = -4) : 
  Real.cos θ = 3/5 := by
sorry

end cos_theta_equals_three_fifths_l702_70236


namespace symmetry_implies_a_pow_b_eq_four_l702_70298

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_implies_a_pow_b_eq_four (a b : ℝ) :
  symmetric_x_axis (a + 2, -2) (4, b) → a^b = 4 := by
  sorry

#check symmetry_implies_a_pow_b_eq_four

end symmetry_implies_a_pow_b_eq_four_l702_70298


namespace problem_solution_l702_70251

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0) : 
  a = 1/5 := by sorry

end problem_solution_l702_70251


namespace doughnuts_given_away_l702_70250

/-- Represents the bakery's doughnut sales and production --/
structure BakeryData where
  total_doughnuts : ℕ
  small_box_capacity : ℕ
  large_box_capacity : ℕ
  small_box_price : ℚ
  large_box_price : ℚ
  discount_rate : ℚ
  small_boxes_sold : ℕ
  large_boxes_sold : ℕ
  large_boxes_discounted : ℕ

/-- Theorem stating the number of doughnuts given away --/
theorem doughnuts_given_away (data : BakeryData) : 
  data.total_doughnuts = 300 ∧
  data.small_box_capacity = 6 ∧
  data.large_box_capacity = 12 ∧
  data.small_box_price = 5 ∧
  data.large_box_price = 9 ∧
  data.discount_rate = 1/10 ∧
  data.small_boxes_sold = 20 ∧
  data.large_boxes_sold = 10 ∧
  data.large_boxes_discounted = 5 →
  data.total_doughnuts - 
    (data.small_boxes_sold * data.small_box_capacity + 
     data.large_boxes_sold * data.large_box_capacity) = 60 := by
  sorry

end doughnuts_given_away_l702_70250


namespace spadesuit_problem_l702_70232

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := by
  sorry

end spadesuit_problem_l702_70232


namespace product_of_cubes_equality_l702_70281

theorem product_of_cubes_equality : 
  (8 / 9 : ℚ)^3 * (-1 / 3 : ℚ)^3 * (3 / 4 : ℚ)^3 = -8 / 729 := by sorry

end product_of_cubes_equality_l702_70281


namespace tenth_term_of_arithmetic_sequence_l702_70278

/-- Given an arithmetic sequence of 20 terms with first term 7 and last term 67,
    the 10th term is equal to 673 / 19. -/
theorem tenth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℚ), 
    (∀ n : ℕ, n < 19 → a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 7 →                                         -- first term
    a 19 = 67 →                                       -- last term
    a 9 = 673 / 19 :=                                 -- 10th term (index 9)
by
  sorry

end tenth_term_of_arithmetic_sequence_l702_70278


namespace sin_2x_value_l702_70292

theorem sin_2x_value (x : ℝ) (h : Real.sin (π + x) + Real.cos (π + x) = 1/2) : 
  Real.sin (2 * x) = -3/4 := by
sorry

end sin_2x_value_l702_70292


namespace min_value_reciprocal_sum_l702_70299

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
sorry

end min_value_reciprocal_sum_l702_70299


namespace billionth_term_is_16_l702_70248

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 112002
  else
    let prev := sequence_term (n - 1)
    prev + 5 * (prev % 10) - 10 * ((prev % 10) / 10)

def is_cyclic (seq : ℕ → ℕ) (cycle_length : ℕ) : Prop :=
  ∀ n : ℕ, seq (n + cycle_length) = seq n

theorem billionth_term_is_16 :
  is_cyclic sequence_term 42 →
  sequence_term (10^9 % 42) = 16 →
  sequence_term (10^9) = 16 :=
by sorry

end billionth_term_is_16_l702_70248


namespace min_alpha_gamma_sum_l702_70257

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (α γ : ℂ) (z : ℂ) : ℂ := (5 + 2*i)*z^3 + (4 + i)*z^2 + α*z + γ

-- State the theorem
theorem min_alpha_gamma_sum (α γ : ℂ) : 
  (f α γ 1).im = 0 → (f α γ i).im = 0 → (f α γ (-1)).im = 0 → 
  Complex.abs α + Complex.abs γ ≥ 7 :=
sorry

end min_alpha_gamma_sum_l702_70257


namespace theta_values_l702_70242

theorem theta_values (a b : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.cos (x + 2 * θ) + b * x + 3
  (f 1 = 5 ∧ f (-1) = 1) → (θ = π / 4 ∨ θ = -π / 4) :=
by sorry

end theta_values_l702_70242


namespace arithmetic_sequence_special_case_l702_70204

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 2)^2 + 12 * (a 2) - 8 = 0 →
  (a 10)^2 + 12 * (a 10) - 8 = 0 →
  a 6 = -6 := by
sorry

end arithmetic_sequence_special_case_l702_70204


namespace sock_selection_l702_70260

theorem sock_selection (n k : ℕ) (h1 : n = 7) (h2 : k = 4) : 
  Nat.choose n k = 35 := by
sorry

end sock_selection_l702_70260


namespace line_through_intersection_with_equal_intercepts_l702_70244

/-- The equation of a line passing through the intersection of two given lines and having equal intercepts on the coordinate axes -/
theorem line_through_intersection_with_equal_intercepts :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x + 2*y - 6 = 0 ∧ x - 2*y + 2 = 0 → a*x + b*y + c = 0) ∧
    (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ a*x₁ + c = 0 ∧ b*x₂ + c = 0 ∧ x₁ = x₂) →
    (a = 1 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) := by
  sorry

end line_through_intersection_with_equal_intercepts_l702_70244


namespace quadratic_inequality_solution_set_l702_70261

theorem quadratic_inequality_solution_set (x : ℝ) :
  x^2 - 3*x + 2 > 0 ↔ x < 1 ∨ x > 2 := by
sorry

end quadratic_inequality_solution_set_l702_70261


namespace city_mpg_equals_highway_mpg_l702_70286

/-- The average miles per gallon (mpg) for an SUV on the highway -/
def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 21 gallons of gasoline -/
def max_distance : ℝ := 256.2

/-- The amount of gasoline in gallons used for the maximum distance -/
def gasoline_amount : ℝ := 21

/-- Theorem: The average mpg in the city is equal to the average mpg on the highway -/
theorem city_mpg_equals_highway_mpg :
  max_distance / gasoline_amount = highway_mpg := by
  sorry


end city_mpg_equals_highway_mpg_l702_70286


namespace max_storage_period_is_56_days_l702_70234

/-- Represents the financial parameters for a wholesale product --/
structure WholesaleProduct where
  wholesalePrice : ℝ
  grossProfitMargin : ℝ
  borrowedCapitalRatio : ℝ
  monthlyInterestRate : ℝ
  dailyStorageCost : ℝ

/-- Calculates the maximum storage period without incurring a loss --/
def maxStoragePeriod (p : WholesaleProduct) : ℕ :=
  sorry

/-- Theorem stating the maximum storage period for the given product --/
theorem max_storage_period_is_56_days (p : WholesaleProduct)
  (h1 : p.wholesalePrice = 500)
  (h2 : p.grossProfitMargin = 0.04)
  (h3 : p.borrowedCapitalRatio = 0.8)
  (h4 : p.monthlyInterestRate = 0.0042)
  (h5 : p.dailyStorageCost = 0.30) :
  maxStoragePeriod p = 56 :=
sorry

end max_storage_period_is_56_days_l702_70234


namespace binomial_1000_500_not_divisible_by_7_l702_70287

theorem binomial_1000_500_not_divisible_by_7 : ¬ (7 ∣ Nat.choose 1000 500) := by
  sorry

end binomial_1000_500_not_divisible_by_7_l702_70287


namespace power_product_result_l702_70262

theorem power_product_result : (-1.5) ^ 2021 * (2/3) ^ 2023 = -(4/9) := by sorry

end power_product_result_l702_70262


namespace bridge_length_l702_70223

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 140 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 235 := by
  sorry

end bridge_length_l702_70223


namespace angle_calculations_l702_70227

/-- Given points P and Q on the terminal sides of angles α and β, 
    prove the values of sin(α - β) and cos(α + β) -/
theorem angle_calculations (P Q : ℝ × ℝ) (α β : ℝ) : 
  P = (-3, 4) → Q = (-1, -2) → 
  (P.1 = (Real.cos α) * Real.sqrt (P.1^2 + P.2^2)) →
  (P.2 = (Real.sin α) * Real.sqrt (P.1^2 + P.2^2)) →
  (Q.1 = (Real.cos β) * Real.sqrt (Q.1^2 + Q.2^2)) →
  (Q.2 = (Real.sin β) * Real.sqrt (Q.1^2 + Q.2^2)) →
  Real.sin (α - β) = -2 * Real.sqrt 5 / 5 ∧ 
  Real.cos (α + β) = 11 * Real.sqrt 5 / 25 := by
  sorry


end angle_calculations_l702_70227


namespace tangent_sum_difference_l702_70253

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := by
  sorry

end tangent_sum_difference_l702_70253


namespace chip_notebook_usage_l702_70207

/-- Calculates the number of packs of notebook paper used by Chip over a given number of weeks. -/
def notebook_packs_used (pages_per_day_per_class : ℕ) (classes : ℕ) (days_per_week : ℕ) 
  (weeks : ℕ) (sheets_per_pack : ℕ) : ℕ :=
  let total_pages := pages_per_day_per_class * classes * days_per_week * weeks
  (total_pages + sheets_per_pack - 1) / sheets_per_pack

/-- Proves that Chip uses 3 packs of notebook paper after 6 weeks. -/
theorem chip_notebook_usage : 
  notebook_packs_used 2 5 5 6 100 = 3 := by
  sorry

end chip_notebook_usage_l702_70207


namespace line_through_points_l702_70226

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/2 : ℝ) • a + (1/2 : ℝ) • b = a + t • (b - a) :=
sorry

end line_through_points_l702_70226


namespace abs_eq_self_iff_nonneg_l702_70201

theorem abs_eq_self_iff_nonneg (x : ℝ) : |x| = x ↔ x ≥ 0 := by
  sorry

end abs_eq_self_iff_nonneg_l702_70201


namespace base_b_perfect_square_l702_70245

-- Define the representation of a number in base b
def base_representation (b : ℕ) : ℕ := b^2 + 4*b + 1

-- Theorem statement
theorem base_b_perfect_square (b : ℕ) (h : b > 4) :
  ∃ n : ℕ, base_representation b = n^2 :=
sorry

end base_b_perfect_square_l702_70245


namespace prime_cube_plus_one_l702_70271

theorem prime_cube_plus_one (p : ℕ) (hp : Prime p) :
  (∃ (x y : ℕ), p^x = y^3 + 1) ↔ p = 2 ∨ p = 3 := by
  sorry

end prime_cube_plus_one_l702_70271


namespace tony_squat_weight_l702_70218

-- Define Tony's lifting capabilities
def curl_weight : ℕ := 90
def military_press_weight : ℕ := 2 * curl_weight
def squat_weight : ℕ := 5 * military_press_weight

-- Theorem to prove
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end tony_squat_weight_l702_70218


namespace line_tangent_to_parabola_l702_70277

/-- A line is tangent to a parabola if and only if the value of k satisfies the tangency condition -/
theorem line_tangent_to_parabola (x y k : ℝ) : 
  (∃ (x₀ y₀ : ℝ), (4 * x₀ + 3 * y₀ + k = 0) ∧ (y₀^2 = 12 * x₀) ∧
    (∀ (x' y' : ℝ), (4 * x' + 3 * y' + k = 0) ∧ (y'^2 = 12 * x') → (x' = x₀ ∧ y' = y₀))) ↔
  (k = 27 / 4) := by
sorry

end line_tangent_to_parabola_l702_70277


namespace square_sum_not_equal_sum_squares_l702_70285

theorem square_sum_not_equal_sum_squares : ∃ (a b : ℝ), a^2 + b^2 ≠ (a + b)^2 := by
  sorry

end square_sum_not_equal_sum_squares_l702_70285


namespace eighth_term_is_15_l702_70269

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term function
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem eighth_term_is_15 : a 8 = 15 := by
  sorry

end eighth_term_is_15_l702_70269


namespace unique_solution_quadratic_l702_70217

theorem unique_solution_quadratic (p : ℝ) : 
  p ≠ 0 ∧ (∃! x, p * x^2 - 8 * x + 2 = 0) ↔ p = 8 := by sorry

end unique_solution_quadratic_l702_70217


namespace range_of_expressions_l702_70205

theorem range_of_expressions (a b : ℝ) 
  (ha : 1 < a ∧ a < 4) 
  (hb : 2 < b ∧ b < 8) : 
  (8 < 2*a + 3*b ∧ 2*a + 3*b < 32) ∧ 
  (-7 < a - b ∧ a - b < 2) := by
sorry

end range_of_expressions_l702_70205


namespace quadratic_equation_condition_l702_70216

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 3 * x - 1 = 0 → (m - 1) ≠ 0) → m ≠ 1 :=
by sorry

end quadratic_equation_condition_l702_70216


namespace marbles_given_l702_70211

theorem marbles_given (drew_initial : ℕ) (marcus_initial : ℕ) (marbles_given : ℕ) : 
  drew_initial = marcus_initial + 24 →
  drew_initial - marbles_given = 25 →
  marcus_initial + marbles_given = 25 →
  marbles_given = 12 := by
sorry

end marbles_given_l702_70211


namespace will_picked_up_38_sticks_l702_70279

/-- The number of sticks originally in the yard -/
def original_sticks : ℕ := 99

/-- The number of sticks left after Will picked some up -/
def remaining_sticks : ℕ := 61

/-- The number of sticks Will picked up -/
def picked_up_sticks : ℕ := original_sticks - remaining_sticks

theorem will_picked_up_38_sticks : picked_up_sticks = 38 := by
  sorry

end will_picked_up_38_sticks_l702_70279


namespace rays_dog_walks_66_blocks_per_day_l702_70283

/-- Represents the number of blocks in each segment of Ray's walk --/
structure WalkSegments where
  toPark : ℕ
  toHighSchool : ℕ
  toHome : ℕ

/-- Calculates the total blocks walked in one trip --/
def totalBlocksPerTrip (w : WalkSegments) : ℕ :=
  w.toPark + w.toHighSchool + w.toHome

/-- Represents Ray's daily dog walking routine --/
structure DailyWalk where
  segments : WalkSegments
  tripsPerDay : ℕ

/-- Calculates the total blocks walked per day --/
def totalBlocksPerDay (d : DailyWalk) : ℕ :=
  (totalBlocksPerTrip d.segments) * d.tripsPerDay

/-- Theorem stating that Ray's dog walks 66 blocks each day --/
theorem rays_dog_walks_66_blocks_per_day :
  ∀ (d : DailyWalk),
    d.segments.toPark = 4 →
    d.segments.toHighSchool = 7 →
    d.segments.toHome = 11 →
    d.tripsPerDay = 3 →
    totalBlocksPerDay d = 66 :=
by
  sorry


end rays_dog_walks_66_blocks_per_day_l702_70283


namespace normal_distribution_symmetry_regression_coefficient_quadratic_inequality_condition_l702_70254

-- Normal distribution properties
def normal_distribution (μ σ : ℝ) (σ_pos : σ > 0) : ℝ → ℝ := sorry

-- Probability measure
def probability (P : Set ℝ → ℝ) : Set ℝ → ℝ := sorry

-- Statement 1
theorem normal_distribution_symmetry 
  (σ : ℝ) (σ_pos : σ > 0) (P : Set ℝ → ℝ) :
  probability P {x | 0 < x ∧ x < 1} = 0.35 →
  probability P {x | 0 < x ∧ x < 2} = 0.7 := sorry

-- Statement 2
theorem regression_coefficient 
  (c k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = c * Real.exp (k * x)) →
  (∀ x, Real.log (y x) = 0.3 * x + 4) →
  c = Real.exp 4 := sorry

-- Statement 3
theorem quadratic_inequality_condition
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (∀ x > 1, a * x^2 - (a + b - 1) * x + b > 0) ↔ a ≥ b - 1 := sorry

end normal_distribution_symmetry_regression_coefficient_quadratic_inequality_condition_l702_70254
