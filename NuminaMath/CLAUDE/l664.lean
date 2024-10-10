import Mathlib

namespace first_fun_friday_march31_l664_66499

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Function to determine if a given date is a Friday -/
def isFriday (d : Date) (startDay : DayOfWeek) : Bool := sorry

/-- Function to count the number of Fridays in a month -/
def countFridays (month : Nat) (startDay : DayOfWeek) : Nat := sorry

/-- Function to determine if a date is a Fun Friday -/
def isFunFriday (d : Date) (startDay : DayOfWeek) : Bool := sorry

/-- Theorem stating that the first Fun Friday of the year is on March 31 -/
theorem first_fun_friday_march31 (startDay : DayOfWeek) :
  startDay = DayOfWeek.Wednesday →
  (∀ d : Date, d.month < 3 → ¬isFunFriday d startDay) →
  isFunFriday { month := 3, day := 31 } startDay :=
sorry

end first_fun_friday_march31_l664_66499


namespace insect_count_l664_66466

theorem insect_count (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : legs_per_insect = 6) :
  total_legs / legs_per_insect = 6 := by
  sorry

end insect_count_l664_66466


namespace playground_paint_ratio_l664_66405

/-- Given a square playground with side length s and diagonal paint lines of width w,
    if one-third of the playground's area is covered in paint,
    then the ratio of s to w is 3/2. -/
theorem playground_paint_ratio (s w : ℝ) (h_positive : s > 0 ∧ w > 0) 
    (h_paint_area : w^2 + (s - w)^2 / 2 = s^2 / 3) : s / w = 3 / 2 := by
  sorry

end playground_paint_ratio_l664_66405


namespace triangle_segment_length_l664_66481

/-- Given a triangle ADE with points B, C, F, and G on its sides, prove that FC = 10.25 -/
theorem triangle_segment_length (DC CB : ℝ) (h1 : DC = 9) (h2 : CB = 8)
  (AD AB ED : ℝ) (h3 : AB = (1/4) * AD) (h4 : ED = (3/4) * AD) : 
  ∃ (FC : ℝ), FC = 10.25 := by
  sorry

end triangle_segment_length_l664_66481


namespace rival_awards_count_l664_66434

/-- The number of awards won by Scott -/
def scott_awards : ℕ := 4

/-- The number of awards won by Jessie relative to Scott -/
def jessie_multiplier : ℕ := 3

/-- The number of awards won by the rival relative to Jessie -/
def rival_multiplier : ℕ := 2

/-- The number of awards won by the rival -/
def rival_awards : ℕ := rival_multiplier * (jessie_multiplier * scott_awards)

theorem rival_awards_count : rival_awards = 24 := by
  sorry

end rival_awards_count_l664_66434


namespace parrot_count_l664_66424

theorem parrot_count (total_birds : ℕ) (remaining_parrots : ℕ) (remaining_crow : ℕ) 
  (h1 : total_birds = 13)
  (h2 : remaining_parrots = 2)
  (h3 : remaining_crow = 1)
  (h4 : ∃ (x : ℕ), total_birds = remaining_parrots + remaining_crow + 2 * x) :
  ∃ (initial_parrots : ℕ), initial_parrots = 7 ∧ 
    ∃ (initial_crows : ℕ), initial_crows + initial_parrots = total_birds :=
by
  sorry

end parrot_count_l664_66424


namespace nearest_integer_to_three_plus_sqrt_five_sixth_power_l664_66441

theorem nearest_integer_to_three_plus_sqrt_five_sixth_power :
  ⌊(3 + Real.sqrt 5)^6 + 1/2⌋ = 20608 := by
  sorry

end nearest_integer_to_three_plus_sqrt_five_sixth_power_l664_66441


namespace simplify_and_evaluate_l664_66476

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = -2) :
  2 * (x - 2*y)^2 - (2*y + x) * (-2*y + x) = 33 := by
  sorry

end simplify_and_evaluate_l664_66476


namespace consecutive_pages_product_l664_66447

theorem consecutive_pages_product (n : ℕ) : 
  n > 0 → n + (n + 1) = 217 → n * (n + 1) = 11772 := by
  sorry

end consecutive_pages_product_l664_66447


namespace susan_spending_equals_2000_l664_66419

/-- The cost of a single pencil in cents -/
def pencil_cost : ℕ := 25

/-- The cost of a single pen in cents -/
def pen_cost : ℕ := 80

/-- The total number of items (pens and pencils) Susan bought -/
def total_items : ℕ := 36

/-- The number of pencils Susan bought -/
def pencils_bought : ℕ := 16

/-- Calculate Susan's total spending in cents -/
def susan_spending : ℕ := pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)

/-- Theorem: Susan's total spending equals $20.00 -/
theorem susan_spending_equals_2000 : susan_spending = 2000 := by sorry

end susan_spending_equals_2000_l664_66419


namespace intersection_A_B_l664_66435

-- Define set A
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end intersection_A_B_l664_66435


namespace square_key_presses_l664_66493

theorem square_key_presses (start : ℝ) (target : ℝ) : ∃ (n : ℕ), n = 4 ∧ start^(2^n) > target ∧ ∀ m < n, start^(2^m) ≤ target :=
by
  -- We define the starting number and the target
  let start := 1.5
  let target := 300
  -- The proof goes here
  sorry

#check square_key_presses

end square_key_presses_l664_66493


namespace water_bottles_divisible_by_kits_l664_66496

/-- Represents the number of emergency-preparedness kits Veronica can make. -/
def num_kits : ℕ := 4

/-- Represents the total number of cans of food Veronica has. -/
def total_cans : ℕ := 12

/-- Represents the number of bottles of water Veronica has. -/
def water_bottles : ℕ := sorry

/-- Theorem stating that the number of water bottles is divisible by the number of kits. -/
theorem water_bottles_divisible_by_kits : 
  water_bottles % num_kits = 0 ∧ 
  total_cans % num_kits = 0 :=
sorry

end water_bottles_divisible_by_kits_l664_66496


namespace f_decreasing_on_interval_l664_66401

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 11, f_deriv x < 0 :=
sorry

end f_decreasing_on_interval_l664_66401


namespace f_minimum_and_equal_values_l664_66428

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

noncomputable def h (t : ℝ) : ℝ :=
  if t ≤ -1 then t * Real.exp (t + 2)
  else if t ≤ 1 then -Real.exp 1
  else (t - 2) * Real.exp t

theorem f_minimum_and_equal_values :
  (∀ t : ℝ, ∀ x ∈ Set.Icc t (t + 2), f x ≥ h t) ∧
  (∀ α β : ℝ, α ≠ β → f α = f β → α + β < 2) := by sorry

end f_minimum_and_equal_values_l664_66428


namespace max_value_of_function_l664_66449

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 5/4) :
  x * (5 - 4*x) ≤ 25/16 ∧ ∃ x₀, x₀ * (5 - 4*x₀) = 25/16 := by
  sorry

end max_value_of_function_l664_66449


namespace area_between_derivative_and_x_axis_l664_66459

open Real

noncomputable def f (x : ℝ) : ℝ := log x + x^2 - 3*x

theorem area_between_derivative_and_x_axis : 
  ∫ (x : ℝ) in (1/2)..1, (1/x + 2*x - 3) = -(3/4 - log 2) :=
sorry

end area_between_derivative_and_x_axis_l664_66459


namespace pyramid_base_edge_length_l664_66452

/-- A square pyramid with a hemisphere resting on top -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Radius of the hemisphere -/
  hemisphereRadius : ℝ
  /-- The hemisphere is tangent to each of the four lateral faces of the pyramid -/
  isTangent : Bool

/-- Calculates the edge-length of the square base of the pyramid -/
def baseEdgeLength (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating that for a pyramid of height 9 cm and a hemisphere of radius 3 cm,
    the edge-length of the square base is 4.5 cm -/
theorem pyramid_base_edge_length :
  let p : PyramidWithHemisphere := {
    pyramidHeight := 9,
    hemisphereRadius := 3,
    isTangent := true
  }
  baseEdgeLength p = 4.5 := by
  sorry

end pyramid_base_edge_length_l664_66452


namespace tony_age_is_six_l664_66482

/-- Represents Tony's work and payment details -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  daysWorked : ℕ
  totalEarned : ℚ

/-- Calculates Tony's age at the beginning of the work period -/
def calculateAge (work : TonyWork) : ℕ :=
  sorry

/-- Theorem stating that Tony's calculated age is 6 -/
theorem tony_age_is_six (work : TonyWork) 
  (h1 : work.hoursPerDay = 3)
  (h2 : work.payPerHourPerYear = 3/4)
  (h3 : work.daysWorked = 60)
  (h4 : work.totalEarned = 945) : 
  calculateAge work = 6 :=
sorry

end tony_age_is_six_l664_66482


namespace triangle_count_relation_l664_66437

/-- The number of non-overlapping triangles formed from 6 points when no three points are collinear -/
def n₀ : ℕ := 20

/-- The number of non-overlapping triangles formed from 6 points when exactly three points are collinear -/
def n₁ : ℕ := 19

/-- The number of non-overlapping triangles formed from 6 points when exactly four points are collinear -/
def n₂ : ℕ := 18

/-- Theorem stating the relationship between n₀, n₁, and n₂ -/
theorem triangle_count_relation : n₀ > n₁ ∧ n₁ > n₂ :=
by sorry

end triangle_count_relation_l664_66437


namespace polynomial_no_x_term_l664_66486

theorem polynomial_no_x_term (n : ℚ) : 
  (∀ x, (x + n) * (3 * x - 1) = 3 * x^2 - n) → n = 1/3 := by
  sorry

end polynomial_no_x_term_l664_66486


namespace partial_fraction_decomposition_l664_66497

theorem partial_fraction_decomposition :
  ∀ (x : ℝ), x ≠ 4 → x ≠ 5 →
  (7 * x - 4) / (x^2 - 9*x + 20) = (-20) / (x - 4) + 31 / (x - 5) :=
by sorry

end partial_fraction_decomposition_l664_66497


namespace sum_interior_angles_pentagon_l664_66440

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540 degrees -/
theorem sum_interior_angles_pentagon : 
  sum_interior_angles pentagon_sides = 540 := by sorry

end sum_interior_angles_pentagon_l664_66440


namespace cupcake_packages_l664_66433

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : 
  initial_cupcakes = 50 → eaten_cupcakes = 5 → cupcakes_per_package = 5 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 9 :=
by
  sorry

end cupcake_packages_l664_66433


namespace distance_between_points_l664_66443

theorem distance_between_points : 
  let x₁ : ℝ := 6
  let y₁ : ℝ := -18
  let x₂ : ℝ := 3
  let y₂ : ℝ := 9
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 738 := by
  sorry

end distance_between_points_l664_66443


namespace quadratic_root_difference_ratio_l664_66463

/-- Given quadratic functions and the differences of their roots, prove the ratio of differences of squared root differences -/
theorem quadratic_root_difference_ratio (a b : ℝ) :
  let f₁ : ℝ → ℝ := λ x ↦ x^2 + 2*x + a
  let f₂ : ℝ → ℝ := λ x ↦ x^2 + b*x - 1
  let f₃ : ℝ → ℝ := λ x ↦ 2*x^2 + (6-b)*x + 3*a + 1
  let f₄ : ℝ → ℝ := λ x ↦ 2*x^2 + (3*b-2)*x - a - 3
  let A := (Real.sqrt (4 - 4*a))
  let B := (Real.sqrt (b^2 + 4))
  let C := (1/2 * Real.sqrt (b^2 - 12*b - 24*a + 28))
  let D := (1/2 * Real.sqrt (9*b^2 - 12*b + 8*a + 28))
  A ≠ B →
  (C^2 - D^2) / (A^2 - B^2) = 2 := by
  sorry

end quadratic_root_difference_ratio_l664_66463


namespace solve_christmas_decorations_problem_l664_66400

def christmas_decorations_problem (decorations_per_box : ℕ) (decorations_used : ℕ) (decorations_given_away : ℕ) : Prop :=
  decorations_per_box = 15 ∧ 
  decorations_used = 35 ∧ 
  decorations_given_away = 25 →
  (decorations_used + decorations_given_away) / decorations_per_box = 4

theorem solve_christmas_decorations_problem :
  ∀ (decorations_per_box : ℕ) (decorations_used : ℕ) (decorations_given_away : ℕ),
  christmas_decorations_problem decorations_per_box decorations_used decorations_given_away :=
by
  sorry

end solve_christmas_decorations_problem_l664_66400


namespace solve_jim_ring_problem_l664_66479

def jim_ring_problem (first_ring_cost : ℝ) : Prop :=
  let second_ring_cost : ℝ := 2 * first_ring_cost
  let sale_price : ℝ := first_ring_cost / 2
  let out_of_pocket : ℝ := first_ring_cost + second_ring_cost - sale_price
  (first_ring_cost = 10000) → (out_of_pocket = 25000)

theorem solve_jim_ring_problem :
  jim_ring_problem 10000 := by
  sorry

end solve_jim_ring_problem_l664_66479


namespace EQ_equals_15_l664_66427

/-- Represents a trapezoid EFGH with a circle tangent to its sides -/
structure TrapezoidWithTangentCircle where
  -- Length of side EF
  EF : ℝ
  -- Length of side FG
  FG : ℝ
  -- Length of side GH
  GH : ℝ
  -- Length of side HE
  HE : ℝ
  -- Assumption that EF is parallel to GH
  EF_parallel_GH : True
  -- Assumption that there exists a circle with center Q on EF tangent to FG and HE
  circle_tangent : True

/-- Theorem stating that EQ = 15 in the given trapezoid configuration -/
theorem EQ_equals_15 (t : TrapezoidWithTangentCircle)
  (h1 : t.EF = 137)
  (h2 : t.FG = 75)
  (h3 : t.GH = 28)
  (h4 : t.HE = 105) :
  ∃ Q : ℝ, Q = 15 ∧ Q ≤ t.EF := by
  sorry

end EQ_equals_15_l664_66427


namespace same_remainder_problem_l664_66494

theorem same_remainder_problem : ∃ (N : ℕ), N > 1 ∧
  N = 23 ∧
  ∀ (M : ℕ), M > N → ¬(1743 % M = 2019 % M ∧ 2019 % M = 3008 % M) :=
by sorry

end same_remainder_problem_l664_66494


namespace equal_focal_distances_l664_66470

/-- The first curve equation -/
def curve1 (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (16 - k) - p.2^2 / k = 1}

/-- The second curve equation -/
def curve2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 9 * p.1^2 + 25 * p.2^2 = 225}

/-- The focal distance of a curve -/
def focalDistance (curve : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating the necessary and sufficient condition for equal focal distances -/
theorem equal_focal_distances :
  ∀ k : ℝ, (focalDistance (curve1 k) = focalDistance curve2) ↔ (0 < k ∧ k < 16) :=
sorry

end equal_focal_distances_l664_66470


namespace always_two_real_roots_specific_case_l664_66450

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := x^2 + (2-m)*x + (1-m)

-- Theorem stating that the equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0 :=
sorry

-- Theorem for the specific case when m < 0 and the difference between roots is 4
theorem specific_case (m : ℝ) (h₁ : m < 0) :
  (∃ (x₁ x₂ : ℝ), x₁ - x₂ = 4 ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0) →
  m = -4 :=
sorry

end always_two_real_roots_specific_case_l664_66450


namespace opposite_of_negative_one_half_l664_66429

theorem opposite_of_negative_one_half : 
  (-(-(1/2 : ℚ))) = (1/2 : ℚ) := by sorry

end opposite_of_negative_one_half_l664_66429


namespace ten_factorial_mod_thirteen_l664_66454

-- Define factorial for natural numbers
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem ten_factorial_mod_thirteen : factorial 10 % 13 = 6 := by
  sorry

end ten_factorial_mod_thirteen_l664_66454


namespace sqrt_sin_cos_identity_l664_66413

theorem sqrt_sin_cos_identity (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end sqrt_sin_cos_identity_l664_66413


namespace binomial_plus_three_l664_66490

theorem binomial_plus_three : Nat.choose 6 2 + 3 = 18 := by
  sorry

end binomial_plus_three_l664_66490


namespace gcd_6a_8b_lower_bound_l664_66467

theorem gcd_6a_8b_lower_bound (a b : ℕ+) (h : Nat.gcd a b = 10) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 10 ∧ Nat.gcd (6 * a') (8 * b') = 20) ∧
  (Nat.gcd (6 * a) (8 * b) ≥ 20) :=
sorry

end gcd_6a_8b_lower_bound_l664_66467


namespace matrix_determinant_l664_66495

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![1, -3, 3; 0, 5, -1; 4, -2, 2]
  Matrix.det A = -40 := by
sorry

end matrix_determinant_l664_66495


namespace hundredth_odd_integer_l664_66417

theorem hundredth_odd_integer : ∀ n : ℕ, n > 0 → (2 * n - 1) = 199 ↔ n = 100 := by sorry

end hundredth_odd_integer_l664_66417


namespace negative_three_x_squared_times_negative_three_x_l664_66468

theorem negative_three_x_squared_times_negative_three_x (x : ℝ) :
  (-3 * x) * (-3 * x)^2 = -27 * x^3 := by
  sorry

end negative_three_x_squared_times_negative_three_x_l664_66468


namespace problem_cube_surface_area_l664_66473

/-- Represents a cube structure -/
structure CubeStructure where
  size : ℕ
  smallCubeSize : ℕ
  removedCubes : ℕ

/-- Calculate the surface area of the cube structure -/
def surfaceArea (c : CubeStructure) : ℕ :=
  sorry

/-- The specific cube structure in the problem -/
def problemCube : CubeStructure :=
  { size := 8
  , smallCubeSize := 2
  , removedCubes := 4 }

/-- Theorem stating that the surface area of the problem cube is 1632 -/
theorem problem_cube_surface_area :
  surfaceArea problemCube = 1632 :=
sorry

end problem_cube_surface_area_l664_66473


namespace candy_given_to_haley_l664_66426

theorem candy_given_to_haley (initial_candy : ℕ) (remaining_candy : ℕ) (candy_given : ℕ) :
  initial_candy = 15 →
  remaining_candy = 9 →
  candy_given = initial_candy - remaining_candy →
  candy_given = 6 := by
sorry

end candy_given_to_haley_l664_66426


namespace condition_sufficient_not_necessary_l664_66458

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x > 0) ∧
  ¬(∀ x : ℝ, x > 0 → 1 < x ∧ x < 2) := by
  sorry

end condition_sufficient_not_necessary_l664_66458


namespace garden_view_theorem_l664_66411

/-- Represents a circular garden with trees -/
structure Garden where
  radius : ℝ
  treeGridSideLength : ℝ
  treeRadius : ℝ

/-- Checks if the view from the gazebo is obstructed -/
def isViewObstructed (g : Garden) : Prop :=
  ∀ θ : ℝ, ∃ (x y : ℤ), (x : ℝ)^2 + (y : ℝ)^2 ≤ g.radius^2 ∧
    ((x : ℝ) - g.treeRadius * Real.cos θ)^2 + ((y : ℝ) - g.treeRadius * Real.sin θ)^2 ≤ g.treeGridSideLength^2

theorem garden_view_theorem (g : Garden) (h1 : g.radius = 50) (h2 : g.treeGridSideLength = 1) :
  (g.treeRadius < 1 / Real.sqrt 2501 → ¬ isViewObstructed g) ∧
  (g.treeRadius = 1 / 50 → isViewObstructed g) := by
  sorry

#check garden_view_theorem

end garden_view_theorem_l664_66411


namespace brady_work_hours_september_l664_66412

/-- Proves that Brady worked 8 hours every day in September given the conditions --/
theorem brady_work_hours_september :
  let hours_per_day_april : ℕ := 6
  let hours_per_day_june : ℕ := 5
  let days_per_month : ℕ := 30
  let average_hours_per_month : ℕ := 190
  let total_months : ℕ := 3
  ∃ (hours_per_day_september : ℕ),
    hours_per_day_september * days_per_month =
      total_months * average_hours_per_month -
      (hours_per_day_april * days_per_month + hours_per_day_june * days_per_month) ∧
    hours_per_day_september = 8 :=
by
  sorry

end brady_work_hours_september_l664_66412


namespace probability_one_letter_each_name_l664_66472

theorem probability_one_letter_each_name :
  let total_cards : ℕ := 12
  let alex_cards : ℕ := 6
  let jamie_cards : ℕ := 6
  let prob_one_each : ℚ := (alex_cards * jamie_cards) / (total_cards * (total_cards - 1))
  prob_one_each = 6 / 11 := by
sorry

end probability_one_letter_each_name_l664_66472


namespace complex_equation_solution_l664_66410

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2*i) * i = b - i) : 
  a + b*i = -1 + 2*i := by
sorry

end complex_equation_solution_l664_66410


namespace cos_power_negative_set_l664_66465

open Set Real

theorem cos_power_negative_set (M : Set ℝ) : 
  M = {x : ℝ | ∀ n : ℕ, cos (2^n * x) < 0} ↔ 
  M = {x : ℝ | ∃ k : ℤ, x = 2*k*π + 2*π/3 ∨ x = 2*k*π - 2*π/3} :=
by sorry

end cos_power_negative_set_l664_66465


namespace negation_of_universal_proposition_l664_66421

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_of_universal_proposition_l664_66421


namespace range_of_a_l664_66488

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l664_66488


namespace problem_solution_l664_66444

theorem problem_solution :
  -- Part 1
  ∀ (a b : ℝ), 10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 ∧
  -- Part 2
  ∀ (x y : ℝ), x^2 - 2*y = -5 → 4*x^2 - 8*y + 24 = 4 ∧
  -- Part 3
  ∀ (a b c d : ℝ),
    a - 2*b = 1009 + 1/2 ∧
    2*b - c = -2024 - 2/3 ∧
    c - d = 1013 + 1/6 →
    (a - c) + (2*b - d) - (2*b - c) = -2 := by
  sorry

end problem_solution_l664_66444


namespace f_g_inequality_implies_a_range_l664_66408

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

def g (x : ℝ) : ℝ := Real.exp x

def H (a : ℝ) (x : ℝ) : ℝ := f a x / g x

theorem f_g_inequality_implies_a_range :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ x₁ > x₂ →
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔
  -1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2 :=
by sorry

end

end f_g_inequality_implies_a_range_l664_66408


namespace abcNegative_neither_sufficient_nor_necessary_l664_66439

-- Define a struct to represent the curve ax^2 + by^2 = c
structure QuadraticCurve where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a QuadraticCurve to be a hyperbola
def isHyperbola (curve : QuadraticCurve) : Prop :=
  sorry

-- Define the condition abc < 0
def abcNegative (curve : QuadraticCurve) : Prop :=
  curve.a * curve.b * curve.c < 0

-- Theorem stating that abcNegative is neither sufficient nor necessary for isHyperbola
theorem abcNegative_neither_sufficient_nor_necessary :
  (∃ curve : QuadraticCurve, abcNegative curve ∧ ¬isHyperbola curve) ∧
  (∃ curve : QuadraticCurve, isHyperbola curve ∧ ¬abcNegative curve) :=
sorry

end abcNegative_neither_sufficient_nor_necessary_l664_66439


namespace product_of_one_plus_roots_l664_66406

theorem product_of_one_plus_roots (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
  sorry

end product_of_one_plus_roots_l664_66406


namespace car_cost_l664_66457

/-- Calculates the cost of the car given the costs of other gifts and total worth --/
theorem car_cost (ring_cost bracelet_cost total_worth : ℕ) 
  (h1 : ring_cost = 4000)
  (h2 : bracelet_cost = 2 * ring_cost)
  (h3 : total_worth = 14000) :
  total_worth - (ring_cost + bracelet_cost) = 2000 := by
  sorry

end car_cost_l664_66457


namespace two_team_property_min_teams_three_team_property_min_teams_l664_66487

/-- A tournament is a relation between teams representing victories -/
def Tournament (α : Type*) := α → α → Prop

/-- In a tournament, team a has defeated team b -/
def Defeated {α : Type*} (t : Tournament α) (a b : α) : Prop := t a b

/-- A tournament satisfies the two-team property if for any two teams,
    there exists a third team that has defeated both -/
def TwoTeamProperty {α : Type*} (t : Tournament α) : Prop :=
  ∀ a b : α, ∃ c : α, Defeated t c a ∧ Defeated t c b

/-- A tournament satisfies the three-team property if for any three teams,
    there exists a fourth team that has defeated all three -/
def ThreeTeamProperty {α : Type*} (t : Tournament α) : Prop :=
  ∀ a b c : α, ∃ d : α, Defeated t d a ∧ Defeated t d b ∧ Defeated t d c

theorem two_team_property_min_teams
  {α : Type*} [Fintype α] (t : Tournament α) (h : TwoTeamProperty t) :
  Fintype.card α ≥ 7 := by
  sorry

theorem three_team_property_min_teams
  {α : Type*} [Fintype α] (t : Tournament α) (h : ThreeTeamProperty t) :
  Fintype.card α ≥ 15 := by
  sorry

end two_team_property_min_teams_three_team_property_min_teams_l664_66487


namespace odd_cube_plus_23_divisible_by_24_l664_66475

theorem odd_cube_plus_23_divisible_by_24 (n : ℤ) (h : Odd n) : 
  ∃ k : ℤ, n^3 + 23*n = 24*k := by
sorry

end odd_cube_plus_23_divisible_by_24_l664_66475


namespace similar_polygons_area_perimeter_ratio_l664_66442

theorem similar_polygons_area_perimeter_ratio :
  ∀ (A₁ A₂ P₁ P₂ : ℝ),
    A₁ > 0 → A₂ > 0 → P₁ > 0 → P₂ > 0 →
    (A₁ / A₂ = 9 / 64) →
    (P₁ / P₂)^2 = (A₁ / A₂) →
    P₁ / P₂ = 3 / 8 := by
  sorry

end similar_polygons_area_perimeter_ratio_l664_66442


namespace digit_150_of_17_70_l664_66445

/-- The decimal representation of 17/70 has a repeating sequence of digits. -/
def decimal_rep_17_70 : ℕ → ℕ
| 0 => 2
| 1 => 4
| n + 2 => match (n + 2) % 6 with
  | 0 => 4
  | 1 => 2
  | 2 => 8
  | 3 => 5
  | 4 => 7
  | 5 => 1
  | _ => 0  -- This case should never occur

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 4. -/
theorem digit_150_of_17_70 : decimal_rep_17_70 149 = 4 := by
  sorry

end digit_150_of_17_70_l664_66445


namespace cost_36_roses_l664_66422

-- Define the proportionality factor
def prop_factor : ℚ := 30 / 18

-- Define the discount rate
def discount_rate : ℚ := 1 / 10

-- Define the discount threshold
def discount_threshold : ℕ := 30

-- Function to calculate the cost of a bouquet before discount
def cost_before_discount (roses : ℕ) : ℚ := prop_factor * roses

-- Function to apply discount if applicable
def apply_discount (cost : ℚ) (roses : ℕ) : ℚ :=
  if roses > discount_threshold then
    cost * (1 - discount_rate)
  else
    cost

-- Theorem stating the cost of 36 roses after discount
theorem cost_36_roses : apply_discount (cost_before_discount 36) 36 = 54 := by
  sorry

end cost_36_roses_l664_66422


namespace divisible_by_35_60_72_between_1000_and_3500_l664_66432

theorem divisible_by_35_60_72_between_1000_and_3500 : 
  ∃! n : ℕ, 1000 < n ∧ n < 3500 ∧ 35 ∣ n ∧ 60 ∣ n ∧ 72 ∣ n :=
by sorry

end divisible_by_35_60_72_between_1000_and_3500_l664_66432


namespace unique_solution_range_l664_66478

theorem unique_solution_range (a : ℝ) : 
  (∃! x : ℕ, x^2 - (a+2)*x + 2 - a < 0) → 
  (1/2 < a ∧ a ≤ 2/3) :=
by sorry

end unique_solution_range_l664_66478


namespace shortest_paths_count_l664_66446

/-- The number of shortest paths on an m × n grid -/
def numShortestPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

/-- Theorem: The number of shortest paths on an m × n grid
    from point A to point B is equal to (m+n choose n) -/
theorem shortest_paths_count (m n : ℕ) :
  numShortestPaths m n = Nat.choose (m + n) n := by
  sorry

end shortest_paths_count_l664_66446


namespace village_survival_time_l664_66420

/-- The number of people a vampire drains per week -/
def vampire_drain_rate : ℕ := 3

/-- The number of people a werewolf eats per week -/
def werewolf_eat_rate : ℕ := 5

/-- The total number of people in the village -/
def village_population : ℕ := 72

/-- The number of weeks the village will last -/
def village_survival_weeks : ℕ := 9

/-- Theorem stating how long the village will last -/
theorem village_survival_time :
  village_population / (vampire_drain_rate + werewolf_eat_rate) = village_survival_weeks :=
by sorry

end village_survival_time_l664_66420


namespace p_amount_l664_66418

theorem p_amount (p q r : ℚ) 
  (h1 : p = (1/6 * p + 1/6 * p) + 32) : p = 48 := by
  sorry

end p_amount_l664_66418


namespace number_of_subsets_of_three_element_set_l664_66477

theorem number_of_subsets_of_three_element_set :
  Finset.card (Finset.powerset {1, 2, 3}) = 8 := by
  sorry

end number_of_subsets_of_three_element_set_l664_66477


namespace no_solution_iff_k_eq_seven_l664_66448

theorem no_solution_iff_k_eq_seven (k : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end no_solution_iff_k_eq_seven_l664_66448


namespace opposite_of_negative_one_third_l664_66451

theorem opposite_of_negative_one_third :
  -((-1 : ℚ) / 3) = 1 / 3 := by sorry

end opposite_of_negative_one_third_l664_66451


namespace polynomial_division_remainder_l664_66485

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X : Polynomial ℚ)^4 + 3 * X^2 - 4 = (X^2 - 3) * q + 14 := by
  sorry

end polynomial_division_remainder_l664_66485


namespace sqrt_13_squared_l664_66431

theorem sqrt_13_squared : (Real.sqrt 13) ^ 2 = 13 := by
  sorry

end sqrt_13_squared_l664_66431


namespace subset_implies_a_leq_two_l664_66423

theorem subset_implies_a_leq_two (a : ℝ) : 
  let A : Set ℝ := {x | x ≥ a}
  let B : Set ℝ := {x | |x - 3| < 1}
  B ⊆ A → a ≤ 2 := by
  sorry

end subset_implies_a_leq_two_l664_66423


namespace square_root_and_arithmetic_square_root_l664_66453

variable (m : ℝ)

theorem square_root_and_arithmetic_square_root :
  (∀ x : ℝ, x^2 = (5 + m)^2 → x = (5 + m) ∨ x = -(5 + m)) ∧
  (Real.sqrt ((5 + m)^2) = |5 + m|) := by
  sorry

end square_root_and_arithmetic_square_root_l664_66453


namespace oranges_per_pack_l664_66471

/-- Proves the number of oranges in each pack given Tammy's orange selling scenario -/
theorem oranges_per_pack (trees : ℕ) (oranges_per_tree_per_day : ℕ) (price_per_pack : ℕ) 
  (total_earnings : ℕ) (days : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  price_per_pack = 2 →
  total_earnings = 840 →
  days = 21 →
  (trees * oranges_per_tree_per_day * days) / (total_earnings / price_per_pack) = 6 := by
  sorry

#check oranges_per_pack

end oranges_per_pack_l664_66471


namespace matrix_product_equality_l664_66438

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 4]

theorem matrix_product_equality :
  A * B = !![23, -5; 24, -20] := by sorry

end matrix_product_equality_l664_66438


namespace power_function_sum_range_l664_66489

def f (x : ℝ) : ℝ := x^2

theorem power_function_sum_range 
  (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ x₂) 
  (h₂ : x₂ ≥ x₃) 
  (h₃ : x₁ + x₂ + x₃ = 1) 
  (h₄ : f x₁ + f x₂ + f x₃ = 1) :
  2/3 ≤ x₁ + x₂ ∧ x₁ + x₂ ≤ 4/3 :=
by sorry

end power_function_sum_range_l664_66489


namespace sqrt_difference_approximation_l664_66430

theorem sqrt_difference_approximation : 
  ∃ ε > 0, |Real.sqrt (49 + 16) - Real.sqrt (36 - 9) - 2.8661| < ε :=
by sorry

end sqrt_difference_approximation_l664_66430


namespace turnip_bag_weights_l664_66498

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (· ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  ∀ t, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 := by sorry

end turnip_bag_weights_l664_66498


namespace pyramid_height_proof_l664_66414

/-- The height of a square-based pyramid with base edge length 10 units,
    given that its volume is equal to the volume of a cube with edge length 5 units. -/
def pyramid_height : ℝ := 3.75

theorem pyramid_height_proof :
  let cube_edge := 5
  let cube_volume := cube_edge ^ 3
  let pyramid_base_edge := 10
  let pyramid_base_area := pyramid_base_edge ^ 2
  pyramid_height = (3 * cube_volume) / pyramid_base_area :=
by sorry

end pyramid_height_proof_l664_66414


namespace polynomial_equation_solution_l664_66456

noncomputable def P (α β : ℝ) (x : ℝ) : ℝ := α * x^4 + α * x^3 + α * x^2 + α * x + β

noncomputable def Q (α : ℝ) (x : ℝ) : ℝ := α * x^3 + α * x

theorem polynomial_equation_solution (α β : ℝ) (hα : α ≠ 0) :
  (∀ x : ℝ, P α β (x^2) + Q α x = P α β x + x^5 * Q α x) ∧
  (∀ P' Q' : ℝ → ℝ, (∀ x : ℝ, P' (x^2) + Q' x = P' x + x^5 * Q' x) →
    (∃ c : ℝ, P' = P (c * α) (c * β) ∧ Q' = Q (c * α))) :=
by sorry

end polynomial_equation_solution_l664_66456


namespace sin_45_degrees_l664_66403

theorem sin_45_degrees :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_45_degrees_l664_66403


namespace product_of_roots_l664_66480

theorem product_of_roots (x : ℝ) : 
  (∃ a b c : ℝ, (x + 3) * (x - 4) = 2 * (x + 1) ∧ 
   a * x^2 + b * x + c = 0 ∧ 
   (∀ r s : ℝ, (a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) → r * s = c / a)) →
  (∃ r s : ℝ, (r + 3) * (r - 4) = 2 * (r + 1) ∧ 
              (s + 3) * (s - 4) = 2 * (s + 1) ∧ 
              r * s = -14) :=
by sorry

end product_of_roots_l664_66480


namespace average_age_of_women_l664_66460

theorem average_age_of_women (A : ℝ) (n : ℕ) : 
  n = 9 → 
  n * (A + 4) - (n * A - (36 + 32) + 104) = 0 → 
  104 / 2 = 52 :=
by sorry

end average_age_of_women_l664_66460


namespace space_station_cost_sharing_l664_66464

/-- The cost of building a space station in trillions of dollars -/
def space_station_cost : ℝ := 5

/-- The number of people sharing the cost in millions -/
def number_of_people : ℝ := 500

/-- The share of each person in dollars -/
def person_share : ℝ := 10000

theorem space_station_cost_sharing :
  (space_station_cost * 1000000) / number_of_people = person_share := by
  sorry

end space_station_cost_sharing_l664_66464


namespace drive_problem_solution_correct_l664_66416

/-- Represents the problem of Sarah's drive to the conference center. -/
structure DriveProblem where
  initial_speed : ℝ  -- Speed in miles per hour
  initial_distance : ℝ  -- Distance covered in the first hour
  speed_increase : ℝ  -- Increase in speed for the rest of the journey
  late_time : ℝ  -- Time in hours Sarah would be late if continuing at initial speed
  early_time : ℝ  -- Time in hours Sarah arrives early with increased speed

/-- The solution to the drive problem. -/
def solve_drive_problem (p : DriveProblem) : ℝ :=
  p.initial_distance

/-- Theorem stating that the solution to the drive problem is correct. -/
theorem drive_problem_solution_correct (p : DriveProblem) 
  (h1 : p.initial_speed = 40)
  (h2 : p.initial_distance = 40)
  (h3 : p.speed_increase = 20)
  (h4 : p.late_time = 0.75)
  (h5 : p.early_time = 0.25) :
  solve_drive_problem p = 40 := by sorry

#check drive_problem_solution_correct

end drive_problem_solution_correct_l664_66416


namespace solve_equation_l664_66409

theorem solve_equation (x : ℝ) (number : ℝ) : 
  x = 32 → 
  35 - (23 - (15 - x)) = number * 2 / (1 / 2) → 
  number = -1.25 := by
sorry

end solve_equation_l664_66409


namespace max_ab_squared_l664_66484

theorem max_ab_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  a * b^2 ≤ 4 :=
sorry

end max_ab_squared_l664_66484


namespace parallelogram_diagonals_fixed_points_l664_66455

/-- A line in a 2D plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A parallelogram in a 2D plane -/
structure Parallelogram :=
  (a b c d : Point)

/-- The diagonal of a parallelogram -/
def diagonal (p : Parallelogram) : Line :=
  sorry

/-- Check if a point lies on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- The intersection point of two lines -/
def intersection (l1 l2 : Line) : Point :=
  sorry

theorem parallelogram_diagonals_fixed_points 
  (l : Line) (A B C D : Point) 
  (hA : on_line A l) (hB : on_line B l) (hC : on_line C l) (hD : on_line D l)
  (p1 p2 : Parallelogram) 
  (hp1 : p1.a = A ∧ p1.c = B) (hp2 : p2.a = C ∧ p2.c = D) :
  ∃ (P Q : Point), 
    (on_line P l ∧ on_line Q l) ∧
    ((on_line (intersection (diagonal p1) l) l ∧ 
      (intersection (diagonal p1) l = P ∨ intersection (diagonal p1) l = Q)) ∧
     (on_line (intersection (diagonal p2) l) l ∧ 
      (intersection (diagonal p2) l = P ∨ intersection (diagonal p2) l = Q))) :=
  sorry

end parallelogram_diagonals_fixed_points_l664_66455


namespace min_value_of_f_l664_66469

open Real

/-- The minimum value of f(x) = (e^x - a)^2 + (e^{-x} - a)^2 for 0 < a < 2 is 2(a - 1)^2 -/
theorem min_value_of_f (a : ℝ) (ha : 0 < a) (ha' : a < 2) :
  (∀ x : ℝ, (exp x - a)^2 + (exp (-x) - a)^2 ≥ 2 * (a - 1)^2) ∧
  (∃ x : ℝ, (exp x - a)^2 + (exp (-x) - a)^2 = 2 * (a - 1)^2) :=
by sorry

end min_value_of_f_l664_66469


namespace wheel_probability_l664_66407

theorem wheel_probability (p_D p_E p_F : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_D + p_E + p_F = 1 → p_F = 3/8 := by
  sorry

end wheel_probability_l664_66407


namespace fraction_simplification_l664_66436

theorem fraction_simplification : 
  (3/7 + 2/3) / (5/11 + 3/8) = 119/90 := by sorry

end fraction_simplification_l664_66436


namespace modulo_23_equivalence_l664_66483

theorem modulo_23_equivalence :
  ∃! n : ℕ, n < 23 ∧ 47582 % 23 = n :=
by
  use 3
  constructor
  · simp
    sorry
  · intro m ⟨hm, hmeq⟩
    sorry

#check modulo_23_equivalence

end modulo_23_equivalence_l664_66483


namespace function_parity_and_ranges_l664_66404

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2*a*x - a

def F (x : ℝ) : ℝ := x * f a x

def g (x : ℝ) : ℝ := -Real.exp x

theorem function_parity_and_ranges :
  (∀ x, f a x = f a (-x) ↔ a = 0) ∧
  (∃ m₁ m₂, m₁ = -16/3 ∧ m₂ = 112/729 ∧
    ∀ m, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      F a x₁ = m ∧ F a x₂ = m ∧ F a x₃ = m) ↔ m₁ < m ∧ m < m₂) ∧
  (∃ a₁ a₂, a₁ = Real.log 2 - 1 ∧ a₂ = 1/2 ∧
    (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.exp 1 ∧ x₁ > x₂ →
      |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔ a₁ ≤ a ∧ a ≤ a₂) :=
by sorry

end

end function_parity_and_ranges_l664_66404


namespace age_difference_of_children_l664_66425

theorem age_difference_of_children (n : ℕ) (sum_ages : ℕ) (eldest_age : ℕ) (d : ℕ) : 
  n = 5 → 
  sum_ages = 50 → 
  eldest_age = 14 → 
  sum_ages = n * eldest_age - (d * (n * (n - 1)) / 2) → 
  d = 2 := by
sorry

end age_difference_of_children_l664_66425


namespace base_8_representation_of_157_digit_count_of_157_base_8_l664_66491

def to_base_8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem base_8_representation_of_157 :
  to_base_8 157 = [2, 3, 5] :=
sorry

theorem digit_count_of_157_base_8 :
  (to_base_8 157).length = 3 :=
sorry

end base_8_representation_of_157_digit_count_of_157_base_8_l664_66491


namespace three_digit_number_transformation_l664_66402

theorem three_digit_number_transformation (n : ℕ) (x y z : ℕ) : 
  x * 100 + y * 10 + z = 178 → 
  n = 2 → 
  (x + n) * 100 + (y - n) * 10 + (z - n) = n * (x * 100 + y * 10 + z) := by
sorry

end three_digit_number_transformation_l664_66402


namespace road_section_last_point_location_l664_66415

/-- Given a road section from start_point to end_point divided into num_sections equal parts,
    the location of the last point is equal to the end_point. -/
theorem road_section_last_point_location
  (start_point end_point : ℝ)
  (num_sections : ℕ)
  (h1 : start_point = 0.35)
  (h2 : end_point = 0.37)
  (h3 : num_sections = 4)
  (h4 : start_point < end_point) :
  start_point + num_sections * ((end_point - start_point) / num_sections) = end_point :=
by sorry

end road_section_last_point_location_l664_66415


namespace sphere_triangle_distance_l664_66474

/-- The distance between the center of a sphere and the plane of a tangent isosceles triangle -/
theorem sphere_triangle_distance (r : ℝ) (a b : ℝ) (h_sphere : r = 8) 
  (h_triangle : a = 13 ∧ b = 10) (h_isosceles : a ≠ b) (h_tangent : True) :
  ∃ (d : ℝ), d = (20 * Real.sqrt 7) / 3 ∧ 
  d^2 = r^2 - (b/2)^2 / (1 - (b/(2*a))^2) := by
  sorry

end sphere_triangle_distance_l664_66474


namespace b_work_time_l664_66461

-- Define the work rates for A, B, C, and D
def A : ℚ := 1 / 5
def C : ℚ := 2 / 5 - A
def B : ℚ := 1 / 4 - C
def D : ℚ := 1 / 2 - B - C

-- State the theorem
theorem b_work_time : (1 : ℚ) / B = 20 := by sorry

end b_work_time_l664_66461


namespace bulbs_chosen_l664_66492

theorem bulbs_chosen (total_bulbs : ℕ) (defective_bulbs : ℕ) (prob_at_least_one_defective : ℝ) :
  total_bulbs = 24 →
  defective_bulbs = 4 →
  prob_at_least_one_defective = 0.3115942028985508 →
  ∃ n : ℕ, n = 2 ∧ (1 - (total_bulbs - defective_bulbs : ℝ) / total_bulbs) ^ n = prob_at_least_one_defective :=
by sorry

end bulbs_chosen_l664_66492


namespace complex_multiplication_equal_parts_l664_66462

theorem complex_multiplication_equal_parts (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
sorry

end complex_multiplication_equal_parts_l664_66462
