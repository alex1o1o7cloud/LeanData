import Mathlib

namespace triangle_properties_l2431_243112

theorem triangle_properties (a b c : ℝ) (h_ratio : (a, b, c) = (5 * 2, 12 * 2, 13 * 2)) 
  (h_perimeter : a + b + c = 60) : 
  (a^2 + b^2 = c^2) ∧ (a * b / 2 > 100) := by
  sorry

end triangle_properties_l2431_243112


namespace households_using_all_brands_l2431_243114

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : ℕ
  none : ℕ
  only_x : ℕ
  only_y : ℕ
  only_z : ℕ
  ratio_all_to_two : ℕ
  ratio_all_to_one : ℕ

/-- Calculates the number of households using all three brands of soap -/
def households_using_all (survey : SoapSurvey) : ℕ :=
  (survey.only_x + survey.only_y + survey.only_z) / survey.ratio_all_to_one

/-- Theorem stating the number of households using all three brands of soap -/
theorem households_using_all_brands (survey : SoapSurvey) 
  (h1 : survey.total = 5000)
  (h2 : survey.none = 1200)
  (h3 : survey.only_x = 800)
  (h4 : survey.only_y = 600)
  (h5 : survey.only_z = 300)
  (h6 : survey.ratio_all_to_two = 5)
  (h7 : survey.ratio_all_to_one = 10) :
  households_using_all survey = 170 := by
  sorry


end households_using_all_brands_l2431_243114


namespace determinant_equality_l2431_243195

theorem determinant_equality (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 7 →
  Matrix.det ![![x + 2*z, y + 2*w], ![z, w]] = 7 := by
  sorry

end determinant_equality_l2431_243195


namespace fourth_term_largest_l2431_243197

theorem fourth_term_largest (x : ℝ) : 
  (5/8 < x ∧ x < 20/21) ↔ 
  (∀ k : ℕ, k ≠ 4 → 
    Nat.choose 10 3 * (5^7) * (3*x)^3 ≥ Nat.choose 10 (k-1) * (5^(10-(k-1))) * (3*x)^(k-1)) :=
by sorry

end fourth_term_largest_l2431_243197


namespace max_m_value_l2431_243168

def M (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 ≤ -1 ∧ p.2 ≤ m}

theorem max_m_value :
  ∃ m : ℝ, m = 1 ∧
  (∀ m' : ℝ, (∀ p ∈ M m', p.1 * 2^p.2 - p.2 - 3*p.1 ≥ 0) →
  m' ≤ m) ∧
  (∀ p ∈ M m, p.1 * 2^p.2 - p.2 - 3*p.1 ≥ 0) :=
sorry

end max_m_value_l2431_243168


namespace derivative_ln_plus_x_l2431_243121

open Real

theorem derivative_ln_plus_x (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x + x) x = (x + 1) / x := by
sorry

end derivative_ln_plus_x_l2431_243121


namespace two_digit_prime_difference_l2431_243172

theorem two_digit_prime_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 90 → x - y = 10 := by
  sorry

end two_digit_prime_difference_l2431_243172


namespace car_cleaning_ratio_l2431_243188

/-- Given the total time spent cleaning a car and the time spent cleaning the outside,
    calculate the ratio of time spent cleaning the inside to the time spent cleaning the outside. -/
theorem car_cleaning_ratio (total_time outside_time inside_time : ℕ) : 
  total_time = outside_time + inside_time → 
  outside_time = 80 → 
  total_time = 100 → 
  (inside_time : ℚ) / outside_time = 1 / 4 := by
  sorry

end car_cleaning_ratio_l2431_243188


namespace nested_squares_segment_length_l2431_243120

/-- Given four nested squares with known segment lengths, prove that the length of GH
    is the sum of lengths AB, CD, and FE. -/
theorem nested_squares_segment_length 
  (AB CD FE : ℝ) 
  (h1 : AB = 11) 
  (h2 : CD = 5) 
  (h3 : FE = 13) : 
  ∃ GH : ℝ, GH = AB + CD + FE :=
by sorry

end nested_squares_segment_length_l2431_243120


namespace range_of_m_l2431_243150

-- Define the propositions p and q
def p (x : ℝ) : Prop := |2*x - 1| ≤ 5
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - 9*m^2 ≤ 0

-- Define the set of x that satisfies ¬p
def not_p_set : Set ℝ := {x | ¬(p x)}

-- Define the set of x that satisfies ¬q
def not_q_set (m : ℝ) : Set ℝ := {x | ¬(q x m)}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, x ∈ not_p_set → x ∈ not_q_set m) →
  (∃ x : ℝ, x ∈ not_q_set m ∧ x ∉ not_p_set) →
  m ∈ Set.Ioo 0 (1/3) ∪ {1/3} :=
sorry

end range_of_m_l2431_243150


namespace two_digit_sum_reverse_l2431_243173

theorem two_digit_sum_reverse : 
  (∃! n : Nat, n = (Finset.filter 
    (fun p : Nat × Nat => 
      let (a, b) := p
      0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (10 * a + b) + (10 * b + a) = 143)
    (Finset.product (Finset.range 10) (Finset.range 10))).card
  ∧ n = 6) := by sorry

end two_digit_sum_reverse_l2431_243173


namespace line_plane_perpendicularity_l2431_243132

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (c : Line) (α β : Plane) :
  contained_in c α → perpendicular c β → plane_perpendicular α β :=
by sorry

end line_plane_perpendicularity_l2431_243132


namespace water_tank_capacity_l2431_243108

theorem water_tank_capacity : ∀ x : ℚ,
  (5/6 : ℚ) * x - 30 = (4/5 : ℚ) * x → x = 900 := by
  sorry

end water_tank_capacity_l2431_243108


namespace toby_speed_proof_l2431_243164

/-- Toby's speed when pulling an unloaded sled -/
def unloaded_speed : ℝ := 20

/-- Distance of the first loaded part of the journey -/
def loaded_distance1 : ℝ := 180

/-- Distance of the first unloaded part of the journey -/
def unloaded_distance1 : ℝ := 120

/-- Distance of the second loaded part of the journey -/
def loaded_distance2 : ℝ := 80

/-- Distance of the second unloaded part of the journey -/
def unloaded_distance2 : ℝ := 140

/-- Total time of the journey -/
def total_time : ℝ := 39

/-- Toby's speed when pulling a loaded sled -/
def loaded_speed : ℝ := 10

theorem toby_speed_proof :
  (loaded_distance1 / loaded_speed + unloaded_distance1 / unloaded_speed +
   loaded_distance2 / loaded_speed + unloaded_distance2 / unloaded_speed) = total_time :=
by sorry

end toby_speed_proof_l2431_243164


namespace solution_when_a_is_3_solution_when_a_is_neg_l2431_243154

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (1 - 2*a) * x - 2 < 0

-- Define the solution set for a = 3
def solution_set_a3 : Set ℝ :=
  {x | -1/3 < x ∧ x < 2}

-- Define the solution set for a < 0
def solution_set_a_neg (a : ℝ) : Set ℝ :=
  if -1/2 < a ∧ a < 0 then
    {x | x < 2 ∨ x > -1/a}
  else if a = -1/2 then
    {x | x ≠ 2}
  else
    {x | x > 2 ∨ x < -1/a}

-- Theorem for a = 3
theorem solution_when_a_is_3 :
  ∀ x, x ∈ solution_set_a3 ↔ quadratic_inequality 3 x :=
sorry

-- Theorem for a < 0
theorem solution_when_a_is_neg :
  ∀ a, a < 0 → ∀ x, x ∈ solution_set_a_neg a ↔ quadratic_inequality a x :=
sorry

end solution_when_a_is_3_solution_when_a_is_neg_l2431_243154


namespace trig_identity_l2431_243199

theorem trig_identity (α : ℝ) : 
  (1 + 1 / Real.cos (2 * α) + Real.tan (2 * α)) * (1 - 1 / Real.cos (2 * α) + Real.tan (2 * α)) = 2 * Real.tan (2 * α) := by
  sorry

end trig_identity_l2431_243199


namespace rectangle_area_l2431_243100

/-- A rectangle with diagonal length x and length twice its width has area (2/5)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ (w l : ℝ),
  w > 0 ∧ l > 0 ∧ l = 2 * w ∧ w^2 + l^2 = x^2 ∧ w * l = (2/5) * x^2 := by
  sorry


end rectangle_area_l2431_243100


namespace cricket_team_ratio_l2431_243109

theorem cricket_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) :
  total_players = 70 →
  throwers = 37 →
  right_handed = 59 →
  (total_players - throwers : ℚ) / (total_players - throwers) = 3 / 4 :=
by sorry

end cricket_team_ratio_l2431_243109


namespace auto_finance_credit_l2431_243111

/-- Proves that the credit extended by automobile finance companies is $40 billion given the specified conditions -/
theorem auto_finance_credit (total_credit : ℝ) (auto_credit_percentage : ℝ) (finance_companies_fraction : ℝ)
  (h1 : total_credit = 342.857)
  (h2 : auto_credit_percentage = 0.35)
  (h3 : finance_companies_fraction = 1/3) :
  finance_companies_fraction * (auto_credit_percentage * total_credit) = 40 := by
  sorry

end auto_finance_credit_l2431_243111


namespace min_green_fraction_of_4x4x4_cube_l2431_243185

/-- Represents a cube with colored unit cubes -/
structure ColoredCube where
  edge_length : ℕ
  total_cubes : ℕ
  blue_cubes : ℕ
  green_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : ColoredCube) : ℕ := 6 * c.edge_length^2

/-- Calculates the minimum visible green surface area -/
def min_green_surface_area (c : ColoredCube) : ℕ := c.green_cubes - 4

theorem min_green_fraction_of_4x4x4_cube (c : ColoredCube) 
  (h1 : c.edge_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.blue_cubes = 56)
  (h4 : c.green_cubes = 8) :
  (min_green_surface_area c : ℚ) / (surface_area c : ℚ) = 1 / 24 := by
  sorry

end min_green_fraction_of_4x4x4_cube_l2431_243185


namespace two_correct_statements_l2431_243103

/-- A statement about triangles -/
inductive TriangleStatement
  | altitudes_intersect : TriangleStatement
  | medians_intersect_inside : TriangleStatement
  | right_triangle_one_altitude : TriangleStatement
  | angle_bisectors_intersect : TriangleStatement

/-- Predicate to check if a statement is correct -/
def is_correct (s : TriangleStatement) : Prop :=
  match s with
  | TriangleStatement.altitudes_intersect => true
  | TriangleStatement.medians_intersect_inside => true
  | TriangleStatement.right_triangle_one_altitude => false
  | TriangleStatement.angle_bisectors_intersect => true

/-- The main theorem to prove -/
theorem two_correct_statements :
  ∃ (s1 s2 : TriangleStatement),
    s1 ≠ s2 ∧
    is_correct s1 ∧
    is_correct s2 ∧
    ∀ (s : TriangleStatement),
      s ≠ s1 ∧ s ≠ s2 → ¬(is_correct s) :=
sorry

end two_correct_statements_l2431_243103


namespace intersection_A_B_l2431_243140

-- Define set A
def A : Set ℝ := {x | ∃ y : ℝ, y^2 = x}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_A_B_l2431_243140


namespace maintenance_building_length_l2431_243193

/-- The length of the maintenance building on a square playground -/
theorem maintenance_building_length : 
  ∀ (playground_side : ℝ) (building_width : ℝ) (uncovered_area : ℝ),
  playground_side = 12 →
  building_width = 5 →
  uncovered_area = 104 →
  ∃ (building_length : ℝ),
    building_length = 8 ∧
    building_length * building_width = playground_side^2 - uncovered_area :=
by sorry

end maintenance_building_length_l2431_243193


namespace sufficient_but_not_necessary_l2431_243161

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x^2 - 5*x + 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x + 6 > 0 ∧ ¬(x > 3)) := by
  sorry

end sufficient_but_not_necessary_l2431_243161


namespace M_greater_than_N_l2431_243110

theorem M_greater_than_N : ∀ x : ℝ, x^2 + 4*x - 2 > 6*x - 5 := by
  sorry

end M_greater_than_N_l2431_243110


namespace curve_and_intersection_l2431_243158

/-- The polar equation of curve C -/
def polar_equation (ρ θ a : ℝ) : Prop :=
  ρ * Real.sqrt (a^2 * Real.sin θ^2 + 4 * Real.cos θ^2) = 2 * a

/-- The Cartesian equation of curve C -/
def cartesian_equation (x y a : ℝ) : Prop :=
  4 * x^2 + a^2 * y^2 = 4 * a^2

/-- The parametric equations of line l -/
def line_equation (x y t : ℝ) : Prop :=
  x = Real.sqrt 3 + t ∧ y = 7 + Real.sqrt 3 * t

/-- Point P -/
def point_P : ℝ × ℝ := (0, 4)

/-- The distance product condition -/
def distance_product (a : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ), line_equation M.1 M.2 (M.1 - Real.sqrt 3) ∧
                   line_equation N.1 N.2 (N.1 - Real.sqrt 3) ∧
                   cartesian_equation M.1 M.2 a ∧
                   cartesian_equation N.1 N.2 a ∧
                   (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 *
                   (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 = 14^2

theorem curve_and_intersection :
  (∀ (ρ θ : ℝ), polar_equation ρ θ a ↔ cartesian_equation (ρ * Real.cos θ) (ρ * Real.sin θ) a) ∧
  (distance_product a → a = 2 * Real.sqrt 21 / 3) := by
  sorry

end curve_and_intersection_l2431_243158


namespace linear_function_is_shifted_odd_exponential_function_is_not_shifted_odd_sine_shifted_odd_condition_cubic_function_not_shifted_odd_condition_l2431_243144

/-- A function is a shifted odd function if there exists a real number m such that
    f(x+m) - f(m) is an odd function over ℝ. -/
def is_shifted_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f (x + m) - f m = -(f (-x + m) - f m)

/-- The function f(x) = 2x + 1 is a shifted odd function. -/
theorem linear_function_is_shifted_odd :
  is_shifted_odd_function (fun x => 2 * x + 1) :=
sorry

/-- The function g(x) = 2^x is not a shifted odd function. -/
theorem exponential_function_is_not_shifted_odd :
  ¬ is_shifted_odd_function (fun x => 2^x) :=
sorry

/-- For f(x) = sin(x + φ) to be a shifted odd function with shift difference π/4,
    φ must equal kπ - π/4 for some integer k. -/
theorem sine_shifted_odd_condition (φ : ℝ) :
  is_shifted_odd_function (fun x => Real.sin (x + φ)) ∧ 
  (∃ m : ℝ, m = π/4 ∧ ∀ x : ℝ, Real.sin (x + m + φ) - Real.sin (m + φ) = -(Real.sin (-x + m + φ) - Real.sin (m + φ))) ↔
  ∃ k : ℤ, φ = k * π - π/4 :=
sorry

/-- For f(x) = x^3 + bx^2 + cx to not be a shifted odd function for any m in [-1/2, +∞),
    b must be greater than 3/2, and c can be any real number. -/
theorem cubic_function_not_shifted_odd_condition (b c : ℝ) :
  (∀ m : ℝ, m ≥ -1/2 → ¬ is_shifted_odd_function (fun x => x^3 + b*x^2 + c*x)) ↔
  b > 3/2 :=
sorry

end linear_function_is_shifted_odd_exponential_function_is_not_shifted_odd_sine_shifted_odd_condition_cubic_function_not_shifted_odd_condition_l2431_243144


namespace bess_throws_20_meters_l2431_243134

/-- Represents the Frisbee throwing scenario -/
structure FrisbeeScenario where
  bess_throws : ℕ           -- Number of times Bess throws
  holly_throws : ℕ          -- Number of times Holly throws
  holly_distance : ℕ        -- Distance Holly throws in meters
  total_distance : ℕ        -- Total distance traveled by all Frisbees in meters

/-- Calculates Bess's throwing distance given a FrisbeeScenario -/
def bess_distance (scenario : FrisbeeScenario) : ℕ :=
  (scenario.total_distance - scenario.holly_throws * scenario.holly_distance) / (2 * scenario.bess_throws)

/-- Theorem stating that Bess's throwing distance is 20 meters in the given scenario -/
theorem bess_throws_20_meters (scenario : FrisbeeScenario) 
  (h1 : scenario.bess_throws = 4)
  (h2 : scenario.holly_throws = 5)
  (h3 : scenario.holly_distance = 8)
  (h4 : scenario.total_distance = 200) :
  bess_distance scenario = 20 := by
  sorry

end bess_throws_20_meters_l2431_243134


namespace log_75843_bounds_l2431_243104

theorem log_75843_bounds : ∃ (c d : ℤ), (c : ℝ) < Real.log 75843 / Real.log 10 ∧ 
  Real.log 75843 / Real.log 10 < (d : ℝ) ∧ c = 4 ∧ d = 5 ∧ c + d = 9 := by
  sorry

#check log_75843_bounds

end log_75843_bounds_l2431_243104


namespace arithmetic_sequence_problem_l2431_243136

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- Theorem: For an arithmetic sequence with a_2 = 1 and S_4 = 8, a_5 = 7 and S_10 = 80 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.a 2 = 1) (h2 : seq.S 4 = 8) : 
    seq.a 5 = 7 ∧ seq.S 10 = 80 := by
  sorry

end arithmetic_sequence_problem_l2431_243136


namespace quadratic_solution_difference_l2431_243166

theorem quadratic_solution_difference (x : ℝ) : 
  (x^2 - 5*x + 12 = 2*x + 60) → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1^2 - 5*x1 + 12 = 2*x1 + 60) ∧ 
    (x2^2 - 5*x2 + 12 = 2*x2 + 60) ∧ 
    |x1 - x2| = Real.sqrt 241 := by
  sorry

end quadratic_solution_difference_l2431_243166


namespace quadratic_radicals_combination_l2431_243106

theorem quadratic_radicals_combination (a : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (1 + a) = k * (4 - 2*a)) ↔ a = 1 :=
by sorry

end quadratic_radicals_combination_l2431_243106


namespace distance_between_points_l2431_243165

theorem distance_between_points (x : ℝ) : 
  |(3 + x) - (3 - x)| = 8 → |x| = 4 := by
sorry

end distance_between_points_l2431_243165


namespace circle_radius_proof_l2431_243183

theorem circle_radius_proof (r : ℝ) : 
  r > 0 → 
  (π * r^2 = 3 * (2 * π * r)) → 
  (π * r^2 + 2 * π * r = 100 * π) → 
  r = 12.5 := by
sorry

end circle_radius_proof_l2431_243183


namespace sqrt_eighteen_div_sqrt_two_eq_three_l2431_243138

theorem sqrt_eighteen_div_sqrt_two_eq_three : 
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end sqrt_eighteen_div_sqrt_two_eq_three_l2431_243138


namespace equation_solutions_l2431_243182

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧
    x1^2 - 4*x1 - 1 = 0 ∧ x2^2 - 4*x2 - 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -3 ∧ x2 = -2 ∧
    (x1 + 3)^2 = x1 + 3 ∧ (x2 + 3)^2 = x2 + 3) :=
by
  sorry

end equation_solutions_l2431_243182


namespace pattern_holds_squares_in_figure_150_l2431_243146

/-- The number of unit squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The sequence of unit squares follows the given pattern for the first four figures -/
theorem pattern_holds : f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 := by sorry

/-- The number of unit squares in figure 150 is 67951 -/
theorem squares_in_figure_150 : f 150 = 67951 := by sorry

end pattern_holds_squares_in_figure_150_l2431_243146


namespace rocket_max_height_l2431_243152

/-- Rocket's maximum height calculation --/
theorem rocket_max_height (a g : ℝ) (τ : ℝ) (h : a > g) (h_a : a = 30) (h_g : g = 10) (h_τ : τ = 30) :
  let v₀ := a * τ
  let y₀ := a * τ^2 / 2
  let t := v₀ / g
  let y_max := y₀ + v₀ * t - g * t^2 / 2
  y_max = 54000 ∧ y_max > 50000 := by
  sorry

#check rocket_max_height

end rocket_max_height_l2431_243152


namespace max_stamps_purchasable_l2431_243124

def stamp_price : ℕ := 35
def discount_threshold : ℕ := 100
def discount_rate : ℚ := 5 / 100
def budget : ℕ := 3200

theorem max_stamps_purchasable :
  let max_stamps := (budget / stamp_price : ℕ)
  let discounted_price := stamp_price * (1 - discount_rate)
  let max_stamps_with_discount := (budget / discounted_price).floor
  (max_stamps_with_discount ≤ discount_threshold) ∧
  (max_stamps = 91) := by
sorry

end max_stamps_purchasable_l2431_243124


namespace inequality_holds_only_for_m_equals_negative_four_l2431_243149

theorem inequality_holds_only_for_m_equals_negative_four :
  ∀ m : ℝ, (∀ x : ℝ, |2*x - m| ≤ |3*x + 6|) ↔ m = -4 := by
  sorry

end inequality_holds_only_for_m_equals_negative_four_l2431_243149


namespace x_greater_than_sin_x_negation_of_implication_and_sufficient_not_necessary_for_or_negation_of_forall_x_minus_ln_x_positive_l2431_243179

-- Statement 1
theorem x_greater_than_sin_x (x : ℝ) (h : x > 0) : x > Real.sin x := by sorry

-- Statement 2
theorem negation_of_implication :
  (¬ (∀ x : ℝ, x - Real.sin x = 0 → x = 0)) ↔
  (∃ x : ℝ, x - Real.sin x ≠ 0 ∧ x ≠ 0) := by sorry

-- Statement 3
theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by sorry

-- Statement 4
theorem negation_of_forall_x_minus_ln_x_positive :
  (¬ (∀ x : ℝ, x - Real.log x > 0)) ↔
  (∃ x : ℝ, x - Real.log x ≤ 0) := by sorry

end x_greater_than_sin_x_negation_of_implication_and_sufficient_not_necessary_for_or_negation_of_forall_x_minus_ln_x_positive_l2431_243179


namespace unique_factorial_sum_l2431_243105

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_factorial_sum (x a b c d : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < x)
  (h6 : a ≤ b) (h7 : b ≤ c) (h8 : c ≤ d) (h9 : d < x)
  (h10 : factorial x = factorial a + factorial b + factorial c + factorial d) :
  x = 4 ∧ a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 :=
by sorry

end unique_factorial_sum_l2431_243105


namespace initial_number_proof_l2431_243198

theorem initial_number_proof (N : ℤ) : (N + 3) % 24 = 0 → N = 21 := by
  sorry

end initial_number_proof_l2431_243198


namespace sallys_garden_area_l2431_243102

/-- Represents a rectangular garden with fence posts. -/
structure GardenFence where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_post_ratio : ℕ

/-- Calculates the area of the garden given its fence configuration. -/
def garden_area (fence : GardenFence) : ℕ :=
  let short_side_posts := (fence.total_posts / 2) / (fence.long_side_post_ratio + 1)
  let long_side_posts := short_side_posts * fence.long_side_post_ratio
  let short_side_length := (short_side_posts - 1) * fence.post_spacing
  let long_side_length := (long_side_posts - 1) * fence.post_spacing
  short_side_length * long_side_length

/-- Theorem stating that Sally's garden has an area of 297 square yards. -/
theorem sallys_garden_area :
  let sally_fence := GardenFence.mk 24 3 3
  garden_area sally_fence = 297 := by
  sorry

end sallys_garden_area_l2431_243102


namespace transformer_current_load_transformer_current_load_is_700A_l2431_243116

theorem transformer_current_load : ℕ → Prop :=
  fun total_load =>
    let units_40A := 3
    let units_60A := 2
    let units_25A := 1
    let running_current_40A := 40
    let running_current_60A := 60
    let running_current_25A := 25
    let starting_multiplier_40A := 2
    let starting_multiplier_60A := 3
    let starting_multiplier_25A := 4
    let total_start_current_40A := units_40A * running_current_40A * starting_multiplier_40A
    let total_start_current_60A := units_60A * running_current_60A * starting_multiplier_60A
    let total_start_current_25A := units_25A * running_current_25A * starting_multiplier_25A
    total_load = total_start_current_40A + total_start_current_60A + total_start_current_25A

theorem transformer_current_load_is_700A : transformer_current_load 700 := by
  sorry

end transformer_current_load_transformer_current_load_is_700A_l2431_243116


namespace committee_probability_l2431_243156

/-- The number of members in the Grammar club -/
def total_members : ℕ := 20

/-- The number of boys in the Grammar club -/
def num_boys : ℕ := 10

/-- The number of girls in the Grammar club -/
def num_girls : ℕ := 10

/-- The size of the committee to be chosen -/
def committee_size : ℕ := 4

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size)) / 
   Nat.choose total_members committee_size = 295 / 323 := by
  sorry

end committee_probability_l2431_243156


namespace max_value_condition_l2431_243159

/-- The expression 2005 - (x + y)^2 takes its maximum value when x = -y -/
theorem max_value_condition (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (x + y)^2 ≥ 2005 - (a + b)^2) → x = -y := by
sorry

end max_value_condition_l2431_243159


namespace total_members_in_math_club_l2431_243135

def math_club (female_members : ℕ) (male_members : ℕ) : Prop :=
  male_members = 2 * female_members

theorem total_members_in_math_club (female_members : ℕ) 
  (h1 : female_members = 6) 
  (h2 : math_club female_members (2 * female_members)) : 
  female_members + 2 * female_members = 18 := by
  sorry

end total_members_in_math_club_l2431_243135


namespace sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos_l2431_243171

open Real

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define f' as the derivative of f
variable (f' : ℝ → ℝ)

-- Axiom: f is an odd function
axiom f_odd (x : ℝ) : f (-x) = -f x

-- Axiom: f' is the derivative of f
axiom f'_is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Axiom: For x in (0, π/2) ∪ (π/2, π), f(x) + f'(x)tan(x) > 0
axiom f_plus_f'_tan_pos (x : ℝ) (h1 : 0 < x) (h2 : x < π) (h3 : x ≠ π/2) :
  f x + f' x * tan x > 0

-- Theorem to prove
theorem sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos :
  Real.sqrt 2 * f (π/4) + f (-π/6) > 0 := by
  sorry

end sqrt_two_f_pi_fourth_plus_f_neg_pi_sixth_pos_l2431_243171


namespace oil_purchase_amount_l2431_243191

/-- Proves that the amount spent on oil is Rs. 600 given the conditions of the problem -/
theorem oil_purchase_amount (original_price : ℝ) (reduced_price : ℝ) (additional_oil : ℝ) 
  (h1 : reduced_price = original_price * 0.75)
  (h2 : reduced_price = 30)
  (h3 : additional_oil = 5) :
  ∃ (amount_spent : ℝ), 
    amount_spent / reduced_price - amount_spent / original_price = additional_oil ∧ 
    amount_spent = 600 := by
  sorry

end oil_purchase_amount_l2431_243191


namespace triangle_properties_l2431_243137

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  c = Real.sqrt 2 →
  Real.cos A = -(Real.sqrt 2) / 4 →
  a = b →
  b = c →
  c = a →
  Real.sin C = Real.sqrt 7 / 4 ∧
  b = 1 ∧
  Real.cos (2 * A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry

end triangle_properties_l2431_243137


namespace monotonicity_condition_l2431_243186

/-- Represents a voting system with n voters and m candidates. -/
structure VotingSystem where
  n : ℕ  -- number of voters
  m : ℕ  -- number of candidates
  k : ℕ  -- number of top choices each voter selects

/-- Represents a poll profile (arrangement of candidate rankings by voters). -/
def PollProfile (vs : VotingSystem) := Fin vs.n → (Fin vs.m → Fin vs.m)

/-- Determines if a candidate is a winner in a given poll profile. -/
def isWinner (vs : VotingSystem) (profile : PollProfile vs) (candidate : Fin vs.m) : Prop :=
  sorry

/-- Determines if one profile is a-good compared to another. -/
def isAGood (vs : VotingSystem) (a : Fin vs.m) (R R' : PollProfile vs) : Prop :=
  ∀ (voter : Fin vs.n) (candidate : Fin vs.m),
    (R voter candidate > R voter a) → (R' voter candidate > R' voter a)

/-- Defines the monotonicity property for a voting system. -/
def isMonotone (vs : VotingSystem) : Prop :=
  ∀ (R R' : PollProfile vs) (a : Fin vs.m),
    isWinner vs R a → isAGood vs a R R' → isWinner vs R' a

/-- The main theorem stating the condition for monotonicity. -/
theorem monotonicity_condition (vs : VotingSystem) :
  isMonotone vs ↔ vs.k > (vs.m * (vs.n - 1)) / vs.n :=
  sorry

end monotonicity_condition_l2431_243186


namespace fixed_point_exponential_function_l2431_243128

theorem fixed_point_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by
  sorry

end fixed_point_exponential_function_l2431_243128


namespace heaviest_weight_in_geometric_progression_l2431_243194

/-- Given four weights in geometric progression, the heaviest can be found using a balance twice -/
theorem heaviest_weight_in_geometric_progression 
  (b : ℝ) (d : ℝ) (h_b_pos : b > 0) (h_d_gt_one : d > 1) :
  ∃ (n : ℕ), n ≤ 2 ∧ 
    (∀ i : Fin 4, b * d ^ i.val ≤ b * d ^ 3) ∧
    (∀ i : Fin 4, i.val ≠ 3 → b * d ^ i.val < b * d ^ 3) := by
  sorry

end heaviest_weight_in_geometric_progression_l2431_243194


namespace inverse_proportion_problem_l2431_243122

theorem inverse_proportion_problem (x y : ℝ) (C : ℝ) :
  (x * y = C) →  -- x and y are inversely proportional
  (x + y = 32) →
  (x - y = 8) →
  (x = 4) →
  y = 60 := by sorry

end inverse_proportion_problem_l2431_243122


namespace telescope_purchase_problem_l2431_243141

theorem telescope_purchase_problem (joan_price karl_price : ℝ) 
  (h1 : joan_price + karl_price = 400)
  (h2 : 2 * joan_price = karl_price + 74) :
  joan_price = 158 := by
  sorry

end telescope_purchase_problem_l2431_243141


namespace special_triangle_angles_l2431_243170

/-- A triangle with a 90° angle that is three times the smallest angle has angles 90°, 60°, and 30° and is right-angled. -/
theorem special_triangle_angles :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a = 90 →
  a = 3 * c →
  (a = 90 ∧ b = 60 ∧ c = 30) ∧ (∃ x, x = 90) :=
by sorry

end special_triangle_angles_l2431_243170


namespace polynomial_factorization_constant_term_l2431_243174

theorem polynomial_factorization_constant_term (a b c d e f : ℝ) 
  (p : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) :
  (∀ x, p x = x^8 - 4*x^7 + 7*x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∀ x, p x = (x - x₁) * (x - x₂) * (x - x₃) * (x - x₄) * (x - x₅) * (x - x₆) * (x - x₇) * (x - x₈)) →
  (x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₆ > 0 ∧ x₇ > 0 ∧ x₈ > 0) →
  f = 1 / 256 := by
  sorry

end polynomial_factorization_constant_term_l2431_243174


namespace equation_solution_l2431_243177

theorem equation_solution : ∃ x : ℝ, 5*x + 9*x = 420 - 12*(x - 4) ∧ x = 18 := by
  sorry

end equation_solution_l2431_243177


namespace complex_modulus_problem_l2431_243126

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = 2 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2431_243126


namespace tan_two_implies_sum_l2431_243176

theorem tan_two_implies_sum (θ : ℝ) (h : Real.tan θ = 2) : 
  2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end tan_two_implies_sum_l2431_243176


namespace combined_age_proof_l2431_243163

def jeremy_age : ℕ := 66

theorem combined_age_proof (amy_age chris_age : ℕ) 
  (h1 : amy_age = jeremy_age / 3)
  (h2 : chris_age = 2 * amy_age) :
  amy_age + jeremy_age + chris_age = 132 := by
  sorry

end combined_age_proof_l2431_243163


namespace fraction_to_sofia_is_one_twelfth_l2431_243178

/-- Represents the initial egg distribution and sharing problem --/
structure EggDistribution where
  mia_eggs : ℕ
  sofia_eggs : ℕ
  pablo_eggs : ℕ
  lucas_eggs : ℕ
  (sofia_eggs_def : sofia_eggs = 3 * mia_eggs)
  (pablo_eggs_def : pablo_eggs = 4 * sofia_eggs)
  (lucas_eggs_def : lucas_eggs = 0)

/-- Calculates the fraction of Pablo's eggs given to Sofia --/
def fraction_to_sofia (d : EggDistribution) : ℚ :=
  let total_eggs := d.mia_eggs + d.sofia_eggs + d.pablo_eggs + d.lucas_eggs
  let equal_share := total_eggs / 4
  let sofia_needs := equal_share - d.sofia_eggs
  sofia_needs / d.pablo_eggs

/-- Theorem stating that the fraction of Pablo's eggs given to Sofia is 1/12 --/
theorem fraction_to_sofia_is_one_twelfth (d : EggDistribution) :
  fraction_to_sofia d = 1 / 12 := by
  sorry

end fraction_to_sofia_is_one_twelfth_l2431_243178


namespace smallest_sum_of_factors_of_72_l2431_243139

theorem smallest_sum_of_factors_of_72 :
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c = 72 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (x y z : ℕ), 
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      x * y * z = 72 →
      x > 0 ∧ y > 0 ∧ z > 0 →
      a + b + c ≤ x + y + z ∧
      a + b + c = 13 :=
by sorry

end smallest_sum_of_factors_of_72_l2431_243139


namespace minimum_cars_with_all_characteristics_l2431_243162

theorem minimum_cars_with_all_characteristics 
  (total : ℕ) 
  (zhiguli dark_colored male_drivers with_passengers : ℕ) 
  (h_total : total = 20)
  (h_zhiguli : zhiguli = 14)
  (h_dark : dark_colored = 15)
  (h_male : male_drivers = 17)
  (h_passengers : with_passengers = 18) :
  total - ((total - zhiguli) + (total - dark_colored) + (total - male_drivers) + (total - with_passengers)) = 4 := by
sorry

end minimum_cars_with_all_characteristics_l2431_243162


namespace even_function_property_l2431_243130

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property
  (f : ℝ → ℝ)
  (h_even : evenFunction f)
  (h_increasing : ∀ x y, x < y → x < 0 → f x < f y)
  (x₁ x₂ : ℝ)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : x₂ > 0)
  (h_abs : |x₁| < |x₂|) :
  f (-x₁) > f (-x₂) := by
sorry

end even_function_property_l2431_243130


namespace starting_lineup_combinations_l2431_243190

/-- The number of members in the basketball team -/
def team_size : ℕ := 12

/-- The number of positions in the starting lineup -/
def lineup_size : ℕ := 5

/-- The number of ways to choose a starting lineup -/
def lineup_choices : ℕ := team_size * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)

theorem starting_lineup_combinations : 
  lineup_choices = 95040 :=
sorry

end starting_lineup_combinations_l2431_243190


namespace cricket_team_selection_l2431_243115

/-- The total number of players in the cricket team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of players to be chosen for the training squad -/
def squad_size : ℕ := 5

/-- The number of ways to choose the training squad under the given restrictions -/
def valid_selections : ℕ := 4356

theorem cricket_team_selection :
  (Nat.choose total_players squad_size) - (Nat.choose (total_players - num_quadruplets) 1) = valid_selections :=
sorry

end cricket_team_selection_l2431_243115


namespace division_of_decimals_l2431_243148

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := by
  sorry

end division_of_decimals_l2431_243148


namespace complex_equation_solution_l2431_243142

theorem complex_equation_solution (a : ℝ) (i : ℂ) (hi : i * i = -1) :
  (a + 2 * i) / (2 + i) = i → a = -1 := by
  sorry

end complex_equation_solution_l2431_243142


namespace prism_intersection_probability_l2431_243143

/-- A rectangular prism with dimensions 2, 3, and 5 units. -/
structure RectangularPrism where
  length : ℕ := 2
  width : ℕ := 3
  height : ℕ := 5

/-- The probability that three randomly chosen vertices of the prism
    form a plane intersecting the prism's interior. -/
def intersectionProbability (p : RectangularPrism) : ℚ :=
  11/14

/-- Theorem stating that the probability of three randomly chosen vertices
    forming a plane that intersects the interior of the given rectangular prism is 11/14. -/
theorem prism_intersection_probability (p : RectangularPrism) :
  intersectionProbability p = 11/14 := by
  sorry

end prism_intersection_probability_l2431_243143


namespace largest_initial_number_l2431_243147

theorem largest_initial_number : 
  ∃ (a b c d e : ℕ), 
    189 + a + b + c + d + e = 200 ∧ 
    a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
    189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
    ∀ (n : ℕ), n > 189 → 
      ¬(∃ (a' b' c' d' e' : ℕ), 
        n + a' + b' + c' + d' + e' = 200 ∧
        a' ≥ 2 ∧ b' ≥ 2 ∧ c' ≥ 2 ∧ d' ≥ 2 ∧ e' ≥ 2 ∧
        n % a' ≠ 0 ∧ n % b' ≠ 0 ∧ n % c' ≠ 0 ∧ n % d' ≠ 0 ∧ n % e' ≠ 0) :=
by sorry

end largest_initial_number_l2431_243147


namespace weight_problem_l2431_243167

/-- Given the average weights of three people and two pairs, prove the weight of one person. -/
theorem weight_problem (A B C : ℝ) : 
  (A + B + C) / 3 = 45 ∧ 
  (A + B) / 2 = 40 ∧ 
  (B + C) / 2 = 43 → 
  B = 31 := by
  sorry

end weight_problem_l2431_243167


namespace K_characterization_l2431_243155

/-- Function that reverses the digits of a positive integer in decimal notation -/
def f (n : ℕ+) : ℕ+ := sorry

/-- The set of all positive integers k such that, for any multiple n of k, k also divides f(n) -/
def K : Set ℕ+ :=
  {k : ℕ+ | ∀ n : ℕ+, k ∣ n → k ∣ f n}

/-- Theorem stating that K is equal to the set {1, 3, 9, 11, 33, 99} -/
theorem K_characterization : K = {1, 3, 9, 11, 33, 99} := by sorry

end K_characterization_l2431_243155


namespace quadratic_inequality_range_l2431_243192

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end quadratic_inequality_range_l2431_243192


namespace sqrt_sum_equation_l2431_243181

theorem sqrt_sum_equation (x : ℝ) (h : x ≥ 1/2) :
  (∃ y ∈ Set.Icc (1/2 : ℝ) 1, x = y ↔ Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1)) = Real.sqrt 2) ∧
  (¬ ∃ y ≥ 1/2, Real.sqrt (y + Real.sqrt (2*y - 1)) + Real.sqrt (y - Real.sqrt (2*y - 1)) = 1) ∧
  (x = 3/2 ↔ Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1)) = 2) :=
by sorry

end sqrt_sum_equation_l2431_243181


namespace problem_solution_l2431_243175

-- Define the solution set
def solution_set (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 1

-- Define the inequality
def inequality (x m : ℝ) : Prop := |x + 2| + |x - m| ≤ 3

theorem problem_solution :
  (∀ x, solution_set x ↔ inequality x 1) ∧
  ∀ a b c : ℝ, a^2 + 2*b^2 + 3*c^2 = 1 → -Real.sqrt 6 ≤ a + 2*b + 3*c ∧ a + 2*b + 3*c ≤ Real.sqrt 6 :=
sorry

end problem_solution_l2431_243175


namespace probability_two_yellow_marbles_l2431_243113

/-- The probability of drawing two yellow marbles successively from a jar -/
theorem probability_two_yellow_marbles 
  (blue : ℕ) (yellow : ℕ) (black : ℕ) 
  (h_blue : blue = 3)
  (h_yellow : yellow = 4)
  (h_black : black = 8) :
  let total := blue + yellow + black
  (yellow / total) * ((yellow - 1) / (total - 1)) = 2 / 35 := by
sorry

end probability_two_yellow_marbles_l2431_243113


namespace dance_attendance_l2431_243107

theorem dance_attendance (girls boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end dance_attendance_l2431_243107


namespace correct_installation_time_l2431_243187

/-- Calculates the time needed to install remaining windows in a skyscraper -/
def time_to_install_remaining (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) : ℕ :=
  (total_windows - installed_windows) * time_per_window

/-- Theorem stating that the time to install remaining windows is correct -/
theorem correct_installation_time (total_windows installed_windows time_per_window : ℕ)
  (h1 : installed_windows ≤ total_windows) :
  time_to_install_remaining total_windows installed_windows time_per_window =
  (total_windows - installed_windows) * time_per_window :=
by sorry

end correct_installation_time_l2431_243187


namespace soda_preference_result_l2431_243180

/-- The number of people who chose "Soda" in a survey about carbonated beverages -/
def soda_preference (total_surveyed : ℕ) (soda_angle : ℕ) : ℕ :=
  (total_surveyed * soda_angle) / 360

/-- Theorem stating that 243 people chose "Soda" in the survey -/
theorem soda_preference_result : soda_preference 540 162 = 243 := by
  sorry

end soda_preference_result_l2431_243180


namespace circle_symmetry_l2431_243169

def original_circle (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 = 3

def symmetric_circle (x y : ℝ) : Prop :=
  x^2 + 2*x + y^2 = 3

def symmetric_wrt_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_wrt_y_axis x₁ y₁ x₂ y₂ →
    symmetric_circle x₂ y₂ :=
by sorry

end circle_symmetry_l2431_243169


namespace intersection_A_complement_B_l2431_243119

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B (A : Set ℕ) 
  (h1 : A ⊆ U) 
  (h2 : B ⊆ U) 
  (h3 : (U \ (A ∪ B)) = {4}) : 
  A ∩ (U \ B) = {3} := by sorry

end intersection_A_complement_B_l2431_243119


namespace nested_fraction_simplification_l2431_243117

theorem nested_fraction_simplification :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end nested_fraction_simplification_l2431_243117


namespace brown_family_seating_l2431_243184

/-- The number of ways to seat n children in a circle. -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat b boys and g girls in a circle
    such that at least two boys are next to each other. -/
def boysNextToEachOther (b g : ℕ) : ℕ :=
  if b > g + 1 then circularArrangements (b + g) else 0

theorem brown_family_seating :
  boysNextToEachOther 5 3 = 5040 := by sorry

end brown_family_seating_l2431_243184


namespace jim_tree_planting_l2431_243189

/-- The age at which Jim started planting a new row of trees every year -/
def start_age : ℕ := sorry

/-- The number of trees Jim has initially -/
def initial_trees : ℕ := 2 * 4

/-- The number of trees Jim plants each year -/
def trees_per_year : ℕ := 4

/-- Jim's age when he doubles his trees -/
def final_age : ℕ := 15

/-- The total number of trees Jim has after doubling on his 15th birthday -/
def total_trees : ℕ := 56

theorem jim_tree_planting :
  2 * (initial_trees + trees_per_year * (final_age - start_age)) = total_trees := by sorry

end jim_tree_planting_l2431_243189


namespace circle_radius_is_two_l2431_243133

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y + 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 2

theorem circle_radius_is_two :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end circle_radius_is_two_l2431_243133


namespace arrangement_count_5_2_l2431_243131

/-- The number of ways to arrange n distinct objects and m pairs of 2 distinct objects each in a row,
    where the objects within each pair must be adjacent -/
def arrangementCount (n m : ℕ) : ℕ :=
  Nat.factorial (n + m) * (Nat.factorial 2)^m

/-- Theorem: The number of ways to arrange 5 distinct objects and 2 pairs of 2 distinct objects each
    in a row, where the objects within each pair must be adjacent, is equal to 7! * (2!)^2 -/
theorem arrangement_count_5_2 :
  arrangementCount 5 2 = 20160 := by
  sorry

end arrangement_count_5_2_l2431_243131


namespace sebastian_ticket_cost_l2431_243101

/-- The total cost of tickets for Sebastian and his parents -/
def total_cost (num_people : ℕ) (ticket_price : ℕ) (service_fee : ℕ) : ℕ :=
  num_people * ticket_price + service_fee

/-- Theorem stating that the total cost for Sebastian's tickets is $150 -/
theorem sebastian_ticket_cost :
  total_cost 3 44 18 = 150 := by
  sorry

end sebastian_ticket_cost_l2431_243101


namespace quadratic_triangle_area_l2431_243145

/-- Given a quadratic function y = ax^2 + bx + c where b^2 - 4ac > 0,
    the area of the triangle formed by its intersections with the x-axis and y-axis is |c|/(2|a|) -/
theorem quadratic_triangle_area (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let f : ℝ → ℝ := λ x => a*x^2 + b*x + c
  let triangle_area := (abs c) / (2 * abs a)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (1/2 * abs (x₁ - x₂) * abs c = triangle_area) :=
sorry

end quadratic_triangle_area_l2431_243145


namespace greatest_divisor_with_remainders_l2431_243123

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (1657 % n = 10 ∧ 2037 % n = 7) → n = 1 :=
by
  sorry

end greatest_divisor_with_remainders_l2431_243123


namespace sum_of_two_squares_l2431_243160

theorem sum_of_two_squares (u : ℕ) (h : Odd u) :
  ∃ (a b : ℕ), (3^(3*u) - 1) / (3^u - 1) = a^2 + b^2 := by
  sorry

end sum_of_two_squares_l2431_243160


namespace quadratic_range_on_interval_l2431_243151

def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_range_on_interval
  (a b c : ℝ)
  (ha : a > 0) :
  let range_min := min (g a b c (-1)) (g a b c 2)
  let range_max := max (g a b c (-1)) (max (g a b c 2) (g a b c (-b/(2*a))))
  ∀ x ∈ Set.Icc (-1 : ℝ) 2,
    range_min ≤ g a b c x ∧ g a b c x ≤ range_max :=
by sorry

end quadratic_range_on_interval_l2431_243151


namespace ratio_problem_l2431_243153

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2)
  (h3 : c / d = 5) :
  d / a = 1 / 30 := by
  sorry

end ratio_problem_l2431_243153


namespace largest_negative_congruent_to_one_mod_seventeen_l2431_243125

theorem largest_negative_congruent_to_one_mod_seventeen :
  ∃ (n : ℤ), 
    n = -1002 ∧ 
    n ≡ 1 [ZMOD 17] ∧ 
    n < 0 ∧ 
    -9999 ≤ n ∧
    ∀ (m : ℤ), m ≡ 1 [ZMOD 17] ∧ m < 0 ∧ -9999 ≤ m → m ≤ n :=
by sorry

end largest_negative_congruent_to_one_mod_seventeen_l2431_243125


namespace binomial_150_150_equals_1_l2431_243118

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_equals_1_l2431_243118


namespace tangent_line_difference_l2431_243196

/-- A curve defined by y = x^3 + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A line defined by y = kx + 1 -/
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 1

theorem tangent_line_difference (a b k : ℝ) :
  (curve a b 1 = 2) →  -- The curve passes through (1, 2)
  (line k 1 = 2) →     -- The line passes through (1, 2)
  (curve_derivative a 1 = k) →  -- The derivative of the curve at x=1 equals the slope of the line
  b - a = 5 := by
    sorry


end tangent_line_difference_l2431_243196


namespace max_plain_cupcakes_l2431_243129

structure Cupcakes :=
  (total : ℕ)
  (blueberries : ℕ)
  (sprinkles : ℕ)
  (frosting : ℕ)
  (pecans : ℕ)

def has_no_ingredients (c : Cupcakes) : ℕ :=
  c.total - (c.blueberries + c.sprinkles + c.frosting + c.pecans)

theorem max_plain_cupcakes (c : Cupcakes) 
  (h_total : c.total = 60)
  (h_blueberries : c.blueberries ≥ c.total / 3)
  (h_sprinkles : c.sprinkles ≥ c.total / 4)
  (h_frosting : c.frosting ≥ c.total / 2)
  (h_pecans : c.pecans ≥ c.total / 5) :
  has_no_ingredients c ≤ 0 :=
sorry

end max_plain_cupcakes_l2431_243129


namespace negation_equivalence_l2431_243127

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end negation_equivalence_l2431_243127


namespace train_average_speed_l2431_243157

/-- Given a train that travels two segments with known distances and times, 
    calculate its average speed. -/
theorem train_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 325) (h2 : time1 = 3.5) 
  (h3 : distance2 = 470) (h4 : time2 = 4) : 
  (distance1 + distance2) / (time1 + time2) = 106 := by
  sorry

#eval (325 + 470) / (3.5 + 4)

end train_average_speed_l2431_243157
