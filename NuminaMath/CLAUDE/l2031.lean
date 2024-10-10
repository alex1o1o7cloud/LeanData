import Mathlib

namespace initial_distance_of_specific_program_l2031_203141

/-- Represents a running program with weekly increments -/
structure RunningProgram where
  initial_distance : ℕ  -- Initial daily running distance
  weeks : ℕ             -- Number of weeks in the program
  increment : ℕ         -- Weekly increment in daily distance

/-- Calculates the final daily running distance after the program -/
def final_distance (program : RunningProgram) : ℕ :=
  program.initial_distance + (program.weeks - 1) * program.increment

/-- Theorem stating the initial distance given the conditions -/
theorem initial_distance_of_specific_program :
  ∃ (program : RunningProgram),
    program.weeks = 5 ∧
    program.increment = 1 ∧
    final_distance program = 7 ∧
    program.initial_distance = 3 := by
  sorry

end initial_distance_of_specific_program_l2031_203141


namespace max_annual_average_profit_l2031_203144

/-- The annual average profit function -/
def f (n : ℕ+) : ℚ :=
  (110 * n - (n^2 + n) - 90) / n

/-- Theorem stating that f(n) reaches its maximum when n = 5 -/
theorem max_annual_average_profit :
  ∀ k : ℕ+, f 5 ≥ f k :=
sorry

end max_annual_average_profit_l2031_203144


namespace bobs_family_children_l2031_203115

/-- Given the following conditions about Bob's family and apple consumption:
  * Bob picked 450 apples in total
  * There are 40 adults in the family
  * Each adult ate 3 apples
  * Each child ate 10 apples
  This theorem proves that there are 33 children in Bob's family. -/
theorem bobs_family_children (total_apples : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ) (apples_per_child : ℕ) :
  total_apples = 450 →
  num_adults = 40 →
  apples_per_adult = 3 →
  apples_per_child = 10 →
  (total_apples - num_adults * apples_per_adult) / apples_per_child = 33 :=
by sorry

end bobs_family_children_l2031_203115


namespace teacher_age_l2031_203137

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 21 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 42 :=
by sorry

end teacher_age_l2031_203137


namespace range_of_special_set_l2031_203148

def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

theorem range_of_special_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : (a + b + c) / 3 = 6)
  (h_median : b = 6)
  (h_min : a = 2) :
  c - a = 8 := by
sorry

end range_of_special_set_l2031_203148


namespace semicircle_perimeter_approx_l2031_203171

/-- The perimeter of a semi-circle with radius 6.83 cm is approximately 35.12 cm. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 6.83
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 35.12) < ε :=
by sorry

end semicircle_perimeter_approx_l2031_203171


namespace quadratic_equality_implies_coefficient_l2031_203113

theorem quadratic_equality_implies_coefficient (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end quadratic_equality_implies_coefficient_l2031_203113


namespace extreme_values_and_monotonicity_l2031_203151

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_monotonicity (a b : ℝ) :
  (f' a b (-1) = 0 ∧ f' a b 2 = 0) →
  (a = -3/2 ∧ b = -6) ∧
  (∀ x, x ∈ Set.Ioo (-1) 2 → (f' (-3/2) (-6) x < 0)) ∧
  (∀ x, (x < -1 ∨ x > 2) → (f' (-3/2) (-6) x > 0)) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-2) 3 → f (-3/2) (-6) x < m) ↔ m > 7/2) :=
by sorry


end extreme_values_and_monotonicity_l2031_203151


namespace sin_cos_pi_12_star_l2031_203194

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - a*b - b^2

-- State the theorem
theorem sin_cos_pi_12_star : 
  star (Real.sin (π/12)) (Real.cos (π/12)) = -(1 + 2*Real.sqrt 3) / 4 := by
  sorry

end sin_cos_pi_12_star_l2031_203194


namespace withdrawn_players_matches_l2031_203108

/-- Represents a table tennis tournament -/
structure TableTennisTournament where
  n : ℕ  -- Total number of players
  r : ℕ  -- Number of matches played among the 3 withdrawn players

/-- The number of matches played by remaining players -/
def remainingMatches (t : TableTennisTournament) : ℕ :=
  (t.n - 3) * (t.n - 4) / 2

/-- The total number of matches played in the tournament -/
def totalMatches (t : TableTennisTournament) : ℕ :=
  remainingMatches t + (3 * 2 - t.r)

/-- Theorem stating the number of matches played among withdrawn players -/
theorem withdrawn_players_matches (t : TableTennisTournament) : 
  t.n > 3 ∧ totalMatches t = 50 → t.r = 1 := by sorry

end withdrawn_players_matches_l2031_203108


namespace workshop_workers_l2031_203189

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- Represents the number of technicians -/
def technicians : ℕ := 5

/-- Represents the average salary of all workers -/
def avg_salary_all : ℚ := 700

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℚ := 800

/-- Represents the average salary of the rest of the workers -/
def avg_salary_rest : ℚ := 650

theorem workshop_workers :
  (avg_salary_all * total_workers : ℚ) = 
  (avg_salary_technicians * technicians : ℚ) + 
  (avg_salary_rest * (total_workers - technicians) : ℚ) := by
  sorry

#check workshop_workers

end workshop_workers_l2031_203189


namespace geometric_sequence_ratio_l2031_203112

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) →
  (a 1 + (2 * a 2 - a 1) / 2 = a 3 / 2) →
  (a 10 + a 11) / (a 8 + a 9) = 3 + 2 * Real.sqrt 2 :=
by sorry

end geometric_sequence_ratio_l2031_203112


namespace limit_x_plus_sin_x_power_sin_x_plus_x_l2031_203105

/-- The limit of (x + sin x)^(sin x + x) as x approaches π is π^π. -/
theorem limit_x_plus_sin_x_power_sin_x_plus_x (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - π| ∧ |x - π| < δ →
    |(x + Real.sin x)^(Real.sin x + x) - π^π| < ε :=
sorry

end limit_x_plus_sin_x_power_sin_x_plus_x_l2031_203105


namespace inequality_proof_l2031_203119

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^4 ≤ x^2 + y^3) : x^3 + y^3 ≤ 2 := by
  sorry

end inequality_proof_l2031_203119


namespace sarahs_weeds_total_l2031_203153

theorem sarahs_weeds_total (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (friday : ℕ) : 
  tuesday = 25 →
  wednesday = 3 * tuesday →
  thursday = wednesday / 5 →
  friday = thursday - 10 →
  tuesday + wednesday + thursday + friday = 120 :=
by sorry

end sarahs_weeds_total_l2031_203153


namespace total_birds_in_pet_store_l2031_203106

/-- Represents the number of birds in a cage -/
structure CageBirds where
  parrots : Nat
  finches : Nat
  canaries : Nat
  parakeets : Nat

/-- The pet store's bird inventory -/
def petStore : List CageBirds := [
  { parrots := 9, finches := 4, canaries := 7, parakeets := 0 },
  { parrots := 5, finches := 10, canaries := 0, parakeets := 8 },
  { parrots := 0, finches := 7, canaries := 3, parakeets := 15 },
  { parrots := 10, finches := 12, canaries := 0, parakeets := 5 }
]

/-- Calculates the total number of birds in a cage -/
def totalBirdsInCage (cage : CageBirds) : Nat :=
  cage.parrots + cage.finches + cage.canaries + cage.parakeets

/-- Theorem: The total number of birds in the pet store is 95 -/
theorem total_birds_in_pet_store :
  (petStore.map totalBirdsInCage).sum = 95 := by
  sorry

end total_birds_in_pet_store_l2031_203106


namespace remainder_divisibility_l2031_203125

theorem remainder_divisibility (N : ℕ) (h : N % 125 = 40) : N % 15 = 10 := by
  sorry

end remainder_divisibility_l2031_203125


namespace two_inequalities_true_l2031_203162

theorem two_inequalities_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
  ∃! n : ℕ, n = 2 ∧ 
    (n = (if x^2 + y^2 < a^2 + b^2 then 1 else 0) +
         (if x^2 - y^2 < a^2 - b^2 then 1 else 0) +
         (if x^2 * y^2 < a^2 * b^2 then 1 else 0) +
         (if x^2 / y^2 < a^2 / b^2 then 1 else 0)) :=
by sorry

end two_inequalities_true_l2031_203162


namespace solution_set_f_geq_3_max_a_value_exists_x_for_a_eq_3_l2031_203175

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the maximum value of a
theorem max_a_value (a : ℝ) :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → a ≤ 3 := by sorry

-- Theorem that 3 is indeed the maximum value
theorem exists_x_for_a_eq_3 :
  ∃ x : ℝ, f x ≤ -3^2 + 3 + 7 := by sorry

end solution_set_f_geq_3_max_a_value_exists_x_for_a_eq_3_l2031_203175


namespace minimum_students_for_photo_l2031_203109

def photo_cost (x : ℝ) : ℝ := 5 + (x - 2) * 0.8

theorem minimum_students_for_photo : 
  ∃ x : ℝ, x ≥ 17 ∧ 
  (∀ y : ℝ, y ≥ x → photo_cost y / y ≤ 1) ∧
  (∀ z : ℝ, z < x → photo_cost z / z > 1) :=
sorry

end minimum_students_for_photo_l2031_203109


namespace probability_A_equals_B_l2031_203127

open Set
open MeasureTheory
open Real

-- Define the set of valid pairs (a, b)
def ValidPairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, b) := p
               cos (cos a) = cos (cos b) ∧
               -5*π/2 ≤ a ∧ a ≤ 5*π/2 ∧
               -5*π/2 ≤ b ∧ b ≤ 5*π/2}

-- Define the set of pairs where A = B
def EqualPairs : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (a, b) := p; a = b}

-- Define the probability measure on ValidPairs
noncomputable def ProbMeasure : Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_A_equals_B :
  ProbMeasure (ValidPairs ∩ EqualPairs) / ProbMeasure ValidPairs = 1/5 :=
sorry

end probability_A_equals_B_l2031_203127


namespace blue_then_red_probability_l2031_203181

/-- The probability of drawing a blue marble first and a red marble second -/
theorem blue_then_red_probability (red white blue : ℕ) 
  (h_red : red = 4)
  (h_white : white = 6)
  (h_blue : blue = 2) : 
  (blue : ℚ) / (red + white + blue) * red / (red + white + blue - 1) = 2 / 33 := by
sorry

end blue_then_red_probability_l2031_203181


namespace max_value_on_circle_l2031_203170

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 10 →
  4*x + 3*y ≤ 32 :=
by sorry

end max_value_on_circle_l2031_203170


namespace line_perpendicular_theorem_l2031_203133

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two planes -/
def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_perpendicular_theorem
  (m n : Line3D) (α β : Plane3D)
  (h1 : ¬ parallel_planes α β)
  (h2 : perpendicular m α)
  (h3 : ¬ parallel n β) :
  perpendicular_lines m n :=
sorry

end line_perpendicular_theorem_l2031_203133


namespace avery_building_time_l2031_203169

theorem avery_building_time (tom_time : ℝ) (joint_work_time : ℝ) (tom_remaining_time : ℝ) 
  (h1 : tom_time = 2)
  (h2 : joint_work_time = 1)
  (h3 : tom_remaining_time = 20.000000000000007 / 60) :
  ∃ (avery_time : ℝ), 
    1 / avery_time + 1 / tom_time + (tom_remaining_time / tom_time) = 1 ∧ 
    avery_time = 3 := by
sorry

end avery_building_time_l2031_203169


namespace sum_of_reciprocal_squares_l2031_203160

theorem sum_of_reciprocal_squares (a b c : ℝ) : 
  a^3 - 12*a^2 + 14*a + 3 = 0 →
  b^3 - 12*b^2 + 14*b + 3 = 0 →
  c^3 - 12*c^2 + 14*c + 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 268/9 := by
sorry

end sum_of_reciprocal_squares_l2031_203160


namespace root_existence_and_bounds_l2031_203124

theorem root_existence_and_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x₁ x₂ : ℝ),
    (1 / x₁ + 1 / (x₁ - a) + 1 / (x₁ + b) = 0) ∧
    (1 / x₂ + 1 / (x₂ - a) + 1 / (x₂ + b) = 0) ∧
    (a / 3 ≤ x₁ ∧ x₁ ≤ 2 * a / 3) ∧
    (-2 * b / 3 ≤ x₂ ∧ x₂ ≤ -b / 3) :=
by sorry

end root_existence_and_bounds_l2031_203124


namespace inequality_proof_l2031_203159

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 := by
  sorry

end inequality_proof_l2031_203159


namespace inscribed_circles_area_limit_l2031_203157

/-- Represents the sum of areas of the first n inscribed circles -/
def S (n : ℕ) (a : ℝ) : ℝ := sorry

/-- The limit of S_n as n approaches infinity -/
def S_limit (a : ℝ) : ℝ := sorry

theorem inscribed_circles_area_limit (a b : ℝ) (h : 0 < a ∧ a ≤ b) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n a - S_limit a| < ε ∧ S_limit a = (π * a^2) / 2 := by
  sorry

end inscribed_circles_area_limit_l2031_203157


namespace boys_to_girls_ratio_l2031_203196

theorem boys_to_girls_ratio (T : ℚ) (G : ℚ) (h : (2/3) * G = (1/4) * T) : 
  (T - G) / G = 5/3 := by
sorry

end boys_to_girls_ratio_l2031_203196


namespace total_legs_farmer_brown_l2031_203126

/-- The number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- The number of legs for a sheep -/
def sheep_legs : ℕ := 4

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := 5

/-- Theorem stating the total number of legs among the animals Farmer Brown fed -/
theorem total_legs_farmer_brown : 
  num_chickens * chicken_legs + num_sheep * sheep_legs = 34 := by
  sorry

end total_legs_farmer_brown_l2031_203126


namespace mirror_area_l2031_203174

/-- The area of a rectangular mirror inside a frame with given external dimensions and frame width -/
theorem mirror_area (frame_height frame_width frame_side_width : ℝ) :
  frame_height = 100 ∧ 
  frame_width = 140 ∧ 
  frame_side_width = 15 →
  (frame_height - 2 * frame_side_width) * (frame_width - 2 * frame_side_width) = 7700 := by
  sorry

end mirror_area_l2031_203174


namespace painted_cube_probability_l2031_203145

/-- Represents a cube with painted faces --/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ

/-- Calculates the number of unit cubes with exactly three painted faces --/
def num_three_painted_faces (cube : PaintedCube) : ℕ :=
  if cube.painted_faces = 2 then 4 else 0

/-- Calculates the number of unit cubes with no painted faces --/
def num_no_painted_faces (cube : PaintedCube) : ℕ :=
  (cube.size - 2) ^ 3

/-- Calculates the total number of unit cubes --/
def total_unit_cubes (cube : PaintedCube) : ℕ :=
  cube.size ^ 3

/-- Calculates the number of ways to choose 2 cubes from the total --/
def choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: The probability of selecting one unit cube with three painted faces
    and one with no painted faces is 9/646 for a 5x5x5 cube with two adjacent
    painted faces --/
theorem painted_cube_probability (cube : PaintedCube)
    (h1 : cube.size = 5)
    (h2 : cube.painted_faces = 2) :
    (num_three_painted_faces cube * num_no_painted_faces cube : ℚ) /
    choose_two (total_unit_cubes cube) = 9 / 646 := by
  sorry

end painted_cube_probability_l2031_203145


namespace bracelet_sale_earnings_l2031_203180

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  total_bracelets : ℕ
  single_price : ℕ
  pair_price : ℕ
  single_sales : ℕ

/-- Calculates the total earnings from selling bracelets -/
def total_earnings (sale : BraceletSale) : ℕ :=
  let remaining_bracelets := sale.total_bracelets - (sale.single_sales / sale.single_price)
  let pair_sales := remaining_bracelets / 2
  (sale.single_sales / sale.single_price) * sale.single_price + pair_sales * sale.pair_price

/-- Theorem stating that the total earnings from the given scenario is $132 -/
theorem bracelet_sale_earnings :
  let sale : BraceletSale := {
    total_bracelets := 30,
    single_price := 5,
    pair_price := 8,
    single_sales := 60
  }
  total_earnings sale = 132 := by sorry

end bracelet_sale_earnings_l2031_203180


namespace square_of_negative_sqrt_five_l2031_203167

theorem square_of_negative_sqrt_five : (-Real.sqrt 5)^2 = 5 := by
  sorry

end square_of_negative_sqrt_five_l2031_203167


namespace employee_hire_year_l2031_203107

/-- Rule of 70 provision: An employee can retire when their age plus years of employment is at least 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1968

/-- The age at which the employee was hired -/
def hire_age : ℕ := 32

/-- The year the employee becomes eligible to retire -/
def retirement_year : ℕ := 2006

theorem employee_hire_year :
  rule_of_70 (hire_age + (retirement_year - hire_year)) hire_age ∧
  ∀ y, y > hire_year → ¬rule_of_70 (hire_age + (y - hire_year)) hire_age :=
by sorry

end employee_hire_year_l2031_203107


namespace ellipse_k_range_l2031_203120

/-- The equation of an ellipse in terms of parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧ 
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ Set.Ioo (-3 : ℝ) (-1/2) ∪ Set.Ioo (-1/2 : ℝ) 2 :=
sorry

end ellipse_k_range_l2031_203120


namespace claps_per_second_is_seventeen_l2031_203149

/-- The number of claps achieved in one minute -/
def claps_per_minute : ℕ := 1020

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of claps per second -/
def claps_per_second : ℚ := claps_per_minute / seconds_per_minute

theorem claps_per_second_is_seventeen : 
  claps_per_second = 17 := by sorry

end claps_per_second_is_seventeen_l2031_203149


namespace hexagon_ratio_l2031_203188

/-- A hexagon with specific properties -/
structure Hexagon where
  area : ℝ
  below_rs_area : ℝ
  triangle_base : ℝ
  xr : ℝ
  rs : ℝ

/-- The theorem statement -/
theorem hexagon_ratio (h : Hexagon) (h_area : h.area = 13)
  (h_bisect : h.below_rs_area = h.area / 2)
  (h_below : h.below_rs_area = 2 + (h.triangle_base * (h.below_rs_area - 2) / h.triangle_base) / 2)
  (h_base : h.triangle_base = 4)
  (h_sum : h.xr + h.rs = h.triangle_base) :
  h.xr / h.rs = 1 := by sorry

end hexagon_ratio_l2031_203188


namespace inequality_system_solution_l2031_203192

theorem inequality_system_solution :
  {x : ℝ | 3*x - 1 ≥ x + 1 ∧ x + 4 > 4*x - 2} = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end inequality_system_solution_l2031_203192


namespace min_abs_sum_squared_matrix_l2031_203117

-- Define the matrix type
def Matrix2x2 (α : Type) := Fin 2 → Fin 2 → α

-- Define the matrix multiplication
def matMul (A B : Matrix2x2 ℤ) : Matrix2x2 ℤ :=
  λ i j => (Finset.univ.sum λ k => A i k * B k j)

-- Define the identity matrix
def identityMatrix : Matrix2x2 ℤ :=
  λ i j => if i = j then 9 else 0

-- Define the absolute value sum
def absSum (a b c d : ℤ) : ℤ :=
  |a| + |b| + |c| + |d|

theorem min_abs_sum_squared_matrix :
  ∃ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (matMul (λ i j => match i, j with
      | 0, 0 => a
      | 0, 1 => b
      | 1, 0 => c
      | 1, 1 => d) (λ i j => match i, j with
      | 0, 0 => a
      | 0, 1 => b
      | 1, 0 => c
      | 1, 1 => d)) = identityMatrix ∧
    (∀ (a' b' c' d' : ℤ),
      a' ≠ 0 → b' ≠ 0 → c' ≠ 0 → d' ≠ 0 →
      (matMul (λ i j => match i, j with
        | 0, 0 => a'
        | 0, 1 => b'
        | 1, 0 => c'
        | 1, 1 => d') (λ i j => match i, j with
        | 0, 0 => a'
        | 0, 1 => b'
        | 1, 0 => c'
        | 1, 1 => d')) = identityMatrix →
      absSum a b c d ≤ absSum a' b' c' d') ∧
    absSum a b c d = 8 :=
by
  sorry


end min_abs_sum_squared_matrix_l2031_203117


namespace model_x_completion_time_l2031_203173

/-- The time (in minutes) it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 30

/-- The number of Model X computers used -/
def num_model_x : ℕ := 20

/-- The time (in minutes) it takes to complete the task when using equal numbers of both models -/
def combined_time : ℝ := 1

/-- The time (in minutes) it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := 60

theorem model_x_completion_time :
  (num_model_x : ℝ) * (1 / model_x_time + 1 / model_y_time) = 1 / combined_time :=
sorry

end model_x_completion_time_l2031_203173


namespace expression_equality_l2031_203184

theorem expression_equality : (2^2 / 3) + (-3^2 + 5) + (-3)^2 * (2/3)^2 = 4/3 := by
  sorry

end expression_equality_l2031_203184


namespace martha_clothes_count_l2031_203187

/-- Calculates the total number of clothes Martha takes home from a shopping trip -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  jackets_bought + free_jackets + tshirts_bought + free_tshirts

/-- Proves that Martha takes home 18 clothes given the conditions of the problem -/
theorem martha_clothes_count :
  total_clothes 4 9 = 18 := by
  sorry

end martha_clothes_count_l2031_203187


namespace second_order_implies_first_order_l2031_203154

/-- A function f: ℝ → ℝ is increasing on an interval D if for any x, y ∈ D, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x y, x ∈ D → y ∈ D → x < y → f x < f y

/-- x₀ is a second-order fixed point of f if f(f(x₀)) = x₀ -/
def SecondOrderFixedPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (f x₀) = x₀

/-- x₀ is a first-order fixed point of f if f(x₀) = x₀ -/
def FirstOrderFixedPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = x₀

theorem second_order_implies_first_order
    (f : ℝ → ℝ) (D : Set ℝ) (x₀ : ℝ)
    (h_inc : IncreasingOn f D)
    (h_x₀ : x₀ ∈ D)
    (h_second : SecondOrderFixedPoint f x₀) :
    FirstOrderFixedPoint f x₀ := by
  sorry

end second_order_implies_first_order_l2031_203154


namespace complex_number_equation_l2031_203190

/-- Given a complex number z = 1 + √2i, prove that z^2 - 2z = -3 -/
theorem complex_number_equation : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 2
  z^2 - 2*z = -3 := by sorry

end complex_number_equation_l2031_203190


namespace julia_played_with_17_kids_on_monday_l2031_203161

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := sorry

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := 34

/-- Theorem stating that Julia played with 17 kids on Monday -/
theorem julia_played_with_17_kids_on_monday :
  monday_kids = 17 :=
by
  sorry

end julia_played_with_17_kids_on_monday_l2031_203161


namespace smallest_five_digit_divisible_by_53_l2031_203193

theorem smallest_five_digit_divisible_by_53 : ∀ n : ℕ, 
  10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 → n ≥ 10017 :=
by sorry

end smallest_five_digit_divisible_by_53_l2031_203193


namespace max_value_log_sum_l2031_203100

theorem max_value_log_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a * b = 1000) :
  Real.sqrt (1 + Real.log a / Real.log 10) + Real.sqrt (1 + Real.log b / Real.log 10) ≤ Real.sqrt 10 := by
  sorry

end max_value_log_sum_l2031_203100


namespace smallest_n_for_factorization_l2031_203158

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, (∃ A B : ℤ, ∀ x, 5*x^2 + n*x + 50 = (5*x + A)*(x + B)) → n ≥ 35 :=
by sorry

end smallest_n_for_factorization_l2031_203158


namespace zero_subset_M_l2031_203110

def M : Set ℤ := {x : ℤ | |x| < 5}

theorem zero_subset_M : {0} ⊆ M := by
  sorry

end zero_subset_M_l2031_203110


namespace floor_abs_negative_real_l2031_203152

theorem floor_abs_negative_real : ⌊|(-54.7 : ℝ)|⌋ = 54 := by sorry

end floor_abs_negative_real_l2031_203152


namespace quadratic_equation_solution_l2031_203198

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - x - 5
  let x₁ : ℝ := (1 + Real.sqrt 21) / 2
  let x₂ : ℝ := (1 - Real.sqrt 21) / 2
  f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end quadratic_equation_solution_l2031_203198


namespace inequality_solution_l2031_203134

-- Define the inequality function
def f (x : ℝ) : ℝ := (x^2 - 4) * (x - 6)^2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

-- Theorem stating that the solution set is correct
theorem inequality_solution : 
  {x : ℝ | f x ≤ 0} = solution_set := by sorry

end inequality_solution_l2031_203134


namespace min_value_theorem_l2031_203103

/-- Given a function y = a^x + b where b > 0, a > 1, and 3 = a + b, 
    the minimum value of (4 / (a - 1)) + (1 / b) is 9/2 -/
theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a > 1) (h3 : 3 = a + b) :
  (∀ x : ℝ, (4 / (a - 1)) + (1 / b) ≥ 9/2) ∧ 
  (∃ x : ℝ, (4 / (a - 1)) + (1 / b) = 9/2) :=
by sorry

end min_value_theorem_l2031_203103


namespace f_property_l2031_203150

noncomputable def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

theorem f_property (x : ℝ) :
  f x + f (1 - x) = 1 ∧
  (2 * (f x)^2 < f (1 - x) ↔ x < 1/2) :=
by sorry

end f_property_l2031_203150


namespace initial_puppies_count_l2031_203116

/-- The number of puppies Sandy's dog initially had -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Sandy gave away -/
def puppies_given_away : ℕ := 4

/-- The number of puppies Sandy has left -/
def puppies_left : ℕ := 4

/-- Theorem stating that the initial number of puppies is 8 -/
theorem initial_puppies_count : initial_puppies = 8 := by sorry

end initial_puppies_count_l2031_203116


namespace tan_triple_angle_l2031_203114

theorem tan_triple_angle (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l2031_203114


namespace non_constant_polynomial_not_always_palindrome_l2031_203123

/-- A number is a palindrome if it reads the same from left to right as it reads from right to left in base 10 -/
def is_palindrome (n : ℤ) : Prop := sorry

/-- The theorem states that for any non-constant polynomial with integer coefficients, 
    there exists a positive integer n such that p(n) is not a palindrome number -/
theorem non_constant_polynomial_not_always_palindrome 
  (p : Polynomial ℤ) (h : ¬ (p.degree = 0)) : 
  ∃ (n : ℕ), ¬ is_palindrome (p.eval n) := by sorry

end non_constant_polynomial_not_always_palindrome_l2031_203123


namespace polygon_sides_from_angle_sum_l2031_203147

/-- The number of sides of a polygon given the sum of its interior angles -/
theorem polygon_sides_from_angle_sum (angle_sum : ℝ) : angle_sum = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = angle_sum := by
  sorry

end polygon_sides_from_angle_sum_l2031_203147


namespace chord_of_ellipse_l2031_203143

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 2*y^2 - 4 = 0

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The midpoint of a line segment -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem chord_of_ellipse :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  ellipse_equation x₁ y₁ ∧ 
  ellipse_equation x₂ y₂ ∧
  (∀ x y : ℝ, line_equation x y ↔ is_midpoint 1 1 x₁ y₁ x₂ y₂) →
  line_equation x₁ y₁ ∧ line_equation x₂ y₂ := by sorry

end chord_of_ellipse_l2031_203143


namespace root_ratio_sum_zero_l2031_203130

theorem root_ratio_sum_zero (a b n p : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) 
  (h_roots : ∃ (x y : ℝ), x / y = a / b ∧ p * x^2 + n * x + n = 0 ∧ p * y^2 + n * y + n = 0) :
  Real.sqrt (a / b) + Real.sqrt (b / a) + Real.sqrt (n / p) = 0 := by
  sorry

end root_ratio_sum_zero_l2031_203130


namespace project_completion_time_l2031_203182

theorem project_completion_time 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (work_days_A : ℝ) 
  (remaining_days_B : ℝ) 
  (h1 : days_A = 10) 
  (h2 : days_B = 15) 
  (h3 : work_days_A = 3) : 
  work_days_A / days_A + remaining_days_B / days_B = 1 :=
by
  sorry

end project_completion_time_l2031_203182


namespace pitchers_prepared_is_six_l2031_203132

/-- Represents the number of glasses of lemonade a single pitcher can serve. -/
def glasses_per_pitcher : ℕ := 5

/-- Represents the total number of glasses of lemonade served. -/
def total_glasses_served : ℕ := 30

/-- Calculates the number of pitchers needed to serve the given number of glasses. -/
def pitchers_needed (total_glasses : ℕ) (glasses_per_pitcher : ℕ) : ℕ :=
  total_glasses / glasses_per_pitcher

/-- Proves that the number of pitchers prepared is 6. -/
theorem pitchers_prepared_is_six :
  pitchers_needed total_glasses_served glasses_per_pitcher = 6 := by
  sorry

end pitchers_prepared_is_six_l2031_203132


namespace polynomial_equality_l2031_203165

theorem polynomial_equality (s t : ℝ) : -1/4 * s * t + 0.25 * s * t = 0 := by
  sorry

end polynomial_equality_l2031_203165


namespace suv_max_distance_l2031_203140

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway : Float
  city : Float

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def maxDistance (efficiency : SUVFuelEfficiency) (fuel : Float) : Float :=
  efficiency.highway * fuel

/-- Theorem stating the maximum distance an SUV can travel with given efficiency and fuel -/
theorem suv_max_distance (efficiency : SUVFuelEfficiency) (fuel : Float) :
  efficiency.highway = 12.2 →
  efficiency.city = 7.6 →
  fuel = 24 →
  maxDistance efficiency fuel = 292.8 := by
  sorry

end suv_max_distance_l2031_203140


namespace equality_comparison_l2031_203176

theorem equality_comparison : 
  (-2^2 ≠ (-2)^2) ∧ 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  ((-3 * 2)^2 ≠ -3^2 * 2^2) := by
  sorry

end equality_comparison_l2031_203176


namespace expected_hits_greater_than_half_l2031_203199

/-- The expected number of hit targets is always greater than or equal to half the number of boys/targets. -/
theorem expected_hits_greater_than_half (n : ℕ) (hn : n > 0) :
  n * (1 - (1 - 1 / n)^n) ≥ n / 2 := by
  sorry

#check expected_hits_greater_than_half

end expected_hits_greater_than_half_l2031_203199


namespace max_value_of_linear_combination_l2031_203179

theorem max_value_of_linear_combination (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  x + 2 * y ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), 4 * x₀ + 3 * y₀ ≤ 10 ∧ 3 * x₀ + 6 * y₀ ≤ 12 ∧ x₀ + 2 * y₀ = 4 :=
sorry

end max_value_of_linear_combination_l2031_203179


namespace clothes_fraction_is_one_eighth_l2031_203185

/-- The fraction of Gina's initial money used to buy clothes -/
def fraction_for_clothes (initial_amount : ℚ) 
  (fraction_to_mom : ℚ) (fraction_to_charity : ℚ) (amount_kept : ℚ) : ℚ :=
  let amount_to_mom := initial_amount * fraction_to_mom
  let amount_to_charity := initial_amount * fraction_to_charity
  let amount_for_clothes := initial_amount - amount_to_mom - amount_to_charity - amount_kept
  amount_for_clothes / initial_amount

theorem clothes_fraction_is_one_eighth :
  fraction_for_clothes 400 (1/4) (1/5) 170 = 1/8 := by
  sorry


end clothes_fraction_is_one_eighth_l2031_203185


namespace binomial_10_2_l2031_203101

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

/-- Theorem: The binomial coefficient (10 choose 2) equals 45 -/
theorem binomial_10_2 : binomial 10 2 = 45 := by sorry

end binomial_10_2_l2031_203101


namespace smallest_of_five_consecutive_integers_sum_2025_l2031_203197

theorem smallest_of_five_consecutive_integers_sum_2025 (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 2025) → n = 403 := by
  sorry

end smallest_of_five_consecutive_integers_sum_2025_l2031_203197


namespace pictures_per_album_l2031_203183

/-- Given a total of 20 pictures sorted equally into 5 albums, prove that each album contains 4 pictures. -/
theorem pictures_per_album :
  let total_pictures : ℕ := 7 + 13
  let num_albums : ℕ := 5
  let pictures_per_album : ℕ := total_pictures / num_albums
  pictures_per_album = 4 := by
  sorry

end pictures_per_album_l2031_203183


namespace negation_of_existence_inequality_l2031_203128

theorem negation_of_existence_inequality (p : Prop) :
  (¬ p ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0) ↔
  (p ↔ ∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0) :=
by sorry

end negation_of_existence_inequality_l2031_203128


namespace intersection_implies_a_value_l2031_203156

def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end intersection_implies_a_value_l2031_203156


namespace triangle_side_length_l2031_203118

open Real

-- Define the triangle
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  -- Given conditions
  c = 4 * sqrt 2 →
  B = π / 4 →  -- 45° in radians
  S = 2 →
  -- Area formula
  S = (1 / 2) * a * c * sin B →
  -- Law of Cosines
  b^2 = a^2 + c^2 - 2*a*c*(cos B) →
  -- Conclusion
  b = 5 := by
sorry


end triangle_side_length_l2031_203118


namespace lcm_20_45_75_l2031_203178

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end lcm_20_45_75_l2031_203178


namespace complement_of_M_l2031_203172

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M :
  (U \ M) = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end complement_of_M_l2031_203172


namespace dolphin_training_hours_l2031_203131

theorem dolphin_training_hours 
  (num_dolphins : ℕ) 
  (num_trainers : ℕ) 
  (hours_per_trainer : ℕ) 
  (h1 : num_dolphins = 4) 
  (h2 : num_trainers = 2) 
  (h3 : hours_per_trainer = 6) : 
  (num_trainers * hours_per_trainer) / num_dolphins = 3 := by
sorry

end dolphin_training_hours_l2031_203131


namespace fraction_equality_l2031_203146

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (4*a + 2*b) / (2*a - 4*b) = 3) : 
  (2*a + 4*b) / (4*a - 2*b) = 9/13 := by
  sorry

end fraction_equality_l2031_203146


namespace cats_total_is_seven_l2031_203122

/-- Calculates the total number of cats given the initial number of cats and the number of kittens -/
def total_cats (initial_cats female_kittens male_kittens : ℕ) : ℕ :=
  initial_cats + female_kittens + male_kittens

/-- Proves that the total number of cats is 7 given the initial conditions -/
theorem cats_total_is_seven :
  total_cats 2 3 2 = 7 := by
  sorry

end cats_total_is_seven_l2031_203122


namespace complex_magnitude_equation_l2031_203195

theorem complex_magnitude_equation (n : ℝ) : 
  (n > 0 ∧ Complex.abs (5 + n * Complex.I) = Real.sqrt 34) → n = 3 := by
  sorry

end complex_magnitude_equation_l2031_203195


namespace square_sum_nonzero_iff_one_nonzero_l2031_203164

theorem square_sum_nonzero_iff_one_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by
  sorry

end square_sum_nonzero_iff_one_nonzero_l2031_203164


namespace vector_dot_product_equals_22_l2031_203136

/-- Given two vectors AB and BC in ℝ², where BC has a magnitude of √10,
    prove that the dot product of AB and AC equals 22. -/
theorem vector_dot_product_equals_22 
  (AB : ℝ × ℝ) 
  (BC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : ∃ t > 0, BC = (3, t)) 
  (h3 : Real.sqrt ((BC.1)^2 + (BC.2)^2) = Real.sqrt 10) : 
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  (AB.1 * AC.1 + AB.2 * AC.2) = 22 := by
  sorry

end vector_dot_product_equals_22_l2031_203136


namespace problem_1_problem_2_l2031_203135

theorem problem_1 : 2023^2 - 2024 * 2022 = 1 := by sorry

theorem problem_2 (a b c : ℝ) : 5 * a^2 * b^3 * (-1/10 * a * b^3 * c) / (1/2 * a * b^2)^3 = -4 * c := by sorry

end problem_1_problem_2_l2031_203135


namespace animal_farm_count_l2031_203177

theorem animal_farm_count (total_legs : ℕ) (chicken_count : ℕ) : 
  total_legs = 26 →
  chicken_count = 5 →
  ∃ (buffalo_count : ℕ),
    2 * chicken_count + 4 * buffalo_count = total_legs ∧
    chicken_count + buffalo_count = 9 :=
by sorry

end animal_farm_count_l2031_203177


namespace circle_equation_with_diameter_PQ_l2031_203139

def P : ℝ × ℝ := (4, 0)
def Q : ℝ × ℝ := (0, 2)

theorem circle_equation_with_diameter_PQ :
  let center := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let radius_squared := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_squared ↔
    (x - 2)^2 + (y - 1)^2 = 5 := by sorry

end circle_equation_with_diameter_PQ_l2031_203139


namespace x_value_in_equation_l2031_203129

theorem x_value_in_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 * x^2 + 24 * x * y = 2 * x^3 + 3 * x^2 * y^2) : x = 8 := by
  sorry

end x_value_in_equation_l2031_203129


namespace original_mean_l2031_203121

theorem original_mean (n : ℕ) (decrement : ℝ) (updated_mean : ℝ) (h1 : n = 50) (h2 : decrement = 34) (h3 : updated_mean = 166) : 
  (n : ℝ) * updated_mean + n * decrement = n * 200 := by
  sorry

end original_mean_l2031_203121


namespace least_three_digit_with_product_24_l2031_203155

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_product_24 :
  (is_three_digit 234) ∧
  (digit_product 234 = 24) ∧
  (∀ m : ℕ, is_three_digit m → digit_product m = 24 → 234 ≤ m) :=
sorry

end least_three_digit_with_product_24_l2031_203155


namespace equation_transformation_l2031_203111

theorem equation_transformation :
  ∀ x : ℝ, x^2 - 6*x = 0 ↔ (x - 3)^2 = 9 := by sorry

end equation_transformation_l2031_203111


namespace omega_range_l2031_203186

theorem omega_range (ω : ℝ) (a b : ℝ) (h_pos : ω > 0) 
  (h_ab : π ≤ a ∧ a < b ∧ b ≤ 2*π) 
  (h_sin : Real.sin (ω*a) + Real.sin (ω*b) = 2) : 
  (9/4 ≤ ω ∧ ω ≤ 5/2) ∨ (13/4 ≤ ω) :=
sorry

end omega_range_l2031_203186


namespace min_total_cards_problem_l2031_203168

def min_total_cards (carlos_cards : ℕ) (matias_diff : ℕ) (ella_multiplier : ℕ) (divisor : ℕ) : ℕ :=
  let matias_cards := carlos_cards - matias_diff
  let jorge_cards := matias_cards
  let ella_cards := ella_multiplier * (jorge_cards + matias_cards)
  let total_cards := carlos_cards + matias_cards + jorge_cards + ella_cards
  ((total_cards + divisor - 1) / divisor) * divisor

theorem min_total_cards_problem :
  min_total_cards 20 6 2 15 = 105 := by sorry

end min_total_cards_problem_l2031_203168


namespace circle_equation_l2031_203102

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y + c = 0

/-- Checks if a point lies on a given circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem to prove -/
theorem circle_equation : ∃ (c : Circle),
  (c.center.x + c.center.y - 2 = 0) ∧
  pointOnCircle ⟨1, -1⟩ c ∧
  pointOnCircle ⟨-1, 1⟩ c ∧
  c.center = ⟨1, 1⟩ ∧
  c.radius = 2 :=
sorry

end circle_equation_l2031_203102


namespace distribute_eight_to_two_groups_l2031_203104

/-- The number of ways to distribute n distinct objects into 2 non-empty groups -/
def distribute_to_two_groups (n : ℕ) : ℕ :=
  2^n - 2

/-- The theorem stating that distributing 8 distinct objects into 2 non-empty groups results in 254 possibilities -/
theorem distribute_eight_to_two_groups :
  distribute_to_two_groups 8 = 254 := by
  sorry

#eval distribute_to_two_groups 8

end distribute_eight_to_two_groups_l2031_203104


namespace triangle_theorem_l2031_203142

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle satisfying the given condition,
    angle C is π/4 and the maximum area when c = 2 is 1 + √2. -/
theorem triangle_theorem (t : Triangle) 
    (h : t.a * Real.cos t.B + t.b * Real.cos t.A - Real.sqrt 2 * t.c * Real.cos t.C = 0) :
    t.C = π / 4 ∧ 
    (t.c = 2 → ∃ (S : ℝ), S = (1 + Real.sqrt 2) ∧ ∀ (S' : ℝ), S' ≤ S) := by
  sorry

#check triangle_theorem

end triangle_theorem_l2031_203142


namespace water_balloon_problem_l2031_203191

/-- The number of water balloons that popped on the ground --/
def popped_balloons (max_rate max_time zach_rate zach_time total_filled : ℕ) : ℕ :=
  max_rate * max_time + zach_rate * zach_time - total_filled

theorem water_balloon_problem :
  popped_balloons 2 30 3 40 170 = 10 := by
  sorry

end water_balloon_problem_l2031_203191


namespace vector_difference_l2031_203138

/-- Given two 2D vectors a and b, prove that their difference is (5, -3) -/
theorem vector_difference (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) (hb : b = (-3, 4)) : 
  a - b = (5, -3) := by
  sorry

end vector_difference_l2031_203138


namespace left_handed_fraction_l2031_203163

/-- Represents the number of participants from each world -/
structure Participants where
  red : ℚ
  blue : ℚ
  green : ℚ

/-- Calculates the total number of participants -/
def total_participants (p : Participants) : ℚ :=
  p.red + p.blue + p.green

/-- Calculates the number of left-handed participants -/
def left_handed_participants (p : Participants) : ℚ :=
  p.red / 3 + 2 * p.blue / 3

/-- The main theorem stating the fraction of left-handed participants -/
theorem left_handed_fraction (p : Participants) 
  (h1 : p.red = 3 * p.blue / 2)  -- ratio of red to blue is 3:2
  (h2 : p.blue = 5 * p.green / 4)  -- ratio of blue to green is 5:4
  : left_handed_participants p / total_participants p = 35 / 99 := by
  sorry

end left_handed_fraction_l2031_203163


namespace day_150_of_year_n_minus_2_is_thursday_l2031_203166

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Represents a day in a year -/
structure DayInYear where
  day : ℕ
  year : Year

def is_leap_year (y : Year) : Prop :=
  sorry

def day_of_week (d : DayInYear) : DayOfWeek :=
  sorry

theorem day_150_of_year_n_minus_2_is_thursday
  (N : Year)
  (h1 : day_of_week ⟨256, N⟩ = DayOfWeek.Wednesday)
  (h2 : is_leap_year ⟨N.value + 1⟩)
  (h3 : day_of_week ⟨164, ⟨N.value + 1⟩⟩ = DayOfWeek.Wednesday) :
  day_of_week ⟨150, ⟨N.value - 2⟩⟩ = DayOfWeek.Thursday :=
sorry

end day_150_of_year_n_minus_2_is_thursday_l2031_203166
