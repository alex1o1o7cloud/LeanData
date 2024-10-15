import Mathlib

namespace NUMINAMATH_CALUDE_function_satisfies_condition_l171_17104

-- Define the function y
def y (x : ℝ) : ℝ := x - 2

-- State the theorem
theorem function_satisfies_condition :
  y 1 = -1 := by sorry

end NUMINAMATH_CALUDE_function_satisfies_condition_l171_17104


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l171_17196

theorem no_solution_fractional_equation :
  ∀ y : ℝ, y ≠ 3 → (y - 2) / (y - 3) ≠ 2 - 1 / (3 - y) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l171_17196


namespace NUMINAMATH_CALUDE_binomial_cube_expansion_problem_solution_l171_17168

theorem binomial_cube_expansion (n : ℕ) : n^3 + 3*(n^2) + 3*n + 1 = (n+1)^3 := by
  sorry

theorem problem_solution : 98^3 + 3*(98^2) + 3*98 + 1 = 99^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_expansion_problem_solution_l171_17168


namespace NUMINAMATH_CALUDE_square_circle_circumradius_infinite_l171_17158

/-- The radius of the circumcircle of a square with side length 1 and an inscribed circle 
    with diameter equal to the square's diagonal is infinite. -/
theorem square_circle_circumradius_infinite :
  let square : Set (ℝ × ℝ) := {p | p.1 ∈ [0, 1] ∧ p.2 ∈ [0, 1]}
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 0.5)^2 + p.2^2 ≤ 0.5^2}
  let figure : Set (ℝ × ℝ) := square ∪ circle
  ¬ ∃ (r : ℝ), r > 0 ∧ ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ figure → (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by sorry


end NUMINAMATH_CALUDE_square_circle_circumradius_infinite_l171_17158


namespace NUMINAMATH_CALUDE_factors_of_1320_l171_17112

theorem factors_of_1320 : Nat.card (Nat.divisors 1320) = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l171_17112


namespace NUMINAMATH_CALUDE_tonis_dimes_l171_17137

/-- Represents the savings of three kids in cents -/
structure Savings where
  teagan : ℕ  -- Teagan's savings in pennies
  rex : ℕ     -- Rex's savings in nickels
  toni : ℕ    -- Toni's savings in dimes
  total : ℕ   -- Total savings in cents

/-- Theorem stating that given the conditions, Toni saved 330 dimes -/
theorem tonis_dimes (s : Savings) : 
  s.teagan = 200 ∧ s.rex = 100 ∧ s.total = 4000 → s.toni = 330 := by
  sorry

#check tonis_dimes

end NUMINAMATH_CALUDE_tonis_dimes_l171_17137


namespace NUMINAMATH_CALUDE_problem_solution_l171_17160

theorem problem_solution (x y : ℝ) 
  (h1 : x = 103) 
  (h2 : x^3*y - 4*x^2*y + 4*x*y = 515400) : 
  y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l171_17160


namespace NUMINAMATH_CALUDE_number_calculation_l171_17183

theorem number_calculation (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 30) : 
  (40/100) * N = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l171_17183


namespace NUMINAMATH_CALUDE_angle_property_l171_17152

theorem angle_property (θ : Real) (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ * Real.cos θ = -1/8) : 
  Real.sin θ - Real.cos θ = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_property_l171_17152


namespace NUMINAMATH_CALUDE_rachel_coloring_books_l171_17105

/-- The number of pictures Rachel still has to color -/
def remaining_pictures (book1_pictures book2_pictures colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem: Rachel has 11 pictures left to color -/
theorem rachel_coloring_books :
  remaining_pictures 23 32 44 = 11 := by
  sorry

end NUMINAMATH_CALUDE_rachel_coloring_books_l171_17105


namespace NUMINAMATH_CALUDE_points_are_collinear_l171_17119

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points are collinear if the slope between any two pairs is equal -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

theorem points_are_collinear : 
  let p1 : Point := ⟨3, 1⟩
  let p2 : Point := ⟨6, 6.4⟩
  let p3 : Point := ⟨8, 10⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_points_are_collinear_l171_17119


namespace NUMINAMATH_CALUDE_distribute_seven_to_twelve_l171_17116

/-- The number of ways to distribute distinct items to recipients -/
def distribute_ways (n_items : ℕ) (n_recipients : ℕ) : ℕ :=
  n_recipients ^ n_items

/-- Theorem: The number of ways to distribute 7 distinct items to 12 recipients,
    where each recipient can receive multiple items, is equal to 12^7 -/
theorem distribute_seven_to_twelve :
  distribute_ways 7 12 = 35831808 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_to_twelve_l171_17116


namespace NUMINAMATH_CALUDE_find_number_l171_17132

theorem find_number : ∃ x : ℝ, 2.12 + 0.345 + x = 2.4690000000000003 ∧ x = 0.0040000000000003 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l171_17132


namespace NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l171_17117

-- Define the variables
def before_dinner : ℕ := 22
def total : ℕ := 37

-- Define the theorem
theorem carrot_sticks_after_dinner :
  total - before_dinner = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l171_17117


namespace NUMINAMATH_CALUDE_tom_total_distance_l171_17154

/-- Calculates the total distance covered by Tom given his swimming and running times and speeds. -/
theorem tom_total_distance (swim_time swim_speed : ℝ) (h1 : swim_time = 2) (h2 : swim_speed = 2)
  (h3 : swim_time > 0) (h4 : swim_speed > 0) : 
  let run_time := swim_time / 2
  let run_speed := 4 * swim_speed
  swim_time * swim_speed + run_time * run_speed = 12 := by
  sorry

#check tom_total_distance

end NUMINAMATH_CALUDE_tom_total_distance_l171_17154


namespace NUMINAMATH_CALUDE_inverse_37_mod_53_l171_17162

theorem inverse_37_mod_53 : ∃ x : ℤ, 37 * x ≡ 1 [ZMOD 53] :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_inverse_37_mod_53_l171_17162


namespace NUMINAMATH_CALUDE_coopers_fence_bricks_l171_17172

/-- Calculates the number of bricks needed for a fence with given dimensions. -/
def bricks_needed (num_walls length height depth : ℕ) : ℕ :=
  num_walls * length * height * depth

/-- Theorem stating the number of bricks needed for Cooper's fence. -/
theorem coopers_fence_bricks : 
  bricks_needed 4 20 5 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_coopers_fence_bricks_l171_17172


namespace NUMINAMATH_CALUDE_cloth_selling_price_l171_17177

/-- Calculates the total selling price of cloth given the quantity, loss per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (loss_per_meter : ℚ) (cost_price_per_meter : ℚ) : ℚ :=
  quantity * (cost_price_per_meter - loss_per_meter)

/-- Proves that the total selling price for 400 meters of cloth is $18,000 given the specified conditions. -/
theorem cloth_selling_price :
  total_selling_price 400 5 50 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l171_17177


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l171_17178

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h1 : pens = 1204) (h2 : pencils = 840) :
  Nat.gcd pens pencils = 28 := by
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l171_17178


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_l171_17169

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^3 = x^3 + y^3

-- Define what it means for a point to be on a line
def on_line (x y a b c : ℝ) : Prop := a*x + b*y + c = 0

-- Define the three lines we expect
def line1 (x y : ℝ) : Prop := on_line x y 1 1 0  -- x + y = 0
def line2 (x y : ℝ) : Prop := on_line x y 1 0 0  -- x = 0
def line3 (x y : ℝ) : Prop := on_line x y 0 1 0  -- y = 0

-- Theorem statement
theorem equation_represents_three_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_l171_17169


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l171_17166

-- Define the sets A and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem intersection_nonempty_implies_a_greater_than_one (a : ℝ) :
  (A ∩ C a).Nonempty → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_one_l171_17166


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l171_17148

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l171_17148


namespace NUMINAMATH_CALUDE_peach_distribution_problem_l171_17164

/-- Represents the distribution of peaches among monkeys -/
structure PeachDistribution where
  total_peaches : ℕ
  num_monkeys : ℕ

/-- Checks if the distribution satisfies the first condition -/
def satisfies_condition1 (d : PeachDistribution) : Prop :=
  2 * 4 + (d.num_monkeys - 2) * 2 + 4 = d.total_peaches

/-- Checks if the distribution satisfies the second condition -/
def satisfies_condition2 (d : PeachDistribution) : Prop :=
  1 * 6 + (d.num_monkeys - 1) * 4 = d.total_peaches + 12

/-- The theorem to be proved -/
theorem peach_distribution_problem :
  ∃ (d : PeachDistribution),
    d.total_peaches = 26 ∧
    d.num_monkeys = 9 ∧
    satisfies_condition1 d ∧
    satisfies_condition2 d :=
by sorry

end NUMINAMATH_CALUDE_peach_distribution_problem_l171_17164


namespace NUMINAMATH_CALUDE_equation_proof_l171_17124

theorem equation_proof : 121 + 2 * 11 * 8 + 64 = 361 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l171_17124


namespace NUMINAMATH_CALUDE_parabola_equation_l171_17176

/-- Represents a parabola with integer coefficients -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  a_pos : 0 < a
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, -2)

/-- The directrix of the parabola -/
def directrix (x y : ℝ) : Prop := 5 * x + 2 * y = 10

/-- Checks if a point is on the parabola -/
def isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.a * x^2 + p.b * x * y + p.c * y^2 + p.d * x + p.e * y + p.f = 0

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation : ∃ (p : Parabola), 
  ∀ (x y : ℝ), isOnParabola p x y ↔ 
    (x - focus.1)^2 + (y - focus.2)^2 = (5 * x + 2 * y - 10)^2 / 29 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l171_17176


namespace NUMINAMATH_CALUDE_sam_distance_walked_sam_walks_25_miles_l171_17121

-- Define the constants
def total_distance : ℝ := 55
def fred_speed : ℝ := 6
def sam_speed : ℝ := 5

-- Define the theorem
theorem sam_distance_walked : ℝ := by
  -- The distance Sam walks
  let d : ℝ := sam_speed * (total_distance / (fred_speed + sam_speed))
  -- Prove that d equals 25
  sorry

-- The main theorem
theorem sam_walks_25_miles :
  sam_distance_walked = 25 := by sorry

end NUMINAMATH_CALUDE_sam_distance_walked_sam_walks_25_miles_l171_17121


namespace NUMINAMATH_CALUDE_solution_count_l171_17153

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem solution_count (a : ℝ) :
  (∀ x > 1, f x ≠ (x - 1) * (a * x - a + 1)) ∨
  (a > 0 ∧ a < 1/2 ∧ (∀ x > 1, f x = (x - 1) * (a * x - a + 1) → 
    ∀ y > 1, y ≠ x → f y ≠ (y - 1) * (a * y - a + 1))) :=
by sorry

end NUMINAMATH_CALUDE_solution_count_l171_17153


namespace NUMINAMATH_CALUDE_visitors_previous_day_l171_17102

/-- The number of visitors to Buckingham Palace over 25 days -/
def total_visitors : ℕ := 949

/-- The number of days over which visitors were counted -/
def total_days : ℕ := 25

/-- The number of visitors on the previous day -/
def previous_day_visitors : ℕ := 246

/-- Theorem stating that the number of visitors on the previous day was 246 -/
theorem visitors_previous_day : previous_day_visitors = 246 := by
  sorry

end NUMINAMATH_CALUDE_visitors_previous_day_l171_17102


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l171_17109

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (α : Real) : Quadrant :=
  sorry

-- Theorem statement
theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α > 0) 
  (h2 : Real.sin α * Real.tan α < 0) : 
  angle_quadrant α = Quadrant.third :=
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l171_17109


namespace NUMINAMATH_CALUDE_total_arrangements_l171_17157

/-- Represents the number of people in the group -/
def total_people : Nat := 6

/-- Represents the number of people who must sit together -/
def group_size : Nat := 3

/-- Calculates the number of ways to arrange the group -/
def arrange_group (n : Nat) : Nat :=
  Nat.factorial n

/-- Calculates the number of ways to choose people for the group -/
def choose_group (n : Nat) : Nat :=
  n

/-- Calculates the number of ways to insert the group -/
def insert_group (n : Nat) : Nat :=
  n * (n - 1)

/-- The main theorem stating the total number of arrangements -/
theorem total_arrangements :
  arrange_group (total_people - group_size) *
  choose_group group_size *
  insert_group (total_people - group_size + 1) = 216 :=
sorry

end NUMINAMATH_CALUDE_total_arrangements_l171_17157


namespace NUMINAMATH_CALUDE_math_test_problems_left_l171_17128

/-- Calculates the number of problems left to solve in a math test -/
def problems_left_to_solve (total_problems : ℕ) (first_20min : ℕ) (second_20min : ℕ) : ℕ :=
  total_problems - (first_20min + second_20min)

/-- Proves that given the conditions, the number of problems left to solve is 45 -/
theorem math_test_problems_left : 
  let total_problems : ℕ := 75
  let first_20min : ℕ := 10
  let second_20min : ℕ := first_20min * 2
  problems_left_to_solve total_problems first_20min second_20min = 45 := by
  sorry

#eval problems_left_to_solve 75 10 20

end NUMINAMATH_CALUDE_math_test_problems_left_l171_17128


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l171_17118

/-- Given a hyperbola and a circle, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∀ x y : ℝ, (x - 2)^2 + y^2 = 1 → 
    ∃ k : ℝ, (x = k ∧ y = k * (b / a)) ∨ (x = -k ∧ y = k * (b / a))) →
  ∃ c : ℝ, c^2 = 3 ∧ (∀ x y : ℝ, x + c * y = 0 ∨ x - c * y = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l171_17118


namespace NUMINAMATH_CALUDE_average_weight_abc_l171_17179

/-- Given three weights a, b, and c, prove that their average is 43 kg -/
theorem average_weight_abc (a b c : ℝ) 
  (hab : (a + b) / 2 = 40)  -- average of a and b is 40 kg
  (hbc : (b + c) / 2 = 43)  -- average of b and c is 43 kg
  (hb : b = 37)             -- weight of b is 37 kg
  : (a + b + c) / 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_abc_l171_17179


namespace NUMINAMATH_CALUDE_probability_black_ball_l171_17197

def total_balls : ℕ := 2 + 3

def black_balls : ℕ := 2

theorem probability_black_ball :
  (black_balls : ℚ) / total_balls = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_black_ball_l171_17197


namespace NUMINAMATH_CALUDE_lighthouse_coverage_l171_17120

/-- Represents a lighthouse with its illumination angle -/
structure Lighthouse where
  angle : ℝ

/-- Represents the Persian Gulf as a circle -/
def PersianGulf : ℝ := 360

/-- The number of lighthouses -/
def num_lighthouses : ℕ := 18

/-- The illumination angle of each lighthouse -/
def lighthouse_angle : ℝ := 20

/-- Proves that the lighthouses can cover the entire Persian Gulf -/
theorem lighthouse_coverage (lighthouses : Fin num_lighthouses → Lighthouse)
  (h1 : ∀ i, (lighthouses i).angle = lighthouse_angle)
  (h2 : lighthouse_angle * num_lighthouses = PersianGulf) :
  ∃ (arrangement : Fin num_lighthouses → ℝ),
    (∀ i, 0 ≤ arrangement i ∧ arrangement i < PersianGulf) ∧
    (∀ x, 0 ≤ x ∧ x < PersianGulf →
      ∃ i, x ∈ Set.Icc (arrangement i) ((arrangement i + (lighthouses i).angle) % PersianGulf)) :=
by sorry

end NUMINAMATH_CALUDE_lighthouse_coverage_l171_17120


namespace NUMINAMATH_CALUDE_range_of_expression_l171_17126

theorem range_of_expression (x y : ℝ) 
  (h1 : x ≥ 0) 
  (h2 : y ≥ x) 
  (h3 : 4 * x + 3 * y ≤ 12) : 
  3 ≤ (x + 2 * y + 3) / (x + 1) ∧ (x + 2 * y + 3) / (x + 1) ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l171_17126


namespace NUMINAMATH_CALUDE_bryce_raisins_l171_17195

theorem bryce_raisins (bryce carter : ℕ) : 
  bryce = carter + 8 →
  carter = bryce / 3 →
  bryce + carter = 44 →
  bryce = 33 := by
sorry

end NUMINAMATH_CALUDE_bryce_raisins_l171_17195


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l171_17108

theorem alpha_plus_beta_equals_two (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α = 1) 
  (h2 : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l171_17108


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l171_17175

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), x = 4 ∧ (2 : ℚ) / 7 < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ∧
  ∀ (y : ℤ), y > x → ¬((2 : ℚ) / 7 < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l171_17175


namespace NUMINAMATH_CALUDE_crow_probability_l171_17151

/-- Represents the number of crows of each color on each tree -/
structure CrowCounts where
  birchWhite : ℕ
  birchBlack : ℕ
  oakWhite : ℕ
  oakBlack : ℕ

/-- The probability of the number of white crows on the birch returning to its initial count -/
def probReturnToInitial (c : CrowCounts) : ℚ :=
  (c.birchBlack * (c.oakBlack + 1) + c.birchWhite * (c.oakWhite + 1)) / (50 * 51)

/-- The probability of the number of white crows on the birch changing -/
def probChange (c : CrowCounts) : ℚ :=
  (c.birchBlack * c.oakWhite + c.birchWhite * c.oakBlack) / (50 * 51)

theorem crow_probability (c : CrowCounts) 
  (h1 : c.birchWhite + c.birchBlack = 50)
  (h2 : c.oakWhite + c.oakBlack = 50)
  (h3 : c.birchWhite > 0)
  (h4 : c.birchBlack ≥ c.birchWhite)
  (h5 : c.oakBlack ≥ c.oakWhite - 1) :
  probReturnToInitial c > probChange c := by
  sorry

end NUMINAMATH_CALUDE_crow_probability_l171_17151


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l171_17149

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 2*x) + 2 * (x + 3*x^3 - 2*x^2 + 4*x^5 - x^3) - 6 * (2 + 2*x - 5*x^3 - 3*x^2 + x^4)

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression))) 0 / 6 = 26 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l171_17149


namespace NUMINAMATH_CALUDE_segment_length_l171_17185

theorem segment_length : Real.sqrt 193 = Real.sqrt ((8 - 1)^2 + (14 - 2)^2) := by sorry

end NUMINAMATH_CALUDE_segment_length_l171_17185


namespace NUMINAMATH_CALUDE_flowchart_requirement_l171_17143

-- Define the structure of a flowchart
structure Flowchart where
  boxes : Set (Operation)
  flowLines : Set (SequenceIndicator)

-- Define operations
inductive Operation
  | process : Operation
  | decision : Operation
  | inputOutput : Operation

-- Define sequence indicators
inductive SequenceIndicator
  | arrow : SequenceIndicator

-- Define the direction of flow
inductive FlowDirection
  | leftToRight : FlowDirection
  | topToBottom : FlowDirection

-- Define the general requirement for drawing a flowchart
def generalRequirement : (FlowDirection × FlowDirection) := (FlowDirection.leftToRight, FlowDirection.topToBottom)

-- Theorem: The general requirement for drawing a flowchart is from left to right, from top to bottom
theorem flowchart_requirement (f : Flowchart) : 
  generalRequirement = (FlowDirection.leftToRight, FlowDirection.topToBottom) := by
  sorry

end NUMINAMATH_CALUDE_flowchart_requirement_l171_17143


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l171_17107

/-- A parallelogram in 2D space --/
structure Parallelogram where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ

/-- The property that defines a parallelogram --/
def isParallelogram (p : Parallelogram) : Prop :=
  (p.a.1 + p.c.1 = p.b.1 + p.d.1) ∧ 
  (p.a.2 + p.c.2 = p.b.2 + p.d.2)

theorem parallelogram_fourth_vertex 
  (p : Parallelogram)
  (h1 : p.a = (-1, 0))
  (h2 : p.b = (3, 0))
  (h3 : p.c = (1, -5))
  (h4 : isParallelogram p) :
  p.d = (1, 5) ∨ p.d = (-3, -5) := by
  sorry

#check parallelogram_fourth_vertex

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l171_17107


namespace NUMINAMATH_CALUDE_least_number_of_cans_l171_17167

def maaza_liters : ℕ := 80
def pepsi_liters : ℕ := 144
def sprite_liters : ℕ := 368

def can_size : ℕ := Nat.gcd maaza_liters (Nat.gcd pepsi_liters sprite_liters)

def total_cans : ℕ := maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size

theorem least_number_of_cans :
  (∀ k : ℕ, k > 0 → (maaza_liters % k = 0 ∧ pepsi_liters % k = 0 ∧ sprite_liters % k = 0) → k ≤ can_size) ∧
  total_cans = 37 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l171_17167


namespace NUMINAMATH_CALUDE_solution_set_inequality_l171_17173

-- Define the function f
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem solution_set_inequality (x : ℝ) :
  f (2 * x^2 - 1) < -1 ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l171_17173


namespace NUMINAMATH_CALUDE_a_range_f_result_l171_17186

noncomputable section

variables (a x x₁ x₂ t : ℝ)

def f (x : ℝ) := Real.exp x - a * x + a

def f' (x : ℝ) := Real.exp x - a

axiom a_positive : a > 0

axiom x₁_less_x₂ : x₁ < x₂

axiom f_roots : f a x₁ = 0 ∧ f a x₂ = 0

axiom t_def : Real.sqrt ((x₂ - 1) / (x₁ - 1)) = t

axiom isosceles_right_triangle : ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo x₁ x₂ ∧ 
  f a x₀ = (x₁ - x₂) / 2

theorem a_range : a > Real.exp 2 := by sorry

theorem f'_negative : f' a (Real.sqrt (x₁ * x₂)) < 0 := by sorry

theorem result : (a - 1) * (t - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_a_range_f_result_l171_17186


namespace NUMINAMATH_CALUDE_age_difference_john_aunt_l171_17180

/-- Represents the ages of family members --/
structure FamilyAges where
  john : ℕ
  father : ℕ
  mother : ℕ
  grandmother : ℕ
  aunt : ℕ

/-- Defines the relationships between family members' ages --/
def valid_family_ages (ages : FamilyAges) : Prop :=
  ages.john * 2 = ages.father ∧
  ages.father = ages.mother + 4 ∧
  ages.grandmother = ages.john * 3 ∧
  ages.aunt = ages.mother * 2 - 5 ∧
  ages.father = 40

/-- Theorem stating the age difference between John and his aunt --/
theorem age_difference_john_aunt (ages : FamilyAges) 
  (h : valid_family_ages ages) : ages.aunt - ages.john = 47 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_john_aunt_l171_17180


namespace NUMINAMATH_CALUDE_f_is_quadratic_l171_17127

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation x^2 - 3x + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem stating that f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l171_17127


namespace NUMINAMATH_CALUDE_completing_square_l171_17188

theorem completing_square (x : ℝ) : x^2 - 4*x - 8 = 0 ↔ (x - 2)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l171_17188


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l171_17142

theorem subtraction_of_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l171_17142


namespace NUMINAMATH_CALUDE_mem_not_veen_l171_17189

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem En Veen : Set U)

-- Define the hypotheses
variable (h1 : Mem ⊆ En)
variable (h2 : En ∩ Veen = ∅)

-- Theorem to prove
theorem mem_not_veen :
  (∀ x, x ∈ Mem → x ∉ Veen) ∧
  (Mem ∩ Veen = ∅) :=
sorry

end NUMINAMATH_CALUDE_mem_not_veen_l171_17189


namespace NUMINAMATH_CALUDE_chord_count_l171_17174

/-- The number of points on the circumference of the circle -/
def n : ℕ := 9

/-- The number of points needed to form a chord -/
def r : ℕ := 2

/-- The number of different chords that can be drawn -/
def num_chords : ℕ := Nat.choose n r

theorem chord_count : num_chords = 36 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l171_17174


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l171_17123

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_even_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (fun x ↦ f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 17 = 1 := by sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l171_17123


namespace NUMINAMATH_CALUDE_triangle_sides_relation_l171_17130

theorem triangle_sides_relation (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_relation_l171_17130


namespace NUMINAMATH_CALUDE_fruit_purchase_problem_l171_17134

theorem fruit_purchase_problem (x y : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ x + y = 7 ∧ 5 * x + 8 * y = 41 → x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_problem_l171_17134


namespace NUMINAMATH_CALUDE_g_value_at_10_l171_17182

theorem g_value_at_10 (g : ℕ → ℝ) 
  (h1 : g 1 = 1)
  (h2 : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = (g (2*m) + g (2*n))/2 + 2) :
  g 10 = 102 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_10_l171_17182


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l171_17115

theorem modulus_of_complex_number (α : Real) (h : π < α ∧ α < 2*π) :
  Complex.abs (1 + Complex.cos α + Complex.I * Complex.sin α) = -2 * Real.cos (α/2) :=
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l171_17115


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l171_17135

theorem quadratic_roots_sum (a b c : ℝ) : 
  (∀ x : ℝ, a * (x^4 + x^2)^2 + b * (x^4 + x^2) + c ≥ a * (x^3 + 2)^2 + b * (x^3 + 2) + c) →
  (∃ r₁ r₂ : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = a * (x - r₁) * (x - r₂)) →
  r₁ + r₂ = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l171_17135


namespace NUMINAMATH_CALUDE_fraction_simplification_l171_17191

theorem fraction_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : z - 1/x ≠ 0) :
  (x*z - 1/y) / (z - 1/x) = z :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l171_17191


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l171_17170

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) →
  k ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l171_17170


namespace NUMINAMATH_CALUDE_unique_solution_l171_17199

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

/-- The equation that n must satisfy -/
def satisfies_equation (n : ℕ) : Prop :=
  factorial (n + 1) + factorial (n + 3) = factorial n * 1540

theorem unique_solution :
  ∃! n : ℕ, n > 0 ∧ satisfies_equation n ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l171_17199


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l171_17110

/-- Given a point P (x_P, y_P) on the curve 2x^2 - y = 0 and a fixed point A (0, -1),
    the midpoint M (x, y) of AP satisfies the equation 8x^2 - 2y - 1 = 0 -/
theorem midpoint_trajectory (x_P y_P x y : ℝ) : 
  (2 * x_P^2 = y_P) →  -- P is on the curve 2x^2 - y = 0
  (x = x_P / 2) →      -- x-coordinate of midpoint
  (y = (y_P - 1) / 2)  -- y-coordinate of midpoint
  → 8 * x^2 - 2 * y - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l171_17110


namespace NUMINAMATH_CALUDE_inequality_proof_l171_17133

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.log (1 + Real.exp 1))
  (hb : b = Real.sqrt (Real.exp 1))
  (hc : c = 2 * Real.exp 1 / 3) :
  c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l171_17133


namespace NUMINAMATH_CALUDE_plus_shape_perimeter_l171_17139

/-- A shape formed by eight congruent squares arranged in a "plus" sign -/
structure PlusShape where
  /-- The side length of each square in the shape -/
  side_length : ℝ
  /-- The total area of the shape -/
  total_area : ℝ
  /-- The shape is formed by eight congruent squares -/
  area_eq : total_area = 8 * side_length ^ 2

/-- The perimeter of a PlusShape -/
def perimeter (shape : PlusShape) : ℝ := 12 * shape.side_length

theorem plus_shape_perimeter (shape : PlusShape) (h : shape.total_area = 648) :
  perimeter shape = 108 := by
  sorry

#check plus_shape_perimeter

end NUMINAMATH_CALUDE_plus_shape_perimeter_l171_17139


namespace NUMINAMATH_CALUDE_simple_interest_rate_l171_17144

theorem simple_interest_rate (P : ℝ) (h : P > 0) : 
  (∃ R : ℝ, R > 0 ∧ P + (P * R * 15) / 100 = 2 * P) → 
  (∃ R : ℝ, R > 0 ∧ P + (P * R * 15) / 100 = 2 * P ∧ R = 100 / 15) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l171_17144


namespace NUMINAMATH_CALUDE_solar_eclipse_viewers_scientific_notation_l171_17181

theorem solar_eclipse_viewers_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 2580000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.58 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solar_eclipse_viewers_scientific_notation_l171_17181


namespace NUMINAMATH_CALUDE_vector_c_value_l171_17111

/-- Given vectors a and b, if vector c satisfies the parallel and perpendicular conditions,
    then c equals the specified vector. -/
theorem vector_c_value (a b c : ℝ × ℝ) : 
  a = (1, 2) →
  b = (2, -3) →
  (∃ k : ℝ, c + a = k • b) →
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) →
  c = (-7/9, -7/3) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l171_17111


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l171_17146

def is_valid_sum (a b c d e f : ℕ) : Prop :=
  a ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  e ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  f ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  100 ≤ a ∧ a ≤ 999 ∧
  100 ≤ d ∧ d ≤ 999 ∧
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) ≤ 1500

theorem smallest_sum_proof :
  ∀ a b c d e f : ℕ,
    is_valid_sum a b c d e f →
    (100 * a + 10 * b + c) + (100 * d + 10 * e + f) ≥ 417 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l171_17146


namespace NUMINAMATH_CALUDE_max_product_sum_300_l171_17136

theorem max_product_sum_300 :
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l171_17136


namespace NUMINAMATH_CALUDE_stem_and_leaf_plot_preserves_information_l171_17138

-- Define the different types of charts
inductive ChartType
  | BarChart
  | PieChart
  | LineChart
  | StemAndLeafPlot

-- Define a property for information preservation
def preserves_all_information (chart : ChartType) : Prop :=
  match chart with
  | ChartType.StemAndLeafPlot => True
  | _ => False

-- Theorem statement
theorem stem_and_leaf_plot_preserves_information :
  ∀ (chart : ChartType), preserves_all_information chart ↔ chart = ChartType.StemAndLeafPlot :=
by
  sorry


end NUMINAMATH_CALUDE_stem_and_leaf_plot_preserves_information_l171_17138


namespace NUMINAMATH_CALUDE_pager_fraction_l171_17193

theorem pager_fraction (total : ℝ) (total_pos : 0 < total) : 
  let cell_phone := (2/3 : ℝ) * total
  let neither := (1/3 : ℝ) * total
  let both := (0.4 : ℝ) * total
  let pager := (0.8 : ℝ) * total
  (cell_phone + (pager - both) = total - neither) →
  (pager / total = 0.8) :=
by
  sorry

end NUMINAMATH_CALUDE_pager_fraction_l171_17193


namespace NUMINAMATH_CALUDE_book_pages_l171_17101

/-- The number of pages Charlie read in the book -/
def total_pages : ℕ :=
  let first_four_days : ℕ := 4 * 45
  let next_three_days : ℕ := 3 * 52
  let last_day : ℕ := 15
  first_four_days + next_three_days + last_day

/-- Theorem stating that the total number of pages in the book is 351 -/
theorem book_pages : total_pages = 351 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l171_17101


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l171_17194

theorem decimal_to_fraction :
  (0.34 : ℚ) = 17 / 50 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l171_17194


namespace NUMINAMATH_CALUDE_incircle_radius_l171_17163

/-- An isosceles triangle with base 10 and height 12 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  is_isosceles : base = 10 ∧ height = 12

/-- The incircle of a triangle -/
def incircle (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: The radius of the incircle of the given isosceles triangle is 10/3 -/
theorem incircle_radius (t : IsoscelesTriangle) : incircle t = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_l171_17163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l171_17122

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n, prove S_8 = 80 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →  -- Definition of S_n
  S 4 = 24 →                                                      -- Given condition
  a 8 = 17 →                                                      -- Given condition
  S 8 = 80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l171_17122


namespace NUMINAMATH_CALUDE_area_of_XYZ_main_theorem_l171_17171

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (AB BC CA : ℝ)

/-- Points on the triangle --/
structure TrianglePoints :=
  (A B C D P Q X Y Z : ℝ × ℝ)

/-- Given triangle ABC with altitude AD and inscribed circle tangent points --/
def given_triangle : Triangle := { AB := 13, BC := 14, CA := 15 }

/-- Theorem: Area of triangle XYZ is 25/4 --/
theorem area_of_XYZ (t : Triangle) (tp : TrianglePoints) : ℝ :=
  let triangle := given_triangle
  25 / 4

/-- Main theorem --/
theorem main_theorem (t : Triangle) (tp : TrianglePoints) : 
  t = given_triangle → 
  tp.D.1 = tp.B.1 ∨ tp.D.1 = tp.C.1 →  -- D is on BC
  (tp.A.1 - tp.D.1)^2 + (tp.A.2 - tp.D.2)^2 = (tp.B.1 - tp.D.1)^2 + (tp.B.2 - tp.D.2)^2 →  -- AD perpendicular to BC
  (tp.P.1 - tp.A.1) * (tp.D.1 - tp.A.1) + (tp.P.2 - tp.A.2) * (tp.D.2 - tp.A.2) = 0 →  -- P on AD
  (tp.Q.1 - tp.A.1) * (tp.D.1 - tp.A.1) + (tp.Q.2 - tp.A.2) * (tp.D.2 - tp.A.2) = 0 →  -- Q on AD
  tp.X.1 = tp.B.1 ∨ tp.X.1 = tp.C.1 →  -- X on BC
  tp.Y.1 = tp.B.1 ∨ tp.Y.1 = tp.C.1 →  -- Y on BC
  (tp.Z.1 - tp.P.1) * (tp.X.1 - tp.P.1) + (tp.Z.2 - tp.P.2) * (tp.X.2 - tp.P.2) = 0 →  -- Z on PX
  (tp.Z.1 - tp.Q.1) * (tp.Y.1 - tp.Q.1) + (tp.Z.2 - tp.Q.2) * (tp.Y.2 - tp.Q.2) = 0 →  -- Z on QY
  area_of_XYZ t tp = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_XYZ_main_theorem_l171_17171


namespace NUMINAMATH_CALUDE_range_of_f_l171_17100

def f (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc 2 5, f x = y}
  S = Set.Icc (-3) 6 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l171_17100


namespace NUMINAMATH_CALUDE_remaining_area_formula_l171_17165

/-- The area of the remaining part when a small square is removed from a larger square -/
def remaining_area (x : ℝ) : ℝ := 9 - x^2

/-- Theorem stating the area of the remaining part when a small square is removed from a larger square -/
theorem remaining_area_formula (x : ℝ) (h : 0 < x ∧ x < 3) : 
  remaining_area x = 9 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_formula_l171_17165


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l171_17113

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) /
  Real.cos (10 * π / 180) = (Real.sqrt 3 + 2) * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l171_17113


namespace NUMINAMATH_CALUDE_hotel_room_charges_l171_17187

theorem hotel_room_charges (G : ℝ) (h1 : G > 0) : 
  let R := G * (1 + 0.19999999999999986)
  let P := R * (1 - 0.25)
  P = G * (1 - 0.1) :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l171_17187


namespace NUMINAMATH_CALUDE_rectangle_area_l171_17190

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 16 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l171_17190


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l171_17114

theorem arithmetic_calculations :
  ((-15) + (-23) - 26 - (-15) = -49) ∧
  ((-1/2 + 2/3 - 1/4) * (-24) = 2) ∧
  ((-24) / (-6) * (-1/4) = -1) ∧
  ((-1)^2024 - (-2)^3 - 3^2 + 2 / (2/3) * (3/2) = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l171_17114


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l171_17161

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 5, -1]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-30, -9; 11, 1]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![5, -8; 7/13, 35/13]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l171_17161


namespace NUMINAMATH_CALUDE_complement_M_intersect_P_l171_17156

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | |x - 1/2| ≤ 5/2}
def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem complement_M_intersect_P :
  (U \ M) ∩ P = {x | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_P_l171_17156


namespace NUMINAMATH_CALUDE_complex_modulus_l171_17103

theorem complex_modulus (z : ℂ) : z = (1 + I) / (2 - I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l171_17103


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l171_17198

/-- Given an arithmetic sequence {a_n} with a_1 = -2014 and S_n as the sum of first n terms,
    if S_{2012}/2012 - S_{10}/10 = 2002, then S_{2016} = 2016 -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)) →  -- Definition of S_n
  (a 1 = -2014) →                                         -- First term condition
  (S 2012 / 2012 - S 10 / 10 = 2002) →                    -- Given condition
  (S 2016 = 2016) :=                                      -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l171_17198


namespace NUMINAMATH_CALUDE_revenue_change_l171_17141

theorem revenue_change
  (initial_price initial_quantity : ℝ)
  (price_increase : ℝ)
  (quantity_decrease : ℝ)
  (h_price : price_increase = 0.4)
  (h_quantity : quantity_decrease = 0.2)
  : (1 + price_increase) * (1 - quantity_decrease) * initial_price * initial_quantity
    = 1.12 * initial_price * initial_quantity :=
by sorry

end NUMINAMATH_CALUDE_revenue_change_l171_17141


namespace NUMINAMATH_CALUDE_triangle_area_l171_17192

/-- Given a right isosceles triangle that shares sides with squares of areas 100, 64, and 100,
    prove that the area of the triangle is 50. -/
theorem triangle_area (a b c : ℝ) (ha : a^2 = 100) (hb : b^2 = 64) (hc : c^2 = 100)
  (right_isosceles : a = c ∧ a^2 + c^2 = b^2) : (1/2) * a * c = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l171_17192


namespace NUMINAMATH_CALUDE_number_divided_by_three_l171_17125

theorem number_divided_by_three : ∃ x : ℤ, x / 3 = x - 24 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l171_17125


namespace NUMINAMATH_CALUDE_age_difference_proof_l171_17184

/-- Represents the age difference between Petra's mother and twice Petra's age --/
def age_difference (petra_age : ℕ) (mother_age : ℕ) : ℕ :=
  mother_age - 2 * petra_age

/-- Theorem stating the age difference between Petra's mother and twice Petra's age --/
theorem age_difference_proof :
  let petra_age : ℕ := 11
  let mother_age : ℕ := 36
  age_difference petra_age mother_age = 14 ∧
  petra_age + mother_age = 47 ∧
  ∃ (n : ℕ), mother_age = 2 * petra_age + n :=
by sorry

end NUMINAMATH_CALUDE_age_difference_proof_l171_17184


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l171_17131

/-- Given the following definitions:
  a = 2√2 + 3√3 + 4√6
  b = -2√2 + 3√3 + 4√6
  c = 2√2 - 3√3 + 4√6
  d = -2√2 - 3√3 + 4√6
  Prove that (1/a + 1/b + 1/c + 1/d)² = 952576/70225 -/
theorem sum_of_reciprocals_squared (a b c d : ℝ) :
  a = 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  b = -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  c = 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  d = -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6 →
  (1/a + 1/b + 1/c + 1/d)^2 = 952576/70225 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l171_17131


namespace NUMINAMATH_CALUDE_current_rate_calculation_l171_17140

/-- Given a boat traveling downstream, calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) -- Speed of the boat in still water (km/hr)
  (distance : ℝ)   -- Distance traveled downstream (km)
  (time : ℝ)       -- Time traveled downstream (minutes)
  (h1 : boat_speed = 42)
  (h2 : distance = 36.67)
  (h3 : time = 44)
  : ∃ (current_rate : ℝ), current_rate = 8 ∧ 
    distance = (boat_speed + current_rate) * (time / 60) :=
by sorry


end NUMINAMATH_CALUDE_current_rate_calculation_l171_17140


namespace NUMINAMATH_CALUDE_angle_rotation_l171_17147

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 60) (h2 : rotation = 630) :
  (initial_angle + rotation) % 360 = 330 ∧ 360 - (initial_angle + rotation) % 360 = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_rotation_l171_17147


namespace NUMINAMATH_CALUDE_max_distance_complex_circle_l171_17129

theorem max_distance_complex_circle (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (max_val : ℝ), max_val = 4 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_circle_l171_17129


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l171_17159

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l171_17159


namespace NUMINAMATH_CALUDE_distance_difference_l171_17150

/-- The line l -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The ellipse C₁ -/
def ellipse_C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point F -/
def point_F : ℝ × ℝ := (1, 0)

/-- Point F₁ -/
def point_F₁ : ℝ × ℝ := (-1, 0)

/-- Theorem stating the difference of distances -/
theorem distance_difference (A B : ℝ × ℝ) 
  (h_line_A : line_l A.1 A.2)
  (h_line_B : line_l B.1 B.2)
  (h_ellipse_A : ellipse_C₁ A.1 A.2)
  (h_ellipse_B : ellipse_C₁ B.1 B.2)
  (h_above : A.2 > B.2) :
  |point_F₁.1 - A.1|^2 + |point_F₁.2 - A.2|^2 - 
  (|point_F₁.1 - B.1|^2 + |point_F₁.2 - B.2|^2) = (6 * Real.sqrt 2 / 7)^2 :=
sorry

end NUMINAMATH_CALUDE_distance_difference_l171_17150


namespace NUMINAMATH_CALUDE_sum_of_digits_in_multiple_of_72_l171_17145

theorem sum_of_digits_in_multiple_of_72 (A B : ℕ) : 
  A < 10 → B < 10 → (A * 100000 + 44610 + B) % 72 = 0 → A + B = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_multiple_of_72_l171_17145


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l171_17155

/-- The lateral area of a cylinder with diameter and height both equal to 4 is 16π. -/
theorem cylinder_lateral_area : 
  ∀ (d h : ℝ), d = 4 → h = 4 → 2 * π * (d / 2) * h = 16 * π :=
by
  sorry


end NUMINAMATH_CALUDE_cylinder_lateral_area_l171_17155


namespace NUMINAMATH_CALUDE_train_length_l171_17106

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 225 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l171_17106
