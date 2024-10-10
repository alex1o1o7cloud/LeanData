import Mathlib

namespace inscribed_cylinder_radius_l3793_379356

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ

/-- Predicate to check if a cylinder is inscribed in a cone -/
def is_inscribed (cylinder : Cylinder) (cone : Cone) : Prop :=
  -- This is a placeholder for the actual geometric condition
  True

theorem inscribed_cylinder_radius (cone : Cone) (cylinder : Cylinder) :
  cone.diameter = 8 →
  cone.altitude = 10 →
  is_inscribed cylinder cone →
  cylinder.radius * 2 = cylinder.radius * 2 →  -- Diameter equals height
  cylinder.radius = 20 / 9 := by
  sorry

end inscribed_cylinder_radius_l3793_379356


namespace fraction_sum_equals_one_l3793_379342

theorem fraction_sum_equals_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : x - y + z = x * y * z) : 1 / x - 1 / y + 1 / z = 1 := by
  sorry

end fraction_sum_equals_one_l3793_379342


namespace problem_8_l3793_379399

theorem problem_8 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + b^2 + c^2 = 63)
  (h2 : 2*a + 3*b + 6*c = 21*Real.sqrt 7) :
  (a/c)^(a/b) = (1/3)^(2/3) := by
  sorry

end problem_8_l3793_379399


namespace tangerine_count_l3793_379365

theorem tangerine_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  initial = 10 → added = 6 → total = initial + added → total = 16 := by
sorry

end tangerine_count_l3793_379365


namespace water_pouring_proof_l3793_379319

/-- Represents the fraction of water remaining after n pourings -/
def remainingWater (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

theorem water_pouring_proof :
  remainingWater 28 = 1 / 15 := by
  sorry

end water_pouring_proof_l3793_379319


namespace basketball_team_selection_l3793_379347

theorem basketball_team_selection (n m k : ℕ) (h1 : n = 18) (h2 : m = 2) (h3 : k = 8) :
  Nat.choose (n - m) (k - m) = 8008 := by
  sorry

end basketball_team_selection_l3793_379347


namespace arithmetic_sequence_sin_property_l3793_379323

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 :=
by
  sorry

end arithmetic_sequence_sin_property_l3793_379323


namespace second_polygon_sides_l3793_379311

theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure positive side length
  50 * (3 * s) = n * s →  -- Same perimeter condition
  n = 150 := by
sorry

end second_polygon_sides_l3793_379311


namespace fraction_sum_equals_one_l3793_379368

theorem fraction_sum_equals_one (m n : ℝ) (h : m ≠ n) :
  m / (m - n) + n / (n - m) = 1 := by
  sorry

end fraction_sum_equals_one_l3793_379368


namespace shortest_distance_between_circles_l3793_379381

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 12*x + y^2 - 6*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 + 8*y + 34 = 0}
  (shortest_distance : ℝ) →
  shortest_distance = Real.sqrt 170 - 3 - Real.sqrt 7 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ),
    p1 ∈ circle1 → p2 ∈ circle2 →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ shortest_distance :=
by
  sorry


end shortest_distance_between_circles_l3793_379381


namespace set_cardinality_lower_bound_l3793_379306

theorem set_cardinality_lower_bound (A : Finset ℤ) (m : ℕ) (hm : m ≥ 2) 
  (B : Fin m → Finset ℤ) (hB : ∀ i, B i ⊆ A) (hB_nonempty : ∀ i, (B i).Nonempty) 
  (hsum : ∀ i, (B i).sum id = m ^ (i : ℕ).succ) : 
  A.card ≥ m / 2 := by
  sorry

end set_cardinality_lower_bound_l3793_379306


namespace range_of_g_l3793_379341

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1

-- Define the range of a function
def has_range (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ y ∈ S, ∃ x, f x = y

-- State the theorem
theorem range_of_g (a : ℝ) :
  (has_range (f a) Set.univ) → (has_range (g a) {y | y ≥ 1}) :=
by sorry

end range_of_g_l3793_379341


namespace rachel_homework_pages_l3793_379388

/-- The number of pages of math homework Rachel has to complete -/
def math_homework : ℕ := 8

/-- The number of pages of biology homework Rachel has to complete -/
def biology_homework : ℕ := 3

/-- The total number of pages of math and biology homework Rachel has to complete -/
def total_homework : ℕ := math_homework + biology_homework

theorem rachel_homework_pages :
  total_homework = 11 :=
by sorry

end rachel_homework_pages_l3793_379388


namespace weight_order_l3793_379348

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ → ℕ := (· * 1000)

/-- Conversion factor from tonnes to grams -/
def t_to_g : ℕ → ℕ := (· * 1000000)

/-- Weight in grams -/
def weight_908g : ℕ := 908

/-- Weight in grams (9kg80g) -/
def weight_9kg80g : ℕ := kg_to_g 9 + 80

/-- Weight in grams (900kg) -/
def weight_900kg : ℕ := kg_to_g 900

/-- Weight in grams (0.09t) -/
def weight_009t : ℕ := t_to_g 0 + 90000

theorem weight_order :
  weight_908g < weight_9kg80g ∧
  weight_9kg80g < weight_009t ∧
  weight_009t < weight_900kg := by
  sorry

end weight_order_l3793_379348


namespace triangle_area_l3793_379318

/-- The area of a triangle with base 9 cm and height 12 cm is 54 cm². -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 9 → height = 12 → area = (1/2) * base * height → 
  area = 54 := by
  sorry

end triangle_area_l3793_379318


namespace geometric_sequence_ratio_l3793_379322

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_5 - a_1 = 15 and a_4 - a_2 = 6, then q = 1/2 or q = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 5 - a 1 = 15 →              -- Condition 1
  a 4 - a 2 = 6 →               -- Condition 2
  q = 1/2 ∨ q = 2 :=            -- Conclusion
by sorry

end geometric_sequence_ratio_l3793_379322


namespace cubic_inequality_range_l3793_379349

theorem cubic_inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, m * x^3 - x^2 + 4*x + 3 ≥ 0) ↔ m ∈ Set.Icc (-6 : ℝ) (-2) := by
  sorry

end cubic_inequality_range_l3793_379349


namespace complex_sum_conjugate_l3793_379312

open Complex

theorem complex_sum_conjugate (α β γ : ℝ) 
  (h : exp (I * α) + exp (I * β) + exp (I * γ) = (1 / 3 : ℂ) + (1 / 2 : ℂ) * I) : 
  exp (-I * α) + exp (-I * β) + exp (-I * γ) = (1 / 3 : ℂ) - (1 / 2 : ℂ) * I := by
  sorry

end complex_sum_conjugate_l3793_379312


namespace max_correct_answers_l3793_379364

/-- Represents the result of a math contest. -/
structure ContestResult where
  correct : ℕ
  blank : ℕ
  incorrect : ℕ
  deriving Repr

/-- Calculates the score for a given contest result. -/
def calculateScore (result : ContestResult) : ℤ :=
  5 * result.correct - 2 * result.incorrect

/-- Checks if a contest result is valid (total questions = 60). -/
def isValidResult (result : ContestResult) : Prop :=
  result.correct + result.blank + result.incorrect = 60

/-- Theorem stating the maximum number of correct answers Evelyn could have. -/
theorem max_correct_answers (result : ContestResult) 
  (h1 : isValidResult result) 
  (h2 : calculateScore result = 150) : 
  result.correct ≤ 38 := by
  sorry

end max_correct_answers_l3793_379364


namespace area_ratio_extended_triangle_l3793_379362

-- Define the triangle ABC and its extensions
structure ExtendedTriangle where
  -- Original equilateral triangle
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Extended sides
  BB' : ℝ
  CC' : ℝ
  AA' : ℝ
  -- Conditions
  equilateral : AB = BC ∧ BC = CA
  extension_BB' : BB' = 2 * AB
  extension_CC' : CC' = 3 * BC
  extension_AA' : AA' = 4 * CA

-- Define the theorem
theorem area_ratio_extended_triangle (t : ExtendedTriangle) :
  (t.AB + t.BB')^2 + (t.BC + t.CC')^2 + (t.CA + t.AA')^2 = 25 * (t.AB^2 + t.BC^2 + t.CA^2) :=
by sorry

end area_ratio_extended_triangle_l3793_379362


namespace trigonometric_simplification_l3793_379337

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
                   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)
  numerator / denominator = 
    (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
    (2 * Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) := by
  sorry

end trigonometric_simplification_l3793_379337


namespace point_in_fourth_quadrant_l3793_379321

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point -/
def given_point : Point2D :=
  { x := 3, y := -4 }

/-- Theorem: The given point lies in the fourth quadrant -/
theorem point_in_fourth_quadrant :
  fourth_quadrant given_point := by
  sorry


end point_in_fourth_quadrant_l3793_379321


namespace weight_of_larger_square_l3793_379360

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two wood squares of different sizes -/
theorem weight_of_larger_square
  (small : WoodSquare)
  (large : WoodSquare)
  (h1 : small.side = 4)
  (h2 : small.weight = 16)
  (h3 : large.side = 6)
  (h4 : large.weight = (large.side^2 / small.side^2) * small.weight) :
  large.weight = 36 := by
  sorry

end weight_of_larger_square_l3793_379360


namespace time_difference_l3793_379343

-- Define constants
def blocks : ℕ := 12
def walk_time_per_block : ℕ := 1  -- in minutes
def bike_time_per_block : ℕ := 20 -- in seconds

-- Define functions
def walk_time : ℕ := blocks * walk_time_per_block

def bike_time_seconds : ℕ := blocks * bike_time_per_block
def bike_time : ℕ := bike_time_seconds / 60

-- Theorem
theorem time_difference : walk_time - bike_time = 8 := by
  sorry


end time_difference_l3793_379343


namespace flagpole_height_is_8_l3793_379376

/-- The height of the flagpole in meters. -/
def flagpole_height : ℝ := 8

/-- The length of the rope in meters. -/
def rope_length : ℝ := flagpole_height + 2

/-- The distance the rope is pulled away from the flagpole in meters. -/
def pull_distance : ℝ := 6

theorem flagpole_height_is_8 :
  flagpole_height = 8 ∧
  rope_length = flagpole_height + 2 ∧
  flagpole_height ^ 2 + pull_distance ^ 2 = rope_length ^ 2 :=
sorry

end flagpole_height_is_8_l3793_379376


namespace number_of_newborns_l3793_379302

/-- Proves the number of newborns in a children's home --/
theorem number_of_newborns (total_children teenagers toddlers newborns : ℕ) : 
  total_children = 40 →
  teenagers = 5 * toddlers →
  toddlers = 6 →
  total_children = teenagers + toddlers + newborns →
  newborns = 4 := by
  sorry

end number_of_newborns_l3793_379302


namespace monica_savings_l3793_379361

def weekly_savings : ℕ := 15
def weeks_to_fill : ℕ := 60
def repetitions : ℕ := 5

theorem monica_savings : weekly_savings * weeks_to_fill * repetitions = 4500 := by
  sorry

end monica_savings_l3793_379361


namespace multiply_by_seven_l3793_379387

theorem multiply_by_seven (x : ℝ) : 7 * x = 50.68 → x = 7.24 := by
  sorry

end multiply_by_seven_l3793_379387


namespace unique_base_solution_l3793_379332

/-- Convert a base-6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number in base b to decimal --/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The unique positive solution to 35₆ = 151ᵦ --/
theorem unique_base_solution : 
  ∃! (b : ℕ), b > 0 ∧ base6ToDecimal 35 = baseBToDecimal 151 b := by sorry

end unique_base_solution_l3793_379332


namespace equation_solution_l3793_379345

theorem equation_solution : ∃ x : ℚ, (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 ∧ x = -1/6 := by
  sorry

end equation_solution_l3793_379345


namespace geometric_sequence_problem_l3793_379353

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 5 = 1 →
  a 9 = 81 →
  a 7 = 9 := by
sorry

end geometric_sequence_problem_l3793_379353


namespace inequality_implies_a_bound_l3793_379324

theorem inequality_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 := by
sorry

end inequality_implies_a_bound_l3793_379324


namespace pipe_fill_time_l3793_379303

/-- The time it takes for a pipe to fill a tank without a leak, given:
  1. With the leak, it takes 12 hours to fill the tank.
  2. The leak alone can empty the full tank in 12 hours. -/
def fill_time_without_leak : ℝ := 6

/-- The time it takes to fill the tank with both the pipe and leak working -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 12

theorem pipe_fill_time :
  fill_time_without_leak = 6 ∧
  (1 / fill_time_without_leak - 1 / leak_empty_time = 1 / fill_time_with_leak) :=
sorry

end pipe_fill_time_l3793_379303


namespace extreme_value_condition_l3793_379378

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem extreme_value_condition (a b : ℝ) :
  f a b 1 = 10 ∧ f_derivative a b 1 = 0 → a = -4 := by
  sorry

end extreme_value_condition_l3793_379378


namespace polynomial_value_at_n_plus_one_l3793_379382

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the polynomial P
noncomputable def P (n : ℕ) (x : ℚ) : ℚ :=
  sorry  -- The actual definition is not provided in the problem statement

-- State the theorem
theorem polynomial_value_at_n_plus_one (n : ℕ) :
  (∀ k : ℕ, k ≤ n → P n k = 1 / binomial n k) →
  P n (n + 1) = if n % 2 = 0 then 1 else 0 :=
sorry

end polynomial_value_at_n_plus_one_l3793_379382


namespace quadratic_intersects_x_axis_twice_l3793_379326

/-- A quadratic function parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 - (2*k - 1) * x + k

/-- The discriminant of the quadratic function f -/
def discriminant (k : ℝ) : ℝ := (2*k - 1)^2 - 4*k*(k - 2)

/-- The condition for f to have two distinct real roots -/
def has_two_distinct_roots (k : ℝ) : Prop :=
  discriminant k > 0 ∧ k ≠ 2

theorem quadratic_intersects_x_axis_twice (k : ℝ) :
  has_two_distinct_roots k ↔ k > -1/4 ∧ k ≠ 2 := by sorry

end quadratic_intersects_x_axis_twice_l3793_379326


namespace fitness_center_member_ratio_l3793_379357

theorem fitness_center_member_ratio 
  (avg_female : ℝ) 
  (avg_male : ℝ) 
  (avg_total : ℝ) 
  (h1 : avg_female = 140) 
  (h2 : avg_male = 180) 
  (h3 : avg_total = 160) :
  ∃ (f m : ℝ), f > 0 ∧ m > 0 ∧ f / m = 1 ∧
  (f * avg_female + m * avg_male) / (f + m) = avg_total :=
by sorry

end fitness_center_member_ratio_l3793_379357


namespace aquafaba_for_angel_food_cakes_l3793_379355

/-- Proves that the number of tablespoons of aquafaba needed for two angel food cakes is 32 -/
theorem aquafaba_for_angel_food_cakes 
  (aquafaba_per_egg : ℕ) 
  (cakes : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (h1 : aquafaba_per_egg = 2)
  (h2 : cakes = 2)
  (h3 : egg_whites_per_cake = 8) : 
  aquafaba_per_egg * cakes * egg_whites_per_cake = 32 :=
by sorry

end aquafaba_for_angel_food_cakes_l3793_379355


namespace balance_after_transfer_l3793_379315

/-- The initial balance in Christina's bank account before the transfer -/
def initial_balance : ℕ := 27004

/-- The amount Christina transferred out of her account -/
def transferred_amount : ℕ := 69

/-- The remaining balance in Christina's account after the transfer -/
def remaining_balance : ℕ := 26935

/-- Theorem stating that the initial balance minus the transferred amount equals the remaining balance -/
theorem balance_after_transfer : 
  initial_balance - transferred_amount = remaining_balance := by sorry

end balance_after_transfer_l3793_379315


namespace rosies_pies_l3793_379380

/-- Given that Rosie can make 3 pies from 12 apples, this theorem proves
    how many pies she can make from 36 apples. -/
theorem rosies_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_three_pies = 12)
  (h2 : total_apples = 36) :
  (total_apples / apples_per_three_pies) * 3 = 9 :=
sorry

end rosies_pies_l3793_379380


namespace watson_class_size_l3793_379389

/-- The number of students in Ms. Watson's class -/
def total_students (kindergartners first_graders second_graders : ℕ) : ℕ :=
  kindergartners + first_graders + second_graders

/-- Theorem stating the total number of students in Ms. Watson's class -/
theorem watson_class_size :
  total_students 14 24 4 = 42 := by
  sorry

end watson_class_size_l3793_379389


namespace quadratic_b_value_l3793_379320

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x - 1

-- Define the property of passing through two points with the same y-coordinate
def passes_through (b : ℝ) : Prop :=
  ∃ y₀ : ℝ, f b 3 = y₀ ∧ f b 9 = y₀

-- Theorem statement
theorem quadratic_b_value :
  ∀ b : ℝ, passes_through b → b = 24 := by sorry

end quadratic_b_value_l3793_379320


namespace complex_fraction_simplification_l3793_379328

theorem complex_fraction_simplification :
  let z : ℂ := (7 + 8*I) / (3 - 4*I)
  z = 53/25 + 52/25 * I := by
  sorry

end complex_fraction_simplification_l3793_379328


namespace tom_hockey_games_this_year_l3793_379327

/-- The number of hockey games Tom went to this year -/
def games_this_year (total_games : ℕ) (last_year_games : ℕ) : ℕ :=
  total_games - last_year_games

/-- Theorem stating that Tom went to 4 hockey games this year -/
theorem tom_hockey_games_this_year :
  games_this_year 13 9 = 4 := by
  sorry

end tom_hockey_games_this_year_l3793_379327


namespace max_value_of_expression_l3793_379379

def numbers : Finset ℕ := {12, 14, 16, 18}

def expression (A B C D : ℕ) : ℕ := A * B + B * C + B * D + C * D

theorem max_value_of_expression :
  ∃ (A B C D : ℕ), A ∈ numbers ∧ B ∈ numbers ∧ C ∈ numbers ∧ D ∈ numbers ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  expression A B C D = 1116 ∧
  ∀ (A' B' C' D' : ℕ), A' ∈ numbers → B' ∈ numbers → C' ∈ numbers → D' ∈ numbers →
  A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
  expression A' B' C' D' ≤ 1116 :=
by sorry

end max_value_of_expression_l3793_379379


namespace distribute_5_3_l3793_379366

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 243 := by
  sorry

end distribute_5_3_l3793_379366


namespace inequality_subtraction_l3793_379392

theorem inequality_subtraction (a b : ℝ) : a < b → a - b < 0 := by
  sorry

end inequality_subtraction_l3793_379392


namespace smallest_possible_d_l3793_379339

theorem smallest_possible_d : 
  ∀ c d : ℝ, 
  (2 < c) → 
  (c < d) → 
  (2 + c ≤ d) → 
  (1 / c + 1 / d ≤ 2) → 
  (∀ d' : ℝ, 
    (∃ c' : ℝ, (2 < c') ∧ (c' < d') ∧ (2 + c' ≤ d') ∧ (1 / c' + 1 / d' ≤ 2)) → 
    d' ≥ 2 + Real.sqrt 3) → 
  d = 2 + Real.sqrt 3 :=
by sorry

end smallest_possible_d_l3793_379339


namespace orange_painted_cubes_l3793_379373

/-- Represents a cube construction with small cubes -/
structure CubeConstruction where
  small_edge : ℝ
  large_edge : ℝ
  all_sides_painted : Bool

/-- Calculates the number of small cubes with only one side painted -/
def cubes_with_one_side_painted (c : CubeConstruction) : ℕ :=
  sorry

/-- Theorem stating the number of small cubes with one side painted in the given construction -/
theorem orange_painted_cubes (c : CubeConstruction) 
  (h1 : c.small_edge = 2)
  (h2 : c.large_edge = 10)
  (h3 : c.all_sides_painted = true) :
  cubes_with_one_side_painted c = 54 := by
  sorry

end orange_painted_cubes_l3793_379373


namespace square_difference_l3793_379367

theorem square_difference : 100^2 - 2 * 100 * 99 + 99^2 = 1 := by
  sorry

end square_difference_l3793_379367


namespace normal_distribution_probability_l3793_379338

/-- A random variable with normal distribution -/
def normal_dist (μ σ : ℝ) : Type := ℝ

/-- Probability measure for the normal distribution -/
noncomputable def P (ξ : normal_dist (-1) σ) (s : Set ℝ) : ℝ := sorry

/-- The statement of the problem -/
theorem normal_distribution_probability (σ : ℝ) (ξ : normal_dist (-1) σ) 
  (h : P ξ {x | -3 ≤ x ∧ x ≤ -1} = 0.4) : 
  P ξ {x | x ≥ 1} = 0.1 := by sorry

end normal_distribution_probability_l3793_379338


namespace train_platform_passing_time_l3793_379304

/-- Calculates the time for a train to pass a platform given its length, time to cross a tree, and platform length -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 2000)
  (h2 : tree_crossing_time = 80)
  (h3 : platform_length = 1200) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 128 :=
by sorry

end train_platform_passing_time_l3793_379304


namespace jennifer_score_l3793_379346

/-- Calculates the score for a modified AMC 8 contest -/
def calculateScore (totalQuestions correctAnswers incorrectAnswers unansweredQuestions : ℕ) : ℤ :=
  2 * correctAnswers - incorrectAnswers

/-- Proves that Jennifer's score in the modified AMC 8 contest is 20 points -/
theorem jennifer_score :
  let totalQuestions : ℕ := 30
  let correctAnswers : ℕ := 15
  let incorrectAnswers : ℕ := 10
  let unansweredQuestions : ℕ := 5
  calculateScore totalQuestions correctAnswers incorrectAnswers unansweredQuestions = 20 := by
  sorry

#eval calculateScore 30 15 10 5

end jennifer_score_l3793_379346


namespace probability_nine_correct_zero_l3793_379396

/-- Represents a matching problem with n pairs -/
structure MatchingProblem (n : ℕ) where
  /-- The number of pairs to match -/
  pairs : ℕ
  /-- Assumption that the number of pairs is positive -/
  positive : 0 < pairs
  /-- Assumption that the number of pairs is equal to n -/
  eq_n : pairs = n

/-- The probability of randomly matching exactly k pairs correctly in a matching problem with n pairs -/
def probability_exact_match (n k : ℕ) (prob : MatchingProblem n) : ℚ :=
  sorry

/-- Theorem stating that the probability of randomly matching exactly 9 pairs correctly in a matching problem with 10 pairs is 0 -/
theorem probability_nine_correct_zero : 
  ∀ (prob : MatchingProblem 10), probability_exact_match 10 9 prob = 0 :=
sorry

end probability_nine_correct_zero_l3793_379396


namespace santiago_roses_count_l3793_379385

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The number of additional roses Mrs. Santiago has compared to Mrs. Garrett -/
def additional_roses : ℕ := 34

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := garrett_roses + additional_roses

theorem santiago_roses_count : santiago_roses = 58 := by
  sorry

end santiago_roses_count_l3793_379385


namespace three_digit_perfect_cube_divisible_by_four_l3793_379325

theorem three_digit_perfect_cube_divisible_by_four :
  ∃! n : ℕ, 100 ≤ 8 * n^3 ∧ 8 * n^3 ≤ 999 ∧ Even n :=
sorry

end three_digit_perfect_cube_divisible_by_four_l3793_379325


namespace largest_number_problem_l3793_379395

theorem largest_number_problem (a b c d e : ℝ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  a + b = 32 →
  a + c = 36 →
  b + c = 37 →
  c + e = 48 →
  d + e = 51 →
  e = 27.5 := by
  sorry

end largest_number_problem_l3793_379395


namespace weight_of_packet_a_l3793_379391

theorem weight_of_packet_a (a b c d e f : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  f = (a + e) / 2 →
  (b + c + d + e + f) / 5 = 81 →
  a = 75 := by
sorry

end weight_of_packet_a_l3793_379391


namespace sandy_puppies_count_l3793_379301

/-- Given that Sandy initially had 8 puppies and gave away 4,
    prove that she now has 4 puppies. -/
theorem sandy_puppies_count (initial_puppies : Nat) (given_away : Nat) :
  initial_puppies = 8 → given_away = 4 → initial_puppies - given_away = 4 := by
  sorry

end sandy_puppies_count_l3793_379301


namespace sixteen_points_configuration_unique_configuration_l3793_379314

/-- Represents a configuration of points on a line -/
structure LineConfiguration where
  totalPoints : ℕ
  pointA : ℕ
  pointB : ℕ

/-- Counts the number of segments that contain a given point -/
def segmentsContainingPoint (config : LineConfiguration) (point : ℕ) : ℕ :=
  (point - 1) * (config.totalPoints - point)

/-- The main theorem stating that the configuration with 16 points satisfies the given conditions -/
theorem sixteen_points_configuration :
  ∃ (config : LineConfiguration),
    config.totalPoints = 16 ∧
    segmentsContainingPoint config config.pointA = 50 ∧
    segmentsContainingPoint config config.pointB = 56 := by
  sorry

/-- Uniqueness theorem: there is only one configuration satisfying the conditions -/
theorem unique_configuration (config1 config2 : LineConfiguration) :
  segmentsContainingPoint config1 config1.pointA = 50 →
  segmentsContainingPoint config1 config1.pointB = 56 →
  segmentsContainingPoint config2 config2.pointA = 50 →
  segmentsContainingPoint config2 config2.pointB = 56 →
  config1.totalPoints = config2.totalPoints := by
  sorry

end sixteen_points_configuration_unique_configuration_l3793_379314


namespace sheep_count_l3793_379350

/-- Given 3 herds of sheep with 20 sheep in each herd, the total number of sheep is 60. -/
theorem sheep_count (num_herds : ℕ) (sheep_per_herd : ℕ) 
  (h1 : num_herds = 3) 
  (h2 : sheep_per_herd = 20) : 
  num_herds * sheep_per_herd = 60 := by
  sorry

end sheep_count_l3793_379350


namespace shoe_matching_problem_l3793_379340

/-- Represents a collection of shoes -/
structure ShoeCollection :=
  (total_pairs : ℕ)
  (color_count : ℕ)
  (indistinguishable : Bool)

/-- 
Given a collection of shoes, returns the minimum number of shoes
needed to guarantee at least one matching pair of the same color
-/
def minShoesForMatch (collection : ShoeCollection) : ℕ :=
  collection.total_pairs + 1

/-- Theorem statement for the shoe matching problem -/
theorem shoe_matching_problem (collection : ShoeCollection) 
  (h1 : collection.total_pairs = 24)
  (h2 : collection.color_count = 2)
  (h3 : collection.indistinguishable = true) :
  minShoesForMatch collection = 25 := by
  sorry

#check shoe_matching_problem

end shoe_matching_problem_l3793_379340


namespace tenth_student_problems_l3793_379313

theorem tenth_student_problems (n : ℕ) : 
  -- Total number of students
  (10 : ℕ) > 0 →
  -- Each problem is solved by exactly 7 students
  ∃ p : ℕ, p > 0 ∧ (7 * p = 36 + n) →
  -- First 9 students each solved 4 problems
  (9 * 4 = 36) →
  -- The number of problems solved by the tenth student is n
  n ≤ p →
  -- Conclusion: The tenth student solved 6 problems
  n = 6 := by
sorry

end tenth_student_problems_l3793_379313


namespace radio_station_survey_l3793_379331

theorem radio_station_survey (males_dont_listen : ℕ) (females_listen : ℕ) 
  (total_listeners : ℕ) (total_non_listeners : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listeners = 180)
  (h4 : total_non_listeners = 120) :
  total_listeners - females_listen = 105 := by
  sorry

end radio_station_survey_l3793_379331


namespace washer_dryer_cost_washer_dryer_cost_proof_l3793_379370

/-- The total cost of a washer-dryer combination is 1200 dollars, given that the washer costs 710 dollars and is 220 dollars more expensive than the dryer. -/
theorem washer_dryer_cost : ℕ → ℕ → ℕ → Prop :=
  fun washer_cost dryer_cost total_cost =>
    washer_cost = 710 ∧
    washer_cost = dryer_cost + 220 ∧
    total_cost = washer_cost + dryer_cost →
    total_cost = 1200

/-- Proof of the washer-dryer cost theorem -/
theorem washer_dryer_cost_proof : washer_dryer_cost 710 490 1200 := by
  sorry

end washer_dryer_cost_washer_dryer_cost_proof_l3793_379370


namespace sin_225_degrees_l3793_379398

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_225_degrees_l3793_379398


namespace intersection_of_A_and_B_l3793_379393

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by sorry

end intersection_of_A_and_B_l3793_379393


namespace pet_store_problem_l3793_379397

/-- The number of ways to distribute pets among Alice, Bob, and Charlie -/
def pet_distribution_ways (num_puppies num_kittens num_hamsters : ℕ) : ℕ :=
  num_kittens * num_hamsters + num_hamsters * num_kittens

/-- Theorem stating the number of ways Alice, Bob, and Charlie can buy pets -/
theorem pet_store_problem :
  let num_puppies : ℕ := 20
  let num_kittens : ℕ := 4
  let num_hamsters : ℕ := 8
  pet_distribution_ways num_puppies num_kittens num_hamsters = 64 :=
by
  sorry

#eval pet_distribution_ways 20 4 8

end pet_store_problem_l3793_379397


namespace trapezoid_segment_length_l3793_379335

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : ℝ
  area_ratio : ℚ

/-- The properties of the trapezoid as described in the problem -/
axiom trapezoid_properties (t : Trapezoid) :
  t.longer_base = t.shorter_base + t.base_difference ∧
  t.base_difference = 150 ∧
  t.midline_ratio = (t.shorter_base + t.longer_base) / 2 ∧
  t.area_ratio = 3 / 2 ∧
  (t.midline_ratio - t.shorter_base) / (t.longer_base - t.midline_ratio) = t.area_ratio

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) :
  ⌊(t.equal_area_segment ^ 2) / 150⌋ = 550 :=
sorry

end trapezoid_segment_length_l3793_379335


namespace number_problem_l3793_379310

theorem number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 6) (h4 : x / y = 6) :
  x * y - (x - y) = 6 / 49 := by
  sorry

end number_problem_l3793_379310


namespace function_composition_theorem_l3793_379329

/-- Given two functions f and g, with f(x) = Ax - 3B² and g(x) = Bx + C,
    where B ≠ 0 and f(g(1)) = 0, prove that A = 3B² / (B + C),
    assuming B + C ≠ 0. -/
theorem function_composition_theorem (A B C : ℝ) 
  (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x - 3 * B^2
  let g : ℝ → ℝ := λ x ↦ B * x + C
  f (g 1) = 0 → A = 3 * B^2 / (B + C) := by
  sorry

end function_composition_theorem_l3793_379329


namespace saber_toothed_frog_tails_l3793_379394

/-- Represents the number of tadpoles of each type -/
structure TadpoleCount where
  triassic : ℕ
  saber : ℕ

/-- Represents the characteristics of each tadpole type -/
structure TadpoleType where
  legs : ℕ
  tails : ℕ

/-- The main theorem to prove -/
theorem saber_toothed_frog_tails 
  (triassic : TadpoleType)
  (saber : TadpoleType)
  (count : TadpoleCount)
  (h1 : triassic.legs = 5)
  (h2 : triassic.tails = 1)
  (h3 : saber.legs = 4)
  (h4 : count.triassic * triassic.legs + count.saber * saber.legs = 100)
  (h5 : count.triassic * triassic.tails + count.saber * saber.tails = 64) :
  saber.tails = 3 := by
  sorry

end saber_toothed_frog_tails_l3793_379394


namespace set_intersection_implies_values_l3793_379358

theorem set_intersection_implies_values (a b : ℤ) : 
  let A : Set ℤ := {1, b, a + b}
  let B : Set ℤ := {a - b, a * b}
  A ∩ B = {-1, 0} →
  a = -1 ∧ b = 0 := by
sorry

end set_intersection_implies_values_l3793_379358


namespace quadratic_roots_problem_l3793_379333

theorem quadratic_roots_problem (p : ℤ) : 
  (∃ u v : ℤ, u > 0 ∧ v > 0 ∧ 
   5 * u^2 - 5 * p * u + (66 * p - 1) = 0 ∧
   5 * v^2 - 5 * p * v + (66 * p - 1) = 0) →
  p = 76 := by
sorry

end quadratic_roots_problem_l3793_379333


namespace watch_correction_proof_l3793_379371

/-- Represents the time loss of a watch in minutes per day -/
def timeLossPerDay : ℝ := 3

/-- Represents the number of days between April 1 at 12 noon and April 10 at 6 P.M. -/
def daysElapsed : ℝ := 9.25

/-- Calculates the positive correction in minutes for the watch -/
def watchCorrection (loss : ℝ) (days : ℝ) : ℝ := loss * days

theorem watch_correction_proof :
  watchCorrection timeLossPerDay daysElapsed = 27.75 := by
  sorry

end watch_correction_proof_l3793_379371


namespace tennis_ball_box_capacity_l3793_379354

theorem tennis_ball_box_capacity :
  ∀ (total_balls : ℕ) (box_capacity : ℕ),
  (4 * box_capacity - 8 = total_balls) →
  (3 * box_capacity + 4 = total_balls) →
  box_capacity = 12 := by
sorry

end tennis_ball_box_capacity_l3793_379354


namespace compound_proposition_true_l3793_379372

theorem compound_proposition_true : 
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∨ (∀ n : ℝ, ∃ m : ℝ, m^2 < n) :=
by sorry

end compound_proposition_true_l3793_379372


namespace particle_final_position_l3793_379377

/-- Represents the position of a particle -/
structure Position where
  x : Int
  y : Int

/-- Calculates the position of the particle after n steps -/
def particle_position (n : Nat) : Position :=
  sorry

/-- The number of complete rectangles after 2023 minutes -/
def complete_rectangles : Nat :=
  sorry

/-- The remaining time after completing the rectangles -/
def remaining_time : Nat :=
  sorry

theorem particle_final_position :
  particle_position (complete_rectangles + 1) = Position.mk 44 1 :=
sorry

end particle_final_position_l3793_379377


namespace abc12_paths_l3793_379363

/-- Represents the number of adjacent letters or numerals --/
def adjacent_count (letter : Char) : Nat :=
  match letter with
  | 'A' => 2  -- Number of B's adjacent to A
  | 'B' => 3  -- Number of C's adjacent to each B
  | 'C' => 2  -- Number of 1's adjacent to each C
  | '1' => 1  -- Number of 2's adjacent to each 1
  | _   => 0  -- For any other character

/-- Calculates the total number of paths to spell ABC12 --/
def total_paths : Nat :=
  adjacent_count 'A' * adjacent_count 'B' * adjacent_count 'C' * adjacent_count '1'

/-- Theorem stating that the number of paths to spell ABC12 is 12 --/
theorem abc12_paths : total_paths = 12 := by
  sorry

end abc12_paths_l3793_379363


namespace probability_all_truth_l3793_379334

theorem probability_all_truth (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (hpA : pA = 0.8) 
  (hpB : pB = 0.6) 
  (hpC : pC = 0.75) : 
  pA * pB * pC = 0.27 := by
  sorry

end probability_all_truth_l3793_379334


namespace cornbread_pieces_l3793_379316

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 20)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end cornbread_pieces_l3793_379316


namespace elena_snow_removal_l3793_379300

/-- The volume of snow Elena removes from her pathway -/
def snow_volume (length width depth : ℝ) (compaction_factor : ℝ) : ℝ :=
  length * width * depth * compaction_factor

/-- Theorem stating the volume of snow Elena removes -/
theorem elena_snow_removal :
  snow_volume 30 3 0.75 0.9 = 60.75 := by
  sorry

end elena_snow_removal_l3793_379300


namespace valid_word_count_l3793_379384

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := {'B', 'C', 'D'}

def is_valid_word (word : List Char) : Prop :=
  word.length = 5 ∧ word.toFinset ⊆ letters ∧ (∃ c ∈ word, c ∈ consonants)

def count_valid_words : ℕ := (letters.powerset.filter (λ s => s.card = 5)).card

theorem valid_word_count : count_valid_words = 7533 := by
  sorry

end valid_word_count_l3793_379384


namespace function_through_point_l3793_379330

/-- Given a function f(x) = x^α that passes through (2, √2), prove f(9) = 3 -/
theorem function_through_point (α : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x ^ α) → f 2 = Real.sqrt 2 → f 9 = 3 := by
  sorry

end function_through_point_l3793_379330


namespace exists_left_identity_element_l3793_379374

variable {T : Type*} [Fintype T]

def LeftIdentityElement (star : T → T → T) (a : T) : Prop :=
  ∀ b : T, star a b = a

theorem exists_left_identity_element
  (star : T → T → T)
  (assoc : ∀ a b c : T, star (star a b) c = star a (star b c))
  (comm : ∀ a b : T, star a b = star b a) :
  ∃ a : T, LeftIdentityElement star a :=
by
  sorry

end exists_left_identity_element_l3793_379374


namespace train_speed_problem_l3793_379351

/-- Proves that given a train journey of 5x km, where 4x km is traveled at 20 kmph,
    and the average speed for the entire journey is 40/3 kmph,
    the speed for the initial x km is 40/7 kmph. -/
theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let total_distance : ℝ := 5 * x
  let second_leg_distance : ℝ := 4 * x
  let second_leg_speed : ℝ := 20
  let average_speed : ℝ := 40 / 3
  let initial_speed : ℝ := 40 / 7
  (total_distance / (x / initial_speed + second_leg_distance / second_leg_speed) = average_speed) :=
by
  sorry


end train_speed_problem_l3793_379351


namespace family_weight_gain_l3793_379344

/-- The weight gained by Orlando, in pounds -/
def orlando_weight : ℕ := 5

/-- The weight gained by Jose, in pounds -/
def jose_weight : ℕ := 2 * orlando_weight + 2

/-- The weight gained by Fernando, in pounds -/
def fernando_weight : ℕ := jose_weight / 2 - 3

/-- The total weight gained by the three family members, in pounds -/
def total_weight : ℕ := orlando_weight + jose_weight + fernando_weight

theorem family_weight_gain : total_weight = 20 := by
  sorry

end family_weight_gain_l3793_379344


namespace g_pow_6_eq_id_l3793_379375

/-- Definition of the function g -/
def g (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := v
  (a + b, b + c, a + c)

/-- Definition of g^n for n ≥ 2 -/
def g_pow (n : ℕ) : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) :=
  match n with
  | 0 => id
  | 1 => g
  | n + 2 => g ∘ (g_pow (n + 1))

/-- Main theorem -/
theorem g_pow_6_eq_id (v : ℝ × ℝ × ℝ) (h1 : v ≠ (0, 0, 0)) 
    (h2 : ∃ (n : ℕ+), g_pow n v = v) : 
  g_pow 6 v = v := by
  sorry

end g_pow_6_eq_id_l3793_379375


namespace quadratic_equation_two_distinct_roots_l3793_379317

theorem quadratic_equation_two_distinct_roots (k : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  x₁^2 - (k + 3) * x₁ + 2 * k + 1 = 0 ∧
  x₂^2 - (k + 3) * x₂ + 2 * k + 1 = 0 :=
by
  sorry

end quadratic_equation_two_distinct_roots_l3793_379317


namespace smallest_number_1755_more_than_sum_of_digits_l3793_379336

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_1755_more_than_sum_of_digits :
  (∀ m : ℕ, m < 1770 → m ≠ sum_of_digits m + 1755) ∧
  1770 = sum_of_digits 1770 + 1755 :=
sorry

end smallest_number_1755_more_than_sum_of_digits_l3793_379336


namespace equal_cubic_expressions_l3793_379386

theorem equal_cubic_expressions (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 = 3)
  (sum_triple_products_eq : a*b*c + b*c*d + c*d*a + d*a*b = 1) :
  a*(1-a)^3 = b*(1-b)^3 ∧ b*(1-b)^3 = c*(1-c)^3 ∧ c*(1-c)^3 = d*(1-d)^3 := by
  sorry

end equal_cubic_expressions_l3793_379386


namespace meeting_point_theorem_l3793_379352

/-- The distance between point A and point B in kilometers -/
def distance_AB : ℝ := 120

/-- Xiao Zhang's speed in km/h -/
def speed_Zhang : ℝ := 60

/-- Xiao Wang's speed in km/h -/
def speed_Wang : ℝ := 40

/-- Time difference between Xiao Zhang and Xiao Wang's departures in hours -/
def time_difference : ℝ := 1

/-- Total travel time for both Xiao Zhang and Xiao Wang in hours -/
def total_time : ℝ := 4

/-- The meeting point of Xiao Zhang and Xiao Wang in km from point A -/
def meeting_point : ℝ := 96

theorem meeting_point_theorem :
  speed_Zhang * time_difference + 
  (speed_Zhang * speed_Wang / (speed_Zhang + speed_Wang)) * 
  (distance_AB - speed_Zhang * time_difference) = meeting_point :=
sorry

end meeting_point_theorem_l3793_379352


namespace sphere_volume_surface_area_relation_l3793_379383

theorem sphere_volume_surface_area_relation (r : ℝ) : 
  (4 / 3 * Real.pi * r^3) = 2 * (4 * Real.pi * r^2) → r = 6 := by
  sorry

end sphere_volume_surface_area_relation_l3793_379383


namespace sequence_properties_l3793_379369

/-- Definition of the sequence and its partial sum -/
def sequence_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (2 * S n) / n + n = 2 * a n + 1

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

/-- Definition of geometric sequence for three terms -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- Main theorem -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence_condition a S →
  (is_arithmetic_sequence a ∧
   (is_geometric_sequence (a 4) (a 7) (a 9) →
    ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78)) :=
by sorry

end sequence_properties_l3793_379369


namespace count_valid_removal_sequences_for_specific_arrangement_l3793_379308

/-- Represents the arrangement of bricks -/
inductive BrickArrangement
| Empty : BrickArrangement
| Add : BrickArrangement → Nat → BrickArrangement

/-- Checks if a removal sequence is valid for a given arrangement -/
def isValidRemovalSequence (arrangement : BrickArrangement) (sequence : List Nat) : Prop := sorry

/-- Counts the number of valid removal sequences for a given arrangement -/
def countValidRemovalSequences (arrangement : BrickArrangement) : Nat := sorry

/-- The specific arrangement of 6 bricks as described in the problem -/
def specificArrangement : BrickArrangement := sorry

theorem count_valid_removal_sequences_for_specific_arrangement :
  countValidRemovalSequences specificArrangement = 10 := by sorry

end count_valid_removal_sequences_for_specific_arrangement_l3793_379308


namespace range_of_m_l3793_379359

/-- The set of x satisfying the condition p -/
def P : Set ℝ := {x | (x + 2) / (x - 10) ≤ 0}

/-- The set of x satisfying the condition q for a given m -/
def Q (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 < 0}

/-- p is a necessary but not sufficient condition for q -/
def NecessaryNotSufficient (m : ℝ) : Prop :=
  (∀ x, x ∈ Q m → x ∈ P) ∧ (∃ x, x ∈ P ∧ x ∉ Q m)

/-- The main theorem stating the range of m -/
theorem range_of_m :
  ∀ m, m > 0 → (NecessaryNotSufficient m ↔ m < 3) :=
sorry

end range_of_m_l3793_379359


namespace vertex_angle_and_side_not_determine_equilateral_l3793_379307

/-- A triangle with side lengths a, b, c and angles A, B, C (opposite to sides a, b, c respectively) -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- An equilateral triangle is a triangle with all sides equal -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- A vertex angle is any of the three angles in a triangle -/
def IsVertexAngle (t : Triangle) (angle : ℝ) : Prop :=
  angle = t.A ∨ angle = t.B ∨ angle = t.C

/-- Statement: Knowing a vertex angle and a side length is not sufficient to uniquely determine an equilateral triangle -/
theorem vertex_angle_and_side_not_determine_equilateral :
  ∃ (t1 t2 : Triangle) (angle side : ℝ),
    IsVertexAngle t1 angle ∧
    IsVertexAngle t2 angle ∧
    (t1.a = side ∨ t1.b = side ∨ t1.c = side) ∧
    (t2.a = side ∨ t2.b = side ∨ t2.c = side) ∧
    IsEquilateral t1 ∧
    ¬IsEquilateral t2 :=
  sorry

end vertex_angle_and_side_not_determine_equilateral_l3793_379307


namespace equation_solution_l3793_379305

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1/3 ∧ 
  (∀ x : ℝ, (x - 1)^2 + 2*x*(x - 1) = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l3793_379305


namespace abs_one_minus_x_gt_one_solution_set_l3793_379390

theorem abs_one_minus_x_gt_one_solution_set :
  {x : ℝ | |1 - x| > 1} = Set.Ioi 2 ∪ Set.Iic 0 := by sorry

end abs_one_minus_x_gt_one_solution_set_l3793_379390


namespace complex_equation_sum_l3793_379309

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + 2 * i) * i = b + i → a + b = -1 := by
  sorry

end complex_equation_sum_l3793_379309
