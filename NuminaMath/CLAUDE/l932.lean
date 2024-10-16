import Mathlib

namespace NUMINAMATH_CALUDE_dodecagon_diagonals_plus_sides_l932_93261

/-- The number of sides in a regular dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The sum of the number of diagonals and sides in a regular dodecagon is 66 -/
theorem dodecagon_diagonals_plus_sides :
  num_diagonals dodecagon_sides + dodecagon_sides = 66 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_plus_sides_l932_93261


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l932_93231

/-- Given that 14 oranges weigh the same as 10 apples, 
    prove that 42 oranges weigh the same as 30 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
  orange_weight > 0 →
  apple_weight > 0 →
  14 * orange_weight = 10 * apple_weight →
  42 * orange_weight = 30 * apple_weight :=
by
  sorry

#check orange_apple_weight_equivalence

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l932_93231


namespace NUMINAMATH_CALUDE_cube_sum_l932_93213

theorem cube_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l932_93213


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l932_93201

theorem function_not_in_first_quadrant
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b < -1) :
  ∀ x > 0, a^x + b < 0 := by
sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l932_93201


namespace NUMINAMATH_CALUDE_max_min_f_l932_93279

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval
def a : ℝ := 0
def b : ℝ := 3

-- Theorem statement
theorem max_min_f :
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ ∀ (y : ℝ), y ∈ Set.Icc a b → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ ∀ (y : ℝ), y ∈ Set.Icc a b → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ Set.Icc a b → f x ≤ 5) ∧
  (∀ (x : ℝ), x ∈ Set.Icc a b → f x ≥ -15) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = 5) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = -15) :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_l932_93279


namespace NUMINAMATH_CALUDE_unpainted_squares_count_l932_93273

/-- Calculates the number of unpainted squares in a grid strip with a repeating pattern -/
def unpainted_squares (width : ℕ) (length : ℕ) (pattern_width : ℕ) 
  (unpainted_per_pattern : ℕ) (unpainted_remainder : ℕ) : ℕ :=
  let complete_patterns := length / pattern_width
  let remainder_columns := length % pattern_width
  complete_patterns * unpainted_per_pattern + unpainted_remainder

/-- The number of unpainted squares in a 5x250 grid with the given pattern is 812 -/
theorem unpainted_squares_count :
  unpainted_squares 5 250 4 13 6 = 812 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_squares_count_l932_93273


namespace NUMINAMATH_CALUDE_crow_worm_consumption_l932_93295

/-- Given that 3 crows eat 30 worms in one hour, prove that 5 crows will eat 100 worms in 2 hours. -/
theorem crow_worm_consumption (crows_per_hour : ℕ → ℕ → ℕ) : 
  crows_per_hour 3 30 = 1  -- 3 crows eat 30 worms in 1 hour
  → crows_per_hour 5 100 = 2  -- 5 crows eat 100 worms in 2 hours
:= by sorry

end NUMINAMATH_CALUDE_crow_worm_consumption_l932_93295


namespace NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_5_l932_93210

open Complex

theorem abs_z_equals_2_sqrt_5 (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2*I = r)
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) : 
  abs z = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_5_l932_93210


namespace NUMINAMATH_CALUDE_initial_employees_correct_l932_93242

/-- Represents the initial number of employees in a company. -/
def initial_employees : ℕ := 450

/-- Represents the monthly salary of each employee in dollars. -/
def salary_per_employee : ℕ := 2000

/-- Represents the fraction of employees remaining after layoffs. -/
def remaining_fraction : ℚ := 2/3

/-- Represents the total amount paid to remaining employees in dollars. -/
def total_paid : ℕ := 600000

/-- Theorem stating that the initial number of employees is correct given the conditions. -/
theorem initial_employees_correct : 
  (initial_employees : ℚ) * remaining_fraction * salary_per_employee = total_paid :=
sorry

end NUMINAMATH_CALUDE_initial_employees_correct_l932_93242


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l932_93256

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2020 = 2023 := by
sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l932_93256


namespace NUMINAMATH_CALUDE_time_puzzle_l932_93259

theorem time_puzzle : 
  ∃ h : ℝ, h = (12 - h) + (2/5) * h ∧ h = 7.5 := by sorry

end NUMINAMATH_CALUDE_time_puzzle_l932_93259


namespace NUMINAMATH_CALUDE_discriminant_divisibility_l932_93282

theorem discriminant_divisibility (a b : ℝ) (n : ℤ) : 
  (∃ x₁ x₂ : ℝ, (2018 * x₁^2 + a * x₁ + b = 0) ∧ 
                (2018 * x₂^2 + a * x₂ + b = 0) ∧ 
                (x₁ - x₂ = n)) → 
  ∃ k : ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := by
sorry

end NUMINAMATH_CALUDE_discriminant_divisibility_l932_93282


namespace NUMINAMATH_CALUDE_peters_remaining_money_l932_93239

/-- Calculates Peter's remaining money after shopping at the market. -/
def remaining_money (initial_amount : ℕ) (potato_kg : ℕ) (potato_price : ℕ) 
  (tomato_kg : ℕ) (tomato_price : ℕ) (cucumber_kg : ℕ) (cucumber_price : ℕ) 
  (banana_kg : ℕ) (banana_price : ℕ) : ℕ :=
  initial_amount - (potato_kg * potato_price + tomato_kg * tomato_price + 
    cucumber_kg * cucumber_price + banana_kg * banana_price)

/-- Proves that Peter's remaining money after shopping is $426. -/
theorem peters_remaining_money : 
  remaining_money 500 6 2 9 3 5 4 3 5 = 426 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l932_93239


namespace NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l932_93296

/-- Given real numbers x, y, z forming an arithmetic sequence with x ≥ y ≥ z ≥ 0,
    and the quadratic equation yx^2 + zx + y = 0 having exactly one root,
    prove that this root is 4. -/
theorem arithmetic_sequence_quadratic_root :
  ∀ (x y z : ℝ),
  (∃ (d : ℝ), y = x - d ∧ z = x - 2*d) →  -- arithmetic sequence condition
  x ≥ y ∧ y ≥ z ∧ z ≥ 0 →                -- ordering condition
  (∀ r : ℝ, y*r^2 + z*r + y = 0 ↔ r = 4) →  -- unique root condition
  ∀ r : ℝ, y*r^2 + z*r + y = 0 → r = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l932_93296


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l932_93247

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  a ≠ b → 
  a^2 - 3*a - 1 = 0 → 
  b^2 - 3*b - 1 = 0 → 
  b/a + a/b = -11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l932_93247


namespace NUMINAMATH_CALUDE_sin_225_degrees_l932_93284

theorem sin_225_degrees :
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l932_93284


namespace NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l932_93290

/-- Calculates the number of revolutions of the back wheel given the diameters of both wheels and the number of revolutions of the front wheel. -/
theorem bicycle_wheel_revolutions 
  (front_diameter : ℝ) 
  (back_diameter : ℝ) 
  (front_revolutions : ℝ) : 
  front_diameter = 28 →
  back_diameter = 20 →
  front_revolutions = 50 →
  (back_diameter / front_diameter) * front_revolutions = 70 := by
sorry

end NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l932_93290


namespace NUMINAMATH_CALUDE_can_determine_contents_l932_93206

/-- Represents the possible contents of a box -/
inductive BoxContent
  | Red
  | White
  | Mixed

/-- Represents a box with a label and actual content -/
structure Box where
  label : BoxContent
  content : BoxContent

/-- The result of opening a box and drawing a ball -/
inductive DrawResult
  | Red
  | White

/-- Represents the state of the puzzle -/
structure PuzzleState where
  boxes : Fin 3 → Box
  all_labels_incorrect : ∀ i, (boxes i).label ≠ (boxes i).content
  contents_distinct : ∀ i j, i ≠ j → (boxes i).content ≠ (boxes j).content

/-- Function to determine the contents of all boxes based on the draw result -/
def determineContents (state : PuzzleState) (draw : DrawResult) : Fin 3 → BoxContent :=
  sorry

theorem can_determine_contents (state : PuzzleState) :
  ∃ (i : Fin 3) (draw : DrawResult),
    determineContents state draw = λ j => (state.boxes j).content :=
  sorry

end NUMINAMATH_CALUDE_can_determine_contents_l932_93206


namespace NUMINAMATH_CALUDE_x_value_from_fraction_equality_l932_93208

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y + 3) / (y^2 + 2*y + 2) →
  x = y^2 + 2*y + 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_from_fraction_equality_l932_93208


namespace NUMINAMATH_CALUDE_diane_needs_38_cents_l932_93297

/-- The cost of the cookies in cents -/
def cookie_cost : ℕ := 65

/-- The amount Diane has in cents -/
def diane_has : ℕ := 27

/-- The additional amount Diane needs in cents -/
def additional_amount : ℕ := cookie_cost - diane_has

theorem diane_needs_38_cents : additional_amount = 38 := by
  sorry

end NUMINAMATH_CALUDE_diane_needs_38_cents_l932_93297


namespace NUMINAMATH_CALUDE_max_cable_connections_l932_93267

/-- Represents the number of computers of brand A -/
def brand_a_count : Nat := 28

/-- Represents the number of computers of brand B -/
def brand_b_count : Nat := 12

/-- Represents the minimum number of connections required per computer -/
def min_connections : Nat := 2

/-- Theorem stating the maximum number of distinct cable connections -/
theorem max_cable_connections :
  brand_a_count * brand_b_count = 336 ∧
  brand_a_count * brand_b_count ≥ brand_a_count * min_connections ∧
  brand_a_count * brand_b_count ≥ brand_b_count * min_connections :=
sorry

end NUMINAMATH_CALUDE_max_cable_connections_l932_93267


namespace NUMINAMATH_CALUDE_planes_perpendicular_l932_93202

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perp : Line → Line → Prop)
variable (perpLP : Line → Plane → Prop)
variable (perpPP : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (a b : Line) (α β : Plane) 
  (hab : a ≠ b) (hαβ : α ≠ β)
  (h1 : perp a b) 
  (h2 : perpLP a α) 
  (h3 : perpLP b β) : 
  perpPP α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l932_93202


namespace NUMINAMATH_CALUDE_smallest_n_with_right_triangle_l932_93209

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The set S containing numbers from 1 to 50 --/
def S : Finset ℕ := Finset.range 50

/-- A property that checks if a subset of size n always contains a right triangle --/
def hasRightTriangle (n : ℕ) : Prop :=
  ∀ (T : Finset ℕ), T ⊆ S → T.card = n →
    ∃ (a b c : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ isRightTriangle a b c

/-- The main theorem stating that 42 is the smallest n satisfying the property --/
theorem smallest_n_with_right_triangle :
  hasRightTriangle 42 ∧ ∀ m < 42, ¬(hasRightTriangle m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_right_triangle_l932_93209


namespace NUMINAMATH_CALUDE_proposition_equivalence_l932_93225

theorem proposition_equivalence (P : Set α) (a b : α) :
  (a ∈ P → b ∉ P) ↔ (b ∈ P → a ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l932_93225


namespace NUMINAMATH_CALUDE_inequality_proof_l932_93263

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ ≥ a₂) (h2 : a₂ ≥ a₃) (h3 : a₃ > 0)
  (h4 : b₁ ≥ b₂) (h5 : b₂ ≥ b₃) (h6 : b₃ > 0)
  (h7 : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (h8 : a₁ - a₃ ≤ b₁ - b₃) :
  a₁ + a₂ + a₃ ≤ 2 * (b₁ + b₂ + b₃) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l932_93263


namespace NUMINAMATH_CALUDE_solution_set_inequality_l932_93255

theorem solution_set_inequality (x : ℝ) : 
  (x^2 - |x - 1| - 1 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l932_93255


namespace NUMINAMATH_CALUDE_num_men_is_seven_l932_93272

/-- Represents the amount of work a person can do per hour -/
structure WorkRate where
  amount : ℝ

/-- The number of men working with 2 boys -/
def numMen : ℕ := sorry

/-- The work rate of a man -/
def manWorkRate : WorkRate := sorry

/-- The work rate of a boy -/
def boyWorkRate : WorkRate := sorry

/-- The ratio of work done by a man to a boy is 4:1 -/
axiom work_ratio : manWorkRate.amount = 4 * boyWorkRate.amount

/-- The group (numMen men and 2 boys) can do 6 times as much work per hour as a man and a boy together -/
axiom group_work_rate : 
  numMen * manWorkRate.amount + 2 * boyWorkRate.amount = 
  6 * (manWorkRate.amount + boyWorkRate.amount)

theorem num_men_is_seven : numMen = 7 := by sorry

end NUMINAMATH_CALUDE_num_men_is_seven_l932_93272


namespace NUMINAMATH_CALUDE_mean_temperature_l932_93264

def temperatures : List Int := [-8, -6, -3, -3, 0, 4, -1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -17 / 7 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_l932_93264


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l932_93217

/-- Given two similar triangles, where the first triangle has sides of 8, 10, and 12,
    and the second triangle has a perimeter of 150, prove that the longest side
    of the second triangle is 60. -/
theorem similar_triangle_longest_side
  (triangle1 : ℝ × ℝ × ℝ)
  (triangle2 : ℝ × ℝ × ℝ)
  (h_triangle1 : triangle1 = (8, 10, 12))
  (h_similar : ∃ (k : ℝ), triangle2 = (8*k, 10*k, 12*k))
  (h_perimeter : triangle2.1 + triangle2.2.1 + triangle2.2.2 = 150)
  : triangle2.2.2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l932_93217


namespace NUMINAMATH_CALUDE_increasing_quadratic_condition_l932_93238

/-- A function f is increasing on an interval [a, +∞) if for any x₁, x₂ in the interval with x₁ < x₂, we have f(x₁) < f(x₂) -/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂

theorem increasing_quadratic_condition (a : ℝ) :
  (IncreasingOn (fun x => x^2 + 2*(a-1)*x + 2) 4) → a ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_condition_l932_93238


namespace NUMINAMATH_CALUDE_brendan_total_wins_l932_93245

/-- Represents the number of matches won in each round of the kickboxing competition -/
structure KickboxingResults where
  round1_wins : Nat
  round2_wins : Nat
  round3_wins : Nat
  round4_wins : Nat

/-- Calculates the total number of matches won across all rounds -/
def total_wins (results : KickboxingResults) : Nat :=
  results.round1_wins + results.round2_wins + results.round3_wins + results.round4_wins

/-- Theorem stating that Brendan's total wins in the kickboxing competition is 18 -/
theorem brendan_total_wins :
  ∃ (results : KickboxingResults),
    results.round1_wins = 6 ∧
    results.round2_wins = 4 ∧
    results.round3_wins = 3 ∧
    results.round4_wins = 5 ∧
    total_wins results = 18 := by
  sorry

end NUMINAMATH_CALUDE_brendan_total_wins_l932_93245


namespace NUMINAMATH_CALUDE_problem_statement_l932_93248

theorem problem_statement (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3) : 
  (x - z) * (y - w) / ((x - y) * (z - w)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l932_93248


namespace NUMINAMATH_CALUDE_winner_third_difference_l932_93220

/-- Represents the vote count for each candidate in the election. -/
structure ElectionResult where
  total_votes : Nat
  num_candidates : Nat
  winner_votes : Nat
  second_votes : Nat
  third_votes : Nat
  fourth_votes : Nat

/-- Theorem stating the difference between the winner's votes and the third opponent's votes. -/
theorem winner_third_difference (e : ElectionResult) 
  (h1 : e.total_votes = 963)
  (h2 : e.num_candidates = 4)
  (h3 : e.winner_votes = 195)
  (h4 : e.second_votes = 142)
  (h5 : e.third_votes = 116)
  (h6 : e.fourth_votes = 90)
  : e.winner_votes - e.third_votes = 79 := by
  sorry


end NUMINAMATH_CALUDE_winner_third_difference_l932_93220


namespace NUMINAMATH_CALUDE_x_value_in_set_l932_93243

theorem x_value_in_set (x : ℝ) : -2 ∈ ({3, 5, x, x^2 + 3*x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_l932_93243


namespace NUMINAMATH_CALUDE_avg_temp_MTWT_is_48_l932_93212

/-- The average temperature for Monday, Tuesday, Wednesday, and Thursday -/
def avg_temp_MTWT : ℝ := sorry

/-- The average temperature for some days -/
def avg_temp_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def avg_temp_TWTF : ℝ := 40

/-- The temperature on Monday -/
def temp_Monday : ℝ := 42

/-- The temperature on Friday -/
def temp_Friday : ℝ := 10

/-- The theorem stating that the average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem avg_temp_MTWT_is_48 : avg_temp_MTWT = 48 := by sorry

end NUMINAMATH_CALUDE_avg_temp_MTWT_is_48_l932_93212


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l932_93271

/-- The surface area of a sphere circumscribing a right circular cone -/
theorem circumscribed_sphere_surface_area (h : ℝ) (s : ℝ) (π : ℝ) : 
  h = 3 → s = 2 → π = Real.pi → 
  (4 * π * ((s^2 * 3 / 9) + (h^2 / 4))) = (43 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l932_93271


namespace NUMINAMATH_CALUDE_sum_first_tenth_l932_93292

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_4_7 : a 4 + a 7 = 2
  prod_5_6 : a 5 * a 6 = -8

/-- The sum of the first and tenth terms of the geometric sequence is -7 -/
theorem sum_first_tenth (seq : GeometricSequence) : seq.a 1 + seq.a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_tenth_l932_93292


namespace NUMINAMATH_CALUDE_cube_sum_of_cyclic_matrix_cube_is_identity_l932_93281

/-- N is a 3x3 matrix with real entries x, y, z -/
def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x, y, z; y, z, x; z, x, y]

/-- The theorem statement -/
theorem cube_sum_of_cyclic_matrix_cube_is_identity
  (x y z : ℝ) (h1 : N x y z ^ 3 = 1) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_cyclic_matrix_cube_is_identity_l932_93281


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l932_93224

theorem distinct_prime_factors_count (n : ℕ) : n = 87 * 89 * 91 * 93 →
  Finset.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l932_93224


namespace NUMINAMATH_CALUDE_switches_in_A_after_process_l932_93260

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 5

/-- The set of all switches -/
def switches : Finset Switch := sorry

/-- The process of advancing switches for 1000 steps -/
def advance_switches : Finset Switch → Finset Switch := sorry

/-- Counts switches in position A -/
def count_switches_in_A : Finset Switch → Nat := sorry

/-- Main theorem: After 1000 steps, 725 switches are in position A -/
theorem switches_in_A_after_process : 
  count_switches_in_A (advance_switches switches) = 725 := by sorry

end NUMINAMATH_CALUDE_switches_in_A_after_process_l932_93260


namespace NUMINAMATH_CALUDE_oliver_presentation_appropriate_l932_93266

/-- Represents a presentation with a given word count. -/
structure Presentation where
  word_count : ℕ

/-- Checks if a presentation is appropriate given the speaking rate and time constraints. -/
def is_appropriate_presentation (p : Presentation) (speaking_rate : ℕ) (min_time : ℕ) (max_time : ℕ) : Prop :=
  let min_words := speaking_rate * min_time
  let max_words := speaking_rate * max_time
  min_words ≤ p.word_count ∧ p.word_count ≤ max_words

theorem oliver_presentation_appropriate :
  let speaking_rate := 120
  let min_time := 40
  let max_time := 55
  let presentation1 := Presentation.mk 5000
  let presentation2 := Presentation.mk 6200
  is_appropriate_presentation presentation1 speaking_rate min_time max_time ∧
  is_appropriate_presentation presentation2 speaking_rate min_time max_time :=
by sorry

end NUMINAMATH_CALUDE_oliver_presentation_appropriate_l932_93266


namespace NUMINAMATH_CALUDE_hyperbola_tangent_coincidence_l932_93299

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define the curve
def curve (a x y : ℝ) : Prop := y = a * x^2 + 1/3

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Define the condition that asymptotes coincide with tangents
def coincide_with_tangents (a : ℝ) : Prop :=
  ∀ x y : ℝ, asymptotes x y → ∃ t : ℝ, curve a t y ∧ 
  (∀ s : ℝ, s ≠ t → curve a s (a * s^2 + 1/3) → (a * s^2 + 1/3 - y) * (s - t) > 0)

-- Theorem statement
theorem hyperbola_tangent_coincidence :
  ∀ a : ℝ, coincide_with_tangents a → a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_coincidence_l932_93299


namespace NUMINAMATH_CALUDE_integer_between_sqrt_11_and_sqrt_19_l932_93229

theorem integer_between_sqrt_11_and_sqrt_19 :
  ∃! x : ℤ, Real.sqrt 11 < x ∧ x < Real.sqrt 19 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_11_and_sqrt_19_l932_93229


namespace NUMINAMATH_CALUDE_cricket_team_size_l932_93288

theorem cricket_team_size :
  ∀ (n : ℕ),
  let captain_age : ℕ := 24
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 21
  let remaining_players_average_age : ℕ := team_average_age - 1
  (n : ℝ) * team_average_age = 
    (n - 2 : ℝ) * remaining_players_average_age + captain_age + wicket_keeper_age →
  n = 11 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l932_93288


namespace NUMINAMATH_CALUDE_min_value_expression_l932_93226

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  ∃ (min : ℝ), min = 1 ∧ ∀ z, z = 2*x^2 + 3*y^2 - 4*x*y → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l932_93226


namespace NUMINAMATH_CALUDE_min_value_of_xy_l932_93291

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1/2) :
  x * y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_xy_l932_93291


namespace NUMINAMATH_CALUDE_linear_system_solution_l932_93205

theorem linear_system_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * ((c₁ * b₂ - c₂ * b₁) / (a₁ * b₂ - a₂ * b₁)) + b₁ * y = c₁)
  (h₂ : a₂ * ((c₁ * b₂ - c₂ * b₁) / (a₁ * b₂ - a₂ * b₁)) + b₂ * y = c₂)
  (h₃ : a₁ * b₂ ≠ a₂ * b₁) :
  y = (c₁ * a₂ - c₂ * a₁) / (b₁ * a₂ - b₂ * a₁) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l932_93205


namespace NUMINAMATH_CALUDE_quadratic_function_point_l932_93265

theorem quadratic_function_point (a b m t : ℝ) : 
  a ≠ 0 →
  (∀ x, a * x^2 - b * x = 2 → x = m) →
  (∀ x, a * x^2 - b * x ≥ -1 → (x ≤ t - 1 ∨ x ≥ -3 - t)) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_point_l932_93265


namespace NUMINAMATH_CALUDE_toby_change_is_seven_l932_93287

/-- Represents the cost of a meal for two people -/
structure MealCost where
  cheeseburger_price : ℚ
  milkshake_price : ℚ
  coke_price : ℚ
  fries_price : ℚ
  cookie_price : ℚ
  cookie_quantity : ℕ
  tax : ℚ

/-- Calculates the change Toby brings home after splitting the bill -/
def toby_change (meal : MealCost) (toby_initial_amount : ℚ) : ℚ :=
  let total_cost := 2 * meal.cheeseburger_price + meal.milkshake_price + meal.coke_price +
                    meal.fries_price + meal.cookie_price * meal.cookie_quantity + meal.tax
  let toby_share := total_cost / 2
  toby_initial_amount - toby_share

/-- Theorem stating that Toby's change is $7 given the specific meal costs -/
theorem toby_change_is_seven :
  let meal := MealCost.mk 3.65 2 1 4 0.5 3 0.2
  toby_change meal 15 = 7 := by sorry


end NUMINAMATH_CALUDE_toby_change_is_seven_l932_93287


namespace NUMINAMATH_CALUDE_light_glow_interval_l932_93286

def seconds_past_midnight (hours minutes seconds : ℕ) : ℕ :=
  hours * 3600 + minutes * 60 + seconds

def start_time : ℕ := seconds_past_midnight 1 57 58
def end_time : ℕ := seconds_past_midnight 3 20 47
def num_glows : ℝ := 354.92857142857144

theorem light_glow_interval :
  let total_time : ℕ := end_time - start_time
  let interval : ℝ := (total_time : ℝ) / num_glows
  ⌊interval⌋ = 14 := by sorry

end NUMINAMATH_CALUDE_light_glow_interval_l932_93286


namespace NUMINAMATH_CALUDE_complex_number_subtraction_l932_93219

theorem complex_number_subtraction (i : ℂ) (h : i * i = -1) :
  (7 - 3 * i) - 3 * (2 + 5 * i) = 1 - 18 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_number_subtraction_l932_93219


namespace NUMINAMATH_CALUDE_exponential_is_self_derivative_l932_93235

theorem exponential_is_self_derivative : 
  ∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x) ∧ (∀ x, deriv f x = f x) :=
sorry

end NUMINAMATH_CALUDE_exponential_is_self_derivative_l932_93235


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l932_93214

theorem quadratic_equation_1 (x : ℝ) : x^2 + 16 = 8*x → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l932_93214


namespace NUMINAMATH_CALUDE_smallest_exponent_sum_l932_93244

theorem smallest_exponent_sum (p q r s : ℕ+) 
  (h_eq : (3^(p:ℕ))^2 + (3^(q:ℕ))^3 + (3^(r:ℕ))^5 = (3^(s:ℕ))^7) : 
  (p:ℕ) + q + r + s ≥ 106 := by
  sorry

end NUMINAMATH_CALUDE_smallest_exponent_sum_l932_93244


namespace NUMINAMATH_CALUDE_y_divisibility_l932_93268

def y : ℕ := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem y_divisibility : 
  (∃ k : ℕ, y = 5 * k) ∧ 
  (∃ k : ℕ, y = 10 * k) ∧ 
  (∃ k : ℕ, y = 20 * k) ∧ 
  (∃ k : ℕ, y = 40 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l932_93268


namespace NUMINAMATH_CALUDE_sams_football_games_l932_93207

/-- Given that Sam went to 14 football games this year and 43 games in total,
    prove that he went to 29 games last year. -/
theorem sams_football_games (games_this_year games_total : ℕ) 
    (h1 : games_this_year = 14)
    (h2 : games_total = 43) :
    games_total - games_this_year = 29 := by
  sorry

end NUMINAMATH_CALUDE_sams_football_games_l932_93207


namespace NUMINAMATH_CALUDE_product_of_square_roots_l932_93262

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (25 * p^2) * Real.sqrt (2 * p^5) = 25 * p^5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l932_93262


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l932_93251

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 a2 b2 : ℝ) : Prop := a1 / b1 = a2 / b2

/-- Definition of the first line l1 -/
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m

/-- Definition of the second line l2 -/
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

/-- Additional condition for m -/
def additional_condition (m : ℝ) : Prop := (3 + m) / 2 ≠ (5 - 3 * m) / 8

theorem parallel_lines_m_value :
  ∃ (m : ℝ), parallel_lines (3 + m) 4 2 (5 + m) ∧ 
             additional_condition m ∧
             m = -7 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l932_93251


namespace NUMINAMATH_CALUDE_circular_track_circumference_l932_93257

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem circular_track_circumference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (meeting_time_minutes : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 16) 
  (h3 : meeting_time_minutes = 36) : 
  speed1 + speed2 * meeting_time_minutes / 60 = 21.6 := by
  sorry

#check circular_track_circumference

end NUMINAMATH_CALUDE_circular_track_circumference_l932_93257


namespace NUMINAMATH_CALUDE_four_roots_implies_a_in_open_interval_l932_93285

def f (x : ℝ) : ℝ := |x^2 + x - 2|

theorem four_roots_implies_a_in_open_interval (a : ℝ) :
  (∃ (w x y z : ℝ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f w - a * |w - 2| = 0 ∧
    f x - a * |x - 2| = 0 ∧
    f y - a * |y - 2| = 0 ∧
    f z - a * |z - 2| = 0 ∧
    (∀ t : ℝ, f t - a * |t - 2| = 0 → t = w ∨ t = x ∨ t = y ∨ t = z)) →
  0 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_four_roots_implies_a_in_open_interval_l932_93285


namespace NUMINAMATH_CALUDE_convergence_of_difference_series_l932_93252

/-- Given two real sequences (a_i) and (b_i) where the series of their squares converge,
    prove that the series of |a_i - b_i|^p converges for all p ≥ 2. -/
theorem convergence_of_difference_series
  (a b : ℕ → ℝ)
  (ha : Summable (λ i => (a i)^2))
  (hb : Summable (λ i => (b i)^2))
  (p : ℝ)
  (hp : p ≥ 2) :
  Summable (λ i => |a i - b i|^p) :=
sorry

end NUMINAMATH_CALUDE_convergence_of_difference_series_l932_93252


namespace NUMINAMATH_CALUDE_distance_and_angle_from_origin_l932_93215

/-- In a rectangular coordinate system, for a point (12, 5): -/
theorem distance_and_angle_from_origin :
  let x : ℝ := 12
  let y : ℝ := 5
  let distance := Real.sqrt (x^2 + y^2)
  let angle := Real.arctan (y / x)
  (distance = 13 ∧ angle = Real.arctan (5 / 12)) := by
  sorry

end NUMINAMATH_CALUDE_distance_and_angle_from_origin_l932_93215


namespace NUMINAMATH_CALUDE_f_inequality_l932_93280

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x ≠ 1, (x - 1) * (deriv f x) < 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem f_inequality (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l932_93280


namespace NUMINAMATH_CALUDE_triangle_inequality_l932_93222

/-- Given a triangle ABC with sides a ≤ b ≤ c, angle bisectors l_a, l_b, l_c,
    and corresponding medians m_a, m_b, m_c, prove that
    h_n/m_n + h_n/m_h_n + l_c/m_m_p > 1 -/
theorem triangle_inequality (a b c : ℝ) (h_sides : 0 < a ∧ a ≤ b ∧ b ≤ c)
  (l_a l_b l_c : ℝ) (h_bisectors : l_a > 0 ∧ l_b > 0 ∧ l_c > 0)
  (m_a m_b m_c : ℝ) (h_medians : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_n m_n m_h_n m_m_p : ℝ) (h_positive : h_n > 0 ∧ m_n > 0 ∧ m_h_n > 0 ∧ m_m_p > 0) :
  h_n / m_n + h_n / m_h_n + l_c / m_m_p > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l932_93222


namespace NUMINAMATH_CALUDE_bus_driver_overtime_pay_increase_l932_93258

/-- Calculates the percentage increase in overtime pay rate for a bus driver -/
theorem bus_driver_overtime_pay_increase 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) : 
  regular_rate = 16 →
  regular_hours = 40 →
  total_compensation = 920 →
  total_hours = 50 →
  ((total_compensation - regular_rate * regular_hours) / (total_hours - regular_hours) - regular_rate) / regular_rate * 100 = 75 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_overtime_pay_increase_l932_93258


namespace NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l932_93227

theorem infinitely_many_prime_divisors 
  (a b c d : ℕ+) 
  (ha : a ≠ b ∧ a ≠ c ∧ a ≠ d) 
  (hb : b ≠ c ∧ b ≠ d) 
  (hc : c ≠ d) : 
  ∃ (s : Set ℕ), Set.Infinite s ∧ 
  (∀ p ∈ s, Prime p ∧ ∃ n : ℕ, p ∣ (a * c^n + b * d^n)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l932_93227


namespace NUMINAMATH_CALUDE_stating_paint_usage_calculation_l932_93278

/-- 
Given an initial amount of paint and usage fractions for two weeks,
calculate the total amount of paint used.
-/
def paint_used (initial_paint : ℝ) (week1_fraction : ℝ) (week2_fraction : ℝ) : ℝ :=
  let week1_usage := initial_paint * week1_fraction
  let remaining_paint := initial_paint - week1_usage
  let week2_usage := remaining_paint * week2_fraction
  week1_usage + week2_usage

/-- 
Theorem stating that given 360 gallons of initial paint, 
using 1/4 of all paint in the first week and 1/6 of the remaining paint 
in the second week results in a total usage of 135 gallons of paint.
-/
theorem paint_usage_calculation :
  paint_used 360 (1/4) (1/6) = 135 := by
  sorry


end NUMINAMATH_CALUDE_stating_paint_usage_calculation_l932_93278


namespace NUMINAMATH_CALUDE_inequality_proof_l932_93249

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l932_93249


namespace NUMINAMATH_CALUDE_vacant_seats_l932_93203

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 75 / 100) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l932_93203


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l932_93254

/-- Given f(x) = ln x - a, if f(x) < x^2 holds for all x > 1, then a ≥ -1 -/
theorem function_inequality_implies_parameter_range (a : ℝ) :
  (∀ x > 1, Real.log x - a < x^2) →
  a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l932_93254


namespace NUMINAMATH_CALUDE_marbles_given_correct_l932_93228

/-- The number of marbles Tyrone gave to Eric -/
def marbles_given : ℕ := sorry

/-- Tyrone's initial number of marbles -/
def tyrone_initial : ℕ := 150

/-- Eric's initial number of marbles -/
def eric_initial : ℕ := 18

/-- Theorem stating the number of marbles Tyrone gave to Eric -/
theorem marbles_given_correct : 
  marbles_given = 24 ∧
  tyrone_initial - marbles_given = 3 * (eric_initial + marbles_given) :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_correct_l932_93228


namespace NUMINAMATH_CALUDE_angle_c_in_triangle_l932_93200

/-- In a triangle ABC, if the sum of angles A and B is 80°, then angle C is 100°. -/
theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_in_triangle_l932_93200


namespace NUMINAMATH_CALUDE_statements_b_and_c_correct_l932_93253

theorem statements_b_and_c_correct :
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b : ℝ), a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) :=
by sorry

end NUMINAMATH_CALUDE_statements_b_and_c_correct_l932_93253


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l932_93234

theorem binomial_coefficient_equality (x : ℕ) : 
  Nat.choose 20 (3 * x) = Nat.choose 20 (x + 4) → x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l932_93234


namespace NUMINAMATH_CALUDE_sqrt_real_iff_nonneg_l932_93223

theorem sqrt_real_iff_nonneg (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_iff_nonneg_l932_93223


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l932_93240

/-- A positive geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 3 →
  a 3 + a 4 = 12 →
  a 4 + a 5 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l932_93240


namespace NUMINAMATH_CALUDE_max_distance_product_l932_93270

/-- Fixed point A -/
def A : ℝ × ℝ := (0, 0)

/-- Fixed point B -/
def B : ℝ × ℝ := (1, 3)

/-- Line through A -/
def line_A (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0

/-- Line through B -/
def line_B (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

/-- Intersection point P -/
def P (m : ℝ) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Product of distances PA and PB -/
def distance_product (m : ℝ) : ℝ := distance (P m) A * distance (P m) B

/-- Theorem: Maximum value of |PA| * |PB| is 5 -/
theorem max_distance_product : 
  ∃ (m : ℝ), ∀ (n : ℝ), distance_product n ≤ distance_product m ∧ distance_product m = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_product_l932_93270


namespace NUMINAMATH_CALUDE_average_not_equal_given_l932_93246

def numbers : List ℝ := [1200, 1300, 1400, 1510, 1530, 1200]
def given_average : ℝ := 1380

theorem average_not_equal_given : (numbers.sum / numbers.length) ≠ given_average := by
  sorry

end NUMINAMATH_CALUDE_average_not_equal_given_l932_93246


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l932_93277

theorem diophantine_equation_solutions (x y : ℕ) : 
  2^(2*x + 1) + 2^x + 1 = y^2 ↔ (x = 4 ∧ y = 23) ∨ (x = 0 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l932_93277


namespace NUMINAMATH_CALUDE_triangle_proof_l932_93218

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The median from vertex A to side BC -/
def median (t : Triangle) : ℝ := sorry

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

theorem triangle_proof (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.A - Real.sqrt 3 * t.c * Real.cos t.A = Real.sqrt 3 * t.a * Real.cos t.C)
  (h2 : t.B = π / 6)
  (h3 : median t = Real.sqrt 7) :
  t.A = π / 6 ∧ area t = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_proof_l932_93218


namespace NUMINAMATH_CALUDE_N_properties_l932_93241

def N : ℕ := 2^2022 + 1

theorem N_properties :
  (∃ k : ℕ, N = 65 * k) ∧
  (∃ a b c d : ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ N = a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_N_properties_l932_93241


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l932_93276

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 120 ∧ 
  n % 8 = 5 ∧ 
  ∀ m : ℕ, m < 120 ∧ m % 8 = 5 → m ≤ n → 
  n = 117 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l932_93276


namespace NUMINAMATH_CALUDE_athletes_simultaneous_return_l932_93211

/-- The time in minutes for Athlete A to complete one lap -/
def timeA : ℕ := 4

/-- The time in minutes for Athlete B to complete one lap -/
def timeB : ℕ := 5

/-- The time in minutes for Athlete C to complete one lap -/
def timeC : ℕ := 6

/-- The length of the circular track in meters -/
def trackLength : ℕ := 1000

/-- The time when all athletes simultaneously return to the starting point -/
def simultaneousReturnTime : ℕ := 60

theorem athletes_simultaneous_return :
  Nat.lcm (Nat.lcm timeA timeB) timeC = simultaneousReturnTime :=
sorry

end NUMINAMATH_CALUDE_athletes_simultaneous_return_l932_93211


namespace NUMINAMATH_CALUDE_female_athletes_in_sample_l932_93221

/-- Calculates the number of female athletes in a stratified sample -/
def femaleAthletesSample (totalAthletes maleAthletes femaleAthletes sampleSize : ℕ) : ℕ :=
  (femaleAthletes * sampleSize) / totalAthletes

/-- Theorem stating the number of female athletes in the sample -/
theorem female_athletes_in_sample :
  femaleAthletesSample 84 48 36 21 = 9 := by
  sorry

#eval femaleAthletesSample 84 48 36 21

end NUMINAMATH_CALUDE_female_athletes_in_sample_l932_93221


namespace NUMINAMATH_CALUDE_set_relationship_l932_93236

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem set_relationship :
  (¬(B (1/5) ⊆ A)) ∧
  (∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5)) :=
sorry

end NUMINAMATH_CALUDE_set_relationship_l932_93236


namespace NUMINAMATH_CALUDE_less_number_proof_l932_93283

theorem less_number_proof (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 96) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_less_number_proof_l932_93283


namespace NUMINAMATH_CALUDE_function_composition_property_l932_93275

theorem function_composition_property (n : ℕ) :
  (∃ (f g : Fin n → Fin n), ∀ i : Fin n, 
    (f (g i) = i ∧ g (f i) ≠ i) ∨ (g (f i) = i ∧ f (g i) ≠ i)) ↔ 
  Even n :=
by sorry

end NUMINAMATH_CALUDE_function_composition_property_l932_93275


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l932_93230

-- Define the constants for the cylinder
def cylinder_height : ℝ := 10
def cylinder_diameter : ℝ := 10

-- Define the theorem
theorem sphere_cylinder_equal_area (r : ℝ) :
  (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) →
  r = 5 := by
  sorry


end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l932_93230


namespace NUMINAMATH_CALUDE_polynomial_factorization_l932_93293

theorem polynomial_factorization (a x y : ℝ) :
  3 * a * x^2 - 3 * a * y^2 = 3 * a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l932_93293


namespace NUMINAMATH_CALUDE_tower_count_mod_1000_l932_93237

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 1 => 4 * T n

/-- The main theorem stating that the number of towers with 9 cubes is congruent to 768 mod 1000 -/
theorem tower_count_mod_1000 : T 9 ≡ 768 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_tower_count_mod_1000_l932_93237


namespace NUMINAMATH_CALUDE_security_system_probability_l932_93289

theorem security_system_probability (p : ℝ) : 
  (1/8 : ℝ) * (1 - p) + (1 - 1/8) * p = 9/40 → p = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_security_system_probability_l932_93289


namespace NUMINAMATH_CALUDE_correct_answer_l932_93232

theorem correct_answer (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l932_93232


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l932_93204

/-- A hyperbola with foci at (-1,0) and (1,0) -/
structure Hyperbola where
  leftFocus : ℝ × ℝ := (-1, 0)
  rightFocus : ℝ × ℝ := (1, 0)

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point a to point b -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

theorem hyperbola_eccentricity (C : Hyperbola) (P : ℝ × ℝ) :
  parabola P.1 P.2 →
  let F₁ := C.leftFocus
  let F₂ := C.rightFocus
  dot_product (vector F₂ P + vector F₂ F₁) (vector F₂ P - vector F₂ F₁) = 0 →
  eccentricity C = 1 + Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l932_93204


namespace NUMINAMATH_CALUDE_infinite_divisible_sequence_l932_93233

theorem infinite_divisible_sequence : 
  ∃ (f : ℕ → ℕ), 
    (∀ k, f k > 0) ∧ 
    (∀ k, k < k.succ → f k < f k.succ) ∧ 
    (∀ k, (2 ^ (f k) + 3 ^ (f k)) % (f k)^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_infinite_divisible_sequence_l932_93233


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_range_l932_93269

/-- The function f(x) = x^3 - 3x has a minimum value on the interval (a, 6-a^2) -/
def has_minimum_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a (6 - a^2), ∀ y ∈ Set.Ioo a (6 - a^2), f x ≤ f y

/-- The main theorem -/
theorem minimum_value_implies_a_range (a : ℝ) :
  has_minimum_on_interval (fun x => x^3 - 3*x) a → a ∈ Set.Icc (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_range_l932_93269


namespace NUMINAMATH_CALUDE_bowl_delivery_fee_l932_93294

/-- The problem of calculating the initial fee for a bowl delivery service -/
theorem bowl_delivery_fee
  (total_bowls : ℕ)
  (safe_delivery_pay : ℕ)
  (loss_penalty : ℕ)
  (lost_bowls : ℕ)
  (broken_bowls : ℕ)
  (total_payment : ℕ)
  (h1 : total_bowls = 638)
  (h2 : safe_delivery_pay = 3)
  (h3 : loss_penalty = 4)
  (h4 : lost_bowls = 12)
  (h5 : broken_bowls = 15)
  (h6 : total_payment = 1825) :
  ∃ (initial_fee : ℕ),
    initial_fee = 100 ∧
    total_payment = initial_fee +
      (total_bowls - lost_bowls - broken_bowls) * safe_delivery_pay -
      (lost_bowls + broken_bowls) * loss_penalty :=
by sorry

end NUMINAMATH_CALUDE_bowl_delivery_fee_l932_93294


namespace NUMINAMATH_CALUDE_fraction_of_week_worked_l932_93274

/-- Proves that given a usual work week of 40 hours, an hourly rate of $15, and a weekly salary of $480, the fraction of the usual week worked is 4/5. -/
theorem fraction_of_week_worked 
  (usual_hours : ℕ) 
  (hourly_rate : ℚ) 
  (weekly_salary : ℚ) 
  (h1 : usual_hours = 40)
  (h2 : hourly_rate = 15)
  (h3 : weekly_salary = 480) :
  (weekly_salary / hourly_rate) / usual_hours = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_week_worked_l932_93274


namespace NUMINAMATH_CALUDE_jenny_sold_192_packs_l932_93298

-- Define the number of boxes sold
def boxes_sold : Float := 24.0

-- Define the number of packs per box
def packs_per_box : Float := 8.0

-- Define the total number of packs sold
def total_packs : Float := boxes_sold * packs_per_box

-- Theorem statement
theorem jenny_sold_192_packs : total_packs = 192.0 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sold_192_packs_l932_93298


namespace NUMINAMATH_CALUDE_isosceles_triangle_l932_93216

theorem isosceles_triangle (A B C : ℝ) (h_sum : A + B + C = π) :
  let f := fun x : ℝ => x^2 - x * Real.cos A * Real.cos B + 2 * Real.sin (C/2)^2
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = (1/2) * x₁ * x₂ → A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l932_93216


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l932_93250

theorem sum_of_coefficients_zero 
  (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ : ℝ) :
  (∀ x : ℝ, (1 + x - x^2)^3 * (1 - 2*x^2)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ + a₁₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l932_93250
