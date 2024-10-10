import Mathlib

namespace fiona_casey_hoodies_l1193_119393

/-- The number of hoodies Fiona and Casey own together -/
def total_hoodies (fiona_hoodies : ℕ) (casey_extra_hoodies : ℕ) : ℕ :=
  fiona_hoodies + (fiona_hoodies + casey_extra_hoodies)

/-- Theorem stating that Fiona and Casey own 8 hoodies in total -/
theorem fiona_casey_hoodies : total_hoodies 3 2 = 8 := by
  sorry

end fiona_casey_hoodies_l1193_119393


namespace linear_function_increasing_condition_l1193_119341

/-- A linear function y = (2m-1)x + 1 -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (2*m - 1)*x + 1

theorem linear_function_increasing_condition 
  (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : y₁ < y₂)
  (h3 : y₁ = linear_function m x₁)
  (h4 : y₂ = linear_function m x₂) :
  m > 1/2 := by
  sorry

end linear_function_increasing_condition_l1193_119341


namespace todd_snow_cone_profit_l1193_119336

/-- Calculates Todd's profit from his snow-cone stand business --/
theorem todd_snow_cone_profit :
  let loan := 300
  let repayment := 330
  let equipment_cost := 120
  let initial_ingredient_cost := 60
  let marketing_cost := 40
  let misc_cost := 10
  let snow_cone_price := 1.75
  let snow_cone_sales := 500
  let custom_cup_price := 2
  let custom_cup_sales := 250
  let ingredient_cost_increase_rate := 0.2
  let snow_cones_before_increase := 300

  let total_initial_expenses := equipment_cost + initial_ingredient_cost + marketing_cost + misc_cost + repayment
  let total_revenue := snow_cone_price * snow_cone_sales + custom_cup_price * custom_cup_sales
  let increased_ingredient_cost := initial_ingredient_cost * ingredient_cost_increase_rate
  let snow_cones_after_increase := snow_cone_sales - snow_cones_before_increase
  let total_expenses := total_initial_expenses + increased_ingredient_cost

  let profit := total_revenue - total_expenses

  profit = 803 := by sorry

end todd_snow_cone_profit_l1193_119336


namespace wicket_keeper_age_l1193_119306

theorem wicket_keeper_age (team_size : ℕ) (team_avg_age : ℝ) (remaining_avg_age : ℝ) : 
  team_size = 11 →
  team_avg_age = 24 →
  remaining_avg_age = team_avg_age - 1 →
  ∃ (wicket_keeper_age : ℝ),
    wicket_keeper_age = team_avg_age + 9 ∧
    (team_size - 2) * remaining_avg_age + wicket_keeper_age + team_avg_age = team_size * team_avg_age :=
by sorry

end wicket_keeper_age_l1193_119306


namespace division_multiplication_equivalence_l1193_119340

theorem division_multiplication_equivalence : 
  (5.8 / 0.001) = (5.8 * 1000) := by sorry

end division_multiplication_equivalence_l1193_119340


namespace coefficient_abc_in_expansion_coefficient_of_ab2c3_l1193_119313

theorem coefficient_abc_in_expansion : ℕ → Prop :=
  fun n => (1 + 1 + 1)^6 = n + sorry

theorem coefficient_of_ab2c3 : coefficient_abc_in_expansion 60 := by
  sorry

end coefficient_abc_in_expansion_coefficient_of_ab2c3_l1193_119313


namespace diamond_5_20_l1193_119394

-- Define the diamond operation
noncomputable def diamond (x y : ℝ) : ℝ := sorry

-- Axioms for the diamond operation
axiom diamond_positive (x y : ℝ) : x > 0 → y > 0 → diamond x y > 0
axiom diamond_eq1 (x y : ℝ) : x > 0 → y > 0 → diamond (x * y) y = x * diamond y y
axiom diamond_eq2 (x : ℝ) : x > 0 → diamond (diamond x 2) x = diamond x 2
axiom diamond_2_2 : diamond 2 2 = 4

-- Theorem to prove
theorem diamond_5_20 : diamond 5 20 = 20 := by sorry

end diamond_5_20_l1193_119394


namespace reflection_equivalence_l1193_119333

-- Define the shape type
inductive Shape
  | OriginalL
  | InvertedL
  | UpsideDownRotatedL
  | VerticallyFlippedL
  | HorizontallyMirroredL
  | UnalteredL

-- Define the reflection operation
def reflectAcrossDiagonal (s : Shape) : Shape :=
  match s with
  | Shape.OriginalL => Shape.HorizontallyMirroredL
  | _ => s  -- For completeness, though we only care about OriginalL

-- State the theorem
theorem reflection_equivalence :
  reflectAcrossDiagonal Shape.OriginalL = Shape.HorizontallyMirroredL :=
by sorry

end reflection_equivalence_l1193_119333


namespace convex_polygon_covered_by_three_similar_l1193_119349

/-- A planar convex polygon. -/
structure PlanarConvexPolygon where
  -- Add necessary fields and properties here
  -- This is a placeholder definition

/-- Similarity between two planar convex polygons. -/
def IsSimilar (P Q : PlanarConvexPolygon) : Prop :=
  -- Define similarity condition here
  sorry

/-- One polygon covers another. -/
def Covers (P Q : PlanarConvexPolygon) : Prop :=
  -- Define covering condition here
  sorry

/-- Union of three polygons. -/
def Union3 (P Q R : PlanarConvexPolygon) : PlanarConvexPolygon :=
  -- Define union operation here
  sorry

/-- A polygon is smaller than another. -/
def IsSmaller (P Q : PlanarConvexPolygon) : Prop :=
  -- Define size comparison here
  sorry

/-- Theorem: Every planar convex polygon can be covered by three smaller similar polygons. -/
theorem convex_polygon_covered_by_three_similar :
  ∀ (M : PlanarConvexPolygon),
  ∃ (N₁ N₂ N₃ : PlanarConvexPolygon),
    IsSimilar N₁ M ∧ IsSimilar N₂ M ∧ IsSimilar N₃ M ∧
    IsSmaller N₁ M ∧ IsSmaller N₂ M ∧ IsSmaller N₃ M ∧
    Covers (Union3 N₁ N₂ N₃) M :=
by sorry

end convex_polygon_covered_by_three_similar_l1193_119349


namespace inequality_proof_l1193_119300

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 - x)^n + (1 + x)^n < 2^n := by
  sorry

end inequality_proof_l1193_119300


namespace cheryl_same_color_probability_l1193_119338

def total_marbles : ℕ := 9
def marbles_per_color : ℕ := 3
def colors : ℕ := 3
def marbles_taken_each : ℕ := 3

theorem cheryl_same_color_probability :
  let total_outcomes := (total_marbles.choose marbles_taken_each) *
                        ((total_marbles - marbles_taken_each).choose marbles_taken_each) *
                        ((total_marbles - 2 * marbles_taken_each).choose marbles_taken_each)
  let favorable_outcomes := colors * ((total_marbles - 2 * marbles_taken_each).choose marbles_taken_each)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 28 := by
  sorry

end cheryl_same_color_probability_l1193_119338


namespace frank_brownies_columns_l1193_119398

/-- The number of columns Frank cut into the pan of brownies -/
def num_columns : ℕ := sorry

/-- The number of rows Frank cut into the pan of brownies -/
def num_rows : ℕ := 3

/-- The total number of people -/
def num_people : ℕ := 6

/-- The number of brownies each person can eat -/
def brownies_per_person : ℕ := 3

/-- The total number of brownies needed -/
def total_brownies : ℕ := num_people * brownies_per_person

theorem frank_brownies_columns :
  num_columns = total_brownies / num_rows :=
by sorry

end frank_brownies_columns_l1193_119398


namespace cos_2alpha_is_zero_l1193_119348

theorem cos_2alpha_is_zero (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (2*α) = Real.cos (π/4 - α)) : Real.cos (2*α) = 0 := by
  sorry

end cos_2alpha_is_zero_l1193_119348


namespace particle_position_after_2023_minutes_l1193_119322

/-- Represents the position of a particle as a pair of integers -/
def Position := ℤ × ℤ

/-- Represents the movement pattern of the particle -/
inductive MovementPattern
| OddSequence
| EvenSequence

/-- Calculates the next position based on the current position, movement pattern, and side length -/
def nextPosition (pos : Position) (pattern : MovementPattern) (side : ℕ) : Position :=
  match pattern with
  | MovementPattern.OddSequence => (pos.1 - side, pos.2 - side)
  | MovementPattern.EvenSequence => (pos.1 + side, pos.2 + side)

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- Theorem stating that the particle's position after 2023 minutes is (-43, -43) -/
theorem particle_position_after_2023_minutes :
  particlePosition 2023 = (-43, -43) :=
  sorry

end particle_position_after_2023_minutes_l1193_119322


namespace square_position_2010_l1193_119345

-- Define the possible positions of the square
inductive SquarePosition
  | ABCD
  | DABC
  | BDAC
  | ACBD
  | CABD
  | DCBA
  | CDAB
  | BADC
  | DBCA

def next_position (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BDAC
  | SquarePosition.DABC => SquarePosition.BDAC
  | SquarePosition.BDAC => SquarePosition.ACBD
  | SquarePosition.ACBD => SquarePosition.CABD
  | SquarePosition.CABD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DBCA
  | SquarePosition.DBCA => SquarePosition.ABCD

def nth_position (n : Nat) : SquarePosition :=
  match n with
  | 0 => SquarePosition.ABCD
  | n + 1 => next_position (nth_position n)

theorem square_position_2010 :
  nth_position 2010 = SquarePosition.BDAC :=
by sorry

end square_position_2010_l1193_119345


namespace system_solution_l1193_119317

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 6 ∧ y₁ = 3 ∧ x₂ = 3 ∧ y₂ = 3/2) ∧
    (∀ x y : ℝ,
      3*x - 2*y > 0 ∧ x > 0 →
      (Real.sqrt ((3*x - 2*y)/(2*x)) + Real.sqrt ((2*x)/(3*x - 2*y)) = 2 ∧
       x^2 - 18 = 2*y*(4*y - 9)) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end system_solution_l1193_119317


namespace circle_radius_decrease_l1193_119392

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.64 * A
  let r' := Real.sqrt (A' / π)
  (r' - r) / r = -0.2 := by
sorry

end circle_radius_decrease_l1193_119392


namespace quadratic_one_root_l1193_119326

theorem quadratic_one_root (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6*m*x + 4*m = 0 → (∀ y : ℝ, y^2 + 6*m*y + 4*m = 0 → x = y)) →
  m = 4/9 := by
sorry

end quadratic_one_root_l1193_119326


namespace particle_movement_probability_l1193_119389

/-- The probability of a particle reaching (n,n) from (0,0) in exactly 2n+k tosses -/
def particle_probability (n k : ℕ) : ℚ :=
  (Nat.choose (2*n + k - 1) (n - 1) : ℚ) * (1 / 2 ^ (2*n + k - 1))

/-- Theorem stating the probability of the particle reaching (n,n) in 2n+k tosses -/
theorem particle_movement_probability (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  particle_probability n k = (Nat.choose (2*n + k - 1) (n - 1) : ℚ) * (1 / 2 ^ (2*n + k - 1)) :=
by
  sorry


end particle_movement_probability_l1193_119389


namespace probability_not_perfect_power_l1193_119378

/-- A number is a perfect power if it can be expressed as x^y where x and y are integers and y > 1 -/
def IsPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are perfect powers -/
def PerfectPowerCount : ℕ := 21

/-- The total count of numbers from 1 to 200 -/
def TotalCount : ℕ := 200

/-- The probability of selecting a number that is not a perfect power -/
def ProbabilityNotPerfectPower : ℚ :=
  (TotalCount - PerfectPowerCount : ℚ) / TotalCount

theorem probability_not_perfect_power :
  ProbabilityNotPerfectPower = 179 / 200 := by
  sorry

end probability_not_perfect_power_l1193_119378


namespace thirtieth_term_of_sequence_l1193_119395

def arithmetic_sequence (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * (a₂ - a₁)

theorem thirtieth_term_of_sequence (a₁ a₂ : ℤ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) :
  arithmetic_sequence a₁ a₂ 30 = 119 := by
  sorry

end thirtieth_term_of_sequence_l1193_119395


namespace sum_fraction_bounds_l1193_119332

theorem sum_fraction_bounds (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (b + c + a) + c / (c + d + b) + d / (d + a + c)
  1 < S ∧ S < 2 := by
  sorry

end sum_fraction_bounds_l1193_119332


namespace expression_simplification_l1193_119352

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := by
sorry

end expression_simplification_l1193_119352


namespace sqrt_190_44_sqrt_176_9_and_18769_integer_n_between_sqrt_l1193_119315

-- Define the table as a function
def f (x : ℝ) : ℝ := x^2

-- Theorem 1
theorem sqrt_190_44 : Real.sqrt 190.44 = 13.8 ∨ Real.sqrt 190.44 = -13.8 := by sorry

-- Theorem 2
theorem sqrt_176_9_and_18769 :
  (abs (Real.sqrt 176.9 - 13.3) < 0.1) ∧ (Real.sqrt 18769 = 137) := by sorry

-- Theorem 3
theorem integer_n_between_sqrt :
  ∀ n : ℕ, (13.5 < Real.sqrt n) ∧ (Real.sqrt n < 13.6) → (n = 183 ∨ n = 184) := by sorry

end sqrt_190_44_sqrt_176_9_and_18769_integer_n_between_sqrt_l1193_119315


namespace ngon_existence_uniqueness_l1193_119325

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields

/-- Represents a point in a plane -/
structure Point where
  -- Add necessary fields

/-- Represents an n-gon -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Checks if a line is perpendicular to a side of a polygon at its midpoint -/
def is_perpendicular_at_midpoint (l : Line) (p : Polygon n) (i : Fin n) : Prop :=
  sorry

/-- Checks if a line is a bisector of an internal or external angle of a polygon -/
def is_angle_bisector (l : Line) (p : Polygon n) (i : Fin n) : Prop :=
  sorry

/-- Represents the solution status of the problem -/
inductive SolutionStatus
| Unique
| Indeterminate
| NoSolution

/-- The main theorem stating the existence and uniqueness of the n-gon -/
theorem ngon_existence_uniqueness 
  (n : ℕ) 
  (lines : Fin n → Line) 
  (condition : (l : Line) → (p : Polygon n) → (i : Fin n) → Prop) : 
  SolutionStatus :=
sorry

end ngon_existence_uniqueness_l1193_119325


namespace stair_steps_left_l1193_119383

theorem stair_steps_left (total : ℕ) (climbed : ℕ) (h1 : total = 96) (h2 : climbed = 74) :
  total - climbed = 22 := by
  sorry

end stair_steps_left_l1193_119383


namespace arrangement_count_l1193_119372

/-- The number of ways to arrange young and elderly people in a line with specific conditions -/
def arrangements (n r : ℕ) : ℕ :=
  (n.factorial * (n - r).factorial) / (n - 2*r).factorial

/-- Theorem stating the number of arrangements for young and elderly people -/
theorem arrangement_count (n r : ℕ) (h : n > 2*r) :
  arrangements n r = (n.factorial * (n - r).factorial) / (n - 2*r).factorial :=
by sorry

end arrangement_count_l1193_119372


namespace wxyz_unique_product_l1193_119386

/-- Represents a letter of the alphabet -/
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Assigns a numeric value to each letter -/
def letterValue : Letter → Nat
| Letter.A => 1  | Letter.B => 2  | Letter.C => 3  | Letter.D => 4
| Letter.E => 5  | Letter.F => 6  | Letter.G => 7  | Letter.H => 8
| Letter.I => 9  | Letter.J => 10 | Letter.K => 11 | Letter.L => 12
| Letter.M => 13 | Letter.N => 14 | Letter.O => 15 | Letter.P => 16
| Letter.Q => 17 | Letter.R => 18 | Letter.S => 19 | Letter.T => 20
| Letter.U => 21 | Letter.V => 22 | Letter.W => 23 | Letter.X => 24
| Letter.Y => 25 | Letter.Z => 26

/-- Represents a four-letter sequence -/
structure FourLetterSequence :=
  (first second third fourth : Letter)

/-- Calculates the product of a four-letter sequence -/
def sequenceProduct (seq : FourLetterSequence) : Nat :=
  (letterValue seq.first) * (letterValue seq.second) * (letterValue seq.third) * (letterValue seq.fourth)

/-- States that WXYZ is the unique four-letter sequence with a product of 29700 -/
theorem wxyz_unique_product :
  ∀ (seq : FourLetterSequence),
    sequenceProduct seq = 29700 →
    seq = FourLetterSequence.mk Letter.W Letter.X Letter.Y Letter.Z :=
by sorry

end wxyz_unique_product_l1193_119386


namespace union_of_M_and_N_l1193_119353

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end union_of_M_and_N_l1193_119353


namespace g_of_seven_equals_twentyone_l1193_119399

/-- Given that g(3x - 8) = 2x + 11 for all real x, prove that g(7) = 21 -/
theorem g_of_seven_equals_twentyone (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 := by
  sorry

end g_of_seven_equals_twentyone_l1193_119399


namespace nikolai_faster_l1193_119363

/-- Represents a mountain goat with its jump distance and number of jumps in a given time -/
structure Goat where
  name : String
  jumpDistance : ℕ
  jumpsPerTime : ℕ

/-- Calculates the distance covered by a goat in one time unit -/
def distancePerTime (g : Goat) : ℕ := g.jumpDistance * g.jumpsPerTime

/-- Calculates the number of jumps needed to cover a given distance -/
def jumpsNeeded (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jumpDistance - 1) / g.jumpDistance

theorem nikolai_faster (gennady nikolai : Goat) (totalDistance : ℕ) : 
  gennady.name = "Gennady" →
  nikolai.name = "Nikolai" →
  gennady.jumpDistance = 6 →
  gennady.jumpsPerTime = 2 →
  nikolai.jumpDistance = 4 →
  nikolai.jumpsPerTime = 3 →
  totalDistance = 2000 →
  distancePerTime gennady = distancePerTime nikolai →
  jumpsNeeded nikolai totalDistance < jumpsNeeded gennady totalDistance := by
  sorry

#eval jumpsNeeded (Goat.mk "Gennady" 6 2) 2000
#eval jumpsNeeded (Goat.mk "Nikolai" 4 3) 2000

end nikolai_faster_l1193_119363


namespace tangent_line_to_x_ln_x_l1193_119382

/-- The line y = 2x - e is tangent to the curve y = x ln x -/
theorem tangent_line_to_x_ln_x : ∃ (x₀ : ℝ), 
  (x₀ * Real.log x₀ = 2 * x₀ - Real.exp 1) ∧ 
  (Real.log x₀ + 1 = 2) := by
  sorry

end tangent_line_to_x_ln_x_l1193_119382


namespace cubic_inequality_l1193_119334

theorem cubic_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  a^3 + b^3 ≤ a*b^2 + a^2*b := by
  sorry

end cubic_inequality_l1193_119334


namespace a_plus_b_value_l1193_119354

theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, (3 * (a * x + b) - 8) = 4 * x + 5) → 
  a + b = 17/3 := by sorry

end a_plus_b_value_l1193_119354


namespace base_n_representation_l1193_119397

/-- Represents a number in base n -/
def BaseN (n : ℕ) (x : ℕ) : Prop := ∃ (d₁ d₀ : ℕ), x = d₁ * n + d₀ ∧ d₀ < n

theorem base_n_representation (n a b : ℕ) : 
  n > 8 → 
  n^2 - a*n + b = 0 → 
  BaseN n a → 
  BaseN n 18 → 
  BaseN n b → 
  BaseN n 80 := by
  sorry

end base_n_representation_l1193_119397


namespace manny_money_left_l1193_119327

/-- The cost of a plastic chair in dollars -/
def chair_cost : ℚ := 55 / 5

/-- The cost of a portable table in dollars -/
def table_cost : ℚ := 3 * chair_cost

/-- Manny's initial amount of money in dollars -/
def initial_money : ℚ := 100

/-- The cost of Manny's purchase (one table and two chairs) in dollars -/
def purchase_cost : ℚ := table_cost + 2 * chair_cost

/-- The amount of money left after Manny's purchase -/
def money_left : ℚ := initial_money - purchase_cost

theorem manny_money_left : money_left = 45 := by sorry

end manny_money_left_l1193_119327


namespace min_teachers_cover_all_subjects_l1193_119330

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- The maximum number of subjects a teacher can teach -/
def maxSubjectsPerTeacher : Nat := 3

/-- The total number of subjects -/
def totalSubjects : Nat := 3

/-- Given the number of teachers for each subject, calculates the minimum number
    of teachers required to cover all subjects -/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  sorry

theorem min_teachers_cover_all_subjects (counts : TeacherCounts) :
  counts.maths = 7 →
  counts.physics = 6 →
  counts.chemistry = 5 →
  minTeachersRequired counts = 7 :=
sorry

end min_teachers_cover_all_subjects_l1193_119330


namespace b_range_l1193_119324

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x

-- Define the derivative of f(x)
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 3*x^2 + b

-- Theorem statement
theorem b_range (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, (f_derivative b x) ≤ 0) →
  b ∈ Set.Iic (-3) :=
by sorry

end b_range_l1193_119324


namespace quadratic_range_l1193_119388

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem quadratic_range :
  ∀ x : ℝ, (2 ≤ quadratic_function x ∧ quadratic_function x < 5) ↔
  (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) := by sorry

end quadratic_range_l1193_119388


namespace average_weight_increase_l1193_119377

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 85 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end average_weight_increase_l1193_119377


namespace math_class_size_l1193_119391

/-- Represents a class of students who took a math test. -/
structure MathClass where
  total_students : ℕ
  both_solvers : ℕ
  harder_solvers : ℕ
  easier_solvers : ℕ

/-- Conditions for the math class problem. -/
def valid_math_class (c : MathClass) : Prop :=
  -- Each student solved at least one problem
  c.total_students = c.both_solvers + c.harder_solvers + c.easier_solvers
  -- Number of students who solved only one problem is one less than twice the number who solved both
  ∧ c.harder_solvers + c.easier_solvers = 2 * c.both_solvers - 1
  -- Total homework solutions from (both + harder) equals total from easier
  ∧ c.both_solvers + 4 * c.harder_solvers = c.easier_solvers

/-- The theorem stating that the class has 32 students. -/
theorem math_class_size :
  ∃ (c : MathClass), valid_math_class c ∧ c.total_students = 32 :=
sorry

end math_class_size_l1193_119391


namespace remaining_miles_l1193_119379

def total_journey : ℕ := 1200
def miles_driven : ℕ := 215

theorem remaining_miles :
  total_journey - miles_driven = 985 := by sorry

end remaining_miles_l1193_119379


namespace mask_quality_most_suitable_l1193_119318

-- Define the survey types
inductive SurveyType
| SecurityCheck
| TeacherRecruitment
| MaskQuality
| StudentVision

-- Define a function to determine if a survey is suitable for sampling
def isSuitableForSampling (survey : SurveyType) : Prop :=
  match survey with
  | SurveyType.MaskQuality => True
  | _ => False

-- Theorem statement
theorem mask_quality_most_suitable :
  ∀ (survey : SurveyType), isSuitableForSampling survey → survey = SurveyType.MaskQuality :=
by sorry

end mask_quality_most_suitable_l1193_119318


namespace maggie_earnings_l1193_119307

/-- The amount Maggie earns for each magazine subscription she sells -/
def earnings_per_subscription : ℝ := 5

/-- The number of subscriptions Maggie sold to her parents -/
def parents_subscriptions : ℕ := 4

/-- The number of subscriptions Maggie sold to her grandfather -/
def grandfather_subscriptions : ℕ := 1

/-- The number of subscriptions Maggie sold to the next-door neighbor -/
def neighbor_subscriptions : ℕ := 2

/-- The total amount Maggie earned from all subscriptions -/
def total_earnings : ℝ := 55

theorem maggie_earnings :
  earnings_per_subscription * (parents_subscriptions + grandfather_subscriptions + 
  neighbor_subscriptions + 2 * neighbor_subscriptions) = total_earnings :=
sorry

end maggie_earnings_l1193_119307


namespace circle_equations_l1193_119337

-- Define the points
def M : ℝ × ℝ := (-1, 1)
def N : ℝ × ℝ := (0, 2)
def Q : ℝ × ℝ := (2, 0)

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 5/2
def C₂ (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 5/2)^2 = 5/2

-- Define the line MN
def MN (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem circle_equations :
  (C₁ M.1 M.2 ∧ C₁ N.1 N.2 ∧ C₁ Q.1 Q.2) ∧
  (∀ x y x' y', C₁ x y ∧ C₂ x' y' → 
    MN ((x + x')/2) ((y + y')/2) ∧
    (x' - x)^2 + (y' - y)^2 = 4 * ((x - 1/2)^2 + (y - 1/2)^2)) :=
by sorry

end circle_equations_l1193_119337


namespace box_height_l1193_119351

/-- Given a rectangular box with width 10 inches, length 20 inches, and height h inches,
    if the area of the triangle formed by the center points of three faces meeting at a corner
    is 40 square inches, then h = (24 * sqrt(21)) / 5 inches. -/
theorem box_height (h : ℝ) : 
  let width : ℝ := 10
  let length : ℝ := 20
  let triangle_area : ℝ := 40
  let diagonal := Real.sqrt (width ^ 2 + length ^ 2)
  let side1 := Real.sqrt (width ^ 2 + (h / 2) ^ 2)
  let side2 := Real.sqrt (length ^ 2 + (h / 2) ^ 2)
  triangle_area = Real.sqrt (
    (diagonal + side1 + side2) *
    (diagonal + side1 - side2) *
    (diagonal - side1 + side2) *
    (-diagonal + side1 + side2)
  ) / 4
  →
  h = 24 * Real.sqrt 21 / 5 := by
sorry

end box_height_l1193_119351


namespace blue_marbles_total_l1193_119357

/-- The number of blue marbles Jason has -/
def jason_blue : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue : ℕ := 24

/-- The total number of blue marbles Jason and Tom have together -/
def total_blue : ℕ := jason_blue + tom_blue

theorem blue_marbles_total : total_blue = 68 := by sorry

end blue_marbles_total_l1193_119357


namespace triangle_max_area_l1193_119364

/-- Given a triangle ABC where:
  - The sides a, b, c are opposite to angles A, B, C respectively
  - a = 2
  - tan A / tan B = 4/3
  The maximum area of the triangle is 1/2 -/
theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) : 
  a = 2 → 
  (Real.tan A) / (Real.tan B) = 4/3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (∃ (S : ℝ), S = (1/2) * b * c * (Real.sin A) ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * (Real.sin A) → S' ≤ 1/2) := by
  sorry

end triangle_max_area_l1193_119364


namespace central_square_area_l1193_119373

theorem central_square_area (side_length : ℝ) (cut_distance : ℝ) :
  side_length = 15 →
  cut_distance = 4 →
  let central_square_side := cut_distance * Real.sqrt 2
  central_square_side ^ 2 = 32 := by
  sorry

end central_square_area_l1193_119373


namespace quadratic_roots_sum_l1193_119304

theorem quadratic_roots_sum (a b : ℝ) : 
  (∃ (p q r : ℝ), p * (Complex.I ^ 2) + q * Complex.I + r = 0 ∧ 
   (3 + a * Complex.I) * ((3 + a * Complex.I) - (b - 2 * Complex.I)) = 0) → 
  a + b = 1 := by
sorry

end quadratic_roots_sum_l1193_119304


namespace cannot_determine_percentage_increase_l1193_119319

/-- Represents a manufacturing machine -/
structure Machine where
  name : String
  production_rate : ℝ

/-- The problem setup -/
def sprocket_problem (time_q : ℝ) : Prop :=
  let machine_a : Machine := ⟨"A", 4⟩
  let machine_q : Machine := ⟨"Q", 440 / time_q⟩
  let machine_p : Machine := ⟨"P", 440 / (time_q + 10)⟩
  let percentage_increase := (machine_q.production_rate - machine_a.production_rate) / machine_a.production_rate * 100

  -- Conditions
  440 > 0 ∧
  time_q > 0 ∧
  machine_p.production_rate < machine_q.production_rate ∧
  -- Question: Can we determine the percentage increase?
  ∃ (x : ℝ), percentage_increase = x

/-- The theorem stating that we cannot determine the percentage increase without knowing time_q -/
theorem cannot_determine_percentage_increase :
  ¬∃ (x : ℝ), ∀ (time_q : ℝ), sprocket_problem time_q → 
    (440 / time_q - 4) / 4 * 100 = x :=
sorry

end cannot_determine_percentage_increase_l1193_119319


namespace smallest_M_with_non_decimal_k_l1193_119375

/-- Sum of digits in base-five representation of n -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-twelve representation of n -/
def k (n : ℕ) : ℕ := sorry

/-- Base-sixteen representation of n as a list of digits -/
def base_sixteen (n : ℕ) : List ℕ := sorry

/-- Checks if a list of base-sixteen digits contains a non-decimal digit -/
def has_non_decimal_digit (digits : List ℕ) : Prop :=
  digits.any (λ d => d ≥ 10)

theorem smallest_M_with_non_decimal_k :
  ∃ M : ℕ, (∀ n < M, ¬has_non_decimal_digit (base_sixteen (k n))) ∧
           has_non_decimal_digit (base_sixteen (k M)) ∧
           M = 24 := by sorry

#eval 24 % 1000  -- Should output 24

end smallest_M_with_non_decimal_k_l1193_119375


namespace expand_expression_l1193_119339

theorem expand_expression (y : ℝ) : 12 * (3 * y - 4) = 36 * y - 48 := by
  sorry

end expand_expression_l1193_119339


namespace mean_home_runs_l1193_119342

def total_players : ℕ := 12
def players_with_5 : ℕ := 3
def players_with_7 : ℕ := 5
def players_with_9 : ℕ := 3
def players_with_11 : ℕ := 1

def total_home_runs : ℕ := 
  5 * players_with_5 + 7 * players_with_7 + 9 * players_with_9 + 11 * players_with_11

theorem mean_home_runs : 
  (total_home_runs : ℚ) / total_players = 88 / 12 := by sorry

end mean_home_runs_l1193_119342


namespace total_spent_is_200_l1193_119371

/-- The amount Pete and Raymond each received in cents -/
def initial_amount : ℕ := 250

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of nickels Pete spent -/
def pete_nickels_spent : ℕ := 4

/-- The number of dimes Raymond has left -/
def raymond_dimes_left : ℕ := 7

/-- Theorem: The total amount spent by Pete and Raymond is 200 cents -/
theorem total_spent_is_200 : 
  (pete_nickels_spent * nickel_value) + 
  (initial_amount - (raymond_dimes_left * dime_value)) = 200 := by
  sorry

end total_spent_is_200_l1193_119371


namespace base3_sum_correct_l1193_119390

/-- Represents a number in base 3 --/
def Base3 : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def toDecimal (n : Base3) : Nat :=
  n.reverse.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The theorem stating that the sum of the given base 3 numbers is correct --/
theorem base3_sum_correct : 
  let a : Base3 := [2]
  let b : Base3 := [2, 0, 1]
  let c : Base3 := [2, 0, 1, 1]
  let d : Base3 := [1, 2, 0, 1, 1]
  let sum : Base3 := [1, 2, 2, 1]
  toDecimal a + toDecimal b + toDecimal c + toDecimal d = toDecimal sum := by
  sorry

end base3_sum_correct_l1193_119390


namespace equation_solution_l1193_119305

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^14 = (14 * x)^7 ↔ x = 2/7 := by sorry

end equation_solution_l1193_119305


namespace imaginary_part_of_reciprocal_l1193_119355

-- Define the complex number z
def z : ℂ := 1 - 2 * Complex.I

-- Theorem statement
theorem imaginary_part_of_reciprocal (z : ℂ) (h : z = 1 - 2 * Complex.I) :
  Complex.im (z⁻¹) = 2 / 5 := by
  sorry

end imaginary_part_of_reciprocal_l1193_119355


namespace irreducible_fraction_l1193_119328

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end irreducible_fraction_l1193_119328


namespace estimate_product_l1193_119384

def approximate_819 : ℕ := 800
def approximate_32 : ℕ := 30

theorem estimate_product : 
  approximate_819 * approximate_32 = 24000 := by sorry

end estimate_product_l1193_119384


namespace triangle_area_product_l1193_119374

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (1/2) * (4/a) * (6/b) = 3 → 
  a * b = 4 := by sorry

end triangle_area_product_l1193_119374


namespace f_two_equals_negative_eight_l1193_119310

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_two_equals_negative_eight
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x^3) :
  f 2 = -8 := by
  sorry

end f_two_equals_negative_eight_l1193_119310


namespace cross_pollinated_percentage_l1193_119302

theorem cross_pollinated_percentage
  (total : ℕ)  -- Total number of trees
  (fuji : ℕ)   -- Number of pure Fuji trees
  (gala : ℕ)   -- Number of pure Gala trees
  (cross : ℕ)  -- Number of cross-pollinated trees
  (h1 : total = fuji + gala + cross)  -- Total trees equation
  (h2 : fuji + cross = 221)           -- Pure Fuji + Cross-pollinated
  (h3 : fuji = (3 * total) / 4)       -- 3/4 of all trees are pure Fuji
  (h4 : gala = 39)                    -- Number of pure Gala trees
  : (cross : ℚ) / total * 100 = 10 := by
  sorry

end cross_pollinated_percentage_l1193_119302


namespace prime_product_sum_difference_l1193_119316

theorem prime_product_sum_difference : ∃ x y : ℕ, 
  x.Prime ∧ y.Prime ∧ 
  x ≠ y ∧ 
  20 < x ∧ x < 40 ∧ 
  20 < y ∧ y < 40 ∧ 
  x * y - (x + y) = 899 := by
  sorry

end prime_product_sum_difference_l1193_119316


namespace expression_evaluation_l1193_119347

theorem expression_evaluation : (20 ^ 40) / (40 ^ 20) = 10 ^ 20 := by
  sorry

end expression_evaluation_l1193_119347


namespace volleyball_team_combinations_l1193_119358

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 6

def choose_starters (n k : ℕ) : ℕ := Nat.choose n k

theorem volleyball_team_combinations : 
  choose_starters (total_players - quadruplets) starters + 
  quadruplets * choose_starters (total_players - quadruplets) (starters - 1) + 
  Nat.choose quadruplets 2 * choose_starters (total_players - quadruplets) (starters - 2) = 7062 := by
  sorry

end volleyball_team_combinations_l1193_119358


namespace number_equation_solution_l1193_119367

theorem number_equation_solution : ∃ x : ℚ, (3 * x + 15 = 6 * x - 10) ∧ (x = 25 / 3) := by
  sorry

end number_equation_solution_l1193_119367


namespace xyz_inequality_l1193_119311

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end xyz_inequality_l1193_119311


namespace constant_term_expansion_l1193_119369

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℝ, k = 24 ∧ k = (3/2) * a^2) → (a = 4 ∨ a = -4) := by
  sorry

end constant_term_expansion_l1193_119369


namespace parallelogram_side_sum_l1193_119308

theorem parallelogram_side_sum (x y : ℝ) : 
  (5 : ℝ) = 10 * y - 3 ∧ (11 : ℝ) = 4 * x + 1 → x + y = 3.3 := by
  sorry

end parallelogram_side_sum_l1193_119308


namespace hyperbola_focus_asymptote_distance_l1193_119385

-- Define the hyperbola
def is_hyperbola (x y m : ℝ) : Prop := x^2 - y^2 / m^2 = 1

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define the distance from focus to asymptote
def focus_asymptote_distance (m : ℝ) : ℝ := m

-- Theorem statement
theorem hyperbola_focus_asymptote_distance (m : ℝ) :
  m_positive m →
  (∃ x y, is_hyperbola x y m) →
  focus_asymptote_distance m = 4 →
  m = 4 :=
by sorry

end hyperbola_focus_asymptote_distance_l1193_119385


namespace unique_point_for_equal_angles_l1193_119331

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (4, 0)

-- Define a chord passing through the focus
def chord (a b : ℝ × ℝ) : Prop :=
  a.1 ≠ b.1 ∨ a.2 ≠ b.2  -- Ensure A and B are distinct points
  ∧ ellipse a.1 a.2      -- A is on the ellipse
  ∧ ellipse b.1 b.2      -- B is on the ellipse
  ∧ (b.2 - a.2) * (a.1 - 1) = (b.1 - a.1) * (a.2 - 0)  -- AB passes through F(1,0)

-- Define the equality of angles APF and BPF
def equal_angles (a b : ℝ × ℝ) : Prop :=
  (a.2 - 0) * (b.1 - 4) = (b.2 - 0) * (a.1 - 4)

theorem unique_point_for_equal_angles :
  ∀ a b : ℝ × ℝ, chord a b → equal_angles a b ∧
  ∀ p : ℝ, p > 0 ∧ p ≠ 4 →
    ∃ c d : ℝ × ℝ, chord c d ∧ ¬(c.2 - 0) * (d.1 - p) = (d.2 - 0) * (c.1 - p) :=
sorry

end unique_point_for_equal_angles_l1193_119331


namespace smallest_gcd_value_l1193_119335

theorem smallest_gcd_value (a b c d : ℕ) : 
  (∃ (gcd_list : List ℕ), 
    gcd_list.length = 6 ∧ 
    1 ∈ gcd_list ∧ 
    2 ∈ gcd_list ∧ 
    3 ∈ gcd_list ∧ 
    4 ∈ gcd_list ∧ 
    5 ∈ gcd_list ∧
    (∃ (N : ℕ), N > 5 ∧ N ∈ gcd_list) ∧
    (∀ (x : ℕ), x ∈ gcd_list → 
      x = Nat.gcd a b ∨ 
      x = Nat.gcd a c ∨ 
      x = Nat.gcd a d ∨ 
      x = Nat.gcd b c ∨ 
      x = Nat.gcd b d ∨ 
      x = Nat.gcd c d)) →
  (∀ (M : ℕ), M > 5 ∧ 
    (∃ (gcd_list : List ℕ), 
      gcd_list.length = 6 ∧ 
      1 ∈ gcd_list ∧ 
      2 ∈ gcd_list ∧ 
      3 ∈ gcd_list ∧ 
      4 ∈ gcd_list ∧ 
      5 ∈ gcd_list ∧
      M ∈ gcd_list ∧
      (∀ (x : ℕ), x ∈ gcd_list → 
        x = Nat.gcd a b ∨ 
        x = Nat.gcd a c ∨ 
        x = Nat.gcd a d ∨ 
        x = Nat.gcd b c ∨ 
        x = Nat.gcd b d ∨ 
        x = Nat.gcd c d)) →
    M ≥ 14) :=
by sorry

end smallest_gcd_value_l1193_119335


namespace print_shop_charge_l1193_119396

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 1.25

/-- The number of color copies -/
def num_copies : ℕ := 60

/-- The additional charge at print shop Y for 60 copies -/
def additional_charge : ℚ := 90

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 2.75

theorem print_shop_charge : 
  charge_y * num_copies = charge_x * num_copies + additional_charge := by
  sorry

end print_shop_charge_l1193_119396


namespace stock_price_calculation_l1193_119301

/-- Given a closing price and percent increase, calculate the opening price of a stock. -/
theorem stock_price_calculation (closing_price : ℝ) (percent_increase : ℝ) (opening_price : ℝ) :
  closing_price = 29 ∧ 
  percent_increase = 3.571428571428581 ∧
  (closing_price - opening_price) / opening_price * 100 = percent_increase →
  opening_price = 28 := by
sorry


end stock_price_calculation_l1193_119301


namespace det_scaled_matrices_l1193_119320

-- Define a 2x2 matrix type
def Matrix2x2 := Fin 2 → Fin 2 → ℝ

-- Define the determinant function for 2x2 matrices
def det (A : Matrix2x2) : ℝ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

-- Define a function to scale all elements of a matrix by a factor
def scaleMatrix (A : Matrix2x2) (k : ℝ) : Matrix2x2 :=
  λ i j ↦ k * A i j

-- Define a function to scale columns of a matrix by different factors
def scaleColumns (A : Matrix2x2) (k1 k2 : ℝ) : Matrix2x2 :=
  λ i j ↦ if j = 0 then k1 * A i j else k2 * A i j

-- State the theorem
theorem det_scaled_matrices (A : Matrix2x2) (h : det A = 3) :
  det (scaleMatrix A 3) = 27 ∧ det (scaleColumns A 4 2) = 24 := by
  sorry

end det_scaled_matrices_l1193_119320


namespace fermat_number_large_prime_factor_l1193_119376

theorem fermat_number_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^(2^n) + 1) ∧ p > 2^(n+2) * (n+1) := by
  sorry

end fermat_number_large_prime_factor_l1193_119376


namespace binomial_expansion_coefficient_l1193_119368

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (1 + x)^10 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + 
               a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7 + a₈*(1-x)^8 + 
               a₉*(1-x)^9 + a₁₀*(1-x)^10 ∧ 
  a₈ = 180 := by
sorry

end binomial_expansion_coefficient_l1193_119368


namespace ambers_age_l1193_119309

theorem ambers_age :
  ∀ (a g : ℕ),
  g = 15 * a →
  g - a = 70 →
  a = 5 := by
sorry

end ambers_age_l1193_119309


namespace faculty_size_l1193_119321

/-- The number of students studying numeric methods -/
def nm : ℕ := 240

/-- The number of students studying automatic control of airborne vehicles -/
def acav : ℕ := 423

/-- The number of students studying both numeric methods and automatic control -/
def nm_acav : ℕ := 134

/-- The number of students studying advanced robotics -/
def ar : ℕ := 365

/-- The number of students studying both numeric methods and advanced robotics -/
def nm_ar : ℕ := 75

/-- The number of students studying both automatic control and advanced robotics -/
def acav_ar : ℕ := 95

/-- The number of students studying all three subjects -/
def all_three : ℕ := 45

/-- The proportion of second year students to total students -/
def second_year_ratio : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 905

theorem faculty_size :
  (nm + acav + ar - nm_acav - nm_ar - acav_ar + all_three : ℚ) / second_year_ratio = total_students := by
  sorry

end faculty_size_l1193_119321


namespace impossible_inequalities_l1193_119366

theorem impossible_inequalities (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) :
  ¬(((a₁ * b₂) / (a₁ + b₂) < (a₂ * b₁) / (a₂ + b₁)) ∧
    ((a₂ * b₃) / (a₂ + b₃) > (a₃ * b₂) / (a₃ + b₂)) ∧
    ((a₃ * b₁) / (a₃ + b₁) > (a₁ * b₃) / (a₁ + b₃))) :=
by sorry

end impossible_inequalities_l1193_119366


namespace f_of_9_eq_836_l1193_119359

/-- The function f(n) = n^3 + n^2 + n + 17 -/
def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

/-- Theorem: The value of f(9) is 836 -/
theorem f_of_9_eq_836 : f 9 = 836 := by sorry

end f_of_9_eq_836_l1193_119359


namespace doctor_visit_cost_is_400_l1193_119361

/-- Represents Tom's medication and doctor visit expenses -/
structure MedicationExpenses where
  pills_per_day : ℕ
  doctor_visits_per_year : ℕ
  pill_cost : ℚ
  insurance_coverage : ℚ
  total_annual_cost : ℚ

/-- Calculates the cost of a single doctor visit -/
def doctor_visit_cost (e : MedicationExpenses) : ℚ :=
  let annual_pills := e.pills_per_day * 365
  let annual_pill_cost := annual_pills * e.pill_cost
  let patient_pill_cost := annual_pill_cost * (1 - e.insurance_coverage)
  let annual_doctor_cost := e.total_annual_cost - patient_pill_cost
  annual_doctor_cost / e.doctor_visits_per_year

/-- Theorem stating that Tom's doctor visit costs $400 -/
theorem doctor_visit_cost_is_400 (e : MedicationExpenses) 
  (h1 : e.pills_per_day = 2)
  (h2 : e.doctor_visits_per_year = 2)
  (h3 : e.pill_cost = 5)
  (h4 : e.insurance_coverage = 4/5)
  (h5 : e.total_annual_cost = 1530) :
  doctor_visit_cost e = 400 := by
  sorry

end doctor_visit_cost_is_400_l1193_119361


namespace find_number_l1193_119314

theorem find_number (x : ℝ) (h : 0.46 * x = 165.6) : x = 360 := by
  sorry

end find_number_l1193_119314


namespace max_area_right_triangle_pen_l1193_119370

/-- The maximum area of a right triangular pen with perimeter 60 feet is 450 square feet. -/
theorem max_area_right_triangle_pen (x y : ℝ) : 
  x > 0 → y > 0 → x + y + Real.sqrt (x^2 + y^2) = 60 → 
  (1/2) * x * y ≤ 450 := by
sorry

end max_area_right_triangle_pen_l1193_119370


namespace motion_solution_correct_l1193_119365

/-- Two bodies moving towards each other with uniform acceleration -/
structure MotionProblem where
  initialDistance : ℝ
  initialVelocityA : ℝ
  accelerationA : ℝ
  initialVelocityB : ℝ
  accelerationB : ℝ

/-- Solution to the motion problem -/
structure MotionSolution where
  time : ℝ
  distanceA : ℝ
  distanceB : ℝ

/-- The function to solve the motion problem -/
def solveMotion (p : MotionProblem) : MotionSolution :=
  { time := 7,
    distanceA := 143.5,
    distanceB := 199.5 }

/-- Theorem stating that the solution is correct -/
theorem motion_solution_correct (p : MotionProblem) :
  p.initialDistance = 343 ∧
  p.initialVelocityA = 3 ∧
  p.accelerationA = 5 ∧
  p.initialVelocityB = 4 ∧
  p.accelerationB = 7 →
  let s := solveMotion p
  s.time = 7 ∧
  s.distanceA = 143.5 ∧
  s.distanceB = 199.5 ∧
  s.distanceA + s.distanceB = p.initialDistance :=
by
  sorry


end motion_solution_correct_l1193_119365


namespace power_five_sum_minus_two_l1193_119360

theorem power_five_sum_minus_two (n : ℕ) : n^5 + n^5 + n^5 + n^5 - 2 * n^5 = 2 * n^5 :=
by
  sorry

end power_five_sum_minus_two_l1193_119360


namespace bagel_bakery_bound_l1193_119346

/-- Definition of a bagel -/
def Bagel (a b : ℕ) : ℕ := 2 * a + 2 * b + 4

/-- Definition of a bakery of order n -/
def Bakery (n : ℕ) : Set (ℕ × ℕ) := sorry

/-- The smallest possible number of cells in a bakery of order n -/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem bagel_bakery_bound :
  ∃ (α : ℝ), ∀ (n : ℕ), n ≥ 8 → Even n →
    ∃ (N : ℕ), ∀ (m : ℕ), m ≥ N →
      (1 / 100 : ℝ) < (f m : ℝ) / m ^ α ∧ (f m : ℝ) / m ^ α < 100 :=
by sorry

end bagel_bakery_bound_l1193_119346


namespace no_function_satisfies_conditions_l1193_119343

theorem no_function_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ) (a b : ℝ),
    (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
    (a > 0 ∧ b > 0) ∧
    (∀ x : ℝ, f (x^2) - (f (a * x + b))^2 ≥ 1/4) := by
  sorry

end no_function_satisfies_conditions_l1193_119343


namespace smallest_value_in_range_l1193_119344

theorem smallest_value_in_range (x : ℝ) (h : 0 < x ∧ x < 2) :
  x^2 ≤ min x (min (3*x) (min (Real.sqrt x) (1/x))) := by
  sorry

end smallest_value_in_range_l1193_119344


namespace chicken_pasta_orders_count_l1193_119323

def chicken_pasta_pieces : ℕ := 2
def barbecue_chicken_pieces : ℕ := 3
def fried_chicken_dinner_pieces : ℕ := 8
def fried_chicken_dinner_orders : ℕ := 2
def barbecue_chicken_orders : ℕ := 3
def total_chicken_pieces : ℕ := 37

theorem chicken_pasta_orders_count : 
  ∃ (chicken_pasta_orders : ℕ), 
    chicken_pasta_orders * chicken_pasta_pieces + 
    barbecue_chicken_orders * barbecue_chicken_pieces + 
    fried_chicken_dinner_orders * fried_chicken_dinner_pieces = 
    total_chicken_pieces ∧ 
    chicken_pasta_orders = 6 := by
  sorry

end chicken_pasta_orders_count_l1193_119323


namespace optimal_rectangle_dimensions_l1193_119362

-- Define the rectangle dimensions
def width : ℝ := 14.625
def length : ℝ := 34.25

-- Define the conditions
def area_constraint (w l : ℝ) : Prop := w * l ≥ 500
def length_constraint (w l : ℝ) : Prop := l = 2 * w + 5

-- Define the perimeter function
def perimeter (w l : ℝ) : ℝ := 2 * (w + l)

theorem optimal_rectangle_dimensions :
  area_constraint width length ∧
  length_constraint width length ∧
  ∀ w l : ℝ, w > 0 → l > 0 →
    area_constraint w l →
    length_constraint w l →
    perimeter width length ≤ perimeter w l :=
sorry

end optimal_rectangle_dimensions_l1193_119362


namespace equilateral_triangle_area_perimeter_ratio_l1193_119387

/-- The ratio of area to perimeter for an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end equilateral_triangle_area_perimeter_ratio_l1193_119387


namespace min_tea_time_l1193_119380

def wash_kettle : ℕ := 1
def boil_water : ℕ := 10
def wash_cups : ℕ := 2
def get_leaves : ℕ := 1
def brew_tea : ℕ := 1

theorem min_tea_time : 
  ∃ (arrangement : ℕ), 
    arrangement = max boil_water (wash_kettle + wash_cups + get_leaves) + brew_tea ∧
    arrangement = 11 ∧
    ∀ (other_arrangement : ℕ), other_arrangement ≥ arrangement :=
by sorry

end min_tea_time_l1193_119380


namespace cube_of_negative_l1193_119350

theorem cube_of_negative (x : ℝ) (h : x^3 = 32.768) : (-x)^3 = -32.768 := by
  sorry

end cube_of_negative_l1193_119350


namespace english_spanish_difference_l1193_119329

/-- The number of hours Ryan spends learning English -/
def hours_english : ℕ := 7

/-- The number of hours Ryan spends learning Chinese -/
def hours_chinese : ℕ := 2

/-- The number of hours Ryan spends learning Spanish -/
def hours_spanish : ℕ := 4

/-- Theorem: Ryan spends 3 more hours on learning English than Spanish -/
theorem english_spanish_difference : hours_english - hours_spanish = 3 := by
  sorry

end english_spanish_difference_l1193_119329


namespace factorial_multiple_of_eight_l1193_119303

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_multiple_of_eight (n : ℕ) :
  (∃ k : ℕ, factorial n = 8 * k) → n ≥ 4 := by
  sorry

end factorial_multiple_of_eight_l1193_119303


namespace triangle_area_l1193_119312

/-- Given a triangle ABC with sides a, b, c and circumradius R, 
    prove that its area is 2√3 / 3 under specific conditions -/
theorem triangle_area (a b c R : ℝ) (h1 : (a^2 - c^2) / (2*R) = (a - b) * Real.sin b)
                                    (h2 : Real.sin b = 2 * Real.sin a)
                                    (h3 : c = 2) :
  (1/2) * a * b * Real.sin ((1/3) * Real.pi) = 2 * Real.sqrt 3 / 3 := by
  sorry

end triangle_area_l1193_119312


namespace any_amount_possible_large_amount_without_change_l1193_119381

/-- Represents the currency system of Bordavia -/
structure BordaviaCurrency where
  m : ℕ  -- value of silver coin
  n : ℕ  -- value of gold coin
  h1 : ∃ (a b : ℕ), a * m + b * n = 10000
  h2 : ∃ (a b : ℕ), a * m + b * n = 1875
  h3 : ∃ (a b : ℕ), a * m + b * n = 3072

/-- Any integer amount of Bourbakis can be obtained using gold and silver coins -/
theorem any_amount_possible (currency : BordaviaCurrency) :
  ∀ k : ℤ, ∃ (a b : ℤ), a * currency.m + b * currency.n = k :=
sorry

/-- Any amount over (mn - 2) Bourbakis can be paid without needing change -/
theorem large_amount_without_change (currency : BordaviaCurrency) :
  ∀ k : ℕ, k > currency.m * currency.n - 2 →
    ∃ (a b : ℕ), a * currency.m + b * currency.n = k :=
sorry

end any_amount_possible_large_amount_without_change_l1193_119381


namespace additional_track_length_l1193_119356

/-- Calculate the additional track length required when reducing grade -/
theorem additional_track_length
  (rise : ℝ)
  (initial_grade : ℝ)
  (reduced_grade : ℝ)
  (h1 : rise = 800)
  (h2 : initial_grade = 0.04)
  (h3 : reduced_grade = 0.025) :
  (rise / reduced_grade) - (rise / initial_grade) = 12000 :=
by sorry

end additional_track_length_l1193_119356
