import Mathlib

namespace complex_root_and_purely_imaginary_l2314_231408

/-- Given that 2-i is a root of x^2 - mx + n = 0 where m and n are real,
    prove that m = 4, n = 5, and if z = a^2 - na + m + (a-m)i is purely imaginary, then a = 1 -/
theorem complex_root_and_purely_imaginary (m n a : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((2 : ℂ) - Complex.I) ^ 2 - m * ((2 : ℂ) - Complex.I) + n = 0 →
  (∃ (b : ℝ), (a ^ 2 - n * a + m : ℂ) + (a - m) * Complex.I = b * Complex.I) →
  (m = 4 ∧ n = 5 ∧ a = 1) :=
by sorry

end complex_root_and_purely_imaginary_l2314_231408


namespace express_y_in_terms_of_x_l2314_231479

/-- Given the equation 3x - y = 9, prove that y can be expressed as 3x - 9 -/
theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x - y = 9) : y = 3 * x - 9 := by
  sorry

end express_y_in_terms_of_x_l2314_231479


namespace emily_oranges_l2314_231441

theorem emily_oranges (betty_oranges sandra_oranges emily_oranges : ℕ) : 
  betty_oranges = 12 →
  sandra_oranges = 3 * betty_oranges →
  emily_oranges = 7 * sandra_oranges →
  emily_oranges = 252 := by
sorry

end emily_oranges_l2314_231441


namespace max_value_implies_m_l2314_231410

open Real

theorem max_value_implies_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = sin (x + π/2) + cos (x - π/2) + m) →
  (∃ x₀, ∀ x, f x ≤ f x₀) →
  (∃ x₁, f x₁ = 2 * sqrt 2) →
  m = sqrt 2 := by
  sorry

end max_value_implies_m_l2314_231410


namespace infinite_sum_equality_l2314_231421

/-- For positive real numbers p and q where p > 2q, the infinite sum
    1/(pq) + 1/(p(3p-2q)) + 1/((3p-2q)(5p-4q)) + 1/((5p-4q)(7p-6q)) + ...
    is equal to 1/((p-2q)p). -/
theorem infinite_sum_equality (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p > 2*q) :
  let f : ℕ → ℝ := λ n => 1 / ((2*n - 1)*p - (2*n - 2)*q) / ((2*n + 1)*p - 2*n*q)
  ∑' n, f n = 1 / ((p - 2*q) * p) := by
  sorry

end infinite_sum_equality_l2314_231421


namespace garden_outer_radius_l2314_231469

/-- Given a circular park with a central fountain and a surrounding garden ring,
    this theorem proves the radius of the garden's outer boundary. -/
theorem garden_outer_radius (fountain_diameter : ℝ) (garden_width : ℝ) :
  fountain_diameter = 12 →
  garden_width = 10 →
  fountain_diameter / 2 + garden_width = 16 := by
  sorry

end garden_outer_radius_l2314_231469


namespace height_difference_l2314_231495

/-- Given three heights in ratio 4 : 5 : 6 with the shortest being 120 cm, 
    prove that the sum of shortest and tallest minus the middle equals 150 cm -/
theorem height_difference (h₁ h₂ h₃ : ℝ) : 
  h₁ / h₂ = 4 / 5 → 
  h₂ / h₃ = 5 / 6 → 
  h₁ = 120 → 
  h₁ + h₃ - h₂ = 150 := by
sorry

end height_difference_l2314_231495


namespace even_function_inequality_l2314_231472

open Real Set

theorem even_function_inequality 
  (f : ℝ → ℝ) 
  (h_even : ∀ x, x ∈ Ioo (-π/2) (π/2) → f x = f (-x))
  (h_deriv : ∀ x, x ∈ Ioo 0 (π/2) → 
    (deriv^[2] f) x * cos x + f x * sin x < 0) :
  ∀ x, x ∈ (Ioo (-π/2) (-π/4) ∪ Ioo (π/4) (π/2)) → 
    f x < Real.sqrt 2 * f (π/4) * cos x := by
  sorry

end even_function_inequality_l2314_231472


namespace course_class_duration_l2314_231494

/-- Proves the duration of each class in a course given the total course duration and other parameters. -/
theorem course_class_duration 
  (weeks : ℕ) 
  (unknown_classes_per_week : ℕ) 
  (known_class_duration : ℕ) 
  (homework_duration : ℕ) 
  (total_course_time : ℕ) 
  (h1 : weeks = 24)
  (h2 : unknown_classes_per_week = 2)
  (h3 : known_class_duration = 4)
  (h4 : homework_duration = 4)
  (h5 : total_course_time = 336) :
  ∃ x : ℕ, x * unknown_classes_per_week * weeks + known_class_duration * weeks + homework_duration * weeks = total_course_time ∧ x = 3 := by
  sorry

#check course_class_duration

end course_class_duration_l2314_231494


namespace markus_to_son_age_ratio_l2314_231425

/-- Represents the ages of Markus, his son, and his grandson. -/
structure FamilyAges where
  markus : ℕ
  son : ℕ
  grandson : ℕ

/-- The conditions given in the problem. -/
def problemConditions (ages : FamilyAges) : Prop :=
  ages.grandson = 20 ∧
  ages.son = 2 * ages.grandson ∧
  ages.markus + ages.son + ages.grandson = 140

/-- The theorem stating that under the given conditions, 
    the ratio of Markus's age to his son's age is 2:1. -/
theorem markus_to_son_age_ratio 
  (ages : FamilyAges) 
  (h : problemConditions ages) : 
  ages.markus * 1 = ages.son * 2 := by
  sorry

#check markus_to_son_age_ratio

end markus_to_son_age_ratio_l2314_231425


namespace larry_win_probability_l2314_231454

/-- The probability of Larry winning a turn-based game against Julius, where:
  * Larry throws first
  * The probability of Larry knocking off the bottle is 3/5
  * The probability of Julius knocking off the bottle is 1/3
  * The winner is the first to knock off the bottle -/
theorem larry_win_probability (p_larry : ℝ) (p_julius : ℝ) 
  (h1 : p_larry = 3/5) 
  (h2 : p_julius = 1/3) :
  p_larry + (1 - p_larry) * (1 - p_julius) * p_larry / (1 - (1 - p_larry) * (1 - p_julius)) = 9/11 :=
by sorry

end larry_win_probability_l2314_231454


namespace median_of_special_list_l2314_231489

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the list -/
def total_elements : ℕ := sum_of_first_n 200

/-- The position of the median elements -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- The value that appears at the median positions -/
def median_value : ℕ := 141

/-- The median of the list -/
def list_median : ℚ := (median_value : ℚ)

theorem median_of_special_list : list_median = 141 := by sorry

end median_of_special_list_l2314_231489


namespace sum_of_y_coefficients_l2314_231415

-- Define the polynomials
def p (x y : ℝ) := 5*x + 3*y - 2
def q (x y : ℝ) := 2*x + 5*y + 7

-- Define the expanded product
def expanded_product (x y : ℝ) := p x y * q x y

-- Define a function to extract coefficients of terms with y
def y_coefficients (x y : ℝ) : List ℝ := 
  [31, 15, 11]  -- Coefficients of xy, y², and y respectively

-- Theorem statement
theorem sum_of_y_coefficients :
  (y_coefficients 0 0).sum = 57 :=
sorry

end sum_of_y_coefficients_l2314_231415


namespace no_perfect_square_pair_l2314_231428

theorem no_perfect_square_pair : ¬∃ (a b : ℕ+), 
  (∃ (k : ℕ+), (a.val ^ 2 + b.val : ℕ) = k.val ^ 2) ∧ 
  (∃ (m : ℕ+), (b.val ^ 2 + a.val : ℕ) = m.val ^ 2) := by
  sorry

end no_perfect_square_pair_l2314_231428


namespace max_value_complex_fraction_l2314_231467

theorem max_value_complex_fraction (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs ((Complex.I * Real.sqrt 3 - z) / (Real.sqrt 2 - z)) ≤ Real.sqrt 7 + Real.sqrt 5 := by
  sorry

end max_value_complex_fraction_l2314_231467


namespace tricycles_in_garage_l2314_231497

/-- The number of tricycles in Zoe's garage --/
def num_tricycles : ℕ := sorry

/-- The total number of wheels in the garage --/
def total_wheels : ℕ := 25

/-- The number of bicycles in the garage --/
def num_bicycles : ℕ := 3

/-- The number of unicycles in the garage --/
def num_unicycles : ℕ := 7

/-- The number of wheels on a bicycle --/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle --/
def wheels_per_tricycle : ℕ := 3

/-- The number of wheels on a unicycle --/
def wheels_per_unicycle : ℕ := 1

/-- Theorem stating that there are 4 tricycles in the garage --/
theorem tricycles_in_garage : num_tricycles = 4 := by
  sorry

end tricycles_in_garage_l2314_231497


namespace sampling_is_systematic_l2314_231403

/-- Represents a student ID number -/
structure StudentID where
  lastThreeDigits : Nat
  inv_range : 1 ≤ lastThreeDigits ∧ lastThreeDigits ≤ 818

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | Stratified
  | Systematic
  | RandomNumberTable

/-- Represents the selection criteria for inspection -/
def isSelected (id : StudentID) : Bool :=
  id.lastThreeDigits % 100 = 16

/-- Theorem stating that the sampling method is systematic -/
theorem sampling_is_systematic (ids : List StudentID) 
  (h1 : ∀ id ∈ ids, 1 ≤ id.lastThreeDigits ∧ id.lastThreeDigits ≤ 818) 
  (h2 : ∀ id ∈ ids, isSelected id ↔ id.lastThreeDigits % 100 = 16) : 
  SamplingMethod.Systematic = SamplingMethod.Systematic := by
  sorry

end sampling_is_systematic_l2314_231403


namespace secret_ballot_best_for_new_member_l2314_231496

/-- Represents a voting method -/
inductive VotingMethod
  | ShowOfHandsAgree
  | ShowOfHandsDisagree
  | SecretBallot
  | RecordedVote

/-- Represents the context of the vote -/
structure VoteContext where
  purpose : String

/-- Defines what it means for a voting method to reflect the true will of students -/
def reflectsTrueWill (method : VotingMethod) (context : VoteContext) : Prop := sorry

/-- Theorem stating that secret ballot best reflects the true will of students for adding a new class committee member -/
theorem secret_ballot_best_for_new_member :
  ∀ (context : VoteContext),
  context.purpose = "adding a new class committee member" →
  ∀ (method : VotingMethod),
  reflectsTrueWill VotingMethod.SecretBallot context →
  reflectsTrueWill method context →
  method = VotingMethod.SecretBallot :=
sorry

end secret_ballot_best_for_new_member_l2314_231496


namespace program_count_l2314_231412

/-- The total number of courses available --/
def total_courses : ℕ := 7

/-- The number of courses in a program --/
def program_size : ℕ := 5

/-- The number of math courses available --/
def math_courses : ℕ := 2

/-- The number of non-math courses available (excluding English) --/
def non_math_courses : ℕ := total_courses - math_courses - 1

/-- The minimum number of math courses required in a program --/
def min_math_courses : ℕ := 2

/-- Calculates the number of ways to choose a program --/
def calculate_programs : ℕ :=
  Nat.choose non_math_courses (program_size - min_math_courses - 1) +
  Nat.choose non_math_courses (program_size - math_courses - 1)

theorem program_count : calculate_programs = 6 := by sorry

end program_count_l2314_231412


namespace stating_isosceles_triangles_properties_l2314_231440

/-- 
Represents the number of isosceles triangles with vertices of the same color 
in a regular (6n+1)-gon with k red vertices and the rest blue.
-/
def P (n : ℕ) (k : ℕ) : ℕ := sorry

/-- 
Theorem stating the properties of P for a regular (6n+1)-gon 
with k red vertices and the rest blue.
-/
theorem isosceles_triangles_properties (n : ℕ) (k : ℕ) : 
  (P n (k + 1) - P n k = 3 * k - 9 * n) ∧ 
  (P n k = 3 * n * (6 * n + 1) - 9 * k * n + (3 * k * (k - 1)) / 2) := by
  sorry

end stating_isosceles_triangles_properties_l2314_231440


namespace least_sum_m_n_l2314_231450

theorem least_sum_m_n (m n : ℕ+) (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^(m : ℕ) = k * n^(n : ℕ)) (h3 : ¬ ∃ k : ℕ, m = k * n) :
  ∀ p q : ℕ+, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^(p : ℕ) = k * q^(q : ℕ)) → 
    (¬ ∃ k : ℕ, p = k * q) → 
    (m + n ≤ p + q) :=
by sorry

end least_sum_m_n_l2314_231450


namespace derivative_f_at_one_l2314_231409

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 2 := by sorry

end derivative_f_at_one_l2314_231409


namespace vector_magnitude_problem_l2314_231481

/-- Given two vectors a and b in ℝ², prove that |a - b| = 5 -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (-2, 1) →
  a + b = (-1, -2) →
  ‖a - b‖ = 5 := by
  sorry

end vector_magnitude_problem_l2314_231481


namespace bug_return_probability_l2314_231423

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The problem statement -/
theorem bug_return_probability : P 8 = 547/2187 := by
  sorry

end bug_return_probability_l2314_231423


namespace fruit_tree_ratio_l2314_231476

theorem fruit_tree_ratio (total_streets : ℕ) (plum_trees pear_trees apricot_trees : ℕ) : 
  total_streets = 18 →
  plum_trees = 3 →
  pear_trees = 3 →
  apricot_trees = 3 →
  (plum_trees + pear_trees + apricot_trees : ℚ) / total_streets = 1 / 2 := by
  sorry

end fruit_tree_ratio_l2314_231476


namespace median_and_altitude_lengths_l2314_231456

/-- Right triangle DEF with given side lengths and midpoint N -/
structure RightTriangleDEF where
  DE : ℝ
  DF : ℝ
  N : ℝ × ℝ
  is_right_angle : DE^2 + DF^2 = (DE + DF)^2 / 2
  side_lengths : DE = 6 ∧ DF = 8
  N_is_midpoint : N = ((DE + DF) / 2, DF / 2)

/-- Theorem about median and altitude lengths in the right triangle -/
theorem median_and_altitude_lengths (t : RightTriangleDEF) :
  let DN := Real.sqrt ((t.DE + t.N.1)^2 + t.N.2^2)
  let altitude := 2 * (t.DE * t.DF) / (t.DE + t.DF)
  DN = 5 ∧ altitude = 4.8 := by
  sorry


end median_and_altitude_lengths_l2314_231456


namespace pizza_combinations_l2314_231465

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 4 + Nat.choose n 3 = 126 := by
  sorry

end pizza_combinations_l2314_231465


namespace number_of_routes_P_to_Q_l2314_231448

/-- Represents the points in the diagram --/
inductive Point : Type
| P | Q | R | S | T

/-- Represents a direct path between two points --/
def DirectPath : Point → Point → Prop :=
  fun p q => match p, q with
  | Point.P, Point.R => True
  | Point.P, Point.S => True
  | Point.R, Point.T => True
  | Point.R, Point.Q => True
  | Point.S, Point.T => True
  | Point.T, Point.Q => True
  | _, _ => False

/-- Represents a route from one point to another --/
def Route : Point → Point → Type :=
  fun p q => List (Σ' x y : Point, DirectPath x y)

/-- Counts the number of routes between two points --/
def countRoutes : Point → Point → Nat :=
  fun p q => sorry

theorem number_of_routes_P_to_Q :
  countRoutes Point.P Point.Q = 3 := by sorry

end number_of_routes_P_to_Q_l2314_231448


namespace hyperbola_eccentricity_l2314_231483

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ∃ (x y : ℝ),
  x ≥ 0 ∧ 
  y = Real.sqrt x ∧ 
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  (∃ (m : ℝ), m * (x + 1) = y ∧ m = 1 / (2 * Real.sqrt x)) →
  (Real.sqrt (a^2 + b^2)) / a = (Real.sqrt 5 + 1) / 2 := by
sorry

end hyperbola_eccentricity_l2314_231483


namespace set_M_equals_three_two_four_three_one_l2314_231438

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | (m - 1)*x - 1 = 0}

-- Define the set M
def M : Set ℝ := {m : ℝ | A ∩ B m = B m}

-- Theorem statement
theorem set_M_equals_three_two_four_three_one : M = {3/2, 4/3, 1} := by
  sorry

end set_M_equals_three_two_four_three_one_l2314_231438


namespace pug_cleaning_theorem_l2314_231422

/-- The number of pugs in the first scenario -/
def num_pugs : ℕ := 4

/-- The time taken by the unknown number of pugs to clean the house -/
def time1 : ℕ := 45

/-- The number of pugs in the second scenario -/
def num_pugs2 : ℕ := 15

/-- The time taken by the known number of pugs to clean the house -/
def time2 : ℕ := 12

/-- The theorem stating that the number of pugs in the first scenario is 4 -/
theorem pug_cleaning_theorem : 
  num_pugs * time1 = num_pugs2 * time2 := by sorry

end pug_cleaning_theorem_l2314_231422


namespace coin_flip_probability_l2314_231434

theorem coin_flip_probability (n : ℕ) : n = 6 →
  (1 + n + n * (n - 1) / 2 : ℚ) / 2^n = 7/32 := by
  sorry

end coin_flip_probability_l2314_231434


namespace complex_equation_solution_l2314_231490

theorem complex_equation_solution (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i := by
  sorry

end complex_equation_solution_l2314_231490


namespace triangle_DEF_angle_D_l2314_231493

theorem triangle_DEF_angle_D (D E F : ℝ) : 
  E = 3 * F → F = 15 → D + E + F = 180 → D = 120 := by sorry

end triangle_DEF_angle_D_l2314_231493


namespace octopus_leg_configuration_l2314_231419

-- Define the possible number of legs for an octopus
inductive LegCount : Type where
  | six : LegCount
  | seven : LegCount
  | eight : LegCount

-- Define the colors of the octopuses
inductive Color : Type where
  | blue : Color
  | green : Color
  | yellow : Color
  | red : Color

-- Define a function to determine if an octopus is telling the truth
def isTruthful (legs : LegCount) : Prop :=
  match legs with
  | LegCount.six => True
  | LegCount.seven => False
  | LegCount.eight => True

-- Define a function to convert LegCount to a natural number
def legCountToNat (legs : LegCount) : Nat :=
  match legs with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

-- Define the claims made by each octopus
def claim (color : Color) : Nat :=
  match color with
  | Color.blue => 28
  | Color.green => 27
  | Color.yellow => 26
  | Color.red => 25

-- Define the theorem
theorem octopus_leg_configuration :
  ∃ (legs : Color → LegCount),
    (legs Color.green = LegCount.six) ∧
    (legs Color.blue = LegCount.seven) ∧
    (legs Color.yellow = LegCount.seven) ∧
    (legs Color.red = LegCount.seven) ∧
    (∀ c, isTruthful (legs c) ↔ (legCountToNat (legs Color.blue) + legCountToNat (legs Color.green) + legCountToNat (legs Color.yellow) + legCountToNat (legs Color.red) = claim c)) :=
sorry

end octopus_leg_configuration_l2314_231419


namespace correlation_relationships_l2314_231444

-- Define the type for relationships
inductive Relationship
  | AppleProductionClimate
  | StudentID
  | TreeDiameterHeight
  | PointCoordinates

-- Define a predicate for correlation relationships
def IsCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleProductionClimate => true
  | Relationship.TreeDiameterHeight => true
  | _ => false

-- Theorem statement
theorem correlation_relationships :
  (∀ r : Relationship, IsCorrelation r ↔ 
    (r = Relationship.AppleProductionClimate ∨ 
     r = Relationship.TreeDiameterHeight)) := by
  sorry

end correlation_relationships_l2314_231444


namespace no_integer_solution_l2314_231447

theorem no_integer_solution : ¬∃ (n : ℕ+), ∃ (k : ℤ), (n.val^(3*n.val - 2) - 3*n.val + 1) / (3*n.val - 2) = k := by
  sorry

end no_integer_solution_l2314_231447


namespace solution_set_nonempty_implies_a_range_l2314_231427

theorem solution_set_nonempty_implies_a_range 
  (h : ∃ x, |x - 3| + |x - a| < 4) : 
  -1 < a ∧ a < 7 := by
sorry

end solution_set_nonempty_implies_a_range_l2314_231427


namespace suit_price_calculation_l2314_231446

theorem suit_price_calculation (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 200 →
  increase_rate = 0.3 →
  discount_rate = 0.3 →
  original_price * (1 + increase_rate) * (1 - discount_rate) = 182 := by
  sorry

#check suit_price_calculation

end suit_price_calculation_l2314_231446


namespace total_door_replacement_cost_l2314_231420

/-- The total cost of replacing doors for John -/
theorem total_door_replacement_cost :
  let num_bedroom_doors : ℕ := 3
  let num_outside_doors : ℕ := 2
  let outside_door_cost : ℕ := 20
  let bedroom_door_cost : ℕ := outside_door_cost / 2
  let total_cost : ℕ := num_bedroom_doors * bedroom_door_cost + num_outside_doors * outside_door_cost
  total_cost = 70 := by sorry

end total_door_replacement_cost_l2314_231420


namespace geometry_relations_l2314_231442

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (a b : Line) (α β : Plane) 
  (h_different_lines : a ≠ b) 
  (h_different_planes : α ≠ β) :
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) ∧
  (perpendicular_planes α β ∧ parallel_line_plane a α → perpendicular_line_plane a β) ∧
  (perpendicular_planes α β ∧ perpendicular_line_plane a β → parallel_line_plane a α) ∧
  (perpendicular_lines a b ∧ perpendicular_line_plane a α ∧ perpendicular_line_plane b β → perpendicular_planes α β) :=
by sorry

end geometry_relations_l2314_231442


namespace parabola_inscribed_triangle_l2314_231402

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
def Parabola (p : ℝ) :=
  {point : Point | point.y^2 = 2 * p * point.x}

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_inscribed_triangle 
  (p : ℝ) 
  (parabola : Parabola p)
  (ABC : Triangle)
  (AFBC : Quadrilateral)
  (h1 : ABC.B.y = 0) -- B is on x-axis
  (h2 : ABC.C.y = 0) -- C is on x-axis
  (h3 : ABC.A.y^2 = 2 * p * ABC.A.x) -- A is on parabola
  (h4 : (ABC.B.x - ABC.A.x) * (ABC.C.x - ABC.A.x) + (ABC.B.y - ABC.A.y) * (ABC.C.y - ABC.A.y) = 0) -- ABC is right-angled
  (h5 : quadrilateralArea AFBC = 8 * p^2) -- Area of AFBC is 8p^2
  : ∃ (D : Point), triangleArea ⟨ABC.A, ABC.C, D⟩ = 15/2 * p^2 :=
sorry

end parabola_inscribed_triangle_l2314_231402


namespace no_natural_numbers_satisfying_condition_l2314_231474

theorem no_natural_numbers_satisfying_condition : 
  ¬∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y - (x + y) = 2021 :=
by sorry

end no_natural_numbers_satisfying_condition_l2314_231474


namespace steak_knife_cost_l2314_231453

/-- The number of steak knife sets -/
def num_sets : ℕ := 2

/-- The number of steak knives in each set -/
def knives_per_set : ℕ := 4

/-- The cost of each set in dollars -/
def cost_per_set : ℚ := 80

/-- The total number of steak knives -/
def total_knives : ℕ := num_sets * knives_per_set

/-- The total cost of all sets in dollars -/
def total_cost : ℚ := num_sets * cost_per_set

/-- The cost of each single steak knife in dollars -/
def cost_per_knife : ℚ := total_cost / total_knives

theorem steak_knife_cost : cost_per_knife = 20 := by
  sorry

end steak_knife_cost_l2314_231453


namespace f_multiplicative_f_derivative_positive_f_derivative_odd_f_satisfies_properties_l2314_231407

/-- A function satisfying specific properties -/
def f (x : ℝ) : ℝ := x^2

/-- Property 1: f(x₁x₂) = f(x₁)f(x₂) for all x₁, x₂ -/
theorem f_multiplicative : ∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂ := by sorry

/-- Property 2: For x ∈ (0, +∞), f'(x) > 0 -/
theorem f_derivative_positive : ∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x := by sorry

/-- Property 3: f'(x) is an odd function -/
theorem f_derivative_odd : ∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ↔ HasDerivAt f (-2 * x) x := by sorry

/-- The main theorem stating that f satisfies all properties -/
theorem f_satisfies_properties : 
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x) ∧
  (∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ↔ HasDerivAt f (-2 * x) x) := by sorry

end f_multiplicative_f_derivative_positive_f_derivative_odd_f_satisfies_properties_l2314_231407


namespace accurate_to_hundreds_place_l2314_231433

/-- Represents a number with a specified precision --/
structure PreciseNumber where
  value : ℝ
  precision : ℕ

/-- Defines what it means for a number to be accurate to a certain place value --/
def accurate_to (n : PreciseNumber) (place : ℕ) : Prop :=
  ∃ (m : ℤ), n.value = (m : ℝ) * (10 : ℝ) ^ place ∧ n.precision = place

/-- The statement to be proved --/
theorem accurate_to_hundreds_place :
  let n : PreciseNumber := ⟨4.0 * 10^3, 2⟩
  accurate_to n 2 :=
sorry

end accurate_to_hundreds_place_l2314_231433


namespace class_sports_census_suitable_l2314_231458

/-- Represents a survey --/
inductive Survey
  | LightBulbLifespan
  | ClassSportsActivity
  | YangtzeRiverFish
  | PlasticBagDisposal

/-- Represents the characteristics of a survey --/
structure SurveyCharacteristics where
  population_size : ℕ
  data_collection_time : ℕ
  resource_intensity : ℕ

/-- Determines if a survey is feasible and practical for a census --/
def is_census_suitable (s : Survey) (c : SurveyCharacteristics) : Prop :=
  c.population_size ≤ 1000 ∧ c.data_collection_time ≤ 7 ∧ c.resource_intensity ≤ 5

/-- The characteristics of the class sports activity survey --/
def class_sports_characteristics : SurveyCharacteristics :=
  { population_size := 30
  , data_collection_time := 1
  , resource_intensity := 2 }

/-- Theorem stating that the class sports activity survey is suitable for a census --/
theorem class_sports_census_suitable :
  is_census_suitable Survey.ClassSportsActivity class_sports_characteristics :=
sorry

end class_sports_census_suitable_l2314_231458


namespace mary_shirts_fraction_l2314_231485

theorem mary_shirts_fraction (blue_initial : ℕ) (brown_initial : ℕ) (total_left : ℕ) :
  blue_initial = 26 →
  brown_initial = 36 →
  total_left = 37 →
  ∃ (f : ℚ), 
    (blue_initial / 2 + brown_initial * (1 - f) = total_left) ∧
    (f = 1/3) :=
by sorry

end mary_shirts_fraction_l2314_231485


namespace estimate_nearsighted_students_l2314_231471

/-- Estimates the number of nearsighted students in a population based on a sample. -/
theorem estimate_nearsighted_students 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (nearsighted_in_sample : ℕ) 
  (h1 : total_students = 400) 
  (h2 : sample_size = 30) 
  (h3 : nearsighted_in_sample = 12) :
  ⌊(total_students : ℚ) * (nearsighted_in_sample : ℚ) / (sample_size : ℚ)⌋ = 160 :=
sorry

end estimate_nearsighted_students_l2314_231471


namespace unique_b_solution_l2314_231436

def base_83_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (83 ^ i)) 0

theorem unique_b_solution : ∃! b : ℤ, 
  (0 ≤ b ∧ b ≤ 20) ∧ 
  (∃ k : ℤ, base_83_to_decimal [2, 5, 7, 3, 6, 4, 5] - b = 17 * k) ∧
  b = 8 := by sorry

end unique_b_solution_l2314_231436


namespace percentage_of_women_in_study_group_l2314_231463

theorem percentage_of_women_in_study_group 
  (percentage_women_lawyers : Real) 
  (probability_selecting_woman_lawyer : Real) :
  let percentage_women := probability_selecting_woman_lawyer / percentage_women_lawyers
  percentage_women_lawyers = 0.4 →
  probability_selecting_woman_lawyer = 0.28 →
  percentage_women = 0.7 := by
sorry

end percentage_of_women_in_study_group_l2314_231463


namespace sufficient_but_not_necessary_l2314_231400

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ y : ℝ, y^2 - 1 > 0 ∧ ¬(y < -1)) := by
  sorry

end sufficient_but_not_necessary_l2314_231400


namespace floor_plus_self_equals_twenty_l2314_231498

theorem floor_plus_self_equals_twenty (s : ℝ) : ⌊s⌋ + s = 20 ↔ s = 10 := by sorry

end floor_plus_self_equals_twenty_l2314_231498


namespace isosceles_triangle_area_l2314_231414

/-- An isosceles triangle with altitude 10 to its base and perimeter 40 has area 75 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 →  -- positive base and side lengths
  2 * s + 2 * b = 40 →  -- perimeter condition
  s^2 = b^2 + 100 →  -- Pythagorean theorem with altitude 10
  b * 10 = 75 := by sorry

end isosceles_triangle_area_l2314_231414


namespace larger_number_problem_l2314_231478

theorem larger_number_problem (x y : ℚ) : 
  (5 * y = 6 * x) → 
  (x + y = 42) → 
  (y > x) →
  y = 252 / 11 := by
sorry

end larger_number_problem_l2314_231478


namespace cost_for_haleighs_pets_l2314_231429

/-- Calculates the cost of leggings for Haleigh's pets -/
def cost_of_leggings (dogs cats spiders parrots chickens octopuses : ℕ)
  (dog_legs cat_legs spider_legs parrot_legs chicken_legs octopus_legs : ℕ)
  (bulk_price : ℚ) (bulk_quantity : ℕ) (regular_price : ℚ) : ℚ :=
  let total_legs := dogs * dog_legs + cats * cat_legs + spiders * spider_legs +
                    parrots * parrot_legs + chickens * chicken_legs + octopuses * octopus_legs
  let total_pairs := total_legs / 2
  let bulk_sets := total_pairs / bulk_quantity
  let remaining_pairs := total_pairs % bulk_quantity
  (bulk_sets * bulk_price) + (remaining_pairs * regular_price)

theorem cost_for_haleighs_pets :
  cost_of_leggings 4 3 2 1 5 3 4 4 8 2 2 8 18 12 2 = 62 := by
  sorry

end cost_for_haleighs_pets_l2314_231429


namespace age_sum_problem_l2314_231435

theorem age_sum_problem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 256 →
  a + b + c = 38 := by
sorry

end age_sum_problem_l2314_231435


namespace bedroom_size_problem_l2314_231462

theorem bedroom_size_problem (total_area : ℝ) (difference : ℝ) :
  total_area = 300 →
  difference = 60 →
  ∃ (smaller larger : ℝ),
    smaller + larger = total_area ∧
    larger = smaller + difference ∧
    smaller = 120 := by
  sorry

end bedroom_size_problem_l2314_231462


namespace sector_area_for_unit_radian_l2314_231482

theorem sector_area_for_unit_radian (arc_length : Real) (h : arc_length = 6) :
  let radius := arc_length  -- From definition of radian: 1 = arc_length / radius
  let sector_area := (1 / 2) * radius * arc_length
  sector_area = 18 := by
  sorry

end sector_area_for_unit_radian_l2314_231482


namespace power_minus_self_even_l2314_231457

theorem power_minus_self_even (a n : ℕ+) : 
  ∃ k : ℤ, (a^n.val - a : ℤ) = 2 * k := by sorry

end power_minus_self_even_l2314_231457


namespace probability_A_B_same_group_l2314_231499

-- Define the score ranges and their frequencies
def score_ranges : List (ℕ × ℕ × ℕ) := [
  (60, 75, 2),
  (75, 90, 3),
  (90, 105, 14),
  (105, 120, 15),
  (120, 135, 12),
  (135, 150, 4)
]

-- Define the total number of students
def total_students : ℕ := 50

-- Define student A's score
def score_A : ℕ := 62

-- Define student B's score
def score_B : ℕ := 140

-- Define the "two-help-one" group formation rule
def two_help_one (s1 s2 s3 : ℕ) : Prop :=
  (s1 ≥ 135 ∧ s1 ≤ 150) ∧ (s2 ≥ 135 ∧ s2 ≤ 150) ∧ (s3 ≥ 60 ∧ s3 < 75)

-- Theorem to prove
theorem probability_A_B_same_group :
  ∃ (p : ℚ), p = 1/4 ∧ 
  (p = (number_of_groups_with_A_and_B : ℚ) / (total_number_of_possible_groups : ℚ)) :=
sorry

end probability_A_B_same_group_l2314_231499


namespace max_area_rectangular_garden_l2314_231406

/-- The maximum area of a rectangular garden with a perimeter of 168 feet and natural number side lengths --/
theorem max_area_rectangular_garden :
  ∃ (w h : ℕ), 
    w + h = 84 ∧ 
    (∀ (x y : ℕ), x + y = 84 → x * y ≤ w * h) ∧
    w * h = 1764 := by
  sorry

end max_area_rectangular_garden_l2314_231406


namespace train_length_calculation_l2314_231459

-- Define the given parameters
def train_speed : ℝ := 60  -- km/h
def man_speed : ℝ := 6     -- km/h
def passing_time : ℝ := 12 -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed : ℝ := train_speed + man_speed
  let relative_speed_mps : ℝ := relative_speed * (5 / 18)
  let train_length : ℝ := relative_speed_mps * passing_time
  train_length = 220 := by sorry

end train_length_calculation_l2314_231459


namespace least_subtraction_for_divisibility_l2314_231461

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 9 ∧ (101054 - k) % 10 = 0 ∧ ∀ (m : ℕ), m < k → (101054 - m) % 10 ≠ 0 := by
  sorry

end least_subtraction_for_divisibility_l2314_231461


namespace largest_n_is_69_l2314_231432

/-- Represents a three-digit number in a given base --/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a three-digit number from a given base to base 10 --/
def to_base_10 (base : ℕ) (num : ThreeDigitNumber base) : ℕ :=
  num.hundreds * base^2 + num.tens * base + num.ones

theorem largest_n_is_69 :
  ∀ (n : ℕ) (base_5 : ThreeDigitNumber 5) (base_9 : ThreeDigitNumber 9),
    n > 0 →
    to_base_10 5 base_5 = n →
    to_base_10 9 base_9 = n →
    base_5.hundreds = base_9.ones →
    base_5.tens = base_9.tens →
    base_5.ones = base_9.hundreds →
    n ≤ 69 :=
sorry

end largest_n_is_69_l2314_231432


namespace quadratic_inequality_l2314_231401

theorem quadratic_inequality (z : ℝ) : z^2 - 40*z + 360 ≤ 16*z ↔ 8 ≤ z ∧ z ≤ 45 := by
  sorry

end quadratic_inequality_l2314_231401


namespace quadratic_distinct_roots_l2314_231431

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) → m < 1 := by
  sorry

end quadratic_distinct_roots_l2314_231431


namespace davids_math_marks_l2314_231443

def english_marks : ℝ := 74
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 90
def average_marks : ℝ := 75.6
def num_subjects : ℕ := 5

theorem davids_math_marks :
  let total_marks := average_marks * num_subjects
  let known_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks
  math_marks = 65 := by sorry

end davids_math_marks_l2314_231443


namespace approximation_of_2026_l2314_231424

def approximate_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem approximation_of_2026 :
  approximate_to_hundredth 2.026 = 2.03 := by
  sorry

end approximation_of_2026_l2314_231424


namespace art_collection_cost_l2314_231464

theorem art_collection_cost (price_first_three : ℝ) (price_fourth : ℝ) : 
  price_first_three = 45000 →
  price_fourth = (price_first_three / 3) * 1.5 →
  price_first_three + price_fourth = 67500 := by
sorry

end art_collection_cost_l2314_231464


namespace geometric_sequence_a_value_l2314_231487

theorem geometric_sequence_a_value (a : ℝ) :
  (1 / (a - 1)) * (a + 1) = (a + 1) * (a^2 - 1) →
  a = 0 := by
sorry

end geometric_sequence_a_value_l2314_231487


namespace ryan_marbles_count_l2314_231477

/-- The number of friends Ryan shares his marbles with -/
def num_friends : ℕ := 9

/-- The number of marbles each friend receives -/
def marbles_per_friend : ℕ := 8

/-- Ryan's total number of marbles -/
def total_marbles : ℕ := num_friends * marbles_per_friend

theorem ryan_marbles_count : total_marbles = 72 := by
  sorry

end ryan_marbles_count_l2314_231477


namespace no_bounded_function_satisfying_inequality_l2314_231484

theorem no_bounded_function_satisfying_inequality :
  ¬∃ (f : ℝ → ℝ), 
    (∃ (M : ℝ), ∀ x, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y, (f (x + y))^2 ≥ (f x)^2 + 2*(f (x*y)) + (f y)^2) := by
  sorry

end no_bounded_function_satisfying_inequality_l2314_231484


namespace line_arrangements_l2314_231491

theorem line_arrangements (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 :=
by sorry

end line_arrangements_l2314_231491


namespace find_a_l2314_231460

def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 2

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 4 * (2 * x + a)) ∧ (deriv (f a)) 2 = 20 → a = 1 := by
  sorry

end find_a_l2314_231460


namespace min_value_expression_min_value_achievable_l2314_231437

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) :
  (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b ≥ Real.sqrt (8 / 3) := by
  sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a ≠ 0 ∧ (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b = Real.sqrt (8 / 3) := by
  sorry

end min_value_expression_min_value_achievable_l2314_231437


namespace urn_problem_l2314_231418

theorem urn_problem (M : ℕ) : 
  (5 / 12 : ℚ) * (10 / (10 + M)) + (7 / 12 : ℚ) * (M / (10 + M)) = 62 / 100 → M = 7 :=
by sorry

end urn_problem_l2314_231418


namespace parabola_points_m_range_l2314_231404

/-- The parabola equation -/
def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3

theorem parabola_points_m_range (a x₁ x₂ y₁ y₂ m : ℝ) : 
  a ≠ 0 →
  parabola a x₁ = y₁ →
  parabola a x₂ = y₂ →
  -2 < x₁ →
  x₁ < 0 →
  m < x₂ →
  x₂ < m + 1 →
  y₁ ≠ y₂ →
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end parabola_points_m_range_l2314_231404


namespace star_neg_two_three_l2314_231473

-- Define the new operation ※
def star (a b : ℤ) : ℤ := a^2 + 2*a*b

-- Theorem statement
theorem star_neg_two_three : star (-2) 3 = -8 := by
  sorry

end star_neg_two_three_l2314_231473


namespace min_n_plus_d_for_arithmetic_sequence_l2314_231475

/-- An arithmetic sequence with positive integral terms -/
def ArithmeticSequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem min_n_plus_d_for_arithmetic_sequence :
  ∀ a : ℕ → ℕ,
  ∀ d : ℕ,
  ArithmeticSequence a d →
  a 1 = 1 →
  (∃ n : ℕ, a n = 51) →
  (∃ n d : ℕ, ArithmeticSequence a d ∧ a 1 = 1 ∧ a n = 51 ∧ n + d = 16 ∧
    ∀ m k : ℕ, ArithmeticSequence a k ∧ a 1 = 1 ∧ a m = 51 → m + k ≥ 16) :=
by sorry

end min_n_plus_d_for_arithmetic_sequence_l2314_231475


namespace cube_less_than_three_times_square_l2314_231452

theorem cube_less_than_three_times_square (x : ℤ) :
  x^3 < 3*x^2 ↔ x = 1 ∨ x = 2 := by
  sorry

end cube_less_than_three_times_square_l2314_231452


namespace triangle_side_length_l2314_231405

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The theorem stating the relationship between sides and angles in the given triangle -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 3)
  (h3 : 3 * t.α + 2 * t.β = Real.pi)
  (h4 : t.α + t.β + t.γ = Real.pi)
  (h5 : 0 < t.a ∧ 0 < t.b ∧ 0 < t.c)
  (h6 : 0 < t.α ∧ 0 < t.β ∧ 0 < t.γ)
  (h7 : t.a / (Real.sin t.α) = t.b / (Real.sin t.β))
  (h8 : t.b / (Real.sin t.β) = t.c / (Real.sin t.γ)) :
  t.c = 4 := by
  sorry


end triangle_side_length_l2314_231405


namespace wall_width_calculation_l2314_231411

theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) :
  mirror_side = 18 →
  wall_length = 20.25 →
  (mirror_side * mirror_side) * 2 = wall_length * (648 / wall_length) :=
by
  sorry

#check wall_width_calculation

end wall_width_calculation_l2314_231411


namespace complex_fraction_simplification_l2314_231416

theorem complex_fraction_simplification :
  (Complex.I - 1) / (1 + Complex.I) = Complex.I := by
  sorry

end complex_fraction_simplification_l2314_231416


namespace sparrow_swallow_system_l2314_231430

/-- Represents the weight of a sparrow in taels -/
def sparrow_weight : ℝ := sorry

/-- Represents the weight of a swallow in taels -/
def swallow_weight : ℝ := sorry

/-- The total weight of five sparrows and six swallows is 16 taels -/
axiom total_weight : 5 * sparrow_weight + 6 * swallow_weight = 16

/-- Exchanging one sparrow with one swallow results in equal weights for both groups -/
axiom exchange_equal : 4 * sparrow_weight + swallow_weight = 5 * swallow_weight + sparrow_weight

/-- The system of equations representing the sparrow and swallow weight problem -/
theorem sparrow_swallow_system :
  (5 * sparrow_weight + 6 * swallow_weight = 16) ∧
  (4 * sparrow_weight + swallow_weight = 5 * swallow_weight + sparrow_weight) :=
sorry

end sparrow_swallow_system_l2314_231430


namespace race_distance_difference_l2314_231426

/-- In a race scenario where:
  * The race distance is 240 meters
  * Runner A finishes in 23 seconds
  * Runner A beats runner B by 7 seconds
This theorem proves that A beats B by 56 meters -/
theorem race_distance_difference (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) :
  race_distance = 240 ∧ 
  a_time = 23 ∧ 
  time_difference = 7 →
  (race_distance - (race_distance / (a_time + time_difference)) * a_time) = 56 :=
by sorry

end race_distance_difference_l2314_231426


namespace brick_length_is_20_l2314_231417

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 27

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks required for the wall -/
def num_bricks : ℕ := 27000

/-- Conversion factor from cubic meters to cubic centimeters -/
def m3_to_cm3 : ℝ := 1000000

theorem brick_length_is_20 :
  brick_length = 20 ∧
  brick_width = 10 ∧
  brick_height = 7.5 ∧
  wall_length = 27 ∧
  wall_width = 2 ∧
  wall_height = 0.75 ∧
  num_bricks = 27000 →
  wall_length * wall_width * wall_height * m3_to_cm3 =
    brick_length * brick_width * brick_height * num_bricks :=
by sorry

end brick_length_is_20_l2314_231417


namespace binary_digit_difference_l2314_231451

theorem binary_digit_difference : ∃ (n m : ℕ), n = 400 ∧ m = 1600 ∧ 
  (Nat.log 2 m + 1) - (Nat.log 2 n + 1) = 2 := by
  sorry

end binary_digit_difference_l2314_231451


namespace complex_fraction_sum_l2314_231488

theorem complex_fraction_sum (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end complex_fraction_sum_l2314_231488


namespace puzzle_pieces_left_l2314_231449

theorem puzzle_pieces_left (total_pieces : ℕ) (num_children : ℕ) (reyn_pieces : ℕ) : 
  total_pieces = 500 →
  num_children = 4 →
  reyn_pieces = 25 →
  total_pieces - (reyn_pieces + 2*reyn_pieces + 3*reyn_pieces + 4*reyn_pieces) = 250 := by
sorry

end puzzle_pieces_left_l2314_231449


namespace geometric_sequence_unique_solution_l2314_231455

/-- A geometric sequence is defined by its first term and common ratio. -/
structure GeometricSequence where
  first_term : ℚ
  common_ratio : ℚ

/-- Get the nth term of a geometric sequence. -/
def GeometricSequence.nth_term (seq : GeometricSequence) (n : ℕ) : ℚ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem geometric_sequence_unique_solution :
  ∃! (seq : GeometricSequence),
    seq.nth_term 2 = 37 + 1/3 ∧
    seq.nth_term 6 = 2 + 1/3 ∧
    seq.first_term = 74 + 2/3 ∧
    seq.common_ratio = 1/2 := by
  sorry

end geometric_sequence_unique_solution_l2314_231455


namespace min_groups_for_class_l2314_231468

theorem min_groups_for_class (total_students : ℕ) (max_group_size : ℕ) (h1 : total_students = 30) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ) (group_size : ℕ),
    num_groups * group_size = total_students ∧
    group_size ≤ max_group_size ∧
    (∀ (other_num_groups : ℕ) (other_group_size : ℕ),
      other_num_groups * other_group_size = total_students →
      other_group_size ≤ max_group_size →
      num_groups ≤ other_num_groups) ∧
    num_groups = 3 :=
by
  sorry

end min_groups_for_class_l2314_231468


namespace die_roll_probability_l2314_231480

def roll_outcome := Fin 6

def is_valid_outcome (m n : roll_outcome) : Prop :=
  m.val + 1 = 2 * (n.val + 1)

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 3

theorem die_roll_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 :=
sorry

end die_roll_probability_l2314_231480


namespace no_solution_implies_a_range_l2314_231466

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ∈ Set.Iic 8 :=
by
  sorry

end no_solution_implies_a_range_l2314_231466


namespace expression_evaluation_l2314_231413

theorem expression_evaluation : (-1)^2 + (1/2 - 7/12 + 5/6) = 7/4 := by
  sorry

end expression_evaluation_l2314_231413


namespace square_side_length_from_hexagons_l2314_231445

/-- The side length of a square formed by repositioning two congruent hexagons cut from a rectangle -/
def square_side_length (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  sorry

/-- The height of each hexagon cut from the rectangle -/
def hexagon_height (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  sorry

theorem square_side_length_from_hexagons
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (h1 : rectangle_width = 9)
  (h2 : rectangle_height = 27)
  (h3 : square_side_length rectangle_width rectangle_height =
        2 * hexagon_height rectangle_width rectangle_height) :
  square_side_length rectangle_width rectangle_height = 9 * Real.sqrt 3 :=
by sorry

end square_side_length_from_hexagons_l2314_231445


namespace rope_cutting_problem_l2314_231486

/-- The number of ropes after n cuts -/
def num_ropes (n : ℕ) : ℕ := 1 + 4 * n

/-- The problem statement -/
theorem rope_cutting_problem :
  ∃ n : ℕ, num_ropes n = 2021 ∧ n = 505 := by sorry

end rope_cutting_problem_l2314_231486


namespace investment_growth_l2314_231470

/-- Calculates the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that an initial investment of $313,021.70 at 6% annual interest for 15 years 
    results in approximately $750,000 -/
theorem investment_growth :
  let initial_investment : ℝ := 313021.70
  let interest_rate : ℝ := 0.06
  let years : ℕ := 15
  let target_amount : ℝ := 750000
  
  abs (future_value initial_investment interest_rate years - target_amount) < 1 := by
  sorry


end investment_growth_l2314_231470


namespace cube_root_of_negative_64_l2314_231439

theorem cube_root_of_negative_64 : ∃ b : ℝ, b^3 = -64 ∧ b = -4 := by sorry

end cube_root_of_negative_64_l2314_231439


namespace equal_probability_for_all_probability_independent_of_method_l2314_231492

/-- The probability of selecting a product given the described selection method -/
def selection_probability (total : ℕ) (remove : ℕ) (select : ℕ) : ℚ :=
  select / total

/-- The selection method ensures equal probability for all products -/
theorem equal_probability_for_all (total : ℕ) (remove : ℕ) (select : ℕ) 
  (h1 : total = 2003)
  (h2 : remove = 3)
  (h3 : select = 50) :
  selection_probability total remove select = 50 / 2003 := by
  sorry

/-- The probability is independent of the specific selection method -/
theorem probability_independent_of_method 
  (simple_random_sampling : (ℕ → ℕ → ℕ → ℚ))
  (systematic_sampling : (ℕ → ℕ → ℕ → ℚ))
  (total : ℕ) (remove : ℕ) (select : ℕ)
  (h1 : total = 2003)
  (h2 : remove = 3)
  (h3 : select = 50) :
  simple_random_sampling total remove select = systematic_sampling (total - remove) select select ∧
  simple_random_sampling total remove select = selection_probability total remove select := by
  sorry

end equal_probability_for_all_probability_independent_of_method_l2314_231492
