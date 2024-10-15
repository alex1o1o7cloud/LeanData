import Mathlib

namespace NUMINAMATH_CALUDE_line_intersecting_ellipse_l1416_141675

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define a line by its slope and y-intercept
def line_equation (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define what it means for a point to be a midpoint of two other points
def is_midpoint (x₁ y₁ x₂ y₂ x y : ℝ) : Prop := 
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

theorem line_intersecting_ellipse (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ → 
  is_on_ellipse x₂ y₂ → 
  is_midpoint x₁ y₁ x₂ y₂ 1 (1/2) →
  ∃ k b, line_equation k b x₁ y₁ ∧ line_equation k b x₂ y₂ ∧ k = -1 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_line_intersecting_ellipse_l1416_141675


namespace NUMINAMATH_CALUDE_probability_to_reach_target_is_79_1024_l1416_141649

/-- Represents a step direction --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- Represents a position on the coordinate plane --/
structure Position :=
  (x : Int) (y : Int)

/-- The probability of a single step in any direction --/
def stepProbability : ℚ := 1/4

/-- The starting position --/
def start : Position := ⟨0, 0⟩

/-- The target position --/
def target : Position := ⟨3, 1⟩

/-- The maximum number of steps allowed --/
def maxSteps : ℕ := 6

/-- Calculates the probability of reaching the target position in at most maxSteps steps --/
noncomputable def probabilityToReachTarget (start : Position) (target : Position) (maxSteps : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove --/
theorem probability_to_reach_target_is_79_1024 :
  probabilityToReachTarget start target maxSteps = 79/1024 :=
by sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_is_79_1024_l1416_141649


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1416_141606

def A : Set ℝ := {-2, -1, 0, 1, 2}

def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

theorem intersection_complement_equality : A ∩ (Set.univ \ B) = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1416_141606


namespace NUMINAMATH_CALUDE_lineup_combinations_count_l1416_141657

/-- The number of ways to choose 6 players from 15 players for 6 specific positions -/
def lineup_combinations : ℕ := 15 * 14 * 13 * 12 * 11 * 10

/-- Theorem stating that the number of ways to choose 6 players from 15 players for 6 specific positions is 3,603,600 -/
theorem lineup_combinations_count : lineup_combinations = 3603600 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_count_l1416_141657


namespace NUMINAMATH_CALUDE_hexagonal_tiling_chromatic_number_l1416_141665

/-- A type representing colors -/
inductive Color
| Red
| Green
| Blue

/-- A type representing a hexagonal tile in the plane -/
structure HexTile :=
  (id : ℕ)

/-- A function type that assigns colors to hexagonal tiles -/
def Coloring := HexTile → Color

/-- Predicate to check if two hexagonal tiles are adjacent (share a side) -/
def adjacent : HexTile → HexTile → Prop := sorry

/-- Predicate to check if a coloring is valid (no adjacent tiles have the same color) -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ h1 h2, adjacent h1 h2 → c h1 ≠ c h2

/-- The main theorem: The minimum number of colors needed is 3 -/
theorem hexagonal_tiling_chromatic_number :
  (∃ c : Coloring, valid_coloring c) ∧
  (∀ c : Coloring, valid_coloring c → (Set.range c).ncard ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_hexagonal_tiling_chromatic_number_l1416_141665


namespace NUMINAMATH_CALUDE_permutation_equality_l1416_141692

-- Define the permutation function
def A (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

-- State the theorem
theorem permutation_equality (n : ℕ) :
  A (2 * n) ^ 3 = 9 * (A n) ^ 3 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equality_l1416_141692


namespace NUMINAMATH_CALUDE_triangle_solutions_l1416_141635

/-- Function to determine the number of triangle solutions given two sides and an angle --/
def triangleSolutionsCount (a b : ℝ) (angleA : Real) : Nat :=
  sorry

theorem triangle_solutions :
  (triangleSolutionsCount 5 4 (120 * π / 180) = 1) ∧
  (triangleSolutionsCount 7 14 (150 * π / 180) = 0) ∧
  (triangleSolutionsCount 9 10 (60 * π / 180) = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_solutions_l1416_141635


namespace NUMINAMATH_CALUDE_expression_evaluation_l1416_141604

theorem expression_evaluation : 
  (-Real.sqrt 27 + Real.cos (30 * π / 180) - (π - Real.sqrt 2) ^ 0 + (-1/2)⁻¹) = 
  -(5 * Real.sqrt 3 + 6) / 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1416_141604


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1416_141645

def f (x : ℝ) : ℝ := x^2 - 6*x + 13

theorem quadratic_minimum :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1416_141645


namespace NUMINAMATH_CALUDE_parabola_equation_l1416_141673

/-- A parabola with vertex at the origin and directrix x = -2 has the equation y^2 = 8x -/
theorem parabola_equation (x y : ℝ) : 
  (∀ p : ℝ, p > 0 → 
    (x - p)^2 + y^2 = (x + p)^2 ∧ 
    p = 2) → 
  y^2 = 8*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1416_141673


namespace NUMINAMATH_CALUDE_seed_distribution_l1416_141601

theorem seed_distribution (n : ℕ) : 
  (n * (n + 1) / 2 : ℚ) + 100 = n * (3 * n + 1) / 2 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_seed_distribution_l1416_141601


namespace NUMINAMATH_CALUDE_roots_equation_sum_l1416_141631

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 4*α - 1 = 0 → β^2 - 4*β - 1 = 0 → 3*α^3 + 4*β^2 = 80 + 35*α := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l1416_141631


namespace NUMINAMATH_CALUDE_oil_production_fraction_l1416_141681

def initial_concentration : ℝ := 0.02
def first_replacement : ℝ := 0.03
def second_replacement : ℝ := 0.015

theorem oil_production_fraction (x : ℝ) 
  (hx_pos : x > 0)
  (hx_le_one : x ≤ 1)
  (h_first_replacement : initial_concentration * (1 - x) + first_replacement * x = initial_concentration + x * (first_replacement - initial_concentration))
  (h_second_replacement : (initial_concentration + x * (first_replacement - initial_concentration)) * (1 - x) + second_replacement * x = initial_concentration) :
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_oil_production_fraction_l1416_141681


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1416_141614

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1416_141614


namespace NUMINAMATH_CALUDE_system_solution_exists_l1416_141697

theorem system_solution_exists : ∃ (x y : ℝ), 
  (x * Real.sqrt (x * y) + y * Real.sqrt (x * y) = 10) ∧ 
  (x^2 + y^2 = 17) := by
sorry

end NUMINAMATH_CALUDE_system_solution_exists_l1416_141697


namespace NUMINAMATH_CALUDE_sum_of_y_values_l1416_141652

theorem sum_of_y_values (x y z : ℝ) : 
  x + y = 7 → 
  x * z = -180 → 
  (x + y + z)^2 = 4 → 
  ∃ y₁ y₂ : ℝ, 
    (x + y₁ = 7 ∧ x * z = -180 ∧ (x + y₁ + z)^2 = 4) ∧
    (x + y₂ = 7 ∧ x * z = -180 ∧ (x + y₂ + z)^2 = 4) ∧
    y₁ ≠ y₂ ∧
    -(y₁ + y₂) = 42 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l1416_141652


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_b_l1416_141655

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 350 * k) :
  Int.gcd (2 * b^3 + 3 * b^2 + 5 * b + 70) b = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_b_l1416_141655


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1416_141662

/-- Given a workshop with workers, prove that the average salary is 8000 --/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 30)
  (h2 : technicians = 10)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l1416_141662


namespace NUMINAMATH_CALUDE_largest_n_for_inequalities_l1416_141607

theorem largest_n_for_inequalities : ∃ (n : ℕ), n = 4 ∧ 
  (∃ (x : ℝ), ∀ (k : ℕ), k ≤ n → (k : ℝ) < x^k ∧ x^k < (k + 1 : ℝ)) ∧
  (∀ (m : ℕ), m > n → ¬∃ (x : ℝ), ∀ (k : ℕ), k ≤ m → (k : ℝ) < x^k ∧ x^k < (k + 1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_inequalities_l1416_141607


namespace NUMINAMATH_CALUDE_hexagon_dimension_theorem_l1416_141625

/-- Represents a hexagon with dimension y -/
structure Hexagon :=
  (y : ℝ)

/-- Represents a rectangle with length and width -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)

/-- Represents a square with side length -/
structure Square :=
  (side : ℝ)

/-- The theorem stating that for an 8x18 rectangle cut into two congruent hexagons 
    that can be repositioned to form a square, the dimension y of the hexagon is 6 -/
theorem hexagon_dimension_theorem (rect : Rectangle) (hex1 hex2 : Hexagon) (sq : Square) :
  rect.length = 18 ∧ 
  rect.width = 8 ∧
  hex1 = hex2 ∧
  rect.length * rect.width = sq.side * sq.side →
  hex1.y = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_dimension_theorem_l1416_141625


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1416_141605

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the line l
def line_l (x y : ℝ) (k₁ : ℝ) : Prop := y = k₁ * (x + 2)

-- Define the point M
def point_M : ℝ × ℝ := (-2, 0)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_line_intersection 
  (P₁ P₂ P : ℝ × ℝ) 
  (k₁ k₂ : ℝ) 
  (h₁ : k₁ ≠ 0)
  (h₂ : ellipse P₁.1 P₁.2)
  (h₃ : ellipse P₂.1 P₂.2)
  (h₄ : line_l P₁.1 P₁.2 k₁)
  (h₅ : line_l P₂.1 P₂.2 k₁)
  (h₆ : P = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2))
  (h₇ : k₂ = P.2 / P.1) :
  k₁ * k₂ = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1416_141605


namespace NUMINAMATH_CALUDE_farm_chickens_count_l1416_141633

theorem farm_chickens_count (chicken_A duck_A chicken_B duck_B : ℕ) : 
  chicken_A + duck_A = 625 →
  chicken_B + duck_B = 748 →
  chicken_B = (chicken_A * 124) / 100 →
  duck_A = (duck_B * 85) / 100 →
  chicken_B = 248 := by
  sorry

end NUMINAMATH_CALUDE_farm_chickens_count_l1416_141633


namespace NUMINAMATH_CALUDE_minyoung_line_size_l1416_141642

/-- Represents a line of people ordered by height -/
structure HeightLine where
  people : ℕ
  tallestToShortest : Fin people → Fin people

/-- A person's position from the tallest in the line -/
def positionFromTallest (line : HeightLine) (person : Fin line.people) : ℕ :=
  line.tallestToShortest person + 1

/-- A person's position from the shortest in the line -/
def positionFromShortest (line : HeightLine) (person : Fin line.people) : ℕ :=
  line.people - line.tallestToShortest person

theorem minyoung_line_size
  (line : HeightLine)
  (minyoung : Fin line.people)
  (h1 : positionFromTallest line minyoung = 2)
  (h2 : positionFromShortest line minyoung = 4) :
  line.people = 5 := by
  sorry

end NUMINAMATH_CALUDE_minyoung_line_size_l1416_141642


namespace NUMINAMATH_CALUDE_waiter_initial_customers_l1416_141632

/-- Calculates the initial number of customers in a waiter's section --/
def initial_customers (tables : ℕ) (people_per_table : ℕ) (left_customers : ℕ) : ℕ :=
  tables * people_per_table + left_customers

/-- Theorem: The initial number of customers in the waiter's section was 62 --/
theorem waiter_initial_customers :
  initial_customers 5 9 17 = 62 := by
  sorry

end NUMINAMATH_CALUDE_waiter_initial_customers_l1416_141632


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1416_141641

theorem solve_exponential_equation (x : ℝ) (h : x ≠ 0) :
  x^(-(2/3) : ℝ) = 4 ↔ x = 1/8 ∨ x = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1416_141641


namespace NUMINAMATH_CALUDE_right_triangle_sin_cos_relation_l1416_141630

theorem right_triangle_sin_cos_relation (A B C : ℝ) (h1 : 0 < A) (h2 : A < π / 2) :
  Real.cos B = 0 → 3 * Real.sin A = 4 * Real.cos A → Real.sin A = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_cos_relation_l1416_141630


namespace NUMINAMATH_CALUDE_propositions_truth_l1416_141621

theorem propositions_truth (a b c : ℝ) (k : ℕ+) :
  (a > b → a^(k : ℝ) > b^(k : ℝ)) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_l1416_141621


namespace NUMINAMATH_CALUDE_find_X_l1416_141619

theorem find_X : ∃ X : ℚ, (X + 43 / 151) * 151 = 2912 ∧ X = 19 := by
  sorry

end NUMINAMATH_CALUDE_find_X_l1416_141619


namespace NUMINAMATH_CALUDE_geometric_sum_property_l1416_141613

-- Define a geometric sequence with positive terms and common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sum_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum : a 1 + a 2 + a 3 = 21) : 
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_property_l1416_141613


namespace NUMINAMATH_CALUDE_arcsin_one_eq_pi_div_two_l1416_141683

-- Define arcsin function
noncomputable def arcsin (x : ℝ) : ℝ :=
  Real.arcsin x

-- State the theorem
theorem arcsin_one_eq_pi_div_two :
  arcsin 1 = π / 2 :=
sorry

end NUMINAMATH_CALUDE_arcsin_one_eq_pi_div_two_l1416_141683


namespace NUMINAMATH_CALUDE_quadratic_equation_has_real_root_l1416_141690

theorem quadratic_equation_has_real_root (a b : ℝ) : 
  ∃ x : ℝ, x^2 + a*x + b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_real_root_l1416_141690


namespace NUMINAMATH_CALUDE_same_terminal_side_l1416_141688

theorem same_terminal_side (k : ℤ) : 
  let angles : List ℝ := [-5*π/3, 2*π/3, 4*π/3, 5*π/3]
  let target : ℝ := -π/3
  let same_side (α : ℝ) : Prop := ∃ n : ℤ, α = 2*π*n + target
  ∀ α ∈ angles, same_side α ↔ α = 5*π/3 :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1416_141688


namespace NUMINAMATH_CALUDE_total_pencils_l1416_141651

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of Chloe's friends who bought the same color box -/
def friends : ℕ := 5

/-- The total number of people who bought color boxes (Chloe and her friends) -/
def total_people : ℕ := friends + 1

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

theorem total_pencils :
  pencils_per_box * total_people = 42 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l1416_141651


namespace NUMINAMATH_CALUDE_nine_possible_H_values_l1416_141694

/-- A function that represents the number formed by digits E, F, G, G, F --/
def EFGGF (E F G : Nat) : Nat := 10000 * E + 1000 * F + 100 * G + 10 * G + F

/-- A function that represents the number formed by digits F, G, E, E, H --/
def FGEEH (F G E H : Nat) : Nat := 10000 * F + 1000 * G + 100 * E + 10 * E + H

/-- A function that represents the number formed by digits H, F, H, H, H --/
def HFHHH (H F : Nat) : Nat := 10000 * H + 1000 * F + 100 * H + 10 * H + H

/-- The main theorem stating that there are exactly 9 possible values for H --/
theorem nine_possible_H_values (E F G H : Nat) :
  (E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10) →  -- E, F, G, H are digits
  (E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H) →  -- E, F, G, H are distinct
  (EFGGF E F G + FGEEH F G E H = HFHHH H F) →  -- The addition equation
  (∃! (s : Finset Nat), s.card = 9 ∧ ∀ h, h ∈ s ↔ ∃ E F G, EFGGF E F G + FGEEH F G E h = HFHHH h F) :=
by sorry


end NUMINAMATH_CALUDE_nine_possible_H_values_l1416_141694


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1416_141696

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the original point A -/
def A (m : ℝ) : Point :=
  { x := -5 * m, y := 2 * m - 1 }

/-- Moves a point up by a given amount -/
def moveUp (p : Point) (amount : ℝ) : Point :=
  { x := p.x, y := p.y + amount }

/-- Checks if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Main theorem -/
theorem point_in_fourth_quadrant (m : ℝ) :
  (moveUp (A m) 3).y = 0 → isInFourthQuadrant (A m) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1416_141696


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l1416_141669

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6,7,8}

-- Define set A
def A : Finset Nat := {1,4,6}

-- Define set B
def B : Finset Nat := {4,5,7}

-- Theorem statement
theorem complement_intersection_equals_set :
  (U \ A) ∩ (U \ B) = {2,3,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l1416_141669


namespace NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_l1416_141617

theorem max_third_term_arithmetic_sequence (a d : ℕ) : 
  a > 0 → d > 0 → a + (a + d) + (a + 2*d) + (a + 3*d) = 50 → 
  ∀ (b e : ℕ), b > 0 → e > 0 → b + (b + e) + (b + 2*e) + (b + 3*e) = 50 → 
  (a + 2*d) ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_l1416_141617


namespace NUMINAMATH_CALUDE_field_trip_vans_l1416_141698

theorem field_trip_vans (total_people : ℕ) (num_buses : ℕ) (people_per_bus : ℕ) (people_per_van : ℕ) :
  total_people = 180 →
  num_buses = 8 →
  people_per_bus = 18 →
  people_per_van = 6 →
  ∃ (num_vans : ℕ), num_vans = 6 ∧ total_people = num_buses * people_per_bus + num_vans * people_per_van :=
by sorry

end NUMINAMATH_CALUDE_field_trip_vans_l1416_141698


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l1416_141647

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 9 ∧ 
  ∃ m : ℕ, p + 2 = m^2 ∧
  m = n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l1416_141647


namespace NUMINAMATH_CALUDE_apple_boxes_theorem_l1416_141674

/-- Calculates the number of boxes of apples after removing rotten ones -/
def calculate_apple_boxes (apples_per_crate : ℕ) (num_crates : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  ((apples_per_crate * num_crates) - rotten_apples) / apples_per_box

/-- Theorem: Given the problem conditions, the number of boxes of apples is 100 -/
theorem apple_boxes_theorem :
  calculate_apple_boxes 180 12 160 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_theorem_l1416_141674


namespace NUMINAMATH_CALUDE_magic_square_y_value_l1416_141672

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (d : ℕ) (e : ℕ) (f : ℕ)
  (g : ℕ) (h : ℕ) (i : ℕ)

/-- Checks if a given 3x3 square is a magic square -/
def is_magic_square (s : MagicSquare) : Prop :=
  let sum := s.a + s.b + s.c
  sum = s.d + s.e + s.f ∧
  sum = s.g + s.h + s.i ∧
  sum = s.a + s.d + s.g ∧
  sum = s.b + s.e + s.h ∧
  sum = s.c + s.f + s.i ∧
  sum = s.a + s.e + s.i ∧
  sum = s.c + s.e + s.g

theorem magic_square_y_value (y : ℕ) :
  ∃ (s : MagicSquare), 
    is_magic_square s ∧ 
    s.a = y ∧ s.b = 25 ∧ s.c = 70 ∧ 
    s.d = 5 → 
    y = 90 := by sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l1416_141672


namespace NUMINAMATH_CALUDE_challenging_polynomial_theorem_l1416_141609

/-- Defines a quadratic polynomial q(x) = x^2 + bx + c -/
def q (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- Defines the composition q(q(x)) -/
def q_comp (b c : ℝ) (x : ℝ) : ℝ := q b c (q b c x)

/-- States that q(q(x)) = 1 has exactly four distinct real solutions -/
def has_four_solutions (b c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    q_comp b c x₁ = 1 ∧ q_comp b c x₂ = 1 ∧ q_comp b c x₃ = 1 ∧ q_comp b c x₄ = 1 ∧
    ∀ (y : ℝ), q_comp b c y = 1 → y = x₁ ∨ y = x₂ ∨ y = x₃ ∨ y = x₄

/-- The product of roots for a quadratic polynomial -/
def root_product (b c : ℝ) : ℝ := c

theorem challenging_polynomial_theorem :
  has_four_solutions (3/4) 1 ∧
  (∀ b c : ℝ, has_four_solutions b c → root_product b c ≤ root_product (3/4) 1) ∧
  q (3/4) 1 (-3) = 31/4 := by sorry

end NUMINAMATH_CALUDE_challenging_polynomial_theorem_l1416_141609


namespace NUMINAMATH_CALUDE_no_rational_solution_sqrt2_equation_l1416_141644

theorem no_rational_solution_sqrt2_equation :
  ∀ (x y z t : ℚ), (x + y * Real.sqrt 2)^2 + (z + t * Real.sqrt 2)^2 ≠ 5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_sqrt2_equation_l1416_141644


namespace NUMINAMATH_CALUDE_solve_equation_l1416_141618

theorem solve_equation : ∃ x : ℚ, 5 * (x - 6) = 3 * (3 - 3 * x) + 9 ∧ x = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1416_141618


namespace NUMINAMATH_CALUDE_orangeade_ratio_l1416_141680

def orangeade_problem (orange_juice water_day1 : ℝ) : Prop :=
  let water_day2 := 2 * water_day1
  let price_day1 := 0.30
  let price_day2 := 0.20
  let volume_day1 := orange_juice + water_day1
  let volume_day2 := orange_juice + water_day2
  (volume_day1 * price_day1 = volume_day2 * price_day2) →
  (orange_juice = water_day1)

theorem orangeade_ratio :
  ∀ (orange_juice water_day1 : ℝ),
  orangeade_problem orange_juice water_day1 :=
sorry

end NUMINAMATH_CALUDE_orangeade_ratio_l1416_141680


namespace NUMINAMATH_CALUDE_pencils_remaining_l1416_141646

/-- Given a box of pencils with an initial count and a number of pencils taken,
    prove that the remaining number of pencils is the difference between the initial count and the number taken. -/
theorem pencils_remaining (initial_count taken : ℕ) : 
  initial_count = 79 → taken = 4 → initial_count - taken = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_remaining_l1416_141646


namespace NUMINAMATH_CALUDE_forum_posts_l1416_141659

/-- A forum with members posting questions and answers -/
structure Forum where
  members : ℕ
  questions_per_hour : ℕ
  answer_ratio : ℕ

/-- Calculate the total number of questions posted in a day -/
def total_questions_per_day (f : Forum) : ℕ :=
  f.members * (f.questions_per_hour * 24)

/-- Calculate the total number of answers posted in a day -/
def total_answers_per_day (f : Forum) : ℕ :=
  f.members * (f.questions_per_hour * 24 * f.answer_ratio)

/-- Theorem stating the number of questions and answers posted in a day -/
theorem forum_posts (f : Forum) 
  (h1 : f.members = 200)
  (h2 : f.questions_per_hour = 3)
  (h3 : f.answer_ratio = 3) :
  total_questions_per_day f = 14400 ∧ total_answers_per_day f = 43200 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_l1416_141659


namespace NUMINAMATH_CALUDE_largest_number_is_4968_l1416_141654

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : ℕ
  first_number : ℕ
  second_number : ℕ
  hTotal : total_students = 5000
  hRange : first_number ≥ 1 ∧ second_number ≤ total_students
  hOrder : first_number < second_number

/-- The largest number in the systematic sample -/
def largest_number (s : SystematicSample) : ℕ :=
  s.first_number + (s.second_number - s.first_number) * ((s.total_students - s.first_number) / (s.second_number - s.first_number))

/-- Theorem stating the largest number in the systematic sample -/
theorem largest_number_is_4968 (s : SystematicSample) 
  (h1 : s.first_number = 18) 
  (h2 : s.second_number = 68) : 
  largest_number s = 4968 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_4968_l1416_141654


namespace NUMINAMATH_CALUDE_tv_discount_theorem_l1416_141661

/-- Represents the price of a TV with successive discounts -/
def discounted_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price of a TV after successive discounts is 63% of the original price -/
theorem tv_discount_theorem :
  let original_price : ℝ := 450
  let discount1 : ℝ := 0.30
  let discount2 : ℝ := 0.10
  let final_price := discounted_price original_price discount1 discount2
  final_price / original_price = 0.63 := by
  sorry


end NUMINAMATH_CALUDE_tv_discount_theorem_l1416_141661


namespace NUMINAMATH_CALUDE_not_always_true_from_false_l1416_141679

-- Define a proposition
variable (P Q R : Prop)

-- Define a logical argument
def logical_argument (premises : Prop) (conclusion : Prop) : Prop :=
  premises → conclusion

-- Define soundness of logical derivation
def sound_derivation (arg : Prop → Prop) : Prop :=
  ∀ (X Y : Prop), (X → Y) → (arg X → arg Y)

-- Theorem statement
theorem not_always_true_from_false :
  ∃ (premises conclusion : Prop) (arg : Prop → Prop),
    (¬premises) ∧ 
    (sound_derivation arg) ∧
    (logical_argument premises conclusion) ∧
    (¬conclusion) :=
sorry

end NUMINAMATH_CALUDE_not_always_true_from_false_l1416_141679


namespace NUMINAMATH_CALUDE_nap_time_is_three_hours_nap_time_in_hours_l1416_141693

/-- Calculates the remaining time for a nap given flight duration and time spent on activities -/
def remaining_nap_time (flight_duration : ℕ) (reading_time : ℕ) (movie_time : ℕ)
  (dinner_time : ℕ) (radio_time : ℕ) (game_time : ℕ) : ℕ :=
  flight_duration - (reading_time + movie_time + dinner_time + radio_time + game_time)

/-- Theorem stating that the remaining time for a nap is 3 hours -/
theorem nap_time_is_three_hours :
  remaining_nap_time 680 120 240 30 40 70 = 180 := by
  sorry

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

/-- Theorem stating that 180 minutes is equal to 3 hours -/
theorem nap_time_in_hours :
  minutes_to_hours (remaining_nap_time 680 120 240 30 40 70) = 3 := by
  sorry

end NUMINAMATH_CALUDE_nap_time_is_three_hours_nap_time_in_hours_l1416_141693


namespace NUMINAMATH_CALUDE_total_components_total_components_proof_l1416_141640

/-- The total number of components of types A, B, and C is 900. -/
theorem total_components : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (total_B : ℕ) (total_C : ℕ) (sample_size : ℕ) (sample_A : ℕ) (sample_C : ℕ) (total : ℕ) =>
    total_B = 300 →
    total_C = 200 →
    sample_size = 45 →
    sample_A = 20 →
    sample_C = 10 →
    total = 900

/-- Proof of the theorem -/
theorem total_components_proof :
  total_components 300 200 45 20 10 900 := by
  sorry

end NUMINAMATH_CALUDE_total_components_total_components_proof_l1416_141640


namespace NUMINAMATH_CALUDE_exists_points_on_hyperbola_with_midpoint_l1416_141678

/-- The hyperbola equation --/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- Definition of a midpoint --/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem exists_points_on_hyperbola_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola x₁ y₁ ∧ 
    hyperbola x₂ y₂ ∧ 
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_exists_points_on_hyperbola_with_midpoint_l1416_141678


namespace NUMINAMATH_CALUDE_total_surface_area_circumscribed_prism_l1416_141667

/-- A prism circumscribed about a sphere -/
structure CircumscribedPrism where
  -- The area of the base of the prism
  base_area : ℝ
  -- The semi-perimeter of the base of the prism
  semi_perimeter : ℝ
  -- The radius of the sphere
  sphere_radius : ℝ
  -- The base area is equal to the product of semi-perimeter and sphere radius
  base_area_eq : base_area = semi_perimeter * sphere_radius

/-- The total surface area of a circumscribed prism is 6 times its base area -/
theorem total_surface_area_circumscribed_prism (p : CircumscribedPrism) :
  ∃ (total_surface_area : ℝ), total_surface_area = 6 * p.base_area :=
by
  sorry

end NUMINAMATH_CALUDE_total_surface_area_circumscribed_prism_l1416_141667


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l1416_141603

theorem dvd_rental_cost (total_dvds : ℕ) (total_cost : ℝ) (known_dvds : ℕ) (known_cost : ℝ) : 
  total_dvds = 7 → 
  total_cost = 12.6 → 
  known_dvds = 3 → 
  known_cost = 1.5 → 
  total_cost - (known_dvds * known_cost) = 8.1 :=
by sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l1416_141603


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l1416_141620

theorem consecutive_integers_sqrt_17 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 17) → (Real.sqrt 17 < b) → (a + b = 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_17_l1416_141620


namespace NUMINAMATH_CALUDE_nancy_target_amount_l1416_141608

def hourly_rate (total_earnings : ℚ) (hours_worked : ℚ) : ℚ :=
  total_earnings / hours_worked

def target_amount (rate : ℚ) (target_hours : ℚ) : ℚ :=
  rate * target_hours

theorem nancy_target_amount 
  (initial_earnings : ℚ) 
  (initial_hours : ℚ) 
  (target_hours : ℚ) 
  (h1 : initial_earnings = 28)
  (h2 : initial_hours = 4)
  (h3 : target_hours = 10) :
  target_amount (hourly_rate initial_earnings initial_hours) target_hours = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_nancy_target_amount_l1416_141608


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1416_141656

theorem inequality_equivalence (x y : ℝ) : y - x < Real.sqrt (x^2) ↔ y < 0 ∨ y < 2*x := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1416_141656


namespace NUMINAMATH_CALUDE_shirts_produced_theorem_l1416_141638

/-- An industrial machine that produces shirts. -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  minutes_worked_today : ℕ

/-- Calculates the total number of shirts produced by the machine today. -/
def shirts_produced_today (machine : ShirtMachine) : ℕ :=
  machine.shirts_per_minute * machine.minutes_worked_today

/-- Theorem stating that a machine producing 6 shirts per minute working for 12 minutes produces 72 shirts. -/
theorem shirts_produced_theorem (machine : ShirtMachine)
    (h1 : machine.shirts_per_minute = 6)
    (h2 : machine.minutes_worked_today = 12) :
    shirts_produced_today machine = 72 := by
  sorry

end NUMINAMATH_CALUDE_shirts_produced_theorem_l1416_141638


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l1416_141639

theorem perfect_square_divisibility (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ (n : ℕ), x = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l1416_141639


namespace NUMINAMATH_CALUDE_shari_walk_distance_l1416_141677

/-- Calculates the distance walked given a constant walking speed, total time, and break time. -/
def distance_walked (speed : ℝ) (total_time : ℝ) (break_time : ℝ) : ℝ :=
  speed * (total_time - break_time)

/-- Proves that walking at 4 miles per hour for 2 hours with a 30-minute break results in 6 miles walked. -/
theorem shari_walk_distance :
  let speed : ℝ := 4
  let total_time : ℝ := 2
  let break_time : ℝ := 0.5
  distance_walked speed total_time break_time = 6 := by sorry

end NUMINAMATH_CALUDE_shari_walk_distance_l1416_141677


namespace NUMINAMATH_CALUDE_women_stockbrokers_increase_l1416_141600

/-- Calculates the final number of women stockbrokers after a percentage increase -/
def final_number (initial : ℕ) (percent_increase : ℕ) : ℕ :=
  initial + (initial * percent_increase) / 100

/-- Theorem: Given 10,000 initial women stockbrokers and a 100% increase, 
    the final number is 20,000 -/
theorem women_stockbrokers_increase : 
  final_number 10000 100 = 20000 := by sorry

end NUMINAMATH_CALUDE_women_stockbrokers_increase_l1416_141600


namespace NUMINAMATH_CALUDE_quadratic_sum_inequality_l1416_141660

theorem quadratic_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≤ 3*b ∧ b ≤ 3*a)
  (hac : a ≤ 3*c ∧ c ≤ 3*a)
  (had : a ≤ 3*d ∧ d ≤ 3*a)
  (hbc : b ≤ 3*c ∧ c ≤ 3*b)
  (hbd : b ≤ 3*d ∧ d ≤ 3*b)
  (hcd : c ≤ 3*d ∧ d ≤ 3*c) :
  a^2 + b^2 + c^2 + d^2 < 2*(a*b + a*c + a*d + b*c + b*d + c*d) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_inequality_l1416_141660


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1416_141616

theorem arithmetic_series_sum : ∀ (a₁ aₙ : ℕ), 
  a₁ = 5 → aₙ = 105 → 
  ∃ (n : ℕ), n > 1 ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ n → ∃ d, a₁ + (k - 1) * d = aₙ) →
  (n * (a₁ + aₙ)) / 2 = 5555 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l1416_141616


namespace NUMINAMATH_CALUDE_right_triangle_one_two_sqrt_three_l1416_141670

theorem right_triangle_one_two_sqrt_three :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 1 ∧ b = Real.sqrt 3 ∧ c = 2 ∧
  a^2 + b^2 = c^2 ∧
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_one_two_sqrt_three_l1416_141670


namespace NUMINAMATH_CALUDE_f_composition_negative_eight_l1416_141699

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then -x^(1/3)
  else x + 2/x - 7

-- State the theorem
theorem f_composition_negative_eight : f (f (-8)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_eight_l1416_141699


namespace NUMINAMATH_CALUDE_symmetric_parabola_b_eq_six_l1416_141612

/-- A function f(x) = x^2 + (a+2)x + 3 with domain [a, b] that is symmetric about x = 1 -/
def symmetric_parabola (a b : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, 
    (∀ x ∈ Set.Icc a b, f x = x^2 + (a+2)*x + 3) ∧ 
    (∀ x ∈ Set.Icc a b, f (2 - x) = f x)

/-- If a parabola is symmetric about x = 1, then b = 6 -/
theorem symmetric_parabola_b_eq_six (a b : ℝ) :
  symmetric_parabola a b → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_parabola_b_eq_six_l1416_141612


namespace NUMINAMATH_CALUDE_unique_solution_is_one_l1416_141666

/-- A function satisfying f(x)f(y) = f(x-y) for all x and y, and is nonzero at some point -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y, f x * f y = f (x - y)) ∧ (∃ x, f x ≠ 0)

/-- The constant function 1 is the unique solution to the functional equation -/
theorem unique_solution_is_one :
  ∀ f : ℝ → ℝ, FunctionalEquation f → (∀ x, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_l1416_141666


namespace NUMINAMATH_CALUDE_limit_of_rational_function_at_four_l1416_141689

theorem limit_of_rational_function_at_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 4| ∧ |x - 4| < δ →
    |((x^2 - 2*x - 8) / (x - 4)) - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_rational_function_at_four_l1416_141689


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1416_141637

/-- Proves that the initial percentage of water in a 40-liter mixture is 10%
    given that adding 5 liters of water results in a 20% water mixture. -/
theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_water = 5)
  (h3 : final_water_percentage = 20)
  (h4 : (initial_volume * x / 100 + added_water) / (initial_volume + added_water) * 100 = final_water_percentage) :
  x = 10 := by
  sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l1416_141637


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1416_141695

theorem sin_2theta_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2) 
  (h3 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6) : 
  Real.sin (2 * θ) = Real.sqrt 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1416_141695


namespace NUMINAMATH_CALUDE_sin_cos_shift_l1416_141610

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x - π / 4) = Real.cos (2 * (x - 3 * π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l1416_141610


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1416_141684

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area : 
  ∀ (r h l : ℝ) (S : ℝ),
    r = 3 →
    h = 4 →
    l^2 = r^2 + h^2 →
    S = π * r * l →
    S = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1416_141684


namespace NUMINAMATH_CALUDE_friends_team_assignments_l1416_141658

/-- The number of ways to assign n distinguishable objects to k distinct categories -/
def assignments (n : ℕ) (k : ℕ+) : ℕ := k.val ^ n

/-- Proof that for 8 friends and 3 teams, the number of assignments is 3^8 -/
theorem friends_team_assignments :
  assignments 8 3 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignments_l1416_141658


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1416_141628

theorem unique_integer_solution :
  ∃! (a : ℤ), ∃ (d e : ℤ), ∀ (x : ℤ), (x - a) * (x - 8) - 3 = (x + d) * (x + e) ∧ a = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1416_141628


namespace NUMINAMATH_CALUDE_problem_1_and_2_l1416_141687

theorem problem_1_and_2 :
  (1/2 * Real.sqrt 24 - Real.sqrt 3 * Real.sqrt 2 = 0) ∧
  ((2 * Real.sqrt 3 + 3 * Real.sqrt 2)^2 = 30 + 12 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_and_2_l1416_141687


namespace NUMINAMATH_CALUDE_largest_t_value_l1416_141602

theorem largest_t_value : ∃ t_max : ℚ, 
  (∀ t : ℚ, (16 * t^2 - 40 * t + 15) / (4 * t - 3) + 7 * t = 5 * t + 2 → t ≤ t_max) ∧ 
  (16 * t_max^2 - 40 * t_max + 15) / (4 * t_max - 3) + 7 * t_max = 5 * t_max + 2 ∧
  t_max = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_largest_t_value_l1416_141602


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1416_141636

theorem rectangle_y_value (y : ℝ) : 
  y > 0 → -- y is positive
  (6 - (-2)) * (y - 2) = 64 → -- area of rectangle is 64
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1416_141636


namespace NUMINAMATH_CALUDE_allowance_proof_l1416_141623

/-- The student's bi-weekly allowance -/
def allowance : ℝ := 233.89

/-- The amount left after all spending -/
def remaining : ℝ := 2.10

theorem allowance_proof :
  allowance * (4/9) * (1/3) * (4/11) * (1/6) = remaining := by sorry

end NUMINAMATH_CALUDE_allowance_proof_l1416_141623


namespace NUMINAMATH_CALUDE_gcd_problem_l1416_141668

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8531) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1416_141668


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1416_141664

-- Define the original number
def original_number : ℝ := 1300000

-- Define the scientific notation components
def coefficient : ℝ := 1.3
def exponent : ℕ := 6

-- Theorem statement
theorem scientific_notation_equivalence :
  original_number = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1416_141664


namespace NUMINAMATH_CALUDE_seashells_sum_l1416_141624

/-- The number of seashells found by Joan, Jessica, and Jeremy -/
def joan_seashells : ℕ := 6
def jessica_seashells : ℕ := 8
def jeremy_seashells : ℕ := 12

/-- The total number of seashells found by Joan, Jessica, and Jeremy -/
def total_seashells : ℕ := joan_seashells + jessica_seashells + jeremy_seashells

theorem seashells_sum : total_seashells = 26 := by
  sorry

end NUMINAMATH_CALUDE_seashells_sum_l1416_141624


namespace NUMINAMATH_CALUDE_jacks_books_l1416_141686

/-- Calculates the number of books in a stack given the stack thickness,
    pages per inch, and pages per book. -/
def number_of_books (stack_thickness : ℕ) (pages_per_inch : ℕ) (pages_per_book : ℕ) : ℕ :=
  (stack_thickness * pages_per_inch) / pages_per_book

/-- Theorem stating that Jack's stack of 12 inches with 80 pages per inch
    and 160 pages per book contains 6 books. -/
theorem jacks_books :
  number_of_books 12 80 160 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacks_books_l1416_141686


namespace NUMINAMATH_CALUDE_point_on_x_axis_distance_to_origin_l1416_141653

/-- If a point P with coordinates (m-2, m+1) is on the x-axis, then the distance from P to the origin is 3. -/
theorem point_on_x_axis_distance_to_origin :
  ∀ m : ℝ, (m + 1 = 0) → Real.sqrt ((m - 2)^2 + 0^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_distance_to_origin_l1416_141653


namespace NUMINAMATH_CALUDE_xiao_ming_error_l1416_141626

theorem xiao_ming_error (x : ℝ) : 
  (x + 1) / 2 - 1 = (x - 2) / 3 → 
  3 * (x + 1) - 1 ≠ 2 * (x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_error_l1416_141626


namespace NUMINAMATH_CALUDE_school_students_count_l1416_141650

theorem school_students_count :
  let blue_percent : ℝ := 0.45
  let red_percent : ℝ := 0.23
  let green_percent : ℝ := 0.15
  let other_count : ℕ := 102
  let total_count : ℕ := 600
  blue_percent + red_percent + green_percent + (other_count : ℝ) / total_count = 1 ∧
  (other_count : ℝ) / total_count = 1 - (blue_percent + red_percent + green_percent) :=
by sorry

end NUMINAMATH_CALUDE_school_students_count_l1416_141650


namespace NUMINAMATH_CALUDE_charge_difference_l1416_141643

/-- Represents the pricing scheme of a psychologist -/
structure PricingScheme where
  firstHourCharge : ℝ
  additionalHourCharge : ℝ
  fiveHourTotal : ℝ
  twoHourTotal : ℝ

/-- Theorem stating the difference in charges for a specific pricing scheme -/
theorem charge_difference (p : PricingScheme) 
  (h1 : p.firstHourCharge > p.additionalHourCharge)
  (h2 : p.firstHourCharge + 4 * p.additionalHourCharge = p.fiveHourTotal)
  (h3 : p.firstHourCharge + p.additionalHourCharge = p.twoHourTotal)
  (h4 : p.fiveHourTotal = 350)
  (h5 : p.twoHourTotal = 161) : 
  p.firstHourCharge - p.additionalHourCharge = 35 := by
  sorry

end NUMINAMATH_CALUDE_charge_difference_l1416_141643


namespace NUMINAMATH_CALUDE_parallel_condition_l1416_141682

/-- Two lines are parallel if they have the same slope -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (a^2-a)x+y=0 -/
def slope1 (a : ℝ) : ℝ := a^2 - a

/-- The slope of the line 2x+y+1=0 -/
def slope2 : ℝ := 2

theorem parallel_condition (a : ℝ) :
  (a = 2 → parallel (slope1 a) slope2) ∧
  (∃ b : ℝ, b ≠ 2 ∧ parallel (slope1 b) slope2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1416_141682


namespace NUMINAMATH_CALUDE_count_pairs_with_harmonic_mean_5_20_l1416_141615

/-- The number of ordered pairs of positive integers with harmonic mean 5^20 -/
def count_pairs : ℕ := 20

/-- Harmonic mean of two numbers -/
def harmonic_mean (x y : ℕ) : ℚ := 2 * x * y / (x + y)

/-- Predicate for valid pairs -/
def is_valid_pair (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ x < y ∧ harmonic_mean x y = 5^20

/-- The main theorem -/
theorem count_pairs_with_harmonic_mean_5_20 :
  (∃ (S : Finset (ℕ × ℕ)), S.card = count_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ S ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_pairs_with_harmonic_mean_5_20_l1416_141615


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1416_141627

theorem inequality_system_solution (x : ℝ) : 
  2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1416_141627


namespace NUMINAMATH_CALUDE_roots_relation_l1416_141611

theorem roots_relation (a b c d : ℝ) : 
  (∀ x, (x - a) * (x - b) - x = 0 ↔ x = c ∨ x = d) →
  (∀ x, (x - c) * (x - d) + x = 0 ↔ x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_roots_relation_l1416_141611


namespace NUMINAMATH_CALUDE_a_3_value_l1416_141634

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = -3

theorem a_3_value (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 7 → a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_3_value_l1416_141634


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1416_141629

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a with a₁ = 2 and a₃ = 4, prove that a₇ = 16 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 4) : 
  a 7 = 16 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l1416_141629


namespace NUMINAMATH_CALUDE_rhombus_q_value_l1416_141648

/-- A rhombus ABCD on a Cartesian plane -/
structure Rhombus where
  P : ℝ
  u : ℝ
  v : ℝ
  A : ℝ × ℝ := (0, 0)
  B : ℝ × ℝ := (P, 1)
  C : ℝ × ℝ := (u, v)
  D : ℝ × ℝ := (1, P)

/-- The sum of u and v coordinates of point C -/
def Q (r : Rhombus) : ℝ := r.u + r.v

/-- Theorem: For a rhombus ABCD with given coordinates, Q equals 2P + 2 -/
theorem rhombus_q_value (r : Rhombus) : Q r = 2 * r.P + 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_q_value_l1416_141648


namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l1416_141691

theorem min_value_absolute_sum (x : ℝ) : 
  ∃ (m : ℝ), (∀ x, |x - 1| + |x + 2| ≥ m) ∧ (∃ x, |x - 1| + |x + 2| = m) ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l1416_141691


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l1416_141676

theorem sqrt_x_plus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l1416_141676


namespace NUMINAMATH_CALUDE_game_c_higher_probability_l1416_141622

-- Define the probability of getting heads
def p_heads : ℚ := 2/3

-- Define the probability of getting tails
def p_tails : ℚ := 1/3

-- Define the probability of winning Game C
def p_game_c : ℚ :=
  let p_first_three := p_heads^3 + p_tails^3
  let p_last_three := p_heads^3 + p_tails^3
  let p_overlap := p_heads^5 + p_tails^5
  p_first_three + p_last_three - p_overlap

-- Define the probability of winning Game D
def p_game_d : ℚ :=
  let p_first_last_two := (p_heads^2 + p_tails^2)^2
  let p_middle_three := p_heads^3 + p_tails^3
  let p_overlap := 2 * (p_heads^4 + p_tails^4)
  p_first_last_two + p_middle_three - p_overlap

-- Theorem statement
theorem game_c_higher_probability :
  p_game_c - p_game_d = 29/81 :=
sorry

end NUMINAMATH_CALUDE_game_c_higher_probability_l1416_141622


namespace NUMINAMATH_CALUDE_z_profit_share_l1416_141671

/-- Calculates the share of profit for a partner in a business --/
def calculate_profit_share (
  x_capital y_capital z_capital : ℕ)  -- Initial capitals
  (x_months y_months z_months : ℕ)    -- Months of investment
  (total_profit : ℕ)                  -- Total annual profit
  : ℕ :=
  let x_share := x_capital * x_months
  let y_share := y_capital * y_months
  let z_share := z_capital * z_months
  let total_share := x_share + y_share + z_share
  (z_share * total_profit) / total_share

/-- Theorem statement for Z's profit share --/
theorem z_profit_share :
  calculate_profit_share 20000 25000 30000 12 12 7 50000 = 14000 := by
  sorry

end NUMINAMATH_CALUDE_z_profit_share_l1416_141671


namespace NUMINAMATH_CALUDE_square_tablecloth_side_length_l1416_141685

-- Define a square tablecloth
structure SquareTablecloth where
  side : ℝ
  area : ℝ
  is_square : area = side * side

-- Theorem statement
theorem square_tablecloth_side_length 
  (tablecloth : SquareTablecloth) 
  (h : tablecloth.area = 5) : 
  tablecloth.side = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_tablecloth_side_length_l1416_141685


namespace NUMINAMATH_CALUDE_envelope_height_l1416_141663

theorem envelope_height (width : ℝ) (area : ℝ) (height : ℝ) : 
  width = 6 → area = 36 → area = width * height → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_envelope_height_l1416_141663
