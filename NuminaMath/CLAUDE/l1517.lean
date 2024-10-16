import Mathlib

namespace NUMINAMATH_CALUDE_expand_polynomial_l1517_151746

theorem expand_polynomial (x : ℝ) : (7 * x^3 - 5 * x + 2) * (4 * x^2) = 28 * x^5 - 20 * x^3 + 8 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1517_151746


namespace NUMINAMATH_CALUDE_first_wave_infections_count_l1517_151713

/-- The number of infections per day during the first wave of coronavirus -/
def first_wave_infections : ℕ := 375

/-- The number of infections per day during the second wave of coronavirus -/
def second_wave_infections : ℕ := 4 * first_wave_infections

/-- The duration of the second wave in days -/
def second_wave_duration : ℕ := 14

/-- The total number of infections during the second wave -/
def total_second_wave_infections : ℕ := 21000

/-- Theorem stating that the number of infections per day during the first wave was 375 -/
theorem first_wave_infections_count : 
  first_wave_infections = 375 ∧ 
  second_wave_infections = 4 * first_wave_infections ∧
  total_second_wave_infections = second_wave_infections * second_wave_duration :=
sorry

end NUMINAMATH_CALUDE_first_wave_infections_count_l1517_151713


namespace NUMINAMATH_CALUDE_number_ratio_l1517_151751

theorem number_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 143) (h4 : y = 104) :
  y / x = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1517_151751


namespace NUMINAMATH_CALUDE_profit_calculation_min_model_A_bicycles_l1517_151731

-- Define the profit functions for models A and B
def profit_A : ℝ := 150
def profit_B : ℝ := 100

-- Define the purchase prices
def price_A : ℝ := 500
def price_B : ℝ := 800

-- Define the total number of bicycles and budget
def total_bicycles : ℕ := 20
def max_budget : ℝ := 13000

-- Theorem for part 1
theorem profit_calculation :
  3 * profit_A + 2 * profit_B = 650 ∧
  profit_A + 2 * profit_B = 350 := by sorry

-- Theorem for part 2
theorem min_model_A_bicycles :
  ∀ m : ℕ,
  (m ≤ total_bicycles ∧ 
   price_A * m + price_B * (total_bicycles - m) ≤ max_budget) →
  m ≥ 10 := by sorry

end NUMINAMATH_CALUDE_profit_calculation_min_model_A_bicycles_l1517_151731


namespace NUMINAMATH_CALUDE_rational_square_plus_one_positive_l1517_151725

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_plus_one_positive_l1517_151725


namespace NUMINAMATH_CALUDE_sameColorPairWithBlueCount_l1517_151785

/-- The number of ways to choose a pair of socks of the same color with at least one blue sock -/
def sameColorPairWithBlue (whiteCount brownCount blueCount greenCount : ℕ) : ℕ :=
  Nat.choose blueCount 2

theorem sameColorPairWithBlueCount :
  sameColorPairWithBlue 5 5 5 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sameColorPairWithBlueCount_l1517_151785


namespace NUMINAMATH_CALUDE_triangle_theorem_l1517_151791

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin (t.B + t.C) + t.a * Real.cos t.B = t.c) 
  (h2 : t.a = 6)
  (h3 : t.b + t.c = 6 + 6 * Real.sqrt 3) : 
  t.A = π / 6 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1517_151791


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1517_151756

theorem quadratic_factorization (x : ℝ) : x^2 + 6*x = 1 ↔ (x + 3)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1517_151756


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1517_151795

def p (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x - 1
def q (x : ℝ) : ℝ := 2 * x^3 - x^2 + 5 * x - 4

theorem coefficient_of_x_squared :
  (∃ a b c d e : ℝ, ∀ x, p x * q x = a * x^5 + b * x^4 + c * x^3 - 31 * x^2 + d * x + e) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1517_151795


namespace NUMINAMATH_CALUDE_ninth_root_unity_sum_l1517_151733

theorem ninth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (2 * Real.pi * I / 9) →
  z^9 = 1 →
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_unity_sum_l1517_151733


namespace NUMINAMATH_CALUDE_sibling_ages_l1517_151711

/-- Represents the ages of the siblings -/
structure SiblingAges where
  maria : ℕ
  ann : ℕ
  david : ℕ
  ethan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : SiblingAges) : Prop :=
  ages.maria = ages.ann - 3 ∧
  ages.maria - 4 = (ages.ann - 4) / 2 ∧
  ages.david = ages.maria + 2 ∧
  (ages.david - 2) + (ages.ann - 2) = 3 * (ages.maria - 2) ∧
  ages.ethan = ages.david - ages.maria ∧
  ages.ann - ages.ethan = 8

/-- The theorem stating the ages of the siblings -/
theorem sibling_ages : 
  ∃ (ages : SiblingAges), satisfiesConditions ages ∧ 
    ages.maria = 7 ∧ ages.ann = 10 ∧ ages.david = 9 ∧ ages.ethan = 2 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_l1517_151711


namespace NUMINAMATH_CALUDE_employee_salary_problem_l1517_151762

theorem employee_salary_problem (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) :
  num_employees = 15 →
  manager_salary = 4200 →
  avg_increase = 150 →
  (∃ (avg_salary : ℕ),
    num_employees * avg_salary + manager_salary = (num_employees + 1) * (avg_salary + avg_increase) ∧
    avg_salary = 1800) :=
by sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l1517_151762


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l1517_151732

/-- A tetrahedron is represented by its six edge lengths -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: For any tetrahedron with only one edge length greater than 1, its volume is at most 1/8 -/
theorem tetrahedron_volume_bound (t : Tetrahedron) 
  (h : ∃! i, t.edges i > 1) : volume t ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l1517_151732


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l1517_151735

/-- Given four real numbers a, b, c, d forming a geometric progression,
    prove that (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 -/
theorem geometric_progression_equality (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l1517_151735


namespace NUMINAMATH_CALUDE_quadratic_roots_fraction_l1517_151722

theorem quadratic_roots_fraction (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_fraction_l1517_151722


namespace NUMINAMATH_CALUDE_inequality_ordering_l1517_151748

theorem inequality_ordering (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_ordering_l1517_151748


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l1517_151719

theorem quadratic_root_equivalence (a b c : ℝ) (h : a ≠ 0) :
  (1 = 1 ∧ a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l1517_151719


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1517_151758

theorem diophantine_equation_solutions :
  ∀ x y z w : ℕ, 2^x * 3^y - 5^x * 7^w = 1 ↔ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1517_151758


namespace NUMINAMATH_CALUDE_starting_lineups_count_l1517_151727

/-- The number of players in the team -/
def totalPlayers : ℕ := 15

/-- The number of players in a starting lineup -/
def lineupSize : ℕ := 6

/-- The number of players who cannot play together -/
def restrictedPlayers : ℕ := 3

/-- The number of possible starting lineups -/
def possibleLineups : ℕ := 3300

/-- Theorem stating the number of possible starting lineups -/
theorem starting_lineups_count :
  (Nat.choose totalPlayers lineupSize) - 
  (Nat.choose (totalPlayers - restrictedPlayers) (lineupSize - restrictedPlayers)) = 
  possibleLineups := by
  sorry

end NUMINAMATH_CALUDE_starting_lineups_count_l1517_151727


namespace NUMINAMATH_CALUDE_range_of_a_l1517_151701

theorem range_of_a (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 1) (ha : ∃ a : ℝ, a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  ∃ a : ℝ, a ∈ Set.Ioo 0 (7 / 27) ∨ a = 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1517_151701


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_sum_zero_l1517_151792

theorem sqrt_sum_zero_implies_power_sum_zero (a b : ℝ) :
  Real.sqrt (a + 1) + Real.sqrt (b - 1) = 0 → a^1011 + b^1011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_sum_zero_l1517_151792


namespace NUMINAMATH_CALUDE_min_moves_for_checkerboard_l1517_151716

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents a 6x6 grid -/
def Grid := Fin 6 → Fin 6 → Cell

/-- Represents a move (changing color of two adjacent cells) -/
structure Move where
  row : Fin 6
  col : Fin 6
  horizontal : Bool

/-- Defines a checkerboard pattern -/
def isCheckerboard (g : Grid) : Prop :=
  ∀ i j, g i j = if (i.val + j.val) % 2 = 0 then Cell.White else Cell.Black

/-- Applies a move to a grid -/
def applyMove (g : Grid) (m : Move) : Grid :=
  sorry

/-- Counts the number of black cells in a grid -/
def blackCellCount (g : Grid) : Nat :=
  sorry

theorem min_moves_for_checkerboard :
  ∀ (initial : Grid) (moves : List Move),
    (∀ i j, initial i j = Cell.White) →
    isCheckerboard (moves.foldl applyMove initial) →
    moves.length ≥ 18 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_for_checkerboard_l1517_151716


namespace NUMINAMATH_CALUDE_aubree_animal_count_l1517_151757

/-- The total number of animals Aubree saw in a day, given the initial counts and changes --/
def total_animals (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 2 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 10
  let afternoon_total := afternoon_beavers + afternoon_chipmunks
  morning_total + afternoon_total

/-- Theorem stating that the total number of animals seen is 130 --/
theorem aubree_animal_count : total_animals 20 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aubree_animal_count_l1517_151757


namespace NUMINAMATH_CALUDE_matt_fem_age_ratio_l1517_151743

theorem matt_fem_age_ratio :
  ∀ (matt_age fem_age : ℕ),
    fem_age = 11 →
    matt_age + fem_age + 4 = 59 →
    matt_age = 4 * fem_age :=
by
  sorry

end NUMINAMATH_CALUDE_matt_fem_age_ratio_l1517_151743


namespace NUMINAMATH_CALUDE_second_day_average_speed_l1517_151761

/-- Represents the driving conditions and results over two days -/
structure DrivingData where
  total_distance : ℝ
  total_time : ℝ
  total_fuel : ℝ
  first_day_time_diff : ℝ
  first_day_speed_diff : ℝ
  first_day_efficiency : ℝ
  second_day_efficiency : ℝ

/-- Theorem stating that given the driving conditions, the average speed on the second day is 35 mph -/
theorem second_day_average_speed
  (data : DrivingData)
  (h1 : data.total_distance = 680)
  (h2 : data.total_time = 18)
  (h3 : data.total_fuel = 22.5)
  (h4 : data.first_day_time_diff = 2)
  (h5 : data.first_day_speed_diff = 5)
  (h6 : data.first_day_efficiency = 25)
  (h7 : data.second_day_efficiency = 30) :
  ∃ (second_day_speed : ℝ),
    second_day_speed = 35 ∧
    (second_day_speed + data.first_day_speed_diff) * (data.total_time / 2 + data.first_day_time_diff / 2) +
    second_day_speed * (data.total_time / 2 - data.first_day_time_diff / 2) = data.total_distance :=
by sorry

#check second_day_average_speed

end NUMINAMATH_CALUDE_second_day_average_speed_l1517_151761


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1517_151714

theorem unique_integer_solution : ∃! x : ℕ+, (4 * x)^2 - x = 2100 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1517_151714


namespace NUMINAMATH_CALUDE_gcd_lcm_product_28_45_l1517_151777

theorem gcd_lcm_product_28_45 : Nat.gcd 28 45 * Nat.lcm 28 45 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_28_45_l1517_151777


namespace NUMINAMATH_CALUDE_range_of_even_function_l1517_151752

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem range_of_even_function (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →
  (∀ x, x ∈ Set.Icc (a - 3) (2 * a) ↔ f a b x ≠ 0) →
  Set.range (f a b) = Set.Icc 3 7 := by
  sorry

#check range_of_even_function

end NUMINAMATH_CALUDE_range_of_even_function_l1517_151752


namespace NUMINAMATH_CALUDE_f_pi_eighth_l1517_151734

noncomputable def f (θ : Real) : Real :=
  (2 * Real.sin (θ / 2) ^ 2 - 1) / (Real.sin (θ / 2) * Real.cos (θ / 2)) + 2 * Real.tan θ

theorem f_pi_eighth : f (Real.pi / 8) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_eighth_l1517_151734


namespace NUMINAMATH_CALUDE_vova_gave_three_l1517_151739

/-- Represents the number of nuts Vova gave to Pavlik -/
def k : ℕ := sorry

/-- Represents Vova's initial number of nuts -/
def V : ℕ := sorry

/-- Represents Pavlik's initial number of nuts -/
def P : ℕ := sorry

/-- Vova has more nuts than Pavlik -/
axiom vova_more : V > P

/-- If Vova gave Pavlik as many nuts as Pavlik had, they would have the same number -/
axiom equal_after_giving : V - P = P + P

/-- Vova gave Pavlik no more than 5 nuts -/
axiom k_at_most_5 : k ≤ 5

/-- The remaining nuts were divided equally among 3 squirrels -/
axiom divisible_by_3 : (V - k) % 3 = 0

/-- The number of nuts Vova gave to Pavlik is 3 -/
theorem vova_gave_three : k = 3 := by sorry

end NUMINAMATH_CALUDE_vova_gave_three_l1517_151739


namespace NUMINAMATH_CALUDE_barn_painted_area_l1517_151720

/-- Calculates the total area to be painted for a rectangular barn --/
def total_painted_area (width length height : ℝ) : ℝ :=
  let wall_area := 2 * (width * height + length * height)
  let floor_ceiling_area := 2 * (width * length)
  wall_area + floor_ceiling_area

/-- Theorem stating that the total area to be painted for the given barn is 1002 sq yd --/
theorem barn_painted_area :
  total_painted_area 15 18 7 = 1002 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l1517_151720


namespace NUMINAMATH_CALUDE_abs_T_equals_512_sqrt_2_l1517_151742

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the expression T
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_512_sqrt_2 : Complex.abs T = 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_512_sqrt_2_l1517_151742


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1517_151741

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
def Vector2D := Point2D

/-- Quadrilateral represented by four points -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Extended quadrilateral with additional points -/
structure ExtendedQuadrilateral where
  Q : Quadrilateral
  A' : Point2D
  D' : Point2D

/-- Vector addition -/
def vectorAdd (v1 v2 : Vector2D) : Vector2D :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

/-- Scalar multiplication of a vector -/
def scalarMul (s : ℝ) (v : Vector2D) : Vector2D :=
  { x := s * v.x, y := s * v.y }

/-- The main theorem to prove -/
theorem quadrilateral_reconstruction 
  (Q : ExtendedQuadrilateral) 
  (h1 : Q.A' = vectorAdd Q.Q.A (scalarMul 1 (vectorAdd Q.Q.B (scalarMul (-1) Q.Q.A))))
  (h2 : Q.D' = vectorAdd Q.Q.D (scalarMul 1 (vectorAdd Q.Q.C (scalarMul (-1) Q.Q.D))))
  : ∃ (p q r s : ℝ), 
    Q.Q.A = vectorAdd 
      (scalarMul p Q.A') 
      (vectorAdd 
        (scalarMul q Q.Q.B) 
        (vectorAdd 
          (scalarMul r Q.Q.C) 
          (scalarMul s Q.D')))
    ∧ p = 0 
    ∧ q = 0 
    ∧ r = 1/4 
    ∧ s = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1517_151741


namespace NUMINAMATH_CALUDE_original_recipe_pasta_amount_l1517_151718

theorem original_recipe_pasta_amount
  (original_servings : ℕ)
  (scaled_servings : ℕ)
  (scaled_pasta : ℝ)
  (h1 : original_servings = 7)
  (h2 : scaled_servings = 35)
  (h3 : scaled_pasta = 10) :
  let pasta_per_person : ℝ := scaled_pasta / scaled_servings
  let original_pasta : ℝ := pasta_per_person * original_servings
  original_pasta = 2 := by sorry

end NUMINAMATH_CALUDE_original_recipe_pasta_amount_l1517_151718


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l1517_151780

theorem pentagon_angle_sum (a b c d q : ℝ) : 
  a = 118 → b = 105 → c = 87 → d = 135 →
  (a + b + c + d + q = 540) →
  q = 95 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l1517_151780


namespace NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l1517_151705

theorem square_roots_and_cube_root_problem (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (3 * a - 14)^2 = k ∧ (a - 2)^2 = k) → 
  ((b - 15)^(1/3) = -3) → 
  (a = 4 ∧ b = -12 ∧ (∀ x : ℝ, x^2 = 4*a + b ↔ x = 2 ∨ x = -2)) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_and_cube_root_problem_l1517_151705


namespace NUMINAMATH_CALUDE_shirts_per_day_l1517_151754

theorem shirts_per_day (total_shirts : ℕ) (reused_shirts : ℕ) (vacation_days : ℕ) : 
  total_shirts = 11 → reused_shirts = 1 → vacation_days = 7 → 
  (total_shirts - reused_shirts) / (vacation_days - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_day_l1517_151754


namespace NUMINAMATH_CALUDE_children_without_candy_l1517_151797

/-- Represents the number of children in the circle -/
def num_children : ℕ := 73

/-- Represents the total number of candies distributed -/
def total_candies : ℕ := 2020

/-- Calculates the position of the nth candy distribution -/
def candy_position (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of unique positions reached -/
def unique_positions : ℕ := 37

theorem children_without_candy :
  num_children - unique_positions = 36 :=
sorry

end NUMINAMATH_CALUDE_children_without_candy_l1517_151797


namespace NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l1517_151786

theorem consecutive_four_plus_one_is_square (n : ℕ) (h : n ≥ 1) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l1517_151786


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1517_151767

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + A*x + B = 0) → A + B = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1517_151767


namespace NUMINAMATH_CALUDE_rachel_age_when_emily_half_her_age_l1517_151772

def emily_current_age : ℕ := 20
def rachel_current_age : ℕ := 24

theorem rachel_age_when_emily_half_her_age :
  ∃ (x : ℕ), 
    (rachel_current_age - x = 2 * (emily_current_age - x)) ∧
    (rachel_current_age - x = 8) := by
  sorry

end NUMINAMATH_CALUDE_rachel_age_when_emily_half_her_age_l1517_151772


namespace NUMINAMATH_CALUDE_pump_emptying_time_l1517_151753

theorem pump_emptying_time (time_B time_together : ℝ) 
  (hB : time_B = 6)
  (hTogether : time_together = 2.4)
  (h_rate : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_pump_emptying_time_l1517_151753


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1517_151781

theorem not_sufficient_not_necessary (a b : ℝ) : 
  ¬(∀ a b, (a ≠ 1 ∧ b ≠ 2) → (a + b ≠ 3)) ∧ 
  ¬(∀ a b, (a + b ≠ 3) → (a ≠ 1 ∧ b ≠ 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1517_151781


namespace NUMINAMATH_CALUDE_raft_sticks_ratio_l1517_151775

theorem raft_sticks_ratio :
  ∀ (simon_sticks gerry_sticks micky_sticks : ℕ),
    simon_sticks = 36 →
    micky_sticks = simon_sticks + gerry_sticks + 9 →
    simon_sticks + gerry_sticks + micky_sticks = 129 →
    gerry_sticks * 3 = simon_sticks * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_ratio_l1517_151775


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1517_151702

/-- A geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 1
  h_geometric : ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem to be proved -/
theorem geometric_sequence_problem (seq : GeometricSequence)
    (h1 : seq.a 3 * seq.a 7 = 72)
    (h2 : seq.a 2 + seq.a 8 = 27) :
  seq.a 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1517_151702


namespace NUMINAMATH_CALUDE_vector_AB_l1517_151736

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define the vector between two points
def vector (p q : Point) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem vector_AB : 
  let A : Point := (-2, 3)
  let B : Point := (3, 2)
  vector A B = (5, -1) := by sorry

end NUMINAMATH_CALUDE_vector_AB_l1517_151736


namespace NUMINAMATH_CALUDE_projection_theorem_l1517_151712

def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_theorem :
  let p := projection
  p (1, -2) = (3/2, -3/2) →
  p (-4, 1) = (-5/2, 5/2) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l1517_151712


namespace NUMINAMATH_CALUDE_max_brownie_pieces_l1517_151749

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The pan dimensions -/
def pan : Rectangle := { length := 24, width := 20 }

/-- The brownie piece dimensions -/
def piece : Rectangle := { length := 4, width := 3 }

/-- Theorem: The maximum number of brownie pieces that can be cut from the pan is 40 -/
theorem max_brownie_pieces : (area pan) / (area piece) = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_brownie_pieces_l1517_151749


namespace NUMINAMATH_CALUDE_parabola_properties_l1517_151779

/-- A parabola is defined by the equation y = -(x-3)^2 --/
def parabola (x y : ℝ) : Prop := y = -(x-3)^2

/-- The axis of symmetry of the parabola --/
def axis_of_symmetry : ℝ := 3

/-- Theorem: The parabola opens downwards and has its axis of symmetry at x=3 --/
theorem parabola_properties :
  (∀ x y : ℝ, parabola x y → y ≤ 0) ∧ 
  (∀ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ < axis_of_symmetry ∧ axis_of_symmetry < x₂ ∧ parabola x₁ y ∧ parabola x₂ y) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1517_151779


namespace NUMINAMATH_CALUDE_joan_football_games_l1517_151768

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l1517_151768


namespace NUMINAMATH_CALUDE_inequality_proof_l1517_151764

theorem inequality_proof (x : ℝ) : 1 + 2 * x^2 ≥ 2 * x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1517_151764


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_inequality_l1517_151765

theorem sin_cos_sixth_power_inequality (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_inequality_l1517_151765


namespace NUMINAMATH_CALUDE_stream_speed_l1517_151788

theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 120)
  (h2 : upstream_distance = 60)
  (h3 : time = 2) :
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_distance / time ∧
    boat_speed - stream_speed = upstream_distance / time ∧
    stream_speed = 15 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1517_151788


namespace NUMINAMATH_CALUDE_factorization_equality_l1517_151799

theorem factorization_equality (a b : ℝ) : (a - b)^2 + 6*(b - a) + 9 = (a - b - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1517_151799


namespace NUMINAMATH_CALUDE_linear_function_two_points_l1517_151715

/-- A linear function passing through exactly two of three given points -/
theorem linear_function_two_points :
  ∃ (f : ℝ → ℝ) (a b : ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (f 0 = 0 ∧ f 1 = 1 ∧ f 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_two_points_l1517_151715


namespace NUMINAMATH_CALUDE_no_solution_cubic_inequality_l1517_151793

theorem no_solution_cubic_inequality :
  ¬∃ x : ℝ, x ≠ 2 ∧ (x^3 - 8) / (x - 2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_cubic_inequality_l1517_151793


namespace NUMINAMATH_CALUDE_amoeba_fill_time_l1517_151784

def amoeba_population (initial : ℕ) (time : ℕ) : ℕ :=
  initial * 2^time

theorem amoeba_fill_time :
  ∀ (tube_capacity : ℕ),
  tube_capacity > 0 →
  (∃ (t : ℕ), amoeba_population 1 t = tube_capacity) →
  (∃ (s : ℕ), amoeba_population 2 s = tube_capacity ∧ s + 1 = t) :=
by sorry

end NUMINAMATH_CALUDE_amoeba_fill_time_l1517_151784


namespace NUMINAMATH_CALUDE_lesser_fraction_l1517_151771

theorem lesser_fraction (x y : ℝ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = (5 - Real.sqrt 7) / 12 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1517_151771


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1517_151798

theorem triangle_angle_sum (A B C : ℝ) : 
  (0 < A) → (0 < B) → (0 < C) →
  (A + B = 90) → (A + B + C = 180) →
  C = 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1517_151798


namespace NUMINAMATH_CALUDE_lottery_theorem_l1517_151703

-- Define the lottery parameters
def total_numbers : ℕ := 90
def numbers_drawn : ℕ := 5
def numbers_played : ℕ := 7
def group_size : ℕ := 10

-- Define the ticket prices and payouts
def ticket_cost : ℕ := 60
def payout_three_match : ℕ := 7000
def payout_two_match : ℕ := 300

-- Define the probability of drawing 3 out of 7 specific numbers
def probability_three_match : ℚ := 119105 / 43949268

-- Define the profit per person
def profit_per_person : ℕ := 4434

-- Theorem statement
theorem lottery_theorem :
  (probability_three_match = 119105 / 43949268) ∧
  (profit_per_person = 4434) := by
  sorry


end NUMINAMATH_CALUDE_lottery_theorem_l1517_151703


namespace NUMINAMATH_CALUDE_M_on_x_axis_M_parallel_to_x_axis_M_distance_from_y_axis_l1517_151769

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a real number m, construct a point M with coordinates (m-1, 2m+3) -/
def M (m : ℝ) : Point := ⟨m - 1, 2 * m + 3⟩

/-- N is a fixed point with coordinates (5, -1) -/
def N : Point := ⟨5, -1⟩

theorem M_on_x_axis (m : ℝ) : 
  M m = ⟨-5/2, 0⟩ ↔ (M m).y = 0 := by sorry

theorem M_parallel_to_x_axis (m : ℝ) :
  M m = ⟨-3, -1⟩ ↔ (M m).y = N.y := by sorry

theorem M_distance_from_y_axis (m : ℝ) :
  (M m = ⟨2, 9⟩ ∨ M m = ⟨-2, 1⟩) ↔ |(M m).x| = 2 := by sorry

end NUMINAMATH_CALUDE_M_on_x_axis_M_parallel_to_x_axis_M_distance_from_y_axis_l1517_151769


namespace NUMINAMATH_CALUDE_horners_method_for_f_l1517_151710

def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

theorem horners_method_for_f :
  f 3 = 21324 := by
sorry

end NUMINAMATH_CALUDE_horners_method_for_f_l1517_151710


namespace NUMINAMATH_CALUDE_prime_triplet_divisibility_l1517_151790

theorem prime_triplet_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  (q * r - 1) % p = 0 ∧
  (p * r - 1) % q = 0 ∧
  (p * q - 1) % r = 0 →
  ({p, q, r} : Set ℕ) = {2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_divisibility_l1517_151790


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1517_151737

/-- If 49m^2 + km + 1 is a perfect square trinomial, then k = ±14 -/
theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), ∀ m, 49 * m^2 + k * m + 1 = (a * m + b)^2) →
  k = 14 ∨ k = -14 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l1517_151737


namespace NUMINAMATH_CALUDE_brick_length_calculation_l1517_151789

theorem brick_length_calculation (courtyard_length courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ) (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16) (h3 : brick_width = 0.1) (h4 : total_bricks = 20000) :
  (courtyard_length * courtyard_width * 10000) / (brick_width * total_bricks) = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l1517_151789


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l1517_151721

/-- The value of m for which the line x = -2 is tangent to the circle x^2 + y^2 + 6x + m = 0 -/
theorem parabola_circle_tangency (m : ℝ) : 
  (∀ y : ℝ, ((-2)^2 + y^2 + 6*(-2) + m = 0) → 
   (∀ x : ℝ, x ≠ -2 → x^2 + y^2 + 6*x + m ≠ 0)) → 
  m = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l1517_151721


namespace NUMINAMATH_CALUDE_sampling_probability_theorem_l1517_151755

/-- Represents the probability of a student being selected in a sampling process -/
def sampling_probability (total_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / total_students

/-- The sampling method described in the problem -/
structure SamplingMethod where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ

/-- Theorem stating that the probability of each student being selected is equal and is 25/1002 -/
theorem sampling_probability_theorem (method : SamplingMethod)
  (h1 : method.total_students = 2004)
  (h2 : method.selected_students = 50)
  (h3 : method.eliminated_students = 4) :
  sampling_probability method.total_students method.selected_students = 25 / 1002 :=
sorry

end NUMINAMATH_CALUDE_sampling_probability_theorem_l1517_151755


namespace NUMINAMATH_CALUDE_total_legs_in_javiers_household_l1517_151774

/-- The number of legs in Javier's household -/
def total_legs : ℕ :=
  let num_humans := 5 -- Javier, his wife, and 3 children
  let num_dogs := 2
  let num_cats := 1
  let legs_per_human := 2
  let legs_per_dog := 4
  let legs_per_cat := 4
  num_humans * legs_per_human + num_dogs * legs_per_dog + num_cats * legs_per_cat

theorem total_legs_in_javiers_household :
  total_legs = 22 := by sorry

end NUMINAMATH_CALUDE_total_legs_in_javiers_household_l1517_151774


namespace NUMINAMATH_CALUDE_least_prime_factor_of_9_4_minus_9_3_l1517_151763

theorem least_prime_factor_of_9_4_minus_9_3 :
  Nat.minFac (9^4 - 9^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_9_4_minus_9_3_l1517_151763


namespace NUMINAMATH_CALUDE_rose_difference_is_34_l1517_151750

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The difference in the number of red roses between Mrs. Santiago and Mrs. Garrett -/
def rose_difference : ℕ := santiago_roses - garrett_roses

theorem rose_difference_is_34 : rose_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_rose_difference_is_34_l1517_151750


namespace NUMINAMATH_CALUDE_counterexample_a_minus_b_zero_l1517_151744

theorem counterexample_a_minus_b_zero : 
  ¬ (∀ a b : ℝ, a - b = 0 → a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_counterexample_a_minus_b_zero_l1517_151744


namespace NUMINAMATH_CALUDE_toy_sword_cost_l1517_151709

theorem toy_sword_cost (total_spent : ℕ) (lego_cost : ℕ) (play_dough_cost : ℕ)
  (lego_sets : ℕ) (toy_swords : ℕ) (play_doughs : ℕ) :
  total_spent = 1940 →
  lego_cost = 250 →
  play_dough_cost = 35 →
  lego_sets = 3 →
  toy_swords = 7 →
  play_doughs = 10 →
  ∃ (sword_cost : ℕ),
    sword_cost = 120 ∧
    total_spent = lego_cost * lego_sets + sword_cost * toy_swords + play_dough_cost * play_doughs :=
by sorry

end NUMINAMATH_CALUDE_toy_sword_cost_l1517_151709


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1517_151766

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 100 → initial_money = 250 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l1517_151766


namespace NUMINAMATH_CALUDE_function_properties_l1517_151726

noncomputable def m (a : ℝ) (t : ℝ) : ℝ := (1/2) * a * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
  if a > -1/2 then a + 2
  else if a > -Real.sqrt 2 / 2 then -a - 1 / (2 * a)
  else Real.sqrt 2

theorem function_properties (a : ℝ) :
  (∀ t : ℝ, Real.sqrt 2 ≤ t ∧ t ≤ 2 → 
    ∃ x : ℝ, m a t = a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)) ∧
  (∀ t : ℝ, Real.sqrt 2 ≤ t ∧ t ≤ 2 → m a t ≤ g a) ∧
  (a ≥ -Real.sqrt 2 → (g a = g (1/a) ↔ (-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2) ∨ a = 1)) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1517_151726


namespace NUMINAMATH_CALUDE_initial_papers_count_l1517_151760

/-- The number of papers Charles initially bought -/
def initial_papers : ℕ := sorry

/-- The number of pictures Charles drew today -/
def pictures_today : ℕ := 6

/-- The number of pictures Charles drew before work yesterday -/
def pictures_before_work : ℕ := 6

/-- The number of pictures Charles drew after work yesterday -/
def pictures_after_work : ℕ := 6

/-- The number of papers Charles has left -/
def papers_left : ℕ := 2

/-- Theorem stating that the initial number of papers is equal to the sum of papers used for pictures and papers left -/
theorem initial_papers_count : 
  initial_papers = pictures_today + pictures_before_work + pictures_after_work + papers_left :=
by sorry

end NUMINAMATH_CALUDE_initial_papers_count_l1517_151760


namespace NUMINAMATH_CALUDE_multiply_monomials_l1517_151759

theorem multiply_monomials (x : ℝ) : 2 * x * (5 * x^2) = 10 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_monomials_l1517_151759


namespace NUMINAMATH_CALUDE_investment_rate_is_five_percent_l1517_151704

/-- Represents an investment account --/
structure Account where
  balance : ℝ
  rate : ℝ

/-- Calculates the interest earned on an account in one year --/
def interest (a : Account) : ℝ := a.balance * a.rate

/-- Represents the investment scenario --/
structure InvestmentScenario where
  account1 : Account
  account2 : Account
  totalInterest : ℝ

/-- The given investment scenario --/
def scenario : InvestmentScenario where
  account1 := { balance := 8000, rate := 0.05 }
  account2 := { balance := 2000, rate := 0.06 }
  totalInterest := 520

/-- Theorem stating that the given scenario satisfies all conditions --/
theorem investment_rate_is_five_percent : 
  scenario.account1.balance = 4 * scenario.account2.balance ∧
  scenario.account2.rate = 0.06 ∧
  interest scenario.account1 + interest scenario.account2 = scenario.totalInterest ∧
  scenario.account1.rate = 0.05 := by
  sorry

#check investment_rate_is_five_percent

end NUMINAMATH_CALUDE_investment_rate_is_five_percent_l1517_151704


namespace NUMINAMATH_CALUDE_johns_allowance_l1517_151729

theorem johns_allowance (allowance : ℚ) : 
  (allowance * (2/5) * (2/3) = 64/100) → allowance = 24/10 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l1517_151729


namespace NUMINAMATH_CALUDE_two_cos_forty_five_equals_sqrt_two_l1517_151706

theorem two_cos_forty_five_equals_sqrt_two : 2 * Real.cos (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_forty_five_equals_sqrt_two_l1517_151706


namespace NUMINAMATH_CALUDE_sum_first_three_special_sequence_l1517_151773

/-- An arithmetic sequence with given fourth, fifth, and sixth terms -/
def ArithmeticSequence (a₄ a₅ a₆ : ℤ) : ℕ → ℤ :=
  fun n => a₄ + (n - 4) * (a₅ - a₄)

/-- The sum of the first three terms of an arithmetic sequence -/
def SumFirstThree (seq : ℕ → ℤ) : ℤ :=
  seq 1 + seq 2 + seq 3

theorem sum_first_three_special_sequence :
  let seq := ArithmeticSequence 4 7 10
  SumFirstThree seq = -6 := by sorry

end NUMINAMATH_CALUDE_sum_first_three_special_sequence_l1517_151773


namespace NUMINAMATH_CALUDE_action_figure_price_l1517_151724

theorem action_figure_price 
  (sneaker_cost : ℕ) 
  (initial_savings : ℕ) 
  (num_figures_sold : ℕ) 
  (money_left : ℕ) : 
  sneaker_cost = 90 →
  initial_savings = 15 →
  num_figures_sold = 10 →
  money_left = 25 →
  (sneaker_cost - initial_savings + money_left) / num_figures_sold = 10 := by
sorry

end NUMINAMATH_CALUDE_action_figure_price_l1517_151724


namespace NUMINAMATH_CALUDE_parabola_focus_l1517_151770

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a))
  ∀ (x y : ℝ), y = a * x^2 → (x - f.1)^2 = 4 * (1 / (4 * a)) * (y - f.2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1517_151770


namespace NUMINAMATH_CALUDE_compound_hydrogen_atoms_l1517_151717

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  al : ℕ
  o : ℕ
  h : ℕ

/-- Represents the atomic weights of elements in g/mol -/
structure AtomicWeights where
  al : ℝ
  o : ℝ
  h : ℝ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (c : Compound) (w : AtomicWeights) : ℝ :=
  c.al * w.al + c.o * w.o + c.h * w.h

/-- The theorem to be proved -/
theorem compound_hydrogen_atoms :
  let c : Compound := { al := 1, o := 3, h := 3 }
  let w : AtomicWeights := { al := 27, o := 16, h := 1 }
  molecularWeight c w = 78 := by
  sorry

end NUMINAMATH_CALUDE_compound_hydrogen_atoms_l1517_151717


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l1517_151776

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence starting at 2, with common difference 4, 
    and last term 2014 has 504 terms -/
theorem arithmetic_sequence_2_to_2014 : 
  arithmetic_sequence_length 2 4 2014 = 504 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2_to_2014_l1517_151776


namespace NUMINAMATH_CALUDE_angle_quadrant_for_defined_log_l1517_151700

theorem angle_quadrant_for_defined_log (θ : Real) :
  (∃ x, x = Real.log (Real.cos θ * Real.tan θ)) →
  (0 ≤ θ ∧ θ < Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_quadrant_for_defined_log_l1517_151700


namespace NUMINAMATH_CALUDE_all_expressions_distinct_exactly_five_distinct_expressions_l1517_151787

/-- Represents the different ways to parenthesize 3^(3^(3^3)) -/
inductive ExpressionType
  | Type1  -- 3^(3^(3^3))
  | Type2  -- 3^((3^3)^3)
  | Type3  -- ((3^3)^3)^3
  | Type4  -- (3^(3^3))^3
  | Type5  -- (3^3)^(3^3)

/-- Evaluates the expression based on its type -/
noncomputable def evaluate (e : ExpressionType) : ℕ :=
  match e with
  | ExpressionType.Type1 => 3^(3^(3^3))
  | ExpressionType.Type2 => 3^((3^3)^3)
  | ExpressionType.Type3 => ((3^3)^3)^3
  | ExpressionType.Type4 => (3^(3^3))^3
  | ExpressionType.Type5 => (3^3)^(3^3)

/-- Theorem stating that all expression types result in distinct values -/
theorem all_expressions_distinct :
  ∀ (e1 e2 : ExpressionType), e1 ≠ e2 → evaluate e1 ≠ evaluate e2 := by
  sorry

/-- Theorem stating that there are exactly 5 distinct ways to parenthesize the expression -/
theorem exactly_five_distinct_expressions :
  ∃! (s : Finset ExpressionType), (∀ e, e ∈ s) ∧ s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_distinct_exactly_five_distinct_expressions_l1517_151787


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1517_151723

/-- Given a train of length 600 m crossing an overbridge of length 100 m in 70 seconds,
    prove that the speed of the train is 36 km/h. -/
theorem train_speed_calculation (train_length : Real) (overbridge_length : Real) (crossing_time : Real)
    (h1 : train_length = 600)
    (h2 : overbridge_length = 100)
    (h3 : crossing_time = 70) :
    (train_length + overbridge_length) / crossing_time * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1517_151723


namespace NUMINAMATH_CALUDE_function_upper_bound_l1517_151778

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)

/-- A function bounded on [0,1] -/
def BoundedOnUnitInterval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x → x ≤ 1 → |f x| ≤ 1997

/-- The main theorem -/
theorem function_upper_bound
  (f : ℝ → ℝ)
  (h1 : SatisfiesInequality f)
  (h2 : BoundedOnUnitInterval f) :
  ∀ x : ℝ, x ≥ 0 → f x ≤ x^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_l1517_151778


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1517_151730

-- Define the circle
def circle_center : ℝ × ℝ := (-2, 1)

-- Define the diameter endpoints
def diameter_endpoint_x : ℝ → ℝ × ℝ := λ a => (a, 0)
def diameter_endpoint_y : ℝ → ℝ × ℝ := λ b => (0, b)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y = 0

-- Theorem statement
theorem circle_equation_proof :
  ∃ (a b : ℝ),
    (diameter_endpoint_x a).1 + (diameter_endpoint_y b).1 = 2 * circle_center.1 ∧
    (diameter_endpoint_x a).2 + (diameter_endpoint_y b).2 = 2 * circle_center.2 →
    ∀ (x y : ℝ),
      (x - circle_center.1)^2 + (y - circle_center.2)^2 = 
        ((diameter_endpoint_x a).1 - (diameter_endpoint_y b).1)^2 / 4 +
        ((diameter_endpoint_x a).2 - (diameter_endpoint_y b).2)^2 / 4 →
      circle_equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1517_151730


namespace NUMINAMATH_CALUDE_jason_total_cost_l1517_151783

def stove_cost : ℚ := 1200
def wall_cost : ℚ := stove_cost / 6
def repair_cost : ℚ := stove_cost + wall_cost
def labor_fee_rate : ℚ := 1/5  -- 20% as a fraction

def total_cost : ℚ := repair_cost + (labor_fee_rate * repair_cost)

theorem jason_total_cost : total_cost = 1680 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_cost_l1517_151783


namespace NUMINAMATH_CALUDE_equation_solution_l1517_151738

theorem equation_solution : 
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + (3 : ℚ) / (n + 2) + n / (n + 2) = 4 ∧ n = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1517_151738


namespace NUMINAMATH_CALUDE_m_range_l1517_151728

theorem m_range (p q : Prop) (m : ℝ) 
  (hp : ∀ x : ℝ, 2*x - x^2 < m)
  (hq : m^2 - 2*m - 3 ≥ 0)
  (hnp : ¬(¬p))
  (hpq : ¬(p ∧ q)) :
  1 < m ∧ m < 3 := by
sorry

end NUMINAMATH_CALUDE_m_range_l1517_151728


namespace NUMINAMATH_CALUDE_parabola_hyperbola_disjunction_l1517_151708

-- Define the propositions
def p : Prop := ∀ y : ℝ, (∃ x : ℝ, x = 4 * y^2) → (∃ x : ℝ, x = 1)

def q : Prop := ∃ x y : ℝ, (x^2 / 4 - y^2 / 5 = -1) ∧ (x = 0 ∧ y = 3)

-- Theorem to prove
theorem parabola_hyperbola_disjunction : p ∨ q := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_disjunction_l1517_151708


namespace NUMINAMATH_CALUDE_cube_surface_area_approx_l1517_151794

-- Define the dimensions of the rectangular prism
def prism_length : ℝ := 10
def prism_width : ℝ := 5
def prism_height : ℝ := 24

-- Define the volume of the rectangular prism
def prism_volume : ℝ := prism_length * prism_width * prism_height

-- Define the edge length of the cube with the same volume
def cube_edge : ℝ := (prism_volume) ^ (1/3)

-- Define the surface area of the cube
def cube_surface_area : ℝ := 6 * (cube_edge ^ 2)

-- Theorem stating that the surface area of the cube is approximately 677.76 square inches
theorem cube_surface_area_approx :
  ∃ ε > 0, |cube_surface_area - 677.76| < ε :=
sorry

end NUMINAMATH_CALUDE_cube_surface_area_approx_l1517_151794


namespace NUMINAMATH_CALUDE_tiles_required_for_room_floor_l1517_151747

def room_length : Real := 6.24
def room_width : Real := 4.32
def tile_side : Real := 0.30

theorem tiles_required_for_room_floor :
  ⌈(room_length * room_width) / (tile_side * tile_side)⌉ = 300 := by
  sorry

end NUMINAMATH_CALUDE_tiles_required_for_room_floor_l1517_151747


namespace NUMINAMATH_CALUDE_arithmetic_sequence_variance_l1517_151745

/-- Given an arithmetic sequence with common difference d,
    prove that if the variance of the first five terms is 2, then d = ±1 -/
theorem arithmetic_sequence_variance (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  ((a 1 - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 2*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 3*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 4*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2) / 5 = 2 →
  d = 1 ∨ d = -1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_variance_l1517_151745


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l1517_151740

theorem function_satisfies_equation (a b c : ℝ) (h : a ≠ b) :
  let f : ℝ → ℝ := λ x ↦ (c / (a - b)) * x
  ∀ x, a * f (x - 1) + b * f (1 - x) = c * x := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l1517_151740


namespace NUMINAMATH_CALUDE_janet_pill_count_l1517_151782

def pills_per_day_first_two_weeks : ℕ := 2 + 3
def pills_per_day_last_two_weeks : ℕ := 2 + 1
def days_per_week : ℕ := 7
def weeks_in_month : ℕ := 4

theorem janet_pill_count :
  (pills_per_day_first_two_weeks * days_per_week * (weeks_in_month / 2)) +
  (pills_per_day_last_two_weeks * days_per_week * (weeks_in_month / 2)) = 112 :=
by sorry

end NUMINAMATH_CALUDE_janet_pill_count_l1517_151782


namespace NUMINAMATH_CALUDE_seven_thousand_six_hundred_scientific_notation_l1517_151796

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem seven_thousand_six_hundred_scientific_notation :
  toScientificNotation 7600 = ScientificNotation.mk 7.6 3 sorry sorry :=
sorry

end NUMINAMATH_CALUDE_seven_thousand_six_hundred_scientific_notation_l1517_151796


namespace NUMINAMATH_CALUDE_x_cubed_coef_sum_l1517_151707

def binomial_coef (n k : ℕ) : ℤ := (-1)^k * (n.choose k)

def expansion_coef (n : ℕ) : ℤ := binomial_coef n 3

theorem x_cubed_coef_sum :
  expansion_coef 5 + expansion_coef 6 + expansion_coef 7 + expansion_coef 8 = -121 :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_coef_sum_l1517_151707
