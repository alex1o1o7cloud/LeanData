import Mathlib

namespace NUMINAMATH_CALUDE_equality_division_property_l3368_336890

theorem equality_division_property (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) :
  a / (c^2) = b / (c^2) := by sorry

end NUMINAMATH_CALUDE_equality_division_property_l3368_336890


namespace NUMINAMATH_CALUDE_motel_payment_savings_l3368_336801

/-- Calculates the savings when choosing monthly payments over weekly payments for a motel stay. -/
theorem motel_payment_savings 
  (weeks_per_month : ℕ) 
  (total_months : ℕ) 
  (weekly_rate : ℕ) 
  (monthly_rate : ℕ) 
  (h1 : weeks_per_month = 4) 
  (h2 : total_months = 3) 
  (h3 : weekly_rate = 280) 
  (h4 : monthly_rate = 1000) : 
  (total_months * weeks_per_month * weekly_rate) - (total_months * monthly_rate) = 360 := by
  sorry

#check motel_payment_savings

end NUMINAMATH_CALUDE_motel_payment_savings_l3368_336801


namespace NUMINAMATH_CALUDE_monomial_sequence_matches_pattern_l3368_336837

/-- A sequence of monomials where the nth term is given by (-1)^n * (n+1) * a^(2n) -/
def monomial_sequence (n : ℕ) (a : ℝ) : ℝ :=
  (-1)^n * (n + 1 : ℝ) * a^(2 * n)

/-- The first few terms of the sequence match the given pattern -/
theorem monomial_sequence_matches_pattern (a : ℝ) :
  (monomial_sequence 1 a = -2 * a^2) ∧
  (monomial_sequence 2 a = 3 * a^4) ∧
  (monomial_sequence 3 a = -4 * a^6) ∧
  (monomial_sequence 4 a = 5 * a^8) ∧
  (monomial_sequence 5 a = -6 * a^10) ∧
  (monomial_sequence 6 a = 7 * a^12) :=
by sorry

end NUMINAMATH_CALUDE_monomial_sequence_matches_pattern_l3368_336837


namespace NUMINAMATH_CALUDE_cube_diagonals_count_l3368_336885

structure Cube where
  vertices : Nat
  edges : Nat

def face_diagonals (c : Cube) : Nat := 12

def space_diagonals (c : Cube) : Nat := 4

def total_diagonals (c : Cube) : Nat := face_diagonals c + space_diagonals c

theorem cube_diagonals_count (c : Cube) (h1 : c.vertices = 8) (h2 : c.edges = 12) :
  total_diagonals c = 16 ∧ face_diagonals c = 12 ∧ space_diagonals c = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonals_count_l3368_336885


namespace NUMINAMATH_CALUDE_tournament_configuration_impossible_l3368_336823

structure Tournament where
  num_teams : Nat
  games_played : Fin num_teams → Nat

def is_valid_configuration (t : Tournament) : Prop :=
  t.num_teams = 12 ∧
  (∃ i : Fin t.num_teams, t.games_played i = 11) ∧
  (∃ i j k : Fin t.num_teams, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    t.games_played i = 9 ∧ t.games_played j = 9 ∧ t.games_played k = 9) ∧
  (∃ i j : Fin t.num_teams, i ≠ j ∧ 
    t.games_played i = 6 ∧ t.games_played j = 6) ∧
  (∃ i j k l : Fin t.num_teams, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    t.games_played i = 4 ∧ t.games_played j = 4 ∧ t.games_played k = 4 ∧ t.games_played l = 4) ∧
  (∃ i j : Fin t.num_teams, i ≠ j ∧ 
    t.games_played i = 1 ∧ t.games_played j = 1)

theorem tournament_configuration_impossible :
  ¬∃ t : Tournament, is_valid_configuration t := by
  sorry

end NUMINAMATH_CALUDE_tournament_configuration_impossible_l3368_336823


namespace NUMINAMATH_CALUDE_remainder_1493825_div_6_l3368_336831

theorem remainder_1493825_div_6 : (1493825 % 6 = 5) := by
  sorry

end NUMINAMATH_CALUDE_remainder_1493825_div_6_l3368_336831


namespace NUMINAMATH_CALUDE_investment_sum_l3368_336805

/-- Proves that if a sum P invested at 18% p.a. for two years yields Rs. 504 more interest
    than if invested at 12% p.a. for the same period, then P = 4200. -/
theorem investment_sum (P : ℚ) : 
  (P * 18 * 2 / 100) - (P * 12 * 2 / 100) = 504 → P = 4200 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l3368_336805


namespace NUMINAMATH_CALUDE_ear_muffs_december_l3368_336830

theorem ear_muffs_december (before_december : ℕ) (total : ℕ) (during_december : ℕ) : 
  before_december = 1346 →
  total = 7790 →
  during_december = total - before_december →
  during_december = 6444 := by
sorry

end NUMINAMATH_CALUDE_ear_muffs_december_l3368_336830


namespace NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l3368_336894

-- Define the size of the grid
def gridSize : ℕ := 5

-- Define the number of dots to choose
def chosenDots : ℕ := 5

-- Define the number of collinear sets of 5 dots in a 5x5 grid
def collinearSets : ℕ := 12

-- Define the total number of ways to choose 5 dots out of 25
def totalCombinations : ℕ := Nat.choose (gridSize * gridSize) chosenDots

-- Theorem statement
theorem collinear_probability_in_5x5_grid :
  (collinearSets : ℚ) / totalCombinations = 2 / 8855 :=
sorry

end NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l3368_336894


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3368_336850

theorem binomial_expansion_problem (n : ℕ) (x : ℝ) :
  (∃ (a b : ℕ), a ≠ b ∧ a > 2 ∧ b > 2 ∧ (Nat.choose n a = Nat.choose n b)) →
  (n = 6 ∧ 
   ∃ (k : ℕ), k = 3 ∧
   ((-1)^k * 2^(n-k) * Nat.choose n k : ℤ) = -160) := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3368_336850


namespace NUMINAMATH_CALUDE_range_of_a_l3368_336871

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) →
  a ≤ -2 ∨ (1/4 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3368_336871


namespace NUMINAMATH_CALUDE_mean_squared_sum_l3368_336814

theorem mean_squared_sum (a b c : ℝ) 
  (h_arithmetic : (a + b + c) / 3 = 7)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 5) :
  a^2 + b^2 + c^2 = 181.8 := by
  sorry

end NUMINAMATH_CALUDE_mean_squared_sum_l3368_336814


namespace NUMINAMATH_CALUDE_pencil_sharpening_l3368_336835

/-- The length sharpened off a pencil is the difference between its original length and its length after sharpening. -/
theorem pencil_sharpening (original_length after_sharpening_length : ℝ) 
  (h1 : original_length = 31.25)
  (h2 : after_sharpening_length = 14.75) :
  original_length - after_sharpening_length = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l3368_336835


namespace NUMINAMATH_CALUDE_amy_local_calls_l3368_336863

/-- Proves that Amy made 15 local calls given the conditions of the problem -/
theorem amy_local_calls :
  ∀ (L I : ℕ),
  (L : ℚ) / I = 5 / 2 →
  L / (I + 3) = 5 / 3 →
  L = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_local_calls_l3368_336863


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3368_336887

def P : Set (ℕ × ℕ) := Set.univ

def f : ℕ → ℕ → ℝ
  | p, q => p * q

theorem f_satisfies_conditions :
  (∀ p q : ℕ, p * q = 0 → f p q = 0) ∧
  (∀ p q : ℕ, p * q ≠ 0 → f p q = 1 + 1/2 * f (p+1) (q-1) + 1/2 * f (p-1) (q+1)) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3368_336887


namespace NUMINAMATH_CALUDE_jake_fewer_peaches_indeterminate_peach_difference_l3368_336851

-- Define the number of apples and peaches for Steven
def steven_apples : ℕ := 52
def steven_peaches : ℕ := 13

-- Define Jake's apples in terms of Steven's
def jake_apples : ℕ := steven_apples + 84

-- Define a variable for Jake's peaches (unknown, but less than Steven's)
variable (jake_peaches : ℕ)

-- Theorem stating that Jake's peaches are fewer than Steven's
theorem jake_fewer_peaches : jake_peaches < steven_peaches := by sorry

-- Theorem stating that the exact difference in peaches cannot be determined
theorem indeterminate_peach_difference :
  ¬ ∃ (diff : ℕ), ∀ (jake_peaches : ℕ), jake_peaches < steven_peaches →
    steven_peaches - jake_peaches = diff := by sorry

end NUMINAMATH_CALUDE_jake_fewer_peaches_indeterminate_peach_difference_l3368_336851


namespace NUMINAMATH_CALUDE_cockatiel_eats_fifty_grams_weekly_l3368_336856

/-- The amount of birdseed a cockatiel eats per week -/
def cockatiel_weekly_consumption (
  boxes_bought : ℕ
  ) (boxes_in_pantry : ℕ
  ) (parrot_weekly_consumption : ℕ
  ) (grams_per_box : ℕ
  ) (weeks_of_feeding : ℕ
  ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let parrot_total_consumption := parrot_weekly_consumption * weeks_of_feeding
  let cockatiel_total_consumption := total_grams - parrot_total_consumption
  cockatiel_total_consumption / weeks_of_feeding

/-- Theorem stating that given the conditions in the problem, 
    the cockatiel eats 50 grams of seeds each week -/
theorem cockatiel_eats_fifty_grams_weekly :
  cockatiel_weekly_consumption 3 5 100 225 12 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cockatiel_eats_fifty_grams_weekly_l3368_336856


namespace NUMINAMATH_CALUDE_abs_negative_two_l3368_336878

theorem abs_negative_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l3368_336878


namespace NUMINAMATH_CALUDE_back_lot_filled_fraction_l3368_336876

/-- Proves that the fraction of the back parking lot filled is 1/2 -/
theorem back_lot_filled_fraction :
  let front_spaces : ℕ := 52
  let back_spaces : ℕ := 38
  let total_spaces : ℕ := front_spaces + back_spaces
  let parked_cars : ℕ := 39
  let available_spaces : ℕ := 32
  let filled_back_spaces : ℕ := total_spaces - parked_cars - available_spaces
  (filled_back_spaces : ℚ) / back_spaces = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_back_lot_filled_fraction_l3368_336876


namespace NUMINAMATH_CALUDE_arccos_value_from_arcsin_inequality_l3368_336849

theorem arccos_value_from_arcsin_inequality (a b : ℝ) :
  Real.arcsin (1 + a^2) - Real.arcsin ((b - 1)^2) ≥ π / 2 →
  Real.arccos (a^2 - b^2) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_value_from_arcsin_inequality_l3368_336849


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3368_336883

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 2 - Complex.I) → z = -1 - 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3368_336883


namespace NUMINAMATH_CALUDE_divided_triangle_perimeter_l3368_336811

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- Theorem stating the relationship between the perimeters of the large and small triangles -/
theorem divided_triangle_perimeter
  (t : DividedTriangle)
  (h1 : t.large_perimeter = 120)
  (h2 : t.num_small_triangles = 9)
  (h3 : t.small_perimeter * 3 = t.large_perimeter) :
  t.small_perimeter = 40 :=
sorry

end NUMINAMATH_CALUDE_divided_triangle_perimeter_l3368_336811


namespace NUMINAMATH_CALUDE_common_tangent_lines_l3368_336858

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_E (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 1

-- Define the potential tangent lines
def line1 (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - Real.sqrt 3 - 2 = 0
def line3 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - Real.sqrt 3 + 2 = 0

-- Define what it means for a line to be tangent to a circle
def is_tangent_to (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y ∧
  ∀ (x' y' : ℝ), line x' y' → circle x' y' → (x' = x ∧ y' = y)

-- State the theorem
theorem common_tangent_lines :
  (is_tangent_to line1 circle_C ∧ is_tangent_to line1 circle_E) ∧
  (is_tangent_to line2 circle_C ∧ is_tangent_to line2 circle_E) ∧
  (is_tangent_to line3 circle_C ∧ is_tangent_to line3 circle_E) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_lines_l3368_336858


namespace NUMINAMATH_CALUDE_exists_alpha_floor_minus_n_even_l3368_336874

theorem exists_alpha_floor_minus_n_even :
  ∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n > 0 → Even (⌊α * n⌋ - n) := by
  sorry

end NUMINAMATH_CALUDE_exists_alpha_floor_minus_n_even_l3368_336874


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3368_336839

theorem quadratic_root_difference :
  let a : ℝ := 3 + 2 * Real.sqrt 2
  let b : ℝ := 5 + Real.sqrt 2
  let c : ℝ := -4
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / a
  root_difference = Real.sqrt (177 - 122 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3368_336839


namespace NUMINAMATH_CALUDE_simplify_sqrt_x6_plus_x3_l3368_336899

theorem simplify_sqrt_x6_plus_x3 (x : ℝ) : 
  Real.sqrt (x^6 + x^3) = |x| * Real.sqrt |x| * Real.sqrt (x^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_x6_plus_x3_l3368_336899


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3368_336828

theorem solution_set_inequality (x : ℝ) :
  (x * (x + 2) < 3) ↔ (-3 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3368_336828


namespace NUMINAMATH_CALUDE_solve_for_b_l3368_336884

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3368_336884


namespace NUMINAMATH_CALUDE_tom_family_plates_l3368_336853

/-- Calculates the total number of plates used by a family during a stay -/
def total_plates_used (family_size : ℕ) (days : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) : ℕ :=
  family_size * days * meals_per_day * plates_per_meal

/-- Theorem: The total number of plates used by Tom's family during their 4-day stay is 144 -/
theorem tom_family_plates : 
  total_plates_used 6 4 3 2 = 144 := by
  sorry


end NUMINAMATH_CALUDE_tom_family_plates_l3368_336853


namespace NUMINAMATH_CALUDE_simplify_expression_l3368_336844

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  10 * x^3 * y^2 / (15 * x^2 * y^3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3368_336844


namespace NUMINAMATH_CALUDE_defective_units_shipped_l3368_336819

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.06)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * total_units) / total_units = 0.0024 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l3368_336819


namespace NUMINAMATH_CALUDE_investment_principal_l3368_336836

/-- Proves that given an investment with a monthly interest payment of $216 and a simple annual interest rate of 9%, the principal amount of the investment is $28,800. -/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 216 →
  annual_rate = 0.09 →
  (monthly_interest * 12) / annual_rate = 28800 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_l3368_336836


namespace NUMINAMATH_CALUDE_correlation_coefficient_inequality_l3368_336870

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x y : List ℝ) : ℝ :=
  sorry

def r₁ : ℝ := linear_correlation_coefficient X Y
def r₂ : ℝ := linear_correlation_coefficient U V

theorem correlation_coefficient_inequality : r₂ < 0 ∧ 0 < r₁ := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_inequality_l3368_336870


namespace NUMINAMATH_CALUDE_solve_shoe_price_l3368_336860

def shoe_price_problem (rebate_percentage : ℝ) (num_pairs : ℕ) (total_rebate : ℝ) : Prop :=
  let original_price := total_rebate / (rebate_percentage * num_pairs : ℝ)
  original_price = 28

theorem solve_shoe_price :
  shoe_price_problem 0.1 5 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_shoe_price_l3368_336860


namespace NUMINAMATH_CALUDE_two_natural_numbers_problem_l3368_336848

theorem two_natural_numbers_problem :
  ∃ (x y : ℕ), x > y ∧ 
    x + y = 5 * (x - y) ∧
    x * y = 24 * (x - y) ∧
    x = 12 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_natural_numbers_problem_l3368_336848


namespace NUMINAMATH_CALUDE_log_difference_inequality_l3368_336886

theorem log_difference_inequality (a b : ℝ) : 
  Real.log a - Real.log b = 3 * b - a → a > b ∧ b > 0 := by sorry

end NUMINAMATH_CALUDE_log_difference_inequality_l3368_336886


namespace NUMINAMATH_CALUDE_sequence_and_sum_formula_l3368_336888

def sequence_a (n : ℕ) : ℚ := (3^n - 1) / 2

def S (n : ℕ) : ℚ := (3^(n+2) - 9) / 8 - n * (n+4) / 4

theorem sequence_and_sum_formula :
  (∀ n : ℕ, ∃ q : ℚ, sequence_a (n+1) + 1/2 = q * (sequence_a n + 1/2)) ∧ 
  (sequence_a 1 + 1/2 = 3/2) ∧
  (sequence_a 4 - sequence_a 1 = 39) →
  (∀ n : ℕ, sequence_a n = (3^n - 1) / 2) ∧
  (∀ n : ℕ, S n = (3^(n+2) - 9) / 8 - n * (n+4) / 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_and_sum_formula_l3368_336888


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3368_336852

/-- Given an arithmetic sequence where the first term is 5 and the common difference is 2,
    prove that the 15th term is equal to 33. -/
theorem fifteenth_term_of_sequence (a : ℕ → ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, a (n + 1) = a n + 2) →
  a 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3368_336852


namespace NUMINAMATH_CALUDE_dihedral_angle_eq_inclination_l3368_336822

/-- A pyramid with an isosceles triangular base and inclined lateral edges -/
structure IsoscelesPyramid where
  -- Angle between equal sides of the base triangle
  α : Real
  -- Angle of inclination of lateral edges to the base plane
  φ : Real
  -- Assumption that α and φ are valid angles
  h_α_range : 0 < α ∧ α < π
  h_φ_range : 0 < φ ∧ φ < π/2

/-- The dihedral angle at the edge connecting the apex to the vertex of angle α -/
def dihedral_angle (p : IsoscelesPyramid) : Real :=
  -- Definition of dihedral angle (to be proved equal to φ)
  sorry

/-- Theorem: The dihedral angle is equal to the inclination angle of lateral edges -/
theorem dihedral_angle_eq_inclination (p : IsoscelesPyramid) :
  dihedral_angle p = p.φ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_eq_inclination_l3368_336822


namespace NUMINAMATH_CALUDE_circle_k_value_l3368_336882

def larger_circle_radius : ℝ := 15
def smaller_circle_radius : ℝ := 10
def point_P : ℝ × ℝ := (9, 12)
def point_S (k : ℝ) : ℝ × ℝ := (0, k)
def QR : ℝ := 5

theorem circle_k_value :
  ∀ k : ℝ,
  (point_P.1^2 + point_P.2^2 = larger_circle_radius^2) →
  ((point_S k).1^2 + (point_S k).2^2 = smaller_circle_radius^2) →
  (larger_circle_radius - smaller_circle_radius = QR) →
  (k = 10 ∨ k = -10) :=
by sorry

end NUMINAMATH_CALUDE_circle_k_value_l3368_336882


namespace NUMINAMATH_CALUDE_perpendicular_bisecting_diagonals_not_imply_square_l3368_336816

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def has_perpendicular_bisecting_diagonals (q : Quadrilateral) : Prop :=
  sorry

def is_square (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_bisecting_diagonals_not_imply_square :
  ¬ (∀ q : Quadrilateral, has_perpendicular_bisecting_diagonals q → is_square q) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisecting_diagonals_not_imply_square_l3368_336816


namespace NUMINAMATH_CALUDE_find_k_l3368_336866

-- Define the polynomials A and B
def A (x k : ℝ) : ℝ := 2 * x^2 + k * x - 6 * x

def B (x k : ℝ) : ℝ := -x^2 + k * x - 1

-- Define the condition for A + 2B to be independent of x
def independent_of_x (k : ℝ) : Prop :=
  ∀ x : ℝ, ∃ c : ℝ, A x k + 2 * B x k = c

-- Theorem statement
theorem find_k : ∃ k : ℝ, independent_of_x k ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_find_k_l3368_336866


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l3368_336834

theorem relationship_between_exponents 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b^2) 
  (h3 : c^(2*y) = a^(3*z)) 
  (h4 : c^(2*y) = d^2) 
  (h5 : a ≠ 0) 
  (h6 : b ≠ 0) 
  (h7 : c ≠ 0) 
  (h8 : d ≠ 0) : 
  9*q*z = 4*x*y := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l3368_336834


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3368_336813

theorem vector_difference_magnitude : 
  let a : ℝ × ℝ := (Real.cos (π / 6), Real.sin (π / 6))
  let b : ℝ × ℝ := (Real.cos (5 * π / 6), Real.sin (5 * π / 6))
  ((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3368_336813


namespace NUMINAMATH_CALUDE_sum_of_squares_l3368_336825

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3368_336825


namespace NUMINAMATH_CALUDE_age_difference_proof_l3368_336868

theorem age_difference_proof (p m n : ℕ) 
  (h1 : 5 * p = 3 * m)  -- p:m = 3:5
  (h2 : 5 * m = 3 * n)  -- m:n = 3:5
  (h3 : p + m + n = 245) : 
  n - p = 80 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3368_336868


namespace NUMINAMATH_CALUDE_f_4_equals_1559_l3368_336862

-- Define the polynomial f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Define Horner's method for this specific polynomial
def horner (x : ℝ) : ℝ := ((((x + 3) * x - 5) * x + 7) * x - 9) * x + 11

-- Theorem stating that f(4) = 1559 using Horner's method
theorem f_4_equals_1559 : f 4 = 1559 ∧ horner 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_f_4_equals_1559_l3368_336862


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3368_336817

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 4 * x + 3 ∧
  (∀ (y : ℝ), y * |y| = 4 * y + 3 → x ≤ y) ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3368_336817


namespace NUMINAMATH_CALUDE_square_sum_given_square_sum_and_product_l3368_336820

theorem square_sum_given_square_sum_and_product
  (x y : ℝ) (h1 : (x + y)^2 = 25) (h2 : x * y = -6) :
  x^2 + y^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_square_sum_and_product_l3368_336820


namespace NUMINAMATH_CALUDE_expression_simplification_l3368_336832

theorem expression_simplification (a : ℚ) (h : a = -2) : 
  ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3*a) / (a^2 - 1)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3368_336832


namespace NUMINAMATH_CALUDE_greatest_x_value_achievable_value_l3368_336869

theorem greatest_x_value (x : ℝ) : 
  (4 * x^2 + 6 * x + 3 = 5) → x ≤ (1/2 : ℝ) :=
by
  sorry

theorem achievable_value : 
  ∃ x : ℝ, (4 * x^2 + 6 * x + 3 = 5) ∧ x = (1/2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_achievable_value_l3368_336869


namespace NUMINAMATH_CALUDE_correct_cracker_distribution_l3368_336859

/-- Represents the distribution of crackers to friends -/
structure CrackerDistribution where
  initial : ℕ
  first_fraction : ℚ
  second_percentage : ℚ
  third_remaining : ℕ

/-- Calculates the number of crackers each friend receives -/
def distribute_crackers (d : CrackerDistribution) : ℕ × ℕ × ℕ := sorry

/-- Theorem stating the correct distribution of crackers -/
theorem correct_cracker_distribution :
  let d := CrackerDistribution.mk 100 (2/3) (37/200) 7
  distribute_crackers d = (66, 6, 7) := by sorry

end NUMINAMATH_CALUDE_correct_cracker_distribution_l3368_336859


namespace NUMINAMATH_CALUDE_distance_EC_l3368_336854

/-- Given five points A, B, C, D, E on a line, with known distances between consecutive points,
    prove that the distance between E and C is 150. -/
theorem distance_EC (A B C D E : ℝ) 
  (h_AB : |A - B| = 30)
  (h_BC : |B - C| = 80)
  (h_CD : |C - D| = 236)
  (h_DE : |D - E| = 86)
  (h_EA : |E - A| = 40)
  (h_line : ∃ (t : ℝ → ℝ), t A < t B ∧ t B < t C ∧ t C < t D ∧ t D < t E) :
  |E - C| = 150 := by
  sorry

end NUMINAMATH_CALUDE_distance_EC_l3368_336854


namespace NUMINAMATH_CALUDE_parking_solution_l3368_336880

def parking_problem (first_level second_level third_level fourth_level : ℕ) : Prop :=
  first_level = 4 ∧
  second_level = first_level + 7 ∧
  third_level > second_level ∧
  fourth_level = 14 ∧
  first_level + second_level + third_level + fourth_level = 46

theorem parking_solution :
  ∀ first_level second_level third_level fourth_level : ℕ,
  parking_problem first_level second_level third_level fourth_level →
  third_level - second_level = 6 :=
by
  sorry

#check parking_solution

end NUMINAMATH_CALUDE_parking_solution_l3368_336880


namespace NUMINAMATH_CALUDE_system_solutions_correct_l3368_336846

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, 2*x + y = 3 ∧ 3*x - 5*y = 11 ∧ x = 2 ∧ y = -1) ∧
  -- System 2
  (∃ a b c : ℝ, a + b + c = 0 ∧ a - b + c = -4 ∧ 4*a + 2*b + c = 5 ∧
                a = 1 ∧ b = 2 ∧ c = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l3368_336846


namespace NUMINAMATH_CALUDE_thomas_total_bill_l3368_336879

/-- Calculates the total bill for an international order including shipping and import taxes. -/
def calculate_total_bill (clothes_cost accessories_cost : ℝ)
  (clothes_shipping_rate accessories_shipping_rate : ℝ)
  (clothes_tax_rate accessories_tax_rate : ℝ) : ℝ :=
  let clothes_shipping := clothes_cost * clothes_shipping_rate
  let accessories_shipping := accessories_cost * accessories_shipping_rate
  let clothes_tax := clothes_cost * clothes_tax_rate
  let accessories_tax := accessories_cost * accessories_tax_rate
  clothes_cost + accessories_cost + clothes_shipping + accessories_shipping + clothes_tax + accessories_tax

/-- Theorem stating that Thomas's total bill is $162.20 -/
theorem thomas_total_bill :
  calculate_total_bill 85 36 0.3 0.15 0.1 0.05 = 162.20 := by
  sorry

end NUMINAMATH_CALUDE_thomas_total_bill_l3368_336879


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3368_336857

theorem remainder_divisibility (x : ℕ) (h : x > 0) :
  (200 % x = 2) → (398 % x = 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3368_336857


namespace NUMINAMATH_CALUDE_largest_positive_root_bound_l3368_336843

theorem largest_positive_root_bound (b₂ b₁ b₀ : ℝ) 
  (h₂ : |b₂| ≤ 3) (h₁ : |b₁| ≤ 5) (h₀ : |b₀| ≤ 3) :
  ∃ s : ℝ, s > 4 ∧ s < 5 ∧
  (∀ x : ℝ, x > 0 → x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ s) ∧
  (∃ b₂' b₁' b₀' : ℝ, |b₂'| ≤ 3 ∧ |b₁'| ≤ 5 ∧ |b₀'| ≤ 3 ∧
    s^3 + b₂'*s^2 + b₁'*s + b₀' = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_bound_l3368_336843


namespace NUMINAMATH_CALUDE_runner_speed_l3368_336893

/-- Proves that a runner covering 11.4 km in 2 minutes has a speed of 95 m/s -/
theorem runner_speed : ∀ (distance : ℝ) (time : ℝ),
  distance = 11.4 ∧ time = 2 →
  (distance * 1000) / (time * 60) = 95 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_l3368_336893


namespace NUMINAMATH_CALUDE_quadratic_radical_sum_l3368_336829

/-- 
Given that √(3b-1) and ∜(7-b) are of the same type of quadratic radical,
where ∜ represents the (a-1)th root, prove that a + b = 5.
-/
theorem quadratic_radical_sum (a b : ℝ) : 
  (a - 1 = 2) → (3*b - 1 = 7 - b) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_sum_l3368_336829


namespace NUMINAMATH_CALUDE_adams_earnings_l3368_336891

/-- Adam's daily earnings problem -/
theorem adams_earnings (daily_earnings : ℝ) : 
  (daily_earnings * 0.9 * 30 = 1080) → daily_earnings = 40 := by
  sorry

end NUMINAMATH_CALUDE_adams_earnings_l3368_336891


namespace NUMINAMATH_CALUDE_exp_13pi_div_2_l3368_336827

/-- Euler's formula -/
axiom euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ

/-- The main theorem -/
theorem exp_13pi_div_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_13pi_div_2_l3368_336827


namespace NUMINAMATH_CALUDE_video_game_lives_l3368_336873

theorem video_game_lives (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) :
  initial_players = 11 →
  quitting_players = 5 →
  total_lives = 30 →
  (total_lives / (initial_players - quitting_players) = 5) :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l3368_336873


namespace NUMINAMATH_CALUDE_all_numbers_on_diagonal_l3368_336800

/-- Represents a 15x15 table with numbers 1 to 15 -/
def Table := Fin 15 → Fin 15 → Fin 15

/-- The property that each number appears exactly once in each row -/
def row_property (t : Table) : Prop :=
  ∀ i j₁ j₂, j₁ ≠ j₂ → t i j₁ ≠ t i j₂

/-- The property that each number appears exactly once in each column -/
def column_property (t : Table) : Prop :=
  ∀ i₁ i₂ j, i₁ ≠ i₂ → t i₁ j ≠ t i₂ j

/-- The property that symmetrically placed numbers are identical -/
def symmetry_property (t : Table) : Prop :=
  ∀ i j, t i j = t j i

/-- The main theorem stating that all numbers appear on the main diagonal -/
theorem all_numbers_on_diagonal (t : Table)
  (h_row : row_property t)
  (h_col : column_property t)
  (h_sym : symmetry_property t) :
  ∀ n : Fin 15, ∃ i : Fin 15, t i i = n :=
sorry

end NUMINAMATH_CALUDE_all_numbers_on_diagonal_l3368_336800


namespace NUMINAMATH_CALUDE_josh_marbles_l3368_336808

theorem josh_marbles (lost : ℕ) (left : ℕ) (initial : ℕ) : 
  lost = 7 → left = 9 → initial = lost + left → initial = 16 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3368_336808


namespace NUMINAMATH_CALUDE_smallest_possible_student_count_l3368_336807

/-- The smallest possible number of students in a classroom with the given seating arrangement --/
def smallest_student_count : ℕ := 42

/-- The number of rows in the classroom --/
def num_rows : ℕ := 5

/-- Represents the number of students in each of the first four rows --/
def students_per_row : ℕ := 8

theorem smallest_possible_student_count :
  (num_rows - 1) * students_per_row + (students_per_row + 2) = smallest_student_count ∧
  smallest_student_count > 40 ∧
  ∀ n : ℕ, n < smallest_student_count →
    (num_rows - 1) * (n / num_rows) + (n / num_rows + 2) ≠ n ∨ n ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_student_count_l3368_336807


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3368_336847

theorem repeating_decimal_sum_difference (x y z : ℚ) :
  x = 5/9 ∧ y = 1/9 ∧ z = 3/9 → x + y - z = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3368_336847


namespace NUMINAMATH_CALUDE_min_dot_product_l3368_336809

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the hyperbola in the first quadrant
def point_on_hyperbola (M : ℝ × ℝ) : Prop :=
  hyperbola M.1 M.2 ∧ M.1 > 0 ∧ M.2 > 0

-- Define the tangent line at point M
def tangent_line (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  point_on_hyperbola M ∧ 
  ∃ (t : ℝ), P = (M.1 + t, M.2 + t) ∧ Q = (M.1 - t, M.2 - t)

-- Define P in the first quadrant
def P_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

-- Define R on the same asymptote as Q
def R_on_asymptote (Q R : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), R = (k * Q.1, k * Q.2)

-- Theorem statement
theorem min_dot_product 
  (M P Q R : ℝ × ℝ) 
  (h1 : tangent_line M P Q)
  (h2 : P_in_first_quadrant P)
  (h3 : R_on_asymptote Q R) :
  ∃ (min_value : ℝ), 
    (∀ (R' : ℝ × ℝ), R_on_asymptote Q R' → 
      (R'.1 - P.1) * (R'.1 - Q.1) + (R'.2 - P.2) * (R'.2 - Q.2) ≥ min_value) ∧
    min_value = -1/2 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l3368_336809


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3368_336842

theorem polynomial_remainder_theorem (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 30) % (4 * x - 8) = 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3368_336842


namespace NUMINAMATH_CALUDE_number_of_pupils_theorem_l3368_336889

/-- The number of pupils sent up for examination -/
def N : ℕ := 28

/-- The average marks of all pupils -/
def overall_average : ℚ := 39

/-- The average marks if 7 specific pupils were not sent up -/
def new_average : ℚ := 45

/-- The marks of the 7 specific pupils -/
def specific_pupils_marks : List ℕ := [25, 12, 15, 19, 31, 18, 27]

/-- The sum of marks of the 7 specific pupils -/
def sum_specific_marks : ℕ := specific_pupils_marks.sum

theorem number_of_pupils_theorem :
  (N * overall_average - sum_specific_marks) / (N - 7) = new_average :=
sorry

end NUMINAMATH_CALUDE_number_of_pupils_theorem_l3368_336889


namespace NUMINAMATH_CALUDE_q_over_p_is_five_thirds_l3368_336861

theorem q_over_p_is_five_thirds (P Q : ℤ) (h : ∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 →
  (P / (x + 6) + Q / (x^2 - 6*x) : ℝ) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x)) :
  (Q : ℚ) / (P : ℚ) = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_q_over_p_is_five_thirds_l3368_336861


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3368_336881

theorem perfect_square_condition (m : ℤ) : 
  (∃ n : ℤ, m^2 + 6*m + 28 = n^2) ↔ (m = 6 ∨ m = -12) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3368_336881


namespace NUMINAMATH_CALUDE_inequality_range_l3368_336812

theorem inequality_range (t : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → 
    (1/8) * (2*t - t^2) ≤ x^2 - 3*x + 2 ∧ 
    x^2 - 3*x + 2 ≤ 3 - t^2) ↔ 
  t ∈ Set.Icc (-1) (1 - Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3368_336812


namespace NUMINAMATH_CALUDE_laundry_detergent_cost_l3368_336815

def budget : ℕ := 60
def shower_gel_cost : ℕ := 4
def shower_gel_quantity : ℕ := 4
def toothpaste_cost : ℕ := 3
def remaining_budget : ℕ := 30

theorem laundry_detergent_cost :
  budget - remaining_budget - (shower_gel_cost * shower_gel_quantity + toothpaste_cost) = 11 := by
  sorry

end NUMINAMATH_CALUDE_laundry_detergent_cost_l3368_336815


namespace NUMINAMATH_CALUDE_min_value_w_l3368_336865

theorem min_value_w (x y : ℝ) : 3 * x^2 + 3 * y^2 + 12 * x - 6 * y + 30 ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_w_l3368_336865


namespace NUMINAMATH_CALUDE_friend_walking_problem_l3368_336867

/-- 
Given two friends walking towards each other on a trail:
- The trail length is 33 km
- They start at opposite ends at the same time
- One friend's speed is 20% faster than the other's
Prove that the faster friend will have walked 18 km when they meet.
-/
theorem friend_walking_problem (v : ℝ) (h_v_pos : v > 0) :
  let trail_length : ℝ := 33
  let speed_ratio : ℝ := 1.2
  let t : ℝ := trail_length / (v * (1 + speed_ratio))
  speed_ratio * v * t = 18 := by sorry

end NUMINAMATH_CALUDE_friend_walking_problem_l3368_336867


namespace NUMINAMATH_CALUDE_expression_simplification_l3368_336855

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / (a^2 - 2*a + 1) / (1 + 2 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3368_336855


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3368_336826

theorem sphere_surface_area (r : ℝ) (h : r = 2) : 4 * Real.pi * r^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3368_336826


namespace NUMINAMATH_CALUDE_average_age_combined_l3368_336872

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 33 →
  num_parents = 55 →
  avg_age_students = 11 →
  avg_age_parents = 33 →
  ((num_students : ℝ) * avg_age_students + (num_parents : ℝ) * avg_age_parents) / 
   ((num_students : ℝ) + (num_parents : ℝ)) = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l3368_336872


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3368_336877

theorem sphere_surface_area (R : ℝ) (r₁ r₂ d : ℝ) : 
  r₁ = 24 → r₂ = 15 → d = 27 → 
  R^2 = r₁^2 + x^2 → 
  R^2 = r₂^2 + (d - x)^2 → 
  4 * π * R^2 = 2500 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3368_336877


namespace NUMINAMATH_CALUDE_marble_difference_l3368_336895

/-- The number of marbles Amon and Rhonda have combined -/
def total_marbles : ℕ := 215

/-- The number of marbles Rhonda has -/
def rhonda_marbles : ℕ := 80

/-- Amon has more marbles than Rhonda -/
axiom amon_has_more : ∃ (amon_marbles : ℕ), amon_marbles > rhonda_marbles ∧ amon_marbles + rhonda_marbles = total_marbles

/-- The difference between Amon's and Rhonda's marbles is 55 -/
theorem marble_difference : ∃ (amon_marbles : ℕ), amon_marbles - rhonda_marbles = 55 := by sorry

end NUMINAMATH_CALUDE_marble_difference_l3368_336895


namespace NUMINAMATH_CALUDE_smallest_twin_egg_number_l3368_336875

def is_twin_egg_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ a ≠ b ∧ n = 1000 * a + 100 * b + 10 * b + a

def F (m : ℕ) : ℚ :=
  let m' := (m % 100) * 100 + (m / 100)
  (m - m') / 11

theorem smallest_twin_egg_number :
  ∃ (m : ℕ),
    is_twin_egg_number m ∧
    ∃ (k : ℕ), F m / 54 = k^2 ∧
    ∀ (n : ℕ), is_twin_egg_number n → (∃ (l : ℕ), F n / 54 = l^2) → m ≤ n ∧
    m = 7117 :=
sorry

end NUMINAMATH_CALUDE_smallest_twin_egg_number_l3368_336875


namespace NUMINAMATH_CALUDE_circular_path_time_increase_l3368_336804

/-- 
Prove that if a person can go round a circular path 8 times in 40 minutes, 
and the diameter of the circle is increased to 10 times the original diameter, 
then the time required to go round the new path once, traveling at the same speed as before, 
is 50 minutes.
-/
theorem circular_path_time_increase 
  (original_rounds : ℕ) 
  (original_time : ℕ) 
  (diameter_increase : ℕ) 
  (h1 : original_rounds = 8) 
  (h2 : original_time = 40) 
  (h3 : diameter_increase = 10) : 
  (original_time / original_rounds) * diameter_increase = 50 := by
  sorry

#check circular_path_time_increase

end NUMINAMATH_CALUDE_circular_path_time_increase_l3368_336804


namespace NUMINAMATH_CALUDE_last_s_replacement_l3368_336803

-- Define the alphabet size
def alphabet_size : ℕ := 26

-- Define the function to calculate the shift for the nth occurrence
def shift (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function to apply the shift modulo alphabet size
def apply_shift (shift : ℕ) : ℕ := shift % alphabet_size

-- Theorem statement
theorem last_s_replacement (occurrences : ℕ) (h : occurrences = 12) :
  apply_shift (shift occurrences) = 0 := by sorry

end NUMINAMATH_CALUDE_last_s_replacement_l3368_336803


namespace NUMINAMATH_CALUDE_power_division_rule_l3368_336838

theorem power_division_rule (x : ℝ) : x^10 / x^2 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3368_336838


namespace NUMINAMATH_CALUDE_parallelogram_area_32_22_l3368_336845

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 and height 22 is 704 -/
theorem parallelogram_area_32_22 : parallelogram_area 32 22 = 704 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_22_l3368_336845


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l3368_336818

/-- Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k,
    the value of h is -3/2. -/
theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a k : ℝ), 3*x^2 + 9*x + 20 = a*(x + 3/2)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l3368_336818


namespace NUMINAMATH_CALUDE_grid_polygon_segment_sum_equality_l3368_336810

/-- A point on a grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A polygon on a grid -/
structure GridPolygon where
  vertices : List GridPoint
  is_convex : Bool
  no_horizontal_vertical_sides : Bool

/-- The sum of lengths of grid segments within a polygon -/
def sum_grid_segments (p : GridPolygon) (direction : Bool) : ℝ :=
  sorry

theorem grid_polygon_segment_sum_equality (p : GridPolygon) 
  (h_convex : p.is_convex = true) 
  (h_no_hv : p.no_horizontal_vertical_sides = true) :
  sum_grid_segments p true = sum_grid_segments p false := by
  sorry

end NUMINAMATH_CALUDE_grid_polygon_segment_sum_equality_l3368_336810


namespace NUMINAMATH_CALUDE_farmers_market_sales_l3368_336802

theorem farmers_market_sales (total_earnings broccoli_sales cauliflower_sales : ℕ) 
  (h1 : total_earnings = 380)
  (h2 : broccoli_sales = 57)
  (h3 : cauliflower_sales = 136) :
  ∃ (spinach_sales : ℕ), 
    spinach_sales = 73 ∧ 
    spinach_sales > (2 * broccoli_sales) / 2 ∧
    total_earnings = broccoli_sales + (2 * broccoli_sales) + cauliflower_sales + spinach_sales :=
by
  sorry


end NUMINAMATH_CALUDE_farmers_market_sales_l3368_336802


namespace NUMINAMATH_CALUDE_odd_cube_plus_one_not_square_l3368_336824

theorem odd_cube_plus_one_not_square (n : ℤ) (h : Odd n) :
  ¬ ∃ x : ℤ, n^3 + 1 = x^2 := by
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_one_not_square_l3368_336824


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3368_336841

/-- An arithmetic sequence is a sequence where the difference between 
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem stating that for an arithmetic sequence satisfying 
    the given condition, 2a_9 - a_10 = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3368_336841


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3368_336896

theorem fifteenth_student_age
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (num_group2 : Nat)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 6)
  (h6 : avg_age_group2 = 16)
  : ℝ :=
by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l3368_336896


namespace NUMINAMATH_CALUDE_relationship_xyz_l3368_336898

noncomputable def x : ℝ := Real.sqrt 2
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 0.7 / Real.log 5

theorem relationship_xyz : z < y ∧ y < x := by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l3368_336898


namespace NUMINAMATH_CALUDE_max_digit_sum_l3368_336821

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n ≤ 2999 ∧ n % 13 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem max_digit_sum :
  ∃ (n : ℕ), is_valid_number n ∧
  ∀ (m : ℕ), is_valid_number m → digit_sum m ≤ digit_sum n ∧
  digit_sum n = 26 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l3368_336821


namespace NUMINAMATH_CALUDE_fish_left_in_tank_l3368_336840

def fish_tank_problem (initial_fish : ℕ) (fish_taken_out : ℕ) : Prop :=
  initial_fish ≥ fish_taken_out ∧ 
  initial_fish - fish_taken_out = 3

theorem fish_left_in_tank : fish_tank_problem 19 16 := by
  sorry

end NUMINAMATH_CALUDE_fish_left_in_tank_l3368_336840


namespace NUMINAMATH_CALUDE_train_length_l3368_336897

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 12 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3368_336897


namespace NUMINAMATH_CALUDE_power_three_250_mod_13_l3368_336864

theorem power_three_250_mod_13 : 3^250 % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_three_250_mod_13_l3368_336864


namespace NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3368_336833

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_diff_and_product_l3368_336833


namespace NUMINAMATH_CALUDE_butterfly_cocoon_time_l3368_336892

theorem butterfly_cocoon_time :
  ∀ (cocoon_time larva_time : ℕ),
    cocoon_time + larva_time = 120 →
    larva_time = 3 * cocoon_time →
    cocoon_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_cocoon_time_l3368_336892


namespace NUMINAMATH_CALUDE_median_squares_sum_l3368_336806

/-- Given a triangle with side lengths 13, 14, and 15, the sum of the squares of its median lengths is 442.5 -/
theorem median_squares_sum (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let median_sum_squares := 3/4 * (a^2 + b^2 + c^2)
  median_sum_squares = 442.5 := by
  sorry

end NUMINAMATH_CALUDE_median_squares_sum_l3368_336806
