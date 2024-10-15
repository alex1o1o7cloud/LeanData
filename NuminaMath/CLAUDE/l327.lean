import Mathlib

namespace NUMINAMATH_CALUDE_age_difference_l327_32739

theorem age_difference (man_age son_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l327_32739


namespace NUMINAMATH_CALUDE_estimate_product_l327_32769

def approximate_819 : ℕ := 800
def approximate_32 : ℕ := 30

theorem estimate_product : 
  approximate_819 * approximate_32 = 24000 := by sorry

end NUMINAMATH_CALUDE_estimate_product_l327_32769


namespace NUMINAMATH_CALUDE_max_y_value_l327_32763

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : x * y * (x + y) = x - y) : 
  y ≤ 1/3 ∧ ∃ (y0 : ℝ), y0 * x * (x + y0) = x - y0 ∧ y0 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l327_32763


namespace NUMINAMATH_CALUDE_solid_surface_area_theorem_l327_32776

def solid_surface_area (s : ℝ) (h : ℝ) : ℝ :=
  let base_area := s^2
  let upper_area := 3 * s^2
  let trapezoid_area := 2 * (s + 3*s) * h
  base_area + upper_area + trapezoid_area

theorem solid_surface_area_theorem :
  solid_surface_area (4 * Real.sqrt 2) (3 * Real.sqrt 2) = 320 := by
  sorry

end NUMINAMATH_CALUDE_solid_surface_area_theorem_l327_32776


namespace NUMINAMATH_CALUDE_collinear_implies_coplanar_exist_coplanar_non_collinear_l327_32751

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a predicate for three points being collinear
def collinear (p q r : Point3D) : Prop := sorry

-- Define a predicate for four points being coplanar
def coplanar (p q r s : Point3D) : Prop := sorry

-- Theorem: If three out of four points are collinear, then all four points are coplanar
theorem collinear_implies_coplanar (p q r s : Point3D) :
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) →
  coplanar p q r s :=
sorry

-- Theorem: There exist four coplanar points where no three are collinear
theorem exist_coplanar_non_collinear :
  ∃ (p q r s : Point3D), coplanar p q r s ∧
    ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) :=
sorry

end NUMINAMATH_CALUDE_collinear_implies_coplanar_exist_coplanar_non_collinear_l327_32751


namespace NUMINAMATH_CALUDE_weight_of_b_l327_32702

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  b = 31 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l327_32702


namespace NUMINAMATH_CALUDE_rabbit_weight_l327_32795

theorem rabbit_weight (k r p : ℝ) 
  (total_weight : k + r + p = 39)
  (rabbit_parrot_weight : r + p = 3 * k)
  (rabbit_kitten_weight : r + k = 1.5 * p) :
  r = 13.65 := by
sorry

end NUMINAMATH_CALUDE_rabbit_weight_l327_32795


namespace NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l327_32720

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

end NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l327_32720


namespace NUMINAMATH_CALUDE_max_area_right_triangle_pen_l327_32796

/-- The maximum area of a right triangular pen with perimeter 60 feet is 450 square feet. -/
theorem max_area_right_triangle_pen (x y : ℝ) : 
  x > 0 → y > 0 → x + y + Real.sqrt (x^2 + y^2) = 60 → 
  (1/2) * x * y ≤ 450 := by
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_pen_l327_32796


namespace NUMINAMATH_CALUDE_marathon_equation_l327_32773

/-- Represents the marathon race scenario -/
theorem marathon_equation (x : ℝ) (distance : ℝ) (speed_ratio : ℝ) (head_start : ℝ) :
  distance > 0 ∧ x > 0 ∧ speed_ratio > 1 ∧ head_start > 0 →
  (distance = 5) ∧ (speed_ratio = 1.5) ∧ (head_start = 12.5 / 60) →
  distance / x = distance / (speed_ratio * x) + head_start :=
by
  sorry

end NUMINAMATH_CALUDE_marathon_equation_l327_32773


namespace NUMINAMATH_CALUDE_smallest_M_with_non_decimal_k_l327_32755

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

end NUMINAMATH_CALUDE_smallest_M_with_non_decimal_k_l327_32755


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l327_32761

def arithmetic_sequence (a₁ a₂ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * (a₂ - a₁)

theorem thirtieth_term_of_sequence (a₁ a₂ : ℤ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) :
  arithmetic_sequence a₁ a₂ 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l327_32761


namespace NUMINAMATH_CALUDE_functional_equation_solution_l327_32726

theorem functional_equation_solution (f : ℝ × ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x, y) + f (y, z) + f (z, x) = 0) :
  ∃ g : ℝ → ℝ, ∀ x y : ℝ, f (x, y) = g x - g y := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l327_32726


namespace NUMINAMATH_CALUDE_plane_equation_proof_l327_32725

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by its direction ratios -/
structure Line3D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A plane in 3D space defined by its equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def line_in_plane (l : Line3D) (plane : Plane) : Prop :=
  plane.A * l.a + plane.B * l.b + plane.C * l.c = 0

/-- The main theorem -/
theorem plane_equation_proof (p : Point3D) (l : Line3D) :
  p = Point3D.mk 0 7 (-7) →
  l = Line3D.mk (-3) 2 1 →
  let plane := Plane.mk 1 1 1 0
  point_on_plane p plane ∧ line_in_plane l plane := by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l327_32725


namespace NUMINAMATH_CALUDE_min_tea_time_l327_32753

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

end NUMINAMATH_CALUDE_min_tea_time_l327_32753


namespace NUMINAMATH_CALUDE_subset_intersection_problem_l327_32794

theorem subset_intersection_problem (a : ℝ) :
  let A := { x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5 }
  let B := { x : ℝ | 3 ≤ x ∧ x ≤ 22 }
  (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_problem_l327_32794


namespace NUMINAMATH_CALUDE_number_equation_solution_l327_32714

theorem number_equation_solution : ∃ x : ℚ, (3 * x + 15 = 6 * x - 10) ∧ (x = 25 / 3) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l327_32714


namespace NUMINAMATH_CALUDE_wxyz_unique_product_l327_32743

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

end NUMINAMATH_CALUDE_wxyz_unique_product_l327_32743


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l327_32705

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, (2*x - 1)^2 = (x + 1)*(3*x + 4)) →
    (∀ x, a*x^2 + b*x + c = 0) ∧
    a = 1 ∧ b = -11 ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l327_32705


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l327_32710

-- Define the polynomial
def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 20

-- Define the property that all roots are positive integers
def all_roots_positive_integers (p : ℤ → ℤ) : Prop :=
  ∀ r : ℤ, p r = 0 → r > 0

-- State the theorem
theorem polynomial_coefficient_B :
  ∀ A B C D : ℤ,
  all_roots_positive_integers (polynomial · A B C D) →
  B = -160 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l327_32710


namespace NUMINAMATH_CALUDE_probability_not_perfect_power_l327_32742

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

end NUMINAMATH_CALUDE_probability_not_perfect_power_l327_32742


namespace NUMINAMATH_CALUDE_cube_vertex_distances_l327_32706

/-- Given a cube with edge length a, after transformation by x₁₄ and x₄₅, 
    the sum of the squares of the distances between vertices 1 and 2, 1 and 4, and 1 and 5 
    is equal to 2a². -/
theorem cube_vertex_distances (a : ℝ) (x₁₄ x₄₅ : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ) 
  (h : a > 0) : 
  ∃ (v₁ v₂ v₄ v₅ : ℝ × ℝ × ℝ), 
    let d₁₂ := ‖v₁ - v₂‖
    let d₁₄ := ‖v₁ - v₄‖
    let d₁₅ := ‖v₁ - v₅‖
    d₁₂^2 + d₁₄^2 + d₁₅^2 = 2 * a^2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_vertex_distances_l327_32706


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l327_32740

theorem fraction_sum_equality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l327_32740


namespace NUMINAMATH_CALUDE_roma_winning_strategy_l327_32700

/-- The game state representing the positions of chips on a board -/
structure GameState where
  k : ℕ  -- number of cells
  n : ℕ  -- number of chips
  positions : List ℕ  -- positions of chips

/-- The rating of a chip at a given position -/
def chipRating (pos : ℕ) : ℕ := 2^pos

/-- The total rating of all chips in the game state -/
def totalRating (state : GameState) : ℕ :=
  state.positions.map chipRating |>.sum

/-- Roma's strategy to maintain or reduce the total rating -/
def romaStrategy (state : GameState) : GameState :=
  sorry

theorem roma_winning_strategy (k n : ℕ) (h : n < 2^(k-3)) :
  ∀ (state : GameState), state.k = k → state.n = n →
    ∀ (finalState : GameState), finalState = (romaStrategy state) →
      ∀ (pos : ℕ), pos ∈ finalState.positions → pos < k - 1 := by
  sorry

end NUMINAMATH_CALUDE_roma_winning_strategy_l327_32700


namespace NUMINAMATH_CALUDE_six_digit_number_problem_l327_32790

theorem six_digit_number_problem : ∃! n : ℕ, 
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ k : ℕ, n = 200000 + k ∧ k < 100000 ∧
  10 * k + 2 = 3 * n ∧
  n = 285714 := by
sorry

end NUMINAMATH_CALUDE_six_digit_number_problem_l327_32790


namespace NUMINAMATH_CALUDE_largest_common_value_l327_32728

/-- The first arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 5 + 10 * m

/-- A common term of both sequences -/
def common_term (k : ℕ) : ℕ := 4 + 10 * k

theorem largest_common_value :
  (∃ n m : ℕ, seq1 n = seq2 m ∧ seq1 n < 1000) ∧
  (∀ n m : ℕ, seq1 n = seq2 m → seq1 n < 1000 → seq1 n ≤ 994) ∧
  (∃ k : ℕ, common_term k = 994 ∧ common_term k = seq1 (2 * k) ∧ common_term k = seq2 k) :=
sorry

end NUMINAMATH_CALUDE_largest_common_value_l327_32728


namespace NUMINAMATH_CALUDE_player_A_wins_l327_32701

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents a row of squares on the game board -/
structure Row :=
  (length : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (tokens : ℕ)
  (row_R : Row)
  (row_S : Row)

/-- Determines if a player has a winning strategy -/
def has_winning_strategy (player : Player) (state : GameState) : Prop :=
  match player with
  | Player.A => state.tokens > 10
  | Player.B => state.tokens ≤ 10

/-- The main theorem stating that Player A has a winning strategy when tokens > 10 -/
theorem player_A_wins (state : GameState) (h1 : state.row_R.length = 1492) (h2 : state.row_S.length = 1989) :
  has_winning_strategy Player.A state ↔ state.tokens > 10 :=
sorry

end NUMINAMATH_CALUDE_player_A_wins_l327_32701


namespace NUMINAMATH_CALUDE_problem_solution_l327_32791

theorem problem_solution (x y : ℚ) 
  (h1 : x + y = 2/3)
  (h2 : x/y = 2/3) : 
  x - y = -2/15 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l327_32791


namespace NUMINAMATH_CALUDE_line_plane_relationship_l327_32770

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (m : Line) (α β : Plane) 
  (h1 : perpPlanes α β) 
  (h2 : perp m α) : 
  para m β ∨ subset m β := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l327_32770


namespace NUMINAMATH_CALUDE_indeterminate_teachers_per_department_l327_32738

/-- Represents a school with departments and teachers -/
structure School where
  departments : ℕ
  total_teachers : ℕ

/-- Defines a function to check if it's possible to determine exact number of teachers per department -/
def can_determine_teachers_per_department (s : School) : Prop :=
  ∃ (teachers_per_dept : ℕ), s.total_teachers = s.departments * teachers_per_dept

/-- Theorem stating that for a school with 7 departments and 140 teachers, 
    it's not always possible to determine the exact number of teachers in each department -/
theorem indeterminate_teachers_per_department :
  ¬ ∀ (s : School), s.departments = 7 ∧ s.total_teachers = 140 → can_determine_teachers_per_department s :=
by
  sorry


end NUMINAMATH_CALUDE_indeterminate_teachers_per_department_l327_32738


namespace NUMINAMATH_CALUDE_sandwich_change_calculation_l327_32798

theorem sandwich_change_calculation (num_sandwiches : ℕ) (cost_per_sandwich : ℕ) (amount_paid : ℕ) : 
  num_sandwiches = 3 → cost_per_sandwich = 5 → amount_paid = 20 → 
  amount_paid - (num_sandwiches * cost_per_sandwich) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_change_calculation_l327_32798


namespace NUMINAMATH_CALUDE_polynomial_coefficient_e_l327_32704

/-- Polynomial Q(x) = 3x^3 + dx^2 + ex + f -/
def Q (d e f : ℝ) (x : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem polynomial_coefficient_e (d e f : ℝ) :
  (Q d e f 0 = 9) →
  (3 + d + e + f = -(f / 3)) →
  (e = -15 - 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_e_l327_32704


namespace NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l327_32756

theorem fermat_number_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^(2^n) + 1) ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l327_32756


namespace NUMINAMATH_CALUDE_john_travel_solution_l327_32764

/-- Represents the problem of calculating the distance John travels -/
def john_travel_problem (initial_speed : ℝ) (speed_increase : ℝ) (initial_time : ℝ) 
  (late_time : ℝ) (early_time : ℝ) : Prop :=
  ∃ (total_distance : ℝ) (total_time : ℝ),
    initial_speed * initial_time = initial_speed ∧
    total_distance = initial_speed * (total_time + late_time / 60) ∧
    total_distance = initial_speed * initial_time + 
      (initial_speed + speed_increase) * (total_time - initial_time - early_time / 60) ∧
    total_distance = 123.4375

/-- The theorem stating that the solution to John's travel problem exists -/
theorem john_travel_solution : 
  john_travel_problem 25 20 1 1.5 0.25 := by sorry

end NUMINAMATH_CALUDE_john_travel_solution_l327_32764


namespace NUMINAMATH_CALUDE_maximum_discount_rate_proof_l327_32786

/-- Represents the maximum discount rate that can be applied to a product. -/
def max_discount_rate : ℝ := 8.8

/-- The cost price of the product in yuan. -/
def cost_price : ℝ := 4

/-- The original selling price of the product in yuan. -/
def original_selling_price : ℝ := 5

/-- The minimum required profit margin as a percentage. -/
def min_profit_margin : ℝ := 10

theorem maximum_discount_rate_proof :
  let discounted_price := original_selling_price * (1 - max_discount_rate / 100)
  let profit := discounted_price - cost_price
  let profit_margin := (profit / cost_price) * 100
  (profit_margin ≥ min_profit_margin) ∧
  (∀ x : ℝ, x > max_discount_rate →
    let new_discounted_price := original_selling_price * (1 - x / 100)
    let new_profit := new_discounted_price - cost_price
    let new_profit_margin := (new_profit / cost_price) * 100
    new_profit_margin < min_profit_margin) :=
by sorry

#check maximum_discount_rate_proof

end NUMINAMATH_CALUDE_maximum_discount_rate_proof_l327_32786


namespace NUMINAMATH_CALUDE_marie_lost_erasers_l327_32736

/-- The number of erasers Marie lost -/
def erasers_lost (initial final : ℕ) : ℕ := initial - final

/-- Theorem stating that Marie lost 42 erasers -/
theorem marie_lost_erasers : 
  let initial := 95
  let final := 53
  erasers_lost initial final = 42 := by
sorry

end NUMINAMATH_CALUDE_marie_lost_erasers_l327_32736


namespace NUMINAMATH_CALUDE_base3_sum_correct_l327_32729

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

end NUMINAMATH_CALUDE_base3_sum_correct_l327_32729


namespace NUMINAMATH_CALUDE_odd_perfect_number_l327_32737

/-- Sum of positive divisors of n -/
def sigma (n : ℕ) : ℕ := sorry

/-- A number is perfect if σ(n) = 2n -/
def isPerfect (n : ℕ) : Prop := sigma n = 2 * n

theorem odd_perfect_number (n : ℕ) (h : n > 0) (h_sigma : (sigma n : ℚ) / n = 5 / 3) :
  isPerfect (5 * n) ∧ Odd (5 * n) := by sorry

end NUMINAMATH_CALUDE_odd_perfect_number_l327_32737


namespace NUMINAMATH_CALUDE_tan_cos_tan_equality_l327_32788

theorem tan_cos_tan_equality : Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_tan_equality_l327_32788


namespace NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l327_32732

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

-- State the theorem
theorem middle_term_of_arithmetic_sequence :
  ∀ y : ℤ, is_arithmetic_sequence (3^2) y (3^4) → y = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l327_32732


namespace NUMINAMATH_CALUDE_g_of_seven_equals_twentyone_l327_32745

/-- Given that g(3x - 8) = 2x + 11 for all real x, prove that g(7) = 21 -/
theorem g_of_seven_equals_twentyone (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_g_of_seven_equals_twentyone_l327_32745


namespace NUMINAMATH_CALUDE_range_of_a_l327_32721

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 5, log10 (x^2 + a*x) = 1) → 
  a ∈ Set.Icc (-3) 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l327_32721


namespace NUMINAMATH_CALUDE_base_n_representation_l327_32782

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

end NUMINAMATH_CALUDE_base_n_representation_l327_32782


namespace NUMINAMATH_CALUDE_power_five_sum_minus_two_l327_32709

theorem power_five_sum_minus_two (n : ℕ) : n^5 + n^5 + n^5 + n^5 - 2 * n^5 = 2 * n^5 :=
by
  sorry

end NUMINAMATH_CALUDE_power_five_sum_minus_two_l327_32709


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_equals_neg_sqrt_2_l327_32707

theorem sin_cos_pi_12_equals_neg_sqrt_2 :
  Real.sin (π / 12) - Real.sqrt 3 * Real.cos (π / 12) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_equals_neg_sqrt_2_l327_32707


namespace NUMINAMATH_CALUDE_trebled_result_l327_32735

theorem trebled_result (initial_number : ℕ) : 
  initial_number = 17 → 
  3 * (2 * initial_number + 5) = 117 := by
  sorry

end NUMINAMATH_CALUDE_trebled_result_l327_32735


namespace NUMINAMATH_CALUDE_stair_steps_left_l327_32785

theorem stair_steps_left (total : ℕ) (climbed : ℕ) (h1 : total = 96) (h2 : climbed = 74) :
  total - climbed = 22 := by
  sorry

end NUMINAMATH_CALUDE_stair_steps_left_l327_32785


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l327_32762

theorem repeating_decimal_to_fraction :
  ∃ (y : ℚ), y = 0.37 + (46 / 99) / 100 ∧ y = 3709 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l327_32762


namespace NUMINAMATH_CALUDE_intersection_theorem_l327_32789

def setA : Set ℝ := {x | (x + 3) * (x - 1) ≤ 0}

def setB : Set ℝ := {x | ∃ y, y = Real.log (x^2 - x - 2)}

theorem intersection_theorem : 
  setA ∩ (setB.compl) = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l327_32789


namespace NUMINAMATH_CALUDE_nikolai_faster_l327_32767

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

end NUMINAMATH_CALUDE_nikolai_faster_l327_32767


namespace NUMINAMATH_CALUDE_f_of_9_eq_836_l327_32708

/-- The function f(n) = n^3 + n^2 + n + 17 -/
def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

/-- Theorem: The value of f(9) is 836 -/
theorem f_of_9_eq_836 : f 9 = 836 := by sorry

end NUMINAMATH_CALUDE_f_of_9_eq_836_l327_32708


namespace NUMINAMATH_CALUDE_withdrawal_recorded_as_negative_l327_32787

-- Define the banking system
structure BankAccount where
  balance : ℤ

-- Define deposit and withdrawal operations
def deposit (account : BankAccount) (amount : ℕ) : BankAccount :=
  { balance := account.balance + amount }

def withdraw (account : BankAccount) (amount : ℕ) : BankAccount :=
  { balance := account.balance - amount }

-- Theorem statement
theorem withdrawal_recorded_as_negative (initial_balance : ℕ) (withdrawal_amount : ℕ) :
  (withdraw (BankAccount.mk initial_balance) withdrawal_amount).balance =
  initial_balance - withdrawal_amount :=
by sorry

end NUMINAMATH_CALUDE_withdrawal_recorded_as_negative_l327_32787


namespace NUMINAMATH_CALUDE_any_amount_possible_large_amount_without_change_l327_32754

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

end NUMINAMATH_CALUDE_any_amount_possible_large_amount_without_change_l327_32754


namespace NUMINAMATH_CALUDE_false_premise_implications_l327_32792

theorem false_premise_implications :
  ∃ (p : Prop) (q r : Prop), 
    (¬p) ∧ (p → q) ∧ (p → r) ∧ q ∧ (¬r) := by
  -- Let p be the false premise 5 = -5
  let p := (5 = -5)
  -- Let q be the true conclusion 25 = 25
  let q := (25 = 25)
  -- Let r be the false conclusion 125 = -125
  let r := (125 = -125)
  
  have h1 : ¬p := by sorry
  have h2 : p → q := by sorry
  have h3 : p → r := by sorry
  have h4 : q := by sorry
  have h5 : ¬r := by sorry

  exact ⟨p, q, r, h1, h2, h3, h4, h5⟩

#check false_premise_implications

end NUMINAMATH_CALUDE_false_premise_implications_l327_32792


namespace NUMINAMATH_CALUDE_boat_length_l327_32719

/-- The length of a boat given specific conditions -/
theorem boat_length (breadth : Real) (sinking_depth : Real) (man_mass : Real) (water_density : Real) :
  breadth = 3 ∧ 
  sinking_depth = 0.01 ∧ 
  man_mass = 210 ∧ 
  water_density = 1000 →
  ∃ (length : Real), length = 7 ∧ 
    man_mass = water_density * (length * breadth * sinking_depth) :=
by sorry

end NUMINAMATH_CALUDE_boat_length_l327_32719


namespace NUMINAMATH_CALUDE_y_value_proof_l327_32778

theorem y_value_proof (x y : ℕ+) 
  (h1 : y = (x : ℚ) * (1/4 : ℚ) * (1/2 : ℚ))
  (h2 : (y : ℚ) * (x : ℚ) / 100 = 100) :
  y = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l327_32778


namespace NUMINAMATH_CALUDE_impossible_inequalities_l327_32772

theorem impossible_inequalities (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) :
  ¬(((a₁ * b₂) / (a₁ + b₂) < (a₂ * b₁) / (a₂ + b₁)) ∧
    ((a₂ * b₃) / (a₂ + b₃) > (a₃ * b₂) / (a₃ + b₂)) ∧
    ((a₃ * b₁) / (a₃ + b₁) > (a₁ * b₃) / (a₁ + b₃))) :=
by sorry

end NUMINAMATH_CALUDE_impossible_inequalities_l327_32772


namespace NUMINAMATH_CALUDE_particle_movement_probability_l327_32750

/-- The probability of a particle reaching (n,n) from (0,0) in exactly 2n+k tosses -/
def particle_probability (n k : ℕ) : ℚ :=
  (Nat.choose (2*n + k - 1) (n - 1) : ℚ) * (1 / 2 ^ (2*n + k - 1))

/-- Theorem stating the probability of the particle reaching (n,n) in 2n+k tosses -/
theorem particle_movement_probability (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  particle_probability n k = (Nat.choose (2*n + k - 1) (n - 1) : ℚ) * (1 / 2 ^ (2*n + k - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_particle_movement_probability_l327_32750


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l327_32731

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.64 * A
  let r' := Real.sqrt (A' / π)
  (r' - r) / r = -0.2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l327_32731


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l327_32766

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 6

def choose_starters (n k : ℕ) : ℕ := Nat.choose n k

theorem volleyball_team_combinations : 
  choose_starters (total_players - quadruplets) starters + 
  quadruplets * choose_starters (total_players - quadruplets) (starters - 1) + 
  Nat.choose quadruplets 2 * choose_starters (total_players - quadruplets) (starters - 2) = 7062 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l327_32766


namespace NUMINAMATH_CALUDE_book_pages_l327_32757

theorem book_pages (x : ℕ) (h1 : x > 0) (h2 : x + (x + 1) = 137) : x + 1 = 69 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l327_32757


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l327_32734

/-- The speed of a boat upstream given its speed in still water and the speed of the current. -/
def speed_upstream (speed_still : ℝ) (speed_current : ℝ) : ℝ :=
  speed_still - speed_current

/-- Theorem: Given a boat with speed 50 km/h in still water and a current with speed 20 km/h,
    the speed of the boat upstream is 30 km/h. -/
theorem boat_upstream_speed :
  let speed_still : ℝ := 50
  let speed_current : ℝ := 20
  speed_upstream speed_still speed_current = 30 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l327_32734


namespace NUMINAMATH_CALUDE_arrangement_count_l327_32724

/-- The number of ways to arrange young and elderly people in a line with specific conditions -/
def arrangements (n r : ℕ) : ℕ :=
  (n.factorial * (n - r).factorial) / (n - 2*r).factorial

/-- Theorem stating the number of arrangements for young and elderly people -/
theorem arrangement_count (n r : ℕ) (h : n > 2*r) :
  arrangements n r = (n.factorial * (n - r).factorial) / (n - 2*r).factorial :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l327_32724


namespace NUMINAMATH_CALUDE_math_class_size_l327_32730

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

end NUMINAMATH_CALUDE_math_class_size_l327_32730


namespace NUMINAMATH_CALUDE_correct_subtraction_l327_32733

theorem correct_subtraction (x : ℤ) (h : x - 63 = 8) : x - 36 = 35 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l327_32733


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l327_32746

/-- The ratio of area to perimeter for an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l327_32746


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l327_32784

/-- The line y = 2x - e is tangent to the curve y = x ln x -/
theorem tangent_line_to_x_ln_x : ∃ (x₀ : ℝ), 
  (x₀ * Real.log x₀ = 2 * x₀ - Real.exp 1) ∧ 
  (Real.log x₀ + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l327_32784


namespace NUMINAMATH_CALUDE_temperature_increase_proof_l327_32775

/-- Represents the temperature increase per century -/
def temperature_increase_per_century : ℝ := 4

/-- Represents the total number of years -/
def total_years : ℕ := 1600

/-- Represents the total temperature change over the given years -/
def total_temperature_change : ℝ := 64

/-- Represents the number of years in a century -/
def years_per_century : ℕ := 100

theorem temperature_increase_proof :
  temperature_increase_per_century * (total_years / years_per_century) = total_temperature_change := by
  sorry

end NUMINAMATH_CALUDE_temperature_increase_proof_l327_32775


namespace NUMINAMATH_CALUDE_equation_holds_except_two_values_l327_32774

theorem equation_holds_except_two_values (a : ℝ) (ha : a ≠ 0) :
  ∀ y : ℝ, y ≠ a → y ≠ -a →
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_except_two_values_l327_32774


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l327_32715

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (1 + x)^10 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + 
               a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7 + a₈*(1-x)^8 + 
               a₉*(1-x)^9 + a₁₀*(1-x)^10 ∧ 
  a₈ = 180 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l327_32715


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l327_32748

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x^2 + 2*x = 0 ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 0 ∧ x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l327_32748


namespace NUMINAMATH_CALUDE_average_weight_increase_l327_32797

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 65 →
  new_weight = 85 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l327_32797


namespace NUMINAMATH_CALUDE_picture_processing_time_l327_32777

theorem picture_processing_time (num_pictures : ℕ) (processing_time_per_picture : ℕ) : 
  num_pictures = 960 → 
  processing_time_per_picture = 2 → 
  (num_pictures * processing_time_per_picture) / 60 = 32 := by
sorry

end NUMINAMATH_CALUDE_picture_processing_time_l327_32777


namespace NUMINAMATH_CALUDE_f_period_and_range_l327_32713

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 * (Real.sin x)^2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_period_and_range :
  (∃ T > 0, is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T') ∧
  (∀ y ∈ Set.range (f ∘ (fun x => x * π / 3)), -Real.sqrt 3 ≤ y ∧ y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_f_period_and_range_l327_32713


namespace NUMINAMATH_CALUDE_doctor_visit_cost_is_400_l327_32783

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

end NUMINAMATH_CALUDE_doctor_visit_cost_is_400_l327_32783


namespace NUMINAMATH_CALUDE_fifth_power_equality_l327_32779

theorem fifth_power_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_equality_l327_32779


namespace NUMINAMATH_CALUDE_constant_term_expansion_l327_32716

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℝ, k = 24 ∧ k = (3/2) * a^2) → (a = 4 ∨ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l327_32716


namespace NUMINAMATH_CALUDE_factor_expression_l327_32722

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l327_32722


namespace NUMINAMATH_CALUDE_fiona_casey_hoodies_l327_32780

/-- The number of hoodies Fiona and Casey own together -/
def total_hoodies (fiona_hoodies : ℕ) (casey_extra_hoodies : ℕ) : ℕ :=
  fiona_hoodies + (fiona_hoodies + casey_extra_hoodies)

/-- Theorem stating that Fiona and Casey own 8 hoodies in total -/
theorem fiona_casey_hoodies : total_hoodies 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fiona_casey_hoodies_l327_32780


namespace NUMINAMATH_CALUDE_blue_marbles_total_l327_32765

/-- The number of blue marbles Jason has -/
def jason_blue : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue : ℕ := 24

/-- The total number of blue marbles Jason and Tom have together -/
def total_blue : ℕ := jason_blue + tom_blue

theorem blue_marbles_total : total_blue = 68 := by sorry

end NUMINAMATH_CALUDE_blue_marbles_total_l327_32765


namespace NUMINAMATH_CALUDE_christina_transfer_l327_32749

/-- The amount Christina transferred out of her bank account -/
def amount_transferred (initial_balance final_balance : ℕ) : ℕ :=
  initial_balance - final_balance

/-- Theorem stating that Christina transferred $69 out of her bank account -/
theorem christina_transfer :
  amount_transferred 27004 26935 = 69 := by
  sorry

end NUMINAMATH_CALUDE_christina_transfer_l327_32749


namespace NUMINAMATH_CALUDE_at_least_one_equals_a_l327_32703

theorem at_least_one_equals_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end NUMINAMATH_CALUDE_at_least_one_equals_a_l327_32703


namespace NUMINAMATH_CALUDE_quadratic_range_l327_32747

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem quadratic_range :
  ∀ x : ℝ, (2 ≤ quadratic_function x ∧ quadratic_function x < 5) ↔
  (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_range_l327_32747


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l327_32793

theorem coloring_book_shelves (initial_stock : ℕ) (acquired : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 2000 →
  acquired = 5000 →
  books_per_shelf = 2 →
  (initial_stock + acquired) / books_per_shelf = 3500 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l327_32793


namespace NUMINAMATH_CALUDE_print_shop_charge_l327_32781

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

end NUMINAMATH_CALUDE_print_shop_charge_l327_32781


namespace NUMINAMATH_CALUDE_triangle_max_area_l327_32768

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

end NUMINAMATH_CALUDE_triangle_max_area_l327_32768


namespace NUMINAMATH_CALUDE_ratio_problem_l327_32799

theorem ratio_problem (a b : ℝ) (h : (9*a - 4*b) / (12*a - 3*b) = 4/7) : 
  a / b = 16 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l327_32799


namespace NUMINAMATH_CALUDE_remaining_miles_l327_32752

def total_journey : ℕ := 1200
def miles_driven : ℕ := 215

theorem remaining_miles :
  total_journey - miles_driven = 985 := by sorry

end NUMINAMATH_CALUDE_remaining_miles_l327_32752


namespace NUMINAMATH_CALUDE_diamond_5_20_l327_32760

-- Define the diamond operation
noncomputable def diamond (x y : ℝ) : ℝ := sorry

-- Axioms for the diamond operation
axiom diamond_positive (x y : ℝ) : x > 0 → y > 0 → diamond x y > 0
axiom diamond_eq1 (x y : ℝ) : x > 0 → y > 0 → diamond (x * y) y = x * diamond y y
axiom diamond_eq2 (x : ℝ) : x > 0 → diamond (diamond x 2) x = diamond x 2
axiom diamond_2_2 : diamond 2 2 = 4

-- Theorem to prove
theorem diamond_5_20 : diamond 5 20 = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_5_20_l327_32760


namespace NUMINAMATH_CALUDE_divisibility_by_35_l327_32717

theorem divisibility_by_35 : 
  {a : ℕ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = 
  {1, 11, 16, 36, 46, 51, 71, 81, 86} := by sorry

end NUMINAMATH_CALUDE_divisibility_by_35_l327_32717


namespace NUMINAMATH_CALUDE_second_day_student_tickets_second_day_student_tickets_is_ten_l327_32712

/-- The price of a student ticket -/
def student_ticket_price : ℕ := 9

/-- The total revenue from the first day of sales -/
def first_day_revenue : ℕ := 79

/-- The total revenue from the second day of sales -/
def second_day_revenue : ℕ := 246

/-- The number of senior citizen tickets sold on the first day -/
def first_day_senior_tickets : ℕ := 4

/-- The number of student tickets sold on the first day -/
def first_day_student_tickets : ℕ := 3

/-- The number of senior citizen tickets sold on the second day -/
def second_day_senior_tickets : ℕ := 12

/-- Calculates the price of a senior citizen ticket based on the first day's sales -/
def senior_ticket_price : ℕ := 
  (first_day_revenue - student_ticket_price * first_day_student_tickets) / first_day_senior_tickets

theorem second_day_student_tickets : ℕ := 
  (second_day_revenue - senior_ticket_price * second_day_senior_tickets) / student_ticket_price

theorem second_day_student_tickets_is_ten : second_day_student_tickets = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_day_student_tickets_second_day_student_tickets_is_ten_l327_32712


namespace NUMINAMATH_CALUDE_expand_and_simplify_l327_32727

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l327_32727


namespace NUMINAMATH_CALUDE_central_square_area_l327_32758

theorem central_square_area (side_length : ℝ) (cut_distance : ℝ) :
  side_length = 15 →
  cut_distance = 4 →
  let central_square_side := cut_distance * Real.sqrt 2
  central_square_side ^ 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_central_square_area_l327_32758


namespace NUMINAMATH_CALUDE_proposition_implication_l327_32718

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l327_32718


namespace NUMINAMATH_CALUDE_unique_solution_implies_k_zero_l327_32723

theorem unique_solution_implies_k_zero (a b k : ℤ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 = a ∧ p.2 = b) ∧ 
    Real.sqrt (↑a - 1) + Real.sqrt (↑b - 1) = Real.sqrt (↑(a * b + k))) → 
  k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_k_zero_l327_32723


namespace NUMINAMATH_CALUDE_amelia_position_100_l327_32711

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Defines Amelia's movement pattern -/
def ameliaMove (n : Nat) : Position :=
  sorry

/-- Theorem stating Amelia's position at p₁₀₀ -/
theorem amelia_position_100 : ameliaMove 100 = Position.mk 0 19 := by
  sorry

end NUMINAMATH_CALUDE_amelia_position_100_l327_32711


namespace NUMINAMATH_CALUDE_motion_solution_correct_l327_32771

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


end NUMINAMATH_CALUDE_motion_solution_correct_l327_32771


namespace NUMINAMATH_CALUDE_frankies_pets_l327_32741

theorem frankies_pets (cats snakes parrots dogs : ℕ) : 
  snakes = cats + 6 →
  parrots = cats - 1 →
  cats + dogs = 6 →
  cats + snakes + parrots + dogs = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_frankies_pets_l327_32741


namespace NUMINAMATH_CALUDE_triangle_area_product_l327_32759

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (1/2) * (4/a) * (6/b) = 3 → 
  a * b = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l327_32759


namespace NUMINAMATH_CALUDE_frank_brownies_columns_l327_32744

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

end NUMINAMATH_CALUDE_frank_brownies_columns_l327_32744
