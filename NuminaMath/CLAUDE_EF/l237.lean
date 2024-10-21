import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l237_23717

/-- Profit function given promotion expense m -/
noncomputable def profit (m : ℝ) : ℝ := 29 - 16 / (m + 1) - m

/-- The promotion expense that maximizes profit -/
def optimal_expense : ℝ := 3

/-- The maximum profit achievable -/
def max_profit : ℝ := 21

theorem profit_maximization :
  (∀ m : ℝ, m ≥ 0 → profit m ≤ max_profit) ∧
  profit optimal_expense = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l237_23717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l237_23731

/-- The length of a bridge in kilometers. -/
noncomputable def bridge_length (walking_speed : ℝ) (crossing_time_minutes : ℝ) : ℝ :=
  walking_speed * (crossing_time_minutes / 60)

/-- Theorem stating that a bridge's length is 1.25 km when a man walking at 5 km/hr crosses it in 15 minutes. -/
theorem bridge_length_calculation :
  bridge_length 5 15 = 1.25 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that 5 * (15 / 60) = 1.25
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l237_23731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitudes_sum_ge_nine_times_inradius_l237_23726

/-- Given a triangle with altitudes h₁, h₂, h₃ and an inscribed circle of radius r,
    the sum of the altitudes is greater than or equal to 9 times the radius. -/
theorem triangle_altitudes_sum_ge_nine_times_inradius 
  (h₁ h₂ h₃ r : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ r > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = (2 * a * b * c) / (a * (b + c - a)) ∧
    h₂ = (2 * a * b * c) / (b * (a + c - b)) ∧
    h₃ = (2 * a * b * c) / (c * (a + b - c)))
  (h_inradius : r = (a * b * c) / ((a + b + c) * (a + b - c))) : 
  h₁ + h₂ + h₃ ≥ 9 * r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_altitudes_sum_ge_nine_times_inradius_l237_23726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inner_triangle_ratio_l237_23747

/-- Represents a triangle with an interior point and parallel lines --/
structure TriangleWithInteriorPoint where
  -- Original triangle
  S : ℝ
  -- Areas of the three inner triangles
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  -- Conditions
  area_positive : S > 0
  inner_triangles_ratio : S₁ = (1 : ℝ) ∧ S₂ = (4 : ℝ) ∧ S₃ = (9 : ℝ)
  sum_of_inner_areas : S₁ + S₂ + S₃ = S

/-- The ratio of the largest inner triangle to the original triangle is 9/14 --/
theorem largest_inner_triangle_ratio (T : TriangleWithInteriorPoint) :
  T.S₃ / T.S = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inner_triangle_ratio_l237_23747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_and_h_above_l237_23734

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

def h (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 2 * a * x - 1

-- State the theorem
theorem f_decreasing_and_h_above (a : ℤ) : 
  (∀ x > 1, ∀ y > x, f y < f x) ∧ 
  (a = 1 ∧ ∀ x > 0, f x < h a x) ∧
  (∀ b : ℤ, b < 1 → ∃ x > 0, f x ≥ h b x) :=
by
  sorry

#check f_decreasing_and_h_above

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_and_h_above_l237_23734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_always_wins_player_a_wins_9_player_a_wins_10_l237_23728

/-- Represents the game state with n "-" signs -/
structure GameState where
  n : ℕ
  h : n > 0

/-- Represents a player's move: changing one or two adjacent "-" signs to "+" -/
inductive Move
  | one  : Move
  | two  : Move

/-- Defines a winning strategy for a player -/
def WinningStrategy (player : ℕ) (depth : ℕ) : Prop :=
  ∀ (state : GameState), depth > 0 → ∃ (move : Move),
    (player = 1 → state.n % 2 = 1 → move = Move.one) ∧
    (player = 1 → state.n % 2 = 0 → move = Move.two) ∧
    (∀ (opponent_move : Move), WinningStrategy ((player % 2) + 1) (depth - 1))

/-- The main theorem: Player A (represented by 1) always has a winning strategy -/
theorem player_a_always_wins (n : ℕ) (h : n > 0) :
  WinningStrategy 1 n := by
  sorry

/-- Specific case: Player A wins when there are 9 "-" signs -/
theorem player_a_wins_9 :
  WinningStrategy 1 9 := by
  sorry

/-- Specific case: Player A wins when there are 10 "-" signs -/
theorem player_a_wins_10 :
  WinningStrategy 1 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_always_wins_player_a_wins_9_player_a_wins_10_l237_23728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_m_half_right_angled_implies_m_sqrt3_l237_23789

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ B.1 - A.1 = t * (C.1 - A.1) ∧ B.2 - A.2 = t * (C.2 - A.2)

-- Define right-angled triangle
def rightAngled (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem 1
theorem collinear_implies_m_half :
  collinear OA OB (OC (1/2)) := by
  sorry

-- Theorem 2
theorem right_angled_implies_m_sqrt3 :
  rightAngled OA OB (OC (1 + Real.sqrt 3)) ∨ rightAngled OA OB (OC (1 - Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_m_half_right_angled_implies_m_sqrt3_l237_23789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stoppage_time_l237_23711

/-- Calculates the stoppage time for a bus given its speeds with and without stops -/
noncomputable def calculate_stoppage_time (speed_without_stops speed_with_stops : ℝ) : ℝ :=
  let distance_difference := speed_without_stops - speed_with_stops
  distance_difference / speed_without_stops * 60

/-- Proves that a bus with given speeds stops for 10 minutes per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54)
  (h2 : speed_with_stops = 45) :
  calculate_stoppage_time speed_without_stops speed_with_stops = 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_stoppage_time 54 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stoppage_time_l237_23711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l237_23774

/-- The distance from a point (x, y) to the line -5y + c = 0 --/
noncomputable def distance_to_line (x y c : ℝ) : ℝ :=
  |c - 5 * y| / 5

/-- The condition for a point (x, y) to be on the circle x^2 + y^2 = 4 --/
def on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

/-- The condition for there to be exactly one point on the circle at distance 1 from the line --/
def exactly_one_point (c : ℝ) : Prop :=
  ∃! (x y : ℝ), on_circle x y ∧ distance_to_line x y c = 1

theorem circle_line_intersection (c : ℝ) :
  exactly_one_point c ↔ c ∈ Set.Ioo (-5 : ℝ) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l237_23774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportion_change_l237_23798

/-- Given positive real numbers x, y, and z, where x and y are inversely proportional
    and z is directly proportional to x and inversely proportional to y,
    prove that when x increases by 20%, both y and z decrease by approximately 16.67% -/
theorem proportion_change (x y z k j m : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hk : k > 0) (hj : j > 0) (hm : m > 0)
  (hxy : x * y = k) (hxz : x * z = j) (hyz : y * z = m) :
  let x' := 1.2 * x
  let y' := y / 1.2
  let z' := z / 1.2
  (abs ((1 - y' / y) * 100 - 16.67) < 0.01) ∧ (abs ((1 - z' / z) * 100 - 16.67) < 0.01) := by
  sorry

#check proportion_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportion_change_l237_23798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_l237_23761

def is_valid_permutation (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    n = 100 * a + 10 * b + c ∧
    (100 * a + 10 * b + c) % 4 = (100 * a + 10 * c + b) % 4 ∧
    (100 * a + 10 * b + c) % 4 = (100 * b + 10 * a + c) % 4 ∧
    (100 * a + 10 * b + c) % 4 = (100 * b + 10 * c + a) % 4 ∧
    (100 * a + 10 * b + c) % 4 = (100 * c + 10 * a + b) % 4 ∧
    (100 * a + 10 * b + c) % 4 = (100 * c + 10 * b + a) % 4

theorem valid_permutations :
  ∀ n : ℕ, is_valid_permutation n ↔ n ∈ ({159, 195, 519, 591, 915, 951} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_l237_23761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_theorem_l237_23756

/-- Represents a square sheet of paper -/
structure Square where
  side_length : ℝ
  rotation : ℝ

/-- Calculates the area of the polygon formed by overlapping rotated squares -/
noncomputable def polygon_area (squares : List Square) : ℝ :=
  sorry

/-- The main theorem stating the area of the polygon -/
theorem polygon_area_theorem (squares : List Square) 
  (h1 : squares.length = 4)
  (h2 : ∀ s ∈ squares, s.side_length = 8)
  (h3 : squares.map Square.rotation = [0, 20, 45, 70]) :
  ∃ ε > 0, |polygon_area squares - 198| < ε :=
by
  sorry

#check polygon_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_theorem_l237_23756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l237_23706

def P : Set ℤ := {x | -4 ≤ x ∧ x ≤ 2}
def Q : Set ℤ := {x | -3 < x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l237_23706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_two_intersecting_lines_l237_23704

/-- Definition of a line in ℝ² -/
def IsLine (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ ∀ (x y : ℝ), (x, y) ∈ s ↔ a * x + b * y + c = 0

/-- The equation (x-1)(x+y+2) = (y-1)(x+y+2) represents two intersecting lines in the xy-plane -/
theorem equation_represents_two_intersecting_lines :
  ∃ (l₁ l₂ : Set (ℝ × ℝ)) (p : ℝ × ℝ),
    (∀ (x y : ℝ), (x - 1) * (x + y + 2) = (y - 1) * (x + y + 2) ↔ (x, y) ∈ l₁ ∪ l₂) ∧
    IsLine l₁ ∧ IsLine l₂ ∧ l₁ ≠ l₂ ∧ p ∈ l₁ ∩ l₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_two_intersecting_lines_l237_23704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_share_is_27_l237_23702

/-- The share of y in rupees when a sum is divided among x, y, and z -/
noncomputable def share_of_y (total : ℝ) (x_share : ℝ) (y_share : ℝ) (z_share : ℝ) : ℝ :=
  (total * y_share) / (x_share + y_share + z_share)

theorem y_share_is_27 :
  let total := 105
  let x_share := 1
  let y_share := 0.45
  let z_share := 0.3
  share_of_y total x_share y_share z_share = 27 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_share_is_27_l237_23702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_72_30_l237_23741

/-- Given a rhombus with diagonals of length d1 and d2, 
    calculate its perimeter -/
noncomputable def rhombusPerimeter (d1 d2 : ℝ) : ℝ :=
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2)

/-- Theorem: The perimeter of a rhombus with diagonals 72 cm and 30 cm is 156 cm -/
theorem rhombus_perimeter_72_30 : 
  rhombusPerimeter 72 30 = 156 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval rhombusPerimeter 72 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_72_30_l237_23741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_π_monomial_properties_l237_23772

-- Define a structure for monomials
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  x_power : ℕ
  y_power : ℕ

-- Define a function to calculate the degree of a monomial
def monomial_degree (m : Monomial ℝ) : ℕ :=
  m.x_power + m.y_power

-- Define our specific monomial
noncomputable def π_monomial : Monomial ℝ :=
  { coeff := Real.pi / 4
  , x_power := 1
  , y_power := 2 }

-- Theorem statement
theorem π_monomial_properties :
  π_monomial.coeff = Real.pi / 4 ∧ monomial_degree π_monomial = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_π_monomial_properties_l237_23772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l237_23769

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- Define symmetry about x = 0
def symmetric_about_zero (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Theorem statement
theorem symmetry_condition (φ : ℝ) :
  symmetric_about_zero (fun x ↦ f (x + φ)) ↔ φ = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l237_23769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_correct_l237_23768

/-- Calculates the total amount paid for a purchase with pizzas and salads, including discount and tax --/
def total_amount_paid (num_pizzas : ℕ) (pizza_price : ℚ) (num_salads : ℕ) (salad_price : ℚ) 
                      (pizza_discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let pizza_cost := num_pizzas * pizza_price
  let pizza_discount := pizza_cost * pizza_discount_rate
  let discounted_pizza_cost := pizza_cost - pizza_discount
  let salad_cost := num_salads * salad_price
  let subtotal := discounted_pizza_cost + salad_cost
  let tax := subtotal * tax_rate
  let total := subtotal + tax
  (total * 100).floor / 100  -- Round down to nearest cent

theorem total_amount_correct : 
  total_amount_paid 3 8 2 6 (1/10) (7/100) = 3595/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_correct_l237_23768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_minus_one_l237_23766

theorem not_prime_sum_minus_one (x y : ℤ) 
  (hx : x > 1) 
  (hy : y > 1) 
  (h_divides : (x + y - 1) ∣ (x^2 + y^2 - 1)) : 
  ¬ Nat.Prime (Int.natAbs (x + y - 1)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_minus_one_l237_23766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_nine_l237_23735

theorem three_digit_numbers_divisible_by_nine : 
  (Finset.filter (fun n => n % 9 = 0) (Finset.range 900 ∪ Finset.range 100)).card = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_nine_l237_23735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_like_equation_implies_power_l237_23744

theorem fermat_like_equation_implies_power (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ m : ℕ, n = p^m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_like_equation_implies_power_l237_23744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_three_fifths_l237_23797

noncomputable def S (n : ℕ) (α : ℝ) : ℝ := (5 : ℝ)^n * Real.sin (n * α)

theorem sin_alpha_three_fifths (α : ℝ) (h : Real.sin α = 3/5) :
  (∀ n : ℕ, n > 0 → ∃ k : ℤ, S n α = k) ∧
  (¬ ∃ n : ℕ, n > 0 ∧ ∃ m : ℤ, S n α = 5 * m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_three_fifths_l237_23797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_rate_equation_l237_23721

/-- Represents the daily paving distance of Team A in meters -/
def x : ℝ := sorry

/-- Theorem stating that the equation correctly represents the relationship between
    the paving rates of Team A and Team B given the problem conditions -/
theorem paving_rate_equation :
  (x > 10) →  -- Ensure x - 10 is positive
  (150 / x = 120 / (x - 10)) ↔
  (x > 0 ∧ -- Ensure division by x is valid
   150 / x = 120 / (x - 10) ∧ -- The equation itself
   x - (x - 10) = 10) -- Team A paves 10m more per day than Team B
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_rate_equation_l237_23721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_open_number_and_max_G_divisible_by_7_l237_23714

/-- Definition of an "open number" -/
def is_open_number (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  let A := 10 * a + d
  let B := 10 * c + b
  A - B = (a + b : Int)  -- Use Int instead of ℕ for subtraction

/-- Definition of G(M) -/
noncomputable def G (M : ℕ) : ℚ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  (b + 13 : ℚ) / (c - a - d : ℚ)  -- Cast to ℚ to ensure division

/-- Theorem stating that 1029 is an "open number" and 8892 is the maximum
    four-digit number M that satisfies the "open number" condition and
    has G(M) divisible by 7 -/
theorem open_number_and_max_G_divisible_by_7 :
  is_open_number 1029 ∧
  (∀ M : ℕ, M ≤ 9999 → is_open_number M → (G M * 7).num % (G M * 7).den = 0 → M ≤ 8892) ∧
  is_open_number 8892 ∧
  (G 8892 * 7).num % (G 8892 * 7).den = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_open_number_and_max_G_divisible_by_7_l237_23714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_de_time_l237_23787

/-- Represents the time taken to fill a tank with different valve combinations -/
structure TankFilling where
  all : ℚ  -- Time taken when all valves are open
  df : ℚ   -- Time taken when valves D and F are open
  ef : ℚ   -- Time taken when valves E and F are open

/-- Calculates the time taken to fill the tank with only valves D and E open -/
def time_de (t : TankFilling) : ℚ :=
  30 / 23

/-- Theorem stating that given the specified filling times for different valve combinations,
    the time taken to fill the tank with only valves D and E open is 30/23 hours -/
theorem tank_filling_de_time (t : TankFilling)
  (h_all : t.all = 5/4)
  (h_df : t.df = 2)
  (h_ef : t.ef = 3) :
  time_de t = 30/23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_de_time_l237_23787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_passed_implies_not_solved_all_not_passed_implies_failed_at_least_one_l237_23781

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (solved_all : Student → Prop)
variable (passed : Student → Prop)
variable (solved : Nat → Student → Prop)  -- Added this line

-- Dr. Evans' statement
variable (evans_statement : ∀ s : Student, solved_all s → passed s)

-- Theorem to prove
theorem not_passed_implies_not_solved_all : 
  ∀ s : Student, ¬(passed s) → ¬(solved_all s) := by
  sorry

-- The logically equivalent statement we want to prove
theorem not_passed_implies_failed_at_least_one (s : Student) : 
  ¬(passed s) → ∃ problem, ¬(solved problem s) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_passed_implies_not_solved_all_not_passed_implies_failed_at_least_one_l237_23781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_properties_sum_distances_not_midpoint_l237_23777

-- Define a line segment AB
structure LineSegment where
  A : ℝ
  B : ℝ

-- Define a point P on the line segment AB
structure PointOnSegment (AB : LineSegment) where
  P : ℝ
  on_segment : AB.A ≤ P ∧ P ≤ AB.B

-- Define what it means for a point to be the midpoint of a line segment
def is_midpoint (AB : LineSegment) (P : ℝ) : Prop :=
  |AB.A - P| = |P - AB.B| ∧ AB.A ≤ P ∧ P ≤ AB.B

-- State the theorem
theorem midpoint_properties (AB : LineSegment) (P : PointOnSegment AB) :
  is_midpoint AB P.P →
  (|AB.A - P.P| = |P.P - AB.B|) ∧
  (|P.P - AB.B| = (1/2) * |AB.A - AB.B|) ∧
  (|AB.A - AB.B| = 2 * |AB.A - P.P|) :=
by sorry

-- Show that the sum of distances doesn't necessarily imply midpoint
theorem sum_distances_not_midpoint (AB : LineSegment) (P : PointOnSegment AB) :
  |AB.A - P.P| + |P.P - AB.B| = |AB.A - AB.B| →
  ¬(is_midpoint AB P.P) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_properties_sum_distances_not_midpoint_l237_23777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_30_factorial_l237_23719

-- Define factorial function
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

-- Define a function to count distinct prime factors
def count_distinct_prime_factors (n : ℕ) : ℕ := (Nat.factors n).toFinset.card

-- Theorem statement
theorem distinct_prime_factors_of_30_factorial :
  count_distinct_prime_factors (factorial 30) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_30_factorial_l237_23719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_givenEquationIsFractional_l237_23750

/-- Represents a variable in an equation. -/
def IsVariable (c : Char) : Prop :=
  c = 'x' ∨ c = 'y' ∨ c = 'z'  -- Simplified for this example

/-- Represents a denominator in an equation. -/
def IsDenominator (s : String) : Prop :=
  s.length > 0 ∧ s.front = '('  -- Simplified representation

/-- Defines a fractional equation as an equation with at least one fraction containing a variable in its denominator. -/
def IsFractionalEquation (eq : String) : Prop :=
  (∃ x, x ∈ eq.data ∧ IsVariable x) ∧ 
  ('=' ∈ eq.data) ∧
  (∃ d : String, d.isPrefixOf eq ∧ IsDenominator d ∧ (∃ x, x ∈ d.data ∧ IsVariable x))

/-- The given equation. -/
def givenEquation : String := "1/(x+1)=1/(2x-3)"

/-- Theorem stating that the given equation is a fractional equation. -/
theorem givenEquationIsFractional : IsFractionalEquation givenEquation := by
  sorry

#check givenEquationIsFractional

end NUMINAMATH_CALUDE_ERRORFEEDBACK_givenEquationIsFractional_l237_23750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l237_23720

theorem cos_alpha_value (α : ℝ) (h : Real.sin (Real.pi + α) = 1/2) :
  Real.cos α = Real.sqrt 3 / 2 ∨ Real.cos α = -(Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l237_23720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_value_of_f_l237_23722

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (4*x^3 + 4*x^2 + 12*x + 23) / (4*x^3 + 4*x^2 + 12*x + 9)

-- Theorem statement
theorem max_integer_value_of_f :
  ∀ y : ℤ, (∃ x : ℝ, f x = ↑y) → y ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_value_of_f_l237_23722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_theorem_l237_23758

/-- Represents a scale factor between map distances and real distances -/
structure MapScale where
  map_dist : ℝ
  real_dist : ℝ
  scale : map_dist / real_dist > 0

/-- Given the actual distance between two mountains, a point's distance on the map,
    and its actual distance, calculate the distance between the mountains on the map -/
noncomputable def map_distance_between_mountains (actual_dist : ℝ) (point_map_dist : ℝ) (point_real_dist : ℝ) : ℝ :=
  (actual_dist * point_map_dist) / point_real_dist

theorem map_distance_theorem (actual_dist : ℝ) (point_map_dist : ℝ) (point_real_dist : ℝ)
  (h1 : actual_dist = 136)
  (h2 : point_map_dist = 42)
  (h3 : point_real_dist = 18.307692307692307)
  : map_distance_between_mountains actual_dist point_map_dist point_real_dist = 312 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_theorem_l237_23758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l237_23767

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * (4:ℝ)^x - k * (2:ℝ)^(x+1) - 4*(k+5)

theorem zero_in_interval (k : ℝ) : ∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f k x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l237_23767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_bill_proof_l237_23790

/-- Given three people ordering food, prove that the unknown order amount is $30 -/
theorem dinner_bill_proof (julie_order : ℝ) (letitia_order : ℝ) (anton_order : ℝ) 
  (tip_per_person : ℝ) (tip_percentage : ℝ) :
  julie_order = 10 →
  letitia_order = 20 →
  tip_per_person = 4 →
  tip_percentage = 0.2 →
  (julie_order + letitia_order + anton_order) * tip_percentage = 3 * tip_per_person →
  anton_order = 30 := by
  intro h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry

#check dinner_bill_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_bill_proof_l237_23790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_major_axis_length_l237_23738

/-- The maximum length of the major axis of an ellipse under specific conditions -/
theorem ellipse_max_major_axis_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_perp : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 + y₁^2/b^2 = 1 → 
    x₂^2/a^2 + y₂^2/b^2 = 1 → 
    y₁ = -x₁ + 1 → 
    y₂ = -x₂ + 1 → 
    x₁*x₂ + y₁*y₂ = 0)
  (h_ecc : ∃ (e : ℝ), 1/2 ≤ e ∧ e ≤ Real.sqrt 2/2 ∧ e^2 = (a^2 - b^2)/a^2) :
  2*a ≤ Real.sqrt 6 ∧ ∃ (a₀ b₀ : ℝ), 2*a₀ = Real.sqrt 6 ∧ a₀ > b₀ ∧ b₀ > 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_major_axis_length_l237_23738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_inequality_l237_23785

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x.natAbs + 8 < 29) → x ≥ -6 ∧ 
  ∃ y : ℤ, y = -6 ∧ 3 * y.natAbs + 8 < 29 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_inequality_l237_23785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l237_23739

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x : ℤ | (x : ℝ)^2 - 2*(x : ℝ) > 0}

theorem intersection_M_N : M ∩ N = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l237_23739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_definition_l237_23710

/-- A geometric solid -/
structure GeometricSolid where
  faces : Set Face

/-- A face of a geometric solid -/
structure Face where
  vertices : Set Point

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Predicate to check if two faces are parallel -/
def are_parallel (f1 f2 : Face) : Prop := sorry

/-- Predicate to check if a face is a quadrilateral -/
def is_quadrilateral (f : Face) : Prop := sorry

/-- Predicate to check if two faces share a common parallel edge -/
def share_parallel_edge (f1 f2 : Face) : Prop := sorry

/-- Definition of a prism -/
def is_prism (s : GeometricSolid) : Prop :=
  ∃ (base1 base2 : Face),
    base1 ∈ s.faces ∧ base2 ∈ s.faces ∧
    are_parallel base1 base2 ∧
    (∀ f, f ∈ s.faces → f ≠ base1 ∧ f ≠ base2 → is_quadrilateral f) ∧
    (∀ f1 f2, f1 ∈ s.faces → f2 ∈ s.faces → f1 ≠ f2 ∧ f1 ≠ base1 ∧ f1 ≠ base2 ∧ f2 ≠ base1 ∧ f2 ≠ base2 →
      share_parallel_edge f1 f2)

/-- Theorem stating the definition of a prism -/
theorem prism_definition (s : GeometricSolid) :
  is_prism s ↔
  (∃ (base1 base2 : Face),
    base1 ∈ s.faces ∧ base2 ∈ s.faces ∧
    are_parallel base1 base2 ∧
    (∀ f, f ∈ s.faces → f ≠ base1 ∧ f ≠ base2 → is_quadrilateral f) ∧
    (∀ f1 f2, f1 ∈ s.faces → f2 ∈ s.faces → f1 ≠ f2 ∧ f1 ≠ base1 ∧ f1 ≠ base2 ∧ f2 ≠ base1 ∧ f2 ≠ base2 →
      share_parallel_edge f1 f2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_definition_l237_23710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l237_23778

def spinner_numbers : List ℕ := [2, 4, 7, 8, 11, 13, 14, 17]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def count_primes (numbers : List ℕ) : ℕ := (numbers.filter is_prime).length

theorem spinner_prime_probability :
  (count_primes spinner_numbers : ℚ) / (spinner_numbers.length : ℚ) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l237_23778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_probability_l237_23773

/-- Represents a square sheet of paper -/
structure Square where
  side : ℝ
  area : ℝ := side * side

/-- Represents a point inside the square -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the folding and cutting process -/
noncomputable def fold_and_cut (s : Square) (p : Point) : ℝ :=
  -- This function would implement the folding and cutting process
  -- and return the area of the region where P must lie for the
  -- remaining piece to be a pentagon
  sorry

/-- The theorem to be proved -/
theorem pentagon_probability (s : Square) : 
  (∫ x in (0 : ℝ)..s.side, ∫ y in (0 : ℝ)..s.side, 
    (fold_and_cut s ⟨x, y⟩) / s.area) = 1/4 := by
  sorry

#check pentagon_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_probability_l237_23773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_problem_l237_23760

theorem set_intersection_problem :
  let M : Set ℤ := {x | x ≤ 3}
  let N : Set ℝ := {x | 1 ≤ Real.exp x ∧ Real.exp x ≤ Real.exp 1}
  (M.image (↑) : Set ℝ) ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_problem_l237_23760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_correct_l237_23724

/-- The number of two-digit numbers where the tens digit is greater than the units digit
    and the sum of the digits is greater than 12 -/
def count_special_numbers : ℕ := 9

theorem count_special_numbers_correct :
  count_special_numbers = 9 :=
by
  -- Unfold the definition
  unfold count_special_numbers
  
  -- The proof would go here. For now, we'll use sorry to skip it.
  sorry

#eval count_special_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_correct_l237_23724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_problem_l237_23752

theorem absolute_value_problem : 
  let x : ℤ := -2016
  ‖‖|x| - x‖ - |x|‖ - x = 4032 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_problem_l237_23752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l237_23709

/-- Represents the hiking scenario -/
structure HikingScenario where
  speed : ℚ  -- Speed in km/min
  initial_distance : ℚ  -- Initial distance hiked east in km
  total_time : ℚ  -- Total time available in minutes

/-- Calculates the total distance hiked east before turning around -/
def total_distance_east (scenario : HikingScenario) : ℚ :=
  scenario.initial_distance + (scenario.total_time - scenario.initial_distance / scenario.speed) * scenario.speed / 2

/-- Theorem stating the total distance hiked east in the given scenario -/
theorem annika_hike_distance (scenario : HikingScenario) 
  (h_speed : scenario.speed = 1 / 10)
  (h_initial : scenario.initial_distance = 5 / 2)
  (h_total_time : scenario.total_time = 35) :
  total_distance_east scenario = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annika_hike_distance_l237_23709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l237_23771

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := h.b

theorem hyperbola_eccentricity_sqrt_5 (h : Hyperbola) 
  (h_focus_dist : focus_to_asymptote_distance h = 2 * h.a) : 
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l237_23771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_locus_is_circle_l237_23707

/-- Predicate to check if a set of points in ℝ² forms a circle. -/
def is_circle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ p : ℝ × ℝ, p ∈ s ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

/-- Given four collinear points A, B, C, D in order, this theorem states that the locus of
    tangency points of circles passing through (A, B) and (C, D) respectively is a circle k
    (except for its intersection points with the line ABCD). -/
theorem tangency_locus_is_circle (A B C D : ℝ) (h_order : A < B ∧ B < C ∧ C < D) :
  ∃ (center : ℝ) (radius : ℝ),
    center ∈ Set.Icc B C ∧
    radius = Real.sqrt ((C - B) * (D - B) * (B - A) * (D - A) / (D - A + 2 * (C - B) + (D - C))^2) ∧
    (∀ P : ℝ × ℝ,
      (P.1 - center)^2 + P.2^2 = radius^2 ∧ P.2 ≠ 0 →
      ∃ (k₁ k₂ : Set (ℝ × ℝ)),
        is_circle k₁ ∧ is_circle k₂ ∧
        (A, 0) ∈ k₁ ∧ (B, 0) ∈ k₁ ∧
        (C, 0) ∈ k₂ ∧ (D, 0) ∈ k₂ ∧
        k₁ ∩ k₂ = {P}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_locus_is_circle_l237_23707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resultant_through_centroid_and_magnitude_l237_23796

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the centroid G of triangle ABC
noncomputable def G (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := 
  (1/3 : ℝ) • (A + B + C)

-- Define an arbitrary point M
variable (M : EuclideanSpace ℝ (Fin 2))

-- Define the forces as vectors
noncomputable def MA (A M : EuclideanSpace ℝ (Fin 2)) := A - M
noncomputable def MB (B M : EuclideanSpace ℝ (Fin 2)) := B - M
noncomputable def MC (C M : EuclideanSpace ℝ (Fin 2)) := C - M

-- Define the resultant force
noncomputable def resultant (A B C M : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := 
  MA A M + MB B M + MC C M

-- Theorem statement
theorem resultant_through_centroid_and_magnitude 
  (A B C M : EuclideanSpace ℝ (Fin 2)) :
  (∃ t : ℝ, G A B C = M + t • resultant A B C M) ∧ 
  ∃ k : ℝ, k • (G A B C - M) = resultant A B C M ∧ k = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_resultant_through_centroid_and_magnitude_l237_23796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_cyclist_intersection_l237_23792

/-- Represents a point in the journey -/
inductive Point where
  | A
  | B
  | C

/-- Represents a vehicle in the journey -/
inductive Vehicle where
  | Car
  | Bus
  | Cyclist

/-- The journey setup -/
structure Journey where
  totalDistance : ℝ
  carDepartureTime : ℝ
  busArrivalTime : ℝ
  cyclistArrivalTime : ℝ
  carPassesC : ℝ

/-- Calculate the speed of a vehicle given distance and time -/
noncomputable def calculateSpeed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Helper function to calculate the distance where bus catches cyclist -/
noncomputable def distance_bus_catches_cyclist (j : Journey) : ℝ :=
  6 -- Simplified for now, actual implementation would go here

/-- The main theorem to prove -/
theorem bus_cyclist_intersection (j : Journey)
  (h1 : j.totalDistance = 10)
  (h2 : j.carDepartureTime = 7)
  (h3 : j.busArrivalTime = 9)
  (h4 : j.cyclistArrivalTime = 10)
  (h5 : j.carPassesC = 2/3 * j.totalDistance) :
  ∃ (d : ℝ), d = 6 ∧ d = distance_bus_catches_cyclist j :=
by
  use 6
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_cyclist_intersection_l237_23792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manu_win_probability_l237_23733

/-- Represents the probability of a coin flip resulting in heads -/
noncomputable def coin_prob : ℝ := 1 / 2

/-- Represents the number of players before Manu -/
def players_before_manu : ℕ := 3

/-- The probability that Manu wins the coin flipping game -/
noncomputable def manu_win_prob : ℝ := 1 / 30

/-- Theorem stating that Manu's probability of winning is 1/30 -/
theorem manu_win_probability : 
  manu_win_prob = ∑' n, coin_prob^(players_before_manu * n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_manu_win_probability_l237_23733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_arrangements_count_l237_23780

/-- Represents a school in the club -/
structure School :=
  (members : Finset Nat)
  (member_count : members.card = 5)

/-- Represents the club with 4 schools -/
structure Club :=
  (schools : Finset School)
  (school_count : schools.card = 4)

/-- Represents a meeting arrangement -/
structure MeetingArrangement :=
  (host : School)
  (host_reps : Finset Nat)
  (other_schools : Finset School)
  (other_reps : Finset Nat)

/-- The number of possible meeting arrangements -/
def meeting_arrangements (c : Club) : ℕ :=
  c.schools.card * (Nat.choose 5 3) * (Nat.choose 3 2) * (5 * 5)

theorem meeting_arrangements_count (c : Club) :
  meeting_arrangements c = 3000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_arrangements_count_l237_23780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l237_23740

theorem series_sum : 
  let i : ℂ := Complex.I
  let series_sum := (Finset.range 45).sum (fun n => i^n * Real.sin (Real.pi / 3 + Real.pi / 2 * n))
  series_sum = Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l237_23740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l237_23793

theorem alpha_value (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (-π/2) 0)
  (h3 : Real.cos (α - β) = 3/5)
  (h4 : Real.sin β = -Real.sqrt 2/10) :
  α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l237_23793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_trip_optimization_l237_23770

/-- Represents the total cost of a truck trip -/
noncomputable def total_cost (x : ℝ) : ℝ := 2340 / x + 13 * x / 18

/-- Represents the optimal speed that minimizes the total cost -/
noncomputable def optimal_speed : ℝ := 18 * Real.sqrt 10

/-- Represents the minimum total cost -/
noncomputable def min_cost : ℝ := 26 * Real.sqrt 10

theorem truck_trip_optimization (x : ℝ) (hx : 50 ≤ x ∧ x ≤ 100) :
  (∀ y, 50 ≤ y ∧ y ≤ 100 → total_cost x ≤ total_cost y) ↔ 
  x = optimal_speed ∧ total_cost x = min_cost := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_trip_optimization_l237_23770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l237_23748

/-- Given a parabola and a line passing through its focus, prove the length of the segment formed by the intersection points. -/
theorem parabola_intersection_length (A B : ℝ × ℝ) (y_M : ℝ) : 
  (∀ x y, x^2 = 8*y → (x, y) ∈ Set.range (λ t : ℝ ↦ (t, t^2/8))) → -- parabola equation
  (∃ m b : ℝ, A.1 = m * A.2 + b ∧ B.1 = m * B.2 + b ∧ 0 = m * 2 + b) → -- line through focus (0, 2)
  y_M = (A.2 + B.2) / 2 → -- midpoint y-coordinate
  y_M = 4 →
  ‖A - B‖ = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l237_23748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l237_23718

/-- The range of a given the specified conditions -/
theorem range_of_a (α : ℝ) (a : ℝ) : 
  (∃ (r : ℝ), r * (3*a - 9) = Real.cos α ∧ r * (a + 2) = Real.sin α) →  -- terminal side condition
  Real.sin (2 * α) ≤ 0 →
  Real.sin α > 0 →
  a ∈ Set.Ioc (-2) 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l237_23718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_solution_l237_23762

theorem quadratic_equations_solution : 
  ∀ (k a b c : ℝ),
  (3^2 - 7*3 + k = 0) →
  (a^2 - 7*a + k = 0) →
  (b^2 - 8*b + (k + 1) = 0) →
  (c^2 - 8*c + (k + 1) = 0) →
  (a + b*c = 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_solution_l237_23762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_3820_l237_23729

/-- Represents the production plan for Yamei Garment Factory -/
structure ProductionPlan where
  jia : ℕ -- number of Type Jia fashion sets
  yi : ℕ -- number of Type Yi fashion sets

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  50 * plan.jia + 45 * plan.yi

/-- Checks if a production plan is valid given the fabric constraints -/
def isValidPlan (plan : ProductionPlan) : Prop :=
  plan.jia + plan.yi = 80 ∧
  (1.1 * plan.jia.toFloat + 0.6 * plan.yi.toFloat : Float) ≤ 70 ∧
  (0.4 * plan.jia.toFloat + 0.9 * plan.yi.toFloat : Float) ≤ 52

/-- The set of all valid production plans -/
def validPlans : Set ProductionPlan :=
  {plan | isValidPlan plan}

/-- Theorem stating that the maximum profit is 3820 yuan -/
theorem max_profit_is_3820 :
  ∃ (plan : ProductionPlan), isValidPlan plan ∧
    profit plan = 3820 ∧
    ∀ (other : ProductionPlan), isValidPlan other → profit other ≤ 3820 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_3820_l237_23729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_product_l237_23727

-- Define the curve y = x + 2/x
noncomputable def on_curve (P : ℝ × ℝ) : Prop :=
  P.2 = P.1 + 2 / P.1 ∧ P.1 > 0

-- Define point A as the foot of the perpendicular from P to y = x
noncomputable def point_A (P : ℝ × ℝ) : ℝ × ℝ :=
  let x := P.1 + 1 / P.1
  (x, x)

-- Define point B as the foot of the perpendicular from P to y-axis
noncomputable def point_B (P : ℝ × ℝ) : ℝ × ℝ :=
  (0, P.2)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Main theorem
theorem perpendicular_product (P : ℝ × ℝ) (h : on_curve P) :
  dot_product (point_A P - P) (point_B P - P) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_product_l237_23727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_to_width_ratio_l237_23779

/-- A box with specific dimensions and volume -/
structure Box where
  height : ℝ
  length : ℝ
  width : ℝ
  volume : ℝ

/-- The properties of the box as described in the problem -/
def problem_box : Box where
  height := 12
  length := 36
  width := 9  -- Calculated width based on the given information
  volume := 3888

/-- The theorem stating the ratio of length to width -/
theorem length_to_width_ratio (b : Box) 
  (h1 : b.height = 12)
  (h2 : b.length = 3 * b.height)
  (h3 : ∃ m : ℝ, b.length = m * b.width)
  (h4 : b.volume = 3888)
  (h5 : b.volume = b.length * b.width * b.height) :
  b.length / b.width = 4 := by
  sorry

#check length_to_width_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_to_width_ratio_l237_23779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_FIGH_theorem_l237_23794

/-- Represents a parallelogram EFGH with a point I on side EH -/
structure ParallelogramWithPoint where
  /-- Area of the parallelogram EFGH -/
  area : ℝ
  /-- Length of side EH -/
  side_length : ℝ
  /-- Ratio of EI to IH -/
  division_ratio : ℝ

/-- Calculates the area of quadrilateral FIGH in the given parallelogram -/
noncomputable def area_FIGH (p : ParallelogramWithPoint) : ℝ :=
  p.area * (1 - p.division_ratio / (1 + p.division_ratio))

/-- Theorem statement -/
theorem area_FIGH_theorem (p : ParallelogramWithPoint) 
  (h1 : p.area = 120)
  (h2 : p.side_length = 15)
  (h3 : p.division_ratio = 3) :
  area_FIGH p = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_FIGH_theorem_l237_23794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_min_max_l237_23732

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

noncomputable def point_P : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def α : ℝ := Real.arctan (-1 / Real.sqrt 3)

theorem f_at_alpha : f α = -3 := by sorry

theorem f_min_max : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), -2 ≤ f x ∧ f x ≤ 1 ∧ 
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), f x₁ = -2) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₂ = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_min_max_l237_23732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_comparison_l237_23764

theorem inequality_comparison (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  (Real.sin (x + 1) > Real.sin x) ∧ 
  (Real.cos (x + 1) < Real.cos x) ∧ 
  ((1 + x)^x < x^(1/2)) ∧ 
  (Real.log (1 + x) > Real.log x) := by
  sorry

#check inequality_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_comparison_l237_23764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_triangles_circumcircles_intersection_concyclic_l237_23745

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A pentagon in the plane -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- An exterior triangle of a pentagon -/
structure ExteriorTriangle where
  P : Point
  Q : Point
  R : Point

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- The circumcircle of a triangle -/
noncomputable def circumcircle (t : ExteriorTriangle) : Circle :=
  sorry

/-- The intersection points of the circumcircles -/
noncomputable def intersectionPoints (p : Pentagon) (ts : List ExteriorTriangle) : List Point :=
  sorry

/-- Checks if a list of points are concyclic -/
def areConcyclic (ps : List Point) : Prop :=
  sorry

/-- Checks if a pentagon is convex -/
def isConvex (p : Pentagon) : Prop :=
  sorry

/-- Checks if a list of triangles are exterior triangles of a pentagon -/
def areExteriorTrianglesOf (ts : List ExteriorTriangle) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem exterior_triangles_circumcircles_intersection_concyclic 
  (p : Pentagon) 
  (h1 : isConvex p) 
  (ts : List ExteriorTriangle) 
  (h2 : areExteriorTrianglesOf ts p) : 
  areConcyclic (intersectionPoints p ts) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_triangles_circumcircles_intersection_concyclic_l237_23745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_unit_distance_l237_23763

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Statement of the theorem
theorem two_color_unit_distance (c : Coloring) :
  ∃ (p1 p2 : Point), distance p1 p2 = 1 ∧ c p1 = c p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_unit_distance_l237_23763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_ratio_at_office_to_remaining_l237_23736

/-- Represents the amount of coffee in ounces --/
structure Coffee where
  amount : ℚ
  deriving Repr

/-- The initial amount of coffee Omar buys --/
def initial_amount : Coffee := ⟨12⟩

/-- The fraction of coffee Omar drinks on the way to work --/
def fraction_drunk_on_way : ℚ := 1/4

/-- The amount of coffee Omar drinks when it's cold --/
def amount_drunk_when_cold : Coffee := ⟨1⟩

/-- The amount of coffee left after Omar is done drinking --/
def final_amount : Coffee := ⟨2⟩

/-- Calculates the amount of coffee remaining after drinking on the way to work --/
def remaining_after_way (initial : Coffee) (fraction : ℚ) : Coffee :=
  ⟨initial.amount * (1 - fraction)⟩

/-- Calculates the amount of coffee Omar drinks at the office --/
def amount_drunk_at_office (remaining : Coffee) (cold : Coffee) (final : Coffee) : Coffee :=
  ⟨remaining.amount - cold.amount - final.amount⟩

/-- The theorem to be proven --/
theorem coffee_ratio_at_office_to_remaining :
  let remaining := remaining_after_way initial_amount fraction_drunk_on_way
  let office_amount := amount_drunk_at_office remaining amount_drunk_when_cold final_amount
  office_amount.amount / remaining.amount = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_ratio_at_office_to_remaining_l237_23736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l237_23799

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt (x - 1) + 4 * Real.sqrt (2 - x)

theorem max_value_of_f :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x ≤ 5 ∧ ∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ 2 ∧ f x₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l237_23799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_plane_centroid_sum_l237_23783

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- x-coordinate of the intersection with the x-axis -/
  a : ℝ
  /-- y-coordinate of the intersection with the y-axis -/
  b : ℝ
  /-- z-coordinate of the intersection with the z-axis -/
  c : ℝ
  /-- Ensure the plane is at the specified distance from the origin -/
  distance_eq : distance = 2
  /-- Ensure the intersections are distinct from the origin -/
  distinct_intersections : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The centroid of the triangle formed by the intersections -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.a / 3, plane.b / 3, plane.c / 3)

/-- The theorem to be proved -/
theorem intersecting_plane_centroid_sum (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_plane_centroid_sum_l237_23783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_song_length_calculation_l237_23703

/-- Represents the length of each of the first two songs in a gig -/
def song_length : ℕ → ℕ := λ x => x

/-- The number of gigs performed -/
def num_gigs : ℕ := 7

/-- The total number of songs played across all gigs -/
def total_songs : ℕ := num_gigs * 3

/-- The total playing time in minutes -/
def total_time : ℕ := 280

theorem song_length_calculation : 
  (∀ x : ℕ, song_length x = x → 
    num_gigs * (2 * song_length x + 2 * song_length x) = total_time) → 
  song_length 10 = 10 := by
  intro h
  have h10 : song_length 10 = 10 := rfl
  exact h10

#check song_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_song_length_calculation_l237_23703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_subsequence_exists_l237_23730

noncomputable def geometric_sequence (n : ℕ) : ℚ := 1 / 2^n

def is_subsequence (a b : ℕ → ℚ) : Prop :=
  ∃ f : ℕ → ℕ, Monotone f ∧ StrictMono f ∧ ∀ n, a n = b (f n)

theorem geometric_subsequence_exists :
  ∃ (sub : ℕ → ℚ),
    is_subsequence sub geometric_sequence ∧
    (∀ n, sub (n + 1) = (1/8 : ℚ) * sub n) ∧
    sub 0 = 1/8 ∧
    Summable sub ∧
    (∑' n, sub n) = 1/7 :=
by
  sorry

#check geometric_subsequence_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_subsequence_exists_l237_23730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l237_23791

/-- The arc length of a sector with given radius and central angle -/
noncomputable def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * 2 * Real.pi * radius

/-- Theorem: The arc length of a sector with radius 5 and central angle 120° is 10π/3 -/
theorem sector_arc_length :
  arcLength 5 120 = (10 / 3) * Real.pi := by
  -- Unfold the definition of arcLength
  unfold arcLength
  -- Simplify the expression
  simp [Real.pi]
  -- Prove the equality
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l237_23791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_triple_intersection_l237_23749

/-- The function f(x) = (x+1)/(|x|+1) -/
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (abs x + 1)

/-- The line l(x) = kx + b -/
def l (k b x : ℝ) : ℝ := k * x + b

theorem slope_range_for_triple_intersection (k b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ = l k b x₁ ∧ f x₂ = l k b x₂ ∧ f x₃ = l k b x₃ ∧
    x₁ + x₂ + x₃ = 0) →
  0 < k ∧ k < 2/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_triple_intersection_l237_23749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_one_black_cell_l237_23715

/-- Represents the number of black cells on a chessboard -/
structure BlackCells where
  value : Nat

/-- Represents a repainting operation on the chessboard -/
def Repaint := BlackCells → BlackCells

/-- The initial number of black cells on the chessboard -/
def initial_black_cells : BlackCells := ⟨32⟩

/-- A repaint operation always changes the number of black cells by an even number -/
axiom repaint_changes_by_even (b : BlackCells) (r : Repaint) :
  ∃ k : Nat, (r b).value = b.value + 2 * k - 8 ∨ (r b).value = b.value - 2 * k + 8

/-- The theorem stating that it's impossible to reach exactly one black cell -/
theorem impossible_one_black_cell :
  ¬∃ (repaints : List Repaint), (List.foldl (λ b r => r b) initial_black_cells repaints).value = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_one_black_cell_l237_23715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_product_sets_l237_23700

/-- A set of seven real numbers where each number is the product of two others in the set -/
def ProductSet : Type := { s : Finset ℝ // s.card = 7 ∧ ∀ x ∈ s, ∃ y z, y ∈ s ∧ z ∈ s ∧ y ≠ z ∧ x = y * z }

/-- The theorem stating that there are infinitely many such sets -/
theorem infinitely_many_product_sets : ∀ n : ℕ, ∃ (sets : Finset ProductSet), sets.card = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_product_sets_l237_23700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_popular_book_buyers_l237_23776

/-- Represents a person who bought books -/
structure Person where
  books : Finset ℕ
  three_books : books.card = 3

/-- Represents the bookstore scenario -/
structure Bookstore where
  people : Finset Person
  total_people : people.card = 510
  common_book : ∀ p q : Person, p ∈ people → q ∈ people → p ≠ q → (p.books ∩ q.books).Nonempty

/-- The minimum number of people who bought the most popular book -/
def most_popular_book_buyers (b : Bookstore) : ℕ :=
  Finset.sup b.people (fun p => (b.people.filter (fun q => (p.books ∩ q.books).Nonempty)).card)

/-- The theorem stating the minimum number of people who bought the most popular book -/
theorem min_popular_book_buyers (b : Bookstore) : most_popular_book_buyers b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_popular_book_buyers_l237_23776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_neg_sin_f_at_four_thirds_pi_l237_23746

/-- The function f as defined in the problem -/
noncomputable def f (θ : ℝ) : ℝ := 
  (Real.sin (θ - 5 * Real.pi) * Real.cos (-Real.pi/2 - θ) * Real.cos (8 * Real.pi - θ)) / 
  (Real.sin (θ - 3 * Real.pi/2) * Real.sin (-θ - 4 * Real.pi))

/-- Theorem stating that f(θ) = -sin(θ) for all θ -/
theorem f_eq_neg_sin (θ : ℝ) : f θ = -Real.sin θ := by sorry

/-- Theorem stating that f(4π/3) = √3/2 -/
theorem f_at_four_thirds_pi : f (4 * Real.pi/3) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_neg_sin_f_at_four_thirds_pi_l237_23746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l237_23765

theorem trigonometric_identity (α : ℝ) : 
  (Real.sin α + 1 / Real.sin α)^2 + (Real.cos α + 1 / Real.cos α)^2 + 4 * Real.sin α * Real.cos α = 
  (7 + 2 * Real.sin (2 * α)) + Real.tan α^2 + (1 / Real.tan α)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l237_23765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpaired_students_three_classes_l237_23788

/-- Represents a class with male and female students -/
structure MyClass where
  males : ℕ
  females : ℕ

/-- Calculates the number of students unable to partner with the opposite gender -/
def unpairedStudents (classes : List MyClass) : ℕ :=
  let totalMales := classes.map (·.males) |>.sum
  let totalFemales := classes.map (·.females) |>.sum
  Int.natAbs (totalFemales - totalMales)

theorem unpaired_students_three_classes :
  let class1 := { males := 17, females := 13 : MyClass }
  let class2 := { males := 14, females := 18 : MyClass }
  let class3 := { males := 15, females := 17 : MyClass }
  let classes := [class1, class2, class3]
  unpairedStudents classes = 2 := by
  sorry

#eval unpairedStudents [
  { males := 17, females := 13 : MyClass },
  { males := 14, females := 18 : MyClass },
  { males := 15, females := 17 : MyClass }
]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpaired_students_three_classes_l237_23788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_calculation_l237_23725

/-- The initially calculated average height of boys in a class -/
noncomputable def initial_average (n : ℕ) (wrong_height correct_height actual_average : ℝ) : ℝ :=
  (n * actual_average + (wrong_height - correct_height)) / n

theorem initial_average_calculation (n : ℕ) (wrong_height correct_height actual_average : ℝ) :
  n = 35 ∧ wrong_height = 166 ∧ correct_height = 106 ∧ actual_average = 180 →
  initial_average n wrong_height correct_height actual_average = 181.71 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_calculation_l237_23725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_calculation_l237_23786

/-- The total distance the dog runs when Wang Ming and his sister meet -/
noncomputable def dog_distance (initial_distance : ℝ) (wang_speed sister_speed dog_speed : ℝ) (final_distance : ℝ) : ℝ :=
  let time := (initial_distance - final_distance) / (wang_speed + sister_speed)
  dog_speed * time

theorem dog_distance_calculation :
  let initial_distance : ℝ := 300
  let wang_speed : ℝ := 50
  let sister_speed : ℝ := 50
  let dog_speed : ℝ := 200
  let final_distance : ℝ := 10
  dog_distance initial_distance wang_speed sister_speed dog_speed final_distance = 580 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_calculation_l237_23786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l237_23755

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem g_properties :
  (∀ y : ℝ, y ≠ 2 → ∃ x : ℝ, g x = y ∧ x = (2 * y + 3) / (y - 2)) ∧
  g 0 = -3/2 ∧
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |g x| > 1/ε) ∧
  g (-3) = 3/5 ∧
  (∀ x : ℝ, x ≠ 2 → g (g x) = x) :=
by
  sorry

#check g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l237_23755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_averageRateOfChangeIs4_l237_23775

/-- The function f(x) = x^2 + 2 -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 2

/-- The average rate of change of f on the interval [1, 3] -/
noncomputable def averageRateOfChange : ℝ := (f 3 - f 1) / (3 - 1)

/-- Theorem: The average rate of change of f(x) = x^2 + 2 on the interval [1, 3] is 4 -/
theorem averageRateOfChangeIs4 : averageRateOfChange = 4 := by
  -- Unfold the definitions
  unfold averageRateOfChange f
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_averageRateOfChangeIs4_l237_23775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l237_23712

-- Define the parabola C
def Parabola (C : Set (ℝ × ℝ)) : Prop :=
  ∃ F : ℝ × ℝ, ∀ P : ℝ × ℝ, P ∈ C ↔ 
    (P.2)^2 = 8 * P.1

-- Define the condition |AF| + |BF| = 8
def DistanceSum (F A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) +
  Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 8

-- Define the perpendicular bisector condition
def PerpendicularBisector (A B : ℝ × ℝ) : Prop :=
  (A.1 - 6)^2 + A.2^2 = (B.1 - 6)^2 + B.2^2

-- Helper function for triangle area
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the main theorem
theorem parabola_properties (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) :
  Parabola C →
  (∀ A B : ℝ × ℝ, A ∈ C → B ∈ C → DistanceSum F A B) →
  (∀ A B : ℝ × ℝ, A ∈ C → B ∈ C → PerpendicularBisector A B) →
  (∃ Q : ℝ × ℝ, Q = (6, 0)) →
  (∀ P : ℝ × ℝ, P ∈ C ↔ P.2^2 = 8 * P.1) ∧
  (∃ maxArea : ℝ, maxArea = (64/9) * Real.sqrt 6 ∧
    ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C →
      area_triangle A (6, 0) B ≤ maxArea) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l237_23712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_five_miles_time_l237_23759

-- Define the running speeds for Jordan and Steve
def jordan_speed : ℝ → ℝ := λ x => x

def steve_speed : ℝ → ℝ := λ x => x

-- Define the relationship between Jordan and Steve's speeds
axiom speed_relation : jordan_speed 2 = steve_speed 3 / 2

-- Define Steve's time for 3 miles
def steve_time : ℝ := 24

-- Define the time it takes Jordan to run x miles
noncomputable def jordan_time (x : ℝ) : ℝ := x * (steve_time / 3)

-- Theorem statement
theorem jordan_five_miles_time :
  jordan_time 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_five_miles_time_l237_23759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_to_dips_l237_23782

/-- A type representing the measurement units in the problem -/
inductive MeasurementUnit
| Dap
| Dop
| Dip

/-- A structure representing a quantity with a numeric value and a unit -/
structure Quantity where
  value : ℚ
  unit : MeasurementUnit

/-- A function to check if two quantities are equivalent -/
def equivalent (q1 q2 : Quantity) : Prop :=
  ∃ (k : ℚ), k > 0 ∧ k * q1.value = q2.value

theorem daps_to_dips (h1 : equivalent (Quantity.mk 5 MeasurementUnit.Dap) (Quantity.mk 4 MeasurementUnit.Dop))
                     (h2 : equivalent (Quantity.mk 3 MeasurementUnit.Dop) (Quantity.mk 8 MeasurementUnit.Dip)) :
  equivalent (Quantity.mk (22.5 : ℚ) MeasurementUnit.Dap) (Quantity.mk 48 MeasurementUnit.Dip) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_to_dips_l237_23782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_squared_distances_l237_23753

/-- Given 9 points on a line, the point that minimizes the sum of squared distances is their average -/
theorem minimize_sum_squared_distances (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : ℝ) :
  let points := [Q₁, Q₂, Q₃, Q₄, Q₅, Q₆, Q₇, Q₈, Q₉]
  let avg := (Q₁ + Q₂ + Q₃ + Q₄ + Q₅ + Q₆ + Q₇ + Q₈ + Q₉) / 9
  let sum_squared_distances (Q : ℝ) := (points.map (fun p => (Q - p)^2)).sum
  ∀ Q : ℝ, sum_squared_distances avg ≤ sum_squared_distances Q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_squared_distances_l237_23753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l237_23723

theorem problem_statement :
  (∃ x : ℝ, x^2 + 2*x + 5 ≤ 4) ∧
  (∀ x : ℝ, 0 < x ∧ x < π/2 → Real.sin x + 4/(Real.sin x) > 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l237_23723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_inequality_l237_23751

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) (h_even : is_even_function f)
    (h_incr : increasing_on f (Set.Iic 0)) :
    ∀ a : ℝ, f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_inequality_l237_23751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_triangle_area_formula_l237_23713

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℚ
  base2 : ℚ
  area : ℚ

/-- The area of a triangle formed by a diagonal and two sides of the trapezoid -/
def diagonal_triangle_area (t : IsoscelesTrapezoid) : ℚ :=
  t.area / 4

theorem diagonal_triangle_area_formula (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 24) 
  (h2 : t.base2 = 40) 
  (h3 : t.area = 480) : 
  diagonal_triangle_area t = 120 := by
  sorry

#eval diagonal_triangle_area { base1 := 24, base2 := 40, area := 480 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_triangle_area_formula_l237_23713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l237_23708

noncomputable def f (x : ℝ) : ℝ := -(Real.cos x) * Real.log (abs x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ k : ℤ, f (Real.pi / 2 + k * Real.pi) = 0) ∧
  (∀ x ∈ Set.Ioo (0 : ℝ) Real.pi, f x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l237_23708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l237_23737

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the angle between two points and the x-axis
noncomputable def angle (p1 p2 : ℝ × ℝ) : ℝ := 
  sorry

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sorry

theorem triangle_similarity (t1 t2 : Triangle)
  (h1 : angle t1.A t1.B = angle t2.A t2.B)
  (h2 : angle t1.B t1.C = angle t2.B t2.C)
  (h3 : angle t1.C t1.A = angle t2.C t2.A) :
  distance t1.A t1.B / distance t1.A t1.C = distance t2.A t2.B / distance t2.A t2.C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l237_23737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_speed_l237_23701

-- Define the given constants
noncomputable def train_length : ℝ := 550
noncomputable def train_speed_kmh : ℝ := 63
noncomputable def crossing_time : ℝ := 32.997

-- Define the function to convert km/h to m/s
noncomputable def km_per_hour_to_m_per_second (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

-- Define the function to calculate relative speed
noncomputable def relative_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

-- Define the function to calculate man's speed in m/s
noncomputable def man_speed_ms (train_speed_ms : ℝ) (rel_speed : ℝ) : ℝ :=
  train_speed_ms - rel_speed

-- Define the function to convert m/s to km/h
noncomputable def m_per_second_to_km_per_hour (speed : ℝ) : ℝ :=
  speed * 3600 / 1000

-- State the theorem
theorem man_walking_speed :
  let train_speed_ms := km_per_hour_to_m_per_second train_speed_kmh
  let rel_speed := relative_speed train_length crossing_time
  let man_speed_ms := man_speed_ms train_speed_ms rel_speed
  let man_speed_kmh := m_per_second_to_km_per_hour man_speed_ms
  abs (man_speed_kmh - 2.992) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walking_speed_l237_23701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_adjacent_face_diagonals_l237_23705

-- Define a cube
structure Cube where
  edge : ℝ
  edge_positive : edge > 0

-- Define a space diagonal of a cube
noncomputable def space_diagonal (c : Cube) : ℝ := c.edge * Real.sqrt 3

-- Define the angle between two space diagonals
noncomputable def angle_between_space_diagonals (c : Cube) : ℝ := Real.arccos (1 / 3)

-- Theorem statement
theorem angle_between_adjacent_face_diagonals (c : Cube) :
  angle_between_space_diagonals c = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_adjacent_face_diagonals_l237_23705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l237_23795

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a slice of a cone -/
structure ConeSlice where
  cone : RightCircularCone
  lowerHeight : ℝ
  upperHeight : ℝ

/-- Calculate the volume of a cone slice -/
noncomputable def volumeOfConeSlice (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * (slice.cone.baseRadius^2) * (slice.upperHeight - slice.lowerHeight)

/-- Calculate the volume ratio of two cone slices -/
noncomputable def volumeRatio (slice1 slice2 : ConeSlice) : ℝ :=
  volumeOfConeSlice slice1 / volumeOfConeSlice slice2

theorem cone_slice_volume_ratio 
  (cone : RightCircularCone) 
  (slice1 : ConeSlice) 
  (slice2 : ConeSlice) 
  (h1 : slice1.cone = cone) 
  (h2 : slice2.cone = cone)
  (h3 : slice1.lowerHeight = (1/3) * cone.height)
  (h4 : slice1.upperHeight = (2/3) * cone.height)
  (h5 : slice2.lowerHeight = 0)
  (h6 : slice2.upperHeight = (1/3) * cone.height) :
  volumeRatio slice1 slice2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l237_23795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_are_butterfly_functions_l237_23743

-- Definition of a butterfly function
def is_butterfly_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f x + x) * (f x - x) ≤ 0

-- Define the two functions
noncomputable def f₁ : ℝ → ℝ := λ x => Real.sin x
noncomputable def f₂ : ℝ → ℝ := λ x => Real.sqrt (x^2 - 1)

-- Theorem statement
theorem both_are_butterfly_functions :
  is_butterfly_function f₁ ∧ is_butterfly_function f₂ := by
  sorry

#check both_are_butterfly_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_are_butterfly_functions_l237_23743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_is_12pi_l237_23784

open Real

-- Define the given constants
noncomputable def sector_radius : ℝ := 5
noncomputable def central_angle : ℝ := 6 * π / 5

-- Define the properties of the cone
noncomputable def slant_height : ℝ := sector_radius
noncomputable def base_circumference : ℝ := central_angle / (2 * π) * (2 * π * sector_radius)
noncomputable def base_radius : ℝ := base_circumference / (2 * π)
noncomputable def cone_height : ℝ := sqrt (slant_height^2 - base_radius^2)

-- Theorem statement
theorem cone_volume_is_12pi :
  (1/3) * π * base_radius^2 * cone_height = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_is_12pi_l237_23784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l237_23742

def score_distribution : List (ℝ × ℝ) := [
  (60, 0.15),
  (75, 0.20),
  (80, 0.25),
  (85, 0.10),
  (90, 0.05),
  (100, 0.25)
]

def total_percentage : ℝ := (score_distribution.map (·.2)).sum

def mean_score : ℝ := (score_distribution.map (λ p => p.1 * p.2)).sum

def median_score : ℝ := 80

theorem exam_score_difference :
  total_percentage = 1 →
  |mean_score - median_score| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_difference_l237_23742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_street_length_is_600_l237_23716

/-- The length of a street in meters, given crossing time and speed -/
noncomputable def street_length (crossing_time_minutes : ℝ) (speed_km_per_hour : ℝ) : ℝ :=
  crossing_time_minutes * (speed_km_per_hour * 1000 / 60)

/-- Theorem: The length of the street is 600 meters -/
theorem street_length_is_600 :
  street_length 2 18 = 600 := by
  -- Unfold the definition of street_length
  unfold street_length
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_street_length_is_600_l237_23716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_x_l237_23754

theorem solve_for_x (a x : ℝ) (h1 : (9 : ℝ)^a = 3) (h2 : Real.log x = a) : x = Real.exp (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_x_l237_23754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l237_23757

/-- A line passing through the origin that forms an equilateral triangle with x = 1 and y = 1 + √3 x has a slope of -√3 -/
noncomputable def slope_of_line_through_origin : ℝ := -Real.sqrt 3

/-- The equation of the line passing through the origin -/
noncomputable def line_through_origin (x : ℝ) : ℝ := slope_of_line_through_origin * x

/-- The first vertex of the triangle -/
noncomputable def vertex1 : ℝ × ℝ := (1, line_through_origin 1)

/-- The second vertex of the triangle -/
noncomputable def vertex2 : ℝ × ℝ := (1, 1 + Real.sqrt 3)

/-- The third vertex of the triangle (the origin) -/
def vertex3 : ℝ × ℝ := (0, 0)

/-- The theorem stating that the perimeter of the triangle is 3 + 6√3 -/
theorem triangle_perimeter : 
  let side_length := Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2)
  3 * side_length = 3 + 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l237_23757
