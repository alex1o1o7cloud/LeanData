import Mathlib

namespace NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l318_31888

theorem profit_percent_for_cost_selling_ratio (cost_price selling_price : ℝ) 
  (h : cost_price / selling_price = 4 / 5) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l318_31888


namespace NUMINAMATH_CALUDE_max_income_at_11_l318_31852

/-- Represents the number of bicycles available for rent -/
def total_bicycles : ℕ := 50

/-- Represents the daily management cost in yuan -/
def management_cost : ℕ := 115

/-- Calculates the number of bicycles rented based on the price -/
def bicycles_rented (price : ℕ) : ℕ :=
  if price ≤ 6 then total_bicycles
  else max (total_bicycles - 3 * (price - 6)) 0

/-- Calculates the net income based on the rental price -/
def net_income (price : ℕ) : ℤ :=
  (price * bicycles_rented price : ℤ) - management_cost

/-- The domain of valid rental prices -/
def valid_price (price : ℕ) : Prop :=
  3 ≤ price ∧ price ≤ 20 ∧ net_income price > 0

theorem max_income_at_11 :
  ∀ price, valid_price price →
    net_income price ≤ net_income 11 :=
  sorry

end NUMINAMATH_CALUDE_max_income_at_11_l318_31852


namespace NUMINAMATH_CALUDE_problem_solution_l318_31858

theorem problem_solution (x y : ℝ) : 
  (0.40 * x = (1/3) * y + 110) → 
  (y = (2/3) * x) → 
  (x = 618.75 ∧ y = 412.5) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l318_31858


namespace NUMINAMATH_CALUDE_smallest_duck_count_is_975_l318_31877

/-- Represents the number of birds in a flock for each type --/
structure FlockSize where
  ducks : Nat
  cranes : Nat
  herons : Nat

/-- Represents the number of flocks for each type of bird --/
structure FlockCount where
  ducks : Nat
  cranes : Nat
  herons : Nat

/-- The smallest number of ducks that satisfies the given conditions --/
def smallest_duck_count (fs : FlockSize) (fc : FlockCount) : Nat :=
  fs.ducks * fc.ducks

/-- Theorem stating the smallest number of ducks observed --/
theorem smallest_duck_count_is_975 (fs : FlockSize) (fc : FlockCount) :
  fs.ducks = 13 →
  fs.cranes = 17 →
  fs.herons = 11 →
  fs.ducks * fc.ducks + fs.cranes * fc.cranes = 15 * fs.herons * fc.herons →
  5 * fc.cranes = 3 * fc.ducks →
  smallest_duck_count fs fc = 975 := by
  sorry

end NUMINAMATH_CALUDE_smallest_duck_count_is_975_l318_31877


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l318_31836

def is_circle (t : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 6*t*x + 8*t*y + 25 = 0

theorem sufficient_not_necessary :
  (∀ t : ℝ, t > 1 → is_circle t) ∧
  (∃ t : ℝ, is_circle t ∧ ¬(t > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l318_31836


namespace NUMINAMATH_CALUDE_basketball_score_proof_l318_31880

theorem basketball_score_proof (jon_score jack_score tom_score : ℕ) : 
  jon_score = 3 →
  jack_score = jon_score + 5 →
  tom_score = jon_score + jack_score - 4 →
  jon_score + jack_score + tom_score = 18 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l318_31880


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l318_31821

theorem pure_imaginary_fraction (m : ℝ) : 
  (∃ k : ℝ, (Complex.I : ℂ) * k = (m + Complex.I) / (1 - Complex.I)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l318_31821


namespace NUMINAMATH_CALUDE_trigonometric_identity_l318_31831

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l318_31831


namespace NUMINAMATH_CALUDE_rosie_pies_l318_31843

/-- Given that Rosie can make 3 pies out of 12 apples, 
    this theorem proves she can make 9 pies out of 36 apples. -/
theorem rosie_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) : 
  apples_per_three_pies = 12 →
  total_apples = 36 →
  (total_apples / apples_per_three_pies) * 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l318_31843


namespace NUMINAMATH_CALUDE_line_l_equation_l318_31884

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨2, 0⟩
def l₁ : Line := ⟨2, 1, -3⟩
def l₂ : Line := ⟨3, -1, 6⟩

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The equation of line l -/
inductive LineL
  | eq1 : LineL  -- 21x + 13y - 42 = 0
  | eq2 : LineL  -- x + y - 2 = 0
  | eq3 : LineL  -- 3x + 4y - 6 = 0

theorem line_l_equation : ∃ (l : LineL), 
  (∀ (A B : Point), pointOnLine A l₁ ∧ pointOnLine B l₂ ∧ 
    pointOnLine M ⟨21, 13, -42⟩ ∧ pointOnLine A ⟨21, 13, -42⟩ ∧ pointOnLine B ⟨21, 13, -42⟩) ∨
  (∀ (A B : Point), pointOnLine A l₁ ∧ pointOnLine B l₂ ∧ 
    pointOnLine M ⟨1, 1, -2⟩ ∧ pointOnLine A ⟨1, 1, -2⟩ ∧ pointOnLine B ⟨1, 1, -2⟩) ∨
  (∀ (A B : Point), pointOnLine A l₁ ∧ pointOnLine B l₂ ∧ 
    pointOnLine M ⟨3, 4, -6⟩ ∧ pointOnLine A ⟨3, 4, -6⟩ ∧ pointOnLine B ⟨3, 4, -6⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_line_l_equation_l318_31884


namespace NUMINAMATH_CALUDE_cubic_root_formula_and_verification_l318_31863

theorem cubic_root_formula_and_verification :
  let x₀ := Real.rpow (3 + (11/9) * Real.sqrt 6) (1/3) + Real.rpow (3 - (11/9) * Real.sqrt 6) (1/3)
  x₀ = 2 ∧ x₀^3 - x₀ - 6 = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_root_formula_and_verification_l318_31863


namespace NUMINAMATH_CALUDE_mary_work_hours_l318_31809

/-- Mary's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, and Friday (each)
  hours_tt : ℕ   -- Hours worked on Tuesday and Thursday (combined)
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Weekly earnings in dollars

/-- Theorem stating Mary's work hours on Tuesday and Thursday --/
theorem mary_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_mwf = 9)
  (h2 : schedule.hourly_rate = 11)
  (h3 : schedule.weekly_earnings = 407)
  (h4 : schedule.weekly_earnings = 
        schedule.hourly_rate * (3 * schedule.hours_mwf + schedule.hours_tt)) :
  schedule.hours_tt = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_work_hours_l318_31809


namespace NUMINAMATH_CALUDE_window_area_calc_l318_31818

/-- The area of a rectangular window -/
def window_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular window with length 6 feet and width 10 feet is 60 square feet -/
theorem window_area_calc :
  window_area 6 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_window_area_calc_l318_31818


namespace NUMINAMATH_CALUDE_quadratic_real_roots_max_integer_a_for_integer_roots_l318_31800

/-- The quadratic equation x^2 - 2ax + 64 = 0 -/
def quadratic_equation (a x : ℝ) : Prop :=
  x^2 - 2*a*x + 64 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ :=
  4*a^2 - 256

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, quadratic_equation a x) ↔ (a ≥ 8 ∨ a ≤ -8) :=
sorry

theorem max_integer_a_for_integer_roots :
  (∃ a : ℕ+, ∃ x y : ℤ, 
    quadratic_equation a x ∧ 
    quadratic_equation a y ∧ 
    (∀ b : ℕ+, b > a → ¬∃ z w : ℤ, quadratic_equation b z ∧ quadratic_equation b w)) →
  (∃ a : ℕ+, a = 17) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_max_integer_a_for_integer_roots_l318_31800


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_equality_l318_31865

theorem sqrt_abs_sum_equality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a^2 + b^2) + |a - b| = a + b ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_equality_l318_31865


namespace NUMINAMATH_CALUDE_undefined_power_implies_m_equals_two_l318_31871

-- Define a proposition that (m-2)^0 is undefined
def is_undefined (m : ℝ) : Prop := (m - 2)^0 ≠ 1

-- Theorem statement
theorem undefined_power_implies_m_equals_two (m : ℝ) :
  is_undefined m → m = 2 := by sorry

end NUMINAMATH_CALUDE_undefined_power_implies_m_equals_two_l318_31871


namespace NUMINAMATH_CALUDE_f1_extrema_f2_extrema_l318_31853

-- Function 1
def f1 (x : ℝ) : ℝ := x^3 + 2*x

theorem f1_extrema :
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f1 x ≥ f1 x₁) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f1 x ≤ f1 x₂) ∧
  (f1 x₁ = -3) ∧ (f1 x₂ = 3) :=
sorry

-- Function 2
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_extrema :
  (∃ x₁ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f2 x ≥ f2 x₁) ∧
  (∃ x₂ ∈ Set.Icc 0 3, ∀ x ∈ Set.Icc 0 3, f2 x ≤ f2 x₂) ∧
  (f2 x₁ = -4) ∧ (f2 x₂ = 2) :=
sorry

end NUMINAMATH_CALUDE_f1_extrema_f2_extrema_l318_31853


namespace NUMINAMATH_CALUDE_solve_for_a_l318_31859

-- Define the function f(x) = x^2 + ax
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem solve_for_a (a : ℝ) (h1 : a < -1) :
  (∀ x : ℝ, f a x ≤ -x) ∧ 
  (∃ x : ℝ, f a x = -x) ∧
  (∀ x : ℝ, f a x ≥ -1/2) ∧
  (∃ x : ℝ, f a x = -1/2) →
  a = -3/2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l318_31859


namespace NUMINAMATH_CALUDE_katies_friends_games_l318_31895

/-- Given that Katie has 57 new games and 39 old games, and she has 62 more games than her friends,
    prove that her friends have 34 new games. -/
theorem katies_friends_games (katie_new : ℕ) (katie_old : ℕ) (difference : ℕ) 
    (h1 : katie_new = 57)
    (h2 : katie_old = 39)
    (h3 : difference = 62) :
  katie_new + katie_old - difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_katies_friends_games_l318_31895


namespace NUMINAMATH_CALUDE_right_side_number_l318_31825

theorem right_side_number (x : ℝ) (some_number : ℝ) 
  (h1 : x + 1 = some_number) (h2 : x = 1) : some_number = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_side_number_l318_31825


namespace NUMINAMATH_CALUDE_OPRQ_shape_l318_31891

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Represents the figure OPRQ -/
structure OPRQ :=
  (O : Point)
  (P : Point)
  (Q : Point)
  (R : Point)
  (h_distinct : P ≠ Q)
  (h_R : R.x = P.x + Q.x ∧ R.y = P.y + Q.y)
  (h_O : O.x = 0 ∧ O.y = 0)

/-- The figure OPRQ is a parallelogram -/
def is_parallelogram (f : OPRQ) : Prop :=
  f.O.x + f.R.x = f.P.x + f.Q.x ∧ f.O.y + f.R.y = f.P.y + f.Q.y

/-- The figure OPRQ is a straight line -/
def is_straight_line (f : OPRQ) : Prop :=
  collinear f.O f.P f.Q ∧ collinear f.O f.P f.R

theorem OPRQ_shape (f : OPRQ) :
  is_parallelogram f ∨ is_straight_line f :=
sorry

end NUMINAMATH_CALUDE_OPRQ_shape_l318_31891


namespace NUMINAMATH_CALUDE_shadow_length_problem_l318_31875

theorem shadow_length_problem (lamp_height person_height initial_distance initial_shadow new_distance : ℝ) :
  lamp_height = 8 →
  initial_distance = 12 →
  initial_shadow = 4 →
  new_distance = 8 →
  person_height / initial_shadow = lamp_height / (initial_distance + initial_shadow) →
  person_height / (lamp_height / new_distance * new_distance - new_distance) = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_problem_l318_31875


namespace NUMINAMATH_CALUDE_game_lasts_12_rounds_l318_31801

/-- Represents the state of the game at any point -/
structure GameState where
  tokens_A : ℕ
  tokens_B : ℕ
  tokens_C : ℕ

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., any player has 0 tokens) -/
def game_ended (state : GameState) : Bool :=
  sorry

/-- Plays the game until it ends, returning the number of rounds played -/
def play_game (initial_state : GameState) : ℕ :=
  sorry

/-- The main theorem stating that the game lasts exactly 12 rounds -/
theorem game_lasts_12_rounds :
  let initial_state : GameState := { tokens_A := 14, tokens_B := 13, tokens_C := 12 }
  play_game initial_state = 12 := by
  sorry

end NUMINAMATH_CALUDE_game_lasts_12_rounds_l318_31801


namespace NUMINAMATH_CALUDE_range_of_a_l318_31848

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) → 
  a ∈ Set.Icc 4 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l318_31848


namespace NUMINAMATH_CALUDE_functional_equation_solution_l318_31856

/-- A function f: R⁺ → R⁺ satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f x + y) * f x = f (x * y + 1)

/-- The theorem stating the solution to the functional equation -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    (∀ x, x > 1 → f x = 1 / x) ∧ f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l318_31856


namespace NUMINAMATH_CALUDE_total_cubes_is_seven_l318_31826

/-- Represents a stack of unit cubes -/
structure CubeStack where
  bottomLayer : Nat
  middleLayer : Nat
  topLayer : Nat

/-- The total number of cubes in a stack -/
def totalCubes (stack : CubeStack) : Nat :=
  stack.bottomLayer + stack.middleLayer + stack.topLayer

/-- Given stack of unit cubes -/
def givenStack : CubeStack :=
  { bottomLayer := 4
  , middleLayer := 2
  , topLayer := 1 }

/-- Theorem: The total number of unit cubes in the given stack is 7 -/
theorem total_cubes_is_seven : totalCubes givenStack = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_cubes_is_seven_l318_31826


namespace NUMINAMATH_CALUDE_kate_savings_ratio_l318_31855

theorem kate_savings_ratio (pen_cost : ℕ) (kate_needs : ℕ) : 
  pen_cost = 30 → kate_needs = 20 → 
  (pen_cost - kate_needs) / pen_cost = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_kate_savings_ratio_l318_31855


namespace NUMINAMATH_CALUDE_painting_class_selection_l318_31806

theorem painting_class_selection (n k : ℕ) (hn : n = 10) (hk : k = 4) :
  Nat.choose n k = 210 := by
  sorry

end NUMINAMATH_CALUDE_painting_class_selection_l318_31806


namespace NUMINAMATH_CALUDE_strawberry_growth_rate_l318_31868

/-- Represents the growth of strawberry plants over time -/
def strawberry_growth (initial_plants : ℕ) (months : ℕ) (plants_given_away : ℕ) (final_plants : ℕ) (growth_rate : ℕ) : Prop :=
  initial_plants + growth_rate * months - plants_given_away = final_plants

/-- Theorem stating that under the given conditions, the growth rate is 7 plants per month -/
theorem strawberry_growth_rate :
  strawberry_growth 3 3 4 20 7 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_growth_rate_l318_31868


namespace NUMINAMATH_CALUDE_place_balls_in_boxes_l318_31827

/-- The number of ways to place 4 out of 5 distinct objects into 3 distinct containers, with no container empty -/
def ways_to_place (n m k : ℕ) : ℕ :=
  (n.choose m) * (m.choose 2) * (k.factorial)

/-- Theorem stating that there are 180 ways to place 4 out of 5 distinct objects into 3 distinct containers, with no container empty -/
theorem place_balls_in_boxes : ways_to_place 5 4 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_place_balls_in_boxes_l318_31827


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l318_31805

theorem sufficient_not_necessary (a b : ℝ) : 
  (a^2 + b^2 ≤ 2 → -1 ≤ a*b ∧ a*b ≤ 1) ∧ 
  ∃ a b : ℝ, -1 ≤ a*b ∧ a*b ≤ 1 ∧ a^2 + b^2 > 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l318_31805


namespace NUMINAMATH_CALUDE_perimeter_equality_y_approximation_l318_31899

-- Define the side length of the square
def y : ℝ := sorry

-- Define the radius of the circle
def r : ℝ := 4

-- Theorem stating the equality of square perimeter and circle circumference
theorem perimeter_equality : 4 * y = 2 * Real.pi * r := sorry

-- Theorem stating the approximate value of y
theorem y_approximation : ∃ (ε : ℝ), ε < 0.005 ∧ |y - 6.28| < ε := sorry

end NUMINAMATH_CALUDE_perimeter_equality_y_approximation_l318_31899


namespace NUMINAMATH_CALUDE_toothpicks_for_ten_squares_toothpicks_for_one_square_toothpicks_for_two_squares_toothpicks_pattern_l318_31807

/-- The number of toothpicks required to form n squares in a row -/
def toothpicks (n : ℕ) : ℕ := 4 + 3 * (n - 1)

theorem toothpicks_for_ten_squares : 
  toothpicks 10 = 31 := by sorry

theorem toothpicks_for_one_square : 
  toothpicks 1 = 4 := by sorry

theorem toothpicks_for_two_squares : 
  toothpicks 2 = 7 := by sorry

theorem toothpicks_pattern (n : ℕ) (h : n > 1) : 
  toothpicks n = toothpicks (n-1) + 3 := by sorry

end NUMINAMATH_CALUDE_toothpicks_for_ten_squares_toothpicks_for_one_square_toothpicks_for_two_squares_toothpicks_pattern_l318_31807


namespace NUMINAMATH_CALUDE_friends_recycled_pounds_l318_31841

/-- The number of pounds of paper recycled to earn one point -/
def pounds_per_point : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_pounds : ℕ := 14

/-- The total number of points earned by Paige and her friends -/
def total_points : ℕ := 4

/-- Calculate the number of points earned for a given number of pounds -/
def points_earned (pounds : ℕ) : ℕ :=
  pounds / pounds_per_point

/-- The number of pounds recycled by Paige's friends -/
def friends_pounds : ℕ := 4

theorem friends_recycled_pounds :
  friends_pounds = total_points * pounds_per_point - points_earned paige_pounds * pounds_per_point :=
by sorry

end NUMINAMATH_CALUDE_friends_recycled_pounds_l318_31841


namespace NUMINAMATH_CALUDE_initial_oranges_count_prove_initial_oranges_count_l318_31845

/-- Given that 35 oranges were taken away and 25 oranges remained,
    prove that the initial number of oranges was 60. -/
theorem initial_oranges_count : ℕ → Prop :=
  fun initial : ℕ =>
    ∀ (taken remaining : ℕ),
      taken = 35 →
      remaining = 25 →
      initial = taken + remaining →
      initial = 60

-- The proof is omitted
theorem prove_initial_oranges_count : initial_oranges_count 60 := by sorry

end NUMINAMATH_CALUDE_initial_oranges_count_prove_initial_oranges_count_l318_31845


namespace NUMINAMATH_CALUDE_sum_of_squares_of_orthogonal_matrix_elements_l318_31813

/-- For a 2x2 matrix A, if A^T = A^(-1), then the sum of squares of its elements is 2 -/
theorem sum_of_squares_of_orthogonal_matrix_elements (a b c d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.transpose = A⁻¹) → a^2 + b^2 + c^2 + d^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_squares_of_orthogonal_matrix_elements_l318_31813


namespace NUMINAMATH_CALUDE_sequence_inequality_l318_31804

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ (k m : ℕ), k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1) :
  ∀ (p q : ℕ), p > 0 → q > 0 → |a p / p - a q / q| < 1 / p + 1 / q :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l318_31804


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l318_31838

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 3 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l318_31838


namespace NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l318_31847

/-- The sale price for a 55% profit given the conditions -/
theorem sale_price_for_55_percent_profit 
  (L : ℝ) -- The price at which the loss equals the profit when sold at $832
  (h1 : ∃ C : ℝ, 832 - C = C - L) -- Condition 1
  (h2 : ∃ C : ℝ, 992 = 0.55 * C) -- Condition 2
  : ∃ C : ℝ, C + 992 = 2795.64 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_for_55_percent_profit_l318_31847


namespace NUMINAMATH_CALUDE_arithmetic_seq_inequality_l318_31878

/-- A positive arithmetic sequence with non-zero common difference -/
structure PosArithmeticSeq where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  diff : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

theorem arithmetic_seq_inequality (seq : PosArithmeticSeq) : seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_inequality_l318_31878


namespace NUMINAMATH_CALUDE_hike_water_theorem_l318_31874

/-- Represents the water consumption during Harry's hike --/
structure HikeWater where
  duration : ℝ  -- Duration of the hike in hours
  distance : ℝ  -- Total distance of the hike in miles
  leak_rate : ℝ  -- Leak rate of the canteen in cups per hour
  last_mile_consumption : ℝ  -- Water consumed in the last mile in cups
  first_miles_consumption : ℝ  -- Water consumed per mile for the first 3 miles
  remaining_water : ℝ  -- Water remaining at the end of the hike in cups

/-- Calculates the initial amount of water in the canteen --/
def initial_water (h : HikeWater) : ℝ :=
  h.leak_rate * h.duration +
  h.last_mile_consumption +
  h.first_miles_consumption * (h.distance - 1) +
  h.remaining_water

/-- Theorem stating that the initial amount of water in the canteen was 10 cups --/
theorem hike_water_theorem (h : HikeWater)
  (h_duration : h.duration = 2)
  (h_distance : h.distance = 4)
  (h_leak_rate : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 3)
  (h_first_miles : h.first_miles_consumption = 1)
  (h_remaining : h.remaining_water = 2) :
  initial_water h = 10 := by
  sorry


end NUMINAMATH_CALUDE_hike_water_theorem_l318_31874


namespace NUMINAMATH_CALUDE_parabolas_symmetric_about_y_axis_l318_31889

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2
def parabola2 (x : ℝ) : ℝ := -2 * x^2
def parabola3 (x : ℝ) : ℝ := x^2

-- Define symmetry about y-axis
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Theorem statement
theorem parabolas_symmetric_about_y_axis :
  symmetric_about_y_axis parabola1 ∧
  symmetric_about_y_axis parabola2 ∧
  symmetric_about_y_axis parabola3 :=
sorry

end NUMINAMATH_CALUDE_parabolas_symmetric_about_y_axis_l318_31889


namespace NUMINAMATH_CALUDE_anie_work_schedule_l318_31864

/-- Represents Anie's work schedule and project details -/
structure WorkSchedule where
  normal_hours : ℝ        -- Normal work hours per day
  extra_hours : ℝ         -- Extra hours worked per day
  project_hours : ℝ       -- Total hours for the project
  days_to_finish : ℝ      -- Number of days to finish the project

/-- Calculates Anie's normal work schedule given the conditions -/
def calculate_normal_schedule (w : WorkSchedule) : Prop :=
  w.extra_hours = 5 ∧ 
  w.project_hours = 1500 ∧ 
  w.days_to_finish = 100 ∧
  w.normal_hours = 10

/-- Theorem stating that Anie's normal work schedule is 10 hours per day -/
theorem anie_work_schedule : 
  ∀ w : WorkSchedule, calculate_normal_schedule w → w.normal_hours = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_anie_work_schedule_l318_31864


namespace NUMINAMATH_CALUDE_james_balloons_count_l318_31887

/-- The number of balloons Amy has -/
def amy_balloons : ℕ := 513

/-- The number of additional balloons James has compared to Amy -/
def james_extra_balloons : ℕ := 709

/-- The total number of balloons James has -/
def james_total_balloons : ℕ := amy_balloons + james_extra_balloons

theorem james_balloons_count : james_total_balloons = 1222 := by
  sorry

end NUMINAMATH_CALUDE_james_balloons_count_l318_31887


namespace NUMINAMATH_CALUDE_correct_lineup_choices_l318_31832

/-- Represents the number of players in a basketball team --/
def team_size : ℕ := 5

/-- Represents the total number of players in the rotation --/
def rotation_size : ℕ := 8

/-- Represents the number of centers in the rotation --/
def num_centers : ℕ := 2

/-- Represents the number of point guards in the rotation --/
def num_point_guards : ℕ := 2

/-- Calculates the number of ways to choose a lineup --/
def lineup_choices : ℕ := sorry

theorem correct_lineup_choices : lineup_choices = 28 := by sorry

end NUMINAMATH_CALUDE_correct_lineup_choices_l318_31832


namespace NUMINAMATH_CALUDE_fourteen_sided_figure_area_l318_31814

/-- A fourteen-sided figure on a 1 cm × 1 cm grid -/
structure FourteenSidedFigure where
  /-- The number of sides of the figure -/
  sides : ℕ
  /-- The figure is on a 1 cm × 1 cm grid -/
  on_unit_grid : Bool
  /-- All edges align with grid lines except one diagonal -/
  edges_align : Bool
  /-- The area of the figure in cm² -/
  area : ℝ

/-- Theorem stating the area of the specific fourteen-sided figure -/
theorem fourteen_sided_figure_area (f : FourteenSidedFigure) 
  (h1 : f.sides = 14)
  (h2 : f.on_unit_grid = true)
  (h3 : f.edges_align = true) :
  f.area = 12.5 := by sorry

end NUMINAMATH_CALUDE_fourteen_sided_figure_area_l318_31814


namespace NUMINAMATH_CALUDE_horseback_trip_speed_l318_31830

/-- The speed of Barry and Jim on the first day of their horseback riding trip -/
def first_day_speed : ℝ := 5

/-- The total distance traveled during the three-day trip -/
def total_distance : ℝ := 115

/-- The duration of travel on the first day -/
def first_day_duration : ℝ := 7

/-- The distance traveled on the second day -/
def second_day_distance : ℝ := 36 + 9

/-- The distance traveled on the third day -/
def third_day_distance : ℝ := 35

theorem horseback_trip_speed :
  first_day_speed * first_day_duration + second_day_distance + third_day_distance = total_distance :=
sorry

end NUMINAMATH_CALUDE_horseback_trip_speed_l318_31830


namespace NUMINAMATH_CALUDE_fraction_of_360_l318_31820

theorem fraction_of_360 : (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6) * 360 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_360_l318_31820


namespace NUMINAMATH_CALUDE_three_solutions_implies_b_equals_three_l318_31872

theorem three_solutions_implies_b_equals_three (a b : ℚ) (ha : |a| > 0) :
  (∃! (s : Finset ℚ), s.card = 3 ∧ ∀ x ∈ s, ‖|x - a| - b‖ = 3) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_implies_b_equals_three_l318_31872


namespace NUMINAMATH_CALUDE_function_is_zero_l318_31873

/-- A function f: ℝ → ℝ is bounded on (0,1) -/
def BoundedOn01 (f : ℝ → ℝ) : Prop :=
  ∃ M > 0, ∀ x ∈ Set.Ioo 0 1, |f x| ≤ M

/-- The functional equation that f satisfies -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x^2 * f x - y^2 * f y = (x^2 - y^2) * f (x + y) - x * y * f (x - y)

theorem function_is_zero
  (f : ℝ → ℝ)
  (hb : BoundedOn01 f)
  (hf : SatisfiesFunctionalEq f) :
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l318_31873


namespace NUMINAMATH_CALUDE_winning_percentage_l318_31849

theorem winning_percentage (total_votes : ℕ) (majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 4500 →
  majority = 900 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = majority :=
by
  sorry

end NUMINAMATH_CALUDE_winning_percentage_l318_31849


namespace NUMINAMATH_CALUDE_brians_math_quiz_l318_31890

theorem brians_math_quiz (x : ℝ) : (x - 11) / 5 = 31 → (x - 5) / 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_brians_math_quiz_l318_31890


namespace NUMINAMATH_CALUDE_watermelon_slices_l318_31898

theorem watermelon_slices (danny_watermelons : ℕ) (sister_watermelon : ℕ) 
  (sister_slices : ℕ) (total_slices : ℕ) (danny_slices : ℕ) : 
  danny_watermelons = 3 → 
  sister_watermelon = 1 → 
  sister_slices = 15 → 
  total_slices = 45 → 
  danny_watermelons * danny_slices + sister_watermelon * sister_slices = total_slices → 
  danny_slices = 10 := by
sorry

end NUMINAMATH_CALUDE_watermelon_slices_l318_31898


namespace NUMINAMATH_CALUDE_mean_temperature_l318_31893

def temperatures : List ℝ := [-6.5, -3, -2, 4, 2.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l318_31893


namespace NUMINAMATH_CALUDE_simplify_expression_l318_31817

theorem simplify_expression : (1 / ((-5^4)^2)) * (-5)^9 = 5 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l318_31817


namespace NUMINAMATH_CALUDE_greatest_possible_median_l318_31844

theorem greatest_possible_median (k m p r s t u : ℕ) : 
  (k + m + p + r + s + t + u) / 7 = 24 →
  0 < k → k < m → m < p → p < r → r < s → s < t → t < u →
  t = 54 →
  k + m ≤ 20 →
  r ≤ 53 ∧ ∃ (k' m' p' r' s' t' u' : ℕ), 
    (k' + m' + p' + r' + s' + t' + u') / 7 = 24 ∧
    0 < k' ∧ k' < m' ∧ m' < p' ∧ p' < r' ∧ r' < s' ∧ s' < t' ∧ t' < u' ∧
    t' = 54 ∧
    k' + m' ≤ 20 ∧
    r' = 53 := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l318_31844


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_Fe2SO43_l318_31896

/-- The atomic weight of Iron in g/mol -/
def atomic_weight_Fe : ℝ := 55.845

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The molecular weight of one mole of Iron(III) sulfate in g/mol -/
def molecular_weight_Fe2SO43 : ℝ :=
  2 * atomic_weight_Fe + 3 * (atomic_weight_S + 4 * atomic_weight_O)

/-- The number of moles of Iron(III) sulfate -/
def moles_Fe2SO43 : ℝ := 3

theorem molecular_weight_3_moles_Fe2SO43 :
  moles_Fe2SO43 * molecular_weight_Fe2SO43 = 1199.619 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_Fe2SO43_l318_31896


namespace NUMINAMATH_CALUDE_only_c_is_perfect_square_l318_31866

def option_a : ℕ := 3^3 * 4^4 * 7^7
def option_b : ℕ := 3^4 * 4^5 * 7^6
def option_c : ℕ := 3^6 * 4^6 * 7^4
def option_d : ℕ := 3^5 * 4^4 * 7^6
def option_e : ℕ := 3^6 * 4^7 * 7^5

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem only_c_is_perfect_square : 
  is_perfect_square option_c ∧ 
  ¬is_perfect_square option_a ∧ 
  ¬is_perfect_square option_b ∧ 
  ¬is_perfect_square option_d ∧ 
  ¬is_perfect_square option_e :=
sorry

end NUMINAMATH_CALUDE_only_c_is_perfect_square_l318_31866


namespace NUMINAMATH_CALUDE_simplify_expression_l318_31870

theorem simplify_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h : c^3 + d^3 = 3*(c + d)) : 
  c/d + d/c - 3/(c*d) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l318_31870


namespace NUMINAMATH_CALUDE_matrix_determinant_l318_31819

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, 5; 1, -3, 2; 3, 6, -1]
  Matrix.det A = 57 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l318_31819


namespace NUMINAMATH_CALUDE_vector_perpendicular_l318_31810

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_perpendicular :
  perpendicular (3 * i.1 - j.1, 3 * i.2 - j.2) (i.1 + 3 * j.1, i.2 + 3 * j.2) := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l318_31810


namespace NUMINAMATH_CALUDE_reporters_not_covering_politics_l318_31803

/-- The percentage of reporters who cover local politics in country X -/
def local_politics_coverage : ℝ := 5

/-- The percentage of reporters who cover politics but not local politics in country X -/
def non_local_politics_coverage : ℝ := 30

/-- The percentage of reporters who cover politics and local politics in country X -/
def local_politics_ratio : ℝ := 100 - non_local_politics_coverage

theorem reporters_not_covering_politics (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (p : ℝ), abs (p - 92.86) < ε ∧ 
  p = 100 - (local_politics_coverage * 100 / local_politics_ratio) :=
sorry

end NUMINAMATH_CALUDE_reporters_not_covering_politics_l318_31803


namespace NUMINAMATH_CALUDE_perpendicular_lines_l318_31860

-- Define the coefficients of the two lines
def line1_coeff (a : ℝ) : ℝ × ℝ := (2, a)
def line2_coeff (a : ℝ) : ℝ × ℝ := (a, 2*a - 1)

-- Define the perpendicularity condition
def are_perpendicular (a : ℝ) : Prop :=
  (line1_coeff a).1 * (line2_coeff a).1 + (line1_coeff a).2 * (line2_coeff a).2 = 0

-- State the theorem
theorem perpendicular_lines (a : ℝ) :
  are_perpendicular a ↔ a = -1/2 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l318_31860


namespace NUMINAMATH_CALUDE_wheel_probability_l318_31882

theorem wheel_probability (p_E p_F p_G p_H p_I : ℝ) : 
  p_E = 1/5 →
  p_F = 3/10 →
  p_G = p_H →
  p_I = 2 * p_G →
  p_E + p_F + p_G + p_H + p_I = 1 →
  p_G = 1/8 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l318_31882


namespace NUMINAMATH_CALUDE_triangle_side_length_l318_31892

theorem triangle_side_length (x y z : ℝ) (Y Z : ℝ) :
  y = 7 →
  z = 3 →
  Real.cos (Y - Z) = 11 / 15 →
  x^2 = y^2 + z^2 - 2 * y * z * Real.cos Y →
  x = Real.sqrt 38.4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l318_31892


namespace NUMINAMATH_CALUDE_max_expression_l318_31846

theorem max_expression : ∀ a b c d : ℕ,
  a = 992 ∧ b = 993 ∧ c = 994 ∧ d = 995 →
  (d * (d + 1) + (d + 1) ≥ a * (a + 7) + (a + 7)) ∧
  (d * (d + 1) + (d + 1) ≥ b * (b + 5) + (b + 5)) ∧
  (d * (d + 1) + (d + 1) ≥ c * (c + 3) + (c + 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_expression_l318_31846


namespace NUMINAMATH_CALUDE_remaining_payment_l318_31829

def deposit_percentage : ℚ := 10 / 100
def deposit_amount : ℚ := 105

theorem remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) :
  deposit_percentage = 10 / 100 →
  deposit_amount = 105 →
  (deposit_amount / deposit_percentage) - deposit_amount = 945 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l318_31829


namespace NUMINAMATH_CALUDE_selling_price_formula_l318_31811

/-- Calculates the selling price of a refrigerator to achieve a specific profit margin -/
def calculate_selling_price (L : ℝ) : ℝ :=
  let first_discount := 0.2
  let second_discount := 0.1
  let additional_costs := 475
  let profit_margin := 0.18
  let discounted_price := L * (1 - first_discount) * (1 - second_discount)
  let total_cost := discounted_price + additional_costs
  total_cost + L * profit_margin

/-- Theorem stating the correct selling price formula -/
theorem selling_price_formula (L : ℝ) :
  calculate_selling_price L = 0.9 * L + 475 := by
  sorry

#eval calculate_selling_price 1000  -- Example calculation

end NUMINAMATH_CALUDE_selling_price_formula_l318_31811


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l318_31812

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_l318_31812


namespace NUMINAMATH_CALUDE_distance_is_130_km_l318_31835

/-- The distance between two vehicles at both 5 hours before and 5 hours after they pass each other -/
def distance_between_vehicles (speed1 speed2 : ℝ) : ℝ :=
  (speed1 + speed2) * 2.5

/-- Theorem stating that the distance between the vehicles is 130 km -/
theorem distance_is_130_km :
  distance_between_vehicles 37 15 = 130 := by sorry

end NUMINAMATH_CALUDE_distance_is_130_km_l318_31835


namespace NUMINAMATH_CALUDE_small_boxes_count_l318_31840

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 300) 
  (h2 : chocolates_per_box = 20) : 
  total_chocolates / chocolates_per_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l318_31840


namespace NUMINAMATH_CALUDE_glendas_wedding_fish_l318_31834

/-- Given a wedding reception setup with tables and fish, calculate the total number of fish -/
def total_fish (total_tables : Nat) (regular_fish : Nat) (special_fish : Nat) : Nat :=
  (total_tables - 1) * regular_fish + special_fish

/-- Theorem: The total number of fish at Glenda's wedding reception is 65 -/
theorem glendas_wedding_fish : total_fish 32 2 3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_glendas_wedding_fish_l318_31834


namespace NUMINAMATH_CALUDE_parabola_equation_l318_31828

/-- Represents a parabola with the given properties -/
structure Parabola where
  -- The equation of the parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  c_pos : c > 0
  gcd_one : Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs) e.natAbs) f.natAbs = 1
  passes_through : a * 2^2 + b * 2 * 8 + c * 8^2 + d * 2 + e * 8 + f = 0
  focus_y : ℤ
  focus_y_is_5 : focus_y = 5
  symmetry_parallel_x : b = 0 ∧ a = 0
  vertex_on_y_axis : d = 0

/-- The theorem stating that the specific equation represents the parabola with given properties -/
theorem parabola_equation : ∃ (p : Parabola), p.a = 0 ∧ p.b = 0 ∧ p.c = 2 ∧ p.d = 9 ∧ p.e = -20 ∧ p.f = 50 :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l318_31828


namespace NUMINAMATH_CALUDE_limit_sqrt_sum_to_infinity_l318_31824

/-- The limit of n(√(n^2+1) + √(n^2-1)) as n approaches infinity is infinity. -/
theorem limit_sqrt_sum_to_infinity :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, n * (Real.sqrt (n^2 + 1) + Real.sqrt (n^2 - 1)) > ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sqrt_sum_to_infinity_l318_31824


namespace NUMINAMATH_CALUDE_daniel_speed_l318_31861

/-- The speeds of runners in a marathon preparation --/
def marathon_speeds (eugene_speed : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  let brianna_speed := (3 / 4) * eugene_speed
  let katie_speed := (4 / 3) * brianna_speed
  let daniel_speed := (5 / 6) * katie_speed
  (eugene_speed, brianna_speed, katie_speed, daniel_speed)

/-- Theorem stating Daniel's speed given Eugene's speed --/
theorem daniel_speed (eugene_speed : ℚ) : 
  (marathon_speeds eugene_speed).2.2.2 = 25 / 6 :=
by
  sorry

#eval marathon_speeds 5

end NUMINAMATH_CALUDE_daniel_speed_l318_31861


namespace NUMINAMATH_CALUDE_quadratic_one_zero_point_l318_31857

/-- A quadratic function with coefficient m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2*x + 3

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := 4 - 12*m

theorem quadratic_one_zero_point (m : ℝ) : 
  (∃! x, f m x = 0) ↔ m = 0 ∨ m = 1/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_point_l318_31857


namespace NUMINAMATH_CALUDE_marble_problem_l318_31815

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = a)
  (h2 : brian = 1.5 * a)
  (h3 : caden = 2.5 * brian)
  (h4 : daryl = 4 * caden)
  (h5 : angela + brian + caden + daryl = 90) :
  a = 72 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l318_31815


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l318_31833

/-- The coordinates of the foci of an ellipse -/
def foci_coordinates (m n : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x = Real.sqrt (n - m) ∧ y = 0 ∨ x = -Real.sqrt (n - m) ∧ y = 0}

/-- Theorem: The coordinates of the foci of the ellipse x²/m + y²/n = -1 where m < n < 0 -/
theorem ellipse_foci_coordinates {m n : ℝ} (hm : m < 0) (hn : n < 0) (hmn : m < n) :
  foci_coordinates m n = {(Real.sqrt (n - m), 0), (-Real.sqrt (n - m), 0)} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l318_31833


namespace NUMINAMATH_CALUDE_water_purifier_max_profit_l318_31869

/-- Represents the cost and selling prices of water purifiers --/
structure WaterPurifier where
  costA : ℕ  -- Cost price of A
  costB : ℕ  -- Cost price of B
  sellA : ℕ  -- Selling price of A
  sellB : ℕ  -- Selling price of B

/-- Calculates the maximum profit for selling water purifiers --/
def maxProfit (w : WaterPurifier) (total : ℕ) : ℕ :=
  let profitA := w.sellA - w.costA
  let profitB := w.sellB - w.costB
  let numA := min (total / 2) (total - (total / 2))
  profitA * numA + profitB * (total - numA)

/-- Theorem stating the maximum profit for the given scenario --/
theorem water_purifier_max_profit :
  ∀ (w : WaterPurifier),
    w.costA = w.costB + 300 →
    40000 / w.costA = 30000 / w.costB →
    w.sellA = 1500 →
    w.sellB = 1100 →
    maxProfit w 400 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_water_purifier_max_profit_l318_31869


namespace NUMINAMATH_CALUDE_tina_fruit_difference_l318_31802

/-- Represents the number of fruits in Tina's bag -/
structure FruitBag where
  oranges : ℕ
  tangerines : ℕ

/-- Calculates the difference between tangerines and oranges after removal -/
def tangerine_orange_difference (bag : FruitBag) (oranges_removed : ℕ) (tangerines_removed : ℕ) : ℤ :=
  (bag.tangerines - tangerines_removed) - (bag.oranges - oranges_removed)

theorem tina_fruit_difference :
  let initial_bag : FruitBag := { oranges := 5, tangerines := 17 }
  let oranges_removed := 2
  let tangerines_removed := 10
  tangerine_orange_difference initial_bag oranges_removed tangerines_removed = 4 := by
  sorry

end NUMINAMATH_CALUDE_tina_fruit_difference_l318_31802


namespace NUMINAMATH_CALUDE_find_n_l318_31850

theorem find_n : ∃ n : ℚ, 1/2 + 2/3 + 3/4 + n/12 = 2 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l318_31850


namespace NUMINAMATH_CALUDE_sphere_volume_from_cube_l318_31839

/-- Given a cube with edge length 3 and all its vertices on the same spherical surface,
    the volume of that sphere is 27√3π/2 -/
theorem sphere_volume_from_cube (edge_length : ℝ) (radius : ℝ) :
  edge_length = 3 →
  radius = (3 * Real.sqrt 3) / 2 →
  (4 / 3) * Real.pi * radius^3 = (27 * Real.sqrt 3 * Real.pi) / 2 := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_from_cube_l318_31839


namespace NUMINAMATH_CALUDE_quadratic_shift_roots_l318_31876

/-- Given a quadratic function f(x) = (x-m)^2 + n that intersects the x-axis at (-1,0) and (3,0),
    prove that the solutions to (x-m+2)^2 + n = 0 are -3 and 1. -/
theorem quadratic_shift_roots (m n : ℝ) : 
  (∀ x, (x - m)^2 + n = 0 ↔ x = -1 ∨ x = 3) →
  (∀ x, (x - m + 2)^2 + n = 0 ↔ x = -3 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_shift_roots_l318_31876


namespace NUMINAMATH_CALUDE_cone_slant_height_l318_31881

/-- The slant height of a cone with surface area 5π and net sector angle 90° is 4. -/
theorem cone_slant_height (r l : ℝ) (h1 : π * r^2 + π * r * l = 5 * π) 
  (h2 : 2 * π * r = 1/4 * (2 * π * l)) : l = 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l318_31881


namespace NUMINAMATH_CALUDE_sqrt_difference_power_l318_31886

theorem sqrt_difference_power (m n : ℕ) :
  ∃ k : ℕ, (Real.sqrt m - Real.sqrt (m - 1)) ^ n = Real.sqrt k - Real.sqrt (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_difference_power_l318_31886


namespace NUMINAMATH_CALUDE_joes_mens_haircuts_l318_31883

def women_haircut_time : ℕ := 50
def men_haircut_time : ℕ := 15
def kids_haircut_time : ℕ := 25
def num_women : ℕ := 3
def num_kids : ℕ := 3
def total_time : ℕ := 255

theorem joes_mens_haircuts :
  ∃ (num_men : ℕ),
    num_men * men_haircut_time +
    num_women * women_haircut_time +
    num_kids * kids_haircut_time = total_time ∧
    num_men = 2 := by
  sorry

end NUMINAMATH_CALUDE_joes_mens_haircuts_l318_31883


namespace NUMINAMATH_CALUDE_speed_ratio_is_five_sixths_l318_31897

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure MotionProblem where
  /-- Speed of object A in yards per minute -/
  vA : ℝ
  /-- Speed of object B in yards per minute -/
  vB : ℝ
  /-- Initial distance of B from point O in yards -/
  initialDistB : ℝ
  /-- Time when A and B are first equidistant from O in minutes -/
  t1 : ℝ
  /-- Time when A and B are again equidistant from O in minutes -/
  t2 : ℝ

/-- The theorem stating the ratio of speeds given the problem conditions -/
theorem speed_ratio_is_five_sixths (p : MotionProblem)
  (h1 : p.initialDistB = 500)
  (h2 : p.t1 = 2)
  (h3 : p.t2 = 10)
  (h4 : p.t1 * p.vA = abs (p.initialDistB - p.t1 * p.vB))
  (h5 : p.t2 * p.vA = abs (p.initialDistB - p.t2 * p.vB)) :
  p.vA / p.vB = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_five_sixths_l318_31897


namespace NUMINAMATH_CALUDE_james_return_to_heavy_lifting_l318_31837

/-- Calculates the total number of days before James can return to heavy lifting after his injury. -/
def time_to_heavy_lifting (initial_pain_days : ℕ) (healing_multiplier : ℕ) (additional_rest_days : ℕ) 
  (light_exercise_weeks : ℕ) (moderate_exercise_weeks : ℕ) (final_wait_weeks : ℕ) : ℕ :=
  let full_healing_days := initial_pain_days * healing_multiplier
  let total_rest_days := full_healing_days + additional_rest_days
  let light_exercise_days := light_exercise_weeks * 7
  let moderate_exercise_days := moderate_exercise_weeks * 7
  let final_wait_days := final_wait_weeks * 7
  total_rest_days + light_exercise_days + moderate_exercise_days + final_wait_days

/-- Theorem stating that James can return to heavy lifting after 60 days. -/
theorem james_return_to_heavy_lifting : 
  time_to_heavy_lifting 3 5 3 2 1 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_return_to_heavy_lifting_l318_31837


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l318_31862

theorem gcd_digits_bound (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 →
  10000 ≤ b ∧ b < 100000 →
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l318_31862


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_10_l318_31822

theorem units_digit_of_2_power_10 : (2^10 : ℕ) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_10_l318_31822


namespace NUMINAMATH_CALUDE_river_flow_volume_l318_31816

/-- Given a river with specified dimensions and flow rate, calculate the volume of water flowing per minute -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 2) 
  (h_width : width = 45) 
  (h_flow_rate : flow_rate_kmph = 3) : 
  depth * width * (flow_rate_kmph * 1000 / 60) = 9000 := by
  sorry

end NUMINAMATH_CALUDE_river_flow_volume_l318_31816


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l318_31823

/-- Given a point P with polar coordinates (1, π), 
    the equation of the line passing through P and perpendicular to the polar axis is ρ = -1 / (cos θ) -/
theorem perpendicular_line_equation (P : ℝ × ℝ) (h : P = (1, π)) :
  ∃ (f : ℝ → ℝ), (∀ θ, f θ = -1 / (Real.cos θ)) ∧ 
  (∀ ρ θ, (ρ * Real.cos θ = -1) ↔ (ρ = f θ)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l318_31823


namespace NUMINAMATH_CALUDE_linear_function_solution_l318_31851

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the domain and range constraints
def domain_constraint (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4
def range_constraint (y : ℝ) : Prop := 3 ≤ y ∧ y ≤ 6

-- Theorem statement
theorem linear_function_solution (k b : ℝ) :
  (∀ x, domain_constraint x → range_constraint (linear_function k b x)) →
  ((k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_solution_l318_31851


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_smallest_perimeter_l318_31867

/-- A quadrilateral represented by four points in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Predicate to check if a quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is inscribed in another -/
def isInscribed (inner outer : Quadrilateral) : Prop := sorry

/-- Function to calculate the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Theorem about the existence of inscribed quadrilaterals with smallest perimeter -/
theorem inscribed_quadrilateral_smallest_perimeter (ABCD : Quadrilateral) :
  (isCyclic ABCD → ∃ (S : Set Quadrilateral), 
    (∀ q ∈ S, isInscribed q ABCD) ∧ 
    (∀ q ∈ S, ∀ p, isInscribed p ABCD → perimeter q ≤ perimeter p) ∧
    (Set.Infinite S)) ∧
  (¬isCyclic ABCD → ¬∃ q : Quadrilateral, 
    isInscribed q ABCD ∧ 
    (∀ p, isInscribed p ABCD → perimeter q ≤ perimeter p) ∧
    (q.A ≠ q.B ∧ q.B ≠ q.C ∧ q.C ≠ q.D ∧ q.D ≠ q.A)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_smallest_perimeter_l318_31867


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l318_31842

/-- Given two lines with different non-zero y-intercepts:
    - First line has slope 8 and x-intercept (r, 0)
    - Second line has slope 4 and x-intercept (q, 0)
    - First line's y-intercept is double that of the second line
    Prove that the ratio of r to q is 1 -/
theorem ratio_of_x_intercepts (r q c : ℝ) : 
  (8 * r + 2 * c = 0) →  -- First line equation at x-intercept
  (4 * q + c = 0) →      -- Second line equation at x-intercept
  r / q = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l318_31842


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l318_31808

/-- Given a curve y = x^3 + ax + 1, if there exists a point where the tangent line is y = 2x + 1, then a = 2 -/
theorem tangent_line_implies_a_equals_two (a : ℝ) : 
  (∃ x₀ y₀ : ℝ, y₀ = x₀^3 + a*x₀ + 1 ∧ 
    (∀ x : ℝ, (x - x₀) * (3*x₀^2 + a) + y₀ = 2*x + 1)) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l318_31808


namespace NUMINAMATH_CALUDE_rancher_problem_l318_31854

theorem rancher_problem (s c : ℕ) : s > 0 ∧ c > 0 ∧ 30 * s + 31 * c = 1200 → s = 9 ∧ c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rancher_problem_l318_31854


namespace NUMINAMATH_CALUDE_solve_systems_of_equations_l318_31885

theorem solve_systems_of_equations :
  -- First system
  (∃ x y : ℝ, 3 * x - y = 8 ∧ 3 * x - 5 * y = -20 → x = 5 ∧ y = 7) ∧
  -- Second system
  (∃ x y : ℝ, x / 3 - y / 2 = -1 ∧ 3 * x - 2 * y = 1 → x = 3 ∧ y = 4) :=
by sorry


end NUMINAMATH_CALUDE_solve_systems_of_equations_l318_31885


namespace NUMINAMATH_CALUDE_max_value_at_two_l318_31879

/-- A function f(x) = ax² + 4(a-1)x - 3 defined on the interval [0,2] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a - 1) * x - 3

/-- The domain of the function -/
def domain : Set ℝ := Set.Icc 0 2

theorem max_value_at_two (a : ℝ) :
  (∀ x ∈ domain, f a x ≤ f a 2) → a ∈ Set.Ici (2/3) :=
sorry

end NUMINAMATH_CALUDE_max_value_at_two_l318_31879


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l318_31894

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  x - 2*y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l318_31894
