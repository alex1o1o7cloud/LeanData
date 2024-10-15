import Mathlib

namespace NUMINAMATH_CALUDE_adrian_days_off_l1228_122893

/-- The number of days Adrian took off in a year -/
def total_holidays : ℕ := 48

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of days Adrian took off each month -/
def days_off_per_month : ℕ := total_holidays / months_in_year

theorem adrian_days_off :
  days_off_per_month = 4 :=
by sorry

end NUMINAMATH_CALUDE_adrian_days_off_l1228_122893


namespace NUMINAMATH_CALUDE_spherical_to_cartesian_coordinates_l1228_122894

/-- Given a point M with spherical coordinates (1, π/3, π/6), 
    prove that its Cartesian coordinates are (3/4, √3/4, 1/2). -/
theorem spherical_to_cartesian_coordinates :
  let r : ℝ := 1
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := r * Real.sin θ * Real.cos φ
  let y : ℝ := r * Real.sin θ * Real.sin φ
  let z : ℝ := r * Real.cos θ
  (x = 3/4) ∧ (y = Real.sqrt 3 / 4) ∧ (z = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_spherical_to_cartesian_coordinates_l1228_122894


namespace NUMINAMATH_CALUDE_angle_bisector_inequality_l1228_122855

/-- A triangle with sides a, b, c and angle bisectors fa, fb, fc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  fa : ℝ
  fb : ℝ
  fc : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_fa : fa > 0
  pos_fb : fb > 0
  pos_fc : fc > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The inequality holds for any triangle -/
theorem angle_bisector_inequality (t : Triangle) :
  1 / t.fa + 1 / t.fb + 1 / t.fc > 1 / t.a + 1 / t.b + 1 / t.c := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_inequality_l1228_122855


namespace NUMINAMATH_CALUDE_season_games_count_l1228_122891

/-- The number of teams in the conference -/
def num_teams : ℕ := 10

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := num_teams * (num_teams - 1) + num_teams * non_conference_games

theorem season_games_count :
  total_games = 150 :=
by sorry

end NUMINAMATH_CALUDE_season_games_count_l1228_122891


namespace NUMINAMATH_CALUDE_eriks_remaining_money_is_43_47_l1228_122804

/-- Calculates the amount of money Erik has left after his purchase --/
def eriks_remaining_money (initial_amount : ℚ) (bread_price carton_price egg_price chocolate_price : ℚ)
  (bread_quantity carton_quantity egg_quantity chocolate_quantity : ℕ)
  (discount_rate tax_rate : ℚ) : ℚ :=
  let total_cost := bread_price * bread_quantity + carton_price * carton_quantity +
                    egg_price * egg_quantity + chocolate_price * chocolate_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  initial_amount - final_cost

/-- Theorem stating that Erik has $43.47 left after his purchase --/
theorem eriks_remaining_money_is_43_47 :
  eriks_remaining_money 86 3 6 4 2 3 3 2 5 (1/10) (1/20) = 43.47 := by
  sorry

end NUMINAMATH_CALUDE_eriks_remaining_money_is_43_47_l1228_122804


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1228_122803

theorem gcd_of_three_numbers : Nat.gcd 7254 (Nat.gcd 10010 22554) = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1228_122803


namespace NUMINAMATH_CALUDE_divisibility_of_powers_l1228_122866

-- Define the polynomial and its greatest positive root
def f (x : ℝ) := x^3 - 3*x^2 + 1

def a : ℝ := sorry

axiom a_is_root : f a = 0

axiom a_is_greatest_positive_root : 
  ∀ x > 0, f x = 0 → x ≤ a

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem divisibility_of_powers : 
  (17 ∣ floor (a^1788)) ∧ (17 ∣ floor (a^1988)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_powers_l1228_122866


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_23_l1228_122880

theorem largest_negative_congruent_to_one_mod_23 : 
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 23] → n ≤ -9994 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_one_mod_23_l1228_122880


namespace NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l1228_122837

theorem solution_to_logarithmic_equation :
  ∃ x : ℝ, (3 * Real.log x - 4 * Real.log 5 = -1) ∧ (x = (62.5 : ℝ) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l1228_122837


namespace NUMINAMATH_CALUDE_sum_of_x_y_z_l1228_122874

theorem sum_of_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_y_z_l1228_122874


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l1228_122817

theorem smaller_root_of_equation : 
  ∃ (x : ℚ), (x - 5/6)^2 + (x - 5/6)*(x - 2/3) = 0 ∧ 
  x = 5/6 ∧ 
  ∀ y, ((y - 5/6)^2 + (y - 5/6)*(y - 2/3) = 0 → y ≥ 5/6) :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l1228_122817


namespace NUMINAMATH_CALUDE_charlotte_boots_cost_l1228_122834

/-- Calculates the amount Charlotte needs to bring to buy discounted boots -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (discount_rate * original_price)

/-- Proves that Charlotte needs to bring $72 for the boots -/
theorem charlotte_boots_cost : discounted_price 90 0.2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_charlotte_boots_cost_l1228_122834


namespace NUMINAMATH_CALUDE_bacteria_in_seventh_generation_l1228_122844

/-- The number of bacteria in a given generation -/
def bacteria_count (generation : ℕ) : ℕ :=
  match generation with
  | 0 => 1  -- First generation
  | n + 1 => 4 * bacteria_count n  -- Subsequent generations

/-- Theorem stating the number of bacteria in the seventh generation -/
theorem bacteria_in_seventh_generation :
  bacteria_count 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_in_seventh_generation_l1228_122844


namespace NUMINAMATH_CALUDE_exists_valid_formula_l1228_122806

def uses_five_twos (formula : ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2) ∧
    ∀ n, formula n = f a b c d e
  where f := λ a b c d e => sorry -- placeholder for the actual formula

def is_valid_formula (formula : ℕ → ℕ) : Prop :=
  uses_five_twos formula ∧
  (∀ n, n ∈ Finset.range 10 → formula (n + 11) = n + 11)

theorem exists_valid_formula : ∃ formula, is_valid_formula formula := by
  sorry

#check exists_valid_formula

end NUMINAMATH_CALUDE_exists_valid_formula_l1228_122806


namespace NUMINAMATH_CALUDE_square_root_equation_solutions_cube_root_equation_solution_l1228_122826

theorem square_root_equation_solutions (x : ℝ) :
  (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by sorry

theorem cube_root_equation_solution (x : ℝ) :
  (x - 2)^3 = -125 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_square_root_equation_solutions_cube_root_equation_solution_l1228_122826


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1228_122848

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1228_122848


namespace NUMINAMATH_CALUDE_inner_triangle_area_l1228_122805

/-- Given a triangle with area T, the area of the smaller triangle formed by
    joining the points that divide each side into three equal segments is 4/9 * T -/
theorem inner_triangle_area (T : ℝ) (h : T > 0) :
  ∃ (inner_area : ℝ), inner_area = (4 / 9) * T := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_area_l1228_122805


namespace NUMINAMATH_CALUDE_x_intercept_distance_l1228_122830

/-- Given two lines with slopes 4 and -2 intersecting at (8, 20),
    the distance between their x-intercepts is 15. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * x - 12) →  -- Equation of line1
  (∀ x, line2 x = -2 * x + 36) →  -- Equation of line2
  line1 8 = 20 →  -- Intersection point
  line2 8 = 20 →  -- Intersection point
  |((36 : ℝ) / 2) - (12 / 4)| = 15 := by
  sorry


end NUMINAMATH_CALUDE_x_intercept_distance_l1228_122830


namespace NUMINAMATH_CALUDE_expression_equals_four_l1228_122847

theorem expression_equals_four :
  (-2022)^0 - 2 * Real.tan (45 * π / 180) + |(-2)| + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l1228_122847


namespace NUMINAMATH_CALUDE_caramel_candies_count_l1228_122841

theorem caramel_candies_count (total : ℕ) (lemon : ℕ) (caramel : ℕ) : 
  lemon = 4 →
  (caramel : ℚ) / (total : ℚ) = 3 / 7 →
  total = lemon + caramel →
  caramel = 3 := by
sorry

end NUMINAMATH_CALUDE_caramel_candies_count_l1228_122841


namespace NUMINAMATH_CALUDE_chessboard_clearable_l1228_122835

/-- Represents the number of chips on a chessboard -/
def Chessboard := Fin 8 → Fin 8 → ℕ

/-- Represents an operation on the chessboard -/
inductive Operation
  | remove_column : Fin 8 → Operation
  | double_row : Fin 8 → Operation

/-- Applies an operation to a chessboard -/
def apply_operation (board : Chessboard) (op : Operation) : Chessboard :=
  match op with
  | Operation.remove_column j => fun i k => if k = j then (board i k).pred else board i k
  | Operation.double_row i => fun k j => if k = i then 2 * (board k j) else board k j

/-- Checks if the board is cleared (all cells are zero) -/
def is_cleared (board : Chessboard) : Prop :=
  ∀ i j, board i j = 0

theorem chessboard_clearable (initial_board : Chessboard) :
  ∃ (ops : List Operation), is_cleared (ops.foldl apply_operation initial_board) :=
sorry

end NUMINAMATH_CALUDE_chessboard_clearable_l1228_122835


namespace NUMINAMATH_CALUDE_linear_regression_change_l1228_122865

/-- Given a linear regression equation y = 2 - 1.5x, prove that when x increases by 1, y decreases by 1.5. -/
theorem linear_regression_change (x y : ℝ) : 
  y = 2 - 1.5 * x → (2 - 1.5 * (x + 1)) = y - 1.5 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_change_l1228_122865


namespace NUMINAMATH_CALUDE_derivative_symmetry_l1228_122876

/-- Given a function f(x) = ax^4 + bx^2 + c, if f'(1) = 2, then f'(-1) = -2 -/
theorem derivative_symmetry (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^4 + b * x^2 + c
  let f' := fun (x : ℝ) => 4 * a * x^3 + 2 * b * x
  f' 1 = 2 → f' (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l1228_122876


namespace NUMINAMATH_CALUDE_sum_of_x_values_l1228_122854

theorem sum_of_x_values (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, Real.sqrt ((x - 2)^2) = 8 ↔ x = x₁ ∨ x = x₂) ∧ x₁ + x₂ = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l1228_122854


namespace NUMINAMATH_CALUDE_sum_85_to_100_l1228_122800

def sum_consecutive_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_85_to_100 :
  sum_consecutive_integers 85 100 = 1480 :=
by sorry

end NUMINAMATH_CALUDE_sum_85_to_100_l1228_122800


namespace NUMINAMATH_CALUDE_janet_stickers_l1228_122812

theorem janet_stickers (S : ℕ) : 
  S > 2 ∧ 
  S % 5 = 2 ∧ 
  S % 11 = 2 ∧ 
  S % 13 = 2 ∧ 
  (∀ T : ℕ, T > 2 ∧ T % 5 = 2 ∧ T % 11 = 2 ∧ T % 13 = 2 → S ≤ T) → 
  S = 717 := by
sorry

end NUMINAMATH_CALUDE_janet_stickers_l1228_122812


namespace NUMINAMATH_CALUDE_percentage_equality_l1228_122838

theorem percentage_equality (x : ℝ) : (15 / 100 * 75 = 2.5 / 100 * x) → x = 450 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1228_122838


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1228_122883

theorem min_values_xy_and_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : xy ≥ 36 ∧ x + y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1228_122883


namespace NUMINAMATH_CALUDE_overhead_cost_reduction_l1228_122801

/-- Represents the cost components of manufacturing a car --/
structure CarCost where
  raw_material : ℝ
  labor : ℝ
  overhead : ℝ

/-- Calculates the total cost of manufacturing a car --/
def total_cost (cost : CarCost) : ℝ :=
  cost.raw_material + cost.labor + cost.overhead

theorem overhead_cost_reduction 
  (initial_cost : CarCost) 
  (new_cost : CarCost) 
  (h1 : initial_cost.raw_material = (4/9) * total_cost initial_cost)
  (h2 : initial_cost.labor = (3/9) * total_cost initial_cost)
  (h3 : initial_cost.overhead = (2/9) * total_cost initial_cost)
  (h4 : new_cost.raw_material = 1.1 * initial_cost.raw_material)
  (h5 : new_cost.labor = 1.08 * initial_cost.labor)
  (h6 : total_cost new_cost = 1.06 * total_cost initial_cost) :
  new_cost.overhead = 0.95 * initial_cost.overhead :=
sorry

end NUMINAMATH_CALUDE_overhead_cost_reduction_l1228_122801


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1228_122842

/-- For a parabola y² = 2px, if the distance from (4, 0) to the focus (p/2, 0) is 5, then p = 8 -/
theorem parabola_focus_distance (p : ℝ) : 
  (∀ y : ℝ, y^2 = 2*p*4) → -- point (4, y) is on the parabola
  ((4 - p/2)^2 + 0^2)^(1/2) = 5 → -- distance from (4, 0) to focus (p/2, 0) is 5
  p = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1228_122842


namespace NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l1228_122879

theorem max_value_3xy_plus_yz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  3*x*y + y*z ≤ Real.sqrt 10 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_3xy_plus_yz_l1228_122879


namespace NUMINAMATH_CALUDE_jerseys_sold_is_two_l1228_122811

/-- The profit made from selling one jersey -/
def profit_per_jersey : ℕ := 76

/-- The total profit made from selling jerseys during the game -/
def total_profit : ℕ := 152

/-- The number of jerseys sold during the game -/
def jerseys_sold : ℕ := total_profit / profit_per_jersey

theorem jerseys_sold_is_two : jerseys_sold = 2 := by sorry

end NUMINAMATH_CALUDE_jerseys_sold_is_two_l1228_122811


namespace NUMINAMATH_CALUDE_sum_always_positive_l1228_122887

/-- A monotonically increasing odd function -/
def MonoIncreasingOdd (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : MonoIncreasingOdd f)
  (h2 : ArithmeticSequence a)
  (h3 : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_always_positive_l1228_122887


namespace NUMINAMATH_CALUDE_partner_calculation_l1228_122833

theorem partner_calculation (x : ℝ) : 3 * (3 * (x + 2) - 2) = 3 * (3 * x + 4) := by
  sorry

#check partner_calculation

end NUMINAMATH_CALUDE_partner_calculation_l1228_122833


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1228_122853

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℚ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_first : a 1 = 7/9)
  (h_thirteenth : a 13 = 4/5) :
  a 7 = 71/90 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1228_122853


namespace NUMINAMATH_CALUDE_max_min_values_l1228_122869

theorem max_min_values (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 = 2) :
  (a + b + c ≤ Real.sqrt 6) ∧
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 * Real.sqrt 6 / 4) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l1228_122869


namespace NUMINAMATH_CALUDE_angle_D_measure_l1228_122808

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Measure of angle E in degrees -/
  angle_E : ℝ
  /-- The triangle is isosceles with angle D congruent to angle F -/
  isosceles : True
  /-- The measure of angle F is three times the measure of angle E -/
  angle_F_eq_three_E : True

/-- Theorem: In the given isosceles triangle, the measure of angle D is 77 1/7 degrees -/
theorem angle_D_measure (t : IsoscelesTriangle) : 
  (3 * t.angle_E : ℝ) = 77 + 1/7 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l1228_122808


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1228_122857

theorem inequality_equivalence (x : ℝ) :
  (x + 1) * (1 / x - 1) > 0 ↔ x ∈ Set.Ioi (-1) ∪ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1228_122857


namespace NUMINAMATH_CALUDE_valid_coloring_iff_odd_l1228_122881

/-- A coloring of edges and diagonals of an n-gon -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin n

/-- Predicate for a valid coloring -/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i →
    ∃ (x y z : Fin n), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
      c x y = i ∧ c y z = j ∧ c z x = k

/-- Theorem: A valid coloring exists if and only if n is odd -/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ c : Coloring n, is_valid_coloring n c) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_coloring_iff_odd_l1228_122881


namespace NUMINAMATH_CALUDE_nina_weekend_earnings_l1228_122831

/-- Calculates the total money made from jewelry sales --/
def total_money_made (necklace_price bracelet_price earring_pair_price ensemble_price : ℚ)
                     (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_pair_price * (earrings_sold / 2) +
  ensemble_price * ensembles_sold

/-- Theorem: Nina's weekend earnings --/
theorem nina_weekend_earnings :
  let necklace_price : ℚ := 25
  let bracelet_price : ℚ := 15
  let earring_pair_price : ℚ := 10
  let ensemble_price : ℚ := 45
  let necklaces_sold : ℕ := 5
  let bracelets_sold : ℕ := 10
  let earrings_sold : ℕ := 20
  let ensembles_sold : ℕ := 2
  total_money_made necklace_price bracelet_price earring_pair_price ensemble_price
                    necklaces_sold bracelets_sold earrings_sold ensembles_sold = 465 :=
by
  sorry

end NUMINAMATH_CALUDE_nina_weekend_earnings_l1228_122831


namespace NUMINAMATH_CALUDE_monotonic_increasing_sine_cosine_function_l1228_122850

theorem monotonic_increasing_sine_cosine_function (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 4), Monotone (fun x => a * Real.sin x + Real.cos x)) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_sine_cosine_function_l1228_122850


namespace NUMINAMATH_CALUDE_triangular_bipyramid_existence_condition_l1228_122849

/-- A triangular bipyramid with four edges of length 1 and two edges of length x -/
structure TriangularBipyramid (x : ℝ) :=
  (edge_length_1 : ℝ := 1)
  (edge_length_x : ℝ := x)
  (num_edges_1 : ℕ := 4)
  (num_edges_x : ℕ := 2)

/-- The existence condition for a triangular bipyramid -/
def exists_triangular_bipyramid (x : ℝ) : Prop :=
  0 < x ∧ x < (Real.sqrt 6 + Real.sqrt 2) / 2

/-- Theorem stating the range of x for which a triangular bipyramid can exist -/
theorem triangular_bipyramid_existence_condition (x : ℝ) :
  (∃ t : TriangularBipyramid x, True) ↔ exists_triangular_bipyramid x :=
sorry

end NUMINAMATH_CALUDE_triangular_bipyramid_existence_condition_l1228_122849


namespace NUMINAMATH_CALUDE_mod_twelve_difference_l1228_122864

theorem mod_twelve_difference (n : ℕ) : (51 ^ n - 27 ^ n) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_difference_l1228_122864


namespace NUMINAMATH_CALUDE_specific_wall_rows_l1228_122814

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  totalBricks : ℕ
  bottomRowBricks : ℕ
  (total_positive : 0 < totalBricks)
  (bottom_positive : 0 < bottomRowBricks)
  (bottom_leq_total : bottomRowBricks ≤ totalBricks)

/-- Calculates the number of rows in a brick wall -/
def numberOfRows (wall : BrickWall) : ℕ :=
  sorry

/-- Theorem stating that a wall with 100 total bricks and 18 bricks in the bottom row has 8 rows -/
theorem specific_wall_rows :
  ∀ (wall : BrickWall),
    wall.totalBricks = 100 →
    wall.bottomRowBricks = 18 →
    numberOfRows wall = 8 :=
  sorry

end NUMINAMATH_CALUDE_specific_wall_rows_l1228_122814


namespace NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l1228_122868

def alphabet_size : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8

theorem probability_of_letter_in_mathematics :
  (unique_letters_in_mathematics : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_mathematics_l1228_122868


namespace NUMINAMATH_CALUDE_curve_equation_l1228_122875

/-- Given a curve ax² + by² = 2 passing through (0, 5/3) and (1, 1), with a + b = 2,
    prove that the equation of the curve is 16/25 * x² + 9/25 * y² = 1 -/
theorem curve_equation (a b : ℝ) :
  (∀ x y : ℝ, a * x^2 + b * y^2 = 2) →
  (a * 0^2 + b * (5/3)^2 = 2) →
  (a * 1^2 + b * 1^2 = 2) →
  (a + b = 2) →
  (∀ x y : ℝ, 16/25 * x^2 + 9/25 * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_l1228_122875


namespace NUMINAMATH_CALUDE_student_count_l1228_122832

theorem student_count (initial_avg : ℝ) (incorrect_height : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 175 →
  incorrect_height = 151 →
  actual_height = 136 →
  actual_avg = 174.5 →
  ∃ n : ℕ, n = 30 ∧ n * actual_avg = n * initial_avg - (incorrect_height - actual_height) :=
by sorry

end NUMINAMATH_CALUDE_student_count_l1228_122832


namespace NUMINAMATH_CALUDE_email_count_proof_l1228_122856

/-- Calculates the total number of emails received in a month with changing email rates -/
def total_emails (days_in_month : ℕ) (initial_rate : ℕ) (new_rate : ℕ) : ℕ :=
  let half_month := days_in_month / 2
  let first_half := initial_rate * half_month
  let second_half := new_rate * half_month
  first_half + second_half

/-- Proves that given the specified conditions, the total number of emails is 675 -/
theorem email_count_proof :
  let days_in_month : ℕ := 30
  let initial_rate : ℕ := 20
  let new_rate : ℕ := 25
  total_emails days_in_month initial_rate new_rate = 675 := by
  sorry


end NUMINAMATH_CALUDE_email_count_proof_l1228_122856


namespace NUMINAMATH_CALUDE_expression_simplification_l1228_122818

theorem expression_simplification (x : ℤ) (h1 : -2 < x) (h2 : x ≤ 2) (h3 : x = 2) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1228_122818


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1228_122839

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 4*x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - x - 3 = 0
  let sol1 : Set ℝ := {3, 1}
  let sol2 : Set ℝ := {(1 + Real.sqrt 13) / 2, (1 - Real.sqrt 13) / 2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1228_122839


namespace NUMINAMATH_CALUDE_quadratic_triple_root_l1228_122878

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is triple the other, 
    then 3b^2 = 16ac -/
theorem quadratic_triple_root (a b c : ℝ) (h : ∃ x y : ℝ, x ≠ 0 ∧ 
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) : 
  3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_l1228_122878


namespace NUMINAMATH_CALUDE_ball_probability_pairs_l1228_122820

theorem ball_probability_pairs : 
  ∃! k : ℕ, ∃ S : Finset (ℕ × ℕ),
    (∀ (m n : ℕ), (m, n) ∈ S ↔ 
      (m > n ∧ n ≥ 4 ∧ m + n ≤ 40 ∧ (m - n)^2 = m + n)) ∧
    S.card = k ∧ k = 3 := by sorry

end NUMINAMATH_CALUDE_ball_probability_pairs_l1228_122820


namespace NUMINAMATH_CALUDE_complex_roots_theorem_l1228_122828

theorem complex_roots_theorem (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (a + 5 * Complex.I) * (b + 6 * Complex.I) = 9 + 61 * Complex.I →
  (a + 5 * Complex.I) + (b + 6 * Complex.I) = 12 + 11 * Complex.I →
  (a, b) = (9, 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_theorem_l1228_122828


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1228_122884

theorem simplify_fraction_product : (144 : ℚ) / 18 * 9 / 108 * 6 / 4 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1228_122884


namespace NUMINAMATH_CALUDE_sum_in_second_quadrant_l1228_122823

/-- Given two complex numbers z₁ and z₂, prove that their sum is in the second quadrant -/
theorem sum_in_second_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = -3 + 4*I) (h₂ : z₂ = 2 - 3*I) : 
  let z := z₁ + z₂
  z.re < 0 ∧ z.im > 0 := by
  sorry

#check sum_in_second_quadrant

end NUMINAMATH_CALUDE_sum_in_second_quadrant_l1228_122823


namespace NUMINAMATH_CALUDE_joshua_toy_cars_l1228_122863

theorem joshua_toy_cars : 
  ∀ (box1 box2 box3 box4 box5 : ℕ),
    box1 = 21 →
    box2 = 31 →
    box3 = 19 →
    box4 = 45 →
    box5 = 27 →
    box1 + box2 + box3 + box4 + box5 = 143 :=
by
  sorry

end NUMINAMATH_CALUDE_joshua_toy_cars_l1228_122863


namespace NUMINAMATH_CALUDE_right_triangle_area_l1228_122836

theorem right_triangle_area (a c : ℝ) (h1 : a = 40) (h2 : c = 41) :
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1228_122836


namespace NUMINAMATH_CALUDE_peters_situps_l1228_122890

theorem peters_situps (greg_situps : ℕ) (ratio : ℚ) : 
  greg_situps = 32 →
  ratio = 3 / 4 →
  ∃ peter_situps : ℕ, peter_situps * 4 = greg_situps * 3 ∧ peter_situps = 24 :=
by sorry

end NUMINAMATH_CALUDE_peters_situps_l1228_122890


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1228_122846

theorem tan_alpha_value (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) :
  Real.tan α = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1228_122846


namespace NUMINAMATH_CALUDE_intersection_M_N_l1228_122873

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1228_122873


namespace NUMINAMATH_CALUDE_rationalize_sqrt3_plus_1_l1228_122809

theorem rationalize_sqrt3_plus_1 :
  (1 : ℝ) / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt3_plus_1_l1228_122809


namespace NUMINAMATH_CALUDE_runners_meet_again_l1228_122824

-- Define the track circumference
def track_circumference : ℝ := 400

-- Define the runners' speeds
def runner1_speed : ℝ := 5.0
def runner2_speed : ℝ := 5.5
def runner3_speed : ℝ := 6.0

-- Define the time when runners meet again
def meeting_time : ℝ := 800

-- Theorem statement
theorem runners_meet_again :
  ∀ (t : ℝ), t = meeting_time →
  (runner1_speed * t) % track_circumference = 
  (runner2_speed * t) % track_circumference ∧
  (runner2_speed * t) % track_circumference = 
  (runner3_speed * t) % track_circumference :=
by
  sorry

#check runners_meet_again

end NUMINAMATH_CALUDE_runners_meet_again_l1228_122824


namespace NUMINAMATH_CALUDE_y_value_proof_l1228_122886

theorem y_value_proof (y : ℚ) (h : 2/3 - 1/4 = 4/y) : y = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1228_122886


namespace NUMINAMATH_CALUDE_expression_value_l1228_122871

theorem expression_value : (19 + 43 / 151) * 151 = 2912 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1228_122871


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1228_122843

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1228_122843


namespace NUMINAMATH_CALUDE_simplify_expression_l1228_122896

theorem simplify_expression : 1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1228_122896


namespace NUMINAMATH_CALUDE_empty_vessel_possible_l1228_122895

/-- Represents a state of water distribution among three vessels --/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a pouring operation from one vessel to another --/
inductive PouringOperation
  | FromAToB
  | FromAToC
  | FromBToA
  | FromBToC
  | FromCToA
  | FromCToB

/-- Applies a pouring operation to a water state --/
def applyPouring (state : WaterState) (op : PouringOperation) : WaterState :=
  match op with
  | PouringOperation.FromAToB => 
      if state.a ≤ state.b then {a := 0, b := state.b + state.a, c := state.c}
      else {a := state.a - state.b, b := 2 * state.b, c := state.c}
  | PouringOperation.FromAToC => 
      if state.a ≤ state.c then {a := 0, b := state.b, c := state.c + state.a}
      else {a := state.a - state.c, b := state.b, c := 2 * state.c}
  | PouringOperation.FromBToA => 
      if state.b ≤ state.a then {a := state.a + state.b, b := 0, c := state.c}
      else {a := 2 * state.a, b := state.b - state.a, c := state.c}
  | PouringOperation.FromBToC => 
      if state.b ≤ state.c then {a := state.a, b := 0, c := state.c + state.b}
      else {a := state.a, b := state.b - state.c, c := 2 * state.c}
  | PouringOperation.FromCToA => 
      if state.c ≤ state.a then {a := state.a + state.c, b := state.b, c := 0}
      else {a := 2 * state.a, b := state.b, c := state.c - state.a}
  | PouringOperation.FromCToB => 
      if state.c ≤ state.b then {a := state.a, b := state.b + state.c, c := 0}
      else {a := state.a, b := 2 * state.b, c := state.c - state.b}

/-- Predicate to check if a water state has an empty vessel --/
def hasEmptyVessel (state : WaterState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem stating that it's always possible to empty a vessel --/
theorem empty_vessel_possible (initialState : WaterState) : 
  ∃ (operations : List PouringOperation), 
    hasEmptyVessel (operations.foldl applyPouring initialState) :=
  sorry

end NUMINAMATH_CALUDE_empty_vessel_possible_l1228_122895


namespace NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l1228_122885

/-- Given two curves C₁ and C₂, prove that if their tangent lines are perpendicular at x = 1, 
    then the parameter a of C₁ must equal -1 / (3e) -/
theorem perpendicular_tangents_imply_a_value (a : ℝ) :
  let C₁ : ℝ → ℝ := λ x => a * x^3 - x^2 + 2 * x
  let C₂ : ℝ → ℝ := λ x => Real.exp x
  let C₁' : ℝ → ℝ := λ x => 3 * a * x^2 - 2 * x + 2
  let C₂' : ℝ → ℝ := λ x => Real.exp x
  (C₁' 1 * C₂' 1 = -1) → a = -1 / (3 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l1228_122885


namespace NUMINAMATH_CALUDE_x_equals_5y_when_squared_difference_equal_l1228_122827

theorem x_equals_5y_when_squared_difference_equal
  (x y : ℕ) -- x and y are natural numbers
  (h : x^2 - 3*x = 25*y^2 - 15*y) -- given equation
  : x = 5*y := by
sorry

end NUMINAMATH_CALUDE_x_equals_5y_when_squared_difference_equal_l1228_122827


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1228_122870

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = c) →
  (C = π / 5) →
  (B = 3 * π / 10) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1228_122870


namespace NUMINAMATH_CALUDE_fried_green_tomatoes_l1228_122840

/-- Given that each tomato is cut into 8 slices and 20 tomatoes are needed to feed a family of 8 for a single meal, 
    prove that 20 slices are needed for a single person's meal. -/
theorem fried_green_tomatoes (slices_per_tomato : ℕ) (tomatoes_for_family : ℕ) (family_size : ℕ) 
  (h1 : slices_per_tomato = 8)
  (h2 : tomatoes_for_family = 20)
  (h3 : family_size = 8) :
  (slices_per_tomato * tomatoes_for_family) / family_size = 20 := by
  sorry

#check fried_green_tomatoes

end NUMINAMATH_CALUDE_fried_green_tomatoes_l1228_122840


namespace NUMINAMATH_CALUDE_max_value_xyz_l1228_122888

/-- Given real numbers x, y, and z that are non-negative and satisfy the equation
    2x + 3xy² + 2z = 36, the maximum value of x²y²z is 144. -/
theorem max_value_xyz (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
    (h_eq : 2*x + 3*x*y^2 + 2*z = 36) :
    x^2 * y^2 * z ≤ 144 :=
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l1228_122888


namespace NUMINAMATH_CALUDE_shortest_track_length_l1228_122810

theorem shortest_track_length (melanie_piece_length martin_piece_length : ℕ) 
  (h1 : melanie_piece_length = 8)
  (h2 : martin_piece_length = 20) :
  Nat.lcm melanie_piece_length martin_piece_length = 40 := by
sorry

end NUMINAMATH_CALUDE_shortest_track_length_l1228_122810


namespace NUMINAMATH_CALUDE_heather_oranges_l1228_122867

theorem heather_oranges (initial : Real) (received : Real) :
  initial = 60.0 → received = 35.0 → initial + received = 95.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_oranges_l1228_122867


namespace NUMINAMATH_CALUDE_monotone_sine_function_l1228_122813

/-- The function f(x) = x + t*sin(2x) is monotonically increasing on ℝ if and only if t ∈ [-1/2, 1/2] -/
theorem monotone_sine_function (t : ℝ) :
  (∀ x : ℝ, Monotone (λ x => x + t * Real.sin (2 * x))) ↔ t ∈ Set.Icc (-1/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_monotone_sine_function_l1228_122813


namespace NUMINAMATH_CALUDE_percentage_of_red_shirts_l1228_122892

theorem percentage_of_red_shirts 
  (total_students : ℕ) 
  (blue_percentage : ℚ) 
  (green_percentage : ℚ) 
  (other_colors : ℕ) 
  (h1 : total_students = 600) 
  (h2 : blue_percentage = 45/100) 
  (h3 : green_percentage = 15/100) 
  (h4 : other_colors = 102) :
  (total_students - (blue_percentage * total_students + green_percentage * total_students + other_colors)) / total_students = 23/100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_red_shirts_l1228_122892


namespace NUMINAMATH_CALUDE_combined_height_theorem_l1228_122899

/-- The conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Maria's height in inches -/
def maria_height_inches : ℝ := 54

/-- Ben's height in inches -/
def ben_height_inches : ℝ := 72

/-- Combined height in centimeters -/
def combined_height_cm : ℝ := (maria_height_inches + ben_height_inches) * inch_to_cm

theorem combined_height_theorem :
  combined_height_cm = 320.04 := by sorry

end NUMINAMATH_CALUDE_combined_height_theorem_l1228_122899


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l1228_122851

/-- An ellipse and a hyperbola are tangent if and only if m = 8/9 -/
theorem ellipse_hyperbola_tangent (m : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1 ∧ 
   ∀ x' y' : ℝ, x'^2 + 9*y'^2 = 9 ∧ x'^2 - m*(y'+3)^2 = 1 → (x', y') = (x, y)) ↔ 
  m = 8/9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangent_l1228_122851


namespace NUMINAMATH_CALUDE_mary_saturday_wage_l1228_122858

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  weekday_hours : Nat
  saturday_hours : Nat
  regular_weekly_earnings : Nat
  saturday_weekly_earnings : Nat

/-- Calculates Mary's Saturday hourly wage --/
def saturday_hourly_wage (schedule : WorkSchedule) : Rat :=
  let regular_hourly_wage := schedule.regular_weekly_earnings / schedule.weekday_hours
  let saturday_earnings := schedule.saturday_weekly_earnings - schedule.regular_weekly_earnings
  saturday_earnings / schedule.saturday_hours

/-- Mary's actual work schedule --/
def mary_schedule : WorkSchedule :=
  { weekday_hours := 37
  , saturday_hours := 4
  , regular_weekly_earnings := 407
  , saturday_weekly_earnings := 483 }

/-- Theorem stating that Mary's Saturday hourly wage is $19 --/
theorem mary_saturday_wage :
  saturday_hourly_wage mary_schedule = 19 := by
  sorry

end NUMINAMATH_CALUDE_mary_saturday_wage_l1228_122858


namespace NUMINAMATH_CALUDE_sum_of_roots_l1228_122898

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1228_122898


namespace NUMINAMATH_CALUDE_odd_function_periodic_l1228_122816

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_periodic 
  (f : ℝ → ℝ) 
  (h1 : IsOdd f) 
  (h2 : ∀ x, f x = f (2 - x)) : 
  IsPeriodic f 4 := by
sorry

end NUMINAMATH_CALUDE_odd_function_periodic_l1228_122816


namespace NUMINAMATH_CALUDE_shaded_area_l1228_122889

/-- The area of the shaded region in a grid with given properties -/
theorem shaded_area (total_area : ℝ) (triangle_base : ℝ) (triangle_height : ℝ)
  (h1 : total_area = 38)
  (h2 : triangle_base = 12)
  (h3 : triangle_height = 4) :
  total_area - (1/2 * triangle_base * triangle_height) = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_l1228_122889


namespace NUMINAMATH_CALUDE_haleys_concert_tickets_l1228_122822

theorem haleys_concert_tickets (ticket_price : ℕ) (extra_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 → extra_tickets = 5 → total_spent = 32 → 
  ∃ (tickets_for_friends : ℕ), 
    ticket_price * (tickets_for_friends + extra_tickets) = total_spent ∧ 
    tickets_for_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_haleys_concert_tickets_l1228_122822


namespace NUMINAMATH_CALUDE_stuffed_animals_count_l1228_122852

/-- The number of stuffed animals McKenna has -/
def mckenna_stuffed_animals : ℕ := 34

/-- The number of stuffed animals Kenley has -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- The number of stuffed animals Tenly has -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

theorem stuffed_animals_count : total_stuffed_animals = 175 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_count_l1228_122852


namespace NUMINAMATH_CALUDE_representations_non_negative_representations_natural_l1228_122882

/-- The number of ways to represent a natural number as a sum of non-negative integers -/
def representationsNonNegative (n m : ℕ) : ℕ := Nat.choose (n + m - 1) n

/-- The number of ways to represent a natural number as a sum of natural numbers -/
def representationsNatural (n m : ℕ) : ℕ := Nat.choose (n - 1) (n - m)

/-- Theorem stating the number of ways to represent n as a sum of m non-negative integers -/
theorem representations_non_negative (n m : ℕ) :
  representationsNonNegative n m = Nat.choose (n + m - 1) n := by sorry

/-- Theorem stating the number of ways to represent n as a sum of m natural numbers -/
theorem representations_natural (n m : ℕ) (h : m ≤ n) :
  representationsNatural n m = Nat.choose (n - 1) (n - m) := by sorry

end NUMINAMATH_CALUDE_representations_non_negative_representations_natural_l1228_122882


namespace NUMINAMATH_CALUDE_unfair_coin_expected_worth_l1228_122861

/-- An unfair coin with given probabilities for heads and tails, and corresponding gains and losses -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ
  prob_sum_one : prob_heads + prob_tails = 1
  prob_nonneg : prob_heads ≥ 0 ∧ prob_tails ≥ 0

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails)

/-- Theorem stating the expected worth of the specific unfair coin -/
theorem unfair_coin_expected_worth :
  ∃ (c : UnfairCoin),
    c.prob_heads = 2/3 ∧
    c.prob_tails = 1/3 ∧
    c.gain_heads = 5 ∧
    c.loss_tails = 6 ∧
    expected_worth c = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_worth_l1228_122861


namespace NUMINAMATH_CALUDE_arithmetic_iff_straight_line_l1228_122877

/-- A sequence of real numbers -/
def Sequence := ℕ+ → ℝ

/-- A sequence of points in 2D space -/
def PointSequence := ℕ+ → ℝ × ℝ

/-- Predicate for arithmetic sequences -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Predicate for points lying on a straight line -/
def on_straight_line (P : PointSequence) : Prop :=
  ∃ m b : ℝ, ∀ n : ℕ+, (P n).2 = m * (P n).1 + b

/-- Main theorem: equivalence between arithmetic sequence and points on a straight line -/
theorem arithmetic_iff_straight_line (a : Sequence) (P : PointSequence) :
  is_arithmetic a ↔ on_straight_line P :=
sorry

end NUMINAMATH_CALUDE_arithmetic_iff_straight_line_l1228_122877


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1228_122872

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  bridge_length = 660 →
  crossing_time = 54.995600351971845 →
  ∃ (speed : ℝ), abs (speed - 54.0036) < 0.0001 ∧ 
  speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1228_122872


namespace NUMINAMATH_CALUDE_street_number_painting_cost_l1228_122815

/-- Calculates the sum of digits for a given range of numbers in an arithmetic sequence -/
def sumDigits (start : ℕ) (diff : ℕ) (count : ℕ) : ℕ :=
  sorry

/-- Calculates the total cost of painting house numbers on a street -/
def totalCost (eastStart eastDiff westStart westDiff houseCount : ℕ) : ℕ :=
  sorry

theorem street_number_painting_cost :
  totalCost 5 5 2 4 25 = 88 :=
sorry

end NUMINAMATH_CALUDE_street_number_painting_cost_l1228_122815


namespace NUMINAMATH_CALUDE_expression_equals_six_l1228_122829

-- Define the expression
def expression : ℚ := 3 * (3 + 3) / 3

-- Theorem statement
theorem expression_equals_six : expression = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l1228_122829


namespace NUMINAMATH_CALUDE_cos_sum_min_value_l1228_122845

theorem cos_sum_min_value (x : ℝ) : |Real.cos x| + |Real.cos (2 * x)| ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_min_value_l1228_122845


namespace NUMINAMATH_CALUDE_plates_needed_is_38_l1228_122825

/-- The number of plates needed for a week given the specified eating patterns -/
def plates_needed : ℕ :=
  let days_with_son := 3
  let days_with_parents := 7 - days_with_son
  let people_with_son := 2
  let people_with_parents := 4
  let plates_per_person_with_son := 1
  let plates_per_person_with_parents := 2
  
  (days_with_son * people_with_son * plates_per_person_with_son) +
  (days_with_parents * people_with_parents * plates_per_person_with_parents)

theorem plates_needed_is_38 : plates_needed = 38 := by
  sorry

end NUMINAMATH_CALUDE_plates_needed_is_38_l1228_122825


namespace NUMINAMATH_CALUDE_confucius_wine_consumption_l1228_122860

theorem confucius_wine_consumption :
  let wine_sequence : List ℚ := [1, 1, 1/2, 1/4, 1/8, 1/16]
  List.sum wine_sequence = 47/16 := by
  sorry

end NUMINAMATH_CALUDE_confucius_wine_consumption_l1228_122860


namespace NUMINAMATH_CALUDE_frisbee_committee_formations_l1228_122859

def num_teams : Nat := 5
def team_size : Nat := 8
def host_committee_size : Nat := 4
def non_host_committee_size : Nat := 2

theorem frisbee_committee_formations :
  (num_teams * (Nat.choose team_size host_committee_size) *
   (Nat.choose team_size non_host_committee_size) ^ (num_teams - 1)) =
  215134600 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_committee_formations_l1228_122859


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1228_122802

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1228_122802


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1228_122819

theorem inequality_solution_set (x : ℝ) : 1 - 3 * (x - 1) < x ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1228_122819


namespace NUMINAMATH_CALUDE_change_in_expression_l1228_122821

theorem change_in_expression (x b : ℝ) (h : b > 0) :
  let f := fun t : ℝ => t^2 - 5*t + 2
  f (x + b) - f x = 2*b*x + b^2 - 5*b :=
by sorry

end NUMINAMATH_CALUDE_change_in_expression_l1228_122821


namespace NUMINAMATH_CALUDE_polynomial_identity_solutions_l1228_122807

variable (x : ℝ)

noncomputable def p (x : ℝ) : ℝ := x^2 + x + 1

theorem polynomial_identity_solutions :
  ∃! (q₁ q₂ : ℝ → ℝ), 
    (∀ x, q₁ x = x^2 + 2*x) ∧ 
    (∀ x, q₂ x = x^2 - 1) ∧ 
    (∀ q : ℝ → ℝ, (∀ x, (p x)^2 - 2*(p x)*(q x) + (q x)^2 - 4*(p x) + 3*(q x) + 3 = 0) → 
      (q = q₁ ∨ q = q₂)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_solutions_l1228_122807


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1228_122862

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define two different planes
variable (α β : Plane)
variable (h_diff : α ≠ β)

-- Define a line m in plane α
variable (m : Line)
variable (h_m_in_α : line_in_plane m α)

-- Theorem statement
theorem perp_planes_necessary_not_sufficient :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1228_122862


namespace NUMINAMATH_CALUDE_impossible_d_greater_than_c_l1228_122897

/-- A decreasing function on positive reals -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f y < f x

theorem impossible_d_greater_than_c
  (f : ℝ → ℝ) (a b c d : ℝ)
  (h_dec : DecreasingFunction f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_prod : f a * f b * f c < 0)
  (h_d : f d = 0) :
  ¬(d > c) := by
sorry

end NUMINAMATH_CALUDE_impossible_d_greater_than_c_l1228_122897
