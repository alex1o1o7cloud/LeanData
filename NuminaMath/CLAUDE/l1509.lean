import Mathlib

namespace NUMINAMATH_CALUDE_radical_simplification_l1509_150971

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (20 * p) * Real.sqrt (10 * p^3) * Real.sqrt (6 * p^4) * Real.sqrt (15 * p^5) = 20 * p^6 * Real.sqrt (15 * p) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l1509_150971


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l1509_150992

def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

theorem loss_percentage_calculation :
  (cost_price - selling_price) / cost_price * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l1509_150992


namespace NUMINAMATH_CALUDE_max_three_digit_divisible_by_15_existence_of_solution_l1509_150960

def is_valid_assignment (n a b c d e : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9 ∧ c ≥ 1 ∧ c ≤ 9 ∧ d ≥ 1 ∧ d ≤ 9 ∧ e ≥ 1 ∧ e ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  n / (a * b + c + d * e) = 15

theorem max_three_digit_divisible_by_15 :
  ∀ n a b c d e : ℕ,
    is_valid_assignment n a b c d e →
    n ≤ 975 :=
by sorry

theorem existence_of_solution :
  ∃ n a b c d e : ℕ,
    is_valid_assignment n a b c d e ∧
    n = 975 :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_divisible_by_15_existence_of_solution_l1509_150960


namespace NUMINAMATH_CALUDE_min_max_values_of_f_l1509_150983

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min_val) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = min_val) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = max_val) ∧
    min_val = -3 * Real.pi / 2 ∧
    max_val = Real.pi / 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_of_f_l1509_150983


namespace NUMINAMATH_CALUDE_line_equation_perpendicular_line_equation_opposite_intercepts_l1509_150918

-- Define the line l
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the perpendicular line
def perpendicularLine : Line := { a := 2, b := 1, c := 3 }

-- Define the condition for a line to pass through a point
def passesThrough (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the condition for two lines to be perpendicular
def isPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the condition for a line to have intercepts with opposite signs
def hasOppositeIntercepts (l : Line) : Prop :=
  (l.a * l.c < 0 ∧ l.b * l.c < 0) ∨ (l.a = 0 ∧ l.b ≠ 0) ∨ (l.a ≠ 0 ∧ l.b = 0)

theorem line_equation_perpendicular (l : Line) :
  passesThrough l P ∧ isPerpendicular l perpendicularLine →
  l = { a := 1, b := -2, c := -4 } :=
sorry

theorem line_equation_opposite_intercepts (l : Line) :
  passesThrough l P ∧ hasOppositeIntercepts l →
  (l = { a := 1, b := 2, c := 0 } ∨ l = { a := 1, b := -1, c := -3 }) :=
sorry

end NUMINAMATH_CALUDE_line_equation_perpendicular_line_equation_opposite_intercepts_l1509_150918


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1509_150986

def total_sum : ℝ := 2743
def second_part : ℝ := 1688
def second_rate : ℝ := 0.05
def first_time : ℝ := 8
def second_time : ℝ := 3

theorem interest_rate_calculation (first_rate : ℝ) : 
  (total_sum - second_part) * first_rate * first_time = second_part * second_rate * second_time →
  first_rate = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1509_150986


namespace NUMINAMATH_CALUDE_collinear_points_theorem_l1509_150998

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_collinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

theorem collinear_points_theorem 
  (e₁ e₂ : V) 
  (h_noncollinear : ¬ is_collinear e₁ e₂)
  (k : ℝ)
  (AB CB CD : V)
  (h_AB : AB = e₁ - k • e₂)
  (h_CB : CB = 2 • e₁ + e₂)
  (h_CD : CD = 3 • e₁ - e₂)
  (h_collinear : is_collinear AB (CD - CB)) :
  k = 2 := by sorry

end NUMINAMATH_CALUDE_collinear_points_theorem_l1509_150998


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1509_150965

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1509_150965


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l1509_150919

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 → Complex.abs (w^3 - 3*w - 2) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l1509_150919


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1509_150937

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((4 + 7 * i) / (4 - 7 * i) + (4 - 7 * i) / (4 + 7 * i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1509_150937


namespace NUMINAMATH_CALUDE_chess_piece_position_l1509_150975

/-- Represents a position on a chess board -/
structure ChessPosition :=
  (column : Nat)
  (row : Nat)

/-- Converts a ChessPosition to a pair of natural numbers -/
def ChessPosition.toPair (pos : ChessPosition) : Nat × Nat :=
  (pos.column, pos.row)

theorem chess_piece_position :
  let piece : ChessPosition := ⟨3, 7⟩
  ChessPosition.toPair piece = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_position_l1509_150975


namespace NUMINAMATH_CALUDE_triangle_side_b_l1509_150927

theorem triangle_side_b (a : ℝ) (A B : ℝ) (h1 : a = 5) (h2 : A = π/6) (h3 : Real.tan B = 3/4) :
  ∃ (b : ℝ), b = 6 ∧ (b / Real.sin B = a / Real.sin A) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_l1509_150927


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l1509_150904

theorem quadratic_roots_and_triangle (α β : ℝ) (p k : ℝ) : 
  α^2 - 10*α + 20 = 0 →
  β^2 - 10*β + 20 = 0 →
  p = α^2 + β^2 →
  k * Real.sqrt 3 = (p^2 / 36) * Real.sqrt 3 →
  p = 60 ∧ k = p^2 / 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l1509_150904


namespace NUMINAMATH_CALUDE_both_save_800_l1509_150914

/-- Represents the financial situation of Anand and Balu -/
structure FinancialSituation where
  anand_income : ℕ
  balu_income : ℕ
  anand_expenditure : ℕ
  balu_expenditure : ℕ

/-- Checks if the given financial situation satisfies the problem conditions -/
def satisfies_conditions (fs : FinancialSituation) : Prop :=
  fs.anand_income * 4 = fs.balu_income * 5 ∧
  fs.anand_expenditure * 2 = fs.balu_expenditure * 3 ∧
  fs.anand_income = 2000

/-- Calculates the savings for a person given their income and expenditure -/
def savings (income : ℕ) (expenditure : ℕ) : ℕ :=
  income - expenditure

/-- Theorem stating that both Anand and Balu save 800 each -/
theorem both_save_800 (fs : FinancialSituation) (h : satisfies_conditions fs) :
  savings fs.anand_income fs.anand_expenditure = 800 ∧
  savings fs.balu_income fs.balu_expenditure = 800 := by
  sorry


end NUMINAMATH_CALUDE_both_save_800_l1509_150914


namespace NUMINAMATH_CALUDE_inequality_holds_l1509_150988

theorem inequality_holds (x : ℝ) (h : 0 < x ∧ x < 1) : 0 < 1 - x^2 ∧ 1 - x^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1509_150988


namespace NUMINAMATH_CALUDE_prob_sum_ge_12_is_zero_l1509_150996

-- Define a uniform random variable on (0,1)
def uniform_01 : Type := {x : ℝ // 0 < x ∧ x < 1}

-- Define the sum of 5 such variables
def sum_5_uniform (X₁ X₂ X₃ X₄ X₅ : uniform_01) : ℝ :=
  X₁.val + X₂.val + X₃.val + X₄.val + X₅.val

-- State the theorem
theorem prob_sum_ge_12_is_zero :
  ∀ X₁ X₂ X₃ X₄ X₅ : uniform_01,
  sum_5_uniform X₁ X₂ X₃ X₄ X₅ < 12 :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_ge_12_is_zero_l1509_150996


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1509_150903

-- Problem 1
theorem problem_1 (a : ℤ) (h : a = -1) : (a + 3)^2 + (3 + a) * (3 - a) = 12 := by sorry

-- Problem 2
theorem problem_2 (x y : ℤ) (hx : x = 2) (hy : y = 3) : 
  (x - 2*y) * (x + 2*y) - (x + 2*y)^2 + 8*y^2 = -24 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1509_150903


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1509_150985

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1509_150985


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l1509_150970

/-- Represents a hyperbola with given asymptotes and passing through a specific point -/
def is_correct_hyperbola (a b : ℝ) : Prop :=
  -- The equation of the hyperbola
  (∀ x y : ℝ, (3 * y^2 / 4) - (x^2 / 3) = 1 ↔ b * y^2 - a * x^2 = a * b) ∧
  -- The asymptotes are y = ±(2/3)x
  (a / b = 3 / 2) ∧
  -- The hyperbola passes through the point (√6, 2)
  (3 * 2^2 / 4 - Real.sqrt 6^2 / 3 = 1)

/-- The standard equation of the hyperbola satisfies the given conditions -/
theorem hyperbola_equation_correct :
  ∃ a b : ℝ, is_correct_hyperbola a b :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_correct_l1509_150970


namespace NUMINAMATH_CALUDE_miles_trumpets_l1509_150909

-- Define the number of body parts (as per typical human attributes)
def hands : Nat := 2
def head : Nat := 1
def fingers : Nat := 10

-- Define the number of each instrument based on the conditions
def guitars : Nat := hands + 2
def trombones : Nat := head + 2
def french_horns : Nat := guitars - 1
def trumpets : Nat := fingers - 3

-- Define the total number of instruments
def total_instruments : Nat := 17

-- Theorem to prove
theorem miles_trumpets :
  guitars + trombones + french_horns + trumpets = total_instruments ∧ trumpets = 7 := by
  sorry

end NUMINAMATH_CALUDE_miles_trumpets_l1509_150909


namespace NUMINAMATH_CALUDE_PropB_implies_PropA_PropA_not_implies_PropB_A_necessary_not_sufficient_for_B_l1509_150907

/-- Proposition A: x ≠ 2 or y ≠ 3 -/
def PropA (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 3

/-- Proposition B: x + y ≠ 5 -/
def PropB (x y : ℝ) : Prop := x + y ≠ 5

/-- Proposition B implies Proposition A -/
theorem PropB_implies_PropA : ∀ x y : ℝ, PropB x y → PropA x y := by sorry

/-- Proposition A does not imply Proposition B -/
theorem PropA_not_implies_PropB : ¬(∀ x y : ℝ, PropA x y → PropB x y) := by sorry

/-- A is a necessary but not sufficient condition for B -/
theorem A_necessary_not_sufficient_for_B : 
  (∀ x y : ℝ, PropB x y → PropA x y) ∧ ¬(∀ x y : ℝ, PropA x y → PropB x y) := by sorry

end NUMINAMATH_CALUDE_PropB_implies_PropA_PropA_not_implies_PropB_A_necessary_not_sufficient_for_B_l1509_150907


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1509_150980

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1509_150980


namespace NUMINAMATH_CALUDE_emerie_dime_count_l1509_150999

/-- Represents the number of coins a person has -/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total number of coins a person has -/
def totalCoins (c : CoinCount) : ℕ := c.quarters + c.nickels + c.dimes

theorem emerie_dime_count 
  (zain_total : ℕ) 
  (emerie : CoinCount)
  (h1 : zain_total = 48)
  (h2 : emerie.quarters = 6)
  (h3 : emerie.nickels = 5)
  (h4 : totalCoins emerie + 10 = zain_total) :
  emerie.dimes = 27 := by
sorry

end NUMINAMATH_CALUDE_emerie_dime_count_l1509_150999


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1509_150968

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1509_150968


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1509_150981

theorem sum_of_solutions : ∃ (S : Finset (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ S ↔ (p.1 * p.2 = 6 * (p.1 + p.2) ∧ p.1 > 0 ∧ p.2 > 0)) ∧ 
  (S.sum (λ p => p.1 + p.2) = 290) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1509_150981


namespace NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l1509_150984

theorem housing_units_without_cable_or_vcr 
  (total : ℝ) 
  (cable : ℝ) 
  (vcr : ℝ) 
  (both : ℝ) 
  (h1 : cable = (1/5) * total) 
  (h2 : vcr = (1/10) * total) 
  (h3 : both = (1/3) * cable) :
  (total - (cable + vcr - both)) / total = 7/10 := by
sorry

end NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l1509_150984


namespace NUMINAMATH_CALUDE_triple_composition_identity_implies_identity_l1509_150944

theorem triple_composition_identity_implies_identity 
  (f : ℝ → ℝ) (hf : Continuous f) (h : ∀ x, f (f (f x)) = x) : 
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_identity_implies_identity_l1509_150944


namespace NUMINAMATH_CALUDE_parallelogram_sides_sum_l1509_150952

theorem parallelogram_sides_sum (x y : ℚ) : 
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -8/5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_sides_sum_l1509_150952


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1509_150913

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, Real.exp x ≤ 1)) ↔ (∃ x : ℝ, Real.exp x > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1509_150913


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1509_150906

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) :
  rp.faces + rp.edges + rp.vertices = 26 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1509_150906


namespace NUMINAMATH_CALUDE_derivative_sin_minus_x_cos_l1509_150982

theorem derivative_sin_minus_x_cos (x : ℝ) :
  deriv (λ x => Real.sin x - x * Real.cos x) x = x * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_minus_x_cos_l1509_150982


namespace NUMINAMATH_CALUDE_locus_is_hexagon_l1509_150991

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a function to check if a triangle is acute-angled
def isAcuteTriangle (t : Triangle3D) : Prop :=
  sorry

-- Define a function to check if a point forms acute-angled triangles with all sides of the base triangle
def formsAcuteTriangles (P : Point3D) (base : Triangle3D) : Prop :=
  sorry

-- Define the locus of points
def locusOfPoints (base : Triangle3D) : Set Point3D :=
  {P | formsAcuteTriangles P base}

-- Theorem statement
theorem locus_is_hexagon (base : Triangle3D) 
  (h : isAcuteTriangle base) : 
  ∃ (hexagon : Set Point3D), locusOfPoints base = hexagon :=
sorry

end NUMINAMATH_CALUDE_locus_is_hexagon_l1509_150991


namespace NUMINAMATH_CALUDE_small_cakes_per_hour_l1509_150932

-- Define the variables
def helpers : ℕ := 10
def hours : ℕ := 3
def large_cakes_needed : ℕ := 20
def small_cakes_needed : ℕ := 700
def large_cakes_per_hour : ℕ := 2

-- Define the theorem
theorem small_cakes_per_hour :
  ∃ (s : ℕ), 
    s * helpers * (hours - (large_cakes_needed / large_cakes_per_hour)) = small_cakes_needed ∧
    s = 35 := by
  sorry

end NUMINAMATH_CALUDE_small_cakes_per_hour_l1509_150932


namespace NUMINAMATH_CALUDE_min_value_fraction_l1509_150911

theorem min_value_fraction (x : ℝ) (h : x > 12) :
  x^2 / (x - 12) ≥ 48 ∧ (x^2 / (x - 12) = 48 ↔ x = 24) := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1509_150911


namespace NUMINAMATH_CALUDE_missing_sale_is_6088_l1509_150989

/-- Calculates the missing sale amount given the sales for five months and the average sale for six months. -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem stating that the missing sale amount is 6088 given the specific sales and average. -/
theorem missing_sale_is_6088 :
  calculate_missing_sale 5921 5468 5568 6433 5922 5900 = 6088 := by
  sorry

#eval calculate_missing_sale 5921 5468 5568 6433 5922 5900

end NUMINAMATH_CALUDE_missing_sale_is_6088_l1509_150989


namespace NUMINAMATH_CALUDE_alice_shopping_cost_l1509_150910

/-- Represents the shopping list and discounts --/
structure ShoppingTrip where
  apple_price : ℕ
  apple_quantity : ℕ
  bread_price : ℕ
  bread_quantity : ℕ
  cereal_price : ℕ
  cereal_quantity : ℕ
  cake_price : ℕ
  cheese_price : ℕ
  cereal_discount : ℕ
  bread_discount : Bool
  coupon_threshold : ℕ
  coupon_value : ℕ

/-- Calculates the total cost of the shopping trip --/
def calculate_total (trip : ShoppingTrip) : ℕ :=
  let apple_cost := trip.apple_price * trip.apple_quantity
  let bread_cost := if trip.bread_discount then trip.bread_price else trip.bread_price * trip.bread_quantity
  let cereal_cost := (trip.cereal_price - trip.cereal_discount) * trip.cereal_quantity
  let total := apple_cost + bread_cost + cereal_cost + trip.cake_price + trip.cheese_price
  if total ≥ trip.coupon_threshold then total - trip.coupon_value else total

/-- Theorem stating that Alice's shopping trip costs $38 --/
theorem alice_shopping_cost : 
  let trip : ShoppingTrip := {
    apple_price := 2,
    apple_quantity := 4,
    bread_price := 4,
    bread_quantity := 2,
    cereal_price := 5,
    cereal_quantity := 3,
    cake_price := 8,
    cheese_price := 6,
    cereal_discount := 1,
    bread_discount := true,
    coupon_threshold := 40,
    coupon_value := 10
  }
  calculate_total trip = 38 := by
sorry

end NUMINAMATH_CALUDE_alice_shopping_cost_l1509_150910


namespace NUMINAMATH_CALUDE_one_third_repeating_one_seventh_repeating_one_ninth_repeating_l1509_150940

def repeating_decimal (n : ℕ) (d : ℕ) (period : List ℕ) : ℚ :=
  (n : ℚ) / (d : ℚ)

theorem one_third_repeating : repeating_decimal 1 3 [3] = 1 / 3 := by sorry

theorem one_seventh_repeating : repeating_decimal 1 7 [1, 4, 2, 8, 5, 7] = 1 / 7 := by sorry

theorem one_ninth_repeating : repeating_decimal 1 9 [1] = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_one_third_repeating_one_seventh_repeating_one_ninth_repeating_l1509_150940


namespace NUMINAMATH_CALUDE_total_pumpkin_pies_l1509_150976

theorem total_pumpkin_pies (pinky helen emily : ℕ) 
  (h1 : pinky = 147) 
  (h2 : helen = 56) 
  (h3 : emily = 89) : 
  pinky + helen + emily = 292 := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkin_pies_l1509_150976


namespace NUMINAMATH_CALUDE_ab_greater_than_sum_l1509_150961

theorem ab_greater_than_sum (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_sum_l1509_150961


namespace NUMINAMATH_CALUDE_condition_relationship_l1509_150917

open Set

def condition_p (x : ℝ) : Prop := |x - 1| < 2
def condition_q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

def set_p : Set ℝ := {x | -1 < x ∧ x < 3}
def set_q : Set ℝ := {x | -1 < x ∧ x < 6}

theorem condition_relationship :
  (∀ x, condition_p x → x ∈ set_p) ∧
  (∀ x, condition_q x → x ∈ set_q) ∧
  set_p ⊂ set_q :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l1509_150917


namespace NUMINAMATH_CALUDE_special_function_properties_l1509_150925

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) - f y = (x + 2*y + 2) * x) ∧ (f 2 = 12)

theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (f 0 = 4) ∧
  (Set.Icc (-1 : ℝ) 5 = {a | ∃ x₀ ∈ Set.Ioo 1 4, f x₀ - 8 = a * x₀}) :=
sorry

end NUMINAMATH_CALUDE_special_function_properties_l1509_150925


namespace NUMINAMATH_CALUDE_equation_solution_l1509_150945

theorem equation_solution : ∃ x : ℝ, 45 * x = 0.6 * 900 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1509_150945


namespace NUMINAMATH_CALUDE_correct_both_problems_l1509_150936

theorem correct_both_problems (total : ℕ) (sets_correct : ℕ) (functions_correct : ℕ) (both_wrong : ℕ) : 
  total = 50 ∧ sets_correct = 40 ∧ functions_correct = 31 ∧ both_wrong = 4 →
  ∃ (both_correct : ℕ), both_correct = 29 ∧ 
    total = sets_correct + functions_correct - both_correct + both_wrong :=
by
  sorry


end NUMINAMATH_CALUDE_correct_both_problems_l1509_150936


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1509_150924

/-- Given plane vectors a, b, and c, if a + 2b is parallel to c, then the x-coordinate of c is -2. -/
theorem vector_parallel_condition (a b c : ℝ × ℝ) :
  a = (3, 1) →
  b = (-1, 1) →
  c.2 = -6 →
  (∃ (k : ℝ), k • (a + 2 • b) = c) →
  c.1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1509_150924


namespace NUMINAMATH_CALUDE_min_sum_mutually_exclusive_events_l1509_150931

theorem min_sum_mutually_exclusive_events (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hA : ℝ) (hB : ℝ) (h_mutually_exclusive : hA + hB = 1) 
  (h_prob_A : hA = 1 / y) (h_prob_B : hB = 4 / x) : 
  x + y ≥ 9 ∧ ∃ x y, x + y = 9 := by
sorry

end NUMINAMATH_CALUDE_min_sum_mutually_exclusive_events_l1509_150931


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_36_l1509_150962

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = x * 1000 + 410 + y

theorem four_digit_divisible_by_36 :
  ∀ n : ℕ, is_valid_number n ∧ n % 36 = 0 ↔ n = 2412 ∨ n = 7416 := by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_36_l1509_150962


namespace NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l1509_150916

theorem arctan_sum_of_cubic_roots (u v w : ℝ) : 
  u^3 - 10*u + 11 = 0 → 
  v^3 - 10*v + 11 = 0 → 
  w^3 - 10*w + 11 = 0 → 
  u + v + w = 0 →
  u*v + v*w + w*u = -10 →
  u*v*w = -11 →
  Real.arctan u + Real.arctan v + Real.arctan w = π/4 := by sorry

end NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l1509_150916


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2007_l1509_150926

theorem units_digit_17_pow_2007 : (17^2007 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2007_l1509_150926


namespace NUMINAMATH_CALUDE_c2h5cl_formed_equals_c2h6_used_l1509_150947

-- Define the chemical reaction
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2h5cl : ℝ
  hcl : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.c2h6 = r.cl2 ∧ r.c2h6 = r.c2h5cl ∧ r.c2h6 = r.hcl

-- Theorem statement
theorem c2h5cl_formed_equals_c2h6_used 
  (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : r.c2h6 = 3) 
  (h3 : r.c2h5cl = 3) : 
  r.c2h5cl = r.c2h6 := by
  sorry


end NUMINAMATH_CALUDE_c2h5cl_formed_equals_c2h6_used_l1509_150947


namespace NUMINAMATH_CALUDE_cone_volume_l1509_150941

/-- Given a cone with base area 2π and lateral area 6π, its volume is 8π/3 -/
theorem cone_volume (r : ℝ) (l : ℝ) (h : ℝ) : 
  (π * r^2 = 2*π) → 
  (π * r * l = 6*π) → 
  (h^2 + r^2 = l^2) →
  (1/3 * π * r^2 * h = 8*π/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1509_150941


namespace NUMINAMATH_CALUDE_property_price_reduction_l1509_150905

theorem property_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 5000)
  (h2 : final_price = 4050)
  (h3 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.1 ∧ initial_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_property_price_reduction_l1509_150905


namespace NUMINAMATH_CALUDE_average_of_new_sequence_eq_l1509_150954

/-- Given a positive integer a, this function returns the average of seven consecutive integers starting with a. -/
def average_of_seven (a : ℤ) : ℚ :=
  (7 * a + 21) / 7

/-- Given a positive integer a, this function returns the average of seven consecutive integers starting with the average of seven consecutive integers starting with a. -/
def average_of_new_sequence (a : ℤ) : ℚ :=
  let b := average_of_seven a
  (7 * ⌊b⌋ + 21) / 7

/-- Theorem stating that the average of the new sequence is equal to a + 6 -/
theorem average_of_new_sequence_eq (a : ℤ) (h : a > 0) : 
  average_of_new_sequence a = a + 6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_new_sequence_eq_l1509_150954


namespace NUMINAMATH_CALUDE_range_of_f_when_m_eq_1_solution_set_of_f_gt_3x_when_m_eq_neg_1_inequality_equivalence_l1509_150955

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| - m * |x - 2|

-- Theorem for the range of f(x) when m = 1
theorem range_of_f_when_m_eq_1 :
  Set.range (f 1) = Set.Icc (-3) 3 := by sorry

-- Theorem for the solution set of f(x) > 3x when m = -1
theorem solution_set_of_f_gt_3x_when_m_eq_neg_1 :
  {x : ℝ | f (-1) x > 3 * x} = Set.Iio 1 := by sorry

-- Additional helper theorem to show the equivalence of the inequality
theorem inequality_equivalence (x : ℝ) :
  f (-1) x > 3 * x ↔ |x + 1| + |x - 2| > 3 * x := by sorry

end NUMINAMATH_CALUDE_range_of_f_when_m_eq_1_solution_set_of_f_gt_3x_when_m_eq_neg_1_inequality_equivalence_l1509_150955


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l1509_150922

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l1509_150922


namespace NUMINAMATH_CALUDE_equation_solution_l1509_150901

theorem equation_solution :
  ∃ x : ℚ, x - 2 ≠ 0 ∧ (2 / (x - 2) = (1 + x) / (x - 2) + 1) ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1509_150901


namespace NUMINAMATH_CALUDE_range_of_a_and_m_l1509_150974

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- Define the theorem
theorem range_of_a_and_m (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_and_m_l1509_150974


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l1509_150997

/-- Given a point P at (3, 5), prove that the distance between P and its reflection over the y-axis is 6 -/
theorem distance_to_reflection_over_y_axis :
  let P : ℝ × ℝ := (3, 5)
  let P' : ℝ × ℝ := (-P.1, P.2)
  Real.sqrt ((P'.1 - P.1)^2 + (P'.2 - P.2)^2) = 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_y_axis_l1509_150997


namespace NUMINAMATH_CALUDE_divisibility_of_forms_l1509_150977

/-- Represents a six-digit number in the form ABCDEF --/
def SixDigitNumber (A B C D E F : ℕ) : ℕ := 
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F

/-- The form PQQPQQ --/
def FormA (P Q : ℕ) : ℕ := SixDigitNumber P Q Q P Q Q

/-- The form PQPQPQ --/
def FormB (P Q : ℕ) : ℕ := SixDigitNumber P Q P Q P Q

/-- The form QPQPQP --/
def FormC (P Q : ℕ) : ℕ := SixDigitNumber Q P Q P Q P

/-- The form PPPPPP --/
def FormD (P : ℕ) : ℕ := SixDigitNumber P P P P P P

/-- The form PPPQQQ --/
def FormE (P Q : ℕ) : ℕ := SixDigitNumber P P P Q Q Q

theorem divisibility_of_forms (P Q : ℕ) :
  (∃ (k : ℕ), FormA P Q = 7 * k) ∧
  (∃ (k : ℕ), FormB P Q = 7 * k) ∧
  (∃ (k : ℕ), FormC P Q = 7 * k) ∧
  (∃ (k : ℕ), FormD P = 7 * k) ∧
  ¬(∀ (P Q : ℕ), ∃ (k : ℕ), FormE P Q = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_forms_l1509_150977


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l1509_150963

/-- A rectangular box with dimensions l, w, and h, wrapped with a square sheet of paper -/
structure Box where
  l : ℝ  -- length
  w : ℝ  -- width
  h : ℝ  -- height
  l_gt_w : l > w

/-- The square sheet of wrapping paper -/
structure WrappingPaper where
  side : ℝ  -- side length of the square sheet

/-- The wrapping configuration -/
structure WrappingConfig (box : Box) (paper : WrappingPaper) where
  centered : Bool  -- box is centered on the paper
  vertices_on_midlines : Bool  -- vertices of longer side on paper midlines
  corners_meet_at_top : Bool  -- unoccupied corners meet at top center

theorem wrapping_paper_area (box : Box) (paper : WrappingPaper) 
    (config : WrappingConfig box paper) : paper.side^2 = 4 * box.l^2 := by
  sorry

#check wrapping_paper_area

end NUMINAMATH_CALUDE_wrapping_paper_area_l1509_150963


namespace NUMINAMATH_CALUDE_velocity_zero_at_two_l1509_150979

-- Define the displacement function
def s (t : ℝ) : ℝ := -2 * t^2 + 8 * t

-- Define the velocity function (derivative of displacement)
def v (t : ℝ) : ℝ := -4 * t + 8

-- Theorem: The time when velocity is 0 is equal to 2
theorem velocity_zero_at_two :
  ∃ t : ℝ, v t = 0 ∧ t = 2 :=
sorry

end NUMINAMATH_CALUDE_velocity_zero_at_two_l1509_150979


namespace NUMINAMATH_CALUDE_refrigerator_profit_percentage_l1509_150946

/-- Calculates the profit percentage for a refrigerator sale --/
theorem refrigerator_profit_percentage 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (transport_cost : ℝ) 
  (installation_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : discounted_price = 13500) 
  (h2 : discount_rate = 0.20) 
  (h3 : transport_cost = 125) 
  (h4 : installation_cost = 250) 
  (h5 : selling_price = 18975) : 
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 36.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_profit_percentage_l1509_150946


namespace NUMINAMATH_CALUDE_action_figure_cost_l1509_150994

theorem action_figure_cost (current : ℕ) (total : ℕ) (cost : ℕ) : 
  current = 7 → total = 16 → cost = 72 → 
  (cost : ℚ) / ((total : ℚ) - (current : ℚ)) = 8 := by sorry

end NUMINAMATH_CALUDE_action_figure_cost_l1509_150994


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1509_150978

theorem purely_imaginary_complex_number (m : ℝ) :
  (3 * m ^ 2 - 8 * m - 3 : ℂ) + (m ^ 2 - 4 * m + 3 : ℂ) * Complex.I = Complex.I * ((m ^ 2 - 4 * m + 3 : ℝ) : ℂ) →
  m = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1509_150978


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l1509_150953

theorem inequality_not_always_hold (a b : ℝ) (h : a > b) : 
  ¬ (∀ c : ℝ, a * c > b * c) := by
sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l1509_150953


namespace NUMINAMATH_CALUDE_john_needs_two_planks_l1509_150958

/-- The number of planks needed for a house wall, given the total number of nails and nails per plank. -/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) : ℕ :=
  total_nails / nails_per_plank

/-- Theorem stating that John needs 2 planks for the house wall. -/
theorem john_needs_two_planks :
  let total_nails : ℕ := 4
  let nails_per_plank : ℕ := 2
  planks_needed total_nails nails_per_plank = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_two_planks_l1509_150958


namespace NUMINAMATH_CALUDE_menelaus_condition_l1509_150938

-- Define the points
variable (A B C D P Q R S O : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define points on sides
def point_on_segment (P A B : Point) : Prop := sorry

-- Define intersection of lines
def lines_intersect (P R Q S O : Point) : Prop := sorry

-- Define quadrilateral with incircle
def has_incircle (A P O S : Point) : Prop := sorry

-- Define the ratio of segments
def segment_ratio (A P B : Point) : ℝ := sorry

-- Main theorem
theorem menelaus_condition 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_P : point_on_segment P A B)
  (h_Q : point_on_segment Q B C)
  (h_R : point_on_segment R C D)
  (h_S : point_on_segment S D A)
  (h_intersect : lines_intersect P R Q S O)
  (h_incircle1 : has_incircle A P O S)
  (h_incircle2 : has_incircle B Q O P)
  (h_incircle3 : has_incircle C R O Q)
  (h_incircle4 : has_incircle D S O R) :
  (segment_ratio A P B) * (segment_ratio B Q C) * 
  (segment_ratio C R D) * (segment_ratio D S A) = 1 := by
  sorry

end NUMINAMATH_CALUDE_menelaus_condition_l1509_150938


namespace NUMINAMATH_CALUDE_vasya_shirt_day_l1509_150942

structure TennisTournament where
  participants : ℕ
  days : ℕ
  matches_per_day : ℕ
  petya_shirt_day : ℕ
  petya_shirt_rank : ℕ
  vasya_shirt_rank : ℕ

def tournament : TennisTournament :=
  { participants := 20
  , days := 19
  , matches_per_day := 10
  , petya_shirt_day := 11
  , petya_shirt_rank := 11
  , vasya_shirt_rank := 15
  }

theorem vasya_shirt_day (t : TennisTournament) (h1 : t = tournament) :
  t.petya_shirt_day + (t.vasya_shirt_rank - t.petya_shirt_rank) = 15 :=
by
  sorry

#check vasya_shirt_day

end NUMINAMATH_CALUDE_vasya_shirt_day_l1509_150942


namespace NUMINAMATH_CALUDE_equation_solution_l1509_150939

theorem equation_solution (x : ℝ) (h : 1 - 9 / x + 9 / x^2 = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1509_150939


namespace NUMINAMATH_CALUDE_complex_calculation_l1509_150908

theorem complex_calculation : (1 - Complex.I)^2 - (4 + 2 * Complex.I) / (1 - 2 * Complex.I) = -4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l1509_150908


namespace NUMINAMATH_CALUDE_root_in_interval_l1509_150934

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ (k : ℤ), ∃ (x₀ : ℝ),
  x₀ > 0 ∧ 
  f x₀ = 0 ∧
  x₀ > k ∧ 
  x₀ < k + 1 ∧
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1509_150934


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_five_l1509_150987

theorem sum_of_squares_divisible_by_five (x y : ℤ) :
  (∃ n : ℤ, (x^2 + y^2) = 5*n) →
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_five_l1509_150987


namespace NUMINAMATH_CALUDE_feeding_sequences_count_l1509_150964

/-- Represents the number of animal pairs in the zoo -/
def num_pairs : Nat := 4

/-- Represents the constraint of alternating genders when feeding -/
def alternating_genders : Bool := true

/-- Represents the condition of starting with a specific male animal -/
def starts_with_male : Bool := true

/-- Calculates the number of possible feeding sequences -/
def feeding_sequences : Nat :=
  (num_pairs) * (num_pairs - 1) * (num_pairs - 1) * (num_pairs - 2) * (num_pairs - 2)

/-- Theorem stating that the number of possible feeding sequences is 144 -/
theorem feeding_sequences_count :
  alternating_genders ∧ starts_with_male → feeding_sequences = 144 := by
  sorry

end NUMINAMATH_CALUDE_feeding_sequences_count_l1509_150964


namespace NUMINAMATH_CALUDE_sandy_change_l1509_150920

/-- The change Sandy received from her purchase of toys -/
def change_received (football_price baseball_price paid : ℚ) : ℚ :=
  paid - (football_price + baseball_price)

/-- Theorem stating the correct change Sandy received -/
theorem sandy_change : change_received 9.14 6.81 20 = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_l1509_150920


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l1509_150902

/-- Given two intersecting circles with centers on the line x - y + c = 0 and
    intersection points A(1, 3) and B(m, -1), prove that m + c = -1 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (∃ (center1 center2 : ℝ × ℝ),
      center1 ∈ circle1 ∧ center2 ∈ circle2 ∧
      center1.1 - center1.2 + c = 0 ∧ center2.1 - center2.2 + c = 0) ∧
    (1, 3) ∈ circle1 ∩ circle2 ∧ (m, -1) ∈ circle1 ∩ circle2) →
  m + c = -1 := by
sorry

end NUMINAMATH_CALUDE_intersecting_circles_sum_l1509_150902


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1509_150930

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Points where a perpendicular from the right focus intersects the hyperbola -/
def intersection_points (h : Hyperbola a b) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The inscribed circle of a triangle -/
def inscribed_circle (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The eccentricity of the hyperbola is (1 + √5) / 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let (A, B) := intersection_points h
  let F₁ := left_focus h
  let (_, _, r) := inscribed_circle A B F₁
  r = a →
  eccentricity h = (1 + Real.sqrt 5) / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1509_150930


namespace NUMINAMATH_CALUDE_dodecahedron_path_count_l1509_150950

/-- Represents a face of the dodecahedron --/
inductive Face
  | Top
  | Bottom
  | UpperRing (n : Fin 5)
  | LowerRing (n : Fin 5)

/-- Represents a valid path on the dodecahedron --/
def ValidPath : List Face → Prop :=
  sorry

/-- The number of valid paths from top to bottom face --/
def numValidPaths : Nat :=
  sorry

/-- Theorem stating that the number of valid paths is 810 --/
theorem dodecahedron_path_count :
  numValidPaths = 810 :=
sorry

end NUMINAMATH_CALUDE_dodecahedron_path_count_l1509_150950


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1509_150957

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set S
def S : Set Nat := {1, 3}

-- Define set T
def T : Set Nat := {4}

-- Theorem statement
theorem complement_union_theorem :
  (Sᶜ ∪ T) = {2, 4} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1509_150957


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_31_l1509_150956

/-- Represents the pricing model of a caterer -/
structure CatererPricing where
  basic_fee : ℕ
  per_person : ℕ
  additional_fee : ℕ

/-- The first caterer's pricing model -/
def caterer1 : CatererPricing := {
  basic_fee := 150,
  per_person := 20,
  additional_fee := 0
}

/-- The second caterer's pricing model -/
def caterer2 : CatererPricing := {
  basic_fee := 250,
  per_person := 15,
  additional_fee := 50
}

/-- Calculate the total cost for a caterer given the number of people -/
def total_cost (c : CatererPricing) (people : ℕ) : ℕ :=
  c.basic_fee + c.per_person * people + c.additional_fee

/-- Theorem stating that 31 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_31 :
  (∀ n : ℕ, n < 31 → total_cost caterer1 n ≤ total_cost caterer2 n) ∧
  (total_cost caterer1 31 > total_cost caterer2 31) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_31_l1509_150956


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l1509_150912

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ) 
  (initially_tagged : ℕ) 
  (second_catch : ℕ) 
  (h1 : total_fish = 1500) 
  (h2 : initially_tagged = 60) 
  (h3 : second_catch = 50) :
  (initially_tagged : ℚ) / total_fish * second_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l1509_150912


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l1509_150990

/-- Given a line intersecting y = x^2 at x₁ and x₂, and the x-axis at x₃ (all non-zero),
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem line_parabola_intersection (x₁ x₂ x₃ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hx₃ : x₃ ≠ 0)
  (h_parabola : ∃ (a b : ℝ), x₁^2 = a*x₁ + b ∧ x₂^2 = a*x₂ + b)
  (h_x_axis : ∃ (a b : ℝ), 0 = a*x₃ + b ∧ (x₁^2 = a*x₁ + b ∨ x₂^2 = a*x₁ + b)) :
  1/x₁ + 1/x₂ = 1/x₃ := by sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l1509_150990


namespace NUMINAMATH_CALUDE_max_distance_between_inscribed_squares_max_distance_is_5_sqrt_2_l1509_150995

/-- The maximum distance between vertices of two squares, where the smaller square
    (perimeter 24) is inscribed in the larger square (perimeter 32) and rotated such that
    one of its vertices lies on the midpoint of one side of the larger square. -/
theorem max_distance_between_inscribed_squares : ℝ :=
  let inner_perimeter : ℝ := 24
  let outer_perimeter : ℝ := 32
  let inner_side : ℝ := inner_perimeter / 4
  let outer_side : ℝ := outer_perimeter / 4
  5 * Real.sqrt 2

/-- Proof that the maximum distance between vertices of the inscribed squares is 5√2. -/
theorem max_distance_is_5_sqrt_2 (inner_perimeter outer_perimeter : ℝ)
  (h1 : inner_perimeter = 24)
  (h2 : outer_perimeter = 32)
  (h3 : ∃ (v : ℝ × ℝ), v.1 = outer_perimeter / 8 ∧ v.2 = 0) :
  max_distance_between_inscribed_squares = 5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_inscribed_squares_max_distance_is_5_sqrt_2_l1509_150995


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1509_150935

theorem other_root_of_quadratic (c : ℝ) : 
  (3 : ℝ) ∈ {x : ℝ | x^2 - 5*x + c = 0} → 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5*x + c = 0 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1509_150935


namespace NUMINAMATH_CALUDE_calculate_expression_l1509_150948

theorem calculate_expression : (-3)^0 + Real.sqrt 8 + (-3)^2 - 4 * (Real.sqrt 2 / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1509_150948


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1509_150929

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (((a : ℂ) - Complex.I) / (1 + Complex.I)).re = 0 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1509_150929


namespace NUMINAMATH_CALUDE_percentage_commutation_l1509_150921

theorem percentage_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 48) : 
  0.4 * (0.3 * x) = 48 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l1509_150921


namespace NUMINAMATH_CALUDE_fourth_sunday_january_l1509_150923

-- Define the year N
def N : ℕ := sorry

-- Define the day of the week as an enumeration
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to determine if a year is a leap year
def isLeapYear (year : ℕ) : Bool := sorry

-- Define a function to get the next day of the week
def nextDay (day : DayOfWeek) : DayOfWeek := sorry

-- Define a function to add days to a given day of the week
def addDays (start : DayOfWeek) (days : ℕ) : DayOfWeek := sorry

-- State the theorem
theorem fourth_sunday_january (h1 : 2000 < N ∧ N < 2100)
  (h2 : addDays DayOfWeek.Tuesday 364 = DayOfWeek.Tuesday)
  (h3 : addDays (nextDay (addDays DayOfWeek.Tuesday 364)) 730 = DayOfWeek.Friday)
  : addDays DayOfWeek.Saturday 22 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_fourth_sunday_january_l1509_150923


namespace NUMINAMATH_CALUDE_nested_sqrt_evaluation_l1509_150969

theorem nested_sqrt_evaluation :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_evaluation_l1509_150969


namespace NUMINAMATH_CALUDE_inequality_proof_l1509_150915

theorem inequality_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a + b < b + c) ∧ (a / (a + b) < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1509_150915


namespace NUMINAMATH_CALUDE_fraction_division_l1509_150973

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l1509_150973


namespace NUMINAMATH_CALUDE_mixed_yellow_ratio_is_quarter_l1509_150966

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the total number of yellow jelly beans in a bag -/
def yellow_count (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellow_ratio

/-- Calculates the ratio of yellow jelly beans to total jelly beans when multiple bags are mixed -/
def mixed_yellow_ratio (bags : List JellyBeanBag) : ℚ :=
  let total_yellow := bags.map yellow_count |>.sum
  let total_beans := bags.map (·.total) |>.sum
  total_yellow / total_beans

theorem mixed_yellow_ratio_is_quarter (bags : List JellyBeanBag) :
  bags = [
    ⟨24, 2/5⟩,
    ⟨30, 3/10⟩,
    ⟨32, 1/4⟩,
    ⟨34, 1/10⟩
  ] →
  mixed_yellow_ratio bags = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_mixed_yellow_ratio_is_quarter_l1509_150966


namespace NUMINAMATH_CALUDE_diana_statues_l1509_150993

/-- Given the amount of paint available and the amount required per statue, 
    calculate the number of statues that can be painted. -/
def statues_paintable (paint_available : ℚ) (paint_per_statue : ℚ) : ℚ :=
  paint_available / paint_per_statue

/-- Theorem: Diana can paint 2 statues with the remaining paint. -/
theorem diana_statues : 
  let paint_available : ℚ := 1/2
  let paint_per_statue : ℚ := 1/4
  statues_paintable paint_available paint_per_statue = 2 := by
  sorry

end NUMINAMATH_CALUDE_diana_statues_l1509_150993


namespace NUMINAMATH_CALUDE_circle_center_sum_l1509_150972

/-- Given a circle with equation x^2 + y^2 = 4x - 12y - 8, 
    the sum of the coordinates of its center is -4. -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 12*y - 8) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 8) ∧ h + k = -4) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1509_150972


namespace NUMINAMATH_CALUDE_complex_vector_relation_l1509_150900

theorem complex_vector_relation (z₁ z₂ z₃ : ℂ) (x y : ℝ)
  (h₁ : z₁ = -1 + 2 * Complex.I)
  (h₂ : z₂ = 1 - Complex.I)
  (h₃ : z₃ = 3 - 2 * Complex.I)
  (h₄ : z₃ = x • z₁ + y • z₂) :
  x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_relation_l1509_150900


namespace NUMINAMATH_CALUDE_smaller_square_area_l1509_150967

theorem smaller_square_area (larger_square_area : ℝ) 
  (h1 : larger_square_area = 144) 
  (h2 : ∀ (side : ℝ), side * side = larger_square_area → 
        ∃ (smaller_side : ℝ), smaller_side = side / 2) : 
  ∃ (smaller_area : ℝ), smaller_area = 72 := by
sorry

end NUMINAMATH_CALUDE_smaller_square_area_l1509_150967


namespace NUMINAMATH_CALUDE_no_real_roots_for_specific_k_l1509_150943

theorem no_real_roots_for_specific_k : ∀ x : ℝ, x^2 + 2*x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_specific_k_l1509_150943


namespace NUMINAMATH_CALUDE_shelly_friends_in_classes_l1509_150933

/-- The number of friends Shelly made in classes -/
def friends_in_classes : ℕ := sorry

/-- The number of friends Shelly made in after-school clubs -/
def friends_in_clubs : ℕ := sorry

/-- The amount of thread needed for each keychain in inches -/
def thread_per_keychain : ℕ := 12

/-- The total amount of thread needed in inches -/
def total_thread : ℕ := 108

/-- Theorem stating that Shelly made 6 friends in classes -/
theorem shelly_friends_in_classes : 
  friends_in_classes = 6 ∧
  friends_in_clubs = friends_in_classes / 2 ∧
  friends_in_classes * thread_per_keychain + friends_in_clubs * thread_per_keychain = total_thread :=
sorry

end NUMINAMATH_CALUDE_shelly_friends_in_classes_l1509_150933


namespace NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l1509_150951

/-- The phase shift of a sine function y = a * sin(bx - c) is c/b to the right when c is positive. -/
theorem sine_phase_shift (a b c : ℝ) (h : c > 0) :
  let f := fun x => a * Real.sin (b * x - c)
  let phase_shift := c / b
  (∀ x, f (x + phase_shift) = a * Real.sin (b * x)) :=
sorry

/-- The phase shift of y = 3 * sin(3x - π/4) is π/12 to the right. -/
theorem specific_sine_phase_shift :
  let f := fun x => 3 * Real.sin (3 * x - π/4)
  let phase_shift := π/12
  (∀ x, f (x + phase_shift) = 3 * Real.sin (3 * x)) :=
sorry

end NUMINAMATH_CALUDE_sine_phase_shift_specific_sine_phase_shift_l1509_150951


namespace NUMINAMATH_CALUDE_four_row_grid_has_27_triangles_l1509_150928

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid :=
  (rows : ℕ)

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows * (grid.rows + 1)) / 2

/-- Counts the number of medium triangles in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid) : ℕ :=
  ((grid.rows - 1) * grid.rows) / 2

/-- Counts the number of large triangles in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid) : ℕ :=
  if grid.rows ≥ 3 then 1 else 0

/-- Counts the total number of triangles in a triangular grid -/
def countTotalTriangles (grid : TriangularGrid) : ℕ :=
  countSmallTriangles grid + countMediumTriangles grid + countLargeTriangles grid

/-- Theorem: A triangular grid with 4 rows contains 27 triangles in total -/
theorem four_row_grid_has_27_triangles :
  countTotalTriangles (TriangularGrid.mk 4) = 27 := by
  sorry

end NUMINAMATH_CALUDE_four_row_grid_has_27_triangles_l1509_150928


namespace NUMINAMATH_CALUDE_square_root_equality_l1509_150949

theorem square_root_equality (x : ℝ) (a : ℝ) 
  (h_pos : x > 0) 
  (h1 : Real.sqrt x = 2 * a - 3) 
  (h2 : Real.sqrt x = 5 - a) : 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_square_root_equality_l1509_150949


namespace NUMINAMATH_CALUDE_sum_bounds_l1509_150959

theorem sum_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b + 1/a + 1/b = 5) : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l1509_150959
