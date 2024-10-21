import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l326_32683

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (4^x + 1)

theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = -1/2) ∧
  (∀ x y, x < y → f a x > f a y) ∧
  (∀ k, (∀ t, f a (t^2 - 2*t) + f a (2*t^2 - k) < 0) → k < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l326_32683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l326_32604

theorem negation_of_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l326_32604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l326_32672

/-- The ellipse C2 obtained from scaling the unit circle -/
def C2 (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- The line l in Cartesian coordinates -/
def l (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- The distance function from a point to a line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + 2*y - 10| / Real.sqrt 5

/-- The theorem stating that (9/5, 8/5) on C2 has minimum distance to l -/
theorem min_distance_point :
  C2 (9/5) (8/5) ∧
  (∀ x y : ℝ, C2 x y → distance_to_line x y ≥ Real.sqrt 5) ∧
  distance_to_line (9/5) (8/5) = Real.sqrt 5 := by
  sorry

#check min_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l326_32672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l326_32687

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2 - 1 / (a - x)^2 else 0

-- State the theorem
theorem k_range (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧
    ∀ x : ℝ, x > 0 → (k^2 * x) + f a (1/4 * x + 1) > 0) ∧
  (∀ k : ℝ, k ≠ 0 →
    ∀ x : ℝ, x > 0 → (k^2 * x) + f a (1/4 * x + 1) > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l326_32687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l326_32685

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → f (-x) = f x) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → f (1/x) = -f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l326_32685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l326_32615

/-- The angle between the asymptotes of a hyperbola with equation x^2/a^2 - y^2 = 1,
    where its left focus coincides with the focus of the parabola y^2 = -8x -/
theorem hyperbola_asymptote_angle (a : ℝ) : 
  (∃ (x y : ℝ), y^2 = -8*x ∧ x^2/a^2 - y^2 = 1 ∧ x = -2) →
  Real.arctan (2 / Real.sqrt 3) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_angle_l326_32615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_constraint_max_value_attained_l326_32611

theorem max_value_constraint (a b c : ℝ) : 
  9*a^2 + 4*b^2 + 25*c^2 = 1 → 8*a + 3*b + 5*c ≤ Real.sqrt 373 / 6 :=
by sorry

theorem max_value_attained : 
  ∃ a b c : ℝ, 9*a^2 + 4*b^2 + 25*c^2 = 1 ∧ 8*a + 3*b + 5*c = Real.sqrt 373 / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_constraint_max_value_attained_l326_32611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roses_more_expensive_than_three_carnations_l326_32664

variable (x y : ℝ)

def rose_price := x
def carnation_price := y

-- Condition 1
def condition1 (x y : ℝ) : Prop := 6 * x + 3 * y > 24

-- Condition 2
def condition2 (x y : ℝ) : Prop := 4 * x + 5 * y < 22

-- Theorem to prove
theorem two_roses_more_expensive_than_three_carnations
  (h1 : condition1 x y) (h2 : condition2 x y) :
  2 * x > 3 * y := by
  sorry

#check two_roses_more_expensive_than_three_carnations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roses_more_expensive_than_three_carnations_l326_32664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_is_three_fifths_l326_32601

def OA : ℝ × ℝ := (1, 1)
def OB : ℝ × ℝ := (4, 1)
def OC : ℝ × ℝ := (4, 5)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
def AC : ℝ × ℝ := (OC.1 - OA.1, OC.2 - OA.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem angle_cosine_is_three_fifths :
  dot_product AB AC / (magnitude AB * magnitude AC) = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_is_three_fifths_l326_32601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_triple_l326_32680

/-- Cylinder volume calculation -/
noncomputable def cylinderVolume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

/-- Original cylinder dimensions -/
def originalRadius : ℝ := 8
def originalHeight : ℝ := 7

/-- New cylinder dimensions -/
def newRadius : ℝ := 8
def newHeight : ℝ := 21

theorem volume_triple :
  cylinderVolume newRadius newHeight = 3 * cylinderVolume originalRadius originalHeight := by
  sorry

#check volume_triple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_triple_l326_32680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l326_32625

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given conditions for the triangle -/
class TriangleConditions (t : Triangle) where
  cos_sin_relation : (Real.cos t.A / t.a) + (Real.cos t.B / t.b) = (Real.sin t.C / t.c)
  side_relation : t.b^2 + t.c^2 - t.a^2 = (6/5) * t.b * t.c

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) [TriangleConditions t] :
  Real.sin t.A * Real.sin t.B = Real.sin t.C ∧ Real.tan t.B = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l326_32625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_share_theorem_l326_32676

/-- The total number of jelly beans -/
def total_jelly_beans : ℕ := 200

/-- The number of jelly beans Emmanuel gets -/
def emmanuel_jelly_beans : ℕ := 100

/-- The ratio of Barry's share to Emmanuel's share -/
def barry_emmanuel_ratio : ℚ := 4/5

/-- Thomas's share of jelly beans as a percentage -/
def thomas_share_percent : ℚ := 10

theorem thomas_share_theorem :
  let barry_jelly_beans := emmanuel_jelly_beans * barry_emmanuel_ratio.num / barry_emmanuel_ratio.den
  let thomas_jelly_beans := total_jelly_beans - (barry_jelly_beans + emmanuel_jelly_beans)
  (thomas_jelly_beans : ℚ) / total_jelly_beans * 100 = thomas_share_percent := by
  sorry

#eval thomas_share_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_share_theorem_l326_32676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jade_rabbit_max_profit_l326_32621

/-- The fixed cost for producing "Jade Rabbit" -/
noncomputable def fixed_cost : ℝ := 20000

/-- The variable cost per item for producing "Jade Rabbit" -/
noncomputable def variable_cost : ℝ := 100

/-- The total revenue function for "Jade Rabbit" production -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else if x > 400 then 80000
  else 0

/-- The profit function for "Jade Rabbit" production -/
noncomputable def f (x : ℝ) : ℝ := R x - variable_cost * x - fixed_cost

/-- Theorem stating the maximum profit and corresponding production quantity -/
theorem jade_rabbit_max_profit :
  ∃ (x_max : ℝ), x_max = 300 ∧ f x_max = 25000 ∧ ∀ x, f x ≤ f x_max := by
  sorry

#check jade_rabbit_max_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jade_rabbit_max_profit_l326_32621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selling_price_and_max_profit_l326_32630

/-- Annual sales profit function -/
noncomputable def annual_sales_profit (x : ℝ) : ℝ := -1/3 * x^3 + 4 * x^2 + 9 * x - 126

/-- Condition on selling price -/
def selling_price_condition (x : ℝ) : Prop := x > 6

/-- Annual sales volume function -/
noncomputable def annual_sales_volume (x : ℝ) : ℝ := -1/3 * x^2 + 2 * x + 21

/-- Theorem stating the optimal selling price and maximum annual sales profit -/
theorem optimal_selling_price_and_max_profit :
  ∃ (x : ℝ),
    selling_price_condition x ∧
    annual_sales_volume 10 = 23/3 ∧
    (∀ (y : ℝ), selling_price_condition y → annual_sales_profit x ≥ annual_sales_profit y) ∧
    x = 9 ∧
    annual_sales_profit x = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selling_price_and_max_profit_l326_32630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_sufficient_not_necessary_l326_32648

open Set
open Function
open Real

variable (a b : ℝ) (hab : a < b)
variable (f : ℝ → ℝ)

def StrictlyIncreasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

def HasPositiveDerivative (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ x ∈ s, DifferentiableAt ℝ f x ∧ deriv f x > 0

theorem positive_derivative_sufficient_not_necessary :
  (HasPositiveDerivative f (Ioo a b) → StrictlyIncreasing f (Ioo a b)) ∧
  ¬(StrictlyIncreasing f (Ioo a b) → HasPositiveDerivative f (Ioo a b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_derivative_sufficient_not_necessary_l326_32648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_squared_plus_sin_l326_32658

theorem min_value_cos_squared_plus_sin :
  ∀ x : ℝ, (Real.cos x) ^ 2 + Real.sin x ≥ -1 ∧ ∃ y : ℝ, (Real.cos y) ^ 2 + Real.sin y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_squared_plus_sin_l326_32658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_division_l326_32605

theorem factorial_division :
  (Nat.factorial 9 : ℕ) = 362880 →
  (Nat.factorial 9 : ℕ) / (Nat.factorial 4 : ℕ) = 15120 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_division_l326_32605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_is_incorrect_correct_approach_differs_l326_32610

-- Define the concepts
def cultural_industry : Prop := sorry
def public_welfare_cultural_undertakings : Prop := sorry
def guarantee_basic_cultural_rights : Prop := sorry

-- Define the statement in question
def statement : Prop := cultural_industry ∧ guarantee_basic_cultural_rights

-- Define the correct approach
def correct_approach : Prop := public_welfare_cultural_undertakings ∧ guarantee_basic_cultural_rights

-- Theorem: The statement is incorrect
theorem statement_is_incorrect : ¬statement :=
by
  sorry

-- Theorem: The correct approach is different from the statement
theorem correct_approach_differs : correct_approach ≠ statement :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_is_incorrect_correct_approach_differs_l326_32610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_l326_32665

-- Define the radii of the circles
noncomputable def small_radius : ℝ := sorry
noncomputable def large_radius : ℝ := sorry

-- Define the condition that the area between circles is four times the area of the smaller circle
def area_condition : Prop :=
  Real.pi * large_radius^2 - Real.pi * small_radius^2 = 4 * (Real.pi * small_radius^2)

-- Theorem to prove
theorem circle_ratio : area_condition → small_radius / large_radius = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_l326_32665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_theorem_l326_32641

/-- Represents the annual interest rate as a percentage -/
noncomputable def interest_rate : ℝ := 15

/-- Represents the time period for the first deposit in years -/
noncomputable def time_period1 : ℝ := 3.5

/-- Represents the time period for the second deposit in years -/
noncomputable def time_period2 : ℝ := 10

/-- Represents the difference in interest earned between the two deposits -/
noncomputable def interest_difference : ℝ := 144

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Theorem stating that the principal amount is approximately 147.69 -/
theorem principal_amount_theorem (P : ℝ) : 
  simple_interest P interest_rate time_period2 - simple_interest P interest_rate time_period1 = interest_difference →
  ∃ ε > 0, |P - 147.69| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_theorem_l326_32641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l326_32653

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- State the theorem
theorem intersection_values (m n : ℝ) :
  A ∩ B m = Set.Ioo (-1) n → m = -1 ∧ n = 1 := by
  sorry

#check intersection_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l326_32653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_range_l326_32654

theorem cos_equation_solution_range (t : ℝ) : 
  t ∈ Set.Icc 0 Real.pi ∧ 
  (∃ x, Real.cos (x + t) = 1 - Real.cos x) ↔ 
  t ∈ Set.Icc 0 ((2 * Real.pi) / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_range_l326_32654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_are_similar_l326_32689

/-- Definition of a Triangle -/
structure Triangle :=
  (angle : Fin 3 → ℝ)
  (side : Fin 3 → ℝ)

/-- Definition of similarity for triangles -/
def are_similar (t1 t2 : Triangle) : Prop :=
  (∀ i : Fin 3, t1.angle i = t2.angle i) ∧ 
  ∃ k > 0, ∀ i : Fin 3, t1.side i = k * t2.side i

/-- Definition of an equilateral triangle -/
def is_equilateral (t : Triangle) : Prop :=
  (∀ i : Fin 3, t.angle i = 60) ∧ (∀ i j : Fin 3, t.side i = t.side j)

/-- Theorem: Any two equilateral triangles are similar -/
theorem equilateral_triangles_are_similar (t1 t2 : Triangle) 
  (h1 : is_equilateral t1) (h2 : is_equilateral t2) : 
  are_similar t1 t2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_are_similar_l326_32689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l326_32636

def sequence_a : ℕ → ℤ
  | 0 => 0
  | n + 1 => -(abs (sequence_a n + n.succ))

theorem a_2017_value : sequence_a 2017 = -1008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l326_32636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_white_balls_l326_32682

theorem estimate_white_balls (total_balls : ℕ) (total_draws : ℕ) (white_draws : ℕ) 
  (h1 : total_balls = 20) 
  (h2 : total_draws = 100) 
  (h3 : white_draws = 40) : 
  Int.floor ((total_balls : ℝ) * (white_draws : ℝ) / (total_draws : ℝ)) = 8 := by
  sorry

#check estimate_white_balls

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_white_balls_l326_32682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l326_32649

noncomputable def P : ℝ × ℝ := (1, 2)
noncomputable def Q : ℝ × ℝ := (3, 6)
noncomputable def R : ℝ × ℝ := (6, 3)
noncomputable def S : ℝ × ℝ := (8, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter : ℝ := distance P Q + distance Q R + distance R S + distance S P

theorem quadrilateral_perimeter :
  perimeter = 2 * Real.sqrt 5 + 10 * Real.sqrt 2 ∧
  (2 : ℝ) + 10 = 12 := by
  sorry

#eval (2 : ℝ) + 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l326_32649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l326_32619

-- Define the parabola function
noncomputable def f (x : ℝ) : ℝ := -1/3 * (x - 2)^2 + 1

-- Theorem statement
theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → x = y ∨ f ((x + y) / 2) > f x) ∧ 
  (f 2 = 1 ∧ ∀ x : ℝ, f x ≤ f 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) :=
by sorry

#check parabola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l326_32619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_axis_lengths_l326_32659

/-- Represents a hyperbola with equation x²/m - y² = 1 -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / m - y^2 = 1

/-- The length of the real axis of a hyperbola -/
noncomputable def real_axis_length (m : ℝ) (h : Hyperbola m) : ℝ := 2 * Real.sqrt m

/-- The length of the imaginary axis of a hyperbola -/
def imaginary_axis_length (m : ℝ) (h : Hyperbola m) : ℝ := 2

theorem hyperbola_axis_lengths 
  (m : ℝ) 
  (h : Hyperbola m) 
  (h_axis_ratio : real_axis_length m h = 3 * imaginary_axis_length m h) :
  real_axis_length m h = 6 ∧ imaginary_axis_length m h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_axis_lengths_l326_32659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_35_pencils_l326_32628

/-- The cost of a set of pencils is directly proportional to the number of pencils it contains. -/
def price_proportional_to_quantity (n : ℕ) (price : ℚ) : Prop :=
  price / n = (9 : ℚ) / 15

/-- The cost of 35 pencils is $21. -/
theorem cost_of_35_pencils : 
  ∃ (price : ℚ), price_proportional_to_quantity 35 price ∧ price = 21 := by
  use 21
  constructor
  · -- Prove that the price is proportional
    simp [price_proportional_to_quantity]
    norm_num
  · -- Prove that the price is 21
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_35_pencils_l326_32628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_complex_numbers_l326_32617

def g (z : ℂ) : ℂ := z^2 - Complex.I * z + 2

def satisfies_conditions (z : ℂ) : Prop :=
  Complex.im z < 0 ∧
  ∃ a b : ℤ, g z = ↑a + ↑b * Complex.I ∧ 
  abs a ≤ 5 ∧ abs b ≤ 5

theorem count_complex_numbers : 
  ∃! (S : Finset ℂ), (∀ z ∈ S, satisfies_conditions z) ∧ S.card = 72 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_complex_numbers_l326_32617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l326_32693

/-- The function f(x) = 1 / (x - 3) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 3)

/-- The domain of f is all real numbers except 3 -/
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l326_32693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_rectangle_with_20_marked_cells_l326_32633

/-- A cell on an infinite chessboard -/
structure Cell where
  x : ℤ
  y : ℤ

/-- A configuration of marked cells on an infinite chessboard -/
def MarkedConfiguration := Finset Cell

/-- A rectangle on an infinite chessboard -/
structure Rectangle where
  x1 : ℤ
  y1 : ℤ
  x2 : ℤ
  y2 : ℤ

/-- The number of marked cells in a rectangle -/
def markedCellsInRectangle (config : MarkedConfiguration) (rect : Rectangle) : ℕ :=
  (config.filter (fun cell => 
    cell.x ≥ rect.x1 ∧ cell.x ≤ rect.x2 ∧ 
    cell.y ≥ rect.y1 ∧ cell.y ≤ rect.y2)).card

/-- Main theorem -/
theorem exists_no_rectangle_with_20_marked_cells : 
  ∃ (config : MarkedConfiguration), 
    config.card = 40 ∧ 
    ∀ (rect : Rectangle), markedCellsInRectangle config rect ≠ 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_no_rectangle_with_20_marked_cells_l326_32633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_a_for_f_gt_a_squared_minus_2a_l326_32631

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for part (1)
theorem solution_set_f_geq_5 :
  {x : ℝ | f x ≥ 5} = Set.Iic (-3) ∪ Set.Ici 2 := by sorry

-- Theorem for part (2)
theorem range_of_a_for_f_gt_a_squared_minus_2a :
  {a : ℝ | ∀ x, f x > a^2 - 2*a} = Set.Ioo (-1) 3 := by sorry

-- Note: Set.Iic (-3) represents (-∞, -3], Set.Ici 2 represents [2, +∞), and Set.Ioo (-1) 3 represents (-1, 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_5_range_of_a_for_f_gt_a_squared_minus_2a_l326_32631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_age_problem_l326_32647

/-- Given that Alice, Bob, and Carl celebrate birthdays on the same day,
    Bob is 2 years older than Alice, Carl is 2 years old today,
    and today is the second of 8 birthdays where Alice's age is a multiple of Carl's age,
    prove that the sum of the digits of Bob's age the next time his age is a multiple of Carl's age is 5. -/
theorem birthday_age_problem (alice_age bob_age carl_age : ℕ) : 
  bob_age = alice_age + 2 →
  carl_age = 2 →
  alice_age % carl_age = 0 →
  (∃ (n : ℕ), n > 0 ∧ (bob_age + n) % (carl_age + n) = 0 ∧ 
    (Nat.digits 10 (bob_age + n)).sum = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_age_problem_l326_32647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_starting_numbers_l326_32643

def machine_step (n : Int) : Int :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def machine_iterate (n : Int) (steps : Nat) : Int :=
  match steps with
  | 0 => n
  | k + 1 => machine_iterate (machine_step n) k

def valid_starting_numbers : List Int :=
  (List.range 10000).filter (fun n => machine_iterate n 6 = 26)

theorem sum_of_valid_starting_numbers :
  valid_starting_numbers.sum = 3302 := by
  sorry

#eval valid_starting_numbers.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_starting_numbers_l326_32643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_through_two_points_l326_32635

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line3D) (p : Point3D) : Prop :=
  ∃ t : ℝ, p.x - l.point.x = t * l.direction.x ∧
            p.y - l.point.y = t * l.direction.y ∧
            p.z - l.point.z = t * l.direction.z

theorem line_through_point (x₁ y₁ z₁ m n p : ℝ) :
  let l := { point := ⟨x₁, y₁, z₁⟩, direction := ⟨m, n, p⟩ : Line3D }
  ∀ x y z : ℝ, (x - x₁) / m = (y - y₁) / n ∧ (y - y₁) / n = (z - z₁) / p →
  pointOnLine l ⟨x, y, z⟩ :=
by sorry

theorem line_through_two_points (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  let p₁ : Point3D := ⟨x₁, y₁, z₁⟩
  let p₂ : Point3D := ⟨x₂, y₂, z₂⟩
  let l := { point := p₁, direction := ⟨x₂ - x₁, y₂ - y₁, z₂ - z₁⟩ : Line3D }
  ∀ x y z : ℝ, (x - x₁) / (x₂ - x₁) = (y - y₁) / (y₂ - y₁) ∧
              (y - y₁) / (y₂ - y₁) = (z - z₁) / (z₂ - z₁) →
  pointOnLine l ⟨x, y, z⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_through_two_points_l326_32635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l326_32639

-- Define a vector type
variable {V : Type*} [NormedAddCommGroup V] [Module ℝ V]

-- Define collinearity
def collinear (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

-- Statement of the theorem
theorem vector_properties :
  (∃ (v : V), v ≠ 0 ∧ ‖v‖ = 0) ∨  -- Negation of "The length of the zero vector is 0"
  (∀ (v w : V), collinear v w → v = w) -- Negation of "Collinear vectors are not necessarily equal"
  → False := by
  sorry

-- The theorem states that the negation of our correct statements leads to a contradiction,
-- which is equivalent to proving that our statements are true.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l326_32639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxNoncongruentTriangles_correct_l326_32668

def maxNoncongruentTriangles (n : ℕ) : ℚ :=
  if n % 2 = 1 then
    ((n + 1) * (n + 3) * (2 * n) + 5) / 24
  else
    (n / 2) * ((n / 2) + 1) * (n + 1) / 3

theorem maxNoncongruentTriangles_correct (n : ℕ) :
  maxNoncongruentTriangles n =
    if n % 2 = 1 then
      ((n + 1) * (n + 3) * (2 * n) + 5) / 24
    else
      (n / 2) * ((n / 2) + 1) * (n + 1) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxNoncongruentTriangles_correct_l326_32668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wins_for_ten_points_l326_32614

/-- The game "Rock, Paper, Scissors" with the following rules:
  * Total number of games: 7
  * Win: 3 points
  * Tie: 1 point
  * Loss: 0 points
  * Xiao Wang's final score: 10 points
-/
structure RockPaperScissors where
  total_games : ℕ
  win_points : ℕ
  tie_points : ℕ
  loss_points : ℕ
  final_score : ℕ
  wins : ℕ
  ties : ℕ
  losses : ℕ

/-- The specific instance of the game we're considering -/
def game : RockPaperScissors where
  total_games := 7
  win_points := 3
  tie_points := 1
  loss_points := 0
  final_score := 10
  wins := 0  -- to be determined
  ties := 0  -- to be determined
  losses := 0  -- to be determined

/-- The theorem stating the minimum number of wins required -/
theorem min_wins_for_ten_points (g : RockPaperScissors) (h1 : g = game) :
  ∃ (w t l : ℕ),
    w + t + l = g.total_games ∧
    w * g.win_points + t * g.tie_points = g.final_score ∧
    ∀ (w' t' l' : ℕ),
      w' + t' + l' = g.total_games →
      w' * g.win_points + t' * g.tie_points = g.final_score →
      w ≤ w' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wins_for_ten_points_l326_32614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_tone_frequency_l326_32684

/-- The frequency ratio between consecutive tones in the Twelve-Tone Equal Temperament system -/
noncomputable def frequency_ratio : ℝ := Real.rpow 2 (1 / 12)

/-- The sequence of frequencies in the Twelve-Tone Equal Temperament system -/
noncomputable def frequency_sequence : ℕ → ℝ
  | 0 => 1
  | n + 1 => frequency_ratio * frequency_sequence n

theorem seventh_tone_frequency :
  frequency_sequence 6 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_tone_frequency_l326_32684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_total_cost_l326_32688

/-- The price Mike paid for the DVD in the store -/
noncomputable def store_price : ℝ := 5

/-- The price of the DVD online -/
noncomputable def online_price : ℝ := 2 * store_price

/-- The shipping cost as a percentage of the online price -/
noncomputable def shipping_percentage : ℝ := 80 / 100

/-- The shipping cost -/
noncomputable def shipping_cost : ℝ := shipping_percentage * online_price

/-- The total amount Steve paid -/
noncomputable def total_cost : ℝ := online_price + shipping_cost

/-- Theorem stating that the total cost for Steve is $18 -/
theorem steve_total_cost : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_total_cost_l326_32688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_roots_l326_32624

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - x - 2

-- Theorem statement
theorem f_has_two_roots : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_roots_l326_32624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_brand_growth_rate_l326_32661

/-- Proves the growth rate of the first brand of computers -/
theorem first_brand_growth_rate 
  (initial_first : ℝ) 
  (initial_second : ℝ) 
  (growth_rate_second : ℝ) 
  (equal_time : ℝ) 
  (growth_rate_first : ℝ)
  (h1 : initial_first = 4.9)
  (h2 : initial_second = 2.5)
  (h3 : growth_rate_second = 0.7)
  (h4 : equal_time = 5.647)
  (h5 : initial_first + growth_rate_first * equal_time = initial_second + growth_rate_second * equal_time) :
  abs (growth_rate_first - 0.275) < 0.001 := by
  sorry

#eval (2.5 + 0.7 * 5.647 - 4.9) / 5.647

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_brand_growth_rate_l326_32661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l326_32667

theorem sin_x_value (x a : ℝ) 
  (h1 : x ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.cos (2*x) = a) : 
  Real.sin x = -Real.sqrt ((1 - a)/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_value_l326_32667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_one_third_l326_32651

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n + 1 => (2 * sequence_a n) / (sequence_a n + 2)

theorem a_5_equals_one_third :
  sequence_a 5 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_one_third_l326_32651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l326_32670

/-- Represents a parabola with vertex (h, k) and parameter a -/
structure Parabola where
  h : ℝ
  k : ℝ
  a : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem parabola_triangle_area :
  ∀ (p : Parabola) (A B M : Point),
    p.h = 1 ∧ p.k = 2 ∧  -- Vertex at (1, 2)
    A.x = -1 ∧ A.y = 0 ∧  -- A is at (-1, 0)
    B.y = 0 ∧  -- B is on x-axis
    M.x = 0 ∧  -- M is on y-axis
    (A.y = p.a * (A.x - p.h)^2 + p.k) ∧  -- A is on the parabola
    (B.y = p.a * (B.x - p.h)^2 + p.k) ∧  -- B is on the parabola
    (M.y = p.a * (M.x - p.h)^2 + p.k) →  -- M is on the parabola
    triangle_area A B M = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l326_32670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_entertainment_is_ten_percent_l326_32627

-- Define Rohan's salary and expenses
noncomputable def salary : ℝ := 10000
noncomputable def food_percentage : ℝ := 40
noncomputable def rent_percentage : ℝ := 20
noncomputable def conveyance_percentage : ℝ := 10
noncomputable def savings : ℝ := 2000

-- Define the function to calculate the entertainment percentage
noncomputable def entertainment_percentage : ℝ :=
  100 - (food_percentage + rent_percentage + conveyance_percentage + (savings / salary * 100))

-- Theorem statement
theorem entertainment_is_ten_percent :
  entertainment_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_entertainment_is_ten_percent_l326_32627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_condition_l326_32652

theorem shaded_areas_equality_condition (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 2) :
  (∃ (r : Real), r > 0 ∧ 
    (r^2 * Real.tan φ) / 2 - (φ * r^2) / 2 = (φ * r^2) / 2) ↔ 
  Real.tan φ = 2 * φ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_condition_l326_32652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_conditions_l326_32699

/-- Moves the last digit of a natural number to the front. -/
def moveLastDigitToFront (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let restOfNumber := n / 10
  lastDigit * (10 ^ (Nat.log 10 restOfNumber + 1)) + restOfNumber

/-- Checks if a number satisfies the problem conditions. -/
def satisfiesConditions (n : ℕ) : Prop :=
  n % 10 = 6 ∧ moveLastDigitToFront n = 4 * n

theorem smallest_number_with_conditions :
  satisfiesConditions 153846 ∧
  ∀ m : ℕ, m < 153846 → ¬satisfiesConditions m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_conditions_l326_32699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_a_2019_l326_32634

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n + 1 / a n

-- State the theorem
theorem integer_part_a_2019 : 
  ⌊a 2019⌋ = 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_a_2019_l326_32634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l326_32600

/-- Calculates the time difference in minutes for a journey given two speeds -/
noncomputable def timeDifference (distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (distance / speed2 - distance / speed1) * 60

theorem journey_time_difference :
  let distance : ℝ := 70
  let speed_ontime : ℝ := 40
  let speed_late : ℝ := 35
  timeDifference distance speed_ontime speed_late = 15 := by
    -- Unfold the definition of timeDifference
    unfold timeDifference
    -- Simplify the expression
    simp
    -- The proof is completed with sorry
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_difference_l326_32600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l326_32691

/-- A grasshopper jumping on a number line -/
noncomputable def Grasshopper (a b : ℝ) := ℝ → ℝ

/-- The grasshopper's jump function -/
noncomputable def jump (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x + a else x - b

/-- A sequence of jumps -/
noncomputable def jumpSequence (a b : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => jump a b (jumpSequence a b n x)

theorem grasshopper_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_irr : Irrational (b / a)) :
  ∃ (n : ℕ) (x : ℝ), x ∈ Set.Icc (-a) b ∧ 
    ∀ (m : ℕ), m ≥ n → |jumpSequence a b m x| < 10^(-6 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l326_32691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_with_parameter_d_l326_32632

/-- Theorem: Collinearity of Points with Parameter d -/
theorem collinear_points_with_parameter_d 
  (a b c d : ℝ) : 
  let p1 : Fin 3 → ℝ := ![2, 0, a]
  let p2 : Fin 3 → ℝ := ![b, 2, 0]
  let p3 : Fin 3 → ℝ := ![0, c, 2]
  let p4 : Fin 3 → ℝ := ![9*d, 9*d, -d]
  (∃ (t u v : ℝ), 
    p2 - p1 = t • (p3 - p1) ∧ 
    p2 - p1 = u • (p4 - p1) ∧ 
    p3 - p1 = v • (p4 - p1)) → 
  d = -1/9 ∨ d = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_with_parameter_d_l326_32632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l326_32694

-- Define the sequence
def a : ℕ → ℚ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | k+1 => 1 / ((List.range (k+1)).sum + 1)

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := (List.range n).map a |>.sum

-- Theorem statement
theorem sequence_sum (n : ℕ) : S n = 2 * n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l326_32694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l326_32607

open Real

theorem omega_range (f : ℝ → ℝ) (f' : ℝ → ℝ) (ω φ t : ℝ) :
  (∀ x, f x = sin (ω * x + φ)) →
  ω > 0 →
  (∀ x, HasDerivAt f (f' x) x) →
  f t = 0 →
  f' t > 0 →
  (∀ x ∈ Set.Ici t ∩ Set.Iio (t + 1), ¬ IsLocalMin f x) →
  ω ∈ Set.Ioo π (3 * π / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l326_32607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l326_32606

/-- Represents an ellipse in a 2D coordinate system -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the properties and equation of the ellipse (C) -/
theorem ellipse_equation (C : Ellipse) (F₁ F₂ A B : Point) :
  -- The center of C is at the origin
  C.a > 0 ∧ C.b > 0 →
  -- F₁ and F₂ are on the x-axis
  F₁.y = 0 ∧ F₂.y = 0 →
  -- The eccentricity of C is 1/2
  (abs F₁.x / C.a) = (1 / 2) →
  -- A and B are on the ellipse C
  (A.x^2 / C.a^2) + (A.y^2 / C.b^2) = 1 →
  (B.x^2 / C.a^2) + (B.y^2 / C.b^2) = 1 →
  -- The perimeter of triangle ABF₂ is 16
  (((A.x - B.x)^2 + (A.y - B.y)^2).sqrt +
   ((A.x - F₂.x)^2 + (A.y - F₂.y)^2).sqrt +
   ((B.x - F₂.x)^2 + (B.y - F₂.y)^2).sqrt) = 16 →
  -- The equation of C is x²/16 + y²/12 = 1
  C.a^2 = 16 ∧ C.b^2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l326_32606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l326_32608

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Calculates the perimeter of a quadrilateral given its four vertices -/
noncomputable def quadrilateralPerimeter (a b c d : Point2D) : ℝ :=
  distance a b + distance b c + distance c d + distance d a

/-- The coordinates of the trapezoid vertices -/
def J : Point2D := ⟨-3, -4⟩
def K : Point2D := ⟨-3, 1⟩
def L : Point2D := ⟨5, 7⟩
def M : Point2D := ⟨5, -4⟩

theorem trapezoid_perimeter : quadrilateralPerimeter J K L M = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l326_32608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l326_32655

structure Envelope where
  length : ℚ
  height : ℚ

def needsExtraCharge (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 3/2 || ratio > 3

def envelopes : List Envelope := [
  ⟨7, 5⟩,
  ⟨10, 2⟩,
  ⟨8, 8⟩,
  ⟨12, 3⟩
]

theorem extra_charge_count : 
  (envelopes.filter needsExtraCharge).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l326_32655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_teams_tied_for_most_wins_l326_32616

def Tournament := Fin 8 → Fin 8 → Option (Fin 2)

def is_valid_tournament (t : Tournament) : Prop :=
  ∀ i j, i ≠ j → t i j = some 0 ↔ t j i = some 1

def wins (t : Tournament) (team : Fin 8) : Nat :=
  (Finset.filter (λ j => t team j = some 0) Finset.univ).card

def max_wins (t : Tournament) : Nat :=
  Finset.sup Finset.univ (wins t)

def teams_with_max_wins (t : Tournament) : Finset (Fin 8) :=
  Finset.filter (λ i => wins t i = max_wins t) Finset.univ

theorem max_teams_tied_for_most_wins :
  ∃ t : Tournament, is_valid_tournament t ∧ (teams_with_max_wins t).card = 7 ∧
    ∀ t' : Tournament, is_valid_tournament t' → (teams_with_max_wins t').card ≤ 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_teams_tied_for_most_wins_l326_32616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l326_32675

noncomputable def f (x : ℝ) : ℝ := x - 1/x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l326_32675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l326_32679

/-- Represents the distribution of scores in the exam -/
structure ScoreDistribution where
  score_70 : ℝ
  score_80 : ℝ
  score_85 : ℝ
  score_90 : ℝ
  score_95 : ℝ
  sum_to_one : score_70 + score_80 + score_85 + score_90 + score_95 = 1

/-- Calculates the mean score given a score distribution -/
def mean_score (d : ScoreDistribution) : ℝ :=
  70 * d.score_70 + 80 * d.score_80 + 85 * d.score_85 + 90 * d.score_90 + 95 * d.score_95

/-- Calculates the median score given a score distribution -/
noncomputable def median_score (d : ScoreDistribution) : ℝ :=
  if d.score_70 + d.score_80 > (1/2 : ℝ) then 80
  else if d.score_70 + d.score_80 + d.score_85 > (1/2 : ℝ) then 85
  else if d.score_70 + d.score_80 + d.score_85 + d.score_90 > (1/2 : ℝ) then 90
  else 95

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score_70 = 0.1)
  (h2 : d.score_80 = 0.25)
  (h3 : d.score_85 = 0.2)
  (h4 : d.score_90 = 0.15) :
  |mean_score d - median_score d| = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l326_32679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l326_32671

theorem order_of_expressions : ∃ (a b c : ℝ), 
  a = (0.9 : ℝ)^2 ∧ 
  b = (2 : ℝ)^(0.9 : ℝ) ∧ 
  c = Real.log 0.9 / Real.log 2 ∧ 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l326_32671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_family_time_l326_32613

noncomputable def hours_in_day : ℝ := 24

noncomputable def sleeping_fraction : ℝ := 1/3
noncomputable def school_fraction : ℝ := 1/6
noncomputable def assignments_fraction : ℝ := 1/12

noncomputable def time_with_family (total_hours sleeping_frac school_frac assignments_frac : ℝ) : ℝ :=
  total_hours * (1 - (sleeping_frac + school_frac + assignments_frac))

theorem steve_family_time : 
  time_with_family hours_in_day sleeping_fraction school_fraction assignments_fraction = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_family_time_l326_32613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_value_l326_32650

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define a as a positive real number not equal to 1
variable (a : ℝ)

-- State the theorem
theorem prove_a_value
  (h1 : ∀ x, g x ≠ 0)
  (h2 : ∀ x, (deriv f x) * g x - f x * (deriv g x) > 0)
  (h3 : ∀ x, f x = a^x * g x)
  (h4 : a > 0)
  (h5 : a ≠ 1)
  (h6 : f 1 / g 1 + f (-1) / g (-1) = 10/3) :
  a = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_value_l326_32650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approx_l326_32669

/-- The radius of a wheel given its total distance covered and number of revolutions --/
noncomputable def wheel_radius (distance : ℝ) (revolutions : ℕ) : ℝ :=
  distance / (2 * Real.pi * (revolutions : ℝ))

/-- Theorem stating that a wheel with given parameters has a specific radius --/
theorem wheel_radius_approx :
  let distance := (4224 : ℝ)
  let revolutions := (3000 : ℕ)
  let calculated_radius := wheel_radius distance revolutions
  ∃ ε > 0, |calculated_radius - 0.224| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_approx_l326_32669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_12_seconds_time_to_travel_380_meters_l326_32644

-- Define the distance function
noncomputable def distance (t : ℝ) : ℝ := 9 * t + (1 / 2) * t^2

-- Theorem 1: After 12 seconds, the car has traveled 180 meters
theorem distance_after_12_seconds :
  distance 12 = 180 := by sorry

-- Theorem 2: It takes 20 seconds to travel 380 meters
theorem time_to_travel_380_meters :
  ∃ t : ℝ, distance t = 380 ∧ t = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_12_seconds_time_to_travel_380_meters_l326_32644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l326_32695

noncomputable def series_term (n : ℕ) : ℝ := n / (5 ^ n)

theorem series_sum : ∑' n, series_term n = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l326_32695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_year_markup_percentage_l326_32696

noncomputable def apply_markup (cost : ℝ) (markup_percent : ℝ) : ℝ :=
  cost * (1 + markup_percent / 100)

noncomputable def apply_discount (price : ℝ) (discount_percent : ℝ) : ℝ :=
  price * (1 - discount_percent / 100)

noncomputable def profit_percent (cost : ℝ) (selling_price : ℝ) : ℝ :=
  (selling_price / cost - 1) * 100

theorem new_year_markup_percentage (cost : ℝ) (new_year_markup : ℝ) 
  (h_cost_positive : cost > 0) :
  let initial_price := apply_markup cost 20
  let new_year_price := apply_markup initial_price new_year_markup
  let final_price := apply_discount new_year_price 20
  profit_percent cost final_price = 20 → new_year_markup = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_year_markup_percentage_l326_32696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_manager_percentage_l326_32603

theorem initial_manager_percentage 
  (total_employees : ℕ) 
  (managers_leaving : ℕ) 
  (remaining_percentage : ℚ) 
  (initial_percentage : ℚ) : Prop :=
  total_employees = 600 ∧
  managers_leaving = 300 ∧
  remaining_percentage = 98/100 ∧
  (initial_percentage / 100 * total_employees - managers_leaving) / 
    (total_employees - managers_leaving) = remaining_percentage →
  initial_percentage = 99/100

example : initial_manager_percentage 600 300 (98/100) (99/100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_manager_percentage_l326_32603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_from_compound_interest_l326_32686

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem simple_interest_from_compound_interest :
  ∀ (P : ℝ),
  compound_interest P 5 2 = 57.40 →
  simple_interest P 5 2 = 56 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_from_compound_interest_l326_32686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_on_interval_g_max_at_sqrt_two_thirds_l326_32698

/-- The function g(x) = 2x - x^3 -/
def g (x : ℝ) : ℝ := 2 * x - x^3

/-- The maximum value of g(x) on [0, 2] -/
noncomputable def max_value : ℝ := (2 * (3 * Real.sqrt 3 - Real.sqrt 2) * Real.sqrt (2/3)) / (3 * Real.sqrt 3)

/-- Theorem stating that max_value is the maximum of g(x) on [0, 2] -/
theorem g_max_on_interval :
  ∀ x ∈ Set.Icc 0 2, g x ≤ max_value := by
  sorry

/-- Theorem stating that max_value is achieved at x = sqrt(2/3) -/
theorem g_max_at_sqrt_two_thirds :
  g (Real.sqrt (2/3)) = max_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_on_interval_g_max_at_sqrt_two_thirds_l326_32698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_inscribed_cone_l326_32618

/-- A cone inscribed around a cube with edge length a -/
structure InscribedCone (a : ℝ) where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The height of the cone -/
  h : ℝ
  /-- Four vertices of the cube lie on the plane of the base of the cone -/
  base_vertices : r > 0
  /-- The other four vertices of the cube lie on the lateral surface of the cone -/
  lateral_vertices : h > a
  /-- The relationship between r and h based on the cube's geometry -/
  geometry : r = (a * h * Real.sqrt 2) / (2 * (h - a))

/-- The volume of the cone -/
noncomputable def cone_volume (a : ℝ) (c : InscribedCone a) : ℝ :=
  (1/3) * Real.pi * c.r^2 * c.h

/-- The theorem stating the minimum volume of the inscribed cone -/
theorem min_volume_inscribed_cone (a : ℝ) (ha : a > 0) :
  ∃ (c : InscribedCone a), ∀ (c' : InscribedCone a),
    cone_volume a c ≤ cone_volume a c' ∧ cone_volume a c = (9/8) * Real.pi * a^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_inscribed_cone_l326_32618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parker_popcorn_experiment_l326_32640

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the average percentage of popped kernels -/
noncomputable def averagePopPercentage (bags : List PopcornBag) : ℝ :=
  let proportions := bags.map (λ bag => (bag.popped : ℝ) / (bag.total : ℝ))
  100 * (proportions.sum / bags.length)

theorem parker_popcorn_experiment :
  let bags : List PopcornBag := [
    { total := 75, popped := 60 },
    { total := 50, popped := 42 },
    { total := 100, popped := 82 }
  ]
  averagePopPercentage bags = 82 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parker_popcorn_experiment_l326_32640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l326_32620

/-- The function f(x) = e^x - x - 2 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

theorem max_k_value (k : ℤ) (h : k > 2) :
  ∃ x > 0, (x - ↑k) * f' x + x + 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l326_32620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_equals_four_over_2007_l326_32646

/-- Calculates the product of fractions in the sequence (4/7) * (7/10) * (10/13) * ... * (2004/2007) -/
def fraction_product : ℚ := 
  let sequence := List.range 668 |>.map (fun i => (4 + 3 * i : ℚ) / (7 + 3 * i))
  sequence.prod

theorem fraction_product_equals_four_over_2007 : fraction_product = 4 / 2007 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_equals_four_over_2007_l326_32646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jennifer_milk_spending_l326_32609

-- Define the initial purchase and rules
def initial_purchase : ℕ := 40
def mark_whole_milk : ℕ := 30
def additional_rule : ℕ → ℕ := λ x ↦ (x / 5) * 6

-- Define the price and discount rules
def price_per_can : ℚ := 2
def discount_rule : ℕ → ℚ := λ x ↦ ((x / 10) * 4 : ℚ)

-- Define the total cans purchased
def total_cans : ℕ := initial_purchase + additional_rule mark_whole_milk

-- Define the total cost before discount
def total_cost_before_discount : ℚ := (total_cans : ℚ) * price_per_can

-- Define the total discount
def total_discount : ℚ := discount_rule total_cans

-- Define the final cost
def final_cost : ℚ := total_cost_before_discount - total_discount

-- Theorem to prove
theorem jennifer_milk_spending :
  final_cost = 124 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jennifer_milk_spending_l326_32609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_roots_sum_of_squares_l326_32663

theorem prime_roots_sum_of_squares (p q R : ℤ) : 
  Nat.Prime p.natAbs → Nat.Prime q.natAbs → p ≠ q →
  p^2 - 13*p + R = 0 → q^2 - 13*q + R = 0 →
  p^2 + q^2 = 125 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_roots_sum_of_squares_l326_32663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l326_32690

theorem equation_solution (x : ℝ) : 
  (3 : ℝ)^((8 : ℝ)^x) = (8 : ℝ)^((3 : ℝ)^x) ↔ x = Real.log (Real.log 2 / Real.log 3) / Real.log (9/8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l326_32690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_is_circle_l326_32637

-- Define the center of the original circle
def O : ℝ × ℝ := (0, 0)

-- Define the radius of the original circle
def R : ℝ := 10

-- Define point P
def P : ℝ × ℝ := (15, 0)

-- Define a point Q on the original circle
def Q : ℝ × ℝ → Prop :=
  λ q => (q.1 - O.1)^2 + (q.2 - O.2)^2 = R^2

-- Define the midpoint M of PQ
noncomputable def M (q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + q.1) / 2, (P.2 + q.2) / 2)

-- Theorem stating the locus of M is a circle
theorem locus_of_M_is_circle :
  ∀ q, Q q →
    ((M q).1 - 7.5)^2 + ((M q).2 - 0)^2 = 5^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_M_is_circle_l326_32637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_time_l326_32692

-- Define the time it takes for A to finish the work alone
noncomputable def A : ℝ := sorry

-- Define the time it takes for B to finish the work alone
def B : ℝ := 10

-- Define the time A and B work together
def together_time : ℝ := 2

-- Define the time B works alone after A leaves
def B_alone_time : ℝ := 3.999999999999999

-- Theorem statement
theorem A_work_time : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_work_time_l326_32692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_key_arrangement_count_l326_32623

/-- Represents the number of keys on the keychain. -/
def total_keys : ℕ := 6

/-- Represents the number of keys that must always be together. -/
def keys_together : ℕ := 2

/-- Represents the number of distinct units to arrange (treating the two keys that must be together as one unit). -/
def units_to_arrange : ℕ := total_keys - keys_together + 1

/-- The number of distinct arrangements of keys on a circular keychain. -/
def circular_key_arrangements : ℕ := 48

/-- Theorem stating that the number of distinct arrangements of keys on a circular keychain,
    where two specific keys must always be together, is equal to 48. -/
theorem circular_key_arrangement_count :
  (units_to_arrange - 1) * Nat.factorial (units_to_arrange - 2) * 2 = circular_key_arrangements :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_key_arrangement_count_l326_32623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_g_range_of_h_l326_32638

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (y < 0 ∨ y > 0) := by sorry

-- For the second function
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x - 1

theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ y ≥ -5 := by sorry

-- For the third function
noncomputable def h (x : ℝ) : ℝ := x + Real.sqrt (x + 1)

theorem range_of_h :
  ∀ y : ℝ, (∃ x : ℝ, h x = y) ↔ y ≥ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_range_of_g_range_of_h_l326_32638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_a_neq_3_l326_32612

/-- Two lines in 3D space -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, ∃ i : Fin 3, l1.point i + t * l1.direction i ≠ l2.point i + u * l2.direction i

theorem lines_skew_iff_a_neq_3 (a : ℝ) :
  let l1 : Line3D := ⟨![1, 2, a], ![2, 3, 4]⟩
  let l2 : Line3D := ⟨![4, 1, 0], ![5, 2, 1]⟩
  are_skew l1 l2 ↔ a ≠ 3 := by
  sorry

#check lines_skew_iff_a_neq_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_a_neq_3_l326_32612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocal_equation_l326_32656

theorem min_sum_reciprocal_equation (a b : ℕ+) (ha : a ≠ b) (heq : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1 / 12) :
  ∃ (m n : ℕ+), m ≠ n ∧ (m : ℚ)⁻¹ + (n : ℚ)⁻¹ = 1 / 12 ∧ m + n = 54 ∧
  ∀ (x y : ℕ+), x ≠ y → (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1 / 12 → x + y ≥ 54 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocal_equation_l326_32656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l326_32678

def sequence_property (a : ℕ → ℤ) : Prop :=
  (a 3 = 12) ∧
  (a 11 = -5) ∧
  (∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 11)

def sum_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) a

theorem sequence_properties (a : ℕ → ℤ) (h : sequence_property a) :
  (a 2017 = 4) ∧
  (∃ n : ℕ, sum_n a n ≤ 100 ∧ sum_n a (n + 1) > 100 ∧ n = 29) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l326_32678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l326_32629

theorem sin_double_angle_problem (α : ℝ) :
  Real.sin α + Real.sin (π / 2 + α) = 2 * Real.sqrt 5 / 5 →
  Real.sin (2 * α) = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_problem_l326_32629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_probability_l326_32673

/-- The number of exam tickets -/
def total_tickets : ℕ := 28

/-- The number of theoretical questions on each ticket -/
def theoretical_questions_per_ticket : ℕ := 2

/-- The number of problems on each ticket -/
def problems_per_ticket : ℕ := 1

/-- The number of theoretical questions the student has prepared -/
def prepared_theoretical_questions : ℕ := 50

/-- The number of problems the student has prepared -/
def prepared_problems : ℕ := 22

/-- The probability of answering all questions correctly on a randomly drawn exam ticket -/
theorem correct_answer_probability :
  (Nat.choose prepared_theoretical_questions theoretical_questions_per_ticket *
   prepared_problems : ℚ) / (Nat.choose (prepared_theoretical_questions + problems_per_ticket)
   theoretical_questions_per_ticket * total_tickets) = 2695 / 4312 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_probability_l326_32673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_l326_32681

theorem complex_expression_evaluation : 
  (27 : Real)^(1/3) + (-1/3)⁻¹ - (3 - Real.pi)^0 + (-1)^2017 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_evaluation_l326_32681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_value_l326_32602

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The conditions on the polynomial P and integer b -/
def PolynomialConditions (P : IntPolynomial) (b : ℤ) : Prop :=
  b > 0 ∧
  (P.eval 1 = b) ∧ (P.eval 4 = b) ∧ (P.eval 5 = b) ∧ (P.eval 8 = b) ∧
  (P.eval 2 = -b) ∧ (P.eval 3 = -b) ∧ (P.eval 6 = -b) ∧ (P.eval 7 = -b)

/-- The theorem stating that 720 is the smallest possible value for b -/
theorem smallest_b_value (P : IntPolynomial) (b : ℤ) 
  (h : PolynomialConditions P b) : b ≥ 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_value_l326_32602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_combination_l326_32645

theorem quadratic_root_combination (a : ℝ) : 
  (∃ k : ℝ, k * Real.sqrt 12 = 5 * Real.sqrt (a + 1)) → 
  (a = 2 ∧ Real.sqrt 12 + 5 * Real.sqrt (a + 1) = 7 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_combination_l326_32645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train2_length_problem_l326_32657

/-- The length of a train that crosses another train in opposite direction -/
noncomputable def train2_length (train1_length : ℝ) (train1_speed : ℝ) (train2_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  (train1_speed + train2_speed) * crossing_time / 3.6 - train1_length

/-- Theorem stating the length of Train 2 given the problem conditions -/
theorem train2_length_problem :
  let train1_length : ℝ := 270
  let train1_speed : ℝ := 120
  let train2_speed : ℝ := 80
  let crossing_time : ℝ := 9
  abs (train2_length train1_length train1_speed train2_speed crossing_time - 229.95) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train2_length_problem_l326_32657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_specific_prism_l326_32674

/-- Represents a rectangular prism with given dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the path length of a dot on the top face of a rectangular prism when rolled for two full revolutions -/
noncomputable def pathLength (prism : RectangularPrism) : ℝ :=
  4 * Real.pi * Real.sqrt ((prism.width / 2) ^ 2 + prism.height ^ 2)

/-- Theorem stating the path length for a specific rectangular prism -/
theorem path_length_specific_prism :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  pathLength prism = 2 * Real.sqrt 5 * Real.pi := by
  sorry

#check path_length_specific_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_specific_prism_l326_32674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_number_problem_l326_32660

theorem six_digit_number_problem (P Q R S T U : Nat) : 
  (P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
   Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
   R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
   S ≠ T ∧ S ≠ U ∧
   T ≠ U) →
  ({P, Q, R, S, T, U} : Finset Nat) = {1, 2, 3, 4, 5, 6} →
  (100 * P + 10 * Q + R) % 4 = 0 →
  (100 * Q + 10 * R + S) % 5 = 0 →
  (100 * R + 10 * S + T) % 2 = 0 →
  (P + Q + R + S + T + U) % 11 = 0 →
  P = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_number_problem_l326_32660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_eq_2_minus_f_neg_l326_32662

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse of f
def f_inv : ℝ → ℝ := sorry

-- Assume f_inv is the inverse of f
axiom f_inv_is_inverse : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the third function
def g : ℝ → ℝ := sorry

-- Assume g is symmetric to f_inv about (1,0)
axiom g_symmetry : ∀ x y, g y = x ↔ f_inv (2 - x) = -y

-- Define the inverse of g
def g_inv : ℝ → ℝ := sorry

-- Theorem: The inverse of g is 2 - f(-x)
theorem g_inv_eq_2_minus_f_neg (x : ℝ) : g_inv x = 2 - f (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_eq_2_minus_f_neg_l326_32662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_candle_l326_32626

-- Define the dimensions of the wax block
def block_length : ℝ := 8
def block_width : ℝ := 3
def block_height : ℝ := 2

-- Define the dimensions of the cylindrical candle
def candle_height : ℝ := 10
def candle_diameter : ℝ := 5

-- Calculate the volume of one wax block
def block_volume : ℝ := block_length * block_width * block_height

-- Calculate the volume of the cylindrical candle
noncomputable def candle_volume : ℝ := Real.pi * (candle_diameter / 2)^2 * candle_height

-- Theorem statement
theorem blocks_needed_for_candle :
  ⌈candle_volume / block_volume⌉ = 5 := by
  sorry

#eval block_volume
-- The following line is commented out because it's noncomputable
-- #eval candle_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_needed_for_candle_l326_32626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_folding_positions_l326_32697

/-- Represents a position where an additional square can be attached to the rectangle. -/
inductive Position
| Corner
| EdgeMiddle
| TopBottomMiddle
| InsideEdge

/-- Represents the 3x2 rectangle made of congruent squares. -/
structure Rectangle where
  length : Nat
  width : Nat
  h_length : length = 3
  h_width : width = 2

/-- Represents the shape after attaching an additional square. -/
structure ExtendedShape where
  base : Rectangle
  additional_square_position : Position

/-- Determines if a given position allows for folding into a valid box. -/
def allows_valid_folding (shape : ExtendedShape) : Bool :=
  match shape.additional_square_position with
  | Position.Corner => true
  | Position.EdgeMiddle => true
  | Position.TopBottomMiddle => false
  | Position.InsideEdge => false

/-- Counts the number of positions that allow valid folding. -/
def count_valid_positions (rectangle : Rectangle) : Nat :=
  let all_positions := [Position.Corner, Position.Corner, Position.Corner, Position.Corner,
                        Position.EdgeMiddle, Position.EdgeMiddle, Position.EdgeMiddle, Position.EdgeMiddle,
                        Position.TopBottomMiddle, Position.TopBottomMiddle,
                        Position.InsideEdge, Position.InsideEdge, Position.InsideEdge]
  (all_positions.map (λ pos => allows_valid_folding ⟨rectangle, pos⟩)).filter id |>.length

/-- The main theorem to be proved. -/
theorem valid_folding_positions (rectangle : Rectangle) :
  count_valid_positions rectangle = 8 := by
  sorry

#eval count_valid_positions ⟨3, 2, rfl, rfl⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_folding_positions_l326_32697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l326_32622

/-- The speed of a train that overtakes a motorbike -/
noncomputable def train_speed (motorbike_speed : ℝ) (overtake_time : ℝ) (train_length : ℝ) : ℝ :=
  motorbike_speed + (train_length / 1000) / (overtake_time / 3600)

/-- Theorem: The speed of the train is 100.00288 kmph -/
theorem train_speed_calculation :
  let motorbike_speed : ℝ := 64
  let overtake_time : ℝ := 40
  let train_length : ℝ := 400.032
  train_speed motorbike_speed overtake_time train_length = 100.00288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l326_32622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_problem_l326_32666

/-- Given a hyperbola with eccentricity 2√3/3 and a line l passing through its vertices
    such that the distance from the origin to l is √3/2, prove that the hyperbola's
    equation is x²/3 - y² = 1 and find the equation of line m passing through (0, -1)
    and intersecting the hyperbola at two points whose position vectors have dot product -23. -/
theorem hyperbola_and_line_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := 2 * Real.sqrt 3 / 3
  let l := {p : ℝ × ℝ | b * p.1 - a * p.2 = a * b}
  let d := Real.sqrt 3 / 2
  let hyperbola := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  (∀ p ∈ l, (abs (b * p.1 - a * p.2) / Real.sqrt (a^2 + b^2) = d)) →
  (e = Real.sqrt (a^2 + b^2) / a) →
  (hyperbola = {p : ℝ × ℝ | p.1^2 / 3 - p.2^2 = 1}) ∧
  (∃ k : ℝ, k = 1/2 ∨ k = -1/2) ∧
  (∀ m : Set (ℝ × ℝ), m = {p : ℝ × ℝ | p.2 = k * p.1 - 1} →
    ∃ M N : ℝ × ℝ, M ∈ hyperbola ∧ N ∈ hyperbola ∧ M ∈ m ∧ N ∈ m ∧
    M.1 * N.1 + M.2 * N.2 = -23) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_problem_l326_32666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_minutes_for_b_grade_l326_32677

/-- Prove that the number of extra minutes earned for each B grade is 1 --/
theorem extra_minutes_for_b_grade (
  normal_recess : ℕ
) (extra_minutes_per_a : ℕ)
  (extra_minutes_per_c : ℕ)
  (minutes_lost_per_d : ℕ)
  (num_a_grades : ℕ)
  (num_b_grades : ℕ)
  (num_c_grades : ℕ)
  (num_d_grades : ℕ)
  (total_recess : ℕ)
  (h1 : normal_recess = 20)
  (h2 : extra_minutes_per_a = 2)
  (h3 : extra_minutes_per_c = 0)
  (h4 : minutes_lost_per_d = 1)
  (h5 : num_a_grades = 10)
  (h6 : num_b_grades = 12)
  (h7 : num_c_grades = 14)
  (h8 : num_d_grades = 5)
  (h9 : total_recess = 47) : 
  ∃ (extra_minutes_per_b : ℕ), 
    normal_recess + 
    (extra_minutes_per_a * num_a_grades) + 
    (extra_minutes_per_b * num_b_grades) + 
    (extra_minutes_per_c * num_c_grades) - 
    (minutes_lost_per_d * num_d_grades) = total_recess ∧
    extra_minutes_per_b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_minutes_for_b_grade_l326_32677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l326_32642

theorem problem_statement :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → (Real.sqrt 2 / 2) * x ≤ Real.sin x ∧ Real.sin x ≤ x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x + x^2 + x^3 / 2 + 2 * (x + 2) * Real.cos x ≤ 4) ↔ a ∈ Set.Iic (-2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l326_32642
