import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_theorem_l3116_311678

theorem circle_radius_theorem (r : ℝ) (h : r > 0) : 3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l3116_311678


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l3116_311696

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_i_minus_one : imaginary_part (Complex.I - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_minus_one_l3116_311696


namespace NUMINAMATH_CALUDE_brick_length_l3116_311654

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: Given a rectangular prism with width 4, height 3, and surface area 164, its length is 10 -/
theorem brick_length (w h SA : ℝ) (hw : w = 4) (hh : h = 3) (hSA : SA = 164) :
  ∃ l : ℝ, surface_area l w h = SA ∧ l = 10 :=
sorry

end NUMINAMATH_CALUDE_brick_length_l3116_311654


namespace NUMINAMATH_CALUDE_defective_units_percentage_l3116_311693

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0024) :
  ∃ (defective_ratio : Real),
    defective_ratio = 0.06 ∧
    shipped_defective_ratio * defective_ratio = total_shipped_defective_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l3116_311693


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l3116_311642

theorem pentagon_area_sum (a b : ℤ) (h1 : 0 < b) (h2 : b < a) : 
  let P := (a, b)
  let Q := (b, a)
  let R := (-b, a)
  let S := (-b, -a)
  let T := (b, -a)
  let pentagon_area := a * (3 * b + a)
  pentagon_area = 792 → a + b = 45 := by
sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l3116_311642


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3116_311636

theorem min_value_abs_sum (x : ℝ) : 
  |x - 4| + |x + 7| + |x - 5| ≥ 4 ∧ ∃ y : ℝ, |y - 4| + |y + 7| + |y - 5| = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3116_311636


namespace NUMINAMATH_CALUDE_f_9_eq_two_thirds_l3116_311664

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is odd -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x-2) = f(x+2) for all x -/
axiom f_period : ∀ x, f (x - 2) = f (x + 2)

/-- f(x) = 3^x - 1 for x in [-2,0] -/
axiom f_def : ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3^x - 1

/-- The main theorem: f(9) = 2/3 -/
theorem f_9_eq_two_thirds : f 9 = 2/3 := by sorry

end NUMINAMATH_CALUDE_f_9_eq_two_thirds_l3116_311664


namespace NUMINAMATH_CALUDE_solve_for_m_l3116_311628

theorem solve_for_m : ∃ m : ℚ, (10 * (1/2 : ℚ) + m = 2) ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3116_311628


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l3116_311602

/-- The marshmallow challenge theorem -/
theorem marshmallow_challenge (haley michael brandon : ℕ) : 
  haley = 8 →
  michael = 3 * haley →
  brandon = michael / 2 →
  haley + michael + brandon = 44 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l3116_311602


namespace NUMINAMATH_CALUDE_wonderful_quadratic_range_l3116_311688

/-- A function is wonderful on a domain if it's monotonic and there exists an interval [a,b] in the domain
    such that the range of f on [a,b] is exactly [a,b] --/
def IsWonderful (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  Monotone f ∧ ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧
    (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
    (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

theorem wonderful_quadratic_range (m : ℝ) :
  IsWonderful (fun x => x^2 + m) (Set.Iic 0) →
  m ∈ Set.Ioo (-1) (-3/4) :=
sorry

end NUMINAMATH_CALUDE_wonderful_quadratic_range_l3116_311688


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3116_311646

theorem geometric_series_first_term (a r : ℝ) (h1 : r ≠ 1) (h2 : |r| < 1) : 
  (a / (1 - r) = 20) → 
  (a^2 / (1 - r^2) = 80) → 
  a = 20/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3116_311646


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l3116_311616

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Define non-coincidence for lines and planes
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

variable (m n : Line)
variable (α β γ : Plane)

-- Axioms
axiom non_coincident_mn : non_coincident_lines m n
axiom non_coincident_αβ : non_coincident_planes α β
axiom non_coincident_βγ : non_coincident_planes β γ
axiom non_coincident_αγ : non_coincident_planes α γ

-- Theorem 1
theorem perpendicular_implies_parallel 
  (h1 : perpendicular m α) (h2 : perpendicular m β) : 
  plane_parallel α β :=
sorry

-- Theorem 2
theorem skew_perpendicular_parallel_implies_perpendicular 
  (h1 : skew m n)
  (h2 : perpendicular m α) (h3 : parallel m β)
  (h4 : perpendicular n β) (h5 : parallel n α) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l3116_311616


namespace NUMINAMATH_CALUDE_zongzi_theorem_l3116_311659

/-- Represents the prices and quantities of zongzi --/
structure ZongziData where
  honey_price : ℝ
  meat_price : ℝ
  honey_quantity : ℕ
  meat_quantity : ℕ
  meat_sold_before : ℕ

/-- Represents the selling prices and profit --/
structure SaleData where
  honey_sell_price : ℝ
  meat_sell_price : ℝ
  meat_price_increase : ℝ
  meat_price_discount : ℝ
  total_profit : ℝ

/-- Main theorem stating the properties of zongzi prices and quantities --/
theorem zongzi_theorem (data : ZongziData) (sale : SaleData) : 
  data.meat_price = data.honey_price + 2.5 ∧ 
  300 / data.meat_price = 2 * (100 / data.honey_price) ∧
  data.honey_quantity = 100 ∧
  data.meat_quantity = 200 ∧
  sale.honey_sell_price = 6 ∧
  sale.meat_sell_price = 10 ∧
  sale.meat_price_increase = 1.1 ∧
  sale.meat_price_discount = 0.9 ∧
  sale.total_profit = 570 →
  data.honey_price = 5 ∧
  data.meat_price = 7.5 ∧
  data.meat_sold_before = 85 := by
  sorry

#check zongzi_theorem

end NUMINAMATH_CALUDE_zongzi_theorem_l3116_311659


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3116_311676

/-- A circle tangent to the y-axis with center on the line x - 3y = 0 and passing through (6, 1) -/
structure TangentCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- The circle is tangent to the y-axis
  tangent_to_y_axis : center.1 = radius
  -- The center is on the line x - 3y = 0
  center_on_line : center.1 = 3 * center.2
  -- The circle passes through (6, 1)
  passes_through_point : (center.1 - 6)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x-3)² + (y-1)² = 9 or (x-111)² + (y-37)² = 111² -/
theorem tangent_circle_equation (c : TangentCircle) :
  (∀ x y, (x - 3)^2 + (y - 1)^2 = 9) ∨
  (∀ x y, (x - 111)^2 + (y - 37)^2 = 111^2) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3116_311676


namespace NUMINAMATH_CALUDE_lindas_tv_cost_l3116_311619

def lindas_problem (original_savings : ℝ) (furniture_fraction : ℝ) : ℝ :=
  original_savings * (1 - furniture_fraction)

theorem lindas_tv_cost :
  lindas_problem 500 (4/5) = 100 := by sorry

end NUMINAMATH_CALUDE_lindas_tv_cost_l3116_311619


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3116_311674

theorem largest_n_for_equation : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6)) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6) →
    m ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3116_311674


namespace NUMINAMATH_CALUDE_satisfying_function_is_identity_l3116_311668

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (f 1 = 1) ∧
  (∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_satisfying_function_is_identity_l3116_311668


namespace NUMINAMATH_CALUDE_inequality_proof_l3116_311672

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3116_311672


namespace NUMINAMATH_CALUDE_katie_sold_four_pastries_l3116_311637

/-- The number of pastries sold at a bake sale -/
def pastries_sold (cupcakes cookies leftover : ℕ) : ℕ :=
  cupcakes + cookies - leftover

/-- Proof that Katie sold 4 pastries at the bake sale -/
theorem katie_sold_four_pastries :
  pastries_sold 7 5 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_katie_sold_four_pastries_l3116_311637


namespace NUMINAMATH_CALUDE_powerlifting_bodyweight_l3116_311687

theorem powerlifting_bodyweight (initial_total : ℝ) (total_gain_percent : ℝ) (weight_gain : ℝ) (final_ratio : ℝ) :
  initial_total = 2200 →
  total_gain_percent = 15 →
  weight_gain = 8 →
  final_ratio = 10 →
  ∃ initial_weight : ℝ,
    initial_weight > 0 ∧
    (initial_total * (1 + total_gain_percent / 100)) / (initial_weight + weight_gain) = final_ratio ∧
    initial_weight = 245 := by
  sorry

#check powerlifting_bodyweight

end NUMINAMATH_CALUDE_powerlifting_bodyweight_l3116_311687


namespace NUMINAMATH_CALUDE_fair_coin_prob_heads_l3116_311679

/-- A fair coin is a coin where the probability of getting heads is equal to the probability of getting tails -/
def is_fair_coin (coin : Type) (prob_heads : coin → ℝ) : Prop :=
  ∀ c : coin, prob_heads c = 1 - prob_heads c

/-- The probability of an event is independent of previous events if the probability remains constant regardless of previous outcomes -/
def is_independent_event {α : Type} (prob : α → ℝ) : Prop :=
  ∀ (a b : α), prob a = prob b

/-- Theorem: For a fair coin, the probability of getting heads on any single toss is 1/2, regardless of previous tosses -/
theorem fair_coin_prob_heads {coin : Type} (prob_heads : coin → ℝ) 
  (h_fair : is_fair_coin coin prob_heads) 
  (h_indep : is_independent_event prob_heads) :
  ∀ c : coin, prob_heads c = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_fair_coin_prob_heads_l3116_311679


namespace NUMINAMATH_CALUDE_general_trigonometric_equation_l3116_311617

theorem general_trigonometric_equation (θ : Real) : 
  Real.sin θ ^ 2 + Real.cos (θ + Real.pi / 6) ^ 2 + Real.sin θ * Real.cos (θ + Real.pi / 6) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_general_trigonometric_equation_l3116_311617


namespace NUMINAMATH_CALUDE_prob_more_heads_10_coins_l3116_311673

def num_coins : ℕ := 10

-- Probability of getting more heads than tails
def prob_more_heads : ℚ := 193 / 512

theorem prob_more_heads_10_coins : 
  (prob_more_heads : ℚ) = 193 / 512 := by sorry

end NUMINAMATH_CALUDE_prob_more_heads_10_coins_l3116_311673


namespace NUMINAMATH_CALUDE_little_john_sweets_expenditure_l3116_311614

theorem little_john_sweets_expenditure
  (initial_amount : ℚ)
  (final_amount : ℚ)
  (amount_per_friend : ℚ)
  (num_friends : ℕ)
  (h1 : initial_amount = 8.5)
  (h2 : final_amount = 4.85)
  (h3 : amount_per_friend = 1.2)
  (h4 : num_friends = 2) :
  initial_amount - final_amount - (↑num_friends * amount_per_friend) = 1.25 :=
by sorry

end NUMINAMATH_CALUDE_little_john_sweets_expenditure_l3116_311614


namespace NUMINAMATH_CALUDE_erdos_szekeres_theorem_l3116_311650

theorem erdos_szekeres_theorem (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → subseq i < subseq j) ∧
    (∀ i j, i < j → seq (subseq i) ≤ seq (subseq j))) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → subseq i < subseq j) ∧
    (∀ i j, i < j → seq (subseq i) ≥ seq (subseq j))) :=
by sorry

end NUMINAMATH_CALUDE_erdos_szekeres_theorem_l3116_311650


namespace NUMINAMATH_CALUDE_expression_simplification_l3116_311677

theorem expression_simplification (a : ℤ) (n : ℕ) (h : n ≠ 1) :
  (a^(3*n) / (a^n - 1) + 1 / (a^n + 1)) - (a^(2*n) / (a^n + 1) + 1 / (a^n - 1)) = a^(2*n) + 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3116_311677


namespace NUMINAMATH_CALUDE_right_triangle_area_l3116_311671

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) (h_hypotenuse : c = 10) : 
  (1 / 2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3116_311671


namespace NUMINAMATH_CALUDE_apples_added_l3116_311625

theorem apples_added (initial_apples final_apples : ℕ) 
  (h1 : initial_apples = 8)
  (h2 : final_apples = 13) :
  final_apples - initial_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_apples_added_l3116_311625


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_equals_area_l3116_311662

theorem right_triangle_hypotenuse_equals_area 
  (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m ≠ n) : 
  let x : ℝ := (m^2 + n^2) / (m * n * (m^2 - n^2))
  let leg1 : ℝ := (m^2 - n^2) * x
  let leg2 : ℝ := 2 * m * n * x
  let hypotenuse : ℝ := (m^2 + n^2) * x
  let area : ℝ := (1/2) * leg1 * leg2
  hypotenuse = area :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_equals_area_l3116_311662


namespace NUMINAMATH_CALUDE_fixed_stable_points_equality_l3116_311658

def f (a x : ℝ) : ℝ := a * x^2 - 1

def isFixedPoint (a x : ℝ) : Prop := f a x = x

def isStablePoint (a x : ℝ) : Prop := f a (f a x) = x

def fixedPointSet (a : ℝ) : Set ℝ := {x | isFixedPoint a x}

def stablePointSet (a : ℝ) : Set ℝ := {x | isStablePoint a x}

theorem fixed_stable_points_equality (a : ℝ) :
  (fixedPointSet a = stablePointSet a ∧ (fixedPointSet a).Nonempty) ↔ -1/4 ≤ a ∧ a ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_stable_points_equality_l3116_311658


namespace NUMINAMATH_CALUDE_value_of_x_l3116_311623

theorem value_of_x : ∀ (x y z w u : ℤ),
  x = y + 12 →
  y = z + 15 →
  z = w + 25 →
  w = u + 10 →
  u = 95 →
  x = 157 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l3116_311623


namespace NUMINAMATH_CALUDE_min_slope_tangent_l3116_311605

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - 1 / (a * x)

theorem min_slope_tangent (a : ℝ) (h : a > 0) :
  let k := (deriv (f a)) 1
  ∀ b > 0, k ≤ (deriv (f b)) 1 ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_slope_tangent_l3116_311605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3116_311610

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : arithmetic_sequence a)
  (h_sub_sequence : ∃ k : ℝ, a 1 + k = (1/2) * a 3 ∧ (1/2) * a 3 + k = 2 * a 2) :
  (a 8 + a 9) / (a 7 + a 8) = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3116_311610


namespace NUMINAMATH_CALUDE_sparrow_distribution_l3116_311632

theorem sparrow_distribution (total : ℕ) (moved : ℕ) (flew_away : ℕ) :
  total = 25 →
  moved = 5 →
  flew_away = 7 →
  (∃ (initial_first initial_second : ℕ),
    initial_first + initial_second = total ∧
    initial_first - moved = 2 * (initial_second + moved - flew_away) ∧
    initial_first = 17 ∧
    initial_second = 8) :=
by sorry

end NUMINAMATH_CALUDE_sparrow_distribution_l3116_311632


namespace NUMINAMATH_CALUDE_unique_solution_system_l3116_311657

theorem unique_solution_system :
  ∃! (a b c d : ℝ),
    (a * b + c + d = 3) ∧
    (b * c + d + a = 5) ∧
    (c * d + a + b = 2) ∧
    (d * a + b + c = 6) ∧
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3116_311657


namespace NUMINAMATH_CALUDE_man_age_twice_son_age_l3116_311640

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given that the man is currently 24 years older than his son and the son is currently 22 years old. -/
theorem man_age_twice_son_age (son_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  man_age + years = 2 * (son_age + years) →
  years = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_age_twice_son_age_l3116_311640


namespace NUMINAMATH_CALUDE_distinct_colorings_count_l3116_311684

/-- Represents the symmetries of a square -/
inductive SquareSymmetry
| Rotation0 | Rotation90 | Rotation180 | Rotation270
| ReflectionSide1 | ReflectionSide2
| ReflectionDiag1 | ReflectionDiag2

/-- Represents a coloring of the square's disks -/
structure SquareColoring :=
(blue1 : Fin 4)
(blue2 : Fin 4)
(red : Fin 4)
(green : Fin 4)

/-- The group of symmetries of a square -/
def squareSymmetryGroup : List SquareSymmetry :=
[SquareSymmetry.Rotation0, SquareSymmetry.Rotation90, SquareSymmetry.Rotation180, SquareSymmetry.Rotation270,
 SquareSymmetry.ReflectionSide1, SquareSymmetry.ReflectionSide2,
 SquareSymmetry.ReflectionDiag1, SquareSymmetry.ReflectionDiag2]

/-- Checks if a coloring is valid (2 blue, 1 red, 1 green) -/
def isValidColoring (c : SquareColoring) : Bool :=
  c.blue1 ≠ c.blue2 ∧ c.blue1 ≠ c.red ∧ c.blue1 ≠ c.green ∧
  c.blue2 ≠ c.red ∧ c.blue2 ≠ c.green ∧ c.red ≠ c.green

/-- Checks if a coloring is fixed by a given symmetry -/
def isFixedBy (c : SquareColoring) (s : SquareSymmetry) : Bool := sorry

/-- Counts the number of colorings fixed by each symmetry -/
def countFixedColorings (s : SquareSymmetry) : Nat := sorry

/-- The main theorem: there are 3 distinct colorings under symmetry -/
theorem distinct_colorings_count :
  (List.sum (List.map countFixedColorings squareSymmetryGroup)) / squareSymmetryGroup.length = 3 := sorry

end NUMINAMATH_CALUDE_distinct_colorings_count_l3116_311684


namespace NUMINAMATH_CALUDE_sine_cosine_arithmetic_progression_l3116_311649

theorem sine_cosine_arithmetic_progression
  (x y z : ℝ)
  (h_sin_ap : 2 * Real.sin y = Real.sin x + Real.sin z)
  (h_sin_increasing : Real.sin x < Real.sin y ∧ Real.sin y < Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_arithmetic_progression_l3116_311649


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l3116_311613

/-- Calculates the number of unit rods in an n-row triangle -/
def unitRods (n : ℕ) : ℕ := n * (3 + 3 * n) / 2

/-- Calculates the number of connectors in an n-row triangle -/
def connectors (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the total number of pieces in an n-row triangle -/
def totalPieces (n : ℕ) : ℕ := unitRods n + connectors (n + 1)

theorem ten_row_triangle_pieces :
  totalPieces 10 = 231 ∧ unitRods 2 = 9 ∧ connectors 3 = 6 := by sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l3116_311613


namespace NUMINAMATH_CALUDE_scale_length_calculation_l3116_311660

/-- Calculates the total length of a scale given the number of equal parts and the length of each part. -/
def totalScaleLength (numParts : ℕ) (partLength : ℝ) : ℝ :=
  numParts * partLength

/-- Theorem: The total length of a scale with 5 equal parts, each 25 inches long, is 125 inches. -/
theorem scale_length_calculation :
  totalScaleLength 5 25 = 125 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_calculation_l3116_311660


namespace NUMINAMATH_CALUDE_symmetry_point_l3116_311648

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to a horizontal line -/
def isSymmetricHorizontal (p q : Point2D) (y_line : ℝ) : Prop :=
  p.x = q.x ∧ y_line - p.y = q.y - y_line

theorem symmetry_point :
  let p : Point2D := ⟨3, -2⟩
  let q : Point2D := ⟨3, 4⟩
  let y_line : ℝ := 1
  isSymmetricHorizontal p q y_line :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_l3116_311648


namespace NUMINAMATH_CALUDE_abie_chips_bought_l3116_311645

theorem abie_chips_bought (initial_bags : ℕ) (given_away : ℕ) (final_bags : ℕ) 
  (h1 : initial_bags = 20)
  (h2 : given_away = 4)
  (h3 : final_bags = 22) :
  final_bags - (initial_bags - given_away) = 6 :=
by sorry

end NUMINAMATH_CALUDE_abie_chips_bought_l3116_311645


namespace NUMINAMATH_CALUDE_edward_money_theorem_l3116_311621

/-- Calculates Edward's earnings from mowing lawns --/
def lawn_earnings (small medium large : ℕ) : ℕ :=
  8 * small + 12 * medium + 15 * large

/-- Calculates Edward's earnings from cleaning gardens --/
def garden_earnings (gardens : ℕ) : ℕ :=
  if gardens = 0 then 0
  else if gardens = 1 then 10
  else if gardens = 2 then 22
  else 22 + 15 * (gardens - 2)

/-- Calculates Edward's total earnings --/
def total_earnings (small medium large gardens : ℕ) : ℕ :=
  lawn_earnings small medium large + garden_earnings gardens

/-- Calculates Edward's final amount of money --/
def edward_final_money (small medium large gardens savings fuel_cost rental_cost : ℕ) : ℕ :=
  total_earnings small medium large gardens + savings - (fuel_cost + rental_cost)

theorem edward_money_theorem :
  edward_final_money 3 1 1 5 7 10 15 = 100 := by
  sorry

#eval edward_final_money 3 1 1 5 7 10 15

end NUMINAMATH_CALUDE_edward_money_theorem_l3116_311621


namespace NUMINAMATH_CALUDE_expression_evaluation_l3116_311691

theorem expression_evaluation : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3116_311691


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3116_311611

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3116_311611


namespace NUMINAMATH_CALUDE_impossible_conditions_l3116_311606

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xc - xa) * (yb - ya) = (xb - xa) * (yc - ya) ∧  -- collinearity check
  (xb - xa)^2 + (yb - ya)^2 = 144 ∧                -- AB = 12
  (xc - xb) * (xa - xb) + (yc - yb) * (ya - yb) = 0 -- ∠ABC = 90°

-- Define a point inside the triangle
def InsideTriangle (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xp - xa) * (yb - ya) < (xb - xa) * (yp - ya) ∧
  (xp - xb) * (yc - yb) < (xc - xb) * (yp - yb) ∧
  (xp - xc) * (ya - yc) < (xa - xc) * (yp - yc)

-- Define the point D on AC
def PointOnAC (D : ℝ × ℝ) (A C : ℝ × ℝ) : Prop :=
  let (xd, yd) := D
  let (xa, ya) := A
  let (xc, yc) := C
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ xd = xa + t * (xc - xa) ∧ yd = ya + t * (yc - ya)

-- Define P being on BD
def POnBD (P : ℝ × ℝ) (B D : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xb, yb) := B
  let (xd, yd) := D
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ xp = xb + t * (xd - xb) ∧ yp = yb + t * (yd - yb)

-- Define BD > 6√2
def BDGreaterThan6Sqrt2 (B D : ℝ × ℝ) : Prop :=
  let (xb, yb) := B
  let (xd, yd) := D
  (xd - xb)^2 + (yd - yb)^2 > 72

-- Define P above the median of BC
def PAboveMedianBC (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  let xm := (xb + xc) / 2
  let ym := (yb + yc) / 2
  (xp - xa) * (ym - ya) > (xm - xa) * (yp - ya)

theorem impossible_conditions (A B C : ℝ × ℝ) (h : Triangle A B C) :
  ¬∃ (P D : ℝ × ℝ), 
    InsideTriangle P A B C ∧ 
    PointOnAC D A C ∧ 
    POnBD P B D ∧ 
    BDGreaterThan6Sqrt2 B D ∧ 
    PAboveMedianBC P A B C :=
  sorry

end NUMINAMATH_CALUDE_impossible_conditions_l3116_311606


namespace NUMINAMATH_CALUDE_original_price_calculation_l3116_311665

theorem original_price_calculation (initial_price : ℚ) : 
  (initial_price * (1 + 20 / 100) * (1 - 10 / 100) = 2) → 
  (initial_price = 100 / 54) := by
sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3116_311665


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3116_311675

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem tenth_term_of_sequence :
  let a₁ : ℤ := 10
  let d : ℤ := -2
  arithmetic_sequence a₁ d 10 = -8 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3116_311675


namespace NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l3116_311639

theorem modulus_of_complex_reciprocal (z : ℂ) : 
  z = (Complex.I - 1)⁻¹ → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l3116_311639


namespace NUMINAMATH_CALUDE_stuffed_animals_difference_l3116_311630

theorem stuffed_animals_difference (mckenna kenley tenly : ℕ) : 
  mckenna = 34 →
  kenley = 2 * mckenna →
  mckenna + kenley + tenly = 175 →
  tenly - kenley = 5 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_difference_l3116_311630


namespace NUMINAMATH_CALUDE_circles_intersect_iff_l3116_311683

/-- Two circles intersect if and only if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
theorem circles_intersect_iff (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_circles_intersect_iff_l3116_311683


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3116_311635

noncomputable section

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop :=
  (x - Real.sqrt a)^2 + (y - Real.sqrt a)^2 = 9

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def exists_P (a : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, circle_M a P.1 P.2 ∧
    (P.1 - point_A.1) * (point_B.1 - point_A.1) +
    (P.2 - point_A.2) * (point_B.2 - point_A.2) = 0

-- State the theorem
theorem circle_intersection_range :
  ∀ a : ℝ, exists_P a ↔ 1/2 ≤ a ∧ a ≤ 25/2 :=
sorry

end

end NUMINAMATH_CALUDE_circle_intersection_range_l3116_311635


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l3116_311638

theorem arithmetic_mean_greater_than_geometric_mean (x y : ℝ) (hx : x = 16) (hy : y = 64) :
  (x + y) / 2 > Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l3116_311638


namespace NUMINAMATH_CALUDE_reciprocal_equation_l3116_311667

theorem reciprocal_equation (x : ℚ) : 
  2 - 1 / (1 - x) = 2 * (1 / (1 - x)) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l3116_311667


namespace NUMINAMATH_CALUDE_student_selection_probability_l3116_311697

theorem student_selection_probability (n : ℕ) : 
  (4 : ℝ) ≥ 0 ∧ (n : ℝ) ≥ 0 →
  (((n + 4) * (n + 3) / 2 - 6) / ((n + 4) * (n + 3) / 2) = 5 / 6) →
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3116_311697


namespace NUMINAMATH_CALUDE_power_of_power_l3116_311626

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3116_311626


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l3116_311653

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the condition for A' and B' to be on the line y = x
def on_diagonal (p : ℝ × ℝ) : Prop := p.1 = p.2

-- Define the condition for AA' and BB' to intersect at C
def intersect_at_C (A' B' : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    C = (A.1 + t₁ * (A'.1 - A.1), A.2 + t₁ * (A'.2 - A.2)) ∧
    C = (B.1 + t₂ * (B'.1 - B.1), B.2 + t₂ * (B'.2 - B.2))

-- State the theorem
theorem length_of_AB_prime : 
  ∃ A' B' : ℝ × ℝ,
    on_diagonal A' ∧ 
    on_diagonal B' ∧ 
    intersect_at_C A' B' ∧ 
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_prime_l3116_311653


namespace NUMINAMATH_CALUDE_liu_hui_author_of_sea_island_arithmetic_l3116_311631

/-- Represents a mathematical work -/
structure MathWork where
  title : String
  author : String
  significance : Bool
  advance_years : ℕ

/-- The Sea Island Arithmetic -/
def sea_island_arithmetic : MathWork :=
  { title := "The Sea Island Arithmetic"
  , author := "Unknown" -- We'll prove this is Liu Hui
  , significance := true
  , advance_years := 1300 }

/-- Theorem: Liu Hui is the author of The Sea Island Arithmetic -/
theorem liu_hui_author_of_sea_island_arithmetic :
  sea_island_arithmetic.author = "Liu Hui" :=
by
  sorry

#check liu_hui_author_of_sea_island_arithmetic

end NUMINAMATH_CALUDE_liu_hui_author_of_sea_island_arithmetic_l3116_311631


namespace NUMINAMATH_CALUDE_digit_count_theorem_l3116_311608

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

def four_digit_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· ≠ 0)).card * (d.card - 1) * (d.card - 2) * (d.card - 3)

def four_digit_even_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· % 2 = 0)).card * (d.filter (· ≠ 0)).card * (d.card - 2) * (d.card - 3)

def four_digit_div5_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· % 5 = 0)).card * (d.filter (· ≠ 0)).card * (d.card - 2) * (d.card - 3)

theorem digit_count_theorem :
  four_digit_no_repeat digits = 720 ∧
  four_digit_even_no_repeat digits = 420 ∧
  four_digit_div5_no_repeat digits = 220 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_theorem_l3116_311608


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l3116_311698

/-- The inradius of a right triangle with side lengths 6, 8, and 10 is 2 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 6 ∧ b = 8 ∧ c = 10 →  -- Side lengths condition
  a^2 + b^2 = c^2 →         -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l3116_311698


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l3116_311669

theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0 ↔ x^2*(y^2 + 2*y - 5) = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l3116_311669


namespace NUMINAMATH_CALUDE_joe_weight_lifting_ratio_l3116_311601

/-- Joe's weight-lifting competition problem -/
theorem joe_weight_lifting_ratio :
  ∀ (total first second : ℕ),
  total = first + second →
  first = 600 →
  total = 1500 →
  first = 2 * (second - 300) →
  first = second :=
λ total first second h1 h2 h3 h4 =>
  sorry

end NUMINAMATH_CALUDE_joe_weight_lifting_ratio_l3116_311601


namespace NUMINAMATH_CALUDE_scientific_notation_of_55000000_l3116_311682

theorem scientific_notation_of_55000000 :
  55000000 = 5.5 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_55000000_l3116_311682


namespace NUMINAMATH_CALUDE_four_xy_even_l3116_311612

theorem four_xy_even (x y : ℕ) (hx : Even x) (hy : Even y) (hxpos : 0 < x) (hypos : 0 < y) : 
  Even (4 * x * y) := by
  sorry

end NUMINAMATH_CALUDE_four_xy_even_l3116_311612


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3116_311690

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = (x + 1)*(x + 3)) → m - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3116_311690


namespace NUMINAMATH_CALUDE_quadratic_residue_product_l3116_311622

theorem quadratic_residue_product (p a b : ℤ) (hp : Prime p) (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) :
  (∃ x : ℤ, x^2 ≡ a [ZMOD p]) → (∃ y : ℤ, y^2 ≡ b [ZMOD p]) →
  (∃ z : ℤ, z^2 ≡ a * b [ZMOD p]) := by
sorry

end NUMINAMATH_CALUDE_quadratic_residue_product_l3116_311622


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3116_311607

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 7 →                     -- Condition 1
  (a 1 + a 2 + a 3 = 21) →      -- Condition 2 (S_3 = 21)
  q = -0.5 ∨ q = 1 :=           -- Conclusion
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3116_311607


namespace NUMINAMATH_CALUDE_common_ratio_is_three_l3116_311647

/-- Geometric sequence with sum of first n terms -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n, a (n + 1) = (a 1) * (a 2 / a 1) ^ n

/-- The common ratio of a geometric sequence is 3 given specific conditions -/
theorem common_ratio_is_three (seq : GeometricSequence)
  (h1 : seq.a 5 = 2 * seq.S 4 + 3)
  (h2 : seq.a 6 = 2 * seq.S 5 + 3) :
  seq.a 2 / seq.a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_three_l3116_311647


namespace NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l3116_311634

theorem magnitude_of_4_minus_15i :
  let z : ℂ := 4 - 15 * I
  Complex.abs z = Real.sqrt 241 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l3116_311634


namespace NUMINAMATH_CALUDE_cost_of_juices_l3116_311618

/-- The cost of juices and sandwiches problem -/
theorem cost_of_juices (sandwich_cost juice_cost : ℚ) : 
  (2 * sandwich_cost = 6) →
  (sandwich_cost + juice_cost = 5) →
  (5 * juice_cost = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_juices_l3116_311618


namespace NUMINAMATH_CALUDE_quadratic_function_m_not_two_l3116_311600

/-- Given a quadratic function y = a(x-m)^2 where a > 0, 
    if it passes through points (-1,p) and (3,q) where p < q, 
    then m ≠ 2 -/
theorem quadratic_function_m_not_two 
  (a m p q : ℝ) 
  (h1 : a > 0)
  (h2 : a * (-1 - m)^2 = p)
  (h3 : a * (3 - m)^2 = q)
  (h4 : p < q) : 
  m ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_m_not_two_l3116_311600


namespace NUMINAMATH_CALUDE_line_parameterization_l3116_311670

/-- Given a line y = 2x + 5 parameterized as (x, y) = (s, -2) + t(3, m), prove that s = -7/2 and m = 6 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ t : ℝ, ∀ x y : ℝ, x = s + 3*t ∧ y = -2 + m*t → y = 2*x + 5) →
  s = -7/2 ∧ m = 6 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3116_311670


namespace NUMINAMATH_CALUDE_range_of_a_l3116_311685

theorem range_of_a (p q : Prop) (a : ℝ) 
  (hp : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (hq : ∃ x : ℝ, x^2 - 4*x + a ≤ 0) :
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3116_311685


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3116_311641

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3116_311641


namespace NUMINAMATH_CALUDE_smallest_square_with_rook_l3116_311680

/-- Represents a chessboard with rooks -/
structure ChessBoard (n : ℕ) where
  size : ℕ := 3 * n
  rooks : Set (ℕ × ℕ)
  beats_entire_board : ∀ (x y : ℕ), x ≤ size ∧ y ≤ size → 
    ∃ (rx ry : ℕ), (rx, ry) ∈ rooks ∧ (rx = x ∨ ry = y)
  beats_at_most_one : ∀ (r1 r2 : ℕ × ℕ), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 →
    (r1.1 = r2.1 ∧ r1.2 ≠ r2.2) ∨ (r1.1 ≠ r2.1 ∧ r1.2 = r2.2)

/-- The main theorem to be proved -/
theorem smallest_square_with_rook (n : ℕ) (h : n > 0) (board : ChessBoard n) :
  (∀ (k : ℕ), k > 2 * n → 
    ∀ (x y : ℕ), x ≤ board.size - k + 1 → y ≤ board.size - k + 1 →
      ∃ (rx ry : ℕ), (rx, ry) ∈ board.rooks ∧ rx ≥ x ∧ rx < x + k ∧ ry ≥ y ∧ ry < y + k) ∧
  (∃ (x y : ℕ), x ≤ board.size - 2 * n + 1 ∧ y ≤ board.size - 2 * n + 1 ∧
    ∀ (rx ry : ℕ), (rx, ry) ∈ board.rooks → (rx < x ∨ rx ≥ x + 2 * n ∨ ry < y ∨ ry ≥ y + 2 * n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_rook_l3116_311680


namespace NUMINAMATH_CALUDE_equation_solution_l3116_311686

theorem equation_solution :
  ∃ x : ℚ, (x - 60) / 3 = (5 - 3 * x) / 4 ∧ x = 255 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3116_311686


namespace NUMINAMATH_CALUDE_isaac_sleep_time_l3116_311655

-- Define a simple representation of time
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

def Time.isAM (t : Time) : Bool :=
  t.hour < 12

def Time.toPM (t : Time) : Time :=
  if t.isAM then { hour := t.hour + 12, minute := t.minute }
  else t

def Time.fromPM (t : Time) : Time :=
  if t.isAM then t
  else { hour := t.hour - 12, minute := t.minute }

def subtractHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute
  let newTotalMinutes := totalMinutes - h * 60
  let newHour := newTotalMinutes / 60
  let newMinute := newTotalMinutes % 60
  { hour := newHour, minute := newMinute }

theorem isaac_sleep_time (wakeUpTime sleepTime : Time) (sleepDuration : Nat) :
  wakeUpTime = { hour := 7, minute := 0 } →
  sleepDuration = 8 →
  sleepTime = (subtractHours wakeUpTime sleepDuration).toPM →
  sleepTime = { hour := 23, minute := 0 } :=
by sorry

end NUMINAMATH_CALUDE_isaac_sleep_time_l3116_311655


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3116_311629

theorem floor_equation_solution (x : ℝ) : 
  (Int.floor (2 * x) + Int.floor (3 * x) = 8 * x - 7 / 2) ↔ (x = 13 / 16 ∨ x = 17 / 16) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3116_311629


namespace NUMINAMATH_CALUDE_mushroom_drying_l3116_311656

/-- Given an initial mass of mushrooms and moisture contents before and after drying,
    calculate the mass of mushrooms after drying. -/
theorem mushroom_drying (initial_mass : ℝ) (initial_moisture : ℝ) (final_moisture : ℝ) :
  initial_mass = 100 →
  initial_moisture = 99 / 100 →
  final_moisture = 98 / 100 →
  (1 - initial_moisture) * initial_mass / (1 - final_moisture) = 50 := by
  sorry

#check mushroom_drying

end NUMINAMATH_CALUDE_mushroom_drying_l3116_311656


namespace NUMINAMATH_CALUDE_complex_quadrant_l3116_311694

theorem complex_quadrant (a : ℝ) (z : ℂ) : 
  z = a^2 - 3*a - 4 + (a - 4)*Complex.I →
  z.re = 0 →
  (a - a*Complex.I).re < 0 ∧ (a - a*Complex.I).im > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l3116_311694


namespace NUMINAMATH_CALUDE_integral_arctan_fraction_l3116_311603

open Real

theorem integral_arctan_fraction (x : ℝ) :
  deriv (fun x => (1/2) * (4 * (arctan x)^2 - log (1 + x^2))) x
  = (4 * arctan x - x) / (1 + x^2) :=
by sorry

end NUMINAMATH_CALUDE_integral_arctan_fraction_l3116_311603


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3116_311643

theorem quadratic_roots_product (b c : ℝ) :
  (∀ x, x^2 + b*x + c = 0 → ∃ y, y^2 + b*y + c = 0 ∧ x * y = 20) →
  c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3116_311643


namespace NUMINAMATH_CALUDE_smallest_m_theorem_l3116_311604

/-- The smallest positive value of m for which the equation 12x^2 - mx - 360 = 0 has integral solutions -/
def smallest_m : ℕ := 12

/-- The equation 12x^2 - mx - 360 = 0 has integral solutions -/
def has_integral_solutions (m : ℤ) : Prop :=
  ∃ x : ℤ, 12 * x^2 - m * x - 360 = 0

/-- The theorem stating that the smallest positive m for which the equation has integral solutions is 12 -/
theorem smallest_m_theorem : 
  (∀ m : ℕ, m < smallest_m → ¬(has_integral_solutions m)) ∧ 
  (has_integral_solutions smallest_m) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_theorem_l3116_311604


namespace NUMINAMATH_CALUDE_x_value_l3116_311627

theorem x_value : ∃ x : ℝ, x = 80 * (1 + 0.12) ∧ x = 89.6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3116_311627


namespace NUMINAMATH_CALUDE_fraction_of_length_equality_l3116_311652

theorem fraction_of_length_equality : (2 / 7 : ℚ) * 3 = (3 / 7 : ℚ) * 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_length_equality_l3116_311652


namespace NUMINAMATH_CALUDE_negative_power_product_l3116_311681

theorem negative_power_product (x : ℝ) : -x^2 * x = -x^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_power_product_l3116_311681


namespace NUMINAMATH_CALUDE_log3_negative_implies_x_negative_but_not_conversely_l3116_311633

-- Define the logarithm function with base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Theorem statement
theorem log3_negative_implies_x_negative_but_not_conversely :
  (∀ x : ℝ, log3 (x + 1) < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ log3 (x + 1) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_log3_negative_implies_x_negative_but_not_conversely_l3116_311633


namespace NUMINAMATH_CALUDE_coefficient_x5_proof_l3116_311661

/-- The coefficient of x^5 in the expansion of (1+x^3)(1-2x)^6 -/
def coefficient_x5 : ℤ := -132

/-- The expansion of (1+x^3)(1-2x)^6 -/
def expansion (x : ℝ) : ℝ := (1 + x^3) * (1 - 2*x)^6

theorem coefficient_x5_proof : 
  (deriv^[5] expansion 0) / 120 = coefficient_x5 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5_proof_l3116_311661


namespace NUMINAMATH_CALUDE_danielle_spending_l3116_311699

/-- Represents the cost and yield of supplies for making popsicles. -/
structure PopsicleSupplies where
  mold_cost : ℕ
  stick_pack_cost : ℕ
  stick_pack_size : ℕ
  juice_bottle_cost : ℕ
  popsicles_per_bottle : ℕ
  remaining_sticks : ℕ

/-- Calculates the total cost of supplies for making popsicles. -/
def total_cost (supplies : PopsicleSupplies) : ℕ :=
  supplies.mold_cost + supplies.stick_pack_cost +
  (supplies.stick_pack_size - supplies.remaining_sticks) / supplies.popsicles_per_bottle * supplies.juice_bottle_cost

/-- Theorem stating that Danielle's total spending on supplies equals $10. -/
theorem danielle_spending (supplies : PopsicleSupplies)
  (h1 : supplies.mold_cost = 3)
  (h2 : supplies.stick_pack_cost = 1)
  (h3 : supplies.stick_pack_size = 100)
  (h4 : supplies.juice_bottle_cost = 2)
  (h5 : supplies.popsicles_per_bottle = 20)
  (h6 : supplies.remaining_sticks = 40) :
  total_cost supplies = 10 := by
    sorry

end NUMINAMATH_CALUDE_danielle_spending_l3116_311699


namespace NUMINAMATH_CALUDE_unique_intersection_l3116_311663

/-- The value of k for which the line x = k intersects the parabola x = -y^2 - 4y + 2 at exactly one point -/
def intersection_k : ℝ := 6

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -y^2 - 4*y + 2

theorem unique_intersection :
  ∀ k : ℝ, (∃! y : ℝ, k = parabola y) ↔ k = intersection_k :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_l3116_311663


namespace NUMINAMATH_CALUDE_difference_of_squares_l3116_311644

theorem difference_of_squares : 55^2 - 45^2 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3116_311644


namespace NUMINAMATH_CALUDE_workers_total_earning_approx_1480_l3116_311692

/-- Represents the daily wage and work days of a worker -/
structure Worker where
  dailyWage : ℝ
  workDays : ℕ

/-- Calculates the total earning of a worker -/
def totalEarning (w : Worker) : ℝ :=
  w.dailyWage * w.workDays

/-- Theorem stating that the total earning of the three workers is approximately 1480 -/
theorem workers_total_earning_approx_1480 
  (a b c : Worker)
  (h_a_days : a.workDays = 16)
  (h_b_days : b.workDays = 9)
  (h_c_days : c.workDays = 4)
  (h_c_wage : c.dailyWage = 71.15384615384615)
  (h_wage_ratio : a.dailyWage / c.dailyWage = 3 / 5 ∧ b.dailyWage / c.dailyWage = 4 / 5) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    abs ((totalEarning a + totalEarning b + totalEarning c) - 1480) < ε :=
sorry

end NUMINAMATH_CALUDE_workers_total_earning_approx_1480_l3116_311692


namespace NUMINAMATH_CALUDE_eggs_left_is_five_l3116_311615

-- Define the problem parameters
def total_eggs : ℕ := 30
def total_cost : ℕ := 500  -- in cents
def price_per_egg : ℕ := 20  -- in cents

-- Define the function to calculate eggs left after recovering capital
def eggs_left_after_recovery : ℕ :=
  total_eggs - (total_cost / price_per_egg)

-- Theorem statement
theorem eggs_left_is_five : eggs_left_after_recovery = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_is_five_l3116_311615


namespace NUMINAMATH_CALUDE_log_expression_equality_l3116_311620

theorem log_expression_equality (a b : ℝ) (ha : a = Real.log 8) (hb : b = Real.log 25) :
  5^(a/b) + 2^(b/a) = Real.sqrt 8 + 5^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l3116_311620


namespace NUMINAMATH_CALUDE_marions_bike_cost_l3116_311689

theorem marions_bike_cost (marion_cost stephanie_cost total : ℕ) : 
  stephanie_cost = 2 * marion_cost →
  total = marion_cost + stephanie_cost →
  total = 1068 →
  marion_cost = 356 := by
sorry

end NUMINAMATH_CALUDE_marions_bike_cost_l3116_311689


namespace NUMINAMATH_CALUDE_friends_pen_cost_l3116_311666

def robertPens : ℕ := 4
def juliaPens : ℕ := 3 * robertPens
def dorothyPens : ℕ := juliaPens / 2
def penCost : ℚ := 3/2

def totalPens : ℕ := robertPens + juliaPens + dorothyPens
def totalCost : ℚ := (totalPens : ℚ) * penCost

theorem friends_pen_cost : totalCost = 33 := by sorry

end NUMINAMATH_CALUDE_friends_pen_cost_l3116_311666


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l3116_311609

/-- Given three positive integers in the ratio 2:3:5 with LCM 180, prove their sum is 60 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a * 3 = b * 2 →
  a * 5 = c * 2 →
  Nat.lcm (Nat.lcm a b) c = 180 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l3116_311609


namespace NUMINAMATH_CALUDE_mysoon_ornament_collection_l3116_311624

/-- The number of ornaments in Mysoon's collection -/
def total_ornaments : ℕ := 20

/-- The number of handmade ornaments -/
def handmade_ornaments : ℕ := total_ornaments / 6 + 10

/-- The number of handmade antique ornaments -/
def handmade_antique_ornaments : ℕ := total_ornaments / 3

theorem mysoon_ornament_collection :
  (handmade_ornaments = total_ornaments / 6 + 10) ∧
  (handmade_antique_ornaments = handmade_ornaments / 2) ∧
  (handmade_antique_ornaments = total_ornaments / 3) →
  total_ornaments = 20 := by
sorry

end NUMINAMATH_CALUDE_mysoon_ornament_collection_l3116_311624


namespace NUMINAMATH_CALUDE_percent_of_y_l3116_311695

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 5 + (3 * y) / 10) / y * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3116_311695


namespace NUMINAMATH_CALUDE_pauls_hourly_wage_l3116_311651

/-- Calculates the hourly wage given the number of hours worked, tax rate, expense rate, and remaining money. -/
def calculate_hourly_wage (hours_worked : ℕ) (tax_rate : ℚ) (expense_rate : ℚ) (remaining_money : ℚ) : ℚ :=
  remaining_money / ((1 - expense_rate) * ((1 - tax_rate) * hours_worked))

/-- Theorem stating that under the given conditions, the hourly wage is $12.50 -/
theorem pauls_hourly_wage :
  let hours_worked : ℕ := 40
  let tax_rate : ℚ := 1/5
  let expense_rate : ℚ := 3/20
  let remaining_money : ℚ := 340
  calculate_hourly_wage hours_worked tax_rate expense_rate remaining_money = 25/2 := by
  sorry


end NUMINAMATH_CALUDE_pauls_hourly_wage_l3116_311651
