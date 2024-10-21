import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1132_113227

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Definition of a point being on the ellipse C -/
def on_ellipse_C (x y : ℝ) : Prop :=
  ellipse_C x y

/-- Definition of the line l passing through (1, 0) -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - 1)

/-- Definition of the area of triangle AOB -/
noncomputable def area_AOB (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (1/2) * abs (x₁ * y₂ - x₂ * y₁)

/-- Main theorem -/
theorem ellipse_and_line_intersection :
  (∀ x y, on_ellipse_C x y → ellipse_C x y) ∧
  on_ellipse_C (Real.sqrt 3) (1/2) ∧
  (∃ m x₁ y₁ x₂ y₂,
    x₁ ≠ x₂ ∧
    on_ellipse_C x₁ y₁ ∧
    on_ellipse_C x₂ y₂ ∧
    line_l m x₁ y₁ ∧
    line_l m x₂ y₂ ∧
    area_AOB x₁ y₁ x₂ y₂ = 4/5 →
    (x₁ + y₁ = 1 ∧ x₂ + y₂ = 1) ∨ (x₁ - y₁ = 1 ∧ x₂ - y₂ = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l1132_113227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_line_length_squared_l1132_113238

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  distance t.A t.B = distance t.B t.C ∧ distance t.B t.C = distance t.C t.A

/-- The fold line in the triangle -/
noncomputable def foldLine (D E F : Point) (touchPoint : Point) : ℝ :=
  let P := Point.mk 0 0  -- Placeholder for the fold point on DE
  let Q := Point.mk 0 0  -- Placeholder for the fold point on DF
  distance P Q

/-- The main theorem -/
theorem fold_line_length_squared 
  (D E F : Point) 
  (touchPoint : Point) :
  isEquilateral (Triangle.mk D E F) →
  distance D E = 15 →
  distance E touchPoint = 11 →
  (foldLine D E F touchPoint)^2 = 112225 / 1225 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_line_length_squared_l1132_113238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_locus_is_circle_l1132_113223

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane, represented by a point it passes through and an angle -/
structure Line2D where
  point : Point2D
  angle : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A circle in a 2D plane -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- The set of points symmetric to a given point with respect to a rotating line -/
def symmetricPointSet (A B : Point2D) : Set Point2D :=
  { p | ∃ (l : Line2D), l.point = B ∧ distance p B = distance A B }

/-- Membership in a circle -/
def inCircle (p : Point2D) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- The theorem stating that the geometric locus is a circle -/
theorem geometric_locus_is_circle (A B : Point2D) :
  symmetricPointSet A B = { p | inCircle p (Circle.mk B (distance A B)) } :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_locus_is_circle_l1132_113223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_a_l1132_113222

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- Define a value for x ≤ 0 to make the function total

-- State the theorem
theorem f_inverse_a (a : ℝ) : f a = f (a + 1) → f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_a_l1132_113222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_average_weight_l1132_113226

noncomputable def average_weight_of_dogs (brown_weight black_weight white_weight grey_weight : ℚ) : ℚ :=
  (brown_weight + black_weight + white_weight + grey_weight) / 4

theorem dogs_average_weight :
  ∀ (brown_weight : ℚ),
    brown_weight = 4 →
    ∀ (black_weight : ℚ),
      black_weight = brown_weight + 1 →
      ∀ (white_weight : ℚ),
        white_weight = 2 * brown_weight →
        ∀ (grey_weight : ℚ),
          grey_weight = black_weight - 2 →
          average_weight_of_dogs brown_weight black_weight white_weight grey_weight = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dogs_average_weight_l1132_113226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x6_is_zero_l1132_113299

/-- The expression to be expanded -/
noncomputable def f (x : ℝ) : ℝ := (x^3 / 3 - 3 / x^2)^9

/-- The coefficient of x^6 in the expansion of f(x) -/
noncomputable def coeff_x6 (f : ℝ → ℝ) : ℝ :=
  (deriv^[6] f 0) / Nat.factorial 6

theorem coeff_x6_is_zero : coeff_x6 f = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x6_is_zero_l1132_113299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_lateral_area_ratio_l1132_113294

theorem cone_cylinder_lateral_area_ratio 
  (r h : ℝ) 
  (hr : r > 0) 
  (hh : h > 0) 
  (cone_axis_equilateral : h = r * Real.sqrt 3) : 
  (2 * Real.pi * r^2) / (2 * Real.pi * r * h) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_lateral_area_ratio_l1132_113294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_ratio_prime_l1132_113205

def my_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 2 - Nat.gcd (my_sequence n) (n + 1)) * my_sequence n

theorem my_sequence_ratio_prime (n : ℕ) :
  n ≠ 0 → (my_sequence (n + 1) / my_sequence n = n ↔ Nat.Prime n ∨ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_ratio_prime_l1132_113205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_time_l1132_113221

/-- The time (in seconds) it takes for a train to overtake a motorbike -/
noncomputable def overtakeTime (trainSpeed : ℝ) (motorbikeSpeed : ℝ) (trainLength : ℝ) : ℝ :=
  trainLength / ((trainSpeed - motorbikeSpeed) * (1000 / 3600))

/-- Theorem stating the time it takes for the train to overtake the motorbike -/
theorem train_overtake_time :
  let trainSpeed : ℝ := 100
  let motorbikeSpeed : ℝ := 64
  let trainLength : ℝ := 120.0096
  overtakeTime trainSpeed motorbikeSpeed trainLength = 12.00096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_overtake_time_l1132_113221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_per_1_5_mile_is_one_dollar_l1132_113287

/-- Represents the fare structure for a taxi ride -/
structure TaxiFare where
  initialFare : ℚ  -- Initial fare for the first 1/5 mile
  totalFare : ℚ    -- Total fare for a 10-mile ride
  totalDistance : ℚ -- Total distance of the ride in miles

/-- Calculates the fare for each 1/5 mile after the first -/
def farePer1_5Mile (tf : TaxiFare) : ℚ :=
  (tf.totalFare - tf.initialFare) / ((tf.totalDistance * 5) - 1)

/-- Theorem stating that given the specified conditions, 
    the fare for each 1/5 mile after the first is $1.00 -/
theorem fare_per_1_5_mile_is_one_dollar 
  (tf : TaxiFare) 
  (h1 : tf.initialFare = 10)
  (h2 : tf.totalFare = 59)
  (h3 : tf.totalDistance = 10) : 
  farePer1_5Mile tf = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fare_per_1_5_mile_is_one_dollar_l1132_113287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_quadrants_II_III_IV_l1132_113240

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 2

-- Define the condition on a
def a_condition (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Define what it means for a point to be in each quadrant
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem f_passes_through_quadrants_II_III_IV (a : ℝ) (h : a_condition a) :
  (∃ x y : ℝ, y = f a x ∧ in_quadrant_II x y) ∧
  (∃ x y : ℝ, y = f a x ∧ in_quadrant_III x y) ∧
  (∃ x y : ℝ, y = f a x ∧ in_quadrant_IV x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_quadrants_II_III_IV_l1132_113240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_terminates_final_state_unique_l1132_113237

/-- Represents the state of the game board -/
structure BoardState where
  a : ℕ
  b : ℕ
  c : ℕ
  deriving Repr

/-- Represents a single move in the game -/
def move (s : BoardState) : Option BoardState :=
  if s.a > Nat.gcd s.b s.c then
    some ⟨s.a - Nat.gcd s.b s.c, s.b, s.c⟩
  else if s.b > Nat.gcd s.a s.c then
    some ⟨s.a, s.b - Nat.gcd s.a s.c, s.c⟩
  else if s.c > Nat.gcd s.a s.b then
    some ⟨s.a, s.b, s.c - Nat.gcd s.a s.b⟩
  else
    none

/-- The game terminates for any initial board state -/
theorem game_terminates (initial : BoardState) : 
  ∃ (n : ℕ) (final : BoardState), (Nat.iterate (Option.getD initial) n (move initial) = final) ∧ (move final = none) :=
sorry

/-- The final state is uniquely determined by the initial state -/
theorem final_state_unique (initial : BoardState) 
  (n1 n2 : ℕ) (final1 final2 : BoardState) :
  (Nat.iterate (Option.getD initial) n1 (move initial) = final1) ∧ (move final1 = none) →
  (Nat.iterate (Option.getD initial) n2 (move initial) = final2) ∧ (move final2 = none) →
  final1 = final2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_terminates_final_state_unique_l1132_113237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celtics_win_in_seven_l1132_113212

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 1/4

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 3/4

/-- The number of games in the series -/
def n_games : ℕ := 7

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

theorem celtics_win_in_seven (h1 : p_celtics + p_lakers = 1) 
  (h2 : p_celtics > 0) (h3 : p_lakers > 0) : 
  (Nat.choose 6 3 : ℚ) * p_celtics^4 * p_lakers^3 = 27/16384 := by
  sorry

#check celtics_win_in_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celtics_win_in_seven_l1132_113212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_power_sum_divisible_l1132_113285

theorem consecutive_odd_power_sum_divisible (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hconsec : q = p + 2) : 
  (p + q) ∣ (p^p + q^q) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_power_sum_divisible_l1132_113285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_repeating_decimal_l1132_113286

/-- The reciprocal of the common fraction form of 0.363636... is 11/4 -/
theorem reciprocal_of_repeating_decimal : 
  (1 / (4 / 11 : ℚ)) = 11/4 := by
  -- Convert the repeating decimal 0.363636... to its fractional form 4/11
  -- Then calculate its reciprocal
  norm_num  -- This tactic should simplify the rational arithmetic
  -- The proof is completed by norm_num, but we can add sorry if needed
  -- sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_repeating_decimal_l1132_113286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_m_value_l1132_113232

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2*x - 2) / (x + 2)

-- Define the theorem
theorem f_monotone_and_m_value :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∃ m : ℝ, m > 1 ∧ f m - f 1 = 1/2 ∧ m = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_m_value_l1132_113232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_11pi_over_2_l1132_113278

theorem cos_alpha_minus_11pi_over_2 (α : ℝ) 
  (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.cos (α - 9*π) = -3/5) : 
  Real.cos (α - 11*π/2) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_11pi_over_2_l1132_113278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_axis_l1132_113224

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)
noncomputable def g (x : ℝ) : ℝ := 2^(1 - x)

theorem symmetry_about_y_axis : 
  ∀ x : ℝ, f (-x) = g x :=
by
  intro x
  simp [f, g]
  sorry

#check symmetry_about_y_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_axis_l1132_113224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_triangle_l1132_113208

/-- Triangle ABC with side lengths a, b, c --/
structure Triangle :=
  (a b c : ℝ)

/-- Four circles with radius r inside triangle ABC --/
structure Circles :=
  (r : ℝ)

/-- Tangency relation between a circle and a line segment or another circle --/
def Tangent (A B : Set ℝ) : Prop := sorry

/-- The configuration of the triangle and circles --/
def Configuration (t : Triangle) (c : Circles) : Prop :=
  t.a = 13 ∧ t.b = 14 ∧ t.c = 15 ∧
  ∃ (O O1 O2 O3 : Set ℝ), 
    (∀ x, x ∈ O ∪ O1 ∪ O2 ∪ O3 → x ∈ Set.Icc 0 1) ∧ 
    (Tangent O1 (Set.Icc 0 t.a) ∧ Tangent O1 (Set.Icc 0 t.b)) ∧
    (Tangent O2 (Set.Icc 0 t.b) ∧ Tangent O2 (Set.Icc 0 t.c)) ∧
    (Tangent O3 (Set.Icc 0 t.c) ∧ Tangent O3 (Set.Icc 0 t.a)) ∧
    Tangent O O1 ∧ Tangent O O2 ∧ Tangent O O3

/-- The main theorem --/
theorem circle_radius_in_triangle (t : Triangle) (c : Circles) 
  (h : Configuration t c) : c.r = 260 / 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_triangle_l1132_113208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skipping_rates_relationship_l1132_113211

/-- Represents the number of skips per minute for Student A -/
def x : ℝ := sorry

/-- The equation representing the relationship between Student A and Student B's skipping rates -/
def skipping_equation (x : ℝ) : Prop :=
  180 / x = 240 / (x + 5)

/-- Theorem stating that the skipping_equation correctly represents the relationship
    between Student A and Student B's skipping rates -/
theorem skipping_rates_relationship (x : ℝ) :
  skipping_equation x ↔
    (180 / x = 240 / (x + 5) ∧
     x > 0 ∧
     x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skipping_rates_relationship_l1132_113211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_jellybean_count_l1132_113255

/-- Represents a cubic box of jellybeans -/
structure JellyboxCube where
  side : ℚ
  capacity : ℕ

/-- Represents a larger box of jellybeans -/
structure JellyboxLarge where
  side : ℚ
  capacity : ℕ

def bert_box : JellyboxCube :=
  { side := 6,
    capacity := 216 }

def carrie_box (bert : JellyboxCube) : JellyboxLarge :=
  { side := 3 * bert.side,
    capacity := (⌊1.1 * (3 * bert.side) ^ 3⌋).toNat }

theorem carrie_jellybean_count :
  (carrie_box bert_box).capacity = 6415 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_jellybean_count_l1132_113255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l1132_113229

-- Define cot function
noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan x

theorem simplify_trig_expression :
  (Real.tan (π / 3))^3 + (cot (π / 3))^3 = -1/3 * (Real.tan (π / 3) + cot (π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l1132_113229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_given_difference_l1132_113217

theorem sin_double_angle_given_difference (x : ℝ) : 
  Real.sin (π/4 - x) = 3/5 → Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_given_difference_l1132_113217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_nine_l1132_113249

theorem not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ (7^n + n^3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_nine_l1132_113249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_decaf_is_30_percent_l1132_113261

/-- Represents the coffee stock problem --/
structure CoffeeStock where
  initial_stock : ℚ
  additional_purchase : ℚ
  decaf_percent_additional : ℚ
  final_decaf_percent : ℚ

/-- Calculates the percentage of decaffeinated coffee in the initial stock --/
def initial_decaf_percent (cs : CoffeeStock) : ℚ :=
  (cs.final_decaf_percent * (cs.initial_stock + cs.additional_purchase) -
   cs.decaf_percent_additional * cs.additional_purchase) / cs.initial_stock * 100

/-- Theorem stating that the initial decaffeinated percentage is 30% --/
theorem initial_decaf_is_30_percent (cs : CoffeeStock)
  (h1 : cs.initial_stock = 400)
  (h2 : cs.additional_purchase = 100)
  (h3 : cs.decaf_percent_additional = 6/10)
  (h4 : cs.final_decaf_percent = 36/100) :
  initial_decaf_percent cs = 30 := by
  sorry

#eval initial_decaf_percent ⟨400, 100, 6/10, 36/100⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_decaf_is_30_percent_l1132_113261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_real_roots_l1132_113215

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else -x^2 + (5/2) * x

-- Theorem statement
theorem two_real_roots :
  ∃ (a b : ℝ), a ≠ b ∧ f a = (1/2) * a + 1 ∧ f b = (1/2) * b + 1 ∧
  ∀ (x : ℝ), f x = (1/2) * x + 1 → x = a ∨ x = b :=
by
  sorry

#check two_real_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_real_roots_l1132_113215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tax_rate_approx_l1132_113201

-- Define the original sales tax rate
noncomputable def original_tax_rate : ℝ := 3.5 / 100

-- Define the market price
noncomputable def market_price : ℝ := 9000

-- Define the difference in tax amount
noncomputable def tax_difference : ℝ := 14.999999999999986

-- Define the new tax rate calculation
noncomputable def new_tax_rate : ℝ := 
  (original_tax_rate * market_price + tax_difference) / market_price

-- Theorem statement
theorem new_tax_rate_approx :
  ∃ ε > 0, |new_tax_rate - 0.0367| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tax_rate_approx_l1132_113201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_height_problem_l1132_113231

/-- Represents a right circular cylinder -/
structure Cylinder where
  circumference : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  (c.circumference ^ 2 * c.height) / (4 * Real.pi)

theorem tank_height_problem (tankM tankB : Cylinder) 
  (hM_circ : tankM.circumference = 8)
  (hB_circ : tankB.circumference = 10)
  (hB_height : tankB.height = 8)
  (hVolume : cylinderVolume tankM = 0.8 * cylinderVolume tankB) :
  tankM.height = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_height_problem_l1132_113231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_product_l1132_113271

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (2, 4)

-- Define a line through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 2

-- Define the intersection points of the line with the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | parabola x y ∧ line_through_focus m x y}

-- Define the x-intercepts of lines AP and AQ
def x_intercepts (P Q : ℝ × ℝ) : Set ℝ :=
  {x | ∃ y, (x = -P.2/2 ∨ x = -Q.2/2) ∧ y = 0}

-- The theorem to prove
theorem parabola_intersection_product :
  ∀ m : ℝ,
  ∀ P Q : ℝ × ℝ,
  P ∈ intersection_points m →
  Q ∈ intersection_points m →
  P ≠ Q →
  P ≠ A →
  Q ≠ A →
  ∀ x₁ x₂, x₁ ∈ x_intercepts P Q → x₂ ∈ x_intercepts P Q →
  |x₁| * |x₂| = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_product_l1132_113271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_for_specific_frame_l1132_113253

/-- Calculates the amount of paint needed for a picture frame. -/
noncomputable def paint_needed (frame_width frame_height frame_thickness : ℝ) (paint_coverage : ℝ) : ℝ :=
  let front_area := frame_width * frame_height - (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness)
  let edge_area := 2 * (frame_width * frame_thickness + frame_height * frame_thickness)
  let total_area := front_area + edge_area
  total_area / paint_coverage

/-- Theorem stating the amount of paint needed for the given frame dimensions. -/
theorem paint_needed_for_specific_frame :
  paint_needed 6 9 1 5 = 11.2 := by
  -- Unfold the definition of paint_needed
  unfold paint_needed
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_needed_for_specific_frame_l1132_113253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_ordered_pairs_count_l1132_113256

theorem distinct_ordered_pairs_count : 
  let S := {(x, y) : ℤ × ℤ | 1 ≤ x ∧ x ≤ 4 ∧ x^4 * y^4 - 18 * x^2 * y^2 + 81 = 0}
  Finset.card (Finset.filter (fun (x, y) => 1 ≤ x ∧ x ≤ 4 ∧ x^4 * y^4 - 18 * x^2 * y^2 + 81 = 0) (Finset.range 5 ×ˢ Finset.range 10)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_ordered_pairs_count_l1132_113256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_radius_correct_l1132_113243

/-- The radius of a spherical bowling ball -/
noncomputable def bowling_ball_radius (weight : ℝ) (density : ℝ) : ℝ :=
  (40 / Real.pi) ^ (1/3)

/-- Theorem: The radius of a spherical bowling ball with given weight and density -/
theorem bowling_ball_radius_correct (weight density : ℝ) 
  (h_weight : weight = 16) 
  (h_density : density = 0.3) : 
  bowling_ball_radius weight density = (40 / Real.pi) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_radius_correct_l1132_113243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_l1132_113245

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_k (k : ℕ) : 
  arithmetic_sum 1 2 (k + 2) = 28 + arithmetic_sum 1 2 k → k = 6 := by
  sorry

#check arithmetic_sequence_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_k_l1132_113245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l1132_113234

def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y ∨ (∀ x y : ℝ, x < y → f a x > f a y)) →
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l1132_113234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_product_l1132_113265

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that AM * AN = CO * CD for a chord CD parallel to AN --/
theorem ellipse_chord_product (a b : ℝ) (h : a > b ∧ b > 0) :
  ∀ (C D N M : ℝ × ℝ),
    let O : ℝ × ℝ := (0, 0)
    let A : ℝ × ℝ := (-a, 0)
    -- C and D are on the ellipse
    (C.1^2 / a^2 + C.2^2 / b^2 = 1) →
    (D.1^2 / a^2 + D.2^2 / b^2 = 1) →
    -- CD is a diameter
    (C.1 = -D.1 ∧ C.2 = -D.2) →
    -- N is on the ellipse
    (N.1^2 / a^2 + N.2^2 / b^2 = 1) →
    -- M is on the minor axis
    (M.1 = 0) →
    -- AN is parallel to CD
    ((N.2 - A.2) / (N.1 - A.1) = (D.2 - C.2) / (D.1 - C.1)) →
    -- Conclusion
    (dist A M * dist A N = dist C O * dist C D) := by
  sorry

/-- Helper function to calculate Euclidean distance --/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_product_l1132_113265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_specific_geometric_series_sum_l1132_113203

theorem geometric_series_sum (a₀ r : ℚ) (n : ℕ) (h : r ≠ 1) :
  (Finset.range n).sum (λ i => a₀ * r ^ i) = a₀ * (1 - r ^ n) / (1 - r) := by sorry

theorem specific_geometric_series_sum :
  (Finset.range 5).sum (λ i => (1 / 4 : ℚ) ^ (i + 1)) = 341 / 1024 := by
  have h : (1 / 4 : ℚ) ≠ 1 := by norm_num
  have eq := geometric_series_sum (1 / 4) (1 / 4) 5 h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_specific_geometric_series_sum_l1132_113203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_uniqueness_l1132_113264

-- Define the quadratic function f(x)
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_uniqueness (a b c : ℝ) :
  (f a b c 0 = 1) ∧ 
  (∀ x, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c = λ x ↦ x^2 - x + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_uniqueness_l1132_113264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_piece_end_condition_l1132_113262

/-- Represents a move in the chessboard game -/
inductive Move
  | horizontal : ℕ → ℕ → ℕ → Move
  | vertical : ℕ → ℕ → ℕ → Move

/-- Represents the state of the chessboard -/
def Board := ℕ → ℕ → Bool

/-- Check if a move is valid on a given board -/
def is_valid_move (b : Board) (m : Move) : Bool := sorry

/-- Apply a move to a board -/
def apply_move (b : Board) (m : Move) : Board := sorry

/-- Check if a board has only one piece remaining -/
def has_one_piece (b : Board) : Bool := sorry

/-- Generate the initial n × n board -/
def initial_board (n : ℕ) : Board := sorry

/-- Main theorem: The game can end with one piece iff n is not divisible by 3 -/
theorem one_piece_end_condition (n : ℕ) :
  (∃ (moves : List Move), has_one_piece (moves.foldl apply_move (initial_board n)) ∧ 
    moves.all (is_valid_move (initial_board n))) ↔ ¬(n % 3 = 0) := by
  sorry

#check one_piece_end_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_piece_end_condition_l1132_113262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1132_113206

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 - 2*x + 3) / Real.log (1/2)

-- State the theorem
theorem f_monotone_increasing :
  ∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
    x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1132_113206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l1132_113284

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def SpecialTriangle (t : Triangle) : Prop :=
  t.A = Real.pi/4 ∧ 
  t.b^2 - t.a^2 = (1/4) * t.c^2 ∧
  (1/2) * t.b * t.c * Real.sin t.A = 5/2

-- State the theorem
theorem special_triangle_properties (t : Triangle) 
  (h : SpecialTriangle t) : Real.tan t.C = 4 ∧ t.b = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l1132_113284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_hyperbola_equation_l1132_113297

/-- An ellipse with given properties -/
structure Ellipse where
  P : ℝ × ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  dist1 : ℝ
  dist2 : ℝ
  major_axis : ℝ × ℝ → ℝ × ℝ → Prop
  perpendicular_line : ℝ × ℝ → ℝ × ℝ → Prop

/-- The conditions of the problem -/
def ellipse_conditions (e : Ellipse) : Prop :=
  e.dist1 = 4 * Real.sqrt 5 / 3 ∧
  e.dist2 = 2 * Real.sqrt 5 / 3 ∧
  (∃ (f : ℝ × ℝ), (f = e.focus1 ∨ f = e.focus2) ∧ e.perpendicular_line e.P f)

/-- The theorem to be proved -/
theorem ellipse_equation (e : Ellipse) (h : ellipse_conditions e) :
  (∀ x y : ℝ, (x^2 / 5 + y^2 / (10/3) = 1) ∨ (x^2 / (10/3) + y^2 / 5 = 1)) :=
sorry

/-- A hyperbola with given properties -/
structure Hyperbola where
  vertex : ℝ × ℝ
  focal_distance : ℝ
  real_axis_length : ℝ

/-- The conditions of the hyperbola problem -/
def hyperbola_conditions (h : Hyperbola) : Prop :=
  h.vertex = (0, 2) ∧
  h.focal_distance = Real.sqrt 5 * h.real_axis_length

/-- The theorem for the hyperbola equation -/
theorem hyperbola_equation (h : Hyperbola) (cond : hyperbola_conditions h) :
  ∀ x y : ℝ, y^2 / 4 - x^2 / 16 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_hyperbola_equation_l1132_113297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1132_113272

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2
  sum_angles : A + B + C = Real.pi

-- Define the specific triangle from the problem
noncomputable def triangle : AcuteTriangle where
  A := Real.arcsin ((3 * Real.sqrt 10) / 10)  -- We'll prove this
  B := Real.arcsin ((2 * Real.sqrt 5) / 5)
  C := Real.pi/4  -- 45°
  a := sorry  -- We don't know the value of a
  b := 4 * Real.sqrt 5
  c := 5 * Real.sqrt 2
  acute := by sorry
  sum_angles := by sorry

-- Theorem to prove
theorem triangle_properties (t : AcuteTriangle) (h : t = triangle) :
  t.c = 5 * Real.sqrt 2 ∧ Real.sin t.A = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1132_113272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_rational_approximation_l1132_113298

theorem pi_rational_approximation (p q : ℕ) (hq : q > 1) :
  |Real.pi - (p : ℝ) / (q : ℝ)| ≥ (q : ℝ)⁻¹^(42 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_rational_approximation_l1132_113298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_matching_l1132_113248

-- Define the parabola
noncomputable def parabola (a : ℝ) (x y : ℝ) : Prop := x^2 = a * y

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 2

-- Define the focus of the parabola
noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ := (0, a / 4)

-- Define the foci of the hyperbola
def hyperbola_foci : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Theorem statement
theorem focus_matching (a : ℝ) :
  (∃ (f : ℝ × ℝ), f ∈ hyperbola_foci ∧ f = parabola_focus a) → a = 8 ∨ a = -8 := by
  sorry

#check focus_matching

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_matching_l1132_113248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_sum_calculation_l1132_113230

/-- Given compound and simple interest for a 2-year period, calculate the principal sum. -/
theorem principal_sum_calculation (P r : ℝ) : 
  P * ((1 + r)^2 - 1) = 11730 →
  P * r * 2 = 10200 →
  abs (P - 16993.46) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_sum_calculation_l1132_113230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_extrema_for_neg_three_F_difference_bound_implies_t_range_l1132_113242

-- Function definitions
noncomputable def f (a x : ℝ) := (a + 1) / 2 * x^2 - a * x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a x : ℝ) := f a x - g x

-- Theorem for question 1
theorem F_extrema_for_neg_three :
  let a := -3
  ∃ (x_min x_max : ℝ), x_min ∈ Set.Icc (Real.exp (-2)) (Real.exp 2) ∧
                        x_max ∈ Set.Icc (Real.exp (-2)) (Real.exp 2) ∧
                        (∀ x ∈ Set.Icc (Real.exp (-2)) (Real.exp 2), 
                          F a x ≥ F a x_min ∧ F a x ≤ F a x_max) ∧
                        F a x_min = -(Real.exp 4) + 3 * (Real.exp 2) - 2 ∧
                        F a x_max = -(Real.exp (-4)) + 3 * (Real.exp (-2)) + 2 := by
  sorry

-- Theorem for question 2
theorem F_difference_bound_implies_t_range :
  ∀ (a : ℝ), a ∈ Set.Ioo (-3) (-2) →
    (∃ (t : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
      |F a x₁ - F a x₂| < a * t + Real.log 2) → 
    t ∈ Set.Iic 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_extrema_for_neg_three_F_difference_bound_implies_t_range_l1132_113242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_expression_equality_second_expression_equality_l1132_113220

-- First expression
theorem first_expression_equality : 
  Real.sqrt (1 * (2 + 1/4)) - (-9.6 : Real)^(0 : Real) - (3 + 3/8 : Real)^(-2/3 : Real) + (1.5 : Real)^(-2 : Real) = 1/2 := by sorry

-- Second expression
theorem second_expression_equality :
  2 * (Real.log 5 / Real.log 10) + (Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) + 
  (Real.log 2 / Real.log 10)^2 + Real.exp (Real.log 3) = Real.log 125 / Real.log 10 + 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_expression_equality_second_expression_equality_l1132_113220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1132_113252

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (5 - 2*x) + Real.sqrt (x^2 - 4*x - 12)

-- State the theorem
theorem f_range :
  ∃ (lower : ℝ), lower = 3 ∧
  (∀ y : ℝ, (∃ x : ℝ, x ≤ -2 ∧ f x = y) ↔ y ≥ lower) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1132_113252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l1132_113268

-- Define the function
noncomputable def f (a b x : ℝ) : ℝ := b + a^(x^2 + 2*x)

-- State the theorem
theorem function_values (a b : ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 0, f a b x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 0, f a b x = 3) ∧
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 0, f a b x ≥ 5/2) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 0, f a b x = 5/2) →
  ((a = 2 ∧ b = 2) ∨ (a = 2/3 ∧ b = 3/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l1132_113268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sequence_l1132_113263

theorem smallest_n_for_sequence : ∃ (n : ℕ) (a : ℕ → ℤ),
  n > 0 ∧
  a 0 = 0 ∧
  a n = 2008 ∧
  (∀ i : ℕ, i > 0 → i ≤ n → |a i - a (i-1)| = i^2) ∧
  (∀ m : ℕ, m > 0 → m < n →
    ¬∃ (b : ℕ → ℤ), b 0 = 0 ∧ b m = 2008 ∧
    (∀ j : ℕ, j > 0 → j ≤ m → |b j - b (j-1)| = j^2)) ∧
  n = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sequence_l1132_113263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1132_113218

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (A + B + C = π) →  -- Sum of angles in a triangle
  (Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2) →
  (a = 2 * Real.sqrt 3) →
  (b + c = 4) →
  -- Conclusions
  (A = 2*π/3) ∧
  (b*c = 4) ∧
  (1/2 * b * c * Real.sin A = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1132_113218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_20_l1132_113293

/-- A figure consisting of a square and additional segments -/
structure Figure where
  /-- Side length of the square ABCE -/
  side_length : ℝ
  /-- Length of segment BD -/
  bd_length : ℝ

/-- Calculate the perimeter of the figure -/
noncomputable def perimeter (f : Figure) : ℝ :=
  4 * f.side_length + 2 * Real.sqrt (f.side_length^2 + f.bd_length^2)

/-- Theorem: The perimeter of the specific figure is 20 cm -/
theorem perimeter_is_20 :
  let f : Figure := { side_length := 4, bd_length := 3 }
  perimeter f = 20 := by
  -- Expand the definition of perimeter
  unfold perimeter
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_20_l1132_113293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1132_113296

def data_set (x : ℝ) : List ℝ := [7, 9, 6, x, 8, 7, 5]

noncomputable def range (l : List ℝ) : ℝ :=
  (l.maximum.getD 0) - (l.minimum.getD 0)

theorem x_value :
  ∀ x : ℝ, range (data_set x) = 6 → x = 11 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_l1132_113296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_theorem_l1132_113213

/-- A geometric progression with given first and fifth terms -/
structure GeometricProgression where
  b1 : ℝ
  b5 : ℝ
  h1 : b1 = Real.sqrt 3
  h5 : b5 = Real.sqrt 243

/-- The common ratio and sixth term of the geometric progression -/
noncomputable def progression_properties (gp : GeometricProgression) : ℝ × ℝ :=
  let q := Real.sqrt 3
  let b6 := 27
  (q, b6)

/-- Theorem stating the properties of the geometric progression -/
theorem geometric_progression_theorem (gp : GeometricProgression) :
  let (q, b6) := progression_properties gp
  (q = Real.sqrt 3 ∨ q = -Real.sqrt 3) ∧ (b6 = 27 ∨ b6 = -27) := by
  sorry

#check geometric_progression_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_theorem_l1132_113213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l1132_113267

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) : 
  a = 9 → b = 10 → θ = 150 * Real.pi / 180 → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) → 
  c = Real.sqrt (181 + 90 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l1132_113267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_workers_first_group_workers_is_three_l1132_113254

/-- The rate of cotton collection per worker per day -/
noncomputable def cotton_rate (kg : ℝ) (workers : ℝ) (days : ℝ) : ℝ :=
  kg / (workers * days)

/-- Proof that the number of workers in the first group is 3 -/
theorem first_group_workers : ℝ := by
  let first_group_kg := 48
  let first_group_days := 4
  let second_group_kg := 72
  let second_group_workers := 9
  let second_group_days := 2
  
  have h1 : cotton_rate first_group_kg (3 : ℝ) first_group_days = 
            cotton_rate second_group_kg second_group_workers second_group_days := by sorry
  
  have h2 : (3 : ℝ) * (first_group_kg / ((3 : ℝ) * first_group_days)) = 
            first_group_kg / first_group_days := by sorry
  
  have h3 : first_group_kg / first_group_days = 
            second_group_kg / (second_group_workers * second_group_days) := by sorry
  
  exact 3

/-- The result of first_group_workers is indeed 3 -/
theorem first_group_workers_is_three : first_group_workers = 3 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_workers_first_group_workers_is_three_l1132_113254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1132_113200

theorem distance_between_points (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = -4)
  (h2 : y₁ = 17)
  (h3 : x₂ = 12)
  (h4 : y₂ = -1)
  : Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2 * Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1132_113200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_prob_theorem_l1132_113260

/-- A binomial distribution with n trials and probability p -/
def binomial (n : ℕ) (p : ℝ) : Type := Unit

/-- The probability of getting at least k successes in a binomial distribution -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem binomial_prob_theorem (p : ℝ) :
  prob_at_least 2 1 p = 5/9 →
  prob_at_least 3 1 p = 19/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_prob_theorem_l1132_113260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1132_113244

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]
noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]

theorem point_transformation (a b : ℝ) :
  let P : Matrix (Fin 2) (Fin 1) ℝ := !![a; b]
  let result : Matrix (Fin 2) (Fin 1) ℝ := N * (M * P)
  result = !![8; 4 * Real.sqrt 3] →
  a = 5 ∧ b = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1132_113244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_z2_minus_z1_l1132_113258

/-- Given two real numbers x and y in the interval [-√(π/2), √(π/2)], 
    a complex number z₁ defined as (cos x²)/(sin y²) + (cos y²)/(sin x²)i with |z₁| = √2,
    and another complex number z₂ defined as x + yi,
    prove that the range of |z₂ - z₁| is the specified set. -/
theorem range_of_z2_minus_z1 (x y : ℝ) (z₁ z₂ : ℂ) 
  (h1 : x ∈ Set.Icc (-Real.sqrt (π/2)) (Real.sqrt (π/2)))
  (h2 : y ∈ Set.Icc (-Real.sqrt (π/2)) (Real.sqrt (π/2)))
  (h3 : z₁ = (Real.cos (x^2) / Real.sin (y^2)) + (Real.cos (y^2) / Real.sin (x^2)) * Complex.I)
  (h4 : Complex.abs z₁ = Real.sqrt 2)
  (h5 : z₂ = x + y * Complex.I) :
  Complex.abs (z₂ - z₁) ∈ 
    Set.Icc (Real.sqrt 2 - Real.sqrt (π/2)) (Real.sqrt (2 - Real.sqrt (2*π) + π/2)) ∪ 
    Set.Ioo (Real.sqrt (2 - Real.sqrt (2*π) + π/2)) (Real.sqrt (2 + Real.sqrt (2*π) + π/2)) ∪
    Set.Ioc (Real.sqrt (2 + Real.sqrt (2*π) + π/2)) (Real.sqrt 2 + Real.sqrt (π/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_z2_minus_z1_l1132_113258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_union_implies_a_range_l1132_113204

-- Define the sets A and B
noncomputable def A : Set ℝ := {x | x > 2 ∨ x ≤ -1}
noncomputable def B (a : ℝ) : Set ℝ := {x | x > a + 1 ∨ x < a}

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x + 1) / (x - 2))
noncomputable def g (a x : ℝ) : ℝ := 1 / Real.sqrt (x^2 - (2*a + 1)*x + a^2 + a)

-- State the theorem
theorem domain_union_implies_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, f x ≠ 0 → x ∈ A) →
  (∀ x : ℝ, g a x ≠ 0 → x ∈ B a) →
  A ∪ B a = B a →
  -1 < a ∧ a ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_union_implies_a_range_l1132_113204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1132_113292

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (M : ℝ), M = -1 ∧ ∀ x, x > 0 → f 0 x ≤ M :=
sorry

-- Theorem for part 2
theorem one_zero_iff_a_positive :
  ∀ a : ℝ, (∃! x, x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_zero_one_zero_iff_a_positive_l1132_113292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_value_l1132_113202

/-- Given a polynomial p(x) with specific values at x = 2, -1, and 4,
    and s(x) as the remainder when p(x) is divided by (x - 2)(x + 1)(x - 4),
    prove that s(3) = 17/3 -/
theorem remainder_value (p : ℝ → ℝ) (s : ℝ → ℝ) 
    (h1 : p 2 = 2)
    (h2 : p (-1) = -2)
    (h3 : p 4 = 5)
    (h4 : ∀ x, ∃ q : ℝ → ℝ, p x = (x - 2) * (x + 1) * (x - 4) * q x + s x) :
  s 3 = 17/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_value_l1132_113202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1132_113209

/-- Given two vectors a and b in ℝ², if a + 2b is parallel to 3a + λb, then λ = 6 -/
theorem parallel_vectors_lambda (a b : ℝ × ℝ) (lambda : ℝ) 
  (h1 : a = (1, 3))
  (h2 : b = (2, 1))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 2 • b = k • (3 • a + lambda • b)) :
  lambda = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l1132_113209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_4_a_range_when_1_in_B_l1132_113210

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a - x^2)

-- Define the domain A and range B
def A (a : ℝ) : Set ℝ := {x | a - x^2 ≥ 0}
def B (a : ℝ) : Set ℝ := {y | ∃ x, f a x = y}

theorem intersection_when_a_4 :
  A 4 ∩ B 4 = Set.Icc 0 2 := by sorry

theorem a_range_when_1_in_B :
  ∀ a : ℝ, 1 ∈ B a → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_4_a_range_when_1_in_B_l1132_113210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_symmetry_center_l1132_113274

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - (Real.sqrt 3 / 2) * Real.sin (2 * x) - 1 / 2

-- Define the theorem
theorem min_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →  -- Angle A is between 0 and π
  f A + 1 = 0 →  -- Condition on angle A
  b + c = 2 →  -- Condition on sides b and c
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →  -- Law of cosines
  a ≥ 1 :=  -- Conclusion: a is greater than or equal to 1
by sorry  -- Proof is omitted

-- Define the symmetry center theorem
theorem symmetry_center :
  ∃ k : ℤ, f (Real.pi / 12 + k * Real.pi / 2) = 0 :=
by sorry  -- Proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_symmetry_center_l1132_113274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1132_113281

noncomputable def f (x : ℝ) : ℝ := (2 * 4^x) / (4^x + 4^(-x))

noncomputable def g (x : ℝ) : ℝ := f x - 1

theorem g_properties :
  (∀ x, g (-x) = -g x) ∧
  (∀ x, -1 < g x ∧ g x < 1) ∧
  (∀ m, m > 1 → g m + g (m - 2) > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1132_113281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_7_simplest_l1132_113250

/-- A function that determines if a given real number is a quadratic radical in its simplest form -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ Nat.Prime n

/-- The given options -/
noncomputable def options : List ℝ := [1, Real.sqrt 7, Real.sqrt 12, 1 / Real.sqrt 13]

/-- Theorem stating that √7 is the simplest quadratic radical among the given options -/
theorem sqrt_7_simplest : 
  ∀ x ∈ options, x ≠ Real.sqrt 7 → ¬(is_simplest_quadratic_radical x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_7_simplest_l1132_113250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1132_113225

/-- Given two lines l₁ and l₂ in ℝ², prove that l₂ has the equation x + 3y = 15 -/
theorem line_equation (l₁ l₂ : Set (ℝ × ℝ)) : 
  (∃ t : ℝ, ∀ x y, (x, y) ∈ l₁ ↔ ∃ s, x = s * 1 ∧ y = s * 3) →
  (∃ k, ∀ x y, (x, y) ∈ l₂ ↔ ∃ t, x = t * (-1) ∧ y = t * k) →
  ((0, 5) ∈ l₂) →
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (x₂ - x₁) * 1 + (y₂ - y₁) * 3 = 0) →
  (∀ x y : ℝ, (x, y) ∈ l₂ ↔ x + 3*y = 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1132_113225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1132_113266

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a chord of a hyperbola -/
structure Chord where
  P : Point
  Q : Point

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := 
  Real.sqrt (1 + b^2 / a^2)

/-- Checks if a chord passes through a focus and is perpendicular to x-axis -/
def passes_through_focus_and_perpendicular (c : Chord) (F : Point) : Prop := 
  c.P.x = F.x ∧ c.Q.x = F.x ∧ c.P.y ≠ c.Q.y

/-- Checks if the angle between two points and a third point is 90 degrees -/
def is_right_angle (P Q F : Point) : Prop := 
  (P.x - F.x) * (Q.x - F.x) + (P.y - F.y) * (Q.y - F.y) = 0

/-- Main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (F₁ F₂ : Point) (c : Chord) :
  passes_through_focus_and_perpendicular c F₁ →
  is_right_angle c.P c.Q F₂ →
  eccentricity h = 1 + Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1132_113266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_partition_exists_l1132_113251

/-- A sequence of weights satisfying the problem conditions -/
structure WeightSequence where
  weights : Fin 2009 → ℕ
  weight_bound : ∀ i, weights i ≤ 1000
  adjacent_diff : ∀ i, i.val + 1 < 2009 → weights i.succ = weights i + 1 ∨ weights i.succ + 1 = weights i
  total_even : Even (Finset.sum Finset.univ weights)

/-- A partition of the weights into two subsets -/
def Partition (w : WeightSequence) := Fin 2009 → Bool

/-- The sum of weights in a partition subset -/
def PartitionSum (w : WeightSequence) (p : Partition w) (b : Bool) : ℕ :=
  Finset.sum (Finset.filter (fun i => p i = b) Finset.univ) w.weights

/-- Theorem stating the existence of an equal partition -/
theorem equal_partition_exists (w : WeightSequence) :
  ∃ p : Partition w, PartitionSum w p true = PartitionSum w p false := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_partition_exists_l1132_113251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_divisors_2016_l1132_113275

def divisors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

noncomputable def expectedValue (n : ℕ) : ℚ :=
  let divs := divisors n
  (1 : ℚ) / (divs.card : ℚ) * (divs.sum (fun d => (d^2 : ℚ) / ((d^2 : ℚ) + n)))

theorem expected_value_of_divisors_2016 :
  expectedValue 2016 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_divisors_2016_l1132_113275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_limit_l1132_113279

/-- The length of the path walked by the nth ant -/
noncomputable def L (n : ℕ) : ℝ := n * (Real.pi / n)

/-- The limit of L(n) as n approaches infinity is π -/
theorem ant_path_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |L n - Real.pi| < ε :=
by
  sorry

#check ant_path_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_limit_l1132_113279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_equidistant_points_l1132_113241

-- Define the point F
def F : ℝ × ℝ := (0, -3)

-- Define the line
def line (y : ℝ) : Prop := y + 5 = 0

-- Define the distance from a point to F
noncomputable def dist_to_F (x y : ℝ) : ℝ := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)

-- Define the distance from a point to the line
def dist_to_line (y : ℝ) : ℝ := |y + 5|

-- Define the locus equation
def locus_eq (x y : ℝ) : Prop := y = (1/4) * x^2 - 4

-- Theorem statement
theorem locus_of_equidistant_points (x y : ℝ) :
  locus_eq x y ↔ dist_to_F x y = dist_to_line y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_equidistant_points_l1132_113241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1132_113270

/-- Given that f(x) is a polynomial and f(x^2 + 2) = x^4 + 6x^2 + 4, 
    prove that f(x^2 - 2) = x^4 - 2x^2 - 4. -/
theorem polynomial_identity (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 2) = x^4 + 6*x^2 + 4) :
  ∀ x : ℝ, f (x^2 - 2) = x^4 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1132_113270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_expression_l1132_113247

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 0 then x^2 + 2*x + 5
  else x^2 - 2*x + 5

-- State the theorem
theorem f_even_expression :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, x < 0 → f x = x^2 + 2*x + 5) →  -- expression for x < 0
  (∀ x, x > 0 → f x = x^2 - 2*x + 5) :=  -- expression for x > 0
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_expression_l1132_113247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_characteristics_l1132_113233

-- Define the probability density function
noncomputable def p (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ 1 then 2 * x
  else 0

-- Define the random variable X
def X : Type := ℝ

-- State the theorem
theorem random_variable_characteristics :
  ∃ (M D σ : ℝ),
    (∫ (x : ℝ), x * p x) = M ∧
    (∫ (x : ℝ), (x - M)^2 * p x) = D ∧
    Real.sqrt D = σ ∧
    M = 2/3 ∧
    D = 1/18 ∧
    σ = 1 / Real.sqrt 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_characteristics_l1132_113233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_144_l1132_113269

/-- Fibonacci-like sequence defined as a₁ = 2, a₂ = 3, aₙ = aₙ₋₁ + aₙ₋₂ for n ≥ 3 -/
def FibLikeSeq : ℕ → ℕ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 3
  | n + 3 => FibLikeSeq (n + 2) + FibLikeSeq (n + 1)

/-- The cost function for guessing a number in a set of size n -/
def GuessCost : ℕ → ℕ
  | 0 => 0
  | n + 1 => n + 1

theorem min_cost_for_144 :
  ∃ n : ℕ, FibLikeSeq n = 144 ∧ GuessCost n = 11 := by
  -- The proof is skipped using sorry
  sorry

#eval FibLikeSeq 10  -- This will evaluate to 144
#eval GuessCost 10   -- This will evaluate to 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_144_l1132_113269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistance_approx_l1132_113219

/-- The combined resistance of two parallel resistors -/
noncomputable def parallel_resistance (x y : ℝ) : ℝ := 1 / (1 / x + 1 / y)

/-- The combined resistance of two parallel resistors with 5 and 6 ohms -/
noncomputable def r : ℝ := parallel_resistance 5 6

theorem parallel_resistance_approx :
  abs (r - 2.73) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistance_approx_l1132_113219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1132_113257

noncomputable def f (x : ℝ) : ℝ := (8*x^4 + 3*x^3 + 7*x^2 + 2*x + 1) / (2*x^4 + x^3 + 5*x^2 + 2*x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, abs x > M → abs (f x - 4) < ε :=
by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1132_113257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1132_113235

noncomputable def f (x : ℝ) : ℝ := Real.log (4 - x^2)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ -2 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1132_113235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_part1_a_value_part2_l1132_113282

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (Real.log (x - 1)) / Real.log a

-- Theorem for part (1)
theorem min_value_part1 :
  ∀ x ∈ Set.Icc 1 2, f (1/4) x ≥ 1/16 := by
  sorry

-- Theorem for part (2)
theorem a_value_part2 (h : ∀ x ∈ Set.Icc 2 3, f a x ≥ 4) (ha : a > 0) (ha' : a ≠ 1) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_part1_a_value_part2_l1132_113282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1132_113239

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = 2 * B ∧
  sin B = sqrt 3 / 3 ∧
  b = 2 →
  cos A = 1 / 3 ∧
  sin C = 5 * sqrt 3 / 9 ∧
  (1 / 2 * a * b * sin C) = 20 * sqrt 2 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1132_113239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1132_113276

noncomputable def v1 : ℝ × ℝ × ℝ := (2, 0, 3)
noncomputable def v2 : ℝ × ℝ × ℝ := (4, 1, -3)
noncomputable def proj_v1 : ℝ × ℝ × ℝ := (2, -1, 1)
noncomputable def proj_v2 : ℝ × ℝ × ℝ := (4/3, -2/3, 2/3)

noncomputable def u : ℝ × ℝ × ℝ := proj_v1

noncomputable def proj (v u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2.1 * u.2.1 + v.2.2 * u.2.2
  let magnitude_squared := u.1 * u.1 + u.2.1 * u.2.1 + u.2.2 * u.2.2
  let scalar := dot_product / magnitude_squared
  (scalar * u.1, scalar * u.2.1, scalar * u.2.2)

theorem projection_theorem :
  proj v1 u = proj_v1 → proj v2 u = proj_v2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1132_113276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_shirts_per_package_l1132_113214

/-- The number of t-shirts per package, given the total number of t-shirts and packages. -/
noncomputable def t_shirts_per_package (total_shirts : ℝ) (num_packages : ℝ) : ℝ :=
  total_shirts / num_packages

/-- Theorem stating that 71.0 t-shirts in 11.83333333 packages results in approximately 6 t-shirts per package. -/
theorem six_shirts_per_package :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |t_shirts_per_package 71.0 11.83333333 - 6| < ε := by
  sorry

#eval (71.0 : Float) / 11.83333333

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_shirts_per_package_l1132_113214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seashell_collection_l1132_113216

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the problem
theorem seashell_collection (total shells : ℕ) (red green : ℕ) : 
  total = 820 →
  red = 144 →
  green = 89 →
  (∃ blue yellow purple : ℕ, 
    (∃ i : ℕ, fib i = blue) ∧ 
    (∃ j : ℕ, fib j = yellow) ∧ 
    (∃ k : ℕ, fib k = purple) ∧
    blue < green ∧
    green < red ∧
    yellow > red ∧
    purple > yellow ∧
    blue + yellow + purple = shells ∧
    red + green + blue + yellow + purple = total) →
  shells = 665 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seashell_collection_l1132_113216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1132_113288

/-- Represents a triangle with its properties -/
structure Triangle where
  angle₁ : ℝ
  angle₂ : ℝ
  angle₃ : ℝ
  side₁ : ℝ
  side₂ : ℝ
  side₃ : ℝ
  area : ℝ
  altitudeToHypotenuse : ℝ
  isRight : Prop

/-- The area of a right triangle with angles of 30 and 60 degrees and an altitude to the hypotenuse of 4 units is 16√3 square units. -/
theorem right_triangle_area (t : Triangle) 
  (h₁ : t.isRight) 
  (h₂ : t.angle₁ = 30 * π / 180) 
  (h₃ : t.angle₂ = 60 * π / 180) 
  (h₄ : t.altitudeToHypotenuse = 4) : t.area = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1132_113288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_max_value_l1132_113236

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- A line y = ax + b tangent to f(x) at some point -/
structure TangentLine where
  a : ℝ
  b : ℝ
  x₀ : ℝ
  tangent_condition : f x₀ = a * x₀ + b
  slope_condition : deriv f x₀ = a

/-- The theorem stating the maximum value of (1-a)/b for tangent lines to f(x) -/
theorem tangent_line_max_value (l : TangentLine) :
  ∃ (M : ℝ), M = 1 / Real.exp 1 ∧ (1 - l.a) / l.b ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_max_value_l1132_113236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chimney_bricks_count_l1132_113280

/-- Represents the number of bricks in the chimney -/
def chimney_bricks : ℕ := 360

/-- Time taken by Brenda to build the chimney alone (in hours) -/
def brenda_time : ℕ := 8

/-- Time taken by Brandon to build the chimney alone (in hours) -/
def brandon_time : ℕ := 12

/-- Decrease in combined output due to wind (in bricks per hour) -/
def wind_decrease : ℕ := 15

/-- Time taken when working together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating the number of bricks in the chimney -/
theorem chimney_bricks_count : 
  chimney_bricks = 
    ((chimney_bricks / brenda_time + chimney_bricks / brandon_time - wind_decrease) * combined_time) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chimney_bricks_count_l1132_113280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1132_113277

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, 6)
def C : ℝ × ℝ := (3, 10)

-- Define the circle
def circleSet : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ r : ℝ, (p.1 - A.1)^2 + (p.2 - A.2)^2 = r^2 ∧
                                               (p.1 - B.1)^2 + (p.2 - B.2)^2 = r^2 ∧
                                               (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2}

-- Theorem statement
theorem tangent_length : 
  ∃ T : ℝ × ℝ, T ∈ circleSet ∧ 
    (∀ p : ℝ × ℝ, p ∈ circleSet → (T.1 - O.1) * (p.1 - T.1) + (T.2 - O.2) * (p.2 - T.2) = 0) ∧
    (T.1 - O.1)^2 + (T.2 - O.2)^2 = 26 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1132_113277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_sum_l1132_113295

/-- Given three quadratic polynomials P, Q, and R over a field F -/
def quadratic_polynomial (F : Type) [Field F] := F → F

variable {F : Type} [Field F]
variable (P Q R : quadratic_polynomial F)

/-- Addition of quadratic polynomials -/
instance : HAdd (quadratic_polynomial F) (quadratic_polynomial F) (quadratic_polynomial F) where
  hAdd f g := λ x => f x + g x

/-- Condition that a quadratic equation has exactly two roots -/
def has_two_roots (f : quadratic_polynomial F) : Prop :=
  ∃ (x y : F), x ≠ y ∧ ∀ z, f z = 0 ↔ (z = x ∨ z = y)

/-- Product of roots of a quadratic equation -/
noncomputable def root_product (f : quadratic_polynomial F) : F :=
  sorry  -- Definition omitted for brevity

/-- Theorem: Product of roots of P + Q + R = 0 -/
theorem product_of_roots_sum (p q r : F) 
  (h1 : has_two_roots (P + Q))
  (h2 : has_two_roots (P + R))
  (h3 : has_two_roots (Q + R))
  (h4 : has_two_roots (P + Q + R))
  (hp : root_product (P + Q) = r)
  (hq : root_product (P + R) = q)
  (hr : root_product (Q + R) = p) :
  root_product (P + Q + R) = (p + q + r) / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_sum_l1132_113295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1132_113228

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define the line passing through P(2,2) with inclination angle π/3
noncomputable def line (t : ℝ) : ℝ × ℝ := (2 + t/2, 2 + (Real.sqrt 3 * t)/2)

-- Define the point P
def P : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem intersection_product : 
  ∃ (A B : ℝ × ℝ), 
    (∃ (t1 t2 : ℝ), 
      circle_eq (line t1).1 (line t1).2 ∧ 
      circle_eq (line t2).1 (line t2).2 ∧
      A = line t1 ∧ 
      B = line t2) →
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)) * 
    (Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1132_113228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_revenue_calculation_l1132_113283

structure Pie where
  slicesPerPie : Nat
  pricePerSlice : Nat

structure BakeryItems where
  pumpkinPie : Pie
  custardPie : Pie
  applePie : Pie
  pecanPie : Pie
  cookiePrice : Nat
  redVelvetCake : Pie

structure BakerySales where
  pumpkinPies : Nat
  custardPies : Nat
  applePies : Nat
  pecanPies : Nat
  cookies : Nat
  redVelvetCakes : Nat

def calculateRevenue (items : BakeryItems) (sales : BakerySales) : Nat :=
  (items.pumpkinPie.slicesPerPie * items.pumpkinPie.pricePerSlice * sales.pumpkinPies) +
  (items.custardPie.slicesPerPie * items.custardPie.pricePerSlice * sales.custardPies) +
  (items.applePie.slicesPerPie * items.applePie.pricePerSlice * sales.applePies) +
  (items.pecanPie.slicesPerPie * items.pecanPie.pricePerSlice * sales.pecanPies) +
  (items.cookiePrice * sales.cookies) +
  (items.redVelvetCake.slicesPerPie * items.redVelvetCake.pricePerSlice * sales.redVelvetCakes)

theorem bakery_revenue_calculation (items : BakeryItems) (sales : BakerySales) :
  items.pumpkinPie = { slicesPerPie := 8, pricePerSlice := 5 } →
  items.custardPie = { slicesPerPie := 6, pricePerSlice := 6 } →
  items.applePie = { slicesPerPie := 10, pricePerSlice := 4 } →
  items.pecanPie = { slicesPerPie := 12, pricePerSlice := 7 } →
  items.cookiePrice = 2 →
  items.redVelvetCake = { slicesPerPie := 8, pricePerSlice := 9 } →
  sales.pumpkinPies = 4 →
  sales.custardPies = 5 →
  sales.applePies = 3 →
  sales.pecanPies = 2 →
  sales.cookies = 15 →
  sales.redVelvetCakes = 6 →
  calculateRevenue items sales = 1090 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_revenue_calculation_l1132_113283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_change_l1132_113259

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  side : ℝ  -- side length of the square base
  height : ℝ  -- height of the pyramid
  volume : ℝ  -- volume of the pyramid

/-- Calculates the volume of a square pyramid -/
noncomputable def calculateVolume (s h : ℝ) : ℝ := (1/3) * s^2 * h

theorem pyramid_volume_change 
  (p : SquarePyramid) 
  (h_volume : p.volume = 60) 
  (h_volume_calc : p.volume = calculateVolume p.side p.height) :
  let new_side := 4 * p.side
  let new_height := 0.75 * p.height
  calculateVolume new_side new_height = 180 := by
  sorry

#check pyramid_volume_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_change_l1132_113259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_period_pi_l1132_113273

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (x + 3 * Real.pi / 2))^2 - 1

theorem f_is_even_and_period_pi :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_period_pi_l1132_113273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2000_value_l1132_113291

def x : ℕ → ℚ
| 0 => -1
| n + 1 => (1 + 2 / (n + 1 : ℚ)) * x n + 4 / (n + 1 : ℚ)

theorem x_2000_value : x 2000 = 2000998 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2000_value_l1132_113291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1132_113246

noncomputable def y : ℝ := 10^(-1998 : ℤ)

theorem largest_expression (a b c d e : ℝ) 
  (ha : a = 4 + y) 
  (hb : b = 4 - y) 
  (hc : c = 4 * y) 
  (hd : d = 4 / y) 
  (he : e = (4 / y)^2) : 
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

#check largest_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l1132_113246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_special_triangles_l1132_113207

/-- Given two right triangles where the incircle of the first triangle is the circumcircle of the second triangle, 
    the ratio of their areas is greater than or equal to 3 + 2√2. -/
theorem area_ratio_of_special_triangles (S S' : ℝ) 
  (h1 : S > 0) 
  (h2 : S' > 0)
  (h3 : ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧
    S = (1/2) * b * c ∧
    S' = ((b * c) / (b + c + Real.sqrt (b^2 + c^2)))^2) : 
  S / S' ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_special_triangles_l1132_113207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1132_113289

/-- Ellipse with center at origin and eccentricity 1/2 -/
structure Ellipse where
  center : ℝ × ℝ
  eccentricity : ℝ
  h_center : center = (0, 0)
  h_eccentricity : eccentricity = 1/2

/-- Parabola with equation y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The right focus of the ellipse coincides with the focus of the parabola -/
def focus_coincide (e : Ellipse) : Prop :=
  (2, 0) ∈ Parabola

/-- A and B are points of intersection of the latus rectum of the parabola and the ellipse -/
def intersection_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1 = 2 ∧ q.2^2 = 45/4}

/-- The length of AB is 6 -/
theorem length_AB (e : Ellipse) 
  (h : focus_coincide e) : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points e ∧ 
              B ∈ intersection_points e ∧
              A ≠ B ∧
              Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1132_113289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_relation_l1132_113290

theorem log_power_relation (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  let p := (Real.log (Real.log a) / Real.log b) / (Real.log a / Real.log b)
  a^p = Real.log a / Real.log b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_power_relation_l1132_113290
