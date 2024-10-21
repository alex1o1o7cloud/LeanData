import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandmother_pill_duration_grandmother_pill_duration_rounded_l932_93257

/-- Calculates the duration in months that a supply of pills will last -/
def pillDuration (totalPills : ℕ) (pillFraction : ℚ) (daysPerDose : ℕ) : ℚ :=
  let daysPerPill : ℚ := daysPerDose / pillFraction
  let totalDays : ℚ := totalPills * daysPerPill
  totalDays / 30

theorem grandmother_pill_duration :
  pillDuration 90 (2/3) 3 = 13.5 := by
  sorry

theorem grandmother_pill_duration_rounded :
  ⌊pillDuration 90 (2/3) 3⌋₊ = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandmother_pill_duration_grandmother_pill_duration_rounded_l932_93257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_five_power_l932_93219

theorem units_digit_of_five_power (n : ℕ) : (5^n : ℕ) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_five_power_l932_93219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_with_property_l932_93249

/-- A function that splits a four-digit number into its hundreds-tens part and ones-tens part -/
def split (n : ℕ) : ℕ × ℕ := (n / 100, n % 100)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The property we're looking for in our number -/
def has_property (n : ℕ) : Prop :=
  let (a, b) := split n
  is_perfect_square (a * b)

/-- The set of all numbers greater than 1818 and less than 10000 with the desired property -/
def valid_numbers : Set ℕ :=
  {n : ℕ | 1818 < n ∧ n < 10000 ∧ has_property n}

theorem next_number_with_property : 
  1832 ∈ valid_numbers ∧ ∀ m ∈ valid_numbers, 1832 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_with_property_l932_93249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_positive_l932_93202

theorem negation_of_forall_positive :
  (∀ x : ℝ, (2 : ℝ)^x + 1 > 0) ↔ ¬(∃ x : ℝ, (2 : ℝ)^x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_positive_l932_93202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_property_l932_93297

-- Define the set of lattice points
def T : Set (ℤ × ℤ) := Set.univ

-- Define the adjacency relation
def adjacent (p q : ℤ × ℤ) : Prop :=
  (abs (p.1 - q.1) + abs (p.2 - q.2) : ℤ) = 1

-- Define the subset S
def S : Set (ℤ × ℤ) :=
  {p | p ∈ T ∧ 5 ∣ (p.1 + 2*p.2)}

-- State the theorem
theorem subset_property :
  ∀ p ∈ T, ∃! q, (q = p ∨ adjacent p q) ∧ q ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_property_l932_93297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l932_93246

/-- The function f as defined in the problem -/
noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

/-- The theorem statement -/
theorem max_value_of_f :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → f x y ≤ f x₀ y₀) ∧
  f x₀ y₀ = 1 / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l932_93246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l932_93223

def S (n : ℕ) : ℤ := n^2 - 3*n + 2

def a : ℕ → ℤ
| 0 => 0  -- Add this case for n = 0
| 1 => 0
| (n+2) => 2*(n+2) - 4

theorem sequence_general_term (n : ℕ) : 
  (n = 1 ∧ a n = S 1) ∨ 
  (n > 1 ∧ a n = S n - S (n-1)) := by
  sorry

#eval a 0  -- Test case for n = 0
#eval a 1  -- Test case for n = 1
#eval a 3  -- Test case for n > 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l932_93223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l932_93228

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def is_convex (q : Quadrilateral) : Prop := sorry

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def angle_cos (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

noncomputable def quadrilateral_area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_properties (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_AD : side_length q.A q.D = 2)
  (h_AB : side_length q.A q.B = 3)
  (h_BD : side_length q.B q.D = 4)
  (h_angle_BCD : angle_cos q.B q.C q.D = 7/8)
  (h_CD : side_length q.C q.D = 6)
  (h_BC_gt_CD : side_length q.B q.C > side_length q.C q.D) :
  (∃ (k : ℝ), q.A.1 - q.D.1 = k * (q.B.1 - q.C.1) ∧ 
              q.A.2 - q.D.2 = k * (q.B.2 - q.C.2)) ∧ 
  (∀ (q' : Quadrilateral), 
    is_convex q' → 
    side_length q'.A q'.D = 2 → 
    side_length q'.A q'.B = 3 → 
    side_length q'.B q'.D = 4 → 
    angle_cos q'.B q'.C q'.D = 7/8 → 
    quadrilateral_area q' ≤ (19 * Real.sqrt 15) / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l932_93228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l932_93201

noncomputable def distance_B (square_side : ℝ) (rotation_angle : ℝ) : ℝ :=
  square_side * Real.sqrt 2 * Real.sin (rotation_angle * Real.pi / 180)

theorem rotated_square_height :
  distance_B 1 30 = Real.sqrt 2 / 2 := by
  sorry

#check rotated_square_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l932_93201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_paper_sector_l932_93277

/-- The height of a cone formed by rolling one sector of a circular sheet --/
noncomputable def cone_height (r : ℝ) (n : ℕ) : ℝ :=
  Real.sqrt (r^2 - (r * 2 * Real.pi / (n * 2 * Real.pi))^2)

/-- Theorem stating the height of the cone formed by the given conditions --/
theorem cone_height_from_paper_sector :
  cone_height 8 4 = 2 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_paper_sector_l932_93277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l932_93285

noncomputable def f (x : ℝ) : ℝ := Real.exp (x / 3)

noncomputable def f' (x : ℝ) : ℝ := (1 / 3) * Real.exp (x / 3)

theorem tangent_line_triangle_area :
  let x₀ : ℝ := 6
  let y₀ : ℝ := Real.exp 2
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  let x_intercept : ℝ := -b / m
  let y_intercept : ℝ := b
  let triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept
  triangle_area = (3 / 2) * Real.exp 2 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l932_93285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_equals_negative_four_l932_93283

/-- Given a real number s that satisfies s^4 - s - 1/2 = 0,
    T is defined as the infinite sum s^3 + 2s^7 + 3s^11 + 4s^15 + ... -/
noncomputable def T (s : ℝ) : ℝ := ∑' n, (n + 1) * s^(4*n + 3)

/-- Theorem stating that if s is a real solution to x^4 - x - 1/2 = 0,
    then T(s) equals -4 -/
theorem T_equals_negative_four (s : ℝ) (hs : s^4 - s - 1/2 = 0) : T s = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_equals_negative_four_l932_93283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_distribution_count_l932_93269

/-- The number of ways to distribute tourists among guides -/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists - 
  (num_guides * (num_guides - 1) ^ num_tourists) + 
  (Nat.choose num_guides 2 * 1 ^ num_tourists)

/-- Theorem stating the correct number of distributions for 9 tourists and 3 guides -/
theorem tourist_distribution_count : distribute_tourists 9 3 = 18150 := by
  rfl

#eval distribute_tourists 9 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_distribution_count_l932_93269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_no_intersection_l932_93281

/-- The circle C with center (-5, 0) and radius r -/
def circleC (x y r : ℝ) : Prop := (x + 5)^2 + y^2 = r^2

/-- The line l: 3x + y + 5 = 0 -/
def lineL (x y : ℝ) : Prop := 3*x + y + 5 = 0

/-- No intersection between circle C and line l -/
def no_intersection (r : ℝ) : Prop := ∀ x y : ℝ, circleC x y r → ¬(lineL x y)

theorem circle_line_no_intersection (r : ℝ) (hr : r > 0) (h_no_intersect : no_intersection r) : 
  r > 0 ∧ r < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_no_intersection_l932_93281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_p_and_q_l932_93206

-- Define the propositions p and q
def p (k : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (2 + k) - y^2 / (3*k + 1) = 1 → True  -- Placeholder for IsHyperbola

def q (k : ℝ) : Prop :=
  ∃ m b : ℝ,  -- Define line using slope-intercept form instead of Line type
    m = k ∧
    1 = m * (-2) + b ∧  -- Line passes through (-2, 1)
    (∃ x1 y1 x2 y2 : ℝ,
      (x1, y1) ≠ (x2, y2) ∧
      y1 = m * x1 + b ∧
      y2 = m * x2 + b ∧
      y1^2 = 4*x1 ∧
      y2^2 = 4*x2)

-- Define the theorem
theorem k_range_for_p_and_q :
  {k : ℝ | p k ∧ q k} = {k : ℝ | -1/3 < k ∧ k < 0 ∨ 0 < k ∧ k < 1/2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_for_p_and_q_l932_93206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l932_93229

noncomputable def f (x a : ℝ) : ℝ := 2 * (Real.cos x)^2 + a * Real.sin (2 * x) + 1

theorem f_properties :
  ∃ (a : ℝ),
    (f (π / 3) a = 0) ∧
    (a = -Real.sqrt 3) ∧
    (∀ (k : ℤ) (x : ℝ),
      (k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6) →
      (∀ (y : ℝ), k * π - 2 * π / 3 ≤ y ∧ y ≤ x → f y a ≤ f x a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l932_93229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l932_93282

theorem angle_sum_bounds (A B C : ℝ) 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) 
  (h3 : 0 < C ∧ C < π/2) 
  (h4 : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) : 
  π/2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_bounds_l932_93282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetry_l932_93224

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem min_omega_for_symmetry :
  ∀ ω : ℝ, ω > 0 →
  (∀ x : ℝ, f ω (x + Real.pi / (2 * ω)) = f ω (-x + Real.pi / (2 * ω))) →
  ω ≥ 1 / 3 ∧ ∃ ω₀ : ℝ, ω₀ = 1 / 3 ∧ ω₀ > 0 ∧
  (∀ x : ℝ, f ω₀ (x + Real.pi / (2 * ω₀)) = f ω₀ (-x + Real.pi / (2 * ω₀))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_symmetry_l932_93224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equation_l932_93289

theorem ceiling_sum_equation : 
  ⌈(Real.sqrt (25 / 9) + 1 / 3 : ℝ)⌉ + ⌈(25 / 9 : ℝ)⌉ + ⌈((25 / 9 : ℝ)^2)⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equation_l932_93289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_zero_F_geq_one_condition_min_difference_l932_93226

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x
def g (a : ℝ) (x : ℝ) : ℝ := a * x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x - g a x

-- Part 1
theorem extreme_point_zero (a : ℝ) : 
  a = 2 → (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → F a x < F a 0 ∨ F a x > F a 0) :=
by sorry

-- Part 2
theorem F_geq_one_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → F a x ≥ 1) → a ≤ 2 :=
by sorry

-- Part 3
theorem min_difference (x₁ x₂ : ℝ) :
  x₁ ≥ 0 → x₂ ≥ 0 → f x₁ = g (1/3) x₂ → x₂ - x₁ ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_zero_F_geq_one_condition_min_difference_l932_93226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_perimeter_l932_93233

/-- Represents a rectangular playground -/
structure Playground where
  width : ℝ
  length : ℝ

/-- Calculates the diagonal of a playground -/
noncomputable def diagonal (p : Playground) : ℝ :=
  Real.sqrt (p.width ^ 2 + p.length ^ 2)

/-- Calculates the area of a playground -/
def area (p : Playground) : ℝ :=
  p.width * p.length

/-- Calculates the perimeter of a playground -/
def perimeter (p : Playground) : ℝ :=
  2 * (p.width + p.length)

/-- Theorem: A rectangular playground with diagonal 30 meters and area 216 square meters has a perimeter of 72 meters -/
theorem playground_perimeter (p : Playground) 
  (h_diagonal : diagonal p = 30)
  (h_area : area p = 216) : 
  perimeter p = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_perimeter_l932_93233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l932_93287

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular_lines (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of a line in the form y = mx + c is m -/
def slope_of_line (m c : ℝ) : ℝ := m

theorem perpendicular_lines_b_value (b : ℝ) :
  perpendicular_lines (slope_of_line 3 (-7)) (slope_of_line (-b/4) 3) → b = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_value_l932_93287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l932_93254

theorem product_of_four_integers (P Q R S : ℕ) :
  P + Q + R + S = 48 →
  P + 3 = Q - 3 →
  P + 3 = R * 3 →
  P + 3 = S / 3 →
  P * Q * R * S = 5832 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l932_93254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_pirate_share_l932_93235

/-- Represents the number of pirates -/
def n : ℕ := 13

/-- Calculates the fraction of remaining coins taken by the k-th pirate -/
def pirate_share (k : ℕ) : ℚ := (k + 1) / n

/-- Calculates the remaining fraction of coins after k pirates have taken their share -/
def remaining_fraction (k : ℕ) : ℚ :=
  if k = 0 then 1
  else (1 - pirate_share k) * remaining_fraction (k - 1)

/-- The smallest number of initial coins that satisfies the conditions -/
def initial_coins : ℕ := n^(n-1) * Nat.factorial (n-1)

/-- The number of coins the last pirate receives -/
def last_pirate_coins : ℕ := Nat.factorial (n-1)

theorem last_pirate_share :
  last_pirate_coins = (initial_coins * (remaining_fraction (n-1))).num :=
sorry

#eval last_pirate_coins -- Should output 479001600

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_pirate_share_l932_93235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_from_equal_inscribed_radii_l932_93211

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  A : V
  B : V
  C : V
  D : V

/-- The radius of the inscribed circle in a triangle --/
noncomputable def inscribedRadius {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B C : V) : ℝ := sorry

/-- A quadrilateral is a rectangle if all its angles are right angles --/
def isRectangle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (q : Quadrilateral V) : Prop := sorry

theorem rectangle_from_equal_inscribed_radii
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (q : Quadrilateral V)
  (h : inscribedRadius q.A q.B q.C = inscribedRadius q.B q.C q.D ∧
       inscribedRadius q.B q.C q.D = inscribedRadius q.C q.D q.A ∧
       inscribedRadius q.C q.D q.A = inscribedRadius q.D q.A q.B) :
  isRectangle q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_from_equal_inscribed_radii_l932_93211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_apples_l932_93212

theorem perfect_apples (total : ℕ) (small_fraction : ℚ) (unripe_fraction : ℚ) :
  total = 30 →
  small_fraction = 1 / 6 →
  unripe_fraction = 1 / 3 →
  ∃ perfect : ℕ, perfect = total - (total * small_fraction).floor - (total * unripe_fraction).floor ∧ perfect = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_apples_l932_93212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_bagels_l932_93220

/-- Represents the number of bagels Jane bought in a week -/
def num_bagels : ℕ := sorry

/-- Represents the number of muffins Jane bought in a week -/
def num_muffins : ℕ := sorry

/-- The cost of a bagel in cents -/
def bagel_cost : ℕ := 90

/-- The cost of a muffin in cents -/
def muffin_cost : ℕ := 60

/-- The total number of items bought in a week -/
def total_items : ℕ := 7

/-- The divisibility factor for the total cost in cents -/
def divisibility_factor : ℕ := 150

theorem janes_bagels :
  (num_bagels + num_muffins = total_items) →
  (bagel_cost * num_bagels + muffin_cost * num_muffins) % divisibility_factor = 0 →
  num_bagels = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janes_bagels_l932_93220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l932_93203

/-- The cubic function f(x) with a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (3/2) * x^2 + 2*x + 1

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3*a*x^2 - 3*x + 2

theorem min_value_of_f (a : ℝ) :
  (f' a 1 = 0) →  -- Tangent line at (1, f(1)) is parallel to x-axis
  (∀ x ∈ Set.Ioo 1 3, f a x ≥ 5/3) ∧ 
  (∃ x ∈ Set.Ioo 1 3, f a x = 5/3) := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l932_93203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_area_eq_three_halves_pi_squared_l932_93218

/-- The area bounded by the graph of y = arccos(cos x) and the x-axis on the interval [0, 3π] -/
noncomputable def bounded_area : ℝ := ∫ x in (0)..(3 * Real.pi), Real.arccos (Real.cos x)

/-- Theorem: The bounded area is equal to 3/2 * π² -/
theorem bounded_area_eq_three_halves_pi_squared :
  bounded_area = 3/2 * Real.pi^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_area_eq_three_halves_pi_squared_l932_93218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_max_value_f_min_value_l932_93264

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - Real.sin x

def is_decreasing_interval (k : ℤ) : Set ℝ :=
  Set.Ioo ((2 * k : ℝ) * Real.pi - Real.pi/3) ((2 * k : ℝ) * Real.pi + Real.pi/3)

theorem f_decreasing_intervals :
  ∀ x : ℝ, (∃ k : ℤ, x ∈ is_decreasing_interval k) ↔ 
  (((1/2) - Real.cos x) < 0) :=
sorry

theorem f_max_value :
  ∃ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧
  f x = Real.pi / 2 ∧
  ∀ y ∈ Set.Icc (-Real.pi) Real.pi, f y ≤ f x :=
sorry

theorem f_min_value :
  ∃ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧
  f x = -Real.pi / 2 ∧
  ∀ y ∈ Set.Icc (-Real.pi) Real.pi, f x ≤ f y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_intervals_f_max_value_f_min_value_l932_93264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_less_than_half_l932_93230

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 else x^2

-- State the theorem
theorem f_inequality_iff_a_less_than_half :
  ∀ a : ℝ, f (a - 1) + f a > 0 ↔ a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_less_than_half_l932_93230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_eq_one_seventh_l932_93227

def b : ℕ → ℚ
  | 0 => 5  -- We define b 0 to be 5 to match b 1 in the original problem
  | 1 => 7
  | (n+2) => b (n+1) / b n

theorem b_2023_eq_one_seventh : b 2023 = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_eq_one_seventh_l932_93227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_color_l932_93265

inductive BeadColor
  | Red
  | Orange
  | Yellow
  | Green
  | Blue

def pattern : List BeadColor := [
  BeadColor.Red, BeadColor.Orange, 
  BeadColor.Yellow, BeadColor.Yellow,
  BeadColor.Green, BeadColor.Green,
  BeadColor.Blue
]

def necklace_length : Nat := 85

theorem last_bead_color (h : necklace_length = 85) :
  pattern[(necklace_length - 1) % pattern.length] = BeadColor.Red := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_bead_color_l932_93265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l932_93236

/-- Given two perpendicular planes S and T sharing a common line m, with points C in S and D in T,
    and their projections C' and D' on m, this function calculates the radius of two equal spheres
    that touch each other and touch S at C and T at D. -/
noncomputable def sphere_radius (c d e : ℝ) : ℝ :=
  (1/2) * (Real.sqrt (2*(c^2 + d^2 + e^2) + (c + d)^2) - (c + d))

/-- Theorem stating that the calculated radius is correct for the given geometric configuration. -/
theorem sphere_radius_correct (c d e : ℝ) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  let r := sphere_radius c d e
  (c - r)^2 + (d - r)^2 + e^2 = 4 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_correct_l932_93236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l932_93245

noncomputable def f (n : ℝ) : ℝ → ℝ := λ x ↦ x^n

theorem power_function_inequality (n : ℝ) (h : f n 2 = 8) :
  f n (2^(1/5)) > f n (Real.sqrt (1/2)) ∧ f n (Real.sqrt (1/2)) > f n (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l932_93245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_65x65_l932_93239

/-- Represents a game board -/
structure Board :=
  (size : Nat)
  (pieces : List (Nat × Nat))

/-- Defines a valid move on the board -/
def validMove (b : Board) (row col : Nat) : Prop :=
  row ≤ b.size ∧ col ≤ b.size ∧
  (b.pieces.filter (λ p => p.1 = row)).length < 2 ∧
  (b.pieces.filter (λ p => p.2 = col)).length < 2

/-- Defines the game rules and winning condition -/
def secondPlayerWins (boardSize : Nat) : Prop :=
  ∀ (game : List (Nat × Nat)),
    (∀ (move : Nat × Nat), move ∈ game → validMove ⟨boardSize, game.take (game.indexOf move)⟩ move.1 move.2) →
    (game.length % 2 = 1) →
    ∃ (nextMove : Nat × Nat), validMove ⟨boardSize, game⟩ nextMove.1 nextMove.2

/-- The main theorem stating that the second player wins on a 65x65 board -/
theorem second_player_wins_65x65 : secondPlayerWins 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_65x65_l932_93239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mandy_black_shirts_l932_93290

/-- The number of packs of black shirts Mandy bought -/
def black_packs : ℕ := sorry

/-- The number of shirts in each pack of black shirts -/
def black_per_pack : ℕ := 5

/-- The number of packs of yellow shirts Mandy bought -/
def yellow_packs : ℕ := 3

/-- The number of shirts in each pack of yellow shirts -/
def yellow_per_pack : ℕ := 2

/-- The total number of shirts Mandy bought -/
def total_shirts : ℕ := 21

theorem mandy_black_shirts :
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = total_shirts →
  black_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mandy_black_shirts_l932_93290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_prime_factorization_sum_l932_93231

-- Define the problem statement
theorem min_x_prime_factorization_sum (x y a b c d : ℕ) : 
  (7 * x ^ 5 = 11 * y ^ 13) →  -- Given equation
  (x = a ^ c * b ^ d) →        -- Prime factorization of x
  (∀ x' y' : ℕ, (7 * x' ^ 5 = 11 * y' ^ 13) → x' ≥ x) →  -- x is minimum
  (a + b + c + d = 31) :=      -- The sum we want to prove
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_prime_factorization_sum_l932_93231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_29000_units_max_profit_value_l932_93247

/-- Annual fixed R&D cost in 10,000 yuan -/
noncomputable def fixed_cost : ℝ := 50

/-- Production cost per 10,000 units in 10,000 yuan -/
noncomputable def production_cost_per_unit : ℝ := 80

/-- Sales revenue function for 0 < x ≤ 20 -/
noncomputable def revenue_function_1 (x : ℝ) : ℝ := 180 - 2*x

/-- Sales revenue function for x > 20 -/
noncomputable def revenue_function_2 (x : ℝ) : ℝ := 70 + 2000/x - 9000/(x*(x+1))

/-- Profit function for 0 < x ≤ 20 -/
noncomputable def profit_function_1 (x : ℝ) : ℝ := 
  x * revenue_function_1 x - production_cost_per_unit * x - fixed_cost

/-- Profit function for x > 20 -/
noncomputable def profit_function_2 (x : ℝ) : ℝ := 
  x * revenue_function_2 x - production_cost_per_unit * x - fixed_cost

/-- The maximum profit is achieved at x = 2.9 (29,000 units) -/
theorem max_profit_at_29000_units :
  ∀ x > 0, profit_function_2 2.9 ≥ profit_function_1 x ∧ 
           profit_function_2 2.9 ≥ profit_function_2 x :=
by sorry

/-- The maximum profit is 1360 (13,600,000 yuan) -/
theorem max_profit_value :
  profit_function_2 2.9 = 1360 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_29000_units_max_profit_value_l932_93247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_converges_sequence_a_2018_value_l932_93207

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => (9 - 4 * sequence_a n) / (4 - sequence_a n)

theorem sequence_a_converges : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_a n - (4 - Real.sqrt 7)| < ε :=
by
  sorry

-- We can't directly evaluate sequence_a 2018 as it's noncomputable
-- Instead, we can state a theorem about its value
theorem sequence_a_2018_value :
  ∃ δ > 0, |sequence_a 2018 - (4 - Real.sqrt 7)| < δ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_converges_sequence_a_2018_value_l932_93207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l932_93294

-- Define the sum of the geometric sequence
noncomputable def S (n : ℕ) (t : ℝ) : ℝ := t * 3^(n-1) - 1/3

-- Define the function y
noncomputable def y (x t : ℝ) : ℝ := ((x+2)*(x+10))/(x+t)

-- Theorem statement
theorem min_value_of_y (t : ℝ) :
  (∀ n : ℕ, S n t = t * 3^(n-1) - 1/3) →
  (∃ x : ℝ, x > 0 ∧ ∀ z : ℝ, z > 0 → y z t ≥ y x t) →
  (∃ x : ℝ, x > 0 ∧ y x t = 16) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_y_l932_93294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_eq_nineteen_point_five_l932_93270

/-- A function satisfying the given equation for all real numbers -/
noncomputable def f : ℝ → ℝ := sorry

/-- The equation that f satisfies for all real x -/
axiom f_eq (x : ℝ) : (2 - x) * f x - 2 * f (3 - x) = -x^3 + 5*x - 18

/-- Theorem stating that f(0) = 19.5 -/
theorem f_zero_eq_nineteen_point_five : f 0 = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_eq_nineteen_point_five_l932_93270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_radii_relation_l932_93260

/-- Predicate to define an equilateral triangle with given side length, inradius, and exradius -/
def is_equilateral_triangle (a ρ ρ' : ℝ) : Prop :=
  ρ = (Real.sqrt 3 / 6) * a ∧ ρ' = (Real.sqrt 3 / 2) * a

/-- For an equilateral triangle with side length a, inradius ρ, and exradius ρ',
    the equation ρ'(ρ + ρ') = a² holds. -/
theorem equilateral_triangle_radii_relation (a ρ ρ' : ℝ) : 
  a > 0 → ρ > 0 → ρ' > 0 → is_equilateral_triangle a ρ ρ' → ρ' * (ρ + ρ') = a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_radii_relation_l932_93260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sidewalk_concrete_order_l932_93279

/-- Represents the dimensions of a sidewalk in feet and inches -/
structure SidewalkDimensions where
  width : ℚ  -- width in feet
  length : ℚ  -- length in feet
  thickness : ℚ  -- thickness in inches

/-- Calculates the volume of concrete needed in cubic yards -/
def concreteVolume (d : SidewalkDimensions) : ℚ :=
  (d.width / 3) * (d.length / 3) * (d.thickness / 36)

/-- Rounds up a rational number to the nearest natural number -/
def ceilRat (q : ℚ) : ℕ :=
  Int.toNat (Int.ceil q)

theorem sidewalk_concrete_order (d : SidewalkDimensions) 
  (h1 : d.width = 4)
  (h2 : d.length = 80)
  (h3 : d.thickness = 4) :
  ceilRat (concreteVolume d) = 4 := by
  sorry

#eval ceilRat (320 / 81)  -- Should output 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sidewalk_concrete_order_l932_93279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ella_received_twelve_l932_93216

/-- The number of strawberries Ella received -/
def ellas_strawberries : ℕ := sorry

/-- The number of strawberries Noah received -/
def noahs_strawberries : ℕ := sorry

/-- Ella received 8 more strawberries than Noah -/
axiom ella_got_more : ellas_strawberries = noahs_strawberries + 8

/-- Noah received one-third of the strawberries Ella received -/
axiom noah_got_third : 3 * noahs_strawberries = ellas_strawberries

/-- Theorem: Ella received 12 strawberries -/
theorem ella_received_twelve : ellas_strawberries = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ella_received_twelve_l932_93216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_one_l932_93280

def is_irreducible (n d : ℕ) : Prop := Nat.gcd n d = 1

def is_valid_fraction (n d : ℕ) : Prop :=
  n ∈ Finset.range 10 \ {0} ∧ 
  d ∈ Finset.range 10 \ {0} ∧ 
  n ≠ d ∧
  is_irreducible n d

theorem fraction_product_one : 
  ∃ (n1 d1 n2 d2 n3 d3 : ℕ),
    is_valid_fraction n1 d1 ∧
    is_valid_fraction n2 d2 ∧
    is_valid_fraction n3 d3 ∧
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ d2 ∧ n1 ≠ d3 ∧
    n2 ≠ n3 ∧ n2 ≠ d1 ∧ n2 ≠ d3 ∧
    n3 ≠ d1 ∧ n3 ≠ d2 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧
    d2 ≠ d3 ∧
    (n1 : ℚ) * n2 * n3 = (d1 : ℚ) * d2 * d3 :=
by
  sorry

#check fraction_product_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_product_one_l932_93280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kSquared_exceeds_critical_value_l932_93237

/-- Represents the contingency table data --/
structure ContingencyTable where
  a : Nat -- on-time trips for company A
  b : Nat -- not on-time trips for company A
  c : Nat -- on-time trips for company B
  d : Nat -- not on-time trips for company B

/-- Calculates the K^2 value given a contingency table --/
noncomputable def calculateKSquared (table : ContingencyTable) : Real :=
  let n := table.a + table.b + table.c + table.d
  (n * (table.a * table.d - table.b * table.c)^2 : Real) /
    ((table.a + table.b) * (table.c + table.d) * (table.a + table.c) * (table.b + table.d))

/-- The critical value for K^2 at 90% confidence level --/
def criticalValue : Real := 2.706

/-- The given contingency table data --/
def surveyData : ContingencyTable := {
  a := 240
  b := 20
  c := 210
  d := 30
}

/-- Theorem stating that the calculated K^2 value is greater than the critical value --/
theorem kSquared_exceeds_critical_value :
  calculateKSquared surveyData > criticalValue := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kSquared_exceeds_critical_value_l932_93237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_and_t_relation_l932_93273

-- Define the polynomial and its roots
def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 8*x - 1

-- Define the roots a, b, c
variable (a b c : ℝ)

-- Define t
noncomputable def t (a b c : ℝ) : ℝ := Real.sqrt a + Real.sqrt b + Real.sqrt c

-- State the theorem
theorem roots_and_t_relation 
  (ha : p a = 0) 
  (hb : p b = 0) 
  (hc : p c = 0) : 
  let t := t a b c
  (t^4 - 20*t^2 + 4*t) = (-8*t^2 + 12*t - 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_and_t_relation_l932_93273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l932_93284

/-- Given a right prism with a rhombus base, this theorem calculates the angle between
    a specific cutting plane and the base plane. -/
theorem angle_between_planes (α k : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : k > 0) :
  Real.arctan (k / (2 * Real.sin α)) = Real.arctan (k / (2 * Real.sin α)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l932_93284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_value_l932_93288

/-- Sequence a_n with given properties -/
def a : ℕ → ℝ := sorry

/-- Partial sum S_n of sequence a_n -/
def S : ℕ → ℝ := sorry

/-- The first term of the sequence is 2 -/
axiom a_1 : a 1 = 2

/-- The relationship between S_n, a_{n+1}, and 4^n -/
axiom sequence_relation (n : ℕ) : 6 * S n = 3 * a (n + 1) + 4^n - 1

/-- The maximum value of S_n is 35 -/
theorem max_S_value : ∃ n : ℕ, S n = 35 ∧ ∀ m : ℕ, S m ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_value_l932_93288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l932_93234

theorem inscribed_squares_area_ratio (t r : ℝ) (h_perimeter : 3 * t = 2 * Real.pi * r) :
  (Real.pi * r / 3)^2 / (2 * r^2) = Real.pi^2 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l932_93234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l932_93298

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

-- Theorem stating that g is an odd function
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l932_93298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_ten_cos_theta_equals_three_l932_93296

/-- Given vectors a and b in R^2, where a = (3, 3) and 2b - a = (-1, 1),
    prove that √10 * cos(θ) = 3, where θ is the angle between a and b. -/
theorem sqrt_ten_cos_theta_equals_three (a b : ℝ × ℝ) (θ : ℝ) : 
  a = (3, 3) → 
  2 • b - a = (-1, 1) → 
  θ = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  Real.sqrt 10 * Real.cos θ = 3 := by
  sorry

#check sqrt_ten_cos_theta_equals_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_ten_cos_theta_equals_three_l932_93296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erin_walk_time_l932_93278

/-- Proves that if Erin walks 3/5 of the way home in 30 minutes at a constant rate,
    it will take her 20 minutes to walk the remaining distance home. -/
theorem erin_walk_time (total_distance : ℝ) (h1 : total_distance > 0) : 
  let initial_fraction : ℚ := 3/5
  let initial_time : ℝ := 30
  let remaining_fraction : ℚ := 1 - initial_fraction
  let remaining_time : ℝ := (initial_time / (initial_fraction : ℝ)) * (remaining_fraction : ℝ)
  remaining_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erin_walk_time_l932_93278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l932_93241

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality (h_odd : ∀ x ∈ Set.Icc (-5 : ℝ) 5, f (-x) = -f x)
  (h_domain : Set.range f ⊆ Set.Icc (-5 : ℝ) 5)
  (h_negative : ∀ x ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioc 2 5, x ≥ 0 → f x < 0) :
  {x | x ∈ Set.Icc (-5 : ℝ) 5 ∧ f x < 0} = Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioc 2 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l932_93241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l932_93259

def sequence_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_property (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, sequence_sum a n = 2 * a n - 4) →
  (∀ n : ℕ+, a n = 2^((n : ℝ) + 1)) :=
by
  intro h
  intro n
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l932_93259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_a_range_correct_l932_93208

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*a*x else (2*a - 1)*x - 3*a + 6

-- State the theorem
theorem f_increasing_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) ↔ a ∈ Set.Icc 1 2 := by
  sorry

-- Define the range of a
def a_range : Set ℝ := Set.Icc 1 2

-- State the main theorem
theorem a_range_correct : 
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) ↔ a ∈ a_range := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_a_range_correct_l932_93208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_is_sqrt_five_l932_93214

/-- The function f(x) = 2x + 5/x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 5 / x

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

/-- Distance from a point (x, y) to the line y = 2x -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |y - line x| / Real.sqrt 5

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem distance_product_is_sqrt_five (x : ℝ) (hx : x ≠ 0) :
  distance_to_line x (f x) * distance_to_y_axis x = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_is_sqrt_five_l932_93214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_expensive_trip_cost_l932_93240

-- Define the cities and distances
structure City :=
  (name : String)

noncomputable def distance (c1 c2 : City) : ℝ :=
  match c1.name, c2.name with
  | "D", "E" => 4000
  | "E", "D" => 4000
  | "D", "F" => 4200
  | "F", "D" => 4200
  | "E", "F" => Real.sqrt (4200^2 - 4000^2)
  | "F", "E" => Real.sqrt (4200^2 - 4000^2)
  | _, _ => 0

-- Define travel costs
def busCost (d : ℝ) : ℝ := 0.20 * d
def planeCost (d : ℝ) : ℝ := 120 + 0.12 * d

-- Define function to choose cheapest travel mode
noncomputable def cheapestCost (d : ℝ) : ℝ := min (busCost d) (planeCost d)

-- Theorem statement
theorem least_expensive_trip_cost 
  (D E F : City)
  (h1 : D.name = "D")
  (h2 : E.name = "E")
  (h3 : F.name = "F")
  : cheapestCost (distance D E) + cheapestCost (distance E F) + cheapestCost (distance F D) = 1480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_expensive_trip_cost_l932_93240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_l932_93252

-- Define a function to represent a line
def Line (a b c : ℝ) := {(x, y) : ℝ × ℝ | a * x + b * y + c = 0}

-- Define a function to check if two lines are perpendicular
def IsPerpendicularTo (l1 l2 : ℝ × ℝ → Prop) : Prop :=
  ∃ a1 b1 a2 b2 : ℝ, (∀ x y, l1 (x, y) ↔ a1 * x + b1 * y = 0) ∧
                     (∀ x y, l2 (x, y) ↔ a2 * x + b2 * y = 0) ∧
                     a1 * a2 + b1 * b2 = 0

theorem perpendicular_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, IsPerpendicularTo (Line (a^2) (-1) 1) (Line 1 (-a) (-2))) ↔ 
  a = -1 ∨ a = 1 :=
sorry

#check perpendicular_lines_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_l932_93252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_is_eight_l932_93221

/-- The numerator of our rational function -/
noncomputable def numerator (x : ℝ) : ℝ := 3*x^8 + 5*x^7 - 7*x^3 + 2

/-- A rational function with the given numerator and a denominator q(x) -/
noncomputable def rational_function (q : ℝ → ℝ) (x : ℝ) : ℝ := numerator x / q x

/-- A function has a horizontal asymptote if it converges to a finite value as x approaches infinity -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The main theorem: The smallest possible degree of q(x) is 8 -/
theorem smallest_degree_is_eight :
  ∀ q : ℝ → ℝ, has_horizontal_asymptote (rational_function q) →
  (∀ p : ℝ → ℝ, has_horizontal_asymptote (rational_function p) → degree q ≤ degree p) →
  degree q = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_degree_is_eight_l932_93221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_free_cells_l932_93263

/-- Represents a grid of size n x n -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Represents an isosceles right triangle with leg length 1 -/
structure Triangle where
  x : ℕ
  y : ℕ
  orientation : Bool  -- true for upper, false for lower

/-- Checks if a triangle covers a given grid segment -/
def covers_segment (t : Triangle) (x y : ℕ) : Bool :=
  sorry

/-- Checks if a triangle occupies half of a grid square -/
def occupies_half_square (t : Triangle) : Bool :=
  sorry

/-- Counts the number of cells without triangles in a grid -/
def count_free_cells (n : ℕ) (triangles : List Triangle) : ℕ :=
  sorry

/-- The main theorem -/
theorem max_free_cells (n : ℕ) (h : n = 100) :
  ∃ (triangles : List Triangle),
    (∀ t, t ∈ triangles → occupies_half_square t) ∧
    (∀ x y : Fin n, ∃! t, t ∈ triangles ∧ covers_segment t x.val y.val) ∧
    count_free_cells n triangles = 2450 ∧
    (∀ triangles' : List Triangle,
      (∀ t, t ∈ triangles' → occupies_half_square t) →
      (∀ x y : Fin n, ∃! t, t ∈ triangles' ∧ covers_segment t x.val y.val) →
      count_free_cells n triangles' ≤ 2450) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_free_cells_l932_93263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_min_distance_l932_93272

noncomputable def trajectory (x y : ℝ) : Prop :=
  y^2 = -4 * (x - 1) ∧ x ≤ 1

noncomputable def min_distance (m : ℝ) : ℝ :=
  if m ≥ -1 then |m - 1| else 2 * Real.sqrt (-m)

theorem circle_trajectory_and_min_distance :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), (x - 3)^2 + y^2 = r^2 ∧ (x + r)^2 + y^2 = 1) →
    (trajectory x y ∧
     ∀ (m : ℝ),
       Real.sqrt ((x - m)^2 + y^2) ≥ min_distance m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_trajectory_and_min_distance_l932_93272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lopez_account_balance_l932_93295

/-- Calculates the final amount in an account after compound interest is applied -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Theorem: Ms. Lopez's account balance after one year -/
theorem lopez_account_balance :
  let principal : ℝ := 100
  let rate : ℝ := 0.20
  let frequency : ℝ := 2
  let time : ℝ := 1
  compound_interest principal rate frequency time = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lopez_account_balance_l932_93295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l932_93255

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 8 * x + 12 * y - 20 = 0

/-- The area of the circle described by the given equation -/
noncomputable def circle_area : ℝ := (33 / 4) * Real.pi

theorem circle_area_proof :
  ∃ (x₀ y₀ r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  circle_area = Real.pi * r^2 := by
  -- Existential introduction
  use 1, -3/2, Real.sqrt (33/4)
  constructor
  
  -- First part: equivalence of equations
  · intro x y
    constructor
    · intro h
      -- Convert the original equation to standard form
      sorry
    · intro h
      -- Convert the standard form back to the original equation
      sorry
  
  -- Second part: area calculation
  · -- Show that the area matches the calculated value
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l932_93255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_Q_l932_93291

def a : Nat := 12345678987654321
def b : Nat := 23456789123
def c : Nat := 11

def Q : Nat := a * b * c

theorem digits_of_Q : (Nat.log Q 10 + 1 : Nat) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_Q_l932_93291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_27_negative_third_l932_93210

theorem power_of_27_negative_third (x : ℝ) : x = 27^(-(1/3 : ℝ)) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_27_negative_third_l932_93210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldbach_max_diff_l932_93209

/-- Two different prime numbers that sum to 156 -/
structure GoldbachPair :=
  (p q : ℕ)
  (p_prime : Nat.Prime p)
  (q_prime : Nat.Prime q)
  (sum_eq : p + q = 156)
  (not_equal : p ≠ q)

/-- The theorem stating the maximum difference between two primes summing to 156 -/
theorem goldbach_max_diff : 
  ∀ (pair : GoldbachPair), (pair.q - pair.p : ℤ).natAbs ≤ 146 ∧ 
  ∃ (max_pair : GoldbachPair), (max_pair.q - max_pair.p : ℤ).natAbs = 146 :=
by
  sorry

#check goldbach_max_diff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldbach_max_diff_l932_93209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l932_93253

noncomputable def f1 (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def f2 (x : ℝ) : ℝ := Real.log 5 / Real.log x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log (1/5)
noncomputable def f4 (x : ℝ) : ℝ := Real.log (1/5) / Real.log x

def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x : ℝ), x > 0 ∧
    (((f1 x = f2 x) ∨ (f1 x = f3 x) ∨ (f1 x = f4 x) ∨
      (f2 x = f3 x) ∨ (f2 x = f4 x) ∨ (f3 x = f4 x)) ∧
     p = (x, f1 x))}

-- Theorem statement
theorem intersection_count :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 3 ∧ (∀ p ∈ intersection_points, p ∈ S) ∧ (∀ p ∈ S, p ∈ intersection_points) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l932_93253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_question_with_different_answers_l932_93261

/-- A type representing questions that can be asked -/
def Question : Type := String

/-- A type representing answers to questions -/
def Answer : Type := String

/-- A function that represents the correct answer to a question at a given time -/
noncomputable def correctAnswer (q : Question) (t : ℝ) : Answer := sorry

/-- A predicate that determines if a person is a truthteller -/
def isTruthteller (person : Question → Answer) : Prop :=
  ∀ q t, person q = correctAnswer q t

/-- The main theorem: there exists a question that a truthteller can answer differently at different times -/
theorem exists_question_with_different_answers :
  ∃ (q : Question) (t₁ t₂ : ℝ) (person : Question → Answer),
    isTruthteller person ∧ t₁ ≠ t₂ ∧ correctAnswer q t₁ ≠ correctAnswer q t₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_question_with_different_answers_l932_93261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_is_11_l932_93274

/-- The cost of an adult ticket for a play, given the following conditions:
  * Child tickets cost $10
  * 23 people attended the performance
  * Total collected from ticket sales: $246
  * 7 children attended the play
-/
def adult_ticket_cost (child_ticket_cost total_attendees total_sales child_attendees : ℕ) : ℕ :=
  if child_ticket_cost = 10 ∧
     total_attendees = 23 ∧
     total_sales = 246 ∧
     child_attendees = 7 ∧
     (total_attendees - child_attendees) * 11 + child_attendees * child_ticket_cost = total_sales
  then 11
  else 0

theorem adult_ticket_cost_is_11 
    (child_ticket_cost total_attendees total_sales child_attendees : ℕ) 
    (h1 : child_ticket_cost = 10)
    (h2 : total_attendees = 23)
    (h3 : total_sales = 246)
    (h4 : child_attendees = 7)
    (h5 : (total_attendees - child_attendees) * 11 + child_attendees * child_ticket_cost = total_sales) :
  adult_ticket_cost child_ticket_cost total_attendees total_sales child_attendees = 11 := by
  simp [adult_ticket_cost, h1, h2, h3, h4, h5]

#eval adult_ticket_cost 10 23 246 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_cost_is_11_l932_93274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l932_93243

noncomputable def f (x : ℝ) := Real.sqrt (4 - Real.sqrt (6 - Real.sqrt x))

theorem domain_of_f : 
  {x : ℝ | f x ≠ 0 ∨ f x = 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 36} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l932_93243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l932_93266

noncomputable def f (x : ℝ) := Real.log (2 - x) + 1

theorem max_ab_value (a b : ℝ) (h1 : f 1 = 1) (h2 : a + b = 1) :
  a * b ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l932_93266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l932_93217

def has_exactly_three_integer_solutions (a : ℝ) : Prop :=
  ∃! (x y z : ℤ), 
    (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    ((↑x : ℝ) / 2 + ((↑x : ℝ) + 1) / 3 > 0) ∧
    (3 * (↑x : ℝ) + 5 * a + 4 > 4 * ((↑x : ℝ) + 1) + 3 * a) ∧
    ((↑y : ℝ) / 2 + ((↑y : ℝ) + 1) / 3 > 0) ∧
    (3 * (↑y : ℝ) + 5 * a + 4 > 4 * ((↑y : ℝ) + 1) + 3 * a) ∧
    ((↑z : ℝ) / 2 + ((↑z : ℝ) + 1) / 3 > 0) ∧
    (3 * (↑z : ℝ) + 5 * a + 4 > 4 * ((↑z : ℝ) + 1) + 3 * a)

theorem inequality_system_solution_range :
  ∀ a : ℝ, has_exactly_three_integer_solutions a ↔ (1 < a ∧ a ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l932_93217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l932_93242

/-- The area of a quadrilateral given its diagonal and offsets -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * (offset1 + offset2)

/-- Theorem: For a quadrilateral with diagonal 30, offset1 10, and area 240, offset2 is 6 -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 30 → offset1 = 10 → area = 240 →
  ∃ (offset2 : ℝ), quadrilateralArea diagonal offset1 offset2 = area ∧ offset2 = 6 := by
  intros h1 h2 h3
  use 6
  apply And.intro
  · simp [quadrilateralArea, h1, h2, h3]
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l932_93242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_detection_distance_for_specific_case_minimum_detection_distance_is_optimal_l932_93232

/-- Represents the minimum distance a sapper needs to walk to detect mines in an equilateral triangle -/
noncomputable def minimum_detection_distance (side_length : ℝ) (detector_range : ℝ) : ℝ :=
  Real.sqrt 7 - Real.sqrt 3 / 2

/-- Theorem stating the minimum detection distance for a specific triangle and detector range -/
theorem minimum_detection_distance_for_specific_case :
  minimum_detection_distance 2 (Real.sqrt 3 / 2) = Real.sqrt 7 - Real.sqrt 3 / 2 :=
by sorry

/-- Predicate indicating whether a point is not covered by the detection path -/
def point_not_covered (side_length : ℝ) (detector_range : ℝ) (path_length : ℝ) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem proving the optimality of the minimum detection distance -/
theorem minimum_detection_distance_is_optimal
  (side_length : ℝ)
  (detector_range : ℝ)
  (h1 : side_length = 2)
  (h2 : detector_range = Real.sqrt 3 / 2) :
  ∀ (path_length : ℝ),
    (path_length ≥ minimum_detection_distance side_length detector_range) ∨
    (∃ (point : ℝ × ℝ), point_not_covered side_length detector_range path_length point) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_detection_distance_for_specific_case_minimum_detection_distance_is_optimal_l932_93232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_explicit_l932_93256

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 3 * a (n + 2) + 18 * a (n + 1) + 2^(n + 1)

/-- Explicit formula for a_n -/
def a_explicit (n : ℕ) : ℚ :=
  (6^n / 32) + (7 * (-3)^n / 16) - (2^n / 16)

/-- Theorem stating that the explicit formula matches the recursive definition -/
theorem a_equals_a_explicit :
  ∀ n : ℕ, a n = a_explicit n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_a_explicit_l932_93256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l932_93262

-- Define the type for positive integers
def PositiveInt := {n : ℤ // n > 0}

-- Define the function type
def RationalToInt := ℚ → ℤ

-- State the theorem
theorem function_characterization (f : RationalToInt) : 
  (∀ (x : ℚ) (a : ℤ) (b : ℕ+), 
    f ((f x + a : ℚ) / (b : ℚ)) = f ((x + a : ℚ) / (b : ℚ))) →
  ((∃ (c : ℤ), ∀ (x : ℚ), f x = c) ∨
   (∀ (x : ℚ), f x = Int.floor x) ∨
   (∀ (x : ℚ), f x = Int.ceil x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l932_93262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_third_quadrant_tan_to_fraction_l932_93286

-- Problem 1
theorem sin_value_third_quadrant (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : α ∈ Set.Icc (Real.pi) (3/2 * Real.pi)) :
  Real.sin α = -3/5 := by sorry

-- Problem 2
theorem tan_to_fraction (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin θ + Real.cos θ) / (2 * Real.sin θ + Real.cos θ) = 4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_third_quadrant_tan_to_fraction_l932_93286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l932_93205

theorem loan_interest_rate : 
  ∀ (principal repayment : ℝ), 
  principal > 0 → 
  repayment > principal →
  principal = 136 →
  repayment = 150 →
  ∃ (rate : ℝ), 
    (rate ≥ 0.099 ∧ rate ≤ 0.101) ∧ 
    repayment = principal * (1 + rate) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_interest_rate_l932_93205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l932_93271

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x

theorem solution_set_of_inequality :
  {x : ℝ | f (2 - x^2) + f (2*x + 1) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l932_93271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plot_dimensions_l932_93215

theorem square_plot_dimensions :
  ∀ (side_length : ℕ) (area : ℕ),
    (side_length * side_length = area) →
    (1000 ≤ area) →
    (area ≤ 9999) →
    (∀ d : ℕ, d ∈ Nat.digits 10 area → Even d) →
    (Nat.digits 10 area).Nodup →
    (side_length = 78 ∧ area = 6084) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plot_dimensions_l932_93215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_coins_theorem_l932_93299

def initial_pennies : ℕ := 36
def initial_nickels : ℕ := 31
def borrowed_nickels : ℕ := 20
def euro_to_dollar_rate : ℚ := 118/100
def nickel_value : ℚ := 5/100
def penny_value : ℚ := 1/100

def remaining_nickels : ℕ := initial_nickels - borrowed_nickels

def total_value : ℚ := 
  (remaining_nickels : ℚ) * nickel_value + (initial_pennies : ℚ) * penny_value

def euros_value : ℚ := total_value / euro_to_dollar_rate

theorem sandy_coins_theorem :
  remaining_nickels = 11 ∧ 
  (euros_value * 100).floor / 100 = 77/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_coins_theorem_l932_93299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_value_l932_93222

theorem square_value (p square : ℝ) (h1 : square + p = 75) (h2 : (square + p) + 2*p = 139) : square = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_value_l932_93222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l932_93250

/-- The function f(x) = 2^(x-1) + x - 5 -/
noncomputable def f (x : ℝ) : ℝ := 2^(x-1) + x - 5

/-- Theorem: There exists a unique solution to the equation 2^(x-1) + x = 5 in the interval (2, 3) -/
theorem solution_in_interval :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l932_93250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_equal_min_questions_plus_one_l932_93258

/-- A binary string of length n is a function from ℕ to Bool where the domain is restricted to {1, ..., n} -/
def BinaryString (n : ℕ) := {i : ℕ // i ≤ n} → Bool

/-- The set of m binary strings of length n -/
def BinaryStringSet (m n : ℕ) := {S : Finset (BinaryString n) // S.card = m}

/-- A question asks about the value of bit i in string j -/
structure Question (m n : ℕ) where
  i : ℕ
  j : ℕ
  h1 : i ≤ n
  h2 : j ≤ m

/-- The minimum number of questions needed to find a new string -/
def minQuestions (m n : ℕ) : ℕ := sorry

theorem min_questions_equal (n : ℕ) :
  minQuestions n n = n := by
  sorry

theorem min_questions_plus_one (n : ℕ) :
  minQuestions (n + 1) n = n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_equal_min_questions_plus_one_l932_93258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l932_93248

noncomputable def f (x : ℝ) := Real.sin (x - 3 * Real.pi / 2) * Real.sin x - Real.sqrt 3 * (Real.cos x) ^ 2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
    T = Real.pi ∧
    (∀ x, f x ≤ 1 - Real.sqrt 3 / 2) ∧
    (∃ x, f x = 1 - Real.sqrt 3 / 2) ∧
    (∀ x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 12), 
     ∀ y ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 12), 
     x ≤ y → f x ≤ f y) ∧
    (∀ x ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3), 
     ∀ y ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3), 
     x ≤ y → f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l932_93248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l932_93244

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x => x ^ α

-- State the theorem
theorem power_function_through_point :
  ∃ α : ℝ, (power_function α 2 = 1/4) ∧ (α = -2) := by
  -- Provide the value of α
  use -2
  
  constructor
  · -- Prove that power_function (-2) 2 = 1/4
    simp [power_function]
    norm_num
  
  · -- Prove that α = -2
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l932_93244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_perimeter_ratio_l932_93204

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a function to calculate the area of a triangle
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Define a function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define a function for the ratio of area to perimeter
noncomputable def areaPerimeterRatio (t : Triangle) : ℝ :=
  area t / perimeter t

-- Define what it means for one triangle to be nested in another
def isNestedIn (t1 t2 : Triangle) : Prop :=
  ∃ (k : ℝ), 0 < k ∧ k ≤ 1 ∧
    t1.a = k * t2.a ∧ t1.b = k * t2.b ∧ t1.c = k * t2.c

-- Theorem statement
theorem max_area_perimeter_ratio (t : Triangle) :
  ∀ (t' : Triangle), isNestedIn t' t → areaPerimeterRatio t' ≤ areaPerimeterRatio t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_perimeter_ratio_l932_93204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l932_93275

/-- The eccentricity of a hyperbola with equation x²/a² - y² = 1 where a > 1 -/
noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ := Real.sqrt (1 + 1 / a^2)

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  1 < hyperbola_eccentricity a ∧ hyperbola_eccentricity a < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l932_93275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_l932_93276

structure Triangle where
  α : ℝ
  β : ℝ
  γ : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def circumcenter (t : Triangle) : (ℝ × ℝ × ℝ) :=
  (Real.sin (2 * t.α), Real.sin (2 * t.β), Real.sin (2 * t.γ))

def incenter (t : Triangle) : (ℝ × ℝ × ℝ) :=
  (t.a, t.b, t.c)

noncomputable def orthocenter (t : Triangle) : (ℝ × ℝ × ℝ) :=
  (Real.tan t.α, Real.tan t.β, Real.tan t.γ)

theorem triangle_centers (t : Triangle) :
  (circumcenter t = (Real.sin (2 * t.α), Real.sin (2 * t.β), Real.sin (2 * t.γ))) ∧
  (incenter t = (t.a, t.b, t.c)) ∧
  (orthocenter t = (Real.tan t.α, Real.tan t.β, Real.tan t.γ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_l932_93276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_specific_number_eq_selections_not_uniform_l932_93213

/-- A circular arrangement of numbers with a selection process -/
structure CircularSelection (n : ℕ) (k : ℕ) where
  h_k_le_n : k ≤ n

/-- The probability of a specific number appearing in the selection -/
def prob_specific_number (cs : CircularSelection n k) : ℚ :=
  k / n

/-- A selection of k numbers from n possibilities -/
def Selection (n : ℕ) (k : ℕ) := Fin k → Fin n

/-- The probability of obtaining a specific selection -/
noncomputable def prob_selection (cs : CircularSelection n k) (s : Selection n k) : ℝ :=
  sorry

/-- Theorem stating that the probability of each specific number is k/n -/
theorem prob_specific_number_eq (cs : CircularSelection n k) :
  prob_specific_number cs = k / n := by
  sorry

/-- Theorem stating that not all selections are equally likely -/
theorem selections_not_uniform (cs : CircularSelection n k) :
  ∃ s₁ s₂ : Selection n k, prob_selection cs s₁ ≠ prob_selection cs s₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_specific_number_eq_selections_not_uniform_l932_93213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_equals_forty_pencils_l932_93251

/-- The number of pencils Patrick purchased -/
def num_pencils : ℕ := 90

/-- The ratio of cost to selling price for the batch of pencils -/
noncomputable def cost_to_sell_ratio : ℝ := 1.4444444444444444

/-- Calculates the number of pencils whose selling price equals the total loss -/
noncomputable def pencils_equal_to_loss (n : ℕ) (r : ℝ) : ℝ :=
  n * (r - 1) / r

theorem loss_equals_forty_pencils :
  pencils_equal_to_loss num_pencils cost_to_sell_ratio = 40 := by
  sorry

#eval num_pencils

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_equals_forty_pencils_l932_93251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_negative_l932_93238

/-- Represents the state of the whiteboard at a given time step -/
structure BoardState where
  R : ℤ  -- Sum of every other number
  S : ℤ  -- Sum of the remaining numbers

/-- The transformation rule for the board state -/
def transform (state : BoardState) : BoardState :=
  { R := 4040 * state.S - state.R,
    S := 4040 * state.R - state.S }

/-- Proposition: For any initial set of 50 consecutive positive integers,
    there will eventually be a negative number on the board -/
theorem eventually_negative 
  (initial : BoardState) 
  (h_initial : ∃ (start : ℤ), 
    initial.R + initial.S = (start + 49) * 50 / 2 ∧ 
    initial.R - initial.S ≠ 0) : 
  ∃ (t : ℕ), (Nat.iterate transform t initial).R < 0 ∨ (Nat.iterate transform t initial).S < 0 :=
by
  sorry

#check eventually_negative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_negative_l932_93238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_polynomial_rational_root_l932_93292

/-- A polynomial of the form x^p - a, where p is prime and a is rational -/
noncomputable def PrimePolynomial (p : ℕ) (a : ℚ) : Polynomial ℚ :=
  Polynomial.X ^ p - Polynomial.C a

/-- The property of being factorizable into two rational polynomials of degree at least 1 -/
def IsFactorizable (f : Polynomial ℚ) : Prop :=
  ∃ (g h : Polynomial ℚ), f = g * h ∧ g.degree ≥ 1 ∧ h.degree ≥ 1

theorem prime_polynomial_rational_root 
  (p : ℕ) (hp : Nat.Prime p) (a : ℚ) (hf : IsFactorizable (PrimePolynomial p a)) :
  ∃ (r : ℚ), (PrimePolynomial p a).eval r = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_polynomial_rational_root_l932_93292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_rate_calculation_l932_93200

/-- The rate of apples per kg -/
def apple_rate (n : ℕ) : Prop := n = 70

/-- The amount of apples purchased in kg -/
def apples_purchased : ℕ := 8

/-- The amount of mangoes purchased in kg -/
def mangoes_purchased : ℕ := 9

/-- The rate of mangoes per kg -/
def mango_rate : ℕ := 55

/-- The total amount paid -/
def total_paid : ℕ := 1055

theorem apple_rate_calculation : apple_rate 70 :=
  by
    -- Unfold the definition of apple_rate
    unfold apple_rate
    -- The definition directly states that 70 = 70, which is true
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_rate_calculation_l932_93200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l932_93268

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in a 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- A point is on a circle if its distance from the center equals the radius -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  distance c.center p = c.radius

/-- A circle is tangent to the y-axis if its distance from the y-axis equals its radius -/
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

theorem circle_equation (c : Circle) (h1 : c.center = (3, 0)) (h2 : tangentToYAxis c) :
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x - 3)^2 + y^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l932_93268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l932_93293

/-- The function f(x) = x^2 - 2x + 1 + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2*x - 2 + a/x

theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
   f' a x₁ = 0 ∧ f' a x₂ = 0 ∧
   (∀ x : ℝ, x > 0 → f' a x = 0 → (x = x₁ ∨ x = x₂))) →
  0 < a ∧ a < 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_imply_a_range_l932_93293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_to_product_l932_93267

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (6 * x) + Real.sin (8 * x) = 2 * Real.sin (7 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_to_product_l932_93267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_not_increasing_l932_93225

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem inverse_proportion_not_increasing :
  ¬ (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₂ > x₁ → f x₂ > f x₁) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_not_increasing_l932_93225
