import Mathlib

namespace NUMINAMATH_CALUDE_girls_who_bought_balloons_l3224_322430

def initial_balloons : ℕ := 3 * 12
def boys_bought : ℕ := 3
def remaining_balloons : ℕ := 21

theorem girls_who_bought_balloons :
  initial_balloons - remaining_balloons - boys_bought = 12 :=
by sorry

end NUMINAMATH_CALUDE_girls_who_bought_balloons_l3224_322430


namespace NUMINAMATH_CALUDE_m_range_l3224_322428

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem m_range (m : ℝ) : 
  (m > 0) → 
  (∀ x, ¬(p x) → ¬(q x m)) → 
  (∃ x, ¬(p x) ∧ (q x m)) → 
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3224_322428


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3224_322431

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 110 → -11 < y ∧ y < -10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3224_322431


namespace NUMINAMATH_CALUDE_number_of_black_balls_l3224_322423

/-- Given a bag with red, white, and black balls, prove the number of black balls -/
theorem number_of_black_balls
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (prob_red : ℚ)
  (prob_white : ℚ)
  (h1 : red_balls = 21)
  (h2 : prob_red = 21 / total_balls)
  (h3 : prob_white = white_balls / total_balls)
  (h4 : prob_red = 42 / 100)
  (h5 : prob_white = 28 / 100)
  (h6 : total_balls = red_balls + white_balls + black_balls) :
  black_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_black_balls_l3224_322423


namespace NUMINAMATH_CALUDE_mira_jogging_hours_l3224_322414

/-- Mira's jogging problem -/
theorem mira_jogging_hours :
  ∀ (h : ℝ),
  (h > 0) →  -- Ensure positive jogging time
  (5 * h * 5 = 50) →  -- Total distance covered in 5 days
  h = 2 := by
sorry

end NUMINAMATH_CALUDE_mira_jogging_hours_l3224_322414


namespace NUMINAMATH_CALUDE_disjoint_sets_property_l3224_322482

theorem disjoint_sets_property (A B : Set ℕ) (h1 : A ∩ B = ∅) (h2 : A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a ≠ b ∧ a > n ∧ b > n ∧
    (({a, b, a + b} : Set ℕ) ⊆ A ∨ ({a, b, a + b} : Set ℕ) ⊆ B) :=
by sorry

end NUMINAMATH_CALUDE_disjoint_sets_property_l3224_322482


namespace NUMINAMATH_CALUDE_f_zero_value_l3224_322489

def is_nonneg_int (x : ℤ) : Prop := x ≥ 0

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ m n, is_nonneg_int m → is_nonneg_int n →
    f (m^2 + n^2) = (f m - f n)^2 + f (2*m*n)

theorem f_zero_value (f : ℤ → ℤ) :
  (∀ x, is_nonneg_int (f x)) →
  functional_equation f →
  8 * f 0 + 9 * f 1 = 2006 →
  f 0 = 118 := by sorry

end NUMINAMATH_CALUDE_f_zero_value_l3224_322489


namespace NUMINAMATH_CALUDE_bank_robbery_culprits_l3224_322458

theorem bank_robbery_culprits (Alexey Boris Veniamin Grigory : Prop) :
  (¬Grigory → Boris ∧ ¬Alexey) →
  (Veniamin → ¬Alexey ∧ ¬Boris) →
  (Grigory → Boris) →
  (Boris → Alexey ∨ Veniamin) →
  (Alexey ∧ Boris ∧ Grigory ∧ ¬Veniamin) :=
by sorry

end NUMINAMATH_CALUDE_bank_robbery_culprits_l3224_322458


namespace NUMINAMATH_CALUDE_friend_rides_80_times_more_l3224_322472

/-- Tommy's effective riding area in square blocks -/
def tommy_area : ℚ := 1

/-- Tommy's friend's riding area in square blocks -/
def friend_area : ℚ := 80

/-- The ratio of Tommy's friend's riding area to Tommy's effective riding area -/
def area_ratio : ℚ := friend_area / tommy_area

theorem friend_rides_80_times_more : area_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_friend_rides_80_times_more_l3224_322472


namespace NUMINAMATH_CALUDE_dorothy_doughnut_price_l3224_322455

/-- Given Dorothy's doughnut business scenario, prove the selling price per doughnut. -/
theorem dorothy_doughnut_price 
  (ingredient_cost : ℚ) 
  (num_doughnuts : ℕ) 
  (profit : ℚ) 
  (h1 : ingredient_cost = 53)
  (h2 : num_doughnuts = 25)
  (h3 : profit = 22) :
  (ingredient_cost + profit) / num_doughnuts = 3 := by
  sorry

#eval (53 + 22) / 25

end NUMINAMATH_CALUDE_dorothy_doughnut_price_l3224_322455


namespace NUMINAMATH_CALUDE_bennys_kids_l3224_322429

/-- Prove that Benny has 18 kids given the conditions of the problem -/
theorem bennys_kids : ℕ :=
  let total_money : ℕ := 360
  let apple_cost : ℕ := 4
  let apples_per_kid : ℕ := 5
  let num_kids : ℕ := 18
  have h1 : total_money ≥ apple_cost * apples_per_kid * num_kids := by sorry
  have h2 : apples_per_kid > 0 := by sorry
  num_kids

/- Proof omitted -/

end NUMINAMATH_CALUDE_bennys_kids_l3224_322429


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l3224_322416

/-- Given vectors a and b in ℝ², prove that the value of t that makes (a - b) perpendicular to (a - t*b) is -11/30 -/
theorem orthogonal_vectors (a b : ℝ × ℝ) (h1 : a = (-3, 1)) (h2 : b = (2, 5)) :
  ∃ t : ℝ, t = -11/30 ∧ (a.1 - b.1, a.2 - b.2) • (a.1 - t * b.1, a.2 - t * b.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l3224_322416


namespace NUMINAMATH_CALUDE_next_number_is_1461_l3224_322447

/-- Represents the sequence generator function -/
def sequenceGenerator (n : ℕ) : ℕ := 
  100 + 15 + (n * (n + 1))

/-- Proves that the next number after 1445 in the sequence is 1461 -/
theorem next_number_is_1461 : 
  ∃ k, sequenceGenerator k = 1445 ∧ sequenceGenerator (k + 1) = 1461 :=
sorry

end NUMINAMATH_CALUDE_next_number_is_1461_l3224_322447


namespace NUMINAMATH_CALUDE_revenue_maximized_at_five_l3224_322470

def revenue (x : ℝ) : ℝ := (400 - 20*x) * (50 + 5*x)

theorem revenue_maximized_at_five :
  ∃ (max : ℝ), revenue 5 = max ∧ ∀ (x : ℝ), revenue x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_five_l3224_322470


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l3224_322457

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type) [AddCommGroup P] [Module L P]

-- Define the perpendicular and parallel relations
variable (perpLine : L → P → Prop)  -- Line perpendicular to plane
variable (perpPlane : P → P → Prop)  -- Plane perpendicular to plane
variable (parallel : P → P → Prop)  -- Plane parallel to plane

-- State the theorem
theorem line_perp_parallel_planes 
  (l : L) (α β : P) 
  (h1 : perpLine l β) 
  (h2 : parallel α β) : 
  perpLine l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l3224_322457


namespace NUMINAMATH_CALUDE_eggs_in_two_boxes_l3224_322461

def eggs_per_box : ℕ := 3
def number_of_boxes : ℕ := 2

theorem eggs_in_two_boxes :
  eggs_per_box * number_of_boxes = 6 := by sorry

end NUMINAMATH_CALUDE_eggs_in_two_boxes_l3224_322461


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l3224_322441

-- Equation 1
theorem equation_one_solution :
  ∃ x : ℝ, (x ≠ 0 ∧ x ≠ 1) ∧ (9 / x = 8 / (x - 1)) → x = 9 :=
sorry

-- Equation 2
theorem equation_two_solution :
  ∃ x : ℝ, (x ≠ 2) ∧ (1 / (x - 2) - 3 = (x - 1) / (2 - x)) → x = 3 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l3224_322441


namespace NUMINAMATH_CALUDE_curve_E_equation_line_l_equation_l3224_322491

/-- The curve E is defined by the constant sum of distances to two fixed points -/
def CurveE (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

/-- The line l passes through (0, -2) and intersects curve E at points C and D -/
def LineL (l : ℝ → ℝ) (C D : ℝ × ℝ) : Prop :=
  l 0 = -2 ∧ CurveE C ∧ CurveE D ∧ C.2 = l C.1 ∧ D.2 = l D.1

/-- The dot product of OC and OD is zero -/
def OrthogonalIntersection (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem curve_E_equation (P : ℝ × ℝ) (h : CurveE P) :
  P.1^2 / 4 + P.2^2 = 1 :=
sorry

theorem line_l_equation (l : ℝ → ℝ) (C D : ℝ × ℝ)
  (hl : LineL l C D) (horth : OrthogonalIntersection C D) :
  (∀ x, l x = 2*x - 2) ∨ (∀ x, l x = -2*x - 2) :=
sorry

end NUMINAMATH_CALUDE_curve_E_equation_line_l_equation_l3224_322491


namespace NUMINAMATH_CALUDE_largest_n_digit_divisible_by_61_l3224_322434

theorem largest_n_digit_divisible_by_61 (n : ℕ+) :
  ∃ (k : ℕ), k = (10^n.val - 1) - ((10^n.val - 1) % 61) ∧ 
  k % 61 = 0 ∧
  k ≤ 10^n.val - 1 ∧
  ∀ m : ℕ, m % 61 = 0 → m ≤ 10^n.val - 1 → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_largest_n_digit_divisible_by_61_l3224_322434


namespace NUMINAMATH_CALUDE_teachers_health_survey_l3224_322400

theorem teachers_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 90)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_teachers_health_survey_l3224_322400


namespace NUMINAMATH_CALUDE_corner_sum_possibilities_l3224_322404

/-- Represents the color of a cell on the board -/
inductive CellColor
| Gold
| Silver

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (colorAt : Nat → Nat → CellColor)
  (numberAt : Nat → Nat → Nat)

/-- Defines a valid board configuration -/
def validBoard (b : Board) : Prop :=
  b.rows = 2016 ∧ b.cols = 2017 ∧
  (∀ i j, b.numberAt i j = 0 ∨ b.numberAt i j = 1) ∧
  (∀ i j, b.colorAt i j ≠ b.colorAt i (j+1)) ∧
  (∀ i j, b.colorAt i j ≠ b.colorAt (i+1) j) ∧
  (∀ i j, b.colorAt i j = CellColor.Gold →
    (b.numberAt i j + b.numberAt (i+1) j + b.numberAt i (j+1) + b.numberAt (i+1) (j+1)) % 2 = 0) ∧
  (∀ i j, b.colorAt i j = CellColor.Silver →
    (b.numberAt i j + b.numberAt (i+1) j + b.numberAt i (j+1) + b.numberAt (i+1) (j+1)) % 2 = 1)

/-- The theorem to be proved -/
theorem corner_sum_possibilities (b : Board) (h : validBoard b) :
  let cornerSum := b.numberAt 0 0 + b.numberAt 0 (b.cols-1) + b.numberAt (b.rows-1) 0 + b.numberAt (b.rows-1) (b.cols-1)
  cornerSum = 0 ∨ cornerSum = 2 ∨ cornerSum = 4 :=
sorry

end NUMINAMATH_CALUDE_corner_sum_possibilities_l3224_322404


namespace NUMINAMATH_CALUDE_floor_problem_l3224_322418

theorem floor_problem (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_problem_l3224_322418


namespace NUMINAMATH_CALUDE_tangent_line_property_l3224_322463

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def hasTangentAt (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

def tangentLineEquation (f : ℝ → ℝ) : Prop :=
  ∃ (y : ℝ), 2 + 2 * y + 1 = 0 ∧ f 2 = y

-- State the theorem
theorem tangent_line_property (f : ℝ → ℝ) 
  (h1 : hasTangentAt f) 
  (h2 : tangentLineEquation f) : 
  f 2 - 2 * (deriv f 2) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_property_l3224_322463


namespace NUMINAMATH_CALUDE_paint_cans_used_l3224_322424

theorem paint_cans_used (initial_capacity : ℕ) (lost_cans : ℕ) (remaining_capacity : ℕ) : 
  initial_capacity = 40 → 
  lost_cans = 4 → 
  remaining_capacity = 30 → 
  ∃ (cans_per_room : ℚ), 
    cans_per_room > 0 ∧
    initial_capacity = (initial_capacity - remaining_capacity) / lost_cans * lost_cans + remaining_capacity ∧
    (initial_capacity : ℚ) / cans_per_room - lost_cans = remaining_capacity / cans_per_room ∧
    remaining_capacity / cans_per_room = 12 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_l3224_322424


namespace NUMINAMATH_CALUDE_convex_polyhedron_max_intersections_non_convex_polyhedron_96_intersections_no_full_intersection_l3224_322476

/-- A polyhedron with a specified number of edges. -/
structure Polyhedron where
  edges : ℕ
  is_convex : Bool

/-- A plane that can intersect edges of a polyhedron. -/
structure IntersectingPlane where
  intersected_edges : ℕ
  passes_through_vertices : Bool

/-- Theorem about the maximum number of edges a plane can intersect in a convex polyhedron. -/
theorem convex_polyhedron_max_intersections
  (p : Polyhedron)
  (plane : IntersectingPlane)
  (h1 : p.edges = 100)
  (h2 : p.is_convex = true)
  (h3 : plane.passes_through_vertices = false) :
  plane.intersected_edges ≤ 66 :=
sorry

/-- Theorem about the existence of a non-convex polyhedron where a plane can intersect 96 edges. -/
theorem non_convex_polyhedron_96_intersections
  (p : Polyhedron)
  (h : p.edges = 100) :
  ∃ (plane : IntersectingPlane), plane.intersected_edges = 96 ∧ p.is_convex = false :=
sorry

/-- Theorem stating that it's impossible for a plane to intersect all 100 edges of a polyhedron. -/
theorem no_full_intersection
  (p : Polyhedron)
  (h : p.edges = 100) :
  ¬ ∃ (plane : IntersectingPlane), plane.intersected_edges = 100 :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_max_intersections_non_convex_polyhedron_96_intersections_no_full_intersection_l3224_322476


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3224_322493

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3224_322493


namespace NUMINAMATH_CALUDE_janet_shampoo_duration_l3224_322446

/-- Calculates the number of days Janet's shampoo will last -/
def shampoo_duration (rose_shampoo : Rat) (jasmine_shampoo : Rat) (usage_per_day : Rat) : Nat :=
  Nat.floor ((rose_shampoo + jasmine_shampoo) / usage_per_day)

/-- Theorem: Janet's shampoo will last for 7 days -/
theorem janet_shampoo_duration :
  shampoo_duration (1/3) (1/4) (1/12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_janet_shampoo_duration_l3224_322446


namespace NUMINAMATH_CALUDE_intersection_slope_of_circles_l3224_322468

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 10 = 0

-- Define the slope of the line passing through the intersection points
def intersection_slope : ℝ := 0.4

-- Theorem statement
theorem intersection_slope_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → 
  ∃ m b : ℝ, m = intersection_slope ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_of_circles_l3224_322468


namespace NUMINAMATH_CALUDE_bird_speed_indeterminate_l3224_322497

/-- A structure representing the problem scenario -/
structure ScenarioData where
  train_speed : ℝ
  bird_distance : ℝ

/-- A function that attempts to calculate the bird's speed -/
def calculate_bird_speed (data : ScenarioData) : Option ℝ :=
  none

/-- Theorem stating that the bird's speed cannot be uniquely determined -/
theorem bird_speed_indeterminate (data : ScenarioData) 
  (h1 : data.train_speed = 60)
  (h2 : data.bird_distance = 120) :
  ∀ (s : ℝ), s > 0 → ∃ (t : ℝ), t > 0 ∧ s * t = data.bird_distance :=
sorry

#check bird_speed_indeterminate

end NUMINAMATH_CALUDE_bird_speed_indeterminate_l3224_322497


namespace NUMINAMATH_CALUDE_primitive_poly_count_l3224_322433

/-- A polynomial with integer coefficients -/
structure IntPoly :=
  (a₂ a₁ a₀ : ℤ)

/-- The set of integers from 1 to 5 -/
def S : Set ℤ := {1, 2, 3, 4, 5}

/-- A polynomial is primitive if the gcd of its coefficients is 1 -/
def isPrimitive (p : IntPoly) : Prop :=
  Nat.gcd p.a₂.natAbs (Nat.gcd p.a₁.natAbs p.a₀.natAbs) = 1

/-- The product of two polynomials -/
def polyMul (p q : IntPoly) : IntPoly :=
  ⟨p.a₂ * q.a₂,
   p.a₂ * q.a₁ + p.a₁ * q.a₂,
   p.a₂ * q.a₀ + p.a₁ * q.a₁ + p.a₀ * q.a₂⟩

/-- The number of pairs of polynomials (f, g) such that f * g is primitive -/
def N : ℕ := sorry

theorem primitive_poly_count :
  N ≡ 689 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_primitive_poly_count_l3224_322433


namespace NUMINAMATH_CALUDE_quadratic_decrease_interval_l3224_322453

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_decrease_interval (b c : ℝ) :
  f b c 1 = 0 → f b c 3 = 0 → 
  ∀ x y : ℝ, x < y → y < 2 → f b c x > f b c y := by sorry

end NUMINAMATH_CALUDE_quadratic_decrease_interval_l3224_322453


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3224_322477

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_property : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2

/-- Theorem: If 2S3 - 3S2 = 15 for an arithmetic sequence, then its common difference is 5 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 - 3 * seq.S 2 = 15) : 
  seq.d = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3224_322477


namespace NUMINAMATH_CALUDE_inverse_sum_equality_l3224_322464

-- Define the function g and its inverse
variable (g : ℝ → ℝ)
variable (g_inv : ℝ → ℝ)

-- Define the given conditions
axiom g_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g
axiom g_4 : g 4 = 6
axiom g_6 : g 6 = 3
axiom g_7 : g 7 = 4

-- State the theorem
theorem inverse_sum_equality :
  g_inv (g_inv 4 + g_inv 6) = g_inv 11 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_equality_l3224_322464


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l3224_322403

theorem product_from_gcd_lcm (a b : ℕ+) : 
  Nat.gcd a b = 8 → Nat.lcm a b = 72 → a * b = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l3224_322403


namespace NUMINAMATH_CALUDE_floor_of_4_7_l3224_322484

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l3224_322484


namespace NUMINAMATH_CALUDE_max_value_x_cubed_minus_y_cubed_l3224_322405

theorem max_value_x_cubed_minus_y_cubed (x y : ℝ) (h : x^2 + y^2 = x + y) :
  ∃ (M : ℝ), M = 1 ∧ x^3 - y^3 ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = x₀ + y₀ ∧ x₀^3 - y₀^3 = M :=
sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_minus_y_cubed_l3224_322405


namespace NUMINAMATH_CALUDE_max_value_of_function_l3224_322411

theorem max_value_of_function (x : ℝ) (h : x < -1) :
  ∃ (M : ℝ), M = -3 ∧ ∀ y, y = x + 1 / (x + 1) → y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3224_322411


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3224_322440

/-- The value of the repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3224_322440


namespace NUMINAMATH_CALUDE_equality_multiplication_l3224_322407

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_l3224_322407


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l3224_322478

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

theorem even_increasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l3224_322478


namespace NUMINAMATH_CALUDE_problem_solution_l3224_322495

theorem problem_solution (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.rpow a (2 * Real.log a / Real.log 3) = 81 * Real.sqrt 3) : 
  1 / a^2 + Real.log a / Real.log 9 = 105/4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3224_322495


namespace NUMINAMATH_CALUDE_distance_between_points_l3224_322486

/-- The distance between two points (3,2,0) and (7,6,0) in 3D space is 4√2. -/
theorem distance_between_points : Real.sqrt 32 = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3224_322486


namespace NUMINAMATH_CALUDE_periodic_odd_function_property_l3224_322421

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_property (f : ℝ → ℝ) (a : ℝ) 
    (h_periodic : is_periodic f 3)
    (h_odd : is_odd f)
    (h_f1 : f 1 > 1)
    (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_property_l3224_322421


namespace NUMINAMATH_CALUDE_squared_sum_of_x_and_y_l3224_322487

theorem squared_sum_of_x_and_y (x y : ℝ) 
  (h : (2*x^2 + 2*y^2 + 3)*(2*x^2 + 2*y^2 - 3) = 27) : 
  x^2 + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_squared_sum_of_x_and_y_l3224_322487


namespace NUMINAMATH_CALUDE_binary_1010101_equals_85_l3224_322450

def binaryToDecimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010101_equals_85 :
  binaryToDecimal [true, false, true, false, true, false, true] = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_equals_85_l3224_322450


namespace NUMINAMATH_CALUDE_square_tiles_count_l3224_322401

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
  | 0 => 3  -- triangular
  | 1 => 4  -- square
  | 2 => 5  -- pentagonal
  | _ => 0  -- should never happen

/-- The proposition that given the conditions, there are 10 square tiles -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 30)
  (h_total_edges : total_edges = 120) :
  ∃ (t s p : ℕ), 
    t + s + p = total_tiles ∧ 
    3*t + 4*s + 5*p = total_edges ∧
    s = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_tiles_count_l3224_322401


namespace NUMINAMATH_CALUDE_inverse_proportion_k_negative_l3224_322474

theorem inverse_proportion_k_negative
  (k : ℝ) (y₁ y₂ : ℝ)
  (h1 : k ≠ 0)
  (h2 : y₁ = k / (-2))
  (h3 : y₂ = k / 5)
  (h4 : y₁ > y₂) :
  k < 0 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_negative_l3224_322474


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l3224_322498

theorem polynomial_identity_sum (d1 d2 d3 e1 e2 e3 : ℝ) : 
  (∀ x : ℝ, x^7 - x^6 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d1*x + e1) * (x^2 + d2*x + e2) * (x^2 + d3*x + e3)) →
  d1*e1 + d2*e2 + d3*e3 = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l3224_322498


namespace NUMINAMATH_CALUDE_impossible_sum_240_l3224_322426

theorem impossible_sum_240 : ¬ ∃ (a b c d e f g h i : ℕ), 
  (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (10 ≤ c ∧ c ≤ 99) ∧
  (10 ≤ d ∧ d ≤ 99) ∧ (10 ≤ e ∧ e ≤ 99) ∧ (10 ≤ f ∧ f ≤ 99) ∧
  (10 ≤ g ∧ g ≤ 99) ∧ (10 ≤ h ∧ h ≤ 99) ∧ (10 ≤ i ∧ i ≤ 99) ∧
  (a % 10 = 9 ∨ a / 10 = 9) ∧ (b % 10 = 9 ∨ b / 10 = 9) ∧
  (c % 10 = 9 ∨ c / 10 = 9) ∧ (d % 10 = 9 ∨ d / 10 = 9) ∧
  (e % 10 = 9 ∨ e / 10 = 9) ∧ (f % 10 = 9 ∨ f / 10 = 9) ∧
  (g % 10 = 9 ∨ g / 10 = 9) ∧ (h % 10 = 9 ∨ h / 10 = 9) ∧
  (i % 10 = 9 ∨ i / 10 = 9) ∧
  a + b + c + d + e + f + g + h + i = 240 :=
by sorry

end NUMINAMATH_CALUDE_impossible_sum_240_l3224_322426


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l3224_322467

/-- Calculates the speed of a bus including stoppages -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (h1 : speed_without_stoppages = 50) 
  (h2 : stoppage_time = 8.4 / 60) : 
  ∃ (speed_with_stoppages : ℝ), 
    speed_with_stoppages = speed_without_stoppages * (1 - stoppage_time) ∧ 
    speed_with_stoppages = 43 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l3224_322467


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l3224_322413

theorem consecutive_product_not_power (m k n : ℕ) (hn : n > 1) :
  m * (m + 1) ≠ k^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l3224_322413


namespace NUMINAMATH_CALUDE_oil_temperature_increase_rate_l3224_322439

def oil_temperature (t : ℕ) : ℝ :=
  if t = 0 then 10
  else if t = 10 then 30
  else if t = 20 then 50
  else if t = 30 then 70
  else if t = 40 then 90
  else 0  -- undefined for other values

theorem oil_temperature_increase_rate :
  ∀ t : ℕ, t < 40 →
    oil_temperature (t + 10) - oil_temperature t = 20 :=
sorry

end NUMINAMATH_CALUDE_oil_temperature_increase_rate_l3224_322439


namespace NUMINAMATH_CALUDE_max_value_of_c_max_value_of_c_achieved_l3224_322452

theorem max_value_of_c (x : ℝ) (c : ℝ) (h1 : x > 1) (h2 : c = 2 - x + 2 * Real.sqrt (x - 1)) :
  c ≤ 2 :=
by sorry

theorem max_value_of_c_achieved (x : ℝ) :
  ∃ c, x > 1 ∧ c = 2 - x + 2 * Real.sqrt (x - 1) ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_c_max_value_of_c_achieved_l3224_322452


namespace NUMINAMATH_CALUDE_georgia_muffins_l3224_322443

theorem georgia_muffins (students : ℕ) (muffins_per_batch : ℕ) (months : ℕ) :
  students = 24 →
  muffins_per_batch = 6 →
  months = 9 →
  (students / muffins_per_batch) * months = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_georgia_muffins_l3224_322443


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3224_322409

theorem arithmetic_calculation : 2^3 + 2 * 5 - 3 + 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3224_322409


namespace NUMINAMATH_CALUDE_max_area_right_quadrilateral_in_circle_l3224_322499

/-- 
Given a circle with radius r, prove that the area of a right quadrilateral inscribed in the circle 
with one side tangent to the circle and one side a chord of the circle is maximized when the 
distance from the center of the circle to the midpoint of the chord is r/2.
-/
theorem max_area_right_quadrilateral_in_circle (r : ℝ) (h : r > 0) :
  ∃ (x y : ℝ),
    x^2 + y^2 = r^2 ∧  -- Pythagorean theorem for right triangle OCE
    (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 → (x + r) * y ≥ (x' + r) * y') ∧  -- Area maximization condition
    x = r / 2  -- The distance that maximizes the area
  := by sorry

end NUMINAMATH_CALUDE_max_area_right_quadrilateral_in_circle_l3224_322499


namespace NUMINAMATH_CALUDE_weaving_increase_proof_l3224_322427

/-- Represents the daily increase in weaving output -/
def daily_increase : ℚ := 16/29

/-- The amount woven on the first day -/
def first_day_output : ℚ := 5

/-- The number of days -/
def num_days : ℕ := 30

/-- The total amount woven in 30 days -/
def total_output : ℚ := 390

theorem weaving_increase_proof :
  first_day_output * num_days + (num_days * (num_days - 1) / 2) * daily_increase = total_output :=
sorry

end NUMINAMATH_CALUDE_weaving_increase_proof_l3224_322427


namespace NUMINAMATH_CALUDE_complex_modulus_from_square_l3224_322437

theorem complex_modulus_from_square (z : ℂ) (h : z^2 = -48 + 64*I) : 
  Complex.abs z = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_from_square_l3224_322437


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l3224_322438

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l3224_322438


namespace NUMINAMATH_CALUDE_distinct_sums_largest_value_l3224_322475

theorem distinct_sums_largest_value (A B C D : ℕ) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + C ≠ B + C ∧ A + C ≠ B + D ∧ A + C ≠ D + A ∧
   B + C ≠ B + D ∧ B + C ≠ D + A ∧
   B + D ≠ D + A) →
  ({A, B, C, D, A + C, B + C, B + D, D + A} : Finset ℕ) = {1, 2, 3, 4, 5, 6, 7, 8} →
  A > B ∧ A > C ∧ A > D →
  A = 12 := by
sorry

end NUMINAMATH_CALUDE_distinct_sums_largest_value_l3224_322475


namespace NUMINAMATH_CALUDE_custom_op_identity_l3224_322462

/-- Custom operation ⊗ defined as k ⊗ l = k^2 - l^2 -/
def custom_op (k l : ℝ) : ℝ := k^2 - l^2

/-- Theorem stating that k ⊗ (k ⊗ k) = k^2 -/
theorem custom_op_identity (k : ℝ) : custom_op k (custom_op k k) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_identity_l3224_322462


namespace NUMINAMATH_CALUDE_scaling_factor_of_similar_cubes_l3224_322422

theorem scaling_factor_of_similar_cubes (v1 v2 : ℝ) (h1 : v1 = 343) (h2 : v2 = 2744) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_scaling_factor_of_similar_cubes_l3224_322422


namespace NUMINAMATH_CALUDE_circle_area_to_circumference_l3224_322415

theorem circle_area_to_circumference (A : ℝ) (h : A = 196 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ A = Real.pi * r^2 ∧ 2 * Real.pi * r = 28 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_to_circumference_l3224_322415


namespace NUMINAMATH_CALUDE_mary_shop_visits_mary_shop_visits_proof_l3224_322412

def shirt_cost : ℝ := 13.04
def jacket_cost : ℝ := 12.27
def total_cost : ℝ := 25.31

theorem mary_shop_visits : ℕ :=
  2

theorem mary_shop_visits_proof :
  (shirt_cost + jacket_cost = total_cost) →
  (∀ (shop : ℕ), shop ≤ mary_shop_visits → 
    (shop = 1 → shirt_cost > 0) ∧ 
    (shop = 2 → jacket_cost > 0)) →
  mary_shop_visits = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_shop_visits_mary_shop_visits_proof_l3224_322412


namespace NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_ln_a_negative_l3224_322469

theorem a_less_than_one_necessary_not_sufficient_for_ln_a_negative :
  (∀ a : ℝ, (Real.log a < 0) → (a < 1)) ∧
  (∃ a : ℝ, a < 1 ∧ ¬(Real.log a < 0)) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_ln_a_negative_l3224_322469


namespace NUMINAMATH_CALUDE_painting_wings_count_l3224_322471

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : Nat
  artifacts_per_wing : Nat
  large_painting_wings : Nat
  small_painting_wings : Nat
  paintings_per_small_wing : Nat

/-- The number of wings dedicated to paintings in the museum -/
def painting_wings (m : Museum) : Nat :=
  m.large_painting_wings + m.small_painting_wings

/-- The number of wings dedicated to artifacts in the museum -/
def artifact_wings (m : Museum) : Nat :=
  m.total_wings - painting_wings m

/-- The total number of paintings in the museum -/
def total_paintings (m : Museum) : Nat :=
  m.large_painting_wings + m.small_painting_wings * m.paintings_per_small_wing

/-- The total number of artifacts in the museum -/
def total_artifacts (m : Museum) : Nat :=
  m.artifacts_per_wing * artifact_wings m

theorem painting_wings_count (m : Museum)
  (h1 : m.total_wings = 8)
  (h2 : total_artifacts m = 4 * total_paintings m)
  (h3 : m.large_painting_wings = 1)
  (h4 : m.small_painting_wings = 2)
  (h5 : m.paintings_per_small_wing = 12)
  (h6 : m.artifacts_per_wing = 20) :
  painting_wings m = 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_wings_count_l3224_322471


namespace NUMINAMATH_CALUDE_function_ratio_bounds_l3224_322445

open Real

theorem function_ratio_bounds (f : ℝ → ℝ) (hf : ∀ x > 0, f x > 0)
  (hf' : ∀ x > 0, 2 * f x < x * (deriv f x) ∧ x * (deriv f x) < 3 * f x) :
  8/27 < f 2 / f 3 ∧ f 2 / f 3 < 4/9 := by
  sorry

end NUMINAMATH_CALUDE_function_ratio_bounds_l3224_322445


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3224_322492

/-- Given two lines in the form ax + by + c = 0, this function returns true if they are perpendicular --/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- Given a line in the form ax + by + c = 0 and a point (x, y), this function returns true if the point lies on the line --/
def point_on_line (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem perpendicular_line_through_point :
  are_perpendicular 1 (-2) 2 1 ∧
  point_on_line 1 (-2) 3 1 2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3224_322492


namespace NUMINAMATH_CALUDE_cleaning_event_children_count_l3224_322485

theorem cleaning_event_children_count (total_members : ℕ) 
  (adult_men_percentage : ℚ) (h1 : total_members = 2000) 
  (h2 : adult_men_percentage = 30 / 100) : 
  total_members - (adult_men_percentage * total_members).num - 
  (2 * (adult_men_percentage * total_members).num) = 200 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_event_children_count_l3224_322485


namespace NUMINAMATH_CALUDE_multiply_725143_by_999999_l3224_322425

theorem multiply_725143_by_999999 : 725143 * 999999 = 725142274857 := by
  sorry

end NUMINAMATH_CALUDE_multiply_725143_by_999999_l3224_322425


namespace NUMINAMATH_CALUDE_geometric_sequence_range_l3224_322483

theorem geometric_sequence_range (a₁ a₂ a₃ a₄ : ℝ) :
  (0 < a₁ ∧ a₁ < 1) →
  (1 < a₂ ∧ a₂ < 2) →
  (2 < a₃ ∧ a₃ < 3) →
  (∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3) →
  (2 * Real.sqrt 2 < a₄ ∧ a₄ < 9) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_range_l3224_322483


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l3224_322419

theorem two_digit_integer_problem :
  ∀ m n : ℕ,
    m ≥ 10 ∧ m ≤ 99 →  -- m is a 2-digit positive integer
    n ≥ 10 ∧ n ≤ 99 →  -- n is a 2-digit positive integer
    n % 25 = 0 →  -- n is a multiple of 25
    (m + n) / 2 = m + n / 100 →  -- average equals decimal representation
    max m n = 50 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l3224_322419


namespace NUMINAMATH_CALUDE_root_product_l3224_322406

theorem root_product (r b c : ℝ) : 
  r^2 = r + 1 → r^6 = b*r + c → b*c = 40 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l3224_322406


namespace NUMINAMATH_CALUDE_exponent_division_l3224_322432

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3224_322432


namespace NUMINAMATH_CALUDE_mark_takes_tablets_for_12_hours_l3224_322466

/-- Represents the number of hours Mark takes Tylenol tablets -/
def hours_taking_tablets (tablets_per_dose : ℕ) (mg_per_tablet : ℕ) (hours_between_doses : ℕ) (total_grams : ℕ) : ℕ :=
  (total_grams * 1000) / (tablets_per_dose * mg_per_tablet) * hours_between_doses

/-- Theorem stating that Mark takes the tablets for 12 hours -/
theorem mark_takes_tablets_for_12_hours :
  hours_taking_tablets 2 500 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mark_takes_tablets_for_12_hours_l3224_322466


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3224_322410

theorem base_10_to_base_7 (n : ℕ) (h : n = 3589) :
  ∃ (a b c d e : ℕ),
    n = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 ∧
    a = 1 ∧ b = 3 ∧ c = 3 ∧ d = 1 ∧ e = 5 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3224_322410


namespace NUMINAMATH_CALUDE_drama_club_revenue_l3224_322454

theorem drama_club_revenue : 
  let total_tickets : ℕ := 1500
  let adult_price : ℕ := 12
  let student_price : ℕ := 6
  let student_tickets : ℕ := 300
  let adult_tickets : ℕ := total_tickets - student_tickets
  let total_revenue : ℕ := adult_tickets * adult_price + student_tickets * student_price
  total_revenue = 16200 := by
sorry

end NUMINAMATH_CALUDE_drama_club_revenue_l3224_322454


namespace NUMINAMATH_CALUDE_largest_common_divisor_660_483_l3224_322473

theorem largest_common_divisor_660_483 : Nat.gcd 660 483 = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_660_483_l3224_322473


namespace NUMINAMATH_CALUDE_fraction_simplification_l3224_322451

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  1 / (x - 1) - 2 / (x^2 - 1) = 1 / (x + 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3224_322451


namespace NUMINAMATH_CALUDE_complex_number_properties_l3224_322494

def z : ℂ := (1 - Complex.I)^2 + 1 + 3 * Complex.I

theorem complex_number_properties :
  (z = 3 + 3 * Complex.I) ∧
  (Complex.abs z = 3 * Real.sqrt 2) ∧
  (∃ (a b : ℝ), z^2 + a * z + b = 1 - Complex.I ∧ a = -6 ∧ b = 10) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3224_322494


namespace NUMINAMATH_CALUDE_salary_increase_l3224_322490

/-- Represents the regression line for a worker's monthly salary based on labor productivity -/
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

/-- Theorem stating that an increase of 1 unit in labor productivity results in a 90 yuan increase in salary -/
theorem salary_increase (x : ℝ) : 
  regression_line (x + 1) - regression_line x = 90 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3224_322490


namespace NUMINAMATH_CALUDE_rubber_bands_distribution_l3224_322460

/-- The number of rubber bands Aira had -/
def aira_bands : ℕ := sorry

/-- The number of rubber bands Samantha had -/
def samantha_bands : ℕ := sorry

/-- The number of rubber bands Joe had -/
def joe_bands : ℕ := sorry

/-- The total number of rubber bands -/
def total_bands : ℕ := sorry

theorem rubber_bands_distribution :
  -- Condition 1 and 2: Equal division resulting in 6 bands each
  total_bands = 3 * 6 ∧
  -- Condition 3: Samantha had 5 more bands than Aira
  samantha_bands = aira_bands + 5 ∧
  -- Condition 4: Aira had 1 fewer band than Joe
  aira_bands + 1 = joe_bands ∧
  -- Total bands is the sum of all individual bands
  total_bands = aira_bands + samantha_bands + joe_bands →
  -- Conclusion: Aira had 4 rubber bands
  aira_bands = 4 := by
sorry

end NUMINAMATH_CALUDE_rubber_bands_distribution_l3224_322460


namespace NUMINAMATH_CALUDE_sufficient_condition_inequality_l3224_322449

theorem sufficient_condition_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_inequality_l3224_322449


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l3224_322465

theorem sum_of_fractions_simplest_form : 
  (6 : ℚ) / 7 + (7 : ℚ) / 9 = (103 : ℚ) / 63 ∧ 
  ∀ n d : ℤ, (n : ℚ) / d = (103 : ℚ) / 63 → (n.gcd d = 1 → n = 103 ∧ d = 63) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l3224_322465


namespace NUMINAMATH_CALUDE_unique_valid_denomination_l3224_322496

def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 120 → ∃ (a b c : ℕ), k = 7 * a + n * b + (n + 2) * c

def is_greatest_unformable (n : ℕ) : Prop :=
  ¬∃ (a b c : ℕ), 120 = 7 * a + n * b + (n + 2) * c

theorem unique_valid_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n ∧ is_greatest_unformable n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_denomination_l3224_322496


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l3224_322402

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- Represents the two parts of a triangle after division along a median -/
structure TriangleParts where
  part1 : Triangle
  part2 : Triangle

theorem triangle_division_theorem (t1 t2 t3 : Triangle) 
  (h_identical : t1 = t2 ∧ t2 = t3) :
  ∃ (p1 p2 p3 : TriangleParts) (result : Triangle),
    (p1.part1.a = t1.a ∧ p1.part1.b = t1.b) ∧
    (p2.part1.a = t2.a ∧ p2.part1.b = t2.b) ∧
    (p3.part1.a = t3.a ∧ p3.part1.b = t3.b) ∧
    (p1.part1.α + p2.part1.α + p3.part1.α = 2 * π) ∧
    (result.a = t1.a ∧ result.b = t1.b ∧ result.c = t1.c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l3224_322402


namespace NUMINAMATH_CALUDE_tens_digit_of_2015_pow_2016_minus_2017_l3224_322480

theorem tens_digit_of_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 / 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_2015_pow_2016_minus_2017_l3224_322480


namespace NUMINAMATH_CALUDE_basketball_shots_l3224_322479

theorem basketball_shots (total_points : ℕ) (three_point_shots : ℕ) : 
  total_points = 26 → 
  three_point_shots = 4 → 
  ∃ (two_point_shots : ℕ), 
    total_points = 3 * three_point_shots + 2 * two_point_shots ∧
    three_point_shots + two_point_shots = 11 :=
by sorry

end NUMINAMATH_CALUDE_basketball_shots_l3224_322479


namespace NUMINAMATH_CALUDE_product_no_linear_quadratic_terms_l3224_322481

theorem product_no_linear_quadratic_terms 
  (p q : ℚ) 
  (h : ∀ x : ℚ, (x + 3*p) * (x^2 - x + 1/3*q) = x^3 + p*q) : 
  p = 1/3 ∧ q = 3 ∧ p^2020 * q^2021 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_no_linear_quadratic_terms_l3224_322481


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l3224_322488

/-- Represents the number of items in each stratum -/
structure Strata :=
  (grains : ℕ)
  (vegetable_oils : ℕ)
  (animal_foods : ℕ)
  (fruits_and_vegetables : ℕ)

/-- Calculates the total number of items across all strata -/
def total_items (s : Strata) : ℕ :=
  s.grains + s.vegetable_oils + s.animal_foods + s.fruits_and_vegetables

/-- Calculates the number of items to sample from a stratum -/
def stratum_sample (total_sample : ℕ) (stratum_size : ℕ) (s : Strata) : ℕ :=
  (total_sample * stratum_size) / (total_items s)

/-- The main theorem to prove -/
theorem stratified_sampling_sum (s : Strata) (total_sample : ℕ) :
  s.grains = 40 →
  s.vegetable_oils = 10 →
  s.animal_foods = 30 →
  s.fruits_and_vegetables = 20 →
  total_sample = 20 →
  (stratum_sample total_sample s.vegetable_oils s +
   stratum_sample total_sample s.fruits_and_vegetables s) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sum_l3224_322488


namespace NUMINAMATH_CALUDE_fraction_equality_l3224_322436

theorem fraction_equality (a b : ℚ) (h : b / a = 5 / 13) : 
  (a - b) / (a + b) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3224_322436


namespace NUMINAMATH_CALUDE_brians_net_commission_l3224_322417

def house_price_1 : ℝ := 157000
def house_price_2 : ℝ := 499000
def house_price_3 : ℝ := 125000
def house_price_4 : ℝ := 275000
def house_price_5 : ℝ := 350000

def commission_rate_1 : ℝ := 0.025
def commission_rate_2 : ℝ := 0.018
def commission_rate_3 : ℝ := 0.02
def commission_rate_4 : ℝ := 0.022
def commission_rate_5 : ℝ := 0.023

def administrative_fee : ℝ := 500

def total_commission : ℝ := 
  house_price_1 * commission_rate_1 +
  house_price_2 * commission_rate_2 +
  house_price_3 * commission_rate_3 +
  house_price_4 * commission_rate_4 +
  house_price_5 * commission_rate_5

def net_commission : ℝ := total_commission - administrative_fee

theorem brians_net_commission : 
  net_commission = 29007 := by sorry

end NUMINAMATH_CALUDE_brians_net_commission_l3224_322417


namespace NUMINAMATH_CALUDE_freds_change_is_correct_l3224_322420

/-- The amount of change Fred received after buying movie tickets and borrowing a movie -/
def freds_change (ticket_price : ℚ) (num_tickets : ℕ) (borrowed_movie_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price)

/-- Theorem: Fred's change is $1.37 -/
theorem freds_change_is_correct : 
  freds_change (92/100 + 5) 2 (79/100 + 6) 20 = 37/100 + 1 :=
by sorry

end NUMINAMATH_CALUDE_freds_change_is_correct_l3224_322420


namespace NUMINAMATH_CALUDE_triangle_side_length_l3224_322444

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (area : ℝ) :
  b = 3 →
  c = 4 →
  area = 3 * Real.sqrt 3 →
  area = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (a = Real.sqrt 13 ∨ a = Real.sqrt 37) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3224_322444


namespace NUMINAMATH_CALUDE_remy_used_19_gallons_l3224_322408

/-- Represents the water usage of three people taking showers -/
structure ShowerUsage where
  roman : ℕ
  remy : ℕ
  riley : ℕ

/-- Defines the conditions of the shower usage problem -/
def validShowerUsage (u : ShowerUsage) : Prop :=
  u.remy = 3 * u.roman + 1 ∧
  u.riley = u.roman + u.remy - 2 ∧
  u.roman + u.remy + u.riley = 48

/-- Theorem stating that if the shower usage is valid, Remy used 19 gallons -/
theorem remy_used_19_gallons (u : ShowerUsage) : 
  validShowerUsage u → u.remy = 19 := by
  sorry

#check remy_used_19_gallons

end NUMINAMATH_CALUDE_remy_used_19_gallons_l3224_322408


namespace NUMINAMATH_CALUDE_f_range_on_interval_l3224_322435

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * cos x - 2 * sin x ^ 2

theorem f_range_on_interval :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2),
  ∃ y ∈ Set.Icc (-5/2) (-2), f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-5/2) (-2) :=
by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l3224_322435


namespace NUMINAMATH_CALUDE_ice_cream_sandwiches_l3224_322456

theorem ice_cream_sandwiches (nieces : ℕ) (sandwiches_per_niece : ℕ) 
  (h1 : nieces = 11) (h2 : sandwiches_per_niece = 13) : 
  nieces * sandwiches_per_niece = 143 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sandwiches_l3224_322456


namespace NUMINAMATH_CALUDE_Q_one_smallest_l3224_322448

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 - 3*x^2 + 6*x - 5

theorem Q_one_smallest : 
  let q1 := Q 1
  let prod_zeros := -5
  let sum_coeff := 1 + (-2) + (-3) + 6 + (-5)
  q1 ≤ prod_zeros ∧ q1 ≤ sum_coeff :=
by sorry

end NUMINAMATH_CALUDE_Q_one_smallest_l3224_322448


namespace NUMINAMATH_CALUDE_tim_income_percentage_l3224_322442

/-- Proves that Tim's income is 60% less than Juan's income given the conditions --/
theorem tim_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = 1.6 * tim)  -- Mart's income is 60% more than Tim's
  (h2 : mart = 0.64 * juan)  -- Mart's income is 64% of Juan's
  : tim = 0.4 * juan :=  -- Tim's income is 40% of Juan's (equivalent to 60% less)
by
  sorry

#check tim_income_percentage

end NUMINAMATH_CALUDE_tim_income_percentage_l3224_322442


namespace NUMINAMATH_CALUDE_trailing_zeros_for_specific_fraction_l3224_322459

/-- The number of trailing zeros in the decimal representation of a rational number -/
def trailingZeros (n d : ℕ) : ℕ :=
  sorry

/-- The main theorem: number of trailing zeros for 1 / (2^3 * 5^7) -/
theorem trailing_zeros_for_specific_fraction :
  trailingZeros 1 (2^3 * 5^7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_for_specific_fraction_l3224_322459
