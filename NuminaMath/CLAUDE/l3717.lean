import Mathlib

namespace NUMINAMATH_CALUDE_modulus_of_Z_l3717_371797

/-- The modulus of the complex number Z = 1 / (i - 1) is equal to √2/2 -/
theorem modulus_of_Z : Complex.abs (1 / (Complex.I - 1)) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l3717_371797


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base_5_l3717_371785

theorem base_conversion_1729_to_base_5 :
  ∃ (a b c d e : ℕ),
    1729 = a * 5^4 + b * 5^3 + c * 5^2 + d * 5^1 + e * 5^0 ∧
    a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 0 ∧ e = 4 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_1729_to_base_5_l3717_371785


namespace NUMINAMATH_CALUDE_quintic_polynomial_minimum_value_l3717_371703

/-- A quintic polynomial with real coefficients -/
def QuinticPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, ∀ x, P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

/-- All complex roots of P have magnitude 1 -/
def AllRootsOnUnitCircle (P : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, P z = 0 → Complex.abs z = 1

theorem quintic_polynomial_minimum_value (P : ℝ → ℝ) 
  (h_quintic : QuinticPolynomial P)
  (h_P0 : P 0 = 2)
  (h_P1 : P 1 = 3)
  (h_roots : AllRootsOnUnitCircle (fun z => P z.re)) :
  (∀ Q : ℝ → ℝ, QuinticPolynomial Q → Q 0 = 2 → Q 1 = 3 → 
    AllRootsOnUnitCircle (fun z => Q z.re) → P 2 ≤ Q 2) ∧ P 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_quintic_polynomial_minimum_value_l3717_371703


namespace NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l3717_371723

theorem sin_three_pi_half_plus_alpha (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  Real.sin (3 * Real.pi / 2 + α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_half_plus_alpha_l3717_371723


namespace NUMINAMATH_CALUDE_total_sharks_l3717_371754

theorem total_sharks (newport_sharks : ℕ) (dana_point_sharks : ℕ) : 
  newport_sharks = 22 → 
  dana_point_sharks = 4 * newport_sharks → 
  newport_sharks + dana_point_sharks = 110 := by
sorry

end NUMINAMATH_CALUDE_total_sharks_l3717_371754


namespace NUMINAMATH_CALUDE_value_of_b_l3717_371770

theorem value_of_b (b : ℚ) (h : b + b/4 = 3) : b = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3717_371770


namespace NUMINAMATH_CALUDE_student_count_proof_l3717_371779

def total_students (group1 group2 group3 group4 : ℕ) : ℕ :=
  group1 + group2 + group3 + group4

theorem student_count_proof :
  let group1 : ℕ := 5
  let group2 : ℕ := 8
  let group3 : ℕ := 7
  let group4 : ℕ := 4
  total_students group1 group2 group3 group4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_count_proof_l3717_371779


namespace NUMINAMATH_CALUDE_february_monthly_fee_calculation_l3717_371712

/-- Represents the monthly membership fee and per-class fee structure -/
structure FeeStructure where
  monthly_fee : ℝ
  per_class_fee : ℝ

/-- Calculates the total bill given a fee structure and number of classes -/
def total_bill (fs : FeeStructure) (classes : ℕ) : ℝ :=
  fs.monthly_fee + fs.per_class_fee * classes

/-- Represents the fee structure with a 10% increase in monthly fee -/
def increased_fee_structure (fs : FeeStructure) : FeeStructure :=
  { monthly_fee := 1.1 * fs.monthly_fee
    per_class_fee := fs.per_class_fee }

theorem february_monthly_fee_calculation 
  (feb_fs : FeeStructure)
  (h1 : total_bill feb_fs 4 = 30.72)
  (h2 : total_bill (increased_fee_structure feb_fs) 8 = 54.72) :
  feb_fs.monthly_fee = 7.47 := by
  sorry

#eval (7.47 : Float).toString

end NUMINAMATH_CALUDE_february_monthly_fee_calculation_l3717_371712


namespace NUMINAMATH_CALUDE_five_congruent_subtriangles_impossible_l3717_371736

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a subdivision of a triangle into five smaller triangles
structure SubdividedTriangle where
  main : Triangle
  sub1 : Triangle
  sub2 : Triangle
  sub3 : Triangle
  sub4 : Triangle
  sub5 : Triangle

-- Theorem statement
theorem five_congruent_subtriangles_impossible (t : SubdividedTriangle) :
  ¬(t.sub1 = t.sub2 ∧ t.sub2 = t.sub3 ∧ t.sub3 = t.sub4 ∧ t.sub4 = t.sub5) :=
by sorry

end NUMINAMATH_CALUDE_five_congruent_subtriangles_impossible_l3717_371736


namespace NUMINAMATH_CALUDE_steve_socks_l3717_371759

theorem steve_socks (total_socks : ℕ) (matching_pairs : ℕ) (mismatching_socks : ℕ) : 
  total_socks = 48 → matching_pairs = 11 → mismatching_socks = total_socks - 2 * matching_pairs → mismatching_socks = 26 := by
  sorry

end NUMINAMATH_CALUDE_steve_socks_l3717_371759


namespace NUMINAMATH_CALUDE_base9_246_to_base10_l3717_371761

/-- Converts a three-digit number from base 9 to base 10 -/
def base9ToBase10 (d2 d1 d0 : Nat) : Nat :=
  d2 * 9^2 + d1 * 9^1 + d0 * 9^0

/-- The base 10 representation of 246 in base 9 is 204 -/
theorem base9_246_to_base10 : base9ToBase10 2 4 6 = 204 := by
  sorry

end NUMINAMATH_CALUDE_base9_246_to_base10_l3717_371761


namespace NUMINAMATH_CALUDE_day_crew_fraction_of_boxes_l3717_371730

/-- Represents the fraction of boxes loaded by the day crew given the relative productivity
    and size of the night crew compared to the day crew. -/
theorem day_crew_fraction_of_boxes
  (night_worker_productivity : ℚ)  -- Productivity of night worker relative to day worker
  (night_crew_size : ℚ)            -- Size of night crew relative to day crew
  (h1 : night_worker_productivity = 1 / 4)
  (h2 : night_crew_size = 4 / 5) :
  (1 : ℚ) / (1 + night_worker_productivity * night_crew_size) = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_day_crew_fraction_of_boxes_l3717_371730


namespace NUMINAMATH_CALUDE_inscribed_circles_theorem_l3717_371717

theorem inscribed_circles_theorem (N : ℕ) (r : ℝ) (h_pos : r > 0) : 
  let R := N * r
  let area_small_circles := N * Real.pi * r^2
  let area_large_circle := Real.pi * R^2
  let area_remaining := area_large_circle - area_small_circles
  (area_small_circles / area_remaining = 1 / 3) → N = 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circles_theorem_l3717_371717


namespace NUMINAMATH_CALUDE_max_black_pieces_l3717_371738

/-- Represents a piece color -/
inductive Color
| Black
| White

/-- Represents the state of the circle -/
def CircleState := List Color

/-- Applies the rule to place new pieces between existing ones -/
def applyRule (state : CircleState) : CircleState :=
  sorry

/-- Removes the original pieces from the circle -/
def removeOriginal (state : CircleState) : CircleState :=
  sorry

/-- Counts the number of black pieces in the circle -/
def countBlack (state : CircleState) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of black pieces is 4 -/
theorem max_black_pieces (initial : CircleState) : 
  initial.length = 5 → 
  ∀ (n : Nat), countBlack (removeOriginal (applyRule initial)) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_black_pieces_l3717_371738


namespace NUMINAMATH_CALUDE_cuboid_max_volume_l3717_371747

/-- The maximum volume of a cuboid with a total edge length of 60 units is 125 cubic units. -/
theorem cuboid_max_volume :
  ∀ x y z : ℝ,
  x > 0 → y > 0 → z > 0 →
  4 * (x + y + z) = 60 →
  x * y * z ≤ 125 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_max_volume_l3717_371747


namespace NUMINAMATH_CALUDE_parallel_vectors_not_always_same_direction_l3717_371734

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (v w : V) : Prop := ∃ k : ℝ, v = k • w

theorem parallel_vectors_not_always_same_direction :
  ∃ (v w : V), parallel v w ∧ ¬(∃ k : ℝ, k > 0 ∧ v = k • w) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_not_always_same_direction_l3717_371734


namespace NUMINAMATH_CALUDE_a_annual_income_l3717_371745

/-- Proves that A's annual income is 403200 given the specified conditions -/
theorem a_annual_income (c_income : ℕ) (h1 : c_income = 12000) : ∃ (a_income b_income : ℕ),
  (a_income : ℚ) / b_income = 5 / 2 ∧
  b_income = c_income + c_income * 12 / 100 ∧
  a_income * 12 = 403200 :=
by sorry

end NUMINAMATH_CALUDE_a_annual_income_l3717_371745


namespace NUMINAMATH_CALUDE_coin_value_theorem_l3717_371750

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the total value of coins in cents given the number of quarters and dimes -/
def total_value (quarters dimes : ℕ) : ℕ := quarter_value * quarters + dime_value * dimes

/-- Calculates the total value of coins in cents if quarters and dimes were swapped -/
def swapped_value (quarters dimes : ℕ) : ℕ := dime_value * quarters + quarter_value * dimes

theorem coin_value_theorem (quarters dimes : ℕ) :
  quarters + dimes = 30 →
  swapped_value quarters dimes = total_value quarters dimes + 150 →
  total_value quarters dimes = 450 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_theorem_l3717_371750


namespace NUMINAMATH_CALUDE_vector_on_line_k_value_l3717_371743

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def line_through (a b : V) : ℝ → V :=
  λ t => a + t • (b - a)

theorem vector_on_line_k_value
  (a b : V) (ha_ne_b : a ≠ b) (k : ℝ) :
  (∃ t : ℝ, line_through a b t = k • a + (5/7 : ℝ) • b) →
  k = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_k_value_l3717_371743


namespace NUMINAMATH_CALUDE_connor_date_cost_l3717_371769

/-- Calculates the total cost of Connor's movie date --/
def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price

/-- Theorem stating that the total cost of Connor's movie date is $36.00 --/
theorem connor_date_cost :
  movie_date_cost 10 11 2.5 = 36 :=
by sorry

end NUMINAMATH_CALUDE_connor_date_cost_l3717_371769


namespace NUMINAMATH_CALUDE_factorial_quotient_l3717_371778

theorem factorial_quotient : Nat.factorial 50 / Nat.factorial 47 = 117600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_quotient_l3717_371778


namespace NUMINAMATH_CALUDE_cooler_cans_count_l3717_371729

/-- Given a cooler with cherry soda and orange pop, where there are twice as many
    cans of orange pop as cherry soda, and there are 8 cherry sodas,
    prove that the total number of cans in the cooler is 24. -/
theorem cooler_cans_count (cherry_soda orange_pop : ℕ) : 
  cherry_soda = 8 →
  orange_pop = 2 * cherry_soda →
  cherry_soda + orange_pop = 24 := by
  sorry

end NUMINAMATH_CALUDE_cooler_cans_count_l3717_371729


namespace NUMINAMATH_CALUDE_opposite_of_gold_is_olive_l3717_371783

-- Define the colors
inductive Color
  | Aqua | Maroon | Olive | Purple | Silver | Gold | Black

-- Define the cube faces
structure CubeFace where
  color : Color

-- Define the cube
structure Cube where
  faces : List CubeFace
  gold_face : CubeFace
  opposite_face : CubeFace

-- Define the cross pattern
structure CrossPattern where
  squares : List CubeFace

-- Function to fold the cross pattern into a cube
def fold_cross_to_cube (cross : CrossPattern) : Cube :=
  sorry

-- Theorem: The face opposite to Gold is Olive
theorem opposite_of_gold_is_olive (cross : CrossPattern) 
  (cube : Cube := fold_cross_to_cube cross) : 
  cube.gold_face.color = Color.Gold → cube.opposite_face.color = Color.Olive :=
sorry

end NUMINAMATH_CALUDE_opposite_of_gold_is_olive_l3717_371783


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l3717_371781

/-- Given the prices of adult and child tickets, the total number of tickets sold,
    and the total revenue, prove the number of adult tickets sold. -/
theorem adult_tickets_sold
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100)
  : ∃ (adult_tickets : ℕ),
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧
    adult_tickets = 500 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l3717_371781


namespace NUMINAMATH_CALUDE_prob_k_white_balls_correct_l3717_371704

/-- The probability of drawing exactly k white balls from an urn containing n white and n black balls,
    when drawing n balls in total. -/
def prob_k_white_balls (n k : ℕ) : ℚ :=
  (Nat.choose n k)^2 / Nat.choose (2*n) n

/-- Theorem stating that the probability of drawing exactly k white balls from an urn
    containing n white balls and n black balls, when drawing n balls in total,
    is equal to (n choose k)^2 / (2n choose n). -/
theorem prob_k_white_balls_correct (n k : ℕ) (h : k ≤ n) :
  prob_k_white_balls n k = (Nat.choose n k)^2 / Nat.choose (2*n) n :=
by sorry

end NUMINAMATH_CALUDE_prob_k_white_balls_correct_l3717_371704


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_exists_y_for_min_max_abs_quadratic_minus_linear_l3717_371700

theorem min_max_abs_quadratic_minus_linear (y : ℝ) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| = 4) ∧
  (∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4) :=
by sorry

theorem exists_y_for_min_max_abs_quadratic_minus_linear :
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_exists_y_for_min_max_abs_quadratic_minus_linear_l3717_371700


namespace NUMINAMATH_CALUDE_date_sum_equality_l3717_371720

/-- Represents a calendar date sequence -/
structure DateSequence where
  x : ℕ  -- Date behind C
  dateA : ℕ := x + 2  -- Date behind A
  dateB : ℕ := x + 11  -- Date behind B
  dateP : ℕ := x + 13  -- Date behind P

/-- Theorem: The sum of dates behind C and P equals the sum of dates behind A and B -/
theorem date_sum_equality (d : DateSequence) : 
  d.x + d.dateP = d.dateA + d.dateB := by
  sorry

end NUMINAMATH_CALUDE_date_sum_equality_l3717_371720


namespace NUMINAMATH_CALUDE_new_rectangle_area_l3717_371708

/-- Given a rectangle with sides a and b, prove the area of a new rectangle constructed from it. -/
theorem new_rectangle_area (a b : ℝ) (h : 0 < a ∧ a < b) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l3717_371708


namespace NUMINAMATH_CALUDE_function_identity_l3717_371721

def is_positive_integer (n : ℤ) : Prop := 0 < n

structure PositiveInteger where
  val : ℤ
  pos : is_positive_integer val

def PositiveIntegerFunction := PositiveInteger → PositiveInteger

theorem function_identity (f : PositiveIntegerFunction) : 
  (∀ (a b : PositiveInteger), ∃ (k : ℤ), a.val ^ 2 + (f a).val * (f b).val = ((f a).val + b.val) * k) →
  (∀ (n : PositiveInteger), (f n).val = n.val) := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3717_371721


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3717_371793

/-- Given that x = 1 is a root of the quadratic equation x^2 + bx - 2 = 0,
    prove that the other root is -2 -/
theorem other_root_of_quadratic (b : ℝ) : 
  (1^2 + b*1 - 2 = 0) → ∃ x : ℝ, x ≠ 1 ∧ x^2 + b*x - 2 = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3717_371793


namespace NUMINAMATH_CALUDE_liz_jump_shots_liz_jump_shots_correct_l3717_371751

theorem liz_jump_shots (initial_deficit : ℕ) (free_throws : ℕ) (three_pointers : ℕ) 
  (opponent_points : ℕ) (final_deficit : ℕ) : ℕ :=
  let free_throw_points := free_throws * 1
  let three_pointer_points := three_pointers * 3
  let total_deficit := initial_deficit + opponent_points
  let points_needed := total_deficit - final_deficit
  let jump_shot_points := points_needed - free_throw_points - three_pointer_points
  jump_shot_points / 2

theorem liz_jump_shots_correct :
  liz_jump_shots 20 5 3 10 8 = 4 := by sorry

end NUMINAMATH_CALUDE_liz_jump_shots_liz_jump_shots_correct_l3717_371751


namespace NUMINAMATH_CALUDE_triangle_side_length_l3717_371773

theorem triangle_side_length 
  (A B C : Real) 
  (AB BC AC : Real) :
  A = π / 3 →
  Real.tan B = 1 / 2 →
  AB = 2 * Real.sqrt 3 + 1 →
  BC = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3717_371773


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3717_371755

-- Define the function f(x)
def f (x : ℝ) := x^3 - 12*x

-- Define the interval
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 16 ∧ min = -16 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3717_371755


namespace NUMINAMATH_CALUDE_cyclists_speed_cyclists_speed_is_10_l3717_371724

/-- Two cyclists traveling in opposite directions for 2.5 hours end up 50 km apart. -/
theorem cyclists_speed : ℝ → Prop :=
  fun speed : ℝ =>
    let time : ℝ := 2.5
    let distance : ℝ := 50
    2 * speed * time = distance

/-- The speed of each cyclist is 10 km/h. -/
theorem cyclists_speed_is_10 : cyclists_speed 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speed_cyclists_speed_is_10_l3717_371724


namespace NUMINAMATH_CALUDE_G_difference_l3717_371753

/-- G is defined as the infinite repeating decimal 0.737373... -/
def G : ℚ := 73 / 99

/-- The difference between the denominator and numerator of G when expressed as a fraction in lowest terms -/
def difference : ℕ := 99 - 73

theorem G_difference : difference = 26 := by sorry

end NUMINAMATH_CALUDE_G_difference_l3717_371753


namespace NUMINAMATH_CALUDE_max_sum_solution_l3717_371798

theorem max_sum_solution : ∃ (a b : ℕ), 
  (2 * a * b + 3 * b = b^2 + 6 * a + 6) ∧ 
  (∀ (x y : ℕ), (2 * x * y + 3 * y = y^2 + 6 * x + 6) → (x + y ≤ a + b)) ∧
  a = 5 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_solution_l3717_371798


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l3717_371758

/-- A regular hexagon with side length 2 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- A circle in the context of our problem -/
structure Circle :=
  (center : Fin 6)  -- Vertex of the hexagon (0 to 5)
  (radius : ℝ)

/-- Three circles touching each other externally -/
def touching_circles (h : RegularHexagon) (c₁ c₂ c₃ : Circle) : Prop :=
  (c₁.center = 0 ∧ c₂.center = 1 ∧ c₃.center = 2) ∧  -- Centers at A, B, C
  (c₁.radius + c₂.radius = h.side_length) ∧
  (c₁.radius + c₃.radius = h.side_length * Real.sqrt 3) ∧
  (c₂.radius + c₃.radius = h.side_length)

theorem smallest_circle_radius 
  (h : RegularHexagon) 
  (c₁ c₂ c₃ : Circle) 
  (touch : touching_circles h c₁ c₂ c₃) :
  min c₁.radius (min c₂.radius c₃.radius) = 2 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l3717_371758


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l3717_371795

/-- Given m > 0, p: (x+2)(x-6) ≤ 0, and q: 2-m ≤ x ≤ 2+m -/
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

/-- If p is a necessary condition for q, then 0 < m ≤ 4 -/
theorem range_of_m (m : ℝ) (h : m > 0) :
  (∀ x, q m x → p x) → 0 < m ∧ m ≤ 4 := by sorry

/-- Given m = 2, if ¬p ∨ ¬q is false, then 0 ≤ x ≤ 4 -/
theorem range_of_x (x : ℝ) :
  ¬(¬(p x) ∨ ¬(q 2 x)) → 0 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l3717_371795


namespace NUMINAMATH_CALUDE_cosine_sum_problem_l3717_371768

theorem cosine_sum_problem (α : Real) 
  (h : Real.sin (π / 2 + α) = 1 / 3) : 
  Real.cos (2 * α) + Real.cos α = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_problem_l3717_371768


namespace NUMINAMATH_CALUDE_lattice_fifth_number_ninth_row_l3717_371784

/-- Given a lattice with 7 numbers in each row, continuing for 9 rows,
    the fifth number in the 9th row is 60. -/
theorem lattice_fifth_number_ninth_row :
  ∀ (lattice : ℕ → ℕ → ℕ),
    (∀ row col, col ≤ 7 → lattice row col = row * col) →
    lattice 9 5 = 60 := by
sorry

end NUMINAMATH_CALUDE_lattice_fifth_number_ninth_row_l3717_371784


namespace NUMINAMATH_CALUDE_max_d_is_one_l3717_371711

/-- The sequence a_n defined as (10^n - 1) / 9 -/
def a (n : ℕ) : ℕ := (10^n - 1) / 9

/-- The greatest common divisor of a_n and a_{n+1} -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- Theorem: The maximum value of d_n is 1 -/
theorem max_d_is_one : ∀ n : ℕ, d n = 1 := by sorry

end NUMINAMATH_CALUDE_max_d_is_one_l3717_371711


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3717_371787

theorem rationalize_denominator : 45 / Real.sqrt 45 = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3717_371787


namespace NUMINAMATH_CALUDE_point_outside_circle_l3717_371718

theorem point_outside_circle (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 → (x - 1)^2 + (y - 1)^2 > 0) ↔ 
  (0 < m ∧ m < 1/4) ∨ m > 1 := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3717_371718


namespace NUMINAMATH_CALUDE_stamps_per_page_l3717_371740

theorem stamps_per_page (a b c : ℕ) (ha : a = 945) (hb : b = 1260) (hc : c = 630) :
  Nat.gcd a (Nat.gcd b c) = 315 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l3717_371740


namespace NUMINAMATH_CALUDE_volume_of_special_parallelepiped_l3717_371710

/-- A rectangular parallelepiped with specific properties -/
structure RectangularParallelepiped where
  /-- Side length of the square face -/
  a : ℝ
  /-- Height perpendicular to the square face -/
  b : ℝ
  /-- The diagonal length is 1 -/
  diagonal_eq_one : 2 * a^2 + b^2 = 1
  /-- The surface area is 1 -/
  surface_area_eq_one : 4 * a * b + 2 * a^2 = 1
  /-- Ensure a and b are positive -/
  a_pos : 0 < a
  b_pos : 0 < b

/-- The volume of a rectangular parallelepiped with the given properties is √2/27 -/
theorem volume_of_special_parallelepiped (p : RectangularParallelepiped) :
  p.a^2 * p.b = Real.sqrt 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_special_parallelepiped_l3717_371710


namespace NUMINAMATH_CALUDE_x_range_l3717_371789

theorem x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) : x > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3717_371789


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3717_371786

/-- Given a quadratic equation mx^2 + x - m^2 + 1 = 0 with -1 as a root, m must equal 1 -/
theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, m*x^2 + x - m^2 + 1 = 0 → x = -1) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3717_371786


namespace NUMINAMATH_CALUDE_calculate_savings_l3717_371701

/-- Given total expenses and savings rate, calculate the amount saved -/
theorem calculate_savings (total_expenses : ℝ) (savings_rate : ℝ) : 
  total_expenses = 24150 ∧ savings_rate = 0.1 → 
  ∃ amount_saved : ℝ, abs (amount_saved - 2683.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_calculate_savings_l3717_371701


namespace NUMINAMATH_CALUDE_find_certain_number_l3717_371716

theorem find_certain_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + x + 45) / 3) + 5 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l3717_371716


namespace NUMINAMATH_CALUDE_second_half_revenue_l3717_371709

/-- Represents the ticket categories --/
inductive TicketCategory
  | A
  | B
  | C

/-- Calculates the total revenue from ticket sales --/
def calculate_revenue (tickets : Nat) (price : Nat) : Nat :=
  tickets * price

/-- Represents the ticket sales data for Richmond Tigers --/
structure TicketSalesData where
  total_tickets : Nat
  first_half_total : Nat
  first_half_A : Nat
  first_half_B : Nat
  first_half_C : Nat
  price_A : Nat
  price_B : Nat
  price_C : Nat

/-- Theorem: The total revenue from the second half of the season is $154,510 --/
theorem second_half_revenue (data : TicketSalesData) 
  (h1 : data.total_tickets = 9570)
  (h2 : data.first_half_total = 3867)
  (h3 : data.first_half_A = 1350)
  (h4 : data.first_half_B = 1150)
  (h5 : data.first_half_C = 1367)
  (h6 : data.price_A = 50)
  (h7 : data.price_B = 40)
  (h8 : data.price_C = 30) :
  calculate_revenue data.first_half_A data.price_A + 
  calculate_revenue data.first_half_B data.price_B + 
  calculate_revenue data.first_half_C data.price_C = 154510 := by
  sorry


end NUMINAMATH_CALUDE_second_half_revenue_l3717_371709


namespace NUMINAMATH_CALUDE_g_of_5_l3717_371776

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3g(2 - x) = 4x^2 - 5x + 1 for all x ∈ ℝ,
    prove that g(5) = -5/4 -/
theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 5 * x + 1) :
  g 5 = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l3717_371776


namespace NUMINAMATH_CALUDE_scientific_notation_141260_l3717_371749

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_141260 :
  toScientificNotation 141260 = ScientificNotation.mk 1.4126 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_141260_l3717_371749


namespace NUMINAMATH_CALUDE_find_a_l3717_371706

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then a^x else -a^(-x)

-- State the theorem
theorem find_a : 
  ∀ a : ℝ, 
  (a > 0) → 
  (a ≠ 1) → 
  (∀ x : ℝ, f a x = -(f a (-x))) → 
  (f a (Real.log 4 / Real.log (1/2)) = -3) → 
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l3717_371706


namespace NUMINAMATH_CALUDE_sample_average_l3717_371766

theorem sample_average (x : ℝ) : 
  (1 + 3 + 2 + 5 + x) / 5 = 3 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_sample_average_l3717_371766


namespace NUMINAMATH_CALUDE_bankers_gain_example_l3717_371726

/-- Calculate the banker's gain given the banker's discount, time, and interest rate. -/
def bankers_gain (bankers_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let face_value := (bankers_discount * 100) / (rate * time)
  let true_discount := (face_value * rate * time) / (100 + rate * time)
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is 360 given the specified conditions. -/
theorem bankers_gain_example : 
  bankers_gain 1360 3 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_example_l3717_371726


namespace NUMINAMATH_CALUDE_log_equation_solution_l3717_371765

theorem log_equation_solution (x : ℝ) :
  0 < x ∧ x ≠ 1 ∧ x < 10 →
  (1 + 2 * (Real.log 2 / Real.log x) * (Real.log (10 - x) / Real.log 4) = 2 / (Real.log x / Real.log 4)) ↔
  (x = 2 ∨ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3717_371765


namespace NUMINAMATH_CALUDE_solve_for_C_l3717_371733

theorem solve_for_C : ∃ C : ℝ, (4 * C - 5 = 23) ∧ (C = 7) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l3717_371733


namespace NUMINAMATH_CALUDE_triangle_acute_angled_l3717_371774

theorem triangle_acute_angled (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (eq : a^3 + b^3 = c^3) :
  c^2 < a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_acute_angled_l3717_371774


namespace NUMINAMATH_CALUDE_min_value_expression_l3717_371760

theorem min_value_expression (x y z : ℝ) (h1 : x * y ≠ 0) (h2 : x + y ≠ 0) :
  ((y + z) / x + 2)^2 + (z / y + 2)^2 + (z / (x + y) - 1)^2 ≥ 5 ∧
  ∃ (x y z : ℝ), x * y ≠ 0 ∧ x + y ≠ 0 ∧
    ((y + z) / x + 2)^2 + (z / y + 2)^2 + (z / (x + y) - 1)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3717_371760


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l3717_371777

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  (Nat.factorial 4) * 
  (Nat.factorial chickens) * 
  (Nat.factorial dogs) * 
  (Nat.factorial cats) * 
  (Nat.factorial rabbits)

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 4 3 5 2 = 414720 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l3717_371777


namespace NUMINAMATH_CALUDE_rotation_equivalence_l3717_371713

/-- Given that a point A is rotated 450 degrees clockwise and y degrees counterclockwise
    about the same center point B, both rotations resulting in the same final position C,
    and y < 360, prove that y = 270. -/
theorem rotation_equivalence (y : ℝ) : 
  (450 % 360 : ℝ) = (360 - y) % 360 → y < 360 → y = 270 := by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l3717_371713


namespace NUMINAMATH_CALUDE_monkey_climb_l3717_371744

/-- Proves that a monkey slips back 2 feet per hour when climbing a 17 ft tree in 15 hours, 
    climbing 3 ft and slipping back a constant distance each hour. -/
theorem monkey_climb (tree_height : ℝ) (total_hours : ℕ) (climb_rate : ℝ) (slip_back : ℝ) : 
  tree_height = 17 →
  total_hours = 15 →
  climb_rate = 3 →
  (total_hours - 1 : ℝ) * (climb_rate - slip_back) + climb_rate = tree_height →
  slip_back = 2 := by
  sorry

#check monkey_climb

end NUMINAMATH_CALUDE_monkey_climb_l3717_371744


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3717_371794

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 16) and (-2, -8) is 7. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (8, 16)
  let p2 : ℝ × ℝ := (-2, -8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 7 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3717_371794


namespace NUMINAMATH_CALUDE_student_selection_problem_l3717_371728

theorem student_selection_problem (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) (n_competitions : ℕ) :
  n_male = 5 →
  n_female = 4 →
  n_select = 3 →
  n_competitions = 2 →
  (Nat.choose (n_male + n_female) n_select - Nat.choose n_male n_select - Nat.choose n_female n_select) *
  (Nat.factorial n_select) = 420 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_problem_l3717_371728


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l3717_371715

theorem trig_fraction_equality (x : ℝ) (h : (1 + Real.sin x) / Real.cos x = -1/2) :
  Real.cos x / (Real.sin x - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l3717_371715


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3717_371714

theorem fifteenth_student_age 
  (total_students : Nat) 
  (average_age : ℝ) 
  (group1_size : Nat) 
  (group1_average : ℝ) 
  (group2_size : Nat) 
  (group2_average : ℝ) 
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_size = 5)
  (h4 : group1_average = 14)
  (h5 : group2_size = 9)
  (h6 : group2_average = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (fifteenth_age : ℝ),
    fifteenth_age = total_students * average_age - (group1_size * group1_average + group2_size * group2_average) ∧
    fifteenth_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3717_371714


namespace NUMINAMATH_CALUDE_school_distance_proof_l3717_371762

/-- The time in hours it takes to drive to school during rush hour -/
def rush_hour_time : ℚ := 18 / 60

/-- The time in hours it takes to drive to school with no traffic -/
def no_traffic_time : ℚ := 12 / 60

/-- The speed increase in mph when there's no traffic -/
def speed_increase : ℚ := 20

/-- The distance to school in miles -/
def distance_to_school : ℚ := 12

theorem school_distance_proof :
  ∃ (rush_hour_speed : ℚ),
    rush_hour_speed * rush_hour_time = distance_to_school ∧
    (rush_hour_speed + speed_increase) * no_traffic_time = distance_to_school := by
  sorry

#check school_distance_proof

end NUMINAMATH_CALUDE_school_distance_proof_l3717_371762


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l3717_371737

theorem pentagon_angle_sum (A B C D E : ℝ) (x y : ℝ) : 
  A = 34 → 
  B = 70 → 
  C = 30 → 
  D = 90 → 
  A + B + C + D + E = 540 → 
  E = 360 - x → 
  180 - y = 120 →
  x + y = 134 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l3717_371737


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3717_371719

/-- Given vectors a and b in ℝ², if a is parallel to b, then the magnitude of b is √5. -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -1 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  ‖b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3717_371719


namespace NUMINAMATH_CALUDE_complementary_angles_proof_l3717_371771

theorem complementary_angles_proof (A B : Real) : 
  A + B = 90 →  -- Angles A and B are complementary
  A = 4 * B →   -- Measure of angle A is 4 times angle B
  A = 72 ∧ B = 18 := by
sorry

end NUMINAMATH_CALUDE_complementary_angles_proof_l3717_371771


namespace NUMINAMATH_CALUDE_double_base_exponent_l3717_371790

theorem double_base_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2 * a)^(2 * b) = a^(2 * b) * y^(2 * b) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_l3717_371790


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l3717_371772

theorem complete_square_equivalence :
  ∀ x : ℝ, 3 * x^2 - 6 * x + 2 = 0 ↔ (x - 1)^2 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l3717_371772


namespace NUMINAMATH_CALUDE_extreme_values_imply_a_b_values_inequality_implies_m_range_l3717_371746

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x - b / x + Real.log x

def g (m x : ℝ) : ℝ := x^2 - 2 * m * x + m

def has_extreme_values (f : ℝ → ℝ) (x₁ x₂ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ (Set.Ioo (x₁ - ε) (x₁ + ε) ∪ Set.Ioo (x₂ - ε) (x₂ + ε)),
    f x ≤ f x₁ ∧ f x ≤ f x₂

theorem extreme_values_imply_a_b_values (a b : ℝ) :
  has_extreme_values (f a b) 1 (1/2) → a = -1/3 ∧ b = -1/3 :=
sorry

theorem inequality_implies_m_range (a b m : ℝ) :
  (a = -1/3 ∧ b = -1/3) →
  (∀ x₁ ∈ Set.Icc (1/2) 2, ∃ x₂ ∈ Set.Icc (1/2) 2, g m x₁ ≥ f a b x₂ - Real.log x₂) →
  m ≤ (3 + Real.sqrt 51) / 6 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_imply_a_b_values_inequality_implies_m_range_l3717_371746


namespace NUMINAMATH_CALUDE_product_of_integers_l3717_371756

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x * y = 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3717_371756


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l3717_371748

def collagen_protein_per_2_scoops : ℕ := 18
def protein_powder_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def arnold_consumption (collagen_scoops protein_scoops : ℕ) : ℕ :=
  (collagen_scoops * collagen_protein_per_2_scoops / 2) + 
  (protein_scoops * protein_powder_per_scoop) + 
  steak_protein

theorem arnold_protein_consumption : 
  arnold_consumption 1 1 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l3717_371748


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l3717_371775

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (hm_ne_n : m ≠ n) 
  (hα_ne_β : α ≠ β) 
  (hm_perp_α : perp m α) 
  (hm_perp_β : perp m β) 
  (hn_perp_α : perp n α) : 
  perp n β := by sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l3717_371775


namespace NUMINAMATH_CALUDE_pet_shop_ducks_l3717_371791

theorem pet_shop_ducks (total : ℕ) (cats : ℕ) (ducks : ℕ) (parrots : ℕ) : 
  cats = 56 →
  ducks = total / 12 →
  ducks = (ducks + parrots) / 4 →
  total = cats + ducks + parrots →
  ducks = 7 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_ducks_l3717_371791


namespace NUMINAMATH_CALUDE_plot_length_is_100_l3717_371725

/-- Proves that the length of a rectangular plot is 100 meters given specific conditions. -/
theorem plot_length_is_100 (width : ℝ) (path_width : ℝ) (gravel_cost_per_sqm : ℝ) (total_gravel_cost : ℝ) :
  width = 65 →
  path_width = 2.5 →
  gravel_cost_per_sqm = 0.4 →
  total_gravel_cost = 340 →
  ∃ (length : ℝ),
    ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * gravel_cost_per_sqm = total_gravel_cost ∧
    length = 100 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_100_l3717_371725


namespace NUMINAMATH_CALUDE_dihedral_angle_inscribed_spheres_l3717_371702

/-- Given two spheres inscribed in a dihedral angle, this theorem proves
    the relationship between the spheres' radii, their position, and the
    measure of the dihedral angle. -/
theorem dihedral_angle_inscribed_spheres 
  (R₁ R₂ : ℝ) -- Radii of the two spheres
  (h_touch : R₁ + R₂ > 0) -- The spheres touch (implied by positive sum of radii)
  (h_ratio : R₁ = 1.5 * R₂) -- Ratio of radii
  (h_angle : Real.cos (45 * π / 180) = Real.sqrt (1 / 2)) -- 45° angle with edge
  : Real.cos (θ / 2) = Real.sqrt ((1 + Real.sqrt (1 / 2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_inscribed_spheres_l3717_371702


namespace NUMINAMATH_CALUDE_seongmin_completion_time_l3717_371799

/-- The number of days it takes Seongmin to complete the task alone -/
def seongmin_days : ℚ := 32

/-- The fraction of work Jinwoo and Seongmin complete together in 8 days -/
def work_together : ℚ := 7/12

/-- The number of days Jinwoo and Seongmin work together -/
def days_together : ℚ := 8

/-- The number of days Jinwoo works alone to complete the remaining work -/
def jinwoo_alone_days : ℚ := 10

theorem seongmin_completion_time :
  let total_work : ℚ := 1
  let work_rate_together : ℚ := work_together / days_together
  let jinwoo_alone_work : ℚ := total_work - work_together
  let jinwoo_work_rate : ℚ := jinwoo_alone_work / jinwoo_alone_days
  let seongmin_work_rate : ℚ := work_rate_together - jinwoo_work_rate
  seongmin_days = total_work / seongmin_work_rate :=
by sorry

end NUMINAMATH_CALUDE_seongmin_completion_time_l3717_371799


namespace NUMINAMATH_CALUDE_pencil_color_fractions_l3717_371763

theorem pencil_color_fractions (L : ℝ) (h1 : L = 9.333333333333332) : 
  let black_fraction : ℝ := 1/8
  let remaining_after_black : ℝ := L - black_fraction * L
  let blue_fraction_of_remaining : ℝ := 7/12
  let white_fraction_of_remaining : ℝ := 1 - blue_fraction_of_remaining
  white_fraction_of_remaining = 5/12 := by
sorry

end NUMINAMATH_CALUDE_pencil_color_fractions_l3717_371763


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3717_371722

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -5) (hy : y = 2) :
  ((x + 2*y)^2 - (x - 2*y)*(2*y + x)) / (4*y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3717_371722


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3717_371705

theorem unique_triple_solution : 
  ∃! (x y z : ℝ), x + y = 4 ∧ x * y - z^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3717_371705


namespace NUMINAMATH_CALUDE_theo_has_winning_strategy_l3717_371780

/-- Game state representing the current turn number and cumulative score -/
structure GameState where
  turn : ℕ
  score : ℕ

/-- Player type -/
inductive Player
| Anatole
| Theo

/-- Game move representing the chosen number and resulting turn score -/
structure GameMove where
  number : ℕ
  turn_score : ℕ

/-- Represents a strategy for a player -/
def Strategy := GameState → GameMove

/-- Checks if a move is valid according to game rules -/
def is_valid_move (p : ℕ) (prev_move : Option GameMove) (current_move : GameMove) : Prop :=
  match prev_move with
  | none => current_move.number > 0
  | some prev => current_move.number > prev.number

/-- Checks if a player wins with a given move -/
def is_winning_move (p : ℕ) (state : GameState) (move : GameMove) : Prop :=
  (p ∣ (move.turn_score * (state.score + state.turn * move.turn_score)))

/-- Theorem stating that Theo has a winning strategy -/
theorem theo_has_winning_strategy (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  ∃ (theo_strategy : Strategy),
    ∀ (anatole_strategy : Strategy),
      ∃ (final_state : GameState),
        final_state.turn < p - 1 ∧
        is_winning_move p final_state (theo_strategy final_state) :=
  sorry

end NUMINAMATH_CALUDE_theo_has_winning_strategy_l3717_371780


namespace NUMINAMATH_CALUDE_rational_numbers_countable_l3717_371788

theorem rational_numbers_countable : ∃ f : ℚ → ℕ+, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_countable_l3717_371788


namespace NUMINAMATH_CALUDE_smaller_circle_with_integer_points_l3717_371727

/-- Given a circle centered at the origin with radius R, there exists a circle
    with radius R/√2 that contains at least as many points with integer coordinates. -/
theorem smaller_circle_with_integer_points (R : ℝ) (R_pos : R > 0) :
  ∃ (R' : ℝ), R' = R / Real.sqrt 2 ∧
  (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 →
    ∃ (x' y' : ℤ), x'^2 + y'^2 ≤ R'^2) :=
by sorry

end NUMINAMATH_CALUDE_smaller_circle_with_integer_points_l3717_371727


namespace NUMINAMATH_CALUDE_inverse_graph_coordinate_sum_l3717_371732

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the theorem
theorem inverse_graph_coordinate_sum :
  (∃ (f : ℝ → ℝ), f 2 = 4 ∧ (∃ (x : ℝ), f⁻¹ x = 2 ∧ x / 4 = 1 / 2)) →
  (∃ (x y : ℝ), y = f⁻¹ x / 4 ∧ x + y = 9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_graph_coordinate_sum_l3717_371732


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3717_371757

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  ∃ (M : ℝ), M = 6 * Real.sqrt 5 ∧ 
  Real.sqrt (3 * x + 4) + Real.sqrt (3 * y + 4) + Real.sqrt (3 * z + 4) ≤ M ∧
  ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 7 ∧
    Real.sqrt (3 * x' + 4) + Real.sqrt (3 * y' + 4) + Real.sqrt (3 * z' + 4) = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3717_371757


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l3717_371792

theorem cubic_sum_inequality (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_of_squares : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 + a*b*c + b*c*d + c*d*a + d*a*b ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l3717_371792


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3717_371735

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence, if a_4 + a_6 + a_8 + a_10 + a_12 = 120, then 2a_10 - a_12 = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3717_371735


namespace NUMINAMATH_CALUDE_min_value_theorem_l3717_371731

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  2 ≤ b / a + 3 / (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3717_371731


namespace NUMINAMATH_CALUDE_book_price_increase_l3717_371767

theorem book_price_increase (initial_price decreased_price final_price : ℝ) 
  (h1 : initial_price = 400)
  (h2 : decreased_price = initial_price * (1 - 0.15))
  (h3 : final_price = 476) :
  (final_price - decreased_price) / decreased_price = 0.4 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l3717_371767


namespace NUMINAMATH_CALUDE_total_amount_theorem_l3717_371752

def calculate_selling_price (purchase_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  purchase_price * (1 - loss_percentage / 100)

def total_amount_received (price1 price2 price3 : ℚ) (loss1 loss2 loss3 : ℚ) : ℚ :=
  calculate_selling_price price1 loss1 +
  calculate_selling_price price2 loss2 +
  calculate_selling_price price3 loss3

theorem total_amount_theorem (price1 price2 price3 loss1 loss2 loss3 : ℚ) :
  price1 = 600 ∧ price2 = 800 ∧ price3 = 1000 ∧
  loss1 = 20 ∧ loss2 = 25 ∧ loss3 = 30 →
  total_amount_received price1 price2 price3 loss1 loss2 loss3 = 1780 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l3717_371752


namespace NUMINAMATH_CALUDE_complement_intersection_equals_l3717_371782

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection_equals : (U \ (A ∩ B)) = {1, 2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_l3717_371782


namespace NUMINAMATH_CALUDE_books_gotten_rid_of_l3717_371796

def initial_stock : ℕ := 27
def shelves_used : ℕ := 3
def books_per_shelf : ℕ := 7

theorem books_gotten_rid_of : 
  initial_stock - (shelves_used * books_per_shelf) = 6 := by
sorry

end NUMINAMATH_CALUDE_books_gotten_rid_of_l3717_371796


namespace NUMINAMATH_CALUDE_parabola_shift_l3717_371764

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola (x + 2) + 2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = 3 * x^2 + 6 * x - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l3717_371764


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l3717_371742

/-- 
Given an equation of the form x²/(4-m) + y²/(m-3) = 1 representing an ellipse with foci on the y-axis,
prove that the range of m is (7/2, 4).
-/
theorem ellipse_foci_y_axis_m_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2/(4-m) + y^2/(m-3) = 1) ∧ 
  (∀ (x y : ℝ), x^2/(4-m) + y^2/(m-3) = 1 → (0 : ℝ) < 4-m ∧ (0 : ℝ) < m-3 ∧ m-3 < 4-m) 
  → 7/2 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_m_range_l3717_371742


namespace NUMINAMATH_CALUDE_exactly_one_even_iff_not_all_odd_or_two_even_l3717_371739

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨
  (Odd a ∧ Even b ∧ Odd c) ∨
  (Odd a ∧ Odd b ∧ Even c)

def all_odd_or_two_even (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∨
  (Even a ∧ Even b) ∨
  (Even a ∧ Even c) ∨
  (Even b ∧ Even c)

theorem exactly_one_even_iff_not_all_odd_or_two_even (a b c : ℕ) :
  exactly_one_even a b c ↔ ¬(all_odd_or_two_even a b c) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_even_iff_not_all_odd_or_two_even_l3717_371739


namespace NUMINAMATH_CALUDE_original_price_calculation_l3717_371707

/-- Proves that if an article is sold for $120 with a 20% gain, its original price was $100. -/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 120 ∧ gain_percent = 20 → 
  selling_price = (100 : ℝ) * (1 + gain_percent / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3717_371707


namespace NUMINAMATH_CALUDE_annette_caitlin_weight_l3717_371741

/-- The combined weight of Annette and Caitlin given the conditions -/
theorem annette_caitlin_weight :
  ∀ (annette caitlin sara : ℝ),
  caitlin + sara = 87 →
  annette = sara + 8 →
  annette + caitlin = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_annette_caitlin_weight_l3717_371741
