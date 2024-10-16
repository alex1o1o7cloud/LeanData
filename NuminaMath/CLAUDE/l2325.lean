import Mathlib

namespace NUMINAMATH_CALUDE_parabola_segment_length_l2325_232546

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus and directrix
def focus : ℝ × ℝ := (2, 0)
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the directrix
def point_on_directrix (P : ℝ × ℝ) : Prop :=
  directrix P.1

-- Define points on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the condition PF = 3MF
def vector_condition (P M : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 
  9 * ((M.1 - focus.1)^2 + (M.2 - focus.2)^2)

-- State the theorem
theorem parabola_segment_length 
  (P M N : ℝ × ℝ) 
  (h1 : point_on_directrix P) 
  (h2 : point_on_parabola M) 
  (h3 : point_on_parabola N) 
  (h4 : vector_condition P M) :
  let MN_length := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  MN_length = 32/3 := by sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l2325_232546


namespace NUMINAMATH_CALUDE_vector_equation_l2325_232534

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)

theorem vector_equation : A - D + D - C - (A - B) = B - C := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2325_232534


namespace NUMINAMATH_CALUDE_triangle_radius_inequalities_l2325_232520

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define perimeter, circumradius, and inradius
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c
def circumradius (t : Triangle) : ℝ := sorry
def inradius (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_radius_inequalities :
  ∃ t1 t2 t3 : Triangle,
    ¬(perimeter t1 > circumradius t1 + inradius t1) ∧
    ¬(perimeter t2 ≤ circumradius t2 + inradius t2) ∧
    ¬(perimeter t3 / 6 < circumradius t3 + inradius t3 ∧ circumradius t3 + inradius t3 < 6 * perimeter t3) :=
  sorry

end NUMINAMATH_CALUDE_triangle_radius_inequalities_l2325_232520


namespace NUMINAMATH_CALUDE_james_cattle_profit_l2325_232582

def cattle_profit (num_cattle : ℕ) (purchase_price : ℕ) (feeding_cost_percentage : ℕ) 
                  (weight_per_cattle : ℕ) (selling_price_per_pound : ℕ) : ℕ :=
  let feeding_cost := purchase_price * feeding_cost_percentage / 100
  let total_cost := purchase_price + feeding_cost
  let selling_price_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_selling_price := num_cattle * selling_price_per_cattle
  total_selling_price - total_cost

theorem james_cattle_profit :
  cattle_profit 100 40000 20 1000 2 = 112000 := by
  sorry

end NUMINAMATH_CALUDE_james_cattle_profit_l2325_232582


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2325_232512

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 135 → n * (180 - interior_angle) = 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2325_232512


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2325_232556

theorem max_value_of_expression (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_one : a + b + c + d = 1) :
  (∀ x y z w : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 → x + y + z + w = 1 →
    (x*y)/(x+y) + (x*z)/(x+z) + (x*w)/(x+w) + 
    (y*z)/(y+z) + (y*w)/(y+w) + (z*w)/(z+w) ≤ 1/2) ∧
  ((a*b)/(a+b) + (a*c)/(a+c) + (a*d)/(a+d) + 
   (b*c)/(b+c) + (b*d)/(b+d) + (c*d)/(c+d) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2325_232556


namespace NUMINAMATH_CALUDE_inverse_function_problem_l2325_232531

/-- Given a function g(x) = 4x - 6 and its relation to the inverse of f(x) = ax + b,
    prove that 4a + 3b = 4 -/
theorem inverse_function_problem (a b : ℝ) :
  (∀ x, (4 * x - 6 : ℝ) = (Function.invFun (fun x => a * x + b) x) - 2) →
  4 * a + 3 * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l2325_232531


namespace NUMINAMATH_CALUDE_shirt_pricing_theorem_l2325_232568

/-- Represents the monthly sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 2600

/-- Represents the profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 50) * (sales_volume x)

/-- The cost price of each shirt -/
def cost_price : ℝ := 50

/-- The constraint that selling price is not less than cost price -/
def price_constraint (x : ℝ) : Prop := x ≥ cost_price

/-- The constraint that profit per unit should not exceed 30% of cost price -/
def profit_constraint (x : ℝ) : Prop := (x - cost_price) / cost_price ≤ 0.3

theorem shirt_pricing_theorem :
  ∃ (x : ℝ), price_constraint x ∧ profit x = 24000 ∧ x = 70 ∧
  ∃ (y : ℝ), price_constraint y ∧ profit_constraint y ∧
    (∀ z, price_constraint z → profit_constraint z → profit z ≤ profit y) ∧
    y = 65 ∧ profit y = 19500 := by sorry

end NUMINAMATH_CALUDE_shirt_pricing_theorem_l2325_232568


namespace NUMINAMATH_CALUDE_frog_jump_distance_l2325_232580

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_grasshopper_diff : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_grasshopper_diff = 39) : 
  grasshopper_jump + frog_grasshopper_diff = 58 := by
sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l2325_232580


namespace NUMINAMATH_CALUDE_power_sum_equality_l2325_232529

theorem power_sum_equality : 2^345 + 3^5 * 3^3 = 2^345 + 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2325_232529


namespace NUMINAMATH_CALUDE_quadrilateral_areas_product_is_square_l2325_232510

/-- Represents a convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by the diagonals -/
  areas : Fin 4 → ℕ

/-- Theorem: The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem quadrilateral_areas_product_is_square (q : ConvexQuadrilateral) :
  ∃ k : ℕ, (q.areas 0) * (q.areas 1) * (q.areas 2) * (q.areas 3) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_areas_product_is_square_l2325_232510


namespace NUMINAMATH_CALUDE_expression_value_l2325_232591

theorem expression_value : 
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2325_232591


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2325_232543

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 + y^3 = 640 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2325_232543


namespace NUMINAMATH_CALUDE_delta_sports_club_ratio_l2325_232593

/-- Proves that the ratio of female to male members is 2/3 given the average ages --/
theorem delta_sports_club_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (35 : ℝ) * f + 30 * m = 32 * (f + m) → f / m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_delta_sports_club_ratio_l2325_232593


namespace NUMINAMATH_CALUDE_work_completion_time_proof_l2325_232563

/-- Represents the time in days for a person to complete the work alone -/
structure WorkTime :=
  (days : ℚ)
  (days_pos : days > 0)

/-- Represents the combined work rate of multiple people -/
def combined_work_rate (work_times : List WorkTime) : ℚ :=
  work_times.map (λ wt => 1 / wt.days) |> List.sum

/-- The time required for the group to complete the work together -/
def group_work_time (work_times : List WorkTime) : ℚ :=
  1 / combined_work_rate work_times

theorem work_completion_time_proof 
  (david_time : WorkTime)
  (john_time : WorkTime)
  (mary_time : WorkTime)
  (h1 : david_time.days = 5)
  (h2 : john_time.days = 9)
  (h3 : mary_time.days = 7) :
  ⌈group_work_time [david_time, john_time, mary_time]⌉ = 3 := by
  sorry

#eval ⌈(315 : ℚ) / 143⌉

end NUMINAMATH_CALUDE_work_completion_time_proof_l2325_232563


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_l2325_232514

theorem isosceles_triangle_areas (a b c : ℝ) (W X Y : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  W = (1/2) * a^2 →
  X = (1/2) * b^2 →
  Y = (1/2) * c^2 →
  W + X = Y := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_l2325_232514


namespace NUMINAMATH_CALUDE_function_equivalence_l2325_232541

-- Define the function f
noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 + 1/x - 2

-- State the theorem
theorem function_equivalence (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) (hx3 : x ≠ -1) :
  f (x - 1/x) = x / (x^2 - 1) - x^2 - 1/x^2 :=
by
  sorry


end NUMINAMATH_CALUDE_function_equivalence_l2325_232541


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2325_232554

/-- Race problem statement -/
theorem race_speed_ratio :
  ∀ (vA vB : ℝ) (d : ℝ),
  d > 0 →
  d / vA = 2 →
  d / vB = 1.5 →
  vA / vB = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2325_232554


namespace NUMINAMATH_CALUDE_susan_bob_cat_difference_l2325_232544

/-- Proves that Susan has 6 more cats than Bob after all exchanges --/
theorem susan_bob_cat_difference : 
  let susan_initial : ℕ := 21
  let bob_initial : ℕ := 3
  let emma_initial : ℕ := 8
  let neighbor_to_susan : ℕ := 12
  let neighbor_to_bob : ℕ := 14
  let neighbor_to_emma : ℕ := 6
  let susan_to_bob : ℕ := 6
  let emma_to_susan : ℕ := 5
  let emma_to_bob : ℕ := 3

  let susan_final := susan_initial + neighbor_to_susan - susan_to_bob + emma_to_susan
  let bob_final := bob_initial + neighbor_to_bob + susan_to_bob + emma_to_bob

  susan_final - bob_final = 6 :=
by sorry

end NUMINAMATH_CALUDE_susan_bob_cat_difference_l2325_232544


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2325_232518

/-- An isosceles right triangle with an inscribed circle -/
structure IsoscelesRightTriangle where
  -- The length of a leg of the triangle
  leg : ℝ
  -- The center of the inscribed circle
  center : ℝ × ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- The area of the inscribed circle is 9π
  circle_area : radius^2 * Real.pi = 9 * Real.pi

/-- The area of an isosceles right triangle with an inscribed circle of area 9π is 36 -/
theorem isosceles_right_triangle_area 
  (triangle : IsoscelesRightTriangle) : triangle.leg^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2325_232518


namespace NUMINAMATH_CALUDE_line_points_relation_l2325_232599

/-- Given a line in the xy-coordinate system with equation x = 2y + 5,
    if (m, n) and (m + 1, n + k) are two points on this line,
    then k = 1/2 -/
theorem line_points_relation (m n k : ℝ) : 
  (m = 2 * n + 5) →  -- (m, n) is on the line
  (m + 1 = 2 * (n + k) + 5) →  -- (m + 1, n + k) is on the line
  k = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_points_relation_l2325_232599


namespace NUMINAMATH_CALUDE_sum_25887_2014_not_even_l2325_232509

theorem sum_25887_2014_not_even : ¬ Even (25887 + 2014) := by
  sorry

end NUMINAMATH_CALUDE_sum_25887_2014_not_even_l2325_232509


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2325_232539

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define point D on AC
def PointOnLine (D A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))

-- Define the distance BD
def DistanceBD (B D : ℝ × ℝ) : Prop :=
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6

-- Define the ratio AD:DC
def RatioADDC (A D C : ℝ × ℝ) : Prop :=
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let DC := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  AD / DC = 18 / 7

-- Theorem statement
theorem triangle_ratio_theorem (A B C D : ℝ × ℝ) :
  Triangle A B C → PointOnLine D A C → DistanceBD B D → RatioADDC A D C :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2325_232539


namespace NUMINAMATH_CALUDE_trap_speed_constant_and_eight_l2325_232562

/-- Representation of a 4-level staircase --/
structure Staircase :=
  (h : ℝ)  -- height of each step
  (b : ℝ)  -- width of each step
  (a : ℝ)  -- length of the staircase
  (v : ℝ)  -- speed of the mouse

/-- The speed of the mouse trap required to catch the mouse --/
def trap_speed (s : Staircase) : ℝ := 8

/-- Theorem stating that the trap speed is constant and equal to 8 cm/s --/
theorem trap_speed_constant_and_eight (s : Staircase) 
  (h_height : s.h = 3)
  (h_width : s.b = 1)
  (h_length : s.a = 8)
  (h_mouse_speed : s.v = 17) :
  trap_speed s = 8 ∧ 
  ∀ (placement : ℝ), 0 ≤ placement ∧ placement ≤ s.a → trap_speed s = 8 := by
  sorry

#check trap_speed_constant_and_eight

end NUMINAMATH_CALUDE_trap_speed_constant_and_eight_l2325_232562


namespace NUMINAMATH_CALUDE_time_after_56_hours_l2325_232567

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Adds hours to a given time -/
def addHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + h * 60
  { hour := (totalMinutes / 60) % 24, minute := totalMinutes % 60 }

theorem time_after_56_hours (start : Time) (h : Nat) :
  start = { hour := 9, minute := 4 } →
  h = 56 →
  addHours start h = { hour := 17, minute := 4 } := by
  sorry

end NUMINAMATH_CALUDE_time_after_56_hours_l2325_232567


namespace NUMINAMATH_CALUDE_identical_numbers_iff_even_l2325_232585

/-- A function that represents the operation of selecting two numbers and replacing them with their sum. -/
def sumOperation (numbers : List ℕ) : List (List ℕ) :=
  sorry

/-- A predicate that checks if all numbers in a list are identical. -/
def allIdentical (numbers : List ℕ) : Prop :=
  sorry

/-- A proposition stating that it's possible to transform n numbers into n identical numbers
    using the sum operation if and only if n is even. -/
theorem identical_numbers_iff_even (n : ℕ) (h : n ≥ 2) :
  (∃ (initial : List ℕ) (final : List ℕ),
    initial.length = n ∧
    final.length = n ∧
    allIdentical final ∧
    final ∈ sumOperation initial) ↔ Even n :=
  sorry

end NUMINAMATH_CALUDE_identical_numbers_iff_even_l2325_232585


namespace NUMINAMATH_CALUDE_minimum_students_l2325_232537

theorem minimum_students (b g : ℕ) : 
  (2 * (b / 2) = 2 * (g * 2 / 3) + 5) →  -- Half of boys equals 2/3 of girls plus 5
  (b ≥ g) →                             -- There are at least as many boys as girls
  (b + g ≥ 17) ∧                        -- The total number of students is at least 17
  (∀ b' g' : ℕ, (2 * (b' / 2) = 2 * (g' * 2 / 3) + 5) → (b' + g' < 17) → (b' < g')) :=
by
  sorry

#check minimum_students

end NUMINAMATH_CALUDE_minimum_students_l2325_232537


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l2325_232594

theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_profit : selling_price = cost_price * (1 + 0.25)) :
  cost_price / selling_price = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l2325_232594


namespace NUMINAMATH_CALUDE_exists_vector_not_in_span_l2325_232547

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-4, -2)

/-- The statement to be proven -/
theorem exists_vector_not_in_span : ∃ d : ℝ × ℝ, ∀ k₁ k₂ : ℝ, d ≠ k₁ • b + k₂ • c := by
  sorry

end NUMINAMATH_CALUDE_exists_vector_not_in_span_l2325_232547


namespace NUMINAMATH_CALUDE_real_part_of_z_l2325_232598

theorem real_part_of_z (z : ℂ) (h1 : Complex.abs (z - 1) = 2) (h2 : Complex.abs (z^2 - 1) = 6) :
  z.re = 5/4 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2325_232598


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2325_232575

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : ℕ, isThreeDigitBase8 n → n ≤ 774 ∨ ¬(7 ∣ base8ToDecimal n) :=
by sorry

#check greatest_3digit_base8_divisible_by_7

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2325_232575


namespace NUMINAMATH_CALUDE_lilith_water_bottle_price_l2325_232500

/-- The regular price per water bottle in Lilith's town -/
def regularPrice : ℚ := 185 / 100

theorem lilith_water_bottle_price :
  let initialBottles : ℕ := 60
  let initialPrice : ℚ := 2
  let shortfall : ℚ := 9
  (initialBottles : ℚ) * regularPrice = initialBottles * initialPrice - shortfall :=
by sorry

end NUMINAMATH_CALUDE_lilith_water_bottle_price_l2325_232500


namespace NUMINAMATH_CALUDE_sum_even_odd_is_odd_l2325_232503

theorem sum_even_odd_is_odd (a b : ℤ) (h1 : Even a) (h2 : Odd b) : Odd (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_even_odd_is_odd_l2325_232503


namespace NUMINAMATH_CALUDE_three_digit_numbers_theorem_l2325_232528

def digits : List Nat := [3, 4, 5, 7, 9]

def isValidNumber (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits

def validNumbers : List Nat :=
  (List.range 900).filter (fun n => n ≥ 300 ∧ isValidNumber n)

theorem three_digit_numbers_theorem :
  validNumbers.length = 60 ∧ validNumbers.sum = 37296 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_theorem_l2325_232528


namespace NUMINAMATH_CALUDE_abc_product_magnitude_l2325_232579

theorem abc_product_magnitude (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_eq : a + 1 / b^2 = b + 1 / c^2 ∧ b + 1 / c^2 = c + 1 / a^2) : 
  |a * b * c| = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_product_magnitude_l2325_232579


namespace NUMINAMATH_CALUDE_sugar_amount_is_two_l2325_232586

-- Define the ratios and quantities
def sugar_to_cheese_ratio : ℚ := 1 / 4
def vanilla_to_cheese_ratio : ℚ := 1 / 2
def eggs_to_vanilla_ratio : ℚ := 2
def eggs_used : ℕ := 8

-- Define the function to calculate sugar used
def sugar_used (eggs : ℕ) : ℚ :=
  (eggs : ℚ) / eggs_to_vanilla_ratio / vanilla_to_cheese_ratio * sugar_to_cheese_ratio

-- Theorem statement
theorem sugar_amount_is_two : sugar_used eggs_used = 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_is_two_l2325_232586


namespace NUMINAMATH_CALUDE_marco_marie_age_ratio_l2325_232552

theorem marco_marie_age_ratio :
  ∀ (x : ℕ) (marco_age marie_age : ℕ),
    marie_age = 12 →
    marco_age = x * marie_age + 1 →
    marco_age + marie_age = 37 →
    (marco_age : ℚ) / marie_age = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_marco_marie_age_ratio_l2325_232552


namespace NUMINAMATH_CALUDE_prob_square_divisor_15_factorial_l2325_232524

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n that are perfect squares -/
def num_square_divisors (n : ℕ) : ℕ := sorry

/-- The total number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The probability of choosing a perfect square divisor from the positive integer divisors of n -/
def prob_square_divisor (n : ℕ) : ℚ :=
  (num_square_divisors n : ℚ) / (num_divisors n : ℚ)

theorem prob_square_divisor_15_factorial :
  prob_square_divisor (factorial 15) = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_square_divisor_15_factorial_l2325_232524


namespace NUMINAMATH_CALUDE_dave_tickets_l2325_232540

theorem dave_tickets (tickets_used : ℕ) (tickets_left : ℕ) : 
  tickets_used = 6 → tickets_left = 7 → tickets_used + tickets_left = 13 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l2325_232540


namespace NUMINAMATH_CALUDE_max_ab_perpendicular_lines_l2325_232538

theorem max_ab_perpendicular_lines (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, 2 * x + (2 * a - 4) * y + 1 = 0 ↔ 2 * b * x + y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (2 * x₁ + (2 * a - 4) * y₁ + 1 = 0 ∧ 2 * x₂ + (2 * a - 4) * y₂ + 1 = 0 ∧ x₁ ≠ x₂) →
    (2 * b * x₁ + y₁ - 2 = 0 ∧ 2 * b * x₂ + y₂ - 2 = 0 ∧ x₁ ≠ x₂) →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₂ - y₁) / (x₂ - x₁)) = -1) →
  ∀ c : ℝ, a * b ≤ c → c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_ab_perpendicular_lines_l2325_232538


namespace NUMINAMATH_CALUDE_find_A_minus_C_l2325_232507

theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84)
  (h2 : B + C = 60)
  (h3 : A = B + B + B + B + B + B)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  A - C = 24 := by
sorry

end NUMINAMATH_CALUDE_find_A_minus_C_l2325_232507


namespace NUMINAMATH_CALUDE_simplify_expression_l2325_232523

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 6) - (2*x + 6)*(3*x - 2) = -3*x^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2325_232523


namespace NUMINAMATH_CALUDE_remainder_3045_div_32_l2325_232516

theorem remainder_3045_div_32 : 3045 % 32 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_3045_div_32_l2325_232516


namespace NUMINAMATH_CALUDE_reflection_property_l2325_232553

/-- A reflection in R² --/
structure Reflection where
  /-- The reflection function --/
  reflect : ℝ × ℝ → ℝ × ℝ

/-- Given a reflection that maps (2, -3) to (-2, 9), it also maps (3, 1) to (-3, 1) --/
theorem reflection_property (r : Reflection) 
  (h1 : r.reflect (2, -3) = (-2, 9)) : 
  r.reflect (3, 1) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_property_l2325_232553


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l2325_232597

-- Define the days of the week
inductive Day : Type
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day
  | Sunday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (addDays d m)

-- Theorem statement
theorem tomorrow_is_saturday (dayBeforeYesterday : Day) :
  addDays dayBeforeYesterday 5 = Day.Monday →
  addDays dayBeforeYesterday 7 = Day.Saturday :=
by
  sorry


end NUMINAMATH_CALUDE_tomorrow_is_saturday_l2325_232597


namespace NUMINAMATH_CALUDE_geometric_sequence_special_property_l2325_232571

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₂ · a₄ = 2a₃ - 1, then a₃ = 1 -/
theorem geometric_sequence_special_property (a : ℕ → ℝ) :
  geometric_sequence a → a 2 * a 4 = 2 * a 3 - 1 → a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_property_l2325_232571


namespace NUMINAMATH_CALUDE_dividend_calculation_l2325_232551

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 13 → quotient = 17 → remainder = 1 → 
  dividend = divisor * quotient + remainder →
  dividend = 222 := by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2325_232551


namespace NUMINAMATH_CALUDE_rod_solution_l2325_232555

/-- Represents a rod divided into parts -/
structure Rod where
  m : ℕ  -- number of red parts
  n : ℕ  -- number of black parts
  x : ℕ  -- number of coinciding lines
  total_segments : ℕ  -- total number of segments after cutting
  longest_segments : ℕ  -- number of longest segments

/-- Conditions for the rod problem -/
def rod_conditions (r : Rod) : Prop :=
  r.m > r.n ∧
  r.total_segments = 170 ∧
  r.longest_segments = 100

/-- Theorem stating the solution to the rod problem -/
theorem rod_solution (r : Rod) (h : rod_conditions r) : r.m = 13 ∧ r.n = 156 :=
sorry

/-- Lemma stating that x + 1 is a common divisor of m and n -/
lemma common_divisor (r : Rod) : (r.x + 1) ∣ r.m ∧ (r.x + 1) ∣ r.n :=
sorry

end NUMINAMATH_CALUDE_rod_solution_l2325_232555


namespace NUMINAMATH_CALUDE_split_bill_example_l2325_232587

/-- Calculates the amount each person should pay when splitting a bill equally -/
def split_bill (num_people : ℕ) (num_bread : ℕ) (bread_price : ℕ) (num_hotteok : ℕ) (hotteok_price : ℕ) : ℕ :=
  ((num_bread * bread_price + num_hotteok * hotteok_price) / num_people)

/-- Theorem stating that given the conditions, each person should pay 1650 won -/
theorem split_bill_example : split_bill 4 5 200 7 800 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_split_bill_example_l2325_232587


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l2325_232559

/-- A triangle with sides 6, 8, and 10 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 6)
  (hb : b = 8)
  (hc : c = 10)

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def longest_altitudes_sum (t : Triangle) : ℝ := sorry

/-- Theorem stating that the sum of the lengths of the two longest altitudes is 14 -/
theorem longest_altitudes_sum_is_14 (t : Triangle) : longest_altitudes_sum t = 14 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l2325_232559


namespace NUMINAMATH_CALUDE_remainder_problem_l2325_232532

theorem remainder_problem (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2325_232532


namespace NUMINAMATH_CALUDE_quadratic_triple_root_relation_l2325_232508

/-- Given a quadratic equation ax^2 + bx + c = 0 where one root is triple the other,
    prove that 3b^2 = 16ac -/
theorem quadratic_triple_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_relation_l2325_232508


namespace NUMINAMATH_CALUDE_red_marbles_count_l2325_232572

/-- The number of red marbles Mary gave to Dan -/
def red_marbles : ℕ := 78 - 64

theorem red_marbles_count : red_marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l2325_232572


namespace NUMINAMATH_CALUDE_prob_both_white_one_third_l2325_232560

/-- Represents a bag of balls -/
structure Bag where
  white : Nat
  yellow : Nat

/-- Calculates the probability of drawing a white ball from a bag -/
def probWhite (bag : Bag) : Rat :=
  bag.white / (bag.white + bag.yellow)

/-- The probability of drawing white balls from both bags -/
def probBothWhite (bagA bagB : Bag) : Rat :=
  probWhite bagA * probWhite bagB

theorem prob_both_white_one_third :
  let bagA : Bag := { white := 1, yellow := 1 }
  let bagB : Bag := { white := 2, yellow := 1 }
  probBothWhite bagA bagB = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_one_third_l2325_232560


namespace NUMINAMATH_CALUDE_sweets_distribution_l2325_232525

theorem sweets_distribution (total_children : Nat) (absent_children : Nat) (extra_sweets : Nat) 
  (h1 : total_children = 256)
  (h2 : absent_children = 64)
  (h3 : extra_sweets = 12) :
  let original_sweets := (total_children - absent_children) * extra_sweets / absent_children
  original_sweets = 36 := by
sorry

end NUMINAMATH_CALUDE_sweets_distribution_l2325_232525


namespace NUMINAMATH_CALUDE_complex_power_of_sqrt2i_l2325_232519

theorem complex_power_of_sqrt2i :
  ∀ z : ℂ, z = Complex.I * Real.sqrt 2 → z^4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_sqrt2i_l2325_232519


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2325_232526

/-- A line that does not pass through the second quadrant -/
theorem line_not_in_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, x - y - a^2 = 0 → ¬(x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2325_232526


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l2325_232513

theorem shaded_area_between_circles (r : ℝ) (chord_length : ℝ) : 
  r = 25 → 
  chord_length = 60 → 
  (π * ((r^2 + (chord_length/2)^2) - r^2)) = 900 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l2325_232513


namespace NUMINAMATH_CALUDE_palindrome_with_five_percentage_l2325_232578

/-- A function that checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool :=
  sorry

/-- A function that checks if a natural number contains the digit 5 -/
def containsFive (n : ℕ) : Bool :=
  sorry

/-- The set of palindromes between 100 and 1000 (inclusive) -/
def palindromes : Finset ℕ :=
  sorry

/-- The set of palindromes between 100 and 1000 (inclusive) containing at least one 5 -/
def palindromesWithFive : Finset ℕ :=
  sorry

theorem palindrome_with_five_percentage :
  (palindromesWithFive.card : ℚ) / palindromes.card * 100 = 37 / 180 * 100 :=
sorry

end NUMINAMATH_CALUDE_palindrome_with_five_percentage_l2325_232578


namespace NUMINAMATH_CALUDE_car_speed_problem_l2325_232535

/-- Proves that given a car traveling for two hours with a speed of 90 km/h in the first hour
    and an average speed of 72.5 km/h over the two hours, the speed in the second hour must be 55 km/h. -/
theorem car_speed_problem (speed_first_hour : ℝ) (average_speed : ℝ) (speed_second_hour : ℝ) :
  speed_first_hour = 90 →
  average_speed = 72.5 →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_second_hour = 55 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2325_232535


namespace NUMINAMATH_CALUDE_divided_square_area_l2325_232589

/-- Represents a square divided into rectangles -/
structure DividedSquare where
  side_length : ℝ
  vertical_lines : ℕ
  horizontal_lines : ℕ

/-- Calculates the total perimeter of all rectangles in a divided square -/
def total_perimeter (s : DividedSquare) : ℝ :=
  4 * s.side_length + 2 * s.side_length * (s.vertical_lines * (s.horizontal_lines + 1) + s.horizontal_lines * (s.vertical_lines + 1))

/-- The main theorem -/
theorem divided_square_area (s : DividedSquare) 
  (h1 : s.vertical_lines = 5)
  (h2 : s.horizontal_lines = 3)
  (h3 : (s.vertical_lines + 1) * (s.horizontal_lines + 1) = 24)
  (h4 : total_perimeter s = 24) :
  s.side_length ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divided_square_area_l2325_232589


namespace NUMINAMATH_CALUDE_game_points_sequence_l2325_232506

theorem game_points_sequence (a : ℕ → ℕ) : 
  a 1 = 2 ∧ 
  a 3 = 5 ∧ 
  a 4 = 8 ∧ 
  a 5 = 12 ∧ 
  a 6 = 17 ∧ 
  (∀ n : ℕ, n > 1 → (a (n + 1) - a n) - (a n - a (n - 1)) = 1) →
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_game_points_sequence_l2325_232506


namespace NUMINAMATH_CALUDE_store_discount_income_increase_l2325_232590

theorem store_discount_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (quantity_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : quantity_increase_rate = 0.2) : 
  let new_price := original_price * (1 - discount_rate)
  let new_quantity := original_quantity * (1 + quantity_increase_rate)
  let original_income := original_price * original_quantity
  let new_income := new_price * new_quantity
  (new_income - original_income) / original_income = 0.08 := by
sorry

end NUMINAMATH_CALUDE_store_discount_income_increase_l2325_232590


namespace NUMINAMATH_CALUDE_max_watch_display_sum_l2325_232521

def is_valid_hour (h : ℕ) : Prop := 1 ≤ h ∧ h ≤ 12

def is_valid_minute (m : ℕ) : Prop := m < 60

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10 + digit_sum (n / 10))

def watch_display_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

theorem max_watch_display_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  watch_display_sum h' m' ≤ watch_display_sum h m ∧
  watch_display_sum h m = 23 :=
sorry

end NUMINAMATH_CALUDE_max_watch_display_sum_l2325_232521


namespace NUMINAMATH_CALUDE_circle_through_points_is_valid_circle_equation_l2325_232583

/-- Given three points in 2D space, this function returns true if they lie on the circle
    described by the equation x^2 + y^2 - 4x - 6y = 0 -/
def points_on_circle (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let f := fun (x y : ℝ) => x^2 + y^2 - 4*x - 6*y
  f p1.1 p1.2 = 0 ∧ f p2.1 p2.2 = 0 ∧ f p3.1 p3.2 = 0

/-- The theorem states that the points (0,0), (4,0), and (-1,1) lie on the circle
    described by the equation x^2 + y^2 - 4x - 6y = 0 -/
theorem circle_through_points :
  points_on_circle (0, 0) (4, 0) (-1, 1) := by
  sorry

/-- The general equation of a circle is x^2 + y^2 + Dx + Ey + F = 0 -/
def is_circle_equation (D E F : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = x^2 + y^2 + D*x + E*y + F

/-- This theorem states that the equation x^2 + y^2 - 4x - 6y = 0 is a valid circle equation -/
theorem is_valid_circle_equation :
  is_circle_equation (-4) (-6) 0 (fun x y => x^2 + y^2 - 4*x - 6*y) := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_is_valid_circle_equation_l2325_232583


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_one_result_l2325_232536

theorem opposite_reciprocal_abs_one_result (a b c d m : ℝ) : 
  (a = -b) → 
  (c * d = 1) → 
  (|m| = 1) → 
  ((a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009) :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_one_result_l2325_232536


namespace NUMINAMATH_CALUDE_triangle_side_altitude_sum_l2325_232574

theorem triangle_side_altitude_sum (x y : ℝ) : 
  x < 75 →
  y < 28 →
  x * 60 = 75 * 28 →
  100 * y = 75 * 28 →
  x + y = 56 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_altitude_sum_l2325_232574


namespace NUMINAMATH_CALUDE_rope_length_l2325_232522

/-- The length of the shorter piece of rope -/
def shorter_piece : ℝ := 20

/-- The length of the longer piece of rope -/
def longer_piece : ℝ := 2 * shorter_piece

/-- The original length of the rope -/
def original_length : ℝ := shorter_piece + longer_piece

/-- Theorem stating that the original length of the rope is 60 meters -/
theorem rope_length : original_length = 60 := by sorry

end NUMINAMATH_CALUDE_rope_length_l2325_232522


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2325_232557

theorem arithmetic_mean_problem (x : ℝ) : 
  (x + 8 + 15 + 2*x + 13 + 2*x + 4) / 5 = 24 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2325_232557


namespace NUMINAMATH_CALUDE_total_pencils_l2325_232517

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) :
  jessica_pencils = 8 →
  sandy_pencils = 8 →
  jason_pencils = 8 →
  jessica_pencils + sandy_pencils + jason_pencils = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l2325_232517


namespace NUMINAMATH_CALUDE_brad_books_this_month_l2325_232501

theorem brad_books_this_month (william_last_month : ℕ) (brad_last_month : ℕ) (william_total : ℕ) (brad_total : ℕ) :
  william_last_month = 6 →
  brad_last_month = 3 * william_last_month →
  william_total = brad_total + 4 →
  william_total = william_last_month + 2 * (brad_total - brad_last_month) →
  brad_total - brad_last_month = 16 := by
  sorry

end NUMINAMATH_CALUDE_brad_books_this_month_l2325_232501


namespace NUMINAMATH_CALUDE_three_number_average_l2325_232581

theorem three_number_average (a b c : ℝ) 
  (h1 : (a + b) / 2 = 26.5)
  (h2 : (b + c) / 2 = 34.5)
  (h3 : (a + c) / 2 = 29)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a + b + c) / 3 = 30 := by
sorry

end NUMINAMATH_CALUDE_three_number_average_l2325_232581


namespace NUMINAMATH_CALUDE_train_length_l2325_232545

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 225 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2325_232545


namespace NUMINAMATH_CALUDE_square_sum_mod_three_solution_l2325_232511

theorem square_sum_mod_three_solution (x y z : ℕ) :
  (x^2 + y^2 + z^2) % 3 = 1 →
  ((x = 3 ∧ y = 3 ∧ z = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 3) ∨
   (x = 2 ∧ y = 3 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_mod_three_solution_l2325_232511


namespace NUMINAMATH_CALUDE_stone_length_proof_l2325_232584

/-- Given a hall and stones with specific dimensions, prove the length of each stone --/
theorem stone_length_proof (hall_length hall_width : ℝ) (stone_width : ℝ) (num_stones : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_width = 0.5)
  (h4 : num_stones = 1800) :
  (hall_length * hall_width * 100) / (stone_width * 10 * num_stones) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stone_length_proof_l2325_232584


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l2325_232549

theorem fraction_equality_solution (x : ℚ) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l2325_232549


namespace NUMINAMATH_CALUDE_prime_pairs_square_sum_l2325_232504

theorem prime_pairs_square_sum (p q : ℕ) : 
  Prime p → Prime q → (∃ n : ℕ, p^2 + 5*p*q + 4*q^2 = n^2) → 
  ((p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_square_sum_l2325_232504


namespace NUMINAMATH_CALUDE_counterexample_twelve_l2325_232569

theorem counterexample_twelve : ∃ n : ℕ, 
  ¬(Nat.Prime n) ∧ (n = 12) ∧ ¬(Nat.Prime (n - 1) ∧ Nat.Prime (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_twelve_l2325_232569


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2325_232558

-- Define the function f
def f (b c x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- State the theorem
theorem min_value_of_reciprocal_sum (b c : ℝ) (x₁ x₂ : ℝ) :
  (∃ (b c : ℝ), f b c (-10) = f b c 12) →  -- f(-10) = f(12)
  (x₁ > 0 ∧ x₂ > 0) →  -- x₁ and x₂ are positive
  (f b c x₁ = 0 ∧ f b c x₂ = 0) →  -- x₁ and x₂ are roots of f(x) = 0
  (∀ y z : ℝ, y > 0 ∧ z > 0 ∧ f b c y = 0 ∧ f b c z = 0 → 1/y + 1/z ≥ 1/x₁ + 1/x₂) →  -- x₁ and x₂ give the minimum value
  1/x₁ + 1/x₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2325_232558


namespace NUMINAMATH_CALUDE_inequality_proof_l2325_232561

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2325_232561


namespace NUMINAMATH_CALUDE_seokgi_jumped_furthest_l2325_232548

/-- Represents the jump distances of three people -/
structure JumpDistances where
  yooseung : ℚ
  shinyoung : ℚ
  seokgi : ℚ

/-- Given the jump distances, proves that Seokgi jumped the furthest -/
theorem seokgi_jumped_furthest (j : JumpDistances)
  (h1 : j.yooseung = 15/8)
  (h2 : j.shinyoung = 2)
  (h3 : j.seokgi = 17/8) :
  j.seokgi > j.yooseung ∧ j.seokgi > j.shinyoung :=
by sorry

end NUMINAMATH_CALUDE_seokgi_jumped_furthest_l2325_232548


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_div_by_7_l2325_232564

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

/-- Checks if a number is a 3-digit base 9 number -/
def isThreeDigitBase9 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_div_by_7 :
  ∀ n : ℕ, isThreeDigitBase9 n → (base9ToBase10 n) % 7 = 0 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_div_by_7_l2325_232564


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2325_232542

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2325_232542


namespace NUMINAMATH_CALUDE_max_cards_purchasable_l2325_232588

theorem max_cards_purchasable (budget : ℚ) (card_cost : ℚ) (h1 : budget = 15/2) (h2 : card_cost = 17/20) :
  ⌊budget / card_cost⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_cards_purchasable_l2325_232588


namespace NUMINAMATH_CALUDE_simplify_fraction_l2325_232533

theorem simplify_fraction : 8 * (15 / 9) * (-45 / 40) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2325_232533


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l2325_232550

-- Define an equilateral triangle ABC with side length s
def equilateral_triangle (A B C : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Define the extended points B', C', and A'
def extended_points (A B C A' B' C' : ℝ × ℝ) (s : ℝ) : Prop :=
  dist B B' = 2*s ∧ dist C C' = 3*s ∧ dist A A' = 4*s

-- Define the area of a triangle given its vertices
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_ratio_theorem (A B C A' B' C' : ℝ × ℝ) (s : ℝ) :
  equilateral_triangle A B C s →
  extended_points A B C A' B' C' s →
  triangle_area A' B' C' / triangle_area A B C = 60 := by sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l2325_232550


namespace NUMINAMATH_CALUDE_sphere_area_is_14pi_l2325_232566

/-- A cuboid with vertices on a sphere's surface -/
structure CuboidOnSphere where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  vertices_on_sphere : Bool

/-- The surface area of a sphere containing a cuboid -/
def sphere_surface_area (c : CuboidOnSphere) : ℝ := sorry

/-- Theorem: The surface area of the sphere is 14π -/
theorem sphere_area_is_14pi (c : CuboidOnSphere) 
  (h1 : c.edge1 = 1) 
  (h2 : c.edge2 = 2) 
  (h3 : c.edge3 = 3) 
  (h4 : c.vertices_on_sphere = true) : 
  sphere_surface_area c = 14 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_area_is_14pi_l2325_232566


namespace NUMINAMATH_CALUDE_house_construction_delay_l2325_232515

/-- Represents the construction of a house -/
structure HouseConstruction where
  totalDays : ℕ
  initialMen : ℕ
  additionalMen : ℕ
  daysBeforeAddition : ℕ

/-- Calculates the total man-days of work for the house construction -/
def totalManDays (h : HouseConstruction) : ℕ :=
  h.initialMen * h.totalDays

/-- Calculates the days behind schedule without additional men -/
def daysBehindSchedule (h : HouseConstruction) : ℕ :=
  let totalWork := h.initialMen * h.daysBeforeAddition + (h.initialMen + h.additionalMen) * (h.totalDays - h.daysBeforeAddition)
  totalWork / h.initialMen - h.totalDays

/-- Theorem stating that the construction would be 80 days behind schedule without additional men -/
theorem house_construction_delay (h : HouseConstruction) 
  (h_total_days : h.totalDays = 100)
  (h_initial_men : h.initialMen = 100)
  (h_additional_men : h.additionalMen = 100)
  (h_days_before_addition : h.daysBeforeAddition = 20) :
  daysBehindSchedule h = 80 := by
  sorry

#eval daysBehindSchedule { totalDays := 100, initialMen := 100, additionalMen := 100, daysBeforeAddition := 20 }

end NUMINAMATH_CALUDE_house_construction_delay_l2325_232515


namespace NUMINAMATH_CALUDE_even_function_max_value_l2325_232530

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_max_value
  (f : ℝ → ℝ)
  (h_even : IsEven f)
  (h_max : ∀ x ∈ Set.Icc (-2) (-1), f x ≤ -2)
  (h_attains : ∃ x ∈ Set.Icc (-2) (-1), f x = -2) :
  (∀ x ∈ Set.Icc 1 2, f x ≤ -2) ∧ (∃ x ∈ Set.Icc 1 2, f x = -2) :=
sorry

end NUMINAMATH_CALUDE_even_function_max_value_l2325_232530


namespace NUMINAMATH_CALUDE_paper_fold_cut_ratio_l2325_232573

theorem paper_fold_cut_ratio : 
  let square_side : ℝ := 6
  let fold_ratio : ℝ := 1/3
  let cut_ratio : ℝ := 2/3
  let small_width : ℝ := square_side * fold_ratio
  let large_width : ℝ := square_side * (1 - fold_ratio) * (1 - cut_ratio)
  let small_perimeter : ℝ := 2 * (square_side + small_width)
  let large_perimeter : ℝ := 2 * (square_side + large_width)
  small_perimeter / large_perimeter = 12/17 := by
sorry

end NUMINAMATH_CALUDE_paper_fold_cut_ratio_l2325_232573


namespace NUMINAMATH_CALUDE_smallest_divisible_by_6_and_35_after_2015_l2325_232505

theorem smallest_divisible_by_6_and_35_after_2015 :
  ∀ n : ℕ, n > 2015 ∧ 6 ∣ n ∧ 35 ∣ n → n ≥ 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_6_and_35_after_2015_l2325_232505


namespace NUMINAMATH_CALUDE_triangle_side_length_l2325_232596

/-- In a triangle ABC, given angle A, angle B, and side AC, prove the length of AB --/
theorem triangle_side_length (A B C : Real) (angleA angleB : Real) (sideAC : Real) :
  -- Conditions
  angleA = 105 * Real.pi / 180 →
  angleB = 45 * Real.pi / 180 →
  sideAC = 2 →
  -- Triangle angle sum property
  angleA + angleB + C = Real.pi →
  -- Sine rule
  sideAC / Real.sin angleB = A / Real.sin C →
  -- Conclusion
  A = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2325_232596


namespace NUMINAMATH_CALUDE_sequence_determination_l2325_232570

/-- A sequence is determined if its terms are uniquely defined by given conditions -/
def is_determined (a : ℕ → ℝ) : Prop := sorry

/-- Arithmetic sequence with given S₁ and S₂ -/
def arithmetic_sequence (a : ℕ → ℝ) (S₁ S₂ : ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ S₁ = a 1 ∧ S₂ = a 1 + a 2

/-- Geometric sequence with given S₁ and S₂ -/
def geometric_sequence_S₁S₂ (a : ℕ → ℝ) (S₁ S₂ : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q^(n - 1) ∧ S₁ = a 1 ∧ S₂ = a 1 + a 1 * q

/-- Geometric sequence with given S₁ and S₃ -/
def geometric_sequence_S₁S₃ (a : ℕ → ℝ) (S₁ S₃ : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q^(n - 1) ∧ S₁ = a 1 ∧ S₃ = a 1 + a 1 * q + a 1 * q^2

/-- Sequence satisfying given recurrence relations -/
def recurrence_sequence (a : ℕ → ℝ) (x y c : ℝ) : Prop :=
  a 1 = c ∧ 
  (∀ n : ℕ, a (2*n + 2) = a (2*n) + x ∧ a (2*n + 1) = a (2*n - 1) + y)

theorem sequence_determination :
  ∀ a : ℕ → ℝ, ∀ S₁ S₂ S₃ x y c : ℝ,
  (is_determined a ↔ arithmetic_sequence a S₁ S₂) ∧
  (is_determined a ↔ geometric_sequence_S₁S₂ a S₁ S₂) ∧
  ¬(is_determined a ↔ geometric_sequence_S₁S₃ a S₁ S₃) ∧
  ¬(is_determined a ↔ recurrence_sequence a x y c) :=
sorry

end NUMINAMATH_CALUDE_sequence_determination_l2325_232570


namespace NUMINAMATH_CALUDE_equilateral_triangle_semi_regular_hexagon_l2325_232576

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a hexagon -/
structure Hexagon :=
  (vertices : Fin 6 → Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Divides each side of a triangle into three equal parts -/
def divideSides (t : Triangle) : Fin 6 → Point := sorry

/-- Forms a hexagon from the division points and opposite vertices -/
def formHexagon (t : Triangle) (divisionPoints : Fin 6 → Point) : Hexagon := sorry

/-- Checks if a hexagon is semi-regular -/
def isSemiRegular (h : Hexagon) : Prop := sorry

/-- Main theorem: The hexagon formed by dividing the sides of an equilateral triangle
    and connecting division points to opposite vertices is semi-regular -/
theorem equilateral_triangle_semi_regular_hexagon 
  (t : Triangle) (h : isEquilateral t) : 
  isSemiRegular (formHexagon t (divideSides t)) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_semi_regular_hexagon_l2325_232576


namespace NUMINAMATH_CALUDE_time_sum_after_advance_l2325_232527

/-- Represents time on a 12-hour digital clock -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  h_valid : hours < 12
  m_valid : minutes < 60
  s_valid : seconds < 60

/-- Calculates the time after a given number of hours, minutes, and seconds -/
def advanceTime (start : Time) (hours minutes seconds : Nat) : Time :=
  sorry

/-- Theorem: After 122 hours, 39 minutes, and 44 seconds from midnight, 
    the sum of resulting hours, minutes, and seconds is 85 -/
theorem time_sum_after_advance : 
  let midnight : Time := ⟨0, 0, 0, by simp, by simp, by simp⟩
  let result := advanceTime midnight 122 39 44
  result.hours + result.minutes + result.seconds = 85 := by
  sorry

end NUMINAMATH_CALUDE_time_sum_after_advance_l2325_232527


namespace NUMINAMATH_CALUDE_non_pine_trees_l2325_232577

theorem non_pine_trees (total : ℕ) (pine_percentage : ℚ) (non_pine : ℕ) : 
  total = 350 → pine_percentage = 70 / 100 → 
  non_pine = total - (pine_percentage * total).floor → non_pine = 105 :=
by sorry

end NUMINAMATH_CALUDE_non_pine_trees_l2325_232577


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2325_232592

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = min) ∧
    max = 16 ∧ min = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2325_232592


namespace NUMINAMATH_CALUDE_triangle_problem_l2325_232565

-- Define a triangle with interior angles a, b, and x
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the property of being an acute triangle
def isAcute (t : Triangle) : Prop :=
  t.a < 90 ∧ t.b < 90 ∧ t.x < 90

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 60)
  (h2 : t.b = 70)
  (h3 : t.a + t.b + t.x = 180) : 
  t.x = 50 ∧ isAcute t := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2325_232565


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l2325_232595

theorem fly_distance_from_ceiling (z : ℝ) : 
  (2:ℝ)^2 + 6^2 + z^2 = 11^2 → z = 9 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l2325_232595


namespace NUMINAMATH_CALUDE_sum_of_roots_l2325_232502

theorem sum_of_roots (p q r : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 72 = 0)
  (hq : 27*q^3 - 243*q^2 + 729*q - 972 = 0)
  (hr : 3*r = 9) : 
  p + q + r = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2325_232502
