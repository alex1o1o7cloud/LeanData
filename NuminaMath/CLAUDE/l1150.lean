import Mathlib

namespace uphill_divisible_by_25_count_l1150_115027

/-- A positive integer is uphill if every digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get! i < (n.digits 10).get! j

/-- A number is divisible by 25 if and only if it ends in 00 or 25. -/
def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

/-- The count of uphill integers divisible by 25 -/
def count_uphill_divisible_by_25 : ℕ := 3

theorem uphill_divisible_by_25_count :
  (∃ S : Finset ℕ, (∀ n ∈ S, is_uphill n ∧ divisible_by_25 n) ∧
                   (∀ n, is_uphill n → divisible_by_25 n → n ∈ S) ∧
                   S.card = count_uphill_divisible_by_25) :=
sorry

end uphill_divisible_by_25_count_l1150_115027


namespace sum_of_coefficients_l1150_115087

theorem sum_of_coefficients (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (4 * x - 2)^6 = b₆ * x^6 + b₅ * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 64 := by
sorry

end sum_of_coefficients_l1150_115087


namespace fraction_of_boys_reading_l1150_115083

theorem fraction_of_boys_reading (total_girls : ℕ) (total_boys : ℕ) 
  (fraction_girls_reading : ℚ) (not_reading : ℕ) :
  total_girls = 12 →
  total_boys = 10 →
  fraction_girls_reading = 5/6 →
  not_reading = 4 →
  (total_boys - (not_reading - (total_girls - (fraction_girls_reading * total_girls).num))) / total_boys = 4/5 := by
  sorry


end fraction_of_boys_reading_l1150_115083


namespace ashley_wedding_guests_l1150_115002

/-- Calculates the number of wedding guests based on champagne requirements. -/
def wedding_guests (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) (bottles_needed : ℕ) : ℕ :=
  (servings_per_bottle / glasses_per_guest) * bottles_needed

/-- Theorem stating that Ashley has 120 wedding guests. -/
theorem ashley_wedding_guests :
  wedding_guests 2 6 40 = 120 := by
  sorry

end ashley_wedding_guests_l1150_115002


namespace not_all_squares_congruent_l1150_115015

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem: It is false that all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (for completeness, not directly used in the proof)
def is_convex (s : Square) : Prop := true
def has_four_right_angles (s : Square) : Prop := true
def has_equal_diagonals (s : Square) : Prop := true
def similar (s1 s2 : Square) : Prop := true

end not_all_squares_congruent_l1150_115015


namespace square_of_binomial_theorem_l1150_115060

-- Define the expressions
def expr_A (x y : ℝ) := (x + y) * (x - y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (-x + y) * (-x - y)
def expr_D (x y : ℝ) := (-x + y) * (x + y)

-- Define what it means for an expression to be a square of a binomial
def is_square_of_binomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ → ℝ), ∀ x y, f x y = (g x y)^2

-- State the theorem
theorem square_of_binomial_theorem :
  is_square_of_binomial expr_A ∧
  ¬(is_square_of_binomial expr_B) ∧
  is_square_of_binomial expr_C ∧
  is_square_of_binomial expr_D :=
sorry

end square_of_binomial_theorem_l1150_115060


namespace tangent_line_to_circle_l1150_115021

/-- The value of k for which the line y = kx (k > 0) is tangent to the circle (x-√3)^2 + y^2 = 1 -/
theorem tangent_line_to_circle (k : ℝ) : 
  k > 0 ∧ 
  (∃ (x y : ℝ), y = k * x ∧ (x - Real.sqrt 3)^2 + y^2 = 1) ∧
  (∀ (x y : ℝ), y = k * x → (x - Real.sqrt 3)^2 + y^2 ≥ 1) →
  k = Real.sqrt 2 / 2 := by
sorry

end tangent_line_to_circle_l1150_115021


namespace line_satisfies_conditions_l1150_115051

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the line
def line (x : ℝ) : ℝ := 6*x - 4

-- Theorem statement
theorem line_satisfies_conditions :
  -- Condition 1: The line passes through (2, 8)
  (line 2 = 8) ∧
  -- Condition 2: There exists a k where x = k intersects both curves 4 units apart
  (∃ k : ℝ, |parabola k - line k| = 4) ∧
  -- Condition 3: The y-intercept is not 0
  (line 0 ≠ 0) :=
by sorry

end line_satisfies_conditions_l1150_115051


namespace rectangular_prism_area_volume_relation_l1150_115026

theorem rectangular_prism_area_volume_relation 
  (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by sorry

end rectangular_prism_area_volume_relation_l1150_115026


namespace point_on_y_axis_l1150_115012

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
def lies_on_y_axis (x y : ℝ) : Prop := x = 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := 6 - 2*m

/-- The y-coordinate of point P -/
def y_coord (m : ℝ) : ℝ := 4 - m

/-- Theorem: If the point P(6-2m, 4-m) lies on the y-axis, then m = 3 -/
theorem point_on_y_axis (m : ℝ) : lies_on_y_axis (x_coord m) (y_coord m) → m = 3 := by
  sorry

end point_on_y_axis_l1150_115012


namespace calculate_english_marks_l1150_115000

/-- Proves that given a student's marks in 4 subjects and an average across 5 subjects,
    we can determine the marks in the fifth subject. -/
theorem calculate_english_marks (math physics chem bio : ℕ) (average : ℚ)
    (h_math : math = 65)
    (h_physics : physics = 82)
    (h_chem : chem = 67)
    (h_bio : bio = 85)
    (h_average : average = 79)
    : ∃ english : ℕ, english = 96 ∧ 
      (english + math + physics + chem + bio : ℚ) / 5 = average :=
by
  sorry

end calculate_english_marks_l1150_115000


namespace triangle_circle_area_l1150_115043

theorem triangle_circle_area (a : ℝ) (h : a > 0) : 
  let angle1 : ℝ := 45 * π / 180
  let angle2 : ℝ := 15 * π / 180
  let angle3 : ℝ := π - angle1 - angle2
  let height : ℝ := a * (Real.sqrt 3 - 1) / (2 * Real.sqrt 3)
  let circle_area : ℝ := π * height^2
  circle_area / 3 = π * a^2 * (2 - Real.sqrt 3) / 18 := by
sorry

end triangle_circle_area_l1150_115043


namespace total_pay_calculation_l1150_115044

def first_job_pay : ℕ := 2125
def pay_difference : ℕ := 375

def second_job_pay : ℕ := first_job_pay - pay_difference

def total_pay : ℕ := first_job_pay + second_job_pay

theorem total_pay_calculation : total_pay = 3875 := by
  sorry

end total_pay_calculation_l1150_115044


namespace intersection_implies_sum_l1150_115054

-- Define the functions
def f (x a b : ℝ) : ℝ := -2 * abs (x - a) + b
def g (x c d : ℝ) : ℝ := 2 * abs (x - c) + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) : 
  (f 1 a b = g 1 c d) ∧ (f 7 a b = g 7 c d) → a + c = 8 := by
  sorry


end intersection_implies_sum_l1150_115054


namespace container_height_l1150_115009

/-- The height of a cylindrical container A, given specific conditions --/
theorem container_height (r_A r_B : ℝ) (h : ℝ → ℝ) :
  r_A = 2 →
  r_B = 3 →
  (∀ x, h x = (2/3 * x - 6)) →
  (π * r_A^2 * x = π * r_B^2 * h x) →
  x = 27 :=
by sorry

end container_height_l1150_115009


namespace min_triangle_area_l1150_115035

/-- Triangle DEF with vertices D(0,0) and E(24,10), and F having integer coordinates -/
structure Triangle where
  F : ℤ × ℤ

/-- Area of triangle DEF given coordinates of F -/
def triangleArea (t : Triangle) : ℚ :=
  let (x, y) := t.F
  (1 : ℚ) / 2 * |10 * x - 24 * y|

/-- The minimum non-zero area of triangle DEF is 5 -/
theorem min_triangle_area :
  ∃ (t : Triangle), triangleArea t > 0 ∧
  ∀ (t' : Triangle), triangleArea t' > 0 → triangleArea t ≤ triangleArea t' :=
sorry

end min_triangle_area_l1150_115035


namespace even_function_implies_b_zero_l1150_115019

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x(x+b) -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ x * (x + b)

/-- If f(x) = x(x+b) is an even function, then b = 0 -/
theorem even_function_implies_b_zero :
  ∀ b : ℝ, IsEven (f b) → b = 0 := by
  sorry

end even_function_implies_b_zero_l1150_115019


namespace solution_first_equation_solutions_second_equation_l1150_115086

-- First equation
theorem solution_first_equation (x : ℝ) :
  27 * (x + 1)^3 = -64 ↔ x = -7/3 := by sorry

-- Second equation
theorem solutions_second_equation (x : ℝ) :
  (x + 1)^2 = 25 ↔ x = 4 ∨ x = -6 := by sorry

end solution_first_equation_solutions_second_equation_l1150_115086


namespace min_sum_products_l1150_115004

theorem min_sum_products (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (h : x + y + z = 3 * x * y * z) : 
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 * a * b * c → 
    x * y + y * z + x * z ≤ a * b + b * c + a * c :=
by sorry

end min_sum_products_l1150_115004


namespace monic_quartic_value_l1150_115025

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value (f : ℝ → ℝ) :
  is_monic_quartic f →
  f (-2) = -4 →
  f 1 = -1 →
  f (-3) = -9 →
  f 5 = -25 →
  f 2 = -64 := by
  sorry

end monic_quartic_value_l1150_115025


namespace journey_time_proof_l1150_115049

/-- Proves that the total journey time is 5 hours given the specified conditions -/
theorem journey_time_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (time1 : ℝ) :
  total_distance = 240 ∧ 
  speed1 = 40 ∧ 
  speed2 = 60 ∧ 
  time1 = 3 →
  speed1 * time1 + (total_distance - speed1 * time1) / speed2 + time1 = 5 := by
  sorry

end journey_time_proof_l1150_115049


namespace sin_180_degrees_l1150_115075

theorem sin_180_degrees : Real.sin (π) = 0 := by
  sorry

end sin_180_degrees_l1150_115075


namespace shaded_area_proof_shaded_area_is_sqrt_288_l1150_115096

theorem shaded_area_proof (small_square_area : ℝ) 
  (h1 : small_square_area = 3) 
  (num_small_squares : ℕ) 
  (h2 : num_small_squares = 9) : ℝ :=
  let small_square_side := Real.sqrt small_square_area
  let small_square_diagonal := small_square_side * Real.sqrt 2
  let large_square_side := 2 * small_square_diagonal + small_square_side
  let large_square_area := large_square_side ^ 2
  let total_small_squares_area := num_small_squares * small_square_area
  let shaded_area := large_square_area - total_small_squares_area
  Real.sqrt 288

theorem shaded_area_is_sqrt_288 : shaded_area_proof 3 rfl 9 rfl = Real.sqrt 288 := by sorry

end shaded_area_proof_shaded_area_is_sqrt_288_l1150_115096


namespace exact_division_condition_l1150_115013

-- Define the polynomial x^4 + 1
def f (x : ℂ) : ℂ := x^4 + 1

-- Define the trinomial x^2 + px + q
def g (p q x : ℂ) : ℂ := x^2 + p*x + q

-- Define the condition for exact division
def is_exact_division (p q : ℂ) : Prop :=
  ∃ (h : ℂ → ℂ), ∀ x, f x = (g p q x) * (h x)

-- State the theorem
theorem exact_division_condition :
  ∀ p q : ℂ, is_exact_division p q ↔ 
    ((p = 0 ∧ q = Complex.I) ∨ 
     (p = 0 ∧ q = -Complex.I) ∨ 
     (p = Real.sqrt 2 ∧ q = 1) ∨ 
     (p = -Real.sqrt 2 ∧ q = 1)) :=
by sorry

end exact_division_condition_l1150_115013


namespace PL_length_l1150_115057

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (topLeft : Point) (bottomRight : Point)

/-- The square WXYZ -/
def square : Rectangle :=
  { topLeft := { x := 0, y := 2 },
    bottomRight := { x := 2, y := 0 } }

/-- The length of PL -/
def PL : ℝ := 1

/-- States that two rectangles are congruent -/
def congruentRectangles (r1 r2 : Rectangle) : Prop :=
  (r1.bottomRight.x - r1.topLeft.x) * (r1.topLeft.y - r1.bottomRight.y) =
  (r2.bottomRight.x - r2.topLeft.x) * (r2.topLeft.y - r2.bottomRight.y)

/-- The theorem to be proved -/
theorem PL_length :
  ∀ (LMNO PQRS : Rectangle),
    congruentRectangles LMNO PQRS →
    PL = 1 :=
by
  sorry


end PL_length_l1150_115057


namespace rectangular_prism_diagonal_l1150_115095

/-- Given a rectangular prism with dimensions a, b, and c, if the total surface area
    is 11 and the sum of all edge lengths is 24, then the length of the body diagonal is 5. -/
theorem rectangular_prism_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)  -- total surface area
  (h2 : 4 * (a + b + c) = 24) :            -- sum of all edge lengths
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end rectangular_prism_diagonal_l1150_115095


namespace bee_count_l1150_115062

theorem bee_count (flowers : ℕ) (bee_difference : ℕ) : 
  flowers = 5 → bee_difference = 2 → flowers - bee_difference = 3 :=
by
  sorry

end bee_count_l1150_115062


namespace cos_2alpha_value_l1150_115038

theorem cos_2alpha_value (α : ℝ) (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) :
  Real.cos (2 * α) = -4/5 := by
  sorry

end cos_2alpha_value_l1150_115038


namespace warehouse_paint_area_l1150_115080

/-- Calculates the area to be painted in a rectangular warehouse with a door. -/
def areaToBePainted (length width height doorWidth doorHeight : ℝ) : ℝ :=
  2 * (length * height + width * height) - (doorWidth * doorHeight)

/-- Theorem stating the area to be painted for the given warehouse dimensions. -/
theorem warehouse_paint_area :
  areaToBePainted 8 6 3.5 1 2 = 96 := by
  sorry

end warehouse_paint_area_l1150_115080


namespace rectangle_perimeter_relation_l1150_115022

/-- Given a figure divided into equal squares, this theorem proves the relationship
    between the perimeters of two rectangles formed by these squares. -/
theorem rectangle_perimeter_relation (square_side : ℝ) 
  (h1 : square_side > 0)
  (h2 : 3 * square_side * 2 + 2 * square_side = 112) : 
  4 * square_side * 2 + 2 * square_side = 140 := by
  sorry

#check rectangle_perimeter_relation

end rectangle_perimeter_relation_l1150_115022


namespace dog_movement_area_calculation_l1150_115063

/-- Represents the dimensions and constraints of a dog tied to a square doghouse --/
structure DogHouseSetup where
  side_length : ℝ
  tie_point_distance : ℝ
  chain_length : ℝ

/-- Calculates the area in which the dog can move --/
def dog_movement_area (setup : DogHouseSetup) : ℝ :=
  sorry

/-- Theorem stating the area in which the dog can move for the given setup --/
theorem dog_movement_area_calculation (ε : ℝ) (h_ε : ε > 0) :
  ∃ (setup : DogHouseSetup),
    setup.side_length = 1.2 ∧
    setup.tie_point_distance = 0.3 ∧
    setup.chain_length = 3 ∧
    |dog_movement_area setup - 23.693| < ε :=
  sorry

end dog_movement_area_calculation_l1150_115063


namespace QY_eq_10_l1150_115032

/-- Circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Point outside the circle -/
def Q : ℝ × ℝ := sorry

/-- Circle C -/
def C : Circle := sorry

/-- Points on the circle -/
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := sorry
def Z : ℝ × ℝ := sorry

/-- Distances -/
def QX : ℝ := sorry
def QY : ℝ := sorry
def QZ : ℝ := sorry

/-- Q is outside C -/
axiom h_Q_outside : Q ∉ {p | (p.1 - C.O.1)^2 + (p.2 - C.O.2)^2 ≤ C.r^2}

/-- QZ is tangent to C at Z -/
axiom h_QZ_tangent : (Z.1 - C.O.1)^2 + (Z.2 - C.O.2)^2 = C.r^2 ∧
  ((Z.1 - Q.1) * (Z.1 - C.O.1) + (Z.2 - Q.2) * (Z.2 - C.O.2) = 0)

/-- X and Y are on C -/
axiom h_X_on_C : (X.1 - C.O.1)^2 + (X.2 - C.O.2)^2 = C.r^2
axiom h_Y_on_C : (Y.1 - C.O.1)^2 + (Y.2 - C.O.2)^2 = C.r^2

/-- QX < QY -/
axiom h_QX_lt_QY : QX < QY

/-- QX = 5 -/
axiom h_QX_eq_5 : QX = 5

/-- QZ = 2(QY - QX) -/
axiom h_QZ_eq : QZ = 2 * (QY - QX)

/-- Power of a Point theorem -/
axiom power_of_point : QX * QY = QZ^2

theorem QY_eq_10 : QY = 10 := by sorry

end QY_eq_10_l1150_115032


namespace samantha_born_1986_l1150_115003

/-- The year of the first Math Kangaroo contest -/
def first_math_kangaroo_year : ℕ := 1991

/-- The age of Samantha when she took the tenth Math Kangaroo -/
def samantha_age_tenth_kangaroo : ℕ := 14

/-- Function to calculate the year of the nth Math Kangaroo contest -/
def math_kangaroo_year (n : ℕ) : ℕ := first_math_kangaroo_year + n - 1

/-- Samantha's birth year -/
def samantha_birth_year : ℕ := math_kangaroo_year 10 - samantha_age_tenth_kangaroo

theorem samantha_born_1986 : samantha_birth_year = 1986 := by
  sorry

end samantha_born_1986_l1150_115003


namespace gift_contribution_total_l1150_115028

/-- Proves that the total contribution is $20 given the specified conditions -/
theorem gift_contribution_total (n : ℕ) (min_contribution max_contribution : ℝ) :
  n = 10 →
  min_contribution = 1 →
  max_contribution = 11 →
  (n - 1 : ℝ) * min_contribution + max_contribution = 20 :=
by sorry

end gift_contribution_total_l1150_115028


namespace cookie_box_weight_limit_l1150_115090

/-- The weight limit of a cookie box in pounds, given the weight of each cookie and the number of cookies it can hold. -/
theorem cookie_box_weight_limit (cookie_weight : ℚ) (box_capacity : ℕ) : 
  cookie_weight = 2 → box_capacity = 320 → (cookie_weight * box_capacity) / 16 = 40 := by
  sorry

end cookie_box_weight_limit_l1150_115090


namespace bikes_total_price_l1150_115098

/-- The total price of Marion's and Stephanie's bikes -/
def total_price (marion_price stephanie_price : ℕ) : ℕ :=
  marion_price + stephanie_price

/-- Theorem stating the total price of Marion's and Stephanie's bikes -/
theorem bikes_total_price :
  ∃ (marion_price stephanie_price : ℕ),
    marion_price = 356 ∧
    stephanie_price = 2 * marion_price ∧
    total_price marion_price stephanie_price = 1068 :=
by
  sorry


end bikes_total_price_l1150_115098


namespace enclosing_triangle_sides_l1150_115091

/-- An isosceles triangle enclosing a circle -/
structure EnclosingTriangle where
  /-- Radius of the enclosed circle -/
  r : ℝ
  /-- Acute angle at the base of the isosceles triangle in radians -/
  θ : ℝ
  /-- Length of the equal sides of the isosceles triangle -/
  a : ℝ
  /-- Length of the base of the isosceles triangle -/
  b : ℝ

/-- The theorem stating the side lengths of the enclosing isosceles triangle -/
theorem enclosing_triangle_sides (t : EnclosingTriangle) 
  (h_r : t.r = 3)
  (h_θ : t.θ = π/6) -- 30° in radians
  : t.a = 4 * Real.sqrt 3 + 6 ∧ t.b = 6 * Real.sqrt 3 + 12 := by
  sorry


end enclosing_triangle_sides_l1150_115091


namespace smallest_AC_solution_exists_l1150_115024

-- Define the triangle and its properties
def Triangle (AC CD : ℕ) : Prop :=
  ∃ (AB BD : ℕ),
    AB = AC ∧  -- AB = AC
    BD * BD = 68 ∧  -- BD² = 68
    AC = (CD * CD + 68) / (2 * CD) ∧  -- Derived from the Pythagorean theorem
    CD < 10 ∧  -- CD is less than 10
    Nat.Prime CD  -- CD is prime

-- State the theorem
theorem smallest_AC :
  ∀ AC CD, Triangle AC CD → AC ≥ 18 :=
by sorry

-- State the existence of a solution
theorem solution_exists :
  ∃ AC CD, Triangle AC CD ∧ AC = 18 :=
by sorry

end smallest_AC_solution_exists_l1150_115024


namespace ceiling_sum_of_square_roots_l1150_115058

theorem ceiling_sum_of_square_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_of_square_roots_l1150_115058


namespace problem_1_problem_2_problem_3_problem_4_l1150_115065

-- Problem 1
theorem problem_1 : -2.4 + 3.5 - 4.6 + 3.5 = 0 := by sorry

-- Problem 2
theorem problem_2 : (-40) - (-28) - (-19) + (-24) = -17 := by sorry

-- Problem 3
theorem problem_3 : (-3 : ℚ) * (5/6 : ℚ) * (-4/5 : ℚ) * (-1/4 : ℚ) = -1/2 := by sorry

-- Problem 4
theorem problem_4 : (-5/7 : ℚ) * (-4/3 : ℚ) / (-15/7 : ℚ) = -4/9 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1150_115065


namespace function_passes_through_point_l1150_115084

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 2
  f 3 = 3 := by sorry

end function_passes_through_point_l1150_115084


namespace count_sequences_with_at_least_three_heads_l1150_115071

/-- The number of distinct sequences of 10 coin flips containing at least 3 heads -/
def sequences_with_at_least_three_heads : ℕ :=
  2^10 - (Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2)

/-- Theorem stating that the number of sequences with at least 3 heads is 968 -/
theorem count_sequences_with_at_least_three_heads :
  sequences_with_at_least_three_heads = 968 := by
  sorry

end count_sequences_with_at_least_three_heads_l1150_115071


namespace equipment_cost_proof_l1150_115016

/-- The number of players on the team -/
def num_players : ℕ := 16

/-- The cost of a jersey in dollars -/
def jersey_cost : ℚ := 25

/-- The cost of shorts in dollars -/
def shorts_cost : ℚ := 15.20

/-- The cost of socks in dollars -/
def socks_cost : ℚ := 6.80

/-- The total cost of equipment for all players -/
def total_cost : ℚ := num_players * (jersey_cost + shorts_cost + socks_cost)

theorem equipment_cost_proof : total_cost = 752 := by
  sorry

end equipment_cost_proof_l1150_115016


namespace hotel_flat_fee_calculation_l1150_115073

/-- A hotel charges a flat fee for the first night and a fixed amount for subsequent nights. -/
structure HotelPricing where
  flatFee : ℝ
  subsequentNightFee : ℝ

/-- Calculate the total cost for a given number of nights -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.subsequentNightFee * (nights - 1)

theorem hotel_flat_fee_calculation (pricing : HotelPricing) :
  totalCost pricing 4 = 185 ∧ totalCost pricing 8 = 350 → pricing.flatFee = 61.25 := by
  sorry

end hotel_flat_fee_calculation_l1150_115073


namespace children_percentage_l1150_115001

def total_passengers : ℕ := 60
def adult_passengers : ℕ := 45

theorem children_percentage : 
  (total_passengers - adult_passengers) / total_passengers * 100 = 25 := by
  sorry

end children_percentage_l1150_115001


namespace recommended_sleep_hours_l1150_115069

theorem recommended_sleep_hours (total_sleep : ℝ) (short_sleep : ℝ) (short_days : ℕ) 
  (normal_days : ℕ) (normal_sleep_percentage : ℝ) 
  (h1 : total_sleep = 30)
  (h2 : short_sleep = 3)
  (h3 : short_days = 2)
  (h4 : normal_days = 5)
  (h5 : normal_sleep_percentage = 0.6)
  (h6 : total_sleep = short_sleep * short_days + normal_sleep_percentage * normal_days * recommended_sleep) :
  recommended_sleep = 8 := by
  sorry

end recommended_sleep_hours_l1150_115069


namespace expand_expression_l1150_115097

theorem expand_expression (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x^2 + 10 * x - 40 := by
  sorry

end expand_expression_l1150_115097


namespace race_result_l1150_115039

-- Define the set of runners
inductive Runner : Type
| P : Runner
| Q : Runner
| R : Runner
| S : Runner
| T : Runner

-- Define the relation "beats" between runners
def beats : Runner → Runner → Prop := sorry

-- Define the relation "finishes_before" between runners
def finishes_before : Runner → Runner → Prop := sorry

-- Define what it means for a runner to finish third
def finishes_third : Runner → Prop := sorry

-- State the theorem
theorem race_result : 
  (beats Runner.P Runner.Q) →
  (beats Runner.P Runner.R) →
  (beats Runner.Q Runner.S) →
  (finishes_before Runner.P Runner.T) →
  (finishes_before Runner.T Runner.Q) →
  (¬ finishes_third Runner.P ∧ ¬ finishes_third Runner.S) ∧
  (∃ (x : Runner), x ≠ Runner.P ∧ x ≠ Runner.S ∧ finishes_third x) :=
by sorry

end race_result_l1150_115039


namespace calculation_proof_l1150_115040

theorem calculation_proof :
  (1 : ℝ) = (1/3)^0 ∧
  3 = Real.sqrt 27 ∧
  3 = |-3| ∧
  1 = Real.tan (π/4) →
  (1/3)^0 + Real.sqrt 27 - |-3| + Real.tan (π/4) = 1 + 3 * Real.sqrt 3 - 2 ∧
  ∀ x : ℝ, (x + 2)^2 - 2*(x - 1) = x^2 + 2*x + 6 :=
by sorry

end calculation_proof_l1150_115040


namespace subtraction_division_fractions_l1150_115050

theorem subtraction_division_fractions : ((3 / 4 : ℚ) - (5 / 8 : ℚ)) / 2 = (1 / 16 : ℚ) := by
  sorry

end subtraction_division_fractions_l1150_115050


namespace soccer_league_games_l1150_115042

/-- The number of games played in a soccer league with given conditions -/
def total_games (n : ℕ) (promo_per_team : ℕ) : ℕ :=
  (n * (n - 1) + n * promo_per_team) / 2

/-- Theorem: In a soccer league with 15 teams, where each team plays every other team twice 
    and has 2 additional promotional games, the total number of games played is 120 -/
theorem soccer_league_games : total_games 15 2 = 120 := by
  sorry

end soccer_league_games_l1150_115042


namespace ellipse_focal_property_l1150_115018

-- Define the ellipse
def ellipse (x y b : ℝ) : Prop := x^2 / 4 + y^2 / b^2 = 1

-- Define the constraint on b
def b_constraint (b : ℝ) : Prop := 0 < b ∧ b < 2

-- Define the maximum value of |BF_2| + |AF_2|
def max_focal_sum (b : ℝ) : Prop := ∃ (A B F_2 : ℝ × ℝ), 
  ∀ (P Q : ℝ × ℝ), dist P F_2 + dist Q F_2 ≤ dist A F_2 + dist B F_2 ∧ 
  dist A F_2 + dist B F_2 = 5

-- Theorem statement
theorem ellipse_focal_property (b : ℝ) :
  b_constraint b →
  (∀ x y, ellipse x y b → max_focal_sum b) →
  b = Real.sqrt 3 := by sorry

end ellipse_focal_property_l1150_115018


namespace sheila_hourly_wage_l1150_115047

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := 3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $11 --/
theorem sheila_hourly_wage :
  let sheila_schedule := WorkSchedule.mk 8 6 396
  hourly_wage sheila_schedule = 11 := by sorry

end sheila_hourly_wage_l1150_115047


namespace log_cutting_ratio_l1150_115055

/-- Given a log of length 20 feet where each linear foot weighs 150 pounds,
    if the log is cut into two equal pieces each weighing 1500 pounds,
    then the ratio of the length of each cut piece to the length of the original log is 1/2. -/
theorem log_cutting_ratio :
  ∀ (original_length cut_length : ℝ) (weight_per_foot cut_weight : ℝ),
    original_length = 20 →
    weight_per_foot = 150 →
    cut_weight = 1500 →
    cut_length * weight_per_foot = cut_weight →
    cut_length / original_length = 1 / 2 := by
  sorry

end log_cutting_ratio_l1150_115055


namespace consecutive_circle_selections_l1150_115074

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement :=
  (total_circles : Nat)
  (long_side_rows : Nat)
  (perpendicular_rows : Nat)

/-- Calculates the number of ways to choose three consecutive circles along the long side -/
def long_side_selections (arr : CircleArrangement) : Nat :=
  (arr.long_side_rows * (arr.long_side_rows + 1)) / 2

/-- Calculates the number of ways to choose three consecutive circles along one perpendicular direction -/
def perpendicular_selections (arr : CircleArrangement) : Nat :=
  (3 * arr.perpendicular_rows + (arr.perpendicular_rows * (arr.perpendicular_rows - 1)) / 2)

/-- The main theorem stating the total number of ways to choose three consecutive circles -/
theorem consecutive_circle_selections (arr : CircleArrangement) 
  (h1 : arr.total_circles = 33)
  (h2 : arr.long_side_rows = 6)
  (h3 : arr.perpendicular_rows = 6) :
  long_side_selections arr + 2 * perpendicular_selections arr = 57 := by
  sorry


end consecutive_circle_selections_l1150_115074


namespace sets_intersection_union_l1150_115081

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sets_intersection_union (a b : ℝ) : 
  (A ∪ B a b = Set.univ) ∧ (A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) → a + b = -7 := by
  sorry

end sets_intersection_union_l1150_115081


namespace weight_difference_proof_l1150_115093

/-- Proves that the difference between the average weight of two departing students
    and Joe's weight is -6.5 kg, given the conditions of the original problem. -/
theorem weight_difference_proof (n : ℕ) (x : ℝ) : 
  -- Joe's weight
  let joe_weight : ℝ := 43
  -- Initial average weight
  let initial_avg : ℝ := 30
  -- New average weight after Joe joins
  let new_avg : ℝ := 31
  -- Number of students in original group
  n = (joe_weight - initial_avg) / (new_avg - initial_avg)
  -- Average weight of two departing students
  → x = (new_avg * (n + 1) - initial_avg * (n - 1)) / 2
  -- Difference between average weight of departing students and Joe's weight
  → x - joe_weight = -6.5 := by
  sorry

end weight_difference_proof_l1150_115093


namespace garden_area_calculation_l1150_115030

def garden_length : ℝ := 18
def garden_width : ℝ := 15
def cutout1_side : ℝ := 4
def cutout2_side : ℝ := 2

theorem garden_area_calculation :
  garden_length * garden_width - (cutout1_side * cutout1_side + cutout2_side * cutout2_side) = 250 := by
  sorry

end garden_area_calculation_l1150_115030


namespace regression_line_correct_l1150_115031

def points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

def regression_line (points : List (ℝ × ℝ)) : ℝ → ℝ := 
  fun x => x + 1

theorem regression_line_correct : 
  regression_line points = fun x => x + 1 := by sorry

end regression_line_correct_l1150_115031


namespace fourth_person_height_l1150_115006

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ - h₁ = 2 →                 -- difference between 1st and 2nd
  h₃ - h₂ = 2 →                 -- difference between 2nd and 3rd
  h₄ - h₃ = 6 →                 -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76  -- average height
  → h₄ = 82 :=                  -- height of 4th person
by sorry

end fourth_person_height_l1150_115006


namespace distance_between_points_l1150_115082

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-3, -4)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end distance_between_points_l1150_115082


namespace third_number_proof_l1150_115053

/-- The smallest number greater than 57 that leaves the same remainder as 25 and 57 when divided by 16 -/
def third_number : ℕ := 73

/-- The common divisor -/
def common_divisor : ℕ := 16

theorem third_number_proof :
  (third_number % common_divisor = 25 % common_divisor) ∧
  (third_number % common_divisor = 57 % common_divisor) ∧
  (third_number > 57) ∧
  (∀ n : ℕ, n > 57 ∧ n < third_number →
    (n % common_divisor ≠ 25 % common_divisor ∨
     n % common_divisor ≠ 57 % common_divisor)) :=
by sorry

end third_number_proof_l1150_115053


namespace base10_216_equals_base9_260_l1150_115046

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 9 to a natural number in base 10 --/
def fromBase9 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if all digits in a list are less than 9 --/
def validBase9Digits (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < 9

theorem base10_216_equals_base9_260 :
  let base9Digits := [2, 6, 0]
  validBase9Digits base9Digits ∧ fromBase9 base9Digits = 216 :=
by
  sorry

end base10_216_equals_base9_260_l1150_115046


namespace sequence_reappearance_l1150_115052

def letter_cycle_length : ℕ := 7
def digit_cycle_length : ℕ := 4

theorem sequence_reappearance :
  Nat.lcm letter_cycle_length digit_cycle_length = 28 := by
  sorry

#check sequence_reappearance

end sequence_reappearance_l1150_115052


namespace area_between_circles_l1150_115085

/-- The area between two concentric circles -/
theorem area_between_circles (R r : ℝ) (h1 : R = 10) (h2 : r = 4) :
  (π * R^2) - (π * r^2) = 84 * π := by
  sorry

end area_between_circles_l1150_115085


namespace school_population_theorem_l1150_115008

theorem school_population_theorem :
  ∀ (boys girls : ℕ),
  boys + girls = 300 →
  girls = (boys * 100) / 300 →
  boys = 225 := by
sorry

end school_population_theorem_l1150_115008


namespace equal_angle_implies_equal_side_l1150_115070

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents the orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

/-- Reflects a point with respect to a line segment -/
def reflect (p : Point) (a : Point) (b : Point) : Point :=
  sorry

/-- Checks if two triangles have an equal angle -/
def have_equal_angle (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if two triangles have an equal side -/
def have_equal_side (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if a triangle is acute -/
def is_acute (t : Triangle) : Prop :=
  sorry

theorem equal_angle_implies_equal_side 
  (ABC : Triangle) 
  (h_acute : is_acute ABC) 
  (H : Point) 
  (h_ortho : H = orthocenter ABC) 
  (A' B' C' : Point) 
  (h_A' : A' = reflect H ABC.B ABC.C) 
  (h_B' : B' = reflect H ABC.C ABC.A) 
  (h_C' : C' = reflect H ABC.A ABC.B) 
  (A'B'C' : Triangle) 
  (h_A'B'C' : A'B'C' = Triangle.mk A' B' C') 
  (h_equal_angle : have_equal_angle ABC A'B'C') :
  have_equal_side ABC A'B'C' :=
sorry

end equal_angle_implies_equal_side_l1150_115070


namespace selection_methods_count_l1150_115034

def total_volunteers : ℕ := 8
def boys : ℕ := 5
def girls : ℕ := 3
def selection_size : ℕ := 3

theorem selection_methods_count : 
  (Nat.choose boys 2 * Nat.choose girls 1) + (Nat.choose boys 1 * Nat.choose girls 2) = 45 := by
  sorry

end selection_methods_count_l1150_115034


namespace cafeteria_pies_l1150_115017

/-- Given a cafeteria with initial apples, apples handed out, and apples per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (apples_handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - apples_handed_out) / apples_per_pie

/-- Theorem stating that with 96 initial apples, 42 apples handed out,
    and 6 apples per pie, the cafeteria can make 9 pies. -/
theorem cafeteria_pies :
  calculate_pies 96 42 6 = 9 := by
  sorry

end cafeteria_pies_l1150_115017


namespace percentage_decrease_l1150_115023

theorem percentage_decrease (initial : ℝ) (increase_percent : ℝ) (final : ℝ) :
  initial = 1500 →
  increase_percent = 20 →
  final = 1080 →
  ∃ (decrease_percent : ℝ),
    final = (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) ∧
    decrease_percent = 40 := by
  sorry

end percentage_decrease_l1150_115023


namespace remainder_equality_l1150_115094

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_equality : 
  (sum_factorials 20) % 21 = (sum_factorials 4) % 21 := by
sorry

end remainder_equality_l1150_115094


namespace prob_three_faces_is_8_27_l1150_115029

/-- Represents a small cube sawed from a larger painted cube -/
structure SmallCube :=
  (painted_faces : Fin 4)

/-- The set of all small cubes obtained from sawing a painted cube -/
def all_cubes : Finset SmallCube := sorry

/-- The set of small cubes with exactly three painted faces -/
def three_face_cubes : Finset SmallCube := sorry

/-- The probability of selecting a small cube with three painted faces -/
def prob_three_faces : ℚ := (three_face_cubes.card : ℚ) / (all_cubes.card : ℚ)

theorem prob_three_faces_is_8_27 : prob_three_faces = 8 / 27 := by sorry

end prob_three_faces_is_8_27_l1150_115029


namespace inequality_system_solution_l1150_115036

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (2 - x < 0) ∧ (-2 * x < 6)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end inequality_system_solution_l1150_115036


namespace x_values_theorem_l1150_115064

theorem x_values_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 :=
by sorry

end x_values_theorem_l1150_115064


namespace x_value_l1150_115092

theorem x_value (x : ℝ) (h1 : x^2 - 2*x = 0) (h2 : x ≠ 0) : x = 2 := by
  sorry

end x_value_l1150_115092


namespace negation_of_p_l1150_115020

open Real

def p : Prop := ∃ x : ℚ, 2^(x : ℝ) - log x < 2

theorem negation_of_p : ¬p ↔ ∀ x : ℚ, 2^(x : ℝ) - log x ≥ 2 := by sorry

end negation_of_p_l1150_115020


namespace raffle_ticket_sales_l1150_115010

theorem raffle_ticket_sales (total_avg : ℝ) (male_avg : ℝ) (female_avg : ℝ) :
  total_avg = 66 →
  male_avg = 58 →
  (1 : ℝ) * male_avg + 2 * female_avg = 3 * total_avg →
  female_avg = 70 := by
  sorry

end raffle_ticket_sales_l1150_115010


namespace largest_integral_x_l1150_115056

theorem largest_integral_x : ∃ (x : ℤ), 
  (∀ (y : ℤ), (1 : ℚ) / 3 < (y : ℚ) / 5 ∧ (y : ℚ) / 5 < 5 / 8 → y ≤ x) ∧
  (1 : ℚ) / 3 < (x : ℚ) / 5 ∧ (x : ℚ) / 5 < 5 / 8 :=
by
  sorry

end largest_integral_x_l1150_115056


namespace geometric_sequence_ratio_l1150_115079

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  property1 : a 5 * a 11 = 3
  property2 : a 3 + a 13 = 4

/-- The theorem stating the possible values of a_15 / a_5 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 15 / seq.a 5 = 1/3 ∨ seq.a 15 / seq.a 5 = 3 := by
  sorry

end geometric_sequence_ratio_l1150_115079


namespace watch_gain_percentage_l1150_115005

/-- Calculates the gain percentage when a watch is sold at a higher price -/
theorem watch_gain_percentage (cost_price : ℝ) (loss_percentage : ℝ) (price_increase : ℝ) : 
  cost_price = 1400 →
  loss_percentage = 10 →
  price_increase = 196 →
  let initial_selling_price := cost_price * (1 - loss_percentage / 100)
  let new_selling_price := initial_selling_price + price_increase
  let gain_amount := new_selling_price - cost_price
  let gain_percentage := (gain_amount / cost_price) * 100
  gain_percentage = 4 := by
  sorry

end watch_gain_percentage_l1150_115005


namespace evaluate_nested_brackets_l1150_115099

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- State the theorem
theorem evaluate_nested_brackets :
  bracket (bracket 100 50 150) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry

end evaluate_nested_brackets_l1150_115099


namespace sqrt_nested_expression_l1150_115072

theorem sqrt_nested_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end sqrt_nested_expression_l1150_115072


namespace a_greater_than_b_l1150_115061

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => a n ^ 2 + 3

def b : ℕ → ℕ
  | 0 => 0
  | n + 1 => b n ^ 2 + 2 ^ (n + 1)

theorem a_greater_than_b : b 2003 < a 2003 := by
  sorry

end a_greater_than_b_l1150_115061


namespace students_not_playing_sports_l1150_115041

theorem students_not_playing_sports (total : ℕ) (soccer : ℕ) (volleyball : ℕ) (one_sport : ℕ) : 
  total = 40 → soccer = 20 → volleyball = 19 → one_sport = 15 → 
  ∃ (both : ℕ), 
    both = soccer + volleyball - one_sport ∧
    total - (soccer + volleyball - both) = 13 := by
  sorry

end students_not_playing_sports_l1150_115041


namespace bobby_candy_problem_l1150_115089

theorem bobby_candy_problem (C : ℕ) : 
  (C + 36 = 16 + 58) → C = 38 := by
sorry

end bobby_candy_problem_l1150_115089


namespace triangle_sides_from_divided_areas_l1150_115078

/-- Given a triangle with an inscribed circle, if the segments from the vertices to the center
    of the inscribed circle divide the triangle's area into parts of 28, 60, and 80,
    then the sides of the triangle are 14, 30, and 40. -/
theorem triangle_sides_from_divided_areas (a b c : ℝ) (r : ℝ) :
  (1/2 * a * r = 28) →
  (1/2 * b * r = 60) →
  (1/2 * c * r = 80) →
  (a = 14 ∧ b = 30 ∧ c = 40) :=
by sorry

end triangle_sides_from_divided_areas_l1150_115078


namespace value_of_expression_l1150_115011

-- Define the function g
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- State the theorem
theorem value_of_expression (p q r s t : ℝ) 
  (h : g p q r s t (-1) = 4) : 
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 := by
  sorry

end value_of_expression_l1150_115011


namespace ball_ratio_problem_l1150_115067

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 5 / 3 →
  white_balls = 15 →
  red_balls = 9 := by
sorry

end ball_ratio_problem_l1150_115067


namespace square_side_length_l1150_115014

theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) (square_side : ℝ) : 
  rectangle_width = 4 →
  rectangle_length = 16 →
  square_side ^ 2 = rectangle_width * rectangle_length →
  square_side = 8 := by
sorry

end square_side_length_l1150_115014


namespace milk_pumping_rate_l1150_115037

/-- Calculates the rate of milk pumped into a tanker given initial conditions --/
theorem milk_pumping_rate 
  (initial_milk : ℝ) 
  (pumping_time : ℝ) 
  (add_rate : ℝ) 
  (add_time : ℝ) 
  (milk_left : ℝ) 
  (h1 : initial_milk = 30000)
  (h2 : pumping_time = 4)
  (h3 : add_rate = 1500)
  (h4 : add_time = 7)
  (h5 : milk_left = 28980) :
  (initial_milk + add_rate * add_time - milk_left) / pumping_time = 2880 := by
  sorry

#check milk_pumping_rate

end milk_pumping_rate_l1150_115037


namespace expression_simplification_l1150_115068

theorem expression_simplification (x : ℝ) (h : x ≠ -1) :
  x / (x + 1) - 3 * x / (2 * (x + 1)) - 1 = (-3 * x - 2) / (2 * (x + 1)) := by
  sorry

end expression_simplification_l1150_115068


namespace geometric_sequence_sum_l1150_115033

theorem geometric_sequence_sum (a q : ℝ) (h1 : a + a * q = 7) (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end geometric_sequence_sum_l1150_115033


namespace green_shirt_pairs_l1150_115059

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ)
  (h1 : red_students = 70)
  (h2 : green_students = 94)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 82)
  (h5 : red_red_pairs = 28)
  : ∃ green_green_pairs : ℕ, green_green_pairs = 40 ∧
    green_green_pairs = total_pairs - red_red_pairs - (red_students - 2 * red_red_pairs) := by
  sorry

end green_shirt_pairs_l1150_115059


namespace eight_million_factorization_roundness_of_eight_million_l1150_115077

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- 8,000,000 can be expressed as 8 × 10^6 -/
theorem eight_million_factorization : (8000000 : ℕ) = 8 * 10^6 := by sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end eight_million_factorization_roundness_of_eight_million_l1150_115077


namespace distance_between_trees_l1150_115048

/-- Given a curved path of length 300 meters with 26 trees planted at equal arc lengths,
    including one at each end, the distance between consecutive trees is 12 meters. -/
theorem distance_between_trees (path_length : ℝ) (num_trees : ℕ) :
  path_length = 300 ∧ num_trees = 26 →
  (path_length / (num_trees - 1 : ℝ)) = 12 := by
  sorry

end distance_between_trees_l1150_115048


namespace factorization_equality_l1150_115045

theorem factorization_equality (a b : ℝ) : 2 * a^2 * b - 8 * b = 2 * b * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l1150_115045


namespace right_triangle_sine_cosine_sum_equality_l1150_115088

theorem right_triangle_sine_cosine_sum_equality (A B C : ℝ) (x y : ℝ) 
  (h1 : A + B + C = π / 2)  -- ∠C is a right angle
  (h2 : 0 ≤ A ∧ A ≤ π / 2)  -- A is an angle in the right triangle
  (h3 : 0 ≤ B ∧ B ≤ π / 2)  -- B is an angle in the right triangle
  (h4 : x = Real.sin A + Real.cos A)  -- Definition of x
  (h5 : y = Real.sin B + Real.cos B)  -- Definition of y
  : x = y := by
  sorry

end right_triangle_sine_cosine_sum_equality_l1150_115088


namespace question_distribution_l1150_115076

-- Define the types for our problem
def TotalQuestions : ℕ := 100
def CorrectAnswersPerStudent : ℕ := 60

-- Define the number of students
def NumStudents : ℕ := 3

-- Define the types of questions
def EasyQuestions (x : ℕ) : Prop := x ≤ TotalQuestions
def MediumQuestions (y : ℕ) : Prop := y ≤ TotalQuestions
def DifficultQuestions (z : ℕ) : Prop := z ≤ TotalQuestions

-- State the theorem
theorem question_distribution 
  (x y z : ℕ) 
  (h1 : EasyQuestions x)
  (h2 : MediumQuestions y)
  (h3 : DifficultQuestions z)
  (h4 : x + y + z = TotalQuestions)
  (h5 : 3 * x + 2 * y + z = NumStudents * CorrectAnswersPerStudent) :
  z - x = 20 :=
sorry

end question_distribution_l1150_115076


namespace quadratic_equation_roots_ratio_l1150_115066

theorem quadratic_equation_roots_ratio (q : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ / r₂ = 3 ∧ 
   r₁^2 + 10*r₁ + q = 0 ∧ r₂^2 + 10*r₂ + q = 0) → 
  q = 18.75 := by
sorry

end quadratic_equation_roots_ratio_l1150_115066


namespace carrots_not_used_l1150_115007

theorem carrots_not_used (total : ℕ) (before_lunch_fraction : ℚ) (end_of_day_fraction : ℚ) : 
  total = 300 →
  before_lunch_fraction = 2 / 5 →
  end_of_day_fraction = 3 / 5 →
  (total - (before_lunch_fraction * total).num - (end_of_day_fraction * (total - (before_lunch_fraction * total).num)).num) = 72 := by
  sorry

end carrots_not_used_l1150_115007
