import Mathlib

namespace concatenation_product_sum_l2709_270969

theorem concatenation_product_sum : ∃! (n m : ℕ), 
  (10 ≤ n ∧ n < 100) ∧ 
  (100 ≤ m ∧ m < 1000) ∧ 
  (1000 * n + m = 9 * n * m) ∧ 
  (n + m = 126) := by
  sorry

end concatenation_product_sum_l2709_270969


namespace intersection_point_is_unique_l2709_270972

-- Define the two lines
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 4

-- Define the intersection point
def intersection_point : ℚ × ℚ := (-12/7, 22/7)

-- Theorem statement
theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point) := by
  sorry

end intersection_point_is_unique_l2709_270972


namespace original_number_proof_l2709_270964

theorem original_number_proof : 
  ∃ x : ℝ, (204 / x = 16) ∧ (x = 12.75) := by
  sorry

end original_number_proof_l2709_270964


namespace square_of_1023_l2709_270920

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end square_of_1023_l2709_270920


namespace function_periodicity_l2709_270912

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity
  (f : ℝ → ℝ)
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/7) + f (x + 1/6)) :
  is_periodic f 1 := by
  sorry

end function_periodicity_l2709_270912


namespace floor_difference_l2709_270927

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := sorry

/-- The number of floors in Building C -/
def floors_C : ℕ := 59

/-- The relationship between floors in Building B and C -/
axiom floors_C_relation : floors_C = 5 * floors_B - 6

/-- The difference in floors between Building A and Building B is 9 -/
theorem floor_difference : floors_B - floors_A = 9 := by sorry

end floor_difference_l2709_270927


namespace volleyball_tickets_l2709_270908

def initial_tickets (jude_tickets andrea_tickets sandra_tickets tickets_left : ℕ) : Prop :=
  andrea_tickets = 2 * jude_tickets ∧
  sandra_tickets = jude_tickets / 2 + 4 ∧
  jude_tickets = 16 ∧
  tickets_left = 40 ∧
  jude_tickets + andrea_tickets + sandra_tickets + tickets_left = 100

theorem volleyball_tickets :
  ∃ (jude_tickets andrea_tickets sandra_tickets tickets_left : ℕ),
    initial_tickets jude_tickets andrea_tickets sandra_tickets tickets_left :=
by
  sorry

end volleyball_tickets_l2709_270908


namespace total_pupils_count_l2709_270954

/-- The number of girls in the school -/
def num_girls : ℕ := 232

/-- The number of boys in the school -/
def num_boys : ℕ := 253

/-- The total number of pupils in the school -/
def total_pupils : ℕ := num_girls + num_boys

/-- Theorem: The total number of pupils in the school is 485 -/
theorem total_pupils_count : total_pupils = 485 := by
  sorry

end total_pupils_count_l2709_270954


namespace function_inequality_existence_l2709_270958

theorem function_inequality_existence (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, x + a * f y ≤ y + f (f x)) ↔ (a < 0 ∨ a = 1) := by
sorry

end function_inequality_existence_l2709_270958


namespace solution_to_equation_l2709_270997

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (3 * x)^5 = (9 * x)^4 → x = 3 := by
sorry

end solution_to_equation_l2709_270997


namespace triangle_6_8_10_is_right_l2709_270988

-- Define a triangle with sides a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

-- Theorem: A triangle with sides 6, 8, and 10 is a right triangle
theorem triangle_6_8_10_is_right : 
  let t : Triangle := { a := 6, b := 8, c := 10 }
  isRightTriangle t := by
  sorry


end triangle_6_8_10_is_right_l2709_270988


namespace tank_water_level_l2709_270942

theorem tank_water_level (tank_capacity : ℝ) (initial_level : ℝ) 
  (empty_percentage : ℝ) (fill_percentage : ℝ) (final_volume : ℝ) :
  tank_capacity = 8000 →
  empty_percentage = 0.4 →
  fill_percentage = 0.3 →
  final_volume = 4680 →
  final_volume = initial_level * (1 - empty_percentage) * (1 + fill_percentage) →
  initial_level / tank_capacity = 0.75 := by
sorry

end tank_water_level_l2709_270942


namespace toy_cost_price_l2709_270968

/-- The cost price of a toy, given the selling conditions -/
def cost_price (selling_price : ℕ) (num_sold : ℕ) (gain_equiv : ℕ) : ℚ :=
  selling_price / (num_sold + gain_equiv)

theorem toy_cost_price :
  let selling_price : ℕ := 25200
  let num_sold : ℕ := 18
  let gain_equiv : ℕ := 3
  cost_price selling_price num_sold gain_equiv = 1200 := by
  sorry

end toy_cost_price_l2709_270968


namespace range_of_a_l2709_270926

-- Define the functions f and g
def f (a x : ℝ) := a - x^2
def g (x : ℝ) := x + 1

-- Define the symmetry condition
def symmetric_about_x_axis (f g : ℝ → ℝ) (a : ℝ) :=
  ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x = -g x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (symmetric_about_x_axis (f a) g a) → -1 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l2709_270926


namespace product_pricing_l2709_270919

/-- Given three products A, B, and C with unknown prices, prove that if 2A + 3B + 1C costs 295 yuan
    and 4A + 3B + 5C costs 425 yuan, then 1A + 1B + 1C costs 120 yuan. -/
theorem product_pricing (a b c : ℝ) 
    (h1 : 2*a + 3*b + c = 295)
    (h2 : 4*a + 3*b + 5*c = 425) : 
  a + b + c = 120 := by
sorry

end product_pricing_l2709_270919


namespace rectangle_area_l2709_270967

/-- The area of a rectangle with vertices at (-7, 1), (1, 1), (1, -6), and (-7, -6) in a rectangular coordinate system is 56 square units. -/
theorem rectangle_area : ℝ := by
  -- Define the vertices of the rectangle
  let v1 : ℝ × ℝ := (-7, 1)
  let v2 : ℝ × ℝ := (1, 1)
  let v3 : ℝ × ℝ := (1, -6)
  let v4 : ℝ × ℝ := (-7, -6)

  -- Calculate the length and width of the rectangle
  let length : ℝ := v2.1 - v1.1
  let width : ℝ := v1.2 - v4.2

  -- Calculate the area of the rectangle
  let area : ℝ := length * width

  -- Prove that the area is equal to 56
  sorry

end rectangle_area_l2709_270967


namespace trapezium_height_l2709_270994

theorem trapezium_height (a b h : ℝ) : 
  a = 20 → b = 18 → (1/2) * (a + b) * h = 247 → h = 13 := by
  sorry

end trapezium_height_l2709_270994


namespace sqrt_trig_identity_l2709_270976

theorem sqrt_trig_identity : 
  Real.sqrt (1 - 2 * Real.cos (π / 2 + 3) * Real.sin (π / 2 - 3)) = -Real.sin 3 - Real.cos 3 := by
  sorry

end sqrt_trig_identity_l2709_270976


namespace quadrilateral_equation_implies_rhombus_l2709_270978

-- Define a quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_d : d > 0

-- Define the condition from the problem
def satisfiesEquation (q : Quadrilateral) : Prop :=
  q.a^4 + q.b^4 + q.c^4 + q.d^4 = 4 * q.a * q.b * q.c * q.d

-- Define a rhombus
def isRhombus (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem statement
theorem quadrilateral_equation_implies_rhombus (q : Quadrilateral) :
  satisfiesEquation q → isRhombus q :=
by sorry

end quadrilateral_equation_implies_rhombus_l2709_270978


namespace B_power_difference_l2709_270932

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_difference :
  B^10 - 3 * B^9 = !![0, 4; 0, -1] := by sorry

end B_power_difference_l2709_270932


namespace problem_1_problem_2_l2709_270965

-- Problem 1
theorem problem_1 (x y : ℝ) : (2*x - 3*y)^2 - (y + 3*x)*(3*x - y) = -5*x^2 - 12*x*y + 10*y^2 := by
  sorry

-- Problem 2
theorem problem_2 : (2+1)*(2^2+1)*(2^4+1)*(2^8+1) - 2^16 = -1 := by
  sorry

end problem_1_problem_2_l2709_270965


namespace cosine_equality_l2709_270993

theorem cosine_equality (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 270) :
  Real.cos (n * π / 180) = Real.cos (890 * π / 180) → n = 10 := by
  sorry

end cosine_equality_l2709_270993


namespace find_special_numbers_l2709_270950

theorem find_special_numbers : ∃ (x y : ℕ), 
  x + y = 2013 ∧ 
  y = 5 * ((x / 100) + 1) ∧ 
  x ≥ y ∧ 
  x = 1913 := by
  sorry

end find_special_numbers_l2709_270950


namespace point_division_l2709_270977

/-- Given two points A and B in a vector space, and a point P on the line segment AB
    such that AP:PB = 4:1, prove that P = (4/5)*A + (1/5)*B -/
theorem point_division (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (A B P : V) (h : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ P - A = (4 * k) • (B - A) ∧ B - P = k • (B - A)) :
  P = (4/5) • A + (1/5) • B := by
  sorry

end point_division_l2709_270977


namespace fourth_tree_grows_more_l2709_270924

/-- Represents the daily growth rates of four trees and their total growth over a period --/
structure TreeGrowth where
  first_tree_rate : ℝ
  second_tree_rate : ℝ
  third_tree_rate : ℝ
  fourth_tree_rate : ℝ
  total_days : ℕ
  total_growth : ℝ

/-- The growth rates satisfy the problem conditions --/
def satisfies_conditions (g : TreeGrowth) : Prop :=
  g.first_tree_rate = 1 ∧
  g.second_tree_rate = 2 * g.first_tree_rate ∧
  g.third_tree_rate = 2 ∧
  g.total_days = 4 ∧
  g.total_growth = 32 ∧
  g.first_tree_rate * g.total_days +
  g.second_tree_rate * g.total_days +
  g.third_tree_rate * g.total_days +
  g.fourth_tree_rate * g.total_days = g.total_growth

/-- The theorem stating the difference in growth rates --/
theorem fourth_tree_grows_more (g : TreeGrowth) 
  (h : satisfies_conditions g) : 
  g.fourth_tree_rate - g.third_tree_rate = 1 := by
  sorry

end fourth_tree_grows_more_l2709_270924


namespace max_triangles_hit_five_times_is_25_l2709_270940

/-- Represents a triangular target divided into smaller equilateral triangles -/
structure Target where
  total_triangles : Nat
  mk_valid : total_triangles = 100

/-- Represents a shot by the sniper -/
structure Shot where
  aimed_triangle : Nat
  hit_triangle : Nat
  mk_valid : hit_triangle = aimed_triangle ∨ 
             hit_triangle = aimed_triangle - 1 ∨ 
             hit_triangle = aimed_triangle + 1

/-- Represents the result of multiple shots -/
def ShotResult := Nat → Nat

/-- The maximum number of triangles that can be hit exactly five times -/
def max_triangles_hit_five_times (t : Target) (shots : List Shot) : Nat :=
  sorry

/-- Theorem stating the maximum number of triangles hit exactly five times -/
theorem max_triangles_hit_five_times_is_25 (t : Target) :
  ∃ (shots : List Shot), max_triangles_hit_five_times t shots = 25 ∧
  ∀ (other_shots : List Shot), max_triangles_hit_five_times t other_shots ≤ 25 :=
sorry

end max_triangles_hit_five_times_is_25_l2709_270940


namespace base7_to_decimal_correct_l2709_270943

/-- Converts a base 7 digit to its decimal (base 10) value -/
def base7ToDecimal (d : ℕ) : ℕ := d

/-- Represents the number 23456 in base 7 as a list of its digits -/
def base7Number : List ℕ := [2, 3, 4, 5, 6]

/-- Converts a list of base 7 digits to its decimal (base 10) equivalent -/
def convertBase7ToDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + (base7ToDecimal d) * 7^i) 0

theorem base7_to_decimal_correct :
  convertBase7ToDecimal base7Number = 6068 := by sorry

end base7_to_decimal_correct_l2709_270943


namespace line_hyperbola_intersection_range_l2709_270962

/-- The range of k values for which the line y = kx + 1 intersects the right branch of the hyperbola 3x^2 - y^2 = 3 at two distinct points -/
theorem line_hyperbola_intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧
    3 * x₁^2 - y₁^2 = 3 ∧ 3 * x₂^2 - y₂^2 = 3) ↔
  -2 < k ∧ k < -Real.sqrt 3 :=
sorry

end line_hyperbola_intersection_range_l2709_270962


namespace sticks_remaining_proof_l2709_270904

/-- The number of sticks originally in the yard -/
def original_sticks : ℕ := 99

/-- The number of sticks Will picked up -/
def picked_up_sticks : ℕ := 38

/-- The number of sticks left after Will picked up some -/
def remaining_sticks : ℕ := original_sticks - picked_up_sticks

theorem sticks_remaining_proof : remaining_sticks = 61 := by
  sorry

end sticks_remaining_proof_l2709_270904


namespace stratified_sampling_medium_stores_l2709_270996

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) 
  (medium_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (sample_size * medium_stores) / total_stores = 5 := by
sorry

end stratified_sampling_medium_stores_l2709_270996


namespace conceived_number_is_seven_l2709_270923

theorem conceived_number_is_seven :
  ∃! (x : ℕ+), (10 * x.val + 7 - x.val ^ 2) / 4 - x.val = 0 :=
by
  sorry

end conceived_number_is_seven_l2709_270923


namespace parabola_transformation_theorem_l2709_270930

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotates a parabola 180 degrees about its vertex -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

/-- Shifts a parabola horizontally -/
def shiftHorizontal (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h - shift, k := p.k }

/-- Shifts a parabola vertically -/
def shiftVertical (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + shift }

/-- Calculates the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := 2 * p.h

theorem parabola_transformation_theorem :
  let original := Parabola.mk 1 3 4
  let transformed := shiftVertical (shiftHorizontal (rotate180 original) 5) (-4)
  sumOfZeros transformed = 16 := by sorry

end parabola_transformation_theorem_l2709_270930


namespace gumball_packages_l2709_270980

theorem gumball_packages (gumballs_per_package : ℕ) (gumballs_eaten : ℕ) : 
  gumballs_per_package = 5 → gumballs_eaten = 20 → 
  (gumballs_eaten / gumballs_per_package : ℕ) = 4 := by
sorry

end gumball_packages_l2709_270980


namespace ellipse_properties_l2709_270903

-- Define the ellipse (C)
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line (l) with slope 1 passing through F(1,0)
def Line (x y : ℝ) : Prop := y = x - 1

-- Define the perpendicular bisector of MN
def PerpendicularBisector (k : ℝ) (x y : ℝ) : Prop :=
  y + (3*k)/(3 + 4*k^2) = -(1/k)*(x - (4*k^2)/(3 + 4*k^2))

theorem ellipse_properties :
  -- Given conditions
  (Ellipse 2 0) →
  (Ellipse 1 0) →
  -- Prove the following
  (∀ x y, Ellipse x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ 
    Line x₁ y₁ ∧ Line x₂ y₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2 : ℝ) = 24/7) ∧
  (∀ k y₀, k ≠ 0 →
    PerpendicularBisector k 0 y₀ →
    -Real.sqrt 3 / 12 ≤ y₀ ∧ y₀ ≤ Real.sqrt 3 / 12) :=
by sorry

end ellipse_properties_l2709_270903


namespace investment_ratio_l2709_270937

theorem investment_ratio (a b c : ℝ) (total_profit b_share : ℝ) :
  b = (2/3) * c →
  a = n * b →
  total_profit = 3300 →
  b_share = 600 →
  b_share / total_profit = b / (a + b + c) →
  a / b = 3 :=
sorry

end investment_ratio_l2709_270937


namespace remainder_sum_l2709_270921

theorem remainder_sum (a b : ℤ) (ha : a % 60 = 53) (hb : b % 45 = 17) : (a + b) % 15 = 5 := by
  sorry

end remainder_sum_l2709_270921


namespace shoe_difference_l2709_270982

/-- Given information about shoe boxes and quantities, prove the difference in pairs of shoes. -/
theorem shoe_difference (pairs_per_box : ℕ) (boxes_of_A : ℕ) (B_to_A_ratio : ℕ) : 
  pairs_per_box = 20 →
  boxes_of_A = 8 →
  B_to_A_ratio = 5 →
  B_to_A_ratio * (pairs_per_box * boxes_of_A) - (pairs_per_box * boxes_of_A) = 640 := by
  sorry

end shoe_difference_l2709_270982


namespace sum_reciprocal_squared_ge_sum_squared_l2709_270956

theorem sum_reciprocal_squared_ge_sum_squared 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 := by
  sorry

end sum_reciprocal_squared_ge_sum_squared_l2709_270956


namespace fibonacci_m_digit_count_fibonacci_5n_plus_2_digits_l2709_270952

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

theorem fibonacci_m_digit_count (m : ℕ) (h : m ≥ 2) :
  ∃ k : ℕ, fib k ≥ 10^(m-1) ∧ fib (k+3) < 10^m ∧ fib (k+4) ≥ 10^m :=
sorry

theorem fibonacci_5n_plus_2_digits (n : ℕ) :
  fib (5*n + 2) ≥ 10^n :=
sorry

end fibonacci_m_digit_count_fibonacci_5n_plus_2_digits_l2709_270952


namespace isosceles_triangle_largest_angle_l2709_270931

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two angles equal to 50°
  α = β ∧ α = 50 ∧
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  max α (max β γ) = 80 :=
sorry

end isosceles_triangle_largest_angle_l2709_270931


namespace trip_distance_proof_l2709_270998

/-- Represents the total distance of the trip in miles -/
def total_distance : ℝ := 90

/-- Represents the distance traveled on battery power in miles -/
def battery_distance : ℝ := 30

/-- Represents the gasoline consumption rate after battery power in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- Represents the overall average fuel efficiency in miles per gallon -/
def average_efficiency : ℝ := 50

/-- Proves that the total trip distance is correct given the conditions -/
theorem trip_distance_proof :
  (total_distance / (gasoline_rate * (total_distance - battery_distance)) = average_efficiency) ∧
  (total_distance > battery_distance) :=
by sorry

end trip_distance_proof_l2709_270998


namespace exists_proportion_with_means_less_than_extremes_l2709_270901

/-- A proportion is represented by four real numbers a, b, c, d such that a : b = c : d -/
def IsProportion (a b c d : ℝ) : Prop := a * d = b * c

/-- Theorem: There exists a proportion where both means are less than both extremes -/
theorem exists_proportion_with_means_less_than_extremes :
  ∃ (a b c d : ℝ), IsProportion a b c d ∧ b < a ∧ b < d ∧ c < a ∧ c < d := by
  sorry

end exists_proportion_with_means_less_than_extremes_l2709_270901


namespace f_has_three_zeros_l2709_270989

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x + 2/x
  else if x > 0 then x * Real.log x - a
  else 0

theorem f_has_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  (-1 / Real.exp 1 < a ∧ a < 0) :=
sorry

end f_has_three_zeros_l2709_270989


namespace simplify_trigonometric_expression_l2709_270905

theorem simplify_trigonometric_expression (x : ℝ) : 
  2 * Real.sin (2 * x) * Real.sin x + Real.cos (3 * x) = Real.cos x := by
sorry

end simplify_trigonometric_expression_l2709_270905


namespace max_sum_given_constraints_l2709_270928

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 36) : 
  x + y ≤ 2 * Real.sqrt 43 :=
by sorry

end max_sum_given_constraints_l2709_270928


namespace janets_sandcastle_height_l2709_270959

/-- Given the heights of two sandcastles, proves that the taller one is the sum of the shorter one's height and the difference between them. -/
theorem janets_sandcastle_height 
  (sisters_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : sisters_height = 2.3333333333333335)
  (h2 : height_difference = 1.3333333333333333) : 
  sisters_height + height_difference = 3.6666666666666665 := by
  sorry

#check janets_sandcastle_height

end janets_sandcastle_height_l2709_270959


namespace mirasol_spending_l2709_270949

/-- Mirasol's spending problem -/
theorem mirasol_spending (initial_amount : ℕ) (coffee_cost : ℕ) (remaining_amount : ℕ) 
  (tumbler_cost : ℕ) :
  initial_amount = 50 →
  coffee_cost = 10 →
  remaining_amount = 10 →
  initial_amount = coffee_cost + tumbler_cost + remaining_amount →
  tumbler_cost = 30 := by
  sorry

end mirasol_spending_l2709_270949


namespace percentage_of_hindu_boys_l2709_270995

-- Define the total number of boys
def total_boys : ℕ := 700

-- Define the percentage of Muslim boys
def muslim_percentage : ℚ := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage : ℚ := 10 / 100

-- Define the number of boys from other communities
def other_boys : ℕ := 126

-- Define the percentage of Hindu boys
def hindu_percentage : ℚ := 28 / 100

-- Theorem statement
theorem percentage_of_hindu_boys :
  hindu_percentage * total_boys =
    total_boys - (muslim_percentage * total_boys + sikh_percentage * total_boys + other_boys) :=
by sorry

end percentage_of_hindu_boys_l2709_270995


namespace remainder_problem_l2709_270946

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 31 = 5 := by
  sorry

end remainder_problem_l2709_270946


namespace greatest_two_digit_multiple_of_17_l2709_270941

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 100 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l2709_270941


namespace parallel_vectors_m_value_l2709_270948

/-- Given plane vectors a and b, if a is parallel to 2b - a, then m = 9/2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (4, 3))
    (h2 : b = (6, m))
    (h3 : ∃ (k : ℝ), a = k • (2 • b - a)) :
  m = 9/2 := by
  sorry

end parallel_vectors_m_value_l2709_270948


namespace grocery_store_order_l2709_270974

theorem grocery_store_order (peas carrots corn : ℕ) 
  (h_peas : peas = 810) 
  (h_carrots : carrots = 954) 
  (h_corn : corn = 675) : 
  ∃ (boxes packs cases : ℕ), 
    boxes * 4 ≥ peas ∧ 
    (boxes - 1) * 4 < peas ∧ 
    packs * 6 = carrots ∧ 
    cases * 5 = corn ∧ 
    boxes = 203 ∧ 
    packs = 159 ∧ 
    cases = 135 := by
  sorry

#check grocery_store_order

end grocery_store_order_l2709_270974


namespace range_of_a_l2709_270913

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - a*t - a ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l2709_270913


namespace unique_positive_solution_l2709_270992

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x := by
  sorry

end unique_positive_solution_l2709_270992


namespace scientific_notation_of_35000000_l2709_270916

theorem scientific_notation_of_35000000 :
  (35000000 : ℝ) = 3.5 * (10 ^ 7) := by sorry

end scientific_notation_of_35000000_l2709_270916


namespace rectangle_cylinder_volume_ratio_l2709_270979

theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 9
  let cylinder_a_radius : ℝ := rectangle_width / (2 * Real.pi)
  let cylinder_a_height : ℝ := rectangle_height
  let cylinder_b_radius : ℝ := rectangle_height / (2 * Real.pi)
  let cylinder_b_height : ℝ := rectangle_width
  let volume_a : ℝ := Real.pi * cylinder_a_radius^2 * cylinder_a_height
  let volume_b : ℝ := Real.pi * cylinder_b_radius^2 * cylinder_b_height
  volume_b / volume_a = 3 / 4 := by
  sorry

end rectangle_cylinder_volume_ratio_l2709_270979


namespace equation_solutions_l2709_270906

theorem equation_solutions :
  (∀ x : ℝ, x^2 + 2*x - 8 = 0 ↔ x = -4 ∨ x = 2) ∧
  (∀ x : ℝ, 2*(x+3)^2 = x*(x+3) ↔ x = -3 ∨ x = -6) := by
  sorry

end equation_solutions_l2709_270906


namespace second_tank_fish_length_is_two_l2709_270917

/-- Represents the fish tank system with given conditions -/
structure FishTankSystem where
  first_tank_size : ℝ
  second_tank_size : ℝ
  first_tank_water : ℝ
  first_tank_fish_length : ℝ
  fish_difference_after_eating : ℕ
  (size_relation : first_tank_size = 2 * second_tank_size)
  (first_tank_water_amount : first_tank_water = 48)
  (first_tank_fish_size : first_tank_fish_length = 3)
  (fish_difference : fish_difference_after_eating = 3)

/-- The length of fish in the second tank -/
def second_tank_fish_length (system : FishTankSystem) : ℝ :=
  2

/-- Theorem stating that the length of fish in the second tank is 2 inches -/
theorem second_tank_fish_length_is_two (system : FishTankSystem) :
  second_tank_fish_length system = 2 := by
  sorry

end second_tank_fish_length_is_two_l2709_270917


namespace apple_pile_count_l2709_270953

-- Define the initial number of apples
def initial_apples : ℕ := 8

-- Define the number of apples added
def added_apples : ℕ := 5

-- Theorem to prove
theorem apple_pile_count : initial_apples + added_apples = 13 := by
  sorry

end apple_pile_count_l2709_270953


namespace massachusetts_avenue_pairings_l2709_270945

/-- Represents the number of possible pairings for n blocks -/
def pairings : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => pairings (n + 1) + pairings n

/-- The 10th Fibonacci number -/
def fib10 : ℕ := pairings 10

theorem massachusetts_avenue_pairings :
  fib10 = 89 :=
by sorry

end massachusetts_avenue_pairings_l2709_270945


namespace banana_change_calculation_emily_banana_change_l2709_270961

/-- Calculates the change received when buying bananas with a discount --/
theorem banana_change_calculation (num_bananas : ℕ) (cost_per_banana : ℚ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) (paid_amount : ℚ) : ℚ :=
  let total_cost := num_bananas * cost_per_banana
  let discounted_cost := if num_bananas > discount_threshold 
    then total_cost * (1 - discount_rate) 
    else total_cost
  paid_amount - discounted_cost

/-- Proves that Emily received $8.65 in change --/
theorem emily_banana_change : 
  banana_change_calculation 5 (30/100) 4 (10/100) 10 = 865/100 := by
  sorry

end banana_change_calculation_emily_banana_change_l2709_270961


namespace train_length_l2709_270984

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 54 →
  crossing_time = 58.9952803775698 →
  bridge_length = 720 →
  ∃ (train_length : ℝ), abs (train_length - 164.93) < 0.01 := by
  sorry

end train_length_l2709_270984


namespace f_properties_l2709_270970

noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 2 → f x < f y) := by
  sorry

end f_properties_l2709_270970


namespace condition_A_not_necessary_nor_sufficient_l2709_270934

-- Define the conditions
def condition_A (θ : Real) (a : Real) : Prop := Real.sqrt (1 + Real.sin θ) = a
def condition_B (θ : Real) (a : Real) : Prop := Real.sin (θ / 2) + Real.cos (θ / 2) = a

-- Theorem statement
theorem condition_A_not_necessary_nor_sufficient :
  ¬(∀ θ a, condition_B θ a → condition_A θ a) ∧
  ¬(∀ θ a, condition_A θ a → condition_B θ a) := by
  sorry

end condition_A_not_necessary_nor_sufficient_l2709_270934


namespace M_characterization_inequality_in_M_l2709_270951

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 4}

-- Theorem 1: Characterization of set M
theorem M_characterization : M = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a^2 + 2*a - 3) * (b^2 + 2*b - 3) ≥ 0 := by sorry

end M_characterization_inequality_in_M_l2709_270951


namespace min_value_on_interval_l2709_270963

def f (x a : ℝ) : ℝ := 3 * x^4 - 8 * x^3 - 18 * x^2 + a

theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-1) 1, f x a = 6) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x a ≤ 6) →
  (∃ x ∈ Set.Icc (-1) 1, f x a = -17) ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x a ≥ -17) := by
  sorry

end min_value_on_interval_l2709_270963


namespace twenty_four_is_eighty_percent_of_thirty_l2709_270938

theorem twenty_four_is_eighty_percent_of_thirty : 
  ∀ x : ℝ, (24 : ℝ) / x = (80 : ℝ) / 100 → x = 30 := by
  sorry

end twenty_four_is_eighty_percent_of_thirty_l2709_270938


namespace runners_meet_time_l2709_270991

/-- Two runners on a circular track meet after approximately 15 seconds --/
theorem runners_meet_time (track_length : ℝ) (speed1 speed2 : ℝ) : 
  track_length = 250 →
  speed1 = 20 * (1000 / 3600) →
  speed2 = 40 * (1000 / 3600) →
  abs (15 - track_length / (speed1 + speed2)) < 0.1 := by
  sorry

#check runners_meet_time

end runners_meet_time_l2709_270991


namespace linda_furniture_fraction_l2709_270985

def original_savings : ℚ := 1200
def tv_cost : ℚ := 300

def furniture_cost : ℚ := original_savings - tv_cost

theorem linda_furniture_fraction :
  furniture_cost / original_savings = 3 / 4 := by sorry

end linda_furniture_fraction_l2709_270985


namespace trigonometric_problem_l2709_270909

theorem trigonometric_problem (α : ℝ) 
  (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1/4) :
  (Real.tan α = -2) ∧ 
  ((Real.sin (2*α) + 1) / (1 + Real.sin (2*α) + Real.cos (2*α)) = -1/2) := by
  sorry

end trigonometric_problem_l2709_270909


namespace isosceles_trapezoid_estate_area_l2709_270986

/-- Represents the scale of the map --/
def map_scale : ℚ := 500 / 2

/-- Represents the length of the diagonals on the map in inches --/
def diagonal_length : ℚ := 10

/-- Calculates the actual length of the diagonal in miles --/
def actual_diagonal_length : ℚ := diagonal_length * map_scale

/-- Represents an isosceles trapezoid estate --/
structure IsoscelesTrapezoidEstate where
  diagonal : ℚ
  area : ℚ

/-- Theorem stating the area of the isosceles trapezoid estate --/
theorem isosceles_trapezoid_estate_area :
  ∃ (estate : IsoscelesTrapezoidEstate),
    estate.diagonal = actual_diagonal_length ∧
    estate.area = 3125000 := by
  sorry


end isosceles_trapezoid_estate_area_l2709_270986


namespace malt_shop_problem_l2709_270935

/-- Represents the number of ounces of chocolate syrup used per shake -/
def syrup_per_shake : ℕ := 4

/-- Represents the number of ounces of chocolate syrup used per cone -/
def syrup_per_cone : ℕ := 6

/-- Represents the number of shakes sold -/
def shakes_sold : ℕ := 2

/-- Represents the total number of ounces of chocolate syrup used -/
def total_syrup_used : ℕ := 14

/-- Represents the number of cones sold -/
def cones_sold : ℕ := 1

theorem malt_shop_problem :
  syrup_per_shake * shakes_sold + syrup_per_cone * cones_sold = total_syrup_used :=
by sorry

end malt_shop_problem_l2709_270935


namespace sandy_fish_count_l2709_270981

def initial_fish : ℕ := 26
def bought_fish : ℕ := 6

theorem sandy_fish_count : initial_fish + bought_fish = 32 := by
  sorry

end sandy_fish_count_l2709_270981


namespace quadratic_minimum_l2709_270957

/-- A quadratic function f(x) = x^2 + px + qx, where p and q are positive constants -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q*x

/-- The theorem stating that the minimum of f occurs at x = -(p+q)/2 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), x_min = -(p + q) / 2 ∧ 
  ∀ (x : ℝ), f p q x_min ≤ f p q x :=
sorry

end quadratic_minimum_l2709_270957


namespace square_area_ratio_l2709_270910

theorem square_area_ratio (s₁ s₂ : ℝ) (h : s₁ = 2 * s₂) : s₁^2 / s₂^2 = 4 := by
  sorry

end square_area_ratio_l2709_270910


namespace john_concert_probability_l2709_270955

theorem john_concert_probability
  (p_rain : ℝ)
  (p_john_if_rain : ℝ)
  (p_john_if_sunny : ℝ)
  (h_rain : p_rain = 0.50)
  (h_john_rain : p_john_if_rain = 0.30)
  (h_john_sunny : p_john_if_sunny = 0.90) :
  p_rain * p_john_if_rain + (1 - p_rain) * p_john_if_sunny = 0.60 :=
by sorry

end john_concert_probability_l2709_270955


namespace sphere_surface_area_of_inscribed_cuboid_l2709_270914

theorem sphere_surface_area_of_inscribed_cuboid (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 6) (h3 : c = 3) :
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let r := d / 2
  4 * Real.pi * r^2 = 16 * Real.pi := by sorry

end sphere_surface_area_of_inscribed_cuboid_l2709_270914


namespace tangency_iff_condition_l2709_270987

/-- The line equation -/
def line_equation (x y : ℝ) : Prop :=
  y = 2 * x + 1

/-- The ellipse equation -/
def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The tangency condition -/
def tangency_condition (a b : ℝ) : Prop :=
  4 * a^2 + b^2 = 1

/-- Theorem stating that the tangency condition is necessary and sufficient -/
theorem tangency_iff_condition (a b : ℝ) :
  (∃! p : ℝ × ℝ, line_equation p.1 p.2 ∧ ellipse_equation p.1 p.2 a b) ↔ tangency_condition a b :=
sorry

end tangency_iff_condition_l2709_270987


namespace new_students_count_l2709_270922

/-- Represents the number of new students who joined the class -/
def new_students : ℕ := sorry

/-- The original average age of the class -/
def original_avg_age : ℕ := 40

/-- The average age of new students -/
def new_students_avg_age : ℕ := 32

/-- The decrease in average age after new students join -/
def avg_age_decrease : ℕ := 4

/-- The original number of students in the class -/
def original_class_size : ℕ := 18

theorem new_students_count :
  (original_class_size * original_avg_age + new_students * new_students_avg_age) / (original_class_size + new_students) = original_avg_age - avg_age_decrease ∧
  new_students = 18 :=
sorry

end new_students_count_l2709_270922


namespace orange_gumdrops_after_replacement_l2709_270907

/-- Represents the number of gumdrops of each color in a jar --/
structure GumdropsJar where
  purple : ℕ
  orange : ℕ
  violet : ℕ
  yellow : ℕ
  white : ℕ
  green : ℕ

/-- Calculates the total number of gumdrops in the jar --/
def total_gumdrops (jar : GumdropsJar) : ℕ :=
  jar.purple + jar.orange + jar.violet + jar.yellow + jar.white + jar.green

/-- Theorem stating the number of orange gumdrops after replacement --/
theorem orange_gumdrops_after_replacement (jar : GumdropsJar) :
  jar.white = 40 ∧
  total_gumdrops jar = 160 ∧
  jar.purple = 40 ∧
  jar.orange = 24 ∧
  jar.violet = 32 ∧
  jar.yellow = 24 →
  jar.orange + (jar.purple / 3) = 37 := by
  sorry

#check orange_gumdrops_after_replacement

end orange_gumdrops_after_replacement_l2709_270907


namespace subset_union_equality_l2709_270939

theorem subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
  (h_nonempty : ∀ i, (A i).Nonempty) : 
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧ 
  (⋃ (i ∈ I), A i) = (⋃ (j ∈ J), A j) := by
sorry

end subset_union_equality_l2709_270939


namespace average_problem_l2709_270960

theorem average_problem (x : ℝ) : (1 + 3 + x) / 3 = 3 → x = 5 := by
  sorry

end average_problem_l2709_270960


namespace polynomial_real_root_iff_b_in_range_l2709_270918

/-- A polynomial of the form x^4 + bx^3 + x^2 + bx + 1 -/
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + b*x^3 + x^2 + b*x + 1

/-- The polynomial has at least one real root -/
def has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, polynomial b x = 0

/-- Theorem: The polynomial has at least one real root if and only if b is in [-3/4, 0) -/
theorem polynomial_real_root_iff_b_in_range :
  ∀ b : ℝ, has_real_root b ↔ -3/4 ≤ b ∧ b < 0 := by sorry

end polynomial_real_root_iff_b_in_range_l2709_270918


namespace rectangular_box_dimensions_sum_l2709_270973

theorem rectangular_box_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 40)
  (h2 : B * C = 90)
  (h3 : C * A = 100) :
  A + B + C = 83/3 := by
sorry

end rectangular_box_dimensions_sum_l2709_270973


namespace father_age_problem_l2709_270947

theorem father_age_problem (father_age son_age : ℕ) : 
  (father_age = 4 * son_age + 4) →
  (father_age + 4 = 2 * (son_age + 4) + 20) →
  father_age = 44 := by
  sorry

end father_age_problem_l2709_270947


namespace min_value_of_f_l2709_270925

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 := by
  sorry

end min_value_of_f_l2709_270925


namespace sum_of_An_and_Bn_l2709_270971

/-- The sum of numbers in the n-th group of positive integers -/
def A (n : ℕ) : ℕ :=
  (2 * n - 1) * (n^2 - n + 1)

/-- The difference between the second and first number in the n-th group of cubes of natural numbers -/
def B (n : ℕ) : ℕ :=
  n^3 - (n - 1)^3

/-- Theorem stating that A_n + B_n = 2n³ for any positive integer n -/
theorem sum_of_An_and_Bn (n : ℕ) : A n + B n = 2 * n^3 := by
  sorry

end sum_of_An_and_Bn_l2709_270971


namespace last_digit_2016_octal_l2709_270933

def decimal_to_octal_last_digit (n : ℕ) : ℕ :=
  n % 8

theorem last_digit_2016_octal : decimal_to_octal_last_digit 2016 = 0 := by
  sorry

end last_digit_2016_octal_l2709_270933


namespace vector_sum_proof_l2709_270915

theorem vector_sum_proof :
  let v1 : Fin 3 → ℝ := ![(-3), 2, (-1)]
  let v2 : Fin 3 → ℝ := ![1, 5, (-3)]
  v1 + v2 = ![(-2), 7, (-4)] := by
  sorry

end vector_sum_proof_l2709_270915


namespace rotten_bananas_percentage_l2709_270936

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 878 / 1000)
  : (total_bananas - (total_oranges + total_bananas - 
     (good_fruits_percentage * (total_oranges + total_bananas)).floor - 
     (rotten_oranges_percentage * total_oranges).floor)) / total_bananas = 8 / 100 := by
  sorry


end rotten_bananas_percentage_l2709_270936


namespace cyclist_rate_problem_l2709_270966

/-- Prove that given two cyclists A and B traveling between Newton and Kingston,
    with the given conditions, the rate of cyclist A is 10 mph. -/
theorem cyclist_rate_problem (rate_A rate_B : ℝ) : 
  rate_B = rate_A + 5 →                   -- B travels 5 mph faster than A
  50 / rate_A = (50 + 10) / rate_B →      -- Time for A to travel 40 miles equals time for B to travel 60 miles
  rate_A = 10 := by
sorry

end cyclist_rate_problem_l2709_270966


namespace average_price_is_86_l2709_270944

def prices : List ℝ := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

theorem average_price_is_86 : 
  (prices.sum / prices.length : ℝ) = 86 := by sorry

end average_price_is_86_l2709_270944


namespace square_area_l2709_270990

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

/-- The line function -/
def g (x : ℝ) : ℝ := 10

theorem square_area : ∃ (x₁ x₂ : ℝ), 
  f x₁ = g x₁ ∧ 
  f x₂ = g x₂ ∧ 
  x₁ ≠ x₂ ∧ 
  (x₂ - x₁)^2 = 41 := by
sorry

end square_area_l2709_270990


namespace andrews_sticker_fraction_l2709_270911

theorem andrews_sticker_fraction 
  (total_stickers : ℕ) 
  (andrews_fraction : ℚ) 
  (bills_fraction : ℚ) 
  (total_given : ℕ) :
  total_stickers = 100 →
  bills_fraction = 3/10 →
  total_given = 44 →
  andrews_fraction * total_stickers + 
    bills_fraction * (total_stickers - andrews_fraction * total_stickers) = total_given →
  andrews_fraction = 1/5 := by
sorry

end andrews_sticker_fraction_l2709_270911


namespace complete_square_result_l2709_270900

theorem complete_square_result (x : ℝ) : 
  x^2 + 6*x - 4 = 0 ↔ (x + 3)^2 = 13 := by
  sorry

end complete_square_result_l2709_270900


namespace base_2_representation_of_123_l2709_270929

theorem base_2_representation_of_123 : 
  ∃ (b : List Bool), 
    (b.length = 7) ∧ 
    (b = [true, true, true, true, false, true, true]) ∧
    (Nat.ofDigits 2 (b.map (fun x => if x then 1 else 0)) = 123) := by
  sorry

end base_2_representation_of_123_l2709_270929


namespace wheel_on_semicircle_diameter_l2709_270902

theorem wheel_on_semicircle_diameter (r_wheel r_semicircle : ℝ) 
  (h_wheel : r_wheel = 8)
  (h_semicircle : r_semicircle = 25) :
  let untouched_length := 2 * (r_semicircle - (r_semicircle^2 - r_wheel^2).sqrt)
  untouched_length = 20 := by
sorry

end wheel_on_semicircle_diameter_l2709_270902


namespace art_class_problem_l2709_270975

theorem art_class_problem (total_students : ℕ) (total_kits : ℕ) (total_artworks : ℕ) 
  (h1 : total_students = 10)
  (h2 : total_kits = 20)
  (h3 : total_artworks = 35)
  (h4 : 2 * total_kits = total_students) -- 1 kit for 2 students
  (h5 : total_students % 2 = 0) -- Ensures even number of students for equal halves
  : ∃ x : ℕ, 
    x * (total_students / 2) + 4 * (total_students / 2) = total_artworks ∧ 
    x = 3 := by
  sorry

end art_class_problem_l2709_270975


namespace negation_equivalence_l2709_270983

theorem negation_equivalence : 
  ¬(∀ x : ℝ, (x ≠ 3 ∧ x ≠ 2) → (x^2 - 5*x + 6 ≠ 0)) ↔ 
  (∀ x : ℝ, (x = 3 ∨ x = 2) → (x^2 - 5*x + 6 = 0)) :=
by sorry

end negation_equivalence_l2709_270983


namespace geometric_sequence_sum_l2709_270999

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_sum (a₁ r : ℝ) (h₁ : a₁ = 1) (h₂ : r = -3) :
  let a := geometric_sequence a₁ r
  a 1 + |a 2| + a 3 + |a 4| + a 5 = 121 := by
  sorry

end geometric_sequence_sum_l2709_270999
