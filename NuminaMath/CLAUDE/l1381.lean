import Mathlib

namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l1381_138127

/-- The parabola y² = 8x with a point P at distance 9 from its focus -/
theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 8*x →                        -- P lies on the parabola
  (x - 2)^2 + y^2 = 9^2 →            -- Distance from P to focus (2, 0) is 9
  x = 7 ∧ y^2 = 56 := by             -- Coordinates of P are (7, ±2√14)
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l1381_138127


namespace NUMINAMATH_CALUDE_smallest_enclosing_circle_l1381_138159

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := y - 2 = 0
def line3 (x y : ℝ) : Prop := x + y - 4 = 0

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  line1 A.1 A.2 ∧ line2 A.1 A.2 ∧
  line2 B.1 B.2 ∧ line3 B.1 B.2 ∧
  line1 C.1 C.2 ∧ line3 C.1 C.2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1.5)^2 = 1.25

-- Theorem statement
theorem smallest_enclosing_circle 
  (A B C : ℝ × ℝ) 
  (h : triangle A B C) :
  ∀ x y : ℝ, 
  (∀ px py : ℝ, (px = A.1 ∧ py = A.2) ∨ (px = B.1 ∧ py = B.2) ∨ (px = C.1 ∧ py = C.2) → 
    (x - px)^2 + (y - py)^2 ≤ 1.25) ↔ 
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_smallest_enclosing_circle_l1381_138159


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l1381_138122

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + 5 * x

-- Theorem 1: Solution set of f(x) ≤ 5x + 3 when a = -1
theorem solution_set_theorem :
  {x : ℝ | |x + 1| + 5 * x ≤ 5 * x + 3} = Set.Icc (-4) 2 := by sorry

-- Theorem 2: Range of a when f(x) ≥ 0 for x ≥ -1
theorem range_of_a_theorem :
  (∀ x ≥ -1, f a x ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_a_theorem_l1381_138122


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1381_138118

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 > 0 → speed2 > 0 → (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km in the first hour and 60 km in the second hour is 75 km/h -/
theorem car_average_speed : 
  let speed1 := 90
  let speed2 := 60
  (speed1 + speed2) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1381_138118


namespace NUMINAMATH_CALUDE_sin_cos_equality_l1381_138194

theorem sin_cos_equality (θ : Real) (h : Real.sin θ * Real.cos θ = 1/2) :
  Real.sin θ - Real.cos θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_l1381_138194


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_seven_l1381_138155

theorem product_of_repeating_decimal_and_seven :
  ∃ (s : ℚ), (s = 456 / 999) ∧ (s * 7 = 118 / 37) := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_seven_l1381_138155


namespace NUMINAMATH_CALUDE_subsets_exist_l1381_138173

/-- A type representing a set of subsets of positive integers -/
def SubsetCollection := Finset (Set ℕ+)

/-- A function that constructs the required subsets -/
def constructSubsets (n : ℕ) : SubsetCollection :=
  sorry

/-- Predicate to check if subsets are pairwise nonintersecting -/
def pairwiseNonintersecting (s : SubsetCollection) : Prop :=
  sorry

/-- Predicate to check if all subsets are nonempty -/
def allNonempty (s : SubsetCollection) : Prop :=
  sorry

/-- Predicate to check if each positive integer can be uniquely expressed
    as a sum of at most n integers from different subsets -/
def uniqueRepresentation (s : SubsetCollection) (n : ℕ) : Prop :=
  sorry

/-- The main theorem stating the existence of the required subsets -/
theorem subsets_exist (n : ℕ) (h : n ≥ 2) :
  ∃ s : SubsetCollection,
    s.card = n ∧
    pairwiseNonintersecting s ∧
    allNonempty s ∧
    uniqueRepresentation s n :=
  sorry

end NUMINAMATH_CALUDE_subsets_exist_l1381_138173


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l1381_138104

theorem inscribed_rectangle_circle_circumference 
  (width : Real) (height : Real) (circle : Real → Prop) 
  (rectangle : Real → Real → Prop) (circumference : Real) :
  width = 9 →
  height = 12 →
  rectangle width height →
  (∀ x y, rectangle x y → circle (Real.sqrt (x^2 + y^2))) →
  circumference = Real.pi * Real.sqrt (width^2 + height^2) →
  circumference = 15 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l1381_138104


namespace NUMINAMATH_CALUDE_shoveling_time_l1381_138129

theorem shoveling_time (kevin dave john allison : ℝ)
  (h_kevin : kevin = 12)
  (h_dave : dave = 8)
  (h_john : john = 6)
  (h_allison : allison = 4) :
  (1 / kevin + 1 / dave + 1 / john + 1 / allison)⁻¹ * 60 = 96 := by
  sorry

end NUMINAMATH_CALUDE_shoveling_time_l1381_138129


namespace NUMINAMATH_CALUDE_writer_tea_and_hours_l1381_138172

structure WriterData where
  sunday_hours : ℝ
  sunday_tea : ℝ
  wednesday_hours : ℝ
  thursday_tea : ℝ

def inverse_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

theorem writer_tea_and_hours (data : WriterData) :
  inverse_proportional data.sunday_hours data.sunday_tea (data.sunday_hours * data.sunday_tea) →
  inverse_proportional data.wednesday_hours (data.sunday_hours * data.sunday_tea / data.wednesday_hours) (data.sunday_hours * data.sunday_tea) ∧
  inverse_proportional (data.sunday_hours * data.sunday_tea / data.thursday_tea) data.thursday_tea (data.sunday_hours * data.sunday_tea) :=
by
  sorry

#check writer_tea_and_hours

end NUMINAMATH_CALUDE_writer_tea_and_hours_l1381_138172


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1381_138120

/-- A geometric sequence with its sum and common ratio -/
structure GeometricSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  q : ℝ
  sum_formula : ∀ n : ℕ+, S n = (a 1) * (1 - q^n.val) / (1 - q)
  term_formula : ∀ n : ℕ+, a n = (a 1) * q^(n.val - 1)

/-- The theorem stating the general term of the specific geometric sequence -/
theorem geometric_sequence_general_term 
  (seq : GeometricSequence) 
  (h1 : seq.S 3 = 14) 
  (h2 : seq.q = 2) :
  ∀ n : ℕ+, seq.a n = 2^n.val :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1381_138120


namespace NUMINAMATH_CALUDE_stairs_height_l1381_138195

theorem stairs_height (h : ℝ) 
  (total_height : 3 * h + h / 2 + (h / 2 + 10) = 70) : h = 15 := by
  sorry

end NUMINAMATH_CALUDE_stairs_height_l1381_138195


namespace NUMINAMATH_CALUDE_bride_groom_age_sum_l1381_138190

theorem bride_groom_age_sum :
  ∀ (groom_age bride_age : ℕ),
    groom_age = 83 →
    bride_age = groom_age + 19 →
    groom_age + bride_age = 185 :=
by
  sorry

end NUMINAMATH_CALUDE_bride_groom_age_sum_l1381_138190


namespace NUMINAMATH_CALUDE_course_choice_related_to_gender_l1381_138157

-- Define the contingency table
def contingency_table := (40, 10, 30, 20)

-- Define the total number of students
def total_students : Nat := 100

-- Define the critical value for α = 0.05
def critical_value : Float := 3.841

-- Function to calculate χ²
def calculate_chi_square (a b c d : Nat) : Float :=
  let n := a + b + c + d
  let numerator := n * (a * d - b * c) ^ 2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  numerator.toFloat / denominator.toFloat

-- Theorem statement
theorem course_choice_related_to_gender (a b c d : Nat) 
  (h1 : (a, b, c, d) = contingency_table) 
  (h2 : a + b + c + d = total_students) : 
  calculate_chi_square a b c d > critical_value :=
by
  sorry


end NUMINAMATH_CALUDE_course_choice_related_to_gender_l1381_138157


namespace NUMINAMATH_CALUDE_least_common_duration_l1381_138191

/-- Represents a business partner -/
structure Partner where
  investment : ℚ
  duration : ℕ

/-- Represents the business venture -/
structure BusinessVenture where
  p : Partner
  q : Partner
  r : Partner
  investmentRatio : Fin 3 → ℚ
  profitRatio : Fin 3 → ℚ

/-- The profit is proportional to the product of investment and duration -/
def profitProportional (bv : BusinessVenture) : Prop :=
  ∃ (k : ℚ), k > 0 ∧
    bv.profitRatio 0 = k * bv.p.investment * bv.p.duration ∧
    bv.profitRatio 1 = k * bv.q.investment * bv.q.duration ∧
    bv.profitRatio 2 = k * bv.r.investment * bv.r.duration

/-- The main theorem -/
theorem least_common_duration (bv : BusinessVenture) 
    (h1 : bv.investmentRatio = ![7, 5, 3])
    (h2 : bv.profitRatio = ![7, 10, 6])
    (h3 : bv.p.duration = 8)
    (h4 : bv.q.duration = 6)
    (h5 : profitProportional bv) :
    bv.r.duration = 6 := by
  sorry

end NUMINAMATH_CALUDE_least_common_duration_l1381_138191


namespace NUMINAMATH_CALUDE_sin_pi_half_plus_two_alpha_l1381_138162

theorem sin_pi_half_plus_two_alpha (y₀ : ℝ) (α : ℝ) : 
  (1/2)^2 + y₀^2 = 1 → 
  Real.cos α = 1/2 →
  Real.sin (π/2 + 2*α) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_half_plus_two_alpha_l1381_138162


namespace NUMINAMATH_CALUDE_A_symmetric_to_B_about_x_axis_l1381_138184

/-- Two points are symmetric about the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other. -/
def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Point A with coordinates (3, 2) -/
def A : ℝ × ℝ := (3, 2)

/-- Point B with coordinates (3, -2) -/
def B : ℝ × ℝ := (3, -2)

/-- Theorem stating that point A is symmetric to point B about the x-axis -/
theorem A_symmetric_to_B_about_x_axis : symmetric_about_x_axis A B := by
  sorry

end NUMINAMATH_CALUDE_A_symmetric_to_B_about_x_axis_l1381_138184


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_no_sum_2014_l1381_138123

def geometric_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2  -- a_1 = 2
  | n + 1 => geometric_sequence n + 2^n

theorem geometric_sequence_general_term :
  ∀ n : ℕ, geometric_sequence n = 2^n :=
sorry

theorem no_sum_2014 :
  ¬ ∃ p q : ℕ, p < q ∧ geometric_sequence p + geometric_sequence q = 2014 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_no_sum_2014_l1381_138123


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1381_138108

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - ((x-4)²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - (x - 4)^2 / M = 1) →
  (∀ x y, y = (4/3) * x ↔ y = (5/Real.sqrt M) * (x - 4)) →
  M = 225 / 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1381_138108


namespace NUMINAMATH_CALUDE_sum_largest_triangles_geq_twice_area_l1381_138179

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  -- This is a simplified representation
  vertices : Set ℝ × ℝ
  is_convex : sorry

/-- The area of a polygon -/
def area (P : ConvexPolygon) : ℝ := sorry

/-- The largest triangle area for a given side of the polygon -/
def largest_triangle_area (P : ConvexPolygon) (side : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- The sum of largest triangle areas for all sides of the polygon -/
def sum_largest_triangle_areas (P : ConvexPolygon) : ℝ := sorry

/-- Theorem: The sum of the areas of the largest triangles formed within P, 
    each having one side coinciding with a side of P, 
    is at least twice the area of P -/
theorem sum_largest_triangles_geq_twice_area (P : ConvexPolygon) :
  sum_largest_triangle_areas P ≥ 2 * area P := by sorry

end NUMINAMATH_CALUDE_sum_largest_triangles_geq_twice_area_l1381_138179


namespace NUMINAMATH_CALUDE_bill_difference_l1381_138136

theorem bill_difference (christine_tip : ℝ) (christine_percent : ℝ)
  (alex_tip : ℝ) (alex_percent : ℝ) :
  christine_tip = 3 →
  christine_percent = 15 →
  alex_tip = 4 →
  alex_percent = 10 →
  christine_tip = (christine_percent / 100) * christine_bill →
  alex_tip = (alex_percent / 100) * alex_bill →
  alex_bill - christine_bill = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l1381_138136


namespace NUMINAMATH_CALUDE_glass_bowl_problem_l1381_138100

/-- The initial rate per bowl given the conditions of the glass bowl sales problem -/
def initial_rate_per_bowl (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) : ℚ :=
  (sold_bowls * selling_price) / (total_bowls * (1 + percentage_gain / 100))

theorem glass_bowl_problem :
  let total_bowls : ℕ := 114
  let sold_bowls : ℕ := 108
  let selling_price : ℚ := 17
  let percentage_gain : ℚ := 23.88663967611336
  abs (initial_rate_per_bowl total_bowls sold_bowls selling_price percentage_gain - 13) < 0.01 := by
  sorry

#eval initial_rate_per_bowl 114 108 17 23.88663967611336

end NUMINAMATH_CALUDE_glass_bowl_problem_l1381_138100


namespace NUMINAMATH_CALUDE_test_probabilities_l1381_138177

theorem test_probabilities (p_first : ℝ) (p_second : ℝ) (p_both : ℝ) 
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_both = 0.45) :
  1 - (p_first + p_second - p_both) = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_test_probabilities_l1381_138177


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1381_138178

/-- The area of the quadrilateral formed by three coplanar squares -/
theorem quadrilateral_area (s₁ s₂ s₃ : ℝ) (hs₁ : s₁ = 3) (hs₂ : s₂ = 5) (hs₃ : s₃ = 7) : 
  let h₁ := s₁ * (s₃ / (s₁ + s₂ + s₃))
  let h₂ := (s₁ + s₂) * (s₃ / (s₁ + s₂ + s₃))
  (h₁ + h₂) * s₂ / 2 = 12.825 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1381_138178


namespace NUMINAMATH_CALUDE_f_composition_value_l1381_138142

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then |x - 1| - 2 else 1 / (1 + x^2)

theorem f_composition_value : f (f 3) = -11/10 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1381_138142


namespace NUMINAMATH_CALUDE_parabola_shift_l1381_138101

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- The original parabola y = -3x^2 -/
def original_parabola : Parabola :=
  { f := fun x => -3 * x^2 }

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { f := fun x => p.f x + v }

/-- The final parabola after shifting -/
def final_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 5) 2

theorem parabola_shift :
  final_parabola.f = fun x => -3 * (x - 5)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l1381_138101


namespace NUMINAMATH_CALUDE_gratuities_calculation_l1381_138146

/-- Calculates the gratuities charged by a restaurant given the total bill, tax rate, and item costs. -/
def calculate_gratuities (total_bill : ℚ) (tax_rate : ℚ) (striploin_cost : ℚ) (wine_cost : ℚ) : ℚ :=
  let bill_before_tax := striploin_cost + wine_cost
  let sales_tax := bill_before_tax * tax_rate
  let bill_with_tax := bill_before_tax + sales_tax
  total_bill - bill_with_tax

/-- Theorem stating that the gratuities charged equals $41 given the problem conditions. -/
theorem gratuities_calculation :
  calculate_gratuities 140 (1/10) 80 10 = 41 := by
  sorry

#eval calculate_gratuities 140 (1/10) 80 10

end NUMINAMATH_CALUDE_gratuities_calculation_l1381_138146


namespace NUMINAMATH_CALUDE_common_ratio_sum_l1381_138140

theorem common_ratio_sum (k p r : ℝ) (h1 : k ≠ 0) (h2 : p ≠ 1) (h3 : r ≠ 1) (h4 : p ≠ r) 
  (h5 : k * p^2 - k * r^2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_sum_l1381_138140


namespace NUMINAMATH_CALUDE_g_divisibility_l1381_138117

def g : ℕ → ℕ
  | 0 => 1
  | n + 1 => g n ^ 2 + g n + 1

theorem g_divisibility (n : ℕ) : 
  (g n ^ 2 + 1) ∣ (g (n + 1) ^ 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_g_divisibility_l1381_138117


namespace NUMINAMATH_CALUDE_scientific_notation_of_15000_l1381_138145

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_15000 :
  toScientificNotation 15000 = ScientificNotation.mk 1.5 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_15000_l1381_138145


namespace NUMINAMATH_CALUDE_square_with_1983_nines_l1381_138171

theorem square_with_1983_nines : ∃ N : ℕ,
  (N^2 = 10 * 88) ∧
  (∃ k : ℕ, N = 10^1984 - 1 + k ∧ k < 10^1984) := by
  sorry

end NUMINAMATH_CALUDE_square_with_1983_nines_l1381_138171


namespace NUMINAMATH_CALUDE_min_students_with_all_characteristics_l1381_138169

theorem min_students_with_all_characteristics
  (total : ℕ) (blue_eyes : ℕ) (lunch_box : ℕ) (glasses : ℕ)
  (h_total : total = 35)
  (h_blue_eyes : blue_eyes = 15)
  (h_lunch_box : lunch_box = 25)
  (h_glasses : glasses = 10) :
  ∃ (n : ℕ), n ≥ 1 ∧ n ≤ min blue_eyes (min lunch_box glasses) ∧
    n ≥ blue_eyes + lunch_box + glasses - 2 * total :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_all_characteristics_l1381_138169


namespace NUMINAMATH_CALUDE_problem_solution_l1381_138161

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |x + a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1) ∧
  (∃ x : ℝ, f a x ≤ |a - 1| ↔ a ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1381_138161


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1381_138158

/-- Given a and b are real numbers satisfying a + bi = (1 + i)i^3, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) (h : (↑a + ↑b * I) = (1 + I) * I^3) : a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1381_138158


namespace NUMINAMATH_CALUDE_max_sides_in_subdivision_l1381_138132

/-- 
Given a convex polygon with n sides and all its diagonals drawn,
the maximum number of sides a polygon in the subdivision can have is n.
-/
theorem max_sides_in_subdivision (n : ℕ) (h : n ≥ 3) :
  ∃ (max_sides : ℕ), max_sides = n ∧ 
  ∀ (subdivided_polygon_sides : ℕ), 
    subdivided_polygon_sides ≤ max_sides :=
by sorry

end NUMINAMATH_CALUDE_max_sides_in_subdivision_l1381_138132


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l1381_138149

theorem trapezium_other_side_length 
  (a : ℝ) -- Area of the trapezium
  (b : ℝ) -- Length of one parallel side
  (h : ℝ) -- Distance between parallel sides
  (x : ℝ) -- Length of the other parallel side
  (h1 : a = 380) -- Area is 380 square centimeters
  (h2 : b = 18)  -- One parallel side is 18 cm
  (h3 : h = 20)  -- Distance between parallel sides is 20 cm
  (h4 : a = (1/2) * (x + b) * h) -- Area formula for trapezium
  : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l1381_138149


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l1381_138143

/-- Given the costs of different combinations of pencils and pens, 
    calculate the cost of three pencils and three pens. -/
theorem pencil_pen_cost (pencil pen : ℝ) 
  (h1 : 3 * pencil + 2 * pen = 3.60)
  (h2 : 2 * pencil + 3 * pen = 3.15) :
  3 * pencil + 3 * pen = 4.05 := by
  sorry


end NUMINAMATH_CALUDE_pencil_pen_cost_l1381_138143


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1381_138135

theorem inequality_solution_set : 
  {x : ℝ | x^2 + x - 2 > 0} = {x : ℝ | x < -2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1381_138135


namespace NUMINAMATH_CALUDE_min_value_m_plus_2n_l1381_138196

/-- The function f(x) = |x-a| where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- The theorem stating the minimum value of m + 2n -/
theorem min_value_m_plus_2n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f 2 x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3) →
  1/m + 1/(2*n) = 2 →
  ∀ k l, k > 0 → l > 0 → 1/k + 1/(2*l) = 2 → m + 2*n ≤ k + 2*l :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_plus_2n_l1381_138196


namespace NUMINAMATH_CALUDE_negation_of_all_x_squared_positive_negation_is_true_l1381_138148

theorem negation_of_all_x_squared_positive :
  (¬ (∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

theorem negation_is_true : ∃ x : ℝ, x^2 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_x_squared_positive_negation_is_true_l1381_138148


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1381_138188

/-- Given a hyperbola C and an ellipse with the following properties:
    - The general equation of C is (x²/a²) - (y²/b²) = 1 where a > 0 and b > 0
    - C has an asymptote equation y = (√5/2)x
    - C shares a common focus with the ellipse x²/12 + y²/3 = 1
    Then, the specific equation of hyperbola C is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2) ∧ 
  (b / a = Real.sqrt 5 / 2) ∧
  (c^2 = 3^2) →
  a^2 = 4 ∧ b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1381_138188


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_formula_arithmetic_sequence_possible_formulas_l1381_138112

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_unique_formula 
  (a₁ d : ℤ) 
  (h1 : arithmetic_sequence a₁ d 11 = 0) 
  (h2 : sum_arithmetic_sequence a₁ d 14 = 98) :
  ∀ n : ℕ, arithmetic_sequence a₁ d n = 22 - 2 * n :=
sorry

theorem arithmetic_sequence_possible_formulas 
  (a₁ d : ℤ) 
  (h1 : a₁ ≥ 6) 
  (h2 : arithmetic_sequence a₁ d 11 > 0) 
  (h3 : sum_arithmetic_sequence a₁ d 14 ≤ 77) :
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 12 - n) ∨ 
  (∀ n : ℕ, arithmetic_sequence a₁ d n = 13 - n) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_formula_arithmetic_sequence_possible_formulas_l1381_138112


namespace NUMINAMATH_CALUDE_exist_a_b_satisfying_conditions_l1381_138153

theorem exist_a_b_satisfying_conditions : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧ a * b * (a - b) = 1 ∧ a^2 + b^2 = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_exist_a_b_satisfying_conditions_l1381_138153


namespace NUMINAMATH_CALUDE_largest_n_with_special_divisor_property_l1381_138128

theorem largest_n_with_special_divisor_property : ∃ N : ℕ, 
  (∀ m : ℕ, m > N → ¬(∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ m ∧ d₂ ∣ m ∧ d₃ ∣ m ∧
    (∀ x : ℕ, x ∣ m → x = 1 ∨ x ≥ d₁) ∧
    (∀ x : ℕ, x ∣ m → x = 1 ∨ x = d₁ ∨ x ≥ d₂) ∧
    (∃ y z : ℕ, y ∣ m ∧ z ∣ m ∧ y > d₃ ∧ z > y) ∧
    d₃ = 21 * d₁)) ∧
  (∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ N ∧ d₂ ∣ N ∧ d₃ ∣ N ∧
    (∀ x : ℕ, x ∣ N → x = 1 ∨ x ≥ d₁) ∧
    (∀ x : ℕ, x ∣ N → x = 1 ∨ x = d₁ ∨ x ≥ d₂) ∧
    (∃ y z : ℕ, y ∣ N ∧ z ∣ N ∧ y > d₃ ∧ z > y) ∧
    d₃ = 21 * d₁) ∧
  N = 441 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_with_special_divisor_property_l1381_138128


namespace NUMINAMATH_CALUDE_divisibility_property_l1381_138116

theorem divisibility_property (n a b c d : ℤ) 
  (hn : n > 0)
  (h1 : n ∣ (a + b + c + d))
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) :
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1381_138116


namespace NUMINAMATH_CALUDE_f_3_equals_7_l1381_138107

-- Define the function f
def f : ℝ → ℝ := fun x => 2*x + 1

-- State the theorem
theorem f_3_equals_7 : f 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_7_l1381_138107


namespace NUMINAMATH_CALUDE_ag_replacement_terminates_l1381_138119

/-- Represents a sequence of As and Gs -/
inductive AGSequence
| empty : AGSequence
| cons : Char → AGSequence → AGSequence

/-- Represents the operation of replacing "AG" with "GAAA" -/
def replaceAG (s : AGSequence) : AGSequence :=
  sorry

/-- Predicate to check if a sequence contains "AG" -/
def containsAG (s : AGSequence) : Prop :=
  sorry

/-- The main theorem stating that the process will eventually terminate -/
theorem ag_replacement_terminates (initial : AGSequence) :
  ∃ (n : ℕ) (final : AGSequence), (∀ k, k ≥ n → replaceAG^[k] initial = final) ∧ ¬containsAG final :=
  sorry

end NUMINAMATH_CALUDE_ag_replacement_terminates_l1381_138119


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1381_138139

/-- The complex number z -/
def z : ℂ := (2 - Complex.I) ^ 2

/-- Theorem: The point corresponding to z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1381_138139


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l1381_138187

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, -3]

theorem vector_addition_and_scalar_multiplication :
  (a + 2 • b) = ![4, -5] := by sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l1381_138187


namespace NUMINAMATH_CALUDE_max_area_rectangle_max_area_520_perimeter_l1381_138113

/-- The perimeter of the rectangle in meters -/
def perimeter : ℝ := 520

/-- Theorem: Maximum area of a rectangle with given perimeter -/
theorem max_area_rectangle (l w : ℝ) (h1 : l > 0) (h2 : w > 0) (h3 : 2 * l + 2 * w = perimeter) :
  l * w ≤ (perimeter / 4) ^ 2 :=
sorry

/-- Corollary: The maximum area of a rectangle with perimeter 520 meters is 16900 square meters -/
theorem max_area_520_perimeter :
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * l + 2 * w = perimeter ∧ l * w = 16900 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_max_area_520_perimeter_l1381_138113


namespace NUMINAMATH_CALUDE_students_liking_both_soda_and_coke_l1381_138175

/-- Given a school with the following conditions:
  - Total number of students: 500
  - Students who like soda: 337
  - Students who like coke: 289
  - Students who neither like soda nor coke: 56
  Prove that the number of students who like both soda and coke is 182. -/
theorem students_liking_both_soda_and_coke 
  (total : ℕ) (soda : ℕ) (coke : ℕ) (neither : ℕ) 
  (h_total : total = 500)
  (h_soda : soda = 337)
  (h_coke : coke = 289)
  (h_neither : neither = 56) :
  soda + coke - total + neither = 182 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_soda_and_coke_l1381_138175


namespace NUMINAMATH_CALUDE_line_AB_equation_l1381_138198

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 2.5)^2 + (y - 0.5)^2 = 2.5

-- Define point P
def P : ℝ × ℝ := (4, 1)

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 3*x + y - 4 = 0

-- Theorem statement
theorem line_AB_equation :
  ∀ x y : ℝ,
  (circle1 x y ∧ circle2 x y) →
  lineAB x y :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_l1381_138198


namespace NUMINAMATH_CALUDE_ellipse_and_tangent_circle_l1381_138121

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the circle tangent to line l -/
def tangent_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4 / 3

/-- Theorem statement -/
theorem ellipse_and_tangent_circle :
  ∀ (x y : ℝ),
  -- Conditions
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ a^2 - b^2 = 1) →  -- Ellipse properties
  (1^2 / 4 + (3/2)^2 / 3 = 1) →  -- Point (1, 3/2) lies on C
  (∃ (m : ℝ), m^2 = 2) →  -- Slope of line l
  -- Conclusions
  (ellipse_C x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (tangent_circle x y ↔ (x + 1)^2 + y^2 = 4 / 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_tangent_circle_l1381_138121


namespace NUMINAMATH_CALUDE_tree_planting_problem_l1381_138144

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Calculates the number of trees that can be planted along a triangle's perimeter -/
def treesAlongPerimeter (t : Triangle) (treeSpacing : ℕ) : ℕ :=
  (t.side1 + t.side2 + t.side3) / treeSpacing

theorem tree_planting_problem :
  let triangle := Triangle.mk 198 180 210
  let treeSpacing := 6
  treesAlongPerimeter triangle treeSpacing = 98 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l1381_138144


namespace NUMINAMATH_CALUDE_probability_queens_or_aces_l1381_138154

/-- Represents a standard deck of 52 cards -/
def standard_deck : ℕ := 52

/-- Number of queens in a standard deck -/
def num_queens : ℕ := 4

/-- Number of aces in a standard deck -/
def num_aces : ℕ := 4

/-- Number of cards drawn -/
def cards_drawn : ℕ := 3

/-- Probability of drawing all queens or at least 2 aces -/
def prob_queens_or_aces : ℚ := 220 / 581747

theorem probability_queens_or_aces :
  let total_ways := standard_deck.choose cards_drawn
  let ways_all_queens := num_queens.choose cards_drawn
  let ways_two_aces := cards_drawn.choose 2 * num_aces.choose 2 * (standard_deck - num_aces)
  let ways_three_aces := num_aces.choose cards_drawn
  (ways_all_queens + ways_two_aces + ways_three_aces : ℚ) / total_ways = prob_queens_or_aces := by
  sorry

end NUMINAMATH_CALUDE_probability_queens_or_aces_l1381_138154


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1381_138103

theorem simplify_fraction_product : (222 : ℚ) / 999 * 111 = 74 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1381_138103


namespace NUMINAMATH_CALUDE_go_kart_tickets_value_l1381_138189

/-- The number of tickets required for a go-kart ride -/
def go_kart_tickets : ℕ := sorry

/-- The number of times Paula rides the go-karts -/
def go_kart_rides : ℕ := 1

/-- The number of times Paula rides the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The number of tickets required for a bumper car ride -/
def bumper_car_tickets : ℕ := 5

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := 24

theorem go_kart_tickets_value : 
  go_kart_tickets = 4 :=
by sorry

end NUMINAMATH_CALUDE_go_kart_tickets_value_l1381_138189


namespace NUMINAMATH_CALUDE_total_liquid_drunk_l1381_138131

/-- Converts pints to cups -/
def pints_to_cups (pints : ℝ) : ℝ := 2 * pints

/-- The amount of coffee Elijah drank in pints -/
def elijah_coffee : ℝ := 8.5

/-- The amount of water Emilio drank in pints -/
def emilio_water : ℝ := 9.5

/-- Theorem: The total amount of liquid drunk by Elijah and Emilio is 36 cups -/
theorem total_liquid_drunk : 
  pints_to_cups elijah_coffee + pints_to_cups emilio_water = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_liquid_drunk_l1381_138131


namespace NUMINAMATH_CALUDE_student_group_allocation_schemes_l1381_138115

theorem student_group_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 12) 
  (h2 : k = 4) 
  (h3 : m = 3) 
  (h4 : n = k * m) : 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m * m^k : ℕ) = 
  (Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 6 3 * 3^4 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_student_group_allocation_schemes_l1381_138115


namespace NUMINAMATH_CALUDE_X_mod_100_l1381_138186

/-- The number of sequences satisfying the given conditions -/
def X : ℕ := sorry

/-- Condition: Each aᵢ is either 0 or a power of 2 -/
def is_valid_element (a : ℕ) : Prop :=
  a = 0 ∨ ∃ k : ℕ, a = 2^k

/-- Condition: aᵢ = a₂ᵢ + a₂ᵢ₊₁ for 1 ≤ i ≤ 1023 -/
def satisfies_sum_condition (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 1023 → a i = a (2*i) + a (2*i + 1)

/-- All conditions for the sequence -/
def valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2047 → is_valid_element (a i)) ∧
  satisfies_sum_condition a ∧
  a 1 = 1024

theorem X_mod_100 : X % 100 = 15 := by sorry

end NUMINAMATH_CALUDE_X_mod_100_l1381_138186


namespace NUMINAMATH_CALUDE_alan_tickets_l1381_138174

theorem alan_tickets (total : ℕ) (alan : ℕ) (marcy : ℕ) 
  (h1 : total = 150)
  (h2 : alan + marcy = total)
  (h3 : marcy = 5 * alan - 6) :
  alan = 26 := by
sorry

end NUMINAMATH_CALUDE_alan_tickets_l1381_138174


namespace NUMINAMATH_CALUDE_unique_a_value_l1381_138106

def A (a : ℝ) : Set ℝ := {2, 3, a^2 - 3*a, a + 2/a + 7}
def B (a : ℝ) : Set ℝ := {|a - 2|, 3}

theorem unique_a_value : ∃! a : ℝ, (4 ∈ A a) ∧ (4 ∉ B a) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1381_138106


namespace NUMINAMATH_CALUDE_paint_together_l1381_138176

/-- The amount of wall Heidi and Tom can paint together in a given time -/
def wall_painted (heidi_time tom_time paint_time : ℚ) : ℚ :=
  paint_time * (1 / heidi_time + 1 / tom_time)

/-- Theorem: Heidi and Tom can paint 5/12 of the wall in 15 minutes -/
theorem paint_together : wall_painted 60 90 15 = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_paint_together_l1381_138176


namespace NUMINAMATH_CALUDE_greatest_k_value_l1381_138192

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1381_138192


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1381_138114

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_square_sum (n : ℕ) : ℕ := (n / 10)^2 + (n % 10)^2

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
            n / digit_sum n = 3 ∧
            n % digit_sum n = 7 ∧
            digit_square_sum n - digit_product n = n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l1381_138114


namespace NUMINAMATH_CALUDE_surface_area_of_cube_structure_l1381_138111

/-- Calculates the surface area of a cube given its side length -/
def cubeSurfaceArea (sideLength : ℝ) : ℝ := 6 * sideLength^2

/-- The structure composed of cubes -/
structure CubeStructure where
  largeCubeSideLength : ℝ
  mediumCubeSideLength : ℝ
  smallCubeSideLength : ℝ
  mediumCubeCount : ℕ
  smallCubeCount : ℕ

/-- Calculates the total surface area of the cube structure -/
def totalSurfaceArea (cs : CubeStructure) : ℝ :=
  cubeSurfaceArea cs.largeCubeSideLength +
  cs.mediumCubeCount * cubeSurfaceArea cs.mediumCubeSideLength +
  cs.smallCubeCount * cubeSurfaceArea cs.smallCubeSideLength

/-- The theorem stating that the total surface area of the given structure is 270 square centimeters -/
theorem surface_area_of_cube_structure :
  let cs : CubeStructure := {
    largeCubeSideLength := 5,
    mediumCubeSideLength := 2,
    smallCubeSideLength := 1,
    mediumCubeCount := 4,
    smallCubeCount := 4
  }
  totalSurfaceArea cs = 270 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_cube_structure_l1381_138111


namespace NUMINAMATH_CALUDE_rachel_apples_l1381_138167

def initial_apples (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : ℕ :=
  num_trees * apples_per_tree + remaining_apples

theorem rachel_apples : initial_apples 3 8 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apples_l1381_138167


namespace NUMINAMATH_CALUDE_angle_complement_supplement_difference_l1381_138163

theorem angle_complement_supplement_difference : 
  ∀ α : ℝ, (90 - α) - (180 - α) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_difference_l1381_138163


namespace NUMINAMATH_CALUDE_ticket_123123123_is_red_l1381_138130

/-- Represents the color of a lottery ticket -/
inductive TicketColor
| Red
| Blue
| Green

/-- Represents a 9-digit lottery ticket number -/
def TicketNumber := Fin 9 → Fin 3

/-- The coloring function for tickets -/
def ticketColor : TicketNumber → TicketColor := sorry

/-- Check if two tickets differ in all places -/
def differInAllPlaces (t1 t2 : TicketNumber) : Prop :=
  ∀ i : Fin 9, t1 i ≠ t2 i

/-- The main theorem to prove -/
theorem ticket_123123123_is_red :
  (∀ t1 t2 : TicketNumber, differInAllPlaces t1 t2 → ticketColor t1 ≠ ticketColor t2) →
  ticketColor (λ i => if i.val % 3 = 0 then 0 else if i.val % 3 = 1 then 1 else 2) = TicketColor.Red →
  ticketColor (λ _ => 1) = TicketColor.Green →
  ticketColor (λ i => i.val % 3) = TicketColor.Red :=
sorry

end NUMINAMATH_CALUDE_ticket_123123123_is_red_l1381_138130


namespace NUMINAMATH_CALUDE_translation_result_l1381_138151

def translate_point (x y dx dy : Int) : (Int × Int) :=
  (x + dx, y - dy)

theorem translation_result :
  let initial_point := (-2, 3)
  let x_translation := 3
  let y_translation := 2
  translate_point initial_point.1 initial_point.2 x_translation y_translation = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l1381_138151


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l1381_138110

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes. -/
theorem ellipse_hyperbola_semi_axes_product : 
  ∀ (a b : ℝ), 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = 2 * Real.sqrt 111 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l1381_138110


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1381_138150

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1381_138150


namespace NUMINAMATH_CALUDE_inverse_sum_modulo_eleven_l1381_138124

theorem inverse_sum_modulo_eleven :
  (((3⁻¹ : ZMod 11) + (5⁻¹ : ZMod 11) + (7⁻¹ : ZMod 11))⁻¹ : ZMod 11) = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_modulo_eleven_l1381_138124


namespace NUMINAMATH_CALUDE_lowest_price_pet_food_l1381_138126

def msrp : ℝ := 45.00
def max_regular_discount : ℝ := 0.30
def additional_discount : ℝ := 0.20

theorem lowest_price_pet_food :
  let regular_discounted_price := msrp * (1 - max_regular_discount)
  let final_price := regular_discounted_price * (1 - additional_discount)
  final_price = 25.20 := by sorry

end NUMINAMATH_CALUDE_lowest_price_pet_food_l1381_138126


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1381_138183

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1381_138183


namespace NUMINAMATH_CALUDE_last_recess_duration_l1381_138170

/- Define the durations of known breaks -/
def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30

/- Define the total time spent outside of class -/
def total_outside_time : ℕ := 80

/- Define the duration of the last recess break -/
def last_recess : ℕ := total_outside_time - (first_recess + second_recess + lunch)

/- Theorem to prove -/
theorem last_recess_duration :
  last_recess = 20 :=
by sorry

end NUMINAMATH_CALUDE_last_recess_duration_l1381_138170


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l1381_138165

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x * (x - 3) ≥ 0}
def B : Set ℝ := {x | x ≤ 2}

-- State the theorem
theorem complement_A_inter_B :
  (Set.compl A) ∩ B = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l1381_138165


namespace NUMINAMATH_CALUDE_parabola_curve_intersection_l1381_138197

/-- A parabola with equation y² = 4x and focus at (1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A curve with equation y = k/x where k > 0 -/
def Curve (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k / p.1 ∧ k > 0}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A point P is perpendicular to the x-axis if its x-coordinate is 1 -/
def isPerpendicular (P : ℝ × ℝ) : Prop :=
  P.1 = 1

theorem parabola_curve_intersection (k : ℝ) :
  ∃ P : ℝ × ℝ, P ∈ Parabola ∧ P ∈ Curve k ∧ isPerpendicular P → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_curve_intersection_l1381_138197


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1381_138181

theorem mathborough_rainfall_2004 (rainfall_2003 rainfall_2004 : ℕ) 
  (h1 : rainfall_2003 = 45)
  (h2 : rainfall_2004 = rainfall_2003 + 3)
  (h3 : ∃ (high_months low_months : ℕ), 
    high_months = 8 ∧ 
    low_months = 12 - high_months ∧
    (high_months * (rainfall_2004 + 5) + low_months * rainfall_2004 = 616)) : 
  rainfall_2004 * 12 + 8 * 5 = 616 := by
  sorry

end NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1381_138181


namespace NUMINAMATH_CALUDE_basketball_store_problem_l1381_138133

/- Define the basketball types -/
inductive BasketballType
| A
| B

/- Define the purchase and selling prices -/
def purchase_price (t : BasketballType) : ℕ :=
  match t with
  | BasketballType.A => 80
  | BasketballType.B => 60

def selling_price (t : BasketballType) : ℕ :=
  match t with
  | BasketballType.A => 120
  | BasketballType.B => 90

/- Define the conditions -/
def condition1 : Prop :=
  20 * purchase_price BasketballType.A + 30 * purchase_price BasketballType.B = 3400

def condition2 : Prop :=
  30 * purchase_price BasketballType.A + 40 * purchase_price BasketballType.B = 4800

def jump_rope_cost : ℕ := 10

/- Define the theorem -/
theorem basketball_store_problem 
  (m n : ℕ) 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : m * selling_price BasketballType.A + n * selling_price BasketballType.B = 5400) :
  (∃ (a b : ℕ), 
    (a * (selling_price BasketballType.A - purchase_price BasketballType.A - jump_rope_cost) + 
     b * (3 * (selling_price BasketballType.B - purchase_price BasketballType.B) - jump_rope_cost) = 600) ∧
    ((a = 12 ∧ b = 3) ∨ (a = 4 ∧ b = 6))) ∧
  (m * (selling_price BasketballType.A - purchase_price BasketballType.A) + 
   n * (selling_price BasketballType.B - purchase_price BasketballType.B) = 1800) :=
by sorry

end NUMINAMATH_CALUDE_basketball_store_problem_l1381_138133


namespace NUMINAMATH_CALUDE_intersection_union_when_m_3_intersection_equals_B_implies_m_range_l1381_138125

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 1}

-- Part 1
theorem intersection_union_when_m_3 :
  (A ∩ B 3) = {x | 1 ≤ x ∧ x ≤ 3} ∧
  (A ∪ B 3) = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2
theorem intersection_equals_B_implies_m_range (m : ℝ) :
  A ∩ B m = B m → 1 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_union_when_m_3_intersection_equals_B_implies_m_range_l1381_138125


namespace NUMINAMATH_CALUDE_exercise_book_count_l1381_138193

/-- Given a ratio of pencils to exercise books and the number of pencils,
    calculate the number of exercise books -/
def calculate_exercise_books (pencil_ratio : ℕ) (book_ratio : ℕ) (num_pencils : ℕ) : ℕ :=
  (num_pencils / pencil_ratio) * book_ratio

/-- Theorem: In a shop with 140 pencils and a pencil to exercise book ratio of 14:3,
    there are 30 exercise books -/
theorem exercise_book_count :
  calculate_exercise_books 14 3 140 = 30 := by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l1381_138193


namespace NUMINAMATH_CALUDE_min_value_of_x_l1381_138134

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 3 ≥ 1 + (1/3) * (Real.log x / Real.log 3)) :
  x ≥ 3 * Real.sqrt 3 ∧ ∀ y : ℝ, y > 0 → Real.log y / Real.log 3 ≥ 1 + (1/3) * (Real.log y / Real.log 3) → y ≥ x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1381_138134


namespace NUMINAMATH_CALUDE_arctan_sum_roots_cubic_l1381_138156

theorem arctan_sum_roots_cubic (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 → 
  x₂^3 - 10*x₂ + 11 = 0 → 
  x₃^3 - 10*x₃ + 11 = 0 → 
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_roots_cubic_l1381_138156


namespace NUMINAMATH_CALUDE_sqrt_defined_for_five_l1381_138141

theorem sqrt_defined_for_five : ∃ (x : ℝ), x = 5 ∧ x - 4 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_defined_for_five_l1381_138141


namespace NUMINAMATH_CALUDE_stating_clock_confusion_times_l1381_138199

/-- Represents the number of degrees the hour hand moves per minute -/
def hourHandSpeed : ℝ := 0.5

/-- Represents the number of degrees the minute hand moves per minute -/
def minuteHandSpeed : ℝ := 6

/-- Represents the total number of degrees in a circle -/
def totalDegrees : ℕ := 360

/-- Represents the number of hours in the given time period -/
def timePeriod : ℕ := 12

/-- Represents the number of times the hands overlap in the given time period -/
def overlapTimes : ℕ := 11

/-- 
  Theorem stating that there are 132 times in a 12-hour period when the clock hands
  can be mistaken for each other, excluding overlaps.
-/
theorem clock_confusion_times : 
  ∃ (confusionTimes : ℕ), 
    confusionTimes = timePeriod * (totalDegrees / (minuteHandSpeed - hourHandSpeed) - 1) - overlapTimes := by
  sorry

end NUMINAMATH_CALUDE_stating_clock_confusion_times_l1381_138199


namespace NUMINAMATH_CALUDE_max_sin_squared_sum_l1381_138180

theorem max_sin_squared_sum (A B C : Real) (a b c : Real) :
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / (Real.sin A) = b / (Real.sin B)) →
  (b / (Real.sin B) = c / (Real.sin C)) →
  ((2 * Real.sin A - Real.sin C) / Real.sin C = (a^2 + b^2 - c^2) / (a^2 + c^2 - b^2)) →
  (∃ (x : Real), x = Real.sin A^2 + Real.sin C^2 ∧ ∀ (y : Real), y = Real.sin A^2 + Real.sin C^2 → y ≤ x) →
  (Real.sin A^2 + Real.sin C^2 ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_sin_squared_sum_l1381_138180


namespace NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l1381_138164

theorem quasi_pythagorean_prime_divisor (a b c : ℕ+) :
  c^2 = a^2 + b^2 + a*b → ∃ p : ℕ, p.Prime ∧ p > 5 ∧ p ∣ c := by
  sorry

end NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l1381_138164


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l1381_138160

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a ≤ 99999 →
    1000 ≤ b ∧ b ≤ 9999 →
    a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l1381_138160


namespace NUMINAMATH_CALUDE_line_slope_equidistant_points_l1381_138138

/-- The slope of a line passing through (4, 4) and equidistant from points (0, 2) and (12, 8) is -2 -/
theorem line_slope_equidistant_points : 
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y - 4 = m * (x - 4) → 
      (x - 0)^2 + (y - 2)^2 = (x - 12)^2 + (y - 8)^2) → 
    m = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_equidistant_points_l1381_138138


namespace NUMINAMATH_CALUDE_max_quotient_value_l1381_138137

theorem max_quotient_value (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) :
  (∀ x' y', 100 ≤ x' ∧ x' ≤ 300 → 900 ≤ y' ∧ y' ≤ 1800 → y' / x' ≤ 18) ∧
  (∃ x' y', 100 ≤ x' ∧ x' ≤ 300 ∧ 900 ≤ y' ∧ y' ≤ 1800 ∧ y' / x' = 18) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1381_138137


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1381_138102

theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Geometric sequence with common ratio 2
  (a 2 + a 4 + a 6 = 3) →       -- Given condition
  (a 5 + a 7 + a 9 = 24) :=     -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1381_138102


namespace NUMINAMATH_CALUDE_intersection_P_T_l1381_138105

def P : Set ℝ := {x | x^2 - x - 2 = 0}
def T : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_P_T : P ∩ T = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_T_l1381_138105


namespace NUMINAMATH_CALUDE_distance_to_village_l1381_138185

theorem distance_to_village (d : ℝ) : 
  (¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 10)) → 
  (d > 7 ∧ d < 8) :=
sorry

end NUMINAMATH_CALUDE_distance_to_village_l1381_138185


namespace NUMINAMATH_CALUDE_somu_age_problem_l1381_138166

/-- Proves that Somu was one-fifth of his father's age 5 years ago -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) :
  somu_age = 10 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 5 := by
  sorry

#check somu_age_problem

end NUMINAMATH_CALUDE_somu_age_problem_l1381_138166


namespace NUMINAMATH_CALUDE_log_expression_1_log_expression_2_l1381_138147

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem for the first expression
theorem log_expression_1 :
  2 * log 3 2 - log 3 (32 / 9) + log 3 8 - (5 : ℝ) ^ (log 5 3) = -1 :=
sorry

-- Theorem for the second expression
theorem log_expression_2 :
  log 2 25 * log 3 4 * log 5 9 = 8 :=
sorry

end NUMINAMATH_CALUDE_log_expression_1_log_expression_2_l1381_138147


namespace NUMINAMATH_CALUDE_distinct_choices_eq_eight_l1381_138152

/-- Represents the set of marbles Tom has -/
inductive Marble : Type
| Red : Marble
| Green : Marble
| Blue : Marble
| Yellow : Marble

/-- The number of each type of marble Tom has -/
def marbleCounts : Marble → ℕ
| Marble.Red => 1
| Marble.Green => 1
| Marble.Blue => 1
| Marble.Yellow => 4

/-- The total number of marbles Tom has -/
def totalMarbles : ℕ := (marbleCounts Marble.Red) + (marbleCounts Marble.Green) + 
                        (marbleCounts Marble.Blue) + (marbleCounts Marble.Yellow)

/-- A function to calculate the number of distinct ways to choose 3 marbles -/
def distinctChoices : ℕ := sorry

/-- Theorem stating that the number of distinct ways to choose 3 marbles is 8 -/
theorem distinct_choices_eq_eight : distinctChoices = 8 := by sorry

end NUMINAMATH_CALUDE_distinct_choices_eq_eight_l1381_138152


namespace NUMINAMATH_CALUDE_octal_253_equals_171_l1381_138182

/-- Converts an octal digit to its decimal representation -/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- The octal representation of the number -/
def octal_number : List Nat := [2, 5, 3]

/-- Converts an octal number to its decimal representation -/
def octal_to_decimal_conversion (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * (8 ^ i)) 0

theorem octal_253_equals_171 :
  octal_to_decimal_conversion octal_number = 171 := by
  sorry

end NUMINAMATH_CALUDE_octal_253_equals_171_l1381_138182


namespace NUMINAMATH_CALUDE_train_length_l1381_138109

theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) : 
  crossing_time = 20 → speed_kmh = 36 → 
  (speed_kmh * 1000 / 3600) * crossing_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1381_138109


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l1381_138168

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem polynomial_value_at_three : f 3 = 925 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l1381_138168
