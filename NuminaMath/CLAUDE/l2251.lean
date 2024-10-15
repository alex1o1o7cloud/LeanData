import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_reflection_l2251_225120

/-- Reflects a point (x, y) about the line y = x -/
def reflect_about_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem circle_center_reflection :
  let original_center : ℝ × ℝ := (8, -3)
  reflect_about_y_equals_x original_center = (-3, 8) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_reflection_l2251_225120


namespace NUMINAMATH_CALUDE_power_product_exponent_l2251_225123

theorem power_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_exponent_l2251_225123


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2251_225175

/-- Represents a triangle XYZ -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Checks if a triangle is isosceles right -/
def isIsoscelesRight (t : Triangle) : Prop := sorry

/-- Calculates the length of a side given two points -/
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem isosceles_right_triangle_area 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : sideLength t.X t.Y > sideLength t.Y t.Z) 
  (h3 : sideLength t.X t.Y = 12.000000000000002) : 
  triangleArea t = 36.000000000000015 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2251_225175


namespace NUMINAMATH_CALUDE_trains_meeting_point_l2251_225131

/-- Proves that two trains traveling towards each other on a 200 km track,
    with train A moving at 60 km/h and train B moving at 90 km/h,
    will meet at a distance of 80 km from train A's starting point. -/
theorem trains_meeting_point (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ)
  (h1 : distance = 200)
  (h2 : speed_A = 60)
  (h3 : speed_B = 90) :
  speed_A * (distance / (speed_A + speed_B)) = 80 :=
by sorry

end NUMINAMATH_CALUDE_trains_meeting_point_l2251_225131


namespace NUMINAMATH_CALUDE_sams_books_l2251_225169

theorem sams_books (joan_books : ℕ) (total_books : ℕ) (h1 : joan_books = 102) (h2 : total_books = 212) :
  total_books - joan_books = 110 := by
  sorry

end NUMINAMATH_CALUDE_sams_books_l2251_225169


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2251_225191

/-- The diagonal length of a rectangular prism with given surface area and total edge length -/
theorem rectangular_prism_diagonal
  (x y z : ℝ)  -- lengths of sides
  (h1 : 2*x*y + 2*x*z + 2*y*z = 22)  -- surface area condition
  (h2 : 4*x + 4*y + 4*z = 24)  -- total edge length condition
  : ∃ d : ℝ, d^2 = x^2 + y^2 + z^2 ∧ d^2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l2251_225191


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l2251_225194

/-- Given an ellipse and a hyperbola with coinciding foci, prove that d^2 = 215/16 -/
theorem ellipse_hyperbola_foci (d : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/d^2 = 1 ↔ x^2/169 - y^2/64 = 1/16) →
  d^2 = 215/16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_l2251_225194


namespace NUMINAMATH_CALUDE_f_properties_l2251_225163

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3) - Real.sin (Real.pi / 2 - x)

theorem f_properties (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : f (α + Real.pi / 6) = 3 / 5) :
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S)) ∧
  f (2 * α) = (24 * Real.sqrt 3 - 7) / 50 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2251_225163


namespace NUMINAMATH_CALUDE_pie_division_l2251_225114

theorem pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 8/9 ∧ num_people = 4 → 
  total_pie / num_people = 2/9 := by sorry

end NUMINAMATH_CALUDE_pie_division_l2251_225114


namespace NUMINAMATH_CALUDE_new_encoding_of_old_message_l2251_225186

/-- Represents the old encoding system --/
def OldEncoding : Type := String

/-- Represents the new encoding system --/
def NewEncoding : Type := String

/-- Decodes a message from the old encoding system --/
def decode (msg : OldEncoding) : String :=
  sorry

/-- Encodes a message using the new encoding system --/
def encode (msg : String) : NewEncoding :=
  sorry

/-- The new encoding rules --/
def newEncodingRules : List (Char × String) :=
  [('A', "21"), ('B', "122"), ('C', "1")]

/-- The theorem to be proved --/
theorem new_encoding_of_old_message :
  let oldMsg : OldEncoding := "011011010011"
  let decodedMsg := decode oldMsg
  encode decodedMsg = "211221121" :=
by sorry

end NUMINAMATH_CALUDE_new_encoding_of_old_message_l2251_225186


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_first_l2251_225190

/-- Proves the minimum speed required for Person B to arrive before Person A --/
theorem min_speed_to_arrive_first (distance : ℝ) (speed_a : ℝ) (delay : ℝ) 
  (h1 : distance = 220)
  (h2 : speed_a = 40)
  (h3 : delay = 0.5)
  (h4 : speed_a > 0) :
  ∃ (min_speed : ℝ), 
    (∀ (speed_b : ℝ), speed_b > min_speed → 
      distance / speed_b + delay < distance / speed_a) ∧
    min_speed = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_first_l2251_225190


namespace NUMINAMATH_CALUDE_strawberries_left_l2251_225107

/-- Given 3.5 baskets of strawberries, with 50 strawberries per basket,
    distributed equally among 24 girls, prove that 7 strawberries are left. -/
theorem strawberries_left (baskets : ℚ) (strawberries_per_basket : ℕ) (girls : ℕ) :
  baskets = 3.5 ∧ strawberries_per_basket = 50 ∧ girls = 24 →
  (baskets * strawberries_per_basket : ℚ) - (↑girls * ↑⌊(baskets * strawberries_per_basket) / girls⌋) = 7 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_left_l2251_225107


namespace NUMINAMATH_CALUDE_canoe_rowing_probability_l2251_225127

def left_oar_prob : ℚ := 3/5
def right_oar_prob : ℚ := 3/5

theorem canoe_rowing_probability :
  let prob_at_least_one_oar := 
    left_oar_prob * right_oar_prob + 
    left_oar_prob * (1 - right_oar_prob) + 
    (1 - left_oar_prob) * right_oar_prob
  prob_at_least_one_oar = 21/25 := by
sorry

end NUMINAMATH_CALUDE_canoe_rowing_probability_l2251_225127


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2251_225119

/-- Given two right triangles with sides 5, 12, and 13, let x be the side length of a square
    inscribed in the first triangle with one vertex at the right angle, and y be the side length
    of a square inscribed in the second triangle with one side on the hypotenuse. -/
def inscribed_squares (x y : ℝ) : Prop :=
  -- First triangle conditions
  5^2 + 12^2 = 13^2 ∧
  x * (12 - x) = 5 * x ∧
  -- Second triangle conditions
  5^2 + 12^2 = 13^2 ∧
  y * (13 - 2*y) = 5 * 12

/-- The ratio of the side lengths of the inscribed squares is 169/220. -/
theorem inscribed_squares_ratio :
  ∀ x y : ℝ, inscribed_squares x y → x / y = 169 / 220 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2251_225119


namespace NUMINAMATH_CALUDE_exists_coverable_parallelepiped_l2251_225189

/-- Represents a parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ+

/-- Checks if three squares can cover a parallelepiped with shared edges -/
def can_cover_with_shared_edges (p : Parallelepiped) (s1 s2 s3 : Square) : Prop :=
  -- The squares cover the surface area of the parallelepiped
  2 * (p.length * p.width + p.length * p.height + p.width * p.height) =
    s1.side * s1.side + s2.side * s2.side + s3.side * s3.side ∧
  -- Each pair of squares shares an edge
  (s1.side = p.length ∨ s1.side = p.width ∨ s1.side = p.height) ∧
  (s2.side = p.length ∨ s2.side = p.width ∨ s2.side = p.height) ∧
  (s3.side = p.length ∨ s3.side = p.width ∨ s3.side = p.height)

/-- Theorem stating the existence of a parallelepiped coverable by three squares with shared edges -/
theorem exists_coverable_parallelepiped :
  ∃ (p : Parallelepiped) (s1 s2 s3 : Square),
    can_cover_with_shared_edges p s1 s2 s3 :=
  sorry

end NUMINAMATH_CALUDE_exists_coverable_parallelepiped_l2251_225189


namespace NUMINAMATH_CALUDE_marble_distribution_l2251_225100

theorem marble_distribution (total_marbles : ℕ) (num_children : ℕ) 
  (h1 : total_marbles = 60) 
  (h2 : num_children = 7) : 
  (num_children - (total_marbles % num_children)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l2251_225100


namespace NUMINAMATH_CALUDE_simplify_expression_l2251_225156

theorem simplify_expression (x : ℝ) : 1 - (2 + (1 - (1 + (2 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2251_225156


namespace NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l2251_225174

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x^2 + 13*x + 20

-- Theorem statement
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-2, -2) :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l2251_225174


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2251_225150

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2251_225150


namespace NUMINAMATH_CALUDE_family_movie_night_l2251_225128

/-- Proves the number of children in a family given ticket prices and payment information --/
theorem family_movie_night (regular_ticket_price : ℕ) 
                            (child_discount : ℕ)
                            (payment : ℕ)
                            (change : ℕ)
                            (num_adults : ℕ) :
  regular_ticket_price = 9 →
  child_discount = 2 →
  payment = 40 →
  change = 1 →
  num_adults = 2 →
  ∃ (num_children : ℕ),
    num_children = 3 ∧
    payment - change = 
      num_adults * regular_ticket_price + 
      num_children * (regular_ticket_price - child_discount) :=
by
  sorry


end NUMINAMATH_CALUDE_family_movie_night_l2251_225128


namespace NUMINAMATH_CALUDE_arts_group_size_l2251_225181

/-- The number of days it takes one student to complete the project -/
def days_for_one_student : ℕ := 60

/-- The number of additional students who joined -/
def additional_students : ℕ := 15

/-- The total number of days worked -/
def total_days_worked : ℕ := 3

/-- The number of days worked with additional students -/
def days_with_additional : ℕ := 2

/-- The total amount of work to be done -/
def total_work : ℚ := 1

theorem arts_group_size :
  ∃ (x : ℕ),
    (x : ℚ) / days_for_one_student + 
    days_with_additional * ((x : ℚ) + additional_students) / days_for_one_student = total_work ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_arts_group_size_l2251_225181


namespace NUMINAMATH_CALUDE_abie_initial_bags_l2251_225182

/-- The number of bags of chips Abie initially had -/
def initial_bags : ℕ := sorry

/-- The number of bags Abie gave away -/
def bags_given_away : ℕ := 4

/-- The number of bags Abie bought -/
def bags_bought : ℕ := 6

/-- The final number of bags Abie has -/
def final_bags : ℕ := 22

/-- Theorem stating that Abie initially had 20 bags of chips -/
theorem abie_initial_bags : 
  initial_bags = 20 ∧ 
  initial_bags - bags_given_away + bags_bought = final_bags :=
sorry

end NUMINAMATH_CALUDE_abie_initial_bags_l2251_225182


namespace NUMINAMATH_CALUDE_total_cement_is_54_4_l2251_225179

/-- Amount of cement used for Lexi's street in tons -/
def lexis_cement : ℝ := 10

/-- Amount of cement used for Tess's street in tons -/
def tess_cement : ℝ := lexis_cement * 1.2

/-- Amount of cement used for Ben's street in tons -/
def bens_cement : ℝ := tess_cement * 0.9

/-- Amount of cement used for Olivia's street in tons -/
def olivias_cement : ℝ := bens_cement * 2

/-- Total amount of cement used for all four streets in tons -/
def total_cement : ℝ := lexis_cement + tess_cement + bens_cement + olivias_cement

theorem total_cement_is_54_4 : total_cement = 54.4 := by
  sorry

end NUMINAMATH_CALUDE_total_cement_is_54_4_l2251_225179


namespace NUMINAMATH_CALUDE_last_digit_322_pow_369_l2251_225118

/-- The last digit of a number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Checks if two numbers have the same last digit -/
def sameLastDigit (a b : ℕ) : Prop := lastDigit a = lastDigit b

theorem last_digit_322_pow_369 : sameLastDigit (322^369) 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_322_pow_369_l2251_225118


namespace NUMINAMATH_CALUDE_negation_of_implication_l2251_225149

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2251_225149


namespace NUMINAMATH_CALUDE_dividend_calculation_l2251_225198

theorem dividend_calculation (divisor quotient remainder dividend : ℤ) :
  divisor = 800 →
  quotient = 594 →
  remainder = -968 →
  dividend = divisor * quotient + remainder →
  dividend = 474232 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2251_225198


namespace NUMINAMATH_CALUDE_probability_intersection_bounds_l2251_225168

theorem probability_intersection_bounds (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 3/4) (hB : P B = 2/3) :
  5/12 ≤ P (A ∩ B) ∧ P (A ∩ B) ≤ 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_intersection_bounds_l2251_225168


namespace NUMINAMATH_CALUDE_total_texts_sent_l2251_225155

/-- The number of texts Sydney sent to Allison, Brittney, and Carol over three days -/
theorem total_texts_sent (
  monday_allison monday_brittney monday_carol : ℕ)
  (tuesday_allison tuesday_brittney tuesday_carol : ℕ)
  (wednesday_allison wednesday_brittney wednesday_carol : ℕ)
  (h1 : monday_allison = 5 ∧ monday_brittney = 5 ∧ monday_carol = 5)
  (h2 : tuesday_allison = 15 ∧ tuesday_brittney = 10 ∧ tuesday_carol = 12)
  (h3 : wednesday_allison = 20 ∧ wednesday_brittney = 18 ∧ wednesday_carol = 7) :
  monday_allison + monday_brittney + monday_carol +
  tuesday_allison + tuesday_brittney + tuesday_carol +
  wednesday_allison + wednesday_brittney + wednesday_carol = 97 :=
by sorry

end NUMINAMATH_CALUDE_total_texts_sent_l2251_225155


namespace NUMINAMATH_CALUDE_binomial_10_3_l2251_225147

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2251_225147


namespace NUMINAMATH_CALUDE_tan_and_g_alpha_l2251_225166

open Real

theorem tan_and_g_alpha (α : ℝ) 
  (h1 : π / 2 < α) (h2 : α < π) 
  (h3 : tan α - (tan α)⁻¹ = -8/3) : 
  tan α = -3 ∧ 
  (sin (π + α) + 4 * cos (2*π + α)) / (sin (π/2 - α) - 4 * sin (-α)) = -7/11 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_g_alpha_l2251_225166


namespace NUMINAMATH_CALUDE_alissa_earring_ratio_l2251_225106

/-- The ratio of Alissa's total earrings to the number of earrings she was given -/
def earring_ratio (barbie_pairs : ℕ) (alissa_total : ℕ) : ℚ :=
  let barbie_total := 2 * barbie_pairs
  let alissa_given := barbie_total / 2
  alissa_total / alissa_given

/-- Theorem stating the ratio of Alissa's total earrings to the number of earrings she was given -/
theorem alissa_earring_ratio :
  let barbie_pairs := 12
  let alissa_total := 36
  earring_ratio barbie_pairs alissa_total = 3 := by
  sorry

end NUMINAMATH_CALUDE_alissa_earring_ratio_l2251_225106


namespace NUMINAMATH_CALUDE_equation_solutions_l2251_225135

theorem equation_solutions :
  (∀ x : ℝ, 3 * x^2 - 4 * x + 5 ≠ 0) ∧
  (∃! s : Set ℝ, s = {-2, 1} ∧ ∀ x ∈ s, (x + 1) * (x + 2) = 2 * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2251_225135


namespace NUMINAMATH_CALUDE_plane_properties_l2251_225142

def plane_equation (x y z : ℝ) : ℝ := 4*x - 3*y - z - 7

def point_M : ℝ × ℝ × ℝ := (2, -1, 4)
def point_N : ℝ × ℝ × ℝ := (3, 2, -1)

def given_plane_normal : ℝ × ℝ × ℝ := (1, 1, 1)

theorem plane_properties :
  (plane_equation point_M.1 point_M.2.1 point_M.2.2 = 0) ∧
  (plane_equation point_N.1 point_N.2.1 point_N.2.2 = 0) ∧
  (4 * given_plane_normal.1 + (-3) * given_plane_normal.2.1 + (-1) * given_plane_normal.2.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_properties_l2251_225142


namespace NUMINAMATH_CALUDE_dot_product_sum_l2251_225180

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem dot_product_sum : 
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by sorry

end NUMINAMATH_CALUDE_dot_product_sum_l2251_225180


namespace NUMINAMATH_CALUDE_earth_surface_cultivation_l2251_225167

theorem earth_surface_cultivation (total : ℝ) (water_percentage : ℝ) (land_percentage : ℝ)
  (desert_ice_fraction : ℝ) (pasture_forest_mountain_fraction : ℝ) :
  water_percentage = 70 →
  land_percentage = 30 →
  water_percentage + land_percentage = 100 →
  desert_ice_fraction = 2/5 →
  pasture_forest_mountain_fraction = 1/3 →
  (land_percentage / 100 * total * (1 - desert_ice_fraction - pasture_forest_mountain_fraction)) / total * 100 = 8 :=
by sorry

end NUMINAMATH_CALUDE_earth_surface_cultivation_l2251_225167


namespace NUMINAMATH_CALUDE_ball_distribution_l2251_225164

theorem ball_distribution (red_balls : ℕ) (white_balls : ℕ) (boxes : ℕ) : 
  red_balls = 17 → white_balls = 10 → boxes = 4 →
  (Nat.choose (white_balls + boxes - 1) (boxes - 1)) * 
  (Nat.choose (red_balls - 1) (boxes - 1)) = 5720 := by
sorry

end NUMINAMATH_CALUDE_ball_distribution_l2251_225164


namespace NUMINAMATH_CALUDE_M_intersect_N_l2251_225108

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem M_intersect_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2251_225108


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2251_225130

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2251_225130


namespace NUMINAMATH_CALUDE_sin_80_minus_sin_20_over_cos_20_l2251_225178

theorem sin_80_minus_sin_20_over_cos_20 :
  (2 * Real.sin (80 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_80_minus_sin_20_over_cos_20_l2251_225178


namespace NUMINAMATH_CALUDE_subtraction_and_divisibility_implies_sum_l2251_225193

theorem subtraction_and_divisibility_implies_sum (a b : Nat) : 
  (741 - (300 + 10*a + 4) = 400 + 10*b + 7) → 
  ((400 + 10*b + 7) % 11 = 0) → 
  (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_subtraction_and_divisibility_implies_sum_l2251_225193


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_condition_l2251_225154

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  2/a + 2/b + 2/c ≥ 2 := by
  sorry

theorem min_value_sum_reciprocals_equality_condition (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  2/a + 2/b + 2/c = 2 ↔ a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_equality_condition_l2251_225154


namespace NUMINAMATH_CALUDE_D_72_l2251_225145

/-- D(n) is the number of ways to write n as a product of factors greater than 1,
    considering the order of factors, and allowing any number of factors (at least one). -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) equals 93 -/
theorem D_72 : D 72 = 93 := by sorry

end NUMINAMATH_CALUDE_D_72_l2251_225145


namespace NUMINAMATH_CALUDE_password_count_correct_l2251_225170

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in the password -/
def num_password_letters : ℕ := 2

/-- The number of digits in the password -/
def num_password_digits : ℕ := 2

/-- The total number of possible passwords -/
def num_possible_passwords : ℕ := (num_letters * (num_letters - 1)) * (num_digits * (num_digits - 1))

theorem password_count_correct :
  num_possible_passwords = num_letters * (num_letters - 1) * num_digits * (num_digits - 1) := by
  sorry

end NUMINAMATH_CALUDE_password_count_correct_l2251_225170


namespace NUMINAMATH_CALUDE_fraction_simplification_l2251_225160

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - 1/y ≠ 0) :
  (y - 1/x) / (x - 1/y) = y/x :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2251_225160


namespace NUMINAMATH_CALUDE_gordons_heavier_bag_weight_l2251_225133

theorem gordons_heavier_bag_weight (trace_bag_count : ℕ) (trace_bag_weight : ℝ)
  (gordon_bag_count : ℕ) (gordon_lighter_bag_weight : ℝ) :
  trace_bag_count = 5 →
  trace_bag_weight = 2 →
  gordon_bag_count = 2 →
  gordon_lighter_bag_weight = 3 →
  trace_bag_count * trace_bag_weight = gordon_lighter_bag_weight + gordon_heavier_bag_weight →
  gordon_heavier_bag_weight = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_gordons_heavier_bag_weight_l2251_225133


namespace NUMINAMATH_CALUDE_triangle_segment_inequality_l2251_225171

/-- Represents a configuration of points in space -/
structure PointConfiguration where
  n : ℕ
  K : ℕ
  T : ℕ
  h_n_ge_2 : n ≥ 2
  h_K_gt_1 : K > 1
  h_no_four_coplanar : True  -- This is a placeholder for the condition

/-- The main theorem -/
theorem triangle_segment_inequality (config : PointConfiguration) :
  9 * (config.T ^ 2) < 2 * (config.K ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_inequality_l2251_225171


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2251_225124

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2251_225124


namespace NUMINAMATH_CALUDE_number_of_persimmons_l2251_225146

/-- Given that there are 18 apples and the sum of apples and persimmons is 33,
    prove that the number of persimmons is 15. -/
theorem number_of_persimmons (apples : ℕ) (total : ℕ) (persimmons : ℕ) 
    (h1 : apples = 18)
    (h2 : apples + persimmons = total)
    (h3 : total = 33) :
    persimmons = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persimmons_l2251_225146


namespace NUMINAMATH_CALUDE_sugar_concentration_of_second_solution_l2251_225192

/-- Given two solutions A and B, where:
    - A is 10% sugar by weight
    - B has an unknown sugar concentration
    - 3/4 of A is mixed with 1/4 of B
    - The resulting mixture is 16% sugar by weight
    This theorem proves that B must be 34% sugar by weight -/
theorem sugar_concentration_of_second_solution
  (W : ℝ) -- Total weight of the original solution
  (h_W_pos : W > 0) -- Assumption that W is positive
  : let A := 0.10 -- Sugar concentration of solution A (10%)
    let final_concentration := 0.16 -- Sugar concentration of final mixture (16%)
    let B := (4 * final_concentration - 3 * A) -- Sugar concentration of solution B
    B = 0.34 -- B is 34% sugar by weight
  := by sorry

end NUMINAMATH_CALUDE_sugar_concentration_of_second_solution_l2251_225192


namespace NUMINAMATH_CALUDE_probability_two_green_marbles_l2251_225197

/-- The probability of drawing two green marbles without replacement from a jar containing 5 red, 3 green, and 7 white marbles is 1/35. -/
theorem probability_two_green_marbles (red green white : ℕ) 
  (h_red : red = 5) 
  (h_green : green = 3) 
  (h_white : white = 7) : 
  (green / (red + green + white)) * ((green - 1) / (red + green + white - 1)) = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_green_marbles_l2251_225197


namespace NUMINAMATH_CALUDE_custom_op_problem_l2251_225183

/-- The custom operation @ defined as a @ b = a × (a + 1) × ... × (a + b - 1) -/
def custom_op (a b : ℕ) : ℕ := 
  (List.range b).foldl (fun acc i => acc * (a + i)) a

/-- Theorem stating that if x @ y @ 2 = 420, then y @ x = 20 -/
theorem custom_op_problem (x y : ℕ) : 
  custom_op x (custom_op y 2) = 420 → custom_op y x = 20 := by
  sorry

#check custom_op_problem

end NUMINAMATH_CALUDE_custom_op_problem_l2251_225183


namespace NUMINAMATH_CALUDE_line_intersects_circle_through_center_l2251_225172

open Real

/-- Proves that a line intersects a circle through its center -/
theorem line_intersects_circle_through_center (α : ℝ) :
  let line := fun (x y : ℝ) => x * cos α - y * sin α = 1
  let circle := fun (x y : ℝ) => (x - cos α)^2 + (y + sin α)^2 = 4
  let center := (cos α, -sin α)
  line center.1 center.2 ∧ circle center.1 center.2 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_through_center_l2251_225172


namespace NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l2251_225129

theorem sin_pi_fourth_plus_alpha (α : ℝ) (h : Real.cos (π/4 - α) = 1/3) :
  Real.sin (π/4 + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_fourth_plus_alpha_l2251_225129


namespace NUMINAMATH_CALUDE_solve_frog_pond_l2251_225199

def frog_pond_problem (initial_frogs : ℕ) : Prop :=
  let tadpoles := 3 * initial_frogs
  let surviving_tadpoles := (2 * tadpoles) / 3
  let total_frogs := initial_frogs + surviving_tadpoles
  (total_frogs = 8) ∧ (total_frogs - 7 = 1)

theorem solve_frog_pond : ∃ (n : ℕ), frog_pond_problem n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_frog_pond_l2251_225199


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2251_225196

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2*y + 3 = 0 ∧ 4*x - a*y + 5 = 0) →
  ((-(1:ℝ)/2) * (4/a) = -1) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2251_225196


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2251_225165

theorem expand_and_simplify : 
  (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2251_225165


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2251_225157

theorem least_subtraction_for_divisibility (n : ℕ) (primes : List ℕ) 
  (h_n : n = 899830)
  (h_primes : primes = [2, 3, 5, 7, 11]) : 
  ∃ (k : ℕ), 
    k = 2000 ∧ 
    (∀ m : ℕ, m < k → ¬((n - m) % (primes.prod) = 0)) ∧ 
    ((n - k) % (primes.prod) = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2251_225157


namespace NUMINAMATH_CALUDE_subset_equality_l2251_225159

theorem subset_equality (h : ℕ) (X S : Set ℕ) : h ≥ 3 →
  X = {n : ℕ | n ≥ 2 * h} →
  S ⊆ X →
  S.Nonempty →
  (∀ a b : ℕ, a ≥ h → b ≥ h → (a + b) ∈ S → (a * b) ∈ S) →
  (∀ a b : ℕ, a ≥ h → b ≥ h → (a * b) ∈ S → (a + b) ∈ S) →
  S = X :=
by sorry

end NUMINAMATH_CALUDE_subset_equality_l2251_225159


namespace NUMINAMATH_CALUDE_additional_rows_l2251_225151

theorem additional_rows (initial_rows : ℕ) (initial_trees_per_row : ℕ) (new_trees_per_row : ℕ) :
  initial_rows = 24 →
  initial_trees_per_row = 42 →
  new_trees_per_row = 28 →
  (initial_rows * initial_trees_per_row) / new_trees_per_row - initial_rows = 12 :=
by sorry

end NUMINAMATH_CALUDE_additional_rows_l2251_225151


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2251_225184

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = -1 → a 2 = 1 →
  ∀ n : ℕ, a n = 2 * n - 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2251_225184


namespace NUMINAMATH_CALUDE_second_largest_of_five_consecutive_sum_90_l2251_225195

theorem second_largest_of_five_consecutive_sum_90 (a b c d e : ℕ) : 
  (a + b + c + d + e = 90) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  d = 19 := by
sorry

end NUMINAMATH_CALUDE_second_largest_of_five_consecutive_sum_90_l2251_225195


namespace NUMINAMATH_CALUDE_standard_deviation_transform_l2251_225140

/-- Given a sample of 10 data points, this function represents their standard deviation. -/
def standard_deviation (x : Fin 10 → ℝ) : ℝ := sorry

/-- This function represents the transformation applied to each data point. -/
def transform (x : ℝ) : ℝ := 3 * x - 1

theorem standard_deviation_transform (x : Fin 10 → ℝ) :
  standard_deviation x = 8 →
  standard_deviation (λ i => transform (x i)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_transform_l2251_225140


namespace NUMINAMATH_CALUDE_octal_to_decimal_23456_l2251_225137

/-- Converts a base-8 digit to its base-10 equivalent --/
def octal_to_decimal (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (8 ^ position)

/-- The base-10 equivalent of 23456 in base-8 --/
def base_10_equivalent : ℕ :=
  octal_to_decimal 6 0 +
  octal_to_decimal 5 1 +
  octal_to_decimal 4 2 +
  octal_to_decimal 3 3 +
  octal_to_decimal 2 4

/-- Theorem: The base-10 equivalent of 23456 in base-8 is 5934 --/
theorem octal_to_decimal_23456 : base_10_equivalent = 5934 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_23456_l2251_225137


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2251_225148

theorem two_digit_number_property (a b k : ℕ) : 
  (a ≥ 1 ∧ a ≤ 9) →  -- a is a single digit (tens place)
  (b ≥ 0 ∧ b ≤ 9) →  -- b is a single digit (ones place)
  (10 * a + b = k * (a + b)) →  -- original number condition
  (10 * b + a = (13 - k) * (a + b)) →  -- interchanged digits condition
  k = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2251_225148


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2251_225138

theorem polynomial_root_sum (b c d e : ℝ) : 
  (∀ x : ℝ, 2*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ↔ x = 4 ∨ x = -3 ∨ x = 5 ∨ x = ((-b-c-d)/2)) →
  (b + c + d) / 2 = 151 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2251_225138


namespace NUMINAMATH_CALUDE_larger_divided_by_smaller_l2251_225117

theorem larger_divided_by_smaller (L S Q : ℕ) : 
  L - S = 1365 →
  S = 270 →
  L = S * Q + 15 →
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_larger_divided_by_smaller_l2251_225117


namespace NUMINAMATH_CALUDE_equation_solution_l2251_225126

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2251_225126


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2251_225121

theorem arithmetic_expression_equality : 72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2251_225121


namespace NUMINAMATH_CALUDE_inequality_solution_l2251_225110

theorem inequality_solution (x : ℝ) : 
  (6 * x^2 + 12 * x - 35) / ((x - 2) * (3 * x + 6)) < 2 ↔ 
  (x > -2 ∧ x < 11/18) ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2251_225110


namespace NUMINAMATH_CALUDE_prime_simultaneous_l2251_225136

theorem prime_simultaneous (p : ℕ) : 
  Nat.Prime p ∧ Nat.Prime (8 * p^2 + 1) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_simultaneous_l2251_225136


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2251_225144

theorem pure_imaginary_condition (x : ℝ) : 
  (∃ (y : ℝ), y ≠ 0 ∧ (x^2 - 1) + (x - 1)*I = y*I) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2251_225144


namespace NUMINAMATH_CALUDE_evaluate_g_l2251_225113

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g : 3 * g 4 - 2 * g (-2) = 47 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l2251_225113


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2251_225141

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (7218 * N) % 6 = 0 ∧ ∀ (M : ℕ), M ≤ 9 ∧ (7218 * M) % 6 = 0 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2251_225141


namespace NUMINAMATH_CALUDE_walter_school_expenses_l2251_225139

/-- Represents Walter's weekly work schedule and earnings --/
structure WalterSchedule where
  job1_weekday_hours : ℝ
  job1_weekend_hours : ℝ
  job1_hourly_rate : ℝ
  job1_weekly_bonus : ℝ
  job1_tax_rate : ℝ
  job2_hours : ℝ
  job2_hourly_rate : ℝ
  job2_tax_rate : ℝ
  job3_hours : ℝ
  job3_hourly_rate : ℝ
  school_allocation_rate : ℝ

/-- Calculates Walter's weekly school expense allocation --/
def calculateSchoolExpenses (schedule : WalterSchedule) : ℝ :=
  let job1_earnings := (schedule.job1_weekday_hours * 5 + schedule.job1_weekend_hours * 2) * schedule.job1_hourly_rate + schedule.job1_weekly_bonus
  let job1_after_tax := job1_earnings * (1 - schedule.job1_tax_rate)
  let job2_earnings := schedule.job2_hours * schedule.job2_hourly_rate
  let job2_after_tax := job2_earnings * (1 - schedule.job2_tax_rate)
  let job3_earnings := schedule.job3_hours * schedule.job3_hourly_rate
  let total_earnings := job1_after_tax + job2_after_tax + job3_earnings
  total_earnings * schedule.school_allocation_rate

/-- Theorem stating that Walter's weekly school expense allocation is approximately $211.69 --/
theorem walter_school_expenses (schedule : WalterSchedule) 
  (h1 : schedule.job1_weekday_hours = 4)
  (h2 : schedule.job1_weekend_hours = 6)
  (h3 : schedule.job1_hourly_rate = 5)
  (h4 : schedule.job1_weekly_bonus = 50)
  (h5 : schedule.job1_tax_rate = 0.1)
  (h6 : schedule.job2_hours = 5)
  (h7 : schedule.job2_hourly_rate = 7)
  (h8 : schedule.job2_tax_rate = 0.05)
  (h9 : schedule.job3_hours = 6)
  (h10 : schedule.job3_hourly_rate = 10)
  (h11 : schedule.school_allocation_rate = 0.75) :
  ∃ ε > 0, |calculateSchoolExpenses schedule - 211.69| < ε := by
  sorry

end NUMINAMATH_CALUDE_walter_school_expenses_l2251_225139


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2251_225161

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

-- Statement to prove
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2251_225161


namespace NUMINAMATH_CALUDE_superinverse_value_l2251_225105

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 9*x^2 + 27*x + 81

-- State that g is bijective
axiom g_bijective : Function.Bijective g

-- Define the superinverse property
def is_superinverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, (f ∘ g) x = Function.invFun g x

-- State that f is the superinverse of g
axiom f_is_superinverse : ∃ f : ℝ → ℝ, is_superinverse f g

-- The theorem to prove
theorem superinverse_value :
  ∃ f : ℝ → ℝ, is_superinverse f g ∧ |f (-289)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_superinverse_value_l2251_225105


namespace NUMINAMATH_CALUDE_exists_intransitive_dice_l2251_225116

/-- Represents a die with 6 faces -/
def Die := Fin 6 → Nat

/-- The probability that one die shows a higher number than another -/
def winProbability (d1 d2 : Die) : ℚ :=
  (Finset.sum Finset.univ (λ i => 
    Finset.sum Finset.univ (λ j => 
      if d1 i > d2 j then 1 else 0
    )
  )) / 36

/-- Predicate for one die winning over another -/
def wins (d1 d2 : Die) : Prop := winProbability d1 d2 > 1/2

/-- Theorem stating the existence of three dice with the desired properties -/
theorem exists_intransitive_dice : ∃ (A B C : Die),
  wins B A ∧ wins C B ∧ wins A C := by sorry

end NUMINAMATH_CALUDE_exists_intransitive_dice_l2251_225116


namespace NUMINAMATH_CALUDE_percentage_problem_l2251_225134

theorem percentage_problem : 
  ∃ x : ℝ, (x / 100 * 50 + 50 / 100 * 860 = 860) ∧ x = 860 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2251_225134


namespace NUMINAMATH_CALUDE_influenza_test_probability_l2251_225188

theorem influenza_test_probability 
  (P : Set Ω → ℝ) 
  (A C : Set Ω) 
  (h1 : P (A ∩ C) / P C = 0.9)
  (h2 : P ((Cᶜ) ∩ (Aᶜ)) / P (Cᶜ) = 0.9)
  (h3 : P C = 0.005)
  : P (C ∩ A) / P A = 9 / 208 := by
  sorry

end NUMINAMATH_CALUDE_influenza_test_probability_l2251_225188


namespace NUMINAMATH_CALUDE_draining_cylinder_height_change_rate_l2251_225115

/-- The rate of change of liquid level height in a draining cylindrical container -/
theorem draining_cylinder_height_change_rate 
  (d : ℝ) -- diameter of the base
  (dV_dt : ℝ) -- rate of volume change (negative for draining)
  (h : ℝ → ℝ) -- height of liquid as a function of time
  (t : ℝ) -- time variable
  (h_diff : Differentiable ℝ h) -- h is differentiable
  (cylinder_volume : ∀ t, π * (d/2)^2 * h t = -dV_dt * t + C) -- volume equation
  (h_positive : ∀ t, h t > 0) -- height is always positive
  (dV_dt_negative : dV_dt < 0) -- volume is decreasing
  (d_positive : d > 0) -- diameter is positive
  (h_init : h 0 > 0) -- initial height is positive
  : d = 2 → dV_dt = -0.01 → deriv h t = -0.01 / π := by
  sorry

end NUMINAMATH_CALUDE_draining_cylinder_height_change_rate_l2251_225115


namespace NUMINAMATH_CALUDE_max_difference_reverse_digits_l2251_225103

theorem max_difference_reverse_digits (q r : ℕ) : 
  (10 ≤ q) ∧ (q < 100) ∧  -- q is a two-digit number
  (10 ≤ r) ∧ (r < 100) ∧  -- r is a two-digit number
  (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ q = 10*x + y ∧ r = 10*y + x) ∧  -- q and r have reversed digits
  (q - r < 30 ∨ r - q < 30) →  -- positive difference is less than 30
  (q - r ≤ 27 ∧ r - q ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_reverse_digits_l2251_225103


namespace NUMINAMATH_CALUDE_symmetry_about_center_three_zeros_existence_l2251_225104

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + 1

-- Theorem for symmetry (Option B)
theorem symmetry_about_center (b : ℝ) :
  ∀ x : ℝ, f 0 b x + f 0 b (-x) = 2 :=
sorry

-- Theorem for three zeros (Option C)
theorem three_zeros_existence (a : ℝ) (h : a > -4) :
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  f a (a^2/4) x = 0 ∧ f a (a^2/4) y = 0 ∧ f a (a^2/4) z = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_center_three_zeros_existence_l2251_225104


namespace NUMINAMATH_CALUDE_lions_and_majestic_l2251_225176

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Lion : U → Prop)
variable (Majestic : U → Prop)
variable (Bird : U → Prop)

-- State the given conditions
variable (h1 : ∀ x, Lion x → Majestic x)
variable (h2 : ∀ x, Bird x → ¬Lion x)

-- Theorem to prove
theorem lions_and_majestic :
  (∀ x, Lion x → ¬Bird x) ∧ (∃ x, Majestic x ∧ ¬Bird x) :=
sorry

end NUMINAMATH_CALUDE_lions_and_majestic_l2251_225176


namespace NUMINAMATH_CALUDE_goldfish_remaining_l2251_225122

def initial_goldfish : ℕ := 15
def fewer_goldfish : ℕ := 11

theorem goldfish_remaining : initial_goldfish - fewer_goldfish = 4 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_remaining_l2251_225122


namespace NUMINAMATH_CALUDE_nine_is_unique_digit_l2251_225173

/-- A function that returns true if a natural number ends with at least k repetitions of digit z -/
def endsWithKDigits (num : ℕ) (k : ℕ) (z : ℕ) : Prop :=
  ∃ m : ℕ, num = m * (10^k) + z * ((10^k - 1) / 9)

/-- The main theorem stating that 9 is the only digit satisfying the condition -/
theorem nine_is_unique_digit : 
  ∀ z : ℕ, z < 10 →
    (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ endsWithKDigits (n^9) k z) ↔ z = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_is_unique_digit_l2251_225173


namespace NUMINAMATH_CALUDE_total_cans_l2251_225101

def bag1 : ℕ := 5
def bag2 : ℕ := 7
def bag3 : ℕ := 12
def bag4 : ℕ := 4
def bag5 : ℕ := 8
def bag6 : ℕ := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_l2251_225101


namespace NUMINAMATH_CALUDE_initial_crayon_packs_l2251_225125

theorem initial_crayon_packs : ℕ := by
  -- Define the cost of one pack of crayons
  let cost_per_pack : ℚ := 5/2

  -- Define the number of additional packs Michael buys
  let additional_packs : ℕ := 2

  -- Define the total value after purchase
  let total_value : ℚ := 15

  -- Define the initial number of packs (to be proven)
  let initial_packs : ℕ := 4

  -- Prove that the initial number of packs is 4
  have h : (cost_per_pack * (initial_packs + additional_packs : ℚ)) = total_value := by sorry

  -- Return the result
  exact initial_packs

end NUMINAMATH_CALUDE_initial_crayon_packs_l2251_225125


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l2251_225102

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line on which the center of C lies
def CenterLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 - 2 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)

-- Define point P
def P : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def XAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define a line given its equation ax + by + c = 0
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem circle_and_line_theorem :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center ∈ CenterLine ∧
    A ∈ Circle center radius ∧
    B ∈ Circle center radius ∧
    (∃ (a b c : ℝ),
      (Line a b c = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 3 = 0} ∨
       Line a b c = {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 + 3 = 0}) ∧
      P ∈ Line a b c ∧
      (∃ (q : ℝ × ℝ), q ∈ XAxis ∧ q ∈ Line a b c) ∧
      (∃ (t : ℝ × ℝ), t ∈ Circle center radius ∧ t ∈ Line a b c)) ∧
    center = (2, 2) ∧
    radius = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l2251_225102


namespace NUMINAMATH_CALUDE_abc_product_l2251_225177

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 30)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 300 / (a * b * c) = 1) :
  a * b * c = 768 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l2251_225177


namespace NUMINAMATH_CALUDE_divisibility_implication_l2251_225132

/-- Represents a three-digit number with non-zero digits -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- The product of digits of a three-digit number -/
def digit_product (n : ThreeDigitNumber) : Nat :=
  n.a * n.b * n.c

theorem divisibility_implication (n : ThreeDigitNumber) :
  (value n % digit_sum n = 0) ∧ (value n % digit_product n = 0) →
  90 * n.a % digit_sum n = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2251_225132


namespace NUMINAMATH_CALUDE_A_subset_B_A_equals_B_iff_l2251_225111

variable (a : ℝ)

def A : Set ℝ := {x | x^2 + a = x}
def B : Set ℝ := {x | (x^2 + a)^2 + a = x}

axiom A_nonempty : A a ≠ ∅

theorem A_subset_B : A a ⊆ B a := by sorry

theorem A_equals_B_iff : 
  A a = B a ↔ -3/4 ≤ a ∧ a ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_A_subset_B_A_equals_B_iff_l2251_225111


namespace NUMINAMATH_CALUDE_binary_sum_proof_l2251_225152

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ ⟨i, bit⟩ acc => acc + if bit then 2^i else 0) 0

theorem binary_sum_proof :
  let b1 := [true, true, false, true]  -- 1101₂
  let b2 := [true, false, true]        -- 101₂
  let b3 := [true, true, true]         -- 111₂
  let b4 := [true, false, false, false, true]  -- 10001₂
  let result := [false, true, false, true, false, true]  -- 101010₂
  binary_to_decimal b1 + binary_to_decimal b2 + binary_to_decimal b3 + binary_to_decimal b4 = binary_to_decimal result :=
by sorry

end NUMINAMATH_CALUDE_binary_sum_proof_l2251_225152


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l2251_225143

theorem pentagon_area_sum (u v : ℤ) : 
  0 < v → v < u → (u^2 + 3*u*v = 150) → u + v = 15 := by
  sorry

#check pentagon_area_sum

end NUMINAMATH_CALUDE_pentagon_area_sum_l2251_225143


namespace NUMINAMATH_CALUDE_sum_abc_equals_109610_l2251_225185

/-- Proves that given the conditions, the sum of a, b, and c is 109610 rupees -/
theorem sum_abc_equals_109610 (a b c : ℕ) : 
  (0.5 / 100 : ℚ) * a = 95 / 100 →  -- 0.5% of a equals 95 paise
  b = 3 * a - 50 →                  -- b is three times the amount of a minus 50
  c = (a - b) ^ 2 →                 -- c is the difference between a and b squared
  a > 0 →                           -- a is a positive integer
  c > 0 →                           -- c is a positive integer
  a + b + c = 109610 := by           -- The sum of a, b, and c is 109610 rupees
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_109610_l2251_225185


namespace NUMINAMATH_CALUDE_train_car_count_l2251_225158

/-- Calculates the number of cars in a train given the observed data -/
def train_cars (cars_observed : ℕ) (observation_time : ℕ) (total_time : ℕ) : ℕ :=
  (cars_observed * total_time) / observation_time

/-- Theorem stating the number of cars in the train -/
theorem train_car_count :
  let cars_observed : ℕ := 8
  let observation_time : ℕ := 12  -- in seconds
  let total_time : ℕ := 3 * 60    -- 3 minutes converted to seconds
  train_cars cars_observed observation_time total_time = 120 := by
  sorry

#eval train_cars 8 12 (3 * 60)

end NUMINAMATH_CALUDE_train_car_count_l2251_225158


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2251_225109

theorem fruit_seller_apples : ∀ (original : ℕ),
  (original : ℝ) * (1 - 0.4) = 420 → original = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2251_225109


namespace NUMINAMATH_CALUDE_geometric_roots_product_l2251_225112

/-- 
Given an equation (x² - mx - 8)(x² - nx - 8) = 0 where m and n are real numbers,
if its four roots form a geometric sequence with the first term being 1,
then the product of m and n is -14.
-/
theorem geometric_roots_product (m n : ℝ) : 
  (∃ a r : ℝ, r ≠ 0 ∧ a = 1 ∧
    (∀ x : ℝ, (x^2 - m*x - 8 = 0 ∨ x^2 - n*x - 8 = 0) ↔ 
      (x = a ∨ x = a*r ∨ x = a*r^2 ∨ x = a*r^3))) →
  m * n = -14 := by
sorry

end NUMINAMATH_CALUDE_geometric_roots_product_l2251_225112


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2251_225187

/-- A random variable following a normal distribution with mean 3 and standard deviation σ -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is less than 2 -/
def prob_X_less_than_2 (σ : ℝ) : ℝ := 0.3

/-- The probability that X is between 2 and 4 -/
def prob_X_between_2_and_4 (σ : ℝ) : ℝ := 1 - 2 * prob_X_less_than_2 σ

theorem normal_distribution_symmetry (σ : ℝ) (h : σ > 0) :
  prob_X_between_2_and_4 σ = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2251_225187


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l2251_225162

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (y : ℝ) : Prop :=
  cubeRoot (30 * y + cubeRoot (30 * y + 26)) = 26

-- Theorem statement
theorem solution_satisfies_equation : equation 585 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l2251_225162


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2251_225153

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 14) : x^3 + y^3 = 580 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2251_225153
