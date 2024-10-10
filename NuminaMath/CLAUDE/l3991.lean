import Mathlib

namespace parallelogram_area_l3991_399161

/-- Represents a parallelogram with given base, height, and one angle -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  angle : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of a parallelogram with base 20 and height 4 is 80 -/
theorem parallelogram_area :
  ∀ (p : Parallelogram), p.base = 20 ∧ p.height = 4 ∧ p.angle = 60 → area p = 80 := by
  sorry

end parallelogram_area_l3991_399161


namespace problem_solution_l3991_399175

theorem problem_solution (a b : ℝ) 
  (h1 : a^2 + 2*b = 0) 
  (h2 : |a^2 - 2*b| = 8) : 
  b + 2023 = 2021 := by
sorry

end problem_solution_l3991_399175


namespace dianes_honey_harvest_l3991_399121

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest (last_year_harvest : ℕ) (increase : ℕ) : 
  last_year_harvest = 2479 → increase = 6085 → last_year_harvest + increase = 8564 := by
  sorry

end dianes_honey_harvest_l3991_399121


namespace point_b_not_on_curve_l3991_399158

/-- The equation of curve C -/
def curve_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 6*a*x - 8*a*y = 0

/-- Point B does not lie on curve C -/
theorem point_b_not_on_curve (a : ℝ) : ¬ curve_equation (2*a) (4*a) a := by
  sorry

end point_b_not_on_curve_l3991_399158


namespace divisors_of_eight_factorial_greater_than_seven_factorial_l3991_399168

theorem divisors_of_eight_factorial_greater_than_seven_factorial :
  (Finset.filter (fun d => d > Nat.factorial 7 ∧ Nat.factorial 8 % d = 0) (Finset.range (Nat.factorial 8 + 1))).card = 7 := by
  sorry

end divisors_of_eight_factorial_greater_than_seven_factorial_l3991_399168


namespace prob_first_odd_given_two_odd_one_even_l3991_399142

/-- Represents the outcome of picking a ball -/
inductive BallType
| Odd
| Even

/-- Represents the result of picking 3 balls -/
structure ThreePickResult :=
  (first second third : BallType)

def is_valid_pick (result : ThreePickResult) : Prop :=
  (result.first = BallType.Odd ∧ result.second = BallType.Odd ∧ result.third = BallType.Even) ∨
  (result.first = BallType.Odd ∧ result.second = BallType.Even ∧ result.third = BallType.Odd) ∨
  (result.first = BallType.Even ∧ result.second = BallType.Odd ∧ result.third = BallType.Odd)

def total_balls : ℕ := 100
def odd_balls : ℕ := 50
def even_balls : ℕ := 50

theorem prob_first_odd_given_two_odd_one_even :
  ∀ (sample_space : Set ThreePickResult) (prob : Set ThreePickResult → ℝ),
  (∀ result ∈ sample_space, is_valid_pick result) →
  (∀ A ⊆ sample_space, 0 ≤ prob A ∧ prob A ≤ 1) →
  prob sample_space = 1 →
  prob {result ∈ sample_space | result.first = BallType.Odd} = 1/2 := by
  sorry

end prob_first_odd_given_two_odd_one_even_l3991_399142


namespace carson_gold_stars_l3991_399165

/-- The total number of gold stars Carson earned over three days -/
def total_gold_stars (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating that Carson earned 26 gold stars in total -/
theorem carson_gold_stars :
  total_gold_stars 7 11 8 = 26 := by
  sorry

end carson_gold_stars_l3991_399165


namespace birthday_age_problem_l3991_399113

theorem birthday_age_problem (current_age : ℕ) : 
  (current_age = 3 * (current_age - 6)) → current_age = 9 := by
  sorry

end birthday_age_problem_l3991_399113


namespace roberta_shopping_trip_l3991_399130

def shopping_trip (initial_amount bag_price_difference lunch_price_fraction : ℚ) : ℚ :=
  let shoe_price := 45
  let bag_price := shoe_price - bag_price_difference
  let lunch_price := bag_price * lunch_price_fraction
  initial_amount - (shoe_price + bag_price + lunch_price)

theorem roberta_shopping_trip :
  shopping_trip 158 17 (1/4) = 78 := by
  sorry

end roberta_shopping_trip_l3991_399130


namespace triangle_inequality_l3991_399146

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a / (b + c)) + Real.sqrt (b / (a + c)) + Real.sqrt (c / (a + b)) ≥ 2 := by
  sorry

end triangle_inequality_l3991_399146


namespace tetrahedron_edge_angle_relation_l3991_399149

/-- Theorem about the relationship between opposite edges and angles in a tetrahedron -/
theorem tetrahedron_edge_angle_relation 
  (a a₁ b b₁ c c₁ : ℝ) 
  (α β γ : ℝ) 
  (h_positive : a > 0 ∧ a₁ > 0 ∧ b > 0 ∧ b₁ > 0 ∧ c > 0 ∧ c₁ > 0)
  (h_angles : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2 ∧ 0 ≤ γ ∧ γ ≤ Real.pi / 2) :
  (a * a₁ * Real.cos α = b * b₁ * Real.cos β + c * c₁ * Real.cos γ) ∨
  (b * b₁ * Real.cos β = a * a₁ * Real.cos α + c * c₁ * Real.cos γ) ∨
  (c * c₁ * Real.cos γ = a * a₁ * Real.cos α + b * b₁ * Real.cos β) := by
  sorry


end tetrahedron_edge_angle_relation_l3991_399149


namespace circle_ellipse_tangent_l3991_399131

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop := x^2 / 18 + y^2 / 2 = 1

-- Define the line PF₁
def line_PF1 (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the point A
def point_A : ℝ × ℝ := (3, 1)

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- State the theorem
theorem circle_ellipse_tangent :
  ∃ (m : ℝ),
    m < 3 ∧
    (∃ (a b : ℝ), a > b ∧ b > 0 ∧
      (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ ellipse_E x y)) ∧
    (∃ e : ℝ, e > 1/2 ∧
      (∀ x y : ℝ, ((x - m)^2 + y^2 = 5) ↔ circle_C x y) ∧
      circle_C point_A.1 point_A.2 ∧
      ellipse_E point_A.1 point_A.2 ∧
      line_PF1 point_P.1 point_P.2) :=
sorry

end circle_ellipse_tangent_l3991_399131


namespace sin_180_degrees_is_zero_l3991_399197

/-- The sine of 180 degrees is 0 -/
theorem sin_180_degrees_is_zero : Real.sin (π) = 0 := by
  sorry

end sin_180_degrees_is_zero_l3991_399197


namespace regular_polygon_with_60_degree_exterior_angle_has_6_sides_l3991_399170

/-- The number of sides of a regular polygon with an exterior angle of 60 degrees is 6. -/
theorem regular_polygon_with_60_degree_exterior_angle_has_6_sides :
  ∀ (n : ℕ), n > 0 →
  (360 / n = 60) →
  n = 6 := by
  sorry

end regular_polygon_with_60_degree_exterior_angle_has_6_sides_l3991_399170


namespace kelly_head_start_l3991_399176

/-- The length of the race in meters -/
def race_length : ℝ := 100

/-- The distance by which Abel lost to Kelly in meters -/
def losing_distance : ℝ := 0.5

/-- The additional distance Abel needs to run to overtake Kelly in meters -/
def overtake_distance : ℝ := 19.9

/-- Kelly's head start in meters -/
def head_start : ℝ := race_length - (race_length - losing_distance - overtake_distance)

theorem kelly_head_start :
  head_start = 20.4 :=
by sorry

end kelly_head_start_l3991_399176


namespace angle_A_is_60_degrees_range_of_b_plus_c_l3991_399128

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle --/
def triangle_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b

/-- Theorem 1: Angle A is 60° --/
theorem angle_A_is_60_degrees (t : Triangle) (h : triangle_condition t) : t.A = π / 3 := by
  sorry

/-- Theorem 2: Range of b + c when a = √3 --/
theorem range_of_b_plus_c (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = Real.sqrt 3) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end angle_A_is_60_degrees_range_of_b_plus_c_l3991_399128


namespace mail_in_rebates_difference_l3991_399135

/-- The number of additional mail-in rebates compared to bills --/
def additional_rebates : ℕ := 3

/-- The total number of stamps needed --/
def total_stamps : ℕ := 21

/-- The number of thank you cards --/
def thank_you_cards : ℕ := 3

/-- The number of bills --/
def bills : ℕ := 2

theorem mail_in_rebates_difference (rebates : ℕ) (job_applications : ℕ) :
  (thank_you_cards + bills + rebates + job_applications + 1 = total_stamps) →
  (job_applications = 2 * rebates) →
  (rebates = bills + additional_rebates) := by
  sorry

end mail_in_rebates_difference_l3991_399135


namespace union_A_B_when_a_is_one_value_set_of_a_when_intersection_empty_l3991_399191

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

-- Theorem for part I
theorem union_A_B_when_a_is_one :
  A 1 ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := by sorry

-- Theorem for part II
theorem value_set_of_a_when_intersection_empty :
  ∀ a : ℝ, a ≥ 0 → (A a ∩ B = ∅ ↔ 0 ≤ a ∧ a ≤ 1/2) := by sorry

end union_A_B_when_a_is_one_value_set_of_a_when_intersection_empty_l3991_399191


namespace total_fruits_count_l3991_399102

-- Define the given conditions
def gerald_apple_bags : ℕ := 5
def gerald_apples_per_bag : ℕ := 30
def gerald_orange_bags : ℕ := 4
def gerald_oranges_per_bag : ℕ := 25

def pam_apple_bags : ℕ := 6
def pam_orange_bags : ℕ := 4

def sue_apple_bags : ℕ := 2 * gerald_apple_bags
def sue_orange_bags : ℕ := gerald_orange_bags / 2

def pam_apples_per_bag : ℕ := 3 * gerald_apples_per_bag
def pam_oranges_per_bag : ℕ := 2 * gerald_oranges_per_bag

def sue_apples_per_bag : ℕ := gerald_apples_per_bag - 10
def sue_oranges_per_bag : ℕ := gerald_oranges_per_bag + 5

-- Theorem statement
theorem total_fruits_count : 
  (gerald_apple_bags * gerald_apples_per_bag + 
   gerald_orange_bags * gerald_oranges_per_bag +
   pam_apple_bags * pam_apples_per_bag + 
   pam_orange_bags * pam_oranges_per_bag +
   sue_apple_bags * sue_apples_per_bag + 
   sue_orange_bags * sue_oranges_per_bag) = 1250 := by
  sorry

end total_fruits_count_l3991_399102


namespace complement_of_at_least_two_defective_l3991_399143

/-- The number of products inspected --/
def n : ℕ := 10

/-- Event A: at least two defective products --/
def event_A (x : ℕ) : Prop := x ≥ 2

/-- The complementary event of A --/
def complement_A (x : ℕ) : Prop := x ≤ 1

/-- Theorem stating that the complement of "at least two defective products" 
    is "at most one defective product" --/
theorem complement_of_at_least_two_defective :
  ∀ x : ℕ, x ≤ n → (¬ event_A x ↔ complement_A x) := by sorry

end complement_of_at_least_two_defective_l3991_399143


namespace path_length_proof_l3991_399100

theorem path_length_proof :
  let rectangle_width : ℝ := 3
  let rectangle_height : ℝ := 4
  let diagonal_length : ℝ := (rectangle_width^2 + rectangle_height^2).sqrt
  let vertical_segments : ℝ := 2 * rectangle_height
  let horizontal_segments : ℝ := 3 * rectangle_width
  diagonal_length + vertical_segments + horizontal_segments = 22 := by
sorry

end path_length_proof_l3991_399100


namespace problem_solution_l3991_399119

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, f x ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → |x + 3| + |x + a| < x + 6) ↔ -1 < a ∧ a < 1) :=
by sorry

end problem_solution_l3991_399119


namespace branch_A_more_profitable_l3991_399129

/-- Represents a branch of the factory -/
inductive Branch
| A
| B

/-- Represents the grade of a product -/
inductive Grade
| A
| B
| C
| D

/-- Processing fee for each grade -/
def processingFee (g : Grade) : ℝ :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Processing cost for each branch -/
def processingCost (b : Branch) : ℝ :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Frequency distribution for each branch and grade -/
def frequency (b : Branch) (g : Grade) : ℝ :=
  match b, g with
  | Branch.A, Grade.A => 0.4
  | Branch.A, Grade.B => 0.2
  | Branch.A, Grade.C => 0.2
  | Branch.A, Grade.D => 0.2
  | Branch.B, Grade.A => 0.28
  | Branch.B, Grade.B => 0.17
  | Branch.B, Grade.C => 0.34
  | Branch.B, Grade.D => 0.21

/-- Average profit for a branch -/
def averageProfit (b : Branch) : ℝ :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem stating that Branch A has higher average profit than Branch B -/
theorem branch_A_more_profitable :
  averageProfit Branch.A > averageProfit Branch.B :=
by sorry


end branch_A_more_profitable_l3991_399129


namespace quadratic_solution_l3991_399180

theorem quadratic_solution (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 →
  a = 10 := by
sorry

end quadratic_solution_l3991_399180


namespace cookies_left_l3991_399199

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies Meena bakes -/
def dozens_baked : ℕ := 5

/-- The number of dozens of cookies Mr. Stone buys -/
def dozens_sold_to_stone : ℕ := 2

/-- The number of cookies Brock buys -/
def cookies_sold_to_brock : ℕ := 7

/-- Calculates the total number of cookies Meena bakes -/
def total_cookies_baked : ℕ := dozens_baked * cookies_per_dozen

/-- Calculates the number of cookies sold to Mr. Stone -/
def cookies_sold_to_stone : ℕ := dozens_sold_to_stone * cookies_per_dozen

/-- Calculates the number of cookies sold to Katy -/
def cookies_sold_to_katy : ℕ := 2 * cookies_sold_to_brock

/-- Calculates the total number of cookies sold -/
def total_cookies_sold : ℕ := cookies_sold_to_stone + cookies_sold_to_brock + cookies_sold_to_katy

/-- Theorem: Meena has 15 cookies left after selling to Mr. Stone, Brock, and Katy -/
theorem cookies_left : total_cookies_baked - total_cookies_sold = 15 := by
  sorry

end cookies_left_l3991_399199


namespace ivy_collectors_edition_dolls_l3991_399107

/-- Proves that Ivy has 20 collectors edition dolls given the conditions -/
theorem ivy_collectors_edition_dolls (dina_dolls ivy_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : dina_dolls = 60) :
  (2 : ℚ) / 3 * ivy_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l3991_399107


namespace geometric_sequence_first_term_l3991_399147

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h1 : ∀ n ≥ 1, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h2 : a 4 = 16) 
  (h3 : a 5 = 32) 
  (h4 : a 6 = 64) : 
  a 1 = 2 := by
sorry

end geometric_sequence_first_term_l3991_399147


namespace inequality_proof_l3991_399138

theorem inequality_proof (x y a b : ℝ) (hx : x > 0) (hy : y > 0) :
  ((a * x + b * y) / (x + y))^2 ≤ (a^2 * x + b^2 * y) / (x + y) := by
  sorry

end inequality_proof_l3991_399138


namespace geometric_sequence_a8_l3991_399117

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the theorem
theorem geometric_sequence_a8 (a₁ : ℝ) (q : ℝ) :
  (a₁ * (a₁ * q^2) = 4) →
  (a₁ * q^8 = 256) →
  (geometric_sequence a₁ q 8 = 128 ∨ geometric_sequence a₁ q 8 = -128) :=
by
  sorry


end geometric_sequence_a8_l3991_399117


namespace determinant_of_specific_matrix_l3991_399108

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 2, 5]
  Matrix.det A = 18 := by sorry

end determinant_of_specific_matrix_l3991_399108


namespace f_derivative_at_zero_l3991_399157

noncomputable def f (x : ℝ) := (2 * x + 1) * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 3 := by sorry

end f_derivative_at_zero_l3991_399157


namespace tank_volume_in_cubic_yards_l3991_399104

/-- Conversion factor from cubic feet to cubic yards -/
def cubicFeetToCubicYards : ℚ := 1 / 27

/-- Volume of the tank in cubic feet -/
def tankVolumeCubicFeet : ℚ := 216

/-- Theorem: The volume of the tank in cubic yards is 8 -/
theorem tank_volume_in_cubic_yards :
  tankVolumeCubicFeet * cubicFeetToCubicYards = 8 := by
  sorry

end tank_volume_in_cubic_yards_l3991_399104


namespace subsets_containing_five_and_six_l3991_399152

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_five_and_six :
  (Finset.filter (λ s : Finset ℕ => 5 ∈ s ∧ 6 ∈ s) (Finset.powerset S)).card = 16 := by
  sorry

end subsets_containing_five_and_six_l3991_399152


namespace sum_of_reciprocals_positive_l3991_399153

theorem sum_of_reciprocals_positive 
  (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end sum_of_reciprocals_positive_l3991_399153


namespace sector_max_area_l3991_399174

/-- A sector is defined by its radius and central angle. -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The perimeter of a sector. -/
def perimeter (s : Sector) : ℝ := s.radius * s.angle + 2 * s.radius

/-- The area of a sector. -/
def area (s : Sector) : ℝ := 0.5 * s.radius^2 * s.angle

/-- Theorem: For a sector with perimeter 4, the area is maximized when the central angle is 2 radians. -/
theorem sector_max_area (s : Sector) (h : perimeter s = 4) :
  area s ≤ area { radius := 1, angle := 2 } := by
  sorry

#check sector_max_area

end sector_max_area_l3991_399174


namespace max_value_x_minus_2y_l3991_399127

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 - 8*x + y^2 - 6*y + 24 = 0) :
  ∃ (max : ℝ), max = Real.sqrt 5 - 2 ∧ ∀ (x' y' : ℝ), x'^2 - 8*x' + y'^2 - 6*y' + 24 = 0 → x' - 2*y' ≤ max :=
by sorry

end max_value_x_minus_2y_l3991_399127


namespace product_power_equals_128y_l3991_399154

theorem product_power_equals_128y (a b : ℤ) (n : ℕ) (h : (a * b) ^ n = 128 * 8) : n = 10 := by
  sorry

end product_power_equals_128y_l3991_399154


namespace scaling_transformation_curve_l3991_399160

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem scaling_transformation_curve (x y x' y' : ℝ) :
  (x' = 5 * x) →
  (y' = 3 * y) →
  (x'^2 + y'^2 = 1) →
  (25 * x^2 + 9 * y^2 = 1) :=
by sorry

end scaling_transformation_curve_l3991_399160


namespace vector_sum_proof_l3991_399181

/-- Given vectors a and b in ℝ², prove that a + 2b = (-3, 4) -/
theorem vector_sum_proof (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-2, 1)) :
  a + 2 • b = (-3, 4) := by
  sorry

end vector_sum_proof_l3991_399181


namespace log_simplification_l3991_399155

theorem log_simplification :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5/4 := by
  sorry

end log_simplification_l3991_399155


namespace union_of_A_and_B_l3991_399112

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Iic 2 ∪ Ioi 3 := by sorry

end union_of_A_and_B_l3991_399112


namespace three_hundredth_term_of_sequence_l3991_399195

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem three_hundredth_term_of_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 8) (h₂ : a₂ = -8) :
  geometric_sequence a₁ (a₂ / a₁) 300 = -8 := by
  sorry

end three_hundredth_term_of_sequence_l3991_399195


namespace fred_final_collection_l3991_399148

/-- Represents the types of coins Fred has --/
inductive Coin
  | Dime
  | Quarter
  | Nickel

/-- Represents Fred's coin collection --/
structure CoinCollection where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

def initial_collection : CoinCollection :=
  { dimes := 7, quarters := 4, nickels := 12 }

def borrowed : CoinCollection :=
  { dimes := 3, quarters := 2, nickels := 0 }

def returned : CoinCollection :=
  { dimes := 0, quarters := 1, nickels := 5 }

def found_cents : ℕ := 50

def cents_per_dime : ℕ := 10

theorem fred_final_collection :
  ∃ (final : CoinCollection),
    final.dimes = 9 ∧
    final.quarters = 3 ∧
    final.nickels = 17 ∧
    final.dimes = initial_collection.dimes - borrowed.dimes + found_cents / cents_per_dime ∧
    final.quarters = initial_collection.quarters - borrowed.quarters + returned.quarters ∧
    final.nickels = initial_collection.nickels + returned.nickels :=
  sorry

end fred_final_collection_l3991_399148


namespace complex_fraction_simplification_l3991_399171

theorem complex_fraction_simplification : 
  (((10^4+324)*(22^4+324)*(34^4+324)*(46^4+324)*(58^4+324)) / 
   ((4^4+324)*(16^4+324)*(28^4+324)*(40^4+324)*(52^4+324))) = 373 := by
  sorry

end complex_fraction_simplification_l3991_399171


namespace parallel_lines_problem_l3991_399120

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, m1 * x + y = b1 ↔ m2 * x + y = b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_problem (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - 1 = 0 ↔ 6 * x + a * y + 2 = 0) →
  a = 3 :=
by sorry

end parallel_lines_problem_l3991_399120


namespace highest_probability_red_card_l3991_399164

theorem highest_probability_red_card (total_cards : Nat) (ace_cards : Nat) (heart_cards : Nat) (king_cards : Nat) (red_cards : Nat) :
  total_cards = 52 →
  ace_cards = 4 →
  heart_cards = 13 →
  king_cards = 4 →
  red_cards = 26 →
  (red_cards : ℚ) / total_cards > (heart_cards : ℚ) / total_cards ∧
  (red_cards : ℚ) / total_cards > (ace_cards : ℚ) / total_cards ∧
  (red_cards : ℚ) / total_cards > (king_cards : ℚ) / total_cards :=
by sorry

end highest_probability_red_card_l3991_399164


namespace incorrect_regression_equation_l3991_399187

-- Define the sample means
def x_mean : ℝ := 2
def y_mean : ℝ := 3

-- Define the proposed linear regression equation
def proposed_equation (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem incorrect_regression_equation :
  ¬(proposed_equation x_mean = y_mean) :=
sorry

end incorrect_regression_equation_l3991_399187


namespace fraction_cube_theorem_l3991_399114

theorem fraction_cube_theorem :
  (2 : ℚ) / 5 ^ 3 = 8 / 125 :=
by sorry

end fraction_cube_theorem_l3991_399114


namespace quadratic_inequality_solution_range_l3991_399145

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (a < -4) ↔ 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) :=
by sorry

end quadratic_inequality_solution_range_l3991_399145


namespace final_elevation_proof_l3991_399172

def elevation_problem (start_elevation : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  start_elevation - rate * time

theorem final_elevation_proof (start_elevation : ℝ) (rate : ℝ) (time : ℝ)
  (h1 : start_elevation = 400)
  (h2 : rate = 10)
  (h3 : time = 5) :
  elevation_problem start_elevation rate time = 350 := by
  sorry

end final_elevation_proof_l3991_399172


namespace polynomial_difference_divisibility_l3991_399198

theorem polynomial_difference_divisibility
  (p : Polynomial ℤ) (b c : ℤ) (h : b ≠ c) :
  (b - c) ∣ (p.eval b - p.eval c) :=
by
  sorry

end polynomial_difference_divisibility_l3991_399198


namespace carnation_bouquets_problem_l3991_399150

/-- Proves that given five bouquets of carnations with specified conditions,
    the sum of carnations in the fourth and fifth bouquets is 34. -/
theorem carnation_bouquets_problem (b1 b2 b3 b4 b5 : ℕ) : 
  b1 = 9 → b2 = 14 → b3 = 18 → 
  (b1 + b2 + b3 + b4 + b5) / 5 = 15 →
  b4 + b5 = 34 := by
sorry

end carnation_bouquets_problem_l3991_399150


namespace quadratic_real_solutions_l3991_399109

theorem quadratic_real_solutions (p : ℝ) :
  (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 := by
  sorry

end quadratic_real_solutions_l3991_399109


namespace remaining_fruits_theorem_l3991_399162

/-- Represents the number of fruits in a bag -/
structure FruitBag where
  apples : ℕ
  oranges : ℕ
  mangoes : ℕ

/-- Calculates the total number of fruits in the bag -/
def FruitBag.total (bag : FruitBag) : ℕ :=
  bag.apples + bag.oranges + bag.mangoes

/-- Represents Luisa's actions on the fruit bag -/
def luisa_action (bag : FruitBag) : FruitBag :=
  { apples := bag.apples - 2,
    oranges := bag.oranges - 4,
    mangoes := bag.mangoes - (2 * bag.mangoes / 3) }

/-- The theorem to be proved -/
theorem remaining_fruits_theorem (initial_bag : FruitBag)
    (h1 : initial_bag.apples = 7)
    (h2 : initial_bag.oranges = 8)
    (h3 : initial_bag.mangoes = 15) :
    (luisa_action initial_bag).total = 14 := by
  sorry


end remaining_fruits_theorem_l3991_399162


namespace intersection_of_three_lines_l3991_399183

/-- Given three lines in the plane that intersect at two points, 
    prove that the parameter a must be either 1 or -2. -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (y₁ + 2*x₁ - 4 = 0 ∧ x₁ - y₁ + 1 = 0 ∧ a*x₁ - y₁ + 2 = 0) ∧
    (y₂ + 2*x₂ - 4 = 0 ∧ x₂ - y₂ + 1 = 0 ∧ a*x₂ - y₂ + 2 = 0) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a = 1 ∨ a = -2 :=
by sorry

end intersection_of_three_lines_l3991_399183


namespace tangent_lines_count_l3991_399192

theorem tangent_lines_count : ∃! (s : Finset ℝ), 
  (∀ x₀ ∈ s, x₀ * Real.exp x₀ * (x₀^2 - x₀ - 4) = 0) ∧ 
  Finset.card s = 3 := by
  sorry

end tangent_lines_count_l3991_399192


namespace all_five_digit_sum_30_div_9_l3991_399136

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem all_five_digit_sum_30_div_9 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 30 → n % 9 = 0 :=
sorry

end all_five_digit_sum_30_div_9_l3991_399136


namespace sum_of_first_20_odd_integers_greater_than_10_l3991_399144

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The 20th term of the arithmetic sequence starting at 11 with common difference 2 -/
def a₂₀ : ℕ := 11 + 19 * 2

theorem sum_of_first_20_odd_integers_greater_than_10 :
  arithmetic_sum 11 2 20 = 600 := by
  sorry

end sum_of_first_20_odd_integers_greater_than_10_l3991_399144


namespace quadratic_equation_roots_and_discriminant_l3991_399105

theorem quadratic_equation_roots_and_discriminant :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := 0
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  let discriminant := b^2 - 4*a*c
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ 
              (x₁ = 0 ∧ x₂ = -5) ∧
              discriminant = 25 :=
by
  sorry

end quadratic_equation_roots_and_discriminant_l3991_399105


namespace suit_price_calculation_l3991_399133

theorem suit_price_calculation (original_price : ℝ) : 
  original_price * 1.25 * 0.75 = 187.5 → original_price = 200 := by
  sorry

end suit_price_calculation_l3991_399133


namespace sphere_surface_area_l3991_399123

/-- The surface area of a sphere, given specific conditions for a hemisphere --/
theorem sphere_surface_area (r : ℝ) 
  (h1 : π * r^2 = 3)  -- area of the base of the hemisphere
  (h2 : 3 * π * r^2 = 9)  -- total surface area of the hemisphere
  : 4 * π * r^2 = 12 := by
  sorry

#check sphere_surface_area

end sphere_surface_area_l3991_399123


namespace max_k_value_l3991_399125

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3/2 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ 
    6 = (3/2)^2 * (x'^2 / y'^2 + y'^2 / x'^2) + (3/2) * (x' / y' + y' / x') :=
sorry

end max_k_value_l3991_399125


namespace power_product_rule_l3991_399103

theorem power_product_rule (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end power_product_rule_l3991_399103


namespace range_of_m_l3991_399101

theorem range_of_m (x y m : ℝ) : 
  (x + 2*y = 4*m) → 
  (2*x + y = 2*m + 1) → 
  (-1 < x - y) → 
  (x - y < 0) → 
  (1/2 < m ∧ m < 1) := by
sorry

end range_of_m_l3991_399101


namespace squared_sum_ge_double_product_l3991_399177

theorem squared_sum_ge_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b ∧ (a^2 + b^2 = 2*a*b ↔ a = b) := by
  sorry

end squared_sum_ge_double_product_l3991_399177


namespace flu_infection_rate_flu_infection_rate_proof_l3991_399185

theorem flu_infection_rate : ℝ → Prop :=
  fun x => (1 + x + x * (1 + x) = 144) → x = 11

-- The proof of the theorem
theorem flu_infection_rate_proof : flu_infection_rate 11 := by
  sorry

end flu_infection_rate_flu_infection_rate_proof_l3991_399185


namespace min_value_sum_squares_l3991_399186

theorem min_value_sum_squares (x y : ℝ) (h : x + y = 4) :
  ∃ (m : ℝ), (∀ a b : ℝ, a + b = 4 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 8 := by
  sorry

end min_value_sum_squares_l3991_399186


namespace trash_cans_redistribution_l3991_399134

/-- The number of trash cans in Veteran's Park after the redistribution -/
def final_trash_cans_veterans_park (initial_veterans_park : ℕ) : ℕ :=
  let initial_central_park := initial_veterans_park / 2 + 8
  let moved_cans := initial_central_park / 2
  initial_veterans_park + moved_cans

/-- Theorem stating that given 24 initial trash cans in Veteran's Park, 
    the final number of trash cans in Veteran's Park is 34 -/
theorem trash_cans_redistribution :
  final_trash_cans_veterans_park 24 = 34 := by
  sorry

end trash_cans_redistribution_l3991_399134


namespace sequence_sum_l3991_399184

theorem sequence_sum (n : ℕ) (S_n : ℝ) (a : ℕ → ℝ) : 
  (∀ k ≥ 1, a k = 1 / (Real.sqrt (k + 1) + Real.sqrt k)) →
  S_n = Real.sqrt 101 - 1 →
  n = 100 := by
  sorry

end sequence_sum_l3991_399184


namespace unique_n_modulo_101_l3991_399169

theorem unique_n_modulo_101 : ∃! n : ℤ, 0 ≤ n ∧ n < 101 ∧ (100 * n) % 101 = 72 % 101 ∧ n = 29 := by
  sorry

end unique_n_modulo_101_l3991_399169


namespace linear_function_composition_function_transformation_l3991_399179

-- Part 1
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  is_linear f → (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
sorry

-- Part 2
theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = 2 * x^2 - x + 1) →
  (∀ x, f x = 2 * x^2 - 3 * x + 2) :=
sorry

end linear_function_composition_function_transformation_l3991_399179


namespace decimal_existence_l3991_399140

theorem decimal_existence :
  (∃ (a b : ℚ), 3.5 < a ∧ a < 3.6 ∧ 3.5 < b ∧ b < 3.6 ∧ a ≠ b) ∧
  (∃ (x y z : ℚ), 0 < x ∧ x < 0.1 ∧ 0 < y ∧ y < 0.1 ∧ 0 < z ∧ z < 0.1 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by sorry

end decimal_existence_l3991_399140


namespace mean_of_combined_sets_l3991_399126

theorem mean_of_combined_sets (set1_count set1_mean set2_count set2_mean : ℚ) 
  (h1 : set1_count = 4)
  (h2 : set1_mean = 15)
  (h3 : set2_count = 8)
  (h4 : set2_mean = 20) :
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 55 / 3 := by
  sorry

end mean_of_combined_sets_l3991_399126


namespace cube_root_seven_to_sixth_l3991_399193

theorem cube_root_seven_to_sixth (x : ℝ) : x = 7^(1/3) → x^6 = 49 := by sorry

end cube_root_seven_to_sixth_l3991_399193


namespace jesse_banana_sharing_l3991_399132

theorem jesse_banana_sharing (total_bananas : ℕ) (bananas_per_friend : ℕ) (h1 : total_bananas = 21) (h2 : bananas_per_friend = 7) :
  total_bananas / bananas_per_friend = 3 :=
by sorry

end jesse_banana_sharing_l3991_399132


namespace repeating_decimal_equals_fraction_l3991_399110

/-- The repeating decimal 0.363636... as a real number -/
def repeating_decimal : ℚ := 36 / 99

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end repeating_decimal_equals_fraction_l3991_399110


namespace ones_digit_sum_powers_2011_l3991_399167

theorem ones_digit_sum_powers_2011 : ∃ n : ℕ, n < 10 ∧ (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 := by
  sorry

end ones_digit_sum_powers_2011_l3991_399167


namespace correct_total_items_l3991_399141

/-- Represents the requirements for a packed lunch --/
structure LunchRequirements where
  sandwiches_per_student : ℕ
  bread_slices_per_sandwich : ℕ
  chips_per_student : ℕ
  apples_per_student : ℕ
  granola_bars_per_student : ℕ

/-- Represents the number of students in each group --/
structure StudentGroups where
  group_a : ℕ
  group_b : ℕ
  group_c : ℕ

/-- Calculates the total number of items needed for packed lunches --/
def calculate_total_items (req : LunchRequirements) (groups : StudentGroups) :
  (ℕ × ℕ × ℕ × ℕ) :=
  let total_students := groups.group_a + groups.group_b + groups.group_c
  let total_bread_slices := total_students * req.sandwiches_per_student * req.bread_slices_per_sandwich
  let total_chips := total_students * req.chips_per_student
  let total_apples := total_students * req.apples_per_student
  let total_granola_bars := total_students * req.granola_bars_per_student
  (total_bread_slices, total_chips, total_apples, total_granola_bars)

/-- Theorem stating the correct calculation of total items needed --/
theorem correct_total_items :
  let req : LunchRequirements := {
    sandwiches_per_student := 2,
    bread_slices_per_sandwich := 4,
    chips_per_student := 1,
    apples_per_student := 3,
    granola_bars_per_student := 1
  }
  let groups : StudentGroups := {
    group_a := 10,
    group_b := 15,
    group_c := 20
  }
  calculate_total_items req groups = (360, 45, 135, 45) :=
by
  sorry


end correct_total_items_l3991_399141


namespace trillion_equals_ten_to_sixteen_l3991_399111

theorem trillion_equals_ten_to_sixteen :
  let ten_thousand : ℕ := 10^4
  let hundred_million : ℕ := 10^8
  let trillion : ℕ := ten_thousand * ten_thousand * hundred_million
  trillion = 10^16 := by
  sorry

end trillion_equals_ten_to_sixteen_l3991_399111


namespace cost_price_calculation_l3991_399124

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) : 
  selling_price = 600 → profit_percentage = 60 → 
  ∃ (cost_price : ℚ), cost_price = 375 ∧ selling_price = cost_price * (1 + profit_percentage / 100) :=
by sorry

end cost_price_calculation_l3991_399124


namespace annulus_area_l3991_399173

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (r s x : ℝ) (hr : r > 0) (hs : s > 0) (hrs : r > s) :
  let P := Real.sqrt (r^2 - s^2)
  x^2 = r^2 - s^2 →
  π * (r^2 - s^2) = π * x^2 := by sorry

end annulus_area_l3991_399173


namespace min_days_team_a_is_ten_l3991_399194

/-- Represents the greening project parameters and constraints -/
structure GreeningProject where
  totalArea : ℝ
  teamARate : ℝ
  teamBRate : ℝ
  teamADailyCost : ℝ
  teamBDailyCost : ℝ
  totalBudget : ℝ

/-- Calculates the minimum number of days Team A should work -/
def minDaysTeamA (project : GreeningProject) : ℝ :=
  sorry

/-- Theorem stating the minimum number of days Team A should work -/
theorem min_days_team_a_is_ten (project : GreeningProject) :
  project.totalArea = 1800 ∧
  project.teamARate = 2 * project.teamBRate ∧
  400 / project.teamARate + 4 = 400 / project.teamBRate ∧
  project.teamADailyCost = 0.4 ∧
  project.teamBDailyCost = 0.25 ∧
  project.totalBudget = 8 →
  minDaysTeamA project = 10 := by
  sorry

#check min_days_team_a_is_ten

end min_days_team_a_is_ten_l3991_399194


namespace crayons_in_drawer_l3991_399190

/-- The number of crayons remaining in a drawer after some are removed. -/
def crayons_remaining (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: Given 7 initial crayons and 3 removed, 4 crayons remain. -/
theorem crayons_in_drawer : crayons_remaining 7 3 = 4 := by
  sorry

end crayons_in_drawer_l3991_399190


namespace factorial_simplification_l3991_399188

theorem factorial_simplification : (12 : ℕ).factorial / ((10 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 1320 / 13 := by
  sorry

end factorial_simplification_l3991_399188


namespace digit_properties_l3991_399156

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem digit_properties :
  ∀ (a b : Nat), a ∈ Digits → b ∈ Digits → a ≠ b →
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x ≠ y → (a + b) * (a * b) ≥ (x + y) * (x * y)) ∧
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x ≠ y → (0 + 1) * (0 * 1) ≤ (x + y) * (x * y)) ∧
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x + y = 10 ↔ 
      ((x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∨ (x = 3 ∧ y = 7) ∨ (x = 4 ∧ y = 6) ∨
       (x = 9 ∧ y = 1) ∨ (x = 8 ∧ y = 2) ∨ (x = 7 ∧ y = 3) ∨ (x = 6 ∧ y = 4))) :=
by
  sorry

end digit_properties_l3991_399156


namespace vegetarian_eaters_l3991_399122

theorem vegetarian_eaters (only_veg : ℕ) (only_non_veg : ℕ) (both : ℕ) 
  (h1 : only_veg = 15) 
  (h2 : only_non_veg = 8) 
  (h3 : both = 11) : 
  only_veg + both = 26 := by
  sorry

end vegetarian_eaters_l3991_399122


namespace min_links_remove_10x10_grid_l3991_399189

/-- Represents a grid with horizontal and vertical lines -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Calculates the total number of links in the grid -/
def total_links (g : Grid) : ℕ :=
  (g.rows * g.vertical_lines) + (g.cols * g.horizontal_lines)

/-- Calculates the number of interior nodes in the grid -/
def interior_nodes (g : Grid) : ℕ :=
  (g.rows - 1) * (g.cols - 1)

/-- The minimum number of links to remove -/
def min_links_to_remove (g : Grid) : ℕ := 41

/-- Theorem stating the minimum number of links to remove for a 10x10 grid -/
theorem min_links_remove_10x10_grid :
  let g : Grid := { rows := 10, cols := 10, horizontal_lines := 11, vertical_lines := 11 }
  min_links_to_remove g = 41 :=
by sorry

end min_links_remove_10x10_grid_l3991_399189


namespace wine_sales_regression_l3991_399137

/-- Linear regression problem for white wine sales and unit cost -/
theorem wine_sales_regression 
  (x_mean : ℝ) 
  (y_mean : ℝ) 
  (sum_x_squared : ℝ) 
  (sum_xy : ℝ) 
  (n : ℕ) 
  (h_x_mean : x_mean = 7/2)
  (h_y_mean : y_mean = 71)
  (h_sum_x_squared : sum_x_squared = 79)
  (h_sum_xy : sum_xy = 1481)
  (h_n : n = 6) :
  let b := (sum_xy - n * x_mean * y_mean) / (sum_x_squared - n * x_mean^2)
  ∃ ε > 0, |b + 1.8182| < ε :=
sorry

end wine_sales_regression_l3991_399137


namespace problem_statement_l3991_399116

theorem problem_statement : (1 / ((-2^4)^2)) * ((-2)^7) = -1/2 := by
  sorry

end problem_statement_l3991_399116


namespace cos_three_pi_halves_l3991_399115

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end cos_three_pi_halves_l3991_399115


namespace difference_of_two_numbers_l3991_399163

theorem difference_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (product_eq : x * y = 200) : 
  |x - y| = 10 := by
sorry

end difference_of_two_numbers_l3991_399163


namespace p_true_and_q_false_p_and_not_q_true_l3991_399159

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Theorem stating that p is true and q is false
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

-- Theorem stating that p ∧ ¬q is true
theorem p_and_not_q_true : p ∧ ¬q := by
  sorry

end p_true_and_q_false_p_and_not_q_true_l3991_399159


namespace x_fourth_equals_one_l3991_399139

theorem x_fourth_equals_one (x : ℝ) 
  (h : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = Real.sqrt 2) : 
  x^4 = 1 := by
sorry

end x_fourth_equals_one_l3991_399139


namespace evaluate_expression_l3991_399151

theorem evaluate_expression : (4^4 - 4*(4-2)^4)^4 = 1358954496 := by
  sorry

end evaluate_expression_l3991_399151


namespace female_rainbow_trout_count_l3991_399196

theorem female_rainbow_trout_count :
  -- Total speckled trout
  ∀ (total_speckled : ℕ),
  -- Male and female speckled trout
  ∀ (male_speckled female_speckled : ℕ),
  -- Male rainbow trout
  ∀ (male_rainbow : ℕ),
  -- Total trout
  ∀ (total_trout : ℕ),
  -- Conditions
  total_speckled = 645 →
  male_speckled = 2 * female_speckled + 45 →
  4 * male_rainbow = 3 * female_speckled →
  20 * male_rainbow = 3 * total_trout →
  -- Conclusion
  total_trout - total_speckled - male_rainbow = 205 :=
by
  sorry

end female_rainbow_trout_count_l3991_399196


namespace parabola_c_value_l3991_399178

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = -3 →   -- vertex at (-3, 1)
  p.x_coord 3 = -1 →   -- passes through (-1, 3)
  p.c = -5/2 := by
    sorry

end parabola_c_value_l3991_399178


namespace exactly_two_true_propositions_l3991_399106

-- Define the propositions
def corresponding_angles_equal : Prop := sorry

def parallel_lines_supplementary_angles : Prop := sorry

def perpendicular_lines_parallel : Prop := sorry

-- Theorem statement
theorem exactly_two_true_propositions :
  (corresponding_angles_equal = false ∧
   parallel_lines_supplementary_angles = true ∧
   perpendicular_lines_parallel = true) :=
by sorry

end exactly_two_true_propositions_l3991_399106


namespace inequalities_theorem_l3991_399118

theorem inequalities_theorem (a b : ℝ) (m n : ℕ) 
    (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n) : 
  (a^n + b^n) * (a^m + b^m) ≤ 2 * (a^(m+n) + b^(m+n)) ∧ 
  (a + b) / 2 * (a^2 + b^2) / 2 * (a^3 + b^3) / 2 ≤ (a^6 + b^6) / 2 :=
by sorry

end inequalities_theorem_l3991_399118


namespace picture_area_l3991_399166

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h_frame_area : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 := by
  sorry

end picture_area_l3991_399166


namespace marks_deposit_l3991_399182

theorem marks_deposit (mark_deposit : ℝ) (bryan_deposit : ℝ) : 
  bryan_deposit = 5 * mark_deposit - 40 →
  mark_deposit + bryan_deposit = 400 →
  mark_deposit = 400 / 6 := by
sorry

end marks_deposit_l3991_399182
