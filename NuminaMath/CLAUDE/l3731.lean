import Mathlib

namespace colored_isosceles_triangle_exists_l3731_373132

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A colored vertex in a polygon -/
def ColoredVertex (n : ℕ) (p : RegularPolygon n) := Fin n

/-- Three vertices form an isosceles triangle -/
def IsIsoscelesTriangle (n : ℕ) (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop := sorry

theorem colored_isosceles_triangle_exists 
  (p : RegularPolygon 5000) 
  (colored : Finset (ColoredVertex 5000 p)) 
  (h : colored.card = 2001) : 
  ∃ (v1 v2 v3 : ColoredVertex 5000 p), 
    v1 ∈ colored ∧ v2 ∈ colored ∧ v3 ∈ colored ∧ 
    IsIsoscelesTriangle 5000 p v1 v2 v3 :=
  sorry

end colored_isosceles_triangle_exists_l3731_373132


namespace min_dot_product_on_ellipse_l3731_373113

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the center and left focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of OP and FP
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem min_dot_product_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y →
  dot_product x y ≥ 6 :=
sorry

end min_dot_product_on_ellipse_l3731_373113


namespace initial_ducks_l3731_373152

theorem initial_ducks (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 20 → total = 33 → initial + joined = total → initial = 13 := by
sorry

end initial_ducks_l3731_373152


namespace belinda_passed_twenty_percent_l3731_373120

/-- The percentage of flyers Belinda passed out -/
def belinda_percentage (total flyers : ℕ) (ryan alyssa scott : ℕ) : ℚ :=
  (total - (ryan + alyssa + scott)) / total * 100

/-- Theorem stating that Belinda passed out 20% of the flyers -/
theorem belinda_passed_twenty_percent :
  belinda_percentage 200 42 67 51 = 20 := by
  sorry

end belinda_passed_twenty_percent_l3731_373120


namespace bob_distance_at_meeting_l3731_373171

/-- The distance between point X and point Y in miles -/
def total_distance : ℝ := 52

/-- Yolanda's walking speed in miles per hour -/
def yolanda_speed : ℝ := 3

/-- Bob's walking speed in miles per hour -/
def bob_speed : ℝ := 4

/-- The time difference between Yolanda's and Bob's start in hours -/
def time_difference : ℝ := 1

/-- The theorem stating that Bob walked 28 miles when they met -/
theorem bob_distance_at_meeting : 
  ∃ (t : ℝ), t > 0 ∧ yolanda_speed * (t + time_difference) + bob_speed * t = total_distance ∧ 
  bob_speed * t = 28 := by
  sorry

end bob_distance_at_meeting_l3731_373171


namespace largest_non_sum_42multiple_composite_l3731_373195

def is_composite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_sum_of_42multiple_and_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 42 * a + b ∧ a > 0 ∧ is_composite b

theorem largest_non_sum_42multiple_composite :
  (∀ n : ℕ, n > 215 → is_sum_of_42multiple_and_composite n) ∧
  ¬is_sum_of_42multiple_and_composite 215 :=
sorry

end largest_non_sum_42multiple_composite_l3731_373195


namespace robin_chocolate_chip_cookies_l3731_373115

/-- Given information about Robin's cookies --/
structure CookieInfo where
  cookies_per_bag : ℕ
  oatmeal_cookies : ℕ
  baggies : ℕ

/-- Calculate the number of chocolate chip cookies --/
def chocolate_chip_cookies (info : CookieInfo) : ℕ :=
  info.baggies * info.cookies_per_bag - info.oatmeal_cookies

/-- Theorem: Robin has 23 chocolate chip cookies --/
theorem robin_chocolate_chip_cookies :
  let info : CookieInfo := {
    cookies_per_bag := 6,
    oatmeal_cookies := 25,
    baggies := 8
  }
  chocolate_chip_cookies info = 23 := by
  sorry

end robin_chocolate_chip_cookies_l3731_373115


namespace cube_root_of_3x_plus_4y_is_3_l3731_373168

theorem cube_root_of_3x_plus_4y_is_3 (x y : ℝ) (h : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 3) :
  (3 * x + 4 * y) ^ (1/3 : ℝ) = 3 := by sorry

end cube_root_of_3x_plus_4y_is_3_l3731_373168


namespace probability_two_same_color_l3731_373103

def total_balls : ℕ := 6
def balls_per_color : ℕ := 2
def num_colors : ℕ := 3
def balls_drawn : ℕ := 3

def total_ways : ℕ := Nat.choose total_balls balls_drawn

def ways_two_same_color : ℕ := num_colors * (Nat.choose balls_per_color 2) * (total_balls - balls_per_color)

theorem probability_two_same_color :
  (ways_two_same_color : ℚ) / total_ways = 3 / 5 := by sorry

end probability_two_same_color_l3731_373103


namespace distribute_negative_three_l3731_373181

theorem distribute_negative_three (a : ℝ) : -3 * (a - 1) = 3 - 3 * a := by
  sorry

end distribute_negative_three_l3731_373181


namespace ball_arrangement_theorem_l3731_373172

/-- The number of ways to arrange 8 balls in a row, with 5 red and 3 white,
    such that exactly 3 consecutive balls are red -/
def arrangement_count : ℕ := 24

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutive_red : ℕ := 3

theorem ball_arrangement_theorem :
  arrangement_count = 24 ∧
  total_balls = 8 ∧
  red_balls = 5 ∧
  white_balls = 3 ∧
  consecutive_red = 3 :=
by sorry

end ball_arrangement_theorem_l3731_373172


namespace original_slices_count_l3731_373133

/-- The number of slices in the original loaf of bread -/
def S : ℕ := 27

/-- The number of slices Andy ate -/
def slices_andy_ate : ℕ := 6

/-- The number of slices Emma used for toast -/
def slices_for_toast : ℕ := 20

/-- The number of slices left after making toast -/
def slices_left : ℕ := 1

/-- Theorem stating that the original number of slices equals the sum of slices eaten,
    used for toast, and left over -/
theorem original_slices_count : S = slices_andy_ate + slices_for_toast + slices_left :=
by sorry

end original_slices_count_l3731_373133


namespace roberto_healthcare_contribution_l3731_373176

/-- Calculates the healthcare contribution in cents per hour given an hourly wage in dollars and a contribution rate. -/
def healthcare_contribution (hourly_wage : ℚ) (contribution_rate : ℚ) : ℚ :=
  hourly_wage * 100 * contribution_rate

/-- Proves that Roberto's healthcare contribution is 50 cents per hour. -/
theorem roberto_healthcare_contribution :
  healthcare_contribution 25 (2/100) = 50 := by
  sorry

#eval healthcare_contribution 25 (2/100)

end roberto_healthcare_contribution_l3731_373176


namespace exam_score_calculation_l3731_373199

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℤ) (correct_answers : ℕ) (marks_per_wrong : ℤ) :
  total_questions = 80 →
  total_marks = 130 →
  correct_answers = 42 →
  marks_per_wrong = -1 →
  ∃ (marks_per_correct : ℤ),
    marks_per_correct * correct_answers + marks_per_wrong * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct = 4 :=
by sorry

end exam_score_calculation_l3731_373199


namespace remainder_theorem_l3731_373145

theorem remainder_theorem (n : ℕ) : (3^(2*n) + 8) % 8 = 1 := by
  sorry

end remainder_theorem_l3731_373145


namespace vitamin_c_content_l3731_373175

/-- The amount of vitamin C (in mg) in one 8-oz glass of apple juice -/
def apple_juice_vc : ℕ := 103

/-- The total amount of vitamin C (in mg) in one 8-oz glass each of apple juice and orange juice -/
def total_vc : ℕ := 185

/-- The amount of vitamin C (in mg) in one 8-oz glass of orange juice -/
def orange_juice_vc : ℕ := total_vc - apple_juice_vc

/-- Theorem: Two 8-oz glasses of apple juice and three 8-oz glasses of orange juice contain 452 mg of vitamin C -/
theorem vitamin_c_content : 2 * apple_juice_vc + 3 * orange_juice_vc = 452 := by
  sorry

end vitamin_c_content_l3731_373175


namespace fraction_problem_l3731_373108

theorem fraction_problem (f : ℚ) : 
  (f * 20 + 5 = 15) → f = 1/2 := by
  sorry

end fraction_problem_l3731_373108


namespace point_in_second_quadrant_l3731_373119

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point P(-1, m^2+1) is in the second quadrant for any real number m -/
theorem point_in_second_quadrant (m : ℝ) : is_in_second_quadrant (-1) (m^2 + 1) := by
  sorry

end point_in_second_quadrant_l3731_373119


namespace car_cost_proof_l3731_373118

def down_payment : ℕ := 8000
def num_payments : ℕ := 48
def monthly_payment : ℕ := 525
def interest_rate : ℚ := 5 / 100

def total_car_cost : ℕ := 34460

theorem car_cost_proof :
  down_payment +
  num_payments * monthly_payment +
  num_payments * (interest_rate * monthly_payment).floor = total_car_cost := by
  sorry

end car_cost_proof_l3731_373118


namespace prime_pairs_theorem_l3731_373180

theorem prime_pairs_theorem : 
  ∀ p q : ℕ, 
    Prime p → Prime q → Prime (p * q + p - 6) → 
    ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end prime_pairs_theorem_l3731_373180


namespace optimal_base_side_l3731_373142

/-- A lidless water tank with square base -/
structure WaterTank where
  volume : ℝ
  baseSide : ℝ
  height : ℝ

/-- The surface area of a lidless water tank -/
def surfaceArea (tank : WaterTank) : ℝ :=
  tank.baseSide ^ 2 + 4 * tank.baseSide * tank.height

/-- The volume constraint for the water tank -/
def volumeConstraint (tank : WaterTank) : Prop :=
  tank.volume = tank.baseSide ^ 2 * tank.height

/-- Theorem: The base side length that minimizes the surface area of a lidless water tank
    with volume 256 cubic units and a square base is 8 units -/
theorem optimal_base_side :
  ∃ (tank : WaterTank),
    tank.volume = 256 ∧
    volumeConstraint tank ∧
    (∀ (other : WaterTank),
      other.volume = 256 →
      volumeConstraint other →
      surfaceArea tank ≤ surfaceArea other) ∧
    tank.baseSide = 8 :=
  sorry

end optimal_base_side_l3731_373142


namespace temperature_conversion_l3731_373183

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 221 → t = 105 := by
  sorry

end temperature_conversion_l3731_373183


namespace fraction_to_decimal_l3731_373141

theorem fraction_to_decimal : (49 : ℚ) / 160 = 0.30625 := by
  sorry

end fraction_to_decimal_l3731_373141


namespace school_distance_is_two_point_five_l3731_373191

/-- The distance from Philip's house to the school in miles -/
def school_distance : ℝ := sorry

/-- The round trip distance to the market in miles -/
def market_round_trip : ℝ := 4

/-- The number of round trips to school per week -/
def school_trips_per_week : ℕ := 8

/-- The number of round trips to the market per week -/
def market_trips_per_week : ℕ := 1

/-- The total mileage for a typical week in miles -/
def total_weekly_mileage : ℝ := 44

theorem school_distance_is_two_point_five : 
  school_distance = 2.5 := by
  sorry

end school_distance_is_two_point_five_l3731_373191


namespace impossible_to_reach_target_l3731_373101

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The initial grid state with all zeros -/
def initial_grid : Grid := fun _ _ => 0

/-- Represents a 2x2 subgrid position in the 3x3 grid -/
inductive SubgridPos
| TopLeft
| TopRight
| BottomLeft
| BottomRight

/-- Applies a 2x2 increment operation to the grid at the specified position -/
def apply_operation (g : Grid) (pos : SubgridPos) : Grid :=
  fun i j =>
    match pos with
    | SubgridPos.TopLeft => if i < 2 && j < 2 then g i j + 1 else g i j
    | SubgridPos.TopRight => if i < 2 && j > 0 then g i j + 1 else g i j
    | SubgridPos.BottomLeft => if i > 0 && j < 2 then g i j + 1 else g i j
    | SubgridPos.BottomRight => if i > 0 && j > 0 then g i j + 1 else g i j

/-- The target grid state we want to prove is impossible to reach -/
def target_grid : Grid :=
  fun i j => if i = 1 && j = 1 then 4 else 1

/-- Theorem stating that it's impossible to reach the target grid from the initial grid
    using any sequence of 2x2 increment operations -/
theorem impossible_to_reach_target :
  ∀ (ops : List SubgridPos),
    (ops.foldl apply_operation initial_grid) ≠ target_grid :=
sorry

end impossible_to_reach_target_l3731_373101


namespace complex_rational_equation_root_l3731_373162

theorem complex_rational_equation_root :
  ∃! x : ℚ, (3*x^2 + 5)/(x-2) - (3*x + 10)/4 + (5 - 9*x)/(x-2) + 2 = 0 ∧ x ≠ 2 :=
by sorry

end complex_rational_equation_root_l3731_373162


namespace fourth_term_of_geometric_progression_l3731_373154

theorem fourth_term_of_geometric_progression :
  let a₁ : ℝ := Real.sqrt 4
  let a₂ : ℝ := (4 : ℝ) ^ (1/4)
  let a₃ : ℝ := (4 : ℝ) ^ (1/8)
  let r : ℝ := a₂ / a₁
  let a₄ : ℝ := a₃ * r
  a₄ = (1/4 : ℝ) ^ (1/8) :=
by sorry

end fourth_term_of_geometric_progression_l3731_373154


namespace student_count_l3731_373109

theorem student_count (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 30 = total) : total = 600 := by
  sorry

end student_count_l3731_373109


namespace equal_sets_imply_a_eq_5_intersection_conditions_imply_a_eq_neg_2_l3731_373196

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem 1
theorem equal_sets_imply_a_eq_5 :
  ∀ a : ℝ, A a = B → a = 5 := by sorry

-- Theorem 2
theorem intersection_conditions_imply_a_eq_neg_2 :
  ∀ a : ℝ, (B ∩ A a ≠ ∅) ∧ (C ∩ A a = ∅) → a = -2 := by sorry

end equal_sets_imply_a_eq_5_intersection_conditions_imply_a_eq_neg_2_l3731_373196


namespace rhombus_area_l3731_373146

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 100 square units. -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 145 →
  d₂ - d₁ = 10 →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  (1/2) * d₁ * d₂ = 100 := by
  sorry

end rhombus_area_l3731_373146


namespace min_surface_area_circumscribed_sphere_l3731_373144

/-- The minimum surface area of a sphere circumscribed around a right rectangular prism --/
theorem min_surface_area_circumscribed_sphere (h : ℝ) (a : ℝ) :
  h = 3 →
  a * a = 7 / 2 →
  ∃ (S : ℝ), S = 16 * Real.pi ∧ ∀ (R : ℝ), R ≥ 2 → 4 * Real.pi * R^2 ≥ S :=
by sorry

end min_surface_area_circumscribed_sphere_l3731_373144


namespace divisor_problem_l3731_373155

theorem divisor_problem (x : ℕ) : x > 0 ∧ 83 = 9 * x + 2 → x = 9 := by
  sorry

end divisor_problem_l3731_373155


namespace pauls_remaining_crayons_l3731_373117

/-- Given that Paul initially had 479 crayons and lost or gave away 345 crayons,
    prove that he has 134 crayons left. -/
theorem pauls_remaining_crayons (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
    (h1 : initial = 479) 
    (h2 : lost = 345) 
    (h3 : remaining = initial - lost) : 
  remaining = 134 := by
  sorry

end pauls_remaining_crayons_l3731_373117


namespace income_calculation_l3731_373173

def original_tax_rate : ℝ := 0.46
def new_tax_rate : ℝ := 0.32
def differential_savings : ℝ := 5040

theorem income_calculation (income : ℝ) :
  (original_tax_rate - new_tax_rate) * income = differential_savings →
  income = 36000 := by
sorry

end income_calculation_l3731_373173


namespace f_divides_characterization_l3731_373127

def f (x : ℕ) : ℕ := x^2 + x + 1

def is_valid (n : ℕ) : Prop :=
  n = 1 ∨ 
  (Nat.Prime n ∧ n % 3 = 1) ∨ 
  (∃ p, Nat.Prime p ∧ p ≠ 3 ∧ n = p^2)

theorem f_divides_characterization (n : ℕ) :
  (∀ k : ℕ, k > 0 → k ∣ n → f k ∣ f n) ↔ is_valid n :=
sorry

end f_divides_characterization_l3731_373127


namespace restaurant_bill_proof_l3731_373111

/-- The total bill for a group of friends dining at a restaurant -/
def total_bill : ℕ := 270

/-- The number of friends dining at the restaurant -/
def num_friends : ℕ := 10

/-- The extra amount each paying friend contributes to cover the non-paying friend -/
def extra_contribution : ℕ := 3

/-- The number of friends who pay the bill -/
def num_paying_friends : ℕ := num_friends - 1

theorem restaurant_bill_proof :
  total_bill = num_paying_friends * (total_bill / num_friends + extra_contribution) :=
sorry

end restaurant_bill_proof_l3731_373111


namespace p_necessary_not_sufficient_for_q_l3731_373170

/-- A function f: ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The condition p: f(x) = x³ + 2x² + mx + 1 is monotonically increasing -/
def p (m : ℝ) : Prop :=
  MonotonicallyIncreasing (fun x => x^3 + 2*x^2 + m*x + 1)

/-- The condition q: m ≥ 8x / (x² + 4) holds for any x > 0 -/
def q (m : ℝ) : Prop :=
  ∀ x, x > 0 → m ≥ 8*x / (x^2 + 4)

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ m, q m → p m) ∧ (∃ m, p m ∧ ¬q m) := by sorry

end p_necessary_not_sufficient_for_q_l3731_373170


namespace decagon_triangles_l3731_373129

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def triangles_from_decagon : ℕ := Nat.choose decagon_vertices triangle_vertices

theorem decagon_triangles :
  triangles_from_decagon = 120 := by sorry

end decagon_triangles_l3731_373129


namespace bob_profit_l3731_373105

/-- Calculates the profit from breeding and selling show dogs -/
def dog_breeding_profit (num_dogs : ℕ) (cost_per_dog : ℚ) (num_puppies : ℕ) (price_per_puppy : ℚ) : ℚ :=
  num_puppies * price_per_puppy - num_dogs * cost_per_dog

/-- Bob's profit from breeding and selling show dogs -/
theorem bob_profit : 
  dog_breeding_profit 2 250 6 350 = 1600 :=
by sorry

end bob_profit_l3731_373105


namespace zeros_of_quadratic_function_l3731_373177

theorem zeros_of_quadratic_function (f : ℝ → ℝ) :
  (f = λ x => x^2 - x - 2) →
  (∀ x, f x = 0 ↔ x = -1 ∨ x = 2) :=
by sorry

end zeros_of_quadratic_function_l3731_373177


namespace least_positive_linear_combination_l3731_373193

theorem least_positive_linear_combination (x y : ℤ) : 
  ∃ (a b : ℤ), 24 * a + 18 * b = 6 ∧ 
  ∀ (c d : ℤ), 24 * c + 18 * d > 0 → 24 * c + 18 * d ≥ 6 := by
  sorry

end least_positive_linear_combination_l3731_373193


namespace expression_simplification_inequality_system_equivalence_l3731_373182

-- Part 1
theorem expression_simplification (a : ℝ) :
  (a - 3)^2 + a*(4 - a) = -2*a + 9 := by sorry

-- Part 2
theorem inequality_system_equivalence (x : ℝ) :
  -2 ≤ x ∧ x < 3 ↔ 3*x - 5 < x + 1 ∧ 2*(2*x - 1) ≥ 3*x - 4 := by sorry

end expression_simplification_inequality_system_equivalence_l3731_373182


namespace whiteboard_washing_time_l3731_373137

theorem whiteboard_washing_time 
  (kids : ℕ) 
  (whiteboards : ℕ) 
  (time : ℕ) 
  (h1 : kids = 4) 
  (h2 : whiteboards = 3) 
  (h3 : time = 20) :
  (1 : ℕ) * 6 * time = kids * whiteboards * 160 :=
sorry

end whiteboard_washing_time_l3731_373137


namespace cube_diagonal_l3731_373130

theorem cube_diagonal (surface_area : ℝ) (h : surface_area = 294) :
  let side_length := Real.sqrt (surface_area / 6)
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 7 * Real.sqrt 3 := by
  sorry

end cube_diagonal_l3731_373130


namespace monochromatic_unit_area_triangle_exists_l3731_373165

-- Define a type for colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Check if all vertices of a triangle have the same color
def monochromatic (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- Main theorem
theorem monochromatic_unit_area_triangle_exists (coloring : Coloring) :
  ∃ t : Triangle, triangleArea t = 1 ∧ monochromatic t coloring := by
  sorry


end monochromatic_unit_area_triangle_exists_l3731_373165


namespace change_received_correct_l3731_373157

/-- Calculates the change received when buying steak -/
def change_received (cost_per_pound : ℝ) (pounds_bought : ℝ) (amount_paid : ℝ) : ℝ :=
  amount_paid - (cost_per_pound * pounds_bought)

/-- Theorem: The change received when buying steak is correct -/
theorem change_received_correct (cost_per_pound : ℝ) (pounds_bought : ℝ) (amount_paid : ℝ) :
  change_received cost_per_pound pounds_bought amount_paid =
  amount_paid - (cost_per_pound * pounds_bought) := by
  sorry

#eval change_received 7 2 20

end change_received_correct_l3731_373157


namespace max_value_of_expression_l3731_373187

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) / (a^3 + b^3 + c^3 - 2*a*b*c)
  A ≤ 6 ∧ (A = 6 ↔ a = b ∧ b = c) := by
  sorry

end max_value_of_expression_l3731_373187


namespace spade_nested_calculation_l3731_373122

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_nested_calculation : spade 3 (spade 4 5) = -72 := by
  sorry

end spade_nested_calculation_l3731_373122


namespace triangle_property_l3731_373163

theorem triangle_property (A B C : Real) (h_triangle : A + B + C = Real.pi) 
  (h_condition : Real.sin A * Real.cos A = Real.sin B * Real.cos B) :
  (A = B ∨ C = Real.pi / 2) ∨ (B = C ∨ A = Real.pi / 2) ∨ (C = A ∨ B = Real.pi / 2) :=
sorry

end triangle_property_l3731_373163


namespace absolute_value_inequality_solution_set_l3731_373143

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end absolute_value_inequality_solution_set_l3731_373143


namespace power_of_256_l3731_373156

theorem power_of_256 : (256 : ℝ) ^ (5/4 : ℝ) = 1024 :=
by
  have h : 256 = 2^8 := by sorry
  sorry

end power_of_256_l3731_373156


namespace buses_in_five_days_l3731_373169

/-- Represents the number of buses leaving a station over multiple days -/
def buses_over_days (buses_per_half_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  buses_per_half_hour * 2 * hours_per_day * days

/-- Theorem stating that 120 buses leave the station over 5 days -/
theorem buses_in_five_days :
  buses_over_days 1 12 5 = 120 := by
  sorry

#eval buses_over_days 1 12 5

end buses_in_five_days_l3731_373169


namespace least_positive_integer_divisible_by_four_primes_l3731_373135

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
   p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
   n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m > 0 → m < n → 
   ¬(∃ q₁ q₂ q₃ q₄ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
     q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
     m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
  n = 210 :=
by sorry

end least_positive_integer_divisible_by_four_primes_l3731_373135


namespace grandmas_salad_ratio_l3731_373102

/-- Prove that the ratio of bacon bits to pickles is 4:1 given the conditions in Grandma's salad --/
theorem grandmas_salad_ratio : 
  ∀ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
    mushrooms = 3 →
    cherry_tomatoes = 2 * mushrooms →
    pickles = 4 * cherry_tomatoes →
    red_bacon_bits = 32 →
    3 * red_bacon_bits = bacon_bits →
    (bacon_bits : ℚ) / pickles = 4 / 1 := by
  sorry

end grandmas_salad_ratio_l3731_373102


namespace jenna_blouses_count_l3731_373184

/-- The number of blouses Jenna needs to dye -/
def num_blouses : ℕ := 100

/-- The number of dots per blouse -/
def dots_per_blouse : ℕ := 20

/-- The amount of dye (in ml) needed per dot -/
def dye_per_dot : ℕ := 10

/-- The number of bottles of dye Jenna needs to buy -/
def num_bottles : ℕ := 50

/-- The volume (in ml) of each bottle of dye -/
def bottle_volume : ℕ := 400

/-- Theorem stating that the number of blouses Jenna needs to dye is correct -/
theorem jenna_blouses_count : 
  num_blouses * (dots_per_blouse * dye_per_dot) = num_bottles * bottle_volume :=
sorry

end jenna_blouses_count_l3731_373184


namespace hyperbola_b_value_l3731_373148

/-- Hyperbola C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 2 - y^2 / 8 = 1

/-- Hyperbola C₂ -/
def C₂ (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Asymptote of C₁ -/
def asymptote_C₁ (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Asymptote of C₂ -/
def asymptote_C₂ (x y a b : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_b_value (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_same_asymptotes : ∀ x y, asymptote_C₁ x y ↔ asymptote_C₂ x y a b)
  (h_focal_length : 4 * Real.sqrt 5 = 2 * Real.sqrt (a^2 + b^2)) :
  b = 4 := by
  sorry

end hyperbola_b_value_l3731_373148


namespace sum_smallest_largest_consecutive_integers_l3731_373192

/-- Given an even number of consecutive integers with arithmetic mean z,
    the sum of the smallest and largest integers is equal to 2z. -/
theorem sum_smallest_largest_consecutive_integers (m : ℕ) (z : ℚ) (h_even : Even m) (h_pos : 0 < m) :
  let b : ℚ := (2 * z * m - m^2 + m) / (2 * m)
  (b + (b + m - 1)) = 2 * z :=
by sorry

end sum_smallest_largest_consecutive_integers_l3731_373192


namespace correct_propositions_count_l3731_373100

-- Define the types of events
inductive EventType
  | Certain
  | Impossible
  | Random

-- Define the propositions
def proposition1 : EventType := EventType.Certain
def proposition2 : EventType := EventType.Impossible
def proposition3 : EventType := EventType.Certain
def proposition4 : EventType := EventType.Random

-- Define a function to check if a proposition is correct
def is_correct (prop : EventType) : Bool :=
  match prop with
  | EventType.Certain => true
  | EventType.Impossible => true
  | EventType.Random => true

-- Theorem: The number of correct propositions is 3
theorem correct_propositions_count :
  (is_correct proposition1).toNat +
  (is_correct proposition2).toNat +
  (is_correct proposition3).toNat +
  (is_correct proposition4).toNat = 3 := by
  sorry


end correct_propositions_count_l3731_373100


namespace min_value_theorem_l3731_373161

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → (x + y + 1) / (x * y) ≥ (a + b + 1) / (a * b)) →
  (a + b + 1) / (a * b) = 4 * Real.sqrt 3 + 7 :=
sorry

end min_value_theorem_l3731_373161


namespace adjacent_smaller_perfect_square_l3731_373149

theorem adjacent_smaller_perfect_square (m : ℕ) (h : ∃ k : ℕ, m = k^2) :
  ∃ n : ℕ, n^2 = m - 2*Int.sqrt m + 1 ∧
    n^2 < m ∧
    ∀ k : ℕ, k^2 < m → k^2 ≤ n^2 :=
sorry

end adjacent_smaller_perfect_square_l3731_373149


namespace inequality_solution_and_absolute_value_bound_l3731_373114

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-1) 2

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |2 * x - a| ≤ 3

-- Theorem statement
theorem inequality_solution_and_absolute_value_bound (a : ℝ) :
  (∀ x, x ∈ solution_set ↔ inequality a x) →
  (a = 1 ∧ ∀ x m, |x - m| < a → |x| < |m| + 1) :=
sorry

end inequality_solution_and_absolute_value_bound_l3731_373114


namespace single_color_bound_l3731_373190

/-- A polygon on a checkered plane --/
structure CheckeredPolygon where
  /-- The area of the polygon --/
  area : ℕ
  /-- The perimeter of the polygon --/
  perimeter : ℕ

/-- The number of squares of a single color in the polygon --/
def singleColorCount (p : CheckeredPolygon) : ℕ := sorry

/-- Theorem: The number of squares of a single color is bounded --/
theorem single_color_bound (p : CheckeredPolygon) :
  singleColorCount p ≥ p.area / 2 - p.perimeter / 8 ∧
  singleColorCount p ≤ p.area / 2 + p.perimeter / 8 := by
  sorry

end single_color_bound_l3731_373190


namespace rhombus_prism_lateral_area_l3731_373138

/-- The lateral surface area of a right prism with a rhombus base and given dimensions. -/
theorem rhombus_prism_lateral_area (d1 d2 h : ℝ) (hd1 : d1 = 9) (hd2 : d2 = 15) (hh : h = 5) :
  4 * (((d1 ^ 2 / 4 + d2 ^ 2 / 4) : ℝ).sqrt) * h = 160 :=
by sorry

#check rhombus_prism_lateral_area

end rhombus_prism_lateral_area_l3731_373138


namespace sum_of_abc_is_zero_l3731_373188

theorem sum_of_abc_is_zero 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hnot_all_equal : ¬(a = b ∧ b = c))
  (heq : a^2 / (2*a^2 + b*c) + b^2 / (2*b^2 + c*a) + c^2 / (2*c^2 + a*b) = 1) :
  a + b + c = 0 := by
sorry

end sum_of_abc_is_zero_l3731_373188


namespace five_workers_required_l3731_373116

/-- Represents the project parameters and progress -/
structure ProjectStatus :=
  (total_days : ℕ)
  (elapsed_days : ℕ)
  (initial_workers : ℕ)
  (completed_fraction : ℚ)

/-- Calculates the minimum number of workers required to complete the project on schedule -/
def min_workers_required (status : ProjectStatus) : ℕ :=
  sorry

/-- Theorem stating that for the given project status, 5 workers are required -/
theorem five_workers_required (status : ProjectStatus) 
  (h1 : status.total_days = 20)
  (h2 : status.elapsed_days = 5)
  (h3 : status.initial_workers = 10)
  (h4 : status.completed_fraction = 1/4) :
  min_workers_required status = 5 := by
  sorry

end five_workers_required_l3731_373116


namespace complex_magnitude_proof_l3731_373131

theorem complex_magnitude_proof : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / i
  Complex.abs z = Real.sqrt 2 := by sorry

end complex_magnitude_proof_l3731_373131


namespace binomial_expansion_problem_l3731_373104

theorem binomial_expansion_problem (n : ℕ) : 
  ((-2 : ℤ) ^ n = 64) →
  (n = 6 ∧ Nat.choose n 2 * 9 = 135) := by
sorry

end binomial_expansion_problem_l3731_373104


namespace percentage_problem_l3731_373158

/-- Given that (P/100 * 1265) / 7 = 271.07142857142856, prove that P = 150 -/
theorem percentage_problem (P : ℝ) : (P / 100 * 1265) / 7 = 271.07142857142856 → P = 150 := by
  sorry

end percentage_problem_l3731_373158


namespace solution_set_f_x_minus_one_gt_two_min_value_x_plus_2y_plus_2z_l3731_373160

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Theorem for part I
theorem solution_set_f_x_minus_one_gt_two :
  {x : ℝ | f (x - 1) > 2} = {x : ℝ | x < -1 ∨ x > 3} :=
sorry

-- Theorem for part II
theorem min_value_x_plus_2y_plus_2z (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) :
  ∃ (m : ℝ), m = -9 ∧ ∀ (x' y' z' : ℝ), f x' ^ 2 + y' ^ 2 + z' ^ 2 = 9 → x' + 2 * y' + 2 * z' ≥ m :=
sorry

end solution_set_f_x_minus_one_gt_two_min_value_x_plus_2y_plus_2z_l3731_373160


namespace contractor_payment_example_l3731_373126

/-- Calculates the total amount a contractor receives given the contract terms and absent days. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  (total_days - absent_days : ℚ) * payment_per_day - (absent_days : ℚ) * fine_per_day

/-- Theorem stating that under the given conditions, the contractor receives Rs. 425. -/
theorem contractor_payment_example : 
  contractor_payment 30 25 7.5 10 = 425 := by
  sorry

end contractor_payment_example_l3731_373126


namespace amusement_park_tickets_l3731_373139

theorem amusement_park_tickets 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_paid : ℕ) 
  (child_tickets : ℕ) : 
  adult_price = 8 → 
  child_price = 5 → 
  total_paid = 201 → 
  child_tickets = 21 → 
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + child_price * child_tickets = total_paid ∧ 
    adult_tickets + child_tickets = 33 :=
by
  sorry

#check amusement_park_tickets

end amusement_park_tickets_l3731_373139


namespace total_monthly_pay_is_12708_l3731_373140

-- Define the structure for an employee
structure Employee where
  name : String
  hours_per_week : ℕ
  hourly_rate : ℕ

-- Define the list of employees
def employees : List Employee := [
  { name := "Fiona", hours_per_week := 40, hourly_rate := 20 },
  { name := "John", hours_per_week := 30, hourly_rate := 22 },
  { name := "Jeremy", hours_per_week := 25, hourly_rate := 18 },
  { name := "Katie", hours_per_week := 35, hourly_rate := 21 },
  { name := "Matt", hours_per_week := 28, hourly_rate := 19 }
]

-- Define the number of weeks in a month
def weeks_in_month : ℕ := 4

-- Calculate the monthly pay for all employees
def total_monthly_pay : ℕ :=
  employees.foldl (fun acc e => acc + e.hours_per_week * e.hourly_rate * weeks_in_month) 0

-- Theorem stating that the total monthly pay is $12,708
theorem total_monthly_pay_is_12708 : total_monthly_pay = 12708 := by
  sorry

end total_monthly_pay_is_12708_l3731_373140


namespace kendra_evening_minivans_l3731_373123

/-- The number of minivans Kendra saw in the afternoon -/
def afternoon_minivans : ℕ := 4

/-- The total number of minivans Kendra saw -/
def total_minivans : ℕ := 5

/-- The number of minivans Kendra saw in the evening -/
def evening_minivans : ℕ := total_minivans - afternoon_minivans

theorem kendra_evening_minivans : evening_minivans = 1 := by
  sorry

end kendra_evening_minivans_l3731_373123


namespace no_solutions_in_interval_l3731_373128

theorem no_solutions_in_interval (x : ℝ) :
  -π ≤ x ∧ x ≤ 3*π →
  ¬(1 / Real.sin x + 1 / Real.cos x = 4) :=
by sorry

end no_solutions_in_interval_l3731_373128


namespace car_speed_comparison_l3731_373194

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 3 / (1 / u + 2 / v)
  let y := (u + 2 * v) / 3
  x ≤ y := by
  sorry

end car_speed_comparison_l3731_373194


namespace square_area_equal_perimeter_triangle_l3731_373124

theorem square_area_equal_perimeter_triangle (a b c s : ℝ) : 
  a = 7.5 ∧ b = 9.3 ∧ c = 12.2 → -- triangle side lengths
  s * 4 = a + b + c →           -- equal perimeters
  s * s = 52.5625 :=            -- square area
by sorry

end square_area_equal_perimeter_triangle_l3731_373124


namespace shelbys_driving_time_l3731_373178

/-- Shelby's driving problem -/
theorem shelbys_driving_time (speed_no_rain speed_rain : ℝ) (total_time total_distance : ℝ) 
  (h1 : speed_no_rain = 40)
  (h2 : speed_rain = 25)
  (h3 : total_time = 3)
  (h4 : total_distance = 85) :
  let rain_time := (total_distance - speed_no_rain * total_time) / (speed_rain - speed_no_rain)
  rain_time * 60 = 140 := by sorry

end shelbys_driving_time_l3731_373178


namespace max_quotient_value_l3731_373136

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 500) (hb : 400 ≤ b ∧ b ≤ 1000) :
  (∀ x y, 100 ≤ x ∧ x ≤ 500 → 400 ≤ y ∧ y ≤ 1000 → b / a ≥ y / x) ∧ b / a ≤ 10 :=
sorry

end max_quotient_value_l3731_373136


namespace probability_is_three_eighths_l3731_373189

/-- Represents a circular field with 8 roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : ℕ := 8)

/-- Represents a geologist on the field -/
structure Geologist :=
  (speed : ℝ)
  (time : ℝ)
  (road : ℕ)

/-- Calculates the distance between two geologists -/
def distance_between (g1 g2 : Geologist) (field : CircularField) : ℝ :=
  sorry

/-- Determines if two geologists are more than 8 km apart -/
def more_than_8km_apart (g1 g2 : Geologist) (field : CircularField) : Prop :=
  distance_between g1 g2 field > 8

/-- Calculates the probability of two geologists being more than 8 km apart -/
def probability_more_than_8km_apart (field : CircularField) : ℝ :=
  sorry

theorem probability_is_three_eighths (field : CircularField) 
  (g1 g2 : Geologist) 
  (h1 : field.num_roads = 8) 
  (h2 : g1.speed = 5) 
  (h3 : g2.speed = 5) 
  (h4 : g1.time = 1) 
  (h5 : g2.time = 1) :
  probability_more_than_8km_apart field = 3/8 :=
sorry

end probability_is_three_eighths_l3731_373189


namespace root_implies_m_value_l3731_373110

theorem root_implies_m_value (m : ℝ) : (3^2 - 4*3 + m = 0) → m = 3 := by
  sorry

end root_implies_m_value_l3731_373110


namespace complex_number_computations_l3731_373198

theorem complex_number_computations :
  let z₁ : ℂ := 1 + 2*I
  let z₂ : ℂ := (1 + I) / (1 - I)
  let z₃ : ℂ := (Real.sqrt 2 + Real.sqrt 3 * I) / (Real.sqrt 3 - Real.sqrt 2 * I)
  (z₁^2 = -3 + 4*I) ∧
  (z₂^6 + z₃ = -1 + Real.sqrt 6 / 5 + ((Real.sqrt 3 + Real.sqrt 2) / 5) * I) := by
sorry

end complex_number_computations_l3731_373198


namespace absolute_value_inequality_l3731_373166

theorem absolute_value_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end absolute_value_inequality_l3731_373166


namespace statement_falsity_l3731_373185

theorem statement_falsity (x : ℝ) : x = -4 ∨ x = -2 → x ∈ Set.Iio 2 ∧ ¬(x^2 < 4) := by
  sorry

end statement_falsity_l3731_373185


namespace inequality_proof_l3731_373112

theorem inequality_proof (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4*a*c := by
  sorry

end inequality_proof_l3731_373112


namespace both_in_picture_probability_l3731_373107

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (sarah sam : Runner) (pictureWidth : ℝ) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem both_in_picture_probability 
  (sarah : Runner) 
  (sam : Runner) 
  (sarah_laptime : sarah.lapTime = 120)
  (sam_laptime : sam.lapTime = 75)
  (sarah_direction : sarah.direction = true)
  (sam_direction : sam.direction = false)
  (picture_width : ℝ)
  (picture_covers_third : picture_width = sarah.lapTime / 3) :
  probabilityBothInPicture sarah sam picture_width = 1/4 := by
  sorry

end both_in_picture_probability_l3731_373107


namespace fraction_product_l3731_373121

theorem fraction_product : (2 : ℚ) / 9 * 5 / 14 = 5 / 63 := by
  sorry

end fraction_product_l3731_373121


namespace opposite_of_negative_2023_l3731_373159

theorem opposite_of_negative_2023 : 
  -((-2023) : ℤ) = (2023 : ℤ) := by sorry

end opposite_of_negative_2023_l3731_373159


namespace brownies_remaining_l3731_373197

/-- Calculates the number of brownies left after consumption -/
def brownies_left (total : ℕ) (tina_daily : ℕ) (tina_days : ℕ) (husband_daily : ℕ) (husband_days : ℕ) (shared : ℕ) : ℕ :=
  total - (tina_daily * tina_days + husband_daily * husband_days + shared)

/-- Proves that given the specific consumption pattern, 5 brownies are left -/
theorem brownies_remaining :
  brownies_left 24 2 5 1 5 4 = 5 := by
  sorry

#eval brownies_left 24 2 5 1 5 4

end brownies_remaining_l3731_373197


namespace vector_to_line_parallel_l3731_373151

/-- A vector pointing from the origin to a line parallel to another vector -/
theorem vector_to_line_parallel (t : ℝ) : ∃ (k : ℝ), ∃ (a b : ℝ),
  (a = 3 * t + 1 ∧ b = t + 1) ∧  -- Point on the line
  (∃ (c : ℝ), a = 3 * c ∧ b = c) ∧  -- Parallel to (3, 1)
  a = 3 * k - 2 ∧ b = k :=  -- The form of the vector
by sorry

end vector_to_line_parallel_l3731_373151


namespace max_distance_from_point_to_unit_circle_l3731_373186

theorem max_distance_from_point_to_unit_circle :
  ∃ (M : ℝ), M = 6 ∧ ∀ z : ℂ, Complex.abs z = 1 →
    Complex.abs (z - (3 - 4*I)) ≤ M ∧
    ∃ w : ℂ, Complex.abs w = 1 ∧ Complex.abs (w - (3 - 4*I)) = M :=
by sorry

end max_distance_from_point_to_unit_circle_l3731_373186


namespace find_p_l3731_373125

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 20

-- State the theorem
theorem find_p : ∃ p : ℝ, f (f (f p)) = 6 ∧ p = 18.25 := by
  sorry

end find_p_l3731_373125


namespace bunny_count_l3731_373167

/-- The number of bunnies coming out of their burrows -/
def num_bunnies : ℕ := 
  let times_per_minute : ℕ := 3
  let hours : ℕ := 10
  let minutes_per_hour : ℕ := 60
  let total_times : ℕ := 36000
  total_times / (times_per_minute * hours * minutes_per_hour)

theorem bunny_count : num_bunnies = 20 := by
  sorry

end bunny_count_l3731_373167


namespace sum_reciprocals_l3731_373134

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / (1 + ω)) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end sum_reciprocals_l3731_373134


namespace semicircle_radius_l3731_373106

/-- The radius of a semicircle with perimeter 180 cm is equal to 180 / (π + 2) cm. -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 180) :
  ∃ r : ℝ, r = perimeter / (Real.pi + 2) ∧ r * (Real.pi + 2) = perimeter := by
  sorry

end semicircle_radius_l3731_373106


namespace at_least_100_triangles_l3731_373150

/-- Represents a set of lines in a plane -/
structure LineSet where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersection : Bool

/-- Counts the number of triangular regions formed by a set of lines -/
def count_triangles (lines : LineSet) : ℕ := sorry

/-- Theorem: 300 lines with given conditions form at least 100 triangles -/
theorem at_least_100_triangles (lines : LineSet) 
  (h1 : lines.num_lines = 300)
  (h2 : lines.no_parallel = true)
  (h3 : lines.no_triple_intersection = true) :
  count_triangles lines ≥ 100 := by sorry

end at_least_100_triangles_l3731_373150


namespace no_four_integers_l3731_373147

theorem no_four_integers (n : ℕ) (hn : n ≥ 1) :
  ¬ ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    n^2 ≤ a ∧ a < (n+1)^2 ∧
    n^2 ≤ b ∧ b < (n+1)^2 ∧
    n^2 ≤ c ∧ c < (n+1)^2 ∧
    n^2 ≤ d ∧ d < (n+1)^2 ∧
    a * d = b * c :=
by sorry

end no_four_integers_l3731_373147


namespace junior_fraction_l3731_373164

/-- Represents the number of students in each category -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : StudentCounts) : Prop :=
  s.freshmen + s.sophomores + s.juniors + s.seniors = 120 ∧
  s.freshmen > 0 ∧ s.sophomores > 0 ∧ s.juniors > 0 ∧ s.seniors > 0 ∧
  s.freshmen = 2 * s.sophomores ∧
  s.juniors = 4 * s.seniors ∧
  (s.freshmen : ℚ) / 2 + (s.sophomores : ℚ) / 3 = (s.juniors : ℚ) * 2 / 3 - (s.seniors : ℚ) / 4

/-- The theorem to be proved -/
theorem junior_fraction (s : StudentCounts) (h : satisfiesConditions s) :
    (s.juniors : ℚ) / (s.freshmen + s.sophomores + s.juniors + s.seniors) = 32 / 167 := by
  sorry

end junior_fraction_l3731_373164


namespace odd_integers_square_l3731_373153

theorem odd_integers_square (a b : ℕ) (ha : Odd a) (hb : Odd b) (hab : ∃ k : ℕ, a^b * b^a = k^2) :
  ∃ m : ℕ, a * b = m^2 := by
sorry

end odd_integers_square_l3731_373153


namespace largest_prime_factor_l3731_373179

def numbers : List Nat := [55, 63, 95, 133, 143]

theorem largest_prime_factor :
  ∃ (n : Nat), n ∈ numbers ∧ 19 ∣ n ∧
  ∀ (m : Nat), m ∈ numbers → ∀ (p : Nat), Prime p → p ∣ m → p ≤ 19 :=
by sorry

end largest_prime_factor_l3731_373179


namespace number_problem_l3731_373174

theorem number_problem (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 := by
  sorry

end number_problem_l3731_373174
