import Mathlib

namespace rectangle_area_ratio_side_length_l810_81059

theorem rectangle_area_ratio_side_length (area_ratio : ℚ) (p q r : ℕ) : 
  area_ratio = 500 / 125 →
  (p * Real.sqrt q : ℝ) / r = Real.sqrt (area_ratio) →
  p + q + r = 4 := by
sorry

end rectangle_area_ratio_side_length_l810_81059


namespace z_in_third_quadrant_l810_81010

def i : ℂ := Complex.I

def z : ℂ := i + 2 * i^2 + 3 * i^3

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_third_quadrant_l810_81010


namespace sock_order_ratio_l810_81055

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
structure SockPrice where
  blue : ℝ

/-- Calculates the total cost of a sock order given the prices -/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  (order.black * 3 * price.blue) + (order.blue * price.blue)

/-- The theorem to be proved -/
theorem sock_order_ratio (original : SockOrder) (price : SockPrice) : 
  original.black = 5 →
  totalCost { black := original.blue, blue := original.black } price = 1.6 * totalCost original price →
  (original.black : ℝ) / original.blue = 5 / 14 := by
  sorry

#check sock_order_ratio

end sock_order_ratio_l810_81055


namespace major_axis_length_major_axis_length_is_four_l810_81011

/-- An ellipse with foci at (5, 1 + √8) and (5, 1 - √8), tangent to y = 1 and x = 1 -/
structure SpecialEllipse where
  /-- First focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- Second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- The ellipse is tangent to the line y = 1 -/
  tangent_y : True
  /-- The ellipse is tangent to the line x = 1 -/
  tangent_x : True
  /-- The first focus is at (5, 1 + √8) -/
  focus1_coord : focus1 = (5, 1 + Real.sqrt 8)
  /-- The second focus is at (5, 1 - √8) -/
  focus2_coord : focus2 = (5, 1 - Real.sqrt 8)

/-- The length of the major axis of the special ellipse is 4 -/
theorem major_axis_length (e : SpecialEllipse) : ℝ := 4

/-- The major axis length of the special ellipse is indeed 4 -/
theorem major_axis_length_is_four (e : SpecialEllipse) : 
  major_axis_length e = 4 := by sorry

end major_axis_length_major_axis_length_is_four_l810_81011


namespace water_usage_problem_l810_81043

/-- Calculates the water charge based on usage --/
def water_charge (usage : ℕ) : ℚ :=
  if usage ≤ 24 then 1.8 * usage
  else 1.8 * 24 + 4 * (usage - 24)

/-- Represents the water usage problem --/
theorem water_usage_problem :
  ∃ (zhang_usage wang_usage : ℕ),
    zhang_usage > 24 ∧
    wang_usage ≤ 24 ∧
    water_charge zhang_usage - water_charge wang_usage = 19.2 ∧
    zhang_usage = 27 ∧
    wang_usage = 20 :=
by
  sorry

#eval water_charge 27  -- Should output 55.2
#eval water_charge 20  -- Should output 36

end water_usage_problem_l810_81043


namespace circle_radius_equals_sphere_surface_area_l810_81012

theorem circle_radius_equals_sphere_surface_area (r : ℝ) : 
  r > 0 → π * r^2 = 4 * π * (2 : ℝ)^2 → r = 4 := by
  sorry

end circle_radius_equals_sphere_surface_area_l810_81012


namespace smallest_780_divisible_by_1125_l810_81092

def is_composed_of_780 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 8 ∨ d = 0

theorem smallest_780_divisible_by_1125 :
  ∀ n : ℕ, n > 0 → is_composed_of_780 n → n % 1125 = 0 → n ≥ 77778000 :=
sorry

end smallest_780_divisible_by_1125_l810_81092


namespace chess_tournament_games_l810_81044

/-- Represents a chess tournament --/
structure ChessTournament where
  participants : ℕ
  total_games : ℕ
  games_per_player : ℕ
  h1 : total_games = participants * (participants - 1) / 2
  h2 : games_per_player = participants - 1

/-- Theorem: In a chess tournament with 20 participants and 190 total games, 
    each participant plays 19 games --/
theorem chess_tournament_games (t : ChessTournament) 
  (h_participants : t.participants = 20) 
  (h_total_games : t.total_games = 190) : 
  t.games_per_player = 19 := by
  sorry


end chess_tournament_games_l810_81044


namespace cosine_sum_product_simplification_l810_81072

theorem cosine_sum_product_simplification (α β : ℝ) :
  Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β = Real.cos α := by
  sorry

end cosine_sum_product_simplification_l810_81072


namespace min_value_x_plus_4y_l810_81033

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 1/(2*y) = 1) : 
  x + 4*y ≥ 3 + 2*Real.sqrt 2 := by
sorry

end min_value_x_plus_4y_l810_81033


namespace square_rectangle_triangle_relation_l810_81023

/-- Square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- Rectangle with width and height -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- Right triangle with base and height -/
structure RightTriangle :=
  (base : ℝ)
  (height : ℝ)

/-- The main theorem -/
theorem square_rectangle_triangle_relation 
  (ABCD : Square)
  (JKHG : Rectangle)
  (EBC : RightTriangle)
  (h1 : JKHG.width = ABCD.side)
  (h2 : EBC.base = ABCD.side)
  (h3 : JKHG.height = EBC.height)
  (h4 : JKHG.width * JKHG.height = 2 * (EBC.base * EBC.height / 2)) :
  EBC.height = 1 := by
  sorry


end square_rectangle_triangle_relation_l810_81023


namespace min_handshakes_30_people_l810_81078

/-- The minimum number of handshakes in a gathering -/
def min_handshakes (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a gathering of 30 people, where each person shakes hands
    with at least three other people, the minimum possible number of handshakes is 45 -/
theorem min_handshakes_30_people :
  let n : ℕ := 30
  let k : ℕ := 3
  min_handshakes n k = 45 := by
  sorry


end min_handshakes_30_people_l810_81078


namespace lauren_change_calculation_l810_81060

/-- Calculates the change Lauren receives after grocery shopping --/
theorem lauren_change_calculation : 
  let hamburger_meat_price : ℝ := 3.50
  let hamburger_meat_weight : ℝ := 2
  let buns_price : ℝ := 1.50
  let lettuce_price : ℝ := 1.00
  let tomato_price_per_pound : ℝ := 2.00
  let tomato_weight : ℝ := 1.5
  let pickles_price : ℝ := 2.50
  let coupon_value : ℝ := 1.00
  let paid_amount : ℝ := 20.00

  let total_cost : ℝ := 
    hamburger_meat_price * hamburger_meat_weight +
    buns_price + 
    lettuce_price + 
    tomato_price_per_pound * tomato_weight + 
    pickles_price - 
    coupon_value

  let change : ℝ := paid_amount - total_cost

  change = 6.00 := by sorry

end lauren_change_calculation_l810_81060


namespace roots_quadratic_equation_l810_81067

theorem roots_quadratic_equation (α β : ℝ) : 
  (∀ x : ℝ, x^2 + 4*x + 2 = 0 ↔ x = α ∨ x = β) → 
  α^3 + 14*β + 5 = -43 := by
  sorry

end roots_quadratic_equation_l810_81067


namespace smaller_number_proof_l810_81040

theorem smaller_number_proof (x y : ℕ+) : 
  (x * y : ℕ) = 323 → 
  (x : ℕ) = (y : ℕ) + 2 → 
  y = 17 := by
sorry

end smaller_number_proof_l810_81040


namespace age_ratio_in_four_years_l810_81074

/-- Represents the ages of Paul and Kim -/
structure Ages where
  paul : ℕ
  kim : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.paul - 8 = 2 * (a.kim - 8)) ∧ 
  (a.paul - 14 = 3 * (a.kim - 14))

/-- The theorem to prove -/
theorem age_ratio_in_four_years (a : Ages) :
  age_conditions a →
  ∃ (x : ℕ), x = 4 ∧ 
    (a.paul + x) * 2 = (a.kim + x) * 3 :=
by sorry

end age_ratio_in_four_years_l810_81074


namespace lcm_hcf_problem_l810_81061

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 → 
  Nat.gcd A B = 30 → 
  A = 231 → 
  B = 300 := by
sorry

end lcm_hcf_problem_l810_81061


namespace quadratic_equation_condition_l810_81082

theorem quadratic_equation_condition (a : ℝ) :
  (∀ x, (a - 2) * x^2 + a * x - 3 = 0 → (a - 2) ≠ 0) ↔ a ≠ 2 :=
by sorry

end quadratic_equation_condition_l810_81082


namespace direction_vector_form_l810_81097

/-- Given a line passing through two points, prove that its direction vector has a specific form -/
theorem direction_vector_form (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-3, 2) → p2 = (2, -3) → 
  (∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) → 
  b = 1 := by
sorry

end direction_vector_form_l810_81097


namespace square_sum_value_l810_81086

theorem square_sum_value (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 15) : x^2 + y^2 = 70 := by
  sorry

end square_sum_value_l810_81086


namespace ratio_of_sums_and_differences_l810_81031

theorem ratio_of_sums_and_differences (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
  (x + 1 / x) / (x - 1 / x) = Real.sqrt 7 / Real.sqrt 6 := by
  sorry

end ratio_of_sums_and_differences_l810_81031


namespace wendy_bought_four_chairs_l810_81021

def furniture_problem (chairs : ℕ) : Prop :=
  let tables : ℕ := 4
  let time_per_piece : ℕ := 6
  let total_time : ℕ := 48
  (chairs + tables) * time_per_piece = total_time

theorem wendy_bought_four_chairs :
  ∃ (chairs : ℕ), furniture_problem chairs ∧ chairs = 4 :=
sorry

end wendy_bought_four_chairs_l810_81021


namespace weakly_increasing_h_implies_b_eq_one_l810_81025

/-- A function is weakly increasing in an interval if it's increasing and its ratio to x is decreasing in that interval --/
def WeaklyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∧
  (∀ x y, a < x ∧ x < y ∧ y ≤ b → f x / x ≥ f y / y)

/-- The function h(x) = x^2 - (b-1)x + b --/
def h (b : ℝ) (x : ℝ) : ℝ := x^2 - (b-1)*x + b

theorem weakly_increasing_h_implies_b_eq_one :
  ∀ b : ℝ, WeaklyIncreasing (h b) 0 1 → b = 1 := by
  sorry

end weakly_increasing_h_implies_b_eq_one_l810_81025


namespace sum_of_cubes_l810_81002

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 8)
  (sum_products_eq : a * b + a * c + b * c = 10)
  (product_eq : a * b * c = -15) :
  a^3 + b^3 + c^3 = 227 := by
sorry

end sum_of_cubes_l810_81002


namespace dozen_chocolates_cost_l810_81083

/-- The cost of a magazine in dollars -/
def magazine_cost : ℝ := 1

/-- The cost of a chocolate bar in dollars -/
def chocolate_cost : ℝ := 2

/-- The number of magazines that cost the same as 4 chocolate bars -/
def magazines_equal_to_4_chocolates : ℕ := 8

theorem dozen_chocolates_cost (h : 4 * chocolate_cost = magazines_equal_to_4_chocolates * magazine_cost) :
  12 * chocolate_cost = 24 := by
  sorry

end dozen_chocolates_cost_l810_81083


namespace square_sum_given_product_and_sum_l810_81054

/-- Given two real numbers p and q satisfying pq = 16 and p + q = 8, 
    prove that p^2 + q^2 = 32. -/
theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 16) (h2 : p + q = 8) : p^2 + q^2 = 32 := by
  sorry

end square_sum_given_product_and_sum_l810_81054


namespace line_point_k_l810_81080

/-- A line is defined by three points it passes through -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Check if a point lies on a given line -/
def lies_on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x1, y1) := l.p1
  let (x2, y2) := l.p2
  let (x3, y3) := l.p3
  let (x, y) := p
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1) ∧
  (y - y2) * (x3 - x2) = (y3 - y2) * (x - x2)

/-- The main theorem -/
theorem line_point_k (l : Line) (k : ℝ) :
  l.p1 = (-1, 1) →
  l.p2 = (2, 5) →
  l.p3 = (5, 9) →
  lies_on_line l (50, k) →
  k = 69 := by
  sorry

end line_point_k_l810_81080


namespace batsman_new_average_l810_81076

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ
  runsIn17thInning : ℝ
  averageIncrease : ℝ

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem stating the batsman's new average after the 17th inning -/
theorem batsman_new_average (b : Batsman) 
  (h1 : b.runsIn17thInning = 74)
  (h2 : b.averageIncrease = 3) : 
  newAverage b = 26 := by
  sorry

#check batsman_new_average

end batsman_new_average_l810_81076


namespace grade_study_sample_size_l810_81046

/-- Represents a statistical study of student grades -/
structure GradeStudy where
  total_students : ℕ
  selected_cards : ℕ

/-- Defines the sample size of a grade study -/
def sample_size (study : GradeStudy) : ℕ := study.selected_cards

/-- Theorem: The sample size of a study with 2000 total students and 200 selected cards is 200 -/
theorem grade_study_sample_size :
  ∀ (study : GradeStudy),
    study.total_students = 2000 →
    study.selected_cards = 200 →
    sample_size study = 200 := by
  sorry

end grade_study_sample_size_l810_81046


namespace min_value_x3_l810_81068

theorem min_value_x3 (x₁ x₂ x₃ : ℝ) 
  (eq1 : x₁ + (1/2) * x₂ + (1/3) * x₃ = 1)
  (eq2 : x₁^2 + (1/2) * x₂^2 + (1/3) * x₃^2 = 3) :
  x₃ ≥ -21/11 ∧ ∃ (x₁' x₂' x₃' : ℝ), 
    x₁' + (1/2) * x₂' + (1/3) * x₃' = 1 ∧
    x₁'^2 + (1/2) * x₂'^2 + (1/3) * x₃'^2 = 3 ∧
    x₃' = -21/11 := by
  sorry

end min_value_x3_l810_81068


namespace dihedral_angle_eq_inclination_l810_81008

/-- A pyramid with an isosceles triangular base and inclined lateral edges -/
structure IsoscelesPyramid where
  -- Angle between equal sides of the base triangle
  α : Real
  -- Angle of inclination of lateral edges to the base plane
  φ : Real
  -- Assumption that α and φ are valid angles
  h_α_range : 0 < α ∧ α < π
  h_φ_range : 0 < φ ∧ φ < π/2

/-- The dihedral angle at the edge connecting the apex to the vertex of angle α -/
def dihedral_angle (p : IsoscelesPyramid) : Real :=
  -- Definition of dihedral angle (to be proved equal to φ)
  sorry

/-- Theorem: The dihedral angle is equal to the inclination angle of lateral edges -/
theorem dihedral_angle_eq_inclination (p : IsoscelesPyramid) :
  dihedral_angle p = p.φ :=
sorry

end dihedral_angle_eq_inclination_l810_81008


namespace clarissa_manuscript_cost_l810_81034

/-- Calculate the total cost for printing, binding, and processing multiple copies of a manuscript with specified requirements. -/
def manuscript_cost (total_pages : ℕ) (color_pages : ℕ) (bw_cost : ℚ) (color_cost : ℚ) 
                    (binding_cost : ℚ) (index_cost : ℚ) (copies : ℕ) (rush_copies : ℕ) 
                    (rush_cost : ℚ) : ℚ :=
  let bw_pages := total_pages - color_pages
  let print_cost := (bw_pages : ℚ) * bw_cost + (color_pages : ℚ) * color_cost
  let additional_cost := binding_cost + index_cost
  let total_per_copy := print_cost + additional_cost
  let total_before_rush := (copies : ℚ) * total_per_copy
  let rush_fee := (rush_copies : ℚ) * rush_cost
  total_before_rush + rush_fee

/-- The total cost for Clarissa's manuscript printing job is $310.00. -/
theorem clarissa_manuscript_cost :
  manuscript_cost 400 50 (5/100) (10/100) 5 2 10 5 3 = 310 := by
  sorry

end clarissa_manuscript_cost_l810_81034


namespace second_planner_cheaper_at_34_l810_81050

/-- Represents the pricing model of an event planner -/
structure PricingModel where
  flatFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (model : PricingModel) (people : ℕ) : ℕ :=
  model.flatFee + model.perPersonFee * people

/-- The pricing model of the first planner -/
def planner1 : PricingModel := { flatFee := 150, perPersonFee := 18 }

/-- The pricing model of the second planner -/
def planner2 : PricingModel := { flatFee := 250, perPersonFee := 15 }

/-- Theorem stating that 34 is the least number of people for which the second planner is cheaper -/
theorem second_planner_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost planner1 n ≤ totalCost planner2 n) ∧
  (totalCost planner2 34 < totalCost planner1 34) :=
by sorry

end second_planner_cheaper_at_34_l810_81050


namespace half_MN_coord_l810_81032

def OM : Fin 2 → ℝ := ![(-2), 3]
def ON : Fin 2 → ℝ := ![(-1), (-5)]

theorem half_MN_coord : 
  (1/2 : ℝ) • (ON - OM) = ![(1/2), (-4)] := by sorry

end half_MN_coord_l810_81032


namespace sportswear_processing_equation_l810_81016

/-- Represents the clothing processing factory problem --/
theorem sportswear_processing_equation 
  (total_sportswear : ℕ) 
  (processed_before_tech : ℕ) 
  (efficiency_increase : ℚ) 
  (total_time : ℚ) 
  (x : ℚ) 
  (h1 : total_sportswear = 400)
  (h2 : processed_before_tech = 160)
  (h3 : efficiency_increase = 1/5)
  (h4 : total_time = 18)
  (h5 : x > 0) :
  (processed_before_tech / x) + ((total_sportswear - processed_before_tech) / ((1 + efficiency_increase) * x)) = total_time :=
sorry

end sportswear_processing_equation_l810_81016


namespace quadratic_inequality_solution_set_l810_81007

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 + (2 - a) * x - 2 < 0}
  (a = 0 → S = {x : ℝ | x < 1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | x < 1 ∨ x > -2/a}) ∧
  (a = -2 → S = {x : ℝ | x ≠ 1}) ∧
  (a < -2 → S = {x : ℝ | x < -2/a ∨ x > 1}) ∧
  (a > 0 → S = {x : ℝ | -2/a < x ∧ x < 1}) := by
sorry

end quadratic_inequality_solution_set_l810_81007


namespace binomial_10_4_l810_81058

theorem binomial_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end binomial_10_4_l810_81058


namespace fifth_month_sale_l810_81001

def sales_1 : ℕ := 5921
def sales_2 : ℕ := 5468
def sales_3 : ℕ := 5568
def sales_4 : ℕ := 6088
def sales_6 : ℕ := 5922
def average_sale : ℕ := 5900
def num_months : ℕ := 6

theorem fifth_month_sale :
  ∃ (sales_5 : ℕ),
    sales_5 = average_sale * num_months - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6433 := by
  sorry

end fifth_month_sale_l810_81001


namespace arccos_neg_half_eq_two_pi_thirds_l810_81036

theorem arccos_neg_half_eq_two_pi_thirds : 
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end arccos_neg_half_eq_two_pi_thirds_l810_81036


namespace coinciding_white_pairs_l810_81015

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  redRed : Nat
  blueBlue : Nat
  redWhite : Nat
  whiteWhite : Nat

/-- The main theorem that proves the number of coinciding white pairs -/
theorem coinciding_white_pairs
  (initial_count : TriangleCount)
  (coinciding : CoincidingPairs)
  (h1 : initial_count.red = 2)
  (h2 : initial_count.blue = 4)
  (h3 : initial_count.white = 6)
  (h4 : coinciding.redRed = 1)
  (h5 : coinciding.blueBlue = 2)
  (h6 : coinciding.redWhite = 2)
  : coinciding.whiteWhite = 4 := by
  sorry

end coinciding_white_pairs_l810_81015


namespace multiplication_problem_l810_81018

-- Define a custom type for single digits
def Digit := { n : Nat // n < 10 }

-- Define a function to convert a two-digit number to a natural number
def twoDigitToNat (d1 d2 : Digit) : Nat := 10 * d1.val + d2.val

-- Define a function to convert a three-digit number to a natural number
def threeDigitToNat (d1 d2 d3 : Digit) : Nat := 100 * d1.val + 10 * d2.val + d3.val

-- Define a function to convert a four-digit number to a natural number
def fourDigitToNat (d1 d2 d3 d4 : Digit) : Nat := 1000 * d1.val + 100 * d2.val + 10 * d3.val + d4.val

theorem multiplication_problem (A B C E F : Digit) :
  A ≠ B → A ≠ C → A ≠ E → A ≠ F →
  B ≠ C → B ≠ E → B ≠ F →
  C ≠ E → C ≠ F →
  E ≠ F →
  Nat.Prime (twoDigitToNat E F) →
  threeDigitToNat A B C * twoDigitToNat E F = fourDigitToNat E F E F →
  A.val + B.val = 1 := by
  sorry

end multiplication_problem_l810_81018


namespace parabola_coeffs_sum_l810_81048

/-- Parabola coefficients -/
structure ParabolaCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Parabola equation -/
def parabola_equation (coeffs : ParabolaCoeffs) (y : ℝ) : ℝ :=
  coeffs.a * y^2 + coeffs.b * y + coeffs.c

/-- Theorem: Parabola coefficients and their sum -/
theorem parabola_coeffs_sum :
  ∀ (coeffs : ParabolaCoeffs),
  parabola_equation coeffs 5 = 6 ∧
  parabola_equation coeffs 3 = 0 ∧
  (∀ y : ℝ, parabola_equation coeffs y = coeffs.a * (y - 3)^2) →
  coeffs.a = 3/2 ∧ coeffs.b = -9 ∧ coeffs.c = 27/2 ∧
  coeffs.a + coeffs.b + coeffs.c = 6 :=
by sorry


end parabola_coeffs_sum_l810_81048


namespace geometric_series_solution_l810_81081

theorem geometric_series_solution (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (6 * n - 2) / k^n = 3) : k = 2 := by
  sorry

end geometric_series_solution_l810_81081


namespace y_divisibility_l810_81006

def y : ℕ := 80 + 120 + 160 + 240 + 360 + 400 + 3600

theorem y_divisibility :
  (∃ k : ℕ, y = 5 * k) ∧
  (∃ k : ℕ, y = 10 * k) ∧
  (∃ k : ℕ, y = 20 * k) ∧
  ¬(∃ k : ℕ, y = 40 * k) :=
by sorry

end y_divisibility_l810_81006


namespace percentage_of_women_l810_81026

theorem percentage_of_women (initial_workers : ℕ) (initial_men_fraction : ℚ) 
  (new_hires : ℕ) : 
  initial_workers = 90 → 
  initial_men_fraction = 2/3 → 
  new_hires = 10 → 
  let total_workers := initial_workers + new_hires
  let initial_women := initial_workers * (1 - initial_men_fraction)
  let total_women := initial_women + new_hires
  (total_women / total_workers : ℚ) * 100 = 40 := by
  sorry

end percentage_of_women_l810_81026


namespace tangent_sum_simplification_l810_81093

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (50 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (40 * π / 180) = -Real.sin (20 * π / 180) := by
  sorry

end tangent_sum_simplification_l810_81093


namespace cube_plus_reciprocal_cube_l810_81030

theorem cube_plus_reciprocal_cube (r : ℝ) (hr : r ≠ 0) 
  (h : (r + 1/r)^2 = 5) : 
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end cube_plus_reciprocal_cube_l810_81030


namespace x_equation_solution_l810_81090

theorem x_equation_solution (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^8 + 2*x^4 = 44387*x - 15088 := by
  sorry

end x_equation_solution_l810_81090


namespace height_ratio_of_cones_l810_81062

/-- The ratio of heights of two right circular cones with the same base circumference -/
theorem height_ratio_of_cones (r : ℝ) (h₁ h₂ : ℝ) :
  r > 0 → h₁ > 0 → h₂ > 0 →
  (2 * Real.pi * r = 20 * Real.pi) →
  ((1/3) * Real.pi * r^2 * h₁ = 400 * Real.pi) →
  h₂ = 40 →
  h₁ / h₂ = 3/10 := by
sorry

end height_ratio_of_cones_l810_81062


namespace equal_average_groups_product_l810_81063

theorem equal_average_groups_product (groups : Fin 3 → List ℕ) : 
  (∀ i : Fin 3, ∀ n ∈ groups i, 1 ≤ n ∧ n ≤ 99) →
  (groups 0).sum + (groups 1).sum + (groups 2).sum = List.sum (List.range 99) →
  (groups 0).length + (groups 1).length + (groups 2).length = 99 →
  (groups 0).sum / (groups 0).length = (groups 1).sum / (groups 1).length →
  (groups 1).sum / (groups 1).length = (groups 2).sum / (groups 2).length →
  ((groups 0).sum / (groups 0).length) * ((groups 1).sum / (groups 1).length) * ((groups 2).sum / (groups 2).length) = 125000 := by
sorry

end equal_average_groups_product_l810_81063


namespace min_value_product_equality_condition_l810_81075

theorem min_value_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) 
  (h_eq : x₁ = π/4 ∧ x₂ = π/4 ∧ x₃ = π/4 ∧ x₄ = π/4) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) = 81 :=
by sorry

end min_value_product_equality_condition_l810_81075


namespace negation_of_existence_negation_of_quadratic_inequality_l810_81069

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∀ x : ℝ, x^2 - x + 1 < 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l810_81069


namespace toms_speed_from_r_to_b_l810_81035

/-- Represents the speed of a journey between two towns -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Represents Tom's entire trip -/
structure TripData where
  rb : Journey
  bc : Journey
  averageSpeed : ℝ

theorem toms_speed_from_r_to_b (trip : TripData) : trip.rb.speed = 60 :=
  by
  have h1 : trip.rb.distance = 2 * trip.bc.distance := by sorry
  have h2 : trip.averageSpeed = 36 := by sorry
  have h3 : trip.bc.speed = 20 := by sorry
  have h4 : trip.averageSpeed = (trip.rb.distance + trip.bc.distance) / 
    (trip.rb.distance / trip.rb.speed + trip.bc.distance / trip.bc.speed) := by sorry
  sorry


end toms_speed_from_r_to_b_l810_81035


namespace items_washed_is_500_l810_81049

/-- Calculates the total number of items washed given the number of loads, towels per load, and shirts per load. -/
def total_items_washed (loads : ℕ) (towels_per_load : ℕ) (shirts_per_load : ℕ) : ℕ :=
  loads * (towels_per_load + shirts_per_load)

/-- Proves that the total number of items washed is 500 given the specific conditions. -/
theorem items_washed_is_500 :
  total_items_washed 20 15 10 = 500 := by
  sorry

end items_washed_is_500_l810_81049


namespace distance_sum_to_axes_l810_81096

/-- The sum of distances from point P(-1, -2) to x-axis and y-axis is 3 -/
theorem distance_sum_to_axes : 
  let P : ℝ × ℝ := (-1, -2)
  abs P.2 + abs P.1 = 3 := by
  sorry

end distance_sum_to_axes_l810_81096


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l810_81000

theorem sum_of_roots_quadratic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let roots := {x : ℝ | f x = d}
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z, z ∈ roots → z = x ∨ z = y) →
  x + y = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x + 8
  let roots := {x : ℝ | f x = 15}
  (∃ x y, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z, z ∈ roots → z = x ∨ z = y) →
  x + y = 2 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l810_81000


namespace exists_alpha_floor_minus_n_even_l810_81009

theorem exists_alpha_floor_minus_n_even :
  ∃ α : ℝ, α > 0 ∧ ∀ n : ℕ, n > 0 → Even (⌊α * n⌋ - n) := by
  sorry

end exists_alpha_floor_minus_n_even_l810_81009


namespace comic_book_stacking_arrangements_l810_81084

def spiderman_comics : ℕ := 6
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 4
def batman_comics : ℕ := 3

def total_comics : ℕ := spiderman_comics + archie_comics + garfield_comics + batman_comics

def comic_groups : ℕ := 4

theorem comic_book_stacking_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * batman_comics.factorial) *
  comic_groups.factorial = 1244160 :=
by sorry

end comic_book_stacking_arrangements_l810_81084


namespace largest_prime_factor_l810_81065

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Prime p ∧ p ∣ (17^4 + 3 * 17^2 + 1 - 16^4) ∧ 
    ∀ q : ℕ, Prime q → q ∣ (17^4 + 3 * 17^2 + 1 - 16^4) → q ≤ p) → 
  (∃ p : ℕ, p = 17 ∧ Prime p ∧ p ∣ (17^4 + 3 * 17^2 + 1 - 16^4) ∧ 
    ∀ q : ℕ, Prime q → q ∣ (17^4 + 3 * 17^2 + 1 - 16^4) → q ≤ p) := by
  sorry

end largest_prime_factor_l810_81065


namespace square_sum_equals_five_l810_81089

theorem square_sum_equals_five (x y : ℝ) (h1 : (x - y)^2 = 25) (h2 : x * y = -10) :
  x^2 + y^2 = 5 := by
  sorry

end square_sum_equals_five_l810_81089


namespace smith_family_laundry_l810_81064

/-- The number of bath towels Kylie uses in one month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels Kylie's daughters use in one month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels Kylie's husband uses in one month -/
def husband_towels : ℕ := 3

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The total number of bath towels used by the Smith family in one month -/
def total_towels : ℕ := kylie_towels + daughters_towels + husband_towels

/-- The number of loads of laundry needed to clean all used towels -/
def loads_needed : ℕ := (total_towels + towels_per_load - 1) / towels_per_load

theorem smith_family_laundry : loads_needed = 3 := by
  sorry

end smith_family_laundry_l810_81064


namespace quadratic_root_problem_l810_81066

theorem quadratic_root_problem (a : ℝ) : 
  ((a - 1) * 1^2 - a * 1 + a^2 = 0) → (a ≠ 1) → (a = -1) := by
  sorry

end quadratic_root_problem_l810_81066


namespace marble_weight_difference_is_8_l810_81053

/-- Calculates the difference in weight between red and yellow marbles -/
def marble_weight_difference (total_marbles : ℕ) (yellow_marbles : ℕ) (blue_red_ratio : ℚ) 
  (yellow_weight : ℝ) (red_yellow_weight_ratio : ℝ) : ℝ :=
  let remaining_marbles := total_marbles - yellow_marbles
  let red_marbles := (remaining_marbles : ℝ) * (1 / (1 + blue_red_ratio)) * (blue_red_ratio / (1 + blue_red_ratio))⁻¹
  let red_weight := yellow_weight * red_yellow_weight_ratio
  red_weight - yellow_weight

theorem marble_weight_difference_is_8 :
  marble_weight_difference 19 5 (3/4) 8 2 = 8 := by
  sorry

end marble_weight_difference_is_8_l810_81053


namespace work_completion_time_l810_81099

/-- Represents the work rate of one person for one hour -/
structure WorkRate where
  man : ℝ
  woman : ℝ

/-- Represents a work scenario -/
structure WorkScenario where
  men : ℕ
  women : ℕ
  hours : ℝ
  days : ℝ

/-- Calculates the total work done in a scenario -/
def totalWork (rate : WorkRate) (scenario : WorkScenario) : ℝ :=
  (scenario.men * rate.man + scenario.women * rate.woman) * scenario.hours * scenario.days

/-- The theorem to be proved -/
theorem work_completion_time 
  (rate : WorkRate)
  (scenario1 scenario2 scenario3 : WorkScenario) :
  scenario1.men = 1 ∧ 
  scenario1.women = 3 ∧ 
  scenario1.hours = 7 ∧
  scenario2.men = 4 ∧ 
  scenario2.women = 4 ∧ 
  scenario2.hours = 3 ∧ 
  scenario2.days = 7 ∧
  scenario3.men = 7 ∧ 
  scenario3.women = 0 ∧ 
  scenario3.hours = 4 ∧ 
  scenario3.days = 5.000000000000001 ∧
  totalWork rate scenario1 = totalWork rate scenario2 ∧
  totalWork rate scenario2 = totalWork rate scenario3
  →
  scenario1.days = 20/3 := by
  sorry

end work_completion_time_l810_81099


namespace complement_A_intersect_B_l810_81013

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {-1, 2, 3}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {2, 3} := by sorry

end complement_A_intersect_B_l810_81013


namespace profit_calculation_l810_81020

theorem profit_calculation (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : marked_price > 0)
  (h3 : 0.8 * marked_price = 1.2 * cost_price) :
  (marked_price - cost_price) / cost_price = 0.5 := by
sorry

end profit_calculation_l810_81020


namespace ball_placement_events_l810_81094

structure Ball :=
  (number : Nat)

structure Box :=
  (number : Nat)

def Placement := Ball → Box

def event_ball1_in_box1 (p : Placement) : Prop :=
  p ⟨1⟩ = ⟨1⟩

def event_ball1_in_box2 (p : Placement) : Prop :=
  p ⟨1⟩ = ⟨2⟩

def mutually_exclusive (e1 e2 : Placement → Prop) : Prop :=
  ∀ p : Placement, ¬(e1 p ∧ e2 p)

def complementary (e1 e2 : Placement → Prop) : Prop :=
  ∀ p : Placement, e1 p ↔ ¬(e2 p)

theorem ball_placement_events :
  (mutually_exclusive event_ball1_in_box1 event_ball1_in_box2) ∧
  ¬(complementary event_ball1_in_box1 event_ball1_in_box2) :=
sorry

end ball_placement_events_l810_81094


namespace count_threes_to_1000_l810_81014

/-- Count of digit 3 appearances when listing integers from 1 to n -/
def count_threes (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the count of digit 3 appearances from 1 to 1000 is 300 -/
theorem count_threes_to_1000 : count_threes 1000 = 300 := by
  sorry

end count_threes_to_1000_l810_81014


namespace train_passes_jogger_train_passes_jogger_time_l810_81045

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : Real) (train_speed : Real) 
  (jogger_lead : Real) (train_length : Real) : Real :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := jogger_lead + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is 24 seconds -/
theorem train_passes_jogger_time :
  train_passes_jogger 9 45 120 120 = 24 := by
  sorry

end train_passes_jogger_train_passes_jogger_time_l810_81045


namespace unique_zero_of_increasing_cubic_l810_81095

/-- Given an increasing function f(x) = x^3 + bx + c on [-1, 1] with f(1/2) * f(-1/2) < 0,
    prove that f has exactly one zero in [-1, 1]. -/
theorem unique_zero_of_increasing_cubic {b c : ℝ} :
  let f : ℝ → ℝ := λ x ↦ x^3 + b*x + c
  (∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x < f y) →
  f (1/2) * f (-1/2) < 0 →
  ∃! x, x ∈ [-1, 1] ∧ f x = 0 :=
by sorry

end unique_zero_of_increasing_cubic_l810_81095


namespace probability_red_ball_3_2_l810_81029

/-- Represents the probability of drawing a red ball from a box containing red and yellow balls. -/
def probability_red_ball (red_balls yellow_balls : ℕ) : ℚ :=
  red_balls / (red_balls + yellow_balls)

/-- Theorem stating that the probability of drawing a red ball from a box with 3 red balls and 2 yellow balls is 3/5. -/
theorem probability_red_ball_3_2 :
  probability_red_ball 3 2 = 3 / 5 := by
  sorry

end probability_red_ball_3_2_l810_81029


namespace video_game_lives_l810_81027

theorem video_game_lives (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) :
  initial_players = 11 →
  quitting_players = 5 →
  total_lives = 30 →
  (total_lives / (initial_players - quitting_players) = 5) :=
by sorry

end video_game_lives_l810_81027


namespace decision_block_two_exits_l810_81091

-- Define the types of program blocks
inductive ProgramBlock
  | Termination
  | InputOutput
  | Processing
  | Decision

-- Define a function to determine if a block has two exit directions
def hasTwoExitDirections (block : ProgramBlock) : Prop :=
  match block with
  | ProgramBlock.Decision => true
  | _ => false

-- Theorem statement
theorem decision_block_two_exits :
  ∀ (block : ProgramBlock),
    hasTwoExitDirections block ↔ block = ProgramBlock.Decision :=
by sorry

end decision_block_two_exits_l810_81091


namespace account_balance_after_transactions_l810_81087

/-- Calculates the final account balance after a series of transactions --/
def finalBalance (initialBalance : ℚ) 
  (transfer1 transfer2 transfer3 transfer4 transfer5 : ℚ)
  (serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 : ℚ) : ℚ :=
  initialBalance - transfer1 - (transfer3 + serviceCharge3) - (transfer5 + serviceCharge5)

/-- Theorem stating the final account balance after the given transactions --/
theorem account_balance_after_transactions 
  (initialBalance : ℚ)
  (transfer1 transfer2 transfer3 transfer4 transfer5 : ℚ)
  (serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 : ℚ)
  (h1 : initialBalance = 400)
  (h2 : transfer1 = 90)
  (h3 : transfer2 = 60)
  (h4 : transfer3 = 50)
  (h5 : transfer4 = 120)
  (h6 : transfer5 = 200)
  (h7 : serviceCharge1 = 0.02 * transfer1)
  (h8 : serviceCharge2 = 0.02 * transfer2)
  (h9 : serviceCharge3 = 0.02 * transfer3)
  (h10 : serviceCharge4 = 0.025 * transfer4)
  (h11 : serviceCharge5 = 0.03 * transfer5) :
  finalBalance initialBalance transfer1 transfer2 transfer3 transfer4 transfer5
    serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 = 53 := by
  sorry


end account_balance_after_transactions_l810_81087


namespace raspberry_carton_is_eight_ounces_l810_81042

/-- Represents the cost and size of fruit cartons, and the amount needed for muffins --/
structure FruitData where
  blueberry_cost : ℚ
  blueberry_size : ℚ
  raspberry_cost : ℚ
  batches : ℕ
  fruit_per_batch : ℚ
  savings : ℚ

/-- Calculates the size of a raspberry carton based on the given data --/
def raspberry_carton_size (data : FruitData) : ℚ :=
  sorry

/-- Theorem stating that the raspberry carton size is 8 ounces --/
theorem raspberry_carton_is_eight_ounces (data : FruitData)
    (h1 : data.blueberry_cost = 5)
    (h2 : data.blueberry_size = 6)
    (h3 : data.raspberry_cost = 3)
    (h4 : data.batches = 4)
    (h5 : data.fruit_per_batch = 12)
    (h6 : data.savings = 22) :
    raspberry_carton_size data = 8 := by
  sorry

end raspberry_carton_is_eight_ounces_l810_81042


namespace even_tower_for_odd_walls_l810_81077

/-- A standard die has opposite faces summing to 7 -/
structure StandardDie where
  faces : Fin 6 → ℕ
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- A tower of dice -/
def DiceTower (n : ℕ) := Fin n → StandardDie

/-- The sum of visible dots on a vertical wall of the tower -/
def wall_sum (tower : DiceTower n) (wall : Fin 4) : ℕ := sorry

theorem even_tower_for_odd_walls (n : ℕ) (tower : DiceTower n) :
  (∀ wall : Fin 4, Odd (wall_sum tower wall)) → Even n := by sorry

end even_tower_for_odd_walls_l810_81077


namespace piece_length_in_cm_l810_81003

-- Define the length of the rod in meters
def rod_length : ℝ := 25.5

-- Define the number of pieces that can be cut from the rod
def num_pieces : ℕ := 30

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem statement
theorem piece_length_in_cm : 
  (rod_length / num_pieces) * meters_to_cm = 85 := by
  sorry

end piece_length_in_cm_l810_81003


namespace bookstore_problem_l810_81098

theorem bookstore_problem (x y : ℕ) 
  (h1 : x + y = 5000)
  (h2 : (x - 400) / 2 - (y + 400) = 400) :
  x - y = 3000 := by
  sorry

end bookstore_problem_l810_81098


namespace inequality_proof_l810_81019

theorem inequality_proof (x : ℝ) (h : x > 0) : x + (2016^2016)/x^2016 ≥ 2017 := by
  sorry

end inequality_proof_l810_81019


namespace yah_to_bah_conversion_l810_81071

/-- Define conversion rates between bahs, rahs, and yahs -/
def bah_to_rah_rate : ℚ := 27 / 18
def rah_to_yah_rate : ℚ := 20 / 12

/-- Theorem stating the equivalence between 800 yahs and 320 bahs -/
theorem yah_to_bah_conversion : 
  ∀ (bahs rahs yahs : ℚ),
  (18 : ℚ) * bahs = (27 : ℚ) * rahs →
  (12 : ℚ) * rahs = (20 : ℚ) * yahs →
  (800 : ℚ) * yahs = (320 : ℚ) * bahs := by
  sorry

end yah_to_bah_conversion_l810_81071


namespace chameleon_color_change_l810_81022

theorem chameleon_color_change (total : ℕ) (blue_factor red_factor : ℕ) : 
  total = 140 ∧ blue_factor = 5 ∧ red_factor = 3 →
  ∃ (initial_blue initial_red changed : ℕ),
    initial_blue + initial_red = total ∧
    changed = initial_blue - (initial_blue / blue_factor) ∧
    initial_red + changed = (initial_red * red_factor) ∧
    changed = 80 := by
  sorry

end chameleon_color_change_l810_81022


namespace urns_can_be_emptied_l810_81024

/-- Represents the two types of operations that can be performed on the urns -/
inductive UrnOperation
  | Remove : ℕ → UrnOperation
  | DoubleFirst : UrnOperation
  | DoubleSecond : UrnOperation

/-- Applies a single operation to the pair of urns -/
def applyOperation (a b : ℕ) (op : UrnOperation) : ℕ × ℕ :=
  match op with
  | UrnOperation.Remove n => (a - min a n, b - min b n)
  | UrnOperation.DoubleFirst => (2 * a, b)
  | UrnOperation.DoubleSecond => (a, 2 * b)

/-- Theorem: Both urns can be made empty after a finite number of operations -/
theorem urns_can_be_emptied (a b : ℕ) :
  ∃ (ops : List UrnOperation), (ops.foldl (fun (pair : ℕ × ℕ) (op : UrnOperation) => applyOperation pair.1 pair.2 op) (a, b)).1 = 0 ∧
                               (ops.foldl (fun (pair : ℕ × ℕ) (op : UrnOperation) => applyOperation pair.1 pair.2 op) (a, b)).2 = 0 := by
  sorry

end urns_can_be_emptied_l810_81024


namespace square_sum_given_difference_and_product_l810_81073

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end square_sum_given_difference_and_product_l810_81073


namespace cos_sin_identity_l810_81047

theorem cos_sin_identity (a : Real) (h : Real.cos (π/6 - a) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + a) - Real.sin (a - π/6)^2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end cos_sin_identity_l810_81047


namespace correct_systematic_sample_l810_81079

def population_size : ℕ := 30
def sample_size : ℕ := 6

def systematic_sampling_interval (pop_size sample_size : ℕ) : ℕ :=
  pop_size / sample_size

def generate_sample (start interval : ℕ) (size : ℕ) : List ℕ :=
  List.range size |>.map (λ i => start + i * interval)

theorem correct_systematic_sample :
  let interval := systematic_sampling_interval population_size sample_size
  let sample := generate_sample 2 interval sample_size
  (interval = 5) ∧ (sample = [2, 7, 12, 17, 22, 27]) := by
  sorry

#eval systematic_sampling_interval population_size sample_size
#eval generate_sample 2 (systematic_sampling_interval population_size sample_size) sample_size

end correct_systematic_sample_l810_81079


namespace function_properties_l810_81052

noncomputable def f (a b x : ℝ) : ℝ := 6 * Real.log x - a * x^2 - 7 * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, x > 0 → HasDerivAt (f a b) ((6 / x) - 2 * a * x - 7) x) →
  HasDerivAt (f a b) 0 2 →
  (a = -1 ∧
   (∀ x, 0 < x ∧ x < 3/2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x > 0)) ∧
   (∀ x, 3/2 < x ∧ x < 2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x < 0)) ∧
   (∀ x, x > 2 → HasDerivAt (f a b) ((x - 2) * (2 * x - 3) / x) x ∧ ((x - 2) * (2 * x - 3) / x > 0)) ∧
   (33/4 - 6 * Real.log (3/2) < b ∧ b < 10 - 6 * Real.log 2)) :=
by sorry

end function_properties_l810_81052


namespace min_monochromatic_triangles_l810_81057

/-- A coloring of edges in a complete graph on 2k vertices. -/
def Coloring (k : ℕ) := Fin (2*k) → Fin (2*k) → Bool

/-- The number of monochromatic triangles in a coloring. -/
def monochromaticTriangles (k : ℕ) (c : Coloring k) : ℕ := sorry

/-- The statement of the problem. -/
theorem min_monochromatic_triangles (k : ℕ) (h : k ≥ 3) :
  ∃ (c : Coloring k), monochromaticTriangles k c = k * (k - 1) * (k - 2) / 3 ∧
  ∀ (c' : Coloring k), monochromaticTriangles k c' ≥ k * (k - 1) * (k - 2) / 3 := by
  sorry

end min_monochromatic_triangles_l810_81057


namespace company_average_salary_l810_81037

theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℚ)
  (avg_salary_associates : ℚ)
  (h1 : num_managers = 15)
  (h2 : num_associates = 75)
  (h3 : avg_salary_managers = 90000)
  (h4 : avg_salary_associates = 30000) :
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) /
  (num_managers + num_associates : ℚ) = 40000 :=
by sorry

end company_average_salary_l810_81037


namespace valid_permutations_count_l810_81004

/-- Given integers 1 to n, where n ≥ 2, this function returns the number of permutations
    satisfying the condition that for all k = 1 to n, the kth element is ≥ k-2 -/
def countValidPermutations (n : ℕ) : ℕ :=
  2 * 3^(n-2)

/-- Theorem stating that for n ≥ 2, the number of permutations of integers 1 to n
    satisfying the condition that for all k = 1 to n, the kth element is ≥ k-2,
    is equal to 2 * 3^(n-2) -/
theorem valid_permutations_count (n : ℕ) (h : n ≥ 2) :
  (Finset.univ.filter (fun p : Fin n → Fin n =>
    ∀ k : Fin n, p k ≥ ⟨k - 2, by sorry⟩)).card = countValidPermutations n := by
  sorry

end valid_permutations_count_l810_81004


namespace total_books_two_months_l810_81085

def books_last_month : ℕ := 4

def books_this_month (n : ℕ) : ℕ := 2 * n

theorem total_books_two_months : 
  books_last_month + books_this_month books_last_month = 12 := by
  sorry

end total_books_two_months_l810_81085


namespace odd_totient_power_of_two_l810_81005

theorem odd_totient_power_of_two (n : ℕ) 
  (h_odd : Odd n)
  (h_phi_n : ∃ k : ℕ, Nat.totient n = 2^k)
  (h_phi_n_plus_one : ∃ m : ℕ, Nat.totient (n+1) = 2^m) :
  (∃ p : ℕ, n+1 = 2^p) ∨ n = 5 := by
sorry

end odd_totient_power_of_two_l810_81005


namespace power_three_315_mod_11_l810_81038

theorem power_three_315_mod_11 : 3^315 % 11 = 1 := by
  sorry

end power_three_315_mod_11_l810_81038


namespace rand_code_is_1236_l810_81051

/-- A coding system that assigns numerical codes to words -/
structure CodeSystem where
  range_code : Nat
  random_code : Nat

/-- The code for a given word in the coding system -/
def word_code (system : CodeSystem) (word : String) : Nat :=
  sorry

/-- Our specific coding system -/
def our_system : CodeSystem :=
  { range_code := 12345, random_code := 123678 }

theorem rand_code_is_1236 :
  word_code our_system "rand" = 1236 := by
  sorry

end rand_code_is_1236_l810_81051


namespace hari_contribution_correct_l810_81088

/-- Calculates Hari's contribution to the capital given the initial conditions of the business partnership --/
def calculate_hari_contribution (praveen_capital : ℕ) (praveen_months : ℕ) (hari_months : ℕ) (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) : ℕ :=
  (3 * praveen_capital * praveen_months) / (2 * hari_months)

theorem hari_contribution_correct :
  let praveen_capital : ℕ := 3780
  let total_months : ℕ := 12
  let hari_join_month : ℕ := 5
  let profit_ratio_praveen : ℕ := 2
  let profit_ratio_hari : ℕ := 3
  let praveen_months : ℕ := total_months
  let hari_months : ℕ := total_months - hari_join_month
  calculate_hari_contribution praveen_capital praveen_months hari_months profit_ratio_praveen profit_ratio_hari = 9720 :=
by
  sorry

#eval calculate_hari_contribution 3780 12 7 2 3

end hari_contribution_correct_l810_81088


namespace total_worth_of_cloth_sold_l810_81039

/-- Calculates the total worth of cloth sold through two agents given their commission rates and amounts -/
theorem total_worth_of_cloth_sold 
  (rate_A rate_B : ℝ) 
  (commission_A commission_B : ℝ) 
  (h1 : rate_A = 0.025) 
  (h2 : rate_B = 0.03) 
  (h3 : commission_A = 21) 
  (h4 : commission_B = 27) : 
  ∃ (total_worth : ℝ), total_worth = commission_A / rate_A + commission_B / rate_B :=
sorry

end total_worth_of_cloth_sold_l810_81039


namespace unique_divisible_by_18_l810_81056

def is_divisible_by_18 (n : ℕ) : Prop := n % 18 = 0

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 520 + x

theorem unique_divisible_by_18 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_18 (four_digit_number x) :=
sorry

end unique_divisible_by_18_l810_81056


namespace equation_solution_l810_81070

theorem equation_solution : ∃ x : ℝ, 90 + 5 * x / (180 / 3) = 91 ∧ x = 12 := by
  sorry

end equation_solution_l810_81070


namespace arithmetic_sequence_terms_l810_81041

/-- An arithmetic sequence is defined by its first term, common difference, and last term. -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ
  last : ℤ

/-- The number of terms in an arithmetic sequence. -/
def numTerms (seq : ArithmeticSequence) : ℤ :=
  (seq.last - seq.first) / seq.diff + 1

/-- Theorem: The arithmetic sequence with first term 13, common difference 3, and last term 73 has exactly 21 terms. -/
theorem arithmetic_sequence_terms : 
  let seq := ArithmeticSequence.mk 13 3 73
  numTerms seq = 21 := by
  sorry

end arithmetic_sequence_terms_l810_81041


namespace square_sum_reciprocal_l810_81017

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 10) : x^2 + 1/x^2 = 98 := by
  sorry

end square_sum_reciprocal_l810_81017


namespace stamp_ratio_problem_l810_81028

theorem stamp_ratio_problem (k a : ℕ) : 
  k > 0 ∧ a > 0 →  -- Initial numbers of stamps are positive
  (k - 12) / (a + 12) = 8 / 6 →  -- Ratio after exchange
  k - 12 = a + 12 + 32 →  -- Kaye has 32 more stamps after exchange
  k / a = 5 / 3 :=  -- Initial ratio
by sorry

end stamp_ratio_problem_l810_81028
