import Mathlib

namespace fraction_addition_l2051_205164

theorem fraction_addition (d : ℝ) : (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := by
  sorry

end fraction_addition_l2051_205164


namespace inscribed_circles_radii_l2051_205173

/-- Three circles inscribed in a corner --/
structure InscribedCircles where
  r : ℝ  -- radius of the small circle
  a : ℝ  -- distance from center of small circle to corner vertex
  x : ℝ  -- radius of the medium circle
  y : ℝ  -- radius of the large circle

/-- The configuration of the inscribed circles --/
def valid_configuration (c : InscribedCircles) : Prop :=
  c.r > 0 ∧ c.a > c.r ∧ c.x > c.r ∧ c.y > c.x

/-- The theorem stating the radii of medium and large circles --/
theorem inscribed_circles_radii (c : InscribedCircles) 
  (h : valid_configuration c) : 
  c.x = (c.a * c.r) / (c.a - c.r) ∧ 
  c.y = (c.a^2 * c.r) / (c.a - c.r)^2 :=
sorry

end inscribed_circles_radii_l2051_205173


namespace q_minus_p_equals_zero_l2051_205166

def P : Set ℕ := {1, 2, 3, 4, 5}
def Q : Set ℕ := {0, 2, 3}

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem q_minus_p_equals_zero : set_difference Q P = {0} := by sorry

end q_minus_p_equals_zero_l2051_205166


namespace pentagon_area_sum_l2051_205194

theorem pentagon_area_sum (u v : ℤ) 
  (h1 : 0 < v) (h2 : v < u) 
  (h3 : u^2 + 3*u*v = 451) : u + v = 21 := by
  sorry

end pentagon_area_sum_l2051_205194


namespace weeks_per_month_l2051_205115

theorem weeks_per_month (months : ℕ) (weekly_rate : ℚ) (monthly_rate : ℚ) (savings : ℚ) :
  months = 3 ∧ 
  weekly_rate = 280 ∧ 
  monthly_rate = 1000 ∧
  savings = 360 →
  (months * monthly_rate + savings) / (months * weekly_rate) = 4 := by
sorry

end weeks_per_month_l2051_205115


namespace circle_radius_l2051_205120

theorem circle_radius (C : ℝ) (h : C = 72 * Real.pi) : C / (2 * Real.pi) = 36 := by
  sorry

end circle_radius_l2051_205120


namespace magnitude_of_a_l2051_205183

def a (t : ℝ) : ℝ × ℝ := (1, t)
def b (t : ℝ) : ℝ × ℝ := (-1, t)

theorem magnitude_of_a (t : ℝ) :
  (2 * a t - b t) • b t = 0 → ‖a t‖ = 2 := by
  sorry

end magnitude_of_a_l2051_205183


namespace helga_extra_hours_l2051_205156

/-- Represents Helga's work schedule and productivity --/
structure HelgaWork where
  articles_per_half_hour : ℕ := 5
  regular_hours_per_day : ℕ := 4
  regular_days_per_week : ℕ := 5
  extra_hours_thursday : ℕ := 2
  total_articles_week : ℕ := 250

/-- Calculates the number of extra hours Helga worked on Friday --/
def extra_hours_friday (hw : HelgaWork) : ℕ :=
  sorry

/-- Theorem stating that Helga worked 3 extra hours on Friday --/
theorem helga_extra_hours (hw : HelgaWork) : extra_hours_friday hw = 3 := by
  sorry

end helga_extra_hours_l2051_205156


namespace inequality_proof_l2051_205186

theorem inequality_proof (x : ℝ) : 
  Real.sqrt (3 * x^2 + 2 * x + 1) + Real.sqrt (3 * x^2 - 4 * x + 2) ≥ Real.sqrt 51 / 3 := by
  sorry

end inequality_proof_l2051_205186


namespace johnson_family_reunion_ratio_l2051_205121

theorem johnson_family_reunion_ratio : 
  let num_children : ℕ := 45
  let num_adults : ℕ := num_children / 3
  let adults_not_blue : ℕ := 10
  let adults_blue : ℕ := num_adults - adults_not_blue
  adults_blue / num_adults = 1 / 3 := by
  sorry

end johnson_family_reunion_ratio_l2051_205121


namespace min_value_cubic_quadratic_l2051_205192

theorem min_value_cubic_quadratic (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : 57 * a + 88 * b + 125 * c ≥ 1148) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 57 * x + 88 * y + 125 * z ≥ 1148 →
  a^3 + b^3 + c^3 + 5*a^2 + 5*b^2 + 5*c^2 ≤ x^3 + y^3 + z^3 + 5*x^2 + 5*y^2 + 5*z^2 :=
by sorry

end min_value_cubic_quadratic_l2051_205192


namespace gasoline_price_increase_percentage_l2051_205143

def highest_price : ℝ := 24
def lowest_price : ℝ := 12

theorem gasoline_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 100 := by
  sorry

end gasoline_price_increase_percentage_l2051_205143


namespace sum_of_three_numbers_l2051_205172

theorem sum_of_three_numbers (a b c : ℤ) (N : ℚ) : 
  a + b + c = 80 ∧ 
  2 * a = N ∧ 
  b - 10 = N ∧ 
  3 * c = N → 
  N = 38 := by sorry

end sum_of_three_numbers_l2051_205172


namespace geometric_sequence_minimum_value_l2051_205198

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_cond : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ x : ℝ, 2 * a 5 + a 4 ≥ m :=
by sorry

end geometric_sequence_minimum_value_l2051_205198


namespace bookshop_inventory_l2051_205111

theorem bookshop_inventory (books_sold : ℕ) (percentage_sold : ℚ) (initial_stock : ℕ) : 
  books_sold = 280 → percentage_sold = 2/5 → initial_stock * percentage_sold = books_sold → 
  initial_stock = 700 := by
sorry

end bookshop_inventory_l2051_205111


namespace prob_six_odd_in_eight_rolls_l2051_205146

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1 / 2

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The number of desired odd rolls -/
def num_odd : ℕ := 6

/-- The probability of getting exactly 6 odd numbers in 8 rolls of a fair 6-sided die -/
theorem prob_six_odd_in_eight_rolls : 
  (Nat.choose num_rolls num_odd : ℚ) * prob_odd ^ num_odd * (1 - prob_odd) ^ (num_rolls - num_odd) = 7 / 64 := by
  sorry

end prob_six_odd_in_eight_rolls_l2051_205146


namespace opposite_corner_not_always_farthest_l2051_205136

/-- A rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- A point on the surface of a box -/
structure SurfacePoint (b : Box) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = b.length) ∨ (y = 0 ∨ y = b.width) ∨ (z = 0 ∨ z = b.height)

/-- The distance between two points on the surface of a box -/
noncomputable def surface_distance (b : Box) (p1 p2 : SurfacePoint b) : ℝ :=
  sorry

/-- The corner opposite to (0, 0, 0) -/
def opposite_corner (b : Box) : SurfacePoint b :=
  { x := b.length, y := b.width, z := b.height,
    on_surface := by simp }

/-- Theorem: The opposite corner is not necessarily the point with the greatest distance from a corner -/
theorem opposite_corner_not_always_farthest (b : Box) :
  ∃ (p : SurfacePoint b), surface_distance b ⟨0, 0, 0, by simp⟩ p > 
                           surface_distance b ⟨0, 0, 0, by simp⟩ (opposite_corner b) :=
sorry

end opposite_corner_not_always_farthest_l2051_205136


namespace rectangle_diagonal_l2051_205184

/-- A rectangle with perimeter 60 cm and area 225 cm² has a diagonal of 15√2 cm. -/
theorem rectangle_diagonal (x y : ℝ) (h_perimeter : x + y = 30) (h_area : x * y = 225) :
  Real.sqrt (x^2 + y^2) = 15 * Real.sqrt 2 := by
  sorry

end rectangle_diagonal_l2051_205184


namespace equation_transformation_l2051_205141

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0 ↔ x^2 * (y^2 - y - 3) = 0 :=
by sorry

end equation_transformation_l2051_205141


namespace rosa_flowers_total_l2051_205129

theorem rosa_flowers_total (initial_flowers : Float) (additional_flowers : Float) :
  initial_flowers = 67.0 →
  additional_flowers = 90.0 →
  initial_flowers + additional_flowers = 157.0 := by
sorry

end rosa_flowers_total_l2051_205129


namespace dilution_calculation_l2051_205150

/-- Calculates the amount of water needed to dilute a shaving lotion to a desired alcohol concentration -/
theorem dilution_calculation (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.6 →
  final_concentration = 0.45 →
  ∃ (water_volume : ℝ),
    water_volume = 4 ∧
    (initial_volume * initial_concentration) / (initial_volume + water_volume) = final_concentration :=
by sorry

end dilution_calculation_l2051_205150


namespace magician_decks_left_l2051_205193

/-- A magician sells magic card decks. -/
structure Magician where
  initial_decks : ℕ  -- Number of decks at the start
  price_per_deck : ℕ  -- Price of each deck in dollars
  earnings : ℕ  -- Total earnings in dollars

/-- Calculate the number of decks left for a magician. -/
def decks_left (m : Magician) : ℕ :=
  m.initial_decks - m.earnings / m.price_per_deck

/-- Theorem: The magician has 3 decks left at the end of the day. -/
theorem magician_decks_left :
  ∀ (m : Magician),
    m.initial_decks = 5 →
    m.price_per_deck = 2 →
    m.earnings = 4 →
    decks_left m = 3 := by
  sorry

end magician_decks_left_l2051_205193


namespace ball_distribution_l2051_205142

theorem ball_distribution (a b c : ℕ) : 
  a + b + c = 45 →
  a + 2 = b - 1 ∧ a + 2 = c - 1 →
  (a, b, c) = (13, 16, 16) :=
by sorry

end ball_distribution_l2051_205142


namespace solution_set_for_a_equals_one_a_range_for_positive_f_l2051_205109

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | -2 < x ∧ x < -2/3} := by sorry

-- Part 2
theorem a_range_for_positive_f :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 3, f a x > 0) → -5/2 < a ∧ a < -2 := by sorry

end solution_set_for_a_equals_one_a_range_for_positive_f_l2051_205109


namespace point_distance_l2051_205124

-- Define the points as real numbers representing their positions on a line
variable (A B C D : ℝ)

-- Define the conditions
variable (h_order : A < B ∧ B < C ∧ C < D)
variable (h_ratio : (B - A) / (C - B) = (D - A) / (D - C))
variable (h_AC : C - A = 3)
variable (h_BD : D - B = 4)

-- State the theorem
theorem point_distance (A B C D : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D)
  (h_ratio : (B - A) / (C - B) = (D - A) / (D - C))
  (h_AC : C - A = 3)
  (h_BD : D - B = 4) : 
  D - A = 6 := by sorry

end point_distance_l2051_205124


namespace discriminant_greater_than_four_l2051_205160

theorem discriminant_greater_than_four (p q : ℝ) 
  (h1 : 999^2 + p * 999 + q < 0) 
  (h2 : 1001^2 + p * 1001 + q < 0) : 
  p^2 - 4*q > 4 := by
sorry

end discriminant_greater_than_four_l2051_205160


namespace triangle_side_relation_l2051_205181

theorem triangle_side_relation (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0 →
  a + c - 2*b = 0 := by
sorry

end triangle_side_relation_l2051_205181


namespace chocolate_boxes_given_away_tom_chocolate_boxes_l2051_205180

theorem chocolate_boxes_given_away (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : ℕ :=
  let total_pieces := total_boxes * pieces_per_box
  let pieces_given_away := total_pieces - pieces_left
  pieces_given_away / pieces_per_box

theorem tom_chocolate_boxes :
  chocolate_boxes_given_away 14 3 18 = 8 := by
  sorry

end chocolate_boxes_given_away_tom_chocolate_boxes_l2051_205180


namespace sin_over_x_satisfies_equation_l2051_205116

open Real

theorem sin_over_x_satisfies_equation (x : ℝ) (hx : x ≠ 0) :
  let y : ℝ → ℝ := fun x => sin x / x
  let y' : ℝ → ℝ := fun x => (x * cos x - sin x) / (x^2)
  x * y' x + y x = cos x := by
  sorry

end sin_over_x_satisfies_equation_l2051_205116


namespace sin_600_degrees_l2051_205128

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_degrees_l2051_205128


namespace gravel_path_cost_example_l2051_205138

/-- Calculates the cost of gravelling a path inside a rectangular plot -/
def gravel_path_cost (length width path_width gravel_cost_per_sqm : ℝ) : ℝ :=
  let total_area := length * width
  let inner_area := (length - 2 * path_width) * (width - 2 * path_width)
  let path_area := total_area - inner_area
  path_area * gravel_cost_per_sqm

/-- Theorem: The cost of gravelling the path is 425 INR -/
theorem gravel_path_cost_example : 
  gravel_path_cost 110 65 2.5 0.5 = 425 := by
sorry

end gravel_path_cost_example_l2051_205138


namespace calculate_interest_rate_l2051_205112

/-- Given simple interest, principal, and time, calculate the interest rate. -/
theorem calculate_interest_rate 
  (simple_interest principal time rate : ℝ) 
  (h1 : simple_interest = 400)
  (h2 : principal = 1200)
  (h3 : time = 4)
  (h4 : simple_interest = principal * rate * time / 100) :
  rate = 400 * 100 / (1200 * 4) :=
by sorry

end calculate_interest_rate_l2051_205112


namespace inequality_proof_l2051_205122

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end inequality_proof_l2051_205122


namespace cubic_function_property_l2051_205159

/-- Given a cubic function f(x) = ax³ + bx² with a maximum at x = 1 and f(1) = 3, prove that a + b = 3 -/
theorem cubic_function_property (a b : ℝ) : 
  let f := fun (x : ℝ) => a * x^3 + b * x^2
  let f' := fun (x : ℝ) => 3 * a * x^2 + 2 * b * x
  (f 1 = 3) → (f' 1 = 0) → (a + b = 3) :=
by
  sorry

end cubic_function_property_l2051_205159


namespace inequality_solution_l2051_205179

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x - 2) ≥ 3 / (x + 2) + 7 / 5) ↔ (x > -2 ∧ x ≠ 2) :=
sorry

end inequality_solution_l2051_205179


namespace difference_after_subtrahend_increase_difference_after_subtrahend_increase_alt_l2051_205188

/-- Given two real numbers with a difference of a, prove that if we increase the subtrahend by 0.5, the new difference is a - 0.5 -/
theorem difference_after_subtrahend_increase (x y a : ℝ) (h : x - y = a) : 
  x - (y + 0.5) = a - 0.5 := by
sorry

/-- Alternative formulation using let bindings for clarity -/
theorem difference_after_subtrahend_increase_alt (a : ℝ) : 
  ∀ x y : ℝ, x - y = a → x - (y + 0.5) = a - 0.5 := by
sorry

end difference_after_subtrahend_increase_difference_after_subtrahend_increase_alt_l2051_205188


namespace mcdonald_farm_weeks_l2051_205110

/-- The number of weeks required for Mcdonald's farm to produce the total number of eggs -/
def weeks_required (saly_eggs ben_eggs total_eggs : ℕ) : ℕ :=
  total_eggs / (saly_eggs + ben_eggs + ben_eggs / 2)

/-- Theorem stating that the number of weeks required is 4 -/
theorem mcdonald_farm_weeks : weeks_required 10 14 124 = 4 := by
  sorry

end mcdonald_farm_weeks_l2051_205110


namespace kelly_initial_apples_l2051_205105

/-- The number of apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly will have after picking -/
def total_apples : ℕ := 105

/-- The initial number of apples Kelly had -/
def initial_apples : ℕ := total_apples - apples_to_pick

theorem kelly_initial_apples :
  initial_apples = 56 :=
sorry

end kelly_initial_apples_l2051_205105


namespace quadratic_inequality_range_l2051_205134

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end quadratic_inequality_range_l2051_205134


namespace area_is_72_l2051_205114

/-- A square with side length 12 and a right triangle in a plane -/
structure Configuration :=
  (square_side : ℝ)
  (square_lower_right : ℝ × ℝ)
  (triangle_base : ℝ)
  (hypotenuse_end : ℝ × ℝ)

/-- The area of the region formed by the portion of the square below the diagonal of the triangle -/
def area_below_diagonal (config : Configuration) : ℝ :=
  sorry

/-- The theorem stating the area is 72 square units -/
theorem area_is_72 (config : Configuration) 
  (h1 : config.square_side = 12)
  (h2 : config.square_lower_right = (12, 0))
  (h3 : config.triangle_base = 12)
  (h4 : config.hypotenuse_end = (24, 0)) :
  area_below_diagonal config = 72 :=
sorry

end area_is_72_l2051_205114


namespace monomial_count_l2051_205135

-- Define what a monomial is
def is_monomial (expr : String) : Bool := sorry

-- Define the set of expressions
def expressions : List String := ["(x+a)/2", "-2", "2x^2y", "b", "7x^2+8x-1"]

-- State the theorem
theorem monomial_count : 
  (expressions.filter is_monomial).length = 3 := by sorry

end monomial_count_l2051_205135


namespace inequality_solution_l2051_205102

theorem inequality_solution (x : ℝ) :
  x ≠ 3 →
  (x * (x + 1) / (x - 3)^2 ≥ 8 ↔ 3 < x ∧ x ≤ 24/7) :=
by sorry

end inequality_solution_l2051_205102


namespace max_value_quadratic_l2051_205195

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 3/2) :
  x * (2 - x) ≤ 1 :=
sorry

end max_value_quadratic_l2051_205195


namespace right_triangle_hypotenuse_l2051_205131

theorem right_triangle_hypotenuse : ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg > 0 →
  hypotenuse > 0 →
  long_leg = 2 * short_leg - 1 →
  (1 / 2) * short_leg * long_leg = 60 →
  short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2 →
  hypotenuse = 17 := by
sorry

end right_triangle_hypotenuse_l2051_205131


namespace roses_per_girl_l2051_205106

/-- Proves that each girl planted 3 roses given the conditions of the problem -/
theorem roses_per_girl (total_students : ℕ) (total_plants : ℕ) (birches : ℕ) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24)
  (h3 : birches = 6)
  (h4 : birches * 3 = total_students - (total_students - birches * 3)) :
  (total_plants - birches) / (total_students - birches * 3) = 3 := by
  sorry

end roses_per_girl_l2051_205106


namespace roots_of_equation_l2051_205182

theorem roots_of_equation (x : ℝ) : (x - 1)^2 = 1 ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_equation_l2051_205182


namespace quadratic_solution_property_l2051_205177

theorem quadratic_solution_property (k : ℚ) : 
  (∃ a b : ℚ, 
    (5 * a^2 + 7 * a + k = 0) ∧ 
    (5 * b^2 + 7 * b + k = 0) ∧ 
    (abs (a - b) = a^2 + b^2)) ↔ 
  (k = 21/25 ∨ k = -21/25) := by
sorry

end quadratic_solution_property_l2051_205177


namespace arithmetic_sequence_finite_negative_terms_l2051_205196

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def has_finite_negative_terms (a : ℕ → ℝ) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a n ≥ 0

theorem arithmetic_sequence_finite_negative_terms
  (a : ℕ → ℝ) (d : ℝ) (h1 : is_arithmetic_sequence a d)
  (h2 : a 1 < 0) (h3 : d > 0) :
  has_finite_negative_terms a :=
sorry

end arithmetic_sequence_finite_negative_terms_l2051_205196


namespace alice_numbers_l2051_205167

theorem alice_numbers : ∃ x y : ℝ, x * y = 12 ∧ x + y = 7 ∧ ({x, y} : Set ℝ) = {3, 4} := by
  sorry

end alice_numbers_l2051_205167


namespace largest_five_digit_with_given_product_l2051_205133

/-- The product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_given_product :
  (∀ n : ℕ, is_five_digit n ∧ digit_product n = 40320 → n ≤ 98752) ∧
  is_five_digit 98752 ∧
  digit_product 98752 = 40320 := by sorry

end largest_five_digit_with_given_product_l2051_205133


namespace arithmetic_sequence_sum_first_ten_l2051_205108

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_first_ten :
  arithmetic_sequence_sum (-3) 6 10 = 240 := by
  sorry

end arithmetic_sequence_sum_first_ten_l2051_205108


namespace fixed_point_on_line_unique_intersection_l2051_205168

-- Define the lines
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem 1: Line l passes through a fixed point
theorem fixed_point_on_line : ∀ k : ℝ, line k (-2) 1 := by sorry

-- Theorem 2: Unique intersection point when k = -3
theorem unique_intersection :
  ∃! k : ℝ, ∃! x y : ℝ, line k x y ∧ line1 x y ∧ line2 x y ∧ k = -3 := by sorry

end fixed_point_on_line_unique_intersection_l2051_205168


namespace length_width_difference_l2051_205104

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

theorem length_width_difference (r : Rectangle) 
  (h1 : perimeter r = 150)
  (h2 : r.length > r.width)
  (h3 : r.width = 45)
  (h4 : r.length = 60) :
  r.length - r.width = 15 := by sorry

end length_width_difference_l2051_205104


namespace geometric_progression_first_term_is_one_l2051_205100

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (a : ℝ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

/-- The product of any two terms in the progression is also a term in the progression. -/
def ProductIsInProgression (a : ℝ → ℝ) : Prop :=
  ∀ i j k : ℕ, ∃ k : ℕ, a i * a j = a k

/-- In a geometric progression where the product of any two terms is also a term in the progression,
    the first term of the progression must be 1. -/
theorem geometric_progression_first_term_is_one
  (a : ℝ → ℝ) (r : ℝ)
  (h1 : IsGeometricProgression a r)
  (h2 : ProductIsInProgression a) :
  a 0 = 1 := by
  sorry

end geometric_progression_first_term_is_one_l2051_205100


namespace min_shapes_for_square_l2051_205175

/-- The area of one shape in square units -/
def shape_area : ℕ := 3

/-- The side length of the square formed by the shapes -/
def square_side : ℕ := 6

/-- The area of the square formed by the shapes -/
def square_area : ℕ := square_side * square_side

/-- The number of shapes required to form the square -/
def num_shapes : ℕ := square_area / shape_area

theorem min_shapes_for_square : 
  ∀ n : ℕ, n < num_shapes → 
  ¬∃ s : ℕ, s * s = n * shape_area ∧ s % shape_area = 0 := by
  sorry

#eval num_shapes  -- Should output 12

end min_shapes_for_square_l2051_205175


namespace adams_clothing_ratio_l2051_205163

theorem adams_clothing_ratio :
  let initial_clothes : ℕ := 36
  let friend_count : ℕ := 3
  let total_donated : ℕ := 126
  let friends_donation := friend_count * initial_clothes
  let adams_kept := initial_clothes - (friends_donation + initial_clothes - total_donated)
  adams_kept = 0 ∧ initial_clothes ≠ 0 →
  (adams_kept : ℚ) / initial_clothes = 0 := by
sorry

end adams_clothing_ratio_l2051_205163


namespace sunday_necklace_production_l2051_205139

/-- The number of necklaces made by the first machine -/
def first_machine_necklaces : ℕ := 45

/-- The ratio of necklaces made by the second machine compared to the first -/
def second_machine_ratio : ℚ := 2.4

/-- The total number of necklaces made on Sunday -/
def total_necklaces : ℕ := 153

theorem sunday_necklace_production :
  (first_machine_necklaces : ℚ) + (first_machine_necklaces : ℚ) * second_machine_ratio = (total_necklaces : ℚ) := by
  sorry

end sunday_necklace_production_l2051_205139


namespace optimal_arrangement_l2051_205118

/-- Represents the housekeeping service company scenario -/
structure CleaningCompany where
  total_cleaners : ℕ
  large_rooms_per_cleaner : ℕ
  small_rooms_per_cleaner : ℕ
  large_room_payment : ℕ
  small_room_payment : ℕ

/-- Calculates the daily income based on the number of cleaners assigned to large rooms -/
def daily_income (company : CleaningCompany) (x : ℕ) : ℕ :=
  company.large_room_payment * company.large_rooms_per_cleaner * x +
  company.small_room_payment * company.small_rooms_per_cleaner * (company.total_cleaners - x)

/-- The main theorem to prove -/
theorem optimal_arrangement (company : CleaningCompany) (x : ℕ) :
  company.total_cleaners = 16 ∧
  company.large_rooms_per_cleaner = 4 ∧
  company.small_rooms_per_cleaner = 5 ∧
  company.large_room_payment = 80 ∧
  company.small_room_payment = 60 ∧
  x = 10 →
  daily_income company x = 5000 := by sorry

end optimal_arrangement_l2051_205118


namespace sum_of_squares_and_products_l2051_205125

theorem sum_of_squares_and_products (x y z : ℝ) 
  (nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (sum_of_squares : x^2 + y^2 + z^2 = 52) 
  (sum_of_products : x*y + y*z + z*x = 24) : 
  x + y + z = 10 := by sorry

end sum_of_squares_and_products_l2051_205125


namespace minimum_value_theorem_l2051_205170

theorem minimum_value_theorem (a b : ℝ) (h : a - 3*b + 6 = 0) :
  ∃ (m : ℝ), m = (1/4 : ℝ) ∧ ∀ x y : ℝ, x - 3*y + 6 = 0 → 2^x + (1/8)^y ≥ m :=
by sorry

end minimum_value_theorem_l2051_205170


namespace nicki_running_mileage_nicki_second_half_mileage_l2051_205153

/-- Calculates the weekly mileage for the second half of the year given the conditions -/
theorem nicki_running_mileage (total_weeks : ℕ) (first_half_weeks : ℕ) 
  (first_half_weekly_miles : ℕ) (total_annual_miles : ℕ) : ℕ :=
  let second_half_weeks := total_weeks - first_half_weeks
  let first_half_total_miles := first_half_weekly_miles * first_half_weeks
  let second_half_total_miles := total_annual_miles - first_half_total_miles
  second_half_total_miles / second_half_weeks

/-- Proves that Nicki ran 30 miles per week in the second half of the year -/
theorem nicki_second_half_mileage :
  nicki_running_mileage 52 26 20 1300 = 30 := by
  sorry

end nicki_running_mileage_nicki_second_half_mileage_l2051_205153


namespace parabola_shift_theorem_l2051_205127

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies a horizontal and vertical shift to a parabola --/
def shift_parabola (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 2 ∧ p.h = 1 ∧ p.k = 3 →
  let p' := shift_parabola p 2 (-1)
  p'.a = 2 ∧ p'.h = -1 ∧ p'.k = 2 := by
  sorry

end parabola_shift_theorem_l2051_205127


namespace no_integer_n_for_real_nth_power_of_complex_l2051_205107

theorem no_integer_n_for_real_nth_power_of_complex : 
  ¬ ∃ (n : ℤ), (Complex.I + n : ℂ)^5 ∈ Set.range Complex.ofReal := by sorry

end no_integer_n_for_real_nth_power_of_complex_l2051_205107


namespace definite_integral_semicircle_l2051_205165

theorem definite_integral_semicircle (f : ℝ → ℝ) (r : ℝ) :
  (∀ x, f x = Real.sqrt (r^2 - x^2)) →
  r > 0 →
  ∫ x in (0)..(r), f x = (π * r^2) / 4 := by
  sorry

end definite_integral_semicircle_l2051_205165


namespace no_solution_functional_equation_l2051_205145

theorem no_solution_functional_equation :
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + f y) = 2*x - f y :=
by sorry

end no_solution_functional_equation_l2051_205145


namespace not_pythagorean_triple_l2051_205178

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem not_pythagorean_triple : 
  (is_pythagorean_triple 3 4 5) ∧ 
  (is_pythagorean_triple 5 12 13) ∧ 
  (is_pythagorean_triple 6 8 10) ∧ 
  ¬(is_pythagorean_triple 7 25 26) := by
  sorry

end not_pythagorean_triple_l2051_205178


namespace james_beverages_consumed_l2051_205174

/-- Represents the number of beverages James drinks in a week. -/
def beverages_consumed_in_week (
  soda_packs : ℕ)
  (sodas_per_pack : ℕ)
  (juice_packs : ℕ)
  (juices_per_pack : ℕ)
  (water_packs : ℕ)
  (waters_per_pack : ℕ)
  (energy_drinks : ℕ)
  (initial_sodas : ℕ)
  (initial_juices : ℕ)
  (mon_wed_sodas : ℕ)
  (mon_wed_juices : ℕ)
  (mon_wed_waters : ℕ)
  (thu_sun_sodas : ℕ)
  (thu_sun_juices : ℕ)
  (thu_sun_waters : ℕ)
  (thu_sun_energy : ℕ) : ℕ :=
  3 * (mon_wed_sodas + mon_wed_juices + mon_wed_waters) +
  4 * (thu_sun_sodas + thu_sun_juices + thu_sun_waters + thu_sun_energy)

/-- Proves that James drinks exactly 50 beverages in a week given the conditions. -/
theorem james_beverages_consumed :
  beverages_consumed_in_week 4 10 3 8 2 15 7 12 5 3 2 1 2 4 1 1 = 50 := by
  sorry

end james_beverages_consumed_l2051_205174


namespace largest_triangle_perimeter_l2051_205190

theorem largest_triangle_perimeter (a b x : ℕ) : 
  a = 8 → b = 11 → x ∈ Set.Icc 4 18 → 
  (∀ y : ℕ, y ∈ Set.Icc 4 18 → a + b + y ≤ a + b + x) →
  a + b + x = 37 := by
sorry

end largest_triangle_perimeter_l2051_205190


namespace unique_number_l2051_205148

def is_valid_number (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- The first digit is 2
  (n / 100000 = 2) ∧
  -- Moving the first digit to the last position results in a number that is three times the original number
  (n % 100000 * 10 + 2 = 3 * n)

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 285714 :=
sorry

end unique_number_l2051_205148


namespace point_b_satisfies_inequality_l2051_205171

def satisfies_inequality (x y : ℝ) : Prop := x + 2 * y - 1 > 0

theorem point_b_satisfies_inequality :
  satisfies_inequality 0 1 ∧
  ¬ satisfies_inequality 1 (-1) ∧
  ¬ satisfies_inequality 1 0 ∧
  ¬ satisfies_inequality (-2) 0 :=
by sorry

end point_b_satisfies_inequality_l2051_205171


namespace prove_h_of_x_l2051_205187

/-- Given that 16x^4 + 5x^3 - 4x + 2 + h(x) = -8x^3 + 7x^2 - 6x + 5,
    prove that h(x) = -16x^4 - 13x^3 + 7x^2 - 2x + 3 -/
theorem prove_h_of_x (x : ℝ) (h : ℝ → ℝ) 
    (eq : 16 * x^4 + 5 * x^3 - 4 * x + 2 + h x = -8 * x^3 + 7 * x^2 - 6 * x + 5) : 
  h x = -16 * x^4 - 13 * x^3 + 7 * x^2 - 2 * x + 3 := by
  sorry

end prove_h_of_x_l2051_205187


namespace rectangular_plot_breadth_l2051_205130

theorem rectangular_plot_breadth : 
  ∀ (length breadth area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 867 →
  breadth = 17 := by
sorry

end rectangular_plot_breadth_l2051_205130


namespace smallest_square_side_length_l2051_205199

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The problem statement -/
theorem smallest_square_side_length 
  (rect1 : Rectangle)
  (rect2 : Rectangle)
  (h1 : rect1.width = 2 ∧ rect1.height = 4)
  (h2 : rect2.width = 4 ∧ rect2.height = 5)
  (h3 : ∀ (s : ℝ), s ≥ 0 → 
    (∃ (x1 y1 x2 y2 : ℝ), 
      0 ≤ x1 ∧ x1 + rect1.width ≤ s ∧
      0 ≤ y1 ∧ y1 + rect1.height ≤ s ∧
      0 ≤ x2 ∧ x2 + rect2.width ≤ s ∧
      0 ≤ y2 ∧ y2 + rect2.height ≤ s ∧
      (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
       y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1))) :
  (∀ (s : ℝ), s ≥ 0 ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      0 ≤ x1 ∧ x1 + rect1.width ≤ s ∧
      0 ≤ y1 ∧ y1 + rect1.height ≤ s ∧
      0 ≤ x2 ∧ x2 + rect2.width ≤ s ∧
      0 ≤ y2 ∧ y2 + rect2.height ≤ s ∧
      (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
       y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1)) → s ≥ 6) ∧
  (∃ (x1 y1 x2 y2 : ℝ), 
    0 ≤ x1 ∧ x1 + rect1.width ≤ 6 ∧
    0 ≤ y1 ∧ y1 + rect1.height ≤ 6 ∧
    0 ≤ x2 ∧ x2 + rect2.width ≤ 6 ∧
    0 ≤ y2 ∧ y2 + rect2.height ≤ 6 ∧
    (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
     y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1)) := by
  sorry

end smallest_square_side_length_l2051_205199


namespace percentage_difference_l2051_205197

theorem percentage_difference (A B x : ℝ) : 
  A > B ∧ B > 0 → A = B * (1 + x / 100) → x = 100 * (A - B) / B := by
  sorry

end percentage_difference_l2051_205197


namespace leftover_value_is_zero_l2051_205169

/-- Represents the number of coins in a roll -/
def roll_size : ℕ := 40

/-- Represents Michael's coin counts -/
def michael_quarters : ℕ := 75
def michael_nickels : ℕ := 123

/-- Represents Sarah's coin counts -/
def sarah_quarters : ℕ := 85
def sarah_nickels : ℕ := 157

/-- Calculates the total number of quarters -/
def total_quarters : ℕ := michael_quarters + sarah_quarters

/-- Calculates the total number of nickels -/
def total_nickels : ℕ := michael_nickels + sarah_nickels

/-- Calculates the number of leftover quarters -/
def leftover_quarters : ℕ := total_quarters % roll_size

/-- Calculates the number of leftover nickels -/
def leftover_nickels : ℕ := total_nickels % roll_size

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the total value of leftover coins in cents -/
def leftover_value : ℕ := leftover_quarters * quarter_value + leftover_nickels * nickel_value

/-- Theorem stating that the value of leftover coins is $0.00 -/
theorem leftover_value_is_zero : leftover_value = 0 := by sorry

end leftover_value_is_zero_l2051_205169


namespace target_shopping_expense_l2051_205158

/-- The total amount spent by Christy and Tanya at Target -/
def total_spent (tanya_face_moisturizer_price : ℕ) 
                (tanya_face_moisturizer_count : ℕ)
                (tanya_body_lotion_price : ℕ)
                (tanya_body_lotion_count : ℕ) : ℕ :=
  let tanya_total := tanya_face_moisturizer_price * tanya_face_moisturizer_count + 
                     tanya_body_lotion_price * tanya_body_lotion_count
  tanya_total * 3

theorem target_shopping_expense :
  total_spent 50 2 60 4 = 1020 :=
sorry

end target_shopping_expense_l2051_205158


namespace quadratic_composite_zeros_l2051_205117

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The function f(x) -/
def f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The composite function f(f(x)) -/
def f_comp_f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  f q (f q x)

/-- The number of distinct real zeros of a function -/
def num_distinct_real_zeros (g : ℝ → ℝ) : ℕ := sorry

theorem quadratic_composite_zeros
  (q : QuadraticFunction)
  (h : f q (1 / q.a) < 0) :
  num_distinct_real_zeros (f_comp_f q) = 4 :=
sorry

end quadratic_composite_zeros_l2051_205117


namespace hammond_statue_weight_l2051_205162

/-- Given Hammond's marble carving scenario, prove the weight of each remaining statue. -/
theorem hammond_statue_weight :
  -- Total weight of marble block
  let total_weight : ℕ := 80
  -- Weight of first statue
  let first_statue : ℕ := 10
  -- Weight of second statue
  let second_statue : ℕ := 18
  -- Weight of discarded marble
  let discarded : ℕ := 22
  -- Number of statues
  let num_statues : ℕ := 4
  -- Weight of each remaining statue
  let remaining_statue_weight : ℕ := (total_weight - first_statue - second_statue - discarded) / (num_statues - 2)
  -- Proof that each remaining statue weighs 15 pounds
  remaining_statue_weight = 15 := by
  sorry

end hammond_statue_weight_l2051_205162


namespace course_choice_theorem_l2051_205126

/-- The number of ways to choose courses for 5 students -/
def course_choice_ways : ℕ := 20

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of courses -/
def num_courses : ℕ := 2

/-- The minimum number of students required for each course -/
def min_students_per_course : ℕ := 2

theorem course_choice_theorem :
  ∀ (ways : ℕ),
  ways = course_choice_ways →
  ways = (num_students.choose min_students_per_course) * num_courses.factorial :=
by sorry

end course_choice_theorem_l2051_205126


namespace triangle_inequality_l2051_205157

theorem triangle_inequality (a b c p q r : ℝ) 
  (triangle_cond : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (sum_zero : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
  sorry

end triangle_inequality_l2051_205157


namespace arithmetic_sequence_15th_term_l2051_205191

-- Define the sequence terms
def term (k : ℕ) (A B : ℝ) : ℝ := (4 + 3 * (k - 1)) * A + (5 + 3 * (k - 1)) * B

-- State the theorem
theorem arithmetic_sequence_15th_term (a b : ℝ) (A B : ℝ) (h1 : A = Real.log a) (h2 : B = Real.log b) :
  (∀ k : ℕ, k ≥ 1 → k ≤ 3 → term k A B = Real.log (a^(4 + 3*(k-1)) * b^(5 + 3*(k-1)))) →
  term 15 A B = Real.log (b^93) := by
  sorry


end arithmetic_sequence_15th_term_l2051_205191


namespace women_average_age_l2051_205140

theorem women_average_age (n : ℕ) (A : ℝ) (age1 age2 : ℕ) :
  n = 10 ∧ 
  age1 = 10 ∧ 
  age2 = 12 ∧ 
  (n * A - age1 - age2 + 2 * ((n * (A + 2)) - (n * A - age1 - age2))) / 2 = 21 :=
by sorry

end women_average_age_l2051_205140


namespace max_player_score_l2051_205155

theorem max_player_score (total_players : ℕ) (total_points : ℕ) (min_points : ℕ) 
  (h1 : total_players = 12)
  (h2 : total_points = 100)
  (h3 : min_points = 7)
  (h4 : ∀ player, player ≥ min_points) :
  ∃ max_score : ℕ, max_score = 23 ∧ 
  (∀ player_score : ℕ, player_score ≤ max_score) ∧
  (∃ player : ℕ, player = max_score) ∧
  (total_points = (total_players - 1) * min_points + max_score) :=
by sorry

end max_player_score_l2051_205155


namespace fraction_expression_value_l2051_205101

theorem fraction_expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, 18 / 7 + x / (2 * q + p) = 3 ∧ x = 6 := by
  sorry

end fraction_expression_value_l2051_205101


namespace nadia_walked_18_km_l2051_205113

/-- The distance Hannah walked in kilometers -/
def hannah_distance : ℝ := sorry

/-- The distance Nadia walked in kilometers -/
def nadia_distance : ℝ := 2 * hannah_distance

/-- The total distance walked by both girls in kilometers -/
def total_distance : ℝ := 27

theorem nadia_walked_18_km :
  nadia_distance = 18 ∧ hannah_distance + nadia_distance = total_distance :=
by sorry

end nadia_walked_18_km_l2051_205113


namespace equation_solution_l2051_205123

theorem equation_solution (x : ℝ) (h : x ≠ -2/3) :
  (3*x + 2) / (3*x^2 - 7*x - 6) = (2*x + 1) / (3*x - 2) ↔
  x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6 :=
by sorry

end equation_solution_l2051_205123


namespace max_earnings_ali_baba_l2051_205119

/-- Represents the weight of the bag when filled with only diamonds -/
def diamond_full_weight : ℝ := 40

/-- Represents the weight of the bag when filled with only gold -/
def gold_full_weight : ℝ := 200

/-- Represents the maximum weight Ali Baba can carry -/
def max_carry_weight : ℝ := 100

/-- Represents the cost of 1 kg of diamonds in dinars -/
def diamond_cost : ℝ := 60

/-- Represents the cost of 1 kg of gold in dinars -/
def gold_cost : ℝ := 20

/-- Represents the objective function to maximize -/
def objective_function (x y : ℝ) : ℝ := diamond_cost * x + gold_cost * y

/-- Theorem stating that the maximum value of the objective function
    under given constraints is 3000 dinars -/
theorem max_earnings_ali_baba :
  ∃ x y : ℝ,
    x ≥ 0 ∧
    y ≥ 0 ∧
    x + y ≤ max_carry_weight ∧
    (x / diamond_full_weight + y / gold_full_weight) ≤ 1 ∧
    objective_function x y = 3000 ∧
    ∀ x' y' : ℝ,
      x' ≥ 0 →
      y' ≥ 0 →
      x' + y' ≤ max_carry_weight →
      (x' / diamond_full_weight + y' / gold_full_weight) ≤ 1 →
      objective_function x' y' ≤ 3000 :=
by sorry

end max_earnings_ali_baba_l2051_205119


namespace rectangle_area_l2051_205152

theorem rectangle_area (square_area : Real) (rectangle_width : Real) (rectangle_length : Real) :
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
  sorry

end rectangle_area_l2051_205152


namespace max_distance_with_specific_tires_l2051_205137

/-- Represents the maximum distance a car can travel with tire switching -/
def maxDistanceWithTireSwitching (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  min frontTireLife rearTireLife

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_with_specific_tires :
  maxDistanceWithTireSwitching 42000 56000 = 42000 := by
  sorry

#check max_distance_with_specific_tires

end max_distance_with_specific_tires_l2051_205137


namespace intersection_implies_a_equals_one_l2051_205154

theorem intersection_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 := by
  sorry

end intersection_implies_a_equals_one_l2051_205154


namespace triangle_circumradius_l2051_205151

/-- Given a triangle ABC where:
  * a, b, c are sides opposite to angles A, B, C respectively
  * a = 2
  * b = 3
  * cos C = 1/3
  Then the radius of the circumcircle is 9√2/8 -/
theorem triangle_circumradius (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = 3 →
  c = (a^2 + b^2 - 2*a*b*(1/3))^(1/2) →
  let r := c / (2 * (1 - (1/3)^2)^(1/2))
  r = 9 * (2^(1/2)) / 8 := by
sorry

end triangle_circumradius_l2051_205151


namespace triangle_side_length_l2051_205185

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →  -- Area condition
  B = π/3 →                                   -- Angle B = 60°
  a^2 + c^2 = 3*a*c →                         -- Given equation
  b = 2 * Real.sqrt 2 := by                   -- Conclusion
sorry


end triangle_side_length_l2051_205185


namespace original_profit_margin_l2051_205161

theorem original_profit_margin
  (original_price : ℝ)
  (original_margin : ℝ)
  (h_price_decrease : ℝ → ℝ → Prop)
  (h_margin_increase : ℝ → ℝ → Prop) :
  h_price_decrease original_price (original_price * (1 - 0.064)) →
  h_margin_increase original_margin (original_margin + 0.08) →
  original_margin = 0.17 :=
by sorry

end original_profit_margin_l2051_205161


namespace average_speed_of_trip_l2051_205103

/-- Proves that the average speed of a trip is 16 km/h given the specified conditions -/
theorem average_speed_of_trip (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
    (second_part_speed : ℝ) (h1 : total_distance = 400) (h2 : first_part_distance = 100) 
    (h3 : first_part_speed = 20) (h4 : second_part_speed = 15) : 
    total_distance / (first_part_distance / first_part_speed + 
    (total_distance - first_part_distance) / second_part_speed) = 16 := by
  sorry

end average_speed_of_trip_l2051_205103


namespace geometric_sequence_fourth_term_l2051_205132

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = (a n : ℚ) * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℕ)
  (h_geometric : is_geometric_sequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 := by
sorry

end geometric_sequence_fourth_term_l2051_205132


namespace limit_equals_one_implies_a_and_b_l2051_205149

/-- Given that a and b are constants such that the limit of (ln(2-x))^2 / (x^2 + ax + b) as x approaches 1 is equal to 1, prove that a = -2 and b = 1. -/
theorem limit_equals_one_implies_a_and_b (a b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((Real.log (2 - x))^2) / (x^2 + a*x + b) - 1| < ε) →
  a = -2 ∧ b = 1 := by
  sorry

end limit_equals_one_implies_a_and_b_l2051_205149


namespace sum_of_phi_plus_one_divisors_l2051_205144

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- A divisor of n is a natural number that divides n without a remainder -/
def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem sum_of_phi_plus_one_divisors (n : ℕ) :
  ∃ (divisors : Finset ℕ), 
    (∀ d ∈ divisors, is_divisor d n) ∧ 
    (Finset.card divisors = phi n + 1) ∧
    (Finset.sum divisors id = n) :=
  sorry

end sum_of_phi_plus_one_divisors_l2051_205144


namespace smallest_dimension_is_eight_l2051_205176

/-- Represents a rectangular crate with dimensions a, b, and c. -/
structure Crate where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a right circular cylinder. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate. -/
def cylinderFitsInCrate (cyl : Cylinder) (cr : Crate) : Prop :=
  2 * cyl.radius ≤ cr.a ∧ 2 * cyl.radius ≤ cr.b ∧ cyl.height ≤ cr.c ∨
  2 * cyl.radius ≤ cr.a ∧ 2 * cyl.radius ≤ cr.c ∧ cyl.height ≤ cr.b ∨
  2 * cyl.radius ≤ cr.b ∧ 2 * cyl.radius ≤ cr.c ∧ cyl.height ≤ cr.a

/-- The main theorem stating that the smallest dimension of the crate is 8 feet. -/
theorem smallest_dimension_is_eight
  (cr : Crate)
  (h1 : cr.b = 8)
  (h2 : cr.c = 12)
  (h3 : ∃ (cyl : Cylinder), cyl.radius = 7 ∧ cylinderFitsInCrate cyl cr) :
  min cr.a (min cr.b cr.c) = 8 := by
  sorry


end smallest_dimension_is_eight_l2051_205176


namespace line_intersects_ellipse_l2051_205147

-- Define the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 20 = 1

-- Theorem statement
theorem line_intersects_ellipse (k : ℝ) :
  ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end line_intersects_ellipse_l2051_205147


namespace rowans_rate_l2051_205189

/-- Rowan's rowing problem -/
theorem rowans_rate (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ)
  (h1 : downstream_distance = 26)
  (h2 : downstream_time = 2)
  (h3 : upstream_time = 4)
  (h4 : downstream_time > 0)
  (h5 : upstream_time > 0) :
  ∃ (still_water_rate : ℝ) (current_rate : ℝ),
    still_water_rate = 9.75 ∧
    (still_water_rate + current_rate) * downstream_time = downstream_distance ∧
    (still_water_rate - current_rate) * upstream_time = downstream_distance :=
by
  sorry


end rowans_rate_l2051_205189
