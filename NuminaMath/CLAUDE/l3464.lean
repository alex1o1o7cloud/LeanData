import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_ratio_l3464_346419

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  c / d = 9/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l3464_346419


namespace NUMINAMATH_CALUDE_henry_collection_cost_l3464_346484

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total_needed : ℕ) (cost_per_figure : ℕ) : ℕ :=
  (total_needed - current) * cost_per_figure

/-- Proof that Henry needs $30 to finish his collection -/
theorem henry_collection_cost : money_needed 3 8 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_henry_collection_cost_l3464_346484


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3464_346464

-- Define the polynomial
def p (x : ℝ) : ℝ := 3 * (x^8 - 2*x^5 + 4*x^3 - 6) - 5 * (x^4 - 3*x^2 + 2) + 2 * (x^6 + 5*x - 8)

-- Theorem: The sum of the coefficients of p is -3
theorem sum_of_coefficients : p 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3464_346464


namespace NUMINAMATH_CALUDE_paco_cookies_proof_l3464_346416

/-- The number of cookies Paco had initially -/
def initial_cookies : ℕ := 2

theorem paco_cookies_proof :
  (∃ (x : ℕ), 
    (x - 2 + 36 = 2 + 34) ∧ 
    (x = initial_cookies)) := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_proof_l3464_346416


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l3464_346467

theorem repeating_decimal_division (a b : ℚ) :
  a = 81 / 99 → b = 36 / 99 → a / b = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l3464_346467


namespace NUMINAMATH_CALUDE_mango_purchase_problem_l3464_346495

/-- The problem of calculating the amount of mangoes purchased --/
theorem mango_purchase_problem (grape_kg : ℕ) (grape_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grape_kg = 8 →
  grape_rate = 80 →
  mango_rate = 55 →
  total_paid = 1135 →
  ∃ (mango_kg : ℕ), mango_kg * mango_rate + grape_kg * grape_rate = total_paid ∧ mango_kg = 9 :=
by sorry

end NUMINAMATH_CALUDE_mango_purchase_problem_l3464_346495


namespace NUMINAMATH_CALUDE_trapezoid_area_l3464_346422

/-- The area of a trapezoid bounded by y=x, y=-x, x=10, and y=10 is 150 square units. -/
theorem trapezoid_area : Real := by
  -- Define the lines bounding the trapezoid
  let line1 : Real → Real := λ x => x
  let line2 : Real → Real := λ x => -x
  let line3 : Real → Real := λ _ => 10
  let vertical_line : Real := 10

  -- Define the trapezoid
  let trapezoid := {(x, y) : Real × Real | 
    (y = line1 x ∨ y = line2 x ∨ y = line3 x) ∧ 
    x ≤ vertical_line ∧ 
    y ≤ line3 x}

  -- Calculate the area of the trapezoid
  let area : Real := 150

  sorry -- Proof goes here

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l3464_346422


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3464_346471

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 1 * a 99 = 21 ∧ a 1 + a 99 = 10) :
  a 3 + a 97 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3464_346471


namespace NUMINAMATH_CALUDE_triangle_third_side_l3464_346448

theorem triangle_third_side (a b c : ℝ) : 
  a = 3 → b = 10 → c > 0 → 
  a + b + c = 6 * (⌊(a + b + c) / 6⌋ : ℝ) →
  c = 11 := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3464_346448


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3464_346479

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposites -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_sum (a b : ℝ) :
  symmetric_about_x_axis (a, 4) (-2, b) → a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3464_346479


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3464_346407

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3464_346407


namespace NUMINAMATH_CALUDE_paper_envelope_problem_l3464_346465

/-- 
Given that each paper envelope can contain 10 white papers and 12 paper envelopes are needed,
prove that the total number of clean white papers is 120.
-/
theorem paper_envelope_problem (papers_per_envelope : ℕ) (num_envelopes : ℕ) 
  (h1 : papers_per_envelope = 10) 
  (h2 : num_envelopes = 12) : 
  papers_per_envelope * num_envelopes = 120 := by
  sorry

end NUMINAMATH_CALUDE_paper_envelope_problem_l3464_346465


namespace NUMINAMATH_CALUDE_maximize_product_l3464_346439

theorem maximize_product (A : ℝ) (h : A > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = A ∧
    ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = A →
      x * y^2 * z^3 ≤ a * b^2 * c^3 ∧
    a = A / 6 ∧ b = A / 3 ∧ c = A / 2 :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l3464_346439


namespace NUMINAMATH_CALUDE_total_cost_is_two_l3464_346461

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 1/10

/-- The cost of 3 pencils and 4 pens in dollars -/
def cost_3p4p : ℚ := 79/50

/-- The cost of a pen in dollars -/
def pen_cost : ℚ := (cost_3p4p - 3 * pencil_cost) / 4

/-- The total cost of 4 pencils and 5 pens in dollars -/
def total_cost : ℚ := 4 * pencil_cost + 5 * pen_cost

theorem total_cost_is_two : total_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_two_l3464_346461


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l3464_346494

/-- Given a cube root of unity ω, prove the equality for any complex numbers a, b, c -/
theorem cube_root_unity_product (ω : ℂ) (a b c : ℂ) 
  (h1 : ω^3 = 1) 
  (h2 : 1 + ω + ω^2 = 0) : 
  (a + b*ω + c*ω^2) * (a + b*ω^2 + c*ω) = a^2 + b^2 + c^2 - a*b - b*c - c*a := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l3464_346494


namespace NUMINAMATH_CALUDE_sum_x_coordinates_of_Q3_l3464_346483

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon from the midpoints of another polygon's sides -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- The sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

theorem sum_x_coordinates_of_Q3 (Q1 : Polygon) 
  (h1 : Q1.vertices.length = 45)
  (h2 : sumXCoordinates Q1 = 180) :
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumXCoordinates Q3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_of_Q3_l3464_346483


namespace NUMINAMATH_CALUDE_baseball_team_ratio_l3464_346445

def baseball_ratio (games_played : ℕ) (games_won : ℕ) : ℚ := 
  games_played / (games_played - games_won)

theorem baseball_team_ratio : 
  let games_played := 10
  let games_won := 5
  baseball_ratio games_played games_won = 2 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_ratio_l3464_346445


namespace NUMINAMATH_CALUDE_potato_cost_is_correct_l3464_346455

/-- The cost of one bag of potatoes from the farmer in rubles -/
def potato_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 100

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 60

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 40

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The additional profit Boris made compared to Andrey in rubles -/
def additional_profit : ℝ := 1200

theorem potato_cost_is_correct : 
  potato_cost * bags_bought * (1 + boris_first_increase / 100) * boris_first_sale +
  potato_cost * bags_bought * (1 + boris_first_increase / 100) * (1 + boris_second_increase / 100) * boris_second_sale -
  potato_cost * bags_bought * (1 + andrey_increase / 100) * bags_bought = additional_profit := by
  sorry

end NUMINAMATH_CALUDE_potato_cost_is_correct_l3464_346455


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l3464_346429

theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (2, 0) →
  B = (0, 4) →
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ↔
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
      x = 2 * (1 - t) + 0 * t ∧
      y = 0 * (1 - t) + 4 * t :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l3464_346429


namespace NUMINAMATH_CALUDE_expression_equivalence_l3464_346430

variables (x y : ℝ)

def P : ℝ := 2*x + 3*y
def Q : ℝ := x - 2*y

theorem expression_equivalence :
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (2*x + 3*y) / (2*x + 10*y) :=
by sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3464_346430


namespace NUMINAMATH_CALUDE_prob_green_is_one_sixth_adding_two_green_balls_makes_prob_one_fourth_l3464_346474

/-- Represents the contents of the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (bag : BagContents) : ℕ :=
  bag.red + bag.yellow + bag.green

/-- Calculates the probability of drawing a green ball -/
def probGreen (bag : BagContents) : ℚ :=
  bag.green / (totalBalls bag)

/-- The initial bag contents -/
def initialBag : BagContents :=
  { red := 6, yellow := 9, green := 3 }

/-- Theorem stating the probability of drawing a green ball is 1/6 -/
theorem prob_green_is_one_sixth :
  probGreen initialBag = 1/6 := by sorry

/-- Adds green balls to the bag -/
def addGreenBalls (bag : BagContents) (n : ℕ) : BagContents :=
  { bag with green := bag.green + n }

/-- Theorem stating that adding 2 green balls makes the probability 1/4 -/
theorem adding_two_green_balls_makes_prob_one_fourth :
  probGreen (addGreenBalls initialBag 2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_prob_green_is_one_sixth_adding_two_green_balls_makes_prob_one_fourth_l3464_346474


namespace NUMINAMATH_CALUDE_pencil_sale_problem_l3464_346489

theorem pencil_sale_problem (total_students : ℕ) (total_pencils : ℕ) 
  (h_total_students : total_students = 10)
  (h_total_pencils : total_pencils = 24)
  (h_first_two : 2 * 2 = 4)  -- First two students bought 2 pencils each
  (h_last_two : 2 * 1 = 2)   -- Last two students bought 1 pencil each
  : ∃ (middle_group : ℕ), 
    middle_group = 6 ∧ 
    middle_group * 3 + 4 + 2 = total_pencils ∧ 
    2 + middle_group + 2 = total_students :=
by sorry

end NUMINAMATH_CALUDE_pencil_sale_problem_l3464_346489


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l3464_346466

/-- Given a configuration of a large semicircle, a circle, and a small semicircle that are all
    pairwise tangent, this theorem proves that the radius of the smaller semicircle is 4 when
    the radius of the large semicircle is 12 and the radius of the circle is 6. -/
theorem small_semicircle_radius
  (R : ℝ) -- Radius of the large semicircle
  (r : ℝ) -- Radius of the circle
  (x : ℝ) -- Radius of the small semicircle
  (h1 : R = 12) -- Given radius of large semicircle
  (h2 : r = 6)  -- Given radius of circle
  (h3 : R > 0 ∧ r > 0 ∧ x > 0) -- All radii are positive
  (h4 : R > r ∧ R > x) -- Large semicircle is the largest
  (h5 : (R - x)^2 + r^2 = (r + x)^2) -- Pythagorean theorem for tangent circles
  : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l3464_346466


namespace NUMINAMATH_CALUDE_probability_white_or_blue_is_half_l3464_346487

/-- Represents the number of marbles of each color in the basket -/
structure MarbleBasket where
  red : ℕ
  white : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of marbles in the basket -/
def totalMarbles (basket : MarbleBasket) : ℕ :=
  basket.red + basket.white + basket.green + basket.blue

/-- Calculates the number of white and blue marbles in the basket -/
def whiteAndBlueMarbles (basket : MarbleBasket) : ℕ :=
  basket.white + basket.blue

/-- The probability of picking a white or blue marble from the basket -/
def probabilityWhiteOrBlue (basket : MarbleBasket) : ℚ :=
  whiteAndBlueMarbles basket / totalMarbles basket

theorem probability_white_or_blue_is_half :
  let basket : MarbleBasket := ⟨4, 3, 9, 10⟩
  probabilityWhiteOrBlue basket = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_or_blue_is_half_l3464_346487


namespace NUMINAMATH_CALUDE_billboard_area_l3464_346476

/-- The area of a rectangular billboard with perimeter 44 feet and width 9 feet is 117 square feet. -/
theorem billboard_area (perimeter width : ℝ) (h1 : perimeter = 44) (h2 : width = 9) :
  let length := (perimeter - 2 * width) / 2
  width * length = 117 :=
by sorry

end NUMINAMATH_CALUDE_billboard_area_l3464_346476


namespace NUMINAMATH_CALUDE_fourth_term_is_two_l3464_346485

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q
  a6_eq_2 : a 6 = 2
  arithmetic_subseq : a 7 - a 5 = a 9 - a 7

/-- The fourth term of the geometric sequence is 2 -/
theorem fourth_term_is_two (seq : GeometricSequence) : seq.a 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_two_l3464_346485


namespace NUMINAMATH_CALUDE_island_closed_path_theorem_l3464_346457

/-- Represents a rectangular county with a diagonal road --/
structure County where
  has_diagonal_road : Bool

/-- Represents a rectangular island composed of counties --/
structure Island where
  counties : List County
  is_rectangular : Bool

/-- Checks if the roads in the counties form a closed path without self-intersection --/
def forms_closed_path (island : Island) : Bool := sorry

/-- Theorem stating that a rectangular island with an odd number of counties can form a closed path
    if and only if it has at least 9 counties --/
theorem island_closed_path_theorem (island : Island) :
  island.is_rectangular ∧ 
  island.counties.length % 2 = 1 ∧
  island.counties.length ≥ 9 ∧
  (∀ c ∈ island.counties, c.has_diagonal_road) →
  forms_closed_path island :=
sorry

end NUMINAMATH_CALUDE_island_closed_path_theorem_l3464_346457


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3464_346402

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 4/9) (h2 : x - y = 2/9) : x^2 - y^2 = 8/81 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3464_346402


namespace NUMINAMATH_CALUDE_tennis_ball_cost_l3464_346415

theorem tennis_ball_cost (num_packs : ℕ) (total_cost : ℚ) (balls_per_pack : ℕ) 
  (h1 : num_packs = 4)
  (h2 : total_cost = 24)
  (h3 : balls_per_pack = 3) :
  total_cost / (num_packs * balls_per_pack) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_cost_l3464_346415


namespace NUMINAMATH_CALUDE_value_of_a_l3464_346421

theorem value_of_a (a : ℝ) : 4 ∈ ({a^2 - 3*a, a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3464_346421


namespace NUMINAMATH_CALUDE_gcd_1248_585_l3464_346454

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1248_585_l3464_346454


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3464_346418

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 5 * a 14 = 5 → a 8 * a 9 * a 10 * a 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3464_346418


namespace NUMINAMATH_CALUDE_max_sum_prism_with_pyramid_l3464_346481

/-- Represents a triangular prism --/
structure TriangularPrism :=
  (faces : Nat)
  (edges : Nat)
  (vertices : Nat)

/-- Represents the result of adding a pyramid to a face of a prism --/
structure PrismWithPyramid :=
  (faces : Nat)
  (edges : Nat)
  (vertices : Nat)

/-- Calculates the sum of faces, edges, and vertices --/
def sumElements (shape : PrismWithPyramid) : Nat :=
  shape.faces + shape.edges + shape.vertices

/-- Adds a pyramid to a triangular face of the prism --/
def addPyramidToTriangularFace (prism : TriangularPrism) : PrismWithPyramid :=
  { faces := prism.faces - 1 + 3,
    edges := prism.edges + 3,
    vertices := prism.vertices + 1 }

/-- Adds a pyramid to a quadrilateral face of the prism --/
def addPyramidToQuadrilateralFace (prism : TriangularPrism) : PrismWithPyramid :=
  { faces := prism.faces - 1 + 4,
    edges := prism.edges + 4,
    vertices := prism.vertices + 1 }

/-- The main theorem to be proved --/
theorem max_sum_prism_with_pyramid :
  let prism := TriangularPrism.mk 5 9 6
  let triangularResult := addPyramidToTriangularFace prism
  let quadrilateralResult := addPyramidToQuadrilateralFace prism
  max (sumElements triangularResult) (sumElements quadrilateralResult) = 28 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_prism_with_pyramid_l3464_346481


namespace NUMINAMATH_CALUDE_problem_solution_l3464_346459

theorem problem_solution : 18 * 36 + 45 * 18 - 9 * 18 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3464_346459


namespace NUMINAMATH_CALUDE_discount_fraction_proof_l3464_346460

/-- Given a purchase of two items with the following conditions:
  1. Each item's full price is $60.
  2. The total spent on both items is $90.
  3. The first item is bought at full price.
  4. The second item is discounted by a certain fraction.

  Prove that the discount fraction on the second item is 1/2. -/
theorem discount_fraction_proof (full_price : ℝ) (total_spent : ℝ) (discount_fraction : ℝ) :
  full_price = 60 →
  total_spent = 90 →
  total_spent = full_price + (1 - discount_fraction) * full_price →
  discount_fraction = (1 : ℝ) / 2 := by
  sorry

#check discount_fraction_proof

end NUMINAMATH_CALUDE_discount_fraction_proof_l3464_346460


namespace NUMINAMATH_CALUDE_ice_cream_probability_l3464_346446

def probability_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem ice_cream_probability : 
  probability_exactly_k_successes 7 3 (3/4) = 945/16384 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_probability_l3464_346446


namespace NUMINAMATH_CALUDE_triangle_tangent_inequality_l3464_346473

/-- Given a triangle ABC with sides a, b, c and tangential segments x, y, z
    from vertices A, B, C to the incircle respectively, if a ≥ b ≥ c,
    then az + by + cx ≥ (a² + b² + c²)/2 ≥ ax + by + cz. -/
theorem triangle_tangent_inequality (a b c x y z : ℝ) 
    (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a = y + z) (h4 : b = x + z) (h5 : c = x + y) :
    a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2 ∧ 
    (a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z := by
  sorry


end NUMINAMATH_CALUDE_triangle_tangent_inequality_l3464_346473


namespace NUMINAMATH_CALUDE_product_real_parts_of_complex_equation_l3464_346470

theorem product_real_parts_of_complex_equation : ∃ (x₁ x₂ : ℂ),
  (x₁^2 - 4*x₁ = -4 - 4*I) ∧
  (x₂^2 - 4*x₂ = -4 - 4*I) ∧
  (x₁ ≠ x₂) ∧
  (Complex.re x₁ * Complex.re x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_product_real_parts_of_complex_equation_l3464_346470


namespace NUMINAMATH_CALUDE_right_column_sum_equals_twenty_l3464_346472

/-- Represents a 3x3 grid of numbers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Check if a grid contains only numbers from 1 to 9 without repetition -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j, g i j ∈ Finset.range 9 ∧ 
  ∀ i j i' j', g i j = g i' j' → i = i' ∧ j = j'

/-- Sum of the bottom row -/
def bottomRowSum (g : Grid) : ℕ :=
  g 2 0 + g 2 1 + g 2 2

/-- Sum of the rightmost column -/
def rightColumnSum (g : Grid) : ℕ :=
  g 0 2 + g 1 2 + g 2 2

theorem right_column_sum_equals_twenty (g : Grid) 
  (hValid : isValidGrid g) 
  (hBottomSum : bottomRowSum g = 20) 
  (hCorner : g 2 2 = 7) : 
  rightColumnSum g = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_column_sum_equals_twenty_l3464_346472


namespace NUMINAMATH_CALUDE_star_value_l3464_346412

-- Define the operation *
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 12) (prod_eq : a * b = 32) : 
  star a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l3464_346412


namespace NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l3464_346417

/-- A polygon with exterior angles measuring 45° has 8 sides. -/
theorem polygon_with_45_degree_exterior_angles_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 45 →
    (n : ℝ) * exterior_angle = 360 →
    n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l3464_346417


namespace NUMINAMATH_CALUDE_factorization_equality_l3464_346463

theorem factorization_equality (x : ℝ) :
  (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3464_346463


namespace NUMINAMATH_CALUDE_triangle_folding_angle_range_l3464_346414

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  valid : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def pointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define perpendicularity of two line segments
def perpendicular (AB CD : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem triangle_folding_angle_range 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_angle_C : angle (B - C) (A - C) = π / 3) 
  (θ : ℝ) 
  (h_angle_BAC : angle (C - A) (B - A) = θ) :
  (∃ M : ℝ × ℝ, 
    pointOnSegment M B C ∧ 
    (∃ B' : ℝ × ℝ, perpendicular (A, B') (C, M))) →
  π / 6 < θ ∧ θ < 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_folding_angle_range_l3464_346414


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l3464_346433

/-- The number of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def pascal_triangle_ones (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probability_of_one (n : ℕ) : ℚ :=
  (pascal_triangle_ones n : ℚ) / (pascal_triangle_elements n : ℚ)

theorem probability_of_one_in_20_rows :
  probability_of_one 20 = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l3464_346433


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3464_346492

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  (q ≠ 1 ∧ q ≠ -1) →                -- Common ratio not equal to ±1
  (a 1 = 1) →                       -- First term is 1
  (a m = a 1 * a 2 * a 3 * a 4 * a 5) →  -- Condition given in the problem
  m = 11 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3464_346492


namespace NUMINAMATH_CALUDE_stamp_collection_increase_l3464_346447

theorem stamp_collection_increase (initial_stamps final_stamps : ℕ) 
  (h1 : initial_stamps = 40)
  (h2 : final_stamps = 48) :
  (((final_stamps - initial_stamps : ℚ) / initial_stamps) * 100 : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_increase_l3464_346447


namespace NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3464_346488

def M : Set ℝ := {x | x^2 - x - 6 ≥ 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_of_N_and_complement_of_M :
  N ∩ (Set.univ \ M) = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3464_346488


namespace NUMINAMATH_CALUDE_function_power_id_implies_bijective_l3464_346401

variable {X : Type*}

def compose_n_times {X : Type*} (f : X → X) : ℕ → (X → X)
  | 0 => id
  | n + 1 => f ∘ (compose_n_times f n)

theorem function_power_id_implies_bijective
  (f : X → X) (k : ℕ) (hk : k > 0) (h : compose_n_times f k = id) :
  Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_function_power_id_implies_bijective_l3464_346401


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l3464_346477

/-- The number of siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 19

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 45

/-- The number of cats sold during the sale -/
def cats_sold : ℕ := 56

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

theorem pet_store_siamese_cats :
  initial_siamese_cats = 19 :=
by
  have h1 : initial_siamese_cats + initial_house_cats = initial_siamese_cats + 45 := by rfl
  have h2 : initial_siamese_cats + initial_house_cats - cats_sold = cats_remaining :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l3464_346477


namespace NUMINAMATH_CALUDE_numPaths_correct_l3464_346486

/-- The number of paths from (0,0) to (m,n) on Z^2, taking steps of +(1,0) or +(0,1) -/
def numPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that numPaths gives the correct number of paths -/
theorem numPaths_correct (m n : ℕ) : 
  numPaths m n = Nat.choose (m + n) m := by sorry

end NUMINAMATH_CALUDE_numPaths_correct_l3464_346486


namespace NUMINAMATH_CALUDE_parallel_lines_l3464_346458

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m : ℝ) : Prop :=
  -m = -(3*m - 2)/m

/-- The first line equation: mx + y + 3 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  m*x + y + 3 = 0

/-- The second line equation: (3m - 2)x + my + 2 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  (3*m - 2)*x + m*y + 2 = 0

/-- The main theorem: lines are parallel iff m = 1 or m = 2 -/
theorem parallel_lines (m : ℝ) : parallel m ↔ (m = 1 ∨ m = 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l3464_346458


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3464_346408

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l3464_346408


namespace NUMINAMATH_CALUDE_max_product_sum_l3464_346420

theorem max_product_sum (A M C : ℕ+) (h : A + M + C = 15) :
  (∀ A' M' C' : ℕ+, A' + M' + C' = 15 →
    A' * M' * C' + A' * M' + M' * C' + C' * A' + A' + M' + C' ≤
    A * M * C + A * M + M * C + C * A + A + M + C) →
  A * M * C + A * M + M * C + C * A + A + M + C = 215 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l3464_346420


namespace NUMINAMATH_CALUDE_combined_age_in_eight_years_l3464_346436

/-- Given the current age and the relation between your age and your brother's age 5 years ago,
    calculate the combined age of you and your brother in 8 years. -/
theorem combined_age_in_eight_years
  (your_current_age : ℕ)
  (h1 : your_current_age = 13)
  (h2 : your_current_age - 5 = (your_current_age + 3) - 5) :
  your_current_age + 8 + (your_current_age + 3) + 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_in_eight_years_l3464_346436


namespace NUMINAMATH_CALUDE_central_park_excess_cans_l3464_346437

def trash_can_problem (central_park : ℕ) (veterans_park : ℕ) : Prop :=
  -- Central Park had some more than half of the number of trash cans as in Veteran's Park
  central_park > veterans_park / 2 ∧
  -- Originally, there were 24 trash cans in Veteran's Park
  veterans_park = 24 ∧
  -- Half of the trash cans from Central Park were moved to Veteran's Park
  -- Now, there are 34 trash cans in Veteran's Park
  central_park / 2 + veterans_park = 34

theorem central_park_excess_cans :
  ∀ central_park veterans_park,
    trash_can_problem central_park veterans_park →
    central_park - veterans_park / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_central_park_excess_cans_l3464_346437


namespace NUMINAMATH_CALUDE_largest_expression_l3464_346440

def expr_a : ℕ := 2 + 3 + 1 + 7
def expr_b : ℕ := 2 * 3 + 1 + 7
def expr_c : ℕ := 2 + 3 * 1 + 7
def expr_d : ℕ := 2 + 3 + 1 * 7
def expr_e : ℕ := 2 * 3 * 1 * 7

theorem largest_expression : 
  expr_e > expr_a ∧ 
  expr_e > expr_b ∧ 
  expr_e > expr_c ∧ 
  expr_e > expr_d := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l3464_346440


namespace NUMINAMATH_CALUDE_expression_simplification_l3464_346469

theorem expression_simplification (x : ℝ) : 
  3*x - 3*(2 - x) + 4*(2 + 3*x) - 5*(1 - 2*x) = 28*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3464_346469


namespace NUMINAMATH_CALUDE_factorial_equality_l3464_346444

theorem factorial_equality : ∃ N : ℕ+, Nat.factorial 7 * Nat.factorial 11 = 18 * Nat.factorial N.val := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l3464_346444


namespace NUMINAMATH_CALUDE_all_propositions_false_l3464_346499

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel (x y : Line ⊕ Plane) : Prop := sorry
def perpendicular (x y : Line ⊕ Plane) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry
def intersects (p q : Plane) (l : Line) : Prop := sorry

-- Define the lines and planes
def m : Line := sorry
def n : Line := sorry
def a : Plane := sorry
def b : Plane := sorry

-- Define the propositions
def proposition1 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    parallel (Sum.inl m) (Sum.inr a) →
    parallel (Sum.inl n) (Sum.inr b) →
    parallel (Sum.inr a) (Sum.inr b) →
    parallel (Sum.inl m) (Sum.inl n)

def proposition2 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    parallel (Sum.inl m) (Sum.inl n) →
    contains a m →
    perpendicular (Sum.inl n) (Sum.inr b) →
    perpendicular (Sum.inr a) (Sum.inr b)

def proposition3 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    intersects a b m →
    parallel (Sum.inl m) (Sum.inl n) →
    (parallel (Sum.inl n) (Sum.inr a) ∧ parallel (Sum.inl n) (Sum.inr b))

def proposition4 : Prop :=
  ∀ (m n : Line) (a b : Plane),
    perpendicular (Sum.inl m) (Sum.inl n) →
    intersects a b m →
    (perpendicular (Sum.inl n) (Sum.inr a) ∨ perpendicular (Sum.inl n) (Sum.inr b))

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_false_l3464_346499


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3464_346428

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3464_346428


namespace NUMINAMATH_CALUDE_base7_subtraction_l3464_346406

/-- Represents a number in base 7 --/
def Base7 : Type := List Nat

/-- Converts a base 7 number to a natural number --/
def to_nat (b : Base7) : Nat :=
  b.foldr (fun digit acc => acc * 7 + digit) 0

/-- Subtracts two base 7 numbers --/
def subtract_base7 (a b : Base7) : Base7 :=
  sorry

theorem base7_subtraction :
  let a : Base7 := [1, 2, 1, 0, 0]
  let b : Base7 := [3, 6, 6, 6]
  subtract_base7 a b = [1, 1, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base7_subtraction_l3464_346406


namespace NUMINAMATH_CALUDE_derivative_at_negative_third_l3464_346438

/-- Given a function f(x) = x^2 + 2f'(-1/3)x, prove that f'(-1/3) = 2/3 -/
theorem derivative_at_negative_third (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + 2 * (deriv f (-1/3)) * x) :
  deriv f (-1/3) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_third_l3464_346438


namespace NUMINAMATH_CALUDE_power_sum_tenth_l3464_346452

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem power_sum_tenth (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_tenth_l3464_346452


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l3464_346409

theorem smallest_factor_for_perfect_square (n : ℕ) : n = 7 ↔ 
  (n > 0 ∧ 
   ∃ (m : ℕ), 1008 * n = m^2 ∧ 
   ∀ (k : ℕ), k > 0 → k < n → ¬∃ (l : ℕ), 1008 * k = l^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l3464_346409


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3464_346424

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.cos α > 0) : 
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3464_346424


namespace NUMINAMATH_CALUDE_min_value_theorem_l3464_346443

theorem min_value_theorem (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 2*m - 1 + n = 0) :
  (4:ℝ)^m + 2^n ≥ 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3464_346443


namespace NUMINAMATH_CALUDE_binomial_divisibility_theorem_l3464_346400

theorem binomial_divisibility_theorem (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, n > 0 ∧ 
    (n ∣ Nat.choose n k) ∧ 
    (∀ m : ℕ, 2 ≤ m → m < k → ¬(n ∣ Nat.choose n m)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_theorem_l3464_346400


namespace NUMINAMATH_CALUDE_f_of_f_3_l3464_346493

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem f_of_f_3 : f (f 3) = 1429 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_3_l3464_346493


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3464_346404

theorem absolute_value_inequality (x : ℝ) : 
  |x + 1| - |x - 4| > 3 ↔ x > 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3464_346404


namespace NUMINAMATH_CALUDE_rogers_broken_crayons_l3464_346423

/-- Given that Roger has 14 crayons in total, 2 new crayons, and 4 used crayons,
    prove that he has 8 broken crayons. -/
theorem rogers_broken_crayons :
  let total_crayons : ℕ := 14
  let new_crayons : ℕ := 2
  let used_crayons : ℕ := 4
  let broken_crayons : ℕ := total_crayons - new_crayons - used_crayons
  broken_crayons = 8 := by
  sorry

end NUMINAMATH_CALUDE_rogers_broken_crayons_l3464_346423


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l3464_346482

/-- The number of packs of green bouncy balls Maggie gave away -/
def green_packs_given_away : ℝ := 4.0

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs_bought : ℝ := 8.0

/-- The number of packs of green bouncy balls Maggie bought -/
def green_packs_bought : ℝ := 4.0

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℝ := 10.0

/-- The total number of bouncy balls Maggie kept -/
def balls_kept : ℕ := 80

theorem maggie_bouncy_balls :
  green_packs_given_away = 
    ((yellow_packs_bought + green_packs_bought) * balls_per_pack - balls_kept) / balls_per_pack :=
by sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l3464_346482


namespace NUMINAMATH_CALUDE_water_moles_theorem_l3464_346441

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  naoh_reactant : ℕ
  h2so4_reactant : ℕ
  na2so4_product : ℕ
  h2o_product : ℕ

/-- The balanced equation for the reaction -/
def balanced_equation : ChemicalEquation :=
  { naoh_reactant := 2
  , h2so4_reactant := 1
  , na2so4_product := 1
  , h2o_product := 2 }

/-- The number of moles of NaOH reacting -/
def naoh_moles : ℕ := 4

/-- The number of moles of H₂SO₄ reacting -/
def h2so4_moles : ℕ := 2

/-- Calculates the number of moles of water produced -/
def water_moles_produced (eq : ChemicalEquation) (naoh : ℕ) : ℕ :=
  (naoh * eq.h2o_product) / eq.naoh_reactant

/-- Theorem stating that 4 moles of water are produced -/
theorem water_moles_theorem :
  water_moles_produced balanced_equation naoh_moles = 4 :=
sorry

end NUMINAMATH_CALUDE_water_moles_theorem_l3464_346441


namespace NUMINAMATH_CALUDE_solve_gloria_pine_trees_l3464_346497

/-- The problem of determining the number of pine trees Gloria has -/
def GloriaPineTrees : Prop :=
  ∃ (num_pine_trees : ℕ),
    let cabin_cost : ℕ := 129000
    let cash : ℕ := 150
    let num_cypress : ℕ := 20
    let num_maple : ℕ := 24
    let cypress_price : ℕ := 100
    let maple_price : ℕ := 300
    let pine_price : ℕ := 200
    let leftover : ℕ := 350
    
    cabin_cost + leftover = 
      cash + num_cypress * cypress_price + num_maple * maple_price + num_pine_trees * pine_price ∧
    num_pine_trees = 600

theorem solve_gloria_pine_trees : GloriaPineTrees := by
  sorry

end NUMINAMATH_CALUDE_solve_gloria_pine_trees_l3464_346497


namespace NUMINAMATH_CALUDE_sequence_sum_bound_l3464_346411

theorem sequence_sum_bound (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, S n^2 - (n^2 + n - 1) * S n - (n^2 + n) = 0) →
  (∀ n : ℕ, b n = (n + 1) / ((n + 2)^2 * (a n)^2)) →
  (∀ n : ℕ, T (n + 1) = T n + b (n + 1)) →
  T 0 = 0 →
  ∀ n : ℕ, 0 < n → T n < 5/64 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_bound_l3464_346411


namespace NUMINAMATH_CALUDE_assistant_coaches_average_age_l3464_346453

/-- The average age of assistant coaches in a sports club --/
theorem assistant_coaches_average_age 
  (total_members : ℕ) 
  (overall_average : ℕ) 
  (girls_count : ℕ) 
  (girls_average : ℕ) 
  (boys_count : ℕ) 
  (boys_average : ℕ) 
  (head_coaches_count : ℕ) 
  (head_coaches_average : ℕ) 
  (assistant_coaches_count : ℕ) 
  (h_total : total_members = 50)
  (h_overall : overall_average = 22)
  (h_girls : girls_count = 30)
  (h_girls_avg : girls_average = 18)
  (h_boys : boys_count = 15)
  (h_boys_avg : boys_average = 20)
  (h_head_coaches : head_coaches_count = 3)
  (h_head_coaches_avg : head_coaches_average = 30)
  (h_assistant_coaches : assistant_coaches_count = 2)
  (h_coaches_total : head_coaches_count + assistant_coaches_count = 5) :
  (total_members * overall_average - 
   girls_count * girls_average - 
   boys_count * boys_average - 
   head_coaches_count * head_coaches_average) / assistant_coaches_count = 85 := by
sorry


end NUMINAMATH_CALUDE_assistant_coaches_average_age_l3464_346453


namespace NUMINAMATH_CALUDE_triangle_angle_c_two_thirds_pi_l3464_346426

theorem triangle_angle_c_two_thirds_pi
  (A B C : Real) (a b c : Real)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : (a + b + c) * (Real.sin A + Real.sin B - Real.sin C) = a * Real.sin B) :
  C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_two_thirds_pi_l3464_346426


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3464_346410

/-- A right circular cone with an inscribed right circular cylinder -/
structure ConeWithCylinder where
  -- Cone properties
  cone_diameter : ℝ
  cone_altitude : ℝ
  -- Cylinder properties
  cylinder_radius : ℝ
  -- Conditions
  cone_diameter_positive : 0 < cone_diameter
  cone_altitude_positive : 0 < cone_altitude
  cylinder_radius_positive : 0 < cylinder_radius
  cylinder_inscribed : cylinder_radius ≤ cone_diameter / 2
  cylinder_height_eq_diameter : cylinder_radius * 2 = cylinder_radius * 2
  shared_axis : True

/-- Theorem: The radius of the inscribed cylinder is 24/5 -/
theorem inscribed_cylinder_radius 
  (c : ConeWithCylinder) 
  (h1 : c.cone_diameter = 16) 
  (h2 : c.cone_altitude = 24) : 
  c.cylinder_radius = 24 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3464_346410


namespace NUMINAMATH_CALUDE_circle_symmetry_range_l3464_346431

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5*a = 0

-- Define the symmetry line equation
def symmetry_line (x y b : ℝ) : Prop :=
  y = x + 2*b

-- Theorem statement
theorem circle_symmetry_range (a b : ℝ) :
  (∃ x y : ℝ, circle_equation x y a ∧ symmetry_line x y b) →
  a + b < 0 ∧ ∀ c, c < 0 → ∃ a' b', a' + b' = c ∧
    ∃ x y : ℝ, circle_equation x y a' ∧ symmetry_line x y b' :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_range_l3464_346431


namespace NUMINAMATH_CALUDE_area_square_on_hypotenuse_l3464_346450

/-- Given a right triangle XYZ with right angle at Y, prove that the area of the square on XZ is 201
    when the sum of areas of a square on XY, a rectangle on YZ, and a square on XZ is 450,
    and YZ is 3 units longer than XY. -/
theorem area_square_on_hypotenuse (x y z : ℝ) (h1 : x^2 + y^2 = z^2)
    (h2 : y = x + 3) (h3 : x^2 + x * y + z^2 = 450) : z^2 = 201 := by
  sorry

end NUMINAMATH_CALUDE_area_square_on_hypotenuse_l3464_346450


namespace NUMINAMATH_CALUDE_unique_number_between_cube_roots_l3464_346475

theorem unique_number_between_cube_roots : ∃! (n : ℕ), 
  n > 0 ∧ 
  24 ∣ n ∧ 
  (9 : ℝ) < (n : ℝ) ^ (1/3 : ℝ) ∧ 
  (n : ℝ) ^ (1/3 : ℝ) < (9.1 : ℝ) ∧
  n = 744 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_between_cube_roots_l3464_346475


namespace NUMINAMATH_CALUDE_headlight_cost_is_180_l3464_346432

/-- Represents the scenario of Chris selling his car with two different offers --/
def car_sale_scenario (asking_price : ℝ) (maintenance_cost : ℝ) (headlight_cost : ℝ) : Prop :=
  let tire_cost := 3 * headlight_cost
  let first_offer := asking_price - maintenance_cost
  let second_offer := asking_price - (headlight_cost + tire_cost)
  (maintenance_cost = asking_price / 10) ∧
  (first_offer - second_offer = 200)

/-- Theorem stating that given the conditions, the headlight replacement cost is $180 --/
theorem headlight_cost_is_180 :
  car_sale_scenario 5200 520 180 :=
sorry

end NUMINAMATH_CALUDE_headlight_cost_is_180_l3464_346432


namespace NUMINAMATH_CALUDE_dandelion_puff_distribution_l3464_346425

theorem dandelion_puff_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h1 : total = 85)
  (h2 : given_away = 36)
  (h3 : friends = 5)
  (h4 : given_away < total) : 
  (total - given_away) / friends = (total - given_away) / (total - given_away) / friends :=
by sorry

end NUMINAMATH_CALUDE_dandelion_puff_distribution_l3464_346425


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3464_346468

theorem max_value_of_expression (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_sum : a + b + c + d = 200) : 
  a * b + b * c + c * d + (1/2) * d * a ≤ 11250 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ 0 ≤ d₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = 200 ∧ 
    a₀ * b₀ + b₀ * c₀ + c₀ * d₀ + (1/2) * d₀ * a₀ = 11250 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3464_346468


namespace NUMINAMATH_CALUDE_remaining_volume_is_five_sixths_l3464_346490

/-- The volume of a tetrahedron formed by planes passing through the midpoints
    of three edges sharing a vertex in a unit cube --/
def tetrahedron_volume : ℚ := 1 / 24

/-- The number of tetrahedra removed from the cube --/
def num_tetrahedra : ℕ := 8

/-- The volume of the remaining solid after removing tetrahedra from a unit cube --/
def remaining_volume : ℚ := 1 - num_tetrahedra * tetrahedron_volume

theorem remaining_volume_is_five_sixths :
  remaining_volume = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_remaining_volume_is_five_sixths_l3464_346490


namespace NUMINAMATH_CALUDE_percentage_of_seats_filled_l3464_346434

/-- Given a public show with 600 seats in total and 330 vacant seats,
    prove that 45% of the seats were filled. -/
theorem percentage_of_seats_filled (total_seats : ℕ) (vacant_seats : ℕ) : 
  total_seats = 600 →
  vacant_seats = 330 →
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_seats_filled_l3464_346434


namespace NUMINAMATH_CALUDE_tian_ji_winning_probability_l3464_346491

/-- Represents the horses of each competitor -/
inductive Horse : Type
  | top : Horse
  | middle : Horse
  | bottom : Horse

/-- Defines the ordering of horses based on their performance -/
def beats (h1 h2 : Horse) : Prop :=
  match h1, h2 with
  | Horse.top, Horse.middle => true
  | Horse.top, Horse.bottom => true
  | Horse.middle, Horse.bottom => true
  | _, _ => false

/-- King Qi's horses -/
def kingQi : Horse → Horse
  | Horse.top => Horse.top
  | Horse.middle => Horse.middle
  | Horse.bottom => Horse.bottom

/-- Tian Ji's horses -/
def tianJi : Horse → Horse
  | Horse.top => Horse.top
  | Horse.middle => Horse.middle
  | Horse.bottom => Horse.bottom

/-- The conditions of the horse performances -/
axiom horse_performance :
  (beats (tianJi Horse.top) (kingQi Horse.middle)) ∧
  (beats (kingQi Horse.top) (tianJi Horse.top)) ∧
  (beats (tianJi Horse.middle) (kingQi Horse.bottom)) ∧
  (beats (kingQi Horse.middle) (tianJi Horse.middle)) ∧
  (beats (kingQi Horse.bottom) (tianJi Horse.bottom))

/-- The probability of Tian Ji's horse winning -/
def winning_probability : ℚ := 1/3

/-- The main theorem to prove -/
theorem tian_ji_winning_probability :
  winning_probability = 1/3 := by sorry

end NUMINAMATH_CALUDE_tian_ji_winning_probability_l3464_346491


namespace NUMINAMATH_CALUDE_ball_travel_distance_l3464_346413

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  let first_rebound := initial_height * rebound_ratio
  let second_rebound := first_rebound * rebound_ratio
  initial_height + first_rebound + first_rebound + second_rebound + second_rebound

/-- Theorem: The ball travels 260 cm when it touches the floor for the third time -/
theorem ball_travel_distance :
  total_distance 104 0.5 = 260 := by
  sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l3464_346413


namespace NUMINAMATH_CALUDE_light_glow_time_l3464_346435

/-- The number of seconds between 1:57:58 am and 3:20:47 am -/
def total_seconds : ℕ := 4969

/-- The maximum number of times the light glowed -/
def max_glows : ℚ := 155.28125

/-- The time it takes for one glow in seconds -/
def time_per_glow : ℕ := 32

theorem light_glow_time :
  (total_seconds : ℚ) / max_glows = time_per_glow := by sorry

end NUMINAMATH_CALUDE_light_glow_time_l3464_346435


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3464_346449

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 3, 4}
  let B : Set ℕ := {0, 1, 3}
  A ∩ B = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3464_346449


namespace NUMINAMATH_CALUDE_olivias_paper_count_l3464_346427

/-- Calculates the total remaining pieces of paper given initial amounts and usage --/
def totalRemainingPieces (initialFolder1 initialFolder2 usedFolder1 usedFolder2 : ℕ) : ℕ :=
  (initialFolder1 - usedFolder1) + (initialFolder2 - usedFolder2)

/-- Theorem stating that given the initial conditions and usage, the total remaining pieces of paper is 130 --/
theorem olivias_paper_count :
  totalRemainingPieces 152 98 78 42 = 130 := by
  sorry

end NUMINAMATH_CALUDE_olivias_paper_count_l3464_346427


namespace NUMINAMATH_CALUDE_chord_slope_l3464_346456

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- Definition of the midpoint of the chord -/
def is_midpoint (x y : ℝ) : Prop := x = 4 ∧ y = 2

/-- Theorem: The slope of the chord is -1/2 -/
theorem chord_slope (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 → is_on_ellipse x2 y2 →
  is_midpoint ((x1 + x2) / 2) ((y1 + y2) / 2) →
  (y2 - y1) / (x2 - x1) = -1/2 := by sorry

end NUMINAMATH_CALUDE_chord_slope_l3464_346456


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l3464_346403

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l3464_346403


namespace NUMINAMATH_CALUDE_y_value_l3464_346498

theorem y_value (x y : ℝ) (h1 : x = 4) (h2 : y = 3 * x) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3464_346498


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3464_346405

theorem inequality_system_solution :
  {x : ℝ | x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2} = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3464_346405


namespace NUMINAMATH_CALUDE_not_all_on_curve_implies_exists_off_curve_l3464_346496

-- Define the necessary types and functions
variable (X Y : Type) -- X and Y represent coordinate types
variable (C : Set (X × Y)) -- C represents the curve
variable (f : X → Y → Prop) -- f represents the equation f(x, y) = 0

-- The main theorem
theorem not_all_on_curve_implies_exists_off_curve :
  (¬ ∀ x y, f x y → (x, y) ∈ C) →
  ∃ x y, f x y ∧ (x, y) ∉ C := by
sorry

end NUMINAMATH_CALUDE_not_all_on_curve_implies_exists_off_curve_l3464_346496


namespace NUMINAMATH_CALUDE_factor_63x_plus_54_l3464_346442

theorem factor_63x_plus_54 : ∀ x : ℝ, 63 * x + 54 = 9 * (7 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_factor_63x_plus_54_l3464_346442


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l3464_346480

/-- The equation of a line passing through a given point with a given angle -/
theorem line_equation_through_point_with_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) 
  (h_x₀ : x₀ = Real.sqrt 3) 
  (h_y₀ : y₀ = -2 * Real.sqrt 3) 
  (h_θ : θ = 135 * π / 180) :
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ 
                 ∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 
                 y - y₀ = Real.tan θ * (x - x₀) :=
sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l3464_346480


namespace NUMINAMATH_CALUDE_most_cars_are_blue_l3464_346451

theorem most_cars_are_blue (total : ℕ) (red blue yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
  sorry

end NUMINAMATH_CALUDE_most_cars_are_blue_l3464_346451


namespace NUMINAMATH_CALUDE_nails_to_buy_l3464_346462

theorem nails_to_buy (initial_nails : ℕ) (found_nails : ℕ) (total_needed : ℕ) : 
  initial_nails = 247 → found_nails = 144 → total_needed = 500 →
  total_needed - (initial_nails + found_nails) = 109 :=
by sorry

end NUMINAMATH_CALUDE_nails_to_buy_l3464_346462


namespace NUMINAMATH_CALUDE_not_necessarily_monotonic_increasing_l3464_346478

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def strictly_increasing_by_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x < f (x + 1)

-- Define monotonic increasing
def monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem statement
theorem not_necessarily_monotonic_increasing 
  (h : strictly_increasing_by_one f) : 
  ¬ (monotonic_increasing f) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_monotonic_increasing_l3464_346478
