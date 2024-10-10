import Mathlib

namespace solution_y_l2424_242477

theorem solution_y (x y : ℝ) 
  (hx : x > 2) 
  (hy : y > 2) 
  (h1 : 1/x + 1/y = 3/4) 
  (h2 : x*y = 8) : 
  y = 4 :=
sorry

end solution_y_l2424_242477


namespace interval_length_theorem_l2424_242432

theorem interval_length_theorem (c d : ℝ) : 
  (∃ (x_min x_max : ℝ), 
    (∀ x : ℝ, c ≤ 3*x + 4 ∧ 3*x + 4 ≤ d ↔ x_min ≤ x ∧ x ≤ x_max) ∧
    x_max - x_min = 15) →
  d - c = 45 := by
sorry

end interval_length_theorem_l2424_242432


namespace milk_tea_sales_l2424_242481

theorem milk_tea_sales (total : ℕ) 
  (h1 : (2 : ℚ) / 5 * total + (3 : ℚ) / 10 * total + 15 = total) 
  (h2 : (3 : ℚ) / 10 * total = 15) : total = 50 := by
  sorry

end milk_tea_sales_l2424_242481


namespace electric_sharpener_advantage_l2424_242416

def pencil_difference (hand_crank_time : ℕ) (electric_time : ℕ) (total_time : ℕ) : ℕ :=
  (total_time / electric_time) - (total_time / hand_crank_time)

theorem electric_sharpener_advantage :
  pencil_difference 45 20 360 = 10 := by
  sorry

end electric_sharpener_advantage_l2424_242416


namespace track_length_is_630_l2424_242494

/-- The length of the circular track in meters -/
def track_length : ℝ := 630

/-- The angle between the starting positions of the two runners in degrees -/
def start_angle : ℝ := 120

/-- The distance run by the first runner (Tom) before the first meeting in meters -/
def first_meeting_distance : ℝ := 120

/-- The additional distance run by the second runner (Jerry) between the first and second meeting in meters -/
def second_meeting_distance : ℝ := 180

/-- Theorem stating that the given conditions imply the track length is 630 meters -/
theorem track_length_is_630 : 
  ∃ (speed_tom speed_jerry : ℝ), 
    speed_tom > 0 ∧ speed_jerry > 0 ∧
    first_meeting_distance / speed_tom = (track_length * start_angle / 360 - first_meeting_distance) / speed_jerry ∧
    (track_length - (track_length * start_angle / 360 - first_meeting_distance) - second_meeting_distance) / speed_tom = 
      (track_length * start_angle / 360 - first_meeting_distance + second_meeting_distance) / speed_jerry :=
by
  sorry


end track_length_is_630_l2424_242494


namespace jack_sugar_final_amount_l2424_242472

/-- Given Jack's sugar transactions, prove the final amount of sugar. -/
theorem jack_sugar_final_amount
  (initial : ℕ)  -- Initial amount of sugar
  (used : ℕ)     -- Amount of sugar used
  (bought : ℕ)   -- Amount of sugar bought
  (h1 : initial = 65)
  (h2 : used = 18)
  (h3 : bought = 50) :
  initial - used + bought = 97 := by
  sorry

end jack_sugar_final_amount_l2424_242472


namespace le_zero_iff_lt_or_eq_l2424_242450

theorem le_zero_iff_lt_or_eq (x : ℝ) : x ≤ 0 ↔ x < 0 ∨ x = 0 := by sorry

end le_zero_iff_lt_or_eq_l2424_242450


namespace mushroom_ratio_l2424_242452

/-- Represents the types of mushrooms -/
inductive MushroomType
  | Spotted
  | Gilled

/-- Represents a mushroom -/
structure Mushroom where
  type : MushroomType

def total_mushrooms : Nat := 30
def gilled_mushrooms : Nat := 3

theorem mushroom_ratio :
  let spotted_mushrooms := total_mushrooms - gilled_mushrooms
  (gilled_mushrooms : Rat) / spotted_mushrooms = 1 / 9 := by
  sorry

end mushroom_ratio_l2424_242452


namespace complex_angle_of_one_plus_i_sqrt_three_l2424_242486

theorem complex_angle_of_one_plus_i_sqrt_three :
  let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end complex_angle_of_one_plus_i_sqrt_three_l2424_242486


namespace simplify_fraction_l2424_242455

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (3 / (x - 1)) + ((x - 3) / (1 - x^2)) = (2*x + 6) / (x^2 - 1) := by
  sorry

end simplify_fraction_l2424_242455


namespace sin_15_cos_75_plus_cos_15_sin_105_eq_1_l2424_242461

theorem sin_15_cos_75_plus_cos_15_sin_105_eq_1 :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sin_15_cos_75_plus_cos_15_sin_105_eq_1_l2424_242461


namespace chinese_dream_probability_l2424_242447

/-- The number of character cards -/
def num_cards : Nat := 3

/-- The total number of possible arrangements -/
def total_arrangements : Nat := Nat.factorial num_cards

/-- The number of arrangements forming the desired phrase -/
def desired_arrangements : Nat := 1

/-- The probability of forming the desired phrase -/
def probability : Rat := desired_arrangements / total_arrangements

theorem chinese_dream_probability :
  probability = 1 / 6 := by
  sorry

end chinese_dream_probability_l2424_242447


namespace base5_to_base8_conversion_l2424_242412

/-- Converts a base-5 number to base-10 -/
def base5_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 -/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem base5_to_base8_conversion :
  base10_to_base8 (base5_to_base10 1234) = 302 := by sorry

end base5_to_base8_conversion_l2424_242412


namespace consecutive_odd_numbers_sum_l2424_242497

theorem consecutive_odd_numbers_sum (k : ℤ) : 
  (2*k - 1) + (2*k + 1) + (2*k + 3) = (2*k - 1) + 128 → 2*k - 1 = 61 := by
  sorry

end consecutive_odd_numbers_sum_l2424_242497


namespace smallest_term_proof_l2424_242413

def arithmetic_sequence (n : ℕ) : ℕ := 7 * n

theorem smallest_term_proof :
  ∀ k : ℕ, 
    (arithmetic_sequence k > 150 ∧ arithmetic_sequence k % 5 = 0) → 
    arithmetic_sequence k ≥ 175 :=
by sorry

end smallest_term_proof_l2424_242413


namespace line_equation_proof_l2424_242482

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  x₀ : ℝ
  y₀ : ℝ
  passes_through : a * x₀ + b * y₀ + c = 0

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem line_equation_proof :
  ∃ (l : Line),
    l.x₀ = 1 ∧
    l.y₀ = 2 ∧
    l.a = 1 ∧
    l.b = 2 ∧
    l.c = -5 ∧
    perpendicular l ⟨2, -1, 1, 0, 0, by sorry⟩ := by
  sorry

end line_equation_proof_l2424_242482


namespace solution_set_equivalence_l2424_242445

/-- Given that the solution set of ax² - bx - 1 ≥ 0 is [1/3, 1/2],
    prove that the solution set of x² - bx - a < 0 is (-3, -2) -/
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ 1/3 ≤ x ∧ x ≤ 1/2) →
  (∀ x, x^2 - b*x - a < 0 ↔ -3 < x ∧ x < -2) :=
by sorry

end solution_set_equivalence_l2424_242445


namespace alternating_sum_of_squares_100_to_1_l2424_242466

/-- The sum of alternating differences of squares from 100² to 1² -/
def alternatingSumOfSquares : ℕ → ℤ
  | 0 => 0
  | n + 1 => (n + 1)^2 - alternatingSumOfSquares n

/-- The main theorem stating that the alternating sum of squares from 100² to 1² equals 5050 -/
theorem alternating_sum_of_squares_100_to_1 :
  alternatingSumOfSquares 100 = 5050 := by
  sorry


end alternating_sum_of_squares_100_to_1_l2424_242466


namespace popcorn_servings_needed_l2424_242415

/-- The number of pieces of popcorn in a serving -/
def serving_size : ℕ := 60

/-- The number of pieces Jared can eat -/
def jared_consumption : ℕ := 150

/-- The number of friends who can eat 80 pieces each -/
def friends_80 : ℕ := 3

/-- The number of friends who can eat 200 pieces each -/
def friends_200 : ℕ := 3

/-- The number of friends who can eat 100 pieces each -/
def friends_100 : ℕ := 4

/-- The number of pieces each friend in the first group can eat -/
def consumption_80 : ℕ := 80

/-- The number of pieces each friend in the second group can eat -/
def consumption_200 : ℕ := 200

/-- The number of pieces each friend in the third group can eat -/
def consumption_100 : ℕ := 100

/-- The theorem stating the number of servings needed -/
theorem popcorn_servings_needed : 
  (jared_consumption + 
   friends_80 * consumption_80 + 
   friends_200 * consumption_200 + 
   friends_100 * consumption_100 + 
   serving_size - 1) / serving_size = 24 :=
sorry

end popcorn_servings_needed_l2424_242415


namespace four_plus_six_equals_ten_l2424_242402

theorem four_plus_six_equals_ten : 4 + 6 = 10 := by
  sorry

end four_plus_six_equals_ten_l2424_242402


namespace smallest_e_value_l2424_242434

theorem smallest_e_value (a b c d e : ℤ) :
  (∃ (x : ℝ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (-3 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (6 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (10 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  (-1/4 : ℝ) ∈ {x : ℝ | a * x^4 + b * x^3 + c * x^2 + d * x + e = 0} →
  e > 0 →
  e ≥ 180 :=
by sorry

end smallest_e_value_l2424_242434


namespace first_equation_is_golden_second_equation_root_values_l2424_242490

/-- Definition of a golden equation -/
def is_golden_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a - b + c = 0

/-- The first equation -/
def first_equation (x : ℝ) : Prop :=
  2 * x^2 + 5 * x + 3 = 0

/-- The second equation -/
def second_equation (x a b : ℝ) : Prop :=
  3 * x^2 - a * x + b = 0

/-- Theorem for the first part -/
theorem first_equation_is_golden :
  is_golden_equation 2 5 3 :=
sorry

/-- Theorem for the second part -/
theorem second_equation_root_values (a b : ℝ) :
  is_golden_equation 3 (-a) b →
  second_equation a a b →
  (a = -1 ∨ a = 3/2) :=
sorry

end first_equation_is_golden_second_equation_root_values_l2424_242490


namespace isosceles_triangle_side_length_l2424_242422

/-- Represents an isosceles triangle DEF -/
structure IsoscelesTriangle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The length of one of the congruent sides -/
  side : ℝ
  /-- Assertion that the base is positive -/
  base_pos : base > 0
  /-- Assertion that the area is positive -/
  area_pos : area > 0
  /-- Assertion that the side is positive -/
  side_pos : side > 0

/-- Theorem stating the relationship between the base, area, and side length of an isosceles triangle -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) 
  (h1 : t.base = 30) 
  (h2 : t.area = 75) : 
  t.side = 5 * Real.sqrt 10 := by
  sorry

end isosceles_triangle_side_length_l2424_242422


namespace same_height_siblings_l2424_242440

-- Define the number of siblings
def num_siblings : ℕ := 5

-- Define the total height of all siblings
def total_height : ℕ := 330

-- Define the height of one sibling
def one_sibling_height : ℕ := 60

-- Define Eliza's height
def eliza_height : ℕ := 68

-- Define the height difference between Eliza and one sibling
def height_difference : ℕ := 2

-- Theorem to prove
theorem same_height_siblings (h : ℕ) : 
  h * 2 + one_sibling_height + eliza_height + (eliza_height + height_difference) = total_height →
  h = 66 := by
  sorry


end same_height_siblings_l2424_242440


namespace first_bank_interest_rate_l2424_242498

/-- Proves that the interest rate of the first bank is 4% given the investment conditions --/
theorem first_bank_interest_rate 
  (total_investment : ℝ)
  (first_bank_investment : ℝ)
  (second_bank_rate : ℝ)
  (total_interest : ℝ)
  (h1 : total_investment = 5000)
  (h2 : first_bank_investment = 1700)
  (h3 : second_bank_rate = 0.065)
  (h4 : total_interest = 282.50)
  : ∃ (first_bank_rate : ℝ), 
    first_bank_rate = 0.04 ∧ 
    first_bank_investment * first_bank_rate + 
    (total_investment - first_bank_investment) * second_bank_rate = 
    total_interest := by
  sorry

end first_bank_interest_rate_l2424_242498


namespace octagon_area_l2424_242484

theorem octagon_area (circle_area : ℝ) (h : circle_area = 256 * Real.pi) :
  ∃ (octagon_area : ℝ), octagon_area = 512 * Real.sqrt 2 := by
  sorry

end octagon_area_l2424_242484


namespace maria_anna_age_sum_prove_maria_anna_age_sum_l2424_242496

theorem maria_anna_age_sum : ℕ → ℕ → Prop :=
  fun maria_age anna_age =>
    (maria_age = anna_age + 5) →
    (maria_age + 7 = 3 * (anna_age - 3)) →
    (maria_age + anna_age = 27)

#check maria_anna_age_sum

theorem prove_maria_anna_age_sum :
  ∃ (maria_age anna_age : ℕ), maria_anna_age_sum maria_age anna_age :=
by
  sorry

end maria_anna_age_sum_prove_maria_anna_age_sum_l2424_242496


namespace tile_arrangement_probability_l2424_242443

theorem tile_arrangement_probability : 
  let total_tiles : ℕ := 7
  let x_tiles : ℕ := 4
  let o_tiles : ℕ := 3
  let favorable_arrangements : ℕ := Nat.choose 4 2
  let total_arrangements : ℕ := Nat.choose total_tiles x_tiles
  (favorable_arrangements : ℚ) / total_arrangements = 6 / 35 := by
sorry

end tile_arrangement_probability_l2424_242443


namespace area_to_paint_is_128_l2424_242423

/-- The area of a rectangle given its height and width -/
def rectangleArea (height width : ℝ) : ℝ := height * width

/-- The area to be painted on a wall with a window and a door -/
def areaToPaint (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) : ℝ :=
  rectangleArea wallHeight wallWidth - rectangleArea windowHeight windowWidth - rectangleArea doorHeight doorWidth

/-- Theorem stating that the area to be painted is 128 square feet -/
theorem area_to_paint_is_128 (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) :
  wallHeight = 10 ∧ wallWidth = 15 ∧ 
  windowHeight = 3 ∧ windowWidth = 5 ∧ 
  doorHeight = 1 ∧ doorWidth = 7 →
  areaToPaint wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth = 128 := by
  sorry

end area_to_paint_is_128_l2424_242423


namespace striped_area_equals_circle_area_l2424_242401

theorem striped_area_equals_circle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_diagonal := Real.sqrt (a^2 + b^2)
  let striped_area := π * (a^2 + b^2) / 4
  let circle_area := π * (rectangle_diagonal / 2)^2
  striped_area = circle_area := by
  sorry

end striped_area_equals_circle_area_l2424_242401


namespace second_term_value_l2424_242421

-- Define a sequence type
def Sequence := ℕ → ℝ

-- Define the Δ operator
def delta (A : Sequence) : Sequence :=
  λ n => A (n + 1) - A n

-- Main theorem
theorem second_term_value (A : Sequence) 
  (h1 : ∀ n, delta (delta A) n = 1)
  (h2 : A 12 = 0)
  (h3 : A 22 = 0) : 
  A 2 = 100 := by
sorry


end second_term_value_l2424_242421


namespace profit_maximized_at_six_l2424_242499

/-- Sales revenue function -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Total production cost function -/
def total_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit function -/
def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

/-- The production quantity that maximizes profit -/
def optimal_quantity : ℝ := 6

theorem profit_maximized_at_six :
  ∀ x > 0, profit x ≤ profit optimal_quantity :=
by sorry

end profit_maximized_at_six_l2424_242499


namespace cone_volume_l2424_242448

/-- The volume of a cone with given slant height and central angle of lateral surface --/
theorem cone_volume (slant_height : ℝ) (central_angle : ℝ) : 
  slant_height = 4 →
  central_angle = (2 * Real.pi) / 3 →
  ∃ (volume : ℝ), volume = (128 * Real.sqrt 2 / 81) * Real.pi := by
  sorry

end cone_volume_l2424_242448


namespace triangle_properties_l2424_242476

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h1 : t.a ≠ t.b)
  (h2 : Real.cos t.A ^ 2 - Real.cos t.B ^ 2 = Real.sqrt 3 * Real.sin t.A * Real.cos t.A - Real.sqrt 3 * Real.sin t.B * Real.cos t.B)
  (h3 : t.c = Real.sqrt 3)
  (h4 : Real.sin t.A = Real.sqrt 2 / 2) :
  t.C = π / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = (3 + Real.sqrt 3) / 4 := by
  sorry


end triangle_properties_l2424_242476


namespace negation_of_existence_negation_of_quadratic_equation_l2424_242436

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) ↔ (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_equation_l2424_242436


namespace restaurant_combinations_l2424_242467

/-- The number of main dishes on the menu -/
def main_dishes : ℕ := 15

/-- The number of appetizer options -/
def appetizer_options : ℕ := 5

/-- The number of people ordering -/
def num_people : ℕ := 2

theorem restaurant_combinations :
  (main_dishes ^ num_people) * appetizer_options = 1125 := by
  sorry

end restaurant_combinations_l2424_242467


namespace stratified_sampling_l2424_242493

theorem stratified_sampling (total_students : ℕ) (class1_students : ℕ) (class2_students : ℕ) 
  (sample_size : ℕ) (h1 : total_students = class1_students + class2_students) 
  (h2 : total_students = 96) (h3 : class1_students = 54) (h4 : class2_students = 42) 
  (h5 : sample_size = 16) :
  (class1_students * sample_size / total_students : ℚ) = 9 ∧
  (class2_students * sample_size / total_students : ℚ) = 7 :=
by sorry

end stratified_sampling_l2424_242493


namespace triangle_area_proof_l2424_242479

theorem triangle_area_proof (A B C : ℝ) (a b c : ℝ) : 
  C = π / 3 →
  c = Real.sqrt 7 →
  b = 3 * a →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry

end triangle_area_proof_l2424_242479


namespace quadratic_roots_transformation_l2424_242441

/-- Given a quadratic equation with roots r and s, prove the value of a in the new equation --/
theorem quadratic_roots_transformation (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 →
  s^2 - 5*s + 6 = 0 →
  r + s = 5 →
  r * s = 6 →
  ∃ b, (r^2 + 1)^2 + (-15)*(r^2 + 1) + b = 0 ∧ (s^2 + 1)^2 + (-15)*(s^2 + 1) + b = 0 :=
by sorry

end quadratic_roots_transformation_l2424_242441


namespace ellipse_axis_endpoint_distance_l2424_242424

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 16(x+2)^2 + 4y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∃ (A' B' : ℝ × ℝ),
    (∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 ↔ ((x, y) ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1})) ∧
    A' ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1 ∧ p.2^2 = 16} ∧
    B' ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1 ∧ (p.1 + 2)^2 = 4} ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoint_distance_l2424_242424


namespace power_fraction_equality_l2424_242483

theorem power_fraction_equality : (1 / ((-8^2)^3)) * (-8)^7 = 8 := by sorry

end power_fraction_equality_l2424_242483


namespace ellipse_minimum_area_l2424_242433

/-- An ellipse containing two specific circles has a minimum area of (3√3/2)π -/
theorem ellipse_minimum_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  π * a * b ≥ (3 * Real.sqrt 3 / 2) * π := by
  sorry

end ellipse_minimum_area_l2424_242433


namespace t_equals_negative_product_l2424_242404

theorem t_equals_negative_product : 
  let t := 1 / (1 - Real.rpow 3 (1/3))
  t = -(1 + Real.rpow 3 (1/3)) * (1 + Real.sqrt 3) := by
  sorry

end t_equals_negative_product_l2424_242404


namespace jesse_money_left_l2424_242462

def jesse_shopping (initial_amount : ℕ) (novel_cost : ℕ) : ℕ :=
  let lunch_cost := 2 * novel_cost
  initial_amount - (novel_cost + lunch_cost)

theorem jesse_money_left : jesse_shopping 50 7 = 29 := by
  sorry

end jesse_money_left_l2424_242462


namespace product_div_3_probability_l2424_242491

/-- The probability of rolling a number not divisible by 3 on a standard 6-sided die -/
def prob_not_div_3 : ℚ := 2/3

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that the product of the numbers rolled on 'num_dice' standard 6-sided dice is divisible by 3 -/
def prob_product_div_3 : ℚ := 1 - prob_not_div_3 ^ num_dice

theorem product_div_3_probability :
  prob_product_div_3 = 211/243 :=
sorry

end product_div_3_probability_l2424_242491


namespace simplified_rational_expression_l2424_242470

theorem simplified_rational_expression (x : ℝ) 
  (h1 : x^2 - 5*x + 6 ≠ 0) 
  (h2 : x^2 - 7*x + 12 ≠ 0) 
  (h3 : x^2 - 5*x + 4 ≠ 0) : 
  (x^2 - 3*x + 2) / (x^2 - 5*x + 6) / ((x^2 - 5*x + 4) / (x^2 - 7*x + 12)) = 1 := by
  sorry

end simplified_rational_expression_l2424_242470


namespace multiplier_value_l2424_242446

theorem multiplier_value (p q : ℕ) (x : ℚ) 
  (h1 : p > 1)
  (h2 : q > 1)
  (h3 : x * (p + 1) = 25 * (q + 1))
  (h4 : p + q ≥ 40)
  (h5 : ∀ p' q' : ℕ, p' > 1 → q' > 1 → x * (p' + 1) = 25 * (q' + 1) → p' + q' < p + q → False) :
  x = 325 := by
  sorry

end multiplier_value_l2424_242446


namespace sum_of_reciprocal_relations_l2424_242417

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 5)
  (h2 : x⁻¹ - y⁻¹ = -9) :
  x + y = -5/14 := by
  sorry

end sum_of_reciprocal_relations_l2424_242417


namespace rice_weight_proof_l2424_242419

/-- Given rice divided equally into 4 containers, each containing 50 ounces,
    prove that the total weight is 12.5 pounds, where 1 pound = 16 ounces. -/
theorem rice_weight_proof (containers : ℕ) (ounces_per_container : ℝ) 
    (ounces_per_pound : ℝ) : 
  containers = 4 →
  ounces_per_container = 50 →
  ounces_per_pound = 16 →
  (containers * ounces_per_container) / ounces_per_pound = 12.5 := by
  sorry

end rice_weight_proof_l2424_242419


namespace x2_value_l2424_242453

def sequence_condition (x : ℕ → ℝ) : Prop :=
  x 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 10 → x (n + 2) = ((x (n + 1) + 1) * (x (n + 1) - 1)) / x n) ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 11 → x n > 0) ∧
  x 12 = 0

theorem x2_value (x : ℕ → ℝ) (h : sequence_condition x) : x 2 = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end x2_value_l2424_242453


namespace x_value_l2424_242408

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := by
  sorry

end x_value_l2424_242408


namespace crossing_stretch_distance_l2424_242464

theorem crossing_stretch_distance :
  ∀ (num_people : ℕ) (run_speed bike_speed : ℝ) (total_time : ℝ),
    num_people = 4 →
    run_speed = 10 →
    bike_speed = 50 →
    total_time = 58 / 3 →
    (5 * (116 / 3) / bike_speed = total_time) :=
by
  sorry

end crossing_stretch_distance_l2424_242464


namespace cards_in_hospital_eq_403_l2424_242475

/-- The number of cards Mariela received while in the hospital -/
def cards_in_hospital : ℕ := 690 - 287

/-- Theorem stating that Mariela received 403 cards while in the hospital -/
theorem cards_in_hospital_eq_403 : cards_in_hospital = 403 := by
  sorry

end cards_in_hospital_eq_403_l2424_242475


namespace odd_as_difference_of_squares_l2424_242428

theorem odd_as_difference_of_squares (n : ℕ) : 2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end odd_as_difference_of_squares_l2424_242428


namespace set_operations_and_range_of_a_l2424_242406

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 1}

-- State the theorem
theorem set_operations_and_range_of_a :
  ∀ a : ℝ,
  (B ∪ C a = B) →
  ((A ∩ B = {x | 3 < x ∧ x < 6}) ∧
   ((Set.compl A ∪ Set.compl B) = {x | x ≤ 3 ∨ x ≥ 6}) ∧
   (a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5))) := by
  sorry

end set_operations_and_range_of_a_l2424_242406


namespace percentage_change_equivalence_l2424_242495

theorem percentage_change_equivalence (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) : 
  N * (1 + r/100) * (1 - s/100) < N ↔ r < 50*s / (100 - s) := by
  sorry

end percentage_change_equivalence_l2424_242495


namespace lindas_age_l2424_242435

theorem lindas_age (jane : ℕ) (linda : ℕ) : 
  linda = 2 * jane + 3 →
  (jane + 5) + (linda + 5) = 28 →
  linda = 13 := by
sorry

end lindas_age_l2424_242435


namespace simplify_expression_l2424_242411

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (x^2)⁻¹ - 2 = (1 - 2*x^2) / x^2 := by
  sorry

end simplify_expression_l2424_242411


namespace min_value_fraction_l2424_242471

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 4) :
  (1/a + 1/(b+1)) ≥ (3 + 2*Real.sqrt 2) / 6 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 4 ∧ 1/a₀ + 1/(b₀+1) = (3 + 2*Real.sqrt 2) / 6 :=
sorry

end min_value_fraction_l2424_242471


namespace units_digit_of_product_l2424_242478

theorem units_digit_of_product : (5^3 * 7^52) % 10 = 5 := by
  sorry

end units_digit_of_product_l2424_242478


namespace arthur_susan_age_difference_l2424_242438

def susan_age : ℕ := 15
def bob_age : ℕ := 11
def tom_age : ℕ := bob_age - 3
def total_age : ℕ := 51

theorem arthur_susan_age_difference : 
  ∃ (arthur_age : ℕ), arthur_age = total_age - susan_age - bob_age - tom_age ∧ arthur_age - susan_age = 2 :=
sorry

end arthur_susan_age_difference_l2424_242438


namespace julie_bought_two_boxes_l2424_242459

/-- Represents the number of boxes of standard paper Julie bought -/
def boxes_bought : ℕ := 2

/-- Represents the number of packages per box -/
def packages_per_box : ℕ := 5

/-- Represents the number of sheets per package -/
def sheets_per_package : ℕ := 250

/-- Represents the number of sheets used per newspaper -/
def sheets_per_newspaper : ℕ := 25

/-- Represents the number of newspapers Julie can print -/
def newspapers_printed : ℕ := 100

/-- Theorem stating that Julie bought 2 boxes of standard paper -/
theorem julie_bought_two_boxes :
  boxes_bought * packages_per_box * sheets_per_package =
  newspapers_printed * sheets_per_newspaper := by
  sorry

end julie_bought_two_boxes_l2424_242459


namespace profit_percentage_calculation_l2424_242410

/-- Calculate the profit percentage given the sale price including tax, tax rate, and cost price -/
theorem profit_percentage_calculation
  (sale_price_with_tax : ℝ)
  (tax_rate : ℝ)
  (cost_price : ℝ)
  (h1 : sale_price_with_tax = 616)
  (h2 : tax_rate = 0.1)
  (h3 : cost_price = 545.13) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 2.73) < 0.01 ∧
    profit_percentage = ((sale_price_with_tax / (1 + tax_rate) - cost_price) / cost_price) * 100 :=
by sorry

end profit_percentage_calculation_l2424_242410


namespace certain_number_proof_l2424_242403

theorem certain_number_proof (h : 213 * 16 = 3408) : ∃ x : ℝ, 0.16 * x = 0.3408 := by
  sorry

end certain_number_proof_l2424_242403


namespace roden_fish_count_l2424_242468

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end roden_fish_count_l2424_242468


namespace prob_at_least_one_diamond_l2424_242420

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of diamond cards in a standard deck -/
def diamondCardCount : ℕ := 13

/-- Probability of drawing at least one diamond when drawing two cards without replacement -/
def probAtLeastOneDiamond : ℚ :=
  1 - (standardDeckSize - diamondCardCount) * (standardDeckSize - diamondCardCount - 1) /
      (standardDeckSize * (standardDeckSize - 1))

theorem prob_at_least_one_diamond :
  probAtLeastOneDiamond = 15 / 34 :=
by sorry

end prob_at_least_one_diamond_l2424_242420


namespace imaginary_part_of_complex_fraction_l2424_242427

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 + 3 * Complex.I) / (3 - Complex.I)
  Complex.im z = 1 := by sorry

end imaginary_part_of_complex_fraction_l2424_242427


namespace inequality_proof_l2424_242437

theorem inequality_proof (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ := by
  sorry

end inequality_proof_l2424_242437


namespace cosine_sine_identity_l2424_242463

theorem cosine_sine_identity (θ : ℝ) 
  (h : Real.cos (π / 6 - θ) = 1 / 3) : 
  Real.cos (5 * π / 6 + θ) - Real.sin (θ - π / 6) ^ 2 = -11 / 9 := by
  sorry

end cosine_sine_identity_l2424_242463


namespace toms_work_schedule_l2424_242430

theorem toms_work_schedule (summer_hours_per_week : ℝ) (summer_weeks : ℕ) 
  (summer_total_earnings : ℝ) (semester_weeks : ℕ) (semester_target_earnings : ℝ) :
  summer_hours_per_week = 40 →
  summer_weeks = 8 →
  summer_total_earnings = 3200 →
  semester_weeks = 24 →
  semester_target_earnings = 2400 →
  let hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
  let semester_hours_per_week := semester_target_earnings / (hourly_wage * semester_weeks)
  semester_hours_per_week = 10 := by
  sorry

end toms_work_schedule_l2424_242430


namespace rhombus_square_equal_area_l2424_242429

/-- The side length of a square with area equal to a rhombus with diagonals 16 and 8 -/
theorem rhombus_square_equal_area (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 8) :
  ∃ (s : ℝ), s > 0 ∧ (d1 * d2) / 2 = s^2 ∧ s = 8 := by
  sorry

end rhombus_square_equal_area_l2424_242429


namespace novels_in_same_box_probability_l2424_242405

/-- The number of empty boxes Sam has -/
def num_boxes : ℕ := 5

/-- The total number of literature books Sam has -/
def total_books : ℕ := 15

/-- The number of novels among Sam's books -/
def num_novels : ℕ := 4

/-- The capacities of Sam's boxes -/
def box_capacities : List ℕ := [3, 4, 4, 2, 2]

/-- The probability of all novels ending up in the same box when packed randomly -/
def novels_in_same_box_prob : ℚ := 1 / 46905750

theorem novels_in_same_box_probability :
  num_boxes = 5 ∧
  total_books = 15 ∧
  num_novels = 4 ∧
  box_capacities = [3, 4, 4, 2, 2] →
  novels_in_same_box_prob = 1 / 46905750 :=
by sorry

end novels_in_same_box_probability_l2424_242405


namespace geometric_sequence_a7_l2424_242473

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 * a 5 = a 3 * a 6 →
  a 9 * a 10 = -8 →
  a 7 = -2 :=
sorry

end geometric_sequence_a7_l2424_242473


namespace average_weight_a_and_b_l2424_242414

/-- Given three weights a, b, and c, prove that their average weight of a and b is 40 kg
    under certain conditions. -/
theorem average_weight_a_and_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- The average weight of a, b, and c is 45 kg
  (b + c) / 2 = 43 →       -- The average weight of b and c is 43 kg
  b = 31 →                 -- The weight of b is 31 kg
  (a + b) / 2 = 40 :=      -- The average weight of a and b is 40 kg
by sorry

end average_weight_a_and_b_l2424_242414


namespace hot_dog_sales_first_innings_l2424_242426

/-- Represents the number of hot dogs in various states --/
structure HotDogSales where
  total : ℕ
  sold_later : ℕ
  left : ℕ

/-- Calculates the number of hot dogs sold in the first three innings --/
def sold_first (s : HotDogSales) : ℕ :=
  s.total - s.sold_later - s.left

/-- Theorem stating that for the given values, the number of hot dogs
    sold in the first three innings is 19 --/
theorem hot_dog_sales_first_innings
  (s : HotDogSales)
  (h1 : s.total = 91)
  (h2 : s.sold_later = 27)
  (h3 : s.left = 45) :
  sold_first s = 19 := by
  sorry

end hot_dog_sales_first_innings_l2424_242426


namespace percentage_multiplication_equality_l2424_242425

theorem percentage_multiplication_equality : ∃ x : ℝ, 45 * x = (45 / 100) * 900 ∧ x = 9 := by
  sorry

end percentage_multiplication_equality_l2424_242425


namespace infinite_geometric_series_first_term_l2424_242469

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 6)
  (h_S : S = 42)
  (h_sum : S = a / (1 - r))
  (h_convergence : abs r < 1) :
  a = 35 :=
sorry

end infinite_geometric_series_first_term_l2424_242469


namespace modular_inverse_15_mod_17_l2424_242400

theorem modular_inverse_15_mod_17 :
  ∃ a : ℕ, a ≤ 16 ∧ (15 * a) % 17 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_15_mod_17_l2424_242400


namespace geometric_sequence_sum_ratio_l2424_242474

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, 
    if S_10 : S_5 = 1 : 2, then (S_5 + S_10 + S_15) / (S_10 - S_5) = -9/2 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_geom : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) 
    (h_ratio : S 10 / S 5 = 1 / 2) :
    (S 5 + S 10 + S 15) / (S 10 - S 5) = -9/2 := by
  sorry

end geometric_sequence_sum_ratio_l2424_242474


namespace rectangular_prism_volume_l2424_242431

/-- The volume of a rectangular prism with face areas √3, √5, and √15 is √15 -/
theorem rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = Real.sqrt 3)
  (h2 : x * z = Real.sqrt 5)
  (h3 : y * z = Real.sqrt 15) :
  x * y * z = Real.sqrt 15 := by
sorry

end rectangular_prism_volume_l2424_242431


namespace class_size_problem_l2424_242460

theorem class_size_problem (class_a class_b class_c : ℕ) : 
  class_a = 2 * class_b →
  class_a = class_c / 3 →
  class_c = 120 →
  class_b = 20 := by
sorry

end class_size_problem_l2424_242460


namespace flier_distribution_l2424_242418

theorem flier_distribution (total : ℕ) (morning_fraction afternoon_fraction evening_fraction : ℚ) : 
  total = 10000 →
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  evening_fraction = 1/3 →
  let remaining_after_morning := total - (morning_fraction * total).num
  let remaining_after_afternoon := remaining_after_morning - (afternoon_fraction * remaining_after_morning).num
  let remaining_after_evening := remaining_after_afternoon - (evening_fraction * remaining_after_afternoon).num
  remaining_after_evening = 4000 := by
sorry

end flier_distribution_l2424_242418


namespace geometric_sequence_ratio_l2424_242488

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : geometric_sequence a q)
  (h_condition : a 5 ^ 2 = 2 * a 3 * a 9) :
  q = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_ratio_l2424_242488


namespace perfect_square_difference_l2424_242487

theorem perfect_square_difference (m n : ℕ+) 
  (h : 2001 * m^2 + m = 2002 * n^2 + n) :
  ∃ k : ℕ, m - n = k^2 := by sorry

end perfect_square_difference_l2424_242487


namespace xy_positive_iff_fraction_positive_solution_set_inequality_l2424_242442

-- Statement A
theorem xy_positive_iff_fraction_positive (x y : ℝ) :
  x * y > 0 ↔ x / y > 0 :=
sorry

-- Statement D
theorem solution_set_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 :=
sorry

end xy_positive_iff_fraction_positive_solution_set_inequality_l2424_242442


namespace quarters_ratio_proof_l2424_242489

def initial_quarters : ℕ := 50
def doubled_quarters : ℕ := initial_quarters * 2
def collected_second_year : ℕ := 3 * 12
def collected_third_year : ℕ := 1 * (12 / 3)
def total_before_loss : ℕ := doubled_quarters + collected_second_year + collected_third_year
def quarters_remaining : ℕ := 105

theorem quarters_ratio_proof :
  (total_before_loss - quarters_remaining) * 4 = total_before_loss :=
by sorry

end quarters_ratio_proof_l2424_242489


namespace jessica_purchases_total_cost_l2424_242465

/-- The cost of Jessica's cat toy in dollars -/
def cat_toy_cost : ℚ := 10.22

/-- The cost of Jessica's cage in dollars -/
def cage_cost : ℚ := 11.73

/-- The total cost of Jessica's purchases in dollars -/
def total_cost : ℚ := cat_toy_cost + cage_cost

theorem jessica_purchases_total_cost :
  total_cost = 21.95 := by sorry

end jessica_purchases_total_cost_l2424_242465


namespace shift_standard_parabola_2_right_l2424_242439

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola :=
  { f := fun x => x^2 }

/-- Theorem: Shifting the standard parabola 2 units right results in y = (x - 2)^2 -/
theorem shift_standard_parabola_2_right :
  (shift_parabola standard_parabola 2).f = fun x => (x - 2)^2 := by
  sorry

end shift_standard_parabola_2_right_l2424_242439


namespace circle_tangent_to_x_axis_l2424_242492

theorem circle_tangent_to_x_axis (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*x + 2*b*y + b^2 = 0 → 
   ∃ p : ℝ × ℝ, p.1^2 + p.2^2 = 0 ∧ p.2 = 0) → 
  b = 2 ∨ b = -2 := by
sorry

end circle_tangent_to_x_axis_l2424_242492


namespace triangle_side_length_l2424_242451

theorem triangle_side_length (a : ℝ) (B C : Real) (h1 : a = 8) (h2 : B = 60) (h3 : C = 75) :
  let A : ℝ := 180 - B - C
  let b : ℝ := a * Real.sin (B * π / 180) / Real.sin (A * π / 180)
  b = 4 * Real.sqrt 6 := by
sorry

end triangle_side_length_l2424_242451


namespace cryptarithm_solution_l2424_242454

def cryptarithm (C L M O P S U W Y : ℕ) : Prop :=
  let MSU := 100 * M + 10 * S + U
  let OLYMP := 10000 * O + 1000 * L + 100 * Y + 10 * M + P
  let MOSCOW := 100000 * M + 10000 * O + 1000 * S + 100 * C + 10 * O + W
  4 * MSU + 2 * OLYMP = MOSCOW

theorem cryptarithm_solution :
  ∃ (C L M O P S U W Y : ℕ),
    C = 5 ∧ L = 7 ∧ M = 1 ∧ O = 9 ∧ P = 2 ∧ S = 4 ∧ U = 3 ∧ W = 6 ∧ Y = 0 ∧
    cryptarithm C L M O P S U W Y ∧
    C ≠ L ∧ C ≠ M ∧ C ≠ O ∧ C ≠ P ∧ C ≠ S ∧ C ≠ U ∧ C ≠ W ∧ C ≠ Y ∧
    L ≠ M ∧ L ≠ O ∧ L ≠ P ∧ L ≠ S ∧ L ≠ U ∧ L ≠ W ∧ L ≠ Y ∧
    M ≠ O ∧ M ≠ P ∧ M ≠ S ∧ M ≠ U ∧ M ≠ W ∧ M ≠ Y ∧
    O ≠ P ∧ O ≠ S ∧ O ≠ U ∧ O ≠ W ∧ O ≠ Y ∧
    P ≠ S ∧ P ≠ U ∧ P ≠ W ∧ P ≠ Y ∧
    S ≠ U ∧ S ≠ W ∧ S ≠ Y ∧
    U ≠ W ∧ U ≠ Y ∧
    W ≠ Y :=
by sorry

end cryptarithm_solution_l2424_242454


namespace arithmetic_geometric_sum_l2424_242444

/-- Given an arithmetic sequence {a_n} with a₁ = 1, common difference d ≠ 0,
    and a₁, a₂, and a₅ forming a geometric sequence, 
    prove that the sum of the first 8 terms (S₈) is equal to 64. -/
theorem arithmetic_geometric_sum (d : ℝ) (h1 : d ≠ 0) : 
  let a : ℕ → ℝ := fun n => 1 + (n - 1) * d
  let S : ℕ → ℝ := fun n => (n * (2 + (n - 1) * d)) / 2
  (a 2)^2 = (a 1) * (a 5) → S 8 = 64 := by
sorry

end arithmetic_geometric_sum_l2424_242444


namespace train_crossing_time_l2424_242485

/-- Proves that a train of given length crossing a bridge of given length in a given time will take 40 seconds to cross a signal post. -/
theorem train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (bridge_crossing_time : ℝ) :
  train_length = 600 →
  bridge_length = 9000 →
  bridge_crossing_time = 600 →
  (train_length / (bridge_length / bridge_crossing_time)) = 40 := by
  sorry

end train_crossing_time_l2424_242485


namespace sin_theta_value_l2424_242449

theorem sin_theta_value (θ : Real) 
  (h1 : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = 1/3 := by
  sorry

end sin_theta_value_l2424_242449


namespace team_performance_l2424_242457

theorem team_performance (total_games : ℕ) (total_points : ℕ) 
  (wins : ℕ) (draws : ℕ) (losses : ℕ) : 
  total_games = 38 →
  total_points = 80 →
  wins + draws + losses = total_games →
  3 * wins + draws = total_points →
  wins > 2 * draws →
  wins > 5 * losses →
  draws = 11 := by
sorry

end team_performance_l2424_242457


namespace function_zero_range_l2424_242480

open Real

theorem function_zero_range (f : ℝ → ℝ) (m : ℝ) :
  (∃ x ∈ Set.Ioo 0 π, f x = 0) →
  (∀ x, f x = 2 * sin (x + π / 4) + m) →
  m ∈ Set.Icc (-2) (Real.sqrt 2) ∧ m ≠ Real.sqrt 2 :=
sorry

end function_zero_range_l2424_242480


namespace function_properties_l2424_242458

def continuous_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x ∈ s, ∀ ε > 0, ∃ δ > 0, ∀ y ∈ s, |y - x| < δ → |f y - f x| < ε

theorem function_properties (f : ℝ → ℝ) 
    (h_cont : continuous_on f (Set.univ : Set ℝ))
    (h_even : ∀ x : ℝ, f (-x) = f x)
    (h_incr : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) > 0)
    (h_zero : f (-1) = 0) :
  (f 3 < f (-4)) ∧ 
  (∀ x : ℝ, f x / x > 0 → (x > 1 ∨ (-1 < x ∧ x < 0))) ∧
  (∃ M : ℝ, ∀ x : ℝ, f x ≥ M) :=
by sorry

end function_properties_l2424_242458


namespace factorization_equality_l2424_242456

theorem factorization_equality (x₁ x₂ : ℝ) :
  x₁^3 - 2*x₁^2*x₂ - x₁ + 2*x₂ = (x₁ - 1) * (x₁ + 1) * (x₁ - 2*x₂) := by
  sorry

end factorization_equality_l2424_242456


namespace mario_haircut_price_l2424_242407

/-- The price of a haircut on a weekday -/
def weekday_price : ℝ := 18

/-- The price of a haircut on a weekend -/
def weekend_price : ℝ := 27

/-- The weekend price is 50% more than the weekday price -/
axiom weekend_price_relation : weekend_price = weekday_price * 1.5

theorem mario_haircut_price : weekday_price = 18 := by
  sorry

end mario_haircut_price_l2424_242407


namespace union_of_A_and_B_l2424_242409

def A : Set ℝ := {x | x = Real.log 1 ∨ x = 1}
def B : Set ℝ := {x | x = -1 ∨ x = 0}

theorem union_of_A_and_B : A ∪ B = {x | x = -1 ∨ x = 0 ∨ x = 1} := by sorry

end union_of_A_and_B_l2424_242409
