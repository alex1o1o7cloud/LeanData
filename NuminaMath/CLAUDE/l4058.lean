import Mathlib

namespace NUMINAMATH_CALUDE_balloon_sum_l4058_405862

theorem balloon_sum (x y : ℝ) (hx : x = 7.5) (hy : y = 5.2) : x + y = 12.7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_sum_l4058_405862


namespace NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l4058_405820

theorem partition_contains_perfect_square_sum (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Set ℕ), (A ∪ B = Finset.range n.succ) → (A ∩ B = ∅) →
  (∃ (x y : ℕ), x ≠ y ∧ ((x ∈ A ∧ y ∈ A) ∨ (x ∈ B ∧ y ∈ B)) ∧ ∃ (z : ℕ), x + y = z^2) :=
by sorry

end NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l4058_405820


namespace NUMINAMATH_CALUDE_a_10_has_many_nines_l4058_405817

def a : ℕ → ℕ
  | 0 => 9
  | n + 1 => 3 * (a n)^4 + 4 * (a n)^3

theorem a_10_has_many_nines : ∃ k : ℕ, k ≥ 1024 ∧ a 10 ≡ 10^k - 1 [ZMOD 10^k] :=
sorry

end NUMINAMATH_CALUDE_a_10_has_many_nines_l4058_405817


namespace NUMINAMATH_CALUDE_sum_remainder_mod_8_l4058_405802

theorem sum_remainder_mod_8 : (7150 + 7151 + 7152 + 7153 + 7154 + 7155) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_8_l4058_405802


namespace NUMINAMATH_CALUDE_x_eq_two_iff_quadratic_eq_zero_l4058_405813

theorem x_eq_two_iff_quadratic_eq_zero : ∀ x : ℝ, x = 2 ↔ x^2 - 4*x + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_eq_two_iff_quadratic_eq_zero_l4058_405813


namespace NUMINAMATH_CALUDE_specific_field_perimeter_l4058_405864

/-- A rectangular field with specific properties -/
structure RectangularField where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_eq : length = breadth + 30
  area_eq : area = length * breadth

/-- The perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.length + field.breadth)

/-- Theorem stating the perimeter of the specific field is 540 meters -/
theorem specific_field_perimeter :
  ∃ (field : RectangularField), field.area = 18000 ∧ perimeter field = 540 := by
  sorry

end NUMINAMATH_CALUDE_specific_field_perimeter_l4058_405864


namespace NUMINAMATH_CALUDE_divisibility_property_l4058_405867

theorem divisibility_property (a b c d : ℤ) (h1 : a ≠ b) (h2 : (a - b) ∣ (a * c + b * d)) :
  (a - b) ∣ (a * d + b * c) := by sorry

end NUMINAMATH_CALUDE_divisibility_property_l4058_405867


namespace NUMINAMATH_CALUDE_expected_black_pairs_in_deck_l4058_405825

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expected_black_pairs (total_cards : ℕ) (black_cards : ℕ) : ℚ :=
  (black_cards : ℚ) * ((black_cards - 1 : ℚ) / (total_cards - 1 : ℚ))

theorem expected_black_pairs_in_deck : 
  expected_black_pairs 52 30 = 870 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_in_deck_l4058_405825


namespace NUMINAMATH_CALUDE_exponent_simplification_l4058_405889

theorem exponent_simplification :
  ((-5^2)^4 * (-5)^11) / ((-5)^3) = 5^16 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l4058_405889


namespace NUMINAMATH_CALUDE_triangle_relationships_l4058_405879

/-- Given a triangle with sides a, b, c, circumradius R, inradius r, and semiperimeter p,
    prove the following relationships. -/
theorem triangle_relationships 
  (a b c R r p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0 ∧ p > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_circumradius : R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)))
  (h_inradius : r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  (a * b * c = 4 * p * r * R) ∧ 
  (a * b + b * c + c * a = r^2 + p^2 + 4 * r * R) := by
  sorry


end NUMINAMATH_CALUDE_triangle_relationships_l4058_405879


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l4058_405855

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ 1 → (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l4058_405855


namespace NUMINAMATH_CALUDE_bob_has_winning_strategy_l4058_405843

/-- Represents a cell in the grid -/
structure Cell :=
  (row : ℕ)
  (col : ℕ)
  (value : ℚ)

/-- Represents the game state -/
structure GameState :=
  (grid : List (List Cell))
  (current_player : Bool)  -- true for Alice, false for Bob

/-- Checks if a cell is part of a continuous path from top to bottom -/
def is_part_of_path (grid : List (List Cell)) (cell : Cell) : Prop :=
  sorry

/-- Determines if there exists a winning path for Alice -/
def exists_winning_path (state : GameState) : Prop :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Cell

/-- Determines if a strategy is winning for Bob -/
def is_winning_strategy_for_bob (strategy : Strategy) : Prop :=
  ∀ (state : GameState), 
    (state.current_player = false) → 
    ¬(exists_winning_path (state))

/-- The main theorem stating that Bob has a winning strategy -/
theorem bob_has_winning_strategy : 
  ∃ (strategy : Strategy), is_winning_strategy_for_bob strategy :=
sorry

end NUMINAMATH_CALUDE_bob_has_winning_strategy_l4058_405843


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l4058_405881

-- Define a digit in base 5
def is_base5_digit (B : ℕ) : Prop := 0 ≤ B ∧ B < 5

-- Define a base greater than or equal to 6
def is_valid_base (c : ℕ) : Prop := c ≥ 6

-- Define the equality BBB_5 = 44_c
def number_equality (B c : ℕ) : Prop := 31 * B = 4 * (c + 1)

-- Theorem statement
theorem smallest_sum_B_plus_c :
  ∀ B c : ℕ,
  is_base5_digit B →
  is_valid_base c →
  number_equality B c →
  (∀ B' c' : ℕ, is_base5_digit B' → is_valid_base c' → number_equality B' c' → B + c ≤ B' + c') →
  B + c = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l4058_405881


namespace NUMINAMATH_CALUDE_simplify_expression_l4058_405851

theorem simplify_expression (n : ℕ) : (2^(n+5) - 3*(2^n)) / (3*(2^(n+3))) = 29 / 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4058_405851


namespace NUMINAMATH_CALUDE_digit_product_sum_28_l4058_405872

/-- Represents a base-10 digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Converts two digits to a two-digit number -/
def toTwoDigitNumber (a b : Digit) : TwoDigitNumber :=
  ⟨a.val * 10 + b.val, by sorry⟩

/-- Converts a digit to a three-digit number where all digits are the same -/
def toThreeDigitSameNumber (e : Digit) : Nat :=
  e.val * 100 + e.val * 10 + e.val

theorem digit_product_sum_28 
  (A B C D E : Digit) 
  (h_unique : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_product : (toTwoDigitNumber A B).val * (toTwoDigitNumber C D).val = toThreeDigitSameNumber E) :
  A.val + B.val + C.val + D.val + E.val = 28 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_sum_28_l4058_405872


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l4058_405835

/-- Given a triangle ABC with side lengths a, b, c, altitudes ha, hb, hc, and circumradius R,
    the ratio of the sum of pairwise products of side lengths to the sum of altitudes
    is equal to the diameter of the circumscribed circle. -/
theorem triangle_ratio_theorem (a b c ha hb hc R : ℝ) :
  a > 0 → b > 0 → c > 0 → ha > 0 → hb > 0 → hc > 0 → R > 0 →
  (a * b + b * c + a * c) / (ha + hb + hc) = 2 * R := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l4058_405835


namespace NUMINAMATH_CALUDE_elliptic_curve_solutions_l4058_405854

theorem elliptic_curve_solutions (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  ∀ (x y : ℤ), y^2 = x^3 - p^2*x ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p ∧ y = 0) ∨ 
    (x = -p ∧ y = 0) ∨ 
    (x = (p^2 + 1)/2 ∧ (y = ((p^2 - 1)/2)*p ∨ y = -((p^2 - 1)/2)*p)) :=
by sorry

end NUMINAMATH_CALUDE_elliptic_curve_solutions_l4058_405854


namespace NUMINAMATH_CALUDE_symmetric_point_in_first_quadrant_l4058_405866

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Symmetry about the x-axis -/
def symmetric_about_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The original point P -/
def P : Point :=
  ⟨2, -3⟩

theorem symmetric_point_in_first_quadrant :
  is_in_first_quadrant (symmetric_about_x_axis P) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_in_first_quadrant_l4058_405866


namespace NUMINAMATH_CALUDE_star_sum_five_l4058_405899

def star (a b : ℕ) : ℕ := a^b + a*b

theorem star_sum_five (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h_star : star a b = 15) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_star_sum_five_l4058_405899


namespace NUMINAMATH_CALUDE_second_square_area_equal_l4058_405890

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- The length of a leg of the isosceles right triangle -/
  leg : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed with two vertices on one leg, one on the hypotenuse, and one on the other leg -/
  inscribed : square_side > 0 ∧ square_side < leg
  /-- The area of the inscribed square is 625 cm² -/
  area_condition : square_side ^ 2 = 625

/-- The area of another inscribed square in the same triangle -/
def second_square_area (triangle : IsoscelesRightTriangleWithSquare) : ℝ :=
  triangle.square_side ^ 2

theorem second_square_area_equal (triangle : IsoscelesRightTriangleWithSquare) :
  second_square_area triangle = 625 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_equal_l4058_405890


namespace NUMINAMATH_CALUDE_train_length_l4058_405844

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h1 : speed_kmh = 144) (h2 : time_sec = 20) :
  speed_kmh * (1000 / 3600) * time_sec = 800 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4058_405844


namespace NUMINAMATH_CALUDE_distance_between_points_l4058_405814

theorem distance_between_points (x : ℝ) : 
  let A := 3 + x
  let B := 3 - x
  |A - B| = 8 → |x| = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l4058_405814


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4058_405849

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 5| = 3 * x + 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4058_405849


namespace NUMINAMATH_CALUDE_sams_sitting_fee_is_correct_l4058_405847

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_sitting_fee : ℝ := 140

/-- The price per sheet for John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The one-time sitting fee for John's Photo World -/
def johns_sitting_fee : ℝ := 125

/-- The price per sheet for Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The number of sheets for which the total price is the same -/
def num_sheets : ℕ := 12

theorem sams_sitting_fee_is_correct :
  johns_price_per_sheet * num_sheets + johns_sitting_fee =
  sams_price_per_sheet * num_sheets + sams_sitting_fee :=
by
  sorry

#check sams_sitting_fee_is_correct

end NUMINAMATH_CALUDE_sams_sitting_fee_is_correct_l4058_405847


namespace NUMINAMATH_CALUDE_train_length_l4058_405856

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 8 → ∃ length : ℝ, 
  (length ≥ 133.36 ∧ length ≤ 133.37) ∧ length = speed * time * (1000 / 3600) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4058_405856


namespace NUMINAMATH_CALUDE_triangle_area_l4058_405840

/-- Given a triangle with perimeter 32 cm and inradius 3.5 cm, its area is 56 cm² -/
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 32) (h2 : r = 3.5) :
  r * p / 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4058_405840


namespace NUMINAMATH_CALUDE_fraction_simplification_l4058_405876

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = (5 * Real.sqrt 2) / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4058_405876


namespace NUMINAMATH_CALUDE_fifth_power_sum_l4058_405887

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 4)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 23)
  (h4 : a * x^4 + b * y^4 = 54) :
  a * x^5 + b * y^5 = 470 := by
  sorry


end NUMINAMATH_CALUDE_fifth_power_sum_l4058_405887


namespace NUMINAMATH_CALUDE_valid_pairs_characterization_l4058_405870

/-- A function that checks if a given pair (m, n) of natural numbers
    satisfies the condition that 2^m * 3^n + 1 is a perfect square. -/
def is_valid_pair (m n : ℕ) : Prop :=
  ∃ x : ℕ, 2^m * 3^n + 1 = x^2

/-- The set of all valid pairs (m, n) that satisfy the condition. -/
def valid_pairs : Set (ℕ × ℕ) :=
  {p | is_valid_pair p.1 p.2}

/-- The theorem stating that the only valid pairs are (3, 1), (4, 1), and (5, 2). -/
theorem valid_pairs_characterization :
  valid_pairs = {(3, 1), (4, 1), (5, 2)} := by
  sorry


end NUMINAMATH_CALUDE_valid_pairs_characterization_l4058_405870


namespace NUMINAMATH_CALUDE_butterfat_mixture_proof_l4058_405809

/-- Proves that mixing 8 gallons of 35% butterfat milk with 12 gallons of 10% butterfat milk
    results in a mixture that is 20% butterfat. -/
theorem butterfat_mixture_proof :
  let x : ℝ := 8 -- Amount of 35% butterfat milk in gallons
  let y : ℝ := 12 -- Amount of 10% butterfat milk in gallons
  let butterfat_high : ℝ := 0.35 -- Percentage of butterfat in high-fat milk
  let butterfat_low : ℝ := 0.10 -- Percentage of butterfat in low-fat milk
  let butterfat_target : ℝ := 0.20 -- Target percentage of butterfat in mixture
  (butterfat_high * x + butterfat_low * y) / (x + y) = butterfat_target :=
by sorry

end NUMINAMATH_CALUDE_butterfat_mixture_proof_l4058_405809


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l4058_405880

theorem soccer_ball_cost (F S : ℝ) 
  (eq1 : 3 * F + S = 155) 
  (eq2 : 2 * F + 3 * S = 220) : 
  S = 50 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l4058_405880


namespace NUMINAMATH_CALUDE_distance_from_origin_l4058_405853

theorem distance_from_origin (x y : ℝ) (h1 : y = 15) (h2 : x > 3)
  (h3 : Real.sqrt ((x - 3)^2 + (y - 8)^2) = 11) :
  Real.sqrt (x^2 + y^2) = Real.sqrt 306 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l4058_405853


namespace NUMINAMATH_CALUDE_cos_equality_theorem_l4058_405805

theorem cos_equality_theorem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (310 * π / 180)) ↔ (n = 50 ∨ n = 310) :=
by sorry

end NUMINAMATH_CALUDE_cos_equality_theorem_l4058_405805


namespace NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l4058_405831

theorem inscribed_sphere_in_cone (a b c : ℝ) : 
  let cone_base_radius : ℝ := 20
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := (120 * (Real.sqrt 13 - 10)) / 27
  sphere_radius = a * Real.sqrt c - b →
  a + b + c = 253 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_in_cone_l4058_405831


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l4058_405893

/-- The percentage of chromium in the new alloy formed by mixing two alloys -/
theorem chromium_percentage_in_new_alloy 
  (chromium_percentage1 : Real) 
  (chromium_percentage2 : Real)
  (weight1 : Real) 
  (weight2 : Real) 
  (h1 : chromium_percentage1 = 12 / 100)
  (h2 : chromium_percentage2 = 8 / 100)
  (h3 : weight1 = 15)
  (h4 : weight2 = 40) : 
  (chromium_percentage1 * weight1 + chromium_percentage2 * weight2) / (weight1 + weight2) = 1 / 11 := by
sorry

#eval (1 / 11 : Float) * 100 -- To show the approximate percentage

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l4058_405893


namespace NUMINAMATH_CALUDE_iphone_sales_l4058_405888

theorem iphone_sales (iphone_price : ℝ) (ipad_count : ℕ) (ipad_price : ℝ)
                     (appletv_count : ℕ) (appletv_price : ℝ) (average_price : ℝ) :
  iphone_price = 1000 →
  ipad_count = 20 →
  ipad_price = 900 →
  appletv_count = 80 →
  appletv_price = 200 →
  average_price = 670 →
  ∃ (iphone_count : ℕ),
    (iphone_count : ℝ) * iphone_price + (ipad_count : ℝ) * ipad_price + (appletv_count : ℝ) * appletv_price =
    average_price * ((iphone_count : ℝ) + (ipad_count : ℝ) + (appletv_count : ℝ)) ∧
    iphone_count = 100 :=
by sorry

end NUMINAMATH_CALUDE_iphone_sales_l4058_405888


namespace NUMINAMATH_CALUDE_alternating_color_probability_l4058_405869

/-- The probability of drawing 10 balls from a box containing 5 white and 5 black balls
    such that the colors alternate is equal to 1/126. -/
theorem alternating_color_probability (n : ℕ) (white_balls black_balls : ℕ) : 
  n = 10 → white_balls = 5 → black_balls = 5 →
  (Nat.choose n white_balls : ℚ)⁻¹ * 2 = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l4058_405869


namespace NUMINAMATH_CALUDE_knight_position_proof_l4058_405885

/-- The total number of people in the line -/
def total_people : ℕ := 2022

/-- The position of the knight from the left -/
def knight_position : ℕ := 48

/-- The ratio of liars to the right compared to the left for each person (except the ends) -/
def liar_ratio : ℕ := 42

theorem knight_position_proof :
  ∀ k : ℕ, 
  1 < k ∧ k < total_people →
  (total_people - k = liar_ratio * (k - 1)) ↔ 
  k = knight_position :=
sorry

end NUMINAMATH_CALUDE_knight_position_proof_l4058_405885


namespace NUMINAMATH_CALUDE_second_boy_marbles_l4058_405858

-- Define the number of marbles for each boy as functions of x
def boy1_marbles (x : ℚ) : ℚ := 4 * x + 2
def boy2_marbles (x : ℚ) : ℚ := 3 * x - 1
def boy3_marbles (x : ℚ) : ℚ := 5 * x + 3

-- Define the total number of marbles
def total_marbles : ℚ := 128

-- Theorem statement
theorem second_boy_marbles :
  ∃ x : ℚ, 
    boy1_marbles x + boy2_marbles x + boy3_marbles x = total_marbles ∧
    boy2_marbles x = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_marbles_l4058_405858


namespace NUMINAMATH_CALUDE_ticket_price_difference_l4058_405857

def total_cost : ℝ := 77
def adult_ticket_cost : ℝ := 19
def num_adults : ℕ := 2
def num_children : ℕ := 3

theorem ticket_price_difference : 
  ∃ (child_ticket_cost : ℝ),
    total_cost = num_adults * adult_ticket_cost + num_children * child_ticket_cost ∧
    adult_ticket_cost - child_ticket_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_difference_l4058_405857


namespace NUMINAMATH_CALUDE_school_travel_speed_l4058_405808

/-- Proves that the speed on the second day is 10 km/hr given the conditions of the problem -/
theorem school_travel_speed 
  (distance : ℝ) 
  (speed_day1 : ℝ) 
  (late_time : ℝ) 
  (early_time : ℝ) 
  (h1 : distance = 2.5) 
  (h2 : speed_day1 = 5) 
  (h3 : late_time = 7 / 60) 
  (h4 : early_time = 8 / 60) : 
  let correct_time := distance / speed_day1
  let actual_time_day2 := correct_time - late_time - early_time
  distance / actual_time_day2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_speed_l4058_405808


namespace NUMINAMATH_CALUDE_intersection_points_distance_squared_l4058_405830

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The squared distance between two points in 2D space -/
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: The squared distance between intersection points of two specific circles is 16 -/
theorem intersection_points_distance_squared
  (c1 : Circle)
  (c2 : Circle)
  (h1 : c1.center = (1, 3))
  (h2 : c1.radius = 3)
  (h3 : c2.center = (1, -4))
  (h4 : c2.radius = 6)
  : ∃ p1 p2 : ℝ × ℝ,
    squaredDistance p1 p2 = 16 ∧
    squaredDistance p1 c1.center = c1.radius^2 ∧
    squaredDistance p1 c2.center = c2.radius^2 ∧
    squaredDistance p2 c1.center = c1.radius^2 ∧
    squaredDistance p2 c2.center = c2.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_distance_squared_l4058_405830


namespace NUMINAMATH_CALUDE_expression_evaluation_l4058_405836

theorem expression_evaluation :
  (∀ a : ℤ, a = -3 → (a + 3)^2 + (2 + a) * (2 - a) = -5) ∧
  (∀ x : ℤ, x = -3 → 2 * x * (3 * x^2 - 4 * x + 1) - 3 * x^2 * (x - 3) = -78) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4058_405836


namespace NUMINAMATH_CALUDE_sixty_degree_iff_arithmetic_progression_l4058_405897

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180

/-- The property that the angles of a triangle are in arithmetic progression -/
def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

/-- Theorem stating that B = 60° is necessary and sufficient for the angles to be in arithmetic progression -/
theorem sixty_degree_iff_arithmetic_progression (t : Triangle) :
  t.B = 60 ↔ angles_in_arithmetic_progression t := by
  sorry

end NUMINAMATH_CALUDE_sixty_degree_iff_arithmetic_progression_l4058_405897


namespace NUMINAMATH_CALUDE_train_passing_time_l4058_405868

theorem train_passing_time (slower_speed faster_speed : ℝ) (train_length : ℝ) : 
  slower_speed = 36 →
  faster_speed = 45 →
  train_length = 90.0072 →
  (train_length / ((slower_speed + faster_speed) * (1000 / 3600))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l4058_405868


namespace NUMINAMATH_CALUDE_students_with_d_grade_l4058_405827

theorem students_with_d_grade (total_students : ℕ) 
  (a_fraction b_fraction c_fraction : ℚ) : 
  total_students = 800 →
  a_fraction = 1/5 →
  b_fraction = 1/4 →
  c_fraction = 1/2 →
  total_students - (total_students * a_fraction + total_students * b_fraction + total_students * c_fraction) = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_with_d_grade_l4058_405827


namespace NUMINAMATH_CALUDE_check_error_l4058_405877

theorem check_error (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →
  100 * y + x - (100 * x + y) = 2376 →
  y = 2 * x + 12 →
  x = 12 ∧ y = 36 := by
sorry

end NUMINAMATH_CALUDE_check_error_l4058_405877


namespace NUMINAMATH_CALUDE_root_ratio_equality_l4058_405812

/-- 
Given a complex polynomial z^4 + az^3 + bz^2 + cz + d with roots p, q, r, s,
if a^2d = c^2 and c ≠ 0, then p/r = s/q.
-/
theorem root_ratio_equality (a b c d p q r s : ℂ) : 
  p * q * r * s = d → 
  p + q + r + s = -a → 
  a^2 * d = c^2 → 
  c ≠ 0 → 
  p / r = s / q := by
sorry

end NUMINAMATH_CALUDE_root_ratio_equality_l4058_405812


namespace NUMINAMATH_CALUDE_circle_radius_problem_l4058_405832

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if two circles are congruent -/
def are_congruent (c1 c2 : Circle) : Prop :=
  c1.radius = c2.radius

theorem circle_radius_problem (A B C D : Circle) :
  are_externally_tangent A B ∧
  are_externally_tangent A C ∧
  are_externally_tangent B C ∧
  is_internally_tangent A D ∧
  is_internally_tangent B D ∧
  is_internally_tangent C D ∧
  are_congruent B C ∧
  A.radius = 1 ∧
  (let (x, y) := D.center; (x - A.center.1)^2 + (y - A.center.2)^2 = A.radius^2) →
  B.radius = 8/9 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l4058_405832


namespace NUMINAMATH_CALUDE_linda_coin_count_l4058_405859

/-- Represents the number of coins Linda has initially and receives from her mother -/
structure CoinCounts where
  initial_dimes : Nat
  initial_quarters : Nat
  initial_nickels : Nat
  additional_dimes : Nat
  additional_quarters : Nat

/-- Calculates the total number of coins Linda has -/
def totalCoins (counts : CoinCounts) : Nat :=
  counts.initial_dimes + counts.initial_quarters + counts.initial_nickels +
  counts.additional_dimes + counts.additional_quarters +
  2 * counts.initial_nickels

theorem linda_coin_count :
  let counts : CoinCounts := {
    initial_dimes := 2,
    initial_quarters := 6,
    initial_nickels := 5,
    additional_dimes := 2,
    additional_quarters := 10
  }
  totalCoins counts = 35 := by
  sorry

end NUMINAMATH_CALUDE_linda_coin_count_l4058_405859


namespace NUMINAMATH_CALUDE_probability_at_least_three_speak_l4058_405834

def probability_of_success : ℚ := 1 / 3

def number_of_trials : ℕ := 7

def minimum_successes : ℕ := 3

theorem probability_at_least_three_speak :
  (1 : ℚ) - (Finset.sum (Finset.range minimum_successes) (λ k =>
    (Nat.choose number_of_trials k : ℚ) *
    probability_of_success ^ k *
    (1 - probability_of_success) ^ (number_of_trials - k)))
  = 939 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_speak_l4058_405834


namespace NUMINAMATH_CALUDE_book_pages_total_l4058_405838

/-- A book with 5 chapters, each containing 111 pages, has a total of 555 pages. -/
theorem book_pages_total (num_chapters : ℕ) (pages_per_chapter : ℕ) :
  num_chapters = 5 → pages_per_chapter = 111 → num_chapters * pages_per_chapter = 555 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_total_l4058_405838


namespace NUMINAMATH_CALUDE_train_length_calculation_l4058_405875

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed_kmh : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed_kmh = 90 →
  time_to_cross = 9.679225661947045 →
  bridge_length = 132 →
  ∃ train_length : ℝ, abs (train_length - 109.98) < 0.01 ∧
    train_length = train_speed_kmh * (1000 / 3600) * time_to_cross - bridge_length :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l4058_405875


namespace NUMINAMATH_CALUDE_square_root_condition_l4058_405894

theorem square_root_condition (x : ℝ) : 
  Real.sqrt ((x - 1)^2) = x - 1 → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_condition_l4058_405894


namespace NUMINAMATH_CALUDE_food_bank_donation_l4058_405816

theorem food_bank_donation (first_week_donation : ℝ) : first_week_donation = 40 :=
  let second_week_donation := 2 * first_week_donation
  let total_donation := first_week_donation + second_week_donation
  let remaining_food := 36
  have h1 : remaining_food = 0.3 * total_donation := by sorry
  have h2 : 36 = 0.3 * (3 * first_week_donation) := by sorry
  have h3 : first_week_donation = 36 / 0.9 := by sorry
  sorry

#check food_bank_donation

end NUMINAMATH_CALUDE_food_bank_donation_l4058_405816


namespace NUMINAMATH_CALUDE_mrs_hilt_bug_count_l4058_405842

theorem mrs_hilt_bug_count (flowers_per_bug : ℕ) (total_flowers : ℕ) (num_bugs : ℕ) : 
  flowers_per_bug = 2 →
  total_flowers = 6 →
  num_bugs * flowers_per_bug = total_flowers →
  num_bugs = 3 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_bug_count_l4058_405842


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4058_405803

theorem quadratic_inequality_solution (x : ℝ) :
  (2 * x - 1) * (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4058_405803


namespace NUMINAMATH_CALUDE_parentheses_removal_correct_l4058_405883

theorem parentheses_removal_correct (a b : ℤ) : -2*a + 3*(b - 1) = -2*a + 3*b - 3 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_correct_l4058_405883


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l4058_405871

theorem final_sum_after_operations (x y D : ℝ) (h : x - y = D) :
  4 * ((x - 5) + (y - 5)) = 4 * (x + y) - 40 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l4058_405871


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l4058_405811

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (product_sum_condition : a * b + a * c + b * c = -6)
  (product_condition : a * b * c = -3) :
  a^3 + b^3 + c^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l4058_405811


namespace NUMINAMATH_CALUDE_acute_angle_specific_circles_l4058_405841

/-- The acute angle formed by two lines intersecting three concentric circles -/
def acute_angle_concentric_circles (r1 r2 r3 : ℝ) (shaded_ratio : ℝ) : ℝ :=
  sorry

/-- The theorem stating the acute angle for the given problem -/
theorem acute_angle_specific_circles :
  acute_angle_concentric_circles 5 3 1 (10/17) = 107/459 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_specific_circles_l4058_405841


namespace NUMINAMATH_CALUDE_solution_set_f_geq_x_min_value_a_min_a_is_three_l4058_405837

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| - |2*x + 3|

-- Theorem for part (I)
theorem solution_set_f_geq_x : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≥ 4/5} := by sorry

-- Theorem for part (II)
theorem min_value_a (m : ℝ) (h : m > 0) :
  (∀ (x y : ℝ), f x ≤ m^y + a/m^y) → a ≥ 3 := by sorry

-- Theorem for the minimum value of a
theorem min_a_is_three :
  ∃ (a : ℝ), a = 3 ∧ 
  (∀ (m : ℝ), m > 0 → ∀ (x y : ℝ), f x ≤ m^y + a/m^y) ∧
  (∀ (a' : ℝ), (∀ (m : ℝ), m > 0 → ∀ (x y : ℝ), f x ≤ m^y + a'/m^y) → a' ≥ a) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_x_min_value_a_min_a_is_three_l4058_405837


namespace NUMINAMATH_CALUDE_mary_shirts_left_l4058_405898

/-- Calculates the number of shirts Mary has left after giving away fractions of each color --/
def shirts_left (blue brown red yellow green : ℕ) : ℕ :=
  let blue_left := blue - (4 * blue / 5)
  let brown_left := brown - (5 * brown / 6)
  let red_left := red - (2 * red / 3)
  let yellow_left := yellow - (3 * yellow / 4)
  let green_left := green - (green / 3)
  blue_left + brown_left + red_left + yellow_left + green_left

/-- The theorem stating that Mary has 45 shirts left --/
theorem mary_shirts_left :
  shirts_left 35 48 27 36 18 = 45 := by sorry

end NUMINAMATH_CALUDE_mary_shirts_left_l4058_405898


namespace NUMINAMATH_CALUDE_tobys_change_is_seven_l4058_405896

/-- Represents the dining scenario and calculates Toby's change --/
def tobys_change (cheeseburger_price : ℚ) (milkshake_price : ℚ) (coke_price : ℚ) 
                 (fries_price : ℚ) (cookie_price : ℚ) (tax : ℚ) 
                 (toby_initial_money : ℚ) : ℚ :=
  let total_cost := 2 * cheeseburger_price + milkshake_price + coke_price + 
                    fries_price + 3 * cookie_price + tax
  let toby_share := total_cost / 2
  toby_initial_money - toby_share

/-- Theorem stating that Toby's change is $7.00 --/
theorem tobys_change_is_seven : 
  tobys_change 3.65 2 1 4 0.5 0.2 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tobys_change_is_seven_l4058_405896


namespace NUMINAMATH_CALUDE_two_numbers_product_sum_l4058_405848

theorem two_numbers_product_sum (x y : ℕ) : 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range 38) id) - x - y = x * y ∧
  y - x = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_product_sum_l4058_405848


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4058_405863

/-- For any natural number n, the polynomial 
    x^(2n) - n^2 * x^(n+1) + 2(n^2 - 1) * x^n + 1 - n^2 * x^(n-1) 
    is divisible by (x-1)^3. -/
theorem polynomial_divisibility (n : ℕ) : 
  ∃ q : Polynomial ℚ, (X : Polynomial ℚ)^(2*n) - n^2 * X^(n+1) + 2*(n^2 - 1) * X^n + 1 - n^2 * X^(n-1) = 
    (X - 1)^3 * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4058_405863


namespace NUMINAMATH_CALUDE_cubic_projection_equality_l4058_405818

/-- A cubic function -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem: For a cubic function and two horizontal lines intersecting it, 
    the difference between the middle x-coordinates equals the sum of the 
    differences between the outer x-coordinates. -/
theorem cubic_projection_equality 
  (a b c d : ℝ) 
  (x₁ x₂ x₃ X₁ X₂ X₃ : ℝ) 
  (y₁ y₂ Y₁ Y₂ : ℝ) 
  (h₁ : x₁ < x₂) (h₂ : x₂ < x₃) 
  (h₃ : X₁ < X₂) (h₄ : X₂ < X₃) 
  (h₅ : cubic_function a b c d x₁ = y₁) 
  (h₆ : cubic_function a b c d x₂ = y₁) 
  (h₇ : cubic_function a b c d x₃ = y₁) 
  (h₈ : cubic_function a b c d X₁ = Y₁) 
  (h₉ : cubic_function a b c d X₂ = Y₁) 
  (h₁₀ : cubic_function a b c d X₃ = Y₁) :
  x₂ - X₂ = (X₁ - x₁) + (X₃ - x₃) := by sorry

end NUMINAMATH_CALUDE_cubic_projection_equality_l4058_405818


namespace NUMINAMATH_CALUDE_range_of_P_l4058_405882

theorem range_of_P (x y : ℝ) (h : x^2/3 + y^2 = 1) : 
  2 ≤ |2*x + y - 4| + |4 - x - 2*y| ∧ |2*x + y - 4| + |4 - x - 2*y| ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_P_l4058_405882


namespace NUMINAMATH_CALUDE_money_sum_l4058_405800

theorem money_sum (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 320) (h3 : C = 20) :
  A + B + C = 500 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l4058_405800


namespace NUMINAMATH_CALUDE_geometric_progression_terms_l4058_405895

/-- A finite geometric progression with first term 3, second term 12, and last term 3072 has 6 terms -/
theorem geometric_progression_terms : 
  ∀ (b : ℕ → ℝ), 
    b 1 = 3 → 
    b 2 = 12 → 
    (∃ n : ℕ, n > 2 ∧ b n = 3072 ∧ ∀ k : ℕ, 1 < k → k < n → b k / b (k-1) = b 2 / b 1) →
    ∃ n : ℕ, n = 6 ∧ b n = 3072 ∧ ∀ k : ℕ, 1 < k → k < n → b k / b (k-1) = b 2 / b 1 :=
by sorry


end NUMINAMATH_CALUDE_geometric_progression_terms_l4058_405895


namespace NUMINAMATH_CALUDE_games_given_solution_l4058_405829

/-- The number of games Henry gave to Neil -/
def games_given : ℕ := sorry

/-- Henry's initial number of games -/
def henry_initial : ℕ := 58

/-- Neil's initial number of games -/
def neil_initial : ℕ := 7

theorem games_given_solution :
  (henry_initial - games_given = 4 * (neil_initial + games_given)) ∧
  games_given = 6 := by sorry

end NUMINAMATH_CALUDE_games_given_solution_l4058_405829


namespace NUMINAMATH_CALUDE_line_passes_through_second_and_fourth_quadrants_l4058_405815

/-- A line with equation y = -2x + b (where b is a constant) always passes through the second and fourth quadrants. -/
theorem line_passes_through_second_and_fourth_quadrants (b : ℝ) :
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    (x₁ < 0 ∧ y₁ > 0 ∧ y₁ = -2*x₁ + b) ∧ 
    (x₂ > 0 ∧ y₂ < 0 ∧ y₂ = -2*x₂ + b) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_second_and_fourth_quadrants_l4058_405815


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4058_405833

theorem least_subtraction_for_divisibility : ∃ (n : ℕ), n = 8 ∧
  20 ∣ (50248 - n) ∧
  ∀ (m : ℕ), m < n → ¬(20 ∣ (50248 - m)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l4058_405833


namespace NUMINAMATH_CALUDE_exists_triangle_with_different_colors_l4058_405821

/-- A color type representing the three possible colors of vertices -/
inductive Color
  | A
  | B
  | C

/-- A graph representing the triangulation -/
structure Graph (α : Type) where
  V : Set α
  E : Set (α × α)

/-- A coloring function that assigns a color to each vertex -/
def Coloring (α : Type) := α → Color

/-- A predicate to check if three vertices form a triangle in the graph -/
def IsTriangle {α : Type} (G : Graph α) (a b c : α) : Prop :=
  a ∈ G.V ∧ b ∈ G.V ∧ c ∈ G.V ∧
  (a, b) ∈ G.E ∧ (b, c) ∈ G.E ∧ (c, a) ∈ G.E

/-- The main theorem statement -/
theorem exists_triangle_with_different_colors {α : Type} (G : Graph α) (f : Coloring α)
  (hA : ∃ a ∈ G.V, f a = Color.A)
  (hB : ∃ b ∈ G.V, f b = Color.B)
  (hC : ∃ c ∈ G.V, f c = Color.C) :
  ∃ x y z : α, IsTriangle G x y z ∧ f x ≠ f y ∧ f y ≠ f z ∧ f z ≠ f x :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_with_different_colors_l4058_405821


namespace NUMINAMATH_CALUDE_sum_expression_l4058_405861

theorem sum_expression (x y z k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) :
  x + y + z = (4 + 3 * k) * x := by
  sorry

end NUMINAMATH_CALUDE_sum_expression_l4058_405861


namespace NUMINAMATH_CALUDE_intersection_of_P_and_M_l4058_405878

-- Define the sets P and M
def P : Set ℝ := {y | ∃ x, y = x^2 - 6*x + 10}
def M : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 8}

-- State the theorem
theorem intersection_of_P_and_M : P ∩ M = {y | 1 ≤ y ∧ y ≤ 9} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_M_l4058_405878


namespace NUMINAMATH_CALUDE_max_sum_squared_integers_l4058_405828

theorem max_sum_squared_integers (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  i + j + k ≤ 77 := by
sorry

end NUMINAMATH_CALUDE_max_sum_squared_integers_l4058_405828


namespace NUMINAMATH_CALUDE_horner_method_for_f_l4058_405874

def f (x : ℝ) : ℝ := x^6 + 2*x^5 + 4*x^3 + 5*x^2 + 6*x + 12

theorem horner_method_for_f :
  f 3 = 588 := by sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l4058_405874


namespace NUMINAMATH_CALUDE_billy_horses_count_l4058_405850

/-- The number of horses Billy has -/
def num_horses : ℕ := 4

/-- The amount of oats (in pounds) each horse eats per feeding -/
def oats_per_feeding : ℕ := 4

/-- The number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- The number of days Billy needs to feed his horses -/
def days_to_feed : ℕ := 3

/-- The total amount of oats (in pounds) Billy needs for all his horses for the given days -/
def total_oats_needed : ℕ := 96

theorem billy_horses_count : 
  num_horses * oats_per_feeding * feedings_per_day * days_to_feed = total_oats_needed :=
sorry

end NUMINAMATH_CALUDE_billy_horses_count_l4058_405850


namespace NUMINAMATH_CALUDE_sam_seashells_l4058_405892

/-- Given that Mary found 47 seashells and the total number of seashells
    found by Sam and Mary is 65, prove that Sam found 18 seashells. -/
theorem sam_seashells (mary_seashells : ℕ) (total_seashells : ℕ)
    (h1 : mary_seashells = 47)
    (h2 : total_seashells = 65) :
    total_seashells - mary_seashells = 18 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l4058_405892


namespace NUMINAMATH_CALUDE_inequality_proof_l4058_405806

open Real

noncomputable def f (a x : ℝ) : ℝ := (a/2) * x^2 - (a-2) * x - 2 * x * log x

theorem inequality_proof (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : 0 < a ∧ a < 2)
  (h_x : x₁ < x₂)
  (h_zeros : ∃ (x : ℝ), x = x₁ ∨ x = x₂ ∧ (deriv (f a)) x = 0) :
  x₂ - x₁ > 4/a - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4058_405806


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l4058_405891

/-- Given that N(4,10) is the midpoint of CD and C(14,6), prove that the sum of D's coordinates is 8 -/
theorem midpoint_coordinate_sum (N C D : ℝ × ℝ) : 
  N = (4, 10) → 
  C = (14, 6) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l4058_405891


namespace NUMINAMATH_CALUDE_smallest_coin_arrangement_l4058_405823

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The number of proper divisors of a positive integer greater than 2 -/
def num_proper_divisors_gt_2 (n : ℕ+) : ℕ := sorry

/-- Checks if all divisors d of n where 2 < d < n, n/d is an integer -/
def all_divisors_divide (n : ℕ+) : Prop := sorry

theorem smallest_coin_arrangement :
  ∃ (n : ℕ+), num_divisors n = 19 ∧ 
              num_proper_divisors_gt_2 n = 17 ∧ 
              all_divisors_divide n ∧
              (∀ m : ℕ+, m < n → 
                (num_divisors m ≠ 19 ∨ 
                 num_proper_divisors_gt_2 m ≠ 17 ∨ 
                 ¬all_divisors_divide m)) ∧
              n = 2700 := by sorry

end NUMINAMATH_CALUDE_smallest_coin_arrangement_l4058_405823


namespace NUMINAMATH_CALUDE_distinct_positions_selection_l4058_405845

theorem distinct_positions_selection (n : ℕ) (k : ℕ) (ways : ℕ) : 
  n = 12 → k = 2 → ways = 132 → ways = n * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_distinct_positions_selection_l4058_405845


namespace NUMINAMATH_CALUDE_convergence_iff_cauchy_l4058_405846

/-- A sequence of real numbers -/
def RealSequence := ℕ → ℝ

/-- Convergence of a sequence -/
def converges (x : RealSequence) : Prop :=
  ∃ (l : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - l| < ε

/-- Cauchy criterion for a sequence -/
def is_cauchy (x : RealSequence) : Prop :=
  ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N → |x m - x n| < ε

/-- Theorem: A sequence of real numbers converges if and only if it satisfies the Cauchy criterion -/
theorem convergence_iff_cauchy (x : RealSequence) :
  converges x ↔ is_cauchy x :=
sorry

end NUMINAMATH_CALUDE_convergence_iff_cauchy_l4058_405846


namespace NUMINAMATH_CALUDE_cookies_with_four_cups_l4058_405819

/-- Represents the number of cookies that can be made with a given amount of flour,
    maintaining a constant ratio of flour to sugar. -/
def cookies_made (flour : ℚ) : ℚ :=
  24 * flour / 3

/-- The ratio of flour to sugar remains constant. -/
axiom constant_ratio : ∀ (f : ℚ), cookies_made f / f = 24 / 3

theorem cookies_with_four_cups :
  cookies_made 4 = 128 :=
sorry

end NUMINAMATH_CALUDE_cookies_with_four_cups_l4058_405819


namespace NUMINAMATH_CALUDE_lottery_tickets_theorem_lottery_tickets_minimality_l4058_405822

/-- The probability of winning with a single lottery ticket -/
def p : ℝ := 0.01

/-- The desired probability of winning at least once -/
def desired_prob : ℝ := 0.95

/-- The number of tickets needed to achieve the desired probability -/
def n : ℕ := 300

/-- Theorem stating that n tickets are sufficient to achieve the desired probability -/
theorem lottery_tickets_theorem :
  1 - (1 - p) ^ n ≥ desired_prob :=
sorry

/-- Theorem stating that n-1 tickets are not sufficient to achieve the desired probability -/
theorem lottery_tickets_minimality :
  1 - (1 - p) ^ (n - 1) < desired_prob :=
sorry

end NUMINAMATH_CALUDE_lottery_tickets_theorem_lottery_tickets_minimality_l4058_405822


namespace NUMINAMATH_CALUDE_craft_store_sales_l4058_405839

theorem craft_store_sales (total_sales : ℕ) : 
  (total_sales / 3 : ℕ) + (total_sales / 4 : ℕ) + 15 = total_sales → 
  total_sales = 36 := by
  sorry

end NUMINAMATH_CALUDE_craft_store_sales_l4058_405839


namespace NUMINAMATH_CALUDE_range_of_a_l4058_405801

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 1 > 0) ∨ (∃ x : ℝ, a*x^2 + 2 ≤ 0) = False →
  a ∈ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l4058_405801


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l4058_405860

/-- An isosceles right triangle with perimeter 4 + 4√2 has a hypotenuse of length 4. -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → -- Side length is positive
  c > 0 → -- Hypotenuse length is positive
  2 * a + c = 4 + 4 * Real.sqrt 2 → -- Perimeter condition
  c = a * Real.sqrt 2 → -- Isosceles right triangle condition
  c = 4 := by
sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l4058_405860


namespace NUMINAMATH_CALUDE_solve_for_w_l4058_405804

theorem solve_for_w (u v w : ℝ) 
  (eq1 : 10 * u + 8 * v + 5 * w = 160)
  (eq2 : v = u + 3)
  (eq3 : w = 2 * v) : 
  w = 13.5714 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_w_l4058_405804


namespace NUMINAMATH_CALUDE_articles_produced_is_y_l4058_405826

/-- Given that x men working x hours a day for x days produce x articles,
    this function calculates the number of articles produced by x men
    working x hours a day for y days. -/
def articles_produced (x y : ℝ) : ℝ :=
  y

/-- Theorem stating that the number of articles produced is y -/
theorem articles_produced_is_y (x y : ℝ) (h : x > 0) :
  articles_produced x y = y :=
by sorry

end NUMINAMATH_CALUDE_articles_produced_is_y_l4058_405826


namespace NUMINAMATH_CALUDE_laundry_problem_solution_l4058_405852

/-- Represents the laundry shop scenario --/
structure LaundryShop where
  price_per_kilo : ℝ
  kilos_two_days_ago : ℝ
  total_earnings : ℝ

/-- Calculates the total kilos of laundry for three days --/
def total_kilos (shop : LaundryShop) : ℝ :=
  shop.kilos_two_days_ago + 
  (shop.kilos_two_days_ago + 5) + 
  2 * (shop.kilos_two_days_ago + 5)

/-- Theorem stating the solution to the laundry problem --/
theorem laundry_problem_solution (shop : LaundryShop) 
  (h1 : shop.price_per_kilo = 2)
  (h2 : shop.total_earnings = 70) :
  shop.kilos_two_days_ago = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_laundry_problem_solution_l4058_405852


namespace NUMINAMATH_CALUDE_percentage_difference_l4058_405810

theorem percentage_difference (p j t : ℝ) 
  (hj : j = 0.75 * p) 
  (ht : t = 0.9375 * p) : 
  (t - j) / t = 0.2 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l4058_405810


namespace NUMINAMATH_CALUDE_increase_by_percentage_l4058_405886

/-- Prove that increasing 500 by 30% results in 650. -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 500 → percentage = 30 → result = initial * (1 + percentage / 100) → result = 650 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l4058_405886


namespace NUMINAMATH_CALUDE_quadratic_transform_coefficient_l4058_405873

/-- Given a quadratic equation 7x - 3 = 2x², prove that when transformed
    to general form ax² + bx + c = 0 with c = 3, the coefficient of x (b) is -7 -/
theorem quadratic_transform_coefficient (x : ℝ) : 
  (7 * x - 3 = 2 * x^2) → 
  ∃ (a b : ℝ), (a * x^2 + b * x + 3 = 0) ∧ (b = -7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_transform_coefficient_l4058_405873


namespace NUMINAMATH_CALUDE_digit_one_more_frequent_than_zero_l4058_405884

def concatenated_sequence (n : ℕ) : String :=
  String.join (List.map toString (List.range n))

def count_digit (s : String) (d : Char) : ℕ :=
  s.toList.filter (· = d) |>.length

theorem digit_one_more_frequent_than_zero (n : ℕ) :
  count_digit (concatenated_sequence n) '1' > count_digit (concatenated_sequence n) '0' :=
sorry

end NUMINAMATH_CALUDE_digit_one_more_frequent_than_zero_l4058_405884


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l4058_405807

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 6) : 
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l4058_405807


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l4058_405865

/-- The cost of the cheaper rice variety in Rs per kg -/
def cost_cheap : ℚ := 9/2

/-- The cost of the more expensive rice variety in Rs per kg -/
def cost_expensive : ℚ := 35/4

/-- The ratio of cheaper rice to more expensive rice in the mixture -/
def mixture_ratio : ℚ := 5/12

/-- The cost of the mixture per kg -/
def mixture_cost : ℚ := 23/4

theorem rice_mixture_cost :
  (cost_cheap * mixture_ratio + cost_expensive * 1) / (mixture_ratio + 1) = mixture_cost := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l4058_405865


namespace NUMINAMATH_CALUDE_green_brunette_percentage_is_54_l4058_405824

/-- Represents the hair and eye color distribution of an island's population -/
structure IslandPopulation where
  blueBrunettes : ℕ
  blueBlondes : ℕ
  greenBlondes : ℕ
  greenBrunettes : ℕ

/-- The proportion of brunettes among blue-eyed inhabitants is 65% -/
def blueBrunettesProportion (pop : IslandPopulation) : Prop :=
  (pop.blueBrunettes : ℚ) / (pop.blueBrunettes + pop.blueBlondes) = 13 / 20

/-- The proportion of blue-eyed among blondes is 70% -/
def blueBlondeProportion (pop : IslandPopulation) : Prop :=
  (pop.blueBlondes : ℚ) / (pop.blueBlondes + pop.greenBlondes) = 7 / 10

/-- The proportion of blondes among green-eyed inhabitants is 10% -/
def greenBlondeProportion (pop : IslandPopulation) : Prop :=
  (pop.greenBlondes : ℚ) / (pop.greenBlondes + pop.greenBrunettes) = 1 / 10

/-- The percentage of green-eyed brunettes in the total population -/
def greenBrunettePercentage (pop : IslandPopulation) : ℚ :=
  (pop.greenBrunettes : ℚ) / (pop.blueBrunettes + pop.blueBlondes + pop.greenBlondes + pop.greenBrunettes) * 100

/-- Theorem stating that the percentage of green-eyed brunettes is 54% -/
theorem green_brunette_percentage_is_54 (pop : IslandPopulation) :
  blueBrunettesProportion pop → blueBlondeProportion pop → greenBlondeProportion pop →
  greenBrunettePercentage pop = 54 := by
  sorry

end NUMINAMATH_CALUDE_green_brunette_percentage_is_54_l4058_405824
