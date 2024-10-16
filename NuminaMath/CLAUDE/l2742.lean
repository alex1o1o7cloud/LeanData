import Mathlib

namespace NUMINAMATH_CALUDE_coral_reading_pages_l2742_274238

def pages_night1 : ℕ := 30

def pages_night2 : ℕ := 2 * pages_night1 - 2

def pages_night3 : ℕ := pages_night1 + pages_night2 + 3

def total_pages : ℕ := pages_night1 + pages_night2 + pages_night3

theorem coral_reading_pages : total_pages = 179 := by
  sorry

end NUMINAMATH_CALUDE_coral_reading_pages_l2742_274238


namespace NUMINAMATH_CALUDE_best_fitting_model_l2742_274227

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Given four regression models, proves that the model with the highest R² has the best fitting effect -/
theorem best_fitting_model
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  model1.r_squared = max model1.r_squared (max model2.r_squared (max model3.r_squared model4.r_squared)) :=
sorry

end NUMINAMATH_CALUDE_best_fitting_model_l2742_274227


namespace NUMINAMATH_CALUDE_joans_grilled_cheese_sandwiches_l2742_274255

/-- Calculates the number of grilled cheese sandwiches Joan makes given the conditions -/
theorem joans_grilled_cheese_sandwiches 
  (total_cheese : ℕ) 
  (ham_sandwiches : ℕ) 
  (cheese_per_ham : ℕ) 
  (cheese_per_grilled : ℕ) 
  (h1 : total_cheese = 50)
  (h2 : ham_sandwiches = 10)
  (h3 : cheese_per_ham = 2)
  (h4 : cheese_per_grilled = 3) :
  (total_cheese - ham_sandwiches * cheese_per_ham) / cheese_per_grilled = 10 := by
  sorry

end NUMINAMATH_CALUDE_joans_grilled_cheese_sandwiches_l2742_274255


namespace NUMINAMATH_CALUDE_expression_equality_l2742_274204

theorem expression_equality : 484 + 2*(22)*(5) + 25 = 729 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2742_274204


namespace NUMINAMATH_CALUDE_candies_sum_l2742_274203

/-- The number of candies Linda has -/
def linda_candies : ℕ := 34

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 28

/-- The total number of candies Linda and Chloe have together -/
def total_candies : ℕ := linda_candies + chloe_candies

theorem candies_sum : total_candies = 62 := by
  sorry

end NUMINAMATH_CALUDE_candies_sum_l2742_274203


namespace NUMINAMATH_CALUDE_donation_amount_l2742_274231

def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def sam_stuffed_animals : ℕ := barbara_stuffed_animals + 5

def barbara_price : ℚ := 2
def trish_price : ℚ := (3 : ℚ) / 2
def sam_price : ℚ := (5 : ℚ) / 2

def total_donation : ℚ := 
  barbara_stuffed_animals * barbara_price + 
  trish_stuffed_animals * trish_price + 
  sam_stuffed_animals * sam_price

theorem donation_amount : total_donation = 80 := by
  sorry

end NUMINAMATH_CALUDE_donation_amount_l2742_274231


namespace NUMINAMATH_CALUDE_geometric_progression_sum_equality_l2742_274271

/-- Proves the equality for sums of geometric progression terms -/
theorem geometric_progression_sum_equality 
  (a q : ℝ) (n : ℕ) (h : q ≠ 1) :
  let S : ℕ → ℝ := λ k => a * (q^k - 1) / (q - 1)
  S n * (S (3*n) - S (2*n)) = (S (2*n) - S n)^2 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_equality_l2742_274271


namespace NUMINAMATH_CALUDE_solution_to_equation_l2742_274273

theorem solution_to_equation (z : ℝ) : 
  (z^2 - 5*z + 6)/(z-2) + (5*z^2 + 11*z - 32)/(5*z - 16) = 1 ↔ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2742_274273


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2742_274202

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →                       -- non-zero common difference
  a 1 = 1 →                     -- a_1 = 1
  (a 3)^2 = a 1 * a 13 →        -- a_1, a_3, a_13 form a geometric sequence
  d = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2742_274202


namespace NUMINAMATH_CALUDE_three_integer_chords_l2742_274229

/-- Represents a circle with a given radius and a point inside it. -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of chords with integer lengths that contain the given point. -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle with radius 13 and a point 5 units from the center,
    there are exactly 3 chords with integer lengths containing the point. -/
theorem three_integer_chords :
  let c := CircleWithPoint.mk 13 5
  countIntegerChords c = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_chords_l2742_274229


namespace NUMINAMATH_CALUDE_jump_rope_time_difference_l2742_274257

-- Define the jump rope times for each person
def cindy_time : ℕ := 12
def betsy_time : ℕ := cindy_time / 2
def tina_time : ℕ := betsy_time * 3

-- Theorem to prove
theorem jump_rope_time_difference : tina_time - cindy_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_jump_rope_time_difference_l2742_274257


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l2742_274288

theorem no_prime_sum_10003 : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 10003 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l2742_274288


namespace NUMINAMATH_CALUDE_max_m_value_l2742_274289

def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ m + 1}

theorem max_m_value (m : ℝ) (h : B m ⊆ A) : m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l2742_274289


namespace NUMINAMATH_CALUDE_film_product_unique_l2742_274299

/-- Represents the alphabet-to-number mapping -/
def letter_value (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- Calculates the product of letter values for a given string -/
def string_product (s : String) : Nat :=
  s.data.foldl (fun acc c => acc * letter_value c) 1

/-- Checks if a string is a valid four-letter combination (all uppercase letters) -/
def is_valid_combination (s : String) : Bool :=
  s.length = 4 && s.data.all (fun c => 'A' ≤ c && c ≤ 'Z')

/-- Theorem: The product of "FILM" is unique among all four-letter combinations -/
theorem film_product_unique :
  ∀ s : String, is_valid_combination s → s ≠ "FILM" →
  string_product s ≠ string_product "FILM" :=
sorry


end NUMINAMATH_CALUDE_film_product_unique_l2742_274299


namespace NUMINAMATH_CALUDE_max_trip_weight_is_750_l2742_274264

/-- Represents the number of crates on a trip -/
inductive NumCrates
  | three
  | four
  | five

/-- The minimum weight of a single crate in kg -/
def minCrateWeight : ℝ := 150

/-- Calculates the maximum weight of crates on a single trip -/
def maxTripWeight (n : NumCrates) : ℝ :=
  match n with
  | .three => 3 * minCrateWeight
  | .four => 4 * minCrateWeight
  | .five => 5 * minCrateWeight

/-- Theorem: The maximum weight of crates on a single trip is 750 kg -/
theorem max_trip_weight_is_750 :
  ∀ n : NumCrates, maxTripWeight n ≤ 750 ∧ ∃ m : NumCrates, maxTripWeight m = 750 :=
by sorry

end NUMINAMATH_CALUDE_max_trip_weight_is_750_l2742_274264


namespace NUMINAMATH_CALUDE_rocket_heights_l2742_274274

theorem rocket_heights (h1 : ℝ) (h2 : ℝ) (height1 : h1 = 500) (height2 : h2 = 2 * h1) :
  h1 + h2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_rocket_heights_l2742_274274


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2742_274272

/-- Represents a notebook with double-sided pages -/
structure Notebook where
  total_sheets : ℕ
  total_pages : ℕ
  pages_per_sheet : ℕ
  h_pages_per_sheet : pages_per_sheet = 2

/-- Calculates the average of remaining page numbers after borrowing sheets -/
def average_remaining_pages (nb : Notebook) (borrowed_sheets : ℕ) : ℚ :=
  let remaining_pages := nb.total_pages - borrowed_sheets * nb.pages_per_sheet
  let sum_remaining := (nb.total_pages * (nb.total_pages + 1) / 2) -
    (borrowed_sheets * nb.pages_per_sheet * (borrowed_sheets * nb.pages_per_sheet + 1) / 2)
  sum_remaining / remaining_pages

/-- Theorem stating that borrowing 12 sheets results in an average of 23 for remaining pages -/
theorem borrowed_sheets_theorem (nb : Notebook)
    (h_total_sheets : nb.total_sheets = 32)
    (h_total_pages : nb.total_pages = 64)
    (borrowed_sheets : ℕ)
    (h_borrowed : borrowed_sheets = 12) :
    average_remaining_pages nb borrowed_sheets = 23 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2742_274272


namespace NUMINAMATH_CALUDE_cylinder_surface_area_from_hemisphere_l2742_274213

/-- Given a hemisphere with total surface area Q and a cylinder with the same base and volume,
    prove that the total surface area of the cylinder is (10/9)Q. -/
theorem cylinder_surface_area_from_hemisphere (Q : ℝ) (R : ℝ) (h : ℝ) :
  Q > 0 →  -- Ensure Q is positive
  Q = 3 * Real.pi * R^2 →  -- Total surface area of hemisphere
  h = (2/3) * R →  -- Height of cylinder with same volume
  (2 * Real.pi * R^2 + 2 * Real.pi * R * h) = (10/9) * Q := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_from_hemisphere_l2742_274213


namespace NUMINAMATH_CALUDE_shortest_diagonal_probability_l2742_274205

/-- The number of sides in the regular polygon -/
def n : ℕ := 20

/-- The total number of diagonals in the polygon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in the polygon -/
def shortest_diagonals : ℕ := n

/-- The probability of selecting a shortest diagonal -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem shortest_diagonal_probability :
  probability = 2 / 17 := by sorry

end NUMINAMATH_CALUDE_shortest_diagonal_probability_l2742_274205


namespace NUMINAMATH_CALUDE_complex_expression_modulus_l2742_274246

theorem complex_expression_modulus : 
  let z : ℂ := (Complex.mk (Real.sqrt 3) (Real.sqrt 2)) * 
               (Complex.mk (Real.sqrt 5) (Real.sqrt 2)) * 
               (Complex.mk (Real.sqrt 5) (Real.sqrt 3)) / 
               ((Complex.mk (Real.sqrt 2) (-Real.sqrt 3)) * 
                (Complex.mk (Real.sqrt 2) (-Real.sqrt 5)))
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_modulus_l2742_274246


namespace NUMINAMATH_CALUDE_circle_condition_l2742_274244

def is_circle (m : ℤ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), 
    x^2 + y^2 + m*x - m*y + 2 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

theorem circle_condition (m : ℤ) : 
  m ∈ ({0, 1, 2, 3} : Set ℤ) →
  (is_circle m ↔ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l2742_274244


namespace NUMINAMATH_CALUDE_power_six_mod_72_l2742_274225

theorem power_six_mod_72 : 6^700 % 72 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_six_mod_72_l2742_274225


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2742_274239

-- Define the polynomials
def p (z : ℝ) : ℝ := 3 * z^2 + 4 * z - 5
def q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the expanded result
def expanded_result (z : ℝ) : ℝ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- Theorem statement
theorem polynomial_expansion (z : ℝ) : p z * q z = expanded_result z := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2742_274239


namespace NUMINAMATH_CALUDE_equation_solution_l2742_274278

theorem equation_solution : ∃! x : ℤ, 27474 + x + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2742_274278


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2742_274210

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a carton with base and top dimensions -/
structure Carton where
  base : Dimensions
  top : Dimensions
  height : ℕ

/-- Represents a soap box with its dimensions and weight -/
structure SoapBox where
  dimensions : Dimensions
  weight : ℕ

def carton : Carton := {
  base := { width := 25, length := 42, height := 0 },
  top := { width := 20, length := 35, height := 0 },
  height := 60
}

def soapBox : SoapBox := {
  dimensions := { width := 7, length := 6, height := 10 },
  weight := 3
}

def maxWeight : ℕ := 150

theorem max_soap_boxes_in_carton :
  let spaceConstraint := (carton.top.width / soapBox.dimensions.width) *
                         (carton.top.length / soapBox.dimensions.length) *
                         (carton.height / soapBox.dimensions.height)
  let weightConstraint := maxWeight / soapBox.weight
  min spaceConstraint weightConstraint = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2742_274210


namespace NUMINAMATH_CALUDE_cakes_served_yesterday_proof_l2742_274235

def cakes_served_yesterday (lunch_today dinner_today total : ℕ) : ℕ :=
  total - (lunch_today + dinner_today)

theorem cakes_served_yesterday_proof :
  cakes_served_yesterday 5 6 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_yesterday_proof_l2742_274235


namespace NUMINAMATH_CALUDE_periodic_even_function_extension_l2742_274286

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = Real.log (1 - x) / Real.log (1/2)) :
  ∀ x ∈ Set.Ioo 1 2, f x = Real.log (x - 1) / Real.log (1/2) := by
sorry

end NUMINAMATH_CALUDE_periodic_even_function_extension_l2742_274286


namespace NUMINAMATH_CALUDE_wire_cutting_l2742_274268

/-- Given a wire of length 50 feet cut into three pieces, prove the lengths of the pieces. -/
theorem wire_cutting (x : ℝ) 
  (h1 : x + (x + 2) + (2*x - 3) = 50) -- Total length equation
  (h2 : x > 0) -- Ensure positive length
  : x = 12.75 ∧ x + 2 = 14.75 ∧ 2*x - 3 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2742_274268


namespace NUMINAMATH_CALUDE_portias_university_students_l2742_274232

theorem portias_university_students :
  ∀ (p l c : ℕ),
  p = 4 * l →
  c = l / 2 →
  p + l + c = 4500 →
  p = 3273 :=
by
  sorry

end NUMINAMATH_CALUDE_portias_university_students_l2742_274232


namespace NUMINAMATH_CALUDE_rational_equation_zero_solution_l2742_274261

theorem rational_equation_zero_solution (x y z : ℚ) :
  x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_zero_solution_l2742_274261


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2742_274270

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 27)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 2) :
  badminton + tennis - total + neither = 11 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2742_274270


namespace NUMINAMATH_CALUDE_dryer_ball_savings_l2742_274277

/-- Calculates the savings from using wool dryer balls instead of dryer sheets over two years -/
theorem dryer_ball_savings :
  let loads_per_month : ℕ := 4 + 5 + 6 + 7
  let loads_per_year : ℕ := loads_per_month * 12
  let sheets_per_box : ℕ := 104
  let boxes_per_year : ℕ := (loads_per_year + sheets_per_box - 1) / sheets_per_box
  let initial_box_price : ℝ := 5.50
  let price_increase_rate : ℝ := 0.025
  let dryer_ball_price : ℝ := 15

  let first_year_cost : ℝ := boxes_per_year * initial_box_price
  let second_year_cost : ℝ := boxes_per_year * (initial_box_price * (1 + price_increase_rate))
  let total_sheet_cost : ℝ := first_year_cost + second_year_cost

  let savings : ℝ := total_sheet_cost - dryer_ball_price

  savings = 18.4125 := by sorry

end NUMINAMATH_CALUDE_dryer_ball_savings_l2742_274277


namespace NUMINAMATH_CALUDE_abs_w_equals_3_fourth_root_2_l2742_274200

-- Define w as a complex number
variable (w : ℂ)

-- State the theorem
theorem abs_w_equals_3_fourth_root_2 (h : w^2 = -18 + 18*I) : 
  Complex.abs w = 3 * (2 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_abs_w_equals_3_fourth_root_2_l2742_274200


namespace NUMINAMATH_CALUDE_f_decreasing_interval_f_extremum_at_3_l2742_274280

/-- The function f(x) = 2x³ - 15x² + 36x - 24 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 15 * x^2 + 36 * x - 24

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 30 * x + 36

/-- Theorem stating that the decreasing interval of f is (2, 3) -/
theorem f_decreasing_interval :
  ∀ x : ℝ, (2 < x ∧ x < 3) ↔ (f' x < 0) :=
sorry

/-- Theorem stating that f has an extremum at x = 3 -/
theorem f_extremum_at_3 :
  f' 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_f_extremum_at_3_l2742_274280


namespace NUMINAMATH_CALUDE_lent_amount_proof_l2742_274228

/-- The amount of money (in Rs.) that A lends to B -/
def lent_amount : ℝ := 1500

/-- The interest rate difference (in decimal form) between B's lending and borrowing rates -/
def interest_rate_diff : ℝ := 0.015

/-- The number of years for which the loan is considered -/
def years : ℝ := 3

/-- B's total gain (in Rs.) over the loan period -/
def total_gain : ℝ := 67.5

theorem lent_amount_proof :
  lent_amount * interest_rate_diff * years = total_gain :=
by sorry

end NUMINAMATH_CALUDE_lent_amount_proof_l2742_274228


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2742_274251

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 * a 5 * a 6 = 8 →
  a 2 = 1 →
  a 2 + a 5 + a 8 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2742_274251


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l2742_274208

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 8*a*b) :
  |((a+b)/(a-b))| = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l2742_274208


namespace NUMINAMATH_CALUDE_second_red_probability_l2742_274215

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ
  black : ℕ
  red : ℕ
  green : ℕ

/-- The probability of drawing a red marble as the second marble -/
def second_red_prob (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.white + bagA.black
  let total_B := bagB.red + bagB.green
  let total_C := bagC.red + bagC.green
  let prob_white_A := bagA.white / total_A
  let prob_black_A := bagA.black / total_A
  let prob_red_B := bagB.red / total_B
  let prob_red_C := bagC.red / total_C
  prob_white_A * prob_red_B + prob_black_A * prob_red_C

theorem second_red_probability :
  let bagA : Bag := { white := 4, black := 5, red := 0, green := 0 }
  let bagB : Bag := { white := 0, black := 0, red := 3, green := 7 }
  let bagC : Bag := { white := 0, black := 0, red := 5, green := 3 }
  second_red_prob bagA bagB bagC = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_second_red_probability_l2742_274215


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2742_274259

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2742_274259


namespace NUMINAMATH_CALUDE_linear_function_intersection_l2742_274279

theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 3 = 0 ∧ x^2 = 36) → (k = 1/2 ∨ k = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l2742_274279


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2742_274265

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n β) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l2742_274265


namespace NUMINAMATH_CALUDE_larger_integer_value_l2742_274216

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  max a b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l2742_274216


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2742_274240

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let r₁ := (-(-10) + Real.sqrt ((-10)^2 - 4*2*3)) / (2*2)
  let r₂ := (-(-10) - Real.sqrt ((-10)^2 - 4*2*3)) / (2*2)
  r₁ + r₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2742_274240


namespace NUMINAMATH_CALUDE_line_slope_angle_l2742_274209

theorem line_slope_angle (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y + 2 = 0) → -- line equation
  (Real.tan (45 * Real.pi / 180) = -1 / a) → -- slope angle is 45°
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_angle_l2742_274209


namespace NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l2742_274247

theorem trillion_equals_ten_to_sixteen :
  let ten_thousand : ℕ := 10^4
  let hundred_million : ℕ := 10^8
  let trillion : ℕ := ten_thousand * ten_thousand * hundred_million
  trillion = 10^16 := by
  sorry

end NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l2742_274247


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2742_274292

theorem rectangle_area_change 
  (l w : ℝ) 
  (h_pos_l : l > 0) 
  (h_pos_w : w > 0) : 
  let new_length := 1.1 * l
  let new_width := 0.9 * w
  let new_area := new_length * new_width
  let original_area := l * w
  new_area / original_area = 0.99 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2742_274292


namespace NUMINAMATH_CALUDE_equal_face_parallelepiped_implies_rhombus_l2742_274241

/-- A parallelepiped with equal parallelogram faces -/
structure EqualFaceParallelepiped where
  /-- The length of the first edge -/
  a : ℝ
  /-- The length of the second edge -/
  b : ℝ
  /-- The length of the third edge -/
  c : ℝ
  /-- All edges have positive length -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  /-- All faces have equal area -/
  equal_faces : a * b = b * c ∧ b * c = a * c

/-- A rhombus is a quadrilateral with all sides equal -/
def is_rhombus (s₁ s₂ s₃ s₄ : ℝ) : Prop :=
  s₁ = s₂ ∧ s₂ = s₃ ∧ s₃ = s₄

/-- If all 6 faces of a parallelepiped are equal parallelograms, then they are rhombuses -/
theorem equal_face_parallelepiped_implies_rhombus (P : EqualFaceParallelepiped) :
  is_rhombus P.a P.a P.a P.a ∧
  is_rhombus P.b P.b P.b P.b ∧
  is_rhombus P.c P.c P.c P.c :=
sorry

end NUMINAMATH_CALUDE_equal_face_parallelepiped_implies_rhombus_l2742_274241


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2742_274218

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x - 1 = Real.sqrt (x - 1)

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l2742_274218


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2742_274290

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2742_274290


namespace NUMINAMATH_CALUDE_y_minus_x_equals_one_tenth_l2742_274252

-- Define the rounding function to the tenths place
def roundToTenths (x : ℚ) : ℚ := ⌊x * 10 + 1/2⌋ / 10

-- Define the given values
def a : ℚ := 545/100
def b : ℚ := 295/100
def c : ℚ := 374/100

-- Define x as the sum of a, b, and c rounded to tenths
def x : ℚ := roundToTenths (a + b + c)

-- Define y as the sum of a, b, and c individually rounded to tenths
def y : ℚ := roundToTenths a + roundToTenths b + roundToTenths c

-- State the theorem
theorem y_minus_x_equals_one_tenth : y - x = 1/10 := by sorry

end NUMINAMATH_CALUDE_y_minus_x_equals_one_tenth_l2742_274252


namespace NUMINAMATH_CALUDE_f_is_direct_proportion_l2742_274263

/-- Definition of direct proportion --/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function we want to prove is directly proportional --/
def f (x : ℝ) : ℝ := -0.1 * x

/-- Theorem stating that f is a direct proportion --/
theorem f_is_direct_proportion : is_direct_proportion f := by
  sorry

end NUMINAMATH_CALUDE_f_is_direct_proportion_l2742_274263


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2742_274256

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_sequence a q n * q

theorem geometric_sequence_sum (a q : ℝ) (h1 : geometric_sequence a q 0 + geometric_sequence a q 1 = 20) 
  (h2 : geometric_sequence a q 2 + geometric_sequence a q 3 = 60) : 
  geometric_sequence a q 4 + geometric_sequence a q 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2742_274256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2742_274253

theorem arithmetic_sequence_terms (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 10 → aₙ = 150 → d = 5 → aₙ = a₁ + (n - 1) * d → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2742_274253


namespace NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l2742_274283

/-- Proves that Ivy has 20 collectors edition dolls given the conditions -/
theorem ivy_collectors_edition_dolls (dina_dolls ivy_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : dina_dolls = 60) :
  (2 : ℚ) / 3 * ivy_dolls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l2742_274283


namespace NUMINAMATH_CALUDE_time_for_b_alone_l2742_274250

/-- Given that:
  1. It takes 'a' hours for A and B to complete the work together.
  2. It takes 'b' hours for A to complete the work alone.
  Prove that the time it takes B alone to complete the work is ab / (b - a) hours. -/
theorem time_for_b_alone (a b : ℝ) (h1 : a > 0) (h2 : b > a) : 
  (1 / a + 1 / (a * b / (b - a)) = 1) := by
sorry

end NUMINAMATH_CALUDE_time_for_b_alone_l2742_274250


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l2742_274230

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l2742_274230


namespace NUMINAMATH_CALUDE_parabola_equation_l2742_274297

/-- A parabola passing through two points on the x-axis -/
structure Parabola where
  a : ℝ
  b : ℝ
  eval : ℝ → ℝ := fun x => a * x^2 + b * x - 5

/-- The parabola passes through the points (-1,0) and (5,0) -/
def passes_through (p : Parabola) : Prop :=
  p.eval (-1) = 0 ∧ p.eval 5 = 0

/-- The theorem stating that the parabola passing through (-1,0) and (5,0) has the equation y = x² - 4x - 5 -/
theorem parabola_equation (p : Parabola) (h : passes_through p) :
  p.a = 1 ∧ p.b = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2742_274297


namespace NUMINAMATH_CALUDE_max_value_implies_a_range_l2742_274254

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 8

-- State the theorem
theorem max_value_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f x ≤ f a) →
  a ∈ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_range_l2742_274254


namespace NUMINAMATH_CALUDE_unique_triple_l2742_274221

theorem unique_triple : ∃! (x y z : ℕ+), 
  (z > 1) ∧ 
  ((y + 1 : ℕ) % x = 0) ∧ 
  ((z - 1 : ℕ) % y = 0) ∧ 
  ((x^2 + 1 : ℕ) % z = 0) ∧
  x = 1 ∧ y = 1 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_l2742_274221


namespace NUMINAMATH_CALUDE_max_value_system_l2742_274206

theorem max_value_system (x y z : ℝ) 
  (eq1 : x - y + z - 1 = 0)
  (eq2 : x * y + 2 * z^2 - 6 * z + 1 = 0) :
  ∃ (M : ℝ), M = 11 ∧ ∀ (x' y' z' : ℝ),
    x' - y' + z' - 1 = 0 →
    x' * y' + 2 * z'^2 - 6 * z' + 1 = 0 →
    (x' - 1)^2 + (y' + 1)^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_system_l2742_274206


namespace NUMINAMATH_CALUDE_marta_number_proof_l2742_274260

/-- A function that checks if a number has three different non-zero digits -/
def has_three_different_nonzero_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10) ∧
  (n / 100) ≠ 0 ∧ ((n / 10) % 10) ≠ 0 ∧ (n % 10) ≠ 0

/-- A function that checks if a number has three identical digits -/
def has_three_identical_digits (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) = ((n / 10) % 10) ∧
  (n / 100) = (n % 10)

theorem marta_number_proof :
  ∀ n : ℕ,
  has_three_different_nonzero_digits n →
  has_three_identical_digits (3 * n) →
  ((n / 10) % 10) = (3 * n / 100) →
  n = 148 :=
by
  sorry

#check marta_number_proof

end NUMINAMATH_CALUDE_marta_number_proof_l2742_274260


namespace NUMINAMATH_CALUDE_triangle_problem_l2742_274295

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  3 * a * Real.cos C = 2 * c * Real.cos A →
  b = 2 * Real.sqrt 5 →
  c = 3 →
  (a = Real.sqrt 5 ∧
   Real.sin (B + π / 4) = Real.sqrt 10 / 10) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2742_274295


namespace NUMINAMATH_CALUDE_ladder_length_l2742_274217

/-- The length of a ladder given specific conditions --/
theorem ladder_length : ∃ (L : ℝ), 
  (∀ (H : ℝ), L^2 = H^2 + 5^2) ∧ 
  (∀ (H : ℝ), L^2 = (H - 4)^2 + 10.658966865741546^2) ∧
  (abs (L - 14.04) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l2742_274217


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2742_274248

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Iic 2 ∪ Ioi 3 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2742_274248


namespace NUMINAMATH_CALUDE_engagement_treats_ratio_l2742_274243

def total_value : ℕ := 158000
def hotel_cost_per_night : ℕ := 4000
def nights_stayed : ℕ := 2
def car_value : ℕ := 30000

theorem engagement_treats_ratio :
  let hotel_total := hotel_cost_per_night * nights_stayed
  let non_house_total := hotel_total + car_value
  let house_value := total_value - non_house_total
  house_value / car_value = 4 := by
sorry

end NUMINAMATH_CALUDE_engagement_treats_ratio_l2742_274243


namespace NUMINAMATH_CALUDE_binomial_6_choose_2_l2742_274294

theorem binomial_6_choose_2 : Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_6_choose_2_l2742_274294


namespace NUMINAMATH_CALUDE_max_value_of_m_l2742_274284

theorem max_value_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0)
  (heq : 5 = m^2 * (a^2/b^2 + b^2/a^2) + m * (a/b + b/a)) :
  m ≤ (-1 + Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_l2742_274284


namespace NUMINAMATH_CALUDE_fraction_product_cube_specific_fraction_product_l2742_274219

theorem fraction_product_cube (a b c d : ℚ) :
  (a / b)^3 * (c / d)^3 = ((a * c) / (b * d))^3 :=
by sorry

theorem specific_fraction_product :
  (8 / 9 : ℚ)^3 * (3 / 5 : ℚ)^3 = 512 / 3375 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cube_specific_fraction_product_l2742_274219


namespace NUMINAMATH_CALUDE_laptop_to_phone_charger_ratio_l2742_274224

/-- Given a person with 4 phone chargers and 24 total chargers, 
    prove that the ratio of laptop chargers to phone chargers is 5. -/
theorem laptop_to_phone_charger_ratio : 
  ∀ (phone_chargers laptop_chargers : ℕ),
    phone_chargers = 4 →
    phone_chargers + laptop_chargers = 24 →
    laptop_chargers / phone_chargers = 5 := by
  sorry

end NUMINAMATH_CALUDE_laptop_to_phone_charger_ratio_l2742_274224


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2742_274237

theorem quadratic_solution_property : ∀ d e : ℝ,
  (4 * d^2 + 8 * d - 48 = 0) →
  (4 * e^2 + 8 * e - 48 = 0) →
  d ≠ e →
  (d - e)^2 + 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2742_274237


namespace NUMINAMATH_CALUDE_intersection_A_B_l2742_274245

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2742_274245


namespace NUMINAMATH_CALUDE_computer_pricing_l2742_274220

/-- 
Given a computer's selling price and profit percentage, 
calculate the new selling price for a different profit percentage.
-/
theorem computer_pricing (initial_price : ℝ) (initial_profit_percent : ℝ) 
  (new_profit_percent : ℝ) (new_price : ℝ) :
  initial_price = (1 + initial_profit_percent / 100) * (initial_price / (1 + initial_profit_percent / 100)) →
  new_price = (1 + new_profit_percent / 100) * (initial_price / (1 + initial_profit_percent / 100)) →
  initial_price = 2240 →
  initial_profit_percent = 40 →
  new_profit_percent = 50 →
  new_price = 2400 :=
by sorry

end NUMINAMATH_CALUDE_computer_pricing_l2742_274220


namespace NUMINAMATH_CALUDE_star_inequality_equivalence_l2742_274242

-- Define the * operation
def star (a b : ℝ) : ℝ := (a + 3*b) - a*b

-- State the theorem
theorem star_inequality_equivalence :
  ∀ x : ℝ, star 5 x < 13 ↔ x > -4 :=
by sorry

end NUMINAMATH_CALUDE_star_inequality_equivalence_l2742_274242


namespace NUMINAMATH_CALUDE_certain_number_problem_l2742_274222

theorem certain_number_problem (x : ℝ) : 
  (x + 40 + 60) / 3 = (10 + 70 + 16) / 3 + 8 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2742_274222


namespace NUMINAMATH_CALUDE_limit_expression_l2742_274201

/-- The limit of (2 - e^(arcsin²(√x)))^(3/x) as x approaches 0 is e^(-3) -/
theorem limit_expression : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |(2 - Real.exp (Real.arcsin (Real.sqrt x))^2)^(3/x) - Real.exp (-3)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_expression_l2742_274201


namespace NUMINAMATH_CALUDE_round_trip_combinations_l2742_274287

def num_flights_A_to_B : ℕ := 2
def num_flights_B_to_A : ℕ := 3

theorem round_trip_combinations : num_flights_A_to_B * num_flights_B_to_A = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_combinations_l2742_274287


namespace NUMINAMATH_CALUDE_largest_power_of_three_dividing_expression_l2742_274223

theorem largest_power_of_three_dividing_expression (m : ℕ) : 
  (∃ (k : ℕ), (3^k : ℕ) ∣ (2^(3^m) + 1)) ∧ 
  (∀ (k : ℕ), k > 2 → ¬((3^k : ℕ) ∣ (2^(3^m) + 1))) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_three_dividing_expression_l2742_274223


namespace NUMINAMATH_CALUDE_ab_greater_than_ac_l2742_274212

theorem ab_greater_than_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_ac_l2742_274212


namespace NUMINAMATH_CALUDE_dore_change_correct_l2742_274234

/-- Calculate the change given to a customer after a purchase. -/
def calculate_change (pants_cost shirt_cost tie_cost amount_paid : ℕ) : ℕ :=
  amount_paid - (pants_cost + shirt_cost + tie_cost)

/-- Theorem stating that the change is correctly calculated for Mr. Doré's purchase. -/
theorem dore_change_correct :
  calculate_change 140 43 15 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dore_change_correct_l2742_274234


namespace NUMINAMATH_CALUDE_complex_multiplication_l2742_274214

theorem complex_multiplication : (1 + Complex.I) ^ 6 * (1 - Complex.I) = -8 - 8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2742_274214


namespace NUMINAMATH_CALUDE_table_covered_area_l2742_274226

/-- Represents a rectangular paper strip -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Represents the overlap between two strips -/
structure Overlap where
  width : ℕ
  length : ℕ

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.length

theorem table_covered_area (strip1 strip2 strip3 : Strip)
  (overlap12 overlap13 overlap23 : Overlap)
  (h1 : strip1.length = 12)
  (h2 : strip2.length = 15)
  (h3 : strip3.length = 9)
  (h4 : strip1.width = 2)
  (h5 : strip2.width = 2)
  (h6 : strip3.width = 2)
  (h7 : overlap12.width = 2)
  (h8 : overlap12.length = 2)
  (h9 : overlap13.width = 1)
  (h10 : overlap13.length = 2)
  (h11 : overlap23.width = 1)
  (h12 : overlap23.length = 2) :
  stripArea strip1 + stripArea strip2 + stripArea strip3 -
  (overlapArea overlap12 + overlapArea overlap13 + overlapArea overlap23) = 64 := by
  sorry

end NUMINAMATH_CALUDE_table_covered_area_l2742_274226


namespace NUMINAMATH_CALUDE_marble_count_l2742_274266

/-- Represents a bag of marbles with red, blue, and green colors -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Theorem: Given a bag of marbles with the specified conditions, 
    prove the total number of marbles and the number of red marbles -/
theorem marble_count (bag : MarbleBag) 
  (ratio : bag.red * 3 * 4 = bag.blue * 2 * 4 ∧ bag.blue * 2 * 4 = bag.green * 2 * 3)
  (green_count : bag.green = 36) :
  bag.red + bag.blue + bag.green = 81 ∧ bag.red = 18 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l2742_274266


namespace NUMINAMATH_CALUDE_exactly_two_true_propositions_l2742_274282

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

end NUMINAMATH_CALUDE_exactly_two_true_propositions_l2742_274282


namespace NUMINAMATH_CALUDE_cory_fruit_orders_l2742_274236

def number_of_orders (apples oranges lemons : ℕ) : ℕ :=
  Nat.factorial (apples + oranges + lemons) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial lemons)

theorem cory_fruit_orders :
  number_of_orders 4 2 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_orders_l2742_274236


namespace NUMINAMATH_CALUDE_inequality_proof_l2742_274275

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  1/(1-a) + 1/(1-b) + 1/(1-c) ≥ 2/(1+a) + 2/(1+b) + 2/(1+c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2742_274275


namespace NUMINAMATH_CALUDE_birthday_age_problem_l2742_274249

theorem birthday_age_problem (current_age : ℕ) : 
  (current_age = 3 * (current_age - 6)) → current_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_birthday_age_problem_l2742_274249


namespace NUMINAMATH_CALUDE_cosine_equality_l2742_274291

theorem cosine_equality (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 270) :
  Real.cos (n * π / 180) = Real.cos (890 * π / 180) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l2742_274291


namespace NUMINAMATH_CALUDE_coffee_shop_optimal_price_l2742_274285

/-- Profit function for the coffee shop -/
def profit (p : ℝ) : ℝ := 150 * p - 4 * p^2 - 200

/-- The constraint on the price -/
def price_constraint (p : ℝ) : Prop := p ≤ 30

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 19

theorem coffee_shop_optimal_price :
  ∃ (p : ℝ), price_constraint p ∧ 
  ∀ (q : ℝ), price_constraint q → profit p ≥ profit q ∧
  p = optimal_price :=
sorry

end NUMINAMATH_CALUDE_coffee_shop_optimal_price_l2742_274285


namespace NUMINAMATH_CALUDE_quadratic_imaginary_root_magnitude_l2742_274211

theorem quadratic_imaginary_root_magnitude (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (2*m - 1)*x + m^2 + 1
  ∃ z : ℂ, (f z.re = 0 ∧ z.im ≠ 0) → Complex.abs (z + m) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_imaginary_root_magnitude_l2742_274211


namespace NUMINAMATH_CALUDE_inequality_range_l2742_274276

theorem inequality_range (x y k : ℝ) : 
  x > 0 → y > 0 → x + y = k → 
  (∀ x y, x > 0 → y > 0 → x + y = k → (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2) ↔ 
  (k > 0 ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2742_274276


namespace NUMINAMATH_CALUDE_pattern_proof_l2742_274269

theorem pattern_proof (n : ℕ) : (2*n - 1) * (2*n + 1) = (2*n)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l2742_274269


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2742_274207

/-- The price ratio of a muffin to a banana -/
def price_ratio (muffin_price banana_price : ℚ) : ℚ :=
  muffin_price / banana_price

/-- Susie's total cost for 4 muffins and 5 bananas -/
def susie_cost (muffin_price banana_price : ℚ) : ℚ :=
  4 * muffin_price + 5 * banana_price

/-- Calvin's total cost for 2 muffins and 12 bananas -/
def calvin_cost (muffin_price banana_price : ℚ) : ℚ :=
  2 * muffin_price + 12 * banana_price

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
    muffin_price > 0 →
    banana_price > 0 →
    calvin_cost muffin_price banana_price = 3 * susie_cost muffin_price banana_price →
    price_ratio muffin_price banana_price = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2742_274207


namespace NUMINAMATH_CALUDE_log2_odd_and_increasing_l2742_274258

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_odd_and_increasing :
  (∀ x > 0, log2 (-x) = -log2 x) ∧
  (∀ x y, 0 ≤ x → x ≤ y → log2 x ≤ log2 y) :=
by sorry

end NUMINAMATH_CALUDE_log2_odd_and_increasing_l2742_274258


namespace NUMINAMATH_CALUDE_simplify_expression_l2742_274267

theorem simplify_expression (x y : ℝ) : 7 * x + 8 - 3 * x + 15 - 2 * y = 4 * x - 2 * y + 23 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2742_274267


namespace NUMINAMATH_CALUDE_max_value_is_110003_l2742_274293

/-- The set of given integers --/
def given_integers : Finset ℤ := {100004, 110003, 102002, 100301, 100041}

/-- Theorem stating that 110003 is the maximum value in the given set of integers --/
theorem max_value_is_110003 : 
  ∀ x ∈ given_integers, x ≤ 110003 ∧ 110003 ∈ given_integers := by
  sorry

#check max_value_is_110003

end NUMINAMATH_CALUDE_max_value_is_110003_l2742_274293


namespace NUMINAMATH_CALUDE_both_readers_count_l2742_274281

def total_workers : ℕ := 72

def saramago_readers : ℕ := total_workers / 4
def kureishi_readers : ℕ := total_workers * 5 / 8

def both_readers : ℕ := 8

theorem both_readers_count :
  saramago_readers + kureishi_readers - both_readers + 
  (saramago_readers - both_readers - 1) = total_workers :=
by sorry

end NUMINAMATH_CALUDE_both_readers_count_l2742_274281


namespace NUMINAMATH_CALUDE_valid_pairs_l2742_274296

def is_valid_pair (m n : ℕ+) : Prop :=
  (m^2 - n) ∣ (m + n^2) ∧ (n^2 - m) ∣ (n + m^2)

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ 
    ((m = 2 ∧ n = 2) ∨ 
     (m = 3 ∧ n = 3) ∨ 
     (m = 1 ∧ n = 2) ∨ 
     (m = 2 ∧ n = 1) ∨ 
     (m = 3 ∧ n = 2) ∨ 
     (m = 2 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2742_274296


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2742_274233

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*k*x₁ - 2 = 0 ∧ x₂^2 - 3*k*x₂ - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2742_274233


namespace NUMINAMATH_CALUDE_dolls_count_l2742_274262

/-- The total number of toys given -/
def total_toys : ℕ := 403

/-- The number of toy cars given to boys -/
def cars_to_boys : ℕ := 134

/-- The number of dolls given to girls -/
def dolls_to_girls : ℕ := total_toys - cars_to_boys

theorem dolls_count : dolls_to_girls = 269 := by
  sorry

end NUMINAMATH_CALUDE_dolls_count_l2742_274262


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2742_274298

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x < 1 ↔ 2*a*x + 3*x > 2*a + 3) → a < -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2742_274298
