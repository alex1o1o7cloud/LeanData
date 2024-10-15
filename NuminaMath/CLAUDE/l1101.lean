import Mathlib

namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1101_110146

/-- The volume of a right circular cone formed by rolling a two-thirds sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 2 / 3
  let base_radius : ℝ := r * sector_fraction
  let cone_height : ℝ := (r^2 - base_radius^2).sqrt
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = (32/3) * π * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1101_110146


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1101_110175

theorem contrapositive_equivalence (a b : ℝ) :
  (((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0) ↔ 
   (a^2 + b^2 = 0 → a = 0 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1101_110175


namespace NUMINAMATH_CALUDE_discount_card_problem_l1101_110150

/-- Proves that given a discount card that costs 20 yuan and provides a 20% discount,
    if a customer saves 12 yuan by using the card, then the original price of the purchase
    before the discount was 160 yuan. -/
theorem discount_card_problem (card_cost discount_rate savings original_price : ℝ)
    (h1 : card_cost = 20)
    (h2 : discount_rate = 0.2)
    (h3 : savings = 12)
    (h4 : card_cost + (1 - discount_rate) * original_price = original_price - savings) :
    original_price = 160 :=
  sorry

end NUMINAMATH_CALUDE_discount_card_problem_l1101_110150


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_a_equals_one_l1101_110190

-- Define the binomial expansion coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the function for the coefficient of x^(-3) in the expansion
def coefficient_x_neg_3 (a : ℝ) : ℝ :=
  (binomial_coefficient 7 2) * (2^2) * (a^5)

-- State the theorem
theorem expansion_coefficient_implies_a_equals_one :
  coefficient_x_neg_3 1 = 84 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_a_equals_one_l1101_110190


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1101_110192

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, 2^x + x^2 > 0) ↔ (∃ x : ℝ, 2^x + x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1101_110192


namespace NUMINAMATH_CALUDE_wheat_D_tallest_and_neatest_l1101_110174

-- Define the wheat types
inductive WheatType
| A
| B
| C
| D

-- Define a function for average height
def averageHeight (t : WheatType) : ℝ :=
  match t with
  | .A => 13
  | .B => 15
  | .C => 13
  | .D => 15

-- Define a function for variance
def variance (t : WheatType) : ℝ :=
  match t with
  | .A => 3.6
  | .B => 6.3
  | .C => 6.3
  | .D => 3.6

-- Define a predicate for tallness
def isTaller (t1 t2 : WheatType) : Prop :=
  averageHeight t1 > averageHeight t2

-- Define a predicate for neatness (lower variance means neater)
def isNeater (t1 t2 : WheatType) : Prop :=
  variance t1 < variance t2

-- Theorem statement
theorem wheat_D_tallest_and_neatest :
  ∀ t : WheatType, t ≠ WheatType.D →
    (isTaller WheatType.D t ∨ averageHeight WheatType.D = averageHeight t) ∧
    (isNeater WheatType.D t ∨ variance WheatType.D = variance t) :=
by sorry

end NUMINAMATH_CALUDE_wheat_D_tallest_and_neatest_l1101_110174


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l1101_110159

theorem least_four_digit_solution (x : ℕ) : x = 1011 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 10 ≡ 19 [ZMOD 7] ∧
     -3 * y + 4 ≡ 2 * y [ZMOD 16]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20]) ∧
  (3 * x + 10 ≡ 19 [ZMOD 7]) ∧
  (-3 * x + 4 ≡ 2 * x [ZMOD 16]) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l1101_110159


namespace NUMINAMATH_CALUDE_total_dots_is_78_l1101_110108

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots on all ladybugs Andre caught -/
def total_dots : ℕ := (monday_ladybugs + tuesday_ladybugs) * dots_per_ladybug

theorem total_dots_is_78 : total_dots = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_dots_is_78_l1101_110108


namespace NUMINAMATH_CALUDE_translation_result_l1101_110106

-- Define the properties of a triangle
structure Triangle :=
  (shape : Type)
  (size : ℝ)
  (orientation : ℝ)

-- Define the translation operation
def translate (t : Triangle) : Triangle := t

-- Define the given shaded triangle
def shaded_triangle : Triangle := sorry

-- Define the options A, B, C, D, E
def option_A : Triangle := sorry
def option_B : Triangle := sorry
def option_C : Triangle := sorry
def option_D : Triangle := sorry
def option_E : Triangle := sorry

-- State the theorem
theorem translation_result :
  ∀ (t : Triangle),
    translate t = t →
    translate shaded_triangle = option_D :=
by sorry

end NUMINAMATH_CALUDE_translation_result_l1101_110106


namespace NUMINAMATH_CALUDE_fraction_inequality_l1101_110188

theorem fraction_inequality (c x y : ℝ) (h1 : c > x) (h2 : x > y) (h3 : y > 0) :
  x / (c - x) > y / (c - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1101_110188


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l1101_110121

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Probability of drawing either 2 or 3 white balls -/
def prob_two_or_three_white : ℚ := 6/7

/-- Probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 13/14

/-- Theorem stating the probabilities of drawing specific combinations of balls -/
theorem ball_drawing_probabilities :
  (prob_two_or_three_white = 6/7) ∧ (prob_at_least_one_black = 13/14) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l1101_110121


namespace NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l1101_110104

def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 4}

theorem intersection_and_union_of_A_and_B :
  (A ∩ B = A) ∧ (A ∪ B = B) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l1101_110104


namespace NUMINAMATH_CALUDE_sheila_attend_probability_l1101_110193

/-- The probability of rain tomorrow -/
def prob_rain : ℝ := 0.4

/-- The probability Sheila will go if it rains -/
def prob_go_if_rain : ℝ := 0.2

/-- The probability Sheila will go if it's sunny -/
def prob_go_if_sunny : ℝ := 0.8

/-- The probability that Sheila will attend the picnic -/
def prob_sheila_attend : ℝ := prob_rain * prob_go_if_rain + (1 - prob_rain) * prob_go_if_sunny

theorem sheila_attend_probability :
  prob_sheila_attend = 0.56 := by sorry

end NUMINAMATH_CALUDE_sheila_attend_probability_l1101_110193


namespace NUMINAMATH_CALUDE_manfred_average_paycheck_l1101_110157

/-- Calculates the average paycheck amount for Manfred's year, rounded to the nearest dollar. -/
def average_paycheck (total_paychecks : ℕ) (initial_paychecks : ℕ) (initial_amount : ℚ) (increase : ℚ) : ℕ :=
  let remaining_paychecks := total_paychecks - initial_paychecks
  let total_amount := initial_paychecks * initial_amount + remaining_paychecks * (initial_amount + increase)
  let average := total_amount / total_paychecks
  (average + 1/2).floor.toNat

/-- Proves that Manfred's average paycheck for the year, rounded to the nearest dollar, is $765. -/
theorem manfred_average_paycheck :
  average_paycheck 26 6 750 20 = 765 := by
  sorry

end NUMINAMATH_CALUDE_manfred_average_paycheck_l1101_110157


namespace NUMINAMATH_CALUDE_range_of_a_l1101_110141

/-- Given that the inequality x^2 + ax - 2 > 0 has a solution in the interval [1,2],
    the range of a is (-1, +∞) -/
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + a*x - 2 > 0) ↔ a ∈ Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1101_110141


namespace NUMINAMATH_CALUDE_nell_initial_ace_cards_l1101_110100

/-- Prove that Nell had 315 Ace cards initially -/
theorem nell_initial_ace_cards 
  (initial_baseball : ℕ)
  (final_ace : ℕ)
  (final_baseball : ℕ)
  (baseball_ace_difference : ℕ)
  (h1 : initial_baseball = 438)
  (h2 : final_ace = 55)
  (h3 : final_baseball = 178)
  (h4 : final_baseball = final_ace + baseball_ace_difference)
  (h5 : baseball_ace_difference = 123) :
  ∃ (initial_ace : ℕ), initial_ace = 315 ∧ 
    initial_ace - final_ace = initial_baseball - final_baseball :=
by
  sorry

end NUMINAMATH_CALUDE_nell_initial_ace_cards_l1101_110100


namespace NUMINAMATH_CALUDE_total_bus_ride_distance_l1101_110113

theorem total_bus_ride_distance :
  let vince_ride : ℚ := 5/8
  let zachary_ride : ℚ := 1/2
  let alice_ride : ℚ := 17/20
  let rebecca_ride : ℚ := 2/5
  vince_ride + zachary_ride + alice_ride + rebecca_ride = 19/8
  := by sorry

end NUMINAMATH_CALUDE_total_bus_ride_distance_l1101_110113


namespace NUMINAMATH_CALUDE_new_person_weight_is_97_l1101_110144

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 97 kg -/
theorem new_person_weight_is_97 :
  weight_of_new_person 8 4 65 = 97 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_97_l1101_110144


namespace NUMINAMATH_CALUDE_work_time_solution_l1101_110107

def work_time (T : ℝ) : Prop :=
  let A_alone := T + 8
  let B_alone := T + 4.5
  (1 / A_alone) + (1 / B_alone) = 1 / T

theorem work_time_solution : ∃ T : ℝ, work_time T ∧ T = 6 := by sorry

end NUMINAMATH_CALUDE_work_time_solution_l1101_110107


namespace NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l1101_110116

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_zeros (l : List Bool) : ℕ :=
  sorry

def count_ones (l : List Bool) : ℕ :=
  sorry

theorem binary_253_ones_minus_zeros :
  let bin_253 := binary_representation 253
  let x := count_zeros bin_253
  let y := count_ones bin_253
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l1101_110116


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l1101_110199

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^11 + i^16 + i^21 + i^26 + i^31 + i^36 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l1101_110199


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1101_110112

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 14*y + 73 = -y^2 + 6*x

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 10 + Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1101_110112


namespace NUMINAMATH_CALUDE_triangle_value_l1101_110185

theorem triangle_value (triangle : ℝ) :
  (∀ x : ℝ, (x - 5) * (x + triangle) = x^2 + 2*x - 35) →
  triangle = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l1101_110185


namespace NUMINAMATH_CALUDE_sum_of_heights_l1101_110162

theorem sum_of_heights (n : ℕ) (h1 : n = 30) (s10 s20 : ℕ) 
  (h2 : s10 = 1450) (h3 : s20 = 3030) : ∃ (s30 : ℕ), s30 = 4610 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_heights_l1101_110162


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1101_110132

/-- Given a parabola with equation x = (1/4m)y^2, its focus has coordinates (m, 0) --/
theorem parabola_focus_coordinates (m : ℝ) (h : m ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | x = (1 / (4 * m)) * y^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (m, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1101_110132


namespace NUMINAMATH_CALUDE_g_of_six_equals_eleven_l1101_110135

theorem g_of_six_equals_eleven (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (4 * x - 2) = x^2 + 2 * x + 3) : 
  g 6 = 11 := by
sorry

end NUMINAMATH_CALUDE_g_of_six_equals_eleven_l1101_110135


namespace NUMINAMATH_CALUDE_multiply_polynomials_l1101_110161

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 + 12*x^4 - 144*x^2 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l1101_110161


namespace NUMINAMATH_CALUDE_fred_gave_25_seashells_l1101_110151

/-- The number of seashells Fred initially had -/
def initial_seashells : ℕ := 47

/-- The number of seashells Fred has now -/
def remaining_seashells : ℕ := 22

/-- The number of seashells Fred gave to Jessica -/
def seashells_given : ℕ := initial_seashells - remaining_seashells

theorem fred_gave_25_seashells : seashells_given = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_gave_25_seashells_l1101_110151


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1101_110103

theorem smallest_integer_solution :
  ∀ y : ℤ, (8 - 3 * y ≤ 23) → y ≥ -5 ∧ ∀ z : ℤ, z < -5 → (8 - 3 * z > 23) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1101_110103


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1101_110169

theorem range_of_a_for_inequality : 
  {a : ℝ | ∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a} = {a : ℝ | -1 ≤ a ∧ a ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1101_110169


namespace NUMINAMATH_CALUDE_xy_is_zero_l1101_110147

theorem xy_is_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_is_zero_l1101_110147


namespace NUMINAMATH_CALUDE_square_cube_sum_condition_l1101_110181

theorem square_cube_sum_condition (n : ℕ) : 
  (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_cube_sum_condition_l1101_110181


namespace NUMINAMATH_CALUDE_radio_cost_price_l1101_110182

/-- Proves that the cost price of a radio is 1500 Rs. given the selling price and loss percentage --/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1110 → loss_percentage = 26 → 
  ∃ (cost_price : ℝ), cost_price = 1500 ∧ selling_price = cost_price * (1 - loss_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l1101_110182


namespace NUMINAMATH_CALUDE_solution_set_inequality_proof_l1101_110111

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3|
def g (x : ℝ) : ℝ := |x - 2|

-- Theorem for the solution set of f(x) + g(x) < 2
theorem solution_set : 
  {x : ℝ | f x + g x < 2} = {x : ℝ | 3/2 < x ∧ x < 7/2} := by sorry

-- Theorem for the inequality proof
theorem inequality_proof (x y : ℝ) (hx : f x ≤ 1) (hy : g y ≤ 1) : 
  |x - 2*y + 1| ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_proof_l1101_110111


namespace NUMINAMATH_CALUDE_cubic_roots_from_conditions_l1101_110129

theorem cubic_roots_from_conditions (p q r : ℂ) :
  p + q + r = 0 →
  p * q + p * r + q * r = -1 →
  p * q * r = -1 →
  {p, q, r} = {x : ℂ | x^3 - x - 1 = 0} := by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_conditions_l1101_110129


namespace NUMINAMATH_CALUDE_length_of_AB_l1101_110122

/-- Given a line segment AB with points P and Q, prove that AB has length 48 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (P - A) / (B - P) = 1 / 4 →  -- P divides AB in ratio 1:4
  (Q - A) / (B - Q) = 2 / 5 →  -- Q divides AB in ratio 2:5
  Q - P = 3 →                  -- Length of PQ is 3
  B - A = 48 := by             -- Length of AB is 48
sorry


end NUMINAMATH_CALUDE_length_of_AB_l1101_110122


namespace NUMINAMATH_CALUDE_div_decimal_equals_sixty_l1101_110189

theorem div_decimal_equals_sixty : (0.24 : ℚ) / (0.004 : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_div_decimal_equals_sixty_l1101_110189


namespace NUMINAMATH_CALUDE_largest_divided_by_smallest_l1101_110176

def numbers : List ℕ := [38, 114, 152, 95]

theorem largest_divided_by_smallest : 
  (List.maximum numbers).get! / (List.minimum numbers).get! = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_divided_by_smallest_l1101_110176


namespace NUMINAMATH_CALUDE_sculpture_and_base_height_l1101_110198

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := feetToInches h.feet + h.inches

/-- Theorem: The combined height of the sculpture and base is 42 inches -/
theorem sculpture_and_base_height :
  let sculpture : Height := { feet := 2, inches := 10 }
  let base_height : ℕ := 8
  heightToInches sculpture + base_height = 42 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_and_base_height_l1101_110198


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1101_110117

/-- Represents a person at the table -/
structure Person :=
  (id : Nat)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Defines the relationship of acquaintance between two people -/
def IsAcquainted (table : Table) (p1 p2 : Person) : Prop := sorry

/-- Counts the number of people between two given positions on the table -/
def PeopleBetween (pos1 pos2 : Fin 40) : Nat := sorry

/-- States that for any two people with an even number between them, there's a common acquaintance -/
def EvenHaveCommonAcquaintance (table : Table) : Prop :=
  ∀ (pos1 pos2 : Fin 40), Even (PeopleBetween pos1 pos2) →
    ∃ (p : Person), IsAcquainted table (table pos1) p ∧ IsAcquainted table (table pos2) p

/-- States that for any two people with an odd number between them, there's no common acquaintance -/
def OddNoCommonAcquaintance (table : Table) : Prop :=
  ∀ (pos1 pos2 : Fin 40), Odd (PeopleBetween pos1 pos2) →
    ∀ (p : Person), ¬(IsAcquainted table (table pos1) p ∧ IsAcquainted table (table pos2) p)

/-- The main theorem stating that no arrangement satisfies both conditions -/
theorem no_valid_arrangement :
  ¬∃ (table : Table), EvenHaveCommonAcquaintance table ∧ OddNoCommonAcquaintance table :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l1101_110117


namespace NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l1101_110196

theorem fourth_root_sum_of_fourth_powers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, c = (a^4 + b^4)^(1/4) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_sum_of_fourth_powers_l1101_110196


namespace NUMINAMATH_CALUDE_vector_decomposition_l1101_110109

def x : Fin 3 → ℝ := ![3, -1, 2]
def p : Fin 3 → ℝ := ![2, 0, 1]
def q : Fin 3 → ℝ := ![1, -1, 1]
def r : Fin 3 → ℝ := ![1, -1, -2]

theorem vector_decomposition :
  x = p + q :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1101_110109


namespace NUMINAMATH_CALUDE_two_card_probability_l1101_110168

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (size : cards.card = 52)

/-- The probability of selecting two cards from a standard 52-card deck
    that are neither of the same value nor the same suit is 12/17. -/
theorem two_card_probability (d : Deck) : 
  let first_card := d.cards.card
  let second_card := d.cards.card - 1
  let favorable_outcomes := 3 * 12
  (favorable_outcomes : ℚ) / second_card = 12 / 17 := by sorry

end NUMINAMATH_CALUDE_two_card_probability_l1101_110168


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_l1101_110177

/-- Ellipse defined by parametric equations x = 2cos(α) and y = √3sin(α) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 2 * Real.cos α ∧ p.2 = Real.sqrt 3 * Real.sin α}

/-- Distance squared from origin to a point -/
def distanceSquared (p : ℝ × ℝ) : ℝ := p.1^2 + p.2^2

/-- Two points are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem ellipse_perpendicular_points_sum (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) (hB : B ∈ Ellipse) (hPerp : perpendicular A B) :
  1 / distanceSquared A + 1 / distanceSquared B = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_points_sum_l1101_110177


namespace NUMINAMATH_CALUDE_min_value_expression_l1101_110163

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y^2 * z = 72) : 
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 120 ∧ 
  (x^2 + 4*x*y + 4*y^2 + 2*z^2 = 120 ↔ x = 6 ∧ y = 3 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1101_110163


namespace NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l1101_110105

-- Define the repeating decimals
def repeating_decimal_37 : ℚ := 37 + 37 / 99
def repeating_decimal_15 : ℚ := 15 + 15 / 99

-- Define the sum of the repeating decimals
def sum : ℚ := repeating_decimal_37 + repeating_decimal_15

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ := 
  ⌊x * 100 + 1/2⌋ / 100

-- Theorem statement
theorem sum_rounded_to_hundredth : 
  round_to_hundredth sum = 52 / 100 := by sorry

end NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l1101_110105


namespace NUMINAMATH_CALUDE_problem_solution_l1101_110171

def f (k : ℝ) (x : ℝ) := k - |x - 3|

theorem problem_solution (k : ℝ) (a b c : ℝ) 
  (h1 : Set.Icc (-1 : ℝ) 1 = {x | f k (x + 3) ≥ 0})
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : 1/a + 1/(2*b) + 1/(3*c) = 1) :
  k = 1 ∧ 1/9 * a + 2/9 * b + 3/9 * c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1101_110171


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l1101_110130

/-- The number of terms in the expansion of ((x^(1/2) + x^(1/3))^12) -/
def total_terms : ℕ := 13

/-- The number of terms with positive integer powers of x in the expansion -/
def integer_power_terms : ℕ := 3

/-- The number of terms without positive integer powers of x in the expansion -/
def non_integer_power_terms : ℕ := total_terms - integer_power_terms

/-- The number of ways to rearrange the terms in the expansion of ((x^(1/2) + x^(1/3))^12)
    so that the terms containing positive integer powers of x are not adjacent to each other -/
def rearrangement_count : ℕ := (Nat.factorial non_integer_power_terms) * (Nat.factorial (non_integer_power_terms + 1) / (Nat.factorial (non_integer_power_terms - 2)))

theorem rearrangement_theorem : 
  rearrangement_count = (Nat.factorial 10) * (Nat.factorial 11 / (Nat.factorial 8)) :=
sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l1101_110130


namespace NUMINAMATH_CALUDE_john_water_savings_l1101_110191

def water_savings (old_flush_volume : ℝ) (flushes_per_day : ℕ) (water_reduction_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let old_daily_usage := old_flush_volume * flushes_per_day
  let old_monthly_usage := old_daily_usage * days_in_month
  let new_flush_volume := old_flush_volume * (1 - water_reduction_percentage)
  let new_daily_usage := new_flush_volume * flushes_per_day
  let new_monthly_usage := new_daily_usage * days_in_month
  old_monthly_usage - new_monthly_usage

theorem john_water_savings :
  water_savings 5 15 0.8 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_john_water_savings_l1101_110191


namespace NUMINAMATH_CALUDE_steve_height_l1101_110155

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Converts a height given in feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet_to_inches feet + inches

/-- Calculates the final height after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  height_to_inches initial_feet initial_inches + growth

theorem steve_height :
  final_height 5 6 6 = 72 := by sorry

end NUMINAMATH_CALUDE_steve_height_l1101_110155


namespace NUMINAMATH_CALUDE_A_intersect_B_l1101_110140

-- Define the universe U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define set A
def A : Set Nat := {2, 4, 6, 8, 10}

-- Define complement of A with respect to U
def C_UA : Set Nat := {1, 3, 5, 7, 9}

-- Define complement of B with respect to U
def C_UB : Set Nat := {1, 4, 6, 8, 9}

-- Define set B (derived from its complement)
def B : Set Nat := U \ C_UB

theorem A_intersect_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1101_110140


namespace NUMINAMATH_CALUDE_triangle_properties_l1101_110180

theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Given conditions
  (c / Real.cos C = (a + b) / (Real.cos A + Real.cos B)) →
  (Real.cos A + Real.cos B ≠ 0) →
  (D.1 = (B + C) / 2) →
  (D.2 = 0) →
  (Real.sqrt ((A - D.1)^2 + D.2^2) = 2) →
  (Real.sqrt ((A - C)^2 + 0^2) = Real.sqrt 7) →
  -- Conclusions
  (C = Real.pi / 3) ∧
  (Real.sqrt ((B - A)^2 + 0^2) = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1101_110180


namespace NUMINAMATH_CALUDE_tan_arctan_five_twelfths_l1101_110124

theorem tan_arctan_five_twelfths : 
  Real.tan (Real.arctan (5 / 12)) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_tan_arctan_five_twelfths_l1101_110124


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1101_110178

theorem modulus_of_complex_fraction : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1101_110178


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1101_110123

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  (a > 0 → S = {x : ℝ | x < -a/4 ∨ x > a/3}) ∧
  (a = 0 → S = {x : ℝ | x ≠ 0}) ∧
  (a < 0 → S = {x : ℝ | x < a/3 ∨ x > -a/4}) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1101_110123


namespace NUMINAMATH_CALUDE_rent_increase_group_size_l1101_110160

theorem rent_increase_group_size 
  (initial_avg : ℝ) 
  (new_avg : ℝ) 
  (increased_rent : ℝ) 
  (increase_rate : ℝ) :
  initial_avg = 800 →
  new_avg = 870 →
  increased_rent = 1400 →
  increase_rate = 0.2 →
  ∃ n : ℕ, 
    n > 0 ∧
    n * new_avg = (n * initial_avg - increased_rent + increased_rent * (1 + increase_rate)) ∧
    n = 4 :=
by sorry

end NUMINAMATH_CALUDE_rent_increase_group_size_l1101_110160


namespace NUMINAMATH_CALUDE_output_increase_percentage_l1101_110128

/-- Represents the increase in output per hour when production increases by 80% and working hours decrease by 10% --/
theorem output_increase_percentage (B : ℝ) (H : ℝ) (B_pos : B > 0) (H_pos : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  (new_output - original_output) / original_output = 1 := by
sorry

end NUMINAMATH_CALUDE_output_increase_percentage_l1101_110128


namespace NUMINAMATH_CALUDE_xy_is_zero_l1101_110120

theorem xy_is_zero (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_is_zero_l1101_110120


namespace NUMINAMATH_CALUDE_laundry_earnings_for_three_days_l1101_110101

def laundry_earnings (charge_per_kilo : ℝ) (day1_kilos : ℝ) : ℝ :=
  let day2_kilos := day1_kilos + 5
  let day3_kilos := 2 * day2_kilos
  charge_per_kilo * (day1_kilos + day2_kilos + day3_kilos)

theorem laundry_earnings_for_three_days :
  laundry_earnings 2 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_laundry_earnings_for_three_days_l1101_110101


namespace NUMINAMATH_CALUDE_smallest_a_inequality_l1101_110165

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  ∀ a : ℝ, a ≥ 2/9 →
    a * (x^2 + y^2 + z^2) + x * y * z ≥ a / 3 + 1 / 27 ∧
    ∀ b : ℝ, b < 2/9 →
      ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧
        b * (x'^2 + y'^2 + z'^2) + x' * y' * z' < b / 3 + 1 / 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_inequality_l1101_110165


namespace NUMINAMATH_CALUDE_real_estate_investment_l1101_110152

theorem real_estate_investment 
  (total_investment : ℝ) 
  (real_estate_ratio : ℝ) 
  (h1 : total_investment = 200000)
  (h2 : real_estate_ratio = 7) : 
  let mutual_funds := total_investment / (real_estate_ratio + 1)
  let real_estate := real_estate_ratio * mutual_funds
  real_estate = 175000 := by
sorry

end NUMINAMATH_CALUDE_real_estate_investment_l1101_110152


namespace NUMINAMATH_CALUDE_expression_simplification_l1101_110119

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1101_110119


namespace NUMINAMATH_CALUDE_other_number_proof_l1101_110131

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 132) : b = 36 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1101_110131


namespace NUMINAMATH_CALUDE_smallest_number_with_divisible_digit_sums_l1101_110136

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the divisibility condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  17 ∣ sumOfDigits n ∧ 17 ∣ sumOfDigits (n + 1)

theorem smallest_number_with_divisible_digit_sums :
  satisfiesCondition 8899 ∧ ∀ m < 8899, ¬satisfiesCondition m := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_divisible_digit_sums_l1101_110136


namespace NUMINAMATH_CALUDE_scalper_discount_l1101_110134

def discount_problem (normal_price : ℝ) (scalper_markup : ℝ) (friend_discount : ℝ) (total_payment : ℝ) : Prop :=
  let website_tickets := 2 * normal_price
  let scalper_tickets := 2 * (normal_price * scalper_markup)
  let friend_ticket := normal_price * friend_discount
  let total_before_discount := website_tickets + scalper_tickets + friend_ticket
  total_before_discount - total_payment = 10

theorem scalper_discount :
  discount_problem 50 2.4 0.6 360 := by
  sorry

end NUMINAMATH_CALUDE_scalper_discount_l1101_110134


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1101_110183

-- Define the floor function
def floor (x : ℚ) : ℤ := Int.floor x

-- Define the theorem
theorem floor_equation_solutions (a : ℚ) (ha : 0 < a) :
  ∀ x : ℕ+, (floor ((3 * x.val + a) / 4) = 2) ↔ (x.val = 1 ∨ x.val = 2 ∨ x.val = 3) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1101_110183


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l1101_110143

theorem function_inequality_implies_upper_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 3, ∃ x₂ ∈ Set.Icc (2 : ℝ) 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l1101_110143


namespace NUMINAMATH_CALUDE_mean_proportional_segment_l1101_110145

theorem mean_proportional_segment (a b c : ℝ) : 
  a = 3 → b = 27 → c^2 = a * b → c = 9 := by sorry

end NUMINAMATH_CALUDE_mean_proportional_segment_l1101_110145


namespace NUMINAMATH_CALUDE_nuts_in_tree_l1101_110138

theorem nuts_in_tree (squirrels : ℕ) (difference : ℕ) (nuts : ℕ) : 
  squirrels = 4 → 
  squirrels = nuts + difference → 
  difference = 2 → 
  nuts = 2 := by sorry

end NUMINAMATH_CALUDE_nuts_in_tree_l1101_110138


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1101_110127

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation :
  ∀ (a b m : ℝ) (P : ℝ × ℝ),
    b > a ∧ a > 0 ∧ m > 0 →
    P.1 = Real.sqrt 5 ∧ P.2 = m →
    P.1^2 / a^2 - P.2^2 / b^2 = 1 →
    P.1 = Real.sqrt (a^2 + b^2) →
    (∃ (A B : ℝ × ℝ),
      (A.2 - P.2) / (A.1 - P.1) = b / a ∧
      (B.2 - P.2) / (B.1 - P.1) = -b / a ∧
      (A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1) = 2) →
    ∀ (x y : ℝ), x^2 - y^2 / 4 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1101_110127


namespace NUMINAMATH_CALUDE_geometric_sum_15_l1101_110115

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_15 (a : ℕ → ℤ) :
  geometric_sequence a →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n * (-2)) →
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_15_l1101_110115


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1101_110195

/-- Calculates the length of a bridge given the train's length, speed, and time to cross. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 255 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1101_110195


namespace NUMINAMATH_CALUDE_parallel_vectors_proportional_components_l1101_110149

/-- Given two 2D vectors a and b, if they are parallel, then their components are proportional. -/
theorem parallel_vectors_proportional_components (a b : ℝ × ℝ) :
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) ↔ ∃ m : ℝ, a = (2, -1) ∧ b = (-1, m) ∧ m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_proportional_components_l1101_110149


namespace NUMINAMATH_CALUDE_min_value_fraction_l1101_110179

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∀ y > 6, x^2 / (x - 6) ≤ y^2 / (y - 6)) → x^2 / (x - 6) = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1101_110179


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1101_110125

theorem complex_fraction_evaluation :
  (Complex.I : ℂ) / (12 + Complex.I) = (1 : ℂ) / 145 + (12 : ℂ) / 145 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1101_110125


namespace NUMINAMATH_CALUDE_rectangle_distance_l1101_110167

theorem rectangle_distance (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 6 →
  large_area = 12 →
  let small_width := small_perimeter / 6
  let small_length := 2 * small_width
  let large_width := 3 * small_width
  let large_length := 2 * small_length
  large_width * large_length = large_area →
  let horizontal_distance := large_length
  let vertical_distance := large_width - small_width
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_distance_l1101_110167


namespace NUMINAMATH_CALUDE_probability_3_1_is_5_over_10_2_l1101_110153

def blue_balls : ℕ := 10
def red_balls : ℕ := 8
def total_balls : ℕ := blue_balls + red_balls
def drawn_balls : ℕ := 4

def probability_3_1 : ℚ :=
  let total_ways := Nat.choose total_balls drawn_balls
  let ways_3blue_1red := Nat.choose blue_balls 3 * Nat.choose red_balls 1
  let ways_1blue_3red := Nat.choose blue_balls 1 * Nat.choose red_balls 3
  (ways_3blue_1red + ways_1blue_3red : ℚ) / total_ways

theorem probability_3_1_is_5_over_10_2 :
  probability_3_1 = 5 / 10.2 := by sorry

end NUMINAMATH_CALUDE_probability_3_1_is_5_over_10_2_l1101_110153


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_1500_l1101_110137

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The difference between the sum of even and odd numbers -/
def even_odd_sum_difference (n : ℕ) : ℤ :=
  arithmetic_sum 0 2 n - arithmetic_sum 1 2 n

theorem even_odd_sum_difference_1500 :
  even_odd_sum_difference 1500 = -1500 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_1500_l1101_110137


namespace NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l1101_110126

theorem smallest_solution_for_floor_equation :
  let x : ℝ := 131 / 11
  ∀ y : ℝ, y > 0 → (⌊y^2⌋ : ℝ) - y * ⌊y⌋ = 10 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_for_floor_equation_l1101_110126


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1101_110154

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₃ + a₅ = 16, a₄ = 8 -/
theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 = 16) :
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1101_110154


namespace NUMINAMATH_CALUDE_plane_speed_theorem_l1101_110170

theorem plane_speed_theorem (v : ℝ) (h1 : v > 0) :
  5 * v + 5 * (3 * v) = 4800 →
  v = 240 ∧ 3 * v = 720 := by
  sorry

end NUMINAMATH_CALUDE_plane_speed_theorem_l1101_110170


namespace NUMINAMATH_CALUDE_intersection_points_existence_and_variability_l1101_110164

/-- The parabola equation -/
def parabola (A : ℝ) (x y : ℝ) : Prop := y = A * x^2

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 + 2 = x^2 + 6 * y

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2 * x - 1

/-- The intersection point satisfies all three equations -/
def is_intersection_point (A : ℝ) (x y : ℝ) : Prop :=
  parabola A x y ∧ hyperbola x y ∧ line x y

/-- The theorem stating that there is at least one intersection point and the number can vary -/
theorem intersection_points_existence_and_variability :
  ∀ A : ℝ, A > 0 →
  (∃ x y : ℝ, is_intersection_point A x y) ∧
  (∃ A₁ A₂ : ℝ, A₁ > 0 ∧ A₂ > 0 ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
      is_intersection_point A₁ x₁ y₁ ∧
      is_intersection_point A₁ x₂ y₂ ∧
      is_intersection_point A₂ x₃ y₃ ∧
      (x₁ ≠ x₂ ∨ y₁ ≠ y₂))) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_existence_and_variability_l1101_110164


namespace NUMINAMATH_CALUDE_oranges_taken_l1101_110173

theorem oranges_taken (initial_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 60)
  (h2 : final_oranges = 25) :
  initial_oranges - final_oranges = 35 := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_l1101_110173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1101_110197

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (n : ℕ) :
  (a 1 = 1) →
  (∀ k : ℕ, a (k + 1) - a k = 3) →
  (a n = 298) →
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1101_110197


namespace NUMINAMATH_CALUDE_white_roses_needed_l1101_110139

/-- Calculates the total number of white roses needed for wedding arrangements -/
theorem white_roses_needed (num_bouquets num_table_decorations roses_per_bouquet roses_per_table_decoration : ℕ) : 
  num_bouquets = 5 → 
  num_table_decorations = 7 → 
  roses_per_bouquet = 5 → 
  roses_per_table_decoration = 12 → 
  num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration = 109 := by
  sorry

#check white_roses_needed

end NUMINAMATH_CALUDE_white_roses_needed_l1101_110139


namespace NUMINAMATH_CALUDE_tetrahedron_vector_sum_same_sign_l1101_110187

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A point is inside a tetrahedron if it can be expressed as a convex combination of the vertices -/
def IsInsideTetrahedron (O A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧
    O = a • A + b • B + c • C + d • D

/-- All real numbers have the same sign if they are all positive or all negative -/
def AllSameSign (α β γ δ : ℝ) : Prop :=
  (α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) ∨ (α < 0 ∧ β < 0 ∧ γ < 0 ∧ δ < 0)

theorem tetrahedron_vector_sum_same_sign
  (O A B C D : V) (α β γ δ : ℝ)
  (h_inside : IsInsideTetrahedron O A B C D)
  (h_sum : α • (A - O) + β • (B - O) + γ • (C - O) + δ • (D - O) = 0) :
  AllSameSign α β γ δ :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_vector_sum_same_sign_l1101_110187


namespace NUMINAMATH_CALUDE_remainder_of_three_to_500_mod_17_l1101_110184

theorem remainder_of_three_to_500_mod_17 : 3^500 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_three_to_500_mod_17_l1101_110184


namespace NUMINAMATH_CALUDE_min_value_expression_l1101_110114

open Real

theorem min_value_expression :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 - 12 ∧
  ∀ (x y : ℝ),
    (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (3 - Real.sqrt 2) * Real.sin x + 1) *
    (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1101_110114


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_l1101_110110

def repeating_decimal : ℚ := 123 / 999

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 374 := by sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_l1101_110110


namespace NUMINAMATH_CALUDE_xy_minus_10_squared_l1101_110158

theorem xy_minus_10_squared (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10)^2 ≥ 64 ∧ 
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) := by
  sorry

end NUMINAMATH_CALUDE_xy_minus_10_squared_l1101_110158


namespace NUMINAMATH_CALUDE_race_time_l1101_110102

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- A beats B by 50 meters
  950 = b.speed * a.time ∧
  -- A beats B by 10 seconds
  b.time = a.time + 10

theorem race_time (a b : Runner) (h : Race a b) : a.time = 200 := by
  sorry

end NUMINAMATH_CALUDE_race_time_l1101_110102


namespace NUMINAMATH_CALUDE_beau_current_age_l1101_110166

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents Beau and his three sons -/
structure Family where
  beau : Person
  son1 : Person
  son2 : Person
  son3 : Person

/-- The age of Beau's sons today -/
def sonAgeToday : ℕ := 16

/-- The theorem stating Beau's current age -/
theorem beau_current_age (f : Family) : 
  (f.son1.age = sonAgeToday) ∧ 
  (f.son2.age = sonAgeToday) ∧ 
  (f.son3.age = sonAgeToday) ∧ 
  (f.beau.age = f.son1.age + f.son2.age + f.son3.age + 3) → 
  f.beau.age = 42 := by
  sorry


end NUMINAMATH_CALUDE_beau_current_age_l1101_110166


namespace NUMINAMATH_CALUDE_cosine_sum_equals_one_l1101_110142

theorem cosine_sum_equals_one (α β γ : Real) 
  (sum_eq_pi : α + β + γ = Real.pi)
  (tan_sum_eq_one : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_equals_one_l1101_110142


namespace NUMINAMATH_CALUDE_larger_square_area_l1101_110148

theorem larger_square_area (small_side : ℝ) (small_triangles : ℕ) (large_triangles : ℕ) :
  small_side = 12 →
  small_triangles = 16 →
  large_triangles = 18 →
  (large_triangles : ℝ) / (small_triangles : ℝ) * (small_side ^ 2) = 162 := by
  sorry

end NUMINAMATH_CALUDE_larger_square_area_l1101_110148


namespace NUMINAMATH_CALUDE_beijing_olympics_village_area_notation_l1101_110133

/-- Expresses 38.66 million in scientific notation -/
theorem beijing_olympics_village_area_notation :
  (38.66 * 1000000 : ℝ) = 3.866 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_beijing_olympics_village_area_notation_l1101_110133


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1101_110156

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 2 * x + (1/2 : ℝ) < 0) ↔
  (∃ x₀ : ℝ, 2 * x₀^2 + 2 * x₀ + (1/2 : ℝ) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1101_110156


namespace NUMINAMATH_CALUDE_divisors_4k_plus_1_ge_4k_minus_1_l1101_110186

/-- The number of divisors of n of the form 4k+1 -/
def divisors_4k_plus_1 (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n of the form 4k-1 -/
def divisors_4k_minus_1 (n : ℕ+) : ℕ := sorry

/-- The difference between the number of divisors of the form 4k+1 and 4k-1 -/
def D (n : ℕ+) : ℤ := (divisors_4k_plus_1 n : ℤ) - (divisors_4k_minus_1 n : ℤ)

theorem divisors_4k_plus_1_ge_4k_minus_1 (n : ℕ+) : D n ≥ 0 := by sorry

end NUMINAMATH_CALUDE_divisors_4k_plus_1_ge_4k_minus_1_l1101_110186


namespace NUMINAMATH_CALUDE_tempo_original_value_l1101_110118

/-- The original value of a tempo given insurance and premium information -/
def original_value (insured_fraction : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) : ℚ :=
  premium_amount / (premium_rate * insured_fraction)

/-- Theorem stating the original value of the tempo given the problem conditions -/
theorem tempo_original_value :
  let insured_fraction : ℚ := 4 / 5
  let premium_rate : ℚ := 13 / 1000
  let premium_amount : ℚ := 910
  original_value insured_fraction premium_rate premium_amount = 87500 := by
sorry

end NUMINAMATH_CALUDE_tempo_original_value_l1101_110118


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1101_110194

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  use -4
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1101_110194


namespace NUMINAMATH_CALUDE_demand_increase_factor_l1101_110172

def demand (p : ℝ) : ℝ := 150 - p

def supply (p : ℝ) : ℝ := 3 * p - 10

def new_demand (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

theorem demand_increase_factor (α : ℝ) :
  (∃ p_initial p_new : ℝ,
    demand p_initial = supply p_initial ∧
    new_demand α p_new = supply p_new ∧
    p_new = 1.25 * p_initial) →
  α = 1.4 := by sorry

end NUMINAMATH_CALUDE_demand_increase_factor_l1101_110172
