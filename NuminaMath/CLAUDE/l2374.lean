import Mathlib

namespace wax_needed_for_SUV_l2374_237473

/-- The amount of wax needed to detail Kellan's SUV -/
def wax_for_SUV : ℕ := by sorry

/-- The amount of wax needed to detail Kellan's car -/
def wax_for_car : ℕ := 3

/-- The amount of wax in the bottle Kellan bought -/
def wax_bought : ℕ := 11

/-- The amount of wax Kellan spilled -/
def wax_spilled : ℕ := 2

/-- The amount of wax left after detailing both vehicles -/
def wax_left : ℕ := 2

theorem wax_needed_for_SUV : 
  wax_for_SUV = 4 := by sorry

end wax_needed_for_SUV_l2374_237473


namespace inequality_implication_l2374_237447

theorem inequality_implication (x y : ℝ) : x < y → -x/2 > -y/2 := by
  sorry

end inequality_implication_l2374_237447


namespace negation_of_quadratic_inequality_l2374_237402

theorem negation_of_quadratic_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0) → 
  (¬p ↔ ∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end negation_of_quadratic_inequality_l2374_237402


namespace complex_product_theorem_l2374_237462

theorem complex_product_theorem (z₁ z₂ : ℂ) : 
  z₁.re = 2 ∧ z₁.im = 1 ∧ z₂.re = 0 ∧ z₂.im = -1 → z₁ * z₂ = 1 - 2*I :=
by sorry

end complex_product_theorem_l2374_237462


namespace blue_paint_calculation_l2374_237467

/-- Represents the ratio of paints (red:blue:yellow:white) -/
structure PaintRatio :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (white : ℕ)

/-- Calculates the amount of blue paint needed given a paint ratio and the amount of white paint used -/
def blue_paint_needed (ratio : PaintRatio) (white_paint : ℕ) : ℕ :=
  (ratio.blue * white_paint) / ratio.white

/-- Theorem stating that given the specific paint ratio and 16 quarts of white paint, 12 quarts of blue paint are needed -/
theorem blue_paint_calculation (ratio : PaintRatio) (h1 : ratio.red = 2) (h2 : ratio.blue = 3) 
    (h3 : ratio.yellow = 1) (h4 : ratio.white = 4) (white_paint : ℕ) (h5 : white_paint = 16) : 
  blue_paint_needed ratio white_paint = 12 := by
  sorry

#eval blue_paint_needed {red := 2, blue := 3, yellow := 1, white := 4} 16

end blue_paint_calculation_l2374_237467


namespace optimal_price_for_target_profit_l2374_237466

-- Define the cost to produce the souvenir
def production_cost : ℝ := 30

-- Define the lower and upper bounds of the selling price
def min_price : ℝ := production_cost
def max_price : ℝ := 54

-- Define the base price and corresponding daily sales
def base_price : ℝ := 40
def base_sales : ℝ := 80

-- Define the rate of change in sales per yuan increase in price
def sales_change_rate : ℝ := -2

-- Define the target daily profit
def target_profit : ℝ := 1200

-- Define the function for daily sales based on price
def daily_sales (price : ℝ) : ℝ :=
  base_sales + sales_change_rate * (price - base_price)

-- Define the function for daily profit based on price
def daily_profit (price : ℝ) : ℝ :=
  (price - production_cost) * daily_sales price

-- Theorem statement
theorem optimal_price_for_target_profit :
  ∃ (price : ℝ), min_price ≤ price ∧ price ≤ max_price ∧ daily_profit price = target_profit ∧ price = 50 := by
  sorry

end optimal_price_for_target_profit_l2374_237466


namespace congruence_power_l2374_237429

theorem congruence_power (a b m n : ℕ) (h : a ≡ b [MOD m]) : a^n ≡ b^n [MOD m] := by
  sorry

end congruence_power_l2374_237429


namespace problem_1_problem_2_l2374_237486

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a > 0) (h4 : b > 0) :
  a + b = 8 := by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) (h : |a - 2| + |b - 3| + |c - 4| = 0) :
  a + b + c = 9 := by sorry

end problem_1_problem_2_l2374_237486


namespace shortest_path_length_l2374_237440

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a circle -/
structure Circle where
  radius : ℝ

/-- Represents a path on a triangle and circle -/
def ShortestPath (t : EquilateralTriangle) (c : Circle) : ℝ := sorry

/-- The theorem stating the length of the shortest path -/
theorem shortest_path_length 
  (t : EquilateralTriangle) 
  (c : Circle) 
  (h1 : t.sideLength = 2) 
  (h2 : c.radius = 1/2) : 
  ShortestPath t c = Real.sqrt (28/3) - 1 := by sorry

end shortest_path_length_l2374_237440


namespace at_most_two_solutions_l2374_237469

theorem at_most_two_solutions (a b c : ℝ) (ha : a > 2000) :
  ¬∃ (x₁ x₂ x₃ : ℤ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (|a * x₁^2 + b * x₁ + c| ≤ 1000) ∧
    (|a * x₂^2 + b * x₂ + c| ≤ 1000) ∧
    (|a * x₃^2 + b * x₃ + c| ≤ 1000) :=
by sorry

end at_most_two_solutions_l2374_237469


namespace sons_age_l2374_237401

theorem sons_age (son_age man_age : ℕ) : 
  man_age = 3 * son_age →
  man_age + 12 = 2 * (son_age + 12) →
  son_age = 12 := by
  sorry

end sons_age_l2374_237401


namespace triangle_f_sign_l2374_237475

/-- Triangle ABC with sides a ≤ b ≤ c, circumradius R, and inradius r -/
structure Triangle where
  a : Real
  b : Real
  c : Real
  R : Real
  r : Real
  h_sides : a ≤ b ∧ b ≤ c
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0

/-- The function f defined for the triangle -/
def f (t : Triangle) : Real := t.a + t.b - 2 * t.R - 2 * t.r

/-- Angle C of the triangle -/
noncomputable def angle_C (t : Triangle) : Real := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem triangle_f_sign (t : Triangle) :
  (f t > 0 ↔ angle_C t < Real.pi / 2) ∧
  (f t = 0 ↔ angle_C t = Real.pi / 2) ∧
  (f t < 0 ↔ angle_C t > Real.pi / 2) := by sorry

end triangle_f_sign_l2374_237475


namespace quadratic_equations_sum_l2374_237457

theorem quadratic_equations_sum (x y : ℝ) : 
  9 * x^2 - 36 * x - 81 = 0 → 
  y^2 + 6 * y + 9 = 0 → 
  (x + y = -1 + Real.sqrt 13) ∨ (x + y = -1 - Real.sqrt 13) := by
sorry

end quadratic_equations_sum_l2374_237457


namespace abc_sum_mod_five_l2374_237478

theorem abc_sum_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 5 = 1 →
  (4 * c) % 5 = 3 →
  (3 * b) % 5 = (2 + b) % 5 →
  (a + b + c) % 5 = 1 := by
sorry

end abc_sum_mod_five_l2374_237478


namespace first_number_is_24_l2374_237481

theorem first_number_is_24 (x : ℝ) : 
  (x + 35 + 58) / 3 = (19 + 51 + 29) / 3 + 6 → x = 24 := by
sorry

end first_number_is_24_l2374_237481


namespace functions_identical_functions_not_identical_l2374_237497

-- Part 1
theorem functions_identical (x : ℝ) (h : x ≠ 0) : x / x^2 = 1 / x := by sorry

-- Part 2
theorem functions_not_identical : ∃ x : ℝ, x ≠ Real.sqrt (x^2) := by sorry

end functions_identical_functions_not_identical_l2374_237497


namespace count_numbers_with_square_factor_eq_41_l2374_237477

def perfect_squares : List Nat := [4, 9, 16, 25, 36, 49, 64, 100]

def is_divisible_by_square (n : Nat) : Bool :=
  perfect_squares.any (λ s => n % s = 0)

def count_numbers_with_square_factor : Nat :=
  (List.range 100).filter is_divisible_by_square |>.length

theorem count_numbers_with_square_factor_eq_41 :
  count_numbers_with_square_factor = 41 := by
  sorry

end count_numbers_with_square_factor_eq_41_l2374_237477


namespace aaron_cards_found_l2374_237471

/-- Given that Aaron initially had 5 cards and ended up with 67 cards,
    prove that he found 62 cards. -/
theorem aaron_cards_found :
  let initial_cards : ℕ := 5
  let final_cards : ℕ := 67
  let cards_found := final_cards - initial_cards
  cards_found = 62 := by sorry

end aaron_cards_found_l2374_237471


namespace sum_of_coefficients_l2374_237428

theorem sum_of_coefficients (a b : ℚ) : 
  (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end sum_of_coefficients_l2374_237428


namespace cos_2alpha_minus_pi_3_l2374_237443

theorem cos_2alpha_minus_pi_3 (α : ℝ) 
  (h : Real.sin (α + π/6) - Real.cos α = 1/3) : 
  Real.cos (2*α - π/3) = 7/9 := by
  sorry

end cos_2alpha_minus_pi_3_l2374_237443


namespace sin_2theta_value_l2374_237458

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ ^ 2) ^ n = 3) : 
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 := by
  sorry

end sin_2theta_value_l2374_237458


namespace octal_digit_reversal_difference_l2374_237432

theorem octal_digit_reversal_difference (A B : Nat) : 
  A ≠ B → 
  A < 8 → 
  B < 8 → 
  ∃ k : Int, (8 * A + B) - (8 * B + A) = 7 * k ∧ k ≠ 0 := by
  sorry

end octal_digit_reversal_difference_l2374_237432


namespace quadratic_equation_solution_l2374_237493

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 2
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 7 ∧ x₂ = 3 - Real.sqrt 7 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end quadratic_equation_solution_l2374_237493


namespace cubes_volume_percentage_l2374_237450

def box_length : ℕ := 8
def box_width : ℕ := 5
def box_height : ℕ := 12
def cube_side : ℕ := 4

def cubes_per_dimension (box_dim : ℕ) (cube_dim : ℕ) : ℕ :=
  box_dim / cube_dim

def total_cubes : ℕ :=
  (cubes_per_dimension box_length cube_side) *
  (cubes_per_dimension box_width cube_side) *
  (cubes_per_dimension box_height cube_side)

def cube_volume : ℕ := cube_side ^ 3
def total_cubes_volume : ℕ := total_cubes * cube_volume
def box_volume : ℕ := box_length * box_width * box_height

theorem cubes_volume_percentage :
  (total_cubes_volume : ℚ) / (box_volume : ℚ) = 4/5 := by
  sorry

end cubes_volume_percentage_l2374_237450


namespace base_8_45327_equals_19159_l2374_237476

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_45327_equals_19159 :
  base_8_to_10 [7, 2, 3, 5, 4] = 19159 := by
  sorry

end base_8_45327_equals_19159_l2374_237476


namespace power_of_power_l2374_237489

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l2374_237489


namespace equation_solutions_l2374_237421

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 49 = 0 ↔ x = 7/2 ∨ x = -7/2) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end equation_solutions_l2374_237421


namespace amoeba_count_after_10_days_l2374_237414

/-- The number of amoebas in the puddle after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  3^days

/-- Theorem stating that after 10 days, there will be 59049 amoebas in the puddle -/
theorem amoeba_count_after_10_days : amoeba_count 10 = 59049 := by
  sorry

end amoeba_count_after_10_days_l2374_237414


namespace sufficient_not_necessary_l2374_237413

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4)) ∧ 
  (∃ a b : ℝ, (a + b > 4 ∧ a * b > 4) ∧ ¬(a > 2 ∧ b > 2)) :=
sorry

end sufficient_not_necessary_l2374_237413


namespace coefficient_x_cube_in_expansion_l2374_237479

theorem coefficient_x_cube_in_expansion : ∃ (c : ℤ), c = -10 ∧ 
  ∀ (x : ℝ), x * (x - 1)^5 = x^6 - 5*x^5 + 10*x^4 + c*x^3 + 5*x^2 - x := by
  sorry

end coefficient_x_cube_in_expansion_l2374_237479


namespace correct_freshman_count_l2374_237482

/-- The number of students the college needs to admit into the freshman class each year
    to maintain a total enrollment of 3400 students, given specific dropout rates for each class. -/
def requiredFreshmen : ℕ :=
  let totalEnrollment : ℕ := 3400
  let freshmanDropoutRate : ℚ := 1/3
  let sophomoreDropouts : ℕ := 40
  let juniorDropoutRate : ℚ := 1/10
  5727

/-- Theorem stating that the required number of freshmen is 5727 -/
theorem correct_freshman_count :
  requiredFreshmen = 5727 :=
by sorry

end correct_freshman_count_l2374_237482


namespace quadratic_inequality_range_l2374_237483

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by sorry

end quadratic_inequality_range_l2374_237483


namespace y_in_terms_of_x_l2374_237426

theorem y_in_terms_of_x (x y : ℝ) (h : y - 2*x = 6) : y = 2*x + 6 := by
  sorry

end y_in_terms_of_x_l2374_237426


namespace min_triangles_for_G_2008_l2374_237403

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- Defines the grid G_n --/
def G (n : Nat) : Set GridPoint :=
  {p : GridPoint | p.x ≥ 1 ∧ p.x ≤ n ∧ p.y ≥ 1 ∧ p.y ≤ n}

/-- Minimum number of triangles needed to cover a grid --/
def minTriangles (n : Nat) : Nat :=
  if n = 2 then 1
  else if n = 3 then 2
  else (n * n) / 3 * 2

/-- Theorem stating the minimum number of triangles needed to cover G_2008 --/
theorem min_triangles_for_G_2008 :
  minTriangles 2008 = 1338 :=
sorry

end min_triangles_for_G_2008_l2374_237403


namespace stop_after_seventh_shot_probability_value_l2374_237451

/-- The maximum number of shots allowed -/
def max_shots : ℕ := 10

/-- The probability of making a shot for student A -/
def shot_probability : ℚ := 2/3

/-- Calculate the score based on the shot number when the student stops -/
def score (n : ℕ) : ℕ := 12 - n

/-- The probability of the specific sequence of shots leading to stopping after the 7th shot -/
def stop_after_seventh_shot_probability : ℚ :=
  (1 - shot_probability) * shot_probability * (1 - shot_probability) *
  1 * (1 - shot_probability) * shot_probability * shot_probability

theorem stop_after_seventh_shot_probability_value :
  stop_after_seventh_shot_probability = 8/729 :=
sorry

end stop_after_seventh_shot_probability_value_l2374_237451


namespace road_trip_cost_l2374_237445

theorem road_trip_cost (initial_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  initial_friends = 5 →
  additional_friends = 3 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    total_cost / initial_friends - total_cost / (initial_friends + additional_friends) = cost_decrease ∧
    total_cost = 200 :=
by sorry

end road_trip_cost_l2374_237445


namespace inequality_proof_l2374_237415

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 1) :
  a^x + b^y + c^z ≥ (4*a*b*c*x*y*z) / (x + y + z - 3)^2 := by
sorry

end inequality_proof_l2374_237415


namespace arithmetic_geometric_mean_problem_l2374_237490

theorem arithmetic_geometric_mean_problem (p q r s : ℝ) : 
  (p + q) / 2 = 10 →
  (q + r) / 2 = 22 →
  (p * q * s)^(1/3) = 20 →
  r - p = 24 := by
sorry

end arithmetic_geometric_mean_problem_l2374_237490


namespace largest_special_square_l2374_237487

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def last_two_digits (n : ℕ) : ℕ := n % 100

def remove_last_two_digits (n : ℕ) : ℕ := (n - last_two_digits n) / 100

theorem largest_special_square : 
  ∀ n : ℕ, 
    (is_square n ∧ 
     n % 100 ≠ 0 ∧ 
     is_square (remove_last_two_digits n)) →
    n ≤ 1681 :=
sorry

end largest_special_square_l2374_237487


namespace binary_to_hexadecimal_conversion_l2374_237400

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  sorry

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hexadecimal (decimal : ℕ) : List ℕ :=
  sorry

theorem binary_to_hexadecimal_conversion :
  let binary : List Bool := [true, false, true, true, false, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let hexadecimal : List ℕ := decimal_to_hexadecimal decimal
  hexadecimal = [2, 2, 5] := by sorry

end binary_to_hexadecimal_conversion_l2374_237400


namespace complex_equation_solution_l2374_237431

theorem complex_equation_solution (z : ℂ) : (1 - I)^2 * z = 3 + 2*I → z = -1 + (3/2)*I := by
  sorry

end complex_equation_solution_l2374_237431


namespace arithmetic_sequence_third_term_l2374_237422

/-- Given an arithmetic sequence of 6 terms where the first term is 11 and the last term is 51,
    prove that the third term is 27. -/
theorem arithmetic_sequence_third_term :
  ∀ (seq : Fin 6 → ℝ),
    (∀ i j : Fin 6, seq (i + 1) - seq i = seq (j + 1) - seq j) →  -- arithmetic sequence
    seq 0 = 11 →  -- first term is 11
    seq 5 = 51 →  -- last term is 51
    seq 2 = 27 :=  -- third term is 27
by sorry

end arithmetic_sequence_third_term_l2374_237422


namespace x_plus_inv_x_eight_l2374_237416

theorem x_plus_inv_x_eight (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end x_plus_inv_x_eight_l2374_237416


namespace vacation_animals_l2374_237434

/-- The total number of animals bought on the last vacation --/
def total_animals (rainbowfish clowns tetras guppies angelfish cichlids : ℕ) : ℕ :=
  rainbowfish + clowns + tetras + guppies + angelfish + cichlids

/-- Theorem stating the total number of animals bought on the last vacation --/
theorem vacation_animals :
  ∃ (rainbowfish clowns tetras guppies angelfish cichlids : ℕ),
    rainbowfish = 40 ∧
    cichlids = rainbowfish / 2 ∧
    angelfish = cichlids + 10 ∧
    guppies = 3 * angelfish ∧
    clowns = 2 * guppies ∧
    tetras = 5 * clowns ∧
    total_animals rainbowfish clowns tetras guppies angelfish cichlids = 1260 := by
  sorry


end vacation_animals_l2374_237434


namespace andrews_grapes_l2374_237424

/-- The amount of grapes Andrew purchased -/
def grapes : ℕ := sorry

/-- The price of grapes per kg -/
def grape_price : ℕ := 98

/-- The amount of mangoes Andrew purchased in kg -/
def mangoes : ℕ := 7

/-- The price of mangoes per kg -/
def mango_price : ℕ := 50

/-- The total amount Andrew paid -/
def total_paid : ℕ := 1428

theorem andrews_grapes : 
  grapes * grape_price + mangoes * mango_price = total_paid ∧ grapes = 11 := by sorry

end andrews_grapes_l2374_237424


namespace imaginary_part_of_complex_fraction_l2374_237494

theorem imaginary_part_of_complex_fraction : Complex.im (Complex.I / (1 - Complex.I)) = 1 / 2 := by
  sorry

end imaginary_part_of_complex_fraction_l2374_237494


namespace fourth_term_equals_seven_l2374_237442

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2, prove that a_4 = 7 -/
theorem fourth_term_equals_seven (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2) : 
    a 4 = 7 := by
  sorry

end fourth_term_equals_seven_l2374_237442


namespace consecutive_integers_square_difference_consecutive_odd_integers_square_difference_l2374_237480

theorem consecutive_integers_square_difference (n : ℤ) : 
  ∃ k : ℤ, (n + 2)^2 - n^2 = 4 * k :=
sorry

theorem consecutive_odd_integers_square_difference (n : ℤ) : 
  ∃ k : ℤ, (2*n + 3)^2 - (2*n - 1)^2 = 8 * k :=
sorry

end consecutive_integers_square_difference_consecutive_odd_integers_square_difference_l2374_237480


namespace initial_girls_count_l2374_237454

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 3 / 10 →
  ((initial_girls : ℚ) - 3) / total = 1 / 5 →
  initial_girls = 9 :=
sorry

end initial_girls_count_l2374_237454


namespace cookies_sold_proof_l2374_237463

/-- The number of packs of cookies sold by Robyn -/
def robyn_sales : ℕ := 47

/-- The number of packs of cookies sold by Lucy -/
def lucy_sales : ℕ := 29

/-- The total number of packs of cookies sold by Robyn and Lucy -/
def total_sales : ℕ := robyn_sales + lucy_sales

theorem cookies_sold_proof : total_sales = 76 := by
  sorry

end cookies_sold_proof_l2374_237463


namespace tile_coverage_proof_l2374_237498

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the ceiling of a fraction represented as a numerator and denominator -/
def ceilingDiv (n d : ℕ) : ℕ := (n + d - 1) / d

theorem tile_coverage_proof (tile : Dimensions) (room : Dimensions) : 
  tile.length = 2 → 
  tile.width = 5 → 
  room.length = feetToInches 3 → 
  room.width = feetToInches 8 → 
  ceilingDiv (area room) (area tile) = 346 := by
  sorry

end tile_coverage_proof_l2374_237498


namespace probability_all_green_apples_l2374_237436

def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples : ℕ := 4
def chosen_apples : ℕ := 3

theorem probability_all_green_apples :
  (Nat.choose green_apples chosen_apples : ℚ) / (Nat.choose total_apples chosen_apples) = 1 / 30 := by
  sorry

end probability_all_green_apples_l2374_237436


namespace min_value_is_zero_l2374_237499

/-- The quadratic function we're minimizing -/
def f (x y : ℝ) : ℝ := 9*x^2 - 24*x*y + 19*y^2 - 6*x - 9*y + 12

/-- The minimum value of f over all real x and y is 0 -/
theorem min_value_is_zero : 
  ∀ x y : ℝ, f x y ≥ 0 ∧ ∃ x₀ y₀ : ℝ, f x₀ y₀ = 0 :=
sorry

end min_value_is_zero_l2374_237499


namespace graduating_class_size_l2374_237430

theorem graduating_class_size 
  (geometry_count : ℕ) 
  (biology_count : ℕ) 
  (overlap_difference : ℕ) 
  (h1 : geometry_count = 144) 
  (h2 : biology_count = 119) 
  (h3 : overlap_difference = 88) :
  geometry_count + biology_count - (biology_count - overlap_difference) = 232 := by
  sorry

end graduating_class_size_l2374_237430


namespace forty_sheep_eat_forty_bags_l2374_237465

/-- The number of bags of grass eaten by a group of sheep -/
def bags_eaten (num_sheep : ℕ) (num_days : ℕ) : ℕ :=
  num_sheep * (num_days / 40)

/-- Theorem: 40 sheep eat 40 bags of grass in 40 days -/
theorem forty_sheep_eat_forty_bags :
  bags_eaten 40 40 = 40 := by
  sorry

end forty_sheep_eat_forty_bags_l2374_237465


namespace student_tickets_sold_l2374_237410

/-- Proves the number of student tickets sold given total tickets, total money, and ticket prices -/
theorem student_tickets_sold 
  (total_tickets : ℕ) 
  (total_money : ℕ) 
  (student_price : ℕ) 
  (nonstudent_price : ℕ) 
  (h1 : total_tickets = 821)
  (h2 : total_money = 1933)
  (h3 : student_price = 2)
  (h4 : nonstudent_price = 3) :
  ∃ (student_tickets : ℕ) (nonstudent_tickets : ℕ),
    student_tickets + nonstudent_tickets = total_tickets ∧
    student_tickets * student_price + nonstudent_tickets * nonstudent_price = total_money ∧
    student_tickets = 530 := by
  sorry

end student_tickets_sold_l2374_237410


namespace rotated_rectangle_area_fraction_l2374_237439

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a rectangle on a 2D grid -/
structure Rectangle where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rectangle given its vertices -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry

/-- Calculates the area of a square grid -/
def gridArea (size : ℤ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem rotated_rectangle_area_fraction :
  let grid_size : ℤ := 6
  let r : Rectangle := {
    v1 := { x := 2, y := 2 },
    v2 := { x := 4, y := 4 },
    v3 := { x := 2, y := 4 },
    v4 := { x := 4, y := 6 }
  }
  rectangleArea r / gridArea grid_size = Real.sqrt 2 / 9 := by
  sorry

end rotated_rectangle_area_fraction_l2374_237439


namespace parallel_vectors_x_value_l2374_237418

/-- Two 2D vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = -6 := by
sorry

end parallel_vectors_x_value_l2374_237418


namespace donnas_card_shop_wage_l2374_237408

/-- Calculates Donna's hourly wage at the card shop based on her weekly earnings --/
theorem donnas_card_shop_wage (
  dog_walking_hours : ℕ)
  (dog_walking_rate : ℚ)
  (dog_walking_days : ℕ)
  (card_shop_hours : ℕ)
  (card_shop_days : ℕ)
  (babysitting_hours : ℕ)
  (babysitting_rate : ℚ)
  (total_earnings : ℚ)
  (h1 : dog_walking_hours = 2)
  (h2 : dog_walking_rate = 10)
  (h3 : dog_walking_days = 5)
  (h4 : card_shop_hours = 2)
  (h5 : card_shop_days = 5)
  (h6 : babysitting_hours = 4)
  (h7 : babysitting_rate = 10)
  (h8 : total_earnings = 305) :
  (total_earnings - (dog_walking_hours * dog_walking_rate * dog_walking_days + babysitting_hours * babysitting_rate)) / (card_shop_hours * card_shop_days) = 33/2 := by
  sorry

#eval (33 : ℚ) / 2

end donnas_card_shop_wage_l2374_237408


namespace max_ski_trips_l2374_237417

/-- Represents the time in minutes for a single trip up and down the mountain -/
def trip_time : ℕ := 15 + 5

/-- Represents the total available time in minutes -/
def total_time : ℕ := 2 * 60

/-- Theorem stating the maximum number of times a person can ski down the mountain in 2 hours -/
theorem max_ski_trips : (total_time / trip_time : ℕ) = 6 := by
  sorry

end max_ski_trips_l2374_237417


namespace seeds_per_row_l2374_237448

/-- Given a garden with potatoes planted in rows, this theorem proves
    the number of seeds in each row when the total number of potatoes
    and the number of rows are known. -/
theorem seeds_per_row (total_potatoes : ℕ) (num_rows : ℕ) 
    (h1 : total_potatoes = 54) 
    (h2 : num_rows = 6) 
    (h3 : total_potatoes % num_rows = 0) : 
  total_potatoes / num_rows = 9 := by
  sorry

end seeds_per_row_l2374_237448


namespace tan_plus_4sin_30_deg_l2374_237412

theorem tan_plus_4sin_30_deg :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  let tan_30 : ℝ := sin_30 / cos_30
  tan_30 + 4 * sin_30 = (Real.sqrt 3 + 6) / 3 := by
  sorry

end tan_plus_4sin_30_deg_l2374_237412


namespace parabola_point_ordering_l2374_237491

-- Define the parabola
def Parabola := ℝ → ℝ

-- Define the properties of the parabola
axiom parabola_increasing (f : Parabola) : ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂
axiom parabola_decreasing (f : Parabola) : ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂

-- Define the points on the parabola
def A (f : Parabola) := f (-2)
def B (f : Parabola) := f 1
def C (f : Parabola) := f 3

-- State the theorem
theorem parabola_point_ordering (f : Parabola) : B f < C f ∧ C f < A f := by sorry

end parabola_point_ordering_l2374_237491


namespace parameterized_line_problem_l2374_237452

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → (Fin 3 → ℝ)

/-- The problem statement as a theorem -/
theorem parameterized_line_problem :
  ∀ (line : ParameterizedLine),
    (line.vector 1 = ![2, 4, 9]) →
    (line.vector 3 = ![1, 1, 2]) →
    (line.vector 4 = ![0.5, -0.5, -1.5]) := by
  sorry

end parameterized_line_problem_l2374_237452


namespace ratio_proof_l2374_237492

theorem ratio_proof (N : ℝ) 
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 20)
  (h2 : 0.4 * N = 240) :
  20 / ((1 / 3) * (2 / 5) * N) = 2 / 15 := by
sorry

end ratio_proof_l2374_237492


namespace original_number_proof_l2374_237407

theorem original_number_proof (x : ℝ) : x * 1.1 = 660 → x = 600 := by
  sorry

end original_number_proof_l2374_237407


namespace rubiks_cube_return_to_original_state_l2374_237461

theorem rubiks_cube_return_to_original_state 
  {S : Type} [Finite S] (f : S → S) : 
  ∃ n : ℕ+, ∀ x : S, (f^[n] x = x) := by
  sorry

end rubiks_cube_return_to_original_state_l2374_237461


namespace hyperbola_equation_l2374_237449

/-- Given a hyperbola and an ellipse with shared foci and related eccentricities,
    prove that the hyperbola has the equation x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), x^2/16 + y^2/9 = 1) ∧
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c^2 = 16 - 9) ∧
  (∃ (e_h e_e : ℝ), e_h = c/a ∧ e_e = c/4 ∧ e_h = 2*e_e) →
  a^2 = 4 ∧ b^2 = 3 :=
by sorry

end hyperbola_equation_l2374_237449


namespace simple_interest_rate_approx_l2374_237409

/-- The rate of simple interest given principal, amount, and time -/
def simple_interest_rate (principal amount : ℕ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that the rate of simple interest is approximately 3.53% -/
theorem simple_interest_rate_approx :
  let rate := simple_interest_rate 12000 17500 13
  ∃ ε > 0, abs (rate - 353/100) < ε ∧ ε < 1/100 :=
sorry

end simple_interest_rate_approx_l2374_237409


namespace remainder_product_l2374_237460

theorem remainder_product (n : ℕ) (d : ℕ) (m : ℕ) (h : d ≠ 0) :
  (n % d) * m = 33 ↔ n = 2345678 ∧ d = 128 ∧ m = 3 := by
  sorry

end remainder_product_l2374_237460


namespace davids_math_marks_l2374_237474

def english_marks : ℕ := 76
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def num_subjects : ℕ := 5

theorem davids_math_marks :
  ∃ (math_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / num_subjects = average_marks ∧
    math_marks = 65 :=
by sorry

end davids_math_marks_l2374_237474


namespace max_value_of_f_on_interval_l2374_237404

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 3 → f y ≤ f x :=
sorry

end max_value_of_f_on_interval_l2374_237404


namespace hassans_orange_trees_l2374_237472

/-- Represents the number of trees in an orchard --/
structure Orchard :=
  (orange : ℕ)
  (apple : ℕ)

/-- The total number of trees in an orchard --/
def Orchard.total (o : Orchard) : ℕ := o.orange + o.apple

theorem hassans_orange_trees :
  ∀ (ahmed hassan : Orchard),
  ahmed.orange = 8 →
  ahmed.apple = 4 * hassan.apple →
  hassan.apple = 1 →
  ahmed.total = hassan.total + 9 →
  hassan.orange = 2 := by
sorry

end hassans_orange_trees_l2374_237472


namespace sum_of_cubes_l2374_237488

theorem sum_of_cubes (a b c d e : ℕ) : 
  a ∈ ({0, 1, 2} : Set ℕ) → 
  b ∈ ({0, 1, 2} : Set ℕ) → 
  c ∈ ({0, 1, 2} : Set ℕ) → 
  d ∈ ({0, 1, 2} : Set ℕ) → 
  e ∈ ({0, 1, 2} : Set ℕ) → 
  a + b + c + d + e = 6 → 
  a^2 + b^2 + c^2 + d^2 + e^2 = 10 → 
  a^3 + b^3 + c^3 + d^3 + e^3 = 18 :=
by sorry

end sum_of_cubes_l2374_237488


namespace line_proof_l2374_237453

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 2 = 0
def line3 (x y : ℝ) : Prop := x + y = 0
def line4 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    line4 x y ∧
    perpendicular (-1) 1 := by
  sorry

end line_proof_l2374_237453


namespace point_on_line_value_l2374_237411

theorem point_on_line_value (a b : ℝ) : 
  b = 3 * a - 2 → 2 * b - 6 * a + 2 = -2 := by
  sorry

end point_on_line_value_l2374_237411


namespace functional_equation_solution_l2374_237406

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the only function satisfying the functional equation is f(x) = 1 - x²/2 -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → ∀ x : ℝ, f x = 1 - x^2 / 2 :=
by sorry

end functional_equation_solution_l2374_237406


namespace perpendicular_line_equation_l2374_237485

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation :
  ∃ (l : Line),
    l.passesThrough ⟨1, -2⟩ ∧
    l.isPerpendicular ⟨2, 3, -1⟩ ∧
    l = ⟨3, -2, -7⟩ := by
  sorry

end perpendicular_line_equation_l2374_237485


namespace complex_second_quadrant_x_range_l2374_237405

theorem complex_second_quadrant_x_range (x : ℝ) :
  let z : ℂ := (x + Complex.I) / (3 - Complex.I)
  (z.re < 0 ∧ z.im > 0) → (-3 < x ∧ x < 1/3) :=
by sorry

end complex_second_quadrant_x_range_l2374_237405


namespace max_puns_purchase_l2374_237455

/-- Represents the cost of each item --/
structure ItemCosts where
  pin : ℕ
  pon : ℕ
  pun : ℕ

/-- Represents the quantity of each item purchased --/
structure Purchase where
  pins : ℕ
  pons : ℕ
  puns : ℕ

/-- Calculates the total cost of a purchase --/
def totalCost (costs : ItemCosts) (purchase : Purchase) : ℕ :=
  costs.pin * purchase.pins + costs.pon * purchase.pons + costs.pun * purchase.puns

/-- Checks if a purchase is valid (at least one of each item) --/
def isValidPurchase (purchase : Purchase) : Prop :=
  purchase.pins ≥ 1 ∧ purchase.pons ≥ 1 ∧ purchase.puns ≥ 1

/-- The main theorem statement --/
theorem max_puns_purchase (costs : ItemCosts) (budget : ℕ) : 
  costs.pin = 3 → costs.pon = 4 → costs.pun = 9 → budget = 108 →
  ∃ (max_puns : ℕ), 
    (∃ (p : Purchase), isValidPurchase p ∧ totalCost costs p = budget ∧ p.puns = max_puns) ∧
    (∀ (p : Purchase), isValidPurchase p → totalCost costs p = budget → p.puns ≤ max_puns) ∧
    max_puns = 10 :=
sorry

end max_puns_purchase_l2374_237455


namespace unique_p_for_three_positive_integer_roots_l2374_237441

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

/-- Predicate to check if a number is a positive integer -/
def is_positive_integer (x : ℝ) : Prop :=
  x > 0 ∧ ∃ n : ℕ, x = n

/-- The main theorem -/
theorem unique_p_for_three_positive_integer_roots :
  ∃! p : ℝ, ∃ x y z : ℝ,
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    is_positive_integer x ∧ is_positive_integer y ∧ is_positive_integer z ∧
    cubic_equation p x = 0 ∧ cubic_equation p y = 0 ∧ cubic_equation p z = 0 ∧
    p = 76 :=
sorry

end unique_p_for_three_positive_integer_roots_l2374_237441


namespace movie_ratio_is_half_l2374_237464

/-- The ratio of movies Theresa saw in 2009 to movies Timothy saw in 2009 -/
def movie_ratio (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℚ :=
  theresa_2009 / timothy_2009

theorem movie_ratio_is_half :
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2009 = 24 →
    timothy_2010 = timothy_2009 + 7 →
    theresa_2010 = 2 * timothy_2010 →
    timothy_2009 + theresa_2009 + timothy_2010 + theresa_2010 = 129 →
    movie_ratio timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 1/2 :=
by sorry


end movie_ratio_is_half_l2374_237464


namespace trigonometric_identity_l2374_237437

theorem trigonometric_identity : 
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end trigonometric_identity_l2374_237437


namespace nonagon_diagonals_l2374_237419

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nonagon_diagonals_l2374_237419


namespace two_integer_b_values_l2374_237423

theorem two_integer_b_values : 
  ∃! (S : Finset ℤ), 
    (Finset.card S = 2) ∧ 
    (∀ b ∈ S, ∃! (T : Finset ℤ), 
      (Finset.card T = 2) ∧ 
      (∀ x ∈ T, x^2 + b*x + 3 ≤ 0) ∧
      (∀ x : ℤ, x^2 + b*x + 3 ≤ 0 → x ∈ T)) := by
sorry

end two_integer_b_values_l2374_237423


namespace division_problem_l2374_237425

theorem division_problem (A : ℕ) (h : 34 = A * 6 + 4) : A = 5 := by
  sorry

end division_problem_l2374_237425


namespace quadratic_coefficient_bounds_l2374_237438

theorem quadratic_coefficient_bounds (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hroots : b^2 - 4*a*c ≥ 0) : 
  (max a (max b c) ≥ 4/9 * (a + b + c)) ∧ 
  (min a (min b c) ≤ 1/4 * (a + b + c)) := by
sorry

end quadratic_coefficient_bounds_l2374_237438


namespace chord_equation_l2374_237459

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a chord of the ellipse
def is_chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

-- Define M as the midpoint of the chord
def M_bisects_chord (A B : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1)/2, (A.2 + B.2)/2)

-- The theorem to prove
theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  is_chord A B →
  M_bisects_chord A B →
  ∃ k m : ℝ, k = -1/2 ∧ m = 4 ∧ 
    ∀ x y : ℝ, (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → y = k*x + m :=
by sorry

end chord_equation_l2374_237459


namespace yoongi_stack_higher_l2374_237456

/-- The height of Box A in centimeters -/
def box_a_height : ℝ := 3

/-- The height of Box B in centimeters -/
def box_b_height : ℝ := 3.5

/-- The number of Box A stacked by Taehyung -/
def taehyung_boxes : ℕ := 16

/-- The number of Box B stacked by Yoongi -/
def yoongi_boxes : ℕ := 14

/-- The total height of Taehyung's stack in centimeters -/
def taehyung_stack_height : ℝ := box_a_height * taehyung_boxes

/-- The total height of Yoongi's stack in centimeters -/
def yoongi_stack_height : ℝ := box_b_height * yoongi_boxes

theorem yoongi_stack_higher :
  yoongi_stack_height > taehyung_stack_height ∧
  yoongi_stack_height - taehyung_stack_height = 1 :=
by sorry

end yoongi_stack_higher_l2374_237456


namespace baker_sales_change_l2374_237427

/-- A baker's weekly pastry sales problem --/
theorem baker_sales_change (price : ℕ) (days_per_week : ℕ) (monday_sales : ℕ) (avg_sales : ℕ) 
  (h1 : price = 5)
  (h2 : days_per_week = 7)
  (h3 : monday_sales = 2)
  (h4 : avg_sales = 5) :
  ∃ (daily_change : ℕ),
    daily_change = 1 ∧
    monday_sales + 
    (monday_sales + daily_change) + 
    (monday_sales + 2 * daily_change) + 
    (monday_sales + 3 * daily_change) + 
    (monday_sales + 4 * daily_change) + 
    (monday_sales + 5 * daily_change) + 
    (monday_sales + 6 * daily_change) = days_per_week * avg_sales :=
by
  sorry

end baker_sales_change_l2374_237427


namespace intersection_of_P_and_Q_l2374_237468

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by
  sorry

end intersection_of_P_and_Q_l2374_237468


namespace factorial_division_l2374_237444

theorem factorial_division : 
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 :=
by
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  sorry

end factorial_division_l2374_237444


namespace largest_integer_fraction_l2374_237484

theorem largest_integer_fraction (n : ℤ) : (n / 11 : ℚ) < 2/3 ↔ n ≤ 7 :=
  sorry

end largest_integer_fraction_l2374_237484


namespace value_of_2a_plus_b_l2374_237495

theorem value_of_2a_plus_b (a b : ℝ) (h : |a + 2| + (b - 5)^2 = 0) : 2*a + b = 1 := by
  sorry

end value_of_2a_plus_b_l2374_237495


namespace tangent_lines_existence_l2374_237433

/-- The range of a for which there exist two different lines tangent to both f(x) and g(x) -/
theorem tangent_lines_existence (a : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₃ ∧ x₁ > 0 ∧ x₃ > 0 ∧
    (2 + a * Real.log x₁) + (a / x₁) * (x₂ - x₁) = (a * x₂^2 + 1) ∧
    (2 + a * Real.log x₃) + (a / x₃) * (x₄ - x₃) = (a * x₄^2 + 1)) ↔
  (a < 0 ∨ a > 2 / (1 + Real.log 2)) :=
sorry

end tangent_lines_existence_l2374_237433


namespace chessboard_cannot_be_tiled_chessboard_with_corner_removed_cannot_be_tiled_l2374_237435

/-- Represents a chessboard -/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a triomino -/
structure Triomino where
  length : Nat
  width : Nat

/-- Function to check if a chessboard can be tiled with triominoes -/
def canBeTiled (board : Chessboard) (triomino : Triomino) : Prop :=
  (board.rows * board.cols) % (triomino.length * triomino.width) = 0

/-- Function to check if a chessboard with one corner removed can be tiled with triominoes -/
def canBeTiledWithCornerRemoved (board : Chessboard) (triomino : Triomino) : Prop :=
  ∃ (colorA colorB colorC : Nat),
    colorA + colorB + colorC = board.rows * board.cols - 1 ∧
    colorA = colorB ∧ colorB = colorC

/-- Theorem: An 8x8 chessboard cannot be tiled with 3x1 triominoes -/
theorem chessboard_cannot_be_tiled :
  ¬ canBeTiled ⟨8, 8⟩ ⟨3, 1⟩ :=
sorry

/-- Theorem: An 8x8 chessboard with one corner removed cannot be tiled with 3x1 triominoes -/
theorem chessboard_with_corner_removed_cannot_be_tiled :
  ¬ canBeTiledWithCornerRemoved ⟨8, 8⟩ ⟨3, 1⟩ :=
sorry

end chessboard_cannot_be_tiled_chessboard_with_corner_removed_cannot_be_tiled_l2374_237435


namespace max_area_inscribed_rectangle_l2374_237470

/-- Given a parabola y^2 = 2px bounded by x = a, the maximum area of an inscribed rectangle 
    with its midline on the parabola's axis is (4a/3) * sqrt(2ap/3) -/
theorem max_area_inscribed_rectangle (p a : ℝ) (hp : p > 0) (ha : a > 0) :
  let parabola := fun y : ℝ => y^2 / (2*p)
  let bound := a
  let inscribed_rectangle_area := fun x : ℝ => 2 * (a - x) * Real.sqrt (2*p*x)
  ∃ max_area : ℝ, max_area = (4*a/3) * Real.sqrt (2*a*p/3) ∧
    ∀ x, 0 < x ∧ x < a → inscribed_rectangle_area x ≤ max_area :=
by sorry

end max_area_inscribed_rectangle_l2374_237470


namespace caravan_hens_l2374_237446

def caravan (num_hens : ℕ) : Prop :=
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let total_heads : ℕ := num_hens + num_goats + num_camels + num_keepers
  let total_feet : ℕ := 2 * num_hens + 4 * num_goats + 4 * num_camels + 2 * num_keepers
  total_feet = total_heads + 224

theorem caravan_hens : ∃ (num_hens : ℕ), caravan num_hens ∧ num_hens = 50 := by
  sorry

end caravan_hens_l2374_237446


namespace simplify_and_abs_l2374_237496

theorem simplify_and_abs (a : ℝ) (h : a = -2) : 
  |12 * a^5 / (72 * a^3)| = 2/3 := by
  sorry

end simplify_and_abs_l2374_237496


namespace similar_triangles_side_length_l2374_237420

-- Define the triangles and their side lengths
structure Triangle :=
  (a b c : ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

-- State the theorem
theorem similar_triangles_side_length 
  (PQR STU : Triangle) 
  (h_similar : similar PQR STU) 
  (h_PQ : PQR.a = 7) 
  (h_QR : PQR.b = 10) 
  (h_ST : STU.a = 4.9) : 
  STU.b = 7 := by
  sorry

end similar_triangles_side_length_l2374_237420
