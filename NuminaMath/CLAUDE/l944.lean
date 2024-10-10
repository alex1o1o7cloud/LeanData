import Mathlib

namespace prep_time_score_relation_student_score_for_six_hours_l944_94497

/-- Represents the direct variation between score and preparation time -/
def score_variation (prep_time : ℝ) (score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ score = k * prep_time

/-- Theorem stating the relationship between preparation time and test score -/
theorem prep_time_score_relation (initial_prep_time initial_score new_prep_time : ℝ) :
  initial_prep_time > 0 →
  initial_score > 0 →
  new_prep_time > 0 →
  score_variation initial_prep_time initial_score →
  score_variation new_prep_time (new_prep_time * initial_score / initial_prep_time) :=
by sorry

/-- Main theorem proving the specific case from the problem -/
theorem student_score_for_six_hours :
  let initial_prep_time : ℝ := 4
  let initial_score : ℝ := 80
  let new_prep_time : ℝ := 6
  score_variation initial_prep_time initial_score →
  score_variation new_prep_time 120 :=
by sorry

end prep_time_score_relation_student_score_for_six_hours_l944_94497


namespace geometric_sequence_product_l944_94491

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation 3x^2 - 2x - 6 = 0 -/
def are_roots (x y : ℝ) : Prop :=
  3 * x^2 - 2 * x - 6 = 0 ∧ 3 * y^2 - 2 * y - 6 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  are_roots (a 1) (a 10) →
  a 4 * a 7 = -2 :=
sorry

end geometric_sequence_product_l944_94491


namespace incorrect_calculation_l944_94443

theorem incorrect_calculation : ¬(3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end incorrect_calculation_l944_94443


namespace baseball_card_value_l944_94439

/-- The value of a baseball card after four years of depreciation --/
def card_value (initial_value : ℝ) (year1_decrease year2_decrease year3_decrease year4_decrease : ℝ) : ℝ :=
  initial_value * (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease) * (1 - year4_decrease)

/-- Theorem stating the final value of the baseball card after four years of depreciation --/
theorem baseball_card_value : 
  card_value 100 0.10 0.12 0.08 0.05 = 69.2208 := by
  sorry


end baseball_card_value_l944_94439


namespace glendas_average_speed_l944_94450

/-- Calculates the average speed given initial and final odometer readings and total time -/
def average_speed (initial_reading : ℕ) (final_reading : ℕ) (total_time : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

/-- Theorem: Glenda's average speed is 55 miles per hour -/
theorem glendas_average_speed :
  let initial_reading := 1221
  let final_reading := 1881
  let total_time := 12
  average_speed initial_reading final_reading total_time = 55 := by
  sorry

end glendas_average_speed_l944_94450


namespace smallest_n_for_powers_l944_94444

theorem smallest_n_for_powers : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), 3^n = a^4) ∧ 
  (∃ (b : ℕ), 2^n = b^6) ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (c : ℕ), 3^m = c^4) → (∃ (d : ℕ), 2^m = d^6) → m ≥ n) ∧
  n = 12 :=
sorry

end smallest_n_for_powers_l944_94444


namespace koi_added_per_day_proof_l944_94456

/-- The number of koi fish added per day to the tank -/
def koi_added_per_day : ℕ := 2

/-- The initial total number of fish in the tank -/
def initial_total_fish : ℕ := 280

/-- The number of goldfish added per day -/
def goldfish_added_per_day : ℕ := 5

/-- The number of days in 3 weeks -/
def days_in_three_weeks : ℕ := 21

/-- The final number of goldfish in the tank -/
def final_goldfish : ℕ := 200

/-- The final number of koi fish in the tank -/
def final_koi : ℕ := 227

theorem koi_added_per_day_proof :
  koi_added_per_day = 2 :=
by sorry

end koi_added_per_day_proof_l944_94456


namespace complement_of_M_in_U_l944_94440

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end complement_of_M_in_U_l944_94440


namespace quadratic_equation_positive_roots_l944_94479

theorem quadratic_equation_positive_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m+2)*x + m + 5 = 0 → x > 0) ↔ -5 < m ∧ m ≤ -4 :=
sorry

end quadratic_equation_positive_roots_l944_94479


namespace fraction_inequality_l944_94421

theorem fraction_inequality (a b : ℝ) :
  (b > 0 ∧ 0 > a → 1/a < 1/b) ∧
  (0 > a ∧ a > b → 1/a < 1/b) ∧
  (a > b ∧ b > 0 → 1/a < 1/b) :=
by sorry

end fraction_inequality_l944_94421


namespace cubic_sum_equality_l944_94423

theorem cubic_sum_equality (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) :
  x^3 + 2*y^3 + 2*z^3 + 6*x*y*z = 24 := by
  sorry

end cubic_sum_equality_l944_94423


namespace student_count_bound_l944_94453

theorem student_count_bound (N M k ℓ : ℕ) (h1 : M = k * N / 100) 
  (h2 : 100 * (M + 1) = ℓ * (N + 3)) (h3 : ℓ < 100) : N ≤ 197 := by
  sorry

end student_count_bound_l944_94453


namespace natural_number_squares_l944_94496

theorem natural_number_squares (x y : ℕ) : 
  1 + x + x^2 + x^3 + x^4 = y^2 ↔ (x = 0 ∧ y = 1) ∨ (x = 3 ∧ y = 11) := by
  sorry

end natural_number_squares_l944_94496


namespace sacred_words_count_l944_94458

-- Define the number of letters in the alien script
variable (n : ℕ)

-- Define the length of sacred words
variable (k : ℕ)

-- Condition that k is less than half of n
variable (h : k < n / 2)

-- Define a function to calculate the number of sacred k-words
def num_sacred_words (n k : ℕ) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * Nat.factorial k / k

-- Theorem statement
theorem sacred_words_count (n k : ℕ) (h : k < n / 2) :
  num_sacred_words n k = n * Nat.choose (n - k - 1) (k - 1) * Nat.factorial k / k :=
by sorry

-- Example for n = 10 and k = 4
example : num_sacred_words 10 4 = 600 :=
by sorry

end sacred_words_count_l944_94458


namespace square_sum_equals_150_l944_94465

theorem square_sum_equals_150 (u v : ℝ) 
  (h1 : u * (u + v) = 50) 
  (h2 : v * (u + v) = 100) : 
  (u + v)^2 = 150 := by
sorry

end square_sum_equals_150_l944_94465


namespace range_of_a_l944_94473

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x

-- Define the proposition P
def P (a : ℝ) : Prop := ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂

-- Define the function inside the logarithm
def g (a : ℝ) (x : ℝ) : ℝ := a*x^2 - x + a

-- Define the proposition Q
def Q (a : ℝ) : Prop := ∀ x, g a x > 0

-- Main theorem
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ 1/2 ∨ a > 1 :=
sorry

end range_of_a_l944_94473


namespace julia_spent_114_l944_94434

/-- The total amount Julia spent on food for her animals -/
def total_spent (weekly_total : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) (rabbit_food_cost : ℕ) : ℕ :=
  let parrot_food_cost := weekly_total - rabbit_food_cost
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost

/-- Proof that Julia spent $114 on food for her animals -/
theorem julia_spent_114 :
  total_spent 30 5 3 12 = 114 := by
  sorry

end julia_spent_114_l944_94434


namespace largest_prime_divisor_101010101_base5_l944_94449

theorem largest_prime_divisor_101010101_base5 :
  let n : ℕ := 5^8 + 5^6 + 5^4 + 5^2 + 1
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ (∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) ∧ p = 601 :=
by sorry

end largest_prime_divisor_101010101_base5_l944_94449


namespace circles_intersect_l944_94433

theorem circles_intersect : ∃ (x y : ℝ), 
  ((x + 1)^2 + (y + 2)^2 = 4) ∧ ((x - 1)^2 + (y + 1)^2 = 9) := by
  sorry

end circles_intersect_l944_94433


namespace contrapositive_equivalence_l944_94482

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = b) → ¬(a^2 - b^2 = 0)) ↔ (a^2 - b^2 = 0 → a = b) := by
  sorry

end contrapositive_equivalence_l944_94482


namespace range_of_t_l944_94406

/-- A function f(x) = x^2 - 2tx + 1 that is decreasing on (-∞, 1] -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t (t : ℝ) : 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → f t x > f t y) →  -- f is decreasing on (-∞, 1]
  (∀ x₁ ∈ Set.Icc 0 (t+1), ∀ x₂ ∈ Set.Icc 0 (t+1), |f t x₁ - f t x₂| ≤ 2) →  -- |f(x₁) - f(x₂)| ≤ 2
  t ∈ Set.Icc 1 (Real.sqrt 2) :=  -- t ∈ [1, √2]
sorry

end range_of_t_l944_94406


namespace sqrt_16_equals_4_l944_94490

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l944_94490


namespace correct_minutes_for_ninth_day_l944_94402

/-- The number of minutes Julia needs to read on the 9th day to achieve the target average -/
def minutes_to_read_on_ninth_day (days_reading_80_min : ℕ) (days_reading_100_min : ℕ) (target_average : ℕ) (total_days : ℕ) : ℕ :=
  let total_minutes_read := days_reading_80_min * 80 + days_reading_100_min * 100
  let target_total_minutes := total_days * target_average
  target_total_minutes - total_minutes_read

/-- Theorem stating the correct number of minutes Julia needs to read on the 9th day -/
theorem correct_minutes_for_ninth_day :
  minutes_to_read_on_ninth_day 6 2 95 9 = 175 := by
  sorry

end correct_minutes_for_ninth_day_l944_94402


namespace tank_weight_l944_94469

/-- Given a tank with the following properties:
  * When four-fifths full, it weighs p kilograms
  * When two-thirds full, it weighs q kilograms
  * The empty tank and other contents weigh r kilograms
  Prove that the total weight of the tank when completely full is (5/2)p + (3/2)q -/
theorem tank_weight (p q r : ℝ) : 
  (∃ (x y : ℝ), x + (4/5) * y = p ∧ x + (2/3) * y = q ∧ x = r) →
  (∃ (z : ℝ), z = (5/2) * p + (3/2) * q ∧ 
    (∀ (x y : ℝ), x + (4/5) * y = p ∧ x + (2/3) * y = q → x + y = z)) :=
by sorry

end tank_weight_l944_94469


namespace chopped_cube_height_chopped_cube_height_value_l944_94414

/-- The height of a 2x2x2 cube with a corner chopped off -/
theorem chopped_cube_height : ℝ :=
  let cube_side : ℝ := 2
  let cut_face_side : ℝ := 2 * Real.sqrt 2
  let cut_face_area : ℝ := Real.sqrt 3 / 4 * cut_face_side^2
  let removed_pyramid_height : ℝ := Real.sqrt 3 / 9
  cube_side - removed_pyramid_height

/-- Theorem stating that the height of the chopped cube is (17√3)/9 -/
theorem chopped_cube_height_value : chopped_cube_height = (17 * Real.sqrt 3) / 9 := by
  sorry


end chopped_cube_height_chopped_cube_height_value_l944_94414


namespace ceiling_neg_seven_fourths_squared_l944_94442

theorem ceiling_neg_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end ceiling_neg_seven_fourths_squared_l944_94442


namespace corn_height_after_three_weeks_l944_94466

/-- The height of corn plants after three weeks of growth -/
def corn_height (initial_height week1_growth : ℕ) : ℕ :=
  let week2_growth := 2 * week1_growth
  let week3_growth := 4 * week2_growth
  initial_height + week1_growth + week2_growth + week3_growth

/-- Theorem stating that the corn height after three weeks is 22 inches -/
theorem corn_height_after_three_weeks :
  corn_height 0 2 = 22 := by
  sorry

end corn_height_after_three_weeks_l944_94466


namespace reflection_line_sum_l944_94470

/-- Given a line y = mx + b, if the reflection of point (2,2) across this line is (10,6), then m + b = 14 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The point (x, y) is on the line y = mx + b
    y = m * x + b ∧ 
    -- The point (x, y) is equidistant from (2,2) and (10,6)
    (x - 2)^2 + (y - 2)^2 = (x - 10)^2 + (y - 6)^2 ∧
    -- The line connecting (2,2) and (10,6) is perpendicular to y = mx + b
    (6 - 2) = -1 / m * (10 - 2)) →
  m + b = 14 := by
sorry


end reflection_line_sum_l944_94470


namespace smallest_sum_four_consecutive_composites_l944_94431

/-- A natural number is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Four consecutive natural numbers are all composite. -/
def FourConsecutiveComposites (n : ℕ) : Prop :=
  IsComposite n ∧ IsComposite (n + 1) ∧ IsComposite (n + 2) ∧ IsComposite (n + 3)

/-- The sum of four consecutive natural numbers starting from n. -/
def SumFourConsecutive (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3)

theorem smallest_sum_four_consecutive_composites :
  (∃ n : ℕ, FourConsecutiveComposites n) ∧
  (∀ m : ℕ, FourConsecutiveComposites m → SumFourConsecutive m ≥ 102) ∧
  (∃ k : ℕ, FourConsecutiveComposites k ∧ SumFourConsecutive k = 102) :=
by sorry

end smallest_sum_four_consecutive_composites_l944_94431


namespace definite_integral_sin_plus_one_l944_94447

theorem definite_integral_sin_plus_one : ∫ x in (-1)..(1), (Real.sin x + 1) = 2 := by sorry

end definite_integral_sin_plus_one_l944_94447


namespace junk_mail_distribution_l944_94467

theorem junk_mail_distribution (total_mail : ℕ) (total_houses : ℕ) 
  (h1 : total_mail = 48) (h2 : total_houses = 8) :
  total_mail / total_houses = 6 := by
  sorry

end junk_mail_distribution_l944_94467


namespace cos_double_angle_for_point_l944_94454

/-- Given a point P(-1, 2) on the terminal side of angle α, prove that cos(2α) = -3/5 -/
theorem cos_double_angle_for_point (α : ℝ) : 
  let P : ℝ × ℝ := (-1, 2)
  (P.1 = -1 ∧ P.2 = 2) → -- P has coordinates (-1, 2)
  (P.1 = -1 * Real.sqrt 5 * Real.cos α ∧ P.2 = 2 * Real.sqrt 5 * Real.sin α) → -- P is on the terminal side of angle α
  Real.cos (2 * α) = -3/5 := by
sorry

end cos_double_angle_for_point_l944_94454


namespace cubic_root_sum_l944_94495

theorem cubic_root_sum (p q s : ℝ) : 
  10 * p^3 - 25 * p^2 + 8 * p - 1 = 0 →
  10 * q^3 - 25 * q^2 + 8 * q - 1 = 0 →
  10 * s^3 - 25 * s^2 + 8 * s - 1 = 0 →
  0 < p → p < 1 →
  0 < q → q < 1 →
  0 < s → s < 1 →
  1 / (1 - p) + 1 / (1 - q) + 1 / (1 - s) = 0.5 :=
by sorry

end cubic_root_sum_l944_94495


namespace zinc_copper_mixture_weight_l944_94417

/-- Proves that given a mixture of zinc and copper in the ratio 9:11, 
    where 27 kg of zinc is used, the total weight of the mixture is 60 kg. -/
theorem zinc_copper_mixture_weight (zinc_weight : ℝ) (copper_weight : ℝ) :
  zinc_weight = 27 →
  zinc_weight / copper_weight = 9 / 11 →
  zinc_weight + copper_weight = 60 := by
sorry

end zinc_copper_mixture_weight_l944_94417


namespace cash_percentage_proof_l944_94410

/-- Calculates the percentage of total amount spent as cash given the total amount and amounts spent on raw materials and machinery. -/
def percentage_spent_as_cash (total_amount raw_materials machinery : ℚ) : ℚ :=
  ((total_amount - (raw_materials + machinery)) / total_amount) * 100

/-- Proves that given a total amount of $250, with $100 spent on raw materials and $125 spent on machinery, the percentage of the total amount spent as cash is 10%. -/
theorem cash_percentage_proof :
  percentage_spent_as_cash 250 100 125 = 10 := by
  sorry

end cash_percentage_proof_l944_94410


namespace starting_lineup_combinations_l944_94471

def team_size : ℕ := 15
def starting_lineup_size : ℕ := 5
def preselected_players : ℕ := 3

theorem starting_lineup_combinations :
  Nat.choose (team_size - preselected_players) (starting_lineup_size - preselected_players) = 66 :=
by sorry

end starting_lineup_combinations_l944_94471


namespace area_ratio_equilateral_triangle_extension_l944_94455

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Extends a side of a triangle by a given factor -/
def extendSide (t : Triangle) (vertex : ℝ × ℝ) (factor : ℝ) : ℝ × ℝ := sorry

theorem area_ratio_equilateral_triangle_extension
  (ABC : Triangle)
  (h_equilateral : ABC.A.1^2 + ABC.A.2^2 = ABC.B.1^2 + ABC.B.2^2 ∧
                   ABC.B.1^2 + ABC.B.2^2 = ABC.C.1^2 + ABC.C.2^2 ∧
                   ABC.C.1^2 + ABC.C.2^2 = ABC.A.1^2 + ABC.A.2^2)
  (B' : ℝ × ℝ)
  (C' : ℝ × ℝ)
  (A' : ℝ × ℝ)
  (h_BB' : B' = extendSide ABC ABC.B 2)
  (h_CC' : C' = extendSide ABC ABC.C 3)
  (h_AA' : A' = extendSide ABC ABC.A 4)
  : area (Triangle.mk A' B' C') / area ABC = 42 := by sorry

end area_ratio_equilateral_triangle_extension_l944_94455


namespace garage_roof_leak_l944_94477

/-- The amount of water leaked from three holes in a garage roof over a 2-hour period -/
def water_leaked (largest_hole_rate : ℚ) (time_hours : ℚ) : ℚ :=
  let medium_hole_rate := largest_hole_rate / 2
  let smallest_hole_rate := medium_hole_rate / 3
  let time_minutes := time_hours * 60
  (largest_hole_rate + medium_hole_rate + smallest_hole_rate) * time_minutes

/-- Theorem stating the total amount of water leaked from three holes in a garage roof over a 2-hour period -/
theorem garage_roof_leak : water_leaked 3 2 = 600 := by
  sorry

end garage_roof_leak_l944_94477


namespace chord_intersection_sum_l944_94400

-- Define the sphere and point S
variable (sphere : Type) (S : sphere)

-- Define the chords
variable (A A' B B' C C' : sphere)

-- Define the lengths
variable (AS BS CS : ℝ)

-- Define the volume ratio
variable (volume_ratio : ℝ)

-- State the theorem
theorem chord_intersection_sum (h1 : AS = 6) (h2 : BS = 3) (h3 : CS = 2)
  (h4 : volume_ratio = 2/9) :
  ∃ (SA' SB' SC' : ℝ), SA' + SB' + SC' = 18 := by
  sorry

end chord_intersection_sum_l944_94400


namespace angle_representation_l944_94446

theorem angle_representation (given_angle : ℝ) : 
  given_angle = -1485 → 
  ∃ (α k : ℝ), 
    given_angle = α + k * 360 ∧ 
    0 ≤ α ∧ α < 360 ∧ 
    k = -5 ∧
    α = 315 := by
  sorry

end angle_representation_l944_94446


namespace product_mod_eight_l944_94438

theorem product_mod_eight : (55 * 57) % 8 = 7 := by
  sorry

end product_mod_eight_l944_94438


namespace inverse_proportion_problem_inverse_proportion_problem_2_l944_94419

/-- Two quantities are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 4 = 36 →
  x 9 = 16 :=
by sorry

theorem inverse_proportion_problem_2 :
  ∀ a b : ℝ → ℝ,
  InverselyProportional a b →
  a 5 = 50 →
  a 10 = 25 :=
by sorry

end inverse_proportion_problem_inverse_proportion_problem_2_l944_94419


namespace bookcase_sum_l944_94487

theorem bookcase_sum (a₁ : ℕ) (d : ℤ) (n : ℕ) (aₙ : ℕ) : 
  a₁ = 32 → 
  d = -3 → 
  aₙ > 0 → 
  aₙ = a₁ + (n - 1) * d → 
  n * (a₁ + aₙ) = 374 → 
  (n : ℤ) * (2 * a₁ + (n - 1) * d) = 374 :=
by sorry

end bookcase_sum_l944_94487


namespace repeating_decimal_36_equals_4_11_l944_94457

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (a * 10 + b) / 99

/-- The theorem states that 0.¯36 is equal to 4/11 -/
theorem repeating_decimal_36_equals_4_11 :
  RepeatingDecimal 3 6 = 4 / 11 := by
  sorry

end repeating_decimal_36_equals_4_11_l944_94457


namespace arithmetic_sequence_sum_l944_94478

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 2 = 2 →
  a 3 + a 4 = 10 →
  a 5 + a 6 = 18 := by
sorry

end arithmetic_sequence_sum_l944_94478


namespace total_concert_attendance_l944_94460

def first_concert_attendance : ℕ := 65899
def second_concert_difference : ℕ := 119

theorem total_concert_attendance : 
  let second_concert_attendance := first_concert_attendance + second_concert_difference
  let third_concert_attendance := 2 * second_concert_attendance
  first_concert_attendance + second_concert_attendance + third_concert_attendance = 263953 := by
sorry

end total_concert_attendance_l944_94460


namespace jacks_total_money_l944_94408

/-- Calculates the total amount of money in dollars given an amount in dollars and euros, with a fixed exchange rate. -/
def total_money_in_dollars (dollars : ℕ) (euros : ℕ) (exchange_rate : ℕ) : ℕ :=
  dollars + euros * exchange_rate

/-- Theorem stating that Jack's total money in dollars is 117 given the problem conditions. -/
theorem jacks_total_money :
  total_money_in_dollars 45 36 2 = 117 := by
  sorry

end jacks_total_money_l944_94408


namespace no_integer_solution_l944_94401

theorem no_integer_solution : ¬ ∃ (x y z : ℤ), (x - y)^3 + (y - z)^3 + (z - x)^3 = 2021 := by
  sorry

end no_integer_solution_l944_94401


namespace solve_equation_l944_94424

theorem solve_equation (m : ℝ) : m + (m + 2) + (m + 4) = 21 → m = 5 := by
  sorry

end solve_equation_l944_94424


namespace largest_five_digit_congruent_18_mod_25_l944_94452

theorem largest_five_digit_congruent_18_mod_25 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≤ 99999 ∧ n % 25 = 18 → 
    n ≤ 99993 :=
by sorry

end largest_five_digit_congruent_18_mod_25_l944_94452


namespace eight_elevenths_rounded_l944_94498

-- Define a function to round a rational number to n decimal places
def round_to_decimal_places (q : ℚ) (n : ℕ) : ℚ :=
  (↑(⌊q * 10^n + 1/2⌋)) / 10^n

-- State the theorem
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 2 = 73/100 := by
  sorry

end eight_elevenths_rounded_l944_94498


namespace exponent_simplification_l944_94415

theorem exponent_simplification :
  3^6 * 6^6 * 3^12 * 6^12 = 18^18 := by
  sorry

end exponent_simplification_l944_94415


namespace visible_red_bus_length_l944_94485

/-- Proves that the visible length of a red bus from a yellow bus is 6 feet, given specific length relationships between red, orange, and yellow buses. -/
theorem visible_red_bus_length 
  (red_bus_length : ℝ)
  (orange_car_length : ℝ)
  (yellow_bus_length : ℝ)
  (h1 : red_bus_length = 4 * orange_car_length)
  (h2 : yellow_bus_length = 3.5 * orange_car_length)
  (h3 : red_bus_length = 48) :
  red_bus_length - yellow_bus_length = 6 := by
  sorry

#check visible_red_bus_length

end visible_red_bus_length_l944_94485


namespace orange_cost_solution_l944_94420

/-- Calculates the cost of an orange given the initial quantities, apple cost, and final earnings -/
def orange_cost (initial_apples initial_oranges : ℕ) (apple_cost : ℚ) 
  (final_apples final_oranges : ℕ) (total_earnings : ℚ) : ℚ :=
  let apples_sold := initial_apples - final_apples
  let oranges_sold := initial_oranges - final_oranges
  let apple_earnings := apples_sold * apple_cost
  let orange_earnings := total_earnings - apple_earnings
  orange_earnings / oranges_sold

theorem orange_cost_solution :
  orange_cost 50 40 (4/5) 10 6 49 = 1/2 := by
  sorry

end orange_cost_solution_l944_94420


namespace least_addend_proof_l944_94425

/-- The least non-negative integer that, when added to 11002, results in a number divisible by 11 -/
def least_addend : ℕ := 9

/-- The original number we start with -/
def original_number : ℕ := 11002

theorem least_addend_proof :
  (∀ k : ℕ, k < least_addend → ¬((original_number + k) % 11 = 0)) ∧
  ((original_number + least_addend) % 11 = 0) :=
sorry

end least_addend_proof_l944_94425


namespace johns_allowance_l944_94499

/-- John's weekly allowance problem -/
theorem johns_allowance :
  ∀ (A : ℚ),
  (A > 0) →
  (3/5 * A + 1/3 * (A - 3/5 * A) + 88/100 = A) →
  A = 33/10 :=
by
  sorry

end johns_allowance_l944_94499


namespace sqrt_of_square_neg_l944_94484

theorem sqrt_of_square_neg (a : ℝ) (h : a < 0) : Real.sqrt (a^2) = -a := by
  sorry

end sqrt_of_square_neg_l944_94484


namespace house_amenities_l944_94464

theorem house_amenities (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ) :
  total = 65 → garage = 50 → pool = 40 → neither = 10 →
  ∃ both : ℕ, both = 35 ∧ garage + pool - both = total - neither :=
by sorry

end house_amenities_l944_94464


namespace textbook_cost_proof_l944_94429

/-- Represents the cost of textbooks and proves the cost of each sale textbook --/
theorem textbook_cost_proof (sale_books : ℕ) (online_books : ℕ) (bookstore_books : ℕ)
  (online_total : ℚ) (total_spent : ℚ) :
  sale_books = 5 →
  online_books = 2 →
  bookstore_books = 3 →
  online_total = 40 →
  total_spent = 210 →
  (sale_books * (total_spent - online_total - 3 * online_total) / sale_books : ℚ) = 10 := by
  sorry

#check textbook_cost_proof

end textbook_cost_proof_l944_94429


namespace negation_equivalence_l944_94488

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 2 < 0) ↔ (∀ x : ℝ, x^2 + x - 2 ≥ 0) := by
  sorry

end negation_equivalence_l944_94488


namespace mikes_games_l944_94468

/-- Given Mike's earnings, expenses, and game cost, prove the number of games he can buy -/
theorem mikes_games (earnings : ℕ) (blade_cost : ℕ) (game_cost : ℕ) 
  (h1 : earnings = 101)
  (h2 : blade_cost = 47)
  (h3 : game_cost = 6) :
  (earnings - blade_cost) / game_cost = 9 := by
  sorry

end mikes_games_l944_94468


namespace decreasing_cubic_condition_l944_94481

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- State the theorem
theorem decreasing_cubic_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ 0 := by
  sorry

end decreasing_cubic_condition_l944_94481


namespace candy_sharing_l944_94409

theorem candy_sharing (hugh tommy melany : ℕ) (h1 : hugh = 8) (h2 : tommy = 6) (h3 : melany = 7) :
  (hugh + tommy + melany) / 3 = 7 := by
  sorry

end candy_sharing_l944_94409


namespace prob_at_least_one_unqualified_is_correct_l944_94462

/-- The total number of products -/
def total_products : ℕ := 6

/-- The number of qualified products -/
def qualified_products : ℕ := 4

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products randomly selected -/
def selected_products : ℕ := 2

/-- The probability of selecting at least one unqualified product -/
def prob_at_least_one_unqualified : ℚ := 3/5

theorem prob_at_least_one_unqualified_is_correct :
  (1 : ℚ) - (Nat.choose qualified_products selected_products : ℚ) / (Nat.choose total_products selected_products : ℚ) = prob_at_least_one_unqualified :=
sorry

end prob_at_least_one_unqualified_is_correct_l944_94462


namespace tangent_slope_at_one_l944_94445

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop := 2 * x^2 + 4 * x + 6 * y = 24

/-- The slope of the tangent line at a given point -/
noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -(2/3 * x + 2/3)

/-- Theorem: The slope of the tangent line to the curve at x = 1 is -4/3 -/
theorem tangent_slope_at_one : 
  tangent_slope 1 = -4/3 := by sorry

end tangent_slope_at_one_l944_94445


namespace necessary_not_sufficient_l944_94411

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end necessary_not_sufficient_l944_94411


namespace complex_sum_to_polar_l944_94426

theorem complex_sum_to_polar : 
  5 * Complex.exp (Complex.I * (3 * Real.pi / 8)) + 
  5 * Complex.exp (Complex.I * (17 * Real.pi / 16)) = 
  10 * Real.cos (5 * Real.pi / 32) * Complex.exp (Complex.I * (23 * Real.pi / 32)) := by
  sorry

end complex_sum_to_polar_l944_94426


namespace fraction_power_four_l944_94476

theorem fraction_power_four : (5 / 3 : ℚ) ^ 4 = 625 / 81 := by
  sorry

end fraction_power_four_l944_94476


namespace probability_at_most_one_defective_is_five_sevenths_l944_94407

def total_products : ℕ := 8
def defective_products : ℕ := 3
def drawn_products : ℕ := 3

def probability_at_most_one_defective : ℚ :=
  (Nat.choose (total_products - defective_products) drawn_products +
   Nat.choose (total_products - defective_products) (drawn_products - 1) * 
   Nat.choose defective_products 1) /
  Nat.choose total_products drawn_products

theorem probability_at_most_one_defective_is_five_sevenths :
  probability_at_most_one_defective = 5 / 7 := by
  sorry

end probability_at_most_one_defective_is_five_sevenths_l944_94407


namespace sqrt_inequality_l944_94441

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end sqrt_inequality_l944_94441


namespace car_speed_calculation_l944_94459

/-- Proves that a car's speed is 52 miles per hour given specific conditions -/
theorem car_speed_calculation (fuel_efficiency : ℝ) (fuel_consumed : ℝ) (time : ℝ)
  (gallon_to_liter : ℝ) (km_to_mile : ℝ) :
  fuel_efficiency = 32 →
  fuel_consumed = 3.9 →
  time = 5.7 →
  gallon_to_liter = 3.8 →
  km_to_mile = 1.6 →
  (fuel_consumed * gallon_to_liter * fuel_efficiency) / (time * km_to_mile) = 52 := by
sorry

end car_speed_calculation_l944_94459


namespace white_pairs_coincide_l944_94403

/-- Represents the number of triangles of each color in one half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

/-- Theorem stating that given the conditions, 7 white pairs must coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 6 ∧ 
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 2 →
  pairs.white_white = 7 := by
  sorry

end white_pairs_coincide_l944_94403


namespace june_video_hours_l944_94492

/-- Calculates the total video hours uploaded in a month with varying upload rates -/
def total_video_hours (days : ℕ) (initial_rate : ℕ) (doubled_rate : ℕ) : ℕ :=
  let half_days := days / 2
  (half_days * initial_rate) + (half_days * doubled_rate)

/-- Proves that the total video hours uploaded in June is 450 -/
theorem june_video_hours :
  total_video_hours 30 10 20 = 450 := by
  sorry

end june_video_hours_l944_94492


namespace problem_polygon_area_l944_94412

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a polygon on a 2D grid --/
structure GridPolygon where
  vertices : List GridPoint

/-- Calculates the area of a polygon on a grid --/
def calculateGridPolygonArea (polygon : GridPolygon) : ℕ :=
  sorry

/-- The specific polygon from the problem --/
def problemPolygon : GridPolygon :=
  { vertices := [
    {x := 0, y := 0}, {x := 1, y := 0}, {x := 1, y := 1}, {x := 2, y := 1},
    {x := 3, y := 0}, {x := 3, y := 1}, {x := 4, y := 0}, {x := 4, y := 1},
    {x := 4, y := 3}, {x := 3, y := 3}, {x := 4, y := 4}, {x := 3, y := 4},
    {x := 2, y := 4}, {x := 0, y := 4}, {x := 0, y := 2}, {x := 0, y := 0}
  ] }

theorem problem_polygon_area : calculateGridPolygonArea problemPolygon = 14 := by
  sorry

end problem_polygon_area_l944_94412


namespace unique_solution_exponential_equation_l944_94437

/-- The equation 3^(3x^3 - 9x^2 + 15x - 5) = 1 has exactly one real solution. -/
theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (3 : ℝ) ^ (3 * x^3 - 9 * x^2 + 15 * x - 5) = 1 := by
  sorry

end unique_solution_exponential_equation_l944_94437


namespace root_relationship_l944_94416

-- Define the first polynomial equation
def f (x : ℝ) : ℝ := x^3 - 6*x^2 - 39*x - 10

-- Define the second polynomial equation
def g (x : ℝ) : ℝ := x^3 + x^2 - 20*x - 50

-- State the theorem
theorem root_relationship :
  (∃ (x y : ℝ), f x = 0 ∧ g y = 0 ∧ x = 2*y) →
  f 10 = 0 ∧ g 5 = 0 := by
  sorry

end root_relationship_l944_94416


namespace largest_x_and_multiples_l944_94494

theorem largest_x_and_multiples :
  let x := Int.floor ((23 - 7) / -3)
  x = -6 ∧
  (x^2 * 1 = 36 ∧ x^2 * 2 = 72 ∧ x^2 * 3 = 108) :=
by sorry

end largest_x_and_multiples_l944_94494


namespace hotel_cost_calculation_l944_94472

theorem hotel_cost_calculation 
  (cost_per_night_per_person : ℕ) 
  (number_of_people : ℕ) 
  (number_of_nights : ℕ) 
  (h1 : cost_per_night_per_person = 40)
  (h2 : number_of_people = 3)
  (h3 : number_of_nights = 3) :
  cost_per_night_per_person * number_of_people * number_of_nights = 360 :=
by sorry

end hotel_cost_calculation_l944_94472


namespace counsel_probability_l944_94404

def CANOE : Finset Char := {'C', 'A', 'N', 'O', 'E'}
def SHRUB : Finset Char := {'S', 'H', 'R', 'U', 'B'}
def FLOW : Finset Char := {'F', 'L', 'O', 'W'}
def COUNSEL : Finset Char := {'C', 'O', 'U', 'N', 'S', 'E', 'L'}

def prob_CANOE : ℚ := 1 / (CANOE.card.choose 2)
def prob_SHRUB : ℚ := 3 / (SHRUB.card.choose 3)
def prob_FLOW : ℚ := 1 / (FLOW.card.choose 4)

theorem counsel_probability :
  prob_CANOE * prob_SHRUB * prob_FLOW = 3 / 100 := by
  sorry

end counsel_probability_l944_94404


namespace area_between_concentric_circles_l944_94430

/-- The area between two concentric circles -/
theorem area_between_concentric_circles 
  (R : ℝ) -- Radius of the outer circle
  (c : ℝ) -- Length of the chord
  (h1 : R = 12) -- Given radius of outer circle
  (h2 : c = 20) -- Given length of chord
  (h3 : c ≤ 2 * R) -- Chord cannot be longer than diameter
  : ∃ (r : ℝ), 0 < r ∧ r < R ∧ π * (R^2 - r^2) = 100 * π :=
sorry

end area_between_concentric_circles_l944_94430


namespace incorrect_operation_l944_94422

theorem incorrect_operation (x y : ℝ) : -2*x*(x - y) ≠ -2*x^2 - 2*x*y := by
  sorry

end incorrect_operation_l944_94422


namespace sphere_area_ratio_l944_94486

theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : A₁ > 0) :
  let A₂ := A₁ * (r₂ / r₁)^2
  r₁ = 4 ∧ r₂ = 6 ∧ A₁ = 37 → A₂ = 83.25 := by
  sorry

end sphere_area_ratio_l944_94486


namespace convex_pentagon_probability_l944_94480

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of chords possible with n points -/
def total_chords : ℕ := n.choose 2

/-- The total number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def favorable_outcomes : ℕ := n.choose k

/-- The probability of k randomly selected chords from n points on a circle forming a convex polygon -/
def probability : ℚ := favorable_outcomes / total_selections

theorem convex_pentagon_probability :
  n = 7 ∧ k = 5 → probability = 1 / 969 := by
  sorry

end convex_pentagon_probability_l944_94480


namespace log_inequalities_l944_94461

theorem log_inequalities (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x < x - 1) ∧ (Real.log x > (x - 1) / x) := by
  sorry

end log_inequalities_l944_94461


namespace combined_yellow_ratio_approx_31_percent_l944_94451

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the total number of yellow jelly beans in a bag -/
def yellow_count (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellow_ratio

/-- Theorem: The ratio of yellow jelly beans to all beans when three bags are combined -/
theorem combined_yellow_ratio_approx_31_percent 
  (bag1 bag2 bag3 : JellyBeanBag)
  (h1 : bag1 = ⟨26, 1/2⟩)
  (h2 : bag2 = ⟨28, 1/4⟩)
  (h3 : bag3 = ⟨30, 1/5⟩) :
  let total_yellow := yellow_count bag1 + yellow_count bag2 + yellow_count bag3
  let total_beans := bag1.total + bag2.total + bag3.total
  abs ((total_yellow / total_beans) - 31/100) < 1/100 := by
  sorry

end combined_yellow_ratio_approx_31_percent_l944_94451


namespace quadratic_two_distinct_roots_negative_four_satisfies_l944_94475

theorem quadratic_two_distinct_roots (c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + c = 0 ∧ y^2 - 4*y + c = 0) → c < 4 :=
by sorry

theorem negative_four_satisfies (c : ℝ) : 
  c = -4 → (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + c = 0 ∧ y^2 - 4*y + c = 0) :=
by sorry

end quadratic_two_distinct_roots_negative_four_satisfies_l944_94475


namespace x_squared_greater_than_x_root_l944_94413

theorem x_squared_greater_than_x_root (x : ℝ) : x ^ 2 > x ^ (1 / 2) ↔ x > 1 := by
  sorry

end x_squared_greater_than_x_root_l944_94413


namespace ratio_of_numbers_l944_94418

def smaller_number : ℝ := 20

def larger_number : ℝ := 6 * smaller_number

theorem ratio_of_numbers : larger_number / smaller_number = 6 := by
  sorry

end ratio_of_numbers_l944_94418


namespace average_of_four_numbers_l944_94448

theorem average_of_four_numbers (n : ℝ) :
  (3 + 16 + 33 + (n + 1)) / 4 = 20 → n = 27 := by
  sorry

end average_of_four_numbers_l944_94448


namespace sequence_general_term_l944_94432

/-- Given a sequence {a_n} with sum of first n terms S_n = (3(3^n + 1)) / 2,
    prove that a_n = 3^n for n ≥ 2 -/
theorem sequence_general_term (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_sum : ∀ k, S k = (3 * (3^k + 1)) / 2) 
    (h_def : ∀ k, k ≥ 2 → a k = S k - S (k-1)) :
  ∀ m, m ≥ 2 → a m = 3^m :=
by sorry

end sequence_general_term_l944_94432


namespace probability_all_genuine_given_equal_weight_l944_94405

/-- Represents the total number of coins -/
def total_coins : ℕ := 12

/-- Represents the number of genuine coins -/
def genuine_coins : ℕ := 9

/-- Represents the number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- Event A: All 4 selected coins are genuine -/
def event_A : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) :=
  sorry

/-- Event B: The combined weight of the first pair equals the combined weight of the second pair -/
def event_B : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) :=
  sorry

/-- The probability measure on the sample space -/
def P : Set (Fin total_coins × Fin total_coins × Fin total_coins × Fin total_coins) → ℚ :=
  sorry

/-- Theorem stating the conditional probability of A given B -/
theorem probability_all_genuine_given_equal_weight :
    P (event_A ∩ event_B) / P event_B = 84 / 113 := by
  sorry

end probability_all_genuine_given_equal_weight_l944_94405


namespace inequality_solution_l944_94436

theorem inequality_solution (x : ℝ) : x - 1 / x > 0 ↔ (-1 < x ∧ x < 0) ∨ x > 1 := by
  sorry

end inequality_solution_l944_94436


namespace loraine_wax_usage_l944_94428

/-- The number of wax sticks used for all animals -/
def total_wax_sticks (large_animal_wax small_animal_wax : ℕ) 
  (small_animal_ratio : ℕ) (small_animal_total_wax : ℕ) : ℕ :=
  small_animal_total_wax + 
  (small_animal_total_wax / small_animal_wax) / small_animal_ratio * large_animal_wax

/-- Proof that Loraine used 20 sticks of wax for all animals -/
theorem loraine_wax_usage : 
  total_wax_sticks 4 2 3 12 = 20 := by
  sorry

end loraine_wax_usage_l944_94428


namespace retailer_profit_percentage_l944_94483

theorem retailer_profit_percentage 
  (cost : ℝ) 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) : 
  cost = 80 → 
  discounted_price = 130 → 
  discount_rate = 0.2 → 
  ((discounted_price / (1 - discount_rate) - cost) / cost) * 100 = 103.125 := by
  sorry

end retailer_profit_percentage_l944_94483


namespace tan_neg_five_pi_fourth_l944_94427

theorem tan_neg_five_pi_fourth : Real.tan (-5 * Real.pi / 4) = -1 := by
  sorry

end tan_neg_five_pi_fourth_l944_94427


namespace bottom_layer_lights_for_specific_tower_l944_94435

/-- Represents a tower with a geometric progression of lights -/
structure LightTower where
  layers : ℕ
  total_lights : ℕ
  ratio : ℕ

/-- Calculates the number of lights on the bottom layer of a tower -/
def bottom_layer_lights (tower : LightTower) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- The theorem stating the number of lights on the bottom layer of the specific tower -/
theorem bottom_layer_lights_for_specific_tower :
  let tower : LightTower := ⟨5, 242, 3⟩
  bottom_layer_lights tower = 162 := by
  sorry

end bottom_layer_lights_for_specific_tower_l944_94435


namespace line_circle_intersection_and_dot_product_l944_94463

-- Define the line l passing through A(0, 1) with slope k
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

-- Define the circle C: (x-2)^2+(y-3)^2=1
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define the point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem line_circle_intersection_and_dot_product
  (k : ℝ) (M N : ℝ × ℝ) :
  (M ∈ line_l k ∧ M ∈ circle_C ∧ N ∈ line_l k ∧ N ∈ circle_C) →
  ((4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3) ∧
  (dot_product (M.1 - point_A.1, M.2 - point_A.2) (N.1 - point_A.1, N.2 - point_A.2) = 7) ∧
  (dot_product (M.1 - origin.1, M.2 - origin.2) (N.1 - origin.1, N.2 - origin.2) = 12 →
    k = 1 ∧ line_l k = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}) := by
  sorry

end line_circle_intersection_and_dot_product_l944_94463


namespace pyramid_rows_equal_ten_l944_94489

/-- The number of spheres in a square-based pyramid with n rows -/
def square_pyramid (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of spheres in a triangle-based pyramid with n rows -/
def triangle_pyramid (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The total number of spheres -/
def total_spheres : ℕ := 605

theorem pyramid_rows_equal_ten :
  ∃ (n : ℕ), n > 0 ∧ square_pyramid n + triangle_pyramid n = total_spheres := by
  sorry

end pyramid_rows_equal_ten_l944_94489


namespace sum_of_solutions_quadratic_sum_of_solutions_specific_l944_94493

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  equation 0 = 0 → sum_of_roots = (-b) / a :=
by
  sorry

-- Specific instance for the given problem
theorem sum_of_solutions_specific :
  let equation := fun x => -48 * x^2 + 96 * x + 180
  let sum_of_roots := 2
  (∀ x, equation x = 0 → x = sum_of_roots ∨ x = 0) :=
by
  sorry

end sum_of_solutions_quadratic_sum_of_solutions_specific_l944_94493


namespace subtraction_problem_l944_94474

theorem subtraction_problem (x N V : ℝ) : 
  x = 10 → 3 * x = (N - x) + V → V = 0 → N = 40 := by
sorry

end subtraction_problem_l944_94474
