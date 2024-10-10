import Mathlib

namespace factored_equation_difference_l2183_218371

theorem factored_equation_difference (p q : ℝ) : 
  (∃ (x : ℝ), x^2 - 6*x + q = 0 ∧ (x - p)^2 = 7) → p - q = 1 := by
  sorry

end factored_equation_difference_l2183_218371


namespace routes_between_plains_cities_l2183_218348

theorem routes_between_plains_cities 
  (total_cities : ℕ) 
  (mountainous_cities : ℕ) 
  (plains_cities : ℕ) 
  (total_routes : ℕ) 
  (mountainous_routes : ℕ) 
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : mountainous_cities + plains_cities = total_cities)
  (h5 : total_routes = 150)
  (h6 : mountainous_routes = 21) :
  total_routes - mountainous_routes - (mountainous_cities * 3 - mountainous_routes * 2) / 2 = 81 := by
  sorry

end routes_between_plains_cities_l2183_218348


namespace no_valid_combination_l2183_218336

def nickel : ℕ := 5
def dime : ℕ := 10
def half_dollar : ℕ := 50

def is_valid_combination (coins : List ℕ) : Prop :=
  coins.all (λ c => c = nickel ∨ c = dime ∨ c = half_dollar) ∧
  coins.length = 6 ∧
  coins.sum = 90

theorem no_valid_combination : ¬ ∃ (coins : List ℕ), is_valid_combination coins := by
  sorry

end no_valid_combination_l2183_218336


namespace min_value_of_expression_l2183_218365

theorem min_value_of_expression (a b : ℤ) (h : a > b) :
  (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' > b' ∧ (((a'^2 + b'^2) / (a'^2 - b'^2)) + ((a'^2 - b'^2) / (a'^2 + b'^2)) : ℚ) = 2 :=
by sorry

end min_value_of_expression_l2183_218365


namespace sum_property_implies_isosceles_l2183_218314

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = π

-- Define a quadrilateral
structure Quadrilateral where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  angle_sum : w + x + y + z = 2 * π

-- Define the property that for any two angles of the triangle, 
-- there is an angle in the quadrilateral equal to their sum
def has_sum_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∃ (i j : Fin 3) (k : Fin 4), 
    i ≠ j ∧ 
    match i, j with
    | 0, 1 | 1, 0 => q.w = t.a + t.b ∨ q.x = t.a + t.b ∨ q.y = t.a + t.b ∨ q.z = t.a + t.b
    | 0, 2 | 2, 0 => q.w = t.a + t.c ∨ q.x = t.a + t.c ∨ q.y = t.a + t.c ∨ q.z = t.a + t.c
    | 1, 2 | 2, 1 => q.w = t.b + t.c ∨ q.x = t.b + t.c ∨ q.y = t.b + t.c ∨ q.z = t.b + t.c
    | _, _ => False

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- The theorem to be proved
theorem sum_property_implies_isosceles (t : Triangle) (q : Quadrilateral) :
  has_sum_property t q → is_isosceles t :=
by sorry

end sum_property_implies_isosceles_l2183_218314


namespace time_to_fill_tank_with_hole_l2183_218388

/-- Time to fill tank with hole present -/
theorem time_to_fill_tank_with_hole 
  (pipe_fill_time : ℝ) 
  (hole_empty_time : ℝ) 
  (h1 : pipe_fill_time = 15) 
  (h2 : hole_empty_time = 60.000000000000014) : 
  (1 : ℝ) / ((1 / pipe_fill_time) - (1 / hole_empty_time)) = 20.000000000000001 := by
  sorry

end time_to_fill_tank_with_hole_l2183_218388


namespace binomial_expansion_properties_l2183_218360

/-- Given a binomial expansion (2x + 1/√x)^n where n is a positive integer,
    if the ratio of binomial coefficients of the second term to the third term is 2:5,
    then n = 6, the coefficient of x^3 is 240, and the sum of binomial terms is 728 -/
theorem binomial_expansion_properties (n : ℕ+) :
  (Nat.choose n 1 : ℚ) / (Nat.choose n 2 : ℚ) = 2 / 5 →
  (n = 6 ∧
   (Nat.choose 6 2 : ℕ) * 2^4 = 240 ∧
   (2^6 * Nat.choose 6 0 + 2^5 * Nat.choose 6 1 + 2^4 * Nat.choose 6 2 +
    2^3 * Nat.choose 6 3 + 2^2 * Nat.choose 6 4 + 2 * Nat.choose 6 5) = 728) :=
by sorry

end binomial_expansion_properties_l2183_218360


namespace imaginary_part_of_complex_fraction_l2183_218334

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((2 + Complex.I) / (1 - 2 * Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2183_218334


namespace repair_cost_calculation_l2183_218316

/-- Proves that the repair cost is $300 given the initial purchase price,
    selling price, and gain percentage. -/
theorem repair_cost_calculation (purchase_price selling_price : ℝ) (gain_percentage : ℝ) :
  purchase_price = 900 →
  selling_price = 1500 →
  gain_percentage = 25 →
  (selling_price / (1 + gain_percentage / 100)) - purchase_price = 300 := by
sorry

end repair_cost_calculation_l2183_218316


namespace allison_video_uploads_l2183_218340

/-- Prove that Allison uploaded 10 one-hour videos daily during the first half of June. -/
theorem allison_video_uploads :
  ∀ (x : ℕ),
  (15 * x + 15 * (2 * x) = 450) →
  x = 10 :=
by
  sorry

end allison_video_uploads_l2183_218340


namespace opposite_of_negative_2023_l2183_218322

theorem opposite_of_negative_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l2183_218322


namespace betty_needs_five_more_l2183_218366

-- Define the cost of the wallet
def wallet_cost : ℕ := 100

-- Define Betty's initial savings
def betty_initial_savings : ℕ := wallet_cost / 2

-- Define the amount Betty's parents give her
def parents_contribution : ℕ := 15

-- Define the amount Betty's grandparents give her
def grandparents_contribution : ℕ := 2 * parents_contribution

-- Define Betty's total savings after contributions
def betty_total_savings : ℕ := betty_initial_savings + parents_contribution + grandparents_contribution

-- Theorem: Betty needs $5 more to buy the wallet
theorem betty_needs_five_more : wallet_cost - betty_total_savings = 5 := by
  sorry

end betty_needs_five_more_l2183_218366


namespace range_of_3a_minus_2b_l2183_218370

theorem range_of_3a_minus_2b (a b : ℝ) 
  (h1 : -3 ≤ a + b ∧ a + b ≤ 2) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 4) : 
  -4 ≤ 3*a - 2*b ∧ 3*a - 2*b ≤ 11 := by sorry

end range_of_3a_minus_2b_l2183_218370


namespace girls_fraction_is_half_l2183_218358

/-- Given a class of students, prove that the fraction of the number of girls
    that equals 1/3 of the total number of students is 1/2, when the ratio of
    boys to girls is 1/2. -/
theorem girls_fraction_is_half (T G B : ℚ) : 
  T > 0 → G > 0 → B > 0 →
  T = G + B →
  B / G = 1 / 2 →
  ∃ (f : ℚ), f * G = (1 / 3) * T ∧ f = 1 / 2 := by
  sorry

end girls_fraction_is_half_l2183_218358


namespace john_payment_amount_l2183_218305

/-- The final amount John needs to pay after late charges -/
def final_amount (original_bill : ℝ) (first_charge : ℝ) (second_charge : ℝ) (third_charge : ℝ) : ℝ :=
  original_bill * (1 + first_charge) * (1 + second_charge) * (1 + third_charge)

/-- Theorem stating the final amount John needs to pay -/
theorem john_payment_amount :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_amount 500 0.02 0.03 0.025 - 538.43| < ε :=
sorry

end john_payment_amount_l2183_218305


namespace skips_mode_is_165_l2183_218395

def skips : List ℕ := [165, 165, 165, 165, 165, 170, 170, 145, 150, 150]

def mode (l : List ℕ) : ℕ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem skips_mode_is_165 : mode skips = 165 := by sorry

end skips_mode_is_165_l2183_218395


namespace cost_per_item_l2183_218390

theorem cost_per_item (total_customers : ℕ) (purchase_percentage : ℚ) (total_profit : ℚ) : 
  total_customers = 100 → 
  purchase_percentage = 80 / 100 → 
  total_profit = 1000 → 
  total_profit / (total_customers * purchase_percentage) = 25 / 2 := by
sorry

end cost_per_item_l2183_218390


namespace triplets_equal_sum_l2183_218373

/-- The number of ordered triplets (m, n, p) of nonnegative integers satisfying m + 3n + 5p ≤ 600 -/
def countTriplets : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + 3 * t.2.1 + 5 * t.2.2 ≤ 600) (Finset.product (Finset.range 601) (Finset.product (Finset.range 201) (Finset.range 121)))).card

/-- The sum of (i+1) for all nonnegative integer solutions of i + 3j + 5k = 600 -/
def sumSolutions : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + 3 * t.2.1 + 5 * t.2.2 = 600) (Finset.product (Finset.range 601) (Finset.product (Finset.range 201) (Finset.range 121)))).sum (fun t => t.1 + 1)

theorem triplets_equal_sum : countTriplets = sumSolutions := by
  sorry

end triplets_equal_sum_l2183_218373


namespace geometric_sequence_ratio_l2183_218344

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (q > 0) →  -- q is positive
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (a 3 * a 9 = 2 * (a 5)^2) →  -- given condition
  q = Real.sqrt 2 := by
sorry

end geometric_sequence_ratio_l2183_218344


namespace cylinder_volume_increase_l2183_218389

/-- Represents the volume multiplication factor of a cylinder when its height is tripled and radius is increased by 300% -/
def cylinder_volume_factor : ℝ := 48

/-- Theorem stating that when a cylinder's height is tripled and its radius is increased by 300%, its volume is multiplied by a factor of 48 -/
theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_r := 4 * r
  let new_h := 3 * h
  (π * new_r^2 * new_h) / (π * r^2 * h) = cylinder_volume_factor :=
by sorry

end cylinder_volume_increase_l2183_218389


namespace shirts_made_yesterday_is_nine_l2183_218341

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 3

/-- The number of minutes the machine worked yesterday -/
def minutes_worked_yesterday : ℕ := 3

/-- The number of shirts made yesterday -/
def shirts_made_yesterday : ℕ := shirts_per_minute * minutes_worked_yesterday

theorem shirts_made_yesterday_is_nine : shirts_made_yesterday = 9 := by
  sorry

end shirts_made_yesterday_is_nine_l2183_218341


namespace functional_relationship_l2183_218350

/-- Given a function y that is the sum of two components y₁ and y₂,
    where y₁ is directly proportional to x and y₂ is inversely proportional to (x-2),
    prove that y = x + 2/(x-2) when y = -1 at x = 1 and y = 5 at x = 3. -/
theorem functional_relationship (y y₁ y₂ : ℝ → ℝ) (k₁ k₂ : ℝ) :
  (∀ x, y x = y₁ x + y₂ x) →
  (∀ x, y₁ x = k₁ * x) →
  (∀ x, y₂ x = k₂ / (x - 2)) →
  y 1 = -1 →
  y 3 = 5 →
  ∀ x, y x = x + 2 / (x - 2) :=
by sorry

end functional_relationship_l2183_218350


namespace dice_arithmetic_progression_probability_l2183_218321

-- Define the number of faces on a die
def die_faces : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 4

-- Define the possible common differences
def common_differences : List ℕ := [1, 2]

-- Define a function to calculate the total number of outcomes
def total_outcomes : ℕ := die_faces ^ num_dice

-- Define a function to calculate the favorable outcomes
def favorable_outcomes : ℕ := sorry

-- The main theorem
theorem dice_arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by sorry

end dice_arithmetic_progression_probability_l2183_218321


namespace high_school_total_students_l2183_218369

/-- Represents a high school with three grades -/
structure HighSchool :=
  (freshman_count : ℕ)
  (sophomore_count : ℕ)
  (senior_count : ℕ)

/-- Represents a stratified sample from the high school -/
structure StratifiedSample :=
  (freshman_sample : ℕ)
  (sophomore_sample : ℕ)
  (senior_sample : ℕ)

/-- The total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.freshman_count + hs.sophomore_count + hs.senior_count

/-- The total number of students in the sample -/
def total_sample (s : StratifiedSample) : ℕ :=
  s.freshman_sample + s.sophomore_sample + s.senior_sample

theorem high_school_total_students 
  (hs : HighSchool) 
  (sample : StratifiedSample) 
  (h1 : hs.freshman_count = 400)
  (h2 : sample.sophomore_sample = 15)
  (h3 : sample.senior_sample = 10)
  (h4 : total_sample sample = 45) :
  total_students hs = 900 :=
sorry

end high_school_total_students_l2183_218369


namespace det_A_zero_l2183_218323

theorem det_A_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h : A = A * B - B * A + A^2 * B - 2 * A * B * A + B * A^2 + A^2 * B * A - A * B * A^2) : 
  Matrix.det A = 0 := by
sorry

end det_A_zero_l2183_218323


namespace area_of_fifth_rectangle_l2183_218342

/-- Given a rectangle divided into five smaller rectangles, prove the area of the fifth rectangle --/
theorem area_of_fifth_rectangle
  (x y n k m : ℝ)
  (a b c d : ℝ)
  (h1 : a = k * (y - n))
  (h2 : b = (m - k) * (y - n))
  (h3 : c = m * (y - n))
  (h4 : d = (x - m) * n)
  (h5 : 0 < x ∧ 0 < y ∧ 0 < n ∧ 0 < k ∧ 0 < m)
  (h6 : n < y ∧ k < m ∧ m < x) :
  x * y - a - b - c - d = x * y - x * n :=
sorry

end area_of_fifth_rectangle_l2183_218342


namespace profit_loss_ratio_l2183_218311

theorem profit_loss_ratio (c x y : ℝ) (hx : x = 0.85 * c) (hy : y = 1.15 * c) :
  y / x = 23 / 17 := by
  sorry

end profit_loss_ratio_l2183_218311


namespace no_further_simplification_l2183_218377

theorem no_further_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  ∀ (f : ℝ → ℝ), f (a/b - b/a + a^2*b^2) = a/b - b/a + a^2*b^2 → f = id := by
  sorry

end no_further_simplification_l2183_218377


namespace unique_cube_property_l2183_218363

theorem unique_cube_property :
  ∃! (n : ℕ), n > 0 ∧ n^3 / 1000 = n :=
by sorry

end unique_cube_property_l2183_218363


namespace sum_difference_equality_l2183_218378

theorem sum_difference_equality : 3.59 + 2.4 - 1.67 = 4.32 := by
  sorry

end sum_difference_equality_l2183_218378


namespace rotate_5_plus_2i_l2183_218310

/-- Rotates a complex number by 90 degrees counter-clockwise around the origin -/
def rotate90 (z : ℂ) : ℂ := z * Complex.I

/-- The result of rotating 5 + 2i by 90 degrees counter-clockwise around the origin -/
theorem rotate_5_plus_2i : rotate90 (5 + 2*Complex.I) = -2 + 5*Complex.I := by
  sorry

end rotate_5_plus_2i_l2183_218310


namespace triangle_perimeter_is_six_l2183_218332

/-- Given a non-isosceles triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its perimeter is 6 under certain conditions. -/
theorem triangle_perimeter_is_six 
  (a b c A B C : ℝ) 
  (h_non_isosceles : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a * (Real.cos (C / 2))^2 + c * (Real.cos (A / 2))^2 = 3 * c / 2)
  (h_sines : 2 * Real.sin (A - B) + b * Real.sin B = a * Real.sin A) :
  a + b + c = 6 := by
  sorry

end triangle_perimeter_is_six_l2183_218332


namespace cubic_function_has_three_roots_l2183_218300

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem cubic_function_has_three_roots :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c :=
sorry

end cubic_function_has_three_roots_l2183_218300


namespace contrapositive_equivalence_l2183_218397

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 ≥ 4 → x ≤ -2 ∨ x ≥ 2)) ↔
  (∀ x : ℝ, (-2 < x ∧ x < 2 → x^2 < 4)) :=
by sorry

end contrapositive_equivalence_l2183_218397


namespace tiffany_score_l2183_218319

/-- The score for each treasure found -/
def points_per_treasure : ℕ := 6

/-- The number of treasures found on the first level -/
def treasures_level1 : ℕ := 3

/-- The number of treasures found on the second level -/
def treasures_level2 : ℕ := 5

/-- Tiffany's total score -/
def total_score : ℕ := points_per_treasure * (treasures_level1 + treasures_level2)

theorem tiffany_score : total_score = 48 := by
  sorry

end tiffany_score_l2183_218319


namespace triangle_isosceles_l2183_218357

-- Define a structure for a triangle
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r

-- Define the condition for triangle existence
def triangleExists (t : Triangle) (n : ℕ) : Prop :=
  t.p^n + t.q^n > t.r^n ∧ t.q^n + t.r^n > t.p^n ∧ t.r^n + t.p^n > t.q^n

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.p = t.q ∨ t.q = t.r ∨ t.r = t.p

-- The main theorem
theorem triangle_isosceles (t : Triangle) 
  (h : ∀ n : ℕ, triangleExists t n) : isIsosceles t := by
  sorry

end triangle_isosceles_l2183_218357


namespace triangle_angle_measure_l2183_218353

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < B → B < π →
  a = b * Real.cos C + c * Real.sin B →
  B = π / 4 :=
sorry

end triangle_angle_measure_l2183_218353


namespace line_passes_through_fixed_point_l2183_218312

/-- The line equation passing through a fixed point -/
def line_equation (m x y : ℝ) : Prop :=
  (m - 2) * x - y + 3 * m + 2 = 0

/-- Theorem stating that the line always passes through the point (-3, 8) -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (-3) 8 := by
  sorry

end line_passes_through_fixed_point_l2183_218312


namespace shrimp_earnings_l2183_218361

/-- Calculates the earnings of each boy from catching and selling shrimp --/
theorem shrimp_earnings (victor_shrimp : ℕ) (austin_diff : ℕ) (price : ℚ) (price_per : ℕ) :
  victor_shrimp = 26 →
  austin_diff = 8 →
  price = 7 →
  price_per = 11 →
  let austin_shrimp := victor_shrimp - austin_diff
  let brian_shrimp := (victor_shrimp + austin_shrimp) / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let total_earnings := (total_shrimp / price_per : ℚ) * price
  total_earnings / 3 = 14 := by
sorry


end shrimp_earnings_l2183_218361


namespace f_value_at_one_l2183_218354

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_value_at_one (m : ℝ) :
  (∀ x ≥ -2, ∀ y ≥ -2, x < y → f m x < f m y) →
  (∀ x ≤ -2, ∀ y ≤ -2, x < y → f m x > f m y) →
  f m 1 = 25 := by
  sorry

end f_value_at_one_l2183_218354


namespace xy_equals_one_l2183_218338

theorem xy_equals_one (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 36 → y = 0.16666666666666666 → x * y = 1 := by
  sorry

end xy_equals_one_l2183_218338


namespace students_neither_music_nor_art_l2183_218379

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 30) 
  (h3 : art = 10) 
  (h4 : both = 10) : 
  total - (music + art - both) = 470 := by
  sorry

end students_neither_music_nor_art_l2183_218379


namespace smallest_power_comparison_l2183_218398

theorem smallest_power_comparison : 127^8 < 63^10 ∧ 63^10 < 33^12 := by
  sorry

end smallest_power_comparison_l2183_218398


namespace fourth_power_complex_equality_l2183_218393

theorem fourth_power_complex_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.mk a b)^4 = (Complex.mk a (-b))^4 → b / a = 1 := by
  sorry

end fourth_power_complex_equality_l2183_218393


namespace giants_playoff_fraction_l2183_218324

theorem giants_playoff_fraction :
  let games_played : ℕ := 20
  let games_won : ℕ := 12
  let games_left : ℕ := 10
  let additional_wins_needed : ℕ := 8
  let total_games : ℕ := games_played + games_left
  let total_wins_needed : ℕ := games_won + additional_wins_needed
  (total_wins_needed : ℚ) / total_games = 2 / 3 := by
  sorry

end giants_playoff_fraction_l2183_218324


namespace election_results_l2183_218352

/-- Represents a candidate in the election -/
inductive Candidate
  | Montoran
  | AjudaPinto
  | VidameOfOussel

/-- Represents a voter group with their preferences -/
structure VoterGroup where
  size : Nat
  preferences : List Candidate

/-- Represents the election setup -/
structure Election where
  totalVoters : Nat
  candidates : List Candidate
  voterGroups : List VoterGroup

/-- One-round voting system -/
def oneRoundWinner (e : Election) : Candidate := sorry

/-- Two-round voting system -/
def twoRoundWinner (e : Election) : Candidate := sorry

/-- Three-round voting system -/
def threeRoundWinner (e : Election) : Candidate := sorry

/-- The election setup based on the problem description -/
def electionSetup : Election :=
  { totalVoters := 100000
  , candidates := [Candidate.Montoran, Candidate.AjudaPinto, Candidate.VidameOfOussel]
  , voterGroups :=
    [ { size := 33000
      , preferences := [Candidate.Montoran, Candidate.AjudaPinto, Candidate.VidameOfOussel]
      }
    , { size := 18000
      , preferences := [Candidate.AjudaPinto, Candidate.Montoran, Candidate.VidameOfOussel]
      }
    , { size := 12000
      , preferences := [Candidate.AjudaPinto, Candidate.VidameOfOussel, Candidate.Montoran]
      }
    , { size := 37000
      , preferences := [Candidate.VidameOfOussel, Candidate.AjudaPinto, Candidate.Montoran]
      }
    ]
  }

theorem election_results (e : Election) :
  e = electionSetup →
  oneRoundWinner e = Candidate.VidameOfOussel ∧
  twoRoundWinner e = Candidate.Montoran ∧
  threeRoundWinner e = Candidate.AjudaPinto :=
sorry

end election_results_l2183_218352


namespace unique_a_for_three_element_set_l2183_218387

theorem unique_a_for_three_element_set : ∃! (a : ℝ), 
  let A : Set ℝ := {a^2, 2-a, 4}
  (Fintype.card A = 3) ∧ (a = 6) := by sorry

end unique_a_for_three_element_set_l2183_218387


namespace men_work_hours_l2183_218383

theorem men_work_hours (men : ℕ) (women : ℕ) (men_days : ℕ) (women_days : ℕ) (women_hours : ℕ) (H : ℚ) :
  men = 15 →
  women = 21 →
  men_days = 21 →
  women_days = 60 →
  women_hours = 3 →
  (3 : ℚ) * men * men_days * H = 2 * women * women_days * women_hours →
  H = 8 := by
sorry

end men_work_hours_l2183_218383


namespace weekly_allowance_calculation_l2183_218343

/-- Represents the daily calorie allowance for a person in their 60's. -/
def daily_allowance : ℕ := 2000

/-- Represents the number of days in a week. -/
def days_in_week : ℕ := 7

/-- Calculates the weekly calorie allowance based on the daily allowance. -/
def weekly_allowance : ℕ := daily_allowance * days_in_week

/-- Proves that the weekly calorie allowance for a person in their 60's
    with an average daily allowance of 2000 calories is equal to 10500 calories. -/
theorem weekly_allowance_calculation :
  weekly_allowance = 10500 := by
  sorry

end weekly_allowance_calculation_l2183_218343


namespace two_integers_sum_l2183_218304

theorem two_integers_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 120) :
  x + y = 15 := by sorry

end two_integers_sum_l2183_218304


namespace quadratic_rational_roots_l2183_218384

theorem quadratic_rational_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a = 1 ∧ b = 2 ∧ c = -3 →
  ∃ (x y : ℚ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end quadratic_rational_roots_l2183_218384


namespace valid_queue_arrangements_correct_l2183_218333

/-- Represents the number of valid queue arrangements for a concert ticket purchase scenario. -/
def validQueueArrangements (m n : ℕ) : ℚ :=
  if n ≥ m then
    (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))
  else 0

/-- Theorem stating the correctness of the validQueueArrangements function. -/
theorem valid_queue_arrangements_correct (m n : ℕ) (h : n ≥ m) :
  validQueueArrangements m n = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by sorry

end valid_queue_arrangements_correct_l2183_218333


namespace circle_area_tripled_l2183_218326

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (1 - Real.sqrt 3) / 2) :=
by sorry

end circle_area_tripled_l2183_218326


namespace equalize_piles_in_three_moves_l2183_218396

/-- Represents a configuration of pin piles -/
structure PinPiles :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a move between two piles -/
inductive Move
  | one_to_two
  | one_to_three
  | two_to_one
  | two_to_three
  | three_to_one
  | three_to_two

/-- Applies a move to a given configuration -/
def apply_move (piles : PinPiles) (move : Move) : PinPiles :=
  match move with
  | Move.one_to_two => PinPiles.mk (piles.pile1 - piles.pile2) (piles.pile2 * 2) piles.pile3
  | Move.one_to_three => PinPiles.mk (piles.pile1 - piles.pile3) piles.pile2 (piles.pile3 * 2)
  | Move.two_to_one => PinPiles.mk (piles.pile1 * 2) (piles.pile2 - piles.pile1) piles.pile3
  | Move.two_to_three => PinPiles.mk piles.pile1 (piles.pile2 - piles.pile3) (piles.pile3 * 2)
  | Move.three_to_one => PinPiles.mk (piles.pile1 * 2) piles.pile2 (piles.pile3 - piles.pile1)
  | Move.three_to_two => PinPiles.mk piles.pile1 (piles.pile2 * 2) (piles.pile3 - piles.pile2)

/-- The main theorem to be proved -/
theorem equalize_piles_in_three_moves :
  ∃ (m1 m2 m3 : Move),
    let initial := PinPiles.mk 11 7 6
    let step1 := apply_move initial m1
    let step2 := apply_move step1 m2
    let step3 := apply_move step2 m3
    step3 = PinPiles.mk 8 8 8 :=
by
  sorry

end equalize_piles_in_three_moves_l2183_218396


namespace basketball_team_enrollment_l2183_218372

theorem basketball_team_enrollment (total : ℕ) (math : ℕ) (both : ℕ) (physics : ℕ) : 
  total = 15 → math = 9 → both = 4 → physics = total - (math - both) → physics = 10 := by
  sorry

end basketball_team_enrollment_l2183_218372


namespace girls_to_boys_ratio_l2183_218329

/-- Proves that in a class with 35 students, where there are seven more girls than boys, 
    the ratio of girls to boys is 3:2 -/
theorem girls_to_boys_ratio (total : ℕ) (girls boys : ℕ) : 
  total = 35 →
  girls = boys + 7 →
  girls + boys = total →
  (girls : ℚ) / (boys : ℚ) = 3 / 2 := by
sorry

end girls_to_boys_ratio_l2183_218329


namespace x_squared_minus_y_squared_l2183_218349

theorem x_squared_minus_y_squared (x y : ℝ) 
  (sum : x + y = 20) 
  (diff : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end x_squared_minus_y_squared_l2183_218349


namespace star_interior_angle_sum_formula_l2183_218346

/-- An n-pointed star is formed from a convex n-gon by extending each side k
    to intersect with side k+3 (modulo n). This function calculates the
    sum of interior angles at the n vertices of the resulting star. -/
def starInteriorAngleSum (n : ℕ) : ℝ :=
  180 * (n - 6 : ℝ)

/-- Theorem stating that for an n-pointed star (n ≥ 5), the sum of
    interior angles at the n vertices is 180(n-6) degrees. -/
theorem star_interior_angle_sum_formula {n : ℕ} (h : n ≥ 5) :
  starInteriorAngleSum n = 180 * (n - 6 : ℝ) := by
  sorry

end star_interior_angle_sum_formula_l2183_218346


namespace patricks_age_l2183_218301

/-- Given that Patrick is half the age of his elder brother Robert, and Robert will turn 30 after 2 years, prove that Patrick's current age is 14 years. -/
theorem patricks_age (robert_age_in_two_years : ℕ) (robert_current_age : ℕ) (patrick_age : ℕ) : 
  robert_age_in_two_years = 30 → 
  robert_current_age = robert_age_in_two_years - 2 →
  patrick_age = robert_current_age / 2 →
  patrick_age = 14 := by
sorry

end patricks_age_l2183_218301


namespace watermelon_stand_problem_l2183_218327

/-- A watermelon stand problem -/
theorem watermelon_stand_problem (total_melons : ℕ) 
  (single_melon_customers : ℕ) (triple_melon_customers : ℕ) :
  total_melons = 46 →
  single_melon_customers = 17 →
  triple_melon_customers = 3 →
  total_melons - (single_melon_customers * 1 + triple_melon_customers * 3) = 20 := by
  sorry

end watermelon_stand_problem_l2183_218327


namespace alcohol_mixture_percentage_l2183_218308

/-- Proves that mixing 100 mL of 10% alcohol solution with 300 mL of 30% alcohol solution 
    results in a 25% alcohol solution -/
theorem alcohol_mixture_percentage :
  let x_volume : ℝ := 100
  let x_percentage : ℝ := 10
  let y_volume : ℝ := 300
  let y_percentage : ℝ := 30
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := (x_volume * x_percentage + y_volume * y_percentage) / 100
  total_alcohol / total_volume * 100 = 25 := by
  sorry

end alcohol_mixture_percentage_l2183_218308


namespace william_max_riding_time_l2183_218317

/-- Represents the maximum number of hours William can ride his horse per day -/
def max_riding_time : ℝ := 6

/-- The total number of days William rode -/
def total_days : ℕ := 6

/-- The number of days William rode for the maximum time -/
def max_time_days : ℕ := 2

/-- The number of days William rode for 1.5 hours -/
def short_ride_days : ℕ := 2

/-- The number of days William rode for half the maximum time -/
def half_time_days : ℕ := 2

/-- The duration of a short ride in hours -/
def short_ride_duration : ℝ := 1.5

/-- The total riding time over all days in hours -/
def total_riding_time : ℝ := 21

theorem william_max_riding_time :
  max_riding_time * max_time_days +
  short_ride_duration * short_ride_days +
  (max_riding_time / 2) * half_time_days = total_riding_time ∧
  max_time_days + short_ride_days + half_time_days = total_days :=
by sorry

end william_max_riding_time_l2183_218317


namespace divisibility_implies_equality_l2183_218394

theorem divisibility_implies_equality (a b : ℕ+) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end divisibility_implies_equality_l2183_218394


namespace no_valid_number_l2183_218303

theorem no_valid_number : ¬∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧  -- 3-digit number
  (∃ (x : ℕ), x < 10 ∧ n = 520 + x) ∧  -- in the form 52x where x is a digit
  (n % 6 = 0) ∧  -- divisible by 6
  (n % 10 = 6)  -- last digit is 6
  := by sorry

end no_valid_number_l2183_218303


namespace f_derivative_at_one_l2183_218367

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem f_derivative_at_one : 
  deriv f 1 = 2 := by sorry

end f_derivative_at_one_l2183_218367


namespace solve_for_t_l2183_218356

-- Define the variables
variable (s t : ℝ)

-- State the theorem
theorem solve_for_t (eq1 : 7 * s + 3 * t = 82) (eq2 : s = 2 * t - 3) : t = 103 / 17 := by
  sorry

end solve_for_t_l2183_218356


namespace smallest_integer_proof_l2183_218380

/-- The smallest positive integer that can be represented as CC₆ and DD₈ -/
def smallest_integer : ℕ := 63

/-- Conversion from base 6 to base 10 -/
def base_6_to_10 (c : ℕ) : ℕ := 6 * c + c

/-- Conversion from base 8 to base 10 -/
def base_8_to_10 (d : ℕ) : ℕ := 8 * d + d

/-- Theorem stating that 63 is the smallest positive integer representable as CC₆ and DD₈ -/
theorem smallest_integer_proof :
  ∃ (c d : ℕ),
    c < 6 ∧ d < 8 ∧
    base_6_to_10 c = smallest_integer ∧
    base_8_to_10 d = smallest_integer ∧
    ∀ (n : ℕ), n > 0 ∧ (∃ (c' d' : ℕ), c' < 6 ∧ d' < 8 ∧ base_6_to_10 c' = n ∧ base_8_to_10 d' = n) →
      n ≥ smallest_integer :=
by sorry

end smallest_integer_proof_l2183_218380


namespace product_maximum_l2183_218382

theorem product_maximum (s : ℝ) (hs : s > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = s → x * y ≥ a * b ∧
  x * y = s^2 / 4 :=
sorry

end product_maximum_l2183_218382


namespace circle_radius_l2183_218374

theorem circle_radius (x y : ℝ) :
  x^2 - 8*x + y^2 + 4*y + 9 = 0 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 11 :=
by sorry

end circle_radius_l2183_218374


namespace division_ways_count_l2183_218385

def number_of_people : ℕ := 6
def number_of_cars : ℕ := 2
def max_capacity_per_car : ℕ := 4

theorem division_ways_count :
  (Finset.sum (Finset.range (min number_of_people (max_capacity_per_car + 1)))
    (λ i => (number_of_people.choose i) * ((number_of_people - i).choose (number_of_people - i)))) = 60 := by
  sorry

end division_ways_count_l2183_218385


namespace sum_of_three_numbers_l2183_218307

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 57)
  (sum_ca : c + a = 62) :
  a + b + c = 77 := by
sorry

end sum_of_three_numbers_l2183_218307


namespace min_value_a2_plus_b2_l2183_218328

theorem min_value_a2_plus_b2 (a b : ℝ) (h : (9 : ℝ) / a^2 + (4 : ℝ) / b^2 = 1) :
  ∀ x y : ℝ, (9 : ℝ) / x^2 + (4 : ℝ) / y^2 = 1 → x^2 + y^2 ≥ 25 :=
by sorry

end min_value_a2_plus_b2_l2183_218328


namespace inverse_co_complementary_angles_equal_l2183_218330

/-- For any two angles α and β, if their co-complementary angles are equal, then α and β are equal. -/
theorem inverse_co_complementary_angles_equal (α β : Real) :
  (90 - α = 90 - β) → α = β := by
  sorry

end inverse_co_complementary_angles_equal_l2183_218330


namespace num_groupings_l2183_218351

/-- The number of ways to distribute n items into 2 non-empty groups -/
def distribute (n : ℕ) : ℕ :=
  2^n - 2

/-- The number of tour guides -/
def num_guides : ℕ := 2

/-- The number of tourists -/
def num_tourists : ℕ := 6

/-- Each guide must have at least one tourist -/
axiom guides_not_empty : distribute num_tourists ≥ 1

/-- Theorem: The number of ways to distribute 6 tourists between 2 guides, 
    with each guide having at least one tourist, is 62 -/
theorem num_groupings : distribute num_tourists = 62 := by
  sorry

end num_groupings_l2183_218351


namespace total_spent_on_fruits_l2183_218345

def total_fruits : ℕ := 32
def plum_cost : ℕ := 2
def peach_cost : ℕ := 1
def plums_bought : ℕ := 20

theorem total_spent_on_fruits : 
  plums_bought * plum_cost + (total_fruits - plums_bought) * peach_cost = 52 := by
  sorry

end total_spent_on_fruits_l2183_218345


namespace min_value_sum_of_squares_l2183_218320

theorem min_value_sum_of_squares (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 9 ∧
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) = 9 :=
sorry

end min_value_sum_of_squares_l2183_218320


namespace brian_tennis_balls_l2183_218364

/-- Given the number of tennis balls for Lily, Frodo, and Brian, prove that Brian has 22 tennis balls. -/
theorem brian_tennis_balls (lily frodo brian : ℕ) 
  (h1 : lily = 3)
  (h2 : frodo = lily + 8)
  (h3 : brian = 2 * frodo) :
  brian = 22 := by
  sorry

end brian_tennis_balls_l2183_218364


namespace simplify_calculations_l2183_218355

theorem simplify_calculations :
  (329 * 101 = 33229) ∧
  (54 * 98 + 46 * 98 = 9800) ∧
  (98 * 125 = 12250) ∧
  (37 * 29 + 37 = 1110) := by
  sorry

end simplify_calculations_l2183_218355


namespace solution_count_correct_l2183_218375

/-- The number of integers n satisfying the equation 1 + ⌊(100n)/103⌋ = ⌈(97n)/100⌉ -/
def solution_count : ℕ := 10300

/-- Function g(n) defined as ⌈(97n)/100⌉ - ⌊(100n)/103⌋ -/
def g (n : ℤ) : ℤ := ⌈(97 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 103⌋

/-- The main theorem stating that the number of solutions is equal to solution_count -/
theorem solution_count_correct :
  (∑' n : ℤ, if 1 + ⌊(100 * n : ℚ) / 103⌋ = ⌈(97 * n : ℚ) / 100⌉ then 1 else 0) = solution_count :=
sorry

/-- Lemma showing the periodic behavior of g(n) -/
lemma g_periodic (n : ℤ) : g (n + 10300) = g n + 3 :=
sorry

/-- Lemma stating that for each residue class modulo 10300, there exists a unique solution -/
lemma unique_solution_per_residue_class (r : ℤ) :
  ∃! n : ℤ, g n = 1 ∧ n ≡ r [ZMOD 10300] :=
sorry

end solution_count_correct_l2183_218375


namespace q_necessary_not_sufficient_l2183_218335

-- Define the propositions
def p (b : ℝ) : Prop := ∃ r : ℝ, r ≠ 0 ∧ b = 1 * r ∧ 9 = b * r

def q (b : ℝ) : Prop := b = 3

-- State the theorem
theorem q_necessary_not_sufficient :
  (∀ b : ℝ, p b → q b) ∧ (∃ b : ℝ, p b ∧ ¬q b) := by
  sorry

end q_necessary_not_sufficient_l2183_218335


namespace number_problem_l2183_218339

theorem number_problem (n : ℝ) (h : (1/3) * (1/4) * n = 18) : (3/10) * n = 64.8 := by
  sorry

end number_problem_l2183_218339


namespace bakery_boxes_l2183_218306

theorem bakery_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (additional_boxes : ℕ) 
  (h1 : total_muffins = 95)
  (h2 : muffins_per_box = 5)
  (h3 : additional_boxes = 9) :
  total_muffins / muffins_per_box - additional_boxes = 10 :=
by sorry

end bakery_boxes_l2183_218306


namespace calculate_wins_l2183_218368

/-- Given a team's home game statistics, calculate the number of wins -/
theorem calculate_wins (total_games losses : ℕ) (h1 : total_games = 56) (h2 : losses = 12) : 
  total_games - losses - (losses / 2) = 38 := by
  sorry

#check calculate_wins

end calculate_wins_l2183_218368


namespace vacuuming_time_ratio_l2183_218302

theorem vacuuming_time_ratio : 
  ∀ (time_downstairs : ℝ),
  time_downstairs > 0 →
  27 = time_downstairs + 5 →
  38 = 27 + time_downstairs →
  (27 : ℝ) / time_downstairs = 27 / 22 :=
by
  sorry

end vacuuming_time_ratio_l2183_218302


namespace farm_milk_production_l2183_218347

/-- Calculates the weekly milk production for a farm -/
def weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * 7

/-- Theorem: A farm with 52 cows, each producing 5 liters of milk per day, produces 1820 liters of milk in a week -/
theorem farm_milk_production :
  weekly_milk_production 52 5 = 1820 := by
  sorry

end farm_milk_production_l2183_218347


namespace scale_division_l2183_218381

/-- Proves that dividing a scale of length 80 inches into 4 equal parts results in each part having a length of 20 inches. -/
theorem scale_division (scale_length : ℕ) (num_parts : ℕ) (part_length : ℕ) 
  (h1 : scale_length = 80) 
  (h2 : num_parts = 4) 
  (h3 : part_length * num_parts = scale_length) : 
  part_length = 20 := by
  sorry

end scale_division_l2183_218381


namespace cyclist_energized_time_l2183_218331

/-- Given a cyclist who rides at different speeds when energized and exhausted,
    prove the time spent energized for a specific total distance and time. -/
theorem cyclist_energized_time
  (speed_energized : ℝ)
  (speed_exhausted : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (h_speed_energized : speed_energized = 22)
  (h_speed_exhausted : speed_exhausted = 15)
  (h_total_distance : total_distance = 154)
  (h_total_time : total_time = 9)
  : ∃ (time_energized : ℝ),
    time_energized * speed_energized +
    (total_time - time_energized) * speed_exhausted = total_distance ∧
    time_energized = 19 / 7 := by
  sorry

end cyclist_energized_time_l2183_218331


namespace exists_scores_with_median_16_l2183_218391

/-- Represents a set of basketball scores -/
def BasketballScores := List ℕ

/-- Calculates the median of a list of natural numbers -/
def median (scores : BasketballScores) : ℚ :=
  sorry

/-- Theorem: There exists a set of basketball scores with a median of 16 -/
theorem exists_scores_with_median_16 : 
  ∃ (scores : BasketballScores), median scores = 16 := by
  sorry

end exists_scores_with_median_16_l2183_218391


namespace sony_games_to_give_away_l2183_218359

theorem sony_games_to_give_away (current_sony_games : ℕ) (target_sony_games : ℕ) :
  current_sony_games = 132 → target_sony_games = 31 →
  current_sony_games - target_sony_games = 101 :=
by
  sorry


end sony_games_to_give_away_l2183_218359


namespace swordfish_pufferfish_ratio_l2183_218325

/-- The ratio of swordfish to pufferfish in an aquarium -/
theorem swordfish_pufferfish_ratio 
  (total_fish : ℕ) 
  (pufferfish : ℕ) 
  (n : ℕ) 
  (h1 : total_fish = 90)
  (h2 : pufferfish = 15)
  (h3 : total_fish = n * pufferfish + pufferfish) :
  (n * pufferfish) / pufferfish = 5 := by
sorry

end swordfish_pufferfish_ratio_l2183_218325


namespace interview_probability_l2183_218386

/-- The number of students enrolled in at least one language class -/
def total_students : ℕ := 30

/-- The number of students enrolled in the German class -/
def german_students : ℕ := 20

/-- The number of students enrolled in the Italian class -/
def italian_students : ℕ := 22

/-- The probability of selecting two students such that at least one is enrolled in German
    and at least one is enrolled in Italian -/
def prob_both_classes : ℚ := 362 / 435

theorem interview_probability :
  prob_both_classes = 1 - (Nat.choose (german_students + italian_students - total_students) 2 +
                           Nat.choose (german_students - (german_students + italian_students - total_students)) 2 +
                           Nat.choose (italian_students - (german_students + italian_students - total_students)) 2) /
                          Nat.choose total_students 2 :=
by sorry

end interview_probability_l2183_218386


namespace complex_in_second_quadrant_l2183_218392

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant :
  let z : ℂ := -2 + I
  second_quadrant z := by
  sorry

end complex_in_second_quadrant_l2183_218392


namespace two_thousand_sixteenth_smallest_n_l2183_218315

/-- The number of ways Yang can reach (n,0) under the given movement rules -/
def a (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition an ≡ 1 (mod 5) -/
def satisfies_condition (n : ℕ) : Prop :=
  a n % 5 = 1

/-- The function that returns the kth smallest positive integer satisfying the condition -/
def kth_smallest (k : ℕ) : ℕ := sorry

theorem two_thousand_sixteenth_smallest_n :
  kth_smallest 2016 = 475756 :=
sorry

end two_thousand_sixteenth_smallest_n_l2183_218315


namespace geometric_sequence_fifth_term_l2183_218309

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 3 = 16)
  (h_sum : a 3 + a 4 = 24) :
  a 5 = 32 := by
sorry

end geometric_sequence_fifth_term_l2183_218309


namespace rectangle_measurement_error_l2183_218337

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * 0.95 = 1.102 → x = 16 := by sorry

end rectangle_measurement_error_l2183_218337


namespace function_equation_solution_l2183_218376

theorem function_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1)) :=
by sorry

end function_equation_solution_l2183_218376


namespace wendy_first_level_treasures_l2183_218318

/-- Represents the game scenario where Wendy finds treasures on two levels --/
structure GameScenario where
  pointsPerTreasure : ℕ
  treasuresOnSecondLevel : ℕ
  totalScore : ℕ

/-- Calculates the number of treasures found on the first level --/
def treasuresOnFirstLevel (game : GameScenario) : ℕ :=
  (game.totalScore - game.pointsPerTreasure * game.treasuresOnSecondLevel) / game.pointsPerTreasure

/-- Theorem stating that Wendy found 4 treasures on the first level --/
theorem wendy_first_level_treasures :
  let game : GameScenario := {
    pointsPerTreasure := 5,
    treasuresOnSecondLevel := 3,
    totalScore := 35
  }
  treasuresOnFirstLevel game = 4 := by sorry

end wendy_first_level_treasures_l2183_218318


namespace first_group_number_is_five_l2183_218313

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  groupSize : ℕ
  numberFromGroup17 : ℕ

/-- The systematic sampling scheme for the given problem -/
def problemSampling : SystematicSampling :=
  { totalStudents := 140
  , sampleSize := 20
  , groupSize := 7
  , numberFromGroup17 := 117
  }

/-- The number drawn from the first group in a systematic sampling -/
def firstGroupNumber (s : SystematicSampling) : ℕ :=
  s.numberFromGroup17 - s.groupSize * (17 - 1)

/-- Theorem stating that the number drawn from the first group is 5 -/
theorem first_group_number_is_five :
  firstGroupNumber problemSampling = 5 := by
  sorry

end first_group_number_is_five_l2183_218313


namespace largest_y_value_l2183_218399

/-- The largest possible value of y for regular polygons Q1 (x-gon) and Q2 (y-gon) -/
theorem largest_y_value (x y : ℕ) : 
  x ≥ y → 
  y ≥ 3 → 
  (x - 2) * y * 29 = (y - 2) * x * 28 → 
  y ≤ 57 :=
by sorry

end largest_y_value_l2183_218399


namespace min_triangle_area_l2183_218362

-- Define the triangle and square
structure Triangle :=
  (X Y Z : ℝ × ℝ)

structure Square :=
  (side : ℝ)
  (area : ℝ)

-- Define the properties
def is_acute_angled (t : Triangle) : Prop := sorry

def square_inscribed (t : Triangle) (s : Square) : Prop := sorry

-- Theorem statement
theorem min_triangle_area 
  (t : Triangle) 
  (s : Square) 
  (h_acute : is_acute_angled t) 
  (h_inscribed : square_inscribed t s) 
  (h_area : s.area = 2017) : 
  ∃ (min_area : ℝ), min_area = 2017/2 ∧ 
  ∀ (actual_area : ℝ), actual_area ≥ min_area := by
  sorry

end min_triangle_area_l2183_218362
