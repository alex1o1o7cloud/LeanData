import Mathlib

namespace isabella_book_purchase_l2038_203865

/-- The number of hardcover volumes bought by Isabella --/
def num_hardcovers : ℕ := 6

/-- The number of paperback volumes bought by Isabella --/
def num_paperbacks : ℕ := 12 - num_hardcovers

/-- The cost of a paperback volume in dollars --/
def paperback_cost : ℕ := 20

/-- The cost of a hardcover volume in dollars --/
def hardcover_cost : ℕ := 30

/-- The total number of volumes --/
def total_volumes : ℕ := 12

/-- The total cost of all volumes in dollars --/
def total_cost : ℕ := 300

theorem isabella_book_purchase :
  num_hardcovers = 6 ∧
  num_hardcovers + num_paperbacks = total_volumes ∧
  num_hardcovers * hardcover_cost + num_paperbacks * paperback_cost = total_cost :=
sorry

end isabella_book_purchase_l2038_203865


namespace absolute_value_inequality_l2038_203844

theorem absolute_value_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 := by sorry

end absolute_value_inequality_l2038_203844


namespace pyramid_volume_is_1280_l2038_203878

/-- Pyramid with square base ABCD and vertex E -/
structure Pyramid where
  baseArea : ℝ
  abeArea : ℝ
  cdeArea : ℝ
  distanceToMidpoint : ℝ

/-- Volume of the pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the volume of the pyramid is 1280 -/
theorem pyramid_volume_is_1280 (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.abeArea = 120)
  (h3 : p.cdeArea = 136)
  (h4 : p.distanceToMidpoint = 17) :
  pyramidVolume p = 1280 := by sorry

end pyramid_volume_is_1280_l2038_203878


namespace solve_equation_l2038_203854

theorem solve_equation (A : ℝ) : 3 + A = 4 → A = 1 := by
  sorry

end solve_equation_l2038_203854


namespace x_value_for_purely_imaginary_square_l2038_203806

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem x_value_for_purely_imaginary_square (x : ℝ) :
  x > 0 → isPurelyImaginary ((x - complex 0 1) ^ 2) → x = 1 := by
  sorry

end x_value_for_purely_imaginary_square_l2038_203806


namespace tangent_line_equation_l2038_203815

/-- The equation of the tangent line to y = xe^(x-1) at (1, 1) is y = 2x - 1 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.exp (x - 1)) → -- Curve equation
  (1 = 1 * Real.exp (1 - 1)) → -- Point (1, 1) satisfies the curve equation
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ 
    (y - 1 = m * (x - 1)) ∧   -- Point-slope form of tangent line
    (m = (1 + 1) * Real.exp (1 - 1)) ∧ -- Slope at x = 1
    (y = 2 * x - 1)) -- Equation of the tangent line
  := by sorry

end tangent_line_equation_l2038_203815


namespace unique_three_digit_number_divisible_by_nine_l2038_203893

theorem unique_three_digit_number_divisible_by_nine :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 5 ∧ 
  (n / 100) % 10 = 3 ∧ 
  n % 9 = 0 ∧
  n = 315 := by
  sorry

end unique_three_digit_number_divisible_by_nine_l2038_203893


namespace triangle_determinant_zero_l2038_203876

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end triangle_determinant_zero_l2038_203876


namespace sum_of_quotients_divisible_by_nine_l2038_203817

theorem sum_of_quotients_divisible_by_nine (n : ℕ) (hn : n > 8) :
  let a : ℕ → ℕ := λ i => (10^(2*i) - 1) / 9
  let q : ℕ → ℕ := λ i => a i / 11
  let s : ℕ → ℕ := λ i => (Finset.range 9).sum (λ j => q (i + j))
  ∀ i : ℕ, i ≤ n - 8 → (s i) % 9 = 0 := by
sorry

end sum_of_quotients_divisible_by_nine_l2038_203817


namespace headphones_savings_visits_l2038_203864

/-- The cost of the headphones in rubles -/
def headphones_cost : ℕ := 275

/-- The cost of a combined pool and sauna visit in rubles -/
def combined_cost : ℕ := 250

/-- The difference between pool-only cost and sauna-only cost in rubles -/
def pool_sauna_diff : ℕ := 200

/-- Calculates the cost of a pool-only visit -/
def pool_only_cost : ℕ := combined_cost - (combined_cost - pool_sauna_diff) / 2

/-- Calculates the savings per visit when choosing pool-only instead of combined -/
def savings_per_visit : ℕ := combined_cost - pool_only_cost

/-- The number of pool-only visits needed to save enough for the headphones -/
def visits_needed : ℕ := (headphones_cost + savings_per_visit - 1) / savings_per_visit

theorem headphones_savings_visits : visits_needed = 11 := by
  sorry

#eval visits_needed

end headphones_savings_visits_l2038_203864


namespace total_arrangements_with_at_least_one_girl_l2038_203866

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def choose (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_people : ℕ := num_boys + num_girls
def num_selected : ℕ := 3
def num_tasks : ℕ := 3

theorem total_arrangements_with_at_least_one_girl : 
  (choose num_people num_selected - choose num_boys num_selected) * factorial num_tasks = 186 := by
  sorry

end total_arrangements_with_at_least_one_girl_l2038_203866


namespace new_average_weight_l2038_203891

theorem new_average_weight (n : ℕ) (w_avg : ℝ) (w_new : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new = 4 →
  (n * w_avg + w_new) / (n + 1) = 27.2 := by
  sorry

end new_average_weight_l2038_203891


namespace four_common_tangents_l2038_203851

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Define the number of common tangent lines
def num_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem four_common_tangents :
  num_common_tangents circle_C1 circle_C2 = 4 := by sorry

end four_common_tangents_l2038_203851


namespace amanda_stroll_time_l2038_203895

/-- Amanda's stroll to Kimberly's house -/
theorem amanda_stroll_time (speed : ℝ) (distance : ℝ) (h1 : speed = 2) (h2 : distance = 6) :
  distance / speed = 3 := by
  sorry

end amanda_stroll_time_l2038_203895


namespace not_pythagorean_triple_8_12_16_l2038_203820

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

/-- Theorem: The set (8, 12, 16) is not a Pythagorean triple -/
theorem not_pythagorean_triple_8_12_16 :
  ¬ is_pythagorean_triple 8 12 16 := by
  sorry

end not_pythagorean_triple_8_12_16_l2038_203820


namespace cos_alpha_plus_seven_pi_twelfths_l2038_203813

theorem cos_alpha_plus_seven_pi_twelfths (α : ℝ) 
  (h : Real.sin (α + π / 12) = 1 / 3) : 
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end cos_alpha_plus_seven_pi_twelfths_l2038_203813


namespace problem_statement_l2038_203890

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
  10 * (6 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end problem_statement_l2038_203890


namespace exists_vertical_line_through_point_l2038_203804

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  slope : Option ℝ
  yIntercept : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  match l.slope with
  | some k => p.y = k * p.x + l.yIntercept
  | none => p.x = l.yIntercept

-- Theorem statement
theorem exists_vertical_line_through_point (b : ℝ) :
  ∃ (l : Line2D), pointOnLine ⟨0, b⟩ l ∧ l.slope = none :=
sorry

end exists_vertical_line_through_point_l2038_203804


namespace gcd_52800_35275_l2038_203869

theorem gcd_52800_35275 : Nat.gcd 52800 35275 = 25 := by
  sorry

end gcd_52800_35275_l2038_203869


namespace rectangle_shorter_side_l2038_203873

theorem rectangle_shorter_side (area perimeter : ℝ) (h_area : area = 104) (h_perimeter : perimeter = 42) :
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * (length + width) = perimeter ∧ 
    min length width = 8 := by
  sorry

end rectangle_shorter_side_l2038_203873


namespace min_value_of_sum_of_fractions_l2038_203801

theorem min_value_of_sum_of_fractions (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a ^ n) + 1 / (1 + b ^ n)) ≥ 1 ∧ 
  (1 / (1 + 1 ^ n) + 1 / (1 + 1 ^ n) = 1) :=
by sorry

end min_value_of_sum_of_fractions_l2038_203801


namespace min_sum_of_squares_l2038_203875

theorem min_sum_of_squares (x y z : ℝ) (h : x*y + y*z + x*z = 4) :
  x^2 + y^2 + z^2 ≥ 4 ∧ ∃ a b c : ℝ, a*b + b*c + a*c = 4 ∧ a^2 + b^2 + c^2 = 4 :=
by sorry

end min_sum_of_squares_l2038_203875


namespace divisibility_of_polynomial_l2038_203853

theorem divisibility_of_polynomial (n : ℤ) : 
  (120 : ℤ) ∣ (n^5 - 5*n^3 + 4*n) := by
  sorry

end divisibility_of_polynomial_l2038_203853


namespace chris_bluray_purchase_l2038_203859

/-- The number of Blu-ray movies Chris bought -/
def num_bluray : ℕ := sorry

/-- The number of DVD movies Chris bought -/
def num_dvd : ℕ := 8

/-- The price of each DVD movie -/
def price_dvd : ℚ := 12

/-- The price of each Blu-ray movie -/
def price_bluray : ℚ := 18

/-- The average price per movie -/
def avg_price : ℚ := 14

theorem chris_bluray_purchase :
  (num_dvd * price_dvd + num_bluray * price_bluray) / (num_dvd + num_bluray) = avg_price ∧
  num_bluray = 4 := by sorry

end chris_bluray_purchase_l2038_203859


namespace water_percentage_in_dried_grapes_l2038_203829

/-- Given that fresh grapes contain 60% water by weight and 30 kg of fresh grapes
    yields 15 kg of dried grapes, prove that the percentage of water in dried grapes is 20%. -/
theorem water_percentage_in_dried_grapes :
  let fresh_grape_weight : ℝ := 30
  let dried_grape_weight : ℝ := 15
  let fresh_water_percentage : ℝ := 60
  let water_weight_fresh : ℝ := fresh_grape_weight * (fresh_water_percentage / 100)
  let solid_weight : ℝ := fresh_grape_weight - water_weight_fresh
  let water_weight_dried : ℝ := dried_grape_weight - solid_weight
  let dried_water_percentage : ℝ := (water_weight_dried / dried_grape_weight) * 100
  dried_water_percentage = 20 := by
sorry

end water_percentage_in_dried_grapes_l2038_203829


namespace concatenated_number_divisibility_l2038_203823

theorem concatenated_number_divisibility
  (n : ℕ) (a : ℕ) (h_n : n > 1) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) :
  let b := a * 10^n + a
  (∃ d : ℕ, b = d * a^2) → b / a^2 = 7 :=
by sorry

end concatenated_number_divisibility_l2038_203823


namespace division_remainder_problem_l2038_203816

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 →
  L = 1636 →
  L = 6 * S + R →
  R < S →
  R = 10 := by
sorry

end division_remainder_problem_l2038_203816


namespace second_friend_shells_l2038_203882

theorem second_friend_shells (jovana_initial : ℕ) (first_friend : ℕ) (total : ℕ) : 
  jovana_initial = 5 → first_friend = 15 → total = 37 → 
  total - (jovana_initial + first_friend) = 17 := by
sorry

end second_friend_shells_l2038_203882


namespace geometric_sum_seven_halves_l2038_203884

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_seven_halves :
  geometric_sum (1/2) (1/2) 7 = 127/128 := by
  sorry

end geometric_sum_seven_halves_l2038_203884


namespace remainder_2984_times_3998_mod_1000_l2038_203855

theorem remainder_2984_times_3998_mod_1000 : (2984 * 3998) % 1000 = 32 := by
  sorry

end remainder_2984_times_3998_mod_1000_l2038_203855


namespace geometric_sequence_sum_l2038_203848

/-- A geometric sequence of positive real numbers -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 4) →
  (a 5 + a 6 = 16) :=
by sorry

end geometric_sequence_sum_l2038_203848


namespace soap_brands_survey_l2038_203800

/-- The number of households that use both brands of soap -/
def households_both_brands : ℕ := 30

/-- The total number of households surveyed -/
def total_households : ℕ := 260

/-- The number of households that use neither brand A nor brand B -/
def households_neither_brand : ℕ := 80

/-- The number of households that use only brand A -/
def households_only_A : ℕ := 60

theorem soap_brands_survey :
  households_both_brands = 30 ∧
  total_households = households_neither_brand + households_only_A + households_both_brands + 3 * households_both_brands :=
by sorry

end soap_brands_survey_l2038_203800


namespace point_on_line_equidistant_from_axes_in_first_quadrant_l2038_203834

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

-- Define the condition for a point being equidistant from coordinate axes
def equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

-- Define the condition for a point being in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem point_on_line_equidistant_from_axes_in_first_quadrant :
  ∃ (x y : ℝ), line_equation x y ∧ equidistant_from_axes x y ∧ in_first_quadrant x y ∧
  (∀ (x' y' : ℝ), line_equation x' y' ∧ equidistant_from_axes x' y' → in_first_quadrant x' y') :=
sorry

end point_on_line_equidistant_from_axes_in_first_quadrant_l2038_203834


namespace matrix_equation_solution_l2038_203846

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 4]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -43/14, 53/14]
  N * A = B := by sorry

end matrix_equation_solution_l2038_203846


namespace equation_solution_l2038_203898

theorem equation_solution (x y : ℝ) : 
  y = 3 * x → 
  (4 * y^2 - 3 * y + 5 = 3 * (8 * x^2 - 3 * y + 1)) ↔ 
  (x = (Real.sqrt 19 - 3) / 4 ∨ x = (-Real.sqrt 19 - 3) / 4) :=
by sorry

end equation_solution_l2038_203898


namespace optimal_price_maximizes_revenue_l2038_203822

/-- Revenue function for book sales -/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The optimal price maximizes revenue -/
theorem optimal_price_maximizes_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue p ≥ revenue q ∧
  p = 19 := by
  sorry

end optimal_price_maximizes_revenue_l2038_203822


namespace distance_to_point_one_zero_l2038_203896

theorem distance_to_point_one_zero (z : ℂ) (h : z * (1 + Complex.I) = 4) :
  Complex.abs (z - 1) = Real.sqrt 5 := by
  sorry

end distance_to_point_one_zero_l2038_203896


namespace urn_probability_theorem_l2038_203835

/-- Represents the probability of drawing specific colored balls from an urn --/
def draw_probability (red white green total : ℕ) : ℚ :=
  (red : ℚ) / total * (white : ℚ) / (total - 1) * (green : ℚ) / (total - 2)

/-- Represents the probability of drawing specific colored balls in any order --/
def draw_probability_any_order (red white green total : ℕ) : ℚ :=
  6 * draw_probability red white green total

theorem urn_probability_theorem (red white green : ℕ) 
  (h_red : red = 15) (h_white : white = 9) (h_green : green = 4) :
  let total := red + white + green
  draw_probability red white green total = 5 / 182 ∧
  draw_probability_any_order red white green total = 15 / 91 := by
  sorry


end urn_probability_theorem_l2038_203835


namespace tan_BAC_equals_three_fourths_l2038_203830

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D and E on sides AB and AC
structure TriangleWithDE extends Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (D_on_AB : D.1 = A.1 + t * (B.1 - A.1) ∧ D.2 = A.2 + t * (B.2 - A.2)) 
  (E_on_AC : E.1 = A.1 + s * (C.1 - A.1) ∧ E.2 = A.2 + s * (C.2 - A.2))
  (t s : ℝ)
  (t_range : 0 < t ∧ t < 1)
  (s_range : 0 < s ∧ s < 1)

-- Define the area of triangle ADE
def area_ADE (t : TriangleWithDE) : ℝ := sorry

-- Define the incircle of quadrilateral BDEC
structure Incircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point K where the incircle touches AB
def point_K (t : TriangleWithDE) (i : Incircle) : ℝ × ℝ := sorry

-- Define the function to calculate tan(BAC)
def tan_BAC (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem tan_BAC_equals_three_fourths 
  (t : TriangleWithDE) 
  (i : Incircle) 
  (h1 : area_ADE t = 0.5)
  (h2 : point_K t i = (t.A.1 + 3, t.A.2))
  (h3 : (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2 = 15^2)
  (h4 : ∃ (center : ℝ × ℝ) (radius : ℝ), 
        (t.B.1 - center.1)^2 + (t.B.2 - center.2)^2 = radius^2 ∧
        (t.D.1 - center.1)^2 + (t.D.2 - center.2)^2 = radius^2 ∧
        (t.E.1 - center.1)^2 + (t.E.2 - center.2)^2 = radius^2 ∧
        (t.C.1 - center.1)^2 + (t.C.2 - center.2)^2 = radius^2) :
  tan_BAC t.toTriangle = 3/4 := by sorry

end tan_BAC_equals_three_fourths_l2038_203830


namespace rectangle_area_theorem_l2038_203827

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The original rectangle -/
def original : Rectangle := { width := 5, height := 7 }

/-- The rectangle after shortening one side by 2 inches -/
def shortened : Rectangle := { width := 3, height := 7 }

/-- The rectangle after shortening the other side by 1 inch -/
def other_shortened : Rectangle := { width := 5, height := 6 }

theorem rectangle_area_theorem :
  original.area = 35 ∧
  shortened.area = 21 →
  other_shortened.area = 30 := by
  sorry

end rectangle_area_theorem_l2038_203827


namespace book_arrangement_count_l2038_203857

-- Define the number of physics and history books
def num_physics_books : ℕ := 4
def num_history_books : ℕ := 6

-- Define the function to calculate the number of arrangements
def num_arrangements (p h : ℕ) : ℕ :=
  2 * (Nat.factorial p) * (Nat.factorial h)

-- Theorem statement
theorem book_arrangement_count :
  num_arrangements num_physics_books num_history_books = 34560 := by
  sorry

end book_arrangement_count_l2038_203857


namespace baseball_cards_cost_theorem_l2038_203802

/-- The cost of a baseball card deck given the total spent and the cost of Digimon card packs -/
def baseball_card_cost (total_spent : ℝ) (digimon_pack_cost : ℝ) (num_digimon_packs : ℕ) : ℝ :=
  total_spent - (digimon_pack_cost * num_digimon_packs)

/-- Theorem: The cost of the baseball cards is $6.06 -/
theorem baseball_cards_cost_theorem (total_spent : ℝ) (digimon_pack_cost : ℝ) (num_digimon_packs : ℕ) :
  total_spent = 23.86 ∧ digimon_pack_cost = 4.45 ∧ num_digimon_packs = 4 →
  baseball_card_cost total_spent digimon_pack_cost num_digimon_packs = 6.06 := by
  sorry

#eval baseball_card_cost 23.86 4.45 4

end baseball_cards_cost_theorem_l2038_203802


namespace only_real_number_line_bijection_is_correct_l2038_203871

-- Define the property of having a square root
def has_square_root (x : ℝ) : Prop := ∃ y : ℝ, y * y = x

-- Define the property of being irrational
def is_irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

-- Define the property of cube root being equal to itself
def cube_root_equals_self (x : ℝ) : Prop := x * x * x = x

-- Define the property of having no square root
def has_no_square_root (x : ℝ) : Prop := ¬ (∃ y : ℝ, y * y = x)

-- Define the one-to-one correspondence between real numbers and points on a line
def real_number_line_bijection : Prop := 
  ∃ f : ℝ → ℝ, Function.Bijective f ∧ (∀ x : ℝ, f x = x)

-- Define the property that the difference of two irrationals is irrational
def irrational_diff_is_irrational : Prop := 
  ∀ x y : ℝ, is_irrational x → is_irrational y → is_irrational (x - y)

theorem only_real_number_line_bijection_is_correct : 
  (¬ (∀ x : ℝ, has_square_root x → is_irrational x)) ∧
  (¬ (∀ x : ℝ, cube_root_equals_self x → (x = 0 ∨ x = 1))) ∧
  (¬ (∀ a : ℝ, has_no_square_root (-a))) ∧
  real_number_line_bijection ∧
  (¬ irrational_diff_is_irrational) :=
by sorry

end only_real_number_line_bijection_is_correct_l2038_203871


namespace amy_school_year_hours_l2038_203843

/-- Calculates the number of hours Amy needs to work per week during the school year --/
def school_year_hours_per_week (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℚ :=
  let summer_total_hours := summer_hours_per_week * summer_weeks
  let hourly_rate := summer_earnings / summer_total_hours
  let school_year_total_hours := school_year_earnings / hourly_rate
  school_year_total_hours / school_year_weeks

/-- Theorem stating that Amy needs to work 9 hours per week during the school year --/
theorem amy_school_year_hours : 
  school_year_hours_per_week 36 10 3000 40 3000 = 9 := by
  sorry

end amy_school_year_hours_l2038_203843


namespace additive_implies_linear_l2038_203845

/-- A function satisfying the given additive property -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A linear function with zero intercept -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x, f x = k * x

/-- If a function satisfies the additive property, then it is a linear function with zero intercept -/
theorem additive_implies_linear (f : ℝ → ℝ) (h : AdditiveFunction f) : LinearFunction f := by
  sorry

end additive_implies_linear_l2038_203845


namespace mult_41_equivalence_l2038_203863

theorem mult_41_equivalence (x y : ℤ) :
  (25 * x + 31 * y) % 41 = 0 ↔ (3 * x + 7 * y) % 41 = 0 := by
  sorry

end mult_41_equivalence_l2038_203863


namespace largest_angle_in_specific_triangle_l2038_203842

/-- The largest angle in a triangle with sides 3√2, 6, and 3√10 is 135° --/
theorem largest_angle_in_specific_triangle : 
  ∀ (a b c : ℝ) (θ : ℝ),
  a = 3 * Real.sqrt 2 →
  b = 6 →
  c = 3 * Real.sqrt 10 →
  c > a ∧ c > b →
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
  θ = 135 * (π / 180) := by
sorry

end largest_angle_in_specific_triangle_l2038_203842


namespace sugar_calculation_l2038_203810

-- Define the original amount of sugar in the recipe
def original_sugar : ℚ := 5 + 3/4

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/3

-- Define the result we want to prove
def result : ℚ := 1 + 11/12

-- Theorem statement
theorem sugar_calculation : 
  recipe_fraction * original_sugar = result := by
  sorry

end sugar_calculation_l2038_203810


namespace revenue_decrease_percent_l2038_203852

theorem revenue_decrease_percent (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let R := T * C
  let T_new := T * (1 - 0.20)
  let C_new := C * (1 + 0.15)
  let R_new := T_new * C_new
  (R - R_new) / R * 100 = 8 := by
sorry

end revenue_decrease_percent_l2038_203852


namespace ages_proof_l2038_203880

/-- Represents the current age of Grant -/
def grant_age : ℕ := 25

/-- Represents the current age of the hospital -/
def hospital_age : ℕ := 40

/-- Represents the current age of the university -/
def university_age : ℕ := 30

/-- Represents the current age of the town library -/
def town_library_age : ℕ := 50

theorem ages_proof :
  (grant_age + 5 = (2 * (hospital_age + 5)) / 3) ∧
  (university_age = hospital_age - 10) ∧
  (town_library_age = university_age + 20) ∧
  (hospital_age < town_library_age) :=
by sorry

end ages_proof_l2038_203880


namespace journey_distance_l2038_203825

/-- A journey with two parts -/
structure Journey where
  total_time : ℝ
  speed1 : ℝ
  time1 : ℝ
  speed2 : ℝ

/-- Calculate the total distance of a journey -/
def total_distance (j : Journey) : ℝ :=
  j.speed1 * j.time1 + j.speed2 * (j.total_time - j.time1)

/-- Theorem: The total distance of the given journey is 240 km -/
theorem journey_distance :
  ∃ (j : Journey),
    j.total_time = 5 ∧
    j.speed1 = 40 ∧
    j.time1 = 3 ∧
    j.speed2 = 60 ∧
    total_distance j = 240 :=
by
  sorry

end journey_distance_l2038_203825


namespace student_failed_marks_l2038_203836

theorem student_failed_marks (total_marks : ℕ) (passing_percentage : ℚ) (student_score : ℕ) : 
  total_marks = 600 → 
  passing_percentage = 33 / 100 → 
  student_score = 125 → 
  (total_marks * passing_percentage).floor - student_score = 73 := by
  sorry

end student_failed_marks_l2038_203836


namespace base_3_10201_equals_100_l2038_203872

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_3_10201_equals_100 :
  base_3_to_10 [1, 0, 2, 0, 1] = 100 := by
  sorry

end base_3_10201_equals_100_l2038_203872


namespace lisas_age_2005_l2038_203819

theorem lisas_age_2005 (lisa_age_2000 grandfather_age_2000 : ℕ) 
  (h1 : lisa_age_2000 * 2 = grandfather_age_2000)
  (h2 : (2000 - lisa_age_2000) + (2000 - grandfather_age_2000) = 3904) :
  lisa_age_2000 + 5 = 37 := by
  sorry

end lisas_age_2005_l2038_203819


namespace twelfth_term_is_twelve_l2038_203862

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  -8 + 2 * (n - 2)

/-- Theorem: The 12th term of the arithmetic sequence is 12 -/
theorem twelfth_term_is_twelve : arithmetic_sequence 12 = 12 := by
  sorry

end twelfth_term_is_twelve_l2038_203862


namespace quadratic_inequality_solution_interval_l2038_203881

theorem quadratic_inequality_solution_interval (k : ℝ) : 
  (k > 0 ∧ ∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (0 < k ∧ k < 16) := by
  sorry

end quadratic_inequality_solution_interval_l2038_203881


namespace circle_plus_five_two_l2038_203811

/-- The custom binary operation ⊕ -/
def circle_plus (x y : ℝ) : ℝ := (x + y + 1) * (x - y)

/-- Theorem stating that 5 ⊕ 2 = 24 -/
theorem circle_plus_five_two : circle_plus 5 2 = 24 := by
  sorry

end circle_plus_five_two_l2038_203811


namespace fraction_to_decimal_l2038_203856

theorem fraction_to_decimal : (22 : ℚ) / 160 = (1375 : ℚ) / 10000 := by sorry

end fraction_to_decimal_l2038_203856


namespace find_other_number_l2038_203814

theorem find_other_number (x y : ℤ) (h1 : 4 * x + 3 * y = 154) (h2 : x = 14 ∨ y = 14) : x = 28 ∨ y = 28 := by
  sorry

end find_other_number_l2038_203814


namespace total_matches_proof_l2038_203885

/-- Represents the number of matches for a team -/
structure MatchRecord where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Calculate the total number of matches played by a team -/
def totalMatches (record : MatchRecord) : ℕ :=
  record.wins + record.draws + record.losses

theorem total_matches_proof
  (home : MatchRecord)
  (rival : MatchRecord)
  (h1 : rival.wins = 2 * home.wins)
  (h2 : home.wins = 3)
  (h3 : home.draws = 4)
  (h4 : rival.draws = 4)
  (h5 : home.losses = 0)
  (h6 : rival.losses = 0) :
  totalMatches home + totalMatches rival = 17 := by
  sorry

end total_matches_proof_l2038_203885


namespace locker_labeling_cost_l2038_203840

/-- Calculates the cost of labeling lockers given the number of lockers and cost per digit -/
def labelingCost (numLockers : ℕ) (costPerDigit : ℚ) : ℚ :=
  let singleDigitCost := (min numLockers 9 : ℕ) * costPerDigit
  let doubleDigitCost := (min (numLockers - 9) 90 : ℕ) * 2 * costPerDigit
  let tripleDigitCost := (min (numLockers - 99) 900 : ℕ) * 3 * costPerDigit
  let quadrupleDigitCost := (max (numLockers - 999) 0 : ℕ) * 4 * costPerDigit
  singleDigitCost + doubleDigitCost + tripleDigitCost + quadrupleDigitCost

theorem locker_labeling_cost :
  labelingCost 2999 (3 / 100) = 32667 / 100 :=
by sorry

end locker_labeling_cost_l2038_203840


namespace jason_newspaper_earnings_l2038_203838

/-- Proves that Jason's earnings from delivering newspapers equals $1.875 --/
theorem jason_newspaper_earnings 
  (fred_initial : ℝ) 
  (jason_initial : ℝ) 
  (emily_initial : ℝ) 
  (fred_increase : ℝ) 
  (jason_increase : ℝ) 
  (emily_increase : ℝ) 
  (h1 : fred_initial = 49) 
  (h2 : jason_initial = 3) 
  (h3 : emily_initial = 25) 
  (h4 : fred_increase = 1.5) 
  (h5 : jason_increase = 1.625) 
  (h6 : emily_increase = 1.4) :
  jason_initial * (jason_increase - 1) = 1.875 := by
  sorry

end jason_newspaper_earnings_l2038_203838


namespace even_function_product_nonnegative_l2038_203867

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_product_nonnegative
  (f : ℝ → ℝ) (h : is_even_function f) :
  ∀ x : ℝ, f x * f (-x) ≥ 0 :=
by
  sorry

end even_function_product_nonnegative_l2038_203867


namespace all_odd_in_M_product_in_M_l2038_203868

-- Define the set M
def M : Set ℤ := {n : ℤ | ∃ (x y : ℤ), n = x^2 - y^2}

-- Statement 1: All odd numbers belong to M
theorem all_odd_in_M : ∀ (k : ℤ), (2 * k + 1) ∈ M := by sorry

-- Statement 3: If a ∈ M and b ∈ M, then ab ∈ M
theorem product_in_M : ∀ (a b : ℤ), a ∈ M → b ∈ M → (a * b) ∈ M := by sorry

end all_odd_in_M_product_in_M_l2038_203868


namespace square_figure_perimeter_l2038_203879

/-- A figure composed of four identical squares with a specific arrangement -/
structure SquareFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The number of squares in the figure -/
  num_squares : ℕ
  /-- The number of exposed sides in the figure's perimeter -/
  exposed_sides : ℕ
  /-- Assertion that the figure is composed of four squares -/
  h_four_squares : num_squares = 4
  /-- Assertion that the total area is 144 cm² -/
  h_total_area : total_area = 144
  /-- Assertion that the exposed sides count is 9 based on the specific arrangement -/
  h_exposed_sides : exposed_sides = 9
  /-- Assertion that the total area is the sum of the areas of individual squares -/
  h_area_sum : total_area = num_squares * square_side ^ 2

/-- The perimeter of the SquareFigure -/
def perimeter (f : SquareFigure) : ℝ :=
  f.exposed_sides * f.square_side

/-- Theorem stating that the perimeter of the SquareFigure is 54 cm -/
theorem square_figure_perimeter (f : SquareFigure) : perimeter f = 54 := by
  sorry

end square_figure_perimeter_l2038_203879


namespace carla_daily_collection_l2038_203889

/-- The number of items Carla needs to collect each day -/
def daily_items (total_leaves total_bugs total_days : ℕ) : ℕ :=
  (total_leaves + total_bugs) / total_days

/-- Proof that Carla needs to collect 5 items per day -/
theorem carla_daily_collection :
  daily_items 30 20 10 = 5 :=
by sorry

end carla_daily_collection_l2038_203889


namespace western_rattlesnake_segments_l2038_203892

/-- The number of segments in Eastern rattlesnakes' tails -/
def eastern_segments : ℕ := 6

/-- The percentage difference in tail size as a fraction -/
def percentage_difference : ℚ := 1/4

/-- The number of segments in Western rattlesnakes' tails -/
def western_segments : ℕ := 8

/-- Theorem stating that the number of segments in Western rattlesnakes' tails is 8,
    given the conditions from the problem -/
theorem western_rattlesnake_segments :
  (western_segments : ℚ) - eastern_segments = percentage_difference * western_segments :=
sorry

end western_rattlesnake_segments_l2038_203892


namespace sum_of_sequences_is_300_l2038_203826

def sequence1 : List ℕ := [2, 13, 24, 35, 46]
def sequence2 : List ℕ := [4, 15, 26, 37, 48]

theorem sum_of_sequences_is_300 : 
  (sequence1.sum + sequence2.sum) = 300 := by
  sorry

end sum_of_sequences_is_300_l2038_203826


namespace exactly_two_true_l2038_203809

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define equilateral triangle
def equilateral (t : Triangle) : Prop := 
  ∀ i : Fin 3, t.angles i = 60

-- Proposition 1
def prop1 : Prop := 
  ∀ t1 t2 : Triangle, t1.area = t2.area → congruent t1 t2

-- Proposition 2
def prop2 : Prop := 
  ∃ a b : ℝ, a * b = 0 ∧ a ≠ 0

-- Proposition 3
def prop3 : Prop := 
  ∀ t : Triangle, ¬equilateral t → ∃ i : Fin 3, t.angles i ≠ 60

-- Main theorem
theorem exactly_two_true : 
  (¬prop1 ∧ prop2 ∧ prop3) ∨
  (prop1 ∧ prop2 ∧ ¬prop3) ∨
  (prop1 ∧ ¬prop2 ∧ prop3) :=
sorry

end exactly_two_true_l2038_203809


namespace meet_once_l2038_203837

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific movement scenario described in the problem --/
def problem_scenario : Movement where
  michael_speed := 6
  truck_speed := 12
  pail_distance := 300
  truck_stop_time := 20
  initial_distance := 300

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once : number_of_meetings problem_scenario = 1 := by
  sorry

end meet_once_l2038_203837


namespace investment_amount_l2038_203821

/-- Proves that given a monthly interest payment of $240 and a simple annual interest rate of 9%,
    the principal amount of the investment is $32,000. -/
theorem investment_amount (monthly_interest : ℝ) (annual_rate : ℝ) (principal : ℝ) :
  monthly_interest = 240 →
  annual_rate = 0.09 →
  principal = monthly_interest / (annual_rate / 12) →
  principal = 32000 := by
  sorry

end investment_amount_l2038_203821


namespace cubic_expression_lower_bound_l2038_203828

theorem cubic_expression_lower_bound (x : ℝ) (h : x^2 - 5*x + 6 > 0) :
  x^3 - 5*x^2 + 6*x + 1 ≥ 1 := by sorry

end cubic_expression_lower_bound_l2038_203828


namespace geometric_sequence_divisibility_l2038_203833

/-- Given a geometric sequence with first term a₁ and second term a₂, 
    find the smallest n for which the n-th term is divisible by 10⁶ -/
theorem geometric_sequence_divisibility
  (a₁ : ℚ)
  (a₂ : ℕ)
  (h₁ : a₁ = 5 / 8)
  (h₂ : a₂ = 25)
  : ∃ n : ℕ, n > 0 ∧ 
    (∀ k < n, ¬(10^6 ∣ (a₁ * (a₂ / a₁)^(k - 1)))) ∧
    (10^6 ∣ (a₁ * (a₂ / a₁)^(n - 1))) ∧
    n = 7 :=
by sorry

end geometric_sequence_divisibility_l2038_203833


namespace distinct_prime_factors_count_l2038_203861

theorem distinct_prime_factors_count (n : ℕ) : n = 95 * 97 * 99 * 101 → Finset.card (Nat.factors n).toFinset = 6 := by
  sorry

end distinct_prime_factors_count_l2038_203861


namespace solve_for_d_l2038_203874

theorem solve_for_d (a c d n : ℝ) (h : n = (c * d * a) / (a - d)) :
  d = (n * a) / (c * d + n) := by
  sorry

end solve_for_d_l2038_203874


namespace tara_megan_money_difference_l2038_203886

/-- The problem of determining how much more money Tara has than Megan. -/
theorem tara_megan_money_difference
  (scooter_cost : ℕ)
  (tara_money : ℕ)
  (megan_money : ℕ)
  (h1 : scooter_cost = 26)
  (h2 : tara_money > megan_money)
  (h3 : tara_money + megan_money = scooter_cost)
  (h4 : tara_money = 15) :
  tara_money - megan_money = 4 := by
  sorry

end tara_megan_money_difference_l2038_203886


namespace polynomial_identity_sum_of_squares_l2038_203897

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ x : ℝ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 := by
  sorry

end polynomial_identity_sum_of_squares_l2038_203897


namespace sqrt_30_between_5_and_6_l2038_203841

theorem sqrt_30_between_5_and_6 : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by
  sorry

end sqrt_30_between_5_and_6_l2038_203841


namespace rowing_distance_with_tide_l2038_203850

/-- Represents the problem of a man rowing with and against the tide. -/
structure RowingProblem where
  /-- The speed of the man rowing in still water (km/h) -/
  manSpeed : ℝ
  /-- The speed of the tide (km/h) -/
  tideSpeed : ℝ
  /-- The distance traveled against the tide (km) -/
  distanceAgainstTide : ℝ
  /-- The time taken to travel against the tide (h) -/
  timeAgainstTide : ℝ
  /-- The time that would have been saved if the tide hadn't changed (h) -/
  timeSaved : ℝ

/-- Theorem stating that given the conditions of the rowing problem, 
    the distance the man can row with the help of the tide in 60 minutes is 5 km. -/
theorem rowing_distance_with_tide (p : RowingProblem) 
  (h1 : p.manSpeed - p.tideSpeed = p.distanceAgainstTide / p.timeAgainstTide)
  (h2 : p.manSpeed + p.tideSpeed = p.distanceAgainstTide / (p.timeAgainstTide - p.timeSaved))
  (h3 : p.distanceAgainstTide = 40)
  (h4 : p.timeAgainstTide = 10)
  (h5 : p.timeSaved = 2) :
  (p.manSpeed + p.tideSpeed) * 1 = 5 := by
  sorry

end rowing_distance_with_tide_l2038_203850


namespace two_numbers_with_specific_means_l2038_203858

theorem two_numbers_with_specific_means :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = 2 * Real.sqrt 3 ∧
  (a + b) / 2 = 6 ∧
  2 / (1 / a + 1 / b) = 2 ∧
  ((a = 6 - 2 * Real.sqrt 6 ∧ b = 6 + 2 * Real.sqrt 6) ∨
   (a = 6 + 2 * Real.sqrt 6 ∧ b = 6 - 2 * Real.sqrt 6)) := by
  sorry

end two_numbers_with_specific_means_l2038_203858


namespace bamboo_with_nine_nodes_l2038_203888

/-- Given a geometric sequence of 9 terms, prove that if the product of the first 3 terms is 3
    and the product of the last 3 terms is 9, then the 5th term is √3. -/
theorem bamboo_with_nine_nodes (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  a 1 * a 2 * a 3 = 3 →             -- Product of first 3 terms
  a 7 * a 8 * a 9 = 9 →             -- Product of last 3 terms
  a 5 = Real.sqrt 3 :=              -- 5th term is √3
by sorry

end bamboo_with_nine_nodes_l2038_203888


namespace table_tennis_pairing_methods_l2038_203849

theorem table_tennis_pairing_methods (total_players : Nat) (male_players : Nat) (female_players : Nat) :
  total_players = male_players + female_players →
  male_players = 5 →
  female_players = 4 →
  (Nat.choose male_players 2) * (Nat.choose female_players 2) * 2 = 120 :=
by sorry

end table_tennis_pairing_methods_l2038_203849


namespace simplify_expression_l2038_203831

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  (a/b + b/a)^2 - 1/(a^2*b^2) = 2/(a*b) := by
  sorry

end simplify_expression_l2038_203831


namespace train_length_is_160_meters_l2038_203812

def train_speed : ℝ := 45 -- km/hr
def crossing_time : ℝ := 30 -- seconds
def bridge_length : ℝ := 215 -- meters

theorem train_length_is_160_meters :
  let speed_mps := train_speed * 1000 / 3600
  let total_distance := speed_mps * crossing_time
  total_distance - bridge_length = 160 := by
  sorry

end train_length_is_160_meters_l2038_203812


namespace triangle_area_is_36_l2038_203807

/-- The area of the triangle bounded by y = x, y = -x, and y = 6 -/
def triangle_area : ℝ := 36

/-- The line y = x -/
def line1 (x : ℝ) : ℝ := x

/-- The line y = -x -/
def line2 (x : ℝ) : ℝ := -x

/-- The line y = 6 -/
def line3 : ℝ := 6

theorem triangle_area_is_36 :
  triangle_area = 36 := by sorry

end triangle_area_is_36_l2038_203807


namespace distinct_prime_factors_of_180_l2038_203887

theorem distinct_prime_factors_of_180 : Nat.card (Nat.factors 180).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_180_l2038_203887


namespace vertical_angles_are_equal_l2038_203899

/-- Two angles are vertical if they are formed by two intersecting lines and are not adjacent. -/
def VerticalAngles (α β : Real) : Prop := sorry

theorem vertical_angles_are_equal (α β : Real) :
  VerticalAngles α β → α = β := by sorry

end vertical_angles_are_equal_l2038_203899


namespace chad_savings_l2038_203818

/-- Represents Chad's financial situation for a year --/
structure ChadFinances where
  savingRate : ℝ
  mowingIncome : ℝ
  birthdayMoney : ℝ
  videoGamesSales : ℝ
  oddJobsIncome : ℝ

/-- Calculates Chad's total savings for the year --/
def totalSavings (cf : ChadFinances) : ℝ :=
  cf.savingRate * (cf.mowingIncome + cf.birthdayMoney + cf.videoGamesSales + cf.oddJobsIncome)

/-- Theorem stating that Chad's savings for the year will be $460 --/
theorem chad_savings :
  ∀ (cf : ChadFinances),
    cf.savingRate = 0.4 ∧
    cf.mowingIncome = 600 ∧
    cf.birthdayMoney = 250 ∧
    cf.videoGamesSales = 150 ∧
    cf.oddJobsIncome = 150 →
    totalSavings cf = 460 :=
by sorry

end chad_savings_l2038_203818


namespace m_range_l2038_203805

-- Define propositions P and Q
def P (m : ℝ) : Prop := |m + 1| ≤ 2
def Q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - m*x + 1 = 0

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, ¬(¬(P m))) →
  (∀ m : ℝ, ¬(P m ∧ Q m)) →
  ∀ m : ℝ, (m > -2 ∧ m ≤ 1) ↔ (P m ∧ ¬(Q m)) :=
sorry

end m_range_l2038_203805


namespace equation_solution_l2038_203832

theorem equation_solution (a : ℝ) (h : a = 0.5) : 
  ∃ x : ℝ, x / (a - 3) = 3 / (a + 2) ∧ x = -3 := by
  sorry

end equation_solution_l2038_203832


namespace sqrt5_diamond_sqrt5_equals_20_l2038_203883

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end sqrt5_diamond_sqrt5_equals_20_l2038_203883


namespace p_plus_q_values_l2038_203870

theorem p_plus_q_values (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 81*p - 162 = 0)
  (hq : 4*q^3 - 24*q^2 + 45*q - 27 = 0) :
  (p + q = 8) ∨ (p + q = 8 + 6*Real.sqrt 3) ∨ (p + q = 8 - 6*Real.sqrt 3) := by
  sorry

end p_plus_q_values_l2038_203870


namespace translation_result_l2038_203803

/-- Represents a 2D point with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Translates a point by given x and y offsets -/
def translate (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_result :
  let initial_point : Point := { x := -5, y := 1 }
  let final_point : Point := translate (translate initial_point 2 0) 0 (-4)
  final_point = { x := -3, y := -3 } := by sorry

end translation_result_l2038_203803


namespace sum_of_distinct_numbers_l2038_203847

theorem sum_of_distinct_numbers (x y u v : ℝ) : 
  x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ y ≠ u ∧ y ≠ v ∧ u ≠ v →
  (x + u) / (x + v) = (y + v) / (y + u) →
  x + y + u + v = 0 := by
sorry

end sum_of_distinct_numbers_l2038_203847


namespace sum_of_complex_roots_of_unity_l2038_203824

open Complex

theorem sum_of_complex_roots_of_unity : 
  let ω : ℂ := exp (Complex.I * Real.pi / 11)
  (ω + ω^3 + ω^5 + ω^7 + ω^9 + ω^11 + ω^13 + ω^15 + ω^17 + ω^19 + ω^21) = 0 := by
  sorry

end sum_of_complex_roots_of_unity_l2038_203824


namespace coronavirus_spread_rate_l2038_203860

/-- The number of people infected after two rounds of novel coronavirus spread -/
def total_infected : ℕ := 121

/-- The number of people initially infected -/
def initial_infected : ℕ := 1

/-- The average number of people infected by one person in each round -/
def m : ℕ := 10

/-- Theorem stating that m = 10 given the conditions of the coronavirus spread -/
theorem coronavirus_spread_rate :
  (initial_infected + m)^2 = total_infected :=
sorry

end coronavirus_spread_rate_l2038_203860


namespace problem_solution_l2038_203839

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end problem_solution_l2038_203839


namespace min_shots_theorem_l2038_203894

/-- Represents a strategy for shooting at windows -/
def ShootingStrategy (n : ℕ) := ℕ → Fin n

/-- Determines if a shooting strategy is successful for all possible target positions -/
def is_successful_strategy (n : ℕ) (strategy : ShootingStrategy n) : Prop :=
  ∀ (start_pos : Fin n), ∃ (k : ℕ), strategy k = min (start_pos + k) (Fin.last n)

/-- The minimum number of shots needed to guarantee hitting the target -/
def min_shots_needed (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem stating the minimum number of shots needed to guarantee hitting the target -/
theorem min_shots_theorem (n : ℕ) : 
  ∃ (strategy : ShootingStrategy n), is_successful_strategy n strategy ∧ 
  (∀ (other_strategy : ShootingStrategy n), 
    is_successful_strategy n other_strategy → 
    (∃ (k : ℕ), ∀ (i : ℕ), i < k → strategy i = other_strategy i) → 
    k ≥ min_shots_needed n) :=
sorry

end min_shots_theorem_l2038_203894


namespace sales_growth_rate_equation_l2038_203808

/-- The average monthly growth rate of a store's sales revenue -/
def average_monthly_growth_rate (march_revenue : ℝ) (may_revenue : ℝ) : ℝ → Prop :=
  λ x => 3 * (1 + x)^2 = 3.63

theorem sales_growth_rate_equation (march_revenue may_revenue : ℝ) 
  (h1 : march_revenue = 30000)
  (h2 : may_revenue = 36300) :
  ∃ x, average_monthly_growth_rate march_revenue may_revenue x :=
by
  sorry

end sales_growth_rate_equation_l2038_203808


namespace hyperbola_eccentricity_l2038_203877

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : a > 0 ∧ b > 0

-- Define the points and conditions
structure HyperbolaIntersection (h : Hyperbola) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_perp : (P.1 - Q.1) * (P.1 - F₁.1) + (P.2 - Q.2) * (P.2 - F₁.2) = 0  -- PQ ⊥ PF₁
  h_equal : (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2  -- |PF₁| = |PQ|
  h_on_hyperbola : P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1 ∧ Q.1^2 / h.a^2 - Q.2^2 / h.b^2 = 1
  h_F₂_on_line : (Q.2 - P.2) * (F₂.1 - P.1) = (Q.1 - P.1) * (F₂.2 - P.2)  -- F₂ is on line PQ

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) (i : HyperbolaIntersection h) :
  h.c / h.a = Real.sqrt (5 - 2 * Real.sqrt 2) :=
sorry

end hyperbola_eccentricity_l2038_203877
