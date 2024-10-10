import Mathlib

namespace polynomial_coefficient_b_l2192_219264

theorem polynomial_coefficient_b (a b c d : ℝ) : 
  (∃ (z w : ℂ), z * w = 9 - 3*I ∧ z + w = -2 - 6*I) →
  (∀ (r : ℂ), r^4 + a*r^3 + b*r^2 + c*r + d = 0 → r.im ≠ 0) →
  b = 58 := by sorry

end polynomial_coefficient_b_l2192_219264


namespace sequence_decreasing_l2192_219254

def x (a : ℝ) (n : ℕ) : ℝ := 2^n * (a^(1/(2*n)) - 1)

theorem sequence_decreasing (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∀ n : ℕ, x a n > x a (n + 1) := by
  sorry

end sequence_decreasing_l2192_219254


namespace molecular_weight_CaCO3_is_100_l2192_219230

/-- The molecular weight of CaCO3 in grams per mole -/
def molecular_weight_CaCO3 : ℝ := 100

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 9

/-- The total molecular weight of the given number of moles in grams -/
def given_total_weight : ℝ := 900

/-- Theorem stating that the molecular weight of CaCO3 is 100 grams/mole -/
theorem molecular_weight_CaCO3_is_100 :
  molecular_weight_CaCO3 = given_total_weight / given_moles :=
by sorry

end molecular_weight_CaCO3_is_100_l2192_219230


namespace cube_root_over_sixth_root_of_eight_l2192_219298

theorem cube_root_over_sixth_root_of_eight (x : ℝ) :
  (8 : ℝ) ^ (1/3) / (8 : ℝ) ^ (1/6) = (8 : ℝ) ^ (1/6) :=
by sorry

end cube_root_over_sixth_root_of_eight_l2192_219298


namespace range_of_a_l2192_219281

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| > 2 → |x| > a) ∧ 
  (∃ x : ℝ, |x| > a ∧ |x + 1| ≤ 2) → 
  a ≤ 1 := by
sorry

end range_of_a_l2192_219281


namespace line_segment_intersection_k_range_l2192_219293

/-- Given points A and B and a line y = kx + 1 that intersects line segment AB, 
    the range of k is [1/2, 1] -/
theorem line_segment_intersection_k_range 
  (A B : ℝ × ℝ) 
  (hA : A = (1, 2)) 
  (hB : B = (2, 1)) 
  (k : ℝ) 
  (h_intersect : ∃ (x y : ℝ), 
    y = k * x + 1 ∧ 
    (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
      x = A.1 + t * (B.1 - A.1) ∧ 
      y = A.2 + t * (B.2 - A.2))) :
  1/2 ≤ k ∧ k ≤ 1 := by sorry

end line_segment_intersection_k_range_l2192_219293


namespace dormitory_students_count_unique_solution_l2192_219272

/-- Represents the number of students in the dormitory -/
def n : ℕ := 6

/-- Represents the number of administrators -/
def m : ℕ := 3

/-- The total number of greeting cards used -/
def total_cards : ℕ := 51

/-- Theorem stating that the number of students in the dormitory is 6 -/
theorem dormitory_students_count :
  (n * (n - 1)) / 2 + n * m + m = total_cards :=
by sorry

/-- Theorem stating that n is the unique solution for the given conditions -/
theorem unique_solution (k : ℕ) :
  (k * (k - 1)) / 2 + k * m + m = total_cards → k = n :=
by sorry

end dormitory_students_count_unique_solution_l2192_219272


namespace sum_of_products_l2192_219224

theorem sum_of_products : 1234 * 2 + 2341 * 2 + 3412 * 2 + 4123 * 2 = 22220 := by
  sorry

end sum_of_products_l2192_219224


namespace semicircle_area_theorem_l2192_219288

noncomputable def semicircle_area (P Q R S T U : Point) : ℝ :=
  let PQ_radius := 2
  let PS_length := Real.sqrt 2
  let QS_length := Real.sqrt 2
  let PT_radius := PQ_radius / 2
  let QU_radius := PQ_radius / 2
  let TU_radius := PS_length
  let triangle_PQS_area := PQ_radius * (Real.sqrt 2) / 2
  (PT_radius^2 * Real.pi / 2) + (QU_radius^2 * Real.pi / 2) + (TU_radius^2 * Real.pi / 2) - triangle_PQS_area

theorem semicircle_area_theorem (P Q R S T U : Point) :
  semicircle_area P Q R S T U = 9 * Real.pi - 2 :=
sorry

end semicircle_area_theorem_l2192_219288


namespace largest_common_term_correct_l2192_219227

/-- First arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 7 * (n + 1)

/-- Second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 8 + 12 * m

/-- Predicate for common terms -/
def isCommonTerm (a : ℕ) : Prop :=
  ∃ n m : ℕ, seq1 n = a ∧ seq2 m = a

/-- The largest common term less than 500 -/
def largestCommonTerm : ℕ := 476

theorem largest_common_term_correct :
  isCommonTerm largestCommonTerm ∧
  largestCommonTerm < 500 ∧
  ∀ x : ℕ, isCommonTerm x → x < 500 → x ≤ largestCommonTerm :=
by sorry

end largest_common_term_correct_l2192_219227


namespace sufficient_condition_for_inequality_l2192_219245

theorem sufficient_condition_for_inequality (x : ℝ) : 
  1 < x ∧ x < 2 → (x + 1) / (x - 1) > 2 := by
  sorry

end sufficient_condition_for_inequality_l2192_219245


namespace baker_cake_difference_l2192_219232

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 47. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ) 
    (h1 : initial = 170) (h2 : sold = 78) (h3 : bought = 31) : 
    sold - bought = 47 := by
  sorry

end baker_cake_difference_l2192_219232


namespace geometric_sequence_a3_l2192_219246

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 4 - a 2 = 6 →
  a 5 - a 1 = 15 →
  a 3 = 4 ∨ a 3 = -4 := by
  sorry

end geometric_sequence_a3_l2192_219246


namespace two_solutions_l2192_219256

-- Define the equation
def equation (x a : ℝ) : Prop := abs (x - 3) = a * x - 1

-- Define the condition for two solutions
theorem two_solutions (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ a ∧ equation x₂ a) ↔ a > 1/3 := by
  sorry

end two_solutions_l2192_219256


namespace max_area_difference_l2192_219218

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 80

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem statement -/
theorem max_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    (∀ r : Rectangle, area r ≤ area r1 ∧ area r ≥ area r2) ∧
    area r1 - area r2 = 1521 ∧
    r1.length = 40 ∧ r1.width = 40 ∧
    r2.length = 1 ∧ r2.width = 79 := by
  sorry


end max_area_difference_l2192_219218


namespace slope_range_for_intersection_l2192_219216

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/20) ∨ m ≥ Real.sqrt (1/20)}

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 + 25 * y^2 = 100

/-- The line equation with slope m and y-intercept 3 -/
def line_equation (m x : ℝ) : ℝ :=
  m * x + 3

theorem slope_range_for_intersection :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_equation x (line_equation m x)) ↔ m ∈ possible_slopes :=
by sorry

end slope_range_for_intersection_l2192_219216


namespace intersection_M_N_l2192_219217

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | 2 * x - 1 > 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end intersection_M_N_l2192_219217


namespace profit_difference_l2192_219262

def business_problem (capital_A capital_B capital_C capital_D capital_E profit_B : ℕ) : Prop :=
  let total_capital := capital_A + capital_B + capital_C + capital_D + capital_E
  let total_profit := profit_B * total_capital / capital_B
  let profit_C := total_profit * capital_C / total_capital
  let profit_E := total_profit * capital_E / total_capital
  profit_E - profit_C = 900

theorem profit_difference :
  business_problem 8000 10000 12000 15000 18000 1500 := by sorry

end profit_difference_l2192_219262


namespace root_implies_m_value_l2192_219249

theorem root_implies_m_value (m : ℝ) : 
  (Complex.I + 1)^2 + m * (Complex.I + 1) + 2 = 0 → m = -2 := by
  sorry

end root_implies_m_value_l2192_219249


namespace trapezoid_x_squared_l2192_219241

/-- A trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  x : ℝ
  shorter_base_length : shorter_base = 50
  longer_base_length : longer_base = shorter_base + 50
  midpoint_ratio : (shorter_base + (shorter_base + longer_base) / 2) / ((shorter_base + longer_base) / 2 + longer_base) = 1 / 2
  equal_area : x > shorter_base ∧ x < longer_base ∧ 
    (x - shorter_base) / (longer_base - shorter_base) = 
    (x - shorter_base) * (x + shorter_base) / ((longer_base - shorter_base) * (longer_base + shorter_base))

theorem trapezoid_x_squared (t : Trapezoid) : t.x^2 = 6875 := by
  sorry

end trapezoid_x_squared_l2192_219241


namespace helpers_count_l2192_219297

theorem helpers_count (pouches_per_pack : ℕ) (team_members : ℕ) (coaches : ℕ) (packs_bought : ℕ) :
  pouches_per_pack = 6 →
  team_members = 13 →
  coaches = 3 →
  packs_bought = 3 →
  (pouches_per_pack * packs_bought) - (team_members + coaches) = 2 :=
by
  sorry

end helpers_count_l2192_219297


namespace base_4_last_digit_l2192_219286

theorem base_4_last_digit (n : ℕ) (h : n = 389) : n % 4 = 1 := by
  sorry

end base_4_last_digit_l2192_219286


namespace D_72_eq_45_l2192_219265

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where order matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- Theorem stating that D(72) is equal to 45 -/
theorem D_72_eq_45 : D 72 = 45 := by sorry

end D_72_eq_45_l2192_219265


namespace speed_conversion_equivalence_l2192_219296

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def given_speed_mps : ℝ := 35.0028

/-- The calculated speed in kilometers per hour -/
def calculated_speed_kmph : ℝ := 126.01008

theorem speed_conversion_equivalence : 
  given_speed_mps * mps_to_kmph = calculated_speed_kmph := by
  sorry

end speed_conversion_equivalence_l2192_219296


namespace man_lot_ownership_l2192_219289

theorem man_lot_ownership (lot_value : ℝ) (sold_fraction : ℝ) (sold_value : ℝ) :
  lot_value = 9200 →
  sold_fraction = 1 / 10 →
  sold_value = 460 →
  (sold_value / sold_fraction) / lot_value = 1 / 2 := by
  sorry

end man_lot_ownership_l2192_219289


namespace bacteria_population_correct_l2192_219223

def bacteria_population (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n/2 + 1)
  else
    2^((n+1)/2)

theorem bacteria_population_correct :
  ∀ n : ℕ,
  (bacteria_population n = 2^(n/2 + 1) ∧ n % 2 = 0) ∨
  (bacteria_population n = 2^((n+1)/2) ∧ n % 2 ≠ 0) :=
by sorry

end bacteria_population_correct_l2192_219223


namespace red_other_side_probability_l2192_219206

structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

def total_cards : ℕ := 9
def black_both_sides : ℕ := 4
def black_red : ℕ := 2
def red_both_sides : ℕ := 3

def is_red (side : Bool) : Prop := side = true

theorem red_other_side_probability :
  let cards : List Card := 
    (List.replicate black_both_sides ⟨false, false⟩) ++
    (List.replicate black_red ⟨false, true⟩) ++
    (List.replicate red_both_sides ⟨true, true⟩)
  let total_red_sides := red_both_sides * 2 + black_red
  let red_both_sides_count := red_both_sides * 2
  (red_both_sides_count : ℚ) / total_red_sides = 3 / 4 :=
by sorry

end red_other_side_probability_l2192_219206


namespace pencils_given_to_dorothy_l2192_219201

/-- Given that Josh had a certain number of pencils initially and was left with
    a smaller number after giving some to Dorothy, prove that the number of
    pencils he gave to Dorothy is the difference between the initial and final amounts. -/
theorem pencils_given_to_dorothy
  (initial_pencils : ℕ)
  (remaining_pencils : ℕ)
  (h1 : initial_pencils = 142)
  (h2 : remaining_pencils = 111)
  (h3 : remaining_pencils < initial_pencils) :
  initial_pencils - remaining_pencils = 31 :=
by sorry

end pencils_given_to_dorothy_l2192_219201


namespace fraction_sum_squared_l2192_219252

theorem fraction_sum_squared (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end fraction_sum_squared_l2192_219252


namespace cone_slant_height_is_10_l2192_219290

/-- The slant height of a cone, given its base radius and that its lateral surface unfolds into a semicircle. -/
def slant_height (base_radius : ℝ) : ℝ :=
  2 * base_radius

theorem cone_slant_height_is_10 :
  let base_radius : ℝ := 5
  slant_height base_radius = 10 :=
by sorry

end cone_slant_height_is_10_l2192_219290


namespace multiply_decimals_l2192_219285

theorem multiply_decimals : 3.6 * 0.3 = 1.08 := by
  sorry

end multiply_decimals_l2192_219285


namespace polynomial_factorization_l2192_219251

theorem polynomial_factorization (x : ℤ) : 
  x^15 + x^8 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^8 - x^7 + x^6 - x + 1) := by
  sorry

end polynomial_factorization_l2192_219251


namespace johns_memory_card_cost_l2192_219294

/-- Calculates the total cost of memory cards for John's photography habit -/
theorem johns_memory_card_cost :
  let pictures_per_day : ℕ := 25
  let years : ℕ := 6
  let days_per_year : ℕ := 365
  let images_per_card : ℕ := 40
  let cost_per_card : ℕ := 75
  let total_pictures : ℕ := pictures_per_day * years * days_per_year
  let cards_needed : ℕ := (total_pictures + images_per_card - 1) / images_per_card
  cards_needed * cost_per_card = 102675 :=
by
  sorry


end johns_memory_card_cost_l2192_219294


namespace completing_square_transformation_l2192_219280

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 8*x - 1 = 0) ↔ ((x - 4)^2 = 17) :=
by sorry

end completing_square_transformation_l2192_219280


namespace expansion_properties_l2192_219283

def binomial_sum (n : ℕ) : ℕ := 2^n

theorem expansion_properties (x : ℝ) :
  let n : ℕ := 8
  let binomial_sum_diff : ℕ := 128
  let largest_coeff_term : ℝ := 70 * x^4
  let x_power_7_term : ℝ := -56 * x^7
  (binomial_sum n - binomial_sum 7 = binomial_sum_diff) ∧
  (∀ k, 0 ≤ k ∧ k ≤ n → |(-1)^k * (n.choose k) * x^(2*n - 3*k)| ≤ |largest_coeff_term|) ∧
  ((-1)^3 * (n.choose 3) * x^(2*n - 3*3) = x_power_7_term) :=
by sorry

end expansion_properties_l2192_219283


namespace student_mistake_difference_l2192_219222

theorem student_mistake_difference : 
  let number := 384
  let correct_fraction := 5 / 16
  let incorrect_fraction := 5 / 6
  let correct_answer := correct_fraction * number
  let incorrect_answer := incorrect_fraction * number
  incorrect_answer - correct_answer = 200 := by
sorry

end student_mistake_difference_l2192_219222


namespace inequality_proof_l2192_219266

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end inequality_proof_l2192_219266


namespace distance_to_focus_l2192_219250

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def P (y : ℝ) : ℝ × ℝ := (4, y)

-- Theorem statement
theorem distance_to_focus (y : ℝ) (h : parabola 4 y) : 
  Real.sqrt ((P y).1 - focus.1)^2 + ((P y).2 - focus.2)^2 = 5 := by sorry

end distance_to_focus_l2192_219250


namespace tiles_for_18_24_room_l2192_219277

/-- Calculates the number of tiles needed for a rectangular room with a double border --/
def tilesNeeded (length width : ℕ) : ℕ :=
  let borderTiles := 2 * (length - 2) + 2 * (length - 4) + 2 * (width - 2) + 2 * (width - 4) + 8
  let innerLength := length - 4
  let innerWidth := width - 4
  let innerArea := innerLength * innerWidth
  let innerTiles := (innerArea + 8) / 9  -- Ceiling division
  borderTiles + innerTiles

/-- The theorem states that for an 18 by 24 foot room, 183 tiles are needed --/
theorem tiles_for_18_24_room : tilesNeeded 24 18 = 183 := by
  sorry

end tiles_for_18_24_room_l2192_219277


namespace factory_production_constraints_l2192_219239

/-- Given a factory producing two products A and B, this theorem states the constraint
conditions for maximizing the total monthly profit. -/
theorem factory_production_constraints
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℝ)
  (x y : ℝ) -- Monthly production of products A and B in kg
  (h_pos_a₁ : a₁ > 0) (h_pos_a₂ : a₂ > 0)
  (h_pos_b₁ : b₁ > 0) (h_pos_b₂ : b₂ > 0)
  (h_pos_c₁ : c₁ > 0) (h_pos_c₂ : c₂ > 0)
  (h_pos_d₁ : d₁ > 0) (h_pos_d₂ : d₂ > 0) :
  (∃ z : ℝ, z = d₁ * x + d₂ * y ∧ -- Total monthly profit
    a₁ * x + a₂ * y ≤ c₁ ∧       -- Constraint on raw material A
    b₁ * x + b₂ * y ≤ c₂ ∧       -- Constraint on raw material B
    x ≥ 0 ∧ y ≥ 0) →             -- Non-negative production constraints
  (a₁ * x + a₂ * y ≤ c₁ ∧
   b₁ * x + b₂ * y ≤ c₂ ∧
   x ≥ 0 ∧ y ≥ 0) :=
by sorry

end factory_production_constraints_l2192_219239


namespace divisibility_by_six_l2192_219221

theorem divisibility_by_six (n : ℕ) 
  (div_by_two : ∃ k : ℕ, n = 2 * k) 
  (div_by_three : ∃ m : ℕ, n = 3 * m) : 
  ∃ p : ℕ, n = 6 * p := by
sorry

end divisibility_by_six_l2192_219221


namespace no_solution_for_equation_l2192_219270

theorem no_solution_for_equation : ¬∃ (x : ℝ), (x - 1) / (x - 3) = 2 - 2 / (3 - x) := by
  sorry

end no_solution_for_equation_l2192_219270


namespace triangle_side_length_l2192_219287

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : Real.cos (A - 2*B) + Real.sin (2*A + B) = 2)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : BC = 6) :
  AB = 3 * (Real.sqrt 5 + 1) :=
sorry

end triangle_side_length_l2192_219287


namespace complex_number_sum_parts_l2192_219211

theorem complex_number_sum_parts (a : ℝ) : 
  let z : ℂ := a / (2 - Complex.I) + (3 - 4 * Complex.I) / 5
  (z.re + z.im = 1) → a = 2 := by
  sorry

end complex_number_sum_parts_l2192_219211


namespace log_ride_cost_l2192_219231

def ferris_wheel_cost : ℕ := 6
def roller_coaster_cost : ℕ := 5
def initial_tickets : ℕ := 2
def additional_tickets_needed : ℕ := 16

theorem log_ride_cost :
  ferris_wheel_cost + roller_coaster_cost + (additional_tickets_needed + initial_tickets - ferris_wheel_cost - roller_coaster_cost) = additional_tickets_needed + initial_tickets :=
by sorry

end log_ride_cost_l2192_219231


namespace shadow_length_l2192_219243

/-- Given two similar right triangles, if one has height 2 and base 4,
    and the other has height 2.5, then the base of the second triangle is 5. -/
theorem shadow_length (h1 h2 b1 b2 : ℝ) : 
  h1 = 2 → h2 = 2.5 → b1 = 4 → h1 / b1 = h2 / b2 → b2 = 5 := by
  sorry

end shadow_length_l2192_219243


namespace total_travel_ways_problem_solution_l2192_219214

/-- Represents the number of transportation options between two cities -/
structure TransportOptions where
  buses : Nat
  trains : Nat
  ferries : Nat

/-- Calculates the total number of ways to travel between two cities -/
def totalWays (options : TransportOptions) : Nat :=
  options.buses + options.trains + options.ferries

/-- Theorem: The total number of ways to travel from A to C via B is the product
    of the number of ways to travel from A to B and from B to C -/
theorem total_travel_ways
  (optionsAB : TransportOptions)
  (optionsBC : TransportOptions) :
  totalWays optionsAB * totalWays optionsBC =
  (optionsAB.buses + optionsAB.trains) * (optionsBC.buses + optionsBC.ferries) :=
by sorry

/-- Given the specific transportation options in the problem -/
def morningOptions : TransportOptions :=
  { buses := 5, trains := 2, ferries := 0 }

def afternoonOptions : TransportOptions :=
  { buses := 3, trains := 0, ferries := 2 }

/-- The main theorem that proves the total number of ways for the specific problem -/
theorem problem_solution :
  totalWays morningOptions * totalWays afternoonOptions = 35 :=
by sorry

end total_travel_ways_problem_solution_l2192_219214


namespace min_value_of_f_range_of_x_when_f_leq_5_l2192_219244

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 4| + |x - 1|

-- Theorem for the minimum value of f(x)
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 :=
sorry

-- Theorem for the range of x when f(x) ≤ 5
theorem range_of_x_when_f_leq_5 :
  ∀ x, f x ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5 :=
sorry

end min_value_of_f_range_of_x_when_f_leq_5_l2192_219244


namespace range_of_f_on_interval_l2192_219278

noncomputable def f (k : ℝ) (c : ℝ) (x : ℝ) : ℝ := x^k + c

theorem range_of_f_on_interval (k : ℝ) (c : ℝ) (h : k > 0) :
  Set.range (fun x => f k c x) ∩ Set.Ici 1 = Set.Ici (1 + c) :=
sorry

end range_of_f_on_interval_l2192_219278


namespace prime_congruence_problem_l2192_219215

theorem prime_congruence_problem (p q : Nat) (n : Nat) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : n > 1)
  (hpOdd : Odd p) (hqOdd : Odd q)
  (hcong1 : q^(n+2) ≡ 3^(n+2) [MOD p^n])
  (hcong2 : p^(n+2) ≡ 3^(n+2) [MOD q^n]) :
  p = 3 ∧ q = 3 := by
sorry

end prime_congruence_problem_l2192_219215


namespace no_right_triangle_perimeter_twice_hypotenuse_l2192_219247

theorem no_right_triangle_perimeter_twice_hypotenuse :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + Real.sqrt (a^2 + b^2) = 2 * Real.sqrt (a^2 + b^2) :=
by sorry

end no_right_triangle_perimeter_twice_hypotenuse_l2192_219247


namespace complex_multiplication_l2192_219260

theorem complex_multiplication (z : ℂ) : z = 2 - I → I^3 * z = -1 - 2*I := by
  sorry

end complex_multiplication_l2192_219260


namespace largest_stamps_per_page_l2192_219229

theorem largest_stamps_per_page (book1 book2 book3 : Nat) 
  (h1 : book1 = 1050) 
  (h2 : book2 = 1260) 
  (h3 : book3 = 1470) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 210 := by
  sorry

end largest_stamps_per_page_l2192_219229


namespace parabola_intercept_sum_l2192_219284

theorem parabola_intercept_sum : ∃ (a b c : ℝ),
  (∀ y : ℝ, 3 * y^2 - 9 * y + 5 = a ↔ y = 0) ∧
  (3 * b^2 - 9 * b + 5 = 0) ∧
  (3 * c^2 - 9 * c + 5 = 0) ∧
  (b ≠ c) ∧
  (a + b + c = 8) :=
by sorry

end parabola_intercept_sum_l2192_219284


namespace S_equals_T_l2192_219267

def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

theorem S_equals_T : S = T := by sorry

end S_equals_T_l2192_219267


namespace statements_B_and_C_are_correct_l2192_219253

theorem statements_B_and_C_are_correct (a b c d : ℝ) :
  (((a * b > 0 ∧ b * c - a * d > 0) → (c / a - d / b > 0)) ∧
   ((a > b ∧ c > d) → (a - d > b - c))) := by
  sorry

end statements_B_and_C_are_correct_l2192_219253


namespace solve_equation_l2192_219238

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
sorry

end solve_equation_l2192_219238


namespace modular_inverse_of_5_mod_33_l2192_219200

theorem modular_inverse_of_5_mod_33 :
  ∃ x : ℕ, x ≥ 0 ∧ x ≤ 32 ∧ (5 * x) % 33 = 1 ∧ x = 20 := by sorry

end modular_inverse_of_5_mod_33_l2192_219200


namespace darius_age_is_8_l2192_219212

-- Define the ages of Jenna and Darius
def jenna_age : ℕ := 13
def darius_age : ℕ := 21 - jenna_age

-- Theorem statement
theorem darius_age_is_8 :
  (jenna_age > darius_age) ∧ 
  (jenna_age + darius_age = 21) ∧
  (jenna_age = 13) →
  darius_age = 8 := by
sorry

end darius_age_is_8_l2192_219212


namespace no_factorial_with_2021_zeros_l2192_219275

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- There is no natural number n such that n! ends with exactly 2021 zeros -/
theorem no_factorial_with_2021_zeros : ∀ n : ℕ, trailingZeros n ≠ 2021 := by
  sorry

end no_factorial_with_2021_zeros_l2192_219275


namespace arithmetic_series_sum_l2192_219259

theorem arithmetic_series_sum : 
  ∀ (a₁ aₙ d : ℚ) (n : ℕ),
  a₁ = 16 → 
  aₙ = 32 → 
  d = 1/3 → 
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ)) / 2 = 1176 :=
by sorry

end arithmetic_series_sum_l2192_219259


namespace single_room_cost_l2192_219269

/-- Proves that the cost of each single room is $35 given the hotel booking information -/
theorem single_room_cost (total_rooms : ℕ) (double_room_cost : ℕ) (total_revenue : ℕ) (double_rooms : ℕ)
  (h1 : total_rooms = 260)
  (h2 : double_room_cost = 60)
  (h3 : total_revenue = 14000)
  (h4 : double_rooms = 196) :
  (total_revenue - double_rooms * double_room_cost) / (total_rooms - double_rooms) = 35 := by
  sorry

#check single_room_cost

end single_room_cost_l2192_219269


namespace quadratic_one_root_l2192_219295

theorem quadratic_one_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 2 * a * x - 1 = 0) → a = -1 :=
sorry

end quadratic_one_root_l2192_219295


namespace f_even_and_increasing_l2192_219271

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_even_and_increasing :
  (∀ x, f x = f (-x)) ∧  -- f is an even function
  (∀ x y, 0 < x → x < y → f x < f y) -- f is monotonically increasing on (0,+∞)
  := by sorry

end f_even_and_increasing_l2192_219271


namespace johns_current_income_l2192_219299

/-- Calculates John's current yearly income based on tax rates and tax increase --/
theorem johns_current_income
  (initial_tax_rate : ℝ)
  (new_tax_rate : ℝ)
  (initial_income : ℝ)
  (tax_increase : ℝ)
  (h1 : initial_tax_rate = 0.20)
  (h2 : new_tax_rate = 0.30)
  (h3 : initial_income = 1000000)
  (h4 : tax_increase = 250000) :
  ∃ current_income : ℝ,
    current_income = 1500000 ∧
    new_tax_rate * current_income - initial_tax_rate * initial_income = tax_increase :=
by
  sorry


end johns_current_income_l2192_219299


namespace parabola_intersection_l2192_219208

/-- Proves that (-3, 55) and (4, -8) are the only intersection points of the parabolas
    y = 3x^2 - 12x - 8 and y = 2x^2 - 10x + 4 -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x - 8
  let g (x : ℝ) := 2 * x^2 - 10 * x + 4
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -3 ∧ y = 55) ∨ (x = 4 ∧ y = -8) := by
  sorry

end parabola_intersection_l2192_219208


namespace min_c_plus_d_l2192_219202

theorem min_c_plus_d (a b c d : ℕ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  a < b ∧ b < c ∧ c < d →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (n : ℕ), a + b + c + d = n^2 →
  11 ≤ c + d ∧ ∃ (a' b' c' d' : ℕ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    a' < b' ∧ b' < c' ∧ c' < d' ∧
    a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ b' ≠ c' ∧ b' ≠ d' ∧ c' ≠ d' ∧
    ∃ (m : ℕ), a' + b' + c' + d' = m^2 ∧
    c' + d' = 11 :=
by sorry

end min_c_plus_d_l2192_219202


namespace coffee_mixture_proof_l2192_219255

/-- The cost of Colombian coffee beans per pound -/
def colombian_cost : ℝ := 5.50

/-- The cost of Peruvian coffee beans per pound -/
def peruvian_cost : ℝ := 4.25

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 40

/-- The desired cost per pound of the mixture -/
def mixture_cost : ℝ := 4.60

/-- The amount of Colombian coffee beans in the mixture -/
def colombian_amount : ℝ := 11.2

theorem coffee_mixture_proof :
  colombian_amount * colombian_cost + (total_weight - colombian_amount) * peruvian_cost = 
  mixture_cost * total_weight :=
by sorry

end coffee_mixture_proof_l2192_219255


namespace matrix_product_is_zero_l2192_219279

def matrix_product_zero (d e f : ℝ) : Prop :=
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -e; -d, 0, f; e, -f, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![d^2, d*e, d*f; d*e, e^2, e*f; d*f, e*f, f^2]
  A * B = 0

theorem matrix_product_is_zero (d e f : ℝ) : matrix_product_zero d e f := by
  sorry

end matrix_product_is_zero_l2192_219279


namespace pencil_pen_combinations_l2192_219248

theorem pencil_pen_combinations (pencil_types : Nat) (pen_types : Nat) :
  pencil_types = 4 → pen_types = 3 → pencil_types * pen_types = 12 := by
  sorry

end pencil_pen_combinations_l2192_219248


namespace building_height_l2192_219210

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_shadow_pos : 0 < building_shadow)
  (h : flagpole_height = 18)
  (s1 : flagpole_shadow = 45)
  (s2 : building_shadow = 60) :
  flagpole_height / flagpole_shadow * building_shadow = 24 := by
sorry


end building_height_l2192_219210


namespace plant_pricing_theorem_l2192_219268

/-- Represents the selling price per plant as a function of the number of plants per pot -/
def selling_price_per_plant (x : ℝ) : ℝ := -0.3 * x + 4.5

/-- Represents the price per pot as a function of the number of plants per pot -/
def price_per_pot (x : ℝ) : ℝ := -0.3 * x^2 + 4.5 * x

/-- Represents the cultivation cost per pot as a function of the number of plants -/
def cultivation_cost (x : ℝ) : ℝ := 2 + 0.3 * x

theorem plant_pricing_theorem :
  ∀ x : ℝ,
  5 ≤ x → x ≤ 12 →
  (selling_price_per_plant x = -0.3 * x + 4.5) ∧
  (price_per_pot x = -0.3 * x^2 + 4.5 * x) ∧
  ((price_per_pot x = 16.2) → (x = 6 ∨ x = 9)) ∧
  (∃ x : ℝ, (x = 12 ∨ x = 15) ∧
    30 * (price_per_pot x) - 40 * (cultivation_cost x) = 100) :=
by sorry


end plant_pricing_theorem_l2192_219268


namespace g_fixed_points_l2192_219213

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 5 ∨ x = 6 := by
  sorry

end g_fixed_points_l2192_219213


namespace parallel_to_y_axis_l2192_219274

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem
theorem parallel_to_y_axis (m n : ℝ) :
  let A : Point2D := ⟨-3, m⟩
  let B : Point2D := ⟨n, -4⟩
  (A.x = B.x) → -- Condition for line AB to be parallel to y-axis
  (n = -3 ∧ m ≠ -4) := by
  sorry


end parallel_to_y_axis_l2192_219274


namespace inequality_system_solution_set_l2192_219292

theorem inequality_system_solution_set : 
  {x : ℝ | (5 - 2*x ≤ 1) ∧ (x - 4 < 0)} = {x : ℝ | 2 ≤ x ∧ x < 4} := by
  sorry

end inequality_system_solution_set_l2192_219292


namespace complex_modulus_problem_l2192_219209

theorem complex_modulus_problem (z : ℂ) : 
  z = 3 + (3 + 4*I) / (4 - 3*I) → Complex.abs z = Real.sqrt 10 := by
sorry

end complex_modulus_problem_l2192_219209


namespace probability_4H_before_3T_is_4_57_l2192_219291

/-- The probability of encountering 4 heads before 3 consecutive tails in fair coin flips -/
def probability_4H_before_3T : ℚ :=
  4 / 57

/-- Theorem stating that the probability of encountering 4 heads before 3 consecutive tails
    in fair coin flips is equal to 4/57 -/
theorem probability_4H_before_3T_is_4_57 :
  probability_4H_before_3T = 4 / 57 := by
  sorry

end probability_4H_before_3T_is_4_57_l2192_219291


namespace smallest_n_satisfying_conditions_l2192_219228

/-- A positive integer n is a perfect square if there exists an integer k such that n = k^2 -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A positive integer n is a perfect fourth power if there exists an integer k such that n = k^4 -/
def IsPerfectFourthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^4

/-- The main theorem stating that 54 is the smallest positive integer satisfying the conditions -/
theorem smallest_n_satisfying_conditions : 
  (∀ m : ℕ, m > 0 ∧ m < 54 → ¬(IsPerfectSquare (2 * m) ∧ IsPerfectFourthPower (3 * m))) ∧ 
  (IsPerfectSquare (2 * 54) ∧ IsPerfectFourthPower (3 * 54)) := by
  sorry

end smallest_n_satisfying_conditions_l2192_219228


namespace convention_handshakes_count_l2192_219236

/-- The number of handshakes at the Interregional Mischief Convention --/
def convention_handshakes (n_gremlins n_imps n_disagreeing_imps n_affected_gremlins : ℕ) : ℕ :=
  let gremlin_handshakes := n_gremlins * (n_gremlins - 1) / 2
  let normal_imp_gremlin_handshakes := (n_imps - n_disagreeing_imps) * n_gremlins
  let affected_imp_gremlin_handshakes := n_disagreeing_imps * (n_gremlins - n_affected_gremlins)
  gremlin_handshakes + normal_imp_gremlin_handshakes + affected_imp_gremlin_handshakes

/-- Theorem stating the number of handshakes at the convention --/
theorem convention_handshakes_count : convention_handshakes 30 20 5 10 = 985 := by
  sorry

end convention_handshakes_count_l2192_219236


namespace password_probability_l2192_219258

def positive_single_digit_numbers : ℕ := 9
def alphabet_size : ℕ := 26
def vowels : ℕ := 5

def even_single_digit_numbers : ℕ := 4
def numbers_greater_than_five : ℕ := 4

theorem password_probability : 
  (even_single_digit_numbers : ℚ) / positive_single_digit_numbers *
  (vowels : ℚ) / alphabet_size *
  (numbers_greater_than_five : ℚ) / positive_single_digit_numbers = 40 / 1053 := by
  sorry

end password_probability_l2192_219258


namespace perpendicular_line_through_point_l2192_219257

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point :
  let l1 : Line := { a := 2, b := -3, c := 9 }
  let l2 : Line := { a := 3, b := 2, c := -1 }
  let p : Point := { x := -1, y := 2 }
  perpendicular l1 l2 ∧ pointOnLine p l2 := by sorry

end perpendicular_line_through_point_l2192_219257


namespace unique_prime_divisor_l2192_219219

theorem unique_prime_divisor : 
  ∃! p : ℕ, p ≥ 5 ∧ Prime p ∧ (p ∣ (p + 3)^(p-3) + (p + 5)^(p-5)) ∧ p = 2813 := by
  sorry

end unique_prime_divisor_l2192_219219


namespace product_of_multiples_l2192_219204

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem product_of_multiples : 
  smallest_two_digit_multiple_of_5 * smallest_three_digit_multiple_of_7 = 1050 := by
  sorry

end product_of_multiples_l2192_219204


namespace last_day_same_as_fifteenth_day_l2192_219263

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a year -/
structure DayInYear where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- A function to determine the day of the week for any day in the year,
    given the day of the week for the 15th day -/
def dayOfWeekFor (fifteenthDay : DayOfWeek) (dayNumber : Nat) : DayOfWeek :=
  sorry

theorem last_day_same_as_fifteenth_day 
  (year : Nat) 
  (h1 : year = 2005) 
  (h2 : (dayOfWeekFor DayOfWeek.Tuesday 15) = DayOfWeek.Tuesday) 
  (h3 : (dayOfWeekFor DayOfWeek.Tuesday 365) = (dayOfWeekFor DayOfWeek.Tuesday 15)) :
  (dayOfWeekFor DayOfWeek.Tuesday 365) = DayOfWeek.Tuesday := by
  sorry

end last_day_same_as_fifteenth_day_l2192_219263


namespace equation_system_solution_l2192_219233

theorem equation_system_solution : ∃ (x y z : ℝ),
  (2 * x - 3 * y - z = 0) ∧
  (x + 3 * y - 14 * z = 0) ∧
  (z = 2) ∧
  ((x^2 + 3*x*y) / (y^2 + z^2) = 7) := by
  sorry

end equation_system_solution_l2192_219233


namespace a_6_value_l2192_219242

/-- An arithmetic sequence where a_2 and a_10 are roots of 2x^2 - x - 7 = 0 -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  2 * (a 2)^2 - (a 2) - 7 = 0 ∧
  2 * (a 10)^2 - (a 10) - 7 = 0

theorem a_6_value (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 6 = 1/4 := by
  sorry

end a_6_value_l2192_219242


namespace finance_club_probability_l2192_219205

theorem finance_club_probability (total_students : ℕ) (interested_fraction : ℚ) 
  (h1 : total_students = 20)
  (h2 : interested_fraction = 3/4) :
  let interested_students := (interested_fraction * total_students).num
  let not_interested_students := total_students - interested_students
  1 - (not_interested_students / total_students) * ((not_interested_students - 1) / (total_students - 1)) = 18/19 := by
sorry

end finance_club_probability_l2192_219205


namespace beanie_babies_total_l2192_219237

/-- The number of beanie babies Lori has -/
def lori_beanie_babies : ℕ := 300

/-- The number of beanie babies Sydney has -/
def sydney_beanie_babies : ℕ := lori_beanie_babies / 15

/-- The initial number of beanie babies Jake has -/
def jake_initial_beanie_babies : ℕ := 2 * sydney_beanie_babies

/-- The number of additional beanie babies Jake gained -/
def jake_additional_beanie_babies : ℕ := (jake_initial_beanie_babies * 20) / 100

/-- The total number of beanie babies Jake has after gaining more -/
def jake_total_beanie_babies : ℕ := jake_initial_beanie_babies + jake_additional_beanie_babies

/-- The total number of beanie babies all three have -/
def total_beanie_babies : ℕ := lori_beanie_babies + sydney_beanie_babies + jake_total_beanie_babies

theorem beanie_babies_total : total_beanie_babies = 368 := by
  sorry

end beanie_babies_total_l2192_219237


namespace complex_number_equality_l2192_219240

theorem complex_number_equality : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end complex_number_equality_l2192_219240


namespace gcd_problem_l2192_219273

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2700 * k) :
  Int.gcd (b^2 + 27*b + 75) (b + 25) = 25 := by
  sorry

end gcd_problem_l2192_219273


namespace alice_pens_count_l2192_219234

/-- Proves that Alice has 60 pens given the conditions of the problem -/
theorem alice_pens_count :
  ∀ (alice_pens clara_pens alice_age clara_age : ℕ),
    clara_pens = (2 * alice_pens) / 5 →
    alice_pens - clara_pens = clara_age - alice_age →
    alice_age = 20 →
    clara_age > alice_age →
    clara_age + 5 = 61 →
    alice_pens = 60 := by
  sorry

end alice_pens_count_l2192_219234


namespace triangle_base_length_l2192_219226

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 9 →
  height = 6 →
  area = (base * height) / 2 →
  base = 3 := by
sorry

end triangle_base_length_l2192_219226


namespace pennies_spent_l2192_219282

/-- Given that Sam initially had 98 pennies and now has 5 pennies left,
    prove that the number of pennies Sam spent is 93. -/
theorem pennies_spent (initial : Nat) (left : Nat) (spent : Nat)
    (h1 : initial = 98)
    (h2 : left = 5)
    (h3 : spent = initial - left) :
  spent = 93 := by
  sorry

end pennies_spent_l2192_219282


namespace cricket_score_product_l2192_219207

def first_ten_scores : List Nat := [11, 6, 7, 5, 12, 8, 3, 10, 9, 4]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem cricket_score_product :
  ∀ (score_11 score_12 : Nat),
    score_11 < 15 →
    score_12 < 15 →
    is_integer ((List.sum first_ten_scores + score_11) / 11) →
    is_integer ((List.sum first_ten_scores + score_11 + score_12) / 12) →
    score_11 * score_12 = 14 :=
by sorry

end cricket_score_product_l2192_219207


namespace largest_root_is_three_l2192_219276

-- Define the cubic polynomial
def cubic (x : ℝ) : ℝ := x^3 - 3*x^2 - 8*x + 15

-- Define the conditions for p, q, and r
def root_conditions (p q r : ℝ) : Prop :=
  p + q + r = 3 ∧ p*q + p*r + q*r = -8 ∧ p*q*r = -15

-- Theorem statement
theorem largest_root_is_three :
  ∃ (p q r : ℝ), root_conditions p q r ∧
  (cubic p = 0 ∧ cubic q = 0 ∧ cubic r = 0) ∧
  (∀ x : ℝ, cubic x = 0 → x ≤ 3) :=
sorry

end largest_root_is_three_l2192_219276


namespace laundry_time_calculation_l2192_219235

theorem laundry_time_calculation (loads : ℕ) (wash_time dry_time : ℕ) : 
  loads = 8 → 
  wash_time = 45 → 
  dry_time = 60 → 
  (loads * (wash_time + dry_time)) / 60 = 14 := by
sorry

end laundry_time_calculation_l2192_219235


namespace debt_settlement_possible_l2192_219225

theorem debt_settlement_possible (vasya_coin_value : ℕ) (petya_coin_value : ℕ) 
  (debt : ℕ) (h1 : vasya_coin_value = 49) (h2 : petya_coin_value = 99) (h3 : debt = 1) :
  ∃ (n m : ℕ), vasya_coin_value * n - petya_coin_value * m = debt :=
by sorry

end debt_settlement_possible_l2192_219225


namespace dot_product_of_specific_vectors_l2192_219220

theorem dot_product_of_specific_vectors :
  let a : ℝ × ℝ := (-2, 4)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2) = 6 := by
  sorry

end dot_product_of_specific_vectors_l2192_219220


namespace f_composition_value_l2192_219261

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem f_composition_value : f (f (f (-2))) = 4 := by sorry

end f_composition_value_l2192_219261


namespace tan_double_alpha_l2192_219203

theorem tan_double_alpha (α β : Real) 
  (h1 : Real.tan (α + β) = 3) 
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (2 * α) = -1 := by
sorry

end tan_double_alpha_l2192_219203
