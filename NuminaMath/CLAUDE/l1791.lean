import Mathlib

namespace final_x_value_l1791_179122

/-- Represents the state of the program at each iteration -/
structure ProgramState where
  x : ℕ
  y : ℕ

/-- Updates the program state according to the given rules -/
def updateState (state : ProgramState) : ProgramState :=
  { x := state.x + 2,
    y := state.y + state.x + 2 }

/-- Checks if the program should continue running -/
def shouldContinue (state : ProgramState) : Bool :=
  state.y < 10000

/-- Computes the final state of the program -/
def finalState : ProgramState :=
  sorry

/-- Proves that the final value of x is 201 -/
theorem final_x_value :
  finalState.x = 201 :=
sorry

end final_x_value_l1791_179122


namespace pencils_remainder_l1791_179130

theorem pencils_remainder : Nat.mod 13254839 7 = 3 := by
  sorry

end pencils_remainder_l1791_179130


namespace inscribed_hexagon_properties_l1791_179133

/-- A regular hexagon inscribed in a circle -/
structure InscribedHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : side_length > 0

/-- Properties of the inscribed hexagon -/
def InscribedHexagon.properties (h : InscribedHexagon) : Prop :=
  let r := h.side_length  -- radius of the circle is equal to side length
  let C := 2 * Real.pi * r  -- circumference of the circle
  let arc_length := C / 6  -- arc length for one side of the hexagon
  let P := 6 * h.side_length  -- perimeter of the hexagon
  C = 10 * Real.pi ∧ 
  arc_length = 5 * Real.pi / 3 ∧ 
  P = 30

/-- Theorem stating the properties of a regular hexagon with side length 5 inscribed in a circle -/
theorem inscribed_hexagon_properties :
  ∀ (h : InscribedHexagon), h.side_length = 5 → h.properties := by
  sorry

end inscribed_hexagon_properties_l1791_179133


namespace parallel_vectors_sum_norm_l1791_179163

/-- Two vectors in ℝ² -/
def a (x : ℝ) : Fin 2 → ℝ := ![x + 1, 2]
def b : Fin 2 → ℝ := ![1, -1]

/-- Parallel vectors have proportional components -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * w i

theorem parallel_vectors_sum_norm (x : ℝ) :
  parallel (a x) b → ‖(a x) + b‖ = Real.sqrt 2 := by
  sorry

end parallel_vectors_sum_norm_l1791_179163


namespace probability_different_colors_eq_137_162_l1791_179175

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probabilityDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.yellow + counts.green
  let pBlue := counts.blue / total
  let pRed := counts.red / total
  let pYellow := counts.yellow / total
  let pGreen := counts.green / total
  pBlue * (1 - pBlue) + pRed * (1 - pRed) + pYellow * (1 - pYellow) + pGreen * (1 - pGreen)

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_eq_137_162 :
  probabilityDifferentColors ⟨6, 5, 4, 3⟩ = 137 / 162 := by
  sorry

end probability_different_colors_eq_137_162_l1791_179175


namespace parabola_intersection_l1791_179180

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

theorem parabola_intersection :
  ∃ (x1 x2 : ℝ),
    x1 = (3 + Real.sqrt 129) / 4 ∧
    x2 = (3 - Real.sqrt 129) / 4 ∧
    parabola1 x1 = parabola2 x1 ∧
    parabola1 x2 = parabola2 x2 ∧
    ∀ (x : ℝ), parabola1 x = parabola2 x → x = x1 ∨ x = x2 :=
by sorry

end parabola_intersection_l1791_179180


namespace quadratic_zero_discriminant_geometric_progression_l1791_179187

/-- 
Given a quadratic equation ax^2 + bx + c = 0 with zero discriminant,
prove that a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) (h : b^2 - 4*a*c = 0) : 
  ∃ (r : ℝ), b = a*r ∧ c = b*r :=
sorry

end quadratic_zero_discriminant_geometric_progression_l1791_179187


namespace arithmetic_sequence_201_l1791_179136

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_201 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_5 : a 5 = 33) 
  (h_45 : a 45 = 153) : 
  a 61 = 201 := by
  sorry

end arithmetic_sequence_201_l1791_179136


namespace dishes_bananas_difference_is_ten_l1791_179167

/-- The number of pears Charles picked -/
def pears_picked : ℕ := 50

/-- The number of dishes Sandrine washed -/
def dishes_washed : ℕ := 160

/-- The number of bananas Charles cooked -/
def bananas_cooked : ℕ := 3 * pears_picked

/-- The difference between dishes washed and bananas cooked -/
def dishes_bananas_difference : ℕ := dishes_washed - bananas_cooked

theorem dishes_bananas_difference_is_ten :
  dishes_bananas_difference = 10 := by sorry

end dishes_bananas_difference_is_ten_l1791_179167


namespace smallest_n_exceeding_million_l1791_179193

def T (n : ℕ) : ℕ := n * 2^(n-1)

theorem smallest_n_exceeding_million :
  (∀ k < 20, T k ≤ 10^6) ∧ T 20 > 10^6 := by sorry

end smallest_n_exceeding_million_l1791_179193


namespace total_puppies_l1791_179199

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

theorem total_puppies : 
  puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4 = 74 := by
  sorry

end total_puppies_l1791_179199


namespace number_of_divisors_of_60_l1791_179147

theorem number_of_divisors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end number_of_divisors_of_60_l1791_179147


namespace largest_binomial_coefficient_seventh_term_l1791_179103

theorem largest_binomial_coefficient_seventh_term :
  let n : ℕ := 8
  let k : ℕ := 6  -- 7th term corresponds to choosing 6 out of 8
  ∀ i : ℕ, i ≤ n → (n.choose k) ≥ (n.choose i) :=
by sorry

end largest_binomial_coefficient_seventh_term_l1791_179103


namespace apple_basket_count_l1791_179185

theorem apple_basket_count : 
  ∀ (total : ℕ) (rotten : ℕ) (good : ℕ),
  rotten = (12 * total) / 100 →
  good = 66 →
  good = total - rotten →
  total = 75 := by
sorry

end apple_basket_count_l1791_179185


namespace car_train_distance_difference_l1791_179189

theorem car_train_distance_difference :
  let train_speed : ℝ := 60
  let car_speed : ℝ := 2 * train_speed
  let travel_time : ℝ := 3
  let train_distance : ℝ := train_speed * travel_time
  let car_distance : ℝ := car_speed * travel_time
  car_distance - train_distance = 180 := by
  sorry

end car_train_distance_difference_l1791_179189


namespace paving_cost_specific_room_l1791_179154

/-- Calculates the cost of paving a floor consisting of two rectangles -/
def paving_cost (length1 width1 length2 width2 cost_per_sqm : ℝ) : ℝ :=
  ((length1 * width1 + length2 * width2) * cost_per_sqm)

/-- Theorem: The cost of paving the specific room is Rs. 26,100 -/
theorem paving_cost_specific_room :
  paving_cost 5.5 3.75 4 3 800 = 26100 := by
  sorry

end paving_cost_specific_room_l1791_179154


namespace average_marks_chemistry_mathematics_l1791_179111

/-- Given that the total marks in physics, chemistry, and mathematics is 140 more than 
    the marks in physics, prove that the average mark in chemistry and mathematics is 70. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) 
  (h : P + C + M = P + 140) : (C + M) / 2 = 70 := by
  sorry

end average_marks_chemistry_mathematics_l1791_179111


namespace special_ellipse_equation_l1791_179105

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The first focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- The second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- A point on the ellipse -/
  P : ℝ × ℝ
  /-- The first focus is at (-4, 0) -/
  h_F₁ : F₁ = (-4, 0)
  /-- The second focus is at (4, 0) -/
  h_F₂ : F₂ = (4, 0)
  /-- The dot product of PF₁ and PF₂ is zero -/
  h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0
  /-- The area of triangle PF₁F₂ is 9 -/
  h_area : abs ((P.1 - F₁.1) * (P.2 - F₂.2) - (P.2 - F₁.2) * (P.1 - F₂.1)) / 2 = 9

/-- The standard equation of the special ellipse -/
def standardEquation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - e.F₁.1)^2 + (p.2 - e.F₁.2)^2 + (p.1 - e.F₂.1)^2 + (p.2 - e.F₂.2)^2 = 100}

/-- The main theorem: The standard equation of the special ellipse is x²/25 + y²/9 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : standardEquation e := by
  sorry

end special_ellipse_equation_l1791_179105


namespace plot_width_l1791_179197

/-- Given a rectangular plot with length 90 meters and perimeter that can be enclosed
    by 52 poles placed 5 meters apart, the width of the plot is 40 meters. -/
theorem plot_width (length : ℝ) (num_poles : ℕ) (pole_distance : ℝ) :
  length = 90 ∧ num_poles = 52 ∧ pole_distance = 5 →
  2 * (length + (num_poles * pole_distance / 2 - length) / 2) = num_poles * pole_distance →
  (num_poles * pole_distance / 2 - length) / 2 = 40 :=
by
  sorry

end plot_width_l1791_179197


namespace geometric_proof_l1791_179181

/-- The problem setup for the geometric proof -/
structure GeometricSetup where
  -- Line l equation
  l : ℝ → ℝ → Prop
  l_def : ∀ x y, l x y ↔ x + 2 * y - 1 = 0

  -- Circle C equations
  C : ℝ → ℝ → Prop
  C_def : ∀ x y, C x y ↔ ∃ φ, x = 3 + 3 * Real.cos φ ∧ y = 3 * Real.sin φ

  -- Ray OM
  α : ℝ
  α_range : 0 < α ∧ α < Real.pi / 2

  -- Function to convert Cartesian to polar coordinates
  to_polar : ℝ × ℝ → ℝ × ℝ

  -- Function to get the length of OP
  OP_length : ℝ

  -- Function to get the length of OQ
  OQ_length : ℝ

/-- The main theorem to be proved -/
theorem geometric_proof (setup : GeometricSetup) : 
  setup.OP_length * setup.OQ_length = 6 → setup.α = Real.pi / 4 := by
  sorry


end geometric_proof_l1791_179181


namespace f_inequality_implies_m_bound_l1791_179132

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

theorem f_inequality_implies_m_bound :
  (∀ x : ℝ, f (2^x - 4^x) + f (m * 2^x - 3) < 0) →
  m < 2 * Real.sqrt 3 - 1 :=
by sorry

end f_inequality_implies_m_bound_l1791_179132


namespace triamoeba_population_after_one_week_l1791_179173

/-- Represents the population of Triamoebas after a given number of days -/
def triamoeba_population (initial_population : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_rate ^ days

/-- Theorem stating that the Triamoeba population after 7 days is 2187 -/
theorem triamoeba_population_after_one_week :
  triamoeba_population 1 3 7 = 2187 := by
  sorry

end triamoeba_population_after_one_week_l1791_179173


namespace expression_evaluation_l1791_179108

theorem expression_evaluation : ((69 + 7 * 8) / 3) * 12 = 500 := by
  sorry

end expression_evaluation_l1791_179108


namespace original_average_calculation_l1791_179102

theorem original_average_calculation (total_pupils : ℕ) 
  (removed_pupils : ℕ) (removed_total : ℕ) (new_average : ℕ) : 
  total_pupils = 21 →
  removed_pupils = 4 →
  removed_total = 71 →
  new_average = 44 →
  (total_pupils * (total_pupils - removed_pupils) * new_average + 
   total_pupils * removed_total) / (total_pupils * total_pupils) = 39 :=
by sorry

end original_average_calculation_l1791_179102


namespace correct_divisor_proof_l1791_179186

theorem correct_divisor_proof (dividend : ℕ) (mistaken_divisor correct_quotient : ℕ) 
  (h1 : dividend % mistaken_divisor = 0)
  (h2 : dividend / mistaken_divisor = 63)
  (h3 : mistaken_divisor = 12)
  (h4 : dividend % correct_quotient = 0)
  (h5 : dividend / correct_quotient = 36) :
  dividend / 36 = 21 := by
  sorry

end correct_divisor_proof_l1791_179186


namespace jack_sugar_calculation_l1791_179170

/-- Given Jack's sugar operations, prove the final amount is correct. -/
theorem jack_sugar_calculation (initial : ℕ) (used : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 65 → used = 18 → bought = 50 → final = 97 → 
  final = initial - used + bought :=
by
  sorry

end jack_sugar_calculation_l1791_179170


namespace fish_cost_is_80_l1791_179126

/-- The cost of fish per kilogram in pesos -/
def fish_cost : ℝ := sorry

/-- The cost of pork per kilogram in pesos -/
def pork_cost : ℝ := sorry

/-- First condition: 530 pesos can buy 4 kg of fish and 2 kg of pork -/
axiom condition1 : 4 * fish_cost + 2 * pork_cost = 530

/-- Second condition: 875 pesos can buy 7 kg of fish and 3 kg of pork -/
axiom condition2 : 7 * fish_cost + 3 * pork_cost = 875

/-- Theorem: The cost of a kilogram of fish is 80 pesos -/
theorem fish_cost_is_80 : fish_cost = 80 := by sorry

end fish_cost_is_80_l1791_179126


namespace triangle_isosceles_condition_l1791_179123

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2b cos C, then the triangle is isosceles with B = C -/
theorem triangle_isosceles_condition (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a = 2 * b * Real.cos C   -- Given condition
  → B = C := by sorry

end triangle_isosceles_condition_l1791_179123


namespace total_lamps_is_147_l1791_179120

/-- The number of lamps per room -/
def lamps_per_room : ℕ := 7

/-- The number of rooms in the hotel -/
def rooms : ℕ := 21

/-- The total number of lamps bought for the hotel -/
def total_lamps : ℕ := lamps_per_room * rooms

/-- Theorem stating that the total number of lamps bought is 147 -/
theorem total_lamps_is_147 : total_lamps = 147 := by sorry

end total_lamps_is_147_l1791_179120


namespace green_balls_removal_l1791_179184

theorem green_balls_removal (total : ℕ) (green_percent : ℚ) (target_percent : ℚ) 
  (h_total : total = 600)
  (h_green_percent : green_percent = 70/100)
  (h_target_percent : target_percent = 60/100) :
  ∃ x : ℕ, 
    (↑x ≤ green_percent * ↑total) ∧ 
    ((green_percent * ↑total - ↑x) / (↑total - ↑x) = target_percent) ∧
    x = 150 := by
  sorry

end green_balls_removal_l1791_179184


namespace pure_imaginary_fraction_l1791_179134

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (a + Complex.I) / (1 + Complex.I)) → a = -1 := by
  sorry

end pure_imaginary_fraction_l1791_179134


namespace rhombus_perimeter_l1791_179106

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 8 * Real.sqrt 41 := by
  sorry

end rhombus_perimeter_l1791_179106


namespace arithmetic_sequence_middle_term_l1791_179159

theorem arithmetic_sequence_middle_term :
  ∀ (a : ℕ → ℤ), 
    (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
    a 0 = 3^2 →                                           -- first term is 3^2
    a 2 = 3^3 →                                           -- third term is 3^3
    a 1 = 18 :=                                           -- second term is 18
by
  sorry

end arithmetic_sequence_middle_term_l1791_179159


namespace sales_tax_calculation_l1791_179190

def total_cost : ℝ := 25
def tax_rate : ℝ := 0.05
def tax_free_cost : ℝ := 18.7

theorem sales_tax_calculation :
  ∃ (taxable_cost : ℝ),
    taxable_cost + tax_free_cost + taxable_cost * tax_rate = total_cost ∧
    taxable_cost * tax_rate = 0.3 := by
  sorry

end sales_tax_calculation_l1791_179190


namespace students_playing_neither_sport_l1791_179168

theorem students_playing_neither_sport 
  (total : ℕ) 
  (football : ℕ) 
  (tennis : ℕ) 
  (both : ℕ) 
  (h1 : total = 60) 
  (h2 : football = 36) 
  (h3 : tennis = 30) 
  (h4 : both = 22) : 
  total - (football + tennis - both) = 16 := by
  sorry

end students_playing_neither_sport_l1791_179168


namespace reverse_product_92565_l1791_179177

def is_reverse (a b : ℕ) : Prop :=
  (Nat.digits 10 a).reverse = Nat.digits 10 b

theorem reverse_product_92565 :
  ∃! (a b : ℕ), a < b ∧ is_reverse a b ∧ a * b = 92565 :=
by
  -- The proof would go here
  sorry

end reverse_product_92565_l1791_179177


namespace max_sum_problem_l1791_179137

theorem max_sum_problem (x y z v w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_v : v > 0) (pos_w : w > 0)
  (sum_cubes : x^3 + y^3 + z^3 + v^3 + w^3 = 2024) : 
  ∃ (M x_M y_M z_M v_M w_M : ℝ),
    (∀ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ 
      a^3 + b^3 + c^3 + d^3 + e^3 = 2024 → 
      a*c + 3*b*c + 4*c*d + 8*c*e ≤ M) ∧
    x_M > 0 ∧ y_M > 0 ∧ z_M > 0 ∧ v_M > 0 ∧ w_M > 0 ∧
    x_M^3 + y_M^3 + z_M^3 + v_M^3 + w_M^3 = 2024 ∧
    x_M*z_M + 3*y_M*z_M + 4*z_M*v_M + 8*z_M*w_M = M ∧
    M + x_M + y_M + z_M + v_M + w_M = 3055 := by
  sorry

end max_sum_problem_l1791_179137


namespace camera_price_difference_l1791_179101

/-- The list price of Camera Y in dollars -/
def list_price : ℚ := 52.99

/-- The discount amount at Best Deals in dollars -/
def best_deals_discount : ℚ := 12

/-- The discount percentage at Market Value -/
def market_value_discount_percent : ℚ := 20

/-- The sale price at Best Deals in dollars -/
def best_deals_price : ℚ := list_price - best_deals_discount

/-- The sale price at Market Value in dollars -/
def market_value_price : ℚ := list_price * (1 - market_value_discount_percent / 100)

/-- The price difference between Market Value and Best Deals in cents -/
def price_difference_cents : ℤ := 
  ⌊(market_value_price - best_deals_price) * 100⌋

theorem camera_price_difference : price_difference_cents = 140 := by
  sorry

end camera_price_difference_l1791_179101


namespace election_result_count_l1791_179151

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of positions to be filled -/
def num_positions : ℕ := 2

/-- The number of ways to select students for positions with at least one female student -/
def ways_with_female : ℕ := total_students.choose num_positions * num_positions.factorial - num_male.choose num_positions * num_positions.factorial

theorem election_result_count : ways_with_female = 14 := by
  sorry

end election_result_count_l1791_179151


namespace factorial_sum_equality_l1791_179100

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 40320 := by
  sorry

end factorial_sum_equality_l1791_179100


namespace find_number_l1791_179145

theorem find_number : ∃ x : ℝ, 3 * (x + 8) = 36 ∧ x = 4 := by sorry

end find_number_l1791_179145


namespace herring_cost_theorem_l1791_179121

def green_herring_price : ℝ := 2.50
def blue_herring_price : ℝ := 4.00
def green_herring_pounds : ℝ := 12
def blue_herring_pounds : ℝ := 7

theorem herring_cost_theorem :
  green_herring_price * green_herring_pounds + blue_herring_price * blue_herring_pounds = 58 :=
by sorry

end herring_cost_theorem_l1791_179121


namespace constant_function_integral_equals_one_l1791_179148

theorem constant_function_integral_equals_one : 
  ∫ x in (0 : ℝ)..1, (1 : ℝ) = 1 := by sorry

end constant_function_integral_equals_one_l1791_179148


namespace theorem_1_theorem_2_l1791_179178

-- Define the conditions
def condition_p (t : ℝ) : Prop := ∀ x : ℝ, (1/2) * x^2 - t*x + 1/2 > 0

def condition_q (t a : ℝ) : Prop := t^2 - (a-1)*t - a < 0

-- Theorem 1
theorem theorem_1 (t : ℝ) : condition_p t → -1 < t ∧ t < 1 := by sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) : 
  (∀ t : ℝ, condition_p t → condition_q t a) ∧ 
  (∃ t : ℝ, condition_q t a ∧ ¬condition_p t) → 
  a > 1 := by sorry

end theorem_1_theorem_2_l1791_179178


namespace dot_product_implies_t_l1791_179149

/-- Given vectors a and b in R^2, if their dot product is -2, then the second component of b is -4 -/
theorem dot_product_implies_t (a b : Fin 2 → ℝ) (h : a 0 = 5 ∧ a 1 = -7 ∧ b 0 = -6) :
  (a 0 * b 0 + a 1 * b 1 = -2) → b 1 = -4 := by
  sorry

end dot_product_implies_t_l1791_179149


namespace quadratic_polynomial_condition_l1791_179109

/-- Given a polynomial of the form 2a*x^4 + 5a*x^3 - 13x^2 - x^4 + 2021 + 2x + b*x^3 - b*x^4 - 13x^3,
    if it is a quadratic polynomial, then a^2 + b^2 = 13 -/
theorem quadratic_polynomial_condition (a b : ℝ) : 
  (∀ x, (2*a - 1 - b) * x^4 + (5*a + b - 13) * x^3 - 13*x^2 + 2*x + 2021 = 0 → 
        ∃ p q r : ℝ, ∀ x, p*x^2 + q*x + r = 0) →
  a^2 + b^2 = 13 := by sorry

end quadratic_polynomial_condition_l1791_179109


namespace initial_men_correct_l1791_179172

/-- The initial number of men working on a project -/
def initial_men : ℕ := 15

/-- The number of days to complete the work with the initial group -/
def initial_days : ℕ := 40

/-- The number of men who leave the project -/
def men_leaving : ℕ := 14

/-- The number of days worked before some men leave -/
def days_before_leaving : ℕ := 16

/-- The number of days to complete the remaining work after some men leave -/
def remaining_days : ℕ := 40

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct : 
  (initial_men : ℚ) * initial_days * (initial_days - days_before_leaving) = 
  (initial_men - men_leaving) * initial_days * remaining_days :=
sorry

end initial_men_correct_l1791_179172


namespace equality_check_l1791_179128

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-1)^3 = -1^3) ∧ 
  ((2/3)^2 ≠ 2^2/3) ∧ 
  ((-2)^2 ≠ -2^2) := by
  sorry

end equality_check_l1791_179128


namespace matrix_inverse_proof_l1791_179139

theorem matrix_inverse_proof : 
  let M : Matrix (Fin 3) (Fin 3) ℚ := !![4/11, 3/11, 0; -1/11, 2/11, 0; 0, 0, 1/3]
  let A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -3, 0; 1, 4, 0; 0, 0, 3]
  M * A = (1 : Matrix (Fin 3) (Fin 3) ℚ) := by sorry

end matrix_inverse_proof_l1791_179139


namespace converse_and_inverse_false_l1791_179182

-- Define quadrilaterals
structure Quadrilateral :=
  (is_rhombus : Bool)
  (is_parallelogram : Bool)

-- The given statement (not used in the proof, but included for completeness)
axiom rhombus_implies_parallelogram :
  ∀ q : Quadrilateral, q.is_rhombus → q.is_parallelogram

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ q : Quadrilateral, q.is_parallelogram ∧ ¬q.is_rhombus) ∧
  (∃ q : Quadrilateral, ¬q.is_rhombus ∧ q.is_parallelogram) := by
  sorry

end converse_and_inverse_false_l1791_179182


namespace tetromino_properties_l1791_179150

/-- A tetromino is a shape made up of 4 squares. -/
structure Tetromino where
  squares : Finset (ℤ × ℤ)
  card_eq_four : squares.card = 4

/-- Two tetrominos are considered identical if they can be superimposed by rotating but not by flipping. -/
def are_identical (t1 t2 : Tetromino) : Prop := sorry

/-- The set of all distinct tetrominos. -/
def distinct_tetrominos : Finset Tetromino := sorry

/-- A 4 × 7 rectangle. -/
def rectangle : Finset (ℤ × ℤ) := sorry

/-- Tiling a rectangle with tetrominos. -/
def tiling (r : Finset (ℤ × ℤ)) (ts : Finset Tetromino) : Prop := sorry

theorem tetromino_properties :
  (distinct_tetrominos.card = 7) ∧
  ¬ (tiling rectangle distinct_tetrominos) := by sorry

end tetromino_properties_l1791_179150


namespace triangle_angle_calculation_l1791_179169

theorem triangle_angle_calculation (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.sin A ≠ 0) (h6 : Real.sin B ≠ 0) 
  (h7 : 3 / Real.sin A = Real.sqrt 3 / Real.sin B) (h8 : A = π/3) : B = π/6 := by
  sorry

end triangle_angle_calculation_l1791_179169


namespace max_product_bound_l1791_179118

theorem max_product_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≤ 9 := by
sorry

end max_product_bound_l1791_179118


namespace investment_growth_l1791_179138

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem: Initial investment of $5000 at 10% p.a. for 2 years yields $6050.000000000001 -/
theorem investment_growth :
  let principal : ℝ := 5000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 6050.000000000001 :=
by sorry

end investment_growth_l1791_179138


namespace rectangle_area_l1791_179164

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

/-- Represents a rectangle defined by four vertices -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Theorem: Area of a specific rectangle -/
theorem rectangle_area (rect : Rectangle)
  (h1 : rect.P = ⟨1, 1⟩)
  (h2 : rect.Q = ⟨-3, 2⟩)
  (h3 : rect.R = ⟨-1, 6⟩)
  (h4 : rect.S = ⟨3, 5⟩)
  (h5 : squaredDistance rect.P rect.Q = squaredDistance rect.R rect.S) -- PQ is one side
  (h6 : squaredDistance rect.P rect.R = squaredDistance rect.Q rect.S) -- PR is a diagonal
  : (squaredDistance rect.P rect.Q * squaredDistance rect.P rect.R : ℝ) = 4 * 51 :=
sorry


end rectangle_area_l1791_179164


namespace item_list_price_l1791_179194

theorem item_list_price (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) → list_price = 40 := by
  sorry

end item_list_price_l1791_179194


namespace x_gt_1_necessary_not_sufficient_for_x_gt_2_l1791_179104

theorem x_gt_1_necessary_not_sufficient_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end x_gt_1_necessary_not_sufficient_for_x_gt_2_l1791_179104


namespace rational_sqrt_two_equation_l1791_179162

theorem rational_sqrt_two_equation (x y : ℚ) (h : x + Real.sqrt 2 * y = 0) : x = 0 ∧ y = 0 := by
  sorry

end rational_sqrt_two_equation_l1791_179162


namespace chocolate_cost_proof_l1791_179174

/-- The cost of the chocolate given the total spent and the cost of the candy bar -/
def chocolate_cost (total_spent : ℝ) (candy_bar_cost : ℝ) : ℝ :=
  total_spent - candy_bar_cost

theorem chocolate_cost_proof (total_spent candy_bar_cost : ℝ) 
  (h1 : total_spent = 13)
  (h2 : candy_bar_cost = 7) :
  chocolate_cost total_spent candy_bar_cost = 6 := by
  sorry

end chocolate_cost_proof_l1791_179174


namespace ascending_order_proof_l1791_179107

theorem ascending_order_proof : 222^2 < 22^22 ∧ 22^22 < 2^222 := by
  sorry

end ascending_order_proof_l1791_179107


namespace remaining_score_is_40_l1791_179140

/-- Represents the score of a dodgeball player -/
structure PlayerScore where
  hitting : ℕ
  catching : ℕ
  eliminating : ℕ

/-- Calculates the total score for a player -/
def totalScore (score : PlayerScore) : ℕ :=
  2 * score.hitting + 5 * score.catching + 10 * score.eliminating

/-- Represents the scores of all players in the game -/
structure GameScores where
  paige : PlayerScore
  brian : PlayerScore
  karen : PlayerScore
  jennifer : PlayerScore
  michael : PlayerScore

/-- The main theorem to prove -/
theorem remaining_score_is_40 (game : GameScores) : 
  totalScore game.paige = 21 →
  totalScore game.brian = 20 →
  game.karen.eliminating = 0 →
  game.jennifer.eliminating = 0 →
  game.michael.eliminating = 0 →
  totalScore game.paige + totalScore game.brian + 
  totalScore game.karen + totalScore game.jennifer + totalScore game.michael = 81 →
  totalScore game.karen + totalScore game.jennifer + totalScore game.michael = 40 := by
  sorry

#check remaining_score_is_40

end remaining_score_is_40_l1791_179140


namespace quadratic_polynomial_solutions_l1791_179117

-- Define a quadratic polynomial
def QuadraticPolynomial (α : Type*) [Field α] := α → α

-- Define the property of having exactly three solutions for (f(x))^3 - 4f(x) = 0
def HasThreeSolutionsCubicMinusFour (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, f x ^ 3 - 4 * f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define the property of having exactly two solutions for (f(x))^2 = 1
def HasTwoSolutionsSquaredEqualsOne (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∀ x : ℝ, f x ^ 2 = 1 ↔ x = y₁ ∨ x = y₂) ∧ y₁ ≠ y₂

-- The theorem statement
theorem quadratic_polynomial_solutions (f : QuadraticPolynomial ℝ) :
  HasThreeSolutionsCubicMinusFour f → HasTwoSolutionsSquaredEqualsOne f := by
  sorry

end quadratic_polynomial_solutions_l1791_179117


namespace triangle_least_perimeter_l1791_179113

theorem triangle_least_perimeter (a b c : ℕ) : 
  a = 24 → b = 37 → c > 0 → a + b > c → a + c > b → b + c > a → 
  (∀ x : ℕ, x > 0 → a + b > x → a + x > b → b + x > a → a + b + c ≤ a + b + x) →
  a + b + c = 75 :=
sorry

end triangle_least_perimeter_l1791_179113


namespace family_weight_ratio_l1791_179146

/-- Given the weights of a family, prove the ratio of child's weight to grandmother's weight -/
theorem family_weight_ratio 
  (total_weight : ℝ) 
  (daughter_child_weight : ℝ) 
  (daughter_weight : ℝ) 
  (h1 : total_weight = 150) 
  (h2 : daughter_child_weight = 60) 
  (h3 : daughter_weight = 42) : 
  ∃ (child_weight grandmother_weight : ℝ), 
    total_weight = grandmother_weight + daughter_weight + child_weight ∧ 
    daughter_child_weight = daughter_weight + child_weight ∧
    child_weight / grandmother_weight = 1 / 5 :=
by sorry

end family_weight_ratio_l1791_179146


namespace average_of_six_numbers_l1791_179195

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 := by
  sorry

end average_of_six_numbers_l1791_179195


namespace function_properties_and_inequality_l1791_179124

/-- Given a function f(x) = ax / (x^2 + b) with specific properties, 
    prove its exact form and a related inequality. -/
theorem function_properties_and_inequality 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 1) 
  (h_def : ∀ x, f x = a * x / (x^2 + b)) 
  (h_f1 : f 1 = 1) 
  (h_max : ∀ x, f x ≤ 3 * Real.sqrt 2 / 4) 
  (h_attains_max : ∃ x, f x = 3 * Real.sqrt 2 / 4) :
  (∀ x, f x = 3 * x / (x^2 + 2)) ∧ 
  (∀ m, (2 < m ∧ m ≤ 4) ↔ 
    (∀ x ∈ Set.Icc 1 2, f x ≤ 3 * m / ((x^2 + 2) * |x - m|))) := by
  sorry

end function_properties_and_inequality_l1791_179124


namespace highest_power_prime_factorial_l1791_179176

def highest_power_of_prime (p n : ℕ) : ℕ := sorry

def sum_of_floor_divisions (p n : ℕ) : ℕ := sorry

theorem highest_power_prime_factorial (p n : ℕ) (h_prime : Nat.Prime p) :
  ∃ k : ℕ, p ^ k ≤ n ∧ n < p ^ (k + 1) ∧
  highest_power_of_prime p n = sum_of_floor_divisions p n :=
sorry

end highest_power_prime_factorial_l1791_179176


namespace log_sum_fifty_twenty_l1791_179158

theorem log_sum_fifty_twenty : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end log_sum_fifty_twenty_l1791_179158


namespace least_subtraction_for_divisibility_problem_solution_l1791_179160

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! k : ℕ, k < d ∧ (n - k) % d = 0 :=
by
  sorry

theorem problem_solution :
  let n := 13294
  let d := 97
  ∃! k : ℕ, k < d ∧ (n - k) % d = 0 ∧ k = 5 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l1791_179160


namespace eight_students_in_neither_l1791_179155

/-- Represents the number of students in various categories of a science club. -/
structure ScienceClub where
  total : ℕ
  biology : ℕ
  chemistry : ℕ
  both : ℕ

/-- Calculates the number of students taking neither biology nor chemistry. -/
def studentsInNeither (club : ScienceClub) : ℕ :=
  club.total - (club.biology + club.chemistry - club.both)

/-- Theorem stating that for the given science club configuration, 
    8 students take neither biology nor chemistry. -/
theorem eight_students_in_neither (club : ScienceClub) 
  (h1 : club.total = 60)
  (h2 : club.biology = 42)
  (h3 : club.chemistry = 35)
  (h4 : club.both = 25) : 
  studentsInNeither club = 8 := by
  sorry

#eval studentsInNeither { total := 60, biology := 42, chemistry := 35, both := 25 }

end eight_students_in_neither_l1791_179155


namespace square_minus_four_equals_negative_three_l1791_179192

theorem square_minus_four_equals_negative_three (a : ℤ) (h : a = -1) : a^2 - 4 = -3 := by
  sorry

end square_minus_four_equals_negative_three_l1791_179192


namespace train_platform_crossing_time_l1791_179135

/-- Given a train of length 1200 m that crosses a tree in 120 seconds,
    prove that the time required for the train to pass a platform of length 400 m is 160 seconds. -/
theorem train_platform_crossing_time :
  let train_length : ℝ := 1200
  let tree_crossing_time : ℝ := 120
  let platform_length : ℝ := 400
  let train_speed : ℝ := train_length / tree_crossing_time
  let total_distance : ℝ := train_length + platform_length
  total_distance / train_speed = 160 := by sorry

end train_platform_crossing_time_l1791_179135


namespace english_only_students_l1791_179142

theorem english_only_students (total : ℕ) (both : ℕ) (french : ℕ) (english : ℕ) : 
  total = 30 ∧ 
  both = 2 ∧ 
  english = 3 * french ∧ 
  total = french + english - both → 
  english - both = 20 := by
sorry

end english_only_students_l1791_179142


namespace sum_equals_42_l1791_179127

/-- An increasing geometric sequence with specific properties -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  geometric : ∃ r : ℝ, r > 1 ∧ ∀ n, a (n + 1) = r * a n
  sum_condition : a 1 + a 3 + a 5 = 21
  a3_value : a 3 = 6

/-- The sum of specific terms in the sequence equals 42 -/
theorem sum_equals_42 (seq : IncreasingGeometricSequence) : seq.a 5 + seq.a 3 + seq.a 9 = 42 := by
  sorry

end sum_equals_42_l1791_179127


namespace A_power_50_l1791_179143

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 1; -12, -3]

theorem A_power_50 : A^50 = !![301, 50; -900, -301] := by sorry

end A_power_50_l1791_179143


namespace sum_1000th_to_1010th_term_l1791_179179

def arithmeticSequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

def sumArithmeticSequence (a₁ d m n : ℕ) : ℕ :=
  ((n - m + 1) * (arithmeticSequence a₁ d m + arithmeticSequence a₁ d n)) / 2

theorem sum_1000th_to_1010th_term :
  sumArithmeticSequence 3 7 1000 1010 = 77341 := by
  sorry

end sum_1000th_to_1010th_term_l1791_179179


namespace equation_equivalence_l1791_179144

theorem equation_equivalence (x y : ℝ) : 2 * y - 4 * x + 5 = 0 ↔ y = 2 * x - 5 / 2 := by
  sorry

end equation_equivalence_l1791_179144


namespace degree_of_product_l1791_179156

/-- The degree of a polynomial resulting from the multiplication of three given expressions -/
theorem degree_of_product : ℕ :=
  let expr1 := (fun x : ℝ => x^5)
  let expr2 := (fun x : ℝ => x + 1/x)
  let expr3 := (fun x : ℝ => 1 + 3/x + 4/x^2 + 5/x^3)
  let product := (fun x : ℝ => expr1 x * expr2 x * expr3 x)
  6

#check degree_of_product

end degree_of_product_l1791_179156


namespace unfair_die_expected_value_l1791_179165

/-- Represents an unfair eight-sided die -/
structure UnfairDie where
  prob_8 : ℚ
  prob_others : ℚ
  sum_to_one : prob_8 + 7 * prob_others = 1
  prob_8_is_3_8 : prob_8 = 3/8

/-- Expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.prob_8 * 8

/-- Theorem stating the expected value of the unfair die is 77/14 -/
theorem unfair_die_expected_value :
  ∀ (d : UnfairDie), expected_value d = 77/14 := by
  sorry

end unfair_die_expected_value_l1791_179165


namespace quadrilateral_area_l1791_179198

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5,
    and two sides with distinct integer lengths has an area of 12. -/
theorem quadrilateral_area : ∀ (A B C D : ℝ × ℝ),
  -- Right angles at B and D
  (B.2 - A.2) * (C.1 - B.1) + (B.1 - A.1) * (C.2 - B.2) = 0 →
  (D.2 - C.2) * (A.1 - D.1) + (D.1 - C.1) * (A.2 - D.2) = 0 →
  -- Diagonal AC = 5
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 25 →
  -- Two sides with distinct integer lengths
  ∃ (a b : ℕ), a ≠ b ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
     (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2) ∨
    ((A.1 - D.1)^2 + (A.2 - D.2)^2 = a^2 ∧
     (D.1 - C.1)^2 + (D.2 - C.2)^2 = b^2) →
  -- Area of ABCD is 12
  abs ((A.1 - C.1) * (B.2 - D.2) - (A.2 - C.2) * (B.1 - D.1)) / 2 = 12 :=
by sorry

end quadrilateral_area_l1791_179198


namespace root_product_cubic_l1791_179125

theorem root_product_cubic (p q r : ℂ) : 
  (3 * p^3 - 8 * p^2 + p - 9 = 0) →
  (3 * q^3 - 8 * q^2 + q - 9 = 0) →
  (3 * r^3 - 8 * r^2 + r - 9 = 0) →
  p * q * r = 3 := by
  sorry

end root_product_cubic_l1791_179125


namespace interest_difference_theorem_l1791_179152

/-- Calculates the difference between the principal and the simple interest --/
def interestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal - (principal * rate * time)

/-- Theorem stating that the difference between the principal and the simple interest
    is 340 for the given conditions --/
theorem interest_difference_theorem :
  interestDifference 500 0.04 8 = 340 := by
  sorry

end interest_difference_theorem_l1791_179152


namespace noah_sales_this_month_l1791_179166

/-- Represents Noah's painting sales --/
structure NoahSales where
  large_price : ℕ
  small_price : ℕ
  last_month_large : ℕ
  last_month_small : ℕ

/-- Calculates Noah's sales for this month --/
def this_month_sales (s : NoahSales) : ℕ :=
  2 * (s.large_price * s.last_month_large + s.small_price * s.last_month_small)

/-- Theorem: Noah's sales for this month equal $1200 --/
theorem noah_sales_this_month (s : NoahSales) 
  (h1 : s.large_price = 60)
  (h2 : s.small_price = 30)
  (h3 : s.last_month_large = 8)
  (h4 : s.last_month_small = 4) :
  this_month_sales s = 1200 := by
  sorry

end noah_sales_this_month_l1791_179166


namespace money_ratio_l1791_179131

def money_problem (total : ℚ) (rene : ℚ) : Prop :=
  ∃ (isha florence : ℚ) (k : ℕ),
    isha = (1/3) * total ∧
    florence = (1/2) * isha ∧
    florence = k * rene ∧
    total = isha + florence + rene ∧
    rene = 300 ∧
    total = 1650 ∧
    florence / rene = 3/2

theorem money_ratio :
  money_problem 1650 300 := by sorry

end money_ratio_l1791_179131


namespace systematic_sampling_theorem_l1791_179183

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_employees : ℕ
  num_groups : ℕ
  group_size : ℕ
  sample_interval : ℕ

/-- Calculates the number to be drawn from a specific group -/
def number_from_group (s : SystematicSampling) (group : ℕ) (position : ℕ) : ℕ :=
  (group - 1) * s.sample_interval + position

/-- The main theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.num_groups = 40)
  (h3 : s.group_size = 5)
  (h4 : s.sample_interval = 5)
  (h5 : number_from_group s 5 3 = 22) :
  number_from_group s 8 3 = 37 := by
  sorry

end systematic_sampling_theorem_l1791_179183


namespace instantaneous_velocity_zero_at_two_l1791_179110

-- Define the motion law
def motion_law (t : ℝ) : ℝ := t^2 - 4*t + 5

-- Define the instantaneous velocity (derivative of motion law)
def instantaneous_velocity (t : ℝ) : ℝ := 2*t - 4

-- Theorem statement
theorem instantaneous_velocity_zero_at_two :
  ∃ (t : ℝ), instantaneous_velocity t = 0 ∧ t = 2 :=
by
  sorry

end instantaneous_velocity_zero_at_two_l1791_179110


namespace more_pockets_than_dollars_per_wallet_l1791_179129

/-- Represents the distribution of dollars, wallets, and pockets -/
structure Distribution where
  total_dollars : ℕ
  num_wallets : ℕ
  num_pockets : ℕ
  dollars_per_pocket : ℕ → ℕ
  dollars_per_wallet : ℕ → ℕ

/-- The conditions of the problem -/
def problem_conditions (d : Distribution) : Prop :=
  d.total_dollars = 2003 ∧
  d.num_wallets > 0 ∧
  d.num_pockets > 0 ∧
  (∀ p, p < d.num_pockets → d.dollars_per_pocket p < d.num_wallets) ∧
  (∀ w, w < d.num_wallets → d.dollars_per_wallet w ≤ d.total_dollars / d.num_wallets)

/-- The theorem to be proved -/
theorem more_pockets_than_dollars_per_wallet (d : Distribution) 
  (h : problem_conditions d) : 
  ∀ w, w < d.num_wallets → d.num_pockets > d.dollars_per_wallet w :=
sorry

end more_pockets_than_dollars_per_wallet_l1791_179129


namespace min_integer_value_of_fraction_l1791_179191

theorem min_integer_value_of_fraction (x : ℝ) : 
  ⌊(4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 3)⌋ ≥ -15 ∧ 
  ∃ y : ℝ, ⌊(4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 3)⌋ = -15 := by
  sorry

end min_integer_value_of_fraction_l1791_179191


namespace function_nonnegative_iff_a_in_range_l1791_179116

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x + 2

-- Define the theorem
theorem function_nonnegative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ [-1, 1] → f a x ≥ 0) ↔ a ∈ [1, 5] := by sorry

end function_nonnegative_iff_a_in_range_l1791_179116


namespace height_relation_l1791_179161

/-- Two right circular cylinders with equal volume and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  equal_volume : r1^2 * h1 = r2^2 * h2  -- cylinders have equal volume
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry

end height_relation_l1791_179161


namespace min_legs_correct_l1791_179153

/-- The length of the circular track in meters -/
def track_length : ℕ := 660

/-- The length of each leg of the race in meters -/
def leg_length : ℕ := 150

/-- The minimum number of legs required for the relay race -/
def min_legs : ℕ := 22

/-- Theorem stating that the minimum number of legs is correct -/
theorem min_legs_correct :
  min_legs = Nat.lcm track_length leg_length / leg_length :=
by sorry

end min_legs_correct_l1791_179153


namespace tetrahedron_volume_bound_l1791_179114

/-- A tetrahedron represented by four vertices in 3D space -/
structure Tetrahedron where
  v1 : Fin 3 → ℝ
  v2 : Fin 3 → ℝ
  v3 : Fin 3 → ℝ
  v4 : Fin 3 → ℝ

/-- A cube represented by its lower and upper bounds in 3D space -/
structure Cube where
  lower : Fin 3 → ℝ
  upper : Fin 3 → ℝ

/-- Function to calculate the volume of a tetrahedron -/
def volume_tetrahedron (t : Tetrahedron) : ℝ := sorry

/-- Function to calculate the volume of a cube -/
def volume_cube (c : Cube) : ℝ := sorry

/-- Function to check if a tetrahedron is inside a cube -/
def is_inside (t : Tetrahedron) (c : Cube) : Prop := sorry

/-- The main theorem: volume of tetrahedron inside unit cube is at most 1/3 -/
theorem tetrahedron_volume_bound (t : Tetrahedron) (c : Cube) :
  is_inside t c →
  (∀ i, c.lower i = 0 ∧ c.upper i = 1) →
  volume_tetrahedron t ≤ (1/3 : ℝ) := by sorry

end tetrahedron_volume_bound_l1791_179114


namespace sum_largest_smallest_even_le_49_l1791_179141

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def largest_even_le_49 : ℕ := 48

def smallest_even_gt_0_le_49 : ℕ := 2

theorem sum_largest_smallest_even_le_49 :
  largest_even_le_49 + smallest_even_gt_0_le_49 = 50 ∧
  is_even largest_even_le_49 ∧
  is_even smallest_even_gt_0_le_49 ∧
  largest_even_le_49 ≤ 49 ∧
  smallest_even_gt_0_le_49 > 0 ∧
  smallest_even_gt_0_le_49 ≤ 49 ∧
  ∀ n, is_even n ∧ n > 0 ∧ n ≤ 49 → n ≤ largest_even_le_49 ∧ n ≥ smallest_even_gt_0_le_49 :=
by sorry

end sum_largest_smallest_even_le_49_l1791_179141


namespace all_propositions_false_l1791_179115

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations between lines and planes
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular (p q : Plane) : Prop := sorry

-- Define the given lines and planes
variable (a b l : Line)
variable (α β γ : Plane)

-- Axioms for different objects
axiom different_lines : a ≠ b ∧ b ≠ l ∧ a ≠ l
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- The four propositions
def proposition1 : Prop := 
  ∀ a b α, parallel a b → contained_in b α → parallel a α

def proposition2 : Prop := 
  ∀ a b α, perpendicular a b → perpendicular b α → parallel a α

def proposition3 : Prop := 
  ∀ l α β, plane_perpendicular α β → contained_in l α → perpendicular l β

def proposition4 : Prop := 
  ∀ l a b α, perpendicular l a → perpendicular l b → 
    contained_in a α → contained_in b α → perpendicular l α

-- Theorem stating all propositions are false
theorem all_propositions_false : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end all_propositions_false_l1791_179115


namespace boat_upstream_distance_l1791_179196

/-- Calculates the distance traveled against the stream in one hour -/
def distance_against_stream (boat_speed : ℝ) (downstream_distance : ℝ) : ℝ :=
  let stream_speed := downstream_distance - boat_speed
  boat_speed - stream_speed

/-- Theorem: Given a boat with speed 4 km/hr in still water that travels 6 km
    downstream in one hour, it will travel 2 km upstream in one hour -/
theorem boat_upstream_distance :
  distance_against_stream 4 6 = 2 := by
  sorry

end boat_upstream_distance_l1791_179196


namespace inequality_condition_l1791_179119

theorem inequality_condition (x y : ℝ) : (x > y ∧ 1 / x > 1 / y) ↔ x * y < 0 := by
  sorry

end inequality_condition_l1791_179119


namespace sieve_of_eratosthenes_complexity_l1791_179112

/-- The Sieve of Eratosthenes algorithm for finding prime numbers up to n. -/
def sieve_of_eratosthenes (n : ℕ) : List ℕ := sorry

/-- The time complexity function for the Sieve of Eratosthenes algorithm. -/
def time_complexity (n : ℕ) : ℝ := sorry

/-- Big O notation for comparing functions. -/
def big_o (f g : ℕ → ℝ) : Prop := 
  ∃ c k : ℝ, c > 0 ∧ ∀ n : ℕ, n ≥ k → f n ≤ c * g n

/-- Theorem stating that the time complexity of the Sieve of Eratosthenes is O(n log(n)^2). -/
theorem sieve_of_eratosthenes_complexity :
  big_o time_complexity (λ n => n * (Real.log n)^2) :=
sorry

end sieve_of_eratosthenes_complexity_l1791_179112


namespace largest_common_term_l1791_179157

def is_in_first_sequence (x : ℕ) : Prop := ∃ n : ℕ, x = 2 + 5 * n

def is_in_second_sequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 8 * m

theorem largest_common_term : 
  (∀ x : ℕ, x ≤ 150 → is_in_first_sequence x → is_in_second_sequence x → x ≤ 117) ∧ 
  is_in_first_sequence 117 ∧ 
  is_in_second_sequence 117 :=
sorry

end largest_common_term_l1791_179157


namespace pet_ownership_l1791_179188

theorem pet_ownership (total : ℕ) (dogs cats other_pets no_pets : ℕ) 
  (dogs_cats : ℕ) (dogs_other : ℕ) (cats_other : ℕ) :
  total = 32 →
  dogs = total / 2 →
  cats = total * 3 / 8 →
  other_pets = 6 →
  no_pets = 5 →
  dogs_cats = 10 →
  dogs_other = 2 →
  cats_other = 9 →
  ∃ (all_three : ℕ),
    all_three = 1 ∧
    dogs + cats + other_pets - dogs_cats - dogs_other - cats_other + all_three = total - no_pets :=
by sorry

end pet_ownership_l1791_179188


namespace sum_of_unit_vector_magnitudes_l1791_179171

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given two unit vectors, prove that the sum of their magnitudes is 2 -/
theorem sum_of_unit_vector_magnitudes
  (a₀ b₀ : E) 
  (ha : ‖a₀‖ = 1) 
  (hb : ‖b₀‖ = 1) : 
  ‖a₀‖ + ‖b₀‖ = 2 := by
sorry

end sum_of_unit_vector_magnitudes_l1791_179171
