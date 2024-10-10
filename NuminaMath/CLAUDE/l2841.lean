import Mathlib

namespace pencil_distribution_l2841_284160

theorem pencil_distribution (num_students : ℕ) (total_pencils : ℕ) (pencils_per_dozen : ℕ) : 
  num_students = 46 → 
  total_pencils = 2208 → 
  pencils_per_dozen = 12 →
  (total_pencils / num_students) / pencils_per_dozen = 4 :=
by
  sorry

end pencil_distribution_l2841_284160


namespace det_A_eq_140_l2841_284169

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, -2; 8, 5, -4; 1, 3, 6]

theorem det_A_eq_140 : Matrix.det A = 140 := by sorry

end det_A_eq_140_l2841_284169


namespace wall_ratio_l2841_284139

theorem wall_ratio (width height length volume : ℝ) :
  width = 4 →
  height = 6 * width →
  volume = width * height * length →
  volume = 16128 →
  length / height = 7 := by
sorry

end wall_ratio_l2841_284139


namespace power_equation_solution_l2841_284197

theorem power_equation_solution (m : ℕ) : (4 : ℝ)^m * 2^3 = 8^5 → m = 6 := by
  sorry

end power_equation_solution_l2841_284197


namespace john_spent_625_l2841_284195

/-- The amount John spent on purchases with a coupon -/
def johnsSpending (vacuumPrice dishwasherPrice couponValue : ℕ) : ℕ :=
  vacuumPrice + dishwasherPrice - couponValue

/-- Theorem stating that John spent $625 -/
theorem john_spent_625 :
  johnsSpending 250 450 75 = 625 := by
  sorry

end john_spent_625_l2841_284195


namespace repeating_decimal_limit_l2841_284133

/-- Define the sequence of partial sums for 0.9999... -/
def partialSum (n : ℕ) : ℚ := 1 - (1 / 10 ^ n)

/-- Theorem: The limit of the sequence of partial sums for 0.9999... is 1 -/
theorem repeating_decimal_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |partialSum n - 1| < ε :=
sorry

end repeating_decimal_limit_l2841_284133


namespace certain_number_theorem_l2841_284110

theorem certain_number_theorem (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 25 := by
  sorry

end certain_number_theorem_l2841_284110


namespace side_ratio_not_imply_right_triangle_l2841_284193

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

/-- Definition of a right triangle --/
def IsRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

/-- The condition a:b:c = 1:2:3 --/
def SideRatio (t : Triangle) : Prop :=
  ∃ (k : ℝ), t.a = k ∧ t.b = 2*k ∧ t.c = 3*k

/-- Theorem: The condition a:b:c = 1:2:3 does not imply a right triangle --/
theorem side_ratio_not_imply_right_triangle :
  ∃ (t : Triangle), SideRatio t ∧ ¬IsRightTriangle t :=
sorry

end side_ratio_not_imply_right_triangle_l2841_284193


namespace symmetric_circle_equation_l2841_284104

/-- Given a circle with equation x²+y²-4x=0, this theorem states that 
    the equation of the circle symmetric to it with respect to the line y=x 
    is x²+y²-4y=0 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ x y, x^2 + y^2 - 4*x = 0 → (x^2 + y^2 - 4*y = 0 ↔ 
    ∃ x' y', x'^2 + y'^2 - 4*x' = 0 ∧ x = y' ∧ y = x')) := by
  sorry

end symmetric_circle_equation_l2841_284104


namespace table_seats_l2841_284149

/-- The number of people sitting at the table -/
def n : ℕ := 10

/-- The sum of seeds taken in the first round -/
def first_round_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of seeds taken in the second round -/
def second_round_sum (n : ℕ) : ℕ := n * (n + 1) / 2 + n^2

/-- The theorem stating that n = 10 satisfies the conditions -/
theorem table_seats : 
  (second_round_sum n - first_round_sum n = 100) ∧ 
  (∀ m : ℕ, second_round_sum m - first_round_sum m = 100 → m = n) := by
  sorry

#check table_seats

end table_seats_l2841_284149


namespace floor_expression_equals_two_l2841_284180

theorem floor_expression_equals_two :
  ⌊(2012^3 : ℝ) / (2010 * 2011) + (2010^3 : ℝ) / (2011 * 2012)⌋ = 2 := by
  sorry

end floor_expression_equals_two_l2841_284180


namespace soldier_count_l2841_284199

/-- The number of soldiers in a group forming a hollow square formation -/
def number_of_soldiers (A : ℕ) : ℕ := ((A + 2 * 3) - 3) * 3 * 4 + 9

/-- The side length of the hollow square formation -/
def side_length : ℕ := 5

theorem soldier_count :
  let A := side_length
  (A - 2 * 2)^2 * 3 + 9 = number_of_soldiers A ∧
  (A - 4) * 4 * 4 + 7 = number_of_soldiers A ∧
  number_of_soldiers A = 105 := by sorry

end soldier_count_l2841_284199


namespace alex_trip_distance_l2841_284109

/-- The distance from Alex's house to the harbor --/
def distance : ℝ := sorry

/-- Alex's initial speed --/
def initial_speed : ℝ := 45

/-- Alex's speed increase --/
def speed_increase : ℝ := 20

/-- Time saved by increasing speed --/
def time_saved : ℝ := 1.75

/-- The total travel time if Alex continued at the initial speed --/
def total_time_initial_speed : ℝ := sorry

theorem alex_trip_distance :
  /- Alex drives 45 miles in the first hour -/
  (initial_speed = 45) →
  /- He would be 1.5 hours late if he continues at the initial speed -/
  (total_time_initial_speed = distance / initial_speed) →
  /- He increases his speed by 20 miles per hour for the rest of the trip -/
  (∃ t : ℝ, t > 0 ∧ t < total_time_initial_speed ∧
    distance = initial_speed + (total_time_initial_speed - t) * (initial_speed + speed_increase)) →
  /- He arrives 15 minutes (0.25 hours) early -/
  (time_saved = 1.75) →
  /- The distance from Alex's house to the harbor is 613 miles -/
  distance = 613 := by sorry

end alex_trip_distance_l2841_284109


namespace notepad_cost_l2841_284141

/-- Given the total cost of notepads, pages per notepad, and total pages bought,
    calculate the cost of each notepad. -/
theorem notepad_cost (total_cost : ℚ) (pages_per_notepad : ℕ) (total_pages : ℕ) :
  total_cost = 10 →
  pages_per_notepad = 60 →
  total_pages = 480 →
  (total_cost / (total_pages / pages_per_notepad : ℚ)) = 1.25 := by
  sorry

end notepad_cost_l2841_284141


namespace simplify_complex_expression_l2841_284142

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that 3(2-i) + i(3+2i) = 4 -/
theorem simplify_complex_expression : 3 * (2 - i) + i * (3 + 2 * i) = (4 : ℂ) := by
  sorry

end simplify_complex_expression_l2841_284142


namespace set_A_properties_l2841_284134

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (A = {-2, 2}) ∧ (2 ∈ A) ∧ (-2 ∈ A) := by
  sorry

end set_A_properties_l2841_284134


namespace alloy_composition_l2841_284140

theorem alloy_composition (m₁ m₂ m₃ m₄ : ℝ) 
  (total_mass : m₁ + m₂ + m₃ + m₄ = 20)
  (first_second_relation : m₁ = 1.5 * m₂)
  (second_third_ratio : m₂ = (3/4) * m₃)
  (third_fourth_ratio : m₃ = (5/6) * m₄) :
  m₄ = 960 / 123 := by
  sorry

end alloy_composition_l2841_284140


namespace no_integer_solutions_l2841_284174

theorem no_integer_solutions : ¬∃ (k : ℕ+) (x : ℤ), 3 * (k : ℤ) * x - 18 = 5 * (k : ℤ) := by
  sorry

end no_integer_solutions_l2841_284174


namespace opposite_to_light_green_is_red_l2841_284178

-- Define the colors
inductive Color
| Red
| White
| Green
| Brown
| LightGreen
| Purple

-- Define a cube
structure Cube where
  faces : Fin 6 → Color
  different_colors : ∀ (i j : Fin 6), i ≠ j → faces i ≠ faces j

-- Define the concept of opposite faces
def opposite (c : Cube) (color1 color2 : Color) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = color1 ∧ c.faces j = color2 ∧
  ∀ (k : Fin 6), k ≠ i ∧ k ≠ j → (c.faces k = Color.White ∨ c.faces k = Color.Brown ∨ 
                                  c.faces k = Color.Purple ∨ c.faces k = Color.Green)

-- Theorem statement
theorem opposite_to_light_green_is_red (c : Cube) :
  opposite c Color.LightGreen Color.Red :=
sorry

end opposite_to_light_green_is_red_l2841_284178


namespace min_value_fraction_sum_l2841_284191

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)) + (y^2 / (x - 2)) ≥ 12 ∧
  ((x^2 / (y - 2)) + (y^2 / (x - 2)) = 12 ↔ x = 4 ∧ y = 4) :=
sorry

end min_value_fraction_sum_l2841_284191


namespace second_shirt_buttons_l2841_284146

/-- The number of buttons on the first type of shirt -/
def buttons_type1 : ℕ := 3

/-- The number of shirts ordered for each type -/
def shirts_per_type : ℕ := 200

/-- The total number of buttons used for all shirts -/
def total_buttons : ℕ := 1600

/-- The number of buttons on the second type of shirt -/
def buttons_type2 : ℕ := 5

theorem second_shirt_buttons :
  buttons_type2 * shirts_per_type + buttons_type1 * shirts_per_type = total_buttons :=
by sorry

end second_shirt_buttons_l2841_284146


namespace sqrt_360000_l2841_284117

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_l2841_284117


namespace sum_of_squares_l2841_284105

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 4*y = 8)
  (eq2 : y^2 + 6*z = 0)
  (eq3 : z^2 + 8*x = -16) :
  x^2 + y^2 + z^2 = 21 := by
  sorry

end sum_of_squares_l2841_284105


namespace field_breadth_is_50_l2841_284124

/-- Proves that the breadth of a field is 50 meters given specific conditions -/
theorem field_breadth_is_50 (field_length : ℝ) (tank_length tank_width tank_depth : ℝ) 
  (field_rise : ℝ) (b : ℝ) : 
  field_length = 90 →
  tank_length = 25 →
  tank_width = 20 →
  tank_depth = 4 →
  field_rise = 0.5 →
  tank_length * tank_width * tank_depth = (field_length * b - tank_length * tank_width) * field_rise →
  b = 50 := by
  sorry

end field_breadth_is_50_l2841_284124


namespace cyclical_fraction_bounds_l2841_284143

theorem cyclical_fraction_bounds (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  1 < (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) ∧ 
  (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) < 4 := by
  sorry

end cyclical_fraction_bounds_l2841_284143


namespace cube_root_of_negative_eight_l2841_284150

theorem cube_root_of_negative_eight :
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by
sorry

end cube_root_of_negative_eight_l2841_284150


namespace exists_value_not_taken_by_phi_at_odd_l2841_284157

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem exists_value_not_taken_by_phi_at_odd :
  ∃ m : ℕ, ∀ n : ℕ, isOdd n → phi n ≠ m := by sorry

end exists_value_not_taken_by_phi_at_odd_l2841_284157


namespace intersection_product_l2841_284122

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection of line l with y-axis
def point_M : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem intersection_product (A B : ℝ × ℝ) :
  curve_C A.1 A.2 →
  curve_C B.1 B.2 →
  line_l A.1 A.2 →
  line_l B.1 B.2 →
  A ≠ B →
  (∃ (t : ℝ), line_l t (point_M.2 + (t - point_M.1))) →
  |point_M.1 - A.1| * |point_M.1 - B.1| = 2 :=
sorry

end intersection_product_l2841_284122


namespace cookies_eaten_l2841_284129

theorem cookies_eaten (initial : Real) (remaining : Real) (eaten : Real) : 
  initial = 28.5 → remaining = 7.25 → eaten = initial - remaining → eaten = 21.25 :=
by sorry

end cookies_eaten_l2841_284129


namespace let_go_to_catch_not_specific_analysis_l2841_284171

/-- Definition of "specific analysis of specific issues" methodology --/
def specific_analysis (methodology : String) : Prop :=
  methodology = "analyzing the particularity of contradictions under the guidance of the universality principle of contradictions"

/-- Set of idioms --/
def idioms : Finset String := 
  {"Prescribe the right medicine for the illness; Make clothes to fit the person",
   "Let go to catch; Attack the east while feigning the west",
   "Act according to the situation; Adapt to local conditions",
   "Teach according to aptitude; Differentiate instruction based on individual differences"}

/-- Predicate to check if an idiom reflects the methodology --/
def reflects_methodology (idiom : String) : Prop :=
  idiom ≠ "Let go to catch; Attack the east while feigning the west"

/-- Theorem stating that "Let go to catch; Attack the east while feigning the west" 
    does not reflect the methodology --/
theorem let_go_to_catch_not_specific_analysis :
  ∃ (idiom : String), idiom ∈ idioms ∧ ¬(reflects_methodology idiom) :=
by
  sorry

#check let_go_to_catch_not_specific_analysis

end let_go_to_catch_not_specific_analysis_l2841_284171


namespace sine_product_ratio_equals_one_l2841_284153

theorem sine_product_ratio_equals_one :
  let d : ℝ := 2 * Real.pi / 15
  (Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d) * Real.sin (12 * d)) /
  (Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d) * Real.sin (6 * d)) = 1 :=
by sorry

end sine_product_ratio_equals_one_l2841_284153


namespace nested_fraction_evaluation_l2841_284125

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (2 + 1 / (1 + 1 / 4))) = 14 / 19 := by
  sorry

end nested_fraction_evaluation_l2841_284125


namespace unique_point_property_l2841_284173

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (-2, 0)

-- Define the point P
def P : ℝ → ℝ × ℝ := λ p => (p, 0)

-- Define a chord passing through the focus
def chord (m : ℝ) (x : ℝ) : ℝ := m * x + 2 * m

-- Define the angle equality condition
def angle_equality (p : ℝ) : Prop :=
  ∀ m : ℝ, ∃ A B : ℝ × ℝ,
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    A.2 = chord m A.1 ∧
    B.2 = chord m B.1 ∧
    (A.2 - 0) / (A.1 - p) = -(B.2 - 0) / (B.1 - p)

-- Theorem statement
theorem unique_point_property :
  ∃! p : ℝ, p > 0 ∧ angle_equality p :=
sorry

end unique_point_property_l2841_284173


namespace price_increase_over_two_years_l2841_284179

theorem price_increase_over_two_years (a b : ℝ) :
  let first_year_increase := 1 + a / 100
  let second_year_increase := 1 + b / 100
  let total_increase := first_year_increase * second_year_increase - 1
  total_increase = (a + b + a * b / 100) / 100 := by
sorry

end price_increase_over_two_years_l2841_284179


namespace total_tickets_sold_l2841_284144

/-- Represents the ticket sales scenario -/
structure TicketSales where
  student_price : ℝ
  adult_price : ℝ
  total_income : ℝ
  student_tickets : ℕ

/-- Theorem stating the total number of tickets sold -/
theorem total_tickets_sold (sale : TicketSales)
  (h1 : sale.student_price = 2)
  (h2 : sale.adult_price = 4.5)
  (h3 : sale.total_income = 60)
  (h4 : sale.student_tickets = 12) :
  ∃ (adult_tickets : ℕ), sale.student_tickets + adult_tickets = 20 :=
by
  sorry

#check total_tickets_sold

end total_tickets_sold_l2841_284144


namespace triangle_and_circle_problem_l2841_284112

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y + Real.sqrt 3)^2 = 1

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (on_C₁ : C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₁ C.1 C.2)
  (counterclockwise : sorry)  -- We would need to define this properly
  (A_coord : A = (2, 0))

-- State the theorem
theorem triangle_and_circle_problem (ABC : Triangle) :
  ABC.B = (-1, Real.sqrt 3) ∧
  ABC.C = (-1, -Real.sqrt 3) ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
    8 ≤ ((P.1 - ABC.B.1)^2 + (P.2 - ABC.B.2)^2) +
        ((P.1 - ABC.C.1)^2 + (P.2 - ABC.C.2)^2) ∧
    ((P.1 - ABC.B.1)^2 + (P.2 - ABC.B.2)^2) +
    ((P.1 - ABC.C.1)^2 + (P.2 - ABC.C.2)^2) ≤ 24 :=
sorry

end triangle_and_circle_problem_l2841_284112


namespace distinct_prime_factors_count_l2841_284114

theorem distinct_prime_factors_count (n : Nat) : n = 85 * 87 * 91 * 94 →
  Finset.card (Nat.factorization n).support = 8 := by
  sorry

end distinct_prime_factors_count_l2841_284114


namespace train_travel_time_l2841_284185

/-- Calculates the actual travel time in hours given the total travel time and break time in minutes. -/
def actualTravelTimeInHours (totalTravelTime breakTime : ℕ) : ℚ :=
  (totalTravelTime - breakTime) / 60

/-- Theorem stating that given a total travel time of 270 minutes with a 30-minute break,
    the actual travel time is 4 hours. -/
theorem train_travel_time :
  actualTravelTimeInHours 270 30 = 4 := by sorry

end train_travel_time_l2841_284185


namespace sandwich_cost_proof_l2841_284103

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 87/100

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 2

/-- The number of sodas purchased -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 646/100

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℚ := 149/100

theorem sandwich_cost_proof :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost :=
by sorry

end sandwich_cost_proof_l2841_284103


namespace triples_divisible_by_1000_l2841_284147

/-- The number of ordered triples (a,b,c) in {1, ..., 2016}³ such that a² + b² + c² ≡ 0 (mod 2017) is divisible by 1000. -/
theorem triples_divisible_by_1000 : ∃ N : ℕ,
  (N = (Finset.filter (fun (t : ℕ × ℕ × ℕ) =>
    let (a, b, c) := t
    1 ≤ a ∧ a ≤ 2016 ∧
    1 ≤ b ∧ b ≤ 2016 ∧
    1 ≤ c ∧ c ≤ 2016 ∧
    (a^2 + b^2 + c^2) % 2017 = 0)
    (Finset.product (Finset.range 2016) (Finset.product (Finset.range 2016) (Finset.range 2016)))).card) ∧
  N % 1000 = 0 :=
by sorry

end triples_divisible_by_1000_l2841_284147


namespace min_value_when_m_eq_one_m_range_when_f_geq_2x_l2841_284106

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| + |m * x - 1|

-- Part 1
theorem min_value_when_m_eq_one :
  (∃ (min : ℝ), ∀ x, f 1 x ≥ min ∧ ∃ x₀ ∈ Set.Icc (-1) 1, f 1 x₀ = min) ∧
  (∀ x, f 1 x = 2 ↔ x ∈ Set.Icc (-1) 1) := by sorry

-- Part 2
theorem m_range_when_f_geq_2x :
  (∀ x, f m x ≥ 2 * x) ↔ m ∈ Set.Iic (-1) ∪ Set.Ici 1 := by sorry

end min_value_when_m_eq_one_m_range_when_f_geq_2x_l2841_284106


namespace milk_processing_profit_comparison_l2841_284135

/-- Represents the profit calculation for a milk processing factory --/
theorem milk_processing_profit_comparison :
  let total_milk : ℝ := 9
  let fresh_milk_profit : ℝ := 500
  let yogurt_profit : ℝ := 1200
  let milk_slice_profit : ℝ := 2000
  let yogurt_capacity : ℝ := 3
  let milk_slice_capacity : ℝ := 1
  let processing_days : ℝ := 4

  let plan1_profit := milk_slice_capacity * processing_days * milk_slice_profit + 
                      (total_milk - milk_slice_capacity * processing_days) * fresh_milk_profit

  let plan2_milk_slice : ℝ := 1.5
  let plan2_yogurt : ℝ := 7.5
  let plan2_profit := plan2_milk_slice * milk_slice_profit + plan2_yogurt * yogurt_profit

  plan2_profit > plan1_profit ∧ 
  plan2_milk_slice + plan2_yogurt = total_milk ∧
  plan2_milk_slice / milk_slice_capacity + plan2_yogurt / yogurt_capacity = processing_days :=
by sorry

end milk_processing_profit_comparison_l2841_284135


namespace calculation_proof_equation_solution_proof_l2841_284128

-- Part 1
theorem calculation_proof :
  0.01⁻¹ + (-1 - 2/7)^0 - Real.sqrt 9 = 98 := by sorry

-- Part 2
theorem equation_solution_proof :
  ∀ x : ℝ, (2 / (x - 3) = 3 / (x - 2)) ↔ (x = 5) := by sorry

end calculation_proof_equation_solution_proof_l2841_284128


namespace johnny_wage_l2841_284154

/-- Given a total earning and hours worked, calculates the hourly wage -/
def hourly_wage (total_earning : ℚ) (hours_worked : ℚ) : ℚ :=
  total_earning / hours_worked

theorem johnny_wage :
  let total_earning : ℚ := 33/2  -- $16.5 represented as a rational number
  let hours_worked : ℚ := 2
  hourly_wage total_earning hours_worked = 33/4  -- $8.25 represented as a rational number
:= by sorry

end johnny_wage_l2841_284154


namespace constant_k_value_l2841_284121

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4) → k = -16 := by
  sorry

end constant_k_value_l2841_284121


namespace sqrt_difference_l2841_284196

theorem sqrt_difference : Real.sqrt 81 - Real.sqrt 144 = -7 := by
  sorry

end sqrt_difference_l2841_284196


namespace inscribed_sphere_radius_regular_tetrahedron_l2841_284190

/-- Given a regular tetrahedron with base area S and volume V,
    the radius R of its inscribed sphere is equal to 3V/(4S) -/
theorem inscribed_sphere_radius_regular_tetrahedron
  (S V : ℝ) (h_S : S > 0) (h_V : V > 0) :
  ∃ R : ℝ, R = (3 * V) / (4 * S) ∧ R > 0 := by
  sorry

end inscribed_sphere_radius_regular_tetrahedron_l2841_284190


namespace geometric_sequence_proof_l2841_284165

theorem geometric_sequence_proof (m : ℝ) :
  (4 / 1 = (2 * m + 8) / 4) → m = 4 := by
  sorry

end geometric_sequence_proof_l2841_284165


namespace tax_reduction_theorem_l2841_284181

/-- Proves that a tax reduction of 20% results in a 4% revenue decrease when consumption increases by 20% -/
theorem tax_reduction_theorem (T C : ℝ) (x : ℝ) 
  (h1 : x > 0)
  (h2 : T > 0)
  (h3 : C > 0)
  (h4 : (T - x / 100 * T) * (C + 20 / 100 * C) = 0.96 * T * C) :
  x = 20 := by
sorry

end tax_reduction_theorem_l2841_284181


namespace isabella_babysitting_afternoons_l2841_284111

/-- Calculates the number of afternoons Isabella babysits per week -/
def babysitting_afternoons (hourly_rate : ℚ) (hours_per_day : ℚ) (total_weeks : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / total_weeks) / (hourly_rate * hours_per_day)

/-- Proves that Isabella babysits 6 afternoons per week -/
theorem isabella_babysitting_afternoons :
  babysitting_afternoons 5 5 7 1050 = 6 := by
  sorry

end isabella_babysitting_afternoons_l2841_284111


namespace ones_digit_of_largest_power_of_two_dividing_32_factorial_l2841_284183

/- Define the factorial function -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/- Define a function to get the largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

/- Define a function to get the ones digit of a number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

/- Theorem statement -/
theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 := by
  sorry


end ones_digit_of_largest_power_of_two_dividing_32_factorial_l2841_284183


namespace bus_trip_distance_l2841_284151

theorem bus_trip_distance (speed : ℝ) (distance : ℝ) : 
  speed = 55 →
  distance / speed - 1 = distance / (speed + 5) →
  distance = 660 := by
sorry

end bus_trip_distance_l2841_284151


namespace isosceles_triangle_with_exterior_angle_ratio_l2841_284168

theorem isosceles_triangle_with_exterior_angle_ratio (α β γ : ℝ) : 
  -- The triangle is isosceles
  β = γ →
  -- Two exterior angles are in the ratio of 1:4
  ∃ (x : ℝ), (180 - α = x ∧ 180 - β = 4*x) ∨ (180 - β = x ∧ 180 - α = 4*x) →
  -- The sum of interior angles is 180°
  α + β + γ = 180 →
  -- The interior angles are 140°, 20°, and 20°
  α = 140 ∧ β = 20 ∧ γ = 20 := by
sorry

end isosceles_triangle_with_exterior_angle_ratio_l2841_284168


namespace negation_quadratic_inequality_l2841_284170

theorem negation_quadratic_inequality (x : ℝ) :
  (x^2 + x - 6 < 0) → (x ≤ 2) :=
sorry

#check negation_quadratic_inequality

end negation_quadratic_inequality_l2841_284170


namespace common_chord_of_circles_l2841_284164

/-- The equation of the common chord of two intersecting circles -/
theorem common_chord_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 10) ∧ ((x-1)^2 + (y-3)^2 = 10) → x + 3*y - 5 = 0 :=
by sorry

end common_chord_of_circles_l2841_284164


namespace madmen_count_l2841_284120

/-- The number of madmen in a psychiatric hospital -/
def num_madmen : ℕ := 20

/-- The number of bites the chief doctor received -/
def chief_doctor_bites : ℕ := 100

/-- Theorem stating the number of madmen in the hospital -/
theorem madmen_count :
  num_madmen = 20 ∧
  (7 * num_madmen = 2 * num_madmen + chief_doctor_bites) :=
by sorry

end madmen_count_l2841_284120


namespace invariant_quotient_division_inequality_l2841_284156

-- Define division with remainder
def div_with_rem (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Property of invariant quotient
theorem invariant_quotient (a b c : ℕ) (h : c ≠ 0) :
  div_with_rem (a * c) (b * c) = (a / b, c * (a % b)) :=
sorry

-- Main theorem
theorem division_inequality :
  div_with_rem 1700 500 ≠ div_with_rem 17 5 :=
sorry

end invariant_quotient_division_inequality_l2841_284156


namespace average_difference_theorem_l2841_284186

/-- The average number of students per teacher -/
def t (total_students : ℕ) (num_teachers : ℕ) : ℚ :=
  total_students / num_teachers

/-- The average number of students per student -/
def s (class_sizes : List ℕ) (total_students : ℕ) : ℚ :=
  (class_sizes.map (λ size => size * (size : ℚ) / total_students)).sum

theorem average_difference_theorem (total_students : ℕ) (num_teachers : ℕ) (class_sizes : List ℕ) :
  total_students = 120 →
  num_teachers = 5 →
  class_sizes = [60, 30, 20, 5, 5] →
  t total_students num_teachers - s class_sizes total_students = -17.25 := by
  sorry

end average_difference_theorem_l2841_284186


namespace bill_lines_count_l2841_284163

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles Bill drew -/
def triangles_drawn : ℕ := 12

/-- The number of squares Bill drew -/
def squares_drawn : ℕ := 8

/-- The number of pentagons Bill drew -/
def pentagons_drawn : ℕ := 4

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 
  triangles_drawn * triangle_sides + 
  squares_drawn * square_sides + 
  pentagons_drawn * pentagon_sides

theorem bill_lines_count : total_lines = 88 := by
  sorry

end bill_lines_count_l2841_284163


namespace angle_ABC_equals_cos_inverse_l2841_284194

/-- The angle ABC given three points A, B, and C in 3D space -/
def angle_ABC (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Converts radians to degrees -/
def to_degrees (x : ℝ) : ℝ := sorry

theorem angle_ABC_equals_cos_inverse :
  let A : ℝ × ℝ × ℝ := (-3, 1, 5)
  let B : ℝ × ℝ × ℝ := (-4, -2, 1)
  let C : ℝ × ℝ × ℝ := (-5, -2, 2)
  to_degrees (angle_ABC A B C) = Real.arccos ((3 * Real.sqrt 13) / 26) := by sorry

end angle_ABC_equals_cos_inverse_l2841_284194


namespace brianna_remaining_money_l2841_284175

theorem brianna_remaining_money
  (m n c : ℝ)
  (h1 : m > 0)
  (h2 : n > 0)
  (h3 : c > 0)
  (h4 : (1/4) * m = (1/2) * n * c) :
  m - ((1/2) * n * c) - ((1/10) * m) = (2/5) * m :=
sorry

end brianna_remaining_money_l2841_284175


namespace derivative_limit_relation_l2841_284136

theorem derivative_limit_relation (f : ℝ → ℝ) (x₀ : ℝ) (h : HasDerivAt f 2 x₀) :
  Filter.Tendsto (fun k => (f (x₀ - k) - f x₀) / (2 * k)) (Filter.atTop.comap (fun k => 1 / k)) (nhds (-1)) := by
  sorry

end derivative_limit_relation_l2841_284136


namespace inequality_proof_l2841_284198

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1 := by
  sorry

end inequality_proof_l2841_284198


namespace sharons_journey_l2841_284119

theorem sharons_journey (normal_time : ℝ) (traffic_time : ℝ) (speed_reduction : ℝ) :
  normal_time = 150 →
  traffic_time = 250 →
  speed_reduction = 15 →
  ∃ (distance : ℝ),
    distance = 80 ∧
    (distance / 4) / (distance / normal_time) +
    ((3 * distance) / 4) / ((distance / normal_time) - (speed_reduction / 60)) = traffic_time :=
by sorry

end sharons_journey_l2841_284119


namespace apples_per_adult_l2841_284126

def total_apples : ℕ := 450
def num_children : ℕ := 33
def apples_per_child : ℕ := 10
def num_adults : ℕ := 40

theorem apples_per_adult :
  (total_apples - num_children * apples_per_child) / num_adults = 3 := by
  sorry

end apples_per_adult_l2841_284126


namespace f_monotone_decreasing_l2841_284113

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (a - 2) * x + b

-- State the theorem
theorem f_monotone_decreasing (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →  -- Symmetry about the origin
  (∀ x ∈ Set.Icc (-4 : ℝ) (4 : ℝ), ∀ y ∈ Set.Icc (-4 : ℝ) (4 : ℝ), x ≤ y → f a b x ≥ f a b y) :=
by sorry

end f_monotone_decreasing_l2841_284113


namespace certain_number_is_26_l2841_284101

/-- The least positive integer divisible by every integer from 10 to 15 inclusive -/
def j : ℕ := sorry

/-- j is divisible by every integer from 10 to 15 inclusive -/
axiom j_divisible : ∀ k : ℕ, 10 ≤ k → k ≤ 15 → k ∣ j

/-- j is the least such positive integer -/
axiom j_least : ∀ m : ℕ, m > 0 → (∀ k : ℕ, 10 ≤ k → k ≤ 15 → k ∣ m) → j ≤ m

/-- The number that j is divided by to get 2310 -/
def x : ℕ := sorry

/-- j divided by x equals 2310 -/
axiom j_div_x : j / x = 2310

theorem certain_number_is_26 : x = 26 := by sorry

end certain_number_is_26_l2841_284101


namespace equation_has_integer_solution_l2841_284176

theorem equation_has_integer_solution : ∃ (x y : ℤ), x^2 - 2 = 7*y := by
  sorry

end equation_has_integer_solution_l2841_284176


namespace arithmetic_sequence_25th_term_l2841_284184

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

theorem arithmetic_sequence_25th_term
  (seq : ArithmeticSequence)
  (h₃ : seq.nthTerm 3 = 7)
  (h₁₈ : seq.nthTerm 18 = 37) :
  seq.nthTerm 25 = 51 := by
  sorry


end arithmetic_sequence_25th_term_l2841_284184


namespace no_cards_below_threshold_l2841_284123

def jungkook_card : ℚ := 0.8
def yoongi_card : ℚ := 1/2
def yoojeong_card : ℚ := 0.9
def yuna_card : ℚ := 1/3

def threshold : ℚ := 0.3

def count_below_threshold (cards : List ℚ) : ℕ :=
  (cards.filter (· < threshold)).length

theorem no_cards_below_threshold :
  count_below_threshold [jungkook_card, yoongi_card, yoojeong_card, yuna_card] = 0 := by
  sorry

end no_cards_below_threshold_l2841_284123


namespace scientific_notation_of_23766400_l2841_284167

theorem scientific_notation_of_23766400 :
  23766400 = 2.37664 * (10 : ℝ)^7 := by
  sorry

end scientific_notation_of_23766400_l2841_284167


namespace simplify_nested_roots_l2841_284155

theorem simplify_nested_roots (a : ℝ) : 
  (((a^16)^(1/3))^(1/4))^3 * (((a^16)^(1/4))^(1/3))^2 = a^(20/3) := by
  sorry

end simplify_nested_roots_l2841_284155


namespace smallest_number_with_conditions_l2841_284138

theorem smallest_number_with_conditions : ∃ x : ℕ, 
  (∃ k : ℤ, (x : ℤ) + 3 = 7 * k) ∧ 
  (∃ m : ℤ, (x : ℤ) - 5 = 8 * m) ∧ 
  (∀ y : ℕ, y < x → ¬((∃ k : ℤ, (y : ℤ) + 3 = 7 * k) ∧ (∃ m : ℤ, (y : ℤ) - 5 = 8 * m))) ∧
  x = 53 :=
by sorry

end smallest_number_with_conditions_l2841_284138


namespace total_fish_count_l2841_284161

theorem total_fish_count (micah kenneth matthias gabrielle : ℕ) : 
  micah = 7 →
  kenneth = 3 * micah →
  matthias = kenneth - 15 →
  gabrielle = 2 * (micah + kenneth + matthias) →
  micah + kenneth + matthias + gabrielle = 102 := by
sorry

end total_fish_count_l2841_284161


namespace factorization_of_2m_squared_minus_8_l2841_284116

theorem factorization_of_2m_squared_minus_8 (m : ℝ) :
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by
  sorry

end factorization_of_2m_squared_minus_8_l2841_284116


namespace distance_from_A_to_x_axis_l2841_284177

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (y : ℝ) : ℝ := |y|

/-- Point A in the Cartesian coordinate system -/
def point_A : ℝ × ℝ := (-5, -9)

theorem distance_from_A_to_x_axis :
  distance_to_x_axis (point_A.2) = 9 := by
  sorry

end distance_from_A_to_x_axis_l2841_284177


namespace cycle_loss_percentage_l2841_284131

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentage_loss (cost_price selling_price : ℕ) : ℚ :=
  (cost_price - selling_price : ℚ) / cost_price * 100

theorem cycle_loss_percentage :
  let cost_price := 2000
  let selling_price := 1800
  percentage_loss cost_price selling_price = 10 := by
sorry

end cycle_loss_percentage_l2841_284131


namespace function_property_implies_range_l2841_284182

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem function_property_implies_range (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_even_function f) 
  (h2 : decreasing_on_nonnegative f) 
  (h3 : f (a + 2) > f (a - 3)) : 
  a < 1/2 := by
sorry

end function_property_implies_range_l2841_284182


namespace infinite_solutions_iff_c_equals_five_l2841_284158

theorem infinite_solutions_iff_c_equals_five (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 := by
  sorry

end infinite_solutions_iff_c_equals_five_l2841_284158


namespace quadrilateral_circumcenter_l2841_284148

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Square
  | NonSquareRectangle
  | NonSquareRhombus
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a function to check if a quadrilateral has a circumcenter
def hasCircumcenter (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Square => True
  | Quadrilateral.NonSquareRectangle => True
  | Quadrilateral.NonSquareRhombus => False
  | Quadrilateral.Parallelogram => False
  | Quadrilateral.IsoscelesTrapezoid => True

-- Theorem stating which quadrilaterals have a circumcenter
theorem quadrilateral_circumcenter :
  ∀ q : Quadrilateral,
    hasCircumcenter q ↔
      (q = Quadrilateral.Square ∨
       q = Quadrilateral.NonSquareRectangle ∨
       q = Quadrilateral.IsoscelesTrapezoid) :=
by sorry

end quadrilateral_circumcenter_l2841_284148


namespace marbles_given_to_mary_l2841_284192

/-- Given that Dan initially had 64 marbles and now has 50 marbles,
    prove that he gave 14 marbles to Mary. -/
theorem marbles_given_to_mary (initial_marbles : ℕ) (current_marbles : ℕ)
    (h1 : initial_marbles = 64)
    (h2 : current_marbles = 50) :
    initial_marbles - current_marbles = 14 := by
  sorry

end marbles_given_to_mary_l2841_284192


namespace triangle_angle_problem_l2841_284115

theorem triangle_angle_problem (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ 2*x < 180 ∧ 
  (x + 2*x + 30 = 180) → x = 50 := by
  sorry

end triangle_angle_problem_l2841_284115


namespace sophie_germain_characterization_l2841_284162

/-- A prime number p is a Sophie Germain prime if 2p + 1 is also prime. -/
def SophieGermainPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

/-- The product of all possible units digits of Sophie Germain primes greater than 6 is 189. -/
axiom units_digit_product : ∃ (S : Finset ℕ), (∀ n ∈ S, n > 6 ∧ SophieGermainPrime n) ∧
  (Finset.prod S (λ n => n % 10) = 189)

theorem sophie_germain_characterization (p : ℕ) (h_prime : Nat.Prime p) (h_greater : p > 6) :
  SophieGermainPrime p ↔ Nat.Prime (2 * p + 1) :=
sorry

end sophie_germain_characterization_l2841_284162


namespace sequence_properties_l2841_284130

/-- Sequence definition -/
def a (n : ℕ) (c : ℤ) : ℤ := -n^2 + 4*n + c

/-- Theorem stating the value of c and the minimum m for which a_m ≤ 0 -/
theorem sequence_properties :
  ∃ (c : ℤ),
    (a 3 c = 24) ∧
    (c = 21) ∧
    (∀ m : ℕ, m > 0 → (a m c ≤ 0 ↔ m ≥ 7)) :=
by sorry

end sequence_properties_l2841_284130


namespace sin_equality_integer_solutions_l2841_284108

theorem sin_equality_integer_solutions (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (750 * π / 180) →
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by sorry

end sin_equality_integer_solutions_l2841_284108


namespace stock_investment_change_l2841_284137

theorem stock_investment_change (initial_investment : ℝ) : 
  initial_investment > 0 → 
  let first_year := initial_investment * (1 + 0.80)
  let second_year := first_year * (1 - 0.30)
  second_year = initial_investment * 1.26 := by
sorry

end stock_investment_change_l2841_284137


namespace paul_spent_three_tickets_l2841_284152

/-- Represents the number of tickets Paul spent on the Ferris wheel -/
def tickets_spent (initial : ℕ) (left : ℕ) : ℕ := initial - left

/-- Theorem stating that Paul spent 3 tickets on the Ferris wheel -/
theorem paul_spent_three_tickets :
  tickets_spent 11 8 = 3 := by
  sorry

end paul_spent_three_tickets_l2841_284152


namespace missing_number_is_twelve_l2841_284159

theorem missing_number_is_twelve : ∃ x : ℕ, 1234562 - x * 3 * 2 = 1234490 ∧ x = 12 := by
  sorry

end missing_number_is_twelve_l2841_284159


namespace circle_passes_through_fixed_point_l2841_284188

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8*y

-- Define the tangent line
def tangent_line (y : ℝ) : Prop := y = 2

-- Define the point that the circle passes through
def fixed_point : ℝ × ℝ := (0, -2)

-- Statement of the theorem
theorem circle_passes_through_fixed_point :
  ∀ (cx cy r : ℝ),
  parabola cx cy →
  (∃ (x : ℝ), tangent_line (cy + r) ∧ (x - cx)^2 + (2 - (cy + r))^2 = r^2) →
  (fixed_point.1 - cx)^2 + (fixed_point.2 - cy)^2 = r^2 := by
  sorry

end circle_passes_through_fixed_point_l2841_284188


namespace geometric_sequence_minimum_value_l2841_284107

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = q * a n

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_cond : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  (∀ q, q > 0 → 2 * a 5 + a 4 ≥ 12 * Real.sqrt 3) ∧
  (∃ q, q > 0 ∧ 2 * a 5 + a 4 = 12 * Real.sqrt 3) :=
sorry

end geometric_sequence_minimum_value_l2841_284107


namespace min_k_value_l2841_284102

theorem min_k_value (f : ℝ → ℝ) (k : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = k * (x^2 - x + 1) - x^4 * (1 - x)^4) →
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) →
  k ≥ 1 / 192 :=
by sorry

end min_k_value_l2841_284102


namespace second_student_male_probability_l2841_284166

/-- The probability that the second student to leave is male, given 2 male and 2 female students -/
def probability_second_male (num_male num_female : ℕ) : ℚ :=
  if num_male = 2 ∧ num_female = 2 then 1/6 else 0

/-- Theorem stating that the probability of the second student to leave being male is 1/6 -/
theorem second_student_male_probability :
  probability_second_male 2 2 = 1/6 := by
  sorry

end second_student_male_probability_l2841_284166


namespace pencils_and_pens_count_pencils_and_pens_count_proof_l2841_284187

theorem pencils_and_pens_count : ℕ → ℕ → Prop :=
  fun initial_pencils initial_pens =>
    (initial_pencils : ℚ) / initial_pens = 4 / 5 ∧
    ((initial_pencils + 1 : ℚ) / (initial_pens - 1) = 7 / 8) →
    initial_pencils + initial_pens = 45

-- The proof goes here
theorem pencils_and_pens_count_proof : ∃ (p q : ℕ), pencils_and_pens_count p q :=
  sorry

end pencils_and_pens_count_pencils_and_pens_count_proof_l2841_284187


namespace non_square_sequence_2003_l2841_284132

/-- The sequence of positive integers with perfect squares removed -/
def non_square_sequence : ℕ → ℕ := sorry

/-- The 2003rd term of the non-square sequence -/
def term_2003 : ℕ := non_square_sequence 2003

theorem non_square_sequence_2003 : term_2003 = 2048 := by sorry

end non_square_sequence_2003_l2841_284132


namespace lab_capacity_l2841_284100

/-- Represents a chemistry lab with work-stations for students -/
structure ChemistryLab where
  total_stations : ℕ
  two_student_stations : ℕ
  three_student_stations : ℕ
  station_sum : total_stations = two_student_stations + three_student_stations

/-- Calculates the total number of students that can use the lab at one time -/
def total_students (lab : ChemistryLab) : ℕ :=
  2 * lab.two_student_stations + 3 * lab.three_student_stations

/-- Theorem stating the number of students that can use the lab at one time -/
theorem lab_capacity (lab : ChemistryLab) 
    (h1 : lab.total_stations = 16)
    (h2 : lab.two_student_stations = 10) :
  total_students lab = 38 := by
  sorry

#eval total_students { total_stations := 16, two_student_stations := 10, three_student_stations := 6, station_sum := rfl }

end lab_capacity_l2841_284100


namespace sphere_radius_ratio_l2841_284127

theorem sphere_radius_ratio (V_large V_small r_large r_small : ℝ) :
  V_large = 500 * Real.pi
  → V_small = 0.25 * V_large
  → V_large = (4/3) * Real.pi * r_large^3
  → V_small = (4/3) * Real.pi * r_small^3
  → r_small / r_large = 1 / (2^(2/3)) := by
  sorry

end sphere_radius_ratio_l2841_284127


namespace even_function_extension_l2841_284189

/-- Given a real-valued function f that is even and defined as ln(x^2 - 2x + 2) for non-negative x,
    prove that f(x) = ln(x^2 + 2x + 2) for negative x -/
theorem even_function_extension (f : ℝ → ℝ) 
    (h_even : ∀ x, f x = f (-x))
    (h_non_neg : ∀ x ≥ 0, f x = Real.log (x^2 - 2*x + 2)) :
    ∀ x < 0, f x = Real.log (x^2 + 2*x + 2) := by
  sorry

end even_function_extension_l2841_284189


namespace women_at_soccer_game_l2841_284145

theorem women_at_soccer_game (adults : ℕ) (adult_women : ℕ) (student_surplus : ℕ) (male_students : ℕ)
  (h1 : adults = 1518)
  (h2 : adult_women = 536)
  (h3 : student_surplus = 525)
  (h4 : male_students = 1257) :
  adult_women + ((adults + student_surplus) - male_students) = 1322 :=
by sorry

end women_at_soccer_game_l2841_284145


namespace completing_square_proof_l2841_284172

theorem completing_square_proof (x : ℝ) : 
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
by sorry

end completing_square_proof_l2841_284172


namespace polynomial_coefficient_sum_l2841_284118

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end polynomial_coefficient_sum_l2841_284118
