import Mathlib

namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l1345_134523

-- Define the set G
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.2 ∧ p.2 ≤ 8 ∧ (p.1 - 3)^2 + 31 = (p.2 - 4)^2 + 8 * Real.sqrt (p.2 * (8 - p.2))}

-- Define the tangent line condition
def isTangentLine (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b ∧ p ∈ G ∧
  ∀ q : ℝ × ℝ, q ∈ G → q.2 ≤ m * q.1 + b

-- Theorem statement
theorem tangent_point_coordinates :
  ∃! p : ℝ × ℝ, p ∈ G ∧ 
    ∃ m : ℝ, m < 0 ∧ 
      isTangentLine m 4 p ∧
      p = (12/5, 8/5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l1345_134523


namespace NUMINAMATH_CALUDE_uncool_parents_count_l1345_134519

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 20)
  (h3 : cool_moms = 25)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 5 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l1345_134519


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1345_134549

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

end NUMINAMATH_CALUDE_total_tickets_sold_l1345_134549


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l1345_134583

theorem sphere_radius_ratio (V_large V_small r_large r_small : ℝ) :
  V_large = 500 * Real.pi
  → V_small = 0.25 * V_large
  → V_large = (4/3) * Real.pi * r_large^3
  → V_small = (4/3) * Real.pi * r_small^3
  → r_small / r_large = 1 / (2^(2/3)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l1345_134583


namespace NUMINAMATH_CALUDE_average_difference_theorem_l1345_134582

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

end NUMINAMATH_CALUDE_average_difference_theorem_l1345_134582


namespace NUMINAMATH_CALUDE_largest_difference_is_62_l1345_134520

/-- Given a list of four digits, returns the largest 2-digit number that can be formed --/
def largest_two_digit (digits : List Nat) : Nat :=
  sorry

/-- Given a list of four digits, returns the smallest 2-digit number that can be formed --/
def smallest_two_digit (digits : List Nat) : Nat :=
  sorry

/-- The set of digits to be used --/
def digit_set : List Nat := [2, 4, 6, 8]

theorem largest_difference_is_62 :
  largest_two_digit digit_set - smallest_two_digit digit_set = 62 :=
sorry

end NUMINAMATH_CALUDE_largest_difference_is_62_l1345_134520


namespace NUMINAMATH_CALUDE_machine_purchase_price_l1345_134526

/-- Given a machine with specified costs and selling price, calculates the original purchase price. -/
theorem machine_purchase_price (repair_cost : ℕ) (transport_cost : ℕ) (profit_percentage : ℚ) (selling_price : ℕ) : 
  repair_cost = 5000 →
  transport_cost = 1000 →
  profit_percentage = 50 / 100 →
  selling_price = 25500 →
  ∃ (purchase_price : ℕ), 
    (purchase_price : ℚ) + repair_cost + transport_cost = 
      selling_price / (1 + profit_percentage) ∧
    purchase_price = 11000 :=
by sorry

end NUMINAMATH_CALUDE_machine_purchase_price_l1345_134526


namespace NUMINAMATH_CALUDE_sequence_properties_l1345_134557

/-- Sequence definition -/
def a (n : ℕ) (c : ℤ) : ℤ := -n^2 + 4*n + c

/-- Theorem stating the value of c and the minimum m for which a_m ≤ 0 -/
theorem sequence_properties :
  ∃ (c : ℤ),
    (a 3 c = 24) ∧
    (c = 21) ∧
    (∀ m : ℕ, m > 0 → (a m c ≤ 0 ↔ m ≥ 7)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1345_134557


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l1345_134542

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_reduction_rate : ℝ)
  (quantity_increase_rate : ℝ)
  (h1 : price_reduction_rate = 0.2)
  (h2 : quantity_increase_rate = 0.5)
  : (((1 - price_reduction_rate) * (1 + quantity_increase_rate) - 1) * 100 = 20) := by
  sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l1345_134542


namespace NUMINAMATH_CALUDE_scientific_notation_of_23766400_l1345_134556

theorem scientific_notation_of_23766400 :
  23766400 = 2.37664 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_23766400_l1345_134556


namespace NUMINAMATH_CALUDE_apples_per_adult_l1345_134535

def total_apples : ℕ := 450
def num_children : ℕ := 33
def apples_per_child : ℕ := 10
def num_adults : ℕ := 40

theorem apples_per_adult :
  (total_apples - num_children * apples_per_child) / num_adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_adult_l1345_134535


namespace NUMINAMATH_CALUDE_madmen_count_l1345_134553

/-- The number of madmen in a psychiatric hospital -/
def num_madmen : ℕ := 20

/-- The number of bites the chief doctor received -/
def chief_doctor_bites : ℕ := 100

/-- Theorem stating the number of madmen in the hospital -/
theorem madmen_count :
  num_madmen = 20 ∧
  (7 * num_madmen = 2 * num_madmen + chief_doctor_bites) :=
by sorry

end NUMINAMATH_CALUDE_madmen_count_l1345_134553


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l1345_134505

/-- Proves that given a total distance of 350 km, where the first 200 km is traveled at 20 km/h,
    and the average speed for the entire trip is 17.5 km/h, the speed for the remaining distance is 15 km/h. -/
theorem bicycle_speed_problem (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 350 →
  first_part_distance = 200 →
  first_part_speed = 20 →
  average_speed = 17.5 →
  (total_distance - first_part_distance) / ((total_distance / average_speed) - (first_part_distance / first_part_speed)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_problem_l1345_134505


namespace NUMINAMATH_CALUDE_sqrt_difference_l1345_134547

theorem sqrt_difference : Real.sqrt 81 - Real.sqrt 144 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_l1345_134547


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1345_134566

theorem rectangle_area_change 
  (l w : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) : 
  let new_length := l * 1.6
  let new_width := w * 0.4
  let initial_area := l * w
  let new_area := new_length * new_width
  (new_area - initial_area) / initial_area = -0.36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1345_134566


namespace NUMINAMATH_CALUDE_pizza_pieces_per_pizza_pizza_pieces_theorem_l1345_134514

theorem pizza_pieces_per_pizza 
  (num_students : ℕ) 
  (pizzas_per_student : ℕ) 
  (total_pieces : ℕ) : ℕ :=
  let total_pizzas := num_students * pizzas_per_student
  total_pieces / total_pizzas

#check pizza_pieces_per_pizza 10 20 1200 = 6

-- Proof
theorem pizza_pieces_theorem :
  pizza_pieces_per_pizza 10 20 1200 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pieces_per_pizza_pizza_pieces_theorem_l1345_134514


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l1345_134593

theorem geometric_sequence_proof (m : ℝ) :
  (4 / 1 = (2 * m + 8) / 4) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l1345_134593


namespace NUMINAMATH_CALUDE_tom_next_birthday_l1345_134515

-- Define the ages as real numbers
def tom_age : ℝ := sorry
def jerry_age : ℝ := sorry
def spike_age : ℝ := sorry

-- Define the relationships between ages
axiom jerry_spike_relation : jerry_age = 1.2 * spike_age
axiom tom_jerry_relation : tom_age = 0.7 * jerry_age

-- Define the sum of ages
axiom age_sum : tom_age + jerry_age + spike_age = 36

-- Theorem to prove
theorem tom_next_birthday : ⌊tom_age⌋ + 1 = 11 := by sorry

end NUMINAMATH_CALUDE_tom_next_birthday_l1345_134515


namespace NUMINAMATH_CALUDE_youth_entertainment_suitable_for_sampling_other_scenarios_not_suitable_for_sampling_l1345_134512

/-- Represents a survey scenario -/
inductive SurveyScenario
| CompanyHealthCheck
| EpidemicTemperatureCheck
| YouthEntertainment
| AirplaneSecurity

/-- Determines if a survey scenario is suitable for sampling -/
def isSuitableForSampling (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.YouthEntertainment => True
  | _ => False

/-- Theorem stating that the youth entertainment survey is suitable for sampling -/
theorem youth_entertainment_suitable_for_sampling :
  isSuitableForSampling SurveyScenario.YouthEntertainment :=
by sorry

/-- Theorem stating that other scenarios are not suitable for sampling -/
theorem other_scenarios_not_suitable_for_sampling (scenario : SurveyScenario) :
  scenario ≠ SurveyScenario.YouthEntertainment →
  ¬ (isSuitableForSampling scenario) :=
by sorry

end NUMINAMATH_CALUDE_youth_entertainment_suitable_for_sampling_other_scenarios_not_suitable_for_sampling_l1345_134512


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1345_134502

theorem chess_tournament_participants (total_games : ℕ) (h : total_games = 231) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ n = 22 ∧ n - 1 = 21 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1345_134502


namespace NUMINAMATH_CALUDE_class_composition_l1345_134567

theorem class_composition (total students : ℕ) (girls boys : ℕ) : 
  students = girls + boys →
  (girls : ℚ) / (students : ℚ) = 60 / 100 →
  ((girls - 1 : ℚ) / ((students - 3) : ℚ)) = 125 / 200 →
  girls = 21 ∧ boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l1345_134567


namespace NUMINAMATH_CALUDE_soldier_count_l1345_134533

/-- The number of soldiers in a group forming a hollow square formation -/
def number_of_soldiers (A : ℕ) : ℕ := ((A + 2 * 3) - 3) * 3 * 4 + 9

/-- The side length of the hollow square formation -/
def side_length : ℕ := 5

theorem soldier_count :
  let A := side_length
  (A - 2 * 2)^2 * 3 + 9 = number_of_soldiers A ∧
  (A - 4) * 4 * 4 + 7 = number_of_soldiers A ∧
  number_of_soldiers A = 105 := by sorry

end NUMINAMATH_CALUDE_soldier_count_l1345_134533


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_exterior_angle_ratio_l1345_134543

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

end NUMINAMATH_CALUDE_isosceles_triangle_with_exterior_angle_ratio_l1345_134543


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l1345_134587

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

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l1345_134587


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1345_134596

theorem smallest_number_with_conditions : ∃ x : ℕ, 
  (∃ k : ℤ, (x : ℤ) + 3 = 7 * k) ∧ 
  (∃ m : ℤ, (x : ℤ) - 5 = 8 * m) ∧ 
  (∀ y : ℕ, y < x → ¬((∃ k : ℤ, (y : ℤ) + 3 = 7 * k) ∧ (∃ m : ℤ, (y : ℤ) - 5 = 8 * m))) ∧
  x = 53 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1345_134596


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l1345_134558

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentage_loss (cost_price selling_price : ℕ) : ℚ :=
  (cost_price - selling_price : ℚ) / cost_price * 100

theorem cycle_loss_percentage :
  let cost_price := 2000
  let selling_price := 1800
  percentage_loss cost_price selling_price = 10 := by
sorry

end NUMINAMATH_CALUDE_cycle_loss_percentage_l1345_134558


namespace NUMINAMATH_CALUDE_invariant_quotient_division_inequality_l1345_134590

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

end NUMINAMATH_CALUDE_invariant_quotient_division_inequality_l1345_134590


namespace NUMINAMATH_CALUDE_sharons_journey_l1345_134572

theorem sharons_journey (normal_time : ℝ) (traffic_time : ℝ) (speed_reduction : ℝ) :
  normal_time = 150 →
  traffic_time = 250 →
  speed_reduction = 15 →
  ∃ (distance : ℝ),
    distance = 80 ∧
    (distance / 4) / (distance / normal_time) +
    ((3 * distance) / 4) / ((distance / normal_time) - (speed_reduction / 60)) = traffic_time :=
by sorry

end NUMINAMATH_CALUDE_sharons_journey_l1345_134572


namespace NUMINAMATH_CALUDE_bus_trip_distance_l1345_134576

theorem bus_trip_distance (speed : ℝ) (distance : ℝ) : 
  speed = 55 →
  distance / speed - 1 = distance / (speed + 5) →
  distance = 660 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_distance_l1345_134576


namespace NUMINAMATH_CALUDE_root_of_cubic_equation_l1345_134597

theorem root_of_cubic_equation :
  let x : ℝ := Real.sin (π / 14)
  (0 < x ∧ x < Real.pi / 13) ∧
  8 * x^3 - 4 * x^2 - 4 * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_of_cubic_equation_l1345_134597


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1345_134571

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1345_134571


namespace NUMINAMATH_CALUDE_stock_investment_change_l1345_134595

theorem stock_investment_change (initial_investment : ℝ) : 
  initial_investment > 0 → 
  let first_year := initial_investment * (1 + 0.80)
  let second_year := first_year * (1 - 0.30)
  second_year = initial_investment * 1.26 := by
sorry

end NUMINAMATH_CALUDE_stock_investment_change_l1345_134595


namespace NUMINAMATH_CALUDE_price_increase_over_two_years_l1345_134586

theorem price_increase_over_two_years (a b : ℝ) :
  let first_year_increase := 1 + a / 100
  let second_year_increase := 1 + b / 100
  let total_increase := first_year_increase * second_year_increase - 1
  total_increase = (a + b + a * b / 100) / 100 := by
sorry

end NUMINAMATH_CALUDE_price_increase_over_two_years_l1345_134586


namespace NUMINAMATH_CALUDE_floor_expression_equals_two_l1345_134529

theorem floor_expression_equals_two :
  ⌊(2012^3 : ℝ) / (2010 * 2011) + (2010^3 : ℝ) / (2011 * 2012)⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_two_l1345_134529


namespace NUMINAMATH_CALUDE_no_divisible_by_ten_l1345_134525

/-- The function g(x) = x^2 + 5x + 3 -/
def g (x : ℤ) : ℤ := x^2 + 5*x + 3

/-- The set T of integers from 0 to 30 -/
def T : Set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

/-- Theorem: There are no integers t in T such that g(t) is divisible by 10 -/
theorem no_divisible_by_ten : ∀ t ∈ T, ¬(g t % 10 = 0) := by sorry

end NUMINAMATH_CALUDE_no_divisible_by_ten_l1345_134525


namespace NUMINAMATH_CALUDE_pencil_distribution_l1345_134536

theorem pencil_distribution (num_students : ℕ) (total_pencils : ℕ) (pencils_per_dozen : ℕ) : 
  num_students = 46 → 
  total_pencils = 2208 → 
  pencils_per_dozen = 12 →
  (total_pencils / num_students) / pencils_per_dozen = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1345_134536


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1345_134551

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n, a (n + 1) = a n * r

/-- The fourth term of a geometric sequence with first term 3 and third term 75 is 375. -/
theorem fourth_term_of_geometric_sequence (a : ℕ → ℕ) (h : IsGeometricSequence a) 
    (h1 : a 1 = 3) (h3 : a 3 = 75) : a 4 = 375 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1345_134551


namespace NUMINAMATH_CALUDE_clay_target_sequences_l1345_134509

theorem clay_target_sequences (n : ℕ) (a b c : ℕ) 
  (h1 : n = 8) 
  (h2 : a = 3) 
  (h3 : b = 3) 
  (h4 : c = 2) 
  (h5 : a + b + c = n) : 
  (Nat.factorial n) / (Nat.factorial a * Nat.factorial b * Nat.factorial c) = 560 :=
by sorry

end NUMINAMATH_CALUDE_clay_target_sequences_l1345_134509


namespace NUMINAMATH_CALUDE_opposite_to_light_green_is_red_l1345_134585

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

end NUMINAMATH_CALUDE_opposite_to_light_green_is_red_l1345_134585


namespace NUMINAMATH_CALUDE_bill_lines_count_l1345_134588

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

end NUMINAMATH_CALUDE_bill_lines_count_l1345_134588


namespace NUMINAMATH_CALUDE_determinant_zero_l1345_134598

theorem determinant_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_determinant_zero_l1345_134598


namespace NUMINAMATH_CALUDE_cyclical_fraction_bounds_l1345_134559

theorem cyclical_fraction_bounds (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  1 < (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) ∧ 
  (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclical_fraction_bounds_l1345_134559


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l1345_134521

theorem right_triangle_arithmetic_sequence (a b c : ℕ) : 
  a < b ∧ b < c →                        -- sides form an increasing sequence
  a + b + c = 840 →                      -- perimeter is 840
  b - a = c - b →                        -- sides form an arithmetic sequence
  a^2 + b^2 = c^2 →                      -- it's a right triangle (Pythagorean theorem)
  c = 350 := by sorry                    -- largest side is 350

end NUMINAMATH_CALUDE_right_triangle_arithmetic_sequence_l1345_134521


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l1345_134501

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ b : ℝ, b < 0 ∧ x^2 + (y - b)^2 = 25 ∧ 3 - b = 5

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  ∃ k : ℝ, y + 3 = k * (x + 3)

-- Define the chord length
def chord_length (x y : ℝ) : Prop :=
  ∃ (c_x c_y : ℝ), circle_C c_x c_y ∧
  ((x - c_x)^2 + (y - c_y)^2) - (((-3) - c_x)^2 + ((-3) - c_y)^2) = 20

-- Theorem statement
theorem circle_and_line_properties :
  ∀ x y : ℝ,
  circle_C x y →
  line_l x y →
  chord_length x y →
  (x^2 + (y + 2)^2 = 25) ∧
  ((x + 2*y + 9 = 0) ∨ (2*x - y + 3 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l1345_134501


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1345_134510

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (-2 + Complex.I) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1345_134510


namespace NUMINAMATH_CALUDE_angle_ABC_equals_cos_inverse_l1345_134539

/-- The angle ABC given three points A, B, and C in 3D space -/
def angle_ABC (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Converts radians to degrees -/
def to_degrees (x : ℝ) : ℝ := sorry

theorem angle_ABC_equals_cos_inverse :
  let A : ℝ × ℝ × ℝ := (-3, 1, 5)
  let B : ℝ × ℝ × ℝ := (-4, -2, 1)
  let C : ℝ × ℝ × ℝ := (-5, -2, 2)
  to_degrees (angle_ABC A B C) = Real.arccos ((3 * Real.sqrt 13) / 26) := by sorry

end NUMINAMATH_CALUDE_angle_ABC_equals_cos_inverse_l1345_134539


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1345_134568

theorem triangle_angle_problem (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ 2*x < 180 ∧ 
  (x + 2*x + 30 = 180) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1345_134568


namespace NUMINAMATH_CALUDE_total_salary_is_616_l1345_134506

/-- The salary of employee N in dollars per week -/
def salary_N : ℝ := 280

/-- The ratio of M's salary to N's salary -/
def salary_ratio : ℝ := 1.2

/-- The salary of employee M in dollars per week -/
def salary_M : ℝ := salary_ratio * salary_N

/-- The total amount paid to both employees per week -/
def total_salary : ℝ := salary_M + salary_N

theorem total_salary_is_616 : total_salary = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_is_616_l1345_134506


namespace NUMINAMATH_CALUDE_constant_k_value_l1345_134554

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4) → k = -16 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l1345_134554


namespace NUMINAMATH_CALUDE_problem_solution_l1345_134534

theorem problem_solution (x y z : ℝ) 
  (h1 : (x + y)^2 + (y + z)^2 + (x + z)^2 = 94)
  (h2 : (x - y)^2 + (y - z)^2 + (x - z)^2 = 26) :
  (x * y + y * z + x * z = 17) ∧
  ((x + 2*y + 3*z)^2 + (y + 2*z + 3*x)^2 + (z + 2*x + 3*y)^2 = 794) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1345_134534


namespace NUMINAMATH_CALUDE_distance_from_A_to_x_axis_l1345_134584

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (y : ℝ) : ℝ := |y|

/-- Point A in the Cartesian coordinate system -/
def point_A : ℝ × ℝ := (-5, -9)

theorem distance_from_A_to_x_axis :
  distance_to_x_axis (point_A.2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_A_to_x_axis_l1345_134584


namespace NUMINAMATH_CALUDE_certain_number_theorem_l1345_134532

theorem certain_number_theorem (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_theorem_l1345_134532


namespace NUMINAMATH_CALUDE_john_spent_625_l1345_134540

/-- The amount John spent on purchases with a coupon -/
def johnsSpending (vacuumPrice dishwasherPrice couponValue : ℕ) : ℕ :=
  vacuumPrice + dishwasherPrice - couponValue

/-- Theorem stating that John spent $625 -/
theorem john_spent_625 :
  johnsSpending 250 450 75 = 625 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_625_l1345_134540


namespace NUMINAMATH_CALUDE_matrix_power_4_l1345_134527

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_4 :
  A ^ 4 = !![(-4 : ℝ), 0; 0, -4] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l1345_134527


namespace NUMINAMATH_CALUDE_train_travel_time_l1345_134591

/-- Calculates the actual travel time in hours given the total travel time and break time in minutes. -/
def actualTravelTimeInHours (totalTravelTime breakTime : ℕ) : ℚ :=
  (totalTravelTime - breakTime) / 60

/-- Theorem stating that given a total travel time of 270 minutes with a 30-minute break,
    the actual travel time is 4 hours. -/
theorem train_travel_time :
  actualTravelTimeInHours 270 30 = 4 := by sorry

end NUMINAMATH_CALUDE_train_travel_time_l1345_134591


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l1345_134589

theorem simplify_nested_roots (a : ℝ) : 
  (((a^16)^(1/3))^(1/4))^3 * (((a^16)^(1/4))^(1/3))^2 = a^(20/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l1345_134589


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_c_equals_five_l1345_134560

theorem infinite_solutions_iff_c_equals_five (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_c_equals_five_l1345_134560


namespace NUMINAMATH_CALUDE_chess_tournament_draws_l1345_134511

/-- Represents a chess tournament with a fixed number of participants. -/
structure ChessTournament where
  n : ℕ  -- number of participants
  lists : Fin n → Fin 12 → Set (Fin n)  -- lists[i][j] is the jth list of participant i
  
  list_rule_1 : ∀ i, lists i 0 = {i}
  list_rule_2 : ∀ i j, j > 0 → lists i j ⊇ lists i (j-1)
  list_rule_12 : ∀ i, lists i 11 ≠ lists i 10

/-- The number of draws in the tournament. -/
def num_draws (t : ChessTournament) : ℕ :=
  (t.n.choose 2) - t.n

theorem chess_tournament_draws (t : ChessTournament) (h : t.n = 12) : 
  num_draws t = 54 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_draws_l1345_134511


namespace NUMINAMATH_CALUDE_sqrt_360000_l1345_134570

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_l1345_134570


namespace NUMINAMATH_CALUDE_side_ratio_not_imply_right_triangle_l1345_134581

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

end NUMINAMATH_CALUDE_side_ratio_not_imply_right_triangle_l1345_134581


namespace NUMINAMATH_CALUDE_notepad_cost_l1345_134552

/-- Given the total cost of notepads, pages per notepad, and total pages bought,
    calculate the cost of each notepad. -/
theorem notepad_cost (total_cost : ℚ) (pages_per_notepad : ℕ) (total_pages : ℕ) :
  total_cost = 10 →
  pages_per_notepad = 60 →
  total_pages = 480 →
  (total_cost / (total_pages / pages_per_notepad : ℚ)) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_notepad_cost_l1345_134552


namespace NUMINAMATH_CALUDE_min_k_value_l1345_134503

-- Define the function f(x) = x(ln x + 1) / (x - 2)
noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1) / (x - 2)

-- State the theorem
theorem min_k_value : 
  (∃ x₀ : ℝ, x₀ > 2 ∧ ∃ k : ℕ, k > 0 ∧ k * (x₀ - 2) > x₀ * (Real.log x₀ + 1)) → 
  (∀ k : ℕ, k > 0 → (∃ x : ℝ, x > 2 ∧ k * (x - 2) > x * (Real.log x + 1)) → k ≥ 5) ∧
  (∃ x : ℝ, x > 2 ∧ 5 * (x - 2) > x * (Real.log x + 1)) :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l1345_134503


namespace NUMINAMATH_CALUDE_probability_one_each_color_l1345_134516

def total_marbles : ℕ := 7
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def yellow_marbles : ℕ := 1
def marbles_drawn : ℕ := 3

theorem probability_one_each_color (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ) (drawn : ℕ)
  (h1 : total = red + blue + green + yellow)
  (h2 : drawn = 3)
  (h3 : red = 2)
  (h4 : blue = 2)
  (h5 : green = 2)
  (h6 : yellow = 1) :
  (red * blue * green : ℚ) / Nat.choose total drawn = 8 / 35 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_each_color_l1345_134516


namespace NUMINAMATH_CALUDE_even_function_extension_l1345_134555

/-- Given a real-valued function f that is even and defined as ln(x^2 - 2x + 2) for non-negative x,
    prove that f(x) = ln(x^2 + 2x + 2) for negative x -/
theorem even_function_extension (f : ℝ → ℝ) 
    (h_even : ∀ x, f x = f (-x))
    (h_non_neg : ∀ x ≥ 0, f x = Real.log (x^2 - 2*x + 2)) :
    ∀ x < 0, f x = Real.log (x^2 + 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_even_function_extension_l1345_134555


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l1345_134545

-- Part 1
theorem calculation_proof :
  0.01⁻¹ + (-1 - 2/7)^0 - Real.sqrt 9 = 98 := by sorry

-- Part 2
theorem equation_solution_proof :
  ∀ x : ℝ, (2 / (x - 3) = 3 / (x - 2)) ↔ (x = 5) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l1345_134545


namespace NUMINAMATH_CALUDE_function_property_implies_range_l1345_134531

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

end NUMINAMATH_CALUDE_function_property_implies_range_l1345_134531


namespace NUMINAMATH_CALUDE_total_fish_count_l1345_134573

theorem total_fish_count (micah kenneth matthias gabrielle : ℕ) : 
  micah = 7 →
  kenneth = 3 * micah →
  matthias = kenneth - 15 →
  gabrielle = 2 * (micah + kenneth + matthias) →
  micah + kenneth + matthias + gabrielle = 102 := by
sorry

end NUMINAMATH_CALUDE_total_fish_count_l1345_134573


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l1345_134569

theorem factorization_of_2m_squared_minus_8 (m : ℝ) :
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l1345_134569


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1345_134575

theorem cube_root_of_negative_eight :
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1345_134575


namespace NUMINAMATH_CALUDE_credit_card_more_beneficial_l1345_134524

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 20000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.01

/-- Represents the annual interest rate on the debit card -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of days in a year for interest calculation -/
def days_in_year : ℝ := 360

/-- Represents the minimum number of days for credit card to be more beneficial -/
def min_days : ℕ := 31

theorem credit_card_more_beneficial :
  ∀ N : ℕ,
  N ≥ min_days →
  (purchase_amount * credit_cashback_rate) + 
  (purchase_amount * annual_interest_rate * N / days_in_year) >
  purchase_amount * debit_cashback_rate :=
sorry

end NUMINAMATH_CALUDE_credit_card_more_beneficial_l1345_134524


namespace NUMINAMATH_CALUDE_marbles_given_to_mary_l1345_134580

/-- Given that Dan initially had 64 marbles and now has 50 marbles,
    prove that he gave 14 marbles to Mary. -/
theorem marbles_given_to_mary (initial_marbles : ℕ) (current_marbles : ℕ)
    (h1 : initial_marbles = 64)
    (h2 : current_marbles = 50) :
    initial_marbles - current_marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_mary_l1345_134580


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_l1345_134562

/-- Given a regular tetrahedron with base area S and volume V,
    the radius R of its inscribed sphere is equal to 3V/(4S) -/
theorem inscribed_sphere_radius_regular_tetrahedron
  (S V : ℝ) (h_S : S > 0) (h_V : V > 0) :
  ∃ R : ℝ, R = (3 * V) / (4 * S) ∧ R > 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_l1345_134562


namespace NUMINAMATH_CALUDE_valid_configuration_iff_n_eq_4_l1345_134550

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  values : Fin n → ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- The condition that no three points are collinear -/
def noThreeCollinear (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) ≠ 0

/-- The condition that the area of any triangle equals the sum of corresponding values -/
def areaEqualsSumOfValues (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) =
      config.values i + config.values j + config.values k

/-- The main theorem stating that a valid configuration exists if and only if n = 4 -/
theorem valid_configuration_iff_n_eq_4 :
  (∃ (config : PointConfiguration n), n > 3 ∧ noThreeCollinear config ∧ areaEqualsSumOfValues config) ↔
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_configuration_iff_n_eq_4_l1345_134550


namespace NUMINAMATH_CALUDE_integral_points_on_line_segment_l1345_134518

def is_on_line_segment (x y : ℤ) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧
  x = (22 : ℤ) + t * ((16 : ℤ) - (22 : ℤ)) ∧
  y = (12 : ℤ) + t * ((17 : ℤ) - (12 : ℤ))

theorem integral_points_on_line_segment :
  ∃! p : ℤ × ℤ, 
    is_on_line_segment p.1 p.2 ∧
    10 ≤ p.1 ∧ p.1 ≤ 30 ∧
    10 ≤ p.2 ∧ p.2 ≤ 30 :=
sorry

end NUMINAMATH_CALUDE_integral_points_on_line_segment_l1345_134518


namespace NUMINAMATH_CALUDE_sophie_germain_characterization_l1345_134574

/-- A prime number p is a Sophie Germain prime if 2p + 1 is also prime. -/
def SophieGermainPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

/-- The product of all possible units digits of Sophie Germain primes greater than 6 is 189. -/
axiom units_digit_product : ∃ (S : Finset ℕ), (∀ n ∈ S, n > 6 ∧ SophieGermainPrime n) ∧
  (Finset.prod S (λ n => n % 10) = 189)

theorem sophie_germain_characterization (p : ℕ) (h_prime : Nat.Prime p) (h_greater : p > 6) :
  SophieGermainPrime p ↔ Nat.Prime (2 * p + 1) :=
sorry

end NUMINAMATH_CALUDE_sophie_germain_characterization_l1345_134574


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1345_134504

theorem circle_intersection_theorem (O₁ O₂ T A B : ℝ × ℝ) : 
  let d := Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2)
  let r₁ := 4
  let r₂ := 6
  d ≥ 6 →
  (∃ C : ℝ × ℝ, (C.1 - O₁.1)^2 + (C.2 - O₁.2)^2 = r₁^2 ∧ 
               (C.1 - O₂.1)^2 + (C.2 - O₂.2)^2 = r₂^2) →
  (A.1 - O₁.1)^2 + (A.2 - O₁.2)^2 = r₂^2 →
  (B.1 - O₁.1)^2 + (B.2 - O₁.2)^2 = r₁^2 →
  Real.sqrt ((A.1 - T.1)^2 + (A.2 - T.2)^2) = 
    1/3 * Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) →
  Real.sqrt ((B.1 - T.1)^2 + (B.2 - T.2)^2) = 
    2/3 * Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1345_134504


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l1345_134579

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)) + (y^2 / (x - 2)) ≥ 12 ∧
  ((x^2 / (y - 2)) + (y^2 / (x - 2)) = 12 ↔ x = 4 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l1345_134579


namespace NUMINAMATH_CALUDE_pencils_and_pens_count_pencils_and_pens_count_proof_l1345_134577

theorem pencils_and_pens_count : ℕ → ℕ → Prop :=
  fun initial_pencils initial_pens =>
    (initial_pencils : ℚ) / initial_pens = 4 / 5 ∧
    ((initial_pencils + 1 : ℚ) / (initial_pens - 1) = 7 / 8) →
    initial_pencils + initial_pens = 45

-- The proof goes here
theorem pencils_and_pens_count_proof : ∃ (p q : ℕ), pencils_and_pens_count p q :=
  sorry

end NUMINAMATH_CALUDE_pencils_and_pens_count_pencils_and_pens_count_proof_l1345_134577


namespace NUMINAMATH_CALUDE_exists_value_not_taken_by_phi_at_odd_l1345_134537

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem exists_value_not_taken_by_phi_at_odd :
  ∃ m : ℕ, ∀ n : ℕ, isOdd n → phi n ≠ m := by sorry

end NUMINAMATH_CALUDE_exists_value_not_taken_by_phi_at_odd_l1345_134537


namespace NUMINAMATH_CALUDE_second_student_male_probability_l1345_134594

/-- The probability that the second student to leave is male, given 2 male and 2 female students -/
def probability_second_male (num_male num_female : ℕ) : ℚ :=
  if num_male = 2 ∧ num_female = 2 then 1/6 else 0

/-- Theorem stating that the probability of the second student to leave being male is 1/6 -/
theorem second_student_male_probability :
  probability_second_male 2 2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_second_student_male_probability_l1345_134594


namespace NUMINAMATH_CALUDE_soccer_handshakes_l1345_134513

theorem soccer_handshakes (team_size : Nat) (referee_count : Nat) : 
  team_size = 11 → referee_count = 3 → 
  (team_size * team_size) + (2 * team_size * referee_count) = 187 := by
  sorry

#check soccer_handshakes

end NUMINAMATH_CALUDE_soccer_handshakes_l1345_134513


namespace NUMINAMATH_CALUDE_intersection_product_l1345_134528

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

end NUMINAMATH_CALUDE_intersection_product_l1345_134528


namespace NUMINAMATH_CALUDE_pizza_consumption_l1345_134522

theorem pizza_consumption (rachel_pizza : ℕ) (bella_pizza : ℕ)
  (h1 : rachel_pizza = 598)
  (h2 : bella_pizza = 354) :
  rachel_pizza + bella_pizza = 952 :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_l1345_134522


namespace NUMINAMATH_CALUDE_power_equation_solution_l1345_134548

theorem power_equation_solution (m : ℕ) : (4 : ℝ)^m * 2^3 = 8^5 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1345_134548


namespace NUMINAMATH_CALUDE_negation_quadratic_inequality_l1345_134538

theorem negation_quadratic_inequality (x : ℝ) :
  (x^2 + x - 6 < 0) → (x ≤ 2) :=
sorry

#check negation_quadratic_inequality

end NUMINAMATH_CALUDE_negation_quadratic_inequality_l1345_134538


namespace NUMINAMATH_CALUDE_missing_number_is_twelve_l1345_134561

theorem missing_number_is_twelve : ∃ x : ℕ, 1234562 - x * 3 * 2 = 1234490 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_twelve_l1345_134561


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l1345_134530

/-- Proves that a tax reduction of 20% results in a 4% revenue decrease when consumption increases by 20% -/
theorem tax_reduction_theorem (T C : ℝ) (x : ℝ) 
  (h1 : x > 0)
  (h2 : T > 0)
  (h3 : C > 0)
  (h4 : (T - x / 100 * T) * (C + 20 / 100 * C) = 0.96 * T * C) :
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l1345_134530


namespace NUMINAMATH_CALUDE_let_go_to_catch_not_specific_analysis_l1345_134578

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

end NUMINAMATH_CALUDE_let_go_to_catch_not_specific_analysis_l1345_134578


namespace NUMINAMATH_CALUDE_det_A_eq_140_l1345_134544

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, -2; 8, 5, -4; 1, 3, 6]

theorem det_A_eq_140 : Matrix.det A = 140 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_140_l1345_134544


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_l1345_134508

/-- The fixed cost for producing products given total cost, marginal cost, and number of products. -/
theorem fixed_cost_calculation (total_cost marginal_cost : ℝ) (n : ℕ) :
  total_cost = 16000 →
  marginal_cost = 200 →
  n = 20 →
  total_cost = (marginal_cost * n) + 12000 :=
by sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_l1345_134508


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1345_134507

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + a*c = 131 := by sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1345_134507


namespace NUMINAMATH_CALUDE_second_shirt_buttons_l1345_134565

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

end NUMINAMATH_CALUDE_second_shirt_buttons_l1345_134565


namespace NUMINAMATH_CALUDE_brianna_remaining_money_l1345_134599

theorem brianna_remaining_money
  (m n c : ℝ)
  (h1 : m > 0)
  (h2 : n > 0)
  (h3 : c > 0)
  (h4 : (1/4) * m = (1/2) * n * c) :
  m - ((1/2) * n * c) - ((1/10) * m) = (2/5) * m :=
sorry

end NUMINAMATH_CALUDE_brianna_remaining_money_l1345_134599


namespace NUMINAMATH_CALUDE_right_triangle_area_l1345_134500

theorem right_triangle_area (a b c : ℝ) (h : a > 0) : 
  a * a = 2 * b * b →  -- 45-45-90 triangle condition
  b = 4 →              -- altitude to hypotenuse is 4
  c = a / 2 →          -- c is half of hypotenuse
  (1/2) * a * b = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1345_134500


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1345_134517

-- Define the proposition
theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ x y z : ℝ, x > y ∧ x * z^2 ≤ y * z^2) ∧
  (∀ x y z : ℝ, x * z^2 > y * z^2 → x > y) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1345_134517


namespace NUMINAMATH_CALUDE_women_at_soccer_game_l1345_134564

theorem women_at_soccer_game (adults : ℕ) (adult_women : ℕ) (student_surplus : ℕ) (male_students : ℕ)
  (h1 : adults = 1518)
  (h2 : adult_women = 536)
  (h3 : student_surplus = 525)
  (h4 : male_students = 1257) :
  adult_women + ((adults + student_surplus) - male_students) = 1322 :=
by sorry

end NUMINAMATH_CALUDE_women_at_soccer_game_l1345_134564


namespace NUMINAMATH_CALUDE_chessboard_square_selection_l1345_134592

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents the number of ways to choose squares from a chessboard -/
def choose_squares (board : Chessboard) (num_squares : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to choose 60 squares from an 11x11 chessboard
    with no adjacent squares is 62 -/
theorem chessboard_square_selection :
  let board : Chessboard := ⟨11⟩
  choose_squares board 60 = 62 := by sorry

end NUMINAMATH_CALUDE_chessboard_square_selection_l1345_134592


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1345_134563

/-- The equation of the common chord of two intersecting circles -/
theorem common_chord_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 10) ∧ ((x-1)^2 + (y-3)^2 = 10) → x + 3*y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1345_134563


namespace NUMINAMATH_CALUDE_cookies_eaten_l1345_134546

theorem cookies_eaten (initial : Real) (remaining : Real) (eaten : Real) : 
  initial = 28.5 → remaining = 7.25 → eaten = initial - remaining → eaten = 21.25 :=
by sorry

end NUMINAMATH_CALUDE_cookies_eaten_l1345_134546


namespace NUMINAMATH_CALUDE_sin_cos_relation_l1345_134541

theorem sin_cos_relation (x : Real) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x ^ 2 - Real.cos x ^ 2 = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l1345_134541
