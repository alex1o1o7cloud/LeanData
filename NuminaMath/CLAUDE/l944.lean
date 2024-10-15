import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_a_l944_94455

theorem max_value_of_a (a b c : ℝ) (sum_eq : a + b + c = 3) (prod_sum_eq : a * b + a * c + b * c = 3) :
  a ≤ 1 + Real.sqrt 2 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 3 ∧ a₀ * b₀ + a₀ * c₀ + b₀ * c₀ = 3 ∧ a₀ = 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l944_94455


namespace NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l944_94435

theorem square_difference_from_sum_and_difference (x y : ℝ) 
  (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l944_94435


namespace NUMINAMATH_CALUDE_pentagon_section_probability_l944_94405

/-- The probability of an arrow stopping in a specific section of a pentagon divided into 5 equal sections is 1/5. -/
theorem pentagon_section_probability :
  ∀ (n : ℕ) (sections : ℕ),
    sections = 5 →
    n ≤ sections →
    (n : ℚ) / (sections : ℚ) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_section_probability_l944_94405


namespace NUMINAMATH_CALUDE_open_box_volume_l944_94465

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_square_side : ℝ)
  (h_sheet_length : sheet_length = 46)
  (h_sheet_width : sheet_width = 36)
  (h_cut_square_side : cut_square_side = 8) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 4800 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l944_94465


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l944_94491

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = 3 * a n) 
  (m n : ℕ) 
  (h_product : a m * a n = 9 * (a 2)^2) :
  (∀ k l : ℕ, 2 / k + 1 / (2 * l) ≥ 3 / 4) ∧ 
  (∃ k l : ℕ, 2 / k + 1 / (2 * l) = 3 / 4) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l944_94491


namespace NUMINAMATH_CALUDE_cereal_eating_time_l944_94474

/-- The time it takes for Mr. Fat and Mr. Thin to eat 5 pounds of cereal together -/
theorem cereal_eating_time (fat_rate thin_rate : ℚ) (total_cereal : ℚ) : 
  fat_rate = 1 / 15 →
  thin_rate = 1 / 45 →
  total_cereal = 5 →
  (total_cereal / (fat_rate + thin_rate) : ℚ) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l944_94474


namespace NUMINAMATH_CALUDE_circle_theorem_l944_94422

structure Circle where
  center : Point
  radius : ℝ

structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

def parallel (l1 l2 : Line) : Prop := sorry

def diameter (c : Circle) (l : Line) : Prop := sorry

def inscribed_angle (c : Circle) (a : Angle) : Prop := sorry

def angle_measure (a : Angle) : ℝ := sorry

theorem circle_theorem (c : Circle) (F B D C A : Point) 
  (FB DC AB FD : Line) (AFB ABF BCD : Angle) :
  diameter c FB →
  parallel FB DC →
  parallel AB FD →
  angle_measure AFB / angle_measure ABF = 3 / 4 →
  inscribed_angle c BCD →
  angle_measure BCD = 330 / 7 := by sorry

end NUMINAMATH_CALUDE_circle_theorem_l944_94422


namespace NUMINAMATH_CALUDE_angle_sum_equality_counterexample_l944_94489

theorem angle_sum_equality_counterexample :
  ∃ (angle1 angle2 : ℝ), 
    angle1 + angle2 = 90 ∧ angle1 = angle2 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_equality_counterexample_l944_94489


namespace NUMINAMATH_CALUDE_dartboard_central_angle_l944_94449

/-- The measure of the central angle of one section in a circular dartboard -/
def central_angle_measure (num_sections : ℕ) (section_probability : ℚ) : ℚ :=
  360 * section_probability

/-- Theorem: The central angle measure for a circular dartboard with 8 equal sections
    and 1/8 probability of landing in each section is 45 degrees -/
theorem dartboard_central_angle :
  central_angle_measure 8 (1/8) = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angle_l944_94449


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l944_94450

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of at least two dice showing the same number when rolling 5 fair 8-sided dice -/
theorem probability_at_least_two_same (numSides : ℕ) (numDice : ℕ) :
  numSides = 8 → numDice = 5 →
  (1 - (numSides.factorial / (numSides - numDice).factorial) / numSides ^ numDice : ℚ) = 6512 / 8192 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l944_94450


namespace NUMINAMATH_CALUDE_absolute_value_sum_greater_than_one_l944_94423

theorem absolute_value_sum_greater_than_one (x y : ℝ) :
  y ≤ -2 → abs x + abs y > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_greater_than_one_l944_94423


namespace NUMINAMATH_CALUDE_vector_angle_problem_l944_94425

theorem vector_angle_problem (α β : Real) (a b : Fin 2 → Real) :
  a 0 = Real.cos α ∧ a 1 = Real.sin α ∧
  b 0 = Real.cos β ∧ b 1 = Real.sin β ∧
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 2 * Real.sqrt 5 / 5 ∧
  0 < α ∧ α < π / 2 ∧
  -π / 2 < β ∧ β < 0 ∧
  Real.sin β = -5 / 13 →
  Real.cos (α - β) = 3 / 5 ∧ Real.sin α = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l944_94425


namespace NUMINAMATH_CALUDE_expected_profit_is_37_l944_94407

/-- Represents the possible product grades produced by the machine -/
inductive ProductGrade
  | GradeA
  | GradeB
  | Defective

/-- Returns the profit for a given product grade -/
def profit (grade : ProductGrade) : ℝ :=
  match grade with
  | ProductGrade.GradeA => 50
  | ProductGrade.GradeB => 30
  | ProductGrade.Defective => -20

/-- Returns the probability of producing a given product grade -/
def probability (grade : ProductGrade) : ℝ :=
  match grade with
  | ProductGrade.GradeA => 0.6
  | ProductGrade.GradeB => 0.3
  | ProductGrade.Defective => 0.1

/-- Calculates the expected profit -/
def expectedProfit : ℝ :=
  (profit ProductGrade.GradeA * probability ProductGrade.GradeA) +
  (profit ProductGrade.GradeB * probability ProductGrade.GradeB) +
  (profit ProductGrade.Defective * probability ProductGrade.Defective)

theorem expected_profit_is_37 : expectedProfit = 37 := by
  sorry

end NUMINAMATH_CALUDE_expected_profit_is_37_l944_94407


namespace NUMINAMATH_CALUDE_james_beat_record_by_296_l944_94440

/-- Calculates the total points scored by James in a football season -/
def james_total_points (
  touchdowns_per_game : ℕ)
  (touchdown_points : ℕ)
  (games_in_season : ℕ)
  (two_point_conversions : ℕ)
  (field_goals : ℕ)
  (field_goal_points : ℕ)
  (extra_point_attempts : ℕ)
  (bonus_touchdown_sets : ℕ)
  (bonus_touchdowns_per_set : ℕ)
  (bonus_multiplier : ℕ) : ℕ :=
  let regular_touchdown_points := touchdowns_per_game * games_in_season * touchdown_points
  let bonus_touchdown_points := bonus_touchdown_sets * bonus_touchdowns_per_set * touchdown_points * bonus_multiplier
  let two_point_conversion_points := two_point_conversions * 2
  let field_goal_points := field_goals * field_goal_points
  let extra_point_points := extra_point_attempts
  regular_touchdown_points + bonus_touchdown_points + two_point_conversion_points + field_goal_points + extra_point_points

/-- Theorem stating that James beat the old record by 296 points -/
theorem james_beat_record_by_296 :
  james_total_points 4 6 15 6 8 3 20 5 3 2 - 300 = 296 := by
  sorry

end NUMINAMATH_CALUDE_james_beat_record_by_296_l944_94440


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l944_94419

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 225 * π → 2 * π * r^2 + π * r^2 = 675 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l944_94419


namespace NUMINAMATH_CALUDE_matrix_power_not_identity_l944_94496

/-- Given a 5x5 complex matrix A with trace 0 and invertible I₅ - A, A⁵ ≠ I₅ -/
theorem matrix_power_not_identity
  (A : Matrix (Fin 5) (Fin 5) ℂ)
  (h_trace : Matrix.trace A = 0)
  (h_invertible : IsUnit (1 - A)) :
  A ^ 5 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_not_identity_l944_94496


namespace NUMINAMATH_CALUDE_crow_speed_l944_94488

/-- Calculates the speed of a crow flying between its nest and a ditch -/
theorem crow_speed (distance : ℝ) (trips : ℕ) (time : ℝ) : 
  distance = 200 → 
  trips = 15 → 
  time = 1.5 → 
  (2 * distance * trips) / (time * 1000) = 4 :=
by sorry

end NUMINAMATH_CALUDE_crow_speed_l944_94488


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l944_94472

theorem roots_quadratic_equation (m p q c : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + c/b)^2 - p*(a + c/b) + q = 0) →
  ((b + c/a)^2 - p*(b + c/a) + q = 0) →
  (q = 3 + 2*c + c^2/3) :=
by sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l944_94472


namespace NUMINAMATH_CALUDE_rebus_solution_l944_94439

theorem rebus_solution :
  ∃! (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    A + B * 10 + B = C * 10 + A ∧
    C = 1 ∧ B = 9 ∧ A = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l944_94439


namespace NUMINAMATH_CALUDE_prove_weekly_pay_l944_94480

def weekly_pay_problem (y_pay : ℝ) (x_percent : ℝ) : Prop :=
  let x_pay := x_percent * y_pay
  let total_pay := x_pay + y_pay
  y_pay = 263.64 ∧ x_percent = 1.2 → total_pay = 580.008

theorem prove_weekly_pay : weekly_pay_problem 263.64 1.2 := by
  sorry

end NUMINAMATH_CALUDE_prove_weekly_pay_l944_94480


namespace NUMINAMATH_CALUDE_inequality_proof_l944_94482

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l944_94482


namespace NUMINAMATH_CALUDE_books_loaned_out_correct_loaned_books_l944_94451

/-- Proves that the number of books loaned out is 160 given the initial and final book counts and return rate -/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) : ℕ :=
  let loaned_books := (initial_books - final_books) / (1 - return_rate)
  160

/-- The number of books loaned out is 160 -/
theorem correct_loaned_books : books_loaned_out 300 244 (65/100) = 160 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_correct_loaned_books_l944_94451


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_triple_expansion_l944_94457

theorem coefficient_of_x_in_triple_expansion (x : ℝ) : 
  let expansion := (1 + x)^3 + (1 + x)^3 + (1 + x)^3
  ∃ a b c d : ℝ, expansion = a + 9*x + b*x^2 + c*x^3 + d*x^4 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_triple_expansion_l944_94457


namespace NUMINAMATH_CALUDE_sum_of_parts_l944_94432

theorem sum_of_parts (x y : ℤ) (h1 : x + y = 54) (h2 : y = 34) : 10 * x + 22 * y = 948 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l944_94432


namespace NUMINAMATH_CALUDE_james_painting_fraction_l944_94420

/-- If a person can paint a wall in a given time, this function calculates
    the fraction of the wall they can paint in a shorter time period. -/
def fractionPainted (totalTime minutes : ℚ) : ℚ :=
  minutes / totalTime

theorem james_painting_fraction :
  fractionPainted 60 15 = 1/4 := by sorry

end NUMINAMATH_CALUDE_james_painting_fraction_l944_94420


namespace NUMINAMATH_CALUDE_literature_class_b_count_l944_94436

/-- In a literature class with the given grade distribution, prove the number of B grades. -/
theorem literature_class_b_count (total : ℕ) (p_a p_b p_c : ℝ) (b_count : ℕ) : 
  total = 25 →
  p_a = 0.8 * p_b →
  p_c = 1.2 * p_b →
  p_a + p_b + p_c = 1 →
  b_count = ⌊(total : ℝ) / 3⌋ →
  b_count = 8 := by
sorry

end NUMINAMATH_CALUDE_literature_class_b_count_l944_94436


namespace NUMINAMATH_CALUDE_fundraising_average_contribution_l944_94456

/-- Proves that the average contribution required from the remaining targeted people
    is $400 / 0.36, given the conditions of the fundraising problem. -/
theorem fundraising_average_contribution
  (total_amount : ℝ) 
  (total_people : ℝ) 
  (h1 : total_amount > 0)
  (h2 : total_people > 0)
  (h3 : 0.6 * total_amount = 0.4 * total_people * 400) :
  (0.4 * total_amount) / (0.6 * total_people) = 400 / 0.36 := by
sorry

end NUMINAMATH_CALUDE_fundraising_average_contribution_l944_94456


namespace NUMINAMATH_CALUDE_solve_equation_l944_94492

theorem solve_equation (p q x : ℚ) : 
  (3 / 4 : ℚ) = p / 60 ∧ 
  (3 / 4 : ℚ) = (p + q) / 100 ∧ 
  (3 / 4 : ℚ) = (x - q) / 140 → 
  x = 135 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l944_94492


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l944_94452

theorem unique_solution_for_exponential_equation :
  ∀ (a b n p : ℕ), 
    p.Prime → 
    2^a + p^b = n^(p-1) → 
    (a = 0 ∧ b = 1 ∧ n = 2 ∧ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l944_94452


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l944_94448

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_equals_four :
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l944_94448


namespace NUMINAMATH_CALUDE_transportation_theorem_l944_94441

/-- Represents the transportation problem between cities A, B, C, and D. -/
structure TransportationProblem where
  supplies_C : ℝ
  supplies_D : ℝ
  cost_C_to_A : ℝ
  cost_C_to_B : ℝ
  cost_D_to_A : ℝ
  cost_D_to_B : ℝ
  x : ℝ  -- Amount transported from D to B

/-- The total transportation cost as a function of x -/
def total_cost (p : TransportationProblem) : ℝ :=
  p.cost_C_to_A * (200 - (p.supplies_D - p.x)) + 
  p.cost_C_to_B * (300 - p.x) + 
  p.cost_D_to_A * (p.supplies_D - p.x) + 
  p.cost_D_to_B * p.x

theorem transportation_theorem (p : TransportationProblem) 
  (h1 : p.supplies_C = 240)
  (h2 : p.supplies_D = 260)
  (h3 : p.cost_C_to_A = 20)
  (h4 : p.cost_C_to_B = 25)
  (h5 : p.cost_D_to_A = 15)
  (h6 : p.cost_D_to_B = 30)
  (h7 : 60 ≤ p.x ∧ p.x ≤ 260) : 
  (∃ (w : ℝ), w = total_cost p ∧ w = 10 * p.x + 10200) ∧
  (∀ (m : ℝ), (∀ (x : ℝ), 60 ≤ x → x ≤ 260 → 
    (10 - m) * x + 10200 ≥ 10320) ↔ (0 < m ∧ m ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_transportation_theorem_l944_94441


namespace NUMINAMATH_CALUDE_circle_circumference_l944_94481

theorem circle_circumference (r : ℝ) (d : ℝ) (C : ℝ) :
  (d = 2 * r) → (C = π * d ∨ C = 2 * π * r) :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l944_94481


namespace NUMINAMATH_CALUDE_girls_in_class_l944_94483

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 35) (h2 : ratio_girls = 3) (h3 : ratio_boys = 4) : 
  (ratio_girls * total) / (ratio_girls + ratio_boys) = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l944_94483


namespace NUMINAMATH_CALUDE_top_three_probability_correct_l944_94413

/-- Represents a knockout tournament with 64 teams. -/
structure Tournament :=
  (teams : Fin 64 → ℕ)
  (distinct_skills : ∀ i j, i ≠ j → teams i ≠ teams j)

/-- The probability of the top three teams finishing in order of their skill levels. -/
def top_three_probability (t : Tournament) : ℚ :=
  512 / 1953

/-- Theorem stating the probability of the top three teams finishing in order of their skill levels. -/
theorem top_three_probability_correct (t : Tournament) : 
  top_three_probability t = 512 / 1953 := by
  sorry

end NUMINAMATH_CALUDE_top_three_probability_correct_l944_94413


namespace NUMINAMATH_CALUDE_subset_condition_implies_a_range_l944_94418

theorem subset_condition_implies_a_range (a : ℝ) : 
  (Finset.powerset {2 * a, a^2 - a}).card = 4 → a ≠ 0 ∧ a ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_implies_a_range_l944_94418


namespace NUMINAMATH_CALUDE_water_formation_l944_94445

/-- Represents the balanced chemical equation for the reaction between NH4Cl and NaOH -/
structure ChemicalReaction where
  nh4cl : ℕ
  naoh : ℕ
  h2o : ℕ
  balanced : nh4cl = naoh ∧ nh4cl = h2o

/-- Calculates the moles of water produced in the reaction -/
def waterProduced (reaction : ChemicalReaction) (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  min nh4cl_moles naoh_moles

theorem water_formation (reaction : ChemicalReaction) 
  (h1 : reaction.nh4cl = 1 ∧ reaction.naoh = 1 ∧ reaction.h2o = 1) 
  (h2 : nh4cl_moles = 3) 
  (h3 : naoh_moles = 3) : 
  waterProduced reaction nh4cl_moles naoh_moles = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_formation_l944_94445


namespace NUMINAMATH_CALUDE_cube_surface_area_l944_94467

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l944_94467


namespace NUMINAMATH_CALUDE_mara_bags_count_l944_94431

/-- Prove that Mara has 12 bags given the conditions of the marble problem -/
theorem mara_bags_count : ∀ (x : ℕ), 
  (x * 2 + 2 = 2 * 13) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_mara_bags_count_l944_94431


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l944_94499

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l944_94499


namespace NUMINAMATH_CALUDE_maciek_purchase_cost_l944_94493

-- Define the pricing structure
def pretzel_price (quantity : ℕ) : ℚ :=
  if quantity > 4 then 3.5 else 4

def chip_price (quantity : ℕ) : ℚ :=
  if quantity > 3 then 6.5 else 7

def soda_price (quantity : ℕ) : ℚ :=
  if quantity > 5 then 1.5 else 2

-- Define Maciek's purchase quantities
def pretzel_quantity : ℕ := 5
def chip_quantity : ℕ := 4
def soda_quantity : ℕ := 6

-- Calculate the total cost
def total_cost : ℚ :=
  pretzel_price pretzel_quantity * pretzel_quantity +
  chip_price chip_quantity * chip_quantity +
  soda_price soda_quantity * soda_quantity

-- Theorem statement
theorem maciek_purchase_cost :
  total_cost = 52.5 := by sorry

end NUMINAMATH_CALUDE_maciek_purchase_cost_l944_94493


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l944_94475

theorem fraction_equality_sum (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 5 / (x - 5)) →
  C + D = 29/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l944_94475


namespace NUMINAMATH_CALUDE_probability_all_red_by_fourth_draw_specific_l944_94479

/-- The probability of drawing all red balls exactly by the 4th draw -/
def probability_all_red_by_fourth_draw (white_balls red_balls : ℕ) : ℚ :=
  let total_balls := white_balls + red_balls
  let prob_white := white_balls / total_balls
  let prob_red := red_balls / total_balls
  prob_white^3 * prob_red

/-- Theorem stating the probability of drawing all red balls exactly by the 4th draw
    given 8 white balls and 2 red balls in a bag -/
theorem probability_all_red_by_fourth_draw_specific :
  probability_all_red_by_fourth_draw 8 2 = 217/5000 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_by_fourth_draw_specific_l944_94479


namespace NUMINAMATH_CALUDE_price_change_theorem_l944_94410

theorem price_change_theorem (initial_price : ℝ) (price_increase : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) :
  price_increase = 32 ∧ discount1 = 10 ∧ discount2 = 15 →
  let increased_price := initial_price * (1 + price_increase / 100)
  let after_discount1 := increased_price * (1 - discount1 / 100)
  let final_price := after_discount1 * (1 - discount2 / 100)
  (final_price - initial_price) / initial_price * 100 = 0.98 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l944_94410


namespace NUMINAMATH_CALUDE_eccentricity_equation_roots_l944_94409

/-- The cubic equation whose roots are the eccentricities of a hyperbola, an ellipse, and a parabola -/
def eccentricity_equation (x : ℝ) : Prop :=
  2 * x^3 - 7 * x^2 + 7 * x - 2 = 0

/-- Definition of eccentricity for an ellipse -/
def is_ellipse_eccentricity (e : ℝ) : Prop :=
  0 ≤ e ∧ e < 1

/-- Definition of eccentricity for a parabola -/
def is_parabola_eccentricity (e : ℝ) : Prop :=
  e = 1

/-- Definition of eccentricity for a hyperbola -/
def is_hyperbola_eccentricity (e : ℝ) : Prop :=
  e > 1

/-- The theorem stating that the roots of the equation correspond to the eccentricities of the three conic sections -/
theorem eccentricity_equation_roots :
  ∃ (e₁ e₂ e₃ : ℝ),
    eccentricity_equation e₁ ∧
    eccentricity_equation e₂ ∧
    eccentricity_equation e₃ ∧
    is_ellipse_eccentricity e₁ ∧
    is_parabola_eccentricity e₂ ∧
    is_hyperbola_eccentricity e₃ :=
  sorry

end NUMINAMATH_CALUDE_eccentricity_equation_roots_l944_94409


namespace NUMINAMATH_CALUDE_house_sale_profit_l944_94444

/-- Calculates the net profit from a house sale and repurchase --/
def netProfit (initialValue : ℝ) (sellProfit : ℝ) (buyLoss : ℝ) : ℝ :=
  let sellPrice := initialValue * (1 + sellProfit)
  let buyPrice := sellPrice * (1 - buyLoss)
  sellPrice - buyPrice

/-- Theorem stating that the net profit is $1725 given the specified conditions --/
theorem house_sale_profit :
  netProfit 15000 0.15 0.10 = 1725 := by
  sorry

#eval netProfit 15000 0.15 0.10

end NUMINAMATH_CALUDE_house_sale_profit_l944_94444


namespace NUMINAMATH_CALUDE_ellipse_equation_l944_94434

/-- The standard equation of an ellipse passing through (-3, 2) with the same foci as x²/9 + y²/4 = 1 -/
theorem ellipse_equation : ∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  ((-3)^2 / a^2 + 2^2 / b^2 = 1) ∧
  (a^2 - b^2 = 9 - 4) ∧
  (a^2 = 15 ∧ b^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l944_94434


namespace NUMINAMATH_CALUDE_taut_if_pred_prime_l944_94494

def is_taut (n : ℕ) : Prop :=
  ∃ (S : Finset (Fin (n^2 - n + 1))),
    S.card = n ∧
    ∀ (a b c d : Fin (n^2 - n + 1)),
      a ∈ S → b ∈ S → c ∈ S → d ∈ S →
      a ≠ b → c ≠ d →
      (a : ℕ) * (d : ℕ) ≠ (b : ℕ) * (c : ℕ)

theorem taut_if_pred_prime (n : ℕ) (h : n ≥ 2) (h_prime : Nat.Prime (n - 1)) :
  is_taut n :=
sorry

end NUMINAMATH_CALUDE_taut_if_pred_prime_l944_94494


namespace NUMINAMATH_CALUDE_terms_before_negative_twenty_l944_94460

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_negative_twenty :
  let a₁ := 100
  let d := -4
  let n := 31
  arithmetic_sequence a₁ d n = -20 ∧ n - 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_negative_twenty_l944_94460


namespace NUMINAMATH_CALUDE_coefficient_x4_is_negative_15_l944_94463

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 5*(x^3 - 2*x^4) + 3*(x^2 - x^4 + 2*x^6) - (2*x^4 + 5*x^3)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -15

/-- Theorem stating that the coefficient of x^4 in the simplified expression is -15 -/
theorem coefficient_x4_is_negative_15 :
  ∃ (f : ℝ → ℝ), ∀ x, expression x = f x + coefficient_x4 * x^4 ∧ 
  ∀ n, n ≠ 4 → (∃ c, ∀ x, f x = c * x^n + (f x - c * x^n)) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_negative_15_l944_94463


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l944_94461

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 2}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x < 2 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l944_94461


namespace NUMINAMATH_CALUDE_sample_is_extracurricular_homework_l944_94443

/-- Represents a student in the survey -/
structure Student where
  id : Nat
  hasExtracurricularHomework : Bool

/-- Represents the survey conducted by the middle school -/
structure Survey where
  totalPopulation : Finset Student
  selectedSample : Finset Student
  sampleSize : Nat

/-- Definition of a valid survey -/
def validSurvey (s : Survey) : Prop :=
  s.totalPopulation.card = 1800 ∧
  s.selectedSample.card = 300 ∧
  s.selectedSample ⊆ s.totalPopulation ∧
  s.sampleSize = s.selectedSample.card

/-- Definition of the sample in the survey -/
def sampleDefinition (s : Survey) : Finset Student :=
  s.selectedSample.filter (λ student => student.hasExtracurricularHomework)

/-- Theorem stating that the sample is the extracurricular homework of 300 students -/
theorem sample_is_extracurricular_homework (s : Survey) (h : validSurvey s) :
  sampleDefinition s = s.selectedSample :=
sorry


end NUMINAMATH_CALUDE_sample_is_extracurricular_homework_l944_94443


namespace NUMINAMATH_CALUDE_sqrt_two_expression_l944_94406

theorem sqrt_two_expression : Real.sqrt 2 * (Real.sqrt 2 + 2) - |Real.sqrt 2 - 2| = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_expression_l944_94406


namespace NUMINAMATH_CALUDE_odd_primes_with_eight_factors_l944_94459

theorem odd_primes_with_eight_factors (w y : ℕ) : 
  Nat.Prime w → 
  Nat.Prime y → 
  w < y → 
  Odd w → 
  Odd y → 
  (Finset.card (Nat.divisors (2 * w * y)) = 8) → 
  w = 3 := by
sorry

end NUMINAMATH_CALUDE_odd_primes_with_eight_factors_l944_94459


namespace NUMINAMATH_CALUDE_decimal_point_problem_l944_94453

theorem decimal_point_problem :
  ∃ (x y : ℝ), y - x = 7.02 ∧ y = 10 * x ∧ x = 0.78 ∧ y = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l944_94453


namespace NUMINAMATH_CALUDE_total_baseball_cards_l944_94485

-- Define the number of people
def num_people : Nat := 4

-- Define the number of cards each person has
def cards_per_person : Nat := 3

-- Theorem to prove
theorem total_baseball_cards : num_people * cards_per_person = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l944_94485


namespace NUMINAMATH_CALUDE_M_subset_N_l944_94433

-- Define set M
def M : Set ℝ := {x | x^2 = x}

-- Define set N
def N : Set ℝ := {x | x ≤ 1}

-- Theorem to prove
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l944_94433


namespace NUMINAMATH_CALUDE_meeting_handshakes_l944_94462

theorem meeting_handshakes (total_handshakes : ℕ) 
  (h1 : total_handshakes = 159) : ∃ (people second_handshakes : ℕ),
  people * (people - 1) / 2 + second_handshakes = total_handshakes ∧
  people = 18 ∧ 
  second_handshakes = 6 := by
sorry

end NUMINAMATH_CALUDE_meeting_handshakes_l944_94462


namespace NUMINAMATH_CALUDE_quadratic_function_range_l944_94473

/-- Given a quadratic function f(x) = x^2 + ax + b, where a and b are real numbers,
    and sets A and B defined as follows:
    A = { x ∈ ℝ | f(x) ≤ 0 }
    B = { x ∈ ℝ | f(f(x)) ≤ 3 }
    If A = B ≠ ∅, then the range of a is [2√3, 6). -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x : ℝ => x^2 + a*x + b
  let A := {x : ℝ | f x ≤ 0}
  let B := {x : ℝ | f (f x) ≤ 3}
  A = B ∧ A.Nonempty → a ∈ Set.Icc (2 * Real.sqrt 3) 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l944_94473


namespace NUMINAMATH_CALUDE_fourth_guard_runs_150_meters_l944_94471

/-- The length of the rectangle in meters -/
def length : ℝ := 200

/-- The width of the rectangle in meters -/
def width : ℝ := 300

/-- The perimeter of the rectangle in meters -/
def perimeter : ℝ := 2 * (length + width)

/-- The total distance run by three guards in meters -/
def three_guards_distance : ℝ := 850

/-- The distance run by the fourth guard in meters -/
def fourth_guard_distance : ℝ := perimeter - three_guards_distance

theorem fourth_guard_runs_150_meters :
  fourth_guard_distance = 150 := by sorry

end NUMINAMATH_CALUDE_fourth_guard_runs_150_meters_l944_94471


namespace NUMINAMATH_CALUDE_equation_solution_l944_94458

theorem equation_solution : ∃ x : ℚ, (1 / 4 + 8 / x = 13 / x + 1 / 8) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l944_94458


namespace NUMINAMATH_CALUDE_two_zeros_implies_a_geq_two_l944_94446

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then 2 * x - a else Real.log (1 - x)

theorem two_zeros_implies_a_geq_two (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(f a x = 0 ∧ f a y = 0 ∧ f a z = 0)) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_implies_a_geq_two_l944_94446


namespace NUMINAMATH_CALUDE_train_passing_time_l944_94487

/-- The time taken for a person to walk the length of a train, given the times it takes for the train to pass the person in opposite and same directions. -/
theorem train_passing_time (t₁ t₂ : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > 0) (h₃ : t₂ > t₁) : 
  let t₃ := (2 * t₁ * t₂) / (t₂ - t₁)
  t₁ = 1 ∧ t₂ = 2 → t₃ = 4 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l944_94487


namespace NUMINAMATH_CALUDE_largest_non_attainable_sum_l944_94490

/-- The set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Set ℕ :=
  {3*n - 1, 6*n + 1, 6*n + 4, 6*n + 7}

/-- A sum is attainable if it can be formed using the given coin denominations -/
def is_attainable (n : ℕ) (sum : ℕ) : Prop :=
  ∃ (a b c d : ℕ), sum = a*(3*n - 1) + b*(6*n + 1) + c*(6*n + 4) + d*(6*n + 7)

/-- The largest non-attainable sum in Limonia -/
def largest_non_attainable (n : ℕ) : ℕ := 6*n^2 + 4*n - 5

/-- Theorem: The largest non-attainable sum in Limonia is 6n^2 + 4n - 5 -/
theorem largest_non_attainable_sum (n : ℕ) :
  (∀ k > largest_non_attainable n, is_attainable n k) ∧
  ¬(is_attainable n (largest_non_attainable n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_non_attainable_sum_l944_94490


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l944_94414

theorem largest_n_satisfying_inequality :
  ∃ (n : ℕ), n^300 < 3^500 ∧ ∀ (m : ℕ), m^300 < 3^500 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l944_94414


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l944_94429

theorem probability_at_least_two_same (n : Nat) (s : Nat) :
  n = 8 →
  s = 8 →
  (1 - (Nat.factorial n) / (s^n : ℚ)) = 415 / 416 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l944_94429


namespace NUMINAMATH_CALUDE_circle_k_range_l944_94411

/-- Represents the equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 5*k = 0

/-- Checks if the equation represents a valid circle -/
def is_circle (k : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y k ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The theorem stating the range of k for which the equation represents a circle -/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l944_94411


namespace NUMINAMATH_CALUDE_pennys_bakery_revenue_l944_94412

/-- The revenue calculation for Penny's bakery --/
theorem pennys_bakery_revenue : 
  ∀ (price_per_slice : ℕ) (slices_per_pie : ℕ) (number_of_pies : ℕ),
    price_per_slice = 7 →
    slices_per_pie = 6 →
    number_of_pies = 7 →
    price_per_slice * slices_per_pie * number_of_pies = 294 := by
  sorry

end NUMINAMATH_CALUDE_pennys_bakery_revenue_l944_94412


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l944_94428

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 6*a*x + 1 < 0) → a ∈ Set.Icc (-1/3) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l944_94428


namespace NUMINAMATH_CALUDE_problem_1_2_l944_94478

theorem problem_1_2 :
  (2 * Real.sqrt 6 + 2 / 3) * Real.sqrt 3 - Real.sqrt 32 = 2 * Real.sqrt 2 + 2 * Real.sqrt 3 / 3 ∧
  (Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) - (Real.sqrt 45 + Real.sqrt 20) / Real.sqrt 5 = -10 :=
by sorry

end NUMINAMATH_CALUDE_problem_1_2_l944_94478


namespace NUMINAMATH_CALUDE_perpendicular_planes_condition_l944_94470

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset_line_plane : Line → Plane → Prop)

variable (α β : Plane)
variable (m : Line)

-- Theorem statement
theorem perpendicular_planes_condition 
  (h_distinct : α ≠ β)
  (h_subset : subset_line_plane m α) :
  (∀ m, subset_line_plane m α → 
    (perpendicular_line_plane m β → perpendicular_plane_plane α β) ∧
    ¬(perpendicular_plane_plane α β → perpendicular_line_plane m β)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_condition_l944_94470


namespace NUMINAMATH_CALUDE_selection_plans_count_l944_94416

def number_of_people : ℕ := 6
def number_of_cities : ℕ := 4
def number_to_select : ℕ := 4
def restricted_people : ℕ := 2
def restricted_city : ℕ := 1

theorem selection_plans_count :
  (number_of_people * (number_of_people - 1) * (number_of_people - 2) * (number_of_people - 3)) -
  (restricted_people * ((number_of_people - 1) * (number_of_people - 2) * (number_of_people - 3))) = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_plans_count_l944_94416


namespace NUMINAMATH_CALUDE_sons_age_l944_94415

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l944_94415


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l944_94438

theorem set_equality_implies_sum_of_powers (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2016 + b^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l944_94438


namespace NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l944_94430

-- Problem 1
theorem max_value_negative_x (x : ℝ) (hx : x < 0) :
  (x^2 + x + 1) / x ≤ -1 :=
sorry

-- Problem 2
theorem min_value_greater_than_negative_one (x : ℝ) (hx : x > -1) :
  ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l944_94430


namespace NUMINAMATH_CALUDE_boys_height_correction_l944_94466

theorem boys_height_correction (n : ℕ) (initial_avg wrong_height actual_avg : ℝ) : 
  n = 35 →
  initial_avg = 183 →
  wrong_height = 166 →
  actual_avg = 181 →
  ∃ (correct_height : ℝ), 
    correct_height = wrong_height + (n * initial_avg - n * actual_avg) ∧
    correct_height = 236 :=
by sorry

end NUMINAMATH_CALUDE_boys_height_correction_l944_94466


namespace NUMINAMATH_CALUDE_team_average_score_l944_94426

theorem team_average_score (player1 player2 player3 player4 : ℝ) 
  (h1 : player1 = 20)
  (h2 : player2 = player1 / 2)
  (h3 : player3 = 6 * player2)
  (h4 : player4 = 3 * player3) :
  (player1 + player2 + player3 + player4) / 4 = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l944_94426


namespace NUMINAMATH_CALUDE_specific_cistern_wet_area_l944_94497

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the wet surface area of a specific cistern -/
theorem specific_cistern_wet_area :
  cisternWetArea 6 4 1.25 = 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_cistern_wet_area_l944_94497


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l944_94468

theorem sqrt_of_sqrt_81 : ∃ (x : ℝ), x^2 = 81 ∧ (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l944_94468


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l944_94437

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 21 + 7 + 12 + y) / 6 = 15 → y = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l944_94437


namespace NUMINAMATH_CALUDE_single_point_ellipse_l944_94495

theorem single_point_ellipse (c : ℝ) : 
  (∃! p : ℝ × ℝ, 4 * p.1^2 + p.2^2 + 16 * p.1 - 6 * p.2 + c = 0) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_single_point_ellipse_l944_94495


namespace NUMINAMATH_CALUDE_no_negative_exponents_l944_94442

theorem no_negative_exponents (a b c d : ℤ) 
  (h : (4 : ℝ)^a + (4 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d + 1) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d := by
  sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l944_94442


namespace NUMINAMATH_CALUDE_triple_involution_properties_l944_94427

/-- A function f: ℝ → ℝ satisfying f(f(f(x))) = x for all x ∈ ℝ -/
def triple_involution (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f (f x)) = x

theorem triple_involution_properties (f : ℝ → ℝ) (h : triple_involution f) :
  (∀ x y : ℝ, f x = f y → x = y) ∧ 
  (¬ (∀ x y : ℝ, x < y → f x > f y)) ∧
  ((∀ x y : ℝ, x < y → f x < f y) → ∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_triple_involution_properties_l944_94427


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l944_94421

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 3*x - 10 > 0) ↔ (x < -2 ∨ x > 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l944_94421


namespace NUMINAMATH_CALUDE_problem_solution_l944_94402

theorem problem_solution : 
  (101 * 99 = 9999) ∧ 
  (32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l944_94402


namespace NUMINAMATH_CALUDE_ferry_river_crossing_l944_94417

/-- Two ferries crossing a river problem -/
theorem ferry_river_crossing (W : ℝ) : 
  W > 0 → -- Width of the river is positive
  (∃ (d₁ d₂ : ℝ), 
    d₁ = 700 ∧ -- First meeting point is 700 feet from one shore
    d₁ + d₂ = W ∧ -- Sum of distances at first meeting equals river width
    W + 400 + (W + (W - 400)) = 3 * W ∧ -- Total distance at second meeting
    2 * (W + 700) = 3 * W) → -- Relationship between meetings and river width
  W = 1400 := by
sorry

end NUMINAMATH_CALUDE_ferry_river_crossing_l944_94417


namespace NUMINAMATH_CALUDE_f_properties_l944_94408

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^x / (a^x + 1)

-- Main theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- 1. The range of f(x) is (0, 1)
  (∀ x, 0 < f a x ∧ f a x < 1) ∧
  -- 2. If the maximum value of f(x) on [-1, 2] is 3/4, then a = √3 or a = 1/3
  (Set.Icc (-1) 2 ⊆ f a ⁻¹' Set.Iio (3/4) → a = Real.sqrt 3 ∨ a = 1/3) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l944_94408


namespace NUMINAMATH_CALUDE_smaller_number_problem_l944_94477

theorem smaller_number_problem (x y : ℝ) (h_sum : x + y = 18) (h_product : x * y = 80) :
  min x y = 8 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l944_94477


namespace NUMINAMATH_CALUDE_children_distribution_l944_94476

theorem children_distribution (n : ℕ) : 
  (6 : ℝ) / n - (6 : ℝ) / (n + 2) = (1 : ℝ) / 4 → n + 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_children_distribution_l944_94476


namespace NUMINAMATH_CALUDE_trip_distance_is_150_l944_94400

/-- Represents the problem of determining the trip distance --/
def TripDistance (D : ℝ) : Prop :=
  let rental_cost_1 : ℝ := 50
  let rental_cost_2 : ℝ := 90
  let gas_efficiency : ℝ := 15  -- km per liter
  let gas_cost : ℝ := 0.9  -- dollars per liter
  let cost_difference : ℝ := 22
  rental_cost_1 + (2 * D / gas_efficiency * gas_cost) = rental_cost_2 - cost_difference

/-- The theorem stating that the trip distance each way is 150 km --/
theorem trip_distance_is_150 : TripDistance 150 := by
  sorry

end NUMINAMATH_CALUDE_trip_distance_is_150_l944_94400


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l944_94404

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.15

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 22

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * num_tshirts

/-- Theorem stating that the total cost of Carrie's t-shirt purchase is $201.30 -/
theorem carries_tshirt_purchase : total_cost = 201.30 := by
  sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l944_94404


namespace NUMINAMATH_CALUDE_prob_other_side_green_l944_94464

/-- Represents a card with two sides --/
inductive Card
| BlueBoth
| BlueGreen
| GreenBoth

/-- The box of cards --/
def box : Finset Card := sorry

/-- The number of cards in the box --/
def num_cards : ℕ := 8

/-- The number of cards that are blue on both sides --/
def num_blue_both : ℕ := 4

/-- The number of cards that are blue on one side and green on the other --/
def num_blue_green : ℕ := 2

/-- The number of cards that are green on both sides --/
def num_green_both : ℕ := 2

/-- Function to check if a given side of a card is green --/
def is_green (c : Card) (side : Bool) : Bool := sorry

/-- The probability of picking a card and observing a green side --/
def prob_green_side : ℚ := sorry

/-- The probability of both sides being green given that one observed side is green --/
def prob_both_green_given_one_green : ℚ := sorry

theorem prob_other_side_green : 
  prob_both_green_given_one_green = 2/3 := sorry

end NUMINAMATH_CALUDE_prob_other_side_green_l944_94464


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l944_94484

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 8 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l944_94484


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l944_94447

/-- Given that 9 oranges weigh the same as 6 apples, prove that 54 oranges
    weigh the same as 36 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
    orange_weight > 0 →
    apple_weight > 0 →
    9 * orange_weight = 6 * apple_weight →
    54 * orange_weight = 36 * apple_weight := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l944_94447


namespace NUMINAMATH_CALUDE_range_of_sin6_plus_cos4_l944_94403

theorem range_of_sin6_plus_cos4 :
  ∀ x : ℝ, 0 ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧
  Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 ∧
  (∃ y : ℝ, Real.sin y ^ 6 + Real.cos y ^ 4 = 0) ∧
  (∃ z : ℝ, Real.sin z ^ 6 + Real.cos z ^ 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_sin6_plus_cos4_l944_94403


namespace NUMINAMATH_CALUDE_factorization_x4_minus_16y4_l944_94401

theorem factorization_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16*y^4 = (x^2 + 4*y^2)*(x + 2*y)*(x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_16y4_l944_94401


namespace NUMINAMATH_CALUDE_sale_price_comparison_l944_94498

theorem sale_price_comparison (x : ℝ) (h : x > 0) : x * 1.3 * 0.85 > x * 1.1 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_comparison_l944_94498


namespace NUMINAMATH_CALUDE_count_positive_rationals_l944_94486

def numbers : List ℚ := [-2023, 1/100, 3/2, 0, 1/5]

theorem count_positive_rationals : 
  (numbers.filter (λ x => x > 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_positive_rationals_l944_94486


namespace NUMINAMATH_CALUDE_discount_savings_l944_94469

/-- Given a store with an 8% discount and a customer who pays $184 for an item, 
    prove that the amount saved is $16. -/
theorem discount_savings (discount_rate : ℝ) (paid_amount : ℝ) (saved_amount : ℝ) : 
  discount_rate = 0.08 →
  paid_amount = 184 →
  saved_amount = 16 →
  paid_amount / (1 - discount_rate) * discount_rate = saved_amount := by
sorry

end NUMINAMATH_CALUDE_discount_savings_l944_94469


namespace NUMINAMATH_CALUDE_inequality_equivalence_l944_94454

theorem inequality_equivalence (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (2 + y) / x < (4 - x) / y ↔
  ((x * y > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧
   (x * y < 0 → (x - 2)^2 + (y + 1)^2 > 5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l944_94454


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l944_94424

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l944_94424
