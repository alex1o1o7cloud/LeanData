import Mathlib

namespace NUMINAMATH_CALUDE_sum_is_even_l3313_331311

theorem sum_is_even (a b p : ℕ) (ha : 4 ∣ a) (hb1 : 6 ∣ b) (hb2 : p ∣ b) (hp : Prime p) :
  Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_is_even_l3313_331311


namespace NUMINAMATH_CALUDE_range_of_m_for_p_or_q_l3313_331381

theorem range_of_m_for_p_or_q (m : ℝ) :
  (∃ x₀ : ℝ, m * x₀^2 + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_or_q_l3313_331381


namespace NUMINAMATH_CALUDE_max_lessons_l3313_331395

/-- Represents the number of shirts the teacher has. -/
def s : ℕ := sorry

/-- Represents the number of pairs of pants the teacher has. -/
def p : ℕ := sorry

/-- Represents the number of pairs of shoes the teacher has. -/
def b : ℕ := sorry

/-- Represents the number of jackets the teacher has. -/
def jackets : ℕ := 2

/-- Represents the total number of possible lessons. -/
def total_lessons : ℕ := 2 * s * p * b

/-- States that one more shirt would allow 36 more lessons. -/
axiom shirt_condition : 2 * (s + 1) * p * b = total_lessons + 36

/-- States that one more pair of pants would allow 72 more lessons. -/
axiom pants_condition : 2 * s * (p + 1) * b = total_lessons + 72

/-- States that one more pair of shoes would allow 54 more lessons. -/
axiom shoes_condition : 2 * s * p * (b + 1) = total_lessons + 54

/-- Theorem stating the maximum number of lessons the teacher could have conducted. -/
theorem max_lessons : total_lessons = 216 := by sorry

end NUMINAMATH_CALUDE_max_lessons_l3313_331395


namespace NUMINAMATH_CALUDE_projection_theorem_l3313_331358

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ := sorry

/-- The plane Q passing through the origin -/
def Q : Plane := sorry

theorem projection_theorem :
  project (6, 4, 6) Q = (4, 6, 2) →
  project (5, 2, 8) Q = (11/6, 31/6, 10/6) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l3313_331358


namespace NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l3313_331305

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (x y : ℝ) : ℝ := |y|

/-- The theorem stating that the distance from (-2, -√5) to the x-axis is √5 -/
theorem distance_from_point_to_x_axis :
  distance_to_x_axis (-2 : ℝ) (-Real.sqrt 5) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l3313_331305


namespace NUMINAMATH_CALUDE_basketball_max_height_l3313_331364

/-- The height function of a basketball -/
def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

/-- The maximum height reached by the basketball -/
theorem basketball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 127 :=
sorry

end NUMINAMATH_CALUDE_basketball_max_height_l3313_331364


namespace NUMINAMATH_CALUDE_intersection_A_B_l3313_331337

def U : Set Int := {-1, 3, 5, 7, 9}
def complement_A : Set Int := {-1, 9}
def B : Set Int := {3, 7, 9}

theorem intersection_A_B :
  let A := U \ complement_A
  (A ∩ B) = {3, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3313_331337


namespace NUMINAMATH_CALUDE_billys_age_l3313_331301

theorem billys_age (my_age billy_age : ℕ) 
  (h1 : my_age = 4 * billy_age)
  (h2 : my_age - billy_age = 12) :
  billy_age = 4 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l3313_331301


namespace NUMINAMATH_CALUDE_decimal_division_proof_l3313_331344

theorem decimal_division_proof : 
  (0.182 : ℚ) / (0.0021 : ℚ) = 86 + 14 / 21 := by sorry

end NUMINAMATH_CALUDE_decimal_division_proof_l3313_331344


namespace NUMINAMATH_CALUDE_waiter_initial_customers_l3313_331334

/-- The number of customers who left the waiter's section -/
def customers_left : ℕ := 12

/-- The number of people at each table after some customers left -/
def people_per_table : ℕ := 3

/-- The number of tables in the waiter's section -/
def number_of_tables : ℕ := 3

/-- The initial number of customers in the waiter's section -/
def initial_customers : ℕ := customers_left + people_per_table * number_of_tables

theorem waiter_initial_customers :
  initial_customers = 21 :=
by sorry

end NUMINAMATH_CALUDE_waiter_initial_customers_l3313_331334


namespace NUMINAMATH_CALUDE_abcdef_hex_to_binary_bits_l3313_331375

theorem abcdef_hex_to_binary_bits : ∃ (n : ℕ), n = 24 ∧ 
  (2^(n-1) : ℕ) ≤ (0xABCDEF : ℕ) ∧ (0xABCDEF : ℕ) < 2^n :=
by sorry

end NUMINAMATH_CALUDE_abcdef_hex_to_binary_bits_l3313_331375


namespace NUMINAMATH_CALUDE_point_on_h_graph_and_coordinate_sum_l3313_331312

/-- Given a function g where g(4) = 8, and h defined as h(x) = 2(g(x))^3,
    prove that (4,1024) is on the graph of h and the sum of its coordinates is 1028 -/
theorem point_on_h_graph_and_coordinate_sum 
  (g : ℝ → ℝ) (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = 2 * (g x)^3)
  (g_value : g 4 = 8) :
  h 4 = 1024 ∧ 4 + 1024 = 1028 := by
  sorry

end NUMINAMATH_CALUDE_point_on_h_graph_and_coordinate_sum_l3313_331312


namespace NUMINAMATH_CALUDE_prob_each_class_one_student_prob_at_least_one_empty_class_prob_exactly_one_empty_class_l3313_331324

/-- The number of newly transferred students -/
def num_students : ℕ := 4

/-- The number of designated classes -/
def num_classes : ℕ := 4

/-- The total number of ways to distribute students into classes -/
def total_distributions : ℕ := num_classes ^ num_students

/-- The number of ways to distribute students such that each class receives one student -/
def each_class_one_student : ℕ := Nat.factorial num_classes

/-- The probability that each class receives one student -/
theorem prob_each_class_one_student :
  (each_class_one_student : ℚ) / total_distributions = 3 / 32 := by sorry

/-- The probability that at least one class does not receive any students -/
theorem prob_at_least_one_empty_class :
  1 - (each_class_one_student : ℚ) / total_distributions = 29 / 32 := by sorry

/-- The number of ways to distribute students such that exactly one class is empty -/
def exactly_one_empty_class : ℕ :=
  (num_classes.choose 1) * (num_classes.choose 2) * ((num_classes - 1).choose 1) * ((num_classes - 2).choose 1)

/-- The probability that exactly one class does not receive any students -/
theorem prob_exactly_one_empty_class :
  (exactly_one_empty_class : ℚ) / total_distributions = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_each_class_one_student_prob_at_least_one_empty_class_prob_exactly_one_empty_class_l3313_331324


namespace NUMINAMATH_CALUDE_coins_player1_l3313_331338

/-- Represents the number of sectors and players -/
def n : ℕ := 9

/-- Represents the number of rotations -/
def rotations : ℕ := 11

/-- Represents the coins received by player 4 -/
def coins_player4 : ℕ := 90

/-- Represents the coins received by player 8 -/
def coins_player8 : ℕ := 35

/-- Theorem stating the number of coins received by player 1 -/
theorem coins_player1 (h1 : n = 9) (h2 : rotations = 11) 
  (h3 : coins_player4 = 90) (h4 : coins_player8 = 35) : 
  ∃ (coins_player1 : ℕ), coins_player1 = 57 :=
sorry


end NUMINAMATH_CALUDE_coins_player1_l3313_331338


namespace NUMINAMATH_CALUDE_units_digit_of_five_consecutive_integers_l3313_331369

theorem units_digit_of_five_consecutive_integers (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_five_consecutive_integers_l3313_331369


namespace NUMINAMATH_CALUDE_shampoo_comparison_l3313_331379

/-- Represents a bottle of shampoo with weight in grams and price in yuan -/
structure ShampooBottle where
  weight : ℚ
  price : ℚ

/-- Calculates the cost per gram of a shampoo bottle -/
def costPerGram (bottle : ShampooBottle) : ℚ :=
  bottle.price / bottle.weight

theorem shampoo_comparison (large small : ShampooBottle)
  (h_large_weight : large.weight = 450)
  (h_large_price : large.price = 36)
  (h_small_weight : small.weight = 150)
  (h_small_price : small.price = 25/2) :
  (∃ (a b : ℕ), a = 72 ∧ b = 25 ∧ large.price / small.price = a / b) ∧
  costPerGram large < costPerGram small :=
sorry

end NUMINAMATH_CALUDE_shampoo_comparison_l3313_331379


namespace NUMINAMATH_CALUDE_cheries_sparklers_l3313_331330

/-- Represents the number of fireworks in a box -/
structure FireworksBox where
  sparklers : ℕ
  whistlers : ℕ

/-- The total number of fireworks in a box -/
def FireworksBox.total (box : FireworksBox) : ℕ := box.sparklers + box.whistlers

theorem cheries_sparklers (koby_box : FireworksBox) 
                          (cherie_box : FireworksBox) 
                          (h1 : koby_box.sparklers = 3)
                          (h2 : koby_box.whistlers = 5)
                          (h3 : cherie_box.whistlers = 9)
                          (h4 : 2 * koby_box.total + cherie_box.total = 33) :
  cherie_box.sparklers = 8 := by
  sorry

#check cheries_sparklers

end NUMINAMATH_CALUDE_cheries_sparklers_l3313_331330


namespace NUMINAMATH_CALUDE_janabel_sales_sum_l3313_331359

theorem janabel_sales_sum (n : ℕ) (a₁ d : ℤ) (h1 : n = 12) (h2 : a₁ = 1) (h3 : d = 4) :
  (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2 = 276 :=
by sorry

end NUMINAMATH_CALUDE_janabel_sales_sum_l3313_331359


namespace NUMINAMATH_CALUDE_ballet_class_size_l3313_331373

/-- The number of large groups formed in the ballet class -/
def large_groups : ℕ := 12

/-- The number of members in each large group -/
def members_per_large_group : ℕ := 7

/-- The total number of members in the ballet class -/
def total_members : ℕ := large_groups * members_per_large_group

theorem ballet_class_size : total_members = 84 := by
  sorry

end NUMINAMATH_CALUDE_ballet_class_size_l3313_331373


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3313_331354

theorem arithmetic_calculation : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3313_331354


namespace NUMINAMATH_CALUDE_complex_power_eight_l3313_331378

theorem complex_power_eight (z : ℂ) : 
  z = (1 - Complex.I * Real.sqrt 3) / 2 → 
  z^8 = -(1 + Complex.I * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_power_eight_l3313_331378


namespace NUMINAMATH_CALUDE_meghan_money_l3313_331372

/-- The total amount of money Meghan had -/
def total_money (hundred_bills : ℕ) (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  100 * hundred_bills + 50 * fifty_bills + 10 * ten_bills

/-- Proof that Meghan had $550 -/
theorem meghan_money : total_money 2 5 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_meghan_money_l3313_331372


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l3313_331367

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequalities (a b c : ℝ) :
  (∀ x, (1/2 : ℝ) ≤ x ∧ x ≤ 2 → f a b c x ≥ 0) ∧
  (∀ x, x < (1/2 : ℝ) ∨ x > 2 → f a b c x < 0) →
  b > 0 ∧ a + b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l3313_331367


namespace NUMINAMATH_CALUDE_solve_equation_l3313_331340

theorem solve_equation : ∃ x : ℚ, 5 * x - 3 * x = 405 - 9 * (x + 4) → x = 369 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3313_331340


namespace NUMINAMATH_CALUDE_gcd_12347_30841_l3313_331374

theorem gcd_12347_30841 : Nat.gcd 12347 30841 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12347_30841_l3313_331374


namespace NUMINAMATH_CALUDE_y_squared_times_three_l3313_331394

theorem y_squared_times_three (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 2 * x - y = 20) : 
  3 * y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_times_three_l3313_331394


namespace NUMINAMATH_CALUDE_min_omega_value_l3313_331396

/-- Given a function f(x) = 2 sin(ωx) with a minimum value of 2 on the interval [-π/3, π/4],
    the minimum value of ω is 3/2 -/
theorem min_omega_value (ω : ℝ) (h : ∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) ≥ 2) :
  ω ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_omega_value_l3313_331396


namespace NUMINAMATH_CALUDE_driving_distance_difference_l3313_331304

/-- Represents a driver's journey --/
structure Journey where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem statement --/
theorem driving_distance_difference 
  (liam : Journey) 
  (zoe : Journey) 
  (mia : Journey) 
  (h1 : zoe.time = liam.time + 2)
  (h2 : zoe.speed = liam.speed + 7)
  (h3 : zoe.distance = liam.distance + 80)
  (h4 : mia.time = liam.time + 3)
  (h5 : mia.speed = liam.speed + 15)
  (h6 : ∀ j : Journey, j.distance = j.speed * j.time) :
  mia.distance - liam.distance = 243 := by
  sorry

end NUMINAMATH_CALUDE_driving_distance_difference_l3313_331304


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3313_331371

/-- Given a quadratic equation with real coefficients x^2 + px + q = 0,
    if b+i and 2-ai (where a and b are real) are its roots, then q = 5 -/
theorem quadratic_root_value (p q a b : ℝ) : 
  (∀ x : ℂ, x^2 + p*x + q = 0 ↔ x = b + I ∨ x = 2 - a*I) →
  q = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3313_331371


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3313_331386

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → (∃ r : ℝ, 10 * r = b ∧ b * r = 2/3) → b = 2 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3313_331386


namespace NUMINAMATH_CALUDE_num_purchasing_plans_eq_600_l3313_331320

/-- The number of different purchasing plans for souvenirs -/
def num_purchasing_plans : ℕ :=
  (Finset.filter (fun (x, y, z) => x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1)
    (Finset.filter (fun (x, y, z) => x + 2*y + 4*z = 101)
      (Finset.product (Finset.range 102)
        (Finset.product (Finset.range 51) (Finset.range 26))))).card

/-- Theorem stating that the number of purchasing plans is 600 -/
theorem num_purchasing_plans_eq_600 : num_purchasing_plans = 600 := by
  sorry

end NUMINAMATH_CALUDE_num_purchasing_plans_eq_600_l3313_331320


namespace NUMINAMATH_CALUDE_john_purchase_profit_l3313_331392

/-- Represents the purchase and sale of items with profit or loss -/
theorem john_purchase_profit (x : ℝ) : 
  let grinder_purchase := 15000
  let grinder_loss_percent := 0.04
  let mobile_profit_percent := 0.15
  let total_profit := 600
  let grinder_sale := grinder_purchase * (1 - grinder_loss_percent)
  let mobile_sale := x * (1 + mobile_profit_percent)
  (mobile_sale - x) - (grinder_purchase - grinder_sale) = total_profit →
  x = 8000 := by
sorry

end NUMINAMATH_CALUDE_john_purchase_profit_l3313_331392


namespace NUMINAMATH_CALUDE_triangle_half_angle_sine_inequality_l3313_331343

theorem triangle_half_angle_sine_inequality (A B C : Real) 
  (h : A + B + C = π) : 
  Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_half_angle_sine_inequality_l3313_331343


namespace NUMINAMATH_CALUDE_proof_by_contradiction_elements_l3313_331362

/-- Elements that can be used in a proof by contradiction -/
inductive ProofByContradictionElement : Type
  | NegationOfConclusion : ProofByContradictionElement
  | KnownConditions : ProofByContradictionElement
  | AxiomsTheoremsDefinitions : ProofByContradictionElement

/-- The set of elements used in proof by contradiction -/
def ProofByContradictionSet : Set ProofByContradictionElement :=
  {ProofByContradictionElement.NegationOfConclusion,
   ProofByContradictionElement.KnownConditions,
   ProofByContradictionElement.AxiomsTheoremsDefinitions}

/-- Theorem stating that the ProofByContradictionSet contains all necessary elements -/
theorem proof_by_contradiction_elements :
  ProofByContradictionElement.NegationOfConclusion ∈ ProofByContradictionSet ∧
  ProofByContradictionElement.KnownConditions ∈ ProofByContradictionSet ∧
  ProofByContradictionElement.AxiomsTheoremsDefinitions ∈ ProofByContradictionSet :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_elements_l3313_331362


namespace NUMINAMATH_CALUDE_amc_scoring_l3313_331351

theorem amc_scoring (total_problems : Nat) (correct_points : Int) (incorrect_points : Int) 
  (unanswered_points : Int) (attempted : Nat) (unanswered : Nat) (min_score : Int) : 
  let min_correct := ((min_score - unanswered * unanswered_points) - 
    (attempted * incorrect_points)) / (correct_points - incorrect_points)
  ⌈min_correct⌉ = 17 :=
by
  sorry

#check amc_scoring 30 7 (-1) 2 25 5 120

end NUMINAMATH_CALUDE_amc_scoring_l3313_331351


namespace NUMINAMATH_CALUDE_negation_of_union_membership_l3313_331370

theorem negation_of_union_membership (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B :=
by sorry

end NUMINAMATH_CALUDE_negation_of_union_membership_l3313_331370


namespace NUMINAMATH_CALUDE_duck_percentage_among_non_swans_l3313_331331

theorem duck_percentage_among_non_swans 
  (duck_percent : ℝ) 
  (swan_percent : ℝ) 
  (eagle_percent : ℝ) 
  (sparrow_percent : ℝ) 
  (h1 : duck_percent = 40)
  (h2 : swan_percent = 20)
  (h3 : eagle_percent = 15)
  (h4 : sparrow_percent = 25)
  (h5 : duck_percent + swan_percent + eagle_percent + sparrow_percent = 100) :
  (duck_percent / (100 - swan_percent)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_duck_percentage_among_non_swans_l3313_331331


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3313_331318

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| - |x - 3| ≥ 0} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3313_331318


namespace NUMINAMATH_CALUDE_quadratic_part_of_equation_l3313_331391

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 8*x + 21 = |x - 5| + 4

-- Define the sum of solutions
def sum_of_solutions : ℝ := 20

-- Theorem to prove
theorem quadratic_part_of_equation :
  ∃ (a b c : ℝ), 
    (∀ x, quadratic_equation x → a*x^2 + b*x + c = |x - 5| + 4) ∧
    (a = 1 ∧ b = -8 ∧ c = 21) :=
sorry

end NUMINAMATH_CALUDE_quadratic_part_of_equation_l3313_331391


namespace NUMINAMATH_CALUDE_custom_op_zero_l3313_331307

def custom_op (a b c : ℝ) : ℝ := 3 * (a - b - c)^2

theorem custom_op_zero (x y z : ℝ) : 
  custom_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_zero_l3313_331307


namespace NUMINAMATH_CALUDE_min_colors_for_pyramid_game_l3313_331322

/-- Represents a pyramid with a regular polygon base -/
structure Pyramid :=
  (base_vertices : ℕ)

/-- The total number of edges in a pyramid -/
def total_edges (p : Pyramid) : ℕ := 2 * p.base_vertices

/-- The maximum degree of any vertex in the pyramid -/
def max_vertex_degree (p : Pyramid) : ℕ := p.base_vertices

/-- The minimal number of colors needed for the coloring game on a pyramid -/
def min_colors_needed (p : Pyramid) : ℕ := p.base_vertices

theorem min_colors_for_pyramid_game (p : Pyramid) (h : p.base_vertices = 2016) :
  min_colors_needed p = 2016 :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_pyramid_game_l3313_331322


namespace NUMINAMATH_CALUDE_number_division_l3313_331360

theorem number_division (x : ℝ) : x - 17 = 55 → x / 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l3313_331360


namespace NUMINAMATH_CALUDE_parabola_directrix_l3313_331314

/-- Given a parabola with equation x^2 = -1/8 * y, its directrix equation is y = 1/32 -/
theorem parabola_directrix (x y : ℝ) : 
  (x^2 = -1/8 * y) → (∃ (k : ℝ), k = 1/32 ∧ k = y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3313_331314


namespace NUMINAMATH_CALUDE_jerrys_remaining_debt_l3313_331335

/-- Given Jerry's debt payments over two months, calculate his remaining debt -/
theorem jerrys_remaining_debt (total_debt : ℕ) (first_payment : ℕ) (additional_payment : ℕ) :
  total_debt = 50 →
  first_payment = 12 →
  additional_payment = 3 →
  total_debt - (first_payment + (first_payment + additional_payment)) = 23 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_remaining_debt_l3313_331335


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_plus_exponential_negation_l3313_331325

theorem negation_of_universal_positive (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem cubic_plus_exponential_negation :
  (¬∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_plus_exponential_negation_l3313_331325


namespace NUMINAMATH_CALUDE_number_comparisons_l3313_331368

theorem number_comparisons :
  (-3.2 > -4.3) ∧ ((1/2 : ℚ) > -1/3) ∧ ((1/4 : ℚ) > 0) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l3313_331368


namespace NUMINAMATH_CALUDE_expression_equality_l3313_331382

theorem expression_equality : -15 + 9 * (6 / 3) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l3313_331382


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3313_331339

theorem polynomial_divisibility (a : ℤ) : 
  ∃ k : ℤ, (3*a + 5)^2 - 4 = (a + 1) * k := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3313_331339


namespace NUMINAMATH_CALUDE_notebook_cost_l3313_331302

/-- Given the cost of items and the total spent, prove the cost of each notebook -/
theorem notebook_cost 
  (pen_cost : ℕ) 
  (folder_cost : ℕ) 
  (num_pens : ℕ) 
  (num_notebooks : ℕ) 
  (num_folders : ℕ) 
  (total_spent : ℕ) 
  (h1 : pen_cost = 1) 
  (h2 : folder_cost = 5) 
  (h3 : num_pens = 3) 
  (h4 : num_notebooks = 4) 
  (h5 : num_folders = 2) 
  (h6 : total_spent = 25) : 
  (total_spent - num_pens * pen_cost - num_folders * folder_cost) / num_notebooks = 3 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3313_331302


namespace NUMINAMATH_CALUDE_solve_seashells_problem_l3313_331399

def seashells_problem (initial_seashells current_seashells : ℕ) : Prop :=
  ∃ (given_seashells : ℕ), 
    initial_seashells = current_seashells + given_seashells

theorem solve_seashells_problem (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_problem initial_seashells current_seashells →
  ∃ (given_seashells : ℕ), given_seashells = initial_seashells - current_seashells :=
by
  sorry

end NUMINAMATH_CALUDE_solve_seashells_problem_l3313_331399


namespace NUMINAMATH_CALUDE_f_at_5_l3313_331366

/-- The polynomial function f(x) = 2x^5 - 5x^4 - 4x^3 + 3x^2 - 524 -/
def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 524

/-- Theorem: The value of f(5) is 2176 -/
theorem f_at_5 : f 5 = 2176 := by
  sorry

end NUMINAMATH_CALUDE_f_at_5_l3313_331366


namespace NUMINAMATH_CALUDE_rational_function_equality_l3313_331377

theorem rational_function_equality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -2) : 
  (x^2 + 5) / (x^3 - 3*x + 2) = 1 / (x + 2) + 2 / (x - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_rational_function_equality_l3313_331377


namespace NUMINAMATH_CALUDE_cube_root_equation_l3313_331308

theorem cube_root_equation (x : ℝ) : 
  x = 2 / (2 - Real.rpow 3 (1/3)) → 
  x = (2 * (2 + Real.rpow 3 (1/3))) / (4 - Real.rpow 9 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_l3313_331308


namespace NUMINAMATH_CALUDE_zilla_savings_proof_l3313_331363

def monthly_savings (total_earnings rent other_expenses : ℝ) : ℝ :=
  total_earnings - rent - other_expenses

theorem zilla_savings_proof 
  (total_earnings : ℝ)
  (rent_percentage : ℝ)
  (rent : ℝ)
  (h1 : rent_percentage = 0.07)
  (h2 : rent = 133)
  (h3 : rent = total_earnings * rent_percentage)
  (h4 : let other_expenses := total_earnings / 2;
        monthly_savings total_earnings rent other_expenses = 817) : 
  ∃ (savings : ℝ), savings = 817 ∧ savings = monthly_savings total_earnings rent (total_earnings / 2) :=
sorry

end NUMINAMATH_CALUDE_zilla_savings_proof_l3313_331363


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l3313_331365

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 8 / 65 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l3313_331365


namespace NUMINAMATH_CALUDE_inequality_proof_l3313_331356

theorem inequality_proof (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  2 * Real.sqrt ((1 - x^2) * (1 - y^2)) ≤ 2 * (1 - x) * (1 - y) + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3313_331356


namespace NUMINAMATH_CALUDE_prob_not_sunny_l3313_331326

/-- Given that the probability of a sunny day is 5/7, 
    prove that the probability of a not sunny day is 2/7 -/
theorem prob_not_sunny (prob_sunny : ℚ) (h : prob_sunny = 5 / 7) :
  1 - prob_sunny = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_sunny_l3313_331326


namespace NUMINAMATH_CALUDE_correct_propositions_count_l3313_331332

/-- Represents a proposition about regression analysis -/
inductive RegressionProposition
  | residualSumOfSquares
  | correlationCoefficient
  | scatterPlotPoints
  | randomError

/-- Determines if a given proposition is correct -/
def is_correct (prop : RegressionProposition) : Bool :=
  match prop with
  | .residualSumOfSquares => true
  | .correlationCoefficient => false
  | .scatterPlotPoints => false
  | .randomError => true

/-- The set of all propositions -/
def all_propositions : List RegressionProposition :=
  [.residualSumOfSquares, .correlationCoefficient, .scatterPlotPoints, .randomError]

/-- Counts the number of correct propositions -/
def count_correct_propositions : Nat :=
  all_propositions.filter is_correct |>.length

/-- Theorem stating that the number of correct propositions is 2 -/
theorem correct_propositions_count :
  count_correct_propositions = 2 := by sorry

end NUMINAMATH_CALUDE_correct_propositions_count_l3313_331332


namespace NUMINAMATH_CALUDE_time_to_paint_one_room_l3313_331310

theorem time_to_paint_one_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (time_for_remaining : ℕ) 
  (h1 : total_rooms = 9) 
  (h2 : painted_rooms = 5) 
  (h3 : time_for_remaining = 32) :
  (time_for_remaining / (total_rooms - painted_rooms) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_room_l3313_331310


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l3313_331306

/-- Represents the colors of balls in the urn -/
inductive Color
| Red
| Blue
| Green

/-- Represents the state of the urn -/
structure UrnState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The operation of drawing a ball and adding a matching one -/
def draw_and_add (state : UrnState) : UrnState → Prop :=
  sorry

/-- Performs the draw_and_add operation n times -/
def perform_operations (n : ℕ) (initial : UrnState) : UrnState → Prop :=
  sorry

/-- The probability of a specific final state after n operations -/
noncomputable def probability_of_state (n : ℕ) (initial final : UrnState) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initial_state : UrnState := ⟨2, 1, 0⟩
  let final_state : UrnState := ⟨3, 3, 3⟩
  probability_of_state 6 initial_state final_state = 2/7 :=
by
  sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l3313_331306


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3313_331388

theorem geometric_sequence_ninth_term :
  let a₁ : ℚ := 5
  let r : ℚ := 3/4
  let n : ℕ := 9
  let aₙ : ℕ → ℚ := λ k => a₁ * r^(k - 1)
  aₙ n = 32805/65536 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3313_331388


namespace NUMINAMATH_CALUDE_purple_balls_count_l3313_331397

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 30 ∧
  yellow = 8 ∧
  red = 9 ∧
  prob = 88/100 ∧
  prob = (white + green + yellow : ℚ) / total →
  total - (white + green + yellow + red) = 0 :=
by sorry

end NUMINAMATH_CALUDE_purple_balls_count_l3313_331397


namespace NUMINAMATH_CALUDE_circle_properties_l3313_331315

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Define the point P
def P : ℝ × ℝ := (9, 1)

-- Theorem statement
theorem circle_properties :
  -- 1. Common chord equation
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → 4*x + 3*y - 23 = 0) ∧
  -- 2. Length of common chord
  (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧
    ∃ c d : ℝ, C₁ c d ∧ C₂ c d ∧ (a ≠ c ∨ b ≠ d) ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 2 * 7^(1/2 : ℝ)) ∧
  -- 3. Tangent lines
  (∀ x y : ℝ, (x = 9 ∨ 9*x + 40*y - 121 = 0) →
    ((x - P.1)^2 + (y - P.2)^2 = 0 ∨
     ∃ t : ℝ, C₂ (x + t) (y + t * (y - P.2) / (x - P.1)) ∧
              (∀ s : ℝ, s ≠ t → ¬C₂ (x + s) (y + s * (y - P.2) / (x - P.1))))) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3313_331315


namespace NUMINAMATH_CALUDE_dave_winfield_home_runs_l3313_331342

theorem dave_winfield_home_runs :
  let aaron_hr : ℕ := 755
  let winfield_hr : ℕ := 465
  aaron_hr = 2 * winfield_hr - 175 →
  winfield_hr = 465 :=
by sorry

end NUMINAMATH_CALUDE_dave_winfield_home_runs_l3313_331342


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3313_331355

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 36 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3313_331355


namespace NUMINAMATH_CALUDE_ellipse_equation_l3313_331321

/-- Given an ellipse centered at the origin with foci on the x-axis,
    focal length 4, and eccentricity √2/2, its equation is x²/8 + y²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let focal_length : ℝ := 4
  let eccentricity : ℝ := Real.sqrt 2 / 2
  x^2 / 8 + y^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3313_331321


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3313_331349

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3*m - 1}
def B (m : ℝ) : Set ℝ := {2*m, -3}

theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, A m ∩ B m = {-3} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l3313_331349


namespace NUMINAMATH_CALUDE_square_of_difference_positive_l3313_331317

theorem square_of_difference_positive {a b : ℝ} (h : a ≠ b) : (a - b)^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_positive_l3313_331317


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l3313_331333

theorem cubic_root_equation_solutions :
  {x : ℝ | (15 * x - 1) ^ (1/3) + (13 * x + 1) ^ (1/3) = 4 * x ^ (1/3)} =
  {0, 1/14, -1/12} := by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l3313_331333


namespace NUMINAMATH_CALUDE_football_team_handedness_l3313_331380

theorem football_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 31)
  (h3 : right_handed = 57)
  (h4 : throwers ≤ right_handed) : 
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_football_team_handedness_l3313_331380


namespace NUMINAMATH_CALUDE_triangle_property_l3313_331345

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c / 2 = t.b - t.a * Real.cos t.C) : 
  (Real.cos t.A = 1 / 2) ∧ 
  (t.a = Real.sqrt 15 → t.b = 4 → t.c^2 - 4*t.c + 1 = 0) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_triangle_property_l3313_331345


namespace NUMINAMATH_CALUDE_stock_price_change_l3313_331361

theorem stock_price_change (x : ℝ) : 
  (1 - x / 100) * 1.1 = 1 + 4.499999999999993 / 100 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l3313_331361


namespace NUMINAMATH_CALUDE_pants_price_decrease_percentage_l3313_331347

theorem pants_price_decrease_percentage (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) : 
  purchase_price = 81 →
  markup_percentage = 0.25 →
  gross_profit = 5.40 →
  let original_price := purchase_price / (1 - markup_percentage)
  let decreased_price := original_price - gross_profit
  let decrease_amount := original_price - decreased_price
  (decrease_amount / original_price) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_decrease_percentage_l3313_331347


namespace NUMINAMATH_CALUDE_min_value_expression_l3313_331303

theorem min_value_expression (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  (a + b^2) * (a^2 - b) / (a * b) = 14 ∧
  ∀ (p q : ℕ), p > 0 → q > 0 → p ≠ q →
    (p + q^2) * (p^2 - q) / (p * q) ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3313_331303


namespace NUMINAMATH_CALUDE_max_y_coordinate_difference_l3313_331313

-- Define the two functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := 2 + x^2 + x^3

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Define the y-coordinates of the intersection points
def y_coordinates : Set ℝ := {y : ℝ | ∃ x ∈ intersection_points, f x = y}

-- Theorem statement
theorem max_y_coordinate_difference :
  ∃ (y1 y2 : ℝ), y1 ∈ y_coordinates ∧ y2 ∈ y_coordinates ∧
  ∀ (z1 z2 : ℝ), z1 ∈ y_coordinates → z2 ∈ y_coordinates →
  |y1 - y2| ≥ |z1 - z2| ∧ |y1 - y2| = 2 :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_difference_l3313_331313


namespace NUMINAMATH_CALUDE_probability_two_slate_rocks_l3313_331319

/-- The probability of selecting two slate rocks from a field with 12 slate rocks, 
    17 pumice rocks, and 8 granite rocks, when choosing 2 rocks at random without replacement. -/
theorem probability_two_slate_rocks (slate : ℕ) (pumice : ℕ) (granite : ℕ) 
  (h_slate : slate = 12) (h_pumice : pumice = 17) (h_granite : granite = 8) :
  let total := slate + pumice + granite
  (slate / total) * ((slate - 1) / (total - 1)) = 132 / 1332 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_slate_rocks_l3313_331319


namespace NUMINAMATH_CALUDE_allie_wildflowers_l3313_331327

/-- The number of wildflowers Allie picked -/
def total_flowers : ℕ := 44

/-- The number of yellow and white flowers -/
def yellow_white : ℕ := 13

/-- The number of red and yellow flowers -/
def red_yellow : ℕ := 17

/-- The number of red and white flowers -/
def red_white : ℕ := 14

/-- The difference between red and white flowers -/
def red_white_diff : ℕ := 4

theorem allie_wildflowers :
  total_flowers = yellow_white + red_yellow + red_white :=
by sorry

end NUMINAMATH_CALUDE_allie_wildflowers_l3313_331327


namespace NUMINAMATH_CALUDE_andrew_family_mask_duration_l3313_331384

/-- Calculates the number of days a package of masks will last for a family -/
def maskDuration (totalMasks : ℕ) (familySize : ℕ) (daysPerMask : ℕ) : ℕ :=
  (totalMasks / familySize) * daysPerMask

/-- Proves that for Andrew's family, 100 masks will last 80 days -/
theorem andrew_family_mask_duration :
  maskDuration 100 5 4 = 80 := by
  sorry

#eval maskDuration 100 5 4

end NUMINAMATH_CALUDE_andrew_family_mask_duration_l3313_331384


namespace NUMINAMATH_CALUDE_square_pyramid_frustum_volume_ratio_is_correct_l3313_331323

def square_pyramid_frustum_volume_ratio : ℚ :=
  let base_edge : ℚ := 24
  let altitude : ℚ := 10
  let small_pyramid_altitude_ratio : ℚ := 1/3
  
  let original_volume : ℚ := (1/3) * base_edge^2 * altitude
  let small_pyramid_base_edge : ℚ := base_edge * small_pyramid_altitude_ratio
  let small_pyramid_volume : ℚ := (1/3) * small_pyramid_base_edge^2 * (altitude * small_pyramid_altitude_ratio)
  let frustum_volume : ℚ := original_volume - small_pyramid_volume
  
  frustum_volume / original_volume

theorem square_pyramid_frustum_volume_ratio_is_correct :
  square_pyramid_frustum_volume_ratio = 924/960 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_frustum_volume_ratio_is_correct_l3313_331323


namespace NUMINAMATH_CALUDE_election_margin_l3313_331376

theorem election_margin (total_votes : ℕ) (vote_swing : ℕ) (final_margin_percent : ℚ) : 
  total_votes = 15000 →
  vote_swing = 3000 →
  final_margin_percent = 20 →
  let initial_winner_votes := (total_votes + vote_swing) / 2 + vote_swing / 2
  let initial_loser_votes := (total_votes - vote_swing) / 2 - vote_swing / 2
  let initial_margin := initial_winner_votes - initial_loser_votes
  initial_margin * 100 / total_votes = final_margin_percent :=
by sorry

end NUMINAMATH_CALUDE_election_margin_l3313_331376


namespace NUMINAMATH_CALUDE_B_power_101_l3313_331387

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 0, 1],
    ![0, 0, 0],
    ![-1, 0, 0]]

theorem B_power_101 : B^101 = B := by sorry

end NUMINAMATH_CALUDE_B_power_101_l3313_331387


namespace NUMINAMATH_CALUDE_perpendicular_impossibility_l3313_331329

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)
variable (non_coincident : Line → Line → Prop)

-- State the theorem
theorem perpendicular_impossibility
  (a b : Line) (α : Plane) (P : Point)
  (h1 : non_coincident a b)
  (h2 : perpendicular a α)
  (h3 : intersect a b P) :
  ¬ (perpendicular b α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_impossibility_l3313_331329


namespace NUMINAMATH_CALUDE_at_least_one_non_integer_distance_l3313_331350

/-- Given four points A, B, C, D on a plane with specified distances,
    prove that at least one of BD or CD is not an integer. -/
theorem at_least_one_non_integer_distance
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (h_AB : dist A B = 1)
  (h_BC : dist B C = 9)
  (h_CA : dist C A = 9)
  (h_AD : dist A D = 7) :
  ¬(∃ (bd cd : ℤ), dist B D = bd ∧ dist C D = cd) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_non_integer_distance_l3313_331350


namespace NUMINAMATH_CALUDE_final_wage_calculation_l3313_331341

/-- Calculates the final wage after a raise and a pay cut -/
theorem final_wage_calculation (initial_wage : ℝ) (raise_percentage : ℝ) (pay_cut_percentage : ℝ) :
  initial_wage = 10 →
  raise_percentage = 0.2 →
  pay_cut_percentage = 0.75 →
  initial_wage * (1 + raise_percentage) * pay_cut_percentage = 9 := by
  sorry

#check final_wage_calculation

end NUMINAMATH_CALUDE_final_wage_calculation_l3313_331341


namespace NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l3313_331385

theorem x_squared_plus_nine_y_squared (x y : ℝ) 
  (h1 : x + 3 * y = 5) (h2 : x * y = -8) : 
  x^2 + 9 * y^2 = 73 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l3313_331385


namespace NUMINAMATH_CALUDE_square_side_increase_l3313_331300

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let b := 2 * a
  let c := b * (1 + 60 / 100)
  c^2 = (a^2 + b^2) * (1 + 104.8 / 100) :=
by sorry

end NUMINAMATH_CALUDE_square_side_increase_l3313_331300


namespace NUMINAMATH_CALUDE_number_of_tables_bought_l3313_331383

/-- Proves that the number of tables bought at the cost price is 15, given the conditions -/
theorem number_of_tables_bought (C S : ℝ) (N : ℕ) : 
  N * C = 20 * S → -- The cost price of N tables equals the selling price of 20 tables
  S = 0.75 * C →   -- The selling price is 75% of the cost price (due to 25% loss)
  N = 15 :=
by sorry

end NUMINAMATH_CALUDE_number_of_tables_bought_l3313_331383


namespace NUMINAMATH_CALUDE_two_thirds_bucket_fill_time_l3313_331352

/-- Given a bucket that takes 3 minutes to fill completely, 
    prove that it takes 2 minutes to fill two-thirds of the bucket. -/
theorem two_thirds_bucket_fill_time :
  let total_time : ℝ := 3  -- Time to fill the entire bucket
  let fraction_to_fill : ℝ := 2/3  -- Fraction of the bucket we want to fill
  (fraction_to_fill * total_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_bucket_fill_time_l3313_331352


namespace NUMINAMATH_CALUDE_fraction_leading_zeros_l3313_331389

-- Define the fraction
def fraction : ℚ := 1 / (2^4 * 5^7)

-- Define a function to count leading zeros in a decimal representation
def countLeadingZeros (q : ℚ) : ℕ :=
  sorry -- Implementation details omitted

-- Theorem statement
theorem fraction_leading_zeros :
  countLeadingZeros fraction = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_leading_zeros_l3313_331389


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3313_331348

theorem magnitude_of_complex_number (z : ℂ) : z = (5 * Complex.I) / (2 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3313_331348


namespace NUMINAMATH_CALUDE_frog_prob_theorem_l3313_331309

/-- A triangular pond with 9 regions -/
structure TriangularPond :=
  (regions : Fin 9)

/-- The frog's position in the pond -/
inductive Position
  | A
  | Adjacent

/-- The probability of the frog being in a specific position after k jumps -/
def probability (k : ℕ) (pos : Position) : ℝ :=
  sorry

/-- The probability of the frog being in region A after 2022 jumps -/
def prob_in_A_after_2022 : ℝ :=
  probability 2022 Position.A

/-- The theorem stating the probability of the frog being in region A after 2022 jumps -/
theorem frog_prob_theorem :
  prob_in_A_after_2022 = 2/9 * (1/2)^1010 + 1/9 :=
sorry

end NUMINAMATH_CALUDE_frog_prob_theorem_l3313_331309


namespace NUMINAMATH_CALUDE_pi_estimation_l3313_331393

theorem pi_estimation (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let π_estimate : ℚ := 4 * (m : ℚ) / (n : ℚ) + 2
  π_estimate = 78 / 25 := by
  sorry

end NUMINAMATH_CALUDE_pi_estimation_l3313_331393


namespace NUMINAMATH_CALUDE_equation_equivalence_l3313_331336

theorem equation_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 9 = 0) ↔ ((x - 1)^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3313_331336


namespace NUMINAMATH_CALUDE_polar_equation_C_max_area_OAB_l3313_331353

-- Define the curves C, C1, and C2
def C (x y : ℝ) : Prop := x^2 + y^2 = |x| + y ∧ y > 0

def C1 (x y t α : ℝ) : Prop := x = t * Real.cos α ∧ y = t * Real.sin α ∧ t > 0

def C2 (x y t α : ℝ) : Prop := x = -t * Real.sin α ∧ y = t * Real.cos α ∧ t > 0 ∧ 0 < α ∧ α < Real.pi / 2

-- Theorem for the polar coordinate equation of C
theorem polar_equation_C : 
  ∀ (ρ θ : ℝ), 0 < θ ∧ θ < Real.pi → 
  (C (ρ * Real.cos θ) (ρ * Real.sin θ) ↔ ρ = |Real.cos θ| + Real.sin θ) :=
sorry

-- Theorem for the maximum area of triangle OAB
theorem max_area_OAB :
  ∃ (x₁ y₁ x₂ y₂ t₁ t₂ α₁ α₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧
    C1 x₁ y₁ t₁ α₁ ∧ C2 x₂ y₂ t₂ α₂ ∧
    (∀ (x₃ y₃ x₄ y₄ t₃ t₄ α₃ α₄ : ℝ),
      C x₃ y₃ ∧ C x₄ y₄ ∧ C1 x₃ y₃ t₃ α₃ ∧ C2 x₄ y₄ t₄ α₄ →
      (1 / 2 : ℝ) * |x₁ * y₂ - x₂ * y₁| ≥ (1 / 2 : ℝ) * |x₃ * y₄ - x₄ * y₃|) ∧
    (1 / 2 : ℝ) * |x₁ * y₂ - x₂ * y₁| = 1 :=
sorry

end NUMINAMATH_CALUDE_polar_equation_C_max_area_OAB_l3313_331353


namespace NUMINAMATH_CALUDE_total_fish_caught_l3313_331316

/-- Given 20 fishermen, where 19 caught 400 fish each and the 20th caught 2400,
    prove that the total number of fish caught is 10000. -/
theorem total_fish_caught (total_fishermen : Nat) (fish_per_fisherman : Nat) (fish_last_fisherman : Nat) :
  total_fishermen = 20 →
  fish_per_fisherman = 400 →
  fish_last_fisherman = 2400 →
  (total_fishermen - 1) * fish_per_fisherman + fish_last_fisherman = 10000 :=
by sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3313_331316


namespace NUMINAMATH_CALUDE_power_division_23_l3313_331390

theorem power_division_23 : (23 : ℕ)^11 / (23 : ℕ)^8 = 12167 := by
  sorry

end NUMINAMATH_CALUDE_power_division_23_l3313_331390


namespace NUMINAMATH_CALUDE_increasing_sin_plus_linear_range_of_a_l3313_331328

/-- A function f : ℝ → ℝ is increasing if for all x₁ x₂, x₁ < x₂ implies f x₁ < f x₂ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- The main theorem: if y = sin x + ax is an increasing function on ℝ, then a ≥ 1 -/
theorem increasing_sin_plus_linear (a : ℝ) :
  IsIncreasing (fun x => Real.sin x + a * x) → a ≥ 1 := by
  sorry

/-- The range of a is [1, +∞) -/
theorem range_of_a (a : ℝ) :
  (IsIncreasing (fun x => Real.sin x + a * x) ↔ a ∈ Set.Ici 1) := by
  sorry

end NUMINAMATH_CALUDE_increasing_sin_plus_linear_range_of_a_l3313_331328


namespace NUMINAMATH_CALUDE_optimal_arrangement_maximizes_sum_l3313_331346

/-- The type of arrangements of numbers from 1 to 1999 in a circle -/
def Arrangement := Fin 1999 → Fin 1999

/-- The sum of products of all sets of 10 consecutive numbers in an arrangement -/
def sumOfProducts (a : Arrangement) : ℕ :=
  sorry

/-- The optimal arrangement of numbers -/
def optimalArrangement : Arrangement :=
  fun i => if i.val % 2 = 0 then (1999 - i.val + 1) else (i.val + 1)

/-- Theorem stating that the optimal arrangement maximizes the sum of products -/
theorem optimal_arrangement_maximizes_sum :
  ∀ a : Arrangement, sumOfProducts a ≤ sumOfProducts optimalArrangement :=
sorry

end NUMINAMATH_CALUDE_optimal_arrangement_maximizes_sum_l3313_331346


namespace NUMINAMATH_CALUDE_circle_in_triangle_l3313_331357

/-- The distance traveled by the center of a circle rolling inside a right triangle -/
def distanceTraveled (a b c r : ℝ) : ℝ :=
  (a - 2*r) + (b - 2*r) + (c - 2*r)

theorem circle_in_triangle (a b c r : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 9) (h_b : b = 12) (h_c : c = 15) (h_r : r = 2) :
  distanceTraveled a b c r = 24 := by
sorry

end NUMINAMATH_CALUDE_circle_in_triangle_l3313_331357


namespace NUMINAMATH_CALUDE_interest_payment_time_l3313_331398

-- Define the principal amount
def principal : ℝ := 8000

-- Define the interest rates
def rate1 : ℝ := 0.08
def rate2 : ℝ := 0.10
def rate3 : ℝ := 0.12

-- Define the time periods
def time1 : ℝ := 4
def time2 : ℝ := 6

-- Define the total interest paid
def totalInterest : ℝ := 12160

-- Function to calculate interest
def calculateInterest (p : ℝ) (r : ℝ) (t : ℝ) : ℝ := p * r * t

-- Theorem statement
theorem interest_payment_time :
  ∃ t : ℝ, 
    calculateInterest principal rate1 time1 +
    calculateInterest principal rate2 time2 +
    calculateInterest principal rate3 (t - (time1 + time2)) = totalInterest ∧
    t = 15 := by sorry

end NUMINAMATH_CALUDE_interest_payment_time_l3313_331398
