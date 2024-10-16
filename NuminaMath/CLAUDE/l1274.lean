import Mathlib

namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_sufficient_not_necessary_l1274_127440

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1: Range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2: Range of a for which ¬p is sufficient but not necessary for ¬q
theorem range_of_a_for_not_p_sufficient_not_necessary (a : ℝ) :
  (∀ x, ¬(p x a) → ¬(q x)) ∧ (∃ x, ¬(q x) ∧ p x a) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_sufficient_not_necessary_l1274_127440


namespace NUMINAMATH_CALUDE_grassy_plot_width_l1274_127460

/-- Proves that the width of a rectangular grassy plot is 55 meters, given specific conditions -/
theorem grassy_plot_width : 
  ∀ (length width path_width : ℝ) (cost_per_sq_meter cost_total : ℝ),
  length = 110 →
  path_width = 2.5 →
  cost_per_sq_meter = 0.5 →
  cost_total = 425 →
  ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * cost_per_sq_meter = cost_total →
  width = 55 := by
sorry

end NUMINAMATH_CALUDE_grassy_plot_width_l1274_127460


namespace NUMINAMATH_CALUDE_animal_distance_calculation_l1274_127498

/-- Calculates the total distance covered by a fox, rabbit, and deer running at their maximum speeds for 120 minutes. -/
theorem animal_distance_calculation :
  let fox_speed : ℝ := 50  -- km/h
  let rabbit_speed : ℝ := 60  -- km/h
  let deer_speed : ℝ := 80  -- km/h
  let time_hours : ℝ := 120 / 60  -- Convert 120 minutes to hours
  let fox_distance := fox_speed * time_hours
  let rabbit_distance := rabbit_speed * time_hours
  let deer_distance := deer_speed * time_hours
  let total_distance := fox_distance + rabbit_distance + deer_distance
  total_distance = 380  -- km
  := by sorry

end NUMINAMATH_CALUDE_animal_distance_calculation_l1274_127498


namespace NUMINAMATH_CALUDE_binomial_30_3_l1274_127457

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l1274_127457


namespace NUMINAMATH_CALUDE_third_cube_edge_l1274_127417

theorem third_cube_edge (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  ∃ x : ℝ, x ^ 3 + a ^ 3 + b ^ 3 = c ^ 3 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_cube_edge_l1274_127417


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1274_127470

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -5) →
  a = 5/4 ∨ a = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1274_127470


namespace NUMINAMATH_CALUDE_additional_chicken_wings_l1274_127486

theorem additional_chicken_wings 
  (num_friends : ℕ) 
  (pre_cooked_wings : ℕ) 
  (wings_per_friend : ℕ) : 
  num_friends = 4 → 
  pre_cooked_wings = 9 → 
  wings_per_friend = 4 → 
  num_friends * wings_per_friend - pre_cooked_wings = 7 := by
  sorry

end NUMINAMATH_CALUDE_additional_chicken_wings_l1274_127486


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1274_127443

theorem matrix_equation_solution :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  M^3 - 3 • M^2 + 2 • M = !![8, 16; 4, 8] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1274_127443


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l1274_127476

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143) (h2 : num_nieces = 11) :
  ∃ (sandwiches_per_niece : ℕ), 
    sandwiches_per_niece * num_nieces = total_sandwiches ∧ 
    sandwiches_per_niece = 13 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l1274_127476


namespace NUMINAMATH_CALUDE_opposite_numbers_system_solution_l1274_127427

theorem opposite_numbers_system_solution :
  ∀ (x y k : ℝ),
  (x = -y) →
  (2 * x + 5 * y = k) →
  (x - 3 * y = 16) →
  (k = -12) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_system_solution_l1274_127427


namespace NUMINAMATH_CALUDE_michelle_payment_l1274_127451

/-- Calculates the total cost of a cell phone plan --/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (texts_sent : ℚ) (hours_used : ℚ) : ℚ :=
  let text_charges := text_cost * texts_sent
  let extra_hours := max (hours_used - included_hours) 0
  let extra_minute_charges := extra_minute_cost * (extra_hours * 60)
  base_cost + text_charges + extra_minute_charges

/-- Theorem stating that Michelle's total payment is $54.00 --/
theorem michelle_payment :
  let base_cost : ℚ := 25
  let included_hours : ℚ := 40
  let text_cost : ℚ := 1 / 10
  let extra_minute_cost : ℚ := 15 / 100
  let texts_sent : ℚ := 200
  let hours_used : ℚ := 41
  calculate_total_cost base_cost included_hours text_cost extra_minute_cost texts_sent hours_used = 54 :=
by
  sorry


end NUMINAMATH_CALUDE_michelle_payment_l1274_127451


namespace NUMINAMATH_CALUDE_solve_jewelry_problem_l1274_127467

/-- Represents the jewelry store inventory problem -/
def jewelry_problem (necklace_capacity : ℕ) (current_necklaces : ℕ) 
  (ring_capacity : ℕ) (current_rings : ℕ) (bracelet_capacity : ℕ) 
  (necklace_cost : ℕ) (ring_cost : ℕ) (bracelet_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (current_bracelets : ℕ),
    necklace_capacity = 12 ∧
    current_necklaces = 5 ∧
    ring_capacity = 30 ∧
    current_rings = 18 ∧
    bracelet_capacity = 15 ∧
    necklace_cost = 4 ∧
    ring_cost = 10 ∧
    bracelet_cost = 5 ∧
    total_cost = 183 ∧
    (necklace_capacity - current_necklaces) * necklace_cost + 
    (ring_capacity - current_rings) * ring_cost + 
    (bracelet_capacity - current_bracelets) * bracelet_cost = total_cost ∧
    current_bracelets = 8

theorem solve_jewelry_problem :
  jewelry_problem 12 5 30 18 15 4 10 5 183 :=
sorry

end NUMINAMATH_CALUDE_solve_jewelry_problem_l1274_127467


namespace NUMINAMATH_CALUDE_longest_segment_in_pie_sector_l1274_127469

theorem longest_segment_in_pie_sector (d : ℝ) (h : d = 12) :
  let r := d / 2
  let sector_angle := 2 * Real.pi / 3
  let chord_length := 2 * r * Real.sin (sector_angle / 2)
  chord_length ^ 2 = 108 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_pie_sector_l1274_127469


namespace NUMINAMATH_CALUDE_append_digits_to_perfect_square_l1274_127415

/-- The number formed by 99 nines in a row -/
def X : ℕ := 10^99 - 1

/-- Theorem stating that there exists a natural number n such that 
    X * 10^100 ≤ n^2 < X * 10^100 + 10^100 -/
theorem append_digits_to_perfect_square :
  ∃ n : ℕ, X * 10^100 ≤ n^2 ∧ n^2 < X * 10^100 + 10^100 := by
  sorry

end NUMINAMATH_CALUDE_append_digits_to_perfect_square_l1274_127415


namespace NUMINAMATH_CALUDE_sam_initial_pennies_l1274_127412

/-- The number of pennies Sam found -/
def pennies_found : ℕ := 93

/-- The total number of pennies Sam has now -/
def total_pennies : ℕ := 191

/-- The initial number of pennies Sam had -/
def initial_pennies : ℕ := total_pennies - pennies_found

theorem sam_initial_pennies : initial_pennies = 98 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_pennies_l1274_127412


namespace NUMINAMATH_CALUDE_new_person_weight_l1274_127488

/-- Proves that the weight of a new person is 65 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1274_127488


namespace NUMINAMATH_CALUDE_domino_swap_incorrect_l1274_127479

/-- Represents a domino with a value from 0 to 9 -/
def Domino : Type := Fin 10

/-- Represents a multiplication problem with 5 dominoes -/
structure DominoMultiplication :=
  (d1 d2 d3 d4 d5 : Domino)

/-- Checks if the domino multiplication is correct -/
def isCorrectMultiplication (dm : DominoMultiplication) : Prop :=
  (dm.d1.val * 10 + dm.d2.val) * dm.d3.val = dm.d4.val * 10 + dm.d5.val

/-- Swaps two dominoes in the multiplication -/
def swapDominoes (dm : DominoMultiplication) (i j : Fin 5) : DominoMultiplication :=
  match i, j with
  | 0, 1 => { d1 := dm.d2, d2 := dm.d1, d3 := dm.d3, d4 := dm.d4, d5 := dm.d5 }
  | 0, 2 => { d1 := dm.d3, d2 := dm.d2, d3 := dm.d1, d4 := dm.d4, d5 := dm.d5 }
  | 0, 3 => { d1 := dm.d4, d2 := dm.d2, d3 := dm.d3, d4 := dm.d1, d5 := dm.d5 }
  | 0, 4 => { d1 := dm.d5, d2 := dm.d2, d3 := dm.d3, d4 := dm.d4, d5 := dm.d1 }
  | 1, 2 => { d1 := dm.d1, d2 := dm.d3, d3 := dm.d2, d4 := dm.d4, d5 := dm.d5 }
  | 1, 3 => { d1 := dm.d1, d2 := dm.d4, d3 := dm.d3, d4 := dm.d2, d5 := dm.d5 }
  | 1, 4 => { d1 := dm.d1, d2 := dm.d5, d3 := dm.d3, d4 := dm.d4, d5 := dm.d2 }
  | 2, 3 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d4, d4 := dm.d3, d5 := dm.d5 }
  | 2, 4 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d5, d4 := dm.d4, d5 := dm.d3 }
  | 3, 4 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d3, d4 := dm.d5, d5 := dm.d4 }
  | _, _ => dm  -- For any other combination, return the original multiplication

theorem domino_swap_incorrect
  (dm : DominoMultiplication)
  (h : isCorrectMultiplication dm)
  (i j : Fin 5)
  (hne : i ≠ j) :
  ¬(isCorrectMultiplication (swapDominoes dm i j)) :=
by sorry

end NUMINAMATH_CALUDE_domino_swap_incorrect_l1274_127479


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1274_127405

open Real

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_pos : ∀ x, x > 0 → f x > 0) 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_eq : ∀ x, x > 0 → deriv f (a / x) = x / f x) :
  ∃ b : ℝ, b > 0 ∧ ∀ x, x > 0 → f x = a^(1 - a/b) * x^(a/b) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1274_127405


namespace NUMINAMATH_CALUDE_union_equals_A_implies_m_zero_or_three_l1274_127458

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equals_A_implies_m_zero_or_three (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_implies_m_zero_or_three_l1274_127458


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1274_127433

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(2*x) * 50^x = 250^3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1274_127433


namespace NUMINAMATH_CALUDE_xyz_product_l1274_127441

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 360)
  (eq2 : y * (z + x) = 405)
  (eq3 : z * (x + y) = 450) :
  x * y * z = 2433 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1274_127441


namespace NUMINAMATH_CALUDE_right_triangle_identification_l1274_127423

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 5 12 13) ∧
  (¬ is_right_triangle 6 8 12) ∧
  (¬ is_right_triangle 6 12 15) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l1274_127423


namespace NUMINAMATH_CALUDE_trapezoid_area_increase_l1274_127490

/-- Represents a trapezoid with a given height -/
structure Trapezoid where
  height : ℝ

/-- Calculates the increase in area when both bases of a trapezoid are increased by a given amount -/
def area_increase (t : Trapezoid) (base_increase : ℝ) : ℝ :=
  t.height * base_increase

/-- Theorem: The area increase of a trapezoid with height 6 cm when both bases are increased by 4 cm is 24 square centimeters -/
theorem trapezoid_area_increase :
  let t : Trapezoid := { height := 6 }
  area_increase t 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_increase_l1274_127490


namespace NUMINAMATH_CALUDE_lucy_cookie_sales_l1274_127407

theorem lucy_cookie_sales : ∀ (first_round second_round total : ℕ),
  first_round = 34 →
  second_round = 27 →
  total = first_round + second_round →
  total = 61 := by
  sorry

end NUMINAMATH_CALUDE_lucy_cookie_sales_l1274_127407


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l1274_127472

/-- Represents the number of employees in each age group -/
structure AgeGroup where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Represents the ratio of employees in each age group -/
structure AgeRatio where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Calculates the stratified sample size for each age group -/
def stratifiedSample (totalPopulation : ℕ) (sampleSize : ℕ) (ratio : AgeRatio) : AgeGroup :=
  let totalRatio := ratio.middleAged + ratio.young + ratio.elderly
  { middleAged := sampleSize * ratio.middleAged / totalRatio,
    young := sampleSize * ratio.young / totalRatio,
    elderly := sampleSize * ratio.elderly / totalRatio }

theorem correct_stratified_sample :
  let totalPopulation : ℕ := 3200
  let sampleSize : ℕ := 400
  let ratio : AgeRatio := { middleAged := 5, young := 3, elderly := 2 }
  let sample : AgeGroup := stratifiedSample totalPopulation sampleSize ratio
  sample.middleAged = 200 ∧ sample.young = 120 ∧ sample.elderly = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l1274_127472


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1274_127473

variables (a b x y : ℝ)

theorem complex_fraction_simplification :
  (a * x * (3 * a^2 * x^2 + 5 * b^2 * y^2) + b * y * (2 * a^2 * x^2 + 4 * b^2 * y^2)) / (a * x + b * y) = 3 * a^2 * x^2 + 4 * b^2 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1274_127473


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l1274_127425

-- Define the isosceles triangle ∆ABC
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the equation of the line containing the altitude
def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the equation of the line containing side BC
def side_BC_line (x y : ℝ) : Prop := 3*x - y - 1 = 0

-- Define the equation of the circumscribed circle
def circumscribed_circle (x y : ℝ) : Prop := (x - 5/2)^2 + (y + 7/2)^2 = 50/4

theorem isosceles_triangle_properties :
  ∃ C : ℝ × ℝ,
    (∀ x y : ℝ, altitude_line x y → side_BC_line x y) ∧
    (∀ x y : ℝ, circumscribed_circle x y) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l1274_127425


namespace NUMINAMATH_CALUDE_sequence_divisibility_l1274_127446

theorem sequence_divisibility (n : ℤ) : 
  (∃ k : ℤ, 7 * n - 3 = 5 * k) ∧ 
  (∀ m : ℤ, 7 * n - 3 ≠ 3 * m) ↔ 
  ∃ t : ℕ, n = 5 * t - 1 ∧ ∀ m : ℕ, t ≠ 3 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l1274_127446


namespace NUMINAMATH_CALUDE_proposition_ranges_l1274_127496

def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, -3 < x ∧ x < 1 → x^2 + 4*x + 9 - m > 0

def prop_q (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x^2 - 2*m*x + 1 < 0

theorem proposition_ranges (m : ℝ) :
  (prop_p m ↔ m < 5) ∧
  (prop_p m ≠ prop_q m ↔ m ≤ 1 ∨ m ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_proposition_ranges_l1274_127496


namespace NUMINAMATH_CALUDE_angle_325_same_terminal_side_as_neg_35_l1274_127480

/-- 
Given an angle θ in degrees, this function returns true if θ has the same terminal side as -35°.
-/
def hasSameTerminalSideAs (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * 360 + (-35)

/-- 
This theorem states that 325° has the same terminal side as -35° and is between 0° and 360°.
-/
theorem angle_325_same_terminal_side_as_neg_35 :
  hasSameTerminalSideAs 325 ∧ 0 ≤ 325 ∧ 325 < 360 := by
  sorry

end NUMINAMATH_CALUDE_angle_325_same_terminal_side_as_neg_35_l1274_127480


namespace NUMINAMATH_CALUDE_x_seventh_x_n_plus_one_l1274_127435

variable (x : ℝ)

-- Define the conditions
axiom x_is_root : x^2 - x - 1 = 0
axiom x_squared : x^2 = x + 1
axiom x_cubed : x^3 = 2*x + 1
axiom x_fourth : x^4 = 3*x + 2
axiom x_fifth : x^5 = 5*x + 3
axiom x_sixth : x^6 = 8*x + 5

-- Define x^n = αx + β
variable (n : ℕ) (α β : ℝ)
axiom x_nth : x^n = α*x + β

-- Theorem statements
theorem x_seventh : x^7 = 13*x + 8 := by sorry

theorem x_n_plus_one : x^(n+1) = (α + β)*x + α := by sorry

end NUMINAMATH_CALUDE_x_seventh_x_n_plus_one_l1274_127435


namespace NUMINAMATH_CALUDE_prob_king_then_ten_l1274_127474

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of 10s in a standard deck -/
def NumTens : ℕ := 4

/-- Probability of drawing a King first and then a 10 from a standard deck -/
theorem prob_king_then_ten : 
  (NumKings : ℚ) / StandardDeck * NumTens / (StandardDeck - 1) = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_then_ten_l1274_127474


namespace NUMINAMATH_CALUDE_pizza_fraction_eaten_l1274_127454

/-- The fraction of pizza eaten after n trips, where each trip consumes one-third of the remaining pizza -/
def fractionEaten (n : ℕ) : ℚ :=
  1 - (2/3)^n

/-- The number of trips to the refrigerator -/
def numTrips : ℕ := 6

theorem pizza_fraction_eaten :
  fractionEaten numTrips = 364 / 729 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_eaten_l1274_127454


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l1274_127447

theorem right_triangle_acute_angle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  (a + b = 90) → (a / b = 7 / 2) → min a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l1274_127447


namespace NUMINAMATH_CALUDE_sams_weight_l1274_127434

/-- Given the weights of Tyler, Sam, and Peter, prove Sam's weight -/
theorem sams_weight (tyler sam peter : ℕ) 
  (h1 : tyler = sam + 25)
  (h2 : peter * 2 = tyler)
  (h3 : peter = 65) : 
  sam = 105 := by sorry

end NUMINAMATH_CALUDE_sams_weight_l1274_127434


namespace NUMINAMATH_CALUDE_hallies_art_earnings_l1274_127462

/-- Calculates the total money Hallie makes from her art -/
def total_money (prize : ℕ) (num_paintings : ℕ) (price_per_painting : ℕ) : ℕ :=
  prize + num_paintings * price_per_painting

/-- Proves that Hallie's total earnings from her art is $300 -/
theorem hallies_art_earnings : total_money 150 3 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_hallies_art_earnings_l1274_127462


namespace NUMINAMATH_CALUDE_max_circle_area_in_square_l1274_127442

/-- The area of the maximum size circle inscribed in a square -/
theorem max_circle_area_in_square (square_side : ℝ) (h : square_side = 10) :
  π * (square_side / 2)^2 = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_max_circle_area_in_square_l1274_127442


namespace NUMINAMATH_CALUDE_f_equation_solution_l1274_127466

def f (x : ℝ) : ℝ := 3 * x - 5

theorem f_equation_solution :
  ∃ x : ℝ, 1 = f (x - 6) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_f_equation_solution_l1274_127466


namespace NUMINAMATH_CALUDE_smallest_valid_number_correct_l1274_127419

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit number
  (n % 2 = 0) ∧  -- Even
  (n % 3 = 0) ∧  -- Divisible by 3
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits.toFinset = {1, 2, 3, 4, 9}  -- Uses each digit exactly once

def smallest_valid_number : ℕ := 14932

theorem smallest_valid_number_correct :
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  ((smallest_valid_number / 10) % 10 = 3) :=
by sorry

#eval smallest_valid_number
#eval (smallest_valid_number / 10) % 10

end NUMINAMATH_CALUDE_smallest_valid_number_correct_l1274_127419


namespace NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l1274_127449

theorem abs_a_minus_b_equals_eight (a b : ℚ) 
  (h : |a + b| + (b - 4)^2 = 0) : 
  |a - b| = 8 := by
sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l1274_127449


namespace NUMINAMATH_CALUDE_equation_solution_l1274_127493

theorem equation_solution :
  ∃ (x : ℚ), (x + 36) / 3 = (7 - 2*x) / 6 ∧ x = -65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1274_127493


namespace NUMINAMATH_CALUDE_ryan_analysis_time_l1274_127444

/-- The number of individuals Ryan is analyzing -/
def num_individuals : ℕ := 3

/-- The number of bones in each individual -/
def bones_per_individual : ℕ := 206

/-- The time (in hours) Ryan spends on initial analysis per bone -/
def initial_analysis_time : ℚ := 1

/-- The additional time (in hours) Ryan spends on research per bone -/
def additional_research_time : ℚ := 1/2

/-- The total time Ryan needs for his analysis -/
def total_analysis_time : ℚ :=
  (num_individuals * bones_per_individual) * (initial_analysis_time + additional_research_time)

theorem ryan_analysis_time : total_analysis_time = 927 := by
  sorry

end NUMINAMATH_CALUDE_ryan_analysis_time_l1274_127444


namespace NUMINAMATH_CALUDE_total_coins_l1274_127420

def coin_distribution (x : ℕ) : Prop :=
  ∃ (pete paul : ℕ),
    paul = x ∧
    pete = 3 * x ∧
    pete = x * (x + 1) ∧
    x > 0

theorem total_coins (x : ℕ) (h : coin_distribution x) : x + 3 * x = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_l1274_127420


namespace NUMINAMATH_CALUDE_square_perimeter_l1274_127468

/-- Theorem: A square with an area of 625 cm² has a perimeter of 100 cm. -/
theorem square_perimeter (s : ℝ) (h_area : s^2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1274_127468


namespace NUMINAMATH_CALUDE_max_digit_sum_18_l1274_127409

/-- Represents a digit (1 to 9) -/
def Digit := {d : ℕ // 1 ≤ d ∧ d ≤ 9}

/-- Calculates the value of a number with n identical digits -/
def digitSum (d : Digit) (n : ℕ) : ℕ := d.val * ((10^n - 1) / 9)

/-- The main theorem -/
theorem max_digit_sum_18 :
  ∃ (a b c : Digit) (n₁ n₂ : ℕ+),
    n₁ ≠ n₂ ∧
    digitSum c (2 * n₁) - digitSum b n₁ = (digitSum a n₁)^2 ∧
    digitSum c (2 * n₂) - digitSum b n₂ = (digitSum a n₂)^2 ∧
    ∀ (a' b' c' : Digit),
      (∃ (m₁ m₂ : ℕ+), m₁ ≠ m₂ ∧
        digitSum c' (2 * m₁) - digitSum b' m₁ = (digitSum a' m₁)^2 ∧
        digitSum c' (2 * m₂) - digitSum b' m₂ = (digitSum a' m₂)^2) →
      a'.val + b'.val + c'.val ≤ a.val + b.val + c.val ∧
      a.val + b.val + c.val = 18 :=
by sorry

end NUMINAMATH_CALUDE_max_digit_sum_18_l1274_127409


namespace NUMINAMATH_CALUDE_unique_number_with_18_factors_l1274_127453

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

theorem unique_number_with_18_factors (x : ℕ) : 
  num_factors x = 18 ∧ 
  18 ∣ x ∧ 
  24 ∣ x → 
  x = 288 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_18_factors_l1274_127453


namespace NUMINAMATH_CALUDE_range_of_m_l1274_127413

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 - x₁ + m - 4 = 0 ∧ 
              x₂^2 - x₂ + m - 4 = 0 ∧ 
              x₁ * x₂ < 0

-- Main theorem
theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬p m) :
  m ≤ 1 - Real.sqrt 2 ∨ (1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1274_127413


namespace NUMINAMATH_CALUDE_polynomial_value_equals_one_l1274_127459

theorem polynomial_value_equals_one (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equals_one_l1274_127459


namespace NUMINAMATH_CALUDE_greenToBlueRatioIs2To3_l1274_127411

/-- Represents a box of crayons with different colors -/
structure CrayonBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  pink : ℕ
  green : ℕ
  h1 : total = red + blue + pink + green

/-- Calculates the ratio of green crayons to blue crayons -/
def greenToBlueRatio (box : CrayonBox) : Rat :=
  box.green / box.blue

/-- Theorem stating that for the given crayon box, the ratio of green to blue crayons is 2:3 -/
theorem greenToBlueRatioIs2To3 (box : CrayonBox) 
    (h2 : box.total = 24)
    (h3 : box.red = 8)
    (h4 : box.blue = 6)
    (h5 : box.pink = 6) :
    greenToBlueRatio box = 2 / 3 := by
  sorry

#eval greenToBlueRatio { total := 24, red := 8, blue := 6, pink := 6, green := 4, h1 := rfl }

end NUMINAMATH_CALUDE_greenToBlueRatioIs2To3_l1274_127411


namespace NUMINAMATH_CALUDE_cube_volume_from_painting_cost_l1274_127416

/-- Given a cube with a surface area that costs 343.98 rupees to paint at a rate of 13 paise per square centimeter, prove that the volume of the cube is 9261 cubic centimeters. -/
theorem cube_volume_from_painting_cost (cost : ℚ) (rate : ℚ) (volume : ℕ) : 
  cost = 343.98 ∧ rate = 13 / 100 → volume = 9261 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_painting_cost_l1274_127416


namespace NUMINAMATH_CALUDE_fraction_problem_l1274_127452

theorem fraction_problem (x : ℝ) (y : ℝ) : 
  (1/3 : ℝ) * (1/4 : ℝ) * x = 18 → 
  y * x = 64.8 → 
  y = 3/10 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1274_127452


namespace NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l1274_127430

/-- Represents the archery contest scenario -/
structure ArcheryContest where
  total_shots : Nat
  chelsea_lead : Nat
  chelsea_min_score : Nat
  opponent_min_score : Nat
  bullseye_score : Nat

/-- Calculates the minimum number of bullseyes needed for Chelsea to guarantee a win -/
def min_bullseyes_needed (contest : ArcheryContest) : Nat :=
  let remaining_shots := contest.total_shots / 2
  let max_opponent_gain := remaining_shots * contest.bullseye_score
  let chelsea_gain_per_bullseye := contest.bullseye_score - contest.chelsea_min_score
  ((max_opponent_gain - contest.chelsea_lead) / chelsea_gain_per_bullseye) + 1

/-- Theorem stating that Chelsea needs at least 52 bullseyes to guarantee a win -/
theorem chelsea_needs_52_bullseyes (contest : ArcheryContest) 
  (h1 : contest.total_shots = 120)
  (h2 : contest.chelsea_lead = 60)
  (h3 : contest.chelsea_min_score = 3)
  (h4 : contest.opponent_min_score = 1)
  (h5 : contest.bullseye_score = 10) :
  min_bullseyes_needed contest ≥ 52 := by
  sorry

#eval min_bullseyes_needed { total_shots := 120, chelsea_lead := 60, chelsea_min_score := 3, opponent_min_score := 1, bullseye_score := 10 }

end NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l1274_127430


namespace NUMINAMATH_CALUDE_children_per_seat_l1274_127408

theorem children_per_seat (total_children : ℕ) (total_seats : ℕ) 
  (h1 : total_children = 58) (h2 : total_seats = 29) : 
  total_children / total_seats = 2 := by
sorry

end NUMINAMATH_CALUDE_children_per_seat_l1274_127408


namespace NUMINAMATH_CALUDE_square_not_always_positive_l1274_127482

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l1274_127482


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l1274_127464

theorem product_of_specific_numbers : 469160 * 9999 = 4690696840 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l1274_127464


namespace NUMINAMATH_CALUDE_sequence_bound_l1274_127410

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j, i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l1274_127410


namespace NUMINAMATH_CALUDE_square_sheet_area_decrease_l1274_127432

theorem square_sheet_area_decrease (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * b = 0.1 * 4 * a) → (1 - (a - b)^2 / a^2 = 0.04) := by
  sorry

end NUMINAMATH_CALUDE_square_sheet_area_decrease_l1274_127432


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1274_127484

/-- The equation of a hyperbola passing through (1, 0) with asymptotes y = ±2x -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2/4 = 1

/-- The focus of the parabola y² = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The asymptotes of the hyperbola -/
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola_equation x y → (asymptote_pos x y ∨ asymptote_neg x y)) ∧
  hyperbola_equation parabola_focus.1 parabola_focus.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1274_127484


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1274_127414

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  (d^2 / 2 : ℝ) = 72 := by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1274_127414


namespace NUMINAMATH_CALUDE_worker_pay_calculation_l1274_127400

/-- Calculate the worker's pay given the following conditions:
  * The total period is 60 days
  * The pay rate for working is Rs. 20 per day
  * The deduction rate for idle days is Rs. 3 per day
  * The number of idle days is 40 days
-/
def worker_pay (total_days : ℕ) (work_rate : ℕ) (idle_rate : ℕ) (idle_days : ℕ) : ℕ :=
  let work_days := total_days - idle_days
  let earnings := work_days * work_rate
  let deductions := idle_days * idle_rate
  earnings - deductions

theorem worker_pay_calculation :
  worker_pay 60 20 3 40 = 280 := by
  sorry

end NUMINAMATH_CALUDE_worker_pay_calculation_l1274_127400


namespace NUMINAMATH_CALUDE_gym_purchase_theorem_l1274_127439

/-- Cost calculation for Option 1 -/
def costOption1 (x : ℕ) : ℚ :=
  1500 + 15 * (x - 20)

/-- Cost calculation for Option 2 -/
def costOption2 (x : ℕ) : ℚ :=
  (1500 + 15 * x) * (9/10)

/-- Cost calculation for the most cost-effective option -/
def costEffectiveOption (x : ℕ) : ℚ :=
  1500 + (x - 20) * 15 * (9/10)

theorem gym_purchase_theorem (x : ℕ) (h : x > 20) :
  (costOption1 40 < costOption2 40) ∧
  (costOption1 100 = costOption2 100) ∧
  (costEffectiveOption 40 < min (costOption1 40) (costOption2 40)) :=
by sorry

end NUMINAMATH_CALUDE_gym_purchase_theorem_l1274_127439


namespace NUMINAMATH_CALUDE_cycle_cost_proof_l1274_127448

def cycle_problem (selling_price : ℕ) (gain_percentage : ℕ) : Prop :=
  let original_cost : ℕ := selling_price / 2
  selling_price = original_cost * (100 + gain_percentage) / 100 ∧
  original_cost = 1000

theorem cycle_cost_proof :
  cycle_problem 2000 100 :=
sorry

end NUMINAMATH_CALUDE_cycle_cost_proof_l1274_127448


namespace NUMINAMATH_CALUDE_chicken_crossing_ratio_l1274_127431

theorem chicken_crossing_ratio (initial_feathers final_feathers cars_dodged : ℕ) 
  (h1 : initial_feathers = 5263)
  (h2 : final_feathers = 5217)
  (h3 : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 := by
sorry

end NUMINAMATH_CALUDE_chicken_crossing_ratio_l1274_127431


namespace NUMINAMATH_CALUDE_sequence_formula_l1274_127491

theorem sequence_formula (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, a (n + 1) / a n = (n + 2) / n) →
  a 1 = 1 →
  ∀ n : ℕ+, a n = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l1274_127491


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1274_127497

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₃ + a₅ = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1274_127497


namespace NUMINAMATH_CALUDE_least_α_is_correct_l1274_127436

/-- An isosceles triangle with two equal angles α° and a third angle β° -/
structure IsoscelesTriangle where
  α : ℕ
  β : ℕ
  is_isosceles : α + α + β = 180
  α_prime : Nat.Prime α
  β_prime : Nat.Prime β
  α_ne_β : α ≠ β

/-- The least possible value of α in an isosceles triangle where α and β are distinct primes -/
def least_α : ℕ := 41

theorem least_α_is_correct (t : IsoscelesTriangle) : t.α ≥ least_α := by
  sorry

end NUMINAMATH_CALUDE_least_α_is_correct_l1274_127436


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1274_127471

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 7 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 28 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1274_127471


namespace NUMINAMATH_CALUDE_inequality_solution_l1274_127421

theorem inequality_solution (x : ℝ) : 
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20 ↔
   x < -2 ∨ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ x > 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1274_127421


namespace NUMINAMATH_CALUDE_average_salary_example_l1274_127401

/-- The average salary of 5 people given their individual salaries -/
def average_salary (a b c d e : ℕ) : ℚ :=
  (a + b + c + d + e : ℚ) / 5

/-- Theorem: The average salary of 5 people with salaries 8000, 5000, 14000, 7000, and 9000 is 8200 -/
theorem average_salary_example : average_salary 8000 5000 14000 7000 9000 = 8200 := by
  sorry

#eval average_salary 8000 5000 14000 7000 9000

end NUMINAMATH_CALUDE_average_salary_example_l1274_127401


namespace NUMINAMATH_CALUDE_karen_start_time_l1274_127499

/-- Proves that Karen starts the race 4 minutes late given the specified conditions. -/
theorem karen_start_time (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_lead : ℝ) : 
  karen_speed = 60 →
  tom_speed = 45 →
  tom_distance = 24 →
  karen_lead = 4 →
  (tom_distance / tom_speed - (tom_distance + karen_lead) / karen_speed) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_karen_start_time_l1274_127499


namespace NUMINAMATH_CALUDE_missing_number_proof_l1274_127465

theorem missing_number_proof (numbers : List ℕ) (missing : ℕ) : 
  numbers = [744, 745, 747, 748, 749, 753, 755, 755] →
  (numbers.sum + missing) / 9 = 750 →
  missing = 804 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1274_127465


namespace NUMINAMATH_CALUDE_square_sum_eq_18_l1274_127428

theorem square_sum_eq_18 (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x^2 + y^2 = 18) : 
  x^2 + y^2 = 18 := by
sorry

end NUMINAMATH_CALUDE_square_sum_eq_18_l1274_127428


namespace NUMINAMATH_CALUDE_max_three_digit_quotient_l1274_127489

theorem max_three_digit_quotient :
  ∃ (a b c : ℕ), 
    a > 5 ∧ b > 5 ∧ c > 5 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∀ (x y z : ℕ), 
      x > 5 ∧ y > 5 ∧ z > 5 ∧ 
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      (100 * a + 10 * b + c : ℚ) / (a + b + c) ≥ (100 * x + 10 * y + z : ℚ) / (x + y + z) ∧
    (100 * a + 10 * b + c : ℚ) / (a + b + c) = 41.125 := by
  sorry

end NUMINAMATH_CALUDE_max_three_digit_quotient_l1274_127489


namespace NUMINAMATH_CALUDE_number_count_l1274_127463

theorem number_count (n : ℕ) (S : ℝ) : 
  S / n = 60 →                  -- average of all numbers is 60
  (58 * 6 : ℝ) = S / n * 6 →    -- average of first 6 numbers is 58
  (65 * 6 : ℝ) = S / n * 6 →    -- average of last 6 numbers is 65
  78 = S / n →                  -- 6th number is 78
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_number_count_l1274_127463


namespace NUMINAMATH_CALUDE_triathlon_running_speed_l1274_127438

/-- Calculates the running speed given swimming speed and average speed -/
def calculate_running_speed (swimming_speed : ℝ) (average_speed : ℝ) : ℝ :=
  2 * average_speed - swimming_speed

/-- Proves that given a swimming speed of 1 mph and an average speed of 4 mph,
    the running speed is 7 mph -/
theorem triathlon_running_speed :
  let swimming_speed : ℝ := 1
  let average_speed : ℝ := 4
  calculate_running_speed swimming_speed average_speed = 7 := by
sorry

#eval calculate_running_speed 1 4

end NUMINAMATH_CALUDE_triathlon_running_speed_l1274_127438


namespace NUMINAMATH_CALUDE_lucas_100_mod_5_l1274_127445

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas (n + 1) + lucas n

theorem lucas_100_mod_5 : lucas 99 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucas_100_mod_5_l1274_127445


namespace NUMINAMATH_CALUDE_zach_babysitting_hours_l1274_127492

def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def lawn_mowing_pay : ℕ := 10
def babysitting_rate : ℕ := 7
def current_savings : ℕ := 65
def additional_needed : ℕ := 6

theorem zach_babysitting_hours :
  ∃ (hours : ℕ),
    bike_cost = current_savings + weekly_allowance + lawn_mowing_pay + babysitting_rate * hours + additional_needed ∧
    hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_zach_babysitting_hours_l1274_127492


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_lambda_l1274_127437

/-- Given two 2D vectors a and b, if a + 3b is parallel to 2a - b, then the second component of b is -8/3 -/
theorem parallel_vectors_imply_lambda (a b : ℝ × ℝ) (h : a = (-3, 2) ∧ b.1 = 4) :
  (∃ (k : ℝ), k ≠ 0 ∧ k • (a + 3 • b) = 2 • a - b) → b.2 = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_lambda_l1274_127437


namespace NUMINAMATH_CALUDE_min_value_on_line_l1274_127485

theorem min_value_on_line (x y : ℝ) (h : x + 2*y + 1 = 0) :
  2^x + 4^y ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_on_line_l1274_127485


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1274_127450

/-- 
Given a system of linear equations:
  a * x + b * y - b * z = c
  a * y + b * x - b * z = c
  a * z + b * y - b * x = c
This theorem states that the system has a unique solution if and only if 
a ≠ 0, a - b ≠ 0, and a + b ≠ 0.
-/
theorem unique_solution_condition (a b c : ℝ) :
  (∃! x y z : ℝ, (a * x + b * y - b * z = c) ∧ 
                 (a * y + b * x - b * z = c) ∧ 
                 (a * z + b * y - b * x = c)) ↔ 
  (a ≠ 0 ∧ a - b ≠ 0 ∧ a + b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1274_127450


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1274_127404

theorem lcm_gcf_problem (n : ℕ) :
  Nat.lcm n 12 = 54 ∧ Nat.gcd n 12 = 8 → n = 36 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1274_127404


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l1274_127494

/-- Proves that a 30% price reduction and 80% sales increase results in a 26% revenue increase -/
theorem price_reduction_sales_increase (P S : ℝ) (P_pos : P > 0) (S_pos : S > 0) :
  let new_price := 0.7 * P
  let new_sales := 1.8 * S
  let original_revenue := P * S
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l1274_127494


namespace NUMINAMATH_CALUDE_team_e_not_played_b_l1274_127406

/-- Represents a soccer team in the tournament -/
inductive Team : Type
  | A | B | C | D | E | F

/-- The number of matches played by each team at a certain point -/
def matches_played (t : Team) : ℕ :=
  match t with
  | Team.A => 5
  | Team.B => 4
  | Team.C => 3
  | Team.D => 2
  | Team.E => 1
  | Team.F => 0

/-- Predicate to check if two teams have played against each other -/
def has_played_against (t1 t2 : Team) : Prop :=
  sorry

/-- The total number of teams in the tournament -/
def total_teams : ℕ := 6

/-- The maximum number of matches a team can play in a round-robin tournament -/
def max_matches : ℕ := total_teams - 1

theorem team_e_not_played_b :
  matches_played Team.A = max_matches ∧
  matches_played Team.E = 1 →
  ¬ has_played_against Team.E Team.B :=
by sorry

end NUMINAMATH_CALUDE_team_e_not_played_b_l1274_127406


namespace NUMINAMATH_CALUDE_b_more_stable_than_a_l1274_127418

/-- Represents a shooter in the competition -/
structure Shooter where
  variance : ℝ

/-- Defines the stability of a shooter based on their variance -/
def is_more_stable (a b : Shooter) : Prop :=
  a.variance < b.variance

/-- Theorem stating that shooter B is more stable than shooter A -/
theorem b_more_stable_than_a :
  let shooterA : Shooter := ⟨1.8⟩
  let shooterB : Shooter := ⟨0.7⟩
  is_more_stable shooterB shooterA :=
by
  sorry

end NUMINAMATH_CALUDE_b_more_stable_than_a_l1274_127418


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l1274_127481

/-- Given a parabola y = ax^2 + bx + c passing through points (-2, 9), (4, 9), and (5, 13),
    the x-coordinate of its vertex is 1. -/
theorem parabola_vertex_x_coordinate
  (a b c : ℝ)
  (h1 : a * (-2)^2 + b * (-2) + c = 9)
  (h2 : a * 4^2 + b * 4 + c = 9)
  (h3 : a * 5^2 + b * 5 + c = 13) :
  (∃ y : ℝ, a * 1^2 + b * 1 + c = y ∧
    ∀ x : ℝ, a * x^2 + b * x + c ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l1274_127481


namespace NUMINAMATH_CALUDE_min_draws_for_sum_30_l1274_127426

-- Define the set of integers from 0 to 20
def integerSet : Set ℕ := {n : ℕ | n ≤ 20}

-- Define a function to check if two numbers in a list sum to 30
def hasPairSum30 (list : List ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ list ∧ b ∈ list ∧ a ≠ b ∧ a + b = 30

-- Theorem: The minimum number of integers to guarantee a pair summing to 30 is 10
theorem min_draws_for_sum_30 :
  ∀ (drawn : List ℕ),
    (∀ n ∈ drawn, n ∈ integerSet) →
    (drawn.length ≥ 10 → hasPairSum30 drawn) ∧
    (∃ subset : List ℕ, subset.length = 9 ∧ ∀ n ∈ subset, n ∈ integerSet ∧ ¬hasPairSum30 subset) :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_sum_30_l1274_127426


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l1274_127402

/-- Represents the ratio of concentrate to water in the orange juice mixture -/
def concentrateToWaterRatio : ℚ := 1 / 3

/-- The number of cans of concentrate required -/
def concentrateCans : ℕ := 35

/-- The volume of each can of concentrate in ounces -/
def canSize : ℕ := 12

/-- The number of servings to be prepared -/
def numberOfServings : ℕ := 280

/-- The size of each serving in ounces -/
def servingSize : ℚ := 6

theorem orange_juice_serving_size :
  (concentrateCans * canSize * (1 + concentrateToWaterRatio)) / numberOfServings = servingSize :=
sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l1274_127402


namespace NUMINAMATH_CALUDE_sequence_difference_equals_170000_l1274_127475

/-- The sum of an arithmetic sequence with first term a, last term l, and n terms -/
def arithmetic_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The difference between two sums of arithmetic sequences -/
def sequence_difference : ℕ :=
  arithmetic_sum 2001 2100 100 - arithmetic_sum 301 400 100

theorem sequence_difference_equals_170000 : sequence_difference = 170000 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_equals_170000_l1274_127475


namespace NUMINAMATH_CALUDE_composition_of_even_and_odd_is_even_l1274_127455

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Theorem statement
theorem composition_of_even_and_odd_is_even
  (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsEven (f ∘ g) := by sorry

end NUMINAMATH_CALUDE_composition_of_even_and_odd_is_even_l1274_127455


namespace NUMINAMATH_CALUDE_triangle_inequality_l1274_127483

/-- For a triangle with side lengths a, b, and c, area 1/4, and circumradius 1,
    √a + √b + √c < 1/a + 1/b + 1/c -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (harea : a * b * c / 4 = 1/4) (hcircum : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1274_127483


namespace NUMINAMATH_CALUDE_problem_statement_l1274_127461

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem problem_statement (a : ℝ) :
  (p a ↔ a ≤ 1) ∧
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a > 1 ∨ (-2 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1274_127461


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1274_127422

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, (a - 3) * x^2 - 4*x + 1 = 0 → (a - 3) ≠ 0) ↔ a ≠ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1274_127422


namespace NUMINAMATH_CALUDE_saras_remaining_money_l1274_127495

/-- Calculates Sara's remaining money after her first paycheck and expenses --/
theorem saras_remaining_money :
  let week1_hours : ℕ := 40
  let week1_rate : ℚ := 11.5
  let week2_regular_hours : ℕ := 40
  let week2_overtime_hours : ℕ := 10
  let week2_rate : ℚ := 12
  let overtime_multiplier : ℚ := 1.5
  let sales : ℚ := 1000
  let commission_rate : ℚ := 0.05
  let tax_rate : ℚ := 0.15
  let insurance_cost : ℚ := 60
  let misc_fees : ℚ := 20
  let tire_cost : ℚ := 410

  let week1_earnings := week1_hours * week1_rate
  let week2_regular_earnings := week2_regular_hours * week2_rate
  let week2_overtime_earnings := week2_overtime_hours * (week2_rate * overtime_multiplier)
  let total_hourly_earnings := week1_earnings + week2_regular_earnings + week2_overtime_earnings
  let commission := sales * commission_rate
  let total_earnings := total_hourly_earnings + commission
  let taxes := total_earnings * tax_rate
  let total_deductions := taxes + insurance_cost + misc_fees
  let net_earnings := total_earnings - total_deductions
  let remaining_money := net_earnings - tire_cost

  remaining_money = 504.5 :=
by sorry

end NUMINAMATH_CALUDE_saras_remaining_money_l1274_127495


namespace NUMINAMATH_CALUDE_scalene_triangle_perimeter_scalene_triangle_perimeter_proof_l1274_127424

/-- A scalene triangle with sides of lengths 15, 10, and 7 has a perimeter of 32. -/
theorem scalene_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p =>
    a = 15 ∧ b = 10 ∧ c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    p = a + b + c →
    p = 32

/-- Proof of the theorem -/
theorem scalene_triangle_perimeter_proof : scalene_triangle_perimeter 15 10 7 32 := by
  sorry

end NUMINAMATH_CALUDE_scalene_triangle_perimeter_scalene_triangle_perimeter_proof_l1274_127424


namespace NUMINAMATH_CALUDE_variation_problem_l1274_127456

theorem variation_problem (c : ℝ) (R S T : ℝ → ℝ) (t : ℝ) :
  (∀ t, R t = c * (S t)^2 / (T t)^2) →
  R 0 = 2 ∧ S 0 = 1 ∧ T 0 = 2 →
  R t = 50 ∧ T t = 5 →
  S t = 12.5 := by
sorry

end NUMINAMATH_CALUDE_variation_problem_l1274_127456


namespace NUMINAMATH_CALUDE_drama_club_adult_ticket_price_l1274_127487

/-- Calculates the adult ticket price for a drama club performance --/
theorem drama_club_adult_ticket_price 
  (total_tickets : ℕ) 
  (student_price : ℕ) 
  (total_amount : ℕ) 
  (student_count : ℕ) 
  (h1 : total_tickets = 1500)
  (h2 : student_price = 6)
  (h3 : total_amount = 16200)
  (h4 : student_count = 300) :
  ∃ (adult_price : ℕ), 
    (total_tickets - student_count) * adult_price + student_count * student_price = total_amount ∧ 
    adult_price = 12 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_adult_ticket_price_l1274_127487


namespace NUMINAMATH_CALUDE_constant_seq_arithmetic_and_geometric_l1274_127478

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A constant sequence with value a -/
def constantSeq (a : ℝ) : Sequence := λ _ => a

/-- An arithmetic sequence -/
def isArithmetic (s : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- A geometric sequence (allowing zero terms) -/
def isGeometric (s : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem constant_seq_arithmetic_and_geometric (a : ℝ) :
  isArithmetic (constantSeq a) ∧ isGeometric (constantSeq a) := by
  sorry

#check constant_seq_arithmetic_and_geometric

end NUMINAMATH_CALUDE_constant_seq_arithmetic_and_geometric_l1274_127478


namespace NUMINAMATH_CALUDE_three_planes_division_l1274_127403

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internal structure of a plane for this problem

/-- The number of regions that a set of planes divides 3D space into -/
def num_regions (planes : List Plane3D) : ℕ := sorry

theorem three_planes_division :
  ∀ (p1 p2 p3 : Plane3D),
  ∃ (min max : ℕ),
    (∀ (n : ℕ), n = num_regions [p1, p2, p3] → min ≤ n ∧ n ≤ max) ∧
    min = 4 ∧ max = 8 := by sorry

end NUMINAMATH_CALUDE_three_planes_division_l1274_127403


namespace NUMINAMATH_CALUDE_probability_of_humanities_course_l1274_127429

/-- Represents a course --/
inductive Course
| Mathematics
| Chinese
| Politics
| Geography
| English
| History
| PhysicalEducation

/-- Represents the time of day --/
inductive TimeOfDay
| Morning
| Afternoon

/-- Defines whether a course is in humanities and social sciences --/
def isHumanities (c : Course) : Bool :=
  match c with
  | Course.Politics | Course.History | Course.Geography => true
  | _ => false

/-- Defines the courses available in each time slot --/
def availableCourses (t : TimeOfDay) : List Course :=
  match t with
  | TimeOfDay.Morning => [Course.Mathematics, Course.Chinese, Course.Politics, Course.Geography]
  | TimeOfDay.Afternoon => [Course.English, Course.History, Course.PhysicalEducation]

theorem probability_of_humanities_course :
  let totalChoices := (availableCourses TimeOfDay.Morning).length * (availableCourses TimeOfDay.Afternoon).length
  let humanitiesChoices := totalChoices - ((availableCourses TimeOfDay.Morning).filter (fun c => !isHumanities c)).length *
                                          ((availableCourses TimeOfDay.Afternoon).filter (fun c => !isHumanities c)).length
  (humanitiesChoices : ℚ) / totalChoices = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_humanities_course_l1274_127429


namespace NUMINAMATH_CALUDE_max_intersection_faces_l1274_127477

def W : Set (Fin 4 → ℝ) := {x | ∀ i, 0 ≤ x i ∧ x i ≤ 1}

def isParallelHyperplane (h : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ x₁ x₂ x₃ x₄, h x₁ x₂ x₃ x₄ ↔ x₁ + x₂ + x₃ + x₄ = k

def intersectionFaces (h : ℝ → ℝ → ℝ → ℝ → Prop) : ℕ :=
  sorry

theorem max_intersection_faces :
  ∀ h, isParallelHyperplane h →
    (∃ x ∈ W, h (x 0) (x 1) (x 2) (x 3)) →
    intersectionFaces h ≤ 8 ∧
    (∃ h', isParallelHyperplane h' ∧
      (∃ x ∈ W, h' (x 0) (x 1) (x 2) (x 3)) ∧
      intersectionFaces h' = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_faces_l1274_127477
