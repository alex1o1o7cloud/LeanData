import Mathlib

namespace NUMINAMATH_CALUDE_molecular_weight_of_one_mole_l220_22092

/-- The molecular weight of aluminum sulfide for a given number of moles -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 4

/-- The molecular weight for the given number of moles -/
def given_weight : ℝ := 600

/-- Theorem: The molecular weight of one mole of aluminum sulfide is 150 g/mol -/
theorem molecular_weight_of_one_mole : 
  molecular_weight 1 = 150 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_of_one_mole_l220_22092


namespace NUMINAMATH_CALUDE_probability_two_red_chips_l220_22031

-- Define the number of red and white chips
def total_red : Nat := 4
def total_white : Nat := 2

-- Define the number of chips in each urn
def chips_per_urn : Nat := 3

-- Define a function to calculate the number of ways to distribute chips
def distribute_chips (r w : Nat) : Nat :=
  Nat.choose total_red r * Nat.choose total_white w

-- Define the probability of drawing a red chip from an urn
def prob_red (red_in_urn total_in_urn : Nat) : Rat :=
  red_in_urn / total_in_urn

-- Theorem statement
theorem probability_two_red_chips :
  -- Calculate the total number of ways to distribute chips
  let total_distributions : Nat :=
    distribute_chips 1 2 + distribute_chips 2 1 + distribute_chips 3 0
  
  -- Calculate the probability for each case
  let prob_case1 : Rat := (distribute_chips 1 2 : Rat) / total_distributions *
    prob_red 1 chips_per_urn * prob_red 3 chips_per_urn
  let prob_case2 : Rat := (distribute_chips 2 1 : Rat) / total_distributions *
    prob_red 2 chips_per_urn * prob_red 2 chips_per_urn
  let prob_case3 : Rat := (distribute_chips 3 0 : Rat) / total_distributions *
    prob_red 3 chips_per_urn * prob_red 1 chips_per_urn

  -- The total probability is the sum of all cases
  prob_case1 + prob_case2 + prob_case3 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_chips_l220_22031


namespace NUMINAMATH_CALUDE_citric_acid_molecular_weight_l220_22081

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in a Citric acid molecule -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in a Citric acid molecule -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in a Citric acid molecule -/
def oxygen_count : ℕ := 7

/-- The molecular weight of Citric acid in g/mol -/
def citric_acid_weight : ℝ := 192.124

theorem citric_acid_molecular_weight :
  (carbon_count : ℝ) * carbon_weight +
  (hydrogen_count : ℝ) * hydrogen_weight +
  (oxygen_count : ℝ) * oxygen_weight =
  citric_acid_weight := by sorry

end NUMINAMATH_CALUDE_citric_acid_molecular_weight_l220_22081


namespace NUMINAMATH_CALUDE_conditions_satisfied_l220_22050

-- Define the points and lengths
variable (P Q R S : ℝ) -- Representing points as real numbers for simplicity
variable (a b c k : ℝ)

-- State the conditions
axiom distinct_collinear : P < Q ∧ Q < R ∧ R < S
axiom positive_lengths : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0
axiom length_PQ : Q - P = a
axiom length_PR : R - P = b
axiom length_PS : S - P = c
axiom b_relation : b = a + k

-- Triangle inequality conditions
axiom triangle_inequality1 : a + (b - a) > c - b
axiom triangle_inequality2 : (b - a) + (c - b) > a
axiom triangle_inequality3 : a + (c - b) > b - a

-- Theorem to prove
theorem conditions_satisfied :
  a < c / 2 ∧ b < 2 * a + c / 2 :=
sorry

end NUMINAMATH_CALUDE_conditions_satisfied_l220_22050


namespace NUMINAMATH_CALUDE_car_distance_l220_22036

/-- Proves that given the ratio of Amar's speed to the car's speed and the distance Amar covers,
    we can calculate the distance the car covers in kilometers. -/
theorem car_distance (amar_speed : ℝ) (car_speed : ℝ) (amar_distance : ℝ) :
  amar_speed / car_speed = 15 / 40 →
  amar_distance = 712.5 →
  ∃ (car_distance : ℝ), car_distance = 1.9 ∧ car_distance * 1000 * (amar_speed / car_speed) = amar_distance :=
by
  sorry

end NUMINAMATH_CALUDE_car_distance_l220_22036


namespace NUMINAMATH_CALUDE_root_product_property_l220_22087

theorem root_product_property (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_product_property_l220_22087


namespace NUMINAMATH_CALUDE_fifth_power_inequality_l220_22003

theorem fifth_power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3*b*c + b^3*a*c + c^3*a*b := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_inequality_l220_22003


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l220_22080

theorem nested_fraction_evaluation : 
  1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l220_22080


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l220_22049

/-- A line that is a perpendicular bisector of a line segment -/
structure PerpendicularBisector where
  a : ℝ
  b : ℝ
  c : ℝ
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  is_perpendicular_bisector : True  -- This is a placeholder for the actual condition

/-- The theorem stating that b = 12 for the given perpendicular bisector -/
theorem perpendicular_bisector_b_value :
  ∀ (pb : PerpendicularBisector), 
  pb.a = 1 ∧ pb.b = 1 ∧ pb.p1 = (2, 4) ∧ pb.p2 = (8, 10) → 
  pb.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l220_22049


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l220_22056

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 → c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l220_22056


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l220_22011

theorem percentage_equation_solution : ∃ x : ℝ, 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 ∧ 
  abs (x - 33.52) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l220_22011


namespace NUMINAMATH_CALUDE_total_jelly_beans_l220_22057

/-- The number of vanilla jelly beans -/
def vanilla : ℕ := 120

/-- The number of grape jelly beans -/
def grape : ℕ := 5 * vanilla + 50

/-- The total number of jelly beans -/
def total : ℕ := vanilla + grape

theorem total_jelly_beans : total = 770 := by sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l220_22057


namespace NUMINAMATH_CALUDE_num_factors_1320_eq_32_l220_22052

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 is equal to its prime factorization -/
axiom prime_fact_1320 : 1320 = 2^3 * 3 * 11 * 5

/-- Theorem: The number of distinct, positive factors of 1320 is 32 -/
theorem num_factors_1320_eq_32 : num_factors_1320 = 32 := by sorry

end NUMINAMATH_CALUDE_num_factors_1320_eq_32_l220_22052


namespace NUMINAMATH_CALUDE_lesser_number_l220_22073

theorem lesser_number (x y : ℝ) (sum_eq : x + y = 60) (diff_eq : x - y = 10) : 
  min x y = 25 := by sorry

end NUMINAMATH_CALUDE_lesser_number_l220_22073


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l220_22077

theorem fraction_equals_zero (x : ℝ) : (x + 2) / (x - 3) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l220_22077


namespace NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l220_22086

/-- The circumference of a circle in which a rectangle with dimensions 10 cm by 24 cm
    is inscribed is equal to 26π cm. -/
theorem circle_circumference_with_inscribed_rectangle : 
  let rectangle_width : ℝ := 10
  let rectangle_height : ℝ := 24
  let diagonal : ℝ := (rectangle_width ^ 2 + rectangle_height ^ 2).sqrt
  let circumference : ℝ := π * diagonal
  circumference = 26 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l220_22086


namespace NUMINAMATH_CALUDE_added_number_proof_l220_22037

theorem added_number_proof (n : ℕ) (original_avg new_avg : ℚ) (added_num : ℚ) : 
  n = 15 →
  original_avg = 17 →
  new_avg = 20 →
  (n : ℚ) * original_avg + added_num = (n + 1 : ℚ) * new_avg →
  added_num = 65 := by
  sorry

end NUMINAMATH_CALUDE_added_number_proof_l220_22037


namespace NUMINAMATH_CALUDE_seventh_grade_class_size_l220_22030

theorem seventh_grade_class_size (girls boys : ℕ) : 
  girls * 3 + boys = 24 → 
  boys / 3 = 6 → 
  girls + boys = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_grade_class_size_l220_22030


namespace NUMINAMATH_CALUDE_profit_is_42_l220_22070

/-- Calculates the profit from selling face masks given the following conditions:
  * 12 boxes of face masks were bought
  * Each box costs $9
  * Each box contains 50 masks
  * 6 boxes were repacked and sold for $5 per 25 pieces
  * Remaining 300 masks were sold in baggies of 10 pieces for $3 each
-/
def calculate_profit (
  total_boxes : ℕ
  ) (cost_per_box : ℕ
  ) (masks_per_box : ℕ
  ) (repacked_boxes : ℕ
  ) (price_per_repack : ℕ
  ) (masks_per_repack : ℕ
  ) (remaining_masks : ℕ
  ) (price_per_baggy : ℕ
  ) (masks_per_baggy : ℕ
  ) : ℕ :=
  let total_cost := total_boxes * cost_per_box
  let repacked_masks := repacked_boxes * masks_per_box
  let repacked_revenue := (repacked_masks / masks_per_repack) * price_per_repack
  let baggy_revenue := (remaining_masks / masks_per_baggy) * price_per_baggy
  let total_revenue := repacked_revenue + baggy_revenue
  total_revenue - total_cost

theorem profit_is_42 : 
  calculate_profit 12 9 50 6 5 25 300 3 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_42_l220_22070


namespace NUMINAMATH_CALUDE_not_divisible_by_361_l220_22041

theorem not_divisible_by_361 (k : ℕ) : ¬(361 ∣ k^2 + 11*k - 22) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_361_l220_22041


namespace NUMINAMATH_CALUDE_mary_nickels_l220_22058

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Mary has 12 nickels after receiving 5 from her dad -/
theorem mary_nickels : total_nickels 7 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_l220_22058


namespace NUMINAMATH_CALUDE_peanuts_in_box_l220_22075

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: If a box initially contains 10 peanuts and 8 more peanuts are added,
    the total number of peanuts in the box is 18. -/
theorem peanuts_in_box : total_peanuts 10 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l220_22075


namespace NUMINAMATH_CALUDE_range_of_m_l220_22004

def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem range_of_m : ∀ m : ℝ, (¬(p m ∨ ¬(q m))) ↔ m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l220_22004


namespace NUMINAMATH_CALUDE_chameleon_color_change_l220_22008

/-- The number of chameleons that changed color in the grove --/
def chameleons_changed : ℕ := 80

/-- The total number of chameleons in the grove --/
def total_chameleons : ℕ := 140

/-- The number of blue chameleons after the color change --/
def blue_after : ℕ → ℕ
| n => n

/-- The number of blue chameleons before the color change --/
def blue_before : ℕ → ℕ
| n => 5 * n

/-- The number of red chameleons before the color change --/
def red_before : ℕ → ℕ
| n => total_chameleons - blue_before n

/-- The number of red chameleons after the color change --/
def red_after : ℕ → ℕ
| n => 3 * (red_before n)

theorem chameleon_color_change :
  ∃ n : ℕ, 
    blue_after n + red_after n = total_chameleons ∧ 
    chameleons_changed = blue_before n - blue_after n :=
  sorry

end NUMINAMATH_CALUDE_chameleon_color_change_l220_22008


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l220_22046

theorem least_addition_for_divisibility :
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), (1056 + m) % 28 = 0 → m ≥ n) ∧
  (1056 + n) % 28 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l220_22046


namespace NUMINAMATH_CALUDE_least_upper_bound_quadratic_form_l220_22090

theorem least_upper_bound_quadratic_form (x₁ x₂ x₃ x₄ : ℝ) (h : x₁ ≠ 0 ∨ x₂ ≠ 0 ∨ x₃ ≠ 0 ∨ x₄ ≠ 0) :
  (x₁ * x₂ + 2 * x₂ * x₃ + x₃ * x₄) / (x₁^2 + x₂^2 + x₃^2 + x₄^2) ≤ (Real.sqrt 2 + 1) / 2 ∧
  ∀ ε > 0, ∃ y₁ y₂ y₃ y₄ : ℝ, (y₁ ≠ 0 ∨ y₂ ≠ 0 ∨ y₃ ≠ 0 ∨ y₄ ≠ 0) ∧
    (y₁ * y₂ + 2 * y₂ * y₃ + y₃ * y₄) / (y₁^2 + y₂^2 + y₃^2 + y₄^2) > (Real.sqrt 2 + 1) / 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_least_upper_bound_quadratic_form_l220_22090


namespace NUMINAMATH_CALUDE_sum_of_squares_l220_22093

theorem sum_of_squares (a b c : ℝ) (h : a + 19 = b + 9 ∧ b + 9 = c + 8) :
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l220_22093


namespace NUMINAMATH_CALUDE_triangle_side_length_l220_22002

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  B = 2 * A ∧
  a = 1 ∧
  b = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l220_22002


namespace NUMINAMATH_CALUDE_y_coordinate_difference_l220_22055

/-- Given two points on a line, prove that the difference between their y-coordinates is 9 -/
theorem y_coordinate_difference (m n : ℝ) : 
  (m = (n / 3) - (2 / 5)) → 
  (m + 3 = ((n + 9) / 3) - (2 / 5)) → 
  ((n + 9) - n = 9) := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_difference_l220_22055


namespace NUMINAMATH_CALUDE_bakery_rolls_distribution_l220_22068

theorem bakery_rolls_distribution (n k : ℕ) (h1 : n = 4) (h2 : k = 3) :
  Nat.choose (n + k - 1) (k - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_distribution_l220_22068


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l220_22006

/-- Given a rectangular plot with the following properties:
  - The length is 10 meters more than the breadth
  - The cost of fencing is 26.50 per meter
  - The total cost of fencing is 5300
  Prove that the length of the plot is 55 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) : 
  length = breadth + 10 →
  26.50 * (2 * (length + breadth)) = 5300 →
  length = 55 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l220_22006


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l220_22084

theorem simplify_and_evaluate (a b : ℝ) :
  -(-a^2 + 2*a*b + b^2) + (-a^2 - a*b + b^2) = -3*a*b ∧
  (a*b = 1 → -(-a^2 + 2*a*b + b^2) + (-a^2 - a*b + b^2) = -3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l220_22084


namespace NUMINAMATH_CALUDE_cricket_team_ratio_proof_l220_22082

def cricket_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) : Prop :=
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  2 * left_handed_non_throwers = right_handed_non_throwers

theorem cricket_team_ratio_proof :
  cricket_team_ratio 64 37 55 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_ratio_proof_l220_22082


namespace NUMINAMATH_CALUDE_marble_jar_problem_l220_22005

theorem marble_jar_problem (r g b : ℕ) :
  r + g = 5 →
  r + b = 7 →
  g + b = 9 →
  r + g + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l220_22005


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l220_22072

def rectangle_lengths : List ℕ := [1, 9, 25, 49, 81, 121]
def common_width : ℕ := 3

theorem sum_of_rectangle_areas :
  (rectangle_lengths.map (λ l => l * common_width)).sum = 858 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l220_22072


namespace NUMINAMATH_CALUDE_rational_roots_count_l220_22020

/-- The number of distinct possible rational roots for a polynomial of the form
    8x^4 + a₃x³ + a₂x² + a₁x + 16 = 0, where a₃, a₂, and a₁ are integers. -/
def num_rational_roots (a₃ a₂ a₁ : ℤ) : ℕ :=
  16

/-- Theorem stating that the number of distinct possible rational roots for the given polynomial
    is always 16, regardless of the values of a₃, a₂, and a₁. -/
theorem rational_roots_count (a₃ a₂ a₁ : ℤ) :
  num_rational_roots a₃ a₂ a₁ = 16 := by
  sorry

end NUMINAMATH_CALUDE_rational_roots_count_l220_22020


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_solution_l220_22042

-- Define the set M as the domain of y = 1 / √(1-2x)
def M : Set ℝ := {x : ℝ | x < 1/2}

-- Define the set N as the range of y = x^2 - 4
def N : Set ℝ := {y : ℝ | y ≥ -4}

-- Theorem stating that the intersection of M and N is {x | -4 ≤ x < 1/2}
theorem M_intersect_N_eq_solution : M ∩ N = {x : ℝ | -4 ≤ x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_solution_l220_22042


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l220_22076

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + z) / 2
  mean1 = mean2 → z = 40 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l220_22076


namespace NUMINAMATH_CALUDE_ship_speed_and_distance_l220_22024

theorem ship_speed_and_distance 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_time = 3)
  (h2 : upstream_time = 4)
  (h3 : current_speed = 3) :
  ∃ (still_water_speed : ℝ) (distance : ℝ),
    still_water_speed = 21 ∧
    distance = 72 ∧
    downstream_time * (still_water_speed + current_speed) = distance ∧
    upstream_time * (still_water_speed - current_speed) = distance :=
by sorry

end NUMINAMATH_CALUDE_ship_speed_and_distance_l220_22024


namespace NUMINAMATH_CALUDE_min_value_of_expression_l220_22069

def expression (a b c : ℕ) : ℚ := ((a + b) / c) / 2

theorem min_value_of_expression :
  ∃ (a b c : ℕ), a ∈ ({2, 3, 5} : Set ℕ) ∧ 
                 b ∈ ({2, 3, 5} : Set ℕ) ∧ 
                 c ∈ ({2, 3, 5} : Set ℕ) ∧ 
                 a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                 (∀ (x y z : ℕ), x ∈ ({2, 3, 5} : Set ℕ) → 
                                 y ∈ ({2, 3, 5} : Set ℕ) → 
                                 z ∈ ({2, 3, 5} : Set ℕ) → 
                                 x ≠ y → y ≠ z → x ≠ z →
                                 expression a b c ≤ expression x y z) ∧
                 expression a b c = 1/2 := by
  sorry

#eval expression 2 3 5

end NUMINAMATH_CALUDE_min_value_of_expression_l220_22069


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_sixth_l220_22099

/-- The units digit of 6^6 is 6 -/
theorem units_digit_of_six_to_sixth (n : ℕ) : n = 6^6 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_sixth_l220_22099


namespace NUMINAMATH_CALUDE_cat_mouse_meet_iff_both_odd_cat_mouse_not_meet_when_sum_odd_cat_mouse_not_meet_when_both_even_l220_22040

/-- Represents the movement of a cat and a mouse on a grid -/
def CatMouseMeet (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ Odd m ∧ Odd n

/-- Theorem stating the conditions for the cat and mouse to meet -/
theorem cat_mouse_meet_iff_both_odd (m n : ℕ) :
  CatMouseMeet m n ↔ (m > 1 ∧ n > 1 ∧ Odd m ∧ Odd n) :=
by sorry

/-- Theorem stating the impossibility of meeting when m + n is odd -/
theorem cat_mouse_not_meet_when_sum_odd (m n : ℕ) :
  m > 1 → n > 1 → Odd (m + n) → ¬(CatMouseMeet m n) :=
by sorry

/-- Theorem stating the impossibility of meeting when both m and n are even -/
theorem cat_mouse_not_meet_when_both_even (m n : ℕ) :
  m > 1 → n > 1 → Even m → Even n → ¬(CatMouseMeet m n) :=
by sorry

end NUMINAMATH_CALUDE_cat_mouse_meet_iff_both_odd_cat_mouse_not_meet_when_sum_odd_cat_mouse_not_meet_when_both_even_l220_22040


namespace NUMINAMATH_CALUDE_set_operations_l220_22065

-- Define the sets A and B
def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) ∧
  ((Aᶜ ∩ Bᶜ) = {x : ℝ | x < -1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l220_22065


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l220_22091

def balanced_about_2 (a b : ℝ) : Prop := a + b = 2

theorem problem_1 : balanced_about_2 3 (-1) := by sorry

theorem problem_2 (x : ℝ) : balanced_about_2 (x - 3) (5 - x) := by sorry

def a (x : ℝ) : ℝ := 2 * x^2 - 3 * (x^2 + x) + 4
def b (x : ℝ) : ℝ := 2 * x - (3 * x - (4 * x + x^2) - 2)

theorem problem_3 : ∀ x : ℝ, a x + b x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l220_22091


namespace NUMINAMATH_CALUDE_complex_power_sum_l220_22053

theorem complex_power_sum (i : ℂ) : i^2 = -1 → i^123 + i^223 + i^323 = -3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l220_22053


namespace NUMINAMATH_CALUDE_complementary_angles_equal_l220_22045

/-- Two angles that are complementary to the same angle are equal. -/
theorem complementary_angles_equal (α β γ : Real) (h1 : α + γ = 90) (h2 : β + γ = 90) : α = β := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_equal_l220_22045


namespace NUMINAMATH_CALUDE_class_average_weight_l220_22044

theorem class_average_weight (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ) :
  students_a = 50 →
  students_b = 50 →
  avg_weight_a = 60 →
  avg_weight_b = 80 →
  (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = 70 :=
by sorry

end NUMINAMATH_CALUDE_class_average_weight_l220_22044


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l220_22035

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = Real.log x}
def N : Set ℝ := {x | x > 0}

-- Statement to prove
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l220_22035


namespace NUMINAMATH_CALUDE_line_in_fourth_quadrant_l220_22014

/-- A line passes through the fourth quadrant if it intersects both the negative x-axis and the positive y-axis. -/
def passes_through_fourth_quadrant (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (a - 2) * x + a * y + 2 * a - 3 = 0

/-- The theorem stating the condition for the line to pass through the fourth quadrant. -/
theorem line_in_fourth_quadrant (a : ℝ) :
  passes_through_fourth_quadrant a ↔ a ∈ Set.Iio 0 ∪ Set.Ioi (3/2) :=
sorry

end NUMINAMATH_CALUDE_line_in_fourth_quadrant_l220_22014


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l220_22038

/-- Calculates the annual rent per square foot for a shop -/
theorem annual_rent_per_square_foot
  (length : ℝ) (width : ℝ) (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 22)
  (h3 : monthly_rent = 2244) :
  (monthly_rent * 12) / (length * width) = 68 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l220_22038


namespace NUMINAMATH_CALUDE_multiples_of_three_l220_22067

theorem multiples_of_three (n : ℕ) : (∃ k, k = 33 ∧ k * 3 = n) ↔ n = 99 := by sorry

end NUMINAMATH_CALUDE_multiples_of_three_l220_22067


namespace NUMINAMATH_CALUDE_divisible_by_11_digit_l220_22029

/-- Checks if a 6-digit number is divisible by 11 -/
def isDivisibleBy11 (a b c d e f : ℕ) : Prop :=
  (a + c + e) - (b + d + f) ≡ 0 [MOD 11] ∨ (b + d + f) - (a + c + e) ≡ 0 [MOD 11]

/-- The theorem stating that 3 is the digit that makes 54321d divisible by 11 -/
theorem divisible_by_11_digit : ∃ (d : ℕ), d < 10 ∧ isDivisibleBy11 5 4 3 2 1 d ∧ ∀ (x : ℕ), x < 10 → isDivisibleBy11 5 4 3 2 1 x → x = d :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_11_digit_l220_22029


namespace NUMINAMATH_CALUDE_factorial_ratio_42_40_l220_22047

theorem factorial_ratio_42_40 : Nat.factorial 42 / Nat.factorial 40 = 1722 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_42_40_l220_22047


namespace NUMINAMATH_CALUDE_expression_value_l220_22025

theorem expression_value : (64 + 27)^2 - (27^2 + 64^2) + 3 * Real.rpow 1728 (1/3) = 3492 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l220_22025


namespace NUMINAMATH_CALUDE_unique_score_170_l220_22022

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct : ℕ
  wrong : ℕ
  score : ℤ

/-- Calculates the score based on the number of correct and wrong answers --/
def calculate_score (ts : TestScore) : ℤ :=
  30 + 4 * ts.correct - ts.wrong

/-- Checks if a TestScore is valid according to the rules --/
def is_valid_score (ts : TestScore) : Prop :=
  ts.correct + ts.wrong ≤ ts.total_questions ∧
  ts.score = calculate_score ts ∧
  ts.score > 90

/-- Theorem stating that 170 is the only score above 90 that uniquely determines the number of correct answers --/
theorem unique_score_170 :
  ∀ (ts : TestScore),
    ts.total_questions = 35 →
    is_valid_score ts →
    (∀ (ts' : TestScore),
      ts'.total_questions = 35 →
      is_valid_score ts' →
      ts'.score = ts.score →
      ts'.correct = ts.correct) →
    ts.score = 170 :=
sorry

end NUMINAMATH_CALUDE_unique_score_170_l220_22022


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l220_22062

theorem complex_imaginary_part (Z : ℂ) (h1 : Z.re = 1) (h2 : Complex.abs Z = 2) :
  Z.im = Real.sqrt 3 ∨ Z.im = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l220_22062


namespace NUMINAMATH_CALUDE_gayle_bicycle_ride_l220_22010

/-- Gayle's bicycle ride problem -/
theorem gayle_bicycle_ride 
  (sunny_speed : ℝ) 
  (rainy_speed : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : sunny_speed = 40)
  (h2 : rainy_speed = 25)
  (h3 : total_distance = 20)
  (h4 : total_time = 50/60) -- Convert 50 minutes to hours
  : ∃ (rainy_time : ℝ), 
    rainy_time = 32/60 ∧ -- Convert 32 minutes to hours
    rainy_time * rainy_speed + (total_time - rainy_time) * sunny_speed = total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_gayle_bicycle_ride_l220_22010


namespace NUMINAMATH_CALUDE_jane_calculation_l220_22023

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - y - z = 7)
  (h2 : x - (y + z) = 19) : 
  x - y = 13 := by sorry

end NUMINAMATH_CALUDE_jane_calculation_l220_22023


namespace NUMINAMATH_CALUDE_jason_coin_difference_l220_22033

/-- Given that Jayden received 300 coins, the total coins given to both boys is 660,
    and Jason received more coins than Jayden, prove that Jason received 60 more coins than Jayden. -/
theorem jason_coin_difference (jayden_coins : ℕ) (total_coins : ℕ) (jason_coins : ℕ)
  (h1 : jayden_coins = 300)
  (h2 : total_coins = 660)
  (h3 : jason_coins + jayden_coins = total_coins)
  (h4 : jason_coins > jayden_coins) :
  jason_coins - jayden_coins = 60 := by
  sorry

end NUMINAMATH_CALUDE_jason_coin_difference_l220_22033


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l220_22078

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l220_22078


namespace NUMINAMATH_CALUDE_cuboids_painted_equals_five_l220_22032

/-- The number of outer faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces painted -/
def total_faces_painted : ℕ := 30

/-- The number of cuboids painted -/
def num_cuboids : ℕ := total_faces_painted / faces_per_cuboid

theorem cuboids_painted_equals_five :
  num_cuboids = 5 :=
by sorry

end NUMINAMATH_CALUDE_cuboids_painted_equals_five_l220_22032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l220_22071

/-- Given an arithmetic sequence {a_n} with common difference d, 
    if a_1 + a_8 + a_15 = 72, then a_5 + 3d = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 8 + a 15 = 72 →           -- given sum condition
  a 5 + 3 * d = 24 := by            -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l220_22071


namespace NUMINAMATH_CALUDE_drummer_drum_sticks_l220_22060

/-- Calculates the total number of drum stick sets used by a drummer over multiple performances. -/
def total_drum_sticks (sticks_per_show : ℕ) (tossed_after_show : ℕ) (num_nights : ℕ) : ℕ :=
  (sticks_per_show + tossed_after_show) * num_nights

/-- Theorem stating that a drummer using 5 sets per show, tossing 6 sets after each show, 
    for 30 nights, uses 330 sets of drum sticks in total. -/
theorem drummer_drum_sticks : total_drum_sticks 5 6 30 = 330 := by
  sorry

end NUMINAMATH_CALUDE_drummer_drum_sticks_l220_22060


namespace NUMINAMATH_CALUDE_min_value_problem_l220_22074

theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l220_22074


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l220_22027

/-- Converts a number from base 11 to base 10 -/
def base11ToBase10 (n : ℕ) : ℤ := sorry

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℤ := sorry

/-- Represents the digit E in base 14 -/
def E : ℕ := 14

theorem base_conversion_subtraction :
  base11ToBase10 373 - base14ToBase10 (4 * 14 * 14 + E * 14 + 5) = -542 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l220_22027


namespace NUMINAMATH_CALUDE_seating_theorem_l220_22095

def seating_arrangements (total_people : ℕ) (rows : ℕ) (people_per_row : ℕ) 
  (specific_front : ℕ) (specific_back : ℕ) : ℕ :=
  let front_arrangements := Nat.descFactorial people_per_row specific_front
  let back_arrangements := Nat.descFactorial people_per_row specific_back
  let remaining_people := total_people - specific_front - specific_back
  let remaining_arrangements := Nat.factorial remaining_people
  front_arrangements * back_arrangements * remaining_arrangements

theorem seating_theorem : 
  seating_arrangements 8 2 4 2 1 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l220_22095


namespace NUMINAMATH_CALUDE_equation_solution_l220_22054

theorem equation_solution (x : ℝ) :
  x ≠ -2 → x ≠ 2 →
  (1 / (x + 2) + (x + 6) / (x^2 - 4) = 1) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l220_22054


namespace NUMINAMATH_CALUDE_greatest_multiple_of_9_l220_22059

def digits : List Nat := [3, 6, 7, 8, 9]

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

def list_to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ l1.toFinset = l2.toFinset

theorem greatest_multiple_of_9 :
  (∀ l : List Nat, l.length = 5 → is_permutation l digits →
    is_multiple_of_9 (list_to_number l) →
    list_to_number l ≤ 98763) ∧
  (list_to_number [9, 8, 7, 6, 3] = 98763) ∧
  (is_multiple_of_9 98763) :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_9_l220_22059


namespace NUMINAMATH_CALUDE_square_circle_area_fraction_l220_22007

theorem square_circle_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let circle_area := π * r^2
  let outside_area := square_area - circle_area
  outside_area / square_area = 1 - π / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_fraction_l220_22007


namespace NUMINAMATH_CALUDE_student_calculation_l220_22018

theorem student_calculation (chosen_number : ℕ) (h : chosen_number = 48) : 
  chosen_number * 5 - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l220_22018


namespace NUMINAMATH_CALUDE_solve_auction_problem_l220_22028

def auction_problem (price_increase : ℕ) (initial_price : ℕ) (final_price : ℕ) (num_bidders : ℕ) : Prop :=
  let total_increase : ℕ := final_price - initial_price
  let total_bids : ℕ := total_increase / price_increase
  let bids_per_person : ℕ := total_bids / num_bidders
  bids_per_person = 5

theorem solve_auction_problem :
  auction_problem 5 15 65 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_auction_problem_l220_22028


namespace NUMINAMATH_CALUDE_sum_equals_power_of_two_l220_22021

theorem sum_equals_power_of_two : 29 + 12 + 23 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_power_of_two_l220_22021


namespace NUMINAMATH_CALUDE_unique_solution_exists_l220_22088

theorem unique_solution_exists : 
  ∃! (a b c : ℕ+), 
    (a.val * b.val + 3 * b.val * c.val = 63) ∧ 
    (a.val * c.val + 3 * b.val * c.val = 39) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l220_22088


namespace NUMINAMATH_CALUDE_solve_burger_problem_l220_22063

/-- Represents the problem of calculating the number of double burgers bought. -/
def BurgerProblem (total_cost : ℚ) (total_burgers : ℕ) (single_cost : ℚ) (double_cost : ℚ) : Prop :=
  ∃ (single_burgers double_burgers : ℕ),
    single_burgers + double_burgers = total_burgers ∧
    single_cost * single_burgers + double_cost * double_burgers = total_cost ∧
    double_burgers = 29

/-- Theorem stating the solution to the burger problem. -/
theorem solve_burger_problem :
  BurgerProblem 64.5 50 1 1.5 :=
sorry

end NUMINAMATH_CALUDE_solve_burger_problem_l220_22063


namespace NUMINAMATH_CALUDE_sum_of_integers_l220_22089

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + e = 9)
  (eq4 : d - e + a = 4)
  (eq5 : e - a + b = 3) : 
  a + b + c + d + e = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l220_22089


namespace NUMINAMATH_CALUDE_negation_equivalence_l220_22034

theorem negation_equivalence (a : ℝ) :
  (¬ ∀ x > 1, 2^x - a > 0) ↔ (∃ x > 1, 2^x - a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l220_22034


namespace NUMINAMATH_CALUDE_six_digits_in_base_b_unique_base_l220_22079

/-- The base in which 500 (base 10) has exactly 6 digits -/
def base_b : ℕ := 3

/-- 500 in base 10 -/
def number : ℕ := 500

theorem six_digits_in_base_b :
  (base_b ^ 5 : ℕ) ≤ number ∧ number < (base_b ^ 6 : ℕ) :=
by sorry

theorem unique_base :
  ∀ b : ℕ, b ≠ base_b → ¬((b ^ 5 : ℕ) ≤ number ∧ number < (b ^ 6 : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_six_digits_in_base_b_unique_base_l220_22079


namespace NUMINAMATH_CALUDE_calorie_calculation_l220_22051

/-- The number of calories in each cookie -/
def cookie_calories : ℕ := 50

/-- The number of cookies Jimmy eats -/
def cookies_eaten : ℕ := 7

/-- The number of crackers Jimmy eats -/
def crackers_eaten : ℕ := 10

/-- The total number of calories Jimmy consumes -/
def total_calories : ℕ := 500

/-- The number of calories in each cracker -/
def cracker_calories : ℕ := 15

theorem calorie_calculation :
  cookie_calories * cookies_eaten + cracker_calories * crackers_eaten = total_calories := by
  sorry

end NUMINAMATH_CALUDE_calorie_calculation_l220_22051


namespace NUMINAMATH_CALUDE_jovana_shells_problem_l220_22013

/-- Given that Jovana has an initial amount of shells and needs a total amount to fill her bucket,
    this function calculates the additional amount of shells needed. -/
def additional_shells_needed (initial_amount total_amount : ℕ) : ℕ :=
  total_amount - initial_amount

/-- Theorem stating that Jovana needs to add 12 more pounds of shells to fill her bucket. -/
theorem jovana_shells_problem :
  let initial_amount : ℕ := 5
  let total_amount : ℕ := 17
  additional_shells_needed initial_amount total_amount = 12 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_problem_l220_22013


namespace NUMINAMATH_CALUDE_odd_numbers_pascal_triangle_l220_22083

/-- 
Given a non-negative integer n, count_ones n returns the number of 1's 
in the binary representation of n.
-/
def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 2) + count_ones (n / 2)

/-- 
Given a non-negative integer n, odd_numbers_in_pascal_row n returns the 
number of odd numbers in the n-th row of Pascal's triangle.
-/
def odd_numbers_in_pascal_row (n : ℕ) : ℕ :=
  2^(count_ones n)

/-- 
Theorem: The number of odd numbers in the n-th row of Pascal's triangle 
is equal to 2^k, where k is the number of 1's in the binary representation of n.
-/
theorem odd_numbers_pascal_triangle (n : ℕ) : 
  odd_numbers_in_pascal_row n = 2^(count_ones n) := by
  sorry


end NUMINAMATH_CALUDE_odd_numbers_pascal_triangle_l220_22083


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l220_22016

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h_total : total_students = 49)
  (h_ratio : boy_ratio = 3 ∧ girl_ratio = 4) : 
  let y := total_students / (boy_ratio + girl_ratio)
  let num_boys := boy_ratio * y
  let num_girls := girl_ratio * y
  num_girls - num_boys = 7 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l220_22016


namespace NUMINAMATH_CALUDE_sum_of_g_and_h_l220_22012

theorem sum_of_g_and_h (a b c d e f g h : ℝ) 
  (avg_abc : (a + b + c) / 3 = 103 / 3)
  (avg_def : (d + e + f) / 3 = 375 / 6)
  (avg_all : (a + b + c + d + e + f + g + h) / 8 = 23 / 2) :
  g + h = -198.5 := by sorry

end NUMINAMATH_CALUDE_sum_of_g_and_h_l220_22012


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l220_22096

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l220_22096


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l220_22043

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^15 + 11^21) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^15 + 11^21) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l220_22043


namespace NUMINAMATH_CALUDE_min_value_of_squares_l220_22064

theorem min_value_of_squares (x y : ℝ) (h : x^3 + y^3 + 3*x*y = 1) :
  ∃ (m : ℝ), m = 1/2 ∧ (∀ a b : ℝ, a^3 + b^3 + 3*a*b = 1 → a^2 + b^2 ≥ m) ∧ (x^2 + y^2 = m) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l220_22064


namespace NUMINAMATH_CALUDE_circle_inside_parabola_radius_l220_22094

/-- A circle inside a parabola y = 4x^2, tangent at two points, has radius a^2/4 -/
theorem circle_inside_parabola_radius (a : ℝ) :
  let parabola := fun x : ℝ => 4 * x^2
  let tangent_point1 := (a, parabola a)
  let tangent_point2 := (-a, parabola (-a))
  let circle_center := (0, a^2)
  let radius := a^2 / 4
  (∀ x y, (x - 0)^2 + (y - a^2)^2 = radius^2 → y ≤ parabola x) ∧
  (circle_center.1 - tangent_point1.1)^2 + (circle_center.2 - tangent_point1.2)^2 = radius^2 ∧
  (circle_center.1 - tangent_point2.1)^2 + (circle_center.2 - tangent_point2.2)^2 = radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_inside_parabola_radius_l220_22094


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_6889_l220_22009

theorem largest_prime_factor_of_6889 : ∃ p : ℕ, p.Prime ∧ p ∣ 6889 ∧ ∀ q : ℕ, q.Prime → q ∣ 6889 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_6889_l220_22009


namespace NUMINAMATH_CALUDE_no_always_positive_f_solution_sets_f_negative_l220_22098

def f (a x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

theorem no_always_positive_f :
  ¬∃ a : ℝ, ∀ x : ℝ, f a x > 0 := by sorry

theorem solution_sets_f_negative (a : ℝ) :
  (a = 0 → {x : ℝ | f a x < 0} = {x : ℝ | x > 1}) ∧
  (a < 0 → {x : ℝ | f a x < 0} = {x : ℝ | x < 1/a ∨ x > 1}) ∧
  (a = 1 → {x : ℝ | f a x < 0} = ∅) ∧
  (a > 1 → {x : ℝ | f a x < 0} = {x : ℝ | 1/a < x ∧ x < 1}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | f a x < 0} = {x : ℝ | 1 < x ∧ x < 1/a}) := by sorry

end NUMINAMATH_CALUDE_no_always_positive_f_solution_sets_f_negative_l220_22098


namespace NUMINAMATH_CALUDE_ant_meeting_point_l220_22017

/-- Triangle with given side lengths --/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Point on the perimeter of the triangle --/
structure PerimeterPoint where
  side : Fin 3
  distance : ℝ

/-- Represents the meeting point of two ants --/
def MeetingPoint (t : Triangle) (w : PerimeterPoint) : Prop :=
  w.side = 1 ∧ w.distance ≤ t.yz

/-- The distance YW --/
def YW (t : Triangle) (w : PerimeterPoint) : ℝ :=
  t.yz - w.distance

/-- Main theorem --/
theorem ant_meeting_point (t : Triangle) (w : PerimeterPoint) :
  t.xy = 8 ∧ t.yz = 10 ∧ t.xz = 12 ∧ MeetingPoint t w →
  YW t w = 3 := by
  sorry

end NUMINAMATH_CALUDE_ant_meeting_point_l220_22017


namespace NUMINAMATH_CALUDE_f_inequality_l220_22000

noncomputable def f (x : ℝ) : ℝ := x * Real.log (Real.sqrt (x^2 + 1) + x) + x^2 - x * Real.sin x

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l220_22000


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l220_22001

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l220_22001


namespace NUMINAMATH_CALUDE_cristina_catches_nicky_l220_22048

/-- Proves that Cristina catches up to Nicky in 27 seconds --/
theorem cristina_catches_nicky (cristina_speed nicky_speed : ℝ) (head_start : ℝ) 
  (h1 : cristina_speed > nicky_speed)
  (h2 : cristina_speed = 5)
  (h3 : nicky_speed = 3)
  (h4 : head_start = 54) :
  (head_start / (cristina_speed - nicky_speed) = 27) := by
  sorry

end NUMINAMATH_CALUDE_cristina_catches_nicky_l220_22048


namespace NUMINAMATH_CALUDE_stock_price_change_l220_22039

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_day1 := initial_price * (1 - 0.15)
  let price_after_day2 := price_after_day1 * (1 + 0.25)
  let percent_change := (price_after_day2 - initial_price) / initial_price * 100
  percent_change = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l220_22039


namespace NUMINAMATH_CALUDE_b_97_mod_36_l220_22015

def b (n : ℕ) : ℕ := 5^n + 7^n

theorem b_97_mod_36 : b 97 % 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_b_97_mod_36_l220_22015


namespace NUMINAMATH_CALUDE_odd_prime_square_root_theorem_l220_22061

theorem odd_prime_square_root_theorem (p : ℕ) (k : ℕ) (h_prime : Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ (m : ℕ), m > 0 ∧ m * m = k * k - p * k) : 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_theorem_l220_22061


namespace NUMINAMATH_CALUDE_no_real_roots_l220_22026

theorem no_real_roots : ∀ x : ℝ, x^2 - x * Real.sqrt 5 + Real.sqrt 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l220_22026


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l220_22085

theorem negative_fraction_comparison : -3/4 > -4/5 := by sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l220_22085


namespace NUMINAMATH_CALUDE_golden_retriever_age_l220_22097

-- Define the weight gain per year
def weight_gain_per_year : ℕ := 11

-- Define the current weight
def current_weight : ℕ := 88

-- Define the age of the golden retriever
def age : ℕ := current_weight / weight_gain_per_year

-- Theorem to prove
theorem golden_retriever_age :
  age = 8 :=
by sorry

end NUMINAMATH_CALUDE_golden_retriever_age_l220_22097


namespace NUMINAMATH_CALUDE_truncated_cube_edges_l220_22066

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  /-- The number of vertices in the original cube -/
  originalVertices : Nat
  /-- The number of edges in the original cube -/
  originalEdges : Nat
  /-- The fraction of each edge removed by truncation -/
  truncationFraction : Rat
  /-- The number of edges affected by truncation at each vertex -/
  edgesAffectedPerVertex : Nat
  /-- The number of new edges created by truncation at each vertex -/
  newEdgesPerVertex : Nat

/-- The number of edges in a truncated cube -/
def edgesInTruncatedCube (c : TruncatedCube) : Nat :=
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex

/-- Theorem stating that a cube with truncated corners has 36 edges -/
theorem truncated_cube_edges :
  ∀ (c : TruncatedCube),
    c.originalVertices = 8 ∧
    c.originalEdges = 12 ∧
    c.truncationFraction = 1/4 ∧
    c.edgesAffectedPerVertex = 2 ∧
    c.newEdgesPerVertex = 3 →
    edgesInTruncatedCube c = 36 :=
by sorry

end NUMINAMATH_CALUDE_truncated_cube_edges_l220_22066


namespace NUMINAMATH_CALUDE_sean_net_profit_l220_22019

/-- Represents the pricing tiers for patches --/
inductive PricingTier
  | small
  | medium
  | large
  | xlarge

/-- Calculates the price per patch based on the pricing tier --/
def price_per_patch (tier : PricingTier) : ℚ :=
  match tier with
  | .small => 12
  | .medium => 11.5
  | .large => 11
  | .xlarge => 10.5

/-- Represents a sale of patches --/
structure Sale :=
  (quantity : ℕ)
  (customers : ℕ)
  (tier : PricingTier)

/-- Calculates the total cost for ordering patches --/
def total_cost (patches : ℕ) : ℚ :=
  let units := (patches + 99) / 100  -- Round up to nearest 100
  1.25 * patches + 20 * units

/-- Calculates the revenue from a sale --/
def sale_revenue (sale : Sale) : ℚ :=
  sale.quantity * sale.customers * price_per_patch sale.tier

/-- Calculates the total revenue from all sales --/
def total_revenue (sales : List Sale) : ℚ :=
  sales.map sale_revenue |> List.sum

/-- The main theorem stating Sean's net profit --/
theorem sean_net_profit (sales : List Sale) 
  (h_sales : sales = [
    {quantity := 15, customers := 5, tier := .small},
    {quantity := 50, customers := 2, tier := .medium},
    {quantity := 25, customers := 1, tier := .large}
  ]) : 
  total_revenue sales - total_cost (sales.map (λ s => s.quantity * s.customers) |> List.sum) = 2035 := by
  sorry


end NUMINAMATH_CALUDE_sean_net_profit_l220_22019
