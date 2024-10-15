import Mathlib

namespace NUMINAMATH_CALUDE_roots_cubic_sum_l28_2830

theorem roots_cubic_sum (a b c : ℝ) (r s : ℝ) (h : a ≠ 0) (h1 : c ≠ 0) : 
  (a * r^2 + b * r + c = 0) → 
  (a * s^2 + b * s + c = 0) → 
  (1 / r^3 + 1 / s^3 = (-b^3 + 3*a*b*c) / c^3) :=
by sorry

end NUMINAMATH_CALUDE_roots_cubic_sum_l28_2830


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l28_2894

theorem trigonometric_expression_equality :
  let sin30 := 1 / 2
  let cos30 := Real.sqrt 3 / 2
  let tan60 := Real.sqrt 3
  2 * sin30 + cos30 * tan60 = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l28_2894


namespace NUMINAMATH_CALUDE_two_year_interest_calculation_l28_2867

def compound_interest (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial_amount * (1 + rate1) * (1 + rate2)

theorem two_year_interest_calculation :
  let initial_amount : ℝ := 7500
  let rate1 : ℝ := 0.20
  let rate2 : ℝ := 0.25
  compound_interest initial_amount rate1 rate2 = 11250 := by
  sorry

end NUMINAMATH_CALUDE_two_year_interest_calculation_l28_2867


namespace NUMINAMATH_CALUDE_red_cards_count_l28_2825

theorem red_cards_count (total_cards : ℕ) (total_credits : ℕ) 
  (red_card_cost blue_card_cost : ℕ) :
  total_cards = 20 →
  total_credits = 84 →
  red_card_cost = 3 →
  blue_card_cost = 5 →
  ∃ (red_cards blue_cards : ℕ),
    red_cards + blue_cards = total_cards ∧
    red_cards * red_card_cost + blue_cards * blue_card_cost = total_credits ∧
    red_cards = 8 :=
by sorry

end NUMINAMATH_CALUDE_red_cards_count_l28_2825


namespace NUMINAMATH_CALUDE_blue_sky_project_exhibition_l28_2800

theorem blue_sky_project_exhibition (n : ℕ) (m : ℕ) :
  n = 6 →
  m = 6 →
  (Nat.choose n 2) * (5^(n - 2)) = (Nat.choose 6 2) * 5^4 :=
by sorry

end NUMINAMATH_CALUDE_blue_sky_project_exhibition_l28_2800


namespace NUMINAMATH_CALUDE_kelly_bought_five_more_paper_l28_2806

/-- Calculates the number of additional pieces of construction paper Kelly bought --/
def additional_construction_paper (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (final_supplies : ℕ) : ℕ :=
  let initial_supplies := students * paper_per_student + glue_bottles
  let remaining_supplies := initial_supplies / 2
  final_supplies - remaining_supplies

/-- Proves that Kelly bought 5 additional pieces of construction paper --/
theorem kelly_bought_five_more_paper : 
  additional_construction_paper 8 3 6 20 = 5 := by
  sorry

#eval additional_construction_paper 8 3 6 20

end NUMINAMATH_CALUDE_kelly_bought_five_more_paper_l28_2806


namespace NUMINAMATH_CALUDE_intersection_M_N_l28_2857

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x : ℕ | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l28_2857


namespace NUMINAMATH_CALUDE_alice_survey_l28_2840

theorem alice_survey (total_students : ℕ) 
  (malfunction_believers : ℕ) 
  (password_believers : ℕ) :
  (malfunction_believers : ℚ) / (total_students : ℚ) = 723/1000 →
  (password_believers : ℚ) / (malfunction_believers : ℚ) = 346/1000 →
  password_believers = 18 →
  total_students = 72 := by
sorry

end NUMINAMATH_CALUDE_alice_survey_l28_2840


namespace NUMINAMATH_CALUDE_product_of_fractions_l28_2811

theorem product_of_fractions : 
  let f (n : ℕ) := (n^4 - 1) / (n^4 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 880 / 91 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l28_2811


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l28_2875

open Set

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {2, 3}

-- Theorem to prove
theorem complement_intersection_problem :
  (U \ M) ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l28_2875


namespace NUMINAMATH_CALUDE_cubic_root_sum_l28_2889

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 7*p^2 + 11*p = 14 →
  q^3 - 7*q^2 + 11*q = 14 →
  r^3 - 7*r^2 + 11*r = 14 →
  p + q + r = 7 →
  p*q + q*r + r*p = 11 →
  p*q*r = 14 →
  p*q/r + q*r/p + r*p/q = -75/14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l28_2889


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l28_2805

theorem triangle_angle_measure (P Q R : ℝ) (h1 : P = 2 * Q) (h2 : R = 5 * Q) (h3 : P + Q + R = 180) : P = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l28_2805


namespace NUMINAMATH_CALUDE_solution_set_implies_ratio_l28_2878

theorem solution_set_implies_ratio (a b : ℝ) (h : ∀ x, (2*a - b)*x + (a + b) > 0 ↔ x > -3) :
  b / a = 5 / 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_ratio_l28_2878


namespace NUMINAMATH_CALUDE_tickets_to_buy_l28_2829

def ferris_wheel_cost : ℕ := 6
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 7
def antonieta_tickets : ℕ := 2

theorem tickets_to_buy : 
  ferris_wheel_cost + roller_coaster_cost + log_ride_cost - antonieta_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_tickets_to_buy_l28_2829


namespace NUMINAMATH_CALUDE_quarter_value_percentage_l28_2884

theorem quarter_value_percentage (num_dimes : ℕ) (num_quarters : ℕ) 
  (h1 : num_dimes = 75) (h2 : num_quarters = 30) : 
  (num_quarters * 25) / (num_dimes * 10 + num_quarters * 25) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_value_percentage_l28_2884


namespace NUMINAMATH_CALUDE_circle_center_sum_l28_2845

theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 10*x - 4*y + 18 → (x - 5)^2 + (y + 2)^2 = 25 ∧ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l28_2845


namespace NUMINAMATH_CALUDE_completing_square_step_l28_2872

theorem completing_square_step (x : ℝ) : x^2 - 4*x + 3 = 0 → x^2 - 4*x + (-2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_step_l28_2872


namespace NUMINAMATH_CALUDE_expand_product_l28_2896

theorem expand_product (x : ℝ) : (5*x + 3) * (3*x^2 + 4) = 15*x^3 + 9*x^2 + 20*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l28_2896


namespace NUMINAMATH_CALUDE_beads_per_necklace_l28_2812

/-- Given that Emily made 6 necklaces and used 18 beads in total,
    prove that each necklace contains 3 beads. -/
theorem beads_per_necklace (total_necklaces : ℕ) (total_beads : ℕ)
    (h1 : total_necklaces = 6)
    (h2 : total_beads = 18) :
    total_beads / total_necklaces = 3 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l28_2812


namespace NUMINAMATH_CALUDE_square_area_with_circles_l28_2814

theorem square_area_with_circles (r : ℝ) (h : r = 7) : 
  (4 * r) ^ 2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l28_2814


namespace NUMINAMATH_CALUDE_parabolas_intersection_l28_2863

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 15
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 12

-- Define the intersection points
def point1 : ℝ × ℝ := (-3, 57)
def point2 : ℝ × ℝ := (12, 237)

theorem parabolas_intersection :
  (∀ x y : ℝ, parabola1 x = parabola2 x ∧ parabola1 x = y → (x, y) = point1 ∨ (x, y) = point2) ∧
  parabola1 (point1.1) = point1.2 ∧
  parabola2 (point1.1) = point1.2 ∧
  parabola1 (point2.1) = point2.2 ∧
  parabola2 (point2.1) = point2.2 ∧
  point1.1 < point2.1 :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l28_2863


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l28_2856

theorem two_digit_number_problem :
  ∀ (x y : ℕ),
    x ≤ 9 ∧ y ≤ 9 ∧  -- Ensuring x and y are single digits
    x + y = 8 ∧  -- Sum of digits is 8
    (10 * x + y) * (10 * y + x) = 1855 →  -- Product condition
    (10 * x + y = 35) ∨ (10 * x + y = 53) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l28_2856


namespace NUMINAMATH_CALUDE_handshake_count_l28_2813

def networking_event (total_people group_a_size group_b_size : ℕ)
  (group_a_fully_acquainted group_a_partially_acquainted : ℕ)
  (group_a_partial_connections : ℕ) : Prop :=
  total_people = group_a_size + group_b_size ∧
  group_a_size = group_a_fully_acquainted + 1 ∧
  group_a_partially_acquainted = 1 ∧
  group_a_partial_connections = 5

theorem handshake_count
  (h : networking_event 40 25 15 24 1 5) :
  (25 * 15) +  -- handshakes between Group A and Group B
  (15 * 14 / 2) +  -- handshakes within Group B
  19  -- handshakes of partially acquainted member in Group A
  = 499 :=
sorry

end NUMINAMATH_CALUDE_handshake_count_l28_2813


namespace NUMINAMATH_CALUDE_range_of_a_l28_2880

def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}

def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) :
  (M a ∪ N = N) ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l28_2880


namespace NUMINAMATH_CALUDE_solution_conditions_l28_2887

/-- Represents the system of equations and conditions for the problem -/
structure EquationSystem where
  x : ℝ
  y : ℝ
  z : ℝ
  a : ℝ
  b : ℝ
  eq1 : 52 * x - 34 * y - z + 9 * a = 0
  eq2 : 49 * x - 31 * y - 3 * z - b = 0
  eq3 : 36 * x - 24 * y + z + 3 * a + 2 * b = 0
  a_pos : a > 0
  b_pos : b > 0

/-- The main theorem stating the conditions for positivity and minimal x -/
theorem solution_conditions (sys : EquationSystem) :
  (sys.a > (2/9) * sys.b → sys.x > 0 ∧ sys.y > 0 ∧ sys.z > 0) ∧
  (sys.x = 9 ∧ sys.y = 14 ∧ sys.z = 1 ∧ sys.a = 1 ∧ sys.b = 4 →
   ∀ (a' b' : ℝ), a' > 0 → b' > 0 → sys.x ≤ 17 * a' - 2 * b') :=
by sorry

end NUMINAMATH_CALUDE_solution_conditions_l28_2887


namespace NUMINAMATH_CALUDE_savings_calculation_l28_2882

def calculate_savings (initial_amount : ℚ) (tax_rate : ℚ) (bike_spending_rate : ℚ) : ℚ :=
  let after_tax := initial_amount * (1 - tax_rate)
  let bike_cost := after_tax * bike_spending_rate
  after_tax - bike_cost

theorem savings_calculation :
  calculate_savings 125 0.2 0.8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l28_2882


namespace NUMINAMATH_CALUDE_sum_mod_five_equals_four_l28_2874

/-- Given positive integers a, b, c less than 5 satisfying certain congruences,
    prove that their sum modulo 5 is 4. -/
theorem sum_mod_five_equals_four
  (a b c : ℕ)
  (ha : 0 < a ∧ a < 5)
  (hb : 0 < b ∧ b < 5)
  (hc : 0 < c ∧ c < 5)
  (h1 : a * b * c % 5 = 1)
  (h2 : 3 * c % 5 = 2)
  (h3 : 4 * b % 5 = (3 + b) % 5) :
  (a + b + c) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_five_equals_four_l28_2874


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l28_2807

theorem angle_in_second_quadrant : 
  let θ := (29 * Real.pi) / 6
  0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l28_2807


namespace NUMINAMATH_CALUDE_floss_leftover_is_five_l28_2843

/-- The amount of floss left over after distributing to students -/
def floss_leftover (num_students : ℕ) (floss_per_student : ℚ) (floss_per_packet : ℕ) : ℚ :=
  let total_floss_needed : ℚ := num_students * floss_per_student
  let packets_needed : ℕ := (total_floss_needed / floss_per_packet).ceil.toNat
  (packets_needed * floss_per_packet : ℚ) - total_floss_needed

/-- Theorem stating the amount of floss left over in the given scenario -/
theorem floss_leftover_is_five :
  floss_leftover 20 (3/2) 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_floss_leftover_is_five_l28_2843


namespace NUMINAMATH_CALUDE_help_user_hours_l28_2892

theorem help_user_hours (total_hours : Real) (software_hours : Real) (other_services_percentage : Real) :
  total_hours = 68.33333333333333 →
  software_hours = 24 →
  other_services_percentage = 0.40 →
  ∃ help_user_hours : Real,
    help_user_hours = total_hours - software_hours - (other_services_percentage * total_hours) ∧
    help_user_hours = 17 := by
  sorry

end NUMINAMATH_CALUDE_help_user_hours_l28_2892


namespace NUMINAMATH_CALUDE_last_two_digits_13_pow_101_base_3_l28_2832

theorem last_two_digits_13_pow_101_base_3 : ∃ n : ℕ, 13^101 ≡ 21 [MOD 9] ∧ n * 9 + 21 = 13^101 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_13_pow_101_base_3_l28_2832


namespace NUMINAMATH_CALUDE_red_bowl_possible_values_l28_2823

/-- Represents the distribution of balls in the three bowls -/
structure BallDistribution where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Checks if a ball distribution is valid according to the problem conditions -/
def is_valid_distribution (d : BallDistribution) : Prop :=
  d.red + d.blue + d.yellow = 27 ∧
  ∃ (red_sum blue_sum yellow_sum : ℕ),
    red_sum + blue_sum + yellow_sum = (27 * 28) / 2 ∧
    d.red > 0 → red_sum / d.red = 15 ∧
    d.blue > 0 → blue_sum / d.blue = 3 ∧
    d.yellow > 0 → yellow_sum / d.yellow = 18

/-- The theorem stating the possible values for the number of balls in the red bowl -/
theorem red_bowl_possible_values :
  ∀ d : BallDistribution, is_valid_distribution d → d.red ∈ ({11, 16, 21} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_red_bowl_possible_values_l28_2823


namespace NUMINAMATH_CALUDE_f_of_five_equals_102_l28_2808

/-- Given a function f(x) = 2x^2 + y where f(2) = 60, prove that f(5) = 102 -/
theorem f_of_five_equals_102 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 60) :
  f 5 = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_102_l28_2808


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l28_2868

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 5 →
    x + 5*x > 20 →
    5*x + 20 > x →
    x + 20 > 5*x →
    (∀ y : ℕ,
      y > 0 →
      y < 5 →
      y + 5*y > 20 →
      5*y + 20 > y →
      y + 20 > 5*y →
      x + 5*x + 20 ≥ y + 5*y + 20) →
    x + 5*x + 20 = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l28_2868


namespace NUMINAMATH_CALUDE_prob_two_red_scheme1_correct_scheme2_more_advantageous_l28_2888

-- Define the number of red and yellow balls
def red_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := red_balls + yellow_balls

-- Define the reward amounts
def reward (red_count : ℕ) : ℝ :=
  match red_count with
  | 0 => 5
  | 1 => 10
  | 2 => 20
  | _ => 0

-- Define the probability of drawing two red balls in Scheme 1
def prob_two_red_scheme1 : ℚ := 1 / 10

-- Define the average earnings for each scheme
def avg_earnings_scheme1 : ℝ := 8.5
def avg_earnings_scheme2 : ℝ := 9.2

-- Theorem 1: Probability of drawing two red balls in Scheme 1
theorem prob_two_red_scheme1_correct :
  prob_two_red_scheme1 = 1 / 10 := by sorry

-- Theorem 2: Scheme 2 is more advantageous
theorem scheme2_more_advantageous :
  avg_earnings_scheme2 > avg_earnings_scheme1 := by sorry

end NUMINAMATH_CALUDE_prob_two_red_scheme1_correct_scheme2_more_advantageous_l28_2888


namespace NUMINAMATH_CALUDE_scribbled_digits_sum_l28_2833

theorem scribbled_digits_sum (x y : ℕ) (a b c : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 →
  ∃ k : ℕ, x * y = k * 100000 + a * 10000 + 3 * 1000 + b * 100 + 1 * 10 + 2 →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 6 := by
sorry

end NUMINAMATH_CALUDE_scribbled_digits_sum_l28_2833


namespace NUMINAMATH_CALUDE_initial_distance_is_15_l28_2861

/-- The initial distance between John and Steve in a speed walking race. -/
def initial_distance (john_speed steve_speed : ℝ) (duration : ℝ) (final_difference : ℝ) : ℝ :=
  john_speed * duration - steve_speed * duration - final_difference

/-- Theorem stating that the initial distance between John and Steve is 15 meters. -/
theorem initial_distance_is_15 :
  initial_distance 4.2 3.7 34 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_is_15_l28_2861


namespace NUMINAMATH_CALUDE_pqr_plus_xyz_eq_zero_l28_2810

theorem pqr_plus_xyz_eq_zero 
  (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := by sorry

end NUMINAMATH_CALUDE_pqr_plus_xyz_eq_zero_l28_2810


namespace NUMINAMATH_CALUDE_p_equiv_not_q_l28_2898

theorem p_equiv_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬(P ∧ Q)) : 
  P ↔ ¬Q := by
  sorry

end NUMINAMATH_CALUDE_p_equiv_not_q_l28_2898


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_is_96_l28_2822

/-- The surface area of a cube with small cubes removed from its corners -/
def modified_cube_surface_area (cube_side : ℝ) (removed_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_side^2
  let removed_area := 8 * 3 * removed_side^2
  let new_exposed_area := 8 * 3 * removed_side^2
  original_surface_area - removed_area + new_exposed_area

/-- Theorem: The surface area of a 4 cm cube with 1 cm cubes removed from each corner is 96 sq.cm -/
theorem modified_cube_surface_area_is_96 :
  modified_cube_surface_area 4 1 = 96 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_is_96_l28_2822


namespace NUMINAMATH_CALUDE_tims_change_l28_2876

/-- Represents the bread purchase scenario -/
structure BreadPurchase where
  loaves : ℕ
  slices_per_loaf : ℕ
  cost_per_slice : ℚ
  payment : ℚ

/-- Calculates the change received in a bread purchase -/
def calculate_change (purchase : BreadPurchase) : ℚ :=
  purchase.payment - (purchase.loaves * purchase.slices_per_loaf * purchase.cost_per_slice)

/-- Theorem: Tim's change is $16.00 -/
theorem tims_change (purchase : BreadPurchase) 
  (h1 : purchase.loaves = 3)
  (h2 : purchase.slices_per_loaf = 20)
  (h3 : purchase.cost_per_slice = 40/100)
  (h4 : purchase.payment = 40) :
  calculate_change purchase = 16 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_l28_2876


namespace NUMINAMATH_CALUDE_three_conical_planet_models_l28_2804

/-- Represents a model of a conical planet --/
structure ConicalPlanetModel where
  /-- The type of coordinate lines in the model --/
  CoordinateLine : Type
  /-- Predicate for whether two coordinate lines intersect --/
  intersects : CoordinateLine → CoordinateLine → Prop
  /-- Predicate for whether a coordinate line self-intersects --/
  self_intersects : CoordinateLine → Prop
  /-- Predicate for whether the constant direction principle holds --/
  constant_direction : Prop

/-- Cylindrical projection model --/
def cylindrical_model : ConicalPlanetModel := sorry

/-- Traditional conical projection model --/
def conical_model : ConicalPlanetModel := sorry

/-- Hybrid model --/
def hybrid_model : ConicalPlanetModel := sorry

/-- Properties of the hybrid model --/
axiom hybrid_model_properties :
  ∀ (l1 l2 : hybrid_model.CoordinateLine),
    l1 ≠ l2 → (hybrid_model.intersects l1 l2 ∧ hybrid_model.intersects l2 l1) ∧
    hybrid_model.self_intersects l1 ∧
    hybrid_model.constant_direction

/-- Theorem stating the existence of three distinct conical planet models --/
theorem three_conical_planet_models :
  ∃ (m1 m2 m3 : ConicalPlanetModel),
    m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
    (m1 = cylindrical_model ∨ m1 = conical_model ∨ m1 = hybrid_model) ∧
    (m2 = cylindrical_model ∨ m2 = conical_model ∨ m2 = hybrid_model) ∧
    (m3 = cylindrical_model ∨ m3 = conical_model ∨ m3 = hybrid_model) := by
  sorry

end NUMINAMATH_CALUDE_three_conical_planet_models_l28_2804


namespace NUMINAMATH_CALUDE_net_population_increase_is_154800_l28_2891

/-- Represents the number of seconds in an hour -/
def secondsPerHour : ℕ := 3600

/-- Represents the number of hours in a day -/
def hoursPerDay : ℕ := 24

/-- Represents the number of peak hours in a day -/
def peakHours : ℕ := 12

/-- Represents the number of off-peak hours in a day -/
def offPeakHours : ℕ := 12

/-- Represents the birth rate during peak hours (people per 2 seconds) -/
def peakBirthRate : ℕ := 7

/-- Represents the birth rate during off-peak hours (people per 2 seconds) -/
def offPeakBirthRate : ℕ := 3

/-- Represents the death rate during peak hours (people per 2 seconds) -/
def peakDeathRate : ℕ := 1

/-- Represents the death rate during off-peak hours (people per 2 seconds) -/
def offPeakDeathRate : ℕ := 2

/-- Represents the net migration rate during peak hours (people entering per 4 seconds) -/
def peakMigrationRate : ℕ := 1

/-- Represents the net migration rate during off-peak hours (people leaving per 6 seconds) -/
def offPeakMigrationRate : ℕ := 1

/-- Calculates the net population increase over a 24-hour period -/
def netPopulationIncrease : ℕ :=
  let peakIncrease := (peakBirthRate * 30 * secondsPerHour * peakHours) -
                      (peakDeathRate * 30 * secondsPerHour * peakHours) +
                      (peakMigrationRate * 15 * secondsPerHour * peakHours)
  let offPeakIncrease := (offPeakBirthRate * 30 * secondsPerHour * offPeakHours) -
                         (offPeakDeathRate * 30 * secondsPerHour * offPeakHours) -
                         (offPeakMigrationRate * 10 * secondsPerHour * offPeakHours)
  peakIncrease + offPeakIncrease

theorem net_population_increase_is_154800 : netPopulationIncrease = 154800 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_is_154800_l28_2891


namespace NUMINAMATH_CALUDE_exponential_fixed_point_l28_2835

/-- The function f(x) = a^(x-1) + 2 passes through the point (1, 3) for all a > 0 and a ≠ 1 -/
theorem exponential_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_fixed_point_l28_2835


namespace NUMINAMATH_CALUDE_age_when_hired_l28_2862

/-- Rule of 70 provision for retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1990

/-- The year the employee became eligible for retirement -/
def retirement_eligibility_year : ℕ := 2009

/-- Years employed before retirement eligibility -/
def years_employed : ℕ := retirement_eligibility_year - hire_year

/-- Theorem stating the employee's age when hired -/
theorem age_when_hired :
  ∃ (age : ℕ), rule_of_70 (age + years_employed) years_employed ∧ age = 51 := by
  sorry

end NUMINAMATH_CALUDE_age_when_hired_l28_2862


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l28_2819

theorem quadratic_function_bound (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  (max (|f 1|) (max (|f 2|) (|f 3|))) ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l28_2819


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l28_2821

/-- Given a sphere of radius 6 cm and a right circular cylinder with equal height and diameter,
    if their surface areas are equal, then the height and diameter of the cylinder are both 12 cm. -/
theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ) :
  r_sphere = 6 →
  h_cylinder = 2 * r_cylinder →
  4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h_cylinder →
  h_cylinder = 12 ∧ (2 * r_cylinder) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_dimensions_l28_2821


namespace NUMINAMATH_CALUDE_octagon_square_ratio_l28_2893

theorem octagon_square_ratio :
  let octagons_per_row : ℕ := 5
  let octagon_rows : ℕ := 4
  let squares_per_row : ℕ := 4
  let square_rows : ℕ := 3
  let total_octagons : ℕ := octagons_per_row * octagon_rows
  let total_squares : ℕ := squares_per_row * square_rows
  (total_octagons : ℚ) / total_squares = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_octagon_square_ratio_l28_2893


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l28_2899

/-- The line equation passing through a fixed point for all values of k -/
def line_equation (k x y : ℝ) : Prop :=
  (2*k - 1) * x - (k - 2) * y - (k + 4) = 0

/-- The theorem stating that the line passes through (2, 3) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k 2 3 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l28_2899


namespace NUMINAMATH_CALUDE_equation_solution_l28_2873

theorem equation_solution :
  ∃ x : ℚ, (x - 1) / 2 - (2 - 3*x) / 3 = 1 ∧ x = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l28_2873


namespace NUMINAMATH_CALUDE_probability_order_l28_2854

-- Define the structure of a deck of cards
structure Card where
  suit : Fin 4
  rank : Fin 13

-- Define the deck
def standardDeck : Finset Card := sorry

-- Define the subsets of cards for each event
def fiveOfHearts : Finset Card := sorry
def jokers : Finset Card := sorry
def fives : Finset Card := sorry
def clubs : Finset Card := sorry
def redCards : Finset Card := sorry

-- Define the probability of drawing a card from a given set
def probability (subset : Finset Card) : ℚ :=
  (subset.card : ℚ) / (standardDeck.card : ℚ)

-- Theorem statement
theorem probability_order :
  probability fiveOfHearts < probability jokers ∧
  probability jokers < probability fives ∧
  probability fives < probability clubs ∧
  probability clubs < probability redCards :=
sorry

end NUMINAMATH_CALUDE_probability_order_l28_2854


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l28_2846

/-- Given a principal P where the simple interest for 2 years at 5% per annum is Rs. 55,
    prove that the compound interest on P at 5% per annum for 2 years is Rs. 56.375. -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 5 * 2) / 100 = 55 →
  P * ((1 + 5/100)^2 - 1) = 56.375 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l28_2846


namespace NUMINAMATH_CALUDE_felix_weight_ratio_l28_2801

/-- The weight ratio of Felix's brother to Felix -/
theorem felix_weight_ratio :
  let felix_lift_ratio : ℝ := 1.5
  let brother_lift_ratio : ℝ := 3
  let felix_lift_weight : ℝ := 150
  let brother_lift_weight : ℝ := 600
  let felix_weight := felix_lift_weight / felix_lift_ratio
  let brother_weight := brother_lift_weight / brother_lift_ratio
  brother_weight / felix_weight = 2 := by
sorry


end NUMINAMATH_CALUDE_felix_weight_ratio_l28_2801


namespace NUMINAMATH_CALUDE_min_value_expression_l28_2881

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ 2) (h3 : 2 ≤ b) (h4 : b ≤ c) (h5 : c ≤ 6) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (6/c - 1)^2 ≥ 16 - 8 * Real.sqrt 3 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 1 ≤ a₀ ∧ a₀ ≤ 2 ∧ 2 ≤ b₀ ∧ b₀ ≤ c₀ ∧ c₀ ≤ 6 ∧
    (a₀ - 1)^2 + (b₀/a₀ - 1)^2 + (c₀/b₀ - 1)^2 + (6/c₀ - 1)^2 = 16 - 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l28_2881


namespace NUMINAMATH_CALUDE_problem_solution_l28_2869

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

theorem problem_solution (a b : ℝ) : 
  (∃ C : Set ℝ, C = {x | a + 1 ≤ x ∧ x ≤ 2*a - 2} ∧ 
   (Aᶜ ∩ C = {x | 6 ≤ x ∧ x ≤ b})) → a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l28_2869


namespace NUMINAMATH_CALUDE_expensive_fluid_price_is_30_l28_2826

/-- Represents the cost of cleaning fluids and drums --/
structure CleaningSupplies where
  total_drums : ℕ
  expensive_drums : ℕ
  cheap_price : ℕ
  total_cost : ℕ

/-- Calculates the price of the more expensive fluid per drum --/
def expensive_fluid_price (supplies : CleaningSupplies) : ℕ :=
  (supplies.total_cost - (supplies.total_drums - supplies.expensive_drums) * supplies.cheap_price) / supplies.expensive_drums

/-- Theorem stating that the price of the more expensive fluid is $30 per drum --/
theorem expensive_fluid_price_is_30 (supplies : CleaningSupplies) 
    (h1 : supplies.total_drums = 7)
    (h2 : supplies.expensive_drums = 2)
    (h3 : supplies.cheap_price = 20)
    (h4 : supplies.total_cost = 160) :
  expensive_fluid_price supplies = 30 := by
  sorry

#eval expensive_fluid_price { total_drums := 7, expensive_drums := 2, cheap_price := 20, total_cost := 160 }

end NUMINAMATH_CALUDE_expensive_fluid_price_is_30_l28_2826


namespace NUMINAMATH_CALUDE_sine_law_application_l28_2839

/-- Given a triangle ABC with sides a and b opposite to angles A and B respectively,
    if a = 2√2, b = 3, and sin A = √2/6, then sin B = 1/4 -/
theorem sine_law_application (a b : ℝ) (A B : ℝ) :
  a = 2 * Real.sqrt 2 →
  b = 3 →
  Real.sin A = Real.sqrt 2 / 6 →
  Real.sin B = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_sine_law_application_l28_2839


namespace NUMINAMATH_CALUDE_sarah_fish_difference_l28_2866

/-- The number of fish each person has -/
structure FishCounts where
  billy : ℕ
  tony : ℕ
  sarah : ℕ
  bobby : ℕ

/-- The conditions of the problem -/
def fish_problem (fc : FishCounts) : Prop :=
  fc.billy = 10 ∧
  fc.tony = 3 * fc.billy ∧
  fc.bobby = 2 * fc.sarah ∧
  fc.sarah > fc.tony ∧
  fc.billy + fc.tony + fc.sarah + fc.bobby = 145

theorem sarah_fish_difference (fc : FishCounts) :
  fish_problem fc → fc.sarah - fc.tony = 5 := by
  sorry

end NUMINAMATH_CALUDE_sarah_fish_difference_l28_2866


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l28_2841

/-- 
Given a rectangular prism with width w, length 2w, and height w/2,
where the sum of all edge lengths is 88 cm,
prove that the volume of the prism is 85184/343 cm³.
-/
theorem rectangular_prism_volume 
  (w : ℝ) 
  (h_edge_sum : 4 * w + 8 * w + 2 * w = 88) :
  (2 * w) * w * (w / 2) = 85184 / 343 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l28_2841


namespace NUMINAMATH_CALUDE_speed_ratio_is_correct_l28_2860

/-- The ratio of 5G to 4G peak download speeds -/
def speed_ratio : ℝ := 7

/-- The size of the folder in MB -/
def folder_size : ℝ := 1400

/-- The peak download speed of 4G in MB/s -/
def speed_4g : ℝ := 50

/-- The time difference in seconds between 4G and 5G downloads -/
def time_difference : ℝ := 24

/-- Theorem stating that the speed ratio is correct given the conditions -/
theorem speed_ratio_is_correct : 
  folder_size / speed_4g - folder_size / (speed_4g * speed_ratio) = time_difference :=
sorry

end NUMINAMATH_CALUDE_speed_ratio_is_correct_l28_2860


namespace NUMINAMATH_CALUDE_organize_objects_groups_l28_2864

/-- The total number of groups created when organizing objects into groups -/
def totalGroups (eggs bananas marbles : ℕ) (eggGroupSize bananaGroupSize marbleGroupSize : ℕ) : ℕ :=
  (eggs / eggGroupSize) + (bananas / bananaGroupSize) + (marbles / marbleGroupSize)

/-- Theorem stating that organizing 57 eggs in groups of 7, 120 bananas in groups of 10,
    and 248 marbles in groups of 8 results in 51 total groups -/
theorem organize_objects_groups : totalGroups 57 120 248 7 10 8 = 51 := by
  sorry

end NUMINAMATH_CALUDE_organize_objects_groups_l28_2864


namespace NUMINAMATH_CALUDE_remainder_13_pow_2000_mod_1000_l28_2815

theorem remainder_13_pow_2000_mod_1000 : 13^2000 % 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2000_mod_1000_l28_2815


namespace NUMINAMATH_CALUDE_complex_product_equals_sqrt_216_l28_2849

-- Define complex numbers p and q
variable (p q : ℂ)

-- Define the real number x
variable (x : ℝ)

-- State the theorem
theorem complex_product_equals_sqrt_216 
  (h1 : Complex.abs p = 3)
  (h2 : Complex.abs q = 5)
  (h3 : p * q = x - 3 * Complex.I) :
  x = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_sqrt_216_l28_2849


namespace NUMINAMATH_CALUDE_unique_irreverent_polynomial_exists_l28_2836

/-- A quadratic polynomial of the form x^2 - px + q -/
structure QuadraticPolynomial where
  p : ℝ
  q : ℝ

/-- The number of distinct real solutions to q(q(x)) = 0 -/
noncomputable def numSolutions (poly : QuadraticPolynomial) : ℕ := sorry

/-- The product of the roots of q(q(x)) = 0 -/
noncomputable def rootProduct (poly : QuadraticPolynomial) : ℝ := sorry

/-- Evaluates q(1) for a given quadratic polynomial -/
def evalAtOne (poly : QuadraticPolynomial) : ℝ :=
  1 - poly.p + poly.q

/-- A quadratic polynomial is irreverent if q(q(x)) = 0 has exactly four distinct real solutions -/
def isIrreverent (poly : QuadraticPolynomial) : Prop :=
  numSolutions poly = 4

theorem unique_irreverent_polynomial_exists :
  ∃! poly : QuadraticPolynomial,
    isIrreverent poly ∧
    (∀ other : QuadraticPolynomial, isIrreverent other → rootProduct poly ≤ rootProduct other) ∧
    ∃ y : ℝ, evalAtOne poly = y :=
  sorry


end NUMINAMATH_CALUDE_unique_irreverent_polynomial_exists_l28_2836


namespace NUMINAMATH_CALUDE_circle_line_intersection_l28_2838

/-- A circle in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : Point) : Prop :=
  p.x^2 + p.y^2 + c.a * p.x + c.b * p.y + c.m = 0

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two points are distinct -/
def Point.distinct (p q : Point) : Prop :=
  p.x ≠ q.x ∨ p.y ≠ q.y

/-- The circle with a given diameter passes through the origin -/
def circle_through_origin (p q : Point) : Prop :=
  ∃ (c : Circle), c.contains p ∧ c.contains q ∧ c.contains origin

/-- The main theorem -/
theorem circle_line_intersection (c : Circle) (l : Line) (p q : Point) :
  c.a = 1 ∧ c.b = -6 ∧ c.c = 1 ∧ c.d = 1 ∧
  l.a = 1 ∧ l.b = 2 ∧ l.c = -3 ∧
  c.contains p ∧ c.contains q ∧
  l.contains p ∧ l.contains q ∧
  Point.distinct p q ∧
  circle_through_origin p q →
  c.m = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l28_2838


namespace NUMINAMATH_CALUDE_jackson_score_l28_2802

/-- Given a basketball team's scoring information, calculate Jackson's score. -/
theorem jackson_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) (h1 : total_score = 72) (h2 : other_players = 7) (h3 : avg_score = 6) : total_score - other_players * avg_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_jackson_score_l28_2802


namespace NUMINAMATH_CALUDE_sum_of_max_min_l28_2818

theorem sum_of_max_min (a b c d : ℚ) : 
  a = 11/100 ∧ b = 98/100 ∧ c = 3/4 ∧ d = 2/3 →
  max a (max b (max c d)) + min a (min b (min c d)) = 109/100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_l28_2818


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l28_2828

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := y = -1/8 * x^2

-- Define the axis of symmetry
def axis_of_symmetry (y : ℝ) : Prop := y = 2

-- Theorem statement
theorem parabola_axis_of_symmetry :
  ∀ x y : ℝ, parabola_eq x y → axis_of_symmetry y := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l28_2828


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l28_2803

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 48) 
  (h2 : b * c = 20) 
  (h3 : c * a = 15) : 
  a * b * c = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l28_2803


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l28_2859

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line2D.passesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line2D.isPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line2D) 
  (point : Point2D) 
  (perp_line : Line2D) : 
  given_line.a = 1 → 
  given_line.b = 1 → 
  given_line.c = -3 →
  point.x = 2 →
  point.y = -1 →
  perp_line.a = 1 →
  perp_line.b = -1 →
  perp_line.c = -3 →
  perp_line.passesThroughPoint point ∧ 
  perp_line.isPerpendicular given_line := by
  sorry

#check perpendicular_line_through_point

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l28_2859


namespace NUMINAMATH_CALUDE_existence_of_sets_l28_2885

theorem existence_of_sets : ∃ (A B C : Set ℕ),
  (A ∩ B).Nonempty ∧
  (A ∩ C).Nonempty ∧
  ((A ∩ B) \ C).Nonempty := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sets_l28_2885


namespace NUMINAMATH_CALUDE_final_price_difference_l28_2871

def total_budget : ℝ := 1500
def tv_budget : ℝ := 1000
def sound_system_budget : ℝ := 500
def tv_discount_flat : ℝ := 150
def tv_discount_percent : ℝ := 0.20
def sound_system_discount_percent : ℝ := 0.15
def tax_rate : ℝ := 0.08

theorem final_price_difference :
  let tv_price := (tv_budget - tv_discount_flat) * (1 - tv_discount_percent)
  let sound_system_price := sound_system_budget * (1 - sound_system_discount_percent)
  let total_before_tax := tv_price + sound_system_price
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount
  total_budget - final_price = 306.60 := by sorry

end NUMINAMATH_CALUDE_final_price_difference_l28_2871


namespace NUMINAMATH_CALUDE_marias_number_l28_2853

theorem marias_number (x : ℝ) : ((3 * (x - 3) + 3) / 3 = 10) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_marias_number_l28_2853


namespace NUMINAMATH_CALUDE_greatest_integer_pi_plus_three_l28_2831

theorem greatest_integer_pi_plus_three :
  ∀ π : ℝ, 3 < π ∧ π < 4 → ⌊π + 3⌋ = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_pi_plus_three_l28_2831


namespace NUMINAMATH_CALUDE_one_fourth_of_5_6_l28_2865

theorem one_fourth_of_5_6 : (5.6 : ℚ) / 4 = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_5_6_l28_2865


namespace NUMINAMATH_CALUDE_problem_solution_l28_2855

theorem problem_solution (A B C : ℕ) 
  (h_diff1 : A ≠ B) (h_diff2 : B ≠ C) (h_diff3 : A ≠ C)
  (h1 : A + B = 84)
  (h2 : B + C = 60)
  (h3 : A = 6 * B) :
  A - C = 24 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l28_2855


namespace NUMINAMATH_CALUDE_solve_linear_equation_l28_2858

/-- Given an equation mx + ny = 6 with two known solutions, prove that m = 4 and n = 2 -/
theorem solve_linear_equation (m n : ℝ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-1) = 6) → 
  (m = 4 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l28_2858


namespace NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l28_2877

/-- The length of a spiral staircase handrail -/
theorem spiral_staircase_handrail_length 
  (turn : Real) -- angle of turn in degrees
  (rise : Real) -- height of staircase in feet
  (radius : Real) -- radius of staircase in feet
  (h1 : turn = 450)
  (h2 : rise = 15)
  (h3 : radius = 4) :
  ∃ (length : Real), 
    abs (length - Real.sqrt (rise^2 + (turn / 360 * 2 * Real.pi * radius)^2)) < 0.1 ∧ 
    abs (length - 17.4) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_spiral_staircase_handrail_length_l28_2877


namespace NUMINAMATH_CALUDE_fraction_problem_l28_2834

theorem fraction_problem (n : ℝ) (f : ℝ) : 
  n = 630.0000000000009 →
  (4/15 * 5/7 * n) > (4/9 * f * n + 8) →
  f = 0.4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l28_2834


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l28_2851

theorem imaginary_part_of_z : 
  let z : ℂ := Complex.I * ((-1 : ℂ) + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l28_2851


namespace NUMINAMATH_CALUDE_max_dominoes_on_problem_board_l28_2850

/-- Represents a cell on the grid board -/
inductive Cell
| White
| Black

/-- Represents the grid board -/
def Board := List (List Cell)

/-- Represents a domino placement on the board -/
structure Domino where
  row : Nat
  col : Nat
  horizontal : Bool

/-- Checks if a domino placement is valid on the board -/
def isValidDomino (board : Board) (domino : Domino) : Bool :=
  sorry

/-- Counts the number of valid domino placements on the board -/
def countValidDominoes (board : Board) (dominoes : List Domino) : Nat :=
  sorry

/-- The specific board layout from the problem -/
def problemBoard : Board :=
  sorry

/-- Theorem stating that the maximum number of dominoes on the problem board is 16 -/
theorem max_dominoes_on_problem_board :
  ∀ (dominoes : List Domino),
    countValidDominoes problemBoard dominoes ≤ 16 :=
  sorry

end NUMINAMATH_CALUDE_max_dominoes_on_problem_board_l28_2850


namespace NUMINAMATH_CALUDE_total_candidates_l28_2897

theorem total_candidates (avg_all : ℝ) (avg_passed : ℝ) (avg_failed : ℝ) (num_passed : ℕ) :
  avg_all = 35 →
  avg_passed = 39 →
  avg_failed = 15 →
  num_passed = 100 →
  ∃ (total : ℕ), total = 120 ∧ 
    (avg_all * total : ℝ) = avg_passed * num_passed + avg_failed * (total - num_passed) :=
by sorry

end NUMINAMATH_CALUDE_total_candidates_l28_2897


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l28_2847

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l28_2847


namespace NUMINAMATH_CALUDE_gcd_1722_966_l28_2879

theorem gcd_1722_966 : Int.gcd 1722 966 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1722_966_l28_2879


namespace NUMINAMATH_CALUDE_sin_cos_tan_relation_l28_2820

theorem sin_cos_tan_relation (A : Real) (q : Real) 
  (h1 : Real.sin A = 3/5)
  (h2 : Real.cos A / Real.tan A = q/15) :
  q = 16 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_tan_relation_l28_2820


namespace NUMINAMATH_CALUDE_absolute_value_fraction_calculation_l28_2844

theorem absolute_value_fraction_calculation : 
  |(-7)| / ((2/3) - (1/5)) - (1/2) * ((-4)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_calculation_l28_2844


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l28_2870

/-- A quadrilateral with an inscribed circle. -/
structure InscribedCircleQuadrilateral where
  /-- The radius of the inscribed circle. -/
  r : ℝ
  /-- The length of AP. -/
  ap : ℝ
  /-- The length of PB. -/
  pb : ℝ
  /-- The length of CQ. -/
  cq : ℝ
  /-- The length of QD. -/
  qd : ℝ
  /-- The circle is tangent to AB at P and to CD at Q. -/
  tangent_condition : True

/-- The theorem stating that for the given quadrilateral with inscribed circle,
    the square of the radius is 647. -/
theorem inscribed_circle_radius_squared
  (quad : InscribedCircleQuadrilateral)
  (h1 : quad.ap = 19)
  (h2 : quad.pb = 26)
  (h3 : quad.cq = 37)
  (h4 : quad.qd = 23) :
  quad.r ^ 2 = 647 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l28_2870


namespace NUMINAMATH_CALUDE_max_prime_factors_l28_2809

theorem max_prime_factors (x y : ℕ+) 
  (h_gcd : (Nat.gcd x y).factors.length = 5)
  (h_lcm : (Nat.lcm x y).factors.length = 20)
  (h_fewer : (x : ℕ).factors.length < (y : ℕ).factors.length) :
  (x : ℕ).factors.length ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_prime_factors_l28_2809


namespace NUMINAMATH_CALUDE_number_equation_l28_2886

theorem number_equation (n : ℚ) : n / 5 + 16 = 58 → n / 15 + 74 = 88 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l28_2886


namespace NUMINAMATH_CALUDE_miranda_monthly_savings_l28_2883

def heels_price : ℕ := 210
def shipping_cost : ℕ := 20
def sister_contribution : ℕ := 50
def saving_months : ℕ := 3
def total_paid : ℕ := 230

theorem miranda_monthly_savings :
  (total_paid - sister_contribution) / saving_months = 60 := by
  sorry

end NUMINAMATH_CALUDE_miranda_monthly_savings_l28_2883


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_51_l28_2837

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Counts the number of three-digit numbers x such that digit_sum(digit_sum(x)) = 4 -/
def count_special_numbers : ℕ := sorry

theorem count_special_numbers_eq_51 : count_special_numbers = 51 := by sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_51_l28_2837


namespace NUMINAMATH_CALUDE_infinite_primes_l28_2895

theorem infinite_primes (S : Finset Nat) (h : ∀ p ∈ S, Nat.Prime p) : 
  ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l28_2895


namespace NUMINAMATH_CALUDE_lychee_harvest_theorem_l28_2817

/-- Represents the lychee harvest data for a single year -/
structure LycheeHarvest where
  red : ℕ
  yellow : ℕ

/-- Calculates the percentage increase between two harvests -/
def percentageIncrease (last : LycheeHarvest) (current : LycheeHarvest) : ℚ :=
  ((current.red - last.red : ℚ) / last.red + (current.yellow - last.yellow : ℚ) / last.yellow) / 2 * 100

/-- Calculates the remaining lychees after selling and family consumption -/
def remainingLychees (harvest : LycheeHarvest) : LycheeHarvest :=
  let redAfterSelling := harvest.red - (2 * harvest.red / 3)
  let yellowAfterSelling := harvest.yellow - (3 * harvest.yellow / 7)
  let redRemaining := redAfterSelling - (3 * redAfterSelling / 5)
  let yellowRemaining := yellowAfterSelling - (4 * yellowAfterSelling / 9)
  { red := redRemaining, yellow := yellowRemaining }

theorem lychee_harvest_theorem (lastYear : LycheeHarvest) (thisYear : LycheeHarvest)
    (h1 : lastYear.red = 350)
    (h2 : lastYear.yellow = 490)
    (h3 : thisYear.red = 500)
    (h4 : thisYear.yellow = 700) :
    percentageIncrease lastYear thisYear = 42.86 ∧
    (remainingLychees thisYear).red = 67 ∧
    (remainingLychees thisYear).yellow = 223 := by
  sorry


end NUMINAMATH_CALUDE_lychee_harvest_theorem_l28_2817


namespace NUMINAMATH_CALUDE_alex_and_carla_weight_l28_2852

/-- Given the weights of pairs of individuals, prove that Alex and Carla weigh 235 pounds together. -/
theorem alex_and_carla_weight
  (alex_ben : ℝ)
  (ben_carla : ℝ)
  (carla_derek : ℝ)
  (alex_derek : ℝ)
  (h1 : alex_ben = 280)
  (h2 : ben_carla = 235)
  (h3 : carla_derek = 260)
  (h4 : alex_derek = 295) :
  ∃ (a b c d : ℝ),
    a + b = alex_ben ∧
    b + c = ben_carla ∧
    c + d = carla_derek ∧
    a + d = alex_derek ∧
    a + c = 235 := by
  sorry

end NUMINAMATH_CALUDE_alex_and_carla_weight_l28_2852


namespace NUMINAMATH_CALUDE_cos_300_degrees_l28_2827

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l28_2827


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l28_2842

theorem quadratic_function_max_value (m : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + 2*x + m
  f (1/2) > f (-1) ∧ f (1/2) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l28_2842


namespace NUMINAMATH_CALUDE_tree_spacing_l28_2816

/-- Given a sidewalk of length 148 feet where 8 trees are to be planted, 
    and each tree occupies 1 square foot, the space between each tree is 20 feet. -/
theorem tree_spacing (sidewalk_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) :
  sidewalk_length = 148 →
  num_trees = 8 →
  tree_space = 1 →
  (sidewalk_length - num_trees * tree_space) / (num_trees - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l28_2816


namespace NUMINAMATH_CALUDE_juice_subtraction_l28_2890

theorem juice_subtraction (initial_juice : ℚ) (given_away : ℚ) :
  initial_juice = 5 →
  given_away = 16 / 3 →
  initial_juice - given_away = -1 / 3 := by
sorry


end NUMINAMATH_CALUDE_juice_subtraction_l28_2890


namespace NUMINAMATH_CALUDE_quadratic_sum_l28_2848

/-- Given a quadratic function f(x) = 4x^2 - 28x - 108, prove that when written in the form
    a(x+b)^2 + c, the sum of a, b, and c is -156.5 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 4 * x^2 - 28 * x - 108) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = -156.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l28_2848


namespace NUMINAMATH_CALUDE_divisible_by_five_l28_2824

theorem divisible_by_five (a b : ℕ) : 
  a > 0 → b > 0 → 5 ∣ (a * b) → ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_five_l28_2824
