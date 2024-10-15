import Mathlib

namespace NUMINAMATH_GPT_find_line_l_l1499_149929

def line_equation (x y: ℤ) : Prop := x - 2 * y = 2

def scaling_transform_x (x: ℤ) : ℤ := x
def scaling_transform_y (y: ℤ) : ℤ := 2 * y

theorem find_line_l :
  ∀ (x y x' y': ℤ),
  x' = scaling_transform_x x →
  y' = scaling_transform_y y →
  line_equation x y →
  x' - y' = 2 := by
  sorry

end NUMINAMATH_GPT_find_line_l_l1499_149929


namespace NUMINAMATH_GPT_males_only_in_band_l1499_149941

theorem males_only_in_band
  (females_in_band : ℕ)
  (males_in_band : ℕ)
  (females_in_orchestra : ℕ)
  (males_in_orchestra : ℕ)
  (females_in_both : ℕ)
  (total_students : ℕ)
  (total_students_in_either : ℕ)
  (hf_in_band : females_in_band = 120)
  (hm_in_band : males_in_band = 90)
  (hf_in_orchestra : females_in_orchestra = 100)
  (hm_in_orchestra : males_in_orchestra = 130)
  (hf_in_both : females_in_both = 80)
  (h_total_students : total_students = 260) :
  total_students_in_either = 260 → 
  (males_in_band - (90 + 130 + 80 - 260 - 120)) = 30 :=
by
  intros h_total_students_in_either
  sorry

end NUMINAMATH_GPT_males_only_in_band_l1499_149941


namespace NUMINAMATH_GPT_number_of_polynomials_is_seven_l1499_149967

-- Definitions of what constitutes a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4*x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/5x" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Given set of algebraic expressions
def expressions : List String := 
  ["3/4*x^2", "3ab", "x+5", "y/5x", "-1", "y/3", "a^2-b^2", "a"]

-- Count the number of polynomials in the given expressions
def count_polynomials (exprs : List String) : Nat :=
  exprs.foldr (fun expr count => if is_polynomial expr then count + 1 else count) 0

theorem number_of_polynomials_is_seven : count_polynomials expressions = 7 :=
  by
    sorry

end NUMINAMATH_GPT_number_of_polynomials_is_seven_l1499_149967


namespace NUMINAMATH_GPT_photograph_area_l1499_149958

def dimensions_are_valid (a b : ℕ) : Prop :=
a > 0 ∧ b > 0 ∧ (a + 4) * (b + 5) = 77

theorem photograph_area (a b : ℕ) (h : dimensions_are_valid a b) : (a * b = 18 ∨ a * b = 14) :=
by 
  sorry

end NUMINAMATH_GPT_photograph_area_l1499_149958


namespace NUMINAMATH_GPT_coin_arrangements_l1499_149960

theorem coin_arrangements (n m : ℕ) (hp_pos : n = 5) (hq_pos : m = 5) :
  ∃ (num_arrangements : ℕ), num_arrangements = 8568 :=
by
  -- Note: 'sorry' is used to indicate here that the proof is omitted.
  sorry

end NUMINAMATH_GPT_coin_arrangements_l1499_149960


namespace NUMINAMATH_GPT_altitudes_reciprocal_sum_eq_reciprocal_inradius_l1499_149978

theorem altitudes_reciprocal_sum_eq_reciprocal_inradius
  (h1 h2 h3 r : ℝ)
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0)
  (h3_pos : h3 > 0)
  (r_pos : r > 0)
  (triangle_area_eq : ∀ (a b c : ℝ),
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧ a + b + c > 0) :
  1 / h1 + 1 / h2 + 1 / h3 = 1 / r := 
by
  sorry

end NUMINAMATH_GPT_altitudes_reciprocal_sum_eq_reciprocal_inradius_l1499_149978


namespace NUMINAMATH_GPT_halfway_between_fractions_l1499_149902

theorem halfway_between_fractions : 
  (2:ℚ) / 9 + (5 / 12) / 2 = 23 / 72 := 
sorry

end NUMINAMATH_GPT_halfway_between_fractions_l1499_149902


namespace NUMINAMATH_GPT_find_m_plus_n_l1499_149992

variable (x n m : ℝ)

def condition : Prop := (x + 5) * (x + n) = x^2 + m * x - 5

theorem find_m_plus_n (hnm : condition x n m) : m + n = 3 := 
sorry

end NUMINAMATH_GPT_find_m_plus_n_l1499_149992


namespace NUMINAMATH_GPT_skylar_total_donations_l1499_149904

-- Define the conditions
def start_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the statement to be proven
theorem skylar_total_donations : 
  (current_age - start_age) * annual_donation = 432000 := by
    sorry

end NUMINAMATH_GPT_skylar_total_donations_l1499_149904


namespace NUMINAMATH_GPT_number_of_possible_heights_is_680_l1499_149940

noncomputable def total_possible_heights : Nat :=
  let base_height := 200 * 3
  let max_additional_height := 200 * (20 - 3)
  let min_height := base_height
  let max_height := base_height + max_additional_height
  let number_of_possible_heights := (max_height - min_height) / 5 + 1
  number_of_possible_heights

theorem number_of_possible_heights_is_680 : total_possible_heights = 680 := by
  sorry

end NUMINAMATH_GPT_number_of_possible_heights_is_680_l1499_149940


namespace NUMINAMATH_GPT_average_percentage_of_10_students_l1499_149931

theorem average_percentage_of_10_students 
  (avg_15_students : ℕ := 80)
  (n_15_students : ℕ := 15)
  (total_students : ℕ := 25)
  (overall_avg : ℕ := 84) : 
  ∃ (x : ℕ), ((n_15_students * avg_15_students + 10 * x) / total_students = overall_avg) → x = 90 := 
sorry

end NUMINAMATH_GPT_average_percentage_of_10_students_l1499_149931


namespace NUMINAMATH_GPT_tv_station_ads_l1499_149959

theorem tv_station_ads (n m : ℕ) :
  n > 1 → 
  ∃ (an : ℕ → ℕ), 
  (an 0 = m) ∧ 
  (∀ k, 1 ≤ k ∧ k < n → an k = an (k - 1) - (k + (1 / 8) * (an (k - 1) - k))) ∧
  an n = 0 →
  (n = 7 ∧ m = 49) :=
by
  intro h
  exists sorry
  sorry

-- The proof steps are omitted

end NUMINAMATH_GPT_tv_station_ads_l1499_149959


namespace NUMINAMATH_GPT_min_max_value_of_expr_l1499_149934

theorem min_max_value_of_expr (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  ∃ m M : ℝ, m = 2 ∧ M = 0 ∧ ∀ x, (x = 3 * (p^3 + q^3 + r^3 + s^3) - 2 * (p^4 + q^4 + r^4 + s^4)) → m ≤ x ∧ x ≤ M :=
sorry

end NUMINAMATH_GPT_min_max_value_of_expr_l1499_149934


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1499_149907

def M := {x : ℝ | 0 < x ∧ x ≤ 3}
def N := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) (haM : a ∈ M) : (a ∈ N → a ∈ M) ∧ ¬(a ∈ M → a ∈ N) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1499_149907


namespace NUMINAMATH_GPT_time_for_C_alone_to_finish_the_job_l1499_149952

variable {A B C : ℝ} -- Declare work rates as real numbers

-- Define the conditions
axiom h1 : A + B = 1/15
axiom h2 : A + B + C = 1/10

-- Define the theorem to prove
theorem time_for_C_alone_to_finish_the_job : C = 1/30 :=
by
  apply sorry

end NUMINAMATH_GPT_time_for_C_alone_to_finish_the_job_l1499_149952


namespace NUMINAMATH_GPT_simplify_expression_l1499_149908

variable (a : ℤ)

theorem simplify_expression : (-2 * a) ^ 3 * a ^ 3 + (-3 * a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1499_149908


namespace NUMINAMATH_GPT_smallest_N_l1499_149974

theorem smallest_N (l m n : ℕ) (N : ℕ) (h_block : N = l * m * n)
  (h_invisible : (l - 1) * (m - 1) * (n - 1) = 120) :
  N = 216 :=
sorry

end NUMINAMATH_GPT_smallest_N_l1499_149974


namespace NUMINAMATH_GPT_greater_number_is_33_l1499_149920

theorem greater_number_is_33 (A B : ℕ) (hcf_11 : Nat.gcd A B = 11) (product_363 : A * B = 363) :
  max A B = 33 :=
by
  sorry

end NUMINAMATH_GPT_greater_number_is_33_l1499_149920


namespace NUMINAMATH_GPT_maximum_area_of_right_angled_triangle_l1499_149968

noncomputable def max_area_right_angled_triangle (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 48) : ℕ := 
  max (a * b / 2) 288

theorem maximum_area_of_right_angled_triangle (a b c : ℕ) 
  (h1 : a^2 + b^2 = c^2)    -- Pythagorean theorem
  (h2 : a + b + c = 48)     -- Perimeter condition
  (h3 : 0 < a)              -- Positive integer side length condition
  (h4 : 0 < b)              -- Positive integer side length condition
  (h5 : 0 < c)              -- Positive integer side length condition
  : max_area_right_angled_triangle a b c h1 h2 = 288 := 
sorry

end NUMINAMATH_GPT_maximum_area_of_right_angled_triangle_l1499_149968


namespace NUMINAMATH_GPT_value_of_x_l1499_149912

theorem value_of_x :
  ∃ x : ℝ, x = 1.13 * 80 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1499_149912


namespace NUMINAMATH_GPT_average_age_of_girls_l1499_149924

variable (B G : ℝ)
variable (age_students age_boys age_girls : ℝ)
variable (ratio_boys_girls : ℝ)

theorem average_age_of_girls :
  age_students = 15.8 ∧ age_boys = 16.2 ∧ ratio_boys_girls = 1.0000000000000044 ∧ B / G = ratio_boys_girls →
  (B * age_boys + G * age_girls) / (B + G) = age_students →
  age_girls = 15.4 :=
by
  intros hconds haverage
  sorry

end NUMINAMATH_GPT_average_age_of_girls_l1499_149924


namespace NUMINAMATH_GPT_minimum_a_plus_2c_l1499_149996

theorem minimum_a_plus_2c (a c : ℝ) (h : (1 / a) + (1 / c) = 1) : a + 2 * c ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_plus_2c_l1499_149996


namespace NUMINAMATH_GPT_susan_coins_value_l1499_149982

-- Define the conditions as Lean functions and statements.
def total_coins (n d : ℕ) := n + d = 30
def value_if_swapped (n : ℕ) := 10 * n + 5 * (30 - n)
def value_original (n : ℕ) := 5 * n + 10 * (30 - n)
def conditions (n : ℕ) := value_if_swapped n = value_original n + 90

-- The proof statement
theorem susan_coins_value (n d : ℕ) (h1 : total_coins n d) (h2 : conditions n) : 5 * n + 10 * d = 180 := by
  sorry

end NUMINAMATH_GPT_susan_coins_value_l1499_149982


namespace NUMINAMATH_GPT_inequality_solution_maximum_expression_l1499_149999

-- Problem 1: Inequality for x
theorem inequality_solution (x : ℝ) : |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 :=
by
  sorry

-- Problem 2: Maximum value for expression within [0, 1]
theorem maximum_expression (a b : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) : 
  ab + (1 - a - b) * (a + b) ≤ 1/3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_maximum_expression_l1499_149999


namespace NUMINAMATH_GPT_product_of_integers_l1499_149962

theorem product_of_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) : x * y = 168 := by
  sorry

end NUMINAMATH_GPT_product_of_integers_l1499_149962


namespace NUMINAMATH_GPT_children_got_on_the_bus_l1499_149918

-- Definitions
def original_children : ℕ := 26
def current_children : ℕ := 64

-- Theorem stating the problem
theorem children_got_on_the_bus : (current_children - original_children = 38) :=
by {
  sorry
}

end NUMINAMATH_GPT_children_got_on_the_bus_l1499_149918


namespace NUMINAMATH_GPT_averages_correct_l1499_149919

variables (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
           marksChemistry totalChemistry marksBiology totalBiology 
           marksHistory totalHistory marksGeography totalGeography : ℕ)

variables (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ)

def Kamal_average_english : Prop :=
  marksEnglish = 76 ∧ totalEnglish = 120 ∧ avgEnglish = (marksEnglish / totalEnglish) * 100

def Kamal_average_math : Prop :=
  marksMath = 65 ∧ totalMath = 150 ∧ avgMath = (marksMath / totalMath) * 100

def Kamal_average_physics : Prop :=
  marksPhysics = 82 ∧ totalPhysics = 100 ∧ avgPhysics = (marksPhysics / totalPhysics) * 100

def Kamal_average_chemistry : Prop :=
  marksChemistry = 67 ∧ totalChemistry = 80 ∧ avgChemistry = (marksChemistry / totalChemistry) * 100

def Kamal_average_biology : Prop :=
  marksBiology = 85 ∧ totalBiology = 100 ∧ avgBiology = (marksBiology / totalBiology) * 100

def Kamal_average_history : Prop :=
  marksHistory = 92 ∧ totalHistory = 150 ∧ avgHistory = (marksHistory / totalHistory) * 100

def Kamal_average_geography : Prop :=
  marksGeography = 58 ∧ totalGeography = 75 ∧ avgGeography = (marksGeography / totalGeography) * 100

theorem averages_correct :
  ∀ (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
      marksChemistry totalChemistry marksBiology totalBiology 
      marksHistory totalHistory marksGeography totalGeography : ℕ),
  ∀ (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ),
  Kamal_average_english marksEnglish totalEnglish avgEnglish →
  Kamal_average_math marksMath totalMath avgMath →
  Kamal_average_physics marksPhysics totalPhysics avgPhysics →
  Kamal_average_chemistry marksChemistry totalChemistry avgChemistry →
  Kamal_average_biology marksBiology totalBiology avgBiology →
  Kamal_average_history marksHistory totalHistory avgHistory →
  Kamal_average_geography marksGeography totalGeography avgGeography →
  avgEnglish = 63.33 ∧ avgMath = 43.33 ∧ avgPhysics = 82 ∧
  avgChemistry = 83.75 ∧ avgBiology = 85 ∧ avgHistory = 61.33 ∧ avgGeography = 77.33 :=
by
  sorry

end NUMINAMATH_GPT_averages_correct_l1499_149919


namespace NUMINAMATH_GPT_problem_statement_l1499_149910

noncomputable def f : ℕ+ → ℝ := sorry

theorem problem_statement (x : ℕ+) :
  (f 1 = 1) →
  (∀ x, f (x + 1) = (2 * f x) / (f x + 2)) →
  f x = 2 / (x + 1) := 
sorry

end NUMINAMATH_GPT_problem_statement_l1499_149910


namespace NUMINAMATH_GPT_original_stickers_l1499_149942

theorem original_stickers (x : ℕ) (h₁ : x * 3 / 4 * 4 / 5 = 45) : x = 75 :=
by
  sorry

end NUMINAMATH_GPT_original_stickers_l1499_149942


namespace NUMINAMATH_GPT_two_digit_number_satisfying_conditions_l1499_149948

theorem two_digit_number_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 8 ∧
  ∀ p ∈ s, ∃ (a b : ℕ), p = (a, b) ∧
    (10 * a + b < 100) ∧
    (a ≥ 2) ∧
    (10 * a + b + 10 * b + a = 110) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_satisfying_conditions_l1499_149948


namespace NUMINAMATH_GPT_total_presents_l1499_149949

variables (ChristmasPresents BirthdayPresents EasterPresents HalloweenPresents : ℕ)

-- Given conditions
def condition1 : ChristmasPresents = 60 := sorry
def condition2 : BirthdayPresents = 3 * EasterPresents := sorry
def condition3 : EasterPresents = (ChristmasPresents / 2) - 10 := sorry
def condition4 : HalloweenPresents = BirthdayPresents - EasterPresents := sorry

-- Proof statement
theorem total_presents (h1 : ChristmasPresents = 60)
    (h2 : BirthdayPresents = 3 * EasterPresents)
    (h3 : EasterPresents = (ChristmasPresents / 2) - 10)
    (h4 : HalloweenPresents = BirthdayPresents - EasterPresents) :
    ChristmasPresents + BirthdayPresents + EasterPresents + HalloweenPresents = 180 :=
sorry

end NUMINAMATH_GPT_total_presents_l1499_149949


namespace NUMINAMATH_GPT_complete_square_identity_l1499_149972

theorem complete_square_identity (x : ℝ) : ∃ (d e : ℤ), (x^2 - 10 * x + 13 = 0 → (x + d)^2 = e ∧ d + e = 7) :=
sorry

end NUMINAMATH_GPT_complete_square_identity_l1499_149972


namespace NUMINAMATH_GPT_growth_rate_correct_max_avg_visitors_correct_l1499_149900

-- Define the conditions from part 1
def visitors_march : ℕ := 80000
def visitors_may : ℕ := 125000

-- Define the monthly average growth rate
def monthly_avg_growth_rate (x : ℝ) : Prop :=
(1 + x)^2 = (visitors_may / visitors_march : ℝ)

-- Define the condition for June
def visitors_june_1_10 : ℕ := 66250
def max_avg_visitors_per_day (y : ℝ) : Prop :=
6.625 + 20 * y ≤ 15.625

-- Prove the monthly growth rate
theorem growth_rate_correct : ∃ x : ℝ, monthly_avg_growth_rate x ∧ x = 0.25 := sorry

-- Prove the max average visitors per day in June
theorem max_avg_visitors_correct : ∃ y : ℝ, max_avg_visitors_per_day y ∧ y = 0.45 := sorry

end NUMINAMATH_GPT_growth_rate_correct_max_avg_visitors_correct_l1499_149900


namespace NUMINAMATH_GPT_calculate_flat_rate_shipping_l1499_149965

noncomputable def flat_rate_shipping : ℝ :=
  17.00

theorem calculate_flat_rate_shipping
  (price_per_shirt : ℝ)
  (num_shirts : ℤ)
  (price_pack_socks : ℝ)
  (num_packs_socks : ℤ)
  (price_per_short : ℝ)
  (num_shorts : ℤ)
  (price_swim_trunks : ℝ)
  (num_swim_trunks : ℤ)
  (total_bill : ℝ)
  (total_items_cost : ℝ)
  (shipping_cost : ℝ) :
  price_per_shirt * num_shirts + 
  price_pack_socks * num_packs_socks + 
  price_per_short * num_shorts +
  price_swim_trunks * num_swim_trunks = total_items_cost →
  total_bill - total_items_cost = shipping_cost →
  total_items_cost > 50 → 
  0.20 * total_items_cost ≠ shipping_cost →
  flat_rate_shipping = 17.00 := 
sorry

end NUMINAMATH_GPT_calculate_flat_rate_shipping_l1499_149965


namespace NUMINAMATH_GPT_find_m_of_quad_roots_l1499_149914

theorem find_m_of_quad_roots
  (a b : ℝ) (m : ℝ)
  (ha : a = 5)
  (hb : b = -4)
  (h_roots : ∀ x : ℂ, (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ x = (2 - Complex.I * Real.sqrt 143) / 5) →
                     (a * x^2 + b * x + m = 0)) :
  m = 7.95 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_m_of_quad_roots_l1499_149914


namespace NUMINAMATH_GPT_find_b_value_l1499_149964

noncomputable def find_b (p q : ℕ) : ℕ := p^2 + q^2

theorem find_b_value
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h_distinct : p ≠ q) (h_roots : p + q = 13 ∧ p * q = 22) :
  find_b p q = 125 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1499_149964


namespace NUMINAMATH_GPT_Carla_final_position_l1499_149936

-- Carla's initial position
def Carla_initial_position : ℤ × ℤ := (10, -10)

-- Function to calculate Carla's new position after each move
def Carla_move (pos : ℤ × ℤ) (direction : ℕ) (distance : ℤ) : ℤ × ℤ :=
  match direction % 4 with
  | 0 => (pos.1, pos.2 + distance)   -- North
  | 1 => (pos.1 + distance, pos.2)   -- East
  | 2 => (pos.1, pos.2 - distance)   -- South
  | 3 => (pos.1 - distance, pos.2)   -- West
  | _ => pos  -- This case will never happen due to the modulo operation

-- Recursive function to simulate Carla's journey
def Carla_journey : ℕ → ℤ × ℤ → ℤ × ℤ 
  | 0, pos => pos
  | n + 1, pos => 
    let next_pos := Carla_move pos n (2 + n / 2 * 2)
    Carla_journey n next_pos

-- Prove that after 100 moves, Carla's position is (-191, -10)
theorem Carla_final_position : Carla_journey 100 Carla_initial_position = (-191, -10) :=
sorry

end NUMINAMATH_GPT_Carla_final_position_l1499_149936


namespace NUMINAMATH_GPT_eugene_used_six_boxes_of_toothpicks_l1499_149911

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end NUMINAMATH_GPT_eugene_used_six_boxes_of_toothpicks_l1499_149911


namespace NUMINAMATH_GPT_initial_legos_l1499_149925

-- Definitions and conditions
def legos_won : ℝ := 17.0
def legos_now : ℝ := 2097.0

-- The statement to prove
theorem initial_legos : (legos_now - legos_won) = 2080 :=
by sorry

end NUMINAMATH_GPT_initial_legos_l1499_149925


namespace NUMINAMATH_GPT_find_number_l1499_149905

-- Define the variables and the conditions as theorems to be proven in Lean.
theorem find_number (x : ℤ) 
  (h1 : (x - 16) % 37 = 0)
  (h2 : (x - 16) / 37 = 23) :
  x = 867 :=
sorry

end NUMINAMATH_GPT_find_number_l1499_149905


namespace NUMINAMATH_GPT_fraction_of_remaining_prize_money_each_winner_receives_l1499_149935

-- Definitions based on conditions
def total_prize_money : ℕ := 2400
def first_winner_fraction : ℚ := 1 / 3
def each_following_winner_prize : ℕ := 160

-- Calculate the first winner's prize
def first_winner_prize : ℚ := first_winner_fraction * total_prize_money

-- Calculate the remaining prize money after the first winner
def remaining_prize_money : ℚ := total_prize_money - first_winner_prize

-- Calculate the fraction of the remaining prize money that each of the next ten winners will receive
def following_winner_fraction : ℚ := each_following_winner_prize / remaining_prize_money

-- Theorem statement
theorem fraction_of_remaining_prize_money_each_winner_receives :
  following_winner_fraction = 1 / 10 :=
sorry

end NUMINAMATH_GPT_fraction_of_remaining_prize_money_each_winner_receives_l1499_149935


namespace NUMINAMATH_GPT_min_pairs_opponents_statement_l1499_149945

-- Problem statement definitions
variables (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2)

-- Required minimum number of pairs of opponents in a parliament
def min_pairs_opponents (h p : ℕ) : ℕ :=
  min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)

-- Proof statement
theorem min_pairs_opponents_statement (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2) :
  ∀ (hp : ℕ), ∃ (pairs : ℕ), 
    pairs = min_pairs_opponents h p :=
  sorry

end NUMINAMATH_GPT_min_pairs_opponents_statement_l1499_149945


namespace NUMINAMATH_GPT_root_poly_ratio_c_d_l1499_149957

theorem root_poly_ratio_c_d (a b c d : ℝ)
  (h₁ : 1 + (-2) + 3 = 2)
  (h₂ : 1 * (-2) + (-2) * 3 + 3 * 1 = -5)
  (h₃ : 1 * (-2) * 3 = -6)
  (h_sum : -b / a = 2)
  (h_pair_prod : c / a = -5)
  (h_prod : -d / a = -6) :
  c / d = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_root_poly_ratio_c_d_l1499_149957


namespace NUMINAMATH_GPT_radius_B_l1499_149955

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end NUMINAMATH_GPT_radius_B_l1499_149955


namespace NUMINAMATH_GPT_proof_problem_l1499_149903

variable {a_n : ℕ → ℤ}
variable {b_n : ℕ → ℤ}
variable {c_n : ℕ → ℤ}
variable {T_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Conditions

-- 1. The common difference d of the arithmetic sequence {a_n} is greater than 0
def common_difference_positive (d : ℤ) : Prop :=
  d > 0

-- 2. a_2 and a_5 are the two roots of the equation x^2 - 12x + 27 = 0
def roots_of_quadratic (a2 a5 : ℤ) : Prop :=
  a2^2 - 12 * a2 + 27 = 0 ∧ a5^2 - 12 * a5 + 27 = 0

-- 3. The sum of the first n terms of the sequence {b_n} is S_n, and it is given that S_n = (3 / 2)(b_n - 1)
def sum_of_b_n (S_n b_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = 3/2 * (b_n n - 1)

-- Define the sequences to display further characteristics

-- 1. Find the general formula for the sequences {a_n} and {b_n}
def general_formula_a (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def general_formula_b (b : ℕ → ℤ) : Prop :=
  ∀ n, b n = 3 ^ n

-- 2. Check if c_n = a_n * b_n and find the sum T_n
def c_n_equals_a_n_times_b_n (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n * b n

def sum_T_n (T c : ℕ → ℤ) : Prop :=
  ∀ n, T n = 3 + (n - 1) * 3^(n + 1)

theorem proof_problem 
  (d : ℤ)
  (a2 a5 : ℤ)
  (S_n b_n : ℕ → ℤ)
  (a_n b_n c_n T_n : ℕ → ℤ) :
  common_difference_positive d ∧
  roots_of_quadratic a2 a5 ∧ 
  sum_of_b_n S_n b_n ∧ 
  general_formula_a a_n ∧ 
  general_formula_b b_n ∧ 
  c_n_equals_a_n_times_b_n a_n b_n c_n ∧ 
  sum_T_n T_n c_n :=
sorry

end NUMINAMATH_GPT_proof_problem_l1499_149903


namespace NUMINAMATH_GPT_arith_seq_ninth_term_value_l1499_149956

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end NUMINAMATH_GPT_arith_seq_ninth_term_value_l1499_149956


namespace NUMINAMATH_GPT_average_salary_of_all_workers_l1499_149961

def totalTechnicians : Nat := 6
def avgSalaryTechnician : Nat := 12000
def restWorkers : Nat := 6
def avgSalaryRest : Nat := 6000
def totalWorkers : Nat := 12
def totalSalary := (totalTechnicians * avgSalaryTechnician) + (restWorkers * avgSalaryRest)

theorem average_salary_of_all_workers : totalSalary / totalWorkers = 9000 := 
by
    -- replace with mathematical proof once available
    sorry

end NUMINAMATH_GPT_average_salary_of_all_workers_l1499_149961


namespace NUMINAMATH_GPT_distinct_cube_units_digits_l1499_149989

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_cube_units_digits_l1499_149989


namespace NUMINAMATH_GPT_cos_inequality_m_range_l1499_149980

theorem cos_inequality_m_range (m : ℝ) : 
  (-1 ≤ 1 - m ∧ 1 - m ≤ 1) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_GPT_cos_inequality_m_range_l1499_149980


namespace NUMINAMATH_GPT_katy_books_l1499_149906

theorem katy_books (june july aug : ℕ) (h1 : june = 8) (h2 : july = 2 * june) (h3 : june + july + aug = 37) :
  july - aug = 3 :=
by sorry

end NUMINAMATH_GPT_katy_books_l1499_149906


namespace NUMINAMATH_GPT_max_d_minus_r_l1499_149981

theorem max_d_minus_r (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) : 
  d - r = 35 :=
sorry

end NUMINAMATH_GPT_max_d_minus_r_l1499_149981


namespace NUMINAMATH_GPT_main_theorem_l1499_149937

-- The condition
def condition (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (2^x - 1) = 4^x - 1

-- The property we need to prove
def proves (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, -1 ≤ x → f x = x^2 + 2*x

-- The main theorem connecting the condition to the desired property
theorem main_theorem (f : ℝ → ℝ) (h : condition f) : proves f :=
sorry

end NUMINAMATH_GPT_main_theorem_l1499_149937


namespace NUMINAMATH_GPT_extreme_points_inequality_l1499_149993

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 + a * Real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - 4 * x + 2

theorem extreme_points_inequality (a : ℝ) (h_a : 0 < a ∧ a < 1) (x0 : ℝ)
  (h_ext : 4 * x0^2 - 4 * x0 + a = 0) (h_min : ∃ x1, x0 + x1 = 1 ∧ x0 < x1 ∧ x1 < 1) :
  g x0 a > 1 / 2 - Real.log 2 :=
sorry

end NUMINAMATH_GPT_extreme_points_inequality_l1499_149993


namespace NUMINAMATH_GPT_compound_interest_calculation_l1499_149901

theorem compound_interest_calculation :
  let SI := (1833.33 * 16 * 6) / 100
  let CI := 2 * SI
  let principal_ci := 8000
  let rate_ci := 20
  let n := Real.log (1.4399995) / Real.log (1 + rate_ci / 100)
  n = 2 := by
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l1499_149901


namespace NUMINAMATH_GPT_pinwheel_area_eq_six_l1499_149963

open Set

/-- Define the pinwheel in a 6x6 grid -/
def is_midpoint (x y : ℤ) : Prop :=
  (x = 3 ∧ (y = 1 ∨ y = 5)) ∨ (y = 3 ∧ (x = 1 ∨ x = 5))

def is_center (x y : ℤ) : Prop :=
  x = 3 ∧ y = 3

def is_triangle_vertex (x y : ℤ) : Prop :=
  is_center x y ∨ is_midpoint x y

-- Main theorem statement
theorem pinwheel_area_eq_six :
  let pinwheel : Set (ℤ × ℤ) := {p | is_triangle_vertex p.1 p.2}
  ∀ A : ℝ, A = 6 :=
by sorry

end NUMINAMATH_GPT_pinwheel_area_eq_six_l1499_149963


namespace NUMINAMATH_GPT_find_k_l1499_149995

def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 60 < f a b c 9 ∧ f a b c 9 < 70)
  (h3 : 90 < f a b c 10 ∧ f a b c 10 < 100)
  (h4 : ∃ k : ℤ, 10000 * k < f a b c 100 ∧ f a b c 100 < 10000 * (k + 1))
  : k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l1499_149995


namespace NUMINAMATH_GPT_normal_CDF_is_correct_l1499_149970

noncomputable def normal_cdf (a σ : ℝ) (x : ℝ) : ℝ :=
  0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2)

theorem normal_CDF_is_correct (a σ : ℝ) (ha : σ > 0) (x : ℝ) :
  (normal_cdf a σ x) = 0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_normal_CDF_is_correct_l1499_149970


namespace NUMINAMATH_GPT_range_of_a_l1499_149976

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Proposition P: f(x) has a root in the interval [-1, 1]
def P (a : ℝ) : Prop := ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0

-- Proposition Q: There is only one real number x satisfying the inequality
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- The theorem stating the range of a if either P or Q is false
theorem range_of_a (a : ℝ) : ¬(P a) ∨ ¬(Q a) → (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1499_149976


namespace NUMINAMATH_GPT_whale_population_ratio_l1499_149997

theorem whale_population_ratio 
  (W_last : ℕ)
  (W_this : ℕ)
  (W_next : ℕ)
  (h1 : W_last = 4000)
  (h2 : W_next = W_this + 800)
  (h3 : W_next = 8800) :
  (W_this / W_last) = 2 := by
  sorry

end NUMINAMATH_GPT_whale_population_ratio_l1499_149997


namespace NUMINAMATH_GPT_original_group_men_l1499_149990

-- Let's define the parameters of the problem
def original_days := 55
def absent_men := 15
def completed_days := 60

-- We need to show that the number of original men (x) is 180
theorem original_group_men (x : ℕ) (h : x * original_days = (x - absent_men) * completed_days) : x = 180 :=
by
  sorry

end NUMINAMATH_GPT_original_group_men_l1499_149990


namespace NUMINAMATH_GPT_find_larger_number_l1499_149991

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1499_149991


namespace NUMINAMATH_GPT_abs_c_five_l1499_149915

theorem abs_c_five (a b c : ℤ) (h_coprime : Int.gcd a (Int.gcd b c) = 1) 
  (h1 : a = 2 * (b + c)) 
  (h2 : b = 3 * (a + c)) : 
  |c| = 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_c_five_l1499_149915


namespace NUMINAMATH_GPT_parking_space_area_l1499_149927

theorem parking_space_area
  (L : ℕ) (W : ℕ)
  (hL : L = 9)
  (hSum : 2 * W + L = 37) : L * W = 126 := 
by
  sorry

end NUMINAMATH_GPT_parking_space_area_l1499_149927


namespace NUMINAMATH_GPT_solve_inequality_range_of_a_l1499_149987

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define the set A
def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- First part: Solve the inequality f(x) ≤ 3a^2 + 1 when a ≠ 0
-- Solution would be translated in a theorem
theorem solve_inequality (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, f x a ≤ 3 * a^2 + 1 → if a > 0 then -a ≤ x ∧ x ≤ 3 * a else -3 * a ≤ x ∧ x ≤ a :=
sorry

-- Second part: Find the range of a if there exists no x0 ∈ A such that f(x0) ≤ A is false
theorem range_of_a (a : ℝ) :
  (∀ x ∈ A, f x a > 0) ↔ a < 1 :=
sorry

end NUMINAMATH_GPT_solve_inequality_range_of_a_l1499_149987


namespace NUMINAMATH_GPT_a2_plus_b2_minus_abc_is_perfect_square_l1499_149947

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem a2_plus_b2_minus_abc_is_perfect_square {a b c : ℕ} (h : 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c) :
  is_perfect_square (a^2 + b^2 - a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_a2_plus_b2_minus_abc_is_perfect_square_l1499_149947


namespace NUMINAMATH_GPT_contrapositive_proof_l1499_149969

theorem contrapositive_proof (x m : ℝ) :
  (m < 0 → (∃ r : ℝ, r * r + 3 * r + m = 0)) ↔
  (¬ (∃ r : ℝ, r * r + 3 * r + m = 0) → m ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proof_l1499_149969


namespace NUMINAMATH_GPT_inclination_angle_x_equals_3_is_90_l1499_149977

-- Define the condition that line x = 3 is vertical
def is_vertical_line (x : ℝ) : Prop := x = 3

-- Define the inclination angle property for a vertical line
def inclination_angle_of_vertical_line_is_90 (x : ℝ) (h : is_vertical_line x) : ℝ :=
90   -- The angle is 90 degrees

-- Theorem statement to prove the inclination angle of the line x = 3 is 90 degrees
theorem inclination_angle_x_equals_3_is_90 :
  inclination_angle_of_vertical_line_is_90 3 (by simp [is_vertical_line]) = 90 :=
sorry  -- proof goes here


end NUMINAMATH_GPT_inclination_angle_x_equals_3_is_90_l1499_149977


namespace NUMINAMATH_GPT_correct_equation_l1499_149917

noncomputable def team_a_initial := 96
noncomputable def team_b_initial := 72
noncomputable def team_b_final (x : ℕ) := team_b_initial - x
noncomputable def team_a_final (x : ℕ) := team_a_initial + x

theorem correct_equation (x : ℕ) : 
  (1 / 3 : ℚ) * (team_a_final x) = (team_b_final x) := 
  sorry

end NUMINAMATH_GPT_correct_equation_l1499_149917


namespace NUMINAMATH_GPT_cos_of_tan_l1499_149938

/-- Given a triangle ABC with angle A such that tan(A) = -5/12, prove cos(A) = -12/13. -/
theorem cos_of_tan (A : ℝ) (h : Real.tan A = -5 / 12) : Real.cos A = -12 / 13 := by
  sorry

end NUMINAMATH_GPT_cos_of_tan_l1499_149938


namespace NUMINAMATH_GPT_countNegativeValues_l1499_149922

-- Define the condition that sqrt(x + 122) is a positive integer
noncomputable def isPositiveInteger (n : ℤ) (x : ℤ) : Prop :=
  ∃ n : ℤ, (n > 0) ∧ (x + 122 = n * n)

-- Define the condition that x is negative
def isNegative (x : ℤ) : Prop :=
  x < 0

-- Prove the number of different negative values of x such that sqrt(x + 122) is a positive integer is 11
theorem countNegativeValues :
  ∃ x_set : Finset ℤ, (∀ x ∈ x_set, isNegative x ∧ isPositiveInteger x (x + 122)) ∧ x_set.card = 11 :=
sorry

end NUMINAMATH_GPT_countNegativeValues_l1499_149922


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1499_149923

theorem sufficient_but_not_necessary_condition 
(a b : ℝ) : (b ≥ 0) → ((a + 1)^2 + b ≥ 0) ∧ (¬ (∀ a b, ((a + 1)^2 + b ≥ 0) → b ≥ 0)) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1499_149923


namespace NUMINAMATH_GPT_relationship_between_x_y_l1499_149954

theorem relationship_between_x_y (x y m : ℝ) (h₁ : x + m = 4) (h₂ : y - 5 = m) : x + y = 9 := 
sorry

end NUMINAMATH_GPT_relationship_between_x_y_l1499_149954


namespace NUMINAMATH_GPT_multiply_469111111_by_99999999_l1499_149921

theorem multiply_469111111_by_99999999 :
  469111111 * 99999999 = 46911111053088889 :=
sorry

end NUMINAMATH_GPT_multiply_469111111_by_99999999_l1499_149921


namespace NUMINAMATH_GPT_striped_nails_painted_l1499_149933

theorem striped_nails_painted (total_nails purple_nails blue_nails : ℕ) (h_total : total_nails = 20)
    (h_purple : purple_nails = 6) (h_blue : blue_nails = 8)
    (h_diff_percent : |(blue_nails:ℚ) / total_nails * 100 - 
    ((total_nails - purple_nails - blue_nails):ℚ) / total_nails * 100| = 10) :
    (total_nails - purple_nails - blue_nails) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_striped_nails_painted_l1499_149933


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1499_149944

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (n : ℕ)
  (a1 : ℤ)
  (d : ℤ)
  (h1 : a1 = 2)
  (h2 : a_n 5 = a_n 1 + 4 * d)
  (h3 : a_n 3 = a_n 1 + 2 * d)
  (h4 : a_n 5 = 3 * a_n 3) :
  S_n 9 = -54 := 
by  
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1499_149944


namespace NUMINAMATH_GPT_sum_difference_l1499_149975

noncomputable def sum_arith_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference :
  let S_even := sum_arith_seq 2 2 1001
  let S_odd := sum_arith_seq 1 2 1002
  S_odd - S_even = 1002 :=
by
  sorry

end NUMINAMATH_GPT_sum_difference_l1499_149975


namespace NUMINAMATH_GPT_solve_quadratic_for_q_l1499_149994

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end NUMINAMATH_GPT_solve_quadratic_for_q_l1499_149994


namespace NUMINAMATH_GPT_right_angled_triangle_hypotenuse_and_altitude_relation_l1499_149985

variables (a b c m : ℝ)

theorem right_angled_triangle_hypotenuse_and_altitude_relation
  (h1 : b^2 + c^2 = a^2)
  (h2 : m^2 = (b - c)^2)
  (h3 : b * c = a * m) :
  m = (a * (Real.sqrt 5 - 1)) / 2 := 
sorry

end NUMINAMATH_GPT_right_angled_triangle_hypotenuse_and_altitude_relation_l1499_149985


namespace NUMINAMATH_GPT_range_of_m_l1499_149939

variables {m x : ℝ}

def p (m : ℝ) : Prop := (16 * (m - 2)^2 - 16 > 0) ∧ (m - 2 < 0)
def q (m : ℝ) : Prop := (9 * m^2 - 4 < 0)
def pq (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(q m)

theorem range_of_m (h : pq m) : m ≤ -2/3 ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1499_149939


namespace NUMINAMATH_GPT_passengers_with_round_trip_tickets_l1499_149979

theorem passengers_with_round_trip_tickets (P R : ℝ) : 
  (0.40 * R = 0.25 * P) → (R / P = 0.625) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_passengers_with_round_trip_tickets_l1499_149979


namespace NUMINAMATH_GPT_star_three_five_l1499_149950

def star (x y : ℕ) := x^2 + 2 * x * y + y^2

theorem star_three_five : star 3 5 = 64 :=
by
  sorry

end NUMINAMATH_GPT_star_three_five_l1499_149950


namespace NUMINAMATH_GPT_hcf_two_numbers_l1499_149913

theorem hcf_two_numbers (H a b : ℕ) (coprime_ab : Nat.gcd a b = 1) 
    (lcm_factors : a * b = 150) (larger_num : H * a = 450 ∨ H * b = 450) : H = 30 := 
by
  sorry

end NUMINAMATH_GPT_hcf_two_numbers_l1499_149913


namespace NUMINAMATH_GPT_inequality_proof_l1499_149946

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y * z) / (Real.sqrt (2 * x^2 * (y + z))) + 
  (y^2 + z * x) / (Real.sqrt (2 * y^2 * (z + x))) + 
  (z^2 + x * y) / (Real.sqrt (2 * z^2 * (x + y))) ≥ 1 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1499_149946


namespace NUMINAMATH_GPT_sqrt_solution_l1499_149998

theorem sqrt_solution (x : ℝ) (h : x = Real.sqrt (1 + x)) : 1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_solution_l1499_149998


namespace NUMINAMATH_GPT_oranges_kilos_bought_l1499_149953

-- Definitions based on the given conditions
variable (O A x : ℝ)

-- Definitions from conditions
def A_value : Prop := A = 29
def equation1 : Prop := x * O + 5 * A = 419
def equation2 : Prop := 5 * O + 7 * A = 488

-- The theorem we want to prove
theorem oranges_kilos_bought {O A x : ℝ} (A_value: A = 29) (h1: x * O + 5 * A = 419) (h2: 5 * O + 7 * A = 488) : x = 5 :=
by
  -- start of proof
  sorry  -- proof omitted

end NUMINAMATH_GPT_oranges_kilos_bought_l1499_149953


namespace NUMINAMATH_GPT_coin_flip_sequences_l1499_149926

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_l1499_149926


namespace NUMINAMATH_GPT_tiffany_total_bags_l1499_149932

-- Define the initial and additional bags correctly
def bags_on_monday : ℕ := 10
def bags_next_day : ℕ := 3
def bags_day_after : ℕ := 7

-- Define the total bags calculation
def total_bags (initial : ℕ) (next : ℕ) (after : ℕ) : ℕ :=
  initial + next + after

-- Prove that the total bags collected is 20
theorem tiffany_total_bags : total_bags bags_on_monday bags_next_day bags_day_after = 20 :=
by
  sorry

end NUMINAMATH_GPT_tiffany_total_bags_l1499_149932


namespace NUMINAMATH_GPT_isosceles_triangle_area_of_triangle_l1499_149966

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom triangle_sides (a b c : ℝ) (A B C : ℝ) : c = 2
axiom cosine_condition (a b c : ℝ) (A B C : ℝ) : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B

-- Questions
theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ)
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B) :
  a = b :=
sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B)
  (h3 : 7 * Real.cos B = 2 * Real.cos C) 
  (h4 : a = b) :
  ∃ S : ℝ, S = Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_area_of_triangle_l1499_149966


namespace NUMINAMATH_GPT_age_ratio_l1499_149973

variables (A B : ℕ)
def present_age_of_A : ℕ := 15
def future_ratio (A B : ℕ) : Prop := (A + 6) / (B + 6) = 7 / 5

theorem age_ratio (A_eq : A = present_age_of_A) (future_ratio_cond : future_ratio A B) : A / B = 5 / 3 :=
sorry

end NUMINAMATH_GPT_age_ratio_l1499_149973


namespace NUMINAMATH_GPT_kelly_games_l1499_149986

theorem kelly_games (initial_games give_away in_stock : ℕ) (h1 : initial_games = 50) (h2 : in_stock = 35) :
  give_away = initial_games - in_stock :=
by {
  -- initial_games = 50
  -- in_stock = 35
  -- Therefore, give_away = initial_games - in_stock
  sorry
}

end NUMINAMATH_GPT_kelly_games_l1499_149986


namespace NUMINAMATH_GPT_abs_add_lt_abs_sub_l1499_149983

variable {a b : ℝ}

theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| :=
sorry

end NUMINAMATH_GPT_abs_add_lt_abs_sub_l1499_149983


namespace NUMINAMATH_GPT_total_cost_correct_l1499_149951

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1499_149951


namespace NUMINAMATH_GPT_find_m_l1499_149928

noncomputable def m_solution (m : ℝ) : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)

theorem find_m :
  ∀ (m : ℝ), Complex.im (m_solution m) ≠ 0 → Complex.re (m_solution m) = 0 → m = 3 / 2 :=
by
  intro m h_im h_re
  sorry

end NUMINAMATH_GPT_find_m_l1499_149928


namespace NUMINAMATH_GPT_find_additional_payment_l1499_149943

-- Definitions used from the conditions
def total_payments : ℕ := 52
def first_partial_payments : ℕ := 25
def second_partial_payments : ℕ := total_payments - first_partial_payments
def first_payment_amount : ℝ := 500
def average_payment : ℝ := 551.9230769230769

-- Condition in Lean
theorem find_additional_payment :
  let total_amount := average_payment * total_payments
  let first_payment_total := first_partial_payments * first_payment_amount
  ∃ x : ℝ, total_amount = first_payment_total + second_partial_payments * (first_payment_amount + x) → x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_additional_payment_l1499_149943


namespace NUMINAMATH_GPT_maximum_distance_point_to_line_l1499_149971

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m - 1) * x + m * y + 2 = 0

-- Statement of the problem to prove
theorem maximum_distance_point_to_line :
  ∀ (x y m : ℝ), circle_C x y → ∃ P : ℝ, line_l m x y → P = 6 :=
by 
  sorry

end NUMINAMATH_GPT_maximum_distance_point_to_line_l1499_149971


namespace NUMINAMATH_GPT_petya_points_l1499_149909

noncomputable def points_after_disqualification : ℕ :=
4

theorem petya_points (players: ℕ) (initial_points: ℕ) (disqualified: ℕ) (new_points: ℕ) : 
  players = 10 → 
  initial_points < (players * (players - 1) / 2) / players → 
  disqualified = 2 → 
  (players - disqualified) * (players - disqualified - 1) / 2 = new_points →
  new_points / (players - disqualified) < points_after_disqualification →
  points_after_disqualification > new_points / (players - disqualified) →
  points_after_disqualification = 4 :=
by 
  intros 
  exact sorry

end NUMINAMATH_GPT_petya_points_l1499_149909


namespace NUMINAMATH_GPT_sam_correct_percent_l1499_149984

variable (y : ℝ)
variable (h_pos : 0 < y)

theorem sam_correct_percent :
  ((8 * y - 3 * y) / (8 * y) * 100) = 62.5 := by
sorry

end NUMINAMATH_GPT_sam_correct_percent_l1499_149984


namespace NUMINAMATH_GPT_number_of_real_z5_is_10_l1499_149916

theorem number_of_real_z5_is_10 :
  ∃ S : Finset ℂ, (∀ z ∈ S, z ^ 30 = 1 ∧ (z ^ 5).im = 0) ∧ S.card = 10 :=
sorry

end NUMINAMATH_GPT_number_of_real_z5_is_10_l1499_149916


namespace NUMINAMATH_GPT_trapezoid_median_properties_l1499_149930

-- Define the variables
variables (a b x : ℝ)

-- State the conditions and the theorem
theorem trapezoid_median_properties (h1 : x = (2 * a) / 3) (h2 : x = b + 3) (h3 : x = (a + b) / 2) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_median_properties_l1499_149930


namespace NUMINAMATH_GPT_g_at_4_l1499_149988

noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 5 / x
noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_4 : g 4 = 8.142857 := by
  sorry

end NUMINAMATH_GPT_g_at_4_l1499_149988
