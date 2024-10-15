import Mathlib

namespace NUMINAMATH_GPT_reduced_price_per_kg_l60_6012

/-- Given that:
1. There is a reduction of 25% in the price of oil.
2. The housewife can buy 5 kgs more for Rs. 700 after the reduction.

Prove that the reduced price per kg of oil is Rs. 35. -/
theorem reduced_price_per_kg (P : ℝ) (R : ℝ) (X : ℝ)
  (h1 : R = 0.75 * P)
  (h2 : 700 = X * P)
  (h3 : 700 = (X + 5) * R)
  : R = 35 := 
sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l60_6012


namespace NUMINAMATH_GPT_find_one_third_of_product_l60_6083

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_find_one_third_of_product_l60_6083


namespace NUMINAMATH_GPT_cubic_ineq_l60_6065

theorem cubic_ineq (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end NUMINAMATH_GPT_cubic_ineq_l60_6065


namespace NUMINAMATH_GPT_B_spends_85_percent_l60_6044

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 4000
def A_savings_percentage : ℝ := 0.05
def A_salary : ℝ := 3000
def B_salary : ℝ := 4000 - A_salary
def equal_savings (S_A S_B : ℝ) : Prop := A_savings_percentage * S_A = (1 - S_B / 100) * B_salary

theorem B_spends_85_percent (S_A S_B : ℝ) (B_spending_percentage : ℝ) :
  combined_salary S_A S_B ∧ S_A = A_salary ∧ equal_savings S_A B_spending_percentage → B_spending_percentage = 0.85 := by
  sorry

end NUMINAMATH_GPT_B_spends_85_percent_l60_6044


namespace NUMINAMATH_GPT_ratio_wy_l60_6015

-- Define the variables and conditions
variables (w x y z : ℚ)
def ratio_wx := w / x = 5 / 4
def ratio_yz := y / z = 7 / 5
def ratio_zx := z / x = 1 / 8

-- Statement to prove
theorem ratio_wy (hwx : ratio_wx w x) (hyz : ratio_yz y z) (hzx : ratio_zx z x) : w / y = 25 / 7 :=
by
  sorry  -- Proof not needed

end NUMINAMATH_GPT_ratio_wy_l60_6015


namespace NUMINAMATH_GPT_find_distance_l60_6063

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_distance_l60_6063


namespace NUMINAMATH_GPT_seating_arrangements_l60_6074

theorem seating_arrangements (n : ℕ) (hn : n = 8) : 
  ∃ (k : ℕ), k = 5760 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l60_6074


namespace NUMINAMATH_GPT_range_of_x_coordinate_l60_6005

theorem range_of_x_coordinate (x : ℝ) : 
  (0 ≤ 2*x + 2 ∧ 2*x + 2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ -1/2) := 
sorry

end NUMINAMATH_GPT_range_of_x_coordinate_l60_6005


namespace NUMINAMATH_GPT_december_sales_multiple_l60_6088

   noncomputable def find_sales_multiple (A : ℝ) (x : ℝ) :=
     x * A = 0.3888888888888889 * (11 * A + x * A)

   theorem december_sales_multiple (A : ℝ) (x : ℝ) (h : find_sales_multiple A x) : x = 7 :=
   by 
     sorry
   
end NUMINAMATH_GPT_december_sales_multiple_l60_6088


namespace NUMINAMATH_GPT_distance_james_rode_l60_6017

def speed : ℝ := 80.0
def time : ℝ := 16.0
def distance : ℝ := speed * time

theorem distance_james_rode :
  distance = 1280.0 :=
by
  -- to show the theorem is sane
  sorry

end NUMINAMATH_GPT_distance_james_rode_l60_6017


namespace NUMINAMATH_GPT_roots_quartic_sum_l60_6053

theorem roots_quartic_sum (p q r : ℝ) 
  (h1 : p^3 - 2*p^2 + 3*p - 4 = 0)
  (h2 : q^3 - 2*q^2 + 3*q - 4 = 0)
  (h3 : r^3 - 2*r^2 + 3*r - 4 = 0)
  (h4 : p + q + r = 2)
  (h5 : p*q + q*r + r*p = 3)
  (h6 : p*q*r = 4) :
  p^4 + q^4 + r^4 = 18 := sorry

end NUMINAMATH_GPT_roots_quartic_sum_l60_6053


namespace NUMINAMATH_GPT_decorations_given_to_friend_l60_6031

-- Definitions of the given conditions
def boxes : ℕ := 6
def decorations_per_box : ℕ := 25
def used_decorations : ℕ := 58
def neighbor_decorations : ℕ := 75

-- The statement of the proof problem
theorem decorations_given_to_friend : 
  (boxes * decorations_per_box) - used_decorations - neighbor_decorations = 17 := 
by 
  sorry

end NUMINAMATH_GPT_decorations_given_to_friend_l60_6031


namespace NUMINAMATH_GPT_john_volunteer_hours_l60_6092

noncomputable def total_volunteer_hours :=
  let first_six_months_hours := 2 * 3 * 6
  let next_five_months_hours := 1 * 2 * 4 * 5
  let december_hours := 3 * 2
  first_six_months_hours + next_five_months_hours + december_hours

theorem john_volunteer_hours : total_volunteer_hours = 82 := by
  sorry

end NUMINAMATH_GPT_john_volunteer_hours_l60_6092


namespace NUMINAMATH_GPT_value_of_2p_plus_q_l60_6050

theorem value_of_2p_plus_q (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p :=
by
  sorry

end NUMINAMATH_GPT_value_of_2p_plus_q_l60_6050


namespace NUMINAMATH_GPT_geom_seq_m_value_l60_6079

/-- Given a geometric sequence {a_n} with a1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11. -/
theorem geom_seq_m_value (q : ℝ) (h_q : q ≠ 1) :
  ∃ (m : ℕ), (m = 11) ∧ (∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n, a (n + 1) = a n * q ) ∧ (a m = a 1 * a 2 * a 3 * a 4 * a 5)) :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_m_value_l60_6079


namespace NUMINAMATH_GPT_walter_age_at_2003_l60_6046

theorem walter_age_at_2003 :
  ∀ (w : ℕ),
  (1998 - w) + (1998 - 3 * w) = 3860 → 
  w + 5 = 39 :=
by
  intros w h
  sorry

end NUMINAMATH_GPT_walter_age_at_2003_l60_6046


namespace NUMINAMATH_GPT_domain_of_g_l60_6029

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊ x^2 - 9 * x + 21 ⌋

theorem domain_of_g :
  { x : ℝ | ∃ y : ℝ, g x = y } = { x : ℝ | x ≤ 4 ∨ x ≥ 5 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_g_l60_6029


namespace NUMINAMATH_GPT_part1_part2_l60_6014

theorem part1 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (6 * Real.sin θ + Real.cos θ) / (3 * Real.sin θ - 2 * Real.cos θ) = 13 / 4 :=
sorry

theorem part2 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
sorry

end NUMINAMATH_GPT_part1_part2_l60_6014


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_length_l60_6090

theorem rhombus_longer_diagonal_length
  (side_length : ℕ) (shorter_diagonal : ℕ) 
  (side_length_eq : side_length = 53) 
  (shorter_diagonal_eq : shorter_diagonal = 50) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 94 := by
  sorry

end NUMINAMATH_GPT_rhombus_longer_diagonal_length_l60_6090


namespace NUMINAMATH_GPT_average_difference_is_7_l60_6066

/-- The differences between Mia's and Liam's study times for each day in one week -/
def daily_differences : List ℤ := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week -/
def number_of_days : ℕ := 7

/-- The total difference over the week -/
def total_difference : ℤ := daily_differences.sum

/-- The average difference per day -/
def average_difference_per_day : ℚ := total_difference / number_of_days

theorem average_difference_is_7 : average_difference_per_day = 7 := by 
  sorry

end NUMINAMATH_GPT_average_difference_is_7_l60_6066


namespace NUMINAMATH_GPT_range_of_a_l60_6039

theorem range_of_a (m : ℝ) (a : ℝ) : 
  m ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  (∀ x₁ x₂ : ℝ, x₁^2 - m * x₁ - 2 = 0 ∧ x₂^2 - m * x₂ - 2 = 0 → a^2 - 5 * a - 3 ≥ |x₁ - x₂|) ↔ (a ≥ 6 ∨ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l60_6039


namespace NUMINAMATH_GPT_total_questions_reviewed_l60_6042

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end NUMINAMATH_GPT_total_questions_reviewed_l60_6042


namespace NUMINAMATH_GPT_certain_event_proof_l60_6057

def Moonlight_in_front_of_bed := "depends_on_time_and_moon_position"
def Lonely_smoke_in_desert := "depends_on_specific_conditions"
def Reach_for_stars_with_hand := "physically_impossible"
def Yellow_River_flows_into_sea := "certain_event"

theorem certain_event_proof : Yellow_River_flows_into_sea = "certain_event" :=
by
  sorry

end NUMINAMATH_GPT_certain_event_proof_l60_6057


namespace NUMINAMATH_GPT_quadratic_ineq_solution_l60_6093

theorem quadratic_ineq_solution (x : ℝ) : x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := 
sorry

end NUMINAMATH_GPT_quadratic_ineq_solution_l60_6093


namespace NUMINAMATH_GPT_find_certain_number_l60_6095

theorem find_certain_number (x certain_number : ℕ) (h: x = 3) (h2: certain_number = 5 * x + 4) : certain_number = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l60_6095


namespace NUMINAMATH_GPT_percentage_regular_cars_l60_6011

theorem percentage_regular_cars (total_cars : ℕ) (truck_percentage : ℚ) (convertibles : ℕ) 
  (h1 : total_cars = 125) (h2 : truck_percentage = 0.08) (h3 : convertibles = 35) : 
  (80 / 125 : ℚ) * 100 = 64 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_regular_cars_l60_6011


namespace NUMINAMATH_GPT_sector_central_angle_l60_6084

theorem sector_central_angle (r θ : ℝ) 
  (h1 : 1 = (1 / 2) * 2 * r) 
  (h2 : 2 = θ * r) : θ = 2 := 
sorry

end NUMINAMATH_GPT_sector_central_angle_l60_6084


namespace NUMINAMATH_GPT_totalShortBushes_l60_6043

namespace ProofProblem

def initialShortBushes : Nat := 37
def additionalShortBushes : Nat := 20

theorem totalShortBushes :
  initialShortBushes + additionalShortBushes = 57 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_totalShortBushes_l60_6043


namespace NUMINAMATH_GPT_line_contains_point_l60_6086

theorem line_contains_point (k : ℝ) : 
  let x := (1 : ℝ) / 3
  let y := -2 
  let line_eq := (3 : ℝ) - 3 * k * x = 4 * y
  line_eq → k = 11 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_contains_point_l60_6086


namespace NUMINAMATH_GPT_gain_percent_of_articles_l60_6022

theorem gain_percent_of_articles (C S : ℝ) (h : 50 * C = 15 * S) : (S - C) / C * 100 = 233.33 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_of_articles_l60_6022


namespace NUMINAMATH_GPT_geometric_series_S_n_div_a_n_l60_6068

-- Define the conditions and the properties of the geometric sequence
variables (a_3 a_5 a_4 a_6 S_n a_n : ℝ) (n : ℕ)
variable (q : ℝ) -- common ratio of the geometric sequence

-- Conditions given in the problem
axiom h1 : a_3 + a_5 = 5 / 4
axiom h2 : a_4 + a_6 = 5 / 8

-- The value we want to prove
theorem geometric_series_S_n_div_a_n : 
  (a_3 + a_5) * q = 5 / 8 → 
  q = 1 / 2 → 
  S_n = a_n * (2^n - 1) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_geometric_series_S_n_div_a_n_l60_6068


namespace NUMINAMATH_GPT_angle_B_lt_pi_div_two_l60_6008

theorem angle_B_lt_pi_div_two 
  (a b c : ℝ) (B : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : B = π / 2 - B)
  (h5 : 2 / b = 1 / a + 1 / c)
  : B < π / 2 := sorry

end NUMINAMATH_GPT_angle_B_lt_pi_div_two_l60_6008


namespace NUMINAMATH_GPT_correct_condition_l60_6038

section proof_problem

variable (a : ℝ)

def cond1 : Prop := (a ^ 6 / a ^ 3 = a ^ 2)
def cond2 : Prop := (2 * a ^ 2 + 3 * a ^ 3 = 5 * a ^ 5)
def cond3 : Prop := (a ^ 4 * a ^ 2 = a ^ 8)
def cond4 : Prop := ((-a ^ 3) ^ 2 = a ^ 6)

theorem correct_condition : cond4 a :=
by
  sorry

end proof_problem

end NUMINAMATH_GPT_correct_condition_l60_6038


namespace NUMINAMATH_GPT_line_through_point_bisects_chord_l60_6023

theorem line_through_point_bisects_chord 
  (x y : ℝ) 
  (h_parabola : y^2 = 16 * x) 
  (h_point : 8 * 2 - 1 - 15 = 0) :
  8 * x - y - 15 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_bisects_chord_l60_6023


namespace NUMINAMATH_GPT_total_molecular_weight_l60_6030

-- Define atomic weights
def atomic_weight (element : String) : Float :=
  match element with
  | "K"  => 39.10
  | "Cr" => 51.996
  | "O"  => 16.00
  | "Fe" => 55.845
  | "S"  => 32.07
  | "Mn" => 54.938
  | _    => 0.0

-- Molecular weights of compounds
def molecular_weight_K2Cr2O7 : Float := 
  2 * atomic_weight "K" + 2 * atomic_weight "Cr" + 7 * atomic_weight "O"

def molecular_weight_Fe2_SO4_3 : Float := 
  2 * atomic_weight "Fe" + 3 * atomic_weight "S" + 12 * atomic_weight "O"

def molecular_weight_KMnO4 : Float := 
  atomic_weight "K" + atomic_weight "Mn" + 4 * atomic_weight "O"

-- Proof statement 
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 + 3 * molecular_weight_Fe2_SO4_3 + 5 * molecular_weight_KMnO4 = 3166.658 :=
by
  sorry

end NUMINAMATH_GPT_total_molecular_weight_l60_6030


namespace NUMINAMATH_GPT_number_of_green_hats_l60_6045

theorem number_of_green_hats 
  (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) 
  : G = 40 :=
sorry

end NUMINAMATH_GPT_number_of_green_hats_l60_6045


namespace NUMINAMATH_GPT_negation_of_universal_l60_6032

-- Definitions based on the provided problem
def prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Main proof problem statement
theorem negation_of_universal : 
  ¬ (∀ x : ℝ, x > 0 → x^2 > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_l60_6032


namespace NUMINAMATH_GPT_count_multiples_3_or_4_but_not_6_l60_6059

def multiples_between (m n k : Nat) : Nat :=
  (k / m) + (k / n) - (k / (m * n))

theorem count_multiples_3_or_4_but_not_6 :
  let count_multiples (d : Nat) := (3000 / d)
  let multiples_of_3 := count_multiples 3
  let multiples_of_4 := count_multiples 4
  let multiples_of_6 := count_multiples 6
  multiples_of_3 + multiples_of_4 - multiples_of_6 = 1250 := by
  sorry

end NUMINAMATH_GPT_count_multiples_3_or_4_but_not_6_l60_6059


namespace NUMINAMATH_GPT_ratio_of_pieces_l60_6097

-- Define the total length of the wire.
def total_length : ℕ := 14

-- Define the length of the shorter piece.
def shorter_piece_length : ℕ := 4

-- Define the length of the longer piece.
def longer_piece_length : ℕ := total_length - shorter_piece_length

-- Define the expected ratio of the lengths.
def ratio : ℚ := shorter_piece_length / longer_piece_length

-- State the theorem to prove.
theorem ratio_of_pieces : ratio = 2 / 5 := 
by {
  -- skip the proof
  sorry
}

end NUMINAMATH_GPT_ratio_of_pieces_l60_6097


namespace NUMINAMATH_GPT_range_of_a_l60_6037

theorem range_of_a 
{α : Type*} [LinearOrderedField α] (a : α) 
(h : ∃ x, x = 3 ∧ (x - a) * (x + 2 * a - 1) ^ 2 * (x - 3 * a) ≤ 0) :
a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l60_6037


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l60_6061

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h₁ : ∀ n k : ℕ, a (n + k) = a n + k * d) 
  (h₂ : a 5 + a 6 + a 7 + a 8 = 20) : a 1 + a 12 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l60_6061


namespace NUMINAMATH_GPT_points_on_fourth_board_l60_6027

theorem points_on_fourth_board (P_1 P_2 P_3 P_4 : ℕ)
 (h1 : P_1 = 30)
 (h2 : P_2 = 38)
 (h3 : P_3 = 41) :
  P_4 = 34 :=
sorry

end NUMINAMATH_GPT_points_on_fourth_board_l60_6027


namespace NUMINAMATH_GPT_find_special_three_digit_numbers_l60_6004

theorem find_special_three_digit_numbers :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 
  (100 * a + 10 * b + (c + 3)) % 10 + (100 * a + 10 * (b + 1) + c).div 10 % 10 + (100 * (a + 1) + 10 * b + c).div 100 % 10 + 3 = 
  (a + b + c) / 3)} → n = 117 ∨ n = 207 ∨ n = 108 :=
by
  sorry

end NUMINAMATH_GPT_find_special_three_digit_numbers_l60_6004


namespace NUMINAMATH_GPT_octagon_perimeter_l60_6020

/-- 
  Represents the side length of the regular octagon
-/
def side_length : ℕ := 12

/-- 
  Represents the number of sides of a regular octagon
-/
def number_of_sides : ℕ := 8

/-- 
  Defines the perimeter of the regular octagon
-/
def perimeter (side_length : ℕ) (number_of_sides : ℕ) : ℕ :=
  side_length * number_of_sides

/-- 
  Proof statement: asserting that the perimeter of a regular octagon
  with a side length of 12 meters is 96 meters
-/
theorem octagon_perimeter :
  perimeter side_length number_of_sides = 96 :=
  sorry

end NUMINAMATH_GPT_octagon_perimeter_l60_6020


namespace NUMINAMATH_GPT_sum_first_100_odd_l60_6060

theorem sum_first_100_odd :
  (Finset.sum (Finset.range 100) (λ x => 2 * (x + 1) - 1)) = 10000 := by
  sorry

end NUMINAMATH_GPT_sum_first_100_odd_l60_6060


namespace NUMINAMATH_GPT_value_of_y_l60_6000

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l60_6000


namespace NUMINAMATH_GPT_angle_CAD_l60_6016

noncomputable def angle_arc (degree: ℝ) (minute: ℝ) : ℝ :=
  degree + minute / 60

theorem angle_CAD :
  angle_arc 117 23 / 2 + angle_arc 42 37 / 2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_angle_CAD_l60_6016


namespace NUMINAMATH_GPT_jen_visits_exactly_two_countries_l60_6082

noncomputable def probability_of_visiting_exactly_two_countries (p_chile p_madagascar p_japan p_egypt : ℝ) : ℝ :=
  let p_chile_madagascar := (p_chile * p_madagascar) * (1 - p_japan) * (1 - p_egypt)
  let p_chile_japan := (p_chile * p_japan) * (1 - p_madagascar) * (1 - p_egypt)
  let p_chile_egypt := (p_chile * p_egypt) * (1 - p_madagascar) * (1 - p_japan)
  let p_madagascar_japan := (p_madagascar * p_japan) * (1 - p_chile) * (1 - p_egypt)
  let p_madagascar_egypt := (p_madagascar * p_egypt) * (1 - p_chile) * (1 - p_japan)
  let p_japan_egypt := (p_japan * p_egypt) * (1 - p_chile) * (1 - p_madagascar)
  p_chile_madagascar + p_chile_japan + p_chile_egypt + p_madagascar_japan + p_madagascar_egypt + p_japan_egypt

theorem jen_visits_exactly_two_countries :
  probability_of_visiting_exactly_two_countries 0.4 0.35 0.2 0.15 = 0.2432 :=
by
  sorry

end NUMINAMATH_GPT_jen_visits_exactly_two_countries_l60_6082


namespace NUMINAMATH_GPT_value_of_x3_plus_inv_x3_l60_6001

theorem value_of_x3_plus_inv_x3 (x : ℝ) (h : 728 = x^6 + 1 / x^6) : 
  x^3 + 1 / x^3 = Real.sqrt 730 :=
sorry

end NUMINAMATH_GPT_value_of_x3_plus_inv_x3_l60_6001


namespace NUMINAMATH_GPT_fraction_problem_l60_6075

theorem fraction_problem :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end NUMINAMATH_GPT_fraction_problem_l60_6075


namespace NUMINAMATH_GPT_find_m_l60_6047

theorem find_m (m : ℝ) (h1 : ∀ x y : ℝ, (x ^ 2 + (y - 2) ^ 2 = 1) → (y = x / m ∨ y = -x / m)) (h2 : 0 < m) :
  m = (Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l60_6047


namespace NUMINAMATH_GPT_minimum_a_l60_6009

theorem minimum_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - (x - a) * |x - a| - 2 ≥ 0) → a ≥ Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_a_l60_6009


namespace NUMINAMATH_GPT_unique_solution_l60_6010

noncomputable def unique_solution_exists : Prop :=
  ∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    (a + b = (c + d + e) / 7) ∧
    (a + d = (b + c + e) / 5) ∧
    (a + b + c + d + e = 24) ∧
    (a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 3 ∧ e = 9)

theorem unique_solution : unique_solution_exists :=
sorry

end NUMINAMATH_GPT_unique_solution_l60_6010


namespace NUMINAMATH_GPT_solutions_shifted_quadratic_l60_6006

theorem solutions_shifted_quadratic (a h k : ℝ) (x1 x2: ℝ)
  (h1 : a * (-1 - h)^2 + k = 0)
  (h2 : a * (3 - h)^2 + k = 0) :
  a * (0 - (h + 1))^2 + k = 0 ∧ a * (4 - (h + 1))^2 + k = 0 :=
by
  sorry

end NUMINAMATH_GPT_solutions_shifted_quadratic_l60_6006


namespace NUMINAMATH_GPT_remainder_of_349_by_17_is_9_l60_6040

theorem remainder_of_349_by_17_is_9 :
  349 % 17 = 9 :=
sorry

end NUMINAMATH_GPT_remainder_of_349_by_17_is_9_l60_6040


namespace NUMINAMATH_GPT_range_of_a_l60_6096

theorem range_of_a (a : ℝ) (e : ℝ) (x : ℝ) (ln : ℝ → ℝ) :
  (∀ x, (1 / e) ≤ x ∧ x ≤ e → (a - x^2 = -2 * ln x)) →
  (1 ≤ a ∧ a ≤ (e^2 - 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l60_6096


namespace NUMINAMATH_GPT_son_age_is_26_l60_6054

-- Definitions based on conditions in the problem
variables (S F : ℕ)
axiom cond1 : F = S + 28
axiom cond2 : F + 2 = 2 * (S + 2)

-- Statement to prove that S = 26
theorem son_age_is_26 : S = 26 :=
by 
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_son_age_is_26_l60_6054


namespace NUMINAMATH_GPT_round_310242_to_nearest_thousand_l60_6071

-- Define the conditions and the target statement
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  if (n % 1000) < 500 then (n / 1000) * 1000 else (n / 1000 + 1) * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 :=
by
  sorry

end NUMINAMATH_GPT_round_310242_to_nearest_thousand_l60_6071


namespace NUMINAMATH_GPT_hyperbola_asymptote_slope_l60_6052

theorem hyperbola_asymptote_slope :
  ∀ {x y : ℝ}, (x^2 / 144 - y^2 / 81 = 1) → (∃ m : ℝ, ∀ x, y = m * x ∨ y = -m * x ∧ m = 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_slope_l60_6052


namespace NUMINAMATH_GPT_avg_books_rounded_l60_6025

def books_read : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4)]

noncomputable def total_books_read (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.fst * pair.snd) 0

noncomputable def total_members (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.snd) 0

noncomputable def average_books_read (books : List (ℕ × ℕ)) : ℤ :=
  Int.ofNat (total_books_read books) / Int.ofNat (total_members books)

theorem avg_books_rounded :
  average_books_read books_read = 3 :=
by 
  sorry

end NUMINAMATH_GPT_avg_books_rounded_l60_6025


namespace NUMINAMATH_GPT_find_dividend_l60_6081

theorem find_dividend (R D Q V : ℤ) (hR : R = 5) (hD1 : D = 3 * Q) (hD2 : D = 3 * R + 3) : V = D * Q + R → V = 113 :=
by 
  sorry

end NUMINAMATH_GPT_find_dividend_l60_6081


namespace NUMINAMATH_GPT_bob_needs_8_additional_wins_to_afford_puppy_l60_6098

variable (n : ℕ) (grand_prize_per_win : ℝ) (total_cost : ℝ)

def bob_total_wins_to_afford_puppy : Prop :=
  total_cost = 1000 ∧ grand_prize_per_win = 100 ∧ n = (total_cost / grand_prize_per_win) - 2

theorem bob_needs_8_additional_wins_to_afford_puppy :
  bob_total_wins_to_afford_puppy 8 100 1000 :=
by {
  sorry
}

end NUMINAMATH_GPT_bob_needs_8_additional_wins_to_afford_puppy_l60_6098


namespace NUMINAMATH_GPT_final_cost_is_correct_l60_6048

noncomputable def calculate_final_cost 
  (price_orange : ℕ)
  (price_mango : ℕ)
  (increase_percent : ℕ)
  (bulk_discount_percent : ℕ)
  (sales_tax_percent : ℕ) : ℕ := 
  let new_price_orange := price_orange + (price_orange * increase_percent) / 100
  let new_price_mango := price_mango + (price_mango * increase_percent) / 100
  let total_cost_oranges := 10 * new_price_orange
  let total_cost_mangoes := 10 * new_price_mango
  let total_cost_before_discount := total_cost_oranges + total_cost_mangoes
  let discount_oranges := (total_cost_oranges * bulk_discount_percent) / 100
  let discount_mangoes := (total_cost_mangoes * bulk_discount_percent) / 100
  let total_cost_after_discount := total_cost_before_discount - discount_oranges - discount_mangoes
  let sales_tax := (total_cost_after_discount * sales_tax_percent) / 100
  total_cost_after_discount + sales_tax

theorem final_cost_is_correct :
  calculate_final_cost 40 50 15 10 8 = 100602 :=
by
  sorry

end NUMINAMATH_GPT_final_cost_is_correct_l60_6048


namespace NUMINAMATH_GPT_zongzi_problem_l60_6033

def zongzi_prices : Prop :=
  ∀ (x y : ℕ), -- x: price of red bean zongzi, y: price of meat zongzi
  10 * x + 12 * y = 136 → -- total cost for the first customer
  y = 2 * x →
  x = 4 ∧ y = 8 -- prices found

def discounted_zongzi_prices : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  20 * a + 30 * b = 270 → -- cost for Xiaohuan's mother
  30 * a + 20 * b = 230 → -- cost for Xiaole's mother
  a = 3 ∧ b = 7 -- discounted prices found

def zongzi_packages (m : ℕ) : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  a = 3 → b = 7 →
  (80 - 4 * m) * (m * a + (40 - m) * b) + (4 * m + 8) * ((40 - m) * a + m * b) = 17280 →
  m ≤ 20 / 2 → -- quantity constraint
  m = 10 -- final m value

-- Statement to prove all together
theorem zongzi_problem :
  zongzi_prices ∧ discounted_zongzi_prices ∧ ∃ (m : ℕ), zongzi_packages m :=
by sorry

end NUMINAMATH_GPT_zongzi_problem_l60_6033


namespace NUMINAMATH_GPT_solve_diophantine_l60_6051

theorem solve_diophantine :
  {xy : ℤ × ℤ | 5 * (xy.1 ^ 2) + 5 * xy.1 * xy.2 + 5 * (xy.2 ^ 2) = 7 * xy.1 + 14 * xy.2} = {(-1, 3), (0, 0), (1, 2)} :=
by sorry

end NUMINAMATH_GPT_solve_diophantine_l60_6051


namespace NUMINAMATH_GPT_parabola_line_intersection_l60_6080

theorem parabola_line_intersection (p : ℝ) (hp : p > 0) 
  (line_eq : ∃ b : ℝ, ∀ x : ℝ, 2 * x + b = 2 * x - p/2) 
  (focus := (p / 4, 0))
  (point_A := (0, -p / 2))
  (area_OAF : 1 / 2 * (p / 4) * (p / 2) = 1) : 
  p = 4 :=
sorry

end NUMINAMATH_GPT_parabola_line_intersection_l60_6080


namespace NUMINAMATH_GPT_find_divisor_l60_6036

theorem find_divisor (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l60_6036


namespace NUMINAMATH_GPT_percentage_reduction_l60_6028

theorem percentage_reduction 
  (original_employees : ℝ)
  (new_employees : ℝ)
  (h1 : original_employees = 208.04597701149424)
  (h2 : new_employees = 181) :
  ((original_employees - new_employees) / original_employees) * 100 = 13.00 :=
by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l60_6028


namespace NUMINAMATH_GPT_intersection_complement_eq_C_l60_6076

def A := { x : ℝ | -3 < x ∧ x < 6 }
def B := { x : ℝ | 2 < x ∧ x < 7 }
def complement_B := { x : ℝ | x ≤ 2 ∨ x ≥ 7 }
def C := { x : ℝ | -3 < x ∧ x ≤ 2 }

theorem intersection_complement_eq_C :
  A ∩ complement_B = C :=
sorry

end NUMINAMATH_GPT_intersection_complement_eq_C_l60_6076


namespace NUMINAMATH_GPT_white_area_of_sign_remains_l60_6094

theorem white_area_of_sign_remains (h1 : (6 * 18 = 108))
  (h2 : 9 = 6 + 3)
  (h3 : 7.5 = 5 + 3 - 0.5)
  (h4 : 13 = 9 + 4)
  (h5 : 9 = 6 + 3)
  (h6 : 38.5 = 9 + 7.5 + 13 + 9)
  : 108 - 38.5 = 69.5 := by
  sorry

end NUMINAMATH_GPT_white_area_of_sign_remains_l60_6094


namespace NUMINAMATH_GPT_sum_three_distinct_zero_l60_6035

variable {R : Type} [Field R]

theorem sum_three_distinct_zero
  (a b c x y : R)
  (h1 : a ^ 3 + a * x + y = 0)
  (h2 : b ^ 3 + b * x + y = 0)
  (h3 : c ^ 3 + c * x + y = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_GPT_sum_three_distinct_zero_l60_6035


namespace NUMINAMATH_GPT_quadratic_function_expression_l60_6034

-- Definitions based on conditions
def quadratic (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
def condition1 (f : ℝ → ℝ) : Prop := (f 0 = 1)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) - f x = 4 * x

-- The theorem we want to prove
theorem quadratic_function_expression (f : ℝ → ℝ) 
  (hf_quad : quadratic f)
  (hf_cond1 : condition1 f)
  (hf_cond2 : condition2 f) : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -2 ∧ c = 1 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end NUMINAMATH_GPT_quadratic_function_expression_l60_6034


namespace NUMINAMATH_GPT_min_value_expression_l60_6069

theorem min_value_expression : ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_min_value_expression_l60_6069


namespace NUMINAMATH_GPT_probability_correct_l60_6064

def total_chips : ℕ := 15
def total_ways_to_draw_2_chips : ℕ := Nat.choose 15 2

def chips_same_color : ℕ := 3 * (Nat.choose 5 2)
def chips_same_number : ℕ := 5 * (Nat.choose 3 2)
def favorable_outcomes : ℕ := chips_same_color + chips_same_number

def probability_same_color_or_number : ℚ := favorable_outcomes / total_ways_to_draw_2_chips

theorem probability_correct :
  probability_same_color_or_number = 3 / 7 :=
by sorry

end NUMINAMATH_GPT_probability_correct_l60_6064


namespace NUMINAMATH_GPT_teachers_like_at_least_one_l60_6087

theorem teachers_like_at_least_one (T C B N: ℕ) 
    (total_teachers : T + C + N = 90)  -- Total number of teachers plus neither equals 90
    (tea_teachers : T = 66)           -- Teachers who like tea is 66
    (coffee_teachers : C = 42)        -- Teachers who like coffee is 42
    (both_beverages : B = 3 * N)      -- Teachers who like both is three times neither
    : T + C - B = 81 :=               -- Teachers who like at least one beverage
by 
  sorry

end NUMINAMATH_GPT_teachers_like_at_least_one_l60_6087


namespace NUMINAMATH_GPT_domain_of_g_l60_6058

def f (x : ℝ) : Prop := x ∈ Set.Icc (-12.0) 6.0

def g (x : ℝ) : Prop := f (3 * x)

theorem domain_of_g : Set.Icc (-4.0) 2.0 = {x : ℝ | g x} := 
by 
    sorry

end NUMINAMATH_GPT_domain_of_g_l60_6058


namespace NUMINAMATH_GPT_range_of_a_l60_6062

theorem range_of_a (a : ℝ) (h1 : 0 < a) :
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 - x - 6 ≤ 0) ∧
  (¬ (∀ x : ℝ, x^2 - x - 6 ≤ 0 → x^2 - 4*a*x + 3*a^2 ≤ 0)) →
  0 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l60_6062


namespace NUMINAMATH_GPT_mean_of_roots_l60_6013

theorem mean_of_roots
  (a b c d k : ℤ)
  (p : ℤ → ℤ)
  (h_poly : ∀ x, p x = (x - a) * (x - b) * (x - c) * (x - d))
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : p k = 4) :
  k = (a + b + c + d) / 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_mean_of_roots_l60_6013


namespace NUMINAMATH_GPT_find_divisor_l60_6002

theorem find_divisor (D Q R d : ℕ) (h1 : D = 159) (h2 : Q = 9) (h3 : R = 6) (h4 : D = d * Q + R) : d = 17 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l60_6002


namespace NUMINAMATH_GPT_nancy_hours_to_work_l60_6077

def tuition := 22000
def scholarship := 3000
def hourly_wage := 10
def parents_contribution := tuition / 2
def student_loan := 2 * scholarship
def total_financial_aid := scholarship + student_loan
def remaining_tuition := tuition - parents_contribution - total_financial_aid
def hours_to_work := remaining_tuition / hourly_wage

theorem nancy_hours_to_work : hours_to_work = 200 := by
  -- This by block demonstrates that a proof would go here
  sorry

end NUMINAMATH_GPT_nancy_hours_to_work_l60_6077


namespace NUMINAMATH_GPT_complete_square_quadratic_t_l60_6019

theorem complete_square_quadratic_t : 
  ∀ x : ℝ, (16 * x^2 - 32 * x - 512 = 0) → (∃ q t : ℝ, (x + q)^2 = t ∧ t = 33) :=
by sorry

end NUMINAMATH_GPT_complete_square_quadratic_t_l60_6019


namespace NUMINAMATH_GPT_union_complement_U_A_B_l60_6070

def U : Set Int := {-1, 0, 1, 2, 3}

def A : Set Int := {-1, 0, 1}

def B : Set Int := {0, 1, 2}

def complement_U_A : Set Int := {u | u ∈ U ∧ u ∉ A}

theorem union_complement_U_A_B : (complement_U_A ∪ B) = {0, 1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_union_complement_U_A_B_l60_6070


namespace NUMINAMATH_GPT_find_x_complementary_l60_6024

-- Define the conditions.
def are_complementary (a b : ℝ) : Prop := a + b = 90

-- The main theorem statement with the condition and conclusion.
theorem find_x_complementary : ∀ x : ℝ, are_complementary (2*x) (3*x) → x = 18 := 
by
  intros x h
  -- sorry is a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_find_x_complementary_l60_6024


namespace NUMINAMATH_GPT_jim_saves_money_by_buying_gallon_l60_6091

theorem jim_saves_money_by_buying_gallon :
  let gallon_price := 8
  let bottle_price := 3
  let ounces_per_gallon := 128
  let ounces_per_bottle := 16
  (ounces_per_gallon / ounces_per_bottle) * bottle_price - gallon_price = 16 :=
by
  sorry

end NUMINAMATH_GPT_jim_saves_money_by_buying_gallon_l60_6091


namespace NUMINAMATH_GPT_investment_initial_amount_l60_6072

noncomputable def initialInvestment (final_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  final_amount / interest_rate^years

theorem investment_initial_amount :
  initialInvestment 705.73 1.12 5 = 400.52 := by
  sorry

end NUMINAMATH_GPT_investment_initial_amount_l60_6072


namespace NUMINAMATH_GPT_remainder_div_14_l60_6099

def S : ℕ := 11065 + 11067 + 11069 + 11071 + 11073 + 11075 + 11077

theorem remainder_div_14 : S % 14 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_14_l60_6099


namespace NUMINAMATH_GPT_card_collection_problem_l60_6041

theorem card_collection_problem 
  (m : ℕ) 
  (h : (2 * m + 1) / 3 = 56) : 
  m = 84 :=
sorry

end NUMINAMATH_GPT_card_collection_problem_l60_6041


namespace NUMINAMATH_GPT_xy_identity_l60_6085

theorem xy_identity (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) : (x^2 + y^2) * (x + y) = 803 := by
  sorry

end NUMINAMATH_GPT_xy_identity_l60_6085


namespace NUMINAMATH_GPT_perimeter_of_regular_pentagon_is_75_l60_6089

-- Define the side length and the property of the figure
def side_length : ℝ := 15
def is_regular_pentagon : Prop := true  -- assuming this captures the regular pentagon property

-- Define the perimeter calculation based on the conditions
def perimeter (n : ℕ) (side_length : ℝ) := n * side_length

-- The theorem to prove
theorem perimeter_of_regular_pentagon_is_75 :
  is_regular_pentagon → perimeter 5 side_length = 75 :=
by
  intro _ -- We don't need to use is_regular_pentagon directly
  rw [side_length]
  norm_num
  sorry

end NUMINAMATH_GPT_perimeter_of_regular_pentagon_is_75_l60_6089


namespace NUMINAMATH_GPT_integers_within_range_l60_6003

def is_within_range (n : ℤ) : Prop :=
  (-1.3 : ℝ) < (n : ℝ) ∧ (n : ℝ) < 2.8

theorem integers_within_range :
  { n : ℤ | is_within_range n } = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_integers_within_range_l60_6003


namespace NUMINAMATH_GPT_find_k_for_one_real_solution_l60_6049

theorem find_k_for_one_real_solution (k : ℤ) :
  (∀ x : ℤ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end NUMINAMATH_GPT_find_k_for_one_real_solution_l60_6049


namespace NUMINAMATH_GPT_problem_1_split_terms_problem_2_split_terms_l60_6056

-- Problem 1 Lean statement
theorem problem_1_split_terms :
  (28 + 5/7) + (-25 - 1/7) = 3 + 4/7 := 
  sorry
  
-- Problem 2 Lean statement
theorem problem_2_split_terms :
  (-2022 - 2/7) + (-2023 - 4/7) + 4046 - 1/7 = 0 := 
  sorry

end NUMINAMATH_GPT_problem_1_split_terms_problem_2_split_terms_l60_6056


namespace NUMINAMATH_GPT_find_A_from_equation_l60_6055

variable (A B C D : ℕ)
variable (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (eq1 : A * 1000 + B * 100 + 82 - 900 + C * 10 + 9 = 4000 + 900 + 30 + D)

theorem find_A_from_equation (A B C D : ℕ) (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq1 : A * 1000 + B * 100 + 82 - (900 + C * 10 + 9) = 4000 + 900 + 30 + D) : A = 5 :=
by sorry

end NUMINAMATH_GPT_find_A_from_equation_l60_6055


namespace NUMINAMATH_GPT_max_gcd_seq_l60_6021

theorem max_gcd_seq (a : ℕ → ℕ) (d : ℕ → ℕ) :
  (∀ n : ℕ, a n = 121 + n^2) →
  (∀ n : ℕ, d n = Nat.gcd (a n) (a (n + 1))) →
  ∃ m : ℕ, ∀ n : ℕ, d n ≤ d m ∧ d m = 99 :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_seq_l60_6021


namespace NUMINAMATH_GPT_find_int_k_l60_6018

theorem find_int_k (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^3) :
  K = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_int_k_l60_6018


namespace NUMINAMATH_GPT_least_number_to_subtract_l60_6026

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (k : ℕ) (hk : 42398 % 15 = k) : k = 8 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l60_6026


namespace NUMINAMATH_GPT_abs_lt_five_implies_interval_l60_6067

theorem abs_lt_five_implies_interval (x : ℝ) : |x| < 5 → -5 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_GPT_abs_lt_five_implies_interval_l60_6067


namespace NUMINAMATH_GPT_tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l60_6007

structure Tetrahedron :=
  (faces : Nat := 4)
  (vertices : Nat := 4)
  (valence : Nat := 3)
  (face_shape : String := "triangular")

structure Cube :=
  (faces : Nat := 6)
  (vertices : Nat := 8)
  (valence : Nat := 3)
  (face_shape : String := "square")

structure Octahedron :=
  (faces : Nat := 8)
  (vertices : Nat := 6)
  (valence : Nat := 4)
  (face_shape : String := "triangular")

structure Dodecahedron :=
  (faces : Nat := 12)
  (vertices : Nat := 20)
  (valence : Nat := 3)
  (face_shape : String := "pentagonal")

structure Icosahedron :=
  (faces : Nat := 20)
  (vertices : Nat := 12)
  (valence : Nat := 5)
  (face_shape : String := "triangular")

theorem tetrahedron_is_self_dual:
  Tetrahedron := by
  sorry

theorem cube_is_dual_to_octahedron:
  Cube × Octahedron := by
  sorry

theorem dodecahedron_is_dual_to_icosahedron:
  Dodecahedron × Icosahedron := by
  sorry

end NUMINAMATH_GPT_tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l60_6007


namespace NUMINAMATH_GPT_harris_flour_amount_l60_6073

noncomputable def flour_needed_by_cakes (cakes : ℕ) : ℕ := cakes * 100

noncomputable def traci_flour : ℕ := 500

noncomputable def total_cakes : ℕ := 9

theorem harris_flour_amount : flour_needed_by_cakes total_cakes - traci_flour = 400 := 
by
  sorry

end NUMINAMATH_GPT_harris_flour_amount_l60_6073


namespace NUMINAMATH_GPT_carpet_needed_for_room_l60_6078

theorem carpet_needed_for_room
  (length_feet : ℕ) (width_feet : ℕ)
  (area_conversion_factor : ℕ)
  (length_given : length_feet = 12)
  (width_given : width_feet = 6)
  (conversion_given : area_conversion_factor = 9) :
  (length_feet * width_feet) / area_conversion_factor = 8 := 
by
  sorry

end NUMINAMATH_GPT_carpet_needed_for_room_l60_6078
