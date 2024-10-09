import Mathlib

namespace ratio_AB_AD_l1790_179016

theorem ratio_AB_AD (a x y : ℝ) (h1 : 0.3 * a^2 = 0.7 * x * y) (h2 : y = a / 10) : x / y = 43 :=
by
  sorry

end ratio_AB_AD_l1790_179016


namespace pints_of_cider_l1790_179043

def pintCider (g : ℕ) (p : ℕ) : ℕ :=
  g / 20 + p / 40

def totalApples (f : ℕ) (h : ℕ) (a : ℕ) : ℕ :=
  f * h * a

theorem pints_of_cider (g p : ℕ) (farmhands : ℕ) (hours : ℕ) (apples_per_hour : ℕ)
  (H1 : g = 1)
  (H2 : p = 2)
  (H3 : farmhands = 6)
  (H4 : hours = 5)
  (H5 : apples_per_hour = 240) :
  pintCider (apples_per_hour * farmhands * hours / 3)
            (apples_per_hour * farmhands * hours * 2 / 3) = 120 :=
by
  sorry

end pints_of_cider_l1790_179043


namespace neg_p_l1790_179071

open Set

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A : Set ℤ := {x | is_odd x}
def B : Set ℤ := {x | is_even x}

-- Proposition p
def p : Prop := ∀ x ∈ A, 2 * x ∈ B

-- Negation of the proposition p
theorem neg_p : ¬p ↔ ∃ x ∈ A, ¬(2 * x ∈ B) := sorry

end neg_p_l1790_179071


namespace simplify_expression_l1790_179096

-- Define the main theorem
theorem simplify_expression 
  (a b x : ℝ) 
  (hx : x = 1 / a * Real.sqrt ((2 * a - b) / b))
  (hc1 : 0 < b / 2)
  (hc2 : b / 2 < a)
  (hc3 : a < b) : 
  (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1 :=
sorry

end simplify_expression_l1790_179096


namespace algebraic_expression_value_l1790_179055

theorem algebraic_expression_value (x : ℝ) (hx : x = Real.sqrt 7 + 1) :
  (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3)) = Real.sqrt 7 - 1 :=
by
  sorry

end algebraic_expression_value_l1790_179055


namespace monster_perimeter_correct_l1790_179056

noncomputable def monster_perimeter (radius : ℝ) (central_angle_missing : ℝ) : ℝ :=
  let full_circle_circumference := 2 * radius * Real.pi
  let arc_length := (1 - central_angle_missing / 360) * full_circle_circumference
  arc_length + 2 * radius

theorem monster_perimeter_correct :
  monster_perimeter 2 90 = 3 * Real.pi + 4 :=
by
  -- The proof would go here
  sorry

end monster_perimeter_correct_l1790_179056


namespace ticket_savings_percentage_l1790_179092

theorem ticket_savings_percentage:
  ∀ (P : ℝ), 9 * P - 6 * P = (1 / 3) * (9 * P) ∧ (33 + 1/3) = 100 * (3 * P / (9 * P)) := 
by
  intros P
  sorry

end ticket_savings_percentage_l1790_179092


namespace arithmetic_seq_fifth_term_l1790_179058

theorem arithmetic_seq_fifth_term (x y : ℝ) 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 2 * x^2 + 3 * y^2) 
  (h2 : a2 = x^2 + 2 * y^2) 
  (h3 : a3 = 2 * x^2 - y^2) 
  (h4 : a4 = x^2 - y^2) 
  (d : ℝ) 
  (hd : d = -x^2 - y^2) 
  (h_arith: ∀ i j k : ℕ, i < j ∧ j < k → a2 - a1 = d ∧ a3 - a2 = d ∧ a4 - a3 = d) : 
  a4 + d = -2 * y^2 := 
by 
  sorry

end arithmetic_seq_fifth_term_l1790_179058


namespace problem_statement_l1790_179077

theorem problem_statement (n m N k : ℕ)
  (h : (n^2 + 1)^(2^k) * (44 * n^3 + 11 * n^2 + 10 * n + 2) = N^m) :
  m = 1 :=
sorry

end problem_statement_l1790_179077


namespace john_bought_soap_l1790_179087

theorem john_bought_soap (weight_per_bar : ℝ) (cost_per_pound : ℝ) (total_spent : ℝ) (h1 : weight_per_bar = 1.5) (h2 : cost_per_pound = 0.5) (h3 : total_spent = 15) : 
  total_spent / (weight_per_bar * cost_per_pound) = 20 :=
by
  -- The proof would go here
  sorry

end john_bought_soap_l1790_179087


namespace minimum_value_sine_shift_l1790_179010

theorem minimum_value_sine_shift :
  ∀ (f : ℝ → ℝ) (φ : ℝ), (∀ x, f x = Real.sin (2 * x + φ)) → |φ| < Real.pi / 2 →
  (∀ x, f (x + Real.pi / 6) = f (-x)) →
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = - Real.sqrt 3 / 2 :=
by
  sorry

end minimum_value_sine_shift_l1790_179010


namespace selling_price_decreased_l1790_179007

theorem selling_price_decreased (d m : ℝ) (hd : d = 0.10) (hm : m = 0.10) :
  (1 - d) * (1 + m) < 1 :=
by
  rw [hd, hm]
  sorry

end selling_price_decreased_l1790_179007


namespace coin_flip_sequences_l1790_179008

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l1790_179008


namespace hyperbolic_identity_l1790_179053

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem hyperbolic_identity (x : ℝ) : (ch x) ^ 2 - (sh x) ^ 2 = 1 := 
sorry

end hyperbolic_identity_l1790_179053


namespace weight_of_b_l1790_179001

-- Definitions based on conditions
variables (A B C : ℝ)

def avg_abc := (A + B + C) / 3 = 45
def avg_ab := (A + B) / 2 = 40
def avg_bc := (B + C) / 2 = 44

-- The theorem to prove
theorem weight_of_b (h1 : avg_abc A B C) (h2 : avg_ab A B) (h3 : avg_bc B C) :
  B = 33 :=
sorry

end weight_of_b_l1790_179001


namespace gold_hammer_weight_l1790_179048

theorem gold_hammer_weight (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_a1 : a 1 = 4) 
  (h_a5 : a 5 = 2) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := 
sorry

end gold_hammer_weight_l1790_179048


namespace percentage_difference_l1790_179036

theorem percentage_difference : 0.70 * 100 - 0.60 * 80 = 22 := 
by
  sorry

end percentage_difference_l1790_179036


namespace sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l1790_179021

-- Prove that the square root of 36 equals ±6
theorem sqrt_36_eq_pm6 : ∃ y : ℤ, y * y = 36 ∧ y = 6 ∨ y = -6 :=
by
  sorry

-- Prove that the arithmetic square root of sqrt(16) equals 2
theorem arith_sqrt_sqrt_16_eq_2 : ∃ z : ℝ, z * z = 16 ∧ z = 4 ∧ 2 * 2 = z :=
by
  sorry

-- Prove that the cube root of -27 equals -3
theorem cube_root_minus_27_eq_minus_3 : ∃ x : ℝ, x * x * x = -27 ∧ x = -3 :=
by
  sorry

end sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l1790_179021


namespace total_ceilings_to_paint_l1790_179006

theorem total_ceilings_to_paint (ceilings_painted_this_week : ℕ) 
                                (ceilings_painted_next_week : ℕ)
                                (ceilings_left_to_paint : ℕ) 
                                (h1 : ceilings_painted_this_week = 12) 
                                (h2 : ceilings_painted_next_week = ceilings_painted_this_week / 4) 
                                (h3 : ceilings_left_to_paint = 13) : 
    ceilings_painted_this_week + ceilings_painted_next_week + ceilings_left_to_paint = 28 :=
by
  sorry

end total_ceilings_to_paint_l1790_179006


namespace even_function_condition_iff_l1790_179067

theorem even_function_condition_iff (m : ℝ) :
    (∀ x : ℝ, (m * 2^x + 2^(-x)) = (m * 2^(-x) + 2^x)) ↔ (m = 1) :=
by
  sorry

end even_function_condition_iff_l1790_179067


namespace david_trip_distance_l1790_179040

theorem david_trip_distance (t : ℝ) (d : ℝ) : 
  (40 * (t + 1) = d) →
  (d - 40 = 60 * (t - 0.75)) →
  d = 130 := 
by
  intro h1 h2
  sorry

end david_trip_distance_l1790_179040


namespace new_mean_after_adding_eleven_l1790_179090

theorem new_mean_after_adding_eleven (nums : List ℝ) (h_len : nums.length = 15) (h_avg : (nums.sum / 15) = 40) :
  ((nums.map (λ x => x + 11)).sum / 15) = 51 := by
  sorry

end new_mean_after_adding_eleven_l1790_179090


namespace division_multiplication_identity_l1790_179027

theorem division_multiplication_identity (a b c d : ℕ) (h1 : b = 6) (h2 : c = 2) (h3 : d = 3) :
  a = 120 → 120 * (b / c) * d = 120 := by
  intro h
  rw [h2, h3, h1]
  sorry

end division_multiplication_identity_l1790_179027


namespace real_solutions_l1790_179003

theorem real_solutions : 
  ∃ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ∧ (x = 7 ∨ x = -2) :=
sorry

end real_solutions_l1790_179003


namespace common_chord_length_of_two_circles_l1790_179035

-- Define the equations of the circles C1 and C2
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y - 4 = 0
def circle2 (x y : ℝ) : Prop := (x + 3 / 2)^2 + (y - 3 / 2)^2 = 11 / 2

-- The theorem stating the length of the common chord
theorem common_chord_length_of_two_circles :
  ∃ l : ℝ, (∀ (x y : ℝ), circle1 x y ↔ circle2 x y) → l = 2 :=
by simp [circle1, circle2]; sorry

end common_chord_length_of_two_circles_l1790_179035


namespace rubble_initial_money_l1790_179028

def initial_money (cost_notebook cost_pen : ℝ) (num_notebooks num_pens : ℕ) (money_left : ℝ) : ℝ :=
  (num_notebooks * cost_notebook + num_pens * cost_pen) + money_left

theorem rubble_initial_money :
  initial_money 4 1.5 2 2 4 = 15 :=
by
  sorry

end rubble_initial_money_l1790_179028


namespace compare_two_sqrt_three_with_three_l1790_179017

theorem compare_two_sqrt_three_with_three : 2 * Real.sqrt 3 > 3 :=
sorry

end compare_two_sqrt_three_with_three_l1790_179017


namespace number_of_violas_l1790_179004

theorem number_of_violas (V : ℕ) 
  (cellos : ℕ := 800) 
  (pairs : ℕ := 70) 
  (probability : ℝ := 0.00014583333333333335) 
  (h : probability = pairs / (cellos * V)) : V = 600 :=
by
  sorry

end number_of_violas_l1790_179004


namespace ratio_of_inscribed_squares_l1790_179030

theorem ratio_of_inscribed_squares (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (hx : x = 60 / 17) (hy : y = 3) :
  x / y = 20 / 17 :=
by
  sorry

end ratio_of_inscribed_squares_l1790_179030


namespace connections_in_computer_lab_l1790_179034

theorem connections_in_computer_lab (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end connections_in_computer_lab_l1790_179034


namespace resistance_between_opposite_vertices_of_cube_l1790_179012

-- Define the parameters of the problem
def resistance_cube_edge : ℝ := 1

-- Define the function to calculate the equivalent resistance
noncomputable def equivalent_resistance_opposite_vertices (R : ℝ) : ℝ :=
  let R1 := R / 3
  let R2 := R / 6
  let R3 := R / 3
  R1 + R2 + R3

-- State the theorem to prove the resistance between two opposite vertices
theorem resistance_between_opposite_vertices_of_cube :
  equivalent_resistance_opposite_vertices resistance_cube_edge = 5 / 6 :=
by
  sorry

end resistance_between_opposite_vertices_of_cube_l1790_179012


namespace max_street_lamps_proof_l1790_179097

noncomputable def max_street_lamps_on_road : ℕ := 1998

theorem max_street_lamps_proof (L : ℕ) (l : ℕ)
    (illuminates : ∀ i, i ≤ max_street_lamps_on_road → 
                  (∃ unique_segment : ℕ, unique_segment ≤ L ∧ unique_segment > L - l )):
  max_street_lamps_on_road = 1998 := by
  sorry

end max_street_lamps_proof_l1790_179097


namespace area_bounded_by_circles_and_x_axis_l1790_179061

/--
Circle C has its center at (5, 5) and radius 5 units.
Circle D has its center at (15, 5) and radius 5 units.
Prove that the area of the region bounded by these circles
and the x-axis is 50 - 25 * π square units.
-/
theorem area_bounded_by_circles_and_x_axis :
  let C_center := (5, 5)
  let D_center := (15, 5)
  let radius := 5
  (2 * (radius * radius) * π / 2) + (10 * radius) = 50 - 25 * π :=
sorry

end area_bounded_by_circles_and_x_axis_l1790_179061


namespace solve_inequality_l1790_179000

theorem solve_inequality (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
  (x^2 - 9) / (x^2 - 1) > 0 ↔ (x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1)) :=
sorry

end solve_inequality_l1790_179000


namespace rain_on_tuesday_l1790_179041

/-- Let \( R_M \) be the event that a county received rain on Monday. -/
def RM : Prop := sorry

/-- Let \( R_T \) be the event that a county received rain on Tuesday. -/
def RT : Prop := sorry

/-- Let \( R_{MT} \) be the event that a county received rain on both Monday and Tuesday. -/
def RMT : Prop := RM ∧ RT

/-- The probability that a county received rain on Monday is 0.62. -/
def prob_RM : ℝ := 0.62

/-- The probability that a county received rain on both Monday and Tuesday is 0.44. -/
def prob_RMT : ℝ := 0.44

/-- The probability that no rain fell on either day is 0.28. -/
def prob_no_rain : ℝ := 0.28

/-- The probability that a county received rain on at least one of the days is 0.72. -/
def prob_at_least_one_day : ℝ := 1 - prob_no_rain

/-- The probability that a county received rain on Tuesday is 0.54. -/
theorem rain_on_tuesday : (prob_at_least_one_day = prob_RM + x - prob_RMT) → (x = 0.54) :=
by
  intros h
  sorry

end rain_on_tuesday_l1790_179041


namespace cost_price_proof_l1790_179095

noncomputable def selling_price : Real := 12000
noncomputable def discount_rate : Real := 0.10
noncomputable def new_selling_price : Real := selling_price * (1 - discount_rate)
noncomputable def profit_rate : Real := 0.08

noncomputable def cost_price : Real := new_selling_price / (1 + profit_rate)

theorem cost_price_proof : cost_price = 10000 := by sorry

end cost_price_proof_l1790_179095


namespace propositions_false_l1790_179045

structure Plane :=
(is_plane : Prop)

structure Line :=
(in_plane : Plane → Prop)

def is_parallel (p1 p2 : Plane) : Prop := sorry
def is_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular (l1 l2 : Line) : Prop := sorry

variable (α β : Plane)
variable (l m : Line)

axiom α_neq_β : α ≠ β
axiom l_in_α : l.in_plane α
axiom m_in_β : m.in_plane β

theorem propositions_false :
  ¬(is_parallel α β → line_parallel l m) ∧ 
  ¬(line_perpendicular l m → is_perpendicular α β) := 
sorry

end propositions_false_l1790_179045


namespace age_composition_is_decline_l1790_179093

-- Define the population and age groups
variable (P : Type)
variable (Y E : P → ℕ) -- Functions indicating the number of young and elderly individuals

-- Assumptions as per the conditions
axiom fewer_young_more_elderly (p : P) : Y p < E p

-- Conclusion: Prove that the population is of Decline type.
def age_composition_decline (p : P) : Prop :=
  Y p < E p

theorem age_composition_is_decline (p : P) : age_composition_decline P Y E p := by
  sorry

end age_composition_is_decline_l1790_179093


namespace anna_chargers_l1790_179091

theorem anna_chargers (P L: ℕ) (h1: L = 5 * P) (h2: P + L = 24): P = 4 := by
  sorry

end anna_chargers_l1790_179091


namespace population_increase_l1790_179088

-- Define the problem conditions
def average_birth_rate := (6 + 10) / 2 / 2  -- the average number of births per second
def average_death_rate := (4 + 8) / 2 / 2  -- the average number of deaths per second
def net_migration_day := 500  -- net migration inflow during the day
def net_migration_night := -300  -- net migration outflow during the night

-- Define the number of seconds in a day
def seconds_in_a_day := 24 * 3600

-- Define the net increase due to births and deaths
def net_increase_births_deaths := (average_birth_rate - average_death_rate) * seconds_in_a_day

-- Define the total net migration
def total_net_migration := net_migration_day + net_migration_night

-- Define the total population net increase
def total_population_net_increase :=
  net_increase_births_deaths + total_net_migration

-- The theorem to be proved
theorem population_increase (h₁ : average_birth_rate = 4)
                           (h₂ : average_death_rate = 3)
                           (h₃ : seconds_in_a_day = 86400) :
  total_population_net_increase = 86600 := by
  sorry

end population_increase_l1790_179088


namespace union_M_N_l1790_179019

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l1790_179019


namespace jordyn_total_cost_l1790_179020

-- Definitions for conditions
def price_cherries : ℝ := 5
def price_olives : ℝ := 7
def number_of_bags : ℕ := 50
def discount_rate : ℝ := 0.10 

-- Define the discounted price function
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate the total cost for Jordyn
def total_cost (price_cherries price_olives : ℝ) (number_of_bags : ℕ) (discount_rate : ℝ) : ℝ :=
  (number_of_bags * discounted_price price_cherries discount_rate) + 
  (number_of_bags * discounted_price price_olives discount_rate)

-- Prove the final cost
theorem jordyn_total_cost : total_cost price_cherries price_olives number_of_bags discount_rate = 540 := by
  sorry

end jordyn_total_cost_l1790_179020


namespace probability_age_less_than_20_l1790_179081

theorem probability_age_less_than_20 (total : ℕ) (ages_gt_30 : ℕ) (ages_lt_20 : ℕ) 
    (h1 : total = 150) (h2 : ages_gt_30 = 90) (h3 : ages_lt_20 = total - ages_gt_30) :
    (ages_lt_20 : ℚ) / total = 2 / 5 :=
by
  simp [h1, h2, h3]
  sorry

end probability_age_less_than_20_l1790_179081


namespace ratio_of_length_to_breadth_l1790_179098

theorem ratio_of_length_to_breadth 
    (breadth : ℝ) (area : ℝ) (h_breadth : breadth = 12) (h_area : area = 432)
    (h_area_formula : area = l * breadth) : 
    l / breadth = 3 :=
sorry

end ratio_of_length_to_breadth_l1790_179098


namespace calculate_fraction_l1790_179026

theorem calculate_fraction : (10^20 / 50^10) = 2^10 := by
  sorry

end calculate_fraction_l1790_179026


namespace find_number_l1790_179023

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 18) : x = 9 :=
sorry

end find_number_l1790_179023


namespace ways_to_sum_420_l1790_179047

theorem ways_to_sum_420 : 
  (∃ n k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 > 0 ∧ n * (2 * k + n - 1) = 840) → (∃ c, c = 11) :=
by
  sorry

end ways_to_sum_420_l1790_179047


namespace parallelogram_area_l1790_179080

-- Definitions
def base_cm : ℕ := 22
def height_cm : ℕ := 21

-- Theorem statement
theorem parallelogram_area : base_cm * height_cm = 462 := by
  sorry

end parallelogram_area_l1790_179080


namespace total_blocks_needed_l1790_179082

theorem total_blocks_needed (length height : ℕ) (block_height : ℕ) (block1_length block2_length : ℕ)
                            (height_blocks : height = 8) (length_blocks : length = 102)
                            (block_height_cond : block_height = 1)
                            (block_lengths : block1_length = 2 ∧ block2_length = 1)
                            (staggered_cond : True) (even_ends : True) :
  ∃ total_blocks, total_blocks = 416 := 
  sorry

end total_blocks_needed_l1790_179082


namespace point_same_side_of_line_l1790_179013

def same_side (p₁ p₂ : ℝ × ℝ) (a b c : ℝ) : Prop :=
  (a * p₁.1 + b * p₁.2 + c > 0) ↔ (a * p₂.1 + b * p₂.2 + c > 0)

theorem point_same_side_of_line :
  same_side (1, 2) (1, 0) 2 (-1) 1 :=
by
  unfold same_side
  sorry

end point_same_side_of_line_l1790_179013


namespace sarah_reads_100_words_per_page_l1790_179044

noncomputable def words_per_page (W_pages : ℕ) (books : ℕ) (hours : ℕ) (pages_per_book : ℕ) (words_per_minute : ℕ) : ℕ :=
  (words_per_minute * 60 * hours) / books / pages_per_book

theorem sarah_reads_100_words_per_page :
  words_per_page 80 6 20 80 40 = 100 := 
sorry

end sarah_reads_100_words_per_page_l1790_179044


namespace solve_system_I_solve_system_II_l1790_179029

theorem solve_system_I (x y : ℝ) (h1 : y = x + 3) (h2 : x - 2 * y + 12 = 0) : x = 6 ∧ y = 9 :=
by
  sorry

theorem solve_system_II (x y : ℝ) (h1 : 4 * (x - y - 1) = 3 * (1 - y) - 2) (h2 : x / 2 + y / 3 = 2) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_system_I_solve_system_II_l1790_179029


namespace sum_of_inscribed_angles_l1790_179094

theorem sum_of_inscribed_angles 
  (n : ℕ) 
  (total_degrees : ℝ)
  (arcs : ℕ)
  (x_arcs : ℕ)
  (y_arcs : ℕ) 
  (arc_angle : ℝ)
  (x_central_angle : ℝ)
  (y_central_angle : ℝ)
  (x_inscribed_angle : ℝ)
  (y_inscribed_angle : ℝ)
  (total_inscribed_angles : ℝ) :
  n = 18 →
  total_degrees = 360 →
  x_arcs = 3 →
  y_arcs = 5 →
  arc_angle = total_degrees / n →
  x_central_angle = x_arcs * arc_angle →
  y_central_angle = y_arcs * arc_angle →
  x_inscribed_angle = x_central_angle / 2 →
  y_inscribed_angle = y_central_angle / 2 →
  total_inscribed_angles = x_inscribed_angle + y_inscribed_angle →
  total_inscribed_angles = 80 := sorry

end sum_of_inscribed_angles_l1790_179094


namespace cannot_sum_to_nine_l1790_179018

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end cannot_sum_to_nine_l1790_179018


namespace unique_solution_for_star_l1790_179046

def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

theorem unique_solution_for_star :
  ∃! y : ℝ, star 2 y = 5 :=
by
  -- We know the definition of star and we need to verify the condition.
  sorry

end unique_solution_for_star_l1790_179046


namespace binomial_product_l1790_179042

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x ^ 2 - 21 * x - 18 := 
sorry

end binomial_product_l1790_179042


namespace prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l1790_179005

theorem prime_in_form_x_squared_plus_16y_squared (p : ℕ) (hprime : Prime p) (h1 : p % 8 = 1) :
  ∃ x y : ℤ, p = x^2 + 16 * y^2 :=
by
  sorry

theorem prime_in_form_4x_squared_plus_4xy_plus_5y_squared (p : ℕ) (hprime : Prime p) (h5 : p % 8 = 5) :
  ∃ x y : ℤ, p = 4 * x^2 + 4 * x * y + 5 * y^2 :=
by
  sorry

end prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l1790_179005


namespace problem1_l1790_179089

theorem problem1 (a : ℝ) 
    (circle_eqn : ∀ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0)
    (line_eqn : ∀ (x y : ℝ), a*x + y + 1 = 0)
    (chord_length : ∀ (x y : ℝ), (ax + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)  -> ((x - x')^2 + (y - y')^2 = 4)) : 
    a = -2 := sorry

end problem1_l1790_179089


namespace first_train_length_correct_l1790_179079

noncomputable def length_of_first_train : ℝ :=
  let speed_first_train := 90 * 1000 / 3600  -- converting to m/s
  let speed_second_train := 72 * 1000 / 3600 -- converting to m/s
  let relative_speed := speed_first_train + speed_second_train
  let distance_apart := 630
  let length_second_train := 200
  let time_to_meet := 13.998880089592832
  let distance_covered := relative_speed * time_to_meet
  let total_distance := distance_apart
  let length_first_train := total_distance - length_second_train
  length_first_train

theorem first_train_length_correct :
  length_of_first_train = 430 :=
by
  -- Place for the proof steps
  sorry

end first_train_length_correct_l1790_179079


namespace original_peaches_l1790_179033

theorem original_peaches (picked: ℕ) (current: ℕ) (initial: ℕ) : 
  picked = 52 → 
  current = 86 → 
  initial = current - picked → 
  initial = 34 := 
by intros h1 h2 h3
   subst h1
   subst h2
   subst h3
   simp

end original_peaches_l1790_179033


namespace Xiaoliang_catches_up_in_h_l1790_179084

-- Define the speeds and head start
def speed_Xiaobin : ℝ := 4  -- Xiaobin's speed in km/h
def speed_Xiaoliang : ℝ := 12  -- Xiaoliang's speed in km/h
def head_start : ℝ := 6  -- Xiaobin's head start in hours

-- Define the additional distance Xiaoliang needs to cover
def additional_distance : ℝ := speed_Xiaobin * head_start

-- Define the hourly distance difference between them
def speed_difference : ℝ := speed_Xiaoliang - speed_Xiaobin

-- Prove that Xiaoliang will catch up with Xiaobin in exactly 3 hours
theorem Xiaoliang_catches_up_in_h : (additional_distance / speed_difference) = 3 :=
by
  sorry

end Xiaoliang_catches_up_in_h_l1790_179084


namespace least_positive_integer_to_multiple_of_4_l1790_179050

theorem least_positive_integer_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ ((563 + n) % 4 = 0) ∧ n = 1 := 
by
  sorry

end least_positive_integer_to_multiple_of_4_l1790_179050


namespace brian_oranges_is_12_l1790_179063

-- Define the number of oranges the person has
def person_oranges : Nat := 12

-- Define the number of oranges Brian has, which is zero fewer than the person's oranges
def brian_oranges : Nat := person_oranges - 0

-- The theorem stating that Brian has 12 oranges
theorem brian_oranges_is_12 : brian_oranges = 12 :=
by
  -- Proof is omitted
  sorry

end brian_oranges_is_12_l1790_179063


namespace molecular_weight_l1790_179075

variable (weight_moles : ℝ) (moles : ℝ)

-- Given conditions
axiom h1 : weight_moles = 699
axiom h2 : moles = 3

-- Concluding statement to prove
theorem molecular_weight : (weight_moles / moles) = 233 := sorry

end molecular_weight_l1790_179075


namespace geometric_sequence_second_term_l1790_179066

theorem geometric_sequence_second_term (a r : ℕ) (h1 : a = 5) (h2 : a * r^4 = 1280) : a * r = 20 :=
by
  sorry

end geometric_sequence_second_term_l1790_179066


namespace volume_of_displaced_water_l1790_179099

-- Defining the conditions of the problem
def cube_side_length : ℝ := 6
def cyl_radius : ℝ := 5
def cyl_height : ℝ := 12
def cube_volume (s : ℝ) : ℝ := s^3

-- Statement: The volume of water displaced by the cube when it is fully submerged in the barrel
theorem volume_of_displaced_water :
  cube_volume cube_side_length = 216 := by
  sorry

end volume_of_displaced_water_l1790_179099


namespace sqrt_of_quarter_l1790_179062

-- Definitions as per conditions
def is_square_root (x y : ℝ) : Prop := x^2 = y

-- Theorem statement proving question == answer given conditions
theorem sqrt_of_quarter : is_square_root 0.5 0.25 ∧ is_square_root (-0.5) 0.25 ∧ (∀ x, is_square_root x 0.25 → (x = 0.5 ∨ x = -0.5)) :=
by
  -- Skipping proof with sorry
  sorry

end sqrt_of_quarter_l1790_179062


namespace max_rectangle_area_l1790_179069

-- Lean statement for the proof problem

theorem max_rectangle_area (x : ℝ) (y : ℝ) (h1 : 2 * x + 2 * y = 24) : ∃ A : ℝ, A = 36 :=
by
  -- Definitions for perimeter and area
  let P := 2 * x + 2 * y
  let A := x * y

  -- Conditions
  have h1 : P = 24 := h1

  -- Setting maximum area and completing the proof
  sorry

end max_rectangle_area_l1790_179069


namespace percent_more_proof_l1790_179022

-- Define the conditions
def y := 150
def x := 120
def is_percent_more (y x p : ℕ) : Prop := y = (1 + p / 100) * x

-- The proof problem statement
theorem percent_more_proof : ∃ p : ℕ, is_percent_more y x p ∧ p = 25 := by
  sorry

end percent_more_proof_l1790_179022


namespace find_difference_square_l1790_179073

theorem find_difference_square (x y c b : ℝ) (h1 : x * y = c^2) (h2 : (1 / x^2) + (1 / y^2) = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := 
by sorry

end find_difference_square_l1790_179073


namespace number_divisible_by_75_l1790_179064

def is_two_digit (x : ℕ) := x >= 10 ∧ x < 100

theorem number_divisible_by_75 {a b : ℕ} (h1 : a * b = 35) (h2 : is_two_digit (10 * a + b)) : (10 * a + b) % 75 = 0 :=
sorry

end number_divisible_by_75_l1790_179064


namespace min_value_of_a_l1790_179065

theorem min_value_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_of_a_l1790_179065


namespace linear_function_passing_points_l1790_179078

theorem linear_function_passing_points :
  ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b) ∧ (k * 0 + b = 3) ∧ (k * (-4) + b = 0)
  →
  (∃ a : ℝ, y = -((3:ℝ) / (4:ℝ)) * x + 3 ∧ (∀ x y : ℝ, y = -((3:ℝ) / (4:ℝ)) * a + 3 → y = 6 → a = -4)) :=
by sorry

end linear_function_passing_points_l1790_179078


namespace unique_solution_only_a_is_2_l1790_179037

noncomputable def unique_solution_inequality (a : ℝ) : Prop :=
  ∀ (p : ℝ → ℝ), (∀ x, 0 ≤ p x ∧ p x ≤ 1 ∧ p x = x^2 - a * x + a) → 
  ∃! x, p x = 1

theorem unique_solution_only_a_is_2 (a : ℝ) (h : unique_solution_inequality a) : a = 2 :=
sorry

end unique_solution_only_a_is_2_l1790_179037


namespace number_is_280_l1790_179074

theorem number_is_280 (x : ℝ) (h : x / 5 + 4 = x / 4 - 10) : x = 280 := 
by 
  sorry

end number_is_280_l1790_179074


namespace farmer_randy_total_acres_l1790_179031

-- Define the conditions
def acres_per_tractor_per_day : ℕ := 68
def tractors_first_2_days : ℕ := 2
def days_first_period : ℕ := 2
def tractors_next_3_days : ℕ := 7
def days_second_period : ℕ := 3

-- Prove the total acres Farmer Randy needs to plant
theorem farmer_randy_total_acres :
  (tractors_first_2_days * acres_per_tractor_per_day * days_first_period) +
  (tractors_next_3_days * acres_per_tractor_per_day * days_second_period) = 1700 :=
by
  -- Here, we would provide the proof, but in this example, we will use sorry.
  sorry

end farmer_randy_total_acres_l1790_179031


namespace male_contestants_count_l1790_179009

-- Define the total number of contestants.
def total_contestants : ℕ := 18

-- A third of the contestants are female.
def female_contestants : ℕ := total_contestants / 3

-- The rest are male.
def male_contestants : ℕ := total_contestants - female_contestants

-- The theorem to prove
theorem male_contestants_count : male_contestants = 12 := by
  -- Proof goes here.
  sorry

end male_contestants_count_l1790_179009


namespace wuyang_math_total_participants_l1790_179051

theorem wuyang_math_total_participants :
  ∀ (x : ℕ), 
  95 * (x + 5) = 75 * (x + 3 + 10) → 
  2 * (x + x + 8) + 9 = 125 :=
by
  intro x h
  sorry

end wuyang_math_total_participants_l1790_179051


namespace scientific_notation_of_600000_l1790_179039

theorem scientific_notation_of_600000 : 600000 = 6 * 10^5 :=
by
  sorry

end scientific_notation_of_600000_l1790_179039


namespace area_of_square_l1790_179083

noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle_given_length_and_breadth (L B : ℝ) : ℝ := L * B

theorem area_of_square (r : ℝ) (B : ℝ) (A : ℝ) 
  (h_length : length_of_rectangle r = (2 / 5) * r) 
  (h_breadth : B = 10) 
  (h_area : A = 160) 
  (h_rectangle_area : area_of_rectangle_given_length_and_breadth ((2 / 5) * r) B = 160) : 
  r = 40 → (r ^ 2 = 1600) := 
by 
  sorry

end area_of_square_l1790_179083


namespace sqrt_product_simplification_l1790_179038

-- Define the main problem
theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := 
by
  sorry

end sqrt_product_simplification_l1790_179038


namespace earnings_per_day_correct_l1790_179060

-- Given conditions
variable (total_earned : ℕ) (days : ℕ) (earnings_per_day : ℕ)

-- Specify the given values from the conditions
def given_conditions : Prop :=
  total_earned = 165 ∧ days = 5 ∧ total_earned = days * earnings_per_day

-- Statement of the problem: proving the earnings per day
theorem earnings_per_day_correct (h : given_conditions total_earned days earnings_per_day) : 
  earnings_per_day = 33 :=
by
  sorry

end earnings_per_day_correct_l1790_179060


namespace surface_area_combination_l1790_179049

noncomputable def smallest_surface_area : ℕ :=
  let s1 := 3
  let s2 := 5
  let s3 := 8
  let surface_area := 6 * (s1 * s1 + s2 * s2 + s3 * s3)
  let overlap_area := (s1 * s1) * 4 + (s2 * s2) * 2 
  surface_area - overlap_area

theorem surface_area_combination :
  smallest_surface_area = 502 :=
by
  -- Proof goes here
  sorry

end surface_area_combination_l1790_179049


namespace find_a_if_f_is_even_l1790_179086

-- Defining f as given in the problem conditions
noncomputable def f (x a : ℝ) : ℝ := (x + a) * 3 ^ (x - 2 + a ^ 2) - (x - a) * 3 ^ (8 - x - 3 * a)

-- Statement of the proof problem with the conditions
theorem find_a_if_f_is_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → (a = -5 ∨ a = 2) :=
by
  sorry

end find_a_if_f_is_even_l1790_179086


namespace original_price_of_dinosaur_model_l1790_179072

-- Define the conditions
theorem original_price_of_dinosaur_model
  (P : ℝ) -- original price of each model
  (kindergarten_models : ℝ := 2)
  (elementary_models : ℝ := 2 * kindergarten_models)
  (total_models : ℝ := kindergarten_models + elementary_models)
  (reduction_percentage : ℝ := 0.05)
  (discounted_price : ℝ := P * (1 - reduction_percentage))
  (total_paid : ℝ := total_models * discounted_price)
  (total_paid_condition : total_paid = 570) :
  P = 100 :=
by
  sorry

end original_price_of_dinosaur_model_l1790_179072


namespace upper_limit_of_sixth_powers_l1790_179054

theorem upper_limit_of_sixth_powers :
  ∃ b : ℕ, (∀ n : ℕ, (∃ a : ℕ, a^6 = n) ∧ n ≤ b → n = 46656) :=
by
  sorry

end upper_limit_of_sixth_powers_l1790_179054


namespace intersection_point_exists_l1790_179025

def line_l (x y : ℝ) : Prop := 2 * x + y = 10
def line_l_prime (x y : ℝ) : Prop := x - 2 * y + 10 = 0
def passes_through (x y : ℝ) (p : ℝ × ℝ) : Prop := p.2 = y ∧ 2 * p.1 - 10 = x

theorem intersection_point_exists :
  ∃ p : ℝ × ℝ, line_l p.1 p.2 ∧ line_l_prime p.1 p.2 ∧ passes_through p.1 p.2 (-10, 0) :=
sorry

end intersection_point_exists_l1790_179025


namespace horse_revolutions_l1790_179015

theorem horse_revolutions :
  ∀ (r_1 r_2 : ℝ) (n : ℕ),
    r_1 = 30 → r_2 = 10 → n = 25 → (r_1 * n) / r_2 = 75 := by
  sorry

end horse_revolutions_l1790_179015


namespace total_working_days_l1790_179070

theorem total_working_days 
  (D : ℕ)
  (A : ℝ)
  (B : ℝ)
  (h1 : A * (D - 2) = 80)
  (h2 : B * (D - 5) = 63)
  (h3 : A * (D - 5) = B * (D - 2) + 2) :
  D = 32 := 
sorry

end total_working_days_l1790_179070


namespace smallest_n_l1790_179059

-- Define the costs.
def cost_red := 10 * 8  -- = 80
def cost_green := 18 * 12  -- = 216
def cost_blue := 20 * 15  -- = 300
def cost_yellow (n : Nat) := 24 * n

-- Define the LCM of the costs.
def LCM_cost : Nat := Nat.lcm (Nat.lcm cost_red cost_green) cost_blue

-- Problem statement: Prove that the smallest value of n such that 24 * n is the LCM of the candy costs is 150.
theorem smallest_n : ∃ n : Nat, cost_yellow n = LCM_cost ∧ n = 150 := 
by {
  -- This part is just a placeholder; the proof steps are omitted.
  sorry
}

end smallest_n_l1790_179059


namespace range_of_m_l1790_179076

theorem range_of_m (m : ℝ) (h : 0 < m)
  (subset_cond : ∀ x y : ℝ, x - 4 ≤ 0 → y ≥ 0 → mx - y ≥ 0 → (x - 2)^2 + (y - 2)^2 ≤ 8) :
  m ≤ 1 :=
sorry

end range_of_m_l1790_179076


namespace jacoby_lottery_winning_l1790_179032

theorem jacoby_lottery_winning :
  let total_needed := 5000
  let job_earning := 20 * 10
  let cookies_earning := 4 * 24
  let total_earnings_before_lottery := job_earning + cookies_earning
  let after_lottery := total_earnings_before_lottery - 10
  let gift_from_sisters := 500 * 2
  let total_earnings_and_gifts := after_lottery + gift_from_sisters
  let total_so_far := total_needed - 3214
  total_so_far - total_earnings_and_gifts = 500 :=
by
  sorry

end jacoby_lottery_winning_l1790_179032


namespace f_neg_one_l1790_179085

-- Assume the function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
-- 1. f(x) is odd: f(-x) = -f(x) for all x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- 2. f(x) = 2^x for all x > 0
axiom f_pos : ∀ x : ℝ, x > 0 → f x = 2^x

-- Proof statement to be filled
theorem f_neg_one : f (-1) = -2 := 
by
  sorry

end f_neg_one_l1790_179085


namespace sum_a6_to_a9_l1790_179057

-- Given definitions and conditions
def sequence_sum (n : ℕ) : ℕ := n^3
def a (n : ℕ) : ℕ := sequence_sum (n + 1) - sequence_sum n

-- Theorem to be proved
theorem sum_a6_to_a9 : a 6 + a 7 + a 8 + a 9 = 604 :=
by sorry

end sum_a6_to_a9_l1790_179057


namespace solve_inequality_2_star_x_l1790_179024

theorem solve_inequality_2_star_x :
  ∀ x : ℝ, 
  6 < (2 * x - 2 - x + 3) ∧ (2 * x - 2 - x + 3) < 7 ↔ 5 < x ∧ x < 6 :=
by sorry

end solve_inequality_2_star_x_l1790_179024


namespace h_h_three_l1790_179068

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end h_h_three_l1790_179068


namespace perimeter_of_park_l1790_179011

def length := 300
def breadth := 200

theorem perimeter_of_park : 2 * (length + breadth) = 1000 := by
  sorry

end perimeter_of_park_l1790_179011


namespace trigonometric_identity_l1790_179052

variable (θ : ℝ) (h : Real.tan θ = 2)

theorem trigonometric_identity : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := 
sorry

end trigonometric_identity_l1790_179052


namespace rectangle_height_l1790_179002

variable (h : ℕ) -- Define h as a natural number for the height

-- Given conditions
def width : ℕ := 32
def area_divided_by_diagonal : ℕ := 576

-- Math proof problem
theorem rectangle_height :
  (1 / 2 * (width * h) = area_divided_by_diagonal) → h = 36 := 
by
  sorry

end rectangle_height_l1790_179002


namespace find_x_given_y_l1790_179014

-- Given x varies inversely as the square of y, we define the relationship
def varies_inversely (x y k : ℝ) : Prop := x = k / y^2

theorem find_x_given_y (k : ℝ) (h_k : k = 4) :
  ∀ (y : ℝ), varies_inversely x y k → y = 2 → x = 1 :=
by
  intros y h_varies h_y_eq
  -- We need to prove the statement here
  sorry

end find_x_given_y_l1790_179014
