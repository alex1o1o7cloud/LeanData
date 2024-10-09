import Mathlib

namespace dried_grapes_weight_l159_15976

def fresh_grapes_weight : ℝ := 30
def fresh_grapes_water_percentage : ℝ := 0.60
def dried_grapes_water_percentage : ℝ := 0.20

theorem dried_grapes_weight :
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  dried_grapes = 15 :=
by
  let non_water_content := fresh_grapes_weight * (1 - fresh_grapes_water_percentage)
  let dried_grapes := non_water_content / (1 - dried_grapes_water_percentage)
  show dried_grapes = 15
  sorry

end dried_grapes_weight_l159_15976


namespace hexagon_bc_de_eq_14_l159_15998

theorem hexagon_bc_de_eq_14
  (α β γ δ ε ζ : ℝ)
  (angle_cond : α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ)
  (AB BC CD DE EF FA : ℝ)
  (sum_AB_BC : AB + BC = 11)
  (diff_FA_CD : FA - CD = 3)
  : BC + DE = 14 := sorry

end hexagon_bc_de_eq_14_l159_15998


namespace periodic_symmetry_mono_f_l159_15973

-- Let f be a function from ℝ to ℝ.
variable (f : ℝ → ℝ)

-- f has the domain of ℝ.
-- f(x) = f(x + 6) for all x ∈ ℝ.
axiom periodic_f : ∀ x : ℝ, f x = f (x + 6)

-- f is monotonically decreasing in (0, 3).
axiom mono_f : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → y < 3 → f y < f x

-- The graph of f is symmetric about the line x = 3.
axiom symmetry_f : ∀ x : ℝ, f x = f (6 - x)

-- Prove that f(3.5) < f(1.5) < f(6.5).
theorem periodic_symmetry_mono_f : f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
sorry

end periodic_symmetry_mono_f_l159_15973


namespace triangle_angle_bisector_proportion_l159_15904

theorem triangle_angle_bisector_proportion
  (a b c x y : ℝ)
  (h : x / c = y / a)
  (h2 : x + y = b) :
  x / c = b / (a + c) :=
sorry

end triangle_angle_bisector_proportion_l159_15904


namespace general_term_of_A_inter_B_l159_15978

def setA : Set ℕ := { n*n + n | n : ℕ }
def setB : Set ℕ := { 3*m - 1 | m : ℕ }

theorem general_term_of_A_inter_B (k : ℕ) :
  let a_k := 9*k^2 - 9*k + 2
  a_k ∈ setA ∩ setB ∧ ∀ n ∈ setA ∩ setB, n = a_k :=
sorry

end general_term_of_A_inter_B_l159_15978


namespace find_brick_length_l159_15995

-- Conditions as given in the problem.
def wall_length : ℝ := 8
def wall_width : ℝ := 6
def wall_height : ℝ := 22.5
def number_of_bricks : ℕ := 6400
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- The volume of the wall in cubic centimeters.
def wall_volume_cm_cube : ℝ := (wall_length * 100) * (wall_width * 100) * (wall_height * 100)

-- Define the volume of one brick based on the unknown length L.
def brick_volume (L : ℝ) : ℝ := L * brick_width * brick_height

-- Define an equivalence for the total volume of the bricks to the volume of the wall.
theorem find_brick_length : 
  ∃ (L : ℝ), wall_volume_cm_cube = brick_volume L * number_of_bricks ∧ L = 2500 := 
by
  sorry

end find_brick_length_l159_15995


namespace upstream_distance_l159_15972

-- Define the conditions
def velocity_current : ℝ := 1.5
def distance_downstream : ℝ := 32
def time : ℝ := 6

-- Define the speed of the man in still water
noncomputable def speed_in_still_water : ℝ := (distance_downstream / time) - velocity_current

-- Define the distance rowed upstream
noncomputable def distance_upstream : ℝ := (speed_in_still_water - velocity_current) * time

-- The theorem statement to be proved
theorem upstream_distance (v c d : ℝ) (h1 : c = 1.5) (h2 : (v + c) * 6 = 32) (h3 : (v - c) * 6 = d) : d = 14 :=
by
  -- Insert the proof here
  sorry

end upstream_distance_l159_15972


namespace tank_fill_time_l159_15905

noncomputable def fill_time (T rA rB rC : ℝ) : ℝ :=
  let cycle_fill := rA + rB + rC
  let cycles := T / cycle_fill
  let cycle_time := 3
  cycles * cycle_time

theorem tank_fill_time
  (T : ℝ) (rA rB rC : ℝ) (hT : T = 800) (hrA : rA = 40) (hrB : rB = 30) (hrC : rC = -20) :
  fill_time T rA rB rC = 48 :=
by
  sorry

end tank_fill_time_l159_15905


namespace probability_red_or_yellow_l159_15931

-- Definitions and conditions
def p_green : ℝ := 0.25
def p_blue : ℝ := 0.35
def total_probability := 1
def p_red_and_yellow := total_probability - (p_green + p_blue)

-- Theorem statement
theorem probability_red_or_yellow :
  p_red_and_yellow = 0.40 :=
by
  -- Here we would prove that the combined probability of selecting either a red or yellow jelly bean is 0.40, given the conditions.
  sorry

end probability_red_or_yellow_l159_15931


namespace sum_of_digits_succ_2080_l159_15907

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_succ_2080 (m : ℕ) (h : sum_of_digits m = 2080) :
  sum_of_digits (m + 1) = 2081 ∨ sum_of_digits (m + 1) = 2090 :=
sorry

end sum_of_digits_succ_2080_l159_15907


namespace total_oranges_picked_l159_15924

theorem total_oranges_picked :
  let Mary_oranges := 14
  let Jason_oranges := 41
  let Amanda_oranges := 56
  Mary_oranges + Jason_oranges + Amanda_oranges = 111 := by
    sorry

end total_oranges_picked_l159_15924


namespace chromium_percentage_is_correct_l159_15948

noncomputable def chromium_percentage_new_alloy (chr_percent1 chr_percent2 weight1 weight2 : ℝ) : ℝ :=
  (chr_percent1 * weight1 + chr_percent2 * weight2) / (weight1 + weight2) * 100

theorem chromium_percentage_is_correct :
  chromium_percentage_new_alloy 0.10 0.06 15 35 = 7.2 :=
by
  sorry

end chromium_percentage_is_correct_l159_15948


namespace part_I_part_II_l159_15928

noncomputable section

def f (x a : ℝ) : ℝ := |x + a| + |x - (1 / a)|

theorem part_I (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -5/2 ∨ x ≥ 5/2 := by
  sorry

theorem part_II (a m : ℝ) (h : ∀ x : ℝ, f x a ≥ |m - 1|) : -1 ≤ m ∧ m ≤ 3 := by
  sorry

end part_I_part_II_l159_15928


namespace sum_zero_of_distinct_and_ratio_l159_15954

noncomputable def distinct (a b c d : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

theorem sum_zero_of_distinct_and_ratio (x y u v : ℝ) 
  (h_distinct : distinct x y u v)
  (h_ratio : (x + u) / (x + v) = (y + v) / (y + u)) : 
  x + y + u + v = 0 := 
sorry

end sum_zero_of_distinct_and_ratio_l159_15954


namespace profit_percentage_l159_15959

/-- If the cost price is 81% of the selling price, then the profit percentage is approximately 23.46%. -/
theorem profit_percentage (SP CP: ℝ) (h : CP = 0.81 * SP) : 
  (SP - CP) / CP * 100 = 23.46 := 
sorry

end profit_percentage_l159_15959


namespace relationship_between_abcd_l159_15919

theorem relationship_between_abcd (a b c d : ℝ) (h : d ≠ 0) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) ↔ a / b = c / d :=
by
  sorry

end relationship_between_abcd_l159_15919


namespace johns_contribution_l159_15913

-- Definitions
variables (A J : ℝ)
axiom h1 : 1.5 * A = 75
axiom h2 : (2 * A + J) / 3 = 75

-- Statement of the proof problem
theorem johns_contribution : J = 125 :=
by
  sorry

end johns_contribution_l159_15913


namespace joan_apples_after_giving_l159_15969

-- Definitions of the conditions
def initial_apples : ℕ := 43
def given_away_apples : ℕ := 27

-- Statement to prove
theorem joan_apples_after_giving : (initial_apples - given_away_apples = 16) :=
by sorry

end joan_apples_after_giving_l159_15969


namespace find_minimum_width_l159_15949

-- Definitions based on the problem conditions
def length_from_width (w : ℝ) : ℝ := w + 12

def minimum_fence_area (w : ℝ) : Prop := w * length_from_width w ≥ 144

-- Proof statement
theorem find_minimum_width : ∃ w : ℝ, w ≥ 6 ∧ minimum_fence_area w :=
sorry

end find_minimum_width_l159_15949


namespace coco_hours_used_l159_15990

noncomputable def electricity_price : ℝ := 0.10
noncomputable def consumption_rate : ℝ := 2.4
noncomputable def total_cost : ℝ := 6.0

theorem coco_hours_used (hours_used : ℝ) : hours_used = total_cost / (consumption_rate * electricity_price) :=
by
  sorry

end coco_hours_used_l159_15990


namespace f_2020_eq_neg_1_l159_15922

noncomputable def f: ℝ → ℝ :=
sorry

axiom f_2_x_eq_neg_f_x : ∀ x: ℝ, f (2 - x) = -f x
axiom f_x_minus_2_eq_f_neg_x : ∀ x: ℝ, f (x - 2) = f (-x)
axiom f_specific : ∀ x : ℝ, -1 < x ∧ x < 1 -> f x = x^2 + 1

theorem f_2020_eq_neg_1 : f 2020 = -1 :=
sorry

end f_2020_eq_neg_1_l159_15922


namespace sufficient_but_not_necessary_l159_15909

variable (m : ℝ)

def P : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0
def Q : Prop := ∀ x : ℝ, 3*x^2 + 4*x + m ≥ 0

theorem sufficient_but_not_necessary : (P m → Q m) ∧ ¬(Q m → P m) :=
by
  sorry

end sufficient_but_not_necessary_l159_15909


namespace find_time_for_products_maximize_salary_l159_15967

-- Assume the conditions and definitions based on the given problem
variables (x y a : ℝ)

-- Condition 1: Time to produce 6 type A and 4 type B products is 170 minutes
axiom cond1 : 6 * x + 4 * y = 170

-- Condition 2: Time to produce 10 type A and 10 type B products is 350 minutes
axiom cond2 : 10 * x + 10 * y = 350


-- Question 1: Validating the time to produce one type A product and one type B product
theorem find_time_for_products : 
  x = 15 ∧ y = 20 := by
  sorry

-- Variables for calculation of Zhang's daily salary
variables (m : ℕ) (base_salary : ℝ := 100) (daily_work: ℝ := 480)

-- Conditions for the piece-rate wages
variables (a_condition: 2 < a ∧ a < 3) 
variables (num_products: m + (28 - m) = 28)

-- Question 2: Finding optimal production plan to maximize daily salary
theorem maximize_salary :
  (2 < a ∧ a < 2.5) → m = 16 ∨ 
  (a = 2.5) → true ∨
  (2.5 < a ∧ a < 3) → m = 28 := by
  sorry

end find_time_for_products_maximize_salary_l159_15967


namespace factorize_difference_of_squares_l159_15921

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end factorize_difference_of_squares_l159_15921


namespace area_percent_less_l159_15937

theorem area_percent_less 
  (r1 r2 : ℝ)
  (h : r1 / r2 = 3 / 10) 
  : 1 - (π * (r1:ℝ)^2 / (π * (r2:ℝ)^2)) = 0.91 := 
by 
  sorry

end area_percent_less_l159_15937


namespace total_revenue_correct_l159_15938

-- Defining the basic parameters
def ticket_price : ℝ := 20
def first_discount_percentage : ℝ := 0.40
def next_discount_percentage : ℝ := 0.15
def first_people : ℕ := 10
def next_people : ℕ := 20
def total_people : ℕ := 48

-- Calculate the discounted prices based on the given percentages
def discounted_price_first : ℝ := ticket_price * (1 - first_discount_percentage)
def discounted_price_next : ℝ := ticket_price * (1 - next_discount_percentage)

-- Calculate the total revenue
def revenue_first : ℝ := first_people * discounted_price_first
def revenue_next : ℝ := next_people * discounted_price_next
def remaining_people : ℕ := total_people - first_people - next_people
def revenue_remaining : ℝ := remaining_people * ticket_price

def total_revenue : ℝ := revenue_first + revenue_next + revenue_remaining

-- The statement to be proved
theorem total_revenue_correct : total_revenue = 820 :=
by
  -- The proof will go here
  sorry

end total_revenue_correct_l159_15938


namespace largest_divisible_by_two_power_l159_15999
-- Import the necessary Lean library

open scoped BigOperators

-- Prime and Multiples calculation based conditions
def primes_count : ℕ := 25
def multiples_of_four_count : ℕ := 25

-- Number of subsets of {1, 2, 3, ..., 100} with more primes than multiples of 4
def N : ℕ :=
  let pow := 2^50
  pow * (pow / 2 - (∑ k in Finset.range 26, Nat.choose 25 k ^ 2))

-- Theorem stating that the largest integer k such that 2^k divides N is 52
theorem largest_divisible_by_two_power :
  ∃ (k : ℕ), (2^k ∣ N) ∧ (∀ m : ℕ, 2^m ∣ N → m ≤ 52) :=
sorry

end largest_divisible_by_two_power_l159_15999


namespace race_probability_l159_15908

theorem race_probability (Px : ℝ) (Py : ℝ) (Pz : ℝ) 
  (h1 : Px = 1 / 6) 
  (h2 : Pz = 1 / 8) 
  (h3 : Px + Py + Pz = 0.39166666666666666) : Py = 0.1 := 
sorry

end race_probability_l159_15908


namespace Yoongi_class_students_l159_15939

theorem Yoongi_class_students (Total_a Total_b Total_ab : ℕ)
  (h1 : Total_a = 18)
  (h2 : Total_b = 24)
  (h3 : Total_ab = 7)
  (h4 : Total_a + Total_b - Total_ab = 35) : 
  Total_a + Total_b - Total_ab = 35 :=
sorry

end Yoongi_class_students_l159_15939


namespace seq_a6_l159_15918

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 3 * a n - 2

theorem seq_a6 (a : ℕ → ℕ) (h : seq a) : a 6 = 1 :=
by
  sorry

end seq_a6_l159_15918


namespace vector_magnitude_sub_l159_15932

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (theta : ℝ) (h_theta : theta = Real.pi / 3)

/-- Given vectors a and b with magnitudes 2 and 3 respectively, and the angle between them is 60 degrees,
    we need to prove that the magnitude of the vector a - b is sqrt(7). -/
theorem vector_magnitude_sub : ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end vector_magnitude_sub_l159_15932


namespace std_dev_of_normal_distribution_l159_15987

theorem std_dev_of_normal_distribution (μ σ : ℝ) (h1: μ = 14.5) (h2: μ - 2 * σ = 11.5) : σ = 1.5 := 
by 
  sorry

end std_dev_of_normal_distribution_l159_15987


namespace num_solutions_20_l159_15993

-- Define the number of integer solutions function
def num_solutions (n : ℕ) : ℕ := 4 * n

-- Given conditions
axiom h1 : num_solutions 1 = 4
axiom h2 : num_solutions 2 = 8

-- Theorem to prove the number of solutions for |x| + |y| = 20 is 80
theorem num_solutions_20 : num_solutions 20 = 80 :=
by sorry

end num_solutions_20_l159_15993


namespace rhombus_perimeter_l159_15943

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 52 :=
by
  sorry

end rhombus_perimeter_l159_15943


namespace walking_west_10_neg_l159_15961

-- Define the condition that walking east for 20 meters is +20 meters
def walking_east_20 := 20

-- Assert that walking west for 10 meters is -10 meters given the east direction definition
theorem walking_west_10_neg : walking_east_20 = 20 → (-10 = -10) :=
by
  intro h
  sorry

end walking_west_10_neg_l159_15961


namespace arithmetic_geometric_sequence_l159_15984

theorem arithmetic_geometric_sequence (a1 d : ℝ) (h1 : a1 = 1) (h2 : d ≠ 0) (h_geom : (a1 + d) ^ 2 = a1 * (a1 + 4 * d)) :
  d = 2 :=
by
  sorry

end arithmetic_geometric_sequence_l159_15984


namespace cream_butterfat_percentage_l159_15960

theorem cream_butterfat_percentage (x : ℝ) (h1 : 1 * (x / 100) + 3 * (5.5 / 100) = 4 * (6.5 / 100)) : 
  x = 9.5 :=
by
  sorry

end cream_butterfat_percentage_l159_15960


namespace factor_difference_of_squares_l159_15975

theorem factor_difference_of_squares (a b p q : ℝ) :
  (∃ c d : ℝ, -a ^ 2 + 9 = c ^ 2 - d ^ 2) ∧
  (¬(∃ c d : ℝ, -a ^ 2 - b ^ 2 = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, p ^ 2 - (-q ^ 2) = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, a ^ 2 - b ^ 3 = c ^ 2 - d ^ 2)) := 
  by 
  sorry

end factor_difference_of_squares_l159_15975


namespace correct_equation_l159_15955

/-- Definitions and conditions used in the problem -/
def jan_revenue := 250
def feb_revenue (x : ℝ) := jan_revenue * (1 + x)
def mar_revenue (x : ℝ) := jan_revenue * (1 + x)^2
def first_quarter_target := 900

/-- Proof problem statement -/
theorem correct_equation (x : ℝ) : 
  jan_revenue + feb_revenue x + mar_revenue x = first_quarter_target := 
by
  sorry

end correct_equation_l159_15955


namespace olivia_dad_spent_l159_15988

def cost_per_meal : ℕ := 7
def number_of_meals : ℕ := 3
def total_cost : ℕ := 21

theorem olivia_dad_spent :
  cost_per_meal * number_of_meals = total_cost :=
by
  sorry

end olivia_dad_spent_l159_15988


namespace complement_intersection_l159_15952

open Set

-- Definitions of U, A, and B
def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Proof statement
theorem complement_intersection : 
  ((U \ A) ∩ (U \ B)) = ({0, 2, 4} : Set ℕ) :=
by sorry

end complement_intersection_l159_15952


namespace unique_real_value_for_equal_roots_l159_15902

-- Definitions of conditions
def quadratic_eq (p : ℝ) : Prop := 
  ∀ x : ℝ, x^2 - (p + 1) * x + p = 0

-- Statement of the problem
theorem unique_real_value_for_equal_roots :
  ∃! p : ℝ, ∀ x y : ℝ, (x^2 - (p+1)*x + p = 0) ∧ (y^2 - (p+1)*y + p = 0) → x = y := 
sorry

end unique_real_value_for_equal_roots_l159_15902


namespace chord_central_angle_l159_15996

-- Given that a chord divides the circumference of a circle in the ratio 5:7
-- Prove that the central angle opposite this chord can be either 75° or 105°
theorem chord_central_angle (x : ℝ) (h : 5 * x + 7 * x = 180) :
  5 * x = 75 ∨ 7 * x = 105 :=
sorry

end chord_central_angle_l159_15996


namespace jessica_money_left_l159_15910

theorem jessica_money_left : 
  let initial_amount := 11.73
  let amount_spent := 10.22
  initial_amount - amount_spent = 1.51 :=
by
  sorry

end jessica_money_left_l159_15910


namespace Al_initial_portion_l159_15982

theorem Al_initial_portion (a b c : ℕ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 150 + 2 * b + 3 * c = 1800) 
  (h3 : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a = 550 :=
by {
  sorry
}

end Al_initial_portion_l159_15982


namespace union_A_B_l159_15927

noncomputable def A : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_A_B : A ∪ B = {-3, -2, 2} := by
  sorry

end union_A_B_l159_15927


namespace total_spending_l159_15997

-- Define the condition of spending for each day
def friday_spending : ℝ := 20
def saturday_spending : ℝ := 2 * friday_spending
def sunday_spending : ℝ := 3 * friday_spending

-- Define the statement to be proven
theorem total_spending : friday_spending + saturday_spending + sunday_spending = 120 :=
by
  -- Provide conditions and calculations here (if needed)
  sorry

end total_spending_l159_15997


namespace power_subtraction_l159_15916

variable {a m n : ℝ}

theorem power_subtraction (hm : a^m = 8) (hn : a^n = 2) : a^(m - 3 * n) = 1 := by
  sorry

end power_subtraction_l159_15916


namespace simplify_expression_l159_15906

theorem simplify_expression : 
  let a := (3 + 2 : ℚ)
  let b := a⁻¹ + 2
  let c := b⁻¹ + 2
  let d := c⁻¹ + 2
  d = 65 / 27 := by
  sorry

end simplify_expression_l159_15906


namespace compute_fraction_l159_15958

theorem compute_fraction :
  ((5 * 4) + 6) / 10 = 2.6 :=
by
  sorry

end compute_fraction_l159_15958


namespace sincos_terminal_side_l159_15964

noncomputable def sincos_expr (α : ℝ) :=
  let P : ℝ × ℝ := (-4, 3)
  let r := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)
  let sinα := P.2 / r
  let cosα := P.1 / r
  sinα + 2 * cosα = -1

theorem sincos_terminal_side :
  sincos_expr α :=
by
  sorry

end sincos_terminal_side_l159_15964


namespace max_ab_squared_l159_15936

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  ∃ x, 0 < x ∧ x < 2 ∧ a = 2 - x ∧ ab^2 = x * (2 - x)^2 :=
sorry

end max_ab_squared_l159_15936


namespace combined_cost_price_l159_15912

theorem combined_cost_price :
  let face_value_A : ℝ := 100
  let discount_A : ℝ := 2
  let purchase_price_A := face_value_A - (discount_A / 100 * face_value_A)
  let brokerage_A := 0.2 / 100 * purchase_price_A
  let total_cost_price_A := purchase_price_A + brokerage_A

  let face_value_B : ℝ := 100
  let premium_B : ℝ := 1.5
  let purchase_price_B := face_value_B + (premium_B / 100 * face_value_B)
  let brokerage_B := 0.2 / 100 * purchase_price_B
  let total_cost_price_B := purchase_price_B + brokerage_B

  let combined_cost_price := total_cost_price_A + total_cost_price_B

  combined_cost_price = 199.899 := by
  sorry

end combined_cost_price_l159_15912


namespace intersection_A_and_B_l159_15965

-- Define the sets based on the conditions
def setA : Set ℤ := {x : ℤ | x^2 - 2 * x - 8 ≤ 0}
def setB : Set ℤ := {x : ℤ | 1 < Real.log x / Real.log 2}

-- State the theorem (Note: The logarithmic condition should translate the values to integers)
theorem intersection_A_and_B : setA ∩ setB = {3, 4} :=
sorry

end intersection_A_and_B_l159_15965


namespace remaining_water_after_45_days_l159_15979

def initial_water : ℝ := 500
def daily_loss : ℝ := 1.2
def days : ℝ := 45

theorem remaining_water_after_45_days :
  initial_water - daily_loss * days = 446 := by
  sorry

end remaining_water_after_45_days_l159_15979


namespace melinda_payment_l159_15985

theorem melinda_payment
  (D C : ℝ)
  (h1 : 3 * D + 4 * C = 4.91)
  (h2 : D = 0.45) :
  5 * D + 6 * C = 7.59 := 
by 
-- proof steps go here
sorry

end melinda_payment_l159_15985


namespace mod_equivalence_l159_15966

theorem mod_equivalence (n : ℤ) (hn₁ : 0 ≤ n) (hn₂ : n < 23) (hmod : -250 % 23 = n % 23) : n = 3 := by
  sorry

end mod_equivalence_l159_15966


namespace exists_parallelogram_marked_cells_l159_15946

theorem exists_parallelogram_marked_cells (n : ℕ) (marked : Finset (Fin n × Fin n)) (h_marked : marked.card = 2 * n) :
  ∃ (a b c d : Fin n × Fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
  ((a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2)) :=
sorry

end exists_parallelogram_marked_cells_l159_15946


namespace initial_persons_count_is_eight_l159_15956

noncomputable def number_of_persons_initially 
  (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : ℝ := 
  (new_weight - old_weight) / avg_increase

theorem initial_persons_count_is_eight 
  (avg_increase : ℝ := 2.5) (old_weight : ℝ := 60) (new_weight : ℝ := 80) : 
  number_of_persons_initially avg_increase old_weight new_weight = 8 :=
by
  sorry

end initial_persons_count_is_eight_l159_15956


namespace eval_expression_l159_15951

theorem eval_expression : 4 * (8 - 3) - 7 = 13 :=
by
  sorry

end eval_expression_l159_15951


namespace find_2a_minus_3b_l159_15992

theorem find_2a_minus_3b
  (a b : ℝ)
  (h1 : a * 2 - b * 1 = 4)
  (h2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 :=
by
  sorry

end find_2a_minus_3b_l159_15992


namespace no_flippy_numbers_divisible_by_11_and_6_l159_15986

def is_flippy (n : ℕ) : Prop :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 = d3 ∧ d3 = d5 ∧ d2 = d4 ∧ d1 ≠ d2) ∨ 
  (d2 = d4 ∧ d4 = d5 ∧ d1 = d3 ∧ d1 ≠ d2)

def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11) = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

def sum_divisible_by_6 (n : ℕ) : Prop :=
  (sum_of_digits n) % 6 = 0

theorem no_flippy_numbers_divisible_by_11_and_6 :
  ∀ n, (10000 ≤ n ∧ n < 100000) → is_flippy n → is_divisible_by_11 n → sum_divisible_by_6 n → false :=
by
  intros n h_range h_flippy h_div11 h_sum6
  sorry

end no_flippy_numbers_divisible_by_11_and_6_l159_15986


namespace exists_real_number_lt_neg_one_l159_15941

theorem exists_real_number_lt_neg_one : ∃ (x : ℝ), x < -1 := by
  sorry

end exists_real_number_lt_neg_one_l159_15941


namespace geometric_sequence_sum_eight_l159_15971

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_eight {a1 q : ℝ} (hq : q ≠ 1) 
  (h4 : sum_geometric_sequence a1 q 4 = -5) 
  (h6 : sum_geometric_sequence a1 q 6 = 21 * sum_geometric_sequence a1 q 2) : 
  sum_geometric_sequence a1 q 8 = -85 :=
sorry

end geometric_sequence_sum_eight_l159_15971


namespace find_divisor_l159_15940

theorem find_divisor (x : ℕ) (h : 180 % x = 0) (h_eq : 70 + 5 * 12 / (180 / x) = 71) : x = 3 := 
by
  -- proof goes here
  sorry

end find_divisor_l159_15940


namespace equilateral_triangle_surface_area_correct_l159_15945

noncomputable def equilateral_triangle_surface_area : ℝ :=
  let side_length := 2
  let A := (0, 0, 0)
  let B := (side_length, 0, 0)
  let C := (side_length / 2, (side_length * (Real.sqrt 3)) / 2, 0)
  let D := (side_length / 2, (side_length * (Real.sqrt 3)) / 6, 0)
  let folded_angle := 90
  let diagonal_length := Real.sqrt (1 + 1 + 3)
  let radius := diagonal_length / 2
  let surface_area := 4 * Real.pi * radius^2
  5 * Real.pi

theorem equilateral_triangle_surface_area_correct :
  equilateral_triangle_surface_area = 5 * Real.pi :=
by
  unfold equilateral_triangle_surface_area
  sorry -- proof omitted

end equilateral_triangle_surface_area_correct_l159_15945


namespace find_d_l159_15901

theorem find_d (a b c d x : ℝ)
  (h1 : ∀ x, 2 ≤ a * (Real.cos (b * x + c)) + d ∧ a * (Real.cos (b * x + c)) + d ≤ 4)
  (h2 : Real.cos (b * 0 + c) = 1) :
  d = 3 :=
sorry

end find_d_l159_15901


namespace find_english_score_l159_15963

-- Define the scores
def M : ℕ := 82
def K : ℕ := M + 5
variable (E : ℕ)

-- The average score condition
axiom avg_condition : (K + E + M) / 3 = 89

-- Our goal is to prove that E = 98
theorem find_english_score : E = 98 :=
by
  -- The proof will go here
  sorry

end find_english_score_l159_15963


namespace smallest_m_l159_15957

theorem smallest_m (m : ℕ) (h1 : m > 0) (h2 : 3 ^ ((m + m ^ 2) / 4) > 500) : m = 5 := 
by sorry

end smallest_m_l159_15957


namespace Andre_final_price_l159_15914

theorem Andre_final_price :
  let treadmill_price := 1350
  let treadmill_discount_rate := 0.30
  let plate_price := 60
  let num_of_plates := 2
  let plate_discount_rate := 0.15
  let sales_tax_rate := 0.07
  let treadmill_discount := treadmill_price * treadmill_discount_rate
  let treadmill_discounted_price := treadmill_price - treadmill_discount
  let total_plate_price := plate_price * num_of_plates
  let plate_discount := total_plate_price * plate_discount_rate
  let plate_discounted_price := total_plate_price - plate_discount
  let total_price_before_tax := treadmill_discounted_price + plate_discounted_price
  let sales_tax := total_price_before_tax * sales_tax_rate
  let final_price := total_price_before_tax + sales_tax
  final_price = 1120.29 := 
by
  repeat { 
    sorry 
  }

end Andre_final_price_l159_15914


namespace flight_duration_sum_l159_15983

theorem flight_duration_sum (h m : ℕ) (h_hours : h = 11) (m_minutes : m = 45) (time_limit : 0 < m ∧ m < 60) :
  h + m = 56 :=
by
  sorry

end flight_duration_sum_l159_15983


namespace total_pieces_of_tomatoes_l159_15991

namespace FarmerTomatoes

variables (rows plants_per_row yield_per_plant : ℕ)

def total_plants (rows plants_per_row : ℕ) := rows * plants_per_row

def total_tomatoes (total_plants yield_per_plant : ℕ) := total_plants * yield_per_plant

theorem total_pieces_of_tomatoes 
  (hrows : rows = 30)
  (hplants_per_row : plants_per_row = 10)
  (hyield_per_plant : yield_per_plant = 20) :
  total_tomatoes (total_plants rows plants_per_row) yield_per_plant = 6000 :=
by
  rw [hrows, hplants_per_row, hyield_per_plant]
  unfold total_plants total_tomatoes
  norm_num
  done

end FarmerTomatoes

end total_pieces_of_tomatoes_l159_15991


namespace smallest_possible_n_l159_15953

theorem smallest_possible_n (n : ℕ) (h_pos: n > 0)
  (h_int: (1/3 : ℚ) + 1/4 + 1/9 + 1/n = (1:ℚ)) : 
  n = 18 :=
sorry

end smallest_possible_n_l159_15953


namespace monotonicity_and_extremes_l159_15917

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonicity_and_extremes :
  (∀ x, f x > f (-3) ∨ f x < f (-3)) ∧
  (∀ x, f x > f 1 ∨ f x < f 1) ∧
  (∀ x, (x < -3 → (∀ y, y < x → f y < f x)) ∧ (x > 1 → (∀ y, y > x → f y < f x))) ∧
  f (-3) = 10 ∧ f 1 = -(2 / 3) :=
sorry

end monotonicity_and_extremes_l159_15917


namespace triangle_PQR_not_right_l159_15970

-- Definitions based on conditions
def isIsosceles (a b c : ℝ) (angle1 angle2 : ℝ) : Prop := (angle1 = angle2) ∧ (a = c)

def perimeter (a b c : ℝ) : ℝ := a + b + c

def isRightTriangle (a b c : ℝ) : Prop := a * a = b * b + c * c

-- Given conditions
def PQR : ℝ := 10
def PRQ : ℝ := 10
def QR : ℝ := 6
def angle_PQR : ℝ := 1
def angle_PRQ : ℝ := 1

-- Lean statement for the proof problem
theorem triangle_PQR_not_right 
  (h1 : isIsosceles PQR QR PRQ angle_PQR angle_PRQ)
  (h2 : QR = 6)
  (h3 : PRQ = 10):
  ¬ isRightTriangle PQR QR PRQ ∧ perimeter PQR QR PRQ = 26 :=
by {
    sorry
}

end triangle_PQR_not_right_l159_15970


namespace base_measurement_zions_house_l159_15968

-- Given conditions
def height_zion_house : ℝ := 20
def total_area_three_houses : ℝ := 1200
def num_houses : ℝ := 3

-- Correct answer
def base_zion_house : ℝ := 40

-- Proof statement (question translated to lean statement)
theorem base_measurement_zions_house :
  ∃ base : ℝ, (height_zion_house = 20 ∧ total_area_three_houses = 1200 ∧ num_houses = 3) →
  base = base_zion_house :=
by
  sorry

end base_measurement_zions_house_l159_15968


namespace course_selection_schemes_count_l159_15947

-- Define the total number of courses
def total_courses : ℕ := 8

-- Define the number of courses to choose
def courses_to_choose : ℕ := 5

-- Define the two specific courses, Course A and Course B
def courseA := 1
def courseB := 2

-- Define the combination function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the count when neither Course A nor Course B is selected
def case1 : ℕ := C 6 5

-- Define the count when exactly one of Course A or Course B is selected
def case2 : ℕ := C 2 1 * C 6 4

-- Combining both cases
theorem course_selection_schemes_count : case1 + case2 = 36 :=
by
  -- These would be replaced with actual combination calculations.
  sorry

end course_selection_schemes_count_l159_15947


namespace average_weight_of_three_l159_15920

theorem average_weight_of_three
  (rachel_weight jimmy_weight adam_weight : ℝ)
  (h1 : rachel_weight = 75)
  (h2 : jimmy_weight = rachel_weight + 6)
  (h3 : adam_weight = rachel_weight - 15) :
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by
  sorry

end average_weight_of_three_l159_15920


namespace not_all_x_heart_x_eq_0_l159_15944

def heartsuit (x y : ℝ) : ℝ := abs (x + y)

theorem not_all_x_heart_x_eq_0 :
  ¬ (∀ x : ℝ, heartsuit x x = 0) :=
by sorry

end not_all_x_heart_x_eq_0_l159_15944


namespace hall_width_length_ratio_l159_15930

theorem hall_width_length_ratio 
  (w l : ℝ) 
  (h1 : w * l = 128) 
  (h2 : l - w = 8) : 
  w / l = 1 / 2 := 
by sorry

end hall_width_length_ratio_l159_15930


namespace find_a_l159_15977

-- Define the given conditions
def parabola_eq (a b c y : ℝ) : ℝ := a * y^2 + b * y + c
def vertex : (ℝ × ℝ) := (3, -1)
def point_on_parabola : (ℝ × ℝ) := (7, 3)

-- Define the theorem to be proved
theorem find_a (a b c : ℝ) (h_eqn : ∀ y, parabola_eq a b c y = x)
  (h_vertex : parabola_eq a b c (-vertex.snd) = vertex.fst)
  (h_point : parabola_eq a b c (point_on_parabola.snd) = point_on_parabola.fst) :
  a = 1 / 4 := 
sorry

end find_a_l159_15977


namespace solution_set_inequality_l159_15915

theorem solution_set_inequality (m : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.exp x + Real.exp (-x))
  (h2 : ∀ x, f (-x) = f x) (h3 : ∀ x, 0 ≤ x → ∀ y, 0 ≤ y → x ≤ y → f x ≤ f y) :
  (f (2 * m) > f (m - 2)) ↔ (m > (2 / 3) ∨ m < -2) :=
  sorry

end solution_set_inequality_l159_15915


namespace quotient_when_m_divided_by_11_is_2_l159_15923

theorem quotient_when_m_divided_by_11_is_2 :
  let n_values := [1, 2, 3, 4, 5]
  let squares := n_values.map (λ n => n^2)
  let remainders := List.eraseDup (squares.map (λ x => x % 11))
  let m := remainders.sum
  m / 11 = 2 :=
by
  sorry

end quotient_when_m_divided_by_11_is_2_l159_15923


namespace remainder_when_sum_is_divided_l159_15934

theorem remainder_when_sum_is_divided (n : ℤ) : ((8 - n) + (n + 5)) % 9 = 4 := by
  sorry

end remainder_when_sum_is_divided_l159_15934


namespace probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l159_15925

-- Probability for different numbers facing up when die is thrown twice
theorem probability_different_numbers :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := n_faces * (n_faces - 1)
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry -- Proof to be filled

-- Probability for sum of numbers being 6 when die is thrown twice
theorem probability_sum_six :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := 5
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 36 :=
by
  sorry -- Proof to be filled

-- Probability for exactly three outcomes being odd when die is thrown five times
theorem probability_three_odds_in_five_throws :
  let n_faces := 6
  let n_throws := 5
  let p_odd := 3 / n_faces
  let p_even := 1 - p_odd
  let binomial_coeff := Nat.choose n_throws 3
  let p_three_odds := (binomial_coeff : ℚ) * (p_odd ^ 3) * (p_even ^ 2)
  p_three_odds = 5 / 16 :=
by
  sorry -- Proof to be filled

end probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l159_15925


namespace find_marks_of_a_l159_15911

theorem find_marks_of_a (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : (A + B + C + D) / 4 = 47)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 48) : 
  A = 43 :=
by
  sorry

end find_marks_of_a_l159_15911


namespace crate_minimum_dimension_l159_15962

theorem crate_minimum_dimension (a : ℕ) (h1 : a ≥ 12) :
  min a (min 8 12) = 8 :=
by
  sorry

end crate_minimum_dimension_l159_15962


namespace range_of_theta_div_4_l159_15994

noncomputable def theta_third_quadrant (k : ℤ) (θ : ℝ) : Prop :=
  (2 * k * Real.pi + Real.pi < θ) ∧ (θ < 2 * k * Real.pi + 3 * Real.pi / 2)

noncomputable def sin_lt_cos (θ : ℝ) : Prop :=
  Real.sin (θ / 4) < Real.cos (θ / 4)

theorem range_of_theta_div_4 (k : ℤ) (θ : ℝ) :
  theta_third_quadrant k θ →
  sin_lt_cos θ →
  (2 * k * Real.pi + 5 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 11 * Real.pi / 8) ∨
  (2 * k * Real.pi + 7 * Real.pi / 4 < θ / 4 ∧ θ / 4 < 2 * k * Real.pi + 15 * Real.pi / 8) := 
  by
    sorry

end range_of_theta_div_4_l159_15994


namespace find_a9_l159_15980

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
def a_n : ℕ → ℝ := sorry   -- The sequence itself is unknown initially.

axiom a3 : a_n 3 = 5
axiom a4_a8 : a_n 4 + a_n 8 = 22

theorem find_a9 : a_n 9 = 41 :=
by
  sorry

end find_a9_l159_15980


namespace box_base_length_max_l159_15926

noncomputable def V (x : ℝ) := x^2 * ((60 - x) / 2)

theorem box_base_length_max 
  (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 60)
  (h3 : ∀ y : ℝ, 0 < y ∧ y < 60 → V x ≥ V y)
  : x = 40 :=
sorry

end box_base_length_max_l159_15926


namespace rectangular_prism_volume_l159_15989

theorem rectangular_prism_volume :
  ∀ (l w h : ℕ), 
  l = 2 * w → 
  w = 2 * h → 
  4 * (l + w + h) = 56 → 
  l * w * h = 64 := 
by
  intros l w h h_l_eq_2w h_w_eq_2h h_edge_len_eq_56
  sorry -- proof not provided

end rectangular_prism_volume_l159_15989


namespace james_milk_left_l159_15974

@[simp] def ounces_in_gallon : ℕ := 128
@[simp] def gallons_james_has : ℕ := 3
@[simp] def ounces_drank : ℕ := 13

theorem james_milk_left :
  (gallons_james_has * ounces_in_gallon - ounces_drank) = 371 :=
by
  sorry

end james_milk_left_l159_15974


namespace vertical_asymptote_l159_15900

theorem vertical_asymptote (x : ℝ) : (4 * x + 6 = 0) -> x = -3 / 2 :=
by
  sorry

end vertical_asymptote_l159_15900


namespace triangle_side_length_difference_l159_15929

theorem triangle_side_length_difference (a b c : ℕ) (hb : b = 8) (hc : c = 3)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  let min_a := 6
  let max_a := 10
  max_a - min_a = 4 :=
by {
  sorry
}

end triangle_side_length_difference_l159_15929


namespace quadratic_roots_transform_l159_15950

theorem quadratic_roots_transform {p q : ℝ} (h1 : 3 * p^2 + 5 * p - 7 = 0) (h2 : 3 * q^2 + 5 * q - 7 = 0) : (p - 2) * (q - 2) = 5 := 
by 
  sorry

end quadratic_roots_transform_l159_15950


namespace smallest_sum_B_d_l159_15903

theorem smallest_sum_B_d :
  ∃ B d : ℕ, (B < 5) ∧ (d > 6) ∧ (125 * B + 25 * B + B = 4 * d + 4) ∧ (B + d = 77) :=
by
  sorry

end smallest_sum_B_d_l159_15903


namespace find_certain_number_l159_15933

theorem find_certain_number (h1 : 213 * 16 = 3408) (x : ℝ) (h2 : x * 2.13 = 0.03408) : x = 0.016 :=
by
  sorry

end find_certain_number_l159_15933


namespace smallest_positive_integer_divisible_by_10_13_14_l159_15981

theorem smallest_positive_integer_divisible_by_10_13_14 : ∃ n : ℕ, n > 0 ∧ (10 ∣ n) ∧ (13 ∣ n) ∧ (14 ∣ n) ∧ n = 910 :=
by {
  sorry
}

end smallest_positive_integer_divisible_by_10_13_14_l159_15981


namespace intersection_points_l159_15935

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)

noncomputable def g (x : ℝ) : ℝ := (-3*x^2 - 6*x + 115) / (x - 2)

theorem intersection_points:
  ∃ (x1 x2 : ℝ), x1 ≠ -3 ∧ x2 ≠ -3 ∧ (f x1 = g x1) ∧ (f x2 = g x2) ∧ 
  (x1 = -11 ∧ f x1 = -2) ∧ (x2 = 3 ∧ f x2 = -2) := 
sorry

end intersection_points_l159_15935


namespace smallest_non_lucky_multiple_of_8_l159_15942

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 : ∃ (m : ℕ), (m > 0) ∧ (m % 8 = 0) ∧ ¬ is_lucky_integer m ∧ m = 16 := sorry

end smallest_non_lucky_multiple_of_8_l159_15942
