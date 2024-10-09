import Mathlib

namespace polar_to_rectangular_l726_72651

theorem polar_to_rectangular (r θ : ℝ) (x y : ℝ) 
  (hr : r = 10) 
  (hθ : θ = (3 * Real.pi) / 4) 
  (hx : x = r * Real.cos θ) 
  (hy : y = r * Real.sin θ) 
  :
  x = -5 * Real.sqrt 2 ∧ y = 5 * Real.sqrt 2 := 
by
  -- We assume that the problem is properly stated
  -- Proof omitted here
  sorry

end polar_to_rectangular_l726_72651


namespace common_ratio_is_two_l726_72630

theorem common_ratio_is_two (a r : ℝ) (h_pos : a > 0) 
  (h_sum : a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r)) : 
  r = 2 := 
by
  sorry

end common_ratio_is_two_l726_72630


namespace line_equations_satisfy_conditions_l726_72698

-- Definitions and conditions:
def intersects_at_distance (k m b : ℝ) : Prop :=
  |(k^2 + 7*k + 12) - (m*k + b)| = 8

def passes_through_point (m b : ℝ) : Prop :=
  7 = 2*m + b

def line_equation_valid (m b : ℝ) : Prop :=
  b ≠ 0

-- Main theorem:
theorem line_equations_satisfy_conditions :
  (line_equation_valid 1 5 ∧ passes_through_point 1 5 ∧ 
  ∃ k, intersects_at_distance k 1 5) ∨
  (line_equation_valid 5 (-3) ∧ passes_through_point 5 (-3) ∧ 
  ∃ k, intersects_at_distance k 5 (-3)) :=
by
  sorry

end line_equations_satisfy_conditions_l726_72698


namespace coefficient_a9_l726_72621

theorem coefficient_a9 (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℤ) :
  (x^2 + x^10 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 +
   a4 * (x + 1)^4 + a5 * (x + 1)^5 + a6 * (x + 1)^6 + a7 * (x + 1)^7 +
   a8 * (x + 1)^8 + a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a10 = 1 →
  a9 = -10 :=
by
  sorry

end coefficient_a9_l726_72621


namespace find_linear_odd_increasing_function_l726_72623

theorem find_linear_odd_increasing_function (f : ℝ → ℝ)
    (h1 : ∀ x, f (f x) = 4 * x)
    (h2 : ∀ x, f x = -f (-x))
    (h3 : ∀ x y, x < y → f x < f y)
    (h4 : ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x) : 
    ∀ x, f x = 2 * x :=
by
  sorry

end find_linear_odd_increasing_function_l726_72623


namespace henry_twice_jill_years_ago_l726_72643

def henry_age : ℕ := 23
def jill_age : ℕ := 17
def sum_of_ages (H J : ℕ) : Prop := H + J = 40

theorem henry_twice_jill_years_ago (H J : ℕ) (H1 : sum_of_ages H J) (H2 : H = 23) (H3 : J = 17) : ∃ x : ℕ, H - x = 2 * (J - x) ∧ x = 11 := 
by
  sorry

end henry_twice_jill_years_ago_l726_72643


namespace minimum_rectangle_area_l726_72679

theorem minimum_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 84) : 
  (l * w) = 41 :=
by sorry

end minimum_rectangle_area_l726_72679


namespace bounded_roots_l726_72601

open Polynomial

noncomputable def P : ℤ[X] := sorry -- Replace with actual polynomial if necessary

theorem bounded_roots (P : ℤ[X]) (n : ℕ) (hPdeg : P.degree = n) (hdec : 1 ≤ n) :
  ∀ k : ℤ, (P.eval k) ^ 2 = 1 → ∃ (r s : ℕ), r + s ≤ n + 2 := 
by 
  sorry

end bounded_roots_l726_72601


namespace collinear_values_k_l726_72649

/-- Define the vectors OA, OB, and OC using the given conditions. -/
def vectorOA (k : ℝ) : ℝ × ℝ := (k, 12)
def vectorOB : ℝ × ℝ := (4, 5)
def vectorOC (k : ℝ) : ℝ × ℝ := (10, k)

/-- Define vectors AB and BC using vector subtraction. -/
def vectorAB (k : ℝ) : ℝ × ℝ := (4 - k, -7)
def vectorBC (k : ℝ) : ℝ × ℝ := (6, k - 5)

/-- Collinearity condition for vectors AB and BC. -/
def collinear (k : ℝ) : Prop :=
  (4 - k) * (k - 5) + 42 = 0

/-- Prove that the value of k is 11 or -2 given the collinearity condition. -/
theorem collinear_values_k : ∀ k : ℝ, collinear k → (k = 11 ∨ k = -2) :=
by
  intros k h
  sorry

end collinear_values_k_l726_72649


namespace traveler_never_returns_home_l726_72626

variable (City : Type)
variable (Distance : City → City → ℝ)

variables (A B C : City)
variables (C_i C_i_plus_one C_i_minus_one : City)

-- Given conditions
axiom travel_far_from_A : ∀ (C : City), C ≠ B → Distance A B > Distance A C
axiom travel_far_from_B : ∀ (D : City), D ≠ C → Distance B C > Distance B D
axiom increasing_distance : ∀ i : ℕ, Distance C_i C_i_plus_one > Distance C_i_minus_one C_i

-- Given condition that C is not A
axiom C_not_eq_A : C ≠ A

-- Proof statement
theorem traveler_never_returns_home : ∀ i : ℕ, C_i ≠ A := sorry

end traveler_never_returns_home_l726_72626


namespace evaluate_expression_l726_72663

theorem evaluate_expression :
  3 + 2*Real.sqrt 3 + 1/(3 + 2*Real.sqrt 3) + 1/(2*Real.sqrt 3 - 3) = 3 + (16 * Real.sqrt 3) / 3 :=
by
  sorry

end evaluate_expression_l726_72663


namespace remainder_add_mod_l726_72668

theorem remainder_add_mod (n : ℕ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := 
by sorry

end remainder_add_mod_l726_72668


namespace bus_stop_time_l726_72676

theorem bus_stop_time 
  (bus_speed_without_stoppages : ℤ)
  (bus_speed_with_stoppages : ℤ)
  (h1 : bus_speed_without_stoppages = 54)
  (h2 : bus_speed_with_stoppages = 36) :
  ∃ t : ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l726_72676


namespace dropped_student_score_l726_72677

theorem dropped_student_score (total_students : ℕ) (remaining_students : ℕ) (initial_average : ℝ) (new_average : ℝ) (x : ℝ) 
  (h1 : total_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 62.5) 
  (h4 : new_average = 63.0) 
  (h5 : total_students * initial_average - remaining_students * new_average = x) : 
  x = 55 := 
sorry

end dropped_student_score_l726_72677


namespace sum_of_tangencies_l726_72648

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 23) (max (2 * x + 5) (5 * x + 17))

noncomputable def q (x : ℝ) : ℝ := sorry  -- since the exact form of q is not specified, we use sorry here

-- Define the tangency condition
def is_tangent (q f : ℝ → ℝ) (x : ℝ) : Prop := (q x = f x) ∧ (deriv q x = deriv f x)

-- Define the three points of tangency
variable {x₄ x₅ x₆ : ℝ}

-- q(x) is tangent to f(x) at points x₄, x₅, x₆
axiom tangent_x₄ : is_tangent q f x₄
axiom tangent_x₅ : is_tangent q f x₅
axiom tangent_x₆ : is_tangent q f x₆

-- Now state the theorem
theorem sum_of_tangencies : x₄ + x₅ + x₆ = -70 / 9 :=
sorry

end sum_of_tangencies_l726_72648


namespace pentagon_number_arrangement_l726_72620

def no_common_divisor_other_than_one (a b : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → (d ∣ a ∧ d ∣ b) → false

def has_common_divisor_greater_than_one (a b : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

theorem pentagon_number_arrangement :
  ∃ (A B C D E : ℕ),
    no_common_divisor_other_than_one A B ∧
    no_common_divisor_other_than_one B C ∧
    no_common_divisor_other_than_one C D ∧
    no_common_divisor_other_than_one D E ∧
    no_common_divisor_other_than_one E A ∧
    has_common_divisor_greater_than_one A C ∧
    has_common_divisor_greater_than_one A D ∧
    has_common_divisor_greater_than_one B D ∧
    has_common_divisor_greater_than_one B E ∧
    has_common_divisor_greater_than_one C E :=
sorry

end pentagon_number_arrangement_l726_72620


namespace pinocchio_optimal_success_probability_l726_72697

def success_prob (s : List ℚ) : ℚ :=
  s.foldr (λ x acc => (x * acc) / (1 - (1 - x) * acc)) 1

theorem pinocchio_optimal_success_probability :
  let success_probs := [9/10, 8/10, 7/10, 6/10, 5/10, 4/10, 3/10, 2/10, 1/10]
  success_prob success_probs = 0.4315 :=
by 
  sorry

end pinocchio_optimal_success_probability_l726_72697


namespace algebraic_expression_value_l726_72607

theorem algebraic_expression_value (a x : ℝ) (h : 3 * a - x = x + 2) (hx : x = 2) : a^2 - 2 * a + 1 = 1 :=
by {
  sorry
}

end algebraic_expression_value_l726_72607


namespace jessica_borrowed_amount_l726_72662

def payment_pattern (hour : ℕ) : ℕ :=
  match (hour % 6) with
  | 1 => 2
  | 2 => 4
  | 3 => 6
  | 4 => 8
  | 5 => 10
  | _ => 12

def total_payment (hours_worked : ℕ) : ℕ :=
  (hours_worked / 6) * 42 + (List.sum (List.map payment_pattern (List.range (hours_worked % 6))))

theorem jessica_borrowed_amount :
  total_payment 45 = 306 :=
by
  -- Proof omitted
  sorry

end jessica_borrowed_amount_l726_72662


namespace annie_age_when_anna_three_times_current_age_l726_72671

theorem annie_age_when_anna_three_times_current_age
  (anna_age : ℕ) (annie_age : ℕ)
  (h1 : anna_age = 13)
  (h2 : annie_age = 3 * anna_age) :
  annie_age + 2 * anna_age = 65 :=
by
  sorry

end annie_age_when_anna_three_times_current_age_l726_72671


namespace minimum_sum_PE_PC_l726_72627

noncomputable def point := (ℝ × ℝ)
noncomputable def length (p1 p2 : point) : ℝ := Real.sqrt (((p1.1 - p2.1)^2) + ((p1.2 - p2.2)^2))

theorem minimum_sum_PE_PC :
  let A : point := (0, 3)
  let B : point := (3, 3)
  let C : point := (3, 0)
  let D : point := (0, 0)
  ∃ P E : point, E.1 = 3 ∧ E.2 = 1 ∧ (∃ t : ℝ, t ≥ 0 ∧ t ≤ 3 ∧ P.1 = 3 - t ∧ P.2 = t) ∧
    (length P E + length P C = Real.sqrt 13) :=
by
  sorry

end minimum_sum_PE_PC_l726_72627


namespace possible_even_and_odd_functions_l726_72653

def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem possible_even_and_odd_functions :
  ∃ p q : ℝ → ℝ, is_even_function p ∧ is_odd_function (p ∘ q) ∧ (¬(∀ x, p (q x) = 0)) :=
by
  sorry

end possible_even_and_odd_functions_l726_72653


namespace joggers_meetings_l726_72612

theorem joggers_meetings (road_length : ℝ)
  (speed_A speed_B : ℝ)
  (start_time : ℝ)
  (meeting_time : ℝ) :
  road_length = 400 → 
  speed_A = 3 → 
  speed_B = 2.5 →
  start_time = 0 → 
  meeting_time = 1200 → 
  ∃ y : ℕ, y = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end joggers_meetings_l726_72612


namespace product_of_four_consecutive_even_numbers_divisible_by_240_l726_72652

theorem product_of_four_consecutive_even_numbers_divisible_by_240 :
  ∀ (n : ℤ), (n % 2 = 0) →
    (n + 2) % 2 = 0 →
    (n + 4) % 2 = 0 →
    (n + 6) % 2 = 0 →
    ((n * (n + 2) * (n + 4) * (n + 6)) % 240 = 0) :=
by
  intro n hn hnp2 hnp4 hnp6
  sorry

end product_of_four_consecutive_even_numbers_divisible_by_240_l726_72652


namespace partI_partII_l726_72655

-- Define the absolute value function
def f (x : ℝ) := |x - 1|

-- Part I: Solve the inequality f(x) - f(x+2) < 1
theorem partI (x : ℝ) (h : f x - f (x + 2) < 1) : x > -1 / 2 := 
sorry

-- Part II: Find the range of values for a such that x - f(x + 1 - a) ≤ 1 for all x in [1,2]
theorem partII (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x - f (x + 1 - a) ≤ 1) : a ≤ 1 ∨ a ≥ 3 := 
sorry

end partI_partII_l726_72655


namespace lilly_fish_count_l726_72642

-- Define the number of fish Rosy has
def rosy_fish : ℕ := 9

-- Define the total number of fish
def total_fish : ℕ := 19

-- Define the statement that Lilly has 10 fish given the conditions
theorem lilly_fish_count : rosy_fish + lilly_fish = total_fish → lilly_fish = 10 := by
  intro h
  sorry

end lilly_fish_count_l726_72642


namespace wrongly_noted_mark_l726_72686

theorem wrongly_noted_mark (x : ℕ) (h_wrong_avg : (30 : ℕ) * 100 = 3000)
    (h_correct_avg : (30 : ℕ) * 98 = 2940) (h_correct_sum : 3000 - x + 10 = 2940) : 
    x = 70 := by
  sorry

end wrongly_noted_mark_l726_72686


namespace olivias_dad_total_spending_l726_72661

def people : ℕ := 5
def meal_cost : ℕ := 12
def drink_cost : ℕ := 3
def dessert_cost : ℕ := 5

theorem olivias_dad_total_spending : 
  (people * meal_cost) + (people * drink_cost) + (people * dessert_cost) = 100 := 
by
  sorry

end olivias_dad_total_spending_l726_72661


namespace point_B_coordinates_l726_72692

theorem point_B_coordinates (A B : ℝ) (hA : A = -2) (hDist : |A - B| = 3) : B = -5 ∨ B = 1 :=
by
  sorry

end point_B_coordinates_l726_72692


namespace cost_of_article_is_308_l726_72666

theorem cost_of_article_is_308 
  (C G : ℝ) 
  (h1 : 348 = C + G)
  (h2 : 350 = C + G + 0.05 * G) : 
  C = 308 :=
by
  sorry

end cost_of_article_is_308_l726_72666


namespace find_num_biology_books_l726_72604

-- Given conditions
def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2548

-- Function to calculate combinations
def combination (n k : ℕ) := n.choose k

-- Statement to be proved
theorem find_num_biology_books (B : ℕ) (h1 : combination num_chemistry_books 2 = 28) 
  (h2 : combination B 2 * 28 = total_ways_to_pick) : B = 14 :=
by 
  -- Proof goes here
  sorry

end find_num_biology_books_l726_72604


namespace value_of_expression_l726_72690

theorem value_of_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 5 = 6 :=
by
  sorry

end value_of_expression_l726_72690


namespace intersecting_circles_l726_72619

theorem intersecting_circles (m n : ℝ) (h_intersect : ∃ c1 c2 : ℝ × ℝ, 
  (c1.1 - c1.2 - 2 = 0) ∧ (c2.1 - c2.2 - 2 = 0) ∧
  ∃ r1 r2 : ℝ, (c1.1 - 1)^2 + (c1.2 - 3)^2 = r1^2 ∧ (c2.1 - 1)^2 + (c2.2 - 3)^2 = r2^2 ∧
  (c1.1 - m)^2 + (c1.2 - n)^2 = r1^2 ∧ (c2.1 - m)^2 + (c2.2 - n)^2 = r2^2) :
  m + n = 4 :=
sorry

end intersecting_circles_l726_72619


namespace total_seats_in_theater_l726_72633

def theater_charges_adults : ℝ := 3.0
def theater_charges_children : ℝ := 1.5
def total_income : ℝ := 510
def number_of_children : ℕ := 60

theorem total_seats_in_theater :
  ∃ (A C : ℕ), C = number_of_children ∧ theater_charges_adults * A + theater_charges_children * C = total_income ∧ A + C = 200 :=
by
  sorry

end total_seats_in_theater_l726_72633


namespace A_inter_B_eq_l726_72600

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 > 1}

theorem A_inter_B_eq : A ∩ B = {-2, 2} := 
by
  sorry

end A_inter_B_eq_l726_72600


namespace f_one_equals_half_f_increasing_l726_72659

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_half (x y : ℝ) : f (x + y) = f x + f y + 1/2

axiom f_half     : f (1/2) = 0

axiom f_positive (x : ℝ) (hx : x > 1/2) : f x > 0

theorem f_one_equals_half : f 1 = 1/2 := 
by 
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2 := 
by 
  sorry

end f_one_equals_half_f_increasing_l726_72659


namespace number_of_fiction_books_l726_72675

theorem number_of_fiction_books (F NF : ℕ) (h1 : F + NF = 52) (h2 : NF = 7 * F / 6) : F = 24 := 
by
  sorry

end number_of_fiction_books_l726_72675


namespace repeating_decimal_fraction_l726_72609

noncomputable def repeating_decimal := 4.66666 -- Assuming repeating forever

theorem repeating_decimal_fraction : repeating_decimal = 14 / 3 :=
by 
  sorry

end repeating_decimal_fraction_l726_72609


namespace inequality_bi_l726_72688

variable {α : Type*} [LinearOrderedField α]

-- Sequence of positive real numbers
variable (a : ℕ → α)
-- Conditions for a_i
variable (ha : ∀ i, i > 0 → i * (a i)^2 ≥ (i + 1) * a (i - 1) * a (i + 1))
-- Positive real numbers x and y
variables (x y : α) (hx : x > 0) (hy : y > 0)
-- Definition of b_i
def b (i : ℕ) : α := x * a i + y * a (i - 1)

theorem inequality_bi (i : ℕ) (hi : i ≥ 2) : i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := 
sorry

end inequality_bi_l726_72688


namespace same_terminal_side_l726_72694

theorem same_terminal_side (θ : ℝ) : (∃ k : ℤ, θ = 2 * k * π - π / 6) → θ = 11 * π / 6 :=
sorry

end same_terminal_side_l726_72694


namespace winning_candidate_votes_l726_72689

theorem winning_candidate_votes  (V W : ℝ) (hW : W = 0.5666666666666664 * V) (hV : V = W + 7636 + 11628) : 
  W = 25216 := 
by 
  sorry

end winning_candidate_votes_l726_72689


namespace find_other_number_l726_72611

theorem find_other_number (a b : ℕ) (gcd_ab : Nat.gcd a b = 45) (lcm_ab : Nat.lcm a b = 1260) (a_eq : a = 180) : b = 315 :=
by
  -- proof goes here
  sorry

end find_other_number_l726_72611


namespace parabolas_pass_through_origin_l726_72606

-- Definition of a family of parabolas
def parabola_family (p q : ℝ) (x : ℝ) : ℝ := -x^2 + p * x + q

-- Definition of vertices lying on y = x^2
def vertex_condition (p q : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = -a^2 + p * a + q)

-- Proving that all such parabolas pass through the point (0, 0)
theorem parabolas_pass_through_origin :
  ∀ (p q : ℝ), vertex_condition p q → parabola_family p q 0 = 0 :=
by
  sorry

end parabolas_pass_through_origin_l726_72606


namespace proof_problem_l726_72610

noncomputable def M : ℕ := 50
noncomputable def T : ℕ := M + Nat.div M 10
noncomputable def W : ℕ := 2 * (M + T)
noncomputable def Th : ℕ := W / 2
noncomputable def total_T_T_W_Th : ℕ := T + W + Th
noncomputable def total_M_T_W_Th : ℕ := M + total_T_T_W_Th
noncomputable def F_S_sun : ℕ := Nat.div (450 - total_M_T_W_Th) 3
noncomputable def car_tolls : ℕ := 150 * 2
noncomputable def bus_tolls : ℕ := 150 * 5
noncomputable def truck_tolls : ℕ := 150 * 10
noncomputable def total_toll : ℕ := car_tolls + bus_tolls + truck_tolls

theorem proof_problem :
  (total_T_T_W_Th = 370) ∧
  (F_S_sun = 10) ∧
  (total_toll = 2550) := by
  sorry

end proof_problem_l726_72610


namespace mul_mod_correct_l726_72617

theorem mul_mod_correct :
  (2984 * 3998) % 1000 = 32 :=
by
  sorry

end mul_mod_correct_l726_72617


namespace combination_add_l726_72681

def combination (n m : ℕ) : ℕ := n.choose m

theorem combination_add {n : ℕ} (h1 : 4 ≤ 9) (h2 : 5 ≤ 9) :
  combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end combination_add_l726_72681


namespace painting_time_l726_72624

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end painting_time_l726_72624


namespace sum_of_possible_amounts_l726_72684

-- Definitions based on conditions:
def possible_quarters_amounts : Finset ℕ := {5, 30, 55, 80}
def possible_dimes_amounts : Finset ℕ := {15, 20, 30, 35, 40, 50, 60, 70, 80, 90}
def both_possible_amounts : Finset ℕ := possible_quarters_amounts ∩ possible_dimes_amounts

-- Statement of the problem:
theorem sum_of_possible_amounts : (both_possible_amounts.sum id) = 110 :=
by
  sorry

end sum_of_possible_amounts_l726_72684


namespace log_ab_is_pi_l726_72687

open Real

noncomputable def log_ab (a b : ℝ) : ℝ :=
(log b) / (log a)

theorem log_ab_is_pi (a b : ℝ)  (ha_pos: 0 < a) (ha_ne_one: a ≠ 1) (hb_pos: 0 < b) 
  (cond1 : log (a ^ 3) = log (b ^ 6)) (cond2 : cos (π * log a) = 1) : log_ab a b = π :=
by
  sorry

end log_ab_is_pi_l726_72687


namespace sophie_aunt_money_l726_72660

noncomputable def totalMoneyGiven (shirts: ℕ) (shirtCost: ℝ) (trousers: ℕ) (trouserCost: ℝ) (additionalItems: ℕ) (additionalItemCost: ℝ) : ℝ :=
  shirts * shirtCost + trousers * trouserCost + additionalItems * additionalItemCost

theorem sophie_aunt_money : totalMoneyGiven 2 18.50 1 63 4 40 = 260 := 
by
  sorry

end sophie_aunt_money_l726_72660


namespace initial_investment_C_l726_72608

def total_investment : ℝ := 425
def increase_A (a : ℝ) : ℝ := 0.05 * a
def increase_B (b : ℝ) : ℝ := 0.08 * b
def increase_C (c : ℝ) : ℝ := 0.10 * c

theorem initial_investment_C (a b c : ℝ) (h1 : a + b + c = total_investment)
  (h2 : increase_A a = increase_B b) (h3 : increase_B b = increase_C c) : c = 100 := by
  sorry

end initial_investment_C_l726_72608


namespace complex_numbers_right_triangle_l726_72613

theorem complex_numbers_right_triangle (z : ℂ) (hz : z ≠ 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ 0 ∧ z₂ ≠ 0 ∧ z₁^3 = z₂ ∧
                 (∃ θ₁ θ₂ : ℝ, z₁ = Complex.exp (Complex.I * θ₁) ∧
                               z₂ = Complex.exp (Complex.I * θ₂) ∧
                               (θ₂ - θ₁ = π/2 ∨ θ₂ - θ₁ = 3 * π/2))) →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end complex_numbers_right_triangle_l726_72613


namespace min_M_value_l726_72685

variable {a b c t : ℝ}

theorem min_M_value (h1 : a < b)
                    (h2 : a > 0)
                    (h3 : b^2 - 4 * a * c ≤ 0)
                    (h4 : b = t + a)
                    (h5 : t > 0)
                    (h6 : c ≥ (t + a)^2 / (4 * a)) :
    ∃ M : ℝ, (∀ x : ℝ, (a * x^2 + b * x + c) ≥ 0) → M = 3 := 
  sorry

end min_M_value_l726_72685


namespace functional_relationship_l726_72632

-- Define the conditions
def directlyProportional (y x k : ℝ) : Prop :=
  y + 6 = k * (x + 1)

def specificCondition1 (x y : ℝ) : Prop :=
  x = 3 ∧ y = 2

-- State the theorem
theorem functional_relationship (k : ℝ) :
  (∀ x y, directlyProportional y x k) →
  specificCondition1 3 2 →
  ∀ x, ∃ y, y = 2 * x - 4 :=
by
  intro directProp
  intro specCond
  sorry

end functional_relationship_l726_72632


namespace power_summation_l726_72645

theorem power_summation :
  (-1:ℤ)^(49) + (2:ℝ)^(3^3 + 5^2 - 48^2) = -1 + 1 / 2 ^ (2252 : ℝ) :=
by
  sorry

end power_summation_l726_72645


namespace consecutive_diff_possible_l726_72695

variable (a b c : ℝ)

def greater_than_2022 :=
  a > 2022 ∨ b > 2022 ∨ c > 2022

def distinct_numbers :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem consecutive_diff_possible :
  greater_than_2022 a b c → distinct_numbers a b c → 
  ∃ (x y z : ℤ), x + 1 = y ∧ y + 1 = z ∧ 
  (a^2 - b^2 = ↑x) ∧ (b^2 - c^2 = ↑y) ∧ (c^2 - a^2 = ↑z) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end consecutive_diff_possible_l726_72695


namespace div_relation_l726_72693

theorem div_relation (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 3) : c / a = 1 / 2 := 
by 
  sorry

end div_relation_l726_72693


namespace range_of_m_l726_72672

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x < y → -3 < x ∧ y < 3 → f x < f y)
  (h2 : ∀ m : ℝ, f (2 * m) < f (m + 1)) : 
  -3/2 < m ∧ m < 1 :=
  sorry

end range_of_m_l726_72672


namespace return_trip_time_l726_72696

-- conditions 
variables (d p w : ℝ) (h1 : d = 90 * (p - w)) (h2 : ∀ t : ℝ, t = d / p → d / (p + w) = t - 15)

--  statement
theorem return_trip_time :
  ∃ t : ℝ, t = 30 ∨ t = 45 :=
by
  -- placeholder proof 
  sorry

end return_trip_time_l726_72696


namespace simplify_expression_l726_72683

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) :
  ( ( (p - 7) / (9 - r) ) * ( (q - 8) / (7 - p) ) * ( (r - 9) / (8 - q) ) ) = -1 := 
by 
  sorry

end simplify_expression_l726_72683


namespace hyperbola_condition_l726_72647

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m + 2)) + (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_condition_l726_72647


namespace double_persons_half_work_l726_72682

theorem double_persons_half_work :
  (∀ (n : ℕ) (d : ℕ), d = 12 → (2 * n) * (d / 2) = n * 3) :=
by
  sorry

end double_persons_half_work_l726_72682


namespace tangent_line_at_1_l726_72625

-- Assume the curve and the point of tangency
noncomputable def curve (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

-- Define the point of tangency
def point_of_tangency : ℝ := 1

-- Define the expected tangent line equation in standard form Ax + By + C = 0
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 5 = 0

theorem tangent_line_at_1 :
  tangent_line point_of_tangency (curve point_of_tangency) := 
sorry

end tangent_line_at_1_l726_72625


namespace recurring_division_l726_72634

def recurring_to_fraction (recurring: ℝ) (part: ℝ): ℝ :=
  part * recurring

theorem recurring_division (recurring: ℝ) (part1 part2: ℝ):
  recurring_to_fraction recurring part1 = 0.63 →
  recurring_to_fraction recurring part2 = 0.18 →
  recurring ≠ 0 →
  (0.63:ℝ)/0.18 = (7:ℝ)/2 :=
by
  intros h1 h2 h3
  rw [recurring_to_fraction] at h1 h2
  sorry

end recurring_division_l726_72634


namespace trigonometric_identity_solution_l726_72658

theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 ↔
  ∃ (k : ℤ), x = Real.pi + 2 * Real.pi * k := by
  sorry

end trigonometric_identity_solution_l726_72658


namespace discount_percentage_l726_72603

theorem discount_percentage (original_price sale_price : ℝ) (h1 : original_price = 150) (h2 : sale_price = 135) : 
  (original_price - sale_price) / original_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l726_72603


namespace amount_spent_on_raw_materials_l726_72641

-- Given conditions
def spending_on_machinery : ℝ := 125
def spending_as_cash (total_amount : ℝ) : ℝ := 0.10 * total_amount
def total_amount : ℝ := 250

-- Mathematically equivalent problem
theorem amount_spent_on_raw_materials :
  (X : ℝ) → X + spending_on_machinery + spending_as_cash total_amount = total_amount →
    X = 100 :=
by
  (intro X h)
  sorry

end amount_spent_on_raw_materials_l726_72641


namespace divisible_by_5_l726_72631

theorem divisible_by_5 (n : ℕ) : (∃ k : ℕ, 2^n - 1 = 5 * k) ∨ (∃ k : ℕ, 2^n + 1 = 5 * k) ∨ (∃ k : ℕ, 2^(2*n) + 1 = 5 * k) :=
sorry

end divisible_by_5_l726_72631


namespace solve_for_z_l726_72665

theorem solve_for_z (z i : ℂ) (h1 : 1 - i*z + 3*i = -1 + i*z + 3*i) (h2 : i^2 = -1) : z = -i := 
  sorry

end solve_for_z_l726_72665


namespace min_value_of_2a_plus_b_l726_72629

variable (a b : ℝ)

def condition := a > 0 ∧ b > 0 ∧ a - 2 * a * b + b = 0

-- Define what needs to be proved
theorem min_value_of_2a_plus_b (h : condition a b) : ∃ a b : ℝ, 2 * a + b = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_of_2a_plus_b_l726_72629


namespace probability_of_head_equal_half_l726_72640

def fair_coin_probability : Prop :=
  ∀ (H T : ℕ), (H = 1 ∧ T = 1 ∧ (H + T = 2)) → ((H / (H + T)) = 1 / 2)

theorem probability_of_head_equal_half : fair_coin_probability :=
sorry

end probability_of_head_equal_half_l726_72640


namespace line_condition_l726_72680

variable (m n Q : ℝ)

theorem line_condition (h1: m = 8 * n + 5) 
                       (h2: m + Q = 8 * (n + 0.25) + 5) 
                       (h3: p = 0.25) : Q = 2 :=
by
  sorry

end line_condition_l726_72680


namespace min_reciprocal_sum_l726_72664

theorem min_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a) + (1 / b) ≥ 2 := by
  sorry

end min_reciprocal_sum_l726_72664


namespace inequality_inequality_must_be_true_l726_72691

variables {a b c d : ℝ}

theorem inequality_inequality_must_be_true
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (a / d) < (b / c) :=
sorry

end inequality_inequality_must_be_true_l726_72691


namespace find_range_of_a_l726_72657

noncomputable def value_range_for_a : Set ℝ := {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4}

theorem find_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0)  ∧
  (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0) ∧
  (¬ (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0) → ¬ (∃ x : ℝ, x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0))
  → a ∈ value_range_for_a :=
sorry

end find_range_of_a_l726_72657


namespace cos_double_angle_l726_72669

theorem cos_double_angle (α β : Real) 
    (h1 : Real.sin α = Real.cos β) 
    (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1 / 2) :
    Real.cos (2 * β) = 2 / 3 :=
by
  sorry

end cos_double_angle_l726_72669


namespace increasing_quadratic_l726_72615

noncomputable def f (a x : ℝ) : ℝ := 3 * x^2 - a * x + 4

theorem increasing_quadratic {a : ℝ} :
  (∀ x ≥ -5, 6 * x - a ≥ 0) ↔ a ≤ -30 :=
by
  sorry

end increasing_quadratic_l726_72615


namespace consecutive_even_numbers_divisible_by_384_l726_72605

theorem consecutive_even_numbers_divisible_by_384 (n : Nat) (h1 : n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) = 384) : n = 6 :=
sorry

end consecutive_even_numbers_divisible_by_384_l726_72605


namespace no_all_same_color_l726_72674

def chameleons_initial_counts (c b m : ℕ) : Prop :=
  c = 13 ∧ b = 15 ∧ m = 17

def chameleon_interaction (c b m : ℕ) : Prop :=
  (∃ c' b' m', c' + b' + m' = c + b + m ∧ 
  ((c' = c - 1 ∧ b' = b - 1 ∧ m' = m + 2) ∨
   (c' = c - 1 ∧ b' = b + 2 ∧ m' = m - 1) ∨
   (c' = c + 2 ∧ b' = b - 1 ∧ m' = m - 1)))

theorem no_all_same_color (c b m : ℕ) (h1 : chameleons_initial_counts c b m) : 
  ¬ (∃ x, c = x ∧ b = x ∧ m = x) := 
sorry

end no_all_same_color_l726_72674


namespace smallest_diff_PR_PQ_l726_72622

theorem smallest_diff_PR_PQ (PQ PR QR : ℤ) (h1 : PQ < PR) (h2 : PR ≤ QR) (h3 : PQ + PR + QR = 2021) : 
  ∃ PQ PR QR : ℤ, PQ < PR ∧ PR ≤ QR ∧ PQ + PR + QR = 2021 ∧ PR - PQ = 1 :=
by
  sorry

end smallest_diff_PR_PQ_l726_72622


namespace candidate_D_votes_l726_72616

theorem candidate_D_votes :
  let total_votes := 10000
  let invalid_votes_percentage := 0.25
  let valid_votes := (1 - invalid_votes_percentage) * total_votes
  let candidate_A_percentage := 0.40
  let candidate_B_percentage := 0.30
  let candidate_C_percentage := 0.20
  let candidate_D_percentage := 1.0 - (candidate_A_percentage + candidate_B_percentage + candidate_C_percentage)
  let candidate_D_votes := candidate_D_percentage * valid_votes
  candidate_D_votes = 750 :=
by
  sorry

end candidate_D_votes_l726_72616


namespace solution_set_inequalities_l726_72638

theorem solution_set_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end solution_set_inequalities_l726_72638


namespace value_not_uniquely_determined_l726_72656

variables (v : Fin 9 → ℤ) (s : Fin 9 → ℤ)

-- Given conditions
axiom announced_sums : ∀ i, s i = v ((i - 1) % 9) + v ((i + 1) % 9)
axiom sums_sequence : s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 12 ∧ s 3 = 18 ∧ s 4 = 24 ∧ s 5 = 31 ∧ s 6 = 40 ∧ s 7 = 48 ∧ s 8 = 53

-- Statement asserting the indeterminacy of v_5
theorem value_not_uniquely_determined (h: s 3 = 18) : 
  ∃ v : Fin 9 → ℤ, sorry :=
sorry

end value_not_uniquely_determined_l726_72656


namespace smallest_special_number_gt_3429_l726_72635

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l726_72635


namespace twenty_four_x_eq_a_cubed_t_l726_72699

-- Define conditions
variables {x : ℝ} {a t : ℝ}
axiom h1 : 2^x = a
axiom h2 : 3^x = t

-- State the theorem
theorem twenty_four_x_eq_a_cubed_t : 24^x = a^3 * t := 
by sorry

end twenty_four_x_eq_a_cubed_t_l726_72699


namespace minimum_value_of_expression_l726_72670

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  a^2 + b^2 + (a + b)^2 + c^2

theorem minimum_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  min_value_expression a b c = 9 :=
  sorry

end minimum_value_of_expression_l726_72670


namespace Kyle_age_l726_72646

-- Let's define the variables for each person's age.
variables (Shelley Kyle Julian Frederick Tyson Casey Sandra David Fiona : ℕ) 

-- Defining conditions based on given problem.
axiom condition1 : Shelley = Kyle - 3
axiom condition2 : Shelley = Julian + 4
axiom condition3 : Julian = Frederick - 20
axiom condition4 : Julian = Fiona + 5
axiom condition5 : Frederick = 2 * Tyson
axiom condition6 : Tyson = 2 * Casey
axiom condition7 : Casey = Fiona - 2
axiom condition8 : Casey = Sandra / 2
axiom condition9 : Sandra = David + 4
axiom condition10 : David = 16

-- The goal is to prove Kyle's age is 23 years old.
theorem Kyle_age : Kyle = 23 :=
by sorry

end Kyle_age_l726_72646


namespace smallest_integer_y_l726_72673

theorem smallest_integer_y : ∃ (y : ℤ), (7 + 3 * y < 25) ∧ (∀ z : ℤ, (7 + 3 * z < 25) → y ≤ z) ∧ y = 5 :=
by
  sorry

end smallest_integer_y_l726_72673


namespace rational_coordinates_l726_72602

theorem rational_coordinates (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 :=
by
  use (1 - x)
  sorry

end rational_coordinates_l726_72602


namespace seminar_duration_total_l726_72618

/-- The first part of the seminar lasted 4 hours and 45 minutes -/
def first_part_minutes := 4 * 60 + 45

/-- The second part of the seminar lasted 135 minutes -/
def second_part_minutes := 135

/-- The closing event lasted 500 seconds -/
def closing_event_minutes := 500 / 60

/-- The total duration of the seminar session in minutes, including the closing event, is 428 minutes -/
theorem seminar_duration_total :
  first_part_minutes + second_part_minutes + closing_event_minutes = 428 := by
  sorry

end seminar_duration_total_l726_72618


namespace perp_lines_a_value_l726_72644

theorem perp_lines_a_value :
  ∀ a : ℝ, ((a + 1) * 1 - 2 * (-a) = 0) → a = 1 :=
by
  intro a
  intro h
  -- We now state that a must satisfy the given condition and show that this leads to a = 1
  -- The proof is left as sorry
  sorry

end perp_lines_a_value_l726_72644


namespace equation_solution_unique_l726_72637

theorem equation_solution_unique (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
    (2 / (x - 3) = 3 / x ↔ x = 9) :=
by
  sorry

end equation_solution_unique_l726_72637


namespace gcd_of_repeated_three_digit_number_is_constant_l726_72614

theorem gcd_of_repeated_three_digit_number_is_constant (m : ℕ) (h1 : 100 ≤ m) (h2 : m < 1000) : 
  ∃ d, d = 1001001 ∧ ∀ n, n = 10010013 * m → (gcd 1001001 n) = 1001001 :=
by
  -- The proof would go here
  sorry

end gcd_of_repeated_three_digit_number_is_constant_l726_72614


namespace c_value_l726_72678

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end c_value_l726_72678


namespace meet_floor_l726_72636

noncomputable def xiaoming_meets_xiaoying (x y meet_floor: ℕ) : Prop :=
  x = 4 → y = 3 → (meet_floor = 22)

theorem meet_floor (x y meet_floor: ℕ) (h1: x = 4) (h2: y = 3) :
  xiaoming_meets_xiaoying x y meet_floor :=
by
  sorry

end meet_floor_l726_72636


namespace distinct_values_for_T_l726_72654

-- Define the conditions given in the problem:
def distinct_digits (n : ℕ) : Prop :=
  n / 1000 ≠ (n / 100 % 10) ∧ n / 1000 ≠ (n / 10 % 10) ∧ n / 1000 ≠ (n % 10) ∧
  (n / 100 % 10) ≠ (n / 10 % 10) ∧ (n / 100 % 10) ≠ (n % 10) ∧
  (n / 10 % 10) ≠ (n % 10)

def Psum (P S T : ℕ) : Prop := P + S = T

-- Main theorem statement:
theorem distinct_values_for_T : ∀ (P S T : ℕ),
  distinct_digits P ∧ distinct_digits S ∧ distinct_digits T ∧
  Psum P S T → 
  (∃ (values : Finset ℕ), values.card = 7 ∧ ∀ val ∈ values, val = T) :=
by
  sorry

end distinct_values_for_T_l726_72654


namespace sum_of_remainders_is_six_l726_72628

def sum_of_remainders (n : ℕ) : ℕ :=
  n % 4 + (n + 1) % 4 + (n + 2) % 4 + (n + 3) % 4

theorem sum_of_remainders_is_six : ∀ n : ℕ, sum_of_remainders n = 6 :=
by
  intro n
  sorry

end sum_of_remainders_is_six_l726_72628


namespace smallest_n_l726_72650

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end smallest_n_l726_72650


namespace distance_to_parabola_focus_l726_72639

theorem distance_to_parabola_focus :
  ∀ (x : ℝ), ((4 : ℝ) = (1 / 4) * x^2) → dist (0, 4) (0, 5) = 5 := 
by
  intro x
  intro hyp
  -- initial conditions indicate the distance is 5 and can be directly given
  sorry

end distance_to_parabola_focus_l726_72639


namespace smallest_cookies_left_l726_72667

theorem smallest_cookies_left (m : ℤ) (h : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end smallest_cookies_left_l726_72667
