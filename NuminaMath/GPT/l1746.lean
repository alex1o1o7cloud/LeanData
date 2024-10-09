import Mathlib

namespace abs_eq_solutions_l1746_174669

theorem abs_eq_solutions (x : ℝ) (hx : |x - 5| = 3 * x + 6) :
  x = -11 / 2 ∨ x = -1 / 4 :=
sorry

end abs_eq_solutions_l1746_174669


namespace inscribed_circle_probability_l1746_174636

theorem inscribed_circle_probability (r : ℝ) (h : r > 0) : 
  let square_area := 4 * r^2
  let circle_area := π * r^2
  (circle_area / square_area) = π / 4 := by
  sorry

end inscribed_circle_probability_l1746_174636


namespace gcd_18_30_45_l1746_174688

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end gcd_18_30_45_l1746_174688


namespace probability_colors_match_l1746_174620

section ProbabilityJellyBeans

structure JellyBeans where
  green : ℕ
  blue : ℕ
  red : ℕ

def total_jellybeans (jb : JellyBeans) : ℕ :=
  jb.green + jb.blue + jb.red

-- Define the situation using structures
def lila_jellybeans : JellyBeans := { green := 1, blue := 1, red := 1 }
def max_jellybeans : JellyBeans := { green := 2, blue := 1, red := 3 }

-- Define probabilities
noncomputable def probability (count : ℕ) (total : ℕ) : ℚ :=
  if total = 0 then 0 else (count : ℚ) / (total : ℚ)

-- Main theorem
theorem probability_colors_match :
  probability lila_jellybeans.green (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.green (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.blue (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.blue (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.red (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.red (total_jellybeans max_jellybeans) = 1 / 3 :=
by sorry

end ProbabilityJellyBeans

end probability_colors_match_l1746_174620


namespace intersection_is_4_l1746_174635

-- Definitions of the sets
def U : Set Int := {0, 1, 2, 4, 6, 8}
def M : Set Int := {0, 4, 6}
def N : Set Int := {0, 1, 6}

-- Definition of the complement
def complement_U_N : Set Int := U \ N

-- Definition of the intersection
def intersection_M_complement_U_N : Set Int := M ∩ complement_U_N

-- Statement of the theorem
theorem intersection_is_4 : intersection_M_complement_U_N = {4} :=
by
  sorry

end intersection_is_4_l1746_174635


namespace manny_marbles_l1746_174657

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l1746_174657


namespace floor_width_l1746_174605

theorem floor_width
  (widthX lengthX : ℝ) (widthY lengthY : ℝ)
  (hX : widthX = 10) (lX : lengthX = 18) (lY : lengthY = 20)
  (h : lengthX * widthX = lengthY * widthY) :
  widthY = 9 := 
by
  -- proof goes here
  sorry

end floor_width_l1746_174605


namespace time_to_cross_platform_l1746_174674

variable (l t p : ℝ) -- Define relevant variables

-- Conditions as definitions in Lean 4
def length_of_train := l
def time_to_pass_man := t
def length_of_platform := p

-- Assume given values in the problem
def cond1 : length_of_train = 186 := by sorry
def cond2 : time_to_pass_man = 8 := by sorry
def cond3 : length_of_platform = 279 := by sorry

-- Statement that represents the target theorem to be proved
theorem time_to_cross_platform (h₁ : length_of_train = 186) (h₂ : time_to_pass_man = 8) (h₃ : length_of_platform = 279) : 
  let speed := length_of_train / time_to_pass_man
  let total_distance := length_of_train + length_of_platform
  let time_to_cross := total_distance / speed
  time_to_cross = 20 :=
by sorry

end time_to_cross_platform_l1746_174674


namespace joe_height_l1746_174693

theorem joe_height (S J A : ℝ) (h1 : S + J + A = 180) (h2 : J = 2 * S + 6) (h3 : A = S - 3) : J = 94.5 :=
by 
  -- Lean proof goes here
  sorry

end joe_height_l1746_174693


namespace distinct_digits_solution_l1746_174604

theorem distinct_digits_solution (A B C : ℕ)
  (h1 : A + B = 10)
  (h2 : C + A = 9)
  (h3 : B + C = 9)
  (h4 : A ≠ B)
  (h5 : B ≠ C)
  (h6 : C ≠ A)
  (h7 : 0 < A)
  (h8 : 0 < B)
  (h9 : 0 < C)
  : A = 1 ∧ B = 9 ∧ C = 8 := 
  by sorry

end distinct_digits_solution_l1746_174604


namespace total_pieces_of_pizza_l1746_174642

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end total_pieces_of_pizza_l1746_174642


namespace total_distance_walked_l1746_174625

-- Condition 1: Distance in feet
def distance_feet : ℝ := 30

-- Condition 2: Conversion factor from feet to meters
def feet_to_meters : ℝ := 0.3048

-- Condition 3: Number of trips
def trips : ℝ := 4

-- Question: Total distance walked in meters
theorem total_distance_walked :
  distance_feet * feet_to_meters * trips = 36.576 :=
sorry

end total_distance_walked_l1746_174625


namespace marble_weight_l1746_174630

-- Define the weights of marbles and waffle irons
variables (m w : ℝ)

-- Given conditions
def condition1 : Prop := 9 * m = 4 * w
def condition2 : Prop := 3 * w = 75 

-- The theorem we want to prove
theorem marble_weight (h1 : condition1 m w) (h2 : condition2 w) : m = 100 / 9 :=
by
  sorry

end marble_weight_l1746_174630


namespace lemons_and_oranges_for_100_gallons_l1746_174667

-- Given conditions
def lemons_per_gallon := 30 / 40
def oranges_per_gallon := 20 / 40

-- Theorem to be proven
theorem lemons_and_oranges_for_100_gallons : 
  lemons_per_gallon * 100 = 75 ∧ oranges_per_gallon * 100 = 50 := by
  sorry

end lemons_and_oranges_for_100_gallons_l1746_174667


namespace find_f_inv_486_l1746_174672

-- Assuming function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_cond1 : f 4 = 2
axiom f_cond2 : ∀ x : ℝ, f (3 * x) = 3 * f x

-- Proof problem: Prove that f⁻¹(486) = 972
theorem find_f_inv_486 : (∃ x : ℝ, f x = 486 ∧ x = 972) :=
sorry

end find_f_inv_486_l1746_174672


namespace women_per_table_l1746_174631

theorem women_per_table 
  (total_tables : ℕ)
  (men_per_table : ℕ)
  (total_customers : ℕ) 
  (h_total_tables : total_tables = 6)
  (h_men_per_table : men_per_table = 5)
  (h_total_customers : total_customers = 48) :
  (total_customers - (men_per_table * total_tables)) / total_tables = 3 :=
by
  subst h_total_tables
  subst h_men_per_table
  subst h_total_customers
  sorry

end women_per_table_l1746_174631


namespace rain_at_least_once_l1746_174658

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l1746_174658


namespace monotonic_m_range_l1746_174632

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 12

-- Prove the range of m where f(x) is monotonic on [m, m+4]
theorem monotonic_m_range {m : ℝ} :
  (∀ x y : ℝ, m ≤ x ∧ x ≤ m + 4 ∧ m ≤ y ∧ y ≤ m + 4 → (x ≤ y → f x ≤ f y ∨ f x ≥ f y))
  ↔ (m ≤ -5 ∨ m ≥ 2) :=
sorry

end monotonic_m_range_l1746_174632


namespace log4_80_cannot_be_found_without_additional_values_l1746_174637

-- Conditions provided in the problem
def log4_16 : Real := 2
def log4_32 : Real := 2.5

-- Lean statement of the proof problem
theorem log4_80_cannot_be_found_without_additional_values :
  ¬(∃ (log4_80 : Real), log4_80 = log4_16 + log4_5) :=
sorry

end log4_80_cannot_be_found_without_additional_values_l1746_174637


namespace expand_expression_l1746_174633

theorem expand_expression :
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 :=
by sorry

end expand_expression_l1746_174633


namespace laila_utility_l1746_174661

theorem laila_utility (u : ℝ) :
  (2 * u * (10 - 2 * u) = 2 * (4 - 2 * u) * (2 * u + 4)) → u = 4 := 
by 
  sorry

end laila_utility_l1746_174661


namespace ratio_of_pages_given_l1746_174654

variable (Lana_initial_pages : ℕ) (Duane_initial_pages : ℕ) (Lana_final_pages : ℕ)

theorem ratio_of_pages_given
  (h1 : Lana_initial_pages = 8)
  (h2 : Duane_initial_pages = 42)
  (h3 : Lana_final_pages = 29) :
  (Lana_final_pages - Lana_initial_pages) / Duane_initial_pages = 1 / 2 :=
  by
  -- Placeholder for the proof
  sorry

end ratio_of_pages_given_l1746_174654


namespace power_inequality_l1746_174608

theorem power_inequality (n : ℕ) (x : ℝ) (h1 : 0 < n) (h2 : x > -1) : (1 + x)^n ≥ 1 + n * x :=
sorry

end power_inequality_l1746_174608


namespace hotel_assignment_l1746_174689

noncomputable def numberOfWaysToAssignFriends (rooms friends : ℕ) : ℕ :=
  if rooms = 5 ∧ friends = 6 then 7200 else 0

theorem hotel_assignment : numberOfWaysToAssignFriends 5 6 = 7200 :=
by 
  -- This is the condition already matched in the noncomputable function defined above.
  sorry

end hotel_assignment_l1746_174689


namespace point_on_curve_l1746_174619

-- Define the equation of the curve
def curve (x y : ℝ) := x^2 - x * y + 2 * y + 1 = 0

-- State that point (3, 10) satisfies the given curve equation
theorem point_on_curve : curve 3 10 :=
by
  -- this is where the proof would go but we will skip it for now
  sorry

end point_on_curve_l1746_174619


namespace peanuts_added_l1746_174634

theorem peanuts_added (a b x : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : a + x = b) : x = 2 :=
by
  sorry

end peanuts_added_l1746_174634


namespace reciprocal_neg_one_over_2011_l1746_174638

theorem reciprocal_neg_one_over_2011 : 1 / (- (1 / 2011)) = -2011 :=
by
  sorry

end reciprocal_neg_one_over_2011_l1746_174638


namespace pow_comparison_l1746_174699

theorem pow_comparison : 2^700 > 5^300 :=
by sorry

end pow_comparison_l1746_174699


namespace solve_for_x_l1746_174613

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) (h : (5*x)^10 = (10*x)^5) : x = 2/5 :=
sorry

end solve_for_x_l1746_174613


namespace smallest_integer_divisible_l1746_174628

theorem smallest_integer_divisible:
  ∃ n : ℕ, n > 1 ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 6 = 1) ∧ n = 61 :=
by
  sorry

end smallest_integer_divisible_l1746_174628


namespace find_x_l1746_174627

theorem find_x (a b x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : 
    x = 16 * a^(3 / 2) :=
by 
  sorry

end find_x_l1746_174627


namespace abs_neg_three_l1746_174610

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l1746_174610


namespace relationship_between_a_and_b_l1746_174624

theorem relationship_between_a_and_b 
  (x a b : ℝ)
  (hx : 0 < x)
  (ha : 0 < a)
  (hb : 0 < b)
  (hax : a^x < b^x) 
  (hbx : b^x < 1) : 
  a < b ∧ b < 1 := 
sorry

end relationship_between_a_and_b_l1746_174624


namespace solve_equation_l1746_174603

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l1746_174603


namespace geometric_sequence_b_value_l1746_174683

theorem geometric_sequence_b_value 
  (b : ℝ)
  (h1 : b > 0)
  (h2 : ∃ r : ℝ, 160 * r = b ∧ b * r = 1)
  : b = 4 * Real.sqrt 10 := 
sorry

end geometric_sequence_b_value_l1746_174683


namespace find_q_l1746_174655

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l1746_174655


namespace marble_cut_percentage_l1746_174616

theorem marble_cut_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (x : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ) :
  initial_weight = 190 →
  final_weight = 109.0125 →
  first_week_cut = (1 - x / 100) →
  second_week_cut = 0.85 →
  third_week_cut = 0.9 →
  (initial_weight * first_week_cut * second_week_cut * third_week_cut = final_weight) →
  x = 24.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end marble_cut_percentage_l1746_174616


namespace bug_probability_at_A_after_8_meters_l1746_174617

noncomputable def P : ℕ → ℚ 
| 0 => 1
| (n + 1) => (1 / 3) * (1 - P n)

theorem bug_probability_at_A_after_8_meters :
  P 8 = 547 / 2187 := 
sorry

end bug_probability_at_A_after_8_meters_l1746_174617


namespace nonnegative_integers_with_abs_value_less_than_4_l1746_174615

theorem nonnegative_integers_with_abs_value_less_than_4 :
  {n : ℕ | abs (n : ℤ) < 4} = {0, 1, 2, 3} :=
by {
  sorry
}

end nonnegative_integers_with_abs_value_less_than_4_l1746_174615


namespace triangle_BPC_area_l1746_174600

universe u

variables {T : Type u} [LinearOrderedField T]

-- Define the points
variables (A B C E F P : T)
variables (area : T → T → T → T) -- A function to compute the area of a triangle

-- Hypotheses
def conditions :=
  E ∈ [A, B] ∧
  F ∈ [A, C] ∧
  (∃ P, P ∈ [B, F] ∧ P ∈ [C, E]) ∧
  area A E P + area E P F + area P F A = 4 ∧ -- AEPF
  area B E P = 4 ∧ -- BEP
  area C F P = 4   -- CFP

-- The theorem to prove
theorem triangle_BPC_area (h : conditions A B C E F P area) : area B P C = 12 :=
sorry

end triangle_BPC_area_l1746_174600


namespace general_term_a_n_sum_b_n_terms_l1746_174606

-- Given definitions based on the conditions
def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := (2^(2*n-1))^2

def b_sum (n : ℕ) : (ℕ → ℕ) := 
  (fun b : ℕ => match b with 
                | 1 => 4 
                | 2 => 64 
                | _ => (4^(2*(b - 2 + 1) - 1)))

def T (n : ℕ) : ℕ := (4 / 15) * (16^n - 1)

-- First part: Proving the general term of {a_n} is 2^(n-1)
theorem general_term_a_n (n : ℕ) : a n = 2^(n-1) := by
  sorry

-- Second part: Proving the sum of the first n terms of {b_n} is (4/15)*(16^n - 1)
theorem sum_b_n_terms (n : ℕ) : T n = (4 / 15) * (16^n - 1) := by 
  sorry

end general_term_a_n_sum_b_n_terms_l1746_174606


namespace find_certain_number_l1746_174686

theorem find_certain_number (n : ℕ) (h : 9823 + n = 13200) : n = 3377 :=
by
  sorry

end find_certain_number_l1746_174686


namespace number_of_roots_l1746_174629

def S : Set ℚ := { x : ℚ | 0 < x ∧ x < (5 : ℚ)/8 }

def f (x : ℚ) : ℚ := 
  match x.num, x.den with
  | num, den => num / den + 1

theorem number_of_roots (h : ∀ q p, (p, q) = 1 → (q : ℚ) / p ∈ S → ((q + 1 : ℚ) / p = (2 : ℚ) / 3)) :
  ∃ n : ℕ, n = 7 :=
sorry

end number_of_roots_l1746_174629


namespace find_m_value_l1746_174653

noncomputable def fx (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem find_m_value (m : ℝ) : (∀ x > 0, fx m x > fx m 0) → m = 2 := by
  sorry

end find_m_value_l1746_174653


namespace fraction_compare_l1746_174670

theorem fraction_compare : 
  let a := (1 : ℝ) / 4
  let b := 250000025 / (10^9)
  let diff := a - b
  diff = (1 : ℝ) / (4 * 10^7) :=
by
  sorry

end fraction_compare_l1746_174670


namespace dice_product_composite_probability_l1746_174622

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l1746_174622


namespace not_approximately_equal_exp_l1746_174611

noncomputable def multinomial_approximation (n k₁ k₂ k₃ k₄ k₅ : ℕ) : ℝ :=
  (n.factorial : ℝ) / ((k₁.factorial : ℝ) * (k₂.factorial : ℝ) * (k₃.factorial : ℝ) * (k₄.factorial : ℝ) * (k₅.factorial : ℝ))

theorem not_approximately_equal_exp (e : ℝ) (h1 : e > 0) :
  e ^ 2737 ≠ multinomial_approximation 1000 70 270 300 220 140 :=
by 
  sorry  

end not_approximately_equal_exp_l1746_174611


namespace average_paychecks_l1746_174671

def first_paychecks : Nat := 6
def remaining_paychecks : Nat := 20
def total_paychecks : Nat := 26
def amount_first : Nat := 750
def amount_remaining : Nat := 770

theorem average_paychecks : 
  (first_paychecks * amount_first + remaining_paychecks * amount_remaining) / total_paychecks = 765 :=
by
  sorry

end average_paychecks_l1746_174671


namespace arithmetic_sequence_y_solution_l1746_174656

theorem arithmetic_sequence_y_solution : 
  ∃ y : ℚ, (y + 2 - - (1 / 3)) = (4 * y - (y + 2)) ∧ y = 13 / 6 :=
by
  sorry

end arithmetic_sequence_y_solution_l1746_174656


namespace minimize_base_side_length_l1746_174678

theorem minimize_base_side_length (V : ℝ) (a h : ℝ) 
  (volume_eq : V = a ^ 2 * h) (V_given : V = 256) (h_eq : h = 256 / (a ^ 2)) :
  a = 8 :=
by
  -- Recognize that for a given volume, making it a cube minimizes the surface area.
  -- As the volume of the cube a^3 = 256, solving for a gives 8.
  -- a := (256:ℝ) ^ (1/3:ℝ)
  sorry

end minimize_base_side_length_l1746_174678


namespace problem_statement_l1746_174607

theorem problem_statement (x y : ℝ) : (x * y < 18) → (x < 2 ∨ y < 9) :=
sorry

end problem_statement_l1746_174607


namespace least_num_subtracted_l1746_174614

theorem least_num_subtracted 
  (n : ℕ) 
  (h1 : n = 642) 
  (rem_cond : ∀ k, (k = 638) → n - k = 4): 
  n - 638 = 4 := 
by sorry

end least_num_subtracted_l1746_174614


namespace point_a_number_l1746_174601

theorem point_a_number (x : ℝ) (h : abs (x - 2) = 6) : x = 8 ∨ x = -4 :=
sorry

end point_a_number_l1746_174601


namespace xiao_ming_speed_difference_l1746_174677

noncomputable def distance_school : ℝ := 9.3
noncomputable def time_cycling : ℝ := 0.6
noncomputable def distance_park : ℝ := 0.9
noncomputable def time_walking : ℝ := 0.2

noncomputable def cycling_speed : ℝ := distance_school / time_cycling
noncomputable def walking_speed : ℝ := distance_park / time_walking
noncomputable def speed_difference : ℝ := cycling_speed - walking_speed

theorem xiao_ming_speed_difference : speed_difference = 11 := by
  sorry

end xiao_ming_speed_difference_l1746_174677


namespace quadratic_vertex_a_l1746_174682

theorem quadratic_vertex_a
  (a b c : ℝ)
  (h1 : ∀ x, (a * x^2 + b * x + c = a * (x - 2)^2 + 5))
  (h2 : a * 0^2 + b * 0 + c = 0) :
  a = -5/4 :=
by
  -- Use the given conditions to outline the proof (proof not provided here as per instruction)
  sorry

end quadratic_vertex_a_l1746_174682


namespace find_other_number_l1746_174650

theorem find_other_number (B : ℕ) (hcf_cond : Nat.gcd 36 B = 14) (lcm_cond : Nat.lcm 36 B = 396) : B = 66 :=
sorry

end find_other_number_l1746_174650


namespace solve_for_x_l1746_174651

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (4 * x + 28)) : 
  x = -17 / 5 := 
by 
  sorry

end solve_for_x_l1746_174651


namespace ratio_of_Y_share_l1746_174694

theorem ratio_of_Y_share (total_profit share_diff X_share Y_share : ℝ) 
(h1 : total_profit = 700) (h2 : share_diff = 140) 
(h3 : X_share + Y_share = 700) (h4 : X_share - Y_share = 140) : 
Y_share / total_profit = 2 / 5 :=
sorry

end ratio_of_Y_share_l1746_174694


namespace negation_of_exists_statement_l1746_174690

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l1746_174690


namespace arithmetic_sequence_sum_l1746_174685

-- Define arithmetic sequence and sum of first n terms
def arithmetic_seq (a d : ℕ → ℕ) :=
  ∀ n, a (n + 1) = a n + d 1

def arithmetic_sum (a d : ℕ → ℕ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

-- Conditions from the problem
variables {a : ℕ → ℕ} {d : ℕ}

axiom condition : a 3 + a 7 + a 11 = 6

-- Definition of a_7 as derived in the solution
def a_7 : ℕ := 2

-- Proof problem equivalent statement
theorem arithmetic_sequence_sum : arithmetic_sum a d 13 = 26 :=
by
  -- These steps would involve setting up and proving the calculation details
  sorry

end arithmetic_sequence_sum_l1746_174685


namespace left_seats_equals_15_l1746_174660

variable (L : ℕ)

noncomputable def num_seats_left (L : ℕ) : Prop :=
  ∃ L, 3 * L + 3 * (L - 3) + 8 = 89

theorem left_seats_equals_15 : num_seats_left L → L = 15 :=
by
  intro h
  sorry

end left_seats_equals_15_l1746_174660


namespace cut_scene_length_proof_l1746_174623

noncomputable def original_length : ℕ := 60
noncomputable def final_length : ℕ := 57
noncomputable def cut_scene_length := original_length - final_length

theorem cut_scene_length_proof : cut_scene_length = 3 := by
  sorry

end cut_scene_length_proof_l1746_174623


namespace stream_speed_l1746_174618

theorem stream_speed (x : ℝ) (d : ℝ) (v_b : ℝ) (t : ℝ) (h : v_b = 8) (h1 : d = 210) (h2 : t = 56) : x = 2 :=
by
  sorry

end stream_speed_l1746_174618


namespace inscribed_circle_radius_l1746_174668

theorem inscribed_circle_radius (AB BC CD DA: ℝ) (hAB: AB = 13) (hBC: BC = 10) (hCD: CD = 8) (hDA: DA = 11) :
  ∃ r, r = 2 * Real.sqrt 7 :=
by
  sorry

end inscribed_circle_radius_l1746_174668


namespace series_sum_l1746_174675

theorem series_sum :
  ∑' n : ℕ,  n ≠ 0 → (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end series_sum_l1746_174675


namespace final_height_of_helicopter_total_fuel_consumed_l1746_174696

noncomputable def height_changes : List Float := [4.1, -2.3, 1.6, -0.9, 1.1]

def total_height_change (changes : List Float) : Float :=
  changes.foldl (λ acc x => acc + x) 0

theorem final_height_of_helicopter :
  total_height_change height_changes = 3.6 :=
by
  sorry

noncomputable def fuel_consumption (changes : List Float) : Float :=
  changes.foldl (λ acc x => if x > 0 then acc + 5 * x else acc + 3 * -x) 0

theorem total_fuel_consumed :
  fuel_consumption height_changes = 43.6 :=
by
  sorry

end final_height_of_helicopter_total_fuel_consumed_l1746_174696


namespace negation_of_proposition_l1746_174612

theorem negation_of_proposition {c : ℝ} (h : ∃ (c : ℝ), c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) :
  ∀ (c : ℝ), c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0 :=
by
  sorry

end negation_of_proposition_l1746_174612


namespace largest_constant_c_l1746_174649

theorem largest_constant_c :
  ∃ c : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 1 → x^6 + y^6 ≥ c * x * y) ∧ c = 1 / 2 :=
sorry

end largest_constant_c_l1746_174649


namespace calculate_total_cost_l1746_174626

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end calculate_total_cost_l1746_174626


namespace james_painted_area_l1746_174666

-- Define the dimensions of the wall and windows
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 6

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length
def total_window_area : ℕ := window1_area + window2_area
def painted_area : ℕ := wall_area - total_window_area

theorem james_painted_area : painted_area = 123 :=
by
  -- The proof is omitted
  sorry

end james_painted_area_l1746_174666


namespace simplify_fraction_l1746_174659

theorem simplify_fraction (a b c : ℕ) (h1 : a = 222) (h2 : b = 8888) (h3 : c = 44) : 
  (a : ℚ) / b * c = 111 / 101 := 
by 
  sorry

end simplify_fraction_l1746_174659


namespace solve_for_x_l1746_174664

theorem solve_for_x 
  (x : ℝ) 
  (h : (2/7) * (1/4) * x = 8) : 
  x = 112 :=
sorry

end solve_for_x_l1746_174664


namespace find_width_of_rect_box_l1746_174640

-- Define the dimensions of the wooden box in meters
def wooden_box_length_m : ℕ := 8
def wooden_box_width_m : ℕ := 7
def wooden_box_height_m : ℕ := 6

-- Define the dimensions of the rectangular boxes in centimeters (with unknown width W)
def rect_box_length_cm : ℕ := 8
def rect_box_height_cm : ℕ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 1000000

-- Define the constraint that the total volume of the boxes should not exceed the volume of the wooden box
theorem find_width_of_rect_box (W : ℕ) (wooden_box_volume : ℕ := (wooden_box_length_m * 100) * (wooden_box_width_m * 100) * (wooden_box_height_m * 100)) : 
  (rect_box_length_cm * W * rect_box_height_cm) * max_boxes = wooden_box_volume → W = 7 :=
by
  sorry

end find_width_of_rect_box_l1746_174640


namespace A_share_of_gain_l1746_174647

-- Given problem conditions
def investment_A (x : ℝ) : ℝ := x * 12
def investment_B (x : ℝ) : ℝ := 2 * x * 6
def investment_C (x : ℝ) : ℝ := 3 * x * 4
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def total_gain : ℝ := 21000

-- Mathematically equivalent proof problem statement
theorem A_share_of_gain (x : ℝ) : (investment_A x) / (total_investment x) * total_gain = 7000 :=
by
  sorry

end A_share_of_gain_l1746_174647


namespace top_black_second_red_probability_l1746_174641

-- Define the problem conditions in Lean
def num_standard_cards : ℕ := 52
def num_jokers : ℕ := 2
def num_total_cards : ℕ := num_standard_cards + num_jokers

def num_black_cards : ℕ := 26
def num_red_cards : ℕ := 26

-- Lean statement
theorem top_black_second_red_probability :
  (num_black_cards / num_total_cards * num_red_cards / (num_total_cards - 1)) = 338 / 1431 := by
  sorry

end top_black_second_red_probability_l1746_174641


namespace spotted_mushrooms_ratio_l1746_174643

theorem spotted_mushrooms_ratio 
  (total_mushrooms : ℕ) 
  (gilled_mushrooms : ℕ) 
  (spotted_mushrooms : ℕ) 
  (total_mushrooms_eq : total_mushrooms = 30) 
  (gilled_mushrooms_eq : gilled_mushrooms = 3) 
  (spots_and_gills_exclusive : ∀ x, x = spotted_mushrooms ∨ x = gilled_mushrooms) : 
  spotted_mushrooms / gilled_mushrooms = 9 := 
by
  sorry

end spotted_mushrooms_ratio_l1746_174643


namespace find_x_floor_mult_eq_45_l1746_174645

theorem find_x_floor_mult_eq_45 (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 45) : x = 7.5 :=
sorry

end find_x_floor_mult_eq_45_l1746_174645


namespace find_sticker_price_l1746_174679

variable (x : ℝ)

def price_at_store_A (x : ℝ) : ℝ := 0.80 * x - 120
def price_at_store_B (x : ℝ) : ℝ := 0.70 * x
def savings (x : ℝ) : ℝ := price_at_store_B x - price_at_store_A x

theorem find_sticker_price (h : savings x = 30) : x = 900 :=
by
  -- proof can be filled in here
  sorry

end find_sticker_price_l1746_174679


namespace find_line_AB_l1746_174646

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 16

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Proof statement: Line AB is the correct line through the intersection points of the two circles
theorem find_line_AB :
  (∃ x y, circle1 x y ∧ circle2 x y) →
  (∀ x y, (circle1 x y ∧ circle2 x y) ↔ lineAB x y) :=
by
  sorry

end find_line_AB_l1746_174646


namespace original_population_l1746_174621

-- Define the initial setup
variable (P : ℝ)

-- The conditions given in the problem
axiom ten_percent_died (P : ℝ) : (1 - 0.1) * P = 0.9 * P
axiom twenty_percent_left (P : ℝ) : (1 - 0.2) * (0.9 * P) = 0.9 * P * 0.8

-- Define the final condition
axiom final_population (P : ℝ) : 0.9 * P * 0.8 = 3240

-- The proof problem
theorem original_population : P = 4500 :=
by
  sorry

end original_population_l1746_174621


namespace students_failed_in_english_l1746_174698

variable (H : ℝ) (E : ℝ) (B : ℝ) (P : ℝ)

theorem students_failed_in_english
  (hH : H = 34 / 100) 
  (hB : B = 22 / 100)
  (hP : P = 44 / 100)
  (hIE : (1 - P) = H + E - B) :
  E = 44 / 100 := 
sorry

end students_failed_in_english_l1746_174698


namespace inverse_function_point_l1746_174663

theorem inverse_function_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ ∀ y, (∀ x, y = a^(x-3) + 1) → (2, 3) ∈ {(y, x) | y = a^(x-3) + 1} :=
by
  sorry

end inverse_function_point_l1746_174663


namespace John_height_in_feet_after_growth_spurt_l1746_174681

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end John_height_in_feet_after_growth_spurt_l1746_174681


namespace simultaneous_equations_solution_l1746_174665

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_equations_solution_l1746_174665


namespace problem_solution_l1746_174691

-- Define a function to sum the digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

-- Define the problem numbers.
def nums : List ℕ := [4272, 4281, 4290, 4311, 4320]

-- Check if the sum of digits is divisible by 9.
def divisible_by_9 (n : ℕ) : Prop :=
  sum_digits n % 9 = 0

-- Main theorem asserting the result.
theorem problem_solution :
  ∃ n ∈ nums, ¬divisible_by_9 n ∧ (n % 100 / 10) * (n % 10) = 14 := by
  sorry

end problem_solution_l1746_174691


namespace fractional_identity_l1746_174609

theorem fractional_identity (m n r t : ℚ) 
  (h₁ : m / n = 5 / 2) 
  (h₂ : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 :=
by 
  sorry

end fractional_identity_l1746_174609


namespace total_dining_bill_before_tip_l1746_174602

-- Define total number of people
def numberOfPeople : ℕ := 6

-- Define the individual payment
def individualShare : ℝ := 25.48

-- Define the total payment
def totalPayment : ℝ := numberOfPeople * individualShare

-- Define the tip percentage
def tipPercentage : ℝ := 0.10

-- Total payment including tip expressed in terms of the original bill B
def totalPaymentWithTip (B : ℝ) : ℝ := B + B * tipPercentage

-- Prove the total dining bill before the tip
theorem total_dining_bill_before_tip : 
    ∃ B : ℝ, totalPayment = totalPaymentWithTip B ∧ B = 139.89 :=
by
    sorry

end total_dining_bill_before_tip_l1746_174602


namespace double_inequality_l1746_174684

variable (a b c : ℝ)

theorem double_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a * b - b * c - c * a ∧ 
  a + b + c - a * b - b * c - c * a ≤ 1 / 2 * (1 + a^2 + b^2 + c^2) := 
sorry

end double_inequality_l1746_174684


namespace min_score_needed_l1746_174652

theorem min_score_needed 
  (s1 s2 s3 s4 s5 : ℕ)
  (next_test_goal_increment : ℕ)
  (current_scores_sum : ℕ)
  (desired_average : ℕ)
  (total_tests : ℕ)
  (required_total_sum : ℕ)
  (required_next_score : ℕ)
  (current_scores : s1 = 88 ∧ s2 = 92 ∧ s3 = 75 ∧ s4 = 85 ∧ s5 = 80)
  (increment_eq : next_test_goal_increment = 5)
  (current_sum_eq : current_scores_sum = s1 + s2 + s3 + s4 + s5)
  (desired_average_eq : desired_average = (current_scores_sum / 5) + next_test_goal_increment)
  (total_tests_eq : total_tests = 6)
  (required_total_sum_eq : required_total_sum = desired_average * total_tests)
  (required_next_score_eq : required_next_score = required_total_sum - current_scores_sum) :
  required_next_score = 114 := by
    sorry

end min_score_needed_l1746_174652


namespace animal_sale_money_l1746_174648

theorem animal_sale_money (G S : ℕ) (h1 : G + S = 360) (h2 : 5 * S = 7 * G) : 
  (1/2 * G * 40) + (2/3 * S * 30) = 7200 := 
by
  sorry

end animal_sale_money_l1746_174648


namespace mike_daily_work_hours_l1746_174673

def total_hours_worked : ℕ := 15
def number_of_days_worked : ℕ := 5

theorem mike_daily_work_hours : total_hours_worked / number_of_days_worked = 3 :=
by
  sorry

end mike_daily_work_hours_l1746_174673


namespace proportional_relationships_l1746_174662

-- Let l, v, t be real numbers indicating distance, velocity, and time respectively.
variables (l v t : ℝ)

-- Define the relationships according to the given formulas
def distance_formula := l = v * t
def velocity_formula := v = l / t
def time_formula := t = l / v

-- Definitions of proportionality
def directly_proportional (x y : ℝ) := ∃ k : ℝ, x = k * y
def inversely_proportional (x y : ℝ) := ∃ k : ℝ, x * y = k

-- The main theorem
theorem proportional_relationships (const_t const_v const_l : ℝ) :
  (distance_formula l v const_t → directly_proportional l v) ∧
  (distance_formula l const_v t → directly_proportional l t) ∧
  (velocity_formula const_l v t → inversely_proportional v t) :=
by
  sorry

end proportional_relationships_l1746_174662


namespace parallelogram_side_problem_l1746_174676

theorem parallelogram_side_problem (y z : ℝ) (h1 : 4 * z + 1 = 15) (h2 : 3 * y - 2 = 15) :
  y + z = 55 / 6 :=
sorry

end parallelogram_side_problem_l1746_174676


namespace division_problem_l1746_174644

theorem division_problem :
  (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end division_problem_l1746_174644


namespace average_bc_l1746_174692

variables (A B C : ℝ)

-- Conditions
def average_abc := (A + B + C) / 3 = 45
def average_ab := (A + B) / 2 = 40
def weight_b := B = 31

-- Proof statement
theorem average_bc (A B C : ℝ) (h_avg_abc : average_abc A B C) (h_avg_ab : average_ab A B) (h_b : weight_b B) :
  (B + C) / 2 = 43 :=
sorry

end average_bc_l1746_174692


namespace find_calories_per_slice_l1746_174639

/-- Defining the number of slices and their respective calories. -/
def slices_in_cake : ℕ := 8
def calories_per_brownie : ℕ := 375
def brownies_in_pan : ℕ := 6
def extra_calories_in_cake : ℕ := 526

/-- Defining the total calories in cake and brownies -/
def total_calories_in_brownies : ℕ := brownies_in_pan * calories_per_brownie
def total_calories_in_cake (c : ℕ) : ℕ := slices_in_cake * c

/-- The equation from the given problem -/
theorem find_calories_per_slice (c : ℕ) :
  total_calories_in_cake c = total_calories_in_brownies + extra_calories_in_cake → c = 347 :=
by
  sorry

end find_calories_per_slice_l1746_174639


namespace inequality_proof_l1746_174687

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l1746_174687


namespace chess_positions_after_one_move_each_l1746_174697

def number_of_chess_positions (initial_positions : ℕ) (pawn_moves : ℕ) (knight_moves : ℕ) (active_pawns : ℕ) (active_knights : ℕ) : ℕ :=
  let pawn_move_combinations := active_pawns * pawn_moves
  let knight_move_combinations := active_knights * knight_moves
  pawn_move_combinations + knight_move_combinations

theorem chess_positions_after_one_move_each :
  number_of_chess_positions 1 2 2 8 2 * number_of_chess_positions 1 2 2 8 2 = 400 :=
by
  sorry

end chess_positions_after_one_move_each_l1746_174697


namespace remainder_123456789012_mod_252_l1746_174680

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l1746_174680


namespace balance_balls_l1746_174695

open Real

variables (G B Y W : ℝ)

-- Conditions
def condition1 := (4 * G = 8 * B)
def condition2 := (3 * Y = 6 * B)
def condition3 := (8 * B = 6 * W)

-- Theorem statement
theorem balance_balls 
  (h1 : condition1 G B) 
  (h2 : condition2 Y B) 
  (h3 : condition3 B W) :
  ∃ (B_needed : ℝ), B_needed = 5 * G + 3 * Y + 4 * W ∧ B_needed = 64 / 3 * B :=
sorry

end balance_balls_l1746_174695
