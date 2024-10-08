import Mathlib

namespace fraction_people_eating_pizza_l160_160944

variable (people : ℕ) (initial_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
variable (fraction : ℚ)

theorem fraction_people_eating_pizza (h1 : people = 15)
    (h2 : initial_pizza = 50)
    (h3 : pieces_per_person = 4)
    (h4 : remaining_pizza = 14)
    (h5 : 4 * 15 * fraction = initial_pizza - remaining_pizza) :
    fraction = 3 / 5 := 
  sorry

end fraction_people_eating_pizza_l160_160944


namespace payment_n_amount_l160_160090

def payment_m_n (m n : ℝ) : Prop :=
  m + n = 550 ∧ m = 1.2 * n

theorem payment_n_amount : ∃ n : ℝ, ∀ m : ℝ, payment_m_n m n → n = 250 :=
by
  sorry

end payment_n_amount_l160_160090


namespace necessary_but_not_sufficient_condition_l160_160759

-- Definitions
variable (f : ℝ → ℝ)

-- Condition that we need to prove
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

def is_symmetric_about_origin (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -g (-x)

-- Necessary and sufficient condition
theorem necessary_but_not_sufficient_condition : 
  (∀ x, |f x| = |f (-x)|) ↔ (∀ x, f x = -f (-x)) ∧ ¬(∀ x, |f x| = |f (-x)| → f x = -f (-x)) := by 
sorry

end necessary_but_not_sufficient_condition_l160_160759


namespace combined_transformation_matrix_l160_160562

-- Definitions for conditions
def dilation_matrix (s : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0], ![0, s]]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

-- Theorem to be proven
theorem combined_transformation_matrix :
  (rotation_matrix_90_ccw * dilation_matrix 4) = ![![0, -4], ![4, 0]] :=
by
  sorry

end combined_transformation_matrix_l160_160562


namespace find_a_l160_160466

theorem find_a (a x : ℝ)
    (h1 : 6 * (x + 8) = 18 * x)
    (h2 : 6 * x - 2 * (a - x) = 2 * a + x) :
    a = 7 :=
  sorry

end find_a_l160_160466


namespace max_value_of_6_f_x_plus_2012_l160_160845

noncomputable def f (x : ℝ) : ℝ :=
  min (min (4*x + 1) (x + 2)) (-2*x + 4)

theorem max_value_of_6_f_x_plus_2012 : ∃ x : ℝ, 6 * f x + 2012 = 2028 :=
sorry

end max_value_of_6_f_x_plus_2012_l160_160845


namespace christine_commission_rate_l160_160686

theorem christine_commission_rate (C : ℝ) (H1 : 24000 ≠ 0) (H2 : 0.4 * (C / 100 * 24000) = 1152) :
  C = 12 :=
by
  sorry

end christine_commission_rate_l160_160686


namespace saucepan_capacity_l160_160349

-- Define the conditions
variable (x : ℝ)
variable (h : 0.28 * x = 35)

-- State the theorem
theorem saucepan_capacity : x = 125 :=
by
  sorry

end saucepan_capacity_l160_160349


namespace arcsin_eq_pi_div_two_solve_l160_160493

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end arcsin_eq_pi_div_two_solve_l160_160493


namespace total_legs_l160_160301

def animals_legs (dogs : Nat) (birds : Nat) (insects : Nat) : Nat :=
  (dogs * 4) + (birds * 2) + (insects * 6)

theorem total_legs :
  animals_legs 3 2 2 = 22 := by
  sorry

end total_legs_l160_160301


namespace pete_total_blocks_traveled_l160_160360

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end pete_total_blocks_traveled_l160_160360


namespace people_distribution_l160_160761

theorem people_distribution
  (total_mentions : ℕ)
  (mentions_house : ℕ)
  (mentions_fountain : ℕ)
  (mentions_bench : ℕ)
  (mentions_tree : ℕ)
  (each_person_mentions : ℕ)
  (total_people : ℕ)
  (facing_house : ℕ)
  (facing_fountain : ℕ)
  (facing_bench : ℕ)
  (facing_tree : ℕ)
  (h_total_mentions : total_mentions = 27)
  (h_mentions_house : mentions_house = 5)
  (h_mentions_fountain : mentions_fountain = 6)
  (h_mentions_bench : mentions_bench = 7)
  (h_mentions_tree : mentions_tree = 9)
  (h_each_person_mentions : each_person_mentions = 3)
  (h_total_people : total_people = 9)
  (h_facing_house : facing_house = 5)
  (h_facing_fountain : facing_fountain = 4)
  (h_facing_bench : facing_bench = 2)
  (h_facing_tree : facing_tree = 9) :
  total_mentions / each_person_mentions = total_people ∧ 
  facing_house = mentions_house ∧
  facing_fountain = total_people - mentions_house ∧
  facing_bench = total_people - mentions_bench ∧
  facing_tree = total_people - mentions_tree :=
by
  sorry

end people_distribution_l160_160761


namespace total_floors_combined_l160_160907

-- Let C be the number of floors in the Chrysler Building
-- Let L be the number of floors in the Leeward Center
-- Given that C = 23 and C = L + 11
-- Prove that the total floors in both buildings combined equals 35

theorem total_floors_combined (C L : ℕ) (h1 : C = 23) (h2 : C = L + 11) : C + L = 35 :=
by
  sorry

end total_floors_combined_l160_160907


namespace solve_inequalities_l160_160495

theorem solve_inequalities (x : ℝ) (h1 : |4 - x| < 5) (h2 : x^2 < 36) : (-1 < x) ∧ (x < 6) :=
by
  sorry

end solve_inequalities_l160_160495


namespace find_abc_value_l160_160179

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variable (h4 : a * b = 30)
variable (h5 : b * c = 54)
variable (h6 : c * a = 45)

theorem find_abc_value : a * b * c = 270 := by
  sorry

end find_abc_value_l160_160179


namespace ab_bc_ca_leq_zero_l160_160504

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l160_160504


namespace minimum_value_l160_160026

/-- The minimum value of the expression (x+2)^2 / (y-2) + (y+2)^2 / (x-2)
    for real numbers x > 2 and y > 2 is 50. -/
theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ z, z = (x + 2) ^ 2 / (y - 2) + (y + 2) ^ 2 / (x - 2) ∧ z = 50 :=
sorry

end minimum_value_l160_160026


namespace problem_statement_l160_160111

variable {A B C D E F H : Point}
variable {a b c : ℝ}

-- Assume the conditions
variable (h_triangle : Triangle A B C)
variable (h_acute : AcuteTriangle h_triangle)
variable (h_altitudes : AltitudesIntersectAt h_triangle H A D B E C F)
variable (h_sides : Sides h_triangle BC a AC b AB c)

-- Statement to prove
theorem problem_statement : AH * AD + BH * BE + CH * CF = 1/2 * (a^2 + b^2 + c^2) :=
sorry

end problem_statement_l160_160111


namespace find_S2012_l160_160243

section Problem

variable {a : ℕ → ℝ} -- Defining the sequence

-- Conditions
def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q, ∀ n, a (n + 1) = a n * q

def sum_S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (Finset.range n).sum a

axiom a1 : a 1 = 2011
axiom recurrence_relation (n : ℕ) : a n + 2*a (n + 1) + a (n + 2) = 0

-- Proof statement
theorem find_S2012 (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ):
  geometric_sequence a →
  (∀ n, S n = sum_S a n) →
  S 2012 = 0 :=
by
  sorry

end Problem

end find_S2012_l160_160243


namespace count_multiples_12_9_l160_160405

theorem count_multiples_12_9 :
  ∃ n : ℕ, n = 8 ∧ (∀ x : ℕ, x % 36 = 0 ∧ 200 ≤ x ∧ x ≤ 500 ↔ ∃ y : ℕ, (x = 36 * y ∧ 200 ≤ 36 * y ∧ 36 * y ≤ 500)) :=
by
  sorry

end count_multiples_12_9_l160_160405


namespace solve_linear_system_l160_160067

theorem solve_linear_system :
  ∃ x y : ℤ, x + 9773 = 13200 ∧ 2 * x - 3 * y = 1544 ∧ x = 3427 ∧ y = 1770 := by
  sorry

end solve_linear_system_l160_160067


namespace sibling_age_difference_l160_160974

theorem sibling_age_difference 
  (x : ℕ) 
  (h : 3 * x + 2 * x + 1 * x = 90) : 
  3 * x - x = 30 := 
by 
  sorry

end sibling_age_difference_l160_160974


namespace geometric_sequence_S6_l160_160968

-- Assume we have a geometric sequence {a_n} and the sum of the first n terms is denoted as S_n
variable (S : ℕ → ℝ)

-- Conditions given in the problem
axiom S2_eq : S 2 = 2
axiom S4_eq : S 4 = 8

-- The goal is to find the value of S 6
theorem geometric_sequence_S6 : S 6 = 26 := 
by 
  sorry

end geometric_sequence_S6_l160_160968


namespace work_completion_days_l160_160101

theorem work_completion_days (A B C : ℕ) (A_rate B_rate C_rate : ℚ) :
  A_rate = 1 / 30 → B_rate = 1 / 55 → C_rate = 1 / 45 →
  1 / (A_rate + B_rate + C_rate) = 55 / 4 :=
by
  intro hA hB hC
  rw [hA, hB, hC]
  sorry

end work_completion_days_l160_160101


namespace solution_set_inequality_l160_160814

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x : ℝ, deriv f x < 1 / 2

theorem solution_set_inequality : {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f (Real.log x / Real.log 2) > (Real.log x / Real.log 2 + 1) / 2} :=
by
  sorry

end solution_set_inequality_l160_160814


namespace simson_line_properties_l160_160341

-- Given a triangle ABC
variables {A B C M P Q R H : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] 
variables [Inhabited M] [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited H]

-- Conditions
def is_point_on_circumcircle (A B C : Type) (M : Type) : Prop :=
sorry  -- formal definition that M is on the circumcircle of triangle ABC

def perpendicular_dropped_to_side (M : Type) (side : Type) (foot : Type) : Prop :=
sorry  -- formal definition of a perpendicular dropping from M to a side

def is_orthocenter (A B C H : Type) : Prop := 
sorry  -- formal definition that H is the orthocenter of triangle ABC

-- Proof Goal 1: The points P, Q, R are collinear (Simson line)
def simson_line (A B C M P Q R : Type) : Prop :=
sorry  -- formal definition and proof that P, Q, R are collinear

-- Proof Goal 2: The Simson line is equidistant from point M and the orthocenter H
def simson_line_equidistant (M H P Q R : Type) : Prop :=
sorry  -- formal definition and proof that Simson line is equidistant from M and H

-- Main theorem combining both proof goals
theorem simson_line_properties 
  (A B C M P Q R H : Type)
  (M_on_circumcircle : is_point_on_circumcircle A B C M)
  (perp_to_BC : perpendicular_dropped_to_side M (B × C) P)
  (perp_to_CA : perpendicular_dropped_to_side M (C × A) Q)
  (perp_to_AB : perpendicular_dropped_to_side M (A × B) R)
  (H_is_orthocenter : is_orthocenter A B C H) :
  simson_line A B C M P Q R ∧ simson_line_equidistant M H P Q R := 
by sorry

end simson_line_properties_l160_160341


namespace pudding_cost_l160_160962

theorem pudding_cost (P : ℝ) (h1 : 75 = 5 * P + 65) : P = 2 :=
sorry

end pudding_cost_l160_160962


namespace parabola_focus_l160_160965

theorem parabola_focus (p : ℝ) (h : 4 = 2 * p * 1^2) : (0, 1 / (4 * 2 * p)) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l160_160965


namespace skittles_distribution_l160_160196

-- Given problem conditions
variable (Brandon_initial : ℕ := 96) (Bonnie_initial : ℕ := 4) 
variable (Brandon_loss : ℕ := 9)
variable (combined_skittles : ℕ := (Brandon_initial - Brandon_loss) + Bonnie_initial)
variable (individual_share : ℕ := combined_skittles / 4)
variable (remainder : ℕ := combined_skittles % 4)
variable (Chloe_share : ℕ := individual_share)
variable (Dylan_share_initial : ℕ := individual_share)
variable (Chloe_to_Dylan : ℕ := Chloe_share / 2)
variable (Dylan_new_share : ℕ := Dylan_share_initial + Chloe_to_Dylan)
variable (Dylan_to_Bonnie : ℕ := Dylan_new_share / 3)
variable (final_Bonnie : ℕ := individual_share + Dylan_to_Bonnie)
variable (final_Chloe : ℕ := Chloe_share - Chloe_to_Dylan)
variable (final_Dylan : ℕ := Dylan_new_share - Dylan_to_Bonnie)

-- The theorem to be proved
theorem skittles_distribution : 
  individual_share = 22 ∧ final_Bonnie = 33 ∧ final_Chloe = 11 ∧ final_Dylan = 22 :=
by
  -- The proof would go here, but it’s not required for this task.
  sorry

end skittles_distribution_l160_160196


namespace cube_volume_l160_160126

variable (V_sphere : ℝ)
variable (V_cube : ℝ)
variable (R : ℝ)
variable (a : ℝ)

theorem cube_volume (h1 : V_sphere = (32 / 3) * Real.pi)
    (h2 : V_sphere = (4 / 3) * Real.pi * R^3)
    (h3 : R = 2)
    (h4 : R = (Real.sqrt 3 / 2) * a)
    (h5 : a = 4 * Real.sqrt 3 / 3) :
    V_cube = (4 * Real.sqrt 3 / 3) ^ 3 :=
  by
    sorry

end cube_volume_l160_160126


namespace smallest_nat_number_l160_160400

theorem smallest_nat_number (x : ℕ) 
  (h1 : ∃ z : ℕ, x + 3 = 5 * z) 
  (h2 : ∃ n : ℕ, x - 3 = 6 * n) : x = 27 := 
sorry

end smallest_nat_number_l160_160400


namespace custom_op_value_l160_160197

variable {a b : ℤ}
def custom_op (a b : ℤ) := 1/a + 1/b

axiom h1 : a + b = 15
axiom h2 : a * b = 56

theorem custom_op_value : custom_op a b = 15/56 :=
by
  sorry

end custom_op_value_l160_160197


namespace find_first_number_l160_160590

theorem find_first_number 
  (second_number : ℕ)
  (increment : ℕ)
  (final_number : ℕ)
  (h1 : second_number = 45)
  (h2 : increment = 11)
  (h3 : final_number = 89)
  : ∃ first_number : ℕ, first_number + increment = second_number := 
by
  sorry

end find_first_number_l160_160590


namespace find_ac_bd_l160_160402

variable (a b c d : ℝ)

axiom cond1 : a^2 + b^2 = 1
axiom cond2 : c^2 + d^2 = 1
axiom cond3 : a * d - b * c = 1 / 7

theorem find_ac_bd : a * c + b * d = 4 * Real.sqrt 3 / 7 := by
  sorry

end find_ac_bd_l160_160402


namespace rectangle_sides_l160_160155

theorem rectangle_sides (a b : ℝ) (h₁ : a < b) (h₂ : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 :=
sorry

end rectangle_sides_l160_160155


namespace max_M_value_l160_160774

noncomputable def M (x y z w : ℝ) : ℝ :=
  x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z

theorem max_M_value (x y z w : ℝ) (h : x + y + z + w = 1) :
  (M x y z w) ≤ 3 / 2 :=
sorry

end max_M_value_l160_160774


namespace infinite_superset_of_infinite_subset_l160_160740

theorem infinite_superset_of_infinite_subset {A B : Set ℕ} (h_subset : B ⊆ A) (h_infinite : Infinite B) : Infinite A := 
sorry

end infinite_superset_of_infinite_subset_l160_160740


namespace laura_saves_more_with_promotion_A_l160_160928

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end laura_saves_more_with_promotion_A_l160_160928


namespace total_books_equals_45_l160_160681

-- Define the number of books bought in each category
def adventure_books : ℝ := 13.0
def mystery_books : ℝ := 17.0
def crime_books : ℝ := 15.0

-- Total number of books bought
def total_books := adventure_books + mystery_books + crime_books

-- The theorem we need to prove
theorem total_books_equals_45 : total_books = 45.0 := by
  -- placeholder for the proof
  sorry

end total_books_equals_45_l160_160681


namespace problem_statement_l160_160319

open Real

theorem problem_statement (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1/a) + (1/b) = 1) (hn_pos : 0 < n) :
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry -- proof to be provided

end problem_statement_l160_160319


namespace mul_value_proof_l160_160181

theorem mul_value_proof :
  ∃ x : ℝ, (8.9 - x = 3.1) ∧ ((x * 3.1) * 2.5 = 44.95) :=
by
  sorry

end mul_value_proof_l160_160181


namespace simplify_and_find_ratio_l160_160830

theorem simplify_and_find_ratio (k : ℤ) : (∃ (c d : ℤ), (∀ x y : ℤ, c = 1 ∧ d = 2 ∧ x = c ∧ y = d → ((6 * k + 12) / 6 = k + 2) ∧ (c / d = 1 / 2))) :=
by
  use 1
  use 2
  sorry

end simplify_and_find_ratio_l160_160830


namespace notebook_costs_2_20_l160_160877

theorem notebook_costs_2_20 (n c : ℝ) (h1 : n + c = 2.40) (h2 : n = 2 + c) : n = 2.20 :=
by
  sorry

end notebook_costs_2_20_l160_160877


namespace investment_doubles_in_9_years_l160_160717

noncomputable def years_to_double (initial_amount : ℕ) (interest_rate : ℕ) : ℕ :=
  72 / interest_rate

theorem investment_doubles_in_9_years :
  ∀ (initial_amount : ℕ) (interest_rate : ℕ) (investment_period_val : ℕ) (expected_value : ℕ),
  initial_amount = 8000 ∧ interest_rate = 8 ∧ investment_period_val = 18 ∧ expected_value = 32000 →
  years_to_double initial_amount interest_rate = 9 :=
by
  intros initial_amount interest_rate investment_period_val expected_value h
  sorry

end investment_doubles_in_9_years_l160_160717


namespace range_of_alpha_l160_160428

open Real

theorem range_of_alpha 
  (α : ℝ) (k : ℤ) :
  (sin α > 0) ∧ (cos α < 0) ∧ (sin α > cos α) →
  (∃ k : ℤ, (2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) ∨ 
  (2 * k * π + (3 * π / 2) < α ∧ α < 2 * k * π + 2 * π)) := 
by 
  sorry

end range_of_alpha_l160_160428


namespace complete_square_transform_l160_160534

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l160_160534


namespace reusable_bag_trips_correct_lowest_carbon_solution_l160_160724

open Real

-- Conditions definitions
def canvas_CO2 := 600 -- in pounds
def polyester_CO2 := 250 -- in pounds
def recycled_plastic_CO2 := 150 -- in pounds
def CO2_per_plastic_bag := 4 / 16 -- 4 ounces per bag, converted to pounds
def bags_per_trip := 8

-- Total CO2 per trip using plastic bags
def CO2_per_trip := CO2_per_plastic_bag * bags_per_trip

-- Proof of correct number of trips
theorem reusable_bag_trips_correct :
  canvas_CO2 / CO2_per_trip = 300 ∧
  polyester_CO2 / CO2_per_trip = 125 ∧
  recycled_plastic_CO2 / CO2_per_trip = 75 :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

-- Proof that recycled plastic bag is the lowest-carbon solution
theorem lowest_carbon_solution :
  min (canvas_CO2 / CO2_per_trip) (min (polyester_CO2 / CO2_per_trip) (recycled_plastic_CO2 / CO2_per_trip)) = recycled_plastic_CO2 / CO2_per_trip :=
by
  -- Here we would provide proofs for each part,
  -- ensuring we are fulfilling the conditions provided
  -- Skipping the proof with sorry
  sorry

end reusable_bag_trips_correct_lowest_carbon_solution_l160_160724


namespace exponent_property_l160_160468

theorem exponent_property :
  4^4 * 9^4 * 4^9 * 9^9 = 36^13 :=
by
  -- Add the proof here
  sorry

end exponent_property_l160_160468


namespace find_rate_of_interest_l160_160419

-- Conditions
def principal : ℕ := 4200
def time : ℕ := 2
def interest_12 : ℕ := principal * 12 * time / 100
def additional_interest : ℕ := 504
def total_interest_r : ℕ := interest_12 + additional_interest

-- Theorem Statement
theorem find_rate_of_interest (r : ℕ) (h : 1512 = principal * r * time / 100) : r = 18 :=
by sorry

end find_rate_of_interest_l160_160419


namespace find_smallest_z_l160_160294

theorem find_smallest_z (x y z : ℤ) (h1 : 7 < x) (h2 : x < 9) (h3 : x < y) (h4 : y < z) 
  (h5 : y - x = 7) : z = 16 :=
by
  sorry

end find_smallest_z_l160_160294


namespace inequality_solution_l160_160519

theorem inequality_solution (x : ℝ) : (1 - 3 * (x - 1) < x) ↔ (x > 1) :=
by sorry

end inequality_solution_l160_160519


namespace find_m_l160_160157

-- Define the conditions
variables {m x1 x2 : ℝ}

-- Given the equation x^2 + mx - 1 = 0 has roots x1 and x2:
-- The sum of the roots x1 + x2 is -m, and the product of the roots x1 * x2 is -1.
-- Furthermore, given that 1/x1 + 1/x2 = -3,
-- Prove that m = -3.

theorem find_m :
  (x1 + x2 = -m) →
  (x1 * x2 = -1) →
  (1 / x1 + 1 / x2 = -3) →
  m = -3 := by
  intros hSum hProd hRecip
  sorry

end find_m_l160_160157


namespace cakes_initially_made_l160_160045

variables (sold bought total initial_cakes : ℕ)

theorem cakes_initially_made (h1 : sold = 105) (h2 : bought = 170) (h3 : total = 186) :
  initial_cakes = total - (sold - bought) :=
by
  rw [h1, h2, h3]
  sorry

end cakes_initially_made_l160_160045


namespace unique_zero_point_condition1_unique_zero_point_condition2_l160_160015

noncomputable def func (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem unique_zero_point_condition1 {a b : ℝ} (h1 : 1 / 2 < a) (h2 : a ≤ Real.exp 2 / 2) (h3 : b > 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

theorem unique_zero_point_condition2 {a b : ℝ} (h1 : 0 < a) (h2 : a < 1 / 2) (h3 : b ≤ 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

end unique_zero_point_condition1_unique_zero_point_condition2_l160_160015


namespace usual_time_to_office_l160_160475

theorem usual_time_to_office (P : ℝ) (T : ℝ) (h1 : T = (3 / 4) * (T + 20)) : T = 60 :=
by
  sorry

end usual_time_to_office_l160_160475


namespace pyramid_base_side_length_l160_160289

theorem pyramid_base_side_length
  (lateral_face_area : Real)
  (slant_height : Real)
  (s : Real)
  (h_lateral_face_area : lateral_face_area = 200)
  (h_slant_height : slant_height = 40)
  (h_area_formula : lateral_face_area = 0.5 * s * slant_height) :
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l160_160289


namespace inequality_solution_set_l160_160631

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} =
  {x : ℝ | 0 < x ∧ x ≤ 1 / 8} ∪ {x : ℝ | 2 < x ∧ x ≤ 6} :=
by
  -- Proof will go here
  sorry

end inequality_solution_set_l160_160631


namespace polynomial_identity_l160_160229

theorem polynomial_identity 
  (P : Polynomial ℤ)
  (a b : ℤ) 
  (h_distinct : a ≠ b)
  (h_eq : P.eval a * P.eval b = -(a - b) ^ 2) : 
  P.eval a + P.eval b = 0 := 
by
  sorry

end polynomial_identity_l160_160229


namespace find_g3_l160_160900

variable {g : ℝ → ℝ}

-- Defining the condition from the problem
def g_condition (x : ℝ) (h : x ≠ 0) : g x - 3 * g (1 / x) = 3^x + x^2 := sorry

-- The main statement to prove
theorem find_g3 : g 3 = - (3 * 3^(1/3) + 1/3 + 36) / 8 := sorry

end find_g3_l160_160900


namespace particular_solution_ODE_l160_160390

theorem particular_solution_ODE (y : ℝ → ℝ) (h : ∀ x, deriv y x + y x * Real.tan x = 0) (h₀ : y 0 = 2) :
  ∀ x, y x = 2 * Real.cos x :=
sorry

end particular_solution_ODE_l160_160390


namespace find_p_l160_160680

variable (A B C D p q u v w : ℝ)
variable (hu : u + v + w = -B / A)
variable (huv : u * v + v * w + w * u = C / A)
variable (huvw : u * v * w = -D / A)
variable (hpq : u^2 + v^2 = -p)
variable (hq : u^2 * v^2 = q)

theorem find_p (A B C D : ℝ) (u v w : ℝ) 
  (H1 : u + v + w = -B / A)
  (H2 : u * v + v * w + w * u = C / A)
  (H3 : u * v * w = -D / A)
  (H4 : v = -u - w)
  : p = (B^2 - 2 * C) / A^2 :=
by sorry

end find_p_l160_160680


namespace no_p_safe_numbers_l160_160234

/-- A number n is p-safe if it differs in absolute value by more than 2 from all multiples of p. -/
def p_safe (n p : ℕ) : Prop := ∀ k : ℤ, abs (n - k * p) > 2 

/-- The main theorem stating that there are no numbers that are simultaneously 5-safe, 
    7-safe, and 9-safe from 1 to 15000. -/
theorem no_p_safe_numbers (n : ℕ) (hp : 1 ≤ n ∧ n ≤ 15000) : 
  ¬ (p_safe n 5 ∧ p_safe n 7 ∧ p_safe n 9) :=
sorry

end no_p_safe_numbers_l160_160234


namespace find_multiple_of_son_age_l160_160570

variable (F S k : ℕ)

theorem find_multiple_of_son_age
  (h1 : F = k * S + 4)
  (h2 : F + 4 = 2 * (S + 4) + 20)
  (h3 : F = 44) :
  k = 4 :=
by
  sorry

end find_multiple_of_son_age_l160_160570


namespace sum_divisible_by_12_l160_160237

theorem sum_divisible_by_12 :
  ((2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12) = 3 := by
  sorry

end sum_divisible_by_12_l160_160237


namespace worker_original_daily_wage_l160_160716

-- Given Conditions
def increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]
def new_total_weekly_salary : ℝ := 1457

-- Define the sum of the weekly increases
def total_increase : ℝ := (1 + increases.get! 0) + (1 + increases.get! 1) + (1 + increases.get! 2) + (1 + increases.get! 3) + (1 + increases.get! 4)

-- Main Theorem
theorem worker_original_daily_wage : ∀ (W : ℝ), total_increase * W = new_total_weekly_salary → W = 242.83 :=
by
  intro W h
  sorry

end worker_original_daily_wage_l160_160716


namespace scout_troop_profit_l160_160935

noncomputable def candy_profit (purchase_bars purchase_rate sell_bars sell_rate donation_fraction : ℕ) : ℕ :=
  let cost_price_per_bar := purchase_rate / purchase_bars
  let total_cost := purchase_bars * cost_price_per_bar
  let effective_cost := total_cost * donation_fraction
  let sell_price_per_bar := sell_rate / sell_bars
  let total_revenue := purchase_bars * sell_price_per_bar
  total_revenue - effective_cost

theorem scout_troop_profit :
  candy_profit 1200 3 4 3 1/2 = 700 := by
  sorry

end scout_troop_profit_l160_160935


namespace square_area_l160_160399

theorem square_area (p : ℝ → ℝ) (a b : ℝ) (h₁ : ∀ x, p x = x^2 + 3 * x + 2) (h₂ : p a = 5) (h₃ : p b = 5) (h₄ : a ≠ b) : (b - a)^2 = 21 :=
by
  sorry

end square_area_l160_160399


namespace perimeter_of_field_l160_160635

theorem perimeter_of_field (b l : ℕ) (h1 : l = b + 30) (h2 : b * l = 18000) : 2 * (l + b) = 540 := 
by 
  -- Proof goes here
sorry

end perimeter_of_field_l160_160635


namespace integer_solution_inequality_l160_160387

theorem integer_solution_inequality (x : ℤ) : ((x - 1)^2 ≤ 4) → ([-1, 0, 1, 2, 3].count x = 5) :=
by
  sorry

end integer_solution_inequality_l160_160387


namespace mod_37_5_l160_160690

theorem mod_37_5 : 37 % 5 = 2 :=
by
  sorry

end mod_37_5_l160_160690


namespace part1_part2_l160_160255

def setA : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def setB (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

theorem part1 : (setB (-1)).union setA = {x : ℝ | -1 < x ∧ x < 3 } := by
  sorry

theorem part2 (k : ℝ) : (setA ∩ setB k = setB k ↔ 0 ≤ k) := by
  sorry

end part1_part2_l160_160255


namespace count_valid_numbers_is_31_l160_160016

def is_valid_digit (n : Nat) : Prop := n = 0 ∨ n = 2 ∨ n = 6 ∨ n = 8

def count_valid_numbers : Nat :=
  let valid_digits := [0, 2, 6, 8]
  let one_digit := valid_digits.filter (λ n => n % 4 = 0)
  let two_digits := valid_digits.product valid_digits |>.filter (λ (a, b) => (10*a + b) % 4 = 0)
  let three_digits := valid_digits.product two_digits |>.filter (λ (a, (b, c)) => (100*a + 10*b + c) % 4 = 0)
  one_digit.length + two_digits.length + three_digits.length

theorem count_valid_numbers_is_31 : count_valid_numbers = 31 := by
  sorry

end count_valid_numbers_is_31_l160_160016


namespace fraction_computation_l160_160253

theorem fraction_computation :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_computation_l160_160253


namespace water_level_in_cubic_tank_is_one_l160_160521

def cubic_tank : Type := {s : ℝ // s > 0}

def water_volume (s : cubic_tank) : ℝ := 
  let ⟨side, _⟩ := s 
  side^3

def water_level (s : cubic_tank) (volume : ℝ) (fill_ratio : ℝ) : ℝ := 
  let ⟨side, _⟩ := s 
  fill_ratio * side

theorem water_level_in_cubic_tank_is_one
  (s : cubic_tank)
  (h1 : water_volume s = 64)
  (h2 : water_volume s / 4 = 16)
  (h3 : 0 < 0.25 ∧ 0.25 ≤ 1) :
  water_level s 16 0.25 = 1 :=
by 
  sorry

end water_level_in_cubic_tank_is_one_l160_160521


namespace twenty_four_multiples_of_4_l160_160323

theorem twenty_four_multiples_of_4 {n : ℕ} : (n = 104) ↔ (∃ k : ℕ, k = 24 ∧ ∀ m : ℕ, (12 ≤ m ∧ m ≤ n) → ∃ t : ℕ, m = 12 + 4 * t ∧ 1 ≤ t ∧ t ≤ 24) := 
by
  sorry

end twenty_four_multiples_of_4_l160_160323


namespace sum_g_squared_l160_160263

noncomputable def g (n : ℕ) : ℝ :=
  ∑' m, if m ≥ 3 then 1 / (m : ℝ)^n else 0

theorem sum_g_squared :
  (∑' n, if n ≥ 3 then (g n)^2 else 0) = 1 / 288 :=
by
  sorry

end sum_g_squared_l160_160263


namespace triangle_area_is_9sqrt2_l160_160601

noncomputable def triangle_area_with_given_medians_and_angle (CM BN : ℝ) (angle_BKM : ℝ) : ℝ :=
  let centroid_division_ratio := (2.0 / 3.0)
  let BK := centroid_division_ratio * BN
  let MK := (1.0 / 3.0) * CM
  let area_BKM := (1.0 / 2.0) * BK * MK * Real.sin angle_BKM
  6.0 * area_BKM

theorem triangle_area_is_9sqrt2 :
  triangle_area_with_given_medians_and_angle 6 4.5 (Real.pi / 4) = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_area_is_9sqrt2_l160_160601


namespace price_per_hotdog_l160_160199

-- The conditions
def hot_dogs_per_hour := 10
def hours := 10
def total_sales := 200

-- Conclusion we need to prove
theorem price_per_hotdog : total_sales / (hot_dogs_per_hour * hours) = 2 := by
  sorry

end price_per_hotdog_l160_160199


namespace cantaloupe_total_l160_160503

theorem cantaloupe_total (Fred Tim Alicia : ℝ) 
  (hFred : Fred = 38.5) 
  (hTim : Tim = 44.2)
  (hAlicia : Alicia = 29.7) : 
  Fred + Tim + Alicia = 112.4 :=
by
  sorry

end cantaloupe_total_l160_160503


namespace cage_cost_correct_l160_160464

noncomputable def total_amount_paid : ℝ := 20
noncomputable def change_received : ℝ := 0.26
noncomputable def cat_toy_cost : ℝ := 8.77
noncomputable def cage_cost := total_amount_paid - change_received

theorem cage_cost_correct : cage_cost = 19.74 := by
  sorry

end cage_cost_correct_l160_160464


namespace find_opposite_of_neg_half_l160_160945

-- Define the given number
def given_num : ℚ := -1/2

-- Define what it means to find the opposite of a number
def opposite (x : ℚ) : ℚ := -x

-- State the theorem
theorem find_opposite_of_neg_half : opposite given_num = 1/2 :=
by
  -- Proof is omitted for now
  sorry

end find_opposite_of_neg_half_l160_160945


namespace simplify_and_evaluate_l160_160683

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3)) - (x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  sorry

end simplify_and_evaluate_l160_160683


namespace no_real_pairs_for_same_lines_l160_160449

theorem no_real_pairs_for_same_lines : ¬ ∃ (a b : ℝ), (∀ x y : ℝ, 2 * x + a * y + b = 0 ↔ b * x - 3 * y + 15 = 0) :=
by {
  sorry
}

end no_real_pairs_for_same_lines_l160_160449


namespace unattainable_y_l160_160867

theorem unattainable_y (x : ℝ) (h1 : x ≠ -3/2) : y = (1 - x) / (2 * x + 3) -> ¬(y = -1 / 2) :=
by sorry

end unattainable_y_l160_160867


namespace fraction_of_x_by_110_l160_160575

theorem fraction_of_x_by_110 (x : ℝ) (f : ℝ) (h1 : 0.6 * x = f * x + 110) (h2 : x = 412.5) : f = 1 / 3 :=
by 
  sorry

end fraction_of_x_by_110_l160_160575


namespace expansion_coefficient_l160_160936

theorem expansion_coefficient :
  ∀ (x : ℝ), (∃ (a₀ a₁ a₂ b : ℝ), x^6 + x^4 = a₀ + a₁ * (x + 2) + a₂ * (x + 2)^2 + b * (x + 2)^3) →
  (a₀ = 0 ∧ a₁ = 0 ∧ a₂ = 0 ∧ b = -168) :=
by
  sorry

end expansion_coefficient_l160_160936


namespace speed_with_stream_l160_160030

variable (V_as V_m V_ws : ℝ)

theorem speed_with_stream (h1 : V_as = 6) (h2 : V_m = 2) : V_ws = V_m + (V_as - V_m) :=
by
  sorry

end speed_with_stream_l160_160030


namespace problem_f_increasing_l160_160411

theorem problem_f_increasing (a : ℝ) 
  (h1 : ∀ x, 2 ≤ x → 0 < x^2 - a * x + 3 * a) 
  (h2 : ∀ x, 2 ≤ x → 0 ≤ 2 * x - a) : 
  -4 < a ∧ a ≤ 4 := by
  sorry

end problem_f_increasing_l160_160411


namespace train_passes_jogger_in_46_seconds_l160_160991

-- Definitions directly from conditions
def jogger_speed_kmh : ℕ := 10
def train_speed_kmh : ℕ := 46
def initial_distance_m : ℕ := 340
def train_length_m : ℕ := 120

-- Additional computed definitions based on conditions
def relative_speed_ms : ℕ := (train_speed_kmh - jogger_speed_kmh) * 1000 / 3600
def total_distance_m : ℕ := initial_distance_m + train_length_m

-- Prove that the time it takes for the train to pass the jogger is 46 seconds
theorem train_passes_jogger_in_46_seconds : total_distance_m / relative_speed_ms = 46 := by
  sorry

end train_passes_jogger_in_46_seconds_l160_160991


namespace part1_part2_l160_160896

theorem part1 (a x y : ℝ) (h1 : 3 * x - y = 2 * a - 5) (h2 : x + 2 * y = 3 * a + 3)
  (hx : x > 0) (hy : y > 0) : a > 1 :=
sorry

theorem part2 (a b : ℝ) (ha : a > 1) (h3 : a - b = 4) (hb : b < 2) : 
  -2 < a + b ∧ a + b < 8 :=
sorry

end part1_part2_l160_160896


namespace perpendicular_lines_k_value_l160_160047

theorem perpendicular_lines_k_value (k : ℝ) : 
  k * (k - 1) + (1 - k) * (2 * k + 3) = 0 ↔ k = -3 ∨ k = 1 :=
by
  sorry

end perpendicular_lines_k_value_l160_160047


namespace support_percentage_correct_l160_160667

-- Define the total number of government employees and the percentage supporting the project
def num_gov_employees : ℕ := 150
def perc_gov_support : ℝ := 0.70

-- Define the total number of citizens and the percentage supporting the project
def num_citizens : ℕ := 800
def perc_citizens_support : ℝ := 0.60

-- Calculate the number of supporters among government employees
def gov_supporters : ℝ := perc_gov_support * num_gov_employees

-- Calculate the number of supporters among citizens
def citizens_supporters : ℝ := perc_citizens_support * num_citizens

-- Calculate the total number of people surveyed and the total number of supporters
def total_surveyed : ℝ := num_gov_employees + num_citizens
def total_supporters : ℝ := gov_supporters + citizens_supporters

-- Define the expected correct answer percentage
def correct_percentage_supporters : ℝ := 61.58

-- Prove that the percentage of overall supporters is equal to the expected correct percentage 
theorem support_percentage_correct :
  (total_supporters / total_surveyed * 100) = correct_percentage_supporters :=
by
  sorry

end support_percentage_correct_l160_160667


namespace sqrt_nested_eq_five_l160_160315

theorem sqrt_nested_eq_five {x : ℝ} (h : x = Real.sqrt (15 + x)) : x = 5 :=
sorry

end sqrt_nested_eq_five_l160_160315


namespace find_second_number_l160_160075

theorem find_second_number
  (first_number : ℕ)
  (second_number : ℕ)
  (h1 : first_number = 45)
  (h2 : first_number / second_number = 5) : second_number = 9 :=
by
  -- Proof goes here
  sorry

end find_second_number_l160_160075


namespace min_sum_weights_l160_160055

theorem min_sum_weights (S : ℕ) (h1 : S > 280) (h2 : S % 70 = 30) : S = 310 :=
sorry

end min_sum_weights_l160_160055


namespace tangent_parallel_points_l160_160942

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∃ (x0 y0 : ℝ), (curve x0 = y0) ∧ 
                 (deriv curve x0 = 4) ∧
                 ((x0 = 1 ∧ y0 = 0) ∨ (x0 = -1 ∧ y0 = -4)) :=
by
  sorry

end tangent_parallel_points_l160_160942


namespace min_rectangles_needed_l160_160166

theorem min_rectangles_needed 
  (type1_corners type2_corners : ℕ)
  (rectangles_cover : ℕ → ℕ)
  (h1 : type1_corners = 12)
  (h2 : type2_corners = 12)
  (h3 : ∀ n, rectangles_cover (3 * n) = n) : 
  (rectangles_cover type2_corners) + (rectangles_cover type1_corners) = 12 := 
sorry

end min_rectangles_needed_l160_160166


namespace lcm_3_15_is_15_l160_160564

theorem lcm_3_15_is_15 : Nat.lcm 3 15 = 15 :=
sorry

end lcm_3_15_is_15_l160_160564


namespace _l160_160874

lemma power_of_a_point_theorem (AP BP CP DP : ℝ) (hAP : AP = 5) (hCP : CP = 2) (h_theorem : AP * BP = CP * DP) :
  BP / DP = 2 / 5 :=
by
  sorry

end _l160_160874


namespace annie_initial_money_l160_160743

def cost_of_hamburgers (n : Nat) : Nat := n * 4
def cost_of_milkshakes (m : Nat) : Nat := m * 5
def total_cost (n m : Nat) : Nat := cost_of_hamburgers n + cost_of_milkshakes m
def initial_money (n m left : Nat) : Nat := total_cost n m + left

theorem annie_initial_money : initial_money 8 6 70 = 132 := by
  sorry

end annie_initial_money_l160_160743


namespace find_p_l160_160510

-- Assume the parametric equations and conditions specified in the problem.
noncomputable def parabola_eqns (p t : ℝ) (M E F : ℝ × ℝ) :=
  ∃ m : ℝ,
    (M = (6, m)) ∧
    (E = (-p / 2, m)) ∧
    (F = (p / 2, 0)) ∧
    (m^2 = 6 * p) ∧
    (|E.1 - F.1|^2 + |E.2 - F.2|^2 = |F.1 - M.1|^2 + |F.2 - M.2|^2) ∧
    (|F.1 - M.1|^2 + |F.2 - M.2|^2 = (F.1 + p / 2)^2 + (F.2 - m)^2)

theorem find_p {p t : ℝ} {M E F : ℝ × ℝ} (h : parabola_eqns p t M E F) : p = 4 :=
by
  sorry

end find_p_l160_160510


namespace initial_black_pens_correct_l160_160404

-- Define the conditions
def initial_blue_pens : ℕ := 9
def removed_blue_pens : ℕ := 4
def remaining_blue_pens : ℕ := initial_blue_pens - removed_blue_pens

def initial_red_pens : ℕ := 6
def removed_red_pens : ℕ := 0
def remaining_red_pens : ℕ := initial_red_pens - removed_red_pens

def total_remaining_pens : ℕ := 25
def removed_black_pens : ℕ := 7

-- Assume B is the initial number of black pens
def B : ℕ := 21

-- Prove the initial number of black pens condition
theorem initial_black_pens_correct : 
  (initial_blue_pens + B + initial_red_pens) - (removed_blue_pens + removed_black_pens) = total_remaining_pens :=
by 
  have h1 : initial_blue_pens - removed_blue_pens = remaining_blue_pens := rfl
  have h2 : initial_red_pens - removed_red_pens = remaining_red_pens := rfl
  have h3 : remaining_blue_pens + (B - removed_black_pens) + remaining_red_pens = total_remaining_pens := sorry
  exact h3

end initial_black_pens_correct_l160_160404


namespace initial_ratio_of_liquids_l160_160346

theorem initial_ratio_of_liquids (A B : ℕ) (H1 : A = 21)
  (H2 : 9 * A = 7 * (B + 9)) :
  A / B = 7 / 6 :=
sorry

end initial_ratio_of_liquids_l160_160346


namespace lines_intersect_value_k_l160_160009

theorem lines_intersect_value_k :
  ∀ (x y k : ℝ), (-3 * x + y = k) → (2 * x + y = 20) → (x = -10) → (k = 70) :=
by
  intros x y k h1 h2 h3
  sorry

end lines_intersect_value_k_l160_160009


namespace sequence_term_l160_160963

open Int

-- Define the sequence {S_n} as stated in the problem
def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

-- Define the sequence {a_n} as the finite difference of {S_n}
def a (n : ℕ) : ℤ := if n = 1 then -1 else S n - S (n - 1)

-- The theorem statement
theorem sequence_term (n : ℕ) (hn : n > 0) : a n = 4 * n - 5 :=
by sorry

end sequence_term_l160_160963


namespace intersection_complement_l160_160518

open Set

variable (U A B : Set ℕ)

-- Given conditions:
def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3}

theorem intersection_complement (U A B : Set ℕ) : 
  U = universal_set → A = set_A → B = set_B → (A ∩ (U \ B)) = {1, 5} := by
  sorry

end intersection_complement_l160_160518


namespace tetrahedron_volume_correct_l160_160174

noncomputable def tetrahedron_volume (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABD_ABC : ℝ) : ℝ :=
  let h_ABD := (2 * area_ABD) / AB
  let h := h_ABD * Real.sin angle_ABD_ABC
  (1 / 3) * area_ABC * h

theorem tetrahedron_volume_correct:
  tetrahedron_volume 3 15 12 (Real.pi / 6) = 20 :=
by
  sorry

end tetrahedron_volume_correct_l160_160174


namespace max_tiles_on_floor_l160_160451

   -- Definitions corresponding to conditions
   def tile_length_1 : ℕ := 35
   def tile_width_1 : ℕ := 30
   def tile_length_2 : ℕ := 30
   def tile_width_2 : ℕ := 35
   def floor_length : ℕ := 1000
   def floor_width : ℕ := 210

   -- Conditions:
   -- 1. Tiles do not overlap.
   -- 2. Tiles are placed with edges jutting against each other on all edges.
   -- 3. A tile can be placed in any orientation so long as its edges are parallel to the edges of the floor.
   -- 4. No tile should overshoot any edge of the floor.

   theorem max_tiles_on_floor :
     let tiles_orientation_1 := (floor_length / tile_length_1) * (floor_width / tile_width_1)
     let tiles_orientation_2 := (floor_length / tile_length_2) * (floor_width / tile_width_2)
     max tiles_orientation_1 tiles_orientation_2 = 198 :=
   by {
     -- The actual proof handling is skipped, as per instructions.
     sorry
   }
   
end max_tiles_on_floor_l160_160451


namespace normal_line_at_point_l160_160853

noncomputable def curve (x : ℝ) : ℝ := (4 * x - x ^ 2) / 4

theorem normal_line_at_point (x0 : ℝ) (h : x0 = 2) :
  ∃ (L : ℝ → ℝ), ∀ (x : ℝ), L x = (2 : ℝ) :=
by
  sorry

end normal_line_at_point_l160_160853


namespace expression_square_minus_three_times_l160_160195

-- Defining the statement
theorem expression_square_minus_three_times (a b : ℝ) : a^2 - 3 * b = a^2 - 3 * b := 
by
  sorry

end expression_square_minus_three_times_l160_160195


namespace value_of_a_l160_160033

theorem value_of_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -3) (h3 : a * x - y = 1) : a = -2 :=
by
  -- Placeholder for the proof
  sorry

end value_of_a_l160_160033


namespace range_of_a_l160_160439

theorem range_of_a (a : ℝ) (x : ℝ) : (∃ x, x^2 - a*x - a ≤ -3) → (a ≤ -6 ∨ a ≥ 2) :=
sorry

end range_of_a_l160_160439


namespace find_number_l160_160202

theorem find_number (some_number : ℤ) : 45 - (28 - (some_number - (15 - 19))) = 58 ↔ some_number = 37 := 
by 
  sorry

end find_number_l160_160202


namespace exists_x_given_y_l160_160531

theorem exists_x_given_y (y : ℝ) : ∃ x : ℝ, x^2 + y^2 = 10 ∧ x^2 - x * y - 3 * y + 12 = 0 := 
sorry

end exists_x_given_y_l160_160531


namespace green_shirt_pairs_l160_160391

theorem green_shirt_pairs (r g : ℕ) (p total_pairs red_pairs : ℕ) :
  r = 63 → g = 69 → p = 66 → red_pairs = 25 → (g - (r - red_pairs * 2)) / 2 = 28 :=
by
  intros hr hg hp hred_pairs
  sorry

end green_shirt_pairs_l160_160391


namespace num_four_digit_with_5_or_7_l160_160858

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end num_four_digit_with_5_or_7_l160_160858


namespace trigonometric_expression_l160_160156

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.sin α = 3 / 5)
  (h2 : α ∈ Set.Ioo (π / 2) π) :
  (Real.cos (2 * α) / (Real.sqrt 2 * Real.sin (α + π / 4))) = -7 / 5 := 
sorry

end trigonometric_expression_l160_160156


namespace compute_fraction_equation_l160_160838

theorem compute_fraction_equation :
  (8 * (2 / 3: ℚ)^4 + 2 = 290 / 81) :=
sorry

end compute_fraction_equation_l160_160838


namespace solve_equation_l160_160120

theorem solve_equation (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60 → x = 4 := by
  sorry

end solve_equation_l160_160120


namespace loss_percentage_is_20_l160_160105

-- Define necessary conditions
def CP : ℕ := 2000
def gain_percent : ℕ := 6
def SP_new : ℕ := CP + ((gain_percent * CP) / 100)
def increase : ℕ := 520

-- Define the selling price condition
def SP : ℕ := SP_new - increase

-- Define the loss percentage condition
def loss_percent : ℕ := ((CP - SP) * 100) / CP

-- Prove the loss percentage is 20%
theorem loss_percentage_is_20 : loss_percent = 20 :=
by sorry

end loss_percentage_is_20_l160_160105


namespace geom_seq_min_value_l160_160729

open Real

/-- 
Theorem: For a geometric sequence {a_n} where a_n > 0 and a_7 = √2/2, 
the minimum value of 1/a_3 + 2/a_11 is 4.
-/
theorem geom_seq_min_value (a : ℕ → ℝ) (a_pos : ∀ n, 0 < a n) (h7 : a 7 = (sqrt 2) / 2) :
  (1 / (a 3) + 2 / (a 11) >= 4) :=
sorry

end geom_seq_min_value_l160_160729


namespace seeds_total_l160_160993

def seedsPerWatermelon : Nat := 345
def numberOfWatermelons : Nat := 27
def totalSeeds : Nat := seedsPerWatermelon * numberOfWatermelons

theorem seeds_total :
  totalSeeds = 9315 :=
by
  sorry

end seeds_total_l160_160993


namespace simplify_expression_l160_160656

theorem simplify_expression : 
  2 ^ (-1: ℤ) + Real.sqrt 16 - (3 - Real.sqrt 3) ^ 0 + |Real.sqrt 2 - 1 / 2| = 3 + Real.sqrt 2 := by
  sorry

end simplify_expression_l160_160656


namespace two_digit_number_with_tens_5_l160_160924

-- Definitions and conditions
variable (A : Nat)

-- Problem statement as a Lean theorem
theorem two_digit_number_with_tens_5 (hA : A < 10) : (10 * 5 + A) = 50 + A := by
  sorry

end two_digit_number_with_tens_5_l160_160924


namespace circle_equation_l160_160344

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let radius : ℝ := dist center point
  dist center point = 1 → 
  (x - 1)^2 + y^2 = radius^2 :=
by
  intros
  sorry

end circle_equation_l160_160344


namespace range_of_m_l160_160043

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (m * x^2 + (m - 3) * x + 1 = 0)) →
  m ∈ Set.Iic 1 := by
  sorry

end range_of_m_l160_160043


namespace prime_solution_l160_160914

theorem prime_solution (p : ℕ) (x y : ℕ) (hp : Prime p) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 → p = 2 ∨ p = 3 :=
by
  sorry

end prime_solution_l160_160914


namespace total_shaded_area_correct_l160_160374
-- Let's import the mathematical library.

-- Define the problem-related conditions.
def first_rectangle_length : ℕ := 4
def first_rectangle_width : ℕ := 15
def second_rectangle_length : ℕ := 5
def second_rectangle_width : ℕ := 12
def third_rectangle_length : ℕ := 2
def third_rectangle_width : ℕ := 2

-- Define the areas based on the problem conditions.
def A1 : ℕ := first_rectangle_length * first_rectangle_width
def A2 : ℕ := second_rectangle_length * second_rectangle_width
def A_overlap_12 : ℕ := first_rectangle_length * second_rectangle_length
def A3 : ℕ := third_rectangle_length * third_rectangle_width

-- Define the total shaded area formula.
def total_shaded_area : ℕ := A1 + A2 - A_overlap_12 + A3

-- Statement of the theorem to prove.
theorem total_shaded_area_correct :
  total_shaded_area = 104 :=
by
  sorry

end total_shaded_area_correct_l160_160374


namespace ratio_new_circumference_to_original_diameter_l160_160669

-- Define the problem conditions
variables (r k : ℝ) (hk : k > 0)

-- Define the Lean theorem to express the proof problem
theorem ratio_new_circumference_to_original_diameter (r k : ℝ) (hk : k > 0) :
  (π * (1 + k / r)) = (2 * π * (r + k)) / (2 * r) :=
by {
  -- Placeholder proof, to be filled in
  sorry
}

end ratio_new_circumference_to_original_diameter_l160_160669


namespace james_payment_l160_160372

theorem james_payment (james_meal : ℕ) (friend_meal : ℕ) (tip_percent : ℕ) (final_payment : ℕ) : 
  james_meal = 16 → 
  friend_meal = 14 → 
  tip_percent = 20 → 
  final_payment = 18 :=
by
  -- Definitions
  let total_bill_before_tip := james_meal + friend_meal
  let tip := total_bill_before_tip * tip_percent / 100
  let final_bill := total_bill_before_tip + tip
  let half_bill := final_bill / 2
  -- Proof (to be filled in)
  sorry

end james_payment_l160_160372


namespace coin_flip_sequences_count_l160_160383

theorem coin_flip_sequences_count : (2 ^ 16) = 65536 :=
by
  sorry

end coin_flip_sequences_count_l160_160383


namespace no_integer_solutions_l160_160579

theorem no_integer_solutions (x y z : ℤ) (h₀ : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solutions_l160_160579


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l160_160961

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : x < -1) : 2 * x ^ 2 + x - 1 > 0 :=
by sorry

theorem not_necessary_condition (h2 : 2 * x ^ 2 + x - 1 > 0) : x > 1/2 ∨ x < -1 :=
by sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l160_160961


namespace sum_mean_median_mode_l160_160389

theorem sum_mean_median_mode (l : List ℚ) (h : l = [1, 2, 2, 3, 3, 3, 3, 4, 5]) :
    let mean := (1 + 2 + 2 + 3 + 3 + 3 + 3 + 4 + 5) / 9
    let median := 3
    let mode := 3
    mean + median + mode = 8.888 :=
by
  sorry

end sum_mean_median_mode_l160_160389


namespace necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l160_160766

theorem necessary_ab_given_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 4) : 
  a + b ≥ 4 :=
sorry

theorem not_sufficient_ab_given_a_b : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4 :=
sorry

end necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l160_160766


namespace isosceles_triangle_perimeter_l160_160637

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l160_160637


namespace general_term_formula_l160_160178

def sequence_sum (n : ℕ) : ℕ := 3 * n^2 - 2 * n

def general_term (n : ℕ) : ℕ := if n = 0 then 0 else 6 * n - 5

theorem general_term_formula (n : ℕ) (h : n > 0) :
  general_term n = sequence_sum n - sequence_sum (n - 1) := by
  sorry

end general_term_formula_l160_160178


namespace integer_sum_of_squares_power_l160_160815

theorem integer_sum_of_squares_power (a p q : ℤ) (k : ℕ) (h : a = p^2 + q^2) : 
  ∃ c d : ℤ, a^k = c^2 + d^2 := 
sorry

end integer_sum_of_squares_power_l160_160815


namespace inscribed_square_proof_l160_160567

theorem inscribed_square_proof :
  (∃ (r : ℝ), 2 * π * r = 72 * π ∧ r = 36) ∧ 
  (∃ (s : ℝ), (2 * (36:ℝ))^2 = 2 * s ^ 2 ∧ s = 36 * Real.sqrt 2) :=
by
  sorry

end inscribed_square_proof_l160_160567


namespace complex_product_conjugate_l160_160603

theorem complex_product_conjugate : (1 + Complex.I) * (1 - Complex.I) = 2 := 
by 
  -- Lean proof goes here
  sorry

end complex_product_conjugate_l160_160603


namespace sum_of_reciprocals_l160_160932

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 1 / x + 1 / y = 8 / 75 :=
by
  sorry

end sum_of_reciprocals_l160_160932


namespace negation_equiv_l160_160129

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 + 1 ≥ 1

-- Negation of the original proposition
def negated_prop : Prop := ∃ x : ℝ, x^2 + 1 < 1

-- Main theorem stating the equivalence
theorem negation_equiv :
  (¬ (∀ x : ℝ, original_prop x)) ↔ negated_prop :=
by sorry

end negation_equiv_l160_160129


namespace glass_sphere_wall_thickness_l160_160852

/-- Mathematically equivalent proof problem statement:
Given a hollow glass sphere with outer diameter 16 cm such that 3/8 of its surface remains dry,
and specific gravity of glass s = 2.523. The wall thickness of the sphere is equal to 0.8 cm. -/
theorem glass_sphere_wall_thickness 
  (outer_diameter : ℝ) (dry_surface_fraction : ℝ) (specific_gravity : ℝ) (required_thickness : ℝ) 
  (uniform_thickness : outer_diameter = 16)
  (dry_surface : dry_surface_fraction = 3 / 8)
  (s : specific_gravity = 2.523) :
  required_thickness = 0.8 :=
by
  sorry

end glass_sphere_wall_thickness_l160_160852


namespace different_rhetorical_device_in_optionA_l160_160827

def optionA_uses_metaphor : Prop :=
  -- Here, define the condition explaining that Option A uses metaphor
  true -- This will denote that Option A uses metaphor 

def optionsBCD_use_personification : Prop :=
  -- Here, define the condition explaining that Options B, C, and D use personification
  true -- This will denote that Options B, C, and D use personification

theorem different_rhetorical_device_in_optionA :
  optionA_uses_metaphor ∧ optionsBCD_use_personification → 
  (∃ (A P : Prop), A ≠ P) :=
by
  -- No proof is required as per instructions
  intro h
  exact Exists.intro optionA_uses_metaphor (Exists.intro optionsBCD_use_personification sorry)

end different_rhetorical_device_in_optionA_l160_160827


namespace triangle_area_l160_160811

variables {A B C D M N: Type}

-- Define the conditions and the proof 
theorem triangle_area
  (α β : ℝ)
  (CD : ℝ)
  (sin_Ratio : ℝ)
  (C_angle : ℝ)
  (MCN_Area : ℝ)
  (M_distance : ℝ)
  (N_distance : ℝ)
  (hCD : CD = Real.sqrt 13)
  (hSinRatio : (Real.sin α) / (Real.sin β) = 4 / 3)
  (hC_angle : C_angle = 120)
  (hMCN_Area : MCN_Area = 3 * Real.sqrt 3)
  (hDistance : M_distance = 2 * N_distance)
  : ∃ ABC_Area, ABC_Area = 27 * Real.sqrt 3 / 2 :=
sorry

end triangle_area_l160_160811


namespace pen_and_notebook_cost_l160_160185

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 17 * p + 5 * n = 200 ∧ p > n ∧ p + n = 16 := 
by
  sorry

end pen_and_notebook_cost_l160_160185


namespace min_area_and_line_eq_l160_160587

theorem min_area_and_line_eq (a b : ℝ) (l : ℝ → ℝ → Prop)
    (h1 : l 3 2)
    (h2: ∀ x y: ℝ, l x y → (x/a + y/b = 1))
    (h3: a > 0)
    (h4: b > 0)
    : 
    a = 6 ∧ b = 4 ∧ 
    (∀ x y : ℝ, l x y ↔ (4 * x + 6 * y - 24 = 0)) ∧ 
    (∃ min_area : ℝ, min_area = 12) :=
by
  sorry

end min_area_and_line_eq_l160_160587


namespace simplify_exponent_multiplication_l160_160892

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end simplify_exponent_multiplication_l160_160892


namespace stairs_left_to_climb_l160_160862

def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

theorem stairs_left_to_climb : total_stairs - climbed_stairs = 22 := by
  sorry

end stairs_left_to_climb_l160_160862


namespace find_x_l160_160321

theorem find_x :
  ∃ x : Real, abs (x - 0.052) < 1e-3 ∧
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + x^2 + 0.0035^2) = 100 :=
by
  sorry

end find_x_l160_160321


namespace unique_suwy_product_l160_160734

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then Char.toNat c - Char.toNat 'A' + 1 else 0

def product_of_chars (l : List Char) : Nat :=
  l.foldr (λ c acc => letter_value c * acc) 1

theorem unique_suwy_product :
  ∀ (l : List Char), l.length = 4 → product_of_chars l = 19 * 21 * 23 * 25 → l = ['S', 'U', 'W', 'Y'] := 
by
  intro l hlen hproduct
  sorry

end unique_suwy_product_l160_160734


namespace fraction_decomposition_l160_160813
noncomputable def A := (48 : ℚ) / 17
noncomputable def B := (-(25 : ℚ) / 17)

theorem fraction_decomposition (A : ℚ) (B : ℚ) :
  ( ∀ x : ℚ, x ≠ -5 ∧ x ≠ 2/3 →
    (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) ) ↔ 
    (A = (48 : ℚ) / 17 ∧ B = (-(25 : ℚ) / 17)) :=
by
  sorry

end fraction_decomposition_l160_160813


namespace average_words_per_puzzle_l160_160064

-- Define the conditions
def uses_up_pencil_every_two_weeks : Prop := ∀ (days_used : ℕ), days_used = 14
def words_to_use_up_pencil : ℕ := 1050
def puzzles_completed_per_day : ℕ := 1

-- Problem statement: Prove the average number of words in each crossword puzzle
theorem average_words_per_puzzle :
  (words_to_use_up_pencil / 14 = 75) :=
by
  -- Definitions used directly from the conditions
  sorry

end average_words_per_puzzle_l160_160064


namespace contrapositive_x_squared_eq_one_l160_160384

theorem contrapositive_x_squared_eq_one (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) := by
  sorry

end contrapositive_x_squared_eq_one_l160_160384


namespace teams_points_l160_160931

-- Definitions of teams and points
inductive Team
| A | B | C | D | E
deriving DecidableEq

def points : Team → ℕ
| Team.A => 6
| Team.B => 5
| Team.C => 4
| Team.D => 3
| Team.E => 2

-- Conditions
axiom no_draws_A : ∀ t : Team, t ≠ Team.A → (points Team.A ≠ points t)
axiom no_loses_B : ∀ t : Team, t ≠ Team.B → (points Team.B > points t) ∨ (points Team.B = points t)
axiom no_wins_D : ∀ t : Team, t ≠ Team.D → (points Team.D < points t)
axiom unique_scores : ∀ (t1 t2 : Team), t1 ≠ t2 → points t1 ≠ points t2

-- Theorem
theorem teams_points :
  points Team.A = 6 ∧
  points Team.B = 5 ∧
  points Team.C = 4 ∧
  points Team.D = 3 ∧
  points Team.E = 2 :=
by
  sorry

end teams_points_l160_160931


namespace packs_with_extra_red_pencils_eq_3_l160_160251

def total_packs : Nat := 15
def regular_red_per_pack : Nat := 1
def total_red_pencils : Nat := 21
def extra_red_per_pack : Nat := 2

theorem packs_with_extra_red_pencils_eq_3 :
  ∃ (packs_with_extra : Nat), packs_with_extra * extra_red_per_pack + (total_packs - packs_with_extra) * regular_red_per_pack = total_red_pencils ∧ packs_with_extra = 3 :=
by
  sorry

end packs_with_extra_red_pencils_eq_3_l160_160251


namespace problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l160_160031

theorem problem_a_lt_b_lt_0_implies_ab_gt_b_sq (a b : ℝ) (h : a < b ∧ b < 0) : ab > b^2 := by
  sorry

end problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l160_160031


namespace find_d_values_l160_160798

open Set

theorem find_d_values :
  ∀ {f : ℝ → ℝ}, ContinuousOn f (Icc 0 1) → (f 0 = f 1) →
  ∃ (d : ℝ), d ∈ Ioo 0 1 ∧ (∀ x₀, x₀ ∈ Icc 0 (1 - d) → (f x₀ = f (x₀ + d))) ↔
  ∃ k : ℕ, d = 1 / k :=
by
  sorry

end find_d_values_l160_160798


namespace price_of_fifth_basket_l160_160233

-- Define the initial conditions
def avg_cost_of_4_baskets (total_cost_4 : ℝ) : Prop :=
  total_cost_4 / 4 = 4

def avg_cost_of_5_baskets (total_cost_5 : ℝ) : Prop :=
  total_cost_5 / 5 = 4.8

-- Theorem statement to be proved
theorem price_of_fifth_basket
  (total_cost_4 : ℝ)
  (h1 : avg_cost_of_4_baskets total_cost_4)
  (total_cost_5 : ℝ)
  (h2 : avg_cost_of_5_baskets total_cost_5) :
  total_cost_5 - total_cost_4 = 8 :=
by
  sorry

end price_of_fifth_basket_l160_160233


namespace initial_hamburgers_count_is_nine_l160_160056

-- Define the conditions
def hamburgers_initial (total_hamburgers : ℕ) (additional_hamburgers : ℕ) : ℕ :=
  total_hamburgers - additional_hamburgers

-- The statement to be proved
theorem initial_hamburgers_count_is_nine :
  hamburgers_initial 12 3 = 9 :=
by
  sorry

end initial_hamburgers_count_is_nine_l160_160056


namespace book_loss_percentage_l160_160889

theorem book_loss_percentage (CP SP_profit SP_loss : ℝ) (L : ℝ) 
  (h1 : CP = 50) 
  (h2 : SP_profit = CP + 0.09 * CP) 
  (h3 : SP_loss = CP - L / 100 * CP) 
  (h4 : SP_profit - SP_loss = 9) : 
  L = 9 :=
by
  sorry

end book_loss_percentage_l160_160889


namespace find_a_share_l160_160131

noncomputable def total_investment (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def total_profit (b_share total_inv b_inv : ℕ) : ℕ :=
  b_share * total_inv / b_inv

noncomputable def a_share (a_inv total_inv total_pft : ℕ) : ℕ :=
  a_inv * total_pft / total_inv

theorem find_a_share
  (a_inv b_inv c_inv b_share : ℕ)
  (h1 : a_inv = 7000)
  (h2 : b_inv = 11000)
  (h3 : c_inv = 18000)
  (h4 : b_share = 880) :
  a_share a_inv (total_investment a_inv b_inv c_inv) (total_profit b_share (total_investment a_inv b_inv c_inv) b_inv) = 560 := 
by
  sorry

end find_a_share_l160_160131


namespace find_other_number_l160_160484

-- Defining the two numbers and their properties
def sum_is_84 (a b : ℕ) : Prop := a + b = 84
def one_is_36 (a b : ℕ) : Prop := a = 36 ∨ b = 36
def other_is_48 (a b : ℕ) : Prop := a = 48 ∨ b = 48

-- The theorem statement
theorem find_other_number (a b : ℕ) (h1 : sum_is_84 a b) (h2 : one_is_36 a b) : other_is_48 a b :=
by {
  sorry
}

end find_other_number_l160_160484


namespace min_initial_questionnaires_l160_160784

theorem min_initial_questionnaires 
(N : ℕ) 
(h1 : 0.60 * (N:ℝ) + 0.60 * (N:ℝ) * 0.80 + 0.60 * (N:ℝ) * (0.80^2) ≥ 750) : 
  N ≥ 513 := sorry

end min_initial_questionnaires_l160_160784


namespace solve_quadratic_for_negative_integer_l160_160007

theorem solve_quadratic_for_negative_integer (N : ℤ) (h_neg : N < 0) (h_eq : 2 * N^2 + N = 20) : N = -4 :=
sorry

end solve_quadratic_for_negative_integer_l160_160007


namespace find_solutions_l160_160071

def system_solutions (x y z : ℝ) : Prop :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solutions :
  ∃ (x y z : ℝ), system_solutions x y z ∧ ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end find_solutions_l160_160071


namespace meal_cost_l160_160266

theorem meal_cost (total_people total_bill : ℕ) (h1 : total_people = 2 + 5) (h2 : total_bill = 21) :
  total_bill / total_people = 3 := by
  sorry

end meal_cost_l160_160266


namespace train_speed_in_kmph_l160_160610

-- Definitions for the given problem conditions
def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 240
def time_to_cross_bridge : ℝ := 20.99832013438925

-- Main theorem statement
theorem train_speed_in_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60.0084 := 
by
  sorry

end train_speed_in_kmph_l160_160610


namespace trigonometric_identity_proof_l160_160356

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h : Real.tan α = (1 + Real.sin β) / Real.cos β)

theorem trigonometric_identity_proof : 2 * α - β = π / 2 := 
by 
  sorry

end trigonometric_identity_proof_l160_160356


namespace inequalities_hold_l160_160207

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b) ≥ 2 := by
  sorry

end inequalities_hold_l160_160207


namespace tan_alpha_tan_beta_value_l160_160376

theorem tan_alpha_tan_beta_value
  (α β : ℝ)
  (h1 : Real.cos (α + β) = 1 / 5)
  (h2 : Real.cos (α - β) = 3 / 5) :
  Real.tan α * Real.tan β = 1 / 2 :=
by
  sorry

end tan_alpha_tan_beta_value_l160_160376


namespace smaller_angle_at_seven_oclock_l160_160605

def degree_measure_of_smaller_angle (hours : ℕ) : ℕ :=
  let complete_circle := 360
  let hour_segments := 12
  let angle_per_hour := complete_circle / hour_segments
  let hour_angle := hours * angle_per_hour
  let smaller_angle := (if hour_angle > complete_circle / 2 then complete_circle - hour_angle else hour_angle)
  smaller_angle

theorem smaller_angle_at_seven_oclock : degree_measure_of_smaller_angle 7 = 150 := 
by
  sorry

end smaller_angle_at_seven_oclock_l160_160605


namespace sandy_painting_area_l160_160130

theorem sandy_painting_area :
  let wall_height := 10
  let wall_length := 15
  let painting_height := 3
  let painting_length := 5
  let wall_area := wall_height * wall_length
  let painting_area := painting_height * painting_length
  let area_to_paint := wall_area - painting_area
  area_to_paint = 135 := 
by 
  sorry

end sandy_painting_area_l160_160130


namespace P_positive_l160_160723

variable (P : ℕ → ℝ)

axiom P_cond_0 : P 0 > 0
axiom P_cond_1 : P 1 > P 0
axiom P_cond_2 : P 2 > 2 * P 1 - P 0
axiom P_cond_3 : P 3 > 3 * P 2 - 3 * P 1 + P 0
axiom P_cond_n : ∀ n, P (n + 4) > 4 * P (n + 3) - 6 * P (n + 2) + 4 * P (n + 1) - P n

theorem P_positive (n : ℕ) (h : n > 0) : P n > 0 := by
  sorry

end P_positive_l160_160723


namespace quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l160_160925

theorem quadrant_606 (θ : ℝ) : θ = 606 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

theorem quadrant_minus_950 (θ : ℝ) : θ = -950 → (90 < (θ % 360) ∧ (θ % 360) < 180) := by
  sorry

theorem same_terminal_side (α k : ℤ) : (α = -457 + k * 360) ↔ (∃ n : ℤ, α = -457 + n * 360) := by
  sorry

theorem quadrant_minus_97 (θ : ℝ) : θ = -97 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

end quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l160_160925


namespace solve_discriminant_l160_160239

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem solve_discriminant : 
  discriminant 2 (2 + 1/2) (1/2) = 2.25 :=
by
  -- The proof can be filled in here
  -- Assuming a = 2, b = 2.5, c = 1/2
  -- discriminant 2 2.5 0.5 will be computed
  sorry

end solve_discriminant_l160_160239


namespace value_of_y_l160_160433

theorem value_of_y (y : ℝ) (h : (3 * y - 9) / 3 = 18) : y = 21 :=
sorry

end value_of_y_l160_160433


namespace solve_expression_l160_160602

theorem solve_expression (x : ℝ) (h : 3 * x - 5 = 10 * x + 9) : 4 * (x + 7) = 20 :=
by
  sorry

end solve_expression_l160_160602


namespace mr_green_potato_yield_l160_160779

theorem mr_green_potato_yield :
  let steps_to_feet := 2.5
  let length_steps := 18
  let width_steps := 25
  let yield_per_sqft := 0.75
  let length_feet := length_steps * steps_to_feet
  let width_feet := width_steps * steps_to_feet
  let area_sqft := length_feet * width_feet
  let expected_yield := area_sqft * yield_per_sqft
  expected_yield = 2109.375 := by sorry

end mr_green_potato_yield_l160_160779


namespace suresh_completion_time_l160_160864

theorem suresh_completion_time (S : ℕ) 
  (ashu_time : ℕ := 30) 
  (suresh_work_time : ℕ := 9) 
  (ashu_remaining_time : ℕ := 12) 
  (ashu_fraction : ℚ := ashu_remaining_time / ashu_time) :
  (suresh_work_time / S + ashu_fraction = 1) → S = 15 :=
by
  intro h
  -- Proof here
  sorry

end suresh_completion_time_l160_160864


namespace largest_angle_consecutive_even_pentagon_l160_160547

theorem largest_angle_consecutive_even_pentagon :
  ∀ (n : ℕ), (2 * n + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 540) →
  (2 * n + 8 = 112) :=
by
  intros n h
  sorry

end largest_angle_consecutive_even_pentagon_l160_160547


namespace juggling_contest_l160_160397

theorem juggling_contest (B : ℕ) (rot_baseball : ℕ := 80)
    (rot_per_apple : ℕ := 101) (num_apples : ℕ := 4)
    (winner_rotations : ℕ := 404) :
    (num_apples * rot_per_apple = winner_rotations) :=
by
  sorry

end juggling_contest_l160_160397


namespace height_of_parallelogram_l160_160566

noncomputable def parallelogram_height (base area : ℝ) : ℝ :=
  area / base

theorem height_of_parallelogram :
  parallelogram_height 8 78.88 = 9.86 :=
by
  -- This is where the proof would go, but it's being omitted as per instructions.
  sorry

end height_of_parallelogram_l160_160566


namespace polygon_sides_l160_160431

theorem polygon_sides (perimeter side_length : ℕ) (h₁ : perimeter = 150) (h₂ : side_length = 15): 
  (perimeter / side_length) = 10 := 
by
  -- Here goes the proof part
  sorry

end polygon_sides_l160_160431


namespace exists_power_of_two_with_last_n_digits_ones_and_twos_l160_160034

theorem exists_power_of_two_with_last_n_digits_ones_and_twos (N : ℕ) (hN : 0 < N) :
  ∃ k : ℕ, ∀ i < N, ∃ (d : ℕ), d = 1 ∨ d = 2 ∧ 
    (2^k % 10^N) / 10^i % 10 = d :=
sorry

end exists_power_of_two_with_last_n_digits_ones_and_twos_l160_160034


namespace total_apples_l160_160363

/-- Problem: 
A fruit stand is selling apples for $2 each. Emmy has $200 while Gerry has $100. 
Prove the total number of apples Emmy and Gerry can buy altogether is 150.
-/
theorem total_apples (p E G : ℕ) (h1: p = 2) (h2: E = 200) (h3: G = 100) : 
  (E / p) + (G / p) = 150 :=
by
  sorry

end total_apples_l160_160363


namespace expression_in_terms_of_x_difference_between_x_l160_160002

variable (E x : ℝ)

theorem expression_in_terms_of_x (h1 : E / (2 * x + 15) = 3) : E = 6 * x + 45 :=
by 
  sorry

variable (x1 x2 : ℝ)

theorem difference_between_x (h1 : E / (2 * x1 + 15) = 3) (h2: E / (2 * x2 + 15) = 3) (h3 : x2 - x1 = 12) : True :=
by 
  sorry

end expression_in_terms_of_x_difference_between_x_l160_160002


namespace solve_equation_l160_160115

theorem solve_equation : ∀ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) → x = 1 :=
by
  intro x
  intro h
  -- The proof would go here
  sorry

end solve_equation_l160_160115


namespace quotient_of_division_l160_160471

theorem quotient_of_division 
  (dividend divisor remainder : ℕ) 
  (h_dividend : dividend = 265) 
  (h_divisor : divisor = 22) 
  (h_remainder : remainder = 1) 
  (h_div : dividend = divisor * (dividend / divisor) + remainder) : 
  (dividend / divisor) = 12 := 
by
  sorry

end quotient_of_division_l160_160471


namespace find_u_l160_160200

variable (α β γ : ℝ)
variables (q s u : ℝ)

-- The first polynomial has roots α, β, γ
axiom roots_first_poly : ∀ x : ℝ, x^3 + 4 * x^2 + 6 * x - 8 = (x - α) * (x - β) * (x - γ)

-- Sum of the roots α + β + γ = -4
axiom sum_roots_first_poly : α + β + γ = -4

-- Product of the roots αβγ = 8
axiom product_roots_first_poly : α * β * γ = 8

-- The second polynomial has roots α + β, β + γ, γ + α
axiom roots_second_poly : ∀ x : ℝ, x^3 + q * x^2 + s * x + u = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))

theorem find_u : u = 32 :=
sorry

end find_u_l160_160200


namespace difference_divisible_by_18_l160_160823

theorem difference_divisible_by_18 (a b : ℤ) : 18 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
by
  sorry

end difference_divisible_by_18_l160_160823


namespace rabbit_speed_l160_160054

theorem rabbit_speed (x : ℕ) :
  2 * (2 * x + 4) = 188 → x = 45 := by
  sorry

end rabbit_speed_l160_160054


namespace find_xy_such_that_product_is_fifth_power_of_prime_l160_160424

theorem find_xy_such_that_product_is_fifth_power_of_prime
  (x y : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (x^2 + y) * (y^2 + x) = p^5) :
  (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) :=
sorry

end find_xy_such_that_product_is_fifth_power_of_prime_l160_160424


namespace A_inter_complement_B_eq_set_minus_one_to_zero_l160_160666

open Set

theorem A_inter_complement_B_eq_set_minus_one_to_zero :
  let U := @univ ℝ
  let A := {x : ℝ | x < 0}
  let B := {x : ℝ | x ≤ -1}
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := 
by
  sorry

end A_inter_complement_B_eq_set_minus_one_to_zero_l160_160666


namespace total_votes_election_l160_160350

theorem total_votes_election (V : ℝ)
    (h1 : 0.55 * 0.8 * V + 2520 = 0.8 * V)
    (h2 : 0.36 > 0) :
    V = 7000 :=
  by
  sorry

end total_votes_election_l160_160350


namespace at_least_one_hit_l160_160327

-- Introduce the predicates
variable (p q : Prop)

-- State the theorem
theorem at_least_one_hit : (¬ (¬ p ∧ ¬ q)) = (p ∨ q) :=
by
  sorry

end at_least_one_hit_l160_160327


namespace equilateral_triangle_l160_160176

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : α + β + γ = π)
  (h8 : a = 2 * Real.sin α)
  (h9 : b = 2 * Real.sin β)
  (h10 : c = 2 * Real.sin γ)
  (h11 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l160_160176


namespace common_root_equation_l160_160285

theorem common_root_equation (a b r : ℝ) (h₁ : a ≠ b)
  (h₂ : r^2 + 2019 * a * r + b = 0)
  (h₃ : r^2 + 2019 * b * r + a = 0) :
  r = 1 / 2019 :=
by
  sorry

end common_root_equation_l160_160285


namespace shaded_region_area_is_correct_l160_160820

noncomputable def area_of_shaded_region : ℝ :=
  let R := 6 -- radius of the larger circle
  let r := R / 2 -- radius of each smaller circle
  let area_large_circle := Real.pi * R^2
  let area_two_small_circles := 2 * Real.pi * r^2
  area_large_circle - area_two_small_circles

theorem shaded_region_area_is_correct :
  area_of_shaded_region = 18 * Real.pi :=
sorry

end shaded_region_area_is_correct_l160_160820


namespace mailman_should_give_junk_mail_l160_160249

-- Definitions from the conditions
def houses_in_block := 20
def junk_mail_per_house := 32

-- The mathematical equivalent proof problem statement in Lean 4
theorem mailman_should_give_junk_mail : 
  junk_mail_per_house * houses_in_block = 640 :=
  by sorry

end mailman_should_give_junk_mail_l160_160249


namespace original_number_is_80_l160_160697

variable (e : ℝ)

def increased_value := 1.125 * e
def decreased_value := 0.75 * e
def difference_condition := increased_value e - decreased_value e = 30

theorem original_number_is_80 (h : difference_condition e) : e = 80 :=
sorry

end original_number_is_80_l160_160697


namespace trapezium_other_side_length_l160_160486

theorem trapezium_other_side_length (x : ℝ) : 
  (1 / 2) * (20 + x) * 13 = 247 → x = 18 :=
by
  sorry

end trapezium_other_side_length_l160_160486


namespace original_chairs_count_l160_160125

theorem original_chairs_count (n : ℕ) (m : ℕ) :
  (∀ k : ℕ, (k % 4 = 0 → k * (2 * n / 4) = k * (3 * n / 4) ) ∧ 
  (m = (4 / 2) * 15) ∧ (n = (4 * m / (2 * m)) - ((2 * m) / m)) ∧ 
  n + (n + 9) = 72) → n = 63 :=
by
  sorry

end original_chairs_count_l160_160125


namespace student_rank_from_right_l160_160147

theorem student_rank_from_right (n m : ℕ) (h1 : n = 8) (h2 : m = 20) : m - (n - 1) = 13 :=
by
  sorry

end student_rank_from_right_l160_160147


namespace find_A_l160_160450

def A : ℕ := 7 * 5 + 3

theorem find_A : A = 38 :=
by
  sorry

end find_A_l160_160450


namespace average_side_length_of_squares_l160_160046

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l160_160046


namespace value_of_g_of_h_at_2_l160_160186

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -5 * x^3 + 4

theorem value_of_g_of_h_at_2 : g (h 2) = 3890 := by
  sorry

end value_of_g_of_h_at_2_l160_160186


namespace incenter_divides_angle_bisector_2_1_l160_160477

def is_incenter_divide_angle_bisector (AB BC AC : ℝ) (O : ℝ) : Prop :=
  AB = 15 ∧ BC = 12 ∧ AC = 18 → O = 2 / 1

theorem incenter_divides_angle_bisector_2_1 :
  is_incenter_divide_angle_bisector 15 12 18 (2 / 1) :=
by
  sorry

end incenter_divides_angle_bisector_2_1_l160_160477


namespace expression_value_l160_160525

theorem expression_value (a b c d : ℤ) (h_a : a = 15) (h_b : b = 19) (h_c : c = 3) (h_d : d = 2) :
  (a - (b - c)) - ((a - b) - c + d) = 4 := 
by
  rw [h_a, h_b, h_c, h_d]
  sorry

end expression_value_l160_160525


namespace range_of_a_l160_160032

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3) / (5-a)) → -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l160_160032


namespace max_marked_points_l160_160312

theorem max_marked_points (segments : ℕ) (ratio : ℚ) (h_segments : segments = 10) (h_ratio : ratio = 3 / 4) : 
  ∃ n, n ≤ (segments * 2 / 2) ∧ n = 10 :=
by
  sorry

end max_marked_points_l160_160312


namespace square_TU_squared_l160_160577

theorem square_TU_squared (P Q R S T U : ℝ × ℝ)
  (side : ℝ) (RT SU PT QU : ℝ)
  (hpqrs : (P.1 - S.1)^2 + (P.2 - S.2)^2 = side^2 ∧ (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side^2 ∧ 
            (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = side^2 ∧ (S.1 - R.1)^2 + (S.2 - R.2)^2 = side^2)
  (hRT : (R.1 - T.1)^2 + (R.2 - T.2)^2 = RT^2)
  (hSU : (S.1 - U.1)^2 + (S.2 - U.2)^2 = SU^2)
  (hPT : (P.1 - T.1)^2 + (P.2 - T.2)^2 = PT^2)
  (hQU : (Q.1 - U.1)^2 + (Q.2 - U.2)^2 = QU^2)
  (side_eq_17 : side = 17) (RT_SU_eq_8 : RT = 8) (PT_QU_eq_15 : PT = 15) :
  (T.1 - U.1)^2 + (T.2 - U.2)^2 = 979.5 :=
by
  -- proof to be filled in
  sorry

end square_TU_squared_l160_160577


namespace profit_inequality_solution_l160_160668

theorem profit_inequality_solution (x : ℝ) (h₁ : 1 ≤ x) (h₂ : x ≤ 10) :
  100 * 2 * (5 * x + 1 - 3 / x) ≥ 3000 ↔ 3 ≤ x ∧ x ≤ 10 :=
by
  sorry

end profit_inequality_solution_l160_160668


namespace perfect_squares_divide_l160_160659

-- Define the problem and the conditions as Lean definitions
def numFactors (base exponent : ℕ) := (exponent / 2) + 1

def countPerfectSquareFactors : ℕ := 
  let choices2 := numFactors 2 3
  let choices3 := numFactors 3 5
  let choices5 := numFactors 5 7
  let choices7 := numFactors 7 9
  choices2 * choices3 * choices5 * choices7

theorem perfect_squares_divide (numFactors : (ℕ → ℕ → ℕ)) 
(countPerfectSquareFactors : ℕ) : countPerfectSquareFactors = 120 :=
by
  -- We skip the proof here
  -- Proof steps would go here if needed
  sorry

end perfect_squares_divide_l160_160659


namespace length_of_box_l160_160701

theorem length_of_box (rate : ℕ) (width : ℕ) (depth : ℕ) (time : ℕ) (volume : ℕ) (length : ℕ) :
  rate = 4 →
  width = 6 →
  depth = 2 →
  time = 21 →
  volume = rate * time →
  length = volume / (width * depth) →
  length = 7 :=
by
  intros
  sorry

end length_of_box_l160_160701


namespace income_is_10000_l160_160748

-- Define the necessary variables: income, expenditure, and savings
variables (income expenditure : ℕ) (x : ℕ)

-- Define the conditions given in the problem
def ratio_condition : Prop := income = 10 * x ∧ expenditure = 7 * x
def savings_condition : Prop := income - expenditure = 3000

-- State the theorem that needs to be proved
theorem income_is_10000 (h_ratio : ratio_condition income expenditure x) (h_savings : savings_condition income expenditure) : income = 10000 :=
sorry

end income_is_10000_l160_160748


namespace solve_x_eq_10000_l160_160480

theorem solve_x_eq_10000 (x : ℝ) (h : 5 * x^(1/4 : ℝ) - 3 * (x / x^(3/4 : ℝ)) = 10 + x^(1/4 : ℝ)) : x = 10000 :=
by
  sorry

end solve_x_eq_10000_l160_160480


namespace sin_beta_value_l160_160410

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 4 / 5) 
  (h4 : Real.cos (α + β) = 5 / 13) : 
  Real.sin β = 33 / 65 := 
by 
  sorry

end sin_beta_value_l160_160410


namespace average_number_of_fish_is_75_l160_160721

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end average_number_of_fish_is_75_l160_160721


namespace savings_of_person_l160_160738

-- Definitions as given in the problem
def income := 18000
def ratio_income_expenditure := 5 / 4

-- Implied definitions based on the conditions and problem context
noncomputable def expenditure := income * (4/5)
noncomputable def savings := income - expenditure

-- Theorem statement
theorem savings_of_person : savings = 3600 :=
by
  -- Placeholder for proof
  sorry

end savings_of_person_l160_160738


namespace temperature_difference_l160_160636

/-- The average temperature at the top of Mount Tai. -/
def T_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai. -/
def T_foot : ℝ := -1

/-- The temperature difference between the average temperature at the foot and the top of Mount Tai is 8 degrees Celsius. -/
theorem temperature_difference : T_foot - T_top = 8 := by
  sorry

end temperature_difference_l160_160636


namespace number_of_zeros_of_f_l160_160933

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

theorem number_of_zeros_of_f : ∃! x : ℝ, 0 < x ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l160_160933


namespace determine_colors_l160_160507

-- Define the colors
inductive Color
| white
| red
| blue

open Color

-- Define the friends
inductive Friend
| Tamara 
| Valya
| Lida

open Friend

-- Define a function from Friend to their dress color and shoes color
def Dress : Friend → Color := sorry
def Shoes : Friend → Color := sorry

-- The problem conditions
axiom cond1 : Dress Tamara = Shoes Tamara
axiom cond2 : Shoes Valya = white
axiom cond3 : Dress Lida ≠ red
axiom cond4 : Shoes Lida ≠ red

-- The proof goal
theorem determine_colors :
  Dress Tamara = red ∧ Shoes Tamara = red ∧
  Dress Valya = blue ∧ Shoes Valya = white ∧
  Dress Lida = white ∧ Shoes Lida = blue :=
sorry

end determine_colors_l160_160507


namespace sales_of_stationery_accessories_l160_160885

def percentage_of_sales_notebooks : ℝ := 25
def percentage_of_sales_markers : ℝ := 40
def total_sales_percentage : ℝ := 100

theorem sales_of_stationery_accessories : 
  percentage_of_sales_notebooks + percentage_of_sales_markers = 65 → 
  total_sales_percentage - (percentage_of_sales_notebooks + percentage_of_sales_markers) = 35 :=
by
  sorry

end sales_of_stationery_accessories_l160_160885


namespace Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l160_160453

def Q (x : ℂ) (n : ℕ) : ℂ := (x + 1)^n + x^n + 1
def P (x : ℂ) : ℂ := x^2 + x + 1

-- Part a) Q(x) is divisible by P(x) if and only if n ≡ 2 (mod 6) or n ≡ 4 (mod 6)
theorem Q_divisible_by_P (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 2 ∨ n % 6 = 4) := sorry

-- Part b) Q(x) is divisible by P(x)^2 if and only if n ≡ 4 (mod 6)
theorem Q_divisible_by_P_squared (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 4 := sorry

-- Part c) Q(x) is never divisible by P(x)^3
theorem Q_not_divisible_by_P_cubed (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^3 ≠ 0 := sorry

end Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l160_160453


namespace alberto_spent_more_l160_160072

noncomputable def alberto_total_before_discount : ℝ := 2457 + 374 + 520
noncomputable def alberto_discount : ℝ := 0.05 * alberto_total_before_discount
noncomputable def alberto_total_after_discount : ℝ := alberto_total_before_discount - alberto_discount

noncomputable def samara_total_before_tax : ℝ := 25 + 467 + 79 + 150
noncomputable def samara_tax : ℝ := 0.07 * samara_total_before_tax
noncomputable def samara_total_after_tax : ℝ := samara_total_before_tax + samara_tax

noncomputable def amount_difference : ℝ := alberto_total_after_discount - samara_total_after_tax

theorem alberto_spent_more : amount_difference = 2411.98 :=
by
  sorry

end alberto_spent_more_l160_160072


namespace count_distinct_ways_l160_160873

theorem count_distinct_ways (p : ℕ × ℕ → ℕ) (h_condition : ∃ j : ℕ × ℕ, j ∈ [(0, 0), (0, 1)] ∧ p j = 4)
  (h_grid_size : ∀ i : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → 1 ≤ p i ∧ p i ≤ 4)
  (h_distinct : ∀ i j : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → j ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → i ≠ j → p i ≠ p j) :
  ∃! l : Finset (ℕ × ℕ → ℕ), l.card = 12 :=
by
  sorry

end count_distinct_ways_l160_160873


namespace larger_number_is_1617_l160_160407

-- Given conditions
variables (L S : ℤ)
axiom condition1 : L - S = 1515
axiom condition2 : L = 16 * S + 15

-- To prove
theorem larger_number_is_1617 : L = 1617 := by
  sorry

end larger_number_is_1617_l160_160407


namespace miranda_heels_cost_l160_160076

theorem miranda_heels_cost (months_saved : ℕ) (savings_per_month : ℕ) (gift_from_sister : ℕ) 
  (h1 : months_saved = 3) (h2 : savings_per_month = 70) (h3 : gift_from_sister = 50) : 
  months_saved * savings_per_month + gift_from_sister = 260 := 
by
  sorry

end miranda_heels_cost_l160_160076


namespace negation_of_P_l160_160528

open Real

theorem negation_of_P :
  (¬ (∀ x : ℝ, x > sin x)) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end negation_of_P_l160_160528


namespace find_middle_part_length_l160_160169

theorem find_middle_part_length (a b c : ℝ) 
  (h1 : a + b + c = 28) 
  (h2 : (a - 0.5 * a) + b + 0.5 * c = 16) :
  b = 4 :=
by
  sorry

end find_middle_part_length_l160_160169


namespace max_even_integers_with_odd_product_l160_160698

theorem max_even_integers_with_odd_product (a b c d e f : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) (h_odd_product : (a * b * c * d * e * f) % 2 = 1) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) := 
sorry

end max_even_integers_with_odd_product_l160_160698


namespace lily_coffee_budget_l160_160929

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end lily_coffee_budget_l160_160929


namespace inequality_system_solution_l160_160138

theorem inequality_system_solution (x : ℝ) : x + 1 > 0 → x - 3 > 0 → x > 3 :=
by
  intros h1 h2
  sorry

end inequality_system_solution_l160_160138


namespace problem_l160_160368

open Real

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : (1 / a) + (4 / b) + (9 / c) ≤ 36 / (a + b + c)) 
  : (2 * b + 3 * c) / (a + b + c) = 13 / 6 :=
sorry

end problem_l160_160368


namespace exists_multiple_of_power_of_2_with_non_zero_digits_l160_160145

theorem exists_multiple_of_power_of_2_with_non_zero_digits (n : ℕ) (hn : n ≥ 1) :
  ∃ a : ℕ, (∀ d ∈ a.digits 10, d = 1 ∨ d = 2) ∧ 2^n ∣ a :=
by
  sorry

end exists_multiple_of_power_of_2_with_non_zero_digits_l160_160145


namespace fraction_of_married_women_l160_160742

theorem fraction_of_married_women (total_employees : ℕ) 
  (women_fraction : ℝ) (married_fraction : ℝ) (single_men_fraction : ℝ)
  (hwf : women_fraction = 0.64) (hmf : married_fraction = 0.60) 
  (hsf : single_men_fraction = 2/3) : 
  ∃ (married_women_fraction : ℝ), married_women_fraction = 3/4 := 
by
  sorry

end fraction_of_married_women_l160_160742


namespace initial_walnuts_l160_160173

theorem initial_walnuts (W : ℕ) (boy_effective : ℕ) (girl_effective : ℕ) (total_walnuts : ℕ) :
  boy_effective = 5 → girl_effective = 3 → total_walnuts = 20 → W + boy_effective + girl_effective = total_walnuts → W = 12 :=
by
  intros h_boy h_girl h_total h_eq
  rw [h_boy, h_girl, h_total] at h_eq
  linarith

end initial_walnuts_l160_160173


namespace barbeck_steve_guitar_ratio_l160_160021

theorem barbeck_steve_guitar_ratio (b s d : ℕ) 
  (h1 : b = s) 
  (h2 : d = 3 * b) 
  (h3 : b + s + d = 27) 
  (h4 : d = 18) : 
  b / s = 2 / 1 := 
by 
  sorry

end barbeck_steve_guitar_ratio_l160_160021


namespace track_width_l160_160303

theorem track_width (r : ℝ) (h1 : 4 * π * r - 2 * π * r = 16 * π) (h2 : 2 * r = r + r) : 2 * r - r = 8 :=
by
  sorry

end track_width_l160_160303


namespace smallest_positive_natural_number_l160_160304

theorem smallest_positive_natural_number (a b c d e : ℕ) 
    (h1 : a = 3) (h2 : b = 5) (h3 : c = 6) (h4 : d = 18) (h5 : e = 23) :
    ∃ (x y : ℕ), x = (e - a) / b - d / c ∨ x = e - d + b - c - a ∧ x = 1 := by
  sorry

end smallest_positive_natural_number_l160_160304


namespace triangle_weight_l160_160371

variables (S C T : ℕ)

def scale1 := (S + C = 8)
def scale2 := (S + 2 * C = 11)
def scale3 := (C + 2 * T = 15)

theorem triangle_weight (h1 : scale1 S C) (h2 : scale2 S C) (h3 : scale3 C T) : T = 6 :=
by 
  sorry

end triangle_weight_l160_160371


namespace james_marbles_left_l160_160568

theorem james_marbles_left :
  ∀ (initial_marbles bags remaining_bags marbles_per_bag left_marbles : ℕ),
  initial_marbles = 28 →
  bags = 4 →
  marbles_per_bag = initial_marbles / bags →
  remaining_bags = bags - 1 →
  left_marbles = remaining_bags * marbles_per_bag →
  left_marbles = 21 :=
by
  intros initial_marbles bags remaining_bags marbles_per_bag left_marbles
  sorry

end james_marbles_left_l160_160568


namespace system_of_equations_solution_l160_160876

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 x4 x5 : ℝ),
  (x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1) ∧
  (x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2) ∧
  (x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) ∧
  (x1 = 1) ∧ (x2 = -1) ∧ (x3 = 1) ∧ (x4 = -1) ∧ (x5 = 1) := by
sorry

end system_of_equations_solution_l160_160876


namespace fixed_point_coordinates_l160_160753

theorem fixed_point_coordinates (a b x y : ℝ) 
  (h1 : a + 2 * b = 1) 
  (h2 : (a * x + 3 * y + b) = 0) :
  x = 1 / 2 ∧ y = -1 / 6 := by
  sorry

end fixed_point_coordinates_l160_160753


namespace range_of_k_l160_160499

/-- If the function y = (k + 1) * x is decreasing on the entire real line, then k < -1. -/
theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x < y → (k + 1) * x > (k + 1) * y) : k < -1 :=
sorry

end range_of_k_l160_160499


namespace smallest_number_is_111111_2_l160_160172

def base9_to_decimal (n : Nat) : Nat :=
  (n / 10) * 9 + (n % 10)

def base6_to_decimal (n : Nat) : Nat :=
  (n / 100) * 36 + ((n % 100) / 10) * 6 + (n % 10)

def base4_to_decimal (n : Nat) : Nat :=
  (n / 1000) * 64

def base2_to_decimal (n : Nat) : Nat :=
  (n / 100000) * 32 + ((n % 100000) / 10000) * 16 + ((n % 10000) / 1000) * 8 + ((n % 1000) / 100) * 4 + ((n % 100) / 10) * 2 + (n % 10)

theorem smallest_number_is_111111_2 :
  let n1 := base9_to_decimal 85
  let n2 := base6_to_decimal 210
  let n3 := base4_to_decimal 1000
  let n4 := base2_to_decimal 111111
  n4 < n1 ∧ n4 < n2 ∧ n4 < n3 := by
    sorry

end smallest_number_is_111111_2_l160_160172


namespace evaluate_expression_l160_160647

theorem evaluate_expression : (532 * 532) - (531 * 533) = 1 := by
  sorry

end evaluate_expression_l160_160647


namespace circle_equation_equivalence_l160_160937

theorem circle_equation_equivalence 
    (x y : ℝ) : 
    x^2 + y^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 :=
sorry

end circle_equation_equivalence_l160_160937


namespace baylor_final_amount_l160_160600

def CDA := 4000
def FCP := (1 / 2) * CDA
def SCP := FCP + (2 / 5) * FCP
def TCP := 2 * (FCP + SCP)
def FDA := CDA + FCP + SCP + TCP

theorem baylor_final_amount : FDA = 18400 := by
  sorry

end baylor_final_amount_l160_160600


namespace eq_fraction_l160_160592

def f(x : ℤ) : ℤ := 3 * x + 4
def g(x : ℤ) : ℤ := 2 * x - 1

theorem eq_fraction : (f (g (f 3))) / (g (f (g 3))) = 79 / 37 := by
  sorry

end eq_fraction_l160_160592


namespace students_more_than_rabbits_by_64_l160_160293

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end students_more_than_rabbits_by_64_l160_160293


namespace find_f_2011_l160_160238

-- Definitions of given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_of_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- Main theorem to be proven
theorem find_f_2011 (f : ℝ → ℝ) 
  (hf_even: is_even_function f) 
  (hf_periodic: is_periodic_of_period f 4) 
  (hf_at_1: f 1 = 1) : 
  f 2011 = 1 := 
by 
  sorry

end find_f_2011_l160_160238


namespace blue_balls_balance_l160_160012

variables {R B O P : ℝ}

-- Given conditions
def cond1 : 4 * R = 8 * B := sorry
def cond2 : 3 * O = 7 * B := sorry
def cond3 : 8 * B = 6 * P := sorry

-- Proof problem: proving equal balance of 5 red balls, 3 orange balls, and 4 purple balls
theorem blue_balls_balance : 5 * R + 3 * O + 4 * P = (67 / 3) * B :=
by
  sorry

end blue_balls_balance_l160_160012


namespace profit_percentage_correct_l160_160749

noncomputable def overall_profit_percentage : ℚ :=
  let cost_radio := 225
  let overhead_radio := 15
  let price_radio := 300
  let cost_watch := 425
  let overhead_watch := 20
  let price_watch := 525
  let cost_mobile := 650
  let overhead_mobile := 30
  let price_mobile := 800
  
  let total_cost_price := (cost_radio + overhead_radio) + (cost_watch + overhead_watch) + (cost_mobile + overhead_mobile)
  let total_selling_price := price_radio + price_watch + price_mobile
  let total_profit := total_selling_price - total_cost_price
  (total_profit * 100 : ℚ) / total_cost_price
  
theorem profit_percentage_correct :
  overall_profit_percentage = 19.05 := by
  sorry

end profit_percentage_correct_l160_160749


namespace ironman_age_greater_than_16_l160_160868

variable (Ironman_age : ℕ)
variable (Thor_age : ℕ := 1456)
variable (CaptainAmerica_age : ℕ := Thor_age / 13)
variable (PeterParker_age : ℕ := CaptainAmerica_age / 7)

theorem ironman_age_greater_than_16
  (Thor_13_times_CaptainAmerica : Thor_age = 13 * CaptainAmerica_age)
  (CaptainAmerica_7_times_PeterParker : CaptainAmerica_age = 7 * PeterParker_age)
  (Thor_age_given : Thor_age = 1456) :
  Ironman_age > 16 :=
by
  sorry

end ironman_age_greater_than_16_l160_160868


namespace bumper_car_rides_correct_l160_160128

def tickets_per_ride : ℕ := 7
def total_tickets : ℕ := 63
def ferris_wheel_rides : ℕ := 5

def tickets_for_bumper_cars : ℕ :=
  total_tickets - ferris_wheel_rides * tickets_per_ride

def bumper_car_rides : ℕ :=
  tickets_for_bumper_cars / tickets_per_ride

theorem bumper_car_rides_correct : bumper_car_rides = 4 :=
by
  sorry

end bumper_car_rides_correct_l160_160128


namespace moles_of_naoh_needed_l160_160291

-- Define the chemical reaction
def balanced_eqn (nh4no3 naoh nano3 nh4oh : ℕ) : Prop :=
  nh4no3 = naoh ∧ nh4no3 = nano3

-- Theorem stating the moles of NaOH required to form 2 moles of NaNO3 from 2 moles of NH4NO3
theorem moles_of_naoh_needed (nh4no3 naoh nano3 nh4oh : ℕ) (h_balanced_eqn : balanced_eqn nh4no3 naoh nano3 nh4oh) 
  (h_nano3: nano3 = 2) (h_nh4no3: nh4no3 = 2) : naoh = 2 :=
by
  unfold balanced_eqn at h_balanced_eqn
  sorry

end moles_of_naoh_needed_l160_160291


namespace find_radius_of_sphere_l160_160643

def radius_of_sphere (width : ℝ) (depth : ℝ) (r : ℝ) : Prop :=
  (width / 2) ^ 2 + (r - depth) ^ 2 = r ^ 2

theorem find_radius_of_sphere (r : ℝ) : radius_of_sphere 30 10 r → r = 16.25 :=
by
  intros h1
  -- sorry is a placeholder for the actual proof
  sorry

end find_radius_of_sphere_l160_160643


namespace total_amount_divided_l160_160523

theorem total_amount_divided (A B C : ℝ) (h1 : A / B = 3 / 4) (h2 : B / C = 5 / 6) (h3 : A = 29491.525423728814) :
  A + B + C = 116000 := 
sorry

end total_amount_divided_l160_160523


namespace solve_for_x_l160_160839

theorem solve_for_x : ∃ x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := 
by
  use -1005
  sorry

end solve_for_x_l160_160839


namespace original_ratio_of_flour_to_baking_soda_l160_160436

-- Define the conditions
def sugar_to_flour_ratio_5_to_5 (sugar flour : ℕ) : Prop :=
  sugar = 2400 ∧ sugar = flour

def baking_soda_mass_condition (flour : ℕ) (baking_soda : ℕ) : Prop :=
  flour = 2400 ∧ (∃ b : ℕ, baking_soda = b ∧ flour / (b + 60) = 8)

-- The theorem statement we need to prove
theorem original_ratio_of_flour_to_baking_soda :
  ∃ flour baking_soda : ℕ,
  sugar_to_flour_ratio_5_to_5 2400 flour ∧
  baking_soda_mass_condition flour baking_soda →
  flour / baking_soda = 10 :=
by
  sorry

end original_ratio_of_flour_to_baking_soda_l160_160436


namespace sum_of_coefficients_l160_160802

theorem sum_of_coefficients 
  (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ)
  (h : (3 * x - 1) ^ 10 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9 + a_10 * x ^ 10) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1023 := 
sorry

end sum_of_coefficients_l160_160802


namespace carwash_num_cars_l160_160733

variable (C : ℕ)

theorem carwash_num_cars 
    (h1 : 5 * 7 + 5 * 6 + C * 5 = 100)
    : C = 7 := 
by
    sorry

end carwash_num_cars_l160_160733


namespace sequence_geometric_and_general_term_sum_of_sequence_l160_160615

theorem sequence_geometric_and_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, S k = 2 * a k - k) : 
  (a 0 = 1) ∧ 
  (∀ k : ℕ, a (k + 1) = 2 * a k + 1) ∧ 
  (∀ k : ℕ, a k = 2^k - 1) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, a k = 2^k - 1)
  (h2 : ∀ k : ℕ, b k = 1 / a (k+1) + 1 / (a k * a (k+1))) :
  T n = 1 - 1 / (2^(n+1) - 1) :=
sorry

end sequence_geometric_and_general_term_sum_of_sequence_l160_160615


namespace solve_inequality_l160_160498

theorem solve_inequality (a x : ℝ) :
  (a > 0 → (a - 1) / a < x ∧ x < 1) ∧ 
  (a = 0 → x < 1) ∧ 
  (a < 0 → x > (a - 1) / a ∨ x < 1) ↔ 
  (ax / (x - 1) < (a - 1) / (x - 1)) :=
sorry

end solve_inequality_l160_160498


namespace find_unique_pair_l160_160756

theorem find_unique_pair (x y : ℝ) :
  (∀ (u v : ℝ), (u * x + v * y = u) ∧ (u * y + v * x = v)) ↔ (x = 1 ∧ y = 0) :=
by
  -- This is to ignore the proof part
  sorry

end find_unique_pair_l160_160756


namespace twelfth_term_geometric_sequence_l160_160401

-- Define the first term and common ratio
def a1 : Int := 5
def r : Int := -3

-- Define the formula for the nth term of the geometric sequence
def nth_term (n : Nat) : Int := a1 * r^(n-1)

-- The statement to be proved: that the twelfth term is -885735
theorem twelfth_term_geometric_sequence : nth_term 12 = -885735 := by
  sorry

end twelfth_term_geometric_sequence_l160_160401


namespace child_ticket_price_correct_l160_160715

-- Definitions based on conditions
def total_collected := 104
def price_adult := 6
def total_tickets := 21
def children_tickets := 11

-- Derived conditions
def adult_tickets := total_tickets - children_tickets
def total_revenue_child (C : ℕ) := children_tickets * C
def total_revenue_adult := adult_tickets * price_adult

-- Main statement to prove
theorem child_ticket_price_correct (C : ℕ) 
  (h1 : total_revenue_child C + total_revenue_adult = total_collected) : 
  C = 4 :=
by
  sorry

end child_ticket_price_correct_l160_160715


namespace hedge_cost_and_blocks_l160_160334

-- Define the costs of each type of block
def costA : Nat := 2
def costB : Nat := 3
def costC : Nat := 4

-- Define the number of each type of block per section
def blocksPerSectionA : Nat := 20
def blocksPerSectionB : Nat := 10
def blocksPerSectionC : Nat := 5

-- Define the number of sections
def sections : Nat := 8

-- Define the total cost calculation
def totalCost : Nat := sections * (blocksPerSectionA * costA + blocksPerSectionB * costB + blocksPerSectionC * costC)

-- Define the total number of each type of block used
def totalBlocksA : Nat := sections * blocksPerSectionA
def totalBlocksB : Nat := sections * blocksPerSectionB
def totalBlocksC : Nat := sections * blocksPerSectionC

-- State the theorem
theorem hedge_cost_and_blocks :
  totalCost = 720 ∧ totalBlocksA = 160 ∧ totalBlocksB = 80 ∧ totalBlocksC = 40 := by
  sorry

end hedge_cost_and_blocks_l160_160334


namespace final_mark_is_correct_l160_160544

def term_mark : ℝ := 80
def term_weight : ℝ := 0.70
def exam_mark : ℝ := 90
def exam_weight : ℝ := 0.30

theorem final_mark_is_correct :
  (term_mark * term_weight + exam_mark * exam_weight) = 83 :=
by
  sorry

end final_mark_is_correct_l160_160544


namespace min_value_f_l160_160201

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem min_value_f (a : ℝ) (h : -2 < a) :
  ∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ≥ m) ∧ 
  ((a ≤ 1 → m = a^2 - 2 * a) ∧ (1 < a → m = -1)) :=
by
  sorry

end min_value_f_l160_160201


namespace minimum_value_of_f_l160_160302

def f (x : ℝ) : ℝ := 5 * x^2 - 20 * x + 1357

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) := 
by 
  use 1337
  sorry

end minimum_value_of_f_l160_160302


namespace sum_of_circle_areas_l160_160082

theorem sum_of_circle_areas (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) : 
  π * r^2 + π * s^2 + π * t^2 = 56 * π := 
by 
  sorry

end sum_of_circle_areas_l160_160082


namespace num_solutions_of_system_eq_two_l160_160070

theorem num_solutions_of_system_eq_two : 
  (∃ n : ℕ, n = 2 ∧ ∀ (x y : ℝ), 
    5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16 ↔ 
    (x, y) = ((-90 + Real.sqrt 31900) / 68, 3 * ((-90 + Real.sqrt 31900) / 68) / 5 + 3) ∨ 
    (x, y) = ((-90 - Real.sqrt 31900) / 68, 3 * ((-90 - Real.sqrt 31900) / 68) / 5 + 3)) :=
sorry

end num_solutions_of_system_eq_two_l160_160070


namespace volume_of_wedge_l160_160711

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l160_160711


namespace complex_number_conditions_l160_160556

open Complex Real

noncomputable def is_real (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 = 0

noncomputable def is_imaginary (a : ℝ) : Prop :=
a ^ 2 - 2 * a - 15 ≠ 0

noncomputable def is_purely_imaginary (a : ℝ) : Prop :=
a ^ 2 - 9 = 0 ∧ a ^ 2 - 2 * a - 15 ≠ 0

theorem complex_number_conditions (a : ℝ) :
  (is_real a ↔ (a = 5 ∨ a = -3))
  ∧ (is_imaginary a ↔ (a ≠ 5 ∧ a ≠ -3))
  ∧ (¬(∃ a : ℝ, is_purely_imaginary a)) :=
by
  sorry

end complex_number_conditions_l160_160556


namespace point_on_x_axis_l160_160926

theorem point_on_x_axis (m : ℝ) (h : m - 2 = 0) :
  (m + 3, m - 2) = (5, 0) :=
by
  sorry

end point_on_x_axis_l160_160926


namespace equivalent_single_discount_l160_160718

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.30
noncomputable def second_discount : ℝ := 0.15
noncomputable def third_discount : ℝ := 0.10

theorem equivalent_single_discount :
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  (1 - (final_price / original_price)) = 0.4645 :=
by
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  sorry

end equivalent_single_discount_l160_160718


namespace john_safe_weight_l160_160232

-- Assuming the conditions provided that form the basis of our problem.
def max_capacity : ℝ := 1000
def safety_margin : ℝ := 0.20
def john_weight : ℝ := 250
def safe_weight (max_capacity safety_margin john_weight : ℝ) : ℝ := 
  (max_capacity * (1 - safety_margin)) - john_weight

-- The main theorem to prove based on the provided problem statement.
theorem john_safe_weight : safe_weight max_capacity safety_margin john_weight = 550 := by
  -- skipping the proof details as instructed
  sorry

end john_safe_weight_l160_160232


namespace pies_not_eaten_with_forks_l160_160541

variables (apple_pe_forked peach_pe_forked cherry_pe_forked chocolate_pe_forked lemon_pe_forked : ℤ)
variables (total_pies types_of_pies : ℤ)

def pies_per_type (total_pies types_of_pies : ℤ) : ℤ :=
  total_pies / types_of_pies

def not_eaten_with_forks (percentage_forked : ℤ) (pies : ℤ) : ℤ :=
  pies - (pies * percentage_forked) / 100

noncomputable def apple_not_forked  := not_eaten_with_forks apple_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def peach_not_forked  := not_eaten_with_forks peach_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def cherry_not_forked := not_eaten_with_forks cherry_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def chocolate_not_forked := not_eaten_with_forks chocolate_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def lemon_not_forked := not_eaten_with_forks lemon_pe_forked (pies_per_type total_pies types_of_pies)

theorem pies_not_eaten_with_forks :
  (apple_not_forked = 128) ∧
  (peach_not_forked = 112) ∧
  (cherry_not_forked = 84) ∧
  (chocolate_not_forked = 76) ∧
  (lemon_not_forked = 140) :=
by sorry

end pies_not_eaten_with_forks_l160_160541


namespace improper_fraction_decomposition_l160_160019

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := 
sorry

end improper_fraction_decomposition_l160_160019


namespace no_solution_in_A_l160_160223

def A : Set ℕ := 
  {n | ∃ k : ℤ, abs (n * Real.sqrt 2022 - 1 / 3 - k) ≤ 1 / 2022}

theorem no_solution_in_A (x y z : ℕ) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A) : 
  20 * x + 21 * y ≠ 22 * z := 
sorry

end no_solution_in_A_l160_160223


namespace muffin_cost_is_correct_l160_160123

variable (M : ℝ)

def total_original_cost (muffin_cost : ℝ) : ℝ := 3 * muffin_cost + 1.45

def discounted_cost (original_cost : ℝ) : ℝ := 0.85 * original_cost

def kevin_paid (discounted_price : ℝ) : Prop := discounted_price = 3.70

theorem muffin_cost_is_correct (h : discounted_cost (total_original_cost M) = 3.70) : M = 0.97 :=
  by
  sorry

end muffin_cost_is_correct_l160_160123


namespace fraction_of_walls_not_illuminated_l160_160821

-- Define given conditions
def point_light_source : Prop := true
def rectangular_room : Prop := true
def flat_mirror_on_wall : Prop := true
def full_height_of_room : Prop := true

-- Define the fraction not illuminated
def fraction_not_illuminated := 17 / 32

-- State the theorem to prove
theorem fraction_of_walls_not_illuminated :
  point_light_source ∧ rectangular_room ∧ flat_mirror_on_wall ∧ full_height_of_room →
  fraction_not_illuminated = 17 / 32 :=
by
  intros h
  sorry

end fraction_of_walls_not_illuminated_l160_160821


namespace symmetric_curve_equation_l160_160573

theorem symmetric_curve_equation (y x : ℝ) :
  (y^2 = 4 * x) → (y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_equation_l160_160573


namespace daniel_earnings_l160_160776

def fabric_monday := 20
def yarn_monday := 15

def fabric_tuesday := 2 * fabric_monday
def yarn_tuesday := yarn_monday + 10

def fabric_wednesday := fabric_tuesday / 4
def yarn_wednesday := yarn_tuesday / 2

def price_per_yard_fabric := 2
def price_per_yard_yarn := 3

def total_fabric := fabric_monday + fabric_tuesday + fabric_wednesday
def total_yarn := yarn_monday + yarn_tuesday + yarn_wednesday

def earnings_fabric := total_fabric * price_per_yard_fabric
def earnings_yarn := total_yarn * price_per_yard_yarn

def total_earnings := earnings_fabric + earnings_yarn

theorem daniel_earnings :
  total_earnings = 299 := by
  sorry

end daniel_earnings_l160_160776


namespace bride_groom_couples_sum_l160_160454

def wedding_reception (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) : Prop :=
  total_guests - friends = couples_guests

theorem bride_groom_couples_sum (B G : ℕ) (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) 
  (h1 : total_guests = 180) (h2 : friends = 100) (h3 : wedding_reception total_guests friends couples_guests) 
  (h4 : couples_guests = 80) : B + G = 40 := 
  by
  sorry

end bride_groom_couples_sum_l160_160454


namespace correct_sum_of_integers_l160_160429

theorem correct_sum_of_integers (a b : ℕ) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := 
  sorry

end correct_sum_of_integers_l160_160429


namespace time_in_future_is_4_l160_160421

def current_time := 5
def future_hours := 1007
def modulo := 12
def future_time := (current_time + future_hours) % modulo

theorem time_in_future_is_4 : future_time = 4 := by
  sorry

end time_in_future_is_4_l160_160421


namespace calculate_ab_l160_160949

theorem calculate_ab {a b c : ℝ} (hc : c ≠ 0) (h1 : (a * b) / c = 4) (h2 : a * (b / c) = 12) : a * b = 12 :=
by
  sorry

end calculate_ab_l160_160949


namespace necessary_but_not_sufficient_l160_160768

def mutually_exclusive (A1 A2 : Prop) : Prop := (A1 ∧ A2) → False
def complementary (A1 A2 : Prop) : Prop := (A1 ∨ A2) ∧ ¬(A1 ∧ A2)

theorem necessary_but_not_sufficient {A1 A2 : Prop}: 
  mutually_exclusive A1 A2 → complementary A1 A2 → (¬(mutually_exclusive A1 A2 → complementary A1 A2) ∧ (complementary A1 A2 → mutually_exclusive A1 A2)) := 
  by
    sorry

end necessary_but_not_sufficient_l160_160768


namespace train_crosses_bridge_in_12_4_seconds_l160_160079

noncomputable def train_crossing_bridge_time (length_train : ℝ) (speed_train_kmph : ℝ) (length_bridge : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (1000 / 3600)
  let total_distance := length_train + length_bridge
  total_distance / speed_train_mps

theorem train_crosses_bridge_in_12_4_seconds :
  train_crossing_bridge_time 110 72 138 = 12.4 :=
by
  sorry

end train_crosses_bridge_in_12_4_seconds_l160_160079


namespace max_months_to_build_l160_160620

theorem max_months_to_build (a b c x : ℝ) (h1 : 1/a + 1/b = 1/6)
                            (h2 : 1/a + 1/c = 1/5)
                            (h3 : 1/c + 1/b = 1/4)
                            (h4 : (1/a + 1/b + 1/c) * x = 1) :
                            x = 4 :=
sorry

end max_months_to_build_l160_160620


namespace exactly_one_correct_proposition_l160_160709

variables (l1 l2 : Line) (alpha : Plane)

-- Definitions for the conditions
def perpendicular_lines (l1 l2 : Line) : Prop := -- definition of perpendicular lines
sorry

def perpendicular_to_plane (l : Line) (alpha : Plane) : Prop := -- definition of line perpendicular to plane
sorry

def line_in_plane (l : Line) (alpha : Plane) : Prop := -- definition of line in a plane
sorry

-- Problem statement
theorem exactly_one_correct_proposition 
  (h1 : perpendicular_lines l1 l2) 
  (h2 : perpendicular_to_plane l1 alpha) 
  (h3 : line_in_plane l2 alpha) : 
  (¬(perpendicular_lines l1 l2 ∧ perpendicular_to_plane l1 alpha → line_in_plane l2 alpha) ∧
   ¬(perpendicular_lines l1 l2 ∧ line_in_plane l2 alpha → perpendicular_to_plane l1 alpha) ∧
   (perpendicular_to_plane l1 alpha ∧ line_in_plane l2 alpha → perpendicular_lines l1 l2)) :=
sorry

end exactly_one_correct_proposition_l160_160709


namespace avg_eggs_per_nest_l160_160982

/-- In the Caribbean, loggerhead turtles lay three million eggs in twenty thousand nests. 
On average, show that there are 150 eggs in each nest. -/

theorem avg_eggs_per_nest 
  (total_eggs : ℕ) 
  (total_nests : ℕ) 
  (h1 : total_eggs = 3000000) 
  (h2 : total_nests = 20000) :
  total_eggs / total_nests = 150 := 
by {
  sorry
}

end avg_eggs_per_nest_l160_160982


namespace parabola_equation_l160_160060

-- Define the conditions of the problem
def parabola_vertex := (0, 0)
def parabola_focus_x_axis := true
def line_eq (x y : ℝ) : Prop := x = y
def midpoint_of_AB (x1 y1 x2 y2 mx my: ℝ) : Prop := (mx, my) = ((x1 + x2) / 2, (y1 + y2) / 2)
def point_P := (1, 1)

theorem parabola_equation (A B : ℝ × ℝ) :
  (parabola_vertex = (0, 0)) →
  (parabola_focus_x_axis) →
  (line_eq A.1 A.2) →
  (line_eq B.1 B.2) →
  midpoint_of_AB A.1 A.2 B.1 B.2 point_P.1 point_P.2 →
  A = (0, 0) ∨ B = (0, 0) →
  B = A ∨ A = (0, 0) → B = (2, 2) →
  ∃ a, ∀ x y, y^2 = a * x → a = 2 :=
sorry

end parabola_equation_l160_160060


namespace no_nat_exists_perfect_cubes_l160_160358

theorem no_nat_exists_perfect_cubes : ¬ ∃ n : ℕ, ∃ a b : ℤ, 2^(n + 1) - 1 = a^3 ∧ 2^(n - 1)*(2^n - 1) = b^3 := 
by
  sorry

end no_nat_exists_perfect_cubes_l160_160358


namespace rectangle_area_l160_160218

theorem rectangle_area (x : ℝ) (l : ℝ) (h1 : 3 * l = x^2 / 10) : 
  3 * l^2 = 3 * x^2 / 10 :=
by sorry

end rectangle_area_l160_160218


namespace sum_n_terms_max_sum_n_l160_160915

variable {a : ℕ → ℚ} (S : ℕ → ℚ)
variable (d a_1 : ℚ)

-- Conditions given in the problem
axiom sum_first_10 : S 10 = 125 / 7
axiom sum_first_20 : S 20 = -250 / 7
axiom sum_arithmetic_seq : ∀ n, S n = n * (a 1 + a n) / 2

-- Define the first term and common difference for the arithmetic sequence
axiom common_difference : ∀ n, a n = a_1 + (n - 1) * d

-- Theorem 1: Sum of the first n terms
theorem sum_n_terms (n : ℕ) : S n = (75 * n - 5 * n^2) / 14 := 
  sorry

-- Theorem 2: Value of n that maximizes S_n
theorem max_sum_n : n = 7 ∨ n = 8 ↔ (∀ m, S m ≤ S 7 ∨ S m ≤ S 8) := 
  sorry

end sum_n_terms_max_sum_n_l160_160915


namespace hexagon_perimeter_l160_160257

-- Define the length of a side of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def perimeter (num_sides side_length : ℕ) : ℕ :=
  num_sides * side_length

-- Theorem stating the perimeter of the hexagon with given side length is 42 inches
theorem hexagon_perimeter : perimeter num_sides side_length = 42 := by
  sorry

end hexagon_perimeter_l160_160257


namespace tensor_A_B_eq_l160_160737

-- Define sets A and B
def A : Set ℕ := {0, 2}
def B : Set ℕ := {x | x^2 - 3 * x + 2 = 0}

-- Define set operation ⊗
def tensor (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

-- Prove that A ⊗ B = {0, 2, 4}
theorem tensor_A_B_eq : tensor A B = {0, 2, 4} :=
by
  sorry

end tensor_A_B_eq_l160_160737


namespace min_dist_AB_l160_160639

-- Definitions of the conditions
structure Point3D where
  x : Float
  y : Float
  z : Float

def O := Point3D.mk 0 0 0
def B := Point3D.mk (Float.sqrt 3) (Float.sqrt 2) 2

def dist (P Q : Point3D) : Float :=
  Float.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points
variables (A : Point3D)
axiom AO_eq_1 : dist A O = 1

-- Minimum value of |AB|
theorem min_dist_AB : dist A B ≥ 2 := 
sorry

end min_dist_AB_l160_160639


namespace xy_not_z_probability_l160_160112

theorem xy_not_z_probability :
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8
  let P := P_X * P_Y * P_not_Z
  P = (1 : ℝ) / 32 :=
by
  -- Definitions based on problem conditions
  let P_X := (1 : ℝ) / 4
  let P_Y := (1 : ℝ) / 3
  let P_not_Z := (3 : ℝ) / 8

  -- Calculate the combined probability
  let P := P_X * P_Y * P_not_Z
  
  -- Check equality with 1/32
  have h : P = (1 : ℝ) / 32 := by sorry
  exact h

end xy_not_z_probability_l160_160112


namespace petya_payment_l160_160554

theorem petya_payment (x y : ℤ) (h₁ : 14 * x + 3 * y = 107) (h₂ : |x - y| ≤ 5) : x + y = 10 :=
sorry

end petya_payment_l160_160554


namespace geometric_sum_eight_terms_l160_160920

theorem geometric_sum_eight_terms (a_1 : ℕ) (S_4 : ℕ) (r : ℕ) (S_8 : ℕ) 
    (h1 : r = 2) (h2 : S_4 = a_1 * (1 + r + r^2 + r^3)) (h3 : S_4 = 30) :
    S_8 = a_1 * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) → S_8 = 510 := 
by sorry

end geometric_sum_eight_terms_l160_160920


namespace find_b_l160_160264

theorem find_b (a b c : ℤ) (h1 : a + b + c = 120) (h2 : a + 4 = b - 12) (h3 : a + 4 = 3 * c) : b = 60 :=
sorry

end find_b_l160_160264


namespace average_rainfall_correct_l160_160655

-- Define the leap year condition and days in February
def leap_year_february_days : ℕ := 29

-- Define total hours in a day
def hours_in_day : ℕ := 24

-- Define total rainfall in February 2012 in inches
def total_rainfall : ℕ := 420

-- Define total hours in February 2012
def total_hours_february : ℕ := leap_year_february_days * hours_in_day

-- Define the average rainfall calculation
def average_rainfall_per_hour : ℚ :=
  total_rainfall / total_hours_february

-- Theorem to prove the average rainfall is 35/58 inches per hour
theorem average_rainfall_correct :
  average_rainfall_per_hour = 35 / 58 :=
by 
  -- Placeholder for proof
  sorry

end average_rainfall_correct_l160_160655


namespace quadratic_inequalities_solution_l160_160053

noncomputable def a : Type := sorry
noncomputable def b : Type := sorry
noncomputable def c : Type := sorry

theorem quadratic_inequalities_solution (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c > 0 ↔ -1/3 < x ∧ x < 2) :
  ∀ y, cx^2 + bx + a < 0 ↔ -3 < y ∧ y < 1/2 :=
sorry

end quadratic_inequalities_solution_l160_160053


namespace calculate_fraction_l160_160621

theorem calculate_fraction :
  (5 * 6 - 4) / 8 = 13 / 4 := 
by
  sorry

end calculate_fraction_l160_160621


namespace max_even_a_exists_max_even_a_l160_160028

theorem max_even_a (a : ℤ): (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k) → a ≤ 8 := sorry

theorem exists_max_even_a : ∃ a : ℤ, (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k ∧ a = 8) := sorry

end max_even_a_exists_max_even_a_l160_160028


namespace chosen_number_is_121_l160_160077

theorem chosen_number_is_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := 
by 
  sorry

end chosen_number_is_121_l160_160077


namespace middle_integer_is_five_l160_160189

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def are_consecutive_odd_integers (a b c : ℤ) : Prop :=
  a < b ∧ b < c ∧ (∃ n : ℤ, a = b - 2 ∧ c = b + 2 ∧ is_odd a ∧ is_odd b ∧ is_odd c)

def sum_is_one_eighth_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_five :
  ∃ (a b c : ℤ), are_consecutive_odd_integers a b c ∧ sum_is_one_eighth_product a b c ∧ b = 5 :=
by
  sorry

end middle_integer_is_five_l160_160189


namespace find_y_from_triangle_properties_l160_160899

-- Define angle measures according to the given conditions
def angle_BAC := 45
def angle_CDE := 72

-- Define the proof problem
theorem find_y_from_triangle_properties
: ∀ (y : ℝ), (∃ (BAC ACB ABC ADC ADE AED DEB : ℝ),
    angle_BAC = 45 ∧
    angle_CDE = 72 ∧
    BAC + ACB + ABC = 180 ∧
    ADC = 180 ∧
    ADE = 180 - angle_CDE ∧
    EAD = angle_BAC ∧
    AED + ADE + EAD = 180 ∧
    DEB = 180 - AED ∧
    y = DEB) →
    y = 153 :=
by sorry

end find_y_from_triangle_properties_l160_160899


namespace percent_employed_females_in_employed_population_l160_160805

def percent_employed (population: ℝ) : ℝ := 0.64 * population
def percent_employed_males (population: ℝ) : ℝ := 0.50 * population
def percent_employed_females (population: ℝ) : ℝ := percent_employed population - percent_employed_males population

theorem percent_employed_females_in_employed_population (population: ℝ) : 
  (percent_employed_females population / percent_employed population) * 100 = 21.875 :=
by
  sorry

end percent_employed_females_in_employed_population_l160_160805


namespace xyz_square_sum_l160_160708

theorem xyz_square_sum {x y z a b c d : ℝ} (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x + y + z = d) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  x^2 + y^2 + z^2 = d^2 - 2 * (a + b + c) :=
sorry

end xyz_square_sum_l160_160708


namespace x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l160_160545

theorem x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842
  (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l160_160545


namespace no_solution_for_ab_ba_l160_160035

theorem no_solution_for_ab_ba (a b x : ℕ)
  (ab ba : ℕ)
  (h_ab : ab = 10 * a + b)
  (h_ba : ba = 10 * b + a) :
  (ab^x - 2 = ba^x - 7) → false :=
by
  sorry

end no_solution_for_ab_ba_l160_160035


namespace rate_of_interest_per_annum_l160_160890

theorem rate_of_interest_per_annum (P R : ℝ) (T : ℝ) 
  (h1 : T = 8)
  (h2 : (P / 5) = (P * R * T) / 100) : 
  R = 2.5 := 
by
  sorry

end rate_of_interest_per_annum_l160_160890


namespace find_values_of_a_l160_160532

def P : Set ℝ := { x | x^2 + x - 6 = 0 }
def S (a : ℝ) : Set ℝ := { x | a * x + 1 = 0 }

theorem find_values_of_a (a : ℝ) : (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) := by
  sorry

end find_values_of_a_l160_160532


namespace walnut_trees_currently_in_park_l160_160463

-- Definitions from the conditions
def total_trees : ℕ := 77
def trees_to_be_planted : ℕ := 44

-- Statement to prove: number of current trees = 33
theorem walnut_trees_currently_in_park : total_trees - trees_to_be_planted = 33 :=
by
  sorry

end walnut_trees_currently_in_park_l160_160463


namespace handshake_problem_l160_160848

theorem handshake_problem (n : ℕ) (h : n * (n - 1) / 2 = 1770) : n = 60 :=
sorry

end handshake_problem_l160_160848


namespace priyas_fathers_age_l160_160088

-- Define Priya's age P and her father's age F
variables (P F : ℕ)

-- Define the conditions
def conditions : Prop :=
  F - P = 31 ∧ P + F = 53

-- Define the theorem to be proved
theorem priyas_fathers_age (h : conditions P F) : F = 42 :=
sorry

end priyas_fathers_age_l160_160088


namespace man_speed_with_the_stream_l160_160977

def speed_with_the_stream (V_m V_s : ℝ) : Prop :=
  V_m + V_s = 2

theorem man_speed_with_the_stream (V_m V_s : ℝ) (h1 : V_m - V_s = 2) (h2 : V_m = 2) : speed_with_the_stream V_m V_s :=
by
  sorry

end man_speed_with_the_stream_l160_160977


namespace giant_exponent_modulo_result_l160_160163

theorem giant_exponent_modulo_result :
  (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 :=
by sorry

end giant_exponent_modulo_result_l160_160163


namespace exponentiation_identity_l160_160679

theorem exponentiation_identity :
  (5^4)^2 = 390625 :=
  by sorry

end exponentiation_identity_l160_160679


namespace percentage_increase_first_year_l160_160571

-- Assume the original price of the painting is P and the percentage increase during the first year is X
variable {P : ℝ} (X : ℝ)

-- Condition: The price decreases by 15% during the second year
def condition_decrease (price : ℝ) : ℝ := price * 0.85

-- Condition: The price at the end of the 2-year period was 93.5% of the original price
axiom condition_end_price : ∀ (P : ℝ), (P + (X/100) * P) * 0.85 = 0.935 * P

-- Proof problem: What was the percentage increase during the first year?
theorem percentage_increase_first_year : X = 10 :=
by 
  sorry

end percentage_increase_first_year_l160_160571


namespace distance_between_midpoints_l160_160904

-- Conditions
def AA' := 68 -- in centimeters
def BB' := 75 -- in centimeters
def CC' := 112 -- in centimeters
def DD' := 133 -- in centimeters

-- Question: Prove the distance between the midpoints of A'C' and B'D' is 14 centimeters
theorem distance_between_midpoints :
  let midpoint_A'C' := (AA' + CC') / 2
  let midpoint_B'D' := (BB' + DD') / 2
  (midpoint_B'D' - midpoint_A'C' = 14) :=
by
  sorry

end distance_between_midpoints_l160_160904


namespace min_height_of_box_with_surface_area_condition_l160_160269

theorem min_height_of_box_with_surface_area_condition {x : ℕ}  
(h : 2*x^2 + 4*x*(x + 6) ≥ 150) (hx: x ≥ 5) : (x + 6) = 11 := by
  sorry

end min_height_of_box_with_surface_area_condition_l160_160269


namespace ab_cd_is_1_or_minus_1_l160_160040

theorem ab_cd_is_1_or_minus_1 (a b c d : ℤ) (h1 : ∃ k₁ : ℤ, a = k₁ * (a * b - c * d))
  (h2 : ∃ k₂ : ℤ, b = k₂ * (a * b - c * d)) (h3 : ∃ k₃ : ℤ, c = k₃ * (a * b - c * d))
  (h4 : ∃ k₄ : ℤ, d = k₄ * (a * b - c * d)) :
  a * b - c * d = 1 ∨ a * b - c * d = -1 := 
sorry

end ab_cd_is_1_or_minus_1_l160_160040


namespace symmetric_line_eq_l160_160490

-- Define points A and B
def A (a : ℝ) : ℝ × ℝ := (a-1, a+1)
def B (a : ℝ) : ℝ × ℝ := (a, a)

-- We want to prove the equation of the line L about which points A and B are symmetric is "x - y + 1 = 0".
theorem symmetric_line_eq (a : ℝ) : 
  ∃ m b, (m = 1) ∧ (b = 1) ∧ (∀ x y, (y = m * x + b) ↔ (x - y + 1 = 0)) :=
sorry

end symmetric_line_eq_l160_160490


namespace lucas_fraction_to_emma_l160_160580

variable (n : ℕ)

-- Define initial stickers
def noah_stickers := n
def emma_stickers := 3 * n
def lucas_stickers := 12 * n

-- Define the final state where each has the same number of stickers
def final_stickers_per_person := (16 * n) / 3

-- Lucas gives some stickers to Emma. Calculate the fraction of Lucas's stickers given to Emma
theorem lucas_fraction_to_emma :
  (7 * n / 3) / (12 * n) = 7 / 36 := by
  sorry

end lucas_fraction_to_emma_l160_160580


namespace percentage_enclosed_by_pentagons_l160_160520

-- Define the condition for the large square and smaller squares.
def large_square_area (b : ℝ) : ℝ := (4 * b) ^ 2

-- Define the condition for the number of smaller squares forming pentagons.
def pentagon_small_squares : ℝ := 10

-- Define the total number of smaller squares within a large square.
def total_small_squares : ℝ := 16

-- Prove that the percentage of the plane enclosed by pentagons is 62.5%.
theorem percentage_enclosed_by_pentagons :
  (pentagon_small_squares / total_small_squares) * 100 = 62.5 :=
by 
  -- The proof is left as an exercise.
  sorry

end percentage_enclosed_by_pentagons_l160_160520


namespace negation_abs_lt_one_l160_160306

theorem negation_abs_lt_one (x : ℝ) : (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_abs_lt_one_l160_160306


namespace distance_to_weekend_class_l160_160322

theorem distance_to_weekend_class:
  ∃ d v : ℝ, (d = v * (1 / 2)) ∧ (d = (v + 10) * (3 / 10)) → d = 7.5 :=
by
  sorry

end distance_to_weekend_class_l160_160322


namespace sophomore_spaghetti_tortellini_ratio_l160_160446

theorem sophomore_spaghetti_tortellini_ratio
    (total_students : ℕ)
    (spaghetti_lovers : ℕ)
    (tortellini_lovers : ℕ)
    (grade_levels : ℕ)
    (spaghetti_sophomores : ℕ)
    (tortellini_sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : spaghetti_lovers = 300)
    (h3 : tortellini_lovers = 120)
    (h4 : grade_levels = 4)
    (h5 : spaghetti_sophomores = spaghetti_lovers / grade_levels)
    (h6 : tortellini_sophomores = tortellini_lovers / grade_levels) :
    (spaghetti_sophomores : ℚ) / (tortellini_sophomores : ℚ) = 5 / 2 := by
  sorry

end sophomore_spaghetti_tortellini_ratio_l160_160446


namespace polar_equation_is_circle_l160_160192

-- Define the polar coordinates equation condition
def polar_equation (r θ : ℝ) : Prop := r = 5

-- Define what it means for a set of points to form a circle centered at the origin with a radius of 5
def is_circle_radius_5 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- State the theorem we want to prove
theorem polar_equation_is_circle (r θ : ℝ) (x y : ℝ) (h1 : polar_equation r θ)
  (h2 : x = r * Real.cos θ) (h3 : y = r * Real.sin θ) : is_circle_radius_5 x y := 
sorry

end polar_equation_is_circle_l160_160192


namespace children_got_off_bus_l160_160097

theorem children_got_off_bus (initial : ℕ) (got_on : ℕ) (after : ℕ) : Prop :=
  initial = 22 ∧ got_on = 40 ∧ after = 2 → initial + got_on - 60 = after


end children_got_off_bus_l160_160097


namespace train_journey_duration_l160_160190

variable (z x : ℝ)
variable (h1 : 1.7 = 1 + 42 / 60)
variable (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7)

theorem train_journey_duration (z x : ℝ)
    (h1 : 1.7 = 1 + 42 / 60)
    (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7):
    z / x = 10 := 
by
  sorry

end train_journey_duration_l160_160190


namespace finished_year_eq_183_l160_160011

theorem finished_year_eq_183 (x : ℕ) (h1 : x < 200) 
  (h2 : x ^ 13 = 258145266804692077858261512663) : x = 183 :=
sorry

end finished_year_eq_183_l160_160011


namespace part1_part2_l160_160311

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x^2 ≤ 5 * x - 4
def q (x a : ℝ) : Prop := x^2 - (a + 2) * x + 2 * a ≤ 0

-- Theorem statement for part (1)
theorem part1 (x : ℝ) (h : p x) : 1 ≤ x ∧ x ≤ 4 := 
by sorry

-- Theorem statement for part (2)
theorem part2 (a : ℝ) : 
  (∀ x, p x → q x a) ∧ (∃ x, p x) ∧ ¬ (∀ x, q x a → p x) → 1 ≤ a ∧ a ≤ 4 := 
by sorry

end part1_part2_l160_160311


namespace knitting_time_total_l160_160837

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end knitting_time_total_l160_160837


namespace largest_divisor_36_l160_160514

theorem largest_divisor_36 (n : ℕ) (h : n > 0) (h_div : 36 ∣ n^3) : 6 ∣ n := 
sorry

end largest_divisor_36_l160_160514


namespace solve_for_a_when_diamond_eq_6_l160_160455

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem solve_for_a_when_diamond_eq_6 (a : ℝ) : diamond a 3 = 6 → a = 8 :=
by
  intros h
  simp [diamond] at h
  sorry

end solve_for_a_when_diamond_eq_6_l160_160455


namespace ab_value_l160_160354

variables {a b : ℝ}

theorem ab_value (h₁ : a - b = 6) (h₂ : a^2 + b^2 = 50) : ab = 7 :=
sorry

end ab_value_l160_160354


namespace incorrect_observation_value_l160_160013

-- Definitions stemming from the given conditions
def initial_mean : ℝ := 100
def corrected_mean : ℝ := 99.075
def number_of_observations : ℕ := 40
def correct_observation_value : ℝ := 50

-- Lean theorem statement to prove the incorrect observation value
theorem incorrect_observation_value (initial_mean corrected_mean correct_observation_value : ℝ) (number_of_observations : ℕ) :
  (initial_mean * number_of_observations - corrected_mean * number_of_observations + correct_observation_value) = 87 := 
sorry

end incorrect_observation_value_l160_160013


namespace solve_for_a_l160_160676

def E (a b c : ℝ) : ℝ := a * b^2 + b * c + c

theorem solve_for_a : (E (-5/8) 3 2 = E (-5/8) 5 3) :=
by
  sorry

end solve_for_a_l160_160676


namespace filtration_minimum_l160_160894

noncomputable def lg : ℝ → ℝ := sorry

theorem filtration_minimum (x : ℕ) (lg2 : ℝ) (lg3 : ℝ) (h1 : lg2 = 0.3010) (h2 : lg3 = 0.4771) :
  (2 / 3 : ℝ) ^ x ≤ 1 / 20 → x ≥ 8 :=
sorry

end filtration_minimum_l160_160894


namespace combine_square_roots_l160_160989

def can_be_combined (x y: ℝ) : Prop :=
  ∃ k: ℝ, y = k * x

theorem combine_square_roots :
  let sqrt12 := 2 * Real.sqrt 3
  let sqrt1_3 := Real.sqrt 1 / Real.sqrt 3
  let sqrt18 := 3 * Real.sqrt 2
  let sqrt27 := 6 * Real.sqrt 3
  can_be_combined (Real.sqrt 3) sqrt12 ∧
  can_be_combined (Real.sqrt 3) sqrt1_3 ∧
  ¬ can_be_combined (Real.sqrt 3) sqrt18 ∧
  can_be_combined (Real.sqrt 3) sqrt27 :=
by
  sorry

end combine_square_roots_l160_160989


namespace no_real_solution_l160_160337

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end no_real_solution_l160_160337


namespace value_of_x_in_terms_of_z_l160_160956

variable {z : ℝ} {x y : ℝ}
  
theorem value_of_x_in_terms_of_z (h1 : y = z + 50) (h2 : x = 0.70 * y) : x = 0.70 * z + 35 := 
  sorry

end value_of_x_in_terms_of_z_l160_160956


namespace calculate_smaller_sphere_radius_l160_160456

noncomputable def smaller_sphere_radius (r1 r2 r3 r4 : ℝ) : ℝ := 
  if h : r1 = 2 ∧ r2 = 2 ∧ r3 = 3 ∧ r4 = 3 then 
    6 / 11 
  else 
    0

theorem calculate_smaller_sphere_radius :
  smaller_sphere_radius 2 2 3 3 = 6 / 11 :=
by
  sorry

end calculate_smaller_sphere_radius_l160_160456


namespace probability_correct_l160_160834

-- Define the problem conditions.
def num_balls : ℕ := 8
def possible_colors : ℕ := 2

-- Probability calculation for a specific arrangement (either configuration of colors).
def probability_per_arrangement : ℚ := (1/2) ^ num_balls

-- Number of favorable arrangements with 4 black and 4 white balls.
def favorable_arrangements : ℕ := Nat.choose num_balls 4

-- The required probability for the solution.
def desired_probability : ℚ := favorable_arrangements * probability_per_arrangement

-- The proof statement to be provided.
theorem probability_correct :
  desired_probability = 35 / 128 := 
by
  sorry

end probability_correct_l160_160834


namespace sum_of_ages_l160_160092

-- Define Henry's and Jill's present ages
def Henry_age : ℕ := 23
def Jill_age : ℕ := 17

-- Define the condition that 11 years ago, Henry was twice the age of Jill
def condition_11_years_ago : Prop := (Henry_age - 11) = 2 * (Jill_age - 11)

-- Theorem statement: sum of Henry's and Jill's present ages is 40
theorem sum_of_ages : Henry_age + Jill_age = 40 :=
by
  -- Placeholder for proof
  sorry

end sum_of_ages_l160_160092


namespace find_g8_l160_160957

variable (g : ℝ → ℝ)

theorem find_g8 (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) (h2 : g 7 = 8) : g 8 = 64 / 7 :=
sorry

end find_g8_l160_160957


namespace point_B_l160_160760

-- Define constants for perimeter and speed factor
def perimeter : ℕ := 24
def speed_factor : ℕ := 2

-- Define the speeds of Jane and Hector
def hector_speed (s : ℕ) : ℕ := s
def jane_speed (s : ℕ) : ℕ := speed_factor * s

-- Define the times until they meet
def time_until_meeting (s : ℕ) : ℚ := perimeter / (hector_speed s + jane_speed s)

-- Distances walked by Hector and Jane upon meeting
noncomputable def hector_distance (s : ℕ) : ℚ := hector_speed s * time_until_meeting s
noncomputable def jane_distance (s : ℕ) : ℚ := jane_speed s * time_until_meeting s

-- Map the perimeter position to a point
def position_on_track (d : ℚ) : ℚ := d % perimeter

-- When they meet
theorem point_B (s : ℕ) (h₀ : 0 < s) : position_on_track (hector_distance s) = position_on_track (jane_distance s) → 
                          position_on_track (hector_distance s) = 8 := 
by 
  sorry

end point_B_l160_160760


namespace volume_ratio_of_cubes_l160_160979

theorem volume_ratio_of_cubes :
  (4^3 / 10^3 : ℚ) = 8 / 125 := by
  sorry

end volume_ratio_of_cubes_l160_160979


namespace percentage_of_third_number_l160_160124

variable (T F S : ℝ)

-- Declare the conditions from step a)
def condition_one : Prop := S = 0.25 * T
def condition_two : Prop := F = 0.20 * S

-- Define the proof problem, proving that F is 5% of T given the conditions
theorem percentage_of_third_number
  (h1 : condition_one T S)
  (h2 : condition_two F S) :
  F = 0.05 * T := by
  sorry

end percentage_of_third_number_l160_160124


namespace group_sum_180_in_range_1_to_60_l160_160182

def sum_of_arithmetic_series (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem group_sum_180_in_range_1_to_60 :
  ∃ (a n : ℕ), 1 ≤ a ∧ a + n - 1 ≤ 60 ∧ sum_of_arithmetic_series a 1 n = 180 :=
by
  sorry

end group_sum_180_in_range_1_to_60_l160_160182


namespace second_divisor_13_l160_160292

theorem second_divisor_13 (N D : ℤ) (k m : ℤ) 
  (h1 : N = 39 * k + 17) 
  (h2 : N = D * m + 4) : 
  D = 13 := 
sorry

end second_divisor_13_l160_160292


namespace no_square_sum_l160_160959

theorem no_square_sum (x y : ℕ) (hxy_pos : 0 < x ∧ 0 < y)
  (hxy_gcd : Nat.gcd x y = 1)
  (hxy_perf : ∃ k : ℕ, x + 3 * y^2 = k^2) : ¬ ∃ z : ℕ, x^2 + 9 * y^4 = z^2 :=
by
  sorry

end no_square_sum_l160_160959


namespace largest_n_in_base10_l160_160806

-- Definitions corresponding to the problem conditions
def n_eq_base8_expr (A B C : ℕ) : ℕ := 64 * A + 8 * B + C
def n_eq_base12_expr (A B C : ℕ) : ℕ := 144 * C + 12 * B + A

-- Problem statement translated into Lean
theorem largest_n_in_base10 (n A B C : ℕ) (h1 : n = n_eq_base8_expr A B C) 
    (h2 : n = n_eq_base12_expr A B C) (hA : A < 8) (hB : B < 8) (hC : C < 12) (h_pos: n > 0) : 
    n ≤ 509 :=
sorry

end largest_n_in_base10_l160_160806


namespace cody_needs_total_steps_l160_160365

theorem cody_needs_total_steps 
  (weekly_steps : ℕ → ℕ)
  (h1 : ∀ n, weekly_steps n = (n + 1) * 1000 * 7)
  (h2 : 4 * 7 * 1000 + 3 * 7 * 1000 + 2 * 7 * 1000 + 1 * 7 * 1000 = 70000) 
  (h3 : 70000 + 30000 = 100000) :
  ∃ total_steps, total_steps = 100000 := 
by
  sorry

end cody_needs_total_steps_l160_160365


namespace min_value_range_l160_160081

theorem min_value_range:
  ∀ (x m n : ℝ), 
    (y = (3 * x + 2) / (x - 1)) → 
    (∀ x ∈ Set.Ioo m n, y ≥ 3 + 5 / (x - 1)) → 
    (y = 8) → 
    n = 2 → 
    (1 ≤ m ∧ m < 2) := by
  sorry

end min_value_range_l160_160081


namespace henry_wins_l160_160967

-- Definitions of conditions
def total_games : ℕ := 14
def losses : ℕ := 2
def draws : ℕ := 10

-- Statement of the theorem
theorem henry_wins : (total_games - losses - draws) = 2 :=
by
  -- Proof goes here
  sorry

end henry_wins_l160_160967


namespace max_quadratic_function_l160_160825

theorem max_quadratic_function :
  ∃ M, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → (x^2 - 2*x - 1 ≤ M)) ∧
       (∀ y : ℝ, y = (x : ℝ) ^ 2 - 2 * x - 1 → x = 3 → y = M) :=
by
  use 2
  sorry

end max_quadratic_function_l160_160825


namespace incorrect_operation_D_l160_160381

theorem incorrect_operation_D (x y: ℝ) : ¬ (-2 * x * (x - y) = -2 * x^2 - 2 * x * y) :=
by sorry

end incorrect_operation_D_l160_160381


namespace middle_of_three_consecutive_integers_is_60_l160_160100

theorem middle_of_three_consecutive_integers_is_60 (n : ℤ)
    (h : (n - 1) + n + (n + 1) = 180) : n = 60 := by
  sorry

end middle_of_three_consecutive_integers_is_60_l160_160100


namespace common_root_quadratic_l160_160370

theorem common_root_quadratic (a x1: ℝ) :
  (x1^2 + a * x1 + 1 = 0) ∧ (x1^2 + x1 + a = 0) ↔ a = -2 :=
sorry

end common_root_quadratic_l160_160370


namespace find_number_l160_160930

theorem find_number : ∃ (x : ℝ), x + 0.303 + 0.432 = 5.485 ↔ x = 4.750 := 
sorry

end find_number_l160_160930


namespace toothpicks_in_12th_stage_l160_160789

def toothpicks_in_stage (n : ℕ) : ℕ :=
  3 * n

theorem toothpicks_in_12th_stage : toothpicks_in_stage 12 = 36 :=
by
  -- Proof steps would go here, including simplification and calculations, but are omitted with 'sorry'.
  sorry

end toothpicks_in_12th_stage_l160_160789


namespace maximum_value_of_objective_function_l160_160824

variables (x y : ℝ)

def objective_function (x y : ℝ) := 3 * x + 2 * y

theorem maximum_value_of_objective_function : 
  (∀ x y, (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4) → objective_function x y ≤ 12) 
  ∧ 
  (∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4 ∧ objective_function x y = 12) :=
sorry

end maximum_value_of_objective_function_l160_160824


namespace veranda_area_correct_l160_160628

-- Define the dimensions of the room.
def room_length : ℕ := 20
def room_width : ℕ := 12

-- Define the width of the veranda.
def veranda_width : ℕ := 2

-- Calculate the total dimensions with the veranda.
def total_length : ℕ := room_length + 2 * veranda_width
def total_width : ℕ := room_width + 2 * veranda_width

-- Calculate the area of the room and the total area including the veranda.
def room_area : ℕ := room_length * room_width
def total_area : ℕ := total_length * total_width

-- Prove that the area of the veranda is 144 m².
theorem veranda_area_correct : total_area - room_area = 144 := by
  sorry

end veranda_area_correct_l160_160628


namespace total_amount_paid_l160_160386

-- Definitions based on the conditions in step a)
def ring_cost : ℕ := 24
def ring_quantity : ℕ := 2

-- Statement to prove that the total cost is $48.
theorem total_amount_paid : ring_quantity * ring_cost = 48 := 
by
  sorry

end total_amount_paid_l160_160386


namespace max_projection_area_l160_160136

noncomputable def maxProjectionArea (a : ℝ) : ℝ :=
  if a > (Real.sqrt 3 / 3) ∧ a <= (Real.sqrt 3 / 2) then
    Real.sqrt 3 / 4
  else if a >= (Real.sqrt 3 / 2) then
    a / 2
  else 
    0  -- if the condition for a is not met, it's an edge case which shouldn't logically occur here

theorem max_projection_area (a : ℝ) (h1 : a > Real.sqrt 3 / 3) (h2 : a <= Real.sqrt 3 / 2 ∨ a >= Real.sqrt 3 / 2) :
  maxProjectionArea a = 
    if a > Real.sqrt 3 / 3 ∧ a <= Real.sqrt 3 / 2 then Real.sqrt 3 / 4
    else if a >= Real.sqrt 3 / 2 then a / 2
    else
      sorry :=
by sorry

end max_projection_area_l160_160136


namespace cubes_sum_formula_l160_160343

theorem cubes_sum_formula (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) : a^3 + b^3 = 238 := 
by 
  sorry

end cubes_sum_formula_l160_160343


namespace sum_of_first_five_terms_l160_160414

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (h_sum_n : ∀ n, S n = n / 2 * (a 1 + a n))
  (h_roots : ∀ x, x^2 - x - 3 = 0 → x = a 2 ∨ x = a 4)
  (h_vieta : a 2 + a 4 = 1) :
  S 5 = 5 / 2 :=
  sorry

end sum_of_first_five_terms_l160_160414


namespace jerusha_earnings_l160_160208

variable (L : ℝ) 

theorem jerusha_earnings (h1 : L + 4 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l160_160208


namespace minimum_m_n_sum_l160_160423

theorem minimum_m_n_sum:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 90 * m = n ^ 3 ∧ m + n = 330 :=
sorry

end minimum_m_n_sum_l160_160423


namespace num_adult_tickets_is_35_l160_160379

noncomputable def num_adult_tickets_sold (A C: ℕ): Prop :=
  A + C = 85 ∧ 5 * A + 2 * C = 275

theorem num_adult_tickets_is_35: ∃ A C: ℕ, num_adult_tickets_sold A C ∧ A = 35 :=
by
  -- Definitions based on the provided conditions
  sorry

end num_adult_tickets_is_35_l160_160379


namespace gcd_problem_l160_160212

-- Define the two numbers
def a : ℕ := 1000000000
def b : ℕ := 1000000005

-- Define the problem to prove the GCD
theorem gcd_problem : Nat.gcd a b = 5 :=
by 
  sorry

end gcd_problem_l160_160212


namespace a_8_is_256_l160_160596

variable (a : ℕ → ℕ)

axiom a_1 : a 1 = 2

axiom a_pq : ∀ p q : ℕ, a (p + q) = a p * a q

theorem a_8_is_256 : a 8 = 256 := by
  sorry

end a_8_is_256_l160_160596


namespace mike_drive_average_rate_l160_160642

open Real

variables (total_distance first_half_distance second_half_distance first_half_speed second_half_speed first_half_time second_half_time total_time avg_rate j : ℝ)

theorem mike_drive_average_rate :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_distance = total_distance / 2 ∧
  first_half_speed = 80 ∧
  first_half_distance / first_half_speed = first_half_time ∧
  second_half_time = 3 * first_half_time ∧
  second_half_distance / second_half_time = second_half_speed ∧
  total_time = first_half_time + second_half_time ∧
  avg_rate = total_distance / total_time →
  j = 40 :=
by
  intro h
  sorry

end mike_drive_average_rate_l160_160642


namespace john_got_rolls_l160_160624

def cost_per_dozen : ℕ := 5
def money_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_got_rolls : (money_spent / cost_per_dozen) * rolls_per_dozen = 36 :=
by sorry

end john_got_rolls_l160_160624


namespace ginger_distance_l160_160318

theorem ginger_distance : 
  ∀ (d : ℝ), (d / 4 - d / 6 = 1 / 16) → (d = 3 / 4) := 
by 
  intro d h
  sorry

end ginger_distance_l160_160318


namespace Mark_paid_total_cost_l160_160599

def length_of_deck : ℝ := 30
def width_of_deck : ℝ := 40
def cost_per_sq_ft_without_sealant : ℝ := 3
def additional_cost_per_sq_ft_sealant : ℝ := 1

def area (length width : ℝ) : ℝ := length * width
def total_cost (area cost_without_sealant cost_sealant : ℝ) : ℝ := 
  area * cost_without_sealant + area * cost_sealant

theorem Mark_paid_total_cost :
  total_cost (area length_of_deck width_of_deck) cost_per_sq_ft_without_sealant additional_cost_per_sq_ft_sealant = 4800 := 
by
  -- Placeholder for proof
  sorry

end Mark_paid_total_cost_l160_160599


namespace solve_fraction_l160_160152

theorem solve_fraction (a b : ℝ) (hab : 3 * a = 2 * b) : (a + b) / b = 5 / 3 :=
by
  sorry

end solve_fraction_l160_160152


namespace price_of_remote_controlled_airplane_l160_160215

theorem price_of_remote_controlled_airplane (x : ℝ) (h : 300 = 0.8 * x) : x = 375 :=
by
  sorry

end price_of_remote_controlled_airplane_l160_160215


namespace henry_apple_weeks_l160_160775

theorem henry_apple_weeks (apples_per_box : ℕ) (boxes : ℕ) (people : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) :
  apples_per_box = 14 → boxes = 3 → people = 2 → apples_per_day = 1 → days_per_week = 7 →
  (apples_per_box * boxes) / (people * apples_per_day * days_per_week) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end henry_apple_weeks_l160_160775


namespace number_composition_l160_160836

theorem number_composition :
  5 * 100000 + 6 * 100 + 3 * 10 + 6 * 0.01 = 500630.06 := 
by 
  sorry

end number_composition_l160_160836


namespace solve_system_equations_l160_160127

variable (x y z : ℝ)

theorem solve_system_equations (h1 : 3 * x = 20 + (20 - x))
    (h2 : y = 2 * x - 5)
    (h3 : z = Real.sqrt (x + 4)) :
  x = 10 ∧ y = 15 ∧ z = Real.sqrt 14 :=
by
  sorry

end solve_system_equations_l160_160127


namespace bus_children_problem_l160_160469

theorem bus_children_problem :
  ∃ X, 5 - 63 + X = 14 ∧ X - 63 = 9 :=
by 
  sorry

end bus_children_problem_l160_160469


namespace quadratic_expression_l160_160650

theorem quadratic_expression (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 + 1 = 0) (h2 : x2^2 - 3 * x2 + 1 = 0) : 
  x1^2 - 2 * x1 + x2 = 2 :=
sorry

end quadratic_expression_l160_160650


namespace perpendicular_lines_slope_l160_160860

theorem perpendicular_lines_slope (m : ℝ) : 
  ((m ≠ -3) ∧ (m ≠ -5) ∧ 
  (- (m + 3) / 4 * - (2 / (m + 5)) = -1)) ↔ m = -13 / 3 := 
sorry

end perpendicular_lines_slope_l160_160860


namespace fraction_divisible_by_n_l160_160151

theorem fraction_divisible_by_n (a b n : ℕ) (h1 : a ≠ b) (h2 : n > 0) (h3 : n ∣ (a^n - b^n)) : n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end fraction_divisible_by_n_l160_160151


namespace parallel_vectors_tan_l160_160252

/-- Given vector a and vector b, and given the condition that a is parallel to b,
prove that the value of tan α is 1/4. -/
theorem parallel_vectors_tan (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.sin α, Real.cos α - 2 * Real.sin α))
  (hb : b = (1, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) : 
  Real.tan α = 1 / 4 := 
by 
  sorry

end parallel_vectors_tan_l160_160252


namespace tables_count_l160_160366

theorem tables_count (c t : Nat) (h1 : c = 8 * t) (h2 : 3 * c + 5 * t = 580) : t = 20 :=
by
  sorry

end tables_count_l160_160366


namespace initial_investment_proof_l160_160004

noncomputable def initial_investment (A : ℝ) (r t : ℕ) : ℝ := 
  A / (1 + r / 100) ^ t

theorem initial_investment_proof : 
  initial_investment 1000 8 8 = 630.17 := sorry

end initial_investment_proof_l160_160004


namespace proof_x_eq_y_l160_160122

variable (x y z : ℝ)

theorem proof_x_eq_y (h1 : x = 6 - y) (h2 : z^2 = x * y - 9) : x = y := 
  sorry

end proof_x_eq_y_l160_160122


namespace original_price_of_trouser_l160_160703

theorem original_price_of_trouser (sale_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 50) (h2 : discount_rate = 0.50) (h3 : sale_price = (1 - discount_rate) * original_price) : 
  original_price = 100 :=
sorry

end original_price_of_trouser_l160_160703


namespace arithmetic_sequence_a5_l160_160829

theorem arithmetic_sequence_a5
  (a : ℕ → ℤ) -- a is the arithmetic sequence function
  (S : ℕ → ℤ) -- S is the sum of the first n terms of the sequence
  (h1 : S 5 = 2 * S 4) -- Condition S_5 = 2S_4
  (h2 : a 2 + a 4 = 8) -- Condition a_2 + a_4 = 8
  (hS : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) -- Definition of S_n
  (ha : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) -- Definition of a_n
: a 5 = 10 := 
by
  -- proof
  sorry

end arithmetic_sequence_a5_l160_160829


namespace gcd_140_396_is_4_l160_160778

def gcd_140_396 : ℕ := Nat.gcd 140 396

theorem gcd_140_396_is_4 : gcd_140_396 = 4 :=
by
  unfold gcd_140_396
  sorry

end gcd_140_396_is_4_l160_160778


namespace geometric_sequence_sum_l160_160713

theorem geometric_sequence_sum (a1 r : ℝ) (S : ℕ → ℝ) :
  S 2 = 3 → S 4 = 15 →
  (∀ n, S n = a1 * (1 - r^n) / (1 - r)) → S 6 = 63 :=
by
  intros hS2 hS4 hSn
  sorry

end geometric_sequence_sum_l160_160713


namespace triangle_right_angle_l160_160225

theorem triangle_right_angle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : A + B = 90)
  (h2 : (a + b) * (a - b) = c ^ 2)
  (h3 : A / B = 1 / 2) :
  C = 90 :=
sorry

end triangle_right_angle_l160_160225


namespace total_amount_in_wallet_l160_160581

theorem total_amount_in_wallet
  (num_10_bills : ℕ)
  (num_20_bills : ℕ)
  (num_5_bills : ℕ)
  (amount_10_bills : ℕ)
  (num_20_bills_eq : num_20_bills = 4)
  (amount_10_bills_eq : amount_10_bills = 50)
  (total_num_bills : ℕ)
  (total_num_bills_eq : total_num_bills = 13)
  (num_10_bills_eq : num_10_bills = amount_10_bills / 10)
  (total_amount : ℕ)
  (total_amount_eq : total_amount = amount_10_bills + num_20_bills * 20 + num_5_bills * 5)
  (num_bills_accounted : ℕ)
  (num_bills_accounted_eq : num_bills_accounted = num_10_bills + num_20_bills)
  (num_5_bills_eq : num_5_bills = total_num_bills - num_bills_accounted)
  : total_amount = 150 :=
by
  sorry

end total_amount_in_wallet_l160_160581


namespace volume_of_rectangular_solid_l160_160702

theorem volume_of_rectangular_solid : 
  let l := 100 -- length in cm
  let w := 20  -- width in cm
  let h := 50  -- height in cm
  let V := l * w * h
  V = 100000 :=
by
  rfl

end volume_of_rectangular_solid_l160_160702


namespace option_C_correct_l160_160990

theorem option_C_correct (x : ℝ) : x^3 * x^2 = x^5 := sorry

end option_C_correct_l160_160990


namespace inequality_proof_l160_160726

variable (a b c : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)

theorem inequality_proof :
  (2 * a + b + c)^2 / (2 * a^2 + (b + c)^2) +
  (a + 2 * b + c)^2 / (2 * b^2 + (c + a)^2) +
  (a + b + 2 * c)^2 / (2 * c^2 + (a + b)^2) ≤ 8 := sorry

end inequality_proof_l160_160726


namespace value_of_m_l160_160645

noncomputable def has_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

noncomputable def has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

theorem value_of_m (m : ℝ) :
  (has_distinct_real_roots 1 m 1 ∧ has_no_real_roots 4 (4 * (m + 2)) 1) ↔ (-3 < m ∧ m < -2) :=
by
  sorry

end value_of_m_l160_160645


namespace total_original_grain_l160_160025

-- Define initial conditions
variables (initial_warehouse1 : ℕ) (initial_warehouse2 : ℕ)
-- Define the amount of grain transported away from the first warehouse
def transported_away := 2500
-- Define the amount of grain in the second warehouse
def warehouse2_initial := 50200

-- Prove the total original amount of grain in the two warehouses
theorem total_original_grain 
  (h1 : transported_away = 2500)
  (h2 : warehouse2_initial = 50200)
  (h3 : initial_warehouse1 - transported_away = warehouse2_initial) : 
  initial_warehouse1 + warehouse2_initial = 102900 :=
sorry

end total_original_grain_l160_160025


namespace smallest_fraction_greater_than_4_over_5_l160_160542

theorem smallest_fraction_greater_than_4_over_5 :
  ∃ (b : ℕ), 10 ≤ b ∧ b < 100 ∧ 77 * 5 > 4 * b ∧ Int.gcd 77 b = 1 ∧
  ∀ (a : ℕ), 10 ≤ a ∧ a < 77 → ¬ ∃ (b' : ℕ), 10 ≤ b' ∧ b' < 100 ∧ a * 5 > 4 * b' ∧ Int.gcd a b' = 1 := by
  sorry

end smallest_fraction_greater_than_4_over_5_l160_160542


namespace cube_surface_area_correct_l160_160164

noncomputable def total_surface_area_of_reassembled_cube : ℝ :=
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let top_bottom_area := 3 * 1 -- Each slab contributes 1 square foot for the top and bottom
  let side_area := 2 * 1 -- Each side slab contributes 1 square foot
  let front_back_area := 2 * 1 -- Each front and back contributes 1 square foot
  top_bottom_area + side_area + front_back_area

theorem cube_surface_area_correct :
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let total_surface_area := total_surface_area_of_reassembled_cube
  total_surface_area = 10 :=
by
  sorry

end cube_surface_area_correct_l160_160164


namespace expression_value_l160_160380

theorem expression_value (x y z : ℤ) (h1: x = 2) (h2: y = -3) (h3: z = 1) :
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := 
by
  sorry

end expression_value_l160_160380


namespace solve_equation_l160_160969

theorem solve_equation (x : ℝ) : x * (x + 1) = 12 → (x = -4 ∨ x = 3) :=
by
  sorry

end solve_equation_l160_160969


namespace total_boys_eq_350_l160_160259

variable (Total : ℕ)
variable (SchoolA : ℕ)
variable (NotScience : ℕ)

axiom h1 : SchoolA = 20 * Total / 100
axiom h2 : NotScience = 70 * SchoolA / 100
axiom h3 : NotScience = 49

theorem total_boys_eq_350 : Total = 350 :=
by
  sorry

end total_boys_eq_350_l160_160259


namespace polynomial_root_expression_l160_160807

theorem polynomial_root_expression (a b : ℂ) 
  (h₁ : a + b = 5) (h₂ : a * b = 6) : 
  a^4 + a^5 * b^3 + a^3 * b^5 + b^4 = 2905 := by
  sorry

end polynomial_root_expression_l160_160807


namespace unique_three_digit_numbers_unique_three_digit_odd_numbers_l160_160803

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end unique_three_digit_numbers_unique_three_digit_odd_numbers_l160_160803


namespace scientific_notation_of_18M_l160_160226

theorem scientific_notation_of_18M : 18000000 = 1.8 * 10^7 :=
by
  sorry

end scientific_notation_of_18M_l160_160226


namespace sum_first_3n_terms_is_36_l160_160258

-- Definitions and conditions
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2
def sum_first_2n_terms (a d : ℤ) (n : ℕ) : ℤ := 2 * n * (2 * a + (2 * n - 1) * d) / 2
def sum_first_3n_terms (a d : ℤ) (n : ℕ) : ℤ := 3 * n * (2 * a + (3 * n - 1) * d) / 2

axiom h1 : ∀ (a d : ℤ) (n : ℕ), sum_first_n_terms a d n = 48
axiom h2 : ∀ (a d : ℤ) (n : ℕ), sum_first_2n_terms a d n = 60

theorem sum_first_3n_terms_is_36 (a d : ℤ) (n : ℕ) : sum_first_3n_terms a d n = 36 := by
  sorry

end sum_first_3n_terms_is_36_l160_160258


namespace andrew_total_hours_l160_160947

theorem andrew_total_hours (days_worked : ℕ) (hours_per_day : ℝ)
    (h1 : days_worked = 3) (h2 : hours_per_day = 2.5) : 
    days_worked * hours_per_day = 7.5 := by
  sorry

end andrew_total_hours_l160_160947


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l160_160555

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {4, 5, 6, 7, 8, 9}
def B : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem union_of_A_and_B : A ∪ B = U := by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {4, 5, 6} := by
  sorry

theorem complement_of_intersection : U \ (A ∩ B) = {1, 2, 3, 7, 8, 9} := by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l160_160555


namespace ratio_of_volumes_l160_160578

theorem ratio_of_volumes (r1 r2 : ℝ) (h : (4 * π * r1^2) / (4 * π * r2^2) = 4 / 9) :
  (4/3 * π * r1^3) / (4/3 * π * r2^3) = 8 / 27 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_volumes_l160_160578


namespace cube_root_approx_l160_160641

open Classical

theorem cube_root_approx (n : ℤ) (x : ℝ) (h₁ : 2^n = x^3) (h₂ : abs (x - 50) <  1) : n = 17 := by
  sorry

end cube_root_approx_l160_160641


namespace cylinder_height_l160_160787

theorem cylinder_height
  (r : ℝ) (SA : ℝ) (h : ℝ)
  (h_radius : r = 3)
  (h_surface_area_given : SA = 30 * Real.pi)
  (h_surface_area_formula : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  h = 2 :=
by
  -- Proof can be written here
  sorry

end cylinder_height_l160_160787


namespace rational_number_addition_l160_160245

theorem rational_number_addition :
  (-206 : ℚ) + (401 + 3 / 4) + (-(204 + 2 / 3)) + (-(1 + 1 / 2)) = -10 - 5 / 12 :=
by
  sorry

end rational_number_addition_l160_160245


namespace find_a1_l160_160538

-- Define the arithmetic sequence and the given conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean (x y z : ℝ) : Prop :=
  y^2 = x * z

def problem_statement (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (arithmetic_sequence a d) ∧ (geometric_mean (a 1) (a 2) (a 4))

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) (h : problem_statement a d) : a 1 = 1 := by
  have h_seq : arithmetic_sequence a d := h.1
  have h_geom : geometric_mean (a 1) (a 2) (a 4) := h.2
  sorry

end find_a1_l160_160538


namespace ab_value_l160_160149

   variable (log2_3 : Real) (b : Real) (a : Real)

   -- Hypotheses
   def log_condition : Prop := log2_3 = 1
   def exp_condition (b : Real) : Prop := (4:Real) ^ b = 3
   
   -- Final statement to prove
   theorem ab_value (h_log2_3 : log_condition log2_3) (h_exp : exp_condition b) 
   (ha : a = 1) : a * b = 1 / 2 := sorry
   
end ab_value_l160_160149


namespace mryak_bryak_problem_l160_160244

variable (m b : ℚ)

theorem mryak_bryak_problem
  (h1 : 3 * m = 5 * b + 10)
  (h2 : 6 * m = 8 * b + 31) :
  7 * m - 9 * b = 38 := sorry

end mryak_bryak_problem_l160_160244


namespace max_total_cut_length_l160_160276

theorem max_total_cut_length :
  let side_length := 30
  let num_pieces := 225
  let area_per_piece := (side_length ^ 2) / num_pieces
  let outer_perimeter := 4 * side_length
  let max_perimeter_per_piece := 10
  (num_pieces * max_perimeter_per_piece - outer_perimeter) / 2 = 1065 :=
by
  sorry

end max_total_cut_length_l160_160276


namespace square_side_length_l160_160310

theorem square_side_length
  (P : ℕ) (A : ℕ) (s : ℕ)
  (h1 : P = 44)
  (h2 : A = 121)
  (h3 : P = 4 * s)
  (h4 : A = s * s) :
  s = 11 :=
sorry

end square_side_length_l160_160310


namespace count_valid_n_l160_160491

theorem count_valid_n : ∃ (count : ℕ), count = 6 ∧ ∀ n : ℕ,
  0 < n ∧ n < 42 → (∃ m : ℕ, m > 0 ∧ n = 42 * m / (m + 1)) :=
by
  sorry

end count_valid_n_l160_160491


namespace journey_time_l160_160746

noncomputable def journey_time_proof : Prop :=
  ∃ t1 t2 t3 : ℝ,
    25 * t1 - 25 * t2 + 25 * t3 = 100 ∧
    5 * t1 + 5 * t2 + 25 * t3 = 100 ∧
    25 * t1 + 5 * t2 + 5 * t3 = 100 ∧
    t1 + t2 + t3 = 8

theorem journey_time : journey_time_proof := by sorry

end journey_time_l160_160746


namespace f_is_odd_l160_160098

open Real

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  (1 + sin x)^(2 * n) - (1 - sin x)^(2 * n)

theorem f_is_odd (n : ℕ) (h : n > 0) : ∀ x : ℝ, f (-x) n = -f x n :=
by
  intros x
  -- Proof goes here
  sorry

end f_is_odd_l160_160098


namespace remaining_area_l160_160277

-- Definitions based on conditions
def large_rectangle_length (x : ℝ) : ℝ := 2 * x + 8
def large_rectangle_width (x : ℝ) : ℝ := x + 6
def hole_length (x : ℝ) : ℝ := 3 * x - 4
def hole_width (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem remaining_area (x : ℝ) : (large_rectangle_length x) * (large_rectangle_width x) - (hole_length x) * (hole_width x) = -x^2 + 21 * x + 52 :=
by
  -- Proof is skipped
  sorry

end remaining_area_l160_160277


namespace track_width_l160_160417

variable (r1 r2 r3 : ℝ)

def cond1 : Prop := 2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi
def cond2 : Prop := 2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi

theorem track_width (h1 : cond1 r1 r2) (h2 : cond2 r2 r3) : r3 - r1 = 25 := by
  sorry

end track_width_l160_160417


namespace forester_total_trees_planted_l160_160205

theorem forester_total_trees_planted (initial_trees monday_trees tuesday_trees wednesday_trees total_trees : ℕ)
    (h1 : initial_trees = 30)
    (h2 : total_trees = 300)
    (h3 : monday_trees = 2 * initial_trees)
    (h4 : tuesday_trees = monday_trees / 3)
    (h5 : wednesday_trees = 2 * tuesday_trees) : 
    (monday_trees + tuesday_trees + wednesday_trees = 120) := by
  sorry

end forester_total_trees_planted_l160_160205


namespace multiple_of_first_number_l160_160791

theorem multiple_of_first_number (F S M : ℕ) (hF : F = 15) (hS : S = 55) (h_relation : S = M * F + 10) : M = 3 :=
by
  -- We are given that F = 15, S = 55 and the relation S = M * F + 10
  -- We need to prove that M = 3
  sorry

end multiple_of_first_number_l160_160791


namespace roots_real_and_equal_l160_160485

theorem roots_real_and_equal :
  ∀ x : ℝ,
  (x^2 - 4 * x * Real.sqrt 5 + 20 = 0) →
  (Real.sqrt ((-4 * Real.sqrt 5)^2 - 4 * 1 * 20) = 0) →
  (∃ r : ℝ, x = r ∧ x = r) :=
by
  intro x h_eq h_discriminant
  sorry

end roots_real_and_equal_l160_160485


namespace smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l160_160859

def is_composite (n : ℕ) : Prop := (∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_square_perimeter_of_isosceles_triangle_with_composite_sides :
  ∃ a b : ℕ,
    is_composite a ∧
    is_composite b ∧
    (2 * a + b) ^ 2 = 256 :=
sorry

end smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l160_160859


namespace min_value_l160_160236

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (1 / x) + (4 / y) ≥ 9 :=
sorry

end min_value_l160_160236


namespace face_card_then_number_card_prob_l160_160987

-- Definitions from conditions
def num_cards := 52
def num_face_cards := 12
def num_number_cards := 40
def total_ways_to_pick_two_cards := 52 * 51

-- Theorem statement
theorem face_card_then_number_card_prob : 
  (num_face_cards * num_number_cards) / total_ways_to_pick_two_cards = (40 : ℚ) / 221 :=
by
  sorry

end face_card_then_number_card_prob_l160_160987


namespace triangles_congruent_alternative_condition_l160_160403

theorem triangles_congruent_alternative_condition
  (A B C A' B' C' : Type)
  (AB A'B' AC A'C' : ℝ)
  (angleA angleA' : ℝ)
  (h1 : AB = A'B')
  (h2 : angleA = angleA')
  (h3 : AC = A'C') :
  ∃ (triangleABC triangleA'B'C' : Type), (triangleABC = triangleA'B'C') :=
by sorry

end triangles_congruent_alternative_condition_l160_160403


namespace find_c_l160_160165

theorem find_c (c : ℝ) (h : ∀ x, 2 < x ∧ x < 6 → -x^2 + c * x + 8 > 0) : c = 8 := 
by
  sorry

end find_c_l160_160165


namespace problem_l160_160535

open Set

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def N : Set ℝ := { x | x < 0 }
def complement_N : Set ℝ := { x | x ≥ 0 }

theorem problem : M ∩ complement_N = { x | 0 ≤ x ∧ x < 3 } :=
by
  sorry

end problem_l160_160535


namespace max_ab_min_fraction_l160_160095

-- Question 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : ab ≤ 25/21 := sorry

-- Question 2: Minimum value of (3/a + 7/b)
theorem min_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : 3/a + 7/b ≥ 10 := sorry

end max_ab_min_fraction_l160_160095


namespace question_1_question_2_l160_160222

open Real

noncomputable def f (x a : ℝ) := abs (x - a) + 3 * x

theorem question_1 :
  {x : ℝ | f x 1 > 3 * x + 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by 
  sorry
  
theorem question_2 (h : {x : ℝ | f x a ≤ 0} = {x : ℝ | x ≤ -1}) :
  a = 2 :=
by 
  sorry

end question_1_question_2_l160_160222


namespace number_of_licenses_l160_160951

-- We define the conditions for the problem
def number_of_letters : ℕ := 3  -- B, C, or D
def number_of_digits : ℕ := 4   -- Four digits following the letter
def choices_per_digit : ℕ := 10 -- Each digit can range from 0 to 9

-- We define the total number of licenses that can be generated
def total_licenses : ℕ := number_of_letters * (choices_per_digit ^ number_of_digits)

-- We now state the theorem to be proved
theorem number_of_licenses : total_licenses = 30000 :=
by
  sorry

end number_of_licenses_l160_160951


namespace cake_heavier_than_bread_l160_160473

-- Definitions
def weight_of_7_cakes_eq_1950_grams (C : ℝ) := 7 * C = 1950
def weight_of_5_cakes_12_breads_eq_2750_grams (C B : ℝ) := 5 * C + 12 * B = 2750

-- Statement
theorem cake_heavier_than_bread (C B : ℝ)
  (h1 : weight_of_7_cakes_eq_1950_grams C)
  (h2 : weight_of_5_cakes_12_breads_eq_2750_grams C B) :
  C - B = 165.47 :=
by {
  sorry
}

end cake_heavier_than_bread_l160_160473


namespace isosceles_triangle_perimeter_l160_160442

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_eq_triangle : a + b + c = 60) (h_eq_sides : a = b) 
  (isosceles_base : c = 15) (isosceles_side1_eq : a = 20) : a + b + c = 55 :=
by
  sorry

end isosceles_triangle_perimeter_l160_160442


namespace find_c_l160_160158

theorem find_c (a b c d : ℕ) (h1 : 8 = 4 * a / 100) (h2 : 4 = d * a / 100) (h3 : 8 = d * b / 100) (h4 : c = b / a) : 
  c = 2 := 
by
  sorry

end find_c_l160_160158


namespace circle_equation_l160_160048

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 10 * y = 0

theorem circle_equation
  (x y : ℝ)
  (center_on_y_axis : ∃ r : ℝ, r > 0 ∧ x^2 + (y - r)^2 = r^2)
  (tangent_to_x_axis : ∃ r : ℝ, r > 0 ∧ y = r)
  (passes_through_point : x = 3 ∧ y = 1) :
  equation_of_circle x y :=
by
  sorry

end circle_equation_l160_160048


namespace union_of_sets_l160_160938

open Set

theorem union_of_sets (A B : Set ℝ) (hA : A = {x | -2 < x ∧ x < 1}) (hB : B = {x | 0 < x ∧ x < 2}) :
  A ∪ B = {x | -2 < x ∧ x < 2} :=
sorry

end union_of_sets_l160_160938


namespace expected_groups_l160_160785

/-- Given a sequence of k zeros and m ones arranged in random order and divided into 
alternating groups of identical digits, the expected value of the total number of 
groups is 1 + 2 * k * m / (k + m). -/
theorem expected_groups (k m : ℕ) (h_pos : k + m > 0) :
  let total_groups := 1 + 2 * k * m / (k + m)
  total_groups = (1 + 2 * k * m / (k + m)) := by
  sorry

end expected_groups_l160_160785


namespace range_of_a_l160_160939

theorem range_of_a (a : ℝ) :
  (¬ ( ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0 ) 
    ∨ 
   ¬ ( ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0 )) 
→ a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end range_of_a_l160_160939


namespace triangle_incircle_ratio_l160_160228

theorem triangle_incircle_ratio
  (a b c : ℝ) (ha : a = 15) (hb : b = 12) (hc : c = 9)
  (r s : ℝ) (hr : r + s = c) (r_lt_s : r < s) :
  r / s = 1 / 2 :=
sorry

end triangle_incircle_ratio_l160_160228


namespace isosceles_triangle_base_length_l160_160559

theorem isosceles_triangle_base_length
  (a : ℕ) (b : ℕ)
  (ha : a = 7) 
  (p : ℕ)
  (hp : p = a + a + b) 
  (hp_perimeter : p = 21) : b = 7 :=
by 
  -- The actual proof will go here, using the provided conditions
  sorry

end isosceles_triangle_base_length_l160_160559


namespace player_A_winning_strategy_l160_160435

-- Define the game state and the player's move
inductive Move
| single (index : Nat) : Move
| double (index : Nat) : Move

-- Winning strategy prop
def winning_strategy (n : Nat) (first_player : Bool) : Prop :=
  ∀ moves : List Move, moves.length ≤ n → (first_player → false) → true

-- Main theorem stating that player A always has a winning strategy
theorem player_A_winning_strategy (n : Nat) (h : n ≥ 1) : winning_strategy n true := 
by 
  -- directly prove the statement
  sorry

end player_A_winning_strategy_l160_160435


namespace part1_part2_l160_160465

noncomputable def cost_prices (x y : ℕ) : Prop := 
  8800 / (y + 4) = 2 * (4000 / x) ∧ 
  x = 40 ∧ 
  y = 44

theorem part1 : ∃ x y : ℕ, cost_prices x y := sorry

noncomputable def minimum_lucky_rabbits (m : ℕ) : Prop := 
  26 * m + 20 * (200 - m) ≥ 4120 ∧ 
  m = 20

theorem part2 : ∃ m : ℕ, minimum_lucky_rabbits m := sorry

end part1_part2_l160_160465


namespace scientific_notation_240000_l160_160395

theorem scientific_notation_240000 :
  240000 = 2.4 * 10^5 :=
by
  sorry

end scientific_notation_240000_l160_160395


namespace initial_necklaces_count_l160_160093

theorem initial_necklaces_count (N : ℕ) 
  (h1 : N - 13 = 37) : 
  N = 50 := 
by
  sorry

end initial_necklaces_count_l160_160093


namespace ratio_of_ages_three_years_ago_l160_160661

theorem ratio_of_ages_three_years_ago (k Y_c : ℕ) (h1 : 45 - 3 = k * (Y_c - 3)) (h2 : (45 + 7) + (Y_c + 7) = 83) : (45 - 3) / (Y_c - 3) = 2 :=
by {
  sorry
}

end ratio_of_ages_three_years_ago_l160_160661


namespace xy_sum_l160_160589

theorem xy_sum (x y : ℝ) (h1 : x^3 + 6 * x^2 + 16 * x = -15) (h2 : y^3 + 6 * y^2 + 16 * y = -17) : x + y = -4 :=
by
  -- The proof can be skipped with 'sorry'
  sorry

end xy_sum_l160_160589


namespace quadratic_equation_same_solutions_l160_160882

theorem quadratic_equation_same_solutions :
  ∃ b c : ℝ, (b, c) = (1, -7) ∧ (∀ x : ℝ, (x - 3 = 4 ∨ 3 - x = 4) ↔ (x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_equation_same_solutions_l160_160882


namespace car_sales_decrease_l160_160536

theorem car_sales_decrease (P N : ℝ) (h1 : 1.30 * P / (N * (1 - D / 100)) = 1.8571 * (P / N)) : D = 30 :=
by
  sorry

end car_sales_decrease_l160_160536


namespace abs_eq_abs_iff_eq_frac_l160_160300

theorem abs_eq_abs_iff_eq_frac {x : ℚ} :
  |x - 3| = |x - 4| → x = 7 / 2 :=
by
  intro h
  sorry

end abs_eq_abs_iff_eq_frac_l160_160300


namespace sophomores_sampled_correct_l160_160883

def stratified_sampling_sophomores (total_students num_sophomores sample_size : ℕ) : ℕ :=
  (num_sophomores * sample_size) / total_students

theorem sophomores_sampled_correct :
  stratified_sampling_sophomores 4500 1500 600 = 200 :=
by
  sorry

end sophomores_sampled_correct_l160_160883


namespace proof_problem_l160_160658

theorem proof_problem (a1 a2 a3 : ℕ) (h1 : a1 = a2 - 1) (h2 : a3 = a2 + 1) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by sorry

end proof_problem_l160_160658


namespace tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l160_160482

def tight_sequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → (1/2 : ℚ) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)

noncomputable def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = a 1 * q ^ (n - 1)

theorem tight_sequence_from_sum_of_terms (S : ℕ → ℚ) (a : ℕ → ℚ) : 
  (∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)) →
  (∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) →
  tight_sequence a :=
sorry

theorem range_of_q_for_tight_sequences (a : ℕ → ℚ) (S : ℕ → ℚ) (q : ℚ) :
  geometric_sequence a q →
  tight_sequence a →
  tight_sequence S →
  (1 / 2 : ℚ) ≤ q ∧ q < 1 :=
sorry

end tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l160_160482


namespace net_profit_from_plant_sales_l160_160472

noncomputable def calculate_net_profit : ℝ :=
  let cost_basil := 2.00
  let cost_mint := 3.00
  let cost_zinnia := 7.00
  let cost_soil := 15.00
  let total_cost := cost_basil + cost_mint + cost_zinnia + cost_soil
  let basil_germinated := 20 * 0.80
  let mint_germinated := 15 * 0.75
  let zinnia_germinated := 10 * 0.70
  let revenue_healthy_basil := 12 * 5.00
  let revenue_small_basil := 8 * 3.00
  let revenue_healthy_mint := 10 * 6.00
  let revenue_small_mint := 4 * 4.00
  let revenue_healthy_zinnia := 5 * 10.00
  let revenue_small_zinnia := 2 * 7.00
  let total_revenue := revenue_healthy_basil + revenue_small_basil + revenue_healthy_mint + revenue_small_mint + revenue_healthy_zinnia + revenue_small_zinnia
  total_revenue - total_cost

theorem net_profit_from_plant_sales : calculate_net_profit = 197.00 := by
  sorry

end net_profit_from_plant_sales_l160_160472


namespace sum_square_divisors_positive_l160_160483

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end sum_square_divisors_positive_l160_160483


namespace karen_cookies_grandparents_l160_160191

theorem karen_cookies_grandparents :
  ∀ (total_cookies cookies_kept class_size cookies_per_person : ℕ)
  (cookies_given_class cookies_left cookies_to_grandparents : ℕ),
  total_cookies = 50 →
  cookies_kept = 10 →
  class_size = 16 →
  cookies_per_person = 2 →
  cookies_given_class = class_size * cookies_per_person →
  cookies_left = total_cookies - cookies_kept - cookies_given_class →
  cookies_to_grandparents = cookies_left →
  cookies_to_grandparents = 8 :=
by
  intros
  sorry

end karen_cookies_grandparents_l160_160191


namespace f_sub_f_inv_eq_2022_l160_160992

def f (n : ℕ) : ℕ := 2 * n
def f_inv (n : ℕ) : ℕ := n

theorem f_sub_f_inv_eq_2022 : f 2022 - f_inv 2022 = 2022 := by
  -- Proof goes here
  sorry

end f_sub_f_inv_eq_2022_l160_160992


namespace not_perfect_square_of_divisor_l160_160118

theorem not_perfect_square_of_divisor (n d : ℕ) (hn : 0 < n) (hd : d ∣ 2 * n^2) :
  ¬ ∃ x : ℕ, n^2 + d = x^2 :=
by
  sorry

end not_perfect_square_of_divisor_l160_160118


namespace water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l160_160755

-- Define the tiered water pricing function
def tiered_water_cost (m : ℕ) : ℝ :=
  if m ≤ 20 then
    1.6 * m
  else if m ≤ 30 then
    1.6 * 20 + 2.4 * (m - 20)
  else
    1.6 * 20 + 2.4 * 10 + 4.8 * (m - 30)

-- Problem 1
theorem water_cost_10_tons : tiered_water_cost 10 = 16 := 
sorry

-- Problem 2
theorem water_cost_27_tons : tiered_water_cost 27 = 48.8 := 
sorry

-- Problem 3
theorem water_cost_between_20_30 (m : ℕ) (h : 20 < m ∧ m < 30) : tiered_water_cost m = 2.4 * m - 16 := 
sorry

-- Problem 4
theorem water_cost_above_30 (m : ℕ) (h : m > 30) : tiered_water_cost m = 4.8 * m - 88 := 
sorry

end water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l160_160755


namespace joan_paid_230_l160_160700

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 := 
by 
  sorry

end joan_paid_230_l160_160700


namespace simplify_and_evaluate_expression_l160_160333

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) : 
  (1 - (3 / (m + 3))) / (m / (m^2 + 6 * m + 9)) = Real.sqrt 2 := 
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l160_160333


namespace manuscript_total_cost_l160_160866

theorem manuscript_total_cost
  (P R1 R2 R3 : ℕ)
  (RateFirst RateRevision : ℕ)
  (hP : P = 300)
  (hR1 : R1 = 55)
  (hR2 : R2 = 35)
  (hR3 : R3 = 25)
  (hRateFirst : RateFirst = 8)
  (hRateRevision : RateRevision = 6) :
  let RemainingPages := P - (R1 + R2 + R3)
  let CostNoRevisions := RemainingPages * RateFirst
  let CostOneRevision := R1 * (RateFirst + RateRevision)
  let CostTwoRevisions := R2 * (RateFirst + 2 * RateRevision)
  let CostThreeRevisions := R3 * (RateFirst + 3 * RateRevision)
  let TotalCost := CostNoRevisions + CostOneRevision + CostTwoRevisions + CostThreeRevisions
  TotalCost = 3600 :=
by
  sorry

end manuscript_total_cost_l160_160866


namespace composite_probability_l160_160844

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l160_160844


namespace find_b_for_intersection_l160_160042

theorem find_b_for_intersection (b : ℝ) :
  (∀ x : ℝ, bx^2 + 2 * x + 3 = 3 * x + 4 → bx^2 - x - 1 = 0) →
  (∀ x : ℝ, x^2 * b - x - 1 = 0 → (1 + 4 * b = 0) → b = -1/4) :=
by
  intros h_eq h_discriminant h_solution
  sorry

end find_b_for_intersection_l160_160042


namespace given_tan_alpha_eq_3_then_expression_eq_8_7_l160_160513

theorem given_tan_alpha_eq_3_then_expression_eq_8_7 (α : ℝ) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := 
by
  sorry

end given_tan_alpha_eq_3_then_expression_eq_8_7_l160_160513


namespace sum_of_first_seven_primes_mod_eighth_prime_l160_160109

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 :=
by
  sorry

end sum_of_first_seven_primes_mod_eighth_prime_l160_160109


namespace sum_of_ages_is_14_l160_160822

/-- Kiana has two older twin brothers and the product of their three ages is 72.
    Prove that the sum of their three ages is 14. -/
theorem sum_of_ages_is_14 (kiana_age twin_age : ℕ) (htwins : twin_age > kiana_age) (h_product : kiana_age * twin_age * twin_age = 72) :
  kiana_age + twin_age + twin_age = 14 :=
sorry

end sum_of_ages_is_14_l160_160822


namespace smallest_positive_period_of_f_l160_160240

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * Real.sin (x + Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x :=
by
  intros
  -- Proof is omitted
  sorry

end smallest_positive_period_of_f_l160_160240


namespace second_number_is_46_l160_160278

theorem second_number_is_46 (sum_is_330 : ∃ (a b c d : ℕ), a + b + c + d = 330)
    (first_is_twice_second : ∀ (b : ℕ), ∃ (a : ℕ), a = 2 * b)
    (third_is_one_third_of_first : ∀ (a : ℕ), ∃ (c : ℕ), c = a / 3)
    (fourth_is_half_difference : ∀ (a b : ℕ), ∃ (d : ℕ), d = (a - b) / 2) :
  ∃ (b : ℕ), b = 46 :=
by
  -- Proof goes here, inserted for illustrating purposes only
  sorry

end second_number_is_46_l160_160278


namespace maximum_value_fraction_sum_l160_160408

theorem maximum_value_fraction_sum (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : 0 < c) (hd : 0 < d) (h1 : a + c = 20) (h2 : (a : ℝ) / b + (c : ℝ) / d < 1) :
  (a : ℝ) / b + (c : ℝ) / d ≤ 1385 / 1386 :=
sorry

end maximum_value_fraction_sum_l160_160408


namespace ratio_of_radii_l160_160901

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end ratio_of_radii_l160_160901


namespace solve_system_l160_160216

theorem solve_system (x y : ℚ) 
  (h₁ : 7 * x - 14 * y = 3) 
  (h₂ : 3 * y - x = 5) : 
  x = 79 / 7 ∧ y = 38 / 7 := 
by 
  sorry

end solve_system_l160_160216


namespace intersection_complement_l160_160260

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 > 4}

-- Define set N
def N : Set ℝ := {x : ℝ | (x - 3) / (x + 1) < 0}

-- Complement of N in U
def complement_N : Set ℝ := {x : ℝ | x <= -1} ∪ {x : ℝ | x >= 3}

-- Final proof to show intersection
theorem intersection_complement :
  M ∩ complement_N = {x : ℝ | x < -2} ∪ {x : ℝ | x >= 3} :=
by
  sorry

end intersection_complement_l160_160260


namespace max_value_a_plus_b_plus_c_l160_160001

-- Definitions used in the problem
def A_n (a n : ℕ) : ℕ := a * (10^n - 1) / 9
def B_n (b n : ℕ) : ℕ := b * (10^n - 1) / 9
def C_n (c n : ℕ) : ℕ := c * (10^(2 * n) - 1) / 9

-- Main statement of the problem
theorem max_value_a_plus_b_plus_c (n : ℕ) (a b c : ℕ) (h : n > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_eq : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n c n1 - B_n b n1 = 2 * (A_n a n1)^2 ∧ C_n c n2 - B_n b n2 = 2 * (A_n a n2)^2) :
  a + b + c ≤ 18 :=
sorry

end max_value_a_plus_b_plus_c_l160_160001


namespace tenth_term_is_513_l160_160230

def nth_term (n : ℕ) : ℕ :=
  2^(n-1) + 1

theorem tenth_term_is_513 : nth_term 10 = 513 := 
by 
  sorry

end tenth_term_is_513_l160_160230


namespace area_of_set_K_l160_160271

open Metric

def set_K :=
  {p : ℝ × ℝ | (abs p.1 + abs (3 * p.2) - 6) * (abs (3 * p.1) + abs p.2 - 6) ≤ 0}

def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Define the area function for a general set s

theorem area_of_set_K : area set_K = 24 :=
  sorry

end area_of_set_K_l160_160271


namespace average_weight_l160_160998

variable (A B C : ℕ)

theorem average_weight (h1 : A + B = 140) (h2 : B + C = 100) (h3 : B = 60) :
  (A + B + C) / 3 = 60 := 
sorry

end average_weight_l160_160998


namespace magnitude_range_l160_160049

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def vector_b : ℝ × ℝ := (Real.sqrt 3, -1)

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_range (θ : ℝ) : 
  0 ≤ (vector_magnitude (2 • vector_a θ - vector_b)) ∧ (vector_magnitude (2 • vector_a θ - vector_b)) ≤ 4 := 
sorry

end magnitude_range_l160_160049


namespace equation_of_line_AC_l160_160265

-- Define the given points A and B
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-3, -5)

-- Define the line m as a predicate
def line_m (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 + 6 = 0

-- Define the condition that line m is the angle bisector of ∠ACB
def is_angle_bisector (A B C : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : Prop := sorry

-- The symmetric point of B with respect to line m
def symmetric_point (B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : (ℝ × ℝ) := sorry

-- Proof statement
theorem equation_of_line_AC :
  ∀ (A B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop),
  A = (1, 1) →
  B = (-3, -5) →
  m = line_m →
  is_angle_bisector A B (symmetric_point B m) m →
  AC = {p : ℝ × ℝ | p.1 = 1} := sorry

end equation_of_line_AC_l160_160265


namespace team_total_score_l160_160672

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end team_total_score_l160_160672


namespace correct_sequence_l160_160102

def step1 := "Collect the admission ticket"
def step2 := "Register"
def step3 := "Written and computer-based tests"
def step4 := "Photography"

theorem correct_sequence : [step2, step4, step1, step3] = ["Register", "Photography", "Collect the admission ticket", "Written and computer-based tests"] :=
by
  sorry

end correct_sequence_l160_160102


namespace triangle_BC_length_l160_160188

theorem triangle_BC_length
  (y_eq_2x2 : ∀ (x : ℝ), ∃ (y : ℝ), y = 2 * x ^ 2)
  (area_ABC : ∃ (A B C : ℝ × ℝ), 
    A = (0, 0) ∧ (∃ (a : ℝ), B = (a, 2 * a ^ 2) ∧ C = (-a, 2 * a ^ 2) ∧ 2 * a ^ 3 = 128))
  : ∃ (a : ℝ), 2 * a = 8 := 
sorry

end triangle_BC_length_l160_160188


namespace number_of_students_l160_160557

noncomputable def is_handshakes_correct (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 
  (1 / 2 : ℚ) * (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) = 1020

theorem number_of_students (m n : ℕ) (h : is_handshakes_correct m n) : m * n = 280 := sorry

end number_of_students_l160_160557


namespace solve_quadratic_eq_l160_160629

theorem solve_quadratic_eq (x y z w d X Y Z W : ℤ) 
    (h1 : w % 2 = z % 2) 
    (h2 : x = 2 * d * (X * Z - Y * W))
    (h3 : y = 2 * d * (X * W + Y * Z))
    (h4 : z = d * (X^2 + Y^2 - Z^2 - W^2))
    (h5 : w = d * (X^2 + Y^2 + Z^2 + W^2)) :
    x^2 + y^2 + z^2 = w^2 :=
sorry

end solve_quadratic_eq_l160_160629


namespace solve_system_of_equations_l160_160699

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 7) (h2 : 2 * x - y = 2) :
  x = 3 ∧ y = 4 :=
by
  sorry

end solve_system_of_equations_l160_160699


namespace value_of_C_l160_160922

theorem value_of_C (C : ℝ) (h : 4 * C + 3 = 25) : C = 5.5 :=
by
  sorry

end value_of_C_l160_160922


namespace value_of_f_at_2_l160_160971

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem value_of_f_at_2 : f 2 = 3 := by
  -- Definition of the function f.
  -- The goal is to prove that f(2) = 3.
  sorry

end value_of_f_at_2_l160_160971


namespace sum_of_three_numbers_l160_160796

noncomputable def lcm_three_numbers (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_of_three_numbers 
  (a b c : ℕ)
  (x : ℕ)
  (h1 : lcm_three_numbers a b c = 180)
  (h2 : a = 2 * x)
  (h3 : b = 3 * x)
  (h4 : c = 5 * x) : a + b + c = 60 :=
by
  sorry

end sum_of_three_numbers_l160_160796


namespace value_of_expression_l160_160496

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := 
by
  sorry

end value_of_expression_l160_160496


namespace value_of_c_l160_160543

-- Define a structure representing conditions of the problem
structure ProblemConditions where
  c : Real

-- Define the problem in terms of given conditions and required proof
theorem value_of_c (conditions : ProblemConditions) : conditions.c = 5 / 2 := by
  sorry

end value_of_c_l160_160543


namespace total_legos_156_l160_160119

def pyramid_bottom_legos (side_length : Nat) : Nat := side_length * side_length
def pyramid_second_level_legos (length : Nat) (width : Nat) : Nat := length * width
def pyramid_third_level_legos (side_length : Nat) : Nat :=
  let total_legos := (side_length * (side_length + 1)) / 2
  total_legos - 3  -- Subtracting 3 Legos for the corners

def pyramid_fourth_level_legos : Nat := 1

def total_pyramid_legos : Nat :=
  pyramid_bottom_legos 10 +
  pyramid_second_level_legos 8 6 +
  pyramid_third_level_legos 4 +
  pyramid_fourth_level_legos

theorem total_legos_156 : total_pyramid_legos = 156 := by
  sorry

end total_legos_156_l160_160119


namespace sufficient_but_not_necessary_condition_l160_160921

def P (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1/x + 4 * x + 6 * m) ≥ 0

def Q (m : ℝ) : Prop :=
  m ≥ -5

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (P m → Q m) ∧ ¬(Q m → P m) := sorry

end sufficient_but_not_necessary_condition_l160_160921


namespace solve_for_y_l160_160250

theorem solve_for_y (y : ℝ) (h : (y * (y^5)^(1/4))^(1/3) = 4) : y = 2^(8/3) :=
by {
  sorry
}

end solve_for_y_l160_160250


namespace last_score_is_87_l160_160280

-- Definitions based on conditions:
def scores : List ℕ := [73, 78, 82, 84, 87, 95]
def total_sum := 499
def final_median := 83

-- Prove that the last score is 87 under given conditions.
theorem last_score_is_87 (h1 : total_sum = 499)
                        (h2 : ∀ n ∈ scores, (499 - n) % 6 ≠ 0)
                        (h3 : final_median = 83) :
  87 ∈ scores := sorry

end last_score_is_87_l160_160280


namespace bench_allocation_l160_160790

theorem bench_allocation (M : ℕ) : (∃ M, M > 0 ∧ 5 * M = 13 * M) → M = 5 :=
by
  sorry

end bench_allocation_l160_160790


namespace central_angle_of_probability_l160_160351

theorem central_angle_of_probability (x : ℝ) (h1 : x / 360 = 1 / 6) : x = 60 := by
  have h2 : x = 60 := by
    linarith
  exact h2

end central_angle_of_probability_l160_160351


namespace mean_equality_l160_160108

theorem mean_equality (x : ℝ) 
  (h : (7 + 9 + 23) / 3 = (16 + x) / 2) : 
  x = 10 := 
sorry

end mean_equality_l160_160108


namespace larger_integer_l160_160652

-- Definitions based on the given conditions
def two_integers (x : ℤ) (y : ℤ) :=
  y = 4 * x ∧ (x + 12) * 2 = y

-- Statement of the problem
theorem larger_integer (x : ℤ) (y : ℤ) (h : two_integers x y) : y = 48 :=
by sorry

end larger_integer_l160_160652


namespace expression_value_l160_160819

theorem expression_value (x : ℝ) (h : x = 3) : x^4 - 4 * x^2 = 45 := by
  sorry

end expression_value_l160_160819


namespace rope_segments_after_folding_l160_160707

theorem rope_segments_after_folding (n : ℕ) (h : n = 6) : 2^n + 1 = 65 :=
by
  rw [h]
  norm_num

end rope_segments_after_folding_l160_160707


namespace find_blue_weights_l160_160062

theorem find_blue_weights (B : ℕ) :
  (2 * B + 15 + 2 = 25) → B = 4 :=
by
  intro h
  sorry

end find_blue_weights_l160_160062


namespace find_xyz_value_l160_160622

noncomputable def xyz_satisfying_conditions (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (x + 1/y = 5) ∧
  (y + 1/z = 2) ∧
  (z + 1/x = 3)

theorem find_xyz_value (x y z : ℝ) (h : xyz_satisfying_conditions x y z) : x * y * z = 1 :=
by
  sorry

end find_xyz_value_l160_160622


namespace total_weight_of_fish_is_correct_l160_160036

noncomputable def totalWeightInFirstTank := 15 * 0.08 + 12 * 0.05

noncomputable def totalWeightInSecondTank := 2 * 15 * 0.08 + 3 * 12 * 0.05

noncomputable def totalWeightInThirdTank := 3 * 15 * 0.08 + 2 * 12 * 0.05 + 5 * 0.14

noncomputable def totalWeightAllTanks := totalWeightInFirstTank + totalWeightInSecondTank + totalWeightInThirdTank

theorem total_weight_of_fish_is_correct : 
  totalWeightAllTanks = 11.5 :=
by         
  sorry

end total_weight_of_fish_is_correct_l160_160036


namespace combined_ratio_l160_160470

theorem combined_ratio (cayley_students fermat_students : ℕ) 
                       (cayley_ratio_boys cayley_ratio_girls fermat_ratio_boys fermat_ratio_girls : ℕ) 
                       (h_cayley : cayley_students = 400) 
                       (h_cayley_ratio : (cayley_ratio_boys, cayley_ratio_girls) = (3, 2)) 
                       (h_fermat : fermat_students = 600) 
                       (h_fermat_ratio : (fermat_ratio_boys, fermat_ratio_girls) = (2, 3)) :
  (480 : ℚ) / 520 = 12 / 13 := 
by 
  sorry

end combined_ratio_l160_160470


namespace number_of_people_l160_160052

theorem number_of_people (total_cookies : ℕ) (cookies_per_person : ℝ) (h1 : total_cookies = 144) (h2 : cookies_per_person = 24.0) : total_cookies / cookies_per_person = 6 := 
by 
  -- Placeholder for actual proof.
  sorry

end number_of_people_l160_160052


namespace exactly_one_three_digit_perfect_cube_divisible_by_25_l160_160017

theorem exactly_one_three_digit_perfect_cube_divisible_by_25 :
  ∃! (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 25 = 0 :=
sorry

end exactly_one_three_digit_perfect_cube_divisible_by_25_l160_160017


namespace rectangular_solid_depth_l160_160769

def SurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth
  (l w A : ℝ)
  (hl : l = 10)
  (hw : w = 9)
  (hA : A = 408) :
  ∃ h : ℝ, SurfaceArea l w h = A ∧ h = 6 :=
by
  use 6
  sorry

end rectangular_solid_depth_l160_160769


namespace problem_T8_l160_160461

noncomputable def a : Nat → ℚ
| 0     => 1/2
| (n+1) => a n / (1 + 3 * a n)

noncomputable def T (n : Nat) : ℚ :=
  (Finset.range n).sum (λ i => 1 / a (i + 1))

theorem problem_T8 : T 8 = 100 :=
sorry

end problem_T8_l160_160461


namespace books_in_series_l160_160332

-- Define the number of movies
def M := 14

-- Define that the number of books is one more than the number of movies
def B := M + 1

-- Theorem statement to prove that the number of books is 15
theorem books_in_series : B = 15 :=
by
  sorry

end books_in_series_l160_160332


namespace no_four_distinct_numbers_l160_160434

theorem no_four_distinct_numbers (x y : ℝ) (h : x ≠ y ∧ 
    (x^(10:ℕ) + (x^(9:ℕ)) * y + (x^(8:ℕ)) * (y^(2:ℕ)) + 
    (x^(7:ℕ)) * (y^(3:ℕ)) + (x^(6:ℕ)) * (y^(4:ℕ)) + 
    (x^(5:ℕ)) * (y^(5:ℕ)) + (x^(4:ℕ)) * (y^(6:ℕ)) + 
    (x^(3:ℕ)) * (y^(7:ℕ)) + (x^(2:ℕ)) * (y^(8:ℕ)) + 
    (x^(1:ℕ)) * (y^(9:ℕ)) + (y^(10:ℕ)) = 1)) : False :=
by
  sorry

end no_four_distinct_numbers_l160_160434


namespace surface_area_l160_160794

theorem surface_area (r : ℝ) (π : ℝ) (V : ℝ) (S : ℝ) 
  (h1 : V = 48 * π) 
  (h2 : V = (4 / 3) * π * r^3) : 
  S = 4 * π * r^2 :=
  sorry

end surface_area_l160_160794


namespace fraction_six_power_l160_160116

theorem fraction_six_power (n : ℕ) (hyp : n = 6 ^ 2024) : n / 6 = 6 ^ 2023 :=
by sorry

end fraction_six_power_l160_160116


namespace simplify_expression_l160_160505

theorem simplify_expression : 4 * (15 / 7) * (21 / -45) = -4 :=
by 
    -- Lean's type system will verify the correctness of arithmetic simplifications.
    sorry

end simplify_expression_l160_160505


namespace log_sum_even_l160_160657

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the condition for maximum value at x = 1
def has_max_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x

-- Main theorem statement: Prove that lg x + lg y is an even function
theorem log_sum_even (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) 
  (hf_max : has_max_value_at (f A ω φ) 1) : 
  ∀ x y : ℝ, Real.log x + Real.log y = Real.log y + Real.log x := by
  sorry

end log_sum_even_l160_160657


namespace average_age_at_marriage_l160_160020

theorem average_age_at_marriage
  (A : ℕ)
  (combined_age_at_marriage : husband_age + wife_age = 2 * A)
  (combined_age_after_5_years : (A + 5) + (A + 5) + 1 = 57) :
  A = 23 := 
sorry

end average_age_at_marriage_l160_160020


namespace sum_of_abcd_l160_160917

theorem sum_of_abcd (a b c d: ℝ) (h₁: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂: c + d = 10 * a) (h₃: c * d = -11 * b) (h₄: a + b = 10 * c) (h₅: a * b = -11 * d)
  : a + b + c + d = 1210 := by
  sorry

end sum_of_abcd_l160_160917


namespace mrs_lee_earnings_percentage_l160_160728

noncomputable def percentage_earnings_june (T : ℝ) : ℝ :=
  let L := 0.5 * T
  let L_June := 1.2 * L
  let total_income_june := T
  (L_June / total_income_june) * 100

theorem mrs_lee_earnings_percentage (T : ℝ) (hT : T ≠ 0) : percentage_earnings_june T = 60 :=
by
  sorry

end mrs_lee_earnings_percentage_l160_160728


namespace saved_percentage_is_correct_l160_160117

def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 5200
def amount_saved : ℝ := 2300

noncomputable def total_expenses : ℝ :=
  rent + milk + groceries + education + petrol + miscellaneous

noncomputable def total_salary : ℝ :=
  total_expenses + amount_saved

noncomputable def percentage_saved : ℝ :=
  (amount_saved / total_salary) * 100

theorem saved_percentage_is_correct :
  percentage_saved = 8.846 := by
  sorry

end saved_percentage_is_correct_l160_160117


namespace value_of_I_l160_160767

variables (T H I S : ℤ)

theorem value_of_I :
  H = 10 →
  T + H + I + S = 50 →
  H + I + T = 35 →
  S + I + T = 40 →
  I = 15 :=
  by
  sorry

end value_of_I_l160_160767


namespace least_element_in_T_l160_160378

variable (S : Finset ℕ)
variable (T : Finset ℕ)
variable (hS : S = Finset.range 16 \ {0})
variable (hT : T.card = 5)
variable (hTsubS : T ⊆ S)
variable (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0))

theorem least_element_in_T (S T : Finset ℕ) (hT : T.card = 5) (hTsubS : T ⊆ S)
  (hCond : ∀ x y, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) : 
  ∃ m ∈ T, m = 5 :=
by
  sorry

end least_element_in_T_l160_160378


namespace original_difference_in_books_l160_160460

theorem original_difference_in_books 
  (x y : ℕ) 
  (h1 : x + y = 5000) 
  (h2 : (1 / 2 : ℚ) * (x - 400) - (y + 400) = 400) : 
  x - y = 3000 := 
by 
  -- Placeholder for the proof
  sorry

end original_difference_in_books_l160_160460


namespace num_ways_to_divide_friends_l160_160800

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end num_ways_to_divide_friends_l160_160800


namespace find_m_of_parallel_lines_l160_160604

theorem find_m_of_parallel_lines (m : ℝ) 
  (H1 : ∃ x y : ℝ, m * x + 2 * y + 6 = 0) 
  (H2 : ∃ x y : ℝ, x + (m - 1) * y + m^2 - 1 = 0) : 
  m = -1 := 
by
  sorry

end find_m_of_parallel_lines_l160_160604


namespace max_value_of_a_l160_160161

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (x + 1) * (1 + Real.log (x + 1)) - a * x

theorem max_value_of_a (a : ℤ) : 
  (∀ x : ℝ, x ≥ -1 → (a : ℝ) * x ≤ (x + 1) * (1 + Real.log (x + 1))) → a ≤ 3 := sorry

end max_value_of_a_l160_160161


namespace range_of_m_l160_160634

theorem range_of_m {m : ℝ} (h1 : m^2 - 1 < 0) (h2 : m > 0) : 0 < m ∧ m < 1 :=
sorry

end range_of_m_l160_160634


namespace simplified_expression_correct_l160_160572

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) :=
  sorry

end simplified_expression_correct_l160_160572


namespace perfect_square_trinomial_l160_160357

theorem perfect_square_trinomial (m x : ℝ) : 
  ∃ a b : ℝ, (4 * x^2 + (m - 3) * x + 1 = (a + b)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_l160_160357


namespace tower_height_proof_l160_160221

-- Definitions corresponding to given conditions
def elev_angle_A : ℝ := 45
def distance_AD : ℝ := 129
def elev_angle_D : ℝ := 60
def tower_height : ℝ := 305

-- Proving the height of Liaoning Broadcasting and Television Tower
theorem tower_height_proof (h : ℝ) (AC CD : ℝ) (h_eq_AC : h = AC) (h_eq_CD_sqrt3 : h = CD * (Real.sqrt 3)) (AC_CD_sum : AC + CD = 129) :
  h = 305 :=
by
  sorry

end tower_height_proof_l160_160221


namespace divisibility_l160_160412

theorem divisibility (a : ℤ) : (5 ∣ a^3) ↔ (5 ∣ a) := 
by sorry

end divisibility_l160_160412


namespace symmetric_to_origin_l160_160843

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end symmetric_to_origin_l160_160843


namespace f_sum_2018_2019_l160_160964

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom even_shifted_function (x : ℝ) : f (x + 1) = f (-x + 1)
axiom f_neg1 : f (-1) = -1

theorem f_sum_2018_2019 : f 2018 + f 2019 = -1 :=
by sorry

end f_sum_2018_2019_l160_160964


namespace proof_problem_l160_160063

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * x)

theorem proof_problem
  (a : ℝ)
  (h1 : ∀ x : ℝ, f (x - 1/2) = f (x + 1/2))
  (h2 : f (-1/4) = a) :
  f (9/4) = -a :=
by sorry

end proof_problem_l160_160063


namespace find_a4_l160_160840

noncomputable def a (n : ℕ) : ℕ := sorry -- Define the arithmetic sequence
def S (n : ℕ) : ℕ := sorry -- Define the sum function for the sequence

theorem find_a4 (h1 : S 5 = 25) (h2 : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l160_160840


namespace short_video_length_l160_160927

theorem short_video_length 
  (videos_per_day : ℕ) 
  (short_videos_factor : ℕ) 
  (weekly_total_minutes : ℕ) 
  (days_in_week : ℕ) 
  (total_videos : videos_per_day = 3)
  (one_video_longer : short_videos_factor = 6)
  (total_weekly_minutes : weekly_total_minutes = 112)
  (days_a_week : days_in_week = 7) :
  ∃ x : ℕ, (videos_per_day * (short_videos_factor + 2)) * days_in_week = weekly_total_minutes ∧ 
            x = 2 := 
by 
  sorry 

end short_video_length_l160_160927


namespace probability_donation_to_A_l160_160714

-- Define population proportions
def prob_O : ℝ := 0.50
def prob_A : ℝ := 0.15
def prob_B : ℝ := 0.30
def prob_AB : ℝ := 0.05

-- Define blood type compatibility predicate
def can_donate_to_A (blood_type : ℝ) : Prop := 
  blood_type = prob_O ∨ blood_type = prob_A

-- Theorem statement
theorem probability_donation_to_A : 
  prob_O + prob_A = 0.65 :=
by
  -- proof skipped
  sorry

end probability_donation_to_A_l160_160714


namespace compute_fraction_l160_160275

theorem compute_fraction (a b c : ℝ) (h1 : a + b = 20) (h2 : b + c = 22) (h3 : c + a = 2022) :
  (a - b) / (c - a) = 1000 :=
by
  sorry

end compute_fraction_l160_160275


namespace arithmetic_progression_contains_sixth_power_l160_160970

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end arithmetic_progression_contains_sixth_power_l160_160970


namespace star_contains_2011_l160_160660

theorem star_contains_2011 :
  ∃ (n : ℕ), n = 183 ∧ 
  (∃ (seq : List ℕ), seq = List.range' (2003) 11 ∧ 2011 ∈ seq) :=
by
  sorry

end star_contains_2011_l160_160660


namespace smallest_n_l160_160394

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end smallest_n_l160_160394


namespace find_fg3_l160_160797

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := (x + 2)^2 - 4 * x

theorem find_fg3 : f (g 3) = 49 :=
by
  sorry

end find_fg3_l160_160797


namespace solve_sum_of_coefficients_l160_160000

theorem solve_sum_of_coefficients (a b : ℝ) 
  (h1 : ∀ x, ax^2 - bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) : a + b = -10 :=
  sorry

end solve_sum_of_coefficients_l160_160000


namespace euler_no_k_divisible_l160_160003

theorem euler_no_k_divisible (n : ℕ) (k : ℕ) (h : k < 5^n - 5^(n-1)) : ¬ (5^n ∣ 2^k - 1) := 
sorry

end euler_no_k_divisible_l160_160003


namespace sqrt_expression_eq_l160_160511

theorem sqrt_expression_eq : 
  (Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3)) = -Real.sqrt 3 + 4 := 
by
  sorry

end sqrt_expression_eq_l160_160511


namespace perp_line_through_point_l160_160193

variable (x y c : ℝ)

def line_perpendicular (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

def perpendicular_line (x y c : ℝ) : Prop :=
  2*x + y + c = 0

theorem perp_line_through_point :
  (line_perpendicular x y) ∧ (perpendicular_line (-2) 3 1) :=
by
  -- The first part asserts that the given line equation holds
  have h1 : line_perpendicular x y := sorry
  -- The second part asserts that our calculated line passes through the point (-2, 3) and is perpendicular
  have h2 : perpendicular_line (-2) 3 1 := sorry
  exact ⟨h1, h2⟩

end perp_line_through_point_l160_160193


namespace f_2_eq_1_l160_160352

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

theorem f_2_eq_1 (a b : ℝ) (h : f a b (-2) = 1) : f a b 2 = 1 :=
by {
  sorry 
}

end f_2_eq_1_l160_160352


namespace differentiable_difference_constant_l160_160091

variable {R : Type*} [AddCommGroup R] [Module ℝ R]

theorem differentiable_difference_constant (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) 
  (h : ∀ x, fderiv ℝ f x = fderiv ℝ g x) : 
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end differentiable_difference_constant_l160_160091


namespace partial_fraction_product_l160_160918

theorem partial_fraction_product (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x^2 - 10 * x + 24 ≠ 0 →
            (x^2 - 25) / (x^3 - 3 * x^2 - 10 * x + 24) = A / (x - 2) + B / (x + 3) + C / (x - 4)) →
  A = 1 ∧ B = 1 ∧ C = 1 →
  A * B * C = 1 := by
  sorry

end partial_fraction_product_l160_160918


namespace part_one_part_two_l160_160649

noncomputable def f (a x : ℝ) : ℝ :=
  |x + (1 / a)| + |x - a + 1|

theorem part_one (a : ℝ) (h : a > 0) (x : ℝ) : f a x ≥ 1 :=
sorry

theorem part_two (a : ℝ) (h : a > 0) : f a 3 < 11 / 2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part_one_part_two_l160_160649


namespace budget_allocation_degrees_l160_160673

theorem budget_allocation_degrees :
  let microphotonics := 12.3
  let home_electronics := 17.8
  let food_additives := 9.4
  let gmo := 21.7
  let industrial_lubricants := 6.2
  let artificial_intelligence := 4.1
  let nanotechnology := 5.3
  let basic_astrophysics := 100 - (microphotonics + home_electronics + food_additives + gmo + industrial_lubricants + artificial_intelligence + nanotechnology)
  (basic_astrophysics * 3.6) + (artificial_intelligence * 3.6) + (nanotechnology * 3.6) = 117.36 :=
by
  sorry

end budget_allocation_degrees_l160_160673


namespace sarees_with_6_shirts_l160_160204

-- Define the prices of sarees, shirts and the equation parameters
variables (S T : ℕ) (X : ℕ)

-- Define the conditions as hypotheses
def condition1 := 2 * S + 4 * T = 1600
def condition2 := 12 * T = 2400
def condition3 := X * S + 6 * T = 1600

-- Define the theorem to prove X = 1 under these conditions
theorem sarees_with_6_shirts
  (h1 : condition1 S T)
  (h2 : condition2 T)
  (h3 : condition3 S T X) : 
  X = 1 :=
sorry

end sarees_with_6_shirts_l160_160204


namespace xy_range_l160_160583

theorem xy_range (x y : ℝ)
  (h1 : x + y = 1)
  (h2 : 1 / 3 ≤ x ∧ x ≤ 2 / 3) :
  2 / 9 ≤ x * y ∧ x * y ≤ 1 / 4 :=
sorry

end xy_range_l160_160583


namespace area_difference_of_circles_l160_160880

theorem area_difference_of_circles (circumference_large: ℝ) (half_radius_relation: ℝ → ℝ) (hl: circumference_large = 36) (hr: ∀ R, half_radius_relation R = R / 2) :
  ∃ R r, R = 18 / π ∧ r = 9 / π ∧ (π * R ^ 2 - π * r ^ 2) = 243 / π :=
by 
  sorry

end area_difference_of_circles_l160_160880


namespace never_2003_pieces_l160_160878

theorem never_2003_pieces :
  ¬∃ n : ℕ, (n = 5 + 4 * k) ∧ (n = 2003) :=
by
  sorry

end never_2003_pieces_l160_160878


namespace smallest_sector_angle_24_l160_160242

theorem smallest_sector_angle_24
  (a : ℕ) (d : ℕ)
  (h1 : ∀ i, i < 8 → ((a + i * d) : ℤ) > 0)
  (h2 : (2 * a + 7 * d = 90)) : a = 24 :=
by
  sorry

end smallest_sector_angle_24_l160_160242


namespace six_digit_ababab_divisible_by_101_l160_160326

theorem six_digit_ababab_divisible_by_101 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) :
  ∃ k : ℕ, 101 * k = 101010 * a + 10101 * b :=
sorry

end six_digit_ababab_divisible_by_101_l160_160326


namespace integer_sum_19_l160_160841

variable (p q r s : ℤ)

theorem integer_sum_19 (h1 : p - q + r = 4) 
                       (h2 : q - r + s = 5) 
                       (h3 : r - s + p = 7) 
                       (h4 : s - p + q = 3) :
                       p + q + r + s = 19 :=
by
  sorry

end integer_sum_19_l160_160841


namespace find_integer_a_l160_160958

theorem find_integer_a (a : ℤ) : (∃ x : ℕ, a * x = 3) ↔ a = 1 ∨ a = 3 :=
by
  sorry

end find_integer_a_l160_160958


namespace pq_square_eq_169_div_4_l160_160879

-- Defining the quadratic equation and the condition on solutions p and q.
def quadratic_eq (x : ℚ) : Prop := 2 * x^2 + 7 * x - 15 = 0

-- Defining the specific solutions p and q.
def p : ℚ := 3 / 2
def q : ℚ := -5

-- The main theorem stating that (p - q)^2 = 169 / 4 given the conditions.
theorem pq_square_eq_169_div_4 (hp : quadratic_eq p) (hq : quadratic_eq q) : (p - q) ^ 2 = 169 / 4 :=
by
  -- Proof is omitted using sorry
  sorry

end pq_square_eq_169_div_4_l160_160879


namespace largest_base_5_five_digit_number_in_decimal_l160_160865

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end largest_base_5_five_digit_number_in_decimal_l160_160865


namespace bisecting_line_of_circle_l160_160261

theorem bisecting_line_of_circle : 
  (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y + 1 = 0 → x - y + 1 = 0) := 
sorry

end bisecting_line_of_circle_l160_160261


namespace val_total_money_l160_160488

theorem val_total_money : 
  ∀ (nickels_initial dimes_initial nickels_found : ℕ),
    nickels_initial = 20 →
    dimes_initial = 3 * nickels_initial →
    nickels_found = 2 * nickels_initial →
    (nickels_initial * 5 + dimes_initial * 10 + nickels_found * 5) / 100 = 9 :=
by
  intros nickels_initial dimes_initial nickels_found h1 h2 h3
  sorry

end val_total_money_l160_160488


namespace cost_per_piece_l160_160162

-- Definitions based on the problem conditions
def total_cost : ℕ := 80         -- Total cost is $80
def num_pizzas : ℕ := 4          -- Luigi bought 4 pizzas
def pieces_per_pizza : ℕ := 5    -- Each pizza was cut into 5 pieces

-- Main theorem statement proving the cost per piece
theorem cost_per_piece :
  (total_cost / (num_pizzas * pieces_per_pizza)) = 4 :=
by
  sorry

end cost_per_piece_l160_160162


namespace inequality_solution_empty_l160_160487

theorem inequality_solution_empty {a : ℝ} :
  (∀ x : ℝ, ¬ (|x+2| + |x-1| < a)) ↔ a ≤ 3 :=
by
  sorry

end inequality_solution_empty_l160_160487


namespace solve_equation_l160_160133

theorem solve_equation (n : ℝ) :
  (3 - 2 * n) / (n + 2) + (3 * n - 9) / (3 - 2 * n) = 2 ↔ 
  n = (25 + Real.sqrt 13) / 18 ∨ n = (25 - Real.sqrt 13) / 18 :=
by
  sorry

end solve_equation_l160_160133


namespace construct_right_triangle_l160_160198

noncomputable def quadrilateral (A B C D : Type) : Prop :=
∃ (AB BC CA : ℝ), 
AB = BC ∧ BC = CA ∧ 
∃ (angle_D : ℝ), 
angle_D = 30

theorem construct_right_triangle (A B C D : Type) (angle_D: ℝ) (AB BC CA : ℝ) 
    (h1 : AB = BC) (h2 : BC = CA) (h3 : angle_D = 30) : 
    exists DA DB DC : ℝ, (DA * DA) + (DC * DC) = (AD * AD) :=
by sorry

end construct_right_triangle_l160_160198


namespace solution_set_l160_160923

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l160_160923


namespace hyperbola_other_asymptote_l160_160910

-- Define the problem conditions
def one_asymptote (x y : ℝ) : Prop := y = 2 * x
def foci_x_coordinate : ℝ := -4

-- Define the equation of the other asymptote
def other_asymptote (x y : ℝ) : Prop := y = -2 * x - 16

-- The statement to be proved
theorem hyperbola_other_asymptote : 
  (∀ x y, one_asymptote x y) → (∀ x, x = -4 → ∃ y, ∃ C, other_asymptote x y ∧ y = C + -2 * x - 8) :=
by
  sorry

end hyperbola_other_asymptote_l160_160910


namespace arithmetic_sequence_eighth_term_l160_160254

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l160_160254


namespace number_of_solutions_l160_160623

theorem number_of_solutions (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 5) :
  (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 → x = -2 :=
sorry

end number_of_solutions_l160_160623


namespace perpendicular_lines_l160_160361

theorem perpendicular_lines (a : ℝ) : 
  (a = -1 → (∀ x y : ℝ, 4 * x - (a + 1) * y + 9 = 0 → x ≠ 0 →  y ≠ 0 → 
  ∃ b : ℝ, (b^2 + 1) * x - b * y + 6 = 0)) ∧ 
  (∀ x y : ℝ, (4 * x - (a + 1) * y + 9 = 0) ∧ (∃ x y : ℝ, (a^2 - 1) * x - a * y + 6 = 0) → a ≠ -1) := 
sorry

end perpendicular_lines_l160_160361


namespace square_table_seats_4_pupils_l160_160061

-- Define the conditions given in the problem
def num_rectangular_tables := 7
def seats_per_rectangular_table := 10
def total_pupils := 90
def num_square_tables := 5

-- Define what we want to prove
theorem square_table_seats_4_pupils (x : ℕ) :
  total_pupils = num_rectangular_tables * seats_per_rectangular_table + num_square_tables * x →
  x = 4 :=
by
  sorry

end square_table_seats_4_pupils_l160_160061


namespace shaded_area_l160_160005

-- Definition for the conditions provided in the problem
def side_length := 6
def area_square := side_length ^ 2
def area_square_unit := area_square * 4

-- The problem and proof statement
theorem shaded_area (sl : ℕ) (asq : ℕ) (nsq : ℕ):
    sl = 6 ∧
    asq = sl ^ 2 ∧
    nsq = asq * 4 →
    nsq - (4 * (sl^2 / 2)) = 72 :=
by
  sorry

end shaded_area_l160_160005


namespace ratio_B_to_A_l160_160582

def work_together_rate : Real := 0.75
def days_for_A : Real := 4

theorem ratio_B_to_A : 
  ∃ (days_for_B : Real), 
    (1/days_for_A + 1/days_for_B = work_together_rate) → 
    (days_for_B / days_for_A = 0.5) :=
by 
  sorry

end ratio_B_to_A_l160_160582


namespace ages_correct_l160_160558

variables (Rehana_age Phoebe_age Jacob_age Xander_age : ℕ)

theorem ages_correct
  (h1 : Rehana_age = 25)
  (h2 : Rehana_age + 5 = 3 * (Phoebe_age + 5))
  (h3 : Jacob_age = 3 * Phoebe_age / 5)
  (h4 : Xander_age = Rehana_age + Jacob_age - 4) : 
  Rehana_age = 25 ∧ Phoebe_age = 5 ∧ Jacob_age = 3 ∧ Xander_age = 24 :=
by
  sorry

end ages_correct_l160_160558


namespace shaded_region_area_proof_l160_160902

/-- Define the geometric properties of the problem -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

noncomputable def shaded_region_area (rect : Rectangle) (circle1 circle2 : Circle) : ℝ :=
  let rect_area := rect.width * rect.height
  let circle_area := (Real.pi * circle1.radius ^ 2) + (Real.pi * circle2.radius ^ 2)
  rect_area - circle_area

theorem shaded_region_area_proof : shaded_region_area 
  {width := 10, height := 12} 
  {radius := 3, center := (0, 0)} 
  {radius := 3, center := (12, 10)} = 120 - 18 * Real.pi :=
by
  sorry

end shaded_region_area_proof_l160_160902


namespace cube_root_inequality_l160_160206

theorem cube_root_inequality (a b : ℝ) (h : a > b) : (a ^ (1/3)) > (b ^ (1/3)) :=
sorry

end cube_root_inequality_l160_160206


namespace part_i_part_ii_l160_160588

-- Define the variables and conditions
variable (a b : ℝ)
variable (h₁ : a > 0)
variable (h₂ : b > 0)
variable (h₃ : a + b = 1 / a + 1 / b)

-- Prove the first part: a + b ≥ 2
theorem part_i : a + b ≥ 2 := by
  sorry

-- Prove the second part: It is impossible for both a² + a < 2 and b² + b < 2 simultaneously
theorem part_ii : ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end part_i_part_ii_l160_160588


namespace percent_defective_units_shipped_for_sale_l160_160154

variable (total_units : ℕ)
variable (defective_units_percentage : ℝ := 0.08)
variable (shipped_defective_units_percentage : ℝ := 0.05)

theorem percent_defective_units_shipped_for_sale :
  defective_units_percentage * shipped_defective_units_percentage * 100 = 0.4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l160_160154


namespace find_x_l160_160591

theorem find_x (x y : ℝ) (h₁ : 2 * x - y = 14) (h₂ : y = 2) : x = 8 :=
by
  sorry

end find_x_l160_160591


namespace jade_and_julia_total_money_l160_160444

theorem jade_and_julia_total_money (x : ℕ) : 
  let jade_initial := 38 
  let julia_initial := jade_initial / 2 
  let jade_after := jade_initial + x 
  let julia_after := julia_initial + x 
  jade_after + julia_after = 57 + 2 * x := by
  sorry

end jade_and_julia_total_money_l160_160444


namespace circle_area_polar_eq_l160_160817

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l160_160817


namespace number_of_truthful_dwarfs_l160_160618

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l160_160618


namespace find_y_l160_160694

-- Define the points and slope conditions
def point_R : ℝ × ℝ := (-3, 4)
def x2 : ℝ := 5

-- Define the y coordinate and its corresponding condition
def y_condition (y : ℝ) : Prop := (y - 4) / (5 - (-3)) = 1 / 2

-- The main theorem stating the conditions and conclusion
theorem find_y (y : ℝ) (h : y_condition y) : y = 8 :=
by
  sorry

end find_y_l160_160694


namespace total_slices_at_picnic_l160_160497

def danny_watermelons : ℕ := 3
def danny_slices_per_watermelon : ℕ := 10
def sister_watermelons : ℕ := 1
def sister_slices_per_watermelon : ℕ := 15

def total_danny_slices : ℕ := danny_watermelons * danny_slices_per_watermelon
def total_sister_slices : ℕ := sister_watermelons * sister_slices_per_watermelon
def total_slices : ℕ := total_danny_slices + total_sister_slices

theorem total_slices_at_picnic : total_slices = 45 :=
by
  sorry

end total_slices_at_picnic_l160_160497


namespace theodore_total_monthly_earning_l160_160678

def total_earnings (stone_statues: Nat) (wooden_statues: Nat) (cost_stone: Nat) (cost_wood: Nat) (tax_rate: Rat) : Rat :=
  let pre_tax_earnings := stone_statues * cost_stone + wooden_statues * cost_wood
  let tax := tax_rate * pre_tax_earnings
  pre_tax_earnings - tax

theorem theodore_total_monthly_earning : total_earnings 10 20 20 5 0.10 = 270 :=
by
  sorry

end theodore_total_monthly_earning_l160_160678


namespace ratio_cube_sphere_surface_area_l160_160448

theorem ratio_cube_sphere_surface_area (R : ℝ) (h1 : R > 0) :
  let Scube := 24 * R^2
  let Ssphere := 4 * Real.pi * R^2
  (Scube / Ssphere) = (6 / Real.pi) :=
by
  sorry

end ratio_cube_sphere_surface_area_l160_160448


namespace problem_l160_160948

def pair_eq (a b c d : ℝ) : Prop := (a = c) ∧ (b = d)

def op_a (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, b * c - a * d)
def op_o (a b c d : ℝ) : ℝ × ℝ := (a + c, b + d)

theorem problem (x y : ℝ) :
  op_a 3 4 x y = (11, -2) →
  op_o 3 4 x y = (4, 6) :=
sorry

end problem_l160_160948


namespace find_smaller_number_l160_160220

theorem find_smaller_number (x y : ℕ) (h₁ : y - x = 1365) (h₂ : y = 6 * x + 15) : x = 270 :=
sorry

end find_smaller_number_l160_160220


namespace balls_in_boxes_l160_160975

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), (balls = 6) → (boxes = 3) → 
  (∃ ways : ℕ, ways = 7) :=
by
  sorry

end balls_in_boxes_l160_160975


namespace trig_equation_solution_l160_160816

open Real

theorem trig_equation_solution (x : ℝ) (k n : ℤ) :
  (sin (2 * x)) ^ 4 + (sin (2 * x)) ^ 3 * (cos (2 * x)) -
  8 * (sin (2 * x)) * (cos (2 * x)) ^ 3 - 8 * (cos (2 * x)) ^ 4 = 0 ↔
  (∃ k : ℤ, x = -π / 8 + (π * k) / 2) ∨ 
  (∃ n : ℤ, x = (1 / 2) * arctan 2 + (π * n) / 2) := sorry

end trig_equation_solution_l160_160816


namespace find_incorrect_option_l160_160085

-- The given conditions from the problem
def incomes : List ℝ := [2, 2.5, 2.5, 2.5, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 13]
def mean_incorrect : Prop := (incomes.sum / incomes.length) = 4
def option_incorrect : Prop := ¬ mean_incorrect

-- The goal is to prove that the statement about the mean being 4 is incorrect
theorem find_incorrect_option : option_incorrect := by
  sorry

end find_incorrect_option_l160_160085


namespace weight_difference_l160_160905

variable (W_A W_D : Nat)

theorem weight_difference : W_A - W_D = 15 :=
by
  -- Given conditions
  have h1 : W_A = 67 := sorry
  have h2 : W_D = 52 := sorry
  -- Proof
  sorry

end weight_difference_l160_160905


namespace does_not_pass_through_third_quadrant_l160_160142

noncomputable def f (a b x : ℝ) : ℝ := a^x + b - 1

theorem does_not_pass_through_third_quadrant (a b : ℝ) (h_a : 0 < a ∧ a < 1) (h_b : 0 < b ∧ b < 1) :
  ¬ ∃ x, f a b x < 0 ∧ x < 0 := sorry

end does_not_pass_through_third_quadrant_l160_160142


namespace r_investment_time_l160_160134

variables (P Q R Profit_p Profit_q Profit_r Tp Tq Tr : ℕ)
variables (h1 : P / Q = 7 / 5)
variables (h2 : Q / R = 5 / 4)
variables (h3 : Profit_p / Profit_q = 7 / 10)
variables (h4 : Profit_p / Profit_r = 7 / 8)
variables (h5 : Tp = 2)
variables (h6 : Tq = t)

theorem r_investment_time (t : ℕ) :
  ∃ Tr : ℕ, Tr = 4 :=
sorry

end r_investment_time_l160_160134


namespace reduced_price_per_dozen_bananas_l160_160074

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price_one_banana (P : ℝ) := 0.60 * P
noncomputable def number_bananas_original (P : ℝ) := 40 / P
noncomputable def number_bananas_reduced (P : ℝ) := 40 / (0.60 * P)
noncomputable def difference_bananas (P : ℝ) := (number_bananas_reduced P) - (number_bananas_original P)

theorem reduced_price_per_dozen_bananas 
  (P : ℝ) 
  (h1 : difference_bananas P = 67) 
  (h2 : P = 16 / 40.2) :
  12 * reduced_price_one_banana P = 2.856 :=
sorry

end reduced_price_per_dozen_bananas_l160_160074


namespace prime_divisors_6270_l160_160630

theorem prime_divisors_6270 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 11 ∧ p5 = 19 ∧ 
  (p1 * p2 * p3 * p4 * p5 = 6270) ∧ 
  (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧ 
  (∀ q, Nat.Prime q ∧ q ∣ 6270 → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4 ∨ q = p5)) := 
by 
  sorry

end prime_divisors_6270_l160_160630


namespace time_difference_180_div_vc_l160_160687

open Real

theorem time_difference_180_div_vc
  (V_A V_B V_C : ℝ)
  (h_ratio : V_A / V_C = 5 ∧ V_B / V_C = 4)
  (start_A start_B start_C : ℝ)
  (h_start_A : start_A = 100)
  (h_start_B : start_B = 80)
  (h_start_C : start_C = 0)
  (race_distance : ℝ)
  (h_race_distance : race_distance = 1200) :
  (race_distance - start_A) / V_A - race_distance / V_C = 180 / V_C := 
sorry

end time_difference_180_div_vc_l160_160687


namespace chess_group_players_l160_160988

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
by
  sorry

end chess_group_players_l160_160988


namespace range_y_over_x_l160_160290

theorem range_y_over_x {x y : ℝ} (h : (x-4)^2 + (y-2)^2 ≤ 4) : 
  ∃ k : ℝ, k = y / x ∧ 0 ≤ k ∧ k ≤ 4/3 :=
sorry

end range_y_over_x_l160_160290


namespace unique_intersection_point_l160_160871

theorem unique_intersection_point (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - 2 * (m + 1) * x - 1 = 0) → x = -1) ↔ m = -2 :=
by
  sorry

end unique_intersection_point_l160_160871


namespace car_arrives_before_bus_l160_160406

theorem car_arrives_before_bus
  (d : ℝ) (s_bus : ℝ) (s_car : ℝ) (v : ℝ)
  (h1 : d = 240)
  (h2 : s_bus = 40)
  (h3 : s_car = v)
  : 56 < v ∧ v < 120 := 
sorry

end car_arrives_before_bus_l160_160406


namespace calories_per_serving_l160_160799

theorem calories_per_serving (x : ℕ) (total_calories bread_calories servings : ℕ)
    (h1: total_calories = 500) (h2: bread_calories = 100) (h3: servings = 2)
    (h4: total_calories = bread_calories + (servings * x)) :
    x = 200 :=
by
  sorry

end calories_per_serving_l160_160799


namespace probability_A_correct_l160_160437

-- Definitions of probabilities
variable (P_A P_B : Prop)
variable (P_AB : Prop := P_A ∧ P_B)
variable (prob_AB : ℝ := 2 / 3)
variable (prob_B_given_A : ℝ := 8 / 9)

-- Lean statement of the mathematical problem
theorem probability_A_correct :
  (P_AB → P_A ∧ P_B) →
  (prob_AB = (2 / 3)) →
  (prob_B_given_A = (2 / 3) / prob_A) →
  (∃ prob_A : ℝ, prob_A = 3 / 4) :=
by
  sorry

end probability_A_correct_l160_160437


namespace inverse_of_f_at_10_l160_160533

noncomputable def f (x : ℝ) : ℝ := 1 + 3^(-x)

theorem inverse_of_f_at_10 :
  f⁻¹ 10 = -2 :=
sorry

end inverse_of_f_at_10_l160_160533


namespace a_plus_b_eq_neg1_l160_160137

theorem a_plus_b_eq_neg1 (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : a + b = -1 :=
by
  sorry

end a_plus_b_eq_neg1_l160_160137


namespace correct_system_of_equations_l160_160375

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : 3 * x = 5 * y - 6)
  (h2 : y = 2 * x - 10) : 
  (3 * x = 5 * y - 6) ∧ (y = 2 * x - 10) :=
by
  sorry

end correct_system_of_equations_l160_160375


namespace draw_contains_chinese_book_l160_160089

theorem draw_contains_chinese_book
  (total_books : ℕ)
  (chinese_books : ℕ)
  (math_books : ℕ)
  (drawn_books : ℕ)
  (h_total : total_books = 12)
  (h_chinese : chinese_books = 10)
  (h_math : math_books = 2)
  (h_drawn : drawn_books = 3) :
  ∃ n, n ≥ 1 ∧ n ≤ drawn_books ∧ n * (chinese_books/total_books) > 1 := 
  sorry

end draw_contains_chinese_book_l160_160089


namespace z_squared_in_second_quadrant_l160_160345
open Complex Real

noncomputable def z : ℂ := exp (π * I / 3)

theorem z_squared_in_second_quadrant : (z^2).re < 0 ∧ (z^2).im > 0 :=
by
  sorry

end z_squared_in_second_quadrant_l160_160345


namespace min_value_l160_160392

theorem min_value : ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (a = 1) → (b = 1) → (c = 1) →
  (∃ x, x = (a^2 + 4 * a + 2) / a ∧ x ≥ 6) ∧
  (∃ y, y = (b^2 + 4 * b + 2) / b ∧ y ≥ 6) ∧
  (∃ z, z = (c^2 + 4 * c + 2) / c ∧ z ≥ 6) →
  (∃ m, m = ((a^2 + 4 * a + 2) * (b^2 + 4 * b + 2) * (c^2 + 4 * c + 2)) / (a * b * c) ∧ m = 216) :=
by {
  sorry
}

end min_value_l160_160392


namespace max_lambda_leq_64_div_27_l160_160508

theorem max_lambda_leq_64_div_27 (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (1:ℝ) + (64:ℝ) / (27:ℝ) * (1 - a) * (1 - b) * (1 - c) ≤ Real.sqrt 3 / Real.sqrt (a + b + c) := 
sorry

end max_lambda_leq_64_div_27_l160_160508


namespace harry_morning_ratio_l160_160248

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end harry_morning_ratio_l160_160248


namespace dice_multiple_3_prob_l160_160783

-- Define the probability calculations for the problem
noncomputable def single_roll_multiple_3_prob: ℝ := 1 / 3
noncomputable def single_roll_not_multiple_3_prob: ℝ := 1 - single_roll_multiple_3_prob
noncomputable def eight_rolls_not_multiple_3_prob: ℝ := (single_roll_not_multiple_3_prob) ^ 8
noncomputable def at_least_one_roll_multiple_3_prob: ℝ := 1 - eight_rolls_not_multiple_3_prob

-- The lean theorem statement
theorem dice_multiple_3_prob : 
  at_least_one_roll_multiple_3_prob = 6305 / 6561 := by 
sorry

end dice_multiple_3_prob_l160_160783


namespace emma_missing_coins_l160_160831

theorem emma_missing_coins (x : ℤ) (h₁ : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  let missing := x - remaining
  missing / x = 1 / 9 :=
by
  sorry

end emma_missing_coins_l160_160831


namespace Eve_age_l160_160415

theorem Eve_age (Adam_age : ℕ) (Eve_age : ℕ) (h1 : Adam_age = 9) (h2 : Eve_age = Adam_age + 5)
  (h3 : ∃ k : ℕ, Eve_age + 1 = k * (Adam_age - 4)) : Eve_age = 14 :=
sorry

end Eve_age_l160_160415


namespace numbers_sum_prod_l160_160517

theorem numbers_sum_prod (x y : ℝ) (h_sum : x + y = 10) (h_prod : x * y = 24) : (x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4) :=
by
  sorry

end numbers_sum_prod_l160_160517


namespace inequality_k_ge_2_l160_160569

theorem inequality_k_ge_2 {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℤ) (h_k : k ≥ 2) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a)) ≥ 3 / 2 :=
by
  sorry

end inequality_k_ge_2_l160_160569


namespace journey_total_distance_l160_160320

/--
Given:
- A person covers 3/5 of their journey by train.
- A person covers 7/20 of their journey by bus.
- A person covers 3/10 of their journey by bicycle.
- A person covers 1/50 of their journey by taxi.
- The rest of the journey (4.25 km) is covered by walking.

Prove:
  D = 15.74 km
where D is the total distance of the journey.
-/
theorem journey_total_distance :
  ∀ (D : ℝ), 3/5 * D + 7/20 * D + 3/10 * D + 1/50 * D + 4.25 = D → D = 15.74 :=
by
  intro D
  sorry

end journey_total_distance_l160_160320


namespace product_not_zero_l160_160745

theorem product_not_zero (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) : (x - 2) * (x - 5) ≠ 0 := 
by 
  sorry

end product_not_zero_l160_160745


namespace f_f_of_2020_l160_160283

def f (x : ℕ) : ℕ :=
  if x ≤ 1 then 1
  else if 1 < x ∧ x ≤ 1837 then 2
  else if 1837 < x ∧ x < 2019 then 3
  else 2018

theorem f_f_of_2020 : f (f 2020) = 3 := by
  sorry

end f_f_of_2020_l160_160283


namespace hungarian_olympiad_problem_l160_160086

-- Define the function A_n as given in the problem
def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n - 1) + 1

-- State the theorem to be proved
theorem hungarian_olympiad_problem (n : ℕ) (h : 0 < n) : 8 ∣ A n :=
by
  sorry

end hungarian_olympiad_problem_l160_160086


namespace circle_equation_l160_160640

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := x + y + 2 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def line2 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def is_solution (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 6 * y - 16 = 0

-- Problem statement in Lean
theorem circle_equation : ∃ x y : ℝ, 
  (line1 x y ∧ circle1 x y ∧ line2 (x / 2) (x / 2)) → is_solution x y :=
sorry

end circle_equation_l160_160640


namespace smallest_possible_a_plus_b_l160_160810

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), (0 < a ∧ 0 < b) ∧ (2^10 * 7^3 = a^b) ∧ (a + b = 350753) :=
sorry

end smallest_possible_a_plus_b_l160_160810


namespace latest_time_for_temperature_at_60_l160_160574

theorem latest_time_for_temperature_at_60
  (t : ℝ) (h : -t^2 + 10 * t + 40 = 60) : t = 12 :=
sorry

end latest_time_for_temperature_at_60_l160_160574


namespace jackson_grade_increase_per_hour_l160_160458

-- Define the necessary variables
variables (v s p G : ℕ)

-- The conditions from the problem
def study_condition1 : v = 9 := sorry
def study_condition2 : s = v / 3 := sorry
def grade_starts_at_zero : G = s * p := sorry
def final_grade : G = 45 := sorry

-- The final problem statement to prove
theorem jackson_grade_increase_per_hour :
  p = 15 :=
by
  -- Add our sorry to indicate the partial proof
  sorry

end jackson_grade_increase_per_hour_l160_160458


namespace percentage_decrease_l160_160175

-- Define the condition given in the problem
def is_increase (pct : ℤ) : Prop := pct > 0
def is_decrease (pct : ℤ) : Prop := pct < 0

-- The main proof statement
theorem percentage_decrease (pct : ℤ) (h : pct = -10) : is_decrease pct :=
by
  sorry

end percentage_decrease_l160_160175


namespace arithmetic_sequence_sum_l160_160617

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m: ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Sum of the first n terms of a sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Specific statement we want to prove
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a)
  (h_S9 : sum_seq a 9 = 72) :
  a 2 + a 4 + a 9 = 24 :=
sorry

end arithmetic_sequence_sum_l160_160617


namespace triangle_area_sqrt2_div2_find_a_c_l160_160336

  -- Problem 1
  -- Prove the area of triangle ABC is sqrt(2)/2
  theorem triangle_area_sqrt2_div2 {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : Real.arcsin (1) = Real.pi / 2) : 
    (1 / 2) * 1 * Real.sqrt 2 = Real.sqrt 2 / 2 := sorry

  -- Problem 2
  -- Prove a = sqrt(7) and c = 2
  theorem find_a_c {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : (1 / 2) * a * Real.sin (Real.arcsin (Real.sqrt 3 / a)) = Real.sqrt 3 / 2) : 
    a = Real.sqrt 7 ∧ c = 2 := sorry

  
end triangle_area_sqrt2_div2_find_a_c_l160_160336


namespace find_b_l160_160039

theorem find_b
  (b : ℝ)
  (h1 : ∃ r : ℝ, 2 * r^2 + b * r - 65 = 0 ∧ r = 5)
  (h2 : 2 * 5^2 + b * 5 - 65 = 0) :
  b = 3 := by
  sorry

end find_b_l160_160039


namespace triangle_area_is_24_l160_160298

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end triangle_area_is_24_l160_160298


namespace zookeeper_feeding_ways_l160_160625

/-- We define the total number of ways the zookeeper can feed all the animals following the rules. -/
def feed_animal_ways : ℕ :=
  6 * 5^2 * 4^2 * 3^2 * 2^2 * 1^2

/-- Theorem statement: The number of ways to feed all the animals is 86400. -/
theorem zookeeper_feeding_ways : feed_animal_ways = 86400 :=
by
  sorry

end zookeeper_feeding_ways_l160_160625


namespace a_beats_b_time_difference_l160_160985

theorem a_beats_b_time_difference
  (d : ℝ) (d_A : ℝ) (d_B : ℝ)
  (t_A : ℝ)
  (h1 : d = 1000)
  (h2 : d_A = d)
  (h3 : d_B = d - 60)
  (h4 : t_A = 235) :
  (t_A - (d_B * t_A / d_A)) = 14.1 :=
by sorry

end a_beats_b_time_difference_l160_160985


namespace num_combinations_l160_160037

-- The conditions given in the problem.
def num_pencil_types : ℕ := 2
def num_eraser_types : ℕ := 3

-- The theorem to prove.
theorem num_combinations (pencils : ℕ) (erasers : ℕ) (h1 : pencils = num_pencil_types) (h2 : erasers = num_eraser_types) : pencils * erasers = 6 :=
by 
  have hp : pencils = 2 := h1
  have he : erasers = 3 := h2
  cases hp
  cases he
  rfl

end num_combinations_l160_160037


namespace fraction_of_plot_occupied_by_beds_l160_160911

-- Define the conditions based on plot area and number of beds
def plot_area : ℕ := 64
def total_beds : ℕ := 13
def outer_beds : ℕ := 12
def central_bed_area : ℕ := 4 * 4

-- The proof statement showing that fraction of the plot occupied by the beds is 15/32
theorem fraction_of_plot_occupied_by_beds : 
  (central_bed_area + (plot_area - central_bed_area)) / plot_area = 15 / 32 := 
sorry

end fraction_of_plot_occupied_by_beds_l160_160911


namespace batsman_total_score_eq_120_l160_160168

/-- A batsman's runs calculation including boundaries, sixes, and running between wickets. -/
def batsman_runs_calculation (T : ℝ) : Prop :=
  let runs_from_boundaries := 5 * 4
  let runs_from_sixes := 5 * 6
  let runs_from_total := runs_from_boundaries + runs_from_sixes
  let runs_from_running := 0.5833333333333334 * T
  T = runs_from_total + runs_from_running

theorem batsman_total_score_eq_120 :
  ∃ T : ℝ, batsman_runs_calculation T ∧ T = 120 :=
sorry

end batsman_total_score_eq_120_l160_160168


namespace statue_original_cost_l160_160616

theorem statue_original_cost (selling_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : selling_price = 620) (h2 : profit_percent = 25) : 
  original_cost = 496 :=
by
  have h3 : profit_percent / 100 + 1 = 1.25 := by sorry
  have h4 : 1.25 * original_cost = selling_price := by sorry
  have h5 : original_cost = 620 / 1.25 := by sorry
  have h6 : 620 / 1.25 = 496 := by sorry
  exact sorry

end statue_original_cost_l160_160616


namespace age_of_participant_who_left_l160_160704

theorem age_of_participant_who_left
  (avg_age_first_room : ℕ)
  (num_people_first_room : ℕ)
  (avg_age_second_room : ℕ)
  (num_people_second_room : ℕ)
  (increase_in_avg_age : ℕ)
  (total_num_people : ℕ)
  (final_avg_age : ℕ)
  (initial_avg_age : ℕ)
  (sum_ages : ℕ)
  (person_left : ℕ) :
  avg_age_first_room = 20 ∧ 
  num_people_first_room = 8 ∧
  avg_age_second_room = 45 ∧
  num_people_second_room = 12 ∧
  increase_in_avg_age = 1 ∧
  total_num_people = num_people_first_room + num_people_second_room ∧
  final_avg_age = initial_avg_age + increase_in_avg_age ∧
  initial_avg_age = (sum_ages) / total_num_people ∧
  sum_ages = (avg_age_first_room * num_people_first_room + avg_age_second_room * num_people_second_room) ∧
  19 * final_avg_age = sum_ages - person_left
  → person_left = 16 :=
by sorry

end age_of_participant_who_left_l160_160704


namespace condition_necessity_not_sufficiency_l160_160863

theorem condition_necessity_not_sufficiency (a : ℝ) : 
  (2 / a < 1 → a^2 > 4) ∧ ¬(2 / a < 1 ↔ a^2 > 4) :=
by {
  sorry
}

end condition_necessity_not_sufficiency_l160_160863


namespace parallel_condition_l160_160722

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x^2, 4 * x)

-- Define the condition for parallelism for two-dimensional vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Define the theorem to prove
theorem parallel_condition (x : ℝ) :
  parallel (vector_a x) (vector_b x) ↔ |x| = 2 :=
by {
  sorry
}

end parallel_condition_l160_160722


namespace room_length_calculation_l160_160268

-- Definitions of the problem conditions
def room_volume : ℝ := 10000
def room_width : ℝ := 10
def room_height : ℝ := 10

-- Statement to prove
theorem room_length_calculation : ∃ L : ℝ, L = room_volume / (room_width * room_height) ∧ L = 100 :=
by
  sorry

end room_length_calculation_l160_160268


namespace dividend_percentage_paid_by_company_l160_160741

-- Define the parameters
def faceValue : ℝ := 50
def investmentReturnPercentage : ℝ := 25
def investmentPerShare : ℝ := 37

-- Define the theorem
theorem dividend_percentage_paid_by_company :
  (investmentReturnPercentage / 100 * investmentPerShare / faceValue * 100) = 18.5 :=
by
  -- The proof is omitted
  sorry

end dividend_percentage_paid_by_company_l160_160741


namespace gumball_difference_l160_160608

theorem gumball_difference :
  ∀ C : ℕ, 19 ≤ (29 + C) / 3 ∧ (29 + C) / 3 ≤ 25 →
  (46 - 28) = 18 :=
by
  intros C h
  sorry

end gumball_difference_l160_160608


namespace ellipse_major_minor_axis_l160_160809

theorem ellipse_major_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) ∧
  (∃ a b : ℝ, a = 2 * b ∧ b^2 = 1 ∧ a^2 = 1/m) →
  m = 1/4 :=
by {
  sorry
}

end ellipse_major_minor_axis_l160_160809


namespace total_distance_biked_l160_160842

-- Definitions of the given conditions
def biking_time_to_park : ℕ := 15
def biking_time_return : ℕ := 25
def average_speed : ℚ := 6 -- miles per hour

-- Total biking time in minutes, then converted to hours
def total_biking_time_minutes : ℕ := biking_time_to_park + biking_time_return
def total_biking_time_hours : ℚ := total_biking_time_minutes / 60

-- Prove that the total distance biked is 4 miles
theorem total_distance_biked : total_biking_time_hours * average_speed = 4 := 
by
  -- proof will be here
  sorry

end total_distance_biked_l160_160842


namespace find_a_l160_160452

noncomputable def f (x : Real) (a : Real) : Real :=
if h : 0 < x ∧ x < 2 then (Real.log x - a * x) 
else 
if h' : -2 < x ∧ x < 0 then sorry
else 
   sorry

theorem find_a (a : Real) : (∀ x : Real, f x a = - f (-x) a) → (∀ x: Real, (0 < x ∧ x < 2) → f x a = Real.log x - a * x) → a > (1 / 2) → (∀ x: Real, (-2 < x ∧ x < 0) → f x a ≥ 1) → a = 1 := 
sorry

end find_a_l160_160452


namespace inequality_holds_for_all_real_l160_160560

theorem inequality_holds_for_all_real (a : ℝ) : a + a^3 - a^4 - a^6 < 1 :=
by
  sorry

end inequality_holds_for_all_real_l160_160560


namespace find_X_l160_160447

theorem find_X :
  let N := 90
  let X := (1 / 15) * N - (1 / 2 * 1 / 3 * 1 / 5 * N)
  X = 3 := by
  sorry

end find_X_l160_160447


namespace kickers_goals_in_first_period_l160_160770

theorem kickers_goals_in_first_period (K : ℕ) 
  (h1 : ∀ n : ℕ, n = K) 
  (h2 : ∀ n : ℕ, n = 2 * K) 
  (h3 : ∀ n : ℕ, n = K / 2) 
  (h4 : ∀ n : ℕ, n = 4 * K) 
  (h5 : K + 2 * K + (K / 2) + 4 * K = 15) : 
  K = 2 := 
by
  sorry

end kickers_goals_in_first_period_l160_160770


namespace range_of_a_l160_160946

-- Define the sets A, B, and C
def set_A (x : ℝ) : Prop := -3 < x ∧ x ≤ 2
def set_B (x : ℝ) : Prop := -1 < x ∧ x < 3
def set_A_int_B (x : ℝ) : Prop := -1 < x ∧ x ≤ 2
def set_C (x : ℝ) (a : ℝ) : Prop := a < x ∧ x < a + 1

-- The target theorem to prove
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, set_C x a → set_A_int_B x) → 
  (-1 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l160_160946


namespace single_burger_cost_l160_160096

theorem single_burger_cost
  (total_cost : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (cost_double_burger : ℝ)
  (remaining_cost : ℝ)
  (single_burgers : ℕ)
  (cost_single_burger : ℝ) :
  total_cost = 64.50 ∧
  total_hamburgers = 50 ∧
  double_burgers = 29 ∧
  cost_double_burger = 1.50 ∧
  remaining_cost = total_cost - (double_burgers * cost_double_burger) ∧
  single_burgers = total_hamburgers - double_burgers ∧
  cost_single_burger = remaining_cost / single_burgers →
  cost_single_burger = 1.00 :=
by
  sorry

end single_burger_cost_l160_160096


namespace three_right_angled_triangles_l160_160671

theorem three_right_angled_triangles 
  (a b c : ℕ)
  (h_area : 1/2 * (a * b) = 2 * (a + b + c))
  (h_pythagorean : a^2 + b^2 = c^2)
  (h_int_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (a = 9 ∧ b = 40 ∧ c = 41) ∨ 
  (a = 10 ∧ b = 24 ∧ c = 26) ∨ 
  (a = 12 ∧ b = 16 ∧ c = 20) := 
sorry

end three_right_angled_triangles_l160_160671


namespace common_difference_is_4_l160_160986

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable {a_4 a_5 S_6 : ℝ}
variable {d : ℝ}

-- Definitions of conditions given in the problem
def a4_cond : a_4 = a_n 4 := sorry
def a5_cond : a_5 = a_n 5 := sorry
def sum_six : S_6 = (6/2) * (2 * a_n 1 + 5 * d) := sorry
def term_sum : a_4 + a_5 = 24 := sorry

-- Proof statement
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l160_160986


namespace product_of_D_l160_160058

theorem product_of_D:
  ∀ (D : ℝ × ℝ), 
  (∃ M C : ℝ × ℝ, 
    M.1 = 4 ∧ M.2 = 3 ∧ 
    C.1 = 6 ∧ C.2 = -1 ∧ 
    M.1 = (C.1 + D.1) / 2 ∧ 
    M.2 = (C.2 + D.2) / 2) 
  → (D.1 * D.2 = 14) :=
sorry

end product_of_D_l160_160058


namespace range_of_a_l160_160369

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l160_160369


namespace total_word_count_is_5000_l160_160665

def introduction : ℕ := 450
def conclusion : ℕ := 3 * introduction
def body_sections : ℕ := 4 * 800

def total_word_count : ℕ := introduction + conclusion + body_sections

theorem total_word_count_is_5000 : total_word_count = 5000 := 
by
  -- Lean proof code will go here.
  sorry

end total_word_count_is_5000_l160_160665


namespace parallel_vectors_l160_160739

noncomputable def vector_a : (ℤ × ℤ) := (1, 3)
noncomputable def vector_b (m : ℤ) : (ℤ × ℤ) := (-2, m)

theorem parallel_vectors (m : ℤ) (h : vector_a = (1, 3) ∧ vector_b m = (-2, m))
  (hp: ∃ k : ℤ, ∀ (a1 a2 b1 b2 : ℤ), (a1, a2) = vector_a ∧ (b1, b2) = (1 + k * (-2), 3 + k * m)):
  m = -6 :=
by
  sorry

end parallel_vectors_l160_160739


namespace Chris_had_before_birthday_l160_160706

-- Define the given amounts
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Define the total birthday money received
def birthday_money : ℕ := grandmother_money + aunt_uncle_money + parents_money

-- Define the amount of money Chris had before his birthday
def money_before_birthday (total_now birthday_money : ℕ) : ℕ := total_now - birthday_money

-- Proposition to prove
theorem Chris_had_before_birthday : money_before_birthday total_money_now birthday_money = 159 := by
  sorry

end Chris_had_before_birthday_l160_160706


namespace max_balls_in_cube_l160_160684

theorem max_balls_in_cube 
  (radius : ℝ) (side_length : ℝ) 
  (ball_volume : ℝ := (4 / 3) * Real.pi * (radius^3)) 
  (cube_volume : ℝ := side_length^3) 
  (max_balls : ℝ := cube_volume / ball_volume) :
  radius = 3 ∧ side_length = 8 → Int.floor max_balls = 4 := 
by
  intro h
  rw [h.left, h.right]
  -- further proof would use numerical evaluation
  sorry

end max_balls_in_cube_l160_160684


namespace sara_picked_peaches_l160_160014

def peaches_original : ℕ := 24
def peaches_now : ℕ := 61
def peaches_picked (p_o p_n : ℕ) : ℕ := p_n - p_o

theorem sara_picked_peaches : peaches_picked peaches_original peaches_now = 37 :=
by
  sorry

end sara_picked_peaches_l160_160014


namespace minimum_value_function_equality_holds_at_two_thirds_l160_160148

noncomputable def f (x : ℝ) : ℝ := 4 / x + 1 / (1 - x)

theorem minimum_value_function (x : ℝ) (hx : 0 < x ∧ x < 1) : f x ≥ 9 := sorry

theorem equality_holds_at_two_thirds : f (2 / 3) = 9 := sorry

end minimum_value_function_equality_holds_at_two_thirds_l160_160148


namespace total_balloons_l160_160502

theorem total_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h₁ : joan_balloons = 40) (h₂ : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := 
by
  sorry

end total_balloons_l160_160502


namespace prob_axisymmetric_and_centrally_symmetric_l160_160551

theorem prob_axisymmetric_and_centrally_symmetric : 
  let card1 := "Line segment"
  let card2 := "Equilateral triangle"
  let card3 := "Parallelogram"
  let card4 := "Isosceles trapezoid"
  let card5 := "Circle"
  let cards := [card1, card2, card3, card4, card5]
  let symmetric_cards := [card1, card5]
  (symmetric_cards.length / cards.length : ℚ) = 2 / 5 :=
by sorry

end prob_axisymmetric_and_centrally_symmetric_l160_160551


namespace min_sum_areas_of_triangles_l160_160153

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1 / 4, 0)

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

def O := (0, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def on_opposite_sides_x_axis (p q : ℝ × ℝ) : Prop := p.2 * q.2 < 0

theorem min_sum_areas_of_triangles 
  (A B : ℝ × ℝ)
  (hA : parabola A)
  (hB : parabola B)
  (hAB : on_opposite_sides_x_axis A B)
  (h_dot : dot_product A B = 2) :
  ∃ m : ℝ, m = 3 := by
  sorry

end min_sum_areas_of_triangles_l160_160153


namespace train_length_l160_160847

noncomputable def length_of_train (time_sec : ℕ) (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000 / 3600) * time_sec

theorem train_length (h_time : 21 = 21) (h_speed : 75.6 = 75.6) :
  length_of_train 21 75.6 = 441 :=
by
  sorry

end train_length_l160_160847


namespace farmer_field_area_l160_160023

variable (x : ℕ) (A : ℕ)

def planned_days : Type := {x : ℕ // 120 * x = 85 * (x + 2) + 40}

theorem farmer_field_area (h : {x : ℕ // 120 * x = 85 * (x + 2) + 40}) : A = 720 :=
by
  sorry

end farmer_field_area_l160_160023


namespace half_percent_of_160_l160_160795

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end half_percent_of_160_l160_160795


namespace find_principal_l160_160512

theorem find_principal
  (R : ℝ) (hR : R = 0.05)
  (I : ℝ) (hI : I = 0.02)
  (A : ℝ) (hA : A = 1120)
  (n : ℕ) (hn : n = 6)
  (R' : ℝ) (hR' : R' = ((1 + R) / (1 + I)) - 1) :
  P = 938.14 :=
by
  have compound_interest_formula := A / (1 + R')^n
  sorry

end find_principal_l160_160512


namespace amount_after_two_years_l160_160897

def present_value : ℝ := 62000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

theorem amount_after_two_years:
  let amount_after_n_years (pv : ℝ) (r : ℝ) (n : ℕ) := pv * (1 + r)^n
  amount_after_n_years present_value rate_of_increase time_period = 78468.75 := 
  by 
    -- This is where your proof would go
    sorry

end amount_after_two_years_l160_160897


namespace cricketer_new_average_l160_160846

variable (A : ℕ) (runs_19th_inning : ℕ) (avg_increase : ℕ)
variable (total_runs_after_18 : ℕ)

theorem cricketer_new_average
  (h1 : runs_19th_inning = 98)
  (h2 : avg_increase = 4)
  (h3 : total_runs_after_18 = 18 * A)
  (h4 : 18 * A + 98 = 19 * (A + 4)) :
  A + 4 = 26 :=
by sorry

end cricketer_new_average_l160_160846


namespace anna_plants_needed_l160_160393

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end anna_plants_needed_l160_160393


namespace eval_expression_l160_160747

theorem eval_expression : (503 * 503 - 502 * 504) = 1 :=
by
  sorry

end eval_expression_l160_160747


namespace mean_of_sequence_l160_160972

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem mean_of_sequence :
  mean [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2] = 17.75 := by
sorry

end mean_of_sequence_l160_160972


namespace expected_value_of_fair_dodecahedral_die_l160_160286

theorem expected_value_of_fair_dodecahedral_die : 
  (1/12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end expected_value_of_fair_dodecahedral_die_l160_160286


namespace wicket_keeper_age_l160_160695

/-- The cricket team consists of 11 members with an average age of 22 years.
    One member is 25 years old, and the wicket keeper is W years old.
    Excluding the 25-year-old and the wicket keeper, the average age of the remaining players is 21 years.
    Prove that the wicket keeper is 6 years older than the average age of the team. -/
theorem wicket_keeper_age (W : ℕ) (team_avg_age : ℕ := 22) (total_team_members : ℕ := 11) 
                          (other_member_age : ℕ := 25) (remaining_avg_age : ℕ := 21) :
    W = 28 → W - team_avg_age = 6 :=
by
  intros
  sorry

end wicket_keeper_age_l160_160695


namespace water_required_l160_160104

-- Definitions based on the conditions
def balanced_equation : Prop := ∀ (NH4Cl H2O NH4OH HCl : ℕ), NH4Cl + H2O = NH4OH + HCl

-- New problem with the conditions translated into Lean
theorem water_required 
  (h_eq : balanced_equation)
  (n : ℕ)
  (m : ℕ)
  (mole_NH4Cl : n = 2 * m)
  (mole_H2O : m = 2) :
  n = m :=
by
  sorry

end water_required_l160_160104


namespace beadshop_profit_on_wednesday_l160_160598

theorem beadshop_profit_on_wednesday (total_profit profit_on_monday profit_on_tuesday profit_on_wednesday : ℝ)
  (h1 : total_profit = 1200)
  (h2 : profit_on_monday = total_profit / 3)
  (h3 : profit_on_tuesday = total_profit / 4)
  (h4 : profit_on_wednesday = total_profit - profit_on_monday - profit_on_tuesday) :
  profit_on_wednesday = 500 := 
sorry

end beadshop_profit_on_wednesday_l160_160598


namespace number_of_teams_l160_160651

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end number_of_teams_l160_160651


namespace largest_possible_expression_value_l160_160057

-- Definition of the conditions.
def distinct_digits (X Y Z : ℕ) : Prop := X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- The main theorem statement.
theorem largest_possible_expression_value : ∀ (X Y Z : ℕ), distinct_digits X Y Z → 
  (100 * X + 10 * Y + Z - 10 * Z - Y - X) ≤ 900 :=
by
  sorry

end largest_possible_expression_value_l160_160057


namespace induction_step_divisibility_l160_160135

theorem induction_step_divisibility {x y : ℤ} (k : ℕ) (h : ∀ n, n = 2*k - 1 → (x^n + y^n) % (x+y) = 0) :
  (x^(2*k+1) + y^(2*k+1)) % (x+y) = 0 :=
sorry

end induction_step_divisibility_l160_160135


namespace integer_solutions_count_l160_160524

theorem integer_solutions_count :
  (∃ (n : ℕ), ∀ (x y : ℤ), x^2 + y^2 = 6 * x + 2 * y + 15 → n = 12) :=
by
  sorry

end integer_solutions_count_l160_160524


namespace mul_exponent_result_l160_160884

theorem mul_exponent_result : 112 * (5^4) = 70000 := 
by 
  sorry

end mul_exponent_result_l160_160884


namespace PropA_neither_sufficient_nor_necessary_for_PropB_l160_160696

variable (a b : ℤ)

-- Proposition A
def PropA : Prop := a + b ≠ 4

-- Proposition B
def PropB : Prop := a ≠ 1 ∧ b ≠ 3

-- The required statement
theorem PropA_neither_sufficient_nor_necessary_for_PropB : ¬(PropA a b → PropB a b) ∧ ¬(PropB a b → PropA a b) :=
by
  sorry

end PropA_neither_sufficient_nor_necessary_for_PropB_l160_160696


namespace percentage_both_correct_l160_160307

theorem percentage_both_correct (p1 p2 pn : ℝ) (h1 : p1 = 0.85) (h2 : p2 = 0.80) (h3 : pn = 0.05) :
  ∃ x, x = 0.70 ∧ x = p1 + p2 - 1 + pn := by
  sorry

end percentage_both_correct_l160_160307


namespace three_digit_odd_sum_count_l160_160561

def countOddSumDigits : Nat :=
  -- Count of three-digit numbers with an odd sum formed by (1, 2, 3, 4, 5)
  24

theorem three_digit_odd_sum_count :
  -- Guarantees that the count of three-digit numbers meeting the criteria is 24
  ∃ n : Nat, n = countOddSumDigits :=
by
  use 24
  sorry

end three_digit_odd_sum_count_l160_160561


namespace tile_difference_correct_l160_160227

def initial_blue_tiles := 23
def initial_green_tiles := 16
def first_border_green_tiles := 6 * 1
def second_border_green_tiles := 6 * 2
def total_green_tiles := initial_green_tiles + first_border_green_tiles + second_border_green_tiles
def difference_tiling := total_green_tiles - initial_blue_tiles

theorem tile_difference_correct : difference_tiling = 11 := by
  sorry

end tile_difference_correct_l160_160227


namespace hcf_of_abc_l160_160213

-- Given conditions
variables (a b c : ℕ)
def lcm_abc := Nat.lcm (Nat.lcm a b) c
def product_abc := a * b * c

-- Statement to prove
theorem hcf_of_abc (H1 : lcm_abc a b c = 1200) (H2 : product_abc a b c = 108000) : 
  Nat.gcd (Nat.gcd a b) c = 90 :=
by
  sorry

end hcf_of_abc_l160_160213


namespace odd_prime_divides_seq_implies_power_of_two_divides_l160_160606

theorem odd_prime_divides_seq_implies_power_of_two_divides (a : ℕ → ℤ) (p n : ℕ)
  (h0 : a 0 = 2)
  (hk : ∀ k, a (k + 1) = 2 * (a k) ^ 2 - 1)
  (h_odd_prime : Nat.Prime p)
  (h_odd : p % 2 = 1)
  (h_divides : ↑p ∣ a n) :
  2^(n + 3) ∣ p^2 - 1 :=
sorry

end odd_prime_divides_seq_implies_power_of_two_divides_l160_160606


namespace intersection_M_N_l160_160445

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l160_160445


namespace number_of_speedster_convertibles_l160_160758

def proof_problem (T : ℕ) :=
  let Speedsters := 2 * T / 3
  let NonSpeedsters := 50
  let TotalInventory := NonSpeedsters * 3
  let SpeedsterConvertibles := 4 * Speedsters / 5
  (Speedsters = 2 * TotalInventory / 3) ∧ (SpeedsterConvertibles = 4 * Speedsters / 5)

theorem number_of_speedster_convertibles : proof_problem 150 → ∃ (x : ℕ), x = 80 :=
by
  -- Provide the definition of Speedsters, NonSpeedsters, TotalInventory, and SpeedsterConvertibles
  sorry

end number_of_speedster_convertibles_l160_160758


namespace real_seq_proof_l160_160854

noncomputable def real_seq_ineq (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1

theorem real_seq_proof (a : ℕ → ℝ) (h : real_seq_ineq a) :
  ∀ k m : ℕ, k > 0 → m > 0 → |a k / k - a m / m| < 1 / k + 1 / m :=
by
  sorry

end real_seq_proof_l160_160854


namespace snow_leopards_arrangement_l160_160808

theorem snow_leopards_arrangement :
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  end_positions * factorial_six = 1440 :=
by
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  show end_positions * factorial_six = 1440
  sorry

end snow_leopards_arrangement_l160_160808


namespace snail_reaches_tree_l160_160762

theorem snail_reaches_tree
  (l1 l2 s : ℝ) 
  (h_l1 : l1 = 4) 
  (h_l2 : l2 = 3) 
  (h_s : s = 40) : 
  ∃ n : ℕ, n = 37 ∧ s - n*(l1 - l2) ≤ l1 :=
  by
    sorry

end snail_reaches_tree_l160_160762


namespace three_digit_factorions_l160_160552

def is_factorion (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  n = Nat.factorial a + Nat.factorial b + Nat.factorial c

theorem three_digit_factorions : ∀ n : ℕ, (100 ≤ n ∧ n < 1000) → is_factorion n → n = 145 :=
by
  sorry

end three_digit_factorions_l160_160552


namespace group_population_l160_160782

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l160_160782


namespace number_is_divisible_by_divisor_l160_160273

-- Defining the number after replacing y with 3
def number : ℕ := 7386038

-- Defining the divisor which we need to prove 
def divisor : ℕ := 7

-- Stating the property that 7386038 is divisible by 7
theorem number_is_divisible_by_divisor : number % divisor = 0 := by
  sorry

end number_is_divisible_by_divisor_l160_160273


namespace Lexie_age_proof_l160_160144

variables (L B S : ℕ)

def condition1 : Prop := L = B + 6
def condition2 : Prop := S = 2 * L
def condition3 : Prop := S - B = 14

theorem Lexie_age_proof (h1 : condition1 L B) (h2 : condition2 S L) (h3 : condition3 S B) : L = 8 :=
by
  sorry

end Lexie_age_proof_l160_160144


namespace FO_gt_DI_l160_160224

-- Definitions and conditions
variables (F I D O : Type) [MetricSpace F] [MetricSpace I] [MetricSpace D] [MetricSpace O]
variables (FI DO DI FO : ℝ) (angle_FIO angle_DIO : ℝ)
variable (convex_FIDO : ConvexQuadrilateral F I D O)

-- Conditions
axiom FI_DO_equal : FI = DO
axiom FI_DO_gt_DI : FI > DI
axiom angles_equal : angle_FIO = angle_DIO

-- Goal
theorem FO_gt_DI : FO > DI :=
sorry

end FO_gt_DI_l160_160224


namespace find_x_for_parallel_vectors_l160_160607

theorem find_x_for_parallel_vectors :
  ∀ (x : ℚ), (∃ a b : ℚ × ℚ, a = (2 * x, 3) ∧ b = (1, 9) ∧ (∃ k : ℚ, (2 * x, 3) = (k * 1, k * 9))) ↔ x = 1 / 6 :=
by 
  sorry

end find_x_for_parallel_vectors_l160_160607


namespace existential_inequality_false_iff_l160_160919

theorem existential_inequality_false_iff {a : ℝ} :
  (∀ x : ℝ, x^2 + a * x - 2 * a ≥ 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
by
  sorry

end existential_inequality_false_iff_l160_160919


namespace max_k_inequality_l160_160008

theorem max_k_inequality (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  ∀ k ≤ 2, ( ( (b - c) ^ 2 * (b + c) / a ) + 
             ( (c - a) ^ 2 * (c + a) / b ) + 
             ( (a - b) ^ 2 * (a + b) / c ) 
             ≥ k * ( a^2 + b^2 + c^2 - a*b - b*c - c*a ) ) :=
by
  sorry

end max_k_inequality_l160_160008


namespace goods_train_cross_platform_time_l160_160396

noncomputable def time_to_cross_platform (speed_kmph : ℝ) (length_train : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_mps : ℝ := speed_kmph * (1000 / 3600)
  let total_distance : ℝ := length_train + length_platform
  total_distance / speed_mps

theorem goods_train_cross_platform_time :
  time_to_cross_platform 72 290.04 230 = 26.002 :=
by
  -- The proof is omitted
  sorry

end goods_train_cross_platform_time_l160_160396


namespace yanni_money_left_in_cents_l160_160751

-- Define the constants based on the conditions
def initial_amount := 0.85
def mother_amount := 0.40
def found_amount := 0.50
def toy_cost := 1.60

-- Function to calculate the total amount
def total_amount := initial_amount + mother_amount + found_amount

-- Function to calculate the money left
def money_left := total_amount - toy_cost

-- Convert the remaining money from dollars to cents
def money_left_in_cents := money_left * 100

-- The theorem to prove
theorem yanni_money_left_in_cents : money_left_in_cents = 15 := by
  -- placeholder for proof, sorry used to skip the proof
  sorry

end yanni_money_left_in_cents_l160_160751


namespace positive_integer_sixk_l160_160425

theorem positive_integer_sixk (n : ℕ) :
  (∃ d1 d2 d3 : ℕ, d1 < d2 ∧ d2 < d3 ∧ d1 + d2 + d3 = n ∧ d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n) ↔ (∃ k : ℕ, n = 6 * k) :=
by
  sorry

end positive_integer_sixk_l160_160425


namespace least_number_divisible_by_11_l160_160685

theorem least_number_divisible_by_11 (n : ℕ) (k : ℕ) (h₁ : n = 2520 * k + 1) (h₂ : 11 ∣ n) : n = 12601 :=
sorry

end least_number_divisible_by_11_l160_160685


namespace division_of_powers_l160_160515

variable {a : ℝ}

theorem division_of_powers (ha : a ≠ 0) : a^5 / a^3 = a^2 :=
by sorry

end division_of_powers_l160_160515


namespace find_number_l160_160241

theorem find_number (x : ℝ) : 
  (3 * x / 5 - 220) * 4 + 40 = 360 → x = 500 :=
by
  intro h
  sorry

end find_number_l160_160241


namespace problem_conditions_l160_160281

open Real

variable {m n : ℝ}

theorem problem_conditions (h1 : 0 < m) (h2 : 0 < n) (h3 : m + n = 2 * m * n) :
  (min (m + n) = 2) ∧ (min (sqrt (m * n)) = 1) ∧
  (min ((n^2) / m + (m^2) / n) = 2) ∧ 
  (max ((sqrt m + sqrt n) / sqrt (m * n)) = 2) :=
by sorry

end problem_conditions_l160_160281


namespace problem_l160_160757

noncomputable def h (p x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15

noncomputable def k (q r x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

theorem problem
  (p q r : ℝ)
  (h_has_distinct_roots: ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ h p a = 0 ∧ h p b = 0 ∧ h p c = 0)
  (h_roots_are_k_roots: ∀ x, h p x = 0 → k q r x = 0) :
  k q r 1 = -3322.25 :=
sorry

end problem_l160_160757


namespace rotate_A_180_about_B_l160_160664

-- Define the points A, B, and C
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (-1, 1)

-- Define the 180 degrees rotation about B
def rotate_180_about (p q : ℝ × ℝ) : ℝ × ℝ :=
  let translated_p := (p.1 - q.1, p.2 - q.2) 
  let rotated_p := (-translated_p.1, -translated_p.2)
  (rotated_p.1 + q.1, rotated_p.2 + q.2)

-- Prove the image of point A after a 180 degrees rotation about point B
theorem rotate_A_180_about_B : rotate_180_about A B = (2, 7) :=
by
  sorry

end rotate_A_180_about_B_l160_160664


namespace estimate_pi_l160_160041

theorem estimate_pi :
  ∀ (r : ℝ) (side_length : ℝ) (total_beans : ℕ) (beans_in_circle : ℕ),
  r = 1 →
  side_length = 2 →
  total_beans = 80 →
  beans_in_circle = 64 →
  (π = 3.2) :=
by
  intros r side_length total_beans beans_in_circle hr hside htotal hin_circle
  sorry

end estimate_pi_l160_160041


namespace original_price_of_shoes_l160_160916

theorem original_price_of_shoes (
  initial_amount : ℝ := 74
) (sweater_cost : ℝ := 9) (tshirt_cost : ℝ := 11) 
  (final_amount_after_refund : ℝ := 51)
  (refund_percentage : ℝ := 0.90)
  (S : ℝ) :
  (initial_amount - sweater_cost - tshirt_cost - S + refund_percentage * S = final_amount_after_refund) -> 
  S = 30 := 
by
  intros h
  sorry

end original_price_of_shoes_l160_160916


namespace avg_speed_correct_l160_160872

noncomputable def avg_speed_round_trip
  (flight_up_speed : ℝ)
  (tailwind_speed : ℝ)
  (tailwind_angle : ℝ)
  (flight_home_speed : ℝ)
  (headwind_speed : ℝ)
  (headwind_angle : ℝ) : ℝ :=
  let effective_tailwind_speed := tailwind_speed * Real.cos (tailwind_angle * Real.pi / 180)
  let ground_speed_to_mother := flight_up_speed + effective_tailwind_speed
  let effective_headwind_speed := headwind_speed * Real.cos (headwind_angle * Real.pi / 180)
  let ground_speed_back_home := flight_home_speed - effective_headwind_speed
  (ground_speed_to_mother + ground_speed_back_home) / 2

theorem avg_speed_correct :
  avg_speed_round_trip 96 12 30 88 15 60 = 93.446 :=
by
  sorry

end avg_speed_correct_l160_160872


namespace number_of_possible_orders_l160_160861

-- Define the total number of bowlers participating in the playoff
def num_bowlers : ℕ := 6

-- Define the number of games
def num_games : ℕ := 5

-- Define the number of possible outcomes per game
def outcomes_per_game : ℕ := 2

-- Prove the total number of possible orders for bowlers to receive prizes
theorem number_of_possible_orders : (outcomes_per_game ^ num_games) = 32 :=
by sorry

end number_of_possible_orders_l160_160861


namespace quadratic_solutions_l160_160367

-- Define the equation x^2 - 6x + 8 = 0
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Lean statement for the equivalence of solutions
theorem quadratic_solutions : ∀ x : ℝ, quadratic_eq x ↔ x = 2 ∨ x = 4 :=
by
  intro x
  dsimp [quadratic_eq]
  sorry

end quadratic_solutions_l160_160367


namespace typesetter_times_l160_160171

theorem typesetter_times (α β γ : ℝ) (h1 : 1 / β - 1 / α = 10)
                                        (h2 : 1 / β - 1 / γ = 6)
                                        (h3 : 9 * (α + β) = 10 * (β + γ)) :
    α = 1 / 20 ∧ β = 1 / 30 ∧ γ = 1 / 24 :=
by {
  sorry
}

end typesetter_times_l160_160171


namespace monotonic_increasing_interval_l160_160272

noncomputable def log_base_1_div_3 (t : ℝ) := Real.log t / Real.log (1/3)

def quadratic (x : ℝ) := 4 + 3 * x - x^2

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), (∀ x, a < x ∧ x < b → (log_base_1_div_3 (quadratic x)) < (log_base_1_div_3 (quadratic (x + ε))) ∧
               ((-1 : ℝ) < x ∧ x < 4) ∧ (quadratic x > 0)) ↔ (a, b) = (3 / 2, 4) :=
by
  sorry

end monotonic_increasing_interval_l160_160272


namespace max_xy_l160_160479

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 6) : xy ≤ 3 / 2 := sorry

end max_xy_l160_160479


namespace red_blood_cells_surface_area_l160_160870

-- Define the body surface area of an adult
def body_surface_area : ℝ := 1800

-- Define the multiplying factor for the surface areas of red blood cells
def multiplier : ℝ := 2000

-- Define the sum of the surface areas of all red blood cells
def sum_surface_area : ℝ := multiplier * body_surface_area

-- Define the expected sum in scientific notation
def expected_sum : ℝ := 3.6 * 10^6

-- The theorem that needs to be proved
theorem red_blood_cells_surface_area :
  sum_surface_area = expected_sum :=
by
  sorry

end red_blood_cells_surface_area_l160_160870


namespace cos_squared_diff_tan_l160_160705

theorem cos_squared_diff_tan (α : ℝ) (h : Real.tan α = 3) :
  Real.cos (α + π/4) ^ 2 - Real.cos (α - π/4) ^ 2 = -3 / 5 :=
by
  sorry

end cos_squared_diff_tan_l160_160705


namespace inequality_solution_set_l160_160663

theorem inequality_solution_set (x : ℝ) : (x-1)/(x+2) > 1 → x < -2 := sorry

end inequality_solution_set_l160_160663


namespace caochong_weighing_equation_l160_160688

-- Definitions for porter weight, stone weight, and the counts in the respective steps
def porter_weight : ℝ := 120
def stone_weight (x : ℝ) : ℝ := x
def first_step_weight (x : ℝ) : ℝ := 20 * stone_weight x + 3 * porter_weight
def second_step_weight (x : ℝ) : ℝ := (20 + 1) * stone_weight x + 1 * porter_weight

-- Theorem stating the equality condition ensuring the same water level
theorem caochong_weighing_equation (x : ℝ) :
  first_step_weight x = second_step_weight x :=
by
  sorry

end caochong_weighing_equation_l160_160688


namespace gcd_max_digits_l160_160952

theorem gcd_max_digits (a b : ℕ) (h_a : a < 10^7) (h_b : b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) : Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_max_digits_l160_160952


namespace find_t_l160_160653

theorem find_t :
  ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 ∧ t = 44 :=
by
  sorry

end find_t_l160_160653


namespace sqrt_5th_of_x_sqrt_4th_x_l160_160413

theorem sqrt_5th_of_x_sqrt_4th_x (x : ℝ) (hx : 0 < x) : Real.sqrt (x * Real.sqrt (x ^ (1 / 4))) = x ^ (1 / 4) :=
by
  sorry

end sqrt_5th_of_x_sqrt_4th_x_l160_160413


namespace player_A_advantage_l160_160316

theorem player_A_advantage (B A : ℤ) (rolls : ℕ) (h : rolls = 36) 
  (game_conditions : ∀ (x : ℕ), (x % 2 = 1 → A = A + x ∧ B = B - x) ∧ 
                      (x % 2 = 0 ∧ x ≠ 2 → A = A - x ∧ B = B + x) ∧ 
                      (x = 2 → A = A ∧ B = B)) : 
  (36 * (1 / 18 : ℚ) = 2) :=
by {
  -- Mathematical proof will be filled here
  sorry
}

end player_A_advantage_l160_160316


namespace total_amount_correct_l160_160006

noncomputable def total_amount : ℝ :=
  let nissin_noodles := 24 * 1.80 * 0.80
  let master_kong_tea := 6 * 1.70 * 0.80
  let shanlin_soup := 5 * 3.40
  let shuanghui_sausage := 3 * 11.20 * 0.90
  nissin_noodles + master_kong_tea + shanlin_soup + shuanghui_sausage

theorem total_amount_correct : total_amount = 89.96 := by
  sorry

end total_amount_correct_l160_160006


namespace cheaper_store_price_difference_in_cents_l160_160662

theorem cheaper_store_price_difference_in_cents :
  let list_price : ℝ := 59.99
  let discount_budget_buys := list_price * 0.15
  let discount_frugal_finds : ℝ := 20
  let sale_price_budget_buys := list_price - discount_budget_buys
  let sale_price_frugal_finds := list_price - discount_frugal_finds
  let difference_in_price := sale_price_budget_buys - sale_price_frugal_finds
  let difference_in_cents := difference_in_price * 100
  difference_in_cents = 1099.15 :=
by
  sorry

end cheaper_store_price_difference_in_cents_l160_160662


namespace eval_expression_l160_160038

theorem eval_expression (x y : ℕ) (h_x : x = 2001) (h_y : y = 2002) :
  (x^3 - 3*x^2*y + 5*x*y^2 - y^3 - 2) / (x * y) = 1999 :=
  sorry

end eval_expression_l160_160038


namespace evaluate_expression_l160_160114

theorem evaluate_expression (a b c : ℝ) (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 :=
by 
  sorry

end evaluate_expression_l160_160114


namespace negation_of_p_l160_160675

-- Define the proposition p
def p : Prop := ∃ x : ℝ, x + 2 ≤ 0

-- Define the negation of p
def not_p : Prop := ∀ x : ℝ, x + 2 > 0

-- State the theorem that the negation of p is not_p
theorem negation_of_p : ¬ p = not_p := by 
  sorry -- Proof not provided

end negation_of_p_l160_160675


namespace no_integer_solutions_l160_160828

theorem no_integer_solutions (m n : ℤ) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2011) :=
by sorry

end no_integer_solutions_l160_160828


namespace geometric_sequence_a4_l160_160438

theorem geometric_sequence_a4 (x a_4 : ℝ) (h1 : 2*x + 2 = (3*x + 3) * (2*x + 2) / x)
  (h2 : x = -4 ∨ x = -1) (h3 : x = -4) : a_4 = -27 / 2 :=
by
  sorry

end geometric_sequence_a4_l160_160438


namespace num_valid_sequences_10_transformations_l160_160818

/-- Define the transformations: 
    L: 90° counterclockwise rotation,
    R: 90° clockwise rotation,
    H: reflection across the x-axis,
    V: reflection across the y-axis. -/
inductive Transformation
| L | R | H | V

/-- Define a function to get the number of valid sequences of transformations
    that bring the vertices E, F, G, H back to their original positions.-/
def countValidSequences : ℕ :=
  56

/-- The theorem to prove that the number of valid sequences
    of 10 transformations resulting in the identity transformation is 56. -/
theorem num_valid_sequences_10_transformations : 
  countValidSequences = 56 :=
sorry

end num_valid_sequences_10_transformations_l160_160818


namespace value_of_a7_minus_a8_l160_160235

variable {a : ℕ → ℤ} (d a₁ : ℤ)

-- Definition that this is an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

-- Given condition
def condition (a : ℕ → ℤ) : Prop :=
  a 2 + a 6 + a 8 + a 10 = 80

-- The goal to prove
theorem value_of_a7_minus_a8 (a : ℕ → ℤ) (h_arith : is_arithmetic_seq a a₁ d)
  (h_cond : condition a) : a 7 - a 8 = 8 :=
sorry

end value_of_a7_minus_a8_l160_160235


namespace final_expression_simplified_l160_160765

variable (b : ℝ)

theorem final_expression_simplified :
  ((3 * b + 6 - 5 * b) / 3) = (-2 / 3) * b + 2 := by
  sorry

end final_expression_simplified_l160_160765


namespace social_logistics_turnover_scientific_notation_l160_160960

noncomputable def total_social_logistics_turnover_2022 : ℝ := 347.6 * (10 ^ 12)

theorem social_logistics_turnover_scientific_notation :
  total_social_logistics_turnover_2022 = 3.476 * (10 ^ 14) :=
by
  sorry

end social_logistics_turnover_scientific_notation_l160_160960


namespace binary_to_decimal_1010101_l160_160980

def bin_to_dec (bin : List ℕ) (len : ℕ): ℕ :=
  List.foldl (λ acc (digit, idx) => acc + digit * 2^idx) 0 (List.zip bin (List.range len))

theorem binary_to_decimal_1010101 : bin_to_dec [1, 0, 1, 0, 1, 0, 1] 7 = 85 :=
by
  simp [bin_to_dec, List.range, List.zip]
  -- Detailed computation can be omitted and sorry used here if necessary
  sorry

end binary_to_decimal_1010101_l160_160980


namespace factorization_identity_l160_160516

theorem factorization_identity (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
by
  sorry

end factorization_identity_l160_160516


namespace true_discount_double_time_l160_160786

theorem true_discount_double_time (PV FV1 FV2 I1 I2 TD1 TD2 : ℕ) 
  (h1 : FV1 = 110)
  (h2 : TD1 = 10)
  (h3 : FV1 - TD1 = PV)
  (h4 : I1 = FV1 - PV)
  (h5 : FV2 = PV + 2 * I1)
  (h6 : TD2 = FV2 - PV) :
  TD2 = 20 := by
  sorry

end true_discount_double_time_l160_160786


namespace train_stop_time_l160_160214

theorem train_stop_time
  (D : ℝ)
  (h1 : D > 0)
  (T_no_stop : ℝ := D / 300)
  (T_with_stop : ℝ := D / 200)
  (T_stop : ℝ := T_with_stop - T_no_stop):
  T_stop = 6 / 60 := by
    sorry

end train_stop_time_l160_160214


namespace circle_represents_valid_a_l160_160194

theorem circle_represents_valid_a (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2 * a * x - 4 * y + 5 * a = 0) → (a > 4 ∨ a < 1) :=
by
  sorry

end circle_represents_valid_a_l160_160194


namespace mod_11_residue_l160_160906

theorem mod_11_residue :
  (312 ≡ 4 [MOD 11]) ∧
  (47 ≡ 3 [MOD 11]) ∧
  (154 ≡ 0 [MOD 11]) ∧
  (22 ≡ 0 [MOD 11]) →
  (312 + 6 * 47 + 8 * 154 + 5 * 22 ≡ 0 [MOD 11]) :=
by
  intros h
  sorry

end mod_11_residue_l160_160906


namespace luke_fish_fillets_l160_160359

theorem luke_fish_fillets (daily_fish : ℕ) (days : ℕ) (fillets_per_fish : ℕ) 
  (h1 : daily_fish = 2) (h2 : days = 30) (h3 : fillets_per_fish = 2) : 
  daily_fish * days * fillets_per_fish = 120 := 
by 
  sorry

end luke_fish_fillets_l160_160359


namespace x_intercept_correct_l160_160540

noncomputable def x_intercept_of_line : ℝ × ℝ :=
if h : (-4 : ℝ) ≠ 0 then (24 / (-4), 0) else (0, 0)

theorem x_intercept_correct : x_intercept_of_line = (-6, 0) := by
  -- proof will be given here
  sorry

end x_intercept_correct_l160_160540


namespace polygon_sides_l160_160995

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end polygon_sides_l160_160995


namespace shirt_selling_price_l160_160426

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end shirt_selling_price_l160_160426


namespace ellipse_transform_circle_l160_160908

theorem ellipse_transform_circle (a b x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b)
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (y' : ℝ)
  (h_transform : y' = (a / b) * y) :
  x^2 + y'^2 = a^2 :=
by
  sorry

end ellipse_transform_circle_l160_160908


namespace kenny_cost_per_book_l160_160501

theorem kenny_cost_per_book (B : ℕ) :
  let lawn_charge := 15
  let mowed_lawns := 35
  let video_game_cost := 45
  let video_games := 5
  let total_earnings := lawn_charge * mowed_lawns
  let spent_on_video_games := video_game_cost * video_games
  let remaining_money := total_earnings - spent_on_video_games
  remaining_money / B = 300 / B :=
by
  sorry

end kenny_cost_per_book_l160_160501


namespace find_all_pairs_l160_160140

def is_solution (m n : ℕ) : Prop := 200 * m + 6 * n = 2006

def valid_pairs : List (ℕ × ℕ) := [(1, 301), (4, 201), (7, 101), (10, 1)]

theorem find_all_pairs :
  ∀ (m n : ℕ), is_solution m n ↔ (m, n) ∈ valid_pairs := by sorry

end find_all_pairs_l160_160140


namespace maria_walk_to_school_l160_160549

variable (w s : ℝ)

theorem maria_walk_to_school (h1 : 25 * w + 13 * s = 38) (h2 : 11 * w + 20 * s = 31) : 
  51 = 51 := by
  sorry

end maria_walk_to_school_l160_160549


namespace proof_problem_l160_160329

-- Definitions
def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Conditions
def condition_A : Prop := is_factor 4 24
def condition_B : Prop := is_divisor 19 152 ∧ ¬ is_divisor 19 96
def condition_E : Prop := is_factor 6 180

-- Proof problem statement
theorem proof_problem : condition_A ∧ condition_B ∧ condition_E :=
by sorry

end proof_problem_l160_160329


namespace length_of_bridge_l160_160913

theorem length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) (length_m : ℝ) :
  speed_kmh = 5 → time_min = 15 → length_m = 1250 :=
by
  sorry

end length_of_bridge_l160_160913


namespace trajectory_is_straight_line_l160_160597

theorem trajectory_is_straight_line (x y : ℝ) (h : x + y = 0) : ∃ m b : ℝ, y = m * x + b :=
by
  use -1
  use 0
  sorry

end trajectory_is_straight_line_l160_160597


namespace ratio_of_areas_l160_160812

theorem ratio_of_areas (R_A R_B : ℝ) 
  (h1 : (1 / 6) * 2 * Real.pi * R_A = (1 / 9) * 2 * Real.pi * R_B) :
  (Real.pi * R_A^2) / (Real.pi * R_B^2) = (4 : ℝ) / 9 :=
by 
  sorry

end ratio_of_areas_l160_160812


namespace brad_start_time_after_maxwell_l160_160029

-- Assuming time is measured in hours, distance in kilometers, and speed in km/h
def meet_time (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) : ℕ :=
  let d_m := t_m * v_m
  let t_b := t_m - 1
  let d_b := t_b * v_b
  d_m + d_b

theorem brad_start_time_after_maxwell (d : ℕ) (v_m : ℕ) (v_b : ℕ) (t_m : ℕ) :
  d = 54 → v_m = 4 → v_b = 6 → t_m = 6 → 
  meet_time d v_m v_b t_m = 54 :=
by
  intros hd hv_m hv_b ht_m
  have : meet_time d v_m v_b t_m = t_m * v_m + (t_m - 1) * v_b := rfl
  rw [hd, hv_m, hv_b, ht_m] at this
  sorry

end brad_start_time_after_maxwell_l160_160029


namespace calculate_total_money_l160_160627

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l160_160627


namespace fraction_zero_l160_160141

theorem fraction_zero (x : ℝ) (h : x ≠ 1) (h₁ : (x + 1) / (x - 1) = 0) : x = -1 :=
sorry

end fraction_zero_l160_160141


namespace siblings_pizza_order_l160_160330

theorem siblings_pizza_order :
  let Alex := 1 / 6
  let Beth := 2 / 5
  let Cyril := 1 / 3
  let Dan := 1 - (Alex + Beth + Cyril)
  Dan > Alex ∧ Alex > Cyril ∧ Cyril > Beth := sorry

end siblings_pizza_order_l160_160330


namespace necessary_not_sufficient_condition_l160_160187

theorem necessary_not_sufficient_condition {a : ℝ} :
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) →
  (¬ (∀ x : ℝ, x ≥ a → |x - 1| < 1)) →
  a ≤ 0 :=
by
  intro h1 h2
  sorry

end necessary_not_sufficient_condition_l160_160187


namespace stock_price_end_second_year_l160_160279

theorem stock_price_end_second_year
  (P₀ : ℝ) (r₁ r₂ : ℝ) 
  (h₀ : P₀ = 150)
  (h₁ : r₁ = 0.80)
  (h₂ : r₂ = 0.30) :
  let P₁ := P₀ + r₁ * P₀
  let P₂ := P₁ - r₂ * P₁
  P₂ = 189 :=
by
  sorry

end stock_price_end_second_year_l160_160279


namespace initial_girls_is_11_l160_160682

-- Definitions of initial parameters and transformations
def initially_girls_percent : ℝ := 0.35
def final_girls_percent : ℝ := 0.25
def three : ℝ := 3

-- 35% of the initial total is girls
def initially_girls (p : ℝ) : ℝ := initially_girls_percent * p
-- After three girls leave and three boys join, the count of girls
def final_girls (p : ℝ) : ℝ := initially_girls p - three

-- Using the condition that after the change, 25% are girls
def proof_problem : Prop := ∀ (p : ℝ), 
  (final_girls p) / p = final_girls_percent →
  (0.1 * p) = 3 → 
  initially_girls p = 11

-- The statement of the theorem to be proved in Lean 4
theorem initial_girls_is_11 : proof_problem := sorry

end initial_girls_is_11_l160_160682


namespace greatest_whole_number_solution_l160_160159

theorem greatest_whole_number_solution :
  ∃ (x : ℕ), (5 * x - 4 < 3 - 2 * x) ∧ ∀ (y : ℕ), (5 * y - 4 < 3 - 2 * y) → y ≤ x ∧ x = 0 :=
by
  sorry

end greatest_whole_number_solution_l160_160159


namespace largest_is_B_l160_160801

noncomputable def A : ℚ := ((2023:ℚ) / 2022) + ((2023:ℚ) / 2024)
noncomputable def B : ℚ := ((2024:ℚ) / 2023) + ((2026:ℚ) / 2023)
noncomputable def C : ℚ := ((2025:ℚ) / 2024) + ((2025:ℚ) / 2026)

theorem largest_is_B : B > A ∧ B > C := by
  sorry

end largest_is_B_l160_160801


namespace trajectory_midpoint_eq_C2_length_CD_l160_160398

theorem trajectory_midpoint_eq_C2 {x y x' y' : ℝ} :
  (x' - 0)^2 + (y' - 4)^2 = 16 →
  x = (x' + 4) / 2 →
  y = y' / 2 →
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  sorry

theorem length_CD {x y Cx Cy Dx Dy : ℝ} :
  ((x - 2)^2 + (y - 2)^2 = 4) →
  (x^2 + (y - 4)^2 = 16) →
  ((Cx - Dx)^2 + (Cy - Dy)^2 = 14) :=
by
  sorry

end trajectory_midpoint_eq_C2_length_CD_l160_160398


namespace bob_and_jim_total_skips_l160_160771

-- Definitions based on conditions
def bob_skips_per_rock : Nat := 12
def jim_skips_per_rock : Nat := 15
def rocks_skipped_by_each : Nat := 10

-- Total skips calculation based on the given conditions
def bob_total_skips : Nat := bob_skips_per_rock * rocks_skipped_by_each
def jim_total_skips : Nat := jim_skips_per_rock * rocks_skipped_by_each
def total_skips : Nat := bob_total_skips + jim_total_skips

-- Theorem statement
theorem bob_and_jim_total_skips : total_skips = 270 := by
  sorry

end bob_and_jim_total_skips_l160_160771


namespace surface_area_of_prism_l160_160594

theorem surface_area_of_prism (l w h : ℕ)
  (h_internal_volume : l * w * h = 24)
  (h_external_volume : (l + 2) * (w + 2) * (h + 2) = 120) :
  2 * ((l + 2) * (w + 2) + (w + 2) * (h + 2) + (h + 2) * (l + 2)) = 148 :=
by
  sorry

end surface_area_of_prism_l160_160594


namespace friends_payment_l160_160788

theorem friends_payment
  (num_friends : ℕ) (num_bread : ℕ) (cost_bread : ℕ) 
  (num_hotteok : ℕ) (cost_hotteok : ℕ) (total_cost : ℕ)
  (cost_per_person : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_bread = 5)
  (h3 : cost_bread = 200)
  (h4 : num_hotteok = 7)
  (h5 : cost_hotteok = 800)
  (h6 : total_cost = num_bread * cost_bread + num_hotteok * cost_hotteok)
  (h7 : cost_per_person = total_cost / num_friends) :
  cost_per_person = 1650 := by
  sorry

end friends_payment_l160_160788


namespace problem_statement_l160_160050

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem problem_statement :
  (M ∩ N) = N :=
by
  sorry

end problem_statement_l160_160050


namespace arithmetic_sequence_max_sum_l160_160898

noncomputable def max_S_n (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  n * a 1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum :
  ∃ d, ∃ a : ℕ → ℝ, 
  (a 1 = 1) ∧ (3 * (a 1 + 7 * d) = 5 * (a 1 + 12 * d)) ∧ 
  (∀ n, max_S_n n a d ≤ max_S_n 20 a d) := 
sorry

end arithmetic_sequence_max_sum_l160_160898


namespace cost_of_bench_eq_150_l160_160051

theorem cost_of_bench_eq_150 (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
sorry

end cost_of_bench_eq_150_l160_160051


namespace cookies_in_each_batch_l160_160609

theorem cookies_in_each_batch (batches : ℕ) (people : ℕ) (consumption_per_person : ℕ) (cookies_per_dozen : ℕ) 
  (total_batches : batches = 4) 
  (total_people : people = 16) 
  (cookies_per_person : consumption_per_person = 6) 
  (dozen_size : cookies_per_dozen = 12) :
  (6 * 16) / 4 / 12 = 2 := 
by {
  sorry
}

end cookies_in_each_batch_l160_160609


namespace cost_of_fencing_each_side_l160_160648

theorem cost_of_fencing_each_side (total_cost : ℕ) (x : ℕ) (h : total_cost = 276) (hx : 4 * x = total_cost) : x = 69 :=
by {
  sorry
}

end cost_of_fencing_each_side_l160_160648


namespace sister_granola_bars_l160_160325

-- Definitions based on conditions
def total_bars := 20
def chocolate_chip_bars := 8
def oat_honey_bars := 6
def peanut_butter_bars := 6

def greg_set_aside_chocolate := 3
def greg_set_aside_oat_honey := 2
def greg_set_aside_peanut_butter := 2

def final_chocolate_chip := chocolate_chip_bars - greg_set_aside_chocolate - 2  -- 2 traded away
def final_oat_honey := oat_honey_bars - greg_set_aside_oat_honey - 4           -- 4 traded away
def final_peanut_butter := peanut_butter_bars - greg_set_aside_peanut_butter

-- Final distribution to sisters
def older_sister_chocolate := 2.5 -- 2 whole bars + 1/2 bar
def younger_sister_peanut := 2.5  -- 2 whole bars + 1/2 bar

theorem sister_granola_bars :
  older_sister_chocolate = 2.5 ∧ younger_sister_peanut = 2.5 :=
by
  sorry

end sister_granola_bars_l160_160325


namespace negation_of_existence_statement_l160_160857

theorem negation_of_existence_statement :
  (¬ (∃ x : ℝ, x^2 + x + 1 < 0)) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l160_160857


namespace distinct_units_digits_of_cube_l160_160777

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l160_160777


namespace purchasing_plan_exists_l160_160500

-- Define the structure for our purchasing plan
structure PurchasingPlan where
  n3 : ℕ
  n6 : ℕ
  n9 : ℕ
  n12 : ℕ
  n15 : ℕ
  n19 : ℕ
  n21 : ℕ
  n30 : ℕ

-- Define the length function to sum up the total length of the purchasing plan
def length (p : PurchasingPlan) : ℕ :=
  3 * p.n3 + 6 * p.n6 + 9 * p.n9 + 12 * p.n12 + 15 * p.n15 + 19 * p.n19 + 21 * p.n21 + 30 * p.n30

-- Define the purchasing options
def options : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

-- Define the requirement
def requiredLength : ℕ := 50

-- State the theorem that there exists a purchasing plan that sums up to the required length
theorem purchasing_plan_exists : ∃ p : PurchasingPlan, length p = requiredLength :=
  sorry

end purchasing_plan_exists_l160_160500


namespace expressions_equal_l160_160080

variable (a b c : ℝ)

theorem expressions_equal (h : a + 2 * b + 2 * c = 0) : a + 2 * b * c = (a + 2 * b) * (a + 2 * c) := 
by 
  sorry

end expressions_equal_l160_160080


namespace digimon_pack_price_l160_160595

-- Defining the given conditions as Lean variables
variables (total_spent baseball_cost : ℝ)
variables (packs_of_digimon : ℕ)

-- Setting given values from the problem
def keith_total_spent : total_spent = 23.86 := sorry
def baseball_deck_cost : baseball_cost = 6.06 := sorry
def number_of_digimon_packs : packs_of_digimon = 4 := sorry

-- Stating the main theorem/problem to prove
theorem digimon_pack_price 
  (h1 : total_spent = 23.86)
  (h2 : baseball_cost = 6.06)
  (h3 : packs_of_digimon = 4) : 
  ∃ (price_per_pack : ℝ), price_per_pack = 4.45 :=
sorry

end digimon_pack_price_l160_160595


namespace max_non_managers_l160_160593

theorem max_non_managers (n_mngrs n_non_mngrs : ℕ) (hmngrs : n_mngrs = 8) 
                (h_ratio : (5 : ℚ) / 24 < (n_mngrs : ℚ) / n_non_mngrs) :
                n_non_mngrs ≤ 38 :=
by {
  sorry
}

end max_non_managers_l160_160593


namespace inequality_solution_reciprocal_inequality_l160_160781

-- Proof Problem (1)
theorem inequality_solution (x : ℝ) : |x-1| + (1/2)*|x-3| < 2 ↔ (1 < x ∧ x < 3) :=
sorry

-- Proof Problem (2)
theorem reciprocal_inequality (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 2) : 
  (1/a) + (1/b) + (1/c) ≥ 9/2 :=
sorry

end inequality_solution_reciprocal_inequality_l160_160781


namespace quadratic_reciprocal_sum_l160_160107

theorem quadratic_reciprocal_sum :
  ∃ (x1 x2 : ℝ), (x1^2 - 5 * x1 + 4 = 0) ∧ (x2^2 - 5 * x2 + 4 = 0) ∧ (x1 ≠ x2) ∧ (x1 + x2 = 5) ∧ (x1 * x2 = 4) ∧ (1 / x1 + 1 / x2 = 5 / 4) :=
sorry

end quadratic_reciprocal_sum_l160_160107


namespace problem1_problem2_l160_160891

-- Problem 1
theorem problem1 : 2 * Real.cos (30 * Real.pi / 180) - Real.tan (60 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) = 1 / 2 :=
by sorry

-- Problem 2
theorem problem2 : (-1) ^ 2023 + 2 * Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) + Real.sin (60 * Real.pi / 180) + (Real.tan (60 * Real.pi / 180)) ^ 2 = 2 + Real.sqrt 2 :=
by sorry

end problem1_problem2_l160_160891


namespace graph_does_not_pass_through_fourth_quadrant_l160_160835

def linear_function (x : ℝ) : ℝ := x + 1

theorem graph_does_not_pass_through_fourth_quadrant : 
  ¬ ∃ x : ℝ, x > 0 ∧ linear_function x < 0 :=
sorry

end graph_does_not_pass_through_fourth_quadrant_l160_160835


namespace time_descend_hill_l160_160420

-- Definitions
def time_to_top : ℝ := 4
def avg_speed_whole_journey : ℝ := 3
def avg_speed_uphill : ℝ := 2.25

-- Theorem statement
theorem time_descend_hill (t : ℝ) 
  (h1 : time_to_top = 4) 
  (h2 : avg_speed_whole_journey = 3) 
  (h3 : avg_speed_uphill = 2.25) : 
  t = 2 := 
sorry

end time_descend_hill_l160_160420


namespace cost_per_box_of_cookies_l160_160849

-- Given conditions
def initial_money : ℝ := 20
def mother_gift : ℝ := 2 * initial_money
def total_money : ℝ := initial_money + mother_gift
def cupcake_price : ℝ := 1.50
def num_cupcakes : ℝ := 10
def cost_cupcakes : ℝ := num_cupcakes * cupcake_price
def money_after_cupcakes : ℝ := total_money - cost_cupcakes
def remaining_money : ℝ := 30
def num_boxes_cookies : ℝ := 5
def money_spent_on_cookies : ℝ := money_after_cupcakes - remaining_money

-- Theorem: Calculate the cost per box of cookies
theorem cost_per_box_of_cookies : (money_spent_on_cookies / num_boxes_cookies) = 3 :=
by
  sorry

end cost_per_box_of_cookies_l160_160849


namespace price_of_turbans_l160_160018

theorem price_of_turbans : 
  ∀ (salary_A salary_B salary_C : ℝ) (months_A months_B months_C : ℕ) (payment_A payment_B payment_C : ℝ)
    (prorated_salary_A prorated_salary_B prorated_salary_C : ℝ),
  salary_A = 120 → 
  salary_B = 150 → 
  salary_C = 180 → 
  months_A = 8 → 
  months_B = 7 → 
  months_C = 10 → 
  payment_A = 80 → 
  payment_B = 87.50 → 
  payment_C = 150 → 
  prorated_salary_A = (salary_A * (months_A / 12 : ℝ)) → 
  prorated_salary_B = (salary_B * (months_B / 12 : ℝ)) → 
  prorated_salary_C = (salary_C * (months_C / 12 : ℝ)) → 
  ∃ (price_A price_B price_C : ℝ),
  price_A = payment_A - prorated_salary_A ∧ 
  price_B = payment_B - prorated_salary_B ∧ 
  price_C = payment_C - prorated_salary_C ∧ 
  price_A = 0 ∧ price_B = 0 ∧ price_C = 0 := 
by
  sorry

end price_of_turbans_l160_160018


namespace no_integer_solutions_l160_160209

theorem no_integer_solutions (x y : ℤ) : 15 * x^2 - 7 * y^2 ≠ 9 :=
by
  sorry

end no_integer_solutions_l160_160209


namespace largest_coins_l160_160355

theorem largest_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n = 146 :=
by
  sorry

end largest_coins_l160_160355


namespace max_f1_l160_160338

-- Define the function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b 

-- Define the condition 
def condition (a : ℝ) (b : ℝ) : Prop := f 0 a b = 4

-- State the theorem
theorem max_f1 (a b: ℝ) (h: condition a b) : 
  ∃ b_max, b_max = 1 ∧ ∀ b, f 1 a b ≤ 7 := 
sorry

end max_f1_l160_160338


namespace production_statistics_relation_l160_160481

noncomputable def a : ℚ := (10 + 12 + 14 + 14 + 15 + 15 + 16 + 17 + 17 + 17) / 10
noncomputable def b : ℚ := (15 + 15) / 2
noncomputable def c : ℤ := 17

theorem production_statistics_relation : c > a ∧ a > b :=
by
  sorry

end production_statistics_relation_l160_160481


namespace probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l160_160509

noncomputable def qualification_rate : ℝ := 0.8
def probability_both_qualified (rate : ℝ) : ℝ := rate * rate
def unqualified_rate (rate : ℝ) : ℝ := 1 - rate
def expected_days (n : ℕ) (p : ℝ) : ℝ := n * p

theorem probability_of_both_qualified_bottles : 
  probability_both_qualified qualification_rate = 0.64 :=
by sorry

theorem expected_number_of_days_with_unqualified_milk :
  expected_days 3 (unqualified_rate qualification_rate) = 1.08 :=
by sorry

end probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l160_160509


namespace find_y_l160_160313

/-- 
  Given: The sum of angles around a point is 360 degrees, 
  and those angles are: 6y, 3y, 4y, and 2y.
  Prove: y = 24 
-/ 
theorem find_y (y : ℕ) (h : 6 * y + 3 * y + 4 * y + 2 * y = 360) : y = 24 :=
sorry

end find_y_l160_160313


namespace solution_set_correct_l160_160638

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3)^2 > 0

theorem solution_set_correct : 
  ∀ x : ℝ, inequality_solution x ↔ (x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 3) := 
by sorry

end solution_set_correct_l160_160638


namespace base7_first_digit_l160_160994

noncomputable def first_base7_digit : ℕ := 625

theorem base7_first_digit (n : ℕ) (h : n = 625) : ∃ (d : ℕ), d = 12 ∧ (d * 49 ≤ n) ∧ (n < (d + 1) * 49) :=
by
  sorry

end base7_first_digit_l160_160994


namespace absolute_value_inequality_l160_160719

variable (a b c d : ℝ)

theorem absolute_value_inequality (h₁ : a + b + c + d > 0) (h₂ : a > c) (h₃ : b > d) : 
  |a + b| > |c + d| := sorry

end absolute_value_inequality_l160_160719


namespace sam_money_left_l160_160309

/- Definitions -/

def initial_dimes : ℕ := 38
def initial_quarters : ℕ := 12
def initial_nickels : ℕ := 25
def initial_pennies : ℕ := 30

def price_per_candy_bar_dimes : ℕ := 4
def price_per_candy_bar_nickels : ℕ := 2
def candy_bars_bought : ℕ := 5

def price_per_lollipop_nickels : ℕ := 6
def price_per_lollipop_pennies : ℕ := 10
def lollipops_bought : ℕ := 2

def price_per_bag_of_chips_quarters : ℕ := 1
def price_per_bag_of_chips_dimes : ℕ := 3
def price_per_bag_of_chips_pennies : ℕ := 5
def bags_of_chips_bought : ℕ := 3

/- Proof problem statement -/

theorem sam_money_left : 
  (initial_dimes * 10 + initial_quarters * 25 + initial_nickels * 5 + initial_pennies * 1) - 
  (
    candy_bars_bought * (price_per_candy_bar_dimes * 10 + price_per_candy_bar_nickels * 5) + 
    lollipops_bought * (price_per_lollipop_nickels * 5 + price_per_lollipop_pennies * 1) +
    bags_of_chips_bought * (price_per_bag_of_chips_quarters * 25 + price_per_bag_of_chips_dimes * 10 + price_per_bag_of_chips_pennies * 1)
  ) = 325 := 
sorry

end sam_money_left_l160_160309


namespace walter_bus_time_l160_160850

/--
Walter wakes up at 6:30 a.m., leaves for the bus at 7:30 a.m., attends 7 classes that each last 45 minutes,
enjoys a 40-minute lunch, and spends 2.5 hours of additional time at school for activities.
He takes the bus home and arrives at 4:30 p.m.
Prove that Walter spends 35 minutes on the bus.
-/
theorem walter_bus_time : 
  let total_time_away := 9 * 60 -- in minutes
  let class_time := 7 * 45 -- in minutes
  let lunch_time := 40 -- in minutes
  let additional_school_time := 2.5 * 60 -- in minutes
  total_time_away - (class_time + lunch_time + additional_school_time) = 35 := 
by
  sorry

end walter_bus_time_l160_160850


namespace fill_in_the_blanks_correctly_l160_160210

def remote_areas_need : String := "what the remote areas need"
def children : String := "children"
def education : String := "education"
def good_textbooks : String := "good textbooks"

-- Defining the grammatical agreement condition
def subject_verb_agreement (s : String) (v : String) : Prop :=
  (s = remote_areas_need ∧ v = "is") ∨ (s = children ∧ v = "are")

-- The main theorem statement
theorem fill_in_the_blanks_correctly : 
  subject_verb_agreement remote_areas_need "is" ∧ subject_verb_agreement children "are" :=
sorry

end fill_in_the_blanks_correctly_l160_160210


namespace trees_per_square_meter_l160_160613

-- Definitions of the given conditions
def side_length : ℕ := 100
def total_trees : ℕ := 120000

def area_of_street : ℤ := side_length * side_length
def area_of_forest : ℤ := 3 * area_of_street

-- The question translated to Lean theorem statement
theorem trees_per_square_meter (h1: area_of_street = side_length * side_length)
    (h2: area_of_forest = 3 * area_of_street) 
    (h3: total_trees = 120000) : 
    (total_trees / area_of_forest) = 4 :=
sorry

end trees_per_square_meter_l160_160613


namespace determinant_of_A_l160_160284

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![8, 5, -4], ![3, 3, 7]]  -- Defining matrix A

def A' : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![5, 4, -2], ![0, 2, 9]]  -- Defining matrix A' after row operations

theorem determinant_of_A' : Matrix.det A' = 55 := by -- Proving that the determinant of A' is 55
  sorry

end determinant_of_A_l160_160284


namespace b_sequence_is_constant_l160_160954

noncomputable def b_sequence_formula (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → ∃ d q : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ (∀ n : ℕ, b (n + 1) = b n * q)) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = b n) ∧
  (∀ n : ℕ, n > 0 → b n = 1)

theorem b_sequence_is_constant (a b : ℕ → ℝ) (h : b_sequence_formula a b) : ∀ n : ℕ, n > 0 → b n = 1 :=
  by
    sorry

end b_sequence_is_constant_l160_160954


namespace total_books_on_shelves_l160_160730

theorem total_books_on_shelves (shelves books_per_shelf : ℕ) (h_shelves : shelves = 350) (h_books_per_shelf : books_per_shelf = 25) :
  shelves * books_per_shelf = 8750 :=
by {
  sorry
}

end total_books_on_shelves_l160_160730


namespace turtles_remaining_on_log_l160_160539

-- Define the initial conditions
def original_turtles := 9
def additional_turtles := (3 * original_turtles) - 2
def total_group := original_turtles + additional_turtles
def frightened_turtles := total_group / 2

-- Theorem statement
theorem turtles_remaining_on_log : total_group - frightened_turtles = 17 :=
by
  sorry

end turtles_remaining_on_log_l160_160539


namespace first_five_terms_series_l160_160170

theorem first_five_terms_series (a : ℕ → ℚ) (h : ∀ n, a n = 1 / (n * (n + 1))) :
  (a 1 = 1 / 2) ∧
  (a 2 = 1 / 6) ∧
  (a 3 = 1 / 12) ∧
  (a 4 = 1 / 20) ∧
  (a 5 = 1 / 30) :=
by
  sorry

end first_five_terms_series_l160_160170


namespace no_nonzero_integer_solution_l160_160855

theorem no_nonzero_integer_solution 
(a b c n : ℤ) (h : 6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * n ^ 2) : 
a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
sorry

end no_nonzero_integer_solution_l160_160855


namespace parabola_hyperbola_focus_vertex_l160_160981

theorem parabola_hyperbola_focus_vertex (p : ℝ) : 
  (∃ (focus_vertex : ℝ × ℝ), focus_vertex = (2, 0) 
    ∧ focus_vertex = (p / 2, 0)) → p = 4 :=
by
  sorry

end parabola_hyperbola_focus_vertex_l160_160981


namespace negation_of_cube_of_every_odd_number_is_odd_l160_160996

theorem negation_of_cube_of_every_odd_number_is_odd:
  ¬ (∀ n : ℤ, (n % 2 = 1 → (n^3 % 2 = 1))) ↔ ∃ n : ℤ, (n % 2 = 1 ∧ ¬ (n^3 % 2 = 1)) := 
by
  sorry

end negation_of_cube_of_every_odd_number_is_odd_l160_160996


namespace natural_numbers_condition_l160_160044

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_numbers_condition (n : ℕ) (p1 p2 : ℕ)
  (hp1_prime : is_prime p1) (hp2_prime : is_prime p2)
  (hn : n = p1 ^ 2) (hn72 : n + 72 = p2 ^ 2) :
  n = 49 ∨ n = 289 :=
  sorry

end natural_numbers_condition_l160_160044


namespace number_of_people_liking_at_least_one_activity_l160_160094

def total_people := 200
def people_like_books := 80
def people_like_songs := 60
def people_like_movies := 30
def people_like_books_and_songs := 25
def people_like_books_and_movies := 15
def people_like_songs_and_movies := 20
def people_like_all_three := 10

theorem number_of_people_liking_at_least_one_activity :
  total_people = 200 →
  people_like_books = 80 →
  people_like_songs = 60 →
  people_like_movies = 30 →
  people_like_books_and_songs = 25 →
  people_like_books_and_movies = 15 →
  people_like_songs_and_movies = 20 →
  people_like_all_three = 10 →
  (people_like_books + people_like_songs + people_like_movies -
   people_like_books_and_songs - people_like_books_and_movies -
   people_like_songs_and_movies + people_like_all_three) = 120 := sorry

end number_of_people_liking_at_least_one_activity_l160_160094


namespace bicycle_saves_time_l160_160670

-- Define the conditions
def time_to_walk : ℕ := 98
def time_saved_by_bicycle : ℕ := 34

-- Prove the question equals the answer
theorem bicycle_saves_time :
  time_saved_by_bicycle = 34 := 
by
  sorry

end bicycle_saves_time_l160_160670


namespace total_cats_l160_160526

-- Define the conditions as constants
def asleep_cats : ℕ := 92
def awake_cats : ℕ := 6

-- State the theorem that proves the total number of cats
theorem total_cats : asleep_cats + awake_cats = 98 := 
by
  -- Proof omitted
  sorry

end total_cats_l160_160526


namespace sarah_correct_answer_percentage_l160_160693

theorem sarah_correct_answer_percentage
  (q1 q2 q3 : ℕ)   -- Number of questions in the first, second, and third tests.
  (p1 p2 p3 : ℕ → ℝ)   -- Percentages of questions Sarah got right in the first, second, and third tests.
  (m : ℕ)   -- Number of calculation mistakes:
  (h_q1 : q1 = 30) (h_q2 : q2 = 20) (h_q3 : q3 = 50)
  (h_p1 : p1 q1 = 0.85) (h_p2 : p2 q2 = 0.75) (h_p3 : p3 q3 = 0.90)
  (h_m : m = 3) :
  ∃ pct_correct : ℝ, pct_correct = 83 :=
by
  sorry

end sarah_correct_answer_percentage_l160_160693


namespace total_batteries_correct_l160_160340

-- Definitions of the number of batteries used in each category
def batteries_flashlight : ℕ := 2
def batteries_toys : ℕ := 15
def batteries_controllers : ℕ := 2

-- The total number of batteries used by Tom
def total_batteries : ℕ := batteries_flashlight + batteries_toys + batteries_controllers

-- The proof statement that needs to be proven
theorem total_batteries_correct : total_batteries = 19 := by
  sorry

end total_batteries_correct_l160_160340


namespace blake_lollipops_count_l160_160113

theorem blake_lollipops_count (lollipop_cost : ℕ) (choc_cost_per_pack : ℕ) 
  (chocolate_packs : ℕ) (total_paid : ℕ) (change_received : ℕ) 
  (total_spent : ℕ) (total_choc_cost : ℕ) (remaining_amount : ℕ) 
  (lollipop_count : ℕ) : 
  lollipop_cost = 2 →
  choc_cost_per_pack = 4 * lollipop_cost →
  chocolate_packs = 6 →
  total_paid = 6 * 10 →
  change_received = 4 →
  total_spent = total_paid - change_received →
  total_choc_cost = chocolate_packs * choc_cost_per_pack →
  remaining_amount = total_spent - total_choc_cost →
  lollipop_count = remaining_amount / lollipop_cost →
  lollipop_count = 4 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end blake_lollipops_count_l160_160113


namespace arithmetic_sequence_property_l160_160353

-- Define arithmetic sequence and given condition
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Lean 4 statement
theorem arithmetic_sequence_property {a : ℕ → ℝ} (h : arithmetic_sequence a) (h1 : a 6 = 30) : a 3 + a 9 = 60 :=
by
  sorry

end arithmetic_sequence_property_l160_160353


namespace extremum_is_not_unique_l160_160689

-- Define the extremum conditionally in terms of unique extremum within an interval for a function
def isExtremum {α : Type*} [Preorder α] (f : α → ℝ) (x : α) :=
  ∀ y, f y ≤ f x ∨ f x ≤ f y

theorem extremum_is_not_unique (α : Type*) [Preorder α] (f : α → ℝ) :
  ¬ ∀ x, isExtremum f x → (∀ y, isExtremum f y → x = y) :=
by
  sorry

end extremum_is_not_unique_l160_160689


namespace percent_decrease_l160_160826

def original_price : ℝ := 100
def sale_price : ℝ := 60

theorem percent_decrease : (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end percent_decrease_l160_160826


namespace m_leq_nine_l160_160457

theorem m_leq_nine (m : ℝ) : (∀ x : ℝ, (x^2 - 4*x + 3 < 0) → (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + m < 0)) → m ≤ 9 :=
by
sorry

end m_leq_nine_l160_160457


namespace tangent_length_external_tangent_length_internal_l160_160382

noncomputable def tangent_length_ext (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R + r) / R)

noncomputable def tangent_length_int (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R - r) / R)

theorem tangent_length_external (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_ext R r a h hAB :=
sorry

theorem tangent_length_internal (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_int R r a h hAB :=
sorry

end tangent_length_external_tangent_length_internal_l160_160382


namespace find_a6_l160_160217

variable (S : ℕ → ℝ) (a : ℕ → ℝ)
variable (h1 : ∀ n ≥ 2, S n = 2 * a n)
variable (h2 : S 5 = 8)

theorem find_a6 : a 6 = 8 :=
by
  sorry

end find_a6_l160_160217


namespace axis_of_symmetry_l160_160339

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (5 - x)) : ∀ x : ℝ, f x = f (2 * 2.5 - x) :=
by
  sorry

end axis_of_symmetry_l160_160339


namespace greatest_GCD_of_product_7200_l160_160976

theorem greatest_GCD_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ ∀ d, (d ∣ a ∧ d ∣ b) → d ≤ 60 :=
by
  sorry

end greatest_GCD_of_product_7200_l160_160976


namespace probability_of_blue_face_l160_160203

theorem probability_of_blue_face (total_faces blue_faces : ℕ) (h_total : total_faces = 8) (h_blue : blue_faces = 5) : 
  blue_faces / total_faces = 5 / 8 :=
by
  sorry

end probability_of_blue_face_l160_160203


namespace magic_square_solution_l160_160027

theorem magic_square_solution (d e k f g h x y : ℤ)
  (h1 : x + 4 + f = 87 + d + f)
  (h2 : x + d + h = 87 + e + h)
  (h3 : x + y + 87 = 4 + d + e)
  (h4 : f + g + h = x + y + 87)
  (h5 : d = x - 83)
  (h6 : e = 2 * x - 170)
  (h7 : y = 3 * x - 274)
  (h8 : f = g)
  (h9 : g = h) :
  x = 62 ∧ y = -88 :=
by
  sorry

end magic_square_solution_l160_160027


namespace sum_lent_correct_l160_160132

noncomputable section

-- Define the principal amount (sum lent)
def P : ℝ := 4464.29

-- Define the interest rate per annum
def R : ℝ := 12.0

-- Define the time period in years
def T : ℝ := 12.0

-- Define the interest after 12 years (using the initial conditions and results)
def I : ℝ := 1.44 * P

-- Define the interest given as "2500 less than double the sum lent" condition
def I_condition : ℝ := 2 * P - 2500

-- Theorem stating the sum lent is the given value P
theorem sum_lent_correct : P = 4464.29 :=
by
  -- Placeholder for the proof
  sorry

end sum_lent_correct_l160_160132


namespace original_percentage_of_acid_l160_160633

theorem original_percentage_of_acid 
  (a w : ℝ) 
  (h1 : a + w = 6) 
  (h2 : a / (a + w + 2) = 15 / 100) 
  (h3 : (a + 2) / (a + w + 4) = 25 / 100) :
  (a / 6) * 100 = 20 :=
  sorry

end original_percentage_of_acid_l160_160633


namespace calculate_expression_l160_160106

def f (x : ℕ) : ℕ := x^2 - 3*x + 4
def g (x : ℕ) : ℕ := 2*x + 1

theorem calculate_expression : f (g 3) - g (f 3) = 23 := by
  sorry

end calculate_expression_l160_160106


namespace vector_subtraction_l160_160409

theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (3, 5)) (h2 : b = (-2, 1)) :
  a - (2 : ℝ) • b = (7, 3) :=
by
  rw [h1, h2]
  simp
  sorry

end vector_subtraction_l160_160409


namespace simplify_and_evaluate_div_expr_l160_160377

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end simplify_and_evaluate_div_expr_l160_160377


namespace percentage_increase_l160_160150

theorem percentage_increase (P : ℝ) (h : 200 * (1 + P/100) * 0.70 = 182) : 
  P = 30 := 
sorry

end percentage_increase_l160_160150


namespace arithmetic_formula_geometric_formula_comparison_S_T_l160_160103

noncomputable def a₁ : ℕ := 16
noncomputable def d : ℤ := -3

def a_n (n : ℕ) : ℤ := -3 * (n : ℤ) + 19
def b_n (n : ℕ) : ℤ := 4^(3 - n)

def S_n (n : ℕ) : ℚ := (-3 * (n : ℚ)^2 + 35 * n) / 2
def T_n (n : ℕ) : ℤ := -n^2 + 3 * n

theorem arithmetic_formula (n : ℕ) : a_n n = -3 * n + 19 :=
sorry

theorem geometric_formula (n : ℕ) : b_n n = 4^(3 - n) :=
sorry

theorem comparison_S_T (n : ℕ) :
  if n = 29 then S_n n = (T_n n : ℚ)
  else if n < 29 then S_n n > (T_n n : ℚ)
  else S_n n < (T_n n : ℚ) :=
sorry

end arithmetic_formula_geometric_formula_comparison_S_T_l160_160103


namespace determine_scores_l160_160121

variables {M Q S K : ℕ}

theorem determine_scores (h1 : Q > M ∨ K > M) 
                          (h2 : M ≠ K) 
                          (h3 : S ≠ Q) 
                          (h4 : S ≠ M) : 
  (Q, S, M) = (Q, S, M) :=
by
  -- We state the theorem as true
  sorry

end determine_scores_l160_160121


namespace find_ratio_l160_160856

-- Definition of the system of equations with k = 5
def system_of_equations (x y z : ℝ) :=
  x + 10 * y + 5 * z = 0 ∧
  2 * x + 5 * y + 4 * z = 0 ∧
  3 * x + 6 * y + 5 * z = 0

-- Proof that if (x, y, z) solves the system, then yz / x^2 = -3 / 49
theorem find_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : system_of_equations x y z) :
  (y * z) / (x ^ 2) = -3 / 49 :=
by
  -- Substitute the system of equations and solve for the ratio.
  sorry

end find_ratio_l160_160856


namespace find_height_of_box_l160_160177

-- Definitions for the problem conditions
def numCubes : ℕ := 24
def volumeCube : ℕ := 27
def lengthBox : ℕ := 8
def widthBox : ℕ := 9
def totalVolumeBox : ℕ := numCubes * volumeCube

-- Problem statement in Lean 4
theorem find_height_of_box : totalVolumeBox = lengthBox * widthBox * 9 :=
by sorry

end find_height_of_box_l160_160177


namespace soda_amount_l160_160611

theorem soda_amount (S : ℝ) (h1 : S / 2 + 2000 = (S - (S / 2 + 2000)) / 2 + 2000) : S = 12000 :=
by
  sorry

end soda_amount_l160_160611


namespace find_omega_increasing_intervals_l160_160792

noncomputable def f (ω x : ℝ) : ℝ :=
  (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (x : ℝ) : ℝ :=
  let ω := 3/2
  f ω (x - (Real.pi / 2))

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : ∀ x : ℝ, f ω (x + 2*Real.pi / (2*ω)) = f ω x) :
  ω = 3/2 :=
  sorry

theorem increasing_intervals (k : ℤ) :
  ∃ a b, 
  a = (2/3 * k * Real.pi + Real.pi / 4) ∧ 
  b = (2/3 * k * Real.pi + 7 * Real.pi / 12) ∧
  ∀ x, a ≤ x ∧ x ≤ b → g x < g (x + 1) :=
  sorry

end find_omega_increasing_intervals_l160_160792


namespace product_equals_permutation_l160_160978

-- Definitions and conditions
def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Given product sequence
def product_seq (n k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldr (λ x y => x * y) 1

-- Problem statement: The product of numbers from 18 to 9 is equivalent to A_{18}^{10}
theorem product_equals_permutation :
  product_seq 18 10 = perm 18 10 :=
by
  sorry

end product_equals_permutation_l160_160978


namespace enrico_earnings_l160_160793

theorem enrico_earnings : 
  let price_per_kg := 0.50
  let weight_rooster1 := 30
  let weight_rooster2 := 40
  let total_earnings := price_per_kg * weight_rooster1 + price_per_kg * weight_rooster2
  total_earnings = 35 := 
by
  sorry

end enrico_earnings_l160_160793


namespace combined_tax_rate_l160_160772

theorem combined_tax_rate
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h_john_income : john_income = 58000)
  (h_john_tax_rate : john_tax_rate = 0.30)
  (h_ingrid_income : ingrid_income = 72000)
  (h_ingrid_tax_rate : ingrid_tax_rate = 0.40) :
  ((john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income)) = 0.3553846154 :=
by
  sorry

end combined_tax_rate_l160_160772


namespace distance_point_to_line_l160_160585

theorem distance_point_to_line : 
  let x0 := 1
  let y0 := 0
  let A := 1
  let B := -2
  let C := 1 
  let dist := (A * x0 + B * y0 + C : ℝ) / Real.sqrt (A^2 + B^2)
  abs dist = 2 * Real.sqrt 5 / 5 :=
by
  -- Using basic principles of Lean and Mathlib to state the equality proof
  sorry

end distance_point_to_line_l160_160585


namespace seq_bound_gt_pow_two_l160_160903

theorem seq_bound_gt_pow_two (a : Fin 101 → ℕ) 
  (h1 : a 1 > a 0) 
  (h2 : ∀ n : Fin 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2 ^ 99 :=
sorry

end seq_bound_gt_pow_two_l160_160903


namespace remainder_invariance_l160_160546

theorem remainder_invariance (S A K : ℤ) (h : ∃ B r : ℤ, S = A * B + r ∧ 0 ≤ r ∧ r < |A|) :
  (∃ B' r' : ℤ, S + A * K = A * B' + r' ∧ r' = r) ∧ (∃ B'' r'' : ℤ, S - A * K = A * B'' + r'' ∧ r'' = r) :=
by
  sorry

end remainder_invariance_l160_160546


namespace parallel_vectors_imply_x_value_l160_160736

theorem parallel_vectors_imply_x_value (x : ℝ) : 
    let a := (1, 2)
    let b := (-1, x)
    (1 / -1:ℝ) = (2 / x) → x = -2 := 
by
  intro h
  sorry

end parallel_vectors_imply_x_value_l160_160736


namespace find_x_for_salt_solution_l160_160492

theorem find_x_for_salt_solution : ∀ (x : ℝ),
  (1 + x) * 0.10 = (x * 0.50) →
  x = 0.25 :=
by
  intros x h
  sorry

end find_x_for_salt_solution_l160_160492


namespace books_from_second_shop_l160_160246

-- Define the conditions
def num_books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1280
def cost_second_shop : ℕ := 880
def total_cost : ℤ := cost_first_shop + cost_second_shop
def average_price_per_book : ℤ := 18

-- Define the statement to be proved
theorem books_from_second_shop (x : ℕ) :
  (num_books_first_shop + x) * average_price_per_book = total_cost →
  x = 55 :=
by
  sorry

end books_from_second_shop_l160_160246


namespace solve_system_eq_l160_160712

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end solve_system_eq_l160_160712


namespace rhombus_area_l160_160305

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 11) (h2 : d2 = 16) : (d1 * d2) / 2 = 88 :=
by {
  -- substitution and proof are omitted, proof body would be provided here
  sorry
}

end rhombus_area_l160_160305


namespace equal_chords_divide_equally_l160_160720

theorem equal_chords_divide_equally 
  {A B C D M : ℝ} 
  (in_circle : ∃ (O : ℝ), (dist O A = dist O B) ∧ (dist O C = dist O D) ∧ (dist O M < dist O A))
  (chords_equal : dist A B = dist C D)
  (intersection_M : dist A M + dist M B = dist C M + dist M D ∧ dist A M = dist C M ∧ dist B M = dist D M) :
  dist A M = dist M B ∧ dist C M = dist M D := 
sorry

end equal_chords_divide_equally_l160_160720


namespace half_angle_in_second_quadrant_l160_160416

theorem half_angle_in_second_quadrant (α : Real) (h1 : 180 < α ∧ α < 270)
        (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
        90 < α / 2 ∧ α / 2 < 180 :=
sorry

end half_angle_in_second_quadrant_l160_160416


namespace megatek_manufacturing_percentage_l160_160146

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ) 
    (h1 : total_degrees = 360) 
    (h2 : manufacturing_degrees = 126) : 
    (manufacturing_degrees / total_degrees) * 100 = 35 := by
  sorry

end megatek_manufacturing_percentage_l160_160146


namespace initial_customers_correct_l160_160950

def initial_customers (remaining : ℕ) (left : ℕ) : ℕ := remaining + left

theorem initial_customers_correct :
  initial_customers 12 9 = 21 :=
by
  sorry

end initial_customers_correct_l160_160950


namespace quadratic_form_sum_const_l160_160780

theorem quadratic_form_sum_const (a b c x : ℝ) (h : 4 * x^2 - 28 * x - 48 = a * (x + b)^2 + c) : 
  a + b + c = -96.5 :=
by
  sorry

end quadratic_form_sum_const_l160_160780


namespace donny_cost_of_apples_l160_160069

def cost_of_apples (small_cost medium_cost big_cost : ℝ) (n_small n_medium n_big : ℕ) : ℝ := 
  n_small * small_cost + n_medium * medium_cost + n_big * big_cost

theorem donny_cost_of_apples :
  cost_of_apples 1.5 2 3 6 6 8 = 45 :=
by
  sorry

end donny_cost_of_apples_l160_160069


namespace toys_per_hour_computation_l160_160832

noncomputable def total_toys : ℕ := 20500
noncomputable def monday_hours : ℕ := 8
noncomputable def tuesday_hours : ℕ := 7
noncomputable def wednesday_hours : ℕ := 9
noncomputable def thursday_hours : ℕ := 6

noncomputable def total_hours_worked : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
noncomputable def toys_produced_each_hour : ℚ := total_toys / total_hours_worked

theorem toys_per_hour_computation :
  toys_produced_each_hour = 20500 / (8 + 7 + 9 + 6) :=
by
  -- Proof goes here
  sorry

end toys_per_hour_computation_l160_160832


namespace diane_initial_amount_l160_160644

theorem diane_initial_amount
  (X : ℝ)        -- the amount Diane started with
  (won_amount : ℝ := 65)
  (total_loss : ℝ := 215)
  (owing_friends : ℝ := 50)
  (final_amount := X + won_amount - total_loss - owing_friends) :
  X = 100 := 
by 
  sorry

end diane_initial_amount_l160_160644


namespace minimum_sum_of_dimensions_l160_160219

   theorem minimum_sum_of_dimensions (a b c : ℕ) (habc : a * b * c = 3003) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
     a + b + c = 45 :=
   sorry
   
end minimum_sum_of_dimensions_l160_160219


namespace geometric_sequence_common_ratio_l160_160422

theorem geometric_sequence_common_ratio {a : ℕ+ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_a3 : a 3 = 1) (h_a5 : a 5 = 4) : q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l160_160422


namespace reduced_admission_price_is_less_l160_160983

-- Defining the conditions
def regular_admission_cost : ℕ := 8
def total_people : ℕ := 2 + 3 + 1
def total_cost_before_6pm : ℕ := 30
def cost_per_person_before_6pm : ℕ := total_cost_before_6pm / total_people

-- Stating the theorem
theorem reduced_admission_price_is_less :
  (regular_admission_cost - cost_per_person_before_6pm) = 3 :=
by
  sorry -- Proof to be filled

end reduced_admission_price_is_less_l160_160983


namespace find_m_l160_160909

open Real

-- Definitions based on problem conditions
def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

-- The dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Prove the final statement using given conditions
theorem find_m (m : ℝ) (h1 : dot_product (vector_a m) vector_b + dot_product vector_b vector_b = 0) :
  m = 8 :=
sorry

end find_m_l160_160909


namespace mimi_spent_on_clothes_l160_160691

theorem mimi_spent_on_clothes :
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  total_spent - total_sneakers_cost = 2600 :=
by
  let total_spent := 8000
  let adidas_cost := 600
  let skechers_cost := 5 * adidas_cost
  let nike_cost := 3 * adidas_cost
  let total_sneakers_cost := adidas_cost + skechers_cost + nike_cost
  show total_spent - total_sneakers_cost = 2600
  sorry

end mimi_spent_on_clothes_l160_160691


namespace large_pile_toys_l160_160362

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end large_pile_toys_l160_160362


namespace ayse_guarantee_win_l160_160725

def can_ayse_win (m n k : ℕ) : Prop :=
  -- Function defining the winning strategy for Ayşe
  sorry -- The exact strategy definition would be here

theorem ayse_guarantee_win :
  ((can_ayse_win 1 2012 2014) ∧ 
   (can_ayse_win 2011 2011 2012) ∧ 
   (can_ayse_win 2011 2012 2013) ∧ 
   (can_ayse_win 2011 2012 2014) ∧ 
   (can_ayse_win 2011 2013 2013)) = true :=
sorry -- Proof goes here

end ayse_guarantee_win_l160_160725


namespace flowers_on_porch_l160_160763

theorem flowers_on_porch (total_plants : ℕ) (flowering_percentage : ℝ) (fraction_on_porch : ℝ) (flowers_per_plant : ℕ) (h1 : total_plants = 80) (h2 : flowering_percentage = 0.40) (h3 : fraction_on_porch = 0.25) (h4 : flowers_per_plant = 5) : (total_plants * flowering_percentage * fraction_on_porch * flowers_per_plant) = 40 :=
by
  sorry

end flowers_on_porch_l160_160763


namespace probability_of_non_touching_square_is_correct_l160_160288

def square_not_touching_perimeter_or_center_probability : ℚ :=
  let total_squares := 100
  let perimeter_squares := 24
  let center_line_squares := 16
  let touching_squares := perimeter_squares + center_line_squares
  let non_touching_squares := total_squares - touching_squares
  non_touching_squares / total_squares

theorem probability_of_non_touching_square_is_correct :
  square_not_touching_perimeter_or_center_probability = 3 / 5 :=
by
  sorry

end probability_of_non_touching_square_is_correct_l160_160288


namespace length_of_room_l160_160626

theorem length_of_room (area_in_sq_inches : ℕ) (length_of_side_in_feet : ℕ) (h1 : area_in_sq_inches = 14400)
  (h2 : length_of_side_in_feet * length_of_side_in_feet = area_in_sq_inches / 144) : length_of_side_in_feet = 10 :=
  by
  sorry

end length_of_room_l160_160626


namespace exists_odd_k_l_m_l160_160943

def odd_nat (n : ℕ) : Prop := n % 2 = 1

theorem exists_odd_k_l_m : 
  ∃ (k l m : ℕ), 
  odd_nat k ∧ odd_nat l ∧ odd_nat m ∧ 
  (k ≠ 0) ∧ (l ≠ 0) ∧ (m ≠ 0) ∧ 
  (1991 * (l * m + k * m + k * l) = k * l * m) :=
by
  sorry

end exists_odd_k_l_m_l160_160943


namespace george_speed_to_school_l160_160432

theorem george_speed_to_school :
  ∀ (d1 d2 v1 v2 v_arrive : ℝ), 
  d1 = 1.0 → d2 = 0.5 → v1 = 3.0 → v2 * (d1 / v1 + d2 / v2) = (d1 + d2) / 4.0 → v_arrive = 12.0 :=
by sorry

end george_speed_to_school_l160_160432


namespace number_of_ways_to_place_letters_l160_160083

-- Define the number of letters and mailboxes
def num_letters : Nat := 3
def num_mailboxes : Nat := 5

-- Define the function to calculate the number of ways to place the letters into mailboxes
def count_ways (n : Nat) (m : Nat) : Nat := m ^ n

-- The theorem to prove
theorem number_of_ways_to_place_letters :
  count_ways num_letters num_mailboxes = 5 ^ 3 :=
by
  sorry

end number_of_ways_to_place_letters_l160_160083


namespace sin_value_given_cos_condition_l160_160881

theorem sin_value_given_cos_condition (theta : ℝ) (h : Real.cos (5 * Real.pi / 12 - theta) = 1 / 3) :
  Real.sin (Real.pi / 12 + theta) = 1 / 3 :=
sorry

end sin_value_given_cos_condition_l160_160881


namespace Tom_green_marbles_l160_160247

-- Define the given variables
def Sara_green_marbles : Nat := 3
def Total_green_marbles : Nat := 7

-- The statement to be proven
theorem Tom_green_marbles : (Total_green_marbles - Sara_green_marbles) = 4 := by
  sorry

end Tom_green_marbles_l160_160247


namespace number_of_spiders_l160_160886

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) 
  (h1 : total_legs = 40) (h2 : legs_per_spider = 8) : 
  (total_legs / legs_per_spider = 5) :=
by
  -- Placeholder for the actual proof
  sorry

end number_of_spiders_l160_160886


namespace mixed_groups_count_l160_160462

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l160_160462


namespace number_of_pages_in_contract_l160_160940

theorem number_of_pages_in_contract (total_pages_copied : ℕ) (copies_per_person : ℕ) (number_of_people : ℕ)
  (h1 : total_pages_copied = 360) (h2 : copies_per_person = 2) (h3 : number_of_people = 9) :
  total_pages_copied / (copies_per_person * number_of_people) = 20 :=
by
  sorry

end number_of_pages_in_contract_l160_160940


namespace pumps_work_hours_l160_160522

theorem pumps_work_hours (d : ℕ) (h_d_pos : d > 0) : 6 * (8 / d) * d = 48 :=
by
  -- The proof is omitted
  sorry

end pumps_work_hours_l160_160522


namespace odd_function_value_sum_l160_160735

theorem odd_function_value_sum
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_fneg1 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end odd_function_value_sum_l160_160735


namespace units_digit_n_l160_160732

theorem units_digit_n (m n : ℕ) (hm : m % 10 = 9) (h : m * n = 18^5) : n % 10 = 2 :=
sorry

end units_digit_n_l160_160732


namespace cube_convex_hull_half_volume_l160_160335

theorem cube_convex_hull_half_volume : 
  ∃ a : ℝ, 0 <= a ∧ a <= 1 ∧ 4 * (a^3) / 6 + 4 * ((1 - a)^3) / 6 = 1 / 2 :=
by
  sorry

end cube_convex_hull_half_volume_l160_160335


namespace units_digit_2_1501_5_1602_11_1703_l160_160282

theorem units_digit_2_1501_5_1602_11_1703 : 
  (2 ^ 1501 * 5 ^ 1602 * 11 ^ 1703) % 10 = 0 :=
  sorry

end units_digit_2_1501_5_1602_11_1703_l160_160282


namespace smallest_a_l160_160231

def f (x : ℕ) : ℕ :=
  if x % 21 = 0 then x / 21
  else if x % 7 = 0 then 3 * x
  else if x % 3 = 0 then 7 * x
  else x + 3

def f_iterate (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate f a x

theorem smallest_a (a : ℕ) : a > 1 ∧ f_iterate a 2 = f 2 ↔ a = 7 := 
sorry

end smallest_a_l160_160231


namespace total_trees_in_forest_l160_160364

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l160_160364


namespace tan_theta_eq_sqrt3_div_3_l160_160506

theorem tan_theta_eq_sqrt3_div_3
  (θ : ℝ)
  (h : (Real.cos θ * Real.sqrt 3 + Real.sin θ) = 2) :
  Real.tan θ = Real.sqrt 3 / 3 := by
  sorry

end tan_theta_eq_sqrt3_div_3_l160_160506


namespace highland_baseball_club_members_l160_160529

-- Define the given costs and expenditures.
def socks_cost : ℕ := 6
def tshirt_cost : ℕ := socks_cost + 7
def cap_cost : ℕ := socks_cost
def total_expenditure : ℕ := 5112
def home_game_cost : ℕ := socks_cost + tshirt_cost
def away_game_cost : ℕ := socks_cost + tshirt_cost + cap_cost
def cost_per_member : ℕ := home_game_cost + away_game_cost

theorem highland_baseball_club_members :
  total_expenditure / cost_per_member = 116 :=
by
  sorry

end highland_baseball_club_members_l160_160529


namespace ticket_sales_total_l160_160494

variable (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ)

def total_money_collected (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - child_tickets
  let total_child := child_tickets * price_child
  let total_adult := adult_tickets * price_adult
  total_child + total_adult

theorem ticket_sales_total :
  price_adult = 6 →
  price_child = 4 →
  total_tickets = 21 →
  child_tickets = 11 →
  total_money_collected price_adult price_child total_tickets child_tickets = 104 :=
by
  intros
  unfold total_money_collected
  simp
  sorry

end ticket_sales_total_l160_160494


namespace find_a_for_inequality_l160_160869

theorem find_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 3) → -2 * x^2 + a * x + 6 > 0) → a = 2 :=
by
  sorry

end find_a_for_inequality_l160_160869


namespace bert_money_left_l160_160183

theorem bert_money_left
  (initial_amount : ℝ)
  (spent_hardware_store_fraction : ℝ)
  (amount_spent_dry_cleaners : ℝ)
  (spent_grocery_store_fraction : ℝ)
  (final_amount : ℝ) :
  initial_amount = 44 →
  spent_hardware_store_fraction = 1/4 →
  amount_spent_dry_cleaners = 9 →
  spent_grocery_store_fraction = 1/2 →
  final_amount = initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners - (spent_grocery_store_fraction * (initial_amount - (spent_hardware_store_fraction * initial_amount) - amount_spent_dry_cleaners)) →
  final_amount = 12 :=
by
  sorry

end bert_money_left_l160_160183


namespace gcd_calculation_l160_160619

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end gcd_calculation_l160_160619


namespace ratio_fifth_term_l160_160059

-- Definitions of arithmetic sequences and sums
def arithmetic_seq_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := n * (2 * a 1 + (n - 1) * d 1) / 2

-- Conditions
variables (S_n S'_n : ℕ → ℕ) (n : ℕ)

-- Given conditions
axiom ratio_sum : ∀ (n : ℕ), S_n n / S'_n n = (5 * n + 3) / (2 * n + 7)
axiom sums_at_9 : S_n 9 = 9 * (S_n 1 + S_n 9) / 2
axiom sums'_at_9 : S'_n 9 = 9 * (S'_n 1 + S'_n 9) / 2

-- Theorem to prove
theorem ratio_fifth_term : (9 * (S_n 1 + S_n 9) / 2) / (9 * (S'_n 1 + S'_n 9) / 2) = 48 / 25 := sorry

end ratio_fifth_term_l160_160059


namespace rectangle_area_l160_160804

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end rectangle_area_l160_160804


namespace cats_in_studio_count_l160_160548

theorem cats_in_studio_count :
  (70 + 40 + 30 + 50
  - 25 - 15 - 20 - 28
  + 5 + 10 + 12
  - 8
  + 12) = 129 :=
by sorry

end cats_in_studio_count_l160_160548


namespace find_numbers_l160_160744

theorem find_numbers : ∃ x y : ℕ, x + y = 2016 ∧ (∃ d : ℕ, d < 10 ∧ (x = 10 * y + d) ∧ x = 1833 ∧ y = 183) :=
by 
  sorry

end find_numbers_l160_160744


namespace equivalent_single_discount_rate_l160_160299

-- Definitions based on conditions
def original_price : ℝ := 120
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.15
def combined_discount_rate : ℝ := 0.3625  -- This is the expected result

-- The proof problem statement
theorem equivalent_single_discount_rate :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 
  (original_price * (1 - combined_discount_rate)) := 
sorry

end equivalent_single_discount_rate_l160_160299


namespace complex_square_l160_160459

theorem complex_square (z : ℂ) (i : ℂ) (h₁ : z = 5 - 3 * i) (h₂ : i * i = -1) : z^2 = 16 - 30 * i :=
by
  rw [h₁]
  sorry

end complex_square_l160_160459


namespace power_of_a_l160_160087

theorem power_of_a (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 := sorry

end power_of_a_l160_160087


namespace three_rays_with_common_point_l160_160324

theorem three_rays_with_common_point (x y : ℝ) :
  (∃ (common : ℝ), ((5 = x - 1 ∧ y + 3 ≤ 5) ∨ 
                     (5 = y + 3 ∧ x - 1 ≤ 5) ∨ 
                     (x - 1 = y + 3 ∧ 5 ≤ x - 1 ∧ 5 ≤ y + 3)) 
  ↔ ((x = 6 ∧ y ≤ 2) ∨ (y = 2 ∧ x ≤ 6) ∨ (y = x - 4 ∧ x ≥ 6))) :=
sorry

end three_rays_with_common_point_l160_160324


namespace no_real_solutions_l160_160441

theorem no_real_solutions :
  ¬ ∃ (a b c d : ℝ), 
  (a^3 + c^3 = 2) ∧ 
  (a^2 * b + c^2 * d = 0) ∧ 
  (b^3 + d^3 = 1) ∧ 
  (a * b^2 + c * d^2 = -6) := 
by
  sorry

end no_real_solutions_l160_160441


namespace expected_value_is_correct_l160_160385

-- Given conditions
def prob_heads : ℚ := 2 / 5
def prob_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def loss_amount_tails : ℚ := -3

-- Expected value calculation
def expected_value : ℚ := prob_heads * win_amount_heads + prob_tails * loss_amount_tails

-- Property to prove
theorem expected_value_is_correct : expected_value = 0.2 := sorry

end expected_value_is_correct_l160_160385


namespace distance_MF_l160_160418

-- Define the conditions for the problem
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def focus : (ℝ × ℝ) := (2, 0)

def lies_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

def distance_to_line (M : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  abs (M.1 - line_x)

def point_M_conditions (M : ℝ × ℝ) : Prop :=
  distance_to_line M (-3) = 6 ∧ lies_on_parabola M

-- The final proof problem statement in Lean
theorem distance_MF (M : ℝ × ℝ) (h : point_M_conditions M) : dist M focus = 5 :=
by sorry

end distance_MF_l160_160418


namespace shots_cost_l160_160342

-- Define the conditions
def golden_retriever_pregnant_dogs : ℕ := 3
def golden_retriever_puppies_per_dog : ℕ := 4
def golden_retriever_shots_per_puppy : ℕ := 2
def golden_retriever_cost_per_shot : ℕ := 5

def german_shepherd_pregnant_dogs : ℕ := 2
def german_shepherd_puppies_per_dog : ℕ := 5
def german_shepherd_shots_per_puppy : ℕ := 3
def german_shepherd_cost_per_shot : ℕ := 8

def bulldog_pregnant_dogs : ℕ := 4
def bulldog_puppies_per_dog : ℕ := 3
def bulldog_shots_per_puppy : ℕ := 4
def bulldog_cost_per_shot : ℕ := 10

-- Define the total cost calculation
def total_puppies (dogs_per_breed puppies_per_dog : ℕ) : ℕ :=
  dogs_per_breed * puppies_per_dog

def total_shot_cost (puppies shots_per_puppy cost_per_shot : ℕ) : ℕ :=
  puppies * shots_per_puppy * cost_per_shot

def total_cost : ℕ :=
  let golden_retriever_puppies := total_puppies golden_retriever_pregnant_dogs golden_retriever_puppies_per_dog
  let german_shepherd_puppies := total_puppies german_shepherd_pregnant_dogs german_shepherd_puppies_per_dog
  let bulldog_puppies := total_puppies bulldog_pregnant_dogs bulldog_puppies_per_dog
  let golden_retriever_cost := total_shot_cost golden_retriever_puppies golden_retriever_shots_per_puppy golden_retriever_cost_per_shot
  let german_shepherd_cost := total_shot_cost german_shepherd_puppies german_shepherd_shots_per_puppy german_shepherd_cost_per_shot
  let bulldog_cost := total_shot_cost bulldog_puppies bulldog_shots_per_puppy bulldog_cost_per_shot
  golden_retriever_cost + german_shepherd_cost + bulldog_cost

-- Statement of the problem
theorem shots_cost (total_cost : ℕ) : total_cost = 840 := by
  -- Proof would go here
  sorry

end shots_cost_l160_160342


namespace last_week_profit_min_selling_price_red_beauty_l160_160710

theorem last_week_profit (x kgs_of_red_beauty x_green : ℕ) 
  (purchase_cost_red_beauty_per_kg selling_cost_red_beauty_per_kg 
  purchase_cost_xiangshan_green_per_kg selling_cost_xiangshan_green_per_kg
  total_weight total_cost all_fruits_profit : ℕ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  selling_cost_red_beauty_per_kg = 35 ->
  purchase_cost_xiangshan_green_per_kg = 5 ->
  selling_cost_xiangshan_green_per_kg = 10 ->
  total_weight = 300 ->
  total_cost = 3000 ->
  x * purchase_cost_red_beauty_per_kg + (total_weight - x) * purchase_cost_xiangshan_green_per_kg = total_cost ->
  all_fruits_profit = x * (selling_cost_red_beauty_per_kg - purchase_cost_red_beauty_per_kg) +
  (total_weight - x) * (selling_cost_xiangshan_green_per_kg - purchase_cost_xiangshan_green_per_kg) -> 
  all_fruits_profit = 2500 := sorry

theorem min_selling_price_red_beauty (last_week_profit : ℕ) (x kgs_of_red_beauty x_green damaged_ratio : ℝ) 
  (purchase_cost_red_beauty_per_kg profit_last_week selling_cost_xiangshan_per_kg 
  total_weight total_cost : ℝ) :
  purchase_cost_red_beauty_per_kg = 20 ->
  profit_last_week = 2500 ->
  damaged_ratio = 0.1 ->
  x = 100 ->
  (profit_last_week = 
    x * (35 - purchase_cost_red_beauty_per_kg) + (total_weight - x) * (10 - 5)) ->
  90 * (purchase_cost_red_beauty_per_kg + (last_week_profit - 15 * (total_weight - x) / 90)) ≥ 1500 ->
  profit_last_week / (90 * (90 * (purchase_cost_red_beauty_per_kg + (2500 - 15 * (300 - x) / 90)))) >=
  (36.7 - 20 / purchase_cost_red_beauty_per_kg) :=
  sorry

end last_week_profit_min_selling_price_red_beauty_l160_160710


namespace other_endpoint_diameter_l160_160612

theorem other_endpoint_diameter (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hO : O = (2, 3)) (hA : A = (-1, -1)) 
  (h_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : B = (5, 7) := by
  sorry

end other_endpoint_diameter_l160_160612


namespace alpha_range_theorem_l160_160895

noncomputable def alpha_range (k : ℤ) (α : ℝ) : Prop :=
  2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi

theorem alpha_range_theorem (α : ℝ) (k : ℤ) (h : |Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) :
  alpha_range k α :=
by
  sorry

end alpha_range_theorem_l160_160895


namespace first_group_number_l160_160143

theorem first_group_number (x : ℕ) (h1 : x + 120 = 126) : x = 6 :=
by
  sorry

end first_group_number_l160_160143


namespace no_perfect_square_abc_sum_l160_160489

theorem no_perfect_square_abc_sum (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  ¬ ∃ m : ℕ, m * m = (100 * a + 10 * b + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) :=
by
  sorry

end no_perfect_square_abc_sum_l160_160489


namespace equation_of_line_m_l160_160999

-- Given conditions
def point (α : Type*) := α × α

def l_eq (p : point ℝ) : Prop := p.1 + 3 * p.2 = 7 -- Equation of line l
def m_intercept : point ℝ := (1, 2) -- Intersection point of l and m
def q : point ℝ := (2, 5) -- Point Q
def q'' : point ℝ := (5, 0) -- Point Q''

-- Proving the equation of line m
theorem equation_of_line_m (m_eq : point ℝ → Prop) :
  (∀ P : point ℝ, m_eq P ↔ P.2 = 2 * P.1 - 2) ↔
  (∃ P : point ℝ, m_eq P ∧ P = (5, 0)) :=
sorry

end equation_of_line_m_l160_160999


namespace piper_gym_sessions_l160_160331

-- Define the conditions and the final statement as a theorem
theorem piper_gym_sessions (session_count : ℕ) (week_days : ℕ) (start_day : ℕ) 
  (alternate_day : ℕ) (skip_day : ℕ): (session_count = 35) ∧ (week_days = 7) ∧ 
  (start_day = 1) ∧ (alternate_day = 2) ∧ (skip_day = 7) → 
  (start_day + ((session_count - 1) / 3) * week_days + ((session_count - 1) % 3) * alternate_day) % week_days = 3 := 
by 
  sorry

end piper_gym_sessions_l160_160331


namespace correct_statement_is_D_l160_160010

axiom three_points_determine_plane : Prop
axiom line_and_point_determine_plane : Prop
axiom quadrilateral_is_planar_figure : Prop
axiom two_intersecting_lines_determine_plane : Prop

theorem correct_statement_is_D : two_intersecting_lines_determine_plane = True := 
by sorry

end correct_statement_is_D_l160_160010


namespace sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l160_160388

-- Problem 1
theorem sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3 : |Real.sqrt 3 - Real.sqrt 2| + Real.sqrt 2 = Real.sqrt 3 := by
  sorry

-- Problem 2
theorem sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2 : Real.sqrt 2 * (Real.sqrt 2 + 2) = 2 + 2 * Real.sqrt 2 := by
  sorry

end sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l160_160388


namespace legacy_earnings_per_hour_l160_160997

-- Define the conditions
def totalFloors : ℕ := 4
def roomsPerFloor : ℕ := 10
def hoursPerRoom : ℕ := 6
def totalEarnings : ℝ := 3600

-- The statement to prove
theorem legacy_earnings_per_hour :
  (totalFloors * roomsPerFloor * hoursPerRoom) = 240 → 
  (totalEarnings / (totalFloors * roomsPerFloor * hoursPerRoom)) = 15 := by
  intros h
  sorry

end legacy_earnings_per_hour_l160_160997


namespace max_value_of_fraction_l160_160430

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l160_160430


namespace total_painting_area_correct_l160_160256

def barn_width : ℝ := 12
def barn_length : ℝ := 15
def barn_height : ℝ := 6

def area_to_be_painted (width length height : ℝ) : ℝ := 
  2 * (width * height + length * height) + width * length

theorem total_painting_area_correct : area_to_be_painted barn_width barn_length barn_height = 828 := 
  by sorry

end total_painting_area_correct_l160_160256


namespace product_of_numbers_l160_160563

theorem product_of_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 150)
  (h2 : 7 * x = n)
  (h3 : y - 10 = n)
  (h4 : z + 10 = n) : x * y * z = 48000 := 
by 
  sorry

end product_of_numbers_l160_160563


namespace trader_profit_l160_160632

theorem trader_profit (donation goal extra profit : ℝ) (half_profit : ℝ) 
  (H1 : donation = 310) (H2 : goal = 610) (H3 : extra = 180)
  (H4 : half_profit = profit / 2) 
  (H5 : half_profit + donation = goal + extra) : 
  profit = 960 := 
by
  sorry

end trader_profit_l160_160632


namespace gcd_lcm_product_l160_160527

theorem gcd_lcm_product (a b : ℕ) (h₀ : a = 15) (h₁ : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 675 :=
by
  sorry

end gcd_lcm_product_l160_160527


namespace maximum_take_home_pay_l160_160180

noncomputable def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - ((x + 10) / 100 * 1000 * x)

theorem maximum_take_home_pay : 
  ∃ x : ℝ, (take_home_pay x = 20250) ∧ (45000 = 1000 * x) :=
by
  sorry

end maximum_take_home_pay_l160_160180


namespace jordan_walk_distance_l160_160078

theorem jordan_walk_distance
  (d t : ℝ)
  (flat_speed uphill_speed walk_speed : ℝ)
  (total_time : ℝ)
  (h1 : flat_speed = 18)
  (h2 : uphill_speed = 6)
  (h3 : walk_speed = 4)
  (h4 : total_time = 3)
  (h5 : d / (3 * 18) + d / (3 * 6) + d / (3 * 4) = total_time) :
  t = 6.6 :=
by
  -- Proof goes here
  sorry

end jordan_walk_distance_l160_160078


namespace bridge_height_at_distance_l160_160110

theorem bridge_height_at_distance :
  (∃ (a : ℝ), ∀ (x : ℝ), (x = 25) → (a * x^2 + 25 = 0)) →
  (∀ (x : ℝ), (x = 10) → (-1/25 * x^2 + 25 = 21)) :=
by
  intro h1
  intro x h2
  have h : 625 * (-1 / 25) * (-1 / 25) = -25 := sorry
  sorry

end bridge_height_at_distance_l160_160110


namespace cross_section_area_correct_l160_160692

noncomputable def cross_section_area (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 33) / 8

theorem cross_section_area_correct
  (AB CC1 : ℝ)
  (h1 : AB = a)
  (h2 : CC1 = 2 * a) :
  cross_section_area a = (3 * a^2 * Real.sqrt 33) / 8 :=
by
  sorry

end cross_section_area_correct_l160_160692


namespace sum_abs_diff_is_18_l160_160875

noncomputable def sum_of_possible_abs_diff (a b c d : ℝ) : ℝ :=
  let possible_values := [
      abs ((a + 2) - (d - 7)),
      abs ((a + 2) - (d + 1)),
      abs ((a + 2) - (d - 1)),
      abs ((a + 2) - (d + 7)),
      abs ((a - 2) - (d - 7)),
      abs ((a - 2) - (d + 1)),
      abs ((a - 2) - (d - 1)),
      abs ((a - 2) - (d + 7))
  ]
  possible_values.foldl (· + ·) 0

theorem sum_abs_diff_is_18 (a b c d : ℝ) (h1 : abs (a - b) = 2) (h2 : abs (b - c) = 3) (h3 : abs (c - d) = 4) :
  sum_of_possible_abs_diff a b c d = 18 := by
  sorry

end sum_abs_diff_is_18_l160_160875


namespace intersection_point_l160_160584

variable (x y : ℝ)

theorem intersection_point :
  (y = 9 / (x^2 + 3)) →
  (x + y = 3) →
  (x = 0) := by
  intros h1 h2
  sorry

end intersection_point_l160_160584


namespace number_of_bags_needed_l160_160727

def cost_corn_seeds : ℕ := 50
def cost_fertilizers_pesticides : ℕ := 35
def cost_labor : ℕ := 15
def profit_percentage : ℝ := 0.10
def price_per_bag : ℝ := 11

theorem number_of_bags_needed (total_cost : ℕ) (total_revenue : ℝ) (num_bags : ℝ) :
  total_cost = cost_corn_seeds + cost_fertilizers_pesticides + cost_labor →
  total_revenue = ↑total_cost + (↑total_cost * profit_percentage) →
  num_bags = total_revenue / price_per_bag →
  num_bags = 10 := 
by
  sorry

end number_of_bags_needed_l160_160727


namespace shortest_altitude_l160_160139

theorem shortest_altitude (a b c : ℕ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) (h4 : a^2 + b^2 = c^2) : ∃ x, x = 9.6 :=
by
  sorry

end shortest_altitude_l160_160139


namespace value_of_b_over_a_l160_160024

def rectangle_ratio (a b : ℝ) : Prop :=
  let d := Real.sqrt (a^2 + b^2)
  let P := 2 * (a + b)
  (b / d) = (d / (a + b))

theorem value_of_b_over_a (a b : ℝ) (h : rectangle_ratio a b) : b / a = 1 :=
by sorry

end value_of_b_over_a_l160_160024


namespace second_player_wins_l160_160297

theorem second_player_wins 
  (pile1 : ℕ) (pile2 : ℕ) (pile3 : ℕ)
  (h1 : pile1 = 10) (h2 : pile2 = 15) (h3 : pile3 = 20) :
  (pile1 - 1) + (pile2 - 1) + (pile3 - 1) % 2 = 0 :=
by
  sorry

end second_player_wins_l160_160297


namespace g_eval_l160_160966

-- Define the function g
def g (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := (2 * a + b) / (c - a)

-- Theorem to prove g(2, 4, -1) = -8 / 3
theorem g_eval :
  g 2 4 (-1) = -8 / 3 := 
by
  sorry

end g_eval_l160_160966


namespace median_hypotenuse_right_triangle_l160_160773

/-- Prove that in a right triangle with legs of lengths 5 and 12,
  the median on the hypotenuse can be either 6 or 6.5. -/
theorem median_hypotenuse_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) :
  ∃ c : ℝ, (c = 6 ∨ c = 6.5) :=
sorry

end median_hypotenuse_right_triangle_l160_160773


namespace range_of_a_l160_160068

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end range_of_a_l160_160068


namespace tony_additional_degrees_l160_160348

-- Definitions for the conditions
def total_years : ℕ := 14
def science_degree_years : ℕ := 4
def physics_degree_years : ℕ := 2
def additional_degree_years : ℤ := total_years - (science_degree_years + physics_degree_years)
def each_additional_degree_years : ℕ := 4
def additional_degrees : ℤ := additional_degree_years / each_additional_degree_years

-- Theorem stating the problem and the answer
theorem tony_additional_degrees : additional_degrees = 2 :=
 by
     sorry

end tony_additional_degrees_l160_160348


namespace apple_tree_total_production_l160_160586

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l160_160586


namespace total_sample_variance_l160_160750

/-- In a survey of the heights (in cm) of high school students at Shuren High School:

 - 20 boys were selected with an average height of 174 cm and a variance of 12.
 - 30 girls were selected with an average height of 164 cm and a variance of 30.

We need to prove that the variance of the total sample is 46.8. -/
theorem total_sample_variance :
  let boys_count := 20
  let girls_count := 30
  let boys_avg := 174
  let girls_avg := 164
  let boys_var := 12
  let girls_var := 30
  let total_count := boys_count + girls_count
  let overall_avg := (boys_avg * boys_count + girls_avg * girls_count) / total_count
  let total_var := 
    (boys_count * (boys_var + (boys_avg - overall_avg)^2) / total_count)
    + (girls_count * (girls_var + (girls_avg - overall_avg)^2) / total_count)
  total_var = 46.8 := by
    sorry

end total_sample_variance_l160_160750


namespace taxi_fare_distance_l160_160764

variable (x : ℝ)

theorem taxi_fare_distance (h1 : 0 ≤ x - 2) (h2 : 3 + 1.2 * (x - 2) = 9) : x = 7 := by
  sorry

end taxi_fare_distance_l160_160764


namespace complement_union_A_B_in_U_l160_160752

open Set Nat

def U : Set ℕ := { x | x < 6 ∧ x > 0 }
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_A_B_in_U : (U \ (A ∪ B)) = {2, 4} := by
  sorry

end complement_union_A_B_in_U_l160_160752


namespace find_negative_a_l160_160674

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3 * x - 22

theorem find_negative_a (a : ℝ) (ha : a < 0) :
  g (g (g 7)) = g (g (g a)) ↔ a = -23 / 3 :=
by
  sorry

end find_negative_a_l160_160674


namespace smallest_n_candy_price_l160_160211

theorem smallest_n_candy_price :
  ∃ n : ℕ, 25 * n = Nat.lcm (Nat.lcm 20 18) 24 ∧ ∀ k : ℕ, k > 0 ∧ 25 * k = Nat.lcm (Nat.lcm 20 18) 24 → n ≤ k :=
sorry

end smallest_n_candy_price_l160_160211


namespace lemonade_quarts_l160_160955

theorem lemonade_quarts (total_parts water_parts lemon_juice_parts : ℕ) (total_gallons gallons_to_quarts : ℚ) 
  (h_ratio : water_parts = 4) (h_ratio_lemon : lemon_juice_parts = 1) (h_total_parts : total_parts = water_parts + lemon_juice_parts)
  (h_total_gallons : total_gallons = 1) (h_gallons_to_quarts : gallons_to_quarts = 4) :
  let volume_per_part := total_gallons / total_parts
  let volume_per_part_quarts := volume_per_part * gallons_to_quarts
  let water_volume := water_parts * volume_per_part_quarts
  water_volume = 16 / 5 :=
by
  sorry

end lemonade_quarts_l160_160955


namespace physics_marks_l160_160530

theorem physics_marks (P C M : ℕ) (h1 : P + C + M = 180) (h2 : P + M = 180) (h3 : P + C = 140) : P = 140 :=
by
  sorry

end physics_marks_l160_160530


namespace max_omega_l160_160646

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem max_omega (ω φ : ℝ) (k k' : ℤ) (hω_pos : ω > 0) (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi / 2) (h1 : f ω φ (-Real.pi / 4) = 0)
  (h2 : ∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x))
  (h3 : ∀ x, x ∈ Set.Ioo (Real.pi / 18) (2 * Real.pi / 9) →
    Monotone (f ω φ)) :
  ω = 5 :=
sorry

end max_omega_l160_160646


namespace find_f1_l160_160443

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 1 / 2 → f x + f ((x + 2) / (1 - 2 * x)) = x) :
  f 1 = 7 / 6 :=
sorry

end find_f1_l160_160443


namespace parabola_directrix_l160_160295

theorem parabola_directrix (x y : ℝ) (h : y = 2 * x^2) : y = - (1 / 8) :=
sorry

end parabola_directrix_l160_160295


namespace translated_B_is_B_l160_160887

def point : Type := ℤ × ℤ

def A : point := (-4, -1)
def A' : point := (-2, 2)
def B : point := (1, 1)
def B' : point := (3, 4)

def translation_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2)

def translate_point (p : point) (v : point) : point :=
  (p.1 + v.1, p.2 + v.2)

theorem translated_B_is_B' : translate_point B (translation_vector A A') = B' :=
by
  sorry

end translated_B_is_B_l160_160887


namespace SomuAge_l160_160941

theorem SomuAge (F S : ℕ) (h1 : S = F / 3) (h2 : S - 8 = (F - 8) / 5) : S = 16 :=
by 
  sorry

end SomuAge_l160_160941


namespace arithmetic_sequence_sum_l160_160467

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end arithmetic_sequence_sum_l160_160467


namespace vector_identity_l160_160553

-- Definitions of the vectors
variables {V : Type*} [AddGroup V]

-- Conditions as Lean definitions
def cond1 (AB BO AO : V) : Prop := AB + BO = AO
def cond2 (AO OM AM : V) : Prop := AO + OM = AM
def cond3 (AM MB AB : V) : Prop := AM + MB = AB

-- The main statement to be proved
theorem vector_identity (AB MB BO BC OM AO AM AC : V) 
  (h1 : cond1 AB BO AO) 
  (h2 : cond2 AO OM AM) 
  (h3 : cond3 AM MB AB) 
  : (AB + MB) + (BO + BC) + OM = AC :=
sorry

end vector_identity_l160_160553


namespace olaf_total_toy_cars_l160_160270

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end olaf_total_toy_cars_l160_160270


namespace problem_statement_l160_160851

-- Definitions for given conditions
variables (a b m n x : ℤ)

-- Assuming conditions: a = -b, mn = 1, and |x| = 2
axiom opp_num : a = -b
axiom recip : m * n = 1
axiom abs_x : |x| = 2

-- Problem statement to prove
theorem problem_statement :
  -2 * m * n + (a + b) / 2023 + x * x = 2 :=
by 
  sorry

end problem_statement_l160_160851


namespace question_l160_160262
-- Importing necessary libraries

-- Stating the problem
theorem question (x : ℤ) (h : (x + 12) / 8 = 9) : 35 - (x / 2) = 5 :=
by {
  sorry
}

end question_l160_160262


namespace max_distance_from_curve_to_line_l160_160893

theorem max_distance_from_curve_to_line
  (θ : ℝ) (t : ℝ)
  (C_polar_eqn : ∀ θ, ∃ (ρ : ℝ), ρ = 2 * Real.cos θ)
  (line_eqn : ∀ t, ∃ (x y : ℝ), x = -1 + t ∧ y = 2 * t) :
  ∃ (max_dist : ℝ), max_dist = (4 * Real.sqrt 5 + 5) / 5 := sorry

end max_distance_from_curve_to_line_l160_160893


namespace solve_for_t_l160_160314

variable (f : ℝ → ℝ)
variable (x t : ℝ)

-- Conditions
def cond1 : Prop := ∀ x, f ((1 / 2) * x - 1) = 2 * x + 3
def cond2 : Prop := f t = 4

-- Theorem statement
theorem solve_for_t (h1 : cond1 f) (h2 : cond2 f t) : t = -3 / 4 := by
  sorry

end solve_for_t_l160_160314


namespace sufficient_not_necessary_l160_160084

noncomputable def f (x a : ℝ) := x^2 - 2*a*x + 1

def no_real_roots (a : ℝ) : Prop := 4*a^2 - 4 < 0

def non_monotonic_interval (a m : ℝ) : Prop := m < a ∧ a < m + 3

def A := {a : ℝ | -1 < a ∧ a < 1}
def B (m : ℝ) := {a : ℝ | m < a ∧ a < m + 3}

theorem sufficient_not_necessary (x : ℝ) (m : ℝ) :
  (x ∈ A → x ∈ B m) → (A ⊆ B m) ∧ (exists a : ℝ, a ∈ B m ∧ a ∉ A) →
  -2 ≤ m ∧ m ≤ -1 := by 
  sorry

end sufficient_not_necessary_l160_160084


namespace firefighter_remaining_money_correct_l160_160022

noncomputable def firefighter_weekly_earnings : ℕ := 30 * 48
noncomputable def firefighter_monthly_earnings : ℕ := firefighter_weekly_earnings * 4
noncomputable def firefighter_rent_expense : ℕ := firefighter_monthly_earnings / 3
noncomputable def firefighter_food_expense : ℕ := 500
noncomputable def firefighter_tax_expense : ℕ := 1000
noncomputable def firefighter_total_expenses : ℕ := firefighter_rent_expense + firefighter_food_expense + firefighter_tax_expense
noncomputable def firefighter_remaining_money : ℕ := firefighter_monthly_earnings - firefighter_total_expenses

theorem firefighter_remaining_money_correct :
  firefighter_remaining_money = 2340 :=
by 
  rfl

end firefighter_remaining_money_correct_l160_160022


namespace salary_of_E_l160_160066

theorem salary_of_E (A B C D E : ℕ) (avg_salary : ℕ) 
  (hA : A = 8000) 
  (hB : B = 5000) 
  (hC : C = 11000) 
  (hD : D = 7000) 
  (h_avg : avg_salary = 8000) 
  (h_total_avg : avg_salary * 5 = A + B + C + D + E) : 
  E = 9000 :=
by {
  sorry
}

end salary_of_E_l160_160066


namespace equation_of_parallel_line_l160_160073

noncomputable def line_parallel_and_intercept (m : ℝ) : Prop :=
  (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ z : ℝ, 3*z + m = 0)

theorem equation_of_parallel_line {m : ℝ} :
  line_parallel_and_intercept m ↔ (∃ x y : ℝ, x + y + 2 = 0) ∨ (∃ x y : ℝ, x + y - 2 = 0) :=
by
  sorry

end equation_of_parallel_line_l160_160073


namespace intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l160_160065

theorem intersect_at_point_m_eq_1_3_n_eq_neg_73_9 
  (m : ℚ) (n : ℚ) : 
  (m^2 + 8 + n = 0) ∧ (3*m - 1 = 0) → 
  (m = 1/3 ∧ n = -73/9) := 
by 
  sorry

theorem lines_parallel_pass_through 
  (m : ℚ) (n : ℚ) :
  (m ≠ 0) → (m^2 = 16) ∧ (3*m - 8 + n = 0) → 
  (m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20) :=
by 
  sorry

theorem lines_perpendicular_y_intercept 
  (m : ℚ) (n : ℚ) :
  (m = 0 ∧ 8*(-1) + n = 0) → 
  (m = 0 ∧ n = 8) :=
by 
  sorry

end intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l160_160065


namespace ab_cd_value_l160_160654

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 0) : 
  a * b + c * d = -31 :=
by
  sorry

end ab_cd_value_l160_160654


namespace distance_from_mountains_l160_160973

/-- Given distances and scales from the problem description -/
def distance_between_mountains_map : ℤ := 312 -- in inches
def actual_distance_between_mountains : ℤ := 136 -- in km
def scale_A : ℤ := 1 -- 1 inch represents 1 km
def scale_B : ℤ := 2 -- 1 inch represents 2 km
def distance_from_mountain_A_map : ℤ := 25 -- in inches
def distance_from_mountain_B_map : ℤ := 40 -- in inches

/-- Prove the actual distances from Ram's camp to the mountains -/
theorem distance_from_mountains (dA dB : ℤ) :
  (dA = distance_from_mountain_A_map * scale_A) ∧ 
  (dB = distance_from_mountain_B_map * scale_B) :=
by {
  sorry -- Proof placeholder
}

end distance_from_mountains_l160_160973


namespace smallest_year_with_digit_sum_16_l160_160296

def sum_of_digits (n : Nat) : Nat :=
  let digits : List Nat := n.digits 10
  digits.foldl (· + ·) 0

theorem smallest_year_with_digit_sum_16 :
  ∃ (y : Nat), 2010 < y ∧ sum_of_digits y = 16 ∧
  (∀ (z : Nat), 2010 < z ∧ sum_of_digits z = 16 → z ≥ y) → y = 2059 :=
by
  sorry

end smallest_year_with_digit_sum_16_l160_160296


namespace smallest_number_diminished_by_10_l160_160953

theorem smallest_number_diminished_by_10 (x : ℕ) (h : ∀ n, x - 10 = 24 * n) : x = 34 := 
  sorry

end smallest_number_diminished_by_10_l160_160953


namespace problem_I_problem_II_l160_160167

theorem problem_I (a b p : ℝ) (F_2 M : ℝ × ℝ)
(h1 : a > b) (h2 : b > 0) (h3 : p > 0)
(h4 : (F_2.1)^2 / a^2 + (F_2.2)^2 / b^2 = 1)
(h5 : M.2^2 = 2 * p * M.1)
(h6 : M.1 = abs (M.2 - F_2.2) - 1)
(h7 : (|F_2.1 - 1|) = 5 / 2) :
    p = 2 ∧ ∃ f : ℝ × ℝ, (f.1)^2 / 9 + (f.2)^2 / 8 = 1 := sorry

theorem problem_II (k m x_0 : ℝ) 
(h8 : k ≠ 0) 
(h9 : m ≠ 0) 
(h10 : km = 1) 
(h11: ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2 = k * A.1 + m ∧ B.2 = k * B.1 + m) ∧
    ((A.1)^2 / 9 + (A.2)^2 / 8 = 1) ∧
    ((B.1)^2 / 9 + (B.2)^2 / 8 = 1) ∧
    (x_0 = (A.1 + B.1) / 2)) :
  -1 < x_0 ∧ x_0 < 0 := sorry

end problem_I_problem_II_l160_160167


namespace work_completion_time_l160_160537

theorem work_completion_time :
  let work_rate_A := 1 / 8
  let work_rate_B := 1 / 6
  let work_rate_C := 1 / 4.8
  (work_rate_A + work_rate_B + work_rate_C) = 1 / 2 :=
by
  sorry

end work_completion_time_l160_160537


namespace proof_problem_l160_160833

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

-- The two conditions
def condition1 (x y : ℝ) : Prop := f x + f y ≤ 0
def condition2 (x y : ℝ) : Prop := f x - f y ≥ 0

-- Equivalent description
def circle_condition (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 ≤ 8
def region1 (x y : ℝ) : Prop := y ≤ x ∧ y ≥ 6 - x
def region2 (x y : ℝ) : Prop := y ≥ x ∧ y ≤ 6 - x

-- The proof statement
theorem proof_problem (x y : ℝ) :
  (condition1 x y ∧ condition2 x y) ↔ 
  (circle_condition x y ∧ (region1 x y ∨ region2 x y)) :=
sorry

end proof_problem_l160_160833


namespace max_value_output_l160_160888

theorem max_value_output (a b c : ℝ) (h_a : a = 3) (h_b : b = 7) (h_c : c = 2) : max (max a b) c = 7 := 
by
  sorry

end max_value_output_l160_160888


namespace seashells_total_l160_160934

theorem seashells_total :
  let monday := 5
  let tuesday := 7 - 3
  let wednesday := (2 * monday) / 2
  let thursday := 3 * 7
  monday + tuesday + wednesday + thursday = 35 :=
by
  sorry

end seashells_total_l160_160934


namespace hyperbola_asymptotes_l160_160347

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (e : ℝ) (he : e = Real.sqrt 3) (h_eq : e = Real.sqrt ((a^2 + b^2) / a^2)) :
  (∀ x : ℝ, y = x * Real.sqrt 2) :=
by
  sorry

end hyperbola_asymptotes_l160_160347


namespace min_a2_plus_b2_l160_160576

theorem min_a2_plus_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_plus_b2_l160_160576


namespace jimmy_bought_3_pens_l160_160474

def cost_of_notebooks (num_notebooks : ℕ) (price_per_notebook : ℕ) : ℕ := num_notebooks * price_per_notebook
def cost_of_folders (num_folders : ℕ) (price_per_folder : ℕ) : ℕ := num_folders * price_per_folder
def total_cost (cost_notebooks cost_folders : ℕ) : ℕ := cost_notebooks + cost_folders
def total_spent (initial_money change : ℕ) : ℕ := initial_money - change
def cost_of_pens (total_spent amount_for_items : ℕ) : ℕ := total_spent - amount_for_items
def num_pens (cost_pens price_per_pen : ℕ) : ℕ := cost_pens / price_per_pen

theorem jimmy_bought_3_pens :
  let pen_price := 1
  let notebook_price := 3
  let num_notebooks := 4
  let folder_price := 5
  let num_folders := 2
  let initial_money := 50
  let change := 25
  let cost_notebooks := cost_of_notebooks num_notebooks notebook_price
  let cost_folders := cost_of_folders num_folders folder_price
  let total_items_cost := total_cost cost_notebooks cost_folders
  let amount_spent := total_spent initial_money change
  let pen_cost := cost_of_pens amount_spent total_items_cost
  num_pens pen_cost pen_price = 3 :=
by
  sorry

end jimmy_bought_3_pens_l160_160474


namespace problem_solution_l160_160550

theorem problem_solution (x : ℝ) :
  (⌊|x^2 - 1|⌋ = 10) ↔ (x ∈ Set.Ioc (-2 * Real.sqrt 3) (-Real.sqrt 11) ∪ Set.Ico (Real.sqrt 11) (2 * Real.sqrt 3)) :=
by
  sorry

end problem_solution_l160_160550


namespace total_cakes_served_l160_160274

theorem total_cakes_served (l : ℝ) (p : ℝ) (s : ℝ) (total_cakes_served_today : ℝ) :
  l = 48.5 → p = 0.6225 → s = 95 → total_cakes_served_today = 108 :=
by
  intros hl hp hs
  sorry

end total_cakes_served_l160_160274


namespace simplify_expression_l160_160267

theorem simplify_expression :
  (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 :=
by
  sorry

end simplify_expression_l160_160267


namespace remainder_101_pow_37_mod_100_l160_160754

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := 
by 
  sorry

end remainder_101_pow_37_mod_100_l160_160754


namespace geometric_sequence_a6_l160_160440

theorem geometric_sequence_a6 :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n) ∧ (a 4 * a 10 = 16) → (a 6 = 2) :=
by
  sorry

end geometric_sequence_a6_l160_160440


namespace andrew_bought_6_kg_of_grapes_l160_160912

def rate_grapes := 74
def rate_mangoes := 59
def kg_mangoes := 9
def total_paid := 975

noncomputable def number_of_kg_grapes := 6

theorem andrew_bought_6_kg_of_grapes :
  ∃ G : ℕ, (rate_grapes * G + rate_mangoes * kg_mangoes = total_paid) ∧ G = number_of_kg_grapes := 
by
  sorry

end andrew_bought_6_kg_of_grapes_l160_160912


namespace parabola_vertex_l160_160317

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end parabola_vertex_l160_160317


namespace number_of_marbles_removed_and_replaced_l160_160160

def bag_contains_red_marbles (r : ℕ) : Prop := r = 12
def total_marbles (t : ℕ) : Prop := t = 48
def probability_not_red_twice (r t : ℕ) : Prop := ((t - r) / t : ℝ) * ((t - r) / t) = 9 / 16

theorem number_of_marbles_removed_and_replaced (r t : ℕ)
  (hr : bag_contains_red_marbles r)
  (ht : total_marbles t)
  (hp : probability_not_red_twice r t) :
  2 = 2 := by
  sorry

end number_of_marbles_removed_and_replaced_l160_160160


namespace fill_tank_in_12_minutes_l160_160099

theorem fill_tank_in_12_minutes (rate1 rate2 rate_out : ℝ) 
  (h1 : rate1 = 1 / 18) (h2 : rate2 = 1 / 20) (h_out : rate_out = 1 / 45) : 
  12 = 1 / (rate1 + rate2 - rate_out) :=
by
  -- sorry will be replaced with the actual proof.
  sorry

end fill_tank_in_12_minutes_l160_160099


namespace sum_is_18_l160_160677

/-- Define the distinct non-zero digits, Hen, Xin, Chun, satisfying the given equation. -/
theorem sum_is_18 (Hen Xin Chun : ℕ) (h1 : Hen ≠ Xin) (h2 : Xin ≠ Chun) (h3 : Hen ≠ Chun)
  (h4 : 1 ≤ Hen ∧ Hen ≤ 9) (h5 : 1 ≤ Xin ∧ Xin ≤ 9) (h6 : 1 ≤ Chun ∧ Chun ≤ 9) :
  Hen + Xin + Chun = 18 :=
sorry

end sum_is_18_l160_160677


namespace relationship_between_x_y_z_l160_160614

theorem relationship_between_x_y_z (x y z : ℕ) (a b c d : ℝ)
  (h1 : x ≤ y ∧ y ≤ z)
  (h2 : (x:ℝ)^a = 70^d ∧ (y:ℝ)^b = 70^d ∧ (z:ℝ)^c = 70^d)
  (h3 : 1/a + 1/b + 1/c = 1/d) :
  x + y = z := 
sorry

end relationship_between_x_y_z_l160_160614


namespace expansive_sequence_in_interval_l160_160184

-- Definition of an expansive sequence
def expansive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (i j : ℕ), (i < j) → (|a i - a j| ≥ 1 / j)

-- Upper bound condition for C
def upper_bound_C (C : ℝ) : Prop :=
  C ≥ 2 * Real.log 2

-- The main statement combining both definitions into a proof problem
theorem expansive_sequence_in_interval (C : ℝ) (a : ℕ → ℝ) 
  (h_exp : expansive_sequence a) (h_bound : upper_bound_C C) :
  ∀ n, 0 ≤ a n ∧ a n ≤ C :=
sorry

end expansive_sequence_in_interval_l160_160184


namespace slope_of_parallel_line_l160_160478

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l160_160478


namespace abs_sum_fraction_le_sum_abs_fraction_l160_160731

variable (a b : ℝ)

theorem abs_sum_fraction_le_sum_abs_fraction (a b : ℝ) :
  (|a + b| / (1 + |a + b|)) ≤ (|a| / (1 + |a|)) + (|b| / (1 + |b|)) :=
sorry

end abs_sum_fraction_le_sum_abs_fraction_l160_160731


namespace total_wheels_in_garage_l160_160328

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l160_160328


namespace range_of_a_l160_160373

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) → (-1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l160_160373


namespace sin_cos_15_eq_1_over_4_l160_160308

theorem sin_cos_15_eq_1_over_4 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := 
by
  sorry

end sin_cos_15_eq_1_over_4_l160_160308


namespace smallest_N_constant_l160_160984

-- Define the property to be proven
theorem smallest_N_constant (a b c : ℝ) 
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) (h₄ : k = 0):
  (a^2 + b^2 + k) / c^2 > 1 / 2 :=
by
  sorry

end smallest_N_constant_l160_160984


namespace magnitude_of_complex_l160_160287

open Complex

theorem magnitude_of_complex :
  abs (Complex.mk (2/3) (-4/5)) = Real.sqrt 244 / 15 :=
by
  -- Placeholder for the actual proof
  sorry

end magnitude_of_complex_l160_160287


namespace prime_related_divisors_circle_l160_160476

variables (n : ℕ)

-- Definitions of prime-related and conditions for n
def is_prime (p: ℕ): Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p
def prime_related (a b : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ (a = p * b ∨ b = p * a)

-- The main statement to be proven
theorem prime_related_divisors_circle (n : ℕ) : 
  (n ≥ 3) ∧ (∀ a b, a ≠ b → (a ∣ n ∧ b ∣ n) → prime_related a b) ↔ ¬ (
    ∃ (p : ℕ) (k : ℕ), is_prime p ∧ (n = p ^ k) ∨ 
    ∃ (m : ℕ), n = m ^ 2 ) :=
sorry

end prime_related_divisors_circle_l160_160476


namespace f_at_2008_l160_160427

noncomputable def f : ℝ → ℝ := sorry
noncomputable def finv : ℝ → ℝ := sorry

axiom f_inverse : ∀ x, f (finv x) = x ∧ finv (f x) = x
axiom f_at_9 : f 9 = 18

theorem f_at_2008 : f 2008 = -1981 :=
by
  sorry

end f_at_2008_l160_160427


namespace ball_bounce_height_lt_one_l160_160565

theorem ball_bounce_height_lt_one :
  ∃ (k : ℕ), 15 * (1/3:ℝ)^k < 1 ∧ k = 3 := 
sorry

end ball_bounce_height_lt_one_l160_160565
