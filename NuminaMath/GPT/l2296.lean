import Mathlib

namespace NUMINAMATH_GPT_fuel_cost_is_50_cents_l2296_229613

-- Define the capacities of the tanks
def small_tank_capacity : ℕ := 60
def large_tank_capacity : ℕ := 60 * 3 / 2 -- 50% larger than small tank

-- Define the number of planes
def number_of_small_planes : ℕ := 2
def number_of_large_planes : ℕ := 2

-- Define the service charge per plane
def service_charge_per_plane : ℕ := 100
def total_service_charge : ℕ :=
  service_charge_per_plane * (number_of_small_planes + number_of_large_planes)

-- Define the total cost to fill all planes
def total_cost : ℕ := 550

-- Define the total fuel capacity
def total_fuel_capacity : ℕ :=
  number_of_small_planes * small_tank_capacity + number_of_large_planes * large_tank_capacity

-- Define the total fuel cost
def total_fuel_cost : ℕ := total_cost - total_service_charge

-- Define the fuel cost per liter
def fuel_cost_per_liter : ℕ :=
  total_fuel_cost / total_fuel_capacity

theorem fuel_cost_is_50_cents :
  fuel_cost_per_liter = 50 / 100 := by
sorry

end NUMINAMATH_GPT_fuel_cost_is_50_cents_l2296_229613


namespace NUMINAMATH_GPT_tethered_dog_area_comparison_l2296_229601

theorem tethered_dog_area_comparison :
  let fence_radius := 20
  let rope_length := 30
  let arrangement1_area := π * (rope_length ^ 2)
  let tether_distance := 12
  let arrangement2_effective_radius := rope_length - tether_distance
  let arrangement2_full_circle_area := π * (arrangement2_effective_radius ^ 2)
  let arrangement2_additional_area := (1 / 4) * π * (tether_distance ^ 2)
  let arrangement2_total_area := arrangement2_full_circle_area + arrangement2_additional_area
  (arrangement1_area - arrangement2_total_area) = 540 * π := 
by
  sorry

end NUMINAMATH_GPT_tethered_dog_area_comparison_l2296_229601


namespace NUMINAMATH_GPT_refills_count_l2296_229634

variable (spent : ℕ) (cost : ℕ)

theorem refills_count (h1 : spent = 40) (h2 : cost = 10) : spent / cost = 4 := 
by
  sorry

end NUMINAMATH_GPT_refills_count_l2296_229634


namespace NUMINAMATH_GPT_possible_point_counts_l2296_229627

theorem possible_point_counts (r b g : ℕ) (d_RB d_RG d_BG : ℕ) :
    r + b + g = 15 →
    r * b * d_RB = 51 →
    r * g * d_RG = 39 →
    b * g * d_BG = 1 →
    (r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3) :=
by {
    sorry
}

end NUMINAMATH_GPT_possible_point_counts_l2296_229627


namespace NUMINAMATH_GPT_mean_proportional_c_l2296_229698

theorem mean_proportional_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 27) (h3 : c^2 = a * b) : c = 9 := by
  sorry

end NUMINAMATH_GPT_mean_proportional_c_l2296_229698


namespace NUMINAMATH_GPT_solve_chris_age_l2296_229603

/-- 
The average of Amy's, Ben's, and Chris's ages is 12. Six years ago, Chris was the same age as Amy is now. In 3 years, Ben's age will be 3/4 of Amy's age at that time. 
How old is Chris now? 
-/
def chris_age : Prop := 
  ∃ (a b c : ℤ), 
    (a + b + c = 36) ∧
    (c - 6 = a) ∧ 
    (b + 3 = 3 * (a + 3) / 4) ∧
    (c = 17)

theorem solve_chris_age : chris_age := 
  by
    sorry

end NUMINAMATH_GPT_solve_chris_age_l2296_229603


namespace NUMINAMATH_GPT_total_points_of_three_players_l2296_229654

-- Definitions based on conditions
def points_tim : ℕ := 30
def points_joe : ℕ := points_tim - 20
def points_ken : ℕ := 2 * points_tim

-- Theorem statement for the total points scored by the three players
theorem total_points_of_three_players :
  points_tim + points_joe + points_ken = 100 :=
by
  -- Proof is to be provided
  sorry

end NUMINAMATH_GPT_total_points_of_three_players_l2296_229654


namespace NUMINAMATH_GPT_all_propositions_correct_l2296_229689

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem all_propositions_correct (m n : ℝ) (a b : α) (h1 : m ≠ 0) (h2 : a ≠ 0) : 
  (∀ (m : ℝ) (a b : α), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : α), (m - n) • a = m • a - n • a) ∧
  (∀ (m : ℝ) (a b : α), m • a = m • b → a = b) ∧
  (∀ (m n : ℝ) (a : α), m • a = n • a → m = n) :=
by {
  sorry
}

end NUMINAMATH_GPT_all_propositions_correct_l2296_229689


namespace NUMINAMATH_GPT_no_solution_in_positive_rationals_l2296_229663

theorem no_solution_in_positive_rationals (n : ℕ) (hn : n > 0) (x y : ℚ) (hx : x > 0) (hy : y > 0) :
  x + y + (1 / x) + (1 / y) ≠ 3 * n :=
sorry

end NUMINAMATH_GPT_no_solution_in_positive_rationals_l2296_229663


namespace NUMINAMATH_GPT_quadratic_roots_opposite_l2296_229675

theorem quadratic_roots_opposite (a : ℝ) (h : ∀ x1 x2 : ℝ, 
  (x1 + x2 = 0 ∧ x1 * x2 = a - 1) ∧
  (x1 - (-(x1)) = 0 ∧ x2 - x1 = 0)) :
  a = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_opposite_l2296_229675


namespace NUMINAMATH_GPT_value_of_a5_l2296_229677

variable (a : ℕ → ℕ)

-- The initial condition
axiom initial_condition : a 1 = 2

-- The recurrence relation
axiom recurrence_relation : ∀ n : ℕ, n ≠ 0 → n * a (n+1) = 2 * (n + 1) * a n

theorem value_of_a5 : a 5 = 160 := 
sorry

end NUMINAMATH_GPT_value_of_a5_l2296_229677


namespace NUMINAMATH_GPT_solve_inequality_l2296_229652

theorem solve_inequality (x : ℝ) : (2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2 : ℝ) (4 / 3) ∨ x ∈ Set.Icc (8 / 3) (6 : ℝ)) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2296_229652


namespace NUMINAMATH_GPT_sqrt_D_rational_sometimes_not_l2296_229682

-- Definitions and conditions
def D (a : ℤ) : ℤ := a^2 + (a + 2)^2 + (a * (a + 2))^2

-- The statement to prove
theorem sqrt_D_rational_sometimes_not (a : ℤ) : ∃ x : ℚ, x = Real.sqrt (D a) ∧ ¬(∃ y : ℤ, x = y) ∨ ∃ y : ℤ, Real.sqrt (D a) = y :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_D_rational_sometimes_not_l2296_229682


namespace NUMINAMATH_GPT_min_value_range_l2296_229655

noncomputable def f (a x : ℝ) := x^2 + a * x

theorem min_value_range (a : ℝ) :
  (∃x : ℝ, ∀y : ℝ, f a (f a x) ≥ f a (f a y)) ∧ (∀x : ℝ, f a x ≥ f a (-a / 2)) →
  a ≤ 0 ∨ a ≥ 2 := sorry

end NUMINAMATH_GPT_min_value_range_l2296_229655


namespace NUMINAMATH_GPT_ice_rink_rental_fee_l2296_229692

/-!
  # Problem:
  An ice skating rink charges $5 for admission and a certain amount to rent skates. 
  Jill can purchase a new pair of skates for $65. She would need to go to the rink 26 times 
  to justify buying the skates rather than renting a pair. How much does the rink charge to rent skates?
-/

/-- Lean statement of the problem. --/
theorem ice_rink_rental_fee 
  (admission_fee : ℝ) (skates_cost : ℝ) (num_visits : ℕ)
  (total_buying_cost : ℝ) (total_renting_cost : ℝ)
  (rental_fee : ℝ) :
  admission_fee = 5 ∧
  skates_cost = 65 ∧
  num_visits = 26 ∧
  total_buying_cost = skates_cost + (admission_fee * num_visits) ∧
  total_renting_cost = (admission_fee + rental_fee) * num_visits ∧
  total_buying_cost = total_renting_cost →
  rental_fee = 2.50 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ice_rink_rental_fee_l2296_229692


namespace NUMINAMATH_GPT_gcd_13924_27018_l2296_229626

theorem gcd_13924_27018 : Int.gcd 13924 27018 = 2 := 
  by
    sorry

end NUMINAMATH_GPT_gcd_13924_27018_l2296_229626


namespace NUMINAMATH_GPT_value_of_a_plus_b_l2296_229605

theorem value_of_a_plus_b (a b c : ℤ) 
    (h1 : a + b + c = 11)
    (h2 : a + b - c = 19)
    : a + b = 15 := 
by
    -- Mathematical details skipped
    sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l2296_229605


namespace NUMINAMATH_GPT_board_officer_election_l2296_229649

def num_ways_choose_officers (total_members : ℕ) (elect_officers : ℕ) : ℕ :=
  -- This will represent the number of ways to choose 4 officers given 30 members
  -- with the conditions on Alice, Bob, Chris, and Dana.
  if total_members = 30 ∧ elect_officers = 4 then
    358800 + 7800 + 7800 + 24
  else
    0

theorem board_officer_election : num_ways_choose_officers 30 4 = 374424 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_board_officer_election_l2296_229649


namespace NUMINAMATH_GPT_max_value_of_f_l2296_229691

noncomputable def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_f (a b : ℝ) (h_symmetric : ∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) :
  ∃ x : ℝ, f x a b = 16 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2296_229691


namespace NUMINAMATH_GPT_range_of_m_l2296_229645

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  (f (m * Real.sin θ) + f (1 - m) > 0) ↔ (m ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_m_l2296_229645


namespace NUMINAMATH_GPT_solve_equation_l2296_229690

theorem solve_equation (x y z t : ℤ) (h : x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0) : x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2296_229690


namespace NUMINAMATH_GPT_couple_ticket_cost_l2296_229648

variable (x : ℝ)

def single_ticket_cost : ℝ := 20
def total_sales : ℝ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16

theorem couple_ticket_cost :
  96 * single_ticket_cost + 16 * x = total_sales →
  x = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_couple_ticket_cost_l2296_229648


namespace NUMINAMATH_GPT_total_number_of_songs_is_30_l2296_229621

-- Define the number of country albums and pop albums
def country_albums : ℕ := 2
def pop_albums : ℕ := 3

-- Define the number of songs per album
def songs_per_album : ℕ := 6

-- Define the total number of albums
def total_albums : ℕ := country_albums + pop_albums

-- Define the total number of songs
def total_songs : ℕ := total_albums * songs_per_album

-- Prove that the total number of songs is 30
theorem total_number_of_songs_is_30 : total_songs = 30 := 
sorry

end NUMINAMATH_GPT_total_number_of_songs_is_30_l2296_229621


namespace NUMINAMATH_GPT_determine_B_l2296_229683

open Set

-- Define the universal set U and the sets A and B
variable (U A B : Set ℕ)

-- Definitions based on the problem conditions
def U_def : U = A ∪ B := 
  by sorry

def cond1 : (U = {1, 2, 3, 4, 5, 6, 7}) := 
  by sorry

def cond2 : (A ∩ (U \ B) = {2, 4, 6}) := 
  by sorry

-- The main statement
theorem determine_B (h1 : U = {1, 2, 3, 4, 5, 6, 7}) (h2 : A ∩ (U \ B) = {2, 4, 6}) : B = {1, 3, 5, 7} :=
  by sorry

end NUMINAMATH_GPT_determine_B_l2296_229683


namespace NUMINAMATH_GPT_number_of_intersections_l2296_229638

theorem number_of_intersections : ∃ (a_values : Finset ℚ), 
  ∀ a ∈ a_values, ∀ x y, y = 2 * x + a ∧ y = x^2 + 3 * a^2 ∧ x = 0 → 
  2 = a_values.card :=
by 
  sorry

end NUMINAMATH_GPT_number_of_intersections_l2296_229638


namespace NUMINAMATH_GPT_combined_weight_cats_l2296_229646

-- Define the weights of the cats
def weight_cat1 := 2
def weight_cat2 := 7
def weight_cat3 := 4

-- Prove the combined weight of the three cats is 13 pounds
theorem combined_weight_cats :
  weight_cat1 + weight_cat2 + weight_cat3 = 13 := by
  sorry

end NUMINAMATH_GPT_combined_weight_cats_l2296_229646


namespace NUMINAMATH_GPT_lunch_choices_l2296_229632

theorem lunch_choices (chickens drinks : ℕ) (h1 : chickens = 3) (h2 : drinks = 2) : chickens * drinks = 6 :=
by
  sorry

end NUMINAMATH_GPT_lunch_choices_l2296_229632


namespace NUMINAMATH_GPT_fg_equals_gf_l2296_229673

theorem fg_equals_gf (m n p q : ℝ) (h : m + q = n + p) : ∀ x : ℝ, (m * (p * x + q) + n = p * (m * x + n) + q) :=
by sorry

end NUMINAMATH_GPT_fg_equals_gf_l2296_229673


namespace NUMINAMATH_GPT_incorrect_statement_B_l2296_229609

axiom statement_A : ¬ (0 > 0 ∨ 0 < 0)
axiom statement_C : ∀ (q : ℚ), (∃ (m : ℤ), q = m) ∨ (∃ (a b : ℤ), b ≠ 0 ∧ q = a / b)
axiom statement_D : abs (0 : ℚ) = 0

theorem incorrect_statement_B : ¬ (∀ (q : ℚ), abs q ≥ 1 → abs 1 = abs q) := sorry

end NUMINAMATH_GPT_incorrect_statement_B_l2296_229609


namespace NUMINAMATH_GPT_determine_functions_l2296_229639

noncomputable def f : (ℝ → ℝ) := sorry

theorem determine_functions (f : ℝ → ℝ)
  (h_domain: ∀ x, 0 < x → 0 < f x)
  (h_eq: ∀ w x y z, 0 < w → 0 < x → 0 < y → 0 < z → w * x = y * z →
    (f w)^2 + (f x)^2 = (f (y^2) + f (z^2)) * (w^2 + x^2) / (y^2 + z^2)) :
  (∀ x, 0 < x → (f x = x ∨ f x = 1 / x)) :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_determine_functions_l2296_229639


namespace NUMINAMATH_GPT_correct_calculation_l2296_229651

theorem correct_calculation (x : ℝ) (h : (x / 2) + 45 = 85) : (2 * x) - 45 = 115 :=
by {
  -- Note: Proof steps are not needed, 'sorry' is used to skip the proof
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l2296_229651


namespace NUMINAMATH_GPT_largest_prime_17p_625_l2296_229635

theorem largest_prime_17p_625 (p : ℕ) (h_prime : Nat.Prime p) (h_sqrt : ∃ q, 17 * p + 625 = q^2) : p = 67 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_17p_625_l2296_229635


namespace NUMINAMATH_GPT_max_value_of_ex1_ex2_l2296_229633

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then exp x else -(x^3)

-- Define the function g
noncomputable def g (x a : ℝ) : ℝ := 
  f (f x) - a

-- Define the condition that g(x) = 0 has two distinct zeros
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0

-- Define the target function h
noncomputable def h (m : ℝ) : ℝ := 
  m^3 * exp (-m)

-- Statement of the final proof
theorem max_value_of_ex1_ex2 (a : ℝ) (hpos : 0 < a) (zeros : has_two_distinct_zeros a) :
  (∃ x1 x2 : ℝ, e^x1 * e^x2 = (27 : ℝ) / (exp 3) ∧ g x1 a = 0 ∧ g x2 a = 0) :=
sorry

end NUMINAMATH_GPT_max_value_of_ex1_ex2_l2296_229633


namespace NUMINAMATH_GPT_oranges_taken_l2296_229659

theorem oranges_taken (initial_oranges remaining_oranges taken_oranges : ℕ) 
  (h1 : initial_oranges = 60) 
  (h2 : remaining_oranges = 25) 
  (h3 : taken_oranges = initial_oranges - remaining_oranges) : 
  taken_oranges = 35 :=
by
  -- Proof is omitted, as instructed.
  sorry

end NUMINAMATH_GPT_oranges_taken_l2296_229659


namespace NUMINAMATH_GPT_total_students_went_to_concert_l2296_229614

/-- There are 12 buses and each bus took 57 students. We want to find out the total number of students who went to the concert. -/
theorem total_students_went_to_concert (num_buses : ℕ) (students_per_bus : ℕ) (total_students : ℕ) 
  (h1 : num_buses = 12) (h2 : students_per_bus = 57) (h3 : total_students = num_buses * students_per_bus) : 
  total_students = 684 := 
by
  sorry

end NUMINAMATH_GPT_total_students_went_to_concert_l2296_229614


namespace NUMINAMATH_GPT_sum_of_x_and_y_l2296_229687

theorem sum_of_x_and_y :
  ∀ (x y : ℚ), (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l2296_229687


namespace NUMINAMATH_GPT_max_cos_alpha_l2296_229660

open Real

-- Define the condition as a hypothesis
def cos_sum_eq (α β : ℝ) : Prop :=
  cos (α + β) = cos α + cos β

-- State the maximum value theorem
theorem max_cos_alpha (α β : ℝ) (h : cos_sum_eq α β) : ∃ α, cos α = sqrt 3 - 1 :=
by
  sorry   -- Proof is omitted

#check max_cos_alpha

end NUMINAMATH_GPT_max_cos_alpha_l2296_229660


namespace NUMINAMATH_GPT_cindy_age_l2296_229602

-- Define the ages involved
variables (C J M G : ℕ)

-- Define the conditions
def jan_age_condition : Prop := J = C + 2
def marcia_age_condition : Prop := M = 2 * J
def greg_age_condition : Prop := G = M + 2
def greg_age_known : Prop := G = 16

-- The statement we need to prove
theorem cindy_age : 
  jan_age_condition C J → 
  marcia_age_condition J M → 
  greg_age_condition M G → 
  greg_age_known G → 
  C = 5 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_cindy_age_l2296_229602


namespace NUMINAMATH_GPT_joel_age_when_dad_twice_l2296_229650

theorem joel_age_when_dad_twice (x joel_age dad_age: ℕ) (h₁: joel_age = 12) (h₂: dad_age = 47) 
(h₃: dad_age + x = 2 * (joel_age + x)) : joel_age + x = 35 :=
by
  rw [h₁, h₂] at h₃ 
  sorry

end NUMINAMATH_GPT_joel_age_when_dad_twice_l2296_229650


namespace NUMINAMATH_GPT_triangle_shape_l2296_229637

-- Let there be a triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively
variables (A B C : ℝ) (a b c : ℝ) (b_ne_1 : b ≠ 1)
          (h1 : (log (b) (C / A)) = (log (sqrt (b)) (2)))
          (h2 : (log (b) (sin B / sin A)) = (log (sqrt (b)) (2)))

-- Define the theorem that states the shape of the triangle
theorem triangle_shape : A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) :=
by
  -- Proof is provided in the solution, skipping proof here
  sorry

end NUMINAMATH_GPT_triangle_shape_l2296_229637


namespace NUMINAMATH_GPT_compare_expression_solve_inequality_l2296_229615

-- Part (1) Problem Statement in Lean 4
theorem compare_expression (x : ℝ) (h : x ≥ -1) : 
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) :=
by sorry

-- Part (2) Problem Statement in Lean 4
theorem solve_inequality (x a : ℝ) (ha : a < 0) : 
  (x^2 - a * x - 6 * a^2 > 0) ↔ (x < 3 * a ∨ x > -2 * a) :=
by sorry

end NUMINAMATH_GPT_compare_expression_solve_inequality_l2296_229615


namespace NUMINAMATH_GPT_evaluate_fraction_l2296_229653

theorem evaluate_fraction :
  (0.5^2 + 0.05^3) / 0.005^3 = 2000100 := by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2296_229653


namespace NUMINAMATH_GPT_f_f_minus_two_l2296_229665

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (1 + x⁻¹))

theorem f_f_minus_two : f (f (-2)) = -8 / 3 := by
  sorry

end NUMINAMATH_GPT_f_f_minus_two_l2296_229665


namespace NUMINAMATH_GPT_jayden_planes_l2296_229657

theorem jayden_planes (W : ℕ) (wings_per_plane : ℕ) (total_wings : W = 108) (wpp_pos : wings_per_plane = 2) :
  ∃ n : ℕ, n = W / wings_per_plane ∧ n = 54 :=
by
  sorry

end NUMINAMATH_GPT_jayden_planes_l2296_229657


namespace NUMINAMATH_GPT_pears_for_apples_l2296_229671

-- Define the costs of apples, oranges, and pears.
variables {cost_apples cost_oranges cost_pears : ℕ}

-- Condition 1: Ten apples cost the same as five oranges
axiom apples_equiv_oranges : 10 * cost_apples = 5 * cost_oranges

-- Condition 2: Three oranges cost the same as four pears
axiom oranges_equiv_pears : 3 * cost_oranges = 4 * cost_pears

-- Theorem: Tyler can buy 13 pears for the price of 20 apples
theorem pears_for_apples : 20 * cost_apples = 13 * cost_pears :=
sorry

end NUMINAMATH_GPT_pears_for_apples_l2296_229671


namespace NUMINAMATH_GPT_chantel_final_bracelets_l2296_229686

-- Definitions of the conditions in Lean
def initial_bracelets_7_days := 7 * 4
def after_school_giveaway := initial_bracelets_7_days - 8
def bracelets_10_days := 10 * 5
def total_after_10_days := after_school_giveaway + bracelets_10_days
def after_soccer_giveaway := total_after_10_days - 12
def crafting_club_bracelets := 4 * 6
def total_after_crafting_club := after_soccer_giveaway + crafting_club_bracelets
def weekend_trip_bracelets := 2 * 3
def total_after_weekend_trip := total_after_crafting_club + weekend_trip_bracelets
def final_total := total_after_weekend_trip - 10

-- Lean statement to prove the final total bracelets
theorem chantel_final_bracelets : final_total = 78 :=
by
  -- Note: The proof is not required, hence the sorry
  sorry

end NUMINAMATH_GPT_chantel_final_bracelets_l2296_229686


namespace NUMINAMATH_GPT_cost_of_song_book_l2296_229630

def cost_of_trumpet : ℝ := 145.16
def total_amount_spent : ℝ := 151.00

theorem cost_of_song_book : (total_amount_spent - cost_of_trumpet) = 5.84 := by
  sorry

end NUMINAMATH_GPT_cost_of_song_book_l2296_229630


namespace NUMINAMATH_GPT_scientific_notation_l2296_229628

theorem scientific_notation (n : ℕ) (h : n = 27000000) : 
  ∃ (m : ℝ) (e : ℤ), n = m * (10 : ℝ) ^ e ∧ m = 2.7 ∧ e = 7 :=
by 
  use 2.7 
  use 7
  sorry

end NUMINAMATH_GPT_scientific_notation_l2296_229628


namespace NUMINAMATH_GPT_incorrect_statement_l2296_229695

open Set

theorem incorrect_statement 
  (M : Set ℝ := {x : ℝ | 0 < x ∧ x < 1})
  (N : Set ℝ := {y : ℝ | 0 < y})
  (R : Set ℝ := univ) : M ∪ N ≠ R :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l2296_229695


namespace NUMINAMATH_GPT_scientific_notation_of_400000_l2296_229612

theorem scientific_notation_of_400000 :
  (400000: ℝ) = 4 * 10^5 :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_400000_l2296_229612


namespace NUMINAMATH_GPT_chef_earns_less_than_manager_l2296_229608

noncomputable def hourly_wage_manager : ℝ := 8.5
noncomputable def hourly_wage_dishwasher : ℝ := hourly_wage_manager / 2
noncomputable def hourly_wage_chef : ℝ := hourly_wage_dishwasher * 1.2
noncomputable def daily_bonus : ℝ := 5
noncomputable def overtime_multiplier : ℝ := 1.5
noncomputable def tax_rate : ℝ := 0.15

noncomputable def manager_hours : ℝ := 10
noncomputable def dishwasher_hours : ℝ := 6
noncomputable def chef_hours : ℝ := 12
noncomputable def standard_hours : ℝ := 8

noncomputable def compute_earnings (hourly_wage : ℝ) (hours_worked : ℝ) : ℝ :=
  let regular_hours := min standard_hours hours_worked
  let overtime_hours := max 0 (hours_worked - standard_hours)
  let regular_pay := regular_hours * hourly_wage
  let overtime_pay := overtime_hours * hourly_wage * overtime_multiplier
  let total_earnings_before_tax := regular_pay + overtime_pay + daily_bonus
  total_earnings_before_tax * (1 - tax_rate)

noncomputable def manager_earnings : ℝ := compute_earnings hourly_wage_manager manager_hours
noncomputable def dishwasher_earnings : ℝ := compute_earnings hourly_wage_dishwasher dishwasher_hours
noncomputable def chef_earnings : ℝ := compute_earnings hourly_wage_chef chef_hours

theorem chef_earns_less_than_manager : manager_earnings - chef_earnings = 18.78 := by
  sorry

end NUMINAMATH_GPT_chef_earns_less_than_manager_l2296_229608


namespace NUMINAMATH_GPT_op_correct_l2296_229694

-- Definition of the operation * for non-zero integers
def op (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 / b)

theorem op_correct (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 12) (h2 : a * b = 32) :
  op a b = 3 / 8 :=
by
  -- Proof, sorry for now
  sorry

end NUMINAMATH_GPT_op_correct_l2296_229694


namespace NUMINAMATH_GPT_average_speed_of_train_l2296_229656

theorem average_speed_of_train
  (distance1 : ℝ) (time1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 240) (h2 : time1 = 3) (h3 : stop_time = 0.5)
  (h4 : distance2 = 450) (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = 81.18 := 
sorry

end NUMINAMATH_GPT_average_speed_of_train_l2296_229656


namespace NUMINAMATH_GPT_find_integer_pairs_l2296_229678

theorem find_integer_pairs :
  { (m, n) : ℤ × ℤ | n^3 + m^3 + 231 = n^2 * m^2 + n * m } = {(4, 5), (5, 4)} :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l2296_229678


namespace NUMINAMATH_GPT_correct_calculation_result_l2296_229680

-- Define the conditions in Lean
variable (num : ℤ) (mistake_mult : ℤ) (result : ℤ)
variable (h_mistake : mistake_mult = num * 10) (h_result : result = 50)

-- The statement we want to prove
theorem correct_calculation_result 
  (h_mistake : mistake_mult = num * 10) 
  (h_result : result = 50) 
  (h_num_correct : num = result / 10) :
  (20 / num = 4) := sorry

end NUMINAMATH_GPT_correct_calculation_result_l2296_229680


namespace NUMINAMATH_GPT_count_five_letter_words_l2296_229606

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end NUMINAMATH_GPT_count_five_letter_words_l2296_229606


namespace NUMINAMATH_GPT_original_price_l2296_229658

theorem original_price (P : ℝ) 
  (h1 : 1.40 * P = P + 700) : P = 1750 :=
by sorry

end NUMINAMATH_GPT_original_price_l2296_229658


namespace NUMINAMATH_GPT_john_initial_candies_l2296_229693

theorem john_initial_candies : ∃ x : ℕ, (∃ (x3 : ℕ), x3 = ((x - 2) / 2) ∧ x3 = 6) ∧ x = 14 := by
  sorry

end NUMINAMATH_GPT_john_initial_candies_l2296_229693


namespace NUMINAMATH_GPT_chocolate_chip_more_than_raisin_l2296_229697

def chocolate_chip_yesterday : ℕ := 19
def chocolate_chip_morning : ℕ := 237
def raisin_cookies : ℕ := 231

theorem chocolate_chip_more_than_raisin : 
  (chocolate_chip_yesterday + chocolate_chip_morning) - raisin_cookies = 25 :=
by 
  sorry

end NUMINAMATH_GPT_chocolate_chip_more_than_raisin_l2296_229697


namespace NUMINAMATH_GPT_average_length_remaining_strings_l2296_229685

theorem average_length_remaining_strings 
  (T1 : ℕ := 6) (avg_length1 : ℕ := 80) 
  (T2 : ℕ := 2) (avg_length2 : ℕ := 70) :
  (6 * avg_length1 - 2 * avg_length2) / 4 = 85 := 
by
  sorry

end NUMINAMATH_GPT_average_length_remaining_strings_l2296_229685


namespace NUMINAMATH_GPT_stadium_breadth_l2296_229676

theorem stadium_breadth (P L B : ℕ) (h1 : P = 800) (h2 : L = 100) :
  2 * (L + B) = P → B = 300 :=
by
  sorry

end NUMINAMATH_GPT_stadium_breadth_l2296_229676


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_l2296_229672

theorem sum_of_three_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_l2296_229672


namespace NUMINAMATH_GPT_find_coordinates_of_P_l2296_229611

-- Define the conditions
variable (x y : ℝ)
def in_second_quadrant := x < 0 ∧ y > 0
def distance_to_x_axis := abs y = 7
def distance_to_y_axis := abs x = 3

-- Define the statement to be proved in Lean 4
theorem find_coordinates_of_P :
  in_second_quadrant x y ∧ distance_to_x_axis y ∧ distance_to_y_axis x → (x, y) = (-3, 7) :=
by
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l2296_229611


namespace NUMINAMATH_GPT_number_of_distinct_arrangements_l2296_229610

-- Given conditions: There are 7 items and we need to choose 4 out of these 7.
def binomial_coefficient (n k : ℕ) : ℕ :=
  (n.choose k)

-- Given condition: Calculate the number of sequences of arranging 4 selected items.
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- The statement in Lean 4 to prove that the number of distinct arrangements is 840.
theorem number_of_distinct_arrangements : binomial_coefficient 7 4 * factorial 4 = 840 :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_arrangements_l2296_229610


namespace NUMINAMATH_GPT_find_c_l2296_229662

theorem find_c (a b c : ℚ) (h1 : ∀ y : ℚ, 1 = a * (3 - 1)^2 + b * (3 - 1) + c) (h2 : ∀ y : ℚ, 4 = a * (1)^2 + b * (1) + c)
  (h3 : ∀ y : ℚ, 1 = a * (y - 1)^2 + 4) : c = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2296_229662


namespace NUMINAMATH_GPT_bus_speeds_l2296_229629

theorem bus_speeds (d t : ℝ) (s₁ s₂ : ℝ)
  (h₀ : d = 48)
  (h₁ : t = 1 / 6) -- 10 minutes in hours
  (h₂ : s₂ = s₁ - 4)
  (h₃ : d / s₂ - d / s₁ = t) :
  s₁ = 36 ∧ s₂ = 32 := 
sorry

end NUMINAMATH_GPT_bus_speeds_l2296_229629


namespace NUMINAMATH_GPT_tangent_line_proof_minimum_a_proof_l2296_229625

noncomputable def f (x : ℝ) := 2 * Real.log x - 3 * x^2 - 11 * x

def tangent_equation_correct : Prop :=
  let y := f 1
  let slope := (2 / 1 - 6 * 1 - 11)
  (slope = -15) ∧ (y = -14) ∧ (∀ x y, y = -15 * (x - 1) + -14 ↔ 15 * x + y - 1 = 0)

def minimum_a_correct : Prop :=
  ∃ a : ℤ, 
    (∀ x, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x - 2) ↔ (a = 2)

theorem tangent_line_proof : tangent_equation_correct := sorry

theorem minimum_a_proof : minimum_a_correct := sorry

end NUMINAMATH_GPT_tangent_line_proof_minimum_a_proof_l2296_229625


namespace NUMINAMATH_GPT_number_of_even_factors_of_n_l2296_229604

def n : ℕ := 2^4 * 3^3 * 5 * 7^2

theorem number_of_even_factors_of_n : 
  (∃ k : ℕ, n = 2^4 * 3^3 * 5 * 7^2 ∧ k = 96) → 
  ∃ count : ℕ, 
    count = 96 ∧ 
    (∀ m : ℕ, 
      (m ∣ n ∧ m % 2 = 0) ↔ 
      (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 2 ∧ m = 2^a * 3^b * 5^c * 7^d)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_even_factors_of_n_l2296_229604


namespace NUMINAMATH_GPT_avg_weight_of_22_boys_l2296_229624

theorem avg_weight_of_22_boys:
  let total_boys := 30
  let avg_weight_8 := 45.15
  let avg_weight_total := 48.89
  let total_weight_8 := 8 * avg_weight_8
  let total_weight_all := total_boys * avg_weight_total
  ∃ A : ℝ, A = 50.25 ∧ 22 * A + total_weight_8 = total_weight_all :=
by {
  sorry 
}

end NUMINAMATH_GPT_avg_weight_of_22_boys_l2296_229624


namespace NUMINAMATH_GPT_smallest_six_factors_l2296_229666

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end NUMINAMATH_GPT_smallest_six_factors_l2296_229666


namespace NUMINAMATH_GPT_radius_increase_l2296_229617

theorem radius_increase (C₁ C₂ : ℝ) (C₁_eq : C₁ = 30) (C₂_eq : C₂ = 40) :
  let r₁ := C₁ / (2 * Real.pi)
  let r₂ := C₂ / (2 * Real.pi)
  r₂ - r₁ = 5 / Real.pi :=
by
  simp [C₁_eq, C₂_eq]
  sorry

end NUMINAMATH_GPT_radius_increase_l2296_229617


namespace NUMINAMATH_GPT_solve_for_x_l2296_229664

theorem solve_for_x (x : ℂ) (h : 5 - 2 * I * x = 4 - 5 * I * x) : x = I / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2296_229664


namespace NUMINAMATH_GPT_max_distance_proof_area_of_coverage_ring_proof_l2296_229631

noncomputable def maxDistanceFromCenterToRadars : ℝ :=
  24 / Real.sin (Real.pi / 7)

noncomputable def areaOfCoverageRing : ℝ :=
  960 * Real.pi / Real.tan (Real.pi / 7)

theorem max_distance_proof :
  ∀ (r n : ℕ) (width : ℝ),  n = 7 → r = 26 → width = 20 → 
  maxDistanceFromCenterToRadars = 24 / Real.sin (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

theorem area_of_coverage_ring_proof :
  ∀ (r n : ℕ) (width : ℝ), n = 7 → r = 26 → width = 20 → 
  areaOfCoverageRing = 960 * Real.pi / Real.tan (Real.pi / 7) :=
by
  intros r n width hn hr hwidth
  sorry

end NUMINAMATH_GPT_max_distance_proof_area_of_coverage_ring_proof_l2296_229631


namespace NUMINAMATH_GPT_find_line_m_l2296_229636

noncomputable def reflect_point_across_line 
  (P : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ :=
  let line_vector := (a, b)
  let scaling_factor := -2 * ((a * P.1 + b * P.2 + c) / (a^2 + b^2))
  ((P.1 + scaling_factor * a), (P.2 + scaling_factor * b))

theorem find_line_m (P P'' : ℝ × ℝ) (a b : ℝ) (c : ℝ := 0)
  (h₁ : P = (2, -3))
  (h₂ : a * 1 + b * 4 = 0)
  (h₃ : P'' = (1, 4))
  (h₄ : reflect_point_across_line (reflect_point_across_line P a b c) a b c = P'') :
  4 * P''.1 - P''.2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_line_m_l2296_229636


namespace NUMINAMATH_GPT_find_amplitude_l2296_229688

-- Conditions
variables (a b c d : ℝ)

theorem find_amplitude
  (h1 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5)
  (h2 : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) :
  a = 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_amplitude_l2296_229688


namespace NUMINAMATH_GPT_acid_solution_l2296_229696

theorem acid_solution (x y : ℝ) (h1 : 0.3 * x + 0.1 * y = 90)
  (h2 : x + y = 600) : x = 150 ∧ y = 450 :=
by
  sorry

end NUMINAMATH_GPT_acid_solution_l2296_229696


namespace NUMINAMATH_GPT_find_BC_length_l2296_229616

theorem find_BC_length
  (area : ℝ) (AB AC : ℝ)
  (h_area : area = 10 * Real.sqrt 3)
  (h_AB : AB = 5)
  (h_AC : AC = 8) :
  ∃ BC : ℝ, BC = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_BC_length_l2296_229616


namespace NUMINAMATH_GPT_find_number_l2296_229618

theorem find_number (x : ℝ) : 61 + x * 12 / (180 / 3) = 62 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2296_229618


namespace NUMINAMATH_GPT_savings_per_month_l2296_229607

-- Define the monthly earnings, total needed for car, and total earnings
def monthly_earnings : ℤ := 4000
def total_needed_for_car : ℤ := 45000
def total_earnings : ℤ := 360000

-- Define the number of months it takes to save the required amount using total earnings and monthly earnings
def number_of_months : ℤ := total_earnings / monthly_earnings

-- Define the monthly savings based on the total needed and number of months
def monthly_savings : ℤ := total_needed_for_car / number_of_months

-- Prove that the monthly savings is £500
theorem savings_per_month : monthly_savings = 500 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_savings_per_month_l2296_229607


namespace NUMINAMATH_GPT_xyz_sum_neg1_l2296_229642

theorem xyz_sum_neg1 (x y z : ℝ) (h : (x + 1)^2 + |y - 2| = -(2 * x - z)^2) : x + y + z = -1 :=
sorry

end NUMINAMATH_GPT_xyz_sum_neg1_l2296_229642


namespace NUMINAMATH_GPT_red_pencil_count_l2296_229679

-- Definitions for provided conditions
def blue_pencils : ℕ := 20
def ratio : ℕ × ℕ := (5, 3)
def red_pencils (blue : ℕ) (rat : ℕ × ℕ) : ℕ := (blue / rat.fst) * rat.snd

-- Theorem statement
theorem red_pencil_count : red_pencils blue_pencils ratio = 12 := 
by
  sorry

end NUMINAMATH_GPT_red_pencil_count_l2296_229679


namespace NUMINAMATH_GPT_molecular_weight_CaO_is_56_08_l2296_229684

-- Define the atomic weights of Calcium and Oxygen
def atomic_weight_Ca := 40.08 -- in g/mol
def atomic_weight_O := 16.00 -- in g/mol

-- Define the molecular weight of the compound
def molecular_weight_CaO := atomic_weight_Ca + atomic_weight_O

-- State the theorem
theorem molecular_weight_CaO_is_56_08 : molecular_weight_CaO = 56.08 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_molecular_weight_CaO_is_56_08_l2296_229684


namespace NUMINAMATH_GPT_expected_plain_zongzi_picked_l2296_229620

-- Definitions and conditions:
def total_zongzi := 10
def red_bean_zongzi := 3
def meat_zongzi := 3
def plain_zongzi := 4

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probabilities
def P_X_0 : ℚ := (choose 6 2 : ℚ) / choose 10 2
def P_X_1 : ℚ := (choose 6 1 * choose 4 1 : ℚ) / choose 10 2
def P_X_2 : ℚ := (choose 4 2 : ℚ) / choose 10 2

-- Expected value of X
def E_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

theorem expected_plain_zongzi_picked : E_X = 4 / 5 := by
  -- Using the definition of E_X and the respective probabilities
  unfold E_X P_X_0 P_X_1 P_X_2
  -- Use the given formula to calculate the values
  -- Remaining steps would show detailed calculations leading to the answer
  sorry

end NUMINAMATH_GPT_expected_plain_zongzi_picked_l2296_229620


namespace NUMINAMATH_GPT_n_squared_divisible_by_36_l2296_229643

theorem n_squared_divisible_by_36 (n : ℕ) (h1 : 0 < n) (h2 : 6 ∣ n) : 36 ∣ n^2 := 
sorry

end NUMINAMATH_GPT_n_squared_divisible_by_36_l2296_229643


namespace NUMINAMATH_GPT_trees_not_pine_trees_l2296_229668

theorem trees_not_pine_trees
  (total_trees : ℕ)
  (percentage_pine : ℝ)
  (number_pine : ℕ)
  (number_not_pine : ℕ)
  (h_total : total_trees = 350)
  (h_percentage : percentage_pine = 0.70)
  (h_pine : number_pine = percentage_pine * total_trees)
  (h_not_pine : number_not_pine = total_trees - number_pine)
  : number_not_pine = 105 :=
sorry

end NUMINAMATH_GPT_trees_not_pine_trees_l2296_229668


namespace NUMINAMATH_GPT_value_of_m_has_positive_root_l2296_229661

theorem value_of_m_has_positive_root (x m : ℝ) (hx : x ≠ 3) :
    ((x + 5) / (x - 3) = 2 - m / (3 - x)) → x > 0 → m = 8 := 
sorry

end NUMINAMATH_GPT_value_of_m_has_positive_root_l2296_229661


namespace NUMINAMATH_GPT_train_cross_time_l2296_229669

theorem train_cross_time (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ) : 
  length_train = 100 →
  length_bridge = 150 →
  speed_kmph = 63 →
  (length_train + length_bridge) / (speed_kmph * (1000 / 3600)) = 14.29 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_l2296_229669


namespace NUMINAMATH_GPT_emery_reading_days_l2296_229641

theorem emery_reading_days (S E : ℕ) (h1 : E = S / 5) (h2 : (E + S) / 2 = 60) : E = 20 := by
  sorry

end NUMINAMATH_GPT_emery_reading_days_l2296_229641


namespace NUMINAMATH_GPT_proof_problem_l2296_229670

noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

theorem proof_problem : ((otimes (otimes 2 3) 4) - otimes 2 (otimes 3 4)) = -224/81 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2296_229670


namespace NUMINAMATH_GPT_ann_age_l2296_229619

variable (A T : ℕ)

-- Condition 1: Tom is currently two times older than Ann
def tom_older : Prop := T = 2 * A

-- Condition 2: The sum of their ages 10 years later will be 38
def age_sum_later : Prop := (A + 10) + (T + 10) = 38

-- Theorem: Ann's current age
theorem ann_age (h1 : tom_older A T) (h2 : age_sum_later A T) : A = 6 :=
by
  sorry

end NUMINAMATH_GPT_ann_age_l2296_229619


namespace NUMINAMATH_GPT_negate_existential_l2296_229667

theorem negate_existential :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + 4 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negate_existential_l2296_229667


namespace NUMINAMATH_GPT_ratio_of_pipe_lengths_l2296_229640

theorem ratio_of_pipe_lengths (L S : ℕ) (h1 : L + S = 177) (h2 : L = 118) (h3 : ∃ k : ℕ, L = k * S) : L / S = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_pipe_lengths_l2296_229640


namespace NUMINAMATH_GPT_inscribed_circle_ratio_l2296_229623

theorem inscribed_circle_ratio (a b c u v : ℕ) (h_triangle : a = 10 ∧ b = 24 ∧ c = 26) 
    (h_tangent_segments : u < v) (h_side_sum : u + v = a) : u / v = 2 / 3 :=
by
    sorry

end NUMINAMATH_GPT_inscribed_circle_ratio_l2296_229623


namespace NUMINAMATH_GPT_average_of_second_class_l2296_229681

variable (average1 : ℝ) (average2 : ℝ) (combined_average : ℝ) (n1 : ℕ) (n2 : ℕ)

theorem average_of_second_class
  (h1 : n1 = 25) 
  (h2 : average1 = 40) 
  (h3 : n2 = 30) 
  (h4 : combined_average = 50.90909090909091) 
  (h5 : n1 + n2 = 55) 
  (h6 : n2 * average2 = 55 * combined_average - n1 * average1) :
  average2 = 60 := by
  sorry

end NUMINAMATH_GPT_average_of_second_class_l2296_229681


namespace NUMINAMATH_GPT_parameterization_properties_l2296_229644

theorem parameterization_properties (a b c d : ℚ)
  (h1 : a * (-1) + b = -3)
  (h2 : c * (-1) + d = 5)
  (h3 : a * 2 + b = 4)
  (h4 : c * 2 + d = 15) :
  a^2 + b^2 + c^2 + d^2 = 790 / 9 :=
sorry

end NUMINAMATH_GPT_parameterization_properties_l2296_229644


namespace NUMINAMATH_GPT_math_problem_l2296_229647

theorem math_problem :
    (50 + 5 * (12 / (180 / 3))^2) * Real.sin (Real.pi / 6) = 25.1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2296_229647


namespace NUMINAMATH_GPT_harriet_smallest_stickers_l2296_229600

theorem harriet_smallest_stickers 
  (S : ℕ) (a b c : ℕ)
  (h1 : S = 5 * a + 3)
  (h2 : S = 11 * b + 3)
  (h3 : S = 13 * c + 3)
  (h4 : S > 3) :
  S = 718 :=
by
  sorry

end NUMINAMATH_GPT_harriet_smallest_stickers_l2296_229600


namespace NUMINAMATH_GPT_area_of_rectangle_l2296_229699

-- Define the given conditions
def length : Real := 5.9
def width : Real := 3
def expected_area : Real := 17.7

theorem area_of_rectangle : (length * width) = expected_area := 
by 
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2296_229699


namespace NUMINAMATH_GPT_count_numbers_with_digit_2_l2296_229674

def contains_digit_2 (n : Nat) : Prop :=
  n / 100 = 2 ∨ (n / 10 % 10) = 2 ∨ (n % 10) = 2

theorem count_numbers_with_digit_2 (N : Nat) (H : 200 ≤ N ∧ N ≤ 499) : 
  Nat.card {n // 200 ≤ n ∧ n ≤ 499 ∧ contains_digit_2 n} = 138 :=
by
  sorry

end NUMINAMATH_GPT_count_numbers_with_digit_2_l2296_229674


namespace NUMINAMATH_GPT_total_points_l2296_229622

theorem total_points (zach_points ben_points : ℝ) (h₁ : zach_points = 42.0) (h₂ : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
  by sorry

end NUMINAMATH_GPT_total_points_l2296_229622
