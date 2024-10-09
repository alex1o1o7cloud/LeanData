import Mathlib

namespace value_of_xyz_l664_66482

theorem value_of_xyz (x y z : ℂ) 
  (h1 : x * y + 5 * y = -20)
  (h2 : y * z + 5 * z = -20)
  (h3 : z * x + 5 * x = -20) :
  x * y * z = 80 := 
by
  sorry

end value_of_xyz_l664_66482


namespace exponent_problem_l664_66430

variable (x m n : ℝ)
variable (h1 : x^m = 3)
variable (h2 : x^n = 5)

theorem exponent_problem : x^(2 * m - 3 * n) = 9 / 125 :=
by 
  sorry

end exponent_problem_l664_66430


namespace converse_of_squared_positive_is_negative_l664_66497

theorem converse_of_squared_positive_is_negative (x : ℝ) :
  (∀ x : ℝ, x < 0 → x^2 > 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
sorry

end converse_of_squared_positive_is_negative_l664_66497


namespace maximum_value_is_16_l664_66438

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
(x^2 - 2 * x * y + 2 * y^2) * (x^2 - 2 * x * z + 2 * z^2) * (y^2 - 2 * y * z + 2 * z^2)

theorem maximum_value_is_16 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  maximum_value x y z ≤ 16 :=
by
  sorry

end maximum_value_is_16_l664_66438


namespace minimize_material_used_l664_66465

theorem minimize_material_used (r h : ℝ) (V : ℝ) (S : ℝ) 
  (volume_formula : π * r^2 * h = V) (volume_given : V = 27 * π) :
  ∃ r, r = 3 :=
by
  sorry

end minimize_material_used_l664_66465


namespace div_factorial_result_l664_66472

-- Define the given condition
def ten_fact : ℕ := 3628800

-- Define four factorial
def four_fact : ℕ := 4 * 3 * 2 * 1

-- State the theorem to be proved
theorem div_factorial_result : ten_fact / four_fact = 151200 :=
by
  -- Sorry is used to skip the proof, only the statement is provided
  sorry

end div_factorial_result_l664_66472


namespace benny_books_l664_66491

variable (B : ℕ) -- the number of books Benny had initially

theorem benny_books (h : B - 10 + 33 = 47) : B = 24 :=
sorry

end benny_books_l664_66491


namespace total_employees_in_buses_l664_66434

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l664_66434


namespace distance_from_D_to_plane_B1EF_l664_66420

theorem distance_from_D_to_plane_B1EF :
  let D := (0, 0, 0)
  let B₁ := (1, 1, 1)
  let E := (1, 1/2, 0)
  let F := (1/2, 1, 0)
  ∃ (d : ℝ), d = 1 := by
  sorry

end distance_from_D_to_plane_B1EF_l664_66420


namespace shelves_of_mystery_books_l664_66450

theorem shelves_of_mystery_books (total_books : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ) (M : ℕ) 
  (h_total_books : total_books = 54) 
  (h_picture_shelves : picture_shelves = 4) 
  (h_books_per_shelf : books_per_shelf = 6)
  (h_mystery_books : total_books - picture_shelves * books_per_shelf = M * books_per_shelf) :
  M = 5 :=
by
  sorry

end shelves_of_mystery_books_l664_66450


namespace num_teachers_in_Oxford_High_School_l664_66458

def classes : Nat := 15
def students_per_class : Nat := 20
def principals : Nat := 1
def total_people : Nat := 349

theorem num_teachers_in_Oxford_High_School : 
  ∃ (teachers : Nat), teachers = total_people - (classes * students_per_class + principals) :=
by
  use 48
  sorry

end num_teachers_in_Oxford_High_School_l664_66458


namespace unique_plants_count_1320_l664_66487

open Set

variable (X Y Z : Finset ℕ)

def total_plants_X : ℕ := 600
def total_plants_Y : ℕ := 480
def total_plants_Z : ℕ := 420
def shared_XY : ℕ := 60
def shared_YZ : ℕ := 70
def shared_XZ : ℕ := 80
def shared_XYZ : ℕ := 30

theorem unique_plants_count_1320 : X.card = total_plants_X →
                                Y.card = total_plants_Y →
                                Z.card = total_plants_Z →
                                (X ∩ Y).card = shared_XY →
                                (Y ∩ Z).card = shared_YZ →
                                (X ∩ Z).card = shared_XZ →
                                (X ∩ Y ∩ Z).card = shared_XYZ →
                                (X ∪ Y ∪ Z).card = 1320 := 
by {
  sorry
}

end unique_plants_count_1320_l664_66487


namespace inequality_holds_for_positive_reals_l664_66449

theorem inequality_holds_for_positive_reals (x y : ℝ) (m n : ℤ) 
  (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (1 - x^n)^m + (1 - y^m)^n ≥ 1 :=
sorry

end inequality_holds_for_positive_reals_l664_66449


namespace average_growth_rate_l664_66442

theorem average_growth_rate (x : ℝ) :
  (7200 * (1 + x)^2 = 8712) → x = 0.10 :=
by
  sorry

end average_growth_rate_l664_66442


namespace cost_of_item_D_is_30_usd_l664_66479

noncomputable def cost_of_item_D_in_usd (total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate : ℝ) : ℝ :=
  let total_spent_with_fee := total_spent * (1 + service_fee_rate)
  let item_D_cost_FC := total_spent_with_fee - items_ABC_spent
  item_D_cost_FC * exchange_rate

theorem cost_of_item_D_is_30_usd
  (total_spent : ℝ)
  (items_ABC_spent : ℝ)
  (tax_paid : ℝ)
  (service_fee_rate : ℝ)
  (exchange_rate : ℝ)
  (h_total_spent : total_spent = 500)
  (h_items_ABC_spent : items_ABC_spent = 450)
  (h_tax_paid : tax_paid = 60)
  (h_service_fee_rate : service_fee_rate = 0.02)
  (h_exchange_rate : exchange_rate = 0.5) :
  cost_of_item_D_in_usd total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate = 30 :=
by
  have h1 : total_spent * (1 + service_fee_rate) = 500 * 1.02 := sorry
  have h2 : 500 * 1.02 - 450 = 60 := sorry
  have h3 : 60 * 0.5 = 30 := sorry
  sorry

end cost_of_item_D_is_30_usd_l664_66479


namespace rhombus_longest_diagonal_l664_66431

theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (h_area : area = 192) (h_ratio : ratio = 4 / 3) :
  ∃ d1 d2 : ℝ, d1 / d2 = 4 / 3 ∧ (d1 * d2) / 2 = 192 ∧ d1 = 16 * Real.sqrt 2 :=
by
  sorry

end rhombus_longest_diagonal_l664_66431


namespace polygon_sides_l664_66404

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l664_66404


namespace remaining_budget_l664_66457

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l664_66457


namespace acetone_mass_percentage_O_l664_66447

-- Definition of atomic masses
def atomic_mass_C := 12.01
def atomic_mass_H := 1.008
def atomic_mass_O := 16.00

-- Definition of the molar mass of acetone
def molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + atomic_mass_O

-- Definition of mass percentage of oxygen in acetone
def mass_percentage_O_acetone := (atomic_mass_O / molar_mass_acetone) * 100

theorem acetone_mass_percentage_O : mass_percentage_O_acetone = 27.55 := by sorry

end acetone_mass_percentage_O_l664_66447


namespace converse_opposite_l664_66466

theorem converse_opposite (x y : ℝ) : (x + y = 0) → (y = -x) :=
by
  sorry

end converse_opposite_l664_66466


namespace minimum_guests_needed_l664_66427

theorem minimum_guests_needed (total_food : ℕ) (max_food_per_guest : ℕ) (guests_needed : ℕ) : 
  total_food = 323 → max_food_per_guest = 2 → guests_needed = Nat.ceil (323 / 2) → guests_needed = 162 :=
by
  intros
  sorry

end minimum_guests_needed_l664_66427


namespace good_eggs_collected_l664_66412

/-- 
Uncle Ben has 550 chickens on his farm, consisting of 49 roosters and the rest being hens. 
Out of these hens, there are three types:
1. Type A: 25 hens do not lay eggs at all.
2. Type B: 155 hens lay 2 eggs per day.
3. Type C: The remaining hens lay 4 eggs every three days.

Moreover, Uncle Ben found that 3% of the eggs laid by Type B and Type C hens go bad before being collected. 
Prove that the total number of good eggs collected by Uncle Ben after one day is 716.
-/
theorem good_eggs_collected 
    (total_chickens : ℕ) (roosters : ℕ) (typeA_hens : ℕ) (typeB_hens : ℕ) 
    (typeB_eggs_per_day : ℕ) (typeC_eggs_per_3days : ℕ) (percent_bad_eggs : ℚ) :
  total_chickens = 550 →
  roosters = 49 →
  typeA_hens = 25 →
  typeB_hens = 155 →
  typeB_eggs_per_day = 2 →
  typeC_eggs_per_3days = 4 →
  percent_bad_eggs = 0.03 →
  (total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day) - 
  round (percent_bad_eggs * ((total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day))) = 716 :=
by
  intros
  sorry

end good_eggs_collected_l664_66412


namespace geometric_sequence_value_l664_66439

theorem geometric_sequence_value 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_condition : a 4 * a 6 * a 8 * a 10 * a 12 = 32) :
  (a 10 ^ 2) / (a 12) = 2 :=
sorry

end geometric_sequence_value_l664_66439


namespace work_required_to_pump_liquid_l664_66480

/-- Calculation of work required to pump a liquid of density ρ out of a parabolic boiler. -/
theorem work_required_to_pump_liquid
  (ρ g H a : ℝ)
  (h_pos : 0 < H)
  (a_pos : 0 < a) :
  ∃ (A : ℝ), A = (π * ρ * g * H^3) / (6 * a^2) :=
by
  -- TODO: Provide the proof.
  sorry

end work_required_to_pump_liquid_l664_66480


namespace minimum_value_reciprocals_l664_66425

theorem minimum_value_reciprocals (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : 2 / Real.sqrt (a^2 + 4 * b^2) = Real.sqrt 2) :
  (1 / a^2 + 1 / b^2) = 9 / 2 :=
sorry

end minimum_value_reciprocals_l664_66425


namespace alice_study_time_for_average_75_l664_66486

variable (study_time : ℕ → ℚ)
variable (score : ℕ → ℚ)

def inverse_relation := ∀ n, study_time n * score n = 120

theorem alice_study_time_for_average_75
  (inverse_relation : inverse_relation study_time score)
  (study_time_1 : study_time 1 = 2)
  (score_1 : score 1 = 60)
  : study_time 2 = 4/3 := by
  sorry

end alice_study_time_for_average_75_l664_66486


namespace unique_solution_a_eq_sqrt3_l664_66451

theorem unique_solution_a_eq_sqrt3 (a : ℝ) :
  (∃! x : ℝ, x^2 - a * |x| + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt3_l664_66451


namespace period_of_sine_plus_cosine_l664_66496

noncomputable def period_sine_cosine_sum (b : ℝ) : ℝ :=
  2 * Real.pi / b

theorem period_of_sine_plus_cosine (b : ℝ) (hb : b = 3) :
  period_sine_cosine_sum b = 2 * Real.pi / 3 :=
by
  rw [hb]
  apply rfl

end period_of_sine_plus_cosine_l664_66496


namespace sum_of_reciprocals_of_squares_l664_66469

open Nat

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h_prod : a * b = 5) : 
  (1 : ℚ) / (a * a) + (1 : ℚ) / (b * b) = 26 / 25 :=
by
  -- proof steps skipping with sorry
  sorry

end sum_of_reciprocals_of_squares_l664_66469


namespace marked_price_percentage_fixed_l664_66499

-- Definitions based on the conditions
def discount_percentage : ℝ := 0.18461538461538467
def profit_percentage : ℝ := 0.06

-- The final theorem statement
theorem marked_price_percentage_fixed (CP MP SP : ℝ) 
  (h1 : SP = CP * (1 + profit_percentage))  
  (h2 : SP = MP * (1 - discount_percentage)) :
  (MP / CP - 1) * 100 = 30 := 
sorry

end marked_price_percentage_fixed_l664_66499


namespace smallest_prime_divisor_524_plus_718_l664_66488

theorem smallest_prime_divisor_524_plus_718 (x y : ℕ) (h1 : x = 5 ^ 24) (h2 : y = 7 ^ 18) :
  ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ p ∣ (x + y) :=
by
  sorry

end smallest_prime_divisor_524_plus_718_l664_66488


namespace average_weight_decrease_l664_66454

theorem average_weight_decrease 
  (weight_old_student : ℝ := 92) 
  (weight_new_student : ℝ := 72) 
  (number_of_students : ℕ := 5) : 
  (weight_old_student - weight_new_student) / ↑number_of_students = 4 :=
by 
  sorry

end average_weight_decrease_l664_66454


namespace scientific_notation_of_463_4_billion_l664_66483

theorem scientific_notation_of_463_4_billion :
  (463.4 * 10^9) = (4.634 * 10^11) := by
  sorry

end scientific_notation_of_463_4_billion_l664_66483


namespace total_lambs_l664_66413

-- Defining constants
def Merry_lambs : ℕ := 10
def Brother_lambs : ℕ := Merry_lambs + 3

-- Proving the total number of lambs
theorem total_lambs : Merry_lambs + Brother_lambs = 23 :=
  by
    -- The actual proof is omitted and a placeholder is put instead
    sorry

end total_lambs_l664_66413


namespace set_D_cannot_form_triangle_l664_66402

theorem set_D_cannot_form_triangle : ¬ (∃ a b c : ℝ, a = 2 ∧ b = 4 ∧ c = 6 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by {
  sorry
}

end set_D_cannot_form_triangle_l664_66402


namespace teachers_on_field_trip_l664_66437

-- Definitions for conditions in the problem
def number_of_students := 12
def cost_per_student_ticket := 1
def cost_per_adult_ticket := 3
def total_cost_of_tickets := 24

-- Main statement
theorem teachers_on_field_trip :
  ∃ (T : ℕ), number_of_students * cost_per_student_ticket + T * cost_per_adult_ticket = total_cost_of_tickets ∧ T = 4 :=
by
  use 4
  sorry

end teachers_on_field_trip_l664_66437


namespace quadrilateral_area_correct_l664_66448

noncomputable def area_of_quadrilateral (n : ℕ) (hn : n > 0) : ℚ :=
  (2 * n^3) / (4 * n^2 - 1)

theorem quadrilateral_area_correct (n : ℕ) (hn : n > 0) :
  ∃ area : ℚ, area = (2 * n^3) / (4 * n^2 - 1) :=
by
  use area_of_quadrilateral n hn
  sorry

end quadrilateral_area_correct_l664_66448


namespace cows_in_group_l664_66423

variable (c h : ℕ)

theorem cows_in_group (hcow : 4 * c + 2 * h = 2 * (c + h) + 18) : c = 9 := 
by 
  sorry

end cows_in_group_l664_66423


namespace emily_necklaces_for_friends_l664_66471

theorem emily_necklaces_for_friends (n b B : ℕ)
  (h1 : n = 26)
  (h2 : b = 2)
  (h3 : B = 52)
  (h4 : n * b = B) : 
  n = 26 :=
by
  sorry

end emily_necklaces_for_friends_l664_66471


namespace sum_faces_edges_vertices_l664_66463

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l664_66463


namespace hyperbola_standard_eq_line_eq_AB_l664_66470

noncomputable def fixed_points : (Real × Real) × (Real × Real) := ((-Real.sqrt 2, 0.0), (Real.sqrt 2, 0.0))

def locus_condition (P : Real × Real) (F1 F2 : Real × Real) : Prop :=
  abs (dist P F2 - dist P F1) = 2

def curve_E (P : Real × Real) : Prop :=
  (P.1 < 0) ∧ (P.1 * P.1 - P.2 * P.2 = 1)

theorem hyperbola_standard_eq :
  ∃ P : Real × Real, locus_condition P (fixed_points.1) (fixed_points.2) ↔ curve_E P :=
sorry

def line_intersects_hyperbola (P : Real × Real) (k : Real) : Prop :=
  P.2 = k * P.1 - 1 ∧ curve_E P

def dist_A_B (A B : Real × Real) : Real :=
  dist A B

theorem line_eq_AB :
  ∃ k : Real, k = -Real.sqrt 5 / 2 ∧
              ∃ A B : Real × Real, line_intersects_hyperbola A k ∧ 
              line_intersects_hyperbola B k ∧ 
              dist_A_B A B = 6 * Real.sqrt 3 ∧
              ∀ x y : Real, y = k * x - 1 ↔ x * (Real.sqrt 5/2) + y + 1 = 0 :=
sorry

end hyperbola_standard_eq_line_eq_AB_l664_66470


namespace at_least_one_gt_one_of_sum_gt_two_l664_66444

theorem at_least_one_gt_one_of_sum_gt_two (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 := 
by sorry

end at_least_one_gt_one_of_sum_gt_two_l664_66444


namespace ceiling_fraction_evaluation_l664_66478

theorem ceiling_fraction_evaluation :
  (Int.ceil ((19 : ℚ) / 8 - Int.ceil ((45 : ℚ) / 19)) / Int.ceil ((45 : ℚ) / 8 + Int.ceil ((8 * 19 : ℚ) / 45))) = 0 :=
by
  sorry

end ceiling_fraction_evaluation_l664_66478


namespace example_theorem_l664_66435

noncomputable def P (A : Set ℕ) : ℝ := sorry

variable (A1 A2 A3 : Set ℕ)

axiom prob_A1 : P A1 = 0.2
axiom prob_A2 : P A2 = 0.3
axiom prob_A3 : P A3 = 0.5

theorem example_theorem : P (A1 ∪ A2) ≤ 0.5 := 
by {
  sorry
}

end example_theorem_l664_66435


namespace initial_bacteria_count_l664_66411

theorem initial_bacteria_count (n : ℕ) : 
  (n * 4^10 = 4194304) → n = 4 :=
by
  sorry

end initial_bacteria_count_l664_66411


namespace convex_k_gons_count_l664_66461

noncomputable def number_of_convex_k_gons (n k : ℕ) : ℕ :=
  if h : n ≥ 2 * k then
    n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k))
  else
    0

theorem convex_k_gons_count (n k : ℕ) (h : n ≥ 2 * k) :
  number_of_convex_k_gons n k = n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k)) :=
by
  sorry

end convex_k_gons_count_l664_66461


namespace monotonic_increasing_interval_l664_66453

noncomputable def f (x : ℝ) : ℝ := sorry

theorem monotonic_increasing_interval :
  (∀ x Δx : ℝ, 0 < x → 0 < Δx → 
  (f (x + Δx) - f x) / Δx = (2 / (Real.sqrt (x + Δx) + Real.sqrt x)) - (1 / (x^2 + x * Δx))) →
  ∀ x : ℝ, 1 < x → (∃ ε > 0, ∀ y, x < y ∧ y < x + ε → f y > f x) :=
by
  intro hyp
  sorry

end monotonic_increasing_interval_l664_66453


namespace find_triples_l664_66475

theorem find_triples (a b c : ℕ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : a^3 + 9 * b^2 + 9 * c + 7 = 1997) :
  (a = 10 ∧ b = 10 ∧ c = 10) :=
by sorry

end find_triples_l664_66475


namespace solve_equation_125_eq_5_25_exp_x_min_2_l664_66421

theorem solve_equation_125_eq_5_25_exp_x_min_2 :
    ∃ x : ℝ, 125 = 5 * (25 : ℝ)^(x - 2) ∧ x = 3 := 
by
  sorry

end solve_equation_125_eq_5_25_exp_x_min_2_l664_66421


namespace trajectory_equation_minimum_AB_l664_66409

/-- Let a moving circle \( C \) passes through the point \( F(0, 1) \).
    The center of the circle \( C \), denoted as \( (x, y) \), is above the \( x \)-axis and the
    distance from \( (x, y) \) to \( F \) is greater than its distance to the \( x \)-axis by 1.
    We aim to prove that the trajectory of the center is \( x^2 = 4y \). -/
theorem trajectory_equation {x y : ℝ} (h : y > 0) (hCF : Real.sqrt (x^2 + (y - 1)^2) - y = 1) : 
  x^2 = 4 * y :=
sorry

/-- Suppose \( A \) and \( B \) are two distinct points on the curve \( x^2 = 4y \). 
    The tangents at \( A \) and \( B \) intersect at \( P \), and \( AP \perp BP \). 
    Then the minimum value of \( |AB| \) is 4. -/
theorem minimum_AB {x₁ x₂ : ℝ} 
  (h₁ : y₁ = (x₁^2) / 4) (h₂ : y₂ = (x₂^2) / 4)
  (h_perp : x₁ * x₂ = -4) : 
  ∃ (d : ℝ), d ≥ 0 ∧ d = 4 :=
sorry

end trajectory_equation_minimum_AB_l664_66409


namespace Fiona_Less_Than_Charles_l664_66401

noncomputable def percentDifference (a b : ℝ) : ℝ :=
  ((a - b) / a) * 100

theorem Fiona_Less_Than_Charles : percentDifference 600 (450 * 1.1) = 17.5 :=
by
  sorry

end Fiona_Less_Than_Charles_l664_66401


namespace camille_total_birds_count_l664_66418

theorem camille_total_birds_count :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry

end camille_total_birds_count_l664_66418


namespace speed_of_other_train_l664_66429

theorem speed_of_other_train (len1 len2 time : ℝ) (v1 v_other : ℝ) :
  len1 = 200 ∧ len2 = 300 ∧ time = 17.998560115190788 ∧ v1 = 40 →
  v_other = ((len1 + len2) / 1000) / (time / 3600) - v1 :=
by
  intros
  sorry

end speed_of_other_train_l664_66429


namespace integer_values_of_f_l664_66473

theorem integer_values_of_f (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a * b ≠ 1) : 
  ∃ k ∈ ({4, 7} : Finset ℕ), 
    (a^2 + b^2 + a * b) / (a * b - 1) = k := 
by
  sorry

end integer_values_of_f_l664_66473


namespace solution_correctness_l664_66481

theorem solution_correctness:
  ∀ (x1 : ℝ) (θ : ℝ), (θ = (5 * Real.pi / 13)) →
  (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) →
  ∃ (x2 : ℝ), (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ 
  (Real.sin x1 - 2 * Real.sin (x2 + θ) = -1) :=
by 
  intros x1 θ hθ hx1;
  sorry

end solution_correctness_l664_66481


namespace prob_lamp_first_factory_standard_prob_lamp_standard_l664_66405

noncomputable def P_B1 : ℝ := 0.35
noncomputable def P_B2 : ℝ := 0.50
noncomputable def P_B3 : ℝ := 0.15

noncomputable def P_B1_A : ℝ := 0.70
noncomputable def P_B2_A : ℝ := 0.80
noncomputable def P_B3_A : ℝ := 0.90

-- Question A
theorem prob_lamp_first_factory_standard : P_B1 * P_B1_A = 0.245 :=
by 
  sorry

-- Question B
theorem prob_lamp_standard : (P_B1 * P_B1_A) + (P_B2 * P_B2_A) + (P_B3 * P_B3_A) = 0.78 :=
by 
  sorry

end prob_lamp_first_factory_standard_prob_lamp_standard_l664_66405


namespace average_employees_per_week_l664_66467

variable (x : ℕ)

theorem average_employees_per_week (h1 : x + 200 > x)
                                   (h2 : x < 200)
                                   (h3 : 2 * 200 = 400) :
  (x + 200 + x + 200 + 200 + 400) / 4 = 250 := by
  sorry

end average_employees_per_week_l664_66467


namespace lowest_fraction_of_job_in_one_hour_l664_66456

-- Define the rates at which each person can work
def rate_A : ℚ := 1/3
def rate_B : ℚ := 1/4
def rate_C : ℚ := 1/6

-- Define the combined rates for each pair of people
def combined_rate_AB : ℚ := rate_A + rate_B
def combined_rate_AC : ℚ := rate_A + rate_C
def combined_rate_BC : ℚ := rate_B + rate_C

-- The Lean 4 statement to prove
theorem lowest_fraction_of_job_in_one_hour : min combined_rate_AB (min combined_rate_AC combined_rate_BC) = 5/12 :=
by 
  -- Here we state that the minimum combined rate is 5/12
  sorry

end lowest_fraction_of_job_in_one_hour_l664_66456


namespace min_value_expression_l664_66433

theorem min_value_expression (k x y z : ℝ) (hk : 0 < k) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x_min y_min z_min : ℝ, (0 < x_min) ∧ (0 < y_min) ∧ (0 < z_min) ∧
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    k * (4 * z / (2 * x + y) + 4 * x / (y + 2 * z) + y / (x + z))
    ≥ 3 * k) ∧
  k * (4 * z_min / (2 * x_min + y_min) + 4 * x_min / (y_min + 2 * z_min) + y_min / (x_min + z_min)) = 3 * k :=
by sorry

end min_value_expression_l664_66433


namespace polar_line_through_centers_l664_66493

-- Definition of the given circles in polar coordinates
def Circle1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def Circle2 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Statement of the problem
theorem polar_line_through_centers (ρ θ : ℝ) :
  (∃ c1 c2 : ℝ × ℝ, Circle1 c1.fst c1.snd ∧ Circle2 c2.fst c2.snd ∧ θ = Real.pi / 4) :=
sorry

end polar_line_through_centers_l664_66493


namespace infinite_series_sum_l664_66422

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 5) ^ (n + 1)) = 5 / 16 :=
sorry

end infinite_series_sum_l664_66422


namespace alfred_gain_percent_l664_66445

-- Definitions based on the conditions
def purchase_price : ℝ := 4700
def repair_costs : ℝ := 800
def selling_price : ℝ := 6000

-- Lean statement to prove gain percent
theorem alfred_gain_percent :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 := by
  sorry

end alfred_gain_percent_l664_66445


namespace quadratic_roots_shifted_l664_66455

theorem quadratic_roots_shifted (a b c : ℝ) (r s : ℝ) 
  (h1 : 4 * r ^ 2 + 2 * r - 9 = 0) 
  (h2 : 4 * s ^ 2 + 2 * s - 9 = 0) :
  c = 51 / 4 := by
  sorry

end quadratic_roots_shifted_l664_66455


namespace AH_HD_ratio_l664_66498

-- Given conditions
variables {A B C H D : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited D]
variables (BC : ℝ) (AC : ℝ) (angle_C : ℝ)
-- We assume the values provided in the problem
variables (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4)

-- Altitudes and orthocenter assumption, representing intersections at orthocenter H
variables (A D H : Type) -- Points to represent A, D, and orthocenter H

noncomputable def AH_H_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) : ℝ :=
  if BC = 6 ∧ AC = 4 * Real.sqrt 2 ∧ angle_C = Real.pi / 4 then 2 else 0

-- We need to prove the ratio AH:HD equals 2 given the conditions
theorem AH_HD_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) :
  AH_H_ratio BC AC angle_C BC_eq AC_eq angle_C_eq = 2 :=
by {
  -- the statement will be proved here
  sorry
}

end AH_HD_ratio_l664_66498


namespace quadratic_function_properties_l664_66440

theorem quadratic_function_properties :
  ∃ a : ℝ, ∃ f : ℝ → ℝ,
    (∀ x : ℝ, f x = a * (x + 1) ^ 2 - 2) ∧
    (f 1 = 10) ∧
    (f (-1) = -2) ∧
    (∀ x : ℝ, x > -1 → f x ≥ f (-1))
:=
by
  sorry

end quadratic_function_properties_l664_66440


namespace compare_minus_abs_val_l664_66403

theorem compare_minus_abs_val :
  -|(-8)| < -6 := 
sorry

end compare_minus_abs_val_l664_66403


namespace length_DE_l664_66495

open Classical

noncomputable def triangle_base_length (ABC_base : ℝ) : ℝ :=
15

noncomputable def is_parallel (DE BC : ℝ) : Prop :=
DE = BC

noncomputable def area_ratio (triangle_small triangle_large : ℝ) : ℝ :=
0.25

theorem length_DE 
  (ABC_base : ℝ)
  (DE : ℝ)
  (BC : ℝ)
  (triangle_small : ℝ)
  (triangle_large : ℝ)
  (h_base : triangle_base_length ABC_base = 15)
  (h_parallel : is_parallel DE BC)
  (h_area : area_ratio triangle_small triangle_large = 0.25)
  (h_similar : true):
  DE = 7.5 :=
by
  sorry

end length_DE_l664_66495


namespace field_dimension_solution_l664_66432

theorem field_dimension_solution (m : ℤ) (H1 : (3 * m + 11) * m = 100) : m = 5 :=
sorry

end field_dimension_solution_l664_66432


namespace mean_of_three_numbers_l664_66407

theorem mean_of_three_numbers (a : Fin 12 → ℕ) (x y z : ℕ) 
  (h1 : (Finset.univ.sum a) / 12 = 40)
  (h2 : ((Finset.univ.sum a) + x + y + z) / 15 = 50) :
  (x + y + z) / 3 = 90 := 
by
  sorry

end mean_of_three_numbers_l664_66407


namespace average_carnations_l664_66417

theorem average_carnations (c1 c2 c3 n : ℕ) (h1 : c1 = 9) (h2 : c2 = 14) (h3 : c3 = 13) (h4 : n = 3) :
  (c1 + c2 + c3) / n = 12 :=
by
  sorry

end average_carnations_l664_66417


namespace amc_problem_l664_66410

theorem amc_problem (a b : ℕ) (h : ∀ n : ℕ, 0 < n → a^n + n ∣ b^n + n) : a = b :=
sorry

end amc_problem_l664_66410


namespace evaluate_expression_l664_66492

theorem evaluate_expression : 3 * (3 * (3 * (3 + 2) + 2) + 2) + 2 = 161 := sorry

end evaluate_expression_l664_66492


namespace max_marks_l664_66484

theorem max_marks (M : ℝ) (h : 0.92 * M = 460) : M = 500 :=
by
  sorry

end max_marks_l664_66484


namespace arithmetic_sequence_sum_l664_66476

theorem arithmetic_sequence_sum (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (m : ℕ) 
  (h1 : S_n m = 0) (h2 : S_n (m - 1) = -2) (h3 : S_n (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l664_66476


namespace minimum_value_of_f_l664_66400

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ -1 / Real.exp 1) ∧ (∃ x : ℝ, x > 0 ∧ f x = -1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_f_l664_66400


namespace inequality_solution_l664_66436

theorem inequality_solution (x : ℝ) : (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) ∨ x = -3 :=
by
sorry

end inequality_solution_l664_66436


namespace train_length_l664_66415

theorem train_length (L : ℕ) 
  (h_tree : L / 120 = L / 200 * 200) 
  (h_platform : (L + 800) / 200 = L / 120) : 
  L = 1200 :=
by
  sorry

end train_length_l664_66415


namespace polygon_sides_l664_66494

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
by
  sorry

end polygon_sides_l664_66494


namespace slope_intercept_condition_l664_66443

theorem slope_intercept_condition (m b : ℚ) (h_m : m = 1/3) (h_b : b = -3/4) : -1 < m * b ∧ m * b < 0 := by
  sorry

end slope_intercept_condition_l664_66443


namespace evaluate_P_l664_66406

noncomputable def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

theorem evaluate_P (y : ℝ) (z : ℝ) (hz : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P 2 = -22 := by
  sorry

end evaluate_P_l664_66406


namespace alpha_identity_l664_66408

theorem alpha_identity (α : ℝ) (hα : α ≠ 0) (h_tan : Real.tan α = -α) : 
    (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := 
by
  sorry

end alpha_identity_l664_66408


namespace whale_tongue_weight_difference_l664_66426

noncomputable def tongue_weight_blue_whale_kg : ℝ := 2700
noncomputable def tongue_weight_fin_whale_kg : ℝ := 1800
noncomputable def kg_to_pounds : ℝ := 2.20462
noncomputable def ton_to_pounds : ℝ := 2000

noncomputable def tongue_weight_blue_whale_tons := (tongue_weight_blue_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def tongue_weight_fin_whale_tons := (tongue_weight_fin_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def weight_difference_tons := tongue_weight_blue_whale_tons - tongue_weight_fin_whale_tons

theorem whale_tongue_weight_difference :
  weight_difference_tons = 0.992079 :=
by
  sorry

end whale_tongue_weight_difference_l664_66426


namespace flat_fee_l664_66474

theorem flat_fee (f n : ℝ) (h1 : f + 4 * n = 320) (h2 : f + 7 * n = 530) : f = 40 := by
  -- Proof goes here
  sorry

end flat_fee_l664_66474


namespace solve_linear_equation_one_variable_with_parentheses_l664_66459

/--
Theorem: Solving a linear equation in one variable that contains parentheses
is equivalent to the process of:
1. Removing the parentheses,
2. Moving terms,
3. Combining like terms, and
4. Making the coefficient of the unknown equal to 1.

Given: a linear equation in one variable that contains parentheses
Prove: The process of solving it is to remove the parentheses, move terms, combine like terms, and make the coefficient of the unknown equal to 1.
-/
theorem solve_linear_equation_one_variable_with_parentheses
  (eq : String) :
  ∃ instructions : String,
    instructions = "remove the parentheses; move terms; combine like terms; make the coefficient of the unknown equal to 1" :=
by
  sorry

end solve_linear_equation_one_variable_with_parentheses_l664_66459


namespace visitors_equal_cats_l664_66477

-- Definition for conditions
def visitors_pets_cats (V C : ℕ) : Prop :=
  (∃ P : ℕ, P = 3 * V ∧ P = 3 * C)

-- Statement of the proof problem
theorem visitors_equal_cats {V C : ℕ}
  (h : visitors_pets_cats V C) : V = C :=
by sorry

end visitors_equal_cats_l664_66477


namespace find_k_value_l664_66490

theorem find_k_value (k : ℚ) :
  (∀ x y : ℚ, (x = 1/3 ∧ y = -8 → -3/4 - 3 * k * x = 7 * y)) → k = 55.25 :=
by
  sorry

end find_k_value_l664_66490


namespace quadratic_b_value_l664_66416

theorem quadratic_b_value (b m : ℝ) (h_b_pos : 0 < b) (h_quad_form : ∀ x, x^2 + b * x + 108 = (x + m)^2 - 4)
  (h_m_pos_sqrt : m = 4 * Real.sqrt 7 ∨ m = -4 * Real.sqrt 7) : b = 8 * Real.sqrt 7 :=
by
  sorry

end quadratic_b_value_l664_66416


namespace popton_school_bus_total_toes_l664_66464

-- Define the number of toes per hand for each race
def toes_per_hand_hoopit : ℕ := 3
def toes_per_hand_neglart : ℕ := 2
def toes_per_hand_zentorian : ℕ := 4

-- Define the number of hands for each race
def hands_per_hoopit : ℕ := 4
def hands_per_neglart : ℕ := 5
def hands_per_zentorian : ℕ := 6

-- Define the number of students from each race on the bus
def num_hoopits : ℕ := 7
def num_neglarts : ℕ := 8
def num_zentorians : ℕ := 5

-- Calculate the total number of toes on the bus
def total_toes_on_bus : ℕ :=
  num_hoopits * (toes_per_hand_hoopit * hands_per_hoopit) +
  num_neglarts * (toes_per_hand_neglart * hands_per_neglart) +
  num_zentorians * (toes_per_hand_zentorian * hands_per_zentorian)

-- Theorem stating the number of toes on the bus
theorem popton_school_bus_total_toes : total_toes_on_bus = 284 :=
by
  sorry

end popton_school_bus_total_toes_l664_66464


namespace find_b_l664_66424

open Real

theorem find_b (b : ℝ) (h : b + ⌈b⌉ = 21.5) : b = 10.5 :=
sorry

end find_b_l664_66424


namespace equation_of_parallel_line_l664_66462

theorem equation_of_parallel_line {x y : ℝ} :
  (∃ b : ℝ, ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 + b = 0)) ↔ 
  (∃ b : ℝ, b = -2 ∧ ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 - 2 = 0)) := 
by 
  sorry

end equation_of_parallel_line_l664_66462


namespace two_digit_numbers_satisfying_l664_66452

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_numbers_satisfying (n : ℕ) : 
  is_two_digit n → n = P n + S n ↔ (n % 10 = 9) :=
by
  sorry

end two_digit_numbers_satisfying_l664_66452


namespace work_days_B_l664_66460

theorem work_days_B (A B: ℕ) (work_per_day_B: ℕ) (total_days : ℕ) (total_units : ℕ) :
  (A = 2 * B) → (work_per_day_B = 1) → (total_days = 36) → (B = 1) → (total_units = total_days * (A + B)) → 
  total_units / work_per_day_B = 108 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_days_B_l664_66460


namespace find_integers_in_range_l664_66446

theorem find_integers_in_range :
  ∀ x : ℤ,
  (20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = 19) ↔
  x = 24 ∨ x = 29 ∨ x = 34 ∨ x = 39 ∨ x = 44 ∨ x = 49 :=
by sorry

end find_integers_in_range_l664_66446


namespace checkerboard_disc_coverage_l664_66489

/-- A circular disc with a diameter of 5 units is placed on a 10 x 10 checkerboard with each square having a side length of 1 unit such that the centers of both the disc and the checkerboard coincide.
    Prove that the number of checkerboard squares that are completely covered by the disc is 36. -/
theorem checkerboard_disc_coverage :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let side_length : ℝ := 1
  let board_size : ℕ := 10
  let disc_center : ℝ × ℝ := (board_size / 2, board_size / 2)
  ∃ (count : ℕ), count = 36 := 
  sorry

end checkerboard_disc_coverage_l664_66489


namespace total_team_points_l664_66414

theorem total_team_points :
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  (A + B + C + D + E + F + G + H = 22) :=
by
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  sorry

end total_team_points_l664_66414


namespace find_slope_l3_l664_66428

/-- Conditions --/
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def line2 (x y : ℝ) : Prop := y = 2
def A : Prod ℝ ℝ := (0, -3)
def area_ABC : ℝ := 5

noncomputable def B : Prod ℝ ℝ := (2, 2)  -- Simultaneous solution of line1 and line2

theorem find_slope_l3 (C : ℝ × ℝ) (slope_l3 : ℝ) :
  line2 C.1 C.2 ∧
  ((0 : ℝ), -3) ∈ {p : ℝ × ℝ | line1 p.1 p.2 → line2 p.1 p.2 } ∧
  C.2 = 2 ∧
  0 ≤ slope_l3 ∧
  area_ABC = 5 →
  slope_l3 = 5 / 4 :=
sorry

end find_slope_l3_l664_66428


namespace range_of_a_for_three_zeros_l664_66419

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end range_of_a_for_three_zeros_l664_66419


namespace other_divisor_l664_66468

theorem other_divisor (x : ℕ) (h1 : 266 % 33 = 2) (h2 : 266 % x = 2) : x = 132 :=
sorry

end other_divisor_l664_66468


namespace kaleb_gave_boxes_l664_66485

theorem kaleb_gave_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (given_boxes : ℕ)
  (h1 : total_boxes = 14) 
  (h2 : pieces_per_box = 6) 
  (h3 : pieces_left = 54) :
  given_boxes = 5 :=
by
  -- Add your proof here
  sorry

end kaleb_gave_boxes_l664_66485


namespace total_unique_handshakes_l664_66441

def num_couples := 8
def num_individuals := num_couples * 2
def potential_handshakes_per_person := num_individuals - 1 - 1
def total_handshakes := num_individuals * potential_handshakes_per_person / 2

theorem total_unique_handshakes : total_handshakes = 112 := sorry

end total_unique_handshakes_l664_66441
