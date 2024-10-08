import Mathlib

namespace fraction_passengers_from_asia_l211_211810

theorem fraction_passengers_from_asia (P : ℕ)
  (hP : P = 108)
  (frac_NA : ℚ) (frac_EU : ℚ) (frac_AF : ℚ)
  (Other_continents : ℕ)
  (h_frac_NA : frac_NA = 1/12)
  (h_frac_EU : frac_EU = 1/4)
  (h_frac_AF : frac_AF = 1/9)
  (h_Other_continents : Other_continents = 42) :
  (P * (1 - (frac_NA + frac_EU + frac_AF)) - Other_continents) / P = 1/6 :=
by
  sorry

end fraction_passengers_from_asia_l211_211810


namespace relationship_among_p_q_a_b_l211_211077

open Int

variables (a b p q : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = p) (h3 : Nat.lcm a b = q)

theorem relationship_among_p_q_a_b : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end relationship_among_p_q_a_b_l211_211077


namespace problem_statement_l211_211083

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end problem_statement_l211_211083


namespace hyperbola_asymptotes_l211_211056

theorem hyperbola_asymptotes (x y : ℝ) : x^2 - 4 * y^2 = -1 → (x = 2 * y) ∨ (x = -2 * y) := 
by
  intro h
  sorry

end hyperbola_asymptotes_l211_211056


namespace determine_y_minus_x_l211_211093

theorem determine_y_minus_x (x y : ℝ) (h1 : x + y = 360) (h2 : x / y = 3 / 5) : y - x = 90 := sorry

end determine_y_minus_x_l211_211093


namespace hyperbola_asymptote_l211_211596

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x, - y^2 = - x^2 / a^2 + 1) ∧ 
  (∀ x y, y + 2 * x = 0) → 
  a = 2 :=
by
  sorry

end hyperbola_asymptote_l211_211596


namespace infinite_colored_points_l211_211911

theorem infinite_colored_points
(P : ℤ → Prop) (red blue : ℤ → Prop)
(h_color : ∀ n : ℤ, (red n ∨ blue n))
(h_red_blue_partition : ∀ n : ℤ, ¬(red n ∧ blue n)) :
  ∃ (C : ℤ → Prop) (k : ℕ), (C = red ∨ C = blue) ∧ ∀ n : ℕ, ∃ m : ℤ, C m ∧ (m % n) = 0 :=
by
  sorry

end infinite_colored_points_l211_211911


namespace andrew_eggs_l211_211781

def andrew_eggs_problem (a b : ℕ) (half_eggs_given_away : ℚ) (remaining_eggs : ℕ) : Prop :=
  a + b - (a + b) * half_eggs_given_away = remaining_eggs

theorem andrew_eggs :
  andrew_eggs_problem 8 62 (1/2 : ℚ) 35 :=
by
  sorry

end andrew_eggs_l211_211781


namespace inequality_problem_l211_211811

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end inequality_problem_l211_211811


namespace floor_sqrt_225_l211_211316

theorem floor_sqrt_225 : Int.floor (Real.sqrt 225) = 15 := by
  sorry

end floor_sqrt_225_l211_211316


namespace ratio_january_february_l211_211231

variable (F : ℕ)

def total_savings := 19 + F + 8 

theorem ratio_january_february (h : total_savings F = 46) : 19 / F = 1 := by
  sorry

end ratio_january_february_l211_211231


namespace part1_part2_l211_211536

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem part1 {a : ℝ} :
  (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
sorry

theorem part2 :
  ∀ x > 0, Real.log x > (1 / Real.exp x) - (2 / (Real.exp 1) * x) :=
sorry

end part1_part2_l211_211536


namespace complete_the_square_l211_211177

-- Definition of the initial condition
def eq1 : Prop := ∀ x : ℝ, x^2 + 4 * x + 1 = 0

-- The goal is to prove if the initial condition holds, then the desired result holds.
theorem complete_the_square (x : ℝ) (h : x^2 + 4 * x + 1 = 0) : (x + 2)^2 = 3 := by
  sorry

end complete_the_square_l211_211177


namespace eval_expression_l211_211149

theorem eval_expression :
    (727 * 727) - (726 * 728) = 1 := by
  sorry

end eval_expression_l211_211149


namespace quadratic_eq_with_roots_l211_211335

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end quadratic_eq_with_roots_l211_211335


namespace Brenda_bakes_20_cakes_a_day_l211_211948

-- Define the conditions
variables (x : ℕ)

-- Other necessary definitions
def cakes_baked_in_9_days (x : ℕ) : ℕ := 9 * x
def cakes_after_selling_half (total_cakes : ℕ) : ℕ := total_cakes.div2

-- Given condition that Brenda has 90 cakes after selling half
def final_cakes_after_selling : ℕ := 90

-- Mathematical statement we want to prove
theorem Brenda_bakes_20_cakes_a_day (x : ℕ) (h : cakes_after_selling_half (cakes_baked_in_9_days x) = final_cakes_after_selling) : x = 20 :=
by sorry

end Brenda_bakes_20_cakes_a_day_l211_211948


namespace minimum_value_of_expression_l211_211054

variable (a b c : ℝ)

noncomputable def expression (a b c : ℝ) := (a + b) / c + (a + c) / b + (b + c) / a

theorem minimum_value_of_expression (hp1 : 0 < a) (hp2 : 0 < b) (hp3 : 0 < c) (h1 : a = 2 * b) (h2 : a = 2 * c) :
  expression a b c = 9.25 := 
sorry

end minimum_value_of_expression_l211_211054


namespace new_shape_perimeter_l211_211003

-- Definitions based on conditions
def square_side : ℕ := 64 / 4
def is_tri_isosceles (a b c : ℕ) : Prop := a = b

-- Definition of given problem setup and perimeter calculation
theorem new_shape_perimeter
  (side : ℕ)
  (tri_side1 tri_side2 base : ℕ)
  (h_square_side : side = 64 / 4)
  (h_tri1 : tri_side1 = side)
  (h_tri2 : tri_side2 = side)
  (h_base : base = side) :
  (side * 5) = 80 :=
by
  sorry

end new_shape_perimeter_l211_211003


namespace equation_of_parallel_line_l211_211833

-- Definitions for conditions from the problem
def point_A : ℝ × ℝ := (3, 2)
def line_eq (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def parallel_slope : ℝ := -4

-- Proof problem statement
theorem equation_of_parallel_line (x y : ℝ) :
  (∃ (m b : ℝ), m = parallel_slope ∧ b = 2 + 4 * 3 ∧ y = m * (x - 3) + b) →
  4 * x + y - 14 = 0 :=
sorry

end equation_of_parallel_line_l211_211833


namespace angle_ABC_tangent_circle_l211_211824

theorem angle_ABC_tangent_circle 
  (BAC ACB : ℝ)
  (h1 : BAC = 70)
  (h2 : ACB = 45)
  (D : Type)
  (incenter : ∀ D : Type, Prop)  -- Represent the condition that D is the incenter
  : ∃ ABC : ℝ, ABC = 65 :=
by
  sorry

end angle_ABC_tangent_circle_l211_211824


namespace find_y_interval_l211_211134

theorem find_y_interval (y : ℝ) (h : y^2 - 8 * y + 12 < 0) : 2 < y ∧ y < 6 :=
sorry

end find_y_interval_l211_211134


namespace temperature_equivalence_l211_211648

theorem temperature_equivalence (x : ℝ) (h : x = (9 / 5) * x + 32) : x = -40 :=
sorry

end temperature_equivalence_l211_211648


namespace factorize_expr1_factorize_expr2_l211_211729

-- Proof Problem 1
theorem factorize_expr1 (a : ℝ) : 
  (a^2 - 4 * a + 4 - 4 * (a - 2) + 4) = (a - 4)^2 :=
sorry

-- Proof Problem 2
theorem factorize_expr2 (x y : ℝ) : 
  16 * x^4 - 81 * y^4 = (4 * x^2 + 9 * y^2) * (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

end factorize_expr1_factorize_expr2_l211_211729


namespace Bobby_candy_chocolate_sum_l211_211448

/-
  Bobby ate 33 pieces of candy, then ate 4 more, and he also ate 14 pieces of chocolate.
  Prove that the total number of pieces of candy and chocolate he ate altogether is 51.
-/

theorem Bobby_candy_chocolate_sum :
  let initial_candy := 33
  let more_candy := 4
  let chocolate := 14
  let total_candy := initial_candy + more_candy
  total_candy + chocolate = 51 :=
by
  -- The theorem asserts the problem; apologies, the proof is not required here.
  sorry

end Bobby_candy_chocolate_sum_l211_211448


namespace each_episode_length_l211_211476

theorem each_episode_length (h_watch_time : ∀ d : ℕ, d = 5 → 2 * 60 * d = 600)
  (h_episodes : 20 > 0) : 600 / 20 = 30 := by
  -- Conditions used:
  -- 1. h_watch_time : John wants to finish a show in 5 days by watching 2 hours a day.
  -- 2. h_episodes : There are 20 episodes.
  -- Goal: Prove that each episode is 30 minutes long.
  sorry

end each_episode_length_l211_211476


namespace jason_gave_seashells_to_tim_l211_211748

-- Defining the conditions
def original_seashells : ℕ := 49
def current_seashells : ℕ := 36

-- The proof statement
theorem jason_gave_seashells_to_tim :
  original_seashells - current_seashells = 13 :=
by
  sorry

end jason_gave_seashells_to_tim_l211_211748


namespace expression_equals_24_l211_211121

-- Given values
def a := 7
def b := 4
def c := 1
def d := 7

-- Statement to prove
theorem expression_equals_24 : (a - b) * (c + d) = 24 := by
  sorry

end expression_equals_24_l211_211121


namespace union_complements_eq_l211_211439

-- Definitions for the universal set U and subsets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

-- Definition of the complements of A and B with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

-- The union of the two complements
def union_complements : Set ℕ := complement_U_A ∪ complement_U_B

-- The target proof statement
theorem union_complements_eq : union_complements = {1, 2, 3, 6, 7} := by
  sorry

end union_complements_eq_l211_211439


namespace max_blue_points_l211_211451

theorem max_blue_points (n : ℕ) (r b : ℕ)
  (h1 : n = 2009)
  (h2 : b + r = n)
  (h3 : ∀(k : ℕ), b ≤ k * (k - 1) / 2 → r ≥ k) :
  b = 1964 :=
by
  sorry

end max_blue_points_l211_211451


namespace hyperbola_eccentricity_l211_211955

theorem hyperbola_eccentricity 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (focus_on_x_axis : ∃ c : ℝ, c > 0)
  (asymptote_eq : ∀ x y : ℝ, (4 + 3 * y = 0) ∨ (4 - 3 * y = 0)) :
  ∃ e : ℝ, e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l211_211955


namespace wheel_revolutions_l211_211217

theorem wheel_revolutions (x y : ℕ) (h1 : y = x + 300)
  (h2 : 10 / (x : ℝ) = 10 / (y : ℝ) + 1 / 60) : 
  x = 300 ∧ y = 600 := 
by sorry

end wheel_revolutions_l211_211217


namespace interest_difference_l211_211924

theorem interest_difference :
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  CI - SI = 36 :=
by
  let P := 10000
  let R := 6
  let T := 2
  let SI := P * R * T / 100
  let CI := P * (1 + R / 100)^T - P
  show CI - SI = 36
  sorry

end interest_difference_l211_211924


namespace roots_of_equation_l211_211922

theorem roots_of_equation :
  (∃ x, (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 3 ∨ x = -4.5)) :=
by
  sorry

end roots_of_equation_l211_211922


namespace difference_between_neutrons_and_electrons_l211_211101

def proton_number : Nat := 118
def mass_number : Nat := 293

def number_of_neutrons : Nat := mass_number - proton_number
def number_of_electrons : Nat := proton_number

theorem difference_between_neutrons_and_electrons :
  (number_of_neutrons - number_of_electrons) = 57 := by
  sorry

end difference_between_neutrons_and_electrons_l211_211101


namespace num_routes_M_to_N_l211_211514

-- Define the relevant points and connections as predicates
def can_reach_directly (x y : String) : Prop :=
  if (x = "C" ∧ y = "N") ∨ (x = "D" ∧ y = "N") ∨ (x = "B" ∧ y = "N") then true else false

def can_reach_via (x y z : String) : Prop :=
  if (x = "A" ∧ y = "C" ∧ z = "N") ∨ (x = "A" ∧ y = "D" ∧ z = "N") ∨ (x = "B" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "B" ∧ y = "C" ∧ z = "N") ∨ (x = "E" ∧ y = "B" ∧ z = "N") ∨ (x = "F" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "F" ∧ y = "B" ∧ z = "N") then true else false

-- Define a function to compute the number of ways from a starting point to "N"
noncomputable def num_routes_to_N : String → ℕ
| "N" => 1
| "C" => 1
| "D" => 1
| "A" => 2 -- from C to N and D to N
| "B" => 4 -- from B to N directly, from B to N via A (2 ways), from B to N via C
| "E" => 4 -- from E to N via B
| "F" => 6 -- from F to N via A (2 ways), from F to N via B (4 ways)
| "M" => 16 -- from M to N via A, B, E, F
| _ => 0

-- The theorem statement
theorem num_routes_M_to_N : num_routes_to_N "M" = 16 :=
by
  sorry

end num_routes_M_to_N_l211_211514


namespace find_number_l211_211956

theorem find_number : ∃ n : ℝ, 50 + (5 * n) / (180 / 3) = 51 ∧ n = 12 := 
by
  use 12
  sorry

end find_number_l211_211956


namespace cost_of_baseball_is_correct_l211_211061

-- Define the costs and total amount spent
def cost_of_marbles : ℝ := 9.05
def cost_of_football : ℝ := 4.95
def total_amount_spent : ℝ := 20.52

-- Define the cost of the baseball
def cost_of_baseball : ℝ := total_amount_spent - (cost_of_marbles + cost_of_football)

-- The theorem we want to prove
theorem cost_of_baseball_is_correct :
  cost_of_baseball = 6.52 := by
  sorry

end cost_of_baseball_is_correct_l211_211061


namespace set_union_complement_l211_211322

-- Definitions based on provided problem statement
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}
def CRQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- The theorem to prove
theorem set_union_complement : P ∪ CRQ = {x | -2 < x ∧ x ≤ 3} :=
by
  -- Skip the proof
  sorry

end set_union_complement_l211_211322


namespace rectangle_area_same_width_l211_211676

theorem rectangle_area_same_width
  (square_area : ℝ) (area_eq : square_area = 36)
  (rect_width_eq_side : ℝ → ℝ → Prop) (width_eq : ∀ s, rect_width_eq_side s s)
  (rect_length_eq_3_times_width : ℝ → ℝ → Prop) (length_eq : ∀ w, rect_length_eq_3_times_width w (3 * w)) :
  (∃ s l w, s = 6 ∧ w = s ∧ l = 3 * w ∧ square_area = s * s ∧ rect_width_eq_side w s ∧ rect_length_eq_3_times_width w l ∧ w * l = 108) :=
by {
  sorry
}

end rectangle_area_same_width_l211_211676


namespace b_car_usage_hours_l211_211479

theorem b_car_usage_hours (h : ℕ) (total_cost_a_b_c : ℕ) 
  (a_usage : ℕ) (b_payment : ℕ) (c_usage : ℕ) 
  (total_cost : total_cost_a_b_c = 720)
  (usage_a : a_usage = 9) 
  (usage_c : c_usage = 13)
  (payment_b : b_payment = 225) 
  (cost_per_hour : ℝ := total_cost_a_b_c / (a_usage + h + c_usage)) :
  b_payment = cost_per_hour * h → h = 10 := 
by
  sorry

end b_car_usage_hours_l211_211479


namespace triangular_number_19_l211_211885

def triangular_number (n : Nat) : Nat :=
  (n + 1) * (n + 2) / 2

theorem triangular_number_19 : triangular_number 19 = 210 := by
  sorry

end triangular_number_19_l211_211885


namespace walkway_and_border_area_correct_l211_211515

-- Definitions based on the given conditions
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3
def walkway_width : ℕ := 2
def border_width : ℕ := 4
def num_rows : ℕ := 4
def num_columns : ℕ := 3

-- Total width calculation
def total_width : ℕ := 
  (flower_bed_width * num_columns) + (walkway_width * (num_columns + 1)) + (border_width * 2)

-- Total height calculation
def total_height : ℕ := 
  (flower_bed_height * num_rows) + (walkway_width * (num_rows + 1)) + (border_width * 2)

-- Total area of the garden including walkways and decorative border
def total_area : ℕ := total_width * total_height

-- Total area of flower beds
def flower_bed_area : ℕ := 
  (flower_bed_width * flower_bed_height) * (num_rows * num_columns)

-- Area of the walkways and decorative border
def walkway_and_border_area : ℕ := total_area - flower_bed_area

theorem walkway_and_border_area_correct : 
  walkway_and_border_area = 912 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end walkway_and_border_area_correct_l211_211515


namespace sum_of_numbers_l211_211298

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a - 2)^2 - (b - 2)^2 = 18): 
  a + b = -2 := 
by 
  sorry

end sum_of_numbers_l211_211298


namespace C_necessary_but_not_sufficient_for_A_l211_211864

variable {A B C : Prop}

-- Given conditions
def sufficient_not_necessary (h : A → B) (hn : ¬(B → A)) := h
def necessary_sufficient := B ↔ C

-- Prove that C is a necessary but not sufficient condition for A
theorem C_necessary_but_not_sufficient_for_A (h₁ : A → B) (hn : ¬(B → A)) (h₂ : B ↔ C) : (C → A) ∧ ¬(A → C) :=
  by
  sorry

end C_necessary_but_not_sufficient_for_A_l211_211864


namespace distance_interval_l211_211005

def distance_to_town (d : ℝ) : Prop :=
  ¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 9)

theorem distance_interval (d : ℝ) : distance_to_town d → d ∈ Set.Ioo 7 8 :=
by
  intro h
  have h1 : d < 8 := by sorry
  have h2 : d > 7 := by sorry
  rw [Set.mem_Ioo]
  exact ⟨h2, h1⟩

end distance_interval_l211_211005


namespace kanul_total_amount_l211_211649

theorem kanul_total_amount (T : ℝ) (R : ℝ) (M : ℝ) (C : ℝ)
  (hR : R = 80000)
  (hM : M = 30000)
  (hC : C = 0.2 * T)
  (hT : T = R + M + C) : T = 137500 :=
by {
  sorry
}

end kanul_total_amount_l211_211649


namespace matrix_count_l211_211457

-- A definition for the type of 3x3 matrices with 1's on the diagonal and * can be 0 or 1
def valid_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 0 = 1 ∧ 
  m 1 1 = 1 ∧ 
  m 2 2 = 1 ∧ 
  (m 0 1 = 0 ∨ m 0 1 = 1) ∧
  (m 0 2 = 0 ∨ m 0 2 = 1) ∧
  (m 1 0 = 0 ∨ m 1 0 = 1) ∧
  (m 1 2 = 0 ∨ m 1 2 = 1) ∧
  (m 2 0 = 0 ∨ m 2 0 = 1) ∧
  (m 2 1 = 0 ∨ m 2 1 = 1)

-- A definition to check that rows are distinct
def distinct_rows (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 ≠ m 1 ∧ m 1 ≠ m 2 ∧ m 0 ≠ m 2

-- Complete proof problem statement
theorem matrix_count : ∃ (n : ℕ), 
  (∀ m : Matrix (Fin 3) (Fin 3) ℕ, valid_matrix m → distinct_rows m) ∧ 
  n = 45 :=
by
  sorry

end matrix_count_l211_211457


namespace largest_n_crates_same_orange_count_l211_211528

theorem largest_n_crates_same_orange_count :
  ∀ (num_crates : ℕ) (min_oranges max_oranges : ℕ),
    num_crates = 200 →
    min_oranges = 100 →
    max_oranges = 130 →
    (∃ (n : ℕ), n = 7 ∧ (∃ (distribution : ℕ → ℕ), 
      (∀ x, min_oranges ≤ x ∧ x ≤ max_oranges) ∧ 
      (∀ x, distribution x ≤ num_crates ∧ 
          ∃ y, distribution y ≥ n))) := sorry

end largest_n_crates_same_orange_count_l211_211528


namespace find_number_of_ducks_l211_211105

variable {D H : ℕ}

-- Definition of the conditions
def total_animals (D H : ℕ) : Prop := D + H = 11
def total_legs (D H : ℕ) : Prop := 2 * D + 4 * H = 30
def number_of_ducks (D : ℕ) : Prop := D = 7

-- Lean statement for the proof problem
theorem find_number_of_ducks (D H : ℕ) (h1 : total_animals D H) (h2 : total_legs D H) : number_of_ducks D :=
by
  sorry

end find_number_of_ducks_l211_211105


namespace total_sheep_l211_211481

variable (x y : ℕ)
/-- Initial condition: After one ram runs away, the ratio of rams to ewes is 7:5. -/
def initial_ratio (x y : ℕ) : Prop := 5 * (x - 1) = 7 * y
/-- Second condition: After the ram returns and one ewe runs away, the ratio of rams to ewes is 5:3. -/
def second_ratio (x y : ℕ) : Prop := 3 * x = 5 * (y - 1)
/-- The total number of sheep in the flock initially is 25. -/
theorem total_sheep (x y : ℕ) 
  (h1 : initial_ratio x y) 
  (h2 : second_ratio x y) : 
  x + y = 25 := 
by sorry

end total_sheep_l211_211481


namespace theta_in_third_or_fourth_quadrant_l211_211919

-- Define the conditions as Lean definitions
def theta_condition (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * Real.pi + (-1 : ℝ)^(k + 1) * (Real.pi / 4)

-- Formulate the statement we need to prove
theorem theta_in_third_or_fourth_quadrant (θ : ℝ) (h : theta_condition θ) :
  ∃ q : ℤ, q = 3 ∨ q = 4 :=
sorry

end theta_in_third_or_fourth_quadrant_l211_211919


namespace geom_prog_all_integers_l211_211274

theorem geom_prog_all_integers (b : ℕ) (r : ℚ) (a c : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, b * r ^ n = a * n + c) ∧ ∃ b_1 : ℤ, b = b_1 →
  (∀ n : ℕ, ∃ b_n : ℤ, b * r ^ n = b_n) :=
by
  sorry

end geom_prog_all_integers_l211_211274


namespace cookies_baked_l211_211288

noncomputable def total_cookies (irin ingrid nell : ℚ) (percentage_ingrid : ℚ) : ℚ :=
  let total_ratio := irin + ingrid + nell
  let proportion_ingrid := ingrid / total_ratio
  let total_cookies := ingrid / (percentage_ingrid / 100)
  total_cookies

theorem cookies_baked (h_ratio: 9.18 + 5.17 + 2.05 = 16.4)
                      (h_percentage : 31.524390243902438 = 31.524390243902438) : 
  total_cookies 9.18 5.17 2.05 31.524390243902438 = 52 :=
by
  -- Placeholder for the proof.
  sorry

end cookies_baked_l211_211288


namespace smallest_integer_representable_l211_211286

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end smallest_integer_representable_l211_211286


namespace slope_of_l_l211_211986

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def parallel_lines (slope : ℝ) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, y = slope * x + m

def intersects_ellipse (slope : ℝ) : Prop :=
  parallel_lines slope ∧ ∃ x y : ℝ, ellipse x y ∧ y = slope * x + (y - slope * x)

theorem slope_of_l {l_slope : ℝ} :
  (∃ (m : ℝ) (x y : ℝ), intersects_ellipse (1 / 4) ∧ (y - l_slope * x = m)) →
  (l_slope = -2) :=
sorry

end slope_of_l_l211_211986


namespace age_of_new_person_l211_211294

theorem age_of_new_person (avg_age : ℝ) (x : ℝ) 
  (h1 : 10 * avg_age - (10 * (avg_age - 3)) = 42 - x) : 
  x = 12 := 
by
  sorry

end age_of_new_person_l211_211294


namespace total_cuts_length_eq_60_l211_211356

noncomputable def total_length_of_cuts (side_length : ℝ) (num_rectangles : ℕ) : ℝ :=
  if side_length = 36 ∧ num_rectangles = 3 then 60 else 0

theorem total_cuts_length_eq_60 :
  ∀ (side_length : ℝ) (num_rectangles : ℕ),
    side_length = 36 ∧ num_rectangles = 3 →
    total_length_of_cuts side_length num_rectangles = 60 := by
  intros
  simp [total_length_of_cuts]
  sorry

end total_cuts_length_eq_60_l211_211356


namespace thread_length_l211_211018

theorem thread_length (x : ℝ) (h : x + (3/4) * x = 21) : x = 12 :=
  sorry

end thread_length_l211_211018


namespace determine_f_peak_tourism_season_l211_211819

noncomputable def f (n : ℕ) : ℝ := 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300

theorem determine_f :
  (∀ n : ℕ, f n = 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300) ∧
  (f 8 - f 2 = 400) ∧
  (f 2 = 100) :=
sorry

theorem peak_tourism_season (n : ℤ) :
  (6 ≤ n ∧ n ≤ 10) ↔ (200 * Real.cos (((Real.pi / 6) * n) + 2 * Real.pi / 3) + 300 >= 400) :=
sorry

end determine_f_peak_tourism_season_l211_211819


namespace part1_part2_l211_211928

def f (x : ℝ) : ℝ := x^2 - 1
def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, |f x| = g x a → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) → a ≤ -2 :=
sorry

end part1_part2_l211_211928


namespace units_digit_k_squared_plus_2_k_is_7_l211_211523

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_k_squared_plus_2_k_is_7 : (k^2 + 2^k) % 10 = 7 :=
by sorry

end units_digit_k_squared_plus_2_k_is_7_l211_211523


namespace radius_excircle_ABC_l211_211250

variables (A B C P Q : Point)
variables (r_ABP r_APQ r_AQC : ℝ) (re_ABP re_APQ re_AQC : ℝ)
variable (r_ABC : ℝ)

-- Conditions
-- Radii of the incircles of triangles ABP, APQ, and AQC are all equal to 1
axiom incircle_ABP : r_ABP = 1
axiom incircle_APQ : r_APQ = 1
axiom incircle_AQC : r_AQC = 1

-- Radii of the corresponding excircles opposite A for ABP, APQ, and AQC are 3, 6, and 5 respectively
axiom excircle_ABP : re_ABP = 3
axiom excircle_APQ : re_APQ = 6
axiom excircle_AQC : re_AQC = 5

-- Radius of the incircle of triangle ABC is 3/2
axiom incircle_ABC : r_ABC = 3 / 2

-- Theorem stating the radius of the excircle of triangle ABC opposite A is 135
theorem radius_excircle_ABC (r_ABC : ℝ) : r_ABC = 3 / 2 → ∀ (re_ABC : ℝ), re_ABC = 135 := 
by
  intros 
  sorry

end radius_excircle_ABC_l211_211250


namespace farm_problem_l211_211519

variable (H R : ℕ)

-- Conditions
def initial_relation : Prop := R = H + 6
def hens_updated : Prop := H + 8 = 20
def current_roosters (H R : ℕ) : ℕ := R + 4

-- Theorem statement
theorem farm_problem (H R : ℕ)
  (h1 : initial_relation H R)
  (h2 : hens_updated H) :
  current_roosters H R = 22 :=
by
  sorry

end farm_problem_l211_211519


namespace linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l211_211504

theorem linear_function_passing_through_point_and_intersecting_another_line (
  k b : ℝ)
  (h1 : (∀ x y : ℝ, y = k * x + b → ((x = 3 ∧ y = -3) ∨ (x = 3/4 ∧ y = 0))))
  (h2 : (∀ x : ℝ, 0 = (4 * x - 3) → x = 3/4))
  : k = -4 / 3 ∧ b = 1 := 
sorry

theorem area_of_triangle (
  k b : ℝ)
  (h1 : k = -4 / 3 ∧ b = 1)
  : 1 / 2 * 3 / 4 * 1 = 3 / 8 := 
sorry

end linear_function_passing_through_point_and_intersecting_another_line_area_of_triangle_l211_211504


namespace bigger_number_in_ratio_l211_211459

theorem bigger_number_in_ratio (x : ℕ) (h : 11 * x = 143) : 8 * x = 104 :=
by
  sorry

end bigger_number_in_ratio_l211_211459


namespace total_winning_team_points_l211_211745

/-!
# Lean 4 Math Proof Problem

Prove that the total points scored by the winning team at the end of the game is 50 points given the conditions provided.
-/

-- Definitions
def losing_team_points_first_quarter : ℕ := 10
def winning_team_points_first_quarter : ℕ := 2 * losing_team_points_first_quarter
def winning_team_points_second_quarter : ℕ := winning_team_points_first_quarter + 10
def winning_team_points_third_quarter : ℕ := winning_team_points_second_quarter + 20

-- Theorem statement
theorem total_winning_team_points : winning_team_points_third_quarter = 50 :=
by
  sorry

end total_winning_team_points_l211_211745


namespace Connie_correct_result_l211_211881

theorem Connie_correct_result :
  ∀ x: ℝ, (200 - x = 100) → (200 + x = 300) :=
by
  intros x h
  have h1 : x = 100 := by linarith [h]
  rw [h1]
  linarith

end Connie_correct_result_l211_211881


namespace retail_price_of_washing_machine_l211_211617

variable (a : ℝ)

theorem retail_price_of_washing_machine :
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price 
  retail_price = 1.04 * a :=
by
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price
  sorry -- Proof skipped

end retail_price_of_washing_machine_l211_211617


namespace sqrt_div_value_l211_211484

open Real

theorem sqrt_div_value (n x : ℝ) (h1 : n = 3600) (h2 : sqrt n / x = 4) : x = 15 :=
by
  sorry

end sqrt_div_value_l211_211484


namespace Margarita_vs_Ricciana_l211_211211

-- Definitions based on the conditions.
def Ricciana_run : ℕ := 20
def Ricciana_jump : ℕ := 4
def Ricciana_total : ℕ := Ricciana_run + Ricciana_jump

def Margarita_run : ℕ := 18
def Margarita_jump : ℕ := 2 * Ricciana_jump - 1
def Margarita_total : ℕ := Margarita_run + Margarita_jump

-- The statement to be proved.
theorem Margarita_vs_Ricciana : (Margarita_total - Ricciana_total = 1) :=
by
  sorry

end Margarita_vs_Ricciana_l211_211211


namespace percentage_increase_of_numerator_l211_211556

theorem percentage_increase_of_numerator (N D : ℝ) (P : ℝ) (h1 : N / D = 0.75)
  (h2 : (N + (P / 100) * N) / (D - (8 / 100) * D) = 15 / 16) :
  P = 15 :=
sorry

end percentage_increase_of_numerator_l211_211556


namespace remainder_xyz_mod7_condition_l211_211480

-- Define variables and conditions
variables (x y z : ℕ)
theorem remainder_xyz_mod7_condition (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 7])
  (h2 : 3 * x + 2 * y + z ≡ 2 [MOD 7])
  (h3 : 2 * x + y + 3 * z ≡ 3 [MOD 7]) :
  (x * y * z % 7) ≡ 1 [MOD 7] := sorry

end remainder_xyz_mod7_condition_l211_211480


namespace area_of_curvilinear_trapezoid_steps_l211_211318

theorem area_of_curvilinear_trapezoid_steps (steps : List String) :
  (steps = ["division", "approximation", "summation", "taking the limit"]) :=
sorry

end area_of_curvilinear_trapezoid_steps_l211_211318


namespace f_4_1981_l211_211899

-- Define the function f with its properties
axiom f : ℕ → ℕ → ℕ

axiom f_0_y (y : ℕ) : f 0 y = y + 1
axiom f_x1_0 (x : ℕ) : f (x + 1) 0 = f x 1
axiom f_x1_y1 (x y : ℕ) : f (x + 1) (y + 1) = f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2 ^ 3964 - 3 :=
sorry

end f_4_1981_l211_211899


namespace prove_ordered_pair_l211_211334

-- Definition of the problem
def satisfies_equation1 (x y : ℚ) : Prop :=
  3 * x - 4 * y = -7

def satisfies_equation2 (x y : ℚ) : Prop :=
  7 * x - 3 * y = 5

-- Definition of the correct answer
def correct_answer (x y : ℚ) : Prop :=
  x = -133 / 57 ∧ y = 64 / 19

-- Main theorem to prove
theorem prove_ordered_pair :
  correct_answer (-133 / 57) (64 / 19) :=
by
  unfold correct_answer
  constructor
  { sorry }
  { sorry }

end prove_ordered_pair_l211_211334


namespace solve_for_x_l211_211572

theorem solve_for_x (x : ℝ) : 3^(3 * x) = Real.sqrt 81 -> x = 2 / 3 :=
by
  sorry

end solve_for_x_l211_211572


namespace number_of_cds_l211_211422

-- Define the constants
def total_money : ℕ := 37
def cd_price : ℕ := 14
def cassette_price : ℕ := 9

theorem number_of_cds (total_money cd_price cassette_price : ℕ) (h_total_money : total_money = 37) (h_cd_price : cd_price = 14) (h_cassette_price : cassette_price = 9) :
  ∃ n : ℕ, n * cd_price + cassette_price = total_money ∧ n = 2 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_cds_l211_211422


namespace total_net_gain_computation_l211_211952

noncomputable def house1_initial_value : ℝ := 15000
noncomputable def house2_initial_value : ℝ := 20000

noncomputable def house1_selling_price : ℝ := 1.15 * house1_initial_value
noncomputable def house2_selling_price : ℝ := 1.2 * house2_initial_value

noncomputable def house1_buy_back_price : ℝ := 0.85 * house1_selling_price
noncomputable def house2_buy_back_price : ℝ := 0.8 * house2_selling_price

noncomputable def house1_profit : ℝ := house1_selling_price - house1_buy_back_price
noncomputable def house2_profit : ℝ := house2_selling_price - house2_buy_back_price

noncomputable def total_net_gain : ℝ := house1_profit + house2_profit

theorem total_net_gain_computation : total_net_gain = 7387.5 :=
by
  sorry

end total_net_gain_computation_l211_211952


namespace derivative_evaluation_at_pi_over_3_l211_211303

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x) + Real.tan x

theorem derivative_evaluation_at_pi_over_3 :
  deriv f (Real.pi / 3) = 3 :=
sorry

end derivative_evaluation_at_pi_over_3_l211_211303


namespace smallest_number_of_tins_needed_l211_211967

variable (A : ℤ) (C : ℚ)

-- Conditions
def wall_area_valid : Prop := 1915 ≤ A ∧ A < 1925
def coverage_per_tin_valid : Prop := 17.5 ≤ C ∧ C < 18.5
def tins_needed_to_cover_wall (A : ℤ) (C : ℚ) : ℚ := A / C
def smallest_tins_needed : ℚ := 111

-- Proof problem statement
theorem smallest_number_of_tins_needed (A : ℤ) (C : ℚ)
    (h1 : wall_area_valid A)
    (h2 : coverage_per_tin_valid C)
    (h3 : 1915 ≤ A)
    (h4 : A < 1925)
    (h5 : 17.5 ≤ C)
    (h6 : C < 18.5) : 
  tins_needed_to_cover_wall A C + 1 ≥ smallest_tins_needed := by
    sorry

end smallest_number_of_tins_needed_l211_211967


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l211_211398

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l211_211398


namespace sum_of_roots_l211_211330

-- sum of roots of first polynomial
def S1 : ℚ := -(-6 / 3)

-- sum of roots of second polynomial
def S2 : ℚ := -(8 / 4)

-- proof statement
theorem sum_of_roots : S1 + S2 = 0 :=
by
  -- placeholders
  sorry

end sum_of_roots_l211_211330


namespace hares_cuts_l211_211186

-- Definitions representing the given conditions
def intermediates_fallen := 10
def end_pieces_fixed := 2
def total_logs := intermediates_fallen + end_pieces_fixed

-- Theorem statement
theorem hares_cuts : total_logs - 1 = 11 := by 
  sorry

end hares_cuts_l211_211186


namespace range_of_a_l211_211817

theorem range_of_a (a : ℝ) (h : ¬ (1^2 - 2*1 + a > 0)) : 1 ≤ a := sorry

end range_of_a_l211_211817


namespace parabola_equation_l211_211080

theorem parabola_equation (a : ℝ) : 
(∀ x y : ℝ, y = x → y = a * x^2)
∧ (∃ P : ℝ × ℝ, P = (2, 2) ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) 
  → A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ y₁ = x₁ ∧ y₂ = x₂ ∧ x₂ = x₁ → 
  ∃ f : ℝ × ℝ, f.fst ≠ 0 ∧ f.snd = 0) →
  a = (1 : ℝ) / 7 := 
sorry

end parabola_equation_l211_211080


namespace calculate_lunch_break_duration_l211_211045

noncomputable def paula_rate (p : ℝ) : Prop := p > 0
noncomputable def helpers_rate (h : ℝ) : Prop := h > 0
noncomputable def apprentice_rate (a : ℝ) : Prop := a > 0
noncomputable def lunch_break_duration (L : ℝ) : Prop := L >= 0

-- Monday's work equation
noncomputable def monday_work (p h a L : ℝ) (monday_work_done : ℝ) :=
  0.6 = (9 - L) * (p + h + a)

-- Tuesday's work equation
noncomputable def tuesday_work (h a L : ℝ) (tuesday_work_done : ℝ) :=
  0.3 = (7 - L) * (h + a)

-- Wednesday's work equation
noncomputable def wednesday_work (p a L : ℝ) (wednesday_work_done : ℝ) :=
  0.1 = (1.2 - L) * (p + a)

-- Final proof statement
theorem calculate_lunch_break_duration (p h a L : ℝ)
  (H1 : paula_rate p)
  (H2 : helpers_rate h)
  (H3 : apprentice_rate a)
  (H4 : lunch_break_duration L)
  (H5 : monday_work p h a L 0.6)
  (H6 : tuesday_work h a L 0.3)
  (H7 : wednesday_work p a L 0.1) :
  L = 1.4 :=
sorry

end calculate_lunch_break_duration_l211_211045


namespace total_jokes_l211_211123

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_jokes_l211_211123


namespace triangle_area_CO_B_l211_211625

-- Define the conditions as given in the problem
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def Q : Point := ⟨0, 15⟩

variable (p : ℝ)
def C : Point := ⟨0, p⟩
def B : Point := ⟨15, 0⟩

-- Prove the area of triangle COB is 15p / 2
theorem triangle_area_CO_B :
  p ≥ 0 → p ≤ 15 → 
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  area = (15 * p) / 2 := 
by
  intros hp0 hp15
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  have : area = (15 * p) / 2 := sorry
  exact this

end triangle_area_CO_B_l211_211625


namespace average_speed_l211_211341

theorem average_speed
  (distance1 : ℝ)
  (time1 : ℝ)
  (distance2 : ℝ)
  (time2 : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (average_speed : ℝ)
  (h1 : distance1 = 90)
  (h2 : time1 = 1)
  (h3 : distance2 = 50)
  (h4 : time2 = 1)
  (h5 : total_distance = distance1 + distance2)
  (h6 : total_time = time1 + time2)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 70 := 
sorry

end average_speed_l211_211341


namespace four_digit_number_exists_l211_211172

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), 
  B = 3 * A ∧ 
  C = A + B ∧ 
  D = 3 * B ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ 
  1000 * A + 100 * B + 10 * C + D = 1349 :=
by {
  sorry 
}

end four_digit_number_exists_l211_211172


namespace james_out_of_pocket_cost_l211_211675

-- Definitions
def doctor_charge : ℕ := 300
def insurance_coverage_percentage : ℝ := 0.80

-- Proof statement
theorem james_out_of_pocket_cost : (doctor_charge : ℝ) * (1 - insurance_coverage_percentage) = 60 := 
by sorry

end james_out_of_pocket_cost_l211_211675


namespace fraction_equals_decimal_l211_211188

theorem fraction_equals_decimal : (1 / 4 : ℝ) = 0.25 := 
sorry

end fraction_equals_decimal_l211_211188


namespace Jazmin_strip_width_l211_211152

theorem Jazmin_strip_width (a b c : ℕ) (ha : a = 44) (hb : b = 33) (hc : c = 55) : Nat.gcd (Nat.gcd a b) c = 11 := by
  sorry

end Jazmin_strip_width_l211_211152


namespace sanctuary_feeding_ways_l211_211292

/-- A sanctuary houses six different pairs of animals, each pair consisting of a male and female.
  The caretaker must feed the animals alternately by gender, meaning no two animals of the same gender 
  can be fed consecutively. Given the additional constraint that the male giraffe cannot be fed 
  immediately before the female giraffe and that the feeding starts with the male lion, 
  there are exactly 7200 valid ways to complete the feeding. -/
theorem sanctuary_feeding_ways : 
  ∃ ways : ℕ, ways = 7200 :=
by sorry

end sanctuary_feeding_ways_l211_211292


namespace stamp_blocks_inequalities_l211_211601

noncomputable def b (n : ℕ) : ℕ := sorry

theorem stamp_blocks_inequalities (n : ℕ) (m : ℕ) (hn : 0 < n) :
  ∃ c d : ℝ, c = 2 / 7 ∧ d = (4 * m^2 + 4 * m + 40) / 5 ∧
    (1 / 7 : ℝ) * n^2 - c * n ≤ b n ∧ 
    b n ≤ (1 / 5 : ℝ) * n^2 + d * n := 
  sorry

end stamp_blocks_inequalities_l211_211601


namespace inequality_ab_bc_ca_l211_211521

open Real

theorem inequality_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a) / (2 * (a + b + c))) := by
sorry

end inequality_ab_bc_ca_l211_211521


namespace total_students_l211_211474

-- Definitions based on the conditions:
def yoongi_left : ℕ := 7
def yoongi_right : ℕ := 5

-- Theorem statement that proves the total number of students given the conditions
theorem total_students (y_left y_right : ℕ) : y_left = yoongi_left -> y_right = yoongi_right -> (y_left + y_right - 1) = 11 := 
by
  intros h1 h2
  rw [h1, h2]
  sorry

end total_students_l211_211474


namespace ants_need_more_hours_l211_211613

theorem ants_need_more_hours (initial_sugar : ℕ) (removal_rate : ℕ) (hours_spent : ℕ) : 
  initial_sugar = 24 ∧ removal_rate = 4 ∧ hours_spent = 3 → 
  (initial_sugar - removal_rate * hours_spent) / removal_rate = 3 :=
by
  intro h
  sorry

end ants_need_more_hours_l211_211613


namespace largest_four_digit_negative_congruent_3_mod_29_l211_211469

theorem largest_four_digit_negative_congruent_3_mod_29 : 
  ∃ (n : ℤ), n < 0 ∧ n ≥ -9999 ∧ (n % 29 = 3) ∧ n = -1012 :=
sorry

end largest_four_digit_negative_congruent_3_mod_29_l211_211469


namespace scooter_travel_time_l211_211912

variable (x : ℝ)
variable (h_speed : x > 0)
variable (h_travel_time : (50 / (x - 1/2)) - (50 / x) = 3/4)

theorem scooter_travel_time : 50 / x = 50 / x := 
  sorry

end scooter_travel_time_l211_211912


namespace graphs_intersect_exactly_one_point_l211_211994

theorem graphs_intersect_exactly_one_point (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 5 * x + 4 = 2 * x - 6 → x = (7 / (2 * k))) ↔ k = (49 / 40) := 
by
  sorry

end graphs_intersect_exactly_one_point_l211_211994


namespace factorize_quadratic_trinomial_l211_211672

theorem factorize_quadratic_trinomial (t : ℝ) : t^2 - 10 * t + 25 = (t - 5)^2 :=
by
  sorry

end factorize_quadratic_trinomial_l211_211672


namespace calc1_calc2_l211_211963

variable (a b : ℝ) 

theorem calc1 : (-b)^2 * (-b)^3 * (-b)^5 = b^10 :=
by sorry

theorem calc2 : (2 * a * b^2)^3 = 8 * a^3 * b^6 :=
by sorry

end calc1_calc2_l211_211963


namespace fraction_calculation_l211_211728

-- Define the initial values of x and y
def x : ℚ := 4 / 6
def y : ℚ := 8 / 10

-- Statement to prove
theorem fraction_calculation : (6 * x^2 + 10 * y) / (60 * x * y) = 11 / 36 := by
  sorry

end fraction_calculation_l211_211728


namespace intersection_M_N_l211_211004

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

-- The theorem stating that the intersection of M and N is {1, 5}
theorem intersection_M_N :
  M ∩ N = {1, 5} :=
  sorry

end intersection_M_N_l211_211004


namespace quadratic_ineq_solution_set_l211_211144

theorem quadratic_ineq_solution_set (a b c : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, 3 < x → x < 6 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, x < (1 / 6) ∨ x > (1 / 3) → cx^2 + bx + a < 0 := by 
  sorry

end quadratic_ineq_solution_set_l211_211144


namespace camera_pictures_olivia_camera_pictures_l211_211259

theorem camera_pictures (phone_pics : Nat) (albums : Nat) (pics_per_album : Nat) (total_pics : Nat) : Prop :=
  phone_pics = 5 →
  albums = 8 →
  pics_per_album = 5 →
  total_pics = albums * pics_per_album →
  total_pics - phone_pics = 35

-- Here's the statement of the theorem followed by a sorry to indicate that the proof is not provided
theorem olivia_camera_pictures (phone_pics albums pics_per_album total_pics : Nat) (h1 : phone_pics = 5) (h2 : albums = 8) (h3 : pics_per_album = 5) (h4 : total_pics = albums * pics_per_album) : total_pics - phone_pics = 35 :=
by
  sorry

end camera_pictures_olivia_camera_pictures_l211_211259


namespace carl_personal_owe_l211_211343

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end carl_personal_owe_l211_211343


namespace rightmost_four_digits_of_5_pow_2023_l211_211754

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end rightmost_four_digits_of_5_pow_2023_l211_211754


namespace conference_center_distance_l211_211126

variables (d t: ℝ)

theorem conference_center_distance
  (h1: ∃ t: ℝ, d = 45 * (t + 1.5))
  (h2: ∃ t: ℝ, d - 45 = 55 * (t - 1.25)):
  d = 478.125 :=
by
  sorry

end conference_center_distance_l211_211126


namespace determine_constants_l211_211938

theorem determine_constants (α β : ℝ) (h_eq : ∀ x, (x - α) / (x + β) = (x^2 - 96 * x + 2210) / (x^2 + 65 * x - 3510))
  (h_num : ∀ x, x^2 - 96 * x + 2210 = (x - 34) * (x - 62))
  (h_denom : ∀ x, x^2 + 65 * x - 3510 = (x - 45) * (x + 78)) :
  α + β = 112 :=
sorry

end determine_constants_l211_211938


namespace trains_meeting_time_l211_211683

noncomputable def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

noncomputable def time_to_meet (L1 L2 D S1 S2 : ℕ) : ℕ := 
  let S1_mps := kmph_to_mps S1
  let S2_mps := kmph_to_mps S2
  let relative_speed := S1_mps + S2_mps
  let total_distance := L1 + L2 + D
  total_distance / relative_speed

theorem trains_meeting_time : time_to_meet 210 120 160 74 92 = 10620 / 1000 :=
by
  sorry

end trains_meeting_time_l211_211683


namespace reciprocal_of_neg_three_l211_211251

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end reciprocal_of_neg_three_l211_211251


namespace johns_elevation_after_descent_l211_211046

def starting_elevation : ℝ := 400
def rate_of_descent : ℝ := 10
def travel_time : ℝ := 5

theorem johns_elevation_after_descent :
  starting_elevation - (rate_of_descent * travel_time) = 350 :=
by
  sorry

end johns_elevation_after_descent_l211_211046


namespace valentina_burger_length_l211_211686

-- Definitions and conditions
def share : ℕ := 6
def total_length (share : ℕ) : ℕ := 2 * share

-- Proof statement
theorem valentina_burger_length : total_length share = 12 := by
  sorry

end valentina_burger_length_l211_211686


namespace maria_total_cost_l211_211518

def price_pencil: ℕ := 8
def price_pen: ℕ := price_pencil / 2
def total_price: ℕ := price_pencil + price_pen

theorem maria_total_cost: total_price = 12 := by
  sorry

end maria_total_cost_l211_211518


namespace range_of_m_l211_211761

theorem range_of_m {m : ℝ} :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end range_of_m_l211_211761


namespace Daria_vacuum_cleaner_problem_l211_211898

theorem Daria_vacuum_cleaner_problem (initial_savings weekly_savings target_savings weeks_needed : ℕ)
  (h1 : initial_savings = 20)
  (h2 : weekly_savings = 10)
  (h3 : target_savings = 120)
  (h4 : weeks_needed = (target_savings - initial_savings) / weekly_savings) : 
  weeks_needed = 10 :=
by
  sorry

end Daria_vacuum_cleaner_problem_l211_211898


namespace intersection_M_N_l211_211311

def M : Set ℝ := {x | x / (x - 1) > 0}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | x > 1} :=
by
  sorry

end intersection_M_N_l211_211311


namespace hyperbola_eccentricity_range_l211_211708

-- Lean 4 statement for the given problem.
theorem hyperbola_eccentricity_range {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h : ∀ (x y : ℝ), y = x * Real.sqrt 3 → y^2 / b^2 - x^2 / a^2 = 1 ∨ ∃ (z : ℝ), y = x * Real.sqrt 3 ∧ z^2 / b^2 - x^2 / a^2 = 1) :
  1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a < 2 :=
by
  sorry

end hyperbola_eccentricity_range_l211_211708


namespace Shelby_drive_time_in_rain_l211_211701

theorem Shelby_drive_time_in_rain (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 3) 
  (h3 : 40 * (3 - x) + 25 * x = 85) : x = 140 / 60 :=
  sorry

end Shelby_drive_time_in_rain_l211_211701


namespace parker_daily_earning_l211_211611

-- Definition of conditions
def total_earned : ℕ := 2646
def weeks_worked : ℕ := 6
def days_per_week : ℕ := 7
def total_days (weeks : ℕ) (days_in_week : ℕ) : ℕ := weeks * days_in_week

-- Proof statement
theorem parker_daily_earning (h : total_days weeks_worked days_per_week = 42) : (total_earned / 42) = 63 :=
by
  sorry

end parker_daily_earning_l211_211611


namespace Louie_monthly_payment_l211_211585

noncomputable def monthly_payment (P : ℕ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  let A := P * (1 + r) ^ n
  A / 3

theorem Louie_monthly_payment : 
  monthly_payment 1000 0.10 3 (3 / 12) = 444 := 
by
  -- computation and rounding
  sorry

end Louie_monthly_payment_l211_211585


namespace mike_baseball_cards_l211_211125

theorem mike_baseball_cards (initial_cards birthday_cards traded_cards : ℕ)
  (h1 : initial_cards = 64) 
  (h2 : birthday_cards = 18) 
  (h3 : traded_cards = 20) :
  initial_cards + birthday_cards - traded_cards = 62 :=
by 
  -- assumption:
  sorry

end mike_baseball_cards_l211_211125


namespace correct_rounded_result_l211_211548

-- Definition of rounding to the nearest hundred
def rounded_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 < 50 then n / 100 * 100 else (n / 100 + 1) * 100

-- Given conditions
def sum : ℕ := 68 + 57

-- The theorem to prove
theorem correct_rounded_result : rounded_to_nearest_hundred sum = 100 :=
by
  -- Proof skipped
  sorry

end correct_rounded_result_l211_211548


namespace triangle_is_right_l211_211853

theorem triangle_is_right {A B C : ℝ} (h : A + B + C = 180) (h1 : A = B + C) : A = 90 :=
by
  sorry

end triangle_is_right_l211_211853


namespace sixth_bar_placement_l211_211429

theorem sixth_bar_placement (f : ℕ → ℕ) (h1 : f 1 = 1) (h2 : f 2 = 121) :
  (∃ n, f 6 = n ∧ (n = 16 ∨ n = 46 ∨ n = 76 ∨ n = 106)) :=
sorry

end sixth_bar_placement_l211_211429


namespace find_X_l211_211629

theorem find_X (X : ℝ) 
  (h : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002) : 
  X = 1.25 := 
sorry

end find_X_l211_211629


namespace angle_of_inclination_l211_211806

theorem angle_of_inclination (α : ℝ) (h: 0 ≤ α ∧ α < 180) (slope_eq : Real.tan (Real.pi * α / 180) = Real.sqrt 3) :
  α = 60 :=
sorry

end angle_of_inclination_l211_211806


namespace min_digs_is_three_l211_211997

/-- Represents an 8x8 board --/
structure Board :=
(dim : ℕ := 8)

/-- Each cell either contains the treasure or a plaque indicating minimum steps --/
structure Cell :=
(content : CellContent)

/-- Possible content of a cell --/
inductive CellContent
| Treasure
| Plaque (steps : ℕ)

/-- Function that returns the minimum number of cells to dig to find the treasure --/
def min_digs_to_find_treasure (board : Board) : ℕ := 3

/-- The main theorem stating the minimum number of cells needed to find the treasure on an 8x8 board --/
theorem min_digs_is_three : 
  ∀ board : Board, min_digs_to_find_treasure board = 3 := 
by 
  intro board
  sorry

end min_digs_is_three_l211_211997


namespace part1_part2_l211_211091

noncomputable def f (a x : ℝ) := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x
noncomputable def g (a x : ℝ) := x^2 + 5 * a^2
noncomputable def F (a x : ℝ) := f a x + g a x

theorem part1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ a ≤ 0 :=
by sorry

theorem part2 (a : ℝ) : ∀ x : ℝ, F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
by sorry

end part1_part2_l211_211091


namespace calculate_fg1_l211_211690

def f (x : ℝ) : ℝ := 4 - 3 * x
def g (x : ℝ) : ℝ := x^3 + 1

theorem calculate_fg1 : f (g 1) = -2 :=
by
  sorry

end calculate_fg1_l211_211690


namespace triangle_angle_y_l211_211002

theorem triangle_angle_y (y : ℝ) (h1 : 2 * y + (y + 10) + 4 * y = 180) : 
  y = 170 / 7 := 
by
  sorry

end triangle_angle_y_l211_211002


namespace number_of_common_tangents_l211_211831

theorem number_of_common_tangents 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (circle2 : ∀ x y : ℝ, 2 * y^2 - 6 * x - 8 * y + 9 = 0) : 
  ∃ n : ℕ, n = 3 :=
by
  -- Proof is skipped
  sorry

end number_of_common_tangents_l211_211831


namespace tg_gamma_half_eq_2_div_5_l211_211393

theorem tg_gamma_half_eq_2_div_5
  (α β γ : ℝ)
  (a b c : ℝ)
  (triangle_angles : α + β + γ = π)
  (tg_half_alpha : Real.tan (α / 2) = 5/6)
  (tg_half_beta : Real.tan (β / 2) = 10/9)
  (ac_eq_2b : a + c = 2 * b):
  Real.tan (γ / 2) = 2 / 5 :=
sorry

end tg_gamma_half_eq_2_div_5_l211_211393


namespace ma_m_gt_mb_l211_211797

theorem ma_m_gt_mb (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m * a > m * b) → m ≥ 0 := 
  sorry

end ma_m_gt_mb_l211_211797


namespace haley_number_of_shirts_l211_211142

-- Define the given information
def washing_machine_capacity : ℕ := 7
def total_loads : ℕ := 5
def number_of_sweaters : ℕ := 33
def number_of_shirts := total_loads * washing_machine_capacity - number_of_sweaters

-- The statement that needs to be proven
theorem haley_number_of_shirts : number_of_shirts = 2 := by
  sorry

end haley_number_of_shirts_l211_211142


namespace ratio_flow_chart_to_total_time_l211_211432

noncomputable def T := 48
noncomputable def D := 18
noncomputable def C := (3 / 8) * T
noncomputable def F := T - C - D

theorem ratio_flow_chart_to_total_time : (F / T) = (1 / 4) := by
  sorry

end ratio_flow_chart_to_total_time_l211_211432


namespace difference_between_advertised_and_actual_mileage_l211_211734

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage_l211_211734


namespace range_of_a_l211_211336

def proposition_P (a : ℝ) := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_Q (a : ℝ) := 5 - 2*a > 1

theorem range_of_a :
  (∃! (p : Prop), (p = proposition_P a ∨ p = proposition_Q a) ∧ p) →
  a ∈ Set.Iic (-2) :=
by
  sorry

end range_of_a_l211_211336


namespace minimum_value_l211_211364

theorem minimum_value (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 2) 
  (h4 : a + b = 1) : 
  ∃ L, L = (3 * a * c / b) + (c / (a * b)) + (6 / (c - 2)) ∧ L = 1 / (a * (1 - a)) := sorry

end minimum_value_l211_211364


namespace translated_function_is_correct_l211_211893

-- Define the original function
def f (x : ℝ) : ℝ := (x - 2) ^ 2 + 2

-- Define the translated function after moving 1 unit to the left
def g (x : ℝ) : ℝ := f (x + 1)

-- Define the final function after moving 1 unit upward
def h (x : ℝ) : ℝ := g x + 1

-- The statement to be proved
theorem translated_function_is_correct :
  ∀ x : ℝ, h x = (x - 1) ^ 2 + 3 :=
by
  -- Proof goes here
  sorry

end translated_function_is_correct_l211_211893


namespace range_of_a_add_b_l211_211441

-- Define the problem and assumptions
variables (a b : ℝ)
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom ab_eq_a_add_b_add_3 : a * b = a + b + 3

-- Define the theorem to prove
theorem range_of_a_add_b : a + b ≥ 6 :=
sorry

end range_of_a_add_b_l211_211441


namespace claudia_groupings_l211_211642

-- Definition of combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def candles_combinations : ℕ := combination 6 3
def flowers_combinations : ℕ := combination 15 12

-- Lean statement
theorem claudia_groupings : candles_combinations * flowers_combinations = 9100 :=
by
  sorry

end claudia_groupings_l211_211642


namespace total_surface_area_l211_211353

noncomputable def calculate_surface_area
  (radius : ℝ) (reflective : Bool) : ℝ :=
  let base_area := (radius^2 * Real.pi)
  let curved_surface_area := (4 * Real.pi * (radius^2)) / 2
  let effective_surface_area := if reflective then 2 * curved_surface_area else curved_surface_area
  effective_surface_area

theorem total_surface_area (r : ℝ) (h₁_reflective : Bool) (h₂_reflective : Bool) :
  r = 8 →
  h₁_reflective = false →
  h₂_reflective = true →
  (calculate_surface_area r h₁_reflective + calculate_surface_area r h₂_reflective) = 384 * Real.pi := 
by
  sorry

end total_surface_area_l211_211353


namespace find_M_l211_211939

theorem find_M (A M C : ℕ) (h1 : (100 * A + 10 * M + C) * (A + M + C) = 2040)
(h2 : (A + M + C) % 2 = 0)
(h3 : A ≤ 9) (h4 : M ≤ 9) (h5 : C ≤ 9) :
  M = 7 := 
sorry

end find_M_l211_211939


namespace greatest_b_value_ineq_l211_211501

theorem greatest_b_value_ineq (b : ℝ) (h : -b^2 + 8 * b - 15 ≥ 0) : b ≤ 5 := 
sorry

end greatest_b_value_ineq_l211_211501


namespace num_readers_sci_fiction_l211_211799

theorem num_readers_sci_fiction (T L B S: ℕ) (hT: T = 250) (hL: L = 88) (hB: B = 18) (hTotal: T = S + L - B) : 
  S = 180 := 
by 
  sorry

end num_readers_sci_fiction_l211_211799


namespace problem_statement_l211_211225

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end problem_statement_l211_211225


namespace relationship_y1_y2_l211_211116

theorem relationship_y1_y2 (k y1 y2 : ℝ) 
  (h1 : y1 = (k^2 + 1) * (-3) - 5) 
  (h2 : y2 = (k^2 + 1) * 4 - 5) : 
  y1 < y2 :=
sorry

end relationship_y1_y2_l211_211116


namespace industrial_lubricants_percentage_l211_211565

theorem industrial_lubricants_percentage :
  let a := 12   -- percentage for microphotonics
  let b := 24   -- percentage for home electronics
  let c := 15   -- percentage for food additives
  let d := 29   -- percentage for genetically modified microorganisms
  let angle_basic_astrophysics := 43.2 -- degrees for basic astrophysics
  let total_angle := 360              -- total degrees in a circle
  let total_budget := 100             -- total budget in percentage
  let e := (angle_basic_astrophysics / total_angle) * total_budget -- percentage for basic astrophysics
  a + b + c + d + e = 92 → total_budget - (a + b + c + d + e) = 8 :=
by
  intros
  sorry

end industrial_lubricants_percentage_l211_211565


namespace ellie_sam_in_photo_probability_l211_211903

-- Definitions of the conditions
def lap_time_ellie := 120 -- seconds
def lap_time_sam := 75 -- seconds
def start_time := 10 * 60 -- 10 minutes in seconds
def photo_duration := 60 -- 1 minute in seconds
def photo_section := 1 / 3 -- fraction of the track captured in the photo

-- The probability that both Ellie and Sam are in the photo section between 10 to 11 minutes
theorem ellie_sam_in_photo_probability :
  let ellie_time := start_time;
  let sam_time := start_time;
  let ellie_range := (ellie_time - (photo_section * lap_time_ellie / 2), ellie_time + (photo_section * lap_time_ellie / 2));
  let sam_range := (sam_time - (photo_section * lap_time_sam / 2), sam_time + (photo_section * lap_time_sam / 2));
  let overlap_start := max ellie_range.1 sam_range.1;
  let overlap_end := min ellie_range.2 sam_range.2;
  let overlap_duration := max 0 (overlap_end - overlap_start);
  let overlap_probability := overlap_duration / photo_duration;
  overlap_probability = 5 / 12 :=
by
  sorry

end ellie_sam_in_photo_probability_l211_211903


namespace coprime_divisible_l211_211618

theorem coprime_divisible (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a ∣ b * c) : a ∣ c :=
by
  sorry

end coprime_divisible_l211_211618


namespace count_four_digit_numbers_divisible_by_17_and_end_in_17_l211_211789

theorem count_four_digit_numbers_divisible_by_17_and_end_in_17 :
  ∃ S : Finset ℕ, S.card = 5 ∧ ∀ n ∈ S, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0 ∧ n % 100 = 17 :=
by
  sorry

end count_four_digit_numbers_divisible_by_17_and_end_in_17_l211_211789


namespace simplify_product_l211_211768

theorem simplify_product (a : ℝ) : (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4) = 120 * a^10 := by
  sorry

end simplify_product_l211_211768


namespace A_beats_B_by_14_meters_l211_211537

theorem A_beats_B_by_14_meters :
  let distance := 70
  let time_A := 20
  let time_B := 25
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let distance_B_in_A_time := speed_B * time_A
  (distance - distance_B_in_A_time) = 14 :=
by
  sorry

end A_beats_B_by_14_meters_l211_211537


namespace c_share_of_profit_l211_211436

theorem c_share_of_profit 
  (x : ℝ) -- The amount invested by B
  (total_profit : ℝ := 11000) -- Total profit
  (A_invest : ℝ := 3 * x) -- A's investment
  (C_invest : ℝ := (3/2) * A_invest) -- C's investment
  (total_invest : ℝ := A_invest + x + C_invest) -- Total investment
  (C_share : ℝ := C_invest / total_invest * total_profit) -- C's share of the profit
  : C_share = 99000 / 17 := 
  by sorry

end c_share_of_profit_l211_211436


namespace speed_of_stream_l211_211621

theorem speed_of_stream
  (v : ℝ)
  (h1 : ∀ t : ℝ, t = 7)
  (h2 : ∀ d : ℝ, d = 72)
  (h3 : ∀ s : ℝ, s = 21)
  : (72 / (21 - v) + 72 / (21 + v) = 7) → v = 3 :=
by
  intro h
  sorry

end speed_of_stream_l211_211621


namespace manny_problem_l211_211542

noncomputable def num_slices_left (num_pies : Nat) (slices_per_pie : Nat) (num_classmates : Nat) (num_teachers : Nat) (num_slices_per_person : Nat) : Nat :=
  let total_slices := num_pies * slices_per_pie
  let total_people := 1 + num_classmates + num_teachers
  let slices_taken := total_people * num_slices_per_person
  total_slices - slices_taken

theorem manny_problem : num_slices_left 3 10 24 1 1 = 4 := by
  sorry

end manny_problem_l211_211542


namespace lisa_more_dresses_than_ana_l211_211656

theorem lisa_more_dresses_than_ana :
  ∀ (total_dresses ana_dresses : ℕ),
    total_dresses = 48 →
    ana_dresses = 15 →
    (total_dresses - ana_dresses) - ana_dresses = 18 :=
by
  intros total_dresses ana_dresses h1 h2
  sorry

end lisa_more_dresses_than_ana_l211_211656


namespace workers_complete_time_l211_211603

theorem workers_complete_time
  (A : ℝ) -- Total work
  (x1 x2 x3 : ℝ) -- Productivities of the workers
  (h1 : x3 = (x1 + x2) / 2)
  (h2 : 10 * x1 = 15 * x2) :
  (A / x1 = 50) ∧ (A / x2 = 75) ∧ (A / x3 = 60) :=
by
  sorry  -- Proof not required

end workers_complete_time_l211_211603


namespace inequality_solution_l211_211178

theorem inequality_solution (x : ℝ) (h₁ : 1 - x < 0) (h₂ : x - 3 ≤ 0) : 1 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_solution_l211_211178


namespace slightly_used_crayons_count_l211_211028

-- Definitions
def total_crayons := 120
def new_crayons := total_crayons * (1/3)
def broken_crayons := total_crayons * (20/100)
def slightly_used_crayons := total_crayons - new_crayons - broken_crayons

-- Theorem statement
theorem slightly_used_crayons_count :
  slightly_used_crayons = 56 :=
by
  sorry

end slightly_used_crayons_count_l211_211028


namespace goods_train_speed_l211_211325

def speed_of_goods_train (length_in_meters : ℕ) (time_in_seconds : ℕ) (speed_of_man_train_kmph : ℕ) : ℕ :=
  let length_in_km := length_in_meters / 1000
  let time_in_hours := time_in_seconds / 3600
  let relative_speed_kmph := (length_in_km * 3600) / time_in_hours
  relative_speed_kmph - speed_of_man_train_kmph

theorem goods_train_speed :
  speed_of_goods_train 280 9 50 = 62 := by
  sorry

end goods_train_speed_l211_211325


namespace negative_expression_l211_211883

theorem negative_expression :
  -(-1) ≠ -1 ∧ (-1)^2 ≠ -1 ∧ |(-1)| ≠ -1 ∧ -|(-1)| = -1 :=
by
  sorry

end negative_expression_l211_211883


namespace valid_digit_distribution_l211_211647

theorem valid_digit_distribution (n : ℕ) : 
  (∃ (d1 d2 d5 others : ℕ), 
    d1 = n / 2 ∧
    d2 = n / 5 ∧
    d5 = n / 5 ∧
    others = n / 10 ∧
    d1 + d2 + d5 + others = n) :=
by
  sorry

end valid_digit_distribution_l211_211647


namespace union_complements_eq_l211_211780

-- Definitions as per conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define complements
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof statement
theorem union_complements_eq :
  (C_UA ∪ C_UB) = {0, 1, 4} :=
by
  sorry

end union_complements_eq_l211_211780


namespace solve_for_a_l211_211427

variable (a u : ℝ)

def eq1 := (3 / a) + (1 / u) = 7 / 2
def eq2 := (2 / a) - (3 / u) = 6

theorem solve_for_a (h1 : eq1 a u) (h2 : eq2 a u) : a = 2 / 3 := 
by
  sorry

end solve_for_a_l211_211427


namespace inequality_sqrt_sum_ge_2_l211_211593
open Real

theorem inequality_sqrt_sum_ge_2 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  sqrt (a^3 / (1 + b * c)) + sqrt (b^3 / (1 + a * c)) + sqrt (c^3 / (1 + a * b)) ≥ 2 :=
by
  sorry

end inequality_sqrt_sum_ge_2_l211_211593


namespace power_of_seven_l211_211162

theorem power_of_seven : 
  (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = (7 ^ (3 / 28)) :=
by
  sorry

end power_of_seven_l211_211162


namespace distance_range_l211_211112

theorem distance_range (A_school_distance : ℝ) (B_school_distance : ℝ) (x : ℝ)
  (hA : A_school_distance = 3) (hB : B_school_distance = 2) :
  1 ≤ x ∧ x ≤ 5 :=
sorry

end distance_range_l211_211112


namespace six_digit_number_divisible_by_504_l211_211638

theorem six_digit_number_divisible_by_504 : 
  ∃ a b c : ℕ, (523 * 1000 + 100 * a + 10 * b + c) % 504 = 0 := by 
sorry

end six_digit_number_divisible_by_504_l211_211638


namespace vertex_angle_isosceles_triangle_l211_211039

theorem vertex_angle_isosceles_triangle (α : ℝ) (β : ℝ) (sum_of_angles : α + α + β = 180) (base_angle : α = 50) :
  β = 80 :=
by
  sorry

end vertex_angle_isosceles_triangle_l211_211039


namespace geometric_series_properties_l211_211626

noncomputable def first_term := (7 : ℚ) / 8
noncomputable def common_ratio := (-1 : ℚ) / 2

theorem geometric_series_properties : 
  common_ratio = -1 / 2 ∧ 
  (first_term * (1 - common_ratio^4) / (1 - common_ratio)) = 35 / 64 := 
by 
  sorry

end geometric_series_properties_l211_211626


namespace maura_classroom_students_l211_211446

theorem maura_classroom_students (T : ℝ) (h1 : Tina_students = T) (h2 : Maura_students = T) (h3 : Zack_students = T / 2) (h4 : Tina_students + Maura_students + Zack_students = 69) : T = 138 / 5 := by
  sorry

end maura_classroom_students_l211_211446


namespace g_of_f_three_l211_211871

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3*x^2 + 3*x + 2

theorem g_of_f_three : g (f 3) = 1952 := by
  sorry

end g_of_f_three_l211_211871


namespace cricket_target_runs_l211_211120

def target_runs (first_10_overs_run_rate remaining_40_overs_run_rate : ℝ) : ℝ :=
  10 * first_10_overs_run_rate + 40 * remaining_40_overs_run_rate

theorem cricket_target_runs : target_runs 4.2 6 = 282 := by
  sorry

end cricket_target_runs_l211_211120


namespace Kylie_coins_left_l211_211237

-- Definitions based on given conditions
def piggyBank := 30
def brother := 26
def father := 2 * brother
def sofa := 15
def totalCoins := piggyBank + brother + father + sofa
def coinsGivenToLaura := totalCoins / 2
def coinsLeft := totalCoins - coinsGivenToLaura

-- Theorem statement
theorem Kylie_coins_left : coinsLeft = 62 := by sorry

end Kylie_coins_left_l211_211237


namespace number_of_bugs_l211_211873

def flowers_per_bug := 2
def total_flowers_eaten := 6

theorem number_of_bugs : total_flowers_eaten / flowers_per_bug = 3 := 
by sorry

end number_of_bugs_l211_211873


namespace divisibility_condition_l211_211278

theorem divisibility_condition (a p q : ℕ) (hp : p > 0) (ha : a > 0) (hq : q > 0) (h : p ≤ q) :
  (p ∣ a^p ↔ p ∣ a^q) :=
sorry

end divisibility_condition_l211_211278


namespace basic_full_fare_l211_211665

theorem basic_full_fare 
  (F R : ℝ)
  (h1 : F + R = 216)
  (h2 : (F + R) + (0.5 * F + R) = 327) :
  F = 210 :=
by
  sorry

end basic_full_fare_l211_211665


namespace find_positive_x_l211_211841

theorem find_positive_x (x : ℝ) (h1 : x * ⌊x⌋ = 72) (h2 : x > 0) : x = 9 :=
by 
  sorry

end find_positive_x_l211_211841


namespace negation_universal_prop_l211_211826

theorem negation_universal_prop :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_universal_prop_l211_211826


namespace condition1_condition2_l211_211508

-- Definition for the coordinates of point P based on given m
def P (m : ℝ) : ℝ × ℝ := (3 * m - 6, m + 1)

-- Condition 1: Point P lies on the x-axis
theorem condition1 (m : ℝ) (hx : P m = (3 * m - 6, 0)) : P m = (-9, 0) := 
by {
  -- Show that if y-coordinate is zero, then m + 1 = 0, hence m = -1
  sorry
}

-- Condition 2: Point A is (-1, 2) and AP is parallel to the y-axis
theorem condition2 (m : ℝ) (A : ℝ × ℝ := (-1, 2)) (hy : (3 * m - 6 = -1)) : P m = (-1, 8/3) :=
by {
  -- Show that if the x-coordinates of A and P are equal, then 3m-6 = -1, hence m = 5/3
  sorry
}

end condition1_condition2_l211_211508


namespace solve_abs_inequality_l211_211731

theorem solve_abs_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8) :=
sorry

end solve_abs_inequality_l211_211731


namespace is_isosceles_triangle_l211_211404

theorem is_isosceles_triangle 
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a * Real.cos B + b * Real.cos C + c * Real.cos A = b * Real.cos A + c * Real.cos B + a * Real.cos C) : 
  (A = B ∨ B = C ∨ A = C) :=
sorry

end is_isosceles_triangle_l211_211404


namespace maximum_k_inequality_l211_211774

open Real

noncomputable def inequality_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : Prop :=
  (x / sqrt (y + z)) + (y / sqrt (z + x)) + (z / sqrt (x + y)) ≥ sqrt (3 / 2) * sqrt (x + y + z)
 
-- This is the theorem statement
theorem maximum_k_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  inequality_problem x y z h1 h2 h3 :=
  sorry

end maximum_k_inequality_l211_211774


namespace sum_of_number_and_reverse_l211_211160

theorem sum_of_number_and_reverse (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) 
(h : (10 * a + b) - (10 * b + a) = 3 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 33 := 
sorry

end sum_of_number_and_reverse_l211_211160


namespace lydia_current_age_l211_211090

def years_for_apple_tree_to_bear_fruit : ℕ := 7
def lydia_age_when_planted_tree : ℕ := 4
def lydia_age_when_eats_apple : ℕ := 11

theorem lydia_current_age 
  (h : lydia_age_when_eats_apple - lydia_age_when_planted_tree = years_for_apple_tree_to_bear_fruit) :
  lydia_age_when_eats_apple = 11 := 
by
  sorry

end lydia_current_age_l211_211090


namespace simplify_expression_l211_211726

theorem simplify_expression (w : ℝ) : (5 - 2 * w) - (4 + 5 * w) = 1 - 7 * w := by 
  sorry

end simplify_expression_l211_211726


namespace handshakes_4_handshakes_n_l211_211029

-- Defining the number of handshakes for n people
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

-- Proving that the number of handshakes for 4 people is 6
theorem handshakes_4 : handshakes 4 = 6 := by
  sorry

-- Proving that the number of handshakes for n people is (n * (n - 1)) / 2
theorem handshakes_n (n : ℕ) : handshakes n = (n * (n - 1)) / 2 := by 
  sorry

end handshakes_4_handshakes_n_l211_211029


namespace second_car_avg_mpg_l211_211466

theorem second_car_avg_mpg 
  (x y : ℝ) 
  (h1 : x + y = 75) 
  (h2 : 25 * x + 35 * y = 2275) : 
  y = 40 := 
by sorry

end second_car_avg_mpg_l211_211466


namespace value_of_m_l211_211862

def f (x m : ℝ) : ℝ := x^2 - 2 * x + m
def g (x m : ℝ) : ℝ := x^2 - 2 * x + 2 * m + 8

theorem value_of_m (m : ℝ) : (3 * f 5 m = g 5 m) → m = -22 :=
by
  intro h
  sorry

end value_of_m_l211_211862


namespace determine_abcd_l211_211568

-- Define a 4-digit natural number abcd in terms of its digits a, b, c, d
def four_digit_number (abcd a b c d : ℕ) :=
  abcd = 1000 * a + 100 * b + 10 * c + d

-- Define the condition given in the problem
def satisfies_condition (abcd a b c d : ℕ) :=
  abcd - (100 * a + 10 * b + c) - (10 * a + b) - a = 1995

-- Define the main theorem statement proving the number is 2243
theorem determine_abcd : ∃ (a b c d abcd : ℕ), four_digit_number abcd a b c d ∧ satisfies_condition abcd a b c d ∧ abcd = 2243 :=
by
  sorry

end determine_abcd_l211_211568


namespace spotlight_distance_l211_211262

open Real

-- Definitions for the ellipsoid parameters
def ellipsoid_parameters (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ a - c = 1.5

-- Given conditions as input parameters
variables (a b c : ℝ)
variables (h_a : a = 2.7) -- semi-major axis half length
variables (h_c : c = 1.5) -- focal point distance

-- Prove that the distance from F2 to F1 is 12 cm
theorem spotlight_distance (h : ellipsoid_parameters a b c) : 2 * a - (a - c) = 12 :=
by sorry

end spotlight_distance_l211_211262


namespace part_1_part_2_l211_211684

variables (a b c : ℝ) (A B C : ℝ)
variable (triangle_ABC : a = b ∧ b = c ∧ A + B + C = 180 ∧ A = 90 ∨ B = 90 ∨ C = 90)
variable (sin_condition : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C)

theorem part_1 (h : a = b) : Real.cos C = 7 / 8 :=
by { sorry }

theorem part_2 (h₁ : B = 90) (h₂ : a = Real.sqrt 2) : b = 2 :=
by { sorry }

end part_1_part_2_l211_211684


namespace value_of_a_5_l211_211092

-- Define the sequence with the general term formula
def a (n : ℕ) : ℕ := 4 * n - 3

-- Prove that the value of a_5 is 17
theorem value_of_a_5 : a 5 = 17 := by
  sorry

end value_of_a_5_l211_211092


namespace average_of_remaining_two_l211_211212

-- Given conditions
def average_of_six (S : ℝ) := S / 6 = 3.95
def average_of_first_two (S1 : ℝ) := S1 / 2 = 4.2
def average_of_next_two (S2 : ℝ) := S2 / 2 = 3.85

-- Prove that the average of the remaining 2 numbers equals 3.8
theorem average_of_remaining_two (S S1 S2 Sr : ℝ) (h1 : average_of_six S) (h2 : average_of_first_two S1) (h3: average_of_next_two S2) (h4 : Sr = S - S1 - S2) :
  Sr / 2 = 3.8 :=
by
  -- We can use the assumptions h1, h2, h3, and h4 to reach the conclusion
  sorry

end average_of_remaining_two_l211_211212


namespace part_two_l211_211270

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 2 * x + a * Real.log x

theorem part_two (a : ℝ) (h : a = 4) (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h_cond : (f m a + f n a) / (m^2 * n^2) = 1) : m + n ≥ 3 :=
sorry

end part_two_l211_211270


namespace percentage_of_students_with_same_grades_l211_211440

noncomputable def same_grade_percentage (students_class : ℕ) (grades_A : ℕ) (grades_B : ℕ) (grades_C : ℕ) (grades_D : ℕ) (grades_E : ℕ) : ℚ :=
  ((grades_A + grades_B + grades_C + grades_D + grades_E : ℚ) / students_class) * 100

theorem percentage_of_students_with_same_grades :
  let students_class := 40
  let grades_A := 3
  let grades_B := 5
  let grades_C := 6
  let grades_D := 2
  let grades_E := 1
  same_grade_percentage students_class grades_A grades_B grades_C grades_D grades_E = 42.5 := by
  sorry

end percentage_of_students_with_same_grades_l211_211440


namespace two_to_the_n_plus_3_is_perfect_square_l211_211971

theorem two_to_the_n_plus_3_is_perfect_square (n : ℕ) (h : ∃ a : ℕ, 2^n + 3 = a^2) : n = 0 := 
sorry

end two_to_the_n_plus_3_is_perfect_square_l211_211971


namespace value_of_coupon_l211_211007

theorem value_of_coupon (price_per_bag : ℝ) (oz_per_bag : ℕ) (cost_per_serving_with_coupon : ℝ) (total_servings : ℕ) :
  price_per_bag = 25 → oz_per_bag = 40 → cost_per_serving_with_coupon = 0.50 → total_servings = 40 →
  (price_per_bag - (cost_per_serving_with_coupon * total_servings)) = 5 :=
by 
  intros hpb hob hcpwcs hts
  sorry

end value_of_coupon_l211_211007


namespace g_eq_g_inv_solution_l211_211978

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem g_eq_g_inv_solution (x : ℝ) : g x = g_inv x ↔ x = 5 / 3 :=
by
  sorry

end g_eq_g_inv_solution_l211_211978


namespace number_at_100th_row_1000th_column_l211_211216

axiom cell_numbering_rule (i j : ℕ) : ℕ

/-- 
  The cell located at the intersection of the 100th row and the 1000th column
  on an infinitely large chessboard, sequentially numbered with specific rules,
  will receive the number 900.
-/
theorem number_at_100th_row_1000th_column : cell_numbering_rule 100 1000 = 900 :=
sorry

end number_at_100th_row_1000th_column_l211_211216


namespace domain_of_tan_2x_plus_pi_over_3_l211_211505

noncomputable def domain_tan_transformed : Set ℝ :=
  {x : ℝ | ∀ (k : ℤ), x ≠ k * (Real.pi / 2) + (Real.pi / 12)}

theorem domain_of_tan_2x_plus_pi_over_3 :
  (∀ x : ℝ, x ∉ domain_tan_transformed ↔ ∃ (k : ℤ), x = k * (Real.pi / 2) + (Real.pi / 12)) :=
sorry

end domain_of_tan_2x_plus_pi_over_3_l211_211505


namespace projection_non_ambiguity_l211_211991

theorem projection_non_ambiguity 
    (a b c : ℝ) 
    (theta : ℝ) 
    (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos theta) : 
    ∃ (c' : ℝ), c' = c * Real.cos theta ∧ a^2 = b^2 + c^2 + 2 * b * c' := 
sorry

end projection_non_ambiguity_l211_211991


namespace geometric_sequence_l211_211837

theorem geometric_sequence (q : ℝ) (a : ℕ → ℝ) (h1 : q > 0) (h2 : a 2 = 1)
  (h3 : a 2 * a 10 = 2 * (a 5)^2) : ∀ n, a n = 2^((n-2:ℝ)/2) := by
  sorry

end geometric_sequence_l211_211837


namespace min_rice_weight_l211_211228

theorem min_rice_weight (o r : ℝ) (h1 : o ≥ 4 + 2 * r) (h2 : o ≤ 3 * r) : r ≥ 4 :=
sorry

end min_rice_weight_l211_211228


namespace number_of_people_purchased_only_book_A_l211_211977

-- Definitions based on the conditions
variable (A B x y z w : ℕ)
variable (h1 : z = 500)
variable (h2 : z = 2 * y)
variable (h3 : w = z)
variable (h4 : x + y + z + w = 2500)
variable (h5 : A = x + z)
variable (h6 : B = y + z)
variable (h7 : A = 2 * B)

-- The statement we want to prove
theorem number_of_people_purchased_only_book_A :
  x = 1000 :=
by
  -- The proof steps will be filled here
  sorry

end number_of_people_purchased_only_book_A_l211_211977


namespace is_odd_function_l211_211827

def f (x : ℝ) : ℝ := x^3 - x

theorem is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end is_odd_function_l211_211827


namespace three_dice_prime_probability_l211_211138

noncomputable def rolling_three_dice_prime_probability : ℚ :=
  sorry

theorem three_dice_prime_probability : rolling_three_dice_prime_probability = 1 / 24 :=
  sorry

end three_dice_prime_probability_l211_211138


namespace fish_per_person_l211_211573

theorem fish_per_person (eyes_per_fish : ℕ) (fish_caught : ℕ) (total_eyes : ℕ) (dog_eyes : ℕ) (oomyapeck_eyes : ℕ) (n_people : ℕ) :
  total_eyes = oomyapeck_eyes + dog_eyes →
  total_eyes = fish_caught * eyes_per_fish →
  n_people = 3 →
  oomyapeck_eyes = 22 →
  dog_eyes = 2 →
  eyes_per_fish = 2 →
  fish_caught / n_people = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end fish_per_person_l211_211573


namespace intersection_M_N_l211_211992

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l211_211992


namespace smallest_multiple_of_3_l211_211732

theorem smallest_multiple_of_3 (a : ℕ) (h : ∀ i j : ℕ, i < 6 → j < 6 → 3 * (a + i) = 3 * (a + 10 + j) → a = 50) : 3 * a = 150 :=
by
  sorry

end smallest_multiple_of_3_l211_211732


namespace probability_of_rain_l211_211351

-- Define the conditions in Lean
variables (x : ℝ) -- probability of rain

-- Known condition: taking an umbrella 20% of the time
def takes_umbrella : Prop := 0.2 = x + ((1 - x) * x)

-- The desired problem statement
theorem probability_of_rain : takes_umbrella x → x = 1 / 9 :=
by
  -- placeholder for the proof
  intro h
  sorry

end probability_of_rain_l211_211351


namespace sqrt_mul_eq_6_l211_211842

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l211_211842


namespace solvability_condition_l211_211301

def is_solvable (p : ℕ) [Fact (Nat.Prime p)] :=
  ∃ α : ℤ, α * (α - 1) + 3 ≡ 0 [ZMOD p] ↔ ∃ β : ℤ, β * (β - 1) + 25 ≡ 0 [ZMOD p]

theorem solvability_condition (p : ℕ) [Fact (Nat.Prime p)] : 
  is_solvable p :=
sorry

end solvability_condition_l211_211301


namespace boat_speed_is_13_l211_211273

noncomputable def boatSpeedStillWater : ℝ := 
  let Vs := 6 -- Speed of the stream in km/hr
  let time := 3.6315789473684212 -- Time taken in hours to travel 69 km downstream
  let distance := 69 -- Distance traveled in km
  (distance - Vs * time) / time

theorem boat_speed_is_13 : boatSpeedStillWater = 13 := by
  sorry

end boat_speed_is_13_l211_211273


namespace number_of_Slurpees_l211_211141

theorem number_of_Slurpees
  (total_money : ℕ)
  (cost_per_Slurpee : ℕ)
  (change : ℕ)
  (spent_money := total_money - change)
  (number_of_Slurpees := spent_money / cost_per_Slurpee)
  (h1 : total_money = 20)
  (h2 : cost_per_Slurpee = 2)
  (h3 : change = 8) :
  number_of_Slurpees = 6 := by
  sorry

end number_of_Slurpees_l211_211141


namespace jane_albert_same_committee_l211_211472

def probability_same_committee (total_MBAs : ℕ) (committee_size : ℕ) (num_committees : ℕ) (favorable_cases : ℕ) (total_cases : ℕ) : ℚ :=
  favorable_cases / total_cases

theorem jane_albert_same_committee :
  probability_same_committee 9 4 3 105 630 = 1 / 6 :=
by
  sorry

end jane_albert_same_committee_l211_211472


namespace race_speeds_l211_211418

theorem race_speeds (x y : ℕ) 
  (h1 : 5 * x + 10 = 5 * y) 
  (h2 : 6 * x = 4 * y) :
  x = 4 ∧ y = 6 :=
by {
  -- Proof will go here, but for now we skip it.
  sorry
}

end race_speeds_l211_211418


namespace range_of_m_l211_211564

theorem range_of_m (m : ℝ) :
  (∃ ρ θ : ℝ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 0 ∧
    (∃ ρ₀ θ₀ : ℝ, ∀ ρ θ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 
      m * ρ₀ * (Real.cos θ₀)^2 + 3 * ρ₀ * (Real.sin θ₀)^2 - 6 * (Real.cos θ₀))) →
  m > 0 ∧ m ≠ 3 := sorry

end range_of_m_l211_211564


namespace caitlin_bracelets_l211_211574

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end caitlin_bracelets_l211_211574


namespace fractions_order_and_non_equality_l211_211969

theorem fractions_order_and_non_equality:
  (37 / 29 < 41 / 31) ∧ (41 / 31 < 31 / 23) ∧ 
  ((37 / 29 ≠ 4 / 3) ∧ (41 / 31 ≠ 4 / 3) ∧ (31 / 23 ≠ 4 / 3)) := by
  sorry

end fractions_order_and_non_equality_l211_211969


namespace max_boxes_fit_l211_211340

theorem max_boxes_fit 
  (L_large W_large H_large : ℕ) 
  (L_small W_small H_small : ℕ) 
  (h1 : L_large = 12) 
  (h2 : W_large = 14) 
  (h3 : H_large = 16) 
  (h4 : L_small = 3) 
  (h5 : W_small = 7) 
  (h6 : H_small = 2) 
  : ((L_large * W_large * H_large) / (L_small * W_small * H_small) = 64) :=
by
  sorry

end max_boxes_fit_l211_211340


namespace translate_B_to_origin_l211_211614

structure Point where
  x : ℝ
  y : ℝ

def translate_right (p : Point) (d : ℕ) : Point := 
  { x := p.x + d, y := p.y }

theorem translate_B_to_origin :
  ∀ (A B : Point) (d : ℕ),
  A = { x := -4, y := 0 } →
  B = { x := 0, y := 2 } →
  (translate_right A d).x = 0 →
  translate_right B d = { x := 4, y := 2 } :=
by
  intros A B d hA hB hA'
  sorry

end translate_B_to_origin_l211_211614


namespace number_divisible_by_19_l211_211775

theorem number_divisible_by_19 (n : ℕ) : (12000 + 3 * 10^n + 8) % 19 = 0 := 
by sorry

end number_divisible_by_19_l211_211775


namespace second_number_is_915_l211_211313

theorem second_number_is_915 :
  ∃ (n1 n2 n3 n4 n5 n6 : ℤ), 
    n1 = 3 ∧ 
    n2 = 915 ∧ 
    n3 = 138 ∧ 
    n4 = 1917 ∧ 
    n5 = 2114 ∧ 
    ∃ x: ℤ, 
      (n1 + n2 + n3 + n4 + n5 + x) / 6 = 12 ∧ 
      n2 = 915 :=
by 
  sorry

end second_number_is_915_l211_211313


namespace base10_to_base4_of_255_l211_211964

theorem base10_to_base4_of_255 :
  (255 : ℕ) = 3 * 4^3 + 3 * 4^2 + 3 * 4^1 + 3 * 4^0 :=
by
  sorry

end base10_to_base4_of_255_l211_211964


namespace seven_pow_fifty_one_mod_103_l211_211575

theorem seven_pow_fifty_one_mod_103 : (7^51 - 1) % 103 = 0 := 
by
  -- Fermat's Little Theorem: If p is a prime number and a is an integer not divisible by p,
  -- then a^(p-1) ≡ 1 ⧸ p.
  -- 103 is prime, so for 7 which is not divisible by 103, we have 7^102 ≡ 1 ⧸ 103.
sorry

end seven_pow_fifty_one_mod_103_l211_211575


namespace field_fence_length_l211_211662

theorem field_fence_length (L : ℝ) (A : ℝ) (W : ℝ) (fencing : ℝ) (hL : L = 20) (hA : A = 210) (hW : A = L * W) : 
  fencing = 2 * W + L → fencing = 41 :=
by
  rw [hL, hA] at hW
  sorry

end field_fence_length_l211_211662


namespace three_numbers_sum_div_by_three_l211_211835

theorem three_numbers_sum_div_by_three (s : Fin 7 → ℕ) : 
  ∃ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (s a + s b + s c) % 3 = 0 := 
sorry

end three_numbers_sum_div_by_three_l211_211835


namespace grades_calculation_l211_211428

-- Defining the conditions
def total_students : ℕ := 22800
def students_per_grade : ℕ := 75

-- Stating the theorem to be proved
theorem grades_calculation : total_students / students_per_grade = 304 := sorry

end grades_calculation_l211_211428


namespace cost_equivalence_l211_211664

theorem cost_equivalence (b a p : ℕ) (h1 : 4 * b = 3 * a) (h2 : 9 * a = 6 * p) : 24 * b = 12 * p :=
  sorry

end cost_equivalence_l211_211664


namespace find_f2_plus_g2_l211_211707

-- Functions f and g are defined
variable (f g : ℝ → ℝ)

-- Conditions based on the problem
def even_function : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function : Prop := ∀ x : ℝ, g (-x) = g x
def function_equation : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

-- Lean Theorem Statement
theorem find_f2_plus_g2 (h1 : even_function f) (h2 : odd_function g) (h3 : function_equation f g) :
  f 2 + g 2 = -4 :=
by
  sorry

end find_f2_plus_g2_l211_211707


namespace volume_percentage_error_l211_211023

theorem volume_percentage_error (L W H : ℝ) (hL : L > 0) (hW : W > 0) (hH : H > 0) :
  let V_true := L * W * H
  let L_meas := 1.08 * L
  let W_meas := 1.12 * W
  let H_meas := 1.05 * H
  let V_calc := L_meas * W_meas * H_meas
  let percentage_error := ((V_calc - V_true) / V_true) * 100
  percentage_error = 25.424 :=
by
  sorry

end volume_percentage_error_l211_211023


namespace jerry_games_before_birthday_l211_211244

def num_games_before (current received : ℕ) : ℕ :=
  current - received

theorem jerry_games_before_birthday : 
  ∀ (current received before : ℕ), current = 9 → received = 2 → before = num_games_before current received → before = 7 :=
by
  intros current received before h_current h_received h_before
  rw [h_current, h_received] at h_before
  exact h_before

end jerry_games_before_birthday_l211_211244


namespace building_height_l211_211808

theorem building_height
  (num_stories_1 : ℕ)
  (height_story_1 : ℕ)
  (num_stories_2 : ℕ)
  (height_story_2 : ℕ)
  (h1 : num_stories_1 = 10)
  (h2 : height_story_1 = 12)
  (h3 : num_stories_2 = 10)
  (h4 : height_story_2 = 15)
  :
  num_stories_1 * height_story_1 + num_stories_2 * height_story_2 = 270 :=
by
  sorry

end building_height_l211_211808


namespace campaign_donation_ratio_l211_211041

theorem campaign_donation_ratio (max_donation : ℝ) 
  (total_money : ℝ) 
  (percent_donations : ℝ) 
  (num_max_donors : ℕ) 
  (half_max_donation : ℝ) 
  (total_raised : ℝ) 
  (half_donation : ℝ) :
  total_money = total_raised * percent_donations →
  half_donation = max_donation / 2 →
  half_max_donation = num_max_donors * max_donation →
  total_money - half_max_donation = 1500 * half_donation →
  (1500 : ℝ) / (num_max_donors : ℝ) = 3 :=
sorry

end campaign_donation_ratio_l211_211041


namespace smallest_four_digit_div_by_53_l211_211759

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l211_211759


namespace smallest_single_discount_more_advantageous_l211_211974

theorem smallest_single_discount_more_advantageous (n : ℕ) :
  (∀ n, 0 < n -> (1 - (n:ℝ)/100) < 0.64 ∧ (1 - (n:ℝ)/100) < 0.658503 ∧ (1 - (n:ℝ)/100) < 0.63) → 
  n = 38 := 
sorry

end smallest_single_discount_more_advantageous_l211_211974


namespace dog_food_amount_l211_211471

theorem dog_food_amount (x : ℕ) (h1 : 3 * x + 6 = 15) : x = 3 :=
by {
  sorry
}

end dog_food_amount_l211_211471


namespace min_value_of_a_l211_211930

theorem min_value_of_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y))) : 
  a ≥ Real.sqrt 2 :=
sorry -- Proof is omitted

end min_value_of_a_l211_211930


namespace not_necessarily_divisor_sixty_four_l211_211691

theorem not_necessarily_divisor_sixty_four (k : ℤ) (h : (k * (k + 1) * (k + 2)) % 8 = 0) :
  ¬ ((k * (k + 1) * (k + 2)) % 64 = 0) := 
sorry

end not_necessarily_divisor_sixty_four_l211_211691


namespace storage_space_remaining_l211_211203

def total_space_remaining (first_floor second_floor: ℕ) (boxes: ℕ) : ℕ :=
  first_floor + second_floor - boxes

theorem storage_space_remaining :
  ∀ (first_floor second_floor boxes: ℕ),
  (first_floor = 2 * second_floor) →
  (boxes = 5000) →
  (boxes = second_floor / 4) →
  total_space_remaining first_floor second_floor boxes = 55000 :=
by
  intros first_floor second_floor boxes h1 h2 h3
  sorry

end storage_space_remaining_l211_211203


namespace find_number_l211_211084

theorem find_number (N : ℝ) (h1 : (4/5) * (3/8) * N = some_number)
                    (h2 : 2.5 * N = 199.99999999999997) :
  N = 79.99999999999999 := 
sorry

end find_number_l211_211084


namespace number_of_factors_60_l211_211022

def prime_factorization_60 : Prop := (60 = 2^2 * 3 * 5)

theorem number_of_factors_60 (h : prime_factorization_60) : 
  12 = ( (2 + 1) * (1 + 1) * (1 + 1) ) := 
by
  sorry

end number_of_factors_60_l211_211022


namespace trees_in_yard_l211_211339

theorem trees_in_yard (L d : ℕ) (hL : L = 250) (hd : d = 5) : 
  (L / d + 1) = 51 := by
  sorry

end trees_in_yard_l211_211339


namespace evaluate_fg_of_2_l211_211936

def f (x : ℝ) : ℝ := x ^ 3
def g (x : ℝ) : ℝ := 4 * x + 5

theorem evaluate_fg_of_2 : f (g 2) = 2197 :=
by
  sorry

end evaluate_fg_of_2_l211_211936


namespace correct_transformation_l211_211332

theorem correct_transformation (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
sorry

end correct_transformation_l211_211332


namespace certain_number_l211_211371

theorem certain_number (p q x : ℝ) (h1 : 3 / p = x) (h2 : 3 / q = 15) (h3 : p - q = 0.3) : x = 6 :=
sorry

end certain_number_l211_211371


namespace maria_tom_weather_probability_l211_211807

noncomputable def probability_exactly_two_clear_days (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * (p ^ (n - 2)) * ((1 - p) ^ 2)

theorem maria_tom_weather_probability :
  probability_exactly_two_clear_days 0.6 5 = 1080 / 3125 :=
by
  sorry

end maria_tom_weather_probability_l211_211807


namespace activity_probability_l211_211082

noncomputable def total_basic_events : ℕ := 3^4
noncomputable def favorable_events : ℕ := Nat.choose 4 2 * Nat.factorial 3

theorem activity_probability :
  (favorable_events : ℚ) / total_basic_events = 4 / 9 :=
by
  sorry

end activity_probability_l211_211082


namespace exist_midpoints_l211_211975
open Classical

noncomputable def h (a b c : ℝ) := (a + b + c) / 3

theorem exist_midpoints (a b c : ℝ) (X Y Z : ℝ) (AX BY CZ : ℝ) :
  (0 < X) ∧ (X < a) ∧
  (0 < Y) ∧ (Y < b) ∧
  (0 < Z) ∧ (Z < c) ∧
  (X + (a - X) = (h a b c)) ∧
  (Y + (b - Y) = (h a b c)) ∧
  (Z + (c - Z) = (h a b c)) ∧
  (AX * BY * CZ = (a - X) * (b - Y) * (c - Z))
  → ∃ (X Y Z : ℝ), X = (a / 2) ∧ Y = (b / 2) ∧ Z = (c / 2) :=
by
  sorry

end exist_midpoints_l211_211975


namespace problem_solved_by_half_participants_l211_211727

variables (n m : ℕ)
variable (solve : ℕ → ℕ → Prop)  -- solve i j means participant i solved problem j

axiom half_n_problems_solved : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)

theorem problem_solved_by_half_participants (h : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)) : 
  ∃ j, j < n ∧ (∃ count, count ≥ m / 2 ∧ (∃ i, i < m → solve i j)) :=
  sorry

end problem_solved_by_half_participants_l211_211727


namespace teamX_total_games_l211_211722

variables (x : ℕ)

-- Conditions
def teamX_wins := (3/4) * x
def teamX_loses := (1/4) * x

def teamY_wins := (2/3) * (x + 10)
def teamY_loses := (1/3) * (x + 10)

-- Question: Prove team X played 20 games
theorem teamX_total_games :
  teamY_wins - teamX_wins = 5 ∧ teamY_loses - teamX_loses = 5 → x = 20 := by
sorry

end teamX_total_games_l211_211722


namespace binary_add_mul_l211_211150

def x : ℕ := 0b101010
def y : ℕ := 0b11010
def z : ℕ := 0b1110
def result : ℕ := 0b11000000000

theorem binary_add_mul : ((x + y) * z) = result := by
  sorry

end binary_add_mul_l211_211150


namespace line_through_point_parallel_l211_211832

theorem line_through_point_parallel (p : ℝ × ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0)
  (hp : a * p.1 + b * p.2 + c = 0) :
  ∃ k : ℝ, a * p.1 + b * p.2 + k = 0 :=
by
  use - (a * p.1 + b * p.2)
  sorry

end line_through_point_parallel_l211_211832


namespace quadratic_roots_identity_l211_211414

noncomputable def a := - (2 / 5 : ℝ)
noncomputable def b := (1 / 5 : ℝ)
noncomputable def quadraticRoots := (a, b)

theorem quadratic_roots_identity :
  a + b ^ 2 = - (9 / 25 : ℝ) := 
by 
  rw [a, b]
  sorry

end quadratic_roots_identity_l211_211414


namespace cos_double_angle_nonpositive_l211_211865

theorem cos_double_angle_nonpositive (α β : ℝ) (φ : ℝ) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := 
sorry

end cos_double_angle_nonpositive_l211_211865


namespace K_time_expression_l211_211918

variable (x : ℝ) 

theorem K_time_expression
  (hyp : (45 / (x - 2 / 5) - 45 / x = 3 / 4)) :
  45 / (x : ℝ) = 45 / x :=
sorry

end K_time_expression_l211_211918


namespace problem_tiles_count_l211_211709

theorem problem_tiles_count (T B : ℕ) (h: 2 * T + 3 * B = 301) (hB: B = 3) : T = 146 := 
by
  sorry

end problem_tiles_count_l211_211709


namespace crown_cost_before_tip_l211_211085

theorem crown_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (crown_cost : ℝ) :
  total_paid = 22000 → tip_percentage = 0.10 → total_paid = crown_cost * (1 + tip_percentage) → crown_cost = 20000 :=
by
  sorry

end crown_cost_before_tip_l211_211085


namespace calories_per_person_l211_211239

theorem calories_per_person (oranges : ℕ) (pieces_per_orange : ℕ) (people : ℕ) (calories_per_orange : ℕ) :
  oranges = 5 →
  pieces_per_orange = 8 →
  people = 4 →
  calories_per_orange = 80 →
  (oranges * pieces_per_orange) / people * ((oranges * calories_per_orange) / (oranges * pieces_per_orange)) = 100 :=
by
  intros h_oranges h_pieces_per_orange h_people h_calories_per_orange
  sorry

end calories_per_person_l211_211239


namespace area_increase_l211_211872

theorem area_increase (l w : ℝ) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_original := l * w
  let A_new := l_new * w_new
  ((A_new - A_original) / A_original) * 100 = 56 := 
by
  sorry

end area_increase_l211_211872


namespace equation_of_line_passing_through_points_l211_211136

-- Definition of the points
def point1 : ℝ × ℝ := (-2, -3)
def point2 : ℝ × ℝ := (4, 7)

-- The statement to prove
theorem equation_of_line_passing_through_points :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (forall (x y : ℝ), 
  y + 3 = (5 / 3) * (x + 2) → 3 * y - 5 * x = 1) := sorry

end equation_of_line_passing_through_points_l211_211136


namespace area_of_octagon_l211_211980

theorem area_of_octagon (a b : ℝ) (hsquare : a ^ 2 = 16)
  (hperimeter : 4 * a = 8 * b) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_octagon_l211_211980


namespace function_properties_l211_211243

theorem function_properties (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 25 ≠ 0 ∧ x^2 - (k - 6) * x + 16 ≠ 0) → 
  (-2 < k ∧ k < 10) :=
by
  intros h
  sorry

end function_properties_l211_211243


namespace pages_read_in_a_year_l211_211840

-- Definition of the problem conditions
def novels_per_month := 4
def pages_per_novel := 200
def months_per_year := 12

-- Theorem statement corresponding to the problem
theorem pages_read_in_a_year (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) : 
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  sorry

end pages_read_in_a_year_l211_211840


namespace conic_section_is_hyperbola_l211_211402

theorem conic_section_is_hyperbola (x y : ℝ) :
  (x - 3)^2 = (3 * y + 4)^2 - 75 → 
  ∃ a b c d e f : ℝ, a * x^2 + b * y^2 + c * x + d * y + e = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 :=
sorry

end conic_section_is_hyperbola_l211_211402


namespace length_of_BC_l211_211315

theorem length_of_BC (b : ℝ) (h : b ^ 4 = 125) : 2 * b = 10 :=
sorry

end length_of_BC_l211_211315


namespace original_number_of_girls_l211_211925

theorem original_number_of_girls (b g : ℕ) (h1 : b = 3 * (g - 20)) (h2 : 4 * (b - 60) = g - 20) : 
  g = 460 / 11 :=
by
  sorry

end original_number_of_girls_l211_211925


namespace percentage_reduction_is_correct_l211_211127

-- Definitions and initial conditions
def initial_price_per_model := 100
def models_for_kindergarten := 2
def models_for_elementary := 2 * models_for_kindergarten
def total_models := models_for_kindergarten + models_for_elementary
def total_cost_without_reduction := total_models * initial_price_per_model
def total_cost_paid := 570

-- Goal statement in Lean 4
theorem percentage_reduction_is_correct :
  (total_models > 5) →
  total_cost_paid = 570 →
  models_for_kindergarten = 2 →
  (total_cost_without_reduction - total_cost_paid) / total_models / initial_price_per_model * 100 = 5 :=
by
  -- sorry to skip the proof
  sorry

end percentage_reduction_is_correct_l211_211127


namespace cranberries_left_in_bog_l211_211442

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end cranberries_left_in_bog_l211_211442


namespace p_as_percentage_of_x_l211_211426

-- Given conditions
variables (x y z w t u p : ℝ)
variables (h1 : 0.37 * z = 0.84 * y)
variables (h2 : y = 0.62 * x)
variables (h3 : 0.47 * w = 0.73 * z)
variables (h4 : w = t - u)
variables (h5 : u = 0.25 * t)
variables (h6 : p = z + t + u)

-- Prove that p is 505.675% of x
theorem p_as_percentage_of_x : p = 5.05675 * x := by
  sorry

end p_as_percentage_of_x_l211_211426


namespace impossible_fifty_pieces_l211_211247

open Nat

theorem impossible_fifty_pieces :
  ¬ ∃ (m : ℕ), 1 + 3 * m = 50 :=
by
  sorry

end impossible_fifty_pieces_l211_211247


namespace petya_friends_l211_211762

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l211_211762


namespace number_of_rabbits_l211_211421

-- Given conditions
variable (r c : ℕ)
variable (cond1 : r + c = 51)
variable (cond2 : 4 * r = 3 * (2 * c) + 4)

-- To prove
theorem number_of_rabbits : r = 31 :=
sorry

end number_of_rabbits_l211_211421


namespace largest_angle_in_ratio_3_4_5_l211_211847

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l211_211847


namespace sum_divisible_by_10_l211_211057

-- Define the problem statement
theorem sum_divisible_by_10 {n : ℕ} : (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 10 = 0 ↔ ∃ t : ℕ, n = 5 * t + 1 :=
by sorry

end sum_divisible_by_10_l211_211057


namespace find_a_l211_211204

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l211_211204


namespace volume_of_larger_part_of_pyramid_proof_l211_211070

noncomputable def volume_of_larger_part_of_pyramid (a b : ℝ) (inclined_angle : ℝ) (area_ratio : ℝ) : ℝ :=
let h_trapezoid := Real.sqrt ((2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 / 4)
let height_pyramid := (1 / 2) * h_trapezoid * Real.tan (inclined_angle)
let volume_total := (1 / 3) * (((a + b) / 2) * Real.sqrt ((a - b) ^ 2 + 4 * h_trapezoid ^ 2) * height_pyramid)
let volume_smaller := (1 / (5 + 7)) * 7 * volume_total
(volume_total - volume_smaller)

theorem volume_of_larger_part_of_pyramid_proof  :
  (volume_of_larger_part_of_pyramid 2 (Real.sqrt 3) (Real.pi / 6) (5 / 7) = 0.875) :=
by
sorry

end volume_of_larger_part_of_pyramid_proof_l211_211070


namespace angies_monthly_salary_l211_211697

theorem angies_monthly_salary 
    (necessities_expense : ℕ)
    (taxes_expense : ℕ)
    (left_over : ℕ)
    (monthly_salary : ℕ) :
  necessities_expense = 42 → 
  taxes_expense = 20 → 
  left_over = 18 → 
  monthly_salary = necessities_expense + taxes_expense + left_over → 
  monthly_salary = 80 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end angies_monthly_salary_l211_211697


namespace distance_between_A_and_C_l211_211970

theorem distance_between_A_and_C :
  ∀ (AB BC CD AD AC : ℝ),
  AB = 3 → BC = 2 → CD = 5 → AD = 6 → AC = 1 := 
by
  intros AB BC CD AD AC hAB hBC hCD hAD
  have h1 : AD = AB + BC + CD := by sorry
  have h2 : 6 = 3 + 2 + AC := by sorry
  have h3 : 6 = 5 + AC := by sorry
  have h4 : AC = 1 := by sorry
  exact h4

end distance_between_A_and_C_l211_211970


namespace det_dilation_matrix_l211_211388

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![12, 0], ![0, 12]]

theorem det_dilation_matrix : Matrix.det E = 144 := by
  sorry

end det_dilation_matrix_l211_211388


namespace circle_diameter_l211_211321

open Real

theorem circle_diameter (r_D : ℝ) (r_C : ℝ) (h_D : r_D = 10) (h_ratio: (π * (r_D ^ 2 - r_C ^ 2)) / (π * r_C ^ 2) = 4) : 2 * r_C = 4 * sqrt 5 :=
by sorry

end circle_diameter_l211_211321


namespace problem1_problem2_l211_211366

variable (x y a b c d : ℝ)
variable (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)

-- Problem 1: Prove (x + y) * (x^2 - x * y + y^2) = x^3 + y^3
theorem problem1 : (x + y) * (x^2 - x * y + y^2) = x^3 + y^3 := sorry

-- Problem 2: Prove ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6)
theorem problem2 (a b c d : ℝ) (h_a : a ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) : 
  ((a^2 * b) / (-c * d^3))^3 / (2 * a / d^3) * (c / (2 * a))^2 = - (a^3 * b^3) / (8 * c * d^6) := 
  sorry

end problem1_problem2_l211_211366


namespace total_cost_for_seven_hard_drives_l211_211711

-- Condition: Two identical hard drives cost $50.
def cost_of_two_hard_drives : ℝ := 50

-- Condition: There is a 10% discount if you buy more than four hard drives.
def discount_rate : ℝ := 0.10

-- Question: What is the total cost in dollars for buying seven of these hard drives?
theorem total_cost_for_seven_hard_drives : (7 * (cost_of_two_hard_drives / 2)) * (1 - discount_rate) = 157.5 := 
by 
  -- def cost_of_one_hard_drive
  let cost_of_one_hard_drive := cost_of_two_hard_drives / 2
  -- def cost_of_seven_hard_drives
  let cost_of_seven_hard_drives := 7 * cost_of_one_hard_drive
  have h₁ : 7 * (cost_of_two_hard_drives / 2) = cost_of_seven_hard_drives := by sorry
  have h₂ : cost_of_seven_hard_drives * (1 - discount_rate) = 157.5 := by sorry
  exact h₂

end total_cost_for_seven_hard_drives_l211_211711


namespace solve_quadratic_l211_211185

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end solve_quadratic_l211_211185


namespace common_difference_d_l211_211016

theorem common_difference_d (a_1 d : ℝ) (h1 : a_1 + 2 * d = 4) (h2 : 9 * a_1 + 36 * d = 18) : d = -1 :=
by sorry

end common_difference_d_l211_211016


namespace intersection_eq_T_l211_211333

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l211_211333


namespace Tony_age_at_end_of_period_l211_211785

-- Definitions based on the conditions in a):
def hours_per_day := 2
def days_worked := 60
def total_earnings := 1140
def earnings_per_hour (age : ℕ) := age

-- The main property we need to prove: Tony's age at the end of the period is 12 years old
theorem Tony_age_at_end_of_period : ∃ age : ℕ, (2 * age * days_worked = total_earnings) ∧ age = 12 :=
by
  sorry

end Tony_age_at_end_of_period_l211_211785


namespace inequality_solution_l211_211491

theorem inequality_solution {x : ℝ} :
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5 / 3 := by
  sorry

end inequality_solution_l211_211491


namespace algebraic_expression_value_l211_211949

-- Define given condition
def condition (x : ℝ) : Prop := 3 * x^2 - 2 * x - 1 = 2

-- Define the target expression
def target_expression (x : ℝ) : ℝ := -9 * x^2 + 6 * x - 1

-- The theorem statement
theorem algebraic_expression_value (x : ℝ) (h : condition x) : target_expression x = -10 := by
  sorry

end algebraic_expression_value_l211_211949


namespace row_length_in_feet_l211_211742

theorem row_length_in_feet (seeds_per_row : ℕ) (space_per_seed : ℕ) (inches_per_foot : ℕ) (H1 : seeds_per_row = 80) (H2 : space_per_seed = 18) (H3 : inches_per_foot = 12) : 
  seeds_per_row * space_per_seed / inches_per_foot = 120 :=
by
  sorry

end row_length_in_feet_l211_211742


namespace sin_four_alpha_l211_211327

theorem sin_four_alpha (α : ℝ) (h1 : Real.sin (2 * α) = -4 / 5) (h2 : -Real.pi / 4 < α ∧ α < Real.pi / 4) :
  Real.sin (4 * α) = -24 / 25 :=
sorry

end sin_four_alpha_l211_211327


namespace joe_lists_count_l211_211403

def num_options (n : ℕ) (k : ℕ) : ℕ := n ^ k

theorem joe_lists_count : num_options 12 3 = 1728 := by
  unfold num_options
  sorry

end joe_lists_count_l211_211403


namespace find_tangent_point_l211_211222

noncomputable def exp_neg (x : ℝ) : ℝ := Real.exp (-x)

theorem find_tangent_point :
  ∃ P : ℝ × ℝ, P = (-Real.log 2, 2) ∧ P.snd = exp_neg P.fst ∧ deriv exp_neg P.fst = -2 :=
by
  sorry

end find_tangent_point_l211_211222


namespace amount_lent_by_A_to_B_l211_211159

theorem amount_lent_by_A_to_B
  (P : ℝ)
  (H1 : P * 0.115 * 3 - P * 0.10 * 3 = 1125) :
  P = 25000 :=
by
  sorry

end amount_lent_by_A_to_B_l211_211159


namespace factorization_problem1_factorization_problem2_l211_211100

variables {a b x y : ℝ}

theorem factorization_problem1 (a b x y : ℝ) : a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) :=
by sorry

theorem factorization_problem2 (a b : ℝ) : a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 :=
by sorry

end factorization_problem1_factorization_problem2_l211_211100


namespace number_of_weavers_l211_211660

theorem number_of_weavers (W : ℕ) 
  (h1 : ∀ t : ℕ, t = 4 → 4 = W * (1 * t)) 
  (h2 : ∀ t : ℕ, t = 16 → 64 = 16 * (1 / (W:ℝ) * t)) : 
  W = 4 := 
by {
  sorry
}

end number_of_weavers_l211_211660


namespace solve_for_b_l211_211957

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 :=
by sorry

end solve_for_b_l211_211957


namespace Moe_has_least_amount_of_money_l211_211131

variables (Money : Type) [LinearOrder Money]
variables (Bo Coe Flo Jo Moe Zoe : Money)
variables (Bo_lt_Flo : Bo < Flo) (Jo_lt_Flo : Jo < Flo)
variables (Moe_lt_Bo : Moe < Bo) (Moe_lt_Coe : Moe < Coe)
variables (Moe_lt_Jo : Moe < Jo) (Jo_lt_Bo : Jo < Bo)
variables (Moe_lt_Zoe : Moe < Zoe) (Zoe_lt_Jo : Zoe < Jo)

theorem Moe_has_least_amount_of_money : ∀ x, x ≠ Moe → Moe < x := by
  sorry

end Moe_has_least_amount_of_money_l211_211131


namespace inequality_sum_squares_products_l211_211612

theorem inequality_sum_squares_products {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_sum_squares_products_l211_211612


namespace pow_div_l211_211352

theorem pow_div (a : ℝ) : (-a) ^ 6 / a ^ 3 = a ^ 3 := by
  sorry

end pow_div_l211_211352


namespace yanni_money_left_in_cents_l211_211852

-- Conditions
def initial_money : ℝ := 0.85
def money_from_mother : ℝ := 0.40
def money_found : ℝ := 0.50
def cost_per_toy : ℝ := 1.60
def number_of_toys : ℕ := 3
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Prove
theorem yanni_money_left_in_cents : 
  (initial_money + money_from_mother + money_found) * 100 = 175 :=
by
  sorry

end yanni_money_left_in_cents_l211_211852


namespace number_of_pairs_l211_211935

theorem number_of_pairs (n : Nat) : 
  (∃ n, n > 2 ∧ ∀ x y : ℝ, (5 * y - 3 * x = 15 ∧ x^2 + y^2 ≤ 16) → True) :=
sorry

end number_of_pairs_l211_211935


namespace james_total_pay_l211_211570

def original_prices : List ℝ := [15, 20, 25, 18, 22, 30]
def discounts : List ℝ := [0.30, 0.50, 0.40, 0.20, 0.45, 0.25]

def discounted_price (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price_after_discount (prices discounts : List ℝ) : ℝ :=
  (List.zipWith discounted_price prices discounts).sum

theorem james_total_pay :
  total_price_after_discount original_prices discounts = 84.50 :=
  by sorry

end james_total_pay_l211_211570


namespace area_of_triangle_bounded_by_lines_l211_211242

def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := - x + 5

theorem area_of_triangle_bounded_by_lines :
  let x_intercept_line1 := -3 / 2
  let x_intercept_line2 := 5
  let base := x_intercept_line2 - x_intercept_line1
  let intersection_x := 2 / 3
  let intersection_y := line1 intersection_x
  let height := intersection_y
  let area := (1 / 2) * base * height
  area = 169 / 12 := 
by
  sorry

end area_of_triangle_bounded_by_lines_l211_211242


namespace tournament_participants_l211_211609

theorem tournament_participants (x : ℕ) (h1 : ∀ g b : ℕ, g = 2 * b)
  (h2 : ∀ p : ℕ, p = 3 * x) 
  (h3 : ∀ G B : ℕ, G + B = (3 * x * (3 * x - 1)) / 2)
  (h4 : ∀ G B : ℕ, G / B = 7 / 9) 
  (h5 : x = 11) :
  3 * x = 33 :=
by
  sorry

end tournament_participants_l211_211609


namespace solution_set_real_implies_conditions_l211_211413

variable {a b c : ℝ}

theorem solution_set_real_implies_conditions (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) : a < 0 ∧ (b^2 - 4 * a * c) < 0 := 
sorry

end solution_set_real_implies_conditions_l211_211413


namespace pq_eq_real_nums_l211_211155

theorem pq_eq_real_nums (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := 
by 
  sorry

end pq_eq_real_nums_l211_211155


namespace circle_area_from_diameter_points_l211_211510

theorem circle_area_from_diameter_points (C D : ℝ × ℝ)
    (hC : C = (-2, 3)) (hD : D = (4, -1)) :
    ∃ (A : ℝ), A = 13 * Real.pi :=
by
  let distance := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  have diameter : distance = Real.sqrt (6^2 + (-4)^2) := sorry -- this follows from the coordinates
  have radius : distance / 2 = Real.sqrt 13 := sorry -- half of the diameter
  exact ⟨13 * Real.pi, sorry⟩ -- area of the circle

end circle_area_from_diameter_points_l211_211510


namespace team_A_champion_probability_l211_211591

/-- Teams A and B are playing a volleyball match.
Team A needs to win one more game to become the champion, while Team B needs to win two more games to become the champion.
The probability of each team winning each game is 0.5. -/
theorem team_A_champion_probability :
  let p_win := (0.5 : ℝ)
  let prob_A_champion := 1 - p_win * p_win
  prob_A_champion = 0.75 := by
  sorry

end team_A_champion_probability_l211_211591


namespace johns_leisure_travel_miles_per_week_l211_211113

-- Define the given conditions
def mpg : Nat := 30
def work_round_trip_miles : Nat := 20 * 2  -- 20 miles to work + 20 miles back home
def work_days_per_week : Nat := 5
def weekly_fuel_usage_gallons : Nat := 8

-- Define the property to prove
theorem johns_leisure_travel_miles_per_week :
  let work_miles_per_week := work_round_trip_miles * work_days_per_week
  let total_possible_miles := weekly_fuel_usage_gallons * mpg
  let leisure_miles := total_possible_miles - work_miles_per_week
  leisure_miles = 40 :=
by
  sorry

end johns_leisure_travel_miles_per_week_l211_211113


namespace find_fifth_integer_l211_211226

theorem find_fifth_integer (x y : ℤ) (h_pos : x > 0)
  (h_mean_median : (x + 2 + x + 7 + x + y) / 5 = x + 7) :
  y = 22 :=
sorry

end find_fifth_integer_l211_211226


namespace lake_coverage_day_17_l211_211857

-- Define the state of lake coverage as a function of day
def lake_coverage (day : ℕ) : ℝ :=
  if day ≤ 20 then 2 ^ (day - 20) else 0

-- Prove that on day 17, the lake was covered by 12.5% algae
theorem lake_coverage_day_17 : lake_coverage 17 = 0.125 :=
by
  sorry

end lake_coverage_day_17_l211_211857


namespace extreme_values_l211_211527

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem extreme_values (x : ℝ) (hx : x ≠ 0) :
  (x = -2 → f x = -4 ∧ ∀ y, y > -2 → f y > -4) ∧
  (x = 2 → f x = 4 ∧ ∀ y, y < 2 → f y > 4) :=
sorry

end extreme_values_l211_211527


namespace arithmetic_sequence_problem_l211_211917

-- Define the arithmetic sequence and given properties
variable {a : ℕ → ℝ} -- an arithmetic sequence such that for all n, a_{n+1} - a_{n} is constant
variable (d : ℝ) (a1 : ℝ) -- common difference 'd' and first term 'a1'

-- Express the terms using the common difference 'd' and first term 'a1'
def a_n (n : ℕ) : ℝ := a1 + (n-1) * d

-- Given condition
axiom given_condition : a_n 3 + a_n 8 = 10

-- Proof goal
theorem arithmetic_sequence_problem : 3 * a_n 5 + a_n 7 = 20 :=
by
  -- Define the sequence in terms of common difference and the first term
  let a_n := fun n => a1 + (n-1) * d
  -- Simplify using the given condition
  sorry

end arithmetic_sequence_problem_l211_211917


namespace difference_in_number_of_girls_and_boys_l211_211921

def ratio_boys_girls (b g : ℕ) : Prop := b * 3 = g * 2

def total_students (b g : ℕ) : Prop := b + g = 30

theorem difference_in_number_of_girls_and_boys
  (b g : ℕ)
  (h1 : ratio_boys_girls b g)
  (h2 : total_students b g) :
  g - b = 6 :=
sorry

end difference_in_number_of_girls_and_boys_l211_211921


namespace right_handed_players_count_l211_211747

theorem right_handed_players_count (total_players throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : left_handed_non_throwers = (total_players - throwers) / 3)
  (h4 : right_handed_non_throwers = total_players - throwers - left_handed_non_throwers)
  (h5 : ∀ n, n = throwers + right_handed_non_throwers) : 
  (throwers + right_handed_non_throwers) = 62 := 
by 
  sorry

end right_handed_players_count_l211_211747


namespace algebra_expression_evaluation_l211_211802

theorem algebra_expression_evaluation (a b c d e : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : e < 0) 
  (h4 : abs e = 1) : 
  (-a * b) ^ 2009 - (c + d) ^ 2010 - e ^ 2011 = 0 := by 
  sorry

end algebra_expression_evaluation_l211_211802


namespace smallest_solution_l211_211494

def polynomial (x : ℝ) := x^4 - 34 * x^2 + 225 = 0

theorem smallest_solution : ∃ x : ℝ, polynomial x ∧ ∀ y : ℝ, polynomial y → x ≤ y := 
sorry

end smallest_solution_l211_211494


namespace sufficient_condition_for_negation_l211_211400

theorem sufficient_condition_for_negation {A B : Prop} (h : B → A) : ¬ A → ¬ B :=
by
  intro hA
  intro hB
  apply hA
  exact h hB

end sufficient_condition_for_negation_l211_211400


namespace faye_earned_total_money_l211_211010

def bead_necklaces : ℕ := 3
def gem_necklaces : ℕ := 7
def price_per_necklace : ℕ := 7

theorem faye_earned_total_money :
  (bead_necklaces + gem_necklaces) * price_per_necklace = 70 :=
by
  sorry

end faye_earned_total_money_l211_211010


namespace expression_simplification_l211_211405

theorem expression_simplification (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 :=
by
  sorry

end expression_simplification_l211_211405


namespace map_distance_to_actual_distance_l211_211910

theorem map_distance_to_actual_distance :
  ∀ (d_map : ℝ) (scale_inch : ℝ) (scale_mile : ℝ), 
    d_map = 15 → scale_inch = 0.25 → scale_mile = 3 →
    (d_map / scale_inch) * scale_mile = 180 :=
by
  intros d_map scale_inch scale_mile h1 h2 h3
  rw [h1, h2, h3]
  sorry

end map_distance_to_actual_distance_l211_211910


namespace imaginary_roots_iff_l211_211608

theorem imaginary_roots_iff {k m : ℝ} (hk : k ≠ 0) : (exists (x : ℝ), k * x^2 + m * x + k = 0 ∧ ∃ (y : ℝ), y * 0 = 0 ∧ y ≠ 0) ↔ m ^ 2 < 4 * k ^ 2 :=
by
  sorry

end imaginary_roots_iff_l211_211608


namespace age_difference_l211_211566

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end age_difference_l211_211566


namespace kanul_spent_on_raw_materials_l211_211678

theorem kanul_spent_on_raw_materials 
    (total_amount : ℝ)
    (spent_machinery : ℝ)
    (spent_cash_percent : ℝ)
    (spent_cash : ℝ)
    (amount_raw_materials : ℝ)
    (h_total : total_amount = 93750)
    (h_machinery : spent_machinery = 40000)
    (h_percent : spent_cash_percent = 20 / 100)
    (h_cash : spent_cash = spent_cash_percent * total_amount)
    (h_sum : total_amount = amount_raw_materials + spent_machinery + spent_cash) : 
    amount_raw_materials = 35000 :=
sorry

end kanul_spent_on_raw_materials_l211_211678


namespace onions_left_on_shelf_l211_211140

def initial_onions : ℕ := 98
def sold_onions : ℕ := 65
def remaining_onions : ℕ := initial_onions - sold_onions

theorem onions_left_on_shelf : remaining_onions = 33 :=
by 
  -- Proof would go here
  sorry

end onions_left_on_shelf_l211_211140


namespace total_road_signs_l211_211051

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end total_road_signs_l211_211051


namespace Q_subset_P_l211_211071

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Statement to prove the relationship between P and Q
theorem Q_subset_P : Q ⊆ P :=
by
  sorry

end Q_subset_P_l211_211071


namespace cost_per_unit_l211_211238

theorem cost_per_unit 
  (units_per_month : ℕ := 400)
  (selling_price_per_unit : ℝ := 440)
  (profit_requirement : ℝ := 40000)
  (C : ℝ) :
  profit_requirement ≤ (units_per_month * selling_price_per_unit) - (units_per_month * C) → C ≤ 340 :=
by
  sorry

end cost_per_unit_l211_211238


namespace SarahsNumber_is_2880_l211_211509

def SarahsNumber (n : ℕ) : Prop :=
  (144 ∣ n) ∧ (45 ∣ n) ∧ (1000 ≤ n ∧ n ≤ 3000)

theorem SarahsNumber_is_2880 : SarahsNumber 2880 :=
  by
  sorry

end SarahsNumber_is_2880_l211_211509


namespace min_value_of_function_l211_211801

noncomputable def f (x y : ℝ) : ℝ := x^2 / (x + 2) + y^2 / (y + 1)

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  f x y ≥ 1 / 4 :=
sorry

end min_value_of_function_l211_211801


namespace square_circle_radius_l211_211485

theorem square_circle_radius (a R : ℝ) (h1 : a^2 = 256) (h2 : R = 10) : R = 10 :=
sorry

end square_circle_radius_l211_211485


namespace inequality_always_holds_l211_211646

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 - m * x - 1 < 0) → -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_always_holds_l211_211646


namespace gun_can_hit_l211_211783

-- Define the constants
variables (v g : ℝ)

-- Define the coordinates in the first quadrant
variables (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0)

-- Prove the condition for a point (x, y) to be in the region that can be hit by the gun
theorem gun_can_hit (hv : v > 0) (hg : g > 0) :
  y ≤ (v^2 / (2 * g)) - (g * x^2 / (2 * v^2)) :=
sorry

end gun_can_hit_l211_211783


namespace max_single_student_books_l211_211114

-- Definitions and conditions
variable (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ)
variable (total_avg_books_per_student : ℕ)

-- Given data
def given_data : Prop :=
  total_students = 20 ∧ no_books = 2 ∧ one_book = 8 ∧
  two_books = 3 ∧ total_avg_books_per_student = 2

-- Maximum number of books any single student could borrow
theorem max_single_student_books (total_students no_books one_book two_books total_avg_books_per_student : ℕ) 
  (h : given_data total_students no_books one_book two_books total_avg_books_per_student) : 
  ∃ max_books_borrowed, max_books_borrowed = 8 :=
by
  sorry

end max_single_student_books_l211_211114


namespace negation_of_exists_gt_one_l211_211654

theorem negation_of_exists_gt_one : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by 
  sorry

end negation_of_exists_gt_one_l211_211654


namespace person_a_age_l211_211420

theorem person_a_age (A B : ℕ) (h1 : A + B = 43) (h2 : A + 4 = B + 7) : A = 23 :=
by sorry

end person_a_age_l211_211420


namespace range_of_a_l211_211370

-- Definitions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

-- Main theorem to prove
theorem range_of_a (a : ℝ) (h : a < 0)
  (h_necessary : ∀ x, ¬ p x a → ¬ q x) 
  (h_not_sufficient : ∃ x, ¬ p x a ∧ q x): 
  a ≤ -4 :=
sorry

end range_of_a_l211_211370


namespace min_calls_required_l211_211245

-- Define the set of people involved in the communication
inductive Person
| A | B | C | D | E | F

-- Function to calculate the minimum number of calls for everyone to know all pieces of gossip
def minCalls : ℕ :=
  9

-- Theorem stating the minimum number of calls required
theorem min_calls_required : minCalls = 9 := by
  sorry

end min_calls_required_l211_211245


namespace no_pos_int_mult_5005_in_form_l211_211073

theorem no_pos_int_mult_5005_in_form (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 49) :
  ¬ ∃ k : ℕ, 5005 * k = 10^j - 10^i := by
  sorry

end no_pos_int_mult_5005_in_form_l211_211073


namespace least_n_ge_100_divides_sum_of_powers_l211_211635

theorem least_n_ge_100_divides_sum_of_powers (n : ℕ) (h₁ : n ≥ 100) :
    77 ∣ (Finset.sum (Finset.range (n + 1)) (λ k => 2^k) - 1) ↔ n = 119 :=
by
  sorry

end least_n_ge_100_divides_sum_of_powers_l211_211635


namespace max_y_coordinate_l211_211954

noncomputable def y_coordinate (θ : Real) : Real :=
  let u := Real.sin θ
  3 * u - 4 * u^3

theorem max_y_coordinate : ∃ θ, y_coordinate θ = 1 := by
  use Real.arcsin (1 / 2)
  sorry

end max_y_coordinate_l211_211954


namespace num_three_digit_numbers_no_repeat_l211_211314

theorem num_three_digit_numbers_no_repeat (digits : Finset ℕ) (h : digits = {1, 2, 3, 4}) :
  (digits.card = 4) →
  ∀ d1 d2 d3, d1 ∈ digits → d2 ∈ digits → d3 ∈ digits →
  d1 ≠ d2 → d1 ≠ d3 → d2 ≠ d3 → 
  3 * 2 * 1 * digits.card = 24 :=
by
  sorry

end num_three_digit_numbers_no_repeat_l211_211314


namespace largest_n_condition_l211_211937

theorem largest_n_condition :
  ∃ n : ℤ, (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 2 * n + 99 = k^2 ∧ ∀ x : ℤ, 
  (∃ m' : ℤ, x^2 = (m' + 1)^3 - m'^3) ∧ ∃ k' : ℤ, 2 * x + 99 = k'^2 → x ≤ 289 :=
sorry

end largest_n_condition_l211_211937


namespace fraction_expression_l211_211913

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end fraction_expression_l211_211913


namespace rectangle_area_l211_211409

noncomputable def area_of_rectangle (radius : ℝ) (ab ad : ℝ) : ℝ :=
  ab * ad

theorem rectangle_area (radius : ℝ) (ad : ℝ) (ab : ℝ) 
  (h_radius : radius = Real.sqrt 5)
  (h_ab_ad_relation : ab = 4 * ad) : 
  area_of_rectangle radius ab ad = 16 / 5 :=
by
  sorry

end rectangle_area_l211_211409


namespace min_value_expression_l211_211645

theorem min_value_expression (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : (a - b) * (b - c) * (c - a) = -16) : 
  ∃ x : ℝ, x = (1 / (a - b)) + (1 / (b - c)) - (1 / (c - a)) ∧ x = 5 / 4 :=
by
  sorry

end min_value_expression_l211_211645


namespace first_term_of_arithmetic_sequence_l211_211252

theorem first_term_of_arithmetic_sequence (T : ℕ → ℝ) (b : ℝ) 
  (h1 : ∀ n : ℕ, T n = (n * (2 * b + (n - 1) * 4)) / 2) 
  (h2 : ∃ d : ℝ, ∀ n : ℕ, T (4 * n) / T n = d) :
  b = 2 :=
by
  sorry

end first_term_of_arithmetic_sequence_l211_211252


namespace Abby_in_seat_3_l211_211337

variables (P : Type) [Inhabited P]
variables (Abby Bret Carl Dana : P)
variables (seat : P → ℕ)

-- Conditions from the problem:
-- Bret is actually sitting in seat #2.
axiom Bret_in_seat_2 : seat Bret = 2

-- False statement 1: Dana is next to Bret.
axiom false_statement_1 : ¬ (seat Dana = 1 ∨ seat Dana = 3)

-- False statement 2: Carl is sitting between Dana and Bret.
axiom false_statement_2 : ¬ (seat Carl = 1)

-- The final translated proof problem:
theorem Abby_in_seat_3 : seat Abby = 3 :=
sorry

end Abby_in_seat_3_l211_211337


namespace james_training_hours_in_a_year_l211_211184

-- Definitions based on conditions
def trains_twice_a_day : ℕ := 2
def hours_per_training : ℕ := 4
def days_trains_per_week : ℕ := 7 - 2
def weeks_per_year : ℕ := 52

-- Resultant computation
def daily_training_hours : ℕ := trains_twice_a_day * hours_per_training
def weekly_training_hours : ℕ := daily_training_hours * days_trains_per_week
def yearly_training_hours : ℕ := weekly_training_hours * weeks_per_year

-- Statement to prove
theorem james_training_hours_in_a_year : yearly_training_hours = 2080 := by
  -- proof goes here
  sorry

end james_training_hours_in_a_year_l211_211184


namespace width_of_foil_covered_prism_l211_211615

theorem width_of_foil_covered_prism (L W H : ℝ) 
    (hW1 : W = 2 * L)
    (hW2 : W = 2 * H)
    (hvol : L * W * H = 128) :
    W + 2 = 8 := 
sorry

end width_of_foil_covered_prism_l211_211615


namespace quadratic_inequality_empty_set_l211_211461

theorem quadratic_inequality_empty_set (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 < 0)) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end quadratic_inequality_empty_set_l211_211461


namespace num_solutions_l211_211328

theorem num_solutions (k : ℤ) :
  (∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    (a^2 + b^2 = k * c * (a + b)) ∧
    (b^2 + c^2 = k * a * (b + c)) ∧
    (c^2 + a^2 = k * b * (c + a))) ↔ k = 1 ∨ k = -2 :=
sorry

end num_solutions_l211_211328


namespace radius_increase_l211_211308

theorem radius_increase (C1 C2 : ℝ) (π : ℝ) (hC1 : C1 = 40) (hC2 : C2 = 50) (hπ : π > 0) : 
  (C2 - C1) / (2 * π) = 5 / π := 
sorry

end radius_increase_l211_211308


namespace probability_of_valid_number_l211_211790

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≠ j → (n % (10^i) / 10^(i-1)) ≠ (n % (10^j) / 10^(j-1))

def digits_in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_number (n : ℕ) : Prop :=
  is_even n ∧ has_distinct_digits n ∧ digits_in_range n

noncomputable def count_valid_numbers : ℕ :=
  2296

noncomputable def total_numbers : ℕ :=
  9000

theorem probability_of_valid_number :
  (count_valid_numbers : ℚ) / total_numbers = 574 / 2250 :=
by sorry

end probability_of_valid_number_l211_211790


namespace James_delivers_2565_bags_in_a_week_l211_211118

noncomputable def total_bags_delivered_in_a_week
  (days_15_bags : ℕ)
  (trips_per_day_15_bags : ℕ)
  (bags_per_trip_15 : ℕ)
  (days_20_bags : ℕ)
  (trips_per_day_20_bags : ℕ)
  (bags_per_trip_20 : ℕ) : ℕ :=
  (days_15_bags * trips_per_day_15_bags * bags_per_trip_15) + (days_20_bags * trips_per_day_20_bags * bags_per_trip_20)

theorem James_delivers_2565_bags_in_a_week :
  total_bags_delivered_in_a_week 3 25 15 4 18 20 = 2565 :=
by
  sorry

end James_delivers_2565_bags_in_a_week_l211_211118


namespace bags_bought_l211_211553

theorem bags_bought (initial_bags : ℕ) (bags_given : ℕ) (final_bags : ℕ) (bags_bought : ℕ) :
  initial_bags = 20 → 
  bags_given = 4 → 
  final_bags = 22 → 
  bags_bought = final_bags - (initial_bags - bags_given) → 
  bags_bought = 6 := 
by
  intros h_initial h_given h_final h_buy
  rw [h_initial, h_given, h_final] at h_buy
  exact h_buy

#check bags_bought

end bags_bought_l211_211553


namespace three_more_than_seven_in_pages_l211_211179

theorem three_more_than_seven_in_pages : 
  ∀ (pages : List Nat), (∀ n, n ∈ pages → 1 ≤ n ∧ n ≤ 530) ∧ (List.length pages = 530) →
  ((List.count 3 (pages.bind (λ n => Nat.digits 10 n))) - (List.count 7 (pages.bind (λ n => Nat.digits 10 n)))) = 100 :=
by
  intros pages h
  sorry

end three_more_than_seven_in_pages_l211_211179


namespace packing_big_boxes_l211_211117

def total_items := 8640
def items_per_small_box := 12
def small_boxes_per_big_box := 6

def num_big_boxes (total_items items_per_small_box small_boxes_per_big_box : ℕ) : ℕ :=
  (total_items / items_per_small_box) / small_boxes_per_big_box

theorem packing_big_boxes : num_big_boxes total_items items_per_small_box small_boxes_per_big_box = 120 :=
by
  sorry

end packing_big_boxes_l211_211117


namespace choose_blue_pair_l211_211674

/-- In a drawer, there are 12 distinguishable socks: 5 white, 3 brown, and 4 blue socks.
    Prove that the number of ways to choose a pair of socks such that both socks are blue is 6. -/
theorem choose_blue_pair (total_socks white_socks brown_socks blue_socks : ℕ)
  (h_total : total_socks = 12) (h_white : white_socks = 5) (h_brown : brown_socks = 3) (h_blue : blue_socks = 4) :
  (blue_socks.choose 2) = 6 :=
by
  sorry

end choose_blue_pair_l211_211674


namespace volume_pyramid_problem_l211_211196

noncomputable def volume_of_pyramid : ℝ :=
  1 / 3 * 10 * 1.5

theorem volume_pyramid_problem :
  ∀ (AB BC CG : ℝ)
  (M : ℝ × ℝ × ℝ),
  AB = 4 →
  BC = 2 →
  CG = 3 →
  M = (2, 5, 1.5) →
  volume_of_pyramid = 5 := 
by
  intros AB BC CG M hAB hBC hCG hM
  sorry

end volume_pyramid_problem_l211_211196


namespace find_k_l211_211310

theorem find_k 
  (k : ℝ)
  (p_eq : ∀ x : ℝ, (4 * x + 3 = k * x - 9) → (x = -3 → (k = 0)))
: k = 0 :=
by sorry

end find_k_l211_211310


namespace cost_of_blue_pill_l211_211486

/-
Statement:
Bob takes two blue pills and one orange pill each day for three weeks.
The cost of a blue pill is $2 more than an orange pill.
The total cost for all pills over the three weeks amounts to $966.
Prove that the cost of one blue pill is $16.
-/

theorem cost_of_blue_pill (days : ℕ) (total_cost : ℝ) (cost_orange : ℝ) (cost_blue : ℝ) 
  (h1 : days = 21) 
  (h2 : total_cost = 966) 
  (h3 : cost_blue = cost_orange + 2) 
  (daily_pill_cost : ℝ)
  (h4 : daily_pill_cost = total_cost / days)
  (h5 : daily_pill_cost = 2 * cost_blue + cost_orange) :
  cost_blue = 16 :=
by
  sorry

end cost_of_blue_pill_l211_211486


namespace fixed_point_coordinates_l211_211582

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 4) ∧ ∀ x, P = (x, a^(x-1) + 3) :=
by
  use (1, 4)
  sorry

end fixed_point_coordinates_l211_211582


namespace not_possible_2020_parts_possible_2023_parts_l211_211751

-- Define the initial number of parts and the operation that adds two parts
def initial_parts : Nat := 1
def operation (n : Nat) : Nat := n + 2

theorem not_possible_2020_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2020) : False :=
sorry

theorem possible_2023_parts
  (is_reachable : ∃ k : Nat, initial_parts + 2 * k = 2023) : True :=
sorry

end not_possible_2020_parts_possible_2023_parts_l211_211751


namespace optimal_garden_dimensions_l211_211767

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), (2 * l + 2 * w = 400 ∧
                l ≥ 100 ∧
                w ≥ 0 ∧ 
                l * w = 10000) :=
by
  sorry

end optimal_garden_dimensions_l211_211767


namespace probability_of_drawing_two_black_two_white_l211_211534

noncomputable def probability_two_black_two_white : ℚ :=
  let total_ways := (Nat.choose 18 4)
  let ways_black := (Nat.choose 10 2)
  let ways_white := (Nat.choose 8 2)
  let favorable_ways := ways_black * ways_white
  favorable_ways / total_ways

theorem probability_of_drawing_two_black_two_white :
  probability_two_black_two_white = 7 / 17 := sorry

end probability_of_drawing_two_black_two_white_l211_211534


namespace proof1_proof2_monotonically_increasing_interval_l211_211634

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - 0.5 * Real.cos (2 * x)

theorem proof1 : ∀ x : ℝ, f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

theorem proof2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → -0.5 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem monotonically_increasing_interval (k : ℤ) : 
∃ lb ub : ℝ, lb = Real.pi / 6 + k * Real.pi ∧ ub = 2 * Real.pi / 3 + k * Real.pi ∧ ∀ x : ℝ, lb ≤ x ∧ x ≤ ub → f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

end proof1_proof2_monotonically_increasing_interval_l211_211634


namespace product_probability_correct_l211_211312

/-- Define probabilities for spins of Paco and Dani --/
def prob_paco := 1 / 5
def prob_dani := 1 / 15

/-- Define the probability that the product of spins is less than 30 --/
def prob_product_less_than_30 : ℚ :=
  (2 / 5) + (1 / 5) * (9 / 15) + (1 / 5) * (7 / 15) + (1 / 5) * (5 / 15)

theorem product_probability_correct : prob_product_less_than_30 = 17 / 25 :=
by sorry

end product_probability_correct_l211_211312


namespace value_of_expression_l211_211026

theorem value_of_expression (x y : ℤ) (hx : x = -5) (hy : y = 8) : 2 * (x - y) ^ 2 - x * y = 378 :=
by
  rw [hx, hy]
  -- The proof goes here.
  sorry

end value_of_expression_l211_211026


namespace round_nearest_hundredth_problem_l211_211859

noncomputable def round_nearest_hundredth (x : ℚ) : ℚ :=
  let shifted := x * 100
  let rounded := if (shifted - shifted.floor) < 0.5 then shifted.floor else shifted.ceil
  rounded / 100

theorem round_nearest_hundredth_problem :
  let A := 34.561
  let B := 34.558
  let C := 34.5539999
  let D := 34.5601
  let E := 34.56444
  round_nearest_hundredth A = 34.56 ∧
  round_nearest_hundredth B = 34.56 ∧
  round_nearest_hundredth C ≠ 34.56 ∧
  round_nearest_hundredth D = 34.56 ∧
  round_nearest_hundredth E = 34.56 :=
sorry

end round_nearest_hundredth_problem_l211_211859


namespace fraction_simplification_l211_211606

theorem fraction_simplification : 
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1 / 3 :=
by
  sorry

end fraction_simplification_l211_211606


namespace SugarWeightLoss_l211_211667

noncomputable def sugar_fraction_lost : Prop :=
  let green_beans_weight := 60
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_lost := (1 / 3) * rice_weight
  let remaining_weight := 120
  let total_initial_weight := green_beans_weight + rice_weight + sugar_weight
  let total_lost := total_initial_weight - remaining_weight
  let sugar_lost := total_lost - rice_lost
  let expected_fraction := (sugar_lost / sugar_weight)
  expected_fraction = (1 / 5)

theorem SugarWeightLoss : sugar_fraction_lost := by
  sorry

end SugarWeightLoss_l211_211667


namespace gcd_459_357_is_51_l211_211102

-- Define the problem statement
theorem gcd_459_357_is_51 : Nat.gcd 459 357 = 51 :=
by
  -- Proof here
  sorry

end gcd_459_357_is_51_l211_211102


namespace number_of_buses_used_l211_211001

-- Definitions based on the conditions
def total_students : ℕ := 360
def students_per_bus : ℕ := 45

-- The theorem we need to prove
theorem number_of_buses_used : total_students / students_per_bus = 8 := 
by sorry

end number_of_buses_used_l211_211001


namespace fraction_power_equality_l211_211571

theorem fraction_power_equality :
  (72000 ^ 4) / (24000 ^ 4) = 81 := 
by
  sorry

end fraction_power_equality_l211_211571


namespace simplify_expression_l211_211254

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end simplify_expression_l211_211254


namespace bucky_savings_excess_l211_211415

def cost_of_game := 60
def saved_amount := 15
def fish_earnings_weekends (fish : String) : ℕ :=
  match fish with
  | "trout" => 5
  | "bluegill" => 4
  | "bass" => 7
  | "catfish" => 6
  | _ => 0

def fish_earnings_weekdays (fish : String) : ℕ :=
  match fish with
  | "trout" => 10
  | "bluegill" => 8
  | "bass" => 14
  | "catfish" => 12
  | _ => 0

def sunday_fish := 10
def weekday_fish := 3
def weekdays := 2

def sunday_fish_distribution := [
  ("trout", 3),
  ("bluegill", 2),
  ("bass", 4),
  ("catfish", 1)
]

noncomputable def sunday_earnings : ℕ :=
  sunday_fish_distribution.foldl (λ acc (fish, count) =>
    acc + count * fish_earnings_weekends fish) 0

noncomputable def weekday_earnings : ℕ :=
  weekdays * weekday_fish * (
    fish_earnings_weekdays "trout" +
    fish_earnings_weekdays "bluegill" +
    fish_earnings_weekdays "bass")

noncomputable def total_earnings : ℕ :=
  sunday_earnings + weekday_earnings

noncomputable def total_savings : ℕ :=
  total_earnings + saved_amount

theorem bucky_savings_excess :
  total_savings - cost_of_game = 76 :=
by sorry

end bucky_savings_excess_l211_211415


namespace red_black_probability_l211_211223

-- Define the number of cards and ranks
def num_cards : ℕ := 64
def num_ranks : ℕ := 16

-- Define the suits and their properties
def suits := 6
def red_suits := 3
def black_suits := 3
def cards_per_suit := num_ranks

-- Define the number of red and black cards
def red_cards := red_suits * cards_per_suit
def black_cards := black_suits * cards_per_suit

-- Prove the probability that the top card is red and the second card is black
theorem red_black_probability : 
  (red_cards * black_cards) / (num_cards * (num_cards - 1)) = 3 / 4 := by 
  sorry

end red_black_probability_l211_211223


namespace total_weight_fruits_in_good_condition_l211_211805

theorem total_weight_fruits_in_good_condition :
  let oranges_initial := 600
  let bananas_initial := 400
  let apples_initial := 300
  let avocados_initial := 200
  let grapes_initial := 100
  let pineapples_initial := 50

  let oranges_rotten := 0.15 * oranges_initial
  let bananas_rotten := 0.05 * bananas_initial
  let apples_rotten := 0.08 * apples_initial
  let avocados_rotten := 0.10 * avocados_initial
  let grapes_rotten := 0.03 * grapes_initial
  let pineapples_rotten := 0.20 * pineapples_initial

  let oranges_good := oranges_initial - oranges_rotten
  let bananas_good := bananas_initial - bananas_rotten
  let apples_good := apples_initial - apples_rotten
  let avocados_good := avocados_initial - avocados_rotten
  let grapes_good := grapes_initial - grapes_rotten
  let pineapples_good := pineapples_initial - pineapples_rotten

  let weight_per_orange := 150 / 1000 -- kg
  let weight_per_banana := 120 / 1000 -- kg
  let weight_per_apple := 100 / 1000 -- kg
  let weight_per_avocado := 80 / 1000 -- kg
  let weight_per_grape := 5 / 1000 -- kg
  let weight_per_pineapple := 1 -- kg

  oranges_good * weight_per_orange +
  bananas_good * weight_per_banana +
  apples_good * weight_per_apple +
  avocados_good * weight_per_avocado +
  grapes_good * weight_per_grape +
  pineapples_good * weight_per_pineapple = 204.585 :=
by
  sorry

end total_weight_fruits_in_good_condition_l211_211805


namespace equilibrium_possible_l211_211305

theorem equilibrium_possible (n : ℕ) : (∃ k : ℕ, 4 * k = n) ∨ (∃ k : ℕ, 4 * k + 3 = n) ↔
  (∃ S1 S2 : Finset ℕ, S1 ∪ S2 = Finset.range (n+1) ∧
                     S1 ∩ S2 = ∅ ∧
                     S1.sum id = S2.sum id) := 
sorry

end equilibrium_possible_l211_211305


namespace cubic_inequality_l211_211639

theorem cubic_inequality :
  {x : ℝ | x^3 - 12*x^2 + 47*x - 60 < 0} = {x : ℝ | 3 < x ∧ x < 5} :=
by
  sorry

end cubic_inequality_l211_211639


namespace smallest_b_value_minimizes_l211_211947

noncomputable def smallest_b_value (a b : ℝ) (c : ℝ := 2) : ℝ :=
  if (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) then b else 0

theorem smallest_b_value_minimizes (a b : ℝ) (c : ℝ := 2) :
  (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) →
  b = 2 :=
by sorry

end smallest_b_value_minimizes_l211_211947


namespace solve_quadratic_inequality_l211_211300

theorem solve_quadratic_inequality (x : ℝ) : 3 * x^2 - 5 * x - 2 < 0 → (-1 / 3 < x ∧ x < 2) :=
by
  intro h
  sorry

end solve_quadratic_inequality_l211_211300


namespace division_result_l211_211529

theorem division_result : (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3)) = 124 / 509 := 
by
  sorry

end division_result_l211_211529


namespace exists_n_l211_211524

def F_n (a n : ℕ) : ℕ :=
  let q := a ^ (1 / n)
  let r := a % n
  q + r

noncomputable def largest_A : ℕ :=
  53590

theorem exists_n (a : ℕ) (h : a ≤ largest_A) :
  ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    F_n (F_n (F_n (F_n (F_n (F_n a n1) n2) n3) n4) n5) n6 = 1 := 
sorry

end exists_n_l211_211524


namespace william_won_more_rounds_than_harry_l211_211700

def rounds_played : ℕ := 15
def william_won_rounds : ℕ := 10
def harry_won_rounds : ℕ := rounds_played - william_won_rounds
def william_won_more_rounds := william_won_rounds > harry_won_rounds

theorem william_won_more_rounds_than_harry : william_won_rounds - harry_won_rounds = 5 := 
by sorry

end william_won_more_rounds_than_harry_l211_211700


namespace inequality_proof_l211_211958

variable (m : ℕ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1)

theorem inequality_proof :
    (m > 0) →
    (x^m / ((1 + y) * (1 + z)) + y^m / ((1 + x) * (1 + z)) + z^m / ((1 + x) * (1 + y)) >= 3/4) :=
by
  intro hm_pos
  -- Proof skipped
  sorry

end inequality_proof_l211_211958


namespace cos2x_quadratic_eq_specific_values_l211_211397

variable (a b c x : ℝ)

axiom eqn1 : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0

noncomputable def quadratic_equation_cos2x 
  (a b c : ℝ) : ℝ × ℝ × ℝ := 
  (a^2, 2*a^2 + 2*a*c - b^2, a^2 + 2*a*c - b^2 + 4*c^2)

theorem cos2x_quadratic_eq 
  (a b c x : ℝ) 
  (h: a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0) :
  (a^2) * (Real.cos (2*x))^2 + 
  (2*a^2 + 2*a*c - b^2) * Real.cos (2*x) + 
  (a^2 + 2*a*c - b^2 + 4*c^2) = 0 :=
sorry

theorem specific_values : 
  quadratic_equation_cos2x 4 2 (-1) = (4, 2, -1) :=
by
  unfold quadratic_equation_cos2x
  simp
  sorry

end cos2x_quadratic_eq_specific_values_l211_211397


namespace inequality_1_inequality_2_inequality_4_l211_211965

theorem inequality_1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

theorem inequality_2 (a : ℝ) : a * (1 - a) ≤ 1 / 4 := sorry

theorem inequality_4 (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := sorry

end inequality_1_inequality_2_inequality_4_l211_211965


namespace min_radius_for_area_l211_211793

theorem min_radius_for_area (r : ℝ) (π : ℝ) (A : ℝ) (h1 : A = 314) (h2 : A = π * r^2) : r ≥ 10 :=
by
  sorry

end min_radius_for_area_l211_211793


namespace largest_possible_median_l211_211078

theorem largest_possible_median 
  (l : List ℕ)
  (h_l : l = [4, 5, 3, 7, 9, 6])
  (h_pos : ∀ n ∈ l, 0 < n)
  (additional : List ℕ)
  (h_additional_pos : ∀ n ∈ additional, 0 < n)
  (h_length : l.length + additional.length = 9) : 
  ∃ median, median = 7 :=
by
  sorry

end largest_possible_median_l211_211078


namespace servant_cash_received_l211_211530

theorem servant_cash_received (annual_cash : ℕ) (turban_price : ℕ) (served_months : ℕ) (total_months : ℕ) (cash_received : ℕ) :
  annual_cash = 90 → turban_price = 50 → served_months = 9 → total_months = 12 → 
  cash_received = (annual_cash + turban_price) * served_months / total_months - turban_price → 
  cash_received = 55 :=
by {
  intros;
  sorry
}

end servant_cash_received_l211_211530


namespace tony_rollercoasters_l211_211345

theorem tony_rollercoasters :
  let s1 := 50 -- speed of the first rollercoaster
  let s2 := 62 -- speed of the second rollercoaster
  let s3 := 73 -- speed of the third rollercoaster
  let s4 := 70 -- speed of the fourth rollercoaster
  let s5 := 40 -- speed of the fifth rollercoaster
  let avg_speed := 59 -- Tony's average speed during the day
  let total_speed := s1 + s2 + s3 + s4 + s5
  total_speed / avg_speed = 5 := sorry

end tony_rollercoasters_l211_211345


namespace find_value_l211_211098

variable (y : ℝ) (Q : ℝ)
axiom condition : 5 * (3 * y + 7 * Real.pi) = Q

theorem find_value : 10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by
  sorry

end find_value_l211_211098


namespace functional_equation_solution_l211_211035

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by
  intro h
  sorry

end functional_equation_solution_l211_211035


namespace sum_of_first_nine_terms_l211_211533

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (n / 2) * (a 0 + a (n - 1))

theorem sum_of_first_nine_terms 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms a S)
  (h_sum_terms : a 2 + a 3 + a 4 + a 5 + a 6 = 20) :
  S 9 = 36 :=
sorry

end sum_of_first_nine_terms_l211_211533


namespace rationalize_denominator_sum_equals_49_l211_211410

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end rationalize_denominator_sum_equals_49_l211_211410


namespace solution_set_of_quadratic_inequality_l211_211232

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l211_211232


namespace find_number_l211_211209

-- Define the conditions: 0.80 * x - 20 = 60
variables (x : ℝ)
axiom condition : 0.80 * x - 20 = 60

-- State the theorem that x = 100 given the condition
theorem find_number : x = 100 :=
by
  sorry

end find_number_l211_211209


namespace Pradeep_marks_l211_211230

variable (T : ℕ) (P : ℕ) (F : ℕ)

def passing_marks := P * T / 100

theorem Pradeep_marks (hT : T = 925) (hP : P = 20) (hF : F = 25) :
  (passing_marks P T) - F = 160 :=
by
  sorry

end Pradeep_marks_l211_211230


namespace minimize_expression_l211_211592

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a ≠ 0)

theorem minimize_expression : 
  (a > b) → (b > c) → (a ≠ 0) → 
  ∃ x : ℝ, x = 4 ∧ ∀ y, y = (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 → x ≤ y := sorry

end minimize_expression_l211_211592


namespace inequality_proof_l211_211488

theorem inequality_proof 
  (x y z w : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w)
  (h_eq : (x^3 + y^3)^4 = z^3 + w^3) :
  x^4 * z + y^4 * w ≥ z * w :=
sorry

end inequality_proof_l211_211488


namespace dealer_profit_percentage_l211_211264

-- Definitions of conditions
def cost_price (C : ℝ) : ℝ := C
def list_price (C : ℝ) : ℝ := 1.5 * C
def discount_rate : ℝ := 0.1
def discounted_price (C : ℝ) : ℝ := (1 - discount_rate) * list_price C
def price_for_45_articles (C : ℝ) : ℝ := 45 * discounted_price C
def cost_for_40_articles (C : ℝ) : ℝ := 40 * cost_price C

-- Statement of the problem
theorem dealer_profit_percentage (C : ℝ) (h₀ : C > 0) :
  (price_for_45_articles C - cost_for_40_articles C) / cost_for_40_articles C * 100 = 35 :=  
sorry

end dealer_profit_percentage_l211_211264


namespace impossible_list_10_numbers_with_given_conditions_l211_211736

theorem impossible_list_10_numbers_with_given_conditions :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 0 ≤ i ∧ i ≤ 7 → (a i * a (i + 1) * a (i + 2)) % 6 = 0) ∧
    (∀ i, 0 ≤ i ∧ i ≤ 8 → (a i * a (i + 1)) % 6 ≠ 0) :=
by
  sorry

end impossible_list_10_numbers_with_given_conditions_l211_211736


namespace value_of_b_l211_211213

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, (-x^2 + b * x - 7 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by
  sorry

end value_of_b_l211_211213


namespace chromium_percentage_new_alloy_l211_211658

theorem chromium_percentage_new_alloy :
  let wA := 15
  let pA := 0.12
  let wB := 30
  let pB := 0.08
  let wC := 20
  let pC := 0.20
  let wD := 35
  let pD := 0.05
  let total_weight := wA + wB + wC + wD
  let total_chromium := (wA * pA) + (wB * pB) + (wC * pC) + (wD * pD)
  total_weight = 100 ∧ total_chromium = 9.95 → total_chromium / total_weight * 100 = 9.95 :=
by
  sorry

end chromium_percentage_new_alloy_l211_211658


namespace ways_to_choose_providers_l211_211498

theorem ways_to_choose_providers : (25 * 24 * 23 * 22 = 303600) :=
by
  sorry

end ways_to_choose_providers_l211_211498


namespace solve_for_x_l211_211020

theorem solve_for_x (a b c x : ℝ) (h : x^2 + b^2 + c = (a + x)^2) : 
  x = (b^2 + c - a^2) / (2 * a) :=
by sorry

end solve_for_x_l211_211020


namespace length_of_CB_l211_211636

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end length_of_CB_l211_211636


namespace adam_tickets_left_l211_211825

def tickets_left (total_tickets : ℕ) (ticket_cost : ℕ) (total_spent : ℕ) : ℕ :=
  total_tickets - total_spent / ticket_cost

theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := 
by
  sorry

end adam_tickets_left_l211_211825


namespace problem1_problem2_l211_211598

-- Problem 1: Prove that (2a^2 b) * a b^2 / 4a^3 = 1/2 b^3
theorem problem1 (a b : ℝ) : (2 * a^2 * b) * (a * b^2) / (4 * a^3) = (1 / 2) * b^3 :=
  sorry

-- Problem 2: Prove that (2x + 5)(x - 3) = 2x^2 - x - 15
theorem problem2 (x : ℝ): (2 * x + 5) * (x - 3) = 2 * x^2 - x - 15 :=
  sorry

end problem1_problem2_l211_211598


namespace theater_tickets_l211_211319

theorem theater_tickets (O B P : ℕ) (h1 : O + B + P = 550) 
  (h2 : 15 * O + 10 * B + 25 * P = 9750) (h3: P = 5 * O) (h4 : O ≥ 50) : 
  B - O = 179 :=
by
  sorry

end theater_tickets_l211_211319


namespace find_corresponding_element_l211_211362

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem find_corresponding_element :
  f (-1, 2) = (-3, 1) :=
by
  sorry

end find_corresponding_element_l211_211362


namespace degree_to_radian_l211_211272

theorem degree_to_radian (h : 1 = (π / 180)) : 60 = π * (1 / 3) := 
sorry

end degree_to_radian_l211_211272


namespace james_spends_on_pistachios_per_week_l211_211367

theorem james_spends_on_pistachios_per_week :
  let cost_per_can := 10
  let ounces_per_can := 5
  let total_ounces_per_5_days := 30
  let days_per_week := 7
  let cost_per_ounce := cost_per_can / ounces_per_can
  let daily_ounces := total_ounces_per_5_days / 5
  let daily_cost := daily_ounces * cost_per_ounce
  daily_cost * days_per_week = 84 :=
by
  sorry

end james_spends_on_pistachios_per_week_l211_211367


namespace cordelia_bleaching_l211_211578

noncomputable def bleaching_time (B : ℝ) : Prop :=
  B + 4 * B + B / 3 = 10

theorem cordelia_bleaching : ∃ B : ℝ, bleaching_time B ∧ B = 1.875 :=
by {
  sorry
}

end cordelia_bleaching_l211_211578


namespace num_students_only_math_l211_211814

def oakwood_ninth_grade_problem 
  (total_students: ℕ)
  (students_in_math: ℕ)
  (students_in_foreign_language: ℕ)
  (students_in_science: ℕ)
  (students_in_all_three: ℕ)
  (students_total_from_ie: ℕ) :=
  (total_students = 120) ∧
  (students_in_math = 85) ∧
  (students_in_foreign_language = 65) ∧
  (students_in_science = 75) ∧
  (students_in_all_three = 20) ∧
  total_students = students_in_math + students_in_foreign_language + students_in_science 
  - (students_total_from_ie) + students_in_all_three - (students_in_all_three)

theorem num_students_only_math 
  (total_students: ℕ := 120)
  (students_in_math: ℕ := 85)
  (students_in_foreign_language: ℕ := 65)
  (students_in_science: ℕ := 75)
  (students_in_all_three: ℕ := 20)
  (students_total_from_ie: ℕ := 45) :
  oakwood_ninth_grade_problem total_students students_in_math students_in_foreign_language students_in_science students_in_all_three students_total_from_ie →
  ∃ (students_only_math: ℕ), students_only_math = 75 :=
by
  sorry

end num_students_only_math_l211_211814


namespace sally_total_fries_is_50_l211_211419

-- Definitions for the conditions
def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 3 * 12
def mark_fraction_given_to_sally : ℕ := mark_initial_fries / 3
def jessica_total_cm_of_fries : ℕ := 240
def fry_length_cm : ℕ := 5
def jessica_total_fries : ℕ := jessica_total_cm_of_fries / fry_length_cm
def jessica_fraction_given_to_sally : ℕ := jessica_total_fries / 2

-- Definition for the question
def total_fries_sally_has (sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally : ℕ) : ℕ :=
  sally_initial_fries + mark_fraction_given_to_sally + jessica_fraction_given_to_sally

-- The theorem to be proved
theorem sally_total_fries_is_50 :
  total_fries_sally_has sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally = 50 :=
sorry

end sally_total_fries_is_50_l211_211419


namespace find_decreased_amount_l211_211271

variables (x y : ℝ)

axiom h1 : 0.20 * x - y = 6
axiom h2 : x = 50.0

theorem find_decreased_amount : y = 4 :=
by
  sorry

end find_decreased_amount_l211_211271


namespace max_servings_l211_211357

-- Define available chunks for each type of fruit
def available_cantaloupe := 150
def available_honeydew := 135
def available_pineapple := 60
def available_watermelon := 220

-- Define the required chunks per serving for each type of fruit
def chunks_per_serving_cantaloupe := 3
def chunks_per_serving_honeydew := 2
def chunks_per_serving_pineapple := 1
def chunks_per_serving_watermelon := 4

-- Define the minimum required servings
def minimum_servings := 50

-- Prove the greatest number of servings that can be made while maintaining the specific ratio
theorem max_servings : 
  ∀ s : ℕ, 
  s * chunks_per_serving_cantaloupe ≤ available_cantaloupe ∧
  s * chunks_per_serving_honeydew ≤ available_honeydew ∧
  s * chunks_per_serving_pineapple ≤ available_pineapple ∧
  s * chunks_per_serving_watermelon ≤ available_watermelon ∧ 
  s ≥ minimum_servings → 
  s = 50 :=
by
  sorry

end max_servings_l211_211357


namespace point_D_eq_1_2_l211_211285

-- Definitions and conditions
def point : Type := ℝ × ℝ

def A : point := (-1, 4)
def B : point := (-4, -1)
def C : point := (4, 7)

-- Translate function
def translate (p : point) (dx dy : ℝ) := (p.1 + dx, p.2 + dy)

-- The translation distances found from A to C
def dx := C.1 - A.1
def dy := C.2 - A.2

-- The point D
def D : point := translate B dx dy

-- Proof objective
theorem point_D_eq_1_2 : D = (1, 2) := by
  sorry

end point_D_eq_1_2_l211_211285


namespace complex_number_solution_l211_211984

theorem complex_number_solution (i : ℂ) (h : i^2 = -1) : (5 / (2 - i) - i = 2) :=
  sorry

end complex_number_solution_l211_211984


namespace intersection_eq_singleton_zero_l211_211407

-- Definition of the sets M and N
def M : Set ℤ := {0, 1}
def N : Set ℤ := { x | ∃ n : ℤ, x = 2 * n }

-- The theorem stating that the intersection of M and N is {0}
theorem intersection_eq_singleton_zero : M ∩ N = {0} :=
by
  sorry

end intersection_eq_singleton_zero_l211_211407


namespace arithmetic_sum_S11_l211_211951

theorem arithmetic_sum_S11 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h_arith : ∀ n, a (n+1) - a n = d) -- The sequence is arithmetic with common difference d
  (h_sum : S n = n * (a 1 + a n) / 2) -- Sum of the first n terms definition
  (h_condition: a 3 + a 6 + a 9 = 54) :
  S 11 = 198 := 
sorry

end arithmetic_sum_S11_l211_211951


namespace values_of_a_l211_211679

noncomputable def quadratic_eq (a x : ℝ) : ℝ :=
(a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

theorem values_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_eq a x = 0 → x ≥ 0) ↔ (a = 3 ∨ (-1 ≤ a ∧ a ≤ 1)) :=
sorry

end values_of_a_l211_211679


namespace maximum_value_expression_l211_211740

-- Definitions
def f (x : ℝ) := -3 * x^2 + 18 * x - 1

-- Lean statement to prove that the maximum value of the function f is 26.
theorem maximum_value_expression : ∃ x : ℝ, f x = 26 :=
sorry

end maximum_value_expression_l211_211740


namespace sum_of_possible_values_of_z_l211_211317

theorem sum_of_possible_values_of_z (x y z : ℂ) 
  (h₁ : z^2 + 5 * x = 10 * z)
  (h₂ : y^2 + 5 * z = 10 * y)
  (h₃ : x^2 + 5 * y = 10 * x) :
  z = 0 ∨ z = 9 / 5 := by
  sorry

end sum_of_possible_values_of_z_l211_211317


namespace sin_vertex_angle_isosceles_triangle_l211_211137

theorem sin_vertex_angle_isosceles_triangle (α β : ℝ) (h_isosceles : β = 2 * α) (tan_base_angle : Real.tan α = 2 / 3) :
  Real.sin β = 12 / 13 := 
sorry

end sin_vertex_angle_isosceles_triangle_l211_211137


namespace mr_a_net_gain_l211_211866

theorem mr_a_net_gain 
  (initial_value : ℝ)
  (sale_profit_percentage : ℝ)
  (buyback_loss_percentage : ℝ)
  (final_sale_price : ℝ) 
  (buyback_price : ℝ)
  (net_gain : ℝ) :
  initial_value = 12000 →
  sale_profit_percentage = 0.15 →
  buyback_loss_percentage = 0.12 →
  final_sale_price = initial_value * (1 + sale_profit_percentage) →
  buyback_price = final_sale_price * (1 - buyback_loss_percentage) →
  net_gain = final_sale_price - buyback_price →
  net_gain = 1656 :=
by
  sorry

end mr_a_net_gain_l211_211866


namespace exists_four_distinct_numbers_with_equal_half_sum_l211_211151

theorem exists_four_distinct_numbers_with_equal_half_sum (S : Finset ℕ) (h_card : S.card = 10) (h_range : ∀ x ∈ S, x ≤ 23) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S) ∧ (a + b = c + d) :=
by
  sorry

end exists_four_distinct_numbers_with_equal_half_sum_l211_211151


namespace cards_thrown_away_l211_211111

theorem cards_thrown_away (h1 : 3 * (52 / 2) + 3 * 52 - 200 = 34) : 34 = 34 :=
by sorry

end cards_thrown_away_l211_211111


namespace endpoints_undetermined_l211_211929

theorem endpoints_undetermined (m : ℝ → ℝ) :
  (∀ x, m x = x - 2) ∧ (∃ mid : ℝ × ℝ, ∃ (x1 x2 y1 y2 : ℝ), 
    mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m mid.1 = mid.2) → 
  ¬ (∃ (x1 x2 y1 y2 : ℝ), mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m ((x1 + x2) / 2) = (y1 + y2) / 2 ∧
    x1 = the_exact_endpoint ∧ x2 = the_exact_other_endpoint) :=
by sorry

end endpoints_undetermined_l211_211929


namespace arith_seq_100th_term_l211_211838

noncomputable def arithSeq (a : ℤ) (n : ℕ) : ℤ :=
  a - 1 + (n - 1) * ((a + 1) - (a - 1))

theorem arith_seq_100th_term (a : ℤ) : arithSeq a 100 = 197 := by
  sorry

end arith_seq_100th_term_l211_211838


namespace perpendicular_vectors_x_value_l211_211033

theorem perpendicular_vectors_x_value
  (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (1, -2)) (hb : b = (-3, x))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -3 / 2 := by
  sorry

end perpendicular_vectors_x_value_l211_211033


namespace mean_score_of_all_students_l211_211944

-- Conditions
def M : ℝ := 90
def A : ℝ := 75
def ratio (m a : ℝ) : Prop := m / a = 2 / 3

-- Question and correct answer
theorem mean_score_of_all_students (m a : ℝ) (hm : ratio m a) : (60 * a + 75 * a) / (5 * a / 3) = 81 := by
  sorry

end mean_score_of_all_students_l211_211944


namespace cos_eq_neg_four_fifths_of_tan_l211_211798

theorem cos_eq_neg_four_fifths_of_tan (α : ℝ) (h_tan : Real.tan α = 3 / 4) (h_interval : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.cos α = -4 / 5 :=
sorry

end cos_eq_neg_four_fifths_of_tan_l211_211798


namespace binomials_product_l211_211187

noncomputable def poly1 (x y : ℝ) : ℝ := 2 * x^2 + 3 * y - 4
noncomputable def poly2 (y : ℝ) : ℝ := y + 6

theorem binomials_product (x y : ℝ) :
  (poly1 x y) * (poly2 y) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 :=
by sorry

end binomials_product_l211_211187


namespace complex_quadrant_l211_211712

open Complex

theorem complex_quadrant (z : ℂ) (h : (1 + I) * z = 2 * I) : 
  z.re > 0 ∧ z.im < 0 :=
  sorry

end complex_quadrant_l211_211712


namespace min_squared_distance_l211_211670

open Real

theorem min_squared_distance : ∀ (x y : ℝ), (3 * x + y = 10) → (x^2 + y^2) ≥ 10 :=
by
  intros x y hxy
  -- Insert the necessary steps or key elements here
  sorry

end min_squared_distance_l211_211670


namespace fraction_equation_l211_211932

theorem fraction_equation (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end fraction_equation_l211_211932


namespace train_length_l211_211867

theorem train_length (L : ℕ) (V : ℕ) (platform_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
    (h1 : V = L / time_pole) 
    (h2 : V = (L + platform_length) / time_platform) :
    L = 300 := 
by 
  -- The proof can be filled here
  sorry

end train_length_l211_211867


namespace vehicle_A_no_speed_increase_needed_l211_211401

noncomputable def V_A := 60 -- Speed of Vehicle A in mph
noncomputable def V_B := 70 -- Speed of Vehicle B in mph
noncomputable def V_C := 50 -- Speed of Vehicle C in mph
noncomputable def dist_AB := 100 -- Initial distance between A and B in ft
noncomputable def dist_AC := 300 -- Initial distance between A and C in ft

theorem vehicle_A_no_speed_increase_needed 
  (V_A V_B V_C : ℝ)
  (dist_AB dist_AC : ℝ)
  (h1 : V_A > V_C)
  (h2 : V_A = 60)
  (h3 : V_B = 70)
  (h4 : V_C = 50)
  (h5 : dist_AB = 100)
  (h6 : dist_AC = 300) : 
  ∀ ΔV : ℝ, ΔV = 0 :=
by
  sorry -- Proof to be filled out

end vehicle_A_no_speed_increase_needed_l211_211401


namespace production_company_keeps_60_percent_l211_211981

noncomputable def openingWeekendRevenue : ℝ := 120
noncomputable def productionCost : ℝ := 60
noncomputable def profit : ℝ := 192
noncomputable def totalRevenue : ℝ := 3.5 * openingWeekendRevenue
noncomputable def amountKept : ℝ := profit + productionCost
noncomputable def percentageKept : ℝ := (amountKept / totalRevenue) * 100

theorem production_company_keeps_60_percent :
  percentageKept = 60 :=
by
  sorry

end production_company_keeps_60_percent_l211_211981


namespace intersection_points_form_line_slope_l211_211559

theorem intersection_points_form_line_slope (s : ℝ) :
  ∃ (m : ℝ), m = 1/18 ∧ ∀ (x y : ℝ),
    (3 * x + y = 5 * s + 6) ∧ (2 * x - 3 * y = 3 * s - 5) →
    ∃ k : ℝ, (y = m * x + k) :=
by
  sorry

end intersection_points_form_line_slope_l211_211559


namespace average_salary_is_8000_l211_211361

def average_salary_all_workers (A : ℝ) :=
  let total_workers := 30
  let technicians := 10
  let technician_salary := 12000
  let rest_workers := total_workers - technicians
  let rest_salary := 6000
  let total_salary := (technicians * technician_salary) + (rest_workers * rest_salary)
  A = total_salary / total_workers

theorem average_salary_is_8000 : average_salary_all_workers 8000 :=
by
  sorry

end average_salary_is_8000_l211_211361


namespace line_slope_point_l211_211355

theorem line_slope_point (m b : ℝ) (h_slope : m = -4) (h_point : ∃ x y : ℝ, (x, y) = (5, 2) ∧ y = m * x + b) : 
  m + b = 18 := by
  sorry

end line_slope_point_l211_211355


namespace total_artworks_l211_211813

theorem total_artworks (students : ℕ) (group1_artworks : ℕ) (group2_artworks : ℕ) (total_students : students = 10) 
    (artwork_group1 : group1_artworks = 5 * 3) (artwork_group2 : group2_artworks = 5 * 4) : 
    group1_artworks + group2_artworks = 35 :=
by
  sorry

end total_artworks_l211_211813


namespace fourth_vertex_of_square_l211_211787

theorem fourth_vertex_of_square (A B C D : ℂ) : 
  A = (2 + 3 * I) ∧ B = (-3 + 2 * I) ∧ C = (-2 - 3 * I) →
  D = (0 - 0.5 * I) :=
sorry

end fourth_vertex_of_square_l211_211787


namespace find_a_b_and_range_of_c_l211_211600

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c

theorem find_a_b_and_range_of_c (c : ℝ) (h1 : ∀ x, 3 * x^2 - 2 * 3 * x - 9 = 0 → x = -1 ∨ x = 3)
    (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 6 → f x 3 (-9) c < c^2 + 4 * c) : 
    (a = 3 ∧ b = -9) ∧ (c > 6 ∨ c < -9) := by
  sorry

end find_a_b_and_range_of_c_l211_211600


namespace zoey_finished_on_monday_l211_211284

def total_days_read (n : ℕ) : ℕ :=
  2 * ((2^n) - 1)

def day_of_week_finished (start_day : ℕ) (total_days : ℕ) : ℕ :=
  (start_day + total_days) % 7

theorem zoey_finished_on_monday :
  day_of_week_finished 1 (total_days_read 18) = 1 :=
by
  sorry

end zoey_finished_on_monday_l211_211284


namespace smallest_four_digit_number_l211_211579

theorem smallest_four_digit_number : 
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (∃ (AB CD : ℕ), 
      n = 1000 * (AB / 10) + 100 * (AB % 10) + CD ∧
      ((AB / 10) * 10 + (AB % 10) + 2) * CD = 100 ∧ 
      n / CD = ((AB / 10) * 10 + (AB % 10) + 1)^2) ∧
    n = 1805 :=
by
  sorry

end smallest_four_digit_number_l211_211579


namespace solve_for_q_l211_211253

theorem solve_for_q :
  ∀ (k l q : ℚ),
    (3 / 4 = k / 108) →
    (3 / 4 = (l + k) / 126) →
    (3 / 4 = (q - l) / 180) →
    q = 148.5 :=
by
  intros k l q hk hl hq
  sorry

end solve_for_q_l211_211253


namespace joans_remaining_kittens_l211_211526

theorem joans_remaining_kittens (initial_kittens given_away : ℕ) (h1 : initial_kittens = 15) (h2 : given_away = 7) : initial_kittens - given_away = 8 := sorry

end joans_remaining_kittens_l211_211526


namespace cuckoo_sounds_from_10_to_16_l211_211848

-- Define a function for the cuckoo sounds per hour considering the clock
def cuckoo_sounds (h : ℕ) : ℕ :=
  if h ≤ 12 then h else h - 12

-- Define the total number of cuckoo sounds from 10:00 to 16:00
def total_cuckoo_sounds : ℕ :=
  (List.range' 10 (16 - 10 + 1)).map cuckoo_sounds |>.sum

theorem cuckoo_sounds_from_10_to_16 : total_cuckoo_sounds = 43 := by
  sorry

end cuckoo_sounds_from_10_to_16_l211_211848


namespace arithmetic_sequence_sum_l211_211995

theorem arithmetic_sequence_sum (a b d : ℕ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ)
  (h1 : a₁ + a₂ + a₃ = 39)
  (h2 : a₄ + a₅ + a₆ = 27)
  (h3 : a₄ = a₁ + 3 * d)
  (h4 : a₅ = a₂ + 3 * d)
  (h5 : a₆ = a₃ + 3 * d)
  (h6 : a₇ = a₄ + 3 * d)
  (h7 : a₈ = a₅ + 3 * d)
  (h8 : a₉ = a₆ + 3 * d) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 81 :=
sorry

end arithmetic_sequence_sum_l211_211995


namespace isosceles_right_triangle_third_angle_l211_211744

/-- In an isosceles right triangle where one of the angles opposite the equal sides measures 45 degrees, 
    the measure of the third angle is 90 degrees. -/
theorem isosceles_right_triangle_third_angle (θ : ℝ) 
  (h1 : θ = 45)
  (h2 : ∀ (a b c : ℝ), a + b + c = 180) : θ + θ + 90 = 180 :=
by
  sorry

end isosceles_right_triangle_third_angle_l211_211744


namespace molecular_weight_one_mole_l211_211359

variable (molecular_weight : ℕ → ℕ)

theorem molecular_weight_one_mole (h : molecular_weight 7 = 2856) :
  molecular_weight 1 = 408 :=
sorry

end molecular_weight_one_mole_l211_211359


namespace cindy_correct_answer_l211_211279

noncomputable def cindy_number (x : ℝ) : Prop :=
  (x - 10) / 5 = 40

theorem cindy_correct_answer (x : ℝ) (h : cindy_number x) : (x - 4) / 10 = 20.6 :=
by
  -- The proof is omitted as instructed
  sorry

end cindy_correct_answer_l211_211279


namespace sally_combinations_l211_211590

theorem sally_combinations :
  let wall_colors := 4
  let flooring_types := 3
  wall_colors * flooring_types = 12 := by
  sorry

end sally_combinations_l211_211590


namespace jack_total_books_is_541_l211_211006

-- Define the number of books in each section
def american_books : ℕ := 6 * 34
def british_books : ℕ := 8 * 29
def world_books : ℕ := 5 * 21

-- Define the total number of books based on the given sections
def total_books : ℕ := american_books + british_books + world_books

-- Prove that the total number of books is 541
theorem jack_total_books_is_541 : total_books = 541 :=
by
  sorry

end jack_total_books_is_541_l211_211006


namespace find_m_l211_211561

-- Define the condition for m to be within the specified range
def valid_range (m : ℤ) : Prop := -180 < m ∧ m < 180

-- Define the relationship with the trigonometric equation to be proven
def tan_eq (m : ℤ) : Prop := Real.tan (m * Real.pi / 180) = Real.tan (1500 * Real.pi / 180)

-- State the main theorem to be proved
theorem find_m (m : ℤ) (h1 : valid_range m) (h2 : tan_eq m) : m = 60 :=
sorry

end find_m_l211_211561


namespace molly_age_l211_211168

theorem molly_age (S M : ℕ) (h1 : S / M = 4 / 3) (h2 : S + 6 = 34) : M = 21 :=
by
  sorry

end molly_age_l211_211168


namespace train_speed_correct_l211_211495

theorem train_speed_correct :
  ∀ (L : ℝ) (V_man : ℝ) (T : ℝ) (V_train : ℝ),
    L = 220 ∧ V_man = 6 * (1000 / 3600) ∧ T = 11.999040076793857 ∧ 
    L / T - V_man = V_train ↔ V_train * 3.6 = 60 :=
by
  intros L V_man T V_train
  sorry

end train_speed_correct_l211_211495


namespace how_many_rocks_l211_211374

section see_saw_problem

-- Conditions
def Jack_weight : ℝ := 60
def Anna_weight : ℝ := 40
def rock_weight : ℝ := 4

-- Theorem statement
theorem how_many_rocks : (Jack_weight - Anna_weight) / rock_weight = 5 :=
by
  -- Proof is omitted, just ensuring the theorem statement
  sorry

end see_saw_problem

end how_many_rocks_l211_211374


namespace root_relationship_l211_211982

theorem root_relationship (a x₁ x₂ : ℝ) 
  (h_eqn : x₁^2 - (2*a + 1)*x₁ + a^2 + 2 = 0)
  (h_roots : x₂ = 2*x₁)
  (h_vieta1 : x₁ + x₂ = 2*a + 1)
  (h_vieta2 : x₁ * x₂ = a^2 + 2) : 
  a = 4 := 
sorry

end root_relationship_l211_211982


namespace man_speed_is_correct_l211_211543

noncomputable def train_length : ℝ := 165
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def time_seconds : ℝ := 9

-- Function to convert speed from kmph to m/s
noncomputable def kmph_to_mps (speed_kmph: ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Function to convert speed from m/s to kmph
noncomputable def mps_to_kmph (speed_mps: ℝ) : ℝ :=
  speed_mps * 3600 / 1000

-- The speed of the train in m/s
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- The relative speed of the train with respect to the man in m/s
noncomputable def relative_speed_mps : ℝ := train_length / time_seconds

-- The speed of the man in m/s
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps

-- The speed of the man in kmph
noncomputable def man_speed_kmph : ℝ := mps_to_kmph man_speed_mps

-- The statement to be proved
theorem man_speed_is_correct : man_speed_kmph = 5.976 := 
sorry

end man_speed_is_correct_l211_211543


namespace find_a1_l211_211628

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then n * a 0 else a 0 * (1 - q ^ n) / (1 - q)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Definitions from conditions
def S_3_eq_a2_plus_10a1 (a_1 a_2 S_3 : ℝ) : Prop :=
S_3 = a_2 + 10 * a_1

def a_5_eq_9 (a_5 : ℝ) : Prop :=
a_5 = 9

-- Main theorem statement
theorem find_a1 (h1 : S_3_eq_a2_plus_10a1 (a 1) (a 2) (sum_of_geometric_sequence a q 3))
                (h2 : a_5_eq_9 (a 5))
                (h3 : q ≠ 0 ∧ q ≠ 1) :
    a 1 = 1 / 9 :=
sorry

end find_a1_l211_211628


namespace contrapositive_l211_211269

theorem contrapositive (m : ℝ) :
  (∀ m > 0, ∃ x : ℝ, x^2 + x - m = 0) ↔ (∀ m ≤ 0, ∀ x : ℝ, x^2 + x - m ≠ 0) := by
  sorry

end contrapositive_l211_211269


namespace larger_integer_is_30_l211_211032

-- Define the problem statement using the given conditions
theorem larger_integer_is_30 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h1 : a / b = 5 / 2) (h2 : a * b = 360) :
  max a b = 30 :=
sorry

end larger_integer_is_30_l211_211032


namespace number_of_female_students_l211_211470

theorem number_of_female_students (M F : ℕ) (h1 : F = M + 6) (h2 : M + F = 82) : F = 44 :=
by
  sorry

end number_of_female_students_l211_211470


namespace at_least_two_equal_l211_211713

theorem at_least_two_equal (x y z : ℝ) (h1 : x * y + z = y * z + x) (h2 : y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l211_211713


namespace johns_score_is_101_l211_211208

variable (c w s : ℕ)
variable (h1 : s = 40 + 5 * c - w)
variable (h2 : s > 100)
variable (h3 : c ≤ 40)
variable (h4 : ∀ s' > 100, s' < s → ∃ c' w', s' = 40 + 5 * c' - w')

theorem johns_score_is_101 : s = 101 := by
  sorry

end johns_score_is_101_l211_211208


namespace factorize_difference_of_squares_l211_211173

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l211_211173


namespace four_digit_numbers_count_l211_211299

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l211_211299


namespace unique_root_ln_eqn_l211_211067

/-- For what values of the parameter \(a\) does the equation
   \(\ln(x - 2a) - 3(x - 2a)^2 + 2a = 0\) have a unique root? -/
theorem unique_root_ln_eqn (a : ℝ) :
  ∃! x : ℝ, (Real.log (x - 2 * a) - 3 * (x - 2 * a) ^ 2 + 2 * a = 0) ↔
  a = (1 + Real.log 6) / 4 :=
sorry

end unique_root_ln_eqn_l211_211067


namespace flight_duration_is_four_hours_l211_211130

def convert_to_moscow_time (local_time : ℕ) (time_difference : ℕ) : ℕ :=
  (local_time - time_difference) % 24

def flight_duration (departure_time arrival_time : ℕ) : ℕ :=
  (arrival_time - departure_time) % 24

def duration_per_flight (total_flight_time : ℕ) (number_of_flights : ℕ) : ℕ :=
  total_flight_time / number_of_flights

theorem flight_duration_is_four_hours :
  let MoscowToBishkekTimeDifference := 3
  let departureMoscowTime := 12
  let arrivalBishkekLocalTime := 18
  let departureBishkekLocalTime := 8
  let arrivalMoscowTime := 10
  let outboundArrivalMoscowTime := convert_to_moscow_time arrivalBishkekLocalTime MoscowToBishkekTimeDifference
  let returnDepartureMoscowTime := convert_to_moscow_time departureBishkekLocalTime MoscowToBishkekTimeDifference
  let outboundDuration := flight_duration departureMoscowTime outboundArrivalMoscowTime
  let returnDuration := flight_duration returnDepartureMoscowTime arrivalMoscowTime
  let totalFlightTime := outboundDuration + returnDuration
  duration_per_flight totalFlightTime 2 = 4 := by
  sorry

end flight_duration_is_four_hours_l211_211130


namespace molecular_weight_bleach_l211_211546

theorem molecular_weight_bleach :
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  molecular_weight = 74.44
:=
by
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  sorry

end molecular_weight_bleach_l211_211546


namespace commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l211_211386

def star (x y : ℕ) : ℕ := (x + 2) * (y + 2) - 2

theorem commutative_star : ∀ x y : ℕ, star x y = star y x := by
  sorry

theorem not_distributive_star : ∃ x y z : ℕ, star x (y + z) ≠ star x y + star x z := by
  sorry

theorem special_case_star_false : ∀ x : ℕ, star (x - 2) (x + 2) ≠ star x x - 2 := by
  sorry

theorem no_identity_star : ¬∃ e : ℕ, ∀ x : ℕ, star x e = x ∧ star e x = x := by
  sorry

-- Associativity requires further verification and does not have a definitive statement yet.

end commutative_star_not_distributive_star_special_case_star_false_no_identity_star_l211_211386


namespace smallest_integer_in_set_A_l211_211506

def set_A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_set_A : ∃ m ∈ set_A, ∀ n ∈ set_A, m ≤ n := 
  sorry

end smallest_integer_in_set_A_l211_211506


namespace not_divisible_by_n_plus_4_l211_211129

theorem not_divisible_by_n_plus_4 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 8*n + 15 = k * (n + 4) :=
sorry

end not_divisible_by_n_plus_4_l211_211129


namespace coefficient_a_must_be_zero_l211_211905

noncomputable def all_real_and_positive_roots (a b c : ℝ) : Prop :=
∀ p : ℝ, p > 0 → ∀ x : ℝ, (a * x^2 + b * x + c + p = 0) → x > 0

theorem coefficient_a_must_be_zero (a b c : ℝ) :
  (all_real_and_positive_roots a b c) → (a = 0) :=
by sorry

end coefficient_a_must_be_zero_l211_211905


namespace turnip_weight_possible_l211_211115

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end turnip_weight_possible_l211_211115


namespace cat_mouse_position_after_moves_l211_211158

-- Define the total number of moves
def total_moves : ℕ := 360

-- Define cat's cycle length and position calculation
def cat_cycle_length : ℕ := 5
def cat_final_position := total_moves % cat_cycle_length

-- Define mouse's cycle length and actual moves per cycle
def mouse_cycle_length : ℕ := 10
def mouse_effective_moves_per_cycle : ℕ := 9
def total_mouse_effective_moves := (total_moves / mouse_cycle_length) * mouse_effective_moves_per_cycle
def mouse_final_position := total_mouse_effective_moves % mouse_cycle_length

theorem cat_mouse_position_after_moves :
  cat_final_position = 0 ∧ mouse_final_position = 4 :=
by
  sorry

end cat_mouse_position_after_moves_l211_211158


namespace valid_integers_count_l211_211189

def count_valid_integers : ℕ :=
  let digits : List ℕ := [0, 1, 2, 3, 4, 6, 7, 8, 9]
  let first_digit_count := 7  -- from 2 to 9 excluding 5
  let second_digit_count := 8
  let third_digit_count := 7
  let fourth_digit_count := 6
  first_digit_count * second_digit_count * third_digit_count * fourth_digit_count

theorem valid_integers_count : count_valid_integers = 2352 := by
  -- intermediate step might include nice counting macros
  sorry

end valid_integers_count_l211_211189


namespace max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l211_211720

open Real

theorem max_min_2sinx_minus_3 : 
  ∀ x : ℝ, 
    -5 ≤ 2 * sin x - 3 ∧ 
    2 * sin x - 3 ≤ -1 :=
by sorry

theorem max_min_7_fourth_sinx_minus_sinx_squared : 
  ∀ x : ℝ, 
    -1/4 ≤ (7/4 + sin x - sin x ^ 2) ∧ 
    (7/4 + sin x - sin x ^ 2) ≤ 2 :=
by sorry

end max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l211_211720


namespace min_value_expr_l211_211630

theorem min_value_expr (x y : ℝ) : 
  ∃ min_val, min_val = 2 ∧ min_val ≤ (x + y)^2 + (x - 1/y)^2 :=
sorry

end min_value_expr_l211_211630


namespace greatest_gcd_f_l211_211445

def f (n : ℕ) : ℕ := 70 + n^2

def g (n : ℕ) : ℕ := Nat.gcd (f n) (f (n + 1))

theorem greatest_gcd_f (n : ℕ) (h : 0 < n) : g n = 281 :=
  sorry

end greatest_gcd_f_l211_211445


namespace trivia_team_l211_211265

theorem trivia_team (total_students groups students_per_group students_not_picked : ℕ) (h1 : total_students = 65)
  (h2 : groups = 8) (h3 : students_per_group = 6) (h4 : students_not_picked = total_students - groups * students_per_group) :
  students_not_picked = 17 :=
sorry

end trivia_team_l211_211265


namespace unpacked_boxes_l211_211387

-- Definitions of boxes per case
def boxesPerCaseLemonChalet : Nat := 12
def boxesPerCaseThinMints : Nat := 15
def boxesPerCaseSamoas : Nat := 10
def boxesPerCaseTrefoils : Nat := 18

-- Definitions of boxes sold by Deborah
def boxesSoldLemonChalet : Nat := 31
def boxesSoldThinMints : Nat := 26
def boxesSoldSamoas : Nat := 17
def boxesSoldTrefoils : Nat := 44

-- The theorem stating the number of boxes that will not be packed to a case
theorem unpacked_boxes :
  boxesSoldLemonChalet % boxesPerCaseLemonChalet = 7 ∧
  boxesSoldThinMints % boxesPerCaseThinMints = 11 ∧
  boxesSoldSamoas % boxesPerCaseSamoas = 7 ∧
  boxesSoldTrefoils % boxesPerCaseTrefoils = 8 := 
by
  sorry

end unpacked_boxes_l211_211387


namespace water_volume_correct_l211_211914

-- Define the conditions
def ratio_water_juice : ℕ := 5
def ratio_juice_water : ℕ := 3
def total_punch_volume : ℚ := 3  -- in liters

-- Define the question and the correct answer
def volume_of_water (ratio_water_juice ratio_juice_water : ℕ) (total_punch_volume : ℚ) : ℚ :=
  (ratio_water_juice * total_punch_volume) / (ratio_water_juice + ratio_juice_water)

-- The proof problem
theorem water_volume_correct : volume_of_water ratio_water_juice ratio_juice_water total_punch_volume = 15 / 8 :=
by
  sorry

end water_volume_correct_l211_211914


namespace vasya_made_a_mistake_l211_211815

theorem vasya_made_a_mistake :
  ∀ x : ℝ, x^4 - 3*x^3 - 2*x^2 - 4*x + 1 = 0 → ¬ x < 0 :=
by sorry

end vasya_made_a_mistake_l211_211815


namespace total_shaded_cubes_l211_211733

/-
The large cube consists of 27 smaller cubes, each face is a 3x3 grid.
Opposite faces are shaded in an identical manner, with each face having 5 shaded smaller cubes.
-/

theorem total_shaded_cubes (number_of_smaller_cubes : ℕ)
  (face_shade_pattern : ∀ (face : ℕ), ℕ)
  (opposite_face_same_shade : ∀ (face1 face2 : ℕ), face1 = face2 → face_shade_pattern face1 = face_shade_pattern face2)
  (faces_possible : ∀ (face : ℕ), face < 6)
  (each_face_shaded_squares : ∀ (face : ℕ), face_shade_pattern face = 5)
  : ∃ (n : ℕ), n = 20 :=
by
  sorry

end total_shaded_cubes_l211_211733


namespace quadratic_real_roots_range_l211_211147

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, (x^2 - x - m = 0)) ↔ m ≥ -1 / 4 :=
by sorry

end quadratic_real_roots_range_l211_211147


namespace most_cost_effective_years_l211_211633

noncomputable def total_cost (x : ℕ) : ℝ := 100000 + 15000 * x + 1000 + 2000 * ((x * (x - 1)) / 2)

noncomputable def average_annual_cost (x : ℕ) : ℝ := total_cost x / x

theorem most_cost_effective_years : ∃ (x : ℕ), x = 10 ∧
  (∀ y : ℕ, y ≠ 10 → average_annual_cost x ≤ average_annual_cost y) :=
by
  sorry

end most_cost_effective_years_l211_211633


namespace zhou_yu_age_equation_l211_211376

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end zhou_yu_age_equation_l211_211376


namespace overall_percentage_loss_l211_211097

noncomputable def original_price : ℝ := 100
noncomputable def increased_price : ℝ := original_price * 1.36
noncomputable def first_discount_price : ℝ := increased_price * 0.90
noncomputable def second_discount_price : ℝ := first_discount_price * 0.85
noncomputable def third_discount_price : ℝ := second_discount_price * 0.80
noncomputable def final_price_with_tax : ℝ := third_discount_price * 1.05
noncomputable def percentage_change : ℝ := ((final_price_with_tax - original_price) / original_price) * 100

theorem overall_percentage_loss : percentage_change = -12.6064 :=
by
  sorry

end overall_percentage_loss_l211_211097


namespace range_of_m_l211_211170

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 :=
sorry

end range_of_m_l211_211170


namespace evaluate_expression_l211_211828

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end evaluate_expression_l211_211828


namespace remainder_of_expression_l211_211110

theorem remainder_of_expression (x y u v : ℕ) (h : x = u * y + v) (Hv : 0 ≤ v ∧ v < y) :
  (if v + 2 < y then (x + 3 * u * y + 2) % y = v + 2
   else (x + 3 * u * y + 2) % y = v + 2 - y) :=
by sorry

end remainder_of_expression_l211_211110


namespace correct_average_l211_211458

theorem correct_average 
  (n : ℕ) (initial_average : ℚ) (wrong_number : ℚ) (correct_number : ℚ) (wrong_average : ℚ)
  (h_n : n = 10) 
  (h_initial : initial_average = 14) 
  (h_wrong_number : wrong_number = 26) 
  (h_correct_number : correct_number = 36) 
  (h_wrong_average : wrong_average = 14) : 
  (initial_average * n - wrong_number + correct_number) / n = 15 := 
by
  sorry

end correct_average_l211_211458


namespace simplify_polynomial_l211_211146

theorem simplify_polynomial :
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  (x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10) :=
by
  sorry

end simplify_polynomial_l211_211146


namespace absolute_inequality_l211_211074

theorem absolute_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := 
sorry

end absolute_inequality_l211_211074


namespace appropriate_grouping_43_neg78_27_neg52_l211_211169

theorem appropriate_grouping_43_neg78_27_neg52 :
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  (a + c) + (b + d) = -60 :=
by
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  sorry

end appropriate_grouping_43_neg78_27_neg52_l211_211169


namespace dartboard_distribution_count_l211_211497

-- Definition of the problem in Lean 4
def count_dartboard_distributions : ℕ :=
  -- We directly use the identified correct answer
  5

theorem dartboard_distribution_count :
  count_dartboard_distributions = 5 :=
sorry

end dartboard_distribution_count_l211_211497


namespace circle_equation_k_range_l211_211756

theorem circle_equation_k_range (k : ℝ) :
  ∀ x y: ℝ, x^2 + y^2 + 4*k*x - 2*y + 4*k^2 - k = 0 →
  k > -1 := 
sorry

end circle_equation_k_range_l211_211756


namespace integer_distances_implies_vertex_l211_211988

theorem integer_distances_implies_vertex (M A B C D : ℝ × ℝ × ℝ)
  (a b c d : ℕ)
  (h_tetrahedron: 
    dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧ 
    dist A C = 2 ∧ dist B D = 2)
  (h_distances: 
    dist M A = a ∧ dist M B = b ∧ dist M C = c ∧ dist M D = d) :
  M = A ∨ M = B ∨ M = C ∨ M = D := 
  sorry

end integer_distances_implies_vertex_l211_211988


namespace max_sum_arithmetic_prog_l211_211861

theorem max_sum_arithmetic_prog (a d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 3 = 327)
  (h2 : S 57 = 57)
  (hS : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  ∃ max_S : ℝ, max_S = 1653 := by
  sorry

end max_sum_arithmetic_prog_l211_211861


namespace school_should_purchase_bookshelves_l211_211053

theorem school_should_purchase_bookshelves
  (x : ℕ)
  (h₁ : x ≥ 20)
  (cost_A : ℕ := 20 * 300 + 100 * (x - 20))
  (cost_B : ℕ := (20 * 300 + 100 * x) * 80 / 100)
  (h₂ : cost_A = cost_B) : x = 40 :=
by sorry

end school_should_purchase_bookshelves_l211_211053


namespace smallest_y2_l211_211214

theorem smallest_y2 :
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  y2 < y1 ∧ y2 < y3 ∧ y2 < y4 :=
by
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  show y2 < y1 ∧ y2 < y3 ∧ y2 < y4
  sorry

end smallest_y2_l211_211214


namespace correct_option_is_D_l211_211280

def p : Prop := 3 ≥ 3
def q : Prop := 3 > 4

theorem correct_option_is_D (hp : p) (hq : ¬ q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬ ¬ p :=
by
  sorry

end correct_option_is_D_l211_211280


namespace quadratic_positive_difference_l211_211644
open Real

theorem quadratic_positive_difference :
  ∀ (x : ℝ), (2*x^2 - 7*x + 1 = x + 31) →
    (abs ((2 + sqrt 19) - (2 - sqrt 19)) = 2 * sqrt 19) :=
by intros x h
   sorry

end quadratic_positive_difference_l211_211644


namespace beth_speed_l211_211012

noncomputable def beth_average_speed (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ) : ℚ :=
  let jerry_time_hours := jerry_time_minutes / 60
  let jerry_distance := jerry_speed * jerry_time_hours
  let beth_distance := jerry_distance + beth_extra_miles
  let beth_time_hours := (jerry_time_minutes + beth_extra_time_minutes) / 60
  beth_distance / beth_time_hours

theorem beth_speed {beth_avg_speed : ℚ}
  (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ)
  (h_jerry_speed : jerry_speed = 40)
  (h_jerry_time : jerry_time_minutes = 30)
  (h_beth_extra_miles : beth_extra_miles = 5)
  (h_beth_extra_time : beth_extra_time_minutes = 20) :
  beth_average_speed jerry_speed jerry_time_minutes beth_extra_miles beth_extra_time_minutes = 30 := 
by 
  -- Leaving out the proof steps
  sorry

end beth_speed_l211_211012


namespace constant_value_l211_211396

theorem constant_value (x y z C : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : x > y) (h4 : y > z) (h5 : z = 2) (h6 : 2 * x + 3 * y + 3 * z = 5 * y + C) : C = 8 :=
by
  sorry

end constant_value_l211_211396


namespace negation_of_proposition_l211_211716

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0)) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_of_proposition_l211_211716


namespace scientific_notation_384000_l211_211256

theorem scientific_notation_384000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 384000 = a * 10 ^ n ∧ 
  a = 3.84 ∧ n = 5 :=
sorry

end scientific_notation_384000_l211_211256


namespace one_cow_one_bag_l211_211858

theorem one_cow_one_bag {days_per_bag : ℕ} (h : 50 * days_per_bag = 50 * 50) : days_per_bag = 50 :=
by
  sorry

end one_cow_one_bag_l211_211858


namespace find_angle_and_area_l211_211791

theorem find_angle_and_area (a b c : ℝ) (C : ℝ)
  (h₁: (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 2 * a * b)
  (h₂: c = 2)
  (h₃: b = 2 * Real.sqrt 2) : 
  C = Real.pi / 4 ∧ a = 2 ∧ (∃ S : ℝ, S = 1 / 2 * a * c ∧ S = 2) :=
by
  -- We assume sorry here since the focus is on setting up the problem statement correctly
  sorry

end find_angle_and_area_l211_211791


namespace ordered_triple_unique_l211_211661

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end ordered_triple_unique_l211_211661


namespace scientific_notation_correct_l211_211763

def million : ℝ := 10^6
def num : ℝ := 1.06
def num_in_million : ℝ := num * million
def scientific_notation : ℝ := 1.06 * 10^6

theorem scientific_notation_correct : num_in_million = scientific_notation :=
by 
  -- The proof is skipped, indicated by sorry
  sorry

end scientific_notation_correct_l211_211763


namespace min_ab_min_expr_min_a_b_l211_211933

-- Define the conditions
variables (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hln : Real.log a + Real.log b = Real.log (a + 9 * b))

-- 1. The minimum value of ab
theorem min_ab : ab = 36 :=
sorry

-- 2. The minimum value of (81 / a^2) + (1 / b^2)
theorem min_expr : (81 / a^2) + (1 / b^2) = (1 / 2) :=
sorry

-- 3. The minimum value of a + b
theorem min_a_b : a + b = 16 :=
sorry

end min_ab_min_expr_min_a_b_l211_211933


namespace relationship_between_y1_y2_l211_211516

theorem relationship_between_y1_y2 (y1 y2 : ℝ) :
    (y1 = -3 * 2 + 4 ∧ y2 = -3 * (-1) + 4) → y1 < y2 :=
by
  sorry

end relationship_between_y1_y2_l211_211516


namespace values_of_a_l211_211685

axiom exists_rat : (x y a : ℚ) → Prop

theorem values_of_a (a : ℚ) (h1 : ∀ x y : ℚ, (x/2 - (2*x - 3*y)/5 = a - 1)) (h2 : ∀ x y : ℚ, (x + 3 = y/3)) :
  0.7 < a ∧ a < 6.4 ↔ (∃ x y : ℚ, x < 0 ∧ y > 0) :=
by
  sorry

end values_of_a_l211_211685


namespace pencils_added_by_Nancy_l211_211703

def original_pencils : ℕ := 27
def total_pencils : ℕ := 72

theorem pencils_added_by_Nancy : ∃ x : ℕ, x = total_pencils - original_pencils := by
  sorry

end pencils_added_by_Nancy_l211_211703


namespace minimum_area_isosceles_trapezoid_l211_211037

theorem minimum_area_isosceles_trapezoid (r x a d : ℝ) (h_circumscribed : a + d = 2 * x) (h_minimal : x ≥ 2 * r) :
  4 * r^2 ≤ (a + d) * r :=
by sorry

end minimum_area_isosceles_trapezoid_l211_211037


namespace sin_70_eq_1_minus_2k_squared_l211_211283

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end sin_70_eq_1_minus_2k_squared_l211_211283


namespace Sam_balloons_correct_l211_211385

def Fred_balloons : Nat := 10
def Dan_balloons : Nat := 16
def Total_balloons : Nat := 72

def Sam_balloons : Nat := Total_balloons - Fred_balloons - Dan_balloons

theorem Sam_balloons_correct : Sam_balloons = 46 := by 
  have H : Sam_balloons = 72 - 10 - 16 := rfl
  simp at H
  exact H

end Sam_balloons_correct_l211_211385


namespace completing_the_square_solution_l211_211042

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_the_square_solution_l211_211042


namespace lower_bound_expression_l211_211477

theorem lower_bound_expression (n : ℤ) (L : ℤ) :
  (∃ k : ℕ, k = 20 ∧
          ∀ n, (L < 4 * n + 7 ∧ 4 * n + 7 < 80)) →
  L = 3 :=
by
  sorry

end lower_bound_expression_l211_211477


namespace find_x_condition_l211_211776

theorem find_x_condition :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  intros x h
  have num_zero : x^2 - 1 = 0 := by
    -- Proof that the numerator is zero
    sorry
  have denom_nonzero : x ≠ -1 := by
    -- Proof that the denominator is non-zero
    sorry
  have x_solves : x = 1 := by
    -- Final proof to show x = 1
    sorry
  exact x_solves

end find_x_condition_l211_211776


namespace cookies_left_l211_211760

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end cookies_left_l211_211760


namespace janet_practiced_days_l211_211794

theorem janet_practiced_days (total_miles : ℕ) (miles_per_day : ℕ) (days_practiced : ℕ) :
  total_miles = 72 ∧ miles_per_day = 8 → days_practiced = total_miles / miles_per_day → days_practiced = 9 :=
by
  sorry

end janet_practiced_days_l211_211794


namespace tan_difference_identity_l211_211021

theorem tan_difference_identity {α : ℝ} (h : Real.tan α = 4 * Real.sin (7 * Real.pi / 3)) :
  Real.tan (α - Real.pi / 3) = Real.sqrt 3 / 7 := 
sorry

end tan_difference_identity_l211_211021


namespace log_identity_l211_211378

theorem log_identity : (Real.log 2)^3 + 3 * (Real.log 2) * (Real.log 5) + (Real.log 5)^3 = 1 :=
by
  sorry

end log_identity_l211_211378


namespace valid_outfit_selections_l211_211157

-- Definitions based on the given conditions
def num_shirts : ℕ := 6
def num_pants : ℕ := 5
def num_hats : ℕ := 6
def num_colors : ℕ := 6

-- The total number of outfits without restrictions
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- The theorem statement to prove the final answer
theorem valid_outfit_selections : total_outfits = 150 :=
by
  have h1 : total_outfits = 6 * 5 * 6 := rfl
  have h2 : 6 * 5 * 6 = 180 := by norm_num
  have h3 : 180 = 150 := sorry -- Here you need to differentiate the invalid outfits using provided restrictions
  exact h3

end valid_outfit_selections_l211_211157


namespace exam_score_impossible_l211_211064

theorem exam_score_impossible (x y : ℕ) : 
  (5 * x + y = 97) ∧ (x + y ≤ 20) → false :=
by
  sorry

end exam_score_impossible_l211_211064


namespace original_class_strength_l211_211882

theorem original_class_strength 
  (orig_avg_age : ℕ) (new_students_num : ℕ) (new_avg_age : ℕ) 
  (avg_age_decrease : ℕ) (orig_strength : ℕ) :
  orig_avg_age = 40 →
  new_students_num = 12 →
  new_avg_age = 32 →
  avg_age_decrease = 4 →
  (orig_strength + new_students_num) * (orig_avg_age - avg_age_decrease) = orig_strength * orig_avg_age + new_students_num * new_avg_age →
  orig_strength = 12 := 
by
  intros
  sorry

end original_class_strength_l211_211882


namespace least_sum_p_q_r_l211_211060

theorem least_sum_p_q_r (p q r : ℕ) (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (h : 17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1)) : p + q + r = 290 :=
  sorry

end least_sum_p_q_r_l211_211060


namespace obtuse_angle_in_second_quadrant_l211_211358

-- Let θ be an angle in degrees
def angle_in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def angle_terminal_side_same (θ₁ θ₂ : ℝ) : Prop := θ₁ % 360 = θ₂ % 360

def angle_in_fourth_quadrant (θ : ℝ) : Prop := -360 < θ ∧ θ < 0 ∧ (θ + 360) > 270

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement D: An obtuse angle is definitely in the second quadrant
theorem obtuse_angle_in_second_quadrant (θ : ℝ) (h : is_obtuse_angle θ) :
  90 < θ ∧ θ < 180 := by
    sorry

end obtuse_angle_in_second_quadrant_l211_211358


namespace usual_time_is_42_l211_211500

noncomputable def usual_time_to_school (R T : ℝ) := T * R
noncomputable def improved_time_to_school (R T : ℝ) := ((7/6) * R) * (T - 6)

theorem usual_time_is_42 (R T : ℝ) :
  (usual_time_to_school R T) = (improved_time_to_school R T) → T = 42 :=
by
  sorry

end usual_time_is_42_l211_211500


namespace find_x_l211_211424

theorem find_x : 
  (∃ x : ℝ, 
    2.5 * ((3.6 * 0.48 * 2.5) / (0.12 * x * 0.5)) = 2000.0000000000002) → 
  x = 0.225 :=
by
  sorry

end find_x_l211_211424


namespace convex_polyhedron_has_triangular_face_l211_211822

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end convex_polyhedron_has_triangular_face_l211_211822


namespace sum_of_possible_remainders_l211_211770

theorem sum_of_possible_remainders (n : ℕ) (h_even : ∃ k : ℕ, n = 2 * k) : 
  let m := 1000 * (2 * n + 6) + 100 * (2 * n + 4) + 10 * (2 * n + 2) + (2 * n)
  let remainder (k : ℕ) := (1112 * k + 6420) % 29
  23 + 7 + 20 = 50 :=
  by
  sorry

end sum_of_possible_remainders_l211_211770


namespace cost_of_iced_coffee_for_2_weeks_l211_211290

def cost_to_last_for_2_weeks (servings_per_bottle servings_per_day price_per_bottle duration_in_days : ℕ) : ℕ :=
  let total_servings_needed := servings_per_day * duration_in_days
  let bottles_needed := total_servings_needed / servings_per_bottle
  bottles_needed * price_per_bottle

theorem cost_of_iced_coffee_for_2_weeks :
  cost_to_last_for_2_weeks 6 3 3 14 = 21 :=
by
  sorry

end cost_of_iced_coffee_for_2_weeks_l211_211290


namespace solve_for_x_l211_211796

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - y = 7)
  (h2 : x + 3 * y = 7) :
  x = 2.8 :=
by
  sorry

end solve_for_x_l211_211796


namespace mila_total_distance_l211_211447

/-- Mila's car consumes a gallon of gas every 40 miles, her full gas tank holds 16 gallons, starting with a full tank, she drove 400 miles, then refueled with 10 gallons, 
and upon arriving at her destination her gas tank was a third full.
Prove that the total distance Mila drove that day is 826 miles. -/
theorem mila_total_distance (consumption_per_mile : ℝ) (tank_capacity : ℝ) (initial_drive : ℝ) (refuel_amount : ℝ) (final_fraction : ℝ)
  (consumption_per_mile_def : consumption_per_mile = 1 / 40)
  (tank_capacity_def : tank_capacity = 16)
  (initial_drive_def : initial_drive = 400)
  (refuel_amount_def : refuel_amount = 10)
  (final_fraction_def : final_fraction = 1 / 3) :
  ∃ total_distance : ℝ, total_distance = 826 :=
by
  sorry

end mila_total_distance_l211_211447


namespace sequence_general_formula_l211_211641

theorem sequence_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2 - 2 * n + 2):
  (a 1 = 1) ∧ (∀ n, 1 < n → a n = S n - S (n - 1)) → 
  (∀ n, a n = if n = 1 then 1 else 2 * n - 3) :=
by
  intro h
  sorry

end sequence_general_formula_l211_211641


namespace average_marks_of_all_students_l211_211062

theorem average_marks_of_all_students (n1 n2 a1 a2 : ℕ) (n1_eq : n1 = 12) (a1_eq : a1 = 40) 
  (n2_eq : n2 = 28) (a2_eq : a2 = 60) : 
  ((n1 * a1 + n2 * a2) / (n1 + n2) : ℕ) = 54 := 
by
  sorry

end average_marks_of_all_students_l211_211062


namespace no_linear_term_in_product_l211_211741

theorem no_linear_term_in_product (m : ℝ) :
  (∀ (x : ℝ), (x - 3) * (3 * x + m) - (3 * x^2 - 3 * m) = 0) → m = 9 :=
by
  intro h
  sorry

end no_linear_term_in_product_l211_211741


namespace oranges_apples_bananas_equiv_l211_211706

-- Define weights
variable (w_orange w_apple w_banana : ℝ)

-- Conditions
def condition1 : Prop := 9 * w_orange = 6 * w_apple
def condition2 : Prop := 4 * w_banana = 3 * w_apple

-- Main problem
theorem oranges_apples_bananas_equiv :
  ∀ (w_orange w_apple w_banana : ℝ),
  (9 * w_orange = 6 * w_apple) →
  (4 * w_banana = 3 * w_apple) →
  ∃ (a b : ℕ), a = 17 ∧ b = 13 ∧ (a + 3/4 * b = (45/9) * 6) :=
by
  intros w_orange w_apple w_banana h1 h2
  -- note: actual proof would go here
  sorry

end oranges_apples_bananas_equiv_l211_211706


namespace expression_meaningful_if_not_three_l211_211845

-- Definition of meaningful expression
def meaningful_expr (x : ℝ) : Prop := (x ≠ 3)

theorem expression_meaningful_if_not_three (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ meaningful_expr x := by
  sorry

end expression_meaningful_if_not_three_l211_211845


namespace find_n_l211_211738

-- Define the values of quarters and dimes in cents
def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10

-- Define the number of quarters and dimes
def num_quarters : ℕ := 15
def num_dimes : ℕ := 25

-- Define the total value in cents corresponding to the quarters
def total_value_quarters : ℕ := num_quarters * value_of_quarter

-- Define the condition where total value by quarters equals total value by n dimes
def equivalent_dimes (n : ℕ) : Prop := total_value_quarters = n * value_of_dime

-- The theorem to prove
theorem find_n : ∃ n : ℕ, equivalent_dimes n ∧ n = 38 := 
by {
  use 38,
  sorry
}

end find_n_l211_211738


namespace intersection_of_M_and_N_l211_211076

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- The theorem to be proved
theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_of_M_and_N_l211_211076


namespace inequality_proof_l211_211156

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) >= 1) :=
by sorry

end inequality_proof_l211_211156


namespace largest_integer_l211_211190

theorem largest_integer (n : ℕ) : n ^ 200 < 5 ^ 300 → n <= 11 :=
by
  sorry

end largest_integer_l211_211190


namespace lowest_dropped_score_l211_211088

theorem lowest_dropped_score (A B C D : ℕ)
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : (A + B + C) / 3 = 55) :
  D = 35 :=
by
  sorry

end lowest_dropped_score_l211_211088


namespace find_f_neg_one_l211_211659

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd f)
(h_pos : ∀ x, 0 < x → f x = x^2 + 1/x) : f (-1) = -2 := 
sorry

end find_f_neg_one_l211_211659


namespace intersection_result_complement_union_result_l211_211241

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_result : A ∩ B = {x | 0 < x ∧ x < 2} :=
by
  sorry

theorem complement_union_result : (compl B) ∪ A = {x | x < 2} :=
by
  sorry

end intersection_result_complement_union_result_l211_211241


namespace find_a_max_and_min_values_l211_211855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + 3*x + a + 1)

theorem find_a (a : ℝ) : (f' a 0) = 2 → a = 1 :=
by {
  -- Proof omitted
  sorry
}

theorem max_and_min_values (a : ℝ) :
  (a = 1) →
  (Real.exp (-2) * (4 - 2 + 1) = (3 / Real.exp 2)) ∧
  (Real.exp (-1) * (1 - 1 + 1) = (1 / Real.exp 1)) ∧
  (Real.exp 2 * (4 + 2 + 1) = (7 * Real.exp 2)) :=
by {
  -- Proof omitted
  sorry
}

end find_a_max_and_min_values_l211_211855


namespace deductive_reasoning_l211_211689

theorem deductive_reasoning (
  deductive_reasoning_form : Prop
): ¬(deductive_reasoning_form → true → correct_conclusion) :=
by sorry

end deductive_reasoning_l211_211689


namespace expected_number_of_2s_when_three_dice_rolled_l211_211443

def probability_of_rolling_2 : ℚ := 1 / 6
def probability_of_not_rolling_2 : ℚ := 5 / 6

theorem expected_number_of_2s_when_three_dice_rolled :
  (0 * (probability_of_not_rolling_2)^3 + 
   1 * 3 * (probability_of_rolling_2) * (probability_of_not_rolling_2)^2 + 
   2 * 3 * (probability_of_rolling_2)^2 * (probability_of_not_rolling_2) + 
   3 * (probability_of_rolling_2)^3) = 
   1 / 2 :=
by
  sorry

end expected_number_of_2s_when_three_dice_rolled_l211_211443


namespace geometric_arithmetic_series_difference_l211_211622

theorem geometric_arithmetic_series_difference :
  let a := 1
  let r := 1 / 2
  let S := a / (1 - r)
  let T := 1 + 2 + 3
  S - T = -4 :=
by
  sorry

end geometric_arithmetic_series_difference_l211_211622


namespace solve_equation_l211_211584

theorem solve_equation : ∀ (x : ℝ), x ≠ 2 → -2 * x^2 = (4 * x + 2) / (x - 2) → x = 1 :=
by
  intros x hx h_eq
  sorry

end solve_equation_l211_211584


namespace uki_total_earnings_l211_211391

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end uki_total_earnings_l211_211391


namespace billiard_angle_correct_l211_211640

-- Definitions for the problem conditions
def center_O : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (0.5, 0)
def radius : ℝ := 1

-- The angle to be proven
def strike_angle (α x : ℝ) := x = (90 - 2 * α)

-- Main theorem statement
theorem billiard_angle_correct :
  ∃ α x : ℝ, (strike_angle α x) ∧ x = 47 + (4 / 60) :=
sorry

end billiard_angle_correct_l211_211640


namespace Mary_works_hours_on_Tuesday_and_Thursday_l211_211095

theorem Mary_works_hours_on_Tuesday_and_Thursday 
  (h_mon_wed_fri : ∀ (d : ℕ), d = 3 → 9 * d = 27)
  (weekly_earnings : ℕ)
  (hourly_rate : ℕ)
  (weekly_hours_mon_wed_fri : ℕ)
  (tue_thu_hours : ℕ) :
  weekly_earnings = 407 →
  hourly_rate = 11 →
  weekly_hours_mon_wed_fri = 9 * 3 →
  weekly_earnings - weekly_hours_mon_wed_fri * hourly_rate = tue_thu_hours * hourly_rate →
  tue_thu_hours = 10 :=
by
  intros hearnings hrate hweek hsub
  sorry

end Mary_works_hours_on_Tuesday_and_Thursday_l211_211095


namespace sum_of_xs_l211_211206

theorem sum_of_xs (x y z : ℂ) : (x + y * z = 8) ∧ (y + x * z = 12) ∧ (z + x * y = 11) → 
    ∃ S, ∀ (xi yi zi : ℂ), (xi + yi * zi = 8) ∧ (yi + xi * zi = 12) ∧ (zi + xi * yi = 11) →
        xi + yi + zi = S :=
by
  sorry

end sum_of_xs_l211_211206


namespace probability_playing_one_instrument_l211_211133

noncomputable def total_people : ℕ := 800
noncomputable def fraction_playing_instruments : ℚ := 1 / 5
noncomputable def number_playing_two_or_more : ℕ := 32

theorem probability_playing_one_instrument :
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  (number_playing_exactly_one / total_people) = 1 / 6.25 :=
by 
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  have key : (number_playing_exactly_one / total_people) = 1 / 6.25 := sorry
  exact key

end probability_playing_one_instrument_l211_211133


namespace pieces_per_block_is_32_l211_211816

-- Define the number of pieces of junk mail given to each house
def pieces_per_house : ℕ := 8

-- Define the number of houses in each block
def houses_per_block : ℕ := 4

-- Calculate the total number of pieces of junk mail given to each block
def total_pieces_per_block : ℕ := pieces_per_house * houses_per_block

-- Prove that the total number of pieces of junk mail given to each block is 32
theorem pieces_per_block_is_32 : total_pieces_per_block = 32 := 
by sorry

end pieces_per_block_is_32_l211_211816


namespace exist_ai_for_xij_l211_211586

theorem exist_ai_for_xij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ a : Fin n → ℝ, ∀ i j : Fin n, x i j = a i - a j :=
by
  sorry

end exist_ai_for_xij_l211_211586


namespace percentage_value_l211_211538

theorem percentage_value (M : ℝ) (h : (25 / 100) * M = (55 / 100) * 1500) : M = 3300 :=
by
  sorry

end percentage_value_l211_211538


namespace inequality_solution_l211_211868

theorem inequality_solution (x : ℝ) : 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2 / 3 := 
sorry

end inequality_solution_l211_211868


namespace sqrt_neg9_sq_l211_211839

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end sqrt_neg9_sq_l211_211839


namespace exists_a_solution_iff_l211_211034

theorem exists_a_solution_iff (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1 / 4 := 
by 
  sorry

end exists_a_solution_iff_l211_211034


namespace sequence_sum_equality_l211_211233

theorem sequence_sum_equality {a_n : ℕ → ℕ} (S_n : ℕ → ℕ) (n : ℕ) (h : n > 0) 
  (h1 : ∀ n, 3 * a_n n = 2 * S_n n + n) : 
  S_n n = (3^((n:ℕ)+1) - 2 * n) / 4 := 
sorry

end sequence_sum_equality_l211_211233


namespace find_angle_l211_211154

theorem find_angle :
  ∃ (x : ℝ), (90 - x = 0.4 * (180 - x)) → x = 30 :=
by
  sorry

end find_angle_l211_211154


namespace min_training_iterations_l211_211953

/-- The model of exponentially decaying learning rate is given by L = L0 * D^(G / G0)
    where
    L  : the learning rate used in each round of optimization,
    L0 : the initial learning rate,
    D  : the decay coefficient,
    G  : the number of training iterations,
    G0 : the decay rate.

    Given:
    - the initial learning rate L0 = 0.5,
    - the decay rate G0 = 18,
    - when G = 18, L = 0.4,

    Prove: 
    The minimum number of training iterations required for the learning rate to decay to below 0.1 (excluding 0.1) is 130.
-/
theorem min_training_iterations
  (L0 : ℝ) (G0 : ℝ) (D : ℝ) (G : ℝ) (L : ℝ)
  (h1 : L0 = 0.5)
  (h2 : G0 = 18)
  (h3 : L = 0.4)
  (h4 : G = 18)
  (h5 : L0 * D^(G / G0) = 0.4)
  : ∃ G, G ≥ 130 ∧ L0 * D^(G / G0) < 0.1 := sorry

end min_training_iterations_l211_211953


namespace find_alpha_beta_l211_211803

-- Define the conditions of the problem
variables (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π < β ∧ β < 2 * π)
variable (h_eq : ∀ x : ℝ, cos (x + α) + sin (x + β) + sqrt 2 * cos x = 0)

-- State the required proof as a theorem
theorem find_alpha_beta : α = 3 * π / 4 ∧ β = 7 * π / 4 :=
by
  sorry

end find_alpha_beta_l211_211803


namespace perfect_square_trinomial_l211_211737

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, a^2 = 1 ∧ b^2 = 1 ∧ x^2 + m * x * y + y^2 = (a * x + b * y)^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l211_211737


namespace sum_quotient_remainder_div9_l211_211715

theorem sum_quotient_remainder_div9 (n : ℕ) (h₁ : n = 248 * 5 + 4) :
  let q := n / 9
  let r := n % 9
  q + r = 140 :=
by
  sorry

end sum_quotient_remainder_div9_l211_211715


namespace probability_of_boys_and_girls_l211_211229

def total_outcomes := Nat.choose 7 4
def only_boys_outcomes := Nat.choose 4 4
def both_boys_and_girls_outcomes := total_outcomes - only_boys_outcomes
def probability := both_boys_and_girls_outcomes / total_outcomes

theorem probability_of_boys_and_girls :
  probability = 34 / 35 :=
by
  sorry

end probability_of_boys_and_girls_l211_211229


namespace volume_correct_l211_211945

open Set Real

-- Define the conditions: the inequality and the constraints on x, y, z
def region (x y z : ℝ) : Prop :=
  abs (z + x + y) + abs (z + x - y) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

-- Define the volume calculation
def volume_of_region : ℝ :=
  62.5

-- State the theorem
theorem volume_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 62.5 :=
by
  intro x y z h
  sorry

end volume_correct_l211_211945


namespace train_rate_first_hour_l211_211438

-- Define the conditions
def rateAtFirstHour (r : ℕ) : Prop :=
  (11 / 2) * (r + (r + 100)) = 660

-- Prove the rate is 10 mph
theorem train_rate_first_hour (r : ℕ) : rateAtFirstHour r → r = 10 :=
by 
  sorry

end train_rate_first_hour_l211_211438


namespace largest_D_l211_211089

theorem largest_D (D : ℝ) : (∀ x y : ℝ, x^2 + 2 * y^2 + 3 ≥ D * (3 * x + 4 * y)) → D ≤ Real.sqrt (12 / 17) :=
by
  sorry

end largest_D_l211_211089


namespace exponent_inequality_l211_211240

theorem exponent_inequality (a b c : ℝ) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : c ≠ 1) (h4 : a > b) (h5 : b > c) (h6 : c > 0) : a ^ b > c ^ b :=
  sorry

end exponent_inequality_l211_211240


namespace david_age_uniq_l211_211705

theorem david_age_uniq (C D E : ℚ) (h1 : C = 4 * D) (h2 : E = D + 7) (h3 : C = E + 1) : D = 8 / 3 := 
by 
  sorry

end david_age_uniq_l211_211705


namespace baseball_weight_l211_211297

theorem baseball_weight
  (weight_total : ℝ)
  (weight_soccer_ball : ℝ)
  (n_soccer_balls : ℕ)
  (n_baseballs : ℕ)
  (total_weight : ℝ)
  (B : ℝ) :
  n_soccer_balls * weight_soccer_ball + n_baseballs * B = total_weight →
  n_soccer_balls = 9 →
  weight_soccer_ball = 0.8 →
  n_baseballs = 7 →
  total_weight = 10.98 →
  B = 0.54 := sorry

end baseball_weight_l211_211297


namespace bank_balance_after_five_years_l211_211788

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem bank_balance_after_five_years :
  let P0 := 5600
  let r1 := 0.03
  let r2 := 0.035
  let r3 := 0.04
  let r4 := 0.045
  let r5 := 0.05
  let D := 2000
  let A1 := compoundInterest P0 r1 1 1
  let A2 := compoundInterest A1 r2 1 1
  let A3 := compoundInterest (A2 + D) r3 1 1
  let A4 := compoundInterest A3 r4 1 1
  let A5 := compoundInterest A4 r5 1 1
  A5 = 9094.2 := by
  sorry

end bank_balance_after_five_years_l211_211788


namespace solve_inequality_l211_211900

theorem solve_inequality {a b : ℝ} (h : -2 * a + 1 < -2 * b + 1) : a > b :=
by
  sorry

end solve_inequality_l211_211900


namespace simplify_and_evaluate_l211_211176

variable (x y : ℤ)

theorem simplify_and_evaluate (h1 : x = 1) (h2 : y = 1) :
    2 * (x - 2 * y) ^ 2 - (2 * y + x) * (-2 * y + x) = 5 := by
    sorry

end simplify_and_evaluate_l211_211176


namespace solve_4_times_3_l211_211015

noncomputable def custom_operation (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem solve_4_times_3 : custom_operation 4 3 = 13 := by
  -- Here the proof would be provided, for now we use sorry
  sorry

end solve_4_times_3_l211_211015


namespace donna_pizza_slices_left_l211_211671

def total_slices_initial : ℕ := 12
def slices_eaten_lunch (slices : ℕ) : ℕ := slices / 2
def slices_remaining_after_lunch (slices : ℕ) : ℕ := slices - slices_eaten_lunch slices
def slices_eaten_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices / 3
def slices_remaining_after_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices - slices_eaten_dinner slices
def slices_shared_friend (slices : ℕ) : ℕ := slices_remaining_after_dinner slices / 4
def slices_remaining_final (slices : ℕ) : ℕ := slices_remaining_after_dinner slices - slices_shared_friend slices

theorem donna_pizza_slices_left : slices_remaining_final total_slices_initial = 3 :=
sorry

end donna_pizza_slices_left_l211_211671


namespace number_of_subsets_l211_211455

theorem number_of_subsets (M : Finset ℕ) (h : M.card = 5) : 2 ^ M.card = 32 := by
  sorry

end number_of_subsets_l211_211455


namespace solution_set_of_inequality_l211_211119

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x^2 + 2 < x} = {x : ℝ | x < -2 / 3} ∪ {x : ℝ | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l211_211119


namespace boys_without_calculators_l211_211384

theorem boys_without_calculators 
  (total_students : ℕ)
  (total_boys : ℕ)
  (students_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (H_total_students : total_students = 30)
  (H_total_boys : total_boys = 20)
  (H_students_with_calculators : students_with_calculators = 25)
  (H_girls_with_calculators : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 13 :=
by
  sorry

end boys_without_calculators_l211_211384


namespace DanielCandies_l211_211567

noncomputable def initialCandies (x : ℝ) : Prop :=
  (3 / 8) * x - (3 / 2) - 16 = 10

theorem DanielCandies : ∃ x : ℝ, initialCandies x ∧ x = 93 :=
by
  use 93
  simp [initialCandies]
  norm_num
  sorry

end DanielCandies_l211_211567


namespace area_of_circle_l211_211563

noncomputable def point : Type := ℝ × ℝ

def A : point := (8, 15)
def B : point := (14, 9)

def is_on_circle (P : point) (r : ℝ) (C : point) : Prop :=
  (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = r ^ 2

def tangent_intersects_x_axis (tangent_point : point) (circle_center : point) : Prop :=
  ∃ x : ℝ, ∃ C : point, C.2 = 0 ∧ tangent_point = C ∧ circle_center = (x, 0)

theorem area_of_circle :
  ∃ C : point, ∃ r : ℝ,
    is_on_circle A r C ∧ 
    is_on_circle B r C ∧ 
    tangent_intersects_x_axis A C ∧ 
    tangent_intersects_x_axis B C ∧ 
    (↑(π * r ^ 2) = (117 * π) / 8) :=
sorry

end area_of_circle_l211_211563


namespace shelves_for_coloring_books_l211_211687

theorem shelves_for_coloring_books (initial_stock sold donated per_shelf remaining total_used needed_shelves : ℕ) 
    (h_initial : initial_stock = 150)
    (h_sold : sold = 55)
    (h_donated : donated = 30)
    (h_per_shelf : per_shelf = 12)
    (h_total_used : total_used = sold + donated)
    (h_remaining : remaining = initial_stock - total_used)
    (h_needed_shelves : (remaining + per_shelf - 1) / per_shelf = needed_shelves) :
    needed_shelves = 6 :=
by
  sorry

end shelves_for_coloring_books_l211_211687


namespace john_needs_20_nails_l211_211197

-- Define the given conditions
def large_planks (n : ℕ) := n = 12
def small_planks (n : ℕ) := n = 10
def nails_for_large_planks (n : ℕ) := n = 15
def nails_for_small_planks (n : ℕ) := n = 5

-- Define the total number of nails needed
def total_nails_needed (n : ℕ) :=
  ∃ (lp sp np_large np_small : ℕ),
  large_planks lp ∧ small_planks sp ∧ nails_for_large_planks np_large ∧ nails_for_small_planks np_small ∧ n = np_large + np_small

-- The theorem statement
theorem john_needs_20_nails : total_nails_needed 20 :=
by { sorry }

end john_needs_20_nails_l211_211197


namespace average_speed_uphill_l211_211844

theorem average_speed_uphill (d : ℝ) (v : ℝ) :
  (2 * d) / ((d / v) + (d / 100)) = 9.523809523809524 → v = 5 :=
by
  intro h1
  sorry

end average_speed_uphill_l211_211844


namespace distinct_primes_eq_1980_l211_211989

theorem distinct_primes_eq_1980 (p q r A : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
    (hne1 : p ≠ q) (hne2 : q ≠ r) (hne3 : p ≠ r) 
    (h1 : 2 * p * q * r + 50 * p * q = A)
    (h2 : 7 * p * q * r + 55 * p * r = A)
    (h3 : 8 * p * q * r + 12 * q * r = A) : 
    A = 1980 := by {
  sorry
}

end distinct_primes_eq_1980_l211_211989


namespace factor_exp_l211_211856

variable (x : ℤ)

theorem factor_exp : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) :=
by
  sorry

end factor_exp_l211_211856


namespace remainder_is_four_l211_211779

def least_number : Nat := 174

theorem remainder_is_four (n : Nat) (m₁ m₂ : Nat) (h₁ : n = least_number / m₁ * m₁ + 4) 
(h₂ : n = least_number / m₂ * m₂ + 4) (h₃ : m₁ = 34) (h₄ : m₂ = 5) : 
  n % m₁ = 4 ∧ n % m₂ = 4 := 
by
  sorry

end remainder_is_four_l211_211779


namespace sugar_per_batch_l211_211099

variable (S : ℝ)

theorem sugar_per_batch :
  (8 * (4 + S) = 44) → (S = 1.5) :=
by
  intro h
  sorry

end sugar_per_batch_l211_211099


namespace maximize_parabola_area_l211_211081

variable {a b : ℝ}

/--
The parabola y = ax^2 + bx is tangent to the line x + y = 4 within the first quadrant. 
Prove that the values of a and b that maximize the area S enclosed by this parabola and 
the x-axis are a = -1 and b = 3, and that the maximum value of S is 9/2.
-/
theorem maximize_parabola_area (hab_tangent : ∃ x y, y = a * x^2 + b * x ∧ y = 4 - x ∧ x > 0 ∧ y > 0) 
  (area_eqn : S = 1/6 * (b^3 / a^2)) : 
  a = -1 ∧ b = 3 ∧ S = 9/2 := 
sorry

end maximize_parabola_area_l211_211081


namespace problem_a_b_c_d_l211_211295

open Real

/-- The main theorem to be proved -/
theorem problem_a_b_c_d
  (a b c d : ℝ)
  (hab : 0 < a) (hcd : 0 < c) (hab' : 0 < b) (hcd' : 0 < d)
  (h1 : a > c) (h2 : b < d)
  (h3 : a + sqrt b ≥ c + sqrt d)
  (h4 : sqrt a + b ≤ sqrt c + d) :
  a + b + c + d > 1 :=
by
  sorry

end problem_a_b_c_d_l211_211295


namespace largest_AB_under_conditions_l211_211972

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_AB_under_conditions :
  ∃ A B C D : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (A + B) % (C + D) = 0 ∧
    is_prime (A + B) ∧ is_prime (C + D) ∧
    (A + B) = 11 :=
sorry

end largest_AB_under_conditions_l211_211972


namespace sufficient_condition_l211_211753

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem sufficient_condition (a : ℝ) : (∀ x y : ℝ, N x y a → M x y) ↔ (a ≥ 5 / 4) := 
sorry

end sufficient_condition_l211_211753


namespace general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l211_211547

-- Part 1: Finding the general term of the arithmetic sequence
theorem general_term_arithmetic_seq (a : ℕ → ℤ) (h1 : a 1 = 25) (h4 : a 4 = 16) :
  ∃ d : ℤ, a n = 28 - 3 * n := 
sorry

-- Part 2: Finding the value of n that maximizes the sum of the first n terms
theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 1 = 25)
  (h4 : a 4 = 16) 
  (ha : ∀ n, a n = 28 - 3 * n) -- Using the result from part 1
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n : ℕ, S n < S (n + 1)) →
  9 = 9 :=
sorry

end general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l211_211547


namespace number_of_valid_consecutive_sum_sets_l211_211718

-- Definition of what it means to be a set of consecutive integers summing to 225
def sum_of_consecutive_integers (n a : ℕ) : Prop :=
  ∃ k : ℕ, (k = (n * (2 * a + n - 1)) / 2) ∧ (k = 225)

-- Prove that there are exactly 4 sets of two or more consecutive positive integers that sum to 225
theorem number_of_valid_consecutive_sum_sets : 
  ∃ (sets : Finset (ℕ × ℕ)), 
    (∀ (n a : ℕ), (n, a) ∈ sets ↔ sum_of_consecutive_integers n a) ∧ 
    (2 ≤ n) ∧ 
    sets.card = 4 := sorry

end number_of_valid_consecutive_sum_sets_l211_211718


namespace mandy_total_shirts_l211_211653

-- Condition definitions
def black_packs : ℕ := 6
def black_shirts_per_pack : ℕ := 7
def yellow_packs : ℕ := 8
def yellow_shirts_per_pack : ℕ := 4

theorem mandy_total_shirts : 
  (black_packs * black_shirts_per_pack + yellow_packs * yellow_shirts_per_pack) = 74 :=
by
  sorry

end mandy_total_shirts_l211_211653


namespace gcd_n_cube_plus_27_n_plus_3_l211_211372

theorem gcd_n_cube_plus_27_n_plus_3 (n : ℕ) (h : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
sorry

end gcd_n_cube_plus_27_n_plus_3_l211_211372


namespace well_depth_and_rope_length_l211_211268

variables (x y : ℝ)

theorem well_depth_and_rope_length :
  (y = x / 4 - 3) ∧ (y = x / 5 + 1) → y = 17 ∧ x = 80 :=
by
  sorry
 
end well_depth_and_rope_length_l211_211268


namespace regular_pyramid_sufficient_condition_l211_211174

-- Define the basic structure of a pyramid
structure Pyramid :=
  (lateral_face_is_equilateral_triangle : Prop)  
  (base_is_square : Prop)  
  (apex_angles_of_lateral_face_are_45_deg : Prop)  
  (projection_of_vertex_at_intersection_of_base_diagonals : Prop)
  (is_regular : Prop)

-- Define the hypothesis conditions
variables 
  (P : Pyramid)
  (h1 : P.lateral_face_is_equilateral_triangle)
  (h2 : P.base_is_square)
  (h3 : P.apex_angles_of_lateral_face_are_45_deg)
  (h4 : P.projection_of_vertex_at_intersection_of_base_diagonals)

-- Define the statement of the proof
theorem regular_pyramid_sufficient_condition :
  (P.lateral_face_is_equilateral_triangle → P.is_regular) ∧ 
  (¬(P.lateral_face_is_equilateral_triangle) → ¬P.is_regular) ↔
  (P.lateral_face_is_equilateral_triangle ∧ ¬P.base_is_square ∧ ¬P.apex_angles_of_lateral_face_are_45_deg ∧ ¬P.projection_of_vertex_at_intersection_of_base_diagonals) := 
by { sorry }


end regular_pyramid_sufficient_condition_l211_211174


namespace tangent_point_condition_l211_211482

open Function

def f (x : ℝ) : ℝ := x^3 - 3 * x
def tangent_line (s : ℝ) (x t : ℝ) : ℝ := (3 * s^2 - 3) * (x - 2) + s^3 - 3 * s

theorem tangent_point_condition (t : ℝ) (h_tangent : ∃s : ℝ, tangent_line s 2 t = t) 
  (h_not_on_curve : ∀ s, (2, t) ≠ (s, f s)) : t = -6 :=
by
  sorry

end tangent_point_condition_l211_211482


namespace stickers_total_correct_l211_211968

-- Define the conditions
def stickers_per_page : ℕ := 10
def pages_total : ℕ := 22

-- Define the total number of stickers
def total_stickers : ℕ := pages_total * stickers_per_page

-- The statement we want to prove
theorem stickers_total_correct : total_stickers = 220 :=
by {
  sorry
}

end stickers_total_correct_l211_211968


namespace line_through_intersection_points_of_circles_l211_211569

theorem line_through_intersection_points_of_circles :
  (∀ x y : ℝ, x^2 + y^2 = 9 ∧ (x + 4)^2 + (y + 3)^2 = 8 → 4 * x + 3 * y + 13 = 0) :=
by
  sorry

end line_through_intersection_points_of_circles_l211_211569


namespace hyperbola_standard_eq_proof_l211_211792

noncomputable def real_axis_length := 6
noncomputable def asymptote_slope := 3 / 2

def hyperbola_standard_eq (a b : ℝ) :=
  ∀ x y : ℝ, (y^2 / a^2 - x^2 / b^2 = 1)

theorem hyperbola_standard_eq_proof (a b : ℝ) 
  (h_a : 2 * a = real_axis_length)
  (h_b : a / b = asymptote_slope) :
  hyperbola_standard_eq 3 2 := 
by
  sorry

end hyperbola_standard_eq_proof_l211_211792


namespace original_square_perimeter_l211_211215

theorem original_square_perimeter (p : ℕ) (x : ℕ) 
  (h1: p = 56) 
  (h2: 28 * x = p) : 4 * (2 * (x + 4 * x)) = 40 :=
by
  sorry

end original_square_perimeter_l211_211215


namespace injective_function_identity_l211_211946

theorem injective_function_identity (f : ℕ → ℕ) (h_inj : Function.Injective f)
  (h : ∀ (m n : ℕ), 0 < m → 0 < n → f (n * f m) ≤ n * m) : ∀ x : ℕ, f x = x :=
by
  sorry

end injective_function_identity_l211_211946


namespace more_birds_than_storks_l211_211234

def initial_storks : ℕ := 5
def initial_birds : ℕ := 3
def additional_birds : ℕ := 4

def total_birds : ℕ := initial_birds + additional_birds

def stork_vs_bird_difference : ℕ := total_birds - initial_storks

theorem more_birds_than_storks : stork_vs_bird_difference = 2 := by
  sorry

end more_birds_than_storks_l211_211234


namespace quadratic_inequality_solution_l211_211993

variable (a x : ℝ)

-- Define the quadratic expression and the inequality condition
def quadratic_inequality (a x : ℝ) : Prop := 
  x^2 - (2 * a + 1) * x + a^2 + a < 0

-- Define the interval in which the inequality holds
def solution_set (a x : ℝ) : Prop :=
  a < x ∧ x < a + 1

-- The main statement to be proven
theorem quadratic_inequality_solution :
  ∀ a x, quadratic_inequality a x ↔ solution_set a x :=
sorry

end quadratic_inequality_solution_l211_211993


namespace no_solution_for_inequalities_l211_211800

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end no_solution_for_inequalities_l211_211800


namespace volume_is_120_l211_211851

namespace volume_proof

-- Definitions from the given conditions
variables (a b c : ℝ)
axiom ab_relation : a * b = 48
axiom bc_relation : b * c = 20
axiom ca_relation : c * a = 15

-- Goal to prove
theorem volume_is_120 : a * b * c = 120 := by
  sorry

end volume_proof

end volume_is_120_l211_211851


namespace equal_cubic_values_l211_211616

theorem equal_cubic_values (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 3) 
  (h3 : a * b * c + b * c * d + c * d * a + d * a * b = 1) :
  a * (1 - a)^3 = b * (1 - b)^3 ∧ 
  b * (1 - b)^3 = c * (1 - c)^3 ∧ 
  c * (1 - c)^3 = d * (1 - d)^3 :=
sorry

end equal_cubic_values_l211_211616


namespace least_positive_nine_n_square_twelve_n_cube_l211_211594

theorem least_positive_nine_n_square_twelve_n_cube :
  ∃ (n : ℕ), 0 < n ∧ (∃ (k1 k2 : ℕ), 9 * n = k1^2 ∧ 12 * n = k2^3) ∧ n = 144 :=
by
  sorry

end least_positive_nine_n_square_twelve_n_cube_l211_211594


namespace ratio_m_of_q_l211_211897

theorem ratio_m_of_q
  (m n p q : ℚ)
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := 
sorry

end ratio_m_of_q_l211_211897


namespace range_of_a_l211_211577

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0 ↔ a ∈ Set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l211_211577


namespace next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l211_211329

-- Problem: Next number after 48 in the sequence
theorem next_number_after_48 (x : ℕ) (h₁ : x % 3 = 0) (h₂ : (x + 1) = 64) : x = 63 := sorry

-- Problem: Eighth number in the sequence
theorem eighth_number_in_sequence (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 8) : n = 168 := sorry

-- Problem: 2013th number in the sequence
theorem two_thousand_thirteenth_number (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 2013) : n = 9120399 := sorry

end next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l211_211329


namespace yeri_change_l211_211869

theorem yeri_change :
  let cost_candies := 5 * 120
  let cost_chocolates := 3 * 350
  let total_cost := cost_candies + cost_chocolates
  let amount_handed_over := 2500
  amount_handed_over - total_cost = 850 :=
by
  sorry

end yeri_change_l211_211869


namespace race_distance_l211_211860

theorem race_distance (T_A T_B : ℝ) (D : ℝ) (V_A V_B : ℝ)
  (h1 : T_A = 23)
  (h2 : T_B = 30)
  (h3 : V_A = D / 23)
  (h4 : V_B = (D - 56) / 30)
  (h5 : D = (D - 56) * (23 / 30) + 56) :
  D = 56 :=
by
  sorry

end race_distance_l211_211860


namespace greatest_value_a4_b4_l211_211999

theorem greatest_value_a4_b4
    (a b : Nat → ℝ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + a 1)
    (h_geom_seq : ∀ n, b (n + 1) = b n * b 1)
    (h_a1b1 : a 1 * b 1 = 20)
    (h_a2b2 : a 2 * b 2 = 19)
    (h_a3b3 : a 3 * b 3 = 14) :
    ∃ m : ℝ, a 4 * b 4 = 8 ∧ ∀ x, a 4 * b 4 ≤ x -> x = 8 := by
  sorry

end greatest_value_a4_b4_l211_211999


namespace find_range_f_l211_211632

noncomputable def greatestIntegerLessEqual (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def f (x y : ℝ) : ℝ :=
  (x + y) / (greatestIntegerLessEqual x * greatestIntegerLessEqual y + greatestIntegerLessEqual x + greatestIntegerLessEqual y + 1)

theorem find_range_f (x y : ℝ) (h1: 0 < x) (h2: 0 < y) (h3: x * y = 1) : 
  ∃ r : ℝ, r = f x y := 
by
  sorry

end find_range_f_l211_211632


namespace courses_selection_l211_211153

-- Definition of the problem
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways person A can choose 2 courses from 4
def total_ways : ℕ := C 4 2 * C 4 2

-- Number of ways both choose exactly the same courses
def same_ways : ℕ := C 4 2

-- Prove the number of ways they can choose such that there is at least one course different
theorem courses_selection :
  total_ways - same_ways = 30 := by
  sorry

end courses_selection_l211_211153


namespace muffins_total_is_83_l211_211694

-- Define the given conditions.
def initial_muffins : Nat := 35
def additional_muffins : Nat := 48

-- Define the total number of muffins.
def total_muffins : Nat := initial_muffins + additional_muffins

-- Statement to prove.
theorem muffins_total_is_83 : total_muffins = 83 := by
  -- Proof is omitted.
  sorry

end muffins_total_is_83_l211_211694


namespace fraction_of_garden_occupied_by_flowerbeds_is_correct_l211_211778

noncomputable def garden_fraction_occupied : ℚ :=
  let garden_length := 28
  let garden_shorter_length := 18
  let triangle_leg := (garden_length - garden_shorter_length) / 2
  let triangle_area := 1 / 2 * triangle_leg^2
  let flowerbeds_area := 2 * triangle_area
  let garden_width : ℚ := 5  -- Assuming the height of the trapezoid as part of the garden rest
  let garden_area := garden_length * garden_width
  flowerbeds_area / garden_area

theorem fraction_of_garden_occupied_by_flowerbeds_is_correct :
  garden_fraction_occupied = 5 / 28 := by
  sorry

end fraction_of_garden_occupied_by_flowerbeds_is_correct_l211_211778


namespace triangle_equi_if_sides_eq_sum_of_products_l211_211030

theorem triangle_equi_if_sides_eq_sum_of_products (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ac) : a = b ∧ b = c :=
by sorry

end triangle_equi_if_sides_eq_sum_of_products_l211_211030


namespace correct_statement_C_l211_211554

theorem correct_statement_C
  (a : ℚ) : a < 0 → |a| = -a := 
by
  sorry

end correct_statement_C_l211_211554


namespace proof_op_l211_211087

def op (A B : ℕ) : ℕ := (A * B) / 2

theorem proof_op (a b c : ℕ) : op (op 4 6) 9 = 54 := by
  sorry

end proof_op_l211_211087


namespace find_initial_quarters_l211_211192

-- Define the initial number of dimes, nickels, and quarters (unknown)
def initial_dimes : ℕ := 2
def initial_nickels : ℕ := 5
def initial_quarters (Q : ℕ) := Q

-- Define the additional coins given by Linda’s mother
def additional_dimes : ℕ := 2
def additional_quarters : ℕ := 10
def additional_nickels : ℕ := 2 * initial_nickels

-- Define the total number of each type of coin after Linda receives the additional coins
def total_dimes : ℕ := initial_dimes + additional_dimes
def total_quarters (Q : ℕ) : ℕ := additional_quarters + initial_quarters Q
def total_nickels : ℕ := initial_nickels + additional_nickels

-- Define the total number of coins
def total_coins (Q : ℕ) : ℕ := total_dimes + total_quarters Q + total_nickels

theorem find_initial_quarters : ∃ Q : ℕ, total_coins Q = 35 ∧ Q = 6 := by
  -- Provide the corresponding proof here
  sorry

end find_initial_quarters_l211_211192


namespace avg_gpa_8th_graders_l211_211863

theorem avg_gpa_8th_graders :
  ∀ (GPA_6th GPA_8th : ℝ),
    GPA_6th = 93 →
    (∀ GPA_7th : ℝ, GPA_7th = GPA_6th + 2 →
    (GPA_6th + GPA_7th + GPA_8th) / 3 = 93 →
    GPA_8th = 91) :=
by
  intros GPA_6th GPA_8th h1 GPA_7th h2 h3
  sorry

end avg_gpa_8th_graders_l211_211863


namespace sum_of_coefficients_is_60_l211_211906

theorem sum_of_coefficients_is_60 :
  ∀ (a b c d e : ℤ), (∀ x : ℤ, 512 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) →
  a + b + c + d + e = 60 :=
by
  intros a b c d e h
  sorry

end sum_of_coefficients_is_60_l211_211906


namespace A_more_than_B_l211_211148

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := A = (1/3) * (B + C)
def condition2 : Prop := B = (2/7) * (A + C)
def condition3 : Prop := A + B + C = 1080

-- Conclusion
theorem A_more_than_B (A B C : ℝ) (h1 : condition1 A B C) (h2 : condition2 A B C) (h3 : condition3 A B C) :
  A - B = 30 :=
sorry

end A_more_than_B_l211_211148


namespace cone_from_sector_l211_211450

theorem cone_from_sector
  (r : ℝ) (slant_height : ℝ)
  (radius_circle : ℝ := 10)
  (angle_sector : ℝ := 252) :
  (r = 7 ∧ slant_height = 10) :=
by
  sorry

end cone_from_sector_l211_211450


namespace derivative_at_0_l211_211631

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x * Real.sin x - 7 * x

theorem derivative_at_0 : deriv f 0 = -6 := 
by
  sorry

end derivative_at_0_l211_211631


namespace tangent_line_through_point_l211_211507

-- Definitions based purely on the conditions given in the problem.
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
def point_on_line (x y : ℝ) : Prop := 3 * x - 4 * y + 25 = 0
def point_given : ℝ × ℝ := (-3, 4)

-- The theorem statement to be proven
theorem tangent_line_through_point : point_on_line point_given.1 point_given.2 := 
sorry

end tangent_line_through_point_l211_211507


namespace power_mod_l211_211522

theorem power_mod (n : ℕ) : (3 ^ 2017) % 17 = 3 := 
by
  sorry

end power_mod_l211_211522


namespace minimum_tanA_9tanB_l211_211503

variable (a b c A B : ℝ)
variable (Aacute : A > 0 ∧ A < π / 2)
variable (h1 : a^2 = b^2 + 2*b*c * Real.sin A)
variable (habc : a = b * Real.sin A)

theorem minimum_tanA_9tanB : 
  ∃ (A B : ℝ), (A > 0 ∧ A < π / 2) ∧ (a^2 = b^2 + 2*b*c * Real.sin A) ∧ (a = b * Real.sin A) ∧ 
  (min ((Real.tan A) - 9*(Real.tan B)) = -2) := 
  sorry

end minimum_tanA_9tanB_l211_211503


namespace find_hidden_data_points_l211_211996

-- Given conditions and data
def student_A_score := 81
def student_B_score := 76
def student_D_score := 80
def student_E_score := 83
def number_of_students := 5
def average_score := 80

-- The total score from the average and number of students
def total_score := average_score * number_of_students

theorem find_hidden_data_points (student_C_score mode_score : ℕ) :
  (student_A_score + student_B_score + student_C_score + student_D_score + student_E_score = total_score) ∧
  (mode_score = 80) :=
by
  sorry

end find_hidden_data_points_l211_211996


namespace tax_rate_calculation_l211_211532

theorem tax_rate_calculation (price_before_tax total_price : ℝ) 
  (h_price_before_tax : price_before_tax = 92) 
  (h_total_price : total_price = 98.90) : 
  (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := 
by 
  -- Proof will be provided here.
  sorry

end tax_rate_calculation_l211_211532


namespace removed_number_is_24_l211_211380

theorem removed_number_is_24
  (S9 : ℕ) (S8 : ℕ) (avg_9 : ℕ) (avg_8 : ℕ) (h1 : avg_9 = 72) (h2 : avg_8 = 78) (h3 : S9 = avg_9 * 9) (h4 : S8 = avg_8 * 8) :
  S9 - S8 = 24 :=
by
  sorry

end removed_number_is_24_l211_211380


namespace total_amount_of_currency_notes_l211_211769

theorem total_amount_of_currency_notes (x y : ℕ) (h1 : x + y = 85) (h2 : 50 * y = 3500) : 100 * x + 50 * y = 5000 := by
  sorry

end total_amount_of_currency_notes_l211_211769


namespace parabola_triangle_areas_l211_211342

-- Define necessary points and expressions
variables (x1 y1 x2 y2 x3 y3 : ℝ)
variables (m n : ℝ)
def parabola_eq (x y : ℝ) := y ^ 2 = 4 * x
def median_line (m n x y : ℝ) := m * x + n * y - m = 0
def areas_sum_sq (S1 S2 S3 : ℝ) := S1 ^ 2 + S2 ^ 2 + S3 ^ 2 = 3

-- Main statement
theorem parabola_triangle_areas :
  (parabola_eq x1 y1 ∧ parabola_eq x2 y2 ∧ parabola_eq x3 y3) →
  (m ≠ 0) →
  (median_line m n 1 0) →
  (x1 + x2 + x3 = 3) →
  ∃ S1 S2 S3 : ℝ, areas_sum_sq S1 S2 S3 :=
by sorry

end parabola_triangle_areas_l211_211342


namespace min_value_of_a_l211_211291

-- Defining the properties of the function f
variable {f : ℝ → ℝ}
variable (even_f : ∀ x, f x = f (-x))
variable (mono_f : ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Necessary condition involving f and a
variable {a : ℝ}
variable (a_condition : f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1)

-- Main statement proving that the minimum value of a is 1/2
theorem min_value_of_a : a = 1/2 :=
sorry

end min_value_of_a_l211_211291


namespace find_max_value_l211_211377

noncomputable def max_value (x y z : ℝ) : ℝ := (x + y) / (x * y * z)

theorem find_max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 2) :
  max_value x y z ≤ 13.5 :=
sorry

end find_max_value_l211_211377


namespace jaymee_is_22_l211_211879

-- Definitions based on the problem conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 2 + 2 * shara_age

-- The theorem we need to prove
theorem jaymee_is_22 : jaymee_age = 22 :=
by
  sorry

end jaymee_is_22_l211_211879


namespace area_correct_l211_211399

noncomputable def area_bounded_curves : ℝ := sorry

theorem area_correct :
  ∃ S, S = area_bounded_curves ∧ S = 12 * pi + 16 := sorry

end area_correct_l211_211399


namespace trip_duration_60_mph_l211_211431

noncomputable def time_at_new_speed (initial_time : ℚ) (initial_speed : ℚ) (new_speed : ℚ) : ℚ :=
  initial_time * (initial_speed / new_speed)

theorem trip_duration_60_mph :
  time_at_new_speed (9 / 2) 70 60 = 5.25 := 
by
  sorry

end trip_duration_60_mph_l211_211431


namespace result_l211_211976

def problem : Float :=
  let sum := 78.652 + 24.3981
  let diff := sum - 0.025
  Float.round (diff * 100) / 100

theorem result :
  problem = 103.03 := by
  sorry

end result_l211_211976


namespace age_ratio_l211_211444

theorem age_ratio (darcie_age : ℕ) (father_age : ℕ) (mother_ratio : ℚ) (mother_fraction : ℚ)
  (h1 : darcie_age = 4)
  (h2 : father_age = 30)
  (h3 : mother_ratio = 4/5)
  (h4 : mother_fraction = mother_ratio * father_age)
  (h5 : mother_fraction = 24) :
  (darcie_age : ℚ) / mother_fraction = 1 / 6 :=
by
  sorry

end age_ratio_l211_211444


namespace sum_ratio_arithmetic_sequence_l211_211959

theorem sum_ratio_arithmetic_sequence
  (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h2 : ∀ k : ℕ, a (k + 1) - a k = a 2 - a 1)
  (h3 : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 :=
sorry

end sum_ratio_arithmetic_sequence_l211_211959


namespace units_digit_of_product_l211_211220

theorem units_digit_of_product : 
  (4 * 6 * 9) % 10 = 6 := 
by
  sorry

end units_digit_of_product_l211_211220


namespace find_f_neg_19_div_3_l211_211717

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 1 then 
    8^x 
  else 
    sorry -- The full definition is complex and not needed for the statement

-- Define the properties of f
lemma f_periodic (x : ℝ) : f (x + 2) = f x := 
  sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := 
  sorry

theorem find_f_neg_19_div_3 : f (-19/3) = -2 :=
  sorry

end find_f_neg_19_div_3_l211_211717


namespace total_number_of_people_l211_211043

variables (A : ℕ) -- Number of adults in the group

-- Conditions
-- Each adult meal costs $8 and the total cost was $72
def cost_per_adult_meal : ℕ := 8
def total_cost : ℕ := 72
def number_of_kids : ℕ := 2

-- Proof problem: Given the conditions, prove the total number of people in the group is 11
theorem total_number_of_people (h : A * cost_per_adult_meal = total_cost) : A + number_of_kids = 11 :=
sorry

end total_number_of_people_l211_211043


namespace sum_first_10_terms_arithmetic_seq_l211_211036

theorem sum_first_10_terms_arithmetic_seq (a : ℕ → ℤ) (h : (a 4)^2 + (a 7)^2 + 2 * (a 4) * (a 7) = 9) :
  ∃ S, S = 10 * (a 4 + a 7) / 2 ∧ (S = 15 ∨ S = -15) := 
by
  sorry

end sum_first_10_terms_arithmetic_seq_l211_211036


namespace solution_set_l211_211069

theorem solution_set:
  (∃ x y : ℝ, x - y = 0 ∧ x^2 + y = 2) ↔ (∃ x y : ℝ, (x = 1 ∧ y = 1) ∨ (x = -2 ∧ y = -2)) :=
by
  sorry

end solution_set_l211_211069


namespace correct_growth_rate_equation_l211_211368

noncomputable def numberOfBikesFirstMonth : ℕ := 1000
noncomputable def additionalBikesThirdMonth : ℕ := 440
noncomputable def monthlyGrowthRate (x : ℝ) : Prop :=
  numberOfBikesFirstMonth * (1 + x)^2 = numberOfBikesFirstMonth + additionalBikesThirdMonth

theorem correct_growth_rate_equation (x : ℝ) : monthlyGrowthRate x :=
by
  sorry

end correct_growth_rate_equation_l211_211368


namespace sin_alpha_through_point_l211_211699

theorem sin_alpha_through_point (α : ℝ) (x y : ℝ) (h : x = -1 ∧ y = 2) (r : ℝ) (h_r : r = Real.sqrt (x^2 + y^2)) :
  Real.sin α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_through_point_l211_211699


namespace total_students_in_school_district_l211_211236

def CampusA_students : Nat :=
  let students_per_grade : Nat := 100
  let num_grades : Nat := 5
  let special_education : Nat := 30
  (students_per_grade * num_grades) + special_education

def CampusB_students : Nat :=
  let students_per_grade : Nat := 120
  let num_grades : Nat := 5
  students_per_grade * num_grades

def CampusC_students : Nat :=
  let students_per_grade : Nat := 150
  let num_grades : Nat := 2
  let international_program : Nat := 50
  (students_per_grade * num_grades) + international_program

def total_students : Nat :=
  CampusA_students + CampusB_students + CampusC_students

theorem total_students_in_school_district : total_students = 1480 := by
  sorry

end total_students_in_school_district_l211_211236


namespace ways_to_seat_people_l211_211483

noncomputable def number_of_ways : ℕ :=
  let choose_people := (Nat.choose 12 8)
  let divide_groups := (Nat.choose 8 4)
  let arrange_circular_table := (Nat.factorial 3)
  choose_people * divide_groups * (arrange_circular_table * arrange_circular_table)

theorem ways_to_seat_people :
  number_of_ways = 1247400 :=
by 
  -- proof goes here
  sorry

end ways_to_seat_people_l211_211483


namespace prob_at_least_one_head_is_7_over_8_l211_211619

-- Define the event and probability calculation
def probability_of_tails_all_three_tosses : ℚ :=
  (1 / 2) ^ 3

def probability_of_at_least_one_head : ℚ :=
  1 - probability_of_tails_all_three_tosses

-- Prove the probability of at least one head is 7/8
theorem prob_at_least_one_head_is_7_over_8 : probability_of_at_least_one_head = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_is_7_over_8_l211_211619


namespace derivative_of_f_eval_deriv_at_pi_over_6_l211_211383

noncomputable def f (x : Real) : Real := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem derivative_of_f : ∀ x, deriv f x = -Real.sin (4 * x) :=
by
  intro x
  sorry

theorem eval_deriv_at_pi_over_6 : deriv f (Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  rw [derivative_of_f]
  sorry

end derivative_of_f_eval_deriv_at_pi_over_6_l211_211383


namespace complex_modulus_z_l211_211049

-- Define the complex number z with given conditions
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I + Complex.I

-- State the theorem to be proven
theorem complex_modulus_z : Complex.abs z = Real.sqrt 2 := 
sorry

end complex_modulus_z_l211_211049


namespace angle_D_is_90_l211_211453

theorem angle_D_is_90 (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : B = 130) (h5 : C + D = 180) :
  D = 90 :=
by
  sorry

end angle_D_is_90_l211_211453


namespace solution_set_absolute_value_sum_eq_three_l211_211369

theorem solution_set_absolute_value_sum_eq_three (m n : ℝ) (h : ∀ x : ℝ, (|2 * x - 3| ≤ 1) ↔ (m ≤ x ∧ x ≤ n)) : m + n = 3 :=
sorry

end solution_set_absolute_value_sum_eq_three_l211_211369


namespace car_speed_second_hour_l211_211052

theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (avg_speed : ℝ)
  (hours : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_first_hour : ℝ)
  (distance_second_hour : ℝ) :
  speed_first_hour = 90 →
  avg_speed = 75 →
  hours = 2 →
  total_time = hours →
  total_distance = avg_speed * total_time →
  distance_first_hour = speed_first_hour * 1 →
  distance_second_hour = total_distance - distance_first_hour →
  distance_second_hour / 1 = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end car_speed_second_hour_l211_211052


namespace find_number_l211_211320

theorem find_number (x : ℕ) (h : x / 46 - 27 = 46) : x = 3358 :=
by
  sorry

end find_number_l211_211320


namespace frac_eq_l211_211059

def my_at (a b : ℕ) := a * b + b^2
def my_hash (a b : ℕ) := a^2 + b + a * b^2

theorem frac_eq : my_at 4 3 / my_hash 4 3 = 21 / 55 :=
by
  sorry

end frac_eq_l211_211059


namespace three_same_colored_balls_l211_211267

theorem three_same_colored_balls (balls : ℕ) (color_count : ℕ) (balls_per_color : ℕ) (h1 : balls = 60) (h2 : color_count = balls / balls_per_color) (h3 : balls_per_color = 6) :
  ∃ n, n = 21 ∧ (∀ picks : ℕ, picks ≥ n → ∃ c, ∃ k ≥ 3, k ≤ balls_per_color ∧ (c < color_count) ∧ (picks / c = k)) :=
sorry

end three_same_colored_balls_l211_211267


namespace determinant_range_l211_211164

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end determinant_range_l211_211164


namespace Angle_Not_Equivalent_l211_211702

theorem Angle_Not_Equivalent (θ : ℤ) : (θ = -750) → (680 % 360 ≠ θ % 360) :=
by
  intro h
  have h1 : 680 % 360 = 320 := by norm_num
  have h2 : -750 % 360 = -30 % 360 := by norm_num
  have h3 : -30 % 360 = 330 := by norm_num
  rw [h, h2, h3]
  sorry

end Angle_Not_Equivalent_l211_211702


namespace johns_total_profit_l211_211224

theorem johns_total_profit
  (cost_price : ℝ) (selling_price : ℝ) (bags_sold : ℕ)
  (h_cost : cost_price = 4) (h_sell : selling_price = 8) (h_bags : bags_sold = 30) :
  (selling_price - cost_price) * bags_sold = 120 := by
    sorry

end johns_total_profit_l211_211224


namespace sum_b_n_l211_211607

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), (∀ n : ℕ, a (n + 1) = q * a n)

theorem sum_b_n (h_geo : is_geometric a) (h_a1 : a 1 = 3) (h_sum_a : ∑' n, a n = 9) (h_bn : ∀ n, b n = a (2 * n)) :
  ∑' n, b n = 18 / 5 :=
sorry

end sum_b_n_l211_211607


namespace job_completion_time_l211_211406

theorem job_completion_time (h1 : ∀ {a d : ℝ}, 4 * (1/a + 1/d) = 1)
                             (h2 : ∀ d : ℝ, d = 11.999999999999998) :
                             (∀ a : ℝ, a = 6) :=
by
  sorry

end job_completion_time_l211_211406


namespace base_of_isosceles_triangle_l211_211784

theorem base_of_isosceles_triangle (b : ℝ) (h1 : 7 + 7 + b = 22) : b = 8 :=
by {
  sorry
}

end base_of_isosceles_triangle_l211_211784


namespace product_of_fractions_l211_211227

theorem product_of_fractions : (2 : ℚ) / 9 * (4 : ℚ) / 5 = 8 / 45 :=
by 
  sorry

end product_of_fractions_l211_211227


namespace extra_bananas_each_child_gets_l211_211281

theorem extra_bananas_each_child_gets
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (absent_children : ℕ)
  (present_children : ℕ)
  (total_bananas : ℕ)
  (bananas_each_present_child_gets : ℕ)
  (extra_bananas : ℕ) :
  total_children = 840 ∧
  bananas_per_child = 2 ∧
  absent_children = 420 ∧
  present_children = total_children - absent_children ∧
  total_bananas = total_children * bananas_per_child ∧
  bananas_each_present_child_gets = total_bananas / present_children ∧
  extra_bananas = bananas_each_present_child_gets - bananas_per_child →
  extra_bananas = 2 :=
by
  sorry

end extra_bananas_each_child_gets_l211_211281


namespace urn_probability_l211_211812

theorem urn_probability :
  ∀ (urn: Finset (ℕ × ℕ)), 
    urn = {(2, 1)} →
    (∀ (n : ℕ) (urn' : Finset (ℕ × ℕ)), n ≤ 5 → urn = urn' → 
      (∃ (r b : ℕ), (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)} ∨ (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)}) → 
    ∃ (p : ℚ), p = 8 / 21)
  := by
    sorry

end urn_probability_l211_211812


namespace exponent_sum_l211_211854

variables (a : ℝ) (m n : ℝ)

theorem exponent_sum (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_sum_l211_211854


namespace pizza_ratio_l211_211623

/-- Define a function that represents the ratio calculation -/
def ratio (a b : ℕ) : ℕ × ℕ := (a / (Nat.gcd a b), b / (Nat.gcd a b))

/-- State the main problem to be proved -/
theorem pizza_ratio (total_slices friend_eats james_eats remaining_slices gcd : ℕ)
  (h1 : total_slices = 8)
  (h2 : friend_eats = 2)
  (h3 : james_eats = 3)
  (h4 : remaining_slices = total_slices - friend_eats)
  (h5 : gcd = Nat.gcd james_eats remaining_slices)
  (h6 : ratio james_eats remaining_slices = (1, 2)) :
  ratio james_eats remaining_slices = (1, 2) :=
by
  sorry

end pizza_ratio_l211_211623


namespace max_product_of_roots_l211_211624

noncomputable def max_prod_roots_m : ℝ :=
  let m := 4.5
  m

theorem max_product_of_roots (m : ℕ) (h : 36 - 8 * m ≥ 0) : m = max_prod_roots_m :=
  sorry

end max_product_of_roots_l211_211624


namespace original_count_l211_211973

-- Conditions
def original_count_eq (ping_pong_balls shuttlecocks : ℕ) : Prop :=
  ping_pong_balls = shuttlecocks

def removal_count (x : ℕ) : Prop :=
  5 * x - 3 * x = 16

-- Theorem to prove the original number of ping-pong balls and shuttlecocks
theorem original_count (ping_pong_balls shuttlecocks : ℕ) (x : ℕ) (h1 : original_count_eq ping_pong_balls shuttlecocks) (h2 : removal_count x) : ping_pong_balls = 40 ∧ shuttlecocks = 40 :=
  sorry

end original_count_l211_211973


namespace seth_spent_more_l211_211257

def cost_ice_cream (cartons : ℕ) (price : ℕ) := cartons * price
def cost_yogurt (cartons : ℕ) (price : ℕ) := cartons * price
def amount_spent (cost_ice : ℕ) (cost_yog : ℕ) := cost_ice - cost_yog

theorem seth_spent_more :
  amount_spent (cost_ice_cream 20 6) (cost_yogurt 2 1) = 118 := by
  sorry

end seth_spent_more_l211_211257


namespace probability_of_die_showing_1_after_5_steps_l211_211218

def prob_showing_1 (steps : ℕ) : ℚ :=
  if steps = 5 then 37 / 192 else 0

theorem probability_of_die_showing_1_after_5_steps :
  prob_showing_1 5 = 37 / 192 :=
sorry

end probability_of_die_showing_1_after_5_steps_l211_211218


namespace range_of_sum_l211_211166

theorem range_of_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b + 1 / a + 9 / b = 10) : 2 ≤ a + b ∧ a + b ≤ 8 :=
sorry

end range_of_sum_l211_211166


namespace extreme_value_sum_l211_211143

noncomputable def f (m n x : ℝ) : ℝ := x^3 + 3 * m * x^2 + n * x + m^2

theorem extreme_value_sum (m n : ℝ) (h1 : f m n (-1) = 0) (h2 : (deriv (f m n)) (-1) = 0) : m + n = 11 := 
sorry

end extreme_value_sum_l211_211143


namespace probability_both_selected_l211_211916

theorem probability_both_selected (P_R : ℚ) (P_V : ℚ) (h1 : P_R = 3 / 7) (h2 : P_V = 1 / 5) :
  P_R * P_V = 3 / 35 :=
by {
  sorry
}

end probability_both_selected_l211_211916


namespace vans_capacity_l211_211013

-- Definitions based on the conditions
def num_students : ℕ := 22
def num_adults : ℕ := 2
def num_vans : ℕ := 3

-- The Lean statement (theorem to be proved)
theorem vans_capacity :
  (num_students + num_adults) / num_vans = 8 := 
by
  sorry

end vans_capacity_l211_211013


namespace nina_money_l211_211668

variable (C : ℝ)

def original_widget_count : ℕ := 6
def new_widget_count : ℕ := 8
def price_reduction : ℝ := 1.5

theorem nina_money (h : original_widget_count * C = new_widget_count * (C - price_reduction)) :
  original_widget_count * C = 36 := by
  sorry

end nina_money_l211_211668


namespace necessary_but_not_sufficient_for_q_implies_range_of_a_l211_211517

variable (a : ℝ)

def p (x : ℝ) := |4*x - 3| ≤ 1
def q (x : ℝ) := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem necessary_but_not_sufficient_for_q_implies_range_of_a :
  (∀ x : ℝ, q a x → p x) → (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end necessary_but_not_sufficient_for_q_implies_range_of_a_l211_211517


namespace interest_paid_percent_l211_211493

noncomputable def down_payment : ℝ := 300
noncomputable def total_cost : ℝ := 750
noncomputable def monthly_payment : ℝ := 57
noncomputable def final_payment : ℝ := 21
noncomputable def num_monthly_payments : ℕ := 9

noncomputable def total_instalments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_paid : ℝ := total_instalments + down_payment
noncomputable def amount_borrowed : ℝ := total_cost - down_payment
noncomputable def interest_paid : ℝ := total_paid - amount_borrowed
noncomputable def interest_percent : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_paid_percent:
  interest_percent = 85.33 := by
  sorry

end interest_paid_percent_l211_211493


namespace solve_quadratic_eq_l211_211282

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 4 * x - 1 = 0) : x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

end solve_quadratic_eq_l211_211282


namespace true_weight_third_object_proof_l211_211145

noncomputable def true_weight_third_object (A a B b C : ℝ) : ℝ :=
  let h := Real.sqrt ((a - b) / (A - B))
  let k := (b * A - a * B) / ((A - B) * (h + 1))
  h * C + k

theorem true_weight_third_object_proof (A a B b C : ℝ) (h := Real.sqrt ((a - b) / (A - B))) (k := (b * A - a * B) / ((A - B) * (h + 1))) :
  true_weight_third_object A a B b C = h * C + k := by
  sorry

end true_weight_third_object_proof_l211_211145


namespace leaves_blew_away_correct_l211_211512

-- Define the initial number of leaves Mikey had.
def initial_leaves : ℕ := 356

-- Define the number of leaves Mikey has left.
def leaves_left : ℕ := 112

-- Define the number of leaves that blew away.
def leaves_blew_away : ℕ := initial_leaves - leaves_left

-- Prove that the number of leaves that blew away is 244.
theorem leaves_blew_away_correct : leaves_blew_away = 244 :=
by sorry

end leaves_blew_away_correct_l211_211512


namespace record_expenditure_l211_211086

theorem record_expenditure (income recording expenditure : ℤ) (h : income = 100 ∧ recording = 100) :
  expenditure = -80 ↔ recording - expenditure = income - 80 :=
by
  sorry

end record_expenditure_l211_211086


namespace cost_of_bananas_l211_211434

theorem cost_of_bananas
  (apple_cost : ℕ)
  (orange_cost : ℕ)
  (banana_cost : ℕ)
  (num_apples : ℕ)
  (num_oranges : ℕ)
  (num_bananas : ℕ)
  (total_paid : ℕ) 
  (discount_threshold : ℕ)
  (discount_amount : ℕ)
  (total_fruits : ℕ)
  (total_without_discount : ℕ) :
  apple_cost = 1 → 
  orange_cost = 2 → 
  num_apples = 5 → 
  num_oranges = 3 → 
  num_bananas = 2 → 
  total_paid = 15 → 
  discount_threshold = 5 → 
  discount_amount = 1 → 
  total_fruits = num_apples + num_oranges + num_bananas →
  total_without_discount = (num_apples * apple_cost) + (num_oranges * orange_cost) + (num_bananas * banana_cost) →
  (total_without_discount - (discount_amount * (total_fruits / discount_threshold))) = total_paid →
  banana_cost = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end cost_of_bananas_l211_211434


namespace cubic_polynomial_solution_l211_211389

theorem cubic_polynomial_solution 
  (p : ℚ → ℚ) 
  (h1 : p 1 = 1)
  (h2 : p 2 = 1 / 4)
  (h3 : p 3 = 1 / 9)
  (h4 : p 4 = 1 / 16)
  (h6 : p 6 = 1 / 36)
  (h0 : p 0 = -1 / 25) : 
  p 5 = 20668 / 216000 :=
sorry

end cubic_polynomial_solution_l211_211389


namespace eyes_that_saw_the_plane_l211_211786

theorem eyes_that_saw_the_plane (students : ℕ) (ratio : ℚ) (eyes_per_student : ℕ) 
  (h1 : students = 200) (h2 : ratio = 3 / 4) (h3 : eyes_per_student = 2) : 
  2 * (ratio * students) = 300 := 
by 
  -- the proof is omitted
  sorry

end eyes_that_saw_the_plane_l211_211786


namespace eggs_ordered_l211_211680

theorem eggs_ordered (E : ℕ) (h1 : E > 0) (h_crepes : E * 1 / 4 = E / 4)
                     (h_cupcakes : 2 / 3 * (3 / 4 * E) = 1 / 2 * E)
                     (h_left : (3 / 4 * E - 2 / 3 * (3 / 4 * E)) = 9) :
  E = 18 := by
  sorry

end eggs_ordered_l211_211680


namespace find_98_real_coins_l211_211199

-- We will define the conditions as variables and state the goal as a theorem.

-- Variables:
variable (Coin : Type) -- Type representing coins
variable [Fintype Coin] -- 100 coins in total, therefore a Finite type
variable (number_of_coins : ℕ) (h100 : number_of_coins = 100)
variable (real : Coin → Prop) -- Predicate indicating if the coin is real
variable (lighter_fake : Coin → Prop) -- Predicate indicating if the coin is the lighter fake
variable (balance_scale : Coin → Coin → Prop) -- Balance scale result

-- Conditions:
axiom real_coins_count : ∃ R : Finset Coin, R.card = 99 ∧ (∀ c ∈ R, real c)
axiom fake_coin_exists : ∃ F : Coin, lighter_fake F ∧ ¬ real F

theorem find_98_real_coins : ∃ S : Finset Coin, S.card = 98 ∧ (∀ c ∈ S, real c) := by
  sorry

end find_98_real_coins_l211_211199


namespace domain_of_f_l211_211210

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : 
  {x : ℝ | ¬ ((x - 3) + (x - 9) = 0)} = 
  {x : ℝ | x ≠ 6} := 
by
  sorry

end domain_of_f_l211_211210


namespace sci_not_218000_l211_211599

theorem sci_not_218000 : 218000 = 2.18 * 10^5 :=
by
  sorry

end sci_not_218000_l211_211599


namespace count_perfect_cubes_l211_211255

theorem count_perfect_cubes (a b : ℕ) (h₁ : 200 < a) (h₂ : a < 1500) (h₃ : b = 6^3) :
  (∃! n : ℕ, 200 < n^3 ∧ n^3 < 1500) :=
sorry

end count_perfect_cubes_l211_211255


namespace max_value_a_l211_211777

theorem max_value_a (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + y = 1) : 
  ∃ a, a = 16 ∧ (∀ x y, (x > 0 → y > 0 → x + y = 1 → a ≤ (1/x) + (9/y))) :=
by 
  use 16
  sorry

end max_value_a_l211_211777


namespace find_a_l211_211710

theorem find_a (a : ℝ) : (-2 * a + 3 = -4) -> (a = 7 / 2) :=
by
  intro h
  sorry

end find_a_l211_211710


namespace distance_between_homes_l211_211475

theorem distance_between_homes (Maxwell_speed : ℝ) (Brad_speed : ℝ) (M_time : ℝ) (B_delay : ℝ) (D : ℝ) 
  (h1 : Maxwell_speed = 4) 
  (h2 : Brad_speed = 6)
  (h3 : M_time = 8)
  (h4 : B_delay = 1) :
  D = 74 :=
by
  sorry

end distance_between_homes_l211_211475


namespace savings_calculation_l211_211079

-- Define the conditions
def income := 17000
def ratio_income_expenditure := 5 / 4

-- Prove that the savings are Rs. 3400
theorem savings_calculation (h : income = 5 * 3400): (income - 4 * 3400) = 3400 :=
by sorry

end savings_calculation_l211_211079


namespace find_sum_l211_211892

variable (a b c d : ℝ)

theorem find_sum :
  (ab + bc + cd + da = 20) →
  (b + d = 4) →
  (a + c = 5) := by
  sorry

end find_sum_l211_211892


namespace Arman_total_earnings_two_weeks_l211_211721

theorem Arman_total_earnings_two_weeks :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let this_week_increase := 0.5
  let initial_rate := 10
  let this_week_rate := initial_rate + this_week_increase
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := this_week_hours * this_week_rate
  let total_earnings := last_week_earnings + this_week_earnings
  total_earnings = 770 := 
by
  sorry

end Arman_total_earnings_two_weeks_l211_211721


namespace difference_in_x_coordinates_is_constant_l211_211163

variable {a x₀ y₀ k : ℝ}

-- Define the conditions
def point_on_x_axis (a : ℝ) : Prop := true

def passes_through_fixed_point_and_tangent (a : ℝ) : Prop :=
  a = 1

def equation_of_curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

def tangent_condition (a x₀ y₀ : ℝ) (k : ℝ) : Prop :=
  a > 2 ∧ y₀ > 0 ∧ y₀^2 = 4 * x₀ ∧ 
  (4 * x₀ - 2 * y₀ * y₀ + y₀^2 = 0)

-- The statement
theorem difference_in_x_coordinates_is_constant (a x₀ y₀ k : ℝ) :
  point_on_x_axis a →
  passes_through_fixed_point_and_tangent a →
  equation_of_curve_C x₀ y₀ →
  tangent_condition a x₀ y₀ k → 
  a - x₀ = 2 :=
by
  intro h1 h2 h3 h4 
  sorry

end difference_in_x_coordinates_is_constant_l211_211163


namespace find_initial_money_l211_211588

def initial_money (s1 s2 s3 : ℝ) : ℝ :=
  let after_store_1 := s1 - (0.4 * s1 + 4)
  let after_store_2 := after_store_1 - (0.5 * after_store_1 + 5)
  let after_store_3 := after_store_2 - (0.6 * after_store_2 + 6)
  after_store_3

theorem find_initial_money (s1 s2 s3 : ℝ) (hs3 : initial_money s1 s2 s3 = 2) : s1 = 90 :=
by
  -- Placeholder for the actual proof
  sorry

end find_initial_money_l211_211588


namespace actual_distance_traveled_l211_211394

theorem actual_distance_traveled (D : ℝ) (h : D / 10 = (D + 20) / 20) : D = 20 :=
  sorry

end actual_distance_traveled_l211_211394


namespace evaluate_expression_at_2_l211_211182

theorem evaluate_expression_at_2 : ∀ (x : ℕ), x = 2 → (x^x)^(x^(x^x)) = 4294967296 := by
  intros x h
  rw [h]
  sorry

end evaluate_expression_at_2_l211_211182


namespace initial_cases_purchased_l211_211966

open Nat

-- Definitions based on conditions

def group1_children := 14
def group2_children := 16
def group3_children := 12
def group4_children := (group1_children + group2_children + group3_children) / 2
def total_children := group1_children + group2_children + group3_children + group4_children

def bottles_per_child_per_day := 3
def days := 3
def total_bottles_needed := total_children * bottles_per_child_per_day * days

def additional_bottles_needed := 255

def bottles_per_case := 24
def initial_bottles := total_bottles_needed - additional_bottles_needed

def cases_purchased := initial_bottles / bottles_per_case

-- Theorem to prove the number of cases purchased initially
theorem initial_cases_purchased : cases_purchased = 13 :=
  sorry

end initial_cases_purchased_l211_211966


namespace test_point_selection_0618_method_l211_211931

theorem test_point_selection_0618_method :
  ∀ (x1 x2 x3 : ℝ),
    1000 + 0.618 * (2000 - 1000) = x1 →
    1000 + (2000 - x1) = x2 →
    x2 < x1 →
    (∀ (f : ℝ → ℝ), f x2 < f x1) →
    x1 + (1000 - x2) = x3 →
    x3 = 1236 :=
by
  intros x1 x2 x3 h1 h2 h3 h4 h5
  sorry

end test_point_selection_0618_method_l211_211931


namespace find_x_l211_211473

theorem find_x : ∃ x : ℝ, (3 * (x + 2 - 6)) / 4 = 3 ∧ x = 8 :=
by
  sorry

end find_x_l211_211473


namespace lila_substituted_value_l211_211468

theorem lila_substituted_value:
  let a := 2
  let b := 3
  let c := 4
  let d := 5
  let f := 6
  ∃ e : ℚ, 20 * e = 2 * (3 - 4 * (5 - (e / 6))) ∧ e = -51 / 28 := sorry

end lila_substituted_value_l211_211468


namespace pencil_fraction_white_part_l211_211829

theorem pencil_fraction_white_part
  (L : ℝ )
  (H1 : L = 9.333333333333332)
  (H2 : (1 / 8) * L + (7 / 12 * 7 / 8) * (7 / 8) * L + W * (7 / 8) * L = L) :
  W = 5 / 12 :=
by
  sorry

end pencil_fraction_white_part_l211_211829


namespace frank_used_2_bags_l211_211462

theorem frank_used_2_bags (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : (total_candy / candy_per_bag) = 2 := 
by
  sorry

end frank_used_2_bags_l211_211462


namespace angle_of_inclination_45_l211_211027

def plane (x y z : ℝ) : Prop := (x = y) ∧ (y = z)
def image_planes (x y : ℝ) : Prop := (x = 45 ∧ y = 45)

theorem angle_of_inclination_45 (t₁₂ : ℝ) :
  ∃ θ: ℝ, (plane t₁₂ t₁₂ t₁₂ → image_planes 45 45 → θ = 45) :=
sorry

end angle_of_inclination_45_l211_211027


namespace solve_system_equations_l211_211704

theorem solve_system_equations (x y : ℝ) :
  (5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8) ↔
  (x = -1 ∧ y = 2) ∨ (x = 11 ∧ y = -7) ∨ (x = -11 ∧ y = 7) ∨ (x = 1 ∧ y = -2) := 
sorry

end solve_system_equations_l211_211704


namespace part1_part2_l211_211927

variable (a b : ℝ) (x : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) : (a^2 / b) + (b^2 / a) ≥ a + b :=
sorry

theorem part2 (h3 : 0 < x) (h4 : x < 1) : 
(∀ y : ℝ, y = ((1 - x)^2 / x) + (x^2 / (1 - x)) → y ≥ 1) ∧ ((1 - x) = x → y = 1) :=
sorry

end part1_part2_l211_211927


namespace remainder_of_product_mod_7_l211_211539

theorem remainder_of_product_mod_7
  (a b c : ℕ)
  (ha : a ≡ 2 [MOD 7])
  (hb : b ≡ 3 [MOD 7])
  (hc : c ≡ 4 [MOD 7]) :
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_mod_7_l211_211539


namespace total_money_l211_211454

variable (A B C: ℕ)
variable (h1: A + C = 200) 
variable (h2: B + C = 350)
variable (h3: C = 200)

theorem total_money : A + B + C = 350 :=
by
  sorry

end total_money_l211_211454


namespace height_of_tank_B_l211_211576

noncomputable def height_tank_A : ℝ := 5
noncomputable def circumference_tank_A : ℝ := 4
noncomputable def circumference_tank_B : ℝ := 10
noncomputable def capacity_ratio : ℝ := 0.10000000000000002

theorem height_of_tank_B {h_B : ℝ} 
  (h_tank_A : height_tank_A = 5)
  (c_tank_A : circumference_tank_A = 4)
  (c_tank_B : circumference_tank_B = 10)
  (capacity_percentage : capacity_ratio = 0.10000000000000002)
  (V_A : ℝ := π * (2 / π)^2 * height_tank_A)
  (V_B : ℝ := π * (5 / π)^2 * h_B)
  (capacity_relation : V_A = capacity_ratio * V_B) :
  h_B = 8 :=
sorry

end height_of_tank_B_l211_211576


namespace compare_trig_values_l211_211620

noncomputable def a : ℝ := Real.tan (-7 * Real.pi / 6)
noncomputable def b : ℝ := Real.cos (23 * Real.pi / 4)
noncomputable def c : ℝ := Real.sin (-33 * Real.pi / 4)

theorem compare_trig_values : c < a ∧ a < b := sorry

end compare_trig_values_l211_211620


namespace sum_of_digits_of_smallest_number_l211_211692

noncomputable def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_of_smallest_number :
  (n : Nat) → (h1 : (Nat.ceil (n / 2) - Nat.ceil (n / 3) = 15)) → 
  sum_of_digits n = 9 :=
by
  sorry

end sum_of_digits_of_smallest_number_l211_211692


namespace reaction_spontaneous_at_high_temperature_l211_211464

theorem reaction_spontaneous_at_high_temperature
  (ΔH : ℝ) (ΔS : ℝ) (T : ℝ) (ΔG : ℝ)
  (h_ΔH_pos : ΔH > 0)
  (h_ΔS_pos : ΔS > 0)
  (h_ΔG_eq : ΔG = ΔH - T * ΔS) :
  (∃ T_high : ℝ, T_high > 0 ∧ ΔG < 0) := sorry

end reaction_spontaneous_at_high_temperature_l211_211464


namespace leak_drain_time_l211_211075

theorem leak_drain_time (P L : ℕ → ℕ) (H1 : ∀ t, P t = 1 / 2) (H2 : ∀ t, P t - L t = 1 / 3) : 
  (1 / L 1) = 6 :=
by
  sorry

end leak_drain_time_l211_211075


namespace no_solutions_988_1991_l211_211886

theorem no_solutions_988_1991 :
    ¬ ∃ (m n : ℤ),
      (988 ≤ m ∧ m ≤ 1991) ∧
      (988 ≤ n ∧ n ≤ 1991) ∧
      m ≠ n ∧
      ∃ (a b : ℤ), (mn + n = a^2 ∧ mn + m = b^2) := sorry

end no_solutions_988_1991_l211_211886


namespace quadratic_distinct_roots_l211_211047

theorem quadratic_distinct_roots (m : ℝ) : 
  ((m - 2) * x ^ 2 + 2 * x + 1 = 0) → (m < 3 ∧ m ≠ 2) :=
by
  sorry

end quadratic_distinct_roots_l211_211047


namespace hexagon_area_l211_211764

theorem hexagon_area (A C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hC : C = (2 * Real.sqrt 3, 2)) : 
  6 * Real.sqrt 3 = 6 * Real.sqrt 3 := 
by sorry

end hexagon_area_l211_211764


namespace number_of_people_got_off_at_third_stop_l211_211950

-- Definitions for each stop
def initial_passengers : ℕ := 0
def passengers_after_first_stop : ℕ := initial_passengers + 7
def passengers_after_second_stop : ℕ := passengers_after_first_stop - 3 + 5
def passengers_after_third_stop (x : ℕ) : ℕ := passengers_after_second_stop - x + 4

-- Final condition stating there are 11 passengers after the third stop
def final_passengers : ℕ := 11

-- Proof goal
theorem number_of_people_got_off_at_third_stop (x : ℕ) :
  passengers_after_third_stop x = final_passengers → x = 2 :=
by
  -- proof goes here
  sorry

end number_of_people_got_off_at_third_stop_l211_211950


namespace range_m_graph_in_quadrants_l211_211688

theorem range_m_graph_in_quadrants (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (m + 2) / x > 0) ∧ (x < 0 → (m + 2) / x < 0))) ↔ m > -2 :=
by 
  sorry

end range_m_graph_in_quadrants_l211_211688


namespace factorize_expression_l211_211038

variable {R : Type} [CommRing R]

theorem factorize_expression (x y : R) : 
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := 
by 
  sorry

end factorize_expression_l211_211038


namespace find_DG_l211_211019

theorem find_DG (a b k l : ℕ) (h1 : a * k = 37 * (a + b)) (h2 : b * l = 37 * (a + b)) : 
  k = 1406 :=
by
  sorry

end find_DG_l211_211019


namespace simplify_expression_l211_211135

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end simplify_expression_l211_211135


namespace qualified_weight_example_l211_211449

-- Define the range of qualified weights
def is_qualified_weight (w : ℝ) : Prop :=
  9.9 ≤ w ∧ w ≤ 10.1

-- State the problem: show that 10 kg is within the qualified range
theorem qualified_weight_example : is_qualified_weight 10 :=
  by
    sorry

end qualified_weight_example_l211_211449


namespace coconuts_for_crab_l211_211896

theorem coconuts_for_crab (C : ℕ) (H1 : 6 * C * 19 = 342) : C = 3 :=
sorry

end coconuts_for_crab_l211_211896


namespace factory_hours_per_day_l211_211487

def factory_produces (hours_per_day : ℕ) : Prop :=
  let refrigerators_per_hour := 90
  let coolers_per_hour := 160
  let total_products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_products_in_5_days := 11250
  total_products_per_hour * (5 * hours_per_day) = total_products_in_5_days

theorem factory_hours_per_day : ∃ h : ℕ, factory_produces h ∧ h = 9 :=
by
  existsi 9
  unfold factory_produces
  sorry

end factory_hours_per_day_l211_211487


namespace find_a_l211_211181

theorem find_a (a : ℝ) : 
  (∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
  ∃ y, y ≤ 6 ∧ 
  (∀ x, x^2 + y^2 = a^2 ∧ 
  x^2 + y^2 + a * y - 6 = 0)) → 
  a = 2 ∨ a = -2 :=
by sorry

end find_a_l211_211181


namespace smallest_sum_B_c_l211_211875

theorem smallest_sum_B_c 
  (B c : ℕ) 
  (h1 : B ≤ 4) 
  (h2 : 6 < c) 
  (h3 : 31 * B = 4 * (c + 1)) : 
  B + c = 34 := 
sorry

end smallest_sum_B_c_l211_211875


namespace minimum_value_is_8_l211_211535

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end minimum_value_is_8_l211_211535


namespace solve_for_y_l211_211823

noncomputable def solve_quadratic := {y : ℂ // 4 + 3 * y^2 = 0.7 * y - 40}

theorem solve_for_y : 
  ∃ y : ℂ, (y = 0.1167 + 3.8273 * Complex.I ∨ y = 0.1167 - 3.8273 * Complex.I) ∧
            (4 + 3 * y^2 = 0.7 * y - 40) :=
by
  sorry

end solve_for_y_l211_211823


namespace goods_train_speed_l211_211096

theorem goods_train_speed
  (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
  (h_train_length : length_train = 250.0416)
  (h_platform_length : length_platform = 270)
  (h_time : time_seconds = 26) :
  (length_train + length_platform) / time_seconds * 3.6 = 72 := by
    sorry

end goods_train_speed_l211_211096


namespace max_profit_l211_211304

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end max_profit_l211_211304


namespace ratio_cost_price_selling_price_l211_211425

theorem ratio_cost_price_selling_price (CP SP : ℝ) (h : SP = 1.5 * CP) : CP / SP = 2 / 3 :=
by
  sorry

end ratio_cost_price_selling_price_l211_211425


namespace more_students_than_rabbits_l211_211306

theorem more_students_than_rabbits :
  let number_of_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let total_students := students_per_classroom * number_of_classrooms
  let total_rabbits := rabbits_per_classroom * number_of_classrooms
  total_students - total_rabbits = 95 := by
  sorry

end more_students_than_rabbits_l211_211306


namespace aaron_pages_sixth_day_l211_211724

theorem aaron_pages_sixth_day 
  (h1 : 18 + 12 + 23 + 10 + 17 + y = 6 * 15) : 
  y = 10 :=
by
  sorry

end aaron_pages_sixth_day_l211_211724


namespace A_completes_job_alone_l211_211874

theorem A_completes_job_alone (efficiency_B efficiency_A total_work days_A : ℝ) :
  efficiency_A = 1.3 * efficiency_B → 
  total_work = (efficiency_A + efficiency_B) * 13 → 
  days_A = total_work / efficiency_A → 
  days_A = 23 :=
by
  intros h1 h2 h3
  sorry

end A_completes_job_alone_l211_211874


namespace factor_and_sum_coeffs_l211_211782

noncomputable def sum_of_integer_coeffs_of_factorization (x y : ℤ) : ℤ :=
  let factors := ([(1 : ℤ), (-1 : ℤ), (5 : ℤ), (1 : ℤ), (6 : ℤ), (1 : ℤ), (1 : ℤ), (5 : ℤ), (-1 : ℤ), (6 : ℤ)])
  factors.sum

theorem factor_and_sum_coeffs (x y : ℤ) :
  (125 * (x^9:ℤ) - 216 * (y^9:ℤ) = (x - y) * (5 * x^2 + x * y + 6 * y^2) * (x + y) * (5 * x^2 - x * y + 6 * y^2))
  ∧ (sum_of_integer_coeffs_of_factorization x y = 24) :=
by
  sorry

end factor_and_sum_coeffs_l211_211782


namespace base6_arithmetic_l211_211580

def base6_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  let n4 := n3 / 10
  let d4 := n4 % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0

def base10_to_base6 (n : ℕ) : ℕ :=
  let b4 := n / 6^4
  let r4 := n % 6^4
  let b3 := r4 / 6^3
  let r3 := r4 % 6^3
  let b2 := r3 / 6^2
  let r2 := r3 % 6^2
  let b1 := r2 / 6^1
  let b0 := r2 % 6^1
  b4 * 10000 + b3 * 1000 + b2 * 100 + b1 * 10 + b0

theorem base6_arithmetic : 
  base10_to_base6 ((base6_to_base10 45321 - base6_to_base10 23454) + base6_to_base10 14553) = 45550 :=
by
  sorry

end base6_arithmetic_l211_211580


namespace calc_sum_of_digits_l211_211752

theorem calc_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10) 
(hm : 10 * 3 + x = 34) (hmy : 34 * (10 * y + 4) = 136) : x + y = 7 :=
sorry

end calc_sum_of_digits_l211_211752


namespace smallest_positive_integer_l211_211435

theorem smallest_positive_integer (n : ℕ) : 13 * n ≡ 567 [MOD 5] ↔ n = 4 := by
  sorry

end smallest_positive_integer_l211_211435


namespace product_of_roots_l211_211696

theorem product_of_roots :
  ∀ a b c : ℚ, (a ≠ 0) → a = 24 → b = 60 → c = -600 → (c / a) = -25 :=
sorry

end product_of_roots_l211_211696


namespace negation_of_every_planet_orbits_the_sun_l211_211878

variables (Planet : Type) (orbits_sun : Planet → Prop)

theorem negation_of_every_planet_orbits_the_sun :
  (¬ ∀ x : Planet, (¬ (¬ (exists x : Planet, true)) → orbits_sun x)) ↔
  ∃ x : Planet, ¬ orbits_sun x :=
by sorry

end negation_of_every_planet_orbits_the_sun_l211_211878


namespace domain_of_f_l211_211031

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3 * x + 2)

theorem domain_of_f :
  {x : ℝ | (x < 1) ∨ (1 < x ∧ x < 2) ∨ (x > 2)} = 
  {x : ℝ | f x ≠ 0} :=
sorry

end domain_of_f_l211_211031


namespace polynomial_coeff_sum_l211_211758

theorem polynomial_coeff_sum (a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x: ℝ, (x - 1) ^ 4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_4 - a_3 + a_2 - a_1 + a_0 = 16 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l211_211758


namespace student_arrangement_l211_211652

theorem student_arrangement :
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  valid_arrangements = 336 :=
by
  let total_arrangements := 720
  let ab_violations := 240
  let cd_violations := 240
  let ab_cd_overlap := 96
  let restricted_arrangements := ab_violations + cd_violations - ab_cd_overlap
  let valid_arrangements := total_arrangements - restricted_arrangements
  exact sorry

end student_arrangement_l211_211652


namespace functional_eq_implies_odd_l211_211894

variable {f : ℝ → ℝ}

def functional_eq (f : ℝ → ℝ) :=
∀ a b, f (a + b) + f (a - b) = 2 * f a * Real.cos b

theorem functional_eq_implies_odd (h : functional_eq f) (hf_non_zero : ¬∀ x, f x = 0) : 
  ∀ x, f (-x) = -f x := 
by
  sorry

end functional_eq_implies_odd_l211_211894


namespace total_people_hired_l211_211344

theorem total_people_hired (H L : ℕ) (hL : L = 1) (payroll : ℕ) (hPayroll : 129 * H + 82 * L = 3952) : H + L = 31 := by
  sorry

end total_people_hired_l211_211344


namespace floor_sum_min_value_l211_211000

theorem floor_sum_min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋) = 7 :=
sorry

end floor_sum_min_value_l211_211000


namespace train_journey_duration_l211_211843

def battery_lifespan (talk_time standby_time : ℝ) :=
  talk_time <= 6 ∧ standby_time <= 210

def full_battery_usage (total_time : ℝ) :=
  (total_time / 2) / 6 + (total_time / 2) / 210 = 1

theorem train_journey_duration (t : ℝ) (h1 : battery_lifespan (t / 2) (t / 2)) (h2 : full_battery_usage t) :
  t = 35 / 3 :=
sorry

end train_journey_duration_l211_211843


namespace positive_difference_in_x_coordinates_l211_211373

-- Define points for line l
def point_l1 : ℝ × ℝ := (0, 10)
def point_l2 : ℝ × ℝ := (2, 0)

-- Define points for line m
def point_m1 : ℝ × ℝ := (0, 3)
def point_m2 : ℝ × ℝ := (10, 0)

-- Define the proof statement with the given problem
theorem positive_difference_in_x_coordinates :
  let y := 20
  let slope_l := (point_l2.2 - point_l1.2) / (point_l2.1 - point_l1.1)
  let intersection_l_x := (y - point_l1.2) / slope_l + point_l1.1
  let slope_m := (point_m2.2 - point_m1.2) / (point_m2.1 - point_m1.1)
  let intersection_m_x := (y - point_m1.2) / slope_m + point_m1.1
  abs (intersection_l_x - intersection_m_x) = 54.67 := 
  sorry -- Proof goes here

end positive_difference_in_x_coordinates_l211_211373


namespace determine_y_l211_211261

theorem determine_y (x y : ℤ) (h1 : x^2 + 4 * x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  intros
  sorry

end determine_y_l211_211261


namespace common_chord_equation_l211_211289

-- Definitions of the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 14*y + 15 = 0

-- Definition of the common chord line
def common_chord_line (x y : ℝ) : Prop := 6*x + 8*y - 3 = 0

-- The theorem to be proved
theorem common_chord_equation :
  (∀ x y, circle1 x y → circle2 x y → common_chord_line x y) :=
by sorry

end common_chord_equation_l211_211289


namespace people_needed_to_mow_lawn_in_4_hours_l211_211926

-- Define the given constants and conditions
def n := 4
def t := 6
def c := n * t -- The total work that can be done in constant hours
def t' := 4

-- Define the new number of people required to complete the work in t' hours
def n' := c / t'

-- Define the problem statement
theorem people_needed_to_mow_lawn_in_4_hours : n' - n = 2 := 
sorry

end people_needed_to_mow_lawn_in_4_hours_l211_211926


namespace james_total_payment_l211_211884

noncomputable def first_pair_cost : ℝ := 40
noncomputable def second_pair_cost : ℝ := 60
noncomputable def discount_applied_to : ℝ := min first_pair_cost second_pair_cost
noncomputable def discount_amount := discount_applied_to / 2
noncomputable def total_before_extra_discount := first_pair_cost + (second_pair_cost - discount_amount)
noncomputable def extra_discount := total_before_extra_discount / 4
noncomputable def final_amount := total_before_extra_discount - extra_discount

theorem james_total_payment : final_amount = 60 := by
  sorry

end james_total_payment_l211_211884


namespace find_f_x_l211_211323

def tan : ℝ → ℝ := sorry  -- tan function placeholder
def cos : ℝ → ℝ := sorry  -- cos function placeholder
def sin : ℝ → ℝ := sorry  -- sin function placeholder

axiom conditions : 
  tan 45 = 1 ∧
  cos 60 = 2 ∧
  sin 90 = 3 ∧
  cos 180 = 4 ∧
  sin 270 = 5

theorem find_f_x :
  ∃ f x, (f x = 6) ∧ 
  (f = tan ∧ x = 360) := 
sorry

end find_f_x_l211_211323


namespace number_of_sets_X_l211_211836

noncomputable def finite_set_problem (M A B : Finset ℕ) : Prop :=
  (M.card = 10) ∧ 
  (A ⊆ M) ∧ 
  (B ⊆ M) ∧ 
  (A ∩ B = ∅) ∧ 
  (A.card = 2) ∧ 
  (B.card = 3) ∧ 
  (∃ (X : Finset ℕ), X ⊆ M ∧ ¬(A ⊆ X) ∧ ¬(B ⊆ X))

theorem number_of_sets_X (M A B : Finset ℕ) (h : finite_set_problem M A B) : 
  ∃ n : ℕ, n = 672 := 
sorry

end number_of_sets_X_l211_211836


namespace sum_of_fourth_powers_of_solutions_l211_211549

theorem sum_of_fourth_powers_of_solutions (x y : ℝ)
  (h : |x^2 - 2 * x + 1/1004| = 1/1004 ∨ |y^2 - 2 * y + 1/1004| = 1/1004) :
  x^4 + y^4 = 20160427280144 / 12600263001 :=
sorry

end sum_of_fourth_powers_of_solutions_l211_211549


namespace area_increase_300_percent_l211_211650

noncomputable def percentage_increase_of_area (d : ℝ) : ℝ :=
  let d' := 2 * d
  let r := d / 2
  let r' := d' / 2
  let A := Real.pi * r^2
  let A' := Real.pi * (r')^2
  100 * (A' - A) / A

theorem area_increase_300_percent (d : ℝ) : percentage_increase_of_area d = 300 :=
by
  sorry

end area_increase_300_percent_l211_211650


namespace range_of_c_l211_211460

theorem range_of_c (a c : ℝ) (ha : a ≥ 1 / 8)
  (h : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 :=
sorry

end range_of_c_l211_211460


namespace sum_of_ages_l211_211891

theorem sum_of_ages (P K : ℕ) (h1 : P - 7 = 3 * (K - 7)) (h2 : P + 2 = 2 * (K + 2)) : P + K = 50 :=
by
  sorry

end sum_of_ages_l211_211891


namespace john_new_weekly_earnings_l211_211395

theorem john_new_weekly_earnings :
  ∀ (original_earnings : ℤ) (percentage_increase : ℝ),
  original_earnings = 60 →
  percentage_increase = 66.67 →
  (original_earnings + (percentage_increase / 100 * original_earnings)) = 100 := 
by
  intros original_earnings percentage_increase h_earnings h_percentage
  rw [h_earnings, h_percentage]
  norm_num
  sorry

end john_new_weekly_earnings_l211_211395


namespace highest_probability_of_red_ball_l211_211132

theorem highest_probability_of_red_ball (red yellow white blue : ℕ) (H1 : red = 5) (H2 : yellow = 4) (H3 : white = 1) (H4 : blue = 3) :
  (red : ℚ) / (red + yellow + white + blue) > (yellow : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (white : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (blue : ℚ) / (red + yellow + white + blue) := 
by {
  sorry
}

end highest_probability_of_red_ball_l211_211132


namespace find_values_of_real_numbers_l211_211293

theorem find_values_of_real_numbers (x y : ℝ)
  (h : 2 * x - 1 + (y + 1) * Complex.I = x - y - (x + y) * Complex.I) :
  x = 3 ∧ y = -2 :=
sorry

end find_values_of_real_numbers_l211_211293


namespace D_coordinates_l211_211587

namespace Parallelogram

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 2 }
def C : Point := { x := 3, y := 1 }

theorem D_coordinates :
  ∃ D : Point, D = { x := 2, y := -1 } ∧ ∀ A B C D : Point, 
    (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) := by
  sorry

end Parallelogram

end D_coordinates_l211_211587


namespace second_number_less_than_first_by_16_percent_l211_211122

variable (X : ℝ)

theorem second_number_less_than_first_by_16_percent
  (h1 : X > 0)
  (first_num : ℝ := 0.75 * X)
  (second_num : ℝ := 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 16 := by
  sorry

end second_number_less_than_first_by_16_percent_l211_211122


namespace no_base_131_cubed_l211_211637

open Nat

theorem no_base_131_cubed (n : ℕ) (k : ℕ) : 
  (4 ≤ n ∧ n ≤ 12) ∧ (1 * n^2 + 3 * n + 1 = k^3) → False := by
  sorry

end no_base_131_cubed_l211_211637


namespace rock_paper_scissors_l211_211354

open Nat

-- Definitions based on problem conditions
def personA_movement (x y z : ℕ) : ℤ :=
  3 * (x : ℤ) - 2 * (y : ℤ) + (z : ℤ)

def personB_movement (x y z : ℕ) : ℤ :=
  3 * (y : ℤ) - 2 * (x : ℤ) + (z : ℤ)

def total_rounds (x y z : ℕ) : ℕ :=
  x + y + z

-- Problem statement
theorem rock_paper_scissors (x y z : ℕ) 
  (h1 : total_rounds x y z = 15)
  (h2 : personA_movement x y z = 17)
  (h3 : personB_movement x y z = 2) : x = 7 :=
by
  sorry

end rock_paper_scissors_l211_211354


namespace candy_cost_l211_211544

theorem candy_cost (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) 
  (total_tickets : ℕ) (candies : ℕ) (cost_per_candy : ℕ) 
  (h1 : tickets_whack_a_mole = 8) (h2 : tickets_skee_ball = 7)
  (h3 : total_tickets = tickets_whack_a_mole + tickets_skee_ball)
  (h4 : candies = 3) (h5 : total_tickets = candies * cost_per_candy) :
  cost_per_candy = 5 :=
by
  sorry

end candy_cost_l211_211544


namespace inequality_solution_l211_211655

theorem inequality_solution (a c : ℝ) (h : ∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ ax^2 + 5*x + c > 0) : a + c = -7 :=
sorry

end inequality_solution_l211_211655


namespace simplify_and_evaluate_l211_211423

theorem simplify_and_evaluate 
  (a b : ℤ)
  (h1 : a = 2)
  (h2 : b = -1) : 
  (2 * a^2 * b - 4 * a * b^2) - 2 * (a * b^2 + a^2 * b) = -12 := 
by
  rw [h1, h2]
  sorry

end simplify_and_evaluate_l211_211423


namespace train_speed_54_kmh_l211_211392

theorem train_speed_54_kmh
  (train_length : ℕ)
  (tunnel_length : ℕ)
  (time_seconds : ℕ)
  (total_distance : ℕ := train_length + tunnel_length)
  (speed_mps : ℚ := total_distance / time_seconds)
  (conversion_factor : ℚ := 3.6) :
  train_length = 300 →
  tunnel_length = 1200 →
  time_seconds = 100 →
  speed_mps * conversion_factor = 54 := 
by
  intros h_train_length h_tunnel_length h_time_seconds
  sorry

end train_speed_54_kmh_l211_211392


namespace A_and_B_finish_together_in_20_days_l211_211773

noncomputable def W_B : ℝ := 1 / 30

noncomputable def W_A : ℝ := 1 / 2 * W_B

noncomputable def W_A_plus_B : ℝ := W_A + W_B

theorem A_and_B_finish_together_in_20_days :
  (1 / W_A_plus_B) = 20 :=
by
  sorry

end A_and_B_finish_together_in_20_days_l211_211773


namespace find_prime_b_l211_211551

-- Define the polynomial function f
def f (n a : ℕ) : ℕ := n^3 - 4 * a * n^2 - 12 * n + 144

-- Define b as a prime number
def b (n : ℕ) (a : ℕ) : ℕ := f n a

-- Theorem statement
theorem find_prime_b (n : ℕ) (a : ℕ) (h : n = 7) (ha : a = 2) (hb : ∃ p : ℕ, Nat.Prime p ∧ p = b n a) :
  b n a = 11 :=
by
  sorry

end find_prime_b_l211_211551


namespace positive_integer_k_l211_211877

theorem positive_integer_k (k x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = k * x * y * z) :
  k = 1 ∨ k = 3 :=
sorry

end positive_integer_k_l211_211877


namespace polynomial_sum_l211_211804

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l211_211804


namespace net_change_correct_l211_211382
-- Import the necessary library

-- Price calculation function
def price_after_changes (initial_price: ℝ) (changes: List (ℝ -> ℝ)): ℝ :=
  changes.foldl (fun price change => change price) initial_price

-- Define each model's price changes
def modelA_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.9, 
  fun price => price * 1.3, 
  fun price => price * 0.85
]

def modelB_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.85, 
  fun price => price * 1.25, 
  fun price => price * 0.80
]

def modelC_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.80, 
  fun price => price * 1.20, 
  fun price => price * 0.95
]

-- Calculate final prices
def final_price_modelA := price_after_changes 1000 modelA_changes
def final_price_modelB := price_after_changes 1500 modelB_changes
def final_price_modelC := price_after_changes 2000 modelC_changes

-- Calculate net changes
def net_change_modelA := final_price_modelA - 1000
def net_change_modelB := final_price_modelB - 1500
def net_change_modelC := final_price_modelC - 2000

-- Set up theorem
theorem net_change_correct:
  net_change_modelA = -5.5 ∧ net_change_modelB = -225 ∧ net_change_modelC = -176 := by
  -- Proof is skipped
  sorry

end net_change_correct_l211_211382


namespace faster_speed_l211_211167

theorem faster_speed (x : ℝ) (h1 : 10 ≠ 0) (h2 : 5 * 10 = 50) (h3 : 50 + 20 = 70) (h4 : 5 = 70 / x) : x = 14 :=
by
  -- proof steps go here
  sorry

end faster_speed_l211_211167


namespace range_of_k_for_intersecting_circles_l211_211940

/-- Given circle \( C \) with equation \( x^2 + y^2 - 8x + 15 = 0 \) and a line \( y = kx - 2 \),
    prove that if there exists at least one point on the line such that a circle with this point
    as the center and a radius of 1 intersects with circle \( C \), then \( 0 \leq k \leq \frac{4}{3} \). -/
theorem range_of_k_for_intersecting_circles (k : ℝ) :
  (∃ (x y : ℝ), y = k * x - 2 ∧ (x - 4) ^ 2 + y ^ 2 - 1 ≤ 1) → 0 ≤ k ∧ k ≤ 4 / 3 :=
by {
  sorry
}

end range_of_k_for_intersecting_circles_l211_211940


namespace kendall_total_distance_l211_211499

def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5
def total_distance : ℝ := 0.67

theorem kendall_total_distance :
  (distance_with_mother + distance_with_father = total_distance) :=
sorry

end kendall_total_distance_l211_211499


namespace ratio_of_shares_l211_211983

theorem ratio_of_shares (A B C : ℝ) (x : ℝ):
  A = 240 → 
  A + B + C = 600 →
  A = x * (B + C) →
  B = (2/3) * (A + C) →
  A / (B + C) = 2 / 3 :=
by
  intros hA hTotal hFraction hB
  sorry

end ratio_of_shares_l211_211983


namespace evaluate_expression_l211_211541

theorem evaluate_expression :
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 :=
by 
  sorry

end evaluate_expression_l211_211541


namespace range_of_m_l211_211723

noncomputable def proof_problem (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) : Prop :=
  ∃ x y : ℝ, (0 < x) ∧ (0 < y) ∧ (1/x + 2/y = 1) ∧ (x + y / 2 < m^2 + 3 * m) ↔ (m < -4 ∨ m > 1)

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) :
  proof_problem x y m hx hy hxy :=
sorry

end range_of_m_l211_211723


namespace coprime_repeating_decimal_sum_l211_211055

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end coprime_repeating_decimal_sum_l211_211055


namespace trigonometric_identity_l211_211349

variable (α : ℝ)

theorem trigonometric_identity :
  4.9 * (Real.sin (7 * Real.pi / 8 - 2 * α))^2 - (Real.sin (9 * Real.pi / 8 - 2 * α))^2 = 
  Real.sin (4 * α) / Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l211_211349


namespace distance_between_points_on_parabola_l211_211106

theorem distance_between_points_on_parabola :
  ∀ (x1 x2 y1 y2 : ℝ), 
    (y1^2 = 4 * x1) → (y2^2 = 4 * x2) → (x2 = x1 + 2) → (|y2 - y1| = 4 * Real.sqrt x2 - 4 * Real.sqrt x1) →
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 8 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4
  sorry

end distance_between_points_on_parabola_l211_211106


namespace twelve_point_five_minutes_in_seconds_l211_211014

-- Definitions
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- Theorem: Prove that 12.5 minutes is 750 seconds
theorem twelve_point_five_minutes_in_seconds : minutes_to_seconds 12.5 = 750 :=
by 
  sorry

end twelve_point_five_minutes_in_seconds_l211_211014


namespace train_length_l211_211985

theorem train_length 
    (t : ℝ) 
    (s_kmh : ℝ) 
    (s_mps : ℝ)
    (h1 : t = 2.222044458665529) 
    (h2 : s_kmh = 162) 
    (h3 : s_mps = s_kmh * (5 / 18))
    (L : ℝ)
    (h4 : L = s_mps * t) : 
  L = 100 := 
sorry

end train_length_l211_211985


namespace smallest_x_for_non_prime_expression_l211_211961

/-- The smallest positive integer x for which x^2 + x + 41 is not a prime number is 40. -/
theorem smallest_x_for_non_prime_expression : ∃ x : ℕ, x > 0 ∧ x^2 + x + 41 = 41 * 41 ∧ (∀ y : ℕ, 0 < y ∧ y < x → Prime (y^2 + y + 41)) := 
sorry

end smallest_x_for_non_prime_expression_l211_211961


namespace sufficient_but_not_necessary_l211_211191

-- Definitions of conditions
def p (x : ℝ) : Prop := 1 / (x + 1) > 0
def q (x : ℝ) : Prop := (1/x > 0)

-- Main theorem statement
theorem sufficient_but_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
sorry

end sufficient_but_not_necessary_l211_211191


namespace quinton_total_fruit_trees_l211_211139

-- Define the given conditions
def num_apple_trees := 2
def width_apple_tree_ft := 10
def space_between_apples_ft := 12
def width_peach_tree_ft := 12
def space_between_peaches_ft := 15
def total_space_ft := 71

-- Definition that calculates the total number of fruit trees Quinton wants to plant
def total_fruit_trees : ℕ := 
  let space_apple_trees := num_apple_trees * width_apple_tree_ft + space_between_apples_ft
  let space_remaining_for_peaches := total_space_ft - space_apple_trees
  1 + space_remaining_for_peaches / (width_peach_tree_ft + space_between_peaches_ft) + num_apple_trees

-- The statement to prove
theorem quinton_total_fruit_trees : total_fruit_trees = 4 := by
  sorry

end quinton_total_fruit_trees_l211_211139


namespace amount_brought_by_sisters_l211_211643

-- Definitions based on conditions
def cost_per_ticket : ℕ := 8
def number_of_tickets : ℕ := 2
def change_received : ℕ := 9

-- Statement to prove
theorem amount_brought_by_sisters :
  (cost_per_ticket * number_of_tickets + change_received) = 25 :=
by
  -- Using assumptions directly
  let total_cost := cost_per_ticket * number_of_tickets
  have total_cost_eq : total_cost = 16 := by sorry
  let amount_brought := total_cost + change_received
  have amount_brought_eq : amount_brought = 25 := by sorry
  exact amount_brought_eq

end amount_brought_by_sisters_l211_211643


namespace number_of_white_balls_l211_211719

-- Definition of the conditions
def total_balls : ℕ := 40
def prob_red : ℝ := 0.15
def prob_black : ℝ := 0.45
def prob_white := 1 - prob_red - prob_black

-- The statement that needs to be proved
theorem number_of_white_balls : (total_balls : ℝ) * prob_white = 16 :=
by
  sorry

end number_of_white_balls_l211_211719


namespace find_a2023_l211_211730

theorem find_a2023 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a n + a (n + 1) = n) : a 2023 = 1012 :=
sorry

end find_a2023_l211_211730


namespace remainder_expression_div_10_l211_211902

theorem remainder_expression_div_10 (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^p + t + 11^t * 6^(p * t)) % 10 = 1 :=
by
  sorry

end remainder_expression_div_10_l211_211902


namespace minimum_value_l211_211560

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end minimum_value_l211_211560


namespace evaluate_expression_at_three_l211_211663

theorem evaluate_expression_at_three : 
  (3^2 + 3 * (3^6) = 2196) :=
by
  sorry -- This is where the proof would go

end evaluate_expression_at_three_l211_211663


namespace kathleen_allowance_l211_211627

theorem kathleen_allowance (x : ℝ) (h1 : Kathleen_middleschool_allowance = x + 2)
(h2 : Kathleen_senior_allowance = 5 + 2 * (x + 2))
(h3 : Kathleen_senior_allowance = 2.5 * Kathleen_middleschool_allowance) :
x = 8 :=
by sorry

end kathleen_allowance_l211_211627


namespace necessary_but_not_sufficient_l211_211180

def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b < 0

theorem necessary_but_not_sufficient (a b c : ℝ) (p : a * b < 0) (q : is_hyperbola a b c) :
  (∀ (a b c : ℝ), is_hyperbola a b c → a * b < 0) ∧ (¬ ∀ (a b c : ℝ), a * b < 0 → is_hyperbola a b c) :=
by
  sorry

end necessary_but_not_sufficient_l211_211180


namespace find_other_endpoint_l211_211880

theorem find_other_endpoint (x1 y1 x2 y2 xm ym : ℝ)
  (midpoint_formula_x : xm = (x1 + x2) / 2)
  (midpoint_formula_y : ym = (y1 + y2) / 2)
  (h_midpoint : xm = -3 ∧ ym = 2)
  (h_endpoint : x1 = -7 ∧ y1 = 6) :
  x2 = 1 ∧ y2 = -2 := 
sorry

end find_other_endpoint_l211_211880


namespace marion_score_correct_l211_211008

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end marion_score_correct_l211_211008


namespace final_apples_count_l211_211904

-- Define the initial conditions
def initial_apples : Nat := 128

def percent_25 (n : Nat) : Nat := n * 25 / 100

def apples_after_selling_to_jill (n : Nat) : Nat := n - percent_25 n

def apples_after_selling_to_june (n : Nat) : Nat := apples_after_selling_to_jill n - percent_25 (apples_after_selling_to_jill n)

def apples_after_giving_to_teacher (n : Nat) : Nat := apples_after_selling_to_june n - 1

-- The theorem stating the problem to be proved
theorem final_apples_count : apples_after_giving_to_teacher initial_apples = 71 := by
  sorry

end final_apples_count_l211_211904


namespace winnie_keeps_balloons_l211_211287

theorem winnie_keeps_balloons : 
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  remainder = 4 :=
by
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  show remainder = 4
  sorry

end winnie_keeps_balloons_l211_211287


namespace min_cylinder_volume_eq_surface_area_l211_211246

theorem min_cylinder_volume_eq_surface_area (r h V S : ℝ) (hr : r > 0) (hh : h > 0)
  (hV : V = π * r^2 * h) (hS : S = 2 * π * r^2 + 2 * π * r * h) (heq : V = S) :
  V = 54 * π :=
by
  -- Placeholder for the actual proof
  sorry

end min_cylinder_volume_eq_surface_area_l211_211246


namespace find_xyz_squares_l211_211058

theorem find_xyz_squares (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end find_xyz_squares_l211_211058


namespace find_original_price_of_dish_l211_211430

noncomputable def original_price_of_dish (P : ℝ) : Prop :=
  let john_paid := (0.9 * P) + (0.15 * P)
  let jane_paid := (0.9 * P) + (0.135 * P)
  john_paid = jane_paid + 0.60 → P = 40

theorem find_original_price_of_dish (P : ℝ) (h : original_price_of_dish P) : P = 40 := by
  sorry

end find_original_price_of_dish_l211_211430


namespace percentage_decrease_l211_211390

theorem percentage_decrease (x y : ℝ) :
  let x' := 0.8 * x
  let y' := 0.7 * y
  let original_expr := x^2 * y^3
  let new_expr := (x')^2 * (y')^3
  let perc_decrease := (original_expr - new_expr) / original_expr * 100
  perc_decrease = 78.048 := by
  sorry

end percentage_decrease_l211_211390


namespace geometric_mean_problem_l211_211040

theorem geometric_mean_problem
  (a : Nat) (a1 : Nat) (a8 : Nat) (r : Rat) 
  (h1 : a1 = 6) (h2 : a8 = 186624) 
  (h3 : a8 = a1 * r^7) 
  : a = a1 * r^3 → a = 1296 := 
by
  sorry

end geometric_mean_problem_l211_211040


namespace division_of_powers_l211_211063

theorem division_of_powers :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 :=
by sorry

end division_of_powers_l211_211063


namespace n_minus_m_l211_211695

variable (m n : ℕ)

def is_congruent_to_5_mod_13 (x : ℕ) : Prop := x % 13 = 5
def is_smallest_three_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 100 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 100 → x ≤ y

def is_smallest_four_digit_integer_congruent_to_5_mod_13 (x : ℕ) : Prop :=
  is_congruent_to_5_mod_13 x ∧ x ≥ 1000 ∧ ∀ y, is_congruent_to_5_mod_13 y → y ≥ 1000 → x ≤ y

theorem n_minus_m
  (h₁ : is_smallest_three_digit_integer_congruent_to_5_mod_13 m)
  (h₂ : is_smallest_four_digit_integer_congruent_to_5_mod_13 n) :
  n - m = 897 := sorry

end n_minus_m_l211_211695


namespace solutionTriangle_l211_211235

noncomputable def solveTriangle (a b : ℝ) (B : ℝ) : (ℝ × ℝ × ℝ) :=
  let A := 30
  let C := 30
  let c := 2
  (A, C, c)

theorem solutionTriangle :
  solveTriangle 2 (2 * Real.sqrt 3) 120 = (30, 30, 2) :=
by
  sorry

end solutionTriangle_l211_211235


namespace cricket_initial_overs_l211_211960

-- Definitions based on conditions
def run_rate_initial : ℝ := 3.2
def run_rate_remaining : ℝ := 12.5
def target_runs : ℝ := 282
def remaining_overs : ℕ := 20

-- Mathematical statement to prove
theorem cricket_initial_overs (x : ℝ) (y : ℝ)
    (h1 : y = run_rate_initial * x)
    (h2 : y + run_rate_remaining * remaining_overs = target_runs) :
    x = 10 :=
sorry

end cricket_initial_overs_l211_211960


namespace union_condition_implies_l211_211347

-- Define set A as per the given condition
def setA : Set ℝ := { x | x * (x - 1) ≤ 0 }

-- Define set B as per the given condition with parameter a
def setB (a : ℝ) : Set ℝ := { x | Real.log x ≤ a }

-- Given condition A ∪ B = A, we need to prove that a ≤ 0
theorem union_condition_implies (a : ℝ) (h : setA ∪ setB a = setA) : a ≤ 0 := 
by
  sorry

end union_condition_implies_l211_211347


namespace population_doubling_time_l211_211452

open Real

noncomputable def net_growth_rate (birth_rate : ℝ) (death_rate : ℝ) : ℝ :=
birth_rate - death_rate

noncomputable def percentage_growth_rate (net_growth_rate : ℝ) (population_base : ℝ) : ℝ :=
(net_growth_rate / population_base) * 100

noncomputable def doubling_time (percentage_growth_rate : ℝ) : ℝ :=
70 / percentage_growth_rate

theorem population_doubling_time :
    let birth_rate := 39.4
    let death_rate := 19.4
    let population_base := 1000
    let net_growth := net_growth_rate birth_rate death_rate
    let percentage_growth := percentage_growth_rate net_growth population_base
    doubling_time percentage_growth = 35 := 
by
    sorry

end population_doubling_time_l211_211452


namespace pie_eating_contest_l211_211360

theorem pie_eating_contest :
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  max (max first_student second_student) third_student - 
  min (min first_student second_student) third_student = 1 / 6 :=
by
  let first_student := (5 : ℚ) / 6
  let second_student := (2 : ℚ) / 3
  let third_student := (3 : ℚ) / 4
  sorry

end pie_eating_contest_l211_211360


namespace gabby_additional_money_needed_l211_211375

theorem gabby_additional_money_needed
  (cost_makeup : ℕ := 65)
  (cost_skincare : ℕ := 45)
  (cost_hair_tool : ℕ := 55)
  (initial_savings : ℕ := 35)
  (money_from_mom : ℕ := 20)
  (money_from_dad : ℕ := 30)
  (money_from_chores : ℕ := 25) :
  (cost_makeup + cost_skincare + cost_hair_tool) - (initial_savings + money_from_mom + money_from_dad + money_from_chores) = 55 := 
by
  sorry

end gabby_additional_money_needed_l211_211375


namespace sum_three_consecutive_odd_integers_l211_211104

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end sum_three_consecutive_odd_integers_l211_211104


namespace q_r_share_difference_l211_211552

theorem q_r_share_difference
  (T : ℝ) -- Total amount of money
  (x : ℝ) -- Common multiple of shares
  (p_share q_share r_share s_share : ℝ) -- Shares before tax
  (p_tax q_tax r_tax s_tax : ℝ) -- Tax percentages
  (h_ratio : p_share = 3 * x ∧ q_share = 7 * x ∧ r_share = 12 * x ∧ s_share = 5 * x) -- Ratio condition
  (h_tax : p_tax = 0.10 ∧ q_tax = 0.15 ∧ r_tax = 0.20 ∧ s_tax = 0.25) -- Tax condition
  (h_difference_pq : q_share * (1 - q_tax) - p_share * (1 - p_tax) = 2400) -- Difference between p and q after tax
  : (r_share * (1 - r_tax) - q_share * (1 - q_tax)) = 2695.38 := sorry

end q_r_share_difference_l211_211552


namespace product_of_numbers_l211_211490

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43.05 := by
  sorry

end product_of_numbers_l211_211490


namespace problem_statement_l211_211979

-- Define the function f1 as the square of the sum of the digits of k
def f1 (k : Nat) : Nat :=
  let sum_digits := (Nat.digits 10 k).sum
  sum_digits * sum_digits

-- Define the recursive function f_{n+1}(k) = f1(f_n(k))
def fn : Nat → Nat → Nat
| 0, k => k
| n+1, k => f1 (fn n k)

theorem problem_statement : fn 1991 (2^1990) = 256 :=
sorry

end problem_statement_l211_211979


namespace larger_square_area_multiple_l211_211276

theorem larger_square_area_multiple (a b : ℕ) (h : a = 4 * b) :
  (a ^ 2) = 16 * (b ^ 2) :=
sorry

end larger_square_area_multiple_l211_211276


namespace books_in_either_but_not_both_l211_211263

theorem books_in_either_but_not_both (shared_books alice_books bob_unique_books : ℕ) 
    (h1 : shared_books = 12) 
    (h2 : alice_books = 26)
    (h3 : bob_unique_books = 8) : 
    (alice_books - shared_books) + bob_unique_books = 22 :=
by
  sorry

end books_in_either_but_not_both_l211_211263


namespace find_f_2008_l211_211681

noncomputable def f (x : ℝ) : ℝ := Real.cos x

noncomputable def f_n (n : ℕ) : (ℝ → ℝ) :=
match n with
| 0     => f
| (n+1) => (deriv (f_n n))

theorem find_f_2008 (x : ℝ) : (f_n 2008) x = Real.cos x := by
  sorry

end find_f_2008_l211_211681


namespace func_g_neither_even_nor_odd_l211_211669

noncomputable def func_g (x : ℝ) : ℝ := (⌈x⌉ : ℝ) - (1 / 3)

theorem func_g_neither_even_nor_odd :
  (¬ ∀ x, func_g (-x) = func_g x) ∧ (¬ ∀ x, func_g (-x) = -func_g x) :=
by
  sorry

end func_g_neither_even_nor_odd_l211_211669


namespace sqrt_41_40_39_38_plus_1_l211_211755

theorem sqrt_41_40_39_38_plus_1 : Real.sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := by
  sorry

end sqrt_41_40_39_38_plus_1_l211_211755


namespace find_a7_in_arithmetic_sequence_l211_211025

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem find_a7_in_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3_a5 : a 3 + a 5 = 10) :
  a 7 = 8 :=
sorry

end find_a7_in_arithmetic_sequence_l211_211025


namespace total_money_spent_l211_211941

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end total_money_spent_l211_211941


namespace no_bounded_sequence_a1_gt_2015_l211_211411

theorem no_bounded_sequence_a1_gt_2015 (a1 : ℚ) (h_a1 : a1 > 2015) : 
  ∀ (a_n : ℕ → ℚ), a_n 1 = a1 → 
  (∀ (n : ℕ), ∃ (p_n q_n : ℕ), p_n > 0 ∧ q_n > 0 ∧ (p_n.gcd q_n = 1) ∧ (a_n n = p_n / q_n) ∧ 
  (a_n (n + 1) = (p_n^2 + 2015) / (p_n * q_n))) → 
  ∃ (M : ℚ), ∀ (n : ℕ), a_n n ≤ M → 
  False :=
sorry

end no_bounded_sequence_a1_gt_2015_l211_211411


namespace number_of_students_l211_211363

theorem number_of_students : 
    ∃ (n : ℕ), 
      (∃ (x : ℕ), 
        (∀ (k : ℕ), x = 4 * k ∧ 5 * x + 1 = n)
      ) ∧ 
      (∃ (y : ℕ), 
        (∀ (k : ℕ), y = 5 * k ∧ 4 * y + 1 = n)
      ) ∧
      n ≤ 30 ∧ 
      n = 21 :=
  sorry

end number_of_students_l211_211363


namespace value_of_expression_l211_211195

theorem value_of_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2) : (1 / 3) * x ^ 8 * y ^ 9 = 2 / 3 :=
by
  -- Proof can be filled in here
  sorry

end value_of_expression_l211_211195


namespace basketball_team_win_rate_l211_211346

theorem basketball_team_win_rate (won_first : ℕ) (total : ℕ) (remaining : ℕ)
    (desired_rate : ℚ) (x : ℕ) (H_won : won_first = 30) (H_total : total = 100)
    (H_remaining : remaining = 55) (H_desired : desired_rate = 13/20) :
    (30 + x) / 100 = 13 / 20 ↔ x = 35 := by
    sorry

end basketball_team_win_rate_l211_211346


namespace trapezoid_diagonals_l211_211545

theorem trapezoid_diagonals {BC AD AB CD AC BD : ℝ} (h b1 b2 : ℝ) 
  (hBC : BC = b1) (hAD : AD = b2) (hAB : AB = h) (hCD : CD = h) 
  (hAC : AC^2 = AB^2 + BC^2) (hBD : BD^2 = CD^2 + AD^2) :
  BD^2 - AC^2 = b2^2 - b1^2 := 
by 
  -- proof is omitted
  sorry

end trapezoid_diagonals_l211_211545


namespace solve_equation_l211_211068

theorem solve_equation (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2)
  (h₃ : (3 * x + 6)/(x^2 + 5 * x + 6) = (3 - x)/(x - 2)) :
  x = 3 ∨ x = -3 :=
sorry

end solve_equation_l211_211068


namespace range_of_x_when_a_equals_1_range_of_a_l211_211581

variable {a x : ℝ}

-- Definitions for conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part (1): Prove the range of x when a = 1 and p ∨ q is true.
theorem range_of_x_when_a_equals_1 (h : a = 1) (h1 : p 1 x ∨ q x) : 1 < x ∧ x < 3 :=
by sorry

-- Part (2): Prove the range of a when p is a necessary but not sufficient condition for q.
theorem range_of_a (h2 : ∀ x, q x → p a x) (h3 : ¬ ∀ x, p a x → q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_x_when_a_equals_1_range_of_a_l211_211581


namespace f_7_eq_neg3_l211_211277

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_interval  : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4

theorem f_7_eq_neg3 : f 7 = -3 :=
  sorry

end f_7_eq_neg3_l211_211277


namespace jars_needed_l211_211531

def hives : ℕ := 5
def honey_per_hive : ℕ := 20
def jar_capacity : ℝ := 0.5
def friend_ratio : ℝ := 0.5

theorem jars_needed : (hives * honey_per_hive) / 2 / jar_capacity = 100 := 
by sorry

end jars_needed_l211_211531


namespace minimum_odd_correct_answers_l211_211221

theorem minimum_odd_correct_answers (students : Fin 50 → Fin 5) :
  (∀ S : Finset (Fin 50), S.card = 40 → 
    (∃ x ∈ S, students x = 3) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, students x₁ = 2 ∧ x₁ ≠ x₂ ∧ students x₂ = 2) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, students x₁ = 1 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ students x₂ = 1 ∧ students x₃ = 1) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, ∃ x₄ ∈ S, students x₁ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ students x₂ = 0 ∧ students x₃ = 0 ∧ students x₄ = 0)) →
  (∃ S : Finset (Fin 50), (∀ x ∈ S, (students x = 1 ∨ students x = 3)) ∧ S.card = 23) :=
by
  sorry

end minimum_odd_correct_answers_l211_211221


namespace sufficient_but_not_necessary_condition_l211_211766

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a - Real.sin x

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, f' a x > 0 → (a > 1)) ∧ (¬∀ x, f' a x ≥ 0 → (a > 1)) := sorry

end sufficient_but_not_necessary_condition_l211_211766


namespace determine_initial_sum_l211_211048

def initial_sum_of_money (P r : ℝ) : Prop :=
  (600 = P + 2 * P * r) ∧ (700 = P + 2 * P * (r + 0.1))

theorem determine_initial_sum (P r : ℝ) (h : initial_sum_of_money P r) : P = 500 :=
by
  cases h with
  | intro h1 h2 =>
    sorry

end determine_initial_sum_l211_211048


namespace gcd_of_12347_and_9876_l211_211011

theorem gcd_of_12347_and_9876 : Nat.gcd 12347 9876 = 7 :=
by
  sorry

end gcd_of_12347_and_9876_l211_211011


namespace opposite_numbers_power_l211_211207

theorem opposite_numbers_power (a b : ℝ) (h : a + b = 0) : (a + b) ^ 2023 = 0 :=
by 
  sorry

end opposite_numbers_power_l211_211207


namespace arithmetic_equation_false_l211_211698

theorem arithmetic_equation_false :
  4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 := by
  sorry

end arithmetic_equation_false_l211_211698


namespace p_is_necessary_not_sufficient_for_q_l211_211651

  variable (x : ℝ)

  def p := |x| ≤ 2
  def q := 0 ≤ x ∧ x ≤ 2

  theorem p_is_necessary_not_sufficient_for_q : (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) :=
  by
    sorry
  
end p_is_necessary_not_sufficient_for_q_l211_211651


namespace find_total_income_l211_211525

theorem find_total_income (I : ℝ) (H : (0.27 * I = 35000)) : I = 129629.63 :=
by
  sorry

end find_total_income_l211_211525


namespace number_of_elements_in_S_l211_211540

def S : Set ℕ := { n : ℕ | ∃ k : ℕ, n > 1 ∧ (10^10 - 1) % n = 0 }

theorem number_of_elements_in_S (h1 : Nat.Prime 9091) :
  ∃ T : Finset ℕ, T.card = 127 ∧ ∀ n, n ∈ T ↔ n ∈ S :=
sorry

end number_of_elements_in_S_l211_211540


namespace pairs_with_green_shirts_l211_211165

theorem pairs_with_green_shirts (r g t p rr_pairs gg_pairs : ℕ)
  (h1 : r = 60)
  (h2 : g = 90)
  (h3 : t = 150)
  (h4 : p = 75)
  (h5 : rr_pairs = 28)
  : gg_pairs = 43 := 
sorry

end pairs_with_green_shirts_l211_211165


namespace num_triangles_from_decagon_l211_211555

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l211_211555


namespace sequence_1001st_term_l211_211324

theorem sequence_1001st_term (a b : ℤ) (h1 : b = 2 * a - 3) : 
  ∃ n : ℤ, n = 1001 → (a + 1000 * (20 * a - 30)) = 30003 := 
by 
  sorry

end sequence_1001st_term_l211_211324


namespace solve_system_of_inequalities_l211_211171

theorem solve_system_of_inequalities (x : ℝ) 
  (h1 : -3 * x^2 + 7 * x + 6 > 0) 
  (h2 : 4 * x - 4 * x^2 > -3) : 
  -1/2 < x ∧ x < 3/2 :=
sorry

end solve_system_of_inequalities_l211_211171


namespace smallest_integer_solution_l211_211795

theorem smallest_integer_solution (n : ℤ) (h : n^3 - 12 * n^2 + 44 * n - 48 ≤ 0) : n = 2 :=
sorry

end smallest_integer_solution_l211_211795


namespace total_reading_materials_l211_211275

theorem total_reading_materials 
  (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (newspapers_per_shelf : ℕ) (graphic_novels_per_shelf : ℕ) 
  (bookshelves : ℕ)
  (h_books : books_per_shelf = 23) 
  (h_magazines : magazines_per_shelf = 61) 
  (h_newspapers : newspapers_per_shelf = 17) 
  (h_graphic_novels : graphic_novels_per_shelf = 29) 
  (h_bookshelves : bookshelves = 37) : 
  (books_per_shelf * bookshelves + magazines_per_shelf * bookshelves + newspapers_per_shelf * bookshelves + graphic_novels_per_shelf * bookshelves) = 4810 := 
by {
  -- Condition definitions are already given; the proof is omitted here.
  sorry
}

end total_reading_materials_l211_211275


namespace people_per_entrance_l211_211478

theorem people_per_entrance (e p : ℕ) (h1 : e = 5) (h2 : p = 1415) : p / e = 283 := by
  sorry

end people_per_entrance_l211_211478


namespace turnips_bag_l211_211103

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end turnips_bag_l211_211103


namespace equilateral_triangle_of_condition_l211_211065

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 0) :
  a = b ∧ b = c := 
sorry

end equilateral_triangle_of_condition_l211_211065


namespace compare_logs_l211_211072

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

theorem compare_logs : b > c ∧ c > a :=
by
  sorry

end compare_logs_l211_211072


namespace Trent_tears_l211_211757

def onions_per_pot := 4
def pots_of_soup := 6
def tears_per_3_onions := 2

theorem Trent_tears:
  (onions_per_pot * pots_of_soup) / 3 * tears_per_3_onions = 16 :=
by
  sorry

end Trent_tears_l211_211757


namespace equation_of_line_l_l211_211463

def point (P : ℝ × ℝ) := P = (2, 1)
def parallel (x y : ℝ) : Prop := 2 * x - y + 2 = 0

theorem equation_of_line_l (c : ℝ) (x y : ℝ) :
  (parallel x y ∧ point (x, y)) →
  2 * x - y + c = 0 →
  c = -3 → 2 * x - y - 3 = 0 :=
by
  intro h1 h2 h3
  sorry

end equation_of_line_l_l211_211463


namespace solutions_of_system_l211_211820

theorem solutions_of_system :
  ∀ (x y : ℝ), (x - 2 * y = 1) ∧ (x^3 - 8 * y^3 - 6 * x * y = 1) ↔ y = (x - 1) / 2 :=
by
  -- Since this is a statement-only task, the detailed proof is omitted.
  -- Insert actual proof here.
  sorry

end solutions_of_system_l211_211820


namespace a3_value_l211_211381

variable {a : ℕ → ℤ} -- Arithmetic sequence as a function from natural numbers to integers
variable {S : ℕ → ℤ} -- Sum of the first n terms

-- Conditions
axiom a1_eq : a 1 = -11
axiom a4_plus_a6_eq : a 4 + a 6 = -6
-- Common difference d
variable {d : ℤ}
axiom d_def : ∀ n, a (n + 1) = a n + d

theorem a3_value : a 3 = -7 := by
  sorry -- Proof not required as per the instructions

end a3_value_l211_211381


namespace potion_combinations_l211_211657

-- Definitions of conditions
def roots : Nat := 3
def minerals : Nat := 5
def incompatible_combinations : Nat := 2

-- Statement of the problem
theorem potion_combinations : (roots * minerals) - incompatible_combinations = 13 := by
  sorry

end potion_combinations_l211_211657


namespace approx_ineq_l211_211743

noncomputable def approx (x : ℝ) : ℝ := 1 + 6 * (-0.002 : ℝ)

theorem approx_ineq (x : ℝ) (h : x = 0.998) : 
  abs ((x^6) - approx x) < 0.001 :=
by
  sorry

end approx_ineq_l211_211743


namespace no_real_m_for_parallel_lines_l211_211412

theorem no_real_m_for_parallel_lines : 
  ∀ (m : ℝ), ∃ (l1 l2 : ℝ × ℝ × ℝ), 
  (l1 = (2, (m + 1), 4)) ∧ (l2 = (m, 3, 4)) ∧ 
  ( ∀ (m : ℝ), -2 / (m + 1) = -m / 3 → false ) :=
by sorry

end no_real_m_for_parallel_lines_l211_211412


namespace jerry_apples_l211_211258

theorem jerry_apples (J : ℕ) (h1 : 20 + 60 + J = 3 * 2 * 20):
  J = 40 :=
sorry

end jerry_apples_l211_211258


namespace negation_of_universal_l211_211597

theorem negation_of_universal (h : ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) : ∃ x : ℝ, x^2 + 2 * x + 5 = 0 :=
sorry

end negation_of_universal_l211_211597


namespace earnings_difference_l211_211492

-- Define the prices and quantities.
def price_A : ℝ := 4
def price_B : ℝ := 3.5
def quantity_A : ℕ := 300
def quantity_B : ℕ := 350

-- Define the earnings for both companies.
def earnings_A := price_A * quantity_A
def earnings_B := price_B * quantity_B

-- State the theorem we intend to prove.
theorem earnings_difference :
  earnings_B - earnings_A = 25 := by
  sorry

end earnings_difference_l211_211492


namespace parallel_lines_not_coincident_l211_211739

theorem parallel_lines_not_coincident (x y : ℝ) (m : ℝ) :
  (∀ y, x + (1 + m) * y = 2 - m ∧ ∀ y, m * x + 2 * y + 8 = 0) → (m =1) := 
sorry

end parallel_lines_not_coincident_l211_211739


namespace worker_savings_multiple_l211_211987

theorem worker_savings_multiple 
  (P : ℝ)
  (P_gt_zero : P > 0)
  (save_fraction : ℝ := 1/3)
  (not_saved_fraction : ℝ := 2/3)
  (total_saved : ℝ := 12 * (save_fraction * P)) :
  ∃ multiple : ℝ, total_saved = multiple * (not_saved_fraction * P) ∧ multiple = 6 := 
by 
  sorry

end worker_savings_multiple_l211_211987


namespace common_point_geometric_progression_passing_l211_211714

theorem common_point_geometric_progression_passing
  (a b c : ℝ) (r : ℝ) (h_b : b = a * r) (h_c : c = a * r^2) :
  ∃ x y : ℝ, (∀ a ≠ 0, a * x + (a * r) * y = a * r^2) → (x = 0 ∧ y = 1) :=
by
  sorry

end common_point_geometric_progression_passing_l211_211714


namespace stuffed_animals_mom_gift_l211_211998

theorem stuffed_animals_mom_gift (x : ℕ) :
  (10 + x) + 3 * (10 + x) = 48 → x = 2 :=
by {
  sorry
}

end stuffed_animals_mom_gift_l211_211998


namespace find_tangent_line_l211_211017

def curve := fun x : ℝ => x^3 + 2 * x + 1
def tangent_point := 1
def tangent_line (x y : ℝ) := 5 * x - y - 1 = 0

theorem find_tangent_line :
  tangent_line tangent_point (curve tangent_point) :=
by
  sorry

end find_tangent_line_l211_211017


namespace total_students_correct_l211_211309

-- Define the number of students who play football, cricket, both and neither.
def play_football : ℕ := 325
def play_cricket : ℕ := 175
def play_both : ℕ := 90
def play_neither : ℕ := 50

-- Define the total number of students
def total_students : ℕ := play_football + play_cricket - play_both + play_neither

-- Prove that the total number of students is 460 given the conditions
theorem total_students_correct : total_students = 460 := by
  sorry

end total_students_correct_l211_211309


namespace negation_statement_l211_211200

variable {α : Type} (teacher generous : α → Prop)

theorem negation_statement :
  ¬ ∀ x, teacher x → generous x ↔ ∃ x, teacher x ∧ ¬ generous x := by
sorry

end negation_statement_l211_211200


namespace range_g_l211_211456

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x + (Real.cos x)^2 - 1/2

noncomputable def g (x : ℝ) : ℝ := 
  let h (x : ℝ) := (Real.sin (2 * x + Real.pi))
  h (x - (5 * Real.pi / 12))

theorem range_g :
  (Set.image g (Set.Icc (-Real.pi/12) (Real.pi/3))) = Set.Icc (-1) (1/2) :=
  sorry

end range_g_l211_211456


namespace deceased_member_income_l211_211589

theorem deceased_member_income (A B C : ℝ) (h1 : (A + B + C) / 3 = 735) (h2 : (A + B) / 2 = 650) : 
  C = 905 :=
by
  sorry

end deceased_member_income_l211_211589


namespace distribution_problem_l211_211876

theorem distribution_problem (cards friends : ℕ) (h1 : cards = 7) (h2 : friends = 9) :
  (Nat.choose friends cards) * (Nat.factorial cards) = 181440 :=
by
  -- According to the combination formula and factorial definition
  -- We can insert specific values and calculations here, but as per the task requirements, 
  -- we are skipping the actual proof.
  sorry

end distribution_problem_l211_211876


namespace service_fee_calculation_l211_211107

-- Problem definitions based on conditions
def cost_food : ℝ := 50
def tip : ℝ := 5
def total_spent : ℝ := 61
def service_fee_percentage (x : ℝ) : Prop := x = (12 / 50) * 100

-- The main statement to be proven, showing that the service fee percentage is 24%
theorem service_fee_calculation : service_fee_percentage 24 :=
by {
  sorry
}

end service_fee_calculation_l211_211107


namespace car_cost_l211_211416

theorem car_cost (days_in_week : ℕ) (sue_days : ℕ) (sister_days : ℕ) 
  (sue_payment : ℕ) (car_cost : ℕ) 
  (h1 : days_in_week = 7)
  (h2 : sue_days = days_in_week - sister_days)
  (h3 : sister_days = 4)
  (h4 : sue_payment = 900)
  (h5 : sue_payment * days_in_week = sue_days * car_cost) :
  car_cost = 2100 := 
by {
  sorry
}

end car_cost_l211_211416


namespace cone_base_circumference_l211_211050

theorem cone_base_circumference (radius : ℝ) (angle : ℝ) (c_base : ℝ) :
  radius = 6 ∧ angle = 180 ∧ c_base = 6 * Real.pi →
  (c_base = (angle / 360) * (2 * Real.pi * radius)) :=
by
  intros h
  rcases h with ⟨h_radius, h_angle, h_c_base⟩
  rw [h_radius, h_angle]
  norm_num
  sorry

end cone_base_circumference_l211_211050


namespace unique_solution_of_system_l211_211248

noncomputable def solve_system_of_equations (x1 x2 x3 x4 x5 x6 x7 : ℝ) : Prop :=
  10 * x1 + 3 * x2 + 4 * x3 + x4 + x5 = 0 ∧
  11 * x2 + 2 * x3 + 2 * x4 + 3 * x5 + x6 = 0 ∧
  15 * x3 + 4 * x4 + 5 * x5 + 4 * x6 + x7 = 0 ∧
  2 * x1 + x2 - 3 * x3 + 12 * x4 - 3 * x5 + x6 + x7 = 0 ∧
  6 * x1 - 5 * x2 + 3 * x3 - x4 + 17 * x5 + x6 = 0 ∧
  3 * x1 + 2 * x2 - 3 * x3 + 4 * x4 + x5 - 16 * x6 + 2 * x7 = 0 ∧
  4 * x1 - 8 * x2 + x3 + x4 - 3 * x5 + 19 * x7 = 0

theorem unique_solution_of_system :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℝ),
    solve_system_of_equations x1 x2 x3 x4 x5 x6 x7 →
    x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0 ∧ x6 = 0 ∧ x7 = 0 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h
  sorry

end unique_solution_of_system_l211_211248


namespace quadratic_solution_symmetry_l211_211830

variable (a b c n : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : a * (-5)^2 + b * (-5) + c = -2.79)
variable (h₂ : a * 1^2 + b * 1 + c = -2.79)
variable (h₃ : a * 2^2 + b * 2 + c = 0)
variable (h₄ : a * 3^2 + b * 3 + c = n)

theorem quadratic_solution_symmetry :
  (x = 3 ∨ x = -7) ↔ (a * x^2 + b * x + c = n) :=
sorry

end quadratic_solution_symmetry_l211_211830


namespace roots_of_polynomial_l211_211934

noncomputable def P (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 - 6 * x

theorem roots_of_polynomial : ∀ x : ℝ, P x = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by 
  -- Here you would provide the proof, but we use sorry to indicate it is left out
  sorry

end roots_of_polynomial_l211_211934


namespace fraction_simplification_l211_211161

-- We define the given fractions
def a := 3 / 7
def b := 2 / 9
def c := 5 / 12
def d := 1 / 4

-- We state the main theorem
theorem fraction_simplification : (a - b) / (c + d) = 13 / 42 := by
  -- Skipping proof for the equivalence problem
  sorry

end fraction_simplification_l211_211161


namespace wrongly_written_height_is_176_l211_211666

-- Definitions and given conditions
def average_height_incorrect := 182
def average_height_correct := 180
def num_boys := 35
def actual_height := 106

-- The difference in total height due to the error
def total_height_incorrect := num_boys * average_height_incorrect
def total_height_correct := num_boys * average_height_correct
def height_difference := total_height_incorrect - total_height_correct

-- The wrongly written height
def wrongly_written_height := actual_height + height_difference

-- Proof statement
theorem wrongly_written_height_is_176 : wrongly_written_height = 176 := by
  sorry

end wrongly_written_height_is_176_l211_211666


namespace truck_left_1_hour_later_l211_211198

theorem truck_left_1_hour_later (v_car v_truck : ℝ) (time_to_pass : ℝ) : 
  v_car = 55 ∧ v_truck = 65 ∧ time_to_pass = 6.5 → 
  1 = time_to_pass - (time_to_pass * (v_car / v_truck)) := 
by
  intros h
  sorry

end truck_left_1_hour_later_l211_211198


namespace range_of_m_l211_211749

theorem range_of_m (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)
  (h_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (-2 * m * x + Real.log x + 3)) :
  ∃ m, m ∈ Set.Icc (1 / (2 * Real.exp 1)) (1 + Real.log 3 / 6) :=
sorry

end range_of_m_l211_211749


namespace find_a_l211_211943

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end find_a_l211_211943


namespace value_of_m_plus_n_l211_211602

noncomputable def exponential_function (a x m n : ℝ) : ℝ :=
  a^(x - m) + n - 3

theorem value_of_m_plus_n (a x m n y : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : exponential_function a 3 m n = 2) : m + n = 7 :=
by
  sorry

end value_of_m_plus_n_l211_211602


namespace line_within_plane_l211_211765

variable (a : Set Point) (α : Set Point)

theorem line_within_plane : a ⊆ α :=
by
  sorry

end line_within_plane_l211_211765


namespace inequality_solution_set_l211_211772

theorem inequality_solution_set : 
  { x : ℝ | (1 - x) * (x + 1) ≤ 0 ∧ x ≠ -1 } = { x : ℝ | x < -1 ∨ x ≥ 1 } :=
sorry

end inequality_solution_set_l211_211772


namespace Anna_s_wear_size_l211_211746

theorem Anna_s_wear_size
  (A : ℕ)
  (Becky_size : ℕ)
  (Ginger_size : ℕ)
  (h1 : Becky_size = 3 * A)
  (h2 : Ginger_size = 2 * Becky_size - 4)
  (h3 : Ginger_size = 8) :
  A = 2 :=
by
  sorry

end Anna_s_wear_size_l211_211746


namespace integer_xyz_zero_l211_211417

theorem integer_xyz_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_xyz_zero_l211_211417


namespace min_value_fraction_l211_211909

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 1) : 
  (1 / x) + (1 / (3 * y)) ≥ 3 :=
sorry

end min_value_fraction_l211_211909


namespace probability_heads_l211_211511

theorem probability_heads (p : ℝ) (q : ℝ) (C_10_5 C_10_6 : ℝ)
  (h1 : q = 1 - p)
  (h2 : C_10_5 = 252)
  (h3 : C_10_6 = 210)
  (eqn : C_10_5 * p^5 * q^5 = C_10_6 * p^6 * q^4) :
  p = 6 / 11 := 
by
  sorry

end probability_heads_l211_211511


namespace sqrt_180_eq_l211_211496

noncomputable def simplify_sqrt_180 : Real := 6 * Real.sqrt 5

theorem sqrt_180_eq : Real.sqrt 180 = simplify_sqrt_180 := 
by
  -- proof omitted
  sorry

end sqrt_180_eq_l211_211496


namespace probability_linda_picks_letter_in_mathematics_l211_211502

def english_alphabet : Finset Char := "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toList.toFinset

def word_mathematics : Finset Char := "MATHEMATICS".toList.toFinset

theorem probability_linda_picks_letter_in_mathematics : 
  (word_mathematics.card : ℚ) / (english_alphabet.card : ℚ) = 4 / 13 := by sorry

end probability_linda_picks_letter_in_mathematics_l211_211502


namespace solve_problem_statement_l211_211467

def problem_statement : Prop :=
  ∃ n, 3^19 % n = 7 ∧ n = 1162261460

theorem solve_problem_statement : problem_statement :=
  sorry

end solve_problem_statement_l211_211467


namespace ellipse_through_points_parabola_equation_l211_211009

-- Ellipse Problem: Prove the standard equation
theorem ellipse_through_points (m n : ℝ) (m_pos : m > 0) (n_pos : n > 0) (m_ne_n : m ≠ n) :
  (m * 0^2 + n * (5/3)^2 = 1) ∧ (m * 1^2 + n * 1^2 = 1) →
  (m = 16 / 25 ∧ n = 9 / 25) → (m * x^2 + n * y^2 = 1) ↔ (16 * x^2 + 9 * y^2 = 225) :=
sorry

-- Parabola Problem: Prove the equation
theorem parabola_equation (p x y : ℝ) (p_pos : p > 0)
  (dist_focus : abs (x + p / 2) = 10) (dist_axis : y^2 = 36) :
  (p = 2 ∨ p = 18) →
  (y^2 = 2 * p * x) ↔ (y^2 = 4 * x ∨ y^2 = 36 * x) :=
sorry

end ellipse_through_points_parabola_equation_l211_211009


namespace count_zero_expressions_l211_211677

/-- Given four specific vector expressions, prove that exactly two of them evaluate to the zero vector. --/
theorem count_zero_expressions
(AB BC CA MB BO OM AC BD CD OA OC CO : ℝ × ℝ)
(H1 : AB + BC + CA = 0)
(H2 : AB + (MB + BO + OM) ≠ 0)
(H3 : AB - AC + BD - CD = 0)
(H4 : OA + OC + BO + CO ≠ 0) :
  (∃ count, count = 2 ∧
      ((AB + BC + CA = 0) → count = count + 1) ∧
      ((AB + (MB + BO + OM) = 0) → count = count + 1) ∧
      ((AB - AC + BD - CD = 0) → count = count + 1) ∧
      ((OA + OC + BO + CO = 0) → count = count + 1)) :=
sorry

end count_zero_expressions_l211_211677


namespace solve_for_x_l211_211908

theorem solve_for_x (x : ℝ) (h : (x - 5)^3 = (1 / 27)⁻¹) : x = 8 :=
sorry

end solve_for_x_l211_211908


namespace calculate_expression_l211_211307

theorem calculate_expression : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end calculate_expression_l211_211307


namespace common_ratio_of_geometric_series_l211_211604

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l211_211604


namespace bianca_deleted_text_files_l211_211408

theorem bianca_deleted_text_files (pictures songs total : ℕ) (h₁ : pictures = 2) (h₂ : songs = 8) (h₃ : total = 17) :
  total - (pictures + songs) = 7 :=
by {
  sorry
}

end bianca_deleted_text_files_l211_211408


namespace linear_function_not_third_quadrant_l211_211673

theorem linear_function_not_third_quadrant (k : ℝ) (h1 : k ≠ 0) (h2 : k < 0) :
  ¬ (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ y = k * x + 1) :=
sorry

end linear_function_not_third_quadrant_l211_211673


namespace value_of_v_3_l211_211520

-- Defining the polynomial
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Given evaluation point
def eval_point : ℝ := -2

-- Horner's method intermediate value v_3
def v_3_using_horner_method (x : ℝ) : ℝ :=
  let V0 := 1
  let V1 := x * V0 - 5
  let V2 := x * V1 + 6
  let V3 := x * V2 -- x^3 term is zero
  V3

-- Statement to prove
theorem value_of_v_3 :
  v_3_using_horner_method eval_point = -40 :=
by 
  -- Proof to be completed later
  sorry

end value_of_v_3_l211_211520


namespace tangerines_left_proof_l211_211266

-- Define the number of tangerines Jimin ate
def tangerinesJiminAte : ℕ := 7

-- Define the total number of tangerines
def totalTangerines : ℕ := 12

-- Define the number of tangerines left
def tangerinesLeft : ℕ := totalTangerines - tangerinesJiminAte

-- Theorem stating the number of tangerines left equals 5
theorem tangerines_left_proof : tangerinesLeft = 5 := 
by
  sorry

end tangerines_left_proof_l211_211266


namespace problem_statement_l211_211202

noncomputable def g : ℝ → ℝ
| x => if x < 0 then -x
            else if x < 5 then x + 3
            else 2 * x ^ 2

theorem problem_statement : g (-6) + g 3 + g 8 = 140 :=
by
  -- Proof goes here
  sorry

end problem_statement_l211_211202


namespace cuboids_painted_l211_211850

-- Let's define the conditions first
def faces_per_cuboid : ℕ := 6
def total_faces_painted : ℕ := 36

-- Now, we state the theorem we want to prove
theorem cuboids_painted (n : ℕ) (h : total_faces_painted = n * faces_per_cuboid) : n = 6 :=
by
  -- Add proof here
  sorry

end cuboids_painted_l211_211850


namespace complex_exponential_sum_angle_l211_211920

theorem complex_exponential_sum_angle :
  ∃ r : ℝ, r ≥ 0 ∧ (e^(Complex.I * 11 * Real.pi / 60) + 
                     e^(Complex.I * 21 * Real.pi / 60) + 
                     e^(Complex.I * 31 * Real.pi / 60) + 
                     e^(Complex.I * 41 * Real.pi / 60) + 
                     e^(Complex.I * 51 * Real.pi / 60) = r * Complex.exp (Complex.I * 31 * Real.pi / 60)) := 
by
  sorry

end complex_exponential_sum_angle_l211_211920


namespace acute_triangle_probability_l211_211990

noncomputable def probability_acute_triangle : ℝ := sorry

theorem acute_triangle_probability :
  probability_acute_triangle = 1 / 4 := sorry

end acute_triangle_probability_l211_211990


namespace problem_l211_211846

def a := 1 / 4
def b := 1 / 2
def c := -3 / 4

def a_n (n : ℕ) : ℚ := 2 * n + 1
def S_n (n : ℕ) : ℚ := (n + 2) * n
def f (n : ℕ) : ℚ := 4 * a * n^2 + (4 * a + 2 * b) * n + (a + b + c)

theorem problem : ∀ n : ℕ, f n = S_n n := by
  sorry

end problem_l211_211846


namespace num_ways_express_2009_as_diff_of_squares_l211_211889

theorem num_ways_express_2009_as_diff_of_squares : 
  ∃ (n : Nat), n = 12 ∧ 
  ∃ (a b : Int), ∀ c, 2009 = a^2 - b^2 ∧ 
  (c = 1 ∨ c = -1) ∧ (2009 = (c * a)^2 - (c * b)^2) :=
sorry

end num_ways_express_2009_as_diff_of_squares_l211_211889


namespace abs_add_lt_abs_add_l211_211890

open Real

theorem abs_add_lt_abs_add {a b : ℝ} (h : a * b < 0) : abs (a + b) < abs a + abs b := 
  sorry

end abs_add_lt_abs_add_l211_211890


namespace negation_universal_proposition_l211_211550

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) → ∃ x : ℝ, x^2 - 2 * x + 1 < 0 :=
by sorry

end negation_universal_proposition_l211_211550


namespace domain_of_f_of_f_l211_211249

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (3 + x)

theorem domain_of_f_of_f :
  {x : ℝ | x ≠ -3 ∧ x ≠ -8 / 5} =
  {x : ℝ | ∃ y : ℝ, f x = y ∧ y ≠ -3 ∧ x ≠ -3} :=
by
  sorry

end domain_of_f_of_f_l211_211249


namespace find_smallest_n_l211_211302

theorem find_smallest_n 
    (a_n : ℕ → ℝ)
    (S_n : ℕ → ℝ)
    (h1 : a_n 1 + a_n 2 = 9 / 2)
    (h2 : S_n 4 = 45 / 8)
    (h3 : ∀ n, S_n n = (1 / 2) * n * (a_n 1 + a_n n)) :
    ∃ n : ℕ, a_n n < 1 / 10 ∧ ∀ m : ℕ, m < n → a_n m ≥ 1 / 10 := 
sorry

end find_smallest_n_l211_211302


namespace other_acute_angle_is_60_l211_211583

theorem other_acute_angle_is_60 (a b c : ℝ) (h_triangle : a + b + c = 180) (h_right : c = 90) (h_acute : a = 30) : b = 60 :=
by 
  -- inserting proof later
  sorry

end other_acute_angle_is_60_l211_211583


namespace incorrect_inequality_l211_211379

theorem incorrect_inequality : ¬ (-2 < -3) :=
by {
  -- Proof goes here
  sorry
}

end incorrect_inequality_l211_211379


namespace philips_painting_total_l211_211895

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end philips_painting_total_l211_211895


namespace geometric_sequence_third_term_l211_211124

theorem geometric_sequence_third_term :
  ∀ (a_1 a_5 : ℚ) (r : ℚ), 
    a_1 = 1 / 2 →
    (a_1 * r^4) = a_5 →
    a_5 = 16 →
    (a_1 * r^2) = 2 := 
by
  intros a_1 a_5 r h1 h2 h3
  sorry

end geometric_sequence_third_term_l211_211124


namespace num_supermarkets_in_US_l211_211260

theorem num_supermarkets_in_US (U C : ℕ) (h1 : U + C = 420) (h2 : U = C + 56) : U = 238 :=
by
  sorry

end num_supermarkets_in_US_l211_211260


namespace determine_m_l211_211219

theorem determine_m (m : ℝ) : (∀ x : ℝ, (0 < x ∧ x < 2) ↔ -1/2 * x^2 + 2 * x + m * x > 0) → m = -1 :=
by
  intro h
  sorry

end determine_m_l211_211219


namespace parallelogram_angle_B_l211_211338

theorem parallelogram_angle_B (A C B D : ℝ) (h₁ : A + C = 110) (h₂ : A = C) : B = 125 :=
by sorry

end parallelogram_angle_B_l211_211338


namespace sin_identity_l211_211915

theorem sin_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (60 * Real.pi / 180 + 2 * α) = 7 / 9 :=
by
  sorry

end sin_identity_l211_211915


namespace cevian_concurrency_l211_211870

theorem cevian_concurrency
  (A B C Z X Y : ℝ)
  (a b c s : ℝ)
  (h1 : s = (a + b + c) / 2)
  (h2 : AZ = s - c) (h3 : ZB = s - b)
  (h4 : BX = s - a) (h5 : XC = s - c)
  (h6 : CY = s - b) (h7 : YA = s - a)
  : (AZ / ZB) * (BX / XC) * (CY / YA) = 1 :=
by
  sorry

end cevian_concurrency_l211_211870


namespace correct_sum_of_integers_l211_211834

theorem correct_sum_of_integers (x y : ℕ) (h1 : x - y = 4) (h2 : x * y = 192) : x + y = 28 := by
  sorry

end correct_sum_of_integers_l211_211834


namespace alyssa_limes_picked_l211_211888

-- Definitions for the conditions
def total_limes : ℕ := 57
def mike_limes : ℕ := 32

-- The statement to be proved
theorem alyssa_limes_picked :
  ∃ (alyssa_limes : ℕ), total_limes - mike_limes = alyssa_limes ∧ alyssa_limes = 25 :=
by
  have alyssa_limes : ℕ := total_limes - mike_limes
  use alyssa_limes
  sorry

end alyssa_limes_picked_l211_211888


namespace jesse_money_left_l211_211907

def initial_money : ℝ := 500
def novel_cost_pounds : ℝ := 13
def num_novels : ℕ := 10
def bookstore_discount : ℝ := 0.20
def exchange_rate_usd_to_pounds : ℝ := 0.7
def lunch_cost_multiplier : ℝ := 3
def lunch_tax_rate : ℝ := 0.12
def lunch_tip_rate : ℝ := 0.18
def jacket_original_euros : ℝ := 120
def jacket_discount : ℝ := 0.30
def jacket_expense_multiplier : ℝ := 2
def exchange_rate_pounds_to_euros : ℝ := 1.15

theorem jesse_money_left : 
  initial_money - (
    ((novel_cost_pounds * num_novels * (1 - bookstore_discount)) / exchange_rate_usd_to_pounds)
    + ((novel_cost_pounds * lunch_cost_multiplier * (1 + lunch_tax_rate + lunch_tip_rate)) / exchange_rate_usd_to_pounds)
    + ((((jacket_original_euros * (1 - jacket_discount)) / exchange_rate_pounds_to_euros) / exchange_rate_usd_to_pounds))
  ) = 174.66 := by
  sorry

end jesse_money_left_l211_211907


namespace inverse_function_l211_211562

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

noncomputable def f_inv (y : ℝ) : ℝ := -2 - Real.sqrt ((1 + y) / 2)

theorem inverse_function :
  ∀ (x : ℝ), x < -2 → f_inv (f x) = x ∧ ∀ (y : ℝ), y > -1 → f (f_inv y) = y :=
by
  sorry

end inverse_function_l211_211562


namespace total_earthworms_in_box_l211_211331

-- Definitions of the conditions
def applesPaidByOkeydokey := 5
def applesPaidByArtichokey := 7
def earthwormsReceivedByOkeydokey := 25
def ratio := earthwormsReceivedByOkeydokey / applesPaidByOkeydokey -- which should be 5

-- Theorem statement proving the total number of earthworms in the box
theorem total_earthworms_in_box :
  (applesPaidByOkeydokey + applesPaidByArtichokey) * ratio = 60 :=
by
  sorry

end total_earthworms_in_box_l211_211331


namespace museum_ticket_cost_l211_211326

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l211_211326


namespace groom_dog_time_l211_211821

theorem groom_dog_time :
  ∃ (D : ℝ), (5 * D + 3 * 0.5 = 14) ∧ (D = 2.5) :=
by
  sorry

end groom_dog_time_l211_211821


namespace sin1993_cos1993_leq_zero_l211_211513

theorem sin1993_cos1993_leq_zero (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) : 
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := 
by 
  sorry

end sin1993_cos1993_leq_zero_l211_211513


namespace min_value_expression_l211_211489

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a + 2) * (1 / b + 2) ≥ 16 :=
sorry

end min_value_expression_l211_211489


namespace number_of_good_carrots_l211_211942

def total_carrots (nancy_picked : ℕ) (mom_picked : ℕ) : ℕ :=
  nancy_picked + mom_picked

def bad_carrots := 14

def good_carrots (total : ℕ) (bad : ℕ) : ℕ :=
  total - bad

theorem number_of_good_carrots :
  good_carrots (total_carrots 38 47) bad_carrots = 71 := by
  sorry

end number_of_good_carrots_l211_211942


namespace part1_part2_l211_211557

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^4 - 4 * x^3 + (3 + m) * x^2 - 12 * x + 12

theorem part1 (m : ℤ) : 
  (∀ x : ℝ, f x m - f (1 - x) m + 4 * x^3 = 0) ↔ (m = 8 ∨ m = 12) := 
sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x m ≥ 0) ↔ (4 ≤ m) := 
sorry

end part1_part2_l211_211557


namespace proof_mod_55_l211_211849

theorem proof_mod_55 (M : ℕ) (h1 : M % 5 = 3) (h2 : M % 11 = 9) : M % 55 = 53 := 
  sorry

end proof_mod_55_l211_211849


namespace age_difference_28_l211_211350

variable (li_lin_age_father_sum li_lin_age_future father_age_future : ℕ)

theorem age_difference_28 
    (h1 : li_lin_age_father_sum = 50)
    (h2 : ∀ x, li_lin_age_future = x → father_age_future = 3 * x - 2)
    (h3 : li_lin_age_future + 4 = li_lin_age_father_sum + 8 - (father_age_future + 4))
    : li_lin_age_father_sum - li_lin_age_future = 28 :=
sorry

end age_difference_28_l211_211350


namespace max_value_of_expression_l211_211205

open Classical
open Real

theorem max_value_of_expression (a b : ℝ) (c : ℝ) (h1 : a^2 + b^2 = c^2 + ab) (h2 : c = 1) :
  ∃ x : ℝ, x = (1 / 2) * b + a ∧ x = (sqrt 21) / 3 := 
sorry

end max_value_of_expression_l211_211205


namespace quadratic_inequality_solution_l211_211365

theorem quadratic_inequality_solution (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
sorry

end quadratic_inequality_solution_l211_211365


namespace find_degree_of_alpha_l211_211201

theorem find_degree_of_alpha
  (x : ℝ)
  (alpha : ℝ := x + 40)
  (beta : ℝ := 3 * x - 40)
  (h_parallel : alpha + beta = 180) :
  alpha = 85 :=
by
  sorry

end find_degree_of_alpha_l211_211201


namespace balloons_remaining_l211_211175

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end balloons_remaining_l211_211175


namespace cake_sharing_l211_211887

theorem cake_sharing (n : ℕ) (alex_share alena_share : ℝ) (h₁ : alex_share = 1 / 11) (h₂ : alena_share = 1 / 14) :
  12 ≤ n ∧ n ≤ 13 :=
by
  sorry

end cake_sharing_l211_211887


namespace common_ratio_l211_211682

-- Definitions for the geometric sequence
variables {a_n : ℕ → ℝ} {S_n q : ℝ}

-- Conditions provided in the problem
def condition1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  S_n 3 = a_n 1 + a_n 2 + a_n 3

def condition2 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2 + a_n 3) = a_n 4 - 2

def condition3 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2) = a_n 3 - 2

-- The theorem we want to prove
theorem common_ratio (a_n : ℕ → ℝ) (q : ℝ) :
  condition2 a_n S_n ∧ condition3 a_n S_n → q = 4 :=
by
  sorry

end common_ratio_l211_211682


namespace regression_equation_l211_211771

-- Define the regression coefficient and correlation
def negatively_correlated (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100

-- The question is to prove that given x and y are negatively correlated,
-- the regression equation is \hat{y} = -2x + 100
theorem regression_equation (x y : ℝ) (h : negatively_correlated x y) :
  (∃ a, a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100) → ∃ (b : ℝ), b = -2 ∧ ∀ (x_val : ℝ), y = b * x_val + 100 :=
by
  sorry

end regression_equation_l211_211771


namespace number_with_150_quarters_is_37_point_5_l211_211044

theorem number_with_150_quarters_is_37_point_5 (n : ℝ) (h : n / (1/4) = 150) : n = 37.5 := 
by 
  sorry

end number_with_150_quarters_is_37_point_5_l211_211044


namespace carl_max_value_carry_l211_211296

variables (rock_weight_3_pound : ℕ := 3) (rock_value_3_pound : ℕ := 9)
          (rock_weight_6_pound : ℕ := 6) (rock_value_6_pound : ℕ := 20)
          (rock_weight_2_pound : ℕ := 2) (rock_value_2_pound : ℕ := 5)
          (weight_limit : ℕ := 20)
          (max_six_pound_rocks : ℕ := 2)

noncomputable def max_value_carry : ℕ :=
  max (2 * rock_value_6_pound + 2 * rock_value_3_pound) 
      (4 * rock_value_3_pound + 4 * rock_value_2_pound)

theorem carl_max_value_carry : max_value_carry = 58 :=
by sorry

end carl_max_value_carry_l211_211296


namespace solve_inequality_l211_211735

theorem solve_inequality (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ x ∈ Set.Ioo (-2) (3) :=
sorry

end solve_inequality_l211_211735


namespace probability_of_digit_six_l211_211750

theorem probability_of_digit_six :
  let total_numbers := 90
  let favorable_numbers := 18
  0 < total_numbers ∧ 0 < favorable_numbers →
  (favorable_numbers / total_numbers : ℚ) = 1 / 5 :=
by
  intros total_numbers favorable_numbers h
  sorry

end probability_of_digit_six_l211_211750


namespace train_speed_l211_211094

def train_length : ℝ := 250
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 32

theorem train_speed :
  (train_length + bridge_length) / time_to_cross = 12.5 :=
by {
  sorry
}

end train_speed_l211_211094


namespace sum_in_base_b_l211_211962

noncomputable def s_in_base (b : ℕ) := 13 + 15 + 17

theorem sum_in_base_b (b : ℕ) (h : (13 * 15 * 17 : ℕ) = 4652) : s_in_base b = 51 := by
  sorry

end sum_in_base_b_l211_211962


namespace option_B_is_not_polynomial_l211_211183

-- Define what constitutes a polynomial
def is_polynomial (expr : String) : Prop :=
  match expr with
  | "-26m" => True
  | "3m+5n" => True
  | "0" => True
  | _ => False

-- Given expressions
def expr_A := "-26m"
def expr_B := "m-n=1"
def expr_C := "3m+5n"
def expr_D := "0"

-- The Lean statement confirming option B is not a polynomial
theorem option_B_is_not_polynomial : ¬is_polynomial expr_B :=
by
  -- Since this statement requires a proof, we use 'sorry' as a placeholder.
  sorry

end option_B_is_not_polynomial_l211_211183


namespace find_a2_plus_a8_l211_211194

variable {a_n : ℕ → ℤ}  -- Assume the sequence is indexed by natural numbers and maps to integers

-- Define the condition in the problem
def seq_property (a_n : ℕ → ℤ) := a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 25

-- Statement to prove
theorem find_a2_plus_a8 (h : seq_property a_n) : a_n 2 + a_n 8 = 10 :=
sorry

end find_a2_plus_a8_l211_211194


namespace travel_time_proportion_l211_211610

theorem travel_time_proportion (D V : ℝ) (hV_pos : V > 0) :
  let Time1 := D / (16 * V)
  let Time2 := 3 * D / (4 * V)
  let TimeTotal := Time1 + Time2
  (Time1 / TimeTotal) = 1 / 13 :=
by
  sorry

end travel_time_proportion_l211_211610


namespace arithmetic_sequence_a6_eq_4_l211_211437

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition: a_n is an arithmetic sequence, so a_(n+1) = a_n + d
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: a_2 = 2
def a_2_eq_2 (a : ℕ → ℝ) : Prop :=
  a 2 = 2

-- Condition: S_4 = 9, where S_n is the sum of first n terms of the sequence
def sum_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

def S_4_eq_9 (S : ℕ → ℝ) : Prop :=
  S 4 = 9

-- Proof: a_6 = 4
theorem arithmetic_sequence_a6_eq_4 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a_2_eq_2 a)
  (h3 : sum_S_n a S) 
  (h4 : S_4_eq_9 S) :
  a 6 = 4 := 
sorry

end arithmetic_sequence_a6_eq_4_l211_211437


namespace percent_Asian_in_West_l211_211595

noncomputable def NE_Asian := 2
noncomputable def MW_Asian := 2
noncomputable def South_Asian := 2
noncomputable def West_Asian := 6

noncomputable def total_Asian := NE_Asian + MW_Asian + South_Asian + West_Asian

theorem percent_Asian_in_West (h1 : total_Asian = 12) : (West_Asian / total_Asian) * 100 = 50 := 
by sorry

end percent_Asian_in_West_l211_211595


namespace compare_doubling_l211_211558

theorem compare_doubling (a b : ℝ) (h : a > b) : 2 * a > 2 * b :=
  sorry

end compare_doubling_l211_211558


namespace find_a_l211_211024

theorem find_a (a : ℝ) (h1 : ∀ θ : ℝ, x = a + 4 * Real.cos θ ∧ y = 1 + 4 * Real.sin θ)
  (h2 : ∃ p : ℝ × ℝ, (3 * p.1 + 4 * p.2 - 5 = 0 ∧ (∃ θ : ℝ, p = (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ))))
  (h3 : ∀ (p1 p2 : ℝ × ℝ), 
        (3 * p1.1 + 4 * p1.2 - 5 = 0 ∧ 3 * p2.1 + 4 * p2.2 - 5 = 0) ∧
        (∃ θ1 : ℝ, p1 = (a + 4 * Real.cos θ1, 1 + 4 * Real.sin θ1)) ∧
        (∃ θ2 : ℝ, p2 = (a + 4 * Real.cos θ2, 1 + 4 * Real.sin θ2)) → p1 = p2) :
  a = 7 := by
  sorry

end find_a_l211_211024


namespace min_value_f_l211_211818

def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

theorem min_value_f : ∃ (x : ℝ), f x = 15 :=
by
  sorry

end min_value_f_l211_211818


namespace cos_minus_sin_of_tan_eq_sqrt3_l211_211901

theorem cos_minus_sin_of_tan_eq_sqrt3 (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (Real.sqrt 3 - 1) / 2 := 
by
  sorry

end cos_minus_sin_of_tan_eq_sqrt3_l211_211901


namespace cheapest_book_price_l211_211809

theorem cheapest_book_price
  (n : ℕ) (c : ℕ) (d : ℕ)
  (h1 : n = 40)
  (h2 : d = 3)
  (h3 : c + d * 19 = 75) :
  c = 18 :=
sorry

end cheapest_book_price_l211_211809


namespace triangle_max_perimeter_l211_211128

noncomputable def max_perimeter_triangle_ABC (a b c : ℝ) (A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) : ℝ := 
  a + b + c

theorem triangle_max_perimeter (a b c A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) :
  max_perimeter_triangle_ABC a b c A B C h1 h2 ≤ 6 * Real.sqrt 3 :=
sorry

end triangle_max_perimeter_l211_211128


namespace pastries_more_than_cakes_l211_211433

def cakes_made : ℕ := 19
def pastries_made : ℕ := 131

theorem pastries_more_than_cakes : pastries_made - cakes_made = 112 :=
by {
  -- Proof will be inserted here
  sorry
}

end pastries_more_than_cakes_l211_211433


namespace smallest_x_for_perfect_cube_l211_211605

theorem smallest_x_for_perfect_cube :
  ∃ (x : ℕ) (h : x > 0), x = 36 ∧ (∃ (k : ℕ), 1152 * x = k ^ 3) := by
  sorry

end smallest_x_for_perfect_cube_l211_211605


namespace remainder_of_division_l211_211193

variable (a : ℝ) (b : ℝ)

theorem remainder_of_division : a = 28 → b = 10.02 → ∃ r : ℝ, 0 ≤ r ∧ r < b ∧ ∃ q : ℤ, a = q * b + r ∧ r = 7.96 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end remainder_of_division_l211_211193


namespace rachel_pizza_eaten_l211_211348

theorem rachel_pizza_eaten (pizza_total : ℕ) (pizza_bella : ℕ) (pizza_rachel : ℕ) :
  pizza_total = pizza_bella + pizza_rachel → pizza_bella = 354 → pizza_total = 952 → pizza_rachel = 598 :=
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  sorry

end rachel_pizza_eaten_l211_211348


namespace max_rabbits_with_long_ears_and_jumping_far_l211_211923

theorem max_rabbits_with_long_ears_and_jumping_far :
  ∃ N : ℕ, N = 27 ∧ 
    (∀ n : ℕ, n > 27 → 
       ¬ (∃ (r1 r2 r3 : ℕ), 
           r1 + r2 + r3 = n ∧ 
           r1 = 13 ∧
           r2 = 17 ∧
           r3 ≥ 3)) :=
sorry

end max_rabbits_with_long_ears_and_jumping_far_l211_211923


namespace root_is_neg_one_then_m_eq_neg_3_l211_211725

theorem root_is_neg_one_then_m_eq_neg_3 (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0 ∧ x = -1) : m = -3 :=
sorry

end root_is_neg_one_then_m_eq_neg_3_l211_211725


namespace dimes_turned_in_l211_211465

theorem dimes_turned_in (total_coins nickels quarters : ℕ) (h1 : total_coins = 11) (h2 : nickels = 2) (h3 : quarters = 7) : 
  ∃ dimes : ℕ, dimes + nickels + quarters = total_coins ∧ dimes = 2 :=
by
  sorry

end dimes_turned_in_l211_211465


namespace option_B_correct_l211_211066

theorem option_B_correct (a b : ℝ) (h : a < b) : a^3 < b^3 := sorry

end option_B_correct_l211_211066


namespace girls_boys_ratio_l211_211108

-- Let g be the number of girls and b be the number of boys.
-- From the conditions, we have:
-- 1. Total students: g + b = 32
-- 2. More girls than boys: g = b + 6

theorem girls_boys_ratio
  (g b : ℕ) -- Declare number of girls and boys as natural numbers
  (h1 : g + b = 32) -- Total number of students
  (h2 : g = b + 6)  -- 6 more girls than boys
  : g = 19 ∧ b = 13 := 
sorry

end girls_boys_ratio_l211_211108


namespace ratio_of_common_differences_l211_211693

variable (a b d1 d2 : ℝ)

theorem ratio_of_common_differences
  (h1 : a + 4 * d1 = b)
  (h2 : a + 5 * d2 = b) :
  d1 / d2 = 5 / 4 := 
by
  sorry

end ratio_of_common_differences_l211_211693


namespace sequence_expression_l211_211109

theorem sequence_expression (a : ℕ → ℕ) (h₀ : a 1 = 33) (h₁ : ∀ n, a (n + 1) - a n = 2 * n) : 
  ∀ n, a n = n^2 - n + 33 :=
by
  sorry

end sequence_expression_l211_211109
