import Mathlib

namespace solve_for_x_l303_303539

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l303_303539


namespace complete_square_l303_303009

theorem complete_square (x : ℝ) : 
  (x ^ 2 - 2 * x = 9) -> ((x - 1) ^ 2 = 10) :=
by
  intro h
  rw [← add_zero (x ^ 2 - 2 * x), ← add_zero (10)]
  calc
    x ^ 2 - 2 * x = 9                   : by rw [h]
             ...  = (x ^ 2 - 2 * x + 1 - 1) : by rw [add_sub_cancel, add_zero]
             ...  = (x - 1) ^ 2 - 1     : by 
                           { rw [sub_eq_add_neg], exact add_sub_cancel _ _}
             ...  = 10 - 1              : by rw [h]
             ...  = 10                  : by rw (sub_sub_cancel)
 

end complete_square_l303_303009


namespace find_minimum_a_l303_303376

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x

theorem find_minimum_a (a : ℝ) :
  (∀ x, 1 ≤ x → 0 ≤ 3 * x^2 + a) ↔ a ≥ -3 :=
by
  sorry

end find_minimum_a_l303_303376


namespace complete_square_transform_l303_303006

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l303_303006


namespace dina_dolls_count_l303_303940

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end dina_dolls_count_l303_303940


namespace cosine_ab_ac_l303_303491

noncomputable def vector_a := (-2, 4, -6)
noncomputable def vector_b := (0, 2, -4)
noncomputable def vector_c := (-6, 8, -10)

noncomputable def a_b : ℝ × ℝ × ℝ := (2, -2, 2)
noncomputable def a_c : ℝ × ℝ × ℝ := (-4, 4, -4)

noncomputable def ab_dot_ac : ℝ := -24

noncomputable def mag_a_b : ℝ := 2 * Real.sqrt 3
noncomputable def mag_a_c : ℝ := 4 * Real.sqrt 3

theorem cosine_ab_ac :
  (ab_dot_ac / (mag_a_b * mag_a_c) = -1) :=
sorry

end cosine_ab_ac_l303_303491


namespace lcm_pairs_count_l303_303507

noncomputable def distinct_pairs_lcm_count : ℕ :=
  sorry

theorem lcm_pairs_count :
  distinct_pairs_lcm_count = 1502 :=
  sorry

end lcm_pairs_count_l303_303507


namespace symmetric_points_origin_l303_303656

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end symmetric_points_origin_l303_303656


namespace dart_within_triangle_probability_l303_303461

theorem dart_within_triangle_probability (s : ℝ) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  (triangle_area / hexagon_area) = 1 / 24 :=
by sorry

end dart_within_triangle_probability_l303_303461


namespace dina_dolls_l303_303937

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end dina_dolls_l303_303937


namespace total_cows_on_farm_l303_303474

-- Defining the conditions
variables (X H : ℕ) -- X is the number of cows per herd, H is the total number of herds
axiom half_cows_counted : 2800 = X * H / 2

-- The theorem stating the total number of cows on the entire farm
theorem total_cows_on_farm (X H : ℕ) (h1 : 2800 = X * H / 2) : 5600 = X * H := 
by 
  sorry

end total_cows_on_farm_l303_303474


namespace fraction_upgraded_sensors_l303_303467

theorem fraction_upgraded_sensors (N U : ℕ) (h1 : N = U / 3) (h2 : U = 3 * N) : 
  (U : ℚ) / (24 * N + U) = 1 / 9 := by
  sorry

end fraction_upgraded_sensors_l303_303467


namespace simplify_and_rationalize_l303_303274

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l303_303274


namespace not_perfect_squares_l303_303625

theorem not_perfect_squares :
  (∀ x : ℝ, x * x ≠ 8 ^ 2041) ∧ (∀ y : ℝ, y * y ≠ 10 ^ 2043) :=
by
  sorry

end not_perfect_squares_l303_303625


namespace value_of_a_plus_b_l303_303509

theorem value_of_a_plus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 1) (h3 : a - b < 0) :
  a + b = -6 ∨ a + b = -4 :=
by
  sorry

end value_of_a_plus_b_l303_303509


namespace candies_per_person_l303_303300

theorem candies_per_person (a b people total_candies candies_per_person : ℕ)
  (h1: a = 17)
  (h2: b = 19)
  (h3: people = 9)
  (h4: total_candies = a + b)
  (h5: candies_per_person = total_candies / people) :
  candies_per_person = 4 :=
by sorry

end candies_per_person_l303_303300


namespace Eve_age_l303_303039

theorem Eve_age (Adam_age : ℕ) (Eve_age : ℕ) (h1 : Adam_age = 9) (h2 : Eve_age = Adam_age + 5)
  (h3 : ∃ k : ℕ, Eve_age + 1 = k * (Adam_age - 4)) : Eve_age = 14 :=
sorry

end Eve_age_l303_303039


namespace simplify_and_rationalize_l303_303270

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l303_303270


namespace simplify_and_rationalize_l303_303278

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l303_303278


namespace combined_weight_of_olivers_bags_l303_303706

-- Define the weights and relationship between the weights
def weight_james_bag : ℝ := 18
def ratio_olivers_to_james : ℝ := 1 / 6
def weight_oliver_one_bag : ℝ := weight_james_bag * ratio_olivers_to_james
def number_of_oliver_bags : ℝ := 2

-- The proof problem statement: proving the combined weight of both Oliver's bags
theorem combined_weight_of_olivers_bags :
  number_of_oliver_bags * weight_oliver_one_bag = 6 := by
  sorry

end combined_weight_of_olivers_bags_l303_303706


namespace certain_number_value_l303_303964

variable {t b c x : ℕ}

theorem certain_number_value 
  (h1 : (t + b + c + 14 + x) / 5 = 12) 
  (h2 : (t + b + c + 29) / 4 = 15) : 
  x = 15 := 
by
  sorry

end certain_number_value_l303_303964


namespace sequence_solution_l303_303062

theorem sequence_solution :
  ∀ (a : ℕ → ℝ), (∀ m n : ℕ, a (m^2 + n^2) = a m ^ 2 + a n ^ 2) →
  (0 ≤ a 0 ∧ a 0 ≤ a 1 ∧ a 1 ≤ a 2 ∧ ∀ n, a n ≤ a (n + 1)) →
  (∀ n, a n = 0) ∨ (∀ n, a n = n) ∨ (∀ n, a n = 1 / 2) :=
sorry

end sequence_solution_l303_303062


namespace inequality_no_solution_l303_303139

theorem inequality_no_solution : 
  ∀ x : ℝ, -2 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 2 → false :=
by sorry

end inequality_no_solution_l303_303139


namespace dina_dolls_count_l303_303939

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end dina_dolls_count_l303_303939


namespace rationalize_denominator_l303_303265

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l303_303265


namespace medians_inequality_l303_303405

  variable {a b c : ℝ} (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)

  noncomputable def median_length (a b c : ℝ) : ℝ :=
    1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2)

  noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
    (a + b + c) / 2

  theorem medians_inequality (m_a m_b m_c s: ℝ)
    (h_ma : m_a = median_length a b c)
    (h_mb : m_b = median_length b c a)
    (h_mc : m_c = median_length c a b)
    (h_s : s = semiperimeter a b c) :
    m_a^2 + m_b^2 + m_c^2 ≥ s^2 := by
  sorry
  
end medians_inequality_l303_303405


namespace max_reciprocal_sum_eq_2_l303_303059

theorem max_reciprocal_sum_eq_2 (r1 r2 t q : ℝ) (h1 : r1 + r2 = t) (h2 : r1 * r2 = q)
  (h3 : ∀ n : ℕ, n > 0 → r1 + r2 = r1^n + r2^n) :
  1 / r1^2010 + 1 / r2^2010 = 2 :=
by
  sorry

end max_reciprocal_sum_eq_2_l303_303059


namespace graph_empty_l303_303622

theorem graph_empty (x y : ℝ) : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 9 ≠ 0 :=
by
  -- Proof omitted
  sorry

end graph_empty_l303_303622


namespace part1_part2_l303_303294

open ArithmeticSequence

variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}
variables (n : ℕ)

-- Conditions
axiom seq_a_arithmetic : arithmetic_sequence a
axiom seq_b_arithmetic : arithmetic_sequence b
axiom sum_of_n_terms : S n = sum_first_n_terms a n
axiom sum_of_n_terms_b : T n = sum_first_n_terms b n
axiom given_ratio : ∀ n, S n / T n = (3 * n + 1) / (n + 3)

-- Prove part 1
theorem part1 : (a 2 + a 20) / (b 7 + b 15) = 8 / 3 := 
    sorry

-- Prove part 2
theorem part2 : {n : ℕ | (a n / b n).is_integer}.card = 2 := 
    sorry

end part1_part2_l303_303294


namespace highest_student_id_in_sample_l303_303968

variable (n : ℕ) (start : ℕ) (interval : ℕ)

theorem highest_student_id_in_sample :
  start = 5 → n = 54 → interval = 9 → 6 = n / interval → start = 5 →
  5 + (interval * (6 - 1)) = 50 :=
by
  sorry

end highest_student_id_in_sample_l303_303968


namespace solveKnight9x9_l303_303515

structure Position where
  x : Nat
  y : Nat

def knightMoves (p : Position) : List Position :=
  [{ x := p.x + 2, y := p.y + 1 }, { x := p.x + 2, y := p.y - 1 },
   { x := p.x - 2, y := p.y + 1 }, { x := p.x - 2, y := p.y - 1 },
   { x := p.x + 1, y := p.y + 2 }, { x := p.x + 1, y := p.y - 2 },
   { x := p.x - 1, y := p.y + 2 }, { x := p.x - 1, y := p.y - 2 }]
  |> List.filter (λ q => q.x > 0 ∧ q.y > 0 ∧ q.x ≤ 9 ∧ q.y ≤ 9)

def canReachAll : Prop :=
  ∀ p : Position, p.x > 0 ∧ p.y > 0 ∧ p.x ≤ 9 ∧ p.y ≤ 9 →
  ∃ path : List Position, path.head = some ⟨1,1⟩ ∧ path.last = some p ∧
  ∀ (h : p ∈ path.tail) (path : List Position), (path.head ∈ knightMoves path.tail.head)

def maxSteps : Nat
def furthestPoints : List Position

theorem solveKnight9x9 : canReachAll ∧ (maxSteps = 6 ∧ furthestPoints = [{ x := 8, y := 8 }, { x := 9, y := 7 }, { x := 9, y := 9 }]) := 
by sorry

end solveKnight9x9_l303_303515


namespace find_function_f_l303_303360

-- The function f maps positive integers to positive integers
def f : ℕ+ → ℕ+ := sorry

-- The statement to be proved
theorem find_function_f (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, (f m)^2 + f n ∣ (m^2 + n)^2) : ∀ n : ℕ+, f n = n :=
sorry

end find_function_f_l303_303360


namespace max_bees_in_largest_beehive_l303_303598

def total_bees : ℕ := 2000000
def beehives : ℕ := 7
def min_ratio : ℚ := 0.7

theorem max_bees_in_largest_beehive (B_max : ℚ) : 
  (6 * (min_ratio * B_max) + B_max = total_bees) → 
  B_max <= 2000000 / 5.2 ∧ B_max.floor = 384615 :=
by
  sorry

end max_bees_in_largest_beehive_l303_303598


namespace find_f_neg_2010_6_l303_303080

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_one (x : ℝ) : f (x + 1) + f x = 3

axiom f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = 2 - x

theorem find_f_neg_2010_6 : f (-2010.6) = 1.4 := by {
  sorry
}

end find_f_neg_2010_6_l303_303080


namespace find_tax_percentage_l303_303679

-- Definitions based on given conditions
def income_total : ℝ := 58000
def income_threshold : ℝ := 40000
def tax_above_threshold_percentage : ℝ := 0.2
def total_tax : ℝ := 8000

-- Let P be the percentage taxed on the first $40,000
variable (P : ℝ)

-- Formulate the problem as a proof goal
theorem find_tax_percentage (h : total_tax = 8000) :
  P = ((total_tax - (tax_above_threshold_percentage * (income_total - income_threshold))) / income_threshold) * 100 :=
by sorry

end find_tax_percentage_l303_303679


namespace ratio_of_areas_is_16_l303_303438

-- Definitions and conditions
variables (a b : ℝ)

-- Given condition: Perimeter of the larger square is 4 times the perimeter of the smaller square
def perimeter_relation (ha : a = 4 * b) : Prop := a = 4 * b

-- Theorem to prove: Ratio of the area of the larger square to the area of the smaller square is 16
theorem ratio_of_areas_is_16 (ha : a = 4 * b) : (a^2 / b^2) = 16 :=
by
  sorry

end ratio_of_areas_is_16_l303_303438


namespace y_range_for_conditions_l303_303834

theorem y_range_for_conditions (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : -9 ≤ y ∧ y < -8 :=
sorry

end y_range_for_conditions_l303_303834


namespace number_of_correct_conclusions_l303_303368

-- Given conditions
variables {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : c > 3)
           (h₂ : a * 25 + b * 5 + c = 0)
           (h₃ : -b / (2 * a) = 2)
           (h₄ : a < 0)

-- Proof should show:
theorem number_of_correct_conclusions 
  (h₀ : a ≠ 0)
  (h₁ : c > 3)
  (h₂ : 25 * a + 5 * b + c = 0)
  (h₃ : - b / (2 * a) = 2)
  (h₄ : a < 0) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ (a * x₁^2 + b * x₁ + c = 2) ∧ (a * x₂^2 + b * x₂ + c = 2)) ∧ 
  (a < -3 / 5) := 
by
  sorry

end number_of_correct_conclusions_l303_303368


namespace solve_for_x_l303_303543

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l303_303543


namespace eight_percent_of_fifty_is_four_l303_303185

theorem eight_percent_of_fifty_is_four : 0.08 * 50 = 4 := by
  sorry

end eight_percent_of_fifty_is_four_l303_303185


namespace total_travel_time_l303_303789

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l303_303789


namespace problem1_problem2_l303_303785

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem1_problem2_l303_303785


namespace find_a_l303_303227

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

-- Define the derivative of function f with respect to x
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

-- Define the condition for the problem
def condition (a : ℝ) : Prop := f' a 1 = 2

-- The statement to be proved
theorem find_a (a : ℝ) (h : condition a) : a = -3 :=
by {
  -- Proof is omitted
  sorry
}

end find_a_l303_303227


namespace max_imaginary_part_angle_l303_303608

def poly (z : Complex) : Complex := z^6 - z^4 + z^2 - 1

theorem max_imaginary_part_angle :
  ∃ θ : Real, θ = 45 ∧ 
  (∃ z : Complex, poly z = 0 ∧ ∀ w : Complex, poly w = 0 → w.im ≤ z.im)
:= sorry

end max_imaginary_part_angle_l303_303608


namespace star_wars_cost_l303_303142

theorem star_wars_cost 
    (LK_cost LK_earn SW_earn: ℕ) 
    (half_profit: ℕ → ℕ)
    (h1: LK_cost = 10)
    (h2: LK_earn = 200)
    (h3: SW_earn = 405)
    (h4: LK_earn - LK_cost = half_profit SW_earn)
    (h5: half_profit SW_earn * 2 = SW_earn - (LK_earn - LK_cost)) :
    ∃ SW_cost : ℕ, SW_cost = 25 := 
by
  sorry

end star_wars_cost_l303_303142


namespace range_of_m_for_point_in_second_quadrant_l303_303106

theorem range_of_m_for_point_in_second_quadrant (m : ℝ) :
  (m - 3 < 0) ∧ (m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  -- The proof will be inserted here.
  sorry

end range_of_m_for_point_in_second_quadrant_l303_303106


namespace right_triangle_set_C_l303_303751

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_set_C_l303_303751


namespace minimum_moves_black_white_swap_l303_303708

-- Define an initial setup of the chessboard
def initial_positions_black := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8)]
def initial_positions_white := [(8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8)]

-- Define chess rules, positions, and switching places
def black_to_white_target := initial_positions_white
def white_to_black_target := initial_positions_black

-- Define a function to count minimal moves (trivial here just for the purpose of this statement)
def min_moves_to_switch_positions := 23

-- The main theorem statement proving necessity of at least 23 moves
theorem minimum_moves_black_white_swap :
  ∀ (black_positions white_positions : List (ℕ × ℕ)),
  black_positions = initial_positions_black →
  white_positions = initial_positions_white →
  min_moves_to_switch_positions ≥ 23 :=
by
  sorry

end minimum_moves_black_white_swap_l303_303708


namespace mass_percentage_K_l303_303493

theorem mass_percentage_K (compound : Type) (m : ℝ) (mass_percentage : ℝ) (h : mass_percentage = 23.81) : mass_percentage = 23.81 :=
by
  sorry

end mass_percentage_K_l303_303493


namespace original_price_calc_l303_303197

theorem original_price_calc (h : 1.08 * x = 2) : x = 100 / 54 := by
  sorry

end original_price_calc_l303_303197


namespace polynomial_integer_roots_k_zero_l303_303593

theorem polynomial_integer_roots_k_zero :
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + 0) ∨
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + k)) →
  k = 0 :=
sorry

end polynomial_integer_roots_k_zero_l303_303593


namespace fraction_n_m_l303_303220

noncomputable def a (k : ℝ) := 2*k + 1
noncomputable def b (k : ℝ) := 3*k + 2
noncomputable def c (k : ℝ) := 3 - 4*k
noncomputable def S (k : ℝ) := a k + 2*(b k) + 3*(c k)

theorem fraction_n_m : 
  (∀ (k : ℝ), -1/2 ≤ k ∧ k ≤ 3/4 → (S (3/4) = 11 ∧ S (-1/2) = 16)) → 
  11/16 = 11 / 16 :=
by
  sorry

end fraction_n_m_l303_303220


namespace find_investment_sum_l303_303588

theorem find_investment_sum (P : ℝ)
  (h1 : SI_15 = P * (15 / 100) * 2)
  (h2 : SI_12 = P * (12 / 100) * 2)
  (h3 : SI_15 - SI_12 = 420) :
  P = 7000 :=
by
  sorry

end find_investment_sum_l303_303588


namespace granddaughter_age_is_12_l303_303045

/-
Conditions:
- Betty is 60 years old.
- Her daughter is 40 percent younger than Betty.
- Her granddaughter is one-third her mother's age.

Question:
- Prove that the granddaughter is 12 years old.
-/

def age_of_Betty := 60

def age_of_daughter (age_of_Betty : ℕ) : ℕ :=
  age_of_Betty - age_of_Betty * 40 / 100

def age_of_granddaughter (age_of_daughter : ℕ) : ℕ :=
  age_of_daughter / 3

theorem granddaughter_age_is_12 (h1 : age_of_Betty = 60) : age_of_granddaughter (age_of_daughter age_of_Betty) = 12 := by
  sorry

end granddaughter_age_is_12_l303_303045


namespace martha_initial_apples_l303_303988

theorem martha_initial_apples :
  ∀ (jane_apples james_apples keep_apples more_to_give initial_apples : ℕ),
    jane_apples = 5 →
    james_apples = jane_apples + 2 →
    keep_apples = 4 →
    more_to_give = 4 →
    initial_apples = jane_apples + james_apples + keep_apples + more_to_give →
    initial_apples = 20 :=
by
  intros jane_apples james_apples keep_apples more_to_give initial_apples
  intro h_jane
  intro h_james
  intro h_keep
  intro h_more
  intro h_initial
  exact sorry

end martha_initial_apples_l303_303988


namespace mangoes_combined_l303_303193

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end mangoes_combined_l303_303193


namespace qin_jiushao_algorithm_correct_operations_l303_303131

def qin_jiushao_algorithm_operations (f : ℝ → ℝ) (x : ℝ) : ℕ × ℕ := sorry

def f (x : ℝ) : ℝ := 4 * x^5 - x^2 + 2
def x : ℝ := 3

theorem qin_jiushao_algorithm_correct_operations :
  qin_jiushao_algorithm_operations f x = (5, 2) :=
sorry

end qin_jiushao_algorithm_correct_operations_l303_303131


namespace second_investment_value_l303_303596

theorem second_investment_value
  (a : ℝ) (r1 r2 rt : ℝ) (x : ℝ)
  (h1 : a = 500)
  (h2 : r1 = 0.07)
  (h3 : r2 = 0.09)
  (h4 : rt = 0.085)
  (h5 : r1 * a + r2 * x = rt * (a + x)) :
  x = 1500 :=
by 
  -- The proof will go here
  sorry

end second_investment_value_l303_303596


namespace problem_proof_l303_303832

theorem problem_proof (M N : ℕ) 
  (h1 : 4 * 63 = 7 * M) 
  (h2 : 4 * N = 7 * 84) : 
  M + N = 183 :=
sorry

end problem_proof_l303_303832


namespace remainder_when_3m_div_by_5_l303_303510

variable (m k : ℤ)

theorem remainder_when_3m_div_by_5 (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end remainder_when_3m_div_by_5_l303_303510


namespace measure_of_angle_is_135_l303_303004

noncomputable def degree_measure_of_angle (x : ℝ) : Prop :=
  (x = 3 * (180 - x)) ∧ (2 * x + (180 - x) = 180) -- Combining all conditions

theorem measure_of_angle_is_135 (x : ℝ) (h : degree_measure_of_angle x) : x = 135 :=
by sorry

end measure_of_angle_is_135_l303_303004


namespace hotel_charge_percentage_l303_303144

theorem hotel_charge_percentage (G R P : ℝ) 
  (hR : R = 1.60 * G) 
  (hP : P = 0.80 * G) : 
  ((R - P) / R) * 100 = 50 := by
  sorry

end hotel_charge_percentage_l303_303144


namespace count_integer_values_of_x_l303_303387

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : {n : ℕ | 121 < n ∧ n ≤ 144}.card = 23 := 
by
  sorry

end count_integer_values_of_x_l303_303387


namespace total_volume_of_quiche_l303_303255

def raw_spinach_volume : ℝ := 40
def cooked_volume_percentage : ℝ := 0.20
def cream_cheese_volume : ℝ := 6
def eggs_volume : ℝ := 4

theorem total_volume_of_quiche :
  raw_spinach_volume * cooked_volume_percentage + cream_cheese_volume + eggs_volume = 18 := by
  sorry

end total_volume_of_quiche_l303_303255


namespace isolating_line_unique_l303_303840

noncomputable def f (x : ℝ) := x^2
noncomputable def g (a x : ℝ) := a * log x

theorem isolating_line_unique (a : ℝ) (hx : ∀ x, f x ≥ g a x ∧ g a x ≥ f x) :
  a = 2 * real.exp 1 := 
sorry

end isolating_line_unique_l303_303840


namespace range_of_a_l303_303517

def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l303_303517


namespace ProbabilityAllisonGreater_l303_303471

-- Define the probability mass functions for the dice
def AllisonCube : ProbabilityMassFunction (Fin 7) := ProbabilityMassFunction.uniform (Fin 7) (Fin.val 6)
def BrianCube : ProbabilityMassFunction (Fin 7) := ProbabilityMassFunction.uniform (Fin 7)
def NoahCube : ProbabilityMassFunction (Fin 7) :=
  ProbabilityMassFunction.mk
    (λ n, if n = Fin 3 3 then 1 / 2 else if n = Fin 3 5 then 1 / 2 else 0)
    sorry  -- proof of sum = 1

-- Define the event that Allison's roll is greater than both Brian's and Noah's
def EventAllisonGreater : Set (Fin 7 × Fin 7 × Fin 7) :=
  {x | x.1 = Fin.ofNat 6 ∧ x.2 < Fin.ofNat 6 ∧ x.3 < Fin.ofNat 6}

-- State the theorem
theorem ProbabilityAllisonGreater :
  (ProbabilityMassFunction.bind AllisonCube
    (λ a, ProbabilityMassFunction.bind BrianCube
      (λ b, ProbabilityMassFunction.map (λ c => (a, b, c)) NoahCube))).prob EventAllisonGreater =
  5 / 12 := 
sorry

end ProbabilityAllisonGreater_l303_303471


namespace geometric_sequence_product_l303_303157

-- Define the geometric sequence sum and the initial conditions
variables {S : ℕ → ℚ} {a : ℕ → ℚ}
variables (q : ℚ) (h1 : a 1 = -1/2)
variables (h2 : S 6 / S 3 = 7 / 8)

-- The main proof problem statement
theorem geometric_sequence_product (h_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 2 * a 4 = 1 / 64 :=
sorry

end geometric_sequence_product_l303_303157


namespace can_restore_axes_l303_303681

noncomputable def restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : Prop :=
  ∃ (B C D : ℝ×ℝ),
    (B.fst = A.fst ∧ B.snd = 0) ∧
    (C.fst = A.fst ∧ C.snd = A.snd) ∧
    (D.fst = A.fst ∧ D.snd = 3 ^ C.fst) ∧
    (∃ (extend_perpendicular : ∀ (x: ℝ), ℝ→ℝ), extend_perpendicular A.snd B.fst = D.snd)

theorem can_restore_axes (A : ℝ×ℝ) (hA : A.snd = 3 ^ A.fst) : restore_axes A hA :=
  sorry

end can_restore_axes_l303_303681


namespace mul_72518_9999_eq_725107482_l303_303016

theorem mul_72518_9999_eq_725107482 : 72518 * 9999 = 725107482 := by
  sorry

end mul_72518_9999_eq_725107482_l303_303016


namespace ribbons_green_count_l303_303104

theorem ribbons_green_count
  (N : ℕ)  -- The total number of ribbons
  (red_ribbons : ℕ := N / 4)   -- Red ribbons are 1/4 of the total
  (blue_ribbons : ℕ := 3 * N / 8)   -- Blue ribbons are 3/8 of the total
  (green_ribbons : ℕ := N / 8)   -- Green ribbons are 1/8 of the total
  (white_ribbons : ℕ := 36) -- The remaining ribbons are white
  (h : N - (red_ribbons + blue_ribbons + green_ribbons) = white_ribbons) :
  green_ribbons = 18 := sorry

end ribbons_green_count_l303_303104


namespace base_7_3516_is_1287_l303_303444

-- Definitions based on conditions
def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 3516 => 3 * 7^3 + 5 * 7^2 + 1 * 7^1 + 6 * 7^0
  | _ => 0

-- Proving the main question
theorem base_7_3516_is_1287 : base7_to_base10 3516 = 1287 := by
  sorry

end base_7_3516_is_1287_l303_303444


namespace total_travel_time_l303_303792

-- Define the necessary distances and speeds
def distance_Washington_to_Idaho : ℝ := 640
def speed_Washington_to_Idaho : ℝ := 80
def distance_Idaho_to_Nevada : ℝ := 550
def speed_Idaho_to_Nevada : ℝ := 50

-- Definitions for time calculations
def time_Washington_to_Idaho : ℝ := distance_Washington_to_Idaho / speed_Washington_to_Idaho
def time_Idaho_to_Nevada : ℝ := distance_Idaho_to_Nevada / speed_Idaho_to_Nevada

-- Problem statement to prove
theorem total_travel_time : time_Washington_to_Idaho + time_Idaho_to_Nevada = 19 := 
by
  sorry

end total_travel_time_l303_303792


namespace tan_cos_solution_count_l303_303689

theorem tan_cos_solution_count : 
  ∃ (n : ℕ), n = 5 ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.tan (2 * x) = Real.cos (x / 2) → x ∈ Set.Icc 0 (2 * Real.pi) :=
sorry

end tan_cos_solution_count_l303_303689


namespace distance_between_red_lights_in_feet_l303_303717

theorem distance_between_red_lights_in_feet :
  let inches_between_lights := 6
  let pattern := [2, 3]
  let foot_in_inches := 12
  let pos_3rd_red := 6
  let pos_21st_red := 51
  let number_of_gaps := pos_21st_red - pos_3rd_red
  let total_distance_in_inches := number_of_gaps * inches_between_lights
  let total_distance_in_feet := total_distance_in_inches / foot_in_inches
  total_distance_in_feet = 22 := by
  sorry

end distance_between_red_lights_in_feet_l303_303717


namespace abc_inequality_l303_303023

-- Define a mathematical statement to encapsulate the problem
theorem abc_inequality (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l303_303023


namespace average_speed_of_trip_l303_303913

theorem average_speed_of_trip :
  let total_distance := 50 -- in kilometers
  let distance1 := 25 -- in kilometers
  let speed1 := 66 -- in kilometers per hour
  let distance2 := 25 -- in kilometers
  let speed2 := 33 -- in kilometers per hour
  let time1 := distance1 / speed1 -- time taken for the first part
  let time2 := distance2 / speed2 -- time taken for the second part
  let total_time := time1 + time2 -- total time for the trip
  let average_speed := total_distance / total_time -- average speed of the trip
  average_speed = 44 := by
{
  sorry
}

end average_speed_of_trip_l303_303913


namespace base3_addition_correct_l303_303040

theorem base3_addition_correct :
  nat.addDigits 3 [2] + nat.addDigits 3 [1,2,1] + nat.addDigits 3 [1,2,1,2] + nat.addDigits 3 [1,2,1,2,1] = nat.addDigits 3 [2,1,1,1] :=
begin
  sorry
end

end base3_addition_correct_l303_303040


namespace right_triangle_set_C_l303_303752

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_set_C_l303_303752


namespace solve_equation_l303_303554

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l303_303554


namespace therapists_next_meeting_day_l303_303776

theorem therapists_next_meeting_day : Nat.lcm (Nat.lcm 5 2) (Nat.lcm 9 3) = 90 := by
  -- Given that Alex works every 5 days,
  -- Brice works every 2 days,
  -- Emma works every 9 days,
  -- and Fiona works every 3 days, we need to show that the LCM of these numbers is 90.
  sorry

end therapists_next_meeting_day_l303_303776


namespace cosine_sine_difference_identity_l303_303054

theorem cosine_sine_difference_identity :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
  - Real.sin (255 * Real.pi / 180) * Real.sin (165 * Real.pi / 180)) = 1 / 2 := by
  -- Proof goes here
  sorry

end cosine_sine_difference_identity_l303_303054


namespace gambler_initial_games_l303_303766

theorem gambler_initial_games (x : ℕ)
  (h1 : ∀ x, ∃ (wins : ℝ), wins = 0.40 * x) 
  (h2 : ∀ x, ∃ (total_games : ℕ), total_games = x + 30)
  (h3 : ∀ x, ∃ (total_wins : ℝ), total_wins = 0.40 * x + 24)
  (h4 : ∀ x, ∃ (final_win_rate : ℝ), final_win_rate = (0.40 * x + 24) / (x + 30))
  (h5 : ∃ (final_win_rate_target : ℝ), final_win_rate_target = 0.60) :
  x = 30 :=
by
  sorry

end gambler_initial_games_l303_303766


namespace min_value_z_l303_303744

theorem min_value_z : ∃ (min_z : ℝ), min_z = 24.1 ∧ 
  ∀ (x y : ℝ), (3 * x ^ 2 + 4 * y ^ 2 + 8 * x - 6 * y + 30) ≥ min_z :=
sorry

end min_value_z_l303_303744


namespace solve_congruence_l303_303720

theorem solve_congruence (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 :=
by {
  sorry
}

end solve_congruence_l303_303720


namespace kamal_marks_physics_correct_l303_303694

-- Definition of the conditions
def kamal_marks_english : ℕ := 76
def kamal_marks_mathematics : ℕ := 60
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 74
def kamal_num_subjects : ℕ := 5

-- Definition of the total marks
def kamal_total_marks : ℕ := kamal_average_marks * kamal_num_subjects

-- Sum of known marks
def kamal_known_marks : ℕ := kamal_marks_english + kamal_marks_mathematics + kamal_marks_chemistry + kamal_marks_biology

-- The expected result for Physics
def kamal_marks_physics : ℕ := 82

-- Proof statement
theorem kamal_marks_physics_correct :
  kamal_total_marks - kamal_known_marks = kamal_marks_physics :=
by
  simp [kamal_total_marks, kamal_known_marks, kamal_marks_physics]
  sorry

end kamal_marks_physics_correct_l303_303694


namespace semifinalists_count_l303_303685

theorem semifinalists_count (n : ℕ) (h : (n - 2) * (n - 3) * (n - 4) = 336) : n = 10 := 
by {
  sorry
}

end semifinalists_count_l303_303685


namespace rationalize_denominator_l303_303266

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l303_303266


namespace money_left_after_transactions_l303_303261

-- Define the coin values and quantities
def dimes := 50
def quarters := 24
def nickels := 40
def pennies := 75

-- Define the item costs
def candy_bar_cost := 6 * 10 + 4 * 5 + 5
def lollipop_cost := 25 + 2 * 10 + 10 - 5 
def bag_of_chips_cost := 2 * 25 + 3 * 10 + 15
def bottle_of_soda_cost := 25 + 6 * 10 + 5 * 5 + 20 - 5

-- Define the number of items bought
def num_candy_bars := 6
def num_lollipops := 3
def num_bags_of_chips := 4
def num_bottles_of_soda := 2

-- Define the initial total money
def total_money := (dimes * 10) + (quarters * 25) + (nickels * 5) + (pennies)

-- Calculate the total cost of items
def total_cost := num_candy_bars * candy_bar_cost + num_lollipops * lollipop_cost + num_bags_of_chips * bag_of_chips_cost + num_bottles_of_soda * bottle_of_soda_cost

-- Calculate the money left after transactions
def money_left := total_money - total_cost

-- Theorem statement to prove
theorem money_left_after_transactions : money_left = 85 := by
  sorry

end money_left_after_transactions_l303_303261


namespace part_one_extreme_value_part_two_max_k_l303_303502

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  x * Real.log x - k * (x - 1)

theorem part_one_extreme_value :
  ∃ x : ℝ, x > 0 ∧ ∀ y > 0, f y 1 ≥ f x 1 ∧ f x 1 = 0 := 
  sorry

theorem part_two_max_k :
  ∀ x : ℝ, ∃ k : ℕ, (1 < x) -> (f x (k:ℝ) + x > 0) ∧ k = 3 :=
  sorry

end part_one_extreme_value_part_two_max_k_l303_303502


namespace lines_parallel_a_eq_sqrt2_l303_303965

theorem lines_parallel_a_eq_sqrt2 (a : ℝ) (h1 : 1 ≠ 0) :
  (∀ a ≠ 0, ((- (1 / (2 * a))) = (- a / 2)) → a = Real.sqrt 2) :=
by
  sorry

end lines_parallel_a_eq_sqrt2_l303_303965


namespace students_errors_proof_l303_303044

noncomputable def students (x y0 y1 y2 y3 y4 y5 : ℕ): ℕ :=
  x + y5 + y4 + y3 + y2 + y1 + y0

noncomputable def errors (x y1 y2 y3 y4 y5 : ℕ): ℕ :=
  6 * x + 5 * y5 + 4 * y4 + 3 * y3 + 2 * y2 + y1

theorem students_errors_proof
  (x y0 y1 y2 y3 y4 y5 : ℕ)
  (h1 : students x y0 y1 y2 y3 y4 y5 = 333)
  (h2 : errors x y1 y2 y3 y4 y5 ≤ 1000) :
  x ≤ y3 + y2 + y1 + y0 :=
by
  sorry

end students_errors_proof_l303_303044


namespace calc_expression_result_l303_303481

theorem calc_expression_result :
  (16^12 * 8^8 / 2^60 = 4096) :=
by
  sorry

end calc_expression_result_l303_303481


namespace smallest_four_digit_divisible_by_4_and_5_l303_303314

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l303_303314


namespace missing_number_approximately_1400_l303_303719

theorem missing_number_approximately_1400 :
  ∃ x : ℤ, x * 54 = 75625 ∧ abs (x - Int.ofNat (75625 / 54)) ≤ 1 :=
by
  sorry

end missing_number_approximately_1400_l303_303719


namespace minimum_cuts_for_polygons_l303_303446

theorem minimum_cuts_for_polygons (initial_pieces desired_pieces : ℕ) (sides : ℕ)
    (h_initial_pieces : initial_pieces = 1) (h_desired_pieces : desired_pieces = 100)
    (h_sides : sides = 20) :
    ∃ (cuts : ℕ), cuts = 1699 ∧
    (∀ current_pieces, current_pieces < desired_pieces → current_pieces + cuts ≥ desired_pieces) :=
by
    sorry

end minimum_cuts_for_polygons_l303_303446


namespace largest_int_lt_100_remainder_3_div_by_8_l303_303363

theorem largest_int_lt_100_remainder_3_div_by_8 : 
  ∃ n, n < 100 ∧ n % 8 = 3 ∧ ∀ m, m < 100 ∧ m % 8 = 3 → m ≤ 99 := by
  sorry

end largest_int_lt_100_remainder_3_div_by_8_l303_303363


namespace cost_price_each_watch_l303_303737

open Real

theorem cost_price_each_watch
  (C : ℝ)
  (h1 : let lossPerc := 0.075 in
        let sp_each := C * (1 - lossPerc) in
        let gainPerc := 0.053 in
        let sp_more := sp_each + 265 in
        let sp_total := 3 * sp_more in
        sp_total = 3 * C * (1 + gainPerc)) :
  C ≈ 2070.31 := by
  sorry

end cost_price_each_watch_l303_303737


namespace PQ_perpendicular_to_KX_l303_303994

def midpoint (A B : Point) : Point := 
  sorry -- Assume the midpoint function is defined

def circumcenter (A B C : Point) : Point := 
  sorry -- Assume the circumcenter function is defined

theorem PQ_perpendicular_to_KX {A B C D K L M N P Q X : Point}
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_equilateral_triangle_outside A B K)
  (h3 : is_equilateral_triangle_outside B C L)
  (h4 : is_equilateral_triangle_outside C D M)
  (h5 : is_equilateral_triangle_outside D A N)
  (hP : P = midpoint B L)
  (hQ : Q = midpoint A N)
  (hX : X = circumcenter C M D) :
  is_perpendicular P Q K X :=
sorry

end PQ_perpendicular_to_KX_l303_303994


namespace raj_house_area_l303_303866

theorem raj_house_area :
  let bedroom_area := 11 * 11
  let bedrooms_total := bedroom_area * 4
  let bathroom_area := 6 * 8
  let bathrooms_total := bathroom_area * 2
  let kitchen_area := 265
  let living_area := kitchen_area
  bedrooms_total + bathrooms_total + kitchen_area + living_area = 1110 :=
by
  -- Proof to be filled in
  sorry

end raj_house_area_l303_303866


namespace longest_piece_length_l303_303410

-- Define the lengths of the ropes
def rope1 : ℕ := 45
def rope2 : ℕ := 75
def rope3 : ℕ := 90

-- Define the greatest common divisor we need to prove
def gcd_of_ropes : ℕ := Nat.gcd rope1 (Nat.gcd rope2 rope3)

-- Goal theorem stating the problem
theorem longest_piece_length : gcd_of_ropes = 15 := by
  sorry

end longest_piece_length_l303_303410


namespace trig_identity_l303_303812

theorem trig_identity (θ : ℝ) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
  sorry

end trig_identity_l303_303812


namespace chloes_test_scores_l303_303192

theorem chloes_test_scores :
  ∃ (scores : List ℕ),
  scores = [93, 92, 86, 82, 79, 78] ∧
  (List.take 4 scores).sum = 339 ∧
  scores.sum / 6 = 85 ∧
  List.Nodup scores ∧
  ∀ score ∈ scores, score < 95 :=
by
  sorry

end chloes_test_scores_l303_303192


namespace total_squares_after_removals_l303_303305

/-- 
Prove that the total number of squares of various sizes on a 5x5 grid,
after removing two 1x1 squares, is 55.
-/
theorem total_squares_after_removals (total_squares_in_5x5_grid: ℕ) (removed_squares: ℕ) : 
  (total_squares_in_5x5_grid = 25 + 16 + 9 + 4 + 1) →
  (removed_squares = 2) →
  (total_squares_in_5x5_grid - removed_squares = 55) :=
sorry

end total_squares_after_removals_l303_303305


namespace num_of_terms_in_arithmetic_sequence_l303_303829

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the first term, common difference, and last term of the sequence
def a : ℕ := 15
def d : ℕ := 4
def last_term : ℕ := 99

-- Define the number of terms in the sequence
def n : ℕ := 22

-- State the theorem
theorem num_of_terms_in_arithmetic_sequence : arithmetic_seq a d n = last_term :=
by
  sorry

end num_of_terms_in_arithmetic_sequence_l303_303829


namespace solve_for_x_l303_303549

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l303_303549


namespace quotient_remainder_div_by_18_l303_303748

theorem quotient_remainder_div_by_18 (M q : ℕ) (h : M = 54 * q + 37) : 
  ∃ k r, M = 18 * k + r ∧ r < 18 ∧ k = 3 * q + 2 ∧ r = 1 :=
by sorry

end quotient_remainder_div_by_18_l303_303748


namespace jaco_budget_for_parents_l303_303407

/-- Assume Jaco has 8 friends, each friend's gift costs $9, and Jaco has a total budget of $100.
    Prove that Jaco's budget for each of his mother and father's gift is $14. -/
theorem jaco_budget_for_parents :
  ∀ (friends_count cost_per_friend total_budget : ℕ), 
  friends_count = 8 → 
  cost_per_friend = 9 → 
  total_budget = 100 → 
  (total_budget - friends_count * cost_per_friend) / 2 = 14 :=
by
  intros friends_count cost_per_friend total_budget h1 h2 h3
  rw [h1, h2, h3]
  have friend_total_cost : friends_count * cost_per_friend = 72 := by norm_num
  have remaining_budget : total_budget - friends_count * cost_per_friend = 28 := by norm_num [friend_total_cost]
  have divided_budget : remaining_budget / 2 = 14 := by norm_num [remaining_budget]
  exact divided_budget

end jaco_budget_for_parents_l303_303407


namespace min_weight_of_automobile_l303_303915

theorem min_weight_of_automobile (ferry_weight_tons: ℝ) (auto_max_weight: ℝ) 
  (max_autos: ℝ) (ferry_weight_pounds: ℝ) (min_auto_weight: ℝ) : 
  ferry_weight_tons = 50 → 
  auto_max_weight = 3200 → 
  max_autos = 62.5 → 
  ferry_weight_pounds = ferry_weight_tons * 2000 → 
  min_auto_weight = ferry_weight_pounds / max_autos → 
  min_auto_weight = 1600 :=
by
  intros
  sorry

end min_weight_of_automobile_l303_303915


namespace possible_values_for_xyz_l303_303569

theorem possible_values_for_xyz:
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
   x + 2 * y = z →
   x^2 - 4 * y^2 + z^2 = 310 →
   (∃ (k : ℕ), k = x * y * z ∧ (k = 11935 ∨ k = 2015))) :=
by
  intros x y z hx hy hz h1 h2
  sorry

end possible_values_for_xyz_l303_303569


namespace f_of_3_l303_303659

def f (x : ℚ) : ℚ := (x + 3) / (x - 6)

theorem f_of_3 : f 3 = -2 := by
  sorry

end f_of_3_l303_303659


namespace girls_more_than_boys_l303_303767

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end girls_more_than_boys_l303_303767


namespace alligators_not_hiding_l303_303473

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75) 
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 :=
by
  -- The proof will go here, which is currently a placeholder.
  sorry

end alligators_not_hiding_l303_303473


namespace polynomial_remainder_l303_303496

theorem polynomial_remainder (P : Polynomial ℝ) (a : ℝ) :
  ∃ (Q : Polynomial ℝ) (r : ℝ), P = Q * (Polynomial.X - Polynomial.C a) + Polynomial.C r ∧ r = (P.eval a) :=
by
  sorry

end polynomial_remainder_l303_303496


namespace no_solution_equation_l303_303399

theorem no_solution_equation (m : ℝ) : 
  ¬∃ x : ℝ, x ≠ 2 ∧ (x - 3) / (x - 2) = m / (2 - x) → m = 1 := 
by 
  sorry

end no_solution_equation_l303_303399


namespace radius_of_circumscribed_sphere_l303_303156

noncomputable def circumscribedSphereRadius (a : ℝ) (α := 60 * Real.pi / 180) : ℝ :=
  5 * a / (4 * Real.sqrt 3)

theorem radius_of_circumscribed_sphere (a : ℝ) :
  circumscribedSphereRadius a = 5 * a / (4 * Real.sqrt 3) := by
  sorry

end radius_of_circumscribed_sphere_l303_303156


namespace angle_between_vectors_eq_2pi_over_3_l303_303816

open Real
open InnerProductSpace

theorem angle_between_vectors_eq_2pi_over_3 (a b : ℝ^3) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∥a∥ = ∥b∥ ∧ ∥a + b∥ = ∥a∥) :
  angle a b = 2 * π / 3 :=
sorry

end angle_between_vectors_eq_2pi_over_3_l303_303816


namespace james_eats_three_l303_303246

variables {p : ℕ} {f : ℕ} {j : ℕ}

-- The initial number of pizza slices
def initial_slices : ℕ := 8

-- The number of slices his friend eats
def friend_slices : ℕ := 2

-- The number of slices left after his friend eats
def remaining_slices : ℕ := initial_slices - friend_slices

-- The number of slices James eats
def james_slices : ℕ := remaining_slices / 2

-- The theorem to prove James eats 3 slices
theorem james_eats_three : james_slices = 3 :=
by
  sorry

end james_eats_three_l303_303246


namespace necessary_and_sufficient_conditions_l303_303419

-- Definitions for sets A and B
def U : Set (ℝ × ℝ) := {p | true}

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

-- Given point P(2, 3)
def P : ℝ × ℝ := (2, 3)

-- Complement of B
def B_complement (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n > 0}

-- Intersection of A and complement of B
def A_inter_B_complement (m n : ℝ) : Set (ℝ × ℝ) := A m ∩ B_complement n

-- Theorem stating the necessary and sufficient conditions for P to belong to A ∩ (complement of B)
theorem necessary_and_sufficient_conditions (m n : ℝ) : 
  P ∈ A_inter_B_complement m n ↔ m > -1 ∧ n < 5 :=
sorry

end necessary_and_sufficient_conditions_l303_303419


namespace only_book_A_l303_303019

variable (numA numB numBoth numOnlyB x : ℕ)
variable (h1 : numA = 2 * numB)
variable (h2 : numBoth = 500)
variable (h3 : numBoth = 2 * numOnlyB)
variable (h4 : numB = numOnlyB + numBoth)
variable (h5 : x = numA - numBoth)

theorem only_book_A : 
  x = 1000 := 
by
  sorry

end only_book_A_l303_303019


namespace equality_of_floor_squares_l303_303325

theorem equality_of_floor_squares (n : ℕ) (hn : 0 < n) :
  (⌊Real.sqrt n + Real.sqrt (n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  (⌊Real.sqrt (4 * n + 1)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  (⌊Real.sqrt (4 * n + 2)⌋ : ℤ) = ⌊Real.sqrt (4 * n + 3)⌋ :=
by
  sorry

end equality_of_floor_squares_l303_303325


namespace rect_area_perimeter_l303_303873

def rect_perimeter (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem rect_area_perimeter (Area Length : ℕ) (hArea : Area = 192) (hLength : Length = 24) :
  ∃ (Width Perimeter : ℕ), Width = Area / Length ∧ Perimeter = rect_perimeter Length Width ∧ Perimeter = 64 :=
by
  sorry

end rect_area_perimeter_l303_303873


namespace log_x2y2_l303_303231

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16 / 11 :=
by
  sorry

end log_x2y2_l303_303231


namespace solve_for_x_l303_303545

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l303_303545


namespace cos_sum_identity_l303_303711

noncomputable theory

open Complex

theorem cos_sum_identity :
  let ω := exp (2 * π * I / 17) in
  ω^17 = 1 →
  (∃ω_conj, ω_conj = conj ω ∧ ω_conj = 1 / ω) →
  cos (2 * π / 17) + cos (6 * π / 17) + cos (8 * π / 17) = (sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l303_303711


namespace number_of_solutions_l303_303068

-- Define the equation and the constraints
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + z = 800

def positive_integer (n : ℕ) : Prop := n > 0

-- The main theorem statement
theorem number_of_solutions : ∃ s, s = 127 ∧ ∀ (x y z : ℕ), positive_integer x → positive_integer y → positive_integer z → equation x y z → s = 127 :=
by
  sorry

end number_of_solutions_l303_303068


namespace simplify_and_rationalize_l303_303273

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l303_303273


namespace vector_decomposition_l303_303586

def x : ℝ×ℝ×ℝ := (8, 0, 5)
def p : ℝ×ℝ×ℝ := (2, 0, 1)
def q : ℝ×ℝ×ℝ := (1, 1, 0)
def r : ℝ×ℝ×ℝ := (4, 1, 2)

theorem vector_decomposition :
  x = (1:ℝ) • p + (-2:ℝ) • q + (2:ℝ) • r :=
by
  sorry

end vector_decomposition_l303_303586


namespace L_shape_count_l303_303198

def L_shape_orientations : List (List (Fin 3 × Fin 3)) :=
  [ [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨1, by simp⟩)],  -- Original orientation
    [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨0, by simp⟩, ⟨1, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩)],  -- 90° rotation
    [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩), (⟨1, by simp⟩, ⟨0, by simp⟩)],  -- 180° rotation
    [(⟨0, by simp⟩, ⟨0, by simp⟩), (⟨0, by simp⟩, ⟨1, by simp⟩), (⟨1, by simp⟩, ⟨1, by simp⟩)] ]-- 270° rotation

def L_shape_in_grid (shape : List (Fin 3 × Fin 3)) : List (Fin 3 × Fin 3) :=
  -- Function to determine if shape is within grid:
  if shape.all (λ (x : Fin 3 × Fin 3), x.1 < 3 ∧ x.2 < 3)
  then shape
  else []

theorem L_shape_count : ∃ n : Nat, n = 48 :=
by { let placements := list.bind L_shape_orientations (λ o, list.map (λ (f : (Fin 3 × Fin 3)), L_shape_in_grid o) [⟨0, by simp⟩, ⟨1, by simp⟩, ⟨2, by simp⟩]),
     have h := placements.length,
     have h48 : h = 48, from sorry,
     exact ⟨h, h48⟩ }

end L_shape_count_l303_303198


namespace rectangles_same_area_l303_303441

theorem rectangles_same_area (x y : ℕ) 
  (h1 : x * y = (x + 4) * (y - 3)) 
  (h2 : x * y = (x + 8) * (y - 4)) : x + y = 10 := 
by
  sorry

end rectangles_same_area_l303_303441


namespace silver_tokens_at_end_l303_303470

theorem silver_tokens_at_end {R B S : ℕ} (x y : ℕ) 
  (hR_init : R = 60) (hB_init : B = 90) 
  (hR_final : R = 60 - 3 * x + y) 
  (hB_final : B = 90 + 2 * x - 4 * y) 
  (h_end_conditions : 0 ≤ R ∧ R < 3 ∧ 0 ≤ B ∧ B < 4) : 
  S = x + y → 
  S = 23 :=
sorry

end silver_tokens_at_end_l303_303470


namespace simplify_expression_l303_303263

theorem simplify_expression (w : ℝ) : 3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 :=
by
  sorry

end simplify_expression_l303_303263


namespace a_left_after_working_days_l303_303900

variable (x : ℕ)  -- x represents the days A worked 

noncomputable def A_work_rate := (1 : ℚ) / 21
noncomputable def B_work_rate := (1 : ℚ) / 28
noncomputable def B_remaining_work := (3 : ℚ) / 4
noncomputable def combined_work_rate := A_work_rate + B_work_rate

theorem a_left_after_working_days 
  (h : combined_work_rate * x + B_remaining_work = 1) : x = 3 :=
by 
  sorry

end a_left_after_working_days_l303_303900


namespace problem_l303_303155

theorem problem :
  ∀ (x y a b : ℝ), 
  |x + y| + |x - y| = 2 → 
  a > 0 → 
  b > 0 → 
  ∀ z : ℝ, 
  z = 4 * a * x + b * y → 
  (∀ (x y : ℝ), |x + y| + |x - y| = 2 → 4 * a * x + b * y ≤ 1) →
  (1 = 4 * a * 1 + b * 1) →
  (1 = 4 * a * (-1) + b * 1) →
  (1 = 4 * a * (-1) + b * (-1)) →
  (1 = 4 * a * 1 + b * (-1)) →
  ∀ a b : ℝ, a > 0 → b > 0 → (1 = 4 * a + b) →
  (a = 1 / 6 ∧ b = 1 / 3) → 
  (1 / a + 1 / b = 9) :=
by
  sorry

end problem_l303_303155


namespace remainder_correct_l303_303945

open Polynomial

noncomputable def polynomial_remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem remainder_correct : polynomial_remainder (X^6 - 2*X^5 + X^4 - X^2 - 2*X + 1)
                                                  ((X^2 - 1)*(X - 2)*(X + 2))
                                                = 2*X^3 - 9*X^2 + 3*X + 2 :=
by
  sorry

end remainder_correct_l303_303945


namespace mangoes_total_l303_303195

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end mangoes_total_l303_303195


namespace bowling_team_avg_weight_l303_303162

noncomputable def total_weight (weights : List ℕ) : ℕ :=
  weights.foldr (· + ·) 0

noncomputable def average_weight (weights : List ℕ) : ℚ :=
  total_weight weights / weights.length

theorem bowling_team_avg_weight :
  let original_weights := [76, 76, 76, 76, 76, 76, 76]
  let new_weights := [110, 60, 85, 65, 100]
  let combined_weights := original_weights ++ new_weights
  average_weight combined_weights = 79.33 := 
by 
  sorry

end bowling_team_avg_weight_l303_303162


namespace max_parrots_l303_303890

-- Define the parameters and conditions for the problem
def N : ℕ := 2018
def Y : ℕ := 1009
def number_of_islanders (R L P : ℕ) := R + L + P = N

-- Define the main theorem
theorem max_parrots (R L P : ℕ) (h : number_of_islanders R L P) (hY : Y = 1009) :
  P = 1009 :=
sorry

end max_parrots_l303_303890


namespace find_x_for_equation_l303_303909

theorem find_x_for_equation :
  ∃ x : ℝ, 9 - 3 / (1 / x) + 3 = 3 ↔ x = 3 := 
by 
  sorry

end find_x_for_equation_l303_303909


namespace base_9_contains_6_or_7_count_l303_303665

def contains_digit_6_or_7 (n : ℕ) : Prop :=
  let digits := (Nat.digits 9 n) in
  List.any digits (λ d, d = 6 ∨ d = 7)

theorem base_9_contains_6_or_7_count :
  ∃ k : ℕ, k = 386 ∧ 
           k = (Finset.filter (λ n, contains_digit_6_or_7 n)
                              (Finset.range 730)).card :=
by
  sorry

end base_9_contains_6_or_7_count_l303_303665


namespace range_of_k_for_quadratic_inequality_l303_303240

theorem range_of_k_for_quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
  sorry

end range_of_k_for_quadratic_inequality_l303_303240


namespace measure_of_angle_y_l303_303806

def is_straight_angle (a : ℝ) := a = 180

theorem measure_of_angle_y (angle_ABC angle_ADB angle_BDA y : ℝ) 
  (h1 : angle_ABC = 117)
  (h2 : angle_ADB = 31)
  (h3 : angle_BDA = 28)
  (h4 : is_straight_angle (angle_ABC + (180 - angle_ABC)))
  : y = 86 := 
by 
  sorry

end measure_of_angle_y_l303_303806


namespace functional_equation_l303_303946

theorem functional_equation 
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : f 0 ≠ 0) :
  f 2009 = 1 :=
sorry

end functional_equation_l303_303946


namespace parking_lot_motorcycles_l303_303844

theorem parking_lot_motorcycles
  (x y : ℕ)
  (h1 : x + y = 24)
  (h2 : 3 * x + 4 * y = 86) : x = 10 :=
by
  sorry

end parking_lot_motorcycles_l303_303844


namespace martin_waste_time_l303_303989

theorem martin_waste_time : 
  let waiting_traffic := 2
  let trying_off_freeway := 4 * waiting_traffic
  let detours := 3 * 30 / 60
  let meal := 45 / 60
  let delays := (20 + 40) / 60
  waiting_traffic + trying_off_freeway + detours + meal + delays = 13.25 := 
by
  sorry

end martin_waste_time_l303_303989


namespace range_of_a_l303_303657

variable (a : ℝ)

def discriminant (a : ℝ) : ℝ := 4 * a ^ 2 - 12

theorem range_of_a
  (h : discriminant a > 0) :
  a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end range_of_a_l303_303657


namespace wrong_observation_value_l303_303149

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean correct_value wrong_value : ℚ) 
  (h₁ : n = 50)
  (h₂ : initial_mean = 36)
  (h₃ : corrected_mean = 36.5)
  (h₄ : correct_value = 60)
  (h₅ : n * corrected_mean = n * initial_mean - wrong_value + correct_value) :
  wrong_value = 35 := by
  have htotal₁ : n * initial_mean = 1800 := by sorry
  have htotal₂ : n * corrected_mean = 1825 := by sorry
  linarith

end wrong_observation_value_l303_303149


namespace A_pow_five_eq_rA_add_sI_l303_303980

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 1; 4, 3]

def I : Matrix (Fin 2) (Fin 2) ℚ :=
  1

theorem A_pow_five_eq_rA_add_sI :
  ∃ (r s : ℚ), (A^5) = r • A + s • I :=
sorry

end A_pow_five_eq_rA_add_sI_l303_303980


namespace concert_total_revenue_l303_303434

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end concert_total_revenue_l303_303434


namespace trig_identity_l303_303353

open Real

theorem trig_identity :
  (1 - 1 / cos (23 * π / 180)) *
  (1 + 1 / sin (67 * π / 180)) *
  (1 - 1 / sin (23 * π / 180)) * 
  (1 + 1 / cos (67 * π / 180)) = 1 :=
by
  sorry

end trig_identity_l303_303353


namespace negate_exactly_one_even_l303_303991

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even (a b c : ℕ) :
  ¬ ((is_even a ∧ is_odd b ∧ is_odd c) ∨ (is_odd a ∧ is_even b ∧ is_odd c) ∨ (is_odd a ∧ is_odd b ∧ is_even c)) ↔ 
  ((is_odd a ∧ is_odd b ∧ is_odd c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c)) :=
sorry

end negate_exactly_one_even_l303_303991


namespace solve_for_x_l303_303546

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l303_303546


namespace log_eqn_l303_303671

theorem log_eqn (a b : ℝ) (h1 : a = (Real.log 400 / Real.log 16))
                          (h2 : b = Real.log 20 / Real.log 2) : a = (1/2) * b :=
sorry

end log_eqn_l303_303671


namespace find_unknown_l303_303594

theorem find_unknown (x : ℝ) :
  300 * 2 + (x + 4) * (1 / 8) = 602 → x = 12 :=
by 
  sorry

end find_unknown_l303_303594


namespace rectangle_area_l303_303244

theorem rectangle_area (AB AC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) : ∃ Area : ℝ, Area = 120 :=
by
  sorry

end rectangle_area_l303_303244


namespace solve_equation_l303_303555

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l303_303555


namespace total_pies_sold_l303_303032

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end total_pies_sold_l303_303032


namespace EquivalenceStatements_l303_303172

-- Define real numbers and sets P, Q
variables {x a b c : ℝ} {P Q : Set ℝ}

-- Prove the necessary equivalences
theorem EquivalenceStatements :
  ((x > 1) → (abs x > 1)) ∧ ((∃ x, x < -1) → (abs x > 1)) ∧
  ((a ∈ P ∩ Q) ↔ (a ∈ P ∧ a ∈ Q)) ∧
  (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (x = 1 ↔ a + b + c = 0) :=
by
  sorry

end EquivalenceStatements_l303_303172


namespace tangent_line_a_value_l303_303676

theorem tangent_line_a_value (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y - 1 = 0 → x^2 + y^2 + 4 * x = 0) → a = -1 / 4 :=
by
  sorry

end tangent_line_a_value_l303_303676


namespace ratio_of_areas_l303_303688

theorem ratio_of_areas (AB BC O : ℝ) (h_diameter : AB = 4) (h_BC : BC = 3)
  (ABD DBE ABDeqDBE : Prop) (x y : ℝ) 
  (h_area_ABCD : x = 7 * y) :
  (x / y) = 7 :=
by
  sorry

end ratio_of_areas_l303_303688


namespace least_integer_k_l303_303674

theorem least_integer_k (k : ℕ) (h : k ^ 3 ∣ 336) : k = 84 :=
sorry

end least_integer_k_l303_303674


namespace sin_x_cos_x_value_l303_303096

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end sin_x_cos_x_value_l303_303096


namespace john_salary_april_l303_303113

theorem john_salary_april 
  (initial_salary : ℤ)
  (raise_percentage : ℤ)
  (cut_percentage : ℤ)
  (bonus : ℤ)
  (february_salary : ℤ)
  (march_salary : ℤ)
  : initial_salary = 3000 →
    raise_percentage = 10 →
    cut_percentage = 15 →
    bonus = 500 →
    february_salary = initial_salary + (initial_salary * raise_percentage / 100) →
    march_salary = february_salary - (february_salary * cut_percentage / 100) →
    march_salary + bonus = 3305 :=
by
  intros
  sorry

end john_salary_april_l303_303113


namespace number_of_small_spheres_l303_303769

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem number_of_small_spheres
  (d_large : ℝ) (d_small : ℝ)
  (h1 : d_large = 6) (h2 : d_small = 2) :
  let V_large := volume_of_sphere (d_large / 2)
  let V_small := volume_of_sphere (d_small / 2)
  V_large / V_small = 27 := 
by
  sorry

end number_of_small_spheres_l303_303769


namespace cumulative_revenue_eq_l303_303151

-- Define the initial box office revenue and growth rate
def initial_revenue : ℝ := 3
def growth_rate (x : ℝ) : ℝ := x

-- Define the cumulative revenue equation after 3 days
def cumulative_revenue (x : ℝ) : ℝ :=
  initial_revenue + initial_revenue * (1 + growth_rate x) + initial_revenue * (1 + growth_rate x) ^ 2

-- State the theorem that proves the equation
theorem cumulative_revenue_eq (x : ℝ) :
  cumulative_revenue x = 10 :=
sorry

end cumulative_revenue_eq_l303_303151


namespace am_gm_inequality_even_sum_l303_303521

theorem am_gm_inequality_even_sum (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h_even : (a + b) % 2 = 0) :
  (a + b : ℚ) / 2 ≥ Real.sqrt (a * b) :=
sorry

end am_gm_inequality_even_sum_l303_303521


namespace smallest_four_digit_divisible_by_4_and_5_l303_303315

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l303_303315


namespace ways_to_select_5_balls_l303_303213

theorem ways_to_select_5_balls (balls : Finset ℕ) (h1 : balls = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) :
  (∑ n in balls.choose 5, if n.sum % 2 = 1 then 1 else 0) = 236 :=
by sorry

end ways_to_select_5_balls_l303_303213


namespace joe_speed_first_part_l303_303693

theorem joe_speed_first_part (v : ℝ) :
  let d1 := 420 -- distance of the first part in miles
  let d2 := 120 -- distance of the second part in miles
  let v2 := 40  -- speed during the second part in miles per hour
  let d_total := d1 + d2 -- total distance
  let avg_speed := 54 -- average speed in miles per hour
  let t1 := d1 / v -- time for the first part
  let t2 := d2 / v2 -- time for the second part
  let t_total := t1 + t2 -- total time
  (d_total / t_total) = avg_speed -> v = 60 :=
by
  intros
  sorry

end joe_speed_first_part_l303_303693


namespace rhind_papyrus_max_bread_l303_303872

theorem rhind_papyrus_max_bread
  (a1 a2 a3 a4 a5 : ℕ) (d : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 100)
  (h2 : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a4 = a1 + 3 * d)
  (h6 : a5 = a1 + 4 * d)
  (h7 : a3 + a4 + a5 = 3 * (a1 + a2)) :
  a5 = 30 :=
by {
  sorry
}

end rhind_papyrus_max_bread_l303_303872


namespace points_on_line_l303_303439

-- Define the points involved
def point1 : ℝ × ℝ := (4, 10)
def point2 : ℝ × ℝ := (-2, -8)
def candidate_points : List (ℝ × ℝ) := [(1, 1), (0, -1), (2, 3), (-1, -5), (3, 7)]
def correct_points : List (ℝ × ℝ) := [(1, 1), (-1, -5), (3, 7)]

-- Define a function to check if a point lies on the line defined by point1 and point2
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (10 - (-8)) / (4 - (-2))
  let b := 10 - m * 4
  p.2 = m * p.1 + b

-- Main theorem statement
theorem points_on_line :
  ∀ p ∈ candidate_points, p ∈ correct_points ↔ lies_on_line p :=
sorry

end points_on_line_l303_303439


namespace cost_of_sneakers_l303_303979

theorem cost_of_sneakers (saved money per_action_figure final_money cost : ℤ) 
  (h1 : saved = 15) 
  (h2 : money = 10) 
  (h3 : per_action_figure = 10) 
  (h4 : final_money = 25) 
  (h5 : money * per_action_figure + saved - cost = final_money) 
  : cost = 90 := 
sorry

end cost_of_sneakers_l303_303979


namespace last_two_digits_of_sum_l303_303310

-- Define factorial, and factorials up to 50 specifically for our problem.
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Sum the last two digits of factorials from 1 to 50
def last_two_digits_sum : ℕ :=
  (fac 1 % 100 + fac 2 % 100 + fac 3 % 100 + fac 4 % 100 + fac 5 % 100 + 
   fac 6 % 100 + fac 7 % 100 + fac 8 % 100 + fac 9 % 100) % 100

theorem last_two_digits_of_sum : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_l303_303310


namespace erased_number_is_one_or_twenty_l303_303328

theorem erased_number_is_one_or_twenty (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 20)
  (h₂ : (210 - x) % 19 = 0) : x = 1 ∨ x = 20 :=
  by sorry

end erased_number_is_one_or_twenty_l303_303328


namespace valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l303_303519

-- Definition: City is divided by roads, and there are initial and additional currency exchange points

structure City := 
(exchange_points : ℕ)   -- Number of exchange points in the city
(parts : ℕ)             -- Number of parts the city is divided into

-- Given: Initial conditions with one existing exchange point and divided parts
def initialCity : City :=
{ exchange_points := 1, parts := 2 }

-- Function to add exchange points in the city
def addExchangePoints (c : City) (new_points : ℕ) : City :=
{ exchange_points := c.exchange_points + new_points, parts := c.parts }

-- Function to verify that each part has exactly two exchange points
def isValidConfiguration (c : City) : Prop :=
c.exchange_points = 2 * c.parts

-- Theorem: Prove that each configuration of new points is valid
theorem valid_first_configuration : 
  isValidConfiguration (addExchangePoints initialCity 3) := 
sorry

theorem valid_second_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_third_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_fourth_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

end valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l303_303519


namespace company_bought_14_02_tons_l303_303912

noncomputable def gravel := 5.91
noncomputable def sand := 8.11
noncomputable def total_material := gravel + sand

theorem company_bought_14_02_tons : total_material = 14.02 :=
by 
  sorry

end company_bought_14_02_tons_l303_303912


namespace number_of_5_digit_numbers_l303_303506

/-- There are 324 five-digit numbers starting with 2 that have exactly three identical digits which are not 2. -/
theorem number_of_5_digit_numbers : ∃ n : ℕ, n = 324 ∧ ∀ (d₁ d₂ : ℕ), 
  (d₁ ≠ 2) ∧ (d₁ ≠ d₂) ∧ (0 ≤ d₁ ∧ d₁ < 10) ∧ (0 ≤ d₂ ∧ d₂ < 10) → 
  n = 4 * 9 * 9 := by
  sorry

end number_of_5_digit_numbers_l303_303506


namespace total_spent_l303_303771

def cost_sandwich : ℕ := 2
def cost_hamburger : ℕ := 2
def cost_hotdog : ℕ := 1
def cost_fruit_juice : ℕ := 2

def selene_sandwiches : ℕ := 3
def selene_fruit_juice : ℕ := 1
def tanya_hamburgers : ℕ := 2
def tanya_fruit_juice : ℕ := 2

def total_selene_spent : ℕ := (selene_sandwiches * cost_sandwich) + (selene_fruit_juice * cost_fruit_juice)
def total_tanya_spent : ℕ := (tanya_hamburgers * cost_hamburger) + (tanya_fruit_juice * cost_fruit_juice)

theorem total_spent : total_selene_spent + total_tanya_spent = 16 := by
  sorry

end total_spent_l303_303771


namespace bananas_distribution_l303_303934

noncomputable def total_bananas : ℝ := 550.5
noncomputable def lydia_bananas : ℝ := 80.25
noncomputable def dawn_bananas : ℝ := lydia_bananas + 93
noncomputable def emily_bananas : ℝ := 198
noncomputable def donna_bananas : ℝ := emily_bananas / 2

theorem bananas_distribution :
  dawn_bananas = 173.25 ∧
  lydia_bananas = 80.25 ∧
  donna_bananas = 99 ∧
  emily_bananas = 198 ∧
  dawn_bananas + lydia_bananas + donna_bananas + emily_bananas = total_bananas :=
by
  sorry

end bananas_distribution_l303_303934


namespace min_ge_n_l303_303532

theorem min_ge_n (x y z n : ℕ) (h : x^n + y^n = z^n) : min x y ≥ n :=
sorry

end min_ge_n_l303_303532


namespace arithmetic_sequence_common_difference_l303_303687

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 1 = 13) (h4 : a_n 4 = 1) : 
  ∃ d : ℤ, d = -4 := by
  sorry

end arithmetic_sequence_common_difference_l303_303687


namespace find_number_l303_303449

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 10) : x = 5 :=
by
  sorry

end find_number_l303_303449


namespace median_eq_range_le_l303_303647

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l303_303647


namespace factorial_product_square_root_square_l303_303617

theorem factorial_product_square_root_square :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 3))^2 = 17280 := 
by
  sorry

end factorial_product_square_root_square_l303_303617


namespace circle_integer_points_l303_303602

theorem circle_integer_points (m n : ℤ) (h : ∃ m n : ℤ, m^2 + n^2 = r ∧ 
  ∃ p q : ℤ, m^2 + n^2 = p ∧ ∃ s t : ℤ, m^2 + n^2 = q ∧ ∃ u v : ℤ, m^2 + n^2 = s ∧ 
  ∃ j k : ℤ, m^2 + n^2 = t ∧ ∃ l w : ℤ, m^2 + n^2 = u ∧ ∃ x y : ℤ, m^2 + n^2 = v ∧ 
  ∃ i b : ℤ, m^2 + n^2 = w ∧ ∃ c d : ℤ, m^2 + n^2 = b ) :
  ∃ r, r = 25 := by
    sorry

end circle_integer_points_l303_303602


namespace noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l303_303382

-- Problem 1: Four-digit numbers with no repeated digits
theorem noRepeatedDigitsFourDigit :
  ∃ (n : ℕ), (n = 120) := sorry

-- Problem 2: Five-digit numbers with no repeated digits and divisible by 5
theorem noRepeatedDigitsFiveDigitDiv5 :
  ∃ (n : ℕ), (n = 216) := sorry

-- Problem 3: Four-digit numbers with no repeated digits and greater than 1325
theorem noRepeatedDigitsFourDigitGreaterThan1325 :
  ∃ (n : ℕ), (n = 181) := sorry

end noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l303_303382


namespace completing_the_square_l303_303011

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end completing_the_square_l303_303011


namespace jordan_buys_rice_l303_303867

variables (r l : ℝ)

theorem jordan_buys_rice
  (price_rice : ℝ := 1.20)
  (price_lentils : ℝ := 0.60)
  (total_pounds : ℝ := 30)
  (total_cost : ℝ := 27.00)
  (eq1 : r + l = total_pounds)
  (eq2 : price_rice * r + price_lentils * l = total_cost) :
  r = 15.0 :=
by
  sorry

end jordan_buys_rice_l303_303867


namespace odd_and_monotonically_decreasing_l303_303777

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

theorem odd_and_monotonically_decreasing :
  is_odd (fun x : ℝ => -x^3) ∧ is_monotonically_decreasing (fun x : ℝ => -x^3) :=
by
  sorry

end odd_and_monotonically_decreasing_l303_303777


namespace exists_y_with_7_coprimes_less_than_20_l303_303299

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1
def connection (a b : ℕ) : ℚ := Nat.lcm a b / (a * b)

theorem exists_y_with_7_coprimes_less_than_20 :
  ∃ y : ℕ, y < 20 ∧ (∃ x : ℕ, connection y x = 1) ∧ (Nat.totient y = 7) :=
by
  sorry

end exists_y_with_7_coprimes_less_than_20_l303_303299


namespace portraits_after_lunch_before_gym_class_l303_303572

-- Define the total number of students in the class
def total_students : ℕ := 24

-- Define the number of students who had their portraits taken before lunch
def students_before_lunch : ℕ := total_students / 3

-- Define the number of students who have not yet had their picture taken after gym class
def students_after_gym_class : ℕ := 6

-- Define the number of students who had their portraits taken before gym class
def students_before_gym_class : ℕ := total_students - students_after_gym_class

-- Define the number of students who had their portraits taken after lunch but before gym class
def students_after_lunch_before_gym_class : ℕ := students_before_gym_class - students_before_lunch

-- Statement of the theorem
theorem portraits_after_lunch_before_gym_class :
  students_after_lunch_before_gym_class = 10 :=
by
  -- The proof is omitted
  sorry

end portraits_after_lunch_before_gym_class_l303_303572


namespace solve_for_x_l303_303234

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 7 = 13) : x = 42 :=
sorry

end solve_for_x_l303_303234


namespace value_of_x2_y2_z2_l303_303120

variable (x y z : ℝ)

theorem value_of_x2_y2_z2 (h1 : x^2 + 3 * y = 4) 
                          (h2 : y^2 - 5 * z = 5) 
                          (h3 : z^2 - 7 * x = -8) : 
                          x^2 + y^2 + z^2 = 20.75 := 
by
  sorry

end value_of_x2_y2_z2_l303_303120


namespace twice_brother_age_l303_303401

theorem twice_brother_age (current_my_age : ℕ) (current_brother_age : ℕ) (years : ℕ) :
  current_my_age = 20 →
  (current_my_age + years) + (current_brother_age + years) = 45 →
  current_my_age + years = 2 * (current_brother_age + years) →
  years = 10 :=
by 
  intros h1 h2 h3
  sorry

end twice_brother_age_l303_303401


namespace scientific_notation_560000_l303_303920

theorem scientific_notation_560000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 560000 = a * 10 ^ n ∧ a = 5.6 ∧ n = 5 :=
by 
  sorry

end scientific_notation_560000_l303_303920


namespace union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l303_303497

open Set

variable (a : ℝ)

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := univ

theorem union_A_B :
  A ∪ B = {x | 1 ≤ x ∧ x ≤ 8} := by
  sorry

theorem compl_A_inter_B :
  (U \ A) ∩ B = {x | 1 ≤ x ∧ x < 2} := by
  sorry

theorem intersection_A_C_not_empty :
  (A ∩ C a ≠ ∅) → a < 8 := by
  sorry

end union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l303_303497


namespace factor_x11_minus_x_l303_303005

theorem factor_x11_minus_x (R : Type*) [CommRing R] : 
  ∃ (p q r s : R[X]), x^11 - x = p * q * r * s :=
by
  sorry

end factor_x11_minus_x_l303_303005


namespace range_of_squared_sum_l303_303813

theorem range_of_squared_sum (x y : ℝ) (h : x^2 + 1 / y^2 = 2) : ∃ z, z = x^2 + y^2 ∧ z ≥ 1 / 2 :=
by
  sorry

end range_of_squared_sum_l303_303813


namespace intersection_of_AB_CD_l303_303404

def point (α : Type*) := (α × α × α)

def A : point ℚ := (5, -8, 9)
def B : point ℚ := (15, -18, 14)
def C : point ℚ := (1, 4, -7)
def D : point ℚ := (3, -4, 11)

def parametric_AB (t : ℚ) : point ℚ :=
  (5 + 10 * t, -8 - 10 * t, 9 + 5 * t)

def parametric_CD (s : ℚ) : point ℚ :=
  (1 + 2 * s, 4 - 8 * s, -7 + 18 * s)

def intersection_point (pi : point ℚ) :=
  ∃ t s : ℚ, parametric_AB t = pi ∧ parametric_CD s = pi

theorem intersection_of_AB_CD : intersection_point (76/15, -118/15, 170/15) :=
  sorry

end intersection_of_AB_CD_l303_303404


namespace value_of_f_5_l303_303177

variable (f : ℕ → ℕ) (x y : ℕ)

theorem value_of_f_5 (h1 : f 2 = 50) (h2 : ∀ x, f x = 2 * x ^ 2 + y) : f 5 = 92 :=
by
  sorry

end value_of_f_5_l303_303177


namespace corrected_mean_l303_303150

open Real

theorem corrected_mean (n : ℕ) (mu_incorrect : ℝ)
                      (x1 y1 x2 y2 x3 y3 : ℝ)
                      (h1 : mu_incorrect = 41)
                      (h2 : n = 50)
                      (h3 : x1 = 48 ∧ y1 = 23)
                      (h4 : x2 = 36 ∧ y2 = 42)
                      (h5 : x3 = 55 ∧ y3 = 28) :
                      ((mu_incorrect * n + (x1 - y1) + (x2 - y2) + (x3 - y3)) / n = 41.92) :=
by
  sorry

end corrected_mean_l303_303150


namespace lucky_lucy_l303_303122

theorem lucky_lucy (a b c d e : ℤ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = 6)
  (hd : d = 8)
  (he : a + b - c + d - e = a + (b - (c + (d - e)))) :
  e = 8 :=
by
  rw [ha, hb, hc, hd] at he
  exact eq_of_sub_eq_zero (by linarith)

end lucky_lucy_l303_303122


namespace train_length_l303_303469

theorem train_length (L S : ℝ) 
  (h1 : L = S * 40) 
  (h2 : L + 1800 = S * 120) : 
  L = 900 := 
by
  sorry

end train_length_l303_303469


namespace cubic_roots_c_over_d_l303_303297

theorem cubic_roots_c_over_d (a b c d : ℤ) (h : a ≠ 0)
  (h_roots : ∃ r1 r2 r3, r1 = -1 ∧ r2 = 3 ∧ r3 = 4 ∧ 
              a * r1 * r2 * r3 + b * (r1 * r2 + r2 * r3 + r3 * r1) + c * (r1 + r2 + r3) + d = 0)
  : (c : ℚ) / d = 5 / 12 := 
sorry

end cubic_roots_c_over_d_l303_303297


namespace geometric_series_common_ratio_l303_303886

theorem geometric_series_common_ratio (a : ℝ) (r : ℝ) (S : ℝ) (h1 : S = a / (1 - r))
  (h2 : S = 16 * (r^2 * S)) : |r| = 1/4 :=
by
  sorry

end geometric_series_common_ratio_l303_303886


namespace find_A_in_terms_of_B_and_C_l303_303418

noncomputable def f (A B : ℝ) (x : ℝ) := A * x - 3 * B^2
noncomputable def g (B C : ℝ) (x : ℝ) := B * x + C

theorem find_A_in_terms_of_B_and_C (A B C : ℝ) (h : B ≠ 0) (h1 : f A B (g B C 1) = 0) : A = 3 * B^2 / (B + C) :=
by sorry

end find_A_in_terms_of_B_and_C_l303_303418


namespace multiplier_condition_l303_303239

theorem multiplier_condition (a b : ℚ) (h : a * b ≤ b) : (b ≥ 0 ∧ a ≤ 1) ∨ (b ≤ 0 ∧ a ≥ 1) :=
by 
  sorry

end multiplier_condition_l303_303239


namespace chives_planted_l303_303115

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end chives_planted_l303_303115


namespace right_triangle_condition_l303_303750

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end right_triangle_condition_l303_303750


namespace parallel_lines_minimum_distance_l303_303974

theorem parallel_lines_minimum_distance :
  ∀ (m n : ℝ) (k : ℝ), 
  k = 2 ∧ ∀ (L1 L2 : ℝ → ℝ), -- we define L1 and L2 as functions
  (L1 = λ y => 2 * y + 3) ∧ (L2 = λ y => k * y - 1) ∧ 
  ((L1 n = m) ∧ (L2 (n + k) = m + 2)) → 
  dist (m, n) (m + 2, n + 2) = 2 * Real.sqrt 2 := 
sorry

end parallel_lines_minimum_distance_l303_303974


namespace Faye_apps_left_l303_303060

theorem Faye_apps_left (total_apps gaming_apps utility_apps deleted_gaming_apps deleted_utility_apps remaining_apps : ℕ)
  (h1 : total_apps = 12) 
  (h2 : gaming_apps = 5) 
  (h3 : utility_apps = total_apps - gaming_apps) 
  (h4 : remaining_apps = total_apps - (deleted_gaming_apps + deleted_utility_apps))
  (h5 : deleted_gaming_apps = gaming_apps) 
  (h6 : deleted_utility_apps = 3) : 
  remaining_apps = 4 :=
by
  sorry

end Faye_apps_left_l303_303060


namespace average_weight_decrease_l303_303876

theorem average_weight_decrease 
  (weight_old_student : ℝ := 92) 
  (weight_new_student : ℝ := 72) 
  (number_of_students : ℕ := 5) : 
  (weight_old_student - weight_new_student) / ↑number_of_students = 4 :=
by 
  sorry

end average_weight_decrease_l303_303876


namespace sides_of_length_five_l303_303355

theorem sides_of_length_five (GH HI : ℝ) (L : ℝ) (total_perimeter : ℝ) :
  GH = 7 → HI = 5 → total_perimeter = 38 → (∃ n m : ℕ, n + m = 6 ∧ n * 7 + m * 5 = 38 ∧ m = 2) := by
  intros hGH hHI hPerimeter
  sorry

end sides_of_length_five_l303_303355


namespace integer_solutions_l303_303943

theorem integer_solutions :
  ∀ (m n : ℤ), (m^3 - n^3 = 2 * m * n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2)) :=
by
  intros m n
  sorry

end integer_solutions_l303_303943


namespace largest_three_digit_congruent_to_twelve_mod_fifteen_l303_303742

theorem largest_three_digit_congruent_to_twelve_mod_fifteen :
  ∃ n : ℕ, 100 ≤ 15 * n + 12 ∧ 15 * n + 12 < 1000 ∧ (15 * n + 12 = 987) :=
sorry

end largest_three_digit_congruent_to_twelve_mod_fifteen_l303_303742


namespace find_abc_l303_303855

theorem find_abc (a b c : ℝ)
  (h1 : ∀ x : ℝ, (x < -6 ∨ (|x - 31| ≤ 1)) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h2 : a < b) :
  a + 2 * b + 3 * c = 76 :=
sorry

end find_abc_l303_303855


namespace total_friends_met_l303_303123

def num_friends_with_pears : Nat := 9
def num_friends_with_oranges : Nat := 6

theorem total_friends_met : num_friends_with_pears + num_friends_with_oranges = 15 :=
by
  sorry

end total_friends_met_l303_303123


namespace total_amount_received_l303_303324

theorem total_amount_received (P R CI: ℝ) (T: ℕ) 
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (P_eq: P = 2828.80 / 0.1664) 
  (R_eq: R = 8) 
  (T_eq: T = 2) : 
  P + CI = 19828.80 := 
by 
  sorry

end total_amount_received_l303_303324


namespace probability_cd_l303_303026

theorem probability_cd (P_A P_B : ℚ) (h1 : P_A = 1/4) (h2 : P_B = 1/3) :
  (1 - P_A - P_B = 5/12) :=
by
  -- Placeholder for the proof
  sorry

end probability_cd_l303_303026


namespace bike_cost_l303_303975

-- Defining the problem conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def leftover : ℚ := 20  -- 20 dollars left over
def quarter_value : ℚ := 0.25

-- Define the total quarters Jenn has
def total_quarters := jars * quarters_per_jar

-- Define the total amount of money from quarters
def total_money_quarters := total_quarters * quarter_value

-- Prove that the cost of the bike is $200
theorem bike_cost : total_money_quarters + leftover - 20 = 200 :=
sorry

end bike_cost_l303_303975


namespace number_of_sides_l303_303056

-- Define the conditions
def interior_angle (n : ℕ) : ℝ := 156

-- The main theorem to prove the number of sides
theorem number_of_sides (n : ℕ) (h : interior_angle n = 156) : n = 15 :=
by
  sorry

end number_of_sides_l303_303056


namespace discount_percentage_of_sale_l303_303892

theorem discount_percentage_of_sale (initial_price sale_coupon saved_amount final_price : ℝ)
    (h1 : initial_price = 125)
    (h2 : sale_coupon = 10)
    (h3 : saved_amount = 44)
    (h4 : final_price = 81) :
    ∃ x : ℝ, x = 0.20 ∧ 
             (initial_price - initial_price * x - sale_coupon) - 
             0.10 * (initial_price - initial_price * x - sale_coupon) = final_price :=
by
  -- Proof should be constructed here
  sorry

end discount_percentage_of_sale_l303_303892


namespace solve_for_x_l303_303556

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l303_303556


namespace nathan_dice_roll_probability_l303_303529

noncomputable def probability_nathan_rolls : ℚ :=
  let prob_less4_first_die : ℚ := 3 / 8
  let prob_greater5_second_die : ℚ := 3 / 8
  prob_less4_first_die * prob_greater5_second_die

theorem nathan_dice_roll_probability : probability_nathan_rolls = 9 / 64 := by
  sorry

end nathan_dice_roll_probability_l303_303529


namespace binary_multiplication_l303_303630

theorem binary_multiplication : (10101 : ℕ) * (101 : ℕ) = 1101001 :=
by sorry

end binary_multiplication_l303_303630


namespace max_sum_of_factors_l303_303119

theorem max_sum_of_factors (heartsuit spadesuit : ℕ) (h : heartsuit * spadesuit = 24) :
  heartsuit + spadesuit ≤ 25 :=
sorry

end max_sum_of_factors_l303_303119


namespace hiker_displacement_l303_303335

theorem hiker_displacement :
  let start_point := (0, 0)
  let move_east := (24, 0)
  let move_north := (0, 20)
  let move_west := (-7, 0)
  let move_south := (0, -9)
  let final_position := (start_point.1 + move_east.1 + move_west.1, start_point.2 + move_north.2 + move_south.2)
  let distance_from_start := Real.sqrt (final_position.1^2 + final_position.2^2)
  distance_from_start = Real.sqrt 410
:= by 
  sorry

end hiker_displacement_l303_303335


namespace mono_increasing_necessary_not_sufficient_problem_statement_l303_303857

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- Define the first condition of p: f(x) is monotonically increasing in (-∞, +∞)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Define the second condition q: m > 4/3
def m_gt_4_over_3 (m : ℝ) : Prop := m > 4/3

-- State the theorem: 
theorem mono_increasing_necessary_not_sufficient (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) := 
by
  sorry

-- Main theorem tying the conditions to the conclusion
theorem problem_statement (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) :=
  by sorry

end mono_increasing_necessary_not_sufficient_problem_statement_l303_303857


namespace indeterminate_equation_solution_l303_303999

theorem indeterminate_equation_solution (x y : ℝ) (n : ℕ) :
  (x^2 + (x + 1)^2 = y^2) ↔ 
  (x = 1/4 * ((1 + Real.sqrt 2)^(2*n + 1) + (1 - Real.sqrt 2)^(2*n + 1) - 2) ∧ 
   y = 1/(2 * Real.sqrt 2) * ((1 + Real.sqrt 2)^(2*n + 1) - (1 - Real.sqrt 2)^(2*n + 1))) := 
sorry

end indeterminate_equation_solution_l303_303999


namespace shortest_player_height_correct_l303_303158

def tallest_player_height : Real := 77.75
def height_difference : Real := 9.5
def shortest_player_height : Real := 68.25

theorem shortest_player_height_correct :
  tallest_player_height - height_difference = shortest_player_height :=
by
  sorry

end shortest_player_height_correct_l303_303158


namespace find_f_of_one_third_l303_303672

-- Define g function according to given condition
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f function according to given condition, valid for x ≠ 0
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x

-- State the theorem we need to prove
theorem find_f_of_one_third : f (1 / 3) = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end find_f_of_one_third_l303_303672


namespace tan_half_angle_sin_cos_expression_l303_303326

-- Proof Problem 1: If α is an angle in the third quadrant and sin α = -5/13, then tan (α / 2) = -5.
theorem tan_half_angle (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -5 := 
by 
  sorry

-- Proof Problem 2: If tan α = 2, then sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5.
theorem sin_cos_expression (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 :=
by 
  sorry

end tan_half_angle_sin_cos_expression_l303_303326


namespace find_a_b_monotonicity_l303_303826

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (x^2 + a * x + b) / x

theorem find_a_b (a b : ℝ) (h_odd : ∀ x ≠ 0, f (-x) a b = -f x a b) (h_eq : f 1 a b = f 4 a b) :
  a = 0 ∧ b = 4 := by sorry

theorem monotonicity (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = x + 4 / x) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 2 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2) ∧
  (∀ x1 x2, 2 < x1 ∧ x1 < x2 → f x1 < f x2) := by sorry

end find_a_b_monotonicity_l303_303826


namespace tips_fraction_to_salary_l303_303347

theorem tips_fraction_to_salary (S T I : ℝ)
  (h1 : I = S + T)
  (h2 : T / I = 0.6923076923076923) :
  T / S = 2.25 := by
  sorry

end tips_fraction_to_salary_l303_303347


namespace area_of_folded_shape_is_two_units_squared_l303_303344

/-- 
A square piece of paper with each side of length 2 units is divided into 
four equal squares along both its length and width. From the top left corner to 
bottom right corner, a line is drawn through the center dividing the square diagonally.
The paper is folded along this line to form a new shape.
We prove that the area of the folded shape is 2 units².
-/
theorem area_of_folded_shape_is_two_units_squared
  (side_len : ℝ)
  (area_original : ℝ)
  (area_folded : ℝ)
  (h1 : side_len = 2)
  (h2 : area_original = side_len * side_len)
  (h3 : area_folded = area_original / 2) :
  area_folded = 2 := by
  -- Place proof here
  sorry

end area_of_folded_shape_is_two_units_squared_l303_303344


namespace rationalize_denominator_l303_303268

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l303_303268


namespace average_age_first_and_fifth_dogs_l303_303580

-- Define the conditions
def first_dog_age : ℕ := 10
def second_dog_age : ℕ := first_dog_age - 2
def third_dog_age : ℕ := second_dog_age + 4
def fourth_dog_age : ℕ := third_dog_age / 2
def fifth_dog_age : ℕ := fourth_dog_age + 20

-- Define the goal statement
theorem average_age_first_and_fifth_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 :=
by
  sorry

end average_age_first_and_fifth_dogs_l303_303580


namespace max_min_diff_value_l303_303856

noncomputable def max_min_diff_c (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : ℝ :=
  (10 / 3) - (-2)

theorem max_min_diff_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : 
  max_min_diff_c a b c h1 h2 = 16 / 3 := 
by 
  sorry

end max_min_diff_value_l303_303856


namespace total_flour_used_l303_303013

def wheat_flour : ℝ := 0.2
def white_flour : ℝ := 0.1

theorem total_flour_used : wheat_flour + white_flour = 0.3 :=
by
  sorry

end total_flour_used_l303_303013


namespace quadratic_factor_n_l303_303814

theorem quadratic_factor_n (n : ℤ) (h : ∃ m : ℤ, (x + 5) * (x + m) = x^2 + 7 * x + n) : n = 10 :=
sorry

end quadratic_factor_n_l303_303814


namespace intersection_point_at_neg4_l303_303730

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end intersection_point_at_neg4_l303_303730


namespace alcohol_quantity_l303_303591

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 4 / 3) (h2 : A / (W + 8) = 4 / 5) : A = 16 := 
by
  sorry

end alcohol_quantity_l303_303591


namespace functional_relationship_minimum_wage_l303_303710

/-- Problem setup and conditions --/
def total_area : ℝ := 1200
def team_A_rate : ℝ := 100
def team_B_rate : ℝ := 50
def team_A_wage : ℝ := 4000
def team_B_wage : ℝ := 3000
def min_days_A : ℝ := 3

/-- Prove Part 1: y as a function of x --/
def y_of_x (x : ℝ) : ℝ := 24 - 2 * x

theorem functional_relationship (x : ℝ) :
  100 * x + 50 * y_of_x x = total_area := by
  sorry

/-- Prove Part 2: Minimum wage calculation --/
def total_wage (a b : ℝ) : ℝ := team_A_wage * a + team_B_wage * b

theorem minimum_wage :
  ∀ (a b : ℝ), 3 ≤ a → a ≤ b → b = 24 - 2 * a → 
  total_wage a b = 56000 → a = 8 ∧ b = 8 := by
  sorry

end functional_relationship_minimum_wage_l303_303710


namespace cos_sum_identity_l303_303713

theorem cos_sum_identity :
  cos (2 * Real.pi / 17) + cos (6 * Real.pi / 17) + cos (8 * Real.pi / 17) = (Real.sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_identity_l303_303713


namespace part_a_part_b_l303_303321

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k  

-- Statement for Part (a)
theorem part_a (n : ℕ) :
  (∑ k in (finset.range (n.div 2 + 1)).filter (λ k, k % 2 = 0), (-1)^(k/2) * binom n k) = 2^(n/2) * real.cos (n * real.pi / 4) := 
sorry

-- Statement for Part (b)
theorem part_b (n : ℕ) :
  (∑ k in (finset.range (n.div 2 + 1)).filter (λ k, k % 2 = 1), (-1)^((k-1)/2) * binom n k) = 2^(n/2) * real.sin (n * real.pi / 4) := 
sorry

end part_a_part_b_l303_303321


namespace odd_function_equiv_l303_303398

noncomputable def odd_function (f : ℝ → ℝ) :=
∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_equiv (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f (x)) ↔ (∀ x : ℝ, f (-(-x)) = -f (-x)) :=
by
  sorry

end odd_function_equiv_l303_303398


namespace coeff_x6_in_expansion_l303_303627

noncomputable def coefficient_x6_expansion : Prop :=
  let p := (1 + 3 * Polynomial.X - Polynomial.X ^ 2) ^ 5
  in Polynomial.coeff p 6 = -370

-- statement without proof
theorem coeff_x6_in_expansion : coefficient_x6_expansion :=
  sorry

end coeff_x6_in_expansion_l303_303627


namespace fish_to_apples_l303_303970

variables (f l r a : ℝ)

theorem fish_to_apples (h1 : 3 * f = 2 * l) (h2 : l = 5 * r) (h3 : l = 3 * a) : f = 2 * a :=
by
  -- We assume the conditions as hypotheses and aim to prove the final statement
  sorry

end fish_to_apples_l303_303970


namespace decreasing_implies_bound_l303_303232

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_implies_bound (b : ℝ) :
  (∀ x > 2, -x + b / x ≤ 0) → b ≤ 4 :=
  sorry

end decreasing_implies_bound_l303_303232


namespace consecutive_even_legs_sum_l303_303148

theorem consecutive_even_legs_sum (x : ℕ) (h : x % 2 = 0) (hx : x ^ 2 + (x + 2) ^ 2 = 34 ^ 2) : x + (x + 2) = 48 := by
  sorry

end consecutive_even_legs_sum_l303_303148


namespace plane_divides_pyramid_l303_303904

noncomputable def volume_of_parts (a h KL KK1: ℝ): ℝ × ℝ :=
  -- Define the pyramid and prism structure and the conditions
  let volume_total := (1/3) * (a^2) * h
  let volume_part1 := 512/15
  let volume_part2 := volume_total - volume_part1
  (⟨volume_part1, volume_part2⟩ : ℝ × ℝ)

theorem plane_divides_pyramid (a h KL KK1: ℝ) 
  (h₁ : a = 8 * Real.sqrt 2) 
  (h₂ : h = 4) 
  (h₃ : KL = 2) 
  (h₄ : KK1 = 1):
  volume_of_parts a h KL KK1 = (512/15, 2048/15) := 
by 
  sorry

end plane_divides_pyramid_l303_303904


namespace eval_expr_equals_1_l303_303137

noncomputable def eval_expr (a b : ℕ) : ℚ :=
  (a + b) / (a * b) / ((a / b) - (b / a))

theorem eval_expr_equals_1 (a b : ℕ) (h₁ : a = 3) (h₂ : b = 2) : eval_expr a b = 1 :=
by
  sorry

end eval_expr_equals_1_l303_303137


namespace number_of_possible_values_of_x_l303_303390

theorem number_of_possible_values_of_x : 
  (∃ x : ℕ, ⌈Real.sqrt x⌉ = 12) → (set.Ico 144 169).card = 25 := 
by
  intros h
  sorry

end number_of_possible_values_of_x_l303_303390


namespace sequence_x_value_l303_303258

theorem sequence_x_value (x : ℕ) (h1 : 3 - 1 = 2) (h2 : 6 - 3 = 3) (h3 : 10 - 6 = 4) (h4 : x - 10 = 5) : x = 15 :=
by
  sorry

end sequence_x_value_l303_303258


namespace number_of_acute_triangles_l303_303778

def num_triangles : ℕ := 7
def right_triangles : ℕ := 2
def obtuse_triangles : ℕ := 3

theorem number_of_acute_triangles :
  num_triangles - right_triangles - obtuse_triangles = 2 := by
  sorry

end number_of_acute_triangles_l303_303778


namespace balls_into_boxes_l303_303667

noncomputable def countDistributions : ℕ :=
  sorry

theorem balls_into_boxes :
  countDistributions = 8 :=
  sorry

end balls_into_boxes_l303_303667


namespace nancy_spelling_problems_l303_303947

structure NancyProblems where
  math_problems : ℝ
  rate : ℝ
  hours : ℝ
  total_problems : ℝ

noncomputable def calculate_spelling_problems (n : NancyProblems) : ℝ :=
  n.total_problems - n.math_problems

theorem nancy_spelling_problems :
  ∀ (n : NancyProblems), n.math_problems = 17.0 ∧ n.rate = 8.0 ∧ n.hours = 4.0 ∧ n.total_problems = 32.0 →
  calculate_spelling_problems n = 15.0 :=
by
  intros
  sorry

end nancy_spelling_problems_l303_303947


namespace six_digit_mod7_l303_303242

theorem six_digit_mod7 (a b c d e f : ℕ) (N : ℕ) (h : N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) (h_div7 : N % 7 = 0) :
    (10^5 * f + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e) % 7 = 0 :=
by
  sorry

end six_digit_mod7_l303_303242


namespace find_a_100_l303_303503

noncomputable def a : Nat → Nat
| 0 => 0
| 1 => 2
| (n+1) => a n + 2 * n

theorem find_a_100 : a 100 = 9902 := 
  sorry

end find_a_100_l303_303503


namespace average_age_of_cricket_team_l303_303758

theorem average_age_of_cricket_team
  (A : ℝ)
  (captain_age : ℝ) (wicket_keeper_age : ℝ)
  (team_size : ℕ) (remaining_players : ℕ)
  (captain_age_eq : captain_age = 24)
  (wicket_keeper_age_eq : wicket_keeper_age = 27)
  (remaining_players_eq : remaining_players = team_size - 2)
  (average_age_condition : (team_size * A - (captain_age + wicket_keeper_age)) = remaining_players * (A - 1)) : 
  A = 21 := by
  sorry

end average_age_of_cricket_team_l303_303758


namespace employees_in_room_l303_303574

-- Define variables
variables (E : ℝ) (M : ℝ) (L : ℝ)

-- Given conditions
def condition1 : Prop := M = 0.99 * E
def condition2 : Prop := (M - L) / E = 0.98
def condition3 : Prop := L = 99.99999999999991

-- Prove statement
theorem employees_in_room (h1 : condition1 E M) (h2 : condition2 E M L) (h3 : condition3 L) : E = 10000 :=
by
  sorry

end employees_in_room_l303_303574


namespace calc_radical_power_l303_303201

theorem calc_radical_power : (Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 16))) ^ 12) = 4096 := sorry

end calc_radical_power_l303_303201


namespace value_of_y_l303_303395

theorem value_of_y (y : ℝ) (α : ℝ) (h₁ : (-3, y) = (x, y)) (h₂ : Real.sin α = -3 / 4) : 
  y = -9 * Real.sqrt 7 / 7 := 
  sorry

end value_of_y_l303_303395


namespace diana_total_cost_l303_303043

noncomputable def shopping_total_cost := 
  let t_shirt_price := 10
  let sweater_price := 25
  let jacket_price := 100
  let jeans_price := 40
  let shoes_price := 70 

  let t_shirt_discount := 0.20
  let sweater_discount := 0.10
  let jacket_discount := 0.15
  let jeans_discount := 0.05
  let shoes_discount := 0.25

  let clothes_tax := 0.06
  let shoes_tax := 0.09

  let t_shirt_qty := 8
  let sweater_qty := 5
  let jacket_qty := 3
  let jeans_qty := 6
  let shoes_qty := 4

  let t_shirt_total := t_shirt_qty * t_shirt_price 
  let sweater_total := sweater_qty * sweater_price 
  let jacket_total := jacket_qty * jacket_price 
  let jeans_total := jeans_qty * jeans_price 
  let shoes_total := shoes_qty * shoes_price 

  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let sweater_discounted := sweater_total * (1 - sweater_discount)
  let jacket_discounted := jacket_total * (1 - jacket_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let shoes_discounted := shoes_total * (1 - shoes_discount)

  let t_shirt_final := t_shirt_discounted * (1 + clothes_tax)
  let sweater_final := sweater_discounted * (1 + clothes_tax)
  let jacket_final := jacket_discounted * (1 + clothes_tax)
  let jeans_final := jeans_discounted * (1 + clothes_tax)
  let shoes_final := shoes_discounted * (1 + shoes_tax)

  t_shirt_final + sweater_final + jacket_final + jeans_final + shoes_final

theorem diana_total_cost : shopping_total_cost = 927.97 :=
by sorry

end diana_total_cost_l303_303043


namespace numWaysToChoosePairs_is_15_l303_303668

def numWaysToChoosePairs : ℕ :=
  let white := Nat.choose 5 2
  let brown := Nat.choose 3 2
  let blue := Nat.choose 2 2
  let black := Nat.choose 2 2
  white + brown + blue + black

theorem numWaysToChoosePairs_is_15 : numWaysToChoosePairs = 15 := by
  -- We will prove this theorem in actual proof
  sorry

end numWaysToChoosePairs_is_15_l303_303668


namespace minimum_square_side_length_l303_303987

theorem minimum_square_side_length (s : ℝ) (h1 : s^2 ≥ 625) (h2 : ∃ (t : ℝ), t = s / 2) : s = 25 :=
by
  sorry

end minimum_square_side_length_l303_303987


namespace calculate_Y_payment_l303_303577

-- Define the known constants
def total_payment : ℝ := 590
def x_to_y_ratio : ℝ := 1.2

-- Main theorem statement, asserting the value of Y's payment
theorem calculate_Y_payment (Y : ℝ) (X : ℝ) 
  (h1 : X = x_to_y_ratio * Y) 
  (h2 : X + Y = total_payment) : 
  Y = 268.18 :=
by
  sorry

end calculate_Y_payment_l303_303577


namespace investment_ratio_l303_303320

theorem investment_ratio (X_investment Y_investment : ℕ) (hX : X_investment = 5000) (hY : Y_investment = 15000) : 
  X_investment * 3 = Y_investment :=
by
  sorry

end investment_ratio_l303_303320


namespace smallest_natural_number_exists_l303_303631

theorem smallest_natural_number_exists (n : ℕ) : (∃ n, ∃ a b c : ℕ, n = 15 ∧ 1998 = a * (5 ^ 4) + b * (3 ^ 4) + c * (1 ^ 4) ∧ a + b + c = 15) :=
sorry

end smallest_natural_number_exists_l303_303631


namespace highest_probability_two_out_of_three_probability_l303_303212

structure Student :=
  (name : String)
  (P_T : ℚ)  -- Probability of passing the theoretical examination
  (P_S : ℚ)  -- Probability of passing the social practice examination

noncomputable def P_earn (student : Student) : ℚ :=
  student.P_T * student.P_S

def student_A := Student.mk "A" (5 / 6) (1 / 2)
def student_B := Student.mk "B" (4 / 5) (2 / 3)
def student_C := Student.mk "C" (3 / 4) (5 / 6)

theorem highest_probability : 
  P_earn student_C > P_earn student_B ∧ P_earn student_B > P_earn student_A :=
by sorry

theorem two_out_of_three_probability :
  (1 - P_earn student_A) * P_earn student_B * P_earn student_C +
  P_earn student_A * (1 - P_earn student_B) * P_earn student_C +
  P_earn student_A * P_earn student_B * (1 - P_earn student_C) =
  115 / 288 :=
by sorry

end highest_probability_two_out_of_three_probability_l303_303212


namespace ac_bd_bound_l303_303498

variables {a b c d : ℝ}

theorem ac_bd_bound (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 4) : |a * c + b * d| ≤ 2 := 
sorry

end ac_bd_bound_l303_303498


namespace simplify_rationalize_denominator_l303_303281

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l303_303281


namespace books_sold_in_store_on_saturday_l303_303600

namespace BookshopInventory

def initial_inventory : ℕ := 743
def saturday_online_sales : ℕ := 128
def sunday_online_sales : ℕ := 162
def shipment_received : ℕ := 160
def final_inventory : ℕ := 502

-- Define the total number of books sold
def total_books_sold (S : ℕ) : ℕ := S + saturday_online_sales + 2 * S + sunday_online_sales

-- Net change in inventory equals total books sold minus shipment received
def net_change_in_inventory (S : ℕ) : ℕ := total_books_sold S - shipment_received

-- Prove that the difference between initial and final inventories equals the net change in inventory
theorem books_sold_in_store_on_saturday : ∃ S : ℕ, net_change_in_inventory S = initial_inventory - final_inventory ∧ S = 37 :=
by
  sorry

end BookshopInventory

end books_sold_in_store_on_saturday_l303_303600


namespace line_bisects_circle_l303_303905

theorem line_bisects_circle (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ x - y = 0) → 
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 = 1) → 
  ∀ x y : ℝ, (x - y = 0) ∨ (x + y = 0) → l x y ∧ C x y → l x y = (x - y = 0) := by
  sorry

end line_bisects_circle_l303_303905


namespace avg_student_headcount_l303_303169

def student_headcount (yr1 yr2 yr3 yr4 : ℕ) : ℕ :=
  (yr1 + yr2 + yr3 + yr4) / 4

theorem avg_student_headcount :
  student_headcount 10600 10800 10500 10400 = 10825 :=
by
  sorry

end avg_student_headcount_l303_303169


namespace find_a_l303_303854

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

theorem find_a (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = 1 / 2 ∨ a = 1 / 3) := by
  sorry

end find_a_l303_303854


namespace prob1_part1_prob1_part2_l303_303859

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2 * a}

theorem prob1_part1 (a : ℝ) (ha : a = 3) :
  A ∪ B a = {x | -2 < x ∧ x < 7} ∧ A ∩ B a = {x | -1 < x ∧ x < 5} :=
by {
  sorry
}

theorem prob1_part2 (h : ∀ x, x ∈ A → x ∈ B a) :
  ∀ a : ℝ, a ≤ 2 :=
by {
  sorry
}

end prob1_part1_prob1_part2_l303_303859


namespace average_age_of_dogs_l303_303583

theorem average_age_of_dogs:
  let age1 := 10 in
  let age2 := age1 - 2 in
  let age3 := age2 + 4 in
  let age4 := age3 / 2 in
  let age5 := age4 + 20 in
  (age1 + age5) / 2 = 18 :=
by 
  sorry

end average_age_of_dogs_l303_303583


namespace concert_total_revenue_l303_303433

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end concert_total_revenue_l303_303433


namespace sin_neg_1740_eq_sqrt3_div_2_l303_303161

theorem sin_neg_1740_eq_sqrt3_div_2 : Real.sin (-1740 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_1740_eq_sqrt3_div_2_l303_303161


namespace pair_C_product_not_36_l303_303319

-- Definitions of the pairs
def pair_A : ℤ × ℤ := (-4, -9)
def pair_B : ℤ × ℤ := (-3, -12)
def pair_C : ℚ × ℚ := (1/2, -72)
def pair_D : ℤ × ℤ := (1, 36)
def pair_E : ℚ × ℚ := (3/2, 24)

-- Mathematical statement for the proof problem
theorem pair_C_product_not_36 :
  pair_C.fst * pair_C.snd ≠ 36 :=
by
  sorry

end pair_C_product_not_36_l303_303319


namespace alternative_plan_cost_is_eleven_l303_303797

-- Defining current cost
def current_cost : ℕ := 12

-- Defining the alternative plan cost in terms of current cost
def alternative_cost : ℕ := current_cost - 1

-- Theorem stating the alternative cost is $11
theorem alternative_plan_cost_is_eleven : alternative_cost = 11 :=
by
  -- This is the proof, which we are skipping with sorry
  sorry

end alternative_plan_cost_is_eleven_l303_303797


namespace shaded_percentage_l303_303747

-- Definition for the six-by-six grid and total squares
def total_squares : ℕ := 36
def shaded_squares : ℕ := 16

-- Definition of the problem: to prove the percentage of shaded squares
theorem shaded_percentage : (shaded_squares : ℚ) / total_squares * 100 = 44.4 :=
by
  sorry

end shaded_percentage_l303_303747


namespace solve_for_x_l303_303537

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l303_303537


namespace quadratic_completion_l303_303881

theorem quadratic_completion (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 26 * x + 81 = (x + b)^2 + c) → b + c = -101 :=
by 
  intro h
  sorry

end quadratic_completion_l303_303881


namespace oranges_left_uneaten_l303_303993

variable (total_oranges : ℕ)
variable (half_oranges ripe_oranges unripe_oranges eaten_ripe_oranges eaten_unripe_oranges uneaten_ripe_oranges uneaten_unripe_oranges total_uneaten_oranges : ℕ)

axiom h1 : total_oranges = 96
axiom h2 : half_oranges = total_oranges / 2
axiom h3 : ripe_oranges = half_oranges
axiom h4 : unripe_oranges = half_oranges
axiom h5 : eaten_ripe_oranges = ripe_oranges / 4
axiom h6 : eaten_unripe_oranges = unripe_oranges / 8
axiom h7 : uneaten_ripe_oranges = ripe_oranges - eaten_ripe_oranges
axiom h8 : uneaten_unripe_oranges = unripe_oranges - eaten_unripe_oranges
axiom h9 : total_uneaten_oranges = uneaten_ripe_oranges + uneaten_unripe_oranges

theorem oranges_left_uneaten : total_uneaten_oranges = 78 := by
  sorry

end oranges_left_uneaten_l303_303993


namespace highest_power_of_3_divides_N_l303_303286

-- Define the range of two-digit numbers and the concatenation function
def concatTwoDigitIntegers : ℕ := sorry  -- Placeholder for the concatenation implementation

-- Integer N formed by concatenating integers from 31 to 68
def N := concatTwoDigitIntegers

-- The statement proving the highest power of 3 dividing N is 3^1
theorem highest_power_of_3_divides_N :
  (∃ k : ℕ, 3^k ∣ N ∧ ¬ 3^(k+1) ∣ N) ∧ 3^1 ∣ N ∧ ¬ 3^2 ∣ N :=
by
  sorry  -- Placeholder for the proof

end highest_power_of_3_divides_N_l303_303286


namespace total_students_course_l303_303018

theorem total_students_course 
  (T : ℕ)
  (H1 : (1 / 5 : ℚ) * T = (1 / 5) * T)
  (H2 : (1 / 4 : ℚ) * T = (1 / 4) * T)
  (H3 : (1 / 2 : ℚ) * T = (1 / 2) * T)
  (H4 : T = (1 / 5 : ℚ) * T + (1 / 4 : ℚ) * T + (1 / 2 : ℚ) * T + 30) : 
  T = 600 :=
sorry

end total_students_course_l303_303018


namespace product_is_zero_l303_303058

def product_series (a : ℤ) : ℤ :=
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * 
  (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_is_zero : product_series 3 = 0 :=
by
  sorry

end product_is_zero_l303_303058


namespace percent_round_trip_tickets_l303_303864

variable (P : ℕ) -- total number of passengers

def passengers_with_round_trip_tickets (P : ℕ) : ℕ :=
  2 * (P / 5 / 2)

theorem percent_round_trip_tickets (P : ℕ) : 
  passengers_with_round_trip_tickets P = 2 * (P / 5 / 2) :=
by
  sorry

end percent_round_trip_tickets_l303_303864


namespace coffee_students_l303_303683

variable (S : ℝ) -- Total number of students
variable (T : ℝ) -- Number of students who chose tea
variable (C : ℝ) -- Number of students who chose coffee

-- Given conditions
axiom h1 : 0.4 * S = 80   -- 40% of the students chose tea
axiom h2 : T = 80         -- Number of students who chose tea is 80
axiom h3 : 0.3 * S = C    -- 30% of the students chose coffee

-- Prove that the number of students who chose coffee is 60
theorem coffee_students : C = 60 := by
  sorry

end coffee_students_l303_303683


namespace betty_needs_more_flies_l303_303494

def betty_frog_food (daily_flies: ℕ) (days_per_week: ℕ) (morning_catch: ℕ) 
  (afternoon_catch: ℕ) (flies_escaped: ℕ) : ℕ :=
  days_per_week * daily_flies - (morning_catch + afternoon_catch - flies_escaped)

theorem betty_needs_more_flies :
  betty_frog_food 2 7 5 6 1 = 4 :=
by
  sorry

end betty_needs_more_flies_l303_303494


namespace coloring_scheme_count_l303_303105

/-- Given the set of points in the Cartesian plane, where each point (m, n) with
    1 <= m, n <= 6 is colored either red or blue, the number of ways to color these points
    such that each unit square has exactly two red vertices is 126. -/
theorem coloring_scheme_count 
  (color : Fin 6 → Fin 6 → Bool)
  (colored_correctly : ∀ m n, (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ 
    (color m n = true ∨ color m n = false) :=
    sorry
  )
  : (∃ valid_coloring : Nat, valid_coloring = 126) :=
  sorry

end coloring_scheme_count_l303_303105


namespace ratio_trumpet_to_running_l303_303520

def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 40

theorem ratio_trumpet_to_running : (trumpet_hours : ℚ) / running_hours = 2 :=
by
  sorry

end ratio_trumpet_to_running_l303_303520


namespace last_two_digits_of_sum_of_first_50_factorials_l303_303309

noncomputable theory

def sum_of_factorials_last_two_digits : ℕ :=
  (List.sum (List.map (λ n, n.factorial % 100) (List.range 10))) % 100

theorem last_two_digits_of_sum_of_first_50_factorials : 
  sum_of_factorials_last_two_digits = 13 :=
by
  -- The proof is omitted as requested.
  sorry

end last_two_digits_of_sum_of_first_50_factorials_l303_303309


namespace union_A_B_eq_C_l303_303077

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
noncomputable def C : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}

theorem union_A_B_eq_C : A ∪ B = C := by
  sorry

end union_A_B_eq_C_l303_303077


namespace right_triangle_area_semi_perimeter_inequality_l303_303130

theorem right_triangle_area_semi_perimeter_inequality 
  (x y : ℝ) (h : x > 0 ∧ y > 0) 
  (p : ℝ := (x + y + Real.sqrt (x^2 + y^2)) / 2)
  (S : ℝ := x * y / 2) 
  (hypotenuse : ℝ := Real.sqrt (x^2 + y^2)) 
  (right_triangle : hypotenuse ^ 2 = x ^ 2 + y ^ 2) : 
  S <= p^2 / 5.5 := 
sorry

end right_triangle_area_semi_perimeter_inequality_l303_303130


namespace part1_part2_l303_303992

-- Part 1: Definition of "consecutive roots quadratic equation"
def consecutive_roots (a b : ℤ) : Prop := a = b + 1 ∨ b = a + 1

-- Statement that for some k and constant term, the roots of the quadratic form consecutive roots
theorem part1 (k : ℤ) : consecutive_roots 7 8 → k = -15 → (∀ x : ℤ, x^2 + k * x + 56 = 0 → x = 7 ∨ x = 8) :=
by
  sorry

-- Part 2: Generalizing to the nth equation
theorem part2 (n : ℕ) : 
  (∀ x : ℤ, x^2 - (2 * n - 1) * x + n * (n - 1) = 0 → x = n ∨ x = n - 1) :=
by
  sorry

end part1_part2_l303_303992


namespace sam_digits_memorized_l303_303262

-- Definitions
def carlos_memorized (c : ℕ) := (c * 6 = 24)
def sam_memorized (s c : ℕ) := (s = c + 6)
def mina_memorized := 24

-- Theorem
theorem sam_digits_memorized (s c : ℕ) (h_c : carlos_memorized c) (h_s : sam_memorized s c) : s = 10 :=
by {
  sorry
}

end sam_digits_memorized_l303_303262


namespace pipes_fill_tank_in_1_5_hours_l303_303997

theorem pipes_fill_tank_in_1_5_hours :
  (1 / 3 + 1 / 9 + 1 / 18 + 1 / 6) = (2 / 3) →
  (1 / (2 / 3)) = (3 / 2) :=
by sorry

end pipes_fill_tank_in_1_5_hours_l303_303997


namespace exists_negative_value_of_f_l303_303322

noncomputable def f : ℝ → ℝ := sorry

axiom f_monotonic (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) : f x < f y
axiom f_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f ((2 * x * y) / (x + y)) ≥ (f x + f y) / 2

theorem exists_negative_value_of_f : ∃ x > 0, f x < 0 := 
sorry

end exists_negative_value_of_f_l303_303322


namespace solve_for_x_l303_303544

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l303_303544


namespace calc_num_int_values_l303_303391

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l303_303391


namespace range_of_a_l303_303241

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - a > 0 ∧ 3 * x - 4 < 5) -> False) ↔ (a ≥ 6) :=
by
  sorry

end range_of_a_l303_303241


namespace median_equality_range_inequality_l303_303641

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l303_303641


namespace find_a_l303_303222

theorem find_a (a : ℝ) (h1 : a^2 + 2 * a - 15 = 0) (h2 : a^2 + 4 * a - 5 ≠ 0) :
  a = 3 :=
by
sorry

end find_a_l303_303222


namespace base6_add_sub_l303_303807

theorem base6_add_sub (a b c : ℕ) (ha : a = 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 6 * 6^1 + 5 * 6^0) (hc : c = 1 * 6^1 + 1 * 6^0) :
  (a + b - c) = 1 * 6^3 + 0 * 6^2 + 5 * 6^1 + 3 * 6^0 :=
by
  -- We should translate the problem context into equivalence
  -- but this part of the actual proof is skipped with sorry.
  sorry

end base6_add_sub_l303_303807


namespace problem_statement_l303_303487

theorem problem_statement (p : ℝ) : 
  (∀ (q : ℝ), q > 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) 
  ↔ (0 ≤ p ∧ p ≤ 7.275) :=
sorry

end problem_statement_l303_303487


namespace solve_problem_1_solve_problem_2_l303_303783

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_problem_1_solve_problem_2_l303_303783


namespace students_taking_neither_l303_303704

theorem students_taking_neither (total biology chemistry both : ℕ)
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  (total - (biology + chemistry - both)) = 10 :=
by {
  sorry
}

end students_taking_neither_l303_303704


namespace corvette_trip_average_rate_l303_303178

theorem corvette_trip_average_rate (total_distance : ℕ) (first_half_distance : ℕ)
  (first_half_rate : ℕ) (second_half_time_multiplier : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_rate = 80 →
  second_half_time_multiplier = 3 →
  total_time = (first_half_distance / first_half_rate) + (second_half_time_multiplier * (first_half_distance / first_half_rate)) →
  (total_distance / total_time) = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end corvette_trip_average_rate_l303_303178


namespace candies_per_basket_l303_303779

noncomputable def chocolate_bars : ℕ := 5
noncomputable def mms : ℕ := 7 * chocolate_bars
noncomputable def marshmallows : ℕ := 6 * mms
noncomputable def total_candies : ℕ := chocolate_bars + mms + marshmallows
noncomputable def baskets : ℕ := 25

theorem candies_per_basket : total_candies / baskets = 10 :=
by
  sorry

end candies_per_basket_l303_303779


namespace income_on_fifth_day_l303_303910

-- Define the incomes for the first four days
def income_day1 := 600
def income_day2 := 250
def income_day3 := 450
def income_day4 := 400

-- Define the average income
def average_income := 500

-- Define the length of days
def days := 5

-- Define the total income for the 5 days
def total_income : ℕ := days * average_income

-- Define the total income for the first 4 days
def total_income_first4 := income_day1 + income_day2 + income_day3 + income_day4

-- Define the income on the fifth day
def income_day5 := total_income - total_income_first4

-- The theorem to prove the income of the fifth day is $800
theorem income_on_fifth_day : income_day5 = 800 := by
  -- proof is not required, so we leave the proof section with sorry
  sorry

end income_on_fifth_day_l303_303910


namespace combined_value_l303_303121

noncomputable def sum_even (a l : ℕ) : ℕ :=
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

noncomputable def sum_odd (a l : ℕ) : ℕ :=
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem combined_value : 
  let i := sum_even 2 500
  let k := sum_even 8 200
  let j := sum_odd 5 133
  2 * i - k + 3 * j = 128867 :=
by
  sorry

end combined_value_l303_303121


namespace option_d_correct_l303_303318

theorem option_d_correct (m n : ℝ) : (m + n) * (m - 2 * n) = m^2 - m * n - 2 * n^2 :=
by
  sorry

end option_d_correct_l303_303318


namespace cos_eq_cos_of_n_l303_303804

theorem cos_eq_cos_of_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (283 * Real.pi / 180)) : n = 77 :=
by sorry

end cos_eq_cos_of_n_l303_303804


namespace sum_of_21st_set_l303_303951

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

def first_element_of_set (n : ℕ) : ℕ := triangular_number n - n + 1

def sum_of_elements_in_set (n : ℕ) : ℕ := 
  n * ((first_element_of_set n + triangular_number n) / 2)

theorem sum_of_21st_set : sum_of_elements_in_set 21 = 4641 := by 
  sorry

end sum_of_21st_set_l303_303951


namespace right_triangle_hypotenuse_l303_303971

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 6) (k : b = 8) (pt : a^2 + b^2 = c^2) : c = 10 := by
  sorry

end right_triangle_hypotenuse_l303_303971


namespace rongrong_bike_speed_l303_303303

theorem rongrong_bike_speed :
  ∃ (x : ℝ), (15 / x - 15 / (4 * x) = 45 / 60) → x = 15 :=
by
  sorry

end rongrong_bike_speed_l303_303303


namespace price_of_each_bottle_is_3_l303_303167

/-- Each bottle of iced coffee has 6 servings. -/
def servings_per_bottle : ℕ := 6

/-- Tricia drinks half a container (bottle) a day. -/
def daily_consumption_rate : ℕ := servings_per_bottle / 2

/-- Number of days in 2 weeks. -/
def duration_days : ℕ := 14

/-- Number of servings Tricia consumes in 2 weeks. -/
def total_servings : ℕ := daily_consumption_rate * duration_days

/-- Number of bottles needed to get the total servings. -/
def bottles_needed : ℕ := total_servings / servings_per_bottle

/-- The total cost of the bottles is $21. -/
def total_cost : ℕ := 21

/-- The price per bottle is the total cost divided by the number of bottles. -/
def price_per_bottle : ℕ := total_cost / bottles_needed

/-- The price of each bottle is $3. -/
theorem price_of_each_bottle_is_3 : price_per_bottle = 3 :=
by
  -- We assume the necessary steps and mathematical verifications have been done.
  sorry

end price_of_each_bottle_is_3_l303_303167


namespace circle_intersection_value_l303_303499

theorem circle_intersection_value {x1 y1 x2 y2 : ℝ} 
  (h_circle : x1^2 + y1^2 = 4)
  (h_non_negative : x1 ≥ 0 ∧ y1 ≥ 0 ∧ x2 ≥ 0 ∧ y2 ≥ 0)
  (h_symmetric : x1 = y2 ∧ x2 = y1) :
  x1^2 + x2^2 = 4 := 
by
  sorry

end circle_intersection_value_l303_303499


namespace juvy_chives_l303_303116

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end juvy_chives_l303_303116


namespace complement_union_l303_303828

def M := { x : ℝ | (x + 3) * (x - 1) < 0 }
def N := { x : ℝ | x ≤ -3 }
def union_set := M ∪ N

theorem complement_union :
  ∀ x : ℝ, x ∈ (⊤ \ union_set) ↔ x ≥ 1 :=
by
  sorry

end complement_union_l303_303828


namespace maximize_pasture_area_l303_303604

theorem maximize_pasture_area
  (barn_length fence_cost budget : ℕ)
  (barn_length_eq : barn_length = 400)
  (fence_cost_eq : fence_cost = 5)
  (budget_eq : budget = 1500) :
  ∃ x y : ℕ, y = 150 ∧
  x > 0 ∧
  2 * x + y = budget / fence_cost ∧
  y = barn_length - 2 * x ∧
  (x * y) = (75 * 150) :=
by
  sorry

end maximize_pasture_area_l303_303604


namespace calc_num_int_values_l303_303392

theorem calc_num_int_values (x : ℕ) (h : 121 ≤ x ∧ x < 144) : ∃ n : ℕ, n = 23 :=
by
  sorry

end calc_num_int_values_l303_303392


namespace log_function_passes_through_point_l303_303025

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x - 1) / Real.log a - 1

theorem log_function_passes_through_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -1 :=
by
  -- To complete the proof, one would argue about the properties of logarithms in specific bases.
  sorry

end log_function_passes_through_point_l303_303025


namespace canonical_equations_of_line_l303_303454

-- Conditions: Two planes given by their equations
def plane1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z + 8 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x + 5 * y - 4 * z + 4 = 0

-- Proving the canonical form of the line
theorem canonical_equations_of_line :
  ∃ x y z, plane1 x y z ∧ plane2 x y z ↔ 
  ∃ t, x = -1 + 5 * t ∧ y = 2 / 5 + 42 * t ∧ z = 60 * t :=
sorry

end canonical_equations_of_line_l303_303454


namespace percentage_increase_l303_303914

def initialProductivity := 120
def totalArea := 1440
def daysInitialProductivity := 2
def daysAheadOfSchedule := 2

theorem percentage_increase :
  let originalDays := totalArea / initialProductivity
  let daysWithIncrease := originalDays - daysAheadOfSchedule
  let daysWithNewProductivity := daysWithIncrease - daysInitialProductivity
  let remainingArea := totalArea - (daysInitialProductivity * initialProductivity)
  let newProductivity := remainingArea / daysWithNewProductivity
  let increase := ((newProductivity - initialProductivity) / initialProductivity) * 100
  increase = 25 :=
by
  sorry

end percentage_increase_l303_303914


namespace simplify_rationalize_denominator_l303_303282

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l303_303282


namespace problem_statement_l303_303233

noncomputable def g (x : ℝ) : ℝ := 3^(x + 1)

theorem problem_statement (x : ℝ) : g (x + 1) - 2 * g x = g x := by
  -- The proof here is omitted
  sorry

end problem_statement_l303_303233


namespace intersection_A_B_l303_303076

def setA (x : ℝ) : Prop := 3 * x + 2 > 0
def setB (x : ℝ) : Prop := (x + 1) * (x - 3) > 0
def A : Set ℝ := { x | setA x }
def B : Set ℝ := { x | setB x }

theorem intersection_A_B : A ∩ B = { x | 3 < x } := by
  sorry

end intersection_A_B_l303_303076


namespace domain_of_f_l303_303623

open Set

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 1)

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_f_l303_303623


namespace mechanic_charge_per_hour_l303_303527

/-- Definitions based on provided conditions -/
def total_amount_paid : ℝ := 300
def part_cost : ℝ := 150
def hours : ℕ := 2

/-- Theorem stating the labor cost per hour is $75 -/
theorem mechanic_charge_per_hour (total_amount_paid part_cost hours : ℝ) : hours = 2 → part_cost = 150 → total_amount_paid = 300 → 
  (total_amount_paid - part_cost) / hours = 75 :=
by
  sorry

end mechanic_charge_per_hour_l303_303527


namespace c_over_e_l303_303729

theorem c_over_e (a b c d e : ℝ) (h1 : 1 * 2 * 3 * a + 1 * 2 * 4 * a + 1 * 3 * 4 * a + 2 * 3 * 4 * a = -d)
  (h2 : 1 * 2 * 3 * 4 = e / a)
  (h3 : 1 * 2 * a + 1 * 3 * a + 1 * 4 * a + 2 * 3 * a + 2 * 4 * a + 3 * 4 * a = c) :
  c / e = 35 / 24 :=
by
  sorry

end c_over_e_l303_303729


namespace eight_b_plus_one_composite_l303_303878

theorem eight_b_plus_one_composite (a b : ℕ) (h₀ : a > b)
  (h₁ : a - b = 5 * b^2 - 4 * a^2) : ∃ (n m : ℕ), 1 < n ∧ 1 < m ∧ (8 * b + 1) = n * m :=
by
  sorry

end eight_b_plus_one_composite_l303_303878


namespace max_books_per_student_l303_303682

-- Define the variables and conditions
variables (students : ℕ) (not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 : ℕ)
variables (avg_books_per_student : ℕ)
variables (remaining_books : ℕ) (max_books : ℕ)

-- Assume given conditions
def conditions : Prop :=
  students = 100 ∧ 
  not_borrowed5 = 5 ∧ 
  borrowed1_20 = 20 ∧ 
  borrowed2_25 = 25 ∧ 
  borrowed3_30 = 30 ∧ 
  borrowed5_20 = 20 ∧ 
  avg_books_per_student = 3

-- Prove the maximum number of books any single student could have borrowed is 50
theorem max_books_per_student (students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student : ℕ) (max_books : ℕ) :
  conditions students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student →
  max_books = 50 :=
by
  sorry

end max_books_per_student_l303_303682


namespace find_x_y_l303_303524

theorem find_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : (x * y / 7) ^ (3 / 2) = x) 
  (h2 : (x * y / 7) = y) : 
  x = 7 ∧ y = 7 ^ (2 / 3) :=
by
  sorry

end find_x_y_l303_303524


namespace complex_expression_evaluation_l303_303354

-- Conditions
def i : ℂ := Complex.I -- Representing the imaginary unit i

-- Defining the inverse of a complex number
noncomputable def complex_inv (z : ℂ) := 1 / z

-- Proof statement
theorem complex_expression_evaluation :
  (i - complex_inv i + 3)⁻¹ = (3 - 2 * i) / 13 := by
sorry

end complex_expression_evaluation_l303_303354


namespace weights_are_equal_l303_303810

variable {n : ℕ}
variables {a : Fin (2 * n + 1) → ℝ}

def weights_condition
    (a : Fin (2 * n + 1) → ℝ) : Prop :=
  ∀ i : Fin (2 * n + 1), ∃ (A B : Finset (Fin (2 * n + 1))),
    A.card = n ∧ B.card = n ∧ A ∩ B = ∅ ∧
    A ∪ B = Finset.univ.erase i ∧
    (A.sum a = B.sum a)

theorem weights_are_equal
    (h : weights_condition a) :
  ∃ k : ℝ, ∀ i : Fin (2 * n + 1), a i = k :=
  sorry

end weights_are_equal_l303_303810


namespace sin_cos_identity_l303_303095

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_identity_l303_303095


namespace number_of_possible_U_l303_303253

open Finset

def U_min : ℕ := (range 60).sum (λ i, i + 10)
def U_max : ℕ := (range 60).sum (λ i, i + 91)

theorem number_of_possible_U (S : Finset ℕ) (h₁ : set.range 60 ⊆ S)
(h₂ : 10 ≤ S.min' (by sorry))
(h₃ : S.max' (by sorry) ≤ 150) :
  ∃ U, U = S.sum id ∧ 2370 ≤ U ∧ U ≤ 7230 → (number_of_possible_U = 4861) := by sorry


end number_of_possible_U_l303_303253


namespace total_cost_of_books_and_pencils_l303_303835

variable (a b : ℕ)

theorem total_cost_of_books_and_pencils (a b : ℕ) : 5 * a + 2 * b = 5 * a + 2 * b := by
  sorry

end total_cost_of_books_and_pencils_l303_303835


namespace complete_square_l303_303008

theorem complete_square (x : ℝ) : 
  (x ^ 2 - 2 * x = 9) -> ((x - 1) ^ 2 = 10) :=
by
  intro h
  rw [← add_zero (x ^ 2 - 2 * x), ← add_zero (10)]
  calc
    x ^ 2 - 2 * x = 9                   : by rw [h]
             ...  = (x ^ 2 - 2 * x + 1 - 1) : by rw [add_sub_cancel, add_zero]
             ...  = (x - 1) ^ 2 - 1     : by 
                           { rw [sub_eq_add_neg], exact add_sub_cancel _ _}
             ...  = 10 - 1              : by rw [h]
             ...  = 10                  : by rw (sub_sub_cancel)
 

end complete_square_l303_303008


namespace total_shopping_cost_l303_303924

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end total_shopping_cost_l303_303924


namespace balance_balls_l303_303259

noncomputable def green_weight := (9 : ℝ) / 4
noncomputable def yellow_weight := (7 : ℝ) / 3
noncomputable def white_weight := (3 : ℝ) / 2

theorem balance_balls (B : ℝ) : 
  5 * green_weight * B + 4 * yellow_weight * B + 3 * white_weight * B = (301 / 12) * B :=
by
  sorry

end balance_balls_l303_303259


namespace min_time_to_pass_l303_303346

noncomputable def tunnel_length : ℝ := 2150
noncomputable def num_vehicles : ℝ := 55
noncomputable def vehicle_length : ℝ := 10
noncomputable def speed_limit : ℝ := 20
noncomputable def max_speed : ℝ := 40

noncomputable def distance_between_vehicles (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then 20 else
if 10 < x ∧ x ≤ 20 then (1/6) * x ^ 2 + (1/3) * x else
0

noncomputable def time_to_pass_through_tunnel (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 10 then (2150 + 10 * 55 + 20 * (55 - 1)) / x else
if 10 < x ∧ x ≤ 20 then (2150 + 10 * 55 + ((1/6) * x^2 + (1/3) * x) * (55 - 1)) / x + 9 * x + 18 else
0

theorem min_time_to_pass : ∃ x : ℝ, (10 < x ∧ x ≤ 20) ∧ x = 17.3 ∧ time_to_pass_through_tunnel x = 329.4 :=
sorry

end min_time_to_pass_l303_303346


namespace percentage_increase_is_20_percent_l303_303291

noncomputable def SP : ℝ := 8600
noncomputable def CP : ℝ := 7166.67
noncomputable def percentageIncrease : ℝ := ((SP - CP) / CP) * 100

theorem percentage_increase_is_20_percent : percentageIncrease = 20 :=
by
  sorry

end percentage_increase_is_20_percent_l303_303291


namespace regular_polygon_sides_l303_303916

theorem regular_polygon_sides (n : ℕ) (h : ∀ i < n, (interior_angle_i : ℝ) = 150) :
  (n = 12) :=
by
  sorry

end regular_polygon_sides_l303_303916


namespace Lacy_correct_percent_l303_303707

theorem Lacy_correct_percent (x : ℝ) (h1 : 7 * x > 0) : ((5 * 100) / 7) = 71.43 :=
by
  sorry

end Lacy_correct_percent_l303_303707


namespace mass_percentage_C_is_54_55_l303_303492

def mass_percentage (C: String) (percentage: ℝ) : Prop :=
  percentage = 54.55

theorem mass_percentage_C_is_54_55 :
  mass_percentage "C" 54.55 :=
by
  unfold mass_percentage
  rfl

end mass_percentage_C_is_54_55_l303_303492


namespace initial_oranges_per_rupee_l303_303337

theorem initial_oranges_per_rupee (loss_rate_gain_rate cost_rate : ℝ) (initial_oranges : ℤ) : 
  loss_rate_gain_rate = 0.92 ∧ cost_rate = 18.4 ∧ 1.25 * cost_rate = 1.25 * 0.92 * (initial_oranges : ℝ) →
  initial_oranges = 14 := by
  sorry

end initial_oranges_per_rupee_l303_303337


namespace songs_per_album_l303_303450

theorem songs_per_album (C P : ℕ) (h1 : 4 * C + 5 * P = 72) (h2 : C = P) : C = 8 :=
by
  sorry

end songs_per_album_l303_303450


namespace usual_time_to_catch_bus_l303_303002

variables (S T T' : ℝ)

theorem usual_time_to_catch_bus
  (h1 : T' = (5 / 4) * T)
  (h2 : T' - T = 6) : T = 24 :=
sorry

end usual_time_to_catch_bus_l303_303002


namespace neg_p_sufficient_not_necessary_q_l303_303072

-- Definitions from the given conditions
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- The theorem stating the mathematical equivalence
theorem neg_p_sufficient_not_necessary_q (a : ℝ) : (¬ p a → q a) ∧ ¬ (q a → ¬ p a) := 
by sorry

end neg_p_sufficient_not_necessary_q_l303_303072


namespace trig_identity_example_l303_303047

theorem trig_identity_example :
  (Real.cos (47 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - 
   Real.sin (47 * Real.pi / 180) * Real.sin (13 * Real.pi / 180)) = 
  (Real.cos (60 * Real.pi / 180)) := by
  sorry

end trig_identity_example_l303_303047


namespace change_in_mean_l303_303874

theorem change_in_mean {a b c d : ℝ} 
  (h1 : (a + b + c + d) / 4 = 10)
  (h2 : (b + c + d) / 3 = 11)
  (h3 : (a + c + d) / 3 = 12)
  (h4 : (a + b + d) / 3 = 13) : 
  ((a + b + c) / 3) = 4 := by 
  sorry

end change_in_mean_l303_303874


namespace smallest_solution_l303_303896

theorem smallest_solution (x : ℝ) (h : x^4 - 16 * x^2 + 63 = 0) :
  x = -3 :=
sorry

end smallest_solution_l303_303896


namespace circle_center_l303_303726

theorem circle_center 
    (x y : ℝ)
    (h : x^2 + y^2 - 4 * x + 6 * y = 0) :
    (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = (x^2 - 4*x + 4) + (y^2 + 6*y + 9) 
    → (x, y) = (2, -3)) :=
sorry

end circle_center_l303_303726


namespace product_and_divisibility_l303_303358

theorem product_and_divisibility (n : ℕ) (h : n = 3) :
  (n-1) * n * (n+1) * (n+2) * (n+3) = 720 ∧ ¬ (720 % 11 = 0) :=
by
  sorry

end product_and_divisibility_l303_303358


namespace cylinder_volume_l303_303083

variables (a : ℝ) (π_ne_zero : π ≠ 0) (two_ne_zero : 2 ≠ 0) 

theorem cylinder_volume (h1 : ∃ (h r : ℝ), (2 * π * r = 2 * a ∧ h = a) 
                        ∨ (2 * π * r = a ∧ h = 2 * a)) :
  (∃ (V : ℝ), V = a^3 / π) ∨ (∃ (V : ℝ), V = a^3 / (2 * π)) :=
by
  sorry

end cylinder_volume_l303_303083


namespace correct_option_l303_303898

def option_A_1 : ℤ := (-2) ^ 2
def option_A_2 : ℤ := -(2 ^ 2)
def option_B_1 : ℤ := (|-2|) ^ 2
def option_B_2 : ℤ := -(2 ^ 2)
def option_C_1 : ℤ := (-2) ^ 3
def option_C_2 : ℤ := -(2 ^ 3)
def option_D_1 : ℤ := (|-2|) ^ 3
def option_D_2 : ℤ := -(2 ^ 3)

theorem correct_option : option_C_1 = option_C_2 ∧ 
  (option_A_1 ≠ option_A_2) ∧ 
  (option_B_1 ≠ option_B_2) ∧ 
  (option_D_1 ≠ option_D_2) :=
by
  sorry

end correct_option_l303_303898


namespace solve_system_eqn_l303_303206

theorem solve_system_eqn :
  ∃ x y : ℚ, 7 * x = -9 - 3 * y ∧ 2 * x = 5 * y - 30 ∧ x = -135 / 41 ∧ y = 192 / 41 :=
by 
  sorry

end solve_system_eqn_l303_303206


namespace sum_of_numbers_gt_1_1_equals_3_9_l303_303317

noncomputable def sum_of_elements_gt_1_1 : Float :=
  let numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
  let numbers_gt_1_1 := List.filter (fun x => x > 1.1) numbers
  List.sum numbers_gt_1_1

theorem sum_of_numbers_gt_1_1_equals_3_9 :
  sum_of_elements_gt_1_1 = 3.9 := by
  sorry

end sum_of_numbers_gt_1_1_equals_3_9_l303_303317


namespace range_of_m_l303_303632

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (m / (2*x - 1) + 3 = 0) ∧ (x > 0)) ↔ (m < 3 ∧ m ≠ 0) :=
by
  sorry

end range_of_m_l303_303632


namespace unique_isolating_line_a_eq_2e_l303_303841

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

theorem unique_isolating_line_a_eq_2e (a : ℝ) (h : a > 0) :
  (∃ k b, ∀ x : ℝ, f x ≥ k * x + b ∧ k * x + b ≥ g a x) → a = 2 * Real.exp 1 :=
sorry

end unique_isolating_line_a_eq_2e_l303_303841


namespace inequality_solution_l303_303295

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x - 3) ≥ 1 ↔ (x > 3 ∨ x ≤ -2) :=
by 
  sorry

end inequality_solution_l303_303295


namespace mr_yadav_yearly_savings_l303_303528

theorem mr_yadav_yearly_savings (S : ℕ) (h1 : S * 3 / 5 * 1 / 2 = 1584) : S * 3 / 5 * 1 / 2 * 12 = 19008 :=
  sorry

end mr_yadav_yearly_savings_l303_303528


namespace min_value_of_F_on_neg_infinity_l303_303327

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions provided in the problem
axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom g_odd : ∀ x : ℝ, g (-x) = - g x
noncomputable def F (x : ℝ) := a * f x + b * g x + 2
axiom F_max_on_pos : ∃ x ∈ (Set.Ioi 0), F x = 5

-- Prove the conclusion of the problem
theorem min_value_of_F_on_neg_infinity : ∃ y ∈ (Set.Iio 0), F y = -1 :=
sorry

end min_value_of_F_on_neg_infinity_l303_303327


namespace shipment_cost_l303_303702

-- Define the conditions
def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def shipping_cost_per_crate : ℝ := 1.5
def surcharge_per_crate : ℝ := 0.5
def flat_fee : ℝ := 10

-- Define the question as a theorem
theorem shipment_cost : 
  let crates := total_weight / weight_per_crate
  let cost_per_crate := shipping_cost_per_crate + surcharge_per_crate
  let total_cost_crates := crates * cost_per_crate
  let total_cost := total_cost_crates + flat_fee
  total_cost = 46 := by
  -- Proof omitted
  sorry

end shipment_cost_l303_303702


namespace monica_expected_winnings_l303_303703

def monica_die_winnings : List ℤ := [2, 3, 5, 7, 0, 0, 0, -4]

def expected_value (values : List ℤ) : ℚ :=
  (List.sum values) / (values.length : ℚ)

theorem monica_expected_winnings :
  expected_value monica_die_winnings = 1.625 := by
  sorry

end monica_expected_winnings_l303_303703


namespace median_equality_range_inequality_l303_303640

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l303_303640


namespace sine_of_pi_minus_alpha_l303_303669

theorem sine_of_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 3) : Real.sin (π - α) = 1 / 3 :=
by
  sorry

end sine_of_pi_minus_alpha_l303_303669


namespace largest_constant_inequality_l303_303624

theorem largest_constant_inequality :
  ∃ C, C = 3 ∧
  (∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 
  C * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂))) :=

sorry

end largest_constant_inequality_l303_303624


namespace find_shorter_parallel_side_l303_303207

variable (x : ℝ) (a : ℝ) (b : ℝ) (h : ℝ)

def is_trapezium_area (a b h : ℝ) (area : ℝ) : Prop :=
  area = 1/2 * (a + b) * h

theorem find_shorter_parallel_side
  (h28 : a = 28)
  (h15 : h = 15)
  (hArea : area = 345)
  (hIsTrapezium : is_trapezium_area a b h area):
  b = 18 := 
sorry

end find_shorter_parallel_side_l303_303207


namespace beth_students_proof_l303_303476

-- Let initial := 150
-- Let joined := 30
-- Let left := 15
-- final := initial + joined - left
-- Prove final = 165

def beth_final_year_students (initial joined left final : ℕ) : Prop :=
  initial = 150 ∧ joined = 30 ∧ left = 15 ∧ final = initial + joined - left

theorem beth_students_proof : ∃ final, beth_final_year_students 150 30 15 final ∧ final = 165 :=
by
  sorry

end beth_students_proof_l303_303476


namespace volleyball_team_starters_l303_303129

noncomputable def volleyball_team_count : ℕ := 14
noncomputable def triplets_count : ℕ := 3
noncomputable def starters_count : ℕ := 6

theorem volleyball_team_starters : 
  (choose (volleyball_team_count - triplets_count) starters_count) + 
  (triplets_count * choose (volleyball_team_count - triplets_count) (starters_count - 1)) = 1848 :=
by sorry

end volleyball_team_starters_l303_303129


namespace intersection_point_at_neg4_l303_303731

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end intersection_point_at_neg4_l303_303731


namespace maximize_profit_l303_303911

noncomputable def R (x : ℝ) : ℝ := 
  if x ≤ 40 then
    40 * x - (1 / 2) * x^2
  else
    1500 - 25000 / x

noncomputable def cost (x : ℝ) : ℝ := 2 + 0.1 * x

noncomputable def f (x : ℝ) : ℝ := R x - cost x

theorem maximize_profit :
  ∃ x : ℝ, x = 50 ∧ f 50 = 300 := by
  sorry

end maximize_profit_l303_303911


namespace find_y_in_triangle_l303_303690

theorem find_y_in_triangle (BAC ABC BCA : ℝ) (y : ℝ) (h1 : BAC = 90)
  (h2 : ABC = 2 * y) (h3 : BCA = y - 10) : y = 100 / 3 :=
by
  -- The proof will be left as sorry
  sorry

end find_y_in_triangle_l303_303690


namespace elizabeth_time_l303_303000

-- Defining the conditions
def tom_time_minutes : ℕ := 120
def time_ratio : ℕ := 4

-- Proving Elizabeth's time
theorem elizabeth_time : tom_time_minutes / time_ratio = 30 := 
by
  sorry

end elizabeth_time_l303_303000


namespace minimize_sum_of_reciprocals_l303_303973

theorem minimize_sum_of_reciprocals (a b : ℕ) (h : 4 * a + b = 6) : 
  a = 1 ∧ b = 2 ∨ a = 2 ∧ b = 1 :=
by
  sorry

end minimize_sum_of_reciprocals_l303_303973


namespace carla_receives_correct_amount_l303_303415

theorem carla_receives_correct_amount (L B C X : ℝ) : 
  (L + B + C + X) / 3 - (C + X) = (L + B - 2 * C - 2 * X) / 3 :=
by
  sorry

end carla_receives_correct_amount_l303_303415


namespace triangles_with_positive_area_l303_303959

-- Define the set of points in the coordinate grid
def points := { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4 }

-- Number of ways to choose 3 points from the grid
def total_triples := Nat.choose 16 3

-- Number of collinear triples
def collinear_triples := 32 + 8 + 4

-- Number of triangles with positive area
theorem triangles_with_positive_area :
  (total_triples - collinear_triples) = 516 :=
by
  -- Definitions for total_triples and collinear_triples.
  -- Proof steps would go here.
  sorry

end triangles_with_positive_area_l303_303959


namespace add_pure_water_to_achieve_solution_l303_303091

theorem add_pure_water_to_achieve_solution
  (w : ℝ) (h_salt_content : 0.15 * 40 = 6) (h_new_concentration : 6 / (40 + w) = 0.1) :
  w = 20 :=
sorry

end add_pure_water_to_achieve_solution_l303_303091


namespace count_integer_values_of_x_l303_303388

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : {n : ℕ | 121 < n ∧ n ≤ 144}.card = 23 := 
by
  sorry

end count_integer_values_of_x_l303_303388


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l303_303649

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l303_303649


namespace solve_for_y_l303_303871

/-- Given the equation 7(2y + 3) - 5 = -3(2 - 5y), solve for y. -/
theorem solve_for_y (y : ℤ) : 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) → y = 22 :=
by
  intros h
  sorry

end solve_for_y_l303_303871


namespace part1a_part1b_part2_part3_l303_303125

-- Definitions for the sequences in columns ①, ②, and ③
def col1 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1)
def col2 (n : ℕ) : ℤ := ((-1 : ℤ) ^ n * (2 * n - 1)) - 2
def col3 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1) * 3

-- Problem statements
theorem part1a : col1 10 = 19 :=
sorry

theorem part1b : col2 15 = -31 :=
sorry

theorem part2 : ¬ ∃ n : ℕ, col2 (n - 1) + col2 n + col2 (n + 1) = 1001 :=
sorry

theorem part3 : ∃ k : ℕ, col1 k + col2 k + col3 k = 599 ∧ k = 301 :=
sorry

end part1a_part1b_part2_part3_l303_303125


namespace largest_stamps_per_page_max_largest_stamps_per_page_l303_303409

theorem largest_stamps_per_page (n : ℕ) :
  (840 % n = 0) ∧ (1008 % n = 0) ∧ (672 % n = 0) → n ≤ 168 :=
by sorry

theorem max_largest_stamps_per_page :
  ∃ n, (840 % n = 0) ∧ (1008 % n = 0) ∧ (672 % n = 0) ∧ n = 168 :=
by {
  use 168,
  split,
  { calc 840 % 168 = 0 : by sorry },
  split,
  { calc 1008 % 168 = 0 : by sorry },
  { calc 672 % 168 = 0 : by sorry },
  exact eq.refl 168
}

end largest_stamps_per_page_max_largest_stamps_per_page_l303_303409


namespace cara_neighbors_l303_303928

def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem cara_neighbors : number_of_pairs 7 = 21 :=
by
  sorry

end cara_neighbors_l303_303928


namespace possible_values_l303_303981

theorem possible_values (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  ∃ S : Set ℝ, S = {x : ℝ | 4 ≤ x} ∧ (1 / a + 1 / b) ∈ S :=
by
  sorry

end possible_values_l303_303981


namespace polynomial_is_positive_for_all_x_l303_303709

noncomputable def P (x : ℝ) : ℝ := x^12 - x^9 + x^4 - x + 1

theorem polynomial_is_positive_for_all_x (x : ℝ) : P x > 0 := 
by
  dsimp [P]
  sorry -- Proof is omitted.

end polynomial_is_positive_for_all_x_l303_303709


namespace probability_interval_l303_303339

open ProbabilityTheory

noncomputable def probability_red_greater_green (x y : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 then if x < y ∧ y < 3 * x then 1 else 0 else 0

theorem probability_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (∫ (x : ℝ) in 0..1, ∫ (y : ℝ) in (x : ℝ)..(min (3 * x) 1), 1) = 5 / 18 := sorry

end probability_interval_l303_303339


namespace number_of_intersection_points_l303_303489

theorem number_of_intersection_points : 
  ∃! (P : ℝ × ℝ), 
    (P.1 ^ 2 + P.2 ^ 2 = 16) ∧ (P.1 = 4) := 
by
  sorry

end number_of_intersection_points_l303_303489


namespace max_reflections_l303_303031

theorem max_reflections (A B D : Point) (n : ℕ) (angle_CDA : ℝ) (incident_angle : ℕ → ℝ)
  (h1 : angle_CDA = 12)
  (h2 : ∀ k : ℕ, k ≤ n → incident_angle k = k * angle_CDA)
  (h3 : incident_angle n = 90) :
  n = 7 := 
sorry

end max_reflections_l303_303031


namespace rudy_first_run_rate_l303_303133

def first_run_rate (R : ℝ) : Prop :=
  let time_first_run := 5 * R
  let time_second_run := 4 * 9.5
  let total_time := time_first_run + time_second_run
  total_time = 88

theorem rudy_first_run_rate : first_run_rate 10 :=
by
  unfold first_run_rate
  simp
  sorry

end rudy_first_run_rate_l303_303133


namespace solve_equation_l303_303551

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l303_303551


namespace total_travel_time_l303_303790

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l303_303790


namespace machines_initially_working_l303_303330

theorem machines_initially_working (N x : ℕ) (h1 : N * 4 * R = x)
  (h2 : 20 * 6 * R = 3 * x) : N = 10 :=
by
  sorry

end machines_initially_working_l303_303330


namespace roots_cubic_properties_l303_303697

theorem roots_cubic_properties (a b c : ℝ) 
    (h1 : ∀ x : ℝ, x^3 - 2 * x^2 + 3 * x - 4 = 0 → x = a ∨ x = b ∨ x = c)
    (h_sum : a + b + c = 2)
    (h_prod_sum : a * b + b * c + c * a = 3)
    (h_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := by
  sorry

end roots_cubic_properties_l303_303697


namespace rowing_time_l303_303030

theorem rowing_time (rowing_speed : ℕ) (current_speed : ℕ) (distance : ℕ) 
  (h_rowing_speed : rowing_speed = 10)
  (h_current_speed : current_speed = 2)
  (h_distance : distance = 24) : 
  2 * distance / (rowing_speed + current_speed) + 2 * distance / (rowing_speed - current_speed) = 5 :=
by
  rw [h_rowing_speed, h_current_speed, h_distance]
  norm_num
  sorry

end rowing_time_l303_303030


namespace simplify_and_rationalize_l303_303272

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l303_303272


namespace kayden_total_processed_l303_303413

-- Definition of the given conditions and final proof problem statement in Lean 4
variable (x : ℕ)  -- x is the number of cartons delivered to each customer

theorem kayden_total_processed (h : 4 * (x - 60) = 160) : 4 * x = 400 :=
by
  sorry

end kayden_total_processed_l303_303413


namespace proof_problem_l303_303643

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l303_303643


namespace ranking_most_economical_l303_303038

theorem ranking_most_economical (c_T c_R c_J q_T q_R q_J : ℝ)
  (hR_cost : c_R = 1.25 * c_T)
  (hR_quantity : q_R = 0.75 * q_J)
  (hJ_quantity : q_J = 2.5 * q_T)
  (hJ_cost : c_J = 1.2 * c_R) :
  ((c_J / q_J) ≤ (c_R / q_R)) ∧ ((c_R / q_R) ≤ (c_T / q_T)) :=
by {
  sorry
}

end ranking_most_economical_l303_303038


namespace find_q_l303_303229

variable (p q : ℝ) (hp : p > 1) (hq : q > 1) (h_cond1 : 1 / p + 1 / q = 1) (h_cond2 : p * q = 9)

theorem find_q : q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l303_303229


namespace proof_complement_union_l303_303985

-- Definition of the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the subset A
def A : Finset ℕ := {0, 3, 4}

-- Definition of the subset B
def B : Finset ℕ := {1, 3}

-- Definition of the complement of A in U
def complement_A : Finset ℕ := U \ A

-- Definition of the union of the complement of A and B
def union_complement_A_B : Finset ℕ := complement_A ∪ B

-- Statement of the theorem
theorem proof_complement_union :
  union_complement_A_B = {1, 2, 3} :=
sorry

end proof_complement_union_l303_303985


namespace spring_membership_decrease_l303_303774

theorem spring_membership_decrease (init_members : ℝ) (increase_percent : ℝ) (total_change_percent : ℝ) 
  (fall_members := init_members * (1 + increase_percent / 100)) 
  (spring_members := init_members * (1 + total_change_percent / 100)) :
  increase_percent = 8 → total_change_percent = -12.52 → 
  (fall_members - spring_members) / fall_members * 100 = 19 :=
by
  intros h1 h2
  -- The complicated proof goes here.
  sorry

end spring_membership_decrease_l303_303774


namespace final_result_after_subtracting_15_l303_303338

theorem final_result_after_subtracting_15 :
  ∀ (n : ℕ) (r : ℕ) (f : ℕ),
  n = 120 → 
  r = n / 6 → 
  f = r - 15 → 
  f = 5 :=
by
  intros n r f hn hr hf
  have h1 : n = 120 := hn
  have h2 : r = n / 6 := hr
  have h3 : f = r - 15 := hf
  sorry

end final_result_after_subtracting_15_l303_303338


namespace prob_four_of_a_kind_after_re_roll_l303_303870

noncomputable def probability_of_four_of_a_kind : ℚ :=
sorry

theorem prob_four_of_a_kind_after_re_roll :
  (probability_of_four_of_a_kind =
    (1 : ℚ) / 6) :=
sorry

end prob_four_of_a_kind_after_re_roll_l303_303870


namespace sandy_saved_last_year_percentage_l303_303978

theorem sandy_saved_last_year_percentage (S : ℝ) (P : ℝ) :
  (this_year_salary: ℝ) → (this_year_savings: ℝ) → 
  (this_year_saved_percentage: ℝ) → (saved_last_year_percentage: ℝ) → 
  this_year_salary = 1.1 * S → 
  this_year_saved_percentage = 6 →
  this_year_savings = (this_year_saved_percentage / 100) * this_year_salary →
  (this_year_savings / ((P / 100) * S)) = 0.66 →
  P = 10 :=
by
  -- The proof is to be filled in here.
  sorry

end sandy_saved_last_year_percentage_l303_303978


namespace simplify_fraction_l303_303482

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)

theorem simplify_fraction : (1 / a) + (1 / b) - (2 * a + b) / (2 * a * b) = 1 / (2 * a) :=
by
  sorry

end simplify_fraction_l303_303482


namespace jazmin_dolls_correct_l303_303634

-- Define the number of dolls Geraldine has.
def geraldine_dolls : ℕ := 2186

-- Define the number of extra dolls Geraldine has compared to Jazmin.
def extra_dolls : ℕ := 977

-- Define the calculation of the number of dolls Jazmin has.
def jazmin_dolls : ℕ := geraldine_dolls - extra_dolls

-- Prove that the number of dolls Jazmin has is 1209.
theorem jazmin_dolls_correct : jazmin_dolls = 1209 := by
  -- Include the required steps in the future proof here.
  sorry

end jazmin_dolls_correct_l303_303634


namespace pies_sold_in_week_l303_303035

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end pies_sold_in_week_l303_303035


namespace heights_inequality_l303_303982

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h₁ : a ≤ b) (h₂ : b ≤ c) :
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) :=
by
  sorry

end heights_inequality_l303_303982


namespace find_m_range_l303_303967

theorem find_m_range (m : ℝ) : 
  (∃ x : ℤ, 2 * (x : ℝ) - 1 ≤ 5 ∧ x - 1 ≥ m ∧ x ≤ 3) ∧ 
  (∃ y : ℤ, 2 * (y : ℝ) - 1 ≤ 5 ∧ y - 1 ≥ m ∧ y ≤ 3 ∧ x ≠ y) → 
  -1 < m ∧ m ≤ 0 := by
  sorry

end find_m_range_l303_303967


namespace avg_height_and_weight_of_class_l303_303969

-- Defining the given conditions
def num_students : ℕ := 70
def num_girls : ℕ := 40
def num_boys : ℕ := 30

def avg_height_30_girls : ℕ := 160
def avg_height_10_girls : ℕ := 156
def avg_height_15_boys_high : ℕ := 170
def avg_height_15_boys_low : ℕ := 160
def avg_weight_girls : ℕ := 55
def avg_weight_boys : ℕ := 60

-- Theorem stating the given question
theorem avg_height_and_weight_of_class :
  ∃ (avg_height avg_weight : ℚ),
    avg_height = (30 * 160 + 10 * 156 + 15 * 170 + 15 * 160) / num_students ∧
    avg_weight = (40 * 55 + 30 * 60) / num_students ∧
    avg_height = 161.57 ∧
    avg_weight = 57.14 :=
by
  -- include the solution steps here if required
  -- examples using appropriate constructs like ring, norm_num, etc.
  sorry

end avg_height_and_weight_of_class_l303_303969


namespace sufficient_not_necessary_condition_l303_303455

variable (x : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 2 → x > 1) ∧ (¬ (x > 1 → x > 2)) := by
sorry

end sufficient_not_necessary_condition_l303_303455


namespace min_translation_phi_l303_303877

theorem min_translation_phi (φ : ℝ) (hφ : φ > 0) : 
  (∃ k : ℤ, φ = (π / 3) - k * π) → φ = π / 3 := 
by 
  sorry

end min_translation_phi_l303_303877


namespace find_unknown_rate_l303_303756

variable {x : ℝ}

theorem find_unknown_rate (h : (3 * 100 + 1 * 150 + 2 * x) / 6 = 150) : x = 225 :=
by 
  sorry

end find_unknown_rate_l303_303756


namespace minimum_questions_to_identify_white_ball_l303_303571

theorem minimum_questions_to_identify_white_ball (n : ℕ) (even_white : ℕ) 
  (h₁ : n = 2004) 
  (h₂ : even_white % 2 = 0) 
  (h₃ : 1 ≤ even_white ∧ even_white ≤ n) :
  ∃ m : ℕ, m = 2003 := 
sorry

end minimum_questions_to_identify_white_ball_l303_303571


namespace geometric_series_first_term_l303_303613

theorem geometric_series_first_term (a : ℝ) (r : ℝ) (s : ℝ) 
  (h1 : r = -1/3) (h2 : s = 12) (h3 : s = a / (1 - r)) : a = 16 :=
by
  -- Placeholder for the proof
  sorry

end geometric_series_first_term_l303_303613


namespace distance_inequality_l303_303423

-- Given definitions for the proof context
variables {A B C P R S T : Point} -- Points in the Euclidean plane
variables {d D : ℝ} -- distances
variables (triangle_ABC : IsAcuteTriangle A B C) -- acute-angled triangle condition
variables (P_in_ABC : PointInTriangle P A B C) -- P is inside the triangle

-- Define maximum and minimum distances
def max_distance (P : Point) (A B C : Point) : ℝ :=
max (dist P A) (max (dist P B) (dist P C))

def min_distance (P : Point) (R S T : Point) : ℝ :=
min (dist P R) (min (dist P S) (dist P T))

-- Main theorem statement: D ≥ 2d, and D = 2d if and only if the triangle is equilateral
theorem distance_inequality (triangle_ABC : IsAcuteTriangle A B C) (P_in_ABC : PointInTriangle P A B C) :
  let D := max_distance P A B C,
      d := min_distance P R S T in
  D ≥ 2 * d ∧ (D = 2 * d ↔ IsEquilateralTriangle A B C) := 
sorry

end distance_inequality_l303_303423


namespace solve_for_x_l303_303547

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l303_303547


namespace dog_food_amount_l303_303615

theorem dog_food_amount (x : ℕ) (h1 : 3 * x + 6 = 15) : x = 3 :=
by {
  sorry
}

end dog_food_amount_l303_303615


namespace average_age_of_dogs_l303_303582

theorem average_age_of_dogs:
  let age1 := 10 in
  let age2 := age1 - 2 in
  let age3 := age2 + 4 in
  let age4 := age3 / 2 in
  let age5 := age4 + 20 in
  (age1 + age5) / 2 = 18 :=
by 
  sorry

end average_age_of_dogs_l303_303582


namespace nancy_seeds_in_big_garden_l303_303863

theorem nancy_seeds_in_big_garden :
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  seeds_in_big_garden = 28 := by
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  sorry

end nancy_seeds_in_big_garden_l303_303863


namespace domain_of_sqrt_expression_l303_303066

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end domain_of_sqrt_expression_l303_303066


namespace quadratic_eq_solutions_l303_303885

theorem quadratic_eq_solutions (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 3 := by
  sorry

end quadratic_eq_solutions_l303_303885


namespace probability_point_between_lines_l303_303101

theorem probability_point_between_lines :
  let l (x : ℝ) := -2 * x + 8
  let m (x : ℝ) := -3 * x + 9
  let area_l := 1 / 2 * 4 * 8
  let area_m := 1 / 2 * 3 * 9
  let area_between := area_l - area_m
  let probability := area_between / area_l
  probability = 0.16 :=
by
  sorry

end probability_point_between_lines_l303_303101


namespace average_salary_technicians_correct_l303_303287

section
variable (average_salary_all : ℝ)
variable (total_workers : ℕ)
variable (average_salary_rest : ℝ)
variable (num_technicians : ℕ)

noncomputable def average_salary_technicians
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : ℝ :=
  12000

theorem average_salary_technicians_correct
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : average_salary_technicians average_salary_all total_workers average_salary_rest num_technicians h1 h2 h3 h4 = 12000 :=
sorry

end

end average_salary_technicians_correct_l303_303287


namespace willam_tax_paid_l303_303432

-- Define our conditions
variables (T : ℝ) (tax_collected : ℝ) (willam_percent : ℝ)

-- Initialize the conditions according to the problem statement
def is_tax_collected (tax_collected : ℝ) : Prop := tax_collected = 3840
def is_farm_tax_levied_on_cultivated_land : Prop := true -- Essentially means we acknowledge it is 50%
def is_willam_taxable_land_percentage (willam_percent : ℝ) : Prop := willam_percent = 0.25

-- The final theorem that states Mr. Willam's tax payment is $960 given the conditions
theorem willam_tax_paid  : 
  ∀ (T : ℝ),
  is_tax_collected 3840 → 
  is_farm_tax_levied_on_cultivated_land →
  is_willam_taxable_land_percentage 0.25 →
  0.25 * 3840 = 960 :=
sorry

end willam_tax_paid_l303_303432


namespace find_marksman_hit_rate_l303_303183

-- Define the conditions
def independent_shots (p : ℝ) (n : ℕ) : Prop :=
  0 ≤ p ∧ p ≤ 1 ∧ (n ≥ 1)

def hit_probability (p : ℝ) (n : ℕ) : ℝ :=
  1 - (1 - p) ^ n

-- Stating the proof problem in Lean
theorem find_marksman_hit_rate (p : ℝ) (n : ℕ) 
  (h_independent : independent_shots p n) 
  (h_prob : hit_probability p n = 80 / 81) : 
  p = 2 / 3 :=
sorry

end find_marksman_hit_rate_l303_303183


namespace granger_bought_12_cans_of_spam_l303_303957

theorem granger_bought_12_cans_of_spam : 
  ∀ (S : ℕ), 
    (3 * 5 + 4 * 2 + 3 * S = 59) → 
    (S = 12) := 
by
  intro S h
  sorry

end granger_bought_12_cans_of_spam_l303_303957


namespace median_equality_and_range_inequality_l303_303637

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l303_303637


namespace solve_for_x_l303_303559

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l303_303559


namespace units_digit_of_quotient_l303_303048

theorem units_digit_of_quotient (n : ℕ) (h1 : n = 1987) : 
  (((4^n + 6^n) / 5) % 10) = 0 :=
by
  have pattern_4 : ∀ (k : ℕ), (4^k) % 10 = if k % 2 = 0 then 6 else 4 := sorry
  have pattern_6 : ∀ (k : ℕ), (6^k) % 10 = 6 := sorry
  have units_sum : (4^1987 % 10 + 6^1987 % 10) % 10 = 0 := sorry
  have multiple_of_5 : (4^1987 + 6^1987) % 5 = 0 := sorry
  sorry

end units_digit_of_quotient_l303_303048


namespace determine_S_l303_303284

theorem determine_S :
  (∃ k : ℝ, (∀ S R T : ℝ, R = k * (S / T)) ∧ (∃ S R T : ℝ, R = 2 ∧ S = 6 ∧ T = 3 ∧ 2 = k * (6 / 3))) →
  (∀ S R T : ℝ, R = 8 ∧ T = 2 → S = 16) :=
by
  sorry

end determine_S_l303_303284


namespace alpha_eq_pi_over_3_l303_303215

theorem alpha_eq_pi_over_3 (α β γ : ℝ) (h1 : 0 < α ∧ α < π) (h2 : α + β + γ = π) 
    (h3 : 2 * Real.sin α + Real.tan β + Real.tan γ = 2 * Real.sin α * Real.tan β * Real.tan γ) :
    α = π / 3 :=
by
  sorry

end alpha_eq_pi_over_3_l303_303215


namespace number_in_scientific_notation_l303_303849

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end number_in_scientific_notation_l303_303849


namespace pole_intersection_height_l303_303168

theorem pole_intersection_height :
  ∀ (d h1 h2 : ℝ), d = 120 ∧ h1 = 30 ∧ h2 = 90 → 
  ∃ y : ℝ, y = 18 :=
by
  sorry

end pole_intersection_height_l303_303168


namespace sufficient_but_not_necessary_l303_303098

variables {a b : ℝ}

theorem sufficient_but_not_necessary (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end sufficient_but_not_necessary_l303_303098


namespace jane_needs_9_more_days_l303_303852

def jane_rate : ℕ := 16
def mark_rate : ℕ := 20
def mark_days : ℕ := 3
def total_vases : ℕ := 248

def vases_by_mark_in_3_days : ℕ := mark_rate * mark_days
def vases_by_jane_and_mark_in_3_days : ℕ := (jane_rate + mark_rate) * mark_days
def remaining_vases_after_3_days : ℕ := total_vases - vases_by_jane_and_mark_in_3_days
def days_jane_needs_alone : ℕ := (remaining_vases_after_3_days + jane_rate - 1) / jane_rate

theorem jane_needs_9_more_days :
  days_jane_needs_alone = 9 :=
by
  sorry

end jane_needs_9_more_days_l303_303852


namespace time_to_cross_tree_l303_303460

theorem time_to_cross_tree (length_train : ℝ) (length_platform : ℝ) (time_to_pass_platform : ℝ) (h1 : length_train = 1200) (h2 : length_platform = 1200) (h3 : time_to_pass_platform = 240) : 
  (length_train / ((length_train + length_platform) / time_to_pass_platform)) = 120 := 
by
    sorry

end time_to_cross_tree_l303_303460


namespace polynomial_coeff_properties_l303_303950

theorem polynomial_coeff_properties :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 : ℤ,
  (∀ x : ℤ, (1 - 2 * x)^7 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) ∧
  a0 = 1 ∧
  (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = -1) ∧
  (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3^7)) :=
sorry

end polynomial_coeff_properties_l303_303950


namespace p_is_necessary_but_not_sufficient_for_q_l303_303071

variable (x : ℝ)
def p := |x| ≤ 2
def q := 0 ≤ x ∧ x ≤ 2

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬ q x := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l303_303071


namespace minValue_equality_l303_303698

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (a + 3 * b) * (b + 3 * c) * (a * c + 3)

theorem minValue_equality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  minValue a b c = 48 :=
sorry

end minValue_equality_l303_303698


namespace tiffany_optimal_area_l303_303443

def optimal_area (A : ℕ) : Prop :=
  ∃ l w : ℕ, l + w = 160 ∧ l ≥ 85 ∧ w ≥ 45 ∧ A = l * w

theorem tiffany_optimal_area : optimal_area 6375 :=
  sorry

end tiffany_optimal_area_l303_303443


namespace max_value_n_for_positive_an_l303_303518

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a d : ℤ) (n : ℤ) := a + (n - 1) * d

-- Define the sum of first n terms of an arithmetic sequence
noncomputable def sum_arith_seq (a d n : ℤ) := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
axiom S15_pos (a d : ℤ) : sum_arith_seq a d 15 > 0
axiom S16_neg (a d : ℤ) : sum_arith_seq a d 16 < 0

-- Proof problem
theorem max_value_n_for_positive_an (a d : ℤ) :
  ∃ n : ℤ, n = 8 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 8) → arithmetic_seq a d m > 0 :=
sorry

end max_value_n_for_positive_an_l303_303518


namespace gcd_max_digits_l303_303805

theorem gcd_max_digits (a b : ℕ) (h_a : a < 10^7) (h_b : b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) : Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_max_digits_l303_303805


namespace horizontal_asymptote_of_f_l303_303052

open Filter Real

def f (x : ℝ) : ℝ := (7 * x^2 - 15) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  tendsto f at_top (𝓝 (7 / 4)) :=
sorry

end horizontal_asymptote_of_f_l303_303052


namespace remainder_of_3x_plus_5y_l303_303860

-- Conditions and parameter definitions
def x (k : ℤ) := 13 * k + 7
def y (m : ℤ) := 17 * m + 11

-- Proof statement
theorem remainder_of_3x_plus_5y (k m : ℤ) : (3 * x k + 5 * y m) % 221 = 76 := by
  sorry

end remainder_of_3x_plus_5y_l303_303860


namespace betty_afternoon_catch_l303_303365

def flies_eaten_per_day := 2
def days_in_week := 7
def flies_needed_for_week := days_in_week * flies_eaten_per_day
def flies_caught_morning := 5
def additional_flies_needed := 4
def flies_currently_have := flies_needed_for_week - additional_flies_needed
def flies_caught_afternoon := flies_currently_have - flies_caught_morning
def flies_escaped := 1

theorem betty_afternoon_catch :
  flies_caught_afternoon + flies_escaped = 6 :=
by
  sorry

end betty_afternoon_catch_l303_303365


namespace beth_final_students_l303_303478

-- Define the initial conditions
def initial_students : ℕ := 150
def students_joined : ℕ := 30
def students_left : ℕ := 15

-- Define the number of students after the first additional year
def after_first_year : ℕ := initial_students + students_joined

-- Define the final number of students after students leaving
def final_students : ℕ := after_first_year - students_left

-- Theorem to prove the number of students in the final year
theorem beth_final_students : 
  final_students = 165 :=
by
  sorry

end beth_final_students_l303_303478


namespace sum_and_product_of_radical_l303_303879

theorem sum_and_product_of_radical (a b : ℝ) (h1 : 2 * a = -4) (h2 : a^2 - b = 1) :
  a + b = 1 :=
sorry

end sum_and_product_of_radical_l303_303879


namespace beam_count_represents_number_of_beams_l303_303041

def price := 6210
def transport_cost_per_beam := 3
def beam_condition (x : ℕ) : Prop := 
  transport_cost_per_beam * x * (x - 1) = price

theorem beam_count_represents_number_of_beams (x : ℕ) :
  beam_condition x → (∃ n : ℕ, x = n) := 
sorry

end beam_count_represents_number_of_beams_l303_303041


namespace fg_minus_gf_l303_303285

noncomputable def f (x : ℝ) : ℝ := 8 * x - 12
noncomputable def g (x : ℝ) : ℝ := x / 4 - 1

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -16 :=
by
  -- We skip the proof.
  sorry

end fg_minus_gf_l303_303285


namespace remainder_check_l303_303200

theorem remainder_check (q : ℕ) (n : ℕ) (h1 : q = 3^19) (h2 : n = 1162261460) : q % n = 7 := by
  rw [h1, h2]
  -- Proof skipped
  sorry

end remainder_check_l303_303200


namespace math_question_l303_303646

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l303_303646


namespace arithmetic_sequence_sum_l303_303701

theorem arithmetic_sequence_sum (S : ℕ → ℕ)
  (h₁ : S 3 = 9)
  (h₂ : S 6 = 36) :
  S 9 - S 6 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l303_303701


namespace sugar_content_of_mixture_l303_303204

theorem sugar_content_of_mixture 
  (volume_juice1 : ℝ) (conc_juice1 : ℝ)
  (volume_juice2 : ℝ) (conc_juice2 : ℝ) 
  (total_volume : ℝ) (total_sugar : ℝ) 
  (resulting_sugar_content : ℝ) :
  volume_juice1 = 2 →
  conc_juice1 = 0.1 →
  volume_juice2 = 3 →
  conc_juice2 = 0.15 →
  total_volume = volume_juice1 + volume_juice2 →
  total_sugar = (conc_juice1 * volume_juice1) + (conc_juice2 * volume_juice2) →
  resulting_sugar_content = (total_sugar / total_volume) * 100 →
  resulting_sugar_content = 13 :=
by
  intros
  sorry

end sugar_content_of_mixture_l303_303204


namespace geraldine_banana_count_l303_303811

variable (b : ℕ) -- the number of bananas Geraldine ate on June 1

theorem geraldine_banana_count 
    (h1 : (5 * b + 80 = 150)) 
    : (b + 32 = 46) :=
by
  sorry

end geraldine_banana_count_l303_303811


namespace binary_addition_to_hex_l303_303895

theorem binary_addition_to_hex :
  let n₁ := (0b11111111111 : ℕ)
  let n₂ := (0b11111111 : ℕ)
  n₁ + n₂ = 0x8FE :=
by {
  sorry
}

end binary_addition_to_hex_l303_303895


namespace jeff_total_distance_l303_303110

-- Define the conditions as constants
def speed1 : ℝ := 80
def time1 : ℝ := 3

def speed2 : ℝ := 50
def time2 : ℝ := 2

def speed3 : ℝ := 70
def time3 : ℝ := 1

def speed4 : ℝ := 60
def time4 : ℝ := 2

def speed5 : ℝ := 45
def time5 : ℝ := 3

def speed6 : ℝ := 40
def time6 : ℝ := 2

def speed7 : ℝ := 30
def time7 : ℝ := 2.5

-- Define the equation for the total distance traveled
def total_distance : ℝ :=
  speed1 * time1 + 
  speed2 * time2 + 
  speed3 * time3 + 
  speed4 * time4 + 
  speed5 * time5 + 
  speed6 * time6 + 
  speed7 * time7

-- Prove that the total distance is equal to 820 miles
theorem jeff_total_distance : total_distance = 820 := by
  sorry

end jeff_total_distance_l303_303110


namespace remainder_problem_l303_303984

theorem remainder_problem (f y z : ℤ) (k m n : ℤ) 
  (h1 : f % 5 = 3) 
  (h2 : y % 5 = 4)
  (h3 : z % 7 = 6)
  (h4 : (f + y) % 15 = 7)
  : (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 :=
by
  sorry

end remainder_problem_l303_303984


namespace transformation_result_l303_303102

def f (x y : ℝ) : ℝ × ℝ := (y, x)
def g (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem transformation_result : g (f (-6) (7)).1 (f (-6) (7)).2 = (-7, 6) :=
by
  sorry

end transformation_result_l303_303102


namespace hypotenuse_length_l303_303036

theorem hypotenuse_length (x y : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = 1350 * Real.pi) 
  (h2 : V2 = 2430 * Real.pi) 
  (h3 : (1/3) * Real.pi * y^2 * x = V1) 
  (h4 : (1/3) * Real.pi * x^2 * y = V2) 
  : Real.sqrt (x^2 + y^2) = Real.sqrt 954 :=
sorry

end hypotenuse_length_l303_303036


namespace simplify_expression_l303_303869

theorem simplify_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) :
  (a^2 - 6 * a + 9) / (a^2 - 2 * a) / (1 - 1 / (a - 2)) = (a - 3) / a :=
sorry

end simplify_expression_l303_303869


namespace circular_garden_area_l303_303331

theorem circular_garden_area
  (r : ℝ) (h_r : r = 16)
  (C A : ℝ) (h_C : C = 2 * Real.pi * r) (h_A : A = Real.pi * r^2)
  (fence_cond : C = 1 / 8 * A) :
  A = 256 * Real.pi := by
  sorry

end circular_garden_area_l303_303331


namespace Michelle_initial_crayons_l303_303861

variable (M : ℕ)  -- M is the number of crayons Michelle initially has
variable (J : ℕ := 2)  -- Janet has 2 crayons
variable (final_crayons : ℕ := 4)  -- After Janet gives her crayons to Michelle, Michelle has 4 crayons

theorem Michelle_initial_crayons : M + J = final_crayons → M = 2 :=
by
  intro h1
  sorry

end Michelle_initial_crayons_l303_303861


namespace arc_measure_BN_l303_303782

variables (M N C A B P : Point)

noncomputable def circle_semicircle (M N C A B P : Point) : Prop :=
  ∃ γ : Circle, (γ.center = C ∧ diameter γ M N) ∧
  (γ.on_circle A ∧ γ.on_circle B) ∧
  (C = midpoint M N) ∧
  (P ∈ line_through C N) ∧
  (∠ C A P = 10) ∧ (∠ C B P = 10) ∧
  (arc_measure γ M A = 40)

theorem arc_measure_BN 
  (M N C A B P : Point) (h : circle_semicircle M N C A B P) :
  arc_measure (Circle.mk C (dist C M)) B N = 20 :=
by 
  sorry

end arc_measure_BN_l303_303782


namespace min_value_of_f_l303_303820

noncomputable def f (x a : ℝ) := Real.exp (x - a) - Real.log (x + a) - 1

theorem min_value_of_f (a : ℝ) : 
  (0 < a) → (∃ x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end min_value_of_f_l303_303820


namespace solve_7_at_8_l303_303833

theorem solve_7_at_8 : (7 * 8) / (7 + 8 + 3) = 28 / 9 := by
  sorry

end solve_7_at_8_l303_303833


namespace taxi_fare_l303_303348

theorem taxi_fare (fare : ℕ → ℝ) (distance : ℕ) :
  (∀ d, d > 10 → fare d = 20 + (d - 10) * (140 / 70)) →
  fare 80 = 160 →
  fare 100 = 200 :=
by
  intros h_fare h_fare_80
  show fare 100 = 200
  sorry

end taxi_fare_l303_303348


namespace max_value_m_l303_303373

theorem max_value_m {m : ℝ} (h : ∀ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → m ≤ Real.tan x + 1) : m = 2 :=
sorry

end max_value_m_l303_303373


namespace Marcus_fit_pies_l303_303254

theorem Marcus_fit_pies (x : ℕ) 
(h1 : ∀ b, (7 * b - 8) = 27) : x = 5 := by
  sorry

end Marcus_fit_pies_l303_303254


namespace units_digit_base8_of_sum_34_8_47_8_l303_303210

def is_units_digit (n m : ℕ) (u : ℕ) := (n + m) % 8 = u

theorem units_digit_base8_of_sum_34_8_47_8 :
  ∀ (n m : ℕ), n = 34 ∧ m = 47 → (is_units_digit (n % 8) (m % 8) 3) :=
by
  intros n m h
  rw [h.1, h.2]
  sorry

end units_digit_base8_of_sum_34_8_47_8_l303_303210


namespace math_question_l303_303645

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l303_303645


namespace height_of_box_l303_303621

-- Define box dimensions
def box_length := 6
def box_width := 6

-- Define spherical radii
def radius_large := 3
def radius_small := 2

-- Define coordinates
def box_volume (h : ℝ) : Prop :=
  ∃ (z : ℝ), z = 2 + Real.sqrt 23 ∧ 
  z + radius_large = h

theorem height_of_box (h : ℝ) : box_volume h ↔ h = 5 + Real.sqrt 23 := by
  sorry

end height_of_box_l303_303621


namespace polynomial_simplification_l303_303136

theorem polynomial_simplification (x : ℝ) : (3 * x^2 + 6 * x - 5) - (2 * x^2 + 4 * x - 8) = x^2 + 2 * x + 3 := 
by 
  sorry

end polynomial_simplification_l303_303136


namespace complete_square_transform_l303_303007

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end complete_square_transform_l303_303007


namespace find_k_l303_303230

noncomputable def vec_a : ℝ × ℝ := (1, 2)
noncomputable def vec_b : ℝ × ℝ := (-3, 2)
noncomputable def vec_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
noncomputable def vec_a_minus_3b : ℝ × ℝ := (10, -4)

theorem find_k :
  ∃! k : ℝ, (vec_k_a_plus_b k).1 * vec_a_minus_3b.2 = (vec_k_a_plus_b k).2 * vec_a_minus_3b.1 ∧ k = -1 / 3 :=
by
  sorry

end find_k_l303_303230


namespace no_positive_n_for_prime_expr_l303_303799

noncomputable def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ m : ℤ, 1 < m → m < p → ¬ (m ∣ p))

theorem no_positive_n_for_prime_expr : 
  ∀ n : ℕ, 0 < n → ¬ is_prime (n^3 - 9 * n^2 + 23 * n - 17) := by
  sorry

end no_positive_n_for_prime_expr_l303_303799


namespace car_win_probability_l303_303589

-- Definitions from conditions
def total_cars : ℕ := 12
def p_X : ℚ := 1 / 6
def p_Y : ℚ := 1 / 10
def p_Z : ℚ := 1 / 8

-- Proof statement: The probability that one of the cars X, Y, or Z will win is 47/120
theorem car_win_probability : p_X + p_Y + p_Z = 47 / 120 := by
  sorry

end car_win_probability_l303_303589


namespace part_a_part_b_l303_303061

theorem part_a (x y : ℂ) : (3 * y + 5 * x * Complex.I = 15 - 7 * Complex.I) ↔ (x = -7/5 ∧ y = 5) := by
  sorry

theorem part_b (x y : ℝ) : (2 * x + 3 * y + (x - y) * Complex.I = 7 + 6 * Complex.I) ↔ (x = 5 ∧ y = -1) := by
  sorry

end part_a_part_b_l303_303061


namespace roy_missed_days_l303_303535

theorem roy_missed_days {hours_per_day days_per_week actual_hours_week missed_days : ℕ}
    (h1 : hours_per_day = 2)
    (h2 : days_per_week = 5)
    (h3 : actual_hours_week = 6)
    (expected_hours_week : ℕ := hours_per_day * days_per_week)
    (missed_hours : ℕ := expected_hours_week - actual_hours_week)
    (missed_days := missed_hours / hours_per_day) :
  missed_days = 2 := by
  sorry

end roy_missed_days_l303_303535


namespace ratio_square_correct_l303_303735

noncomputable def ratio_square (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) : ℝ :=
  let k := a / b
  let x := k * k
  x

theorem ratio_square_correct (a b : ℝ) (h : a / b = b / Real.sqrt (a^2 + b^2)) :
  ratio_square a b h = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end ratio_square_correct_l303_303735


namespace combined_weight_of_olivers_bags_l303_303705

theorem combined_weight_of_olivers_bags (w_james : ℕ) (w_oliver : ℕ) (w_combined : ℕ) 
  (h1 : w_james = 18) 
  (h2 : w_oliver = w_james / 6) 
  (h3 : w_combined = 2 * w_oliver) : 
  w_combined = 6 := 
by
  sorry

end combined_weight_of_olivers_bags_l303_303705


namespace sum_of_roots_l303_303252

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 2 * x1^2 - k * x1 = 2 * c) 
  (h2 : 2 * x2^2 - k * x2 = 2 * c) (h3 : x1 ≠ x2) : x1 + x2 = k / 2 := 
sorry

end sum_of_roots_l303_303252


namespace line_always_passes_fixed_point_l303_303379

theorem line_always_passes_fixed_point:
  ∀ a x y, x = 5 → y = -3 → (a * x + (2 * a - 1) * y + a - 3 = 0) :=
by
  intros a x y h1 h2
  rw [h1, h2]
  sorry

end line_always_passes_fixed_point_l303_303379


namespace triangle_area_l303_303190

def point := (ℚ × ℚ)

def vertex1 : point := (3, -3)
def vertex2 : point := (3, 4)
def vertex3 : point := (8, -3)

theorem triangle_area :
  let base := (vertex3.1 - vertex1.1 : ℚ)
  let height := (vertex2.2 - vertex1.2 : ℚ)
  (base * height / 2) = 17.5 :=
by
  sorry

end triangle_area_l303_303190


namespace beth_final_students_l303_303477

-- Define the initial conditions
def initial_students : ℕ := 150
def students_joined : ℕ := 30
def students_left : ℕ := 15

-- Define the number of students after the first additional year
def after_first_year : ℕ := initial_students + students_joined

-- Define the final number of students after students leaving
def final_students : ℕ := after_first_year - students_left

-- Theorem to prove the number of students in the final year
theorem beth_final_students : 
  final_students = 165 :=
by
  sorry

end beth_final_students_l303_303477


namespace problem_1_problem_2_problem_3_problem_4_l303_303787

-- Problem 1
theorem problem_1 : 4.7 + (-2.5) - (-5.3) - 7.5 = 0 := by
  sorry

-- Problem 2
theorem problem_2 : 18 + 48 / (-2)^2 - (-4)^2 * 5 = -50 := by
  sorry

-- Problem 3
theorem problem_3 : -1^4 + (-2)^2 / 4 * (5 - (-3)^2) = -5 := by
  sorry

-- Problem 4
theorem problem_4 : (-19 + 15 / 16) * 8 = -159 + 1 / 2 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l303_303787


namespace alloy_cut_weight_l303_303430

variable (a b x : ℝ)
variable (ha : 0 ≤ a ∧ a ≤ 1) -- assuming copper content is a fraction between 0 and 1
variable (hb : 0 ≤ b ∧ b ≤ 1)
variable (h : a ≠ b)
variable (hx : 0 < x ∧ x < 40) -- x is strictly between 0 and 40 (since 0 ≤ x ≤ 40)

theorem alloy_cut_weight (A B : ℝ) (hA : A = 40) (hB : B = 60) (h1 : (a * x + b * (A - x)) / 40 = (b * x + a * (B - x)) / 60) : x = 24 :=
by
  sorry

end alloy_cut_weight_l303_303430


namespace present_age_of_father_l303_303880

-- Definitions based on the conditions
variables (F S : ℕ)
axiom cond1 : F = 3 * S + 3
axiom cond2 : F + 3 = 2 * (S + 3) + 8

-- The theorem to prove
theorem present_age_of_father : F = 27 :=
by
  sorry

end present_age_of_father_l303_303880


namespace age_of_eldest_child_l303_303903

-- Define the conditions as hypotheses
def child_ages_sum_equals_50 (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50

-- Define the main theorem to prove the age of the eldest child
theorem age_of_eldest_child (x : ℕ) (h : child_ages_sum_equals_50 x) : x + 8 = 14 :=
sorry

end age_of_eldest_child_l303_303903


namespace Earl_rate_36_l303_303057

theorem Earl_rate_36 (E : ℝ) (h1 : E + (2 / 3) * E = 60) : E = 36 :=
by {
  sorry
}

end Earl_rate_36_l303_303057


namespace quadratic_inequality_solution_set_conclusions_l303_303677

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set_conclusions (h1 : ∀ x, -1 ≤ x ∧ x ≤ 2 → ax^2 + bx + c ≥ 0)
(h2 : ∀ x, x < -1 ∨ x > 2 → ax^2 + bx + c < 0) :
(a + b = 0) ∧ (a + b + c > 0) ∧ (c > 0) ∧ ¬ (b < 0) := by
sorry

end quadratic_inequality_solution_set_conclusions_l303_303677


namespace set_D_cannot_form_triangle_l303_303610

theorem set_D_cannot_form_triangle : ¬ (∃ a b c : ℝ, a = 2 ∧ b = 4 ∧ c = 6 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by {
  sorry
}

end set_D_cannot_form_triangle_l303_303610


namespace problem_statement_l303_303575

-- Define the basic problem setup
def defect_rate (p : ℝ) := p = 0.01
def sample_size (n : ℕ) := n = 200

-- Define the binomial distribution
noncomputable def binomial_expectation (n : ℕ) (p : ℝ) := n * p
noncomputable def binomial_variance (n : ℕ) (p : ℝ) := n * p * (1 - p)

-- The actual statement that we will prove
theorem problem_statement (p : ℝ) (n : ℕ) (X : ℕ → ℕ) 
  (h_defect_rate : defect_rate p) 
  (h_sample_size : sample_size n) 
  (h_distribution : ∀ k, X k = (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))) 
  : binomial_expectation n p = 2 ∧ binomial_variance n p = 1.98 :=
by
  sorry

end problem_statement_l303_303575


namespace xyz_squared_sum_l303_303931

def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[0, 3 * y, 2 * z],
   [2 * x, 2 * y, z],
   [x, -y, -z]]

def N_orthogonal (x y z : ℝ) : Prop :=
  (N x y z)ᵀ ⬝ (N x y z) = 1

theorem xyz_squared_sum (x y z : ℝ) (hN : N_orthogonal x y z) :
  x^2 + y^2 + z^2 = 46 / 105 :=
sorry

end xyz_squared_sum_l303_303931


namespace katie_earnings_l303_303412

theorem katie_earnings :
  4 * 3 + 3 * 7 + 2 * 5 + 5 * 2 = 53 := 
by 
  sorry

end katie_earnings_l303_303412


namespace range_of_a_l303_303090

-- Conditions for sets A and B
def SetA := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def SetB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 2}

-- Main statement to show that A ∪ B = A implies the range of a is [-2, 0]
theorem range_of_a (a : ℝ) : (SetB a ⊆ SetA) → (-2 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l303_303090


namespace probability_of_divisibility_l303_303837

-- Definitions based on given conditions

def digits : Finset ℕ := {1, 2, 3, 5, 5, 8, 0}

noncomputable def probability_divisible_by_30 : ℚ :=
  5 / 21

theorem probability_of_divisibility :
  let arrangements := digits.to_finset.permutations.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  let divisible_by_30 := arrangements.filter (λ n, n % 30 = 0)
  (divisible_by_30.card : ℚ) / (arrangements.card : ℚ) = probability_divisible_by_30 :=
by
  sorry

end probability_of_divisibility_l303_303837


namespace sequence_general_term_l303_303288

-- Define the sequence using a recurrence relation for clarity in formal proof
def a (n : ℕ) : ℕ :=
  if h : n > 0 then 2^n + 1 else 3

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → a n = 2^n + 1 := 
by 
  sorry

end sequence_general_term_l303_303288


namespace perfect_square_l303_303815

-- Define natural numbers m and n and the condition mn ∣ m^2 + n^2 + m
variables (m n : ℕ)

-- Define the condition as a hypothesis
def condition (m n : ℕ) : Prop := (m * n) ∣ (m ^ 2 + n ^ 2 + m)

-- The main theorem statement: if the condition holds, then m is a perfect square
theorem perfect_square (m n : ℕ) (h : condition m n) : ∃ k : ℕ, m = k ^ 2 :=
sorry

end perfect_square_l303_303815


namespace area_of_BEIH_l303_303755

structure Point where
  x : ℚ
  y : ℚ

def B : Point := ⟨0, 0⟩
def A : Point := ⟨0, 2⟩
def D : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩
def E : Point := ⟨0, 1⟩
def F : Point := ⟨1, 0⟩
def I : Point := ⟨2/5, 6/5⟩
def H : Point := ⟨2/3, 2/3⟩

def quadrilateral_area (p1 p2 p3 p4 : Point) : ℚ :=
  (1/2 : ℚ) * 
  ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - 
   (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

theorem area_of_BEIH : quadrilateral_area B E I H = 7 / 15 := sorry

end area_of_BEIH_l303_303755


namespace composite_has_at_least_three_divisors_l303_303763

def is_composite (n : ℕ) : Prop := ∃ d, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_divisors (n : ℕ) (h : is_composite n) : ∃ a b c, a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c :=
sorry

end composite_has_at_least_three_divisors_l303_303763


namespace probability_same_color_socks_l303_303411

-- Define the total number of socks and the groups
def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

-- Define combinatorial functions to calculate combinations
def comb (n m : ℕ) : ℕ := n.choose m

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  comb blue_socks 2 +
  comb green_socks 2 +
  comb red_socks 2

-- Calculate the total number of possible outcomes
def total_outcomes : ℕ := comb total_socks 2

-- Calculate the probability as a ratio of favorable outcomes to total outcomes
def probability := favorable_outcomes / total_outcomes

-- Prove the probability is 19/45
theorem probability_same_color_socks : probability = 19 / 45 := by
  sorry

end probability_same_color_socks_l303_303411


namespace average_age_of_first_and_fifth_fastest_dogs_l303_303579

-- Definitions based on the conditions
def first_dog_age := 10
def second_dog_age := first_dog_age - 2
def third_dog_age := second_dog_age + 4
def fourth_dog_age := third_dog_age / 2
def fifth_dog_age := fourth_dog_age + 20

-- Statement to prove
theorem average_age_of_first_and_fifth_fastest_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 := by
  -- Add your proof here
  sorry

end average_age_of_first_and_fifth_fastest_dogs_l303_303579


namespace average_age_first_and_fifth_dogs_l303_303581

-- Define the conditions
def first_dog_age : ℕ := 10
def second_dog_age : ℕ := first_dog_age - 2
def third_dog_age : ℕ := second_dog_age + 4
def fourth_dog_age : ℕ := third_dog_age / 2
def fifth_dog_age : ℕ := fourth_dog_age + 20

-- Define the goal statement
theorem average_age_first_and_fifth_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 :=
by
  sorry

end average_age_first_and_fifth_dogs_l303_303581


namespace travel_time_difference_l303_303761

theorem travel_time_difference 
  (speed : ℝ) (d1 d2 : ℝ) (h_speed : speed = 50) (h_d1 : d1 = 475) (h_d2 : d2 = 450) : 
  (d1 - d2) / speed * 60 = 30 := 
by 
  sorry

end travel_time_difference_l303_303761


namespace cyclic_quadrilateral_ptolemy_l303_303998

theorem cyclic_quadrilateral_ptolemy 
  (a b c d : ℝ) 
  (h : a + b + c + d = Real.pi) :
  Real.sin (a + b) * Real.sin (b + c) = Real.sin a * Real.sin c + Real.sin b * Real.sin d :=
by
  sorry

end cyclic_quadrilateral_ptolemy_l303_303998


namespace solve_problem_1_solve_problem_2_l303_303784

/-
Problem 1:
Given the equation 2(x - 1)^2 = 18, prove that x = 4 or x = -2.
-/
theorem solve_problem_1 (x : ℝ) : 2 * (x - 1)^2 = 18 → (x = 4 ∨ x = -2) :=
by
  sorry

/-
Problem 2:
Given the equation x^2 - 4x - 3 = 0, prove that x = 2 + √7 or x = 2 - √7.
-/
theorem solve_problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 → (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end solve_problem_1_solve_problem_2_l303_303784


namespace scientific_notation_correct_l303_303851

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end scientific_notation_correct_l303_303851


namespace Sup_iid_rvs_eq_sup_P_l303_303251

open Probability MeasureTheory

noncomputable theory

variables {Ω : Type*} {ξ ξ₁ ξ₂ : Ω → ℝ} 

-- Assuming ξ, ξ₁, ξ₂, ... are i.i.d. random variables
axiom i.i.d. : ∀ n, (xi : @MeasureTheory.Measure.toOuterMeasure Ω PMeasureSpace.real ℝ) == ξ₁

-- Defining the random variable supremum and x*
def xi_sup : ℝ := ⨆ n, ξ₁
def x_star : ℝ := ⨆ (x : ℝ) (h : PMeasureSpace.real < 1), x

-- Problem statement in Lean 4
theorem Sup_iid_rvs_eq_sup_P {Ω : Type*} [MeasureSpace Ω] [is_probability_measure P] : 
  (i.i.d. ξ₁ ξ₂) → (P ((λ x, xi_sup = x_star) = (1 : ℝ)) :=
by sorry

end Sup_iid_rvs_eq_sup_P_l303_303251


namespace find_f_f_2_l303_303500

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_f_2 :
  f (f 2) = 14 :=
by
sorry

end find_f_f_2_l303_303500


namespace solve_for_x_l303_303550

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l303_303550


namespace estimate_pi_l303_303357

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

end estimate_pi_l303_303357


namespace x_cubed_gt_y_squared_l303_303290

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_gt_y_squared_l303_303290


namespace smallest_fraction_division_l303_303808

theorem smallest_fraction_division (a b : ℕ) (h_coprime : Nat.gcd a b = 1) 
(h1 : ∃ n, (25 * a = n * 21 * b)) (h2 : ∃ m, (15 * a = m * 14 * b)) : (a = 42) ∧ (b = 5) := 
sorry

end smallest_fraction_division_l303_303808


namespace sum_of_largest_and_smallest_l303_303308

theorem sum_of_largest_and_smallest (d1 d2 d3 d4 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 3) (h4 : d4 = 9) :
  let largest := 9631
  let smallest := 1369
  largest + smallest = 11000 :=
by
  let largest := 9631
  let smallest := 1369
  sorry

end sum_of_largest_and_smallest_l303_303308


namespace percentage_of_female_officers_on_duty_l303_303995

theorem percentage_of_female_officers_on_duty :
  ∀ (total_on_duty : ℕ) (half_on_duty : ℕ) (total_female_officers : ℕ), 
  total_on_duty = 204 → half_on_duty = total_on_duty / 2 → total_female_officers = 600 → 
  ((half_on_duty: ℚ) / total_female_officers) * 100 = 17 :=
by
  intro total_on_duty half_on_duty total_female_officers
  intros h1 h2 h3
  sorry

end percentage_of_female_officers_on_duty_l303_303995


namespace cute_angle_of_isosceles_cute_triangle_l303_303673

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end cute_angle_of_isosceles_cute_triangle_l303_303673


namespace calculate_gross_profit_l303_303757

theorem calculate_gross_profit (sales_price : ℝ) (cost : ℝ) (gross_profit : ℝ) 
    (h1 : sales_price = 81)
    (h2 : gross_profit = 1.70 * cost)
    (h3 : sales_price = cost + gross_profit) : gross_profit = 51 :=
by
  sorry

end calculate_gross_profit_l303_303757


namespace find_number_l303_303746

theorem find_number (x : ℝ) 
  (h : 0.4 * x + (0.3 * 0.2) = 0.26) : x = 0.5 := 
by
  sorry

end find_number_l303_303746


namespace sin_x_cos_x_value_l303_303097

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end sin_x_cos_x_value_l303_303097


namespace positive_difference_of_perimeters_is_zero_l303_303796

-- Definitions of given conditions
def rect1_length : ℕ := 5
def rect1_width : ℕ := 1
def rect2_first_rect_length : ℕ := 3
def rect2_first_rect_width : ℕ := 2
def rect2_second_rect_length : ℕ := 1
def rect2_second_rect_width : ℕ := 2

-- Perimeter calculation functions
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)
def rect1_perimeter := perimeter rect1_length rect1_width
def rect2_extended_length : ℕ := rect2_first_rect_length + rect2_second_rect_length
def rect2_extended_width : ℕ := rect2_first_rect_width
def rect2_perimeter := perimeter rect2_extended_length rect2_extended_width

-- The positive difference of the perimeters
def positive_difference (a b : ℕ) : ℕ := if a > b then a - b else b - a

-- The Lean 4 statement to be proven
theorem positive_difference_of_perimeters_is_zero :
    positive_difference rect1_perimeter rect2_perimeter = 0 := by
  sorry

end positive_difference_of_perimeters_is_zero_l303_303796


namespace finite_negatives_condition_l303_303235

-- Define the sequence terms
def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + n * d

-- Define the condition for finite negative terms
def has_finite_negatives (a1 d : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, arithmetic_seq a1 d n ≥ 0

-- Theorem that proves the desired statement
theorem finite_negatives_condition (a1 d : ℝ) (h1 : a1 < 0) (h2 : d > 0) :
  has_finite_negatives a1 d :=
sorry

end finite_negatives_condition_l303_303235


namespace fifth_term_is_19_l303_303403

-- Define the first term and the common difference
def a₁ : Int := 3
def d : Int := 4

-- Define the formula for the nth term in the arithmetic sequence
def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

-- Define the Lean 4 statement proving that the 5th term is 19
theorem fifth_term_is_19 : arithmetic_sequence 5 = 19 :=
by
  sorry -- Proof to be filled in

end fifth_term_is_19_l303_303403


namespace inf_solutions_l303_303868

theorem inf_solutions (x y z : ℤ) : 
  ∃ (infinitely many relatively prime solutions : ℕ), x^2 + y^2 = z^5 + z :=
sorry

end inf_solutions_l303_303868


namespace solution_set_of_inequality_l303_303884

theorem solution_set_of_inequality (a x : ℝ) (h : a > 0) : 
  (x^2 - (a + 1/a + 1) * x + a + 1/a < 0) ↔ (1 < x ∧ x < a + 1/a) :=
by sorry

end solution_set_of_inequality_l303_303884


namespace min_value_geq_9div2_l303_303523

noncomputable def min_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) : ℝ := 
  (x + y + z : ℝ) * ((1 : ℝ) / (x + y) + (1 : ℝ) / (x + z) + (1 : ℝ) / (y + z))

theorem min_value_geq_9div2 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) :
  min_value x y z hx hy hz h_sum ≥ 9 / 2 := 
sorry

end min_value_geq_9div2_l303_303523


namespace domain_of_sqrt_expression_l303_303065

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end domain_of_sqrt_expression_l303_303065


namespace quadratic_switch_real_roots_l303_303504

theorem quadratic_switch_real_roots (a b c u v w : ℝ) (ha : a ≠ u)
  (h_root1 : b^2 - 4 * a * c ≥ 0)
  (h_root2 : v^2 - 4 * u * w ≥ 0)
  (hwc : w * c > 0) :
  (b^2 - 4 * u * c ≥ 0) ∨ (v^2 - 4 * a * w ≥ 0) :=
sorry

end quadratic_switch_real_roots_l303_303504


namespace natural_number_base_conversion_l303_303184

theorem natural_number_base_conversion (n : ℕ) (h7 : n = 4 * 7 + 1) (h9 : n = 3 * 9 + 2) : 
  n = 3 * 8 + 5 := 
by 
  sorry

end natural_number_base_conversion_l303_303184


namespace tangent_intersection_locus_l303_303084

theorem tangent_intersection_locus :
  ∀ (l : ℝ → ℝ) (C : ℝ → ℝ), 
  (∀ x > 0, C x = x + 1/x) →
  (∃ k : ℝ, ∀ x, l x = k * x + 1) →
  ∃ (P : ℝ × ℝ), (P = (2, 2)) ∨ (P = (2, 5/2)) :=
by sorry

end tangent_intersection_locus_l303_303084


namespace find_p_q_sum_l303_303203

theorem find_p_q_sum (p q : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = 0 → 3 * x ^ 2 - p * x + q = 0) →
  p = 24 ∧ q = 45 ∧ p + q = 69 :=
by
  intros h
  have h3 := h 3 (by ring)
  have h5 := h 5 (by ring)
  sorry

end find_p_q_sum_l303_303203


namespace line_equation_through_point_l303_303803

theorem line_equation_through_point 
  (x y : ℝ)
  (h1 : (5, 2) ∈ {p : ℝ × ℝ | p.2 = p.1 * (2 / 5)})
  (h2 : (5, 2) ∈ {p : ℝ × ℝ | p.1 / 6 + p.2 / 12 = 1}) 
  (h3 : (5,2) ∈ {p : ℝ × ℝ | 2 * p.1 = p.2 }) :
  (2 * x + y - 12 = 0 ∨ 
   2 * x - 5 * y = 0) := 
sorry

end line_equation_through_point_l303_303803


namespace simple_interest_problem_l303_303179

theorem simple_interest_problem 
  (P R : ℝ)
  (h1 : 600 = (P * R * 10) / 100)
  (h2 : ∃ (P : ℝ), (R = 6000 / P) ∧ (600 = (P * (6000 / P) * 10) / 100))
  : 
  let I1 := (P * R * 5) / 100,
      I2 := (3 * P * R * 5) / 100
  in I1 + I2 = 1200 :=
by
  sorry

end simple_interest_problem_l303_303179


namespace observer_height_proof_l303_303925

noncomputable def height_observer (d m α β : ℝ) : ℝ :=
  let cot_alpha := 1 / Real.tan α
  let cot_beta := 1 / Real.tan β
  let u := (d * (m * cot_beta - d)) / (2 * d - m * (cot_beta - cot_alpha))
  20 + Real.sqrt (400 + u * m * cot_alpha - u^2)

theorem observer_height_proof :
  height_observer 290 40 (11.4 * Real.pi / 180) (4.7 * Real.pi / 180) = 52 := sorry

end observer_height_proof_l303_303925


namespace not_possible_select_seven_distinct_weights_no_equal_subsets_l303_303948

theorem not_possible_select_seven_distinct_weights_no_equal_subsets :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 27 → s.card = 7 → ∃ (a b : Finset ℕ), a ≠ b ∧ a ⊆ s ∧ b ⊆ s ∧ a.sum id = b.sum id :=
by
  intro s hs hcard
  sorry

end not_possible_select_seven_distinct_weights_no_equal_subsets_l303_303948


namespace increasing_function_unique_root_proof_l303_303822

noncomputable def increasing_function_unique_root (f : ℝ → ℝ) :=
  (∀ x y : ℝ, x < y → f x ≤ f y) -- condition for increasing function
  ∧ ∃! x : ℝ, f x = 0 -- exists exactly one root

theorem increasing_function_unique_root_proof
  (f : ℝ → ℝ)
  (h_inc : ∀ x y : ℝ, x < y → f x ≤ f y)
  (h_ex : ∃ x : ℝ, f x = 0) :
  ∃! x : ℝ, f x = 0 := sorry

end increasing_function_unique_root_proof_l303_303822


namespace ammonium_iodide_molecular_weight_l303_303046

theorem ammonium_iodide_molecular_weight :
  let N := 14.01
  let H := 1.008
  let I := 126.90
  let NH4I_weight := (1 * N) + (4 * H) + (1 * I)
  NH4I_weight = 144.942 :=
by
  -- The proof will go here
  sorry

end ammonium_iodide_molecular_weight_l303_303046


namespace possible_values_y_l303_303417

theorem possible_values_y (x : ℝ) (h : x^2 + 4 * (x / (x - 2))^2 = 45) : 
  ∃ y : ℝ, y = 2 ∨ y = 16 :=
sorry

end possible_values_y_l303_303417


namespace x_minus_y_values_l303_303217

theorem x_minus_y_values (x y : ℝ) (h₁ : |x + 1| = 4) (h₂ : (y + 2)^2 = 4) (h₃ : x + y ≥ -5) :
  x - y = -5 ∨ x - y = 3 ∨ x - y = 7 :=
by
  sorry

end x_minus_y_values_l303_303217


namespace beth_students_proof_l303_303475

-- Let initial := 150
-- Let joined := 30
-- Let left := 15
-- final := initial + joined - left
-- Prove final = 165

def beth_final_year_students (initial joined left final : ℕ) : Prop :=
  initial = 150 ∧ joined = 30 ∧ left = 15 ∧ final = initial + joined - left

theorem beth_students_proof : ∃ final, beth_final_year_students 150 30 15 final ∧ final = 165 :=
by
  sorry

end beth_students_proof_l303_303475


namespace revenue_change_l303_303759

theorem revenue_change (x : ℝ) 
  (increase_in_1996 : ∀ R : ℝ, R * (1 + x/100) > R) 
  (decrease_in_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) < R * (1 + x/100)) 
  (decrease_from_1995_to_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) = R * 0.96): 
  x = 20 :=
by
  sorry

end revenue_change_l303_303759


namespace log_49_48_in_terms_of_a_and_b_l303_303953

-- Define the constants and hypotheses
variable (a b : ℝ)
variable (h1 : a = Real.logb 7 3)
variable (h2 : b = Real.logb 7 4)

-- Define the statement to be proved
theorem log_49_48_in_terms_of_a_and_b (a b : ℝ) (h1 : a = Real.logb 7 3) (h2 : b = Real.logb 7 4) :
  Real.logb 49 48 = (a + 2 * b) / 2 :=
by
  sorry

end log_49_48_in_terms_of_a_and_b_l303_303953


namespace trigonometric_expression_evaluation_l303_303202

theorem trigonometric_expression_evaluation :
  let tan30 := (Real.sqrt 3) / 3
  let sin60 := (Real.sqrt 3) / 2
  let cot60 := 1 / (Real.sqrt 3)
  let tan60 := Real.sqrt 3
  let cos45 := (Real.sqrt 2) / 2
  (3 * tan30) / (1 - sin60) + (cot60 + Real.cos (Real.pi * 70 / 180))^0 - tan60 / (cos45^4) = 7 :=
by
  -- This is where the proof would go
  sorry

end trigonometric_expression_evaluation_l303_303202


namespace acceleration_inverse_square_distance_l303_303340

noncomputable def s (t : ℝ) : ℝ := t^(2/3)

noncomputable def v (t : ℝ) : ℝ := (deriv s t : ℝ)

noncomputable def a (t : ℝ) : ℝ := (deriv v t : ℝ)

theorem acceleration_inverse_square_distance
  (t : ℝ) (h : t ≠ 0) :
  ∃ k : ℝ, k = -2/9 ∧ a t = k / (s t)^2 :=
sorry

end acceleration_inverse_square_distance_l303_303340


namespace gcd_number_between_75_and_90_is_5_l303_303566

theorem gcd_number_between_75_and_90_is_5 :
  ∃ n : ℕ, 75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 15 n = 5 :=
sorry

end gcd_number_between_75_and_90_is_5_l303_303566


namespace model_y_completion_time_l303_303029

theorem model_y_completion_time
  (rate_model_x : ℕ → ℝ)
  (rate_model_y : ℕ → ℝ)
  (num_model_x : ℕ)
  (num_model_y : ℕ)
  (time_model_x : ℝ)
  (combined_rate : ℝ)
  (same_number : num_model_y = num_model_x)
  (task_completion_x : ∀ x, rate_model_x x = 1 / time_model_x)
  (total_model_x : num_model_x = 24)
  (task_completion_y : ∀ y, rate_model_y y = 1 / y)
  (one_minute_completion : num_model_x * rate_model_x 1 + num_model_y * rate_model_y 36 = combined_rate)
  : 36 = time_model_x * 2 :=
by
  sorry

end model_y_completion_time_l303_303029


namespace average_age_increase_by_one_l303_303724

-- Definitions based on the conditions.
def initial_average_age : ℕ := 14
def initial_students : ℕ := 10
def new_students_average_age : ℕ := 17
def new_students : ℕ := 5

-- Helper calculation for the total age of initial students.
def total_age_initial_students := initial_students * initial_average_age

-- Helper calculation for the total age of new students.
def total_age_new_students := new_students * new_students_average_age

-- Helper calculation for the total age of all students.
def total_age_all_students := total_age_initial_students + total_age_new_students

-- Helper calculation for the number of all students.
def total_students := initial_students + new_students

-- Calculate the new average age.
def new_average_age := total_age_all_students / total_students

-- The goal is to prove the increase in average age is 1 year.
theorem average_age_increase_by_one :
  new_average_age - initial_average_age = 1 :=
by
  -- Proof goes here
  sorry

end average_age_increase_by_one_l303_303724


namespace inequality_holds_l303_303568

theorem inequality_holds (m : ℝ) (h : 0 ≤ m ∧ m < 12) :
  ∀ x : ℝ, 3 * m * x ^ 2 + m * x + 1 > 0 :=
sorry

end inequality_holds_l303_303568


namespace highest_degree_divisibility_l303_303628

-- Definition of the problem settings
def prime_number := 1991
def number_1 := 1990 ^ (1991 ^ 1002)
def number_2 := 1992 ^ (1501 ^ 1901)
def combined_number := number_1 + number_2

-- Statement of the proof to be formalized
theorem highest_degree_divisibility (k : ℕ) : k = 1001 ∧ prime_number ^ k ∣ combined_number := by
  sorry

end highest_degree_divisibility_l303_303628


namespace luther_latest_line_count_l303_303420

theorem luther_latest_line_count :
  let silk := 10
  let cashmere := silk / 2
  let blended := 2
  silk + cashmere + blended = 17 :=
by
  sorry

end luther_latest_line_count_l303_303420


namespace fraction_of_juniors_equals_seniors_l303_303972

theorem fraction_of_juniors_equals_seniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J * 7 = 4 * (J + S)) : J / S = 4 / 3 :=
sorry

end fraction_of_juniors_equals_seniors_l303_303972


namespace inequality_proof_l303_303224

theorem inequality_proof (x : ℝ) (n : ℕ) (h : 3 * x ≥ -1) : (1 + x) ^ n ≥ 1 + n * x :=
sorry

end inequality_proof_l303_303224


namespace original_daily_production_l303_303180

theorem original_daily_production (x N : ℕ) (h1 : N = (x - 3) * 31 + 60) (h2 : N = (x + 3) * 25 - 60) : x = 8 :=
sorry

end original_daily_production_l303_303180


namespace problem_statement_l303_303522

def f (x : ℕ) : ℝ := sorry

theorem problem_statement (h_cond : ∀ k : ℕ, f k ≤ (k : ℝ) ^ 2 → f (k + 1) ≤ (k + 1 : ℝ) ^ 2)
    (h_f7 : f 7 = 50) : ∀ k : ℕ, k ≤ 7 → f k > (k : ℝ) ^ 2 :=
sorry

end problem_statement_l303_303522


namespace ratio_Rose_to_Mother_l303_303260

variable (Rose_age : ℕ) (Mother_age : ℕ)

-- Define the conditions
axiom sum_of_ages : Rose_age + Mother_age = 100
axiom Rose_is_25 : Rose_age = 25
axiom Mother_is_75 : Mother_age = 75

-- Define the main theorem to prove the ratio
theorem ratio_Rose_to_Mother : (Rose_age : ℚ) / (Mother_age : ℚ) = 1 / 3 := by
  sorry

end ratio_Rose_to_Mother_l303_303260


namespace ab_value_l303_303819

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end ab_value_l303_303819


namespace f_2_values_l303_303495

theorem f_2_values (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, |f x - f y| = |x - y|)
  (hf1 : f 1 = 3) :
  f 2 = 2 ∨ f 2 = 4 :=
sorry

end f_2_values_l303_303495


namespace find_rectangle_width_l303_303562

noncomputable def area_of_square_eq_5times_area_of_rectangle (s l : ℝ) (w : ℝ) :=
  s^2 = 5 * (l * w)

noncomputable def perimeter_of_square_eq_160 (s : ℝ) :=
  4 * s = 160

theorem find_rectangle_width : ∃ w : ℝ, ∀ l : ℝ, 
  area_of_square_eq_5times_area_of_rectangle 40 l w ∧
  perimeter_of_square_eq_160 40 → 
  w = 10 :=
by
  sorry

end find_rectangle_width_l303_303562


namespace find_a_from_conditions_l303_303733

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end find_a_from_conditions_l303_303733


namespace tetrahedron_volume_formula_l303_303858

variables (r₀ S₀ S₁ S₂ S₃ V : ℝ)

theorem tetrahedron_volume_formula
  (h : V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀) :
  V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ :=
by { sorry }

end tetrahedron_volume_formula_l303_303858


namespace least_common_denominator_l303_303935

-- Define the list of numbers
def numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the least common multiple function
noncomputable def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Define the main theorem
theorem least_common_denominator : lcm_list numbers = 2520 := 
  by sorry

end least_common_denominator_l303_303935


namespace count_integer_values_of_x_l303_303386

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l303_303386


namespace speed_second_boy_l303_303445

theorem speed_second_boy (v : ℝ) (t : ℝ) (d : ℝ) (s₁ : ℝ) :
  s₁ = 4.5 ∧ t = 9.5 ∧ d = 9.5 ∧ (d = (v - s₁) * t) → v = 5.5 :=
by
  intros h
  obtain ⟨hs₁, ht, hd, hev⟩ := h
  sorry

end speed_second_boy_l303_303445


namespace cos_sum_simplified_l303_303712

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end cos_sum_simplified_l303_303712


namespace find_savings_l303_303017

theorem find_savings (income expenditure : ℕ) (ratio_income_expenditure : ℕ × ℕ) (income_value : income = 40000)
    (ratio_condition : ratio_income_expenditure = (8, 7)) :
    income - expenditure = 5000 :=
by
  sorry

end find_savings_l303_303017


namespace proof_of_problem_l303_303618

def problem_statement : Prop :=
  2 * Real.cos (Real.pi / 4) + abs (Real.sqrt 2 - 3)
  - (1 / 3) ^ (-2 : ℤ) + (2021 - Real.pi) ^ 0 = -5

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l303_303618


namespace max_bars_scenario_a_max_bars_scenario_b_l303_303453

-- Define the game conditions and the maximum bars Ivan can take in each scenario.

def max_bars_taken (initial_bars : ℕ) : ℕ :=
  if initial_bars = 14 then 13 else 13

theorem max_bars_scenario_a :
  max_bars_taken 13 = 13 :=
by sorry

theorem max_bars_scenario_b :
  max_bars_taken 14 = 13 :=
by sorry

end max_bars_scenario_a_max_bars_scenario_b_l303_303453


namespace pow_mul_eq_add_l303_303484

variable (a : ℝ)

theorem pow_mul_eq_add : a^2 * a^3 = a^5 := 
by 
  sorry

end pow_mul_eq_add_l303_303484


namespace incorrect_statement_about_zero_l303_303611

theorem incorrect_statement_about_zero :
  ¬ (0 > 0) :=
by
  sorry

end incorrect_statement_about_zero_l303_303611


namespace solve_for_x_l303_303548

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l303_303548


namespace range_of_a_l303_303501

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), (0 < x1 ∧ x1 < x2 ∧ x2 < 1) → (a * x2 - x2^3) - (a * x1 - x1^3) > x2 - x1) : a ≥ 4 :=
sorry


end range_of_a_l303_303501


namespace max_n_arithmetic_sequences_l303_303781

theorem max_n_arithmetic_sequences (a b : ℕ → ℤ) 
  (ha : ∀ n, a n = 1 + (n - 1) * 1)  -- Assuming x = 1 for simplicity, as per solution x = y = 1
  (hb : ∀ n, b n = 1 + (n - 1) * 1)  -- Assuming y = 1
  (a1 : a 1 = 1)
  (b1 : b 1 = 1)
  (a2_leq_b2 : a 2 ≤ b 2)
  (hn : ∃ n, a n * b n = 1764) :
  ∃ n, n = 44 ∧ a n * b n = 1764 :=
by
  sorry

end max_n_arithmetic_sequences_l303_303781


namespace distance_between_stripes_l303_303345

theorem distance_between_stripes (d₁ d₂ L W : ℝ) (h : ℝ)
  (h₁ : d₁ = 60)  -- distance between parallel curbs
  (h₂ : L = 30)  -- length of the curb between stripes
  (h₃ : d₂ = 80)  -- length of each stripe
  (area_eq : W * L = 1800) -- area of the parallelogram with base L
: h = 22.5 :=
by
  -- This is to assume the equation derived from area calculation
  have area_eq' : d₂ * h = 1800 := by sorry
  -- Solving for h using the derived area equation
  have h_calc : h = 1800 / 80 := by sorry
  -- Simplifying the result
  have h_simplified : h = 22.5 := by sorry
  exact h_simplified

end distance_between_stripes_l303_303345


namespace compare_y1_y2_l303_303653

theorem compare_y1_y2 (m y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 2*(-1) + m) 
  (h2 : y2 = 2^2 - 2*2 + m) : 
  y1 > y2 := 
sorry

end compare_y1_y2_l303_303653


namespace smallest_3a_plus_1_l303_303508

theorem smallest_3a_plus_1 (a : ℝ) (h : 8 * a ^ 2 + 6 * a + 2 = 4) : 
  ∃ a, (8 * a ^ 2 + 6 * a + 2 = 4) ∧ min (3 * (-1) + 1) (3 * (1 / 4) + 1) = -2 :=
by {
  sorry
}

end smallest_3a_plus_1_l303_303508


namespace no_beverages_l303_303684

noncomputable def businessmen := 30
def coffee := 15
def tea := 13
def water := 6
def coffee_tea := 7
def tea_water := 3
def coffee_water := 2
def all_three := 1

theorem no_beverages (businessmen coffee tea water coffee_tea tea_water coffee_water all_three):
  businessmen - (coffee + tea + water - coffee_tea - tea_water - coffee_water + all_three) = 7 :=
by sorry

end no_beverages_l303_303684


namespace solution_set_f1_geq_4_min_value_pq_l303_303088

-- Define the function f(x) for the first question
def f1 (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for part (I)
theorem solution_set_f1_geq_4 (x : ℝ) : f1 x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 :=
by
  sorry

-- Define the function f(x) for the second question
def f2 (m x : ℝ) : ℝ := |x - m| + |x - 3|

-- Theorem for part (II)
theorem min_value_pq (p q m : ℝ) (h_pos_p : p > 0) (h_pos_q : q > 0)
    (h_eq : 1 / p + 1 / (2 * q) = m)
    (h_min_f : ∀ x : ℝ, f2 m x ≥ 3) :
    pq = 1 / 18 :=
by
  sorry

end solution_set_f1_geq_4_min_value_pq_l303_303088


namespace pictures_per_album_l303_303024

-- Define the problem conditions
def picturesFromPhone : Nat := 35
def picturesFromCamera : Nat := 5
def totalAlbums : Nat := 5

-- Define the total number of pictures
def totalPictures : Nat := picturesFromPhone + picturesFromCamera

-- Define what we need to prove
theorem pictures_per_album :
  totalPictures / totalAlbums = 8 := by
  sorry

end pictures_per_album_l303_303024


namespace Harkamal_total_payment_l303_303663

theorem Harkamal_total_payment :
  let cost_grapes := 10 * 70
  let cost_mangoes := 9 * 55
  let cost_apples := 12 * 80
  let cost_papayas := 7 * 45
  let cost_oranges := 15 * 30
  let cost_bananas := 5 * 25
  cost_grapes + cost_mangoes + cost_apples + cost_papayas + cost_oranges + cost_bananas = 3045 := by
  sorry

end Harkamal_total_payment_l303_303663


namespace solve_for_x_l303_303557

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l303_303557


namespace quadratic_variation_y_l303_303099

theorem quadratic_variation_y (k : ℝ) (x y : ℝ) (h1 : y = k * x^2) (h2 : (25 : ℝ) = k * (5 : ℝ)^2) :
  y = 25 :=
by
sorry

end quadratic_variation_y_l303_303099


namespace quadratic_real_roots_l303_303085

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_roots_l303_303085


namespace quiche_total_volume_l303_303256

theorem quiche_total_volume :
  ∀ (raw_spinach cream_cheese eggs : ℕ),
    raw_spinach = 40 →
    cream_cheese = 6 →
    eggs = 4 →
    let cooked_spinach := raw_spinach * 20 / 100 in
    cooked_spinach + cream_cheese + eggs = 18 :=
by
  intros raw_spinach cream_cheese eggs h_raw h_cream h_eggs
  simp [h_raw, h_cream, h_eggs]
  -- rewriting the calculation step
  let cooked_spinach := raw_spinach * 20 / 100
  have h_cooked : cooked_spinach = 8 := by norm_num
  rw h_cooked
  norm_num
  -- closing the final proof by setting result to 18
  sorry

end quiche_total_volume_l303_303256


namespace freshmen_minus_sophomores_eq_24_l303_303323

def total_students := 800
def percent_juniors := 27 / 100
def percent_not_sophomores := 75 / 100
def number_seniors := 160

def number_juniors := percent_juniors * total_students
def number_not_sophomores := percent_not_sophomores * total_students
def number_sophomores := total_students - number_not_sophomores
def number_freshmen := total_students - (number_juniors + number_sophomores + number_seniors)

theorem freshmen_minus_sophomores_eq_24 :
  number_freshmen - number_sophomores = 24 :=
sorry

end freshmen_minus_sophomores_eq_24_l303_303323


namespace square_garden_perimeter_l303_303143

theorem square_garden_perimeter (q p : ℝ) (h : q = 2 * p + 20) : p = 40 :=
sorry

end square_garden_perimeter_l303_303143


namespace Nancy_folders_l303_303862

def n_initial : ℕ := 43
def n_deleted : ℕ := 31
def n_per_folder : ℕ := 6
def n_folders : ℕ := (n_initial - n_deleted) / n_per_folder

theorem Nancy_folders : n_folders = 2 := by
  sorry

end Nancy_folders_l303_303862


namespace locus_of_point_parabola_l303_303397

/-- If the distance from point P to the point F (4, 0) is one unit less than its distance to the line x + 5 = 0, then the equation of the locus of point P is y^2 = 16x. -/
theorem locus_of_point_parabola :
  ∀ P : ℝ × ℝ, dist P (4, 0) + 1 = abs (P.1 + 5) → P.2^2 = 16 * P.1 :=
by
  sorry

end locus_of_point_parabola_l303_303397


namespace sum_of_m_integers_l303_303512

theorem sum_of_m_integers :
  ∀ (m : ℤ), 
    (∀ (x : ℚ), (x - 10) / 5 ≤ -1 - x / 5 ∧ x - 1 > -m / 2) → 
    (∃ x_max x_min : ℤ, x_max + x_min = -2 ∧ 
                        (x_max ≤ 5 / 2 ∧ x_min ≤ 5 / 2) ∧ 
                        (1 - m / 2 < x_min ∧ 1 - m / 2 < x_max)) →
  (10 < m ∧ m ≤ 12) → m = 11 ∨ m = 12 → 11 + 12 = 23 :=
by sorry

end sum_of_m_integers_l303_303512


namespace Jaco_budget_for_parents_gifts_l303_303406

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end Jaco_budget_for_parents_gifts_l303_303406


namespace overall_percentage_decrease_l303_303977

theorem overall_percentage_decrease (P x y : ℝ) (hP : P = 100) 
  (h : (P - (x / 100) * P) - (y / 100) * (P - (x / 100) * P) = 55) : 
  ((P - 55) / P) * 100 = 45 := 
by 
  sorry

end overall_percentage_decrease_l303_303977


namespace geometric_sequence_inequality_l303_303245

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)

-- Conditions
def geometric_sequence (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ) : Prop :=
  a₂ = a₁ * q ∧
  a₃ = a₁ * q^2 ∧
  a₄ = a₁ * q^3 ∧
  a₅ = a₁ * q^4 ∧
  a₆ = a₁ * q^5 ∧
  a₇ = a₁ * q^6 ∧
  a₈ = a₁ * q^7

theorem geometric_sequence_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)
  (h_seq : geometric_sequence a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q)
  (h_a₁_pos : 0 < a₁)
  (h_q_ne_1 : q ≠ 1) :
  a₁ + a₈ > a₄ + a₅ :=
by 
-- Proof omitted
sorry

end geometric_sequence_inequality_l303_303245


namespace B_and_C_together_l303_303191

theorem B_and_C_together (A B C : ℕ) (h1 : A + B + C = 1000) (h2 : A + C = 700) (h3 : C = 300) :
  B + C = 600 :=
by
  sorry

end B_and_C_together_l303_303191


namespace amusement_park_admission_fees_l303_303431

theorem amusement_park_admission_fees
  (num_children : ℕ) (num_adults : ℕ)
  (fee_child : ℝ) (fee_adult : ℝ)
  (total_people : ℕ) (expected_total_fees : ℝ) :
  num_children = 180 →
  fee_child = 1.5 →
  fee_adult = 4.0 →
  total_people = 315 →
  expected_total_fees = 810 →
  num_children + num_adults = total_people →
  (num_children : ℝ) * fee_child + (num_adults : ℝ) * fee_adult = expected_total_fees := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amusement_park_admission_fees_l303_303431


namespace difference_of_results_l303_303614

theorem difference_of_results (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (h_diff: a ≠ b) :
  (70 * a - 7 * a) - (70 * b - 7 * b) = 0 :=
by
  sorry

end difference_of_results_l303_303614


namespace both_inequalities_equiv_l303_303349

theorem both_inequalities_equiv (x : ℝ) : (x - 3)/(2 - x) ≥ 0 ↔ (3 - x)/(x - 2) ≥ 0 := by
  sorry

end both_inequalities_equiv_l303_303349


namespace negation_proposition_true_l303_303152

theorem negation_proposition_true (x : ℝ) : (¬ (|x| > 1 → x > 1)) ↔ (|x| ≤ 1 → x ≤ 1) :=
by sorry

end negation_proposition_true_l303_303152


namespace B_profit_percentage_l303_303772

theorem B_profit_percentage (cost_price_A : ℝ) (profit_A : ℝ) (selling_price_C : ℝ) 
  (h1 : cost_price_A = 154) 
  (h2 : profit_A = 0.20) 
  (h3 : selling_price_C = 231) : 
  (selling_price_C - (cost_price_A * (1 + profit_A))) / (cost_price_A * (1 + profit_A)) * 100 = 25 :=
by
  sorry

end B_profit_percentage_l303_303772


namespace complex_number_value_l303_303367

open Complex

theorem complex_number_value (a : ℝ) 
  (h1 : z = (2 + a * I) / (1 + I)) 
  (h2 : (z.re, z.im) ∈ { p : ℝ × ℝ | p.2 = -p.1 }) : 
  a = 0 :=
by
  sorry

end complex_number_value_l303_303367


namespace quadratic_root_range_specific_m_value_l303_303074

theorem quadratic_root_range (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1^2 - 2 * (1 - m) * x1 + m^2 = 0 ∧ x2^2 - 2 * (1 - m) * x2 + m^2 = 0 ↔ m ≤ 1/2 :=
by
  sorry

theorem specific_m_value (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2 * (1 - m) * x1 + m^2 = 0)
  (h2 : x2^2 - 2 * (1 - m) * x2 + m^2 = 0) (h3 : x1^2 + 12 * m + x2^2 = 10) : 
  m = -3 :=
by
  sorry

end quadratic_root_range_specific_m_value_l303_303074


namespace initial_ratio_proof_l303_303329

variable (p q : ℕ) -- Define p and q as non-negative integers

-- Condition: The initial total volume of the mixture is 30 liters
def initial_volume (p q : ℕ) : Prop := p + q = 30

-- Condition: Adding 12 liters of q changes the ratio to 3:4
def new_ratio (p q : ℕ) : Prop := p * 4 = (q + 12) * 3

-- The final goal: prove the initial ratio is 3:2
def initial_ratio (p q : ℕ) : Prop := p * 2 = q * 3

-- The main proof problem statement
theorem initial_ratio_proof (p q : ℕ) 
  (h1 : initial_volume p q) 
  (h2 : new_ratio p q) : initial_ratio p q :=
  sorry

end initial_ratio_proof_l303_303329


namespace perpendicular_lines_intersect_at_point_l303_303739

theorem perpendicular_lines_intersect_at_point :
  ∀ (d k : ℝ), 
  (∀ x y, 3 * x - 4 * y = d ↔ 8 * x + k * y = d) → 
  (∃ x y, x = 2 ∧ y = -3 ∧ 3 * x - 4 * y = d ∧ 8 * x + k * y = d) → 
  d = -2 :=
by sorry

end perpendicular_lines_intersect_at_point_l303_303739


namespace mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l303_303257

-- Definition of Mr. Wang's movements
def movements : List Int := [6, -3, 10, -8, 12, -7, -10]

-- Definitions of given conditions
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.3

theorem mr_wang_returned_to_1st_floor :
  (List.sum movements = 0) :=
by
  sorry

theorem mr_wang_electricity_consumption :
  (List.sum (movements.map Int.natAbs) * floor_height * electricity_per_meter = 50.4) :=
by
  sorry

end mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l303_303257


namespace find_x_for_abs_expression_zero_l303_303801

theorem find_x_for_abs_expression_zero (x : ℚ) : |5 * x - 2| = 0 → x = 2 / 5 := by
  sorry

end find_x_for_abs_expression_zero_l303_303801


namespace slices_per_friend_l303_303780

theorem slices_per_friend (total_slices friends : ℕ) (h1 : total_slices = 16) (h2 : friends = 4) : (total_slices / friends) = 4 :=
by
  sorry

end slices_per_friend_l303_303780


namespace jacos_budget_l303_303408

theorem jacos_budget :
  (friends : Nat) (friend_gift_cost total_budget : Nat)
  (jaco_remainder_budget : Nat)
  : friends = 8 →
  friend_gift_cost = 9 →
  total_budget = 100 →
  jaco_remainder_budget = total_budget - (friends * friend_gift_cost) →
  (jaco_remainder_budget / 2) = 14 := by
  intros friends friend_gift_cost total_budget jaco_remainder_budget friends_eq friend_gift_cost_eq total_budget_eq jaco_remainder_budget_eq
  rw [friends_eq, friend_gift_cost_eq, total_budget_eq, jaco_remainder_budget_eq]
  simp
  sorry

end jacos_budget_l303_303408


namespace total_pies_sold_l303_303033

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end total_pies_sold_l303_303033


namespace real_inequality_l303_303955

theorem real_inequality
  (a1 a2 a3 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (S : ℝ)
  (hS : S = a1 + a2 + a3)
  (h4 : ∀ i ∈ [a1, a2, a3], (i^2 / (i - 1) > S)) :
  (1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1) := 
by
  sorry

end real_inequality_l303_303955


namespace coby_travel_time_l303_303793

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l303_303793


namespace dina_dolls_l303_303938

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end dina_dolls_l303_303938


namespace distribute_places_l303_303394

open Nat

theorem distribute_places (places schools : ℕ) (h_places : places = 7) (h_schools : schools = 3) : 
  ∃ n : ℕ, n = (Nat.choose (places - 1) (schools - 1)) ∧ n = 15 :=
by
  rw [h_places, h_schools]
  use 15
  , sorry

end distribute_places_l303_303394


namespace grove_town_fall_expenditure_l303_303725

-- Define the expenditures at the end of August and November
def expenditure_end_of_august : ℝ := 3.0
def expenditure_end_of_november : ℝ := 5.5

-- Define the spending during fall months (September, October, November)
def spending_during_fall_months : ℝ := 2.5

-- Statement to be proved
theorem grove_town_fall_expenditure :
  expenditure_end_of_november - expenditure_end_of_august = spending_during_fall_months :=
by
  sorry

end grove_town_fall_expenditure_l303_303725


namespace problem_equivalent_l303_303384

theorem problem_equivalent :
  2^1998 - 2^1997 - 2^1996 + 2^1995 = 3 * 2^1995 :=
by
  sorry

end problem_equivalent_l303_303384


namespace select_group_odd_number_of_girl_friends_l303_303762

variables {Girl Boy : Type}
variable (friends_with : Boy → Girl → Prop)

axiom each_boy_has_girl_friend : ∀ b : Boy, ∃ g : Girl, friends_with b g

theorem select_group_odd_number_of_girl_friends:
  ∃ (group : Finset (Girl ⊕ Boy)), 
  (group.card * 2 ≥ (Finset.univ : Finset (Girl ⊕ Boy)).card) ∧ 
  (∀ (b : Boy), b ∈ group →
    (group.filter_sum_left Girl Boy).filter (λ g : Girl, friends_with b g).card % 2 = 1) :=
sorry

end select_group_odd_number_of_girl_friends_l303_303762


namespace scientific_notation_correct_l303_303850

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end scientific_notation_correct_l303_303850


namespace least_positive_integer_special_property_l303_303944

/-- 
  Prove that 9990 is the least positive integer whose digits sum to a multiple of 27 
  and the number itself is not a multiple of 27.
-/
theorem least_positive_integer_special_property : ∃ n : ℕ, 
  n > 0 ∧ 
  (Nat.digits 10 n).sum % 27 = 0 ∧ 
  n % 27 ≠ 0 ∧ 
  ∀ m : ℕ, (m > 0 ∧ (Nat.digits 10 m).sum % 27 = 0 ∧ m % 27 ≠ 0 → n ≤ m) := 
by
  sorry

end least_positive_integer_special_property_l303_303944


namespace part1_part2_part3a_part3b_l303_303375

open Real

variable (θ : ℝ) (m : ℝ)

-- Conditions
axiom theta_domain : 0 < θ ∧ θ < 2 * π
axiom quadratic_eq : ∀ x : ℝ, 2 * x^2 - (sqrt 3 + 1) * x + m = 0
axiom roots_eq_theta : ∀ x : ℝ, (x = sin θ ∨ x = cos θ)

-- Proof statements
theorem part1 : 1 - cos θ ≠ 0 → 1 - tan θ ≠ 0 → 
  (sin θ / (1 - cos θ) + cos θ / (1 - tan θ)) = (3 + 5 * sqrt 3) / 4 := sorry

theorem part2 : sin θ * cos θ = m / 2 → m = sqrt 3 / 4 := sorry

theorem part3a : sin θ = sqrt 3 / 2 ∧ cos θ = 1 / 2 → θ = π / 3 := sorry

theorem part3b : sin θ = 1 / 2 ∧ cos θ = sqrt 3 / 2 → θ = π / 6 := sorry

end part1_part2_part3a_part3b_l303_303375


namespace count_multiples_200_to_400_l303_303383

def count_multiples_in_range (a b n : ℕ) : ℕ :=
  (b / n) - ((a + n - 1) / n) + 1

theorem count_multiples_200_to_400 :
  count_multiples_in_range 200 400 78 = 3 :=
by
  sorry

end count_multiples_200_to_400_l303_303383


namespace managers_participation_l303_303926

theorem managers_participation (teams : ℕ) (people_per_team : ℕ) (employees : ℕ) (total_people : teams * people_per_team = 6) (num_employees : employees = 3) :
  teams * people_per_team - employees = 3 :=
by
  sorry

end managers_participation_l303_303926


namespace initial_distance_l303_303633

-- Define conditions
def fred_speed : ℝ := 4
def sam_speed : ℝ := 4
def sam_distance_when_meet : ℝ := 20

-- States that the initial distance between Fred and Sam is 40 miles considering the given conditions.
theorem initial_distance (d : ℝ) (fred_speed_eq : fred_speed = 4) (sam_speed_eq : sam_speed = 4) (sam_distance_eq : sam_distance_when_meet = 20) :
  d = 40 :=
  sorry

end initial_distance_l303_303633


namespace product_of_two_numbers_ratio_l303_303307

theorem product_of_two_numbers_ratio {x y : ℝ}
  (h1 : x + y = (5/3) * (x - y))
  (h2 : x * y = 5 * (x - y)) :
  x * y = 56.25 := sorry

end product_of_two_numbers_ratio_l303_303307


namespace students_not_making_cut_l303_303301

theorem students_not_making_cut :
  let girls := 39
  let boys := 4
  let called_back := 26
  let total := girls + boys
  total - called_back = 17 :=
by
  -- add the proof here
  sorry

end students_not_making_cut_l303_303301


namespace five_point_questions_l303_303587

-- Defining the conditions as Lean statements
def question_count (x y : ℕ) : Prop := x + y = 30
def total_points (x y : ℕ) : Prop := 5 * x + 10 * y = 200

-- The theorem statement that states x equals the number of 5-point questions
theorem five_point_questions (x y : ℕ) (h1 : question_count x y) (h2 : total_points x y) : x = 20 :=
sorry -- Proof is omitted

end five_point_questions_l303_303587


namespace carla_marbles_l303_303788

theorem carla_marbles (before now bought : ℝ) (h_before : before = 187.0) (h_now : now = 321) : bought = 134 :=
by
  sorry

end carla_marbles_l303_303788


namespace Adam_total_shopping_cost_l303_303922

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end Adam_total_shopping_cost_l303_303922


namespace eq1_solution_eq2_solution_l303_303138

theorem eq1_solution (x : ℝ) : (x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2) ↔ (x^2 - 6 * x + 1 = 0) :=
by
  sorry

theorem eq2_solution (x : ℝ) : (x = 1 ∨ x = -5 / 2) ↔ (2 * x^2 + 3 * x - 5 = 0) :=
by
  sorry

end eq1_solution_eq2_solution_l303_303138


namespace car_miles_per_gallon_l303_303619

-- Define the conditions
def distance_home : ℕ := 220
def additional_distance : ℕ := 100
def total_distance : ℕ := distance_home + additional_distance
def tank_capacity : ℕ := 16 -- in gallons
def miles_per_gallon : ℕ := total_distance / tank_capacity

-- State the goal
theorem car_miles_per_gallon : miles_per_gallon = 20 := by
  sorry

end car_miles_per_gallon_l303_303619


namespace solve_fraction_eq_zero_l303_303100

theorem solve_fraction_eq_zero (x : ℝ) (h : (x - 3) / (2 * x + 5) = 0) (h2 : 2 * x + 5 ≠ 0) : x = 3 :=
sorry

end solve_fraction_eq_zero_l303_303100


namespace vets_recommend_yummy_dog_kibble_l303_303908

theorem vets_recommend_yummy_dog_kibble :
  (let total_vets := 1000
   let percentage_puppy_kibble := 20
   let vets_puppy_kibble := (percentage_puppy_kibble * total_vets) / 100
   let diff_yummy_puppy := 100
   let vets_yummy_kibble := vets_puppy_kibble + diff_yummy_puppy
   let percentage_yummy_kibble := (vets_yummy_kibble * 100) / total_vets
   percentage_yummy_kibble = 30) :=
by
  sorry

end vets_recommend_yummy_dog_kibble_l303_303908


namespace last_four_digits_of_5_pow_2011_l303_303530

theorem last_four_digits_of_5_pow_2011 : (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_of_5_pow_2011_l303_303530


namespace pencils_in_total_l303_303359

theorem pencils_in_total
  (rows : ℕ) (pencils_per_row : ℕ) (total_pencils : ℕ)
  (h1 : rows = 14)
  (h2 : pencils_per_row = 11)
  (h3 : total_pencils = rows * pencils_per_row) :
  total_pencils = 154 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end pencils_in_total_l303_303359


namespace solution_set_l303_303699

noncomputable def domain := Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x, x ∈ domain → x ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2)
axiom f_odd : ∀ x, f x + f (-x) = 0
def f' : ℝ → ℝ := sorry
axiom derivative_condition : ∀ x, 0 < x ∧ x < Real.pi / 2 → f' x * Real.cos x + f x * Real.sin x < 0

theorem solution_set :
  {x | f x < Real.sqrt 2 * f (Real.pi / 4) * Real.cos x} = {x | Real.pi / 4 < x ∧ x < Real.pi / 2} :=
sorry

end solution_set_l303_303699


namespace negation_equiv_l303_303221

open Classical

-- Proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 1 = 0

-- Negation of proposition p
def neg_p : Prop := ∀ x : ℝ, x^2 - x + 1 ≠ 0

-- Statement to prove the equivalence of the negation of p and neg_p
theorem negation_equiv :
  ¬p ↔ neg_p := 
sorry

end negation_equiv_l303_303221


namespace fraction_addition_l303_303298

/--
The value of 2/5 + 1/3 is 11/15.
-/
theorem fraction_addition :
  (2 / 5 : ℚ) + (1 / 3) = 11 / 15 := 
sorry

end fraction_addition_l303_303298


namespace trip_time_difference_l303_303027

def travel_time (distance speed : ℕ) : ℕ :=
  distance / speed

theorem trip_time_difference
  (speed : ℕ)
  (speed_pos : 0 < speed)
  (distance1 : ℕ)
  (distance2 : ℕ)
  (time_difference : ℕ)
  (h1 : distance1 = 540)
  (h2 : distance2 = 600)
  (h_speed : speed = 60)
  (h_time_diff : time_difference = (travel_time distance2 speed) - (travel_time distance1 speed) * 60)
  : time_difference = 60 :=
by
  sorry

end trip_time_difference_l303_303027


namespace min_buses_needed_l303_303343

theorem min_buses_needed (total_students : ℕ) (bus45_capacity : ℕ) (bus40_capacity : ℕ) : 
  total_students = 530 ∧ bus45_capacity = 45 ∧ bus40_capacity = 40 → 
  ∃ (n : ℕ), n = 12 :=
by 
  intro h
  obtain ⟨htotal, hbus45, hbus40⟩ := h
  -- Proof would go here...
  sorry

end min_buses_needed_l303_303343


namespace students_failed_l303_303163

theorem students_failed (Q : ℕ) (x : ℕ) (h1 : 4 * Q < 56) (h2 : x = Nat.lcm 3 (Nat.lcm 7 2)) (h3 : x < 56) :
  let R := x - (x / 3 + x / 7 + x / 2) 
  R = 1 := 
by
  sorry

end students_failed_l303_303163


namespace y_intercept_of_line_l303_303883

/-- Let m be the slope of a line and (x_intercept, 0) be the x-intercept of the same line.
    If the line passes through the point (3, 0) and has a slope of -3, then its y-intercept is (0, 9). -/
theorem y_intercept_of_line 
    (m : ℝ) (x_intercept : ℝ) (x1 y1 : ℝ)
    (h1 : m = -3)
    (h2 : (x_intercept, 0) = (3, 0)) :
    (0, -m * x_intercept) = (0, 9) :=
by sorry

end y_intercept_of_line_l303_303883


namespace variance_of_scores_l303_303918

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end variance_of_scores_l303_303918


namespace geometric_sequence_analogy_l303_303823

variables {a_n b_n : ℕ → ℕ} {S T : ℕ → ℕ}

-- Conditions for the arithmetic sequence
def is_arithmetic_sequence_sum (S : ℕ → ℕ) :=
  S 8 - S 4 = 2 * (S 4) ∧ S 12 - S 8 = 2 * (S 8 - S 4)

-- Conditions for the geometric sequence
def is_geometric_sequence_product (T : ℕ → ℕ) :=
  (T 8 / T 4) = (T 4) ∧ (T 12 / T 8) = (T 8 / T 4)

-- Statement of the proof problem
theorem geometric_sequence_analogy
  (h_arithmetic : is_arithmetic_sequence_sum S)
  (h_geometric_nil : is_geometric_sequence_product T) :
  T 4 / T 4 = 1 ∧
  (T 8 / T 4) / (T 8 / T 4) = 1 ∧
  (T 12 / T 8) / (T 12 / T 8) = 1 := 
by
  sorry

end geometric_sequence_analogy_l303_303823


namespace sum_of_squares_of_consecutive_even_integers_l303_303292

theorem sum_of_squares_of_consecutive_even_integers (n : ℤ) (h : (2 * n - 2) * (2 * n) * (2 * n + 2) = 12 * ((2 * n - 2) + (2 * n) + (2 * n + 2))) :
  (2 * n - 2) ^ 2 + (2 * n) ^ 2 + (2 * n + 2) ^ 2 = 440 :=
by
  sorry

end sum_of_squares_of_consecutive_even_integers_l303_303292


namespace find_x_l303_303723

theorem find_x {x : ℝ} :
  (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 → x = 6 :=
by
  intro h
  -- Solution steps would go here, but they are omitted.
  sorry

end find_x_l303_303723


namespace cos_beta_acos_l303_303655

theorem cos_beta_acos {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_cos_α : Real.cos α = 1 / 7) (h_cos_sum : Real.cos (α + β) = -11 / 14) :
  Real.cos β = 1 / 2 := by
  sorry

end cos_beta_acos_l303_303655


namespace certain_number_is_60_l303_303302

theorem certain_number_is_60 
  (A J C : ℕ) 
  (h1 : A = 4) 
  (h2 : C = 8) 
  (h3 : A = (1 / 2) * J) :
  3 * (A + J + C) = 60 :=
by sorry

end certain_number_is_60_l303_303302


namespace layla_earnings_l303_303561

-- Define the hourly rates for each family
def rate_donaldson : ℕ := 15
def rate_merck : ℕ := 18
def rate_hille : ℕ := 20
def rate_johnson : ℕ := 22
def rate_ramos : ℕ := 25

-- Define the hours Layla worked for each family
def hours_donaldson : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def hours_johnson : ℕ := 4
def hours_ramos : ℕ := 2

-- Calculate the earnings for each family
def earnings_donaldson : ℕ := rate_donaldson * hours_donaldson
def earnings_merck : ℕ := rate_merck * hours_merck
def earnings_hille : ℕ := rate_hille * hours_hille
def earnings_johnson : ℕ := rate_johnson * hours_johnson
def earnings_ramos : ℕ := rate_ramos * hours_ramos

-- Calculate total earnings
def total_earnings : ℕ :=
  earnings_donaldson + earnings_merck + earnings_hille + earnings_johnson + earnings_ramos

-- The assertion that Layla's total earnings are $411
theorem layla_earnings : total_earnings = 411 := by
  sorry

end layla_earnings_l303_303561


namespace larger_number_l303_303727

/-- The difference of two numbers is 1375 and the larger divided by the smaller gives a quotient of 6 and a remainder of 15. 
Prove that the larger number is 1647. -/
theorem larger_number (L S : ℕ) 
  (h1 : L - S = 1375) 
  (h2 : L = 6 * S + 15) : 
  L = 1647 := 
sorry

end larger_number_l303_303727


namespace james_eats_slices_l303_303247

theorem james_eats_slices :
  let initial_slices := 8 in
  let friend_eats := 2 in
  let remaining_slices := initial_slices - friend_eats in
  let james_eats := remaining_slices / 2 in
  james_eats = 3 := 
by 
  sorry

end james_eats_slices_l303_303247


namespace sum_q_t_8_eq_128_l303_303696

-- Define the type of 8-tuples where each entry is either 0 or 1
def T := {t: Fin 8 → ℕ // ∀ i, t i = 0 ∨ t i = 1}

-- Define q_t as the polynomial of degree at most 7
noncomputable def q_t (t: T) (x: ℕ) :=
  Polynomial.sum (fun i : Fin 8 => if t.val i = 1 then Polynomial.monomial i 1 else 0)

-- Define the polynomial q(x)
noncomputable def q (x: ℕ) :=
  @Finset.sum (T) (q_t · x) _ Finset.univ

-- The theorem we aim to prove
theorem sum_q_t_8_eq_128:
  q 8 = 128 :=
sorry

end sum_q_t_8_eq_128_l303_303696


namespace trig_expression_zero_l303_303223

theorem trig_expression_zero (α : ℝ) (h : Real.tan α = 2) : 
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := 
by
  sorry

end trig_expression_zero_l303_303223


namespace part_a_part_b_l303_303932

-- Define the system of equations
def system_of_equations (x y z p : ℝ) :=
  x^2 - 3 * y + p = z ∧ y^2 - 3 * z + p = x ∧ z^2 - 3 * x + p = y

-- Part (a) proof problem statement
theorem part_a (p : ℝ) (hp : p ≥ 4) :
  (p > 4 → ¬ ∃ (x y z : ℝ), system_of_equations x y z p) ∧
  (p = 4 → ∀ (x y z : ℝ), system_of_equations x y z 4 → x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

-- Part (b) proof problem statement
theorem part_b (p : ℝ) (hp : 1 < p ∧ p < 4) :
  ∀ (x y z : ℝ), system_of_equations x y z p → x = y ∧ y = z :=
by sorry

end part_a_part_b_l303_303932


namespace storage_temperature_overlap_l303_303374

theorem storage_temperature_overlap (T_A_min T_A_max T_B_min T_B_max : ℝ) 
  (hA : T_A_min = 0)
  (hA' : T_A_max = 5)
  (hB : T_B_min = 2)
  (hB' : T_B_max = 7) : 
  (max T_A_min T_B_min, min T_A_max T_B_max) = (2, 5) := by 
{
  sorry -- The proof is omitted as per instructions.
}

end storage_temperature_overlap_l303_303374


namespace domain_of_f_l303_303356

noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / Real.sqrt (3*x - 9)

theorem domain_of_f : ∀ x : ℝ, (3 < x) ↔ (∃ y : ℝ, f y ≠ y) :=
by
  sorry

end domain_of_f_l303_303356


namespace three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l303_303424

theorem three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one (n : ℕ) (h : n > 1) : ¬(2^n - 1) ∣ (3^n - 1) :=
sorry

end three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l303_303424


namespace midpoint_of_segment_l303_303743

theorem midpoint_of_segment (A B : (ℤ × ℤ)) (hA : A = (12, 3)) (hB : B = (-8, -5)) :
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1 :=
by
  sorry

end midpoint_of_segment_l303_303743


namespace sum_of_numbers_l303_303927

theorem sum_of_numbers :
  1357 + 7531 + 3175 + 5713 = 17776 :=
by
  sorry

end sum_of_numbers_l303_303927


namespace find_divisor_l303_303126

/-- Given a dividend of 15698, a quotient of 89, and a remainder of 14, find the divisor. -/
theorem find_divisor :
  ∃ D : ℕ, 15698 = 89 * D + 14 ∧ D = 176 :=
by
  sorry

end find_divisor_l303_303126


namespace positive_difference_solutions_of_abs_eq_l303_303745

theorem positive_difference_solutions_of_abs_eq (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 15) (h2 : 2 * x2 - 3 = -15) : |x1 - x2| = 15 := by
  sorry

end positive_difference_solutions_of_abs_eq_l303_303745


namespace correct_options_l303_303899

-- Definitions of conditions in Lean 
def is_isosceles (T : Triangle) : Prop := sorry -- Define isosceles triangle
def is_right_angle (T : Triangle) : Prop := sorry -- Define right-angled triangle
def similar (T₁ T₂ : Triangle) : Prop := sorry -- Define similarity of triangles
def equal_vertex_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal vertex angle
def equal_base_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal base angle

-- Theorem statement to verify correct options (2) and (4)
theorem correct_options {T₁ T₂ : Triangle} :
  (is_right_angle T₁ ∧ is_right_angle T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) ∧ 
  (equal_vertex_angle T₁ T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) :=
sorry -- proof not required

end correct_options_l303_303899


namespace find_m_l303_303087

open Real

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (sin x ^ 4 + cos x ^ 4) + m * (sin x + cos x) ^ 4

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x m ≤ 5) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x m = 5) :=
sorry

end find_m_l303_303087


namespace product_not_divisible_by_prime_l303_303135

theorem product_not_divisible_by_prime (p a b : ℕ) (hp : Prime p) (ha : 1 ≤ a) (hpa : a < p) (hb : 1 ≤ b) (hpb : b < p) : ¬ (p ∣ (a * b)) :=
by
  sorry

end product_not_divisible_by_prime_l303_303135


namespace fraction_increase_by_three_l303_303842

variables (a b : ℝ)

theorem fraction_increase_by_three : 
  3 * (2 * a * b / (3 * a - 4 * b)) = 2 * (3 * a * 3 * b) / (3 * (3 * a) - 4 * (3 * b)) :=
by
  sorry

end fraction_increase_by_three_l303_303842


namespace axis_of_parabola_l303_303208

-- Define the given equation of the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8 * y

-- Define the standard form of a vertical parabola and the value we need to prove (axis of the parabola)
def standard_form (p y : ℝ) : Prop := y = 2

-- The proof problem: Given the equation of the parabola, prove the equation of its axis.
theorem axis_of_parabola : 
  ∀ x y : ℝ, (parabola x y) → (standard_form y 2) :=
by
  intros x y h
  sorry

end axis_of_parabola_l303_303208


namespace solve_for_x_l303_303540

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l303_303540


namespace find_real_roots_l303_303802

theorem find_real_roots : 
  {x : ℝ | x^9 + (9 / 8) * x^6 + (27 / 64) * x^3 - x + (219 / 512) = 0} =
  {1 / 2, (-1 + Real.sqrt 13) / 4, (-1 - Real.sqrt 13) / 4} :=
by
  sorry

end find_real_roots_l303_303802


namespace problem_statement_l303_303081

variable (x1 x2 x3 x4 x5 x6 x7 : ℝ)

theorem problem_statement
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 20)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 145) :
  16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 380 :=
sorry

end problem_statement_l303_303081


namespace perpendicular_vectors_l303_303662

def vector (α : Type) := (α × α)
def dot_product {α : Type} [Add α] [Mul α] (a b : vector α) : α :=
  a.1 * b.1 + a.2 * b.2

theorem perpendicular_vectors
    (a : vector ℝ) (b : vector ℝ)
    (h : dot_product a b = 0)
    (ha : a = (2, 4))
    (hb : b = (-1, n)) : 
    n = 1 / 2 := 
  sorry

end perpendicular_vectors_l303_303662


namespace find_a_l303_303159

theorem find_a (a : ℝ) :
  (∀ x : ℝ, deriv (fun x => a * x^3 - 2) x * x = 1) → a = 1 / 3 :=
by
  intro h
  have slope_at_minus_1 := h (-1)
  sorry -- here we stop as proof isn't needed

end find_a_l303_303159


namespace polynomial_has_one_positive_real_solution_l303_303958

-- Define the polynomial
def f (x : ℝ) : ℝ := x ^ 10 + 4 * x ^ 9 + 7 * x ^ 8 + 2023 * x ^ 7 - 2024 * x ^ 6

-- The proof problem statement
theorem polynomial_has_one_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ f x = 0 := by
  sorry

end polynomial_has_one_positive_real_solution_l303_303958


namespace product_of_primes_95_l303_303888

theorem product_of_primes_95 (p q : Nat) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p + q = 95) : p * q = 178 := sorry

end product_of_primes_95_l303_303888


namespace red_ball_probability_l303_303400

theorem red_ball_probability :
  let total_balls := 7
  let red_balls_initial := 4
  let white_balls_initial := 3
  let first_ball_red := True
  let total_balls_after_first := total_balls - 1
  let red_balls_after_first := red_balls_initial - 1
  let probability_first_red := red_balls_initial / total_balls
  let probability_second_red_given_first_red := red_balls_after_first / total_balls_after_first
  probability_second_red_given_first_red = 1/2 :=
by
  sorry

end red_ball_probability_l303_303400


namespace average_earning_week_l303_303563

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ)
  (h1 : (D1 + D2 + D3 + D4) / 4 = 25)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 20) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 24 :=
by
  sorry

end average_earning_week_l303_303563


namespace Lily_points_l303_303001

variable (x y z : ℕ) -- points for inner ring (x), middle ring (y), and outer ring (z)

-- Tom's score
axiom Tom_score : 3 * x + y + 2 * z = 46

-- John's score
axiom John_score : x + 3 * y + 2 * z = 34

-- Lily's score
def Lily_score : ℕ := 40

theorem Lily_points : ∀ (x y z : ℕ), 3 * x + y + 2 * z = 46 → x + 3 * y + 2 * z = 34 → Lily_score = 40 := by
  intros x y z Tom_score John_score
  sorry

end Lily_points_l303_303001


namespace mod_inverse_11_mod_1105_l303_303485

theorem mod_inverse_11_mod_1105 : (11 * 201) % 1105 = 1 :=
  by 
    sorry

end mod_inverse_11_mod_1105_l303_303485


namespace describe_graph_l303_303173

theorem describe_graph :
  ∀ (x y : ℝ), ((x + y) ^ 2 = x ^ 2 + y ^ 2 + 4 * x) ↔ (x = 0 ∨ y = 2) := 
by
  sorry

end describe_graph_l303_303173


namespace proof_problem_l303_303370

def p : Prop := ∃ k : ℕ, 0 = 2 * k
def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem proof_problem : p ∨ q :=
by
  sorry

end proof_problem_l303_303370


namespace count_scalene_triangles_under_16_l303_303567

theorem count_scalene_triangles_under_16 : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (a b c : ℕ), 
  a < b ∧ b < c ∧ a + b + c < 16 ∧ a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a, b, c) ∈ [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 5), (3, 5, 6), (4, 5, 6)] :=
by sorry

end count_scalene_triangles_under_16_l303_303567


namespace complex_problem_l303_303225

def is_imaginary_unit (x : ℂ) : Prop := x^2 = -1

theorem complex_problem (a b : ℝ) (i : ℂ) (h1 : (a - 2 * i) / i = (b : ℂ) + i) (h2 : is_imaginary_unit i) :
  a - b = 1 := 
sorry

end complex_problem_l303_303225


namespace last_two_digits_sum_factorials_l303_303312

theorem last_two_digits_sum_factorials : 
  (∑ i in Finset.range 50, Nat.factorial i) % 100 = 13 := sorry

end last_two_digits_sum_factorials_l303_303312


namespace proof_problem_l303_303644

variable {α : Type*} [LinearOrderedField α] 
variable (x1 x2 x3 x4 x5 x6 : α) 
variable (h1 : x1 = min x1 x2 ⊓ x1 x3 ⊓ x1 x4 ⊓ x1 x5 ⊓ x1 x6)
variable (h6 : x6 = max x1 x2 ⊔ x1 x3 ⊔ x1 x4 ⊔ x1 x5 ⊔ x1 x6)

-- Definitions of medians and ranges
def median (s : Finset α) : α := 
  let n := s.card
  if n % 2 = 1 then s.sort (≤).nth (n / 2) 
  else (s.sort (≤).nth (n / 2 - 1) + s.sort (≤).nth (n / 2)) / 2

def range (s : Finset α) : α := s.max' (Finset.nonempty_sort _) - s.min' (Finset.nonempty_sort _)

theorem proof_problem :
  median {x2, x3, x4, x5} = median {x1, x2, x3, x4, x5, x6} ∧
  range {x2, x3, x4, x5} ≤ range {x1, x2, x3, x4, x5, x6} :=
by
  sorry

end proof_problem_l303_303644


namespace cos_square_minus_sin_square_15_l303_303049

theorem cos_square_minus_sin_square_15 (cos_30 : Real.cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2) : 
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = (Real.sqrt 3) / 2 := 
by 
  sorry

end cos_square_minus_sin_square_15_l303_303049


namespace fraction_increase_by_3_l303_303236

theorem fraction_increase_by_3 (x y : ℝ) (h₁ : x' = 3 * x) (h₂ : y' = 3 * y) : 
  (x' * y') / (x' - y') = 3 * (x * y) / (x - y) :=
by
  sorry

end fraction_increase_by_3_l303_303236


namespace barbell_percentage_increase_l303_303691

def old_barbell_cost : ℕ := 250
def new_barbell_cost : ℕ := 325

theorem barbell_percentage_increase :
  (new_barbell_cost - old_barbell_cost : ℚ) / old_barbell_cost * 100 = 30 := 
by
  sorry

end barbell_percentage_increase_l303_303691


namespace factor_expression_l303_303214

theorem factor_expression (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2) ^ 2 :=
by sorry

end factor_expression_l303_303214


namespace sum_f_1_to_2017_l303_303334

noncomputable def f (x : ℝ) : ℝ :=
  if x % 6 < -1 then -(x % 6 + 2) ^ 2 else x % 6

theorem sum_f_1_to_2017 : (List.sum (List.map f (List.range' 1 2017))) = 337 :=
  sorry

end sum_f_1_to_2017_l303_303334


namespace probability_of_sum_leq_10_l303_303165

open Nat

-- Define the three dice roll outcomes
def dice_outcomes := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := 6 ^ 3

-- Count the number of valid outcomes where the sum of three dice is less than or equal to 10
def count_valid_outcomes : ℕ := 75  -- This is determined through combinatorial calculations or software

-- Define the desired probability
def desired_probability := (count_valid_outcomes : ℚ) / total_outcomes

-- Prove that the desired probability equals 25/72
theorem probability_of_sum_leq_10 :
  desired_probability = 25 / 72 :=
by sorry

end probability_of_sum_leq_10_l303_303165


namespace find_original_expenditure_l303_303576

def original_expenditure (x : ℝ) := 35 * x
def new_expenditure (x : ℝ) := 42 * (x - 1)

theorem find_original_expenditure :
  ∃ x, 35 * x + 42 = 42 * (x - 1) ∧ original_expenditure x = 420 :=
by
  sorry

end find_original_expenditure_l303_303576


namespace two_digit_product_GCD_l303_303293

-- We define the condition for two-digit integer numbers
def two_digit_num (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Lean statement capturing the conditions
theorem two_digit_product_GCD :
  ∃ (a b : ℕ), two_digit_num a ∧ two_digit_num b ∧ a * b = 1728 ∧ Nat.gcd a b = 12 := 
by {
  sorry -- The proof steps would go here
}

end two_digit_product_GCD_l303_303293


namespace coby_travel_time_l303_303794

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l303_303794


namespace Adam_total_shopping_cost_l303_303921

theorem Adam_total_shopping_cost :
  let sandwiches := 3
  let sandwich_cost := 3
  let water_cost := 2
  (sandwiches * sandwich_cost + water_cost) = 11 := 
by
  sorry

end Adam_total_shopping_cost_l303_303921


namespace max_value_of_f_on_interval_l303_303226

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 2

theorem max_value_of_f_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 1 :=
by
  sorry

end max_value_of_f_on_interval_l303_303226


namespace parabola_min_value_l303_303378

theorem parabola_min_value (x : ℝ) : (∃ x, x^2 + 10 * x + 21 = -4) := sorry

end parabola_min_value_l303_303378


namespace average_sqft_per_person_texas_l303_303843

theorem average_sqft_per_person_texas :
  let population := 17000000
  let area_sqmiles := 268596
  let usable_land_percentage := 0.8
  let sqfeet_per_sqmile := 5280 * 5280
  let total_sqfeet := area_sqmiles * sqfeet_per_sqmile
  let usable_sqfeet := usable_land_percentage * total_sqfeet
  let avg_sqfeet_per_person := usable_sqfeet / population
  352331 <= avg_sqfeet_per_person ∧ avg_sqfeet_per_person < 500000 :=
by
  sorry

end average_sqft_per_person_texas_l303_303843


namespace find_ellipse_equation_l303_303654

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ c : ℝ, a > b ∧ b > 0 ∧ 4 * a = 16 ∧ |c| = 2 ∧ a^2 = b^2 + c^2

theorem find_ellipse_equation :
  (∃ (a b : ℝ), ellipse_equation a b) → (∃ b : ℝ, (a = 4) ∧ (b > 0) ∧ (b^2 = 12) ∧ (∀ x y : ℝ, (x^2 / 16) + (y^2 / 12) = 1)) :=
by {
  sorry
}

end find_ellipse_equation_l303_303654


namespace eggs_distribution_l303_303599

theorem eggs_distribution
  (total_eggs : ℕ)
  (eggs_per_adult : ℕ)
  (num_adults : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (eggs_per_girl : ℕ)
  (total_eggs_def : total_eggs = 3 * 12)
  (eggs_per_adult_def : eggs_per_adult = 3)
  (num_adults_def : num_adults = 3)
  (num_girls_def : num_girls = 7)
  (num_boys_def : num_boys = 10)
  (eggs_per_girl_def : eggs_per_girl = 1) :
  ∃ eggs_per_boy : ℕ, eggs_per_boy - eggs_per_girl = 1 :=
by {
  sorry
}

end eggs_distribution_l303_303599


namespace simplify_expression_l303_303715

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := 
by sorry

end simplify_expression_l303_303715


namespace mn_sum_l303_303093

theorem mn_sum {m n : ℤ} (h : ∀ x : ℤ, (x + 8) * (x - 1) = x^2 + m * x + n) : m + n = -1 :=
by
  sorry

end mn_sum_l303_303093


namespace dessert_eating_contest_l303_303736

theorem dessert_eating_contest (a b c : ℚ) 
  (h1 : a = 5/6) 
  (h2 : b = 7/8) 
  (h3 : c = 1/2) :
  b - a = 1/24 ∧ a - c = 1/3 := 
by 
  sorry

end dessert_eating_contest_l303_303736


namespace rabbit_weight_l303_303341

theorem rabbit_weight (a b c : ℕ) (h1 : a + b + c = 30) (h2 : a + c = 2 * b) (h3 : a + b = c) :
  a = 5 := by
  sorry

end rabbit_weight_l303_303341


namespace median_equality_range_inequality_l303_303639

open List

variables (x : List ℝ) (h₁ : length x = 6) (h₂ : ∀ y ∈ x, x[0] ≤ y) (h₃ : ∀ y ∈ x, y ≤ x[5])

def average (l : List ℝ) : ℝ := (l.foldl (fun x y => x + y) 0) / (l.length)

theorem median_equality :
  (average (x.drop 1 |>.pop) = average x) ∧ (nth (x.drop 2) 1 = nth x 2) ∧ (nth (x.drop 2) 2 = nth x 3) := 
sorry

theorem range_inequality :
  (nth x 5 - nth x 0 >= nth x 4 - nth x 1) :=
sorry

end median_equality_range_inequality_l303_303639


namespace smallest_four_digit_divisible_by_4_and_5_l303_303316

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l303_303316


namespace Nancy_needs_5_loads_l303_303124

/-- Definition of the given problem conditions. -/
def pieces_of_clothing (shirts sweaters socks jeans : ℕ) : ℕ :=
  shirts + sweaters + socks + jeans

def washing_machine_capacity : ℕ := 12

def loads_required (total_clothing capacity : ℕ) : ℕ :=
  (total_clothing + capacity - 1) / capacity -- integer division with rounding up

/-- Theorem statement. -/
theorem Nancy_needs_5_loads :
  loads_required (pieces_of_clothing 19 8 15 10) washing_machine_capacity = 5 :=
by
  -- Insert proof here when needed.
  sorry

end Nancy_needs_5_loads_l303_303124


namespace rowing_downstream_speed_l303_303182

-- Define the given conditions
def V_u : ℝ := 60  -- speed upstream in kmph
def V_s : ℝ := 75  -- speed in still water in kmph

-- Define the problem statement
theorem rowing_downstream_speed : ∃ (V_d : ℝ), V_s = (V_u + V_d) / 2 ∧ V_d = 90 :=
by
  sorry

end rowing_downstream_speed_l303_303182


namespace tim_drinks_amount_l303_303381

theorem tim_drinks_amount (H : ℚ := 2/7) (T : ℚ := 5/8) : 
  (T * H) = 5/28 :=
by sorry

end tim_drinks_amount_l303_303381


namespace hours_week3_and_4_l303_303051

variable (H3 H4 : Nat)

def hours_worked_week1_and_2 : Nat := 35 + 35
def extra_hours_worked_week3_and_4 : Nat := 26
def total_hours_week3_and_4 : Nat := hours_worked_week1_and_2 + extra_hours_worked_week3_and_4

theorem hours_week3_and_4 :
  H3 + H4 = total_hours_week3_and_4 := by
sorry

end hours_week3_and_4_l303_303051


namespace circle_equation_tangent_l303_303209

theorem circle_equation_tangent (h : ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = 25)) :
    ∃ c : ℝ × ℝ, c = (1, 2) ∧ ∃ r : ℝ, r = 5 ∧ ∀ x y : ℝ, (4 * x + 3 * y - 35 ≠ 0) → ((x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2) := 
by
    sorry

end circle_equation_tangent_l303_303209


namespace cos_double_angle_l303_303393

theorem cos_double_angle (α : ℝ) (h : Real.sin α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = 1 / 3 :=
by
  sorry

end cos_double_angle_l303_303393


namespace centroid_of_triangle_l303_303488

theorem centroid_of_triangle :
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  ( (x1 + x2 + x3) / 3 = 8 / 3 ∧ (y1 + y2 + y3) / 3 = -5 / 3 ) :=
by
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  have centroid_x : (x1 + x2 + x3) / 3 = 8 / 3 := sorry
  have centroid_y : (y1 + y2 + y3) / 3 = -5 / 3 := sorry
  exact ⟨centroid_x, centroid_y⟩

end centroid_of_triangle_l303_303488


namespace smallest_positive_q_with_property_l303_303053

theorem smallest_positive_q_with_property :
  ∃ q : ℕ, (
    q > 0 ∧
    ∀ m : ℕ, (1 ≤ m ∧ m ≤ 1006) →
    ∃ n : ℤ, 
      (m * q : ℤ) / 1007 < n ∧
      (m + 1) * q / 1008 > n) ∧
   q = 2015 := 
sorry

end smallest_positive_q_with_property_l303_303053


namespace area_of_triangle_l303_303907

-- Define the lines in terms of functions
def line1 (x : ℝ) : ℝ := 6
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the points of intersection based on solving the equations of lines
def point1 : ℝ × ℝ := (4, line1 4)
def point2 : ℝ × ℝ := (-4, line1 (-4))
def point3 : ℝ × ℝ := (0, line2 0)

-- Translate problem to Lean proof statement
theorem area_of_triangle :
  let v1 := (4, 6)
  let v2 := (-4, 6)
  let v3 := (0, 2)
  let area := 1 / 2 * abs ((fst v1 * snd v2 + fst v2 * snd v3 + fst v3 * snd v1) -
                           (snd v1 * fst v2 + snd v2 * fst v3 + snd v3 * fst v1))
  area = 16 :=
sorry

end area_of_triangle_l303_303907


namespace length_of_diagonal_l303_303362

theorem length_of_diagonal (area : ℝ) (h1 h2 : ℝ) (d : ℝ) 
  (h_area : area = 75)
  (h_offsets : h1 = 6 ∧ h2 = 4) :
  d = 15 :=
by
  -- Given the conditions and formula, we can conclude
  sorry

end length_of_diagonal_l303_303362


namespace remainder_of_number_mod_1000_l303_303695

-- Definitions according to the conditions
def num_increasing_8_digit_numbers_with_zero : ℕ := Nat.choose 17 8

-- The main statement to be proved
theorem remainder_of_number_mod_1000 : 
  (num_increasing_8_digit_numbers_with_zero % 1000) = 310 :=
by
  sorry

end remainder_of_number_mod_1000_l303_303695


namespace find_angle_B_l303_303108

noncomputable def angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) : ℝ :=
  B

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
sorry

end find_angle_B_l303_303108


namespace total_shopping_cost_l303_303923

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end total_shopping_cost_l303_303923


namespace probability_sum_is_prime_l303_303740

open_locale classical

-- Definitions for the sectors of Spinner 1 and Spinner 2
def spinner1_sectors := {2, 3, 4}
def spinner2_sectors := {1, 3, 5}

-- Function to determine if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The set of prime sums from the possible outcomes of the spinners
def prime_sums (s1 s2 : set ℕ) : set ℕ :=
  { x | x ∈ {a + b | a ∈ s1, b ∈ s2} ∧ is_prime x }

-- The probability of landing on a prime sum
def probability_of_prime_sum : ℚ :=
  ∑ x in prime_sums spinner1_sectors spinner2_sectors, 1 / (spinner1_sectors.card * spinner2_sectors.card)

theorem probability_sum_is_prime : probability_of_prime_sum = 5 / 9 :=
by sorry

end probability_sum_is_prime_l303_303740


namespace incorrect_statement_l303_303753

theorem incorrect_statement :
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  (statementA = "correct") ∧ 
  (statementB = "correct") ∧ 
  (statementC = "correct") ∧ 
  (statementD = "incorrect") :=
by
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  have hA : statementA = "correct" := sorry
  have hB : statementB = "correct" := sorry
  have hC : statementC = "correct" := sorry
  have hD : statementD = "incorrect" := sorry
  exact ⟨hA, hB, hC, hD⟩

end incorrect_statement_l303_303753


namespace area_of_smaller_circle_l303_303306

theorem area_of_smaller_circle
  (PA AB : ℝ)
  (r s : ℝ)
  (tangent_at_T : true) -- placeholder; represents the tangency condition
  (common_tangents : true) -- placeholder; represents the external tangents condition
  (PA_eq_AB : PA = AB) :
  PA = 5 →
  AB = 5 →
  r = 2 * s →
  ∃ (s : ℝ) (area : ℝ), s = 5 / (2 * (Real.sqrt 2)) ∧ area = (Real.pi * s^2) ∧ area = (25 * Real.pi) / 8 := by
  intros hPA hAB h_r_s
  use 5 / (2 * (Real.sqrt 2))
  use (Real.pi * (5 / (2 * (Real.sqrt 2)))^2)
  simp [←hPA,←hAB]
  sorry

end area_of_smaller_circle_l303_303306


namespace minimize_distance_postman_l303_303603

-- Let x be a function that maps house indices to coordinates.
def optimalPostOfficeLocation (n: ℕ) (x : ℕ → ℝ) : ℝ :=
  if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2)

theorem minimize_distance_postman (n: ℕ) (x : ℕ → ℝ)
  (h_sorted : ∀ i j, i < j → x i < x j) :
  optimalPostOfficeLocation n x = if n % 2 = 1 then 
    x (n / 2 + 1)
  else 
    x (n / 2) := 
  sorry

end minimize_distance_postman_l303_303603


namespace fib_150_mod_9_l303_303722

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib n + fib (n + 1)

-- State the main theorem
theorem fib_150_mod_9 : (fib 150) % 9 = 8 := 
sorry

end fib_150_mod_9_l303_303722


namespace exists_function_f_l303_303891

-- Define the problem statement
theorem exists_function_f :
  ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (abs (x + 1)) = x^2 + 2 * x :=
sorry

end exists_function_f_l303_303891


namespace find_a4_l303_303824

variable {a_n : ℕ → ℝ}
variable (S_n : ℕ → ℝ)

noncomputable def Sn := 1/2 * 5 * (a_n 1 + a_n 5)

axiom h1 : S_n 5 = 25
axiom h2 : a_n 2 = 3

theorem find_a4 : a_n 4 = 5 := sorry

end find_a4_l303_303824


namespace wallpaper_job_completion_l303_303741

theorem wallpaper_job_completion (x : ℝ) (y : ℝ) 
  (h1 : ∀ a b : ℝ, (a = 1.5) → (7/x + (7-a)/(x-3) = 1)) 
  (h2 : y = x - 3) 
  (h3 : x - y = 3) : 
  (x = 14) ∧ (y = 11) :=
sorry

end wallpaper_job_completion_l303_303741


namespace visit_orders_l303_303175

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_permutations_cities (pohang busan geoncheon gimhae gyeongju : Type) : ℕ :=
  factorial 4

theorem visit_orders (pohang busan geoncheon gimhae gyeongju : Type) :
  num_permutations_cities pohang busan geoncheon gimhae gyeongju = 24 :=
by
  unfold num_permutations_cities
  norm_num
  sorry

end visit_orders_l303_303175


namespace three_distinct_divisors_l303_303448

theorem three_distinct_divisors (M : ℕ) : (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ M ∧ b ∣ M ∧ c ∣ M ∧ (∀ d, d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬ d ∣ M)) ↔ (∃ p : ℕ, Prime p ∧ M = p^2) := 
by sorry

end three_distinct_divisors_l303_303448


namespace simplify_and_rationalize_l303_303271

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l303_303271


namespace playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l303_303369

def hasWinningStrategyA (n : ℕ) : Prop :=
  n ≥ 8

def hasWinningStrategyB (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

def draw (n : ℕ) : Prop :=
  n = 6 ∨ n = 7

theorem playerA_winning_strategy (n : ℕ) : n ≥ 8 → hasWinningStrategyA n :=
by
  sorry

theorem playerB_winning_strategy (n : ℕ) : (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) → hasWinningStrategyB n :=
by
  sorry

theorem no_winning_strategy (n : ℕ) : n = 6 ∨ n = 7 → draw n :=
by
  sorry

end playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l303_303369


namespace number_of_teachers_l303_303342

theorem number_of_teachers (total_population sample_size teachers_within_sample students_within_sample : ℕ) 
    (h_total_population : total_population = 3000) 
    (h_sample_size : sample_size = 150) 
    (h_students_within_sample : students_within_sample = 140) 
    (h_teachers_within_sample : teachers_within_sample = sample_size - students_within_sample) 
    (h_ratio : (total_population - students_within_sample) * sample_size = total_population * teachers_within_sample) : 
    total_population - students_within_sample = 200 :=
by {
  sorry
}

end number_of_teachers_l303_303342


namespace missing_angles_sum_l303_303402

theorem missing_angles_sum 
  (calculated_sum : ℕ) 
  (missed_angles_sum : ℕ)
  (total_corrections : ℕ)
  (polygon_angles : ℕ) 
  (h1 : calculated_sum = 2797) 
  (h2 : total_corrections = 2880) 
  (h3 : polygon_angles = total_corrections - calculated_sum) : 
  polygon_angles = 83 := by
  sorry

end missing_angles_sum_l303_303402


namespace simplify_expression_l303_303714

theorem simplify_expression :
  (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := 
by {
  sorry
}

end simplify_expression_l303_303714


namespace chives_planted_l303_303114

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end chives_planted_l303_303114


namespace class_average_gpa_l303_303902

theorem class_average_gpa (n : ℕ) (hn : 0 < n) :
  ((1/3 * n) * 45 + (2/3 * n) * 60) / n = 55 :=
by
  sorry

end class_average_gpa_l303_303902


namespace find_a_perpendicular_line_l303_303238

theorem find_a_perpendicular_line (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 3 * y + 1 = 0) → (2 * x + 2 * y - 3 = 0) → (-(a / 3) * (-1) = -1)) → 
  a = -3 :=
by
  sorry

end find_a_perpendicular_line_l303_303238


namespace min_log_expression_is_zero_l303_303364

open Real

noncomputable def min_log_expression (a b c : ℝ) (h_cond : a ≥ b ∧ b ≥ c ∧ c > 1) : ℝ :=
  (log a / log (a^3 / b)) + (log b / log (b^3 / c))

theorem min_log_expression_is_zero (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 1) :
  min_log_expression a b c h = 0 :=
sorry

end min_log_expression_is_zero_l303_303364


namespace min_value_expr_ge_52_l303_303629

open Real

theorem min_value_expr_ge_52 (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (sin x + 3 * (1 / sin x)) ^ 2 + (cos x + 3 * (1 / cos x)) ^ 2 ≥ 52 := 
by
  sorry

end min_value_expr_ge_52_l303_303629


namespace triangles_with_two_colors_l303_303020

theorem triangles_with_two_colors {n : ℕ} 
  (h1 : ∀ (p : Finset ℝ) (hn : p.card = n) 
      (e : p → p → Prop), 
      (∀ (x y : p), e x y → e x y = red ∨ e x y = yellow ∨ e x y = green) /\
      (∀ (a b c : p), 
        (e a b = red ∨ e a b = yellow ∨ e a b = green) ∧ 
        (e b c = red ∨ e b c = yellow ∨ e b c = green) ∧ 
        (e a c = red ∨ e a c = yellow ∨ e a c = green) → 
        (e a b ≠ e b c ∨ e b c ≠ e a c ∨ e a b ≠ e a c))) :
  n < 13 := 
sorry

end triangles_with_two_colors_l303_303020


namespace lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l303_303853

def lamps_on_again (n : ℕ) (steps : ℕ → Bool → Bool) : ∃ M : ℕ, ∀ s, (s ≥ M) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

theorem lamps_on_after_n2_minus_n_plus_1 (n : ℕ) (k : ℕ) (hk : n = 2^k + 1) (steps : ℕ → Bool → Bool) : 
∀ s, (s ≥ n^2 - n + 1) → (n > 1 → ∀ i : ℕ, steps i true = true) := 
sorry

end lamps_on_after_n2_minus_1_lamps_on_after_n2_minus_n_plus_1_l303_303853


namespace tan_15_degree_l303_303906

theorem tan_15_degree : 
  let a := 45 * (Real.pi / 180)
  let b := 30 * (Real.pi / 180)
  Real.tan (a - b) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_15_degree_l303_303906


namespace median_equal_range_not_greater_l303_303636

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l303_303636


namespace largest_power_of_two_divides_n_l303_303800

noncomputable def largestPowerOfTwo (n : ℤ) : ℤ :=
  let v2 := PadicValuation 2 (padicVal 2 n)
  v2

theorem largest_power_of_two_divides_n (a b : ℕ) (ha : a = 15) (hb : b = 13) :
  largestPowerOfTwo (a^4 - b^4) = 16 := by
  -- Proof would go here
  sorry

end largest_power_of_two_divides_n_l303_303800


namespace real_solutions_l303_303361

theorem real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6)) = 1 / 12) ↔ (x = 12 ∨ x = -4) :=
by
  sorry

end real_solutions_l303_303361


namespace no_super_plus_good_exists_at_most_one_super_plus_good_l303_303930

def is_super_plus_good (board : ℕ → ℕ → ℕ) (n : ℕ) (i j : ℕ) : Prop :=
  (∀ k, k < n → board i k ≤ board i j) ∧ 
  (∀ k, k < n → board k j ≥ board i j)

def arrangement (n : ℕ) := { board : ℕ → ℕ → ℕ // ∀ i j, i < n → j < n → 1 ≤ board i j ∧ board i j ≤ n * n }

-- Prove that in some arrangements, there is no super-plus-good number.
theorem no_super_plus_good_exists (n : ℕ) (h₁ : n = 8) :
  ∃ (b : arrangement n), ∀ i j, ¬ is_super_plus_good b.val n i j := sorry

-- Prove that in every arrangement, there is at most one super-plus-good number.
theorem at_most_one_super_plus_good (n : ℕ) (h : n = 8) :
  ∀ (b : arrangement n), ∃! i j, is_super_plus_good b.val n i j := sorry

end no_super_plus_good_exists_at_most_one_super_plus_good_l303_303930


namespace louie_monthly_payment_l303_303590

noncomputable def compound_interest_payment (P : ℝ) (r : ℝ) (n : ℕ) (t_months : ℕ) : ℝ :=
  let t_years := t_months / 12
  let A := P * (1 + r / ↑n)^(↑n * t_years)
  A / t_months

theorem louie_monthly_payment : compound_interest_payment 1000 0.10 1 3 = 444 :=
by
  sorry

end louie_monthly_payment_l303_303590


namespace wooden_toys_count_l303_303990

theorem wooden_toys_count :
  ∃ T : ℤ, 
    10 * 40 + 20 * T - (10 * 36 + 17 * T) = 64 ∧ T = 8 :=
by
  use 8
  sorry

end wooden_toys_count_l303_303990


namespace solve_equation_l303_303552

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l303_303552


namespace donald_oranges_l303_303055

-- Define the initial number of oranges
def initial_oranges : ℕ := 4

-- Define the number of additional oranges found
def additional_oranges : ℕ := 5

-- Define the total number of oranges as the sum of initial and additional oranges
def total_oranges : ℕ := initial_oranges + additional_oranges

-- Theorem stating that the total number of oranges is 9
theorem donald_oranges : total_oranges = 9 := by
    -- Proof not provided, so we put sorry to indicate that this is a place for the proof.
    sorry

end donald_oranges_l303_303055


namespace part1_part2_l303_303380

noncomputable def set_A (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def set_B : Set ℝ := {x : ℝ | x < -1 ∨ x > 1}

theorem part1 (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a > 3) :=
by sorry

theorem part2 (a : ℝ) : (set_A a ∪ set_B = Set.univ) ↔ (-2 ≤ a ∧ a ≤ -1 / 2) :=
by sorry

end part1_part2_l303_303380


namespace simplify_expression_l303_303716

theorem simplify_expression (x y : ℤ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := 
by 
  sorry 

end simplify_expression_l303_303716


namespace range_of_a_l303_303228

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1) < a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l303_303228


namespace min_value_inequality_l303_303079

theorem min_value_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 :=
by
  sorry

end min_value_inequality_l303_303079


namespace number_of_lines_passing_through_point_and_forming_given_area_l303_303838

theorem number_of_lines_passing_through_point_and_forming_given_area :
  ∃ l : ℝ → ℝ, (∀ x y : ℝ, l 1 = 1) ∧ (∃ (a b : ℝ), abs ((1/2) * a * b) = 2)
  → (∃ n : ℕ, n = 4) :=
by
  sorry

end number_of_lines_passing_through_point_and_forming_given_area_l303_303838


namespace haji_mother_tough_weeks_l303_303505

/-- Let's define all the conditions: -/
def tough_week_revenue : ℕ := 800
def good_week_revenue : ℕ := 2 * tough_week_revenue
def number_of_good_weeks : ℕ := 5
def total_revenue : ℕ := 10400

/-- Let's define the proofs for intermediate steps: -/
def good_weeks_revenue : ℕ := number_of_good_weeks * good_week_revenue
def tough_weeks_revenue : ℕ := total_revenue - good_weeks_revenue
def number_of_tough_weeks : ℕ := tough_weeks_revenue / tough_week_revenue

/-- Now the theorem which states that the number of tough weeks is 3. -/
theorem haji_mother_tough_weeks : number_of_tough_weeks = 3 := by
  sorry

end haji_mother_tough_weeks_l303_303505


namespace total_travel_time_l303_303791

-- Define the necessary distances and speeds
def distance_Washington_to_Idaho : ℝ := 640
def speed_Washington_to_Idaho : ℝ := 80
def distance_Idaho_to_Nevada : ℝ := 550
def speed_Idaho_to_Nevada : ℝ := 50

-- Definitions for time calculations
def time_Washington_to_Idaho : ℝ := distance_Washington_to_Idaho / speed_Washington_to_Idaho
def time_Idaho_to_Nevada : ℝ := distance_Idaho_to_Nevada / speed_Idaho_to_Nevada

-- Problem statement to prove
theorem total_travel_time : time_Washington_to_Idaho + time_Idaho_to_Nevada = 19 := 
by
  sorry

end total_travel_time_l303_303791


namespace range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l303_303089

-- Define the propositions p and q
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := x^2 - 5 * x + 6 < 0

-- Question 1: When a = 1, if p ∧ q is true, determine the range of x
theorem range_of_x_when_a_is_1_and_p_and_q_are_true :
  ∀ x, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
by
  sorry

-- Question 2: If p is a necessary but not sufficient condition for q, determine the range of a
theorem range_of_a_when_p_necessary_for_q :
  ∀ a, (∀ x, q x → p x a) ∧ ¬ (∀ x, p x a → q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l303_303089


namespace f_of_7_l303_303765

theorem f_of_7 (f : ℝ → ℝ) (h : ∀ (x : ℝ), f (4 * x - 1) = x^2 + 2 * x + 2) :
    f 7 = 10 := by
  sorry

end f_of_7_l303_303765


namespace concert_revenue_l303_303436

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end concert_revenue_l303_303436


namespace add_n_to_constant_l303_303086

theorem add_n_to_constant (y n : ℝ) (h_eq : y^4 - 20 * y + 1 = 22) (h_n : n = 3) : y^4 - 20 * y + 4 = 25 :=
by
  sorry

end add_n_to_constant_l303_303086


namespace files_deleted_l303_303421

-- Definitions based on the conditions
def initial_files : ℕ := 93
def files_per_folder : ℕ := 8
def num_folders : ℕ := 9

-- The proof problem
theorem files_deleted : initial_files - (files_per_folder * num_folders) = 21 :=
by
  sorry

end files_deleted_l303_303421


namespace digit_place_value_ratio_l303_303107

theorem digit_place_value_ratio : 
  let num := 43597.2468
  let digit5_place_value := 10    -- tens place
  let digit2_place_value := 0.1   -- tenths place
  digit5_place_value / digit2_place_value = 100 := 
by 
  sorry

end digit_place_value_ratio_l303_303107


namespace simplify_and_rationalize_l303_303269

theorem simplify_and_rationalize :
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l303_303269


namespace sin_theta_value_l303_303825

theorem sin_theta_value (a : ℝ) (h : a ≠ 0) (h_tan : Real.tan θ = -a) (h_point : P = (a, -1)) : Real.sin θ = -Real.sqrt 2 / 2 :=
sorry

end sin_theta_value_l303_303825


namespace legs_walking_on_ground_l303_303028

def number_of_horses : ℕ := 14
def number_of_men : ℕ := number_of_horses
def legs_per_man : ℕ := 2
def legs_per_horse : ℕ := 4
def half (n : ℕ) : ℕ := n / 2

theorem legs_walking_on_ground :
  (half number_of_men) * legs_per_man + (half number_of_horses) * legs_per_horse = 42 :=
by
  sorry

end legs_walking_on_ground_l303_303028


namespace problem_statement_l303_303092

variable (x P : ℝ)

theorem problem_statement
  (h1 : x^2 - 5 * x + 6 < 0)
  (h2 : P = x^2 + 5 * x + 6) :
  (20 < P) ∧ (P < 30) :=
sorry

end problem_statement_l303_303092


namespace symmetric_intersection_points_eq_y_axis_l303_303839

theorem symmetric_intersection_points_eq_y_axis (k : ℝ) :
  (∀ x y : ℝ, (y = k * x + 1) ∧ (x^2 + y^2 + k * x - y - 9 = 0) → (∃ x' : ℝ, y = k * (-x') + 1 ∧ (x'^2 + y^2 + k * x' - y - 9 = 0) ∧ x' = -x)) →
  k = 0 :=
by
  sorry

end symmetric_intersection_points_eq_y_axis_l303_303839


namespace right_triangle_angle_ratio_l303_303882

theorem right_triangle_angle_ratio
  (a b : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) 
  (h : a / b = 5 / 4)
  (h3 : a + b = 90) :
  (a = 50) ∧ (b = 40) :=
by
  sorry

end right_triangle_angle_ratio_l303_303882


namespace girls_more_than_boys_l303_303768

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end girls_more_than_boys_l303_303768


namespace simplify_and_rationalize_l303_303277

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l303_303277


namespace fraction_identity_l303_303585

theorem fraction_identity (f : ℚ) (h : 32 * f^2 = 2^3) : f = 1 / 2 :=
sorry

end fraction_identity_l303_303585


namespace decreasing_interval_eqn_l303_303660

def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem decreasing_interval_eqn {a : ℝ} : (∀ x : ℝ, x < 6 → deriv (f a) x < 0) ↔ a ≥ 6 :=
sorry

end decreasing_interval_eqn_l303_303660


namespace solution_set_of_inequality_l303_303486

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
-- f(x) is symmetric about the origin
variable (symmetric_f : ∀ x, f (-x) = -f x)
-- f(2) = 2
variable (f_at_2 : f 2 = 2)
-- For any 0 < x2 < x1, the slope condition holds
variable (slope_cond : ∀ x1 x2, 0 < x2 ∧ x2 < x1 → (f x1 - f x2) / (x1 - x2) < 1)

theorem solution_set_of_inequality :
  {x : ℝ | f x - x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end solution_set_of_inequality_l303_303486


namespace problem1_problem2_l303_303786

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem1_problem2_l303_303786


namespace john_spending_l303_303901

theorem john_spending (X : ℝ) 
  (H1 : X * (1 / 4) + X * (1 / 3) + X * (1 / 6) + 6 = X) : 
  X = 24 := 
sorry

end john_spending_l303_303901


namespace least_pos_integer_to_yield_multiple_of_5_l303_303584

theorem least_pos_integer_to_yield_multiple_of_5 (n : ℕ) (h : n > 0) :
  ((567 + n) % 5 = 0) ↔ (n = 3) :=
by {
  sorry
}

end least_pos_integer_to_yield_multiple_of_5_l303_303584


namespace median_equal_range_not_greater_l303_303635

variable {α : Type} [LinearOrder α] {x1 x2 x3 x4 x5 x6 : α}

-- Define the conditions:
-- x1 is the minimum value and x6 is the maximum value in the set {x1, x2, x3, x4, x5, x6}
variable (hx_min : x1 ≤ x2 ∧ x1 ≤ x3 ∧ x1 ≤ x4 ∧ x1 ≤ x5 ∧ x1 ≤ x6)
variable (hx_max : x6 ≥ x2 ∧ x6 ≥ x3 ∧ x6 ≥ x4 ∧ x6 ≥ x5 ∧ x6 ≥ x1)

-- Prove that the median of {x2, x3, x4, x5} is equal to the median of {x1, x2, x3, x4, x5, x6}
theorem median_equal :
  (x2 + x3 + x4 + x5) / 4 = (x1 + x2 + x3 + x4 + x5 + x6) / 6 := by
  sorry

-- Prove that the range of {x2, x3, x4, x5} is not greater than the range of {x1, x2, x3, x4, x5, x6}
theorem range_not_greater :
  (x5 - x2) ≤ (x6 - x1) := by
  sorry

end median_equal_range_not_greater_l303_303635


namespace x_cubed_gt_y_squared_l303_303289

theorem x_cubed_gt_y_squared (x y : ℝ) (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end x_cubed_gt_y_squared_l303_303289


namespace parabola_ellipse_sum_distances_l303_303601

noncomputable def sum_distances_intersection_points (b c : ℝ) : ℝ :=
  2 * Real.sqrt b + 2 * Real.sqrt c

theorem parabola_ellipse_sum_distances
  (A B : ℝ)
  (h1 : A > 0) -- semi-major axis condition implied
  (h2 : B > 0) -- semi-minor axis condition implied
  (ellipse_eq : ∀ x y, (x^2) / A^2 + (y^2) / B^2 = 1)
  (focus_shared : ∃ f : ℝ, f = Real.sqrt (A^2 - B^2))
  (directrix_parabola : ∃ d : ℝ, d = B) -- directrix condition
  (intersections : ∃ (b c : ℝ), (b > 0 ∧ c > 0)) -- existence of such intersection points
  : sum_distances_intersection_points b c = 2 * Real.sqrt b + 2 * Real.sqrt c :=
sorry  -- proof omitted

end parabola_ellipse_sum_distances_l303_303601


namespace population_increase_rate_correct_l303_303154

variable (P0 P1 : ℕ)
variable (r : ℚ)

-- Given conditions
def initial_population := P0 = 200
def population_after_one_year := P1 = 220

-- Proof problem statement
theorem population_increase_rate_correct :
  initial_population P0 →
  population_after_one_year P1 →
  r = (P1 - P0 : ℚ) / P0 * 100 →
  r = 10 :=
by
  sorry

end population_increase_rate_correct_l303_303154


namespace problem_value_of_m_l303_303377

theorem problem_value_of_m (m : ℝ)
  (h1 : (m + 1) * x ^ (m ^ 2 - 3) = y)
  (h2 : m ^ 2 - 3 = 1)
  (h3 : m + 1 < 0) : 
  m = -2 := 
  sorry

end problem_value_of_m_l303_303377


namespace concert_revenue_l303_303435

-- Defining the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := ticket_price_adult / 2
def attendees_adults : ℕ := 183
def attendees_children : ℕ := 28

-- Defining the total revenue calculation based on the conditions
def total_revenue : ℕ :=
  attendees_adults * ticket_price_adult +
  attendees_children * ticket_price_child

-- The theorem to prove the total revenue
theorem concert_revenue : total_revenue = 5122 := by
  sorry

end concert_revenue_l303_303435


namespace hyperbola_asymptotes_angle_l303_303821

-- Define the given conditions and the proof problem
theorem hyperbola_asymptotes_angle (a b c : ℝ) (e : ℝ) (h1 : e = 2) 
  (h2 : e = c / a) (h3 : c = 2 * a) (h4 : b^2 + a^2 = c^2) : 
  ∃ θ : ℝ, θ = 60 :=
by 
  sorry -- Proof is omitted

end hyperbola_asymptotes_angle_l303_303821


namespace number_of_possible_values_of_x_l303_303389

theorem number_of_possible_values_of_x : 
  (∃ x : ℕ, ⌈Real.sqrt x⌉ = 12) → (set.Ico 144 169).card = 25 := 
by
  intros h
  sorry

end number_of_possible_values_of_x_l303_303389


namespace salad_chopping_l303_303022

theorem salad_chopping (tom_rate : ℝ) (tammy_rate : ℝ) (total_salad : ℝ)
  (h1 : tom_rate = 2 / 3)
  (h2 : tammy_rate = 3 / 2)
  (h3 : total_salad = 65) :
  let tom_share := (tom_rate / (tom_rate + tammy_rate)) * total_salad
  let tammy_share := (tammy_rate / (tom_rate + tammy_rate)) * total_salad
  let difference := tammy_share - tom_share
  let percentage_difference := (difference / tom_share) * 100
  percentage_difference = 125 := 
by {
  -- sorry to skip the proof
  sorry
}

end salad_chopping_l303_303022


namespace smallest_four_digit_divisible_by_4_and_5_l303_303313

theorem smallest_four_digit_divisible_by_4_and_5 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ ∀ m, 1000 ≤ m ∧ m < 10000 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_4_and_5_l303_303313


namespace rationalize_denominator_l303_303264

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l303_303264


namespace football_games_this_year_l303_303111

theorem football_games_this_year 
  (total_games : ℕ) 
  (games_last_year : ℕ) 
  (games_this_year : ℕ) 
  (h1 : total_games = 9) 
  (h2 : games_last_year = 5) 
  (h3 : total_games = games_last_year + games_this_year) : 
  games_this_year = 4 := 
sorry

end football_games_this_year_l303_303111


namespace area_of_ABCD_l303_303533

noncomputable def quadrilateral_area (AB BC AD DC : ℝ) : ℝ :=
  let area_ABC := 1 / 2 * AB * BC
  let area_ADC := 1 / 2 * AD * DC
  area_ABC + area_ADC

theorem area_of_ABCD {AB BC AD DC AC : ℝ}
  (h1 : AC = 5)
  (h2 : AB * AB + BC * BC = 25)
  (h3 : AD * AD + DC * DC = 25)
  (h4 : AB ≠ AD)
  (h5 : BC ≠ DC) :
  quadrilateral_area AB BC AD DC = 12 :=
sorry

end area_of_ABCD_l303_303533


namespace brenda_trays_l303_303350

-- Define main conditions
def cookies_per_tray : ℕ := 80
def cookies_per_box : ℕ := 60
def cost_per_box : ℕ := 350
def total_cost : ℕ := 1400  -- Using cents for calculation to avoid float numbers

-- State the problem
theorem brenda_trays :
  (total_cost / cost_per_box) * cookies_per_box / cookies_per_tray = 3 := 
by
  sorry

end brenda_trays_l303_303350


namespace expected_number_of_digits_l303_303616

open ProbabilityTheory

def fair_dodecahedral_die := { x : ℕ // 1 ≤ x ∧ x ≤ 12 }
def num_digits (n : ℕ) : ℕ := if n < 10 then 1 else 2

theorem expected_number_of_digits :
  measure_theory.integral (measure_theory.measure_space.comap (λ n : fair_dodecahedral_die, n.val) 
  (measure_theory.measure_space.measure_univ : measure_theory.measure fair_dodecahedral_die)) num_digits = 1.25 :=
begin
  sorry
end

end expected_number_of_digits_l303_303616


namespace geometric_sequence_s4_l303_303818

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0, a1, q => 0
| (n+1), a1, q => a1 * (1 - q^(n+1)) / (1 - q)

variable (a1 q : ℝ) (n : ℕ)

theorem geometric_sequence_s4  (h1 : a1 * (q^1) * (q^3) = 16) (h2 : geometric_sequence_sum 2 a1 q + a1 * (q^2) = 7) :
  geometric_sequence_sum 3 a1 q = 15 :=
sorry

end geometric_sequence_s4_l303_303818


namespace average_age_of_first_and_fifth_fastest_dogs_l303_303578

-- Definitions based on the conditions
def first_dog_age := 10
def second_dog_age := first_dog_age - 2
def third_dog_age := second_dog_age + 4
def fourth_dog_age := third_dog_age / 2
def fifth_dog_age := fourth_dog_age + 20

-- Statement to prove
theorem average_age_of_first_and_fifth_fastest_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 := by
  -- Add your proof here
  sorry

end average_age_of_first_and_fifth_fastest_dogs_l303_303578


namespace log_49_48_proof_l303_303952

variable (a b : ℝ)
variable (conditions : (1 / 7) ^ a = (1 / 3) ∧ Real.log 7 4 = b)

noncomputable def log_49_48_in_terms_of_a_b : ℝ :=
  if (1 / 7) ^ a = (1 / 3) ∧ Real.log 7 4 = b then
    (a + 2 * b) / 2
  else
    0

theorem log_49_48_proof : log_49_48_in_terms_of_a_b a b = Real.log 49 48 := by
  sorry

end log_49_48_proof_l303_303952


namespace min_tiles_to_cover_region_l303_303466

noncomputable def num_tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area

theorem min_tiles_to_cover_region : num_tiles_needed 6 2 36 72 = 216 :=
by 
  -- This is the format needed to include the assumptions and reach the conclusion
  sorry

end min_tiles_to_cover_region_l303_303466


namespace not_mutually_exclusive_option_D_l303_303609

-- Definitions for mutually exclusive events
def mutually_exclusive (event1 event2 : Prop) : Prop := ¬ (event1 ∧ event2)

-- Conditions as given in the problem
def eventA1 : Prop := True -- Placeholder for "score is greater than 8"
def eventA2 : Prop := True -- Placeholder for "score is less than 6"

def eventB1 : Prop := True -- Placeholder for "90 seeds germinate"
def eventB2 : Prop := True -- Placeholder for "80 seeds germinate"

def eventC1 : Prop := True -- Placeholder for "pass rate is higher than 70%"
def eventC2 : Prop := True -- Placeholder for "pass rate is 70%"

def eventD1 : Prop := True -- Placeholder for "average score is not lower than 90"
def eventD2 : Prop := True -- Placeholder for "average score is not higher than 120"

-- Lean proof statement
theorem not_mutually_exclusive_option_D :
  mutually_exclusive eventA1 eventA2 ∧
  mutually_exclusive eventB1 eventB2 ∧
  mutually_exclusive eventC1 eventC2 ∧
  ¬ mutually_exclusive eventD1 eventD2 :=
sorry

end not_mutually_exclusive_option_D_l303_303609


namespace shaded_area_in_design_l303_303692

theorem shaded_area_in_design (side_length : ℝ) (radius : ℝ)
  (h1 : side_length = 30) (h2 : radius = side_length / 6)
  (h3 : 6 * (π * radius^2) = 150 * π) :
  (side_length^2) - 6 * (π * radius^2) = 900 - 150 * π := 
by
  sorry

end shaded_area_in_design_l303_303692


namespace a_2_value_general_terms_T_n_value_l303_303457

-- Definitions based on conditions
def S (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of sequence {a_n}

def a (n : ℕ) : ℕ := (S n + 2) / 2  -- a_n is the arithmetic mean of S_n and 2

def b (n : ℕ) : ℕ := 2 * n - 1  -- Given general term for b_n

-- Prove a_2 = 4
theorem a_2_value : a 2 = 4 := 
by
  sorry

-- Prove the general terms
theorem general_terms (n : ℕ) : a n = 2^n ∧ b n = 2 * n - 1 := 
by
  sorry

-- Definition and sum of the first n terms of c_n
def c (n : ℕ) : ℕ := a n * b n

def T (n : ℕ) : ℕ := (2 * n - 3) * 2^(n + 1) + 6  -- Given sum of the first n terms of {c_n}

-- Prove T_n = (2n - 3)2^(n+1) + 6
theorem T_n_value (n : ℕ) : T n = (2 * n - 3) * 2^(n + 1) + 6 :=
by
  sorry

end a_2_value_general_terms_T_n_value_l303_303457


namespace fraction_of_tips_in_august_is_five_eighths_l303_303754

-- Definitions
def average_tips (other_tips_total : ℤ) (n : ℤ) : ℤ := other_tips_total / n
def total_tips (other_tips : ℤ) (august_tips : ℤ) : ℤ := other_tips + august_tips
def fraction (numerator : ℤ) (denominator : ℤ) : ℚ := (numerator : ℚ) / (denominator : ℚ)

-- Given conditions
variables (A : ℤ) -- average monthly tips for the other 6 months (March to July and September)
variables (other_months : ℤ := 6)
variables (tips_total_other : ℤ := other_months * A) -- total tips for the 6 other months
variables (tips_august : ℤ := 10 * A) -- tips for August
variables (total_tips_all : ℤ := tips_total_other + tips_august) -- total tips for all months

-- Prove the statement
theorem fraction_of_tips_in_august_is_five_eighths :
  fraction tips_august total_tips_all = 5 / 8 := by sorry

end fraction_of_tips_in_august_is_five_eighths_l303_303754


namespace john_reads_days_per_week_l303_303112

-- Define the conditions
def john_reads_books_per_day := 4
def total_books_read := 48
def total_weeks := 6

-- Theorem statement
theorem john_reads_days_per_week :
  (total_books_read / john_reads_books_per_day) / total_weeks = 2 :=
by
  sorry

end john_reads_days_per_week_l303_303112


namespace calculate_monthly_rent_l303_303775

theorem calculate_monthly_rent (P : ℝ) (R : ℝ) (T : ℝ) (M : ℝ) (rent : ℝ) :
  P = 12000 →
  R = 0.06 →
  T = 400 →
  M = 0.1 →
  rent = 103.70 :=
by
  intros hP hR hT hM
  sorry

end calculate_monthly_rent_l303_303775


namespace simplify_and_rationalize_l303_303275

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l303_303275


namespace lake_coverage_day_17_l303_303773

-- Define the state of lake coverage as a function of day
def lake_coverage (day : ℕ) : ℝ :=
  if day ≤ 20 then 2 ^ (day - 20) else 0

-- Prove that on day 17, the lake was covered by 12.5% algae
theorem lake_coverage_day_17 : lake_coverage 17 = 0.125 :=
by
  sorry

end lake_coverage_day_17_l303_303773


namespace solve_system_of_equations_l303_303427

theorem solve_system_of_equations :
  ∀ x y z : ℝ,
  (3 * x * y - 5 * y * z - x * z = 3 * y) →
  (x * y + y * z = -y) →
  (-5 * x * y + 4 * y * z + x * z = -4 * y) →
  (x = 2 ∧ y = -1 / 3 ∧ z = -3) ∨ 
  (y = 0 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l303_303427


namespace find_common_difference_l303_303075

def common_difference (S_odd S_even n : ℕ) (d : ℤ) : Prop :=
  S_even - S_odd = n / 2 * d

theorem find_common_difference :
  ∃ d : ℤ, common_difference 132 112 20 d ∧ d = -2 :=
  sorry

end find_common_difference_l303_303075


namespace length_second_train_is_125_l303_303452

noncomputable def length_second_train (speed_faster speed_slower distance1 : ℕ) (time_minutes : ℝ) : ℝ :=
  let relative_speed_m_per_minute := (speed_faster - speed_slower) * 1000 / 60
  let total_distance_covered := relative_speed_m_per_minute * time_minutes
  total_distance_covered - distance1

theorem length_second_train_is_125 :
  length_second_train 50 40 125 1.5 = 125 :=
  by sorry

end length_second_train_is_125_l303_303452


namespace angle_sum_x_y_l303_303846

def angle_A := 36
def angle_B := 80
def angle_C := 24

def target_sum : ℕ := 140

theorem angle_sum_x_y (angle_A angle_B angle_C : ℕ) (x y : ℕ) : 
  angle_A = 36 → angle_B = 80 → angle_C = 24 → x + y = 140 := by 
  intros _ _ _
  sorry

end angle_sum_x_y_l303_303846


namespace chairs_per_row_l303_303573

-- Definition of the given conditions
def rows : ℕ := 20
def people_per_chair : ℕ := 5
def total_people : ℕ := 600

-- The statement to be proven
theorem chairs_per_row (x : ℕ) (h : rows * (x * people_per_chair) = total_people) : x = 6 := 
by sorry

end chairs_per_row_l303_303573


namespace unit_price_first_purchase_l303_303199

theorem unit_price_first_purchase (x y : ℝ) (h1 : x * y = 500000) 
    (h2 : 1.4 * x * (y + 10000) = 770000) : x = 5 :=
by
  -- Proof details here
  sorry

end unit_price_first_purchase_l303_303199


namespace simplify_fraction_l303_303942

theorem simplify_fraction:
  ((1/2 - 1/3) / (3/7 + 1/9)) * (1/4) = 21/272 :=
by
  sorry

end simplify_fraction_l303_303942


namespace solve_for_x_l303_303542

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l303_303542


namespace mirror_side_length_l303_303187

theorem mirror_side_length (width length : ℝ) (area_wall : ℝ) (area_mirror : ℝ) (side_length : ℝ) 
  (h1 : width = 28) 
  (h2 : length = 31.5) 
  (h3 : area_wall = width * length)
  (h4 : area_mirror = area_wall / 2) 
  (h5 : area_mirror = side_length ^ 2) : 
  side_length = 21 := 
by 
  sorry

end mirror_side_length_l303_303187


namespace cubics_inequality_l303_303949

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 :=
sorry

end cubics_inequality_l303_303949


namespace unique_elements_in_set_l303_303827

theorem unique_elements_in_set (a : ℤ) (h : a ∈ ({0, 1, 2} : Finset ℤ)) :
  ({1, a^2 - a - 1, a^2 - 2*a + 2} : Finset ℤ).card = 3 ↔ a = 0 := by
  sorry

end unique_elements_in_set_l303_303827


namespace tangent_sum_formula_application_l303_303809

-- Define the problem's parameters and statement
noncomputable def thirty_three_degrees_radian := Real.pi * 33 / 180
noncomputable def seventeen_degrees_radian := Real.pi * 17 / 180
noncomputable def twenty_eight_degrees_radian := Real.pi * 28 / 180

theorem tangent_sum_formula_application :
  Real.tan seventeen_degrees_radian + Real.tan twenty_eight_degrees_radian + Real.tan seventeen_degrees_radian * Real.tan twenty_eight_degrees_radian = 1 := 
sorry

end tangent_sum_formula_application_l303_303809


namespace domain_of_expression_l303_303063

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end domain_of_expression_l303_303063


namespace number_of_solutions_l303_303140

theorem number_of_solutions (h₁ : ∀ x, 50 * x % 100 = 0 → (x % 2 = 0)) 
                            (h₂ : ∀ x, (x % 2 = 0) → (∀ k, 1 ≤ k ∧ k ≤ 49 → (k * x % 100 ≠ 0)))
                            (h₃ : ∀ x, 1 ≤ x ∧ x ≤ 100) : 
  ∃ count, count = 20 := 
by {
  -- Here, we usually would provide a method to count all valid x values meeting the conditions,
  -- but we skip the proof as instructed.
  sorry
}

end number_of_solutions_l303_303140


namespace max_4x3_y3_l303_303416

theorem max_4x3_y3 (x y : ℝ) (h1 : x ≤ 2) (h2 : y ≤ 3) (h3 : x + y = 3) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : 
  4 * x^3 + y^3 ≤ 33 :=
sorry

end max_4x3_y3_l303_303416


namespace problem_equivalence_l303_303012

section ProblemDefinitions

def odd_function_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def statement_A (f : ℝ → ℝ) : Prop :=
  (∀ x < 0, f x = -Real.log (-x)) →
  odd_function_condition f →
  ∀ x > 0, f x ≠ -Real.log x

def statement_B (a : ℝ) : Prop :=
  Real.logb a (1 / 2) < 1 →
  (0 < a ∧ a < 1 / 2) ∨ (1 < a)

def statement_C : Prop :=
  ∀ x, (Real.logb 2 (Real.sqrt (x-1)) = (1/2) * Real.logb 2 x)

def statement_D (x1 x2 : ℝ) : Prop :=
  (x1 + Real.log x1 = 2) →
  (Real.log (1 - x2) - x2 = 1) →
  x1 + x2 = 1

end ProblemDefinitions

structure MathProofProblem :=
  (A : ∀ f : ℝ → ℝ, statement_A f)
  (B : ∀ a : ℝ, statement_B a)
  (C : statement_C)
  (D : ∀ x1 x2 : ℝ, statement_D x1 x2)

theorem problem_equivalence : MathProofProblem :=
  { A := sorry,
    B := sorry,
    C := sorry,
    D := sorry }

end problem_equivalence_l303_303012


namespace number_of_candidates_l303_303451

theorem number_of_candidates
  (n : ℕ)
  (h : n * (n - 1) = 132) : 
  n = 12 :=
sorry

end number_of_candidates_l303_303451


namespace isosceles_triangle_base_angles_l303_303511

theorem isosceles_triangle_base_angles 
  (α β : ℝ) -- α and β are the base angles
  (h : α = β)
  (height leg : ℝ)
  (h_height_leg : height = leg / 2) : 
  α = 75 ∨ α = 15 :=
by
  sorry

end isosceles_triangle_base_angles_l303_303511


namespace number_of_integers_with_6_or_7_as_digit_in_base9_l303_303664

/-- 
  There are 729 smallest positive integers written in base 9.
  We want to determine how many of these integers use the digits 6 or 7 (or both) at least once.
-/
theorem number_of_integers_with_6_or_7_as_digit_in_base9 : 
  ∃ n : ℕ, n = 729 ∧ ∃ m : ℕ, m = n - 7^3 := sorry

end number_of_integers_with_6_or_7_as_digit_in_base9_l303_303664


namespace solve_for_x_l303_303560

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l303_303560


namespace tom_boxes_needed_l303_303304

-- Definitions of given conditions
def room_length : ℕ := 16
def room_width : ℕ := 20
def box_coverage : ℕ := 10
def already_covered : ℕ := 250

-- The total area of the living room
def total_area : ℕ := room_length * room_width

-- The remaining area that needs to be covered
def remaining_area : ℕ := total_area - already_covered

-- The number of boxes required to cover the remaining area
def boxes_needed : ℕ := remaining_area / box_coverage

-- The theorem statement
theorem tom_boxes_needed : boxes_needed = 7 := by
  -- The proof will go here
  sorry

end tom_boxes_needed_l303_303304


namespace guest_bedroom_area_l303_303893

theorem guest_bedroom_area 
  (master_bedroom_bath_area : ℝ)
  (kitchen_guest_bath_living_area : ℝ)
  (total_rent : ℝ)
  (rate_per_sqft : ℝ)
  (num_guest_bedrooms : ℕ)
  (area_guest_bedroom : ℝ) :
  master_bedroom_bath_area = 500 →
  kitchen_guest_bath_living_area = 600 →
  total_rent = 3000 →
  rate_per_sqft = 2 →
  num_guest_bedrooms = 2 →
  (total_rent / rate_per_sqft) - (master_bedroom_bath_area + kitchen_guest_bath_living_area) / num_guest_bedrooms = area_guest_bedroom → 
  area_guest_bedroom = 200 := by
  sorry

end guest_bedroom_area_l303_303893


namespace binom_six_two_l303_303050

-- Define the binomial coefficient function
def binom (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_six_two : binom 6 2 = 15 := by
  sorry

end binom_six_two_l303_303050


namespace area_of_enclosed_triangle_l303_303919

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
1 / 2 * b * h

noncomputable def base_length (x1 x2 : ℝ) : ℝ :=
x2 - x1

noncomputable def height (y1 y2 : ℝ) : ℝ :=
y1 - y2

-- Coordinates of the vertices from the solution:
def vertex1 : ℝ × ℝ := (1 / 2, 2)
def vertex2 : ℝ × ℝ := (4, 2)
def vertex3 : ℝ × ℝ := (6 / 5, 17 / 5)

-- Base along y = 2
def base : ℝ := base_length (vertex1.1) (vertex2.1)

-- Height from vertex3 to line y = 2
def height_triangle : ℝ := height (vertex3.2) 2

-- The area of the triangle
def area_triangle : ℝ := area_of_triangle base height_triangle

-- Statement to prove
theorem area_of_enclosed_triangle : area_triangle = 2.45 :=
sorry

end area_of_enclosed_triangle_l303_303919


namespace difference_between_Annette_and_Sara_l303_303042

-- Define the weights of the individuals
variables (A C S B E : ℝ)

-- Conditions given in the problem
def condition1 := A + C = 95
def condition2 := C + S = 87
def condition3 := A + S = 97
def condition4 := C + B = 100
def condition5 := A + C + B = 155
def condition6 := A + S + B + E = 240
def condition7 := E = 1.25 * C

-- The theorem that we want to prove
theorem difference_between_Annette_and_Sara (A C S B E : ℝ)
  (h1 : condition1 A C)
  (h2 : condition2 C S)
  (h3 : condition3 A S)
  (h4 : condition4 C B)
  (h5 : condition5 A C B)
  (h6 : condition6 A S B E)
  (h7 : condition7 C E) :
  A - S = 8 :=
by {
  sorry
}

end difference_between_Annette_and_Sara_l303_303042


namespace simplify_and_rationalize_l303_303276

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) := 
  sorry

end simplify_and_rationalize_l303_303276


namespace range_of_m_l303_303216

-- Defining the conditions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Defining the main theorem
theorem range_of_m (m : ℝ) : (∀ x : ℝ, q x → p x m) ∧ ¬ (∀ x : ℝ, p x m → q x) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end range_of_m_l303_303216


namespace solve_for_x_l303_303536

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l303_303536


namespace min_dot_product_l303_303675

noncomputable def ellipse_eq_p (x y : ℝ) : Prop :=
    x^2 / 9 + y^2 / 8 = 1

noncomputable def dot_product_op_fp (x y : ℝ) : ℝ :=
    x^2 + x + y^2

theorem min_dot_product : 
    (∀ x y : ℝ, ellipse_eq_p x y → dot_product_op_fp x y = 6) := 
sorry

end min_dot_product_l303_303675


namespace madeline_has_five_boxes_l303_303526

theorem madeline_has_five_boxes 
    (total_crayons_per_box : ℕ)
    (not_used_fraction1 : ℚ)
    (not_used_fraction2 : ℚ)
    (used_fraction2 : ℚ)
    (total_boxes_not_used : ℚ)
    (total_unused_crayons : ℕ)
    (unused_in_last_box : ℚ)
    (total_boxes : ℕ) :
    total_crayons_per_box = 24 →
    not_used_fraction1 = 5 / 8 →
    not_used_fraction2 = 1 / 3 →
    used_fraction2 = 2 / 3 →
    total_boxes_not_used = 4 →
    total_unused_crayons = 70 →
    total_boxes = 5 :=
by
  -- Insert proof here
  sorry

end madeline_has_five_boxes_l303_303526


namespace candy_bar_cost_correct_l303_303933

-- Definitions based on conditions
def candy_bar_cost := 3
def chocolate_cost := candy_bar_cost + 5
def total_cost := chocolate_cost + candy_bar_cost

-- Assertion to be proved
theorem candy_bar_cost_correct :
  total_cost = 11 → candy_bar_cost = 3 :=
by
  intro h
  simp [total_cost, chocolate_cost, candy_bar_cost] at h
  sorry

end candy_bar_cost_correct_l303_303933


namespace ring_stack_vertical_distance_l303_303037

theorem ring_stack_vertical_distance :
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  ∃ n, (top_ring_outer_diameter - bottom_ring_outer_diameter) / decrement + 1 = n ∧
       n * ring_thickness = 260 :=
by {
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  sorry
}

end ring_stack_vertical_distance_l303_303037


namespace Tim_total_score_l303_303468

/-- Given the following conditions:
1. A single line is worth 1000 points.
2. A tetris is worth 8 times a single line.
3. If a single line and a tetris are made consecutively, the score of the tetris doubles.
4. If two tetrises are scored back to back, an additional 5000-point bonus is awarded.
5. If a player scores a single, double and triple line consecutively, a 3000-point bonus is awarded.
6. Tim scored 6 singles, 4 tetrises, 2 doubles, and 1 triple during his game.
7. He made a single line and a tetris consecutively once, scored 2 tetrises back to back, 
   and scored a single, double and triple consecutively.
Prove that Tim’s total score is 54000 points.
-/
theorem Tim_total_score :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6 * single_points
  let tetrises := 4 * tetris_points
  let base_score := singles + tetrises
  let consecutive_tetris_bonus := tetris_points
  let back_to_back_tetris_bonus := 5000
  let consecutive_lines_bonus := 3000
  let total_score := base_score + consecutive_tetris_bonus + back_to_back_tetris_bonus + consecutive_lines_bonus
  total_score = 54000 := by
  sorry

end Tim_total_score_l303_303468


namespace mean_temperature_is_88_75_l303_303153

def temperatures : List ℕ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 : (List.sum temperatures : ℚ) / temperatures.length = 88.75 := by
  sorry

end mean_temperature_is_88_75_l303_303153


namespace find_range_of_x_l303_303371

-- Conditions
variable (f : ℝ → ℝ)
variable (even_f : ∀ x : ℝ, f x = f (-x))
variable (mono_incr_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Equivalent proof statement
theorem find_range_of_x (x : ℝ) :
  f (Real.log (abs (x + 1)) / Real.log (1 / 2)) < f (-1) ↔ x ∈ Set.Ioo (-3 : ℝ) (-3 / 2) ∪ Set.Ioo (-1 / 2) 1 := by
  sorry

end find_range_of_x_l303_303371


namespace Connie_savings_l303_303620

theorem Connie_savings (cost_of_watch : ℕ) (extra_needed : ℕ) (saved_amount : ℕ) : 
  cost_of_watch = 55 → 
  extra_needed = 16 → 
  saved_amount = cost_of_watch - extra_needed → 
  saved_amount = 39 := 
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end Connie_savings_l303_303620


namespace pow_mul_eq_add_l303_303483

variable (a : ℝ)

theorem pow_mul_eq_add : a^2 * a^3 = a^5 := 
by 
  sorry

end pow_mul_eq_add_l303_303483


namespace probability_of_multiple_of_42_is_zero_l303_303014

-- Given conditions
def factors_200 : Set ℕ := {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200}
def multiple_of_42 (n : ℕ) : Prop := n % 42 = 0

-- Problem statement: the probability of selecting a multiple of 42 from the factors of 200 is 0.
theorem probability_of_multiple_of_42_is_zero : 
  ∀ (n : ℕ), n ∈ factors_200 → ¬ multiple_of_42 n := 
by
  sorry

end probability_of_multiple_of_42_is_zero_l303_303014


namespace age_difference_between_brother_and_cousin_is_five_l303_303986

variable (Lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ)

-- Conditions
axiom lexie_age_def : Lexie_age = 8
axiom grandma_age_def : grandma_age = 68
axiom lexie_brother_condition : Lexie_age = brother_age + 6
axiom lexie_sister_condition : sister_age = 2 * Lexie_age
axiom uncle_grandma_condition : uncle_age = grandma_age - 12
axiom cousin_brother_condition : cousin_age = brother_age + 5

-- Goal
theorem age_difference_between_brother_and_cousin_is_five : 
  Lexie_age = 8 → grandma_age = 68 → brother_age = Lexie_age - 6 → cousin_age = brother_age + 5 → cousin_age - brother_age = 5 :=
by sorry

end age_difference_between_brother_and_cousin_is_five_l303_303986


namespace prove_f_10_l303_303218

variable (f : ℝ → ℝ)

-- Conditions from the problem
def condition : Prop := ∀ x : ℝ, f (3 ^ x) = x

-- Statement of the problem
theorem prove_f_10 (h : condition f) : f 10 = Real.log 10 / Real.log 3 :=
by
  sorry

end prove_f_10_l303_303218


namespace juvy_chives_l303_303117

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end juvy_chives_l303_303117


namespace joan_seashells_initially_l303_303248

variable (mikeGave joanTotal : ℕ)

theorem joan_seashells_initially (h : mikeGave = 63) (t : joanTotal = 142) : joanTotal - mikeGave = 79 := 
by
  sorry

end joan_seashells_initially_l303_303248


namespace simplify_rationalize_denominator_l303_303283

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l303_303283


namespace age_sum_l303_303176

theorem age_sum (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 10) : a + b + c = 27 := by
  sorry

end age_sum_l303_303176


namespace marble_ratio_l303_303166

theorem marble_ratio (A J C : ℕ) (h1 : 3 * (A + J + C) = 60) (h2 : A = 4) (h3 : C = 8) : A / J = 1 / 2 :=
by sorry

end marble_ratio_l303_303166


namespace rationalize_denominator_l303_303267

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end rationalize_denominator_l303_303267


namespace mateen_garden_area_l303_303686

theorem mateen_garden_area :
  ∃ (L W : ℝ), (20 * L = 1000) ∧ (8 * (2 * L + 2 * W) = 1000) ∧ (L * W = 625) :=
by
  sorry

end mateen_garden_area_l303_303686


namespace overtime_percentage_increase_l303_303463

-- Define basic conditions
def basic_hours := 40
def total_hours := 48
def basic_pay := 20
def total_wage := 25

-- Calculate overtime hours and wages
def overtime_hours := total_hours - basic_hours
def overtime_pay := total_wage - basic_pay

-- Define basic and overtime hourly rates
def basic_hourly_rate := basic_pay / basic_hours
def overtime_hourly_rate := overtime_pay / overtime_hours

-- Calculate and state the theorem for percentage increase
def percentage_increase := ((overtime_hourly_rate - basic_hourly_rate) / basic_hourly_rate) * 100

theorem overtime_percentage_increase :
  percentage_increase = 25 :=
by
  sorry

end overtime_percentage_increase_l303_303463


namespace crayons_lost_or_given_away_correct_l303_303531

def initial_crayons : ℕ := 606
def remaining_crayons : ℕ := 291
def crayons_lost_or_given_away : ℕ := initial_crayons - remaining_crayons

theorem crayons_lost_or_given_away_correct :
  crayons_lost_or_given_away = 315 :=
by
  sorry

end crayons_lost_or_given_away_correct_l303_303531


namespace simplest_quadratic_radical_problem_l303_303472

/-- The simplest quadratic radical -/
def simplest_quadratic_radical (r : ℝ) : Prop :=
  ((∀ a b : ℝ, r = a * b → b = 1 ∧ a = r) ∧ (∀ a b : ℝ, r ≠ a / b))

theorem simplest_quadratic_radical_problem :
  (simplest_quadratic_radical (Real.sqrt 6)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 8)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt (1/3))) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 4)) :=
by
  sorry

end simplest_quadratic_radical_problem_l303_303472


namespace negation_of_existential_l303_303437

theorem negation_of_existential (P : Prop) :
  (¬ (∃ x : ℝ, x ^ 3 > 0)) ↔ (∀ x : ℝ, x ^ 3 ≤ 0) :=
by
  sorry

end negation_of_existential_l303_303437


namespace geometric_sequence_min_value_l303_303651

theorem geometric_sequence_min_value
  (q : ℝ) (a : ℕ → ℝ)
  (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n)
  (h_geom : ∀ k, a k = q ^ k)
  (h_eq : a m * (a n) ^ 2 = (a 4) ^ 2)
  (h_sum : m + 2 * n = 8) :
  ∀ (f : ℝ), f = (2 / m + 1 / n) → f ≥ 1 :=
by
  sorry

end geometric_sequence_min_value_l303_303651


namespace ab_range_l303_303836

theorem ab_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a * b = a + b) : 1 / 4 ≤ a * b :=
sorry

end ab_range_l303_303836


namespace inequality_for_pos_reals_equality_condition_l303_303983

open Real

theorem inequality_for_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / c + c / b = 4 * a / (a + b)) ↔ (a = b ∧ b = c) :=
by
  -- Theorem Statement Proof Skeleton
  sorry

end inequality_for_pos_reals_equality_condition_l303_303983


namespace neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l303_303734

theorem neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one :
  ¬(∃ x : ℝ, x^2 < 1) ↔ ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 := 
by 
  sorry

end neg_exists_x_sq_lt_one_eqv_forall_x_real_x_leq_neg_one_or_x_geq_one_l303_303734


namespace mangoes_total_l303_303196

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end mangoes_total_l303_303196


namespace distance_between_points_l303_303171

noncomputable def distance (x1 y1 x2 y2 : ℝ) := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points : 
  distance (-3) (1/2) 4 (-7) = Real.sqrt 105.25 := 
by 
  sorry

end distance_between_points_l303_303171


namespace average_salary_of_employees_l303_303564

theorem average_salary_of_employees
  (A : ℝ)  -- Define the average monthly salary A of 18 employees
  (h1 : 18*A + 5800 = 19*(A + 200))  -- Condition given in the problem
  : A = 2000 :=  -- The conclusion we need to prove
by
  sorry

end average_salary_of_employees_l303_303564


namespace find_a_from_conditions_l303_303732

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end find_a_from_conditions_l303_303732


namespace sum_of_perimeters_triangles_l303_303612

theorem sum_of_perimeters_triangles (a : ℕ → ℕ) (side_length : ℕ) (P : ℕ → ℕ):
  (∀ n : ℕ, a 0 = side_length ∧ P 0 = 3 * a 0) →
  (∀ n : ℕ, a (n + 1) = a n / 2 ∧ P (n + 1) = 3 * a (n + 1)) →
  (side_length = 45) →
  ∑' n, P n = 270 :=
by
  -- the proof would continue here
  sorry

end sum_of_perimeters_triangles_l303_303612


namespace avg_not_equal_median_equal_std_dev_less_range_not_greater_l303_303650

section

variables (x : List ℝ) (h_sorted : x = x.sorted) (hx1 : ∀ y ∈ x, y ≥ x.head!) 
(hx6 : ∀ y ∈ x, y ≤ x.getLast $ by simp [List.isEmpty]) (h_len : x.length = 6)
(h_min : x.head! = x.nthLe 0 sorry) (h_max : x.getLast $ by simp [List.isEmpty] = x.nthLe 5 sorry)

-- Prove 1: The average of x_2, x_3, x_4, x_5 is not equal to the average of x_1, x_2, ..., x_6
theorem avg_not_equal (hx1 : x_1 = x.nthLe 0 sorry) (hx6 : x_6 = x.nthLe 5 sorry): 
  (x.drop 1).take 4).sum / 4 ≠ x.sum / 6 := sorry

-- Prove 2: The median of x_2, x_3, x_4, x_5 is equal to the median of x_1, x_2, ..., x_6
theorem median_equal : 
  ((x.drop 1).take 4)).nthLe 1 sorry + ((x.drop 1).take 4)).nthLe 2 sorry) / 2 = (x.nthLe 2 sorry + x.nthLe 3 sorry) / 2 := sorry

-- Prove 3: The standard deviation of x_2, x_3, x_4, x_5 is less than the standard deviation of x_1, x_2, ..., x_6
theorem std_dev_less : 
  standard_deviation ((x.drop 1).take 4)) < standard_deviation x := sorry

-- Prove 4: The range of x_2, x_3, x_4, x_5 is not greater than the range of x_1, x_2, ..., x_6
theorem range_not_greater : 
  ((x.drop 1).take 4)).nthLe 3 sorry - ((x.drop 1).take 4)).nthLe 0 sorry ≤ x.nthLe 5 sorry - x.nthLe 0 sorry := sorry

end

end avg_not_equal_median_equal_std_dev_less_range_not_greater_l303_303650


namespace sum_of_squares_diagonals_of_rhombus_l303_303966

theorem sum_of_squares_diagonals_of_rhombus (d1 d2 : ℝ) (h : (d1 / 2)^2 + (d2 / 2)^2 = 4) : d1^2 + d2^2 = 16 :=
sorry

end sum_of_squares_diagonals_of_rhombus_l303_303966


namespace condition1_condition2_condition3_condition4_l303_303127

-- Proof for the equivalence of conditions and point descriptions

theorem condition1 (x y : ℝ) : 
  (x >= -2) ↔ ∃ y : ℝ, x = -2 ∨ x > -2 := 
by
  sorry

theorem condition2 (x y : ℝ) : 
  (-2 < x ∧ x < 2) ↔ ∃ y : ℝ, -2 < x ∧ x < 2 := 
by
  sorry

theorem condition3 (x y : ℝ) : 
  (|x| < 2) ↔ -2 < x ∧ x < 2 :=
by
  sorry

theorem condition4 (x y : ℝ) : 
  (|x| ≥ 2) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by 
  sorry

end condition1_condition2_condition3_condition4_l303_303127


namespace count_integer_values_of_x_l303_303385

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l303_303385


namespace find_x2_y2_l303_303205

theorem find_x2_y2 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : xy + x + y = 35) (h4 : xy * (x + y) = 360) : x^2 + y^2 = 185 := by
  sorry

end find_x2_y2_l303_303205


namespace hyperbolas_same_asymptotes_l303_303147

theorem hyperbolas_same_asymptotes :
  (∀ x y, (x^2 / 4 - y^2 / 9 = 1) → (∃ k, y = k * x)) →
  (∀ x y, (y^2 / 18 - x^2 / N = 1) → (∃ k, y = k * x)) →
  N = 8 :=
by sorry

end hyperbolas_same_asymptotes_l303_303147


namespace differential_solution_l303_303109

theorem differential_solution (C : ℝ) : 
  ∃ y : ℝ → ℝ, (∀ x : ℝ, y x = C * (1 + x^2)) := 
by
  sorry

end differential_solution_l303_303109


namespace matches_C_won_l303_303069

variable (A_wins B_wins D_wins total_matches wins_C : ℕ)

theorem matches_C_won 
  (hA : A_wins = 3)
  (hB : B_wins = 1)
  (hD : D_wins = 0)
  (htot : total_matches = 6)
  (h_sum_wins: A_wins + B_wins + D_wins + wins_C = total_matches)
  : wins_C = 2 :=
by
  sorry

end matches_C_won_l303_303069


namespace median_equality_range_inequality_l303_303642

variable {x1 x2 x3 x4 x5 x6 : ℝ}

-- Given conditions
def is_min_max (x1 x6 : ℝ) (xs : List ℝ) : Prop :=
  x1 = xs.minimum ∧ x6 = xs.maximum

-- Propositions to prove
theorem median_equality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x3 + x4) / 2 = [x1, x2, x3, x4, x5, x6].median :=
sorry

theorem range_inequality (xs : List ℝ) (h : is_min_max x1 x6 [x1, x2, x3, x4, x5, x6]) :
  (x5 - x2) ≤ (x6 - x1) :=
sorry

end median_equality_range_inequality_l303_303642


namespace domain_of_expression_l303_303064

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end domain_of_expression_l303_303064


namespace number_of_people_l303_303678

theorem number_of_people (x : ℕ) : 
  (x % 10 = 1) ∧
  (x % 9 = 1) ∧
  (x % 8 = 1) ∧
  (x % 7 = 1) ∧
  (x % 6 = 1) ∧
  (x % 5 = 1) ∧
  (x % 4 = 1) ∧
  (x % 3 = 1) ∧
  (x % 2 = 1) ∧
  (x < 5000) →
  x = 2521 :=
sorry

end number_of_people_l303_303678


namespace mangoes_combined_l303_303194

variable (Alexis Dilan Ashley : ℕ)

theorem mangoes_combined :
  (Alexis = 60) → (Alexis = 4 * (Dilan + Ashley)) → (Alexis + Dilan + Ashley = 75) := 
by
  intros h₁ h₂
  sorry

end mangoes_combined_l303_303194


namespace shoe_store_ratio_l303_303606

theorem shoe_store_ratio
  (marked_price : ℝ)
  (discount : ℝ) (discount_eq : discount = 1/4)
  (cost_factor : ℝ) (cost_factor_eq : cost_factor = 2/3) :
  (cost_factor * (1 - discount) * marked_price / marked_price) = 1/2 := 
by
  -- Insert proof here
  sorry

end shoe_store_ratio_l303_303606


namespace factorization_result_l303_303145

theorem factorization_result (a b : ℤ) (h1 : 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) :
  a + 2 * b = 20 :=
by
  sorry

end factorization_result_l303_303145


namespace parabola_min_area_l303_303954

-- Definition of the parabola C with vertex at the origin and focus on the positive y-axis
-- (Conditions 1 and 2)
def parabola_eq (x y : ℝ) : Prop := x^2 = 6 * y

-- Line l defined by mx + y - 3/2 = 0 (Condition 3)
def line_eq (m x y : ℝ) : Prop := m * x + y - 3 / 2 = 0

-- Formal statement combining all conditions to prove the equivalent Lean statement
theorem parabola_min_area :
  (∀ x y : ℝ, parabola_eq x y ↔ x^2 = 6 * y) ∧
  (∀ m x y : ℝ, line_eq m x y ↔ m * x + y - 3 / 2 = 0) →
  (parabola_eq 0 0) ∧ (∃ y > 0, parabola_eq 0 y ∧ line_eq 0 0 (y/2) ∧ y = 3 / 2) ∧
  ∀ A B P : ℝ, line_eq 0 A B ∧ line_eq 0 B P ∧ A^2 + B^2 > 0 → 
  ∃ min_S : ℝ, min_S = 9 :=
by
  sorry

end parabola_min_area_l303_303954


namespace pies_sold_in_week_l303_303034

def daily_pies := 8
def days_in_week := 7
def total_pies := 56

theorem pies_sold_in_week : daily_pies * days_in_week = total_pies :=
by
  sorry

end pies_sold_in_week_l303_303034


namespace max_followers_1009_l303_303889

noncomputable def maxFollowers (N Y : Nat) (knights : Nat) (liars : Nat) (followers : Nat) : Nat :=
  if N = 2018 ∧ Y = 1009 ∧ (knights + liars + followers = N) then
    1009
  else
    sorry

theorem max_followers_1009 :
  ∃ followers, maxFollowers 2018 1009 knights liars followers = 1009 :=
by {
  use 1009,
  have h1 : 2018 = (knights + liars + 1009),
  have h2 : (1009 = 2018 - 1009),
  exact_and h1 h2,
  sorry
}

end max_followers_1009_l303_303889


namespace max_value_of_n_l303_303067

theorem max_value_of_n (A B : ℤ) (h1 : A * B = 48) : 
  ∃ n, (∀ n', (∃ A' B', (A' * B' = 48) ∧ (n' = 2 * B' + 3 * A')) → n' ≤ n) ∧ n = 99 :=
by
  sorry

end max_value_of_n_l303_303067


namespace shanghai_population_scientific_notation_l303_303565

theorem shanghai_population_scientific_notation :
  16.3 * 10^6 = 1.63 * 10^7 :=
sorry

end shanghai_population_scientific_notation_l303_303565


namespace determine_beta_l303_303956

-- Define a structure for angles in space
structure Angle where
  measure : ℝ

-- Define the conditions
def alpha : Angle := ⟨30⟩
def parallel_sides (a b : Angle) : Prop := true  -- Simplification for the example, should be defined properly for general case

-- The theorem to be proved
theorem determine_beta (α β : Angle) (h1 : α = Angle.mk 30) (h2 : parallel_sides α β) : β = Angle.mk 30 ∨ β = Angle.mk 150 := by
  sorry

end determine_beta_l303_303956


namespace train_crossing_pole_time_l303_303607

/-- 
Given the conditions:
1. The train is running at a speed of 60 km/hr.
2. The length of the train is 66.66666666666667 meters.
Prove that it takes 4 seconds for the train to cross the pole.
-/
theorem train_crossing_pole_time :
  let speed_km_hr := 60
  let length_m := 66.66666666666667
  let conversion_factor := 1000 / 3600
  let speed_m_s := speed_km_hr * conversion_factor
  let time := length_m / speed_m_s
  time = 4 :=
by
  sorry

end train_crossing_pole_time_l303_303607


namespace find_student_hourly_rate_l303_303465

-- Definitions based on conditions
def janitor_work_time : ℝ := 8  -- Janitor can clean the school in 8 hours
def student_work_time : ℝ := 20  -- Student can clean the school in 20 hours
def janitor_hourly_rate : ℝ := 21  -- Janitor is paid $21 per hour
def cost_difference : ℝ := 8  -- The cost difference between janitor alone and both together is $8

-- The value we need to prove
def student_hourly_rate := 7

theorem find_student_hourly_rate
  (janitor_work_time : ℝ)
  (student_work_time : ℝ)
  (janitor_hourly_rate : ℝ)
  (cost_difference : ℝ) :
  S = 7 :=
by
  -- Calculations and logic can be filled here to prove the theorem
  sorry

end find_student_hourly_rate_l303_303465


namespace math_club_team_selection_l303_303422

noncomputable def choose (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.descFactorial n k / Nat.factorial k else 0

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_selected := 4
  let girls_selected := 4
  choose boys boys_selected * choose girls girls_selected = 103950 := 
by simp [choose]; sorry

end math_club_team_selection_l303_303422


namespace largest_term_quotient_l303_303525

theorem largest_term_quotient (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_S_def : ∀ n, S n = (n * (a 0 + a n)) / 2)
  (h_S15_pos : S 15 > 0)
  (h_S16_neg : S 16 < 0) :
  ∃ m, 1 ≤ m ∧ m ≤ 15 ∧
       ∀ k, (1 ≤ k ∧ k ≤ 15) → (S m / a m) ≥ (S k / a k) ∧ m = 8 := 
sorry

end largest_term_quotient_l303_303525


namespace dog_bones_remaining_l303_303332

noncomputable def initial_bones : ℕ := 350
noncomputable def factor : ℕ := 9
noncomputable def found_bones : ℕ := factor * initial_bones
noncomputable def total_bones : ℕ := initial_bones + found_bones
noncomputable def bones_given_away : ℕ := 120
noncomputable def bones_remaining : ℕ := total_bones - bones_given_away

theorem dog_bones_remaining : bones_remaining = 3380 :=
by
  sorry

end dog_bones_remaining_l303_303332


namespace fraction_of_remaining_paint_used_l303_303249

theorem fraction_of_remaining_paint_used (total_paint : ℕ) (first_week_fraction : ℚ) (total_used : ℕ) :
  total_paint = 360 ∧ first_week_fraction = 1/6 ∧ total_used = 120 →
  (total_used - first_week_fraction * total_paint) / (total_paint - first_week_fraction * total_paint) = 1/5 :=
  by
    sorry

end fraction_of_remaining_paint_used_l303_303249


namespace binomial_sum_mod_prime_l303_303118

theorem binomial_sum_mod_prime (n : ℕ) (p : ℕ) (h : Prime p) (H : p = 2023) :
  (∑ k in Finset.range 101, Nat.choose 2020 (k + 3)) % 2023 = 578 :=
by
  sorry

end binomial_sum_mod_prime_l303_303118


namespace proof_problem_l303_303661

variables (a b : ℝ)
variable (h : a ≠ b)
variable (h1 : a * Real.exp a = b * Real.exp b)
variable (p : Prop := Real.log a + a = Real.log b + b)
variable (q : Prop := (a + 1) * (b + 1) < 0)

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l303_303661


namespace least_5_digit_divisible_by_12_15_18_l303_303015

theorem least_5_digit_divisible_by_12_15_18 : 
  ∃ n, n >= 10000 ∧ n < 100000 ∧ (180 ∣ n) ∧ n = 10080 :=
by
  -- Proof goes here
  sorry

end least_5_digit_divisible_by_12_15_18_l303_303015


namespace oxygen_atoms_l303_303764

theorem oxygen_atoms (x : ℤ) (h : 27 + 16 * x + 3 = 78) : x = 3 := 
by 
  sorry

end oxygen_atoms_l303_303764


namespace solve_quadratic_eqn_l303_303440

theorem solve_quadratic_eqn :
  ∃ x₁ x₂ : ℝ, (x - 6) * (x + 2) = 0 ↔ (x = x₁ ∨ x = x₂) ∧ x₁ = 6 ∧ x₂ = -2 :=
by
  sorry

end solve_quadratic_eqn_l303_303440


namespace prob_Z_l303_303462

theorem prob_Z (P_X P_Y P_W P_Z : ℚ) (hX : P_X = 1/4) (hY : P_Y = 1/3) (hW : P_W = 1/6) 
(hSum : P_X + P_Y + P_Z + P_W = 1) : P_Z = 1/4 := 
by
  -- The proof will be filled in later
  sorry

end prob_Z_l303_303462


namespace line_passes_through_fixed_point_l303_303670

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  mx + 3 * y + n = 0 → (x, y) = (1/2, -1/6) :=
by
  sorry

end line_passes_through_fixed_point_l303_303670


namespace cost_price_per_metre_l303_303917

theorem cost_price_per_metre (total_selling_price : ℕ) (total_metres : ℕ) (loss_per_metre : ℕ)
  (h1 : total_selling_price = 9000)
  (h2 : total_metres = 300)
  (h3 : loss_per_metre = 6) :
  (total_selling_price + (loss_per_metre * total_metres)) / total_metres = 36 :=
by
  sorry

end cost_price_per_metre_l303_303917


namespace sum_factorials_last_two_digits_l303_303311

/-- Prove that the last two digits of the sum of factorials from 1! to 50! is equal to 13,
    given that for any n ≥ 10, n! ends in at least two zeros. -/
theorem sum_factorials_last_two_digits :
  (∑ n in finset.range 50, (n!) % 100) % 100 = 13 := 
sorry

end sum_factorials_last_two_digits_l303_303311


namespace find_dot_AP_BC_l303_303847

-- Defining the lengths of the sides of the triangle.
def length_AB : ℝ := 13
def length_BC : ℝ := 14
def length_CA : ℝ := 15

-- Defining the provided dot product conditions at point P.
def dot_BP_CA : ℝ := 18
def dot_CP_BA : ℝ := 32

-- The target is to prove the final dot product.
theorem find_dot_AP_BC :
  ∃ (AP BC : ℝ), BC = 14 → dot_BP_CA = 18 → dot_CP_BA = 32 → (AP * BC = 14) :=
by
  -- proof goes here
  sorry

end find_dot_AP_BC_l303_303847


namespace value_of_expression_l303_303189

theorem value_of_expression : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end value_of_expression_l303_303189


namespace sin_cos_pi_12_eq_l303_303570

theorem sin_cos_pi_12_eq:
  (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = 1 / 4 :=
by
  sorry

end sin_cos_pi_12_eq_l303_303570


namespace rectangle_area_l303_303132

open Classical

noncomputable def point := {x : ℝ × ℝ // x.1 >= 0 ∧ x.2 >= 0}

structure Triangle :=
  (X Y Z : point)

structure Rectangle :=
  (P Q R S : point)

def height_from (t : Triangle) : ℝ :=
  8

def xz_length (t : Triangle) : ℝ :=
  15

def ps_on_xz (r : Rectangle) (t : Triangle) : Prop :=
  r.S.val.1 = r.P.val.1 ∧ r.S.val.1 = t.X.val.1 ∧ r.S.val.2 = 0 ∧ r.P.val.2 = 0

def pq_is_one_third_ps (r : Rectangle) : Prop :=
  dist r.P.1 r.Q.1 = (1/3) * dist r.P.1 r.S.1

theorem rectangle_area : ∀ (R : Rectangle) (T : Triangle),
  height_from T = 8 → xz_length T = 15 → ps_on_xz R T → pq_is_one_third_ps R →
  (dist R.P.1 R.Q.1) * (dist R.P.1 R.S.1) = 4800/169 :=
by
  intros
  sorry

end rectangle_area_l303_303132


namespace solve_for_x_l303_303538

theorem solve_for_x (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 :=
by sorry

end solve_for_x_l303_303538


namespace probability_no_shaded_rectangle_l303_303597

theorem probability_no_shaded_rectangle :
  let n := 2003
  let total_rectangles := ((n + 1) * (n + 1) * (n + 1) * (n + 1)) / 4
  let shaded_rectangles := ((n / 2 + 1) * (n / 2 + 1) * (n / 2 + 1) * (n / 2 + 1))
  let p := 1 - (shaded_rectangles / total_rectangles)
  p = (Rat.ofInt 1001) / (Rat.ofInt 2003) :=
by
  sorry

end probability_no_shaded_rectangle_l303_303597


namespace rationalize_and_divide_l303_303534

theorem rationalize_and_divide :
  (8 / Real.sqrt 8 / 2) = Real.sqrt 2 :=
by
  sorry

end rationalize_and_divide_l303_303534


namespace fill_cistern_l303_303894

noncomputable def pipe_fill_rate (min_to_fill : ℕ) : ℚ := 1 / min_to_fill

theorem fill_cistern 
  (p_rate : ℚ := pipe_fill_rate 12) 
  (q_rate : ℚ := pipe_fill_rate 15) 
  (combined_rate := p_rate + q_rate)
  (initial_fill_time : ℕ := 4) 
  (initial_fill := initial_fill_time * combined_rate)
  (remaining_fill := 1 - initial_fill)
  (total_time_after_turn_off : ℕ := (remaining_fill / q_rate).toNat) :
  total_time_after_turn_off = 6 := 
by
  -- Using given conditions to calculate the total_time_after_turn_off
  have calc_1 : p_rate = 1 / 12 := rfl
  have calc_2 : q_rate = 1 / 15 := rfl
  have calc_3 : combined_rate = 1 / 12 + 1 / 15 := by rw [calc_1, calc_2]
  have calc_4 : 1 / 12 = 5 / 60 := by norm_num
  have calc_5 : 1 / 15 = 4 / 60 := by norm_num
  have calc_6 : combined_rate = 9 / 60 := by rw [calc_3, calc_4, calc_5]; norm_num
  have calc_7 : initial_fill = 4 * (9 / 60) := by rw [calc_6]; norm_num
  have fill_frac : initial_fill = 3 / 5 := rfl
  have remain_frac : remaining_fill = 2 / 5 := by rw [fill_frac]; norm_num
  have calc_8 : (remaining_fill / q_rate).toNat = 6 := by
    rw [remain_frac, q_rate, div_div_eq_mul_div, inv_eq_one_div, div_eq_mul_one_div, <-
      inv_mul_cancel, Nat.inv_of_nat]
    norm_num
  exact calc_8

#eval fill_cistern  -- Running this line checks if the theorem holds correctly.

end fill_cistern_l303_303894


namespace star_polygon_net_of_pyramid_l303_303996

theorem star_polygon_net_of_pyramid (R r : ℝ) (h : R > r) : R > 2 * r :=
by
  sorry

end star_polygon_net_of_pyramid_l303_303996


namespace distance_AB_l303_303865

theorem distance_AB : 
  let A := -1
  let B := 2020
  |A - B| = 2021 := by
  sorry

end distance_AB_l303_303865


namespace shifted_linear_function_correct_l303_303146

def original_function (x : ℝ) : ℝ := 5 * x - 8
def shifted_function (x : ℝ) : ℝ := original_function x + 4

theorem shifted_linear_function_correct (x : ℝ) :
  shifted_function x = 5 * x - 4 :=
by
  sorry

end shifted_linear_function_correct_l303_303146


namespace infinitely_many_sum_of_squares_exceptions_l303_303425

-- Define the predicate for a number being expressible as a sum of two squares
def is_sum_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

-- Define the main theorem
theorem infinitely_many_sum_of_squares_exceptions : 
  ∃ f : ℕ → ℕ, (∀ k : ℕ, is_sum_of_squares (f k)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k - 1)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k + 1)) ∧ (∀ k1 k2 : ℕ, k1 ≠ k2 → f k1 ≠ f k2) :=
sorry

end infinitely_many_sum_of_squares_exceptions_l303_303425


namespace tan_theta_minus_pi_over_4_l303_303831

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over_4_l303_303831


namespace union_of_sets_l303_303078

open Set

variable (A B : Set ℝ)

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x ∧ x < 4}

theorem union_of_sets :
  A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 4} :=
sorry

end union_of_sets_l303_303078


namespace orange_gumdrops_after_replacement_l303_303181

noncomputable def total_gumdrops : ℕ :=
  100

noncomputable def initial_orange_gumdrops : ℕ :=
  10

noncomputable def initial_blue_gumdrops : ℕ :=
  40

noncomputable def replaced_blue_gumdrops : ℕ :=
  initial_blue_gumdrops / 3

theorem orange_gumdrops_after_replacement : 
  (initial_orange_gumdrops + replaced_blue_gumdrops) = 23 :=
by
  sorry

end orange_gumdrops_after_replacement_l303_303181


namespace total_blocks_fell_l303_303976

-- Definitions based on the conditions
def first_stack_height := 7
def second_stack_height := first_stack_height + 5
def third_stack_height := second_stack_height + 7

def first_stack_fallen_blocks := first_stack_height  -- All blocks fell down
def second_stack_fallen_blocks := second_stack_height - 2  -- 2 blocks left standing
def third_stack_fallen_blocks := third_stack_height - 3  -- 3 blocks left standing

-- Total fallen blocks
def total_fallen_blocks := first_stack_fallen_blocks + second_stack_fallen_blocks + third_stack_fallen_blocks

-- Theorem to prove the total number of fallen blocks
theorem total_blocks_fell : total_fallen_blocks = 33 :=
by
  -- Proof omitted, statement given as required
  sorry

end total_blocks_fell_l303_303976


namespace find_d_q_l303_303372

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def b_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  b1 * q^(n - 1)

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) / 2) * d

-- Sum of the first n terms of a geometric sequence
noncomputable def T_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  if q = 1 then n * b1
  else b1 * (1 - q^n) / (1 - q)

theorem find_d_q (a1 b1 d q : ℕ) (h1 : ∀ n : ℕ, n > 0 →
  n^2 * (T_n b1 q n + 1) = 2^n * S_n a1 d n) : d = 2 ∧ q = 2 :=
by
  sorry

end find_d_q_l303_303372


namespace place_value_diff_7669_l303_303170

theorem place_value_diff_7669 :
  let a := 6 * 10
  let b := 6 * 100
  b - a = 540 :=
by
  let a := 6 * 10
  let b := 6 * 100
  have h : b - a = 540 := by sorry
  exact h

end place_value_diff_7669_l303_303170


namespace simplify_rationalize_denominator_l303_303280

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l303_303280


namespace cost_price_of_each_watch_l303_303738

-- Define the given conditions.
def sold_at_loss (C : ℝ) := 0.925 * C
def total_transaction_price (C : ℝ) := 3 * C * 1.053
def sold_for_more (C : ℝ) := 0.925 * C + 265

-- State the theorem to prove the cost price of each watch.
theorem cost_price_of_each_watch (C : ℝ) :
  3 * sold_for_more C = total_transaction_price C → C = 2070.31 :=
by
  intros h
  sorry

end cost_price_of_each_watch_l303_303738


namespace ratio_result_l303_303366

theorem ratio_result (p q r s : ℚ) 
(h1 : p / q = 2) 
(h2 : q / r = 4 / 5) 
(h3 : r / s = 3) : 
  s / p = 5 / 24 :=
sorry

end ratio_result_l303_303366


namespace magic_triangle_max_sum_l303_303516

theorem magic_triangle_max_sum :
  ∃ (a b c d e f : ℕ), ((a = 5 ∨ a = 6 ∨ a = 7 ∨ a = 8 ∨ a = 9 ∨ a = 10) ∧
                        (b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8 ∨ b = 9 ∨ b = 10) ∧
                        (c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10) ∧
                        (d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9 ∨ d = 10) ∧
                        (e = 5 ∨ e = 6 ∨ e = 7 ∨ e = 8 ∨ e = 9 ∨ e = 10) ∧
                        (f = 5 ∨ f = 6 ∨ f = 7 ∨ f = 8 ∨ f = 9 ∨ f = 10) ∧
                        (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
                        (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
                        (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
                        (d ≠ e) ∧ (d ≠ f) ∧
                        (e ≠ f) ∧
                        (a + b + c = 24) ∧ (c + d + e = 24) ∧ (e + f + a = 24)) :=
sorry

end magic_triangle_max_sum_l303_303516


namespace sacks_filled_l303_303929

theorem sacks_filled (pieces_per_sack : ℕ) (total_pieces : ℕ) (h1 : pieces_per_sack = 20) (h2 : total_pieces = 80) : (total_pieces / pieces_per_sack) = 4 :=
by {
  sorry
}

end sacks_filled_l303_303929


namespace hunting_dog_catches_fox_l303_303336

theorem hunting_dog_catches_fox :
  ∀ (V_1 V_2 : ℝ) (t : ℝ),
  V_1 / V_2 = 10 ∧ 
  t * V_2 = (10 / (V_2) + t) →
  (V_1 * t) = 100 / 9 :=
by
  intros V_1 V_2 t h
  sorry

end hunting_dog_catches_fox_l303_303336


namespace box_volume_l303_303770

theorem box_volume
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12)
  (h4 : l = h + 1) :
  l * w * h = 120 := 
sorry

end box_volume_l303_303770


namespace minimize_transfers_l303_303728

-- Define the initial number of pieces in each supermarket
def pieces_in_A := 15
def pieces_in_B := 7
def pieces_in_C := 11
def pieces_in_D := 3
def pieces_in_E := 14

-- Define the target number of pieces in each supermarket after transfers
def target_pieces := 10

-- Define a function to compute the total number of pieces
def total_pieces := pieces_in_A + pieces_in_B + pieces_in_C + pieces_in_D + pieces_in_E

-- Define the minimum number of transfers needed
def min_transfers := 12

-- The main theorem: proving that the minimum number of transfers is 12
theorem minimize_transfers : 
  total_pieces = 5 * target_pieces → 
  ∃ (transfers : ℕ), transfers = min_transfers :=
by
  -- This represents the proof section, we leave it as sorry
  sorry

end minimize_transfers_l303_303728


namespace sum_of_arithmetic_sequence_9_terms_l303_303082

-- Define the odd function and its properties
variables {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = -f (x)) 
          (h2 : ∀ x y, x < y → f x < f y)

-- Define the shifted function g
noncomputable def g (x : ℝ) := f (x - 5)

-- Define the arithmetic sequence with non-zero common difference
variables {a : ℕ → ℝ} (d : ℝ) (h3 : d ≠ 0) 
          (h4 : ∀ n, a (n + 1) = a n + d)

-- Condition given by the problem
variable (h5 : g (a 1) + g (a 9) = 0)

-- Proof obligation
theorem sum_of_arithmetic_sequence_9_terms :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 45 :=
sorry

end sum_of_arithmetic_sequence_9_terms_l303_303082


namespace weight_of_new_person_l303_303875

/-- The average weight of 10 persons increases by 7.2 kg when a new person
replaces one who weighs 65 kg. Prove that the weight of the new person is 137 kg. -/
theorem weight_of_new_person (W_new : ℝ) (W_old : ℝ) (n : ℝ) (increase : ℝ) 
  (h1 : W_old = 65) (h2 : n = 10) (h3 : increase = 7.2) 
  (h4 : W_new = W_old + n * increase) : W_new = 137 := 
by
  -- proof to be done later
  sorry

end weight_of_new_person_l303_303875


namespace problem_proof_l303_303447

-- Define I, J, and K respectively to be 9^20, 3^41, 3
def I : ℕ := 9^20
def J : ℕ := 3^41
def K : ℕ := 3

theorem problem_proof : I + I + I = J := by
  -- Lean structure placeholder
  sorry

end problem_proof_l303_303447


namespace calculate_value_l303_303351

theorem calculate_value : (245^2 - 225^2) / 20 = 470 :=
by
  sorry

end calculate_value_l303_303351


namespace solve_for_x_l303_303558

noncomputable def equation (x : ℝ) := (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2

theorem solve_for_x (h : ∀ x, x ≠ 3) : equation (-7 / 6) :=
by
  sorry

end solve_for_x_l303_303558


namespace chord_length_l303_303352

-- Define radii of the circles
def r1 : ℝ := 5
def r2 : ℝ := 12
def r3 : ℝ := r1 + r2

-- Define the centers of the circles
variable (O1 O2 O3 : ℝ)

-- Define the points of tangency and foot of the perpendicular
def T1 : ℝ := O1 + r1
def T2 : ℝ := O2 + r2
def T : ℝ := O3 - r3

-- Given the conditions
theorem chord_length (m n p : ℤ) : 
  (∃ (C1 C2 C3 : ℝ) (tangent1 tangent2 : ℝ),
    C1 = r1 ∧ C2 = r2 ∧ C3 = r3 ∧
    -- Externally tangent: distance between centers of C1 and C2 is r1 + r2
    dist O1 O2 = r1 + r2 ∧
    -- Internally tangent: both C1 and C2 are tangent to C3
    dist O1 O3 = r3 - r1 ∧
    dist O2 O3 = r3 - r2 ∧
    -- The chord in C3 is a common external tangent to C1 and C2
    tangent1 = O3 + ((O1 * O2) - (O1 * O3)) / r1 ∧
    tangent2 = O3 + ((O2 * O1) - (O2 * O3)) / r2 ∧
    m = 10 ∧ n = 546 ∧ p = 7 ∧
    m + n + p = 563)
  := sorry

end chord_length_l303_303352


namespace david_on_sixth_platform_l303_303003

theorem david_on_sixth_platform 
  (h₁ : walter_initial_fall = 4)
  (h₂ : walter_additional_fall = 3 * walter_initial_fall)
  (h₃ : total_fall = walter_initial_fall + walter_additional_fall)
  (h₄ : total_platforms = 8)
  (h₅ : total_height = total_fall)
  (h₆ : platform_height = total_height / total_platforms)
  (h₇ : david_fall_distance = walter_initial_fall)
  : (total_height - david_fall_distance) / platform_height = 6 := 
  by sorry

end david_on_sixth_platform_l303_303003


namespace product_sum_even_l303_303700

theorem product_sum_even (m n : ℤ) : Even (m * n * (m + n)) := 
sorry

end product_sum_even_l303_303700


namespace min_value_expression_l303_303073

theorem min_value_expression (x y : ℝ) (h : y^2 - 2*x + 4 = 0) : 
  ∃ z : ℝ, z = x^2 + y^2 + 2*x ∧ z = -8 :=
by
  sorry

end min_value_expression_l303_303073


namespace gcd_2873_1349_gcd_4562_275_l303_303479

theorem gcd_2873_1349 : Nat.gcd 2873 1349 = 1 := 
sorry

theorem gcd_4562_275 : Nat.gcd 4562 275 = 1 := 
sorry

end gcd_2873_1349_gcd_4562_275_l303_303479


namespace max_load_truck_l303_303333

theorem max_load_truck (bag_weight : ℕ) (num_bags : ℕ) (remaining_load : ℕ) 
  (h1 : bag_weight = 8) (h2 : num_bags = 100) (h3 : remaining_load = 100) : 
  bag_weight * num_bags + remaining_load = 900 :=
by
  -- We leave the proof step intentionally, as per instructions.
  sorry

end max_load_truck_l303_303333


namespace speed_of_stream_l303_303160

variable (D : ℝ) -- The distance rowed in both directions
variable (vs : ℝ) -- The speed of the stream
variable (Vb : ℝ := 78) -- The speed of the boat in still water

theorem speed_of_stream (h : (D / (Vb - vs) = 2 * (D / (Vb + vs)))) : vs = 26 := by
    sorry

end speed_of_stream_l303_303160


namespace largest_divisor_of_composite_sum_and_square_l303_303211

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_divisor_of_composite_sum_and_square (n : ℕ) (h : is_composite n) : ( ∃ (k : ℕ), ∀ n : ℕ, is_composite n → ∃ m : ℕ, n + n^2 = m * k) → k = 2 :=
by
  sorry

end largest_divisor_of_composite_sum_and_square_l303_303211


namespace black_grid_probability_l303_303459

theorem black_grid_probability : 
  (let n := 4
   let unit_squares := n * n
   let pairs := unit_squares / 2
   let probability_each_pair := (1:ℝ) / 4
   let total_probability := probability_each_pair ^ pairs
   total_probability = (1:ℝ) / 65536) :=
by
  let n := 4
  let unit_squares := n * n
  let pairs := unit_squares / 2
  let probability_each_pair := (1:ℝ) / 4
  let total_probability := probability_each_pair ^ pairs
  sorry

end black_grid_probability_l303_303459


namespace condition_for_ellipse_l303_303456

-- Definition of the problem conditions
def is_ellipse (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (5 - m > 0) ∧ (m - 2 ≠ 5 - m)

noncomputable def necessary_not_sufficient_condition (m : ℝ) : Prop :=
  (2 < m) ∧ (m < 5)

-- The theorem to be proved
theorem condition_for_ellipse (m : ℝ) : 
  (necessary_not_sufficient_condition m) → (is_ellipse m) :=
by
  -- proof to be written here
  sorry

end condition_for_ellipse_l303_303456


namespace simplify_rationalize_denominator_l303_303279

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l303_303279


namespace solve_equation_l303_303553

theorem solve_equation (x : ℚ) : 
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 → 
  x ≠ 3 → 
  x ≠ -2 → 
  x = -7 / 6 :=
by
  sorry

end solve_equation_l303_303553


namespace number_in_scientific_notation_l303_303848

/-- Condition: A constant corresponding to the number we are converting. -/
def number : ℕ := 9000000000

/-- Condition: The correct answer we want to prove. -/
def correct_answer : ℕ := 9 * 10^9

/-- Proof Problem: Prove that the number equals the correct_answer when expressed in scientific notation. -/
theorem number_in_scientific_notation : number = correct_answer := by
  sorry

end number_in_scientific_notation_l303_303848


namespace sin_double_angle_value_l303_303070

open Real

theorem sin_double_angle_value (x : ℝ) 
  (h1 : sin (x + π/3) * cos (x - π/6) + sin (x - π/6) * cos (x + π/3) = 5 / 13)
  (h2 : -π/3 ≤ x ∧ x ≤ π/6) :
  sin (2 * x) = (5 * sqrt 3 - 12) / 26 :=
by
  sorry

end sin_double_angle_value_l303_303070


namespace book_arrangement_count_l303_303961

-- Define the conditions
def total_books : ℕ := 6
def identical_books : ℕ := 3
def different_books : ℕ := total_books - identical_books

-- Prove the number of arrangements
theorem book_arrangement_count : (total_books.factorial / identical_books.factorial) = 120 := by
  sorry

end book_arrangement_count_l303_303961


namespace solve_for_y_l303_303798

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem solve_for_y (y : ℝ) : star 2 y = 10 → y = 0 := by
  intro h
  sorry

end solve_for_y_l303_303798


namespace pow_neg_cubed_squared_l303_303480

variable (a : ℝ)

theorem pow_neg_cubed_squared : 
  (-a^3)^2 = a^6 := 
by 
  sorry

end pow_neg_cubed_squared_l303_303480


namespace total_players_on_ground_l303_303103

def cricket_players : ℕ := 15
def hockey_players : ℕ := 12
def football_players : ℕ := 13
def softball_players : ℕ := 15

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 55 := 
by
  sorry

end total_players_on_ground_l303_303103


namespace find_points_l303_303128

theorem find_points :
  ∀ (x₀ : ℝ), (∃ (x₀ : ℝ), (M : ℝ×ℝ) → M = (x₀, -13/6) ∧ (∃ (k₁ k₂ : ℝ),
    k₁ + k₂ = 2 * x₀ ∧ k₁ * k₂ = -13/3 ∧
    (k₂ - k₁) / (1 + k₂ * k₁)) = sqrt 3) →
      (x₀ = 2 ∨ x₀ = -2) :=
by
  sorry

end find_points_l303_303128


namespace find_other_number_l303_303887

theorem find_other_number (x : ℕ) (h : x + 42 = 96) : x = 54 :=
by {
  sorry
}

end find_other_number_l303_303887


namespace black_cards_taken_out_l303_303428

theorem black_cards_taken_out (total_black_cards remaining_black_cards : ℕ)
  (h1 : total_black_cards = 26) (h2 : remaining_black_cards = 21) :
  total_black_cards - remaining_black_cards = 5 :=
by
  sorry

end black_cards_taken_out_l303_303428


namespace value_of_x_l303_303513

theorem value_of_x : 
  ∀ (x y z : ℕ), 
  (x = y / 3) ∧ 
  (y = z / 6) ∧ 
  (z = 72) → 
  x = 4 :=
by
  intros x y z h
  have h1 : y = z / 6 := h.2.1
  have h2 : z = 72 := h.2.2
  have h3 : x = y / 3 := h.1
  sorry

end value_of_x_l303_303513


namespace solve_for_x_l303_303541

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l303_303541


namespace no_infinite_non_constant_arithmetic_progression_with_powers_l303_303941

theorem no_infinite_non_constant_arithmetic_progression_with_powers (a b : ℕ) (b_ge_2 : b ≥ 2) : 
  ¬ ∃ (f : ℕ → ℕ) (d : ℕ), (∀ n : ℕ, f n = (a^(b + n*d)) ∧ b ≥ 2) := sorry

end no_infinite_non_constant_arithmetic_progression_with_powers_l303_303941


namespace inequality_system_solution_l303_303721

theorem inequality_system_solution (x : ℝ) : 
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 :=
by
  sorry

end inequality_system_solution_l303_303721


namespace mutually_exclusive_event_of_hitting_target_at_least_once_l303_303464

-- Definitions from conditions
def two_shots_fired : Prop := true

def complementary_events (E F : Prop) : Prop :=
  E ∨ F ∧ ¬(E ∧ F)

def hitting_target_at_least_once : Prop := true -- Placeholder for the event of hitting at least one target
def both_shots_miss : Prop := true              -- Placeholder for the event that both shots miss

-- Statement to prove
theorem mutually_exclusive_event_of_hitting_target_at_least_once
  (h1 : two_shots_fired)
  (h2 : complementary_events hitting_target_at_least_once both_shots_miss) :
  hitting_target_at_least_once = ¬both_shots_miss := 
sorry

end mutually_exclusive_event_of_hitting_target_at_least_once_l303_303464


namespace value_of_a_minus_b_l303_303962

theorem value_of_a_minus_b (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : ab = 6) : a - b = 5 ∨ a - b = -5 := 
by
  sorry

end value_of_a_minus_b_l303_303962


namespace inequality_problem_l303_303652

theorem inequality_problem
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_sum : a + b + c ≤ 3) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by sorry

end inequality_problem_l303_303652


namespace positive_real_triangle_inequality_l303_303760

theorem positive_real_triangle_inequality
    (a b c : ℝ)
    (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h : 5 * a * b * c > a^3 + b^3 + c^3) :
    a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end positive_real_triangle_inequality_l303_303760


namespace greatest_number_in_consecutive_multiples_l303_303592

theorem greatest_number_in_consecutive_multiples (s : Set ℕ) (h₁ : ∃ m : ℕ, s = {n | ∃ k < 100, n = 8 * (m + k)} ∧ m = 14) :
  (∃ n ∈ s, ∀ x ∈ s, x ≤ n) →
  ∃ n ∈ s, n = 904 :=
by
  sorry

end greatest_number_in_consecutive_multiples_l303_303592


namespace right_triangle_condition_l303_303749

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end right_triangle_condition_l303_303749


namespace find_number_l303_303021

theorem find_number (N : ℕ) (h1 : ∃ k : ℤ, N = 13 * k + 11) (h2 : ∃ m : ℤ, N = 17 * m + 9) : N = 89 := 
sorry

end find_number_l303_303021


namespace sandy_initial_payment_l303_303134

variable (P : ℝ) 

theorem sandy_initial_payment
  (h1 : (1.2 : ℝ) * (P + 200) = 1200) :
  P = 800 :=
by
  -- Proof goes here
  sorry

end sandy_initial_payment_l303_303134


namespace buns_cost_eq_1_50_l303_303414

noncomputable def meat_cost : ℝ := 2 * 3.50
noncomputable def tomato_cost : ℝ := 1.5 * 2.00
noncomputable def pickles_cost : ℝ := 2.50 - 1.00
noncomputable def lettuce_cost : ℝ := 1.00
noncomputable def total_other_items_cost : ℝ := meat_cost + tomato_cost + pickles_cost + lettuce_cost
noncomputable def total_amount_spent : ℝ := 20.00 - 6.00
noncomputable def buns_cost : ℝ := total_amount_spent - total_other_items_cost

theorem buns_cost_eq_1_50 : buns_cost = 1.50 := by
  sorry

end buns_cost_eq_1_50_l303_303414


namespace count_numbers_without_1_or_2_l303_303960

/-- The number of whole numbers between 1 and 2000 that do not contain the digits 1 or 2 is 511. -/
theorem count_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 511 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2000 →
      ¬ (∃ d : ℕ, (k.digits 10).contains d ∧ (d = 1 ∨ d = 2)) → n = 511) :=
sorry

end count_numbers_without_1_or_2_l303_303960


namespace relay_team_orders_l303_303250

noncomputable def jordan_relay_orders : Nat :=
  let friends := [1, 2, 3] -- Differentiate friends; let's represent A by 1, B by 2, C by 3
  let choices_for_jordan_third := 2 -- Ways if Jordan runs third
  let choices_for_jordan_fourth := 2 -- Ways if Jordan runs fourth
  choices_for_jordan_third + choices_for_jordan_fourth

theorem relay_team_orders :
  jordan_relay_orders = 4 :=
by
  sorry

end relay_team_orders_l303_303250


namespace fifi_pink_hangers_l303_303680

theorem fifi_pink_hangers :
  ∀ (g b y p : ℕ), 
  g = 4 →
  b = g - 1 →
  y = b - 1 →
  16 = g + b + y + p →
  p = 7 :=
by
  intros
  sorry

end fifi_pink_hangers_l303_303680


namespace counting_unit_difference_l303_303595

-- Definitions based on conditions
def magnitude_equality : Prop := 75 = 75.0
def counting_unit_75 : Nat := 1
def counting_unit_75_0 : Nat := 1 / 10

-- Proof problem stating that 75 and 75.0 do not have the same counting units.
theorem counting_unit_difference : 
  ¬ (counting_unit_75 = counting_unit_75_0) :=
by sorry

end counting_unit_difference_l303_303595


namespace ways_to_partition_6_into_4_boxes_l303_303666

theorem ways_to_partition_6_into_4_boxes : 
  ∃ (s : Finset (Finset ℕ)), (∀ (x ∈ s), ∃ (a b c d : ℕ), x = {a, b, c, d} ∧ a + b + c + d = 6) ∧ s.card = 9 :=
sorry

end ways_to_partition_6_into_4_boxes_l303_303666


namespace seven_people_arrangement_l303_303426

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def perm (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem seven_people_arrangement : 
  (perm 5 5) * (perm 6 2) = 3600 := by
sorry

end seven_people_arrangement_l303_303426


namespace problem1_problem2_l303_303817

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + b|
noncomputable def g (x a b : ℝ) : ℝ := -x^2 - a*x - b

-- Problem 1: Prove that a + b = 3
theorem problem1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : ∀ x, f x a b ≤ 3) : a + b = 3 := 
sorry

-- Problem 2: Prove that 1/2 < a < 3
theorem problem2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 3) 
  (h₃ : ∀ x, x ≥ a → g x a b < f x a b) : 1/2 < a ∧ a < 3 := 
sorry

end problem1_problem2_l303_303817


namespace solve_quadratic_eq_l303_303718

theorem solve_quadratic_eq {x : ℝ} (h : x^2 - 5*x + 6 = 0) : x = 2 ∨ x = 3 :=
sorry

end solve_quadratic_eq_l303_303718


namespace find_m_l303_303237

def setA (m : ℝ) : Set ℝ := {1, m - 2}
def setB : Set ℝ := {2}

theorem find_m (m : ℝ) (H : setA m ∩ setB = {2}) : m = 4 :=
by
  sorry

end find_m_l303_303237


namespace exists_real_ge_3_l303_303174

-- Definition of the existential proposition
theorem exists_real_ge_3 : ∃ x : ℝ, x ≥ 3 :=
sorry

end exists_real_ge_3_l303_303174


namespace median_eq_range_le_l303_303648

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l303_303648


namespace school_accomodation_proof_l303_303186

theorem school_accomodation_proof
  (total_classrooms : ℕ) 
  (fraction_classrooms_45 : ℕ) 
  (fraction_classrooms_38 : ℕ)
  (fraction_classrooms_32 : ℕ)
  (fraction_classrooms_25 : ℕ)
  (desks_45 : ℕ)
  (desks_38 : ℕ)
  (desks_32 : ℕ)
  (desks_25 : ℕ)
  (student_capacity_limit : ℕ) :
  total_classrooms = 50 ->
  fraction_classrooms_45 = (3 / 10) * total_classrooms -> 
  fraction_classrooms_38 = (1 / 4) * total_classrooms -> 
  fraction_classrooms_32 = (1 / 5) * total_classrooms -> 
  fraction_classrooms_25 = (total_classrooms - fraction_classrooms_45 - fraction_classrooms_38 - fraction_classrooms_32) ->
  desks_45 = 15 * 45 -> 
  desks_38 = 12 * 38 -> 
  desks_32 = 10 * 32 -> 
  desks_25 = fraction_classrooms_25 * 25 -> 
  student_capacity_limit = 1800 -> 
  fraction_classrooms_45 * 45 +
  fraction_classrooms_38 * 38 +
  fraction_classrooms_32 * 32 + 
  fraction_classrooms_25 * 25 = 1776 + sorry
  :=
sorry

end school_accomodation_proof_l303_303186


namespace completing_the_square_l303_303010

theorem completing_the_square (x : ℝ) : 
  x^2 - 2 * x = 9 → (x - 1)^2 = 10 :=
by
  intro h
  sorry

end completing_the_square_l303_303010


namespace number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l303_303243

def number_of_seatings (n : ℕ) : ℕ := Nat.factorial n

theorem number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other :
  let k := 2      -- Kolya and Olya as a unit
  let remaining := 3 -- The remaining people
  let pairs := 4 -- Pairs of seats that Kolya and Olya can take
  let arrangements_kolya_olya := pairs * 2 -- Each pair can have Kolya and Olya in 2 arrangements
  let arrangements_remaining := number_of_seatings remaining 
  arrangements_kolya_olya * arrangements_remaining = 48 := by
{
  -- This would be the location for the proof implementation
  sorry
}

end number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l303_303243


namespace append_five_new_number_l303_303396

theorem append_five_new_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) : 
  10 * (10 * t + u) + 5 = 100 * t + 10 * u + 5 :=
by sorry

end append_five_new_number_l303_303396


namespace sponge_cake_eggs_l303_303490

theorem sponge_cake_eggs (eggs flour sugar total desiredCakeMass : ℕ) 
  (h_recipe : eggs = 300) 
  (h_flour : flour = 120)
  (h_sugar : sugar = 100) 
  (h_total : total = 520) 
  (h_desiredMass : desiredCakeMass = 2600) :
  (eggs * desiredCakeMass / total) = 1500 := by
  sorry

end sponge_cake_eggs_l303_303490


namespace sufficient_conditions_for_equation_l303_303936

theorem sufficient_conditions_for_equation 
  (a b c : ℤ) :
  (a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c) →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_conditions_for_equation_l303_303936


namespace n_divisible_by_6_l303_303429

open Int -- Open integer namespace for convenience

theorem n_divisible_by_6 (m n : ℤ)
    (h1 : ∃ (a b : ℤ), a + b = -m ∧ a * b = -n)
    (h2 : ∃ (c d : ℤ), c + d = m ∧ c * d = n) :
    6 ∣ n := 
sorry

end n_divisible_by_6_l303_303429


namespace x_sq_plus_inv_sq_l303_303963

theorem x_sq_plus_inv_sq (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
  sorry

end x_sq_plus_inv_sq_l303_303963


namespace median_equality_and_range_inequality_l303_303638

theorem median_equality_and_range_inequality
  (x : Fin 6 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i ≠ x j)
  (hx1_min : ∀ i, x 0 ≤ x i)
  (hx6_max : ∀ i, x i ≤ x 5) :
  median ({ x 1, x 2, x 3, x 4 } : Finset ℝ) = median ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) ∧
  range ({ x 1, x 2, x 3, x 4 } : Finset ℝ) ≤ range ({ x 0, x 1, x 2, x 3, x 4, x 5 } : Finset ℝ) := 
sorry

end median_equality_and_range_inequality_l303_303638


namespace scout_troop_profit_calc_l303_303605

theorem scout_troop_profit_calc
  (candy_bars : ℕ := 1200)
  (purchase_rate : ℚ := 3/6)
  (sell_rate : ℚ := 2/3) :
  (candy_bars * sell_rate - candy_bars * purchase_rate) = 200 :=
by
  sorry

end scout_troop_profit_calc_l303_303605


namespace proof_fraction_l303_303164

def find_fraction (x : ℝ) : Prop :=
  (2 / 9) * x = 10 → (2 / 5) * x = 18

-- Optional, you can define x based on the condition:
noncomputable def certain_number : ℝ := 10 * (9 / 2)

theorem proof_fraction :
  find_fraction certain_number :=
by
  intro h
  sorry

end proof_fraction_l303_303164


namespace min_value_arithmetic_sequence_l303_303219

theorem min_value_arithmetic_sequence (d : ℝ) (n : ℕ) (hd : d ≠ 0) (a1 : ℝ) (ha1 : a1 = 1)
(geo : (1 + 2 * d)^2 = 1 + 12 * d) (Sn : ℝ) (hSn : Sn = n^2) (an : ℝ) (han : an = 2 * n - 1) :
  ∀ (n : ℕ), n > 0 → (2 * Sn + 8) / (an + 3) ≥ 5 / 2 :=
by sorry

end min_value_arithmetic_sequence_l303_303219


namespace hexagon_coloring_count_l303_303626

-- Defining the conditions
def has7Colors : Type := Fin 7

-- Hexagon vertices
inductive Vertex
| A | B | C | D | E | F

-- Adjacent vertices
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B => true
| Vertex.B, Vertex.C => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.E => true
| Vertex.E, Vertex.F => true
| Vertex.F, Vertex.A => true
| _, _ => false

-- Non-adjacent vertices (diagonals)
def non_adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.D => true
| Vertex.B, Vertex.E => true
| Vertex.C, Vertex.E => true
| Vertex.C, Vertex.F => true
| Vertex.D, Vertex.F => true
| Vertex.E, Vertex.A => true
| Vertex.F, Vertex.A => true
| Vertex.F, Vertex.B => true
| _, _ => false

-- Coloring function
def valid_coloring (coloring : Vertex → has7Colors) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2, non_adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2 v3, adjacent v1 v2 → adjacent v2 v3 → adjacent v1 v3 → coloring v1 ≠ coloring v3)

noncomputable def count_valid_colorings : Nat :=
  -- This is a placeholder for the count function
  sorry

theorem hexagon_coloring_count : count_valid_colorings = 21000 := 
  sorry

end hexagon_coloring_count_l303_303626


namespace final_weights_are_correct_l303_303141

-- Definitions of initial weights and reduction percentages per week
def initial_weight_A : ℝ := 300
def initial_weight_B : ℝ := 450
def initial_weight_C : ℝ := 600
def initial_weight_D : ℝ := 750

def reduction_A_week1 : ℝ := 0.20 * initial_weight_A
def reduction_B_week1 : ℝ := 0.15 * initial_weight_B
def reduction_C_week1 : ℝ := 0.30 * initial_weight_C
def reduction_D_week1 : ℝ := 0.25 * initial_weight_D

def weight_A_after_week1 : ℝ := initial_weight_A - reduction_A_week1
def weight_B_after_week1 : ℝ := initial_weight_B - reduction_B_week1
def weight_C_after_week1 : ℝ := initial_weight_C - reduction_C_week1
def weight_D_after_week1 : ℝ := initial_weight_D - reduction_D_week1

def reduction_A_week2 : ℝ := 0.25 * weight_A_after_week1
def reduction_B_week2 : ℝ := 0.30 * weight_B_after_week1
def reduction_C_week2 : ℝ := 0.10 * weight_C_after_week1
def reduction_D_week2 : ℝ := 0.20 * weight_D_after_week1

def weight_A_after_week2 : ℝ := weight_A_after_week1 - reduction_A_week2
def weight_B_after_week2 : ℝ := weight_B_after_week1 - reduction_B_week2
def weight_C_after_week2 : ℝ := weight_C_after_week1 - reduction_C_week2
def weight_D_after_week2 : ℝ := weight_D_after_week1 - reduction_D_week2

def reduction_A_week3 : ℝ := 0.15 * weight_A_after_week2
def reduction_B_week3 : ℝ := 0.10 * weight_B_after_week2
def reduction_C_week3 : ℝ := 0.20 * weight_C_after_week2
def reduction_D_week3 : ℝ := 0.30 * weight_D_after_week2

def weight_A_after_week3 : ℝ := weight_A_after_week2 - reduction_A_week3
def weight_B_after_week3 : ℝ := weight_B_after_week2 - reduction_B_week3
def weight_C_after_week3 : ℝ := weight_C_after_week2 - reduction_C_week3
def weight_D_after_week3 : ℝ := weight_D_after_week2 - reduction_D_week3

def reduction_A_week4 : ℝ := 0.10 * weight_A_after_week3
def reduction_B_week4 : ℝ := 0.20 * weight_B_after_week3
def reduction_C_week4 : ℝ := 0.25 * weight_C_after_week3
def reduction_D_week4 : ℝ := 0.15 * weight_D_after_week3

def final_weight_A : ℝ := weight_A_after_week3 - reduction_A_week4
def final_weight_B : ℝ := weight_B_after_week3 - reduction_B_week4
def final_weight_C : ℝ := weight_C_after_week3 - reduction_C_week4
def final_weight_D : ℝ := weight_D_after_week3 - reduction_D_week4

theorem final_weights_are_correct :
  final_weight_A = 137.7 ∧ 
  final_weight_B = 192.78 ∧ 
  final_weight_C = 226.8 ∧ 
  final_weight_D = 267.75 :=
by
  unfold final_weight_A final_weight_B final_weight_C final_weight_D
  sorry

end final_weights_are_correct_l303_303141


namespace bubble_pass_probability_l303_303795

-- Define the conditions and question
variable (s : Fin 35 → ℝ)
variable (distinct : ∀ i ≠ j, s i ≠ s j)

-- Define the event of a single bubble pass
-- For simplicity, we do not model the complete bubble pass algorithm,
-- but define what needs to be shown: the probability calculation outcome.

theorem bubble_pass_probability (p q : ℕ) (h : p / q = 1 / 1650 ∧ Int.gcd p q = 1) :
  p + q = 1651 :=
by
  sorry

end bubble_pass_probability_l303_303795


namespace units_digit_of_1583_pow_1246_l303_303830

theorem units_digit_of_1583_pow_1246 : 
  (1583^1246) % 10 = 9 := 
sorry

end units_digit_of_1583_pow_1246_l303_303830


namespace average_price_per_pen_l303_303458

def total_cost_pens_pencils : ℤ := 690
def number_of_pencils : ℕ := 75
def price_per_pencil : ℤ := 2
def number_of_pens : ℕ := 30

theorem average_price_per_pen :
  (total_cost_pens_pencils - number_of_pencils * price_per_pencil) / number_of_pens = 18 :=
by
  sorry

end average_price_per_pen_l303_303458


namespace find_k_value_l303_303658

theorem find_k_value (k : ℝ) (x : ℝ) :
  -x^2 - (k + 12) * x - 8 = -(x - 2) * (x - 4) → k = -18 :=
by
  intro h
  sorry

end find_k_value_l303_303658


namespace sin_cos_identity_l303_303094

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 := 
by
  sorry

end sin_cos_identity_l303_303094


namespace tom_watches_movies_total_duration_l303_303845

-- Define the running times for each movie
def M := 120
def A := M - 30
def B := A + 10
def D := 2 * B - 20

-- Define the number of times Tom watches each movie
def watch_B := 2
def watch_A := 3
def watch_M := 1
def watch_D := 4

-- Calculate the total time spent watching each movie
def total_time_B := watch_B * B
def total_time_A := watch_A * A
def total_time_M := watch_M * M
def total_time_D := watch_D * D

-- Calculate the total duration Tom spends watching these movies in a week
def total_duration := total_time_B + total_time_A + total_time_M + total_time_D

-- The statement to prove
theorem tom_watches_movies_total_duration :
  total_duration = 1310 := 
by
  sorry

end tom_watches_movies_total_duration_l303_303845


namespace min_students_with_blue_eyes_and_backpack_l303_303514

theorem min_students_with_blue_eyes_and_backpack :
  ∀ (students : Finset ℕ), 
  (∀ s, s ∈ students → s = 1) →
  ∃ A B : Finset ℕ, 
    A.card = 18 ∧ B.card = 24 ∧ students.card = 35 ∧ 
    (A ∩ B).card ≥ 7 :=
by
  sorry

end min_students_with_blue_eyes_and_backpack_l303_303514


namespace entry_exit_options_l303_303296

theorem entry_exit_options :
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  (total_gates * total_gates = 49) :=
by {
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  show total_gates * total_gates = 49
  sorry
}

end entry_exit_options_l303_303296


namespace sum_of_five_consecutive_even_integers_l303_303897

theorem sum_of_five_consecutive_even_integers (a : ℤ) (h : a + (a + 4) = 150) :
  a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385 :=
by
  sorry

end sum_of_five_consecutive_even_integers_l303_303897


namespace original_number_of_people_l303_303442

-- Defining the conditions
variable (n : ℕ) -- number of people originally
variable (total_cost : ℕ := 375)
variable (equal_cost_split : n > 0 ∧ total_cost = 375) -- total cost is $375 and n > 0
variable (cost_condition : 375 / n + 50 = 375 / 5)

-- The proof statement
theorem original_number_of_people (h1 : total_cost = 375) (h2 : 375 / n + 50 = 375 / 5) : n = 15 :=
by
  sorry

end original_number_of_people_l303_303442


namespace tank_length_is_25_l303_303188

noncomputable def cost_to_paise (cost_in_rupees : ℕ) : ℕ :=
  cost_in_rupees * 100

noncomputable def total_area_plastered (total_cost_in_paise : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  total_cost_in_paise / cost_per_sq_m

noncomputable def length_of_tank (width height cost_in_rupees rate : ℕ) : ℕ :=
  let total_cost_in_paise := cost_to_paise cost_in_rupees
  let total_area := total_area_plastered total_cost_in_paise rate
  let area_eq := total_area = (2 * (height * width) + 2 * (6 * height) + (height * width))
  let simplified_eq := total_area - 144 = 24 * height
  (total_area - 144) / 24

theorem tank_length_is_25 (width height cost_in_rupees rate : ℕ) : 
  width = 12 → height = 6 → cost_in_rupees = 186 → rate = 25 → length_of_tank width height cost_in_rupees rate = 25 :=
  by
    intros hwidth hheight hcost hrate
    unfold length_of_tank
    rw [hwidth, hheight, hcost, hrate]
    simp
    sorry

end tank_length_is_25_l303_303188
