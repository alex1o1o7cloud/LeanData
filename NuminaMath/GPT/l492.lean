import Mathlib

namespace common_ratio_is_0_88_second_term_is_475_2_l492_49245

-- Define the first term and the sum of the infinite geometric series
def first_term : Real := 540
def sum_infinite_series : Real := 4500

-- Required properties of the common ratio
def common_ratio (r : Real) : Prop :=
  abs r < 1 ∧ sum_infinite_series = first_term / (1 - r)

-- Prove the common ratio is 0.88 given the conditions
theorem common_ratio_is_0_88 : ∃ r : Real, common_ratio r ∧ r = 0.88 :=
by 
  sorry

-- Calculate the second term of the series
def second_term (r : Real) : Real := first_term * r

-- Prove the second term is 475.2 given the common ratio is 0.88
theorem second_term_is_475_2 : second_term 0.88 = 475.2 :=
by 
  sorry

end common_ratio_is_0_88_second_term_is_475_2_l492_49245


namespace part_a_part_b_l492_49263

-- Definition for combination
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Proof problems as Lean statements
theorem part_a : combination 30 2 = 435 := by
  sorry

theorem part_b : combination 30 3 = 4060 := by
  sorry

end part_a_part_b_l492_49263


namespace max_plates_l492_49210

def cost_pan : ℕ := 3
def cost_pot : ℕ := 5
def cost_plate : ℕ := 11
def total_cost : ℕ := 100
def min_pans : ℕ := 2
def min_pots : ℕ := 2

theorem max_plates (p q r : ℕ) :
  p >= min_pans → q >= min_pots → (cost_pan * p + cost_pot * q + cost_plate * r = total_cost) → r = 7 :=
by
  intros h_p h_q h_cost
  sorry

end max_plates_l492_49210


namespace unique_solution_l492_49239

theorem unique_solution:
  ∃! (x y z : ℕ), 2^x + 9 * 7^y = z^3 ∧ x = 0 ∧ y = 1 ∧ z = 4 :=
by
  sorry

end unique_solution_l492_49239


namespace intersection_A_B_l492_49270

def setA : Set ℝ := {x : ℝ | x > -1}
def setB : Set ℝ := {x : ℝ | x < 3}
def setIntersection : Set ℝ := {x : ℝ | x > -1 ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = setIntersection :=
by sorry

end intersection_A_B_l492_49270


namespace valid_three_digit_numbers_l492_49287

   noncomputable def three_digit_num_correct (A : ℕ) : Prop :=
     (100 ≤ A ∧ A < 1000) ∧ (1000000 + A = A * A)

   theorem valid_three_digit_numbers (A : ℕ) :
     three_digit_num_correct A → (A = 625 ∨ A = 376) :=
   by
     sorry
   
end valid_three_digit_numbers_l492_49287


namespace competition_participants_l492_49298

theorem competition_participants (n : ℕ) :
    (100 < n ∧ n < 200) ∧
    (n % 4 = 2) ∧
    (n % 5 = 2) ∧
    (n % 6 = 2)
    → (n = 122 ∨ n = 182) :=
by
  intro h
  sorry

end competition_participants_l492_49298


namespace bobs_sisters_mile_time_l492_49292

theorem bobs_sisters_mile_time (bobs_current_time_minutes : ℕ) (bobs_current_time_seconds : ℕ) (improvement_percentage : ℝ) :
  bobs_current_time_minutes = 10 → bobs_current_time_seconds = 40 → improvement_percentage = 9.062499999999996 →
  bobs_sisters_time_minutes = 9 ∧ bobs_sisters_time_seconds = 42 :=
by
  -- Definitions from conditions
  let bobs_time_in_seconds := bobs_current_time_minutes * 60 + bobs_current_time_seconds
  let improvement_in_seconds := bobs_time_in_seconds * improvement_percentage / 100
  let target_time_in_seconds := bobs_time_in_seconds - improvement_in_seconds
  let bobs_sisters_time_minutes := target_time_in_seconds / 60
  let bobs_sisters_time_seconds := target_time_in_seconds % 60
  
  sorry

end bobs_sisters_mile_time_l492_49292


namespace intersection_points_count_l492_49227

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := |3 * x + 6|
def f2 (x : ℝ) : ℝ := -|4 * x - 4|

-- Prove the number of intersection points is 2
theorem intersection_points_count : 
  (∃ x1 y1, (f1 x1 = y1) ∧ (f2 x1 = y1)) ∧ 
  (∃ x2 y2, (f1 x2 = y2) ∧ (f2 x2 = y2) ∧ x1 ≠ x2) :=
sorry

end intersection_points_count_l492_49227


namespace abs_neg_product_eq_product_l492_49223

variable (a b : ℝ)

theorem abs_neg_product_eq_product (h1 : a < 0) (h2 : 0 < b) : |-a * b| = a * b := by
  sorry

end abs_neg_product_eq_product_l492_49223


namespace empty_set_subset_zero_set_l492_49272

-- Define the sets
def zero_set : Set ℕ := {0}
def empty_set : Set ℕ := ∅

-- State the problem
theorem empty_set_subset_zero_set : empty_set ⊂ zero_set :=
sorry

end empty_set_subset_zero_set_l492_49272


namespace cylinder_radius_in_cone_l492_49258

theorem cylinder_radius_in_cone (d h r : ℝ) (h_d : d = 20) (h_h : h = 24) (h_cylinder : 2 * r = r):
  r = 60 / 11 :=
by
  sorry

end cylinder_radius_in_cone_l492_49258


namespace monogram_count_l492_49256

theorem monogram_count :
  ∃ (n : ℕ), n = 156 ∧
    (∃ (beforeM : Fin 13) (afterM : Fin 14),
      ∀ (a : Fin 13) (b : Fin 14),
        a < b → (beforeM = a ∧ afterM = b) → n = 12 * 13
    ) :=
by {
  sorry
}

end monogram_count_l492_49256


namespace problem_solution_l492_49255

def tens_digit_is_odd (n : ℕ) : Bool :=
  let m := (n * n + n) / 10 % 10
  m % 2 = 1

def count_tens_digit_odd : ℕ :=
  List.range 50 |>.filter tens_digit_is_odd |>.length

theorem problem_solution : count_tens_digit_odd = 25 :=
  sorry

end problem_solution_l492_49255


namespace calculate_x_n_minus_inverse_x_n_l492_49250

theorem calculate_x_n_minus_inverse_x_n
  (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) (x : ℝ) (h : x - 1/x = 2 * Real.sin θ) (n : ℕ) (hn : 0 < n) :
  x^n - 1/x^n = 2 * Real.sinh (n * θ) :=
by sorry

end calculate_x_n_minus_inverse_x_n_l492_49250


namespace dispatch_plans_count_l492_49290

theorem dispatch_plans_count:
  -- conditions
  let total_athletes := 9
  let basketball_players := 5
  let soccer_players := 6
  let both_players := 2
  let only_basketball := 3
  let only_soccer := 4
  -- proof
  (both_players.choose 2 + both_players * only_basketball + both_players * only_soccer + only_basketball * only_soccer) = 28 :=
by
  sorry

end dispatch_plans_count_l492_49290


namespace division_decomposition_l492_49222

theorem division_decomposition (a b : ℕ) (h₁ : a = 36) (h₂ : b = 3)
    (h₃ : 30 / b = 10) (h₄ : 6 / b = 2) (h₅ : 10 + 2 = 12) :
    a / b = (30 / b) + (6 / b) := 
sorry

end division_decomposition_l492_49222


namespace suitcase_weight_on_return_l492_49274

def initial_weight : ℝ := 5
def perfume_count : ℝ := 5
def perfume_weight_oz : ℝ := 1.2
def chocolate_weight_lb : ℝ := 4
def soap_count : ℝ := 2
def soap_weight_oz : ℝ := 5
def jam_count : ℝ := 2
def jam_weight_oz : ℝ := 8
def oz_per_lb : ℝ := 16

theorem suitcase_weight_on_return :
  initial_weight + (perfume_count * perfume_weight_oz / oz_per_lb) + chocolate_weight_lb +
  (soap_count * soap_weight_oz / oz_per_lb) + (jam_count * jam_weight_oz / oz_per_lb) = 11 := 
  by
  sorry

end suitcase_weight_on_return_l492_49274


namespace smallest_four_digit_integer_l492_49240

theorem smallest_four_digit_integer (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : ∀ d ∈ [1, 5, 6], n % d = 0)
  (h3 : ∀ d1 d2, d1 ≠ d2 → d1 ∈ [1, 5, 6] → d2 ∈ [1, 5, 6] → d1 ≠ d2) :
  n = 1560 :=
by
  sorry

end smallest_four_digit_integer_l492_49240


namespace max_value_of_expression_l492_49296

theorem max_value_of_expression (x : Real) :
  (x^4 / (x^8 + 2 * x^6 - 3 * x^4 + 5 * x^3 + 8 * x^2 + 5 * x + 25)) ≤ (1 / 15) :=
sorry

end max_value_of_expression_l492_49296


namespace find_x_for_dot_product_l492_49277

theorem find_x_for_dot_product :
  let a : (ℝ × ℝ) := (1, -1)
  let b : (ℝ × ℝ) := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 1) ↔ x = 1 :=
by
  sorry

end find_x_for_dot_product_l492_49277


namespace solve_x_l492_49247

theorem solve_x (x : ℝ) (hx : (1/x + 1/(2*x) + 1/(3*x) = 1/12)) : x = 22 :=
  sorry

end solve_x_l492_49247


namespace total_figurines_l492_49233

theorem total_figurines:
  let basswood_blocks := 25
  let butternut_blocks := 30
  let aspen_blocks := 35
  let oak_blocks := 40
  let cherry_blocks := 45
  let basswood_figs_per_block := 3
  let butternut_figs_per_block := 4
  let aspen_figs_per_block := 2 * basswood_figs_per_block
  let oak_figs_per_block := 5
  let cherry_figs_per_block := 7
  let basswood_total := basswood_blocks * basswood_figs_per_block
  let butternut_total := butternut_blocks * butternut_figs_per_block
  let aspen_total := aspen_blocks * aspen_figs_per_block
  let oak_total := oak_blocks * oak_figs_per_block
  let cherry_total := cherry_blocks * cherry_figs_per_block
  let total_figs := basswood_total + butternut_total + aspen_total + oak_total + cherry_total
  total_figs = 920 := by sorry

end total_figurines_l492_49233


namespace books_borrowed_l492_49276

theorem books_borrowed (initial_books : ℕ) (additional_books : ℕ) (remaining_books : ℕ) : 
  initial_books = 300 → 
  additional_books = 10 * 5 → 
  remaining_books = 210 → 
  initial_books + additional_books - remaining_books = 140 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  sorry

end books_borrowed_l492_49276


namespace basketball_team_girls_l492_49282

theorem basketball_team_girls (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3) * G = 18) : 
  G = 18 :=
by
  have h3 : G - (1 / 3) * G = 30 - 18 := by sorry
  have h4 : (2 / 3) * G = 12 := by sorry
  have h5 : G = 12 * (3 / 2) := by sorry
  have h6 : G = 18 := by sorry
  exact h6

end basketball_team_girls_l492_49282


namespace arithmetic_geometric_sequence_l492_49236

theorem arithmetic_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) 
(h1 : S 3 = 2) 
(h2 : S 6 = 18) 
(h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
: S 10 / S 5 = 33 := 
sorry

end arithmetic_geometric_sequence_l492_49236


namespace entree_cost_14_l492_49286

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l492_49286


namespace initial_crayons_per_box_l492_49226

-- Define the initial total number of crayons in terms of x
def total_initial_crayons (x : ℕ) : ℕ := 4 * x

-- Define the crayons given to Mae
def crayons_to_Mae : ℕ := 5

-- Define the crayons given to Lea
def crayons_to_Lea : ℕ := 12

-- Define the remaining crayons
def remaining_crayons : ℕ := 15

-- Prove that the initial number of crayons per box is 8 given the conditions
theorem initial_crayons_per_box (x : ℕ) : total_initial_crayons x - crayons_to_Mae - crayons_to_Lea = remaining_crayons → x = 8 :=
by
  intros h
  sorry

end initial_crayons_per_box_l492_49226


namespace dogwood_trees_after_5_years_l492_49207

theorem dogwood_trees_after_5_years :
  let current_trees := 39
  let trees_planted_today := 41
  let growth_rate_today := 2 -- trees per year
  let trees_planted_tomorrow := 20
  let growth_rate_tomorrow := 4 -- trees per year
  let years := 5
  let total_planted_trees := trees_planted_today + trees_planted_tomorrow
  let total_initial_trees := current_trees + total_planted_trees
  let total_growth_today := growth_rate_today * years
  let total_growth_tomorrow := growth_rate_tomorrow * years
  let total_growth := total_growth_today + total_growth_tomorrow
  let final_tree_count := total_initial_trees + total_growth
  final_tree_count = 130 := by
  sorry

end dogwood_trees_after_5_years_l492_49207


namespace sample_capacity_l492_49209

theorem sample_capacity (f : ℕ) (r : ℚ) (n : ℕ) (h₁ : f = 40) (h₂ : r = 0.125) (h₃ : r * n = f) : n = 320 :=
sorry

end sample_capacity_l492_49209


namespace solve_cubic_inequality_l492_49243

theorem solve_cubic_inequality :
  { x : ℝ | x^3 + x^2 - 7 * x + 6 < 0 } = { x : ℝ | -3 < x ∧ x < 1 ∨ 1 < x ∧ x < 2 } :=
by
  sorry

end solve_cubic_inequality_l492_49243


namespace summer_sales_is_2_million_l492_49202

def spring_sales : ℝ := 4.8
def autumn_sales : ℝ := 7
def winter_sales : ℝ := 2.2
def spring_percentage : ℝ := 0.3

theorem summer_sales_is_2_million :
  ∃ (total_sales : ℝ), total_sales = (spring_sales / spring_percentage) ∧
  ∃ summer_sales : ℝ, total_sales = spring_sales + summer_sales + autumn_sales + winter_sales ∧
  summer_sales = 2 :=
by
  sorry

end summer_sales_is_2_million_l492_49202


namespace volume_of_sphere_in_cone_l492_49234

theorem volume_of_sphere_in_cone :
  let diameter_of_base := 16 * Real.sqrt 2
  let radius_of_base := diameter_of_base / 2
  let side_length := radius_of_base * 2 / Real.sqrt 2
  let inradius := side_length / 2
  let r := inradius
  let V := (4 / 3) * Real.pi * r^3
  V = (2048 / 3) * Real.pi := by
  sorry

end volume_of_sphere_in_cone_l492_49234


namespace gumballs_remaining_l492_49228

theorem gumballs_remaining (Alicia_gumballs : ℕ) (Pedro_gumballs : ℕ) (Total_gumballs : ℕ) (Gumballs_taken_out : ℕ)
  (h1 : Alicia_gumballs = 20)
  (h2 : Pedro_gumballs = Alicia_gumballs + 3 * Alicia_gumballs)
  (h3 : Total_gumballs = Alicia_gumballs + Pedro_gumballs)
  (h4 : Gumballs_taken_out = 40 * Total_gumballs / 100) :
  Total_gumballs - Gumballs_taken_out = 60 := by
  sorry

end gumballs_remaining_l492_49228


namespace max_edges_partitioned_square_l492_49217

theorem max_edges_partitioned_square (n v e : ℕ) 
  (h : v - e + n = 1) : e ≤ 3 * n + 1 := 
sorry

end max_edges_partitioned_square_l492_49217


namespace eval_expression_l492_49230

theorem eval_expression : (2: ℤ)^2 - 3 * (2: ℤ) + 2 = 0 := by
  sorry

end eval_expression_l492_49230


namespace add_mul_of_3_l492_49232

theorem add_mul_of_3 (a b : ℤ) (ha : ∃ m : ℤ, a = 6 * m) (hb : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end add_mul_of_3_l492_49232


namespace probability_stopping_after_three_draws_l492_49281

def draws : List (List ℕ) := [
  [2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0], [0, 0, 1],
  [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [0, 3, 1], [3, 2, 0], [1, 2, 2], [1, 0, 3], [2, 3, 3]
]

def favorable_sequences (seqs : List (List ℕ)) : List (List ℕ) :=
  seqs.filter (λ seq => 0 ∈ seq ∧ 1 ∈ seq)

def probability_of_drawing_zhong_hua (seqs : List (List ℕ)) : ℚ :=
  (favorable_sequences seqs).length / seqs.length

theorem probability_stopping_after_three_draws :
  probability_of_drawing_zhong_hua draws = 5 / 18 := by
sorry

end probability_stopping_after_three_draws_l492_49281


namespace algebraic_expression_equality_l492_49297

variable {x : ℝ}

theorem algebraic_expression_equality (h : x^2 + 3*x + 8 = 7) : 3*x^2 + 9*x - 2 = -5 := 
by
  sorry

end algebraic_expression_equality_l492_49297


namespace largest_n_divisible_l492_49252

theorem largest_n_divisible (n : ℕ) : (n^3 + 150) % (n + 15) = 0 → n ≤ 2385 := by
  sorry

end largest_n_divisible_l492_49252


namespace range_of_m_l492_49269

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) (h : f (2 * m - 1) + f (3 - m) > 0) : m > -2 :=
by
  sorry

end range_of_m_l492_49269


namespace unique_real_x_satisfies_eq_l492_49265

theorem unique_real_x_satisfies_eq (x : ℝ) (h : x ≠ 0) : (7 * x)^5 = (14 * x)^4 ↔ x = 16 / 7 :=
by sorry

end unique_real_x_satisfies_eq_l492_49265


namespace total_turtles_rabbits_l492_49251

-- Number of turtles and rabbits on Happy Island
def turtles_happy : ℕ := 120
def rabbits_happy : ℕ := 80

-- Number of turtles and rabbits on Lonely Island
def turtles_lonely : ℕ := turtles_happy / 3
def rabbits_lonely : ℕ := turtles_lonely

-- Number of turtles and rabbits on Serene Island
def rabbits_serene : ℕ := 2 * rabbits_lonely
def turtles_serene : ℕ := (3 * rabbits_lonely) / 4

-- Number of turtles and rabbits on Tranquil Island
def turtles_tranquil : ℕ := (turtles_happy - turtles_serene) + 5
def rabbits_tranquil : ℕ := turtles_tranquil

-- Proving the total numbers
theorem total_turtles_rabbits :
    turtles_happy = 120 ∧ rabbits_happy = 80 ∧
    turtles_lonely = 40 ∧ rabbits_lonely = 40 ∧
    turtles_serene = 30 ∧ rabbits_serene = 80 ∧
    turtles_tranquil = 95 ∧ rabbits_tranquil = 95 ∧
    (turtles_happy + turtles_lonely + turtles_serene + turtles_tranquil = 285) ∧
    (rabbits_happy + rabbits_lonely + rabbits_serene + rabbits_tranquil = 295) := 
    by
        -- Here we prove each part step by step using the definitions and conditions provided above
        sorry

end total_turtles_rabbits_l492_49251


namespace intersection_a_four_range_of_a_l492_49216

variable {x a : ℝ}

-- Problem 1: Intersection of A and B for a = 4
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a^2 + 2}

theorem intersection_a_four : A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} := 
by  sorry

-- Problem 2: Range of a given condition
theorem range_of_a (a : ℝ) (h1 : a > -3/2) (h2 : ∀ x ∈ A a, x ∈ B a) : 1 ≤ a ∧ a ≤ 3 := 
by  sorry

end intersection_a_four_range_of_a_l492_49216


namespace tara_had_more_l492_49288

theorem tara_had_more (M T X : ℕ) (h1 : T = 15) (h2 : M + T = 26) (h3 : T = M + X) : X = 4 :=
by 
  sorry

end tara_had_more_l492_49288


namespace meaningful_expression_range_l492_49229

theorem meaningful_expression_range (x : ℝ) : (∃ y : ℝ, y = (1 / (Real.sqrt (x - 2)))) ↔ (x > 2) := 
sorry

end meaningful_expression_range_l492_49229


namespace find_time_l492_49214

theorem find_time (s z t : ℝ) (h : ∀ s, 0 ≤ s ∧ s ≤ 7 → z = s^2 + 2 * s) : 
  z = 35 ∧ z = t^2 + 2 * t + 20 → 0 ≤ t ∧ t ≤ 7 → t = 3 :=
by
  sorry

end find_time_l492_49214


namespace length_of_rectangular_garden_l492_49204

-- Define the perimeter and breadth conditions
def perimeter : ℕ := 950
def breadth : ℕ := 100

-- The formula for the perimeter of a rectangle
def formula (L B : ℕ) : ℕ := 2 * (L + B)

-- State the theorem
theorem length_of_rectangular_garden (L : ℕ) 
  (h1 : perimeter = 2 * (L + breadth)) : 
  L = 375 := 
by
  sorry

end length_of_rectangular_garden_l492_49204


namespace fifteen_horses_fifteen_bags_l492_49219

-- Definitions based on the problem
def days_for_one_horse_one_bag : ℝ := 1  -- It takes 1 day for 1 horse to eat 1 bag of grain

-- Theorem statement
theorem fifteen_horses_fifteen_bags {d : ℝ} (h : d = days_for_one_horse_one_bag) :
  d = 1 :=
by
  sorry

end fifteen_horses_fifteen_bags_l492_49219


namespace determine_k_value_l492_49285

theorem determine_k_value : (5 ^ 1002 + 6 ^ 1001) ^ 2 - (5 ^ 1002 - 6 ^ 1001) ^ 2 = 24 * 30 ^ 1001 :=
by
  sorry

end determine_k_value_l492_49285


namespace positive_integers_satisfying_inequality_l492_49205

theorem positive_integers_satisfying_inequality (x : ℕ) (hx : x > 0) : 4 - x > 1 ↔ x = 1 ∨ x = 2 :=
by
  sorry

end positive_integers_satisfying_inequality_l492_49205


namespace max_extra_time_matches_l492_49279

theorem max_extra_time_matches (number_teams : ℕ) 
    (points_win : ℕ) (points_lose : ℕ) 
    (points_win_extra : ℕ) (points_lose_extra : ℕ) 
    (total_matches_2016 : number_teams = 2016)
    (pts_win_3 : points_win = 3)
    (pts_lose_0 : points_lose = 0)
    (pts_win_extra_2 : points_win_extra = 2)
    (pts_lose_extra_1 : points_lose_extra = 1) :
    ∃ N, N = 1512 := 
by {
  sorry
}

end max_extra_time_matches_l492_49279


namespace least_of_10_consecutive_odd_integers_average_154_l492_49294

theorem least_of_10_consecutive_odd_integers_average_154 (x : ℤ)
  (h_avg : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18)) / 10 = 154) :
  x = 145 :=
by 
  sorry

end least_of_10_consecutive_odd_integers_average_154_l492_49294


namespace negation_of_original_prop_l492_49280

variable (a : ℝ)
def original_prop (x : ℝ) : Prop := x^2 + a * x + 1 < 0

theorem negation_of_original_prop :
  ¬ (∃ x : ℝ, original_prop a x) ↔ ∀ x : ℝ, ¬ original_prop a x :=
by sorry

end negation_of_original_prop_l492_49280


namespace smallest_n_l492_49235

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end smallest_n_l492_49235


namespace probability_is_half_l492_49267

noncomputable def probability_at_least_35_cents : ℚ :=
  let total_outcomes := 32
  let successful_outcomes := 8 + 4 + 4 -- from solution steps (1, 2, 3)
  successful_outcomes / total_outcomes

theorem probability_is_half :
  probability_at_least_35_cents = 1 / 2 := by
  -- proof details are not required as per instructions
  sorry

end probability_is_half_l492_49267


namespace smallest_digit_N_divisible_by_6_l492_49200

theorem smallest_digit_N_divisible_by_6 : 
  ∃ N : ℕ, N < 10 ∧ 
          (14530 + N) % 6 = 0 ∧
          ∀ M : ℕ, M < N → (14530 + M) % 6 ≠ 0 := sorry

end smallest_digit_N_divisible_by_6_l492_49200


namespace rectangle_area_l492_49246

theorem rectangle_area (p : ℝ) (l : ℝ) (h1 : 2 * (l + 2 * l) = p) :
  l * 2 * l = p^2 / 18 :=
by
  sorry

end rectangle_area_l492_49246


namespace necessary_but_not_sufficient_condition_l492_49237

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x < 5) : (x < 2 → x < 5) ∧ ¬(x < 5 → x < 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l492_49237


namespace eggs_town_hall_l492_49206

-- Definitions of given conditions
def eggs_club_house : ℕ := 40
def eggs_park : ℕ := 25
def total_eggs_found : ℕ := 80

-- Problem statement
theorem eggs_town_hall : total_eggs_found - (eggs_club_house + eggs_park) = 15 := by
  sorry

end eggs_town_hall_l492_49206


namespace least_integer_a_divisible_by_240_l492_49295

theorem least_integer_a_divisible_by_240 (a : ℤ) (h1 : 240 ∣ a^3) : a ≥ 60 := by
  sorry

end least_integer_a_divisible_by_240_l492_49295


namespace count_three_digit_with_f_l492_49220

open Nat

def f : ℕ → ℕ := sorry 

axiom f_add_add (a b : ℕ) : f (a + b) = f (f a + b)
axiom f_add_small (a b : ℕ) (h : a + b < 10) : f (a + b) = f a + f b
axiom f_10 : f 10 = 1

theorem count_three_digit_with_f (hN : ∀ n : ℕ, f 2^(3^(4^5)) = f n):
  ∃ k, k = 100 ∧ ∀ n, 100 ≤ n ∧ n < 1000 → (f n = f 2^(3^(4^5))) :=
sorry

end count_three_digit_with_f_l492_49220


namespace max_value_m_l492_49283

theorem max_value_m (m n : ℕ) (h : 8 * m + 9 * n = m * n + 6) : m ≤ 75 := 
sorry

end max_value_m_l492_49283


namespace divides_prime_factors_l492_49221

theorem divides_prime_factors (a b : ℕ) (p : ℕ → ℕ → Prop) (k l : ℕ → ℕ) (n : ℕ) : 
  (a ∣ b) ↔ (∀ i : ℕ, i < n → k i ≤ l i) :=
by
  sorry

end divides_prime_factors_l492_49221


namespace floor_length_l492_49208

theorem floor_length (b l : ℝ)
  (h1 : l = 3 * b)
  (h2 : 3 * b^2 = 484 / 3) :
  l = 22 := 
sorry

end floor_length_l492_49208


namespace problem_solution_l492_49299

-- Definitions based on conditions
def valid_sequence (b : Fin 7 → Nat) : Prop :=
  (∀ i j : Fin 7, i ≤ j → b i ≥ b j) ∧ 
  (∀ i : Fin 7, b i ≤ 1500) ∧ 
  (∀ i : Fin 7, (b i + i) % 3 = 0)

-- The main theorem
theorem problem_solution :
  (∃ b : Fin 7 → Nat, valid_sequence b) →
  @Nat.choose 506 7 % 1000 = 506 :=
sorry

end problem_solution_l492_49299


namespace stuffed_animal_total_l492_49262

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l492_49262


namespace bottles_recycled_l492_49284

theorem bottles_recycled (start_bottles : ℕ) (recycle_ratio : ℕ) (answer : ℕ)
  (h_start : start_bottles = 256) (h_recycle : recycle_ratio = 4) : answer = 85 :=
sorry

end bottles_recycled_l492_49284


namespace domain_of_f_l492_49257

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - 2 * x) + Real.log (1 + 2 * x)

theorem domain_of_f : {x : ℝ | 1 - 2 * x > 0 ∧ 1 + 2 * x > 0} = {x : ℝ | -1 / 2 < x ∧ x < 1 / 2} :=
by
    sorry

end domain_of_f_l492_49257


namespace functions_identified_l492_49225

variable (n : ℕ) (hn : n > 1)
variable {f : ℕ → ℝ → ℝ}

-- Define the conditions f1, f2, ..., fn
axiom cond_1 (x y : ℝ) : f 1 x + f 1 y = f 2 x * f 2 y
axiom cond_2 (x y : ℝ) : f 2 (x^2) + f 2 (y^2) = f 3 x * f 3 y
axiom cond_3 (x y : ℝ) : f 3 (x^3) + f 3 (y^3) = f 4 x * f 4 y
-- ... Similarly define conditions up to cond_n
axiom cond_n (x y : ℝ) : f n (x^n) + f n (y^n) = f 1 x * f 1 y

theorem functions_identified (i : ℕ) (hi₁ : 1 ≤ i) (hi₂ : i ≤ n) (x : ℝ) :
  f i x = 0 ∨ f i x = 2 :=
sorry

end functions_identified_l492_49225


namespace simplified_expr_l492_49211

theorem simplified_expr : 
  (Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2) ^ 2) = (8 + 2 * Real.sqrt 2) := 
by 
  sorry

end simplified_expr_l492_49211


namespace tickets_problem_l492_49218

theorem tickets_problem (A C : ℝ) 
  (h1 : A + C = 200) 
  (h2 : 3 * A + 1.5 * C = 510) : C = 60 :=
by
  sorry

end tickets_problem_l492_49218


namespace correct_propositions_l492_49275

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n + a (n+1) > 2 * a n

def prop1 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  a 2 > a 1 → ∀ n > 1, a n > a (n-1)

def prop4 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  ∃ d, ∀ n > 1, a n > a 1 + (n-1) * d

theorem correct_propositions {a : ℕ → ℝ}
  (h : sequence_condition a) :
  (prop1 a h) ∧ (prop4 a h) := 
sorry

end correct_propositions_l492_49275


namespace red_paint_amount_l492_49241

theorem red_paint_amount (r w : ℕ) (hrw : r / w = 5 / 7) (hwhite : w = 21) : r = 15 :=
by {
  sorry
}

end red_paint_amount_l492_49241


namespace new_plants_description_l492_49254

-- Condition: Anther culture of diploid corn treated with colchicine.
def diploid_corn := Type
def colchicine_treatment (plant : diploid_corn) : Prop := -- assume we have some method to define it
sorry

def anther_culture (plant : diploid_corn) (treated : colchicine_treatment plant) : Type := -- assume we have some method to define it
sorry

-- Describe the properties of new plants
def is_haploid (plant : diploid_corn) : Prop := sorry
def has_no_homologous_chromosomes (plant : diploid_corn) : Prop := sorry
def cannot_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def has_homologous_chromosomes_in_somatic_cells (plant : diploid_corn) : Prop := sorry
def can_form_fertile_gametes (plant : diploid_corn) : Prop := sorry
def is_homozygous_or_heterozygous (plant : diploid_corn) : Prop := sorry
def is_definitely_homozygous (plant : diploid_corn) : Prop := sorry
def is_diploid (plant : diploid_corn) : Prop := sorry

-- Equivalent math proof problem
theorem new_plants_description (plant : diploid_corn) (treated : colchicine_treatment plant) : 
  is_haploid (anther_culture plant treated) ∧ 
  has_homologous_chromosomes_in_somatic_cells (anther_culture plant treated) ∧ 
  can_form_fertile_gametes (anther_culture plant treated) ∧ 
  is_homozygous_or_heterozygous (anther_culture plant treated) := sorry

end new_plants_description_l492_49254


namespace total_seeds_planted_l492_49224

theorem total_seeds_planted 
    (seeds_per_bed : ℕ) 
    (seeds_grow_per_bed : ℕ) 
    (total_flowers : ℕ) 
    (h1 : seeds_per_bed = 15) 
    (h2 : seeds_grow_per_bed = 60) 
    (h3 : total_flowers = 220) : 
    ∃ (total_seeds : ℕ), total_seeds = 85 := 
by
    sorry

end total_seeds_planted_l492_49224


namespace guessing_game_l492_49268

-- Define the conditions
def number : ℕ := 33
def result : ℕ := 2 * 51 - 3

-- Define the factor (to be proven)
def factor (n r : ℕ) : ℕ := r / n

-- The theorem to be proven
theorem guessing_game (n r : ℕ) (h1 : n = 33) (h2 : r = 2 * 51 - 3) : 
  factor n r = 3 := by
  -- Placeholder for the actual proof
  sorry

end guessing_game_l492_49268


namespace alchemerion_age_problem_l492_49244

theorem alchemerion_age_problem
  (A S F : ℕ)  -- Declare the ages as natural numbers
  (h1 : A = 3 * S)  -- Condition 1: Alchemerion is 3 times his son's age
  (h2 : F = 2 * A + 40)  -- Condition 2: His father’s age is 40 years more than twice his age
  (h3 : A + S + F = 1240)  -- Condition 3: Together they are 1240 years old
  (h4 : A = 360)  -- Condition 4: Alchemerion is 360 years old
  : 40 = F - 2 * A :=  -- Conclusion: The number of years more than twice Alchemerion’s age is 40
by
  sorry  -- Proof can be filled in here

end alchemerion_age_problem_l492_49244


namespace problem1_problem2_l492_49266

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + (a - 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem problem1 (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) → a = 2 ∨ a = 3 := sorry

theorem problem2 (m : ℝ) : (∀ x, x ∈ A → x ∈ C m) → m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) := sorry

end problem1_problem2_l492_49266


namespace problem1_problem2_problem3_problem4_problem5_problem6_l492_49273

-- Proof for 238 + 45 × 5 = 463
theorem problem1 : 238 + 45 * 5 = 463 := by
  sorry

-- Proof for 65 × 4 - 128 = 132
theorem problem2 : 65 * 4 - 128 = 132 := by
  sorry

-- Proof for 900 - 108 × 4 = 468
theorem problem3 : 900 - 108 * 4 = 468 := by
  sorry

-- Proof for 369 + (512 - 215) = 666
theorem problem4 : 369 + (512 - 215) = 666 := by
  sorry

-- Proof for 758 - 58 × 9 = 236
theorem problem5 : 758 - 58 * 9 = 236 := by
  sorry

-- Proof for 105 × (81 ÷ 9 - 3) = 630
theorem problem6 : 105 * (81 / 9 - 3) = 630 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l492_49273


namespace initial_concentration_alcohol_l492_49212

theorem initial_concentration_alcohol (x : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 100)
    (h2 : 0.44 * 10 = (x / 100) * 2 + 3.6) :
    x = 40 :=
sorry

end initial_concentration_alcohol_l492_49212


namespace factor_expression_l492_49291

theorem factor_expression (x : ℝ) : 35 * x ^ 13 + 245 * x ^ 26 = 35 * x ^ 13 * (1 + 7 * x ^ 13) :=
by {
  sorry
}

end factor_expression_l492_49291


namespace floor_S_value_l492_49264

noncomputable def floor_S (a b c d : ℝ) : ℝ :=
  a + b + c + d

theorem floor_S_value (a b c d : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_sum_sq : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (h_product : a * c = 1008 ∧ b * d = 1008) :
  ⌊floor_S a b c d⌋ = 117 :=
by
  sorry

end floor_S_value_l492_49264


namespace find_m_from_parallel_l492_49261

theorem find_m_from_parallel (m : ℝ) : 
  (∃ (A B : ℝ×ℝ), A = (-2, m) ∧ B = (m, 4) ∧
  (∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -1 ∧
  (a * (B.1 - A.1) + b * (B.2 - A.2) = 0)) ) 
  → m = -8 :=
by
  sorry

end find_m_from_parallel_l492_49261


namespace raise_percentage_to_original_l492_49271

-- Let original_salary be a variable representing the original salary.
-- Since the salary was reduced by 50%, the reduced_salary is half of the original_salary.
-- We need to prove that to get the reduced_salary back to the original_salary, 
-- it must be increased by 100%.

noncomputable def original_salary : ℝ := sorry
noncomputable def reduced_salary : ℝ := original_salary * 0.5

theorem raise_percentage_to_original :
  (original_salary - reduced_salary) / reduced_salary * 100 = 100 :=
sorry

end raise_percentage_to_original_l492_49271


namespace value_of_expression_l492_49248

theorem value_of_expression (a b c d : ℝ) (h : a + b + c + d = 4) : 12 * a - 6 * b + 3 * c - 2 * d = 40 :=
by sorry

end value_of_expression_l492_49248


namespace relationship_M_N_l492_49249

theorem relationship_M_N (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) 
  (M : ℝ) (hM : M = a * b) (N : ℝ) (hN : N = a + b - 1) : M > N :=
by
  sorry

end relationship_M_N_l492_49249


namespace part_one_solution_set_part_two_range_of_m_l492_49253

-- Part I
theorem part_one_solution_set (x : ℝ) : (|x + 1| + |x - 2| - 5 > 0) ↔ (x > 3 ∨ x < -2) :=
sorry

-- Part II
theorem part_two_range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) ↔ (m ≤ 1) :=
sorry

end part_one_solution_set_part_two_range_of_m_l492_49253


namespace sum_of_first_15_odd_positive_integers_l492_49242

theorem sum_of_first_15_odd_positive_integers :
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  S_n = 225 :=
by
  let a := 1
  let d := 2
  let n := 15
  let l := a + (n - 1) * d
  let S_n := (n / 2) * (a + l)
  have : S_n = 225 := sorry
  exact this

end sum_of_first_15_odd_positive_integers_l492_49242


namespace tangent_line_slope_at_one_l492_49213

variable {f : ℝ → ℝ}

theorem tangent_line_slope_at_one (h : ∀ x, f x = e * x - e) : deriv f 1 = e :=
by sorry

end tangent_line_slope_at_one_l492_49213


namespace find_x_l492_49201

theorem find_x (y : ℝ) (x : ℝ) : 
  (5 + 2*x) / (7 + 3*x + y) = (3 + 4*x) / (4 + 2*x + y) ↔ 
  x = (-19 + Real.sqrt 329) / 16 ∨ x = (-19 - Real.sqrt 329) / 16 :=
by
  sorry

end find_x_l492_49201


namespace quadratic_polynomial_with_conditions_l492_49293

theorem quadratic_polynomial_with_conditions :
  ∃ (a b c : ℝ), 
  (∀ x : ℂ, x = -3 - 4 * Complex.I ∨ x = -3 + 4 * Complex.I → a * x^2 + b * x + c = 0)
  ∧ b = -10 
  ∧ a = -5/3 
  ∧ c = -125/3 := 
sorry

end quadratic_polynomial_with_conditions_l492_49293


namespace how_long_it_lasts_l492_49215

-- Define a structure to hold the conditions
structure MoneySpending where
  mowing_income : ℕ
  weeding_income : ℕ
  weekly_expense : ℕ

-- Example conditions given in the problem
def lukesEarnings : MoneySpending :=
{ mowing_income := 9,
  weeding_income := 18,
  weekly_expense := 3 }

-- Main theorem proving the number of weeks he can sustain his spending
theorem how_long_it_lasts (data : MoneySpending) : 
  (data.mowing_income + data.weeding_income) / data.weekly_expense = 9 := by
  sorry

end how_long_it_lasts_l492_49215


namespace num_positive_integers_le_500_l492_49278

-- Define a predicate to state that a number is a perfect square
def is_square (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

-- Define the main theorem
theorem num_positive_integers_le_500 (n : ℕ) :
  (∃ (ns : Finset ℕ), (∀ x ∈ ns, x ≤ 500 ∧ is_square (21 * x)) ∧ ns.card = 4) :=
by
  sorry

end num_positive_integers_le_500_l492_49278


namespace area_of_side_face_l492_49231

theorem area_of_side_face (l w h : ℝ)
  (h_front_top : w * h = 0.5 * (l * h))
  (h_top_side : l * h = 1.5 * (w * h))
  (h_volume : l * w * h = 3000) :
  w * h = 200 := 
sorry

end area_of_side_face_l492_49231


namespace max_colors_l492_49238

theorem max_colors (n : ℕ) (color : ℕ → ℕ → ℕ)
  (h_color_property : ∀ i j : ℕ, i < 2^n → j < 2^n → color i j = color j ((i + j) % 2^n)) :
  ∃ (c : ℕ), c ≤ 2^n ∧ (∀ i j : ℕ, i < 2^n → j < 2^n → color i j < c) :=
sorry

end max_colors_l492_49238


namespace compute_expression_l492_49260

variable (a b : ℝ)

theorem compute_expression : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := 
by sorry

end compute_expression_l492_49260


namespace equal_share_payment_l492_49259

theorem equal_share_payment (A B C : ℝ) (h : A < B ∧ B < C) :
  (B + C - 2 * A) / 3 + (A + B - 2 * C) / 3 = ((A + B + C) * 2 / 3) - B :=
by
  sorry

end equal_share_payment_l492_49259


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l492_49289

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l492_49289


namespace probability_green_ball_eq_l492_49203

noncomputable def prob_green_ball : ℚ := 
  1 / 3 * (5 / 18) + 1 / 3 * (1 / 2) + 1 / 3 * (1 / 2)

theorem probability_green_ball_eq : 
  prob_green_ball = 23 / 54 := 
  by
  sorry

end probability_green_ball_eq_l492_49203
