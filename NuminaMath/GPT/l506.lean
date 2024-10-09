import Mathlib

namespace gcd_12a_18b_l506_50672

theorem gcd_12a_18b (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a.gcd b = 15) : (12 * a).gcd (18 * b) = 90 :=
by sorry

end gcd_12a_18b_l506_50672


namespace intersection_M_N_l506_50683

def M (x : ℝ) : Prop := abs (x - 1) ≥ 2

def N (x : ℝ) : Prop := x^2 - 4 * x ≥ 0

def P (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4

theorem intersection_M_N (x : ℝ) : (M x ∧ N x) → P x :=
by
  sorry

end intersection_M_N_l506_50683


namespace number_of_roof_tiles_l506_50628

def land_cost : ℝ := 50
def bricks_cost_per_1000 : ℝ := 100
def roof_tile_cost : ℝ := 10
def land_required : ℝ := 2000
def bricks_required : ℝ := 10000
def total_construction_cost : ℝ := 106000

theorem number_of_roof_tiles :
  let land_total := land_cost * land_required
  let bricks_total := (bricks_required / 1000) * bricks_cost_per_1000
  let remaining_cost := total_construction_cost - (land_total + bricks_total)
  let roof_tiles := remaining_cost / roof_tile_cost
  roof_tiles = 500 := by
  sorry

end number_of_roof_tiles_l506_50628


namespace race_distance_l506_50627

theorem race_distance {d x y z : ℝ} :
  (d / x = (d - 25) / y) →
  (d / y = (d - 15) / z) →
  (d / x = (d - 37) / z) →
  d = 125 :=
by
  intros h1 h2 h3
  -- Insert proof here
  sorry

end race_distance_l506_50627


namespace katie_initial_candies_l506_50601

theorem katie_initial_candies (K : ℕ) (h1 : K + 23 - 8 = 23) : K = 8 :=
sorry

end katie_initial_candies_l506_50601


namespace max_chocolates_eaten_by_Ben_l506_50625

-- Define the situation with Ben and Carol sharing chocolates
variable (b c k : ℕ) -- b for Ben, c for Carol, k is the multiplier

-- Define the conditions
def chocolates_shared (b c : ℕ) : Prop := b + c = 30
def carol_eats_multiple (b c k : ℕ) : Prop := c = k * b ∧ k > 0

-- The theorem statement that we want to prove
theorem max_chocolates_eaten_by_Ben 
  (h1 : chocolates_shared b c) 
  (h2 : carol_eats_multiple b c k) : 
  b ≤ 15 := by
  sorry

end max_chocolates_eaten_by_Ben_l506_50625


namespace fraction_irreducible_l506_50643
-- Import necessary libraries

-- Define the problem to prove
theorem fraction_irreducible (n: ℕ) (h: n > 0) : gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end fraction_irreducible_l506_50643


namespace subsets_of_A_value_of_a_l506_50660

def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 - a*x + 2 = 0}

theorem subsets_of_A : 
  (A = {1, 2} ∧ (∀ S, S ⊆ A → S = ∅ ∨ S = {1} ∨ S = {2} ∨ S = {1, 2}))  :=
by
  sorry

theorem value_of_a (a : ℝ) (B_non_empty : B a ≠ ∅) (B_subset_A : ∀ x, x ∈ B a → x ∈ A): 
  a = 3 :=
by
  sorry

end subsets_of_A_value_of_a_l506_50660


namespace max_gcd_of_sequence_l506_50658

/-- Define the sequence as a function. -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- Define the greatest common divisor of the sequence terms. -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- State the theorem of the maximum value of d. -/
theorem max_gcd_of_sequence : ∃ n : ℕ, d n = 401 := sorry

end max_gcd_of_sequence_l506_50658


namespace max_g_at_8_l506_50602

noncomputable def g : ℝ → ℝ :=
  sorry -- We define g here abstractly, with nonnegative coefficients

axiom g_nonneg_coeffs : ∀ x, 0 ≤ g x
axiom g_at_4 : g 4 = 16
axiom g_at_16 : g 16 = 256

theorem max_g_at_8 : g 8 ≤ 64 :=
by sorry

end max_g_at_8_l506_50602


namespace min_sum_dimensions_l506_50636

theorem min_sum_dimensions (a b c : ℕ) (h : a * b * c = 2310) : a + b + c ≥ 52 :=
sorry

end min_sum_dimensions_l506_50636


namespace distance_between_points_l506_50673

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_between_points :
  distance (-3, 4, 0) (2, -1, 6) = Real.sqrt 86 :=
by
  sorry

end distance_between_points_l506_50673


namespace problem_statement_l506_50666

noncomputable def log_three_four : ℝ := Real.log 4 / Real.log 3
noncomputable def a : ℝ := Real.log (log_three_four) / Real.log (3/4)
noncomputable def b : ℝ := Real.rpow (3/4 : ℝ) 0.5
noncomputable def c : ℝ := Real.rpow (4/3 : ℝ) 0.5

theorem problem_statement : a < b ∧ b < c :=
by
  sorry

end problem_statement_l506_50666


namespace crayons_given_to_friends_l506_50698

def initial_crayons : ℕ := 440
def lost_crayons : ℕ := 106
def remaining_crayons : ℕ := 223

theorem crayons_given_to_friends :
  initial_crayons - remaining_crayons - lost_crayons = 111 := 
by
  sorry

end crayons_given_to_friends_l506_50698


namespace sum_of_angles_in_segments_outside_pentagon_l506_50624

theorem sum_of_angles_in_segments_outside_pentagon 
  (α β γ δ ε : ℝ) 
  (hα : α = 0.5 * (360 - arc_BCDE))
  (hβ : β = 0.5 * (360 - arc_CDEA))
  (hγ : γ = 0.5 * (360 - arc_DEAB))
  (hδ : δ = 0.5 * (360 - arc_EABC))
  (hε : ε = 0.5 * (360 - arc_ABCD)) 
  (arc_BCDE arc_CDEA arc_DEAB arc_EABC arc_ABCD : ℝ) :
  α + β + γ + δ + ε = 720 := 
by 
  sorry

end sum_of_angles_in_segments_outside_pentagon_l506_50624


namespace broth_for_third_l506_50610

theorem broth_for_third (b : ℚ) (h : b = 6 + 3/4) : b / 3 = 2 + 1/4 := by
  sorry

end broth_for_third_l506_50610


namespace three_digit_number_parity_count_equal_l506_50654

/-- Prove the number of three-digit numbers with all digits having the same parity is equal to the number of three-digit numbers where adjacent digits have different parity. -/
theorem three_digit_number_parity_count_equal :
  ∃ (same_parity_count alternating_parity_count : ℕ),
    same_parity_count = alternating_parity_count ∧
    -- Condition for digits of the same parity
    same_parity_count = (4 * 5 * 5) + (5 * 5 * 5) ∧
    -- Condition for alternating parity digits (patterns EOE and OEO)
    alternating_parity_count = (4 * 5 * 5) + (5 * 5 * 5) := by
  sorry

end three_digit_number_parity_count_equal_l506_50654


namespace rice_containers_l506_50651

theorem rice_containers (total_weight_pounds : ℚ) (weight_per_container_ounces : ℚ) (pound_to_ounces : ℚ) : 
  total_weight_pounds = 29/4 → 
  weight_per_container_ounces = 29 → 
  pound_to_ounces = 16 → 
  (total_weight_pounds * pound_to_ounces) / weight_per_container_ounces = 4 := 
by
  intros h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end rice_containers_l506_50651


namespace maria_bought_9_hardcover_volumes_l506_50600

def total_volumes (h p : ℕ) : Prop := h + p = 15
def total_cost (h p : ℕ) : Prop := 10 * p + 30 * h = 330

theorem maria_bought_9_hardcover_volumes (h p : ℕ) (h_vol : total_volumes h p) (h_cost : total_cost h p) : h = 9 :=
by
  sorry

end maria_bought_9_hardcover_volumes_l506_50600


namespace part1_part2_l506_50694

-- Defining the function f(x) and the given conditions
def f (x a : ℝ) := x^2 - a * x + 2 * a - 2

-- Given conditions
variables (a : ℝ)
axiom f_condition : ∀ (x : ℝ), f (2 + x) a * f (2 - x) a = 4
axiom a_gt_0 : a > 0
axiom fx_bounds : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → 1 ≤ f x a ∧ f x a ≤ 3

-- To prove (part 1)
theorem part1 (h : f 2 a + f 3 a = 6) : a = 2 := sorry

-- To prove (part 2)
theorem part2 : (4 - (2 * Real.sqrt 6) / 3) ≤ a ∧ a ≤ 5 / 2 := sorry

end part1_part2_l506_50694


namespace grunters_win_all_6_games_l506_50696

-- Define the probability of the Grunters winning a single game
def probability_win_single_game : ℚ := 3 / 5

-- Define the number of games
def number_of_games : ℕ := 6

-- Calculate the probability of winning all games (all games are independent)
def probability_win_all_games (p : ℚ) (n : ℕ) : ℚ := p ^ n

-- Prove that the probability of the Grunters winning all 6 games is exactly 729/15625
theorem grunters_win_all_6_games :
  probability_win_all_games probability_win_single_game number_of_games = 729 / 15625 :=
by
  sorry

end grunters_win_all_6_games_l506_50696


namespace guzman_boxes_l506_50639

noncomputable def total_doughnuts : Nat := 48
noncomputable def doughnuts_per_box : Nat := 12

theorem guzman_boxes :
  ∃ (N : Nat), N = total_doughnuts / doughnuts_per_box ∧ N = 4 :=
by
  use 4
  sorry

end guzman_boxes_l506_50639


namespace popsicle_sticks_left_l506_50616

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l506_50616


namespace scientific_notation_of_3930_billion_l506_50693

theorem scientific_notation_of_3930_billion :
  (3930 * 10^9) = 3.93 * 10^12 :=
sorry

end scientific_notation_of_3930_billion_l506_50693


namespace find_second_number_l506_50657

theorem find_second_number (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4)
  (h3 : y / z = 4 / 7) :
  y = 240 / 7 :=
by sorry

end find_second_number_l506_50657


namespace jane_buys_4_bagels_l506_50689

theorem jane_buys_4_bagels (b m : ℕ) (h1 : b + m = 7) (h2 : (80 * b + 60 * m) % 100 = 0) : b = 4 := 
by sorry

end jane_buys_4_bagels_l506_50689


namespace coat_price_reduction_l506_50655

theorem coat_price_reduction (original_price : ℝ) (reduction_percent : ℝ)
  (price_is_500 : original_price = 500)
  (reduction_is_30 : reduction_percent = 0.30) :
  original_price * reduction_percent = 150 :=
by
  sorry

end coat_price_reduction_l506_50655


namespace marbles_in_jar_l506_50617

theorem marbles_in_jar (M : ℕ) (h1 : ∀ n : ℕ, n = 20 → ∀ m : ℕ, m = M / n → ∀ a b : ℕ, a = n + 2 → b = m - 1 → ∀ k : ℕ, k = M / a → k = b) : M = 220 :=
by 
  sorry

end marbles_in_jar_l506_50617


namespace snowflake_stamps_count_l506_50684

theorem snowflake_stamps_count (S : ℕ) (truck_stamps : ℕ) (rose_stamps : ℕ) :
  truck_stamps = S + 9 →
  rose_stamps = S + 9 - 13 →
  S + truck_stamps + rose_stamps = 38 →
  S = 11 :=
by
  intros h1 h2 h3
  sorry

end snowflake_stamps_count_l506_50684


namespace temp_difference_l506_50680

theorem temp_difference
  (temp_beijing : ℤ) 
  (temp_hangzhou : ℤ) 
  (h_beijing : temp_beijing = -10) 
  (h_hangzhou : temp_hangzhou = -1) : 
  temp_beijing - temp_hangzhou = -9 := 
by 
  rw [h_beijing, h_hangzhou] 
  sorry

end temp_difference_l506_50680


namespace compare_neg_fractions_l506_50681

theorem compare_neg_fractions : (-3 / 4) > (-5 / 6) :=
sorry

end compare_neg_fractions_l506_50681


namespace union_A_B_l506_50686

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_A_B : A ∪ B = {1, 2, 3} := 
by
  sorry

end union_A_B_l506_50686


namespace committee_count_with_president_l506_50649

-- Define the conditions
def total_people : ℕ := 12
def committee_size : ℕ := 5
def remaining_people : ℕ := 11
def president_inclusion : ℕ := 1

-- Define the calculation of binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

-- State the problem in Lean 4
theorem committee_count_with_president : 
  binomial remaining_people (committee_size - president_inclusion) = 330 :=
sorry

end committee_count_with_president_l506_50649


namespace expression_independent_of_a_l506_50677

theorem expression_independent_of_a (a : ℝ) :
  7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 :=
by sorry

end expression_independent_of_a_l506_50677


namespace matrix_det_is_zero_l506_50659

noncomputable def matrixDetProblem (a b : ℝ) : ℝ :=
  Matrix.det ![
    ![1, Real.cos (a - b), Real.sin a],
    ![Real.cos (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

theorem matrix_det_is_zero (a b : ℝ) : matrixDetProblem a b = 0 :=
  sorry

end matrix_det_is_zero_l506_50659


namespace upstream_speed_is_8_l506_50619

-- Definitions of given conditions
def downstream_speed : ℝ := 13
def stream_speed : ℝ := 2.5
def man's_upstream_speed : ℝ := downstream_speed - 2 * stream_speed

-- Theorem to prove
theorem upstream_speed_is_8 : man's_upstream_speed = 8 :=
by
  rw [man's_upstream_speed, downstream_speed, stream_speed]
  sorry

end upstream_speed_is_8_l506_50619


namespace rohan_monthly_salary_l506_50626

theorem rohan_monthly_salary (s : ℝ) 
  (h_food : s * 0.40 = f)
  (h_rent : s * 0.20 = hr) 
  (h_entertainment : s * 0.10 = e)
  (h_conveyance : s * 0.10 = c)
  (h_savings : s * 0.20 = 1000) : 
  s = 5000 := 
sorry

end rohan_monthly_salary_l506_50626


namespace sum_of_faces_edges_vertices_rectangular_prism_l506_50687

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l506_50687


namespace original_curve_equation_l506_50675

theorem original_curve_equation (x y : ℝ) (θ : ℝ) (hθ : θ = π / 4)
  (h : (∃ P : ℝ × ℝ, P = (x, y) ∧ (∃ P' : ℝ × ℝ, P' = (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ) ∧ ((P'.fst)^2 - (P'.snd)^2 = 2)))) :
  x * y = -1 :=
sorry

end original_curve_equation_l506_50675


namespace bulbs_in_bathroom_and_kitchen_l506_50678

theorem bulbs_in_bathroom_and_kitchen
  (bedroom_bulbs : Nat)
  (basement_bulbs : Nat)
  (garage_bulbs : Nat)
  (bulbs_per_pack : Nat)
  (packs_needed : Nat)
  (total_bulbs : Nat)
  (H1 : bedroom_bulbs = 2)
  (H2 : basement_bulbs = 4)
  (H3 : garage_bulbs = basement_bulbs / 2)
  (H4 : bulbs_per_pack = 2)
  (H5 : packs_needed = 6)
  (H6 : total_bulbs = packs_needed * bulbs_per_pack) :
  (total_bulbs - (bedroom_bulbs + basement_bulbs + garage_bulbs) = 4) :=
by
  sorry

end bulbs_in_bathroom_and_kitchen_l506_50678


namespace probability_no_coinciding_sides_l506_50691

theorem probability_no_coinciding_sides :
  let total_triangles := Nat.choose 10 3
  let unfavorable_outcomes := 60 + 10
  let favorable_outcomes := total_triangles - unfavorable_outcomes
  favorable_outcomes / total_triangles = 5 / 12 := by
  sorry

end probability_no_coinciding_sides_l506_50691


namespace sqrt_square_identity_l506_50650

-- Define the concept of square root
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Problem statement: prove (sqrt 12321)^2 = 12321
theorem sqrt_square_identity (x : ℝ) : (sqrt x) ^ 2 = x := by
  sorry

-- Specific instance for the given number
example : (sqrt 12321) ^ 2 = 12321 := sqrt_square_identity 12321

end sqrt_square_identity_l506_50650


namespace largest_value_of_m_exists_l506_50688

theorem largest_value_of_m_exists (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 30) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) : 
  ∃ m : ℝ, (m = min (a * b) (min (b * c) (c * a))) ∧ (m = 2) := sorry

end largest_value_of_m_exists_l506_50688


namespace juanita_spends_more_l506_50663

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end juanita_spends_more_l506_50663


namespace gain_percentage_second_book_l506_50609

theorem gain_percentage_second_book (CP1 CP2 SP1 SP2 : ℝ)
  (h1 : CP1 = 350) 
  (h2 : CP1 + CP2 = 600)
  (h3 : SP1 = CP1 - (0.15 * CP1))
  (h4 : SP1 = SP2) :
  SP2 = CP2 + (19 / 100 * CP2) :=
by
  sorry

end gain_percentage_second_book_l506_50609


namespace percentage_difference_is_20_l506_50629

/-
Given:
Height of sunflowers from Packet A = 192 inches
Height of sunflowers from Packet B = 160 inches

Show:
Percentage difference in height between Packet A and Packet B is 20%.
-/

-- Definitions of heights
def height_packet_A : ℤ := 192
def height_packet_B : ℤ := 160

-- Definition of percentage difference formula
def percentage_difference (hA hB : ℤ) : ℤ := ((hA - hB) * 100) / hB

-- Theorem statement
theorem percentage_difference_is_20 :
  percentage_difference height_packet_A height_packet_B = 20 :=
sorry

end percentage_difference_is_20_l506_50629


namespace mul_digits_example_l506_50695

theorem mul_digits_example (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : C = 2) (h8 : D = 5) : A + B = 2 := by
  sorry

end mul_digits_example_l506_50695


namespace parallel_lines_parallel_lines_solution_l506_50699

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) → a = -1 ∨ a = 2 :=
sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) ∧ 
  ((a = -1 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0)) ∨ 
  (a = 2 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0))) :=
sorry

end parallel_lines_parallel_lines_solution_l506_50699


namespace roots_of_polynomial_l506_50612

   -- We need to define the polynomial and then state that the roots are exactly {0, 3, -5}
   def polynomial (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x)

   theorem roots_of_polynomial :
     {x : ℝ | polynomial x = 0} = {0, 3, -5} :=
   by
     sorry
   
end roots_of_polynomial_l506_50612


namespace sum_of_squares_base_6_l506_50662

def to_base (n b : ℕ) : ℕ := sorry

theorem sum_of_squares_base_6 :
  let squares := (List.range 12).map (λ x => x.succ ^ 2);
  let squares_base6 := squares.map (λ x => to_base x 6);
  (squares_base6.sum) = to_base 10515 6 :=
by sorry

end sum_of_squares_base_6_l506_50662


namespace flour_needed_for_one_batch_l506_50669

theorem flour_needed_for_one_batch (F : ℝ) (h1 : 8 * F + 8 * 1.5 = 44) : F = 4 := 
by
    sorry

end flour_needed_for_one_batch_l506_50669


namespace problem1_problem2_problem3_problem4_l506_50635

theorem problem1 : -16 - (-12) - 24 + 18 = -10 := 
by
  sorry

theorem problem2 : 0.125 + (1 / 4) + (-9 / 4) + (-0.25) = -2 := 
by
  sorry

theorem problem3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := 
by
  sorry

theorem problem4 : (-2 + 3) * 3 - (-2)^3 / 4 = 5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l506_50635


namespace min_value_expression_l506_50634

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    4.5 ≤ (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) :=
by
  sorry

end min_value_expression_l506_50634


namespace rectangle_area_increase_l506_50648

theorem rectangle_area_increase
  (l w : ℝ)
  (h₀ : l > 0) -- original length is positive
  (h₁ : w > 0) -- original width is positive
  (length_increase : l' = 1.3 * l) -- new length after increase
  (width_increase : w' = 1.15 * w) -- new width after increase
  (new_area : A' = l' * w') -- new area after increase
  (original_area : A = l * w) -- original area
  :
  ((A' / A) * 100 - 100) = 49.5 := by
  sorry

end rectangle_area_increase_l506_50648


namespace two_digit_factors_of_2_pow_18_minus_1_l506_50676

-- Define the main problem statement: 
-- How many two-digit factors does 2^18 - 1 have?

theorem two_digit_factors_of_2_pow_18_minus_1 : 
  ∃ n : ℕ, n = 5 ∧ ∀ f : ℕ, (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100) ↔ (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100 ∧ ∃ k : ℕ, (2^18 - 1) = k * f) :=
by sorry

end two_digit_factors_of_2_pow_18_minus_1_l506_50676


namespace boy_two_girls_work_completion_days_l506_50632

-- Work rates definitions
def man_work_rate := 1 / 6
def woman_work_rate := 1 / 18
def girl_work_rate := 1 / 12
def team_work_rate := 1 / 3

-- Boy's work rate
def boy_work_rate := 1 / 36

-- Combined work rate of boy and two girls
def boy_two_girls_work_rate := boy_work_rate + 2 * girl_work_rate

-- Prove that the number of days it will take for a boy and two girls to complete the work is 36 / 7
theorem boy_two_girls_work_completion_days : (1 / boy_two_girls_work_rate) = 36 / 7 :=
by
  sorry

end boy_two_girls_work_completion_days_l506_50632


namespace loss_eq_cost_price_of_x_balls_l506_50603

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end loss_eq_cost_price_of_x_balls_l506_50603


namespace probability_of_one_defective_l506_50690

theorem probability_of_one_defective :
  (2 : ℕ) ≤ 5 → (0 : ℕ) ≤ 2 → (0 : ℕ) ≤ 3 →
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = (3 / 5 : ℚ) :=
by
  intros h1 h2 h3
  let total_outcomes := Nat.choose 5 2
  let favorable_outcomes := Nat.choose 3 1 * Nat.choose 2 1
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  have : total_outcomes = 10 := by sorry
  have : favorable_outcomes = 6 := by sorry
  have : probability = (6 / 10 : ℚ) := by sorry
  have : (6 / 10 : ℚ) = (3 / 5 : ℚ) := by sorry
  exact this

end probability_of_one_defective_l506_50690


namespace y_intercept_of_line_l506_50692

theorem y_intercept_of_line :
  ∃ y : ℝ, (∃ x : ℝ, x = 0 ∧ 2 * x - 3 * y = 6) ∧ y = -2 :=
sorry

end y_intercept_of_line_l506_50692


namespace mul_large_numbers_l506_50671

theorem mul_large_numbers : 300000 * 300000 * 3 = 270000000000 := by
  sorry

end mul_large_numbers_l506_50671


namespace wire_around_field_l506_50637

theorem wire_around_field (A L : ℕ) (hA : A = 69696) (hL : L = 15840) : L / (4 * (Nat.sqrt A)) = 15 :=
by
  sorry

end wire_around_field_l506_50637


namespace polynomial_factorization_l506_50682

variable (x y : ℝ)

theorem polynomial_factorization (m : ℝ) :
  (∃ (a b : ℝ), 6 * x^2 - 5 * x * y - 4 * y^2 - 11 * x + 22 * y + m = (3 * x - 4 * y + a) * (2 * x + y + b)) →
  m = -10 :=
sorry

end polynomial_factorization_l506_50682


namespace percentage_earth_fresh_water_l506_50670

theorem percentage_earth_fresh_water :
  let portion_land := 3 / 10
  let portion_water := 1 - portion_land
  let percent_salt_water := 97 / 100
  let percent_fresh_water := 1 - percent_salt_water
  100 * (portion_water * percent_fresh_water) = 2.1 :=
by
  sorry

end percentage_earth_fresh_water_l506_50670


namespace plums_for_20_oranges_l506_50642

noncomputable def oranges_to_pears (oranges : ℕ) : ℕ :=
  (oranges / 5) * 3

noncomputable def pears_to_plums (pears : ℕ) : ℕ :=
  (pears / 4) * 6

theorem plums_for_20_oranges :
  oranges_to_pears 20 = 12 ∧ pears_to_plums 12 = 18 :=
by
  sorry

end plums_for_20_oranges_l506_50642


namespace student_weekly_allowance_l506_50664

theorem student_weekly_allowance (A : ℝ) :
  (3 / 4) * (1 / 3) * ((2 / 5) * A + 4) - 2 = 0 ↔ A = 100/3 := sorry

end student_weekly_allowance_l506_50664


namespace monochromatic_triangle_probability_l506_50614

noncomputable def probability_monochromatic_triangle : ℚ := sorry

theorem monochromatic_triangle_probability :
  -- Condition: Each of the 6 sides and the 9 diagonals of a regular hexagon are randomly and independently colored red, blue, or green with equal probability.
  -- Proof: The probability that at least one triangle whose vertices are among the vertices of the hexagon has all its sides of the same color is equal to 872/1000.
  probability_monochromatic_triangle = 872 / 1000 :=
sorry

end monochromatic_triangle_probability_l506_50614


namespace coconut_grove_l506_50638

theorem coconut_grove (x : ℕ) :
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) = 300 * x → x = 2 :=
by
  intro h
  -- We can leave the proof part to prove this later.
  sorry

end coconut_grove_l506_50638


namespace area_covered_by_three_layers_l506_50679

theorem area_covered_by_three_layers (A B C : ℕ) (total_wallpaper : ℕ := 300)
  (wall_area : ℕ := 180) (two_layer_coverage : ℕ := 30) :
  A + 2 * B + 3 * C = total_wallpaper ∧ B + C = total_wallpaper - wall_area ∧ B = two_layer_coverage → 
  C = 90 :=
by
  sorry

end area_covered_by_three_layers_l506_50679


namespace triangle_properties_l506_50605

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (angle_A : A = 30) (angle_B : B = 45) (side_a : a = Real.sqrt 2) :
  b = 2 ∧ (1 / 2) * a * b * Real.sin (105 * Real.pi / 180) = (Real.sqrt 3 + 1) / 2 := by
sorry

end triangle_properties_l506_50605


namespace candy_bars_per_friend_l506_50640

-- Definitions based on conditions
def total_candy_bars : ℕ := 24
def spare_candy_bars : ℕ := 10
def number_of_friends : ℕ := 7

-- The problem statement as a Lean theorem
theorem candy_bars_per_friend :
  (total_candy_bars - spare_candy_bars) / number_of_friends = 2 := 
by
  sorry

end candy_bars_per_friend_l506_50640


namespace max_squares_covered_l506_50667

theorem max_squares_covered 
    (board_square_side : ℝ) 
    (card_side : ℝ) 
    (n : ℕ) 
    (h1 : board_square_side = 1) 
    (h2 : card_side = 2) 
    (h3 : ∀ x y : ℝ, (x*x + y*y ≤ card_side*card_side) → card_side*card_side ≤ 4) :
    n ≤ 9 := sorry

end max_squares_covered_l506_50667


namespace bill_new_win_percentage_l506_50611

theorem bill_new_win_percentage :
  ∀ (initial_games : ℕ) (initial_win_percentage : ℚ) (additional_games : ℕ) (losses_in_additional_games : ℕ),
  initial_games = 200 →
  initial_win_percentage = 0.63 →
  additional_games = 100 →
  losses_in_additional_games = 43 →
  ((initial_win_percentage * initial_games + (additional_games - losses_in_additional_games)) / (initial_games + additional_games)) * 100 = 61 := 
by
  intros initial_games initial_win_percentage additional_games losses_in_additional_games h1 h2 h3 h4
  sorry

end bill_new_win_percentage_l506_50611


namespace eggs_left_over_l506_50608

theorem eggs_left_over (David_eggs Ella_eggs Fiona_eggs : ℕ)
  (hD : David_eggs = 45)
  (hE : Ella_eggs = 58)
  (hF : Fiona_eggs = 29) :
  (David_eggs + Ella_eggs + Fiona_eggs) % 10 = 2 :=
by
  sorry

end eggs_left_over_l506_50608


namespace f_of_1789_l506_50674

-- Definitions as per conditions
def f : ℕ → ℕ := sorry -- This will be the function definition satisfying the conditions

axiom f_f_n (n : ℕ) (h : n > 0) : f (f n) = 4 * n + 9
axiom f_2_k (k : ℕ) : f (2^k) = 2^(k+1) + 3

-- Prove f(1789) = 3581 given the conditions.
theorem f_of_1789 : f 1789 = 3581 := 
sorry

end f_of_1789_l506_50674


namespace consecutive_even_sum_l506_50661

theorem consecutive_even_sum : 
  ∃ n : ℕ, 
  (∃ x : ℕ, (∀ i : ℕ, i < n → (2 * i + x = 14 → i = 2) → 
  2 * x + (n - 1) * n = 52) ∧ n = 4) :=
by
  sorry

end consecutive_even_sum_l506_50661


namespace arithmetic_sequence_term_l506_50618

theorem arithmetic_sequence_term :
  ∀ a : ℕ → ℕ, (a 1 = 1) → (∀ n : ℕ, a (n + 1) - a n = 2) → (a 6 = 11) :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_term_l506_50618


namespace ratio_smaller_triangle_to_trapezoid_area_l506_50630

theorem ratio_smaller_triangle_to_trapezoid_area (a b : ℕ) (sqrt_three : ℝ) 
  (h_a : a = 10) (h_b : b = 2) (h_sqrt_three : sqrt_three = Real.sqrt 3) :
  ( ( (sqrt_three / 4 * (b ^ 2)) / 
      ( (sqrt_three / 4 * (a ^ 2)) - 
         (sqrt_three / 4 * (b ^ 2)))) = 1 / 24 ) := 
by
  -- conditions from the problem
  have h1: a = 10 := by exact h_a
  have h2: b = 2 := by exact h_b
  have h3: sqrt_three = Real.sqrt 3 := by exact h_sqrt_three
  sorry

end ratio_smaller_triangle_to_trapezoid_area_l506_50630


namespace Tabitha_age_proof_l506_50641

variable (Tabitha_age current_hair_colors: ℕ)
variable (Adds_new_color_per_year: ℕ)
variable (initial_hair_colors: ℕ)
variable (years_passed: ℕ)

theorem Tabitha_age_proof (h1: Adds_new_color_per_year = 1)
                          (h2: initial_hair_colors = 2)
                          (h3: ∀ years_passed, Tabitha_age  = 15 + years_passed)
                          (h4: Adds_new_color_per_year  = 1 )
                          (h5: current_hair_colors =  8 - 3)
                          (h6: current_hair_colors  =  initial_hair_colors + 3)
                          : Tabitha_age = 18 := 
by {
  sorry  -- Proof omitted
}

end Tabitha_age_proof_l506_50641


namespace fish_weight_l506_50646

theorem fish_weight (θ H T : ℝ) (h1 : θ = 4) (h2 : H = θ + 0.5 * T) (h3 : T = H + θ) : H + T + θ = 32 :=
by
  sorry

end fish_weight_l506_50646


namespace distinct_paper_count_l506_50613

theorem distinct_paper_count (n : ℕ) :
  let sides := 4  -- 4 rotations and 4 reflections
  let identity_fixed := n^25 
  let rotation_90_fixed := n^7
  let rotation_270_fixed := n^7
  let rotation_180_fixed := n^13
  let reflection_fixed := n^15
  (1 / 8) * (identity_fixed + 4 * reflection_fixed + rotation_180_fixed + 2 * rotation_90_fixed) 
  = (1 / 8) * (n^25 + 4 * n^15 + n^13 + 2 * n^7) :=
  by 
    sorry

end distinct_paper_count_l506_50613


namespace range_of_a_l506_50685

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l506_50685


namespace max_value_of_expression_l506_50644

variable (a b c d : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : a + b + c + d = 3)

theorem max_value_of_expression :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end max_value_of_expression_l506_50644


namespace original_radius_l506_50615

theorem original_radius (r : Real) (h : Real) (z : Real) 
  (V : Real) (Vh : Real) (Vr : Real) :
  h = 3 → 
  V = π * r^2 * h → 
  Vh = π * r^2 * (h + 3) → 
  Vr = π * (r + 3)^2 * h → 
  Vh - V = z → 
  Vr - V = z →
  r = 3 + 3 * Real.sqrt 2 :=
by
  sorry

end original_radius_l506_50615


namespace initial_cookie_count_l506_50652

variable (cookies_left_after_week : ℕ)
variable (cookies_taken_each_day : ℕ)
variable (total_cookies_taken_in_four_days : ℕ)
variable (initial_cookies : ℕ)
variable (days_per_week : ℕ)

theorem initial_cookie_count :
  cookies_left_after_week = 28 →
  total_cookies_taken_in_four_days = 24 →
  days_per_week = 7 →
  (∀ d (h : d ∈ Finset.range days_per_week), cookies_taken_each_day = 6) →
  initial_cookies = 52 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_cookie_count_l506_50652


namespace beanie_babies_total_l506_50668

theorem beanie_babies_total
  (Lori_beanie_babies : ℕ) (Sydney_beanie_babies : ℕ)
  (h1 : Lori_beanie_babies = 15 * Sydney_beanie_babies)
  (h2 : Lori_beanie_babies = 300) :
  Lori_beanie_babies + Sydney_beanie_babies = 320 :=
sorry

end beanie_babies_total_l506_50668


namespace culture_growth_l506_50697

/-- Define the initial conditions and growth rates of the bacterial culture -/
def initial_cells : ℕ := 5

def growth_rate1 : ℕ := 3
def growth_rate2 : ℕ := 2

def cycle_duration : ℕ := 3
def first_phase_duration : ℕ := 6
def second_phase_duration : ℕ := 6

def total_duration : ℕ := 12

/-- Define the hypothesis that calculates the number of cells at any point in time based on the given rules -/
theorem culture_growth : 
    (initial_cells * growth_rate1^ (first_phase_duration / cycle_duration) 
    * growth_rate2^ (second_phase_duration / cycle_duration)) = 180 := 
sorry

end culture_growth_l506_50697


namespace seeds_total_l506_50623

noncomputable def seeds_planted (x : ℕ) (y : ℕ) (z : ℕ) : ℕ :=
x + y + z

theorem seeds_total (x : ℕ) (H1 :  y = 5 * x) (H2 : x + y = 156) (z : ℕ) 
(H3 : z = 4) : seeds_planted x y z = 160 :=
by
  sorry

end seeds_total_l506_50623


namespace circle_ratio_l506_50653

theorem circle_ratio (R r : ℝ) (h₁ : R > 0) (h₂ : r > 0) 
                     (h₃ : π * R^2 - π * r^2 = 3 * π * r^2) : R = 2 * r :=
by
  sorry

end circle_ratio_l506_50653


namespace person_A_work_days_l506_50647

theorem person_A_work_days (A : ℕ) (h1 : ∀ (B : ℕ), B = 45) (h2 : 4 * (1/A + 1/45) = 2/9) : A = 30 := 
by
  sorry

end person_A_work_days_l506_50647


namespace find_angle_y_l506_50621

theorem find_angle_y (ABC BAC BCA DCE CED y : ℝ)
  (h1 : ABC = 80) (h2 : BAC = 60)
  (h3 : ABC + BAC + BCA = 180)
  (h4 : CED = 90)
  (h5 : DCE = BCA)
  (h6 : DCE + CED + y = 180) :
  y = 50 :=
by
  sorry

end find_angle_y_l506_50621


namespace pupils_like_burgers_total_l506_50631

theorem pupils_like_burgers_total (total_pupils pizza_lovers both_lovers : ℕ) :
  total_pupils = 200 →
  pizza_lovers = 125 →
  both_lovers = 40 →
  (pizza_lovers - both_lovers) + (total_pupils - pizza_lovers - both_lovers) + both_lovers = 115 :=
by
  intros h_total h_pizza h_both
  rw [h_total, h_pizza, h_both]
  sorry

end pupils_like_burgers_total_l506_50631


namespace same_terminal_side_angle_l506_50622

theorem same_terminal_side_angle (k : ℤ) : 
  (∃ k : ℤ, - (π / 6) = 2 * k * π + a) → a = 11 * π / 6 :=
sorry

end same_terminal_side_angle_l506_50622


namespace part_i_l506_50633

theorem part_i (n : ℤ) : (∃ k : ℤ, n = 225 * k + 99) ↔ (n % 9 = 0 ∧ (n + 1) % 25 = 0) :=
by 
  sorry

end part_i_l506_50633


namespace correct_option_D_l506_50607

noncomputable def total_students := 40
noncomputable def male_students := 25
noncomputable def female_students := 15
noncomputable def class_president := 1
noncomputable def prob_class_president := class_president / total_students
noncomputable def prob_class_president_from_females := 0

theorem correct_option_D
  (h1 : total_students = 40)
  (h2 : male_students = 25)
  (h3 : female_students = 15)
  (h4 : class_president = 1) :
  prob_class_president = 1 / 40 ∧ prob_class_president_from_females = 0 := 
by
  sorry

end correct_option_D_l506_50607


namespace twelfth_term_is_three_l506_50620

-- Define the first term and the common difference of the arithmetic sequence
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 4

-- Define the nth term of an arithmetic sequence
def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

-- Prove that the twelfth term is equal to 3
theorem twelfth_term_is_three : nth_term first_term common_difference 12 = 3 := 
  by 
    sorry

end twelfth_term_is_three_l506_50620


namespace probability_two_white_balls_is_4_over_15_l506_50604

-- Define the conditions of the problem
def total_balls : ℕ := 15
def white_balls_initial : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 2 -- Note: Even though not explicitly required, it's part of the context

-- Calculate the probability of drawing two white balls without replacement
noncomputable def probability_two_white_balls : ℚ :=
  (white_balls_initial / total_balls) * ((white_balls_initial - 1) / (total_balls - 1))

-- The theorem to prove
theorem probability_two_white_balls_is_4_over_15 :
  probability_two_white_balls = 4 / 15 := by
  sorry

end probability_two_white_balls_is_4_over_15_l506_50604


namespace todd_money_left_l506_50645

-- Define the initial amount of money Todd has
def initial_amount : ℕ := 20

-- Define the number of candy bars Todd buys
def number_of_candy_bars : ℕ := 4

-- Define the cost per candy bar
def cost_per_candy_bar : ℕ := 2

-- Define the total cost of the candy bars
def total_cost : ℕ := number_of_candy_bars * cost_per_candy_bar

-- Define the final amount of money Todd has left
def final_amount : ℕ := initial_amount - total_cost

-- The statement to be proven in Lean
theorem todd_money_left : final_amount = 12 := by
  -- The proof is omitted
  sorry

end todd_money_left_l506_50645


namespace sum_of_angles_l506_50665

theorem sum_of_angles (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ = 67.5) (h₂ : θ₂ = 157.5) (h₃ : θ₃ = 247.5) (h₄ : θ₄ = 337.5) :
  θ₁ + θ₂ + θ₃ + θ₄ = 810 :=
by
  -- These parameters are used only to align with provided conditions
  let r₁ := 1
  let r₂ := r₁
  let r₃ := r₁
  let r₄ := r₁
  have z₁ := r₁ * (Complex.cos θ₁ + Complex.sin θ₁ * Complex.I)
  have z₂ := r₂ * (Complex.cos θ₂ + Complex.sin θ₂ * Complex.I)
  have z₃ := r₃ * (Complex.cos θ₃ + Complex.sin θ₃ * Complex.I)
  have z₄ := r₄ * (Complex.cos θ₄ + Complex.sin θ₄ * Complex.I)
  sorry

end sum_of_angles_l506_50665


namespace problem_1_l506_50656

theorem problem_1 (a : ℝ) : (1 + a * x) * (1 + x) ^ 5 = 1 + 5 * x + 5 * i * x^2 → a = -1 := sorry

end problem_1_l506_50656


namespace find_A_l506_50606

theorem find_A (A B : ℕ) (h1 : 10 * A + 7 + (30 + B) = 73) : A = 3 := by
  sorry

end find_A_l506_50606
