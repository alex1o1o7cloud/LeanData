import Mathlib

namespace num_occupied_third_floor_rooms_l52_52540

-- Definitions based on conditions
def first_floor_rent : Int := 15
def second_floor_rent : Int := 20
def third_floor_rent : Int := 2 * first_floor_rent
def rooms_per_floor : Int := 3
def monthly_earnings : Int := 165

-- The proof statement
theorem num_occupied_third_floor_rooms : 
  let total_full_occupancy_cost := rooms_per_floor * first_floor_rent + rooms_per_floor * second_floor_rent + rooms_per_floor * third_floor_rent
  let revenue_difference := total_full_occupancy_cost - monthly_earnings
  revenue_difference / third_floor_rent = 1 → rooms_per_floor - revenue_difference / third_floor_rent = 2 :=
by
  sorry

end num_occupied_third_floor_rooms_l52_52540


namespace gcd_9247_4567_eq_1_l52_52151

theorem gcd_9247_4567_eq_1 : Int.gcd 9247 4567 = 1 := sorry

end gcd_9247_4567_eq_1_l52_52151


namespace minyoung_gave_nine_notebooks_l52_52697

theorem minyoung_gave_nine_notebooks (original left given : ℕ) (h1 : original = 17) (h2 : left = 8) (h3 : given = original - left) : given = 9 :=
by
  rw [h1, h2] at h3
  exact h3

end minyoung_gave_nine_notebooks_l52_52697


namespace marc_watching_episodes_l52_52558

theorem marc_watching_episodes :
  ∀ (n : ℕ) (f : ℝ),
  n = 50 → f = 1/10 → 
  n / (n * f) = 10 := 
by
  intro n f hn hf
  rw [hn, hf]
  norm_num
  sorry

end marc_watching_episodes_l52_52558


namespace maximize_probability_when_C_second_game_l52_52439

variable {p1 p2 p3 : ℝ}
variables (h1 : p1 > 0) (h2 : p2 > p1) (h3 : p3 > p2)

noncomputable def P_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def P_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_when_C_second_game : P_C > P_A ∧ P_C > P_B :=
by { sorry }

end maximize_probability_when_C_second_game_l52_52439


namespace water_park_admission_l52_52585

def adult_admission_charge : ℝ := 1
def child_admission_charge : ℝ := 0.75
def children_accompanied : ℕ := 3
def total_admission_charge (adults : ℝ) (children : ℝ) : ℝ := adults + children

theorem water_park_admission :
  let adult_charge := adult_admission_charge
  let children_charge := children_accompanied * child_admission_charge
  total_admission_charge adult_charge children_charge = 3.25 :=
by sorry

end water_park_admission_l52_52585


namespace carol_lollipops_l52_52049

theorem carol_lollipops (total_lollipops : ℝ) (first_day_lollipops : ℝ) (delta_lollipops : ℝ) :
  total_lollipops = 150 → delta_lollipops = 5 →
  (first_day_lollipops + (first_day_lollipops + 5) + (first_day_lollipops + 10) +
  (first_day_lollipops + 15) + (first_day_lollipops + 20) + (first_day_lollipops + 25) = total_lollipops) →
  (first_day_lollipops = 12.5) →
  (first_day_lollipops + 15 = 27.5) :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end carol_lollipops_l52_52049


namespace cupboard_cost_price_l52_52417

noncomputable def cost_price_of_cupboard (C : ℝ) : Prop :=
  let SP := 0.88 * C
  let NSP := 1.12 * C
  NSP - SP = 1650

theorem cupboard_cost_price : ∃ (C : ℝ), cost_price_of_cupboard C ∧ C = 6875 := by
  sorry

end cupboard_cost_price_l52_52417


namespace contradiction_example_l52_52733

theorem contradiction_example (a b c d : ℝ) 
(h1 : a + b = 1) 
(h2 : c + d = 1) 
(h3 : ac + bd > 1) : 
¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_example_l52_52733


namespace min_digits_fraction_l52_52420

theorem min_digits_fraction : 
  let num := 987654321
  let denom := 2^27 * 5^3
  ∃ (digits : ℕ), (10^digits * num = 987654321 * 2^27 * 5^3) ∧ digits = 27 := 
by
  sorry

end min_digits_fraction_l52_52420


namespace probability_difference_one_is_one_third_l52_52780

open Finset

-- Let an urn contain 6 balls numbered from 1 to 6
def urn := {1, 2, 3, 4, 5, 6}

-- Define the event that two balls are drawn
def two_balls : Finset (ℕ × ℕ) := {(x, y) | x ∈ urn ∧ y ∈ urn ∧ x < y}

-- Define the event that the difference between two drawn balls is 1
def difference_one (x y : ℕ) : Prop := abs (x - y) = 1

-- Calculate the total number of ways to draw two balls from the urn
def total_draws : ℕ := two_balls.card

-- Calculate the number of favorable outcomes
def favorable_draws : ℕ := (two_balls.filter (λ (p : ℕ × ℕ), difference_one p.1 p.2)).card

-- Calculate the probability of the difference being 1
noncomputable def probability_difference_one : ℚ := favorable_draws / total_draws

-- State the theorem
theorem probability_difference_one_is_one_third :
  probability_difference_one = 1 / 3 :=
sorry

end probability_difference_one_is_one_third_l52_52780


namespace correct_option_D_l52_52106

def U : Set ℕ := {1, 2, 4, 6, 8}
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

theorem correct_option_D : A ∩ complement_U_B = {1} := by
  sorry

end correct_option_D_l52_52106


namespace logarithmic_expression_l52_52302

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression :
  let log2 := lg 2
  let log5 := lg 5
  log2 + log5 = 1 →
  (log2^3 + 3 * log2 * log5 + log5^3 = 1) :=
by
  intros log2 log5 h
  sorry

end logarithmic_expression_l52_52302


namespace total_spent_l52_52416

theorem total_spent (deck_price : ℕ) (victor_decks : ℕ) (friend_decks : ℕ)
  (h1 : deck_price = 8)
  (h2 : victor_decks = 6)
  (h3 : friend_decks = 2) :
  deck_price * victor_decks + deck_price * friend_decks = 64 :=
by
  sorry

end total_spent_l52_52416


namespace repeating_decimal_sum_l52_52016

theorem repeating_decimal_sum : 
  let x := 0.123123 in
  let frac := 41 / 333 in
  sum_n_d (frac) = 374 :=
begin
  sorry
end

end repeating_decimal_sum_l52_52016


namespace largest_both_writers_editors_l52_52783

-- Define the conditions
def writers : ℕ := 45
def editors_gt : ℕ := 38
def total_attendees : ℕ := 90
def both_writers_editors (x : ℕ) : ℕ := x
def neither_writers_editors (x : ℕ) : ℕ := x / 2

-- Define the main proof statement
theorem largest_both_writers_editors :
  ∃ x : ℕ, x ≤ 4 ∧
  (writers + (editors_gt + (0 : ℕ)) + neither_writers_editors x + both_writers_editors x = total_attendees) :=
sorry

end largest_both_writers_editors_l52_52783


namespace min_value_exponential_sub_l52_52807

theorem min_value_exponential_sub (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : x + 2 * y = x * y) : ∃ y₀ > 0, ∀ y > 1, e^y - 8 / x ≥ e :=
by
  sorry

end min_value_exponential_sub_l52_52807


namespace function_intersection_at_most_one_l52_52132

theorem function_intersection_at_most_one (f : ℝ → ℝ) (a : ℝ) :
  ∃! b, f b = a := sorry

end function_intersection_at_most_one_l52_52132


namespace count_possible_third_side_lengths_l52_52672

theorem count_possible_third_side_lengths : ∀ (n : ℤ), 2 < n ∧ n < 14 → ∃ s : Finset ℤ, s.card = 11 ∧ ∀ x ∈ s, 2 < x ∧ x < 14 := by
  sorry

end count_possible_third_side_lengths_l52_52672


namespace CDs_per_rack_l52_52311

theorem CDs_per_rack (racks_on_shelf : ℕ) (CDs_on_shelf : ℕ) (h1 : racks_on_shelf = 4) (h2 : CDs_on_shelf = 32) : 
  CDs_on_shelf / racks_on_shelf = 8 :=
by
  sorry

end CDs_per_rack_l52_52311


namespace range_of_f_l52_52798

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f : Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) = Set.range f :=
  sorry

end range_of_f_l52_52798


namespace Buffy_whiskers_l52_52206

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l52_52206


namespace compute_y_value_l52_52188

theorem compute_y_value : 
  (∑' n : ℕ, (1 / 3)^n) * (∑' n : ℕ, (-1 / 3)^n) = ∑' n : ℕ, (1 / (9 : ℝ))^n := 
by 
  sorry

end compute_y_value_l52_52188


namespace angle_A_range_l52_52686

open Real

theorem angle_A_range (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) (h3 : 0 < A ∧ A < π) : 
  π / 2 < A ∧ A < 3 * π / 4 :=
by
  sorry

end angle_A_range_l52_52686


namespace vector_addition_l52_52806

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_addition :
  2 • a + b = (1, 2) :=
by
  sorry

end vector_addition_l52_52806


namespace find_a3_in_arith_geo_seq_l52_52215

theorem find_a3_in_arith_geo_seq
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : S 6 / S 3 = -19 / 8)
  (h2 : a 4 - a 2 = -15 / 8) :
  a 3 = 9 / 4 :=
sorry

end find_a3_in_arith_geo_seq_l52_52215


namespace cubic_roots_inequality_l52_52543

theorem cubic_roots_inequality (a b c : ℝ) (h : ∃ (α β γ : ℝ), (x : ℝ) → x^3 + a * x^2 + b * x + c = (x - α) * (x - β) * (x - γ)) :
  3 * b ≤ a^2 :=
sorry

end cubic_roots_inequality_l52_52543


namespace loss_percentage_l52_52632

theorem loss_percentage (C S : ℕ) (H1 : C = 750) (H2 : S = 600) : (C - S) * 100 / C = 20 := by
  sorry

end loss_percentage_l52_52632


namespace distance_between_feet_of_perpendiculars_eq_area_over_radius_l52_52565
noncomputable def area (ABC : Type) : ℝ := sorry
noncomputable def circumradius (ABC : Type) : ℝ := sorry

theorem distance_between_feet_of_perpendiculars_eq_area_over_radius
  (ABC : Type)
  (area_ABC : ℝ)
  (R : ℝ)
  (h_area : area ABC = area_ABC)
  (h_radius : circumradius ABC = R) :
  ∃ (m : ℝ), m = area_ABC / R := sorry

end distance_between_feet_of_perpendiculars_eq_area_over_radius_l52_52565


namespace compute_fraction_l52_52015

theorem compute_fraction : ((5 * 7) - 3) / 9 = 32 / 9 := by
  sorry

end compute_fraction_l52_52015


namespace sum_of_last_two_digits_of_fibonacci_factorial_series_l52_52609

def last_two_digits (n : Nat) : Nat :=
  n % 100

def relevant_factorials : List Nat := [
  last_two_digits (Nat.factorial 1),
  last_two_digits (Nat.factorial 1),
  last_two_digits (Nat.factorial 2),
  last_two_digits (Nat.factorial 3),
  last_two_digits (Nat.factorial 5),
  last_two_digits (Nat.factorial 8)
]

def sum_last_two_digits : Nat :=
  relevant_factorials.sum

theorem sum_of_last_two_digits_of_fibonacci_factorial_series :
  sum_last_two_digits = 5 := by
  sorry

end sum_of_last_two_digits_of_fibonacci_factorial_series_l52_52609


namespace nancy_small_gardens_l52_52559

theorem nancy_small_gardens (total_seeds big_garden_seeds small_garden_seed_count : ℕ) 
    (h1 : total_seeds = 52) 
    (h2 : big_garden_seeds = 28) 
    (h3 : small_garden_seed_count = 4) : 
    (total_seeds - big_garden_seeds) / small_garden_seed_count = 6 := by 
    sorry

end nancy_small_gardens_l52_52559


namespace triangle_inequality_for_powers_l52_52418

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, (a ^ n + b ^ n > c ^ n)) ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b = c) :=
sorry

end triangle_inequality_for_powers_l52_52418


namespace xyz_value_l52_52943

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
  sorry

end xyz_value_l52_52943


namespace sector_area_l52_52814

-- Define the properties and conditions
def perimeter_of_sector (r l : ℝ) : Prop :=
  l + 2 * r = 8

def central_angle_arc_length (r : ℝ) : ℝ :=
  2 * r

-- Theorem to prove the area of the sector
theorem sector_area (r : ℝ) (l : ℝ) 
  (h_perimeter : perimeter_of_sector r l) 
  (h_arc_length : l = central_angle_arc_length r) : 
  1 / 2 * l * r = 4 := 
by
  -- This is the place where the proof would go; we use sorry to indicate it's incomplete
  sorry

end sector_area_l52_52814


namespace roots_poly_cond_l52_52846

theorem roots_poly_cond (α β p q γ δ : ℝ) 
  (h1 : α ^ 2 + p * α - 1 = 0) 
  (h2 : β ^ 2 + p * β - 1 = 0) 
  (h3 : γ ^ 2 + q * γ - 1 = 0) 
  (h4 : δ ^ 2 + q * δ - 1 = 0)
  (h5 : γ * δ = -1) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = -(p - q) ^ 2 := 
by 
  sorry

end roots_poly_cond_l52_52846


namespace sum_reciprocal_l52_52332

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end sum_reciprocal_l52_52332


namespace original_problem_theorem_l52_52547

-- Define the third sequence in general
def third_sequence_term (j : ℕ) (n : ℕ) (hj : 1 ≤ j ∧ j ≤ n-2) : ℕ :=
  4 * j + 4

-- Define the term in the last sequence given the provided recurrence relation
noncomputable def last_sequence_term (n : ℕ) : ℕ :=
  (n + 1) * 2^(n - 2)

-- Main theorem combining both parts
theorem original_problem_theorem (n j : ℕ) (hn : 1 ≤ n) (hj : 1 ≤ j ∧ j ≤ n-2) :
  third_sequence_term j n hj = 4 * j + 4 ∧ last_sequence_term n = (n + 1) * 2^(n - 2) :=
begin
  split,
  { -- Proof for the third sequence term is not needed
    sorry },
  { -- Proof for the term in the last sequence is not needed
    sorry },
end

end original_problem_theorem_l52_52547


namespace part1_solution_set_part2_range_of_a_l52_52949

noncomputable def f (x a : ℝ) : ℝ := -x^2 + a * x + 4

def g (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem part1_solution_set (a : ℝ := 1) :
  {x : ℝ | f x a ≥ g x} = { x : ℝ | -1 ≤ x ∧ x ≤ (Real.sqrt 17 - 1) / 2 } :=
by
  sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x ∈ [-1,1], f x a ≥ g x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end part1_solution_set_part2_range_of_a_l52_52949


namespace units_digit_uniform_l52_52059

-- Definitions
def domain : Finset ℕ := Finset.range 15

def pick : Type := { n // n ∈ domain }

def uniform_pick : pick := sorry

-- Statement of the theorem
theorem units_digit_uniform :
  ∀ (J1 J2 K : pick), 
  ∃ d : ℕ, d < 10 ∧ (J1.val + J2.val + K.val) % 10 = d
:= sorry

end units_digit_uniform_l52_52059


namespace tiling_vertex_squares_octagons_l52_52922

theorem tiling_vertex_squares_octagons (m n : ℕ) 
  (h1 : 135 * n + 90 * m = 360) : 
  m = 1 ∧ n = 2 :=
by
  sorry

end tiling_vertex_squares_octagons_l52_52922


namespace Marc_watch_episodes_l52_52557

theorem Marc_watch_episodes : ∀ (episodes per_day : ℕ), episodes = 50 → per_day = episodes / 10 → (episodes / per_day) = 10 :=
by
  intros episodes per_day h1 h2
  sorry

end Marc_watch_episodes_l52_52557


namespace simplify_expression_l52_52504

theorem simplify_expression :
  ((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4) = 125 / 12 := by
  sorry

end simplify_expression_l52_52504


namespace find_angle_A_l52_52350

theorem find_angle_A (a b : ℝ) (B A : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = Real.pi / 4) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l52_52350


namespace tom_strokes_over_par_l52_52895

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l52_52895


namespace max_prob_two_consecutive_wins_l52_52437

/-
Given probabilities of winning against A, B, and C are p1, p2, and p3 respectively,
and p3 > p2 > p1 > 0, prove that the probability of winning two consecutive games
is maximum when the chess player plays against C in the second game.
-/

variables {p1 p2 p3 : ℝ}
variables (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

theorem max_prob_two_consecutive_wins :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in PC > PA ∧ PC > PB :=
by {
    sorry
}

end max_prob_two_consecutive_wins_l52_52437


namespace MargaretsMeanScore_l52_52212

theorem MargaretsMeanScore :
  ∀ (scores : List ℕ)
    (cyprian_mean : ℝ)
    (highest_lowest_different : Prop),
    scores = [82, 85, 88, 90, 92, 95, 97, 99] →
    cyprian_mean = 88.5 →
    highest_lowest_different →
    ∃ (margaret_mean : ℝ), margaret_mean = 93.5 := by
  sorry

end MargaretsMeanScore_l52_52212


namespace tower_no_knights_l52_52390

-- Define the problem conditions in Lean

variable {T : Type} -- Type for towers
variable {K : Type} -- Type for knights

variable (towers : Fin 9 → T)
variable (knights : Fin 18 → K)

-- Movement of knights: each knight moves to a neighboring tower every hour (either clockwise or counterclockwise)
variable (moves : K → (T → T))

-- Each knight stands watch at each tower exactly once over the course of the night
variable (stands_watch : ∀ k : K, ∀ t : T, ∃ hour : Fin 9, moves k t = towers hour)

-- Condition: at one time (say hour 1), each tower had at least two knights on watch
variable (time1 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 1
variable (cond1 : ∀ i : Fin 9, 2 ≤ time1 1 i)

-- Condition: at another time (say hour 2), exactly five towers each had exactly one knight on watch
variable (time2 : Fin 9 → Fin 9 → ℕ) -- Number of knights at each tower at hour 2
variable (cond2 : ∃ seq : Fin 5 → Fin 9, (∀ i : Fin 5, time2 2 (seq i) = 1) ∧ ∀ j : Fin 4, i ≠ j → 1 ≠ seq j)

-- Prove: there exists a time when one of the towers had no knights at all
theorem tower_no_knights : ∃ hour : Fin 9, ∃ i : Fin 9, moves (knights i) (towers hour) = towers hour ∧ (∀ knight : K, moves knight (towers hour) ≠ towers hour) :=
sorry

end tower_no_knights_l52_52390


namespace factorize_expression_l52_52496

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end factorize_expression_l52_52496


namespace solve_y_l52_52310

theorem solve_y (y : ℝ) (h : (y ^ (7 / 8)) = 4) : y = 2 ^ (16 / 7) :=
sorry

end solve_y_l52_52310


namespace cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l52_52083

theorem cos_alpha_minus_11pi_div_12_eq_neg_2_div_3
  (α : ℝ)
  (h : Real.sin (7 * Real.pi / 12 + α) = 2 / 3) :
  Real.cos (α - 11 * Real.pi / 12) = -(2 / 3) :=
by
  sorry

end cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l52_52083


namespace daisies_per_bouquet_l52_52629

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end daisies_per_bouquet_l52_52629


namespace remainder_when_3x_7y_5z_div_31517_l52_52556

theorem remainder_when_3x_7y_5z_div_31517
  (x y z : ℕ)
  (hx : x % 23 = 9)
  (hy : y % 29 = 15)
  (hz : z % 37 = 12) :
  (3 * x + 7 * y - 5 * z) % 31517 = ((69 * (x / 23) + 203 * (y / 29) - 185 * (z / 37) + 72) % 31517) := 
sorry

end remainder_when_3x_7y_5z_div_31517_l52_52556


namespace problem_l52_52079

open Function

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + m) + a (n - m) = 2 * a n

theorem problem (h_arith : arithmetic_sequence a) (h_eq : a 1 + 3 * a 6 + a 11 = 10) :
  a 5 + a 7 = 4 := 
sorry

end problem_l52_52079


namespace toys_left_l52_52060

-- Given conditions
def initial_toys := 7
def sold_toys := 3

-- Proven statement
theorem toys_left : initial_toys - sold_toys = 4 := by
  sorry

end toys_left_l52_52060


namespace vehicle_speeds_l52_52148

theorem vehicle_speeds (V_A V_B V_C : ℝ) (d_AB d_AC : ℝ) (decel_A : ℝ)
  (V_A_eff : ℝ) (delta_V_A : ℝ) :
  V_A = 70 → V_B = 50 → V_C = 65 →
  decel_A = 5 → V_A_eff = V_A - decel_A → 
  d_AB = 40 → d_AC = 250 →
  delta_V_A = 10 →
  (d_AB / (V_A_eff + delta_V_A - V_B) < d_AC / (V_A_eff + delta_V_A + V_C)) :=
by
  intros hVA hVB hVC hdecel hV_A_eff hdAB hdAC hdelta_V_A
  -- the proof would be filled in here
  sorry

end vehicle_speeds_l52_52148


namespace part1_solution_set_part2_range_of_a_l52_52518

noncomputable def f (x : ℝ) : ℝ := abs (4 * x - 1) - abs (x + 2)

-- Part 1: Prove the solution set of f(x) < 8 is -9 / 5 < x < 11 / 3
theorem part1_solution_set : {x : ℝ | f x < 8} = {x : ℝ | -9 / 5 < x ∧ x < 11 / 3} :=
sorry

-- Part 2: Prove the range of a such that the inequality has a solution
theorem part2_range_of_a (a : ℝ) : (∃ x : ℝ, f x + 5 * abs (x + 2) < a^2 - 8 * a) ↔ (a < -1 ∨ a > 9) :=
sorry

end part1_solution_set_part2_range_of_a_l52_52518


namespace sail_pressure_l52_52133

def pressure (k A V : ℝ) : ℝ := k * A * V^2

theorem sail_pressure (k : ℝ)
  (h_k : k = 1 / 800) 
  (A : ℝ) 
  (V : ℝ) 
  (P : ℝ)
  (h_initial : A = 1 ∧ V = 20 ∧ P = 0.5) 
  (A2 : ℝ) 
  (V2 : ℝ) 
  (h_doubled : A2 = 2 ∧ V2 = 30) :
  pressure k A2 V2 = 2.25 :=
by
  sorry

end sail_pressure_l52_52133


namespace profit_percentage_l52_52754

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 75) : 
  ((selling_price - cost_price) / cost_price) * 100 = 25 := by
  sorry

end profit_percentage_l52_52754


namespace sarah_bus_time_l52_52705

noncomputable def totalTimeAway : ℝ := (4 + 15/60) + (5 + 15/60)  -- 9.5 hours
noncomputable def totalTimeAwayInMinutes : ℝ := totalTimeAway * 60  -- 570 minutes

noncomputable def timeInClasses : ℝ := 8 * 45  -- 360 minutes
noncomputable def timeInLunch : ℝ := 30  -- 30 minutes
noncomputable def timeInExtracurricular : ℝ := 1.5 * 60  -- 90 minutes
noncomputable def totalTimeInSchoolActivities : ℝ := timeInClasses + timeInLunch + timeInExtracurricular  -- 480 minutes

noncomputable def timeOnBus : ℝ := totalTimeAwayInMinutes - totalTimeInSchoolActivities  -- 90 minutes

theorem sarah_bus_time : timeOnBus = 90 := by
  sorry

end sarah_bus_time_l52_52705


namespace parallel_lines_intersect_parabola_l52_52897

theorem parallel_lines_intersect_parabola {a k b c x1 x2 x3 x4 : ℝ} 
    (h₁ : x1 < x2) 
    (h₂ : x3 < x4) 
    (intersect1 : ∀ y : ℝ, y = k * x1 + b ∧ y = a * x1^2 ∧ y = k * x2 + b ∧ y = a * x2^2) 
    (intersect2 : ∀ y : ℝ, y = k * x3 + c ∧ y = a * x3^2 ∧ y = k * x4 + c ∧ y = a * x4^2) :
    (x3 - x1) = (x2 - x4) := 
by 
    sorry

end parallel_lines_intersect_parabola_l52_52897


namespace ratio_a_d_l52_52246

variables (a b c d : ℕ)

-- Given conditions
def ratio_ab := 8 / 3
def ratio_bc := 1 / 5
def ratio_cd := 3 / 2
def b_value := 27

theorem ratio_a_d (h₁ : a / b = ratio_ab)
                  (h₂ : b / c = ratio_bc)
                  (h₃ : c / d = ratio_cd)
                  (h₄ : b = b_value) :
  a / d = 4 / 5 :=
sorry

end ratio_a_d_l52_52246


namespace twelve_factorial_mod_thirteen_l52_52076

theorem twelve_factorial_mod_thirteen : (12! % 13) = 12 := by
  sorry

end twelve_factorial_mod_thirteen_l52_52076


namespace xiao_ming_correctly_answered_question_count_l52_52038

-- Define the given conditions as constants and variables
def total_questions : ℕ := 20
def points_per_correct : ℕ := 8
def points_deducted_per_incorrect : ℕ := 5
def total_score : ℕ := 134

-- Prove that the number of correctly answered questions is 18
theorem xiao_ming_correctly_answered_question_count :
  ∃ (correct_count incorrect_count : ℕ), 
      correct_count + incorrect_count = total_questions ∧
      correct_count * points_per_correct - 
      incorrect_count * points_deducted_per_incorrect = total_score ∧
      correct_count = 18 :=
by
  sorry

end xiao_ming_correctly_answered_question_count_l52_52038


namespace remaining_amount_spent_on_watermelons_l52_52295

def pineapple_cost : ℕ := 7
def total_spent : ℕ := 38
def pineapples_purchased : ℕ := 2

theorem remaining_amount_spent_on_watermelons:
  total_spent - (pineapple_cost * pineapples_purchased) = 24 :=
by
  sorry

end remaining_amount_spent_on_watermelons_l52_52295


namespace find_starting_number_l52_52871

theorem find_starting_number : 
  ∃ x : ℕ, (∀ k : ℕ, (k < 12 → (x + 3 * k) ≤ 46) ∧ 12 = (46 - x) / 3 + 1) 
  ∧ x = 12 := 
by 
  sorry

end find_starting_number_l52_52871


namespace solution_set_of_inequality_l52_52408

theorem solution_set_of_inequality:
  {x : ℝ | 1 < abs (2 * x - 1) ∧ abs (2 * x - 1) < 3} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ 
  {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l52_52408


namespace max_value_f_max_value_f_at_13_l52_52715

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_f : ∀ x : ℝ, f x ≤ 1 / 3 := by
  sorry

theorem max_value_f_at_13 : ∃ x : ℝ, f x = 1 / 3 := by
  sorry

end max_value_f_max_value_f_at_13_l52_52715


namespace complex_number_solution_l52_52514

open Complex

theorem complex_number_solution (z : ℂ) (h : (1 + I) * z = 2 * I) : z = 1 + I :=
sorry

end complex_number_solution_l52_52514


namespace average_speed_distance_div_time_l52_52436

theorem average_speed_distance_div_time (distance : ℕ) (time_minutes : ℕ) (average_speed : ℕ) : 
  distance = 8640 → time_minutes = 36 → average_speed = distance / (time_minutes * 60) → average_speed = 4 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  assumption

end average_speed_distance_div_time_l52_52436


namespace ratio_of_metals_l52_52223

theorem ratio_of_metals (G C S : ℝ) (h1 : 11 * G + 5 * C + 7 * S = 9 * (G + C + S)) : 
  G / C = 1 / 2 ∧ G / S = 1 :=
by
  sorry

end ratio_of_metals_l52_52223


namespace combined_frosting_rate_l52_52044

theorem combined_frosting_rate (time_Cagney time_Lacey total_time : ℕ) (Cagney_rate Lacey_rate : ℚ) :
  (time_Cagney = 20) →
  (time_Lacey = 30) →
  (total_time = 5 * 60) →
  (Cagney_rate = 1 / time_Cagney) →
  (Lacey_rate = 1 / time_Lacey) →
  ((Cagney_rate + Lacey_rate) * total_time) = 25 :=
by
  intros
  -- conditions are given and used in the statement.
  -- proof follows from these conditions. 
  sorry

end combined_frosting_rate_l52_52044


namespace binomial_7_2_l52_52471

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l52_52471


namespace z_sum_of_squares_eq_101_l52_52855

open Complex

noncomputable def z_distances_sum_of_squares (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : ℝ :=
  abs (z - (1 + 1 * I)) ^ 2 + abs (z - (5 - 5 * I)) ^ 2

theorem z_sum_of_squares_eq_101 (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : 
  z_distances_sum_of_squares z h = 101 :=
by
  sorry

end z_sum_of_squares_eq_101_l52_52855


namespace james_meat_sales_l52_52539

theorem james_meat_sales
  (beef_pounds : ℕ)
  (pork_pounds : ℕ)
  (meat_per_meal : ℝ)
  (meal_price : ℝ)
  (total_meat : ℝ)
  (number_of_meals : ℝ)
  (total_money : ℝ)
  (h1 : beef_pounds = 20)
  (h2 : pork_pounds = beef_pounds / 2)
  (h3 : meat_per_meal = 1.5)
  (h4 : meal_price = 20)
  (h5 : total_meat = beef_pounds + pork_pounds)
  (h6 : number_of_meals = total_meat / meat_per_meal)
  (h7 : total_money = number_of_meals * meal_price) :
  total_money = 400 := by
  sorry

end james_meat_sales_l52_52539


namespace Buffy_whiskers_l52_52207

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l52_52207


namespace pleasant_goat_paths_l52_52463

-- Define the grid points A, B, and C
structure Point :=
  (x : ℕ)
  (y : ℕ)

def A : Point := { x := 0, y := 0 }
def C : Point := { x := 3, y := 3 }  -- assuming some grid layout
def B : Point := { x := 1, y := 1 }

-- Define a statement to count the number of shortest paths
def shortest_paths_count (A B C : Point) : ℕ := sorry

-- Proving the shortest paths from A to C avoiding B is 81
theorem pleasant_goat_paths : shortest_paths_count A B C = 81 := 
sorry

end pleasant_goat_paths_l52_52463


namespace number_of_two_digit_primes_with_digit_sum_12_l52_52201

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_digit_sum_12 : 
  ∃! n, is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 12 :=
by
  sorry

end number_of_two_digit_primes_with_digit_sum_12_l52_52201


namespace find_j_l52_52545

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_j
  (a b c : ℤ)
  (h1 : f a b c 2 = 0)
  (h2 : 200 < f a b c 10 ∧ f a b c 10 < 300)
  (h3 : 400 < f a b c 9 ∧ f a b c 9 < 500)
  (j : ℤ)
  (h4 : 1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1)) :
  j = 36 := sorry

end find_j_l52_52545


namespace find_Q_div_P_l52_52989

variable (P Q : ℚ)
variable (h_eq : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
  P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))

theorem find_Q_div_P : Q / P = -6 / 13 := by
  sorry

end find_Q_div_P_l52_52989


namespace number_of_members_l52_52757

theorem number_of_members (n : ℕ) (h : n^2 = 5929) : n = 77 :=
sorry

end number_of_members_l52_52757


namespace sequence_sum_l52_52134

noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry
noncomputable def a₅ : ℝ := sorry
noncomputable def a₆ : ℝ := sorry
noncomputable def a₇ : ℝ := sorry
noncomputable def a₈ : ℝ := sorry
noncomputable def q : ℝ := sorry

axiom condition_1 : a₁ + a₂ + a₃ + a₄ = 1
axiom condition_2 : a₅ + a₆ + a₇ + a₈ = 2
axiom condition_3 : q^4 = 2

theorem sequence_sum : q = (2:ℝ)^(1/4) → a₁ + a₂ + a₃ + a₄ = 1 → 
  (a₁ * q^16 + a₂ * q^17 + a₃ * q^18 + a₄ * q^19) = 16 := 
by
  intros hq hsum_s4
  sorry

end sequence_sum_l52_52134


namespace min_rainfall_on_fourth_day_l52_52143

theorem min_rainfall_on_fourth_day : 
  let capacity_ft := 6
  let drain_per_day_in := 3
  let rain_first_day_in := 10
  let rain_second_day_in := 2 * rain_first_day_in
  let rain_third_day_in := 1.5 * rain_second_day_in
  let total_rain_first_three_days_in := rain_first_day_in + rain_second_day_in + rain_third_day_in
  let total_drain_in := 3 * drain_per_day_in
  let water_level_start_fourth_day_in := total_rain_first_three_days_in - total_drain_in
  let capacity_in := capacity_ft * 12
  capacity_in = water_level_start_fourth_day_in + 21 :=
by
  sorry

end min_rainfall_on_fourth_day_l52_52143


namespace eval_at_3_l52_52795

theorem eval_at_3 : (3^3)^(3^3) = 27^27 :=
by sorry

end eval_at_3_l52_52795


namespace angle_terminal_side_equiv_l52_52460

theorem angle_terminal_side_equiv (α : ℝ) (k : ℤ) :
  (∃ k : ℤ, α = 30 + k * 360) ↔ (∃ β : ℝ, β = 30 ∧ α % 360 = β % 360) :=
by
  sorry

end angle_terminal_side_equiv_l52_52460


namespace Montoya_budget_spent_on_food_l52_52583

-- Define the fractions spent on groceries and going out to eat
def groceries_fraction : ℝ := 0.6
def eating_out_fraction : ℝ := 0.2

-- Define the total fraction spent on food
def total_food_fraction (g : ℝ) (e : ℝ) : ℝ := g + e

-- The theorem to prove
theorem Montoya_budget_spent_on_food : total_food_fraction groceries_fraction eating_out_fraction = 0.8 := 
by
  -- the proof will go here
  sorry

end Montoya_budget_spent_on_food_l52_52583


namespace determine_coefficients_l52_52794

theorem determine_coefficients (a b c : ℝ) (x y : ℝ) :
  (x = 3/4 ∧ y = 5/8) →
  (a * (x - 1) + 2 * y = 1) →
  (b * |x - 1| + c * y = 3) →
  (a = 1 ∧ b = 2 ∧ c = 4) := 
by 
  intros 
  sorry

end determine_coefficients_l52_52794


namespace mindmaster_code_count_l52_52100

theorem mindmaster_code_count :
  let colors := 7
  let slots := 5
  (colors ^ slots) = 16807 :=
by
  -- Define the given conditions
  let colors := 7
  let slots := 5
  -- Proof statement to be inserted here
  sorry

end mindmaster_code_count_l52_52100


namespace maisy_earns_more_l52_52251

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l52_52251


namespace find_number_of_dimes_l52_52156

def total_value (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25 + half_dollars * 50

def number_of_coins (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies + nickels + dimes + quarters + half_dollars

theorem find_number_of_dimes
  (pennies nickels dimes quarters half_dollars : Nat)
  (h_value : total_value pennies nickels dimes quarters half_dollars = 163)
  (h_coins : number_of_coins pennies nickels dimes quarters half_dollars = 13)
  (h_penny : 1 ≤ pennies)
  (h_nickel : 1 ≤ nickels)
  (h_dime : 1 ≤ dimes)
  (h_quarter : 1 ≤ quarters)
  (h_half_dollar : 1 ≤ half_dollars) :
  dimes = 3 :=
sorry

end find_number_of_dimes_l52_52156


namespace fair_coin_second_head_l52_52905

theorem fair_coin_second_head (P : ℝ) 
  (fair_coin : ∀ outcome : ℝ, outcome = 0.5) :
  P = 0.5 :=
by
  sorry

end fair_coin_second_head_l52_52905


namespace monica_study_ratio_l52_52700

theorem monica_study_ratio :
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  weekend = wednesday + thursday + friday :=
by
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  sorry

end monica_study_ratio_l52_52700


namespace arithmetic_sequence_a1a6_eq_l52_52685

noncomputable def a_1 : ℤ := 2
noncomputable def d : ℤ := 1
noncomputable def a_n (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem arithmetic_sequence_a1a6_eq :
  (a_1 * a_n 6) = 14 := by 
  sorry

end arithmetic_sequence_a1a6_eq_l52_52685


namespace problem_inequality_l52_52952

theorem problem_inequality 
  (a b c d : ℝ)
  (h1 : d > 0)
  (h2 : a ≥ b)
  (h3 : b ≥ c)
  (h4 : c ≥ d)
  (h5 : a * b * c * d = 1) : 
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1 / 3)) :=
sorry

end problem_inequality_l52_52952


namespace fraction_of_work_left_l52_52612

theorem fraction_of_work_left (a_days b_days : ℕ) (together_days : ℕ) 
    (h_a : a_days = 15) (h_b : b_days = 20) (h_together : together_days = 4) : 
    (1 - together_days * ((1/a_days : ℚ) + (1/b_days))) = 8/15 := by
  sorry

end fraction_of_work_left_l52_52612


namespace matrix_mult_correct_l52_52650

-- Definition of matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 1],
  ![4, -2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![7, -3],
  ![2, 4]
]

-- The goal is to prove that A * B yields the matrix C
def matrix_product : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![23, -5],
  ![24, -20]
]

theorem matrix_mult_correct : A * B = matrix_product := by
  -- Proof omitted
  sorry

end matrix_mult_correct_l52_52650


namespace evaluate_expression_l52_52063

theorem evaluate_expression (x c : ℕ) (h1 : x = 3) (h2 : c = 2) : 
  ((x^2 + c)^2 - (x^2 - c)^2) = 72 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l52_52063


namespace relationship_among_abc_l52_52675

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

theorem relationship_among_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_among_abc_l52_52675


namespace algebraic_expression_value_l52_52064

theorem algebraic_expression_value (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (  ( ((x + 2)^2 * (x^2 - 2 * x + 4)^2) / ( (x^3 + 8)^2 ))^2
   * ( ((x - 2)^2 * (x^2 + 2 * x + 4)^2) / ( (x^3 - 8)^2 ))^2 ) = 1 :=
by
  sorry

end algebraic_expression_value_l52_52064


namespace siamese_cats_initially_l52_52761

theorem siamese_cats_initially (house_cats: ℕ) (cats_sold: ℕ) (cats_left: ℕ) (initial_siamese: ℕ) :
  house_cats = 5 → 
  cats_sold = 10 → 
  cats_left = 8 → 
  (initial_siamese + house_cats - cats_sold = cats_left) → 
  initial_siamese = 13 :=
by
  intros h1 h2 h3 h4
  sorry

end siamese_cats_initially_l52_52761


namespace maximize_prob_of_consecutive_wins_l52_52442

variable {p1 p2 p3 : ℝ}
variable (h1 : 0 < p1) (h2 : p1 < p2) (h3 : p2 < p3)

def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_prob_of_consecutive_wins : P_C > P_A ∧ P_C > P_B :=
by sorry

end maximize_prob_of_consecutive_wins_l52_52442


namespace find_positive_integer_cube_root_divisible_by_21_l52_52797

theorem find_positive_integer_cube_root_divisible_by_21 (m : ℕ) (h1: m = 735) :
  m % 21 = 0 ∧ 9 < (m : ℝ)^(1/3) ∧ (m : ℝ)^(1/3) < 9.1 :=
by {
  sorry
}

end find_positive_integer_cube_root_divisible_by_21_l52_52797


namespace each_album_contains_correct_pictures_l52_52276

def pictures_in_each_album (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat) :=
  (pictures_per_album_phone + pictures_per_album_camera)

theorem each_album_contains_correct_pictures (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat)
  (h1 : pictures_phone = 80)
  (h2 : pictures_camera = 40)
  (h3 : albums = 10)
  (h4 : pictures_per_album_phone = 8)
  (h5 : pictures_per_album_camera = 4)
  : pictures_in_each_album pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera = 12 := by
  sorry

end each_album_contains_correct_pictures_l52_52276


namespace total_cost_of_projectors_and_computers_l52_52406

theorem total_cost_of_projectors_and_computers :
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  (n_p * c_p + n_c * c_c) = 175200 := by
  let n_p := 8
  let c_p := 7500
  let n_c := 32
  let c_c := 3600
  sorry 

end total_cost_of_projectors_and_computers_l52_52406


namespace largest_x_satisfying_inequality_l52_52308

theorem largest_x_satisfying_inequality :
  (∃ x : ℝ, 
    (∀ y : ℝ, |(y^2 - 4 * y - 39601)| ≥ |(y^2 + 4 * y - 39601)| → y ≤ x) ∧ 
    |(x^2 - 4 * x - 39601)| ≥ |(x^2 + 4 * x - 39601)|
  ) → x = 199 := 
sorry

end largest_x_satisfying_inequality_l52_52308


namespace binom_7_2_eq_21_l52_52485

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l52_52485


namespace length_of_bridge_l52_52275

theorem length_of_bridge 
  (lenA : ℝ) (speedA : ℝ) (lenB : ℝ) (speedB : ℝ) (timeA : ℝ) (timeB : ℝ) (startAtSameTime : Prop)
  (h1 : lenA = 120) (h2 : speedA = 12.5) (h3 : lenB = 150) (h4 : speedB = 15.28) 
  (h5 : timeA = 30) (h6 : timeB = 25) : 
  (∃ X : ℝ, X = 757) :=
by
  sorry

end length_of_bridge_l52_52275


namespace manager_salary_calculation_l52_52830

theorem manager_salary_calculation :
  let percent_marketers := 0.60
  let salary_marketers := 50000
  let percent_engineers := 0.20
  let salary_engineers := 80000
  let percent_sales_reps := 0.10
  let salary_sales_reps := 70000
  let percent_managers := 0.10
  let total_average_salary := 75000
  let total_contribution := percent_marketers * salary_marketers + percent_engineers * salary_engineers + percent_sales_reps * salary_sales_reps
  let managers_total_contribution := total_average_salary - total_contribution
  let manager_salary := managers_total_contribution / percent_managers
  manager_salary = 220000 :=
by
  sorry

end manager_salary_calculation_l52_52830


namespace time_after_6666_seconds_l52_52841

noncomputable def initial_time : Nat := 3 * 3600
noncomputable def additional_seconds : Nat := 6666

-- Function to convert total seconds to "HH:MM:SS" format
def time_in_seconds (h m s : Nat) : Nat :=
  h*3600 + m*60 + s

noncomputable def new_time : Nat :=
  initial_time + additional_seconds

-- Convert the new total time back to "HH:MM:SS" format (expected: 4:51:06)
def hours (secs : Nat) : Nat := secs / 3600
def minutes (secs : Nat) : Nat := (secs % 3600) / 60
def seconds (secs : Nat) : Nat := (secs % 3600) % 60

theorem time_after_6666_seconds :
  hours new_time = 4 ∧ minutes new_time = 51 ∧ seconds new_time = 6 :=
by
  sorry

end time_after_6666_seconds_l52_52841


namespace range_of_mn_l52_52515

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x

theorem range_of_mn (m n : ℝ)
  (h₁ : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4)
  (h₂ : ∀ z, -5 ≤ z ∧ z ≤ 4 → ∃ x, f x = z ∧ m ≤ x ∧ x ≤ n) :
  1 ≤ m + n ∧ m + n ≤ 7 :=
by
  sorry

end range_of_mn_l52_52515


namespace find_a6_l52_52677

noncomputable def a_n (n : ℕ) : ℝ := sorry
noncomputable def S_n (n : ℕ) : ℝ := sorry
noncomputable def r : ℝ := sorry

axiom h_pos : ∀ n, a_n n > 0
axiom h_s3 : S_n 3 = 14
axiom h_a3 : a_n 3 = 8

theorem find_a6 : a_n 6 = 64 := by sorry

end find_a6_l52_52677


namespace right_triangle_area_l52_52664

theorem right_triangle_area (a : ℝ) (h1 : a > 0) :
  let b := (2 / 3) * a,
      c := (∛13 / 3) * a,
      area := a * b / 2 in 
  area = 8 / 3 :=
by {
  sorry
}

end right_triangle_area_l52_52664


namespace remaining_amount_division_l52_52425

-- Definitions
def total_amount : ℕ := 2100
def number_of_participants : ℕ := 8
def amount_already_raised : ℕ := 150

-- Proof problem statement
theorem remaining_amount_division :
  (total_amount - amount_already_raised) / (number_of_participants - 1) = 279 :=
by
  sorry

end remaining_amount_division_l52_52425


namespace first_term_of_geometric_sequence_l52_52130

theorem first_term_of_geometric_sequence
  (a r : ℚ) -- where a is the first term and r is the common ratio
  (h1 : a * r^4 = 45) -- fifth term condition
  (h2 : a * r^5 = 60) -- sixth term condition
  : a = 1215 / 256 := 
sorry

end first_term_of_geometric_sequence_l52_52130


namespace binomial_7_2_l52_52473

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l52_52473


namespace pattern_D_cannot_form_tetrahedron_l52_52790

theorem pattern_D_cannot_form_tetrahedron :
  (¬ ∃ (f : ℝ × ℝ → ℝ × ℝ),
      f (0, 0) = (1, 1) ∧ f (1, 0) = (1, -1) ∧ f (2, 0) = (-1, 1) ∧ f (3, 0) = (-1, -1)) :=
by
  -- proof will go here
  sorry

end pattern_D_cannot_form_tetrahedron_l52_52790


namespace disk_diameter_solution_l52_52150

noncomputable def disk_diameter_condition : Prop :=
∃ x : ℝ, 
  (4 * Real.sqrt 3 + 2 * Real.pi) * x^2 - 12 * x + Real.sqrt 3 = 0 ∧
  x < Real.sqrt 3 / 6 ∧ 
  2 * x = 0.36

theorem disk_diameter_solution : exists (x : ℝ), 
  disk_diameter_condition := 
sorry

end disk_diameter_solution_l52_52150


namespace find_height_of_cylinder_l52_52803

theorem find_height_of_cylinder (r SA : ℝ) (h : ℝ) (h_r : r = 3) (h_SA : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h → h = 2 :=
by
  sorry

end find_height_of_cylinder_l52_52803


namespace strokes_over_par_l52_52891

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l52_52891


namespace find_sam_age_l52_52570

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l52_52570


namespace find_original_number_l52_52128

def digitsGPA (A B C : ℕ) : Prop := B^2 = A * C
def digitsAPA (X Y Z : ℕ) : Prop := 2 * Y = X + Z

theorem find_original_number (A B C X Y Z : ℕ) :
  100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C ≤ 999 ∧
  digitsGPA A B C ∧
  100 * X + 10 * Y + Z = (100 * A + 10 * B + C) - 200 ∧
  digitsAPA X Y Z →
  (100 * A + 10 * B + C) = 842 :=
sorry

end find_original_number_l52_52128


namespace physics_marks_l52_52158

theorem physics_marks
  (P C M : ℕ)
  (h1 : P + C + M = 240)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 80 :=
by
  sorry

end physics_marks_l52_52158


namespace range_of_m_l52_52111

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) * (2 * x - 1) - m * x + m

def exists_unique_int_n (m : ℝ) : Prop :=
∃! n : ℤ, f n m < 0

theorem range_of_m {m : ℝ} (h : m < 1) (h2 : exists_unique_int_n m) : 
  (Real.exp 1) * (1 / 2) ≤ m ∧ m < 1 :=
sorry

end range_of_m_l52_52111


namespace abs_neg_one_third_l52_52127

theorem abs_neg_one_third : abs (-1/3) = 1/3 := by
  sorry

end abs_neg_one_third_l52_52127


namespace beetles_consumed_per_day_l52_52099

-- Definitions
def bird_eats_beetles (n : Nat) : Nat := 12 * n
def snake_eats_birds (n : Nat) : Nat := 3 * n
def jaguar_eats_snakes (n : Nat) : Nat := 5 * n
def crocodile_eats_jaguars (n : Nat) : Nat := 2 * n

-- Initial values
def initial_jaguars : Nat := 6
def initial_crocodiles : Nat := 30
def net_increase_birds : Nat := 4
def net_increase_snakes : Nat := 2
def net_increase_jaguars : Nat := 1

-- Proof statement
theorem beetles_consumed_per_day : 
  bird_eats_beetles (snake_eats_birds (jaguar_eats_snakes initial_jaguars)) = 1080 := 
by 
  sorry

end beetles_consumed_per_day_l52_52099


namespace somu_fathers_age_ratio_l52_52579

noncomputable def somus_age := 16

def proof_problem (S F : ℕ) : Prop :=
  S = 16 ∧ 
  (S - 8 = (1 / 5) * (F - 8)) ∧
  (S / F = 1 / 3)

theorem somu_fathers_age_ratio (S F : ℕ) : proof_problem S F :=
by
  sorry

end somu_fathers_age_ratio_l52_52579


namespace triangle_area_solution_l52_52682

noncomputable def triangle_area_problem 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  : ℝ := (1 / 2) * a * c * Real.sin B

theorem triangle_area_solution 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  (ha : a = 12)
  (hb : b = 6 * Real.sin (π / 3))
  (hA : A = π / 2)
  (hB : B = π / 3)
  (hC : C = π / 6) 
  : triangle_area_problem a b c A B C h1 h2 h3 = 18 * Real.sqrt 3 := by
  sorry

end triangle_area_solution_l52_52682


namespace dealer_pricing_l52_52426

theorem dealer_pricing
  (cost_price : ℝ)
  (discount : ℝ := 0.10)
  (profit : ℝ := 0.20)
  (num_articles_sold : ℕ := 45)
  (num_articles_cost : ℕ := 40)
  (selling_price_per_article : ℝ := (num_articles_cost : ℝ) / num_articles_sold)
  (actual_cost_price_per_article : ℝ := selling_price_per_article / (1 + profit))
  (listed_price_per_article : ℝ := selling_price_per_article / (1 - discount)) :
  100 * ((listed_price_per_article - actual_cost_price_per_article) / actual_cost_price_per_article) = 33.33 := by
  sorry

end dealer_pricing_l52_52426


namespace coordinates_of_P_l52_52671

def point (x y : ℝ) := (x, y)

def A : (ℝ × ℝ) := point 1 1
def B : (ℝ × ℝ) := point 4 0

def vector_sub (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2)

def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem coordinates_of_P
  (P : ℝ × ℝ)
  (hP : vector_sub P A = scalar_mult 3 (vector_sub B P)) :
  P = (11 / 2, -1 / 2) :=
by
  sorry

end coordinates_of_P_l52_52671


namespace buffy_whiskers_l52_52203

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l52_52203


namespace diff_roots_eq_sqrt_2p2_add_2p_sub_2_l52_52306

theorem diff_roots_eq_sqrt_2p2_add_2p_sub_2 (p : ℝ) :
  let a := 1
  let b := -2 * p
  let c := p^2 - p + 1
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let r1 := (-b + sqrt_discriminant) / (2 * a)
  let r2 := (-b - sqrt_discriminant) / (2 * a)
  r1 - r2 = Real.sqrt (2*p^2 + 2*p - 2) :=
by
  sorry

end diff_roots_eq_sqrt_2p2_add_2p_sub_2_l52_52306


namespace largest_prime_factor_of_1729_l52_52004

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l52_52004


namespace inequality_hold_l52_52379

theorem inequality_hold (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧ 
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ 1/8 :=
sorry

end inequality_hold_l52_52379


namespace probability_of_exactly_one_instrument_l52_52285

-- Definitions
def total_people : ℕ := 800
def fraction_play_at_least_one_instrument : ℚ := 2 / 5
def num_play_two_or_more_instruments : ℕ := 96

-- Calculation
def num_play_at_least_one_instrument := fraction_play_at_least_one_instrument * total_people
def num_play_exactly_one_instrument := num_play_at_least_one_instrument - num_play_two_or_more_instruments

-- Probability calculation
def probability_play_exactly_one_instrument := num_play_exactly_one_instrument / total_people

-- Proof statement
theorem probability_of_exactly_one_instrument :
  probability_play_exactly_one_instrument = 0.28 := by
  sorry

end probability_of_exactly_one_instrument_l52_52285


namespace buffy_whiskers_l52_52209

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l52_52209


namespace first_group_number_l52_52898

theorem first_group_number (x : ℕ) (h1 : x + 120 = 126) : x = 6 :=
by
  sorry

end first_group_number_l52_52898


namespace total_vehicles_in_lanes_l52_52454

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end total_vehicles_in_lanes_l52_52454


namespace total_markings_on_stick_l52_52032

noncomputable def markings (n m : ℕ) : ℕ := 
  (0..n).toFinset.card + (0..m).toFinset.card - (0..(n*m / (n.gcd m))).toFinset.card - 2

theorem total_markings_on_stick : markings 4 5 = 9 :=
by sorry

end total_markings_on_stick_l52_52032


namespace three_digit_multiples_of_3_and_11_l52_52520

theorem three_digit_multiples_of_3_and_11 : 
  ∃ n, n = 27 ∧ ∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 33 = 0 ↔ ∃ k, x = 33 * k ∧ 4 ≤ k ∧ k ≤ 30 :=
by
  sorry

end three_digit_multiples_of_3_and_11_l52_52520


namespace find_x_in_interval_l52_52200

noncomputable def a : ℝ := Real.sqrt 2014 - Real.sqrt 2013

theorem find_x_in_interval :
  ∀ x : ℝ, (0 < x) → (x < Real.pi) →
  (a^(Real.tan x ^ 2) + (Real.sqrt 2014 + Real.sqrt 2013)^(-Real.tan x ^ 2) = 2 * a^3) →
  (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) := by
  -- add proof here
  sorry

end find_x_in_interval_l52_52200


namespace train_length_correct_l52_52768

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l52_52768


namespace A_union_B_subset_B_A_intersection_B_subset_B_l52_52087

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3 * x - 10 <= 0}
def B (m : ℝ) : Set ℝ := {x | m - 4 <= x ∧ x <= 3 * m + 2}

-- Problem 1: Prove the range of m if A ∪ B = B
theorem A_union_B_subset_B (m : ℝ) : (A ∪ B m = B m) → (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

-- Problem 2: Prove the range of m if A ∩ B = B
theorem A_intersection_B_subset_B (m : ℝ) : (A ∩ B m = B m) → (m < -3) :=
by
  sorry

end A_union_B_subset_B_A_intersection_B_subset_B_l52_52087


namespace sector_to_cone_volume_l52_52084

theorem sector_to_cone_volume (θ : ℝ) (A : ℝ) (V : ℝ) (l r h : ℝ) :
  θ = (2 * Real.pi / 3) →
  A = (3 * Real.pi) →
  A = (1 / 2 * l^2 * θ) →
  θ = (r / l * 2 * Real.pi) →
  h = Real.sqrt (l^2 - r^2) →
  V = (1 / 3 * Real.pi * r^2 * h) →
  V = (2 * Real.sqrt 2 * Real.pi / 3) :=
by
  intros hθ hA hAeq hθeq hh hVeq
  sorry

end sector_to_cone_volume_l52_52084


namespace gcd_of_elements_in_T_is_one_l52_52363

open Set

variables {n : ℕ} (U : Finset ℕ) (S T : Finset ℕ) (s d : ℕ)

noncomputable def greatest_common_divisor := Finset.gcd

theorem gcd_of_elements_in_T_is_one (hU_sub : U = Finset.range (n + 1)) (hS_sub : S ⊆ U)
  (hT_sub : T ⊆ U) (hS_nonempty : S.nonempty) (hs_def : greatest_common_divisor S = s)
  (hs_not_one : s ≠ 1) (hd_def : d = Finset.min' (Finset.filter (fun x => x > 1 ∧ s % x = 0) (Finset.range (s + 1))) (by sorry))
  (hT_size : ∥T∥ ≥ 1 + n / d) :
  greatest_common_divisor T = 1 := by
  sorry

end gcd_of_elements_in_T_is_one_l52_52363


namespace option_A_option_B_option_C_option_D_l52_52956

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end option_A_option_B_option_C_option_D_l52_52956


namespace double_counted_toddlers_l52_52466

def number_of_toddlers := 21
def missed_toddlers := 3
def billed_count := 26

theorem double_counted_toddlers : 
  ∃ (D : ℕ), (number_of_toddlers + D - missed_toddlers = billed_count) ∧ D = 8 :=
by
  sorry

end double_counted_toddlers_l52_52466


namespace triangle_perimeter_inequality_l52_52881

theorem triangle_perimeter_inequality (x : ℕ) (h₁ : 15 + 24 > x) (h₂ : 15 + x > 24) (h₃ : 24 + x > 15) 
    (h₄ : ∃ x : ℕ, x > 9 ∧ x < 39) : 15 + 24 + x = 49 :=
by { sorry }

end triangle_perimeter_inequality_l52_52881


namespace fixed_monthly_fee_l52_52863

theorem fixed_monthly_fee (f h : ℝ) 
  (feb_bill : f + h = 18.72)
  (mar_bill : f + 3 * h = 33.78) :
  f = 11.19 :=
by
  sorry

end fixed_monthly_fee_l52_52863


namespace max_watches_two_hours_l52_52694

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end max_watches_two_hours_l52_52694


namespace compare_neg_two_powers_l52_52185

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end compare_neg_two_powers_l52_52185


namespace isosceles_triangle_perimeter_l52_52524

theorem isosceles_triangle_perimeter :
  ∃ P : ℕ, (P = 15 ∨ P = 18) ∧ ∀ (a b c : ℕ), (a = 7 ∨ b = 7 ∨ c = 7) ∧ (a = 4 ∨ b = 4 ∨ c = 4) → ((a = 7 ∨ a = 4) ∧ (b = 7 ∨ b = 4) ∧ (c = 7 ∨ c = 4)) ∧ P = a + b + c :=
by
  sorry

end isosceles_triangle_perimeter_l52_52524


namespace combined_weight_of_contents_l52_52435

theorem combined_weight_of_contents
    (weight_pencil : ℝ := 28.3)
    (weight_eraser : ℝ := 15.7)
    (weight_paperclip : ℝ := 3.5)
    (weight_stapler : ℝ := 42.2)
    (num_pencils : ℕ := 5)
    (num_erasers : ℕ := 3)
    (num_paperclips : ℕ := 4)
    (num_staplers : ℕ := 2) :
    num_pencils * weight_pencil +
    num_erasers * weight_eraser +
    num_paperclips * weight_paperclip +
    num_staplers * weight_stapler = 287 := 
sorry

end combined_weight_of_contents_l52_52435


namespace tan_double_angle_sum_l52_52815

theorem tan_double_angle_sum (α : ℝ) (h : Real.tan α = 3 / 2) :
  Real.tan (2 * α + Real.pi / 4) = -7 / 17 := 
sorry

end tan_double_angle_sum_l52_52815


namespace rem_value_is_correct_l52_52651

def rem (x y : ℚ) : ℚ :=
  x - y * (Int.floor (x / y))

theorem rem_value_is_correct : rem (-5/9) (7/3) = 16/9 := by
  sorry

end rem_value_is_correct_l52_52651


namespace cost_of_bricks_l52_52602

theorem cost_of_bricks
  (N: ℕ)
  (half_bricks:ℕ)
  (full_price: ℝ)
  (discount_percentage: ℝ)
  (n_half: half_bricks = N / 2)
  (P1: full_price = 0.5)
  (P2: discount_percentage = 0.5):
  (half_bricks * (full_price * discount_percentage) + 
  half_bricks * full_price = 375) := 
by sorry

end cost_of_bricks_l52_52602


namespace sum_reciprocal_l52_52331

open Real

theorem sum_reciprocal (y : ℝ) (h₁ : y^3 + (1 / y)^3 = 110) : y + (1 / y) = 5 :=
sorry

end sum_reciprocal_l52_52331


namespace real_solutions_l52_52069

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end real_solutions_l52_52069


namespace avg_displacement_per_man_l52_52752

-- Problem definition as per the given conditions
def num_men : ℕ := 50
def tank_length : ℝ := 40  -- 40 meters
def tank_width : ℝ := 20   -- 20 meters
def rise_in_water_level : ℝ := 0.25  -- 25 cm -> 0.25 meters

-- Given the conditions, we need to prove the average displacement per man
theorem avg_displacement_per_man :
  (tank_length * tank_width * rise_in_water_level) / num_men = 4 := by
  sorry

end avg_displacement_per_man_l52_52752


namespace complement_A_U_l52_52555

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U
def C_U_A : Set ℕ := U \ A

-- Theorem: The complement of A with respect to U is {2, 4}
theorem complement_A_U : C_U_A = {2, 4} := by
  sorry

end complement_A_U_l52_52555


namespace sqrt_inequality_l52_52551

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := 
by
  sorry

end sqrt_inequality_l52_52551


namespace arithmetic_progression_sum_at_least_66_l52_52229

-- Define the sum of the first n terms of an arithmetic progression
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the conditions for the arithmetic progression
def arithmetic_prog_conditions (a1 d : ℤ) (n : ℕ) :=
  sum_first_n_terms a1 d n ≥ 66

-- The main theorem to prove
theorem arithmetic_progression_sum_at_least_66 (n : ℕ) :
  (n >= 3 ∧ n <= 14) → arithmetic_prog_conditions 25 (-3) n :=
by
  sorry

end arithmetic_progression_sum_at_least_66_l52_52229


namespace unique_real_solution_l52_52309

theorem unique_real_solution :
  ∀ x : ℝ, (x > 0 → (x ^ 16 + 1) * (x ^ 12 + x ^ 8 + x ^ 4 + 1) = 18 * x ^ 8 → x = 1) :=
by
  introv
  sorry

end unique_real_solution_l52_52309


namespace minimum_value_proof_l52_52837

variables {A B C : ℝ}
variable (triangle_ABC : 
  ∀ {A B C : ℝ}, 
  (A > 0 ∧ A < π / 2) ∧ 
  (B > 0 ∧ B < π / 2) ∧ 
  (C > 0 ∧ C < π / 2))

noncomputable def minimum_value (A B C : ℝ) :=
  3 * (Real.tan B) * (Real.tan C) + 
  2 * (Real.tan A) * (Real.tan C) + 
  1 * (Real.tan A) * (Real.tan B)

theorem minimum_value_proof (h : 
  ∀ (A B C : ℝ), 
  (1 / (Real.tan A * Real.tan B)) + 
  (1 / (Real.tan B * Real.tan C)) + 
  (1 / (Real.tan C * Real.tan A)) = 1) 
  : minimum_value A B C = 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_proof_l52_52837


namespace missing_fraction_is_73_div_60_l52_52888

-- Definition of the given fractions
def fraction1 : ℚ := 1/3
def fraction2 : ℚ := 1/2
def fraction3 : ℚ := -5/6
def fraction4 : ℚ := 1/5
def fraction5 : ℚ := 1/4
def fraction6 : ℚ := -5/6

-- Total sum provided in the problem
def total_sum : ℚ := 50/60  -- 0.8333333333333334 in decimal form

-- The summation of given fractions
def sum_of_fractions : ℚ := fraction1 + fraction2 + fraction3 + fraction4 + fraction5 + fraction6

-- The statement to prove that the missing fraction is 73/60
theorem missing_fraction_is_73_div_60 : (total_sum - sum_of_fractions) = 73/60 := by
  sorry

end missing_fraction_is_73_div_60_l52_52888


namespace binomial_7_2_eq_21_l52_52478

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l52_52478


namespace obtuse_dihedral_angles_l52_52261

theorem obtuse_dihedral_angles (AOB BOC COA : ℝ) (h1 : AOB > 90) (h2 : BOC > 90) (h3 : COA > 90) :
  ∃ α β γ : ℝ, α > 90 ∧ β > 90 ∧ γ > 90 :=
sorry

end obtuse_dihedral_angles_l52_52261


namespace Roshesmina_pennies_l52_52967

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end Roshesmina_pennies_l52_52967


namespace olafs_dad_points_l52_52864

-- Let D be the number of points Olaf's dad scored.
def dad_points : ℕ := sorry

-- Olaf scored three times more points than his dad.
def olaf_points (dad_points : ℕ) : ℕ := 3 * dad_points

-- Total points scored is 28.
def total_points (dad_points olaf_points : ℕ) : Prop := dad_points + olaf_points = 28

theorem olafs_dad_points (D : ℕ) :
  (D + olaf_points D = 28) → (D = 7) :=
by
  sorry

end olafs_dad_points_l52_52864


namespace original_price_is_125_l52_52913

noncomputable def original_price (sold_price : ℝ) (discount_percent : ℝ) : ℝ :=
  sold_price / ((100 - discount_percent) / 100)

theorem original_price_is_125 : original_price 120 4 = 125 :=
by
  sorry

end original_price_is_125_l52_52913


namespace polynomials_equal_l52_52095

theorem polynomials_equal (f g : Polynomial ℝ) (n : ℕ) (x : Fin (n + 1) → ℝ) :
  (∀ i, f.eval (x i) = g.eval (x i)) → f = g :=
by
  sorry

end polynomials_equal_l52_52095


namespace new_pressure_of_nitrogen_gas_l52_52464

variable (p1 p2 v1 v2 k : ℝ)

theorem new_pressure_of_nitrogen_gas :
  (∀ p v, p * v = k) ∧ (p1 = 8) ∧ (v1 = 3) ∧ (p1 * v1 = k) ∧ (v2 = 7.5) →
  p2 = 3.2 :=
by
  intro h
  sorry

end new_pressure_of_nitrogen_gas_l52_52464


namespace product_of_two_numbers_l52_52878

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l52_52878


namespace middle_digit_base7_l52_52294

theorem middle_digit_base7 (a b c : ℕ) 
  (h1 : N = 49 * a + 7 * b + c) 
  (h2 : N = 81 * c + 9 * b + a)
  (h3 : a < 7 ∧ b < 7 ∧ c < 7) : 
  b = 0 :=
by sorry

end middle_digit_base7_l52_52294


namespace heaviest_box_difference_l52_52142

theorem heaviest_box_difference (a b c d : ℕ) (h : a < b) (h1 : b < c) (h2 : c < d)
  (pairs : multiset ℕ) (weights : multiset ℕ)
  (hpairs : pairs = [a + b, a + c, a + d, b + c, b + d, c + d])
  (hweights : weights = [22, 23, 27, 29, 30]) :
  (d - a) = 7 :=
by {
  sorry
}

end heaviest_box_difference_l52_52142


namespace find_a_parallel_lines_l52_52681

theorem find_a_parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, x * a + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = k * (x * a + 2 * y + 2)) ↔ a = -6 := by
  sorry

end find_a_parallel_lines_l52_52681


namespace eval_power_l52_52661

theorem eval_power {a m n : ℕ} : (a^m)^n = a^(m * n) := by
  sorry

example : (3^2)^4 = 6561 := by
  rw eval_power
  norm_num

end eval_power_l52_52661


namespace sum_of_vars_l52_52679

variables (a b c d k p : ℝ)

theorem sum_of_vars (h1 : a^2 + b^2 + c^2 + d^2 = 390)
                    (h2 : ab + bc + ca + ad + bd + cd = 5)
                    (h3 : ad + bd + cd = k)
                    (h4 : (a * b * c * d)^2 = p) :
                    a + b + c + d = 20 :=
by
  -- placeholder for the proof
  sorry

end sum_of_vars_l52_52679


namespace real_solutions_l52_52071

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end real_solutions_l52_52071


namespace total_vehicles_l52_52451

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end total_vehicles_l52_52451


namespace distribute_paper_clips_l52_52371

theorem distribute_paper_clips (total_paper_clips boxes : ℕ) (h_total : total_paper_clips = 81) (h_boxes : boxes = 9) : total_paper_clips / boxes = 9 := by
  sorry

end distribute_paper_clips_l52_52371


namespace spherical_to_rectangular_coordinates_l52_52054

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_coordinates :
  sphericalToRectangular 10 (5 * Real.pi / 4) (Real.pi / 4) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l52_52054


namespace max_value_of_x_neg_two_in_interval_l52_52399

noncomputable def x_neg_two_max_value_in_interval (a b : ℝ) (f : ℝ → ℝ) :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ 4

theorem max_value_of_x_neg_two_in_interval:
  x_neg_two_max_value_in_interval (1/2) 2 (λ x, x^(-2)) :=
by {
  intros x hx,
  have h1 : (1 / 2 : ℝ) > 0 := by linarith,
  have h2 : (2 : ℝ) > 0 := by linarith,
  have hx1 : x^(-2) ≤ 4, {
    rw [← one_div_eq_inv, ← one_div_eq_inv, div_le_div_iff],
    calc
      x^(-2) = 1 / x^2 : by rw [← rpow_neg (-2) x, rpow_nat_cast, rpow_neg (-2)], sorry, apply or.intro_right, linarith, sorry},
  exact hx1
}

end max_value_of_x_neg_two_in_interval_l52_52399


namespace sum_op_two_triangles_l52_52582

def op (a b c : ℕ) : ℕ := 2 * a - b + c

theorem sum_op_two_triangles : op 3 7 5 + op 6 2 8 = 22 := by
  sorry

end sum_op_two_triangles_l52_52582


namespace B_is_empty_l52_52923

def A : Set ℤ := {0}
def B : Set ℤ := {x | x > 8 ∧ x < 5}
def C : Set ℕ := {x | x - 1 = 0}
def D : Set ℤ := {x | x > 4}

theorem B_is_empty : B = ∅ := by
  sorry

end B_is_empty_l52_52923


namespace relationship_among_abc_l52_52938

noncomputable def a : ℝ := Real.logb 11 10
noncomputable def b : ℝ := (Real.logb 11 9) ^ 2
noncomputable def c : ℝ := Real.logb 10 11

theorem relationship_among_abc : b < a ∧ a < c :=
  sorry

end relationship_among_abc_l52_52938


namespace product_sqrt_50_l52_52726

theorem product_sqrt_50 (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : a^2 < 50) (h₄ : 50 < b^2) : a * b = 56 := by
  sorry

end product_sqrt_50_l52_52726


namespace roots_of_Q_are_fifth_powers_of_roots_of_P_l52_52689

def P (x : ℝ) : ℝ := x^3 - 3 * x + 1

noncomputable def Q (y : ℝ) : ℝ := y^3 + 15 * y^2 - 198 * y + 1

theorem roots_of_Q_are_fifth_powers_of_roots_of_P : 
  ∀ α β γ : ℝ, (P α = 0) ∧ (P β = 0) ∧ (P γ = 0) →
  (Q (α^5) = 0) ∧ (Q (β^5) = 0) ∧ (Q (γ^5) = 0) := 
by 
  intros α β γ h
  sorry

end roots_of_Q_are_fifth_powers_of_roots_of_P_l52_52689


namespace bullet_train_pass_time_l52_52291

noncomputable def time_to_pass (length_train : ℕ) (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) : ℝ := 
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * 1000 / 3600
  length_train / relative_speed_mps

def length_train := 350
def speed_train_kmph := 75
def speed_man_kmph := 12

theorem bullet_train_pass_time : 
  abs (time_to_pass length_train speed_train_kmph speed_man_kmph - 14.47) < 0.01 :=
by
  sorry

end bullet_train_pass_time_l52_52291


namespace complex_real_part_of_product_l52_52962

theorem complex_real_part_of_product (z1 z2 : ℂ) (i : ℂ) 
  (hz1 : z1 = 4 + 29 * Complex.I)
  (hz2 : z2 = 6 + 9 * Complex.I)
  (hi : i = Complex.I) : 
  ((z1 - z2) * i).re = 20 := 
by
  sorry

end complex_real_part_of_product_l52_52962


namespace binomial_7_2_l52_52488

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l52_52488


namespace ceil_square_of_neg_five_thirds_l52_52314

theorem ceil_square_of_neg_five_thirds : Int.ceil ((-5 / 3:ℚ)^2) = 3 := by
  sorry

end ceil_square_of_neg_five_thirds_l52_52314


namespace false_statements_count_is_3_l52_52701

-- Define the statements
def statement1_false : Prop := ¬ (1 ≠ 1)     -- Not exactly one statement is false
def statement2_false : Prop := ¬ (2 ≠ 2)     -- Not exactly two statements are false
def statement3_false : Prop := ¬ (3 ≠ 3)     -- Not exactly three statements are false
def statement4_false : Prop := ¬ (4 ≠ 4)     -- Not exactly four statements are false
def statement5_false : Prop := ¬ (5 ≠ 5)     -- Not all statements are false

-- Prove that the number of false statements is 3
theorem false_statements_count_is_3 :
  (statement1_false → statement2_false →
  statement3_false → statement4_false →
  statement5_false → (3 = 3)) := by
  sorry

end false_statements_count_is_3_l52_52701


namespace ink_left_is_50_percent_l52_52147

variables (A1 A2 : ℕ)
variables (length width : ℕ)
variables (total_area used_area : ℕ)

-- Define the conditions
def total_area_of_squares := 3 * (4 * 4)
def total_area_of_rectangles := 2 * (6 * 2)
def ink_left_percentage := ((total_area_of_squares - total_area_of_rectangles) * 100) / total_area_of_squares

-- The theorem to prove
theorem ink_left_is_50_percent : ink_left_percentage = 50 :=
by
  rw [total_area_of_squares, total_area_of_rectangles]
  norm_num
  exact rfl
  sorry -- Proof omitted

end ink_left_is_50_percent_l52_52147


namespace smallest_integer_20p_larger_and_19p_smaller_l52_52930

theorem smallest_integer_20p_larger_and_19p_smaller :
  ∃ (N x y : ℕ), N = 162 ∧ N = 12 / 10 * x ∧ N = 81 / 100 * y :=
by
  sorry

end smallest_integer_20p_larger_and_19p_smaller_l52_52930


namespace line_MN_parallel_to_y_axis_l52_52673

-- Definition of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector between two points
def vector_between (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y,
    z := Q.z - P.z }

-- Given points M and N
def M : Point := { x := 3, y := -2, z := 1 }
def N : Point := { x := 3, y := 2, z := 1 }

-- The vector \overrightarrow{MN}
def vec_MN : Point := vector_between M N

-- Theorem: The vector between points M and N is parallel to the y-axis
theorem line_MN_parallel_to_y_axis : vec_MN = {x := 0, y := 4, z := 0} := by
  sorry

end line_MN_parallel_to_y_axis_l52_52673


namespace find_third_number_l52_52523

-- Define the conditions
def equation1_valid : Prop := (5 * 3 = 15) ∧ (5 * 2 = 10) ∧ (2 * 1000 + 3 * 100 + 5 = 1022)
def equation2_valid : Prop := (9 * 2 = 18) ∧ (9 * 4 = 36) ∧ (4 * 1000 + 2 * 100 + 9 = 3652)

-- The theorem to prove
theorem find_third_number (h1 : equation1_valid) (h2 : equation2_valid) : (7 * 2 = 14) ∧ (7 * 5 = 35) ∧ (5 * 1000 + 2 * 100 + 7 = 547) :=
by 
  sorry

end find_third_number_l52_52523


namespace total_amount_spent_l52_52226
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end total_amount_spent_l52_52226


namespace compare_neg_two_cubed_l52_52183

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end compare_neg_two_cubed_l52_52183


namespace ratio_of_hexagon_areas_l52_52243

open Real

-- Define the given conditions about the hexagon and the midpoints
structure Hexagon :=
  (s : ℝ)
  (regular : True)
  (midpoints : True)

theorem ratio_of_hexagon_areas (h : Hexagon) : 
  let s := 2
  ∃ (area_ratio : ℝ), area_ratio = 4 / 7 :=
by
  sorry

end ratio_of_hexagon_areas_l52_52243


namespace worker_b_time_l52_52610

theorem worker_b_time (T_B : ℝ) : 
  (1 / 10) + (1 / T_B) = 1 / 6 → T_B = 15 := by
  intro h
  sorry

end worker_b_time_l52_52610


namespace purely_imaginary_value_of_m_third_quadrant_value_of_m_l52_52321

theorem purely_imaginary_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 2 * m ≠ 0) → m = -1/2 :=
by
  sorry

theorem third_quadrant_value_of_m (m : ℝ) :
  (2 * m^2 - 3 * m - 2 < 0) ∧ (m^2 - 2 * m < 0) → 0 < m ∧ m < 2 :=
by
  sorry

end purely_imaginary_value_of_m_third_quadrant_value_of_m_l52_52321


namespace find_remainder_l52_52910

theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = 686) (h2 : divisor = 36) (h3 : quotient = 19) :
  ∃ remainder, dividend = (divisor * quotient) + remainder ∧ remainder = 2 :=
by
  sorry

end find_remainder_l52_52910


namespace rope_length_loss_l52_52731

theorem rope_length_loss
  (stories_needed : ℕ)
  (feet_per_story : ℕ)
  (pieces_of_rope : ℕ)
  (feet_per_rope : ℕ)
  (total_feet_needed : ℕ)
  (total_feet_bought : ℕ)
  (percentage_lost : ℕ) :
  
  stories_needed = 6 →
  feet_per_story = 10 →
  pieces_of_rope = 4 →
  feet_per_rope = 20 →
  total_feet_needed = stories_needed * feet_per_story →
  total_feet_bought = pieces_of_rope * feet_per_rope →
  total_feet_needed <= total_feet_bought →
  percentage_lost = ((total_feet_bought - total_feet_needed) * 100) / total_feet_bought →
  percentage_lost = 25 :=
by
  intros h_stories h_feet_story h_pieces h_feet_rope h_total_needed h_total_bought h_needed_bought h_percentage
  sorry

end rope_length_loss_l52_52731


namespace player_A_success_l52_52563

/-- Representation of the problem conditions --/
structure GameState where
  coins : ℕ
  boxes : ℕ
  n_coins : ℕ 
  n_boxes : ℕ 
  arrangement: ℕ → ℕ 
  (h_coins : coins ≥ 2012)
  (h_boxes : boxes = 2012)
  (h_initial_distribution : (∀ b, arrangement b ≥ 1))
  
/-- The main theorem for player A to ensure at least 1 coin in each box --/
theorem player_A_success (s : GameState) : 
  s.coins ≥ 4022 → (∀ b, s.arrangement b ≥ 1) :=
by
  sorry

end player_A_success_l52_52563


namespace cards_ratio_l52_52361

variable (x : ℕ)

def partially_full_decks_cards := 3 * x
def full_decks_cards := 3 * 52
def total_cards_before := 200 + 34

theorem cards_ratio (h : 3 * x + full_decks_cards = total_cards_before) : x / 52 = 1 / 2 :=
by sorry

end cards_ratio_l52_52361


namespace prob_no_infection_correct_prob_one_infection_correct_l52_52776

-- Probability that no chicken is infected
def prob_no_infection (p_not_infected : ℚ) (n : ℕ) : ℚ := p_not_infected^n

-- Given
def p_not_infected : ℚ := 4 / 5
def n : ℕ := 5

-- Expected answer for no chicken infected
def expected_prob_no_infection : ℚ := 1024 / 3125

-- Lean statement
theorem prob_no_infection_correct : 
  prob_no_infection p_not_infected n = expected_prob_no_infection := by
  sorry

-- Probability that exactly one chicken is infected
def prob_one_infection (p_infected : ℚ) (p_not_infected : ℚ) (n : ℕ) : ℚ := 
  (n * p_not_infected^(n-1) * p_infected)

-- Given
def p_infected : ℚ := 1 / 5

-- Expected answer for exactly one chicken infected
def expected_prob_one_infection : ℚ := 256 / 625

-- Lean statement
theorem prob_one_infection_correct : 
  prob_one_infection p_infected p_not_infected n = expected_prob_one_infection := by
  sorry

end prob_no_infection_correct_prob_one_infection_correct_l52_52776


namespace linear_inequality_solution_set_l52_52283

variable (x : ℝ)

theorem linear_inequality_solution_set :
  ∀ x : ℝ, (2 * x - 4 > 0) → (x > 2) := 
by
  sorry

end linear_inequality_solution_set_l52_52283


namespace tree_height_at_3_years_l52_52040

-- Define the conditions as Lean definitions
def tree_height (years : ℕ) : ℕ :=
  2 ^ years

-- State the theorem using the defined conditions
theorem tree_height_at_3_years : tree_height 6 = 32 → tree_height 3 = 4 := by
  intro h
  sorry

end tree_height_at_3_years_l52_52040


namespace Martha_points_l52_52861

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end Martha_points_l52_52861


namespace train_length_correct_l52_52767

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l52_52767


namespace person_A_work_days_l52_52732

theorem person_A_work_days (x : ℝ) (h1 : 0 < x) 
                                 (h2 : ∃ b_work_rate, b_work_rate = 1 / 30) 
                                 (h3 : 5 * (1 / x + 1 / 30) = 0.5) : 
  x = 15 :=
by
-- Proof omitted
sorry

end person_A_work_days_l52_52732


namespace largest_K_inequality_l52_52245

theorem largest_K_inequality :
  ∃ K : ℕ, (K < 12) ∧ (10 * K = 110) := by
  use 11
  sorry

end largest_K_inequality_l52_52245


namespace sum_of_cube_faces_l52_52168

-- Define the cube numbers as consecutive integers starting from 15.
def cube_faces (faces : List ℕ) : Prop :=
  faces = [15, 16, 17, 18, 19, 20]

-- Define the condition that the sum of numbers on opposite faces is the same.
def opposite_faces_condition (pairs : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) = 35

theorem sum_of_cube_faces : ∃ faces : List ℕ, cube_faces faces ∧ (∃ pairs : List (ℕ × ℕ), opposite_faces_condition pairs ∧ faces.sum = 105) :=
by
  sorry

end sum_of_cube_faces_l52_52168


namespace moles_of_CaCl2_l52_52319

/-- 
We are given the reaction: CaCO3 + 2 HCl → CaCl2 + CO2 + H2O 
with 2 moles of HCl and 1 mole of CaCO3. We need to prove that the number 
of moles of CaCl2 formed is 1.
-/
theorem moles_of_CaCl2 (HCl: ℝ) (CaCO3: ℝ) (reaction: CaCO3 + 2 * HCl = 1): CaCO3 = 1 → HCl = 2 → CaCl2 = 1 :=
by
  -- importing the required context for chemical equations and stoichiometry
  sorry

end moles_of_CaCl2_l52_52319


namespace number_of_true_propositions_l52_52145

theorem number_of_true_propositions : 
  (∃ x y : ℝ, (x * y = 1) ↔ (x = y⁻¹ ∨ y = x⁻¹)) ∧
  (¬(∀ x : ℝ, (x > -3) → x^2 - x - 6 ≤ 0)) ∧
  (¬(∀ a b : ℝ, (a > b) → (a^2 < b^2))) ∧
  (¬(∀ x : ℝ, (x - 1/x > 0) → (x > -1))) →
  True := by
  sorry

end number_of_true_propositions_l52_52145


namespace maximum_probability_second_game_C_l52_52441

variables {p1 p2 p3 p : ℝ}

-- Define the probabilities and their conditions
axiom h1 : p3 > p2
axiom h2 : p2 > p1
axiom h3 : p1 > 0

-- Define the probabilities of winning two consecutive games in different orders
def P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def P_C := 2 * (p3 * (p1 + p2) - 2 * p1 * p2 * p3)

-- The main statement we need to prove
theorem maximum_probability_second_game_C : P_C > P_A ∧ P_C > P_B :=
by
  sorry

end maximum_probability_second_game_C_l52_52441


namespace value_of_m_l52_52883

theorem value_of_m :
  ∃ m : ℝ, (3 - 1) / (m + 2) = 1 → m = 0 :=
by 
  sorry

end value_of_m_l52_52883


namespace difference_mean_median_is_neg_half_l52_52866

-- Definitions based on given conditions
def scoreDistribution : List (ℕ × ℚ) :=
  [(65, 0.05), (75, 0.25), (85, 0.4), (95, 0.2), (105, 0.1)]

-- Defining the total number of students as 100 for easier percentage calculations
def totalStudents := 100

-- Definition to compute mean
def mean : ℚ :=
  scoreDistribution.foldl (λ acc (score, percentage) => acc + (↑score * percentage)) 0

-- Median score based on the distribution conditions
def median : ℚ := 85

-- Proving the proposition that the difference between the mean and the median is -0.5
theorem difference_mean_median_is_neg_half :
  median - mean = -0.5 :=
sorry

end difference_mean_median_is_neg_half_l52_52866


namespace strictly_increasing_arithmetic_seq_l52_52707

theorem strictly_increasing_arithmetic_seq 
  (s : ℕ → ℕ) 
  (hs_incr : ∀ n, s n < s (n + 1)) 
  (hs_seq1 : ∃ D1, ∀ n, s (s n) = s (s 0) + n * D1) 
  (hs_seq2 : ∃ D2, ∀ n, s (s n + 1) = s (s 0 + 1) + n * D2) : 
  ∃ d, ∀ n, s (n + 1) = s n + d :=
sorry

end strictly_increasing_arithmetic_seq_l52_52707


namespace factorize_expression_l52_52495

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end factorize_expression_l52_52495


namespace range_of_b_l52_52958

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  - (1/2) * (x - 2)^2 + b * Real.log x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 1 < x → f x b ≤ f 1 b) → b ≤ -1 :=
by
  sorry

end range_of_b_l52_52958


namespace amount_paid_for_peaches_l52_52112

noncomputable def cost_of_berries : ℝ := 7.19
noncomputable def change_received : ℝ := 5.98
noncomputable def total_bill : ℝ := 20

theorem amount_paid_for_peaches :
  total_bill - change_received - cost_of_berries = 6.83 :=
by
  sorry

end amount_paid_for_peaches_l52_52112


namespace combination_7_2_l52_52475

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l52_52475


namespace find_111th_digit_in_fraction_l52_52900

theorem find_111th_digit_in_fraction :
  let frac : ℚ := 33 / 555
  (decimal_rep : String) := "0.0overline594"
  (repeating_cycle_len : ℕ := 3)
  (position_mod : ℕ := 110 % repeating_cycle_len)
  (digit := "594".nth (position_mod).getD '0')
in digit = '9' :=
by
  sorry

end find_111th_digit_in_fraction_l52_52900


namespace value_of_m_l52_52397

theorem value_of_m (m : ℝ) : (m + 1, 3) ∈ {p : ℝ × ℝ | p.1 + p.2 + 1 = 0} → m = -5 :=
by
  intro h
  sorry

end value_of_m_l52_52397


namespace combination_7_2_l52_52474

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l52_52474


namespace num_valid_combinations_l52_52921

-- Definitions based on the conditions
def num_herbs := 4
def num_gems := 6
def num_incompatible_gems := 3
def num_incompatible_herbs := 2

-- Statement to be proved
theorem num_valid_combinations :
  (num_herbs * num_gems) - (num_incompatible_gems * num_incompatible_herbs) = 18 :=
by
  sorry

end num_valid_combinations_l52_52921


namespace john_purchased_large_bottles_l52_52240

noncomputable def large_bottle_cost : ℝ := 1.75
noncomputable def small_bottle_cost : ℝ := 1.35
noncomputable def num_small_bottles : ℝ := 690
noncomputable def avg_price_paid : ℝ := 1.6163438256658595
noncomputable def total_small_cost : ℝ := num_small_bottles * small_bottle_cost
noncomputable def total_cost (L : ℝ) : ℝ := large_bottle_cost * L + total_small_cost
noncomputable def total_bottles (L : ℝ) : ℝ := L + num_small_bottles

theorem john_purchased_large_bottles : ∃ L : ℝ, 
  (total_cost L / total_bottles L = avg_price_paid) ∧ 
  (L = 1380) := 
sorry

end john_purchased_large_bottles_l52_52240


namespace monthly_rent_is_3600_rs_l52_52920

def shop_length_feet : ℕ := 20
def shop_width_feet : ℕ := 15
def annual_rent_per_square_foot_rs : ℕ := 144

theorem monthly_rent_is_3600_rs :
  (shop_length_feet * shop_width_feet) * annual_rent_per_square_foot_rs / 12 = 3600 :=
by sorry

end monthly_rent_is_3600_rs_l52_52920


namespace suitable_bases_for_346_l52_52932

theorem suitable_bases_for_346 (b : ℕ) (hb : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0) : b = 6 ∨ b = 7 :=
sorry

end suitable_bases_for_346_l52_52932


namespace maisy_earns_more_l52_52252

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l52_52252


namespace classroom_students_count_l52_52593

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end classroom_students_count_l52_52593


namespace shifted_polynomial_roots_are_shifted_l52_52850

noncomputable def original_polynomial : Polynomial ℝ := Polynomial.X ^ 3 - 5 * Polynomial.X + 7
noncomputable def shifted_polynomial : Polynomial ℝ := Polynomial.X ^ 3 + 6 * Polynomial.X ^ 2 + 7 * Polynomial.X + 5

theorem shifted_polynomial_roots_are_shifted :
  (∀ (a b c : ℝ), (original_polynomial.eval a = 0) ∧ (original_polynomial.eval b = 0) ∧ (original_polynomial.eval c = 0) 
    → (shifted_polynomial.eval (a - 2) = 0) ∧ (shifted_polynomial.eval (b - 2) = 0) ∧ (shifted_polynomial.eval (c - 2) = 0)) :=
by
  sorry

end shifted_polynomial_roots_are_shifted_l52_52850


namespace eval_expr1_eval_expr2_l52_52046

theorem eval_expr1 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := 
by
  sorry

theorem eval_expr2 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^2 + b^2) / (a + b) = 5.8 :=
by
  sorry

end eval_expr1_eval_expr2_l52_52046


namespace hypotenuse_of_right_triangle_l52_52604

theorem hypotenuse_of_right_triangle (a b : ℕ) (ha : a = 140) (hb : b = 336) :
  Nat.sqrt (a * a + b * b) = 364 := by
  sorry

end hypotenuse_of_right_triangle_l52_52604


namespace find_r_l52_52799

theorem find_r (r : ℝ) (h : ⌊r⌋ + r = 16.5) : r = 8.5 :=
sorry

end find_r_l52_52799


namespace find_a_l52_52656

def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

theorem find_a (a : ℝ) (h : F a 3 4 = F a 2 5) : a = 1 / 2 :=
by
  sorry

end find_a_l52_52656


namespace average_speed_round_trip_l52_52640

theorem average_speed_round_trip (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (2 * m * n) / (m + n) = (2 * (m * n)) / (m + n) :=
  sorry

end average_speed_round_trip_l52_52640


namespace range_of_m_l52_52519

theorem range_of_m (m x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (h3 : (1 - 2 * m) / x1 < (1 - 2 * m) / x2) : m < 1 / 2 :=
sorry

end range_of_m_l52_52519


namespace percentage_shaded_is_18_75_l52_52355

-- conditions
def total_squares: ℕ := 16
def shaded_squares: ℕ := 3

-- claim to prove
theorem percentage_shaded_is_18_75 :
  ((shaded_squares : ℝ) / total_squares) * 100 = 18.75 := 
by
  sorry

end percentage_shaded_is_18_75_l52_52355


namespace det_my_matrix_l52_52646

def my_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, 1], ![-5, 5, -4], ![3, 3, 6]]

theorem det_my_matrix : my_matrix.det = 96 := by
  sorry

end det_my_matrix_l52_52646


namespace intersection_A_B_eq_B_l52_52081

-- Define set A
def setA : Set ℝ := { x : ℝ | x > -3 }

-- Define set B
def setB : Set ℝ := { x : ℝ | x ≥ 2 }

-- Theorem statement of proving the intersection of setA and setB is setB itself
theorem intersection_A_B_eq_B : setA ∩ setB = setB :=
by
  -- proof skipped
  sorry

end intersection_A_B_eq_B_l52_52081


namespace calvin_score_l52_52047

theorem calvin_score (C : ℚ) (h_paislee_score : (3/4) * C = 125) : C = 167 := 
  sorry

end calvin_score_l52_52047


namespace find_m_values_l52_52338

-- Defining the sets and conditions
def A : Set ℝ := { x | x ^ 2 - 9 * x - 10 = 0 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- Stating the proof problem
theorem find_m_values : {m | A ∪ B m = A} = {0, 1, -1 / 10} :=
by
  sorry

end find_m_values_l52_52338


namespace find_integer_l52_52710

theorem find_integer (a b c d : ℕ) (h1 : a + b + c + d = 18) 
  (h2 : b + c = 11) (h3 : a - d = 3) (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  10^3 * a + 10^2 * b + 10 * c + d = 5262 ∨ 10^3 * a + 10^2 * b + 10 * c + d = 5622 := 
by
  sorry

end find_integer_l52_52710


namespace pop_spending_original_l52_52264

-- Given conditions
def total_spent := 150
def crackle_spending (P : ℝ) := 3 * P
def snap_spending (P : ℝ) := 2 * crackle_spending P

-- Main statement to prove
theorem pop_spending_original : ∃ P : ℝ, snap_spending P + crackle_spending P + P = total_spent ∧ P = 15 :=
by
  sorry

end pop_spending_original_l52_52264


namespace grandpa_rank_l52_52992

theorem grandpa_rank (mom dad grandpa : ℕ) 
  (h1 : mom < dad) 
  (h2 : dad < grandpa) : 
  ∀ rank: ℕ, rank = 3 := 
by
  sorry

end grandpa_rank_l52_52992


namespace exactly_one_greater_than_one_l52_52720

theorem exactly_one_greater_than_one (x1 x2 x3 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h4 : x1 * x2 * x3 = 1)
  (h5 : x1 + x2 + x3 > (1 / x1) + (1 / x2) + (1 / x3)) :
  (x1 > 1 ∧ x2 ≤ 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 > 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 ≤ 1 ∧ x3 > 1) :=
sorry

end exactly_one_greater_than_one_l52_52720


namespace solve_equation1_solve_equation2_l52_52706

-- Lean 4 statements for the given problems:
theorem solve_equation1 (x : ℝ) (h : x ≠ 0) : (2 / x = 3 / (x + 2)) ↔ (x = 4) := by
  sorry

theorem solve_equation2 (x : ℝ) (h : x ≠ 2) : ¬(5 / (x - 2) + 1 = (x - 7) / (2 - x)) := by
  sorry

end solve_equation1_solve_equation2_l52_52706


namespace row_trip_time_example_l52_52033

noncomputable def round_trip_time
    (rowing_speed : ℝ)
    (current_speed : ℝ)
    (total_distance : ℝ) : ℝ :=
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  let one_way_distance := total_distance / 2
  let time_to_place := one_way_distance / downstream_speed
  let time_back := one_way_distance / upstream_speed
  time_to_place + time_back

theorem row_trip_time_example :
  round_trip_time 10 2 96 = 10 := by
  sorry

end row_trip_time_example_l52_52033


namespace unique_array_count_l52_52035

theorem unique_array_count (n m : ℕ) (h_conds : n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :
  ∃! (n m : ℕ), (n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :=
by
  sorry

end unique_array_count_l52_52035


namespace find_p_l52_52074

variables (a b c p : ℝ)

theorem find_p 
  (h1 : 9 / (a + b) = 13 / (c - b)) : 
  p = 22 :=
sorry

end find_p_l52_52074


namespace evaluate_f_of_composed_g_l52_52851

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 2

theorem evaluate_f_of_composed_g :
  f (2 + g 3) = 17 :=
by
  sorry

end evaluate_f_of_composed_g_l52_52851


namespace part1_part2_l52_52948

-- Part (1): Solution set of the inequality
theorem part1 (x : ℝ) : (|x - 1| + |x + 1| ≤ 8 - x^2) ↔ (-2 ≤ x) ∧ (x ≤ 2) :=
by
  sorry

-- Part (2): Range of real number t
theorem part2 (t : ℝ) (m n : ℝ) (x : ℝ) (h1 : m + n = 4) (h2 : m > 0) (h3 : n > 0) :  
  |x-t| + |x+t| = (4 * m^2 + n) / (m * n) → t ≥ 9 / 8 ∨ t ≤ -9 / 8 :=
by
  sorry

end part1_part2_l52_52948


namespace envelope_addressing_equation_l52_52759

theorem envelope_addressing_equation (x : ℝ) :
  (800 / 10 + 800 / x + 800 / 5) * (3 / 800) = 1 / 3 :=
  sorry

end envelope_addressing_equation_l52_52759


namespace eight_diamond_three_l52_52075

def diamond (x y : ℤ) : ℤ := sorry

axiom diamond_zero (x : ℤ) : diamond x 0 = x
axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x
axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

theorem eight_diamond_three : diamond 8 3 = 39 :=
sorry

end eight_diamond_three_l52_52075


namespace oliver_first_coupon_redeem_on_friday_l52_52865

-- Definitions of conditions in the problem
def has_coupons (n : ℕ) := n = 8
def uses_coupon_every_9_days (days : ℕ) := days = 9
def is_closed_on_monday (day : ℕ) := day % 7 = 1  -- Assuming 1 represents Monday
def does_not_redeem_on_closed_day (redemption_days : List ℕ) :=
  ∀ day ∈ redemption_days, day % 7 ≠ 1

-- Main theorem statement
theorem oliver_first_coupon_redeem_on_friday : 
  ∃ (first_redeem_day: ℕ), 
  has_coupons 8 ∧ uses_coupon_every_9_days 9 ∧
  is_closed_on_monday 1 ∧ 
  does_not_redeem_on_closed_day [first_redeem_day, first_redeem_day + 9, first_redeem_day + 18, first_redeem_day + 27, first_redeem_day + 36, first_redeem_day + 45, first_redeem_day + 54, first_redeem_day + 63] ∧ 
  first_redeem_day % 7 = 5 := sorry

end oliver_first_coupon_redeem_on_friday_l52_52865


namespace sufficient_condition_for_proposition_l52_52810

theorem sufficient_condition_for_proposition :
  ∀ (a : ℝ), (0 < a ∧ a < 4) → (∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) := 
sorry

end sufficient_condition_for_proposition_l52_52810


namespace condition_sufficient_necessary_l52_52288

theorem condition_sufficient_necessary (x : ℝ) :
  (1 / 3) ^ x < 1 → x > 0 ∧ (0 < x ∧ x < 1) :=
by {
  intros h,
  have h1 : (1 / 3) ^ 0 = 1, by norm_num,
  have h2 : x > 0, by {
    apply (real.rpow_lt_rpow_iff (by norm_num) zero_lt_one).mpr,
    rwa h1,
  },
  have h3 : x < 1, by {
    have h_aux : 1 / x > 1, from by { assumption },
    linarith [(by linear_combination h_aux : 1 / x < 1)],
  },
  exact ⟨h2, ⟨h2, h3⟩⟩
}

end condition_sufficient_necessary_l52_52288


namespace son_l52_52634

variable (S M : ℕ)
variable h1 : M = S + 26
variable h2 : M + 2 = 2 * (S + 2)

theorem son's_age_is_24 : S = 24 :=
by
  sorry

end son_l52_52634


namespace enrico_earnings_l52_52061

def roosterPrice (weight: ℕ) : ℝ :=
  if weight < 20 then weight * 0.80
  else if weight ≤ 35 then weight * 0.65
  else weight * 0.50

theorem enrico_earnings :
  roosterPrice 15 + roosterPrice 30 + roosterPrice 40 + roosterPrice 50 = 76.50 := 
by
  sorry

end enrico_earnings_l52_52061


namespace num_people_second_hour_l52_52293

theorem num_people_second_hour 
  (n1_in n2_in n1_left n2_left : ℕ) 
  (rem_hour1 rem_hour2 : ℕ)
  (h1 : n1_in = 94)
  (h2 : n1_left = 27)
  (h3 : n2_left = 9)
  (h4 : rem_hour2 = 76)
  (h5 : rem_hour1 = n1_in - n1_left)
  (h6 : rem_hour2 = rem_hour1 + n2_in - n2_left) :
  n2_in = 18 := 
  by 
  sorry

end num_people_second_hour_l52_52293


namespace distribute_items_l52_52368

open Nat

def g (n k : ℕ) : ℕ :=
  -- This is a placeholder for the actual function definition
  sorry

theorem distribute_items (n k : ℕ) (h : n ≥ k ∧ k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
by
  sorry

end distribute_items_l52_52368


namespace batsman_average_l52_52433

theorem batsman_average (A : ℝ) (h1 : 24 * A < 95) 
                        (h2 : 24 * A + 95 = 25 * (A + 3.5)) : A + 3.5 = 11 :=
by
  sorry

end batsman_average_l52_52433


namespace unique_fraction_10_percent_increase_l52_52305

theorem unique_fraction_10_percent_increase :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ Int.gcd x y = 1 ∧ (y + 1) * (10 * x) = 11 * x * (y + 1) :=
by
  sorry

end unique_fraction_10_percent_increase_l52_52305


namespace area_of_quadrilateral_NLMK_l52_52809

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_quadrilateral_NLMK 
  (AB BC AC AK CN CL : ℝ)
  (h_AB : AB = 13)
  (h_BC : BC = 20)
  (h_AC : AC = 21)
  (h_AK : AK = 4)
  (h_CN : CN = 1)
  (h_CL : CL = 20 / 21) : 
  triangle_area AB BC AC - 
  (1 * CL / (BC * AC) * triangle_area AB BC AC) - 
  (9 * (BC - CN) / (AB * BC) * triangle_area AB BC AC) -
  (16 * 41 / (169 * 21) * triangle_area AB BC AC) = 
  493737 / 11830 := 
sorry

end area_of_quadrilateral_NLMK_l52_52809


namespace total_students_correct_l52_52116

theorem total_students_correct (H : ℕ)
  (B : ℕ := 2 * H)
  (P : ℕ := H + 5)
  (S : ℕ := 3 * (H + 5))
  (h1 : B = 30)
  : (B + H + P + S) = 125 := by
  sorry

end total_students_correct_l52_52116


namespace hitting_probability_l52_52034

theorem hitting_probability (P_miss : ℝ) (P_6 P_7 P_8 P_9 P_10 : ℝ) :
  P_miss = 0.2 →
  P_6 = 0.1 →
  P_7 = 0.2 →
  P_8 = 0.3 →
  P_9 = 0.15 →
  P_10 = 0.05 →
  1 - P_miss = 0.8 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hitting_probability_l52_52034


namespace range_of_f_gt_f_of_quadratic_l52_52657

-- Define the function f and its properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Define the problem statement
theorem range_of_f_gt_f_of_quadratic (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_inc : is_increasing_on_pos f) :
  {x : ℝ | f x > f (x^2 - 2*x + 2)} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end range_of_f_gt_f_of_quadratic_l52_52657


namespace zero_function_l52_52498

noncomputable def f : ℝ → ℝ := sorry

theorem zero_function :
  (∀ x y : ℝ, f x + f y = f (f x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro h
  sorry

end zero_function_l52_52498


namespace distance_between_first_and_last_tree_l52_52493

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h₁ : n = 8)
  (h₂ : d = 75)
  : (d / ((4 - 1) : ℕ)) * (n - 1) = 175 := sorry

end distance_between_first_and_last_tree_l52_52493


namespace Mehki_is_10_years_older_than_Jordyn_l52_52373

def Zrinka_age : Nat := 6
def Mehki_age : Nat := 22
def Jordyn_age : Nat := 2 * Zrinka_age

theorem Mehki_is_10_years_older_than_Jordyn : Mehki_age - Jordyn_age = 10 :=
by
  sorry

end Mehki_is_10_years_older_than_Jordyn_l52_52373


namespace largest_sequence_sum_45_l52_52589

theorem largest_sequence_sum_45 
  (S: ℕ → ℕ)
  (h_S: ∀ n, S n = n * (n + 1) / 2)
  (h_sum: ∃ m: ℕ, S m = 45):
  (∃ k: ℕ, k ≤ 9 ∧ S k = 45) ∧ (∀ m: ℕ, S m ≤ 45 → m ≤ 9) :=
by
  sorry

end largest_sequence_sum_45_l52_52589


namespace find_q_l52_52654

theorem find_q (p q d : ℝ) (h₁ : (-p / 3) = q) (h₂ : 1 + p + q + d = q) (h₃ : d = 7) : q = 8 / 3 :=
by
  sorry

end find_q_l52_52654


namespace find_prime_pairs_l52_52963

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem find_prime_pairs :
  ∀ m n : ℕ,
  is_prime m → is_prime n → (m < n ∧ n < 5 * m) → is_prime (m + 3 * n) →
  (m = 2 ∧ (n = 3 ∨ n = 5 ∨ n = 7)) :=
by
  sorry

end find_prime_pairs_l52_52963


namespace work_alone_days_l52_52615

theorem work_alone_days (d : ℝ) (p q : ℝ) (h1 : q = 10) (h2 : 2 * (1/d + 1/q) = 0.3) : d = 20 :=
by
  sorry

end work_alone_days_l52_52615


namespace bus_seats_capacity_l52_52832

-- Define the conditions
variable (x : ℕ) -- number of people each seat can hold
def left_side_seats := 15
def right_side_seats := left_side_seats - 3
def back_seat_capacity := 7
def total_capacity := left_side_seats * x + right_side_seats * x + back_seat_capacity

-- State the theorem
theorem bus_seats_capacity :
  total_capacity x = 88 → x = 3 := by
  sorry

end bus_seats_capacity_l52_52832


namespace total_pennies_l52_52964

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end total_pennies_l52_52964


namespace largest_prime_factor_1729_l52_52014

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l52_52014


namespace distance_between_trees_l52_52745

-- Lean 4 statement for the proof problem
theorem distance_between_trees (n : ℕ) (yard_length : ℝ) (h_n : n = 26) (h_length : yard_length = 600) :
  yard_length / (n - 1) = 24 :=
by
  sorry

end distance_between_trees_l52_52745


namespace area_of_rectangle_l52_52391

theorem area_of_rectangle (M N P Q R S X Y : Type) 
  (PQ : ℝ) (PX XY YQ : ℝ) (R_perpendicular_to_PQ S_perpendicular_to_PQ : Prop) 
  (R_through_M S_through_Q : Prop) 
  (segment_lengths : PQ = PX + XY + YQ) : PQ = 5 ∧ PX = 1 ∧ XY = 2 ∧ YQ = 2 
  → 2 * (1/2 * PQ * 2) = 10 :=
  sorry

end area_of_rectangle_l52_52391


namespace largest_prime_factor_of_1729_l52_52007

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l52_52007


namespace natural_numbers_divisible_by_6_l52_52508

theorem natural_numbers_divisible_by_6 :
  {n : ℕ | 2 ≤ n ∧ n ≤ 88 ∧ 6 ∣ n} = {n | n = 6 * k ∧ 1 ≤ k ∧ k ≤ 14} :=
by
  sorry

end natural_numbers_divisible_by_6_l52_52508


namespace meeting_equation_correct_l52_52611

-- Define the conditions
def distance : ℝ := 25
def time : ℝ := 3
def speed_Xiaoming : ℝ := 4
def speed_Xiaogang (x : ℝ) : ℝ := x

-- The target equation derived from conditions which we need to prove valid.
theorem meeting_equation_correct (x : ℝ) : 3 * (speed_Xiaoming + speed_Xiaogang x) = distance :=
by
  sorry

end meeting_equation_correct_l52_52611


namespace map_scale_l52_52117

theorem map_scale (cm12_km90 : 12 * (1 / 90) = 1) : 20 * (90 / 12) = 150 :=
by
  sorry

end map_scale_l52_52117


namespace farmer_rows_of_tomatoes_l52_52169

def num_rows (total_tomatoes yield_per_plant plants_per_row : ℕ) : ℕ :=
  (total_tomatoes / yield_per_plant) / plants_per_row

theorem farmer_rows_of_tomatoes (total_tomatoes yield_per_plant plants_per_row : ℕ)
    (ht : total_tomatoes = 6000)
    (hy : yield_per_plant = 20)
    (hp : plants_per_row = 10) :
    num_rows total_tomatoes yield_per_plant plants_per_row = 30 := 
by
  sorry

end farmer_rows_of_tomatoes_l52_52169


namespace max_watched_hours_l52_52696

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end max_watched_hours_l52_52696


namespace flowers_sold_l52_52505

theorem flowers_sold (lilacs roses gardenias : ℕ) 
  (h1 : lilacs = 10)
  (h2 : roses = 3 * lilacs)
  (h3 : gardenias = lilacs / 2) : 
  lilacs + roses + gardenias = 45 :=
by
  sorry

end flowers_sold_l52_52505


namespace axis_of_symmetry_l52_52124

theorem axis_of_symmetry :
  ∀ x : ℝ, (g : ℝ → ℝ), (f : ℝ → ℝ) 
  (g = λ x , 3 * Real.sin (2 * x - π / 6))
  (f = λ x , 3 * Real.sin (4 * x + π / 6)),
  (∃ (k : ℤ), x = k * π / 2 + π / 3) :=
begin
  sorry
end

end axis_of_symmetry_l52_52124


namespace math_problems_l52_52955

theorem math_problems (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (a * (6 - a) ≤ 9) ∧
  (ab = a + b + 3 → ab ≥ 9) ∧
  ¬(∀ x : ℝ, 0 < x → x^2 + 4 / (x^2 + 3) ≥ 1) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end math_problems_l52_52955


namespace find_initial_time_l52_52623

-- The initial distance d
def distance : ℕ := 288

-- Conditions
def initial_condition (v t : ℕ) : Prop :=
  distance = v * t

def new_condition (t : ℕ) : Prop :=
  distance = 32 * (3 * t / 2)

-- Proof Problem Statement
theorem find_initial_time (v t : ℕ) (h1 : initial_condition v t)
  (h2 : new_condition t) : t = 6 := by
  sorry

end find_initial_time_l52_52623


namespace tangent_lines_ln_e_proof_l52_52825

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end tangent_lines_ln_e_proof_l52_52825


namespace max_watched_hours_l52_52695

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end max_watched_hours_l52_52695


namespace count_n_with_product_zero_l52_52502

theorem count_n_with_product_zero : 
  (Finset.filter (λ n, 1 ≤ n ∧ n ≤ 2016 ∧ 
    ∃ k, 0 ≤ k ∧ k < n ∧ (2 + Real.cos (4 * Real.pi * k / n) + Real.sin (4 * Real.pi * k / n) * Complex.i) ^ n = 1) 
      (Finset.range (2017))).card = 504 :=
sorry

end count_n_with_product_zero_l52_52502


namespace magazine_page_height_l52_52253

theorem magazine_page_height
  (charge_per_sq_inch : ℝ := 8)
  (half_page_cost : ℝ := 432)
  (page_width : ℝ := 12) : 
  ∃ h : ℝ, (1/2) * h * page_width * charge_per_sq_inch = half_page_cost :=
by sorry

end magazine_page_height_l52_52253


namespace sum_and_times_l52_52409

theorem sum_and_times 
  (a : ℕ) (ha : a = 99) 
  (b : ℕ) (hb : b = 301) 
  (c : ℕ) (hc : c = 200) : 
  a + b = 2 * c :=
by 
  -- skipping proof 
  sorry

end sum_and_times_l52_52409


namespace find_significance_level_l52_52414

noncomputable def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![[10, 40], [20, 30]]

def n := 100

def chi_squared_value (n : ℕ) (a b c d : ℕ) : ℚ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def K_squared := chi_squared_value n 10 40 20 30

def k0 := 3.841
def k0_prob := 0.05

theorem find_significance_level :
  K_squared <= 4.762 :=
by
  sorry

end find_significance_level_l52_52414


namespace martha_total_points_l52_52859

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end martha_total_points_l52_52859


namespace ratio_of_inscribed_squares_in_isosceles_right_triangle_l52_52043

def isosceles_right_triangle (a b : ℝ) (leg : ℝ) : Prop :=
  let a_square_inscribed := a = leg
  let b_square_inscribed := b = leg
  a_square_inscribed ∧ b_square_inscribed

theorem ratio_of_inscribed_squares_in_isosceles_right_triangle (a b leg : ℝ)
  (h : isosceles_right_triangle a b leg) :
  leg = 6 ∧ a = leg ∧ b = leg → a / b = 1 := 
by {
  sorry -- the proof will go here
}

end ratio_of_inscribed_squares_in_isosceles_right_triangle_l52_52043


namespace sum_of_remainders_l52_52018

theorem sum_of_remainders (a b c : ℕ) 
  (ha : a % 15 = 12) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 9 := 
by 
  sorry

end sum_of_remainders_l52_52018


namespace volume_of_cube_l52_52730

theorem volume_of_cube (SA : ℝ) (H : SA = 600) : (10^3 : ℝ) = 1000 :=
by
  sorry

end volume_of_cube_l52_52730


namespace sister_height_on_birthday_l52_52561

theorem sister_height_on_birthday (previous_height : ℝ) (growth_rate : ℝ)
    (h_previous_height : previous_height = 139.65)
    (h_growth_rate : growth_rate = 0.05) :
    previous_height * (1 + growth_rate) = 146.6325 :=
by
  -- Proof omitted
  sorry

end sister_height_on_birthday_l52_52561


namespace difference_is_four_l52_52925

open Nat

-- Assume we have a 5x5x5 cube
def cube_side_length : ℕ := 5
def total_unit_cubes : ℕ := cube_side_length ^ 3

-- Define the two configurations
def painted_cubes_config1 : ℕ := 65  -- Two opposite faces and one additional face
def painted_cubes_config2 : ℕ := 61  -- Three adjacent faces

-- The difference in the number of unit cubes with at least one painted face
def painted_difference : ℕ := painted_cubes_config1 - painted_cubes_config2

theorem difference_is_four :
    painted_difference = 4 := by
  sorry

end difference_is_four_l52_52925


namespace series_sum_equals_one_sixth_l52_52056

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end series_sum_equals_one_sixth_l52_52056


namespace is_not_prime_390629_l52_52357

theorem is_not_prime_390629 : ¬ Prime 390629 :=
sorry

end is_not_prime_390629_l52_52357


namespace product_of_two_numbers_l52_52875

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l52_52875


namespace hexagon_perimeter_l52_52396

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

end hexagon_perimeter_l52_52396


namespace min_balls_draw_l52_52273

def box1_red := 40
def box1_green := 30
def box1_yellow := 25
def box1_blue := 15

def box2_red := 35
def box2_green := 25
def box2_yellow := 20

def min_balls_to_draw_to_get_20_balls_of_single_color (totalRed totalGreen totalYellow totalBlue : ℕ) : ℕ :=
  let maxNoColor :=
    (min totalRed 19) + (min totalGreen 19) + (min totalYellow 19) + (min totalBlue 15)
  maxNoColor + 1

theorem  min_balls_draw {r1 r2 g1 g2 y1 y2 b1 : ℕ} :
  r1 = box1_red -> g1 = box1_green -> y1 = box1_yellow -> b1 = box1_blue ->
  r2 = box2_red -> g2 = box2_green -> y2 = box2_yellow ->
  min_balls_to_draw_to_get_20_balls_of_single_color (r1 + r2) (g1 + g2) (y1 + y2) b1 = 73 :=
by
  intros
  unfold min_balls_to_draw_to_get_20_balls_of_single_color
  sorry

end min_balls_draw_l52_52273


namespace integer_solution_range_l52_52348

theorem integer_solution_range {m : ℝ} : 
  (∀ x : ℤ, -1 ≤ x → x < m → (x = -1 ∨ x = 0)) ↔ (0 < m ∧ m ≤ 1) :=
by 
  sorry

end integer_solution_range_l52_52348


namespace product_of_three_numbers_l52_52411

theorem product_of_three_numbers : 
  ∃ x y z : ℚ, x + y + z = 30 ∧ x = 3 * (y + z) ∧ y = 6 * z ∧ x * y * z = 23625 / 686 :=
by
  sorry

end product_of_three_numbers_l52_52411


namespace john_income_increase_l52_52980

theorem john_income_increase :
  let initial_job_income := 60
  let initial_freelance_income := 40
  let initial_online_sales_income := 20

  let new_job_income := 120
  let new_freelance_income := 60
  let new_online_sales_income := 35

  let weeks_per_month := 4

  let initial_monthly_income := (initial_job_income + initial_freelance_income + initial_online_sales_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income + new_online_sales_income) * weeks_per_month
  
  let percentage_increase := 100 * (new_monthly_income - initial_monthly_income) / initial_monthly_income

  percentage_increase = 79.17 := by
  sorry

end john_income_increase_l52_52980


namespace tangent_line_problem_l52_52827

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end tangent_line_problem_l52_52827


namespace choosing_six_adjacent_numbers_number_of_ways_no_consecutive_numbers_l52_52997

theorem choosing_six_adjacent_numbers 
  (s : Finset ℕ) 
  (h₁ : s ⊆ Finset.range 49) 
  (h₂ : s.card = 6): 
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ (a = b + 1 ∨ a = b - 1) := 
by
  -- placeholder for the actual proof
  sorry

theorem number_of_ways_no_consecutive_numbers :
  (Finset.range 44).choose 6 = Nat.choose 49 6 - Nat.choose 44 6 :=
by
  -- placeholder for the actual proof
  sorry

end choosing_six_adjacent_numbers_number_of_ways_no_consecutive_numbers_l52_52997


namespace maximum_unique_walks_l52_52874

-- Define the conditions
def starts_at_A : Prop := true
def crosses_bridge_1_first : Prop := true
def finishes_at_B : Prop := true
def six_bridges_linking_two_islands_and_banks : Prop := true

-- Define the theorem to prove the maximum number of unique walks is 6
theorem maximum_unique_walks : starts_at_A ∧ crosses_bridge_1_first ∧ finishes_at_B ∧ six_bridges_linking_two_islands_and_banks → ∃ n, n = 6 :=
by
  intros
  existsi 6
  sorry

end maximum_unique_walks_l52_52874


namespace hyperbola_equation_is_correct_l52_52219

-- Given Conditions
def hyperbola_eq (x y : ℝ) (a : ℝ) : Prop := (x^2) / (a^2) - (y^2) / 4 = 1
def asymptote_eq (x y : ℝ) : Prop := y = (1 / 2) * x

-- Correct answer to be proven
def hyperbola_correct (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 4 = 1

theorem hyperbola_equation_is_correct (x y : ℝ) (a : ℝ) :
  (hyperbola_eq x y a) → (asymptote_eq x y) → (a = 4) → hyperbola_correct x y :=
by 
  intros h_hyperbola h_asymptote h_a
  sorry

end hyperbola_equation_is_correct_l52_52219


namespace no_prime_roots_of_quadratic_l52_52784

open Int Nat

theorem no_prime_roots_of_quadratic (k : ℤ) :
  ¬ (∃ p q : ℤ, Prime p ∧ Prime q ∧ p + q = 107 ∧ p * q = k) :=
by
  sorry

end no_prime_roots_of_quadratic_l52_52784


namespace sam_age_l52_52575

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l52_52575


namespace bee_paths_to_hive_6_correct_l52_52782

noncomputable def num_paths_to_hive_6 : ℕ := 21

theorem bee_paths_to_hive_6_correct
  (start_pos : ℕ)
  (end_pos : ℕ)
  (bee_can_only_crawl : Prop)
  (bee_can_move_right : Prop)
  (bee_can_move_upper_right : Prop)
  (bee_can_move_lower_right : Prop)
  (total_hives : ℕ)
  (start_pos_is_initial : start_pos = 0)
  (end_pos_is_six : end_pos = 6) :
  num_paths_to_hive_6 = 21 :=
by
  sorry

end bee_paths_to_hive_6_correct_l52_52782


namespace product_modulo_l52_52384

theorem product_modulo : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m := 
  sorry

end product_modulo_l52_52384


namespace simplify_expression_l52_52122

theorem simplify_expression :
  let a := 2
  let b := -3
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := sorry

end simplify_expression_l52_52122


namespace largest_divisor_of_n4_minus_n2_is_12_l52_52714

theorem largest_divisor_of_n4_minus_n2_is_12 : ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  intro n
  -- Placeholder for proof; the detailed steps of the proof go here
  sorry

end largest_divisor_of_n4_minus_n2_is_12_l52_52714


namespace martha_total_points_l52_52858

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end martha_total_points_l52_52858


namespace compare_f_neg_x1_neg_x2_l52_52808

noncomputable def f : ℝ → ℝ := sorry

theorem compare_f_neg_x1_neg_x2 
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x)) 
  (h2 : ∀ x y : ℝ, 1 ≤ x → 1 ≤ y → x < y → f x < f y)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx2 : x2 > 0)
  (hx1x2 : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
by sorry

end compare_f_neg_x1_neg_x2_l52_52808


namespace evaluate_expression_l52_52062

theorem evaluate_expression :
  (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 :=
by
  sorry

end evaluate_expression_l52_52062


namespace find_a_and_b_find_sqrt_4a_plus_b_l52_52816

section

variable {a b : ℤ}

-- Define the conditions
def condition1 (a b : ℤ) := ((3 * a - 14)^2 = (a - 2)^2) 
def condition2 (b : ℤ) := (b - 15) ^ (3 : ℝ) = -27

-- Prove the given values of a and b
theorem find_a_and_b : ∃ a b : ℤ, condition1 a b ∧ condition2 b :=
begin
  use [4, -12],
  split,
  -- prove condition1
  { unfold condition1, norm_num },
  -- prove condition2
  { unfold condition2, norm_num }
end

-- Prove the square root of 4a + b
theorem find_sqrt_4a_plus_b (a b : ℤ) (h₁ : condition1 a b) (h₂ : condition2 b) : 
  sqrt (4 * a + b) = 2 ∨ sqrt (4 * a + b) = -2 :=
begin
  -- use given conditions to prove the statement
  have ha : a = 4, from sorry,
  have hb : b = -12, from sorry,
  rw [ha, hb],
  norm_num,
  left,
  norm_num,
end

end

end find_a_and_b_find_sqrt_4a_plus_b_l52_52816


namespace maximum_p_l52_52986

noncomputable def p (a b c : ℝ) : ℝ :=
  (2 / (a ^ 2 + 1)) - (2 / (b ^ 2 + 1)) + (3 / (c ^ 2 + 1))

theorem maximum_p (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : abc + a + c = b) : 
  p a b c ≤ 10 / 3 ∧ ∃ a b c, abc + a + c = b ∧ p a b c = 10 / 3 :=
sorry

end maximum_p_l52_52986


namespace ab_is_4_l52_52090

noncomputable def ab_value (a b : ℝ) : ℝ :=
  8 / (0.5 * (8 / a) * (8 / b))

theorem ab_is_4 (a b : ℝ) (ha : a > 0) (hb : b > 0) (area_condition : ab_value a b = 8) : a * b = 4 :=
  by
  sorry

end ab_is_4_l52_52090


namespace large_rectangle_perimeter_correct_l52_52918

def perimeter_of_square (p : ℕ) : ℕ :=
  p / 4

def perimeter_of_rectangle (p : ℕ) (l : ℕ) : ℕ :=
  (p - 2 * l) / 2

def perimeter_of_large_rectangle (side_length_of_square side_length_of_rectangle : ℕ) : ℕ :=
  let height := side_length_of_square + 2 * side_length_of_rectangle
  let width := 3 * side_length_of_square
  2 * (height + width)

theorem large_rectangle_perimeter_correct :
  let side_length_of_square := perimeter_of_square 24
  let side_length_of_rectangle := perimeter_of_rectangle 16 side_length_of_square
  perimeter_of_large_rectangle side_length_of_square side_length_of_rectangle = 52 :=
by
  sorry

end large_rectangle_perimeter_correct_l52_52918


namespace recycling_drive_target_l52_52312

-- Define the collection totals for each section
def section_collections_first_week : List ℝ := [260, 290, 250, 270, 300, 310, 280, 265]

-- Compute total collection for the first week
def total_first_week (collections: List ℝ) : ℝ := collections.sum

-- Compute collection for the second week with a 10% increase
def second_week_increase (collection: ℝ) : ℝ := collection * 1.10
def total_second_week (collections: List ℝ) : ℝ := (collections.map second_week_increase).sum

-- Compute collection for the third week with a 30% increase from the second week
def third_week_increase (collection: ℝ) : ℝ := collection * 1.30
def total_third_week (collections: List ℝ) : ℝ := (collections.map (second_week_increase)).sum * 1.30

-- Total target collection is the sum of collections for three weeks
def target (collections: List ℝ) : ℝ := total_first_week collections + total_second_week collections + total_third_week collections

-- Main theorem to prove
theorem recycling_drive_target : target section_collections_first_week = 7854.25 :=
by
  sorry -- skipping the proof

end recycling_drive_target_l52_52312


namespace length_of_AD_in_parallelogram_l52_52838

theorem length_of_AD_in_parallelogram
  (x : ℝ)
  (AB BC CD : ℝ)
  (AB_eq : AB = x + 3)
  (BC_eq : BC = x - 4)
  (CD_eq : CD = 16)
  (parallelogram_ABCD : AB = CD ∧ AD = BC) :
  AD = 9 := by
sorry

end length_of_AD_in_parallelogram_l52_52838


namespace books_finished_correct_l52_52254

def miles_traveled : ℕ := 6760
def miles_per_book : ℕ := 450
def books_finished (miles_traveled miles_per_book : ℕ) : ℕ :=
  miles_traveled / miles_per_book

theorem books_finished_correct :
  books_finished miles_traveled miles_per_book = 15 :=
by
  -- The steps of the proof would go here
  sorry

end books_finished_correct_l52_52254


namespace chess_tournament_total_players_l52_52098

-- Define the conditions

def total_points_calculation (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 132

def games_played (n : ℕ) : ℕ :=
  ((n + 12) * (n + 11)) / 2

theorem chess_tournament_total_players :
  ∃ n, total_points_calculation n = games_played n ∧ n + 12 = 34 :=
by {
  -- Assume n is found such that all conditions are satisfied
  use 22,
  -- Provide the necessary equations and conditions
  sorry
}

end chess_tournament_total_players_l52_52098


namespace smallest_integer_no_inverse_mod_77_66_l52_52742

theorem smallest_integer_no_inverse_mod_77_66 :
  ∃ a : ℕ, 0 < a ∧ a = 11 ∧ gcd a 77 > 1 ∧ gcd a 66 > 1 :=
by
  sorry

end smallest_integer_no_inverse_mod_77_66_l52_52742


namespace quotient_multiple_of_y_l52_52153

theorem quotient_multiple_of_y (x y m : ℤ) (h1 : x = 11 * y + 4) (h2 : 2 * x = 8 * m * y + 3) (h3 : 13 * y - x = 1) : m = 3 :=
by
  sorry

end quotient_multiple_of_y_l52_52153


namespace range_of_a_for_f_increasing_l52_52326

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a_for_f_increasing :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_for_f_increasing_l52_52326


namespace math_problem_l52_52581

variables (x y : ℝ)

noncomputable def question_value (x y : ℝ) : ℝ := (2 * x - 5 * y) / (5 * x + 2 * y)

theorem math_problem 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (cond : (5 * x - 2 * y) / (2 * x + 3 * y) = 1) : 
  question_value x y = -5 / 31 :=
sorry

end math_problem_l52_52581


namespace function_odd_on_domain_l52_52394

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem function_odd_on_domain :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x :=
by
  intros x h
  sorry

end function_odd_on_domain_l52_52394


namespace jellybean_probability_l52_52163

theorem jellybean_probability :
  let total_jellybeans := 12,
      red_jellybeans := 5,
      blue_jellybeans := 3,
      white_jellybeans := 4,
      total_pick := 3,
      two_blue_one_white := (choose blue_jellybeans 2) * (choose white_jellybeans 1),
      total_ways := choose total_jellybeans total_pick in
  (two_blue_one_white / total_ways : ℚ) = 3 / 55 :=
by
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 4
  let total_pick := 3
  let two_blue_one_white := choose blue_jellybeans 2 * choose white_jellybeans 1
  let total_ways := choose total_jellybeans total_pick
  have h1 : (two_blue_one_white : ℚ) = 12 := by sorry
  have h2 : (total_ways : ℚ) = 220 := by sorry
  calc
    (two_blue_one_white / total_ways : ℚ)
        = 12 / 220 : by rw [h1, h2]
    ... = 3 / 55 : by norm_num

end jellybean_probability_l52_52163


namespace intercept_form_conversion_normal_form_conversion_l52_52086

-- Definitions for given conditions.
def plane_eq (x y z : ℝ) : Prop :=
  2 * x - 2 * y + z - 20 = 0

def intercept_form (x y z : ℝ) : Prop :=
  (x / 10) + (y / -10) + (z / 20) = 1

def normal_form (x y z : ℝ) : Prop :=
  -(2 / 3) * x + (2 / 3) * y - (1 / 3) * z + (20 / 3) = 0

-- Theorem statements to prove the conversions.
theorem intercept_form_conversion (x y z : ℝ) :
  plane_eq x y z → intercept_form x y z :=
by
  sorry

theorem normal_form_conversion (x y z : ℝ) :
  plane_eq x y z → normal_form x y z :=
by
  sorry

end intercept_form_conversion_normal_form_conversion_l52_52086


namespace remainder_of_12_factorial_mod_13_l52_52077

open Nat

theorem remainder_of_12_factorial_mod_13 : (factorial 12) % 13 = 12 := by
  -- Wilson's Theorem: For a prime number \( p \), \( (p-1)! \equiv -1 \pmod{p} \)
  -- Given \( p = 13 \), we have \( 12! \equiv -1 \pmod{13} \)
  -- Thus, it follows that the remainder is 12
  sorry

end remainder_of_12_factorial_mod_13_l52_52077


namespace eggs_remainder_and_full_cartons_l52_52456

def abigail_eggs := 48
def beatrice_eggs := 63
def carson_eggs := 27
def carton_size := 15

theorem eggs_remainder_and_full_cartons :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  ∃ (full_cartons left_over : ℕ),
    total_eggs = full_cartons * carton_size + left_over ∧
    left_over = 3 ∧
    full_cartons = 9 :=
by
  sorry

end eggs_remainder_and_full_cartons_l52_52456


namespace prob_A_union_B_compl_l52_52354

open ProbabilityTheory

def outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 2, 3, 4}
def B_compl : Set ℕ := {5, 6}

def P (s : Set ℕ) : ℚ := s.card / outcomes.card

theorem prob_A_union_B_compl : P (A ∪ B_compl) = 2 / 3 :=
by
  have hA : P A = 1 / 3 := by simp [P, A, outcomes, Finset.card]
  have hB_compl : P B_compl = 1 / 3 := by simp [P, B_compl, outcomes, Finset.card]
  have h_disjoint : Disjoint A B_compl := by simp [Set.Disjoint, Set.Inter_eq_empty]
  calc
    P (A ∪ B_compl) = P A + P B_compl : eq.symm (probability_union_disjoint h_disjoint)
    ... = 1 / 3 + 1 / 3 : by rw [hA, hB_compl]
    ... = 2 / 3 : by norm_num

end prob_A_union_B_compl_l52_52354


namespace division_problem_l52_52290

theorem division_problem : 8900 / 6 / 4 = 1483.3333 :=
by sorry

end division_problem_l52_52290


namespace lorelei_roses_l52_52536

theorem lorelei_roses :
  let red_flowers := 12
  let pink_flowers := 18
  let yellow_flowers := 20
  let orange_flowers := 8
  let lorelei_red := (50 / 100) * red_flowers
  let lorelei_pink := (50 / 100) * pink_flowers
  let lorelei_yellow := (25 / 100) * yellow_flowers
  let lorelei_orange := (25 / 100) * orange_flowers
  lorelei_red + lorelei_pink + lorelei_yellow + lorelei_orange = 22 :=
by
  sorry

end lorelei_roses_l52_52536


namespace range_of_a_l52_52378

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

def neg_p : Prop := ∃ x : ℝ, a * x^2 + a * x + 1 < 0

theorem range_of_a (h : neg_p a) : a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
  sorry

end range_of_a_l52_52378


namespace correct_operation_l52_52423

variable (a b : ℝ)

theorem correct_operation : (a^2 * a^3 = a^5) :=
by sorry

end correct_operation_l52_52423


namespace miranda_can_stuff_10_pillows_l52_52699

def feathers_needed_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

theorem miranda_can_stuff_10_pillows :
  (goose_total_feathers / goose_feathers_per_pound + duck_total_feathers / duck_feathers_per_pound) / feathers_needed_per_pillow = 10 :=
by
  sorry

end miranda_can_stuff_10_pillows_l52_52699


namespace train_length_l52_52775

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l52_52775


namespace find_middle_part_length_l52_52763

theorem find_middle_part_length (a b c : ℝ) 
  (h1 : a + b + c = 28) 
  (h2 : (a - 0.5 * a) + b + 0.5 * c = 16) :
  b = 4 :=
by
  sorry

end find_middle_part_length_l52_52763


namespace mark_reading_time_l52_52991

-- Definitions based on conditions
def daily_reading_hours : ℕ := 3
def days_in_week : ℕ := 7
def weekly_increase : ℕ := 6

-- Proof statement
theorem mark_reading_time : daily_reading_hours * days_in_week + weekly_increase = 27 := by
  -- placeholder for the proof
  sorry

end mark_reading_time_l52_52991


namespace prob_calculation_l52_52270

noncomputable def normal_distribution (mu sigma : ℝ) : Measure ℝ :=
  MeasureTheory.Measure.dirac mu

open MeasureTheory

def prob_le (x : ℝ) (dist : Measure ℝ) (a : ℝ) : ℝ := dist {y | y ≤ a}
def prob_interval (dist : Measure ℝ) (a b : ℝ) : ℝ := dist {y | a < y ∧ y ≤ b}

axiom normal_property {x : ℝ} {σ : ℝ} :
  ∀ (mu : ℝ), (∀ (a : ℝ), x ∼ (fun y => Measure.dirac y) mu σ)  →  prob_le 2 (normal_distribution 3 σ) = 0.3

theorem prob_calculation (σ : ℝ) (h : prob_le 2 (normal_distribution 3 σ) = 0.3) :
  prob_interval (normal_distribution 3 σ) 3 4 = 0.2 :=
sorry

end prob_calculation_l52_52270


namespace two_digit_numbers_division_condition_l52_52152

theorem two_digit_numbers_division_condition {n x y q : ℕ} (h1 : 10 * x + y = n)
  (h2 : n % 6 = x)
  (h3 : n / 10 = 3) (h4 : n % 10 = y) :
  n = 33 ∨ n = 39 := 
sorry

end two_digit_numbers_division_condition_l52_52152


namespace inscribed_rectangle_area_l52_52644

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end inscribed_rectangle_area_l52_52644


namespace product_value_l52_52282

noncomputable def product_of_sequence : ℝ :=
  (1/3) * 9 * (1/27) * 81 * (1/243) * 729 * (1/2187) * 6561

theorem product_value : product_of_sequence = 729 := by
  sorry

end product_value_l52_52282


namespace fourth_term_is_neg6_term_is_150_at_16_positive_terms_after_7_l52_52713

-- Define the sequence {a_n} using the general term formula
def seq (n : ℕ) : ℤ := n^2 - 7 * n + 6

-- Prove that the fourth term of the sequence is -6
theorem fourth_term_is_neg6 : seq 4 = -6 := by
  sorry

-- Prove that there exists an n = 16 such that a_n = 150
theorem term_is_150_at_16 : ∃ n : ℕ, seq n = 150 := by
  use 16
  sorry

-- Prove that all terms of the sequence are positive for n ≥ 7
theorem positive_terms_after_7 (n : ℕ) : n ≥ 7 → seq n > 0 := by
  sorry

end fourth_term_is_neg6_term_is_150_at_16_positive_terms_after_7_l52_52713


namespace expression_equals_20_over_9_l52_52181

noncomputable def complex_fraction_expression := 
  let a := 11 + 1 / 9
  let b := 3 + 2 / 5
  let c := 1 + 2 / 17
  let d := 8 + 2 / 5
  let e := 3.6
  let f := 2 + 6 / 25
  ((a - b * c) - d / e) / f

theorem expression_equals_20_over_9 : complex_fraction_expression = 20 / 9 :=
by
  sorry

end expression_equals_20_over_9_l52_52181


namespace phone_charges_equal_l52_52624

theorem phone_charges_equal (x : ℝ) : 
  (0.60 + 14 * x = 0.08 * 18) → (x = 0.06) :=
by
  intro h
  have : 14 * x = 1.44 - 0.60 := sorry
  have : 14 * x = 0.84 := sorry
  have : x = 0.06 := sorry
  exact this

end phone_charges_equal_l52_52624


namespace sum_of_irreducible_fractions_is_integer_iff_same_denominator_l52_52262

theorem sum_of_irreducible_fractions_is_integer_iff_same_denominator
  (a b c d A : ℤ) (h_irred1 : Int.gcd a b = 1) (h_irred2 : Int.gcd c d = 1) (h_sum : (a : ℚ) / b + (c : ℚ) / d = A) :
  b = d := 
by
  sorry

end sum_of_irreducible_fractions_is_integer_iff_same_denominator_l52_52262


namespace intersection_of_A_and_B_l52_52942

open Set

-- Definition of set A
def A : Set ℤ := {1, 2, 3}

-- Definition of set B
def B : Set ℤ := {x | x < -1 ∨ 0 < x ∧ x < 2}

-- The theorem to prove A ∩ B = {1}
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  -- Proof logic here
  sorry

end intersection_of_A_and_B_l52_52942


namespace total_weight_of_envelopes_l52_52461

theorem total_weight_of_envelopes :
  (8.5 * 880 / 1000) = 7.48 :=
by
  sorry

end total_weight_of_envelopes_l52_52461


namespace intersection_subset_proper_l52_52365

-- Definitions of P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- The problem statement to prove
theorem intersection_subset_proper : P ∩ Q ⊂ P := by
  sorry

end intersection_subset_proper_l52_52365


namespace principal_sum_investment_l52_52766

theorem principal_sum_investment 
    (P R : ℝ) 
    (h1 : (P * 5 * (R + 2)) / 100 - (P * 5 * R) / 100 = 180)
    (h2 : (P * 5 * (R + 3)) / 100 - (P * 5 * R) / 100 = 270) :
    P = 1800 :=
by
  -- These are the hypotheses generated for Lean, the proof steps are omitted
  sorry

end principal_sum_investment_l52_52766


namespace compare_f_values_l52_52818

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem compare_f_values :
  f 0.6 > f (-0.5) ∧ f (-0.5) > f 0 := by
  sorry

end compare_f_values_l52_52818


namespace g_is_odd_l52_52687

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  sorry

end g_is_odd_l52_52687


namespace abe_age_sum_l52_52410

theorem abe_age_sum (h : abe_age = 29) : abe_age + (abe_age - 7) = 51 :=
by
  sorry

end abe_age_sum_l52_52410


namespace identify_mathematicians_l52_52778

def famous_people := List (Nat × String)

def is_mathematician : Nat → Bool
| 1 => false  -- Bill Gates
| 2 => true   -- Gauss
| 3 => false  -- Yuan Longping
| 4 => false  -- Nobel
| 5 => true   -- Chen Jingrun
| 6 => true   -- Hua Luogeng
| 7 => false  -- Gorky
| 8 => false  -- Einstein
| _ => false  -- default case

theorem identify_mathematicians (people : famous_people) : 
  (people.filter (fun (n, _) => is_mathematician n)) = [(2, "Gauss"), (5, "Chen Jingrun"), (6, "Hua Luogeng")] :=
by sorry

end identify_mathematicians_l52_52778


namespace distance_focus_to_asymptote_of_hyperbola_l52_52800

open Real

noncomputable def distance_from_focus_to_asymptote_of_hyperbola : ℝ :=
  let a := 2
  let b := 1
  let c := sqrt (a^2 + b^2)
  let foci1 := (sqrt (a^2 + b^2), 0)
  let foci2 := (-sqrt (a^2 + b^2), 0)
  let asymptote_slope := a / b
  let distance_formula := (|abs (sqrt 5)|) / (sqrt (1 + asymptote_slope^2))
  distance_formula

theorem distance_focus_to_asymptote_of_hyperbola :
  distance_from_focus_to_asymptote_of_hyperbola = 1 :=
sorry

end distance_focus_to_asymptote_of_hyperbola_l52_52800


namespace largest_prime_factor_of_1729_l52_52008

theorem largest_prime_factor_of_1729 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
begin
  have h1 : 1729 = 7 * 13 * 19, by norm_num,
  have h_prime_19 : prime 19, by norm_num,

  -- Introducing definitions and conditions
  use 19,
  split,
  { exact h_prime_19 },
  split,
  { 
    -- Proof that 19 divides 1729
    rw h1,
    exact dvd_mul_of_dvd_right (dvd_mul_right 19 13) 7,
  },
  {
    intros q q_conditions,
    obtain ⟨hq_prime, hq_divides⟩ := q_conditions,
    -- Prove that if q is another prime factor of 1729, q must be less than or equal to 19
    have factors_7_13_19 : {7, 13, 19} ⊆ {q | prime q ∧ q ∣ 1729}, 
    { 
      rw h1,
      simp only [set.mem_set_of_eq, and_imp],
      intros p hp_prime hp_dvd,
      exact ⟨hp_prime, hp_dvd⟩,
    },
    apply set.mem_of_subset factors_7_13_19,
    exact ⟨hq_prime, hq_divides⟩,
  }
end

end largest_prime_factor_of_1729_l52_52008


namespace buffy_whiskers_l52_52205

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l52_52205


namespace find_cost_expensive_module_l52_52975

-- Defining the conditions
def cost_cheaper_module : ℝ := 2.5
def total_modules : ℕ := 22
def num_cheaper_modules : ℕ := 21
def total_stock_value : ℝ := 62.5

-- The goal is to find the cost of the more expensive module 
def cost_expensive_module (cost_expensive_module : ℝ) : Prop :=
  num_cheaper_modules * cost_cheaper_module + cost_expensive_module = total_stock_value

-- The mathematically equivalent proof problem
theorem find_cost_expensive_module : cost_expensive_module 10 :=
by
  unfold cost_expensive_module
  norm_num
  sorry

end find_cost_expensive_module_l52_52975


namespace binomial_7_2_eq_21_l52_52477

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l52_52477


namespace Sam_age_l52_52569

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l52_52569


namespace simplify_and_sum_of_exponents_l52_52576

-- Define the given expression
def radicand (x y z : ℝ) : ℝ := 40 * x ^ 5 * y ^ 7 * z ^ 9

-- Define what cube root stands for
noncomputable def cbrt (a : ℝ) := a ^ (1 / 3 : ℝ)

-- Define the simplified expression outside the cube root
noncomputable def simplified_outside_exponents (x y z : ℝ) : ℝ := x * y * z ^ 3

-- Define the sum of the exponents outside the radical
def sum_of_exponents_outside (x y z : ℝ) : ℝ := (1 + 1 + 3 : ℝ)

-- Statement of the problem in Lean
theorem simplify_and_sum_of_exponents (x y z : ℝ) :
  sum_of_exponents_outside x y z = 5 :=
by 
  sorry

end simplify_and_sum_of_exponents_l52_52576


namespace problem_statement_l52_52159

theorem problem_statement (k : ℕ) (h : 35^k ∣ 1575320897) : 7^k - k^7 = 1 := by
  sorry

end problem_statement_l52_52159


namespace base_eight_to_ten_l52_52736

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end base_eight_to_ten_l52_52736


namespace inequality_holds_l52_52937

theorem inequality_holds (a b : ℝ) (h : a ≠ b) : a^4 + 6 * a^2 * b^2 + b^4 > 4 * a * b * (a^2 + b^2) := 
by
  sorry

end inequality_holds_l52_52937


namespace find_divisor_l52_52166

theorem find_divisor (n : ℕ) (h_n : n = 36) : 
  ∃ D : ℕ, ((n + 10) * 2 / D) - 2 = 44 → D = 2 :=
by
  use 2
  intros h
  sorry

end find_divisor_l52_52166


namespace find_square_tiles_l52_52165

variable {s p : ℕ}

theorem find_square_tiles (h1 : s + p = 30) (h2 : 4 * s + 5 * p = 110) : s = 20 :=
by
  sorry

end find_square_tiles_l52_52165


namespace triangle_inequality_l52_52636

-- Define the nondegenerate condition for the triangle's side lengths.
def nondegenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter condition for the triangle.
def triangle_perimeter (a b c : ℝ) (p : ℝ) : Prop :=
  a + b + c = p

-- The main theorem to prove the given inequality.
theorem triangle_inequality (a b c : ℝ) (h_non_deg : nondegenerate_triangle a b c) (h_perim : triangle_perimeter a b c 1) :
  abs ((a - b) / (c + a * b)) + abs ((b - c) / (a + b * c)) + abs ((c - a) / (b + a * c)) < 2 :=
by
  sorry

end triangle_inequality_l52_52636


namespace gino_initial_sticks_l52_52324

-- Definitions based on the conditions
def given_sticks : ℕ := 50
def remaining_sticks : ℕ := 13
def initial_sticks (x y : ℕ) : ℕ := x + y

-- Theorem statement based on the mathematically equivalent proof problem
theorem gino_initial_sticks :
  initial_sticks given_sticks remaining_sticks = 63 :=
by
  sorry

end gino_initial_sticks_l52_52324


namespace sam_age_l52_52574

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l52_52574


namespace cube_planes_l52_52978

theorem cube_planes {k : ℕ → ℝ} (h : ∀ n, k n = n) :
  ∃ (planes : ℕ → (ℝ × ℝ × ℝ × ℝ)), 
    (∀ n, planes n = (1, 2, 4, k n)) ∧ 
    (∀ n, ∃ d, d = |(k (n + 1) - k n) / real.sqrt (1^2 + 2^2 + 4^2)| ∧ 
      d = 1 / real.sqrt 21) :=
begin
  sorry,
end

end cube_planes_l52_52978


namespace negation_of_existence_l52_52401

theorem negation_of_existence :
  ¬ (∃ (x_0 : ℝ), x_0^2 - x_0 + 1 ≤ 0) ↔ ∀ (x : ℝ), x^2 - x + 1 > 0 :=
by
  sorry

end negation_of_existence_l52_52401


namespace rhombus_diagonal_length_l52_52873

theorem rhombus_diagonal_length (d1 : ℝ) : 
  (d1 * 12) / 2 = 60 → d1 = 10 := 
by 
  sorry

end rhombus_diagonal_length_l52_52873


namespace positive_difference_perimeters_l52_52189

def perimeter_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def perimeter_cross_shape : ℕ := 
  let top_and_bottom := 3 + 3 -- top and bottom edges
  let left_and_right := 3 + 3 -- left and right edges
  let internal_subtraction := 4
  top_and_bottom + left_and_right - internal_subtraction

theorem positive_difference_perimeters :
  let length := 4
  let width := 3
  perimeter_rectangle length width - perimeter_cross_shape = 6 :=
by
  let length := 4
  let width := 3
  sorry

end positive_difference_perimeters_l52_52189


namespace score_order_l52_52531

variable (A B C D : ℕ)

-- Condition 1: B + D = A + C
axiom h1 : B + D = A + C
-- Condition 2: A + B > C + D + 10
axiom h2 : A + B > C + D + 10
-- Condition 3: D > B + C + 20
axiom h3 : D > B + C + 20
-- Condition 4: A + B + C + D = 200
axiom h4 : A + B + C + D = 200

-- Question to prove: Order is Donna > Alice > Brian > Cindy
theorem score_order : D > A ∧ A > B ∧ B > C :=
by
  sorry

end score_order_l52_52531


namespace number_of_ways_to_choose_six_with_consecutives_l52_52670

theorem number_of_ways_to_choose_six_with_consecutives :
  ∀ (s : Finset ℕ), (s.card = 6) → (∀ i ∈ s, i ∈ (Finset.range 50 \ {0})) →
  ∃ (t u : Finset ℕ), t ∪ u = s ∧ (∀ i j ∈ t, abs (i - j) = 1) ∧ (t.card ≥ 2) ∧
  (nat.choose 49 6 - nat.choose 44 6) :=
by
  sorry

end number_of_ways_to_choose_six_with_consecutives_l52_52670


namespace exit_forest_strategy_l52_52552

/-- A strategy ensuring the parachutist will exit the forest with a path length of less than 2.5l -/
theorem exit_forest_strategy (l : Real) : 
  ∃ (path_length : Real), path_length < 2.5 * l :=
by
  use 2.278 * l
  sorry

end exit_forest_strategy_l52_52552


namespace percentage_of_students_play_sports_l52_52729

def total_students : ℕ := 400
def soccer_percentage : ℝ := 0.125
def soccer_players : ℕ := 26

theorem percentage_of_students_play_sports : 
  ∃ P : ℝ, (soccer_percentage * P = soccer_players) → (P / total_students * 100 = 52) :=
by
  sorry

end percentage_of_students_play_sports_l52_52729


namespace sugar_flour_ratio_10_l52_52296

noncomputable def sugar_to_flour_ratio (sugar flour : ℕ) : ℕ :=
  sugar / flour

theorem sugar_flour_ratio_10 (sugar flour : ℕ) (hs : sugar = 50) (hf : flour = 5) : sugar_to_flour_ratio sugar flour = 10 :=
by
  rw [hs, hf]
  unfold sugar_to_flour_ratio
  norm_num
  -- sorry

end sugar_flour_ratio_10_l52_52296


namespace maximize_area_of_quadrilateral_l52_52639

theorem maximize_area_of_quadrilateral (k : ℝ) (h0 : 0 < k) (h1 : k < 1) 
    (hE : ∀ E : ℝ, E = 2 * k) (hF : ∀ F : ℝ, F = 2 * k) :
    k = 1/2 ∧ (2 * (1 - k) ^ 2) = 1/2 := 
by 
  sorry

end maximize_area_of_quadrilateral_l52_52639


namespace mike_weekly_avg_time_l52_52994

theorem mike_weekly_avg_time :
  let mon_wed_fri_tv := 4 -- hours per day on Mon, Wed, Fri
  let tue_thu_tv := 3 -- hours per day on Tue, Thu
  let weekend_tv := 5 -- hours per day on weekends
  let num_mon_wed_fri := 3 -- days
  let num_tue_thu := 2 -- days
  let num_weekend := 2 -- days
  let num_days_week := 7 -- days
  let num_video_game_days := 3 -- days
  let weeks := 4 -- weeks
  let mon_wed_fri_total := mon_wed_fri_tv * num_mon_wed_fri
  let tue_thu_total := tue_thu_tv * num_tue_thu
  let weekend_total := weekend_tv * num_weekend
  let weekly_tv_time := mon_wed_fri_total + tue_thu_total + weekend_total
  let daily_avg_tv_time := weekly_tv_time / num_days_week
  let daily_video_game_time := daily_avg_tv_time / 2
  let weekly_video_game_time := daily_video_game_time * num_video_game_days
  let total_tv_time_4_weeks := weekly_tv_time * weeks
  let total_video_game_time_4_weeks := weekly_video_game_time * weeks
  let total_time_4_weeks := total_tv_time_4_weeks + total_video_game_time_4_weeks
  let weekly_avg_time := total_time_4_weeks / weeks
  weekly_avg_time = 34 := sorry

end mike_weekly_avg_time_l52_52994


namespace sum_first_5_arithmetic_l52_52613

theorem sum_first_5_arithmetic (u : ℕ → ℝ) (h : u 3 = 0) : 
  (u 1 + u 2 + u 3 + u 4 + u 5) = 0 :=
sorry

end sum_first_5_arithmetic_l52_52613


namespace midpoint_one_sixth_one_twelfth_l52_52195

theorem midpoint_one_sixth_one_twelfth : (1 : ℚ) / 8 = (1 / 6 + 1 / 12) / 2 := by
  sorry

end midpoint_one_sixth_one_twelfth_l52_52195


namespace fraction_sum_l52_52180

theorem fraction_sum : (1 / 4 : ℚ) + (3 / 8) = 5 / 8 :=
by
  sorry

end fraction_sum_l52_52180


namespace find_P_l52_52691

noncomputable def P (x : ℝ) : ℝ :=
  4 * x^3 - 6 * x^2 - 12 * x

theorem find_P (a b c : ℝ) (h_root : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_roots : ∀ x, x^3 - 2 * x^2 - 4 * x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c)
  (h_Pa : P a = b + 2 * c)
  (h_Pb : P b = 2 * a + c)
  (h_Pc : P c = a + 2 * b)
  (h_Psum : P (a + b + c) = -20) :
  ∀ x, P x = 4 * x^3 - 6 * x^2 - 12 * x :=
by
  sorry

end find_P_l52_52691


namespace range_of_m_l52_52377

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : y1 = x1^2 - 4*x1 + 3)
  (h2 : y2 = x2^2 - 4*x2 + 3) (h3 : -1 < x1) (h4 : x1 < 1)
  (h5 : m > 0) (h6 : m-1 < x2) (h7 : x2 < m) (h8 : y1 ≠ y2) :
  (2 ≤ m ∧ m ≤ 3) ∨ (m ≥ 6) :=
sorry

end range_of_m_l52_52377


namespace tom_strokes_over_par_l52_52894

theorem tom_strokes_over_par (holes_played : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) :
  holes_played = 9 → avg_strokes_per_hole = 4 → par_per_hole = 3 → 
  (holes_played * avg_strokes_per_hole - holes_played * par_per_hole) = 9 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    9 * 4 - 9 * 3 = 36 - 27 : by simp
               ... = 9       : by norm_num

end tom_strokes_over_par_l52_52894


namespace max_2b_div_a_l52_52344

theorem max_2b_div_a (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : 
  ∃ max_val, max_val = (2 * b) / a ∧ max_val = (32 / 3) :=
by
  sorry

end max_2b_div_a_l52_52344


namespace prove_value_range_for_a_l52_52819

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a*x + 7 + a) / (x + 1)

noncomputable def g (x : ℝ) : ℝ := 
  - ((x + 1) + (8 / (x + 1))) + 6

theorem prove_value_range_for_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 4) ↔ (a ≥ 1 / 3) :=
sorry

end prove_value_range_for_a_l52_52819


namespace count_special_four_digit_integers_is_100_l52_52951

def count_special_four_digit_integers : Nat := sorry

theorem count_special_four_digit_integers_is_100 :
  count_special_four_digit_integers = 100 :=
sorry

end count_special_four_digit_integers_is_100_l52_52951


namespace range_of_sum_l52_52214

variable {x y t : ℝ}

theorem range_of_sum :
  (1 = x^2 + 4*y^2 - 2*x*y) ∧ (x < 0) ∧ (y < 0) →
  -2 <= x + 2*y ∧ x + 2*y < 0 :=
by {
  sorry
}

end range_of_sum_l52_52214


namespace sports_probability_boy_given_sports_probability_l52_52097

variable (x : ℝ) -- Number of girls

def number_of_boys := 1.5 * x
def boys_liking_sports := 0.4 * number_of_boys x
def girls_liking_sports := 0.2 * x
def total_students := x + number_of_boys x
def total_students_liking_sports := boys_liking_sports x + girls_liking_sports x

theorem sports_probability : (total_students_liking_sports x) / (total_students x) = 8 / 25 := 
sorry

theorem boy_given_sports_probability :
  (boys_liking_sports x) / (total_students_liking_sports x) = 3 / 4 := 
sorry

end sports_probability_boy_given_sports_probability_l52_52097


namespace solve_for_x_l52_52271

theorem solve_for_x (x : ℝ) (h : 2 * x + 10 = (1 / 2) * (5 * x + 30)) : x = -10 :=
sorry

end solve_for_x_l52_52271


namespace alpha_proportional_l52_52872

theorem alpha_proportional (alpha beta gamma : ℝ) (h1 : ∀ β γ, (β = 15 ∧ γ = 3) → α = 5)
    (h2 : beta = 30) (h3 : gamma = 6) : alpha = 2.5 :=
sorry

end alpha_proportional_l52_52872


namespace zach_fill_time_l52_52692

theorem zach_fill_time : 
  ∀ (t : ℕ), 
  (∀ (max_time max_rate zach_rate popped total : ℕ), 
    max_time = 30 → 
    max_rate = 2 → 
    zach_rate = 3 → 
    popped = 10 → 
    total = 170 → 
    (max_time * max_rate + t * zach_rate - popped = total) → 
    t = 40) := 
sorry

end zach_fill_time_l52_52692


namespace maximum_black_squares_l52_52553

theorem maximum_black_squares (n : ℕ) (h : n ≥ 2) : 
  (n % 2 = 0 → ∃ b : ℕ, b = (n^2 - 4) / 2) ∧ 
  (n % 2 = 1 → ∃ b : ℕ, b = (n^2 - 1) / 2) := 
by sorry

end maximum_black_squares_l52_52553


namespace possible_c_value_l52_52336

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem possible_c_value (a b c : ℝ) 
  (h1 : f (-1) a b c = f (-2) a b c) 
  (h2 : f (-2) a b c = f (-3) a b c) 
  (h3 : 0 ≤ f (-1) a b c) 
  (h4 : f (-1) a b c ≤ 3) : 
  6 ≤ c ∧ c ≤ 9 := sorry

end possible_c_value_l52_52336


namespace binomial_7_2_l52_52480

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l52_52480


namespace matrix_determinant_zero_l52_52304

theorem matrix_determinant_zero (a b : ℝ) : 
  Matrix.det ![
    ![1, Real.sin (2 * a), Real.sin a],
    ![Real.sin (2 * a), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 0 := 
by 
  sorry

end matrix_determinant_zero_l52_52304


namespace man_work_alone_l52_52633

theorem man_work_alone (W: ℝ) (M S: ℝ)
  (hS: S = W / 6.67)
  (hMS: M + S = W / 4):
  W / M = 10 :=
by {
  -- This is a placeholder for the proof
  sorry
}

end man_work_alone_l52_52633


namespace Cantor_set_compact_perfect_nowhere_dense_l52_52120

open Set Filter TopologicalSpace

def Cantor_set := ⋂ (n : ℕ), ⋃ (s : Finset (Fin n)), (⋂ i ∈ s, Icc ((i + 1 : ℝ)/(3 * (n + 1))) ((i + 2 : ℝ)/(3 * (n + 1))))

theorem Cantor_set_compact_perfect_nowhere_dense :
  isCompact Cantor_set ∧
  (∀ x ∈ Cantor_set, ∃ y ≠ x, y ∈ Cantor_set) ∧
  ∀ U, is_open U -> ¬(U ⊆ Cantor_set) :=
by
  sorry

end Cantor_set_compact_perfect_nowhere_dense_l52_52120


namespace max_sum_of_factors_of_48_l52_52325

theorem max_sum_of_factors_of_48 (d Δ : ℕ) (h : d * Δ = 48) : d + Δ ≤ 49 :=
sorry

end max_sum_of_factors_of_48_l52_52325


namespace farm_field_area_l52_52091

theorem farm_field_area
  (planned_daily_plough : ℕ)
  (actual_daily_plough : ℕ)
  (extra_days : ℕ)
  (remaining_area : ℕ)
  (total_days_hectares : ℕ → ℕ) :
  planned_daily_plough = 260 →
  actual_daily_plough = 85 →
  extra_days = 2 →
  remaining_area = 40 →
  total_days_hectares (total_days_hectares (1 + 2) * 85 + 40) = 312 :=
by
  sorry

end farm_field_area_l52_52091


namespace largest_prime_factor_of_1729_is_19_l52_52000

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l52_52000


namespace jack_walked_distance_l52_52103

theorem jack_walked_distance (time_in_hours : ℝ) (rate : ℝ) (expected_distance : ℝ) : 
  time_in_hours = 1 + 15 / 60 ∧ 
  rate = 6.4 →
  expected_distance = 8 → 
  rate * time_in_hours = expected_distance :=
by 
  intros h
  sorry

end jack_walked_distance_l52_52103


namespace Mary_younger_than_Albert_l52_52777

-- Define the basic entities and conditions
def Betty_age : ℕ := 11
def Albert_age : ℕ := 4 * Betty_age
def Mary_age : ℕ := Albert_age / 2

-- Define the property to prove
theorem Mary_younger_than_Albert : Albert_age - Mary_age = 22 :=
by 
  sorry

end Mary_younger_than_Albert_l52_52777


namespace square_side_length_l52_52160

theorem square_side_length (s : ℝ) (h : s^2 + s - 4 * s = 4) : s = 4 :=
sorry

end square_side_length_l52_52160


namespace quadratic_no_solution_l52_52402

theorem quadratic_no_solution 
  (p q r s : ℝ) (h1 : p^2 < 4 * q) (h2 : r^2 < 4 * s) :
  (1009 * p + 1008 * r)^2 < 4 * 2017 * (1009 * q + 1008 * s) :=
by
  sorry

end quadratic_no_solution_l52_52402


namespace pencil_probability_l52_52352

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem pencil_probability : 
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 14 :=
by
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := 6
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination non_defective_pencils selected_pencils
  let probability := non_defective_ways / total_ways
  have h : probability = 5 / 14 := sorry
  exact h

end pencil_probability_l52_52352


namespace cannot_determine_total_movies_l52_52272

def number_of_books : ℕ := 22
def books_read : ℕ := 12
def books_to_read : ℕ := 10
def movies_watched : ℕ := 56

theorem cannot_determine_total_movies (n : ℕ) (h1 : books_read + books_to_read = number_of_books) : n ≠ movies_watched → n = 56 → False := 
by 
  intro h2 h3
  sorry

end cannot_determine_total_movies_l52_52272


namespace value_of_a_l52_52908

variable (a : ℝ)

theorem value_of_a (h1 : (0.5 / 100) * a = 0.80) : a = 160 := by
  sorry

end value_of_a_l52_52908


namespace sequence_term_value_l52_52940

theorem sequence_term_value :
  ∃ (a : ℕ → ℚ), a 1 = 2 ∧ (∀ n, a (n + 1) = a n + 1 / 2) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_value_l52_52940


namespace digit_for_divisibility_by_5_l52_52419

theorem digit_for_divisibility_by_5 (B : ℕ) (B_digit_condition : B < 10) :
  (∃ k : ℕ, 6470 + B = 5 * k) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_for_divisibility_by_5_l52_52419


namespace union_A_B_l52_52820

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x > 2}

theorem union_A_B :
  A ∪ B = {x : ℝ | 1 ≤ x} := sorry

end union_A_B_l52_52820


namespace find_x_l52_52984

variables (a b c k : ℝ) (h : k ≠ 0)

theorem find_x (x y z : ℝ)
  (h1 : (xy + k) / (x + y) = a)
  (h2 : (xz + k) / (x + z) = b)
  (h3 : (yz + k) / (y + z) = c) :
  x = 2 * a * b * c * d / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) := sorry

end find_x_l52_52984


namespace cos_double_angle_l52_52944

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 1 / 3) :
  Real.cos (2 * α) = 7 / 9 :=
sorry

end cos_double_angle_l52_52944


namespace prove_k_eq_one_l52_52550

theorem prove_k_eq_one 
  (n m k : ℕ) 
  (h_positive : 0 < n)  -- implies n, and hence n-1, n+1, are all positive
  (h_eq : (n-1) * n * (n+1) = m^k): 
  k = 1 := 
sorry

end prove_k_eq_one_l52_52550


namespace albert_mary_age_ratio_l52_52458

theorem albert_mary_age_ratio
  (A M B : ℕ)
  (h1 : A = 4 * B)
  (h2 : M = A - 14)
  (h3 : B = 7)
  :
  A / M = 2 := 
by sorry

end albert_mary_age_ratio_l52_52458


namespace probability_one_from_each_l52_52459

theorem probability_one_from_each (cards : Finset (Fin 9)) 
  (alex tommy : Finset (Fin 9)) 
  (h_card_count : cards.card = 9)
  (h_alex : alex.card = 4)
  (h_tommy : tommy.card = 5)
  (h_alex_union_tommy : alex ∪ tommy = cards)
  (h_disjoint : Disjoint alex tommy) :
  probability (λ (x : Fin 9 × Fin 9), x.1 ∈ alex ∧ x.2 ∈ tommy ∨ x.1 ∈ tommy ∧ x.2 ∈ alex) 
    (Finset.powersetLen 2 cards).toFinset = 5 / 9 :=
by
  sorry

end probability_one_from_each_l52_52459


namespace find_second_quadrant_point_l52_52779

def is_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem find_second_quadrant_point :
  (is_second_quadrant (2, 3) = false) ∧
  (is_second_quadrant (2, -3) = false) ∧
  (is_second_quadrant (-2, -3) = false) ∧
  (is_second_quadrant (-2, 3) = true) := 
sorry

end find_second_quadrant_point_l52_52779


namespace denise_crayons_l52_52191

theorem denise_crayons (c : ℕ) :
  (∀ f p : ℕ, f = 30 ∧ p = 7 → c = f * p) → c = 210 :=
by
  intro h
  specialize h 30 7 ⟨rfl, rfl⟩
  exact h

end denise_crayons_l52_52191


namespace Jakes_height_is_20_l52_52119

-- Define the conditions
def Sara_width : ℤ := 12
def Sara_height : ℤ := 24
def Sara_depth : ℤ := 24
def Jake_width : ℤ := 16
def Jake_depth : ℤ := 18
def volume_difference : ℤ := 1152

-- Volume calculation
def Sara_volume : ℤ := Sara_width * Sara_height * Sara_depth

-- Prove Jake's height is 20 inches
theorem Jakes_height_is_20 :
  ∃ h : ℤ, (Sara_volume - (Jake_width * h * Jake_depth) = volume_difference) ∧ h = 20 :=
by
  sorry

end Jakes_height_is_20_l52_52119


namespace abs_eq_4_l52_52257

theorem abs_eq_4 (a : ℝ) : |a| = 4 ↔ a = 4 ∨ a = -4 :=
by
  sorry

end abs_eq_4_l52_52257


namespace sunzi_classic_equation_l52_52235

theorem sunzi_classic_equation (x : ℕ) : 3 * (x - 2) = 2 * x + 9 :=
  sorry

end sunzi_classic_equation_l52_52235


namespace squares_of_natural_numbers_l52_52867

theorem squares_of_natural_numbers (x y z : ℕ) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) : ∃ a b c : ℕ, x = a^2 ∧ y = b^2 ∧ z = c^2 := 
by
  sorry

end squares_of_natural_numbers_l52_52867


namespace solution1_solution2_l52_52817

noncomputable def problem1 : Prop :=
  ∃ (a b : ℤ), 
  (∃ (n : ℤ), 3*a - 14 = n ∧ a - 2 = n) ∧ 
  (b - 15 = -27) ∧ 
  a = 4 ∧ 
  b = -12 ∧ 
  (4*a + b = 4)

noncomputable def problem2 : Prop :=
  ∀ (a b : ℤ), 
  (a = 4) ∧ 
  (b = -12) → 
  (4*a + b = 4) → 
  (∃ n, n^2 = 4 ∧ (n = 2 ∨ n = -2))

theorem solution1 : problem1 := by { sorry }
theorem solution2 : problem2 := by { sorry }

end solution1_solution2_l52_52817


namespace math_preference_related_to_gender_l52_52935

-- Definitions for conditions
def total_students : ℕ := 100
def male_students : ℕ := 55
def female_students : ℕ := total_students - male_students -- 45
def likes_math : ℕ := 40
def female_likes_math : ℕ := 20
def female_not_like_math : ℕ := female_students - female_likes_math -- 25
def male_likes_math : ℕ := likes_math - female_likes_math -- 20
def male_not_like_math : ℕ := male_students - male_likes_math -- 35

-- Calculate Chi-square
def chi_square (a b c d : ℕ) : Float :=
  let numerator := (total_students * (a * d - b * c)^2).toFloat
  let denominator := ((a + b) * (c + d) * (a + c) * (b + d)).toFloat
  numerator / denominator

def k_square : Float := chi_square 20 35 20 25 -- Calculate with given values

-- Prove the result
theorem math_preference_related_to_gender :
  k_square > 7.879 :=
by
  sorry

end math_preference_related_to_gender_l52_52935


namespace factorize_x2_minus_2x_plus_1_l52_52065

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end factorize_x2_minus_2x_plus_1_l52_52065


namespace find_m_of_odd_number_sequence_l52_52501

theorem find_m_of_odd_number_sequence : 
  ∃ m : ℕ, m > 1 ∧ (∃ a : ℕ, a = m * (m - 1) + 1 ∧ a = 2023) ↔ m = 45 :=
by
    sorry

end find_m_of_odd_number_sequence_l52_52501


namespace base9_add_subtract_l52_52187

theorem base9_add_subtract :
  let n1 := 3 * 9^2 + 5 * 9 + 1
  let n2 := 4 * 9^2 + 6 * 9 + 5
  let n3 := 1 * 9^2 + 3 * 9 + 2
  let n4 := 1 * 9^2 + 4 * 9 + 7
  (n1 + n2 + n3 - n4 = 8 * 9^2 + 4 * 9 + 7) :=
by
  sorry

end base9_add_subtract_l52_52187


namespace largest_prime_factor_of_1729_l52_52005

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l52_52005


namespace largest_prime_factor_1729_l52_52009

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l52_52009


namespace probability_not_black_correct_l52_52164

def total_balls : ℕ := 8 + 9 + 3

def non_black_balls : ℕ := 8 + 3

def probability_not_black : ℚ := non_black_balls / total_balls

theorem probability_not_black_correct : probability_not_black = 11 / 20 := 
by 
  -- The proof would go here
  sorry

end probability_not_black_correct_l52_52164


namespace g_cross_horizontal_asymptote_at_l52_52202

noncomputable def g (x : ℝ) : ℝ :=
  (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

theorem g_cross_horizontal_asymptote_at (x : ℝ) : g x = 3 ↔ x = 13 / 4 :=
by
  sorry

end g_cross_horizontal_asymptote_at_l52_52202


namespace intersection_M_N_l52_52554

noncomputable def M := {x : ℕ | x < 6}
noncomputable def N := {x : ℕ | x^2 - 11 * x + 18 < 0}
noncomputable def intersection := {x : ℕ | x ∈ M ∧ x ∈ N}

theorem intersection_M_N : intersection = {3, 4, 5} := by
  sorry

end intersection_M_N_l52_52554


namespace probability_black_pen_l52_52351

-- Define the total number of pens and the specific counts
def total_pens : ℕ := 5 + 6 + 7
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the probability calculation
def probability (total : ℕ) (count : ℕ) : ℚ := count / total

-- State the theorem
theorem probability_black_pen :
  probability total_pens black_pens = 1 / 3 :=
by sorry

end probability_black_pen_l52_52351


namespace mike_books_l52_52600

theorem mike_books (tim_books : ℕ) (total_books : ℕ) (h1 : tim_books = 22) (h2 : total_books = 42) :
  ∃ (mike_books : ℕ), mike_books = total_books - tim_books :=
by {
  use 20,
  rw [h1, h2],
  norm_num,
  sorry
}

end mike_books_l52_52600


namespace find_n_equal_roots_l52_52053

theorem find_n_equal_roots (x n : ℝ) (hx : x ≠ 2) : n = -1 ↔
  let a := 1
  let b := -2
  let c := -(n^2 + 2 * n)
  b^2 - 4 * a * c = 0 :=
by
  sorry

end find_n_equal_roots_l52_52053


namespace total_vehicles_l52_52452

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end total_vehicles_l52_52452


namespace largest_prime_factor_of_1729_l52_52003

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  nat.find_greatest_prime n

theorem largest_prime_factor_of_1729 : largest_prime_factor 1729 = 19 :=
sorry

end largest_prime_factor_of_1729_l52_52003


namespace steven_owes_jeremy_l52_52979

-- Define the payment per room
def payment_per_room : ℚ := 13 / 3

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 5 / 2

-- Calculate the total amount owed
def total_amount_owed : ℚ := payment_per_room * rooms_cleaned

-- The theorem statement to prove
theorem steven_owes_jeremy :
  total_amount_owed = 65 / 6 :=
by
  sorry

end steven_owes_jeremy_l52_52979


namespace lisa_flight_distance_l52_52372

-- Define the given speed and time
def speed : ℝ := 32
def time : ℝ := 8

-- Define the distance formula
def distance (v : ℝ) (t : ℝ) : ℝ := v * t

-- State the theorem to be proved
theorem lisa_flight_distance : distance speed time = 256 := by
sorry

end lisa_flight_distance_l52_52372


namespace circle_condition_m_l52_52231

theorem circle_condition_m (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_m_l52_52231


namespace smallest_solution_of_equation_l52_52491

theorem smallest_solution_of_equation : 
    ∃ x : ℝ, x*|x| = 3 * x - 2 ∧ 
            ∀ y : ℝ, y*|y| = 3 * y - 2 → x ≤ y :=
sorry

end smallest_solution_of_equation_l52_52491


namespace ratio_father_to_children_after_5_years_l52_52796

def father's_age := 15
def sum_children_ages := father's_age / 3

def father's_age_after_5_years := father's_age + 5
def sum_children_ages_after_5_years := sum_children_ages + 10

theorem ratio_father_to_children_after_5_years :
  father's_age_after_5_years / sum_children_ages_after_5_years = 4 / 3 := by
  sorry

end ratio_father_to_children_after_5_years_l52_52796


namespace product_of_three_numbers_l52_52889

theorem product_of_three_numbers (a b c : ℝ) 
  (h₁ : a + b + c = 45)
  (h₂ : a = 2 * (b + c))
  (h₃ : c = 4 * b) : 
  a * b * c = 1080 := 
sorry

end product_of_three_numbers_l52_52889


namespace exists_strictly_positive_c_l52_52911

theorem exists_strictly_positive_c {a : ℕ → ℕ → ℝ} (h_diag_pos : ∀ i, a i i > 0)
  (h_off_diag_neg : ∀ i j, i ≠ j → a i j < 0) :
  ∃ (c : ℕ → ℝ), (∀ i, 
    0 < c i) ∧ 
    ((∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 > 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 < 0) ∨ 
     (∀ k, a k 1 * c 1 + a k 2 * c 2 + a k 3 * c 3 = 0)) :=
by
  sorry

end exists_strictly_positive_c_l52_52911


namespace min_score_guarantees_payoff_l52_52614

-- Defining the probability of a single roll being a six
def prob_single_six : ℚ := 1 / 6 

-- Defining the event of rolling two sixes independently
def prob_two_sixes : ℚ := prob_single_six * prob_single_six

-- Defining the score of two die rolls summing up to 12
def is_score_twelve (a b : ℕ) : Prop := a + b = 12

-- Proving the probability of Jim scoring 12 in two rolls guarantees some monetary payoff.
theorem min_score_guarantees_payoff :
  (prob_two_sixes = 1/36) :=
by
  sorry

end min_score_guarantees_payoff_l52_52614


namespace fill_cistern_time_l52_52167

-- Definitions based on conditions
def rate_A : ℚ := 1 / 8
def rate_B : ℚ := 1 / 16
def rate_C : ℚ := -1 / 12

-- Combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Time to fill the cistern
def time_to_fill := 1 / combined_rate

-- Lean statement of the proof
theorem fill_cistern_time : time_to_fill = 9.6 := by
  sorry

end fill_cistern_time_l52_52167


namespace sum_of_factors_of_30_multiplied_by_2_equals_144_l52_52281

-- We define the factors of 30
def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- We define the function to multiply each factor by 2 and sum them
def sum_factors_multiplied_by_2 (factors : List ℕ) : ℕ :=
  factors.foldl (λ acc x => acc + 2 * x) 0

-- The final statement to be proven
theorem sum_of_factors_of_30_multiplied_by_2_equals_144 :
  sum_factors_multiplied_by_2 factors_of_30 = 144 :=
by sorry

end sum_of_factors_of_30_multiplied_by_2_equals_144_l52_52281


namespace tom_strokes_over_par_l52_52893

theorem tom_strokes_over_par (holes_played : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) :
  holes_played = 9 → avg_strokes_per_hole = 4 → par_per_hole = 3 → 
  (holes_played * avg_strokes_per_hole - holes_played * par_per_hole) = 9 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    9 * 4 - 9 * 3 = 36 - 27 : by simp
               ... = 9       : by norm_num

end tom_strokes_over_par_l52_52893


namespace apples_per_box_l52_52598

-- Defining the given conditions
variable (apples_per_crate : ℤ)
variable (number_of_crates : ℤ)
variable (rotten_apples : ℤ)
variable (number_of_boxes : ℤ)

-- Stating the facts based on given conditions
def total_apples := apples_per_crate * number_of_crates
def remaining_apples := total_apples - rotten_apples

-- The statement to prove
theorem apples_per_box 
    (hc1 : apples_per_crate = 180)
    (hc2 : number_of_crates = 12)
    (hc3 : rotten_apples = 160)
    (hc4 : number_of_boxes = 100) :
    (remaining_apples apples_per_crate number_of_crates rotten_apples) / number_of_boxes = 20 := 
sorry

end apples_per_box_l52_52598


namespace tom_strokes_over_par_l52_52896

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l52_52896


namespace base_eight_to_base_ten_l52_52739

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end base_eight_to_base_ten_l52_52739


namespace abs_iff_neg_one_lt_x_lt_one_l52_52912

theorem abs_iff_neg_one_lt_x_lt_one (x : ℝ) : |x| < 1 ↔ -1 < x ∧ x < 1 :=
by
  sorry

end abs_iff_neg_one_lt_x_lt_one_l52_52912


namespace distance_from_point_to_line_is_7_l52_52317

open Real

-- Definition of the point and the line.
def point : Vector ℝ 3 := ⟨[1, 2, 3], by simp⟩
def line (t : ℝ) : Vector ℝ 3 := ⟨[6 + 3 * t, 7 + 2 * t, 7 - 2 * t], by simp [add_assoc, add_mul, zero_add]⟩

-- The distance computation
def distance (p1 p2 : Vector ℝ 3) :=
  (p1 - p2).norm

-- The goal is to prove that the distance from the point to the line at the closest point is 7.
theorem distance_from_point_to_line_is_7 : ∃ t : ℝ, distance point (line t) = 7 :=
sorry

end distance_from_point_to_line_is_7_l52_52317


namespace value_of_a_plus_d_l52_52023

variable {R : Type} [LinearOrderedField R]
variables {a b c d : R}

theorem value_of_a_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 13 := by
  sorry

end value_of_a_plus_d_l52_52023


namespace no_solution_inequality_l52_52287

theorem no_solution_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| < 4 * x - 1 ∧ x < a) ↔ a ≤ (2/3) := by sorry

end no_solution_inequality_l52_52287


namespace percentage_of_singles_l52_52530

/-- In a baseball season, Lisa had 50 hits. Among her hits were 2 home runs, 
2 triples, 8 doubles, and 1 quadruple. The rest of her hits were singles. 
What percent of her hits were singles? --/
theorem percentage_of_singles
  (total_hits : ℕ := 50)
  (home_runs : ℕ := 2)
  (triples : ℕ := 2)
  (doubles : ℕ := 8)
  (quadruples : ℕ := 1)
  (non_singles := home_runs + triples + doubles + quadruples)
  (singles := total_hits - non_singles) :
  (singles : ℚ) / (total_hits : ℚ) * 100 = 74 := by
  sorry

end percentage_of_singles_l52_52530


namespace cupcakes_frosted_in_5_minutes_l52_52045

theorem cupcakes_frosted_in_5_minutes :
  (let r_cagney := (1 : ℚ) / 20;
       r_lacey := (1 : ℚ) / 30;
       combined_rate := r_cagney + r_lacey in 
       300 * combined_rate = 25) := 
by {
  -- Define Cagney's and Lacey's rates
  let r_cagney := (1 : ℚ) / 20,
  let r_lacey := (1 : ℚ) / 30,

  -- Calculate combined rate
  let combined_rate := r_cagney + r_lacey,

  -- Express the total number of cupcakes frosted in 300 seconds
  have h : 300 * combined_rate = 25, by {
    calc 300 * combined_rate
          = 300 * ((1 / 20) + (1 / 30)) : by { refl }
      ... = 300 * ((3 / 60) + (2 / 60)) : by { congr; field_simp [ne_of_gt (show 20 > 0, by norm_num)] }
      ... = 300 * (5 / 60) : by { congr; field_simp [ne_of_gt (show 30 > 0, by norm_num)] }
      ... = 300 * (1 / 12) : by { norm_num }
      ... = 25 : by norm_num,
  },
  exact h,
}

end cupcakes_frosted_in_5_minutes_l52_52045


namespace eyes_per_ant_proof_l52_52115

noncomputable def eyes_per_ant (s a e_s E : ℕ) : ℕ :=
  let e_spiders := s * e_s
  let e_ants := E - e_spiders
  e_ants / a

theorem eyes_per_ant_proof : eyes_per_ant 3 50 8 124 = 2 :=
by
  sorry

end eyes_per_ant_proof_l52_52115


namespace fractional_sum_identity_l52_52988

noncomputable def distinct_real_roots (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem fractional_sum_identity :
  ∀ (p q r A B C : ℝ),
  (x^3 - 22*x^2 + 80*x - 67 = (x - p) * (x - q) * (x - r)) →
  distinct_real_roots (λ x => x^3 - 22*x^2 + 80*x - 67) p q r →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (1 / (A) + 1 / (B) + 1 / (C) = 244) :=
by 
  intros p q r A B C h_poly h_distinct h_fractional
  sorry

end fractional_sum_identity_l52_52988


namespace class_5_matches_l52_52836

theorem class_5_matches (matches_c1 matches_c2 matches_c3 matches_c4 matches_c5 : ℕ)
  (C1 : matches_c1 = 2)
  (C2 : matches_c2 = 4)
  (C3 : matches_c3 = 4)
  (C4 : matches_c4 = 3) :
  matches_c5 = 3 :=
sorry

end class_5_matches_l52_52836


namespace max_f_value_l52_52269

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x / (x^2 + m)

theorem max_f_value (m : ℝ) : 
  (m > 1) ↔ (∀ x : ℝ, f x m < 1) ∧ ¬((∀ x : ℝ, f x m < 1) → (m > 1)) :=
by
  sorry

end max_f_value_l52_52269


namespace no_perfect_square_l52_52998

-- Define the given polynomial
def poly (n : ℕ) : ℤ := n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3

-- The theorem to prove
theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, poly n = k^2 := by
  sorry

end no_perfect_square_l52_52998


namespace Lakota_spent_l52_52541

-- Define the conditions
def U : ℝ := 9.99
def Mackenzies_cost (N : ℝ) : ℝ := 3 * N + 8 * U
def cost_of_Lakotas_disks (N : ℝ) : ℝ := 6 * N + 2 * U

-- State the theorem
theorem Lakota_spent (N : ℝ) (h : Mackenzies_cost N = 133.89) : cost_of_Lakotas_disks N = 127.92 :=
by
  sorry

end Lakota_spent_l52_52541


namespace batsman_avg_after_17th_inning_l52_52753

def batsman_average : Prop :=
  ∃ (A : ℕ), 
    (A + 3 = (16 * A + 92) / 17) → 
    (A + 3 = 44)

theorem batsman_avg_after_17th_inning : batsman_average :=
by
  sorry

end batsman_avg_after_17th_inning_l52_52753


namespace arccos_cos_11_l52_52051

theorem arccos_cos_11 : Real.arccos (Real.cos 11) = 1.425 :=
by
  sorry

end arccos_cos_11_l52_52051


namespace max_min_values_of_f_l52_52398

noncomputable def f (x : ℝ) : ℝ := x^2

theorem max_min_values_of_f : 
  (∀ x, -3 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 0) :=
by
  sorry

end max_min_values_of_f_l52_52398


namespace inequality_amgm_l52_52110

theorem inequality_amgm (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) := 
by 
  sorry

end inequality_amgm_l52_52110


namespace simplify_expression_l52_52577

theorem simplify_expression (a1 a2 a3 a4 : ℝ) (h1 : 1 - a1 ≠ 0) (h2 : 1 - a2 ≠ 0) (h3 : 1 - a3 ≠ 0) (h4 : 1 - a4 ≠ 0) :
  1 + a1 / (1 - a1) + a2 / ((1 - a1) * (1 - a2)) + a3 / ((1 - a1) * (1 - a2) * (1 - a3)) + 
  (a4 - a1) / ((1 - a1) * (1 - a2) * (1 - a3) * (1 - a4)) = 
  1 / ((1 - a2) * (1 - a3) * (1 - a4)) :=
by
  sorry

end simplify_expression_l52_52577


namespace no_square_cube_l52_52987

theorem no_square_cube (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, k^2 = n * (n + 1) * (n + 2) * (n + 3)) ∧ ¬ (∃ l : ℕ, l^3 = n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end no_square_cube_l52_52987


namespace find_value_l52_52327

variable (a b c : Int)

-- Conditions from the problem
axiom abs_a_eq_two : |a| = 2
axiom b_eq_neg_seven : b = -7
axiom neg_c_eq_neg_five : -c = -5

-- Proof problem
theorem find_value : a^2 + (-b) + (-c) = 6 := by
  sorry

end find_value_l52_52327


namespace probability_sqrt_lt_7_of_random_two_digit_number_l52_52606

theorem probability_sqrt_lt_7_of_random_two_digit_number : 
  (∃ p : ℚ, (∀ n, 10 ≤ n ∧ n ≤ 99 → n < 49 → ∃ k, k = p) ∧ p = 13 / 30) := 
by
  sorry

end probability_sqrt_lt_7_of_random_two_digit_number_l52_52606


namespace x_when_y_is_125_l52_52727

noncomputable def C : ℝ := (2^2) * (5^2)

theorem x_when_y_is_125 
  (x y : ℝ) 
  (h_pos : x > 0 ∧ y > 0) 
  (h_inv : x^2 * y^2 = C) 
  (h_initial : y = 5) 
  (h_x_initial : x = 2) 
  (h_y : y = 125) : 
  x = 2 / 25 :=
by
  sorry

end x_when_y_is_125_l52_52727


namespace probability_red_side_first_on_third_roll_l52_52907

noncomputable def red_side_probability_first_on_third_roll : ℚ :=
  let p_non_red := 7 / 10
  let p_red := 3 / 10
  (p_non_red * p_non_red * p_red)

theorem probability_red_side_first_on_third_roll :
  red_side_probability_first_on_third_roll = 147 / 1000 := 
sorry

end probability_red_side_first_on_third_roll_l52_52907


namespace parker_net_income_after_taxes_l52_52376

noncomputable def parker_income : Real := sorry

theorem parker_net_income_after_taxes :
  let daily_pay := 63
  let hours_per_day := 8
  let hourly_rate := daily_pay / hours_per_day
  let overtime_rate := 1.5 * hourly_rate
  let overtime_hours_per_weekend_day := 3
  let weekends_in_6_weeks := 6
  let days_per_week := 7
  let total_days_in_6_weeks := days_per_week * weekends_in_6_weeks
  let regular_earnings := daily_pay * total_days_in_6_weeks
  let total_overtime_earnings := overtime_rate * overtime_hours_per_weekend_day * 2 * weekends_in_6_weeks
  let gross_income := regular_earnings + total_overtime_earnings
  let tax_rate := 0.1
  let net_income_after_taxes := gross_income * (1 - tax_rate)
  net_income_after_taxes = 2764.125 := by sorry

end parker_net_income_after_taxes_l52_52376


namespace binomial_7_2_l52_52481

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l52_52481


namespace find_principal_amount_l52_52073

def interest_rate_first_year : ℝ := 0.10
def compounding_periods_first_year : ℕ := 2
def interest_rate_second_year : ℝ := 0.12
def compounding_periods_second_year : ℕ := 4
def diff_interest : ℝ := 12

theorem find_principal_amount (P : ℝ)
  (h1_first : interest_rate_first_year / (compounding_periods_first_year : ℝ) = 0.05)
  (h1_second : interest_rate_second_year / (compounding_periods_second_year : ℝ) = 0.03)
  (compounded_amount : ℝ := P * (1 + 0.05)^(compounding_periods_first_year) * (1 + 0.03)^compounding_periods_second_year)
  (simple_interest : ℝ := P * (interest_rate_first_year + interest_rate_second_year) / 2 * 2)
  (h_diff : compounded_amount - P - simple_interest = diff_interest) : P = 597.01 :=
sorry

end find_principal_amount_l52_52073


namespace find_k_l52_52247

def line_p (x y : ℝ) : Prop := y = -2 * x + 3
def line_q (x y k : ℝ) : Prop := y = k * x + 4
def intersection (x y k : ℝ) : Prop := line_p x y ∧ line_q x y k

theorem find_k (k : ℝ) (h_inter : intersection 1 1 k) : k = -3 :=
sorry

end find_k_l52_52247


namespace sum_min_max_x_y_l52_52258

theorem sum_min_max_x_y (x y : ℕ) (h : 6 * x + 7 * y = 2012): 288 + 335 = 623 :=
by
  sorry

end sum_min_max_x_y_l52_52258


namespace cylinder_area_ratio_l52_52527

theorem cylinder_area_ratio (r h : ℝ) (h_eq : h = 2 * r * Real.sqrt π) :
  let S_lateral := 2 * π * r * h
  let S_total := S_lateral + 2 * π * r^2
  S_total / S_lateral = 1 + (1 / (2 * Real.sqrt π)) := by
sorry

end cylinder_area_ratio_l52_52527


namespace math_problem_l52_52467

theorem math_problem :
  (Int.ceil ((18: ℚ) / 5 * (-25 / 4)) - Int.floor ((18 / 5) * Int.floor (-25 / 4))) = 4 := 
by
  sorry

end math_problem_l52_52467


namespace jill_arrives_before_jack_l52_52843

def pool_distance : ℝ := 2
def jill_speed : ℝ := 12
def jack_speed : ℝ := 4
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem jill_arrives_before_jack
    (d : ℝ) (v_jill : ℝ) (v_jack : ℝ) (convert : ℝ → ℝ)
    (h_d : d = pool_distance)
    (h_vj : v_jill = jill_speed)
    (h_vk : v_jack = jack_speed)
    (h_convert : convert = hours_to_minutes) :
  convert (d / v_jack) - convert (d / v_jill) = 20 := by
  sorry

end jill_arrives_before_jack_l52_52843


namespace abs_a_gt_b_l52_52936

theorem abs_a_gt_b (a b : ℝ) (h : a > b) : |a| > b :=
sorry

end abs_a_gt_b_l52_52936


namespace inequality1_inequality2_l52_52232

theorem inequality1 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + a * b * c ≥ 2 * Real.sqrt 3 :=
by
  sorry

theorem inequality2 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (1 / A) + (1 / B) + (1 / C) ≥ 9 / Real.pi :=
by
  sorry

end inequality1_inequality2_l52_52232


namespace correct_option_l52_52217

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem correct_option : M ∪ (U \ N) = U :=
by
  sorry

end correct_option_l52_52217


namespace intersection_point_of_diagonals_l52_52719

noncomputable def intersection_of_diagonals (k m b : Real) : Real × Real :=
  let A := (0, b)
  let B := (0, -b)
  let C := (2 * b / (k - m), 2 * b * k / (k - m) - b)
  let D := (-2 * b / (k - m), -2 * b * k / (k - m) + b)
  (0, 0)

theorem intersection_point_of_diagonals (k m b : Real) :
  intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_point_of_diagonals_l52_52719


namespace john_total_payment_l52_52238

def cost_per_toy := 3
def number_of_toys := 5
def discount_rate := 0.2

theorem john_total_payment :
  (number_of_toys * cost_per_toy) - ((number_of_toys * cost_per_toy) * discount_rate) = 12 :=
by
  sorry

end john_total_payment_l52_52238


namespace find_number_l52_52028

theorem find_number (x : ℝ) (h : x * 2 + (12 + 4) * (1/8) = 602) : x = 300 :=
by
  sorry

end find_number_l52_52028


namespace molecular_weight_of_aluminum_part_in_Al2_CO3_3_l52_52318

def total_molecular_weight_Al2_CO3_3 : ℝ := 234
def atomic_weight_Al : ℝ := 26.98
def num_atoms_Al_in_Al2_CO3_3 : ℕ := 2

theorem molecular_weight_of_aluminum_part_in_Al2_CO3_3 :
  num_atoms_Al_in_Al2_CO3_3 * atomic_weight_Al = 53.96 :=
by
  sorry

end molecular_weight_of_aluminum_part_in_Al2_CO3_3_l52_52318


namespace factory_hours_per_day_l52_52969

def hour_worked_forth_machine := 12
def production_rate_per_hour := 2
def selling_price_per_kg := 50
def total_earnings := 8100

def h := 23

theorem factory_hours_per_day
  (num_machines : ℕ)
  (num_machines := 3)
  (production_first_three : ℕ := num_machines * production_rate_per_hour * h)
  (production_fourth : ℕ := hour_worked_forth_machine * production_rate_per_hour)
  (total_production : ℕ := production_first_three + production_fourth)
  (total_earnings_eq : total_production * selling_price_per_kg = total_earnings) :
  h = 23 := by
  sorry

end factory_hours_per_day_l52_52969


namespace strokes_over_par_l52_52892

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end strokes_over_par_l52_52892


namespace symmetric_line_eq_l52_52088

-- Define points A and B
def A (a : ℝ) : ℝ × ℝ := (a-1, a+1)
def B (a : ℝ) : ℝ × ℝ := (a, a)

-- We want to prove the equation of the line L about which points A and B are symmetric is "x - y + 1 = 0".
theorem symmetric_line_eq (a : ℝ) : 
  ∃ m b, (m = 1) ∧ (b = 1) ∧ (∀ x y, (y = m * x + b) ↔ (x - y + 1 = 0)) :=
sorry

end symmetric_line_eq_l52_52088


namespace temperature_equivalence_l52_52595

theorem temperature_equivalence (x : ℝ) (h : x = (9 / 5) * x + 32) : x = -40 :=
sorry

end temperature_equivalence_l52_52595


namespace uma_income_l52_52429

theorem uma_income
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 5000)
  (h2 : 3 * x - 2 * y = 5000) :
  4 * x = 20000 :=
by
  sorry

end uma_income_l52_52429


namespace poly_coeff_sum_l52_52341

theorem poly_coeff_sum :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ,
  (∀ x : ℤ, ((x^2 + 1) * (x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 + a_11 * x^11))
  ∧ a_0 = -512) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510) :=
by
  sorry

end poly_coeff_sum_l52_52341


namespace pages_read_on_wednesday_l52_52927

theorem pages_read_on_wednesday (W : ℕ) (h : 18 + W + 23 = 60) : W = 19 :=
by {
  sorry
}

end pages_read_on_wednesday_l52_52927


namespace summer_camp_students_l52_52138

theorem summer_camp_students (x : ℕ)
  (h1 : (1 / 6) * x = n_Shanghai)
  (h2 : n_Tianjin = 24)
  (h3 : (1 / 4) * x = n_Chongqing)
  (h4 : n_Beijing = (3 / 2) * (n_Shanghai + n_Tianjin)) :
  x = 180 :=
by
  sorry

end summer_camp_students_l52_52138


namespace part1_part2_l52_52431

-- Part 1: Inequality solution
theorem part1 (x : ℝ) :
  (1 / 3 * x - (3 * x + 4) / 6 ≤ 2 / 3) → (x ≥ -8) := 
by
  intro h
  sorry

-- Part 2: System of inequalities solution
theorem part2 (x : ℝ) :
  (4 * (x + 1) ≤ 7 * x + 13) ∧ ((x + 2) / 3 - x / 2 > 1) → (-3 ≤ x ∧ x < -2) := 
by
  intro h
  sorry

end part1_part2_l52_52431


namespace total_children_l52_52224

variable (S C B T : ℕ)

theorem total_children (h1 : T < 19) 
                       (h2 : S = 3 * C) 
                       (h3 : B = S / 2) 
                       (h4 : T = B + S + 1) : 
                       T = 10 := 
  sorry

end total_children_l52_52224


namespace remainder_when_divided_by_x_minus_2_l52_52903

def f (x : ℝ) : ℝ := x^5 - 6 * x^4 + 11 * x^3 + 21 * x^2 - 17 * x + 10

theorem remainder_when_divided_by_x_minus_2 : (f 2) = 84 := by
  sorry

end remainder_when_divided_by_x_minus_2_l52_52903


namespace range_of_m_l52_52526

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem range_of_m (m : ℝ) (h : second_quadrant (m-3) (m-2)) : 2 < m ∧ m < 3 :=
sorry

end range_of_m_l52_52526


namespace floor_of_smallest_zero_l52_52853
noncomputable def g (x : ℝ) := 3 * Real.sin x - Real.cos x + 2 * Real.tan x
def smallest_zero (s : ℝ) : Prop := s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem floor_of_smallest_zero (s : ℝ) (h : smallest_zero s) : ⌊s⌋ = 4 :=
sorry

end floor_of_smallest_zero_l52_52853


namespace binomial_7_2_l52_52482

theorem binomial_7_2 :
  Nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l52_52482


namespace sin_double_angle_of_tan_pi_sub_alpha_eq_two_l52_52343

theorem sin_double_angle_of_tan_pi_sub_alpha_eq_two 
  (α : Real) 
  (h : Real.tan (Real.pi - α) = 2) : 
  Real.sin (2 * α) = -4 / 5 := 
  by sorry

end sin_double_angle_of_tan_pi_sub_alpha_eq_two_l52_52343


namespace jane_coffees_l52_52236

open Nat

theorem jane_coffees (b m c n : Nat) 
  (h1 : b + m + c = 6)
  (h2 : 75 * b + 60 * m + 100 * c = 100 * n) :
  c = 1 :=
by sorry

end jane_coffees_l52_52236


namespace distinct_colored_cube_patterns_l52_52734

-- Define the colors
inductive Color
| Yellow
| Black
| Red
| Pink

-- Define a coloring of the cube
structure Coloring :=
(faces : Fin 6 → Color)

-- Define the group of symmetries of a cube
-- This would involve defining the rotations and considering color blindness

-- Define what it means for Mr. Li to consider two colorings the same
-- This typically involves permutations of red and pink faces

structure Cube :=
(symmetry : Fin 6 ≃ Fin 6)

theorem distinct_colored_cube_patterns :
  -- There are 5 distinct colorings of the cube according to Mr. Li’s recognition method.
  ℕ := sorry -- Correct answer is 5

end distinct_colored_cube_patterns_l52_52734


namespace window_design_ratio_l52_52642

theorem window_design_ratio (AB AD r : ℝ)
  (h1 : AB = 40)
  (h2 : AD / AB = 4 / 3)
  (h3 : r = AB / 2) :
  ((AD - AB) * AB) / (π * r^2 / 2) = 8 / (3 * π) :=
by
  sorry

end window_design_ratio_l52_52642


namespace Grace_minus_Lee_l52_52339

-- Definitions for the conditions
def Grace_calculation : ℤ := 12 - (3 * 4 - 2)
def Lee_calculation : ℤ := (12 - 3) * 4 - 2

-- Statement of the problem to prove
theorem Grace_minus_Lee : Grace_calculation - Lee_calculation = -32 := by
  sorry

end Grace_minus_Lee_l52_52339


namespace division_of_fractions_l52_52603

theorem division_of_fractions :
  (10 / 21) / (4 / 9) = 15 / 14 :=
by
  -- Proof will be provided here 
  sorry

end division_of_fractions_l52_52603


namespace faye_coloring_books_l52_52316

theorem faye_coloring_books (x : ℕ) : 34 - x + 48 = 79 → x = 3 :=
by
  sorry

end faye_coloring_books_l52_52316


namespace daisies_per_bouquet_l52_52630

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end daisies_per_bouquet_l52_52630


namespace toms_total_profit_l52_52241

def total_earnings_mowing : ℕ := 4 * 12 + 3 * 15 + 1 * 20
def total_earnings_side_jobs : ℕ := 2 * 10 + 3 * 8 + 1 * 12
def total_earnings : ℕ := total_earnings_mowing + total_earnings_side_jobs
def total_expenses : ℕ := 17 + 5
def total_profit : ℕ := total_earnings - total_expenses

theorem toms_total_profit : total_profit = 147 := by
  -- Proof omitted
  sorry

end toms_total_profit_l52_52241


namespace missing_digit_first_digit_l52_52755

-- Definitions derived from conditions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_divisible_by_six (n : ℕ) : Prop := n % 6 = 0
def multiply_by_two (d : ℕ) : ℕ := 2 * d

-- Main statement to prove
theorem missing_digit_first_digit (d : ℕ) (n : ℕ) 
  (h1 : multiply_by_two d = n) 
  (h2 : is_three_digit_number n) 
  (h3 : is_divisible_by_six n)
  (h4 : d = 2)
  : n / 100 = 2 :=
sorry

end missing_digit_first_digit_l52_52755


namespace function_domain_l52_52490

open Set

noncomputable def domain_of_function : Set ℝ :=
  {x | x ≠ 2}

theorem function_domain :
  domain_of_function = {x : ℝ | x ≠ 2} :=
by sorry

end function_domain_l52_52490


namespace minimum_value_expr_l52_52668

theorem minimum_value_expr (x : ℝ) (h : x > 2) :
  ∃ y, y = (x^2 - 6 * x + 8) / (2 * x - 4) ∧ y = -1/2 := sorry

end minimum_value_expr_l52_52668


namespace calculate_total_cost_l52_52173

-- Define the cost per workbook
def cost_per_workbook (x : ℝ) : ℝ := x

-- Define the number of workbooks
def number_of_workbooks : ℝ := 400

-- Define the total cost calculation
def total_cost (x : ℝ) : ℝ := number_of_workbooks * cost_per_workbook x

-- State the theorem to prove
theorem calculate_total_cost (x : ℝ) : total_cost x = 400 * x :=
by sorry

end calculate_total_cost_l52_52173


namespace choices_of_N_l52_52179

def base7_representation (N : ℕ) : ℕ := 
  (N / 49) * 100 + ((N % 49) / 7) * 10 + (N % 7)

def base8_representation (N : ℕ) : ℕ := 
  (N / 64) * 100 + ((N % 64) / 8) * 10 + (N % 8)

theorem choices_of_N : 
  ∃ (N_set : Finset ℕ), 
    (∀ N ∈ N_set, 100 ≤ N ∧ N < 1000 ∧ 
      ((base7_representation N * base8_representation N) % 100 = (3 * N) % 100)) 
    ∧ N_set.card = 15 :=
by
  sorry

end choices_of_N_l52_52179


namespace cone_height_l52_52833

theorem cone_height (r l h : ℝ) (h_r : r = 1) (h_l : l = 4) : h = Real.sqrt 15 :=
by
  -- proof steps would go here
  sorry

end cone_height_l52_52833


namespace heaviest_lightest_difference_l52_52141

-- Define 4 boxes' weights
variables {a b c d : ℕ}

-- Define given pairwise weights
axiom w1 : a + b = 22
axiom w2 : a + c = 23
axiom w3 : c + d = 30
axiom w4 : b + d = 29

-- Define the inequality among the weights
axiom h1 : a < b
axiom h2 : b < c
axiom h3 : c < d

-- Prove the heaviest box is 7 kg heavier than the lightest
theorem heaviest_lightest_difference : d - a = 7 :=
by sorry

end heaviest_lightest_difference_l52_52141


namespace solve_for_n_l52_52123

theorem solve_for_n (n : ℕ) (h : (16^n) * (16^n) * (16^n) * (16^n) * (16^n) = 256^5) : n = 2 := by
  sorry

end solve_for_n_l52_52123


namespace sum_of_reciprocals_l52_52136

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 := 
sorry

end sum_of_reciprocals_l52_52136


namespace _l52_52643

noncomputable def waiter_fraction_from_tips (S T I : ℝ) : Prop :=
  T = (5 / 2) * S ∧
  I = S + T ∧
  T / I = 5 / 7

lemma waiter_tips_fraction_theorem (S T I : ℝ) : waiter_fraction_from_tips S T I → T / I = 5 / 7 :=
by
  intro h
  rw [waiter_fraction_from_tips] at h
  obtain ⟨h₁, h₂, h₃⟩ := h
  exact h₃

end _l52_52643


namespace nancy_age_l52_52560

variable (n g : ℕ)

theorem nancy_age (h1 : g = 10 * n) (h2 : g - n = 45) : n = 5 :=
by
  sorry

end nancy_age_l52_52560


namespace Series_value_l52_52057

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end Series_value_l52_52057


namespace values_of_x_l52_52959

theorem values_of_x (x : ℝ) : (x+2)*(x-9) < 0 ↔ -2 < x ∧ x < 9 := 
by
  sorry

end values_of_x_l52_52959


namespace find_colored_copies_l52_52649

variable (cost_c cost_w total_copies total_cost : ℝ)
variable (colored_copies white_copies : ℝ)

def colored_copies_condition (cost_c cost_w total_copies total_cost : ℝ) :=
  ∃ (colored_copies white_copies : ℝ),
    colored_copies + white_copies = total_copies ∧
    cost_c * colored_copies + cost_w * white_copies = total_cost

theorem find_colored_copies :
  colored_copies_condition 0.10 0.05 400 22.50 → 
  ∃ (c : ℝ), c = 50 :=
by 
  sorry

end find_colored_copies_l52_52649


namespace find_3m_plus_n_l52_52364

theorem find_3m_plus_n (m n : ℕ) (h1 : m > n) (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 3 * m + n = 46 :=
sorry

end find_3m_plus_n_l52_52364


namespace quadratic_inequality_solution_l52_52503

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 - 6 * x - 16 > 0) ↔ (x < -2 ∨ x > 8) :=
sorry

end quadratic_inequality_solution_l52_52503


namespace geometric_sequence_common_ratio_and_general_formula_l52_52840

variable (a : ℕ → ℝ)

theorem geometric_sequence_common_ratio_and_general_formula (h₁ : a 1 = 1) (h₃ : a 3 = 4) : 
  (∃ q : ℝ, q = 2 ∨ q = -2 ∧ (∀ n : ℕ, a n = 2^(n-1) ∨ a n = (-2)^(n-1))) := 
by
  sorry

end geometric_sequence_common_ratio_and_general_formula_l52_52840


namespace thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l52_52186

theorem thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one : 37 * 23 = 851 := by
  sorry

end thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l52_52186


namespace cd_percentage_cheaper_l52_52621

theorem cd_percentage_cheaper (cost_cd cost_book cost_album difference percentage : ℝ) 
  (h1 : cost_book = cost_cd + 4)
  (h2 : cost_book = 18)
  (h3 : cost_album = 20)
  (h4 : difference = cost_album - cost_cd)
  (h5 : percentage = (difference / cost_album) * 100) : 
  percentage = 30 :=
sorry

end cd_percentage_cheaper_l52_52621


namespace total_cost_proof_l52_52722

-- Definitions for the problem conditions
def basketball_cost : ℕ := 48
def volleyball_cost : ℕ := basketball_cost - 18
def basketball_quantity : ℕ := 3
def volleyball_quantity : ℕ := 5
def total_basketball_cost : ℕ := basketball_cost * basketball_quantity
def total_volleyball_cost : ℕ := volleyball_cost * volleyball_quantity
def total_cost : ℕ := total_basketball_cost + total_volleyball_cost

-- Theorem to be proved
theorem total_cost_proof : total_cost = 294 :=
by
  sorry

end total_cost_proof_l52_52722


namespace probability_of_snow_during_holiday_l52_52666

theorem probability_of_snow_during_holiday
  (P_snow_Friday : ℝ)
  (P_snow_Monday : ℝ)
  (P_snow_independent : true) -- Placeholder since we assume independence
  (h_Friday: P_snow_Friday = 0.30)
  (h_Monday: P_snow_Monday = 0.45) :
  ∃ P_snow_holiday, P_snow_holiday = 0.615 :=
by
  sorry

end probability_of_snow_during_holiday_l52_52666


namespace twelve_people_pairing_l52_52413

noncomputable def num_ways_to_pair : ℕ := sorry

theorem twelve_people_pairing :
  (∀ (n : ℕ), n = 12 → (∃ f : ℕ → ℕ, ∀ i, f i = 2 ∨ f i = 12 ∨ f i = 7) → num_ways_to_pair = 3) := 
sorry

end twelve_people_pairing_l52_52413


namespace classroom_student_count_l52_52591

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end classroom_student_count_l52_52591


namespace marked_price_of_jacket_l52_52916

variable (x : ℝ) -- Define the variable x as a real number representing the marked price.

-- Define the conditions as a Lean theorem statement
theorem marked_price_of_jacket (cost price_sold profit : ℝ) (h1 : cost = 350) (h2 : price_sold = 0.8 * x) (h3 : profit = price_sold - cost) : 
  x = 550 :=
by
  -- We would solve the proof here using provided conditions
  sorry

end marked_price_of_jacket_l52_52916


namespace spectral_density_stationary_random_function_l52_52320

open Complex

-- Defining the correlation function
def correlation_function (τ : ℝ) : ℝ :=
  Real.exp (-Real.abs τ) * Real.cos τ

-- Defining the spectral density
noncomputable def spectral_density (ω : ℝ) : ℝ :=
  1 / (2 * Real.pi * (1 + ω^2))

-- The theorem we want to prove
theorem spectral_density_stationary_random_function : 
  ∀ ω : ℝ, s_x ω = spectral_density ω := 
sorry

end spectral_density_stationary_random_function_l52_52320


namespace largest_prime_factor_of_1729_l52_52002

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l52_52002


namespace N_is_85714_l52_52405

theorem N_is_85714 (N : ℕ) (hN : 10000 ≤ N ∧ N < 100000) 
  (P : ℕ := 200000 + N) 
  (Q : ℕ := 10 * N + 2) 
  (hQ_eq_3P : Q = 3 * P) 
  : N = 85714 := 
by 
  sorry

end N_is_85714_l52_52405


namespace solve_cos_sin_eq_one_l52_52381

open Real

theorem solve_cos_sin_eq_one (n : ℕ) (hn : n > 0) :
  {x : ℝ | cos x ^ n - sin x ^ n = 1} = {x : ℝ | ∃ k : ℤ, x = k * π} :=
by
  sorry

end solve_cos_sin_eq_one_l52_52381


namespace sam_age_l52_52573

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end sam_age_l52_52573


namespace black_white_tile_ratio_l52_52652

theorem black_white_tile_ratio :
  let original_black_tiles := 10
  let original_white_tiles := 15
  let total_tiles_in_original_square := original_black_tiles + original_white_tiles
  let side_length_of_original_square := Int.sqrt total_tiles_in_original_square -- this should be 5
  let side_length_of_extended_square := side_length_of_original_square + 2
  let total_black_tiles_in_border := 4 * (side_length_of_extended_square - 1) / 2 -- Each border side starts and ends with black
  let total_white_tiles_in_border := (side_length_of_extended_square * 4 - 4) - total_black_tiles_in_border 
  let new_total_black_tiles := original_black_tiles + total_black_tiles_in_border
  let new_total_white_tiles := original_white_tiles + total_white_tiles_in_border
  (new_total_black_tiles / gcd new_total_black_tiles new_total_white_tiles) / 
  (new_total_white_tiles / gcd new_total_black_tiles new_total_white_tiles) = 26 / 23 :=
by
  sorry

end black_white_tile_ratio_l52_52652


namespace perfume_weight_is_six_ounces_l52_52104

def weight_in_pounds (ounces : ℕ) : ℕ := ounces / 16

def initial_weight := 5  -- Initial suitcase weight in pounds
def final_weight := 11   -- Final suitcase weight in pounds
def chocolate := 4       -- Weight of chocolate in pounds
def soap := 2 * 5        -- Weight of 2 bars of soap in ounces
def jam := 2 * 8         -- Weight of 2 jars of jam in ounces

def total_additional_weight :=
  chocolate + (weight_in_pounds soap) + (weight_in_pounds jam)

def perfume_weight_in_pounds := final_weight - initial_weight - total_additional_weight

def perfume_weight_in_ounces := perfume_weight_in_pounds * 16

theorem perfume_weight_is_six_ounces : perfume_weight_in_ounces = 6 := by sorry

end perfume_weight_is_six_ounces_l52_52104


namespace find_other_number_l52_52126

theorem find_other_number
  (x y lcm hcf : ℕ)
  (h_lcm : Nat.lcm x y = lcm)
  (h_hcf : Nat.gcd x y = hcf)
  (h_x : x = 462)
  (h_lcm_value : lcm = 2310)
  (h_hcf_value : hcf = 30) :
  y = 150 :=
by
  sorry

end find_other_number_l52_52126


namespace geom_seq_a1_l52_52535

-- Define a geometric sequence.
def geom_seq (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q ^ n

-- Given conditions
def a2 (a : ℕ → ℝ) : Prop := a 1 = 2 -- because a2 = a(1) in zero-indexed
def a5 (a : ℕ → ℝ) : Prop := a 4 = -54 -- because a5 = a(4) in zero-indexed

-- Prove that a1 = -2/3
theorem geom_seq_a1 (a : ℕ → ℝ) (a1 q : ℝ) (h_geom : geom_seq a a1 q)
  (h_a2 : a2 a) (h_a5 : a5 a) : a1 = -2 / 3 :=
by
  sorry

end geom_seq_a1_l52_52535


namespace cafeteria_apples_count_l52_52723

def initial_apples : ℕ := 17
def used_monday : ℕ := 2
def bought_monday : ℕ := 23
def used_tuesday : ℕ := 4
def bought_tuesday : ℕ := 15
def used_wednesday : ℕ := 3

def final_apples (initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday : ℕ) : ℕ :=
  initial_apples - used_monday + bought_monday - used_tuesday + bought_tuesday - used_wednesday

theorem cafeteria_apples_count :
  final_apples initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday = 46 :=
by
  sorry

end cafeteria_apples_count_l52_52723


namespace radius_increase_l52_52743

-- Definitions and conditions
def initial_circumference : ℝ := 24
def final_circumference : ℝ := 30
def circumference_radius_relation (C : ℝ) (r : ℝ) : Prop := C = 2 * Real.pi * r

-- Required proof statement
theorem radius_increase (r1 r2 Δr : ℝ)
  (h1 : circumference_radius_relation initial_circumference r1)
  (h2 : circumference_radius_relation final_circumference r2)
  (h3 : Δr = r2 - r1) :
  Δr = 3 / Real.pi :=
by
  sorry

end radius_increase_l52_52743


namespace maximum_ratio_squared_l52_52849

theorem maximum_ratio_squared (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ge : a ≥ b)
  (x y : ℝ) (h_x : 0 ≤ x) (h_xa : x < a) (h_y : 0 ≤ y) (h_yb : y < b)
  (h_eq1 : a^2 + y^2 = b^2 + x^2)
  (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (a / b)^2 ≤ 4 / 3 :=
sorry

end maximum_ratio_squared_l52_52849


namespace min_inv_sum_four_l52_52680

theorem min_inv_sum_four (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  4 ≤ (1 / a + 1 / b) := 
sorry

end min_inv_sum_four_l52_52680


namespace option_C_correct_l52_52422

theorem option_C_correct : (Real.sqrt 2) * (Real.sqrt 6) = 2 * (Real.sqrt 3) :=
by sorry

end option_C_correct_l52_52422


namespace colored_line_midpoint_l52_52447

theorem colored_line_midpoint (L : ℝ → Prop) (p1 p2 : ℝ) :
  (L p1 → L p2) →
  (∃ A B C : ℝ, L A = L B ∧ L B = L C ∧ 2 * B = A + C ∧ L A = L C) :=
sorry

end colored_line_midpoint_l52_52447


namespace candy_mixture_price_l52_52971

theorem candy_mixture_price
  (a : ℝ)
  (h1 : 0 < a) -- Assuming positive amount of money spent, to avoid division by zero
  (p1 p2 : ℝ)
  (h2 : p1 = 2)
  (h3 : p2 = 3)
  (h4 : p2 * (a / p2) = p1 * (a / p1)) -- Condition that the total cost for each type is equal.
  : ( (p1 * (a / p1) + p2 * (a / p2)) / (a / p1 + a / p2) = 2.4 ) :=
  sorry

end candy_mixture_price_l52_52971


namespace right_triangle_consecutive_sides_l52_52972

theorem right_triangle_consecutive_sides (n : ℕ) (n_pos : 0 < n) :
    (n+1)^2 + n^2 = (n+2)^2 ↔ (n = 3) :=
by
  sorry

end right_triangle_consecutive_sides_l52_52972


namespace maximum_area_of_triangle_l52_52370

noncomputable def parabola (p : ℝ) : ℝ := -p^2 + 8 * p - 15

def area_of_triangle (p : ℝ) : ℝ :=
  let A := (2, 5)
  let B := (5, 10)
  let C := (p, parabola p)
  abs (2 * 10 + 5 * parabola p + p * 5 - 5 * 2 - 10 * p - parabola p * 2) / 2

theorem maximum_area_of_triangle : 
  ∃ p (h : 0 ≤ p ∧ p ≤ 5), area_of_triangle p = 112.5 / 24 :=
sorry

end maximum_area_of_triangle_l52_52370


namespace linda_original_savings_l52_52747

theorem linda_original_savings (S : ℝ) (h1 : 3 / 4 * S = 300 + 300) :
  S = 1200 :=
by
  sorry -- The proof is not required.

end linda_original_savings_l52_52747


namespace min_time_needed_l52_52620

-- Define the conditions and required time for shoeing horses
def num_blacksmiths := 48
def num_horses := 60
def hooves_per_horse := 4
def time_per_hoof := 5
def total_hooves := num_horses * hooves_per_horse
def total_time_one_blacksmith := total_hooves * time_per_hoof
def min_time (num_blacksmiths : Nat) (total_time_one_blacksmith : Nat) : Nat :=
  total_time_one_blacksmith / num_blacksmiths

-- Prove that the minimum time needed is 25 minutes
theorem min_time_needed : min_time num_blacksmiths total_time_one_blacksmith = 25 :=
by
  sorry

end min_time_needed_l52_52620


namespace range_of_k_l52_52947
noncomputable def quadratic_nonnegative (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 - 4 * x + 3 ≥ 0

theorem range_of_k (k : ℝ) : quadratic_nonnegative k ↔ k ∈ Set.Ici (4 / 3) :=
by
  sorry

end range_of_k_l52_52947


namespace solve_system_l52_52506

theorem solve_system : ∀ (x y : ℝ), x + 2 * y = 1 ∧ 2 * x + y = 2 → x + y = 1 :=
by
  intros x y h
  cases h with h₁ h₂
  sorry

end solve_system_l52_52506


namespace double_angle_cosine_calculation_l52_52647

theorem double_angle_cosine_calculation :
    2 * (Real.cos (Real.pi / 12))^2 - 1 = Real.cos (Real.pi / 6) := 
by
    sorry

end double_angle_cosine_calculation_l52_52647


namespace geometric_sequence_Sn_geometric_sequence_Sn_l52_52970

noncomputable def Sn (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1/3 then (27/2) - (1/2) * 3^(n - 3)
  else if q = 3 then (3^n - 1) / 2
  else 0

theorem geometric_sequence_Sn (a1 : ℝ) (n : ℕ) (h1 : a1 * (1/3) = 3)
  (h2 : a1 + a1 * (1/3)^2 = 10) : 
  Sn a1 (1/3) n = (27/2) - (1/2) * 3^(n - 3) :=
by
  sorry

theorem geometric_sequence_Sn' (a1 : ℝ) (n : ℕ) (h1 : a1 * 3 = 3) 
  (h2 : a1 + a1 * 3^2 = 10) : 
  Sn a1 3 n = (3^n - 1) / 2 :=
by
  sorry

end geometric_sequence_Sn_geometric_sequence_Sn_l52_52970


namespace weight_of_one_bowling_ball_l52_52993

-- Definitions from the problem conditions
def weight_canoe := 36
def num_canoes := 4
def num_bowling_balls := 9

-- Calculate the total weight of the canoes
def total_weight_canoes := num_canoes * weight_canoe

-- Prove the weight of one bowling ball
theorem weight_of_one_bowling_ball : (total_weight_canoes / num_bowling_balls) = 16 := by
  sorry

end weight_of_one_bowling_ball_l52_52993


namespace value_of_M_l52_52135

theorem value_of_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 :=
sorry

end value_of_M_l52_52135


namespace compare_neg_two_powers_l52_52184

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end compare_neg_two_powers_l52_52184


namespace biology_marks_l52_52793

theorem biology_marks 
  (e : ℕ) (m : ℕ) (p : ℕ) (c : ℕ) (a : ℕ) (n : ℕ) (b : ℕ) 
  (h_e : e = 96) (h_m : m = 95) (h_p : p = 82) (h_c : c = 97) (h_a : a = 93) (h_n : n = 5)
  (h_total : e + m + p + c + b = a * n) :
  b = 95 :=
by 
  sorry

end biology_marks_l52_52793


namespace percentage_of_25_of_fifty_percent_of_500_l52_52421

-- Define the constants involved
def fifty_percent_of_500 := 0.50 * 500  -- 50% of 500

-- Prove the equivalence
theorem percentage_of_25_of_fifty_percent_of_500 : (25 / fifty_percent_of_500) * 100 = 10 := by
  -- Place proof steps here
  sorry

end percentage_of_25_of_fifty_percent_of_500_l52_52421


namespace smallest_multiple_of_seven_gt_neg50_l52_52279

theorem smallest_multiple_of_seven_gt_neg50 : ∃ (n : ℤ), n % 7 = 0 ∧ n > -50 ∧ ∀ (m : ℤ), m % 7 = 0 → m > -50 → n ≤ m :=
sorry

end smallest_multiple_of_seven_gt_neg50_l52_52279


namespace g_inv_undefined_at_one_l52_52230

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

theorem g_inv_undefined_at_one :
  ∀ (x : ℝ), (∃ (y : ℝ), g y = x ∧ ¬ ∃ (z : ℝ), g z = y ∧ g z = 1) ↔ x = 1 :=
by
  sorry

end g_inv_undefined_at_one_l52_52230


namespace angle_complementary_supplementary_l52_52026

theorem angle_complementary_supplementary (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle1 + angle3 = 180)
  (h3 : angle3 = 125) :
  angle2 = 35 :=
by 
  sorry

end angle_complementary_supplementary_l52_52026


namespace lifespan_difference_l52_52882

variable (H : ℕ)

theorem lifespan_difference (H : ℕ) (bat_lifespan : ℕ) (frog_lifespan : ℕ) (total_lifespan : ℕ) 
    (hb : bat_lifespan = 10)
    (hf : frog_lifespan = 4 * H)
    (ht : H + bat_lifespan + frog_lifespan = total_lifespan)
    (t30 : total_lifespan = 30) :
    bat_lifespan - H = 6 :=
by
  -- here would be the proof
  sorry

end lifespan_difference_l52_52882


namespace beneficiary_received_32_176_l52_52041

noncomputable def A : ℝ := 19520 / 0.728
noncomputable def B : ℝ := 1.20 * A
noncomputable def C : ℝ := 1.44 * A
noncomputable def D : ℝ := 1.728 * A

theorem beneficiary_received_32_176 :
    round B = 32176 :=
by
    sorry

end beneficiary_received_32_176_l52_52041


namespace cube_painting_distinct_ways_l52_52190

theorem cube_painting_distinct_ways : ∃ n : ℕ, n = 7 := sorry

end cube_painting_distinct_ways_l52_52190


namespace exists_geometric_weak_arithmetic_l52_52939

theorem exists_geometric_weak_arithmetic (m : ℕ) (hm : 3 ≤ m) :
  ∃ (k : ℕ) (a : ℕ → ℕ), 
    (∀ i, 1 ≤ i → i ≤ m → a i = k^(m - i)*(k + 1)^(i - 1)) ∧
    ((∀ i, 1 ≤ i → i < m → a i < a (i + 1)) ∧ 
    ∃ (x : ℕ → ℕ) (d : ℕ), 
      (x 0 ≤ a 1 ∧ 
      ∀ i, 1 ≤ i → i < m → (x i ≤ a (i + 1) ∧ a (i + 1) < x (i + 1)) ∧ 
      ∀ i, 0 ≤ i → i < m - 1 → x (i + 1) - x i = d)) :=
by
  sorry

end exists_geometric_weak_arithmetic_l52_52939


namespace find_real_numbers_l52_52928

theorem find_real_numbers (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end find_real_numbers_l52_52928


namespace percentage_change_area_l52_52428

theorem percentage_change_area
    (L B : ℝ)
    (Area_original : ℝ) (Area_new : ℝ)
    (Length_new : ℝ) (Breadth_new : ℝ) :
    Area_original = L * B →
    Length_new = L / 2 →
    Breadth_new = 3 * B →
    Area_new = Length_new * Breadth_new →
    (Area_new - Area_original) / Area_original * 100 = 50 :=
  by
  intro h_orig_area hl_new hb_new ha_new
  sorry

end percentage_change_area_l52_52428


namespace train_length_calculation_l52_52770

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l52_52770


namespace solve_a_value_l52_52385

theorem solve_a_value (a b k : ℝ) 
  (h1 : a^3 * b^2 = k)
  (h2 : a = 5)
  (h3 : b = 2) :
  ∃ a', b = 8 → a' = 2.5 :=
by
  sorry

end solve_a_value_l52_52385


namespace parabola_constants_sum_l52_52389

-- Definition based on the given conditions
structure Parabola where
  a: ℝ
  b: ℝ
  c: ℝ
  vertex_x: ℝ
  vertex_y: ℝ
  point_x: ℝ
  point_y: ℝ

-- Definitions of the specific parabola based on the problem's conditions
noncomputable def givenParabola : Parabola := {
  a := -1/4,
  b := -5/2,
  c := -1/4,
  vertex_x := 6,
  vertex_y := -5,
  point_x := 2,
  point_y := -1
}

-- Theorem proving the required value of a + b + c
theorem parabola_constants_sum : givenParabola.a + givenParabola.b + givenParabola.c = -3.25 :=
  by
  sorry

end parabola_constants_sum_l52_52389


namespace percentage_supports_policy_l52_52171

theorem percentage_supports_policy
    (men_support_percentage : ℝ)
    (women_support_percentage : ℝ)
    (num_men : ℕ)
    (num_women : ℕ)
    (total_surveyed : ℕ)
    (total_supporters : ℕ)
    (overall_percentage : ℝ) :
    (men_support_percentage = 0.70) →
    (women_support_percentage = 0.75) →
    (num_men = 200) →
    (num_women = 800) →
    (total_surveyed = num_men + num_women) →
    (total_supporters = (men_support_percentage * num_men) + (women_support_percentage * num_women)) →
    (overall_percentage = (total_supporters / total_surveyed) * 100) →
    overall_percentage = 74 :=
by
  intros
  sorry

end percentage_supports_policy_l52_52171


namespace solution_set_inequality_l52_52525

theorem solution_set_inequality (m : ℝ) (x : ℝ) 
  (h : 3 - m < 0) : (2 - m) * x + m > 2 ↔ x < 1 :=
by
  sorry

end solution_set_inequality_l52_52525


namespace num_of_lists_is_correct_l52_52027

theorem num_of_lists_is_correct :
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  total_lists = 50625 :=
by
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  show total_lists = 50625
  sorry

end num_of_lists_is_correct_l52_52027


namespace cost_per_box_types_l52_52756

-- Definitions based on conditions
def cost_type_B := 1500
def cost_type_A := cost_type_B + 500

-- Given conditions
def condition1 : cost_type_A = cost_type_B + 500 := by sorry
def condition2 : 6000 / (cost_type_B + 500) = 4500 / cost_type_B := by sorry

-- Theorem to be proved
theorem cost_per_box_types :
  cost_type_A = 2000 ∧ cost_type_B = 1500 ∧
  (∃ (m : ℕ), 20 ≤ m ∧ m ≤ 25 ∧ 2000 * (50 - m) + 1500 * m ≤ 90000) ∧
  (∃ (a b : ℕ), 2500 * a + 3500 * b = 87500 ∧ a + b ≤ 33) :=
sorry

end cost_per_box_types_l52_52756


namespace average_candies_correct_l52_52596

def candy_counts : List ℕ := [16, 22, 30, 26, 18, 20]
def num_members : ℕ := 6
def total_candies : ℕ := List.sum candy_counts
def average_candies : ℕ := total_candies / num_members

theorem average_candies_correct : average_candies = 22 := by
  -- Proof is omitted, as per instructions
  sorry

end average_candies_correct_l52_52596


namespace plums_in_basket_l52_52619

theorem plums_in_basket (initial : ℕ) (added : ℕ) (total : ℕ) (h_initial : initial = 17) (h_added : added = 4) : total = 21 := by
  sorry

end plums_in_basket_l52_52619


namespace cos_frac_less_sin_frac_l52_52848

theorem cos_frac_less_sin_frac : 
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  a < b :=
by
  let a := Real.cos (3 / 2)
  let b := Real.sin (1 / 10)
  sorry -- proof skipped

end cos_frac_less_sin_frac_l52_52848


namespace Miranda_can_stuff_pillows_l52_52698

theorem Miranda_can_stuff_pillows:
  let pounds_per_pillow := 2 in
  let goose_feathers_per_pound := 300 in
  let duck_feathers_per_pound := 500 in
  let total_goose_feathers := 3600 in
  let total_duck_feathers := 4000 in
  let goose_feathers_in_pounds := total_goose_feathers / goose_feathers_per_pound in
  let duck_feathers_in_pounds := total_duck_feathers / duck_feathers_per_pound in
  let total_feathers_in_pounds := goose_feathers_in_pounds + duck_feathers_in_pounds in
  let pillows_stuffed := total_feathers_in_pounds / pounds_per_pillow in
  pillows_stuffed = 10 := by
  sorry

end Miranda_can_stuff_pillows_l52_52698


namespace product_segment_doubles_l52_52842

-- Define the problem conditions and proof statement in Lean.
theorem product_segment_doubles
  (a b e : ℝ)
  (d : ℝ := (a * b) / e)
  (e' : ℝ := e / 2)
  (d' : ℝ := (a * b) / e') :
  d' = 2 * d := 
  sorry

end product_segment_doubles_l52_52842


namespace find_B_l52_52366

theorem find_B (B: ℕ) (h1: 5457062 % 2 = 0 ∧ 200 * B % 4 = 0) (h2: 5457062 % 5 = 0 ∧ B % 5 = 0) (h3: 5450062 % 8 = 0 ∧ 100 * B % 8 = 0) : B = 0 :=
sorry

end find_B_l52_52366


namespace magician_inequality_l52_52448

theorem magician_inequality (N : ℕ) : 
  (N - 1) * 10^(N - 2) ≥ 10^N → N ≥ 101 :=
by
  sorry

end magician_inequality_l52_52448


namespace sin_double_angle_value_l52_52078

theorem sin_double_angle_value 
  (h1 : Real.pi / 2 < α ∧ α < β ∧ β < 3 * Real.pi / 4)
  (h2 : Real.cos (α - β) = 12 / 13)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.sin (2 * α) = -16 / 65 :=
by
  sorry

end sin_double_angle_value_l52_52078


namespace maisy_earnings_increase_l52_52250

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l52_52250


namespace class_b_students_l52_52050

theorem class_b_students (total_students : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total_students = 100 → sample_size = 10 → class_a_sample = 4 → 
  (total_students - total_students * class_a_sample / sample_size = 60) :=
by
  intros
  sorry

end class_b_students_l52_52050


namespace binomial_7_2_l52_52472

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end binomial_7_2_l52_52472


namespace part_1_part_2_l52_52828

-- Conditions and definitions
noncomputable def triangle_ABC (a b c S : ℝ) (A B C : ℝ) :=
  a * Real.sin B = -b * Real.sin (A + Real.pi / 3) ∧
  S = Real.sqrt 3 / 4 * c^2

-- 1. Prove A = 5 * Real.pi / 6
theorem part_1 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  A = 5 * Real.pi / 6 :=
  sorry

-- 2. Prove sin C = sqrt 7 / 14 given S = sqrt 3 / 4 * c^2
theorem part_2 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  Real.sin C = Real.sqrt 7 / 14 :=
  sorry

end part_1_part_2_l52_52828


namespace polynomial_roots_l52_52929

theorem polynomial_roots :
  Polynomial.roots (Polynomial.C 4 * Polynomial.X ^ 5 +
                    Polynomial.C 13 * Polynomial.X ^ 4 +
                    Polynomial.C (-30) * Polynomial.X ^ 3 +
                    Polynomial.C 8 * Polynomial.X ^ 2) =
  {0, 0, 1 / 2, -2 + 2 * Real.sqrt 2, -2 - 2 * Real.sqrt 2} :=
by
  sorry

end polynomial_roots_l52_52929


namespace list_length_eq_12_l52_52340

-- Define a list of numbers in the sequence
def seq : List ℝ := [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5]

-- Define the theorem that states the number of elements in the sequence
theorem list_length_eq_12 : seq.length = 12 := 
by 
  -- Proof here
  sorry

end list_length_eq_12_l52_52340


namespace part_I_part_II_l52_52517

noncomputable def f (x : ℝ) := |x - 2| - |2 * x + 1|

theorem part_I :
  { x : ℝ | f x ≤ 0 } = { x : ℝ | x ≤ -3 ∨ x ≥ (1 : ℝ) / 3 } :=
by
  sorry

theorem part_II :
  ∀ x : ℝ, f x - 2 * m^2 ≤ 4 * m :=
by
  sorry

end part_I_part_II_l52_52517


namespace effective_average_speed_l52_52635

def rowing_speed_with_stream := 16 -- km/h
def rowing_speed_against_stream := 6 -- km/h
def stream1_effect := 2 -- km/h
def stream2_effect := -1 -- km/h
def stream3_effect := 3 -- km/h
def opposing_wind := 1 -- km/h

theorem effective_average_speed :
  ((rowing_speed_with_stream + stream1_effect - opposing_wind) + 
   (rowing_speed_against_stream + stream2_effect - opposing_wind) + 
   (rowing_speed_with_stream + stream3_effect - opposing_wind)) / 3 = 13 := 
by
  sorry

end effective_average_speed_l52_52635


namespace total_cows_in_herd_l52_52445

theorem total_cows_in_herd {n : ℚ} (h1 : 1/3 + 1/6 + 1/9 = 11/18) 
                           (h2 : (1 - 11/18) = 7/18) 
                           (h3 : 8 = (7/18) * n) : 
                           n = 144/7 :=
by sorry

end total_cows_in_herd_l52_52445


namespace conditional_probability_B0_conditional_probability_B1_conditional_probability_B2_probability_distribution_X_l52_52412

open locale classical
open ProbabilityTheory

-- Definitions of the conditions
def event_A (boxes : Finset ℕ) (checked_boxes : Finset ℕ) : Prop :=
  ∀ i ∈ checked_boxes, boxes i = 0

def event_B0 (boxes : Finset ℕ) : Prop :=
  ∀ i ∈ boxes, boxes i = 0

def event_B1 (boxes : Finset ℕ) : Prop :=
  ∃ i ∈ boxes, boxes i = 1 ∧ ∀ j ∈ boxes, j ≠ i → boxes j = 0

def event_B2 (boxes : Finset ℕ) : Prop :=
  ∃ i j ∈ boxes, i ≠ j ∧ boxes i = 1 ∧ boxes j = 1 ∧ ∀ k ∈ boxes, k ≠ i ∧ k ≠ j → boxes k = 0

noncomputable def P (boxes : Finset ℕ) (p : Prop) : ℝ := sorry

-- Proof statements
theorem conditional_probability_B0 (boxes : Finset ℕ) (checked_boxes : Finset ℕ) :
  (event_B0 boxes → event_A boxes checked_boxes) = 1 := sorry

theorem conditional_probability_B1 (boxes : Finset ℕ) (checked_boxes : Finset ℕ) :
  (event_B1 boxes → event_A boxes checked_boxes) = 4 / 5 := sorry

theorem conditional_probability_B2 (boxes : Finset ℕ) (checked_boxes : Finset ℕ) :
  (event_B2 boxes → event_A boxes checked_boxes) = 12 / 19 := sorry

theorem probability_distribution_X :
  ∀ (X : ℕ), 
    (X = 0 → P X (X = 0) = 877 / 950) ∧ 
    (X = 1 → P X (X = 1) = 70 / 950) ∧ 
    (X = 2 → P X (X = 2) = 3 / 950) := sorry

end conditional_probability_B0_conditional_probability_B1_conditional_probability_B2_probability_distribution_X_l52_52412


namespace determine_good_numbers_l52_52449

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), (∀ k : Fin n, ∃ m : ℕ, k.1 + (a k).1 + 1 = m * m)

theorem determine_good_numbers :
  is_good_number 13 ∧ is_good_number 15 ∧ is_good_number 17 ∧ is_good_number 19 ∧ ¬is_good_number 11 :=
by
  sorry

end determine_good_numbers_l52_52449


namespace total_erasers_is_35_l52_52787

def Celine : ℕ := 10

def Gabriel : ℕ := Celine / 2

def Julian : ℕ := Celine * 2

def total_erasers : ℕ := Celine + Gabriel + Julian

theorem total_erasers_is_35 : total_erasers = 35 :=
  by
  sorry

end total_erasers_is_35_l52_52787


namespace solve_for_x_l52_52870

theorem solve_for_x :
  ∀ x : ℕ, 100^4 = 5^x → x = 8 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l52_52870


namespace fraction_subtraction_l52_52277

theorem fraction_subtraction : 
  (3 + 6 + 9) = 18 ∧ (2 + 5 + 8) = 15 ∧ (2 + 5 + 8) = 15 ∧ (3 + 6 + 9) = 18 →
  (18 / 15 - 15 / 18) = 11 / 30 :=
by
  intro h
  sorry

end fraction_subtraction_l52_52277


namespace arithmetic_sequence_1001th_term_l52_52131

theorem arithmetic_sequence_1001th_term (p q : ℚ)
    (h1 : p + 3 * q = 12)
    (h2 : 12 + 3 * q = 3 * p - q) :
    (p + (1001 - 1) * (3 * q) = 5545) :=
by
  sorry

end arithmetic_sequence_1001th_term_l52_52131


namespace neg_A_is_square_of_int_l52_52946

theorem neg_A_is_square_of_int (x y z : ℤ) (A : ℤ) (h1 : A = x * y + y * z + z * x) 
  (h2 : A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1)) : ∃ k : ℤ, -A = k^2 :=
by
  sorry

end neg_A_is_square_of_int_l52_52946


namespace power_function_through_point_l52_52395

noncomputable def f : ℝ → ℝ := sorry

theorem power_function_through_point (h : ∀ x, ∃ a : ℝ, f x = x^a) (h1 : f 3 = 27) :
  f x = x^3 :=
sorry

end power_function_through_point_l52_52395


namespace polynomial_divisible_l52_52885

theorem polynomial_divisible (A B : ℝ) (h : ∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + A * x + B = 0) : A + B = -1 :=
by
  sorry

end polynomial_divisible_l52_52885


namespace h_inch_approx_l52_52586

noncomputable def h_cm : ℝ := 14.5 - 2 * 1.7
noncomputable def cm_to_inch (cm : ℝ) : ℝ := cm / 2.54
noncomputable def h_inch : ℝ := cm_to_inch h_cm

theorem h_inch_approx : abs (h_inch - 4.37) < 1e-2 :=
by
  -- The proof is omitted
  sorry

end h_inch_approx_l52_52586


namespace total_amount_spent_l52_52105

-- Definitions based on the conditions
def games_this_month := 11
def cost_per_ticket_this_month := 25
def total_cost_this_month := games_this_month * cost_per_ticket_this_month

def games_last_month := 17
def cost_per_ticket_last_month := 30
def total_cost_last_month := games_last_month * cost_per_ticket_last_month

def games_next_month := 16
def cost_per_ticket_next_month := 35
def total_cost_next_month := games_next_month * cost_per_ticket_next_month

-- Lean statement for the proof problem
theorem total_amount_spent :
  total_cost_this_month + total_cost_last_month + total_cost_next_month = 1345 :=
by
  -- proof goes here
  sorry

end total_amount_spent_l52_52105


namespace percent_women_surveryed_equal_40_l52_52533

theorem percent_women_surveryed_equal_40
  (W M : ℕ) 
  (h1 : W + M = 100)
  (h2 : (W / 100 * 1 / 10 : ℚ) + (M / 100 * 1 / 4 : ℚ) = (19 / 100 : ℚ))
  (h3 : (9 / 10 : ℚ) * (W / 100 : ℚ) + (3 / 4 : ℚ) * (M / 100 : ℚ) = (1 - 19 / 100 : ℚ)) :
  W = 40 := 
sorry

end percent_women_surveryed_equal_40_l52_52533


namespace cheryl_used_total_material_correct_amount_l52_52648

def material_used (initial leftover : ℚ) : ℚ := initial - leftover

def total_material_used 
  (initial_a initial_b initial_c leftover_a leftover_b leftover_c : ℚ) : ℚ :=
  material_used initial_a leftover_a + material_used initial_b leftover_b + material_used initial_c leftover_c

theorem cheryl_used_total_material_correct_amount :
  total_material_used (2/9) (1/8) (3/10) (4/18) (1/12) (3/15) = 17/120 :=
by
  sorry

end cheryl_used_total_material_correct_amount_l52_52648


namespace find_b_l52_52497

theorem find_b (x : ℝ) (b : ℝ) :
  (∃ t u : ℝ, (bx^2 + 18 * x + 9) = (t * x + u)^2 ∧ u^2 = 9 ∧ 2 * t * u = 18 ∧ t^2 = b) →
  b = 9 :=
by
  sorry

end find_b_l52_52497


namespace cylinder_volume_l52_52786

theorem cylinder_volume (short_side long_side : ℝ) (h_short_side : short_side = 12) (h_long_side : long_side = 18) : 
  ∀ (r h : ℝ) (h_radius : r = short_side / 2) (h_height : h = long_side), 
    volume = π * r^2 * h := 
by
  sorry

end cylinder_volume_l52_52786


namespace quadrant_of_theta_l52_52521

theorem quadrant_of_theta (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin θ < 0) : (0 < θ ∧ θ < π/2) ∨ (3*π/2 < θ ∧ θ < 2*π) :=
by
  sorry

end quadrant_of_theta_l52_52521


namespace product_of_four_consecutive_integers_is_perfect_square_l52_52566

-- Define the main statement we want to prove
theorem product_of_four_consecutive_integers_is_perfect_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  -- Proof is omitted
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l52_52566


namespace points_on_ray_MA_l52_52941

-- Define the distance function
def dist (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Define points A and B
variables (A B : ℝ × ℝ)

-- Define the midpoint M of segment AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define a predicate that characterizes the points P on ray MA excluding midpoint M
def on_ray_MA_except_M (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  let M := midpoint A B in
  (P ≠ M) ∧ (dist A P < dist M P ∨ (P.1 - A.1) / (M.1 - A.1) = (P.2 - A.2) / (M.2 - A.2))

-- The theorem to prove
theorem points_on_ray_MA (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  dist P B > dist P A ↔ on_ray_MA_except_M P A B :=
sorry

end points_on_ray_MA_l52_52941


namespace sum_of_two_digit_factors_l52_52658

theorem sum_of_two_digit_factors (a b : ℕ) (h : a * b = 5681) (h1 : 10 ≤ a) (h2 : a < 100) (h3 : 10 ≤ b) (h4 : b < 100) : a + b = 154 :=
by
  sorry

end sum_of_two_digit_factors_l52_52658


namespace Danny_bottle_caps_l52_52792

theorem Danny_bottle_caps (r w c : ℕ) (h1 : r = 11) (h2 : c = r + 1) : c = 12 := by
  sorry

end Danny_bottle_caps_l52_52792


namespace combination_7_2_l52_52476

theorem combination_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end combination_7_2_l52_52476


namespace width_of_plot_is_correct_l52_52915

-- Definitions based on the given conditions
def cost_per_acre_per_month : ℝ := 60
def total_monthly_rent : ℝ := 600
def length_of_plot : ℝ := 360
def sq_feet_per_acre : ℝ := 43560

-- Theorems to be proved based on the conditions and the correct answer
theorem width_of_plot_is_correct :
  let number_of_acres := total_monthly_rent / cost_per_acre_per_month
  let total_sq_footage := number_of_acres * sq_feet_per_acre
  let width_of_plot := total_sq_footage / length_of_plot
  width_of_plot = 1210 :=
by 
  sorry

end width_of_plot_is_correct_l52_52915


namespace tissue_properties_l52_52904

noncomputable def actual_diameter (magnified_diameter magnification_factor : ℝ) : ℝ :=
  magnified_diameter / magnification_factor

noncomputable def radius (diameter : ℝ) : ℝ :=
  diameter / 2

noncomputable def area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

noncomputable def circumference (radius : ℝ) : ℝ :=
  2 * Real.pi * radius

theorem tissue_properties :
  ∀ (magnified_diameter magnification_factor : ℝ),
  magnified_diameter = 5 →
  magnification_factor = 1000 →
  actual_diameter magnified_diameter magnification_factor = 0.005 ∧
  area (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.000019635 ∧
  circumference (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.01570795 :=
by
  intros magnified_diameter magnification_factor h1 h2
  have h_diameter : actual_diameter magnified_diameter magnification_factor = 0.005
  {
    rw [h1, h2]
    norm_num
  }
  have h_radius : radius (actual_diameter magnified_diameter magnification_factor) = 0.0025
  {
    rw [h_diameter]
    norm_num
  }
  have h_area : area (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.000019635
  {
    simp [h_radius, area]
    norm_num
    simp [Real.pi]
  }
  have h_circumference : circumference (radius (actual_diameter magnified_diameter magnification_factor)) ≈ 0.01570795
  {
    simp [h_radius, circumference]
    norm_num
    simp [Real.pi]
  }
  exact ⟨h_diameter, h_area, h_circumference⟩

end tissue_properties_l52_52904


namespace flux_vector_field_through_Sigma_l52_52802

open Real Set MeasureTheory

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (y^2 + z^2, x*y + y^2, x*z + z)

noncomputable def surface_Σ : Set (ℝ × ℝ × ℝ) :=
  {p | (∃ z ∈ Icc (0:ℝ) (1:ℝ), (p.1)^2 + (p.2)^2 = 1 ∧ p.3 = z) ∨
       (p.3 = 0 ∧ (p.1)^2 + (p.2)^2 ≤ 1) ∨
       (p.3 = 1 ∧ (p.1)^2 + (p.2)^2 ≤ 1) }

theorem flux_vector_field_through_Sigma :
  ∫ (p : ℝ × ℝ × ℝ) in surface_Σ, 
  (vector_field p.1 p.2 p.3).1 * p.1 + (vector_field p.1 p.2 p.3).2 * p.2 + 
  (vector_field p.1 p.2 p.3).3 * p.3 = 2 * π :=
sorry

end flux_vector_field_through_Sigma_l52_52802


namespace find_minimum_m_l52_52708

theorem find_minimum_m (m : ℕ) (h1 : 1350 + 36 * m < 2136) (h2 : 1500 + 45 * m ≥ 2365) :
  m = 20 :=
by
  sorry

end find_minimum_m_l52_52708


namespace bill_new_win_percentage_l52_52465

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

end bill_new_win_percentage_l52_52465


namespace sum_and_product_of_roots_l52_52280

-- Define the polynomial equation and the conditions on the roots
def cubic_eqn (x : ℝ) : Prop := 3 * x ^ 3 - 18 * x ^ 2 + 27 * x - 6 = 0

-- The Lean statement for the given problem
theorem sum_and_product_of_roots (p q r : ℝ) :
  cubic_eqn p ∧ cubic_eqn q ∧ cubic_eqn r →
  (p + q + r = 6) ∧ (p * q * r = 2) :=
by
  sorry

end sum_and_product_of_roots_l52_52280


namespace train_length_l52_52773

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l52_52773


namespace problem_1_problem_2_l52_52337

open Set Real

-- Definition of the sets A, B, and the complement of B in the real numbers
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Proof problem (1): Prove that A ∩ (complement of B) = [1, 2]
theorem problem_1 : (A ∩ (compl B)) = {x | 1 ≤ x ∧ x ≤ 2} := sorry

-- Proof problem (2): Prove that the set of values for the real number a such that C(a) ∩ A = C(a)
-- is (-∞, 3]
theorem problem_2 : { a : ℝ | C a ⊆ A } = { a : ℝ | a ≤ 3 } := sorry

end problem_1_problem_2_l52_52337


namespace bus_speed_excluding_stoppages_l52_52494

variable (v : ℝ) -- Speed of the bus excluding stoppages

-- Conditions
def bus_stops_per_hour := 45 / 60 -- 45 minutes converted to hours
def effective_driving_time := 1 - bus_stops_per_hour -- Effective time driving in an hour

-- Given Condition
def speed_including_stoppages := 12 -- Speed including stoppages in km/hr

theorem bus_speed_excluding_stoppages 
  (h : effective_driving_time * v = speed_including_stoppages) : 
  v = 48 :=
sorry

end bus_speed_excluding_stoppages_l52_52494


namespace part1_l52_52617

theorem part1 (a b c : ℚ) (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : c^3 = 27) (h4 : a * b < 0) (h5 : b * c > 0) : 
  a * b - b * c + c * a = -33 := by
  sorry

end part1_l52_52617


namespace largest_prime_factor_1729_l52_52010

theorem largest_prime_factor_1729 :
  ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
sorry

end largest_prime_factor_1729_l52_52010


namespace binomial_7_2_l52_52487

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l52_52487


namespace greater_number_is_18_l52_52137

theorem greater_number_is_18 (x y : ℕ) (h₁ : x + y = 30) (h₂ : x - y = 6) : x = 18 :=
by
  sorry

end greater_number_is_18_l52_52137


namespace root_expression_value_l52_52985

theorem root_expression_value (p q r : ℝ) (hpq : p + q + r = 15) (hpqr : p * q + q * r + r * p = 25) (hpqrs : p * q * r = 10) :
  (p / (2 / p + q * r) + q / (2 / q + r * p) + r / (2 / r + p * q) = 175 / 12) :=
by sorry

end root_expression_value_l52_52985


namespace find_white_balls_l52_52684

-- Define a structure to hold the probabilities and total balls
structure BallProperties where
  totalBalls : Nat
  probRed : Real
  probBlack : Real

-- Given data as conditions
def givenData : BallProperties := 
  { totalBalls := 50, probRed := 0.15, probBlack := 0.45 }

-- The statement to prove the number of white balls
theorem find_white_balls (data : BallProperties) : 
  data.totalBalls = 50 →
  data.probRed = 0.15 →
  data.probBlack = 0.45 →
  ∃ whiteBalls : Nat, whiteBalls = 20 :=
by
  sorry

end find_white_balls_l52_52684


namespace percentage_failed_both_l52_52973

theorem percentage_failed_both (p_hindi p_english p_pass_both x : ℝ)
  (h₁ : p_hindi = 0.25)
  (h₂ : p_english = 0.5)
  (h₃ : p_pass_both = 0.5)
  (h₄ : (p_hindi + p_english - x) = 0.5) : 
  x = 0.25 := 
sorry

end percentage_failed_both_l52_52973


namespace percent_of_employed_people_who_are_females_l52_52977

theorem percent_of_employed_people_who_are_females (p_employed p_employed_males : ℝ) 
  (h1 : p_employed = 64) (h2 : p_employed_males = 48) : 
  100 * (p_employed - p_employed_males) / p_employed = 25 :=
by
  sorry

end percent_of_employed_people_who_are_females_l52_52977


namespace race_time_l52_52909

theorem race_time (v_A v_B : ℝ) (t tB : ℝ) (h1 : 200 / v_A = t) (h2 : 144 / v_B = t) (h3 : 200 / v_B = t + 7) : t = 18 :=
by
  sorry

end race_time_l52_52909


namespace find_y_l52_52170

noncomputable def x : Real := 1.6666666666666667
def y : Real := 5

theorem find_y (h : x ≠ 0) (h1 : (x * y) / 3 = x^2) : y = 5 := 
by sorry

end find_y_l52_52170


namespace sum_minimal_area_k_l52_52718

def vertices_triangle_min_area (k : ℤ) : Prop :=
  let x1 := 1
  let y1 := 7
  let x2 := 13
  let y2 := 16
  let x3 := 5
  ((y1 - k) * (x2 - x1) ≠ (x1 - x3) * (y2 - y1))

def minimal_area_sum_k : ℤ :=
  9 + 11

theorem sum_minimal_area_k :
  ∃ k1 k2 : ℤ, vertices_triangle_min_area k1 ∧ vertices_triangle_min_area k2 ∧ k1 + k2 = 20 := 
sorry

end sum_minimal_area_k_l52_52718


namespace series_sum_equals_one_sixth_l52_52055

noncomputable def series_sum : ℝ :=
  ∑' n, 2^n / (7^(2^n) + 1)

theorem series_sum_equals_one_sixth : series_sum = 1 / 6 :=
by
  sorry

end series_sum_equals_one_sixth_l52_52055


namespace remainder_div_P_by_D_plus_D_l52_52667

theorem remainder_div_P_by_D_plus_D' 
  (P Q D R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D + D') = R :=
by
  -- Proof is not required.
  sorry

end remainder_div_P_by_D_plus_D_l52_52667


namespace polynomial_factorization_l52_52404

-- Definitions from conditions
def p (x : ℝ) : ℝ := x^6 - 2 * x^4 + 6 * x^3 + x^2 - 6 * x + 9
def q (x : ℝ) : ℝ := (x^3 - x + 3)^2

-- The theorem statement proving question == answer given conditions
theorem polynomial_factorization : ∀ x : ℝ, p x = q x :=
by
  sorry

end polynomial_factorization_l52_52404


namespace prove_logical_proposition_l52_52092

theorem prove_logical_proposition (p q : Prop) (hp : p) (hq : ¬q) : (¬p ∨ ¬q) :=
by
  sorry

end prove_logical_proposition_l52_52092


namespace smallest_z_value_l52_52839

theorem smallest_z_value :
  ∀ w x y z : ℤ, (∃ k : ℤ, w = 2 * k - 1 ∧ x = 2 * k + 1 ∧ y = 2 * k + 3 ∧ z = 2 * k + 5) ∧
    w^3 + x^3 + y^3 = z^3 →
    z = 9 :=
sorry

end smallest_z_value_l52_52839


namespace find_number_l52_52322

-- Definitions and conditions
def unknown_number (x : ℝ) : Prop :=
  (14 / 100) * x = 98

-- Theorem to prove
theorem find_number (x : ℝ) : unknown_number x → x = 700 := by
  sorry

end find_number_l52_52322


namespace minimal_volume_block_l52_52039

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end minimal_volume_block_l52_52039


namespace math_problems_l52_52954

theorem math_problems (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (a * (6 - a) ≤ 9) ∧
  (ab = a + b + 3 → ab ≥ 9) ∧
  ¬(∀ x : ℝ, 0 < x → x^2 + 4 / (x^2 + 3) ≥ 1) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end math_problems_l52_52954


namespace problem_solution_l52_52822

theorem problem_solution (x : ℝ) (h : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
by
  sorry

end problem_solution_l52_52822


namespace card_distribution_count_l52_52996

theorem card_distribution_count : 
  ∃ (methods : ℕ), methods = 18 ∧ 
  ∃ (cards : Finset ℕ),
  ∃ (envelopes : Finset (Finset ℕ)), 
  cards = {1, 2, 3, 4, 5, 6} ∧ 
  envelopes.card = 3 ∧ 
  (∀ e ∈ envelopes, (e.card = 2) ∧ ({1, 2} ⊆ e → ∃ e1 e2, {e1, e2} ∈ envelopes ∧ {e1, e2} ⊆ cards \ {1, 2})) ∧ 
  (∀ c1 ∈ cards, ∃ e ∈ envelopes, c1 ∈ e) :=
by
  sorry

end card_distribution_count_l52_52996


namespace norma_initial_cards_l52_52374

def initial_card_count (lost: ℕ) (left: ℕ) : ℕ :=
  lost + left

theorem norma_initial_cards : initial_card_count 70 18 = 88 :=
  by
    -- skipping proof
    sorry

end norma_initial_cards_l52_52374


namespace downstream_speed_l52_52919

-- Define constants based on conditions given 
def V_upstream : ℝ := 30
def V_m : ℝ := 35

-- Define the speed of the stream based on the given conditions and upstream speed
def V_s : ℝ := V_m - V_upstream

-- The downstream speed is the man's speed in still water plus the stream speed
def V_downstream : ℝ := V_m + V_s

-- Theorem to be proved
theorem downstream_speed : V_downstream = 40 :=
by
  -- The actual proof steps are omitted
  sorry

end downstream_speed_l52_52919


namespace minimum_value_expression_l52_52107

theorem minimum_value_expression (a b c d e f : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
(h_sum : a + b + c + d + e + f = 7) : 
  ∃ min_val : ℝ, min_val = 63 ∧ 
  (∀ a b c d e f : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 0 < f → a + b + c + d + e + f = 7 → 
  (1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f) ≥ min_val) := 
sorry

end minimum_value_expression_l52_52107


namespace find_ac_bd_l52_52618

variable (a b c d : ℝ)

axiom cond1 : a^2 + b^2 = 1
axiom cond2 : c^2 + d^2 = 1
axiom cond3 : a * d - b * c = 1 / 7

theorem find_ac_bd : a * c + b * d = 4 * Real.sqrt 3 / 7 := by
  sorry

end find_ac_bd_l52_52618


namespace Xiaokang_position_l52_52906

theorem Xiaokang_position :
  let east := 150
  let west := 100
  let total_walks := 3
  (east - west - west = -50) :=
sorry

end Xiaokang_position_l52_52906


namespace lcm_gcd_product_difference_l52_52199
open Nat

theorem lcm_gcd_product_difference :
  (Nat.lcm 12 9) * (Nat.gcd 12 9) - (Nat.gcd 15 9) = 105 :=
by
  sorry

end lcm_gcd_product_difference_l52_52199


namespace train_length_l52_52774

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l52_52774


namespace binom_7_2_eq_21_l52_52484

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l52_52484


namespace john_pays_12_dollars_l52_52237

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end john_pays_12_dollars_l52_52237


namespace min_value_x_add_2y_l52_52513

theorem min_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end min_value_x_add_2y_l52_52513


namespace greatest_value_of_sum_l52_52740

variable (x y : ℝ)

-- Conditions
axiom sum_of_squares : x^2 + y^2 = 130
axiom product : x * y = 36

-- Statement to prove
theorem greatest_value_of_sum : x + y ≤ Real.sqrt 202 := sorry

end greatest_value_of_sum_l52_52740


namespace solve_equation_l52_52265

theorem solve_equation :
  ∀ x : ℝ, 18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = 4.5 ∨ x = -3) :=
by
  sorry

end solve_equation_l52_52265


namespace ring_display_capacity_l52_52446

def necklace_capacity : ℕ := 12
def current_necklaces : ℕ := 5
def ring_capacity : ℕ := 18
def bracelet_capacity : ℕ := 15
def current_bracelets : ℕ := 8
def necklace_cost : ℕ := 4
def ring_cost : ℕ := 10
def bracelet_cost : ℕ := 5
def total_cost : ℕ := 183

theorem ring_display_capacity : ring_capacity + (total_cost - ((necklace_capacity - current_necklaces) * necklace_cost + (bracelet_capacity - current_bracelets) * bracelet_cost)) / ring_cost = 30 := by
  sorry

end ring_display_capacity_l52_52446


namespace sum_faces_edges_vertices_of_octagonal_pyramid_l52_52665

-- We define an octagonal pyramid with the given geometric properties.
structure OctagonalPyramid :=
  (base_vertices : ℕ) -- the number of vertices of the base
  (base_edges : ℕ)    -- the number of edges of the base
  (apex : ℕ)          -- the single apex of the pyramid
  (faces : ℕ)         -- the total number of faces: base face + triangular faces
  (edges : ℕ)         -- the total number of edges
  (vertices : ℕ)      -- the total number of vertices

-- Now we instantiate the structure based on the conditions.
def octagonalPyramid : OctagonalPyramid :=
  { base_vertices := 8,
    base_edges := 8,
    apex := 1,
    faces := 9,
    edges := 16,
    vertices := 9 }

-- We prove that the total number of faces, edges, and vertices sum to 34.
theorem sum_faces_edges_vertices_of_octagonal_pyramid : 
  (octagonalPyramid.faces + octagonalPyramid.edges + octagonalPyramid.vertices = 34) :=
by
  -- The proof steps are omitted as per instruction.
  sorry

end sum_faces_edges_vertices_of_octagonal_pyramid_l52_52665


namespace Buffy_whiskers_l52_52208

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l52_52208


namespace average_height_plants_l52_52434

theorem average_height_plants (h1 h3 : ℕ) (h1_eq : h1 = 27) (h3_eq : h3 = 9)
  (prop : ∀ (h2 h4 : ℕ), (h2 = h1 / 3 ∨ h2 = h1 * 3) ∧ (h3 = h2 / 3 ∨ h3 = h2 * 3) ∧ (h4 = h3 / 3 ∨ h4 = h3 * 3)) : 
  ((27 + h2 + 9 + h4) / 4 = 12) :=
by 
  sorry

end average_height_plants_l52_52434


namespace rectangle_length_l52_52746

theorem rectangle_length (side_square length_rectangle width_rectangle wire_length : ℝ) 
    (h1 : side_square = 12) 
    (h2 : width_rectangle = 6) 
    (h3 : wire_length = 4 * side_square) 
    (h4 : wire_length = 2 * width_rectangle + 2 * length_rectangle) : 
    length_rectangle = 18 := 
by 
  sorry

end rectangle_length_l52_52746


namespace dog_speed_is_16_kmh_l52_52631

variable (man's_speed : ℝ := 4) -- man's speed in km/h
variable (total_path_length : ℝ := 625) -- total path length in meters
variable (remaining_distance : ℝ := 81) -- remaining distance in meters

theorem dog_speed_is_16_kmh :
  let total_path_length_km := total_path_length / 1000
  let remaining_distance_km := remaining_distance / 1000
  let man_covered_distance_km := total_path_length_km - remaining_distance_km
  let time := man_covered_distance_km / man's_speed
  let dog_total_distance_km := 4 * (2 * total_path_length_km)
  let dog_speed := dog_total_distance_km / time
  dog_speed = 16 :=
by
  sorry

end dog_speed_is_16_kmh_l52_52631


namespace largest_side_of_rectangle_l52_52375

theorem largest_side_of_rectangle :
  ∃ (l w : ℝ), (2 * l + 2 * w = 240) ∧ (l * w = 12 * 240) ∧ (l = 86.835 ∨ w = 86.835) :=
by
  -- Actual proof would be here
  sorry

end largest_side_of_rectangle_l52_52375


namespace probability_both_hit_l52_52538

-- Define the probabilities of hitting the target for shooters A and B.
def prob_A_hits : ℝ := 0.7
def prob_B_hits : ℝ := 0.8

-- Define the independence condition (not needed as a direct definition but implicitly acknowledges independence).
axiom A_and_B_independent : true

-- The statement we want to prove: the probability that both shooters hit the target.
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.56 :=
by
  -- Placeholder for proof
  sorry

end probability_both_hit_l52_52538


namespace exists_common_point_l52_52702

-- Definitions: Rectangle and the problem conditions
structure Rectangle :=
(x_min y_min x_max y_max : ℝ)
(h_valid : x_min ≤ x_max ∧ y_min ≤ y_max)

def rectangles_intersect (R1 R2 : Rectangle) : Prop :=
¬(R1.x_max < R2.x_min ∨ R2.x_max < R1.x_min ∨ R1.y_max < R2.y_min ∨ R2.y_max < R1.y_min)

def all_rectangles_intersect (rects : List Rectangle) : Prop :=
∀ (R1 R2 : Rectangle), R1 ∈ rects → R2 ∈ rects → rectangles_intersect R1 R2

-- Theorem: Existence of a common point
theorem exists_common_point (rects : List Rectangle) (h_intersect : all_rectangles_intersect rects) : 
  ∃ (T : ℝ × ℝ), ∀ (R : Rectangle), R ∈ rects → 
    R.x_min ≤ T.1 ∧ T.1 ≤ R.x_max ∧ 
    R.y_min ≤ T.2 ∧ T.2 ≤ R.y_max := 
sorry

end exists_common_point_l52_52702


namespace cylinder_height_and_diameter_l52_52139

/-- The surface area of a sphere is the same as the curved surface area of a right circular cylinder.
    The height and diameter of the cylinder are the same, and the radius of the sphere is 4 cm.
    Prove that the height and diameter of the cylinder are both 8 cm. --/
theorem cylinder_height_and_diameter (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ)
  (h1 : r_sphere = 4)
  (h2 : 4 * π * r_sphere^2 = 2 * π * r_cylinder * h_cylinder)
  (h3 : h_cylinder = 2 * r_cylinder) :
  h_cylinder = 8 ∧ r_cylinder = 4 :=
by
  -- Proof to be completed
  sorry

end cylinder_height_and_diameter_l52_52139


namespace problem_statement_l52_52157

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x - Real.sqrt x ≤ y - 1 / 4 ∧ y - 1 / 4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1 / 4 ∧ x - 1 / 4 ≤ y + Real.sqrt y :=
sorry

end problem_statement_l52_52157


namespace compare_neg5_neg7_l52_52789

theorem compare_neg5_neg7 : -5 > -7 := 
by
  sorry

end compare_neg5_neg7_l52_52789


namespace parallelogram_angle_l52_52427

theorem parallelogram_angle (a b : ℕ) (h : a + b = 180) (exceed_by_10 : b = a + 10) : a = 85 := by
  -- proof skipped
  sorry

end parallelogram_angle_l52_52427


namespace ice_bag_cost_correct_l52_52462

def total_cost_after_discount (cost_small cost_large : ℝ) (num_bags num_small : ℕ) (discount_rate : ℝ) : ℝ :=
  let num_large := num_bags - num_small
  let total_cost_before_discount := num_small * cost_small + num_large * cost_large
  let discount := discount_rate * total_cost_before_discount
  total_cost_before_discount - discount

theorem ice_bag_cost_correct :
  total_cost_after_discount 0.80 1.46 30 18 0.12 = 28.09 :=
by
  sorry

end ice_bag_cost_correct_l52_52462


namespace binomial_22_5_computation_l52_52510

theorem binomial_22_5_computation (h1 : Nat.choose 20 3 = 1140) (h2 : Nat.choose 20 4 = 4845) (h3 : Nat.choose 20 5 = 15504) :
    Nat.choose 22 5 = 26334 := by
  sorry

end binomial_22_5_computation_l52_52510


namespace factorize_x2_minus_2x_plus_1_l52_52066

theorem factorize_x2_minus_2x_plus_1 :
  ∀ (x : ℝ), x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  intro x
  linarith

end factorize_x2_minus_2x_plus_1_l52_52066


namespace buffy_whiskers_l52_52204

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l52_52204


namespace find_a2_l52_52945

variable (x : ℝ)
variable (a₀ a₁ a₂ a₃ : ℝ)
axiom condition : ∀ x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

theorem find_a2 : a₂ = 6 :=
by
  -- The proof that involves verifying the Taylor series expansion will come here
  sorry

end find_a2_l52_52945


namespace propositions_A_and_D_true_l52_52154

theorem propositions_A_and_D_true :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧ (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) :=
by
  sorry

end propositions_A_and_D_true_l52_52154


namespace roger_expenses_fraction_l52_52704

theorem roger_expenses_fraction {B t s n : ℝ} (h1 : t = 0.25 * (B - s))
  (h2 : s = 0.10 * (B - t)) (h3 : n = 5) :
  (t + s + n) / B = 0.41 :=
sorry

end roger_expenses_fraction_l52_52704


namespace find_original_one_digit_number_l52_52031

theorem find_original_one_digit_number (x : ℕ) (h1 : x < 10) (h2 : (x + 10) * (x + 10) / x = 72) : x = 2 :=
sorry

end find_original_one_digit_number_l52_52031


namespace cut_ribbon_l52_52430

theorem cut_ribbon
    (length_ribbon : ℝ)
    (points : ℝ × ℝ × ℝ × ℝ × ℝ)
    (h_length : length_ribbon = 5)
    (h_points : points = (1, 2, 3, 4, 5)) :
    points.2.1 = (11 / 15) * length_ribbon :=
by
    sorry

end cut_ribbon_l52_52430


namespace range_of_c_over_a_l52_52330

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + b + c = 0) : -2 < c / a ∧ c / a < -1 :=
by {
  sorry
}

end range_of_c_over_a_l52_52330


namespace gamma_start_time_correct_l52_52301

noncomputable def trisection_points (AB : ℕ) : Prop := AB ≥ 3

structure Walkers :=
  (d : ℕ) -- Total distance AB
  (Vα : ℕ) -- Speed of person α
  (Vβ : ℕ) -- Speed of person β
  (Vγ : ℕ) -- Speed of person γ

def meeting_times (w : Walkers) := 
  w.Vα = w.d / 72 ∧ 
  w.Vβ = w.d / 36 ∧ 
  w.Vγ = w.Vβ

def start_times_correct (startA timeA_meetC : ℕ) (startB timeB_reachesA: ℕ) (startC_latest: ℕ): Prop :=
  startA = 0 ∧ 
  startB = 12 ∧
  timeA_meetC = 24 ∧ 
  timeB_reachesA = 30 ∧
  startC_latest = 16

theorem gamma_start_time_correct (AB : ℕ) (w : Walkers) (t : Walkers → Prop) : 
  trisection_points AB → 
  meeting_times w →
  start_times_correct 0 24 12 30 16 → 
  ∃ tγ_start, tγ_start = 16 :=
sorry

end gamma_start_time_correct_l52_52301


namespace find_y_plus_one_over_y_l52_52333

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end find_y_plus_one_over_y_l52_52333


namespace no_valid_sum_seventeen_l52_52359

def std_die (n : ℕ) : Prop := n ∈ [1, 2, 3, 4, 5, 6]

def valid_dice (a b c d : ℕ) : Prop := std_die a ∧ std_die b ∧ std_die c ∧ std_die d

def sum_dice (a b c d : ℕ) : ℕ := a + b + c + d

def prod_dice (a b c d : ℕ) : ℕ := a * b * c * d

theorem no_valid_sum_seventeen (a b c d : ℕ) (h_valid : valid_dice a b c d) (h_prod : prod_dice a b c d = 360) : sum_dice a b c d ≠ 17 :=
sorry

end no_valid_sum_seventeen_l52_52359


namespace eating_time_175_seconds_l52_52444

variable (Ponchik_time Neznaika_time : ℝ)
variable (Ponchik_rate Neznaika_rate : ℝ)

theorem eating_time_175_seconds
    (hP_rate : Ponchik_rate = 1 / Ponchik_time)
    (hP_time : Ponchik_time = 5)
    (hN_rate : Neznaika_rate = 1 / Neznaika_time)
    (hN_time : Neznaika_time = 7)
    (combined_rate := Ponchik_rate + Neznaika_rate)
    (total_minutes := 1 / combined_rate)
    (total_seconds := total_minutes * 60):
    total_seconds = 175 := by
  sorry

end eating_time_175_seconds_l52_52444


namespace sum_of_four_interior_edges_l52_52297

-- Define the given conditions
def is_two_inch_frame (w : ℕ) := w = 2
def frame_area (A : ℕ) := A = 68
def outer_edge_length (L : ℕ) := L = 15

-- Define the inner dimensions calculation function
def inner_dimensions (outerL outerH frameW : ℕ) := 
  (outerL - 2 * frameW, outerH - 2 * frameW)

-- Define the final question in Lean 4 reflective of the equivalent proof problem
theorem sum_of_four_interior_edges (w A L y : ℕ) 
  (h1 : is_two_inch_frame w) 
  (h2 : frame_area A)
  (h3 : outer_edge_length L)
  (h4 : 15 * y - (15 - 2 * w) * (y - 2 * w) = A)
  : 2 * (15 - 2 * w) + 2 * (y - 2 * w) = 26 := 
sorry

end sum_of_four_interior_edges_l52_52297


namespace original_volume_of_ice_l52_52176

theorem original_volume_of_ice (V : ℝ) 
  (h1 : V * (1/4) * (1/4) = 0.4) : 
  V = 6.4 :=
sorry

end original_volume_of_ice_l52_52176


namespace train_length_is_correct_l52_52641

noncomputable def train_length (speed_kmph : ℝ) (time_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_sec
  total_distance - bridge_length

theorem train_length_is_correct :
  train_length 60 20.99832013438925 240 = 110 :=
by
  sorry

end train_length_is_correct_l52_52641


namespace daisies_per_bouquet_is_7_l52_52627

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end daisies_per_bouquet_is_7_l52_52627


namespace total_vehicles_in_lanes_l52_52453

theorem total_vehicles_in_lanes :
  ∀ (lanes : ℕ) (trucks_per_lane cars_total trucks_total : ℕ),
  lanes = 4 →
  trucks_per_lane = 60 →
  trucks_total = trucks_per_lane * lanes →
  cars_total = 2 * trucks_total →
  (trucks_total + cars_total) = 2160 :=
by intros lanes trucks_per_lane cars_total trucks_total hlanes htrucks_per_lane htrucks_total hcars_total
   -- sorry added to skip the proof
   sorry

end total_vehicles_in_lanes_l52_52453


namespace race_order_l52_52025

theorem race_order (overtakes_G_S_L : (ℕ × ℕ × ℕ))
  (h1 : overtakes_G_S_L.1 = 10)
  (h2 : overtakes_G_S_L.2.1 = 4)
  (h3 : overtakes_G_S_L.2.2 = 6)
  (h4 : ¬(overtakes_G_S_L.2.1 > 0 ∧ overtakes_G_S_L.2.2 > 0))
  (h5 : ∀ i j k : ℕ, i ≠ j → j ≠ k → k ≠ i)
  : overtakes_G_S_L = (10, 4, 6) :=
sorry

end race_order_l52_52025


namespace maximize_probability_l52_52443

variable {p1 p2 p3 : ℝ}
variable {p1_gt_zero : p1 > 0}
variable {h1 : p3 > p2}
variable {h2 : p2 > p1}

def probability_p_A : ℝ := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def probability_p_B : ℝ := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def probability_p_C : ℝ := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability :
  probability_p_C > probability_p_A ∧ probability_p_C > probability_p_B := by
  sorry

end maximize_probability_l52_52443


namespace stratified_sampling_difference_l52_52407

theorem stratified_sampling_difference
  (male_athletes : ℕ := 56)
  (female_athletes : ℕ := 42)
  (sample_size : ℕ := 28)
  (H_total : male_athletes + female_athletes = 98)
  (H_sample_frac : sample_size = 28)
  : (56 * (sample_size / 98) - 42 * (sample_size / 98) = 4) :=
sorry

end stratified_sampling_difference_l52_52407


namespace Sam_age_l52_52568

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l52_52568


namespace ensure_two_different_colors_ensure_two_yellow_balls_l52_52974

-- First statement: Ensuring two balls of different colors
theorem ensure_two_different_colors (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 11 ∧ 
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, draws i ≠ draws j := 
sorry

-- Second statement: Ensuring two yellow balls
theorem ensure_two_yellow_balls (balls_red balls_white balls_yellow : Nat)
  (hr : balls_red = 10) (hw : balls_white = 10) (hy : balls_yellow = 10) :
  ∃ n, n >= 22 ∧
       ∀ draws : Fin n → Fin (balls_red + balls_white + balls_yellow), 
       ∃ i j, (draws i).val - balls_red - balls_white < balls_yellow ∧ 
              (draws j).val - balls_red - balls_white < balls_yellow ∧
              draws i = draws j := 
sorry

end ensure_two_different_colors_ensure_two_yellow_balls_l52_52974


namespace total_consumer_installment_credit_l52_52178

-- Conditions
def auto_instalment_credit (C : ℝ) : ℝ := 0.2 * C
def auto_finance_extends_1_third (auto_installment : ℝ) : ℝ := 57
def student_loans (C : ℝ) : ℝ := 0.15 * C
def credit_card_debt (C : ℝ) (auto_installment : ℝ) : ℝ := 0.25 * C
def other_loans (C : ℝ) : ℝ := 0.4 * C

-- Correct Answer
theorem total_consumer_installment_credit (C : ℝ) :
  auto_instalment_credit C / 3 = auto_finance_extends_1_third (auto_instalment_credit C) ∧
  student_loans C = 80 ∧
  credit_card_debt C (auto_instalment_credit C) = auto_instalment_credit C + 100 ∧
  credit_card_debt C (auto_instalment_credit C) = 271 →
  C = 1084 := 
by
  sorry

end total_consumer_installment_credit_l52_52178


namespace pipe_tank_fill_time_l52_52562

/-- 
Given:
1. Pipe A fills the tank in 2 hours.
2. The leak empties the tank in 4 hours.
Prove: 
The tank is filled in 4 hours when both Pipe A and the leak are working together.
 -/
theorem pipe_tank_fill_time :
  let A := 1 / 2 -- rate at which Pipe A fills the tank (tank per hour)
  let L := 1 / 4 -- rate at which the leak empties the tank (tank per hour)
  let net_rate := A - L -- net rate of filling the tank
  net_rate > 0 → (1 / net_rate) = 4 := 
by
  intros
  sorry

end pipe_tank_fill_time_l52_52562


namespace buffy_whiskers_l52_52210

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l52_52210


namespace chess_player_max_consecutive_win_prob_l52_52440

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l52_52440


namespace triangle_is_isosceles_l52_52829

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) 
  (h1 : c = 2 * a * Real.cos B) 
  (h2 : a = b) :
  ∃ (isIsosceles : Bool), isIsosceles := 
sorry

end triangle_is_isosceles_l52_52829


namespace right_triangle_area_eq_8_over_3_l52_52663

-- Definitions arising from the conditions in the problem
variable (a b c : ℝ)

-- The conditions as Lean definitions
def condition1 : Prop := b = (2/3) * a
def condition2 : Prop := b = (2/3) * c

-- The question translated into a proof problem: proving that the area of the triangle equals 8/3
theorem right_triangle_area_eq_8_over_3 (h1 : condition1 a b) (h2 : condition2 b c) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 8/3 :=
by
  sorry

end right_triangle_area_eq_8_over_3_l52_52663


namespace profit_equation_example_l52_52289

noncomputable def profit_equation (a b : ℝ) (x : ℝ) : Prop :=
  a * (1 + x) ^ 2 = b

theorem profit_equation_example :
  profit_equation 250 360 x :=
by
  have : 25 * (1 + x) ^ 2 = 36 := sorry
  sorry

end profit_equation_example_l52_52289


namespace calc_value_l52_52468

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end calc_value_l52_52468


namespace purely_imaginary_condition_l52_52854

-- Define the necessary conditions
def real_part_eq_zero (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0
def imaginary_part_neq_zero (m : ℝ) : Prop := m^2 - 3 * m + 2 ≠ 0

-- State the theorem to be proved
theorem purely_imaginary_condition (m : ℝ) :
  real_part_eq_zero m ∧ imaginary_part_neq_zero m ↔ m = -1/2 :=
sorry

end purely_imaginary_condition_l52_52854


namespace find_sam_age_l52_52571

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l52_52571


namespace largest_integer_le_zero_l52_52349

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero (x k : ℝ) (h1 : f x = 0) (h2 : 2 < x) (h3 : x < 3) : k ≤ x ∧ k = 2 :=
by
  sorry

end largest_integer_le_zero_l52_52349


namespace probability_sqrt_two_digit_lt_7_l52_52605

theorem probability_sqrt_two_digit_lt_7 : 
  let two_digit_set := Finset.Icc 10 99
  let favorable_set := Finset.Icc 10 48
  (favorable_set.card : ℚ) / two_digit_set.card = 13 / 30 :=
by sorry

end probability_sqrt_two_digit_lt_7_l52_52605


namespace garage_travel_time_correct_l52_52844

theorem garage_travel_time_correct :
  let floors := 12
  let gate_interval := 3
  let gate_time := 2 * 60 -- 2 minutes in seconds
  let distance_per_floor := 800
  let speed := 10 in
  let num_gates := floors / gate_interval
  let total_gate_time := num_gates * gate_time
  let time_per_floor := distance_per_floor / speed
  let total_drive_time := floors * time_per_floor in
  total_gate_time + total_drive_time = 1440 := by
  sorry

end garage_travel_time_correct_l52_52844


namespace angle_measure_x_l52_52534

theorem angle_measure_x
    (angle_CBE : ℝ)
    (angle_EBD : ℝ)
    (angle_ABE : ℝ)
    (sum_angles_TRIA : ∀ a b c : ℝ, a + b + c = 180)
    (sum_straight_ANGLE : ∀ a b : ℝ, a + b = 180) :
    angle_CBE = 124 → angle_EBD = 33 → angle_ABE = 19 → x = 91 :=
by
    sorry

end angle_measure_x_l52_52534


namespace find_omega_find_g_min_max_l52_52516

open Real

def f (ω x : ℝ) := sin (ω * x - π / 6) + sin (ω * x - π / 2)
def g (x : ℝ) := sqrt 3 * sin (x - π / 12)

theorem find_omega (ω : ℝ) (h₀ : 0 < ω) (h₁ : ω < 3) (h₂ : f ω (π / 6) = 0) : ω = 2 :=
sorry

theorem find_g_min_max : 
  ∃ (min max : ℝ), min = -3 / 2 ∧ max = sqrt 3 ∧
    ∀ (x : ℝ), -π / 4 ≤ x ∧ x ≤ 3 * π / 4 → (g x ≥ min ∧ g x ≤ max) :=
sorry

end find_omega_find_g_min_max_l52_52516


namespace polynomial_division_l52_52751

variable (a p x : ℝ)

theorem polynomial_division :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end polynomial_division_l52_52751


namespace binomial_7_2_l52_52486

theorem binomial_7_2 : nat.choose 7 2 = 21 :=
by
  sorry

end binomial_7_2_l52_52486


namespace remainder_product_div_17_l52_52741

theorem remainder_product_div_17 :
  (2357 ≡ 6 [MOD 17]) → (2369 ≡ 4 [MOD 17]) → (2384 ≡ 0 [MOD 17]) →
  (2391 ≡ 9 [MOD 17]) → (3017 ≡ 9 [MOD 17]) → (3079 ≡ 0 [MOD 17]) →
  (3082 ≡ 3 [MOD 17]) →
  ((2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) ≡ 0 [MOD 17]) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end remainder_product_div_17_l52_52741


namespace total_ticket_sales_l52_52890

def ticket_price : Type := 
  ℕ → ℕ

def total_individual_sales (student_count adult_count child_count senior_count : ℕ) (prices : ticket_price) : ℝ :=
  (student_count * prices 6 + adult_count * prices 8 + child_count * prices 4 + senior_count * prices 7)

def total_group_sales (group_student_count group_adult_count group_child_count group_senior_count : ℕ) (prices : ticket_price) : ℝ :=
  let total_price := (group_student_count * prices 6 + group_adult_count * prices 8 + group_child_count * prices 4 + group_senior_count * prices 7)
  if (group_student_count + group_adult_count + group_child_count + group_senior_count) > 10 then 
    total_price - 0.10 * total_price 
  else 
    total_price

theorem total_ticket_sales
  (prices : ticket_price)
  (student_count adult_count child_count senior_count : ℕ)
  (group_student_count group_adult_count group_child_count group_senior_count : ℕ)
  (total_sales : ℝ) :
  student_count = 20 →
  adult_count = 12 →
  child_count = 15 →
  senior_count = 10 →
  group_student_count = 5 →
  group_adult_count = 8 →
  group_child_count = 10 →
  group_senior_count = 9 →
  prices 6 = 6 →
  prices 8 = 8 →
  prices 4 = 4 →
  prices 7 = 7 →
  total_sales = (total_individual_sales student_count adult_count child_count senior_count prices) + (total_group_sales group_student_count group_adult_count group_child_count group_senior_count prices) →
  total_sales = 523.30 := by
  sorry

end total_ticket_sales_l52_52890


namespace minibus_children_count_l52_52578

theorem minibus_children_count
  (total_seats : ℕ)
  (seats_with_3_children : ℕ)
  (seats_with_2_children : ℕ)
  (children_per_seat_3 : ℕ)
  (children_per_seat_2 : ℕ)
  (h_seats_count : total_seats = 7)
  (h_seats_distribution : seats_with_3_children = 5 ∧ seats_with_2_children = 2)
  (h_children_per_seat : children_per_seat_3 = 3 ∧ children_per_seat_2 = 2) :
  seats_with_3_children * children_per_seat_3 + seats_with_2_children * children_per_seat_2 = 19 :=
by
  sorry

end minibus_children_count_l52_52578


namespace speed_conversion_l52_52113

theorem speed_conversion (speed_kmph : ℕ) (conversion_rate : ℚ) : (speed_kmph = 600) ∧ (conversion_rate = 0.6) → (speed_kmph * conversion_rate / 60 = 6) :=
by
  sorry

end speed_conversion_l52_52113


namespace solution_set_of_inequality_l52_52725

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l52_52725


namespace equation_of_midpoint_trajectory_l52_52811

theorem equation_of_midpoint_trajectory
  (M : ℝ × ℝ)
  (hM : M.1 ^ 2 + M.2 ^ 2 = 1)
  (N : ℝ × ℝ := (2, 0))
  (P : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) :
  (P.1 - 1) ^ 2 + P.2 ^ 2 = 1 / 4 := 
sorry

end equation_of_midpoint_trajectory_l52_52811


namespace boys_needed_to_change_ratio_l52_52683

variables (x B G : ℕ)

theorem boys_needed_to_change_ratio (h1 : B + G = 48)
                                   (h2 : B * 5 = G * 3)
                                   (h3 : (B + x) * 3 = 5 * G) :
                                   x = 32 :=
by sorry

end boys_needed_to_change_ratio_l52_52683


namespace arithmetic_series_sum_correct_l52_52303

-- Define the parameters of the arithmetic series
def a : ℤ := -53
def l : ℤ := 3
def d : ℤ := 2

-- Define the number of terms in the series
def n : ℕ := 29

-- The expected sum of the series
def expected_sum : ℤ := -725

-- Define the nth term formula
noncomputable def nth_term (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the arithmetic series
noncomputable def arithmetic_series_sum (a l : ℤ) (n : ℕ) : ℤ :=
  (n * (a + l)) / 2

-- Statement of the proof problem
theorem arithmetic_series_sum_correct :
  arithmetic_series_sum a l n = expected_sum := by
  sorry

end arithmetic_series_sum_correct_l52_52303


namespace total_trapezoid_area_l52_52030

def large_trapezoid_area (AB CD altitude_L : ℝ) : ℝ :=
  0.5 * (AB + CD) * altitude_L

def small_trapezoid_area (EF GH altitude_S : ℝ) : ℝ :=
  0.5 * (EF + GH) * altitude_S

def total_area (large_area small_area : ℝ) : ℝ :=
  large_area + small_area

theorem total_trapezoid_area :
  large_trapezoid_area 60 30 15 + small_trapezoid_area 25 10 5 = 762.5 :=
by
  -- proof goes here
  sorry

end total_trapezoid_area_l52_52030


namespace surface_area_of_sphere_l52_52094

theorem surface_area_of_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 2)
  (h4 : ∀ d, d = Real.sqrt (a^2 + b^2 + c^2)) : 
  4 * Real.pi * (d / 2)^2 = 9 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l52_52094


namespace john_hourly_rate_with_bonus_l52_52362

theorem john_hourly_rate_with_bonus:
  ∀ (daily_wage : ℝ) (work_hours : ℕ) (bonus : ℝ) (extra_hours : ℕ),
    daily_wage = 80 →
    work_hours = 8 →
    bonus = 20 →
    extra_hours = 2 →
    (daily_wage + bonus) / (work_hours + extra_hours) = 10 :=
by
  intros daily_wage work_hours bonus extra_hours
  intros h1 h2 h3 h4
  -- sorry: the proof is omitted
  sorry

end john_hourly_rate_with_bonus_l52_52362


namespace cute_2020_all_integers_cute_l52_52899

-- Definition of "cute" integer
def is_cute (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), n = a^2 + b^3 + c^3 + d^5

-- Proof problem 1: Assert that 2020 is cute
theorem cute_2020 : is_cute 2020 :=
sorry

-- Proof problem 2: Assert that every integer is cute
theorem all_integers_cute (n : ℤ) : is_cute n :=
sorry

end cute_2020_all_integers_cute_l52_52899


namespace multiplication_of_variables_l52_52021

theorem multiplication_of_variables 
  (a b c d : ℚ)
  (h1 : 3 * a + 2 * b + 4 * c + 6 * d = 48)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : 2 * c - 2 = d) :
  a * b * c * d = -58735360 / 81450625 := 
sorry

end multiplication_of_variables_l52_52021


namespace total_amount_spent_l52_52225
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end total_amount_spent_l52_52225


namespace find_number_l52_52669

theorem find_number : ∃ (x : ℤ), 45 + 3 * x = 72 ∧ x = 9 := by
  sorry

end find_number_l52_52669


namespace side_length_of_square_l52_52387

theorem side_length_of_square (s : ℝ) (h : s^2 = 6 * (4 * s)) : s = 24 :=
by sorry

end side_length_of_square_l52_52387


namespace Sam_age_l52_52567

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end Sam_age_l52_52567


namespace product_of_two_numbers_l52_52879

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l52_52879


namespace opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l52_52403

theorem opposite_neg_abs_five_minus_six : -|5 - 6| = -1 := by
  sorry

theorem opposite_of_neg_abs (h : -|5 - 6| = -1) : -(-1) = 1 := by
  sorry

theorem math_problem_proof : -(-|5 - 6|) = 1 := by
  apply opposite_of_neg_abs
  apply opposite_neg_abs_five_minus_six

end opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l52_52403


namespace train_length_calculation_l52_52772

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l52_52772


namespace find_valid_pairs_l52_52509

theorem find_valid_pairs :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 12 →
  (∃ C : ℤ, ∀ (n : ℕ), 0 < n → (a^n + b^(n+9)) % 13 = C % 13) ↔
  (a, b) = (1, 1) ∨ (a, b) = (4, 4) ∨ (a, b) = (10, 10) ∨ (a, b) = (12, 12) := 
by
  sorry

end find_valid_pairs_l52_52509


namespace melissa_games_l52_52862

noncomputable def total_points_scored := 91
noncomputable def points_per_game := 7
noncomputable def number_of_games_played := total_points_scored / points_per_game

theorem melissa_games : number_of_games_played = 13 :=
by 
  sorry

end melissa_games_l52_52862


namespace jelly_bean_problem_l52_52917

variables {p_r p_o p_y p_g : ℝ}

theorem jelly_bean_problem :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 :=
by
  intros p_r_eq p_o_eq sum_eq
  -- The proof would proceed here, but we avoid proof details
  sorry

end jelly_bean_problem_l52_52917


namespace double_acute_angle_is_less_than_180_degrees_l52_52082

theorem double_acute_angle_is_less_than_180_degrees (alpha : ℝ) (h : 0 < alpha ∧ alpha < 90) : 2 * alpha < 180 :=
sorry

end double_acute_angle_is_less_than_180_degrees_l52_52082


namespace exists_multiple_of_power_of_2_with_non_zero_digits_l52_52260

theorem exists_multiple_of_power_of_2_with_non_zero_digits (n : ℕ) (hn : n ≥ 1) :
  ∃ a : ℕ, (∀ d ∈ a.digits 10, d = 1 ∨ d = 2) ∧ 2^n ∣ a :=
by
  sorry

end exists_multiple_of_power_of_2_with_non_zero_digits_l52_52260


namespace total_money_spent_l52_52227

def cost_life_journey_cd : ℕ := 100
def cost_day_life_cd : ℕ := 50
def cost_when_rescind_cd : ℕ := 85
def number_of_cds_each : ℕ := 3

theorem total_money_spent :
  number_of_cds_each * cost_life_journey_cd +
  number_of_cds_each * cost_day_life_cd +
  number_of_cds_each * cost_when_rescind_cd = 705 :=
sorry

end total_money_spent_l52_52227


namespace probability_point_in_square_l52_52765

theorem probability_point_in_square (r : ℝ) (hr : 0 < r) :
  (∃ p : ℝ, p = 2 / Real.pi) :=
by
  sorry

end probability_point_in_square_l52_52765


namespace find_diameters_l52_52709

theorem find_diameters (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : x ≠ z) :
  x + y + z = 26 ∧ x^2 + y^2 + z^2 = 338 :=
  sorry

end find_diameters_l52_52709


namespace billing_error_l52_52162

theorem billing_error (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) 
    (h : 100 * y + x - (100 * x + y) = 2970) : y - x = 30 ∧ 10 ≤ x ∧ x ≤ 69 ∧ 40 ≤ y ∧ y ≤ 99 := 
by
  sorry

end billing_error_l52_52162


namespace marble_prism_weight_l52_52360

theorem marble_prism_weight :
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  volume * density = 86400 :=
by
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  sorry

end marble_prism_weight_l52_52360


namespace halve_second_column_l52_52499

-- Definitions of given matrices
variable (f g h i : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := ![![f, g], ![h, i]])
variable (N : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, (1/2)]])

-- Proof statement to be proved
theorem halve_second_column (hf : f ≠ 0) (hh : h ≠ 0) : N * A = ![![f, (1/2) * g], ![h, (1/2) * i]] := by
  sorry

end halve_second_column_l52_52499


namespace sum_3n_terms_l52_52080

variable {a_n : ℕ → ℝ} -- Definition of the sequence
variable {S : ℕ → ℝ} -- Definition of the sum function

-- Conditions
axiom sum_n_terms (n : ℕ) : S n = 3
axiom sum_2n_terms (n : ℕ) : S (2 * n) = 15

-- Question and correct answer
theorem sum_3n_terms (n : ℕ) : S (3 * n) = 63 := 
sorry -- Proof to be provided

end sum_3n_terms_l52_52080


namespace monotonically_increasing_interval_l52_52307

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / Real.exp 1 → (Real.log x + 1) > 0 :=
by
  intros x hx
  sorry

end monotonically_increasing_interval_l52_52307


namespace classroom_student_count_l52_52590

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end classroom_student_count_l52_52590


namespace trains_meet_in_32_seconds_l52_52749

noncomputable def train_meeting_time
  (length_train1 : ℕ)
  (length_train2 : ℕ)
  (initial_distance : ℕ)
  (speed_train1_kmph : ℕ)
  (speed_train2_kmph : ℕ)
  : ℕ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let total_distance := length_train1 + length_train2 + initial_distance
  total_distance / relative_speed

theorem trains_meet_in_32_seconds :
  train_meeting_time 400 200 200 54 36 = 32 := 
by
  sorry

end trains_meet_in_32_seconds_l52_52749


namespace min_value_l52_52512

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) := 
sorry

end min_value_l52_52512


namespace find_f_neg1_l52_52218

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_neg1 :
  (∀ x, f (-x) = -f x) →
  (∀ x, (0 < x) → f x = 2 * x * (x + 1)) →
  f (-1) = -4 := by
  intros h1 h2
  sorry

end find_f_neg1_l52_52218


namespace edwards_initial_money_l52_52926

variable (spent1 spent2 current remaining : ℕ)

def initial_money (spent1 spent2 current remaining : ℕ) : ℕ :=
  spent1 + spent2 + current

theorem edwards_initial_money :
  spent1 = 9 → spent2 = 8 → remaining = 17 →
  initial_money spent1 spent2 remaining remaining = 34 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end edwards_initial_money_l52_52926


namespace divide_milk_l52_52492

theorem divide_milk : (3 / 5 : ℚ) = 3 / 5 := by {
    sorry
}

end divide_milk_l52_52492


namespace Series_value_l52_52058

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end Series_value_l52_52058


namespace largest_prime_factor_1729_l52_52012

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l52_52012


namespace symmetric_curve_equation_l52_52711

theorem symmetric_curve_equation (y x : ℝ) :
  (y^2 = 4 * x) → (y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_equation_l52_52711


namespace present_age_of_son_l52_52284

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 25) (h2 : F + 2 = 2 * (S + 2)) : S = 23 :=
by
  sorry

end present_age_of_son_l52_52284


namespace solution_set_of_inequality_l52_52594

theorem solution_set_of_inequality :
  { x : ℝ | x > 0 ∧ x < 1 } = { x : ℝ | 1 / x > 1 } :=
by
  sorry

end solution_set_of_inequality_l52_52594


namespace X_Y_Z_sum_eq_17_l52_52388

variable {X Y Z : ℤ}

def base_ten_representation_15_fac (X Y Z : ℤ) : Prop :=
  Z = 0 ∧ (28 + X + Y) % 9 = 8 ∧ (X - Y) % 11 = 11

theorem X_Y_Z_sum_eq_17 (X Y Z : ℤ) (h : base_ten_representation_15_fac X Y Z) : X + Y + Z = 17 :=
by
  sorry

end X_Y_Z_sum_eq_17_l52_52388


namespace tangent_lines_ln_e_proof_l52_52824

noncomputable def tangent_tangent_ln_e : Prop :=
  ∀ (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) 
  (h₁_eq : x₂ = -Real.log x₁)
  (h₂_eq : Real.log x₁ - 1 = (Real.exp x₂) * (1 - x₂)),
  (2 / (x₁ - 1) + x₂ = -1)

theorem tangent_lines_ln_e_proof : tangent_tangent_ln_e :=
  sorry

end tangent_lines_ln_e_proof_l52_52824


namespace polar_bear_trout_l52_52659

/-
Question: How many buckets of trout does the polar bear eat daily?
Conditions:
  1. The polar bear eats some amount of trout and 0.4 bucket of salmon daily.
  2. The polar bear eats a total of 0.6 buckets of fish daily.
Answer: 0.2 buckets of trout daily.
-/

theorem polar_bear_trout (trout salmon total : ℝ) 
  (h1 : salmon = 0.4)
  (h2 : total = 0.6)
  (h3 : trout + salmon = total) :
  trout = 0.2 :=
by
  -- The proof will be provided here
  sorry

end polar_bear_trout_l52_52659


namespace operation_hash_12_6_l52_52931

axiom operation_hash (r s : ℝ) : ℝ

-- Conditions
axiom condition_1 : ∀ r : ℝ, operation_hash r 0 = r
axiom condition_2 : ∀ r s : ℝ, operation_hash r s = operation_hash s r
axiom condition_3 : ∀ r s : ℝ, operation_hash (r + 2) s = (operation_hash r s) + 2 * s + 2

-- Proof statement
theorem operation_hash_12_6 : operation_hash 12 6 = 168 :=
by
  sorry

end operation_hash_12_6_l52_52931


namespace total_money_spent_l52_52228

def cost_life_journey_cd : ℕ := 100
def cost_day_life_cd : ℕ := 50
def cost_when_rescind_cd : ℕ := 85
def number_of_cds_each : ℕ := 3

theorem total_money_spent :
  number_of_cds_each * cost_life_journey_cd +
  number_of_cds_each * cost_day_life_cd +
  number_of_cds_each * cost_when_rescind_cd = 705 :=
sorry

end total_money_spent_l52_52228


namespace find_y_l52_52017

theorem find_y (y : ℝ) (h : 2 * y / 3 = 12) : y = 18 :=
by
  sorry

end find_y_l52_52017


namespace min_value_expression_l52_52089

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a + (a/b^2) + b) ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l52_52089


namespace polar_to_cartesian_l52_52101

theorem polar_to_cartesian :
  ∃ (x y : ℝ), x = 2 * Real.cos (Real.pi / 6) ∧ y = 2 * Real.sin (Real.pi / 6) ∧ 
  (x, y) = (Real.sqrt 3, 1) :=
by
  use (2 * Real.cos (Real.pi / 6)), (2 * Real.sin (Real.pi / 6))
  -- The proof will show the necessary steps
  sorry

end polar_to_cartesian_l52_52101


namespace cone_height_l52_52735

theorem cone_height
  (V1 V2 V : ℝ)
  (h1 h2 : ℝ)
  (fact1 : h1 = 10)
  (fact2 : h2 = 2)
  (h : ∀ m : ℝ, V1 = V * (10 ^ 3) / (m ^ 3) ∧ V2 = V * ((m - 2) ^ 3) / (m ^ 3))
  (equal_volumes : V1 + V2 = V) :
  (∃ m : ℝ, m = 13.897) :=
by
  sorry

end cone_height_l52_52735


namespace factorize_expression_l52_52067

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end factorize_expression_l52_52067


namespace product_B_sampling_l52_52721

theorem product_B_sampling (a : ℕ) (h_seq : a > 0) :
  let A := a
  let B := 2 * a
  let C := 4 * a
  let total := A + B + C
  total = 7 * a →
  let total_drawn := 140
  B / total * total_drawn = 40 :=
by sorry

end product_B_sampling_l52_52721


namespace smallest_odd_integer_of_set_l52_52400

theorem smallest_odd_integer_of_set (S : Set Int) 
  (h1 : ∃ m : Int, m ∈ S ∧ m = 149)
  (h2 : ∃ n : Int, n ∈ S ∧ n = 159)
  (h3 : ∀ a b : Int, a ∈ S → b ∈ S → a ≠ b → (a - b) % 2 = 0) : 
  ∃ s : Int, s ∈ S ∧ s = 137 :=
by sorry

end smallest_odd_integer_of_set_l52_52400


namespace find_sam_age_l52_52572

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l52_52572


namespace arithmetic_mean_of_fractions_l52_52193
-- Import the Mathlib library to use fractional arithmetic

-- Define the problem in Lean
theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 :=
by
  let a : ℚ := 3 / 8
  let b : ℚ := 5 / 9
  have := (a + b) / 2 = 67 / 144
  sorry

end arithmetic_mean_of_fractions_l52_52193


namespace initial_food_supplies_l52_52174

theorem initial_food_supplies (x : ℝ) 
  (h1 : (3 / 5) * x - (3 / 5) * ((3 / 5) * x) = 96) : x = 400 :=
by
  sorry

end initial_food_supplies_l52_52174


namespace right_pan_at_least_left_pan_l52_52234

theorem right_pan_at_least_left_pan (weights : ℕ → ℕ) {left_pan right_pan : Finset ℕ}
  (h_left_distinct : ∀ x ∈ left_pan, ∃ n : ℕ, x = 2^n) 
  (h_eq : left_pan.sum weights = right_pan.sum weights) :
  left_pan.card ≤ right_pan.card :=
sorry

end right_pan_at_least_left_pan_l52_52234


namespace flour_needed_l52_52638

theorem flour_needed (flour_per_40_cookies : ℝ) (cookies : ℕ) (desired_cookies : ℕ) (flour_needed : ℝ) 
  (h1 : flour_per_40_cookies = 3) (h2 : cookies = 40) (h3 : desired_cookies = 100) :
  flour_needed = 7.5 :=
by
  sorry

end flour_needed_l52_52638


namespace simplify_fraction_l52_52869

theorem simplify_fraction (k : ℤ) : 
  (∃ (a b : ℤ), a = 1 ∧ b = 2 ∧ (6 * k + 12) / 6 = a * k + b) → (1 / 2 : ℚ) = (1 / 2 : ℚ) := 
by
  intro h
  sorry

end simplify_fraction_l52_52869


namespace min_value_f_when_a_eq_one_range_of_a_for_inequality_l52_52856

noncomputable def f (x a : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Question 1: When a = 1, find the minimum value of the function f(x)
theorem min_value_f_when_a_eq_one : ∃ x : ℝ, ∀ y : ℝ, f y 1 ≥ f x 1 ∧ f x 1 = 4 :=
by
  sorry

-- Question 2: For which values of a does f(x) ≥ 4/a + 1 hold for all real numbers x
theorem range_of_a_for_inequality : (∀ x : ℝ, f x a ≥ 4 / a + 1) ↔ (a < 0 ∨ a = 2) :=
by
  sorry

end min_value_f_when_a_eq_one_range_of_a_for_inequality_l52_52856


namespace compare_neg_two_cubed_l52_52182

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end compare_neg_two_cubed_l52_52182


namespace difference_english_math_l52_52968

/-- There are 30 students who pass in English and 20 students who pass in Math. -/
axiom passes_in_english : ℕ
axiom passes_in_math : ℕ
axiom both_subjects : ℕ
axiom only_english : ℕ
axiom only_math : ℕ

/-- Definitions based on the problem conditions -/
axiom number_passes_in_english : only_english + both_subjects = 30
axiom number_passes_in_math : only_math + both_subjects = 20

/-- The difference between the number of students who pass only in English
    and the number of students who pass only in Math is 10. -/
theorem difference_english_math : only_english - only_math = 10 :=
by
  sorry

end difference_english_math_l52_52968


namespace arccos_sin_eq_pi_div_two_sub_1_72_l52_52052

theorem arccos_sin_eq_pi_div_two_sub_1_72 :
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 :=
sorry

end arccos_sin_eq_pi_div_two_sub_1_72_l52_52052


namespace john_pays_after_discount_l52_52239

theorem john_pays_after_discount :
  ∀ (num_toys : ℕ) (cost_per_toy : ℕ) (discount_rate : ℚ),
  num_toys = 5 → cost_per_toy = 3 → discount_rate = 0.20 →
  let total_cost := num_toys * cost_per_toy in
  let discount := discount_rate * ↑total_cost in
  let amount_paid := total_cost - discount in
  amount_paid = 12 :=
by
  intros num_toys cost_per_toy discount_rate hnum_toys hcost_per_toy hdiscount_rate
  rw [hnum_toys, hcost_per_toy, hdiscount_rate]
  let total_cost := num_toys * cost_per_toy
  let discount := discount_rate * ↑total_cost
  let amount_paid := total_cost - discount
  have htotal_cost : total_cost = 15 := by norm_num
  have hdiscount : discount = 3 := by norm_num
  have hamount_paid : amount_paid = 12 := by norm_num
  exact hamount_paid

end john_pays_after_discount_l52_52239


namespace unique_outfits_count_l52_52744

theorem unique_outfits_count (s : Fin 5) (p : Fin 6) (restricted_pairings : (Fin 1 × Fin 2) → Prop) 
  (r : restricted_pairings (0, 0) ∧ restricted_pairings (0, 1)) : ∃ n, n = 28 ∧ 
  ∃ (outfits : Fin 5 → Fin 6 → Prop), 
    (∀ s p, outfits s p) ∧ 
    (∀ p, ¬outfits 0 p ↔ p = 0 ∨ p = 1) := by
  sorry

end unique_outfits_count_l52_52744


namespace maximum_side_length_l52_52125

theorem maximum_side_length 
    (D E F : ℝ) 
    (a b c : ℝ) 
    (h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1)
    (h_a : a = 12)
    (h_perimeter : a + b + c = 40) : 
    ∃ max_side : ℝ, max_side = 7 + Real.sqrt 23 / 2 :=
by
  sorry

end maximum_side_length_l52_52125


namespace remainder_when_divided_by_6_l52_52823

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 6 = 4 :=
  sorry

end remainder_when_divided_by_6_l52_52823


namespace inequality_solution_l52_52383

theorem inequality_solution (x : ℝ) : (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) ↔ (-1/2 ≤ x ∧ x < 1) :=
by
  sorry

end inequality_solution_l52_52383


namespace average_velocity_of_particle_l52_52760

theorem average_velocity_of_particle (t : ℝ) (s : ℝ → ℝ) (h_s : ∀ t, s t = t^2 + 1) :
  (s 2 - s 1) / (2 - 1) = 3 :=
by {
  sorry
}

end average_velocity_of_particle_l52_52760


namespace greatest_whole_number_lt_100_with_odd_factors_l52_52255

theorem greatest_whole_number_lt_100_with_odd_factors :
  ∃ n, n < 100 ∧ (∃ p : ℕ, n = p * p) ∧ 
    ∀ m, (m < 100 ∧ (∃ q : ℕ, m = q * q)) → m ≤ n :=
sorry

end greatest_whole_number_lt_100_with_odd_factors_l52_52255


namespace ink_percentage_left_l52_52146

def area_of_square (side: ℕ) := side * side
def area_of_rectangle (length: ℕ) (width: ℕ) := length * width
def total_area_marker_can_paint (num_squares: ℕ) (square_side: ℕ) :=
  num_squares * area_of_square square_side
def total_area_colored (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ) :=
  num_rectangles * area_of_rectangle rect_length rect_width

def fraction_of_ink_used (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  (total_area_colored num_rectangles rect_length rect_width : ℚ)
    / (total_area_marker_can_paint num_squares square_side : ℚ)

def percentage_ink_left (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  100 * (1 - fraction_of_ink_used num_rectangles rect_length rect_width num_squares square_side)

theorem ink_percentage_left :
  percentage_ink_left 2 6 2 3 4 = 50 := by
  sorry

end ink_percentage_left_l52_52146


namespace part1_part2_l52_52335

noncomputable def f (a : ℝ) (a_pos : a > 1) (x : ℝ) : ℝ :=
  a^x + (x - 2) / (x + 1)

-- Statement for part 1
theorem part1 (a : ℝ) (a_pos : a > 1) : ∀ x : ℝ, -1 < x → f a a_pos x ≤ f a a_pos (x + ε) → 0 < ε := sorry

-- Statement for part 2
theorem part2 (a : ℝ) (a_pos : a > 1) : ¬ ∃ x : ℝ, x < 0 ∧ f a a_pos x = 0 := sorry

end part1_part2_l52_52335


namespace Martha_points_l52_52860

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end Martha_points_l52_52860


namespace sad_girls_count_l52_52256

variables (total_children happy_children sad_children neither_children : ℕ)
variables (total_boys total_girls happy_boys sad_children total_sad_boys : ℕ)

theorem sad_girls_count :
  total_children = 60 ∧ 
  happy_children = 30 ∧ 
  sad_children = 10 ∧ 
  neither_children = 20 ∧ 
  total_boys = 17 ∧ 
  total_girls = 43 ∧ 
  happy_boys = 6 ∧ 
  neither_boys = 5 ∧ 
  sad_children = total_sad_boys + (sad_children - total_sad_boys) ∧ 
  total_sad_boys = total_boys - happy_boys - neither_boys → 
  (sad_children - total_sad_boys = 4) := 
by
  intros h
  sorry

end sad_girls_count_l52_52256


namespace basketball_player_height_l52_52886

noncomputable def player_height (H : ℝ) : Prop :=
  let reach := 22 / 12
  let jump := 32 / 12
  let total_rim_height := 10 + (6 / 12)
  H + reach + jump = total_rim_height

theorem basketball_player_height : ∃ H : ℝ, player_height H → H = 6 :=
by
  use 6
  sorry

end basketball_player_height_l52_52886


namespace tangent_line_problem_l52_52826

theorem tangent_line_problem 
  (x1 x2 : ℝ)
  (h1 : (1 / x1) = Real.exp x2)
  (h2 : Real.log x1 - 1 = Real.exp x2 * (1 - x2)) :
  (2 / (x1 - 1) + x2 = -1) :=
by 
  sorry

end tangent_line_problem_l52_52826


namespace atomic_number_order_l52_52196

-- Define that elements A, B, C, D, and E are in the same period
variable (A B C D E : Type)

-- Define conditions based on the problem
def highest_valence_oxide_basic (x : Type) : Prop := sorry
def basicity_greater (x y : Type) : Prop := sorry
def gaseous_hydride_stability (x y : Type) : Prop := sorry
def smallest_ionic_radius (x : Type) : Prop := sorry

-- Assume conditions given in the problem
axiom basic_oxides : highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
axiom basicity_order : basicity_greater B A
axiom hydride_stabilities : gaseous_hydride_stability C D
axiom smallest_radius : smallest_ionic_radius E

-- Prove that the order of atomic numbers from smallest to largest is B, A, E, D, C
theorem atomic_number_order :
  ∃ (A B C D E : Type), highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
  ∧ basicity_greater B A ∧ gaseous_hydride_stability C D ∧ smallest_ionic_radius E
  ↔ B = B ∧ A = A ∧ E = E ∧ D = D ∧ C = C := sorry

end atomic_number_order_l52_52196


namespace max_value_a4_b2_c2_d2_l52_52544

theorem max_value_a4_b2_c2_d2
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  a^4 + b^2 + c^2 + d^2 ≤ 100 :=
sorry

end max_value_a4_b2_c2_d2_l52_52544


namespace jack_shoes_time_l52_52688

theorem jack_shoes_time (J : ℝ) (h : J + 2 * (J + 3) = 18) : J = 4 :=
by
  sorry

end jack_shoes_time_l52_52688


namespace integer_powers_of_reciprocal_sum_l52_52548

variable (x: ℝ)

theorem integer_powers_of_reciprocal_sum (hx : x ≠ 0) (hx_int : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ k : ℤ, x^n + 1/x^n = k :=
by
  sorry

end integer_powers_of_reciprocal_sum_l52_52548


namespace find_second_number_l52_52597

theorem find_second_number 
  (k : ℕ)
  (h_k_is_1 : k = 1)
  (h_div_1657 : ∃ q1 : ℕ, 1657 = k * q1 + 10)
  (h_div_x : ∃ q2 : ℕ, ∀ x : ℕ, x = k * q2 + 7 → x = 1655) 
: ∃ x : ℕ, x = 1655 :=
by
  sorry

end find_second_number_l52_52597


namespace range_of_d_l52_52093

theorem range_of_d (d : ℝ) : (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) ↔ d ≥ 1 :=
sorry

end range_of_d_l52_52093


namespace total_tickets_sold_l52_52298

-- Define the conditions
variables (V G : ℕ)

-- Condition 1: Total revenue from VIP and general admission
def total_revenue_eq : Prop := 40 * V + 15 * G = 7500

-- Condition 2: There are 212 fewer VIP tickets than general admission
def vip_tickets_eq : Prop := V = G - 212

-- Main statement to prove: the total number of tickets sold
theorem total_tickets_sold (h1 : total_revenue_eq V G) (h2 : vip_tickets_eq V G) : V + G = 370 :=
sorry

end total_tickets_sold_l52_52298


namespace min_increase_air_quality_days_l52_52242

theorem min_increase_air_quality_days {days_in_year : ℕ} (last_year_ratio next_year_ratio : ℝ) (good_air_days : ℕ) :
  days_in_year = 365 → last_year_ratio = 0.6 → next_year_ratio > 0.7 →
  (good_air_days / days_in_year < last_year_ratio → ∀ n: ℕ, good_air_days + n ≥ 37) :=
by
  intros hdays_in_year hlast_year_ratio hnext_year_ratio h_good_air_days
  sorry

end min_increase_air_quality_days_l52_52242


namespace discriminant_zero_no_harmonic_progression_l52_52345

theorem discriminant_zero_no_harmonic_progression (a b c : ℝ) 
    (h_disc : b^2 = 24 * a * c) : 
    ¬ (2 * (1 / b) = (1 / a) + (1 / c)) := 
sorry

end discriminant_zero_no_harmonic_progression_l52_52345


namespace correlation_coefficient_interpretation_l52_52020

-- Definitions and problem statement

/-- 
Theorem: Correct interpretation of the correlation coefficient r.
Given r in (-1, 1):
The closer |r| is to zero, the weaker the correlation between the two variables.
-/
theorem correlation_coefficient_interpretation (r : ℝ) (h : -1 < r ∧ r < 1) :
  (r > 0 -> false) ∧ (r > 1 -> false) ∧ (0 < r -> false) ∧ (|r| -> Prop) :=
sorry

end correlation_coefficient_interpretation_l52_52020


namespace find_p_probability_of_match_ending_after_4_games_l52_52292

variables (p : ℚ)

-- Conditions translated to Lean definitions
def probability_first_game_win : ℚ := 1 / 2

def probability_consecutive_games_win : ℚ := 5 / 16

-- Definitions based on conditions
def prob_second_game_win_if_won_first : ℚ := (1 + p) / 2

def prob_winning_consecutive_games (prob_first_game : ℚ) (prob_second_game_if_won_first : ℚ) : ℚ :=
prob_first_game * prob_second_game_if_won_first

-- Main Theorem Statements to be proved
theorem find_p 
    (h_eq : prob_winning_consecutive_games probability_first_game_win (prob_second_game_win_if_won_first p) = probability_consecutive_games_win) :
    p = 1 / 4 :=
sorry

-- Given p = 1/4, probabilities for each scenario the match ends after 4 games
def prob_scenario1 : ℚ := (1 / 2) * ((1 + 1/4) / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2)
def prob_scenario2 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2)
def prob_scenario3 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2) * ((1 + 1/4) / 2)

def total_probability_ending_in_4_games : ℚ :=
2 * (prob_scenario1 + prob_scenario2 + prob_scenario3)

theorem probability_of_match_ending_after_4_games (hp : p = 1 / 4) :
    total_probability_ending_in_4_games = 165 / 512 :=
sorry

end find_p_probability_of_match_ending_after_4_games_l52_52292


namespace find_num_white_balls_l52_52233

theorem find_num_white_balls
  (W : ℕ)
  (total_balls : ℕ := 15 + W)
  (prob_black : ℚ := 7 / total_balls)
  (given_prob : ℚ := 0.38095238095238093) :
  prob_black = given_prob → W = 3 :=
by
  intro h
  sorry

end find_num_white_balls_l52_52233


namespace negation_of_exists_l52_52716

open Set Real

theorem negation_of_exists (x : Real) :
  ¬ (∃ x ∈ Icc 0 1, x^3 + x^2 > 1) ↔ ∀ x ∈ Icc 0 1, x^3 + x^2 ≤ 1 := 
by sorry

end negation_of_exists_l52_52716


namespace hexagon_side_length_l52_52717

-- Define the conditions for the side length of a hexagon where the area equals the perimeter
theorem hexagon_side_length (s : ℝ) (h1 : (3 * Real.sqrt 3 / 2) * s^2 = 6 * s) :
  s = 4 * Real.sqrt 3 / 3 :=
sorry

end hexagon_side_length_l52_52717


namespace find_second_discount_l52_52588

theorem find_second_discount 
    (list_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (h₁ : list_price = 65)
    (h₂ : final_price = 57.33)
    (h₃ : first_discount = 0.10)
    (h₄ : (list_price - (first_discount * list_price)) = 58.5)
    (h₅ : final_price = 58.5 - (second_discount * 58.5)) :
    second_discount = 0.02 := 
by
  sorry

end find_second_discount_l52_52588


namespace product_of_two_numbers_l52_52880

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end product_of_two_numbers_l52_52880


namespace real_solutions_l52_52072

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end real_solutions_l52_52072


namespace largest_prime_factor_of_1729_l52_52006

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p :=
by
  -- 1729 can be factored as 7 * 11 * 23
  have h1 : 1729 = 7 * 247 := by norm_num
  have h2 : 247 = 11 * 23 := by norm_num
  -- All factors need to be prime
  have p7 : prime 7 := by norm_num
  have p11 : prime 11 := by norm_num
  have p23 : prime 23 := by norm_num
  -- Combining these results
  use 23
  split
  {
    exact p23
  }
  split
  {
    -- Showing 23 divides 1729
    rw h1
    rw h2
    exact dvd_mul_of_dvd_right (dvd_mul_left 11 23) 7
  }
  {
    -- Show 23 is the largest prime factor
    intros q hprime hdiv
    rw h1 at hdiv
    rw h2 at hdiv
    cases hdiv
    {
      -- Case for q = 7
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    cases hdiv
    {
      -- Case for q = 11
      exfalso
      have h := prime.ne_one hprime
      norm_num at h
    }
    {
      -- Case for q = 23
      exfalso
      exact prime.ne_one hprime
    }
  }
    sorry

end largest_prime_factor_of_1729_l52_52006


namespace trigonometric_solution_l52_52382

theorem trigonometric_solution (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, (n % 2 = 0 → (x = k * real.pi)) ∨ 
          (n % 2 = 1 → (x = 2 * k * real.pi ∨ x = 2 * k * real.pi - real.pi / 2)) 
:=
sorry

end trigonometric_solution_l52_52382


namespace binary_to_decimal_l52_52655

-- Define the binary number 10011_2
def binary_10011 : ℕ := bit0 (bit1 (bit1 (bit0 (bit1 0))))

-- Define the expected decimal value
def decimal_19 : ℕ := 19

-- State the theorem to convert binary 10011 to decimal
theorem binary_to_decimal :
  binary_10011 = decimal_19 :=
sorry

end binary_to_decimal_l52_52655


namespace solve_ab_c_eq_l52_52197

theorem solve_ab_c_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_eq : 11^a + 3^b = c^2) :
  a = 4 ∧ b = 5 ∧ c = 122 :=
by
  sorry

end solve_ab_c_eq_l52_52197


namespace calc_value_l52_52469

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end calc_value_l52_52469


namespace find_three_numbers_l52_52587

theorem find_three_numbers (x y z : ℝ) 
  (h1 : x - y = 12) 
  (h2 : (x + y) / 4 = 7) 
  (h3 : z = 2 * y) 
  (h4 : x + z = 24) : 
  x = 20 ∧ y = 8 ∧ z = 16 := by
  sorry

end find_three_numbers_l52_52587


namespace cosine_value_l52_52674

theorem cosine_value (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos (α + Real.pi / 6) = -1 / 3 :=
by
  sorry

end cosine_value_l52_52674


namespace mod_exp_l52_52608

theorem mod_exp (n : ℕ) : (5^303) % 11 = 4 :=
  by sorry

end mod_exp_l52_52608


namespace probability_of_both_selected_l52_52748

theorem probability_of_both_selected :
  let pX := 1 / 5
  let pY := 2 / 7
  (pX * pY) = 2 / 35 :=
by
  let pX := 1 / 5
  let pY := 2 / 7
  show (pX * pY) = 2 / 35
  sorry

end probability_of_both_selected_l52_52748


namespace perimeter_of_triangle_l52_52626

theorem perimeter_of_triangle (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 7) (h2 : num_sides = 3) : 
  num_sides * side_length = 21 :=
by
  sorry

end perimeter_of_triangle_l52_52626


namespace product_of_xy_l52_52960

-- Define the problem conditions
variables (x y : ℝ)
-- Define the condition that |x-3| and |y+1| are opposite numbers
def opposite_abs_values := |x - 3| = - |y + 1|

-- State the theorem
theorem product_of_xy (h : opposite_abs_values x y) : x * y = -3 :=
sorry -- Proof is omitted

end product_of_xy_l52_52960


namespace george_painting_problem_l52_52323

theorem george_painting_problem (n : ℕ) (hn : n = 9) :
  let choices := nat.choose (n - 1) 1 in
  choices = 8 :=
by
  -- Import necessary combinatorial definitions and theorems
  have h : n - 1 = 8 := by linarith [hn],
  rw h,
  show nat.choose 8 1 = 8,
  apply nat.choose_one_right,
  sorry

end george_painting_problem_l52_52323


namespace num_ordered_triples_l52_52450

theorem num_ordered_triples : 
  {n : ℕ // ∃ (a b c : ℤ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = (2 * (a * b + b * c + c * a)) / 3 ∧ n = 3} :=
sorry

end num_ordered_triples_l52_52450


namespace line_through_intersections_l52_52392

-- Conditions
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem statement
theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → x - y - 3 = 0 :=
by
  sorry

end line_through_intersections_l52_52392


namespace complex_point_location_l52_52328

open Complex

noncomputable def quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "on the axis"

theorem complex_point_location :
  ∀ (z : ℂ), (z - 2 * I) * (1 + I) = abs (1 - sqrt 3 * I) → quadrant z = "first quadrant" :=
by
  intros z h
  have h₁ : abs (1 - sqrt 3 * I) = 2 := by sorry
  have h₂ : (z - 2 * I) * (1 + I) = 2 := by rw [h, h₁]
  have h₃ : z = 1 + I := by sorry
  rw [h₃]
  unfold quadrant
  simp

end complex_point_location_l52_52328


namespace odd_function_ln_negx_l52_52546

theorem odd_function_ln_negx (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_positive : ∀ x, x > 0 → f x = Real.log x) :
  ∀ x, x < 0 → f x = -Real.log (-x) :=
by 
  intros x hx_neg
  have hx_pos : -x > 0 := by linarith
  rw [← h_positive (-x) hx_pos, h_odd x]
  sorry

end odd_function_ln_negx_l52_52546


namespace trapezoid_condition_l52_52564

-- Define the problem statement in Lean
theorem trapezoid_condition (A B C D M N : Point) (h1 : M = midpoint A B) (h2 : N = midpoint C D)
  (h3 : area (quadrilateral A M N B) = area (quadrilateral M N C D)) : 
  parallel (line A D) (line B C) :=
sorry

end trapezoid_condition_l52_52564


namespace tuesday_snow_correct_l52_52102

-- Define the snowfall amounts as given in the conditions
def monday_snow : ℝ := 0.32
def total_snow : ℝ := 0.53

-- Define the amount of snow on Tuesday as per the question to be proved
def tuesday_snow : ℝ := total_snow - monday_snow

-- State the theorem to prove that the snowfall on Tuesday is 0.21 inches
theorem tuesday_snow_correct : tuesday_snow = 0.21 := by
  -- Proof skipped with sorry
  sorry

end tuesday_snow_correct_l52_52102


namespace find_smallest_value_of_sum_of_squares_l52_52096
noncomputable def smallest_value (x y z : ℚ) := x^2 + y^2 + z^2

theorem find_smallest_value_of_sum_of_squares :
  ∃ (x y z : ℚ), (x + 4) * (y - 4) = 0 ∧ 3 * z - 2 * y = 5 ∧ smallest_value x y z = 457 / 9 :=
by
  sorry

end find_smallest_value_of_sum_of_squares_l52_52096


namespace domain_ln_x_plus_one_l52_52801

theorem domain_ln_x_plus_one :
  ∀ (x : ℝ), ∃ (y : ℝ), y = Real.log (x + 1) ↔ x > -1 :=
by sorry

end domain_ln_x_plus_one_l52_52801


namespace graph_paper_squares_below_line_l52_52393

theorem graph_paper_squares_below_line
  (h : ∀ (x y : ℕ), 12 * x + 247 * y = 2976)
  (square_size : ℕ) 
  (xs : ℕ) (ys : ℕ)
  (line_eq : ∀ (x y : ℕ), y = 247 * x / 12)
  (n_squares : ℕ) :
  n_squares = 1358
  := by
    sorry

end graph_paper_squares_below_line_l52_52393


namespace journey_time_proof_l52_52259

noncomputable def journey_time_on_wednesday (d s x : ℝ) : ℝ :=
  d / s

theorem journey_time_proof (d s x : ℝ) (usual_speed_nonzero : s ≠ 0) :
  (journey_time_on_wednesday d s x) = 11 * x :=
by
  have thursday_speed : ℝ := 1.1 * s
  have thursday_time : ℝ := d / thursday_speed
  have time_diff : ℝ := (d / s) - thursday_time
  have reduced_time_eq_x : time_diff = x := by sorry
  have journey_time_eq : (d / s) = 11 * x := by sorry
  exact journey_time_eq

end journey_time_proof_l52_52259


namespace outermost_diameter_l52_52835

def radius_of_fountain := 6 -- derived from the information that 12/2 = 6
def width_of_garden := 9
def width_of_inner_walking_path := 3
def width_of_outer_walking_path := 7

theorem outermost_diameter :
  2 * (radius_of_fountain + width_of_garden + width_of_inner_walking_path + width_of_outer_walking_path) = 50 :=
by
  sorry

end outermost_diameter_l52_52835


namespace per_can_price_difference_cents_l52_52622

   theorem per_can_price_difference_cents :
     let bulk_warehouse_price_per_case := 12.0
     let bulk_warehouse_cans_per_case := 48
     let bulk_warehouse_discount := 0.10
     let local_store_price_per_case := 6.0
     let local_store_cans_per_case := 12
     let local_store_promotion_factor := 1.5 -- represents the effect of the promotion (3 cases for the price of 2.5 cases)
     let bulk_warehouse_price_per_can := (bulk_warehouse_price_per_case * (1 - bulk_warehouse_discount)) / bulk_warehouse_cans_per_case
     let local_store_price_per_can := (local_store_price_per_case * local_store_promotion_factor) / (local_store_cans_per_case * 3)
     let price_difference_cents := (local_store_price_per_can - bulk_warehouse_price_per_can) * 100
     price_difference_cents = 19.17 :=
   by
     sorry
   
end per_can_price_difference_cents_l52_52622


namespace hyperbola_condition_l52_52537

variables (a b : ℝ)
def e1 : (ℝ × ℝ) := (2, 1)
def e2 : (ℝ × ℝ) := (2, -1)

theorem hyperbola_condition (h1 : e1 = (2, 1)) (h2 : e2 = (2, -1)) (p : ℝ × ℝ)
  (h3 : p = (2 * a + 2 * b, a - b)) :
  4 * a * b = 1 :=
sorry

end hyperbola_condition_l52_52537


namespace lcm_9_16_21_eq_1008_l52_52278

theorem lcm_9_16_21_eq_1008 : Nat.lcm (Nat.lcm 9 16) 21 = 1008 := by
  sorry

end lcm_9_16_21_eq_1008_l52_52278


namespace total_onions_l52_52380

theorem total_onions (sara sally fred amy matthew : Nat) 
  (hs : sara = 40) (hl : sally = 55) 
  (hf : fred = 90) (ha : amy = 25) 
  (hm : matthew = 75) :
  sara + sally + fred + amy + matthew = 285 := 
by
  sorry

end total_onions_l52_52380


namespace rearrangement_count_correct_l52_52805

def original_number := "1234567890"

def is_valid_rearrangement (n : String) : Prop :=
  n.length = 10 ∧ n.front ≠ '0'
  
def count_rearrangements (n : String) : ℕ :=
  if n = original_number 
  then 232
  else 0

theorem rearrangement_count_correct :
  count_rearrangements original_number = 232 :=
sorry


end rearrangement_count_correct_l52_52805


namespace binom_7_2_eq_21_l52_52483

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n k := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem binom_7_2_eq_21 : binom 7 2 = 21 := by
  sorry

end binom_7_2_eq_21_l52_52483


namespace real_roots_of_cubic_equation_l52_52192

theorem real_roots_of_cubic_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (x^3 - 2 * x + 1)^2 = 9) ∧ S.card = 2 := 
by
  sorry

end real_roots_of_cubic_equation_l52_52192


namespace triangle_non_existent_l52_52222

theorem triangle_non_existent (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (tangent_condition : (c^2) = 2 * (a^2) + 2 * (b^2)) : False := by
  sorry

end triangle_non_existent_l52_52222


namespace h_at_3_l52_52845

noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := Real.sqrt (f x) - 3
noncomputable def h (x : ℝ) := g (f x)

theorem h_at_3 : h 3 = Real.sqrt 43 - 3 := by
  sorry

end h_at_3_l52_52845


namespace arithmetic_result_l52_52804

/-- Define the constants involved in the arithmetic operation. -/
def a : ℕ := 999999999999
def b : ℕ := 888888888888
def c : ℕ := 111111111111

/-- The theorem stating that the given arithmetic operation results in the expected answer. -/
theorem arithmetic_result :
  a - b + c = 222222222222 :=
by
  sorry

end arithmetic_result_l52_52804


namespace harmonic_mean_of_1_3_1_div_2_l52_52902

noncomputable def harmonicMean (a b c : ℝ) : ℝ :=
  let reciprocals := [1 / a, 1 / b, 1 / c]
  (reciprocals.sum) / reciprocals.length

theorem harmonic_mean_of_1_3_1_div_2 : harmonicMean 1 3 (1 / 2) = 9 / 10 :=
  sorry

end harmonic_mean_of_1_3_1_div_2_l52_52902


namespace simplify_expression_l52_52653

theorem simplify_expression (x : ℝ) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) :=
sorry

end simplify_expression_l52_52653


namespace inequality_proof_l52_52549

variables {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (min (a * b) (b * c)) (c * a) ≥ 1) :
  (↑((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) ^ (1 / 3 : ℝ)) ≤ ((a + b + c) / 3) ^ 2 + 1 :=
by
  sorry

end inequality_proof_l52_52549


namespace floor_double_l52_52868

theorem floor_double (a : ℝ) (h : 0 < a) : 
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ :=
sorry

end floor_double_l52_52868


namespace unique_solution_condition_l52_52522

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l52_52522


namespace katie_earnings_l52_52982

def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

theorem katie_earnings : bead_necklaces + gemstone_necklaces * cost_per_necklace = 21 := 
by
  sorry

end katie_earnings_l52_52982


namespace journey_total_distance_l52_52175

theorem journey_total_distance (D : ℝ) 
  (h1 : (D / 3) / 21 + (D / 3) / 14 + (D / 3) / 6 = 12) : 
  D = 126 :=
sorry

end journey_total_distance_l52_52175


namespace find_a_of_line_slope_l52_52676

theorem find_a_of_line_slope (a : ℝ) (h1 : a > 0)
  (h2 : ∃ (b : ℝ), (a, 5) = (b * 1, b * 2) ∧ (2, a) = (b * 1, 2 * b) ∧ b = 1) 
  : a = 3 := 
sorry

end find_a_of_line_slope_l52_52676


namespace math_problem_l52_52852

def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 2 * x + 5

theorem math_problem : f (g 4) - g (f 4) = 129 := by
  sorry

end math_problem_l52_52852


namespace remainder_13_pow_2000_mod_1000_l52_52607

theorem remainder_13_pow_2000_mod_1000 :
  (13^2000) % 1000 = 1 := 
by 
  sorry

end remainder_13_pow_2000_mod_1000_l52_52607


namespace area_of_rectangle_l52_52762

theorem area_of_rectangle (P : ℝ) (w : ℝ) (h : ℝ) (A : ℝ) 
  (hP : P = 28) 
  (hw : w = 6) 
  (hP_formula : P = 2 * (h + w)) 
  (hA_formula : A = h * w) : 
  A = 48 :=
by
  sorry

end area_of_rectangle_l52_52762


namespace find_k_l52_52678

theorem find_k (k : ℝ) (x₁ x₂ : ℝ) (h_distinct_roots : (2*k + 3)^2 - 4*k^2 > 0)
  (h_roots : ∀ (x : ℝ), x^2 + (2*k + 3)*x + k^2 = 0 ↔ x = x₁ ∨ x = x₂)
  (h_reciprocal_sum : 1/x₁ + 1/x₂ = -1) : k = 3 :=
by
  sorry

end find_k_l52_52678


namespace interest_difference_l52_52857

noncomputable def annual_amount (P r t : ℝ) : ℝ :=
P * (1 + r)^t

noncomputable def monthly_amount (P r n t : ℝ) : ℝ :=
P * (1 + r / n)^(n * t)

theorem interest_difference
  (P : ℝ)
  (r : ℝ)
  (n : ℕ)
  (t : ℝ)
  (annual_compounded : annual_amount P r t = 8000 * (1 + 0.10)^3)
  (monthly_compounded : monthly_amount P r 12 3 = 8000 * (1 + 0.10 / 12) ^ (12 * 3)) :
  (monthly_amount P r 12 t - annual_amount P r t) = 142.80 := 
sorry

end interest_difference_l52_52857


namespace vector_dot_product_l52_52950

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Vector addition and scalar multiplication
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematical statement to prove
theorem vector_dot_product : dot_product (vec_add (scalar_mul 2 vec_a) vec_b) vec_a = 6 :=
by
  -- Sorry is used to skip the proof; it's just a placeholder.
  sorry

end vector_dot_product_l52_52950


namespace range_of_m_for_two_distinct_zeros_l52_52346

noncomputable def quadratic_discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem range_of_m_for_two_distinct_zeros :
  ∀ (m : ℝ), quadratic_discriminant 1 (2*m) (m+2) > 0 ↔ (m < -1 ∨ m > 2) :=
begin
  intro m,
  rw [quadratic_discriminant, pow_two, mul_assoc, mul_comm],
  apply (lt_or_gt_of_ne (ne_of_gt (sub_pos_of_lt (by sorry)))).symm,
end

end range_of_m_for_two_distinct_zeros_l52_52346


namespace sin_cos_sum_identity_l52_52785

noncomputable def trigonometric_identity (x y z w : ℝ) := 
  (Real.sin x * Real.cos y + Real.sin z * Real.cos w) = Real.sqrt 2 / 2

theorem sin_cos_sum_identity :
  trigonometric_identity 347 148 77 58 :=
by sorry

end sin_cos_sum_identity_l52_52785


namespace largest_prime_factor_of_1729_l52_52001

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, nat.prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l52_52001


namespace highway_length_l52_52274

theorem highway_length 
  (speed_car1 speed_car2 : ℕ) (time : ℕ)
  (h_speed_car1 : speed_car1 = 54)
  (h_speed_car2 : speed_car2 = 57)
  (h_time : time = 3) : 
  speed_car1 * time + speed_car2 * time = 333 := by
  sorry

end highway_length_l52_52274


namespace projectile_reaches_height_at_first_l52_52129

noncomputable def reach_height (t : ℝ) : ℝ :=
-16 * t^2 + 80 * t

theorem projectile_reaches_height_at_first (t : ℝ) :
  reach_height t = 36 → t = 0.5 :=
by
  -- The proof can be provided here
  sorry

end projectile_reaches_height_at_first_l52_52129


namespace sphere_radius_l52_52580

theorem sphere_radius (r_A r_B : ℝ) (h₁ : r_A = 40) (h₂ : (4 * π * r_A^2) / (4 * π * r_B^2) = 16) : r_B = 20 :=
  sorry

end sphere_radius_l52_52580


namespace original_number_of_men_l52_52022

theorem original_number_of_men (W : ℝ) (M : ℝ) (total_work : ℝ) :
  (M * W * 11 = (M + 10) * W * 8) → M = 27 :=
by
  sorry

end original_number_of_men_l52_52022


namespace ruby_total_classes_l52_52263

noncomputable def average_price_per_class (pack_cost : ℝ) (pack_classes : ℕ) : ℝ :=
  pack_cost / pack_classes

noncomputable def additional_class_price (average_price : ℝ) : ℝ :=
  average_price + (1/3 * average_price)

noncomputable def total_classes_taken (total_payment : ℝ) (pack_cost : ℝ) (pack_classes : ℕ) : ℕ :=
  let avg_price := average_price_per_class pack_cost pack_classes
  let additional_price := additional_class_price avg_price
  let additional_classes := (total_payment - pack_cost) / additional_price
  pack_classes + Nat.floor additional_classes -- We use Nat.floor to convert from real to natural number of classes

theorem ruby_total_classes 
  (pack_cost : ℝ) 
  (pack_classes : ℕ) 
  (total_payment : ℝ) 
  (h_pack_cost : pack_cost = 75) 
  (h_pack_classes : pack_classes = 10) 
  (h_total_payment : total_payment = 105) :
  total_classes_taken total_payment pack_cost pack_classes = 13 :=
by
  -- The proof would go here
  sorry

end ruby_total_classes_l52_52263


namespace unique_prime_range_start_l52_52532

theorem unique_prime_range_start (N : ℕ) (hN : N = 220) (h1 : ∀ n, N ≥ n → n ≥ 211 → ¬Prime n) (h2 : Prime 211) : N - 8 = 212 :=
by
  sorry

end unique_prime_range_start_l52_52532


namespace product_of_two_numbers_l52_52877

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l52_52877


namespace factorize_expression_l52_52068

theorem factorize_expression (x : ℝ) : x^2 - 2 * x + 1 = (x - 1)^2 :=
by
  sorry

end factorize_expression_l52_52068


namespace correlation_coefficient_correct_option_l52_52019

variable (r : ℝ)

-- Definitions of Conditions
def positive_correlation : Prop := r > 0 → ∀ x y : ℝ, x * y > 0
def range_r : Prop := -1 < r ∧ r < 1
def correlation_strength : Prop := |r| < 1 → (∀ ε : ℝ, 0 < ε → ∃ δ : ℝ, 0 < δ ∧ δ < ε ∧ |r| < δ)

-- Theorem statement
theorem correlation_coefficient_correct_option :
  (positive_correlation r) ∧
  (range_r r) ∧
  (correlation_strength r) →
  (r ≠ 0 → |r| < 1) :=
by
  sorry

end correlation_coefficient_correct_option_l52_52019


namespace tom_spends_total_cost_l52_52601

theorem tom_spends_total_cost :
  (let total_bricks := 1000
       half_bricks := total_bricks / 2
       full_price := 0.50
       half_price := full_price / 2
       cost_half := half_bricks * half_price
       cost_full := half_bricks * full_price
       total_cost := cost_half + cost_full
   in total_cost = 375) := 
by
  let total_bricks := 1000
  let half_bricks := total_bricks / 2
  let full_price := 0.50
  let half_price := full_price / 2
  let cost_half := half_bricks * half_price
  let cost_full := half_bricks * full_price
  let total_cost := cost_half + cost_full
  show total_cost = 375 from sorry

end tom_spends_total_cost_l52_52601


namespace plane_speed_east_l52_52637

def plane_travel_problem (v : ℕ) : Prop :=
  let time : ℕ := 35 / 10 
  let distance_east := v * time
  let distance_west := 275 * time
  let total_distance := distance_east + distance_west
  total_distance = 2100

theorem plane_speed_east : ∃ v : ℕ, plane_travel_problem v ∧ v = 325 :=
sorry

end plane_speed_east_l52_52637


namespace simplify_expression_l52_52924

theorem simplify_expression (x y : ℝ) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / (1 / 2 * x * y)^2 = 8 * x - 12 * y := by
  sorry

end simplify_expression_l52_52924


namespace bricks_required_l52_52625

-- Definitions
def courtyard_length : ℕ := 20  -- in meters
def courtyard_breadth : ℕ := 16  -- in meters
def brick_length : ℕ := 20  -- in centimeters
def brick_breadth : ℕ := 10  -- in centimeters

-- Statement to prove
theorem bricks_required :
  ((courtyard_length * 100) * (courtyard_breadth * 100)) / (brick_length * brick_breadth) = 16000 :=
sorry

end bricks_required_l52_52625


namespace range_of_c_l52_52329

variable {a b c : ℝ} -- Declare the variables

-- Define the conditions
def triangle_condition (a b : ℝ) : Prop :=
|a + b - 4| + (a - b + 2)^2 = 0

-- Define the proof problem
theorem range_of_c {a b c : ℝ} (h : triangle_condition a b) : 2 < c ∧ c < 4 :=
sorry -- Proof to be completed

end range_of_c_l52_52329


namespace cost_of_traveling_roads_l52_52036

def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 40
def road_width : ℕ := 10
def cost_per_sqm : ℕ := 3

def area_road_parallel_length : ℕ := road_width * lawn_length
def area_road_parallel_breadth : ℕ := road_width * lawn_breadth
def area_intersection : ℕ := road_width * road_width

def total_area_roads : ℕ := area_road_parallel_length + area_road_parallel_breadth - area_intersection
def total_cost : ℕ := total_area_roads * cost_per_sqm

theorem cost_of_traveling_roads : total_cost = 3300 :=
by
  sorry

end cost_of_traveling_roads_l52_52036


namespace percent_of_absent_students_l52_52999

noncomputable def absent_percentage : ℚ :=
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := boys * (1/5 : ℚ)
  let absent_girls := girls * (1/4 : ℚ)
  let total_absent := absent_boys + absent_girls
  (total_absent / total_students) * 100

theorem percent_of_absent_students : absent_percentage = 22.5 := sorry

end percent_of_absent_students_l52_52999


namespace plantingMethodsCalculation_l52_52268

noncomputable def numPlantingMethods : Nat :=
  let totalSeeds := 5
  let endChoices := 3 * 2 -- Choosing 2 seeds for the ends from 3 remaining types
  let middleChoices := 6 -- Permutations of (A, B, another type) = 3! = 6
  endChoices * middleChoices

theorem plantingMethodsCalculation : numPlantingMethods = 24 := by
  sorry

end plantingMethodsCalculation_l52_52268


namespace area_of_rectangle_l52_52161

namespace RectangleArea

variable (l b : ℕ)
variable (h1 : l = 3 * b)
variable (h2 : 2 * (l + b) = 88)

theorem area_of_rectangle : l * b = 363 :=
by
  -- We will prove this in Lean 
  sorry

end RectangleArea

end area_of_rectangle_l52_52161


namespace yao_ming_mcgrady_probability_l52_52781

theorem yao_ming_mcgrady_probability
        (p : ℝ) (q : ℝ)
        (h1 : p = 0.8)
        (h2 : q = 0.7) :
        (2 * p * (1 - p)) * (2 * q * (1 - q)) = 0.1344 := 
by
  sorry

end yao_ming_mcgrady_probability_l52_52781


namespace find_scalars_l52_52847

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1],
    ![4, 3]]

def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0],
    ![0, 1]]

theorem find_scalars (r s : ℤ) (h : B^6 = r • B + s • I) :
  (r = 1125) ∧ (s = -1875) :=
sorry

end find_scalars_l52_52847


namespace garrett_bought_peanut_granola_bars_l52_52213

def garrett_granola_bars (t o : ℕ) (h_t : t = 14) (h_o : o = 6) : ℕ :=
  t - o

theorem garrett_bought_peanut_granola_bars : garrett_granola_bars 14 6 rfl rfl = 8 :=
  by
    unfold garrett_granola_bars
    rw [Nat.sub_eq_of_eq_add]
    sorry

end garrett_bought_peanut_granola_bars_l52_52213


namespace min_possible_value_of_coefficient_x_l52_52821

theorem min_possible_value_of_coefficient_x 
  (c d : ℤ) 
  (h1 : c * d = 15) 
  (h2 : ∃ (C : ℤ), C = c + d) 
  (h3 : c ≠ d ∧ c ≠ 34 ∧ d ≠ 34) :
  (∃ (C : ℤ), C = c + d ∧ C = 34) :=
sorry

end min_possible_value_of_coefficient_x_l52_52821


namespace solution_A_l52_52990

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end solution_A_l52_52990


namespace max_prob_win_two_consecutive_is_C_l52_52438

-- Definitions based on conditions
def p1 : ℝ := sorry -- Probability of winning against A
def p2 : ℝ := sorry -- Probability of winning against B
def p3 : ℝ := sorry -- Probability of winning against C

-- Condition p3 > p2 > p1 > 0
axiom h_p3_gt_p2 : p3 > p2
axiom h_p2_gt_p1 : p2 > p1
axiom h_p1_gt_0 : p1 > 0

-- Prove the maximum probability of winning two consecutive games
theorem max_prob_win_two_consecutive_is_C :
  let P_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      P_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      P_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)
  in P_C > P_A ∧ P_C > P_B :=
by
  sorry

end max_prob_win_two_consecutive_is_C_l52_52438


namespace maisy_earnings_increase_l52_52249

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l52_52249


namespace graph_is_empty_l52_52194

/-- The given equation 3x² + 4y² - 12x - 16y + 36 = 0 has no real solutions. -/
theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 :=
by
  intro x y
  sorry

end graph_is_empty_l52_52194


namespace ellipse_fixed_point_l52_52216

theorem ellipse_fixed_point (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (c : ℝ) (h3 : c = 1) 
    (h4 : a = 2) (h5 : b = Real.sqrt 3) :
    (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
        ∃ M : ℝ × ℝ, (M.1 = 4) ∧ 
        ∃ Q : ℝ × ℝ, (Q.1= (P.1) ∧ Q.2 = - (P.2)) ∧ 
            ∃ fixed_point : ℝ × ℝ, (fixed_point.1 = 5 / 2) ∧ (fixed_point.2 = 0) ∧ 
            ∃ k, (Q.2 - M.2) = k * (Q.1 - M.1) ∧ 
            ∃ l, fixed_point.2 = l * (fixed_point.1 - M.1)) :=
sorry

end ellipse_fixed_point_l52_52216


namespace arithmetic_sequence_S9_l52_52983

theorem arithmetic_sequence_S9 :
  ∀ {a : ℕ → ℤ} {S : ℕ → ℤ},
  (∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2) →
  a 2 = 3 →
  S 4 = 16 →
  S 9 = 81 :=
by
  intro a S h_S h_a2 h_S4
  sorry

end arithmetic_sequence_S9_l52_52983


namespace value_of_X_l52_52342

def M : ℕ := 2024 / 4
def N : ℕ := M / 2
def X : ℕ := M + N

theorem value_of_X : X = 759 := by
  sorry

end value_of_X_l52_52342


namespace total_area_of_strips_l52_52933

def strip1_length := 12
def strip1_width := 1
def strip2_length := 8
def strip2_width := 2
def num_strips1 := 2
def num_strips2 := 2
def overlap_area_per_strip := 2
def num_overlaps := 4
def total_area_covered := 48

theorem total_area_of_strips : 
  num_strips1 * (strip1_length * strip1_width) + 
  num_strips2 * (strip2_length * strip2_width) - 
  num_overlaps * overlap_area_per_strip = total_area_covered := sorry

end total_area_of_strips_l52_52933


namespace min_visible_sum_of_prism_faces_l52_52172

theorem min_visible_sum_of_prism_faces :
  let corners := 8
  let edges := 8
  let face_centers := 12
  let min_corner_sum := 6 -- Each corner dice can show 1, 2, and 3
  let min_edge_sum := 3    -- Each edge dice can show 1 and 2
  let min_face_center_sum := 1 -- Each face center dice can show 1
  let total_sum := corners * min_corner_sum + edges * min_edge_sum + face_centers * min_face_center_sum
  total_sum = 84 := 
by
  -- The proof is omitted
  sorry

end min_visible_sum_of_prism_faces_l52_52172


namespace daisies_per_bouquet_is_7_l52_52628

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end daisies_per_bouquet_is_7_l52_52628


namespace combinatorial_identity_l52_52511

theorem combinatorial_identity :
  (nat.choose 22 5) = 26334 := by
  have h1 : nat.choose 20 3 = 1140 := sorry
  have h2 : nat.choose 20 4 = 4845 := sorry
  have h3 : nat.choose 20 5 = 15504 := sorry
  sorry

end combinatorial_identity_l52_52511


namespace product_of_two_numbers_l52_52876

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 48) (h_gcd : gcd a b = 8) : a * b = 384 :=
by sorry

end product_of_two_numbers_l52_52876


namespace option_A_option_B_option_C_option_D_l52_52957

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end option_A_option_B_option_C_option_D_l52_52957


namespace train_length_calculation_l52_52771

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l52_52771


namespace shifted_parabola_eq_l52_52712

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -(x^2)

-- Define the transformation for shifting left 2 units
def shift_left (x : ℝ) : ℝ := x + 2

-- Define the transformation for shifting down 3 units
def shift_down (y : ℝ) : ℝ := y - 3

-- Define the new parabola equation after shifting
def new_parabola (x : ℝ) : ℝ := shift_down (original_parabola (shift_left x))

-- The theorem to be proven
theorem shifted_parabola_eq : new_parabola x = -(x + 2)^2 - 3 := by
  sorry

end shifted_parabola_eq_l52_52712


namespace problem_solution_l52_52507

theorem problem_solution (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : 2 * x + y = 2) : x + y = 1 :=
by
  sorry

end problem_solution_l52_52507


namespace base7_and_base13_addition_l52_52315

def base7_to_nat (a b c : ℕ) : ℕ := a * 49 + b * 7 + c

def base13_to_nat (a b c : ℕ) : ℕ := a * 169 + b * 13 + c

theorem base7_and_base13_addition (a b c d e f : ℕ) :
  a = 5 → b = 3 → c = 6 → d = 4 → e = 12 → f = 5 →
  base7_to_nat a b c + base13_to_nat d e f = 1109 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  unfold base7_to_nat base13_to_nat
  sorry

end base7_and_base13_addition_l52_52315


namespace union_A_B_subset_B_A_l52_52690

-- Condition definitions
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Problem 1: If m = 4, prove A ∪ B = {2, 4, 8}
theorem union_A_B (m : ℝ) (h : m = 4) : A ∪ B m = {2, 4, 8} :=
sorry

-- Problem 2: If B ⊆ A, find the range for m
theorem subset_B_A (m : ℝ) (h : B m ⊆ A) : 
  m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1 / 2 :=
sorry

end union_A_B_subset_B_A_l52_52690


namespace number_of_ways_to_choose_marbles_l52_52981

theorem number_of_ways_to_choose_marbles 
  (total_marbles : ℕ) 
  (red_count green_count blue_count : ℕ) 
  (total_choice chosen_rgb_count remaining_choice : ℕ) 
  (h_total_marbles : total_marbles = 15) 
  (h_red_count : red_count = 2) 
  (h_green_count : green_count = 2) 
  (h_blue_count : blue_count = 2) 
  (h_total_choice : total_choice = 5) 
  (h_chosen_rgb_count : chosen_rgb_count = 2) 
  (h_remaining_choice : remaining_choice = 3) :
  ∃ (num_ways : ℕ), num_ways = 3300 :=
sorry

end number_of_ways_to_choose_marbles_l52_52981


namespace largest_prime_factor_1729_l52_52013

open Nat

theorem largest_prime_factor_1729 : 
  ∃ p, prime p ∧ p ∣ 1729 ∧ (∀ q, prime q ∧ q ∣ 1729 → q ≤ p) := 
by
  sorry

end largest_prime_factor_1729_l52_52013


namespace area_of_square_l52_52457

theorem area_of_square 
  (a : ℝ)
  (h : 4 * a = 28) :
  a^2 = 49 :=
sorry

end area_of_square_l52_52457


namespace gala_arrangements_l52_52029

theorem gala_arrangements :
  let original_programs := 10
  let added_programs := 3
  let total_positions := original_programs + 1 - 2 -- Excluding first and last
  (total_positions * (total_positions - 1) * (total_positions - 2)) / 6 = 165 :=
by sorry

end gala_arrangements_l52_52029


namespace tan_five_pi_over_four_eq_one_l52_52489

theorem tan_five_pi_over_four_eq_one : Real.tan (5 * Real.pi / 4) = 1 :=
by sorry

end tan_five_pi_over_four_eq_one_l52_52489


namespace max_minutes_sleep_without_missing_happy_moment_l52_52703

def isHappyMoment (h m : ℕ) : Prop :=
  (h = 4 * m ∨ m = 4 * h) ∧ h < 24 ∧ m < 60

def sleepDurationMax : ℕ :=
  239

theorem max_minutes_sleep_without_missing_happy_moment :
  ∀ (sleepDuration : ℕ), sleepDuration ≤ 239 :=
sorry

end max_minutes_sleep_without_missing_happy_moment_l52_52703


namespace total_money_made_from_jerseys_l52_52584

def price_per_jersey : ℕ := 76
def jerseys_sold : ℕ := 2

theorem total_money_made_from_jerseys : price_per_jersey * jerseys_sold = 152 := 
by
  -- The actual proof steps will go here
  sorry

end total_money_made_from_jerseys_l52_52584


namespace minimum_fourth_day_rain_l52_52144

def rainstorm_duration : Nat := 4
def area_capacity_feet : Nat := 6
def area_capacity_inches : Nat := area_capacity_feet * 12 -- Convert to inches
def drainage_rate : Nat := 3 -- inches per day
def rainfall_day1 : Nat := 10
def rainfall_day2 : Nat := 2 * rainfall_day1
def rainfall_day3 : Nat := (3 * rainfall_day1) -- 50% more than Day 2
def total_rain_first_three_days : Nat := rainfall_day1 + rainfall_day2 + rainfall_day3
def drained_amount : Nat := 3 * drainage_rate
def effective_capacity : Nat := area_capacity_inches - drained_amount
def overflow_capacity_left : Nat := effective_capacity - total_rain_first_three_days

theorem minimum_fourth_day_rain : Nat :=
  overflow_capacity_left + 1 = 4

end minimum_fourth_day_rain_l52_52144


namespace smallest_multiple_6_15_l52_52500

theorem smallest_multiple_6_15 (b : ℕ) (hb1 : b % 6 = 0) (hb2 : b % 15 = 0) :
  ∃ (b : ℕ), (b > 0) ∧ (b % 6 = 0) ∧ (b % 15 = 0) ∧ (∀ x : ℕ, (x > 0) ∧ (x % 6 = 0) ∧ (x % 15 = 0) → x ≥ b) :=
sorry

end smallest_multiple_6_15_l52_52500


namespace determine_c_l52_52887

-- Definitions of the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 1

-- Hypothesis for the sequence to be geometric
def geometric_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∃ c, ∀ n, a (n + 1) + c = r * (a n + c)

-- The goal to prove
theorem determine_c (a : ℕ → ℕ) (c : ℕ) (r := 2) :
  seq a →
  geometric_seq a c →
  c = 1 :=
by
  intros h_seq h_geo
  sorry

end determine_c_l52_52887


namespace find_k_l52_52528

theorem find_k (k x y : ℝ) (h_ne_zero : k ≠ 0) (h_x : x = 4) (h_y : y = -1/2) (h_eq : y = k / x) : k = -2 :=
by
  -- This is where the proof would go
  sorry

end find_k_l52_52528


namespace problem_statement_l52_52953

theorem problem_statement (y : ℝ) (h : 8 / y^3 = y / 32) : y = 4 :=
by
  sorry

end problem_statement_l52_52953


namespace triangle_inequality_l52_52542

variables (a b c S : ℝ) (S_def : S = (a + b + c) / 2)

theorem triangle_inequality 
  (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) 
  ≤ 3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) :=
sorry

end triangle_inequality_l52_52542


namespace tickets_difference_l52_52177

-- Definitions of conditions
def tickets_won : Nat := 19
def tickets_for_toys : Nat := 12
def tickets_for_clothes : Nat := 7

-- Theorem statement: Prove that the difference between tickets used for toys and tickets used for clothes is 5
theorem tickets_difference : (tickets_for_toys - tickets_for_clothes = 5) := by
  sorry

end tickets_difference_l52_52177


namespace total_worth_of_stock_l52_52455

theorem total_worth_of_stock (X : ℝ) :
  (0.30 * 0.10 * X + 0.40 * -0.05 * X + 0.30 * -0.10 * X = -500) → X = 25000 :=
by
  intro h
  -- Proof to be completed
  sorry

end total_worth_of_stock_l52_52455


namespace XiaoMing_selection_l52_52140

def final_positions (n : Nat) : List Nat :=
  if n <= 2 then
    List.range n
  else
    final_positions (n / 2) |>.filter (λ k => k % 2 = 0) |>.map (λ k => k / 2)

theorem XiaoMing_selection (n : Nat) (h : n = 32) : final_positions n = [16, 32] :=
  by
  sorry

end XiaoMing_selection_l52_52140


namespace correct_sampling_methods_l52_52149

-- Define conditions for the sampling problems
structure SamplingProblem where
  scenario: String
  samplingMethod: String

-- Define the three scenarios
def firstScenario : SamplingProblem :=
  { scenario := "Draw 5 bottles from 15 bottles of drinks for food hygiene inspection", samplingMethod := "Simple random sampling" }

def secondScenario : SamplingProblem :=
  { scenario := "Sample 20 staff members from 240 staff members in a middle school", samplingMethod := "Stratified sampling" }

def thirdScenario : SamplingProblem :=
  { scenario := "Select 25 audience members from a full science and technology report hall", samplingMethod := "Systematic sampling" }

-- Main theorem combining all conditions and proving the correct answer
theorem correct_sampling_methods :
  (firstScenario.samplingMethod = "Simple random sampling") ∧
  (secondScenario.samplingMethod = "Stratified sampling") ∧
  (thirdScenario.samplingMethod = "Systematic sampling") :=
by
  sorry -- Proof is omitted

end correct_sampling_methods_l52_52149


namespace base_eight_to_ten_l52_52737

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end base_eight_to_ten_l52_52737


namespace initial_floors_l52_52645

-- Define the conditions given in the problem
def austin_time := 60 -- Time Austin takes in seconds to reach the ground floor
def jake_time := 90 -- Time Jake takes in seconds to reach the ground floor
def jake_steps_per_sec := 3 -- Jake descends 3 steps per second
def steps_per_floor := 30 -- There are 30 steps per floor

-- Define the total number of steps Jake descends
def total_jake_steps := jake_time * jake_steps_per_sec

-- Define the number of floors descended in terms of total steps and steps per floor
def num_floors := total_jake_steps / steps_per_floor

-- Theorem stating the number of floors is 9
theorem initial_floors : num_floors = 9 :=
by 
  -- Provide the basic proof structure
  sorry

end initial_floors_l52_52645


namespace window_width_l52_52037

theorem window_width (h_pane_height : ℕ) (h_to_w_ratio_num : ℕ) (h_to_w_ratio_den : ℕ) (gaps : ℕ) 
(border : ℕ) (columns : ℕ) 
(panes_per_row : ℕ) (pane_height : ℕ) 
(heights_equal : h_pane_height = pane_height)
(ratio : h_to_w_ratio_num * pane_height = h_to_w_ratio_den * panes_per_row)
: columns * (h_to_w_ratio_den * pane_height / h_to_w_ratio_num) + 
  gaps + 2 * border = 57 := sorry

end window_width_l52_52037


namespace ratio_of_juniors_to_freshmen_l52_52831

variables (f j : ℕ) 

theorem ratio_of_juniors_to_freshmen (h1 : (1/4 : ℚ) * f = (1/2 : ℚ) * j) :
  j = f / 2 :=
by
  sorry

end ratio_of_juniors_to_freshmen_l52_52831


namespace racers_final_segment_l52_52048

def final_racer_count : Nat := 9

def segment_eliminations (init_count: Nat) : Nat :=
  let seg1 := init_count - Int.toNat (Nat.sqrt init_count)
  let seg2 := seg1 - seg1 / 3
  let seg3 := seg2 - (seg2 / 4 + (2 ^ 2))
  let seg4 := seg3 - seg3 / 3
  let seg5 := seg4 / 2
  let seg6 := seg5 - (seg5 * 3 / 4)
  seg6

theorem racers_final_segment
  (init_count: Nat)
  (h: init_count = 225) :
  segment_eliminations init_count = final_racer_count :=
  by
  rw [h]
  unfold segment_eliminations
  sorry

end racers_final_segment_l52_52048


namespace expression_value_l52_52109

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x + y + z = 0) (h2 : xy + xz + yz ≠ 0) :
  (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)^2) = 3 / (x^2 + xy + y^2)^2 :=
by
  sorry

end expression_value_l52_52109


namespace ratio_of_radii_l52_52353

theorem ratio_of_radii (r R : ℝ) (hR : R > 0) (hr : r > 0)
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l52_52353


namespace find_a_l52_52221

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x

theorem find_a (a : ℝ) (h : (deriv (f a)) e = 3) : a = 3 / 2 :=
by
-- placeholder for the proof
sorry

end find_a_l52_52221


namespace sum_of_integers_is_24_l52_52114

theorem sum_of_integers_is_24 (x y : ℕ) (hx : x > y) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 :=
by
  sorry

end sum_of_integers_is_24_l52_52114


namespace intersection_M_N_eq_2_4_l52_52244

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℕ := {x | ∃ y, y = Real.log (6 - x) ∧ x < 6}

theorem intersection_M_N_eq_2_4 : M ∩ N = {2, 4} :=
by sorry

end intersection_M_N_eq_2_4_l52_52244


namespace base_eight_to_base_ten_l52_52738

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end base_eight_to_base_ten_l52_52738


namespace Susan_has_10_dollars_left_l52_52266

def initial_amount : ℝ := 80
def food_expense : ℝ := 15
def rides_expense : ℝ := 3 * food_expense
def games_expense : ℝ := 10
def total_expense : ℝ := food_expense + rides_expense + games_expense
def remaining_amount : ℝ := initial_amount - total_expense

theorem Susan_has_10_dollars_left : remaining_amount = 10 := by
  sorry

end Susan_has_10_dollars_left_l52_52266


namespace total_hours_difference_l52_52662

-- Definitions based on conditions
def hours_learning_english := 6
def hours_learning_chinese := 2
def hours_learning_spanish := 3
def hours_learning_french := 1

-- Calculation of total time spent on English and Chinese
def total_hours_english_chinese := hours_learning_english + hours_learning_chinese

-- Calculation of total time spent on Spanish and French
def total_hours_spanish_french := hours_learning_spanish + hours_learning_french

-- Calculation of the difference in hours spent
def hours_difference := total_hours_english_chinese - total_hours_spanish_french

-- Statement to prove
theorem total_hours_difference : hours_difference = 4 := by
  sorry

end total_hours_difference_l52_52662


namespace real_solutions_l52_52070

theorem real_solutions :
  ∃ x : ℝ, 
    (x = 9 ∨ x = 5) ∧ 
    (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
     1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10) := 
by 
  sorry  

end real_solutions_l52_52070


namespace find_y_plus_one_over_y_l52_52334

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end find_y_plus_one_over_y_l52_52334


namespace simplify_and_rationalize_l52_52121

theorem simplify_and_rationalize :
  let x := (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 17)
  x = 3 * Real.sqrt 84885 / 1309 := sorry

end simplify_and_rationalize_l52_52121


namespace probability_of_valid_pairs_l52_52415

open MeasureTheory Probability

noncomputable def fair_dice_space : MassFunction (ℕ × ℕ) :=
  MassFunction.uniform (finset.univ.product finset.univ)

def valid_pair (p : ℕ × ℕ) : Prop :=
  let (a, b) := p in
  a + b ≤ 10 ∧ (a > 3 ∨ b > 3)

theorem probability_of_valid_pairs : 
  fair_dice_space.probOf valid_pair = 2 / 3 :=
sorry

end probability_of_valid_pairs_l52_52415


namespace number_of_polynomials_l52_52198

theorem number_of_polynomials :
  let satisfies_conditions :=
    λ (n : ℕ) (a : ℕ → ℤ),
      n + ((Finset.range (n + 1)).sum (λ i, |a i|)) = 5 ∧
      ∃ i, i ≤ n ∧ a i < 0
  in
  (Finset.sum (Finset.range 6) 
    (λ n, (Finset.piFinset' (Finset.range (n + 1)) (λ _, Finset.univ.filter (λ x, satisfies_conditions n (λ i, x i)))).card)) = 6 := 
by 
  sorry

end number_of_polynomials_l52_52198


namespace initial_kola_volume_l52_52914

theorem initial_kola_volume (V : ℝ) (S : ℝ) :
  S = 0.14 * V →
  (S + 3.2) / (V + 20) = 0.14111111111111112 →
  V = 340 :=
by
  intro h_S h_equation
  sorry

end initial_kola_volume_l52_52914


namespace buffy_whiskers_l52_52211

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l52_52211


namespace systematic_sampling_interval_l52_52432

def population_size : ℕ := 2000
def sample_size : ℕ := 50
def interval (N n : ℕ) : ℕ := N / n

theorem systematic_sampling_interval :
  interval population_size sample_size = 40 := by
  sorry

end systematic_sampling_interval_l52_52432


namespace inequality_proof_l52_52750

theorem inequality_proof (a b x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l52_52750


namespace max_watches_two_hours_l52_52693

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end max_watches_two_hours_l52_52693


namespace euler_disproof_l52_52995

theorem euler_disproof :
  ∃ (n : ℕ), 0 < n ∧ (133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144) :=
by
  sorry

end euler_disproof_l52_52995


namespace machine_A_sprockets_per_hour_l52_52248

theorem machine_A_sprockets_per_hour 
  (A T_Q : ℝ)
  (h1 : 550 = 1.1 * A * T_Q)
  (h2 : 550 = A * (T_Q + 10)) 
  : A = 5 :=
by
  sorry

end machine_A_sprockets_per_hour_l52_52248


namespace find_n_l52_52220

theorem find_n (m n : ℕ) (h1: m = 34)
               (h2: (1^(m+1) / 5^(m+1)) * (1^n / 4^n) = 1 / (2 * 10^35)) : 
               n = 18 :=
by
  sorry

end find_n_l52_52220


namespace value_of_expression_l52_52108

theorem value_of_expression :
  let x := 1
  let y := -1
  let z := 0
  2 * x + 3 * y + 4 * z = -1 :=
by
  sorry

end value_of_expression_l52_52108


namespace largest_prime_factor_1729_l52_52011

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def largest_prime_factor (n : ℕ) : ℕ :=
  Nat.greatest (λ p : ℕ => Nat.Prime p ∧ p ∣ n)

theorem largest_prime_factor_1729 : largest_prime_factor 1729 = 19 := 
by
  sorry

end largest_prime_factor_1729_l52_52011


namespace quotient_remainder_threefold_l52_52155

theorem quotient_remainder_threefold (a b c d : ℤ)
  (h : a = b * c + d) :
  3 * a = 3 * b * c + 3 * d :=
by sorry

end quotient_remainder_threefold_l52_52155


namespace find_value_of_p_l52_52813

variable (x y : ℝ)

/-- Given that the hyperbola has the equation x^2 / 4 - y^2 / 12 = 1
    and the eccentricity e = 2, and that the parabola x = 2 * p * y^2 has its focus at (e, 0), 
    prove that the value of the real number p is 1/8. -/
theorem find_value_of_p :
  (∃ (p : ℝ), 
    (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) ∧ 
    (∀ (x y : ℝ), x = 2 * p * y^2) ∧
    (2 = 2)) →
    ∃ (p : ℝ), p = 1/8 :=
by 
  sorry

end find_value_of_p_l52_52813


namespace r_earns_per_day_l52_52286

variables (P Q R S : ℝ)

theorem r_earns_per_day
  (h1 : P + Q + R + S = 240)
  (h2 : P + R + S = 160)
  (h3 : Q + R = 150)
  (h4 : Q + R + S = 650 / 3) :
  R = 70 :=
by
  sorry

end r_earns_per_day_l52_52286


namespace score_difference_l52_52834

noncomputable def mean_score (scores pcts : List ℕ) : ℚ := 
  (List.zipWith (· * ·) scores pcts).sum / 100

def median_score (scores pcts : List ℕ) : ℚ := 75

theorem score_difference :
  let scores := [60, 75, 85, 95]
  let pcts := [20, 50, 15, 15]
  abs (median_score scores pcts - mean_score scores pcts) = 1.5 := by
  sorry

end score_difference_l52_52834


namespace third_red_yellow_flash_is_60_l52_52764

-- Define the flashing intervals for red, yellow, and green lights
def red_interval : Nat := 3
def yellow_interval : Nat := 4
def green_interval : Nat := 8

-- Define the function for finding the time of the third occurrence of only red and yellow lights flashing together
def third_red_yellow_flash : Nat :=
  let lcm_red_yellow := Nat.lcm red_interval yellow_interval
  let times := (List.range (100)).filter (fun t => t % lcm_red_yellow = 0 ∧ t % green_interval ≠ 0)
  times[2] -- Getting the third occurrence

-- Prove that the third occurrence time is 60 seconds
theorem third_red_yellow_flash_is_60 :
  third_red_yellow_flash = 60 :=
  by
    -- Proof goes here
    sorry

end third_red_yellow_flash_is_60_l52_52764


namespace isosceles_triangle_and_sin_cos_range_l52_52529

theorem isosceles_triangle_and_sin_cos_range 
  (A B C : ℝ) (a b c : ℝ) 
  (hA_pos : 0 < A) (hA_lt_pi_div_2 : A < π / 2) (h_triangle : a * Real.cos B = b * Real.cos A) :
  (A = B ∧
  ∃ x, x = Real.sin B + Real.cos (A + π / 6) ∧ (1 / 2 < x ∧ x ≤ 1)) :=
by
  sorry

end isosceles_triangle_and_sin_cos_range_l52_52529


namespace students_more_than_pets_l52_52660

theorem students_more_than_pets
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (birds_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ)
  (total_rabbits : ℕ)
  (total_birds : ℕ)
  (total_pets : ℕ)
  (difference : ℕ)
  : students_per_classroom = 22 → 
    rabbits_per_classroom = 3 → 
    birds_per_classroom = 2 → 
    number_of_classrooms = 5 → 
    total_students = students_per_classroom * number_of_classrooms → 
    total_rabbits = rabbits_per_classroom * number_of_classrooms → 
    total_birds = birds_per_classroom * number_of_classrooms → 
    total_pets = total_rabbits + total_birds → 
    difference = total_students - total_pets →
    difference = 85 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end students_more_than_pets_l52_52660


namespace largest_possible_difference_l52_52042

theorem largest_possible_difference (A_est : ℕ) (B_est : ℕ) (A : ℝ) (B : ℝ)
(hA_est : A_est = 40000) (hB_est : B_est = 70000)
(hA_range : 36000 ≤ A ∧ A ≤ 44000)
(hB_range : 60870 ≤ B ∧ B ≤ 82353) :
  abs (B - A) = 46000 :=
by sorry

end largest_possible_difference_l52_52042


namespace classroom_students_count_l52_52592

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end classroom_students_count_l52_52592


namespace weights_difference_l52_52728

-- Definitions based on conditions
def A : ℕ := 36
def ratio_part : ℕ := A / 4
def B : ℕ := 5 * ratio_part
def C : ℕ := 6 * ratio_part

-- Theorem to prove
theorem weights_difference :
  (A + C) - B = 45 := by
  sorry

end weights_difference_l52_52728


namespace total_annual_interest_l52_52299

theorem total_annual_interest 
    (principal1 principal2 : ℝ)
    (rate1 rate2 : ℝ)
    (time : ℝ)
    (h1 : principal1 = 26000)
    (h2 : rate1 = 0.08)
    (h3 : principal2 = 24000)
    (h4 : rate2 = 0.085)
    (h5 : time = 1) :
    principal1 * rate1 * time + principal2 * rate2 * time = 4120 := 
sorry

end total_annual_interest_l52_52299


namespace eq_condition_implies_inequality_l52_52347

theorem eq_condition_implies_inequality (a : ℝ) (h_neg_root : 2 * a - 4 < 0) : (a - 3) * (a - 4) > 0 :=
by {
  sorry
}

end eq_condition_implies_inequality_l52_52347


namespace angle_z_value_l52_52976

theorem angle_z_value
  (ABC BAC : ℝ)
  (h1 : ABC = 70)
  (h2 : BAC = 50)
  (h3 : ∀ BCA : ℝ, BCA + ABC + BAC = 180) :
  ∃ z : ℝ, z = 30 :=
by
  sorry

end angle_z_value_l52_52976


namespace binomial_7_2_eq_21_l52_52479

def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binomial_7_2_eq_21 : binomial 7 2 = 21 :=
by
  sorry

end binomial_7_2_eq_21_l52_52479


namespace geometric_sequence_product_l52_52356

theorem geometric_sequence_product
    (a : ℕ → ℝ)
    (r : ℝ)
    (h₀ : a 1 = 1 / 9)
    (h₃ : a 4 = 3)
    (h_geom : ∀ n, a (n + 1) = a n * r) :
    (a 1) * (a 2) * (a 3) * (a 4) * (a 5) = 1 :=
sorry

end geometric_sequence_product_l52_52356


namespace tina_jumps_more_than_cindy_l52_52788

def cindy_jumps : ℕ := 12
def betsy_jumps : ℕ := cindy_jumps / 2
def tina_jumps : ℕ := betsy_jumps * 3

theorem tina_jumps_more_than_cindy : tina_jumps - cindy_jumps = 6 := by
  sorry

end tina_jumps_more_than_cindy_l52_52788


namespace daniel_candy_removal_l52_52791

theorem daniel_candy_removal (n k : ℕ) (h1 : n = 24) (h2 : k = 4) : ∃ m : ℕ, n % k = 0 → m = 0 :=
by
  sorry

end daniel_candy_removal_l52_52791


namespace integral_solution_l52_52616

noncomputable def definite_integral : ℝ :=
  ∫ x in (-2 : ℝ)..(0 : ℝ), (x + 2)^2 * (Real.cos (3 * x))

theorem integral_solution :
  definite_integral = (12 - 2 * Real.sin 6) / 27 :=
sorry

end integral_solution_l52_52616


namespace count_integer_solutions_l52_52961

theorem count_integer_solutions :
  (2 * 9^2 + 5 * 9 * -4 + 3 * (-4)^2 = 30) →
  ∃ S : Finset (ℤ × ℤ), (∀ x y : ℤ, ((2 * x ^ 2 + 5 * x * y + 3 * y ^ 2 = 30) ↔ (x, y) ∈ S)) ∧ 
  S.card = 16 :=
by sorry

end count_integer_solutions_l52_52961


namespace highest_possible_rubidium_concentration_l52_52313

noncomputable def max_rubidium_concentration (R C F : ℝ) : Prop :=
  (R + C + F > 0) →
  (0.10 * R + 0.08 * C + 0.05 * F) / (R + C + F) = 0.07 ∧
  (0.05 * F) / (R + C + F) ≤ 0.02 →
  (0.10 * R) / (R + C + F) = 0.01

theorem highest_possible_rubidium_concentration :
  ∃ R C F : ℝ, max_rubidium_concentration R C F :=
sorry

end highest_possible_rubidium_concentration_l52_52313


namespace volleyball_match_win_probability_l52_52267

/-- Probability of a team winning a best-of-five match 3:0 given different probabilities for each set -/
theorem volleyball_match_win_probability :
  let p₁ := 2/3 -- Probability of winning any of the first 4 sets
  let p₅ := 1/2 -- Probability of winning the fifth set
  p₁ * p₁ * p₁ = 8/27 :=
by
  sorry

end volleyball_match_win_probability_l52_52267


namespace hiker_final_distance_l52_52758

theorem hiker_final_distance :
  let east := 24
  let north := 7
  let west := 15
  let south := 5
  let net_east := east - west
  let net_north := north - south
  net_east = 9 ∧ net_north = 2 →
  Real.sqrt ((net_east)^2 + (net_north)^2) = Real.sqrt 85 :=
by
  intros
  sorry

end hiker_final_distance_l52_52758


namespace probability_of_opposite_middle_vertex_l52_52300

noncomputable def ant_moves_to_opposite_middle_vertex_prob : ℚ := 1 / 2

-- Specification of the problem conditions
structure Octahedron :=
  (middle_vertices : Finset ℕ) -- Assume some identification of middle vertices
  (adjacent_vertices : ℕ → Finset ℕ) -- Function mapping a vertex to its adjacent vertices
  (is_middle_vertex : ℕ → Prop) -- Predicate to check if a vertex is a middle vertex
  (is_top_or_bottom_vertex : ℕ → Prop) -- Predicate to check if a vertex is a top or bottom vertex
  (start_vertex : ℕ)

variables (O : Octahedron)

-- Main theorem statement
theorem probability_of_opposite_middle_vertex :
  ∃ A B : ℕ, A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B) →
  (∀ (A B : ℕ), (A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B)) →
    ant_moves_to_opposite_middle_vertex_prob = 1 / 2) := sorry

end probability_of_opposite_middle_vertex_l52_52300


namespace coefficient_x5_in_product_l52_52901

noncomputable def P : Polynomial ℤ := 
  Polynomial.C 1 * Polynomial.X ^ 6 +
  Polynomial.C (-2) * Polynomial.X ^ 5 +
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C (-6) * Polynomial.X +
  Polynomial.C 7

noncomputable def Q : Polynomial ℤ := 
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C 6 * Polynomial.X +
  Polynomial.C (-8)

theorem coefficient_x5_in_product (p q : Polynomial ℤ) :
  (p * q).coeff 5 = 2 :=
by
  have P := 
    Polynomial.C 1 * Polynomial.X ^ 6 +
    Polynomial.C (-2) * Polynomial.X ^ 5 +
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C (-6) * Polynomial.X +
    Polynomial.C 7
  have Q := 
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C 6 * Polynomial.X +
    Polynomial.C (-8)

  sorry

end coefficient_x5_in_product_l52_52901


namespace train_length_correct_l52_52769

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l52_52769


namespace people_in_circle_l52_52118

theorem people_in_circle (n : ℕ) (h : ∃ k : ℕ, k * 2 + 7 = 18) : n = 22 :=
by
  sorry

end people_in_circle_l52_52118


namespace negation_of_proposition_l52_52884

theorem negation_of_proposition :
    (¬ ∃ (x : ℝ), (Real.exp x - x - 1 < 0)) ↔ (∀ (x : ℝ), Real.exp x - x - 1 ≥ 0) :=
by
  sorry

end negation_of_proposition_l52_52884


namespace g_symmetry_value_h_m_interval_l52_52085

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 12)) ^ 2

noncomputable def g (x : ℝ) : ℝ :=
  1 + 1 / 2 * Real.sin (2 * x)

noncomputable def h (x : ℝ) : ℝ :=
  f x + g x

theorem g_symmetry_value (k : ℤ) : 
  g (k * Real.pi / 2 - Real.pi / 12) = (3 + (-1) ^ k) / 4 :=
by
  sorry

theorem h_m_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc (- Real.pi / 12) (5 * Real.pi / 12), |h x - m| ≤ 1) ↔ (1 ≤ m ∧ m ≤ 9 / 4) :=
by
  sorry

end g_symmetry_value_h_m_interval_l52_52085


namespace jerry_clock_reading_l52_52358

noncomputable def clock_reading_after_pills (pills : ℕ) (start_time : ℕ) (interval : ℕ) : ℕ :=
(start_time + (pills - 1) * interval) % 12

theorem jerry_clock_reading :
  clock_reading_after_pills 150 12 5 = 1 :=
by
  sorry

end jerry_clock_reading_l52_52358


namespace problem_statement_l52_52812

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 - x else 2 - (x % 2)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, f (x + 1) + f x = 3) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 - x) →
  f (-2007.5) = 1.5 :=
by sorry

end problem_statement_l52_52812


namespace total_pennies_l52_52965

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end total_pennies_l52_52965


namespace problem_statement_l52_52424

theorem problem_statement : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end problem_statement_l52_52424


namespace question_a_gt_b_neither_sufficient_nor_necessary_l52_52367

theorem question_a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by
  sorry

end question_a_gt_b_neither_sufficient_nor_necessary_l52_52367


namespace young_member_age_diff_l52_52024

-- Definitions
def A : ℝ := sorry    -- Average age of committee members 4 years ago
def O : ℝ := sorry    -- Age of the old member
def N : ℝ := sorry    -- Age of the new member

-- Hypotheses
axiom avg_same : ∀ (t : ℝ), t = t
axiom replacement : 10 * A + 4 * 10 - 40 = 10 * A

-- Theorem
theorem young_member_age_diff : O - N = 40 := by
  -- proof goes here
  sorry

end young_member_age_diff_l52_52024


namespace investment_scientific_notation_l52_52386

def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ (1650000000 = a * 10^n)

theorem investment_scientific_notation :
  ∃ a n, is_scientific_notation a n ∧ a = 1.65 ∧ n = 9 :=
sorry

end investment_scientific_notation_l52_52386


namespace field_trip_students_l52_52724

theorem field_trip_students 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (total_students : ℕ) 
  (h1 : seats_per_bus = 2) 
  (h2 : buses_needed = 7) 
  (h3 : total_students = seats_per_bus * buses_needed) : 
  total_students = 14 :=
by 
  rw [h1, h2] at h3
  assumption

end field_trip_students_l52_52724


namespace units_digit_k_squared_plus_two_exp_k_eq_7_l52_52369

/-- Define k as given in the problem -/
def k : ℕ := 2010^2 + 2^2010

/-- Final statement that needs to be proved -/
theorem units_digit_k_squared_plus_two_exp_k_eq_7 : (k^2 + 2^k) % 10 = 7 := 
by
  sorry

end units_digit_k_squared_plus_two_exp_k_eq_7_l52_52369


namespace fox_jeans_price_l52_52934

theorem fox_jeans_price (pony_price : ℝ)
                        (total_savings : ℝ)
                        (total_discount_rate : ℝ)
                        (pony_discount_rate : ℝ)
                        (fox_discount_rate : ℝ)
                        (fox_price : ℝ) :
    pony_price = 18 ∧
    total_savings = 8.91 ∧
    total_discount_rate = 0.22 ∧
    pony_discount_rate = 0.1099999999999996 ∧
    fox_discount_rate = 0.11 →
    (3 * fox_discount_rate * fox_price + 2 * pony_discount_rate * pony_price = total_savings) →
    fox_price = 15 :=
by
  intros h h_eq
  rcases h with ⟨h_pony, h_savings, h_total_rate, h_pony_rate, h_fox_rate⟩
  sorry

end fox_jeans_price_l52_52934


namespace books_count_l52_52599

theorem books_count (Tim_books Total_books Mike_books : ℕ) (h1 : Tim_books = 22) (h2 : Total_books = 42) : Mike_books = 20 :=
by
  sorry

end books_count_l52_52599


namespace Claire_plans_to_buy_five_cookies_l52_52470

theorem Claire_plans_to_buy_five_cookies :
  let initial_amount := 100
  let latte_cost := 3.75
  let croissant_cost := 3.50
  let days := 7
  let cookie_cost := 1.25
  let remaining_amount := 43
  let daily_expense := latte_cost + croissant_cost
  let weekly_expense := daily_expense * days
  let total_spent := initial_amount - remaining_amount
  let cookie_spent := total_spent - weekly_expense
  let cookies := cookie_spent / cookie_cost
  cookies = 5 :=
by {
  sorry
}

end Claire_plans_to_buy_five_cookies_l52_52470


namespace Roshesmina_pennies_l52_52966

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end Roshesmina_pennies_l52_52966
