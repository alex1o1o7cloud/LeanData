import Mathlib

namespace sunil_total_amount_l2392_239270

noncomputable def principal (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  CI / ((1 + R / 100) ^ T - 1)

noncomputable def total_amount (CI : ℝ) (R : ℝ) (T : ℕ) : ℝ :=
  let P := principal CI R T
  P + CI

theorem sunil_total_amount (CI : ℝ) (R : ℝ) (T : ℕ) :
  CI = 420 → R = 10 → T = 2 → total_amount CI R T = 2420 := by
  intros hCI hR hT
  rw [hCI, hR, hT]
  sorry

end sunil_total_amount_l2392_239270


namespace line_passes_through_point_has_correct_equation_l2392_239267

theorem line_passes_through_point_has_correct_equation :
  (∃ (L : ℝ × ℝ → Prop), (L (-2, 5)) ∧ (∃ m : ℝ, m = -3 / 4 ∧ ∀ (x y : ℝ), L (x, y) ↔ y - 5 = -3 / 4 * (x + 2))) →
  ∀ x y : ℝ, (3 * x + 4 * y - 14 = 0) ↔ (y - 5 = -3 / 4 * (x + 2)) :=
by
  intro h_L
  sorry

end line_passes_through_point_has_correct_equation_l2392_239267


namespace relationship_of_arithmetic_progression_l2392_239255

theorem relationship_of_arithmetic_progression (x y z d : ℝ) (h1 : x + (y - z) + d = y + (z - x))
    (h2 : y + (z - x) + d = z + (x - y))
    (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
    x = y + d / 2 ∧ z = y + d := by
  sorry

end relationship_of_arithmetic_progression_l2392_239255


namespace solution_set_f_2_minus_x_l2392_239263

def f (x : ℝ) (a : ℝ) (b : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_f_2_minus_x (a b : ℝ) (h_even : b - 2 * a = 0)
  (h_mono : 0 < a) :
  {x : ℝ | f (2 - x) a b > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_2_minus_x_l2392_239263


namespace abs_gt_x_iff_x_lt_0_l2392_239208

theorem abs_gt_x_iff_x_lt_0 (x : ℝ) : |x| > x ↔ x < 0 := 
by
  sorry

end abs_gt_x_iff_x_lt_0_l2392_239208


namespace sarahs_score_l2392_239273

theorem sarahs_score (g s : ℕ) (h1 : s = g + 60) (h2 : s + g = 260) : s = 160 :=
sorry

end sarahs_score_l2392_239273


namespace C_gets_more_than_D_l2392_239292

-- Define the conditions
def proportion_B := 3
def share_B : ℕ := 3000
def proportion_C := 5
def proportion_D := 4

-- Define the parts based on B's share
def part_value := share_B / proportion_B

-- Define the shares based on the proportions
def share_C := proportion_C * part_value
def share_D := proportion_D * part_value

-- Prove the final statement about the difference
theorem C_gets_more_than_D : share_C - share_D = 1000 :=
by
  -- Proof goes here
  sorry

end C_gets_more_than_D_l2392_239292


namespace numSpaceDiagonals_P_is_241_l2392_239234

noncomputable def numSpaceDiagonals (vertices : ℕ) (edges : ℕ) (tri_faces : ℕ) (quad_faces : ℕ) : ℕ :=
  let total_segments := (vertices * (vertices - 1)) / 2
  let face_diagonals := 2 * quad_faces
  total_segments - edges - face_diagonals

theorem numSpaceDiagonals_P_is_241 :
  numSpaceDiagonals 26 60 24 12 = 241 := by 
  sorry

end numSpaceDiagonals_P_is_241_l2392_239234


namespace Milly_took_extra_balloons_l2392_239236

theorem Milly_took_extra_balloons :
  let total_packs := 3 + 2
  let balloons_per_pack := 6
  let total_balloons := total_packs * balloons_per_pack
  let even_split := total_balloons / 2
  let Floretta_balloons := 8
  let Milly_extra_balloons := even_split - Floretta_balloons
  Milly_extra_balloons = 7 := by
  sorry

end Milly_took_extra_balloons_l2392_239236


namespace sara_pumpkins_l2392_239206

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end sara_pumpkins_l2392_239206


namespace rajas_income_l2392_239282

theorem rajas_income (I : ℝ) 
  (h1 : 0.60 * I + 0.10 * I + 0.10 * I + 5000 = I) : I = 25000 :=
by
  sorry

end rajas_income_l2392_239282


namespace tan_alpha_eq_neg2_l2392_239218

theorem tan_alpha_eq_neg2 {α : ℝ} {x y : ℝ} (hx : x = -2) (hy : y = 4) (hM : (x, y) = (-2, 4)) :
  Real.tan α = -2 :=
by
  sorry

end tan_alpha_eq_neg2_l2392_239218


namespace neg_distance_represents_west_l2392_239212

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end neg_distance_represents_west_l2392_239212


namespace latte_cost_l2392_239288

theorem latte_cost :
  ∃ (latte_cost : ℝ), 
    2 * 2.25 + 3.50 + 0.50 + 2 * 2.50 + 3.50 + 2 * latte_cost = 25.00 ∧ 
    latte_cost = 4.00 :=
by
  use 4.00
  simp
  sorry

end latte_cost_l2392_239288


namespace determine_valid_m_l2392_239281

-- The function given in the problem
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + m + 2

-- The range of values for m
def valid_m (m : ℝ) : Prop := -1/4 ≤ m ∧ m ≤ 0

-- The condition that f is increasing on (-∞, 2)
def increasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < a → f x ≤ f y

-- The main statement we want to prove
theorem determine_valid_m (m : ℝ) :
  increasing_on_interval (f m) 2 ↔ valid_m m :=
sorry

end determine_valid_m_l2392_239281


namespace trig_proof_1_trig_proof_2_l2392_239266

variables {α : ℝ}

-- Given condition
def tan_alpha (a : ℝ) := Real.tan a = -3

-- Proof problem statement
theorem trig_proof_1 (h : tan_alpha α) :
  (3 * Real.sin α - 3 * Real.cos α) / (6 * Real.cos α + Real.sin α) = -4 := sorry

theorem trig_proof_2 (h : tan_alpha α) :
  1 / (Real.sin α * Real.cos α + 1 + Real.cos (2 * α)) = -10 := sorry

end trig_proof_1_trig_proof_2_l2392_239266


namespace tan_theta_value_l2392_239209

theorem tan_theta_value (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 :=
sorry

end tan_theta_value_l2392_239209


namespace solve_negative_integer_sum_l2392_239202

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l2392_239202


namespace dilation_result_l2392_239230

noncomputable def dilation (c a : ℂ) (k : ℝ) : ℂ := k * (c - a) + a

theorem dilation_result :
  dilation (3 - 1* I) (1 + 2* I) 4 = 9 + 6* I :=
by
  sorry

end dilation_result_l2392_239230


namespace polygon_diagonals_with_restricted_vertices_l2392_239217

theorem polygon_diagonals_with_restricted_vertices
  (vertices : ℕ) (non_contributing_vertices : ℕ)
  (h_vertices : vertices = 35)
  (h_non_contributing_vertices : non_contributing_vertices = 5) :
  (vertices - non_contributing_vertices) * (vertices - non_contributing_vertices - 3) / 2 = 405 :=
by {
  sorry
}

end polygon_diagonals_with_restricted_vertices_l2392_239217


namespace smaller_pack_size_l2392_239216

theorem smaller_pack_size {x : ℕ} (total_eggs large_pack_size large_packs : ℕ) (eggs_in_smaller_packs : ℕ) :
  total_eggs = 79 → large_pack_size = 11 → large_packs = 5 → eggs_in_smaller_packs = total_eggs - large_pack_size * large_packs →
  x * 1 = eggs_in_smaller_packs → x = 24 :=
by sorry

end smaller_pack_size_l2392_239216


namespace johanna_loses_half_turtles_l2392_239296

theorem johanna_loses_half_turtles
  (owen_turtles_initial : ℕ)
  (johanna_turtles_fewer : ℕ)
  (owen_turtles_after_month : ℕ)
  (owen_turtles_final : ℕ)
  (johanna_donates_rest_to_owen : ℚ → ℚ)
  (x : ℚ)
  (hx1 : owen_turtles_initial = 21)
  (hx2 : johanna_turtles_fewer = 5)
  (hx3 : owen_turtles_after_month = owen_turtles_initial * 2)
  (hx4 : owen_turtles_final = owen_turtles_after_month + johanna_donates_rest_to_owen (1 - x))
  (hx5 : owen_turtles_final = 50) :
  x = 1 / 2 :=
by
  sorry

end johanna_loses_half_turtles_l2392_239296


namespace sum_of_squares_l2392_239232

variable (a b c : ℝ)
variable (S : ℝ)

theorem sum_of_squares (h1 : ab + bc + ac = 131)
                       (h2 : a + b + c = 22) :
  a^2 + b^2 + c^2 = 222 :=
by
  -- Proof would be placed here
  sorry

end sum_of_squares_l2392_239232


namespace largest_A_l2392_239243

namespace EquivalentProofProblem

def F (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f (3 * x) ≥ f (f (2 * x)) + x

theorem largest_A (f : ℝ → ℝ) (hf : F f) (x : ℝ) (hx : x > 0) : 
  ∃ A, (∀ (f : ℝ → ℝ), F f → ∀ x, x > 0 → f x ≥ A * x) ∧ A = 1 / 2 :=
sorry

end EquivalentProofProblem

end largest_A_l2392_239243


namespace speed_of_first_boy_l2392_239260

theorem speed_of_first_boy (x : ℝ) (h1 : 7.5 > 0) (h2 : 16 > 0) (h3 : 32 > 0) (h4 : 32 = 16 * (x - 7.5)) : x = 9.5 :=
by
  sorry

end speed_of_first_boy_l2392_239260


namespace speed_of_man_l2392_239248

theorem speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ)
  (relative_speed_km_h : ℝ)
  (h_train_length : train_length = 440)
  (h_train_speed : train_speed_kmph = 60)
  (h_time : time_seconds = 24)
  (h_relative_speed : relative_speed_km_h = (train_length / time_seconds) * 3.6):
  (relative_speed_km_h - train_speed_kmph) = 6 :=
by sorry

end speed_of_man_l2392_239248


namespace tallest_is_Justina_l2392_239264

variable (H G I J K : ℝ)

axiom height_conditions1 : H < G
axiom height_conditions2 : G < J
axiom height_conditions3 : K < I
axiom height_conditions4 : I < G

theorem tallest_is_Justina : J > G ∧ J > H ∧ J > I ∧ J > K :=
by
  sorry

end tallest_is_Justina_l2392_239264


namespace find_a_f_greater_than_1_l2392_239262

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) := x^2 * Real.exp x - a * Real.log x

-- Condition: Slope at x = 1 is 3e - 1
theorem find_a (a : ℝ) (h : deriv (fun x => f x a) 1 = 3 * Real.exp 1 - 1) : a = 1 := sorry

-- Given a = 1
theorem f_greater_than_1 (x : ℝ) (hx : x > 0) : f x 1 > 1 := sorry

end find_a_f_greater_than_1_l2392_239262


namespace greatest_N_consecutive_sum_50_l2392_239251

theorem greatest_N_consecutive_sum_50 :
  ∃ N a : ℤ, (N > 0) ∧ (N * (2 * a + N - 1) = 100) ∧ (N = 100) :=
by
  sorry

end greatest_N_consecutive_sum_50_l2392_239251


namespace two_a_minus_b_l2392_239226

-- Definitions of vector components and parallelism condition
def is_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0
def vector_a : ℝ × ℝ := (1, -2)

-- Given assumptions
variable (m : ℝ)
def vector_b : ℝ × ℝ := (m, 4)

-- Theorem statement
theorem two_a_minus_b (h : is_parallel vector_a (vector_b m)) : 2 • vector_a - vector_b m = (4, -8) :=
sorry

end two_a_minus_b_l2392_239226


namespace part1_part2_l2392_239268

noncomputable def f (x a : ℝ) : ℝ := |(x - a)| + |(x + 2)|

-- Part (1)
theorem part1 (x : ℝ) (h : f x 1 ≤ 7) : -4 ≤ x ∧ x ≤ 3 :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2 * a + 1) : a ≤ 1 :=
by
  sorry

end part1_part2_l2392_239268


namespace rectangles_260_261_272_273_have_similar_property_l2392_239214

-- Defining a rectangle as a structure with width and height
structure Rectangle where
  width : ℕ
  height : ℕ

-- Given conditions
def r1 : Rectangle := ⟨16, 10⟩
def r2 : Rectangle := ⟨23, 7⟩

-- Hypothesis function indicating the dissection trick causing apparent equality
def dissection_trick (r1 r2 : Rectangle) : Prop :=
  (r1.width * r1.height : ℕ) = (r2.width * r2.height : ℕ) + 1

-- The statement of the proof problem
theorem rectangles_260_261_272_273_have_similar_property :
  ∃ (r3 r4 : Rectangle) (r5 r6 : Rectangle),
    dissection_trick r3 r4 ∧ dissection_trick r5 r6 ∧
    r3.width * r3.height = 260 ∧ r4.width * r4.height = 261 ∧
    r5.width * r5.height = 272 ∧ r6.width * r6.height = 273 :=
  sorry

end rectangles_260_261_272_273_have_similar_property_l2392_239214


namespace spiral_2018_position_l2392_239233

def T100_spiral : Matrix ℕ ℕ ℕ := sorry -- Definition of T100 as a spiral matrix

def pos_2018 := (34, 95) -- The given position we need to prove

theorem spiral_2018_position (i j : ℕ) (h₁ : T100_spiral 34 95 = 2018) : (i, j) = pos_2018 := by  
  sorry

end spiral_2018_position_l2392_239233


namespace total_heads_l2392_239252

def total_legs : ℕ := 45
def num_cats : ℕ := 7
def legs_per_cat : ℕ := 4
def captain_legs : ℕ := 1
def legs_humans := total_legs - (num_cats * legs_per_cat) - captain_legs
def num_humans := legs_humans / 2

theorem total_heads : (num_cats + (num_humans + 1)) = 15 := by
  sorry

end total_heads_l2392_239252


namespace arithmetic_sequence_a2015_l2392_239280

theorem arithmetic_sequence_a2015 :
  (∀ n : ℕ, n > 0 → (∃ a_n a_n1 : ℝ,
    a_n1 = a_n + 2 ∧ a_n + a_n1 = 4 * n - 58))
  → (∃ a_2015 : ℝ, a_2015 = 4000) :=
by
  intro h
  sorry

end arithmetic_sequence_a2015_l2392_239280


namespace susan_hours_per_day_l2392_239211

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end susan_hours_per_day_l2392_239211


namespace book_vs_necklace_price_difference_l2392_239265

-- Problem-specific definitions and conditions
def necklace_price : ℕ := 34
def limit_price : ℕ := 70
def overspent : ℕ := 3
def total_spent : ℕ := limit_price + overspent
def book_price : ℕ := total_spent - necklace_price

-- Lean statement to prove the correct answer
theorem book_vs_necklace_price_difference :
  book_price - necklace_price = 5 := by
  sorry

end book_vs_necklace_price_difference_l2392_239265


namespace lockers_remaining_open_l2392_239261

-- Define the number of lockers and students
def num_lockers : ℕ := 1000

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to count perfect squares up to a given number
def count_perfect_squares_up_to (n : ℕ) : ℕ :=
  Nat.sqrt n

-- Theorem statement
theorem lockers_remaining_open : 
  count_perfect_squares_up_to num_lockers = 31 :=
by
  -- Proof left out because it's not necessary to provide
  sorry

end lockers_remaining_open_l2392_239261


namespace roots_opposite_signs_l2392_239223

theorem roots_opposite_signs (a b c: ℝ) 
  (h1 : (b^2 - a * c) > 0)
  (h2 : (b^4 - a^2 * c^2) < 0) :
  a * c < 0 :=
sorry

end roots_opposite_signs_l2392_239223


namespace number_of_members_l2392_239220

theorem number_of_members (n : ℕ) (h : n^2 = 9216) : n = 96 :=
sorry

end number_of_members_l2392_239220


namespace infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l2392_239254

noncomputable def grid_size := 10
noncomputable def initial_infected_count_1 := 9
noncomputable def initial_infected_count_2 := 10

def condition (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n → 
  infected + steps * (infected / 2) < grid_size * grid_size

def can_infect_entire_grid (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n ∧ (
  ∃ t : ℕ, infected + t * (infected / 2) = grid_size * grid_size)

theorem infection_does_not_spread_with_9_cells :
  ¬ can_infect_entire_grid initial_infected_count_1 :=
by
  sorry

theorem minimum_infected_cells_needed :
  condition initial_infected_count_2 :=
by
  sorry

end infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l2392_239254


namespace quadrilateral_area_l2392_239286

theorem quadrilateral_area (c d : ℤ) (h1 : 0 < d) (h2 : d < c) (h3 : 2 * ((c : ℝ) ^ 2 - (d : ℝ) ^ 2) = 18) : 
  c + d = 9 :=
by
  sorry

end quadrilateral_area_l2392_239286


namespace unique_nets_of_a_cube_l2392_239201

-- Definitions based on the conditions and the properties of the cube
def is_net (net: ℕ) : Prop :=
  -- A placeholder definition of a valid net
  sorry

def is_distinct_by_rotation_or_reflection (net1 net2: ℕ) : Prop :=
  -- Two nets are distinct if they cannot be transformed into each other by rotation or reflection
  sorry

-- The statement to be proved
theorem unique_nets_of_a_cube : ∃ n, n = 11 ∧ (∀ net, is_net net → ∃! net', is_net net' ∧ is_distinct_by_rotation_or_reflection net net') :=
sorry

end unique_nets_of_a_cube_l2392_239201


namespace sum_of_coefficients_l2392_239253

noncomputable def u : ℕ → ℕ
| 0       => 5
| (n + 1) => u n + (3 + 4 * (n - 1))

theorem sum_of_coefficients :
  (2 + -3 + 6) = 5 :=
by {
  sorry
}

end sum_of_coefficients_l2392_239253


namespace distance_between_parallel_lines_eq_l2392_239205

open Real

theorem distance_between_parallel_lines_eq
  (h₁ : ∀ (x y : ℝ), 3 * x + y - 3 = 0 → Prop)
  (h₂ : ∀ (x y : ℝ), 6 * x + 2 * y + 1 = 0 → Prop) :
  ∃ d : ℝ, d = (7 / 20) * sqrt 10 :=
sorry

end distance_between_parallel_lines_eq_l2392_239205


namespace problem_a_problem_b_l2392_239229

-- Define the polynomial P(x) = ax^3 + bx
def P (a b x : ℤ) : ℤ := a * x^3 + b * x

-- Define when a pair (a, b) is n-good
def is_ngood (n a b : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ P a b m - P a b k → n ∣ m - k

-- Define when a pair (a, b) is very good
def is_verygood (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ (infinitely_many_n : ℕ), is_ngood n a b

-- Problem (a): Find a pair (1, -51^2) which is 51-good but not very good
theorem problem_a : ∃ a b : ℤ, a = 1 ∧ b = -(51^2) ∧ is_ngood 51 a b ∧ ¬is_verygood a b := 
by {
  sorry
}

-- Problem (b): Show that all 2010-good pairs are very good
theorem problem_b : ∀ a b : ℤ, is_ngood 2010 a b → is_verygood a b := 
by {
  sorry
}

end problem_a_problem_b_l2392_239229


namespace Winnie_lollipops_remain_l2392_239256

theorem Winnie_lollipops_remain :
  let cherry_lollipops := 45
  let wintergreen_lollipops := 116
  let grape_lollipops := 4
  let shrimp_cocktail_lollipops := 229
  let total_lollipops := cherry_lollipops + wintergreen_lollipops + grape_lollipops + shrimp_cocktail_lollipops
  let friends := 11
  total_lollipops % friends = 9 :=
by
  sorry

end Winnie_lollipops_remain_l2392_239256


namespace arithmetic_sqrt_25_l2392_239241

-- Define the arithmetic square root condition
def is_arithmetic_sqrt (x a : ℝ) : Prop :=
  0 ≤ x ∧ x^2 = a

-- Lean statement to prove the arithmetic square root of 25 is 5
theorem arithmetic_sqrt_25 : is_arithmetic_sqrt 5 25 :=
by 
  sorry

end arithmetic_sqrt_25_l2392_239241


namespace simplify_fraction_l2392_239271

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l2392_239271


namespace product_divisible_by_10_l2392_239289

noncomputable def probability_divisible_by_10 (n : ℕ) (h : n > 1) : ℝ :=
  1 - (8^n + 5^n - 4^n) / 9^n

theorem product_divisible_by_10 (n : ℕ) (h : n > 1) :
  probability_divisible_by_10 n h = 1 - (8^n + 5^n - 4^n)/(9^n) :=
by
  sorry

end product_divisible_by_10_l2392_239289


namespace no_such_natural_number_exists_l2392_239293

theorem no_such_natural_number_exists :
  ¬ ∃ (n : ℕ), (∃ (m k : ℤ), 2 * n - 5 = 9 * m ∧ n - 2 = 15 * k) :=
by
  sorry

end no_such_natural_number_exists_l2392_239293


namespace greatest_possible_sum_l2392_239297

noncomputable def eight_products_sum_max : ℕ :=
  let a := 3
  let b := 4
  let c := 5
  let d := 8
  let e := 6
  let f := 7
  7 * (c + d) * (e + f)

theorem greatest_possible_sum (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) :
  eight_products_sum_max = 1183 :=
by
  sorry

end greatest_possible_sum_l2392_239297


namespace segment_length_reflection_l2392_239257

theorem segment_length_reflection (F : ℝ × ℝ) (F' : ℝ × ℝ)
  (hF : F = (-4, -2)) (hF' : F' = (4, -2)) :
  dist F F' = 8 :=
by
  sorry

end segment_length_reflection_l2392_239257


namespace average_GPA_of_class_l2392_239246

theorem average_GPA_of_class (n : ℕ) (h1 : n > 0) 
  (GPA1 : ℝ := 60) (GPA2 : ℝ := 66) 
  (students_ratio1 : ℝ := 1 / 3) (students_ratio2 : ℝ := 2 / 3) :
  let total_students := (students_ratio1 * n + students_ratio2 * n)
  let total_GPA := (students_ratio1 * n * GPA1 + students_ratio2 * n * GPA2)
  let average_GPA := total_GPA / total_students
  average_GPA = 64 := by
    sorry

end average_GPA_of_class_l2392_239246


namespace probability_of_experts_winning_l2392_239250

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l2392_239250


namespace alice_bob_meet_l2392_239210

theorem alice_bob_meet (t : ℝ) 
(h1 : ∀ s : ℝ, s = 30 * t) 
(h2 : ∀ b : ℝ, b = 29.5 * 60 ∨ b = 30.5 * 60)
(h3 : ∀ a : ℝ, a = 30 * t)
(h4 : ∀ a b : ℝ, a = b):
(t = 59 ∨ t = 61) :=
by
  sorry

end alice_bob_meet_l2392_239210


namespace reinforcement_arrival_days_l2392_239244

theorem reinforcement_arrival_days (x : ℕ) (h : x = 2000) (provisions_days : ℕ) (provisions_days_initial : provisions_days = 54) 
(reinforcement : ℕ) (reinforcement_val : reinforcement = 1300) (remaining_days : ℕ) (remaining_days_val : remaining_days = 20) 
(total_men : ℕ) (total_men_val : total_men = 3300) (equation : 2000 * (54 - x) = 3300 * 20) : x = 21 := 
by
  have eq1 : 2000 * 54 - 2000 * x = 3300 * 20 := by sorry
  have eq2 : 108000 - 2000 * x = 66000 := by sorry
  have eq3 : 2000 * x = 42000 := by sorry
  have eq4 : x = 21000 / 2000 := by sorry
  have eq5 : x = 21 := by sorry
  sorry

end reinforcement_arrival_days_l2392_239244


namespace sum_of_integers_l2392_239221

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 54) : x + y = 21 :=
by
  sorry

end sum_of_integers_l2392_239221


namespace solution_set_of_even_function_l2392_239225

theorem solution_set_of_even_function (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) 
  (h_def : ∀ x, 0 < x → f x = x^2 - 2*x - 3) : 
  { x : ℝ | f x > 0 } = { x | x > 3 } ∪ { x | x < -3 } :=
sorry

end solution_set_of_even_function_l2392_239225


namespace number_of_lines_with_negative_reciprocal_intercepts_l2392_239283

-- Define the point (-2, 4)
def point : ℝ × ℝ := (-2, 4)

-- Define the condition that intercepts are negative reciprocals
def are_negative_reciprocals (a b : ℝ) : Prop :=
  a * b = -1

-- Define the proof problem: 
-- Number of lines through point (-2, 4) with intercepts negative reciprocals of each other
theorem number_of_lines_with_negative_reciprocal_intercepts :
  ∃ n : ℕ, n = 2 ∧ 
  ∀ (a b : ℝ), are_negative_reciprocals a b →
  (∃ m k : ℝ, (k * (-2) + m = 4) ∧ ((m ⁻¹ = a ∧ k = b) ∨ (k = a ∧ m ⁻¹ = b))) :=
sorry

end number_of_lines_with_negative_reciprocal_intercepts_l2392_239283


namespace ab_value_l2392_239231

/-- 
  Given the conditions:
  - a - b = 10
  - a^2 + b^2 = 210
  Prove that ab = 55.
-/
theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 210) : a * b = 55 :=
by
  sorry

end ab_value_l2392_239231


namespace arithmetic_expression_value_l2392_239228

theorem arithmetic_expression_value :
  68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 :=
by
  sorry

end arithmetic_expression_value_l2392_239228


namespace ab_minus_a_plus_b_eq_two_l2392_239294

theorem ab_minus_a_plus_b_eq_two
  (a b : ℝ)
  (h1 : a + 1 ≠ 0)
  (h2 : b - 1 ≠ 0)
  (h3 : a + (1 / (a + 1)) = b + (1 / (b - 1)) - 2)
  (h4 : a - b + 2 ≠ 0)
: ab - a + b = 2 :=
sorry

end ab_minus_a_plus_b_eq_two_l2392_239294


namespace leila_money_left_l2392_239227

theorem leila_money_left (initial_money spent_on_sweater spent_on_jewelry total_spent left_money : ℕ) 
  (h1 : initial_money = 160) 
  (h2 : spent_on_sweater = 40) 
  (h3 : spent_on_jewelry = 100) 
  (h4 : total_spent = spent_on_sweater + spent_on_jewelry) 
  (h5 : total_spent = 140) : 
  initial_money - total_spent = 20 := by
  sorry

end leila_money_left_l2392_239227


namespace intersection_M_N_l2392_239219

-- Definitions of the domains M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | x > 0}

-- The goal is to prove that the intersection of M and N is equal to (0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l2392_239219


namespace todd_initial_gum_l2392_239291

theorem todd_initial_gum (x : ℝ)
(h1 : 150 = 0.25 * x)
(h2 : x + 150 = 890) :
x = 712 :=
by
  -- Here "by" is used to denote the beginning of proof block
  sorry -- Proof will be filled in later.

end todd_initial_gum_l2392_239291


namespace quadrilateral_area_l2392_239285

theorem quadrilateral_area (a b c d : ℝ) (horizontally_vertically_apart : a = b + 1 ∧ b = c + 1 ∧ c = d + 1 ∧ d = a + 1) : 
  area_of_quadrilateral = 6 :=
sorry

end quadrilateral_area_l2392_239285


namespace even_multiples_of_25_l2392_239207

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0

theorem even_multiples_of_25 (a b : ℕ) (h1 : 249 ≤ a) (h2 : b ≤ 501) :
  (a = 250 ∨ a = 275 ∨ a = 300 ∨ a = 350 ∨ a = 400 ∨ a = 450) →
  (b = 275 ∨ b = 300 ∨ b = 350 ∨ b = 400 ∨ b = 450 ∨ b = 500) →
  (∃ n, n = 5 ∧ ∀ m, (is_multiple_of_25 m ∧ is_even m ∧ a ≤ m ∧ m ≤ b) ↔ m ∈ [a, b]) :=
by sorry

end even_multiples_of_25_l2392_239207


namespace ordered_triple_exists_l2392_239275

theorem ordered_triple_exists (a b c : ℝ) (h₁ : 4 < a) (h₂ : 4 < b) (h₃ : 4 < c)
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) :=
sorry

end ordered_triple_exists_l2392_239275


namespace complement_intersection_l2392_239200

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection (hU : ∀ x, x ∈ U) (hA : ∀ x, x ∈ A) (hB : ∀ x, x ∈ B) :
    (U \ A) ∩ (U \ B) = {7, 9} :=
by
  sorry

end complement_intersection_l2392_239200


namespace total_seats_l2392_239237

theorem total_seats (s : ℕ) 
  (first_class : ℕ := 30) 
  (business_class : ℕ := (20 * s) / 100) 
  (premium_economy : ℕ := 15) 
  (economy_class : ℕ := s - first_class - business_class - premium_economy) 
  (total : first_class + business_class + premium_economy + economy_class = s) 
  : s = 288 := 
sorry

end total_seats_l2392_239237


namespace increased_speed_l2392_239277

theorem increased_speed
  (d : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) 
  (h1 : d = 2) 
  (h2 : s1 = 2) 
  (h3 : t1 = 1)
  (h4 : t2 = 2 / 3)
  (h5 : s1 * t1 = d)
  (h6 : s2 * t2 = d) :
  s2 - s1 = 1 := 
sorry

end increased_speed_l2392_239277


namespace root_difference_l2392_239278

theorem root_difference (p : ℝ) (r s : ℝ) :
  (r + s = p) ∧ (r * s = (p^2 - 1) / 4) ∧ (r ≥ s) → r - s = 1 :=
by
  intro h
  sorry

end root_difference_l2392_239278


namespace identical_digits_time_l2392_239274

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l2392_239274


namespace eq1_solution_eq2_solution_l2392_239222

theorem eq1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 :=
by
  sorry

theorem eq2_solution (x : ℝ) (h : (1 / 2) * x - 6 = (3 / 4) * x) : x = -24 :=
by
  sorry

end eq1_solution_eq2_solution_l2392_239222


namespace interest_earned_l2392_239249

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) := P * (1 + r) ^ t

theorem interest_earned :
  let P := 2000
  let r := 0.05
  let t := 5
  let A := compound_interest P r t
  A - P = 552.56 :=
by
  sorry

end interest_earned_l2392_239249


namespace positive_integers_sum_digits_less_than_9000_l2392_239204

theorem positive_integers_sum_digits_less_than_9000 : 
  ∃ n : ℕ, n = 47 ∧ ∀ x : ℕ, (1 ≤ x ∧ x < 9000 ∧ (Nat.digits 10 x).sum = 5) → (Nat.digits 10 x).length = n :=
sorry

end positive_integers_sum_digits_less_than_9000_l2392_239204


namespace straight_line_cannot_intersect_all_segments_l2392_239295

/-- A broken line in the plane with 11 segments -/
structure BrokenLine :=
(segments : Fin 11 → (ℝ × ℝ) × (ℝ × ℝ))
(closed_chain : ∀ i : Fin 11, i.val < 10 → (segments ⟨i.val + 1, sorry⟩).fst = (segments i).snd)

/-- A straight line that doesn't contain the vertices of the broken line -/
structure StraightLine :=
(is_not_vertex : (ℝ × ℝ) → Prop)

/-- The main theorem stating the impossibility of a straight line intersecting all segments -/
theorem straight_line_cannot_intersect_all_segments (line : StraightLine) (brokenLine: BrokenLine) :
  ∃ i : Fin 11, ¬∃ t : ℝ, ∃ x y : ℝ, 
    brokenLine.segments i = ((x, y), (x + t, y + t)) ∧ 
    ¬line.is_not_vertex (x, y) ∧ 
    ¬line.is_not_vertex (x + t, y + t) :=
sorry

end straight_line_cannot_intersect_all_segments_l2392_239295


namespace gcd_a_b_eq_one_l2392_239299

def a : ℕ := 123^2 + 235^2 + 347^2
def b : ℕ := 122^2 + 234^2 + 348^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_a_b_eq_one_l2392_239299


namespace cost_of_water_l2392_239276

theorem cost_of_water (total_cost sandwiches_cost : ℕ) (num_sandwiches sandwich_price water_price : ℕ) 
  (h1 : total_cost = 11) 
  (h2 : sandwiches_cost = num_sandwiches * sandwich_price) 
  (h3 : num_sandwiches = 3) 
  (h4 : sandwich_price = 3) 
  (h5 : total_cost = sandwiches_cost + water_price) : 
  water_price = 2 :=
by
  sorry

end cost_of_water_l2392_239276


namespace rectangle_area_l2392_239215

theorem rectangle_area (a b c : ℝ) :
  a = 15 ∧ b = 12 ∧ c = 1 / 3 →
  ∃ (AD AB : ℝ), 
  AD = (180 / 17) ∧ AB = (60 / 17) ∧ 
  (AD * AB = 10800 / 289) :=
by sorry

end rectangle_area_l2392_239215


namespace car_cost_l2392_239239

-- Define the weekly allowance in the first year
def first_year_allowance_weekly : ℕ := 50

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Calculate the total first year savings
def first_year_savings : ℕ := first_year_allowance_weekly * weeks_in_year

-- Define the hourly wage and weekly hours worked in the second year
def hourly_wage : ℕ := 9
def weekly_hours_worked : ℕ := 30

-- Calculate the total second year earnings
def second_year_earnings : ℕ := hourly_wage * weekly_hours_worked * weeks_in_year

-- Define the weekly spending in the second year
def weekly_spending : ℕ := 35

-- Calculate the total second year spending
def second_year_spending : ℕ := weekly_spending * weeks_in_year

-- Calculate the total second year savings
def second_year_savings : ℕ := second_year_earnings - second_year_spending

-- Calculate the total savings after two years
def total_savings : ℕ := first_year_savings + second_year_savings

-- Define the additional amount needed
def additional_amount_needed : ℕ := 2000

-- Calculate the total cost of the car
def total_cost_of_car : ℕ := total_savings + additional_amount_needed

-- Theorem statement
theorem car_cost : total_cost_of_car = 16820 := by
  -- The proof is omitted; it is enough to state the theorem
  sorry

end car_cost_l2392_239239


namespace Donggil_cleaning_time_l2392_239240

-- Define the total area of the school as A.
variable (A : ℝ)

-- Define the cleaning rates of Daehyeon (D) and Donggil (G).
variable (D G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (D + G) * 8 = (7 / 12) * A
def condition2 : Prop := D * 10 = (5 / 12) * A

-- The goal is to prove that Donggil can clean the entire area alone in 32 days.
theorem Donggil_cleaning_time : condition1 A D G ∧ condition2 A D → 32 * G = A :=
by
  sorry

end Donggil_cleaning_time_l2392_239240


namespace find_principal_sum_l2392_239287

theorem find_principal_sum (P : ℝ) (r : ℝ) (A2 : ℝ) (A3 : ℝ) : 
  (A2 = 7000) → (A3 = 9261) → 
  (A2 = P * (1 + r)^2) → (A3 = P * (1 + r)^3) → 
  P = 4000 :=
by
  intro hA2 hA3 hA2_eq hA3_eq
  -- here, we assume the proof steps leading to P = 4000
  sorry

end find_principal_sum_l2392_239287


namespace three_digit_sum_seven_l2392_239272

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l2392_239272


namespace range_of_m_l2392_239259

theorem range_of_m (x m : ℝ) (h1 : 2 * x - m ≤ 3) (h2 : -5 < x) (h3 : x < 4) :
  ∃ m, ∀ (x : ℝ), (-5 < x ∧ x < 4) → (2 * x - m ≤ 3) ↔ (m ≥ 5) :=
by sorry

end range_of_m_l2392_239259


namespace mushroom_problem_l2392_239258

variables (x1 x2 x3 x4 : ℕ)

theorem mushroom_problem
  (h1 : x1 + x2 = 6)
  (h2 : x1 + x3 = 7)
  (h3 : x2 + x3 = 9)
  (h4 : x2 + x4 = 11)
  (h5 : x3 + x4 = 12)
  (h6 : x1 + x4 = 9) :
  x1 = 2 ∧ x2 = 4 ∧ x3 = 5 ∧ x4 = 7 := 
  by
    sorry

end mushroom_problem_l2392_239258


namespace classify_curve_l2392_239224

-- Define the curve equation
def curve_equation (m : ℝ) : Prop := 
  ∃ (x y : ℝ), ((m - 3) * x^2 + (5 - m) * y^2 = 1)

-- Define the conditions for types of curves
def is_circle (m : ℝ) : Prop := 
  m = 4 ∧ (curve_equation m)

def is_ellipse (m : ℝ) : Prop := 
  (3 < m ∧ m < 5 ∧ m ≠ 4) ∧ (curve_equation m)

def is_hyperbola (m : ℝ) : Prop := 
  ((m > 5 ∨ m < 3) ∧ (curve_equation m))

-- Main theorem stating the type of curve
theorem classify_curve (m : ℝ) : 
  (is_circle m) ∨ (is_ellipse m) ∨ (is_hyperbola m) :=
sorry

end classify_curve_l2392_239224


namespace no_positive_integer_n_exists_l2392_239238

theorem no_positive_integer_n_exists {n : ℕ} (hn : n > 0) :
  ¬ ((∃ k, 5 * 10^(k - 1) ≤ 2^n ∧ 2^n < 6 * 10^(k - 1)) ∧
     (∃ m, 2 * 10^(m - 1) ≤ 5^n ∧ 5^n < 3 * 10^(m - 1))) :=
sorry

end no_positive_integer_n_exists_l2392_239238


namespace subtract_23_result_l2392_239203

variable {x : ℕ}

theorem subtract_23_result (h : x + 30 = 55) : x - 23 = 2 :=
sorry

end subtract_23_result_l2392_239203


namespace rosa_total_pages_called_l2392_239235

variable (P_last P_this : ℝ)

theorem rosa_total_pages_called (h1 : P_last = 10.2) (h2 : P_this = 8.6) : P_last + P_this = 18.8 :=
by sorry

end rosa_total_pages_called_l2392_239235


namespace coordinates_of_C_l2392_239242

noncomputable def point := (ℚ × ℚ)

def A : point := (2, 8)
def B : point := (6, 14)
def M : point := (4, 11)
def L : point := (6, 6)
def C : point := (14, 2)

-- midpoint formula definition
def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main statement to prove
theorem coordinates_of_C (hM : is_midpoint M A B) : C = (14, 2) :=
  sorry

end coordinates_of_C_l2392_239242


namespace number_of_triangles_l2392_239284

open Nat

/-- Each side of a square is divided into 8 equal parts, and using the divisions
as vertices (not including the vertices of the square), the number of different 
triangles that can be obtained is 3136. -/
theorem number_of_triangles (n : ℕ := 7) :
  (n * 4).choose 3 - 4 * n.choose 3 = 3136 := 
sorry

end number_of_triangles_l2392_239284


namespace min_distance_squared_l2392_239269

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end min_distance_squared_l2392_239269


namespace bicycles_in_garage_l2392_239298

theorem bicycles_in_garage 
  (B : ℕ) 
  (h1 : 4 * 3 = 12) 
  (h2 : 7 * 1 = 7) 
  (h3 : 2 * B + 12 + 7 = 25) : 
  B = 3 := 
by
  sorry

end bicycles_in_garage_l2392_239298


namespace range_of_a_l2392_239290

open Real

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 6) ∨ (a ≥ 5 ∨ a ≤ 1) ∧ ¬((0 < a ∧ a < 6) ∧ (a ≥ 5 ∨ a ≤ 1)) ↔ 
  (a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)) :=
by sorry

end range_of_a_l2392_239290


namespace three_point_seven_five_as_fraction_l2392_239245

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end three_point_seven_five_as_fraction_l2392_239245


namespace point_in_fourth_quadrant_l2392_239213

theorem point_in_fourth_quadrant (m : ℝ) (h1 : m + 2 > 0) (h2 : m < 0) : -2 < m ∧ m < 0 := by
  sorry

end point_in_fourth_quadrant_l2392_239213


namespace perfect_square_trinomial_l2392_239279

theorem perfect_square_trinomial (x : ℝ) : 
  let a := x
  let b := 1 / 2
  2 * a * b = x :=
by
  sorry

end perfect_square_trinomial_l2392_239279


namespace best_fitting_model_is_model_3_l2392_239247

-- Define models with their corresponding R^2 values
def R_squared_model_1 : ℝ := 0.72
def R_squared_model_2 : ℝ := 0.64
def R_squared_model_3 : ℝ := 0.98
def R_squared_model_4 : ℝ := 0.81

-- Define a proposition that model 3 has the best fitting effect
def best_fitting_model (R1 R2 R3 R4 : ℝ) : Prop :=
  R3 = max (max R1 R2) (max R3 R4)

-- State the theorem that we need to prove
theorem best_fitting_model_is_model_3 :
  best_fitting_model R_squared_model_1 R_squared_model_2 R_squared_model_3 R_squared_model_4 :=
by
  sorry

end best_fitting_model_is_model_3_l2392_239247
