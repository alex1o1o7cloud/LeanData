import Mathlib

namespace circle_form_eq_standard_form_l77_77417

theorem circle_form_eq_standard_form :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 11 := 
by
  intro x y
  sorry

end circle_form_eq_standard_form_l77_77417


namespace rectangular_garden_area_l77_77641

theorem rectangular_garden_area (w l : ℝ) 
  (h1 : l = 3 * w + 30) 
  (h2 : 2 * (l + w) = 800) : w * l = 28443.75 := 
by
  sorry

end rectangular_garden_area_l77_77641


namespace probability_of_genuine_given_defective_l77_77880

-- Definitions based on the conditions
def num_total_products : ℕ := 7
def num_genuine_products : ℕ := 4
def num_defective_products : ℕ := 3

def probability_event_A : ℝ := (num_defective_products : ℝ) / (num_total_products : ℝ)
def probability_event_AB : ℝ := (num_defective_products : ℝ * num_genuine_products : ℝ) / (num_total_products : ℝ * (num_total_products - 1))

-- Statement of the theorem
theorem probability_of_genuine_given_defective : 
  probability_event_AB / probability_event_A = 2 / 3 :=
by
  sorry

end probability_of_genuine_given_defective_l77_77880


namespace find_vector_at_6_l77_77838

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vec_add (v1 v2 : Vector3D) : Vector3D :=
  { x := v1.x + v2.x, y := v1.y + v2.y, z := v1.z + v2.z }

def vec_scale (c : ℝ) (v : Vector3D) : Vector3D :=
  { x := c * v.x, y := c * v.y, z := c * v.z }

noncomputable def vector_at_t (a d : Vector3D) (t : ℝ) : Vector3D :=
  vec_add a (vec_scale t d)

theorem find_vector_at_6 :
  let a := { x := 2, y := -1, z := 3 }
  let d := { x := 1, y := 2, z := -1 }
  vector_at_t a d 6 = { x := 8, y := 11, z := -3 } :=
by
  sorry

end find_vector_at_6_l77_77838


namespace compound_interest_second_year_l77_77807

variables {P r CI_2 CI_3 : ℝ}

-- Given conditions as definitions in Lean
def interest_rate : ℝ := 0.05
def year_3_interest : ℝ := 1260
def relation_between_CI2_and_CI3 (CI_2 CI_3 : ℝ) : Prop :=
  CI_3 = CI_2 * (1 + interest_rate)

-- The theorem to prove
theorem compound_interest_second_year :
  relation_between_CI2_and_CI3 CI_2 year_3_interest ∧
  r = interest_rate →
  CI_2 = 1200 := 
sorry

end compound_interest_second_year_l77_77807


namespace sine_beta_value_l77_77874

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < (π / 2))
variable (h2 : 0 < β ∧ β < (π / 2))
variable (h3 : Real.cos α = 4 / 5)
variable (h4 : Real.cos (α + β) = 3 / 5)

theorem sine_beta_value : Real.sin β = 7 / 25 :=
by
  -- The proof will go here
  sorry

end sine_beta_value_l77_77874


namespace range_of_k_l77_77757

theorem range_of_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ k ≤ 1 :=
by sorry

end range_of_k_l77_77757


namespace relationship_among_abc_l77_77877

theorem relationship_among_abc 
  (f : ℝ → ℝ)
  (h_symm : ∀ x, f (x) = f (-x))
  (h_def : ∀ x, 0 < x → f x = |Real.log x / Real.log 2|)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = f (1 / 3))
  (hb : b = f (-4))
  (hc : c = f 2) :
  c < a ∧ a < b :=
by
  sorry

end relationship_among_abc_l77_77877


namespace smallest_multiple_of_3_l77_77441

open Finset

theorem smallest_multiple_of_3 (cards : Finset ℕ) (cond1 : {1, 2, 6} ⊆ cards) :
  ∃ x, x ∈ {12, 16, 21, 26, 61, 62} ∧ (x % 3 = 0) ∧ (∀ y, y ∈ {12, 16, 21, 26, 61, 62} ∧ (y % 3 = 0) → x ≤ y) :=
by
  sorry

end smallest_multiple_of_3_l77_77441


namespace find_person_10_number_l77_77857

theorem find_person_10_number (n : ℕ) (a : ℕ → ℕ)
  (h1 : n = 15)
  (h2 : 2 * a 10 = a 9 + a 11)
  (h3 : 2 * a 3 = a 2 + a 4)
  (h4 : a 10 = 8)
  (h5 : a 3 = 7) :
  a 10 = 8 := 
by sorry

end find_person_10_number_l77_77857


namespace minimum_value_expression_l77_77741

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, 
    (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    x = (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c)) ∧
    x = -17 + 12 * Real.sqrt 2 := 
sorry

end minimum_value_expression_l77_77741


namespace inequality_positive_numbers_l77_77936

theorem inequality_positive_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (x + 2 * y + 3 * z)) + (y / (y + 2 * z + 3 * x)) + (z / (z + 2 * x + 3 * y)) ≤ 4 / 3 :=
by
  sorry

end inequality_positive_numbers_l77_77936


namespace smallest_four_digit_multiple_of_18_l77_77522

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l77_77522


namespace product_equals_one_l77_77016

theorem product_equals_one (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1 / (1 + x + x^2)) + (1 / (1 + y + y^2)) + (1 / (1 + x + y)) = 1) : 
  x * y = 1 :=
by
  sorry

end product_equals_one_l77_77016


namespace part_one_part_two_l77_77543

noncomputable def problem_conditions (θ : ℝ) : Prop :=
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  ∃ m : ℝ, (∀ x : ℝ, x^2 - (Real.sqrt 3 - 1) * x + m = 0 → (x = sin_theta ∨ x = cos_theta))

theorem part_one (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let m := sin_theta * cos_theta
  m = (3 - 2 * Real.sqrt 3) / 2 :=
sorry

theorem part_two (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let tan_theta := sin_theta / cos_theta
  (cos_theta - sin_theta * tan_theta) / (1 - tan_theta) = Real.sqrt 3 - 1 :=
sorry

end part_one_part_two_l77_77543


namespace system_inequalities_1_l77_77633

theorem system_inequalities_1 (x : ℝ) (h1 : 2 * x ≥ x - 1) (h2 : 4 * x + 10 > x + 1) :
  x ≥ -1 :=
sorry

end system_inequalities_1_l77_77633


namespace distribute_books_l77_77910

-- Definition of books and people
def num_books : Nat := 2
def num_people : Nat := 10

-- The main theorem statement that we need to prove.
theorem distribute_books : (num_people ^ num_books) = 100 :=
by
  -- Proof body
  sorry

end distribute_books_l77_77910


namespace ab_eq_six_l77_77076

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77076


namespace arithmetic_sequence_sum_l77_77873

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h1 : a 1 + a 5 = 2) 
  (h2 : a 2 + a 14 = 12) : 
  (10 / 2) * (2 * a 1 + 9 * d) = 35 :=
by
  sorry

end arithmetic_sequence_sum_l77_77873


namespace range_of_a_l77_77428

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (x^2 + a*x + 4 < 0)) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end range_of_a_l77_77428


namespace a_six_between_three_and_four_l77_77024

theorem a_six_between_three_and_four (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 := 
sorry

end a_six_between_three_and_four_l77_77024


namespace ice_cream_scoops_l77_77988

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end ice_cream_scoops_l77_77988


namespace regular_hexagon_area_inscribed_in_circle_l77_77449

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l77_77449


namespace pastries_made_correct_l77_77984

-- Definitions based on conditions
def cakes_made := 14
def cakes_sold := 97
def pastries_sold := 8
def cakes_more_than_pastries := 89

-- Definition of the function to compute pastries made
def pastries_made (cakes_made cakes_sold pastries_sold cakes_more_than_pastries : ℕ) : ℕ :=
  cakes_sold - cakes_more_than_pastries

-- The statement to prove
theorem pastries_made_correct : pastries_made cakes_made cakes_sold pastries_sold cakes_more_than_pastries = 8 := by
  unfold pastries_made
  norm_num
  sorry

end pastries_made_correct_l77_77984


namespace solve_sum_of_coefficients_l77_77902

theorem solve_sum_of_coefficients (a b : ℝ) 
  (h1 : ∀ x, ax^2 - bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) : a + b = -10 :=
  sorry

end solve_sum_of_coefficients_l77_77902


namespace initial_money_l77_77664

theorem initial_money (M : ℝ)
  (clothes : M * (1 / 3) = M - M * (2 / 3))
  (food : (M - M * (1 / 3)) * (1 / 5) = (M - M * (1 / 3)) - ((M - M * (1 / 3)) * (4 / 5)))
  (travel : ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4) = ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5)))) * (3 / 4))
  (left : ((M - M * (1 / 3)) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4))) = 400)
  : M = 1000 := 
sorry

end initial_money_l77_77664


namespace max_volume_tank_l77_77980

theorem max_volume_tank (a b h : ℝ) (ha : a ≤ 1.5) (hb : b ≤ 1.5) (hh : h = 1.5) :
  a * b * h ≤ 3.375 :=
by {
  sorry
}

end max_volume_tank_l77_77980


namespace number_of_pupils_l77_77827

theorem number_of_pupils (n : ℕ) 
  (h1 : 83 - 63 = 20) 
  (h2 : (20 : ℝ) / n = 1 / 2) : 
  n = 40 := 
sorry

end number_of_pupils_l77_77827


namespace find_divisor_l77_77687

variable (x y : ℝ)
variable (h1 : (x - 5) / 7 = 7)
variable (h2 : (x - 2) / y = 4)

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 2) / y = 4) : y = 13 := by
  sorry

end find_divisor_l77_77687


namespace who_plays_chess_l77_77148

def person_plays_chess (A B C : Prop) : Prop := 
  (A ∧ ¬ B ∧ ¬ C) ∨ (¬ A ∧ B ∧ ¬ C) ∨ (¬ A ∧ ¬ B ∧ C)

axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom one_statement_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Definition translating the statements made by A, B, and C
def A_plays := true
def B_not_plays := true
def A_not_plays := ¬ A_plays

-- Axiom stating that only one of A's, B's, or C's statements are true
axiom only_one_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Prove that B is the one who knows how to play Chinese chess
theorem who_plays_chess : B_plays :=
by
  -- Insert proof steps here
  sorry

end who_plays_chess_l77_77148


namespace dusty_change_l77_77699

def price_single_layer : ℕ := 4
def price_double_layer : ℕ := 7
def number_of_single_layers : ℕ := 7
def number_of_double_layers : ℕ := 5
def amount_paid : ℕ := 100

theorem dusty_change :
  amount_paid - (number_of_single_layers * price_single_layer + number_of_double_layers * price_double_layer) = 37 := 
by
  sorry

end dusty_change_l77_77699


namespace ab_equals_six_l77_77110

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77110


namespace product_of_ab_l77_77068

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77068


namespace distinct_paths_in_grid_l77_77605

/-- In a 6x5 grid, there are exactly 462 distinct paths from the bottom-left corner to the top-right corner, when only moving right or up, with exactly 6 right steps and 5 up steps. -/
theorem distinct_paths_in_grid : @nat.choose 11 5 = 462 := 
by sorry

end distinct_paths_in_grid_l77_77605


namespace probability_less_than_one_third_l77_77572

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l77_77572


namespace Rickey_took_30_minutes_l77_77789

variables (R P : ℝ)

-- Define the conditions
def Prejean_speed_is_three_quarters_of_Rickey := P = 4 / 3 * R
def total_time_is_70 := R + P = 70

-- Define the statement to prove
theorem Rickey_took_30_minutes 
  (h1 : Prejean_speed_is_three_quarters_of_Rickey R P) 
  (h2 : total_time_is_70 R P) : R = 30 :=
by
  sorry

end Rickey_took_30_minutes_l77_77789


namespace stepa_and_petya_are_wrong_l77_77986

-- Define the six-digit number where all digits are the same.
def six_digit_same (a : ℕ) : ℕ := a * 111111

-- Define the sum of distinct prime divisors of 1001 and 111.
def prime_divisor_sum : ℕ := 7 + 11 + 13 + 3 + 37

-- Define the sum of prime divisors when a is considered.
def additional_sum (a : ℕ) : ℕ :=
  if (a = 2) || (a = 6) || (a = 8) then 2
  else if (a = 5) then 5
  else 0

-- Summarize the possible correct sums
def correct_sums (a : ℕ) : ℕ := prime_divisor_sum + additional_sum a

-- The proof statement
theorem stepa_and_petya_are_wrong (a : ℕ) :
  correct_sums a ≠ 70 ∧ correct_sums a ≠ 80 := 
by {
  sorry
}

end stepa_and_petya_are_wrong_l77_77986


namespace hexagon_area_of_circle_l77_77446

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l77_77446


namespace plumber_total_cost_l77_77484

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end plumber_total_cost_l77_77484


namespace sum_of_b_for_quadratic_has_one_solution_l77_77997

theorem sum_of_b_for_quadratic_has_one_solution :
  (∀ x : ℝ, 3 * x^2 + (b+6) * x + 1 = 0 → 
    ∀ Δ : ℝ, Δ = (b + 6)^2 - 4 * 3 * 1 → 
    Δ = 0 → 
    b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) → 
  (-6 + 2 * Real.sqrt 3 + -6 - 2 * Real.sqrt 3 = -12) := 
by
  sorry

end sum_of_b_for_quadratic_has_one_solution_l77_77997


namespace product_ab_l77_77141

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77141


namespace greatest_three_digit_multiple_of_17_l77_77232

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77232


namespace balloon_totals_l77_77774

-- Definitions
def Joan_blue := 40
def Joan_red := 30
def Joan_green := 0
def Joan_yellow := 0

def Melanie_blue := 41
def Melanie_red := 0
def Melanie_green := 20
def Melanie_yellow := 0

def Eric_blue := 0
def Eric_red := 25
def Eric_green := 0
def Eric_yellow := 15

-- Total counts
def total_blue := Joan_blue + Melanie_blue + Eric_blue
def total_red := Joan_red + Melanie_red + Eric_red
def total_green := Joan_green + Melanie_green + Eric_green
def total_yellow := Joan_yellow + Melanie_yellow + Eric_yellow

-- Statement of the problem
theorem balloon_totals :
  total_blue = 81 ∧
  total_red = 55 ∧
  total_green = 20 ∧
  total_yellow = 15 :=
by
  -- Proof omitted
  sorry

end balloon_totals_l77_77774


namespace smallest_angle_measure_in_triangle_l77_77763

theorem smallest_angle_measure_in_triangle (a b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : c > 2 * Real.sqrt 2) :
  ∃ x : ℝ, x = 140 ∧ C < x :=
sorry

end smallest_angle_measure_in_triangle_l77_77763


namespace ab_equals_six_l77_77114

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77114


namespace mom_prepared_pieces_l77_77531

-- Define the conditions
def jane_pieces : ℕ := 4
def total_eaters : ℕ := 3

-- Define the hypothesis that each of the eaters ate an equal number of pieces
def each_ate_equal (pieces : ℕ) : Prop := pieces = jane_pieces

-- The number of pieces Jane's mom prepared
theorem mom_prepared_pieces : total_eaters * jane_pieces = 12 :=
by
  -- Placeholder for actual proof
  sorry

end mom_prepared_pieces_l77_77531


namespace ab_equals_six_l77_77056

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77056


namespace closest_total_population_of_cities_l77_77381

theorem closest_total_population_of_cities 
    (n_cities : ℕ) (avg_population_lower avg_population_upper : ℕ)
    (h_lower : avg_population_lower = 3800) (h_upper : avg_population_upper = 4200) :
  (25:ℕ) * (4000:ℕ) = 100000 :=
by
  sorry

end closest_total_population_of_cities_l77_77381


namespace art_piece_increase_l77_77957

theorem art_piece_increase (initial_price : ℝ) (multiplier : ℝ) (future_increase : ℝ) (h1 : initial_price = 4000) (h2 : multiplier = 3) :
  future_increase = (multiplier * initial_price) - initial_price :=
by
  rw [h1, h2]
  norm_num
  sorry

end art_piece_increase_l77_77957


namespace number_of_attendees_choosing_water_l77_77502

variables {total_attendees : ℕ} (juice_percent water_percent : ℚ)

-- Conditions
def attendees_juice (total_attendees : ℕ) : ℚ := 0.7 * total_attendees
def attendees_water (total_attendees : ℕ) : ℚ := 0.3 * total_attendees
def attendees_juice_given := (attendees_juice total_attendees) = 140

-- Theorem statement
theorem number_of_attendees_choosing_water 
  (h1 : juice_percent = 0.7) 
  (h2 : water_percent = 0.3) 
  (h3 : attendees_juice total_attendees = 140) : 
  attendees_water total_attendees = 60 :=
sorry

end number_of_attendees_choosing_water_l77_77502


namespace ab_equals_six_l77_77118

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77118


namespace correct_divisor_l77_77604

theorem correct_divisor (X D : ℕ) (h1 : X / 72 = 24) (h2 : X / D = 48) : D = 36 :=
sorry

end correct_divisor_l77_77604


namespace faster_cow_days_to_eat_one_bag_l77_77473

-- Conditions as assumptions
def num_cows : ℕ := 60
def num_husks : ℕ := 150
def num_days : ℕ := 80
def faster_cows : ℕ := 20
def normal_cows : ℕ := num_cows - faster_cows
def faster_rate : ℝ := 1.3

-- The question translated to Lean 4 statement
theorem faster_cow_days_to_eat_one_bag :
  (faster_cows * faster_rate + normal_cows) / num_cows * (num_husks / num_days) = 1 / 27.08 :=
sorry

end faster_cow_days_to_eat_one_bag_l77_77473


namespace ab_equals_six_l77_77113

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77113


namespace sheela_monthly_income_eq_l77_77631

-- Defining the conditions
def sheela_deposit : ℝ := 4500
def percentage_of_income : ℝ := 0.28

-- Define Sheela's monthly income as I
variable (I : ℝ)

-- The theorem to prove
theorem sheela_monthly_income_eq : (percentage_of_income * I = sheela_deposit) → (I = 16071.43) :=
by
  sorry

end sheela_monthly_income_eq_l77_77631


namespace sum_binom_equals_220_l77_77495

/-- The binomial coefficient C(n, k) -/
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

/-- Prove that the sum C(2, 2) + C(3, 2) + C(4, 2) + ... + C(11, 2) equals 220 using the 
    binomial coefficient property C(n, r+1) + C(n, r) = C(n+1, r+1) -/
theorem sum_binom_equals_220 :
  binom 2 2 + binom 3 2 + binom 4 2 + binom 5 2 + binom 6 2 + binom 7 2 + 
  binom 8 2 + binom 9 2 + binom 10 2 + binom 11 2 = 220 := by
sorry

end sum_binom_equals_220_l77_77495


namespace simplify_expression_l77_77546

-- Define the given condition as a hypothesis
theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 :=
by
  sorry -- Proof will be provided here.

end simplify_expression_l77_77546


namespace monotone_function_sol_l77_77001

noncomputable def monotone_function (f : ℤ → ℤ) :=
  ∀ x y : ℤ, f x ≤ f y → x ≤ y

theorem monotone_function_sol
  (f : ℤ → ℤ)
  (H1 : monotone_function f)
  (H2 : ∀ x y : ℤ, f (x^2005 + y^2005) = f x ^ 2005 + f y ^ 2005) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end monotone_function_sol_l77_77001


namespace total_amount_l77_77828

theorem total_amount
  (x y z : ℝ)
  (hy : y = 0.45 * x)
  (hz : z = 0.50 * x)
  (y_share : y = 27) :
  x + y + z = 117 :=
by
  sorry

end total_amount_l77_77828


namespace min_arithmetic_series_sum_l77_77429

-- Definitions from the conditions
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def arithmetic_series_sum (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (a1 + (n-1) * d / 2)

-- Theorem statement
theorem min_arithmetic_series_sum (a2 a7 : ℤ) (h1 : a2 = -7) (h2 : a7 = 3) :
  ∃ n, (n * (a2 + (n - 1) * 2 / 2) = (n * n) - 10 * n) ∧
  (∀ m, n* (a2 + (m - 1) * 2 / 2) ≥ n * (n * n - 10 * n)) :=
sorry

end min_arithmetic_series_sum_l77_77429


namespace ab_equals_six_l77_77123

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77123


namespace math_problem_l77_77036

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem math_problem
  (omega phi : ℝ)
  (h1 : omega > 0)
  (h2 : |phi| < Real.pi / 2)
  (h3 : ∀ x, f x = Real.sin (omega * x + phi))
  (h4 : ∀ k : ℤ, f (k * Real.pi) = f 0) 
  (h5 : f 0 = 1 / 2) :
  (omega = 2) ∧
  (∀ x, f (x + Real.pi / 6) = f (-x + Real.pi / 6)) ∧
  (∀ k : ℤ, 
    ∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    ∀ y, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    x < y → f x ≤ f y) :=
by
  sorry

end math_problem_l77_77036


namespace sum_of_inserted_numbers_l77_77771

variable {x y : ℝ} -- Variables x and y are real numbers

-- Conditions
axiom geometric_sequence_condition : x^2 = 3 * y
axiom arithmetic_sequence_condition : 2 * y = x + 9

-- Goal: Prove that x + y = 45 / 4 (which is 11 1/4)
theorem sum_of_inserted_numbers : x + y = 45 / 4 :=
by
  -- Utilize axioms and conditions
  sorry

end sum_of_inserted_numbers_l77_77771


namespace greatest_three_digit_multiple_of_17_l77_77251

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77251


namespace art_piece_increase_is_correct_l77_77959

-- Define the conditions
def initial_price : ℝ := 4000
def future_multiplier : ℝ := 3
def future_price : ℝ := future_multiplier * initial_price

-- Define the goal
-- Proof that the increase in price is equal to $8000
theorem art_piece_increase_is_correct : future_price - initial_price = 8000 := 
by {
  -- We put sorry here to skip the actual proof
  sorry
}

end art_piece_increase_is_correct_l77_77959


namespace ab_equals_six_l77_77111

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77111


namespace gcd_4536_13440_216_l77_77192

def gcd_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_4536_13440_216 : gcd_of_three_numbers 4536 13440 216 = 216 :=
by
  sorry

end gcd_4536_13440_216_l77_77192


namespace cos_A_and_sin_2B_minus_A_l77_77762

variable (A B C a b c : ℝ)
variable (h1 : a * Real.sin A = 4 * b * Real.sin B)
variable (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2))

theorem cos_A_and_sin_2B_minus_A :
  Real.cos A = -Real.sqrt 5 / 5 ∧ Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_A_and_sin_2B_minus_A_l77_77762


namespace perimeter_eq_28_l77_77767

theorem perimeter_eq_28 (PQ QR TS TU : ℝ) (h2 : PQ = 4) (h3 : QR = 4) 
(h5 : TS = 8) (h7 : TU = 4) : 
PQ + QR + TS + TS - TU + TU + TU = 28 := by
  sorry

end perimeter_eq_28_l77_77767


namespace powers_of_2_form_6n_plus_8_l77_77858

noncomputable def is_power_of_two (x : ℕ) : Prop := ∃ k : ℕ, x = 2 ^ k

def of_the_form (n : ℕ) : ℕ := 6 * n + 8

def is_odd_greater_than_one (k : ℕ) : Prop := k % 2 = 1 ∧ k > 1

theorem powers_of_2_form_6n_plus_8 (k : ℕ) (n : ℕ) :
  (2 ^ k = of_the_form n) ↔ is_odd_greater_than_one k :=
sorry

end powers_of_2_form_6n_plus_8_l77_77858


namespace k_minus_2_divisible_by_3_l77_77923

theorem k_minus_2_divisible_by_3
  (k : ℕ)
  (a : ℕ → ℤ)
  (h_a0_pos : 0 < k)
  (h_seq : ∀ n ≥ 1, a n = (a (n - 1) + n^k) / n) :
  (k - 2) % 3 = 0 :=
sorry

end k_minus_2_divisible_by_3_l77_77923


namespace ice_cream_scoops_total_l77_77993

noncomputable def scoops_of_ice_cream : ℕ :=
let single_cone : ℕ := 1 in
let double_cone : ℕ := single_cone * 2 in
let banana_split : ℕ := single_cone * 3 in
let waffle_bowl : ℕ := banana_split + 1 in
single_cone + double_cone + banana_split + waffle_bowl

theorem ice_cream_scoops_total : scoops_of_ice_cream = 10 :=
sorry

end ice_cream_scoops_total_l77_77993


namespace geom_seq_min_value_l77_77916

theorem geom_seq_min_value :
  let a1 := 2
  ∃ r : ℝ, ∀ a2 a3,
    a2 = 2 * r ∧ 
    a3 = 2 * r^2 →
    3 * a2 + 6 * a3 = -3/2 := by
  sorry

end geom_seq_min_value_l77_77916


namespace AM_GM_inequality_equality_case_of_AM_GM_l77_77941

theorem AM_GM_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (x / y) + (y / x) ≥ 2 :=
by
  sorry

theorem equality_case_of_AM_GM (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : ((x / y) + (y / x) = 2) ↔ (x = y) :=
by
  sorry

end AM_GM_inequality_equality_case_of_AM_GM_l77_77941


namespace percentage_of_difference_is_50_l77_77756

noncomputable def percentage_of_difference (x y : ℝ) (p : ℝ) :=
  (p / 100) * (x - y) = 0.20 * (x + y)

noncomputable def y_is_percentage_of_x (x y : ℝ) :=
  y = 0.42857142857142854 * x

theorem percentage_of_difference_is_50 (x y : ℝ) (p : ℝ)
  (h1 : percentage_of_difference x y p)
  (h2 : y_is_percentage_of_x x y) :
  p = 50 :=
by
  sorry

end percentage_of_difference_is_50_l77_77756


namespace greatest_three_digit_multiple_of_17_l77_77193

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77193


namespace possible_values_ceil_x_squared_l77_77367

theorem possible_values_ceil_x_squared (x : ℝ) (h : ⌈x⌉ = 9) : (finset.Icc 65 81).card = 17 := by
  sorry

end possible_values_ceil_x_squared_l77_77367


namespace average_weight_of_whole_class_l77_77672

theorem average_weight_of_whole_class :
  let students_A := 50
  let students_B := 50
  let avg_weight_A := 60
  let avg_weight_B := 80
  let total_students := students_A + students_B
  let total_weight_A := students_A * avg_weight_A
  let total_weight_B := students_B * avg_weight_B
  let total_weight := total_weight_A + total_weight_B
  let avg_weight := total_weight / total_students
  avg_weight = 70 := 
by 
  sorry

end average_weight_of_whole_class_l77_77672


namespace ab_equals_6_l77_77104

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77104


namespace ratio_of_medians_to_sides_l77_77510

theorem ratio_of_medians_to_sides (a b c : ℝ) (m_a m_b m_c : ℝ) 
  (h1: m_a = 1/2 * (2 * b^2 + 2 * c^2 - a^2)^(1/2))
  (h2: m_b = 1/2 * (2 * a^2 + 2 * c^2 - b^2)^(1/2))
  (h3: m_c = 1/2 * (2 * a^2 + 2 * b^2 - c^2)^(1/2)) :
  (m_a*m_a + m_b*m_b + m_c*m_c) / (a*a + b*b + c*c) = 3/4 := 
by 
  sorry

end ratio_of_medians_to_sides_l77_77510


namespace find_y_l77_77978

noncomputable def x : ℝ := 3.3333333333333335

theorem find_y (y x: ℝ) (h1: x = 3.3333333333333335) (h2: x * 10 / y = x^2) :
  y = 3 :=
by
  sorry

end find_y_l77_77978


namespace simplify_expression_l77_77632

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 :=
sorry

end simplify_expression_l77_77632


namespace freeze_alcohol_time_l77_77878

theorem freeze_alcohol_time :
  ∀ (init_temp freeze_temp : ℝ)
    (cooling_rate : ℝ), 
    init_temp = 12 → 
    freeze_temp = -117 → 
    cooling_rate = 1.5 →
    (freeze_temp - init_temp) / cooling_rate = -129 / cooling_rate :=
by
  intros init_temp freeze_temp cooling_rate h1 h2 h3
  rw [h2, h1, h3]
  exact sorry

end freeze_alcohol_time_l77_77878


namespace find_hourly_rate_l77_77158

-- Definitions of conditions in a)
def hourly_rate : ℝ := sorry  -- This is what we will find.
def hours_worked : ℝ := 3
def tip_percentage : ℝ := 0.2
def total_paid : ℝ := 54

-- Functions based on the conditions
def cost_without_tip (rate : ℝ) : ℝ := hours_worked * rate
def tip_amount (rate : ℝ) : ℝ := tip_percentage * (cost_without_tip rate)
def total_cost (rate : ℝ) : ℝ := (cost_without_tip rate) + (tip_amount rate)

-- The goal is to prove that the rate is 15
theorem find_hourly_rate : total_cost 15 = total_paid :=
by
  sorry

end find_hourly_rate_l77_77158


namespace isosceles_triangle_leg_length_l77_77349

theorem isosceles_triangle_leg_length
  (P : ℝ) (base : ℝ) (L : ℝ)
  (h_isosceles : true)
  (h_perimeter : P = 24)
  (h_base : base = 10)
  (h_perimeter_formula : P = base + 2 * L) :
  L = 7 := 
by
  sorry

end isosceles_triangle_leg_length_l77_77349


namespace greatest_three_digit_multiple_of_17_l77_77281

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77281


namespace geometric_sequence_a3_l77_77384

variable {a : ℕ → ℝ} (h1 : a 1 > 0) (h2 : a 2 * a 4 = 25)
def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (h_geom : geometric_sequence a) : 
  a 3 = 5 := 
by
  sorry

end geometric_sequence_a3_l77_77384


namespace trishul_investment_less_than_raghu_l77_77657

noncomputable def VishalInvestment (T : ℝ) : ℝ := 1.10 * T

noncomputable def TotalInvestment (T : ℝ) (R : ℝ) : ℝ :=
  T + VishalInvestment T + R

def RaghuInvestment : ℝ := 2100

def TotalSumInvested : ℝ := 6069

theorem trishul_investment_less_than_raghu :
  ∃ T : ℝ, TotalInvestment T RaghuInvestment = TotalSumInvested → (RaghuInvestment - T) / RaghuInvestment * 100 = 10 := by
  sorry

end trishul_investment_less_than_raghu_l77_77657


namespace greatest_three_digit_multiple_of_17_l77_77264

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77264


namespace calculate_neg_three_minus_one_l77_77987

theorem calculate_neg_three_minus_one : -3 - 1 = -4 := by
  sorry

end calculate_neg_three_minus_one_l77_77987


namespace greatest_three_digit_multiple_of_17_l77_77201

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77201


namespace probability_red_blue_given_draw_l77_77773

/-- Probability that the only marbles in the bag are red and blue given the draw sequence red, blue, red is 27 / 35 -/
theorem probability_red_blue_given_draw :
  ∀ (marbles : set string), 
  marbles = {"red", "green", "blue"} → 
  let subsets := {s ∈ marbles.powerset \ {∅}} in
  let prob_red_blue_red := (1 / 2) ^ 3 in
  let prob_red_blue_green := (1 / 3) ^ 3 in
  let prior_red_blue := 1 / 7 in
  let prior_red_blue_green := 1 / 7 in
  let total_prob_red_blue_red := (prob_red_blue_red * prior_red_blue) + (prob_red_blue_green * prior_red_blue_green) in
  (prob_red_blue_red * prior_red_blue) / total_prob_red_blue_red = 27 / 35 :=
by
  intros,
  sorry

end probability_red_blue_given_draw_l77_77773


namespace probability_of_interval_l77_77580

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l77_77580


namespace sheila_saving_years_l77_77166

theorem sheila_saving_years 
  (initial_amount : ℝ) 
  (monthly_saving : ℝ) 
  (secret_addition : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : 
  initial_amount = 3000 ∧ 
  monthly_saving = 276 ∧ 
  secret_addition = 7000 ∧ 
  final_amount = 23248 → 
  years = 4 := 
sorry

end sheila_saving_years_l77_77166


namespace angle_equality_l77_77154

variables {Point Circle : Type}
variables (K O1 O2 P1 P2 Q1 Q2 M1 M2 : Point)
variables (W1 W2 : Circle)
variables (midpoint : Point → Point → Point)
variables (is_center : Point → Circle → Prop)
variables (intersects_at : Circle → Circle → Point → Prop)
variables (common_tangent_points : Circle → Circle → (Point × Point) × (Point × Point) → Prop)
variables (intersect_circle_at : Circle → Line → Point → Point → Prop)
variables (angle : Point → Point → Point → ℝ) -- to denote the angle measure between three points

-- Conditions
axiom K_intersection : intersects_at W1 W2 K
axiom O1_center : is_center O1 W1
axiom O2_center : is_center O2 W2
axiom tangents_meet_at : common_tangent_points W1 W2 ((P1, Q1), (P2, Q2))
axiom M1_midpoint : M1 = midpoint P1 Q1
axiom M2_midpoint : M2 = midpoint P2 Q2

-- The statement to prove
theorem angle_equality : angle O1 K O2 = angle M1 K M2 := 
  sorry

end angle_equality_l77_77154


namespace li_family_cinema_cost_l77_77382

theorem li_family_cinema_cost :
  let standard_ticket_price := 10
  let child_discount := 0.4
  let senior_discount := 0.3
  let handling_fee := 5
  let num_adults := 2
  let num_children := 1
  let num_seniors := 1
  let child_ticket_price := (1 - child_discount) * standard_ticket_price
  let senior_ticket_price := (1 - senior_discount) * standard_ticket_price
  let total_ticket_cost := num_adults * standard_ticket_price + num_children * child_ticket_price + num_seniors * senior_ticket_price
  let final_cost := total_ticket_cost + handling_fee
  final_cost = 38 :=
by
  sorry

end li_family_cinema_cost_l77_77382


namespace quadratic_roots_condition_l77_77759

theorem quadratic_roots_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ x₁^2 + (3 * a - 1) * x₁ + a + 8 = 0 ∧
  x₂^2 + (3 * a - 1) * x₂ + a + 8 = 0) → a < -2 :=
by
  sorry

end quadratic_roots_condition_l77_77759


namespace product_of_ab_l77_77071

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77071


namespace min_k_for_xyz_sum_l77_77012

open Finset

theorem min_k_for_xyz_sum (S : Finset ℕ) (hS : S = (Finset.range 2012).map (Nat.succ)) :
  ∃ (k : ℕ), k = 1008 ∧ 
  ∀ A : Finset ℕ, A ⊆ S → A.card = k →
  (∃ x y z a b c : ℕ, {x, y, z} ⊆ A ∧ {a, b, c} ⊆ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 
    ∧ x = a + b ∧ y = b + c ∧ z = c + a) :=
by
  sorry

end min_k_for_xyz_sum_l77_77012


namespace rachel_age_when_emily_half_her_age_l77_77337

theorem rachel_age_when_emily_half_her_age (emily_current_age rachel_current_age : ℕ) 
  (h1 : emily_current_age = 20) 
  (h2 : rachel_current_age = 24) 
  (age_difference : ℕ) 
  (h3 : rachel_current_age - emily_current_age = age_difference) 
  (emily_age_when_half : ℕ) 
  (rachel_age_when_half : ℕ) 
  (h4 : emily_age_when_half = rachel_age_when_half / 2)
  (h5 : rachel_age_when_half = emily_age_when_half + age_difference) :
  rachel_age_when_half = 8 :=
by
  sorry

end rachel_age_when_emily_half_her_age_l77_77337


namespace sum_of_all_alternating_sums_eq_5120_l77_77736

open BigOperators
open Finset

noncomputable def alternating_sum (s : Finset ℕ) : ℕ :=
  s.sort (· > ·).alternating_sum
  where
    alternating_sum : List ℕ → ℕ
    | [] => 0
    | [a] => a
    | a :: b :: rest => a - b + alternating_sum rest

theorem sum_of_all_alternating_sums_eq_5120 :
  let subsets := univ.powerset \ {∅}
  (∑ s in subsets, alternating_sum s) = 5120 :=
by
  let subsets := univ.powerset \ {∅}
  have h₁ : subsets.card = (2 ^ 10) - 1 := by
    simp [Finset.card_powerset, Finset.card_univ]
    ring
  have h₂ : ∑ s in subsets, alternating_sum s = 5120 := sorry
  exact h₂


end sum_of_all_alternating_sums_eq_5120_l77_77736


namespace problem_g3_1_l77_77559

theorem problem_g3_1 (a : ℝ) : 
  (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = a ↔ a = 2005 := 
sorry

end problem_g3_1_l77_77559


namespace shop_dimension_is_100_l77_77642

-- Given conditions
def monthly_rent : ℕ := 1300
def annual_rent_per_sqft : ℕ := 156

-- Define annual rent
def annual_rent : ℕ := monthly_rent * 12

-- Define dimension to prove
def dimension_of_shop : ℕ := annual_rent / annual_rent_per_sqft

-- The theorem statement
theorem shop_dimension_is_100 :
  dimension_of_shop = 100 :=
by
  sorry

end shop_dimension_is_100_l77_77642


namespace average_of_remaining_numbers_l77_77947

theorem average_of_remaining_numbers 
    (nums : List ℝ) 
    (h_length : nums.length = 12) 
    (h_avg_90 : (nums.sum) / 12 = 90) 
    (h_contains_65_85 : 65 ∈ nums ∧ 85 ∈ nums) 
    (nums' := nums.erase 65)
    (nums'' := nums'.erase 85) : 
   nums''.length = 10 ∧ nums''.sum / 10 = 93 :=
by
  sorry

end average_of_remaining_numbers_l77_77947


namespace triplet_sums_to_two_l77_77999

theorem triplet_sums_to_two :
  (3 / 4 + 1 / 4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3 / 5 + 4 / 5 + 3 / 5 = 2) ∧
  (2 - 3 + 3 = 2) :=
by
  sorry

end triplet_sums_to_two_l77_77999


namespace interval_length_l77_77950

theorem interval_length (a b : ℝ) (h : ∀ x : ℝ, a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) :
  (b - a = 57) :=
sorry

end interval_length_l77_77950


namespace product_of_ab_l77_77070

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77070


namespace greatest_three_digit_multiple_of_17_l77_77208

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77208


namespace initial_men_count_l77_77955

theorem initial_men_count (M : ℕ) (F : ℕ) (h1 : F = M * 22) (h2 : (M + 2280) * 5 = M * 20) : M = 760 := by
  sorry

end initial_men_count_l77_77955


namespace ab_equals_six_l77_77061

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77061


namespace discriminant_of_quadratic_equation_l77_77964

theorem discriminant_of_quadratic_equation :
  let a := 5
  let b := -9
  let c := 4
  (b^2 - 4 * a * c = 1) :=
by {
  sorry
}

end discriminant_of_quadratic_equation_l77_77964


namespace expected_value_coins_basilio_l77_77730

noncomputable def expected_coins_basilio (n : ℕ) (p : ℚ) : ℚ :=
  let X_Y := Binomial n p
  let E_X_Y := (n * p : ℚ)
  let X_minus_Y := 0.5
  (E_X_Y + X_minus_Y) / 2

theorem expected_value_coins_basilio :
  expected_coins_basilio 20 (1/2) = 5.25 := by
  sorry

end expected_value_coins_basilio_l77_77730


namespace greatest_three_digit_multiple_of_17_l77_77211

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77211


namespace hyperbola_asymptotes_l77_77745

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := (y^2 / 4) - (x^2 / 9) = 1

-- Define the standard form of hyperbola asymptotes equations
def asymptotes_eq (x y : ℝ) : Prop := 2 * x + 3 * y = 0 ∨ 2 * x - 3 * y = 0

-- The final proof statement
theorem hyperbola_asymptotes (x y : ℝ) (h : hyperbola_eq x y) : asymptotes_eq x y :=
    sorry

end hyperbola_asymptotes_l77_77745


namespace hyperbola_focal_length_range_l77_77549

theorem hyperbola_focal_length_range (m : ℝ) (h1 : m > 0)
    (h2 : ∀ x y, x^2 - y^2 / m^2 ≠ 1 → y ≠ m * x ∧ y ≠ -m * x)
    (h3 : ∀ x y, x^2 + (y + 2)^2 = 1 → x^2 + y^2 / m^2 ≠ 1) :
    ∃ c : ℝ, 2 < 2 * Real.sqrt (1 + m^2) ∧ 2 * Real.sqrt (1 + m^2) < 4 :=
by
  sorry

end hyperbola_focal_length_range_l77_77549


namespace tournament_start_count_l77_77377

theorem tournament_start_count (x : ℝ) (h1 : (0.1 * x = 30)) : x = 300 :=
by
  sorry

end tournament_start_count_l77_77377


namespace angle_ratio_l77_77356

theorem angle_ratio (x y α β : ℝ)
  (h1 : y = x + β)
  (h2 : 2 * y = 2 * x + α) :
  α / β = 2 :=
by
  sorry

end angle_ratio_l77_77356


namespace sum_of_squares_l77_77369

-- Define conditions
def condition1 (a b : ℝ) : Prop := a - b = 6
def condition2 (a b : ℝ) : Prop := a * b = 7

-- Define what we want to prove
def target (a b : ℝ) : Prop := a^2 + b^2 = 50

-- Main theorem stating the required proof
theorem sum_of_squares (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) : target a b :=
by sorry

end sum_of_squares_l77_77369


namespace triangle_area_l77_77181

/-- Proof that the area of a triangle with side lengths 9 cm, 40 cm, and 41 cm is 180 square centimeters, 
    given that these lengths form a right triangle. -/
theorem triangle_area : ∀ (a b c : ℕ), a = 9 → b = 40 → c = 41 → a^2 + b^2 = c^2 → (a * b) / 2 = 180 := by
  intros a b c ha hb hc hpyth
  sorry

end triangle_area_l77_77181


namespace proposition_p_proposition_not_q_proof_p_and_not_q_l77_77026

variable (p : Prop)
variable (q : Prop)
variable (r : Prop)

theorem proposition_p : (∃ x0 : ℝ, x0 > 2) := sorry

theorem proposition_not_q : ¬ (∀ x : ℝ, x^3 > x^2) := sorry

theorem proof_p_and_not_q : (∃ x0 : ℝ, x0 > 2) ∧ ¬ (∀ x : ℝ, x^3 > x^2) :=
by
  exact ⟨proposition_p, proposition_not_q⟩

end proposition_p_proposition_not_q_proof_p_and_not_q_l77_77026


namespace calculate_area_of_shaded_region_l77_77442

namespace Proof

noncomputable def AreaOfShadedRegion (width height : ℝ) (divisions : ℕ) : ℝ :=
  let small_width := width
  let small_height := height / divisions
  let area_of_small := small_width * small_height
  let shaded_in_small := area_of_small / 2
  let total_shaded := divisions * shaded_in_small
  total_shaded

theorem calculate_area_of_shaded_region :
  AreaOfShadedRegion 3 14 4 = 21 := by
  sorry

end Proof

end calculate_area_of_shaded_region_l77_77442


namespace greatest_three_digit_multiple_of_17_l77_77266

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77266


namespace greatest_three_digit_multiple_of_17_l77_77254

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77254


namespace remainder_of_division_l77_77658

theorem remainder_of_division :
  ∃ R : ℕ, 176 = (19 * 9) + R ∧ R = 5 :=
by
  sorry

end remainder_of_division_l77_77658


namespace jon_and_mary_frosting_l77_77389

-- Jon frosts a cupcake every 40 seconds
def jon_frost_rate : ℚ := 1 / 40

-- Mary frosts a cupcake every 24 seconds
def mary_frost_rate : ℚ := 1 / 24

-- Combined frosting rate of Jon and Mary
def combined_frost_rate : ℚ := jon_frost_rate + mary_frost_rate

-- Total time in seconds for 12 minutes
def total_time_seconds : ℕ := 12 * 60

-- Calculate the total number of cupcakes frosted in 12 minutes
def total_cupcakes_frosted (time_seconds : ℕ) (rate : ℚ) : ℚ :=
  time_seconds * rate

theorem jon_and_mary_frosting : total_cupcakes_frosted total_time_seconds combined_frost_rate = 48 := by
  sorry

end jon_and_mary_frosting_l77_77389


namespace fish_worth_apples_l77_77908

-- Defining the variables
variables (f l r a : ℝ)

-- Conditions based on the problem
def condition1 : Prop := 5 * f = 3 * l
def condition2 : Prop := l = 6 * r
def condition3 : Prop := 3 * r = 2 * a

-- The statement of the problem
theorem fish_worth_apples (h1 : condition1 f l) (h2 : condition2 l r) (h3 : condition3 r a) : f = 12 / 5 * a :=
by
  sorry

end fish_worth_apples_l77_77908


namespace average_is_1380_l77_77820

def avg_of_numbers : Prop := 
  (1200 + 1300 + 1400 + 1510 + 1520 + 1530 + 1200) / 7 = 1380

theorem average_is_1380 : avg_of_numbers := by
  sorry

end average_is_1380_l77_77820


namespace greatest_three_digit_multiple_of_17_l77_77213

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77213


namespace some_athletes_not_members_honor_society_l77_77995

universe u

variable {U : Type u} -- Assume U is our universe of discourse, e.g., individuals.
variables (Athletes Disciplined HonorSociety : U → Prop)

-- Conditions
def some_athletes_not_disciplined := ∃ x, Athletes x ∧ ¬Disciplined x
def all_honor_society_disciplined := ∀ x, HonorSociety x → Disciplined x

-- Correct Answer
theorem some_athletes_not_members_honor_society :
  some_athletes_not_disciplined Athletes Disciplined →
  all_honor_society_disciplined HonorSociety Disciplined →
  ∃ y, Athletes y ∧ ¬HonorSociety y :=
by
  intros h1 h2
  sorry

end some_athletes_not_members_honor_society_l77_77995


namespace hexagon_area_of_circle_l77_77447

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l77_77447


namespace probability_less_than_third_l77_77563

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l77_77563


namespace top_angle_isosceles_triangle_l77_77028

open Real

theorem top_angle_isosceles_triangle (A B C : ℝ) (abc_is_isosceles : (A = B ∨ B = C ∨ A = C))
  (angle_A : A = 40) : (B = 40 ∨ B = 100) :=
sorry

end top_angle_isosceles_triangle_l77_77028


namespace ab_value_l77_77046

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77046


namespace drawing_two_black_balls_probability_equals_half_l77_77845

noncomputable def total_number_of_events : ℕ := 6

noncomputable def number_of_black_draw_events : ℕ := 3

noncomputable def probability_of_drawing_two_black_balls : ℚ :=
  number_of_black_draw_events / total_number_of_events

theorem drawing_two_black_balls_probability_equals_half :
  probability_of_drawing_two_black_balls = 1 / 2 :=
by
  sorry

end drawing_two_black_balls_probability_equals_half_l77_77845


namespace polar_to_cartesian_l77_77882

theorem polar_to_cartesian (ρ θ x y : ℝ) (h1 : ρ = 2 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l77_77882


namespace gcd_779_209_589_eq_19_l77_77861

theorem gcd_779_209_589_eq_19 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end gcd_779_209_589_eq_19_l77_77861


namespace green_apples_count_l77_77695

variables (G R : ℕ)

def total_apples_collected (G R : ℕ) : Prop :=
  R + G = 496

def relation_red_green (G R : ℕ) : Prop :=
  R = 3 * G

theorem green_apples_count (G R : ℕ) (h1 : total_apples_collected G R) (h2 : relation_red_green G R) :
  G = 124 :=
by sorry

end green_apples_count_l77_77695


namespace greatest_three_digit_multiple_of_17_l77_77265

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77265


namespace find_difference_square_l77_77041

theorem find_difference_square (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 6) :
  (x - y)^2 = 25 :=
by
  sorry

end find_difference_square_l77_77041


namespace statement_1_correct_statement_3_correct_correct_statements_l77_77824

-- Definition for Acute Angles
def is_acute_angle (α : Real) : Prop :=
  0 < α ∧ α < 90

-- Definition for First Quadrant Angles
def is_first_quadrant_angle (β : Real) : Prop :=
  ∃ k : Int, k * 360 < β ∧ β < 90 + k * 360

-- Conditions
theorem statement_1_correct (α : Real) : is_acute_angle α → is_first_quadrant_angle α :=
sorry

theorem statement_3_correct (β : Real) : is_first_quadrant_angle β :=
sorry

-- Final Proof Statement
theorem correct_statements (α β : Real) :
  (is_acute_angle α → is_first_quadrant_angle α) ∧ (is_first_quadrant_angle β) :=
⟨statement_1_correct α, statement_3_correct β⟩

end statement_1_correct_statement_3_correct_correct_statements_l77_77824


namespace smallest_four_digit_multiple_of_18_l77_77521

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l77_77521


namespace find_January_salary_l77_77671

-- Definitions and conditions
variables (J F M A May : ℝ)
def avg_Jan_to_Apr : Prop := (J + F + M + A) / 4 = 8000
def avg_Feb_to_May : Prop := (F + M + A + May) / 4 = 8300
def May_salary : Prop := May = 6500

-- Theorem statement
theorem find_January_salary (h1 : avg_Jan_to_Apr J F M A) 
                            (h2 : avg_Feb_to_May F M A May) 
                            (h3 : May_salary May) : 
                            J = 5300 :=
sorry

end find_January_salary_l77_77671


namespace union_A_B_inter_complB_A_l77_77362

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set A
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define the complement of B with respect to U
def compl_B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 6}

-- Problem (1): Prove that A ∪ B = {x | -3 < x ∧ x ≤ 6}
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 6} := by
  sorry

-- Problem (2): Prove that (compl_B) ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6}
theorem inter_complB_A : compl_B ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6} := by 
  sorry

end union_A_B_inter_complB_A_l77_77362


namespace satisfies_negative_inverse_l77_77709

noncomputable def f1 (x : ℝ) : ℝ := x - 1/x
noncomputable def f2 (x : ℝ) : ℝ := x + 1/x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f4 (x : ℝ) : ℝ :=
  if x < 1 then x
  else if x = 1 then 0
  else -1/x

theorem satisfies_negative_inverse :
  { f | (∀ x : ℝ, f (1 / x) = -f x) } = {f1, f3, f4} :=
sorry

end satisfies_negative_inverse_l77_77709


namespace geometric_sequence_x_l77_77032

theorem geometric_sequence_x (x : ℝ) (h : 1 * x = x ∧ x * x = 9) : x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_x_l77_77032


namespace value_of_expression_l77_77554

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 :=
by
  sorry

end value_of_expression_l77_77554


namespace total_elephants_l77_77176

-- Definitions and Hypotheses
def W : ℕ := 70
def G : ℕ := 3 * W

-- Proposition
theorem total_elephants : W + G = 280 := by
  sorry

end total_elephants_l77_77176


namespace math_competition_probs_l77_77766

-- Definitions related to the problem conditions
def boys : ℕ := 3
def girls : ℕ := 3
def total_students := boys + girls
def total_combinations := (total_students.choose 2)

-- Definition of the probabilities
noncomputable def prob_exactly_one_boy : ℚ := 0.6
noncomputable def prob_at_least_one_boy : ℚ := 0.8
noncomputable def prob_at_most_one_boy : ℚ := 0.8

-- Lean statement for the proof problem
theorem math_competition_probs :
  prob_exactly_one_boy = 0.6 ∧
  prob_at_least_one_boy = 0.8 ∧
  prob_at_most_one_boy = 0.8 :=
by
  sorry

end math_competition_probs_l77_77766


namespace molecular_weight_constant_l77_77456

-- Given the molecular weight of a compound
def molecular_weight (w : ℕ) := w = 1188

-- Statement about molecular weight of n moles
def weight_of_n_moles (n : ℕ) := n * 1188

theorem molecular_weight_constant (moles : ℕ) : 
  ∀ (w : ℕ), molecular_weight w → ∀ (n : ℕ), weight_of_n_moles n = n * w :=
by
  intro w h n
  sorry

end molecular_weight_constant_l77_77456


namespace greatest_3_digit_multiple_of_17_l77_77289

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77289


namespace ab_equals_6_l77_77107

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77107


namespace even_three_digit_numbers_less_than_600_l77_77440

def count_even_three_digit_numbers : ℕ :=
  let hundreds_choices := 5
  let tens_choices := 6
  let units_choices := 3
  hundreds_choices * tens_choices * units_choices

theorem even_three_digit_numbers_less_than_600 : count_even_three_digit_numbers = 90 := by
  -- sorry ensures that the statement type checks even without the proof.
  sorry

end even_three_digit_numbers_less_than_600_l77_77440


namespace average_monthly_balance_correct_l77_77333

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 250
def april_balance : ℕ := 250
def may_balance : ℕ := 150
def june_balance : ℕ := 100

def total_balance : ℕ :=
  january_balance + february_balance + march_balance + april_balance + may_balance + june_balance

def number_of_months : ℕ := 6

def average_monthly_balance : ℕ :=
  total_balance / number_of_months

theorem average_monthly_balance_correct :
  average_monthly_balance = 175 := by
  sorry

end average_monthly_balance_correct_l77_77333


namespace smallest_four_digit_multiple_of_18_l77_77525

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l77_77525


namespace evaluate_expression_l77_77856

theorem evaluate_expression :
  (15 - 14 + 13 - 12 + 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) / (1 - 2 + 3 - 4 + 5 - 6 + 7) = 2 :=
by
  sorry

end evaluate_expression_l77_77856


namespace vector_angle_acuteness_l77_77020

theorem vector_angle_acuteness (x : ℝ) : 
  ∀ (a b : ℝ × ℝ), a = (1, 2) ∧ b = (x, 4) → 
    (∃ (θ : ℝ), θ ∈ (0, π/2) ∧ 
      ∀ ⦃x : ℝ⦄, x > -8 ∧ x ≠ 2 → 
        ((x > -8 ∧ x < 2) ∨ (x > 2))) := 
by
  sorry

end vector_angle_acuteness_l77_77020


namespace ryan_days_learning_l77_77007

-- Definitions based on conditions
def hours_per_day_chinese : ℕ := 4
def total_hours_chinese : ℕ := 24

-- Theorem stating the number of days Ryan learns
theorem ryan_days_learning : total_hours_chinese / hours_per_day_chinese = 6 := 
by 
  -- Divide the total hours spent on Chinese learning by hours per day
  sorry

end ryan_days_learning_l77_77007


namespace initial_milk_quantity_l77_77434

theorem initial_milk_quantity 
  (milk_left_in_tank : ℕ) -- the remaining milk in the tank
  (pumping_rate : ℕ) -- the rate at which milk was pumped out
  (pumping_hours : ℕ) -- hours during which milk was pumped out
  (adding_rate : ℕ) -- the rate at which milk was added
  (adding_hours : ℕ) -- hours during which milk was added 
  (initial_milk : ℕ) -- initial milk collected
  (h1 : milk_left_in_tank = 28980) -- condition 3
  (h2 : pumping_rate = 2880) -- condition 1 (rate)
  (h3 : pumping_hours = 4) -- condition 1 (hours)
  (h4 : adding_rate = 1500) -- condition 2 (rate)
  (h5 : adding_hours = 7) -- condition 2 (hours)
  : initial_milk = 30000 :=
by
  sorry

end initial_milk_quantity_l77_77434


namespace largest_prime_factor_sum_of_four_digit_numbers_l77_77541

theorem largest_prime_factor_sum_of_four_digit_numbers 
  (a b c d : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
  (h3 : 1 ≤ b) (h4 : b ≤ 9) 
  (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 1 ≤ d) (h8 : d ≤ 9) 
  (h_diff : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  : Nat.gcd 6666 (a + b + c + d) = 101 :=
sorry

end largest_prime_factor_sum_of_four_digit_numbers_l77_77541


namespace ab_eq_six_l77_77082

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77082


namespace greatest_three_digit_multiple_of_17_l77_77197

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77197


namespace probability_at_least_one_six_l77_77684

theorem probability_at_least_one_six :
  let p_six := 1 / 6
  let p_not_six := 5 / 6
  let p_not_six_three_rolls := p_not_six ^ 3
  let p_at_least_one_six := 1 - p_not_six_three_rolls
  p_at_least_one_six = 91 / 216 :=
by
  sorry

end probability_at_least_one_six_l77_77684


namespace ab_equals_six_l77_77120

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77120


namespace speed_of_second_half_l77_77694

theorem speed_of_second_half (total_time : ℕ) (speed_first_half : ℕ) (total_distance : ℕ)
  (h1 : total_time = 15) (h2 : speed_first_half = 21) (h3 : total_distance = 336) :
  2 * total_distance / total_time - speed_first_half * (total_time / 2) / (total_time / 2) = 24 :=
by
  -- Proof omitted
  sorry

end speed_of_second_half_l77_77694


namespace power_product_to_seventh_power_l77_77042

theorem power_product_to_seventh_power :
  (2 ^ 14) * (2 ^ 21) = (32 ^ 7) :=
by
  sorry

end power_product_to_seventh_power_l77_77042


namespace roots_cubic_sum_l77_77919

theorem roots_cubic_sum:
  (∃ p q r : ℝ, 
     (p^3 - p^2 + p - 2 = 0) ∧ 
     (q^3 - q^2 + q - 2 = 0) ∧ 
     (r^3 - r^2 + r - 2 = 0)) 
  → 
  (∃ p q r : ℝ, p^3 + q^3 + r^3 = 4) := 
by 
  sorry

end roots_cubic_sum_l77_77919


namespace decimal_expansion_of_fraction_l77_77735

/-- 
Theorem: The decimal expansion of 13 / 375 is 0.034666...
-/
theorem decimal_expansion_of_fraction : 
  let numerator := 13
  let denominator := 375
  let resulting_fraction := (numerator * 2^3) / (denominator * 2^3)
  let decimal_expansion := 0.03466666666666667
  (resulting_fraction : ℝ) = decimal_expansion :=
sorry

end decimal_expansion_of_fraction_l77_77735


namespace greatest_three_digit_multiple_of_17_l77_77233

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77233


namespace time_to_chop_an_onion_is_4_minutes_l77_77700

noncomputable def time_to_chop_pepper := 3
noncomputable def time_to_grate_cheese_per_omelet := 1
noncomputable def time_to_cook_omelet := 5
noncomputable def peppers_needed := 4
noncomputable def onions_needed := 2
noncomputable def omelets_needed := 5
noncomputable def total_time := 50

theorem time_to_chop_an_onion_is_4_minutes : 
  (total_time - (peppers_needed * time_to_chop_pepper + omelets_needed * time_to_grate_cheese_per_omelet + omelets_needed * time_to_cook_omelet)) / onions_needed = 4 := by sorry

end time_to_chop_an_onion_is_4_minutes_l77_77700


namespace ratio_apples_pie_to_total_is_one_to_two_l77_77835

variable (x : ℕ) -- number of apples Paul put aside for pie
variable (total_apples : ℕ := 62) 
variable (fridge_apples : ℕ := 25)
variable (muffin_apples : ℕ := 6)

def apples_pie_ratio (x total_apples : ℕ) : ℕ := x / gcd x total_apples

theorem ratio_apples_pie_to_total_is_one_to_two :
  x + fridge_apples + muffin_apples = total_apples -> apples_pie_ratio x total_apples = 1 / 2 :=
by
  sorry

end ratio_apples_pie_to_total_is_one_to_two_l77_77835


namespace math_problem_l77_77844

noncomputable def x : ℝ := 24

theorem math_problem : ∀ (x : ℝ), x = 3/8 * x + 15 → x = 24 := 
by 
  intro x
  intro h
  sorry

end math_problem_l77_77844


namespace remainder_div_product_l77_77025

theorem remainder_div_product (P D D' D'' Q R Q' R' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = Q' * D' + R') 
  (h3 : Q' = Q'' * D'' + R'') :
  P % (D * D' * D'') = D * D' * R'' + D * R' + R := 
sorry

end remainder_div_product_l77_77025


namespace age_proof_l77_77416

theorem age_proof (A B C D : ℕ) 
  (h1 : A = D + 16)
  (h2 : B = D + 8)
  (h3 : C = D + 4)
  (h4 : A - 6 = 3 * (D - 6))
  (h5 : A - 6 = 2 * (B - 6))
  (h6 : A - 6 = (C - 6) + 4) 
  : A = 30 ∧ B = 22 ∧ C = 18 ∧ D = 14 :=
sorry

end age_proof_l77_77416


namespace total_rainfall_in_Springdale_l77_77503

theorem total_rainfall_in_Springdale
    (rainfall_first_week rainfall_second_week : ℝ)
    (h1 : rainfall_second_week = 1.5 * rainfall_first_week)
    (h2 : rainfall_second_week = 12) :
    (rainfall_first_week + rainfall_second_week = 20) :=
by
  sorry

end total_rainfall_in_Springdale_l77_77503


namespace plumber_total_cost_l77_77485

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end plumber_total_cost_l77_77485


namespace ab_eq_six_l77_77081

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77081


namespace find_x_perpendicular_l77_77553

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b (x : ℝ) : ℝ × ℝ := (-3, x)

-- Define the condition that the dot product of vectors a and b is zero
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Statement we need to prove
theorem find_x_perpendicular (x : ℝ) (h : perpendicular a (b x)) : x = -1 :=
by sorry

end find_x_perpendicular_l77_77553


namespace find_weekly_allowance_l77_77750

noncomputable def weekly_allowance (A : ℝ) : Prop :=
  let spent_at_arcade := (3/5) * A
  let remaining_after_arcade := A - spent_at_arcade
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  remaining_after_toy_store = 1.20

theorem find_weekly_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 4.50 := 
  sorry

end find_weekly_allowance_l77_77750


namespace solve_for_x_l77_77893

theorem solve_for_x (x : ℝ) (h1 : x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 := sorry

end solve_for_x_l77_77893


namespace greatest_three_digit_multiple_of_17_l77_77255

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77255


namespace probability_dice_sum_12_l77_77814

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := 25

theorem probability_dice_sum_12 :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 216 := by
  sorry

end probability_dice_sum_12_l77_77814


namespace initial_mat_weavers_l77_77800

variable (num_weavers : ℕ) (rate : ℕ → ℕ → ℕ) -- rate weaver_count duration_in_days → mats_woven

-- Given Conditions
def condition1 := rate num_weavers 4 = 4
def condition2 := rate (2 * num_weavers) 8 = 16

-- Theorem to Prove
theorem initial_mat_weavers : num_weavers = 4 :=
by
  sorry

end initial_mat_weavers_l77_77800


namespace total_surfers_calculation_l77_77431

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end total_surfers_calculation_l77_77431


namespace combined_total_capacity_l77_77813

theorem combined_total_capacity (A B C : ℝ) 
  (hA : 0.35 * A + 48 = 3 / 4 * A)
  (hB : 0.45 * B + 36 = 0.95 * B)
  (hC : 0.20 * C - 24 = 0.10 * C) :
  A + B + C = 432 := 
by 
  sorry

end combined_total_capacity_l77_77813


namespace initial_volume_of_solution_l77_77976

variable (V : ℝ)
variables (h1 : 0.10 * V = 0.08 * (V + 16))
variables (V_correct : V = 64)

theorem initial_volume_of_solution : V = 64 := by
  sorry

end initial_volume_of_solution_l77_77976


namespace distance_from_A_to_y_axis_l77_77171

variable (x y : ℝ)

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem distance_from_A_to_y_axis (hx : x = -2) (hy : y = 1) :
  distance_to_y_axis x = 2 :=
by
  rw [hx]
  simp [distance_to_y_axis]
  norm_num

#eval distance_from_A_to_y_axis (by rfl) (by rfl)

end distance_from_A_to_y_axis_l77_77171


namespace ab_eq_six_l77_77083

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77083


namespace apples_ratio_l77_77983

theorem apples_ratio (initial_apples rickis_apples end_apples samsons_apples : ℕ)
(h_initial : initial_apples = 74)
(h_ricki : rickis_apples = 14)
(h_end : end_apples = 32)
(h_samson : initial_apples - rickis_apples - end_apples = samsons_apples) :
  samsons_apples / Nat.gcd samsons_apples rickis_apples = 2 ∧ rickis_apples / Nat.gcd samsons_apples rickis_apples = 1 :=
by
  sorry

end apples_ratio_l77_77983


namespace ab_equals_six_l77_77122

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77122


namespace sufficient_but_not_necessary_condition_l77_77998

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l77_77998


namespace reciprocal_proof_l77_77981

theorem reciprocal_proof :
  (-2) * (-(1 / 2)) = 1 := 
by 
  sorry

end reciprocal_proof_l77_77981


namespace find_dividend_l77_77455

-- Definitions from conditions
def divisor : ℕ := 14
def quotient : ℕ := 12
def remainder : ℕ := 8

-- The problem statement to prove
theorem find_dividend : (divisor * quotient + remainder) = 176 := by
  sorry

end find_dividend_l77_77455


namespace ones_digit_exponent_73_l77_77010

theorem ones_digit_exponent_73 (n : ℕ) : 
  (73 ^ n) % 10 = 7 ↔ n % 4 = 3 := 
sorry

end ones_digit_exponent_73_l77_77010


namespace cos_x_plus_2y_eq_one_l77_77743

theorem cos_x_plus_2y_eq_one (x y a : ℝ) 
  (hx : -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4)
  (hy : -Real.pi / 4 ≤ y ∧ y ≤ Real.pi / 4)
  (h_eq1 : x^3 + Real.sin x - 2 * a = 0)
  (h_eq2 : 4 * y^3 + (1 / 2) * Real.sin (2 * y) + a = 0) : 
  Real.cos (x + 2 * y) = 1 := 
sorry -- Proof goes here

end cos_x_plus_2y_eq_one_l77_77743


namespace P_is_circumcenter_of_excenters_l77_77780

open EuclideanGeometry

structure Triangle :=
  (A B C : Point)

structure IncenterSystem (T : Triangle) :=
  (P : Point)
  (Pa : Point)
  (Pb : Point)
  (Pc : Point)
  (D : Point)
  (E : Point)
  (F : Point)
  (cond1 : Perp P D T.B T.C)
  (cond2 : Perp P E T.C T.A)
  (cond3 : Perp P F T.A T.B)
  (cond4 : (dist_sq P T.A) + (dist_sq P D) = (dist_sq P T.B) + (dist_sq P E))
  (cond5 : (dist_sq P T.B) + (dist_sq P E) = (dist_sq P T.C) + (dist_sq P F))

noncomputable def circumcenter_of_excenters (T : Triangle) (I_a I_b I_c : Point) :=
  ∃ (P : Point), IncenterSystem T ∧ P = circumcenter ⟨I_a, I_b, I_c⟩

theorem P_is_circumcenter_of_excenters (T : Triangle) (I_a I_b I_c : Point) 
  (h : ∃ (P : Point), IncenterSystem T ∧ P = circumcenter ⟨I_a, I_b, I_c⟩) :
  ∀ P, IncenterSystem T → P = circumcenter ⟨I_a, I_b, I_c⟩ := by
  intro P h_incenterSystem
  sorry

end P_is_circumcenter_of_excenters_l77_77780


namespace hector_gumballs_remaining_l77_77885

def gumballs_remaining (gumballs : ℕ) (given_todd : ℕ) (given_alisha : ℕ) (given_bobby : ℕ) : ℕ :=
  gumballs - (given_todd + given_alisha + given_bobby)

theorem hector_gumballs_remaining :
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  gumballs_remaining gumballs given_todd given_alisha given_bobby = 6 :=
by 
  let gumballs := 45
  let given_todd := 4
  let given_alisha := 2 * given_todd
  let given_bobby := 4 * given_alisha - 5
  show gumballs_remaining gumballs given_todd given_alisha given_bobby = 6
  sorry

end hector_gumballs_remaining_l77_77885


namespace probability_first_grade_probability_at_least_one_second_grade_l77_77378

-- Define conditions
def total_products : ℕ := 10
def first_grade_products : ℕ := 8
def second_grade_products : ℕ := 2
def inspected_products : ℕ := 2
def total_combinations : ℕ := Nat.choose total_products inspected_products
def first_grade_combinations : ℕ := Nat.choose first_grade_products inspected_products
def mixed_combinations : ℕ := first_grade_products * second_grade_products
def second_grade_combinations : ℕ := Nat.choose second_grade_products inspected_products

-- Define probabilities
def P_A : ℚ := first_grade_combinations / total_combinations
def P_B1 : ℚ := mixed_combinations / total_combinations
def P_B2 : ℚ := second_grade_combinations / total_combinations
def P_B : ℚ := P_B1 + P_B2

-- Statements
theorem probability_first_grade : P_A = 28 / 45 := sorry
theorem probability_at_least_one_second_grade : P_B = 17 / 45 := sorry

end probability_first_grade_probability_at_least_one_second_grade_l77_77378


namespace rectangles_with_perimeter_equals_area_l77_77860

theorem rectangles_with_perimeter_equals_area (a b : ℕ) (h : 2 * (a + b) = a * b) : (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 4 ∧ b = 4) :=
  sorry

end rectangles_with_perimeter_equals_area_l77_77860


namespace greatest_three_digit_multiple_of_17_l77_77212

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77212


namespace equivalent_equation_l77_77143

theorem equivalent_equation (x y : ℝ) 
  (x_ne_0 : x ≠ 0) (x_ne_3 : x ≠ 3) 
  (y_ne_0 : y ≠ 0) (y_ne_5 : y ≠ 5)
  (main_equation : (3 / x) + (4 / y) = 1 / 3) : 
  x = 9 * y / (y - 12) :=
sorry

end equivalent_equation_l77_77143


namespace greatest_three_digit_multiple_of_17_l77_77267

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77267


namespace T_shape_perimeter_l77_77962

/-- Two rectangles each measuring 3 inch × 5 inch are placed to form the letter T.
The overlapping area between the two rectangles is 1.5 inch. -/
theorem T_shape_perimeter:
  let l := 5 -- inches
  let w := 3 -- inches
  let overlap := 1.5 -- inches
  -- perimeter of one rectangle
  let P := 2 * (l + w)
  -- total perimeter accounting for overlap
  let total_perimeter := 2 * P - 2 * overlap
  total_perimeter = 29 :=
by
  sorry

end T_shape_perimeter_l77_77962


namespace greatest_three_digit_multiple_of_17_l77_77236

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77236


namespace football_team_matches_l77_77150

theorem football_team_matches (total_matches loses total_points: ℕ) 
  (points_win points_draw points_lose wins draws: ℕ)
  (h1: total_matches = 15)
  (h2: loses = 4)
  (h3: total_points = 29)
  (h4: points_win = 3)
  (h5: points_draw = 1)
  (h6: points_lose = 0)
  (h7: wins + draws + loses = total_matches)
  (h8: points_win * wins + points_draw * draws = total_points) :
  wins = 9 ∧ draws = 2 :=
sorry


end football_team_matches_l77_77150


namespace greatest_three_digit_multiple_of_17_l77_77234

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77234


namespace simplify_complex_fraction_l77_77797

theorem simplify_complex_fraction :
  (⟨-4, -6⟩ : ℂ) / (⟨5, -2⟩ : ℂ) = ⟨-(32 : ℚ) / 21, -(38 : ℚ) / 21⟩ := 
sorry

end simplify_complex_fraction_l77_77797


namespace georgia_vs_texas_license_plates_l77_77018

theorem georgia_vs_texas_license_plates :
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 :=
by
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  show georgia_plates - texas_plates = 731161600
  sorry

end georgia_vs_texas_license_plates_l77_77018


namespace remaining_area_exclude_smaller_rectangles_l77_77318

-- Conditions from part a)
variables (x : ℕ)
def large_rectangle_area := (x + 8) * (x + 6)
def small1_rectangle_area := (2 * x - 1) * (x - 1)
def small2_rectangle_area := (x - 3) * (x - 5)

-- Proof statement from part c)
theorem remaining_area_exclude_smaller_rectangles :
  large_rectangle_area x - (small1_rectangle_area x - small2_rectangle_area x) = 25 * x + 62 :=
by
  sorry

end remaining_area_exclude_smaller_rectangles_l77_77318


namespace greatest_three_digit_multiple_of_17_l77_77256

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77256


namespace probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l77_77979

theorem probability_one_piece_is_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) : 
  (if (piece_lengths.1 = 2 ∧ piece_lengths.2 ≠ 2) ∨ (piece_lengths.1 ≠ 2 ∧ piece_lengths.2 = 2) then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 2 / 5 :=
sorry

theorem probability_both_pieces_longer_than_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) :
  (if piece_lengths.1 > 2 ∧ piece_lengths.2 > 2 then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 1 / 3 :=
sorry

end probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l77_77979


namespace speed_of_boat_in_still_water_12_l77_77677

theorem speed_of_boat_in_still_water_12 (d b c : ℝ) (h1 : d = (b - c) * 5) (h2 : d = (b + c) * 3) (hb : b = 12) : b = 12 :=
by
  sorry

end speed_of_boat_in_still_water_12_l77_77677


namespace number_of_teachers_l77_77390

-- Definitions from the problem conditions
def num_students : Nat := 1500
def classes_per_student : Nat := 6
def classes_per_teacher : Nat := 5
def students_per_class : Nat := 25

-- The proof problem statement
theorem number_of_teachers : 
  (num_students * classes_per_student / students_per_class) / classes_per_teacher = 72 := by
  sorry

end number_of_teachers_l77_77390


namespace ab_value_l77_77045

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77045


namespace integral_sin_pi_over_2_to_pi_l77_77006

theorem integral_sin_pi_over_2_to_pi : ∫ x in (Real.pi / 2)..Real.pi, Real.sin x = 1 := by
  sorry

end integral_sin_pi_over_2_to_pi_l77_77006


namespace smallest_four_digit_multiple_of_18_l77_77527

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l77_77527


namespace ball_hits_ground_approx_time_l77_77948

-- Conditions
def height (t : ℝ) : ℝ := -6.1 * t^2 + 4.5 * t + 10

-- Main statement to be proved
theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, (height t = 0) ∧ (abs (t - 1.70) < 0.01) :=
sorry

end ball_hits_ground_approx_time_l77_77948


namespace find_number_l77_77610

theorem find_number (x : ℝ) : 
  (x + 72 = (2 * x) / (2 / 3)) → x = 36 :=
by
  intro h
  sorry

end find_number_l77_77610


namespace find_mnp_l77_77705

noncomputable def equation_rewrite (a b x y : ℝ) (m n p : ℕ): Prop :=
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 1) ∧
  (a^m * x - a^n) * (a^p * y - a^3) = a^5 * b^5

theorem find_mnp (a b x y : ℝ): 
  equation_rewrite a b x y 2 1 4 ∧ (2 * 1 * 4 = 8) :=
by 
  sorry

end find_mnp_l77_77705


namespace value_of_square_l77_77556

variable (x y : ℝ)

theorem value_of_square (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y) ^ 2 = 90 := sorry

end value_of_square_l77_77556


namespace greatest_three_digit_multiple_of_17_l77_77225

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77225


namespace greatest_three_digit_multiple_of_17_l77_77284

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77284


namespace coefficient_of_x_eq_2_l77_77881

variable (a : ℝ)

theorem coefficient_of_x_eq_2 (h : (5 * (-2)) + (4 * a) = 2) : a = 3 :=
sorry

end coefficient_of_x_eq_2_l77_77881


namespace slope_of_line_AF_parabola_l77_77169

theorem slope_of_line_AF_parabola (A : ℝ × ℝ)
  (hA_on_parabola : A.snd ^ 2 = 4 * A.fst)
  (h_dist_focus : Real.sqrt ((A.fst - 1) ^ 2 + A.snd ^ 2) = 4) :
  (A.snd / (A.fst - 1) = Real.sqrt 3 ∨ A.snd / (A.fst - 1) = -Real.sqrt 3) :=
sorry

end slope_of_line_AF_parabola_l77_77169


namespace find_value_added_l77_77612

open Classical

variable (n : ℕ) (avg_initial avg_final : ℝ)

-- Initial conditions
axiom avg_then_sum (n : ℕ) (avg : ℝ) : n * avg = 600

axiom avg_after_addition (n : ℕ) (avg : ℝ) : n * avg = 825

theorem find_value_added (n : ℕ) (avg_initial avg_final : ℝ) (h1 : n * avg_initial = 600) (h2 : n * avg_final = 825) :
  avg_final - avg_initial = 15 := by
  -- Proof goes here
  sorry

end find_value_added_l77_77612


namespace sum_of_square_face_is_13_l77_77915

-- Definitions based on conditions
variables (x₁ x₂ x₃ x₄ x₅ : ℕ)

-- Conditions
axiom h₁ : x₁ + x₂ + x₃ = 7
axiom h₂ : x₁ + x₂ + x₄ = 8
axiom h₃ : x₁ + x₃ + x₄ = 9
axiom h₄ : x₂ + x₃ + x₄ = 10

-- Properties
axiom h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15

-- Goal to prove
theorem sum_of_square_face_is_13 (h₁ : x₁ + x₂ + x₃ = 7) (h₂ : x₁ + x₂ + x₄ = 8) 
  (h₃ : x₁ + x₃ + x₄ = 9) (h₄ : x₂ + x₃ + x₄ = 10) (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15): 
  x₅ + x₁ + x₂ + x₄ = 13 :=
sorry

end sum_of_square_face_is_13_l77_77915


namespace lean_problem_l77_77740

noncomputable theory
open Classical

variables {Ω : Type*} {P : ProbabilityMeasure Ω}
variables {A B : Set Ω}

theorem lean_problem :
  P A = 0.6 ∧ P B = 0.2 →
  (B ⊆ A → P (A ∪ B) = 0.6) ∧
  (Disjoint A B → P (A ∪ B) = 0.8) ∧
  (Indep A B → P (Aᶜ ∩ Bᶜ) = 0.32) := 
by
  intros hP
  obtain ⟨hPA, hPB⟩ := hP
  split
  · intro hSubset
    rw [P.union_eq_left hSubset]
    exact hPA
  split
  · intro hDisjoint
    rw [P.union_eq_add_of_disjoint hDisjoint]
    rw [hPA, hPB]
    norm_num
  · intro hIndep
    rw [P.inter_compl_eq_prod_of_indep hIndep]
    rw [hPA, hPB]
    norm_num
  sorry

end lean_problem_l77_77740


namespace expected_coins_basilio_per_day_l77_77716

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end expected_coins_basilio_per_day_l77_77716


namespace find_larger_number_l77_77370

variable (x y : ℕ)

theorem find_larger_number (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 := 
by 
  sorry

end find_larger_number_l77_77370


namespace greatest_three_digit_multiple_of_17_l77_77214

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77214


namespace car_avg_mpg_B_to_C_is_11_11_l77_77679

noncomputable def avg_mpg_B_to_C (D : ℝ) : ℝ :=
  let avg_mpg_total := 42.857142857142854
  let x := (100 : ℝ) / 9
  let total_distance := (3 / 2) * D
  let total_gallons := (D / 40) + (D / (2 * x))
  (total_distance / total_gallons)

/-- Prove the car's average miles per gallon from town B to town C is 100/9 mpg. -/
theorem car_avg_mpg_B_to_C_is_11_11 (D : ℝ) (h1 : D > 0):
  avg_mpg_B_to_C D = 100 / 9 :=
by
  sorry

end car_avg_mpg_B_to_C_is_11_11_l77_77679


namespace ab_value_l77_77096

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77096


namespace acute_angle_implies_x_range_l77_77019

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end acute_angle_implies_x_range_l77_77019


namespace art_piece_increase_l77_77958

theorem art_piece_increase (initial_price : ℝ) (multiplier : ℝ) (future_increase : ℝ) (h1 : initial_price = 4000) (h2 : multiplier = 3) :
  future_increase = (multiplier * initial_price) - initial_price :=
by
  rw [h1, h2]
  norm_num
  sorry

end art_piece_increase_l77_77958


namespace ab_equals_six_l77_77129

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77129


namespace probability_merlin_dismissed_l77_77615

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l77_77615


namespace divide_square_into_smaller_squares_l77_77411

def P (n : Nat) : Prop :=
  ∃ f : ℕ → ℕ, (∀ m, m < n → f m ≠ 0) ∧ (∀ s, s ∈ finset.range n → s = n)

theorem divide_square_into_smaller_squares (n : Nat) (h : n > 5) : P n := sorry

end divide_square_into_smaller_squares_l77_77411


namespace function_range_of_roots_l77_77896

theorem function_range_of_roots (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : a > 1 := 
sorry

end function_range_of_roots_l77_77896


namespace range_of_a_l77_77927

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h_roots : x1 < 1 ∧ 1 < x2) (h_eq : ∀ x, x^2 + a * x - 2 = (x - x1) * (x - x2)) : a < 1 :=
sorry

end range_of_a_l77_77927


namespace greatest_three_digit_multiple_of_17_l77_77196

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77196


namespace probability_less_than_one_third_l77_77567

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l77_77567


namespace new_profit_percentage_l77_77842

def original_cost (c : ℝ) : ℝ := c
def original_selling_price (c : ℝ) : ℝ := 1.2 * c
def new_cost (c : ℝ) : ℝ := 0.9 * c
def new_selling_price (c : ℝ) : ℝ := 1.05 * 1.2 * c

theorem new_profit_percentage (c : ℝ) (hc : c > 0) :
  ((new_selling_price c - new_cost c) / new_cost c) * 100 = 40 :=
by
  sorry

end new_profit_percentage_l77_77842


namespace books_taken_out_on_Tuesday_l77_77812

theorem books_taken_out_on_Tuesday (T : ℕ) (initial_books : ℕ) (returned_books : ℕ) (withdrawn_books : ℕ) (final_books : ℕ) :
  initial_books = 250 ∧
  returned_books = 35 ∧
  withdrawn_books = 15 ∧
  final_books = 150 →
  T = 120 :=
by
  sorry

end books_taken_out_on_Tuesday_l77_77812


namespace average_price_per_book_l77_77794

theorem average_price_per_book (books1_cost : ℕ) (books1_count : ℕ)
    (books2_cost : ℕ) (books2_count : ℕ)
    (h1 : books1_cost = 6500) (h2 : books1_count = 65)
    (h3 : books2_cost = 2000) (h4 : books2_count = 35) :
    (books1_cost + books2_cost) / (books1_count + books2_count) = 85 :=
by
    sorry

end average_price_per_book_l77_77794


namespace original_decimal_number_l77_77475

theorem original_decimal_number (I : ℤ) (d : ℝ) (h1 : 0 ≤ d) (h2 : d < 1) (h3 : I + 4 * (I + d) = 21.2) : I + d = 4.3 :=
by
  sorry

end original_decimal_number_l77_77475


namespace cuboidal_box_area_l77_77804

/-- Given conditions about a cuboidal box:
    - The area of one face is 72 cm²
    - The area of an adjacent face is 60 cm²
    - The volume of the cuboidal box is 720 cm³,
    Prove that the area of the third adjacent face is 120 cm². -/
theorem cuboidal_box_area (l w h : ℝ) (h1 : l * w = 72) (h2 : w * h = 60) (h3 : l * w * h = 720) :
  l * h = 120 :=
sorry

end cuboidal_box_area_l77_77804


namespace area_of_rectangular_field_l77_77645

-- Define the conditions
variables (l w : ℝ)

def perimeter_condition : Prop := 2 * l + 2 * w = 100
def length_width_relation : Prop := l = 3 * w

-- Define the area
def area : ℝ := l * w

-- Prove the area given the conditions
theorem area_of_rectangular_field (h1 : perimeter_condition l w) (h2 : length_width_relation l w) : area l w = 468.75 :=
by sorry

end area_of_rectangular_field_l77_77645


namespace train_passing_time_l77_77752

-- Definitions based on the conditions
def length_T1 : ℕ := 800
def speed_T1_kmph : ℕ := 108
def length_T2 : ℕ := 600
def speed_T2_kmph : ℕ := 72

-- Converting kmph to mps
def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600
def speed_T1_mps : ℕ := convert_kmph_to_mps speed_T1_kmph
def speed_T2_mps : ℕ := convert_kmph_to_mps speed_T2_kmph

-- Calculating relative speed and total length
def relative_speed_T1_T2 : ℕ := speed_T1_mps - speed_T2_mps
def total_length_T1_T2 : ℕ := length_T1 + length_T2

-- Proving the time to pass
theorem train_passing_time : total_length_T1_T2 / relative_speed_T1_T2 = 140 := by
  sorry

end train_passing_time_l77_77752


namespace quadratic_inequality_l77_77761

theorem quadratic_inequality (a : ℝ) 
  (x₁ x₂ : ℝ) (h_roots : ∀ x, x^2 + (3 * a - 1) * x + a + 8 = 0) 
  (h_distinct : x₁ ≠ x₂)
  (h_x1_lt_1 : x₁ < 1) (h_x2_gt_1 : x₂ > 1) : 
  a < -2 := 
by
  sorry

end quadratic_inequality_l77_77761


namespace range_of_k_l77_77351

theorem range_of_k
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : a^2 + c^2 = 16)
  (h2 : b^2 + c^2 = 25) : 
  9 < a^2 + b^2 ∧ a^2 + b^2 < 41 :=
by
  sorry

end range_of_k_l77_77351


namespace sum_non_prime_between_50_60_eq_383_start_number_is_50_l77_77182

def is_non_prime (n : ℕ) : Prop := ∃ d, 1 < d ∧ d < n ∧ d ∣ n

def non_prime_numbers_between (start finish : ℕ) : List ℕ :=
  (List.range (finish - start - 1)).map (λ i => start + i + 1) |>.filter is_non_prime

theorem sum_non_prime_between_50_60_eq_383 : (non_prime_numbers_between 50 60).sum = 383 :=
sorry

theorem start_number_is_50 (n : ℕ) (hn : (non_prime_numbers_between n 60).sum = 383) : n = 50 :=
by {
  have h : non_prime_numbers_between 50 60 = [51, 52, 54, 55, 56, 57, 58] := 
  begin 
    sorry
  end,
  have sum_is_383 : (non_prime_numbers_between 50 60).sum = 383 := sum_non_prime_between_50_60_eq_383,
  have h₁ : (non_prime_numbers_between n 60).sum = (non_prime_numbers_between 50 60).sum := by rw hn,
  exact h₁.symm.trans sum_is_383
}

end sum_non_prime_between_50_60_eq_383_start_number_is_50_l77_77182


namespace modulo_sum_remainder_l77_77613

theorem modulo_sum_remainder (a b: ℤ) (k j: ℤ) 
  (h1 : a = 84 * k + 77) 
  (h2 : b = 120 * j + 113) :
  (a + b) % 42 = 22 := by
  sorry

end modulo_sum_remainder_l77_77613


namespace find_p_value_l77_77872

theorem find_p_value (D E F : ℚ) (α β : ℚ)
  (h₁: D ≠ 0) 
  (h₂: E^2 - 4*D*F ≥ 0) 
  (hαβ: D * (α^2 + β^2) + E * (α + β) + 2*F = 2*D^2 - E^2) :
  ∃ p : ℚ, (p = (2*D*F - E^2 - 2*D^2) / D^2) :=
sorry

end find_p_value_l77_77872


namespace ab_equals_six_l77_77128

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77128


namespace largest_c_for_range_of_f_l77_77863

def has_real_roots (a b c : ℝ) : Prop :=
  b * b - 4 * a * c ≥ 0

theorem largest_c_for_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x + c = 7) ↔ c ≤ 37 / 4 := by
  sorry

end largest_c_for_range_of_f_l77_77863


namespace f_is_periodic_f_nat_exact_l77_77548

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_eq (x y : ℝ) : f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom f_0_nonzero : f 0 ≠ 0
axiom f_1_zero : f 1 = 0

theorem f_is_periodic : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  by
    use 4
    sorry

theorem f_nat_exact (n : ℕ) : f n = Real.cos (n * Real.pi / 2) :=
  by
    sorry

end f_is_periodic_f_nat_exact_l77_77548


namespace domain_log_base_2_l77_77172

theorem domain_log_base_2 (x : ℝ) : (1 - x > 0) ↔ (x < 1) := by
  sorry

end domain_log_base_2_l77_77172


namespace find_a_value_l77_77744

theorem find_a_value 
  (a : ℝ)
  (h : abs (1 - (-1 / (4 * a))) = 2) :
  a = 1 / 4 ∨ a = -1 / 12 :=
sorry

end find_a_value_l77_77744


namespace son_work_rate_l77_77467

theorem son_work_rate (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : 1 / S = 20 :=
by
  sorry

end son_work_rate_l77_77467


namespace inequality_a_squared_plus_b_squared_l77_77739

variable (a b : ℝ)

theorem inequality_a_squared_plus_b_squared (h : a > b) : a^2 + b^2 > ab := 
sorry

end inequality_a_squared_plus_b_squared_l77_77739


namespace greatest_three_digit_multiple_of_17_l77_77240

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77240


namespace linda_babysitting_hours_l77_77396

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end linda_babysitting_hours_l77_77396


namespace find_k_l77_77352

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 1/a + 2/b = 1) : k = 18 := by
  sorry

end find_k_l77_77352


namespace circle_area_ratio_l77_77817

theorem circle_area_ratio (O X P : ℝ) (rOx rOp : ℝ) (h1 : rOx = rOp / 3) :
  (π * rOx^2) / (π * rOp^2) = 1 / 9 :=
by 
  -- Import required theorems and add assumptions as necessary
  -- Continue the proof based on Lean syntax
  sorry

end circle_area_ratio_l77_77817


namespace dice_probability_correct_l77_77866

-- Definitions of conditions
def is_standard_die (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6}
def valid_roll (a b c d e : ℕ) : Prop := is_standard_die a ∧ is_standard_die b ∧ is_standard_die c ∧ is_standard_die d ∧ is_standard_die e
def no_die_is_one (a b c d e : ℕ) : Prop := a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1 ∧ e ≠ 1
def sum_of_two_is_ten (a b c d e : ℕ) : Prop := (a + b = 10) ∨ (a + c = 10) ∨ (a + d = 10) ∨ (a + e = 10) ∨ (b + c = 10) ∨ (b + d = 10) ∨ (b + e = 10) ∨ (c + d = 10) ∨ (c + e = 10) ∨ (d + e = 10)

-- Probability calculation
noncomputable def probability (P : ℝ) : Prop :=
  ∃ a b c d e : ℕ,
    valid_roll a b c d e ∧ no_die_is_one a b c d e ∧ sum_of_two_is_ten a b c d e ∧
    P = ((5.0 / 6.0) ^ 5) * 10.0 * (1.0 / 12.0)

-- Final theorem statement
theorem dice_probability_correct : probability (2604.1667 / 7776) := sorry

end dice_probability_correct_l77_77866


namespace greatest_three_digit_multiple_of_17_l77_77278

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77278


namespace expected_coins_basilio_l77_77712

-- Define the number of courtyards and the probability of receiving a coin
def num_courtyards : ℕ := 20
def prob_coin : ℝ := 0.5

-- Define the random variable for the total number of coins received
noncomputable def X_Y : ℕ → ℝ := λ n, (n : ℝ) * prob_coin!

-- Expected value of the total number of coins received
noncomputable def E_X_Y : ℝ := X_Y num_courtyards

-- Define the random variable for the difference in coins received by Basilio and Alisa
noncomputable def X_Y_diff : ℕ → ℝ := λ _, (prob_coin)

-- Expected value of the difference in coins received
noncomputable def E_X_Y_diff : ℝ := X_Y_diff 1

-- Define the expected number of coins received by Basilio
noncomputable def E_X : ℝ := (E_X_Y + E_X_Y_diff) / 2

-- Prove the expected number of coins received by Basilio
theorem expected_coins_basilio : E_X = 5.25 := by
  -- using sorry to skip the proof
  sorry

end expected_coins_basilio_l77_77712


namespace money_split_l77_77710

theorem money_split (donna_share friend_share : ℝ) (h1 : donna_share = 32.50) (h2 : friend_share = 32.50) :
  donna_share + friend_share = 65 :=
by
  sorry

end money_split_l77_77710


namespace rectangle_difference_length_width_l77_77372

theorem rectangle_difference_length_width (x y p d : ℝ) (h1 : x + y = p / 2) (h2 : x^2 + y^2 = d^2) (h3 : x > y) : 
  x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 := sorry

end rectangle_difference_length_width_l77_77372


namespace ab_equals_6_l77_77098

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77098


namespace integer_solutions_m3_eq_n3_plus_n_l77_77731

theorem integer_solutions_m3_eq_n3_plus_n (m n : ℤ) (h : m^3 = n^3 + n) : m = 0 ∧ n = 0 :=
sorry

end integer_solutions_m3_eq_n3_plus_n_l77_77731


namespace geometric_sequence_common_ratio_eq_one_third_l77_77547

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio_eq_one_third
  (h_geom : geometric_sequence a_n q)
  (h_increasing : ∀ n, a_n n < a_n (n + 1))
  (h_a1 : a_n 1 = -2)
  (h_recurrence : ∀ n, 3 * (a_n n + a_n (n + 2)) = 10 * a_n (n + 1)) :
  q = 1 / 3 :=
by
  sorry

end geometric_sequence_common_ratio_eq_one_third_l77_77547


namespace total_cost_of_feeding_pets_for_one_week_l77_77639

-- Definitions based on conditions
def turtle_food_per_weight : ℚ := 1 / (1 / 2)
def turtle_weight : ℚ := 30
def turtle_food_qty_per_jar : ℚ := 15
def turtle_food_cost_per_jar : ℚ := 3

def bird_food_per_weight : ℚ := 2
def bird_weight : ℚ := 8
def bird_food_qty_per_bag : ℚ := 40
def bird_food_cost_per_bag : ℚ := 5

def hamster_food_per_weight : ℚ := 1.5 / (1 / 2)
def hamster_weight : ℚ := 3
def hamster_food_qty_per_box : ℚ := 20
def hamster_food_cost_per_box : ℚ := 4

-- Theorem stating the equivalent proof problem
theorem total_cost_of_feeding_pets_for_one_week :
  let turtle_food_needed := (turtle_weight * turtle_food_per_weight)
  let turtle_jars_needed := turtle_food_needed / turtle_food_qty_per_jar
  let turtle_cost := turtle_jars_needed * turtle_food_cost_per_jar
  let bird_food_needed := (bird_weight * bird_food_per_weight)
  let bird_bags_needed := bird_food_needed / bird_food_qty_per_bag
  let bird_cost := if bird_bags_needed < 1 then bird_food_cost_per_bag else bird_bags_needed * bird_food_cost_per_bag
  let hamster_food_needed := (hamster_weight * hamster_food_per_weight)
  let hamster_boxes_needed := hamster_food_needed / hamster_food_qty_per_box
  let hamster_cost := if hamster_boxes_needed < 1 then hamster_food_cost_per_box else hamster_boxes_needed * hamster_food_cost_per_box
  turtle_cost + bird_cost + hamster_cost = 21 :=
by
  sorry

end total_cost_of_feeding_pets_for_one_week_l77_77639


namespace sum_of_two_digit_divisors_l77_77917

theorem sum_of_two_digit_divisors (d : ℕ) (h_pos : d > 0) (h_mod : 145 % d = 4) : d = 47 := 
by sorry

end sum_of_two_digit_divisors_l77_77917


namespace ab_equals_six_l77_77119

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77119


namespace croissants_left_l77_77000

-- Definitions based on conditions
def total_croissants : ℕ := 17
def vegans : ℕ := 3
def allergic_to_chocolate : ℕ := 2
def any_type : ℕ := 2
def guests : ℕ := 7
def plain_needed : ℕ := vegans + allergic_to_chocolate
def plain_baked : ℕ := plain_needed
def choc_baked : ℕ := total_croissants - plain_baked

-- Assuming choc_baked > plain_baked as given
axiom croissants_greater_condition : choc_baked > plain_baked

-- Theorem to prove
theorem croissants_left (total_croissants vegans allergic_to_chocolate any_type guests : ℕ) 
    (plain_needed plain_baked choc_baked : ℕ) 
    (croissants_greater_condition : choc_baked > plain_baked) : 
    (choc_baked - guests + any_type) = 3 := 
by sorry

end croissants_left_l77_77000


namespace hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l77_77969

-- Define a hexagon with legal points and triangles

structure Hexagon :=
  (A B C D E F : ℝ)

-- Legal point occurs when certain conditions on intersection between diagonals hold
def legal_point (h : Hexagon) (x : ℝ) (y : ℝ) : Prop :=
  -- Placeholder, we need to define the exact condition based on problem constraints.
  sorry

-- Function to check if a division is legal based on defined rules
def legal_triangle_division (h : Hexagon) (n : ℕ) : Prop :=
  -- Placeholder, this requires a definition based on how points and triangles are formed
  sorry

-- Prove the specific cases
theorem hexagon_six_legal_triangles (h : Hexagon) : legal_triangle_division h 6 :=
  sorry

theorem hexagon_ten_legal_triangles (h : Hexagon) : legal_triangle_division h 10 :=
  sorry

theorem hexagon_two_thousand_fourteen_legal_triangles (h : Hexagon)  : legal_triangle_division h 2014 :=
  sorry

end hexagon_six_legal_triangles_hexagon_ten_legal_triangles_hexagon_two_thousand_fourteen_legal_triangles_l77_77969


namespace find_ages_of_siblings_l77_77188

-- Define the ages of the older brother and the younger sister as variables x and y
variables (x y : ℕ)

-- Define the conditions as provided in the problem
def condition1 : Prop := x = 4 * y
def condition2 : Prop := x + 3 = 3 * (y + 3)

-- State that the system of equations defined by condition1 and condition2 is consistent
theorem find_ages_of_siblings (x y : ℕ) (h1 : x = 4 * y) (h2 : x + 3 = 3 * (y + 3)) : 
  (x = 4 * y) ∧ (x + 3 = 3 * (y + 3)) :=
by 
  exact ⟨h1, h2⟩

end find_ages_of_siblings_l77_77188


namespace domain_and_range_of_f_l77_77035

noncomputable def f (a x : ℝ) : ℝ := Real.log (a - a * x) / Real.log a

theorem domain_and_range_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a - a * x > 0 → x < 1) ∧ 
  (∀ t : ℝ, 0 < t ∧ t < a → ∃ x : ℝ, t = a - a * x) :=
by
  sorry

end domain_and_range_of_f_l77_77035


namespace smallest_four_digit_multiple_of_18_l77_77517

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l77_77517


namespace price_per_foot_l77_77316

theorem price_per_foot (area : ℝ) (cost : ℝ) (side_length : ℝ) (perimeter : ℝ) 
  (h1 : area = 289) (h2 : cost = 3740) 
  (h3 : side_length^2 = area) (h4 : perimeter = 4 * side_length) : 
  (cost / perimeter = 55) :=
by
  sorry

end price_per_foot_l77_77316


namespace number_of_sides_l77_77590

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l77_77590


namespace ab_equals_six_l77_77057

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77057


namespace natural_number_triplets_l77_77732

theorem natural_number_triplets :
  ∀ (a b c : ℕ), a^3 + b^3 + c^3 = (a * b * c)^2 → 
    (a = 3 ∧ b = 2 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 1) ∨ (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) := 
by
  sorry

end natural_number_triplets_l77_77732


namespace quadratic_roots_condition_l77_77758

theorem quadratic_roots_condition (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ x₁^2 + (3 * a - 1) * x₁ + a + 8 = 0 ∧
  x₂^2 + (3 * a - 1) * x₂ + a + 8 = 0) → a < -2 :=
by
  sorry

end quadratic_roots_condition_l77_77758


namespace probability_of_more_twos_than_fives_eq_8_over_27_l77_77894

open ProbabilityTheory

noncomputable def probability_more_twos_than_fives : ℝ :=
  let num_faces := 6
  let num_dice := 3
  let total_outcomes := num_faces ^ num_dice
  let same_num_twos_and_fives_outcomes := 64 + 24
  let probability_same_num_twos_and_fives := same_num_twos_and_fives_outcomes / total_outcomes
  let probability_twos_eq_fives := probability_same_num_twos_and_fives
  in (1 - probability_twos_eq_fives) / 2

theorem probability_of_more_twos_than_fives_eq_8_over_27 :
  probability_more_twos_than_fives = 8 / 27 := 
by sorry

end probability_of_more_twos_than_fives_eq_8_over_27_l77_77894


namespace range_of_a_monotonically_decreasing_l77_77897

noncomputable def f (x a : ℝ) := x^3 - a * x^2 + 1

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < 2) ∧ (0 < y ∧ y < 2) → x < y → f x a ≥ f y a) → (a ≥ 3) :=
by
  sorry

end range_of_a_monotonically_decreasing_l77_77897


namespace expected_coins_basilio_20_l77_77721

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end expected_coins_basilio_20_l77_77721


namespace animals_per_aquarium_l77_77439

theorem animals_per_aquarium
  (saltwater_aquariums : ℕ)
  (saltwater_animals : ℕ)
  (h1 : saltwater_aquariums = 56)
  (h2 : saltwater_animals = 2184)
  : saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end animals_per_aquarium_l77_77439


namespace transform_eq_l77_77870

theorem transform_eq (m n x y : ℕ) (h1 : m + x = n + y) (h2 : x = y) : m = n :=
sorry

end transform_eq_l77_77870


namespace totalPeoplePresent_l77_77493

-- Defining the constants based on the problem conditions
def associateProfessors := 2
def assistantProfessors := 7

def totalPencils := 11
def totalCharts := 16

-- The main proof statement
theorem totalPeoplePresent :
  (∃ (A B : ℕ), (2 * A + B = totalPencils) ∧ (A + 2 * B = totalCharts)) →
  (associateProfessors + assistantProfessors = 9) :=
  by
  sorry

end totalPeoplePresent_l77_77493


namespace expected_value_coins_basilio_l77_77727

noncomputable def expected_coins_basilio (n : ℕ) (p : ℚ) : ℚ :=
  let X_Y := Binomial n p
  let E_X_Y := (n * p : ℚ)
  let X_minus_Y := 0.5
  (E_X_Y + X_minus_Y) / 2

theorem expected_value_coins_basilio :
  expected_coins_basilio 20 (1/2) = 5.25 := by
  sorry

end expected_value_coins_basilio_l77_77727


namespace total_baseball_fans_l77_77907

theorem total_baseball_fans (Y M B : ℕ)
  (h1 : Y = 3 / 2 * M)
  (h2 : M = 88)
  (h3 : B = 5 / 4 * M) :
  Y + M + B = 330 :=
by
  sorry

end total_baseball_fans_l77_77907


namespace inverse_mod_187_l77_77506

theorem inverse_mod_187 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 186 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end inverse_mod_187_l77_77506


namespace triangle_inequality_l77_77825

noncomputable def f (K : ℝ) (x : ℝ) : ℝ :=
  (x^4 + K * x^2 + 1) / (x^4 + x^2 + 1)

theorem triangle_inequality (K : ℝ) (a b c : ℝ) :
  (-1 / 2) < K ∧ K < 4 → ∃ (A B C : ℝ), A = f K a ∧ B = f K b ∧ C = f K c ∧ A + B > C ∧ A + C > B ∧ B + C > A :=
by
  sorry

end triangle_inequality_l77_77825


namespace dart_prob_center_square_l77_77474

noncomputable def hexagon_prob (s : ℝ) : ℝ :=
  let square_area := s^2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  square_area / hexagon_area

theorem dart_prob_center_square (s : ℝ) : hexagon_prob s = 2 * Real.sqrt 3 / 9 :=
by
  -- Proof omitted
  sorry

end dart_prob_center_square_l77_77474


namespace common_measure_angle_l77_77159

theorem common_measure_angle (α β : ℝ) (m n : ℕ) (h : α = β * (m / n)) : α / m = β / n :=
by 
  sorry

end common_measure_angle_l77_77159


namespace greatest_three_digit_multiple_of_17_l77_77210

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77210


namespace monotonicity_f_max_value_f_l77_77746

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 1

theorem monotonicity_f :
  (∀ x, 0 < x ∧ x < Real.exp 1 → f x < f (Real.exp 1)) ∧
  (∀ x, x > Real.exp 1 → f x < f (Real.exp 1)) :=
sorry

theorem max_value_f (m : ℝ) (hm : m > 0) :
  (2 * m ≤ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log (2 * m)) / (2 * m) - 1) ∧
  (m ≥ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log m) / m - 1) ∧
  (Real.exp 1 / 2 < m ∧ m < Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = 1 / Real.exp 1 - 1) :=
sorry

end monotonicity_f_max_value_f_l77_77746


namespace product_ab_l77_77140

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77140


namespace gcd_m_n_eq_one_l77_77340

/-- Mathematical definitions of m and n. --/
def m : ℕ := 123^2 + 235^2 + 347^2
def n : ℕ := 122^2 + 234^2 + 348^2

/-- Listing the conditions and deriving the result that gcd(m, n) = 1. --/
theorem gcd_m_n_eq_one : gcd m n = 1 :=
by sorry

end gcd_m_n_eq_one_l77_77340


namespace percentage_of_cobalt_is_15_l77_77686

-- Define the given percentages of lead and copper
def percent_lead : ℝ := 25
def percent_copper : ℝ := 60

-- Define the weights of lead and copper used in the mixture
def weight_lead : ℝ := 5
def weight_copper : ℝ := 12

-- Define the total weight of the mixture
def total_weight : ℝ := weight_lead + weight_copper

-- Prove that the percentage of cobalt is 15%
theorem percentage_of_cobalt_is_15 :
  (100 - (percent_lead + percent_copper) = 15) :=
by
  sorry

end percentage_of_cobalt_is_15_l77_77686


namespace probability_less_than_one_third_l77_77576

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l77_77576


namespace cos_sin_15_deg_l77_77014

theorem cos_sin_15_deg :
  400 * (Real.cos (15 * Real.pi / 180))^5 +  (Real.sin (15 * Real.pi / 180))^5 / (Real.cos (15 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 100 := 
sorry

end cos_sin_15_deg_l77_77014


namespace greatest_three_digit_multiple_of_17_l77_77195

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77195


namespace probability_of_interval_l77_77579

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l77_77579


namespace g_one_minus_g_four_l77_77419

theorem g_one_minus_g_four (g : ℝ → ℝ)
  (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ x : ℝ, g (x + 1) - g x = 5) :
  g 1 - g 4 = -15 :=
sorry

end g_one_minus_g_four_l77_77419


namespace sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l77_77423

theorem sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms :
    let a := 63
    let b := 25
    a + b = 88 := by
  sorry

end sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l77_77423


namespace jogger_distance_ahead_l77_77977

theorem jogger_distance_ahead
  (train_speed_km_hr : ℝ) (jogger_speed_km_hr : ℝ)
  (train_length_m : ℝ) (time_seconds : ℝ)
  (relative_speed_m_s : ℝ) (distance_covered_m : ℝ)
  (D : ℝ)
  (h1 : train_speed_km_hr = 45)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_length_m = 100)
  (h4 : time_seconds = 25)
  (h5 : relative_speed_m_s = 36 * (5/18))
  (h6 : distance_covered_m = 10 * 25)
  (h7 : D + train_length_m = distance_covered_m) :
  D = 150 :=
by sorry

end jogger_distance_ahead_l77_77977


namespace positive_integer_fraction_l77_77755

theorem positive_integer_fraction (p : ℕ) (h1 : p > 0) (h2 : (3 * p + 25) / (2 * p - 5) > 0) :
  3 ≤ p ∧ p ≤ 35 :=
by
  sorry

end positive_integer_fraction_l77_77755


namespace gas_tank_size_l77_77793

-- Conditions from part a)
def advertised_mileage : ℕ := 35
def actual_mileage : ℕ := 31
def total_miles_driven : ℕ := 372

-- Question and the correct answer in the context of conditions
theorem gas_tank_size (h1 : actual_mileage = advertised_mileage - 4) 
                      (h2 : total_miles_driven = 372) 
                      : total_miles_driven / actual_mileage = 12 := 
by sorry

end gas_tank_size_l77_77793


namespace smallest_four_digit_multiple_of_18_l77_77518

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l77_77518


namespace solve_for_x_l77_77383

theorem solve_for_x (h_perimeter_square : ∀(s : ℝ), 4 * s = 64)
  (h_height_triangle : ∀(h : ℝ), h = 48)
  (h_area_equal : ∀(s h x : ℝ), s * s = 1/2 * h * x) : 
  x = 32 / 3 := by
  sorry

end solve_for_x_l77_77383


namespace probability_merlin_dismissed_l77_77620

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l77_77620


namespace problem_a2014_l77_77770

-- Given conditions
def seq (a : ℕ → ℕ) := a 1 = 1 ∧ ∀ n, a (n + 1) = a n + 1

-- Prove the required statement
theorem problem_a2014 (a : ℕ → ℕ) (h : seq a) : a 2014 = 2014 :=
by sorry

end problem_a2014_l77_77770


namespace jordan_length_eq_six_l77_77704

def carol_length := 12
def carol_width := 15
def jordan_width := 30

theorem jordan_length_eq_six
  (h1 : carol_length * carol_width = jordan_width * jordan_length) : 
  jordan_length = 6 := by
  sorry

end jordan_length_eq_six_l77_77704


namespace Q_joined_after_4_months_l77_77934

namespace Business

-- Definitions
def P_cap := 4000
def Q_cap := 9000
def P_time := 12
def profit_ratio := (2 : ℚ) / 3

-- Statement to prove
theorem Q_joined_after_4_months (x : ℕ) (h : P_cap * P_time / (Q_cap * (12 - x)) = profit_ratio) :
  x = 4 := 
sorry

end Business

end Q_joined_after_4_months_l77_77934


namespace ab_equals_6_l77_77103

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77103


namespace exterior_angle_of_regular_polygon_l77_77595

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l77_77595


namespace Rickey_took_30_minutes_l77_77790

variables (R P : ℝ)

-- Define the conditions
def Prejean_speed_is_three_quarters_of_Rickey := P = 4 / 3 * R
def total_time_is_70 := R + P = 70

-- Define the statement to prove
theorem Rickey_took_30_minutes 
  (h1 : Prejean_speed_is_three_quarters_of_Rickey R P) 
  (h2 : total_time_is_70 R P) : R = 30 :=
by
  sorry

end Rickey_took_30_minutes_l77_77790


namespace binary_to_base4_conversion_l77_77706

theorem binary_to_base4_conversion : ∀ (a b c d e : ℕ), 
  1101101101 = (11 * 2^8) + (01 * 2^6) + (10 * 2^4) + (11 * 2^2) + 01 -> 
  a = 3 -> b = 1 -> c = 2 -> d = 3 -> e = 1 -> 
  (a*10000 + b*1000 + c*100 + d*10 + e : ℕ) = 31131 :=
by
  -- proof will go here
  sorry

end binary_to_base4_conversion_l77_77706


namespace regular_polygon_exterior_angle_l77_77587

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l77_77587


namespace probability_merlin_dismissed_l77_77622

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l77_77622


namespace product_ab_l77_77133

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77133


namespace smallest_positive_number_is_option_B_l77_77013

theorem smallest_positive_number_is_option_B :
  let A := 8 - 2 * Real.sqrt 17
  let B := 2 * Real.sqrt 17 - 8
  let C := 25 - 7 * Real.sqrt 5
  let D := 40 - 9 * Real.sqrt 2
  let E := 9 * Real.sqrt 2 - 40
  0 < B ∧ (A ≤ 0 ∨ B < A) ∧ (C ≤ 0 ∨ B < C) ∧ (D ≤ 0 ∨ B < D) ∧ (E ≤ 0 ∨ B < E) :=
by
  sorry

end smallest_positive_number_is_option_B_l77_77013


namespace greatest_three_digit_multiple_of_17_l77_77203

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77203


namespace exactly_one_root_in_interval_l77_77179

theorem exactly_one_root_in_interval (p q : ℝ) (h : q * (q + p + 1) < 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ (x^2 + p * x + q = 0) := sorry

end exactly_one_root_in_interval_l77_77179


namespace lowest_score_within_two_std_devs_l77_77829

variable (mean : ℝ) (std_dev : ℝ) (jack_score : ℝ)

def within_two_std_devs (mean : ℝ) (std_dev : ℝ) (score : ℝ) : Prop :=
  score >= mean - 2 * std_dev

theorem lowest_score_within_two_std_devs :
  mean = 60 → std_dev = 10 → within_two_std_devs mean std_dev jack_score → (40 ≤ jack_score) :=
by
  intros h1 h2 h3
  change mean = 60 at h1
  change std_dev = 10 at h2
  sorry

end lowest_score_within_two_std_devs_l77_77829


namespace smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l77_77848

theorem smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2 :
  ∀ (x y z w : ℝ),
    x = Real.sqrt 3 →
    y = -1 / 3 →
    z = -2 →
    w = 0 →
    (x > 1) ∧ (y < 0) ∧ (z < 0) ∧ (|y| = 1 / 3) ∧ (|z| = 2) ∧ (w = 0) →
    min (min (min x y) z) w = z :=
by
  intros x y z w hx hy hz hw hcond
  sorry

end smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l77_77848


namespace Kenny_jumping_jacks_wednesday_l77_77777

variable (Sunday Monday Tuesday Wednesday Thursday Friday Saturday : ℕ)
variable (LastWeekTotal : ℕ := 324)
variable (SundayJumpingJacks : ℕ := 34)
variable (MondayJumpingJacks : ℕ := 20)
variable (TuesdayJumpingJacks : ℕ := 0)
variable (SomeDayJumpingJacks : ℕ := 64)
variable (FridayJumpingJacks : ℕ := 23)
variable (SaturdayJumpingJacks : ℕ := 61)

def Kenny_jumping_jacks_this_week (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ) : ℕ :=
  SundayJumpingJacks + MondayJumpingJacks + TuesdayJumpingJacks + WednesdayJumpingJacks + ThursdayJumpingJacks + FridayJumpingJacks + SaturdayJumpingJacks

def Kenny_jumping_jacks_to_beat (weekTotal : ℕ) : ℕ :=
  LastWeekTotal + 1

theorem Kenny_jumping_jacks_wednesday : 
  ∃ (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ), 
  Kenny_jumping_jacks_this_week WednesdayJumpingJacks ThursdayJumpingJacks = LastWeekTotal + 1 ∧ 
  (WednesdayJumpingJacks = 59 ∧ ThursdayJumpingJacks = 64) ∨ (WednesdayJumpingJacks = 64 ∧ ThursdayJumpingJacks = 59) :=
by
  sorry

end Kenny_jumping_jacks_wednesday_l77_77777


namespace greatest_three_digit_multiple_of_17_l77_77269

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77269


namespace product_of_numbers_l77_77190

variable (x y : ℕ)

theorem product_of_numbers : x + y = 120 ∧ x - y = 6 → x * y = 3591 := by
  sorry

end product_of_numbers_l77_77190


namespace total_surfers_calculation_l77_77430

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end total_surfers_calculation_l77_77430


namespace greatest_three_digit_multiple_of_17_l77_77239

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77239


namespace slope_tangent_at_point_l77_77951

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem slope_tangent_at_point : (deriv f 1) = 1 := 
by
  sorry

end slope_tangent_at_point_l77_77951


namespace ab_equals_six_l77_77058

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77058


namespace ab_value_l77_77049

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77049


namespace Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l77_77637

def cost_supermarket_A (x : ℝ) : ℝ :=
  200 + 0.8 * (x - 200)

def cost_supermarket_B (x : ℝ) : ℝ :=
  100 + 0.85 * (x - 100)

theorem Li_Minghui_should_go_to_supermarket_B_for_300_yuan :
  cost_supermarket_B 300 < cost_supermarket_A 300 := by
  sorry

theorem cost_equal_for_500_yuan :
  cost_supermarket_A 500 = cost_supermarket_B 500 := by
  sorry

end Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l77_77637


namespace Mary_takes_3_children_l77_77787

def num_children (C : ℕ) : Prop :=
  ∃ (C : ℕ), 2 + C = 5

theorem Mary_takes_3_children (C : ℕ) : num_children C → C = 3 :=
by
  intro h
  sorry

end Mary_takes_3_children_l77_77787


namespace rhombus_diagonal_solution_l77_77682

variable (d1 : ℝ) (A : ℝ)

def rhombus_other_diagonal (d1 d2 A : ℝ) : Prop :=
  A = (d1 * d2) / 2

theorem rhombus_diagonal_solution (h1 : d1 = 16) (h2 : A = 80) : rhombus_other_diagonal d1 10 A :=
by
  rw [h1, h2]
  sorry

end rhombus_diagonal_solution_l77_77682


namespace teamA_worked_days_l77_77689

def teamA_days_to_complete := 10
def teamB_days_to_complete := 15
def teamC_days_to_complete := 20
def total_days := 6
def teamA_halfway_withdrew := true

theorem teamA_worked_days : 
  ∀ (T_A T_B T_C total: ℕ) (halfway_withdrawal: Bool),
    T_A = teamA_days_to_complete ->
    T_B = teamB_days_to_complete ->
    T_C = teamC_days_to_complete ->
    total = total_days ->
    halfway_withdrawal = teamA_halfway_withdrew ->
    (total / 2) * (1 / T_A + 1 / T_B + 1 / T_C) = 3 := 
by 
  sorry

end teamA_worked_days_l77_77689


namespace inequality_reciprocal_l77_77754

theorem inequality_reciprocal (a b : ℝ)
  (h : a * b > 0) : a > b ↔ 1 / a < 1 / b := 
sorry

end inequality_reciprocal_l77_77754


namespace cos_beta_minus_alpha_l77_77021

open Real

theorem cos_beta_minus_alpha :
  ∀ α β : ℝ,
  cos α = -3 / 5 ∧ α ∈ set.Ioo (π / 2) π ∧ sin β = -12 / 13 ∧ β ∈ set.Ioo π (3 * π / 2) →
  cos (β - α) = -33 / 65 :=
by
  intros α β h
  rcases h with ⟨hcosα, hα_range, hsinβ, hβ_range⟩
  sorry

end cos_beta_minus_alpha_l77_77021


namespace total_of_three_new_observations_l77_77806

theorem total_of_three_new_observations (avg9 : ℕ) (num9 : ℕ) 
(new_obs : ℕ) (new_avg_diff : ℕ) (new_num : ℕ) 
(total9 : ℕ) (new_avg : ℕ) (total12 : ℕ) : 
avg9 = 15 ∧ num9 = 9 ∧ new_obs = 3 ∧ new_avg_diff = 2 ∧
new_num = num9 + new_obs ∧ new_avg = avg9 - new_avg_diff ∧
total9 = num9 * avg9 ∧ total9 + 3 * (new_avg) = total12 → 
total12 - total9 = 21 := by sorry

end total_of_three_new_observations_l77_77806


namespace linda_babysitting_hours_l77_77395

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end linda_babysitting_hours_l77_77395


namespace divide_two_equal_parts_divide_four_equal_parts_l77_77949

-- the figure is bounded by three semicircles
def figure_bounded_by_semicircles 
-- two have the same radius r1 
(r1 r2 r3 : ℝ) 
-- the third has twice the radius r3 = 2 * r1
(h_eq : r3 = 2 * r1) 
-- Let's denote the figure as F
(F : Type) :=
-- conditions for r1 and r2
r1 > 0 ∧ r2 = r1 ∧ r3 = 2 * r1

-- Prove the figure can be divided into two equal parts.
theorem divide_two_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 : F), H1 ≠ H2 ∧ H1 = H2 :=
sorry

-- Prove the figure can be divided into four equal parts.
theorem divide_four_equal_parts 
{r1 r2 r3 : ℝ} 
{h_eq : r3 = 2 * r1} 
{F : Type} 
(h_bounded : figure_bounded_by_semicircles r1 r2 r3 h_eq F) : 
∃ (H1 H2 H3 H4 : F), H1 ≠ H2 ∧ H2 ≠ H3 ∧ H3 ≠ H4 ∧ H1 = H2 ∧ H2 = H3 ∧ H3 = H4 :=
sorry

end divide_two_equal_parts_divide_four_equal_parts_l77_77949


namespace months_after_withdrawal_and_advance_eq_eight_l77_77471

-- Define initial conditions
def initial_investment_A : ℝ := 3000
def initial_investment_B : ℝ := 4000
def withdrawal_A : ℝ := 1000
def advancement_B : ℝ := 1000
def total_profit : ℝ := 630
def share_A : ℝ := 240
def share_B : ℝ := total_profit - share_A

-- Define the main proof problem
theorem months_after_withdrawal_and_advance_eq_eight
  (initial_investment_A : ℝ) (initial_investment_B : ℝ)
  (withdrawal_A : ℝ) (advancement_B : ℝ)
  (total_profit : ℝ) (share_A : ℝ) (share_B : ℝ) : 
  ∃ x : ℝ, 
  (3000 * x + 2000 * (12 - x)) / (4000 * x + 5000 * (12 - x)) = 240 / 390 ∧
  x = 8 :=
sorry

end months_after_withdrawal_and_advance_eq_eight_l77_77471


namespace geometric_probability_l77_77586

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l77_77586


namespace abs_ineq_real_solution_range_l77_77899

theorem abs_ineq_real_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x + 3| < a) ↔ a > 7 :=
sorry

end abs_ineq_real_solution_range_l77_77899


namespace find_k_l77_77355

theorem find_k : ∀ (x y k : ℤ), (x = -y) → (2 * x + 5 * y = k) → (x - 3 * y = 16) → (k = -12) :=
by
  intros x y k h1 h2 h3
  sorry

end find_k_l77_77355


namespace range_of_m_l77_77891

theorem range_of_m : 3 < 3 * Real.sqrt 2 - 1 ∧ 3 * Real.sqrt 2 - 1 < 4 :=
by
  have h1 : 3 * Real.sqrt 2 < 5 := sorry
  have h2 : 4 < 3 * Real.sqrt 2 := sorry
  exact ⟨by linarith, by linarith⟩

end range_of_m_l77_77891


namespace max_zoo_area_l77_77655

theorem max_zoo_area (length width x y : ℝ) (h1 : length = 16) (h2 : width = 8 - x) (h3 : y = x * (8 - x)) : 
  ∃ M, ∀ x, 0 < x ∧ x < 8 → y ≤ M ∧ M = 16 :=
by
  sorry

end max_zoo_area_l77_77655


namespace ab_equals_six_l77_77124

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77124


namespace temperature_difference_l77_77422

theorem temperature_difference 
  (lowest: ℤ) (highest: ℤ) 
  (h_lowest : lowest = -4)
  (h_highest : highest = 5) :
  highest - lowest = 9 := 
by
  --relies on the correctness of problem and given simplyifying
  sorry

end temperature_difference_l77_77422


namespace range_of_m_l77_77890

theorem range_of_m : 
  ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := 
by
  -- the proof will go here
  sorry

end range_of_m_l77_77890


namespace area_of_quadrilateral_l77_77846

theorem area_of_quadrilateral (A B C : ℝ) (h1 : A + B = C) (h2 : A = 16) (h3 : B = 16) :
  (C - A - B) / 2 = 8 :=
by
  sorry

end area_of_quadrilateral_l77_77846


namespace carnival_friends_l77_77326

theorem carnival_friends (F : ℕ) (h1 : 865 % F ≠ 0) (h2 : 873 % F = 0) : F = 3 :=
by
  -- proof is not required
  sorry

end carnival_friends_l77_77326


namespace divide_fractions_l77_77004

theorem divide_fractions :
  (7 / 3) / (5 / 4) = (28 / 15) :=
by
  sorry

end divide_fractions_l77_77004


namespace geometric_sequence_a5_l77_77879

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (hq : q = 2) (h_a2a6 : a 2 * a 6 = 16) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_l77_77879


namespace solve_quadratic_eq_l77_77968

theorem solve_quadratic_eq (x : ℝ) :
  (3 * (2 * x + 1) = (2 * x + 1)^2) →
  (x = -1/2 ∨ x = 1) :=
by
  sorry

end solve_quadratic_eq_l77_77968


namespace inequality_holds_l77_77002

theorem inequality_holds (x y : ℝ) : (y - x^2 < abs x) ↔ (y < x^2 + abs x) := by
  sorry

end inequality_holds_l77_77002


namespace probability_merlin_dismissed_l77_77618

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l77_77618


namespace ab_value_l77_77043

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77043


namespace inequality_pos_distinct_l77_77937

theorem inequality_pos_distinct (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    (a + b + c) * (1/a + 1/b + 1/c) > 9 := by
  sorry

end inequality_pos_distinct_l77_77937


namespace intersection_of_sets_l77_77037

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets :
  setA ∩ setB = { z : ℝ | z ∈ [-1, 1] } :=
sorry

end intersection_of_sets_l77_77037


namespace ab_eq_six_l77_77080

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77080


namespace ice_cream_scoops_total_l77_77992

noncomputable def scoops_of_ice_cream : ℕ :=
let single_cone : ℕ := 1 in
let double_cone : ℕ := single_cone * 2 in
let banana_split : ℕ := single_cone * 3 in
let waffle_bowl : ℕ := banana_split + 1 in
single_cone + double_cone + banana_split + waffle_bowl

theorem ice_cream_scoops_total : scoops_of_ice_cream = 10 :=
sorry

end ice_cream_scoops_total_l77_77992


namespace housing_price_growth_l77_77375

theorem housing_price_growth (x : ℝ) (h₁ : (5500 : ℝ) > 0) (h₂ : (7000 : ℝ) > 0) :
  5500 * (1 + x) ^ 2 = 7000 := 
sorry

end housing_price_growth_l77_77375


namespace expected_coins_basilio_l77_77713

-- Define the number of courtyards and the probability of receiving a coin
def num_courtyards : ℕ := 20
def prob_coin : ℝ := 0.5

-- Define the random variable for the total number of coins received
noncomputable def X_Y : ℕ → ℝ := λ n, (n : ℝ) * prob_coin!

-- Expected value of the total number of coins received
noncomputable def E_X_Y : ℝ := X_Y num_courtyards

-- Define the random variable for the difference in coins received by Basilio and Alisa
noncomputable def X_Y_diff : ℕ → ℝ := λ _, (prob_coin)

-- Expected value of the difference in coins received
noncomputable def E_X_Y_diff : ℝ := X_Y_diff 1

-- Define the expected number of coins received by Basilio
noncomputable def E_X : ℝ := (E_X_Y + E_X_Y_diff) / 2

-- Prove the expected number of coins received by Basilio
theorem expected_coins_basilio : E_X = 5.25 := by
  -- using sorry to skip the proof
  sorry

end expected_coins_basilio_l77_77713


namespace find_u_l77_77017

theorem find_u (u : ℝ) : (∃ x : ℝ, x = ( -15 - Real.sqrt 145 ) / 8 ∧ 4 * x^2 + 15 * x + u = 0) ↔ u = 5 := by
  sorry

end find_u_l77_77017


namespace triangle_area_l77_77191

theorem triangle_area :
  let A := (2, -3)
  let B := (2, 4)
  let C := (8, 0) 
  let base := (4 - (-3))
  let height := (8 - 2)
  let area := (1 / 2) * base * height
  area = 21 := 
by 
  sorry

end triangle_area_l77_77191


namespace greatest_three_digit_multiple_of_17_l77_77276

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77276


namespace greatest_three_digit_multiple_of_17_l77_77253

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77253


namespace greatest_three_digit_multiple_of_17_l77_77244

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77244


namespace sum_in_correct_range_l77_77703

-- Define the mixed numbers
def mixed1 := 1 + 1/4
def mixed2 := 4 + 1/3
def mixed3 := 6 + 1/12

-- Their sum
def sumMixed := mixed1 + mixed2 + mixed3

-- Correct sum in mixed number form
def correctSum := 11 + 2/3

-- Range we need to check
def lowerBound := 11 + 1/2
def upperBound := 12

theorem sum_in_correct_range : sumMixed = correctSum ∧ lowerBound < correctSum ∧ correctSum < upperBound := by
  sorry

end sum_in_correct_range_l77_77703


namespace ab_equals_6_l77_77099

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77099


namespace parallelogram_to_rhombus_l77_77816

theorem parallelogram_to_rhombus {a b m1 m2 x : ℝ} (h_area : a * m1 = x * m2) (h_proportion : b / m1 = x / m2) : x = Real.sqrt (a * b) :=
by
  -- Proof goes here
  sorry

end parallelogram_to_rhombus_l77_77816


namespace total_scoops_l77_77991

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l77_77991


namespace spent_on_video_game_l77_77388

def saved_September : ℕ := 30
def saved_October : ℕ := 49
def saved_November : ℕ := 46
def money_left : ℕ := 67
def total_saved := saved_September + saved_October + saved_November

theorem spent_on_video_game : total_saved - money_left = 58 := by
  -- proof steps go here
  sorry

end spent_on_video_game_l77_77388


namespace minimum_value_of_xy_l77_77371

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  18 ≤ x * y :=
sorry

end minimum_value_of_xy_l77_77371


namespace flowchart_structure_correct_l77_77303

-- Definitions based on conditions
def flowchart_typically_has_one_start : Prop :=
  ∃ (start : Nat), start = 1

def flowchart_typically_has_one_or_more_ends : Prop :=
  ∃ (ends : Nat), ends ≥ 1

-- Theorem for the correct statement
theorem flowchart_structure_correct :
  (flowchart_typically_has_one_start ∧ flowchart_typically_has_one_or_more_ends) →
  (∃ (start : Nat) (ends : Nat), start = 1 ∧ ends ≥ 1) :=
by
  sorry

end flowchart_structure_correct_l77_77303


namespace largest_three_digit_multiple_of_17_l77_77218

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77218


namespace total_amount_is_4200_l77_77670

variables (p q r : ℕ)
variable (total_amount : ℕ)
variable (r_has_two_thirds : total_amount / 3 * 2 = 2800)
variable (r_value : r = 2800)

theorem total_amount_is_4200 (h1 : total_amount / 3 * 2 = 2800) (h2 : r = 2800) : total_amount = 4200 :=
by
  sorry

end total_amount_is_4200_l77_77670


namespace greatest_three_digit_multiple_of_17_l77_77215

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77215


namespace solve_for_nabla_l77_77558

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 := 
by
  sorry

end solve_for_nabla_l77_77558


namespace largest_common_multiple_3_5_l77_77507

theorem largest_common_multiple_3_5 (n : ℕ) :
  (n < 10000) ∧ (n ≥ 1000) ∧ (n % 3 = 0) ∧ (n % 5 = 0) → n ≤ 9990 :=
sorry

end largest_common_multiple_3_5_l77_77507


namespace mary_balloons_correct_l77_77788

-- Define the number of black balloons Nancy has
def nancy_balloons : ℕ := 7

-- Define the multiplier that represents how many times more balloons Mary has compared to Nancy
def multiplier : ℕ := 4

-- Define the number of black balloons Mary has in terms of Nancy's balloons and the multiplier
def mary_balloons : ℕ := nancy_balloons * multiplier

-- The statement we want to prove
theorem mary_balloons_correct : mary_balloons = 28 :=
by
  sorry

end mary_balloons_correct_l77_77788


namespace large_apple_probability_l77_77909

open ProbabilityTheory

variables (A1 A2 B : Prop)

def P (e : Prop) [MeasureTheory.ProbabilityMeasure e] := MeasureTheory.measure e

variables (hA1 : P A1 = 9 / 10)
          (hA2 : P A2 = 1 / 10)
          (hBA1 : P (B | A1) = 19 / 20)
          (hBA2 : P (B | A2) = 1 / 50)

theorem large_apple_probability :
  P (A1 | B) = 855 / 857 :=
by
  sorry

end large_apple_probability_l77_77909


namespace largest_number_in_L_shape_l77_77697

theorem largest_number_in_L_shape (x : ℤ) (sum : ℤ) (h : sum = 2015) : x = 676 :=
by
  sorry

end largest_number_in_L_shape_l77_77697


namespace ab_equals_six_l77_77117

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77117


namespace average_annual_cost_reduction_l77_77189

theorem average_annual_cost_reduction (x : ℝ) (h : (1 - x) ^ 2 = 0.64) : x = 0.2 :=
sorry

end average_annual_cost_reduction_l77_77189


namespace correct_judgement_l77_77895

noncomputable def f (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 2 then (1 / 2) * Real.sqrt (4 - x^2)
else - (1 / 2) * Real.sqrt (x^2 - 4)

noncomputable def F (x : ℝ) : ℝ := f x + x

theorem correct_judgement : (∀ y : ℝ, ∃ x : ℝ, (f x = y) ↔ (y ∈ Set.Iic 1)) ∧ (∃! x : ℝ, F x = 0) :=
by
  sorry

end correct_judgement_l77_77895


namespace distance_from_A_to_y_axis_is_2_l77_77170

-- Define the point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- The theorem to prove
theorem distance_from_A_to_y_axis_is_2 : distance_to_y_axis point_A = 2 :=
by
  sorry

end distance_from_A_to_y_axis_is_2_l77_77170


namespace box_volume_l77_77480

theorem box_volume (length width side_len : ℕ) (h1 : length = 48) (h2 : width = 36) (h3 : side_len = 8) :
  (length - 2 * side_len) * (width - 2 * side_len) * side_len = 5120 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l77_77480


namespace open_box_volume_l77_77479

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l77_77479


namespace total_games_played_l77_77775

theorem total_games_played (games_attended games_missed : ℕ) 
  (h_attended : games_attended = 395) 
  (h_missed : games_missed = 469) : 
  games_attended + games_missed = 864 := 
by
  sorry

end total_games_played_l77_77775


namespace find_y_l77_77971

theorem find_y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) (h4 : z = 2) : y = 3 :=
    sorry

end find_y_l77_77971


namespace golf_balls_count_l77_77435

theorem golf_balls_count (dozen_count : ℕ) (balls_per_dozen : ℕ) (total_balls : ℕ) 
  (h1 : dozen_count = 13) 
  (h2 : balls_per_dozen = 12) 
  (h3 : total_balls = dozen_count * balls_per_dozen) : 
  total_balls = 156 := 
sorry

end golf_balls_count_l77_77435


namespace open_box_volume_l77_77478

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l77_77478


namespace greatest_three_digit_multiple_of_17_l77_77279

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77279


namespace common_number_in_lists_l77_77840

theorem common_number_in_lists (nums : List ℚ) (h_len : nums.length = 9)
  (h_first_five_avg : (nums.take 5).sum / 5 = 7)
  (h_last_five_avg : (nums.drop 4).sum / 5 = 9)
  (h_total_avg : nums.sum / 9 = 73/9) :
  ∃ x, x ∈ nums.take 5 ∧ x ∈ nums.drop 4 ∧ x = 7 := 
sorry

end common_number_in_lists_l77_77840


namespace greatest_three_digit_multiple_of_17_l77_77243

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77243


namespace stratified_sampling_model_A_l77_77683

theorem stratified_sampling_model_A (r_A r_B r_C n x : ℕ) 
  (r_A_eq : r_A = 2) (r_B_eq : r_B = 3) (r_C_eq : r_C = 5) 
  (n_eq : n = 80) : 
  (r_A * n / (r_A + r_B + r_C) = x) -> x = 16 := 
by 
  intros h
  rw [r_A_eq, r_B_eq, r_C_eq, n_eq] at h
  norm_num at h
  exact h.symm

end stratified_sampling_model_A_l77_77683


namespace product_ab_l77_77137

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77137


namespace find_integer_pairs_l77_77973

theorem find_integer_pairs :
  { (m, n) : ℤ × ℤ | n^3 + m^3 + 231 = n^2 * m^2 + n * m } = {(4, 5), (5, 4)} :=
by
  sorry

end find_integer_pairs_l77_77973


namespace positive_function_characterization_l77_77859

theorem positive_function_characterization (f : ℝ → ℝ) (h₁ : ∀ x, x > 0 → f x > 0) (h₂ : ∀ a b : ℝ, a > 0 → b > 0 → a * b ≤ 0.5 * (a * f a + b * (f b)⁻¹)) :
  ∃ C > 0, ∀ x > 0, f x = C * x :=
sorry

end positive_function_characterization_l77_77859


namespace relationship_y1_y2_y3_l77_77898

variables {m y_1 y_2 y_3 : ℝ}

theorem relationship_y1_y2_y3 :
  (∃ (m : ℝ), (y_1 = (-1)^2 - 2*(-1) + m) ∧ (y_2 = 2^2 - 2*2 + m) ∧ (y_3 = 3^2 - 2*3 + m)) →
  y_2 < y_1 ∧ y_1 = y_3 :=
by
  sorry

end relationship_y1_y2_y3_l77_77898


namespace daisies_bought_l77_77324

-- Definitions from the given conditions
def cost_per_flower : ℕ := 6
def num_roses : ℕ := 7
def total_spent : ℕ := 60

-- Proving the number of daisies Maria bought
theorem daisies_bought : ∃ (D : ℕ), D = 3 ∧ total_spent = num_roses * cost_per_flower + D * cost_per_flower :=
by
  sorry

end daisies_bought_l77_77324


namespace team_a_wins_3_2_prob_l77_77972

-- Definitions for the conditions in the problem
def prob_win_first_four : ℚ := 2 / 3
def prob_win_fifth : ℚ := 1 / 2

-- Definitions related to combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Main statement: Proving the probability of winning 3:2
theorem team_a_wins_3_2_prob :
  (C 4 2 * (prob_win_first_four ^ 2) * ((1 - prob_win_first_four) ^ 2) * prob_win_fifth) = 4 / 27 := 
sorry

end team_a_wins_3_2_prob_l77_77972


namespace total_scoops_l77_77990

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l77_77990


namespace greatest_three_digit_multiple_of_17_l77_77206

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77206


namespace smallest_cube_ends_in_584_l77_77511

theorem smallest_cube_ends_in_584 (n : ℕ) : n^3 ≡ 584 [MOD 1000] → n = 34 := by
  sorry

end smallest_cube_ends_in_584_l77_77511


namespace product_ab_l77_77132

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77132


namespace greatest_three_digit_multiple_of_17_l77_77275

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77275


namespace factor_ax2_minus_ay2_l77_77426

variable (a x y : ℝ)

theorem factor_ax2_minus_ay2 : a * x^2 - a * y^2 = a * (x + y) * (x - y) := 
sorry

end factor_ax2_minus_ay2_l77_77426


namespace largest_k_value_l77_77341

theorem largest_k_value (a b c d : ℕ) (k : ℝ)
  (h1 : a + b = c + d)
  (h2 : 2 * (a * b) = c * d)
  (h3 : a ≥ b) :
  (∀ k', (∀ a b (h1_b : a + b = c + d)
              (h2_b : 2 * a * b = c * d)
              (h3_b : a ≥ b), (a : ℝ) / (b : ℝ) ≥ k') → k' ≤ k) → k = 3 + 2 * Real.sqrt 2 :=
sorry

end largest_k_value_l77_77341


namespace total_elephants_in_two_parks_is_280_l77_77178

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end total_elephants_in_two_parks_is_280_l77_77178


namespace regular_hexagon_area_l77_77454

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l77_77454


namespace angle_DNE_l77_77961

theorem angle_DNE (DE EF FD : ℝ) (EFD END FND : ℝ) 
  (h1 : DE = 2 * EF) 
  (h2 : EF = FD) 
  (h3 : EFD = 34) 
  (h4 : END = 3) 
  (h5 : FND = 18) : 
  ∃ DNE : ℝ, DNE = 104 :=
by 
  sorry

end angle_DNE_l77_77961


namespace greatest_three_digit_multiple_of_17_l77_77259

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77259


namespace greatest_three_digit_multiple_of_17_l77_77268

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77268


namespace class_average_score_l77_77332

theorem class_average_score :
  let total_students := 40
  let absent_students := 2
  let present_students := total_students - absent_students
  let initial_avg := 92
  let absent_scores := [100, 100]
  let initial_total_score := initial_avg * present_students
  let total_final_score := initial_total_score + absent_scores.sum
  let final_avg := total_final_score / total_students
  final_avg = 92.4 := by
  sorry

end class_average_score_l77_77332


namespace joan_seashells_count_l77_77914

variable (total_seashells_given_to_sam : ℕ) (seashells_left_with_joan : ℕ)

theorem joan_seashells_count
  (h_given : total_seashells_given_to_sam = 43)
  (h_left : seashells_left_with_joan = 27) :
  total_seashells_given_to_sam + seashells_left_with_joan = 70 :=
sorry

end joan_seashells_count_l77_77914


namespace train_cross_pole_time_l77_77488

-- Definitions based on the conditions
def train_speed_kmh := 54
def train_length_m := 105
def train_speed_ms := (train_speed_kmh * 1000) / 3600
def expected_time := train_length_m / train_speed_ms

-- Theorem statement, encapsulating the problem
theorem train_cross_pole_time : expected_time = 7 := by
  sorry

end train_cross_pole_time_l77_77488


namespace greatest_three_digit_multiple_of_17_l77_77286

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77286


namespace range_of_a_l77_77742

def A (x : ℝ) : Prop := abs (x - 4) < 2 * x

def B (x a : ℝ) : Prop := x * (x - a) ≥ (a + 6) * (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) → a ≤ -14 / 3 :=
  sorry

end range_of_a_l77_77742


namespace scoops_for_mom_l77_77163

/-- 
  Each scoop of ice cream costs $2.
  Pierre gets 3 scoops.
  The total bill is $14.
  Prove that Pierre's mom gets 4 scoops.
-/
theorem scoops_for_mom
  (scoop_cost : ℕ)
  (pierre_scoops : ℕ)
  (total_bill : ℕ) :
  scoop_cost = 2 → pierre_scoops = 3 → total_bill = 14 → 
  (total_bill - pierre_scoops * scoop_cost) / scoop_cost = 4 := 
by
  intros h1 h2 h3
  sorry

end scoops_for_mom_l77_77163


namespace greatest_three_digit_multiple_of_17_l77_77250

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77250


namespace ratio_diagonals_to_sides_l77_77644

-- Definition of the number of diagonals in a polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the condition
def n : ℕ := 5

-- Proof statement that the ratio of the number of diagonals to the number of sides is 1
theorem ratio_diagonals_to_sides (n_eq_5 : n = 5) : 
  (number_of_diagonals n) / n = 1 :=
by {
  -- Proof would go here, but is omitted
  sorry
}

end ratio_diagonals_to_sides_l77_77644


namespace geometric_sequence_ratio_l77_77928

noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio 
  (S : ℕ → ℝ) 
  (hS12 : S 12 = 1)
  (hS6 : S 6 = 2)
  (geom_property : ∀ a r, (S n = a * (1 - r^n) / (1 - r))) :
  S 18 / S 6 = 3 / 4 :=
by
  sorry

end geometric_sequence_ratio_l77_77928


namespace greatest_three_digit_multiple_of_17_l77_77271

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77271


namespace craig_distance_ridden_farther_l77_77654

/-- Given that Craig rode the bus for 3.83 miles and walked for 0.17 miles,
    prove that the distance he rode farther than he walked is 3.66 miles. -/
theorem craig_distance_ridden_farther :
  let distance_bus := 3.83
  let distance_walked := 0.17
  distance_bus - distance_walked = 3.66 :=
by
  let distance_bus := 3.83
  let distance_walked := 0.17
  show distance_bus - distance_walked = 3.66
  sorry

end craig_distance_ridden_farther_l77_77654


namespace rickey_time_l77_77791

/-- Prejean's speed in a race was three-quarters that of Rickey. 
If they both took a total of 70 minutes to run the race, 
the total number of minutes that Rickey took to finish the race is 40. -/
theorem rickey_time (t : ℝ) (h1 : ∀ p : ℝ, p = (3/4) * t) (h2 : t + (3/4) * t = 70) : t = 40 := 
by
  sorry

end rickey_time_l77_77791


namespace range_of_a_l77_77146

theorem range_of_a :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := 
by
  sorry

end range_of_a_l77_77146


namespace one_belt_one_road_l77_77537

theorem one_belt_one_road (m n : ℝ) :
  (∀ x y : ℝ, y = x^2 - 2 * x + n ↔ (x, y) ∈ { p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 1 }) →
  (∀ x y : ℝ, y = m * x + 1 ↔ (x, y) ∈ { q : ℝ × ℝ | q.1 = 0 ∧ q.2 = 1 }) →
  (∀ x y : ℝ, y = x^2 - 2 * x + 1 → y = 0) →
  m = -1 ∧ n = 1 :=
by
  intros h1 h2 h3
  sorry

end one_belt_one_road_l77_77537


namespace blankets_warm_nathan_up_l77_77298

theorem blankets_warm_nathan_up :
  (∀ (blankets_added half_blankets: ℕ), blankets_added = (half_blankets) -> half_blankets = 7 -> ∃ warmth: ℕ, warmth = blankets_added * 3  ∧ warmth = 21) :=
begin
  intros,
  sorry
end

end blankets_warm_nathan_up_l77_77298


namespace arrange_descending_order_l77_77869

noncomputable def a := 8 ^ 0.7
noncomputable def b := 8 ^ 0.9
noncomputable def c := 2 ^ 0.8

theorem arrange_descending_order :
    b > a ∧ a > c := by
  sorry

end arrange_descending_order_l77_77869


namespace expected_coins_basilio_20_l77_77719

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end expected_coins_basilio_20_l77_77719


namespace exterior_angle_polygon_l77_77596

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l77_77596


namespace determine_y_l77_77144

variable (x y : ℝ)

theorem determine_y (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := 
by 
  sorry

end determine_y_l77_77144


namespace tan_beta_is_six_over_seventeen_l77_77348
-- Import the Mathlib library

-- Define the problem in Lean
theorem tan_beta_is_six_over_seventeen
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 4 / 5)
  (h2 : Real.tan (α - β) = 2 / 3) :
  Real.tan β = 6 / 17 := 
by
  sorry

end tan_beta_is_six_over_seventeen_l77_77348


namespace find_q_l77_77469

theorem find_q (q : ℤ) (x : ℤ) (y : ℤ) (h1 : x = 55 + 2 * q) (h2 : y = 4 * q + 41) (h3 : x = y) : q = 7 :=
by
  sorry

end find_q_l77_77469


namespace system_of_equations_correct_l77_77603

def question_statement (x y : ℕ) : Prop :=
  x + y = 12 ∧ 6 * x = 3 * 4 * y

theorem system_of_equations_correct
  (x y : ℕ)
  (h1 : x + y = 12)
  (h2 : 6 * x = 3 * 4 * y)
: question_statement x y :=
by
  unfold question_statement
  exact ⟨h1, h2⟩

end system_of_equations_correct_l77_77603


namespace cost_of_one_book_l77_77160

theorem cost_of_one_book (m : ℕ) (H1: 1100 < 900 + 9 * m ∧ 900 + 9 * m < 1200)
                                (H2: 1500 < 1300 + 13 * m ∧ 1300 + 13 * m < 1600) : 
                                m = 23 :=
by {
  sorry
}

end cost_of_one_book_l77_77160


namespace product_divisible_by_15_l77_77707

theorem product_divisible_by_15 (n : ℕ) (hn1 : n % 2 = 1) (hn2 : n > 0) :
  15 ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end product_divisible_by_15_l77_77707


namespace solve_logarithmic_inequality_l77_77649

theorem solve_logarithmic_inequality :
  {x : ℝ | 2 * (Real.log x / Real.log 0.5)^2 + 9 * (Real.log x / Real.log 0.5) + 9 ≤ 0} = 
  {x : ℝ | 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8} :=
sorry

end solve_logarithmic_inequality_l77_77649


namespace range_of_k_l77_77599

theorem range_of_k (k : ℝ) : (x^2 + k * y^2 = 2) ∧ (k > 0) ∧ (k < 1) ↔ (0 < k ∧ k < 1) :=
by
  sorry

end range_of_k_l77_77599


namespace complex_real_number_l77_77823

-- Definition of the complex number z
def z (a : ℝ) : ℂ := (a^2 + 2011) + (a - 1) * Complex.I

-- The proof problem statement
theorem complex_real_number (a : ℝ) (h : z a = (a^2 + 2011 : ℂ)) : a = 1 :=
by
  sorry

end complex_real_number_l77_77823


namespace exterior_angle_of_regular_polygon_l77_77594

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l77_77594


namespace probability_less_than_one_third_l77_77573

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l77_77573


namespace find_S5_l77_77347

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a 1 + n * d
axiom sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_S5 (h : a 1 + a 3 + a 5 = 3) : S 5 = 5 :=
by
  sorry

end find_S5_l77_77347


namespace total_surfers_is_60_l77_77433

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end total_surfers_is_60_l77_77433


namespace M_subset_N_l77_77551

def M (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 2) + (Real.pi / 4)
def N (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 4) + (Real.pi / 2)

theorem M_subset_N : ∀ x, M x → N x := 
by
  sorry

end M_subset_N_l77_77551


namespace value_of_1_plus_i_cubed_l77_77676

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- Condition: i^2 = -1
lemma i_squared : i ^ 2 = -1 := by
  unfold i
  exact Complex.I_sq

-- The proof statement
theorem value_of_1_plus_i_cubed : 1 + i ^ 3 = 1 - i := by
  sorry

end value_of_1_plus_i_cubed_l77_77676


namespace interval_probability_l77_77560

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l77_77560


namespace integer_values_of_f_l77_77392

theorem integer_values_of_f (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a * b ≠ 1) : 
  ∃ k ∈ ({4, 7} : Finset ℕ), 
    (a^2 + b^2 + a * b) / (a * b - 1) = k := 
by
  sorry

end integer_values_of_f_l77_77392


namespace ab_equals_6_l77_77101

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77101


namespace probability_less_than_one_third_l77_77574

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l77_77574


namespace expected_coins_basilio_20_l77_77722

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end expected_coins_basilio_20_l77_77722


namespace sqrt_a_squared_b_l77_77142

variable {a b : ℝ}

theorem sqrt_a_squared_b (h: a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end sqrt_a_squared_b_l77_77142


namespace project_selection_l77_77764

noncomputable def binomial : ℕ → ℕ → ℕ 
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binomial n k + binomial n (k+1)

theorem project_selection :
  (binomial 5 2 * binomial 3 2) + (binomial 3 1 * binomial 5 1) = 45 := 
sorry

end project_selection_l77_77764


namespace payment_denotation_is_correct_l77_77996

-- Define the initial condition of receiving money
def received_amount : ℤ := 120

-- Define the payment amount
def payment_amount : ℤ := 85

-- The expected payoff
def expected_payment_denotation : ℤ := -85

-- Theorem stating that the payment should be denoted as -85 yuan
theorem payment_denotation_is_correct : (payment_amount = -expected_payment_denotation) :=
by
  sorry

end payment_denotation_is_correct_l77_77996


namespace problem_statement_l77_77926

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement :
  (∀ x y : ℝ, f x + f y = f (x + y)) →
  f 3 = 4 →
  f 0 + f (-3) = -4 :=
by
  intros h1 h2
  sorry

end problem_statement_l77_77926


namespace expected_coins_for_cat_basilio_l77_77724

open MeasureTheory

noncomputable def expected_coins_cat : ℝ := let n := 20 in 
  let p := 0.5 in 
  let E_XY := n * p in 
  let E_X_Y := 0.5 * 0 + 0.5 * 1 in 
  (E_XY + E_X_Y) / 2

theorem expected_coins_for_cat_basilio : expected_coins_cat = 5.25 :=
by
  sorry

end expected_coins_for_cat_basilio_l77_77724


namespace polar_to_cartesian_l77_77357

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.cos θ) :
  ∃ x y : ℝ, (x=ρ*Real.cos θ ∧ y=ρ*Real.sin θ) ∧ (x-1)^2 + y^2 = 1 :=
by
  sorry

end polar_to_cartesian_l77_77357


namespace remove_parentheses_l77_77165

variable (a b c : ℝ)

theorem remove_parentheses :
  -3 * a - (2 * b - c) = -3 * a - 2 * b + c :=
by
  sorry

end remove_parentheses_l77_77165


namespace area_of_inscribed_hexagon_l77_77444

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l77_77444


namespace product_ab_l77_77131

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77131


namespace largest_of_four_numbers_l77_77489

variables {x y z w : ℕ}

theorem largest_of_four_numbers
  (h1 : x + y + z = 180)
  (h2 : x + y + w = 197)
  (h3 : x + z + w = 208)
  (h4 : y + z + w = 222) :
  max x (max y (max z w)) = 89 :=
sorry

end largest_of_four_numbers_l77_77489


namespace box_volume_l77_77481

theorem box_volume (length width side_len : ℕ) (h1 : length = 48) (h2 : width = 36) (h3 : side_len = 8) :
  (length - 2 * side_len) * (width - 2 * side_len) * side_len = 5120 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l77_77481


namespace product_ab_l77_77136

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77136


namespace range_of_function_is_correct_l77_77343

def range_of_quadratic_function : Set ℝ :=
  {y | ∃ x : ℝ, y = -x^2 - 6 * x - 5}

theorem range_of_function_is_correct :
  range_of_quadratic_function = {y | y ≤ 4} :=
by
  -- sorry allows skipping the actual proof step
  sorry

end range_of_function_is_correct_l77_77343


namespace necessary_and_sufficient_condition_l77_77809

theorem necessary_and_sufficient_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1^2 - m * x1 - 1 = 0 ∧ x2^2 - m * x2 - 1 = 0) ↔ m > 1.5 :=
by
  sorry

end necessary_and_sufficient_condition_l77_77809


namespace remainder_of_2_pow_2017_mod_11_l77_77297

theorem remainder_of_2_pow_2017_mod_11 : (2 ^ 2017) % 11 = 7 := by
  sorry

end remainder_of_2_pow_2017_mod_11_l77_77297


namespace part1_l77_77542

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}
def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem part1 (a : ℝ) (h : a = 0) : A a ∩ B = {x | -1 < x ∧ x < 1} :=
by
  -- Proof here
  sorry

end part1_l77_77542


namespace find_f_4_l77_77418

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2

theorem find_f_4 : f 4 = 2 := 
by {
    -- The proof is omitted as per the task.
    sorry
}

end find_f_4_l77_77418


namespace ratio_proof_l77_77472

theorem ratio_proof (X: ℕ) (h: 150 * 2 = 300 * X) : X = 1 := by
  sorry

end ratio_proof_l77_77472


namespace final_answer_l77_77482

theorem final_answer : (848 / 8) - 100 = 6 := 
by
  sorry

end final_answer_l77_77482


namespace find_m_n_sum_l77_77361

theorem find_m_n_sum (m n : ℝ) :
  ( ∀ x, -3 < x ∧ x < 6 → x^2 - m * x - 6 * n < 0 ) →
  m + n = 6 :=
by
  sorry

end find_m_n_sum_l77_77361


namespace series_sum_eq_one_sixth_l77_77003

noncomputable def a (n : ℕ) : ℝ := 2^n / (7^(2^n) + 1)

theorem series_sum_eq_one_sixth :
  (∑' (n : ℕ), a n) = 1 / 6 :=
sorry

end series_sum_eq_one_sixth_l77_77003


namespace grazing_months_for_b_l77_77665

/-
  We define the problem conditions and prove that b put his oxen for grazing for 5 months.
-/

theorem grazing_months_for_b (x : ℕ) :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let c_oxen := 15
  let c_months := 3
  let total_rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * x
  let c_ox_months := c_oxen * c_months
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  (c_share : ℚ) / total_rent = (c_ox_months : ℚ) / total_ox_months →
  x = 5 :=
by
  sorry

end grazing_months_for_b_l77_77665


namespace min_square_sum_l77_77646

theorem min_square_sum (a b m n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 15 * a + 16 * b = m * m) (h4 : 16 * a - 15 * b = n * n) : 481 ≤ min (m * m) (n * n) :=
sorry

end min_square_sum_l77_77646


namespace mary_picked_nine_lemons_l77_77796

def num_lemons_sally := 7
def total_num_lemons := 16
def num_lemons_mary := total_num_lemons - num_lemons_sally

theorem mary_picked_nine_lemons :
  num_lemons_mary = 9 := by
  sorry

end mary_picked_nine_lemons_l77_77796


namespace blankets_warmth_increase_l77_77299

-- Conditions
def blankets_in_closet : ℕ := 14
def blankets_used : ℕ := blankets_in_closet / 2
def degree_per_blanket : ℕ := 3

-- Goal: Prove that the total temperature increase is 21 degrees.
theorem blankets_warmth_increase : blankets_used * degree_per_blanket = 21 :=
by
  sorry

end blankets_warmth_increase_l77_77299


namespace original_number_of_men_l77_77826

variable (M W : ℕ)

def original_work_condition := M * W / 60 = W
def larger_group_condition := (M + 8) * W / 50 = W

theorem original_number_of_men : original_work_condition M W ∧ larger_group_condition M W → M = 48 :=
by
  sorry

end original_number_of_men_l77_77826


namespace ab_equals_six_l77_77126

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77126


namespace probability_of_interval_l77_77578

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l77_77578


namespace divide_square_into_smaller_squares_l77_77412

-- Definition of the property P(n)
def P (n : ℕ) : Prop := ∃ (f : ℕ → ℕ), ∀ i, i < n → (f i > 0)

-- Proposition for the problem
theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
sorry

end divide_square_into_smaller_squares_l77_77412


namespace age_difference_l77_77304

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end age_difference_l77_77304


namespace sum_of_all_possible_values_of_abs_b_l77_77409

theorem sum_of_all_possible_values_of_abs_b {a b : ℝ}
  {r s : ℝ} (hr : r^3 + a * r + b = 0) (hs : s^3 + a * s + b = 0)
  (hr4 : (r + 4)^3 + a * (r + 4) + b + 240 = 0) (hs3 : (s - 3)^3 + a * (s - 3) + b + 240 = 0) :
  |b| = 20 ∨ |b| = 42 →
  20 + 42 = 62 :=
by
  sorry

end sum_of_all_possible_values_of_abs_b_l77_77409


namespace greatest_three_digit_multiple_of_17_l77_77283

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77283


namespace find_z_l77_77354

def z_value (i : ℂ) (z : ℂ) : Prop := z * (1 - (2 * i)) = 2 + (4 * i)

theorem find_z (i z : ℂ) (hi : i^2 = -1) (h : z_value i z) : z = - (2 / 5) + (8 / 5) * i := by
  sorry

end find_z_l77_77354


namespace total_employees_l77_77325

theorem total_employees (female_employees managers male_associates female_managers : ℕ)
  (h_female_employees : female_employees = 90)
  (h_managers : managers = 40)
  (h_male_associates : male_associates = 160)
  (h_female_managers : female_managers = 40) :
  female_employees - female_managers + male_associates + managers = 250 :=
by {
  sorry
}

end total_employees_l77_77325


namespace range_of_m_l77_77034

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 9 * x - m

theorem range_of_m (H : ∃ (x_0 : ℝ), x_0 ≠ 0 ∧ f 0 x_0 = f 0 x_0) : 0 < m ∧ m < 1 / 2 :=
sorry

end range_of_m_l77_77034


namespace ab_equals_six_l77_77054

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77054


namespace two_digit_integers_l77_77625

theorem two_digit_integers (x y m : ℕ) (h1 : 10 ≤ x ∧ x < 100)
                           (h2 : 10 ≤ y ∧ y < 100)
                           (h3 : y = (x % 10) * 10 + x / 10)
                           (h4 : x^2 - y^2 = 9 * m^2) :
  x + y + 2 * m = 143 :=
sorry

end two_digit_integers_l77_77625


namespace find_a_l77_77364

open Real

def are_perpendicular (l1 l2 : Real × Real × Real) : Prop :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  a1 * a2 + b1 * b2 = 0

theorem find_a (a : Real) :
  let l1 := (a + 2, 1 - a, -1)
  let l2 := (a - 1, 2 * a + 3, 2)
  are_perpendicular l1 l2 → a = 1 ∨ a = -1 :=
by
  intro h
  sorry

end find_a_l77_77364


namespace product_ab_l77_77138

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77138


namespace charts_per_associate_professor_l77_77982

-- Definitions
def A : ℕ := 3
def B : ℕ := 4
def C : ℕ := 1

-- Conditions based on the given problem
axiom h1 : 2 * A + B = 10
axiom h2 : A * C + 2 * B = 11
axiom h3 : A + B = 7

-- The theorem to be proven
theorem charts_per_associate_professor : C = 1 := by
  sorry

end charts_per_associate_professor_l77_77982


namespace seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l77_77459

-- Define the sequences
def a_sq (n : ℕ) : ℕ := n ^ 2
def a_cube (n : ℕ) : ℕ := n ^ 3

-- First proof problem statement
theorem seq_satisfies_recurrence_sq :
  (a_sq 0 = 0) ∧ (a_sq 1 = 1) ∧ (a_sq 2 = 4) ∧ (a_sq 3 = 9) ∧ (a_sq 4 = 16) →
  (∀ n : ℕ, n ≥ 3 → a_sq n = 3 * a_sq (n - 1) - 3 * a_sq (n - 2) + a_sq (n - 3)) :=
by
  sorry

-- Second proof problem statement
theorem seq_satisfies_recurrence_cube :
  (a_cube 0 = 0) ∧ (a_cube 1 = 1) ∧ (a_cube 2 = 8) ∧ (a_cube 3 = 27) ∧ (a_cube 4 = 64) →
  (∀ n : ℕ, n ≥ 4 → a_cube n = 4 * a_cube (n - 1) - 6 * a_cube (n - 2) + 4 * a_cube (n - 3) - a_cube (n - 4)) :=
by
  sorry

end seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l77_77459


namespace mod_2_200_sub_3_l77_77830

theorem mod_2_200_sub_3 (h1 : 2^1 % 7 = 2) (h2 : 2^2 % 7 = 4) (h3 : 2^3 % 7 = 1) : (2^200 - 3) % 7 = 1 := 
by
  sorry

end mod_2_200_sub_3_l77_77830


namespace final_temperature_l77_77386

variable (initial_temp : ℝ := 40)
variable (double_temp : ℝ := initial_temp * 2)
variable (reduce_by_dad : ℝ := double_temp - 30)
variable (reduce_by_mother : ℝ := reduce_by_dad * 0.70)
variable (increase_by_sister : ℝ := reduce_by_mother + 24)

theorem final_temperature : increase_by_sister = 59 := by
  sorry

end final_temperature_l77_77386


namespace angles_equal_l77_77779

variables {A B C M W L T : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace M] [MetricSpace W] [MetricSpace L] [MetricSpace T]

-- A, B, C are points of the triangle ABC with incircle k.
-- Line_segment AC is longer than line segment BC.
-- M is the intersection of median from C.
-- W is the intersection of angle bisector from C.
-- L is the intersection of altitude from C.
-- T is the point where the tangent from M to the incircle k, different from AB, touches k.
def triangle_ABC (A B C : Type*) : Prop := sorry
def incircle_k (A B C : Type*) (k : Type*) : Prop := sorry
def longer_AC (A B C : Type*) : Prop := sorry
def intersection_median_C (M C : Type*) : Prop := sorry
def intersection_angle_bisector_C (W C : Type*) : Prop := sorry
def intersection_altitude_C (L C : Type*) : Prop := sorry
def tangent_through_M (M T k : Type*) : Prop := sorry
def touches_k (T k : Type*) : Prop := sorry
def angle_eq (M T W L : Type*) : Prop := sorry

theorem angles_equal (A B C M W L T k : Type*)
  (h_triangle : triangle_ABC A B C)
  (h_incircle : incircle_k A B C k)
  (h_longer_AC : longer_AC A B C)
  (h_inter_median : intersection_median_C M C)
  (h_inter_bisector : intersection_angle_bisector_C W C)
  (h_inter_altitude : intersection_altitude_C L C)
  (h_tangent : tangent_through_M M T k)
  (h_touches : touches_k T k) :
  angle_eq M T W L := 
sorry


end angles_equal_l77_77779


namespace polynomial_sum_of_coefficients_l77_77156

theorem polynomial_sum_of_coefficients {v : ℕ → ℝ} (h1 : v 1 = 7)
  (h2 : ∀ n : ℕ, v (n + 1) - v n = 5 * n - 2) :
  ∃ (a b c : ℝ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) :=
by
  sorry

end polynomial_sum_of_coefficients_l77_77156


namespace initial_men_employed_l77_77490

theorem initial_men_employed (M : ℕ) 
  (h1 : ∀ m d, m * d = 2 * 10)
  (h2 : ∀ m t, (m + 30) * t = 10 * 30) : 
  M = 75 :=
by
  sorry

end initial_men_employed_l77_77490


namespace largest_three_digit_multiple_of_17_l77_77219

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77219


namespace find_x_for_g_statement_l77_77365

noncomputable def g (x : ℝ) : ℝ := (x + 4) ^ (1/3) / 5 ^ (1/3)

theorem find_x_for_g_statement (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -13 / 3 := by
  sorry

end find_x_for_g_statement_l77_77365


namespace triangle_area_ratio_l77_77607

theorem triangle_area_ratio {A B C : ℝ} {a b c : ℝ} 
  (h : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) 
  (S1 : ℝ) (S2 : ℝ) :
  S1 / S2 = 1 / (3 * Real.pi) :=
sorry

end triangle_area_ratio_l77_77607


namespace smallest_four_digit_multiple_of_18_l77_77515

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l77_77515


namespace greatest_three_digit_multiple_of_17_l77_77272

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77272


namespace ab_equals_six_l77_77112

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77112


namespace roots_of_quadratic_l77_77648

theorem roots_of_quadratic (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
by
  sorry

end roots_of_quadratic_l77_77648


namespace greatest_three_digit_multiple_of_17_l77_77277

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77277


namespace ab_value_l77_77094

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77094


namespace triangle_area_circumcircle_area_ratio_l77_77608

theorem triangle_area_circumcircle_area_ratio {A B C a b c : ℝ} (h1 : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) :
  let S₁ := (1 / 2) * a * b * Real.sin C in
  let S₂ := Real.pi * (b / (2 * Real.sin B)) ^ 2 in
  S₁ / S₂ = 1 / (3 * Real.pi) :=
by
  sorry

end triangle_area_circumcircle_area_ratio_l77_77608


namespace num_interesting_quadruples_l77_77498

theorem num_interesting_quadruples : 
  ∑ (a : ℕ) in Finset.Icc 1 15, 
  ∑ (b : ℕ) in Finset.Icc (a + 1) 15, 
  ∑ (c : ℕ) in Finset.Icc (b + 1) 15, 
  ∑ (d : ℕ) in Finset.Icc (c + 1) 15, 
  if a + d > 2 * (b + c) then 1 else 0 = 500 :=
by
  sorry

end num_interesting_quadruples_l77_77498


namespace product_of_ab_l77_77069

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77069


namespace compute_binom_value_l77_77753

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else x * binom (x - 1) (k - 1) / k

theorem compute_binom_value : 
  (binom (1/2) 2014 * 4^2014 / binom 4028 2014) = -1/4027 :=
by 
  sorry

end compute_binom_value_l77_77753


namespace product_of_ab_l77_77072

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77072


namespace greatest_3_digit_multiple_of_17_l77_77292

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77292


namespace constants_sum_l77_77414

theorem constants_sum (A B C D : ℕ) 
  (h : ∀ n : ℕ, n ≥ 4 → n^4 = A * (n.choose 4) + B * (n.choose 3) + C * (n.choose 2) + D * (n.choose 1)) 
  : A + B + C + D = 75 :=
by
  sorry

end constants_sum_l77_77414


namespace long_fur_brown_dogs_l77_77147

-- Defining the basic parameters given in the problem
def total_dogs : ℕ := 45
def long_fur : ℕ := 26
def brown_dogs : ℕ := 30
def neither_long_fur_nor_brown : ℕ := 8

-- Statement of the theorem
theorem long_fur_brown_dogs : ∃ LB : ℕ, LB = 27 ∧ total_dogs = long_fur + brown_dogs - LB + neither_long_fur_nor_brown :=
by {
  -- skipping the proof
  sorry
}

end long_fur_brown_dogs_l77_77147


namespace find_x_l77_77468

theorem find_x (h : 0.60 / x = 6 / 2) : x = 0.20 :=
by
  sorry

end find_x_l77_77468


namespace matching_pairs_less_than_21_in_at_least_61_positions_l77_77819

theorem matching_pairs_less_than_21_in_at_least_61_positions :
  ∀ (disks : ℕ) (total_sectors : ℕ) (red_sectors : ℕ) (max_overlap : ℕ) (rotations : ℕ),
  disks = 2 →
  total_sectors = 1965 →
  red_sectors = 200 →
  max_overlap = 20 →
  rotations = total_sectors →
  (∃ positions, positions = total_sectors - (red_sectors * red_sectors / (max_overlap + 1)) ∧ positions ≤ rotations) →
  positions = 61 :=
by {
  -- Placeholder to provide the structure of the theorem.
  sorry
}

end matching_pairs_less_than_21_in_at_least_61_positions_l77_77819


namespace product_of_three_numbers_l77_77952

theorem product_of_three_numbers (a b c : ℚ) 
  (h₁ : a + b + c = 30)
  (h₂ : a = 6 * (b + c))
  (h₃ : b = 5 * c) : 
  a * b * c = 22500 / 343 := 
sorry

end product_of_three_numbers_l77_77952


namespace expected_coins_basilio_per_day_l77_77718

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end expected_coins_basilio_per_day_l77_77718


namespace find_k_value_l77_77539

variable {a : ℕ → ℕ} {S : ℕ → ℕ} 

axiom sum_of_first_n_terms (n : ℕ) (hn : n > 0) : S n = a n / n
axiom exists_Sk_inequality (k : ℕ) (hk : k > 0) : 1 < S k ∧ S k < 9

theorem find_k_value 
  (k : ℕ) (hk : k > 0) (hS : S k = a k / k) (hSk : 1 < S k ∧ S k < 9)
  (h_cond : ∀ n > 0, S n = n * S n ∧ S (n - 1) = S n * (n - 1)) : 
  k = 4 :=
sorry

end find_k_value_l77_77539


namespace sam_distance_l77_77399

theorem sam_distance (miles_marguerite : ℕ) (hours_marguerite : ℕ) (hours_sam : ℕ) 
  (speed_increase : ℚ) (avg_speed_marguerite : ℚ) (speed_sam : ℚ) (distance_sam : ℚ) :
  miles_marguerite = 120 ∧ hours_marguerite = 3 ∧ hours_sam = 4 ∧ speed_increase = 1.20 ∧
  avg_speed_marguerite = miles_marguerite / hours_marguerite ∧ 
  speed_sam = avg_speed_marguerite * speed_increase ∧
  distance_sam = speed_sam * hours_sam →
  distance_sam = 192 :=
by
  intros h
  sorry

end sam_distance_l77_77399


namespace pens_sold_during_promotion_l77_77320

theorem pens_sold_during_promotion (x y n : ℕ) 
  (h_profit: 12 * x + 7 * y = 2011)
  (h_n: n = 2 * x + y) : 
  n = 335 := by
  sorry

end pens_sold_during_promotion_l77_77320


namespace number_of_sides_l77_77592

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l77_77592


namespace probability_intervals_l77_77581

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l77_77581


namespace rhombus_diagonal_l77_77168

theorem rhombus_diagonal (d1 d2 area : ℝ) (h1 : d1 = 20) (h2 : area = 160) (h3 : area = (d1 * d2) / 2) :
  d2 = 16 :=
by
  rw [h1, h2] at h3
  linarith

end rhombus_diagonal_l77_77168


namespace percentage_increase_of_cube_surface_area_l77_77660

-- Basic setup definitions and conditions
variable (a : ℝ)

-- Step 1: Initial surface area
def initial_surface_area : ℝ := 6 * a^2

-- Step 2: New edge length after 50% growth
def new_edge_length : ℝ := 1.5 * a

-- Step 3: New surface area after edge growth
def new_surface_area : ℝ := 6 * (new_edge_length a)^2

-- Step 4: Surface area after scaling by 1.5
def scaled_surface_area : ℝ := new_surface_area a * (1.5)^2

-- Prove the percentage increase
theorem percentage_increase_of_cube_surface_area :
  (scaled_surface_area a - initial_surface_area a) / initial_surface_area a * 100 = 406.25 := by
  sorry

end percentage_increase_of_cube_surface_area_l77_77660


namespace range_of_x_minus_y_l77_77344

variable (x y : ℝ)
variable (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3)

theorem range_of_x_minus_y : -1 < x - y ∧ x - y < 5 := 
by {
  sorry
}

end range_of_x_minus_y_l77_77344


namespace sum_of_seven_digits_is_33_l77_77939

/-
  Seven different digits from the set {1, 2, 3, 4, 5, 6, 7, 8, 9}
  are placed in the squares of a grid where a vertical column of four squares
  and a horizontal row of five squares intersect at two squares. 
  The sum of the entries in the vertical column is 30 and 
  the sum of the entries in the horizontal row is 25. 
  Prove that the sum of the seven distinct digits used is 33.
-/

theorem sum_of_seven_digits_is_33 :
  ∃ (digits : Finset ℕ),
  digits.card = 7 ∧ digits ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (∃ (a b c d e h i : ℕ), 
    digits = {a, b, c, d, e, h, i} ∧ 
    {a, b, c, d}.sum = 30 ∧ 
    {e, b, c, h, i}.sum = 25) 
  → digits.sum = 33 :=
by
  sorry

end sum_of_seven_digits_is_33_l77_77939


namespace ab_value_l77_77093

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77093


namespace total_elephants_in_two_parks_is_280_l77_77177

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end total_elephants_in_two_parks_is_280_l77_77177


namespace sylvia_buttons_l77_77415

theorem sylvia_buttons (n : ℕ) (h₁: n % 10 = 0) (h₂: n ≥ 80):
  (∃ w : ℕ, w = (n - (n / 2) - (n / 5) - 8)) ∧ (n - (n / 2) - (n / 5) - 8 = 1) :=
by
  sorry

end sylvia_buttons_l77_77415


namespace cody_initial_money_l77_77852

variable (x : ℤ)

theorem cody_initial_money :
  (x + 9 - 19 = 35) → (x = 45) :=
by
  intro h
  sorry

end cody_initial_money_l77_77852


namespace problem_statement_l77_77994

theorem problem_statement : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17 / 15 :=
by
  -- Prove the equivalence as outlined above.
  sorry

end problem_statement_l77_77994


namespace train_length_equals_sixty_two_point_five_l77_77970

-- Defining the conditions
noncomputable def calculate_train_length (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_faster_train - speed_slower_train
  let relative_speed_ms := (relative_speed_kmh * 5) / 18
  let distance_covered := relative_speed_ms * time_seconds
  distance_covered / 2

theorem train_length_equals_sixty_two_point_five :
  calculate_train_length 46 36 45 = 62.5 :=
sorry

end train_length_equals_sixty_two_point_five_l77_77970


namespace sequence_nth_term_mod_2500_l77_77496

def sequence_nth_term (n : ℕ) : ℕ :=
  -- this is a placeholder function definition; the actual implementation to locate the nth term is skipped
  sorry

theorem sequence_nth_term_mod_2500 : (sequence_nth_term 2500) % 7 = 1 := 
sorry

end sequence_nth_term_mod_2500_l77_77496


namespace necessary_but_not_sufficient_condition_l77_77461

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m < 1) → (∀ x y : ℝ, (x - m) ^ 2 + y ^ 2 = m ^ 2 → (x, y) ≠ (1, 1)) :=
sorry

end necessary_but_not_sufficient_condition_l77_77461


namespace ab_equals_six_l77_77127

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77127


namespace arithmetic_sequence_property_l77_77029

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_cond : a 1 + a 3 = 2) : 
  a 2 = 1 :=
by 
  sorry

end arithmetic_sequence_property_l77_77029


namespace product_of_ab_l77_77067

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77067


namespace hexagon_area_of_circle_l77_77448

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l77_77448


namespace fraction_zero_value_x_l77_77600

theorem fraction_zero_value_x (x : ℝ) (h1 : (x - 2) / (1 - x) = 0) (h2 : 1 - x ≠ 0) : x = 2 := 
sorry

end fraction_zero_value_x_l77_77600


namespace deductive_reasoning_example_l77_77470

-- Definitions for the conditions
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
def Iron : Type := sorry

-- The problem statement
theorem deductive_reasoning_example (H1 : ∀ x, Metal x → ConductsElectricity x) (H2 : Metal Iron) : ConductsElectricity Iron :=
by sorry

end deductive_reasoning_example_l77_77470


namespace product_of_ab_l77_77065

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77065


namespace smallest_four_digit_multiple_of_18_l77_77528

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l77_77528


namespace isosceles_triangle_of_cosine_equality_l77_77913

variable {A B C : ℝ}
variable {a b c : ℝ}

/-- Prove that in triangle ABC, if a*cos(B) = b*cos(A), then a = b implying A = B --/
theorem isosceles_triangle_of_cosine_equality 
(h1 : a * Real.cos B = b * Real.cos A) :
a = b :=
sorry

end isosceles_triangle_of_cosine_equality_l77_77913


namespace pizza_slices_all_toppings_l77_77483

theorem pizza_slices_all_toppings (x : ℕ) :
  (16 = (8 - x) + (12 - x) + (6 - x) + x) → x = 5 := by
  sorry

end pizza_slices_all_toppings_l77_77483


namespace triangle_cut_l77_77023

theorem triangle_cut (A B C : Type) [PointClass ℝ A] [PointClass ℝ B] [PointClass ℝ C]
  (angle_A : ∠BAC = 30) (angle_B : ∠ABC = 70) (angle_C : ∠ACB = 80)
  (M : PointClass ℝ (Midpoint A C)) (H : PointClass ℝ (Altitude A B C)) (L : PointClass ℝ (Bisector A H B)) :
  AreParallel (median A H C) (bisector A H B) :=
  begin
    sorry
  end

end triangle_cut_l77_77023


namespace ab_eq_six_l77_77078

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77078


namespace greatest_three_digit_multiple_of_17_l77_77231

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77231


namespace sqrt_calculation_l77_77811

theorem sqrt_calculation : Real.sqrt (36 * Real.sqrt 16) = 12 := 
by
  sorry

end sqrt_calculation_l77_77811


namespace smallest_perimeter_scalene_triangle_l77_77659

theorem smallest_perimeter_scalene_triangle (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) :
  a + b + c = 9 := 
sorry

end smallest_perimeter_scalene_triangle_l77_77659


namespace area_of_inscribed_hexagon_l77_77443

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l77_77443


namespace intersection_l77_77152

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / (2 * x - 6)

noncomputable def g (x : ℝ) (a b c d k : ℝ) : ℝ := -2 * x - 4 + k / (x - d)

theorem intersection (a b c k : ℝ) (h_d : d = 3) (h_k : k = 36) : 
  ∃ (x y : ℝ), x ≠ -3 ∧ (f x = g x 0 0 0 d k) ∧ (x, y) = (6.8, -32 / 19) :=
by
  sorry

end intersection_l77_77152


namespace largest_three_digit_multiple_of_17_l77_77223

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77223


namespace range_of_a_l77_77933

noncomputable def my_function (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (x + 1) ^ 2

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ → my_function a x₁ - my_function a x₂ ≥ 4 * (x₁ - x₂)) → a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l77_77933


namespace ab_value_l77_77088

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77088


namespace arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l77_77886

-- Statement for Problem 1: Number of ways to draw five numbers forming an arithmetic progression
theorem arithmetic_progression_five_numbers :
  ∃ (N : ℕ), N = 968 :=
  sorry

-- Statement for Problem 2: Number of ways to draw four numbers forming an arithmetic progression with a fifth number being arbitrary
theorem arithmetic_progression_four_numbers :
  ∃ (N : ℕ), N = 111262 :=
  sorry

end arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l77_77886


namespace fuel_consumption_per_100_km_l77_77183

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end fuel_consumption_per_100_km_l77_77183


namespace greatest_three_digit_multiple_of_17_l77_77242

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77242


namespace min_value_f_l77_77864

theorem min_value_f (x : ℝ) (h : 0 < x) : 
  ∃ c: ℝ, c = 2.5 ∧ (∀ x, 0 < x → x^2 + 1 / x^2 + 1 / (x^2 + 1 / x^2) ≥ c) :=
by sorry

end min_value_f_l77_77864


namespace greatest_three_digit_multiple_of_17_l77_77274

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77274


namespace seating_arrangement_l77_77187

theorem seating_arrangement :
  let seats := ["A", "B", "C", "D", "E", "F"]
  let families := {1, 2}
  let num_people := 6
  -- 3 adults and 3 children, no two same-family members sit together 
  (2 * (Finset.univ : Finset (Fin 3)).card.factorial * (Finset.univ : Finset (Fin 3)).card.factorial) = 72 :=
by
  sorry

end seating_arrangement_l77_77187


namespace relationship_coefficients_l77_77943

-- Definitions based directly on the conditions
def has_extrema (a b c : ℝ) : Prop := b^2 - 3 * a * c > 0
def passes_through_origin (x1 x2 y1 y2 : ℝ) : Prop := x1 * y2 = x2 * y1

-- Main statement proving the relationship among the coefficients
theorem relationship_coefficients (a b c d : ℝ) (x1 x2 y1 y2 : ℝ)
  (h_extrema : has_extrema a b c)
  (h_line : passes_through_origin x1 x2 y1 y2)
  (hx1 : x1 ≠ 0) (hx2 : x2 ≠ 0)
  (h_y1 : y1 = a * x1^3 + b * x1^2 + c * x1 + d)
  (h_y2 : y2 = a * x2^3 + b * x2^2 + c * x2 + d) :
  9 * a * d = b * c :=
sorry

end relationship_coefficients_l77_77943


namespace product_of_ab_l77_77075

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77075


namespace tables_difference_l77_77841

theorem tables_difference (N O : ℕ) (h1 : N + O = 40) (h2 : 6 * N + 4 * O = 212) : N - O = 12 :=
sorry

end tables_difference_l77_77841


namespace oranges_left_to_be_sold_l77_77932

theorem oranges_left_to_be_sold : 
  let total_oranges := 7 * 12,
      reserved_for_friend := total_oranges / 4,
      remaining_after_reservation := total_oranges - reserved_for_friend,
      sold_yesterday := remaining_after_reservation * 3 / 7,
      left_after_sale := remaining_after_reservation - sold_yesterday,
      rotten_today := 4,
      left_today := left_after_sale - rotten_today in
  left_today = 32 :=
by
  sorry

end oranges_left_to_be_sold_l77_77932


namespace tangent_line_circle_l77_77540

theorem tangent_line_circle (k : ℝ) (h1 : k = Real.sqrt 3) (h2 : ∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) :
  (k = Real.sqrt 3 → (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1)) ∧ (¬ (∀ (k : ℝ), (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) → k = Real.sqrt 3)) :=
  sorry

end tangent_line_circle_l77_77540


namespace min_cube_edge_division_l77_77778

theorem min_cube_edge_division (n : ℕ) (h : n^3 ≥ 1996) : n = 13 :=
by {
  sorry
}

end min_cube_edge_division_l77_77778


namespace min_value_of_a_plus_2b_l77_77353

theorem min_value_of_a_plus_2b (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_eq : 1 / a + 2 / b = 4) : a + 2 * b = 9 / 4 :=
by
  sorry

end min_value_of_a_plus_2b_l77_77353


namespace valid_license_plates_l77_77606

def letters := 26
def digits := 10
def totalPlates := letters^3 * digits^4

theorem valid_license_plates : totalPlates = 175760000 := by
  sorry

end valid_license_plates_l77_77606


namespace range_of_m_l77_77394

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : (1 / (a - b)) + (1 / (b - c)) ≥ m / (a - c)) :
  m ≤ 4 :=
sorry

end range_of_m_l77_77394


namespace triangle_is_either_isosceles_or_right_angled_l77_77374

theorem triangle_is_either_isosceles_or_right_angled
  (A B : Real)
  (a b c : Real)
  (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  : a = b ∨ a^2 + b^2 = c^2 :=
sorry

end triangle_is_either_isosceles_or_right_angled_l77_77374


namespace greatest_three_digit_multiple_of_17_l77_77207

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77207


namespace complement_of_A_in_U_l77_77363

-- Conditions definitions
def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {x | 2 * x - 5 < 0}

-- Theorem stating the question and the correct answer
theorem complement_of_A_in_U :
  U \ A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  -- The proof will go here
  sorry

end complement_of_A_in_U_l77_77363


namespace total_points_other_team_members_l77_77769

variable (x y : ℕ)

theorem total_points_other_team_members :
  (1 / 3 * x + 3 / 8 * x + 18 + y = x) ∧ (y ≤ 24) → y = 17 :=
by
  intro h
  have h1 : 1 / 3 * x + 3 / 8 * x + 18 + y = x := h.1
  have h2 : y ≤ 24 := h.2
  sorry

end total_points_other_team_members_l77_77769


namespace ab_equals_six_l77_77121

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77121


namespace quadratic_to_standard_form_l77_77647

theorem quadratic_to_standard_form (a b c : ℝ) (x : ℝ) :
  (20 * x^2 + 240 * x + 3200 = a * (x + b)^2 + c) → (a + b + c = 2506) :=
  sorry

end quadratic_to_standard_form_l77_77647


namespace repeating_decimal_sum_l77_77505

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9
noncomputable def repeating_decimal_7 : ℚ := 7 / 9

theorem repeating_decimal_sum : 
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 - repeating_decimal_7 = -1 / 3 :=
by {
  sorry
}

end repeating_decimal_sum_l77_77505


namespace range_of_m_l77_77889

theorem range_of_m : 
  ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := 
by
  -- the proof will go here
  sorry

end range_of_m_l77_77889


namespace geometric_probability_l77_77584

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l77_77584


namespace partial_fraction_identity_l77_77339

theorem partial_fraction_identity
  (P Q R : ℝ)
  (h1 : -2 = P + Q)
  (h2 : 1 = Q + R)
  (h3 : -1 = P + R) :
  (P, Q, R) = (-2, 0, 1) :=
by
  sorry

end partial_fraction_identity_l77_77339


namespace children_too_heavy_l77_77486

def Kelly_weight : ℝ := 34
def Sam_weight : ℝ := 40
def Daisy_weight : ℝ := 28
def Megan_weight := 1.1 * Kelly_weight
def Mike_weight := Megan_weight + 5

def Total_weight := Kelly_weight + Sam_weight + Daisy_weight + Megan_weight + Mike_weight
def Bridge_limit : ℝ := 130

theorem children_too_heavy :
  Total_weight - Bridge_limit = 51.8 :=
by
  sorry

end children_too_heavy_l77_77486


namespace scientific_notation_141260_million_l77_77161

theorem scientific_notation_141260_million :
  (141260 * 10^6 : ℝ) = 1.4126 * 10^11 := 
sorry

end scientific_notation_141260_million_l77_77161


namespace greatest_three_digit_multiple_of_17_l77_77263

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77263


namespace g_f_3_eq_1476_l77_77624

def f (x : ℝ) : ℝ := x^3 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_f_3_eq_1476 : g (f 3) = 1476 :=
by
  sorry

end g_f_3_eq_1476_l77_77624


namespace wheat_acres_l77_77803

def cultivate_crops (x y : ℕ) : Prop :=
  (42 * x + 30 * y = 18600) ∧ (x + y = 500) 

theorem wheat_acres : ∃ y, ∃ x, 
  cultivate_crops x y ∧ y = 200 :=
by {sorry}

end wheat_acres_l77_77803


namespace probability_less_than_one_third_l77_77577

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l77_77577


namespace ab_value_l77_77053

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77053


namespace fraction_classification_l77_77460

theorem fraction_classification (x y : ℤ) :
  (∃ a b : ℤ, a/b = x/(x+1)) ∧ ¬(∃ a b : ℤ, a/b = x/2 + 1) ∧ ¬(∃ a b : ℤ, a/b = x/2) ∧ ¬(∃ a b : ℤ, a/b = xy/3) :=
by sorry

end fraction_classification_l77_77460


namespace f_greater_than_fp_3_2_l77_77747

noncomputable def f (x : ℝ) (a : ℝ) := a * (x - Real.log x) + (2 * x - 1) / (x ^ 2)
noncomputable def f' (x : ℝ) (a : ℝ) := (a * x^3 - a * x^2 + 2 - 2*x) / x^3

theorem f_greater_than_fp_3_2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  f x 1 > f' x 1 + 3 / 2 := sorry

end f_greater_than_fp_3_2_l77_77747


namespace proof_statement_l77_77330

def convert_base_9_to_10 (n : Nat) : Nat :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def convert_base_6_to_10 (n : Nat) : Nat :=
  2 * 6^2 + 2 * 6^1 + 1 * 6^0

def problem_statement : Prop :=
  convert_base_9_to_10 324 - convert_base_6_to_10 221 = 180

theorem proof_statement : problem_statement := 
  by
    sorry

end proof_statement_l77_77330


namespace Megan_seashells_needed_l77_77930

-- Let x be the number of additional seashells needed
def seashells_needed (total_seashells desired_seashells : Nat) : Nat :=
  desired_seashells - total_seashells

-- Given conditions
def current_seashells : Nat := 19
def desired_seashells : Nat := 25

-- The equivalent proof problem
theorem Megan_seashells_needed : seashells_needed current_seashells desired_seashells = 6 := by
  sorry

end Megan_seashells_needed_l77_77930


namespace smaller_cuboid_width_l77_77836

theorem smaller_cuboid_width
  (length_orig width_orig height_orig : ℕ)
  (length_small height_small : ℕ)
  (num_small_cuboids : ℕ)
  (volume_orig : ℕ := length_orig * width_orig * height_orig)
  (volume_small : ℕ := length_small * width_small * height_small)
  (H1 : length_orig = 18)
  (H2 : width_orig = 15)
  (H3 : height_orig = 2)
  (H4 : length_small = 5)
  (H5 : height_small = 3)
  (H6 : num_small_cuboids = 6)
  (H_volume_match : num_small_cuboids * volume_small = volume_orig)
  : width_small = 6 := by
  sorry

end smaller_cuboid_width_l77_77836


namespace ab_value_l77_77087

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77087


namespace ab_value_l77_77095

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77095


namespace steak_amount_per_member_l77_77436

theorem steak_amount_per_member : 
  ∀ (num_members steaks_needed ounces_per_steak total_ounces each_amount : ℕ),
    num_members = 5 →
    steaks_needed = 4 →
    ounces_per_steak = 20 →
    total_ounces = steaks_needed * ounces_per_steak →
    each_amount = total_ounces / num_members →
    each_amount = 16 :=
by
  intros num_members steaks_needed ounces_per_steak total_ounces each_amount
  intro h_members h_steaks h_ounces_per_steak h_total_ounces h_each_amount
  sorry

end steak_amount_per_member_l77_77436


namespace phase_shift_cosine_l77_77509

theorem phase_shift_cosine (x : ℝ) : 2 * x + (Real.pi / 2) = 0 → x = - (Real.pi / 4) :=
by
  intro h
  sorry

end phase_shift_cosine_l77_77509


namespace production_days_l77_77015

variable (n : ℕ) (average_past : ℝ := 50) (production_today : ℝ := 115) (new_average : ℝ := 55)

theorem production_days (h1 : average_past * n + production_today = new_average * (n + 1)) : 
    n = 12 := 
by 
  sorry

end production_days_l77_77015


namespace greatest_three_digit_multiple_of_17_l77_77229

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77229


namespace highland_baseball_club_members_l77_77401

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

end highland_baseball_club_members_l77_77401


namespace difference_of_squares_l77_77667

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 :=
by
  sorry

end difference_of_squares_l77_77667


namespace stadium_height_l77_77734

theorem stadium_height
  (l w d : ℕ) (h : ℕ) 
  (hl : l = 24) 
  (hw : w = 18) 
  (hd : d = 34) 
  (h_eq : d^2 = l^2 + w^2 + h^2) : 
  h = 16 := by 
  sorry

end stadium_height_l77_77734


namespace no_real_solution_for_inequality_l77_77500

theorem no_real_solution_for_inequality :
  ¬ ∃ a : ℝ, ∃ x : ℝ, ∀ b : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 :=
by
  sorry

end no_real_solution_for_inequality_l77_77500


namespace expected_coins_basilio_per_day_l77_77715

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end expected_coins_basilio_per_day_l77_77715


namespace geometric_progression_sum_ratio_l77_77901

theorem geometric_progression_sum_ratio (a : ℝ) (r n : ℕ) (hn : r = 3)
  (h : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28) : n = 6 :=
by
  -- Place the steps of the proof here, which are not required as per instructions.
  sorry

end geometric_progression_sum_ratio_l77_77901


namespace edith_novel_count_l77_77005

-- Definitions based on conditions
variables (N W : ℕ)

-- Conditions from the problem
def condition1 : Prop := N = W / 2
def condition2 : Prop := N + W = 240

-- Target statement
theorem edith_novel_count (N W : ℕ) (h1 : N = W / 2) (h2 : N + W = 240) : N = 80 :=
by
  sorry

end edith_novel_count_l77_77005


namespace find_A_l77_77494

theorem find_A (A B : ℚ) (h1 : B - A = 211.5) (h2 : B = 10 * A) : A = 23.5 :=
by sorry

end find_A_l77_77494


namespace bumper_cars_initial_count_l77_77327

variable {X : ℕ}

theorem bumper_cars_initial_count (h : (X - 6) + 3 = 6) : X = 9 := 
by
  sorry

end bumper_cars_initial_count_l77_77327


namespace hyperbola_hkabc_sum_l77_77906

theorem hyperbola_hkabc_sum :
  ∃ h k a b : ℝ, h = 3 ∧ k = -1 ∧ a = 2 ∧ b = Real.sqrt 46 ∧ h + k + a + b = 4 + Real.sqrt 46 :=
by
  use 3
  use -1
  use 2
  use Real.sqrt 46
  simp
  sorry

end hyperbola_hkabc_sum_l77_77906


namespace total_elephants_l77_77175

-- Definitions and Hypotheses
def W : ℕ := 70
def G : ℕ := 3 * W

-- Proposition
theorem total_elephants : W + G = 280 := by
  sorry

end total_elephants_l77_77175


namespace correct_average_is_18_l77_77309

theorem correct_average_is_18 (incorrect_avg : ℕ) (incorrect_num : ℕ) (true_num : ℕ) (n : ℕ) 
  (h1 : incorrect_avg = 16) (h2 : incorrect_num = 25) (h3 : true_num = 45) (h4 : n = 10) : 
  (incorrect_avg * n + (true_num - incorrect_num)) / n = 18 :=
by
  sorry

end correct_average_is_18_l77_77309


namespace parabola_constant_term_l77_77688

theorem parabola_constant_term 
  (b c : ℝ)
  (h1 : 2 = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c)
  (h2 : 2 = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c) : 
  c = 8 :=
by
  sorry

end parabola_constant_term_l77_77688


namespace find_pastries_made_l77_77985

variable (cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries : ℕ)

def baker_conditions := (cakes_made = 157) ∧ 
                        (total_cakes_sold = 158) ∧ 
                        (total_pastries_sold = 147) ∧ 
                        (more_cakes_than_pastries = 11) ∧ 
                        (extra_cakes = total_cakes_sold - cakes_made) ∧ 
                        (pastries_made = cakes_made - more_cakes_than_pastries)

theorem find_pastries_made : 
  baker_conditions cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries → 
  pastries_made = 146 :=
by
  sorry

end find_pastries_made_l77_77985


namespace max_non_managers_l77_77668

-- Definitions of the problem conditions
variable (m n : ℕ)
variable (h : m = 8)
variable (hratio : (7:ℚ) / 24 < m / n)

-- The theorem we need to prove
theorem max_non_managers (m n : ℕ) (h : m = 8) (hratio : ((7:ℚ) / 24 < m / n)) :
  n ≤ 27 := 
sorry

end max_non_managers_l77_77668


namespace antifreeze_solution_l77_77837

theorem antifreeze_solution (x : ℝ) 
  (h1 : 26 * x + 13 * 0.54 = 39 * 0.58) : 
  x = 0.6 := 
by 
  sorry

end antifreeze_solution_l77_77837


namespace hyunwoo_family_saving_l77_77887

def daily_water_usage : ℝ := 215
def saving_factor : ℝ := 0.32

theorem hyunwoo_family_saving:
  daily_water_usage * saving_factor = 68.8 := by
  sorry

end hyunwoo_family_saving_l77_77887


namespace probability_intervals_l77_77583

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l77_77583


namespace NumberOfStudentsEnrolledOnlyInEnglish_l77_77376

-- Definition of the problem's variables and conditions
variables (TotalStudents BothEnglishAndGerman TotalGerman OnlyEnglish OnlyGerman : ℕ)
variables (h1 : TotalStudents = 52)
variables (h2 : BothEnglishAndGerman = 12)
variables (h3 : TotalGerman = 22)
variables (h4 : TotalStudents = OnlyEnglish + OnlyGerman + BothEnglishAndGerman)
variables (h5 : OnlyGerman = TotalGerman - BothEnglishAndGerman)

-- Theorem to prove the number of students enrolled only in English
theorem NumberOfStudentsEnrolledOnlyInEnglish : OnlyEnglish = 30 :=
by
  -- Insert the necessary proof steps here to derive the number of students enrolled only in English from the given conditions
  sorry

end NumberOfStudentsEnrolledOnlyInEnglish_l77_77376


namespace greatest_three_digit_multiple_of_17_l77_77257

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77257


namespace greatest_three_digit_multiple_of_17_l77_77261

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77261


namespace ab_value_l77_77044

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77044


namespace product_ab_l77_77139

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77139


namespace largest_inscribed_square_l77_77772

-- Define the problem data
noncomputable def s : ℝ := 15
noncomputable def h : ℝ := s * (Real.sqrt 3) / 2
noncomputable def y : ℝ := s - h

-- Statement to prove
theorem largest_inscribed_square :
  y = (30 - 15 * Real.sqrt 3) / 2 := by
  sorry

end largest_inscribed_square_l77_77772


namespace ab_value_l77_77048

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77048


namespace probability_less_than_third_l77_77564

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l77_77564


namespace expected_coins_basilio_l77_77714

-- Define the number of courtyards and the probability of receiving a coin
def num_courtyards : ℕ := 20
def prob_coin : ℝ := 0.5

-- Define the random variable for the total number of coins received
noncomputable def X_Y : ℕ → ℝ := λ n, (n : ℝ) * prob_coin!

-- Expected value of the total number of coins received
noncomputable def E_X_Y : ℝ := X_Y num_courtyards

-- Define the random variable for the difference in coins received by Basilio and Alisa
noncomputable def X_Y_diff : ℕ → ℝ := λ _, (prob_coin)

-- Expected value of the difference in coins received
noncomputable def E_X_Y_diff : ℝ := X_Y_diff 1

-- Define the expected number of coins received by Basilio
noncomputable def E_X : ℝ := (E_X_Y + E_X_Y_diff) / 2

-- Prove the expected number of coins received by Basilio
theorem expected_coins_basilio : E_X = 5.25 := by
  -- using sorry to skip the proof
  sorry

end expected_coins_basilio_l77_77714


namespace num_members_in_league_l77_77402

-- Definitions based on conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def shorts_cost : ℕ := tshirt_cost
def total_cost_per_member : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)
def total_league_cost : ℕ := 4719

-- Theorem statement
theorem num_members_in_league : (total_league_cost / total_cost_per_member) = 74 :=
by
  sorry

end num_members_in_league_l77_77402


namespace prob_two_red_balls_l77_77975

-- Define the initial conditions for the balls in the bag
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 2
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the probability of picking a red ball first
def prob_red1 : ℚ := red_balls / total_balls

-- Define the remaining number of balls and the probability of picking a red ball second
def remaining_red_balls : ℕ := red_balls - 1
def remaining_total_balls : ℕ := total_balls - 1
def prob_red2 : ℚ := remaining_red_balls / remaining_total_balls

-- Define the combined probability of both events
def prob_both_red : ℚ := prob_red1 * prob_red2

-- Statement of the theorem to be proved
theorem prob_two_red_balls : prob_both_red = 5 / 39 := by
  sorry

end prob_two_red_balls_l77_77975


namespace distinct_midpoints_at_least_2n_minus_3_l77_77346

open Set

theorem distinct_midpoints_at_least_2n_minus_3 
  (n : ℕ) 
  (points : Finset (ℝ × ℝ)) 
  (h_points_card : points.card = n) :
  ∃ (midpoints : Finset (ℝ × ℝ)), 
    midpoints.card ≥ 2 * n - 3 := 
sorry

end distinct_midpoints_at_least_2n_minus_3_l77_77346


namespace day_of_week_150th_day_previous_year_l77_77385

theorem day_of_week_150th_day_previous_year (N : ℕ) 
  (h1 : (275 % 7 = 4))  -- Thursday is 4th day of the week if starting from Sunday as 0
  (h2 : (215 % 7 = 4))  -- Similarly, Thursday is 4th day of the week
  : (150 % 7 = 6) :=     -- Proving the 150th day of year N-1 is a Saturday (Saturday as 6th day of the week)
sorry

end day_of_week_150th_day_previous_year_l77_77385


namespace exterior_angle_polygon_l77_77597

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l77_77597


namespace greatest_three_digit_multiple_of_17_l77_77241

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77241


namespace license_plate_difference_l77_77638

theorem license_plate_difference :
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  california_plates - texas_plates = 281216000 :=
by
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  have h1 : california_plates = 456976 * 1000 := by sorry
  have h2 : texas_plates = 17576 * 10000 := by sorry
  have h3 : 456976000 - 175760000 = 281216000 := by sorry
  exact h3

end license_plate_difference_l77_77638


namespace ab_equals_6_l77_77108

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77108


namespace find_remainder_l77_77643

theorem find_remainder (a : ℕ) :
  (a ^ 100) % 73 = 2 ∧ (a ^ 101) % 73 = 69 → a % 73 = 71 :=
by
  sorry

end find_remainder_l77_77643


namespace ab_equals_six_l77_77109

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77109


namespace smallest_integer_l77_77145

theorem smallest_integer (k : ℕ) (n : ℕ) (h936 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : 2^5 ∣ 936 * k)
  (h3 : 3^3 ∣ 936 * k)
  (h4 : 12^2 ∣ 936 * k) : 
  k = 36 :=
by {
  sorry
}

end smallest_integer_l77_77145


namespace largest_three_digit_multiple_of_17_l77_77217

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77217


namespace Linda_needs_15_hours_to_cover_fees_l77_77398

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end Linda_needs_15_hours_to_cover_fees_l77_77398


namespace age_difference_proof_l77_77306

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end age_difference_proof_l77_77306


namespace fill_bucket_completely_l77_77904

theorem fill_bucket_completely (t : ℕ) : (2/3 : ℚ) * t = 100 → t = 150 :=
by
  intro h
  sorry

end fill_bucket_completely_l77_77904


namespace milk_fraction_in_cup1_is_one_third_l77_77783

-- Define the initial state of the cups
structure CupsState where
  cup1_tea : ℚ  -- amount of tea in cup1
  cup1_milk : ℚ -- amount of milk in cup1
  cup2_tea : ℚ  -- amount of tea in cup2
  cup2_milk : ℚ -- amount of milk in cup2

def initial_cups_state : CupsState := {
  cup1_tea := 8,
  cup1_milk := 0,
  cup2_tea := 0,
  cup2_milk := 8
}

-- Function to transfer a fraction of tea from cup 1 to cup 2
def transfer_tea (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea * (1 - frac),
  cup1_milk := s.cup1_milk,
  cup2_tea := s.cup2_tea + s.cup1_tea * frac,
  cup2_milk := s.cup2_milk
}

-- Function to transfer a fraction of the mixture from cup 2 to cup 1
def transfer_mixture (s : CupsState) (frac : ℚ) : CupsState := {
  cup1_tea := s.cup1_tea + (frac * s.cup2_tea),
  cup1_milk := s.cup1_milk + (frac * s.cup2_milk),
  cup2_tea := s.cup2_tea * (1 - frac),
  cup2_milk := s.cup2_milk * (1 - frac)
}

-- Define the state after each transfer
def state_after_tea_transfer := transfer_tea initial_cups_state (1 / 4)
def final_state := transfer_mixture state_after_tea_transfer (1 / 3)

-- Prove the fraction of milk in the first cup is 1/3
theorem milk_fraction_in_cup1_is_one_third : 
  (final_state.cup1_milk / (final_state.cup1_tea + final_state.cup1_milk)) = 1 / 3 :=
by
  -- skipped proof
  sorry

end milk_fraction_in_cup1_is_one_third_l77_77783


namespace rickey_time_l77_77792

/-- Prejean's speed in a race was three-quarters that of Rickey. 
If they both took a total of 70 minutes to run the race, 
the total number of minutes that Rickey took to finish the race is 40. -/
theorem rickey_time (t : ℝ) (h1 : ∀ p : ℝ, p = (3/4) * t) (h2 : t + (3/4) * t = 70) : t = 40 := 
by
  sorry

end rickey_time_l77_77792


namespace calc_result_l77_77331

theorem calc_result (a : ℤ) : 3 * a - 5 * a + a = -a := by
  sorry

end calc_result_l77_77331


namespace triangles_same_base_height_have_equal_areas_l77_77818

theorem triangles_same_base_height_have_equal_areas 
  (b1 h1 b2 h2 : ℝ) 
  (A1 A2 : ℝ) 
  (h1_nonneg : 0 ≤ h1) 
  (h2_nonneg : 0 ≤ h2) 
  (A1_eq : A1 = b1 * h1 / 2) 
  (A2_eq : A2 = b2 * h2 / 2) :
  (A1 = A2 ↔ b1 * h1 = b2 * h2) ∧ (b1 = b2 ∧ h1 = h2 → A1 = A2) :=
by {
  sorry
}

end triangles_same_base_height_have_equal_areas_l77_77818


namespace find_y_l77_77805

theorem find_y (y : ℝ) (h : (15 + 28 + y) / 3 = 25) : y = 32 := by
  sorry

end find_y_l77_77805


namespace exists_integer_in_seq_l77_77768

noncomputable def x_seq (x : ℕ → ℚ) := ∀ n : ℕ, x (n + 1) = x n + 1 / ⌊x n⌋

theorem exists_integer_in_seq {x : ℕ → ℚ} (h1 : 1 < x 1) (h2 : x_seq x) : 
  ∃ n : ℕ, ∃ k : ℤ, x n = k :=
sorry

end exists_integer_in_seq_l77_77768


namespace geometric_sequence_condition_l77_77538

theorem geometric_sequence_condition (A B q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn_def : ∀ n, S n = A * q^n + B) (hq_ne_zero : q ≠ 0) :
  (∀ n, a n = S n - S (n-1)) → (A = -B) ↔ (∀ n, a n = A * (q - 1) * q^(n-1)) := 
sorry

end geometric_sequence_condition_l77_77538


namespace expected_coins_for_cat_basilio_l77_77723

open MeasureTheory

noncomputable def expected_coins_cat : ℝ := let n := 20 in 
  let p := 0.5 in 
  let E_XY := n * p in 
  let E_X_Y := 0.5 * 0 + 0.5 * 1 in 
  (E_XY + E_X_Y) / 2

theorem expected_coins_for_cat_basilio : expected_coins_cat = 5.25 :=
by
  sorry

end expected_coins_for_cat_basilio_l77_77723


namespace interval_probability_l77_77561

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l77_77561


namespace exists_F_squared_l77_77408

theorem exists_F_squared (n : ℕ) : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F (F n)) = n^2 := 
sorry

end exists_F_squared_l77_77408


namespace greatest_three_digit_multiple_of_17_l77_77273

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77273


namespace employee_saves_86_25_l77_77681

def initial_purchase_price : ℝ := 500
def markup_rate : ℝ := 0.15
def employee_discount_rate : ℝ := 0.15

def retail_price : ℝ := initial_purchase_price * (1 + markup_rate)
def employee_discount_amount : ℝ := retail_price * employee_discount_rate
def employee_savings : ℝ := retail_price - (retail_price - employee_discount_amount)

theorem employee_saves_86_25 :
  employee_savings = 86.25 := 
sorry

end employee_saves_86_25_l77_77681


namespace simplify_and_evaluate_l77_77942

variable (a : ℝ)
noncomputable def given_expression : ℝ :=
    (3 * a / (a^2 - 4)) * (1 - 2 / a) - (4 / (a + 2))

theorem simplify_and_evaluate (h : a = Real.sqrt 2 - 1) : 
  given_expression a = 1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l77_77942


namespace ab_eq_six_l77_77085

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77085


namespace greatest_3_digit_multiple_of_17_l77_77294

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77294


namespace anna_walk_distance_l77_77492

theorem anna_walk_distance (d: ℚ) 
  (hd: 22 * 1.25 - 4 * 1.25 = d)
  (d2: d = 3.7): d = 3.7 :=
by 
  sorry

end anna_walk_distance_l77_77492


namespace figurine_cost_is_one_l77_77849

-- Definitions from the conditions
def cost_per_tv : ℕ := 50
def num_tvs : ℕ := 5
def num_figurines : ℕ := 10
def total_spent : ℕ := 260

-- The price of a single figurine
def cost_per_figurine (total_spent num_tvs cost_per_tv num_figurines : ℕ) : ℕ :=
  (total_spent - num_tvs * cost_per_tv) / num_figurines

-- The theorem statement
theorem figurine_cost_is_one : cost_per_figurine total_spent num_tvs cost_per_tv num_figurines = 1 :=
by
  sorry

end figurine_cost_is_one_l77_77849


namespace probability_of_interval_l77_77569

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l77_77569


namespace saree_final_sale_price_in_inr_l77_77321

noncomputable def finalSalePrice (initialPrice: ℝ) (discounts: List ℝ) (conversionRate: ℝ) : ℝ :=
  let finalUSDPrice := discounts.foldl (fun acc discount => acc * (1 - discount)) initialPrice
  finalUSDPrice * conversionRate

theorem saree_final_sale_price_in_inr
  (initialPrice : ℝ := 150)
  (discounts : List ℝ := [0.20, 0.15, 0.05])
  (conversionRate : ℝ := 75)
  : finalSalePrice initialPrice discounts conversionRate = 7267.5 :=
by
  sorry

end saree_final_sale_price_in_inr_l77_77321


namespace regular_hexagon_area_inscribed_in_circle_l77_77450

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l77_77450


namespace ab_value_l77_77051

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77051


namespace inequality_example_l77_77738

theorem inequality_example (a b c : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (sum_eq_one : a + b + c = 1) :
  (a + 1 / a) * (b + 1 / b) * (c + 1 / c) ≥ 1000 / 27 := 
by 
  sorry

end inequality_example_l77_77738


namespace greatest_three_digit_multiple_of_17_l77_77199

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77199


namespace value_of_a_l77_77601

theorem value_of_a (a : ℝ) : (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  intro h
  have h1 : 2 = a - 1 := sorry
  have h2 : 4 = a + 1 := sorry
  have h3 : a = 3 := sorry
  exact h3

end value_of_a_l77_77601


namespace regular_polygon_exterior_angle_l77_77588

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l77_77588


namespace greatest_3_digit_multiple_of_17_l77_77295

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77295


namespace number_of_terms_in_arithmetic_sequence_l77_77504

theorem number_of_terms_in_arithmetic_sequence 
  (a d n l : ℤ) (h1 : a = 7) (h2 : d = 2) (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : n = 70 := 
by sorry

end number_of_terms_in_arithmetic_sequence_l77_77504


namespace no_solution_for_k_eq_six_l77_77533

theorem no_solution_for_k_eq_six :
  ∀ x k : ℝ, k = 6 → (x ≠ 2 ∧ x ≠ 7) → (x - 1) / (x - 2) = (x - k) / (x - 7) → false :=
by 
  intros x k hk hnx_eq h_eq
  sorry

end no_solution_for_k_eq_six_l77_77533


namespace age_difference_l77_77305

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end age_difference_l77_77305


namespace probability_less_than_third_l77_77565

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l77_77565


namespace susan_spent_total_l77_77162

-- Definitions for the costs and quantities
def pencil_cost : ℝ := 0.25
def pen_cost : ℝ := 0.80
def total_items : ℕ := 36
def pencils_bought : ℕ := 16

-- Question: How much did Susan spend?
theorem susan_spent_total : (pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)) = 20 :=
by
    -- definition goes here
    sorry

end susan_spent_total_l77_77162


namespace original_weight_of_marble_l77_77692

theorem original_weight_of_marble (W : ℝ) (h1 : W * 0.75 * 0.85 * 0.90 = 109.0125) : W = 190 :=
by
  sorry

end original_weight_of_marble_l77_77692


namespace f_iterate_result_l77_77781

def f (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1 else 4*n - 3

theorem f_iterate_result : f (f (f 1)) = 17 :=
by
  sorry

end f_iterate_result_l77_77781


namespace greatest_three_digit_multiple_of_17_l77_77235

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77235


namespace value_of_expression_l77_77627

variables {a b c : ℝ}

theorem value_of_expression (h1 : a * b * c = 10) (h2 : a + b + c = 15) (h3 : a * b + b * c + c * a = 25) :
  (2 + a) * (2 + b) * (2 + c) = 128 := 
sorry

end value_of_expression_l77_77627


namespace product_of_ab_l77_77066

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77066


namespace ratio_xz_y2_l77_77530

-- Define the system of equations
def system (k x y z : ℝ) : Prop := 
  x + k * y + 4 * z = 0 ∧ 
  4 * x + k * y - 3 * z = 0 ∧ 
  3 * x + 5 * y - 4 * z = 0

-- Our main theorem to prove the value of xz / y^2 given the system with k = 7.923
theorem ratio_xz_y2 (x y z : ℝ) (h : system 7.923 x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  ∃ r : ℝ, r = (x * z) / (y ^ 2) :=
sorry

end ratio_xz_y2_l77_77530


namespace scientific_notation_of_12400_l77_77652

theorem scientific_notation_of_12400 :
  12400 = 1.24 * 10^4 :=
sorry

end scientific_notation_of_12400_l77_77652


namespace lcm_even_numbers_between_14_and_21_l77_77508

-- Define the even numbers between 14 and 21
def evenNumbers := [14, 16, 18, 20]

-- Define a function to compute the LCM of a list of integers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Theorem statement: the LCM of the even numbers between 14 and 21 equals 5040
theorem lcm_even_numbers_between_14_and_21 :
  lcm_list evenNumbers = 5040 :=
by
  sorry

end lcm_even_numbers_between_14_and_21_l77_77508


namespace total_surfers_is_60_l77_77432

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end total_surfers_is_60_l77_77432


namespace find_value_of_y_l77_77368

theorem find_value_of_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = 7) : y = 52 :=
by
  sorry

end find_value_of_y_l77_77368


namespace final_result_is_106_l77_77693

def chosen_number : ℕ := 122
def multiplied_by_2 (x : ℕ) : ℕ := 2 * x
def subtract_138 (y : ℕ) : ℕ := y - 138

theorem final_result_is_106 : subtract_138 (multiplied_by_2 chosen_number) = 106 :=
by
  -- proof is omitted
  sorry

end final_result_is_106_l77_77693


namespace unique_ab_for_interval_condition_l77_77534

theorem unique_ab_for_interval_condition : 
  ∃! (a b : ℝ), (∀ x, (0 ≤ x ∧ x ≤ 1) → |x^2 - a * x - b| ≤ 1 / 8) ∧ a = 1 ∧ b = -1 / 8 := by
  sorry

end unique_ab_for_interval_condition_l77_77534


namespace range_of_m_l77_77892

theorem range_of_m : 3 < 3 * Real.sqrt 2 - 1 ∧ 3 * Real.sqrt 2 - 1 < 4 :=
by
  have h1 : 3 * Real.sqrt 2 < 5 := sorry
  have h2 : 4 < 3 * Real.sqrt 2 := sorry
  exact ⟨by linarith, by linarith⟩

end range_of_m_l77_77892


namespace greatest_three_digit_multiple_of_17_l77_77198

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77198


namespace probability_a_squared_plus_b_divisible_by_3_l77_77319

theorem probability_a_squared_plus_b_divisible_by_3 : 
  let a_vals := {n ∈ (Finset.range 11).erase 0 | n ∈ ∅ ∪ (Finset.range 10).map Nat.negSuccPnatCoe}
  let b_vals := {(m : ℤ) | m ∈ (Finset.range 11).erase 0}.image (λ x, - (x : ℤ))
  ∃ a ∈ a_vals, ∃ b ∈ b_vals, ↑((Finset.filter (λ x, (fst x) ^ 2 + snd x ≡ 0 [MOD 3]) ((a_vals ×ˢ b_vals)).card) / (a_vals.card * b_vals.card)) = 0.37 :=
by
  sorry

end probability_a_squared_plus_b_divisible_by_3_l77_77319


namespace withdrawal_amount_in_2008_l77_77636

noncomputable def total_withdrawal (a : ℕ) (p : ℝ) : ℝ :=
  (a / p) * ((1 + p) - (1 + p)^8)

theorem withdrawal_amount_in_2008 (a : ℕ) (p : ℝ) (h_pos : 0 < p) (h_neg_one_lt : -1 < p) :
  total_withdrawal a p = (a / p) * ((1 + p) - (1 + p)^8) :=
by
  -- Conditions
  -- Starting from May 10th, 2001, multiple annual deposits.
  -- Annual interest rate p > 0 and p > -1.
  sorry

end withdrawal_amount_in_2008_l77_77636


namespace expected_value_coins_basilio_l77_77728

noncomputable def expected_coins_basilio (n : ℕ) (p : ℚ) : ℚ :=
  let X_Y := Binomial n p
  let E_X_Y := (n * p : ℚ)
  let X_minus_Y := 0.5
  (E_X_Y + X_minus_Y) / 2

theorem expected_value_coins_basilio :
  expected_coins_basilio 20 (1/2) = 5.25 := by
  sorry

end expected_value_coins_basilio_l77_77728


namespace value_of_expression_l77_77366

theorem value_of_expression (m n : ℝ) (h : m + n = -2) : 5 * m^2 + 5 * n^2 + 10 * m * n = 20 := 
by
  sorry

end value_of_expression_l77_77366


namespace probability_merlin_dismissed_l77_77617

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l77_77617


namespace total_pencils_bought_l77_77464

theorem total_pencils_bought (x y : ℕ) (y_pos : 0 < y) (initial_cost : y * (x + 10) = 5 * x) (later_cost : (4 * y) * (x + 10) = 20 * x) :
    x = 15 → (40 = x + x + 10) ∨ x = 40 → (90 = x + (x + 10)) :=
by
  sorry

end total_pencils_bought_l77_77464


namespace necessary_but_not_sufficient_l77_77602

variables {a b : ℝ}

theorem necessary_but_not_sufficient (h : a > 0) (h₁ : a > b) (h₂ : a⁻¹ > b⁻¹) : 
  b < 0 :=
sorry

end necessary_but_not_sufficient_l77_77602


namespace original_kittens_count_l77_77815

theorem original_kittens_count 
  (K : ℕ) 
  (h1 : K - 3 + 9 = 12) : 
  K = 6 := by
sorry

end original_kittens_count_l77_77815


namespace exists_positive_root_in_form_l77_77011

theorem exists_positive_root_in_form :
  ∃ (a b : ℝ), (a + b * real.sqrt 3) > 0 ∧ (a + b * real.sqrt 3) ^ 3 - 4 * (a + b * real.sqrt 3) ^ 2 - 2 * (a + b * real.sqrt 3) - real.sqrt 3 = 0 :=
sorry

end exists_positive_root_in_form_l77_77011


namespace smallest_four_digit_multiple_of_18_l77_77512

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l77_77512


namespace Linda_needs_15_hours_to_cover_fees_l77_77397

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end Linda_needs_15_hours_to_cover_fees_l77_77397


namespace hausdorff_dimension_union_sup_l77_77940

open Set

noncomputable def Hausdorff_dimension (A : Set ℝ) : ℝ :=
sorry -- Definition for Hausdorff dimension is nontrivial and can be added here

theorem hausdorff_dimension_union_sup {A : ℕ → Set ℝ} :
  Hausdorff_dimension (⋃ i, A i) = ⨆ i, Hausdorff_dimension (A i) :=
sorry

end hausdorff_dimension_union_sup_l77_77940


namespace problem_statement_l77_77536

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l77_77536


namespace f_a_plus_b_eq_neg_one_l77_77748

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if x ≥ 0 then x * (x - b) else a * x * (x + 2)

theorem f_a_plus_b_eq_neg_one (a b : ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) a b = -f x a b) 
  (ha : a = -1) 
  (hb : b = 2) : 
  f (a + b) a b = -1 :=
by
  sorry

end f_a_plus_b_eq_neg_one_l77_77748


namespace second_solution_concentration_l77_77765

def volume1 : ℝ := 5
def concentration1 : ℝ := 0.04
def volume2 : ℝ := 2.5
def concentration_final : ℝ := 0.06
def total_silver1 : ℝ := volume1 * concentration1
def total_volume : ℝ := volume1 + volume2
def total_silver_final : ℝ := total_volume * concentration_final

theorem second_solution_concentration :
  ∃ (C2 : ℝ), total_silver1 + volume2 * C2 = total_silver_final ∧ C2 = 0.1 := 
by 
  sorry

end second_solution_concentration_l77_77765


namespace ab_equals_6_l77_77102

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77102


namespace volume_of_box_l77_77476

theorem volume_of_box (L W S : ℕ) (hL : L = 48) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 5120 := by
sorry

end volume_of_box_l77_77476


namespace greatest_three_digit_multiple_of_17_l77_77200

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77200


namespace largest_three_digit_multiple_of_17_l77_77224

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77224


namespace puzzle_pieces_l77_77387

theorem puzzle_pieces (x : ℝ) (h : x + 2 * 1.5 * x = 4000) : x = 1000 :=
  sorry

end puzzle_pieces_l77_77387


namespace sin_330_eq_neg_half_l77_77953

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_half_l77_77953


namespace greatest_three_digit_multiple_of_17_l77_77226

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77226


namespace ab_equals_six_l77_77116

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77116


namespace rachel_bought_3_tables_l77_77630

-- Definitions from conditions
def chairs := 7
def minutes_per_furniture := 4
def total_minutes := 40

-- Define the number of tables Rachel bought
def number_of_tables (chairs : ℕ) (minutes_per_furniture : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes - (chairs * minutes_per_furniture)) / minutes_per_furniture

-- Lean theorem stating the proof problem
theorem rachel_bought_3_tables : number_of_tables chairs minutes_per_furniture total_minutes = 3 :=
by
  sorry

end rachel_bought_3_tables_l77_77630


namespace roots_reciprocal_l77_77373

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 - 1 = 0) (h2 : x2^2 - 3 * x2 - 1 = 0) 
                         (h_sum : x1 + x2 = 3) (h_prod : x1 * x2 = -1) :
  (1 / x1) + (1 / x2) = -3 :=
by
  sorry

end roots_reciprocal_l77_77373


namespace ab_equals_six_l77_77063

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77063


namespace zero_of_f_l77_77186

noncomputable def f (x : ℝ) : ℝ := Real.logb 5 (x - 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 2 :=
by
  use 2
  unfold f
  sorry -- Skip the proof steps, as instructed.

end zero_of_f_l77_77186


namespace common_roots_l77_77854

noncomputable def p (x a : ℝ) := x^3 + a * x^2 + 14 * x + 7
noncomputable def q (x b : ℝ) := x^3 + b * x^2 + 21 * x + 15

theorem common_roots (a b : ℝ) (r s : ℝ) (hr : r ≠ s)
  (hp : p r a = 0) (hp' : p s a = 0)
  (hq : q r b = 0) (hq' : q s b = 0) :
  a = 5 ∧ b = 4 :=
by sorry

end common_roots_l77_77854


namespace mandy_quarters_l77_77784

theorem mandy_quarters (q : ℕ) : 
  40 < q ∧ q < 400 ∧ 
  q % 6 = 2 ∧ 
  q % 7 = 2 ∧ 
  q % 8 = 2 →
  (q = 170 ∨ q = 338) :=
by
  intro h
  sorry

end mandy_quarters_l77_77784


namespace greatest_three_digit_multiple_of_17_l77_77216

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77216


namespace correct_option_is_D_l77_77302

-- Define the expressions to be checked
def exprA (x : ℝ) := 3 * x + 2 * x = 5 * x^2
def exprB (x : ℝ) := -2 * x^2 * x^3 = 2 * x^5
def exprC (x y : ℝ) := (y + 3 * x) * (3 * x - y) = y^2 - 9 * x^2
def exprD (x y : ℝ) := (-2 * x^2 * y)^3 = -8 * x^6 * y^3

theorem correct_option_is_D (x y : ℝ) : 
  ¬ exprA x ∧ 
  ¬ exprB x ∧ 
  ¬ exprC x y ∧ 
  exprD x y := by
  -- The proof would be provided here
  sorry

end correct_option_is_D_l77_77302


namespace oranges_left_to_be_sold_l77_77931

-- Defining the initial conditions
def seven_dozen_oranges : ℕ := 7 * 12
def reserved_for_friend (total : ℕ) : ℕ := total / 4
def remaining_after_reserve (total reserved : ℕ) : ℕ := total - reserved
def sold_yesterday (remaining : ℕ) : ℕ := 3 * remaining / 7
def remaining_after_sale (remaining sold : ℕ) : ℕ := remaining - sold
def remaining_after_rotten (remaining : ℕ) : ℕ := remaining - 4

-- Statement to prove
theorem oranges_left_to_be_sold (total reserved remaining sold final : ℕ) :
  total = seven_dozen_oranges →
  reserved = reserved_for_friend total →
  remaining = remaining_after_reserve total reserved →
  sold = sold_yesterday remaining →
  final = remaining_after_sale remaining sold - 4 →
  final = 32 :=
by
  sorry

end oranges_left_to_be_sold_l77_77931


namespace total_songs_megan_bought_l77_77463

-- Definitions for the problem conditions
def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7
def total_albums : ℕ := country_albums + pop_albums

-- Theorem stating the conclusion we need to prove
theorem total_songs_megan_bought : total_albums * songs_per_album = 70 :=
by
  sorry

end total_songs_megan_bought_l77_77463


namespace smallest_m_exists_l77_77698

theorem smallest_m_exists : ∃ (m : ℕ), (∀ n : ℕ, (n > 0) → ((10000 * n % 53 = 0) → (m ≤ n))) ∧ (10000 * m % 53 = 0) :=
by
  sorry

end smallest_m_exists_l77_77698


namespace geom_seq_ratio_l77_77782

variable {a_1 r : ℚ}
variable {S : ℕ → ℚ}

-- The sum of the first n terms of a geometric sequence
def geom_sum (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * (1 - r^n) / (1 - r)

-- Given conditions
axiom Sn_def : ∀ n, S n = geom_sum a_1 r n
axiom condition : S 10 / S 5 = 1 / 2

-- Theorem to prove
theorem geom_seq_ratio (h : r ≠ 1) : S 15 / S 5 = 3 / 4 :=
by
  -- proof omitted
  sorry

end geom_seq_ratio_l77_77782


namespace ab_equals_six_l77_77062

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77062


namespace rational_xyz_squared_l77_77350

theorem rational_xyz_squared
  (x y z : ℝ)
  (hx : ∃ r1 : ℚ, x + y * z = r1)
  (hy : ∃ r2 : ℚ, y + z * x = r2)
  (hz : ∃ r3 : ℚ, z + x * y = r3)
  (hxy : x^2 + y^2 = 1) :
  ∃ r4 : ℚ, x * y * z^2 = r4 := 
sorry

end rational_xyz_squared_l77_77350


namespace pages_left_l77_77487

variable (a b : ℕ)

theorem pages_left (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

end pages_left_l77_77487


namespace determine_real_coins_l77_77656

def has_fake_coin (coins : List ℝ) : Prop :=
  ∃ fake_coin ∈ coins, (∀ coin ∈ coins, coin ≠ fake_coin)

theorem determine_real_coins (coins : List ℝ) (h : has_fake_coin coins) (h_length : coins.length = 101) :
  ∃ real_coins : List ℝ, ∀ r ∈ real_coins, r ∈ coins ∧ real_coins.length ≥ 50 :=
by
  sorry

end determine_real_coins_l77_77656


namespace smallest_four_digit_multiple_of_18_l77_77529

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l77_77529


namespace solve_inequality_l77_77799

theorem solve_inequality (x : ℝ) :
  (4 * x^4 + x^2 + 4 * x - 5 * x^2 * |x + 2| + 4) ≥ 0 ↔ 
  x ∈ Set.Iic (-1) ∪ Set.Icc ((1 - Real.sqrt 33) / 8) ((1 + Real.sqrt 33) / 8) ∪ Set.Ici 2 :=
by
  sorry

end solve_inequality_l77_77799


namespace smallest_four_digit_multiple_of_18_l77_77519

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l77_77519


namespace greatest_three_digit_multiple_of_17_l77_77270

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77270


namespace ab_value_l77_77091

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77091


namespace no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l77_77629

theorem no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122 :
  ¬ ∃ x y : ℤ, x^2 + 3 * x * y - 2 * y^2 = 122 := sorry

end no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l77_77629


namespace regular_polygon_sides_l77_77691

theorem regular_polygon_sides (n : ℕ) (h : ∀ (x : ℕ), x = 180 * (n - 2) / n → x = 144) :
  n = 10 :=
sorry

end regular_polygon_sides_l77_77691


namespace ab_value_l77_77047

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77047


namespace volume_of_box_l77_77477

theorem volume_of_box (L W S : ℕ) (hL : L = 48) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 5120 := by
sorry

end volume_of_box_l77_77477


namespace none_of_these_l77_77920

theorem none_of_these (s x y : ℝ) (hs : s > 1) (hx2y_ne_zero : x^2 * y ≠ 0) (hineq : x * s^2 > y * s^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < y / x) :=
by
  sorry

end none_of_these_l77_77920


namespace batsman_average_increase_l77_77313

-- Definitions to capture the initial conditions
def runs_scored_in_17th_inning : ℕ := 74
def average_after_17_innings : ℕ := 26

-- Statement to prove the increment in average is 3 runs per inning
theorem batsman_average_increase (A : ℕ) (initial_avg : ℕ)
  (h_initial_runs : 16 * initial_avg + 74 = 17 * 26) :
  26 - initial_avg = 3 :=
by
  sorry

end batsman_average_increase_l77_77313


namespace area_of_inscribed_hexagon_l77_77445

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l77_77445


namespace range_of_a_l77_77883

variable (a : ℝ)

def p : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ (x : ℝ), x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ∈ Set.Iic (-2) ∪ {1} := by
  sorry

end range_of_a_l77_77883


namespace stamps_ratio_l77_77851

noncomputable def number_of_stamps_bought := 300
noncomputable def total_stamps_after_purchase := 450
noncomputable def number_of_stamps_before_purchase := total_stamps_after_purchase - number_of_stamps_bought

theorem stamps_ratio : (number_of_stamps_before_purchase : ℚ) / number_of_stamps_bought = 1 / 2 := by
  have h : number_of_stamps_before_purchase = total_stamps_after_purchase - number_of_stamps_bought := rfl
  rw [h]
  norm_num
  sorry

end stamps_ratio_l77_77851


namespace fuel_consumption_per_100_km_l77_77184

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end fuel_consumption_per_100_km_l77_77184


namespace min_value_ge_9_l77_77888

noncomputable def minValue (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : ℝ :=
  1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2

theorem min_value_ge_9 (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 2)) : minValue θ h ≥ 9 := 
  sorry

end min_value_ge_9_l77_77888


namespace greatest_three_digit_multiple_of_17_l77_77238

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77238


namespace ordered_pair_A_B_l77_77174

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 6
noncomputable def linear_function (x : ℝ) : ℝ := -2 / 3 * x + 2

noncomputable def points_intersect (x1 x2 x3 y1 y2 y3 : ℝ) : Prop :=
  cubic_function x1 = y1 ∧ cubic_function x2 = y2 ∧ cubic_function x3 = y3 ∧
  2 * x1 + 3 * y1 = 6 ∧ 2 * x2 + 3 * y2 = 6 ∧ 2 * x3 + 3 * y3 = 6

theorem ordered_pair_A_B (x1 x2 x3 y1 y2 y3 A B : ℝ)
  (h_intersect : points_intersect x1 x2 x3 y1 y2 y3) 
  (h_sum_x : x1 + x2 + x3 = A)
  (h_sum_y : y1 + y2 + y3 = B) :
  (A, B) = (2, 14 / 3) :=
by {
  sorry
}

end ordered_pair_A_B_l77_77174


namespace min_value_of_x_plus_y_l77_77871

theorem min_value_of_x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
sorry

end min_value_of_x_plus_y_l77_77871


namespace probability_merlin_dismissed_l77_77621

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l77_77621


namespace tan_A_value_l77_77039

open Real

theorem tan_A_value (A : ℝ) (h1 : sin A * (sin A + sqrt 3 * cos A) = -1 / 2) (h2 : 0 < A ∧ A < π) :
  tan A = -sqrt 3 / 3 :=
sorry

end tan_A_value_l77_77039


namespace probability_merlin_dismissed_l77_77614

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l77_77614


namespace interval_probability_l77_77562

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l77_77562


namespace class1_qualified_l77_77317

variables (Tardiness : ℕ → ℕ) -- Tardiness function mapping days to number of tardy students

def classQualified (mean variance median mode : ℕ) : Prop :=
  (mean = 2 ∧ variance = 2) ∨
  (mean = 3 ∧ median = 3) ∨
  (mean = 2 ∧ variance > 0) ∨
  (median = 2 ∧ mode = 2)

def eligible (Tardiness : ℕ → ℕ) : Prop :=
  ∀ i, i < 5 → Tardiness i ≤ 5

theorem class1_qualified : 
  (∀ Tardiness, (∃ mean variance median mode,
    classQualified mean variance median mode 
    ∧ mean = 2 ∧ variance = 2 
    ∧ eligible Tardiness)) → 
  (∀ Tardiness, eligible Tardiness) :=
by
  sorry

end class1_qualified_l77_77317


namespace smallest_four_digit_multiple_of_18_l77_77513

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l77_77513


namespace greatest_three_digit_multiple_of_17_l77_77285

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77285


namespace geometric_sequence_common_ratio_l77_77157

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum_ratio : (a 0 + a 1 + a 2) / a 2 = 7) :
  q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l77_77157


namespace abigail_lost_money_l77_77847

-- Conditions
def initial_money := 11
def money_spent := 2
def money_left := 3

-- Statement of the problem as a Lean theorem
theorem abigail_lost_money : initial_money - money_spent - money_left = 6 := by
  sorry

end abigail_lost_money_l77_77847


namespace work_hours_l77_77400

-- Let h be the number of hours worked
def hours_worked (total_paid part_cost hourly_rate : ℕ) : ℕ :=
  (total_paid - part_cost) / hourly_rate

-- Given conditions
def total_paid : ℕ := 300
def part_cost : ℕ := 150
def hourly_rate : ℕ := 75

-- The statement to be proved
theorem work_hours :
  hours_worked total_paid part_cost hourly_rate = 2 :=
by
  -- Provide the proof here
  sorry

end work_hours_l77_77400


namespace stuart_segments_return_l77_77413

theorem stuart_segments_return (r1 r2 : ℝ) (tangent_chord : ℝ)
  (angle_ABC : ℝ) (h1 : r1 < r2) (h2 : tangent_chord = r1 * 2)
  (h3 : angle_ABC = 75) :
  ∃ (n : ℕ), n = 24 ∧ tangent_chord * n = 360 * (n / 24) :=
by {
  sorry
}

end stuart_segments_return_l77_77413


namespace pictures_at_museum_l77_77663

variable (M : ℕ)

-- Definitions from conditions
def pictures_at_zoo : ℕ := 50
def pictures_deleted : ℕ := 38
def pictures_left : ℕ := 20

-- Theorem to prove the total number of pictures taken including the museum pictures
theorem pictures_at_museum :
  pictures_at_zoo + M - pictures_deleted = pictures_left → M = 8 :=
by
  sorry

end pictures_at_museum_l77_77663


namespace general_term_of_sequence_l77_77033

theorem general_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : ∀ n, S n = 2 * a n - 1) 
    (a₁ : a 1 = 1) :
  ∀ n, a n = 2^(n - 1) := 
sorry

end general_term_of_sequence_l77_77033


namespace geometric_probability_l77_77585

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l77_77585


namespace original_proposition_true_converse_false_l77_77550

-- Lean 4 statement for the equivalent proof problem
theorem original_proposition_true_converse_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬((a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_false_l77_77550


namespace ab_equals_6_l77_77106

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77106


namespace car_speeds_l77_77974

-- Definitions and conditions
def distance_AB : ℝ := 200
def distance_meet : ℝ := 80
def car_A_speed : ℝ := sorry -- To Be Proved
def car_B_speed : ℝ := sorry -- To Be Proved

axiom car_B_faster (x : ℝ) : car_B_speed = car_A_speed + 30
axiom time_equal (x : ℝ) : (distance_meet / car_A_speed) = ((distance_AB - distance_meet) / car_B_speed)

-- Proof (only statement, without steps)
theorem car_speeds : car_A_speed = 60 ∧ car_B_speed = 90 :=
  by
  have car_A_speed := 60
  have car_B_speed := 90
  sorry

end car_speeds_l77_77974


namespace maximal_value_6tuple_l77_77626

theorem maximal_value_6tuple :
  ∀ (a b c d e f : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ 
  a + b + c + d + e + f = 6 → 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ((a, b, c, d, e, f) = (0, 0, t, 2, 2, 2 - t) ∨ 
   (a, b, c, d, e, f) = (0, t, 2, 2 - t, 0, 0) ∨ 
   (a, b, c, d, e, f) = (t, 2, 2 - t, 0, 0, 0) ∨ 
   (a, b, c, d, e, f) = (2, 2 - t, 0, 0, 0, t) ∨
   (a, b, c, d, e, f) = (2 - t, 0, 0, 0, t, 2) ∨
   (a, b, c, d, e, f) = (0, 0, 0, t, 2, 2 - t))) := 
sorry

end maximal_value_6tuple_l77_77626


namespace coloring_15_segments_impossible_l77_77609

theorem coloring_15_segments_impossible :
  ¬ ∃ (colors : Fin 15 → Fin 3) (adj : Fin 15 → Fin 2),
    ∀ i j, adj i = adj j → i ≠ j → colors i ≠ colors j :=
by
  sorry

end coloring_15_segments_impossible_l77_77609


namespace ice_cream_scoops_l77_77989

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end ice_cream_scoops_l77_77989


namespace total_accidents_l77_77151

noncomputable def A (k x : ℕ) : ℕ := 96 + k * x

theorem total_accidents :
  let k_morning := 1
  let k_evening := 3
  let x_morning := 2000
  let x_evening := 1000
  A k_morning x_morning + A k_evening x_evening = 5192 := by
  sorry

end total_accidents_l77_77151


namespace regular_hexagon_area_l77_77452

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l77_77452


namespace math_proof_l77_77801

noncomputable def side_length_of_smaller_square (d e f : ℕ) : ℝ :=
  (d - Real.sqrt e) / f

def are_positive_integers (d e f : ℕ) : Prop := d > 0 ∧ e > 0 ∧ f > 0
def is_not_divisible_by_square_of_any_prime (e : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p * p ∣ e)

def proof_problem : Prop :=
  ∃ (d e f : ℕ),
    are_positive_integers d e f ∧
    is_not_divisible_by_square_of_any_prime e ∧
    side_length_of_smaller_square d e f = (4 - Real.sqrt 10) / 3 ∧
    d + e + f = 17

theorem math_proof : proof_problem := sorry

end math_proof_l77_77801


namespace zeros_of_quadratic_l77_77651

theorem zeros_of_quadratic : ∃ x : ℝ, x^2 - x - 2 = 0 -> (x = -1 ∨ x = 2) :=
by
  sorry

end zeros_of_quadratic_l77_77651


namespace option_D_forms_triangle_l77_77462

theorem option_D_forms_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 9) : 
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end option_D_forms_triangle_l77_77462


namespace cakes_left_l77_77850

def initial_cakes : ℕ := 62
def additional_cakes : ℕ := 149
def cakes_sold : ℕ := 144

theorem cakes_left : (initial_cakes + additional_cakes) - cakes_sold = 67 :=
by
  sorry

end cakes_left_l77_77850


namespace regular_polygon_exterior_angle_l77_77589

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l77_77589


namespace total_ingredients_l77_77180

theorem total_ingredients (b f s : ℕ) (h_ratio : 2 * f = 5 * f) (h_flour : f = 15) : b + f + s = 30 :=
by 
  sorry

end total_ingredients_l77_77180


namespace problem1_problem2_l77_77905

-- Definitions for the number of combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1
theorem problem1 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 4) + (C r 3 * C w 1) + (C r 2 * C w 2) = 115 := 
sorry

-- Problem 2
theorem problem2 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 2 * C w 3) + (C r 3 * C w 2) + (C r 4 * C w 1) = 186 := 
sorry

end problem1_problem2_l77_77905


namespace problem_statement_l77_77535

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l77_77535


namespace expected_coins_for_cat_basilio_l77_77725

open MeasureTheory

noncomputable def expected_coins_cat : ℝ := let n := 20 in 
  let p := 0.5 in 
  let E_XY := n * p in 
  let E_X_Y := 0.5 * 0 + 0.5 * 1 in 
  (E_XY + E_X_Y) / 2

theorem expected_coins_for_cat_basilio : expected_coins_cat = 5.25 :=
by
  sorry

end expected_coins_for_cat_basilio_l77_77725


namespace inequality_proof_l77_77164

variable {a b c d : ℝ}

theorem inequality_proof
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_pos_d : 0 < d)
  (h_inequality : a / b < c / d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := 
by
  sorry

end inequality_proof_l77_77164


namespace largest_possible_sum_l77_77311

def max_sum_pair_mult_48 : Prop :=
  ∃ (heartsuit clubsuit : ℕ), (heartsuit * clubsuit = 48) ∧ (heartsuit + clubsuit = 49) ∧ 
  (∀ (h c : ℕ), (h * c = 48) → (h + c ≤ 49))

theorem largest_possible_sum : max_sum_pair_mult_48 :=
  sorry

end largest_possible_sum_l77_77311


namespace initial_gasohol_amount_l77_77315

variable (x : ℝ)

def gasohol_ethanol_percentage (initial_gasohol : ℝ) := 0.05 * initial_gasohol
def mixture_ethanol_percentage (initial_gasohol : ℝ) := gasohol_ethanol_percentage initial_gasohol + 3

def optimal_mixture (total_volume : ℝ) := 0.10 * total_volume

theorem initial_gasohol_amount :
  ∀ (initial_gasohol : ℝ), 
  mixture_ethanol_percentage initial_gasohol = optimal_mixture (initial_gasohol + 3) →
  initial_gasohol = 54 :=
by
  intros
  sorry

end initial_gasohol_amount_l77_77315


namespace expected_coins_basilio_20_l77_77720

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end expected_coins_basilio_20_l77_77720


namespace time_to_install_rest_of_windows_l77_77843

-- Definition of the given conditions:
def num_windows_needed : ℕ := 10
def num_windows_installed : ℕ := 6
def install_time_per_window : ℕ := 5

-- Statement that we aim to prove:
theorem time_to_install_rest_of_windows :
  install_time_per_window * (num_windows_needed - num_windows_installed) = 20 := by
  sorry

end time_to_install_rest_of_windows_l77_77843


namespace max_expression_value_l77_77009

theorem max_expression_value {x y : ℝ} (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) :
  x^2 + y^2 ≤ 10 :=
sorry

end max_expression_value_l77_77009


namespace largest_constant_inequality_l77_77862

theorem largest_constant_inequality (C : ℝ) (h : ∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) : 
  C ≤ 2 / Real.sqrt 3 :=
sorry

end largest_constant_inequality_l77_77862


namespace min_z_value_l77_77038

variable (x y z : ℝ)

theorem min_z_value (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  z = x - y → z = -1 :=
by sorry

end min_z_value_l77_77038


namespace brother_reading_time_l77_77405

variable (my_time_in_hours : ℕ)
variable (speed_ratio : ℕ)

theorem brother_reading_time
  (h1 : my_time_in_hours = 3)
  (h2 : speed_ratio = 4) :
  my_time_in_hours * 60 / speed_ratio = 45 := 
by
  sorry

end brother_reading_time_l77_77405


namespace polynomial_identity_l77_77155

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_identity (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h : ∀ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = g (f x)) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end polynomial_identity_l77_77155


namespace greatest_three_digit_multiple_of_17_l77_77260

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77260


namespace ab_equals_six_l77_77064

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77064


namespace piecewise_function_continuity_l77_77925

theorem piecewise_function_continuity :
  (∀ x, if x > (3 : ℝ) 
        then 2 * (a : ℝ) * x + 4 = (x : ℝ) ^ 2 - 1
        else if x < -1 
        then 3 * (x : ℝ) - (c : ℝ) = (x : ℝ) ^ 2 - 1
        else (x : ℝ) ^ 2 - 1 = (x : ℝ) ^ 2 - 1) →
  a = 2 / 3 →
  c = -3 →
  a + c = -7 / 3 :=
by
  intros h ha hc
  simp [ha, hc]
  sorry

end piecewise_function_continuity_l77_77925


namespace draw_13_cards_no_straight_flush_l77_77314

theorem draw_13_cards_no_straight_flush :
  let deck_size := 52
  let suit_count := 4
  let rank_count := 13
  let non_straight_flush_draws (n : ℕ) := 3^n - 3
  n = rank_count →
  ∀ (draw : ℕ), draw = non_straight_flush_draws n :=
by
-- Proof would be here
sorry

end draw_13_cards_no_straight_flush_l77_77314


namespace exterior_angle_polygon_l77_77598

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l77_77598


namespace min_value_proof_l77_77921

theorem min_value_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x - 2 * y + 3 * z = 0) : 3 = 3 :=
by
  sorry

end min_value_proof_l77_77921


namespace greatest_three_digit_multiple_of_17_l77_77287

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77287


namespace real_mul_eq_zero_iff_l77_77661

theorem real_mul_eq_zero_iff (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end real_mul_eq_zero_iff_l77_77661


namespace greatest_three_digit_multiple_of_17_l77_77227

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77227


namespace greatest_three_digit_multiple_of_17_l77_77282

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77282


namespace greatest_three_digit_multiple_of_17_l77_77248

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77248


namespace archie_initial_marbles_l77_77696

theorem archie_initial_marbles (M : ℝ) (h1 : 0.6 * M + 0.5 * 0.4 * M = M - 20) : M = 100 :=
sorry

end archie_initial_marbles_l77_77696


namespace largest_three_digit_multiple_of_17_l77_77222

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77222


namespace solution_set_of_inequality_l77_77675

theorem solution_set_of_inequality :
  { x : ℝ | abs (x - 4) + abs (3 - x) < 2 } = { x : ℝ | 2.5 < x ∧ x < 4.5 } := sorry

end solution_set_of_inequality_l77_77675


namespace max_distance_l77_77650

noncomputable def starting_cost : ℝ := 10
noncomputable def additional_cost_per_km : ℝ := 1.5
noncomputable def round_up : ℝ := 1
noncomputable def total_fare : ℝ := 19

theorem max_distance (x : ℝ) : (starting_cost + additional_cost_per_km * (x - 4)) = total_fare → x = 10 :=
by sorry

end max_distance_l77_77650


namespace greatest_three_digit_multiple_of_17_l77_77237

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77237


namespace greatest_3_digit_multiple_of_17_l77_77296

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77296


namespace greatest_three_digit_multiple_of_17_l77_77230

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77230


namespace value_of_m_l77_77031

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end value_of_m_l77_77031


namespace probability_merlin_dismissed_l77_77623

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l77_77623


namespace probability_merlin_dismissed_l77_77616

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l77_77616


namespace distinct_sum_values_l77_77359

open Finset

noncomputable def is_arithmetic_seq (B : ℕ → ℝ) := ∃ d : ℝ, ∀ i : ℕ, B (i+1) = B i + d

theorem distinct_sum_values (n : ℕ) (B : ℕ → ℝ) (h1 : is_arithmetic_seq B) :
  (card (image (λ (p : ℕ × ℕ), B p.1 + B p.2)
     (filter (λ (p : ℕ × ℕ), p.1 < p.2) (range n).product (range n)))) = 2 * n - 3 := sorry

end distinct_sum_values_l77_77359


namespace expected_coins_for_cat_basilio_l77_77726

open MeasureTheory

noncomputable def expected_coins_cat : ℝ := let n := 20 in 
  let p := 0.5 in 
  let E_XY := n * p in 
  let E_X_Y := 0.5 * 0 + 0.5 * 1 in 
  (E_XY + E_X_Y) / 2

theorem expected_coins_for_cat_basilio : expected_coins_cat = 5.25 :=
by
  sorry

end expected_coins_for_cat_basilio_l77_77726


namespace equivalent_statements_l77_77965

variable (P Q R : Prop)

theorem equivalent_statements :
  ((¬ P ∧ ¬ Q) → R) ↔ (P ∨ Q ∨ R) :=
sorry

end equivalent_statements_l77_77965


namespace triangle_properties_l77_77379

noncomputable def triangle_side_lengths (m1 m2 m3 : ℝ) : Prop :=
  ∃ a b c s,
    m1 = 20 ∧
    m2 = 24 ∧
    m3 = 30 ∧
    a = 36.28 ∧
    b = 30.24 ∧
    c = 24.19 ∧
    s = 362.84

theorem triangle_properties :
  triangle_side_lengths 20 24 30 :=
by
  sorry

end triangle_properties_l77_77379


namespace second_lock_less_than_three_times_first_l77_77776

variable (first_lock_time : ℕ := 5)
variable (second_lock_time : ℕ)
variable (combined_lock_time : ℕ := 60)

-- Assuming the second lock time is a fraction of the combined lock time
axiom h1 : 5 * second_lock_time = combined_lock_time

theorem second_lock_less_than_three_times_first : (3 * first_lock_time - second_lock_time) = 3 :=
by
  -- prove that the theorem is true based on given conditions.
  sorry

end second_lock_less_than_three_times_first_l77_77776


namespace quotient_with_zero_in_middle_l77_77674

theorem quotient_with_zero_in_middle : 
  ∃ (op : ℕ → ℕ → ℕ), 
  (op = Nat.add ∧ ((op 6 4) / 3).digits 10 = [3, 0, 3]) := 
by 
  sorry

end quotient_with_zero_in_middle_l77_77674


namespace part1_l77_77834

theorem part1 (a : ℤ) (h : a = -2) : 
  ((a^2 + a) / (a^2 - 3 * a)) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2 / 3 := by
  rw [h]
  sorry

end part1_l77_77834


namespace probability_less_than_one_third_l77_77566

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l77_77566


namespace number_of_elements_in_set_S_l77_77938

-- Define the set S and its conditions
variable (S : Set ℝ) (n : ℝ) (sumS : ℝ)

-- Conditions given in the problem
axiom avg_S : sumS / n = 6.2
axiom new_avg_S : (sumS + 8) / n = 7

-- The statement to be proved
theorem number_of_elements_in_set_S : n = 10 := by 
  sorry

end number_of_elements_in_set_S_l77_77938


namespace john_candies_l77_77785

theorem john_candies (mark_candies : ℕ) (peter_candies : ℕ) (total_candies : ℕ) (equal_share : ℕ) (h1 : mark_candies = 30) (h2 : peter_candies = 25) (h3 : total_candies = 90) (h4 : equal_share * 3 = total_candies) : 
  (total_candies - mark_candies - peter_candies = 35) :=
by
  sorry

end john_candies_l77_77785


namespace range_of_a_l77_77737

variable (A B : Set ℝ) (a : ℝ)

def setA : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def setB : Set ℝ := {x | (2^(1 - x) + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

theorem range_of_a :
  A ⊆ B ↔ (-4 ≤ a) ∧ (a ≤ -1) :=
by
  sorry

end range_of_a_l77_77737


namespace trader_profit_l77_77322

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def discount_price (P : ℝ) : ℝ := 0.95 * P
noncomputable def selling_price (P : ℝ) : ℝ := 1.52 * P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def percent_profit (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) (hP : 0 < P) : percent_profit P = 52 := by 
  sorry

end trader_profit_l77_77322


namespace parabola_y_axis_intersection_l77_77424

theorem parabola_y_axis_intersection:
  (∀ x y : ℝ, y = -2 * (x - 1)^2 - 3 → x = 0 → y = -5) :=
by
  intros x y h_eq h_x
  sorry

end parabola_y_axis_intersection_l77_77424


namespace ab_value_l77_77052

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77052


namespace probability_merlin_dismissed_l77_77619

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l77_77619


namespace perimeter_of_square_B_l77_77635

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_B_l77_77635


namespace scientific_notation_448000_l77_77810

theorem scientific_notation_448000 :
  ∃ a n, (448000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 :=
by
  sorry

end scientific_notation_448000_l77_77810


namespace greatest_three_digit_multiple_of_17_l77_77205

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77205


namespace hourly_wage_calculation_l77_77167

variable (H : ℝ)
variable (hours_per_week : ℝ := 40)
variable (wage_per_widget : ℝ := 0.16)
variable (widgets_per_week : ℝ := 500)
variable (total_earnings : ℝ := 580)

theorem hourly_wage_calculation :
  (hours_per_week * H + widgets_per_week * wage_per_widget = total_earnings) →
  H = 12.5 :=
by
  intro h_equation
  -- Proof steps would go here
  sorry

end hourly_wage_calculation_l77_77167


namespace no_real_number_pairs_satisfy_equation_l77_77040

theorem no_real_number_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ¬ (1 / a + 1 / b = 1 / (2 * a + 3 * b)) :=
by
  intros a b ha hb
  sorry

end no_real_number_pairs_satisfy_equation_l77_77040


namespace particular_solution_satisfies_l77_77342

noncomputable def particular_solution (x : ℝ) : ℝ :=
  (1/3) * Real.exp (-4 * x) - (1/3) * Real.exp (2 * x) + (x ^ 2 + 3 * x) * Real.exp (2 * x)

def initial_conditions (f df : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ df 0 = 1

def differential_equation (f df ddf : ℝ → ℝ) : Prop :=
  ∀ x, ddf x + 2 * df x - 8 * f x = (12 * x + 20) * Real.exp (2 * x)

theorem particular_solution_satisfies :
  ∃ C1 C2 : ℝ, initial_conditions (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
              (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) ∧ 
              differential_equation (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
                                  (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) 
                                  (λ x => 16 * C1 * Real.exp (-4 * x) + 4 * C2 * Real.exp (2 * x) + (4 * x^2 + 12 * x + 1) * Real.exp (2 * x)) :=
sorry

end particular_solution_satisfies_l77_77342


namespace greatest_three_digit_multiple_of_17_l77_77249

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77249


namespace product_ab_l77_77135

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77135


namespace sufficient_but_not_necessary_l77_77833

-- Let's define the conditions and the theorem to be proved in Lean 4
theorem sufficient_but_not_necessary : ∀ x : ℝ, (x > 1 → x > 0) ∧ ¬(∀ x : ℝ, x > 0 → x > 1) := by
  sorry

end sufficient_but_not_necessary_l77_77833


namespace bridget_apples_l77_77328

theorem bridget_apples (x : ℕ) (h1 : x - 2 ≥ 0) (h2 : (x - 2) / 3 = 0 → false)
    (h3 : (2 * (x - 2) / 3) - 5 = 6) : x = 20 :=
by
  sorry

end bridget_apples_l77_77328


namespace ab_equals_six_l77_77115

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l77_77115


namespace balloons_left_l77_77966

def total_balloons (r w g c: Nat) : Nat := r + w + g + c

def num_friends : Nat := 10

theorem balloons_left (r w g c : Nat) (total := total_balloons r w g c) (h_r : r = 24) (h_w : w = 38) (h_g : g = 68) (h_c : c = 75) :
  total % num_friends = 5 := by
  sorry

end balloons_left_l77_77966


namespace mixed_oil_rate_is_correct_l77_77666

def rate_of_mixed_oil (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℕ :=
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2)

theorem mixed_oil_rate_is_correct :
  rate_of_mixed_oil 10 50 5 68 = 56 := by
  sorry

end mixed_oil_rate_is_correct_l77_77666


namespace car_travel_distance_l77_77678

theorem car_travel_distance :
  let a := 36
  let d := -12
  let n := 4
  let S := (n / 2) * (2 * a + (n - 1) * d)
  S = 72 := by
    sorry

end car_travel_distance_l77_77678


namespace f_1987_is_3_l77_77555

noncomputable def f : ℕ → ℕ :=
sorry

axiom f_is_defined : ∀ x : ℕ, f x ≠ 0
axiom f_initial : f 1 = 3
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b) + 1

theorem f_1987_is_3 : f 1987 = 3 :=
by
  -- Here we would provide the mathematical proof
  sorry

end f_1987_is_3_l77_77555


namespace problem_statement_l77_77345

theorem problem_statement (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
(a + b = 2) ∧ ¬( (a^2 + a > 2) ∧ (b^2 + b > 2) ) := by
  sorry

end problem_statement_l77_77345


namespace largest_value_f12_l77_77393

theorem largest_value_f12 (f : ℝ → ℝ) (hf_poly : ∀ x, f x ≥ 0) 
  (hf_6 : f 6 = 24) (hf_24 : f 24 = 1536) :
  f 12 ≤ 192 :=
sorry

end largest_value_f12_l77_77393


namespace smallest_four_digit_multiple_of_18_l77_77520

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l77_77520


namespace fraction_difference_in_simplest_form_l77_77008

noncomputable def difference_fraction : ℚ := (5 / 19) - (2 / 23)

theorem fraction_difference_in_simplest_form :
  difference_fraction = 77 / 437 := by sorry

end fraction_difference_in_simplest_form_l77_77008


namespace heroes_can_reduce_heads_to_zero_l77_77956

-- Definition of the Hero strikes
def IlyaMurometsStrikes (H : ℕ) : ℕ := H / 2 - 1
def DobrynyaNikitichStrikes (H : ℕ) : ℕ := 2 * H / 3 - 2
def AlyoshaPopovichStrikes (H : ℕ) : ℕ := 3 * H / 4 - 3

-- The ultimate goal is proving this theorem
theorem heroes_can_reduce_heads_to_zero (H : ℕ) : 
  ∃ (n : ℕ), ∀ i ≤ n, 
  (if i % 3 = 0 then H = 0 
   else if i % 3 = 1 then IlyaMurometsStrikes H = 0 
   else if i % 3 = 2 then DobrynyaNikitichStrikes H = 0 
   else AlyoshaPopovichStrikes H = 0)
:= sorry

end heroes_can_reduce_heads_to_zero_l77_77956


namespace increasing_interval_of_f_l77_77808

open Real

def f (x : ℝ) : ℝ := sqrt 3 * cos (x / 2) ^ 2 - 1 / 2 * sin x - sqrt 3 / 2

theorem increasing_interval_of_f :
  ∀ x, x ∈ Icc 0 π → f (x) = cos (x + π / 6) →
  (∀ x1 x2, x1 ∈ Icc (5 * π / 6) π → x2 ∈ Icc (5 * π / 6) π → x1 < x2 → f(x1) < f(x2)) :=
by
  sorry

end increasing_interval_of_f_l77_77808


namespace students_at_table_l77_77410

def numStudents (candies : ℕ) (first_last : ℕ) (st_len : ℕ) : Prop :=
  candies - 1 = st_len * first_last

theorem students_at_table 
  (candies : ℕ)
  (first_last : ℕ)
  (st_len : ℕ)
  (h1 : candies = 120) 
  (h2 : first_last = 1) :
  (st_len = 7 ∨ st_len = 17) :=
by
  sorry

end students_at_table_l77_77410


namespace smallest_four_digit_multiple_of_18_l77_77523

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l77_77523


namespace regular_polygon_sides_l77_77865

theorem regular_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 144 * n) : n = 10 := 
by
  sorry

end regular_polygon_sides_l77_77865


namespace inequality_bound_l77_77922

theorem inequality_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ( (2 * a + b + c)^2 / (2 * a ^ 2 + (b + c) ^2) + 
    (2 * b + c + a)^2 / (2 * b ^ 2 + (c + a) ^2) + 
    (2 * c + a + b)^2 / (2 * c ^ 2 + (a + b) ^2) ) ≤ 8 := 
sorry

end inequality_bound_l77_77922


namespace ab_value_l77_77050

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l77_77050


namespace find_a6_a7_l77_77030

variable {a : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom sum_given : a 2 + a 3 + a 10 + a 11 = 48

theorem find_a6_a7 (arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d) (h : a 2 + a 3 + a 10 + a 11 = 48) :
  a 6 + a 7 = 24 :=
by
  sorry

end find_a6_a7_l77_77030


namespace pentagon_largest_angle_l77_77437

theorem pentagon_largest_angle
    (P Q : ℝ)
    (hP : P = 55)
    (hQ : Q = 120)
    (R S T : ℝ)
    (hR_eq_S : R = S)
    (hT : T = 2 * R + 20):
    R + S + T + P + Q = 540 → T = 192.5 :=
by
    sorry

end pentagon_largest_angle_l77_77437


namespace largest_perimeter_l77_77868

noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

noncomputable def condition (n1 n2 n3 n4 : ℕ) : Prop :=
  2 * interior_angle n1 + interior_angle n2 + interior_angle n3 = 360

theorem largest_perimeter
  {n1 n2 n3 n4 : ℕ}
  (h : n1 = n4)
  (h_condition : condition n1 n2 n3 n4) :
  4 * n1 + 2 * n2 + 2 * n3 - 8 ≤ 22 :=
sorry

end largest_perimeter_l77_77868


namespace factorization_of_expression_l77_77338

theorem factorization_of_expression (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := by
  sorry

end factorization_of_expression_l77_77338


namespace greatest_three_digit_multiple_of_17_l77_77280

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77280


namespace probability_less_than_one_third_l77_77568

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l77_77568


namespace c_completion_days_l77_77308

noncomputable def work_rate (days: ℕ) := (1 : ℝ) / days

theorem c_completion_days : 
  ∀ (W : ℝ) (Ra Rb Rc : ℝ) (Dc : ℕ),
  Ra = work_rate 30 → Rb = work_rate 30 → Rc = work_rate Dc →
  (Ra + Rb + Rc) * 8 + (Ra + Rb) * 4 = W → 
  Dc = 40 :=
by
  intros W Ra Rb Rc Dc hRa hRb hRc hW
  sorry

end c_completion_days_l77_77308


namespace expected_coins_basilio_l77_77711

-- Define the number of courtyards and the probability of receiving a coin
def num_courtyards : ℕ := 20
def prob_coin : ℝ := 0.5

-- Define the random variable for the total number of coins received
noncomputable def X_Y : ℕ → ℝ := λ n, (n : ℝ) * prob_coin!

-- Expected value of the total number of coins received
noncomputable def E_X_Y : ℝ := X_Y num_courtyards

-- Define the random variable for the difference in coins received by Basilio and Alisa
noncomputable def X_Y_diff : ℕ → ℝ := λ _, (prob_coin)

-- Expected value of the difference in coins received
noncomputable def E_X_Y_diff : ℝ := X_Y_diff 1

-- Define the expected number of coins received by Basilio
noncomputable def E_X : ℝ := (E_X_Y + E_X_Y_diff) / 2

-- Prove the expected number of coins received by Basilio
theorem expected_coins_basilio : E_X = 5.25 := by
  -- using sorry to skip the proof
  sorry

end expected_coins_basilio_l77_77711


namespace shaded_area_is_correct_l77_77425

def area_of_rectangle (l w : ℕ) : ℕ := l * w

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

def area_of_shaded_region : ℕ :=
  let length := 8
  let width := 4
  let area_rectangle := area_of_rectangle length width
  let area_triangle := area_of_triangle length width
  area_rectangle - area_triangle

theorem shaded_area_is_correct : area_of_shaded_region = 16 :=
by
  sorry

end shaded_area_is_correct_l77_77425


namespace min_students_l77_77628

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : (b + g) % 5 = 2) : 
  b + g = 57 :=
sorry

end min_students_l77_77628


namespace fabric_problem_l77_77653

theorem fabric_problem
  (x y : ℝ)
  (h1 : y > 0)
  (cost_second_piece := x)
  (cost_first_piece := x + 126)
  (cost_per_meter_first := (x + 126) / y)
  (cost_per_meter_second := x / y)
  (h2 : 4 * cost_per_meter_first - 3 * cost_per_meter_second = 135)
  (h3 : 3 * cost_per_meter_first + 4 * cost_per_meter_second = 382.5) :
  y = 5.6 ∧ cost_per_meter_first = 67.5 ∧ cost_per_meter_second = 45 :=
sorry

end fabric_problem_l77_77653


namespace smallest_four_digit_multiple_of_18_l77_77514

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l77_77514


namespace monroe_collection_legs_l77_77403

theorem monroe_collection_legs : 
  let ants := 12 
  let spiders := 8 
  let beetles := 15 
  let centipedes := 5 
  let legs_ants := 6 
  let legs_spiders := 8 
  let legs_beetles := 6 
  let legs_centipedes := 100
  (ants * legs_ants + spiders * legs_spiders + beetles * legs_beetles + centipedes * legs_centipedes = 726) := 
by 
  sorry

end monroe_collection_legs_l77_77403


namespace ab_eq_six_l77_77086

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77086


namespace intersection_P_Q_l77_77360

def P := {x : ℝ | x^2 - 9 < 0}
def Q := {y : ℤ | ∃ x : ℤ, y = 2*x}

theorem intersection_P_Q :
  {x : ℝ | x ∈ P ∧ (∃ n : ℤ, x = 2*n)} = {-2, 0, 2} :=
by
  sorry

end intersection_P_Q_l77_77360


namespace common_root_polynomials_l77_77855

theorem common_root_polynomials (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end common_root_polynomials_l77_77855


namespace olivia_wallet_after_shopping_l77_77406

variable (initial_wallet : ℝ := 200) 
variable (groceries : ℝ := 65)
variable (shoes_original_price : ℝ := 75)
variable (shoes_discount_rate : ℝ := 0.15)
variable (belt : ℝ := 25)

theorem olivia_wallet_after_shopping :
  initial_wallet - (groceries + (shoes_original_price - shoes_original_price * shoes_discount_rate) + belt) = 46.25 := by
  sorry

end olivia_wallet_after_shopping_l77_77406


namespace greatest_3_digit_multiple_of_17_l77_77291

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77291


namespace product_of_two_numbers_l77_77640

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 :=
by
  sorry

end product_of_two_numbers_l77_77640


namespace ab_value_l77_77089

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77089


namespace square_of_integer_l77_77391

theorem square_of_integer (n : ℕ) (h : ∃ l : ℤ, l^2 = 1 + 12 * (n^2 : ℤ)) :
  ∃ m : ℤ, 2 + 2 * Int.sqrt (1 + 12 * (n^2 : ℤ)) = m^2 := by
  sorry

end square_of_integer_l77_77391


namespace ab_value_l77_77090

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77090


namespace ab_equals_six_l77_77060

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77060


namespace probability_of_interval_l77_77571

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l77_77571


namespace cyclist_wait_20_minutes_l77_77420

noncomputable def cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (time_passed_minutes : ℝ) : ℝ :=
  let time_passed_hours := time_passed_minutes / 60
  let distance := cyclist_speed * time_passed_hours
  let hiker_catch_up_time := distance / hiker_speed
  hiker_catch_up_time * 60

theorem cyclist_wait_20_minutes :
  cyclist_wait_time 5 20 5 = 20 :=
by
  -- Definitions according to given conditions
  let hiker_speed := 5 -- miles per hour
  let cyclist_speed := 20 -- miles per hour
  let time_passed_minutes := 5
  -- Required result
  let result_needed := 20
  -- Using the cyclist_wait_time function
  show cyclist_wait_time hiker_speed cyclist_speed time_passed_minutes = result_needed
  sorry

end cyclist_wait_20_minutes_l77_77420


namespace train_crossing_time_l77_77669

theorem train_crossing_time 
    (length : ℝ) (speed_kmph : ℝ) 
    (conversion_factor: ℝ) (speed_mps: ℝ) 
    (time : ℝ) :
  length = 400 ∧ speed_kmph = 144 ∧ conversion_factor = 1000 / 3600 ∧ speed_mps = speed_kmph * conversion_factor ∧ time = length / speed_mps → time = 10 := 
by 
  sorry

end train_crossing_time_l77_77669


namespace total_legs_walking_on_ground_l77_77680

def horses : ℕ := 16
def men : ℕ := 16

def men_walking := men / 2
def men_riding := men / 2

def legs_per_man := 2
def legs_per_horse := 4

def legs_for_men_walking := men_walking * legs_per_man
def legs_for_horses := horses * legs_per_horse

theorem total_legs_walking_on_ground : legs_for_men_walking + legs_for_horses = 80 := 
by
  sorry

end total_legs_walking_on_ground_l77_77680


namespace isosceles_triangle_area_l77_77491

theorem isosceles_triangle_area (s b : ℝ) (h₁ : s + b = 20) (h₂ : b^2 + 10^2 = s^2) : 
  1/2 * 2 * b * 10 = 75 :=
by sorry

end isosceles_triangle_area_l77_77491


namespace divide_equal_parts_l77_77438

theorem divide_equal_parts (m n: ℕ) (h₁: (m + n) % 2 = 0) (h₂: gcd m n ∣ ((m + n) / 2)) : ∃ a b: ℕ, a = b ∧ a + b = m + n ∧ a ≤ m + n ∧ b ≤ m + n :=
sorry

end divide_equal_parts_l77_77438


namespace number_of_possible_monograms_l77_77404

-- Define the set of letters before 'M'
def letters_before_M : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}

-- Define the set of letters after 'M'
def letters_after_M : Finset Char := {'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

-- State the theorem 
theorem number_of_possible_monograms : 
  (letters_before_M.card * letters_after_M.card) = 156 :=
by
  sorry

end number_of_possible_monograms_l77_77404


namespace greatest_three_digit_multiple_of_17_l77_77247

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77247


namespace smallest_four_digit_multiple_of_18_l77_77516

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l77_77516


namespace reflected_ray_eqn_l77_77690

theorem reflected_ray_eqn : 
  ∃ a b c : ℝ, (∀ x y : ℝ, 2 * x - y + 5 = 0 → (a * x + b * y + c = 0)) → -- Condition for the line
  (∀ x y : ℝ, x = 1 ∧ y = 3 → (a * x + b * y + c = 0)) → -- Condition for point (1, 3)
  (a = 1 ∧ b = -5 ∧ c = 14) := -- Assertion about the line equation
by
  sorry

end reflected_ray_eqn_l77_77690


namespace product_of_ab_l77_77073

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77073


namespace density_is_not_vector_l77_77323

/-- Conditions definition -/
def is_vector (quantity : String) : Prop :=
quantity = "Buoyancy" ∨ quantity = "Wind speed" ∨ quantity = "Displacement"

/-- Problem statement -/
theorem density_is_not_vector : ¬ is_vector "Density" := 
by 
sorry

end density_is_not_vector_l77_77323


namespace number_of_sides_l77_77591

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l77_77591


namespace simplify_and_evaluate_expression_l77_77798

theorem simplify_and_evaluate_expression :
  ∀ x : ℤ, -1 ≤ x ∧ x ≤ 2 →
  (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2) →
  ( ( (x^2 - 1) / (x^2 - 2*x + 1) + ((x^2 - 2*x) / (x - 2)) / x ) = 1 ) :=
by
  intros x hx_constraints x_ne_criteria
  sorry

end simplify_and_evaluate_expression_l77_77798


namespace parabolas_pass_through_origin_l77_77853

-- Definition of a family of parabolas
def parabola_family (p q : ℝ) (x : ℝ) : ℝ := -x^2 + p * x + q

-- Definition of vertices lying on y = x^2
def vertex_condition (p q : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = -a^2 + p * a + q)

-- Proving that all such parabolas pass through the point (0, 0)
theorem parabolas_pass_through_origin :
  ∀ (p q : ℝ), vertex_condition p q → parabola_family p q 0 = 0 :=
by
  sorry

end parabolas_pass_through_origin_l77_77853


namespace largest_three_digit_multiple_of_17_l77_77221

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77221


namespace find_C_l77_77802

theorem find_C (A B C : ℕ) (h1 : (8 + 4 + A + 7 + 3 + B + 2) % 3 = 0)
  (h2 : (5 + 2 + 9 + A + B + 4 + C) % 3 = 0) : C = 2 :=
by
  sorry

end find_C_l77_77802


namespace greatest_three_digit_multiple_of_17_l77_77202

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77202


namespace product_of_ab_l77_77074

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l77_77074


namespace find_angle_C_l77_77532

-- Definitions based on conditions
variables (α β γ : ℝ) -- Angles of the triangle

-- Condition: Angles between the altitude and the angle bisector at vertices A and B are equal
-- This implies α = β
def angles_equal (α β : ℝ) : Prop :=
  α = β

-- Condition: Sum of the angles in a triangle is 180 degrees
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Condition: Angle at vertex C is greater than angles at vertices A and B
def c_greater_than_a_and_b (α γ : ℝ) : Prop :=
  γ > α

-- The proof problem: Prove γ = 120 degrees given the conditions
theorem find_angle_C (α β γ : ℝ) (h1 : angles_equal α β) (h2 : angles_sum_to_180 α β γ) (h3 : c_greater_than_a_and_b α γ) : γ = 120 :=
by
  sorry

end find_angle_C_l77_77532


namespace expected_value_coins_basilio_l77_77729

noncomputable def expected_coins_basilio (n : ℕ) (p : ℚ) : ℚ :=
  let X_Y := Binomial n p
  let E_X_Y := (n * p : ℚ)
  let X_minus_Y := 0.5
  (E_X_Y + X_minus_Y) / 2

theorem expected_value_coins_basilio :
  expected_coins_basilio 20 (1/2) = 5.25 := by
  sorry

end expected_value_coins_basilio_l77_77729


namespace negation_of_p_l77_77884

variable (f : ℝ → ℝ)

theorem negation_of_p :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔ (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
by
  sorry

end negation_of_p_l77_77884


namespace greatest_x_for_lcm_l77_77421

theorem greatest_x_for_lcm (x : ℕ) (h_lcm : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
by
  sorry

end greatest_x_for_lcm_l77_77421


namespace isosceles_triangle_top_angle_l77_77027

theorem isosceles_triangle_top_angle (A B C : Type) [triangle A B C] (isosceles : is_isosceles_triangle A B C) (angle_A : ∠A = 40) : 
∠top_angle(A B C) = 40 ∨ ∠top_angle(A B C) = 100 :=
begin
  sorry
end

end isosceles_triangle_top_angle_l77_77027


namespace ab_equals_6_l77_77105

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77105


namespace lines_non_intersect_l77_77708

theorem lines_non_intersect (k : ℝ) : 
  (¬∃ t s : ℝ, (1 + 2 * t = -1 + 3 * s ∧ 3 - 5 * t = 4 + k * s)) → 
  k = -15 / 2 :=
by
  intro h
  -- Now left to define proving steps using sorry
  sorry

end lines_non_intersect_l77_77708


namespace y_work_days_eq_10_l77_77831

noncomputable def work_days_y (W d : ℝ) : Prop :=
  let work_rate_x := W / 30
  let work_rate_y := W / 15
  let days_x_remaining := 10.000000000000002
  let work_done_by_y := d * work_rate_y
  let work_done_by_x := days_x_remaining * work_rate_x
  work_done_by_y + work_done_by_x = W

/-- The number of days y worked before leaving the job is 10 -/
theorem y_work_days_eq_10 (W : ℝ) : work_days_y W 10 :=
by
  sorry

end y_work_days_eq_10_l77_77831


namespace greatest_three_digit_multiple_of_17_l77_77228

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l77_77228


namespace art_piece_increase_is_correct_l77_77960

-- Define the conditions
def initial_price : ℝ := 4000
def future_multiplier : ℝ := 3
def future_price : ℝ := future_multiplier * initial_price

-- Define the goal
-- Proof that the increase in price is equal to $8000
theorem art_piece_increase_is_correct : future_price - initial_price = 8000 := 
by {
  -- We put sorry here to skip the actual proof
  sorry
}

end art_piece_increase_is_correct_l77_77960


namespace angle_sum_around_point_l77_77458

theorem angle_sum_around_point (x : ℝ) (h : 2 * x + 140 = 360) : x = 110 := 
  sorry

end angle_sum_around_point_l77_77458


namespace correct_operation_l77_77301

variable {R : Type*} [CommRing R] (x y : R)

theorem correct_operation : x * (1 + y) = x + x * y :=
by sorry

end correct_operation_l77_77301


namespace eval_expression_l77_77954

theorem eval_expression : 8 - (6 / (4 - 2)) = 5 := 
sorry

end eval_expression_l77_77954


namespace ab_value_l77_77092

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77092


namespace incorrect_number_read_l77_77946

theorem incorrect_number_read (incorrect_avg correct_avg : ℕ) (n correct_number incorrect_sum correct_sum : ℕ)
  (h1 : incorrect_avg = 17)
  (h2 : correct_avg = 20)
  (h3 : n = 10)
  (h4 : correct_number = 56)
  (h5 : incorrect_sum = n * incorrect_avg)
  (h6 : correct_sum = n * correct_avg)
  (h7 : correct_sum - incorrect_sum = correct_number - X) :
  X = 26 :=
by
  sorry

end incorrect_number_read_l77_77946


namespace inequality_solution_l77_77963

theorem inequality_solution (x : ℝ) (h : 1 / (x - 2) < 4) : x < 2 ∨ x > 9 / 4 :=
sorry

end inequality_solution_l77_77963


namespace greatest_three_digit_multiple_of_17_l77_77204

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l77_77204


namespace greatest_3_digit_multiple_of_17_l77_77293

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77293


namespace expected_coins_basilio_per_day_l77_77717

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end expected_coins_basilio_per_day_l77_77717


namespace ab_eq_six_l77_77084

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77084


namespace pyramid_z_value_l77_77912

-- Define the conditions and the proof problem
theorem pyramid_z_value {z x y : ℕ} :
  (x = z * y) →
  (8 = z * x) →
  (40 = x * y) →
  (10 = y * x) →
  z = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end pyramid_z_value_l77_77912


namespace age_difference_proof_l77_77307

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end age_difference_proof_l77_77307


namespace valid_pairs_iff_l77_77499

noncomputable def valid_pairs (a b : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ a * (⌊ b * n ⌋ : ℝ) = b * (⌊ a * n ⌋ : ℝ)

theorem valid_pairs_iff (a b : ℝ) : valid_pairs a b ↔
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (m n : ℤ), a = m ∧ b = n)) :=
by sorry

end valid_pairs_iff_l77_77499


namespace find_x_l77_77733

theorem find_x (x : ℝ) : |2 * x - 6| = 3 * x + 1 ↔ x = 1 := 
by 
  sorry

end find_x_l77_77733


namespace avg_chem_math_l77_77185

-- Given conditions
variables (P C M : ℕ)
axiom total_marks : P + C + M = P + 130

-- The proof problem
theorem avg_chem_math : (C + M) / 2 = 65 :=
by sorry

end avg_chem_math_l77_77185


namespace three_minus_pi_to_zero_l77_77702

theorem three_minus_pi_to_zero : (3 - Real.pi) ^ 0 = 1 := by
  -- proof goes here
  sorry

end three_minus_pi_to_zero_l77_77702


namespace greatest_three_digit_multiple_of_17_l77_77258

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77258


namespace regular_hexagon_area_l77_77453

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l77_77453


namespace complex_fraction_value_l77_77821

theorem complex_fraction_value :
  1 + (1 / (2 + (1 / (2 + 2)))) = 13 / 9 :=
by
  sorry

end complex_fraction_value_l77_77821


namespace greatest_three_digit_multiple_of_17_l77_77262

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l77_77262


namespace ab_eq_six_l77_77079

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77079


namespace max_a_if_monotonically_increasing_l77_77173

noncomputable def f (x a : ℝ) : ℝ := x^3 + Real.exp x - a * x

theorem max_a_if_monotonically_increasing (a : ℝ) : 
  (∀ x, 0 ≤ x → 3 * x^2 + Real.exp x - a ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end max_a_if_monotonically_increasing_l77_77173


namespace paco_salty_cookies_left_l77_77407

-- Define the initial number of salty cookies Paco had
def initial_salty_cookies : ℕ := 26

-- Define the number of salty cookies Paco ate
def eaten_salty_cookies : ℕ := 9

-- The theorem statement that Paco had 17 salty cookies left
theorem paco_salty_cookies_left : initial_salty_cookies - eaten_salty_cookies = 17 := 
 by
  -- Here we skip the proof by adding sorry
  sorry

end paco_salty_cookies_left_l77_77407


namespace Donovan_Mitchell_current_average_l77_77335

theorem Donovan_Mitchell_current_average 
    (points_per_game_goal : ℕ) 
    (games_played : ℕ) 
    (total_games_goal : ℕ) 
    (average_needed_remaining_games : ℕ)
    (points_needed : ℕ) 
    (remaining_games : ℕ) 
    (x : ℕ) 
    (h₁ : games_played = 15) 
    (h₂ : total_games_goal = 20) 
    (h₃ : points_per_game_goal = 30) 
    (h₄ : remaining_games = total_games_goal - games_played)
    (h₅ : average_needed_remaining_games = 42) 
    (h₆ : points_needed = remaining_games * average_needed_remaining_games) 
    (h₇ : points_needed = 210)  
    (h₈ : points_per_game_goal * total_games_goal = 600) 
    (h₉ : games_played * x + points_needed = 600) : 
    x = 26 :=
by {
  sorry
}

end Donovan_Mitchell_current_average_l77_77335


namespace ineq_one_of_two_sqrt_amgm_l77_77312

-- Lean 4 statement for Question 1
theorem ineq_one_of_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

-- Lean 4 statement for Question 2
theorem sqrt_amgm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
sorry

end ineq_one_of_two_sqrt_amgm_l77_77312


namespace jenny_hours_left_l77_77611

theorem jenny_hours_left 
    (h_research : ℕ := 10)
    (h_proposal : ℕ := 2)
    (h_visual_aids : ℕ := 5)
    (h_editing : ℕ := 3)
    (h_total : ℕ := 25) :
    h_total - (h_research + h_proposal + h_visual_aids + h_editing) = 5 := by
  sorry

end jenny_hours_left_l77_77611


namespace exists_x_in_interval_iff_m_lt_3_l77_77900

theorem exists_x_in_interval_iff_m_lt_3 (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2 * x > m) ↔ m < 3 :=
by
  sorry

end exists_x_in_interval_iff_m_lt_3_l77_77900


namespace original_number_l77_77662

theorem original_number (x y : ℕ) (h1 : x + y = 859560) (h2 : y = 859560 % 456) : x = 859376 ∧ 456 ∣ x :=
by
  sorry

end original_number_l77_77662


namespace find_range_f_l77_77924

noncomputable def greatestIntegerLessEqual (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def f (x y : ℝ) : ℝ :=
  (x + y) / (greatestIntegerLessEqual x * greatestIntegerLessEqual y + greatestIntegerLessEqual x + greatestIntegerLessEqual y + 1)

theorem find_range_f (x y : ℝ) (h1: 0 < x) (h2: 0 < y) (h3: x * y = 1) : 
  ∃ r : ℝ, r = f x y := 
by
  sorry

end find_range_f_l77_77924


namespace quadratic_inequality_l77_77760

theorem quadratic_inequality (a : ℝ) 
  (x₁ x₂ : ℝ) (h_roots : ∀ x, x^2 + (3 * a - 1) * x + a + 8 = 0) 
  (h_distinct : x₁ ≠ x₂)
  (h_x1_lt_1 : x₁ < 1) (h_x2_gt_1 : x₂ > 1) : 
  a < -2 := 
by
  sorry

end quadratic_inequality_l77_77760


namespace ab_equals_6_l77_77100

variable (a b : ℝ)
theorem ab_equals_6 (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_equals_6_l77_77100


namespace smallest_n_l77_77457

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def meets_condition (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ l : ℕ, 1 ≤ l ∧ l ≤ n + 1 ∧ ¬ is_divisible (n^2 - n + 1) l

theorem smallest_n : ∃ n : ℕ, meets_condition n ∧ n = 5 :=
by
  sorry

end smallest_n_l77_77457


namespace arithmetic_sequence_sum_l77_77544

theorem arithmetic_sequence_sum :
  ∀ {a : ℕ → ℕ} {S : ℕ → ℕ},
  (∀ n, a (n + 1) - a n = a 1 - a 0) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 1 + a 9 = 18 →
  a 4 = 7 →
  S 8 = 64 :=
by
  intros a S h_arith_seq h_sum_formula h_a1_a9 h_a4
  sorry

end arithmetic_sequence_sum_l77_77544


namespace greatest_three_digit_multiple_of_17_l77_77209

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l77_77209


namespace solution_set_ineq_l77_77427

theorem solution_set_ineq (x : ℝ) : (1 < x ∧ x ≤ 3) ↔ (x - 3) / (x - 1) ≤ 0 := sorry

end solution_set_ineq_l77_77427


namespace sum_of_digits_is_32_l77_77673

/-- 
Prove that the sum of digits \( A, B, C, D, E \) is 32 given the constraints
1. \( A, B, C, D, E \) are single digits.
2. The sum of the units column 3E results in 1 (units place of 2011).
3. The sum of the hundreds column 3A and carry equals 20 (hundreds place of 2011).
-/
theorem sum_of_digits_is_32
  (A B C D E : ℕ)
  (h1 : A < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : D < 10)
  (h5 : E < 10)
  (units_condition : 3 * E % 10 = 1)
  (hundreds_condition : ∃ carry: ℕ, carry < 10 ∧ 3 * A + carry = 20) :
  A + B + C + D + E = 32 := 
sorry

end sum_of_digits_is_32_l77_77673


namespace ab_eq_six_l77_77077

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l77_77077


namespace Winnie_lollipops_remain_l77_77967

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

end Winnie_lollipops_remain_l77_77967


namespace winning_strategy_for_B_l77_77685

theorem winning_strategy_for_B (N : ℕ) (h : N < 15) : N = 7 ↔ (∃ strategy : (Fin 6 → ℕ) → ℕ, ∀ f : Fin 6 → ℕ, (strategy f) % 1001 = 0) :=
by
  sorry

end winning_strategy_for_B_l77_77685


namespace regular_hexagon_area_inscribed_in_circle_l77_77451

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l77_77451


namespace portion_spent_in_second_store_l77_77786

theorem portion_spent_in_second_store (M : ℕ) (X : ℕ) (H : M = 180)
  (H1 : M - (M / 2 + 14) = 76)
  (H2 : X + 16 = 76)
  (H3 : M = (M / 2 + 14) + (X + 16)) :
  (X : ℚ) / M = 1 / 3 :=
by 
  sorry

end portion_spent_in_second_store_l77_77786


namespace find_probability_greater_than_three_l77_77552

noncomputable def X : ℝ := Binomial 4 (1 / 4)

noncomputable def Y (μ σ : ℝ) : Measure ℝ := Normal μ σ^2

axiom E_X (X : ℝ) : E[X] = 1
axiom E_Y (Y : ℝ) (μ : ℝ) : E[Y] = μ

axiom P_abs_Y_lt_1 (Y : ℝ) (σ : ℝ) : P (|Y| < 1) = 0.4

theorem find_probability_greater_than_three (μ σ : ℝ) (hσ : σ > 0) :
  (∃ μ, E[X] = E[Y μ σ] ∧ P_abs_Y_lt_1 (Y μ σ) σ) → P (Y μ σ > 3) = 0.1 :=
by
  sorry

end find_probability_greater_than_three_l77_77552


namespace largest_three_digit_multiple_of_17_l77_77220

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l77_77220


namespace none_of_these_are_perfect_squares_l77_77300

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

theorem none_of_these_are_perfect_squares :
  ¬ is_perfect_square (19! * 20! / 2) ∧
  ¬ is_perfect_square (20! * 21! / 2) ∧
  ¬ is_perfect_square (21! * 22! / 2) ∧
  ¬ is_perfect_square (22! * 23! / 2) ∧
  ¬ is_perfect_square (23! * 24! / 2) :=
by
  sorry

end none_of_these_are_perfect_squares_l77_77300


namespace radius_of_circle_l77_77022

variables (O P A B : Type) [MetricSpace O] [MetricSpace P] [MetricSpace A] [MetricSpace B]
variables (circle_radius : ℝ) (PA PB OP : ℝ)

theorem radius_of_circle
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  (circle_radius : ℝ)
  : circle_radius = 7 :=
by sorry

end radius_of_circle_l77_77022


namespace greatest_three_digit_multiple_of_17_l77_77245

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77245


namespace solve_for_x_l77_77334

noncomputable def solution_x : ℝ := -1011.5

theorem solve_for_x (x : ℝ) (h : (2023 + x)^2 = x^2) : x = solution_x :=
by sorry

end solve_for_x_l77_77334


namespace probability_less_than_one_third_l77_77575

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l77_77575


namespace smallest_degree_of_f_l77_77466

theorem smallest_degree_of_f (p : Polynomial ℂ) (hp_deg : p.degree < 1992)
  (hp0 : p.eval 0 ≠ 0) (hp1 : p.eval 1 ≠ 0) (hp_1 : p.eval (-1) ≠ 0) :
  ∃ f g : Polynomial ℂ, 
    (Polynomial.derivative^[1992] (p / (X^3 - X))) = f / g ∧ f.degree = 3984 := 
sorry

end smallest_degree_of_f_l77_77466


namespace exterior_angle_of_regular_polygon_l77_77593

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l77_77593


namespace minimum_value_of_expression_l77_77839

theorem minimum_value_of_expression (a b : ℝ) (h : 1 / a + 2 / b = 1) : 4 * a^2 + b^2 ≥ 32 :=
by sorry

end minimum_value_of_expression_l77_77839


namespace greatest_three_digit_multiple_of_17_l77_77252

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l77_77252


namespace highest_daily_profit_and_total_profit_l77_77465

def cost_price : ℕ := 6
def standard_price : ℕ := 10

def price_relative (day : ℕ) : ℤ := 
  match day with
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => -1
  | 5 => -2
  | _ => 0

def quantity_sold (day : ℕ) : ℕ :=
  match day with
  | 1 => 7
  | 2 => 12
  | 3 => 15
  | 4 => 32
  | 5 => 34
  | _ => 0

noncomputable def selling_price (day : ℕ) : ℤ := standard_price + price_relative day

noncomputable def profit_per_pen (day : ℕ) : ℤ := (selling_price day) - cost_price

noncomputable def daily_profit (day : ℕ) : ℤ := (profit_per_pen day) * (quantity_sold day)

theorem highest_daily_profit_and_total_profit 
  (h_highest_profit: daily_profit 4 = 96) 
  (h_total_profit: daily_profit 1 + daily_profit 2 + daily_profit 3 + daily_profit 4 + daily_profit 5 = 360) : 
  True :=
by
  sorry

end highest_daily_profit_and_total_profit_l77_77465


namespace unique_solution_for_a_l77_77867

def system_has_unique_solution (a : ℝ) (x y : ℝ) : Prop :=
(x^2 + y^2 + 2 * x ≤ 1) ∧ (x - y + a = 0)

theorem unique_solution_for_a (a x y : ℝ) :
  (system_has_unique_solution 3 x y ∨ system_has_unique_solution (-1) x y)
  ∧ (((a = 3) → (x, y) = (-2, 1)) ∨ ((a = -1) → (x, y) = (0, -1))) :=
sorry

end unique_solution_for_a_l77_77867


namespace greatest_3_digit_multiple_of_17_l77_77290

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l77_77290


namespace total_kids_attended_camp_l77_77336

theorem total_kids_attended_camp :
  let n1 := 34044
  let n2 := 424944
  n1 + n2 = 458988 := 
by {
  sorry
}

end total_kids_attended_camp_l77_77336


namespace problem_statement_l77_77149

variable {Point Line Plane : Type}

-- Definitions for perpendicular and parallel
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def perp_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given variables
variable (a b c d : Line) (α β : Plane)

-- Conditions
axiom a_perp_b : perpendicular a b
axiom c_perp_d : perpendicular c d
axiom a_perp_alpha : perp_to_plane a α
axiom c_perp_alpha : perp_to_plane c α

-- Required proof
theorem problem_statement : perpendicular c b :=
by sorry

end problem_statement_l77_77149


namespace perimeter_of_square_B_l77_77634

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_B_l77_77634


namespace product_ab_l77_77134

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l77_77134


namespace spend_on_candy_l77_77935

variable (initial_money spent_on_oranges spent_on_apples remaining_money spent_on_candy : ℕ)

-- Conditions
axiom initial_amount : initial_money = 95
axiom spent_on_oranges_value : spent_on_oranges = 14
axiom spent_on_apples_value : spent_on_apples = 25
axiom remaining_amount : remaining_money = 50

-- Question as a theorem
theorem spend_on_candy :
  spent_on_candy = initial_money - (spent_on_oranges + spent_on_apples) - remaining_money :=
by sorry

end spend_on_candy_l77_77935


namespace range_of_t_l77_77358

noncomputable def a_n (t : ℝ) (n : ℕ) : ℝ :=
  if n > 8 then ((1 / 3) - t) * (n:ℝ) + 2 else t ^ (n - 7)

theorem range_of_t (t : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → a_n t n > a_n t (n + 1)) →
  (1/2 < t ∧ t < 1) :=
by
  intros h
  -- The proof would go here.
  sorry

end range_of_t_l77_77358


namespace ab_equals_six_l77_77130

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77130


namespace geometric_sequence_sum_8_l77_77875

variable {a : ℝ} 

-- conditions
def geometric_series_sum_4 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3

def geometric_series_sum_8 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 + a * r^6 + a * r^7

theorem geometric_sequence_sum_8 (r : ℝ) (S4 : ℝ) (S8 : ℝ) (hr : r = 2) (hS4 : S4 = 1) :
  (∃ a : ℝ, geometric_series_sum_4 r a = S4 ∧ geometric_series_sum_8 r a = S8) → S8 = 17 :=
by
  sorry

end geometric_sequence_sum_8_l77_77875


namespace number_of_boys_in_class_l77_77945

theorem number_of_boys_in_class (n : ℕ)
  (avg_height : ℕ) (incorrect_height : ℕ) (actual_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : avg_height = 180)
  (h2 : incorrect_height = 156)
  (h3 : actual_height = 106)
  (h4 : actual_avg_height = 178)
  : n = 25 :=
by 
  -- We have the following conditions:
  -- Incorrect total height = avg_height * n
  -- Difference due to incorrect height = incorrect_height - actual_height
  -- Correct total height = avg_height * n - (incorrect_height - actual_height)
  -- Total height according to actual average = actual_avg_height * n
  -- Equating both, we have:
  -- avg_height * n - (incorrect_height - actual_height) = actual_avg_height * n
  -- We know avg_height, incorrect_height, actual_height, actual_avg_height from h1, h2, h3, h4
  -- Substituting these values and solving:
  -- 180n - (156 - 106) = 178n
  -- 180n - 50 = 178n
  -- 2n = 50
  -- n = 25
  sorry

end number_of_boys_in_class_l77_77945


namespace original_amount_of_money_l77_77497

-- Define the conditions
variables (x : ℕ) -- daily allowance

-- Spending details
def spend_10_days := 6 * 10 - 6 * x
def spend_15_days := 15 * 3 - 3 * x

-- Lean proof statement
theorem original_amount_of_money (h : spend_10_days = spend_15_days) : (6 * 10 - 6 * x) = 30 :=
by
  sorry

end original_amount_of_money_l77_77497


namespace journey_length_25_km_l77_77822

theorem journey_length_25_km:
  ∀ (D T : ℝ),
  (D = 100 * T) →
  (D = 50 * (T + 15/60)) →
  D = 25 :=
by
  intros D T h1 h2
  sorry

end journey_length_25_km_l77_77822


namespace dubblefud_red_balls_l77_77911

theorem dubblefud_red_balls (R B G : ℕ) 
  (h1 : 3^R * 7^B * 11^G = 5764801)
  (h2 : B = G) :
  R = 7 :=
by
  sorry

end dubblefud_red_balls_l77_77911


namespace find_x_l77_77557
-- The first priority is to ensure the generated Lean code can be built successfully.

theorem find_x (x : ℤ) (h : 9823 + x = 13200) : x = 3377 :=
by
  sorry

end find_x_l77_77557


namespace length_of_AD_l77_77380

theorem length_of_AD (AB BC CD DE : ℝ) (right_angle_B right_angle_C : Prop) :
  AB = 6 → BC = 7 → CD = 25 → DE = 15 → AD = Real.sqrt 274 :=
by
  intros
  sorry

end length_of_AD_l77_77380


namespace smallest_four_digit_multiple_of_18_l77_77526

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l77_77526


namespace probability_of_interval_l77_77570

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l77_77570


namespace probability_intervals_l77_77582

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l77_77582


namespace calculate_gfg3_l77_77918

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem calculate_gfg3 : g (f (g 3)) = 192 := by
  sorry

end calculate_gfg3_l77_77918


namespace number_of_white_tiles_l77_77701

theorem number_of_white_tiles (n : ℕ) : 
  ∃ a_n : ℕ, a_n = 4 * n + 2 :=
sorry

end number_of_white_tiles_l77_77701


namespace ab_value_l77_77097

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l77_77097


namespace calc_expression_l77_77329

theorem calc_expression : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end calc_expression_l77_77329


namespace smallest_four_digit_multiple_of_18_l77_77524

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l77_77524


namespace total_possible_match_sequences_l77_77944

theorem total_possible_match_sequences :
  let num_teams := 2
  let team_size := 7
  let possible_sequences := 2 * (Nat.choose (2 * team_size - 1) (team_size - 1))
  possible_sequences = 3432 :=
by
  sorry

end total_possible_match_sequences_l77_77944


namespace ab_equals_six_l77_77055

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77055


namespace min_xy_value_l77_77545

theorem min_xy_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (hlog : Real.log x / Real.log 2 * Real.log y / Real.log 2 = 1) : x * y = 4 :=
by sorry

end min_xy_value_l77_77545


namespace greatest_three_digit_multiple_of_17_l77_77246

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l77_77246


namespace ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l77_77310

-- Problem 1
theorem ab_eq_6_pos_or_neg (a b : ℚ) (h : a * b = 6) : a + b > 0 ∨ a + b < 0 := sorry

-- Problem 2
theorem max_ab_when_sum_neg5 (a b : ℤ) (h : a + b = -5) : a * b ≤ 6 := sorry

-- Problem 3
theorem ab_lt_0_sign_of_sum (a b : ℚ) (h : a * b < 0) : (a + b > 0 ∨ a + b = 0 ∨ a + b < 0) := sorry

end ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l77_77310


namespace complement_A_U_l77_77929

-- Define the universal set U and set A as given in the problem.
def U : Set ℕ := { x | x ≥ 3 }
def A : Set ℕ := { x | x * x ≥ 10 }

-- Prove that the complement of A with respect to U is {3}.
theorem complement_A_U :
  (U \ A) = {3} :=
by
  sorry

end complement_A_U_l77_77929


namespace complete_work_in_days_l77_77832

def rate_x : ℚ := 1 / 10
def rate_y : ℚ := 1 / 15
def rate_z : ℚ := 1 / 20

def combined_rate : ℚ := rate_x + rate_y + rate_z

theorem complete_work_in_days :
  1 / combined_rate = 60 / 13 :=
by
  -- Proof will go here
  sorry

end complete_work_in_days_l77_77832


namespace ab_equals_six_l77_77059

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l77_77059


namespace min_value_problem_l77_77153

theorem min_value_problem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 3 * b = 1) :
    (1 / a) + (3 / b) ≥ 16 :=
sorry

end min_value_problem_l77_77153


namespace smaller_angle_at_8_15_l77_77751

def angle_minute_hand_at_8_15: ℝ := 90
def angle_hour_hand_at_8: ℝ := 240
def additional_angle_hour_hand_at_8_15: ℝ := 7.5
def total_angle_hour_hand_at_8_15 := angle_hour_hand_at_8 + additional_angle_hour_hand_at_8_15

theorem smaller_angle_at_8_15 :
  min (abs (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))
      (abs (360 - (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))) = 157.5 :=
by
  sorry

end smaller_angle_at_8_15_l77_77751


namespace greatest_three_digit_multiple_of_17_l77_77194

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l77_77194


namespace greatest_three_digit_multiple_of_17_l77_77288

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l77_77288


namespace angles_equal_or_cofunctions_equal_l77_77795

def cofunction (θ : ℝ) : ℝ := sorry -- Define the co-function (e.g., sine and cosine)

theorem angles_equal_or_cofunctions_equal (θ₁ θ₂ : ℝ) :
  θ₁ = θ₂ ∨ cofunction θ₁ = cofunction θ₂ → θ₁ = θ₂ :=
sorry

end angles_equal_or_cofunctions_equal_l77_77795


namespace min_k_value_l77_77876

noncomputable def f (k x : ℝ) : ℝ := k * (x^2 - x + 1) - x^4 * (1 - x)^4

theorem min_k_value : ∃ k : ℝ, (k = 1 / 192) ∧ ∀ x : ℝ, (0 ≤ x) → (x ≤ 1) → (f k x ≥ 0) :=
by
  existsi (1 / 192)
  sorry

end min_k_value_l77_77876


namespace find_a_l77_77903

theorem find_a (a x : ℝ) (h1 : 3 * x + 5 = 11) (h2 : 6 * x + 3 * a = 22) : a = 10 / 3 :=
by
  -- the proof will go here
  sorry

end find_a_l77_77903


namespace parallel_lines_slope_l77_77501

theorem parallel_lines_slope (a : ℝ) :
  (∃ (a : ℝ), ∀ x y, (3 * y - a = 9 * x + 1) ∧ (y - 2 = (2 * a - 3) * x)) → a = 3 :=
by
  sorry

end parallel_lines_slope_l77_77501


namespace ab_equals_six_l77_77125

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l77_77125


namespace sufficient_but_not_necessary_l77_77749

noncomputable def p (m : ℝ) : Prop :=
  -6 ≤ m ∧ m ≤ 6

noncomputable def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 9 ≠ 0

theorem sufficient_but_not_necessary (m : ℝ) :
  (p m → q m) ∧ (q m → ¬ p m) :=
by
  sorry

end sufficient_but_not_necessary_l77_77749
