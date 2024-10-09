import Mathlib

namespace total_salmon_l2241_224118

def male_salmon : Nat := 712261
def female_salmon : Nat := 259378

theorem total_salmon :
  male_salmon + female_salmon = 971639 := by
  sorry

end total_salmon_l2241_224118


namespace modulus_of_z_l2241_224113

open Complex

theorem modulus_of_z (z : ℂ) (h : z^2 = (3/4 : ℝ) - I) : abs z = Real.sqrt 5 / 2 := 
  sorry

end modulus_of_z_l2241_224113


namespace find_m_l2241_224135

theorem find_m {m : ℝ} (a b : ℝ × ℝ) (H : a = (3, m) ∧ b = (2, -1)) (H_dot : a.1 * b.1 + a.2 * b.2 = 0) : m = 6 := 
by
  sorry

end find_m_l2241_224135


namespace no_girl_can_avoid_losing_bet_l2241_224197

theorem no_girl_can_avoid_losing_bet
  (G1 G2 G3 : Prop)
  (h1 : G1 ↔ ¬G2)
  (h2 : G2 ↔ ¬G3)
  (h3 : G3 ↔ ¬G1)
  : G1 ∧ G2 ∧ G3 → False := by
  sorry

end no_girl_can_avoid_losing_bet_l2241_224197


namespace relative_prime_in_consecutive_integers_l2241_224167

theorem relative_prime_in_consecutive_integers (n : ℤ) : 
  ∃ k, n ≤ k ∧ k ≤ n + 5 ∧ ∀ m, n ≤ m ∧ m ≤ n + 5 ∧ m ≠ k → Int.gcd k m = 1 :=
sorry

end relative_prime_in_consecutive_integers_l2241_224167


namespace benny_eggs_l2241_224145

theorem benny_eggs (dozen_count : ℕ) (eggs_per_dozen : ℕ) (total_eggs : ℕ) 
  (h1 : dozen_count = 7) 
  (h2 : eggs_per_dozen = 12) 
  (h3 : total_eggs = dozen_count * eggs_per_dozen) : 
  total_eggs = 84 := 
by 
  sorry

end benny_eggs_l2241_224145


namespace square_area_from_diagonal_l2241_224190

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end square_area_from_diagonal_l2241_224190


namespace point_distance_5_5_l2241_224132

-- Define the distance function in the context of the problem
def distance_from_origin (x : ℝ) : ℝ := abs x

-- Formalize the proposition
theorem point_distance_5_5 (x : ℝ) : distance_from_origin x = 5.5 → (x = -5.5 ∨ x = 5.5) :=
by
  intro h
  simp [distance_from_origin] at h
  sorry

end point_distance_5_5_l2241_224132


namespace scarves_per_box_l2241_224149

theorem scarves_per_box (S M : ℕ) (h1 : S = M) (h2 : 6 * (S + M) = 60) : S = 5 :=
by
  sorry

end scarves_per_box_l2241_224149


namespace workers_days_not_worked_l2241_224196

theorem workers_days_not_worked (W N : ℕ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 :=
sorry

end workers_days_not_worked_l2241_224196


namespace no_integer_solution_exists_l2241_224155

theorem no_integer_solution_exists :
  ¬ ∃ m n : ℤ, m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_integer_solution_exists_l2241_224155


namespace part1_part2_l2241_224192

def P (x : ℝ) : Prop := |x - 1| > 2
def S (x : ℝ) (a : ℝ) : Prop := x^2 - (a + 1) * x + a > 0

theorem part1 (a : ℝ) (h : a = 2) : ∀ x, S x a ↔ x < 1 ∨ x > 2 :=
by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 1) : ∀ x, (P x → S x a) → (-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l2241_224192


namespace jonah_total_lemonade_l2241_224173

theorem jonah_total_lemonade : 
  0.25 + 0.4166666666666667 + 0.25 + 0.5833333333333334 = 1.5 :=
by
  sorry

end jonah_total_lemonade_l2241_224173


namespace trapezoid_area_l2241_224121

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_l2241_224121


namespace truncated_cone_contact_radius_l2241_224144

theorem truncated_cone_contact_radius (R r r' ζ : ℝ)
  (h volume_condition : ℝ)
  (R_pos : 0 < R)
  (r_pos : 0 < r)
  (r'_pos : 0 < r')
  (ζ_pos : 0 < ζ)
  (h_eq : h = 2 * R)
  (volume_condition_eq :
    (2 : ℝ) * ((4 / 3) * Real.pi * R^3) = 
    (2 / 3) * Real.pi * h * (r^2 + r * r' + r'^2)) :
  ζ = (2 * R * Real.sqrt 5) / 5 :=
by
  sorry

end truncated_cone_contact_radius_l2241_224144


namespace solve_equation_l2241_224189

theorem solve_equation (x : ℝ) : 
  (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ↔ 
    (x = 1 ∨ x = -16 ∨ x = 4 ∨ x = -4) :=
by
  sorry

end solve_equation_l2241_224189


namespace initial_apples_l2241_224154

theorem initial_apples (C : ℝ) (h : C + 7.0 = 27) : C = 20.0 := by
  sorry

end initial_apples_l2241_224154


namespace tycho_jogging_schedule_count_l2241_224158

-- Definition of the conditions
def non_consecutive_shot_schedule (days : Finset ℕ) : Prop :=
  ∀ day ∈ days, ∀ next_day ∈ days, day < next_day → next_day - day > 1

-- Definition stating there are exactly seven valid schedules
theorem tycho_jogging_schedule_count :
  ∃ (S : Finset (Finset ℕ)), (∀ s ∈ S, s.card = 3 ∧ non_consecutive_shot_schedule s) ∧ S.card = 7 := 
sorry

end tycho_jogging_schedule_count_l2241_224158


namespace combined_avg_score_l2241_224146

theorem combined_avg_score (x : ℕ) : 
  let avgA := 65
  let avgB := 90 
  let avgC := 77 
  let ratioA := 4 
  let ratioB := 6 
  let ratioC := 5 
  let total_students := 15 * x 
  let total_score := (ratioA * avgA + ratioB * avgB + ratioC * avgC) * x
  (total_score / total_students) = 79 := 
by
  sorry

end combined_avg_score_l2241_224146


namespace last_integer_in_sequence_is_one_l2241_224151

theorem last_integer_in_sequence_is_one :
  ∀ seq : ℕ → ℕ, (seq 0 = 37) ∧ (∀ n, seq (n + 1) = seq n / 2) → (∃ n, seq (n + 1) = 0 ∧ seq n = 1) :=
by
  sorry

end last_integer_in_sequence_is_one_l2241_224151


namespace expenditure_ratio_l2241_224153

theorem expenditure_ratio 
  (I1 : ℝ) (I2 : ℝ) (E1 : ℝ) (E2 : ℝ) (S1 : ℝ) (S2 : ℝ)
  (h1 : I1 = 3500)
  (h2 : I2 = (4 / 5) * I1)
  (h3 : S1 = I1 - E1)
  (h4 : S2 = I2 - E2)
  (h5 : S1 = 1400)
  (h6 : S2 = 1400) : 
  E1 / E2 = 3 / 2 :=
by
  -- Steps of the proof will go here
  sorry

end expenditure_ratio_l2241_224153


namespace equivalent_expression_l2241_224150

noncomputable def problem_statement (α β γ δ p q : ℝ) :=
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (p^2 - q^2) + 4

theorem equivalent_expression
  (α β γ δ p q : ℝ)
  (h1 : ∀ x, x^2 + p * x + 2 = 0 → (x = α ∨ x = β))
  (h2 : ∀ x, x^2 + q * x + 2 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
by sorry

end equivalent_expression_l2241_224150


namespace unique_solution_in_z3_l2241_224127

theorem unique_solution_in_z3 (x y z : ℤ) (h : x^3 + 2 * y^3 = 4 * z^3) : 
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end unique_solution_in_z3_l2241_224127


namespace car_distance_problem_l2241_224131

-- A definition for the initial conditions.
def initial_conditions (D : ℝ) (S : ℝ) (T : ℝ) : Prop :=
  T = 6 ∧ S = 50 ∧ (3/2 * T = 9)

-- The statement corresponding to the given problem.
theorem car_distance_problem (D : ℝ) (S : ℝ) (T : ℝ) :
  initial_conditions D S T → D = 450 :=
by
  -- leave the proof as an exercise.
  sorry

end car_distance_problem_l2241_224131


namespace set_C_is_correct_l2241_224180

open Set

noncomputable def set_A : Set ℝ := {x | x ^ 2 - x - 12 ≤ 0}
noncomputable def set_B : Set ℝ := {x | (x + 1) / (x - 1) < 0}
noncomputable def set_C : Set ℝ := {x | x ∈ set_A ∧ x ∉ set_B}

theorem set_C_is_correct : set_C = {x | -3 ≤ x ∧ x ≤ -1} ∪ {x | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_C_is_correct_l2241_224180


namespace remainder_of_square_l2241_224160

theorem remainder_of_square (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := 
by 
  sorry

end remainder_of_square_l2241_224160


namespace intersection_A_B_l2241_224191

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_A_B_l2241_224191


namespace find_number_l2241_224134

theorem find_number (x : ℤ) (h : 72516 * x = 724797420) : x = 10001 :=
by
  sorry

end find_number_l2241_224134


namespace min_cost_proof_l2241_224101

-- Define the costs and servings for each ingredient
def pasta_cost : ℝ := 1.12
def pasta_servings_per_box : ℕ := 5

def meatballs_cost : ℝ := 5.24
def meatballs_servings_per_pack : ℕ := 4

def tomato_sauce_cost : ℝ := 2.31
def tomato_sauce_servings_per_jar : ℕ := 5

def tomatoes_cost : ℝ := 1.47
def tomatoes_servings_per_pack : ℕ := 4

def lettuce_cost : ℝ := 0.97
def lettuce_servings_per_head : ℕ := 6

def olives_cost : ℝ := 2.10
def olives_servings_per_jar : ℕ := 8

def cheese_cost : ℝ := 2.70
def cheese_servings_per_block : ℕ := 7

-- Define the number of people to serve
def number_of_people : ℕ := 8

-- The total cost calculated
def total_cost : ℝ := 
  (2 * pasta_cost) +
  (2 * meatballs_cost) +
  (2 * tomato_sauce_cost) +
  (2 * tomatoes_cost) +
  (2 * lettuce_cost) +
  (1 * olives_cost) +
  (2 * cheese_cost)

-- The minimum total cost
def min_total_cost : ℝ := 29.72

theorem min_cost_proof : total_cost = min_total_cost :=
by sorry

end min_cost_proof_l2241_224101


namespace find_x_for_equation_l2241_224163

theorem find_x_for_equation : ∃ x : ℝ, (1 / 2) + ((2 / 3) * x + 4) - (8 / 16) = 4.25 ↔ x = 0.375 := 
by
  sorry

end find_x_for_equation_l2241_224163


namespace sum_of_center_coords_l2241_224188

theorem sum_of_center_coords (x y : ℝ) :
  (∃ k : ℝ, (x + 2)^2 + (y + 3)^2 = k ∧ (x^2 + y^2 = -4 * x - 6 * y + 5)) -> x + y = -5 :=
by
sorry

end sum_of_center_coords_l2241_224188


namespace number_of_girls_in_basketball_club_l2241_224114

-- Define the number of members in the basketball club
def total_members : ℕ := 30

-- Define the number of members who attended the practice session
def attended : ℕ := 18

-- Define the unknowns: number of boys (B) and number of girls (G)
variables (B G : ℕ)

-- Define the conditions provided in the problem
def condition1 : Prop := B + G = total_members
def condition2 : Prop := B + (1 / 3) * G = attended

-- Define the theorem to prove
theorem number_of_girls_in_basketball_club (B G : ℕ) (h1 : condition1 B G) (h2 : condition2 B G) : G = 18 :=
sorry

end number_of_girls_in_basketball_club_l2241_224114


namespace eighth_grade_girls_l2241_224108

theorem eighth_grade_girls
  (G : ℕ) 
  (boys : ℕ) 
  (h1 : boys = 2 * G - 16) 
  (h2 : G + boys = 68) : 
  G = 28 :=
by
  sorry

end eighth_grade_girls_l2241_224108


namespace range_of_values_abs_range_of_values_l2241_224104

noncomputable def problem (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1

theorem range_of_values (x y : ℝ) (h : problem x y) :
  2 ≤ (2 * x + y - 1) / x ∧ (2 * x + y - 1) / x ≤ 10 / 3 :=
sorry

theorem abs_range_of_values (x y : ℝ) (h : problem x y) :
  5 - Real.sqrt 2 ≤ abs (x + y + 1) ∧ abs (x + y + 1) ≤ 5 + Real.sqrt 2 :=
sorry

end range_of_values_abs_range_of_values_l2241_224104


namespace value_of_b_l2241_224109

theorem value_of_b (b x : ℝ) (h1 : 2 * x + 7 = 3) (h2 : b * x - 10 = -2) : b = -4 :=
by
  sorry

end value_of_b_l2241_224109


namespace side_length_of_square_l2241_224130

theorem side_length_of_square :
  ∃ n : ℝ, n^2 = 9/16 ∧ n = 3/4 :=
sorry

end side_length_of_square_l2241_224130


namespace complex_powers_sum_zero_l2241_224162

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end complex_powers_sum_zero_l2241_224162


namespace eccentricity_of_ellipse_l2241_224141

variable (a b c d1 d2 : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (h4 : 2 * c = (d1 + d2) / 2)
variable (h5 : d1 + d2 = 2 * a)

theorem eccentricity_of_ellipse : (c / a) = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l2241_224141


namespace smallest_common_multiple_of_8_and_6_l2241_224139

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ, n > 0 ∧ (8 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (8 ∣ m) ∧ (6 ∣ m)) → n ≤ m :=
by
  sorry

end smallest_common_multiple_of_8_and_6_l2241_224139


namespace pen_cost_l2241_224100

def pencil_cost : ℝ := 1.60
def elizabeth_money : ℝ := 20.00
def num_pencils : ℕ := 5
def num_pens : ℕ := 6

theorem pen_cost (pen_cost : ℝ) : 
  elizabeth_money - (num_pencils * pencil_cost) = num_pens * pen_cost → 
  pen_cost = 2 :=
by 
  sorry

end pen_cost_l2241_224100


namespace doubles_tournament_handshakes_l2241_224133

theorem doubles_tournament_handshakes :
  let num_teams := 3
  let players_per_team := 2
  let total_players := num_teams * players_per_team
  let handshakes_per_player := total_players - 2
  let total_handshakes := total_players * handshakes_per_player / 2
  total_handshakes = 12 :=
by
  sorry

end doubles_tournament_handshakes_l2241_224133


namespace product_plus_one_is_square_l2241_224106

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : x * y + 1 = (x + 1) ^ 2 :=
by
  sorry

end product_plus_one_is_square_l2241_224106


namespace sides_of_length_five_l2241_224123

theorem sides_of_length_five (GH HI : ℝ) (L : ℝ) (total_perimeter : ℝ) :
  GH = 7 → HI = 5 → total_perimeter = 38 → (∃ n m : ℕ, n + m = 6 ∧ n * 7 + m * 5 = 38 ∧ m = 2) := by
  intros hGH hHI hPerimeter
  sorry

end sides_of_length_five_l2241_224123


namespace num_true_propositions_eq_two_l2241_224111

open Classical

theorem num_true_propositions_eq_two (p q : Prop) :
  (if (p ∧ q) then 1 else 0) + (if (p ∨ q) then 1 else 0) + (if (¬p) then 1 else 0) + (if (¬q) then 1 else 0) = 2 :=
by sorry

end num_true_propositions_eq_two_l2241_224111


namespace find_a_l2241_224117

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  x * (x + 1)
else
  -((-x) * ((-x) + 1))

theorem find_a (a : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, x >= 0 → f x = x * (x + 1)) (h_a: f a = -2) : a = -1 :=
sorry

end find_a_l2241_224117


namespace cookie_division_l2241_224194

theorem cookie_division (C : ℝ) (blue_fraction : ℝ := 1/4) (green_fraction_of_remaining : ℝ := 5/9)
  (remaining_fraction : ℝ := 3/4) (green_fraction : ℝ := 5/12) :
  blue_fraction + green_fraction = 2/3 := by
  sorry

end cookie_division_l2241_224194


namespace total_cost_proof_l2241_224182

-- Define the conditions
def length_grass_field : ℝ := 75
def width_grass_field : ℝ := 55
def width_path : ℝ := 2.5
def area_path : ℝ := 6750
def cost_per_sq_m : ℝ := 10

-- Calculate the outer dimensions
def outer_length : ℝ := length_grass_field + 2 * width_path
def outer_width : ℝ := width_grass_field + 2 * width_path

-- Calculate the area of the entire field including the path
def area_entire_field : ℝ := outer_length * outer_width

-- Calculate the area of the grass field without the path
def area_grass_field : ℝ := length_grass_field * width_grass_field

-- Calculate the area of the path
def area_calculated_path : ℝ := area_entire_field - area_grass_field

-- Calculate the total cost of constructing the path
noncomputable def total_cost : ℝ := area_calculated_path * cost_per_sq_m

-- The theorem to prove
theorem total_cost_proof :
  area_calculated_path = area_path ∧ total_cost = 6750 :=
by
  sorry

end total_cost_proof_l2241_224182


namespace angle_I_measure_l2241_224187

theorem angle_I_measure {x y : ℝ} 
  (h1 : x = y - 50) 
  (h2 : 3 * x + 2 * y = 540)
  : y = 138 := 
by 
  sorry

end angle_I_measure_l2241_224187


namespace sqrt_of_0_01_l2241_224195

theorem sqrt_of_0_01 : Real.sqrt 0.01 = 0.1 :=
by
  sorry

end sqrt_of_0_01_l2241_224195


namespace books_count_l2241_224143

theorem books_count (books_per_box : ℕ) (boxes : ℕ) (total_books : ℕ) 
  (h1 : books_per_box = 3)
  (h2 : boxes = 8)
  (h3 : total_books = books_per_box * boxes) : 
  total_books = 24 := 
by 
  rw [h1, h2] at h3
  exact h3

end books_count_l2241_224143


namespace range_of_a_l2241_224198

theorem range_of_a : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), ((a^2 - 1) * x^2 + (a + 1) * x + 1) > 0) → 1 ≤ a ∧ a ≤ 5 / 3 := 
by
  sorry

end range_of_a_l2241_224198


namespace find_x_set_l2241_224179

theorem find_x_set (x : ℝ) : ((x - 2) ^ 2 < 3 * x + 4) ↔ (0 ≤ x ∧ x < 7) := 
sorry

end find_x_set_l2241_224179


namespace winning_candidate_percentage_l2241_224148

theorem winning_candidate_percentage
  (total_votes : ℕ)
  (vote_majority : ℕ)
  (winning_candidate_votes : ℕ)
  (losing_candidate_votes : ℕ) :
  total_votes = 400 →
  vote_majority = 160 →
  winning_candidate_votes = total_votes * 70 / 100 →
  losing_candidate_votes = total_votes - winning_candidate_votes →
  winning_candidate_votes - losing_candidate_votes = vote_majority →
  winning_candidate_votes = 280 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end winning_candidate_percentage_l2241_224148


namespace find_prime_pair_l2241_224112

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_prime_pair (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) (h_prime : is_prime (p^5 - q^5)) : (p, q) = (3, 2) := 
  sorry

end find_prime_pair_l2241_224112


namespace probability_ace_spades_then_king_spades_l2241_224126

theorem probability_ace_spades_then_king_spades :
  ∃ (p : ℚ), (p = 1/52 * 1/51) := sorry

end probability_ace_spades_then_king_spades_l2241_224126


namespace difference_of_squares_l2241_224136

theorem difference_of_squares : (540^2 - 460^2 = 80000) :=
by
  have a := 540
  have b := 460
  have identity := (a + b) * (a - b)
  sorry

end difference_of_squares_l2241_224136


namespace compare_logs_l2241_224176

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log 3 / Real.log 5
noncomputable def c := Real.log 5 / Real.log 8

theorem compare_logs : a < b ∧ b < c := by
  sorry

end compare_logs_l2241_224176


namespace sum_areas_of_square_and_rectangle_l2241_224138

theorem sum_areas_of_square_and_rectangle (s w l : ℝ) 
  (h1 : s^2 + w * l = 130)
  (h2 : 4 * s - 2 * (w + l) = 20)
  (h3 : l = 2 * w) : 
  s^2 + 2 * w^2 = 118 :=
by
  -- Provide space for proof
  sorry

end sum_areas_of_square_and_rectangle_l2241_224138


namespace center_and_radius_of_circle_l2241_224102

theorem center_and_radius_of_circle (x y : ℝ) : 
  (x + 1)^2 + (y - 2)^2 = 4 → (x = -1 ∧ y = 2 ∧ ∃ r, r = 2) := 
by
  intro h
  sorry

end center_and_radius_of_circle_l2241_224102


namespace a_7_value_l2241_224152

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

-- Given conditions
def geometric_sequence_positive_terms (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

def geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (a 0 * (1 - ((a (1 + n)) / a 0))) / (1 - (a 1 / a 0))

def S_4_eq_3S_2 (S : ℕ → ℝ) : Prop :=
S 4 = 3 * S 2

def a_3_eq_2 (a : ℕ → ℝ) : Prop :=
a 3 = 2

-- The statement to prove
theorem a_7_value (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  geometric_sequence_positive_terms a →
  geometric_sequence_sum a S →
  S_4_eq_3S_2 S →
  a_3_eq_2 a →
  a 7 = 8 :=
by
  sorry

end a_7_value_l2241_224152


namespace initial_amount_spent_l2241_224115

theorem initial_amount_spent (X : ℝ) 
    (h_bread : X - 3 ≥ 0) 
    (h_candy : X - 3 - 2 ≥ 0) 
    (h_turkey : X - 3 - 2 - (1/3) * (X - 3 - 2) ≥ 0) 
    (h_remaining : X - 3 - 2 - (1/3) * (X - 3 - 2) = 18) : X = 32 := 
sorry

end initial_amount_spent_l2241_224115


namespace number_of_homework_situations_l2241_224107

theorem number_of_homework_situations (teachers students : ℕ) (homework_options : students = 4 ∧ teachers = 3) :
  teachers ^ students = 81 :=
by
  sorry

end number_of_homework_situations_l2241_224107


namespace graphs_intersection_points_l2241_224171

theorem graphs_intersection_points {g : ℝ → ℝ} (h_injective : Function.Injective g) :
  ∃ (x1 x2 x3 : ℝ), (g (x1^3) = g (x1^5)) ∧ (g (x2^3) = g (x2^5)) ∧ (g (x3^3) = g (x3^5)) ∧ 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ (x : ℝ), (g (x^3) = g (x^5)) → (x = x1 ∨ x = x2 ∨ x = x3) := 
by
  sorry

end graphs_intersection_points_l2241_224171


namespace bake_sale_comparison_l2241_224128

theorem bake_sale_comparison :
  let tamara_small_brownies := 4 * 2
  let tamara_large_brownies := 12 * 3
  let tamara_cookies := 36 * 1.5
  let tamara_total := tamara_small_brownies + tamara_large_brownies + tamara_cookies

  let sarah_muffins := 24 * 1.75
  let sarah_choco_cupcakes := 7 * 2.5
  let sarah_vanilla_cupcakes := 8 * 2
  let sarah_strawberry_cupcakes := 15 * 2.75
  let sarah_total := sarah_muffins + sarah_choco_cupcakes + sarah_vanilla_cupcakes + sarah_strawberry_cupcakes

  sarah_total - tamara_total = 18.75 := by
  sorry

end bake_sale_comparison_l2241_224128


namespace total_players_is_60_l2241_224170

-- Define the conditions
def Cricket_players : ℕ := 25
def Hockey_players : ℕ := 20
def Football_players : ℕ := 30
def Softball_players : ℕ := 18

def Cricket_and_Hockey : ℕ := 5
def Cricket_and_Football : ℕ := 8
def Cricket_and_Softball : ℕ := 3
def Hockey_and_Football : ℕ := 4
def Hockey_and_Softball : ℕ := 6
def Football_and_Softball : ℕ := 9

def Cricket_Hockey_and_Football_not_Softball : ℕ := 2

-- Define total unique players present on the ground
def total_unique_players : ℕ :=
  Cricket_players + Hockey_players + Football_players + Softball_players -
  (Cricket_and_Hockey + Cricket_and_Football + Cricket_and_Softball +
   Hockey_and_Football + Hockey_and_Softball + Football_and_Softball) +
  Cricket_Hockey_and_Football_not_Softball

-- Statement
theorem total_players_is_60:
  total_unique_players = 60 :=
by
  sorry

end total_players_is_60_l2241_224170


namespace single_reduction_equivalent_l2241_224164

theorem single_reduction_equivalent (P : ℝ) (P_pos : 0 < P) : 
  (P - (P - 0.30 * P)) / P = 0.70 := 
by
  -- Let's denote the original price by P, 
  -- apply first 25% and then 60% reduction 
  -- and show that it's equivalent to a single 70% reduction
  sorry

end single_reduction_equivalent_l2241_224164


namespace polar_equation_parabola_l2241_224186

/-- Given a polar equation 4 * ρ * (sin(θ / 2))^2 = 5, prove that it represents a parabola in Cartesian coordinates. -/
theorem polar_equation_parabola (ρ θ : ℝ) (h : 4 * ρ * (Real.sin (θ / 2))^ 2 = 5) : 
  ∃ (a : ℝ), a ≠ 0 ∧ (∃ b c : ℝ, ∀ x y : ℝ, (y^2 = a * (x + b)) ∨ (x = c ∨ y = 0)) := 
sorry

end polar_equation_parabola_l2241_224186


namespace coffee_price_increase_l2241_224122

variable (C : ℝ) -- cost per pound of green tea and coffee in June
variable (P_green_tea_july : ℝ := 0.1) -- price of green tea per pound in July
variable (mixture_cost : ℝ := 3.15) -- cost of mixture of equal quantities of green tea and coffee for 3 lbs
variable (green_tea_cost_per_lb_july : ℝ := 0.1) -- cost per pound of green tea in July
variable (green_tea_weight : ℝ := 1.5) -- weight of green tea in the mixture in lbs
variable (coffee_weight : ℝ := 1.5) -- weight of coffee in the mixture in lbs
variable (coffee_cost_per_lb_july : ℝ := 2.0) -- cost per pound of coffee in July

theorem coffee_price_increase :
  C = 1 → mixture_cost = 3.15 →
  P_green_tea_july * C = green_tea_cost_per_lb_july →
  green_tea_weight * green_tea_cost_per_lb_july + coffee_weight * coffee_cost_per_lb_july = mixture_cost →
  (coffee_cost_per_lb_july - C) / C * 100 = 100 :=
by
  intros
  sorry

end coffee_price_increase_l2241_224122


namespace zero_in_P_two_not_in_P_l2241_224161

variables (P : Set Int)

-- Conditions
def condition_1 := ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0
def condition_2 := ∃ x ∈ P, x % 2 = 0 ∧ ∃ y ∈ P, y % 2 ≠ 0 
def condition_3 := 1 ∉ P
def condition_4 := ∀ x y, x ∈ P → y ∈ P → x + y ∈ P

-- Proving 0 ∈ P
theorem zero_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 0 ∈ P := 
sorry

-- Proving 2 ∉ P
theorem two_not_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 2 ∉ P := 
sorry

end zero_in_P_two_not_in_P_l2241_224161


namespace eval_expr_at_2_l2241_224147

def expr (x : ℝ) : ℝ := (3 * x + 4)^2

theorem eval_expr_at_2 : expr 2 = 100 :=
by sorry

end eval_expr_at_2_l2241_224147


namespace general_term_of_an_l2241_224140

theorem general_term_of_an (a : ℕ → ℕ) (h1 : a 1 = 1)
    (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) :
    ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end general_term_of_an_l2241_224140


namespace solve_for_x_l2241_224119

theorem solve_for_x (x : ℚ) : (x + 4) / (x - 3) = (x - 2) / (x + 2) -> x = -2 / 11 := by
  sorry

end solve_for_x_l2241_224119


namespace cannot_determine_b_l2241_224178

theorem cannot_determine_b 
  (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 12.345) 
  (h_ineq : a > b ∧ b > c ∧ c > d) : 
  ¬((b = 12.345) ∨ (b > 12.345) ∨ (b < 12.345)) :=
sorry

end cannot_determine_b_l2241_224178


namespace part1_part2_l2241_224166

-- Define set A
def set_A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }

-- Define set B depending on m
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 ≤ x ∧ x ≤ m + 1 }

-- Part 1: When m = -3, find A ∩ B
theorem part1 : set_B (-3) ∩ set_A = { x | -3 ≤ x ∧ x ≤ -2 } := 
sorry

-- Part 2: Find the range of m such that B ⊆ A
theorem part2 (m : ℝ) : set_B m ⊆ set_A ↔ m ≥ -1 :=
sorry

end part1_part2_l2241_224166


namespace Tonya_initial_stamps_l2241_224184

theorem Tonya_initial_stamps :
  ∀ (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (jimmy_matchbooks : ℕ) (tonya_remaining_stamps : ℕ),
  stamps_per_match = 12 →
  matches_per_matchbook = 24 →
  jimmy_matchbooks = 5 →
  tonya_remaining_stamps = 3 →
  tonya_remaining_stamps + (jimmy_matchbooks * matches_per_matchbook) / stamps_per_match = 13 := 
by
  intros stamps_per_match matches_per_matchbook jimmy_matchbooks tonya_remaining_stamps
  sorry

end Tonya_initial_stamps_l2241_224184


namespace function_above_x_axis_l2241_224129

theorem function_above_x_axis (m : ℝ) : 
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) ↔ m < 2 + 2 * Real.sqrt 2 :=
sorry

end function_above_x_axis_l2241_224129


namespace triathlon_minimum_speeds_l2241_224174

theorem triathlon_minimum_speeds (x : ℝ) (T : ℝ := 80) (total_time : ℝ := (800 / x + 20000 / (7.5 * x) + 4000 / (3 * x))) :
  total_time ≤ T → x ≥ 60 ∧ 3 * x = 180 ∧ 7.5 * x = 450 :=
by
  sorry

end triathlon_minimum_speeds_l2241_224174


namespace inverse_proportionality_example_l2241_224116

theorem inverse_proportionality_example (k : ℝ) (x : ℝ) (y : ℝ) (h1 : 5 * 10 = k) (h2 : x * 40 = k) : x = 5 / 4 :=
by
  -- sorry is used to skip the proof.
  sorry

end inverse_proportionality_example_l2241_224116


namespace painter_rooms_painted_l2241_224172

theorem painter_rooms_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ) 
    (h1 : total_rooms = 9) (h2 : hours_per_room = 8) (h3 : remaining_hours = 32) : 
    total_rooms - (remaining_hours / hours_per_room) = 5 :=
by
  sorry

end painter_rooms_painted_l2241_224172


namespace text_messages_in_march_l2241_224142

theorem text_messages_in_march
  (nov_texts : ℕ)
  (dec_texts : ℕ)
  (jan_texts : ℕ)
  (feb_texts : ℕ)
  (double_pattern : ∀ n m : ℕ, m = 2 * n)
  (h_nov : nov_texts = 1)
  (h_dec : dec_texts = 2 * nov_texts)
  (h_jan : jan_texts = 2 * dec_texts)
  (h_feb : feb_texts = 2 * jan_texts) : 
  ∃ mar_texts : ℕ, mar_texts = 2 * feb_texts ∧ mar_texts = 16 := 
by
  sorry

end text_messages_in_march_l2241_224142


namespace line_equation_l2241_224156

theorem line_equation (θ : Real) (b : Real) (h1 : θ = 45) (h2 : b = 2) : (y = x + b) :=
by
  -- Assume θ = 45°. The corresponding slope is k = tan(θ) = 1.
  -- Since the y-intercept b = 2, the equation of the line y = mx + b = x + 2.
  sorry

end line_equation_l2241_224156


namespace eval_expression_l2241_224183

theorem eval_expression : (4^2 - 2^3) = 8 := by
  sorry

end eval_expression_l2241_224183


namespace paco_initial_salty_cookies_l2241_224199

variable (S : ℕ)
variable (sweet_cookies : ℕ := 40)
variable (salty_cookies_eaten1 : ℕ := 28)
variable (sweet_cookies_eaten : ℕ := 15)
variable (extra_salty_cookies_eaten : ℕ := 13)

theorem paco_initial_salty_cookies 
  (h1 : salty_cookies_eaten1 = 28)
  (h2 : sweet_cookies_eaten = 15)
  (h3 : extra_salty_cookies_eaten = 13)
  (h4 : sweet_cookies = 40)
  : (S = (salty_cookies_eaten1 + (extra_salty_cookies_eaten + sweet_cookies_eaten))) :=
by
  -- starting with the equation S = number of salty cookies Paco
  -- initially had, which should be equal to the total salty 
  -- cookies he ate.
  sorry

end paco_initial_salty_cookies_l2241_224199


namespace number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l2241_224193

def cars_sold_each_day_first_three_days : ℕ := 5
def days_first_period : ℕ := 3
def quota : ℕ := 50
def cars_remaining_after_next_four_days : ℕ := 23
def days_next_period : ℕ := 4

theorem number_of_cars_sold_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period) - cars_remaining_after_next_four_days = 12 :=
by
  sorry

theorem cars_sold_each_day_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period - cars_remaining_after_next_four_days) / days_next_period = 3 :=
by
  sorry

end number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l2241_224193


namespace Riverdale_High_students_l2241_224177

theorem Riverdale_High_students
  (f j : ℕ)
  (h1 : (3 / 7) * f + (3 / 4) * j = 234)
  (h2 : f + j = 420) :
  f = 64 ∧ j = 356 := by
  sorry

end Riverdale_High_students_l2241_224177


namespace second_container_mass_l2241_224105

-- Given conditions
def height1 := 4 -- height of first container in cm
def width1 := 2 -- width of first container in cm
def length1 := 8 -- length of first container in cm
def mass1 := 64 -- mass of material the first container can hold in grams

def height2 := 3 * height1 -- height of second container in cm
def width2 := 2 * width1 -- width of second container in cm
def length2 := length1 -- length of second container in cm

def volume (height width length : ℤ) : ℤ := height * width * length

-- The proof statement
theorem second_container_mass : volume height2 width2 length2 = 6 * volume height1 width1 length1 → 6 * mass1 = 384 :=
by
  sorry

end second_container_mass_l2241_224105


namespace log_domain_is_pos_real_l2241_224185

noncomputable def domain_log : Set ℝ := {x | x > 0}
noncomputable def domain_reciprocal : Set ℝ := {x | x ≠ 0}
noncomputable def domain_sqrt : Set ℝ := {x | x ≥ 0}
noncomputable def domain_exp : Set ℝ := {x | true}

theorem log_domain_is_pos_real :
  (domain_log = {x : ℝ | 0 < x}) ∧ 
  (domain_reciprocal = {x : ℝ | x ≠ 0}) ∧ 
  (domain_sqrt = {x : ℝ | 0 ≤ x}) ∧ 
  (domain_exp = {x : ℝ | true}) →
  domain_log = {x : ℝ | 0 < x} :=
by
  intro h
  sorry

end log_domain_is_pos_real_l2241_224185


namespace total_number_of_legs_l2241_224137

def kangaroos : ℕ := 23
def goats : ℕ := 3 * kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_number_of_legs : 
  (kangaroos * legs_of_kangaroo + goats * legs_of_goat) = 322 := by
  sorry

end total_number_of_legs_l2241_224137


namespace compute_expression_l2241_224168

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end compute_expression_l2241_224168


namespace calculate_expr_l2241_224169

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem calculate_expr : ((x^3 * y^2)^2 * (x / y^3)) = x^7 * y :=
by sorry

end calculate_expr_l2241_224169


namespace at_most_one_cube_l2241_224175

theorem at_most_one_cube (a : ℕ → ℕ) (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2018) :
  ∃! n, ∃ m : ℕ, a n = m ^ 3 := sorry

end at_most_one_cube_l2241_224175


namespace find_k_for_circle_radius_5_l2241_224165

theorem find_k_for_circle_radius_5 (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) → k = -27 :=
by
  sorry

end find_k_for_circle_radius_5_l2241_224165


namespace sum_arithmetic_sequence_has_max_value_l2241_224120

noncomputable section
open Classical

-- Defining an arithmetic sequence with first term a1 and common difference d
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + d * (n - 1)

-- Defining the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

-- The main statement to prove: Sn has a maximum value given conditions a1 > 0 and d < 0
theorem sum_arithmetic_sequence_has_max_value (a1 d : ℝ) (h1 : a1 > 0) (h2 : d < 0) :
  ∃ M, ∀ n, sum_arithmetic_sequence a1 d n ≤ M :=
by
  sorry

end sum_arithmetic_sequence_has_max_value_l2241_224120


namespace part_a_exists_part_b_impossible_l2241_224159

def gridSize : Nat := 7 * 14
def cellCount (x y : Nat) : Nat := 4 * x + 3 * y
def x_equals_y_condition (x y : Nat) : Prop := x = y
def x_greater_y_condition (x y : Nat) : Prop := x > y

theorem part_a_exists (x y : Nat) (h : cellCount x y = gridSize) : ∃ (x y : Nat), x_equals_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry

theorem part_b_impossible (x y : Nat) (h : cellCount x y = gridSize) : ¬ ∃ (x y : Nat), x_greater_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry


end part_a_exists_part_b_impossible_l2241_224159


namespace faucet_leakage_volume_l2241_224181

def leakage_rate : ℝ := 0.1
def time_seconds : ℝ := 14400
def expected_volume : ℝ := 1.4 * 10^3

theorem faucet_leakage_volume : 
  leakage_rate * time_seconds = expected_volume := 
by
  -- proof
  sorry

end faucet_leakage_volume_l2241_224181


namespace incorrect_option_l2241_224125

-- Definitions and conditions from the problem
def p (x : ℝ) : Prop := (x - 2) * Real.sqrt (x^2 - 3*x + 2) ≥ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, k * x^2 - k * x - 1 < 0

-- The Lean 4 statement to verify the problem
theorem incorrect_option :
  (¬ ∃ x, p x) ∧ (∃ k, q k) ∧
  (∀ k, -4 < k ∧ k ≤ 0 → q k) →
  (∃ x, ¬p x) :=
  by
  sorry

end incorrect_option_l2241_224125


namespace volume_ratio_l2241_224157

theorem volume_ratio (x : ℝ) (h : x > 0) : 
  let V_Q := x^3
  let V_P := (3 * x)^3
  (V_Q / V_P) = (1 / 27) :=
by
  sorry

end volume_ratio_l2241_224157


namespace total_scoops_needed_l2241_224103

def cups_of_flour : ℕ := 4
def cups_of_sugar : ℕ := 3
def cups_of_milk : ℕ := 2

def flour_scoop_size : ℚ := 1 / 4
def sugar_scoop_size : ℚ := 1 / 3
def milk_scoop_size : ℚ := 1 / 2

theorem total_scoops_needed : 
  (cups_of_flour / flour_scoop_size) + (cups_of_sugar / sugar_scoop_size) + (cups_of_milk / milk_scoop_size) = 29 := 
by {
  sorry
}

end total_scoops_needed_l2241_224103


namespace cube_modulo_9_l2241_224110

theorem cube_modulo_9 (N : ℤ) (h : N % 9 = 2 ∨ N % 9 = 5 ∨ N % 9 = 8) : 
  (N^3) % 9 = 8 :=
by sorry

end cube_modulo_9_l2241_224110


namespace sum_second_largest_and_smallest_l2241_224124

theorem sum_second_largest_and_smallest :
  let numbers := [10, 11, 12, 13, 14]
  ∃ second_largest second_smallest, (List.nthLe numbers 3 sorry = second_largest ∧ List.nthLe numbers 1 sorry = second_smallest ∧ second_largest + second_smallest = 24) :=
sorry

end sum_second_largest_and_smallest_l2241_224124
