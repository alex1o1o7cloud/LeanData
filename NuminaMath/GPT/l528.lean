import Mathlib

namespace part_a_roots_part_b_sum_l528_52808

theorem part_a_roots : ∀ x : ℝ, 2^x = x + 1 ↔ x = 0 ∨ x = 1 :=
by 
  intros x
  sorry

theorem part_b_sum (f : ℝ → ℝ) (h : ∀ x : ℝ, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 :=
by 
  sorry

end part_a_roots_part_b_sum_l528_52808


namespace find_n_l528_52876

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_prime : Nat.Prime (n^4 - 16 * n^2 + 100)) : n = 3 := 
sorry

end find_n_l528_52876


namespace expected_value_is_20_point_5_l528_52821

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def coin_heads_probability : ℚ := 1 / 2

noncomputable def expected_value : ℚ :=
  coin_heads_probability * (penny_value + nickel_value + dime_value + quarter_value)

theorem expected_value_is_20_point_5 :
  expected_value = 20.5 := by
  sorry

end expected_value_is_20_point_5_l528_52821


namespace exists_nonneg_coefs_some_n_l528_52805

-- Let p(x) be a polynomial with real coefficients
variable (p : Polynomial ℝ)

-- Assumption: p(x) > 0 for all x >= 0
axiom positive_poly : ∀ x : ℝ, x ≥ 0 → p.eval x > 0 

theorem exists_nonneg_coefs_some_n :
  ∃ n : ℕ, ∀ k : ℕ, Polynomial.coeff ((1 + Polynomial.X)^n * p) k ≥ 0 :=
sorry

end exists_nonneg_coefs_some_n_l528_52805


namespace points_three_units_away_from_neg_two_on_number_line_l528_52889

theorem points_three_units_away_from_neg_two_on_number_line :
  ∃! p1 p2 : ℤ, |p1 + 2| = 3 ∧ |p2 + 2| = 3 ∧ p1 ≠ p2 ∧ (p1 = -5 ∨ p2 = -5) ∧ (p1 = 1 ∨ p2 = 1) :=
sorry

end points_three_units_away_from_neg_two_on_number_line_l528_52889


namespace no_such_n_l528_52847

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end no_such_n_l528_52847


namespace k_times_a_plus_b_l528_52811

/-- Given a quadrilateral with vertices P(ka, kb), Q(kb, ka), R(-ka, -kb), and S(-kb, -ka),
where a and b are consecutive integers with a > b > 0, and k is an odd integer.
It is given that the area of PQRS is 50.
Prove that k(a + b) = 5. -/
theorem k_times_a_plus_b (a b k : ℤ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a = b + 1)
  (h4 : Odd k)
  (h5 : 2 * k^2 * (a - b) * (a + b) = 50) :
  k * (a + b) = 5 := by
  sorry

end k_times_a_plus_b_l528_52811


namespace single_discount_equivalence_l528_52898

theorem single_discount_equivalence (original_price : ℝ) (first_discount second_discount : ℝ) (final_price : ℝ) :
  original_price = 50 →
  first_discount = 0.30 →
  second_discount = 0.10 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  ((original_price - final_price) / original_price) * 100 = 37 := by
  sorry

end single_discount_equivalence_l528_52898


namespace second_consecutive_odd_integer_l528_52897

theorem second_consecutive_odd_integer (x : ℤ) 
  (h1 : ∃ x, x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) 
  (h2 : (x + 2) + (x + 4) = x + 17) : 
  (x + 2) = 13 :=
by
  sorry

end second_consecutive_odd_integer_l528_52897


namespace cone_base_radius_l528_52841

open Real

theorem cone_base_radius (r_sector : ℝ) (θ_sector : ℝ) : 
    r_sector = 6 ∧ θ_sector = 120 → (∃ r : ℝ, 2 * π * r = θ_sector * π * r_sector / 180 ∧ r = 2) :=
by
  sorry

end cone_base_radius_l528_52841


namespace remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l528_52881

structure ArtCollection where
  medieval : ℕ
  renaissance : ℕ
  modern : ℕ

def AliciaArtCollection : ArtCollection := {
  medieval := 70,
  renaissance := 120,
  modern := 150
}

def donationPercentages : ArtCollection := {
  medieval := 65,
  renaissance := 30,
  modern := 45
}

def remainingArtPieces (initial : ℕ) (percent : ℕ) : ℕ :=
  initial - ((percent * initial) / 100)

theorem remaining_medieval_art_pieces :
  remainingArtPieces AliciaArtCollection.medieval donationPercentages.medieval = 25 := by
  sorry

theorem remaining_renaissance_art_pieces :
  remainingArtPieces AliciaArtCollection.renaissance donationPercentages.renaissance = 84 := by
  sorry

theorem remaining_modern_art_pieces :
  remainingArtPieces AliciaArtCollection.modern donationPercentages.modern = 83 := by
  sorry

end remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l528_52881


namespace deepak_age_l528_52800

theorem deepak_age (A D : ℕ)
  (h1 : A / D = 2 / 3)
  (h2 : A + 5 = 25) :
  D = 30 := 
by
  sorry

end deepak_age_l528_52800


namespace min_floor_sum_l528_52883

-- Definitions of the conditions
variables (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24)

-- Our main theorem statement
theorem min_floor_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 24) :
  (Nat.floor ((a+b) / c) + Nat.floor ((b+c) / a) + Nat.floor ((c+a) / b)) = 6 := 
sorry

end min_floor_sum_l528_52883


namespace remainder_N_div_5_is_1_l528_52846

-- The statement proving the remainder of N when divided by 5 is 1
theorem remainder_N_div_5_is_1 (N : ℕ) (h1 : N % 2 = 1) (h2 : N % 35 = 1) : N % 5 = 1 :=
sorry

end remainder_N_div_5_is_1_l528_52846


namespace simplify_and_evaluate_expression_l528_52866

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end simplify_and_evaluate_expression_l528_52866


namespace spacy_subsets_15_l528_52806

def spacy_subsets_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | (k + 5) => spacy_subsets_count (k + 4) + spacy_subsets_count k

theorem spacy_subsets_15 : spacy_subsets_count 15 = 181 :=
sorry

end spacy_subsets_15_l528_52806


namespace gray_region_area_l528_52880

theorem gray_region_area 
  (r : ℝ) 
  (h1 : ∀ r : ℝ, (3 * r) - r = 3) 
  (h2 : r = 1.5) 
  (inner_circle_area : ℝ := π * r * r) 
  (outer_circle_area : ℝ := π * (3 * r) * (3 * r)) : 
  outer_circle_area - inner_circle_area = 18 * π := 
by
  sorry

end gray_region_area_l528_52880


namespace radius_of_first_cylinder_l528_52873

theorem radius_of_first_cylinder :
  ∀ (rounds1 rounds2 : ℕ) (r2 r1 : ℝ), rounds1 = 70 → rounds2 = 49 → r2 = 20 → 
  (2 * Real.pi * r1 * rounds1 = 2 * Real.pi * r2 * rounds2) → r1 = 14 :=
by
  sorry

end radius_of_first_cylinder_l528_52873


namespace oranges_for_juice_l528_52869

theorem oranges_for_juice (total_oranges : ℝ) (exported_percentage : ℝ) (juice_percentage : ℝ) :
  total_oranges = 7 →
  exported_percentage = 0.30 →
  juice_percentage = 0.60 →
  (total_oranges * (1 - exported_percentage) * juice_percentage) = 2.9 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end oranges_for_juice_l528_52869


namespace meadow_area_l528_52807

theorem meadow_area (x : ℝ) (h1 : ∀ y : ℝ, y = x / 2 + 3) (h2 : ∀ z : ℝ, z = 1 / 3 * (x / 2 - 3) + 6) :
  (x / 2 + 3) + (1 / 3 * (x / 2 - 3) + 6) = x → x = 24 := by
  sorry

end meadow_area_l528_52807


namespace number_of_students_l528_52818

theorem number_of_students (n T : ℕ) (h1 : T = n * 90) 
(h2 : T - 120 = (n - 3) * 95) : n = 33 := 
by
sorry

end number_of_students_l528_52818


namespace percentage_of_l_equals_150_percent_k_l528_52878

section

variables (j k l m : ℝ) (x : ℝ)

-- Given conditions
axiom cond1 : 1.25 * j = 0.25 * k
axiom cond2 : 1.50 * k = x / 100 * l
axiom cond3 : 1.75 * l = 0.75 * m
axiom cond4 : 0.20 * m = 7.00 * j

-- Proof statement
theorem percentage_of_l_equals_150_percent_k : x = 50 :=
sorry

end

end percentage_of_l_equals_150_percent_k_l528_52878


namespace reflect_center_is_image_center_l528_52850

def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem reflect_center_is_image_center : 
  reflect_over_y_eq_neg_x (3, -4) = (4, -3) :=
by
  -- Proof is omitted as per instructions.
  -- This proof would show the reflection of the point (3, -4) over the line y = -x resulting in (4, -3).
  sorry

end reflect_center_is_image_center_l528_52850


namespace remainder_of_x_div_9_l528_52851

theorem remainder_of_x_div_9 (x : ℕ) (hx_pos : 0 < x) (h : (6 * x) % 9 = 3) : x % 9 = 5 :=
by {
  sorry
}

end remainder_of_x_div_9_l528_52851


namespace monotone_increasing_interval_l528_52879

def f (x : ℝ) := x^2 - 2

theorem monotone_increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y :=
by
  sorry

end monotone_increasing_interval_l528_52879


namespace factorize_x4_minus_81_l528_52822

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l528_52822


namespace two_largest_divisors_difference_l528_52819

theorem two_largest_divisors_difference (N : ℕ) (h : N > 1) (a : ℕ) (ha : a ∣ N) (h6a : 6 * a ∣ N) :
  (N / 2 : ℚ) / (N / 3 : ℚ) = 1.5 := by
  sorry

end two_largest_divisors_difference_l528_52819


namespace find_c_for_minimum_value_l528_52877

-- Definitions based on the conditions
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Main statement to be proved
theorem find_c_for_minimum_value (c : ℝ) : (∀ x, (3*x^2 - 4*c*x + c^2) = 0) → c = 3 :=
by
  sorry

end find_c_for_minimum_value_l528_52877


namespace joggers_meet_again_at_correct_time_l528_52894

-- Define the joggers and their lap times
def bob_lap_time := 3
def carol_lap_time := 5
def ted_lap_time := 8

-- Calculate the Least Common Multiple (LCM) of their lap times
def lcm_joggers := Nat.lcm (Nat.lcm bob_lap_time carol_lap_time) ted_lap_time

-- Start time is 9:00 AM
def start_time := 9 * 60  -- in minutes

-- The time (in minutes) we get back together is start_time plus the LCM
def earliest_meeting_time := start_time + lcm_joggers

-- Convert the meeting time to hours and minutes
def hours := earliest_meeting_time / 60
def minutes := earliest_meeting_time % 60

-- Define an expected result
def expected_meeting_hour := 11
def expected_meeting_minute := 0

-- Prove that all joggers will meet again at the correct time
theorem joggers_meet_again_at_correct_time :
  hours = expected_meeting_hour ∧ minutes = expected_meeting_minute :=
by
  -- Here you would provide the proof, but we'll use sorry for brevity
  sorry

end joggers_meet_again_at_correct_time_l528_52894


namespace find_percentage_l528_52845

variable (P : ℝ)

/-- A number P% that satisfies the condition is 65. -/
theorem find_percentage (h : ((P / 100) * 40 = ((5 / 100) * 60) + 23)) : P = 65 :=
sorry

end find_percentage_l528_52845


namespace combined_mpg_l528_52853

-- Definitions based on the conditions
def ray_miles : ℕ := 150
def tom_miles : ℕ := 100
def ray_mpg : ℕ := 30
def tom_mpg : ℕ := 20

-- Theorem statement
theorem combined_mpg : (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 25 := by
  sorry

end combined_mpg_l528_52853


namespace find_larger_number_l528_52836

theorem find_larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : 2 * (x + y) = 40) : x = 12.5 :=
by 
  sorry

end find_larger_number_l528_52836


namespace rectangle_with_perpendicular_diagonals_is_square_l528_52843

-- Define rectangle and its properties
structure Rectangle where
  length : ℝ
  width : ℝ
  opposite_sides_equal : length = width

-- Define the condition that the diagonals of the rectangle are perpendicular
axiom perpendicular_diagonals {r : Rectangle} : r.length = r.width → True

-- Define the square property that a rectangle with all sides equal is a square
structure Square extends Rectangle where
  all_sides_equal : length = width

-- The main theorem to be proven
theorem rectangle_with_perpendicular_diagonals_is_square (r : Rectangle) (h : r.length = r.width) : Square := by
  sorry

end rectangle_with_perpendicular_diagonals_is_square_l528_52843


namespace sample_size_student_congress_l528_52875

-- Definitions based on the conditions provided in the problem
def num_classes := 40
def students_per_class := 3

-- Theorem statement for the mathematically equivalent proof problem
theorem sample_size_student_congress : 
  (num_classes * students_per_class) = 120 := 
by 
  sorry

end sample_size_student_congress_l528_52875


namespace simplify_expression_l528_52824

variable (a b : ℚ)

theorem simplify_expression (ha : a = -2) (hb : b = 1/5) :
  2 * (a^2 * b - 2 * a * b) - 3 * (a^2 * b - 3 * a * b) + a^2 * b = -2 := by
  -- Proof can be filled here
  sorry

end simplify_expression_l528_52824


namespace maximize_profit_l528_52803

theorem maximize_profit : 
  ∃ (a b : ℕ), 
  a ≤ 8 ∧ 
  b ≤ 7 ∧ 
  2 * a + b ≤ 19 ∧ 
  a + b ≤ 12 ∧ 
  10 * a + 6 * b ≥ 72 ∧ 
  (a * 450 + b * 350) = 4900 :=
by
  sorry

end maximize_profit_l528_52803


namespace central_angle_of_sector_l528_52830

noncomputable def central_angle (l S r : ℝ) : ℝ :=
  2 * S / r^2

theorem central_angle_of_sector (r : ℝ) (h₁ : 4 * r / 2 = 4) (h₂ : r = 2) : central_angle 4 4 r = 2 :=
by
  sorry

end central_angle_of_sector_l528_52830


namespace part_a_l528_52854

-- Part (a)
theorem part_a (x : ℕ)  : (x^2 - x + 2) % 7 = 0 → x % 7 = 4 := by 
  sorry

end part_a_l528_52854


namespace wire_not_used_l528_52857

variable (total_wire length_cut_parts parts_used : ℕ)

theorem wire_not_used (h1 : total_wire = 50) (h2 : length_cut_parts = 5) (h3 : parts_used = 3) : 
  total_wire - (parts_used * (total_wire / length_cut_parts)) = 20 := 
  sorry

end wire_not_used_l528_52857


namespace players_at_least_two_sciences_l528_52865

-- Define the conditions of the problem
def total_players : Nat := 30
def players_biology : Nat := 15
def players_chemistry : Nat := 10
def players_physics : Nat := 5
def players_all_three : Nat := 3

-- Define the main theorem we want to prove
theorem players_at_least_two_sciences :
  (players_biology + players_chemistry + players_physics 
    - players_all_three - total_players) = 9 :=
sorry

end players_at_least_two_sciences_l528_52865


namespace range_g_l528_52835

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x + 1
noncomputable def g (a x : ℝ) : ℝ := x^2 + a * x + 1

theorem range_g (a : ℝ) (h : Set.range (λ x => f a x) = Set.univ) : Set.range (λ x => g a x) = { y : ℝ | 1 ≤ y } := by
  sorry

end range_g_l528_52835


namespace relationship_xyz_w_l528_52887

theorem relationship_xyz_w (x y z w : ℝ) (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) :
  x = 2 * z - w := 
sorry

end relationship_xyz_w_l528_52887


namespace general_formula_a_sum_T_max_k_value_l528_52810

-- Given conditions
noncomputable def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 6 else n + 5
noncomputable def b (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))
noncomputable def T (n : ℕ) : ℚ := (3 * n) / (2 * n + 1)

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = if n = 1 then 6 else n + 5 :=
by sorry

theorem sum_T (n : ℕ) : T n = (3 * n) / (2 * n + 1) :=
by sorry

theorem max_k_value (k : ℕ) : k = 19 → ∀ n : ℕ, T n > k / 20 :=
by sorry

end general_formula_a_sum_T_max_k_value_l528_52810


namespace right_triangle_legs_l528_52895

theorem right_triangle_legs (a b : ℕ) (h : a^2 + b^2 = 100) (h_r: a + b - 10 = 4) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
sorry

end right_triangle_legs_l528_52895


namespace geometric_series_sum_l528_52872

theorem geometric_series_sum (a r : ℝ)
  (h₁ : a / (1 - r) = 15)
  (h₂ : a / (1 - r^4) = 9) :
  r = 1 / 3 :=
sorry

end geometric_series_sum_l528_52872


namespace apple_tree_fruits_production_l528_52842

def apple_production (first_season : ℕ) (second_season : ℕ) (third_season : ℕ): ℕ :=
  first_season + second_season + third_season

theorem apple_tree_fruits_production :
  let first_season := 200
  let second_season := 160    -- 200 - 20% of 200
  let third_season := 320     -- 2 * 160
  apple_production first_season second_season third_season = 680 := by
  -- This is where the proof would go
  sorry

end apple_tree_fruits_production_l528_52842


namespace hyperbola_asymptotes_l528_52858

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), 16 * x^2 - 9 * y^2 = -144 → (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intros x y h1
  sorry

end hyperbola_asymptotes_l528_52858


namespace evaluate_expression_l528_52848

theorem evaluate_expression :
  2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 :=
by
  sorry

end evaluate_expression_l528_52848


namespace sum_smallest_largest_l528_52886

theorem sum_smallest_largest (z b : ℤ) (n : ℤ) (h_even_n : (n % 2 = 0)) (h_mean : z = (n * b + ((n - 1) * n) / 2) / n) : 
  (2 * (z - (n - 1) / 2) + n - 1) = 2 * z := by
  sorry

end sum_smallest_largest_l528_52886


namespace Bernoulli_inequality_l528_52863

theorem Bernoulli_inequality (n : ℕ) (a : ℝ) (h : a > -1) : (1 + a)^n ≥ n * a + 1 := 
sorry

end Bernoulli_inequality_l528_52863


namespace probability_jqk_3_13_l528_52804

def probability_jack_queen_king (total_cards jacks queens kings : ℕ) : ℚ :=
  (jacks + queens + kings) / total_cards

theorem probability_jqk_3_13 :
  probability_jack_queen_king 52 4 4 4 = 3 / 13 := by
  sorry

end probability_jqk_3_13_l528_52804


namespace total_cookies_in_box_l528_52888

-- Definitions from the conditions
def oldest_son_cookies : ℕ := 4
def youngest_son_cookies : ℕ := 2
def days_box_lasts : ℕ := 9

-- Total cookies consumed per day
def daily_cookies_consumption : ℕ := oldest_son_cookies + youngest_son_cookies

-- Theorem statement: total number of cookies in the box
theorem total_cookies_in_box : (daily_cookies_consumption * days_box_lasts) = 54 := by
  sorry

end total_cookies_in_box_l528_52888


namespace win_probability_l528_52827

theorem win_probability (P_lose : ℚ) (h : P_lose = 5 / 8) : (1 - P_lose = 3 / 8) :=
by
  -- Provide the proof here if needed, but skip it
  sorry

end win_probability_l528_52827


namespace polynomial_multiple_of_six_l528_52859

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end polynomial_multiple_of_six_l528_52859


namespace price_of_turban_l528_52815

theorem price_of_turban : 
  ∃ T : ℝ, (9 / 12) * (90 + T) = 40 + T ↔ T = 110 :=
by
  sorry

end price_of_turban_l528_52815


namespace problem_l528_52871

theorem problem (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) : 
  x * y * z = 4 := 
sorry

end problem_l528_52871


namespace arithmetic_sequence_ratio_l528_52884

-- Definitions based on conditions
variables {a_n b_n : ℕ → ℕ} -- Arithmetic sequences
variables {A_n B_n : ℕ → ℕ} -- Sums of the first n terms

-- Given condition
axiom sums_of_arithmetic_sequences (n : ℕ) : A_n n / B_n n = (7 * n + 1) / (4 * n + 27)

-- Theorem to prove
theorem arithmetic_sequence_ratio :
  ∀ (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℕ), 
    (∀ n, A_n n / B_n n = (7 * n + 1) / (4 * n + 27)) → 
    a_6 / b_6 = 78 / 71 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l528_52884


namespace equal_shipments_by_truck_l528_52864

theorem equal_shipments_by_truck (T : ℕ) (hT1 : 120 % T = 0) (hT2 : T ≠ 5) : T = 2 :=
by
  sorry

end equal_shipments_by_truck_l528_52864


namespace algebra_expression_value_l528_52832

theorem algebra_expression_value (a b : ℝ) 
  (h₁ : a - b = 5) 
  (h₂ : a * b = -1) : 
  (2 * a + 3 * b - 2 * a * b) 
  - (a + 4 * b + a * b) 
  - (3 * a * b + 2 * b - 2 * a) = 21 := 
by
  sorry

end algebra_expression_value_l528_52832


namespace max_grapes_leftover_l528_52833

-- Define variables and conditions
def total_grapes (n : ℕ) : ℕ := n
def kids : ℕ := 5
def grapes_leftover (n : ℕ) : ℕ := n % kids

-- The proposition we need to prove
theorem max_grapes_leftover (n : ℕ) (h : n ≥ 5) : grapes_leftover n = 4 :=
sorry

end max_grapes_leftover_l528_52833


namespace f_2009_value_l528_52861

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_function (f : ℝ → ℝ) : ∀ x, f x = f (-x)
axiom odd_function (g : ℝ → ℝ) : ∀ x, g x = -g (-x)
axiom f_value : f 1 = 0
axiom g_def : ∀ x, g x = f (x - 1)

theorem f_2009_value : f 2009 = 0 :=
by
  sorry

end f_2009_value_l528_52861


namespace brownies_pieces_l528_52801

theorem brownies_pieces (tray_length tray_width piece_length piece_width : ℕ) 
  (h1 : tray_length = 24) 
  (h2 : tray_width = 16) 
  (h3 : piece_length = 2) 
  (h4 : piece_width = 2) : 
  tray_length * tray_width / (piece_length * piece_width) = 96 :=
by sorry

end brownies_pieces_l528_52801


namespace reservoir_water_level_at_6_pm_l528_52837

/-
  Initial conditions:
  - initial_water_level: Water level at 8 a.m.
  - increase_rate: Rate of increase in water level from 8 a.m. to 12 p.m.
  - decrease_rate: Rate of decrease in water level from 12 p.m. to 6 p.m.
  - start_increase_time: Starting time of increase (in hours from 8 a.m.)
  - end_increase_time: Ending time of increase (in hours from 8 a.m.)
  - start_decrease_time: Starting time of decrease (in hours from 12 p.m.)
  - end_decrease_time: Ending time of decrease (in hours from 12 p.m.)
-/
def initial_water_level : ℝ := 45
def increase_rate : ℝ := 0.6
def decrease_rate : ℝ := 0.3
def start_increase_time : ℝ := 0 -- 8 a.m. in hours from 8 a.m.
def end_increase_time : ℝ := 4 -- 12 p.m. in hours from 8 a.m.
def start_decrease_time : ℝ := 0 -- 12 p.m. in hours from 12 p.m.
def end_decrease_time : ℝ := 6 -- 6 p.m. in hours from 12 p.m.

theorem reservoir_water_level_at_6_pm :
  initial_water_level
  + (end_increase_time - start_increase_time) * increase_rate
  - (end_decrease_time - start_decrease_time) * decrease_rate
  = 45.6 :=
by
  sorry

end reservoir_water_level_at_6_pm_l528_52837


namespace prove_d_value_l528_52828

-- Definitions of the conditions
def d (x : ℝ) : ℝ := x^4 - 2*x^3 + x^2 - 12*x - 5

-- The statement to prove
theorem prove_d_value (x : ℝ) (h : x^2 - 2*x - 5 = 0) : d x = 25 :=
sorry

end prove_d_value_l528_52828


namespace zeros_in_square_of_999_999_999_l528_52892

noncomputable def number_of_zeros_in_square (n : ℕ) : ℕ :=
  if n ≥ 1 then n - 1 else 0

theorem zeros_in_square_of_999_999_999 :
  number_of_zeros_in_square 9 = 8 :=
sorry

end zeros_in_square_of_999_999_999_l528_52892


namespace find_P_l528_52840

noncomputable def parabola_vertex : ℝ × ℝ := (0, 0)
noncomputable def parabola_focus : ℝ × ℝ := (0, -1)
noncomputable def point_P : ℝ × ℝ := (20 * Real.sqrt 6, -120)
noncomputable def PF_distance : ℝ := 121

def parabola_equation (x y : ℝ) : Prop :=
  x^2 = -4 * y

def parabola_condition (x y : ℝ) : Prop :=
  (parabola_equation x y) ∧ 
  (Real.sqrt (x^2 + (y + 1)^2) = PF_distance)

theorem find_P : parabola_condition (point_P.1) (point_P.2) :=
by
  sorry

end find_P_l528_52840


namespace solve_for_b_l528_52855

theorem solve_for_b (b : ℚ) : 
  (∃ m1 m2 : ℚ, 3 * m1 - 2 * 1 + 4 = 0 ∧ 5 * m2 + b * 1 - 1 = 0 ∧ m1 * m2 = -1) → b = 15 / 2 :=
by
  sorry

end solve_for_b_l528_52855


namespace ellipse_equation_and_fixed_point_proof_l528_52867

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l528_52867


namespace no_valid_x_l528_52829

-- Definitions based on given conditions
variables {m n x : ℝ}
variables (hm : m > 0) (hn : n < 0)

-- Theorem statement
theorem no_valid_x (hm : m > 0) (hn : n < 0) :
  ¬ ∃ x, (x - m)^2 - (x - n)^2 = (m - n)^2 :=
by
  sorry

end no_valid_x_l528_52829


namespace pagoda_lights_l528_52890

/-- From afar, the magnificent pagoda has seven layers, with red lights doubling on each
ascending floor, totaling 381 lights. How many lights are there at the very top? -/
theorem pagoda_lights :
  ∃ x, (1 + 2 + 4 + 8 + 16 + 32 + 64) * x = 381 ∧ x = 3 :=
by
  sorry

end pagoda_lights_l528_52890


namespace geometric_sequence_problem_l528_52856

-- Step d) Rewrite the problem in Lean 4 statement
theorem geometric_sequence_problem 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (q : ℝ) 
  (h1 : ∀ n, n > 0 → a_n n = 1 * q^(n-1)) 
  (h2 : 1 + q + q^2 = 7)
  (h3 : 6 * 1 * q = 1 + 3 + 1 * q^2 + 4)
  :
  (∀ n, a_n n = 2^(n-1)) ∧ 
  (∀ n, T_n n = 4 - (n+2) / 2^(n-1)) :=
  sorry

end geometric_sequence_problem_l528_52856


namespace overtime_hours_l528_52870

theorem overtime_hours
  (regularPayPerHour : ℝ)
  (regularHours : ℝ)
  (totalPay : ℝ)
  (overtimeRate : ℝ) 
  (h1 : regularPayPerHour = 3)
  (h2 : regularHours = 40)
  (h3 : totalPay = 168)
  (h4 : overtimeRate = 2 * regularPayPerHour) :
  (totalPay - (regularPayPerHour * regularHours)) / overtimeRate = 8 :=
by
  sorry

end overtime_hours_l528_52870


namespace solution_set_for_f_geq_zero_l528_52874

theorem solution_set_for_f_geq_zero (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f3 : f 3 = 0) (h_cond : ∀ x : ℝ, x < 0 → x * (deriv f x) < f x) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | -3 < x ∧ x < 0} ∪ {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_for_f_geq_zero_l528_52874


namespace avg_temp_correct_l528_52814

-- Defining the temperatures for each day from March 1st to March 5th
def day_1_temp := 55.0
def day_2_temp := 59.0
def day_3_temp := 60.0
def day_4_temp := 57.0
def day_5_temp := 64.0

-- Calculating the average temperature
def avg_temp := (day_1_temp + day_2_temp + day_3_temp + day_4_temp + day_5_temp) / 5.0

-- Proving that the average temperature equals 59.0°F
theorem avg_temp_correct : avg_temp = 59.0 := sorry

end avg_temp_correct_l528_52814


namespace find_AD_length_l528_52816

variables (A B C D O : Point)
variables (BO OD AO OC AB AD : ℝ)

def quadrilateral_properties (BO OD AO OC AB : ℝ) (O : Point) : Prop :=
  BO = 3 ∧ OD = 9 ∧ AO = 5 ∧ OC = 2 ∧ AB = 7

theorem find_AD_length (h : quadrilateral_properties BO OD AO OC AB O) : AD = Real.sqrt 151 :=
by
  sorry

end find_AD_length_l528_52816


namespace base_rate_of_first_company_is_7_l528_52825

noncomputable def telephone_company_base_rate_proof : Prop :=
  ∃ (base_rate1 base_rate2 charge_per_minute1 charge_per_minute2 minutes : ℝ),
  base_rate1 = 7 ∧
  charge_per_minute1 = 0.25 ∧
  base_rate2 = 12 ∧
  charge_per_minute2 = 0.20 ∧
  minutes = 100 ∧
  (base_rate1 + charge_per_minute1 * minutes) =
  (base_rate2 + charge_per_minute2 * minutes) ∧
  base_rate1 = 7

theorem base_rate_of_first_company_is_7 :
  telephone_company_base_rate_proof :=
by
  -- The proof step will go here
  sorry

end base_rate_of_first_company_is_7_l528_52825


namespace sin_neg_30_eq_neg_one_half_l528_52891

theorem sin_neg_30_eq_neg_one_half :
  Real.sin (-30 * Real.pi / 180) = -1 / 2 := 
by
  sorry -- Proof is skipped

end sin_neg_30_eq_neg_one_half_l528_52891


namespace austin_needs_six_weeks_l528_52834

theorem austin_needs_six_weeks
  (work_rate: ℕ) (hours_monday hours_wednesday hours_friday: ℕ) (bicycle_cost: ℕ) 
  (weekly_hours: ℕ := hours_monday + hours_wednesday + hours_friday) 
  (weekly_earnings: ℕ := weekly_hours * work_rate) 
  (weeks_needed: ℕ := bicycle_cost / weekly_earnings):
  work_rate = 5 ∧ hours_monday = 2 ∧ hours_wednesday = 1 ∧ hours_friday = 3 ∧ bicycle_cost = 180 ∧ weeks_needed = 6 :=
by {
  sorry
}

end austin_needs_six_weeks_l528_52834


namespace sum_of_integers_l528_52885

theorem sum_of_integers (a b : ℤ) (h : (Int.sqrt (a - 2023) + |b + 2023| = 1)) : a + b = 1 ∨ a + b = -1 :=
by
  sorry

end sum_of_integers_l528_52885


namespace playground_perimeter_l528_52862

-- Defining the conditions
def length : ℕ := 100
def breadth : ℕ := 500
def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

-- The theorem to prove
theorem playground_perimeter : perimeter length breadth = 1200 := 
by
  -- The actual proof will be filled later
  sorry

end playground_perimeter_l528_52862


namespace pipe_fill_time_without_leak_l528_52820

theorem pipe_fill_time_without_leak (T : ℕ) :
  let pipe_with_leak_time := 10
  let leak_empty_time := 10
  ((1 / T : ℚ) - (1 / leak_empty_time) = (1 / pipe_with_leak_time)) →
  T = 5 := 
sorry

end pipe_fill_time_without_leak_l528_52820


namespace common_ratio_infinite_geometric_series_l528_52809

theorem common_ratio_infinite_geometric_series :
  let a₁ := (4 : ℚ) / 7
  let a₂ := (16 : ℚ) / 49
  let a₃ := (64 : ℚ) / 343
  let r := a₂ / a₁
  r = 4 / 7 :=
by
  sorry

end common_ratio_infinite_geometric_series_l528_52809


namespace selling_price_correct_l528_52812

-- Define the conditions
def cost_price : ℝ := 900
def gain_percentage : ℝ := 0.2222222222222222

-- Define the selling price calculation
def profit := cost_price * gain_percentage
def selling_price := cost_price + profit

-- The problem statement in Lean 4
theorem selling_price_correct : selling_price = 1100 := 
by
  -- Proof to be filled in later
  sorry

end selling_price_correct_l528_52812


namespace compound_carbon_atoms_l528_52896

-- Definition of data given in the problem.
def molecular_weight : ℕ := 60
def hydrogen_atoms : ℕ := 4
def oxygen_atoms : ℕ := 2
def carbon_atomic_weight : ℕ := 12
def hydrogen_atomic_weight : ℕ := 1
def oxygen_atomic_weight : ℕ := 16

-- Statement to prove the number of carbon atoms in the compound.
theorem compound_carbon_atoms : 
  (molecular_weight - (hydrogen_atoms * hydrogen_atomic_weight + oxygen_atoms * oxygen_atomic_weight)) / carbon_atomic_weight = 2 := 
by
  sorry

end compound_carbon_atoms_l528_52896


namespace train_length_proof_l528_52844

def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 5 / 18

theorem train_length_proof (speed_kmph : ℕ) (platform_length_m : ℕ) (crossing_time_s : ℕ) (speed_mps : ℕ) (distance_covered_m : ℕ) (train_length_m : ℕ) :
  speed_kmph = 72 →
  platform_length_m = 270 →
  crossing_time_s = 26 →
  speed_mps = convert_kmph_to_mps speed_kmph →
  distance_covered_m = speed_mps * crossing_time_s →
  train_length_m = distance_covered_m - platform_length_m →
  train_length_m = 250 :=
by
  intros h_speed h_platform h_time h_conv h_dist h_train_length
  sorry

end train_length_proof_l528_52844


namespace solve_for_x_l528_52839

theorem solve_for_x (x : ℝ) (h : (2 / 7) * (1 / 3) * x = 14) : x = 147 :=
sorry

end solve_for_x_l528_52839


namespace can_divide_cube_into_71_l528_52831

theorem can_divide_cube_into_71 : 
  ∃ (n : ℕ), n = 71 ∧ 
  (∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = f k + 7) ∧ f n = 71) :=
by
  sorry

end can_divide_cube_into_71_l528_52831


namespace solve_for_a_l528_52849

theorem solve_for_a (a x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
by sorry

end solve_for_a_l528_52849


namespace fewest_cookies_l528_52852

theorem fewest_cookies
  (area_art_cookies : ℝ)
  (area_roger_cookies : ℝ)
  (area_paul_cookies : ℝ)
  (area_trisha_cookies : ℝ)
  (h_art : area_art_cookies = 12)
  (h_roger : area_roger_cookies = 8)
  (h_paul : area_paul_cookies = 6)
  (h_trisha : area_trisha_cookies = 6)
  (dough : ℝ) :
  (dough / area_art_cookies) < (dough / area_roger_cookies) ∧
  (dough / area_art_cookies) < (dough / area_paul_cookies) ∧
  (dough / area_art_cookies) < (dough / area_trisha_cookies) := by
  sorry

end fewest_cookies_l528_52852


namespace no_integer_solutions_l528_52882

theorem no_integer_solutions :
  ∀ n m : ℤ, (n^2 + (n+1)^2 + (n+2)^2) ≠ m^2 :=
by
  intro n m
  sorry

end no_integer_solutions_l528_52882


namespace greatest_three_digit_divisible_by_3_5_6_l528_52899

theorem greatest_three_digit_divisible_by_3_5_6 : 
    ∃ n : ℕ, 
        (100 ≤ n ∧ n ≤ 999) ∧ 
        (∃ k₃ : ℕ, n = 3 * k₃) ∧ 
        (∃ k₅ : ℕ, n = 5 * k₅) ∧ 
        (∃ k₆ : ℕ, n = 6 * k₆) ∧ 
        (∀ m : ℕ, (100 ≤ m ∧ m ≤ 999) ∧ (∃ k₃ : ℕ, m = 3 * k₃) ∧ (∃ k₅ : ℕ, m = 5 * k₅) ∧ (∃ k₆ : ℕ, m = 6 * k₆) → m ≤ 990) := by
  sorry

end greatest_three_digit_divisible_by_3_5_6_l528_52899


namespace smallest_n_l528_52826

theorem smallest_n (n : ℕ) : 
  (25 * n = (Nat.lcm 10 (Nat.lcm 16 18)) → n = 29) :=
by sorry

end smallest_n_l528_52826


namespace remainder_equality_l528_52868

theorem remainder_equality (a b s t d : ℕ) (h1 : a > b) (h2 : a % d = s % d) (h3 : b % d = t % d) :
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d :=
by
  sorry

end remainder_equality_l528_52868


namespace solution_set_inequality_l528_52838

theorem solution_set_inequality (x : ℝ) : |5 - x| < |x - 2| + |7 - 2 * x| ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3.5 :=
by
  sorry

end solution_set_inequality_l528_52838


namespace henry_added_water_l528_52802

theorem henry_added_water (F : ℕ) (h2 : F = 32) (α β : ℚ) (h3 : α = 3/4) (h4 : β = 7/8) :
  (F * β) - (F * α) = 4 := by
  sorry

end henry_added_water_l528_52802


namespace darcy_commute_l528_52893

theorem darcy_commute (d w r t x time_walk train_time : ℝ) 
  (h1 : d = 1.5)
  (h2 : w = 3)
  (h3 : r = 20)
  (h4 : train_time = t + x)
  (h5 : time_walk = 15 + train_time)
  (h6 : time_walk = d / w * 60)  -- Time taken to walk in minutes
  (h7 : t = d / r * 60)  -- Time taken on train in minutes
  : x = 10.5 :=
sorry

end darcy_commute_l528_52893


namespace exists_special_N_l528_52823

open Nat

theorem exists_special_N :
  ∃ N : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 150 → N % i = 0 ∨ i = 127 ∨ i = 128) ∧ 
  ¬ (N % 127 = 0) ∧ ¬ (N % 128 = 0) :=
by
  sorry

end exists_special_N_l528_52823


namespace arithmetic_identity_l528_52813

theorem arithmetic_identity : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by
  sorry

end arithmetic_identity_l528_52813


namespace BC_at_least_17_l528_52817

-- Given conditions
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
-- Distances given
variables (AB AC EC BD BC : ℝ)
variables (AB_pos : AB = 7)
variables (AC_pos : AC = 15)
variables (EC_pos : EC = 9)
variables (BD_pos : BD = 26)
-- Triangle Inequalities
variables (triangle_ABC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], AC - AB < BC)
variables (triangle_DEC : ∀ {x y z : Type} [MetricSpace x] [MetricSpace y] [MetricSpace z], BD - EC < BC)

-- Proof statement
theorem BC_at_least_17 : BC ≥ 17 := by
  sorry

end BC_at_least_17_l528_52817


namespace geometric_sum_l528_52860

theorem geometric_sum 
  (a : ℕ → ℝ) (q : ℝ) (h1 : a 2 + a 4 = 32) (h2 : a 6 + a 8 = 16) 
  (h_seq : ∀ n, a (n+2) = a n * q ^ 2):
  a 10 + a 12 + a 14 + a 16 = 12 :=
by
  -- Proof needs to be written here
  sorry

end geometric_sum_l528_52860
