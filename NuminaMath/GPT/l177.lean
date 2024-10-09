import Mathlib

namespace find_a_purely_imaginary_z1_z2_l177_17749

noncomputable def z1 (a : ℝ) : ℂ := ⟨a^2 - 3, a + 5⟩
noncomputable def z2 (a : ℝ) : ℂ := ⟨a - 1, a^2 + 2 * a - 1⟩

theorem find_a_purely_imaginary_z1_z2 (a : ℝ)
    (h_imaginary : ∃ b : ℝ, z2 a - z1 a = ⟨0, b⟩) : 
    a = -1 :=
sorry

end find_a_purely_imaginary_z1_z2_l177_17749


namespace smallest_value_y_l177_17797

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end smallest_value_y_l177_17797


namespace florist_sold_roses_l177_17759

theorem florist_sold_roses (x : ℕ) (h1 : 5 - x + 34 = 36) : x = 3 :=
by sorry

end florist_sold_roses_l177_17759


namespace material_for_7_quilts_l177_17752

theorem material_for_7_quilts (x : ℕ) (h1 : ∀ y : ℕ, y = 7 * x) (h2 : 36 = 12 * x) : 7 * x = 21 := 
by 
  sorry

end material_for_7_quilts_l177_17752


namespace volume_in_region_l177_17727

def satisfies_conditions (x y : ℝ) : Prop :=
  |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15

def in_region (x y : ℝ) : Prop :=
  satisfies_conditions x y

theorem volume_in_region (x y p m n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (hn : n ≠ 0) (V : ℝ) 
  (hvol : V = (m * Real.pi) / (n * Real.sqrt p))
  (hprime : m.gcd n = 1 ∧ ¬(∃ k : ℕ, k^2 ∣ p ∧ k ≥ 2)) 
  (hpoints : ∀ (x y : ℝ), in_region x y → 3 * y - x = 15) : 
  m + n + p = 365 := 
sorry

end volume_in_region_l177_17727


namespace area_of_path_correct_l177_17746

noncomputable def area_of_path (length_field : ℝ) (width_field : ℝ) (path_width : ℝ) : ℝ :=
  let length_total := length_field + 2 * path_width
  let width_total := width_field + 2 * path_width
  let area_total := length_total * width_total
  let area_field := length_field * width_field
  area_total - area_field

theorem area_of_path_correct :
  area_of_path 75 55 3.5 = 959 := 
by
  sorry

end area_of_path_correct_l177_17746


namespace boards_cannot_be_covered_by_dominos_l177_17721

-- Definitions of the boards
def board_6x4 := (6 : ℕ) * (4 : ℕ)
def board_5x5 := (5 : ℕ) * (5 : ℕ)
def board_L_shaped := (5 : ℕ) * (5 : ℕ) - (2 : ℕ) * (2 : ℕ)
def board_3x7 := (3 : ℕ) * (7 : ℕ)
def board_plus_shaped := (3 : ℕ) * (3 : ℕ) + (1 : ℕ) * (3 : ℕ)

-- Definition to check if a board can't be covered by dominoes
def cannot_be_covered_by_dominos (n : ℕ) : Prop := n % 2 = 1

-- Theorem stating which specific boards cannot be covered by dominoes
theorem boards_cannot_be_covered_by_dominos :
  cannot_be_covered_by_dominos board_5x5 ∧
  cannot_be_covered_by_dominos board_L_shaped ∧
  cannot_be_covered_by_dominos board_3x7 :=
by
  -- Proof here
  sorry

end boards_cannot_be_covered_by_dominos_l177_17721


namespace average_value_is_2020_l177_17707

namespace CardsAverage

theorem average_value_is_2020 (n : ℕ) (h : (2020 * 3 * ((n * (n + 1)) + 2) = n * (n + 1) * (2 * n + 1) + 6 * (n + 1))) : n = 3015 := 
by
  sorry

end CardsAverage

end average_value_is_2020_l177_17707


namespace vector_sum_is_correct_l177_17773

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (0, 1)

-- Define the vectors AB and AC
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vectorAC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- State the theorem
theorem vector_sum_is_correct : vectorAB + vectorAC = (-3, -1) :=
by
  sorry

end vector_sum_is_correct_l177_17773


namespace constant_term_exists_l177_17768

theorem constant_term_exists (n : ℕ) (h : n = 6) : 
  (∃ r : ℕ, 2 * n - 3 * r = 0) ∧ 
  (∃ n' r' : ℕ, n' ≠ 6 ∧ 2 * n' - 3 * r' = 0) := by
  sorry

end constant_term_exists_l177_17768


namespace probability_colored_ball_l177_17713

theorem probability_colored_ball (total_balls blue_balls green_balls white_balls : ℕ)
  (h_total : total_balls = 40)
  (h_blue : blue_balls = 15)
  (h_green : green_balls = 5)
  (h_white : white_balls = 20)
  (h_disjoint : total_balls = blue_balls + green_balls + white_balls) :
  (blue_balls + green_balls) / total_balls = 1 / 2 := by
  -- Proof skipped
  sorry

end probability_colored_ball_l177_17713


namespace programmer_debugging_hours_l177_17704

theorem programmer_debugging_hours 
  (total_hours : ℕ)
  (flow_chart_fraction coding_fraction : ℚ)
  (flow_chart_fraction_eq : flow_chart_fraction = 1/4)
  (coding_fraction_eq : coding_fraction = 3/8)
  (hours_worked : total_hours = 48) :
  ∃ (debugging_hours : ℚ), debugging_hours = 18 := 
by
  sorry

end programmer_debugging_hours_l177_17704


namespace gcd_exponentiation_l177_17774

def m : ℕ := 2^2050 - 1
def n : ℕ := 2^2040 - 1

theorem gcd_exponentiation : Nat.gcd m n = 1023 := by
  sorry

end gcd_exponentiation_l177_17774


namespace range_of_b_l177_17783

theorem range_of_b (b : ℝ) : 
  (¬ (4 ≤ 3 * 3 + b) ∧ (4 ≤ 3 * 4 + b)) ↔ (-8 ≤ b ∧ b < -5) := 
by
  sorry

end range_of_b_l177_17783


namespace number_of_players_l177_17710

theorem number_of_players (n : ℕ) (G : ℕ) (h : G = 2 * n * (n - 1)) : n = 19 :=
by {
  sorry
}

end number_of_players_l177_17710


namespace employee_payment_l177_17769

theorem employee_payment (X Y : ℝ) (h1 : X + Y = 528) (h2 : X = 1.2 * Y) : Y = 240 :=
by
  sorry

end employee_payment_l177_17769


namespace inspection_arrangements_l177_17787

-- Definitions based on conditions
def liberal_arts_classes : ℕ := 2
def science_classes : ℕ := 3
def num_students (classes : ℕ) : ℕ := classes

-- Main theorem statement
theorem inspection_arrangements (liberal_arts_classes science_classes : ℕ)
  (h1: liberal_arts_classes = 2) (h2: science_classes = 3) : 
  num_students liberal_arts_classes * num_students science_classes = 24 :=
by {
  -- Given there are 2 liberal arts classes and 3 science classes,
  -- there are exactly 24 ways to arrange the inspections as per the conditions provided.
  sorry
}

end inspection_arrangements_l177_17787


namespace total_votes_cast_l177_17790

theorem total_votes_cast (V : ℝ) (h1 : ∃ x : ℝ, x = 0.31 * V) (h2 : ∃ y : ℝ, y = x + 2451) :
  V = 6450 :=
by
  sorry

end total_votes_cast_l177_17790


namespace part1_part2_l177_17742

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem part1 (a x : ℝ) (h : a > 0) : f a x + a / Real.exp 1 > 0 := by
  sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f (-1/2) x1 = f (-1/2) x2) : x1 + x2 < 2 := by
  sorry

end part1_part2_l177_17742


namespace f_2014_odd_f_2014_not_even_l177_17728

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 1 / x
| (n + 1), x => 1 / (x + f n x)

theorem f_2014_odd :
  ∀ x : ℝ, f 2014 x = - f 2014 (-x) :=
sorry

theorem f_2014_not_even :
  ∃ x : ℝ, f 2014 x ≠ f 2014 (-x) :=
sorry

end f_2014_odd_f_2014_not_even_l177_17728


namespace negation_q_sufficient_not_necessary_negation_p_l177_17782

theorem negation_q_sufficient_not_necessary_negation_p :
  (∃ x : ℝ, (∃ p : 16 - x^2 < 0, (x ∈ [-4, 4]))) →
  (∃ x : ℝ, (∃ q : x^2 + x - 6 > 0, (x ∈ [-3, 2]))) :=
sorry

end negation_q_sufficient_not_necessary_negation_p_l177_17782


namespace find_g_expression_l177_17700

theorem find_g_expression (f g : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = 2 * x + 3)
  (h2 : ∀ x : ℝ, g (x + 2) = f x) :
  ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end find_g_expression_l177_17700


namespace bacteria_original_count_l177_17711

theorem bacteria_original_count (current: ℕ) (increase: ℕ) (hc: current = 8917) (hi: increase = 8317) : current - increase = 600 :=
by
  sorry

end bacteria_original_count_l177_17711


namespace motorcycle_time_l177_17715

theorem motorcycle_time (v_m v_b d t_m : ℝ) 
  (h1 : 12 * v_m + 9 * v_b = d)
  (h2 : 21 * v_b + 8 * v_m = d)
  (h3 : v_m = 3 * v_b) :
  t_m = 15 :=
by
  sorry

end motorcycle_time_l177_17715


namespace right_triangle_height_l177_17720

theorem right_triangle_height
  (h : ℕ)
  (base : ℕ)
  (rectangle_area : ℕ)
  (same_area : (1 / 2 : ℚ) * base * h = rectangle_area)
  (base_eq_width : base = 5)
  (rectangle_area_eq : rectangle_area = 45) :
  h = 18 :=
by
  sorry

end right_triangle_height_l177_17720


namespace average_annual_population_increase_l177_17789

theorem average_annual_population_increase 
    (initial_population : ℝ) 
    (final_population : ℝ) 
    (years : ℝ) 
    (initial_population_pos : initial_population > 0) 
    (years_pos : years > 0)
    (initial_population_eq : initial_population = 175000) 
    (final_population_eq : final_population = 297500) 
    (years_eq : years = 10) : 
    (final_population - initial_population) / initial_population / years * 100 = 7 :=
by
    sorry

end average_annual_population_increase_l177_17789


namespace anne_distance_l177_17753

-- Definitions based on conditions
def Time : ℕ := 5
def Speed : ℕ := 4
def Distance : ℕ := Speed * Time

-- Proof statement
theorem anne_distance : Distance = 20 := by
  sorry

end anne_distance_l177_17753


namespace complex_modulus_square_l177_17786

open Complex

theorem complex_modulus_square (a b : ℝ) (h : 5 * (a + b * I) + 3 * Complex.abs (a + b * I) = 15 - 16 * I) :
  (Complex.abs (a + b * I))^2 = 256 / 25 :=
by sorry

end complex_modulus_square_l177_17786


namespace max_repeating_sequence_length_l177_17757

theorem max_repeating_sequence_length (p q n α β d : ℕ) (h_prime: Nat.gcd p q = 1)
  (hq : q = (2 ^ α) * (5 ^ β) * d) (hd_coprime: Nat.gcd d 10 = 1) (h_repeat: 10 ^ n ≡ 1 [MOD d]) :
  ∃ s, s ≤ n * (10 ^ n - 1) ∧ (10 ^ s ≡ 1 [MOD d^2]) :=
by
  sorry

end max_repeating_sequence_length_l177_17757


namespace greatest_multiple_of_5_and_6_less_than_1000_l177_17730

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end greatest_multiple_of_5_and_6_less_than_1000_l177_17730


namespace digits_sum_is_23_l177_17771

/-
Juan chooses a five-digit positive integer.
Maria erases the ones digit and gets a four-digit number.
The sum of this four-digit number and the original five-digit number is 52,713.
What can the sum of the five digits of the original number be?
-/

theorem digits_sum_is_23 (x y : ℕ) (h1 : 1000 ≤ x) (h2 : x ≤ 9999) (h3 : y ≤ 9) (h4 : 11 * x + y = 52713) :
  (x / 1000) + (x / 100 % 10) + (x / 10 % 10) + (x % 10) + y = 23 :=
by {
  sorry -- Proof goes here.
}

end digits_sum_is_23_l177_17771


namespace remainder_is_210_l177_17798

-- Define necessary constants and theorems
def x : ℕ := 2^35
def dividend : ℕ := 2^210 + 210
def divisor : ℕ := 2^105 + 2^63 + 1

theorem remainder_is_210 : (dividend % divisor) = 210 :=
by 
  -- Assume the calculation steps in the preceding solution are correct.
  -- No need to manually re-calculate as we've directly taken from the solution.
  sorry

end remainder_is_210_l177_17798


namespace quadratic_has_real_roots_iff_l177_17762

theorem quadratic_has_real_roots_iff (a : ℝ) :
  (∃ (x : ℝ), a * x^2 - 4 * x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) := by
  sorry

end quadratic_has_real_roots_iff_l177_17762


namespace ellipse_formula_max_area_triangle_l177_17745

-- Definitions for Ellipse part
def ellipse_eq (x y a : ℝ) := (x^2 / a^2) + (y^2 / 3) = 1
def eccentricity (a : ℝ) := (Real.sqrt (a^2 - 3)) / a = 1 / 2

-- Definition for Circle intersection part
def circle_intersection_cond (t : ℝ) := (0 < t) ∧ (t < (2 * Real.sqrt 21) / 7)

-- Main theorem for ellipse equation
theorem ellipse_formula (a : ℝ) (h1 : a > Real.sqrt 3) (h2 : eccentricity a) :
  ellipse_eq x y 2 :=
sorry

-- Main theorem for maximum area of triangle ABC
theorem max_area_triangle (t : ℝ) (h : circle_intersection_cond t) :
  ∃ S, S = (3 * Real.sqrt 7) / 7 :=
sorry

end ellipse_formula_max_area_triangle_l177_17745


namespace value_of_expression_l177_17734

theorem value_of_expression : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end value_of_expression_l177_17734


namespace chocolates_eaten_by_robert_l177_17763

theorem chocolates_eaten_by_robert (nickel_ate : ℕ) (robert_ate_more : ℕ) (H1 : nickel_ate = 3) (H2 : robert_ate_more = 4) :
  nickel_ate + robert_ate_more = 7 :=
by {
  sorry
}

end chocolates_eaten_by_robert_l177_17763


namespace correct_formula_for_xy_l177_17739

theorem correct_formula_for_xy :
  (∀ x y, (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) →
    y = x^2 + x + 1) :=
sorry

end correct_formula_for_xy_l177_17739


namespace find_sample_size_l177_17764

theorem find_sample_size : ∃ n : ℕ, n ∣ 36 ∧ (n + 1) ∣ 35 ∧ n = 6 := by
  use 6
  simp
  sorry

end find_sample_size_l177_17764


namespace income_after_selling_more_l177_17744

theorem income_after_selling_more (x y : ℝ)
  (h1 : 26 * x + 14 * y = 264) 
  : 39 * x + 21 * y = 396 := 
by 
  sorry

end income_after_selling_more_l177_17744


namespace garden_perimeter_is_44_l177_17777

-- Define the original garden's side length given the area
noncomputable def original_side_length (A : ℕ) := Nat.sqrt A

-- Given condition: Area of the original garden is 49 square meters
def original_area := 49

-- Define the new side length after expanding each side by 4 meters
def new_side_length (original_side : ℕ) := original_side + 4

-- Define the perimeter of the new garden given the new side length
def new_perimeter (new_side : ℕ) := 4 * new_side

-- Proof statement: The perimeter of the new garden given the original area is 44 meters
theorem garden_perimeter_is_44 : new_perimeter (new_side_length (original_side_length original_area)) = 44 := by
  -- This is where the proof would go
  sorry

end garden_perimeter_is_44_l177_17777


namespace find_initial_volume_l177_17729

noncomputable def initial_volume_of_solution (V : ℝ) : Prop :=
  let initial_jasmine := 0.05 * V
  let added_jasmine := 8
  let added_water := 2
  let new_total_volume := V + added_jasmine + added_water
  let new_jasmine := 0.125 * new_total_volume
  initial_jasmine + added_jasmine = new_jasmine

theorem find_initial_volume : ∃ V : ℝ, initial_volume_of_solution V ∧ V = 90 :=
by
  use 90
  unfold initial_volume_of_solution
  sorry

end find_initial_volume_l177_17729


namespace b6_b8_value_l177_17712

def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d
def nonzero_sequence (a : ℕ → ℕ) := ∀ n : ℕ, a n ≠ 0
def geometric_seq (b : ℕ → ℕ) := ∃ r : ℕ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℕ) (d : ℕ) 
  (h_arith : arithmetic_seq a) 
  (h_nonzero : nonzero_sequence a) 
  (h_cond1 : 2 * a 3 = a 1^2) 
  (h_cond2 : a 1 = d)
  (h_geo : geometric_seq b)
  (h_b13 : b 13 = a 2)
  (h_b1 : b 1 = a 1) :
  b 6 * b 8 = 72 := 
sorry

end b6_b8_value_l177_17712


namespace min_value_of_y_l177_17718

noncomputable def y (x : ℝ) : ℝ := x^2 + 26 * x + 7

theorem min_value_of_y : ∃ x : ℝ, y x = -162 :=
by
  use -13
  sorry

end min_value_of_y_l177_17718


namespace initial_amount_l177_17754

theorem initial_amount (cost_bread cost_butter cost_juice total_remain total_amount : ℕ) :
  cost_bread = 2 →
  cost_butter = 3 →
  cost_juice = 2 * cost_bread →
  total_remain = 6 →
  total_amount = cost_bread + cost_butter + cost_juice + total_remain →
  total_amount = 15 := by
  intros h_bread h_butter h_juice h_remain h_total
  sorry

end initial_amount_l177_17754


namespace count_valid_n_l177_17792

theorem count_valid_n : ∃ (n : ℕ), n < 200 ∧ (∃ (m : ℕ), (m % 4 = 0) ∧ (∃ (k : ℤ), n = 4 * k + 2 ∧ m = 4 * k * (k + 1))) ∧ (∃ k_range : ℕ, k_range = 50) :=
sorry

end count_valid_n_l177_17792


namespace alice_probability_l177_17791

noncomputable def probability_picking_exactly_three_green_marbles : ℚ :=
  let binom : ℚ := 35 -- binomial coefficient (7 choose 3)
  let prob_green : ℚ := 8 / 15 -- probability of picking a green marble
  let prob_purple : ℚ := 7 / 15 -- probability of picking a purple marble
  binom * (prob_green ^ 3) * (prob_purple ^ 4)

theorem alice_probability :
  probability_picking_exactly_three_green_marbles = 34454336 / 136687500 := by
  sorry

end alice_probability_l177_17791


namespace sum_real_imag_l177_17722

theorem sum_real_imag (z : ℂ) (hz : z = 3 - 4 * I) : z.re + z.im = -1 :=
by {
  -- Because the task asks for no proof, we're leaving it with 'sorry'.
  sorry
}

end sum_real_imag_l177_17722


namespace maria_cookies_left_l177_17766

theorem maria_cookies_left
    (total_cookies : ℕ) -- Maria has 60 cookies
    (friend_share : ℕ) -- 20% of the initial cookies goes to the friend
    (family_share : ℕ) -- 1/3 of the remaining cookies goes to the family
    (eaten_cookies : ℕ) -- Maria eats 4 cookies
    (neighbor_share : ℕ) -- Maria gives 1/6 of the remaining cookies to neighbor
    (initial_cookies : total_cookies = 60)
    (friend_fraction : friend_share = total_cookies * 20 / 100)
    (remaining_after_friend : ℕ := total_cookies - friend_share)
    (family_fraction : family_share = remaining_after_friend / 3)
    (remaining_after_family : ℕ := remaining_after_friend - family_share)
    (eaten : eaten_cookies = 4)
    (remaining_after_eating : ℕ := remaining_after_family - eaten_cookies)
    (neighbor_fraction : neighbor_share = remaining_after_eating / 6)
    (neighbor_integerized : neighbor_share = 4) -- assumed whole number for neighbor's share
    (remaining_after_neighbor : ℕ := remaining_after_eating - neighbor_share) : 
    remaining_after_neighbor = 24 :=
sorry  -- The statement matches the problem, proof is left out

end maria_cookies_left_l177_17766


namespace balloon_arrangement_count_l177_17735

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l177_17735


namespace structure_burns_in_65_seconds_l177_17756

noncomputable def toothpick_grid_burn_time (m n : ℕ) (toothpicks : ℕ) (burn_time : ℕ) : ℕ :=
  if (m = 3 ∧ n = 5 ∧ toothpicks = 38 ∧ burn_time = 10) then 65 else 0

theorem structure_burns_in_65_seconds : toothpick_grid_burn_time 3 5 38 10 = 65 := by
  sorry

end structure_burns_in_65_seconds_l177_17756


namespace minimum_additional_marbles_l177_17738

-- Definitions corresponding to the conditions
def friends := 12
def initial_marbles := 40

-- Sum of the first n natural numbers definition
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the necessary number of additional marbles
theorem minimum_additional_marbles (h1 : friends = 12) (h2 : initial_marbles = 40) : 
  ∃ additional_marbles, additional_marbles = sum_first_n friends - initial_marbles := by
  sorry

end minimum_additional_marbles_l177_17738


namespace total_surface_area_calc_l177_17740

/-- Given a cube with a total volume of 1 cubic foot, cut into four pieces by three parallel cuts:
1) The first cut is 0.4 feet from the top.
2) The second cut is 0.3 feet below the first.
3) The third cut is 0.1 feet below the second.
Prove that the total surface area of the new solid is 6 square feet. -/
theorem total_surface_area_calc :
  ∀ (A B C D : ℝ), 
    A = 0.4 → 
    B = 0.3 → 
    C = 0.1 → 
    D = 1 - (A + B + C) → 
    (6 : ℝ) = 6 := 
by 
  intros A B C D hA hB hC hD 
  sorry

end total_surface_area_calc_l177_17740


namespace height_of_brick_l177_17761

-- Definitions of wall dimensions
def L_w : ℝ := 700
def W_w : ℝ := 600
def H_w : ℝ := 22.5

-- Number of bricks
def n : ℝ := 5600

-- Definitions of brick dimensions (length and width)
def L_b : ℝ := 25
def W_b : ℝ := 11.25

-- Main theorem: Prove the height of each brick
theorem height_of_brick : ∃ h : ℝ, h = 6 :=
by
  -- Will add the proof steps here eventually
  sorry

end height_of_brick_l177_17761


namespace fraction_allocated_for_school_l177_17748

-- Conditions
def days_per_week : ℕ := 5
def hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 5
def allocation_for_school : ℕ := 75

-- Proof statement
theorem fraction_allocated_for_school :
  let weekly_hours := days_per_week * hours_per_day
  let weekly_earnings := weekly_hours * earnings_per_hour
  allocation_for_school / weekly_earnings = 3 / 4 := 
by
  sorry

end fraction_allocated_for_school_l177_17748


namespace polynomial_square_binomial_l177_17794

-- Define the given polynomial and binomial
def polynomial (x : ℚ) (a : ℚ) : ℚ :=
  25 * x^2 + 40 * x + a

def binomial (x b : ℚ) : ℚ :=
  (5 * x + b)^2

-- Theorem to state the problem
theorem polynomial_square_binomial (a : ℚ) : 
  (∃ b, polynomial x a = binomial x b) ↔ a = 16 :=
by
  sorry

end polynomial_square_binomial_l177_17794


namespace remainder_9_plus_y_mod_31_l177_17772

theorem remainder_9_plus_y_mod_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (9 + y) % 31 = 18 :=
sorry

end remainder_9_plus_y_mod_31_l177_17772


namespace rectangle_sides_equal_perimeter_and_area_l177_17737

theorem rectangle_sides_equal_perimeter_and_area (x y : ℕ) (h : 2 * x + 2 * y = x * y) : 
    (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 4) :=
by sorry

end rectangle_sides_equal_perimeter_and_area_l177_17737


namespace count_integer_values_l177_17770

-- Statement of the problem in Lean 4
theorem count_integer_values (x : ℤ) : 
  (7 * x^2 + 23 * x + 20 ≤ 30) → 
  ∃ (n : ℕ), n = 6 :=
sorry

end count_integer_values_l177_17770


namespace jackson_souvenirs_total_l177_17747

def jacksons_collections := 
  let hermit_crabs := 120
  let spiral_shells_per_hermit_crab := 8
  let starfish_per_spiral_shell := 5
  let sand_dollars_per_starfish := 3
  let coral_structures_per_sand_dollars := 4
  let spiral_shells := hermit_crabs * spiral_shells_per_hermit_crab
  let starfish := spiral_shells * starfish_per_spiral_shell
  let sand_dollars := starfish * sand_dollars_per_starfish
  let coral_structures := sand_dollars / coral_structures_per_sand_dollars
  hermit_crabs + spiral_shells + starfish + sand_dollars + coral_structures

theorem jackson_souvenirs_total : jacksons_collections = 22880 := by sorry

end jackson_souvenirs_total_l177_17747


namespace product_of_integers_between_sqrt_115_l177_17714

theorem product_of_integers_between_sqrt_115 :
  ∃ a b : ℕ, 100 < 115 ∧ 115 < 121 ∧ a = 10 ∧ b = 11 ∧ a * b = 110 := by
  sorry

end product_of_integers_between_sqrt_115_l177_17714


namespace pencils_total_l177_17733

theorem pencils_total (original_pencils : ℕ) (added_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : original_pencils = 41) 
  (h2 : added_pencils = 30) 
  (h3 : total_pencils = original_pencils + added_pencils) : 
  total_pencils = 71 := 
by
  sorry

end pencils_total_l177_17733


namespace find_m_l177_17716

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the function to calculate m * a + b
def m_a_plus_b (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3 * m + 2)

-- Define the vector a - 2 * b
def a_minus_2b : ℝ × ℝ := (4, -1)

-- Define the condition for parallelism
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The theorem that states the equivalence
theorem find_m (m : ℝ) (H : parallel (m_a_plus_b m) a_minus_2b) : m = -1/2 :=
by
  sorry

end find_m_l177_17716


namespace geometric_seq_sum_l177_17779

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, S n = a 0 * (1 - q^n) / (1 - q))
    (h2 : S 10 = 10) (h3 : S 30 = 70) (hq_pos : 0 < q) :
    S 40 = 150 := by
  sorry

end geometric_seq_sum_l177_17779


namespace count_triangles_in_figure_l177_17767

-- Define the structure of the grid with the given properties.
def grid_structure : Prop :=
  ∃ (n1 n2 n3 n4 : ℕ), 
  n1 = 3 ∧  -- First row: 3 small triangles
  n2 = 2 ∧  -- Second row: 2 small triangles
  n3 = 1 ∧  -- Third row: 1 small triangle
  n4 = 1    -- 1 large inverted triangle

-- The problem statement
theorem count_triangles_in_figure (h : grid_structure) : 
  ∃ (total_triangles : ℕ), total_triangles = 9 :=
sorry

end count_triangles_in_figure_l177_17767


namespace tiles_finite_initial_segment_l177_17780

theorem tiles_finite_initial_segment (S : ℕ → Prop) (hTiling : ∀ n : ℕ, ∃ m : ℕ, m ≥ n ∧ S m) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → S n :=
by
  sorry

end tiles_finite_initial_segment_l177_17780


namespace find_ordered_pairs_l177_17796

theorem find_ordered_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m ∣ 3 * n - 2 ∧ 2 * n ∣ 3 * m - 2) ↔ (m, n) = (2, 2) ∨ (m, n) = (10, 14) ∨ (m, n) = (14, 10) :=
by
  sorry

end find_ordered_pairs_l177_17796


namespace newspaper_target_l177_17765

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end newspaper_target_l177_17765


namespace find_m_range_l177_17709

-- Definitions
def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3
def q (x m : ℝ) (h : m > 0) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0

-- Problem Statement
theorem find_m_range : 
  (∀ (x : ℝ) (h : m > 0), (¬ (p x)) → (¬ (q x m h))) ∧ 
  (∃ (x : ℝ), ¬ (p x) ∧ ¬ (q x m h)) → 
  ∃ (m : ℝ), m ≥ 3 := 
sorry

end find_m_range_l177_17709


namespace shopkeeper_profit_percentage_l177_17731

theorem shopkeeper_profit_percentage (P : ℝ) : (70 / 100) * (1 + P / 100) = 1 → P = 700 / 3 :=
by
  sorry

end shopkeeper_profit_percentage_l177_17731


namespace solve_for_first_expedition_weeks_l177_17743

-- Define the variables according to the given conditions.
variables (x : ℕ)
variables (days_in_week : ℕ := 7)
variables (total_days_on_island : ℕ := 126)

-- Define the total number of weeks spent on the expeditions.
def total_weeks_on_expeditions (x : ℕ) : ℕ := 
  x + (x + 2) + 2 * (x + 2)

-- Convert total days to weeks.
def total_weeks := total_days_on_island / days_in_week

-- Prove the equation
theorem solve_for_first_expedition_weeks
  (h : total_weeks_on_expeditions x = total_weeks):
  x = 3 :=
by
  sorry

end solve_for_first_expedition_weeks_l177_17743


namespace horizontal_distance_l177_17784

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l177_17784


namespace lines_intersect_l177_17788

def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 4 * v, 9 - v)

theorem lines_intersect :
  ∃ s v : ℚ, (line1 s) = (line2 v) ∧ (line1 s) = (-17/5, 53/5) := 
sorry

end lines_intersect_l177_17788


namespace apple_eating_contest_l177_17793

theorem apple_eating_contest (a z : ℕ) (h_most : a = 8) (h_fewest : z = 1) : a - z = 7 :=
by
  sorry

end apple_eating_contest_l177_17793


namespace circles_positional_relationship_l177_17701

theorem circles_positional_relationship
  (r1 r2 d : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 5)
  (h3 : d = 3) :
  d < r2 - r1 := 
by
  sorry

end circles_positional_relationship_l177_17701


namespace multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l177_17705

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem multiple_of_4 : y % 4 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_8 : y % 8 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_16 : y % 16 = 0 := by
  -- proof needed
  sorry

theorem not_multiple_of_32 : y % 32 ≠ 0 := by
  -- proof needed
  sorry

end multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l177_17705


namespace jim_can_bake_loaves_l177_17724

-- Define the amounts of flour in different locations
def flour_cupboard : ℕ := 200  -- in grams
def flour_counter : ℕ := 100   -- in grams
def flour_pantry : ℕ := 100    -- in grams

-- Define the amount of flour required for one loaf of bread
def flour_per_loaf : ℕ := 200  -- in grams

-- Total loaves Jim can bake
def loaves_baked (f_c f_k f_p f_r : ℕ) : ℕ :=
  (f_c + f_k + f_p) / f_r

-- Theorem to prove the solution
theorem jim_can_bake_loaves :
  loaves_baked flour_cupboard flour_counter flour_pantry flour_per_loaf = 2 :=
by
  -- Proof is omitted
  sorry

end jim_can_bake_loaves_l177_17724


namespace ratio_of_second_to_first_l177_17755

noncomputable def building_heights (H1 H2 H3 : ℝ) : Prop :=
  H1 = 600 ∧ H3 = 3 * (H1 + H2) ∧ H1 + H2 + H3 = 7200

theorem ratio_of_second_to_first (H1 H2 H3 : ℝ) (h : building_heights H1 H2 H3) :
  H1 ≠ 0 → (H2 / H1 = 2) :=
by
  unfold building_heights at h
  rcases h with ⟨h1, h3, h_total⟩
  sorry -- Steps of solving are skipped

end ratio_of_second_to_first_l177_17755


namespace div_binomial_expansion_l177_17750

theorem div_binomial_expansion
  (a n b : Nat)
  (hb : a^n ∣ b) :
  a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end div_binomial_expansion_l177_17750


namespace perfect_square_trinomial_l177_17778

theorem perfect_square_trinomial (m : ℤ) (h : ∃ b : ℤ, (x : ℤ) → x^2 - 10 * x + m = (x + b)^2) : m = 25 :=
sorry

end perfect_square_trinomial_l177_17778


namespace fly_least_distance_l177_17758

noncomputable def leastDistance (r : ℝ) (h : ℝ) (start_dist : ℝ) (end_dist : ℝ) : ℝ := 
  let C := 2 * Real.pi * r
  let R := Real.sqrt (r^2 + h^2)
  let θ := C / R
  let A := (start_dist, 0)
  let B := (Real.cos (θ / 2) * end_dist, Real.sin (θ / 2) * end_dist)
  Real.sqrt ((B.fst - A.fst)^2 + (B.snd - A.snd)^2)

theorem fly_least_distance : 
  leastDistance 600 (200 * Real.sqrt 7) 125 (375 * Real.sqrt 2) = 625 := 
sorry

end fly_least_distance_l177_17758


namespace cartesian_to_polar_curve_C_l177_17732

theorem cartesian_to_polar_curve_C (x y : ℝ) (θ ρ : ℝ) 
  (h1 : x = ρ * Real.cos θ)
  (h2 : y = ρ * Real.sin θ)
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * Real.cos θ :=
sorry

end cartesian_to_polar_curve_C_l177_17732


namespace gcd_polynomial_l177_17725

theorem gcd_polynomial (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^5 + 125) (n + 5) = if n % 5 = 0 then 5 else 1 :=
by
  sorry

end gcd_polynomial_l177_17725


namespace expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l177_17775

-- Proof for (x + 3y)^2 = x^2 + 6xy + 9y^2
theorem expand_x_plus_3y_squared (x y : ℝ) : 
  (x + 3 * y) ^ 2 = x ^ 2 + 6 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (2x + 3y)^2 = 4x^2 + 12xy + 9y^2
theorem expand_2x_plus_3y_squared (x y : ℝ) : 
  (2 * x + 3 * y) ^ 2 = 4 * x ^ 2 + 12 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (m^3 + n^5)^2 = m^6 + 2m^3n^5 + n^10
theorem expand_m3_plus_n5_squared (m n : ℝ) : 
  (m ^ 3 + n ^ 5) ^ 2 = m ^ 6 + 2 * m ^ 3 * n ^ 5 + n ^ 10 := 
  sorry

-- Proof for (5x - 3y)^2 = 25x^2 - 30xy + 9y^2
theorem expand_5x_minus_3y_squared (x y : ℝ) : 
  (5 * x - 3 * y) ^ 2 = 25 * x ^ 2 - 30 * x * y + 9 * y ^ 2 := 
  sorry

-- Proof for (3m^5 - 4n^2)^2 = 9m^10 - 24m^5n^2 + 16n^4
theorem expand_3m5_minus_4n2_squared (m n : ℝ) : 
  (3 * m ^ 5 - 4 * n ^ 2) ^ 2 = 9 * m ^ 10 - 24 * m ^ 5 * n ^ 2 + 16 * n ^ 4 := 
  sorry

end expand_x_plus_3y_squared_expand_2x_plus_3y_squared_expand_m3_plus_n5_squared_expand_5x_minus_3y_squared_expand_3m5_minus_4n2_squared_l177_17775


namespace value_of_w_l177_17760

-- Define the positivity of w
def positive_integer (w : ℕ) := w > 0

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Define the function which encapsulates the problem
def problem_condition (w : ℕ) := sum_of_digits (10^w - 74)

-- The main proof problem
theorem value_of_w (w : ℕ) (h : positive_integer w) : problem_condition w = 17 :=
by
  sorry

end value_of_w_l177_17760


namespace isosceles_triangle_k_value_l177_17723

theorem isosceles_triangle_k_value 
(side1 : ℝ)
(side2 side3 : ℝ)
(k : ℝ)
(h1 : side1 = 3 ∨ side2 = 3 ∨ side3 = 3)
(h2 : side1 = side2 ∨ side1 = side3 ∨ side2 = side3)
(h3 : Polynomial.eval side1 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side2 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side3 (Polynomial.C k + Polynomial.X ^ 2) = 0) :
k = 3 ∨ k = 4 :=
sorry

end isosceles_triangle_k_value_l177_17723


namespace simplify_expression_l177_17719

theorem simplify_expression (x : ℝ) : 
  (4 * x + 6 * x^3 + 8 - (3 - 6 * x^3 - 4 * x)) = 12 * x^3 + 8 * x + 5 := 
by
  sorry

end simplify_expression_l177_17719


namespace length_of_room_l177_17708

theorem length_of_room 
  (width : ℝ) (cost : ℝ) (rate : ℝ) (area : ℝ) (length : ℝ) 
  (h1 : width = 3.75) 
  (h2 : cost = 24750) 
  (h3 : rate = 1200) 
  (h4 : area = cost / rate) 
  (h5 : area = length * width) : 
  length = 5.5 :=
sorry

end length_of_room_l177_17708


namespace lcm_18_30_l177_17795

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l177_17795


namespace max_value_of_expression_l177_17736

-- Define the variables and condition.
variable (x y z : ℝ)
variable (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)

-- State the theorem.
theorem max_value_of_expression :
  (8 * x + 5 * y + 15 * z) ≤ 4.54 :=
sorry

end max_value_of_expression_l177_17736


namespace minimum_a_l177_17706

noncomputable def f (x : ℝ) := x - Real.exp (x - Real.exp 1)

theorem minimum_a (a : ℝ) (x1 x2 : ℝ) (hx : x2 - x1 ≥ Real.exp 1)
  (hy : Real.exp x1 = 1 + Real.log (x2 - a)) : a ≥ Real.exp 1 - 1 :=
by
  sorry

end minimum_a_l177_17706


namespace club_positions_l177_17703

def num_ways_to_fill_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

theorem club_positions : num_ways_to_fill_positions 12 = 665280 := by 
  sorry

end club_positions_l177_17703


namespace joe_fruit_probability_l177_17717

theorem joe_fruit_probability :
  let prob_same := (1 / 4) ^ 3
  let total_prob_same := 4 * prob_same
  let prob_diff := 1 - total_prob_same
  prob_diff = 15 / 16 :=
by
  sorry

end joe_fruit_probability_l177_17717


namespace range_of_t_for_obtuse_triangle_l177_17776

def is_obtuse_triangle (a b c : ℝ) : Prop := ∃t : ℝ, a = t - 1 ∧ b = t + 1 ∧ c = t + 3

theorem range_of_t_for_obtuse_triangle :
  ∀ t : ℝ, is_obtuse_triangle (t-1) (t+1) (t+3) → (3 < t ∧ t < 7) :=
by
  intros t ht
  sorry

end range_of_t_for_obtuse_triangle_l177_17776


namespace greatest_integer_part_expected_winnings_l177_17726

noncomputable def expected_winnings_one_envelope : ℝ := 500

noncomputable def expected_winnings_two_envelopes : ℝ := 625

noncomputable def expected_winnings_three_envelopes : ℝ := 695.3125

theorem greatest_integer_part_expected_winnings :
  ⌊expected_winnings_three_envelopes⌋ = 695 :=
by 
  sorry

end greatest_integer_part_expected_winnings_l177_17726


namespace circles_intersect_l177_17785

def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 3 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem circles_intersect :
  (∃ (x y : ℝ), circle_eq1 x y ∧ circle_eq2 x y) :=
sorry

end circles_intersect_l177_17785


namespace danil_claim_false_l177_17751

theorem danil_claim_false (E O : ℕ) (hE : E % 2 = 0) (hO : O % 2 = 0) (h : O = E + 15) : false :=
by sorry

end danil_claim_false_l177_17751


namespace gcd_increase_by_9_l177_17781

theorem gcd_increase_by_9 (m n d : ℕ) (h1 : d = Nat.gcd m n) (h2 : 9 * d = Nat.gcd (m + 6) n) : d = 3 ∨ d = 6 :=
by
  sorry

end gcd_increase_by_9_l177_17781


namespace blake_change_l177_17799

def cost_oranges : ℕ := 40
def cost_apples : ℕ := 50
def cost_mangoes : ℕ := 60
def initial_money : ℕ := 300

def total_cost : ℕ := cost_oranges + cost_apples + cost_mangoes
def change : ℕ := initial_money - total_cost

theorem blake_change : change = 150 := by
  sorry

end blake_change_l177_17799


namespace clea_ride_time_l177_17702

-- Definitions: Let c be Clea's walking speed without the bag and s be the speed of the escalator

variables (c s : ℝ)

-- Conditions translated into equations
def distance_without_bag := 80 * c
def distance_with_bag_and_escalator := 38 * (0.7 * c + s)

-- The problem: Prove that the time t for Clea to ride down the escalator while just standing on it with the bag is 57 seconds.
theorem clea_ride_time :
  (38 * (0.7 * c + s) = 80 * c) ->
  (t = 80 * (38 / 53.4)) ->
  t = 57 :=
sorry

end clea_ride_time_l177_17702


namespace different_total_scores_l177_17741

noncomputable def basket_scores (x y z : ℕ) : ℕ := x + 2 * y + 3 * z

def total_baskets := 7
def score_range := {n | 7 ≤ n ∧ n ≤ 21}

theorem different_total_scores : 
  ∃ (count : ℕ), count = 15 ∧ 
  ∀ n ∈ score_range, ∃ (x y z : ℕ), x + y + z = total_baskets ∧ basket_scores x y z = n :=
sorry

end different_total_scores_l177_17741
