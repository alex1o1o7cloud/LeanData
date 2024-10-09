import Mathlib

namespace probability_of_color_difference_l971_97127

noncomputable def probability_of_different_colors (n m : ℕ) : ℚ :=
  (Nat.choose n m : ℚ) * (1/2)^n

theorem probability_of_color_difference :
  probability_of_different_colors 8 4 = 35/128 :=
by
  sorry

end probability_of_color_difference_l971_97127


namespace total_flowers_l971_97140

def pieces (f : String) : Nat :=
  if f == "roses" ∨ f == "lilies" ∨ f == "sunflowers" ∨ f == "daisies" then 40 else 0

theorem total_flowers : 
  pieces "roses" + pieces "lilies" + pieces "sunflowers" + pieces "daisies" = 160 := 
by
  sorry


end total_flowers_l971_97140


namespace total_eggs_l971_97193

theorem total_eggs (eggs_today eggs_yesterday : ℕ) (h_today : eggs_today = 30) (h_yesterday : eggs_yesterday = 19) : eggs_today + eggs_yesterday = 49 :=
by
  sorry

end total_eggs_l971_97193


namespace festival_second_day_attendance_l971_97139

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end festival_second_day_attendance_l971_97139


namespace alex_buys_15_pounds_of_corn_l971_97149

theorem alex_buys_15_pounds_of_corn:
  ∃ (c b : ℝ), c + b = 30 ∧ 1.20 * c + 0.60 * b = 27.00 ∧ c = 15.0 :=
by
  sorry

end alex_buys_15_pounds_of_corn_l971_97149


namespace tins_per_case_is_24_l971_97197

def total_cases : ℕ := 15
def damaged_percentage : ℝ := 0.05
def remaining_tins : ℕ := 342

theorem tins_per_case_is_24 (x : ℕ) (h : (1 - damaged_percentage) * (total_cases * x) = remaining_tins) : x = 24 :=
  sorry

end tins_per_case_is_24_l971_97197


namespace sugar_used_in_two_minutes_l971_97190

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l971_97190


namespace arithmetic_sequence_line_l971_97162

theorem arithmetic_sequence_line (A B C x y : ℝ) :
  (2 * B = A + C) → (A * 1 + B * -2 + C = 0) :=
by
  intros h
  sorry

end arithmetic_sequence_line_l971_97162


namespace number_of_paintings_l971_97131

def is_valid_painting (grid : Matrix (Fin 3) (Fin 3) Bool) : Prop :=
  ∀ i j, grid i j = true → 
    (∀ k, k.succ < 3 → grid k j = true → ¬ grid (k.succ) j = false) ∧
    (∀ l, l.succ < 3 → grid i l = true → ¬ grid i (l.succ) = false)

theorem number_of_paintings : 
  ∃ n, n = 50 ∧ 
       ∃ f : Finset (Matrix (Fin 3) (Fin 3) Bool), 
         (∀ grid ∈ f, is_valid_painting grid) ∧ 
         Finset.card f = n :=
sorry

end number_of_paintings_l971_97131


namespace line_plane_intersection_l971_97183

theorem line_plane_intersection :
  (∃ t : ℝ, (x, y, z) = (3 + t, 1 - t, -5) ∧ (3 + t) + 7 * (1 - t) + 3 * (-5) + 11 = 0) →
  (x, y, z) = (4, 0, -5) :=
sorry

end line_plane_intersection_l971_97183


namespace twelve_edge_cubes_painted_faces_l971_97122

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces_l971_97122


namespace range_of_eccentricity_l971_97116

theorem range_of_eccentricity
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)
  (h4 : c^2 - b^2 + a * c < 0) :
  0 < c / a ∧ c / a < 1 / 2 :=
sorry

end range_of_eccentricity_l971_97116


namespace minimum_value_of_ratio_l971_97106

theorem minimum_value_of_ratio 
  {a b c : ℝ} (h_a : a ≠ 0) 
  (h_f'0 : 2 * a * 0 + b > 0)
  (h_f_nonneg : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  (∃ x : ℝ, a * x^2 + b * x + c ≥ 0) ∧ (1 + (a + c) / b = 2) := sorry

end minimum_value_of_ratio_l971_97106


namespace seat_arrangement_l971_97166

theorem seat_arrangement (seats : ℕ) (people : ℕ) (min_empty_between : ℕ) : 
  seats = 9 ∧ people = 3 ∧ min_empty_between = 2 → 
  ∃ ways : ℕ, ways = 60 :=
by
  intro h
  sorry

end seat_arrangement_l971_97166


namespace next_in_step_distance_l971_97161

theorem next_in_step_distance
  (jack_stride jill_stride : ℕ)
  (h1 : jack_stride = 64)
  (h2 : jill_stride = 56) :
  Nat.lcm jack_stride jill_stride = 448 := by
  sorry

end next_in_step_distance_l971_97161


namespace max_abs_value_inequality_l971_97118

theorem max_abs_value_inequality (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ (a b : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) ∧ |20 * a + 14 * b| + |20 * a - 14 * b| = 80 := 
sorry

end max_abs_value_inequality_l971_97118


namespace melanie_missed_games_l971_97187

-- Define the total number of soccer games played and the number attended by Melanie
def total_games : ℕ := 64
def attended_games : ℕ := 32

-- Statement to be proven
theorem melanie_missed_games : total_games - attended_games = 32 := by
  -- Placeholder for the proof
  sorry

end melanie_missed_games_l971_97187


namespace probability_of_at_least_one_l971_97134

theorem probability_of_at_least_one (P_1 P_2 : ℝ) (h1 : 0 ≤ P_1 ∧ P_1 ≤ 1) (h2 : 0 ≤ P_2 ∧ P_2 ≤ 1) :
  1 - (1 - P_1) * (1 - P_2) = P_1 + P_2 - P_1 * P_2 :=
by
  sorry

end probability_of_at_least_one_l971_97134


namespace unique_corresponding_point_l971_97182

-- Define the points for the squares
structure Point := (x : ℝ) (y : ℝ)

structure Square :=
  (a b c d : Point)

def contains (sq1 sq2: Square) : Prop :=
  sq2.a.x >= sq1.a.x ∧ sq2.a.y >= sq1.a.y ∧
  sq2.b.x <= sq1.b.x ∧ sq2.b.y >= sq1.b.y ∧
  sq2.c.x <= sq1.c.x ∧ sq2.c.y <= sq1.c.y ∧
  sq2.d.x >= sq1.d.x ∧ sq2.d.y <= sq1.d.y

theorem unique_corresponding_point
  (sq1 sq2 : Square)
  (h1 : contains sq1 sq2)
  (h2 : sq1.a.x - sq1.c.x = sq2.a.x - sq2.c.x ∧ sq1.a.y - sq1.c.y = sq2.a.y - sq2.c.y):
  ∃! (O : Point), ∃ O' : Point, contains sq1 sq2 ∧ 
  (O.x - sq1.a.x) / (sq1.b.x - sq1.a.x) = (O'.x - sq2.a.x) / (sq2.b.x - sq2.a.x) ∧ 
  (O.y - sq1.a.y) / (sq1.d.y - sq1.a.y) = (O'.y - sq2.a.y) / (sq2.d.y - sq2.a.y) := 
sorry

end unique_corresponding_point_l971_97182


namespace min_average_annual_growth_rate_l971_97102

theorem min_average_annual_growth_rate (M : ℝ) (x : ℝ) (h : M * (1 + x)^2 = 2 * M) : x = Real.sqrt 2 - 1 :=
by
  sorry

end min_average_annual_growth_rate_l971_97102


namespace fred_grew_38_cantelopes_l971_97137

def total_cantelopes : Nat := 82
def tim_cantelopes : Nat := 44
def fred_cantelopes : Nat := total_cantelopes - tim_cantelopes

theorem fred_grew_38_cantelopes : fred_cantelopes = 38 :=
by
  sorry

end fred_grew_38_cantelopes_l971_97137


namespace min_green_beads_l971_97133

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end min_green_beads_l971_97133


namespace function_symmetric_and_monotonic_l971_97124

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^4 - 2 * Real.sin x * Real.cos x - (Real.sin x)^4

theorem function_symmetric_and_monotonic :
  (∀ x, f (x + (3/8) * π) = f (x - (3/8) * π)) ∧
  (∀ x y, x ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → y ∈  Set.Icc (-(π / 8)) ((3 * π) / 8) → x < y → f x > f y) :=
by
  sorry

end function_symmetric_and_monotonic_l971_97124


namespace max_value_of_reciprocal_powers_l971_97189

variable {R : Type*} [CommRing R]
variables (s q r₁ r₂ : R)

-- Condition: the roots of the polynomial
def is_roots_of_polynomial (s q r₁ r₂ : R) : Prop :=
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ (r₁ + r₂ = r₁ ^ 2 + r₂ ^ 2) ∧ (r₁ + r₂ = r₁^10 + r₂^10)

-- The theorem that needs to be proven
theorem max_value_of_reciprocal_powers (s q r₁ r₂ : ℝ) (h : is_roots_of_polynomial s q r₁ r₂):
  (∃ r₁ r₂, r₁ + r₂ = s ∧ r₁ * r₂ = q ∧
             r₁ + r₂ = r₁^2 + r₂^2 ∧
             r₁ + r₂ = r₁^10 + r₂^10) →
  (r₁^ 11 ≠ 0 ∧ r₂^11 ≠ 0 ∧
  ((1 / r₁^11) + (1 / r₂^11) = 2)) :=
by
  sorry

end max_value_of_reciprocal_powers_l971_97189


namespace aira_rubber_bands_l971_97119

theorem aira_rubber_bands (total_bands : ℕ) (bands_each : ℕ) (samantha_extra : ℕ) (aira_fewer : ℕ)
  (h1 : total_bands = 18) 
  (h2 : bands_each = 6) 
  (h3 : samantha_extra = 5) 
  (h4 : aira_fewer = 1) : 
  ∃ x : ℕ, x + (x + samantha_extra) + (x + aira_fewer) = total_bands ∧ x = 4 :=
by
  sorry

end aira_rubber_bands_l971_97119


namespace num_positive_divisors_of_720_multiples_of_5_l971_97159

theorem num_positive_divisors_of_720_multiples_of_5 :
  (∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ c = 1) →
  ∃ (n : ℕ), n = 15 :=
by
  -- Proof will go here
  sorry

end num_positive_divisors_of_720_multiples_of_5_l971_97159


namespace smallest_number_is_correct_largest_number_is_correct_l971_97170

def initial_sequence := "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960"

def remove_digits (n : ℕ) (s : String) : String := sorry  -- Placeholder function for removing n digits

noncomputable def smallest_number_after_removal (s : String) : String :=
  -- Function to find the smallest number possible after removing digits
  remove_digits 100 s

noncomputable def largest_number_after_removal (s : String) : String :=
  -- Function to find the largest number possible after removing digits
  remove_digits 100 s

theorem smallest_number_is_correct : smallest_number_after_removal initial_sequence = "123450" :=
  sorry

theorem largest_number_is_correct : largest_number_after_removal initial_sequence = "56758596049" :=
  sorry

end smallest_number_is_correct_largest_number_is_correct_l971_97170


namespace dvd_count_correct_l971_97101

def total_dvds (store_dvds online_dvds : Nat) : Nat :=
  store_dvds + online_dvds

theorem dvd_count_correct :
  total_dvds 8 2 = 10 :=
by
  sorry

end dvd_count_correct_l971_97101


namespace Romeo_bars_of_chocolate_l971_97142

theorem Romeo_bars_of_chocolate 
  (cost_per_bar : ℕ) (packaging_cost : ℕ) (total_sale : ℕ) (profit : ℕ) (x : ℕ) :
  cost_per_bar = 5 →
  packaging_cost = 2 →
  total_sale = 90 →
  profit = 55 →
  (total_sale - (cost_per_bar + packaging_cost) * x = profit) →
  x = 5 :=
by
  sorry

end Romeo_bars_of_chocolate_l971_97142


namespace triangle_obtuse_l971_97164

variable {a b c : ℝ}

theorem triangle_obtuse (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ C : ℝ, 0 ≤ C ∧ C ≤ π ∧ Real.cos C = -1/4 ∧ C > Real.pi / 2 :=
by
  sorry

end triangle_obtuse_l971_97164


namespace sin_cos_sum_inequality_l971_97112

theorem sin_cos_sum_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 := 
sorry

end sin_cos_sum_inequality_l971_97112


namespace shaded_square_ratio_l971_97104

theorem shaded_square_ratio (side_length : ℝ) (H : side_length = 5) :
  let large_square_area := side_length ^ 2
  let shaded_square_area := (side_length / 2) ^ 2
  shaded_square_area / large_square_area = 1 / 4 :=
by
  sorry

end shaded_square_ratio_l971_97104


namespace Jerry_paid_more_last_month_l971_97156

def Debt_total : ℕ := 50
def Debt_remaining : ℕ := 23
def Paid_2_months_ago : ℕ := 12
def Paid_last_month : ℕ := 27 - Paid_2_months_ago

theorem Jerry_paid_more_last_month :
  Paid_last_month - Paid_2_months_ago = 3 :=
by
  -- Calculation for Paid_last_month
  have h : Paid_last_month = 27 - 12 := by rfl
  -- Compute the difference
  have diff : 15 - 12 = 3 := by rfl
  exact diff

end Jerry_paid_more_last_month_l971_97156


namespace find_number_l971_97123

theorem find_number : ∃ n : ℕ, n = (15 * 6) + 5 := 
by sorry

end find_number_l971_97123


namespace max_regions_11_l971_97198

noncomputable def max_regions (n : ℕ) : ℕ :=
  1 + n * (n + 1) / 2

theorem max_regions_11 : max_regions 11 = 67 := by
  unfold max_regions
  norm_num

end max_regions_11_l971_97198


namespace one_and_one_third_of_x_is_36_l971_97120

theorem one_and_one_third_of_x_is_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := 
sorry

end one_and_one_third_of_x_is_36_l971_97120


namespace find_theta_ratio_l971_97114

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem find_theta_ratio (θ : ℝ) 
  (h : det2x2 (Real.sin θ) 2 (Real.cos θ) 3 = 0) : 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 := 
by 
  sorry

end find_theta_ratio_l971_97114


namespace count_valid_numbers_l971_97136

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ d₁ ≠ d₂ ∧
    (∀ k, (n / 10^k % 10 = d₁ ∨ n / 10^k % 10 = d₂))

theorem count_valid_numbers : 
  ∃ (k : ℕ), k = 2151 ∧ (∀ n, is_valid_number n → n < 100000 → n ≤ k) :=
by
  sorry

end count_valid_numbers_l971_97136


namespace integer_solution_unique_l971_97105

theorem integer_solution_unique (w x y z : ℤ) :
  w^2 + 11 * x^2 - 8 * y^2 - 12 * y * z - 10 * z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry
 
end integer_solution_unique_l971_97105


namespace remainder_eval_at_4_l971_97125

def p : ℚ → ℚ := sorry

def r (x : ℚ) : ℚ := sorry

theorem remainder_eval_at_4 :
  (p 1 = 2) →
  (p 3 = 5) →
  (p (-2) = -2) →
  (∀ x, ∃ q : ℚ → ℚ, p x = (x - 1) * (x - 3) * (x + 2) * q x + r x) →
  r 4 = 38 / 7 :=
sorry

end remainder_eval_at_4_l971_97125


namespace find_number_l971_97155

-- Define the main condition and theorem.
theorem find_number (x : ℤ) : 45 - (x - (37 - (15 - 19))) = 58 ↔ x = 28 :=
by
  sorry  -- placeholder for the proof

end find_number_l971_97155


namespace final_hair_length_is_14_l971_97169

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end final_hair_length_is_14_l971_97169


namespace cds_unique_to_either_l971_97186

-- Declare the variables for the given problem
variables (total_alice_shared : ℕ) (total_alice : ℕ) (unique_bob : ℕ)

-- The given conditions in the problem
def condition_alice : Prop := total_alice_shared + unique_bob + (total_alice - total_alice_shared) = total_alice

-- The theorem to prove: number of CDs in either Alice's or Bob's collection but not both is 19
theorem cds_unique_to_either (h1 : total_alice = 23) 
                             (h2 : total_alice_shared = 12) 
                             (h3 : unique_bob = 8) : 
                             (total_alice - total_alice_shared) + unique_bob = 19 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end cds_unique_to_either_l971_97186


namespace sum_of_integers_l971_97130

theorem sum_of_integers (x y : ℕ) (h1 : x = y + 3) (h2 : x^3 - y^3 = 63) : x + y = 5 :=
by
  sorry

end sum_of_integers_l971_97130


namespace jason_retirement_age_l971_97141

def age_at_retirement (initial_age years_to_chief extra_years_ratio years_after_masterchief : ℕ) : ℕ :=
  initial_age + years_to_chief + (years_to_chief * extra_years_ratio / 100) + years_after_masterchief

theorem jason_retirement_age :
  age_at_retirement 18 8 25 10 = 46 :=
by
  sorry

end jason_retirement_age_l971_97141


namespace radius_of_inscribed_circle_l971_97121

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] (triangle : Triangle A B C)

-- Given conditions
def AC : ℝ := 24
def BC : ℝ := 10
def AB : ℝ := 26

-- Statement to be proved
theorem radius_of_inscribed_circle (hAC : triangle.side_length A C = AC)
                                   (hBC : triangle.side_length B C = BC)
                                   (hAB : triangle.side_length A B = AB) :
  triangle.incircle_radius = 4 :=
by sorry

end radius_of_inscribed_circle_l971_97121


namespace alternating_draws_probability_l971_97184

noncomputable def probability_alternating_draws : ℚ :=
  let total_draws := 11
  let white_balls := 5
  let black_balls := 6
  let successful_sequences := 1
  let total_sequences := @Nat.choose total_draws black_balls
  successful_sequences / total_sequences

theorem alternating_draws_probability :
  probability_alternating_draws = 1 / 462 := by
  sorry

end alternating_draws_probability_l971_97184


namespace kombucha_bottles_after_refund_l971_97196

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end kombucha_bottles_after_refund_l971_97196


namespace find_x_l971_97179

theorem find_x (x : ℝ) : |2 * x - 6| = 3 * x + 1 ↔ x = 1 := 
by 
  sorry

end find_x_l971_97179


namespace smallest_m_satisfying_condition_l971_97146

def D (n : ℕ) : Finset ℕ := (n.divisors : Finset ℕ)

def F (n i : ℕ) : Finset ℕ :=
  (D n).filter (λ a => a % 4 = i)

def f (n i : ℕ) : ℕ :=
  (F n i).card

theorem smallest_m_satisfying_condition :
  ∃ m : ℕ, f m 0 + f m 1 - f m 2 - f m 3 = 2017 ∧
           m = 2^34 * 3^6 * 7^2 * 11^2 :=
by
  sorry

end smallest_m_satisfying_condition_l971_97146


namespace negation_of_p_l971_97168

variable {x : ℝ}

def p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_p_l971_97168


namespace brad_trips_to_fill_barrel_l971_97151

noncomputable def bucket_volume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

noncomputable def barrel_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem brad_trips_to_fill_barrel :
  let r_bucket := 8  -- radius of the hemisphere bucket in inches
  let r_barrel := 8  -- radius of the cylindrical barrel in inches
  let h_barrel := 20 -- height of the cylindrical barrel in inches
  let V_bucket := bucket_volume r_bucket
  let V_barrel := barrel_volume r_barrel h_barrel
  (Nat.ceil (V_barrel / V_bucket) = 4) :=
by
  sorry

end brad_trips_to_fill_barrel_l971_97151


namespace bus_passengers_total_l971_97160

theorem bus_passengers_total (children_percent : ℝ) (adults_number : ℝ) (H1 : children_percent = 0.25) (H2 : adults_number = 45) :
  ∃ T : ℝ, T = 60 :=
by
  sorry

end bus_passengers_total_l971_97160


namespace angle_A_and_shape_of_triangle_l971_97167

theorem angle_A_and_shape_of_triangle 
  (a b c : ℝ)
  (h1 : a^2 - c^2 = a * c - b * c)
  (h2 : ∃ r : ℝ, a = b * r ∧ c = b / r)
  (h3 : ∃ B C : Type, B = A ∧ C ≠ A ) :
  ∃ (A : ℝ), A = 60 ∧ a = b ∧ b = c := 
sorry

end angle_A_and_shape_of_triangle_l971_97167


namespace sum_symmetric_prob_43_l971_97195

def prob_symmetric_sum_43_with_20 : Prop :=
  let n_dice := 9
  let min_sum := n_dice * 1
  let max_sum := n_dice * 6
  let midpoint := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * midpoint - 20
  symmetric_sum = 43

theorem sum_symmetric_prob_43 (n_dice : ℕ) (h₁ : n_dice = 9) (h₂ : ∀ i : ℕ, i ≥ 1 ∧ i ≤ 6) :
  prob_symmetric_sum_43_with_20 :=
by
  sorry

end sum_symmetric_prob_43_l971_97195


namespace units_digit_of_exponentiated_product_l971_97103

theorem units_digit_of_exponentiated_product :
  (2 ^ 2101 * 5 ^ 2102 * 11 ^ 2103) % 10 = 0 := 
sorry

end units_digit_of_exponentiated_product_l971_97103


namespace point_on_inverse_proportion_l971_97172

theorem point_on_inverse_proportion :
  ∀ (k x y : ℝ), 
    (∀ (x y: ℝ), (x = -2 ∧ y = 6) → y = k / x) →
    k = -12 →
    y = k / x →
    (x = 1 ∧ y = -12) :=
by
  sorry

end point_on_inverse_proportion_l971_97172


namespace fourth_power_sum_l971_97180

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 6) : 
  a^4 + b^4 + c^4 = 4.5 :=
by
  sorry

end fourth_power_sum_l971_97180


namespace problem_l971_97165

noncomputable def a : Real := 9^(1/3)
noncomputable def b : Real := 3^(2/5)
noncomputable def c : Real := 4^(1/5)

theorem problem (a := 9^(1/3)) (b := 3^(2/5)) (c := 4^(1/5)) : a > b ∧ b > c := by
  sorry

end problem_l971_97165


namespace average_percentage_reduction_equation_l971_97158

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end average_percentage_reduction_equation_l971_97158


namespace find_misread_solution_l971_97175

theorem find_misread_solution:
  ∃ a b : ℝ, 
  a = 5 ∧ b = 2 ∧ 
    (a^2 - 2 * a * b + b^2 = 9) ∧ 
    (∀ x y : ℝ, (5 * x + 4 * y = 23) ∧ (3 * x - 2 * y = 5) → (x = 3) ∧ (y = 2)) := by
    sorry

end find_misread_solution_l971_97175


namespace problem1_solution_problem2_solution_l971_97177

-- Conditions for Problem 1
def problem1_condition (x : ℝ) : Prop := 
  5 * (x - 20) + 2 * x = 600

-- Proof for Problem 1 Goal
theorem problem1_solution (x : ℝ) (h : problem1_condition x) : x = 100 := 
by sorry

-- Conditions for Problem 2
def problem2_condition (m : ℝ) : Prop :=
  (360 / m) + (540 / (1.2 * m)) = (900 / 100)

-- Proof for Problem 2 Goal
theorem problem2_solution (m : ℝ) (h : problem2_condition m) : m = 90 := 
by sorry

end problem1_solution_problem2_solution_l971_97177


namespace find_K_l971_97171

theorem find_K (Z K : ℕ) (hZ1 : 1000 < Z) (hZ2 : Z < 8000) (hK : Z = K^3) : 11 ≤ K ∧ K ≤ 19 :=
sorry

end find_K_l971_97171


namespace find_w_l971_97100

theorem find_w {w : ℝ} : (3, w^3) ∈ {p : ℝ × ℝ | ∃ x, p = (x, x^2 - 1)} → w = 2 :=
by
  sorry

end find_w_l971_97100


namespace third_side_length_of_triangle_l971_97108

theorem third_side_length_of_triangle {a b c : ℝ} (h1 : a^2 - 7 * a + 12 = 0) (h2 : b^2 - 7 * b + 12 = 0) 
  (h3 : a ≠ b) (h4 : a = 3 ∨ a = 4) (h5 : b = 3 ∨ b = 4) : 
  (c = 5 ∨ c = Real.sqrt 7) := by
  sorry

end third_side_length_of_triangle_l971_97108


namespace proof_of_intersection_l971_97129

open Set

theorem proof_of_intersection :
  let U := ℝ
  let M := compl { x : ℝ | x^2 > 4 }
  let N := { x : ℝ | 1 < x ∧ x ≤ 3 }
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := by
sorry

end proof_of_intersection_l971_97129


namespace sin_cos_value_l971_97147

noncomputable def tan_plus_pi_div_two_eq_two (θ : ℝ) : Prop :=
  Real.tan (θ + Real.pi / 2) = 2

theorem sin_cos_value (θ : ℝ) (h : tan_plus_pi_div_two_eq_two θ) :
  Real.sin θ * Real.cos θ = -2 / 5 :=
sorry

end sin_cos_value_l971_97147


namespace mixed_number_subtraction_l971_97111

theorem mixed_number_subtraction :
  2 + 5 / 6 - (1 + 1 / 3) = 3 / 2 := by
sorry

end mixed_number_subtraction_l971_97111


namespace option_c_correct_l971_97194

theorem option_c_correct (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x < 2 → x^2 - a ≤ 0) : 4 < a :=
by
  sorry

end option_c_correct_l971_97194


namespace original_rectangle_area_l971_97138

theorem original_rectangle_area : 
  ∃ (a b : ℤ), (a + b = 20) ∧ (a * b = 96) := by
  sorry

end original_rectangle_area_l971_97138


namespace emily_initial_cards_l971_97143

theorem emily_initial_cards (x : ℤ) (h1 : x + 7 = 70) : x = 63 :=
by
  sorry

end emily_initial_cards_l971_97143


namespace pyramid_volume_l971_97191

noncomputable def volume_of_pyramid (l : ℝ) : ℝ :=
  (l^3 / 24) * (Real.sqrt (Real.sqrt 2 + 1))

theorem pyramid_volume (l : ℝ) (α β : ℝ)
  (hα : α = π / 8)
  (hβ : β = π / 4)
  (hl : l = 6) :
  volume_of_pyramid l = 9 * Real.sqrt (Real.sqrt 2 + 1) := by
  sorry

end pyramid_volume_l971_97191


namespace potential_values_of_k_l971_97173

theorem potential_values_of_k :
  ∃ k : ℚ, ∀ (a b : ℕ), 
  (10 * a + b = k * (a + b)) ∧ (10 * b + a = (13 - k) * (a + b)) → k = 11/2 :=
by
  sorry

end potential_values_of_k_l971_97173


namespace find_numbers_l971_97154

theorem find_numbers (x y : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : 100 ≤ y ∧ y ≤ 999) (h3 : 1000 * x + y = 7 * x * y) :
  x = 143 ∧ y = 143 :=
by
  sorry

end find_numbers_l971_97154


namespace ratio_of_DN_NF_l971_97128

theorem ratio_of_DN_NF (D E F N : Type) (DE EF DF DN NF p q: ℕ) (h1 : DE = 18) (h2 : EF = 28) (h3 : DF = 34) 
(h4 : DN + NF = DF) (h5 : DN = 22) (h6 : NF = 11) (h7 : p = 101) (h8 : q = 50) : p + q = 151 := 
by 
  sorry

end ratio_of_DN_NF_l971_97128


namespace combined_salaries_BCDE_l971_97176

-- Define the given conditions
def salary_A : ℕ := 10000
def average_salary : ℕ := 8400
def num_individuals : ℕ := 5

-- Define the total salary of all individuals
def total_salary_all : ℕ := average_salary * num_individuals

-- Define the proof problem
theorem combined_salaries_BCDE : (total_salary_all - salary_A) = 32000 := by
  sorry

end combined_salaries_BCDE_l971_97176


namespace number_of_terms_in_sequence_l971_97199

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem number_of_terms_in_sequence : 
  ∃ n : ℕ, arithmetic_sequence (-3) 4 n = 53 ∧ n = 15 :=
by
  use 15
  constructor
  · unfold arithmetic_sequence
    norm_num
  · norm_num

end number_of_terms_in_sequence_l971_97199


namespace unique_bounded_sequence_exists_l971_97113

variable (a : ℝ) (n : ℕ) (hn_pos : n > 0)

theorem unique_bounded_sequence_exists :
  ∃! (x : ℕ → ℝ), (x 0 = 0) ∧ (x (n+1) = 0) ∧
                   (∀ i, 1 ≤ i ∧ i ≤ n → (1/2) * (x (i+1) + x (i-1)) = x i + x i ^ 3 - a ^ 3) ∧
                   (∀ i, i ≤ n + 1 → |x i| ≤ |a|) := by
  sorry

end unique_bounded_sequence_exists_l971_97113


namespace percentage_of_invalid_papers_l971_97132

theorem percentage_of_invalid_papers (total_papers : ℕ) (valid_papers : ℕ) (invalid_papers : ℕ) (percentage_invalid : ℚ) 
  (h1 : total_papers = 400) 
  (h2 : valid_papers = 240) 
  (h3 : invalid_papers = total_papers - valid_ppapers)
  (h4 : percentage_invalid = (invalid_papers : ℚ) / total_papers * 100) : 
  percentage_invalid = 40 :=
by
  sorry

end percentage_of_invalid_papers_l971_97132


namespace find_principal_amount_l971_97126

theorem find_principal_amount
  (r : ℝ := 0.05)  -- Interest rate (5% per annum)
  (t : ℕ := 2)    -- Time period (2 years)
  (diff : ℝ := 20) -- Given difference between CI and SI
  (P : ℝ := 8000) -- Principal amount to prove
  : P * (1 + r) ^ t - P - P * r * t = diff :=
by
  sorry

end find_principal_amount_l971_97126


namespace two_person_subcommittees_from_six_l971_97188

theorem two_person_subcommittees_from_six :
  (Nat.choose 6 2) = 15 := by
  sorry

end two_person_subcommittees_from_six_l971_97188


namespace total_packs_sold_l971_97157

theorem total_packs_sold (lucy_packs : ℕ) (robyn_packs : ℕ) (h1 : lucy_packs = 19) (h2 : robyn_packs = 16) : lucy_packs + robyn_packs = 35 :=
by
  sorry

end total_packs_sold_l971_97157


namespace trig_identity_proof_l971_97117

noncomputable def cos_30 := Real.cos (Real.pi / 6)
noncomputable def sin_60 := Real.sin (Real.pi / 3)
noncomputable def sin_30 := Real.sin (Real.pi / 6)
noncomputable def cos_60 := Real.cos (Real.pi / 3)

theorem trig_identity_proof :
  (1 - (1 / cos_30)) * (1 + (2 / sin_60)) * (1 - (1 / sin_30)) * (1 + (2 / cos_60)) = (25 - 10 * Real.sqrt 3) / 3 := by
  sorry

end trig_identity_proof_l971_97117


namespace wheel_distance_travelled_l971_97135

noncomputable def radius : ℝ := 3
noncomputable def num_revolutions : ℝ := 3
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def total_distance (r : ℝ) (n : ℝ) : ℝ := n * circumference r

theorem wheel_distance_travelled :
  total_distance radius num_revolutions = 18 * Real.pi :=
by 
  sorry

end wheel_distance_travelled_l971_97135


namespace line_representation_l971_97109

variable {R : Type*} [Field R]
variable (f : R → R → R)
variable (x0 y0 : R)

def not_on_line (P : R × R) (f : R → R → R) : Prop :=
  f P.1 P.2 ≠ 0

theorem line_representation (P : R × R) (hP : not_on_line P f) :
  ∃ l : R → R → Prop, (∀ x y, l x y ↔ f x y - f P.1 P.2 = 0) ∧ (l P.1 P.2) ∧ 
  ∀ x y, f x y = 0 → ∃ n : R, ∀ x1 y1, (l x1 y1 → f x1 y1 = n * (f x y)) :=
sorry

end line_representation_l971_97109


namespace sufficient_but_not_necessary_condition_l971_97185

noncomputable def f (x a : ℝ) : ℝ := abs (x - a)

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≤ -2) ↔ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a)) ∧ ¬ (∀ x y : ℝ, (-1 ≤ x) → (x ≤ y) → (f x a ≤ f y a) → (a ≤ -2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l971_97185


namespace total_prize_amount_l971_97107

theorem total_prize_amount:
  ∃ P : ℝ, 
  (∃ n m : ℝ, n = 15 ∧ m = 15 ∧ ((2 / 5) * P = (3 / 5) * n * 285) ∧ P = 2565 * 2.5 + 6 * 15 ∧ ∀ i : ℕ, i < m → i ≥ 0 → P ≥ 15)
  ∧ P = 6502.5 :=
sorry

end total_prize_amount_l971_97107


namespace last_two_digits_of_9_pow_2008_l971_97174

theorem last_two_digits_of_9_pow_2008 : (9 ^ 2008) % 100 = 21 := 
by
  sorry

end last_two_digits_of_9_pow_2008_l971_97174


namespace range_of_a_for_no_extreme_points_l971_97181

theorem range_of_a_for_no_extreme_points :
  ∀ (a : ℝ), (∀ x : ℝ, x * (x - 2 * a) * x + 1 ≠ 0) ↔ -1 ≤ a ∧ a ≤ 1 := sorry

end range_of_a_for_no_extreme_points_l971_97181


namespace regular_polygon_sides_l971_97153

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l971_97153


namespace price_reduction_eq_l971_97192

theorem price_reduction_eq (x : ℝ) (price_original price_final : ℝ) 
    (h1 : price_original = 400) 
    (h2 : price_final = 200) 
    (h3 : price_final = price_original * (1 - x) * (1 - x)) :
  400 * (1 - x)^2 = 200 :=
by
  sorry

end price_reduction_eq_l971_97192


namespace coins_of_each_type_l971_97163

theorem coins_of_each_type (x : ℕ) (h : x + x / 2 + x / 4 = 70) : x = 40 :=
sorry

end coins_of_each_type_l971_97163


namespace area_of_shaded_region_l971_97178

theorem area_of_shaded_region (r R : ℝ) (π : ℝ) (h1 : R = 3 * r) (h2 : 2 * r = 6) : 
  (π * R^2) - (π * r^2) = 72 * π :=
by
  sorry

end area_of_shaded_region_l971_97178


namespace gcd_mn_mn_squared_l971_97110

theorem gcd_mn_mn_squared (m n : ℕ) (h : Nat.gcd m n = 1) : ({d : ℕ | d = Nat.gcd (m + n) (m ^ 2 + n ^ 2)} ⊆ {1, 2}) := 
sorry

end gcd_mn_mn_squared_l971_97110


namespace red_gumballs_count_l971_97150

def gumballs_problem (R B G : ℕ) : Prop :=
  B = R / 2 ∧
  G = 4 * B ∧
  R + B + G = 56

theorem red_gumballs_count (R B G : ℕ) (h : gumballs_problem R B G) : R = 16 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end red_gumballs_count_l971_97150


namespace games_lost_l971_97115

theorem games_lost (total_games won_games : ℕ) (h_total : total_games = 12) (h_won : won_games = 8) :
  (total_games - won_games) = 4 :=
by
  -- Placeholder for the proof
  sorry

end games_lost_l971_97115


namespace combined_degrees_l971_97148

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) (h1 : summer_degrees = jolly_degrees + 5) (h2 : summer_degrees = 150) : summer_degrees + jolly_degrees = 295 := 
by
  sorry

end combined_degrees_l971_97148


namespace job_completion_l971_97145

theorem job_completion (x y z : ℝ) 
  (h1 : 1/x + 1/y = 1/2) 
  (h2 : 1/y + 1/z = 1/4) 
  (h3 : 1/z + 1/x = 1/2.4) 
  (h4 : 1/x + 1/y + 1/z = 7/12) : 
  x = 3 := 
sorry

end job_completion_l971_97145


namespace handshake_count_l971_97152

def gathering_handshakes (total_people : ℕ) (know_each_other : ℕ) (know_no_one : ℕ) : ℕ :=
  let group2_handshakes := know_no_one * (total_people - 1)
  group2_handshakes / 2

theorem handshake_count :
  gathering_handshakes 30 20 10 = 145 :=
by
  sorry

end handshake_count_l971_97152


namespace bob_paid_24_percent_of_SRP_l971_97144

theorem bob_paid_24_percent_of_SRP
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ) -- Marked Price (MP)
  (price_bob_paid : ℝ) -- Price Bob Paid
  (h1 : MP = 0.60 * P) -- Condition 1: MP is 60% of SRP
  (h2 : price_bob_paid = 0.40 * MP) -- Condition 2: Bob paid 40% of the MP
  : (price_bob_paid / P) * 100 = 24 := -- Bob paid 24% of the SRP
by
  sorry

end bob_paid_24_percent_of_SRP_l971_97144
