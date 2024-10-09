import Mathlib

namespace captain_co_captain_selection_l1911_191196

theorem captain_co_captain_selection 
  (men women : ℕ)
  (h_men : men = 12) 
  (h_women : women = 12) : 
  (men * (men - 1) + women * (women - 1)) = 264 := 
by
  -- Since we are skipping the proof here, we use sorry.
  sorry

end captain_co_captain_selection_l1911_191196


namespace balloons_left_l1911_191185

theorem balloons_left (yellow blue pink violet friends : ℕ) (total_balloons remainder : ℕ) 
  (hy : yellow = 20) (hb : blue = 24) (hp : pink = 50) (hv : violet = 102) (hf : friends = 9)
  (ht : total_balloons = yellow + blue + pink + violet) (hr : total_balloons % friends = remainder) : 
  remainder = 7 :=
by
  sorry

end balloons_left_l1911_191185


namespace hyperbola_eccentricity_l1911_191178

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), (b = 5) → (c = 3) → (c^2 = a^2 + b) → (a > 0) →
  (a + c = 3) → (e = c / a) → (e = 3 / 2) :=
by
  intros a b c hb hc hc2 ha hac he
  sorry

end hyperbola_eccentricity_l1911_191178


namespace value_of_f_5_l1911_191123

theorem value_of_f_5 (f : ℕ → ℕ) (y : ℕ)
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : f 5 = 62 :=
sorry

end value_of_f_5_l1911_191123


namespace lcm_16_24_45_l1911_191145

-- Define the numbers
def a : Nat := 16
def b : Nat := 24
def c : Nat := 45

-- State the theorem that the least common multiple of these numbers is 720
theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end lcm_16_24_45_l1911_191145


namespace vacant_seats_l1911_191121

theorem vacant_seats (total_seats : ℕ) (filled_percent vacant_percent : ℚ) 
  (h_total : total_seats = 600)
  (h_filled_percent : filled_percent = 75)
  (h_vacant_percent : vacant_percent = 100 - filled_percent)
  (h_vacant_percent_25 : vacant_percent = 25) :
  (25 / 100) * 600 = 150 :=
by 
  -- this is the final answer we want to prove, replace with sorry to skip the proof just for statement validation
  sorry

end vacant_seats_l1911_191121


namespace probability_male_monday_female_tuesday_l1911_191113

structure Volunteers where
  men : ℕ
  women : ℕ
  total : ℕ

def group : Volunteers := {men := 2, women := 2, total := 4}

def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_male_monday_female_tuesday :
  let n := permutations group.total 2
  let m := combinations group.men 1 * combinations group.women 1
  (m / n : ℚ) = 1 / 3 :=
by
  sorry

end probability_male_monday_female_tuesday_l1911_191113


namespace relationship_a_b_c_l1911_191134

open Real

theorem relationship_a_b_c (x : ℝ) (hx1 : e < x) (hx2 : x < e^2)
  (a : ℝ) (ha : a = log x)
  (b : ℝ) (hb : b = (1 / 2) ^ log x)
  (c : ℝ) (hc : c = exp (log x)) :
  c > a ∧ a > b :=
by {
  -- we state the theorem without providing the proof for now
  sorry
}

end relationship_a_b_c_l1911_191134


namespace area_of_triangle_l1911_191140

-- Definitions of the conditions
def hypotenuse_AC (a b c : ℝ) : Prop := c = 50
def sum_of_legs (a b : ℝ) : Prop := a + b = 70
def pythagorean_theorem (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- The main theorem statement
theorem area_of_triangle (a b c : ℝ) (h1 : hypotenuse_AC a b c)
  (h2 : sum_of_legs a b) (h3 : pythagorean_theorem a b c) : 
  (1/2) * a * b = 300 := 
by
  sorry

end area_of_triangle_l1911_191140


namespace total_distance_walked_l1911_191116

def distance_to_fountain : ℕ := 30
def number_of_trips : ℕ := 4
def round_trip_distance : ℕ := 2 * distance_to_fountain

theorem total_distance_walked : (number_of_trips * round_trip_distance) = 240 := by
  sorry

end total_distance_walked_l1911_191116


namespace find_cost_price_l1911_191162

theorem find_cost_price (C : ℝ) (h1 : 1.12 * C + 18 = 1.18 * C) : C = 300 :=
by
  sorry

end find_cost_price_l1911_191162


namespace sequence_an_l1911_191190

theorem sequence_an (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := 
by
  sorry

end sequence_an_l1911_191190


namespace no_solution_for_k_eq_4_l1911_191142

theorem no_solution_for_k_eq_4 (x k : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : (k = 4) → ¬ ((x - 3) * (x - 8) = (x - k) * (x - 4)) :=
by
  sorry

end no_solution_for_k_eq_4_l1911_191142


namespace oil_leakage_during_repair_l1911_191183

variables (initial_leak: ℚ) (initial_hours: ℚ) (repair_hours: ℚ) (reduction: ℚ) (total_leak: ℚ)

theorem oil_leakage_during_repair
    (h1 : initial_leak = 2475)
    (h2 : initial_hours = 7)
    (h3 : repair_hours = 5)
    (h4 : reduction = 0.75)
    (h5 : total_leak = 6206) :
    (total_leak - initial_leak = 3731) :=
by
  sorry

end oil_leakage_during_repair_l1911_191183


namespace circle_radius_l1911_191188

theorem circle_radius (A r : ℝ) (h1 : A = 64 * Real.pi) (h2 : A = Real.pi * r^2) : r = 8 := 
by
  sorry

end circle_radius_l1911_191188


namespace mindy_earns_k_times_more_than_mork_l1911_191147

-- Given the following conditions:
-- Mork's tax rate: 0.45
-- Mindy's tax rate: 0.25
-- Combined tax rate: 0.29
-- Mindy earns k times more than Mork

theorem mindy_earns_k_times_more_than_mork (M : ℝ) (k : ℝ) (hM : M > 0) :
  (0.45 * M + 0.25 * k * M) / (M * (1 + k)) = 0.29 → k = 4 :=
by
  sorry

end mindy_earns_k_times_more_than_mork_l1911_191147


namespace age_problem_l1911_191148

theorem age_problem (S F : ℕ) (h1 : F = S + 27) (h2 : F + 2 = 2 * (S + 2)) :
  S = 25 := by
  sorry

end age_problem_l1911_191148


namespace ratio_of_side_lengths_l1911_191149

theorem ratio_of_side_lengths (t s : ℕ) (ht : 2 * t + (20 - 2 * t) = 20) (hs : 4 * s = 20) :
  t / s = 4 / 3 :=
by
  sorry

end ratio_of_side_lengths_l1911_191149


namespace weight_of_milk_l1911_191160

def max_bag_capacity : ℕ := 20
def green_beans : ℕ := 4
def carrots : ℕ := 2 * green_beans
def fit_more : ℕ := 2
def current_weight : ℕ := max_bag_capacity - fit_more
def total_weight_of_green_beans_and_carrots : ℕ := green_beans + carrots

theorem weight_of_milk : (current_weight - total_weight_of_green_beans_and_carrots) = 6 := by
  -- Proof to be written here
  sorry

end weight_of_milk_l1911_191160


namespace compare_f_l1911_191182

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem compare_f (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 = 0) : 
  f x1 < f x2 :=
by sorry

end compare_f_l1911_191182


namespace Dan_has_five_limes_l1911_191184

-- Define the initial condition of limes Dan had
def initial_limes : Nat := 9

-- Define the limes Dan gave to Sara
def limes_given : Nat := 4

-- Define the remaining limes Dan has
def remaining_limes : Nat := initial_limes - limes_given

-- The theorem we need to prove, i.e., the remaining limes Dan has is 5
theorem Dan_has_five_limes : remaining_limes = 5 := by
  sorry

end Dan_has_five_limes_l1911_191184


namespace remainder_division_l1911_191103

theorem remainder_division (n r : ℕ) (k : ℤ) (h1 : n % 25 = r) (h2 : (n + 15) % 5 = r) (h3 : 0 ≤ r ∧ r < 25) : r = 5 :=
sorry

end remainder_division_l1911_191103


namespace difference_of_squares_l1911_191153

variable (x y : ℚ)

theorem difference_of_squares (h1 : x + y = 3 / 8) (h2 : x - y = 1 / 8) : x^2 - y^2 = 3 / 64 := 
by
  sorry

end difference_of_squares_l1911_191153


namespace abs_expression_value_l1911_191152

theorem abs_expression_value : (abs (2 * Real.pi - abs (Real.pi - 9))) = 3 * Real.pi - 9 := 
by
  sorry

end abs_expression_value_l1911_191152


namespace even_and_multiple_of_3_l1911_191127

theorem even_and_multiple_of_3 (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) (h2 : ∃ n : ℤ, b = 6 * n) :
  (∃ m : ℤ, a + b = 2 * m) ∧ (∃ p : ℤ, a + b = 3 * p) :=
by
  sorry

end even_and_multiple_of_3_l1911_191127


namespace jane_mistake_corrected_l1911_191104

-- Conditions translated to Lean definitions
variables (x y z : ℤ)
variable (h1 : x - (y + z) = 15)
variable (h2 : x - y + z = 7)

-- Statement to prove
theorem jane_mistake_corrected : x - y = 11 :=
by
  -- Placeholder for the proof
  sorry

end jane_mistake_corrected_l1911_191104


namespace solve_linear_system_l1911_191111

theorem solve_linear_system (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = -10 - 4 * y)
  (h3 : x + y = 14 - 4 * z)
  : 2 * x + 2 * y + 2 * z = 8 :=
by
  sorry

end solve_linear_system_l1911_191111


namespace quadrilateral_inequality_l1911_191158

theorem quadrilateral_inequality (A C : ℝ) (AB AC AD BC CD : ℝ) (h1 : A + C < 180) (h2 : A > 0) (h3 : C > 0) (h4 : AB > 0) (h5 : AC > 0) (h6 : AD > 0) (h7 : BC > 0) (h8 : CD > 0) : 
  AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l1911_191158


namespace double_angle_value_l1911_191199

theorem double_angle_value : 2 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 2 := 
sorry

end double_angle_value_l1911_191199


namespace system_of_equations_solution_l1911_191133

theorem system_of_equations_solution (x y z : ℝ) :
  (4 * x^2 / (1 + 4 * x^2) = y ∧
   4 * y^2 / (1 + 4 * y^2) = z ∧
   4 * z^2 / (1 + 4 * z^2) = x) →
  ((x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) :=
by
  sorry

end system_of_equations_solution_l1911_191133


namespace unique_k_satisfying_eq_l1911_191122

theorem unique_k_satisfying_eq (k : ℤ) :
  (∀ a b c : ℝ, (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ k = -1 :=
sorry

end unique_k_satisfying_eq_l1911_191122


namespace cubic_poly_real_roots_l1911_191164

theorem cubic_poly_real_roots (a b c d : ℝ) (h : a ≠ 0) : 
  ∃ (min_roots max_roots : ℕ), 1 ≤ min_roots ∧ max_roots ≤ 3 ∧ min_roots = 1 ∧ max_roots = 3 :=
by
  sorry

end cubic_poly_real_roots_l1911_191164


namespace dress_designs_count_l1911_191174

inductive Color
| red | green | blue | yellow

inductive Pattern
| stripes | polka_dots | floral | geometric | plain

def patterns_for_color (c : Color) : List Pattern :=
  match c with
  | Color.red    => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.green  => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]
  | Color.blue   => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.yellow => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]

noncomputable def number_of_dress_designs : ℕ :=
  (patterns_for_color Color.red).length +
  (patterns_for_color Color.green).length +
  (patterns_for_color Color.blue).length +
  (patterns_for_color Color.yellow).length

theorem dress_designs_count : number_of_dress_designs = 18 :=
  by
  sorry

end dress_designs_count_l1911_191174


namespace commercial_duration_l1911_191119

/-- Michael was watching a TV show, which was aired for 1.5 hours. 
    During this time, there were 3 commercials. 
    The TV show itself, not counting commercials, was 1 hour long. 
    Prove that each commercial lasted 10 minutes. -/
theorem commercial_duration (total_time : ℝ) (num_commercials : ℕ) (show_time : ℝ)
  (h1 : total_time = 1.5) (h2 : num_commercials = 3) (h3 : show_time = 1) :
  (total_time - show_time) / num_commercials * 60 = 10 := 
sorry

end commercial_duration_l1911_191119


namespace average_marks_combined_l1911_191108

theorem average_marks_combined (P C M B E : ℕ) (h : P + C + M + B + E = P + 280) : 
  (C + M + B + E) / 4 = 70 :=
by 
  sorry

end average_marks_combined_l1911_191108


namespace largest_lucky_number_l1911_191187

theorem largest_lucky_number (n : ℕ) (h₀ : n = 160) (h₁ : ∀ k, 160 > k → k > 0) (h₂ : ∀ k, k ≡ 7 [MOD 16] → k ≤ 160) : 
  ∃ k, k = 151 := 
sorry

end largest_lucky_number_l1911_191187


namespace exists_pos_integer_n_l1911_191117

theorem exists_pos_integer_n (n : ℕ) (hn_pos : n > 0) (h : ∃ m : ℕ, m * m = 1575 * n) : n = 7 :=
sorry

end exists_pos_integer_n_l1911_191117


namespace sin_of_7pi_over_6_l1911_191125

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l1911_191125


namespace girls_tried_out_l1911_191107

-- Definitions for conditions
def boys_trying_out : ℕ := 4
def students_called_back : ℕ := 26
def students_did_not_make_cut : ℕ := 17

-- Definition to calculate total students who tried out
def total_students_who_tried_out : ℕ := students_called_back + students_did_not_make_cut

-- Proof statement
theorem girls_tried_out : ∀ (G : ℕ), G + boys_trying_out = total_students_who_tried_out → G = 39 :=
by
  intro G
  intro h
  rw [total_students_who_tried_out, boys_trying_out] at h
  sorry

end girls_tried_out_l1911_191107


namespace total_marks_calculation_l1911_191138

def average (total_marks : ℕ) (num_candidates : ℕ) : ℕ := total_marks / num_candidates
def total_marks (average : ℕ) (num_candidates : ℕ) : ℕ := average * num_candidates

theorem total_marks_calculation
  (num_candidates : ℕ)
  (average_marks : ℕ)
  (range_min : ℕ)
  (range_max : ℕ)
  (h1 : num_candidates = 250)
  (h2 : average_marks = 42)
  (h3 : range_min = 10)
  (h4 : range_max = 80) :
  total_marks average_marks num_candidates = 10500 :=
by 
  sorry

end total_marks_calculation_l1911_191138


namespace find_m_of_quadratic_root_l1911_191173

theorem find_m_of_quadratic_root
  (m : ℤ) 
  (h : ∃ x : ℤ, x^2 - (m+3)*x + m + 2 = 0 ∧ x = 81) : 
  m = 79 :=
by
  sorry

end find_m_of_quadratic_root_l1911_191173


namespace flowers_left_l1911_191144

theorem flowers_left (flowers_picked_A : Nat) (flowers_picked_M : Nat) (flowers_given : Nat)
  (h_a : flowers_picked_A = 16)
  (h_m : flowers_picked_M = 16)
  (h_g : flowers_given = 18) :
  flowers_picked_A + flowers_picked_M - flowers_given = 14 :=
by
  sorry

end flowers_left_l1911_191144


namespace parabola_example_l1911_191198

theorem parabola_example (p : ℝ) (hp : p > 0)
    (h_intersect : ∀ x y : ℝ, y = x - p / 2 ∧ y^2 = 2 * p * x → ((x - p / 2)^2 = 2 * p * x))
    (h_AB : ∀ A B : ℝ × ℝ, A.2 = A.1 - p / 2 ∧ B.2 = B.1 - p / 2 ∧ |A.1 - B.1| = 8) :
    p = 2 := 
sorry

end parabola_example_l1911_191198


namespace units_digit_G_n_for_n_eq_3_l1911_191179

def G (n : ℕ) : ℕ := 2 ^ 2 ^ 2 ^ n + 1

theorem units_digit_G_n_for_n_eq_3 : (G 3) % 10 = 7 := 
by 
  sorry

end units_digit_G_n_for_n_eq_3_l1911_191179


namespace Diego_half_block_time_l1911_191193

def problem_conditions_and_solution : Prop :=
  ∃ (D : ℕ), (3 * 60 + D * 60) / 2 = 240 ∧ D = 5

theorem Diego_half_block_time :
  problem_conditions_and_solution :=
by
  sorry

end Diego_half_block_time_l1911_191193


namespace percent_motorists_receive_tickets_l1911_191177

theorem percent_motorists_receive_tickets (n : ℕ) (h1 : (25 : ℕ) % 100 = 25) (h2 : (20 : ℕ) % 100 = 20) :
  (75 * n / 100) = (20 * n / 100) :=
by
  sorry

end percent_motorists_receive_tickets_l1911_191177


namespace sum_of_cubes_l1911_191129

theorem sum_of_cubes
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (h1 : (x + y)^2 = 2500) 
  (h2 : x * y = 500) :
  x^3 + y^3 = 50000 := 
by
  sorry

end sum_of_cubes_l1911_191129


namespace cube_volume_surface_area_l1911_191172

-- Define volume and surface area conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 3 * x
def surface_area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x

-- The main theorem statement
theorem cube_volume_surface_area (x : ℝ) (s : ℝ) :
  volume_condition x s → surface_area_condition x s → x = 5832 :=
by
  intros h_volume h_area
  sorry

end cube_volume_surface_area_l1911_191172


namespace factory_output_l1911_191159

variable (a : ℝ)
variable (n : ℕ)
variable (r : ℝ)

-- Initial condition: the output value increases by 10% each year for 5 years
def annual_growth (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Theorem statement
theorem factory_output (a : ℝ) : annual_growth a 1.1 5 = 1.1^5 * a :=
by
  sorry

end factory_output_l1911_191159


namespace number_of_black_balls_l1911_191120

theorem number_of_black_balls
  (total_balls : ℕ)  -- define the total number of balls
  (B : ℕ)            -- define B as the number of black balls
  (prob_red : ℚ := 1/4) -- define the probability of drawing a red ball as 1/4
  (red_balls : ℕ := 3)  -- define the number of red balls as 3
  (h1 : total_balls = red_balls + B) -- total balls is the sum of red and black balls
  (h2 : red_balls / total_balls = prob_red) -- given probability
  : B = 9 :=              -- we need to prove that B is 9
by
  sorry

end number_of_black_balls_l1911_191120


namespace solve_for_x_l1911_191194

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
sorry

end solve_for_x_l1911_191194


namespace bonnie_roark_wire_length_ratio_l1911_191130

-- Define the conditions
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length_per_piece : ℕ := 8
def roark_wire_length_per_piece : ℕ := 2
def bonnie_cube_volume : ℕ := 8 * 8 * 8
def roark_total_cube_volume : ℕ := bonnie_cube_volume
def roark_unit_cube_volume : ℕ := 1
def roark_unit_cube_wires : ℕ := 12

-- Calculate Bonnie's total wire length
noncomputable def bonnie_total_wire_length : ℕ := bonnie_wire_pieces * bonnie_wire_length_per_piece

-- Calculate the number of Roark's unit cubes
noncomputable def roark_number_of_unit_cubes : ℕ := roark_total_cube_volume / roark_unit_cube_volume

-- Calculate the total wire used by Roark
noncomputable def roark_total_wire_length : ℕ := roark_number_of_unit_cubes * roark_unit_cube_wires * roark_wire_length_per_piece

-- Calculate the ratio of Bonnie's total wire length to Roark's total wire length
noncomputable def wire_length_ratio : ℚ := bonnie_total_wire_length / roark_total_wire_length

-- State the theorem
theorem bonnie_roark_wire_length_ratio : wire_length_ratio = 1 / 128 := 
by 
  sorry

end bonnie_roark_wire_length_ratio_l1911_191130


namespace minimum_value_proof_l1911_191143

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end minimum_value_proof_l1911_191143


namespace longest_line_segment_l1911_191189

theorem longest_line_segment (total_length_cm : ℕ) (h : total_length_cm = 3000) :
  ∃ n : ℕ, 2 * (n * (n + 1) / 2) ≤ total_length_cm ∧ n = 54 :=
by
  use 54
  sorry

end longest_line_segment_l1911_191189


namespace monotonic_quadratic_range_l1911_191171

-- Define a quadratic function
noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- The theorem
theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≤ quadratic a x₂) ∨
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≥ quadratic a x₂) →
  (a ≤ 2 ∨ 3 ≤ a) :=
sorry

end monotonic_quadratic_range_l1911_191171


namespace circle_radius_tangents_l1911_191166

theorem circle_radius_tangents
  (AB CD EF r : ℝ)
  (circle_tangent_AB : AB = 5)
  (circle_tangent_CD : CD = 11)
  (circle_tangent_EF : EF = 15) :
  r = 2.5 := by
  sorry

end circle_radius_tangents_l1911_191166


namespace expression_evaluation_l1911_191141

theorem expression_evaluation :
  (0.15)^3 - (0.06)^3 / (0.15)^2 + 0.009 + (0.06)^2 = 0.006375 :=
by
  sorry

end expression_evaluation_l1911_191141


namespace tilly_bag_cost_l1911_191128

noncomputable def cost_per_bag (n s P τ F : ℕ) : ℕ :=
  let revenue := n * s
  let total_sales_tax := n * (s * τ / 100)
  let total_additional_expenses := total_sales_tax + F
  (revenue - (P + total_additional_expenses)) / n

theorem tilly_bag_cost :
  let n := 100
  let s := 10
  let P := 300
  let τ := 5
  let F := 50
  cost_per_bag n s P τ F = 6 :=
  by
    let n := 100
    let s := 10
    let P := 300
    let τ := 5
    let F := 50
    have : cost_per_bag n s P τ F = 6 := sorry
    exact this

end tilly_bag_cost_l1911_191128


namespace maximize_revenue_l1911_191163

-- Define the problem conditions
def is_valid (x y : ℕ) : Prop :=
  x + y ≤ 60 ∧ 6 * x + 30 * y ≤ 600

-- Define the objective function
def revenue (x y : ℕ) : ℚ :=
  2.5 * x + 7.5 * y

-- State the theorem with the given conditions
theorem maximize_revenue : 
  (∃ x y : ℕ, is_valid x y ∧ ∀ a b : ℕ, is_valid a b → revenue x y >= revenue a b) ∧
  ∃ x y, is_valid x y ∧ revenue x y = revenue 50 10 := 
sorry

end maximize_revenue_l1911_191163


namespace set_in_quadrant_I_l1911_191112

theorem set_in_quadrant_I (x y : ℝ) (h1 : y ≥ 3 * x) (h2 : y ≥ 5 - x) (h3 : y < 7) : 
  x > 0 ∧ y > 0 :=
sorry

end set_in_quadrant_I_l1911_191112


namespace ducks_counted_l1911_191132

theorem ducks_counted (x y : ℕ) (h1 : x + y = 300) (h2 : 2 * x + 4 * y = 688) : x = 256 :=
by
  sorry

end ducks_counted_l1911_191132


namespace value_of_y_at_x8_l1911_191169

theorem value_of_y_at_x8 (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = k * x^(1 / 3)) (h2 : f 64 = 4) : f 8 = 2 :=
sorry

end value_of_y_at_x8_l1911_191169


namespace max_quarters_l1911_191101

theorem max_quarters (total_value : ℝ) (n_quarters n_nickels n_dimes : ℕ) 
  (h1 : n_nickels = n_quarters) 
  (h2 : n_dimes = 2 * n_quarters)
  (h3 : 0.25 * n_quarters + 0.05 * n_nickels + 0.10 * n_dimes = total_value)
  (h4 : total_value = 3.80) : 
  n_quarters = 7 := 
by
  sorry

end max_quarters_l1911_191101


namespace translate_parabola_l1911_191136

theorem translate_parabola :
  (∀ x, y = 1/2 * x^2 + 1 → y = 1/2 * (x - 1)^2 - 2) :=
by
  sorry

end translate_parabola_l1911_191136


namespace hyperbola_iff_m_lt_0_l1911_191165

theorem hyperbola_iff_m_lt_0 (m : ℝ) : (m < 0) ↔ (∃ x y : ℝ,  x^2 + m * y^2 = m) :=
by sorry

end hyperbola_iff_m_lt_0_l1911_191165


namespace T_number_square_l1911_191168

theorem T_number_square (a b : ℤ) : ∃ c d : ℤ, (a^2 + a * b + b^2)^2 = c^2 + c * d + d^2 := by
  sorry

end T_number_square_l1911_191168


namespace find_n_l1911_191102

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  11 + (n - 1) * 6

-- State the problem
theorem find_n (n : ℕ) : 
  (∀ m : ℕ, m ≥ n → arithmetic_sequence m > 2017) ↔ n = 336 :=
by
  sorry

end find_n_l1911_191102


namespace sum_of_undefined_domain_values_l1911_191115

theorem sum_of_undefined_domain_values :
  ∀ (x : ℝ), (x = 0 ∨ (1 + 1/x) = 0 ∨ (1 + 1/(1 + 1/x)) = 0 ∨ (1 + 1/(1 + 1/(1 + 1/x))) = 0) →
  x = 0 ∧ x = -1 ∧ x = -1/2 ∧ x = -1/3 →
  (0 + (-1) + (-1/2) + (-1/3) = -11/6) := sorry

end sum_of_undefined_domain_values_l1911_191115


namespace Sara_taller_than_Joe_l1911_191186

noncomputable def Roy_height := 36

noncomputable def Joe_height := Roy_height + 3

noncomputable def Sara_height := 45

theorem Sara_taller_than_Joe : Sara_height - Joe_height = 6 :=
by
  sorry

end Sara_taller_than_Joe_l1911_191186


namespace find_smaller_number_l1911_191197

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end find_smaller_number_l1911_191197


namespace least_number_remainder_l1911_191154

theorem least_number_remainder (N k : ℕ) (h : N = 18 * k + 4) : N = 256 :=
by
  sorry

end least_number_remainder_l1911_191154


namespace distinct_ratios_zero_l1911_191105

theorem distinct_ratios_zero (p q r : ℝ) (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) 
  (h : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 :=
sorry

end distinct_ratios_zero_l1911_191105


namespace range_of_f_l1911_191175

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x / Real.log 2

theorem range_of_f : Set.range f = Set.Ici 3 :=
by
  sorry

end range_of_f_l1911_191175


namespace modulus_product_l1911_191118

open Complex -- to open the complex namespace

-- Define the complex numbers
def z1 : ℂ := 10 - 5 * Complex.I
def z2 : ℂ := 7 + 24 * Complex.I

-- State the theorem to prove
theorem modulus_product : abs (z1 * z2) = 125 * Real.sqrt 5 := by
  sorry

end modulus_product_l1911_191118


namespace hamburgers_sold_last_week_l1911_191192

theorem hamburgers_sold_last_week (avg_hamburgers_per_day : ℕ) (days_in_week : ℕ) 
    (h_avg : avg_hamburgers_per_day = 9) (h_days : days_in_week = 7) : 
    avg_hamburgers_per_day * days_in_week = 63 :=
by
  -- Avg hamburgers per day times number of days
  sorry

end hamburgers_sold_last_week_l1911_191192


namespace solution_set_quadratic_l1911_191139

-- Define the quadratic equation as a function
def quadratic_eq (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- The theorem to prove
theorem solution_set_quadratic :
  {x : ℝ | quadratic_eq x = 0} = {1, 2} :=
by
  sorry

end solution_set_quadratic_l1911_191139


namespace difference_between_eights_l1911_191155

theorem difference_between_eights (value_tenths : ℝ) (value_hundredths : ℝ) (h1 : value_tenths = 0.8) (h2 : value_hundredths = 0.08) : 
  value_tenths - value_hundredths = 0.72 :=
by 
  sorry

end difference_between_eights_l1911_191155


namespace fencing_rate_3_rs_per_meter_l1911_191126

noncomputable def rate_per_meter (A_hectares : ℝ) (total_cost : ℝ) : ℝ := 
  let A_m2 := A_hectares * 10000
  let r := Real.sqrt (A_m2 / Real.pi)
  let C := 2 * Real.pi * r
  total_cost / C

theorem fencing_rate_3_rs_per_meter : rate_per_meter 17.56 4456.44 = 3.00 :=
by 
  sorry

end fencing_rate_3_rs_per_meter_l1911_191126


namespace yuan_to_scientific_notation_l1911_191109

/-- Express 2.175 billion yuan in scientific notation,
preserving three significant figures. --/
theorem yuan_to_scientific_notation (a : ℝ) (h : a = 2.175 * 10^9) : a = 2.18 * 10^9 :=
sorry

end yuan_to_scientific_notation_l1911_191109


namespace reciprocal_neg_2023_l1911_191135

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l1911_191135


namespace find_value_of_a_l1911_191167

theorem find_value_of_a 
  (P : ℝ × ℝ)
  (a : ℝ)
  (α : ℝ)
  (point_on_terminal_side : P = (-4, a))
  (sin_cos_condition : Real.sin α * Real.cos α = Real.sqrt 3 / 4) : 
  a = -4 * Real.sqrt 3 ∨ a = - (4 * Real.sqrt 3 / 3) :=
sorry

end find_value_of_a_l1911_191167


namespace financial_outcome_l1911_191106

theorem financial_outcome :
  let initial_value : ℝ := 12000
  let selling_price : ℝ := initial_value * 1.20
  let buying_price : ℝ := selling_price * 0.85
  let financial_outcome : ℝ := buying_price - initial_value
  financial_outcome = 240 :=
by
  sorry

end financial_outcome_l1911_191106


namespace slips_with_number_three_l1911_191157

theorem slips_with_number_three : 
  ∀ (total_slips : ℕ) (number3 number8 : ℕ) (E : ℚ), 
  total_slips = 15 → 
  E = 5.6 → 
  number3 + number8 = total_slips → 
  (number3 : ℚ) / total_slips * 3 + (number8 : ℚ) / total_slips * 8 = E →
  number3 = 8 :=
by
  intros total_slips number3 number8 E h1 h2 h3 h4
  sorry

end slips_with_number_three_l1911_191157


namespace range_a_mul_b_sub_three_half_l1911_191124

theorem range_a_mul_b_sub_three_half (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : b = (1 + Real.sqrt 5) / 2 * a) :
  (∃ l u : ℝ, ∀ f, l ≤ f ∧ f < u ↔ f = a * (b - 3 / 2)) :=
sorry

end range_a_mul_b_sub_three_half_l1911_191124


namespace boat_distance_ratio_l1911_191156

theorem boat_distance_ratio :
  ∀ (D_u D_d : ℝ),
  (3.6 = (D_u + D_d) / ((D_u / 4) + (D_d / 6))) →
  D_u / D_d = 4 :=
by
  intros D_u D_d h
  sorry

end boat_distance_ratio_l1911_191156


namespace carpooling_plans_l1911_191161

def last_digits (jia : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ) (friend4 : ℕ) : Prop :=
  jia = 0 ∧ friend1 = 0 ∧ friend2 = 2 ∧ friend3 = 1 ∧ friend4 = 5

def total_car_plans : Prop :=
  ∀ (jia friend1 friend2 friend3 friend4 : ℕ),
    last_digits jia friend1 friend2 friend3 friend4 →
    (∃ num_ways : ℕ, num_ways = 64)

theorem carpooling_plans : total_car_plans :=
sorry

end carpooling_plans_l1911_191161


namespace excess_calories_l1911_191100

-- Conditions
def calories_from_cheezits (bags: ℕ) (ounces_per_bag: ℕ) (calories_per_ounce: ℕ) : ℕ :=
  bags * ounces_per_bag * calories_per_ounce

def calories_from_chocolate_bars (bars: ℕ) (calories_per_bar: ℕ) : ℕ :=
  bars * calories_per_bar

def calories_from_popcorn (calories: ℕ) : ℕ :=
  calories

def calories_burned_running (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_swimming (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_cycling (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

-- Hypothesis
def total_calories_consumed : ℕ :=
  calories_from_cheezits 3 2 150 + calories_from_chocolate_bars 2 250 + calories_from_popcorn 500

def total_calories_burned : ℕ :=
  calories_burned_running 40 12 + calories_burned_swimming 30 15 + calories_burned_cycling 20 10

-- Theorem
theorem excess_calories : total_calories_consumed - total_calories_burned = 770 := by
  sorry

end excess_calories_l1911_191100


namespace problem_correct_l1911_191114

noncomputable def S : Set ℕ := {x | x^2 - x = 0}
noncomputable def T : Set ℕ := {x | x ∈ Set.univ ∧ 6 % (x - 2) = 0}

theorem problem_correct : S ∩ T = ∅ :=
by sorry

end problem_correct_l1911_191114


namespace female_officers_on_duty_percentage_l1911_191151

   def percentage_of_females_on_duty (total_on_duty : ℕ) (female_on_duty : ℕ) (total_females : ℕ) : ℕ :=
   (female_on_duty * 100) / total_females
  
   theorem female_officers_on_duty_percentage
     (total_on_duty : ℕ) (h1 : total_on_duty = 180)
     (female_on_duty : ℕ) (h2 : female_on_duty = total_on_duty / 2)
     (total_females : ℕ) (h3 : total_females = 500) :
     percentage_of_females_on_duty total_on_duty female_on_duty total_females = 18 :=
   by
     rw [h1, h2, h3]
     sorry
   
end female_officers_on_duty_percentage_l1911_191151


namespace plane_ticket_price_l1911_191181

theorem plane_ticket_price :
  ∀ (P : ℕ),
  (20 * 155) + 2900 = 30 * P →
  P = 200 := 
by
  sorry

end plane_ticket_price_l1911_191181


namespace angle_addition_l1911_191191

open Real

theorem angle_addition (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : tan α = 1 / 3) (h₄ : cos β = 3 / 5) : α + 3 * β = 3 * π / 4 :=
by
  sorry

end angle_addition_l1911_191191


namespace K9_le_89_K9_example_171_l1911_191137

section weights_proof

def K (n : ℕ) (P : ℕ) : ℕ := sorry -- Assume the definition of K given by the problem

theorem K9_le_89 : ∀ P, K 9 P ≤ 89 := by
  sorry -- Proof to be filled

def example_weight : ℕ := 171

theorem K9_example_171 : K 9 example_weight = 89 := by
  sorry -- Proof to be filled

end weights_proof

end K9_le_89_K9_example_171_l1911_191137


namespace find_x_in_inches_l1911_191150

theorem find_x_in_inches (x : ℝ) :
  let area_smaller_square := 9 * x^2
  let area_larger_square := 36 * x^2
  let area_triangle := 9 * x^2
  area_smaller_square + area_larger_square + area_triangle = 1950 → x = (5 * Real.sqrt 13) / 3 :=
by
  sorry

end find_x_in_inches_l1911_191150


namespace Peter_speed_is_correct_l1911_191195

variable (Peter_speed : ℝ)

def Juan_speed : ℝ := Peter_speed + 3

def distance_Peter_in_1_5_hours : ℝ := 1.5 * Peter_speed

def distance_Juan_in_1_5_hours : ℝ := 1.5 * Juan_speed Peter_speed

theorem Peter_speed_is_correct (h : distance_Peter_in_1_5_hours Peter_speed + distance_Juan_in_1_5_hours Peter_speed = 19.5) : Peter_speed = 5 :=
by
  sorry

end Peter_speed_is_correct_l1911_191195


namespace Ann_age_is_46_l1911_191146

theorem Ann_age_is_46
  (a b : ℕ) 
  (h1 : a + b = 72)
  (h2 : b = (a / 3) + 2 * (a - b)) : a = 46 :=
by
  sorry

end Ann_age_is_46_l1911_191146


namespace complement_union_eq_l1911_191131

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l1911_191131


namespace sum_gcd_lcm_of_4_and_10_l1911_191110

theorem sum_gcd_lcm_of_4_and_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 :=
by
  sorry

end sum_gcd_lcm_of_4_and_10_l1911_191110


namespace unique_b_for_quadratic_l1911_191180

theorem unique_b_for_quadratic (c : ℝ) (h_c : c ≠ 0) : (∃! b : ℝ, b > 0 ∧ (2*b + 2/b)^2 - 4*c = 0) → c = 4 :=
by
  sorry

end unique_b_for_quadratic_l1911_191180


namespace bicycle_wheels_l1911_191170

theorem bicycle_wheels :
  ∃ b : ℕ, 
  (∃ (num_bicycles : ℕ) (num_tricycles : ℕ) (wheels_per_tricycle : ℕ) (total_wheels : ℕ),
    num_bicycles = 16 ∧ 
    num_tricycles = 7 ∧ 
    wheels_per_tricycle = 3 ∧ 
    total_wheels = 53 ∧ 
    16 * b + num_tricycles * wheels_per_tricycle = total_wheels) ∧ 
  b = 2 :=
by
  sorry

end bicycle_wheels_l1911_191170


namespace initial_boys_count_l1911_191176

theorem initial_boys_count (B : ℕ) (boys girls : ℕ)
  (h1 : boys = 3 * B)                             -- The ratio of boys to girls is 3:4
  (h2 : girls = 4 * B)                            -- The ratio of boys to girls is 3:4
  (h3 : boys - 10 = 4 * (girls - 20))             -- The final ratio after transfer is 4:5
  : boys = 90 :=                                  -- Prove initial boys count was 90
by 
  sorry

end initial_boys_count_l1911_191176
