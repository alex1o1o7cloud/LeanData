import Mathlib

namespace find_integer_n_l1762_176260

theorem find_integer_n (n : ℤ) (h1 : n ≥ 3) (h2 : ∃ k : ℚ, k * k = (n^2 - 5) / (n + 1)) : n = 3 := by
  sorry

end find_integer_n_l1762_176260


namespace laura_running_speed_l1762_176285

noncomputable def running_speed (x : ℝ) : ℝ := x^2 - 1

noncomputable def biking_speed (x : ℝ) : ℝ := 3 * x + 2

noncomputable def biking_time (x: ℝ) : ℝ := 30 / (biking_speed x)

noncomputable def running_time (x: ℝ) : ℝ := 5 / (running_speed x)

noncomputable def total_motion_time (x : ℝ) : ℝ := biking_time x + running_time x

-- Laura's total workout duration without transition time
noncomputable def required_motion_time : ℝ := 140 / 60

theorem laura_running_speed (x : ℝ) (hx : total_motion_time x = required_motion_time) :
  running_speed x = 83.33 :=
sorry

end laura_running_speed_l1762_176285


namespace third_smallest_is_four_probability_l1762_176213

noncomputable def probability_third_smallest_is_four : ℚ :=
  let total_ways := Nat.choose 12 7
  let favorable_ways := (Nat.choose 3 2) * (Nat.choose 8 4)
  favorable_ways / total_ways

theorem third_smallest_is_four_probability : 
  probability_third_smallest_is_four = 35 / 132 := 
sorry

end third_smallest_is_four_probability_l1762_176213


namespace school_avg_GPA_l1762_176258

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l1762_176258


namespace maximum_value_of_k_l1762_176293

noncomputable def max_k (m : ℝ) : ℝ := 
  if 0 < m ∧ m < 1 / 2 then 
    1 / m + 2 / (1 - 2 * m) 
  else 
    0

theorem maximum_value_of_k : ∀ m : ℝ, (0 < m ∧ m < 1 / 2) → (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m) ≥ k) → k ≤ 8) :=
  sorry

end maximum_value_of_k_l1762_176293


namespace smallest_base_l1762_176296

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end smallest_base_l1762_176296


namespace wire_cut_equal_area_l1762_176280

theorem wire_cut_equal_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a / b = 2 / Real.sqrt Real.pi) ↔ (a^2 / 16 = b^2 / (4 * Real.pi)) :=
by
  sorry

end wire_cut_equal_area_l1762_176280


namespace ordered_triples_lcm_sum_zero_l1762_176233

theorem ordered_triples_lcm_sum_zero :
  ∀ (x y z : ℕ), 
    (0 < x) → 
    (0 < y) → 
    (0 < z) → 
    Nat.lcm x y = 180 →
    Nat.lcm x z = 450 →
    Nat.lcm y z = 600 →
    x + y + z = 120 →
    false := 
by
  intros x y z hx hy hz hxy hxz hyz hs
  sorry

end ordered_triples_lcm_sum_zero_l1762_176233


namespace parabola_directrix_l1762_176255

noncomputable def equation_of_directrix (a h k : ℝ) : ℝ :=
  k - 1 / (4 * a)

theorem parabola_directrix:
  ∀ (a h k : ℝ), a = -3 ∧ h = 1 ∧ k = -2 → equation_of_directrix a h k = - 23 / 12 :=
by
  intro a h k
  intro h_ahk
  sorry

end parabola_directrix_l1762_176255


namespace min_hours_to_pass_message_ge_55_l1762_176208

theorem min_hours_to_pass_message_ge_55 : 
  ∃ (n: ℕ), (∀ m: ℕ, m < n → 2^(m+1) - 2 ≤ 55) ∧ 2^(n+1) - 2 > 55 :=
by sorry

end min_hours_to_pass_message_ge_55_l1762_176208


namespace find_c_value_l1762_176292

def f (c : ℝ) (x : ℝ) : ℝ := c * x^4 + (c^2 - 3) * x^2 + 1

theorem find_c_value (c : ℝ) :
  (∀ x < -1, deriv (f c) x < 0) ∧ 
  (∀ x, -1 < x → x < 0 → deriv (f c) x > 0) → 
  c = 1 :=
by 
  sorry

end find_c_value_l1762_176292


namespace age_problem_solution_l1762_176217

theorem age_problem_solution 
  (x : ℕ) 
  (xiaoxiang_age : ℕ := 5) 
  (father_age : ℕ := 48) 
  (mother_age : ℕ := 42) 
  (h : (father_age + x) + (mother_age + x) = 6 * (xiaoxiang_age + x)) : 
  x = 15 :=
by {
  -- To be proved
  sorry
}

end age_problem_solution_l1762_176217


namespace cos_A_equals_one_third_l1762_176264

-- Noncomputable context as trigonometric functions are involved.
noncomputable def cosA_in_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  let law_of_cosines : (a * Real.cos B) = (3 * c - b) * Real.cos A := sorry
  (Real.cos A = 1 / 3)

-- Define the problem statement to be proved
theorem cos_A_equals_one_third (a b c A B C : ℝ) 
  (h1 : a = Real.cos B)
  (h2 : a * Real.cos B = (3 * c - b) * Real.cos A) :
  Real.cos A = 1 / 3 := 
by 
  -- Placeholder for the actual proof
  sorry

end cos_A_equals_one_third_l1762_176264


namespace patio_length_four_times_width_l1762_176214

theorem patio_length_four_times_width (w l : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 :=
by
  sorry

end patio_length_four_times_width_l1762_176214


namespace range_of_b_l1762_176210

theorem range_of_b (f g : ℝ → ℝ) (a b : ℝ)
  (hf : ∀ x, f x = Real.exp x - 1)
  (hg : ∀ x, g x = -x^2 + 4*x - 3)
  (h : f a = g b) :
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by
  sorry

end range_of_b_l1762_176210


namespace minimum_value_sqrt_m2_n2_l1762_176241

theorem minimum_value_sqrt_m2_n2 
  (a b m n : ℝ)
  (h1 : a^2 + b^2 = 3)
  (h2 : m*a + n*b = 3) : 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ Real.sqrt (m^2 + n^2) = k :=
by
  sorry

end minimum_value_sqrt_m2_n2_l1762_176241


namespace man_born_year_l1762_176250

theorem man_born_year (x : ℕ) : 
  (x^2 - x = 1806) ∧ (x^2 - x < 1850) ∧ (40 < x) ∧ (x < 50) → x = 43 :=
by
  sorry

end man_born_year_l1762_176250


namespace union_A_B_complement_intersect_B_intersection_sub_C_l1762_176225

-- Define set A
def A : Set ℝ := {x | -5 < x ∧ x < 1}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 8}

-- Define set C with variable parameter a
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Problem (1): Prove A ∪ B = { x | -5 < x < 8 }
theorem union_A_B : A ∪ B = {x | -5 < x ∧ x < 8} := 
by sorry

-- Problem (1): Prove (complement R A) ∩ B = { x | 1 ≤ x < 8 }
theorem complement_intersect_B : (Aᶜ) ∩ B = {x | 1 ≤ x ∧ x < 8} :=
by sorry

-- Problem (2): If A ∩ B ⊆ C, prove a ≥ 1
theorem intersection_sub_C (a : ℝ) (h : A ∩ B ⊆ C a) : 1 ≤ a :=
by sorry

end union_A_B_complement_intersect_B_intersection_sub_C_l1762_176225


namespace ratio_area_circle_to_triangle_l1762_176226

theorem ratio_area_circle_to_triangle (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
    (π * r) / (h + r) = (π * r ^ 2) / (r * (h + r)) := sorry

end ratio_area_circle_to_triangle_l1762_176226


namespace nuts_in_trail_mix_l1762_176231

theorem nuts_in_trail_mix :
  let walnuts := 0.25
  let almonds := 0.25
  walnuts + almonds = 0.50 :=
by
  sorry

end nuts_in_trail_mix_l1762_176231


namespace problem_statement_l1762_176242

variables (P Q : Prop)

theorem problem_statement (h1 : ¬P) (h2 : ¬(P ∧ Q)) : ¬(P ∨ Q) :=
sorry

end problem_statement_l1762_176242


namespace chimney_bricks_l1762_176228

theorem chimney_bricks (x : ℝ) 
  (h1 : ∀ x, Brenda_rate = x / 8) 
  (h2 : ∀ x, Brandon_rate = x / 12) 
  (h3 : Combined_rate = (Brenda_rate + Brandon_rate - 15)) 
  (h4 : x = Combined_rate * 6) 
  : x = 360 := 
by 
  sorry

end chimney_bricks_l1762_176228


namespace evaluate_fraction_l1762_176283

-- Define the custom operations x@y and x#y
def op_at (x y : ℝ) : ℝ := x * y - y^2
def op_hash (x y : ℝ) : ℝ := x + y - x * y^2 + x^2

-- State the proof goal
theorem evaluate_fraction : (op_at 7 3) / (op_hash 7 3) = -3 :=
by
  -- Calculations to prove the theorem
  sorry

end evaluate_fraction_l1762_176283


namespace area_of_inscribed_square_l1762_176224

theorem area_of_inscribed_square :
  let parabola := λ x => x^2 - 10 * x + 21
  ∃ (t : ℝ), parabola (5 + t) = -2 * t ∧ (2 * t)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end area_of_inscribed_square_l1762_176224


namespace probability_penny_dime_halfdollar_tails_is_1_over_8_l1762_176259

def probability_penny_dime_halfdollar_tails : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_penny_dime_halfdollar_tails_is_1_over_8 :
  probability_penny_dime_halfdollar_tails = 1 / 8 :=
by
  sorry

end probability_penny_dime_halfdollar_tails_is_1_over_8_l1762_176259


namespace linda_savings_l1762_176249

theorem linda_savings (S : ℝ) 
  (h1 : ∃ f : ℝ, f = 0.9 * 1/2 * S) -- She spent half of her savings on furniture with a 10% discount
  (h2 : ∃ t : ℝ, t = 1/2 * S * 1.05) -- The rest of her savings, spent on TV, had a 5% sales tax applied
  (h3 : 1/2 * S * 1.05 = 300) -- The total cost of the TV after tax was $300
  : S = 571.42 := 
sorry

end linda_savings_l1762_176249


namespace count_three_digit_integers_divisible_by_11_and_5_l1762_176243

def count_three_digit_multiples (a b: ℕ) : ℕ :=
  let lcm := Nat.lcm a b
  let first_multiple := (100 + lcm - 1) / lcm
  let last_multiple := 999 / lcm
  last_multiple - first_multiple + 1

theorem count_three_digit_integers_divisible_by_11_and_5 : 
  count_three_digit_multiples 11 5 = 17 := by 
  sorry

end count_three_digit_integers_divisible_by_11_and_5_l1762_176243


namespace alpha_inverse_proportional_beta_l1762_176295

theorem alpha_inverse_proportional_beta (α β : ℝ) (k : ℝ) :
  (∀ β1 α1, α1 * β1 = k) → (4 * 2 = k) → (β = -3) → (α = -8/3) :=
by
  sorry

end alpha_inverse_proportional_beta_l1762_176295


namespace clubsuit_problem_l1762_176256

def clubsuit (x y : ℤ) : ℤ :=
  (x^2 + y^2) * (x - y)

theorem clubsuit_problem : clubsuit 2 (clubsuit 3 4) = 16983 := 
by 
  sorry

end clubsuit_problem_l1762_176256


namespace number_exceeds_80_by_120_l1762_176275

theorem number_exceeds_80_by_120 : ∃ x : ℝ, x = 0.80 * x + 120 ∧ x = 600 :=
by sorry

end number_exceeds_80_by_120_l1762_176275


namespace lucy_fish_bought_l1762_176218

def fish_bought (fish_original fish_now : ℕ) : ℕ :=
  fish_now - fish_original

theorem lucy_fish_bought : fish_bought 212 492 = 280 :=
by
  sorry

end lucy_fish_bought_l1762_176218


namespace sum_powers_seventh_l1762_176236

/-- Given the sequence values for sums of powers of 'a' and 'b', prove the value of the sum of the 7th powers. -/
theorem sum_powers_seventh (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := 
  sorry

end sum_powers_seventh_l1762_176236


namespace malfunctioning_clock_fraction_correct_l1762_176248

noncomputable def malfunctioning_clock_correct_time_fraction : ℚ := 5 / 8

theorem malfunctioning_clock_fraction_correct :
  malfunctioning_clock_correct_time_fraction = 5 / 8 := 
by
  sorry

end malfunctioning_clock_fraction_correct_l1762_176248


namespace area_of_isosceles_triangle_l1762_176204

theorem area_of_isosceles_triangle
  (h : ℝ)
  (s : ℝ)
  (b : ℝ)
  (altitude : h = 10)
  (perimeter : s + (s - 2) + 2 * b = 40)
  (pythagoras : b^2 + h^2 = s^2) :
  (b * h) = 81.2 :=
by
  sorry

end area_of_isosceles_triangle_l1762_176204


namespace option_A_option_B_option_C_option_D_l1762_176251

theorem option_A : (-(-1) : ℤ) ≠ -|(-1 : ℤ)| := by
  sorry

theorem option_B : ((-3)^2 : ℤ) ≠ -(3^2 : ℤ) := by
  sorry

theorem option_C : ((-4)^3 : ℤ) = -(4^3 : ℤ) := by
  sorry

theorem option_D : ((2^2 : ℚ)/3) ≠ ((2/3)^2 : ℚ) := by
  sorry

end option_A_option_B_option_C_option_D_l1762_176251


namespace angle_measure_is_zero_l1762_176262

-- Definitions corresponding to conditions
variable (x : ℝ)

def complement (x : ℝ) := 90 - x
def supplement (x : ℝ) := 180 - x

-- Final proof statement
theorem angle_measure_is_zero (h : complement x = (1 / 2) * supplement x) : x = 0 :=
  sorry

end angle_measure_is_zero_l1762_176262


namespace at_most_two_even_l1762_176298

-- Assuming the negation of the proposition
def negate_condition (a b c : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0

-- Proposition to prove by contradiction
theorem at_most_two_even 
  (a b c : ℕ) 
  (h : negate_condition a b c) 
  : False :=
sorry

end at_most_two_even_l1762_176298


namespace gcd_2873_1349_gcd_4562_275_l1762_176291

theorem gcd_2873_1349 : Nat.gcd 2873 1349 = 1 := 
sorry

theorem gcd_4562_275 : Nat.gcd 4562 275 = 1 := 
sorry

end gcd_2873_1349_gcd_4562_275_l1762_176291


namespace total_time_to_watch_movie_l1762_176211

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l1762_176211


namespace inequality_solution_l1762_176290

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l1762_176290


namespace ryan_bus_meet_exactly_once_l1762_176278

-- Define respective speeds of Ryan and the bus
def ryan_speed : ℕ := 6 
def bus_speed : ℕ := 15 

-- Define bench placement and stop times
def bench_distance : ℕ := 300 
def regular_stop_time : ℕ := 45 
def extra_stop_time : ℕ := 90 

-- Initial positions
def ryan_initial_position : ℕ := 0
def bus_initial_position : ℕ := 300

-- Distance function D(t)
noncomputable def distance_at_time (t : ℕ) : ℤ :=
  let bus_travel_time : ℕ := 15  -- time for bus to travel 225 feet
  let bus_stop_time : ℕ := 45  -- time for bus to stop during regular stops
  let extended_stop_time : ℕ := 90  -- time for bus to stop during 3rd bench stops
  sorry -- calculation of distance function

-- Problem to prove: Ryan and the bus meet exactly once
theorem ryan_bus_meet_exactly_once : ∃ t₁ t₂ : ℕ, t₁ ≠ t₂ ∧ distance_at_time t₁ = 0 ∧ distance_at_time t₂ ≠ 0 := 
  sorry

end ryan_bus_meet_exactly_once_l1762_176278


namespace polar_to_cartesian_l1762_176282

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.sin θ) : 
  ∀ (x y : ℝ) (h₁ : x = ρ * Real.cos θ) (h₂ : y = ρ * Real.sin θ), 
    x^2 + (y - 1)^2 = 1 :=
by
  sorry

end polar_to_cartesian_l1762_176282


namespace like_terms_sum_l1762_176229

theorem like_terms_sum (m n : ℕ) (a b : ℝ) :
  (∀ c d : ℝ, -4 * a^(2 * m) * b^(3) = c * a^(6) * b^(n + 1)) →
  m + n = 5 :=
by 
  intro h
  sorry

end like_terms_sum_l1762_176229


namespace alice_bob_numbers_count_101_l1762_176268

theorem alice_bob_numbers_count_101 : 
  ∃ n : ℕ, (∀ x, 3 ≤ x ∧ x ≤ 2021 → (∃ k l, x = 3 + 5 * k ∧ x = 2021 - 4 * l)) → n = 101 :=
by
  sorry

end alice_bob_numbers_count_101_l1762_176268


namespace find_square_value_l1762_176209

theorem find_square_value (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : (8 * y - 4)^2 = 202 := 
by
  sorry

end find_square_value_l1762_176209


namespace number_line_4_units_away_l1762_176274

theorem number_line_4_units_away (x : ℝ) : |x + 3.2| = 4 ↔ (x = 0.8 ∨ x = -7.2) :=
by
  sorry

end number_line_4_units_away_l1762_176274


namespace part_one_solution_set_part_two_range_of_a_l1762_176238

def f (x : ℝ) : ℝ := abs (2 * x - 4) + abs (x + 1)

theorem part_one_solution_set :
  { x : ℝ | f x ≤ 9 } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

theorem part_two_range_of_a (a : ℝ) (B := { x : ℝ | x^2 - 3 * x < 0 })
  (A := { x : ℝ | f x < 2 * x + a }) :
  B ⊆ A → 5 ≤ a :=
sorry

end part_one_solution_set_part_two_range_of_a_l1762_176238


namespace bus_trip_distance_l1762_176200

variable (D : ℝ) (S : ℝ := 50)

theorem bus_trip_distance :
  (D / (S + 5) = D / S - 1) → D = 550 := by
  sorry

end bus_trip_distance_l1762_176200


namespace problem_1_problem_2_l1762_176287

-- Define the given conditions
variables (a c : ℝ) (cosB : ℝ)
variables (b : ℝ) (S : ℝ)

-- Assuming the values for the variables
axiom h₁ : a = 4
axiom h₂ : c = 3
axiom h₃ : cosB = 1 / 8

-- Prove that b = sqrt(22)
theorem problem_1 : b = Real.sqrt 22 := by
  sorry

-- Prove that the area of triangle ABC is 9 * sqrt(7) / 4
theorem problem_2 : S = 9 * Real.sqrt 7 / 4 := by 
  sorry

end problem_1_problem_2_l1762_176287


namespace number_of_zeros_l1762_176202

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x - 3

theorem number_of_zeros (b : ℝ) : 
  ∃ x₁ x₂ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ x₁ ≠ x₂ := by
  sorry

end number_of_zeros_l1762_176202


namespace Claire_plans_to_buy_five_cookies_l1762_176206

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

end Claire_plans_to_buy_five_cookies_l1762_176206


namespace fraction_evaluation_l1762_176267

theorem fraction_evaluation :
  (7 / 18 * (9 / 2) + 1 / 6) / ((40 / 3) - (15 / 4) / (5 / 16)) * (23 / 8) =
  4 + 17 / 128 :=
by
  -- conditions based on mixed number simplification
  have h1 : 4 + 1 / 2 = (9 : ℚ) / 2 := by sorry
  have h2 : 13 + 1 / 3 = (40 : ℚ) / 3 := by sorry
  have h3 : 3 + 3 / 4 = (15 : ℚ) / 4 := by sorry
  have h4 : 2 + 7 / 8 = (23 : ℚ) / 8 := by sorry
  -- the main proof
  sorry

end fraction_evaluation_l1762_176267


namespace max_xy_value_l1762_176289

theorem max_xy_value (x y : ℕ) (h : 27 * x + 35 * y ≤ 1000) : x * y ≤ 252 :=
sorry

end max_xy_value_l1762_176289


namespace largest_expression_is_A_l1762_176203

def expr_A := 1 - 2 + 3 + 4
def expr_B := 1 + 2 - 3 + 4
def expr_C := 1 + 2 + 3 - 4
def expr_D := 1 + 2 - 3 - 4
def expr_E := 1 - 2 - 3 + 4

theorem largest_expression_is_A : expr_A = 6 ∧ expr_A > expr_B ∧ expr_A > expr_C ∧ expr_A > expr_D ∧ expr_A > expr_E :=
  by sorry

end largest_expression_is_A_l1762_176203


namespace solve_equation_l1762_176299

-- Define the equation as a Lean proposition
def equation (x : ℝ) : Prop :=
  (6 * x + 3) / (3 * x^2 + 6 * x - 9) = 3 * x / (3 * x - 3)

-- Define the solution set
def solution (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2

-- Define the condition to avoid division by zero
def valid (x : ℝ) : Prop := x ≠ 1

-- State the theorem
theorem solve_equation (x : ℝ) (h : equation x) (hv : valid x) : solution x :=
by
  sorry

end solve_equation_l1762_176299


namespace exists_points_same_color_one_meter_apart_l1762_176253

-- Predicate to describe points in the 2x2 square
structure Point where
  x : ℝ
  y : ℝ
  h_x : 0 ≤ x ∧ x ≤ 2
  h_y : 0 ≤ y ∧ y ≤ 2

-- Function to describe the color assignment
def color (p : Point) : Prop := sorry -- True = Black, False = White

-- The main theorem to be proven
theorem exists_points_same_color_one_meter_apart :
  ∃ p1 p2 : Point, color p1 = color p2 ∧ dist (p1.1, p1.2) (p2.1, p2.2) = 1 :=
by
  sorry

end exists_points_same_color_one_meter_apart_l1762_176253


namespace range_of_m_l1762_176220

theorem range_of_m (m : ℝ) : 
  ((0 - m)^2 + (0 + m)^2 < 4) → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
by
  sorry

end range_of_m_l1762_176220


namespace bella_stamps_l1762_176215

theorem bella_stamps :
  let snowflake_cost := 1.05
  let truck_cost := 1.20
  let rose_cost := 0.90
  let butterfly_cost := 1.15
  let snowflake_spent := 15.75
  
  let snowflake_stamps := snowflake_spent / snowflake_cost
  let truck_stamps := snowflake_stamps + 11
  let rose_stamps := truck_stamps - 17
  let butterfly_stamps := 1.5 * rose_stamps
  
  let total_stamps := snowflake_stamps + truck_stamps + rose_stamps + butterfly_stamps
  
  total_stamps = 64 := by
  sorry

end bella_stamps_l1762_176215


namespace triangle_inequality_l1762_176240

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 < a / (b + c) + b / (c + a) + c / (a + b) ∧ a / (b + c) + b / (c + a) + c / (a + b) < 2 := 
sorry

end triangle_inequality_l1762_176240


namespace min_ab_eq_11_l1762_176265

theorem min_ab_eq_11 (a b : ℕ) (h : 23 * a - 13 * b = 1) : a + b = 11 :=
sorry

end min_ab_eq_11_l1762_176265


namespace find_f_2004_l1762_176272

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom odd_g : ∀ x : ℝ, g (-x) = -g x
axiom g_eq_f_shift : ∀ x : ℝ, g x = f (x - 1)
axiom g_one : g 1 = 2003

theorem find_f_2004 : f 2004 = 2003 :=
  sorry

end find_f_2004_l1762_176272


namespace solve_equation_l1762_176244

theorem solve_equation :
  ∀ (x m n : ℕ), 
    0 < x → 0 < m → 0 < n → 
    x^m = 2^(2 * n + 1) + 2^n + 1 →
    (x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4) :=
by
  sorry

end solve_equation_l1762_176244


namespace range_of_m_l1762_176223

theorem range_of_m (y : ℝ) (x : ℝ) (xy_ne_zero : x * y ≠ 0) :
  (x^2 + 4 * y^2 = (m^2 + 3 * m) * x * y) → -4 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l1762_176223


namespace equivalent_expression_l1762_176234

def evaluate_expression : ℚ :=
  let part1 := (2/3) * ((35/100) * 250)
  let part2 := ((75/100) * 150) / 16
  let part3 := (1/2) * ((40/100) * 500)
  part1 - part2 + part3

theorem equivalent_expression :
  evaluate_expression = 151.3020833333 :=  
by 
  sorry

end equivalent_expression_l1762_176234


namespace function_neither_even_nor_odd_l1762_176288

noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^3))

theorem function_neither_even_nor_odd :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) := by
  sorry

end function_neither_even_nor_odd_l1762_176288


namespace football_team_progress_l1762_176222

theorem football_team_progress (loss gain : ℤ) (h_loss : loss = -5) (h_gain : gain = 8) :
  (loss + gain = 3) :=
by
  sorry

end football_team_progress_l1762_176222


namespace green_ball_count_l1762_176269

theorem green_ball_count 
  (total_balls : ℕ)
  (n_red n_blue n_green : ℕ)
  (h_total : n_red + n_blue + n_green = 50)
  (h_red : ∀ (A : Finset ℕ), A.card = 34 -> ∃ a ∈ A, a < n_red)
  (h_blue : ∀ (A : Finset ℕ), A.card = 35 -> ∃ a ∈ A, a < n_blue)
  (h_green : ∀ (A : Finset ℕ), A.card = 36 -> ∃ a ∈ A, a < n_green)
  : n_green = 15 ∨ n_green = 16 ∨ n_green = 17 :=
by
  sorry

end green_ball_count_l1762_176269


namespace sequence_solution_l1762_176219

theorem sequence_solution (a : ℕ → ℝ) :
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m + n) = a m + a n - m * n) ∧ 
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m * n) = m^2 * a n + n^2 * a m + 2 * a m * a n) →
    (∀ n, a n = -n*(n-1)/2) ∨ (∀ n, a n = -n^2/2) :=
  by
  sorry

end sequence_solution_l1762_176219


namespace ball_arrangement_problem_l1762_176221

-- Defining the problem statement and conditions
theorem ball_arrangement_problem : 
  (∃ (A : ℕ), 
    (∀ (b : Fin 6 → ℕ), 
      (b 0 = 1 ∨ b 1 = 1) ∧ (b 0 = 2 ∨ b 1 = 2) ∧ -- 1 adjacent to 2
      b 4 ≠ 5 ∧ b 4 ≠ 6 ∧                 -- 5 not adjacent to 6 condition
      b 5 ≠ 5 ∧ b 5 ≠ 6     -- Add all other necessary conditions for arrangement
    ) →
    A = 144)
:= sorry

end ball_arrangement_problem_l1762_176221


namespace scientific_notation_of_138000_l1762_176277

noncomputable def scientific_notation_equivalent (n : ℕ) (a : ℝ) (exp : ℤ) : Prop :=
  n = a * (10:ℝ)^exp

theorem scientific_notation_of_138000 : scientific_notation_equivalent 138000 1.38 5 :=
by
  sorry

end scientific_notation_of_138000_l1762_176277


namespace good_horse_catches_up_l1762_176205

noncomputable def catch_up_days : ℕ := sorry

theorem good_horse_catches_up (x : ℕ) :
  (∀ (good_horse_speed slow_horse_speed head_start_duration : ℕ),
    good_horse_speed = 200 →
    slow_horse_speed = 120 →
    head_start_duration = 10 →
    200 * x = 120 * x + 120 * 10) →
  catch_up_days = x :=
by
  intro h
  have := h 200 120 10 rfl rfl rfl
  sorry

end good_horse_catches_up_l1762_176205


namespace function_for_negative_x_l1762_176245

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem function_for_negative_x {f : ℝ → ℝ} :
  odd_function f → given_function f → ∀ x, x < 0 → f x = x * (1 + x) :=
by
  intros h1 h2
  sorry

end function_for_negative_x_l1762_176245


namespace chennai_to_hyderabad_distance_l1762_176254

-- Definitions of the conditions
def david_speed := 50 -- mph
def lewis_speed := 70 -- mph
def meet_point := 250 -- miles from Chennai

-- Theorem statement
theorem chennai_to_hyderabad_distance :
  ∃ D T : ℝ, lewis_speed * T = D + (D - meet_point) ∧ david_speed * T = meet_point ∧ D = 300 :=
by
  sorry

end chennai_to_hyderabad_distance_l1762_176254


namespace not_divisible_by_24_l1762_176230

theorem not_divisible_by_24 : 
  ¬ (121416182022242628303234 % 24 = 0) := 
by
  sorry

end not_divisible_by_24_l1762_176230


namespace compare_fractions_l1762_176246

theorem compare_fractions : -(2 / 3 : ℚ) < -(3 / 5 : ℚ) :=
by sorry

end compare_fractions_l1762_176246


namespace find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l1762_176271

theorem find_zeros_of_quadratic {a b : ℝ} (h_a : a = 1) (h_b : b = -2) :
  ∀ x, (a * x^2 + b * x + b - 1 = 0) ↔ (x = 3 ∨ x = -1) := sorry

theorem range_of_a_for_two_distinct_zeros :
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + b - 1 = 0 ∧ a * x2^2 + b * x2 + b - 1 = 0) ↔ (0 < a ∧ a < 1) := sorry

end find_zeros_of_quadratic_range_of_a_for_two_distinct_zeros_l1762_176271


namespace simplify_expression_l1762_176281

theorem simplify_expression (x y z : ℤ) (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by
  sorry

end simplify_expression_l1762_176281


namespace minimum_cost_is_8600_l1762_176294

-- Defining the conditions
def shanghai_units : ℕ := 12
def nanjing_units : ℕ := 6
def suzhou_needs : ℕ := 10
def changsha_needs : ℕ := 8
def cost_shanghai_suzhou : ℕ := 400
def cost_shanghai_changsha : ℕ := 800
def cost_nanjing_suzhou : ℕ := 300
def cost_nanjing_changsha : ℕ := 500

-- Defining the function for total shipping cost
def total_shipping_cost (x : ℕ) : ℕ :=
  cost_shanghai_suzhou * x +
  cost_shanghai_changsha * (shanghai_units - x) +
  cost_nanjing_suzhou * (suzhou_needs - x) +
  cost_nanjing_changsha * (x - (shanghai_units - suzhou_needs))

-- Define the minimum shipping cost function
def minimum_shipping_cost : ℕ :=
  total_shipping_cost 10

-- State the theorem to prove
theorem minimum_cost_is_8600 : minimum_shipping_cost = 8600 :=
sorry

end minimum_cost_is_8600_l1762_176294


namespace calculate_l1762_176279

def q (x y : ℤ) : ℤ :=
  if x > 0 ∧ y ≥ 0 then x + 2*y
  else if x < 0 ∧ y ≤ 0 then x - 3*y
  else 4*x + 2*y

theorem calculate : q (q 2 (-2)) (q (-3) 1) = -4 := 
  by
    sorry

end calculate_l1762_176279


namespace find_x_l1762_176239

theorem find_x (x y z : ℕ) 
  (h1 : x + y = 74) 
  (h2 : (x + y) + y + z = 164) 
  (h3 : z - y = 16) : 
  x = 37 :=
sorry

end find_x_l1762_176239


namespace find_sum_of_distinct_numbers_l1762_176247

variable {R : Type} [LinearOrderedField R]

theorem find_sum_of_distinct_numbers (p q r s : R) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p ∧ r * s = -13 * q)
  (h2 : p + q = 12 * r ∧ p * q = -13 * s) :
  p + q + r + s = 2028 := 
by 
  sorry

end find_sum_of_distinct_numbers_l1762_176247


namespace smallest_positive_multiple_of_32_l1762_176276

theorem smallest_positive_multiple_of_32 : ∃ (n : ℕ), n > 0 ∧ n % 32 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 32 = 0 → n ≤ m :=
by
  sorry

end smallest_positive_multiple_of_32_l1762_176276


namespace interest_percentage_correct_l1762_176273

noncomputable def encyclopedia_cost : ℝ := 1200
noncomputable def down_payment : ℝ := 500
noncomputable def monthly_payment : ℝ := 70
noncomputable def final_payment : ℝ := 45
noncomputable def num_monthly_payments : ℕ := 12
noncomputable def total_installment_payments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_cost_paid : ℝ := total_installment_payments + down_payment
noncomputable def amount_borrowed : ℝ := encyclopedia_cost - down_payment
noncomputable def interest_paid : ℝ := total_cost_paid - encyclopedia_cost
noncomputable def interest_percentage : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_percentage_correct : interest_percentage = 26.43 := by
  sorry

end interest_percentage_correct_l1762_176273


namespace parabola_translation_l1762_176232

theorem parabola_translation :
  ∀ x : ℝ, (x^2 + 3) = ((x + 1)^2 + 3) :=
by
  skip -- proof is not needed; this is just the statement according to the instruction
  sorry

end parabola_translation_l1762_176232


namespace profit_equation_l1762_176261

noncomputable def price_and_profit (x : ℝ) : ℝ :=
  (1 + 0.5) * x * 0.8 - x

theorem profit_equation : ∀ x : ℝ, price_and_profit x = 8 → ((1 + 0.5) * x * 0.8 - x = 8) :=
 by intros x h
    exact h

end profit_equation_l1762_176261


namespace decreasing_interval_l1762_176286

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end decreasing_interval_l1762_176286


namespace exists_positive_integers_for_hexagon_area_l1762_176227

theorem exists_positive_integers_for_hexagon_area (S : ℕ) (a b : ℕ) (hS : S = 2016) :
  2 * (a^2 + b^2 + a * b) = S → ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 2 * (a^2 + b^2 + a * b) = S :=
by
  sorry

end exists_positive_integers_for_hexagon_area_l1762_176227


namespace ball_bounces_17_times_to_reach_below_2_feet_l1762_176212

theorem ball_bounces_17_times_to_reach_below_2_feet:
  ∃ k: ℕ, (∀ n, n < k → (800 * ((2: ℝ) / 3) ^ n) ≥ 2) ∧ (800 * ((2: ℝ) / 3) ^ k < 2) ∧ k = 17 :=
by
  sorry

end ball_bounces_17_times_to_reach_below_2_feet_l1762_176212


namespace inequality_proof_l1762_176263

theorem inequality_proof (x y z : ℝ) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ Real.sqrt 3 * (x * y + y * z + z * x) := 
  sorry

end inequality_proof_l1762_176263


namespace solution_set_of_inequality_l1762_176216

theorem solution_set_of_inequality (x : ℝ) : ((x - 1) * (2 - x) ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end solution_set_of_inequality_l1762_176216


namespace jimin_and_seokjin_total_l1762_176284

def Jimin_coins := (5 * 100) + (1 * 50)
def Seokjin_coins := (2 * 100) + (7 * 10)
def total_coins := Jimin_coins + Seokjin_coins

theorem jimin_and_seokjin_total : total_coins = 820 :=
by
  sorry

end jimin_and_seokjin_total_l1762_176284


namespace dot_product_is_2_l1762_176257

variable (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_is_2 (ha : a = (1, 0)) (hb : b = (2, 1)) :
  dot_product a b = 2 := by
  sorry

end dot_product_is_2_l1762_176257


namespace leo_current_weight_l1762_176252

theorem leo_current_weight (L K : ℕ) 
  (h1 : L + 10 = 3 * K / 2) 
  (h2 : L + K = 160)
  : L = 92 :=
sorry

end leo_current_weight_l1762_176252


namespace cindy_envelopes_left_l1762_176201

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l1762_176201


namespace time_to_be_d_miles_apart_l1762_176235

def mary_walk_rate := 4 -- Mary's walking rate in miles per hour
def sharon_walk_rate := 6 -- Sharon's walking rate in miles per hour
def time_to_be_3_miles_apart := 0.3 -- Time in hours to be 3 miles apart
def initial_distance := 3 -- They are 3 miles apart after 0.3 hours

theorem time_to_be_d_miles_apart (d: ℝ) : ∀ t: ℝ,
  (mary_walk_rate + sharon_walk_rate) * t = d ↔ 
  t = d / (mary_walk_rate + sharon_walk_rate) :=
by
  intros
  sorry

end time_to_be_d_miles_apart_l1762_176235


namespace find_cos_2beta_l1762_176297

noncomputable def cos_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (htan : Real.tan α = 1 / 7) (hcos : Real.cos (α + β) = 2 * Real.sqrt 5 / 5) : Real :=
  2 * (Real.cos β)^2 - 1

theorem find_cos_2beta (α β : ℝ) (h1: 0 < α ∧ α < π / 2) (h2: 0 < β ∧ β < π / 2)
  (htan: Real.tan α = 1 / 7) (hcos: Real.cos (α + β) = 2 * Real.sqrt 5 / 5) :
  cos_2beta α β h1 h2 htan hcos = 4 / 5 := 
sorry

end find_cos_2beta_l1762_176297


namespace trig_identity_example_l1762_176266

theorem trig_identity_example:
  (Real.sin (63 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) + 
  Real.cos (63 * Real.pi / 180) * Real.cos (108 * Real.pi / 180)) = 
  Real.sqrt 2 / 2 := 
by 
  sorry

end trig_identity_example_l1762_176266


namespace expression_simplification_l1762_176237

theorem expression_simplification (a : ℝ) (h : a ≠ 1) (h_beta : 1 = 1):
  (2^(Real.log (a) / Real.log (Real.sqrt 2)) - 
   3^((Real.log (a^2+1)) / (Real.log 27)) - 
   2 * a) / 
  (7^(4 * (Real.log (a) / Real.log 49)) - 
   5^((0.5 * Real.log (a)) / (Real.log (Real.sqrt 5))) - 1) = a^2 + a + 1 :=
by
  sorry

end expression_simplification_l1762_176237


namespace value_of_a5_max_sum_first_n_value_l1762_176207

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_first_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem value_of_a5 (a d a5 : ℤ) :
  a5 = 4 ↔ (2 * a + 4 * d) + (a + 4 * d) + (a + 8 * d) = (a + 5 * d) + 8 :=
  sorry

theorem max_sum_first_n_value (a d : ℤ) (n : ℕ) (max_n : ℕ) :
  a = 16 →
  d = -3 →
  (∀ i, sum_first_n a d i ≤ sum_first_n a d max_n) →
  max_n = 6 :=
  sorry

end value_of_a5_max_sum_first_n_value_l1762_176207


namespace john_profit_proof_l1762_176270

-- Define the conditions
variables 
  (parts_cost : ℝ := 800)
  (selling_price_multiplier : ℝ := 1.4)
  (monthly_build_quantity : ℝ := 60)
  (monthly_rent : ℝ := 5000)
  (monthly_extra_expenses : ℝ := 3000)

-- Define the computed variables based on conditions
def selling_price_per_computer := parts_cost * selling_price_multiplier
def total_revenue := monthly_build_quantity * selling_price_per_computer
def total_cost_of_components := monthly_build_quantity * parts_cost
def total_expenses := monthly_rent + monthly_extra_expenses
def profit_per_month := total_revenue - total_cost_of_components - total_expenses

-- The theorem statement of the proof
theorem john_profit_proof : profit_per_month = 11200 := 
by
  sorry

end john_profit_proof_l1762_176270
