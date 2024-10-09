import Mathlib

namespace shaded_region_perimeter_l1644_164477

theorem shaded_region_perimeter (r : ℝ) (θ : ℝ) (h₁ : r = 2) (h₂ : θ = 90) : 
  (2 * r + (2 * π * r * (1 - θ / 180))) = π + 4 := 
by sorry

end shaded_region_perimeter_l1644_164477


namespace third_derivative_correct_l1644_164464

noncomputable def func (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_correct :
  (deriv^[3] func) x = (4 / (1 + x^2)^2) :=
sorry

end third_derivative_correct_l1644_164464


namespace bowling_ball_weight_l1644_164417

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 9 * b = 6 * c) 
  (h2 : 4 * c = 120) : 
  b = 20 :=
by 
  sorry

end bowling_ball_weight_l1644_164417


namespace inequality_proof_l1644_164499

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1 / a - 1 / b + 1 / c) ≥ 1 :=
by
  sorry

end inequality_proof_l1644_164499


namespace hyperbola_focus_l1644_164429

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus
  (a b : ℝ)
  (hEq : ∀ x y : ℝ, ((x - 1)^2 / a^2) - ((y - 10)^2 / b^2) = 1):
  (1 + c 7 3, 10) = (1 + Real.sqrt (7^2 + 3^2), 10) :=
by
  sorry

end hyperbola_focus_l1644_164429


namespace negation_of_p_l1644_164416

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 := by
  sorry

end negation_of_p_l1644_164416


namespace sum_gcd_lcm_is_39_l1644_164449

theorem sum_gcd_lcm_is_39 : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by 
  sorry

end sum_gcd_lcm_is_39_l1644_164449


namespace solve_eq1_solve_eq2_l1644_164406

theorem solve_eq1 (x : ℝ) : x^2 - 6*x - 7 = 0 → x = 7 ∨ x = -1 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 1 = 2*x → x = 1 ∨ x = -1/3 :=
by
  sorry

end solve_eq1_solve_eq2_l1644_164406


namespace log_comparison_l1644_164493

open Real

noncomputable def a := log 4 / log 5  -- a = log_5(4)
noncomputable def b := log 5 / log 3  -- b = log_3(5)
noncomputable def c := log 5 / log 4  -- c = log_4(5)

theorem log_comparison : a < c ∧ c < b := 
by
  sorry

end log_comparison_l1644_164493


namespace lcm_quadruples_count_l1644_164450

-- Define the problem conditions
variables (r s : ℕ) (hr : r > 0) (hs : s > 0)

-- Define the mathematical problem statement
theorem lcm_quadruples_count :
  ( ∀ (a b c d : ℕ),
    lcm (lcm a b) c = lcm (lcm a b) d ∧
    lcm (lcm a b) c = lcm (lcm a c) d ∧
    lcm (lcm a b) c = lcm (lcm b c) d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a = 3 ^ r * 7 ^ s ∧
    b = 3 ^ r * 7 ^ s ∧
    c = 3 ^ r * 7 ^ s ∧
    d = 3 ^ r * 7 ^ s 
  → ∃ n, n = (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2)) :=
sorry

end lcm_quadruples_count_l1644_164450


namespace euclidean_steps_arbitrarily_large_l1644_164440

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

theorem euclidean_steps_arbitrarily_large (n : ℕ) (h : n ≥ 2) :
  gcd (fib (n+1)) (fib n) = gcd (fib 1) (fib 0) := 
sorry

end euclidean_steps_arbitrarily_large_l1644_164440


namespace problem_l1644_164423

theorem problem (x : ℝ) (h : x + 1/x = 10) :
  (x^2 + 1/x^2 = 98) ∧ (x^3 + 1/x^3 = 970) :=
by
  sorry

end problem_l1644_164423


namespace evaluate_expression_l1644_164404

theorem evaluate_expression (c : ℕ) (h : c = 4) : (c^c - c * (c - 1)^(c - 1))^c = 148^4 := 
by 
  sorry

end evaluate_expression_l1644_164404


namespace triangle_angle_y_l1644_164491

theorem triangle_angle_y (y : ℝ) (h : y + 3 * y + 45 = 180) : y = 33.75 :=
by
  have h1 : 4 * y + 45 = 180 := by sorry
  have h2 : 4 * y = 135 := by sorry
  have h3 : y = 33.75 := by sorry
  exact h3

end triangle_angle_y_l1644_164491


namespace mean_age_correct_l1644_164403

def children_ages : List ℕ := [6, 6, 9, 12]

def number_of_children : ℕ := 4

def sum_of_ages (ages : List ℕ) : ℕ := ages.sum

def mean_age (ages : List ℕ) (num_children : ℕ) : ℚ :=
  sum_of_ages ages / num_children

theorem mean_age_correct :
  mean_age children_ages number_of_children = 8.25 := by
  sorry

end mean_age_correct_l1644_164403


namespace amount_saved_per_person_l1644_164461

-- Definitions based on the conditions
def original_price := 60
def discounted_price := 48
def number_of_people := 3
def discount := original_price - discounted_price

-- Proving that each person paid 4 dollars less.
theorem amount_saved_per_person : discount / number_of_people = 4 :=
by
  sorry

end amount_saved_per_person_l1644_164461


namespace value_of_y_l1644_164452

theorem value_of_y (y : ℝ) (h : |y| = |y - 3|) : y = 3 / 2 :=
sorry

end value_of_y_l1644_164452


namespace equivalent_annual_rate_correct_l1644_164458

noncomputable def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def effective_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate / 100)^4

noncomputable def equivalent_annual_rate (annual_rate : ℝ) : ℝ :=
  (effective_annual_rate (quarterly_rate annual_rate) - 1) * 100

theorem equivalent_annual_rate_correct :
  equivalent_annual_rate 8 = 8.24 := 
by
  sorry

end equivalent_annual_rate_correct_l1644_164458


namespace inverse_proportion_relation_l1644_164496

variable (k : ℝ) (y1 y2 : ℝ) (h1 : y1 = - (2 / (-1))) (h2 : y2 = - (2 / (-2)))

theorem inverse_proportion_relation : y1 > y2 := by
  sorry

end inverse_proportion_relation_l1644_164496


namespace train_speed_approx_l1644_164425

noncomputable def distance_in_kilometers (d : ℝ) : ℝ :=
d / 1000

noncomputable def time_in_hours (t : ℝ) : ℝ :=
t / 3600

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ :=
distance_in_kilometers d / time_in_hours t

theorem train_speed_approx (d t : ℝ) (h_d : d = 200) (h_t : t = 5.80598713393251) :
  abs (speed_in_kmh d t - 124.019) < 1e-3 :=
by
  rw [h_d, h_t]
  simp only [distance_in_kilometers, time_in_hours, speed_in_kmh]
  norm_num
  -- We're using norm_num to deal with numerical approximations and constants
  -- The actual calculations can be verified through manual checks or external tools but in Lean we skip this step.
  sorry

end train_speed_approx_l1644_164425


namespace vertex_x_coord_l1644_164473

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Conditions based on given points
def conditions (a b c : ℝ) : Prop :=
  quadratic a b c 2 = 4 ∧
  quadratic a b c 8 =4 ∧
  quadratic a b c 10 = 13

-- Statement to prove the x-coordinate of the vertex is 5
theorem vertex_x_coord (a b c : ℝ) (h : conditions a b c) : 
  (-(b) / (2 * a)) = 5 :=
by
  sorry

end vertex_x_coord_l1644_164473


namespace calculate_expression_l1644_164474

theorem calculate_expression (x : ℝ) (h₁ : x ≠ 5) (h₂ : x = 4) : (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  sorry

end calculate_expression_l1644_164474


namespace area_of_lune_l1644_164492

theorem area_of_lune :
  ∃ (A L : ℝ), A = (3/2) ∧ L = 2 ∧
  (Lune_area : ℝ) = (9 * Real.sqrt 3 / 4) - (55 * π / 24) →
  Lune_area = (9 * Real.sqrt 3 / 4) - (55 * π / 24) :=
by
  sorry

end area_of_lune_l1644_164492


namespace smallest_m_l1644_164465

theorem smallest_m (m : ℕ) (h1 : 7 ≡ 2 [MOD 5]) : 
  (7^m ≡ m^7 [MOD 5]) ↔ (m = 7) :=
by sorry

end smallest_m_l1644_164465


namespace probability_drawing_balls_l1644_164432

theorem probability_drawing_balls :
  let total_balls := 15
  let red_balls := 10
  let blue_balls := 5
  let drawn_balls := 4
  let num_ways_to_draw_4_balls := Nat.choose total_balls drawn_balls
  let num_ways_to_draw_3_red_1_blue := (Nat.choose red_balls 3) * (Nat.choose blue_balls 1)
  let num_ways_to_draw_1_red_3_blue := (Nat.choose red_balls 1) * (Nat.choose blue_balls 3)
  let total_favorable_outcomes := num_ways_to_draw_3_red_1_blue + num_ways_to_draw_1_red_3_blue
  let probability := total_favorable_outcomes / num_ways_to_draw_4_balls
  probability = (140 : ℚ) / 273 :=
sorry

end probability_drawing_balls_l1644_164432


namespace handshakes_at_networking_event_l1644_164448

noncomputable def total_handshakes (n : ℕ) (exclude : ℕ) : ℕ :=
  (n * (n - 1 - exclude)) / 2

theorem handshakes_at_networking_event : total_handshakes 12 1 = 60 := by
  sorry

end handshakes_at_networking_event_l1644_164448


namespace three_digit_minuends_count_l1644_164421

theorem three_digit_minuends_count :
  ∀ a b c : ℕ, a - c = 4 ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  (∃ n : ℕ, n = 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c - 396 = 100 * c + 10 * b + a) →
  ∃ count : ℕ, count = 50 :=
by
  sorry

end three_digit_minuends_count_l1644_164421


namespace rectangle_area_l1644_164422

theorem rectangle_area (x : ℝ) (h : (2*x - 3) * (3*x + 4) = 20 * x - 12) : x = 7 / 2 :=
sorry

end rectangle_area_l1644_164422


namespace scientific_notation_l1644_164486

def significant_digits : ℝ := 4.032
def exponent : ℤ := 11
def original_number : ℝ := 403200000000

theorem scientific_notation : original_number = significant_digits * 10 ^ exponent := 
by
  sorry

end scientific_notation_l1644_164486


namespace F_2021_F_integer_F_divisibility_l1644_164443

/- Part 1 -/
def F (n : ℕ) : ℕ := 
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  let n' := 1000 * c + 100 * d + 10 * a + b
  (n + n') / 101

theorem F_2021 : F 2021 = 41 :=
  sorry

/- Part 2 -/
theorem F_integer (a b c d : ℕ) (ha : 1 ≤ a) (hb : a ≤ 9) (hc : 0 ≤ b) (hd : b ≤ 9)
(hc' : 0 ≤ c) (hd' : c ≤ 9) (hc'' : 0 ≤ d) (hd'' : d ≤ 9) :
  let n := 1000 * a + 100 * b + 10 * c + d
  let n' := 1000 * c + 100 * d + 10 * a + b
  F n = (101 * (10 * a + b + 10 * c + d)) / 101 :=
  sorry

/- Part 3 -/
theorem F_divisibility (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 5 ≤ b ∧ b ≤ 9) :
  let s := 3800 + 10 * a + b
  let t := 1000 * b + 100 * a + 13
  (3 * F t - F s) % 8 = 0 ↔ s = 3816 ∨ s = 3847 ∨ s = 3829 :=
  sorry

end F_2021_F_integer_F_divisibility_l1644_164443


namespace polygon_sides_l1644_164418

theorem polygon_sides (n : ℕ) 
  (H : (n * (n - 3)) / 2 = 3 * n) : n = 9 := 
sorry

end polygon_sides_l1644_164418


namespace sum_of_ages_l1644_164480

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l1644_164480


namespace probability_all_black_after_rotation_l1644_164490

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end probability_all_black_after_rotation_l1644_164490


namespace moral_of_saying_l1644_164424

/-!
  Comrade Mao Zedong said: "If you want to know the taste of a pear, you must change the pear and taste it yourself." 
  Prove that this emphasizes "Practice is the source of knowledge" (option C) over the other options.
-/

def question := "What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?"

def options := ["Knowledge is the driving force behind the development of practice", 
                "Knowledge guides practice", 
                "Practice is the source of knowledge", 
                "Practice has social and historical characteristics"]

def correct_answer := "Practice is the source of knowledge"

theorem moral_of_saying : (question, options[2]) ∈ [("What is the moral of Comrade Mao Zedong's famous saying about tasting a pear?", 
                                                      "Practice is the source of knowledge")] := by 
  sorry

end moral_of_saying_l1644_164424


namespace line_intersects_hyperbola_l1644_164438

theorem line_intersects_hyperbola 
  (k : ℝ)
  (hyp : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) :
  -Real.sqrt 15 / 3 < k ∧ k < -1 := 
sorry


end line_intersects_hyperbola_l1644_164438


namespace find_triple_sum_l1644_164481

theorem find_triple_sum (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = 1 - 4 * y)
  (h3 : x + y = -12 - 4 * z) :
  3 * x + 3 * y + 3 * z = 9 / 2 := 
sorry

end find_triple_sum_l1644_164481


namespace compute_expression_l1644_164439

theorem compute_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (h3 : x = 1 / z^2) : 
  (x - 1 / x) * (z^2 + 1 / z^2) = x^2 - z^4 :=
by
  sorry

end compute_expression_l1644_164439


namespace find_a_plus_b_l1644_164436

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end find_a_plus_b_l1644_164436


namespace fisherman_total_fish_l1644_164484

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l1644_164484


namespace find_angle_A_l1644_164488

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 2) (h3 : B = Real.pi / 4) : A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l1644_164488


namespace point_in_fourth_quadrant_l1644_164453

def point : ℝ × ℝ := (3, -2)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l1644_164453


namespace cyclist_north_speed_l1644_164495

variable {v : ℝ} -- Speed of the cyclist going north.

-- Conditions: 
def speed_south := 15 -- Speed of the cyclist going south (15 kmph).
def time := 2 -- The time after which they are 50 km apart (2 hours).
def distance := 50 -- The distance they are apart after 2 hours (50 km).

-- Theorem statement:
theorem cyclist_north_speed :
    (v + speed_south) * time = distance → v = 10 := by
  intro h
  sorry

end cyclist_north_speed_l1644_164495


namespace correct_simplification_l1644_164413

theorem correct_simplification (x y : ℝ) (hy : y ≠ 0):
  3 * x^4 * y / (x^2 * y) = 3 * x^2 :=
by
  sorry

end correct_simplification_l1644_164413


namespace average_difference_l1644_164444

theorem average_difference :
  let avg1 := (10 + 30 + 50) / 3
  let avg2 := (20 + 40 + 6) / 3
  avg1 - avg2 = 8 := by
  sorry

end average_difference_l1644_164444


namespace taxi_fare_l1644_164468

theorem taxi_fare :
  ∀ (initial_fee rate_per_increment increment_distance total_distance : ℝ),
    initial_fee = 2.35 →
    rate_per_increment = 0.35 →
    increment_distance = (2 / 5) →
    total_distance = 3.6 →
    (initial_fee + rate_per_increment * (total_distance / increment_distance)) = 5.50 :=
by
  intros initial_fee rate_per_increment increment_distance total_distance
  intro h1 h2 h3 h4
  sorry -- Proof is not required.

end taxi_fare_l1644_164468


namespace triangle_area_l1644_164415

theorem triangle_area : 
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  0.5 * base * height = 24.0 :=
by
  let p1 := (0, 0)
  let p2 := (0, 6)
  let p3 := (8, 15)
  let base := 6
  let height := 8
  sorry

end triangle_area_l1644_164415


namespace find_b_l1644_164482

theorem find_b (a c S : ℝ) (h₁ : a = 5) (h₂ : c = 2) (h₃ : S = 4) : 
  b = Real.sqrt 17 ∨ b = Real.sqrt 41 := by
  sorry

end find_b_l1644_164482


namespace campaign_fliers_l1644_164469

theorem campaign_fliers (total_fliers : ℕ) (fraction_morning : ℚ) (fraction_afternoon : ℚ) 
  (remaining_fliers_after_morning : ℕ) (remaining_fliers_after_afternoon : ℕ) :
  total_fliers = 1000 → fraction_morning = 1/5 → fraction_afternoon = 1/4 → 
  remaining_fliers_after_morning = total_fliers - total_fliers * fraction_morning → 
  remaining_fliers_after_afternoon = remaining_fliers_after_morning - remaining_fliers_after_morning * fraction_afternoon → 
  remaining_fliers_after_afternoon = 600 := 
by
  sorry

end campaign_fliers_l1644_164469


namespace cube_inequality_l1644_164462

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cube_inequality_l1644_164462


namespace wuxi_GDP_scientific_notation_l1644_164414

theorem wuxi_GDP_scientific_notation :
  14800 = 1.48 * 10^4 :=
sorry

end wuxi_GDP_scientific_notation_l1644_164414


namespace correct_option_C_l1644_164455

variable (a : ℝ)

theorem correct_option_C : (a^2 * a = a^3) :=
by sorry

end correct_option_C_l1644_164455


namespace geometric_sequence_general_formula_arithmetic_sequence_sum_l1644_164478

-- Problem (I)
theorem geometric_sequence_general_formula (a : ℕ → ℝ) (q a1 : ℝ)
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : a 1 + a 2 = 6)
  (h3 : a 1 * a 2 = a 3) :
  a n = 2 ^ n :=
sorry

-- Problem (II)
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (h1 : ∀ n, a n = 2 ^ n)
  (h2 : ∀ n, S n = (n * (b 1 + b n)) / 2)
  (h3 : ∀ n, S (2 * n + 1) = b n * b (n + 1))
  (h4 : ∀ n, b n = 2 * n + 1) :
  T n = 5 - (2 * n + 5) / 2 ^ n :=
sorry

end geometric_sequence_general_formula_arithmetic_sequence_sum_l1644_164478


namespace jungkook_colored_paper_count_l1644_164410

theorem jungkook_colored_paper_count :
  (3 * 10) + 8 = 38 :=
by sorry

end jungkook_colored_paper_count_l1644_164410


namespace milk_transfer_proof_l1644_164454

theorem milk_transfer_proof :
  ∀ (A B C x : ℝ), 
  A = 1232 →
  B = A - 0.625 * A → 
  C = A - B → 
  B + x = C - x → 
  x = 154 :=
by
  intros A B C x hA hB hC hEqual
  sorry

end milk_transfer_proof_l1644_164454


namespace compute_expression_l1644_164420

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end compute_expression_l1644_164420


namespace citizen_income_l1644_164428

theorem citizen_income (tax_paid : ℝ) (base_income : ℝ) (base_rate excess_rate : ℝ) (income : ℝ) 
  (h1 : 0 < base_income) (h2 : base_rate * base_income = 4400) (h3 : tax_paid = 8000)
  (h4 : excess_rate = 0.20) (h5 : base_rate = 0.11)
  (h6 : tax_paid = base_rate * base_income + excess_rate * (income - base_income)) :
  income = 58000 :=
sorry

end citizen_income_l1644_164428


namespace bus_travel_time_kimovsk_moscow_l1644_164479

noncomputable def travel_time_kimovsk_moscow (d1 d2 d3: ℝ) (max_speed: ℝ) (t_kt: ℝ) (t_nm: ℝ) : Prop :=
  35 ≤ d1 ∧ d1 ≤ 35 ∧
  60 ≤ d2 ∧ d2 ≤ 60 ∧
  200 ≤ d3 ∧ d3 ≤ 200 ∧
  max_speed <= 60 ∧
  2 ≤ t_kt ∧ t_kt ≤ 2 ∧
  5 ≤ t_nm ∧ t_nm ≤ 5 ∧
  (5 + 7/12 : ℝ) ≤ t_kt + t_nm ∧ t_kt + t_nm ≤ 6

theorem bus_travel_time_kimovsk_moscow
  (d1 d2 d3 : ℝ) (max_speed : ℝ) (t_kt : ℝ) (t_nm : ℝ) :
  travel_time_kimovsk_moscow d1 d2 d3 max_speed t_kt t_nm := 
by
  sorry

end bus_travel_time_kimovsk_moscow_l1644_164479


namespace tan_theta_eq_neg3_then_expr_eq_5_div_2_l1644_164433

theorem tan_theta_eq_neg3_then_expr_eq_5_div_2
  (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5 / 2 := 
sorry

end tan_theta_eq_neg3_then_expr_eq_5_div_2_l1644_164433


namespace expected_male_teachers_in_sample_l1644_164494

theorem expected_male_teachers_in_sample 
  (total_male total_female sample_size : ℕ) 
  (h1 : total_male = 56) 
  (h2 : total_female = 42) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 :=
by
  sorry

end expected_male_teachers_in_sample_l1644_164494


namespace expression_evaluation_l1644_164411

def eval_expression : Int := 
  let a := -2 ^ 3
  let b := abs (2 - 3)
  let c := -2 * (-1) ^ 2023
  a + b + c

theorem expression_evaluation :
  eval_expression = -5 :=
by
  sorry

end expression_evaluation_l1644_164411


namespace fraction_zero_implies_x_is_minus_one_l1644_164466

variable (x : ℝ)

theorem fraction_zero_implies_x_is_minus_one (h : (x^2 - 1) / (1 - x) = 0) : x = -1 :=
sorry

end fraction_zero_implies_x_is_minus_one_l1644_164466


namespace determine_words_per_page_l1644_164400

noncomputable def wordsPerPage (totalPages : ℕ) (wordsPerPage : ℕ) (totalWordsMod : ℕ) : ℕ :=
if totalPages * wordsPerPage % 250 = totalWordsMod ∧ wordsPerPage <= 200 then wordsPerPage else 0

theorem determine_words_per_page :
  wordsPerPage 150 198 137 = 198 :=
by 
  sorry

end determine_words_per_page_l1644_164400


namespace molecular_weight_of_10_moles_l1644_164489

-- Define the molecular weight of a compound as a constant
def molecular_weight (compound : Type) : ℝ := 840

-- Prove that the molecular weight of 10 moles of the compound is the same as the molecular weight of 1 mole of the compound
theorem molecular_weight_of_10_moles (compound : Type) :
  molecular_weight compound = 840 :=
by
  -- Proof
  sorry

end molecular_weight_of_10_moles_l1644_164489


namespace smallest_number_of_students_l1644_164441

/--
At a school, the ratio of 10th-graders to 8th-graders is 3:2, 
and the ratio of 10th-graders to 9th-graders is 5:3. 
Prove that the smallest number of students from these grades is 34.
-/
theorem smallest_number_of_students {G8 G9 G10 : ℕ} 
  (h1 : 3 * G8 = 2 * G10) (h2 : 5 * G9 = 3 * G10) : 
  G10 + G8 + G9 = 34 :=
by
  sorry

end smallest_number_of_students_l1644_164441


namespace nonagon_diagonals_l1644_164434

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end nonagon_diagonals_l1644_164434


namespace sum_of_coefficients_l1644_164401

-- Define the polynomial expansion and the target question
theorem sum_of_coefficients
  (x : ℝ)
  (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℝ)
  (h : (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + 
                        b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0) :
  (b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0) = 729 :=
by {
  -- We substitute x = 1 and show that the polynomial equals 729
  sorry
}

end sum_of_coefficients_l1644_164401


namespace thought_number_is_24_l1644_164456

variable (x : ℝ)

theorem thought_number_is_24 (h : x / 4 + 9 = 15) : x = 24 := by
  sorry

end thought_number_is_24_l1644_164456


namespace friendships_structure_count_l1644_164498

/-- In a group of 8 individuals, where each person has exactly 3 friends within the group,
there are 420 different ways to structure these friendships. -/
theorem friendships_structure_count : 
  ∃ (structure_count : ℕ), 
    structure_count = 420 ∧ 
    (∀ (G : Fin 8 → Fin 8 → Prop), 
      (∀ i, ∃! (j₁ j₂ j₃ : Fin 8), G i j₁ ∧ G i j₂ ∧ G i j₃) ∧ 
      (∀ i j, G i j → G j i) ∧ 
      (structure_count = 420)) := 
by
  sorry

end friendships_structure_count_l1644_164498


namespace find_a_geometric_sequence_l1644_164427

theorem find_a_geometric_sequence (a : ℤ) (T : ℕ → ℤ) (b : ℕ → ℤ) :
  (∀ n, T n = 3 ^ n + a) →
  b 1 = T 1 →
  (∀ n, n ≥ 2 → b n = T n - T (n - 1)) →
  (∀ n, n ≥ 2 → (∃ r, r * b n = b (n - 1))) →
  a = -1 :=
by
  sorry

end find_a_geometric_sequence_l1644_164427


namespace meals_for_children_l1644_164472

theorem meals_for_children (C : ℕ)
  (H1 : 70 * C = 70 * 45)
  (H2 : 70 * 45 = 2 * 45 * 35) :
  C = 90 :=
by
  sorry

end meals_for_children_l1644_164472


namespace range_of_k_for_distinct_real_roots_l1644_164409

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k - 1) * x1^2 - 2 * x1 + 1 = 0 ∧ (k - 1) * x2^2 - 2 * x2 + 1 = 0) →
    k < 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l1644_164409


namespace woman_born_second_half_20th_century_l1644_164483

theorem woman_born_second_half_20th_century (x : ℕ) (hx : 45 < x ∧ x < 50) (h_year : x * x = 2025) :
  x * x - x = 1980 :=
by {
  -- Add the crux of the problem here.
  sorry
}

end woman_born_second_half_20th_century_l1644_164483


namespace n_minus_m_l1644_164407

theorem n_minus_m (m n : ℤ) (h_m : m - 2 = 3) (h_n : n + 1 = 2) : n - m = -4 := sorry

end n_minus_m_l1644_164407


namespace town_population_growth_l1644_164402

noncomputable def populationAfterYears (population : ℝ) (year1Increase : ℝ) (year2Increase : ℝ) : ℝ :=
  let populationAfterFirstYear := population * (1 + year1Increase)
  let populationAfterSecondYear := populationAfterFirstYear * (1 + year2Increase)
  populationAfterSecondYear

theorem town_population_growth :
  ∀ (initialPopulation : ℝ) (year1Increase : ℝ) (year2Increase : ℝ),
    initialPopulation = 1000 → year1Increase = 0.10 → year2Increase = 0.20 →
      populationAfterYears initialPopulation year1Increase year2Increase = 1320 :=
by
  intros initialPopulation year1Increase year2Increase h1 h2 h3
  rw [h1, h2, h3]
  have h4 : populationAfterYears 1000 0.10 0.20 = 1320 := sorry
  exact h4

end town_population_growth_l1644_164402


namespace solve_system_l1644_164408

theorem solve_system 
    (x y z : ℝ) 
    (h1 : x + y - 2 + 4 * x * y = 0) 
    (h2 : y + z - 2 + 4 * y * z = 0) 
    (h3 : z + x - 2 + 4 * z * x = 0) :
    (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
sorry

end solve_system_l1644_164408


namespace highest_qualification_number_possible_l1644_164471

theorem highest_qualification_number_possible (n : ℕ) (qualifies : ℕ → ℕ → Prop)
    (h512 : n = 512)
    (hqualifies : ∀ a b, qualifies a b ↔ (a < b ∧ b - a ≤ 2)): 
    ∃ k, k = 18 ∧ (∀ m, qualifies m k → m < k) :=
by
  sorry

end highest_qualification_number_possible_l1644_164471


namespace actual_area_l1644_164475

open Real

theorem actual_area
  (scale : ℝ)
  (mapped_area_cm2 : ℝ)
  (actual_area_cm2 : ℝ)
  (actual_area_m2 : ℝ)
  (h_scale : scale = 1 / 50000)
  (h_mapped_area : mapped_area_cm2 = 100)
  (h_proportion : mapped_area_cm2 / actual_area_cm2 = scale ^ 2)
  : actual_area_m2 = 2.5 * 10^7 :=
by
  sorry

end actual_area_l1644_164475


namespace sum_of_coefficients_l1644_164463

theorem sum_of_coefficients:
  (∀ x : ℝ, (2*x - 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) →
  a_1 + a_3 + a_5 = -364 :=
by
  sorry

end sum_of_coefficients_l1644_164463


namespace sum_of_digits_l1644_164405

theorem sum_of_digits (d : ℕ) (h1 : d % 5 = 0) (h2 : 3 * d - 75 = d) : 
  (d / 10 + d % 10) = 11 :=
by {
  -- Placeholder for the proof
  sorry
}

end sum_of_digits_l1644_164405


namespace original_number_is_509_l1644_164497

theorem original_number_is_509 (n : ℕ) (h : n - 5 = 504) : n = 509 :=
by {
    sorry
}

end original_number_is_509_l1644_164497


namespace solve_abs_equation_l1644_164430

theorem solve_abs_equation (x : ℝ) :
  |2 * x - 1| + |x - 2| = |x + 1| ↔ 1 / 2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end solve_abs_equation_l1644_164430


namespace years_passed_l1644_164426

def initial_ages : List ℕ := [19, 34, 37, 42, 48]

def new_ages (x : ℕ) : List ℕ :=
  initial_ages.map (λ age => age + x)

-- Hypothesis: The new ages fit the following stem-and-leaf plot structure
def valid_stem_and_leaf (ages : List ℕ) : Bool :=
  ages = [25, 31, 34, 37, 43, 48]

theorem years_passed : ∃ x : ℕ, valid_stem_and_leaf (new_ages x) := by
  sorry

end years_passed_l1644_164426


namespace mix_alcohol_solutions_l1644_164431

-- Definitions capturing the conditions from part (a)
def volume_solution_y : ℝ := 600
def percent_alcohol_x : ℝ := 0.1
def percent_alcohol_y : ℝ := 0.3
def desired_percent_alcohol : ℝ := 0.25

-- The resulting Lean statement to prove question == answer given conditions
theorem mix_alcohol_solutions (Vx : ℝ) (h : (percent_alcohol_x * Vx + percent_alcohol_y * volume_solution_y) / (Vx + volume_solution_y) = desired_percent_alcohol) : Vx = 200 :=
sorry

end mix_alcohol_solutions_l1644_164431


namespace line_transformation_equiv_l1644_164445

theorem line_transformation_equiv :
  (∀ x y: ℝ, (2 * x - y - 3 = 0) ↔
    (7 * (x + 2 * y) - 5 * (-x + 4 * y) - 18 = 0)) :=
sorry

end line_transformation_equiv_l1644_164445


namespace odd_expression_l1644_164451

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_expression (k m : ℤ) (o := 2 * k + 3) (n := 2 * m) :
  is_odd (o^2 + n * o) :=
by sorry

end odd_expression_l1644_164451


namespace P_intersection_Q_is_singleton_l1644_164419

theorem P_intersection_Q_is_singleton :
  {p : ℝ × ℝ | p.1 + p.2 = 3} ∩ {p : ℝ × ℝ | p.1 - p.2 = 5} = { (4, -1) } :=
by
  -- The proof steps would go here.
  sorry

end P_intersection_Q_is_singleton_l1644_164419


namespace ordered_pairs_unique_solution_l1644_164435

theorem ordered_pairs_unique_solution :
  ∃! (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (b^2 - 4 * c = 0) ∧ (c^2 - 4 * b = 0) :=
sorry

end ordered_pairs_unique_solution_l1644_164435


namespace value_of_3Y5_l1644_164412

def Y (a b : ℤ) : ℤ := b + 10 * a - a^2 - b^2

theorem value_of_3Y5 : Y 3 5 = 1 := sorry

end value_of_3Y5_l1644_164412


namespace noodles_given_to_William_l1644_164446

def initial_noodles : ℝ := 54.0
def noodles_left : ℝ := 42.0
def noodles_given : ℝ := initial_noodles - noodles_left

theorem noodles_given_to_William : noodles_given = 12.0 := 
by
  sorry -- Proof to be filled in

end noodles_given_to_William_l1644_164446


namespace parabola_through_points_with_h_l1644_164460

noncomputable def quadratic_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_through_points_with_h (
    a h k : ℝ) 
    (H0 : quadratic_parabola a h k 0 = 4)
    (H1 : quadratic_parabola a h k 6 = 5)
    (H2 : a < 0)
    (H3 : 0 < h)
    (H4 : h < 6) : 
    h = 4 := 
sorry

end parabola_through_points_with_h_l1644_164460


namespace prime_sum_product_l1644_164467

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hsum : p + q = 102) (hgt : p > 30 ∨ q > 30) :
  p * q = 2201 := 
sorry

end prime_sum_product_l1644_164467


namespace find_original_b_l1644_164485

variable {a b c : ℝ}
variable (H_inv_prop : a * b = c) (H_a_increase : 1.20 * a * 80 = c)

theorem find_original_b : b = 96 :=
  by
  sorry

end find_original_b_l1644_164485


namespace simplify_fraction_l1644_164459

theorem simplify_fraction :
  (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end simplify_fraction_l1644_164459


namespace line_passes_fixed_point_l1644_164437

theorem line_passes_fixed_point (k b : ℝ) (h : -1 = (k + b) / 2) :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ y = k * x + b :=
by
  sorry

end line_passes_fixed_point_l1644_164437


namespace y_intercept_of_line_l1644_164487

def equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

theorem y_intercept_of_line : equation 0 (-2) :=
by
  sorry

end y_intercept_of_line_l1644_164487


namespace min_xy_min_x_plus_y_l1644_164457

theorem min_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : xy ≥ 36 :=
sorry  

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_xy_min_x_plus_y_l1644_164457


namespace angle_C_modified_l1644_164442

theorem angle_C_modified (A B C : ℝ) (h_eq_triangle: A = B) (h_C_modified: C = A + 40) (h_sum_angles: A + B + C = 180) : 
  C = 86.67 := 
by 
  sorry

end angle_C_modified_l1644_164442


namespace smallest_possible_value_of_other_integer_l1644_164476

theorem smallest_possible_value_of_other_integer (x b : ℕ) (h_gcd_lcm : ∀ m n : ℕ, m = 36 → gcd m n = x + 5 → lcm m n = x * (x + 5)) : 
  b > 0 → ∃ b, b = 1 ∧ gcd 36 b = x + 5 ∧ lcm 36 b = x * (x + 5) := 
by {
   sorry 
}

end smallest_possible_value_of_other_integer_l1644_164476


namespace quadratic_inequality_solution_l1644_164447

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end quadratic_inequality_solution_l1644_164447


namespace equal_expressions_l1644_164470

theorem equal_expressions : (-2)^3 = -(2^3) :=
by sorry

end equal_expressions_l1644_164470
