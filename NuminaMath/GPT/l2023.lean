import Mathlib

namespace sin_neg_thirtyone_sixths_pi_l2023_202393

theorem sin_neg_thirtyone_sixths_pi : Real.sin (-31 / 6 * Real.pi) = 1 / 2 :=
by 
  sorry

end sin_neg_thirtyone_sixths_pi_l2023_202393


namespace num_real_roots_eq_two_l2023_202392

theorem num_real_roots_eq_two : 
  ∀ x : ℝ, (∃ r : ℕ, r = 2 ∧ (abs (x^2 - 1) = 1/10 * (x + 9/10) → x = r)) := sorry

end num_real_roots_eq_two_l2023_202392


namespace smallest_a_satisfies_sin_condition_l2023_202315

open Real

theorem smallest_a_satisfies_sin_condition :
  ∃ (a : ℝ), (∀ x : ℤ, sin (a * x + 0) = sin (45 * x)) ∧ 0 ≤ a ∧ ∀ b : ℝ, (∀ x : ℤ, sin (b * x + 0) = sin (45 * x)) ∧ 0 ≤ b → 45 ≤ b :=
by
  -- To be proved.
  sorry

end smallest_a_satisfies_sin_condition_l2023_202315


namespace range_of_a_l2023_202374

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 ≤ x) → ∀ y : ℝ, (1 ≤ y) → (x ≤ y) → (Real.exp (abs (x - a)) ≤ Real.exp (abs (y - a)))) : a ≤ 1 :=
sorry

end range_of_a_l2023_202374


namespace pepper_left_l2023_202378

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem pepper_left (h1 : initial_pepper = 0.25) (h2 : used_pepper = 0.16) :
  initial_pepper - used_pepper = remaining_pepper :=
by
  sorry

end pepper_left_l2023_202378


namespace evaluate_statements_l2023_202311

-- Defining what it means for angles to be vertical
def vertical_angles (α β : ℝ) : Prop := α = β

-- Defining what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- Defining what supplementary angles are
def supplementary (α β : ℝ) : Prop := α + β = 180

-- Define the geometric properties for perpendicular and parallel lines
def unique_perpendicular_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, m * x + p.2 = l x

def unique_parallel_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, (l x ≠ m * x + p.2) ∧ (∀ y, y ≠ p.2 → l y ≠ m * y)

theorem evaluate_statements :
  (¬ ∃ α β, α = β ∧ vertical_angles α β) ∧
  (¬ ∃ α β, supplementary α β ∧ complementary α β) ∧
  ∃ l p, unique_perpendicular_through_point l p ∧
  ∃ l p, unique_parallel_through_point l p →
  2 = 2
  :=
by
  sorry  -- Proof is omitted

end evaluate_statements_l2023_202311


namespace find_numbers_l2023_202360

theorem find_numbers 
  (a b c d : ℝ)
  (h1 : b / c = c / a)
  (h2 : a + b + c = 19)
  (h3 : b - c = c - d)
  (h4 : b + c + d = 12) :
  (a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2) :=
sorry

end find_numbers_l2023_202360


namespace tom_age_ratio_l2023_202384

theorem tom_age_ratio (T N : ℝ) (h1 : T - N = 3 * (T - 4 * N)) : T / N = 5.5 :=
by
  sorry

end tom_age_ratio_l2023_202384


namespace value_of_a_l2023_202319

theorem value_of_a (a : ℝ) (h : abs (2 * a + 1) = 3) :
  a = -2 ∨ a = 1 :=
sorry

end value_of_a_l2023_202319


namespace probability_of_two_red_balls_l2023_202316

theorem probability_of_two_red_balls :
  let red_balls := 4
  let blue_balls := 4
  let green_balls := 2
  let total_balls := red_balls + blue_balls + green_balls
  let prob_red1 := (red_balls : ℚ) / total_balls
  let prob_red2 := ((red_balls - 1 : ℚ) / (total_balls - 1))
  (prob_red1 * prob_red2 = (2 : ℚ) / 15) :=
by
  sorry

end probability_of_two_red_balls_l2023_202316


namespace largest_of_set_l2023_202366

theorem largest_of_set : 
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  c = 2 ∧ (d < b ∧ b < a ∧ a < c) := by
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  sorry

end largest_of_set_l2023_202366


namespace function_domain_l2023_202328

open Set

noncomputable def domain_of_function : Set ℝ :=
  {x | x ≠ 2}

theorem function_domain :
  domain_of_function = {x : ℝ | x ≠ 2} :=
by sorry

end function_domain_l2023_202328


namespace purely_imaginary_complex_l2023_202362

theorem purely_imaginary_complex (a : ℝ) : (a - 2) = 0 → a = 2 :=
by
  intro h
  exact eq_of_sub_eq_zero h

end purely_imaginary_complex_l2023_202362


namespace power_of_i_l2023_202386

theorem power_of_i (i : ℂ) (h₀ : i^2 = -1) : i^(2016) = 1 :=
by {
  -- Proof will go here
  sorry
}

end power_of_i_l2023_202386


namespace fraction_of_square_above_line_l2023_202331

theorem fraction_of_square_above_line :
  let A := (2, 1)
  let B := (5, 1)
  let C := (5, 4)
  let D := (2, 4)
  let P := (2, 3)
  let Q := (5, 1)
  ∃ f : ℚ, f = 2 / 3 := 
by
  -- Placeholder for the proof
  sorry

end fraction_of_square_above_line_l2023_202331


namespace avg_diff_noah_liam_l2023_202357

-- Define the daily differences over 14 days
def daily_differences : List ℤ := [5, 0, 15, -5, 10, 10, -10, 5, 5, 10, -5, 15, 0, 5]

-- Define the function to calculate the average difference
def average_daily_difference (daily_diffs : List ℤ) : ℚ :=
  (daily_diffs.sum : ℚ) / daily_diffs.length

-- The proposition we want to prove
theorem avg_diff_noah_liam : average_daily_difference daily_differences = 60 / 14 := by
  sorry

end avg_diff_noah_liam_l2023_202357


namespace find_a_l2023_202333

theorem find_a (a : ℝ) (x y : ℝ) :
  (x^2 - 4*x + y^2 = 0) →
  ((x - a)^2 + y^2 = 4*((x - 1)^2 + y^2)) →
  a = -2 :=
by
  intros h_circle h_distance
  sorry

end find_a_l2023_202333


namespace smallest_n_l2023_202322

theorem smallest_n (n : ℕ) (hn1 : ∃ k, 5 * n = k^4) (hn2: ∃ m, 4 * n = m^3) : n = 2000 :=
sorry

end smallest_n_l2023_202322


namespace unique_b_for_unique_solution_l2023_202370

theorem unique_b_for_unique_solution (c : ℝ) (h₁ : c ≠ 0) :
  (∃ b : ℝ, b > 0 ∧ ∃! x : ℝ, x^2 + (b + (2 / b)) * x + c = 0) →
  c = 2 :=
by
  -- sorry will go here to indicate the proof is to be filled in
  sorry

end unique_b_for_unique_solution_l2023_202370


namespace min_value_x2_y2_z2_l2023_202345

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^2 + y^2 + z^2 - 3 * x * y * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 := 
sorry

end min_value_x2_y2_z2_l2023_202345


namespace average_of_11_numbers_l2023_202350

theorem average_of_11_numbers (a b c d e f g h i j k : ℝ)
  (h_first_6_avg : (a + b + c + d + e + f) / 6 = 98)
  (h_last_6_avg : (f + g + h + i + j + k) / 6 = 65)
  (h_6th_number : f = 318) :
  ((a + b + c + d + e + f + g + h + i + j + k) / 11) = 60 :=
by
  sorry

end average_of_11_numbers_l2023_202350


namespace last_person_is_knight_l2023_202361

def KnightLiarsGame1 (n : ℕ) : Prop :=
  let m := 10
  let p := 13
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

def KnightLiarsGame2 (n : ℕ) : Prop :=
  let m := 12
  let p := 9
  (∀ i < n, (i % 2 = 0 → true) ∧ (i % 2 = 1 → true)) ∧ 
  (m % 2 ≠ p % 2 → (n - 1) % 2 = 1)

theorem last_person_is_knight :
  ∃ n, KnightLiarsGame1 n ∧ KnightLiarsGame2 n :=
by 
  sorry

end last_person_is_knight_l2023_202361


namespace smallest_triangle_perimeter_l2023_202372

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def smallest_possible_prime_perimeter : ℕ :=
  31

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  a > 5 ∧ b > 5 ∧ c > 5 ∧
                  is_prime a ∧ is_prime b ∧ is_prime c ∧
                  triangle_inequality a b c ∧
                  is_prime (a + b + c) ∧
                  a + b + c = smallest_possible_prime_perimeter :=
sorry

end smallest_triangle_perimeter_l2023_202372


namespace fundraiser_goal_eq_750_l2023_202395

def bronze_donations := 10 * 25
def silver_donations := 7 * 50
def gold_donations   := 1 * 100
def total_collected  := bronze_donations + silver_donations + gold_donations
def amount_needed    := 50
def total_goal       := total_collected + amount_needed

theorem fundraiser_goal_eq_750 : total_goal = 750 :=
by
  sorry

end fundraiser_goal_eq_750_l2023_202395


namespace solve_equation_l2023_202346

theorem solve_equation (x : ℝ) : x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 := 
by sorry

end solve_equation_l2023_202346


namespace sum_of_absolute_values_l2023_202354

theorem sum_of_absolute_values (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 4 * n + 2) →
  a 1 = -1 →
  (∀ n, 1 < n → a n = 2 * n - 5) →
  ((abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) +
    abs (a 6) + abs (a 7) + abs (a 8) + abs (a 9) + abs (a 10)) = 66) :=
by
  intros hS a1_eq ha_eq
  sorry

end sum_of_absolute_values_l2023_202354


namespace part_a_l2023_202303

def system_of_equations (x y z a : ℝ) := 
  (x - a * y = y * z) ∧ (y - a * z = z * x) ∧ (z - a * x = x * y)

theorem part_a (x y z : ℝ) : 
  system_of_equations x y z 0 ↔ (x = 0 ∧ y = 0 ∧ z = 0) 
  ∨ (∃ x, y = x ∧ z = 1) 
  ∨ (∃ x, y = -x ∧ z = -1) := 
  sorry

end part_a_l2023_202303


namespace value_of_y_when_x_is_neg2_l2023_202332

theorem value_of_y_when_x_is_neg2 :
  ∃ (k b : ℝ), (k + b = 2) ∧ (-k + b = -4) ∧ (∀ x, y = k * x + b) ∧ (x = -2) → (y = -7) := 
sorry

end value_of_y_when_x_is_neg2_l2023_202332


namespace kyle_and_miles_marbles_l2023_202343

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end kyle_and_miles_marbles_l2023_202343


namespace probability_four_or_more_same_value_l2023_202323

theorem probability_four_or_more_same_value :
  let n := 5 -- number of dice
  let d := 10 -- number of sides on each die
  let event := "at least four of the five dice show the same value"
  let probability := (23 : ℚ) / 5000 -- given probability
  n = 5 ∧ d = 10 ∧ event = "at least four of the five dice show the same value" →
  (probability = 23 / 5000) := 
by
  intros
  sorry

end probability_four_or_more_same_value_l2023_202323


namespace typists_retype_time_l2023_202355

theorem typists_retype_time
  (x y : ℕ)
  (h1 : (x / 2) + (y / 2) = 25)
  (h2 : 1 / x + 1 / y = 1 / 12) :
  (x = 20 ∧ y = 30) ∨ (x = 30 ∧ y = 20) :=
by
  sorry

end typists_retype_time_l2023_202355


namespace both_miss_probability_l2023_202368

-- Define the probabilities of hitting the target for Persons A and B 
def prob_hit_A : ℝ := 0.85
def prob_hit_B : ℝ := 0.8

-- Calculate the probabilities of missing the target
def prob_miss_A : ℝ := 1 - prob_hit_A
def prob_miss_B : ℝ := 1 - prob_hit_B

-- Prove that the probability of both missing the target is 0.03
theorem both_miss_probability : prob_miss_A * prob_miss_B = 0.03 :=
by
  sorry

end both_miss_probability_l2023_202368


namespace am_gm_inequality_l2023_202321

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : (1 + x) * (1 + y) * (1 + z) ≥ 8 :=
sorry

end am_gm_inequality_l2023_202321


namespace cistern_wet_surface_area_l2023_202365

def cistern_length : ℝ := 4
def cistern_width : ℝ := 8
def water_depth : ℝ := 1.25

def area_bottom (l w : ℝ) : ℝ := l * w
def area_pair1 (l h : ℝ) : ℝ := 2 * (l * h)
def area_pair2 (w h : ℝ) : ℝ := 2 * (w * h)
def total_wet_surface_area (l w h : ℝ) : ℝ := area_bottom l w + area_pair1 l h + area_pair2 w h

theorem cistern_wet_surface_area : total_wet_surface_area cistern_length cistern_width water_depth = 62 := 
by 
  sorry

end cistern_wet_surface_area_l2023_202365


namespace rectangle_area_l2023_202330

theorem rectangle_area (x : ℝ) (h1 : (x^2 + (3*x)^2) = (15*Real.sqrt 2)^2) :
  (x * (3 * x)) = 135 := 
by
  sorry

end rectangle_area_l2023_202330


namespace boys_cannot_score_twice_l2023_202337

-- Define the total number of points in the tournament
def total_points_in_tournament : ℕ := 15

-- Define the number of boys and girls
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 4

-- Define the points scored by boys and girls
axiom points_by_boys : ℕ
axiom points_by_girls : ℕ

-- The conditions
axiom total_points_condition : points_by_boys + points_by_girls = total_points_in_tournament
axiom boys_twice_girls_condition : points_by_boys = 2 * points_by_girls

-- The statement to prove
theorem boys_cannot_score_twice : False :=
  by {
    -- Note: provide a sketch to illustrate that under the given conditions the statement is false
    sorry
  }

end boys_cannot_score_twice_l2023_202337


namespace May4th_Sunday_l2023_202398

theorem May4th_Sunday (x : ℕ) (h_sum : x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 80) : 
  (4 % 7) = 0 :=
by
  sorry

end May4th_Sunday_l2023_202398


namespace sum_x1_x2_range_l2023_202341

variable {x₁ x₂ : ℝ}

-- Definition of x₁ being the real root of the equation x * 2^x = 1
def is_root_1 (x : ℝ) : Prop :=
  x * 2^x = 1

-- Definition of x₂ being the real root of the equation x * log_2 x = 1
def is_root_2 (x : ℝ) : Prop :=
  x * Real.log x / Real.log 2 = 1

theorem sum_x1_x2_range (hx₁ : is_root_1 x₁) (hx₂ : is_root_2 x₂) :
  2 < x₁ + x₂ :=
sorry

end sum_x1_x2_range_l2023_202341


namespace prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l2023_202335

-- Definitions
def fair_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Question 1: Probability that a + b >= 9
theorem prob_sum_geq_9 (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  a + b ≥ 9 → (∃ (valid_outcomes : Finset (ℕ × ℕ)),
    valid_outcomes = {(3, 6), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 3), (6, 4), (6, 5), (6, 6)} ∧
    valid_outcomes.card = 10 ∧
    10 / 36 = 5 / 18) :=
sorry

-- Question 2: Probability that the line ax + by + 5 = 0 is tangent to the circle x^2 + y^2 = 1
theorem prob_tangent_line (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (tangent_outcomes : Finset (ℕ × ℕ)),
    tangent_outcomes = {(3, 4), (4, 3)} ∧
    a^2 + b^2 = 25 ∧
    tangent_outcomes.card = 2 ∧
    2 / 36 = 1 / 18) :=
sorry

-- Question 3: Probability that the lengths a, b, and 5 form an isosceles triangle
theorem prob_isosceles_triangle (a b : ℕ) (ha : a ∈ fair_die) (hb : b ∈ fair_die) :
  (∃ (isosceles_outcomes : Finset (ℕ × ℕ)),
    isosceles_outcomes = {(1, 5), (2, 5), (3, 3), (3, 5), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6)} ∧
    isosceles_outcomes.card = 14 ∧
    14 / 36 = 7 / 18) :=
sorry

end prob_sum_geq_9_prob_tangent_line_prob_isosceles_triangle_l2023_202335


namespace normal_pumping_rate_l2023_202313

-- Define the conditions and the proof problem
def pond_capacity : ℕ := 200
def drought_factor : ℚ := 2/3
def fill_time : ℕ := 50

theorem normal_pumping_rate (R : ℚ) :
  (drought_factor * R) * (fill_time : ℚ) = pond_capacity → R = 6 :=
by
  sorry

end normal_pumping_rate_l2023_202313


namespace right_triangle_area_l2023_202389

theorem right_triangle_area (h b : ℝ) (hypotenuse : h = 5) (base : b = 3) :
  ∃ a : ℝ, a = 1 / 2 * b * (Real.sqrt (h^2 - b^2)) ∧ a = 6 := 
by
  sorry

end right_triangle_area_l2023_202389


namespace sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l2023_202390

theorem sum_of_two_terms_is_term_iff_a_is_multiple_of_d
    (a d : ℤ) 
    (n k : ℕ) 
    (h : ∀ (p : ℕ), a + d * n + (a + d * k) = a + d * p)
    : ∃ m : ℤ, a = d * m :=
sorry

end sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l2023_202390


namespace tangential_quadrilateral_difference_l2023_202348

-- Definitions of the conditions given in the problem
def is_cyclic_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the quadrilateral vertices lie on a circle
def is_tangential_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the sides are tangent to a common incircle
def point_tangency (a b c : ℝ) : Prop := sorry

-- Main theorem
theorem tangential_quadrilateral_difference (AB BC CD DA : ℝ) (x y : ℝ) 
  (h1 : is_cyclic_quadrilateral AB BC CD DA)
  (h2 : is_tangential_quadrilateral AB BC CD DA)
  (h3 : AB = 80) (h4 : BC = 140) (h5 : CD = 120) (h6 : DA = 100)
  (h7 : point_tangency x y CD)
  (h8 : x + y = 120) :
  |x - y| = 80 := 
sorry

end tangential_quadrilateral_difference_l2023_202348


namespace div_coeff_roots_l2023_202327

theorem div_coeff_roots :
  ∀ (a b c d e : ℝ), (∀ x, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4)
  → (d / e = -25 / 12) :=
by
  intros a b c d e h
  sorry

end div_coeff_roots_l2023_202327


namespace max_units_of_material_A_l2023_202363

theorem max_units_of_material_A (x y z : ℕ) 
    (h1 : 3 * x + 5 * y + 7 * z = 62)
    (h2 : 2 * x + 4 * y + 6 * z = 50) : x ≤ 5 :=
by
    sorry 

end max_units_of_material_A_l2023_202363


namespace edge_length_of_small_cube_l2023_202342

-- Define the parameters
def volume_cube : ℕ := 1000
def num_small_cubes : ℕ := 8
def remaining_volume : ℕ := 488

-- Define the main theorem
theorem edge_length_of_small_cube (x : ℕ) :
  (volume_cube - num_small_cubes * x^3 = remaining_volume) → x = 4 := 
by 
  sorry

end edge_length_of_small_cube_l2023_202342


namespace find_speed_l2023_202344

variables (x : ℝ) (V : ℝ)

def initial_speed (x : ℝ) (V : ℝ) : Prop := 
  let time_initial := x / V
  let time_second := (2 * x) / 20
  let total_distance := 3 * x
  let average_speed := 26.25
  average_speed = total_distance / (time_initial + time_second)

theorem find_speed (x : ℝ) (h : initial_speed x V) : V = 70 :=
by sorry

end find_speed_l2023_202344


namespace projectile_max_height_l2023_202339

def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

theorem projectile_max_height : ∃ t : ℝ, h t = 60 := 
sorry

end projectile_max_height_l2023_202339


namespace solve_system_l2023_202309

theorem solve_system : 
  ∃ x y : ℚ, (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ∧ x = 1/2 ∧ y = -3 :=
by
  sorry

end solve_system_l2023_202309


namespace minimize_x_plus_y_on_circle_l2023_202399

theorem minimize_x_plus_y_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : x + y ≥ 2 :=
by
  sorry

end minimize_x_plus_y_on_circle_l2023_202399


namespace math_problem_l2023_202312

/-- Lean translation of the mathematical problem.
Given \(a, b \in \mathbb{R}\) such that \(a^2 + b^2 = a^2 b^2\) and 
\( |a| \neq 1 \) and \( |b| \neq 1 \), prove that 
\[
\frac{a^7}{(1 - a)^2} - \frac{a^7}{(1 + a)^2} = 
\frac{b^7}{(1 - b)^2} - \frac{b^7}{(1 + b)^2}.
\]
-/
theorem math_problem 
  (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  (a^7 / (1 - a)^2 - a^7 / (1 + a)^2) = 
  (b^7 / (1 - b)^2 - b^7 / (1 + b)^2) := 
by 
  -- Proof is omitted for this exercise.
  sorry

end math_problem_l2023_202312


namespace connect_5_points_four_segments_l2023_202358

theorem connect_5_points_four_segments (A B C D E : Type) (h : ∀ (P Q R : Type), P ≠ Q ∧ Q ≠ R ∧ R ≠ P)
: ∃ (n : ℕ), n = 135 := 
  sorry

end connect_5_points_four_segments_l2023_202358


namespace inequalities_hold_l2023_202352

theorem inequalities_hold (a b c x y z : ℝ) (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (x * y * z ≤ a * b * c) :=
by
  sorry

end inequalities_hold_l2023_202352


namespace two_lines_perpendicular_to_same_line_are_parallel_l2023_202382

/- Define what it means for two lines to be perpendicular -/
def perpendicular (l m : Line) : Prop :=
  -- A placeholder definition for perpendicularity, replace with the actual definition
  sorry

/- Define what it means for two lines to be parallel -/
def parallel (l m : Line) : Prop :=
  -- A placeholder definition for parallelism, replace with the actual definition
  sorry

/- Given: Two lines l1 and l2 that are perpendicular to the same line l3 -/
variables (l1 l2 l3 : Line)
variable (h1 : perpendicular l1 l3)
variable (h2 : perpendicular l2 l3)

/- Prove: l1 and l2 are parallel to each other -/
theorem two_lines_perpendicular_to_same_line_are_parallel :
  parallel l1 l2 :=
  sorry

end two_lines_perpendicular_to_same_line_are_parallel_l2023_202382


namespace fans_received_all_items_l2023_202385

theorem fans_received_all_items (n : ℕ) (h1 : (∀ k : ℕ, k * 45 ≤ n → (k * 45) ∣ n))
                                (h2 : (∀ k : ℕ, k * 50 ≤ n → (k * 50) ∣ n))
                                (h3 : (∀ k : ℕ, k * 100 ≤ n → (k * 100) ∣ n))
                                (capacity_full : n = 5000) :
  n / Nat.lcm 45 (Nat.lcm 50 100) = 5 :=
by
  sorry

end fans_received_all_items_l2023_202385


namespace max_minute_hands_l2023_202388

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end max_minute_hands_l2023_202388


namespace range_of_k_l2023_202329

theorem range_of_k (k : ℝ) : (∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 - 2*k*x + k)) ↔ (k ∈ Set.Iic 0 ∨ k ∈ Set.Ici 1) :=
by
  sorry

end range_of_k_l2023_202329


namespace f_prime_at_zero_l2023_202301

-- Lean definition of the conditions.
def a (n : ℕ) : ℝ := 2 * (2 ^ (1/7)) ^ (n - 1)

-- The function f(x) based on the given conditions.
noncomputable def f (x : ℝ) : ℝ := 
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * 
  (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

-- The main goal to prove: f'(0) = 2^12
theorem f_prime_at_zero : deriv f 0 = 2^12 := by
  sorry

end f_prime_at_zero_l2023_202301


namespace total_points_zach_ben_l2023_202364

theorem total_points_zach_ben (zach_points ben_points : ℝ) (h1 : zach_points = 42.0) (h2 : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
by
  sorry

end total_points_zach_ben_l2023_202364


namespace twentyfive_percent_in_usd_l2023_202314

variable (X : ℝ)
variable (Y : ℝ) (hY : Y > 0)

theorem twentyfive_percent_in_usd : 0.25 * X * Y = (0.25 : ℝ) * X * Y := by
  sorry

end twentyfive_percent_in_usd_l2023_202314


namespace ratio_john_to_total_cost_l2023_202376

noncomputable def cost_first_8_years := 8 * 10000
noncomputable def cost_next_10_years := 10 * 20000
noncomputable def university_tuition := 250000
noncomputable def cost_john_paid := 265000
noncomputable def total_cost := cost_first_8_years + cost_next_10_years + university_tuition

theorem ratio_john_to_total_cost : (cost_john_paid / total_cost : ℚ) = 1 / 2 := by
  sorry

end ratio_john_to_total_cost_l2023_202376


namespace lecture_room_configuration_l2023_202305

theorem lecture_room_configuration (m n : ℕ) (boys_per_row girls_per_column unoccupied_chairs : ℕ) :
    boys_per_row = 6 →
    girls_per_column = 8 →
    unoccupied_chairs = 15 →
    (m * n = boys_per_row * m + girls_per_column * n + unoccupied_chairs) →
    (m = 71 ∧ n = 7) ∨
    (m = 29 ∧ n = 9) ∨
    (m = 17 ∧ n = 13) ∨
    (m = 15 ∧ n = 15) ∨
    (m = 11 ∧ n = 27) ∨
    (m = 9 ∧ n = 69) :=
by
  intros h1 h2 h3 h4
  sorry

end lecture_room_configuration_l2023_202305


namespace distance_AF_l2023_202356

theorem distance_AF (A B C D E F : ℝ×ℝ)
  (h1 : A = (0, 0))
  (h2 : B = (5, 0))
  (h3 : C = (5, 5))
  (h4 : D = (0, 5))
  (h5 : E = (2.5, 5))
  (h6 : ∃ k : ℝ, F = (k, 2 * k) ∧ dist F C = 5) :
  dist A F = Real.sqrt 5 :=
by
  sorry

end distance_AF_l2023_202356


namespace find_constants_l2023_202387

noncomputable section

theorem find_constants (P Q R : ℝ)
  (h : ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
    (5*x^2 + 7*x) / ((x - 2) * (x - 4)^2) =
    P / (x - 2) + Q / (x - 4) + R / (x - 4)^2) :
  P = 3.5 ∧ Q = 1.5 ∧ R = 18 :=
by
  sorry

end find_constants_l2023_202387


namespace total_pairs_is_11_l2023_202369

-- Definitions for the conditions
def soft_lens_price : ℕ := 150
def hard_lens_price : ℕ := 85
def total_sales_last_week : ℕ := 1455

-- Variables
variables (H S : ℕ)

-- Condition that she sold 5 more pairs of soft lenses than hard lenses
def sold_more_soft : Prop := S = H + 5

-- Equation for total sales
def total_sales_eq : Prop := (hard_lens_price * H) + (soft_lens_price * S) = total_sales_last_week

-- Total number of pairs of contact lenses sold
def total_pairs_sold : ℕ := H + S

-- The theorem to prove
theorem total_pairs_is_11 (H S : ℕ) (h1 : sold_more_soft H S) (h2 : total_sales_eq H S) : total_pairs_sold H S = 11 :=
sorry

end total_pairs_is_11_l2023_202369


namespace volume_common_solid_hemisphere_cone_l2023_202326

noncomputable def volume_common_solid (r : ℝ) : ℝ := 
  let V_1 := (2/3) * Real.pi * (r^3 - (3 * r / 5)^3)
  let V_2 := Real.pi * ((r / 5)^2) * (r - (r / 15))
  V_1 + V_2

theorem volume_common_solid_hemisphere_cone (r : ℝ) :
  volume_common_solid r = (14 * Real.pi * r^3) / 25 := 
by
  sorry

end volume_common_solid_hemisphere_cone_l2023_202326


namespace max_k_constant_for_right_triangle_l2023_202359

theorem max_k_constant_for_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a ≤ b) (h2 : b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3*Real.sqrt 2) * a * b * c :=
by 
  sorry

end max_k_constant_for_right_triangle_l2023_202359


namespace polynomial_identity_l2023_202310

theorem polynomial_identity :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end polynomial_identity_l2023_202310


namespace intersection_A_B_l2023_202347

def A (y : ℝ) : Prop := ∃ x : ℝ, y = -x^2 + 2*x - 1
def B (y : ℝ) : Prop := ∃ x : ℝ, y = 2*x + 1

theorem intersection_A_B :
  {y : ℝ | A y} ∩ {y : ℝ | B y} = {y : ℝ | y ≤ 0} :=
sorry

end intersection_A_B_l2023_202347


namespace inequality_proof_l2023_202397

theorem inequality_proof (a b c x y z : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) :
  (a^2 * x^2 / ((b * y + c * z) * (b * z + c * y)) + 
   b^2 * y^2 / ((a * x + c * z) * (a * z + c * x)) +
   c^2 * z^2 / ((a * x + b * y) * (a * y + b * x))) ≥ 3 / 4 := 
by
  sorry

end inequality_proof_l2023_202397


namespace emma_correct_percentage_l2023_202375

theorem emma_correct_percentage (t : ℕ) (lt : t > 0)
  (liam_correct_alone : ℝ := 0.70)
  (liam_overall_correct : ℝ := 0.82)
  (emma_correct_alone : ℝ := 0.85)
  (joint_error_rate : ℝ := 0.05)
  (liam_solved_together_correct : ℝ := liam_overall_correct * t - liam_correct_alone * (t / 2)) :
  (emma_correct_alone * (t / 2) + (1 - joint_error_rate) * liam_solved_together_correct) / t * 100 = 87.15 :=
by
  sorry

end emma_correct_percentage_l2023_202375


namespace isosceles_triangle_perimeter_l2023_202381

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) (h₃ : a > b) : a + a + b = 15 :=
by
  sorry

end isosceles_triangle_perimeter_l2023_202381


namespace fixed_point_of_line_l2023_202334

theorem fixed_point_of_line (k : ℝ) : ∃ (p : ℝ × ℝ), p = (-3, 4) ∧ ∀ (x y : ℝ), (y - 4 = -k * (x + 3)) → (-3, 4) = (x, y) :=
by
  sorry

end fixed_point_of_line_l2023_202334


namespace teddy_bears_count_l2023_202373

theorem teddy_bears_count (toys_count : ℕ) (toy_cost : ℕ) (total_money : ℕ) (teddy_bear_cost : ℕ) : 
  toys_count = 28 → 
  toy_cost = 10 → 
  total_money = 580 → 
  teddy_bear_cost = 15 →
  ((total_money - toys_count * toy_cost) / teddy_bear_cost) = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end teddy_bears_count_l2023_202373


namespace original_proposition_false_converse_false_inverse_false_contrapositive_false_l2023_202338

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop := 
  (a * b ≤ 0) → (a ≤ 0 ∨ b ≤ 0)

-- Define the converse
def converse (a b : ℝ) : Prop := 
  (a ≤ 0 ∨ b ≤ 0) → (a * b ≤ 0)

-- Define the inverse
def inverse (a b : ℝ) : Prop := 
  (a * b > 0) → (a > 0 ∧ b > 0)

-- Define the contrapositive
def contrapositive (a b : ℝ) : Prop := 
  (a > 0 ∧ b > 0) → (a * b > 0)

-- Prove that the original proposition is false
theorem original_proposition_false : ∀ (a b : ℝ), ¬ original_proposition a b :=
by sorry

-- Prove that the converse is false
theorem converse_false : ∀ (a b : ℝ), ¬ converse a b :=
by sorry

-- Prove that the inverse is false
theorem inverse_false : ∀ (a b : ℝ), ¬ inverse a b :=
by sorry

-- Prove that the contrapositive is false
theorem contrapositive_false : ∀ (a b : ℝ), ¬ contrapositive a b :=
by sorry

end original_proposition_false_converse_false_inverse_false_contrapositive_false_l2023_202338


namespace problem_1_problem_2_problem_3_problem_4_l2023_202307

-- Problem 1
theorem problem_1 (x y : ℝ) : 
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
by
  sorry

-- Problem 4
theorem problem_4 : 2010^2 - 2011 * 2009 = 1 :=
by
  sorry

end problem_1_problem_2_problem_3_problem_4_l2023_202307


namespace length_of_segment_l2023_202383

theorem length_of_segment (x : ℝ) : 
  |x - (27^(1/3))| = 5 →
  ∃ a b : ℝ, a - b = 10 ∧ (|a - (27^(1/3))| = 5 ∧ |b - (27^(1/3))| = 5) :=
by
  sorry

end length_of_segment_l2023_202383


namespace coloring_ways_l2023_202320

-- Definitions for colors
inductive Color
| red
| green

open Color

-- Definition of the coloring function
def color (n : ℕ) : Color := sorry

-- Conditions:
-- 1. Each positive integer is colored either red or green
def condition1 (n : ℕ) : n > 0 → (color n = red ∨ color n = green) := sorry

-- 2. The sum of any two different red numbers is a red number
def condition2 (r1 r2 : ℕ) : r1 ≠ r2 → color r1 = red → color r2 = red → color (r1 + r2) = red := sorry

-- 3. The sum of any two different green numbers is a green number
def condition3 (g1 g2 : ℕ) : g1 ≠ g2 → color g1 = green → color g2 = green → color (g1 + g2) = green := sorry

-- The required theorem
theorem coloring_ways : ∃! (f : ℕ → Color), 
  (∀ n, n > 0 → (f n = red ∨ f n = green)) ∧ 
  (∀ r1 r2, r1 ≠ r2 → f r1 = red → f r2 = red → f (r1 + r2) = red) ∧
  (∀ g1 g2, g1 ≠ g2 → f g1 = green → f g2 = green → f (g1 + g2) = green) :=
sorry

end coloring_ways_l2023_202320


namespace arithmetic_sequence_product_l2023_202353

theorem arithmetic_sequence_product (b : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ n, b (n + 1) > b n)
  (h2 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l2023_202353


namespace quadrilateral_area_l2023_202377

noncomputable def AreaOfQuadrilateral (AB AC AD : ℝ) : ℝ :=
  let BC := Real.sqrt (AC^2 - AB^2)
  let CD := Real.sqrt (AC^2 - AD^2)
  let AreaABC := (1 / 2) * AB * BC
  let AreaACD := (1 / 2) * AD * CD
  AreaABC + AreaACD

theorem quadrilateral_area :
  AreaOfQuadrilateral 5 13 12 = 60 :=
by
  sorry

end quadrilateral_area_l2023_202377


namespace total_amount_is_200_l2023_202396

-- Given conditions
def sam_amount : ℕ := 75
def billy_amount : ℕ := 2 * sam_amount - 25

-- Theorem to prove
theorem total_amount_is_200 : billy_amount + sam_amount = 200 :=
by
  sorry

end total_amount_is_200_l2023_202396


namespace ratio_A_to_B_l2023_202302

theorem ratio_A_to_B (total_weight_X : ℕ) (weight_B : ℕ) (weight_A : ℕ) (h₁ : total_weight_X = 324) (h₂ : weight_B = 270) (h₃ : weight_A = total_weight_X - weight_B):
  weight_A / gcd weight_A weight_B = 1 ∧ weight_B / gcd weight_A weight_B = 5 :=
by
  sorry

end ratio_A_to_B_l2023_202302


namespace fraction_to_terminanting_decimal_l2023_202380

theorem fraction_to_terminanting_decimal : (47 / (5^4 * 2) : ℚ) = 0.0376 := 
by 
  sorry

end fraction_to_terminanting_decimal_l2023_202380


namespace hyperbola_distance_to_foci_l2023_202391

theorem hyperbola_distance_to_foci
  (E : ∀ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1)
  (F1 F2 : ℝ)
  (P : ℝ)
  (dist_PF1 : P = 5)
  (a : ℝ)
  (ha : a = 3): 
  |P - F2| = 11 :=
by
  sorry

end hyperbola_distance_to_foci_l2023_202391


namespace total_tickets_correct_l2023_202318

-- Define the initial number of tickets Tate has
def initial_tickets_Tate : ℕ := 32

-- Define the additional tickets Tate buys
def additional_tickets_Tate : ℕ := 2

-- Calculate the total number of tickets Tate has
def total_tickets_Tate : ℕ := initial_tickets_Tate + additional_tickets_Tate

-- Define the number of tickets Peyton has (half of Tate's total tickets)
def tickets_Peyton : ℕ := total_tickets_Tate / 2

-- Calculate the total number of tickets Tate and Peyton have together
def total_tickets_together : ℕ := total_tickets_Tate + tickets_Peyton

-- Prove that the total number of tickets together equals 51
theorem total_tickets_correct : total_tickets_together = 51 := by
  sorry

end total_tickets_correct_l2023_202318


namespace geometric_sequence_sum_l2023_202304

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) -- a_n is a sequence of real numbers
  (q : ℝ) -- q is the common ratio
  (h1 : a 1 + a 2 = 20) -- first condition
  (h2 : a 3 + a 4 = 80) -- second condition
  (h_geom : ∀ n, a (n + 1) = a n * q) -- property of geometric sequence
  : a 5 + a 6 = 320 := 
sorry

end geometric_sequence_sum_l2023_202304


namespace days_wages_l2023_202306

theorem days_wages (S W_a W_b : ℝ) 
    (h1 : S = 28 * W_b) 
    (h2 : S = 12 * (W_a + W_b)) 
    (h3 : S = 21 * W_a) : 
    true := 
by sorry

end days_wages_l2023_202306


namespace calc_square_difference_and_square_l2023_202394

theorem calc_square_difference_and_square (a b : ℤ) (h1 : a = 7) (h2 : b = 3)
  (h3 : a^2 = 49) (h4 : b^2 = 9) : (a^2 - b^2)^2 = 1600 := by
  sorry

end calc_square_difference_and_square_l2023_202394


namespace cost_of_four_dozen_l2023_202379

-- Defining the conditions
def cost_of_three_dozen (cost : ℚ) : Prop :=
  cost = 25.20

-- The theorem to prove the cost of four dozen apples at the same rate
theorem cost_of_four_dozen (cost : ℚ) :
  cost_of_three_dozen cost →
  (4 * (cost / 3) = 33.60) :=
by
  sorry

end cost_of_four_dozen_l2023_202379


namespace train_length_and_speed_l2023_202308

theorem train_length_and_speed (L_bridge : ℕ) (t_cross : ℕ) (t_on_bridge : ℕ) (L_train : ℕ) (v_train : ℕ)
  (h_bridge : L_bridge = 1000)
  (h_t_cross : t_cross = 60)
  (h_t_on_bridge : t_on_bridge = 40)
  (h_crossing_eq : (L_bridge + L_train) / t_cross = v_train)
  (h_on_bridge_eq : L_bridge / t_on_bridge = v_train) : 
  L_train = 200 ∧ v_train = 20 := 
  by
  sorry

end train_length_and_speed_l2023_202308


namespace new_books_count_l2023_202317

-- Defining the conditions
def num_adventure_books : ℕ := 13
def num_mystery_books : ℕ := 17
def num_used_books : ℕ := 15

-- Proving the number of new books Sam bought
theorem new_books_count : (num_adventure_books + num_mystery_books) - num_used_books = 15 :=
by
  sorry

end new_books_count_l2023_202317


namespace mark_birth_year_proof_l2023_202325

-- Conditions
def current_year := 2021
def janice_age := 21
def graham_age := 2 * janice_age
def mark_age := graham_age + 3
def mark_birth_year (current_year : ℕ) (mark_age : ℕ) := current_year - mark_age

-- Statement to prove
theorem mark_birth_year_proof : 
  mark_birth_year current_year mark_age = 1976 := by
  sorry

end mark_birth_year_proof_l2023_202325


namespace trip_time_maple_to_oak_l2023_202371

noncomputable def total_trip_time (d1 d2 v1 v2 t_break : ℝ) : ℝ :=
  (d1 / v1) + t_break + (d2 / v2)

theorem trip_time_maple_to_oak : 
  total_trip_time 210 210 50 40 0.5 = 5.75 :=
by
  sorry

end trip_time_maple_to_oak_l2023_202371


namespace math_problem_l2023_202340

theorem math_problem 
  (a : ℤ) 
  (h_a : a = -1) 
  (b : ℚ) 
  (h_b : b = 0) 
  (c : ℕ) 
  (h_c : c = 1)
  : a^2024 + 2023 * b - c^2023 = 0 := by
  sorry

end math_problem_l2023_202340


namespace isosceles_triangle_perimeter_l2023_202351

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h_iso : ¬(4 + 4 > 9 ∧ 4 + 9 > 4 ∧ 9 + 4 > 4))
  (h_ineq : (9 + 9 > 4) ∧ (9 + 4 > 9) ∧ (4 + 9 > 9)) : 2 * b + a = 22 :=
by sorry

end isosceles_triangle_perimeter_l2023_202351


namespace prism_distance_to_plane_l2023_202336

theorem prism_distance_to_plane
  (side_length : ℝ)
  (volume : ℝ)
  (h : ℝ)
  (base_is_square : side_length = 6)
  (volume_formula : volume = (1 / 3) * h * (side_length ^ 2)) :
  h = 8 := 
  by sorry

end prism_distance_to_plane_l2023_202336


namespace uncovered_area_l2023_202367

theorem uncovered_area {s₁ s₂ : ℝ} (hs₁ : s₁ = 10) (hs₂ : s₂ = 4) : 
  (s₁^2 - 2 * s₂^2) = 68 := by
  sorry

end uncovered_area_l2023_202367


namespace doughnut_machine_completion_l2023_202300

noncomputable def completion_time (start_time : ℕ) (partial_duration : ℕ) : ℕ :=
  start_time + 4 * partial_duration

theorem doughnut_machine_completion :
  let start_time := 8 * 60  -- 8:00 AM in minutes
  let partial_completion_time := 11 * 60 + 40  -- 11:40 AM in minutes
  let one_fourth_duration := partial_completion_time - start_time
  completion_time start_time one_fourth_duration = (22 * 60 + 40) := -- 10:40 PM in minutes
by
  sorry

end doughnut_machine_completion_l2023_202300


namespace ab_value_l2023_202349

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end ab_value_l2023_202349


namespace ratio_of_volumes_of_spheres_l2023_202324

theorem ratio_of_volumes_of_spheres (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a / b = 1 / 2 ∧ b / c = 2 / 3) : a^3 / b^3 = 1 / 8 ∧ b^3 / c^3 = 8 / 27 :=
by
  sorry

end ratio_of_volumes_of_spheres_l2023_202324
