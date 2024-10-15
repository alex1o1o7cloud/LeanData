import Mathlib

namespace NUMINAMATH_GPT_joan_kittens_total_l723_72368

-- Definition of the initial conditions
def joan_original_kittens : ℕ := 8
def neighbor_original_kittens : ℕ := 6
def joan_gave_away : ℕ := 2
def neighbor_gave_away : ℕ := 4
def joan_adopted_from_neighbor : ℕ := 3

-- The final number of kittens Joan has
def joan_final_kittens : ℕ :=
  let joan_remaining := joan_original_kittens - joan_gave_away
  let neighbor_remaining := neighbor_original_kittens - neighbor_gave_away
  let adopted := min joan_adopted_from_neighbor neighbor_remaining
  joan_remaining + adopted

theorem joan_kittens_total : joan_final_kittens = 8 := 
by 
  -- Lean proof would go here, but adding sorry for now
  sorry

end NUMINAMATH_GPT_joan_kittens_total_l723_72368


namespace NUMINAMATH_GPT_binom_30_3_eq_4060_l723_72358

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_GPT_binom_30_3_eq_4060_l723_72358


namespace NUMINAMATH_GPT_general_term_formula_l723_72326

theorem general_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = 3 * n ^ 2 - 2 * n) → 
  (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ a 1 = S 1 → 
  ∀ n, a n = 6 * n - 5 := 
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l723_72326


namespace NUMINAMATH_GPT_compound_ratio_is_one_fourteenth_l723_72320

theorem compound_ratio_is_one_fourteenth :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) = 1 / 14 :=
by sorry

end NUMINAMATH_GPT_compound_ratio_is_one_fourteenth_l723_72320


namespace NUMINAMATH_GPT_range_of_a_l723_72309

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := 
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (1 - a) + f (1 - 2 * a) < 0

-- Theorem statement
theorem range_of_a (h_decreasing : decreasing_on f (Set.Ioo (-1) 1))
                   (h_odd : odd_function f)
                   (h_condition : condition f a) :
  0 < a ∧ a < 2 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l723_72309


namespace NUMINAMATH_GPT_min_value_exists_max_value_exists_l723_72335

noncomputable def y (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem min_value_exists :
  (∃ k : ℤ, y (π / 6 + 2 * k * π) = -2) ∧ (∃ k : ℤ, y (5 * π / 6 + 2 * k * π) = -2) :=
by 
  sorry

theorem max_value_exists :
  ∃ k : ℤ, y (-π / 2 + 2 * k * π) = 7 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_exists_max_value_exists_l723_72335


namespace NUMINAMATH_GPT_correct_range_of_x_l723_72360

variable {x : ℝ}

noncomputable def isosceles_triangle (x y : ℝ) : Prop :=
  let perimeter := 2 * y + x
  let relationship := y = - (1/2) * x + 8
  perimeter = 16 ∧ relationship

theorem correct_range_of_x (x y : ℝ) (h : isosceles_triangle x y) : 0 < x ∧ x < 8 :=
by
  -- The proof of the theorem is omitted
  sorry

end NUMINAMATH_GPT_correct_range_of_x_l723_72360


namespace NUMINAMATH_GPT_wheel_sum_even_and_greater_than_10_l723_72353

-- Definitions based on conditions
def prob_even_A : ℚ := 3 / 8
def prob_odd_A : ℚ := 5 / 8
def prob_even_B : ℚ := 1 / 4
def prob_odd_B : ℚ := 3 / 4

-- Event probabilities from solution steps
def prob_both_even : ℚ := prob_even_A * prob_even_B
def prob_both_odd : ℚ := prob_odd_A * prob_odd_B
def prob_even_sum : ℚ := prob_both_even + prob_both_odd
def prob_even_sum_greater_10 : ℚ := 1 / 3

-- Compute final probability
def final_probability : ℚ := prob_even_sum * prob_even_sum_greater_10

-- The statement that needs proving
theorem wheel_sum_even_and_greater_than_10 : final_probability = 3 / 16 := by
  sorry

end NUMINAMATH_GPT_wheel_sum_even_and_greater_than_10_l723_72353


namespace NUMINAMATH_GPT_common_difference_minimum_sum_value_l723_72355

variable {α : Type}
variables (a : ℕ → ℤ) (d : ℤ)
variables (S : ℕ → ℚ)

-- Conditions: Arithmetic sequence property and specific initial values
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

axiom a1_eq_neg3 : a 1 = -3
axiom condition : 11 * a 5 = 5 * a 8 - 13

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (d : ℤ) : ℚ :=
  (↑n / 2) * (2 * a 1 + ↑((n - 1) * d))

-- Prove the common difference and the minimum sum value
theorem common_difference : d = 31 / 9 :=
sorry

theorem minimum_sum_value : S 1 = -2401 / 840 :=
sorry

end NUMINAMATH_GPT_common_difference_minimum_sum_value_l723_72355


namespace NUMINAMATH_GPT_quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l723_72390

theorem quad_eq1_solution (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  sorry

theorem quad_eq2_solution (x : ℝ) : 2 * x^2 - 7 * x + 5 = 0 → x = 5 / 2 ∨ x = 1 :=
by
  sorry

theorem quad_eq3_solution (x : ℝ) : (x + 3)^2 - 2 * (x + 3) = 0 → x = -3 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_quad_eq1_solution_quad_eq2_solution_quad_eq3_solution_l723_72390


namespace NUMINAMATH_GPT_asymptotic_lines_of_hyperbola_l723_72373

open Real

-- Given: Hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- To Prove: Asymptotic lines equation
theorem asymptotic_lines_of_hyperbola : 
  ∀ x y : ℝ, hyperbola x y → (y = x ∨ y = -x) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_asymptotic_lines_of_hyperbola_l723_72373


namespace NUMINAMATH_GPT_greatest_power_of_two_factor_l723_72349

theorem greatest_power_of_two_factor (a b c d : ℕ) (h1 : a = 10) (h2 : b = 1006) (h3 : c = 6) (h4 : d = 503) :
  ∃ k : ℕ, 2^k ∣ (a^b - c^d) ∧ ∀ j : ℕ, 2^j ∣ (a^b - c^d) → j ≤ 503 :=
sorry

end NUMINAMATH_GPT_greatest_power_of_two_factor_l723_72349


namespace NUMINAMATH_GPT_difference_between_scores_l723_72318

variable (H F : ℕ)
variable (h_hajar_score : H = 24)
variable (h_sum_scores : H + F = 69)
variable (h_farah_higher : F > H)

theorem difference_between_scores : F - H = 21 := by
  sorry

end NUMINAMATH_GPT_difference_between_scores_l723_72318


namespace NUMINAMATH_GPT_elena_deductions_in_cents_l723_72366

-- Definitions based on the conditions
def cents_per_dollar : ℕ := 100
def hourly_wage_in_dollars : ℕ := 25
def hourly_wage_in_cents : ℕ := hourly_wage_in_dollars * cents_per_dollar
def tax_rate : ℚ := 0.02
def health_benefit_rate : ℚ := 0.015

-- The problem to prove
theorem elena_deductions_in_cents:
  (tax_rate * hourly_wage_in_cents) + (health_benefit_rate * hourly_wage_in_cents) = 87.5 := 
by
  sorry

end NUMINAMATH_GPT_elena_deductions_in_cents_l723_72366


namespace NUMINAMATH_GPT_second_grade_students_sampled_l723_72367

-- Definitions corresponding to conditions in a)
def total_students := 2000
def mountain_climbing_fraction := 2 / 5
def running_ratios := (2, 3, 5)
def sample_size := 200

-- Calculation of total running participants based on ratio
def total_running_students :=
  total_students * (1 - mountain_climbing_fraction)

def a := 2 * (total_running_students / (2 + 3 + 5))
def b := 3 * (total_running_students / (2 + 3 + 5))
def c := 5 * (total_running_students / (2 + 3 + 5))

def running_sample_size := sample_size * (3 / 5) --since the ratio is 3:5

-- The statement to prove
theorem second_grade_students_sampled : running_sample_size * (3 / (2+3+5)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_second_grade_students_sampled_l723_72367


namespace NUMINAMATH_GPT_bald_eagle_dive_time_l723_72394

-- Definitions as per the conditions in the problem
def speed_bald_eagle : ℝ := 100
def speed_peregrine_falcon : ℝ := 2 * speed_bald_eagle
def time_peregrine_falcon : ℝ := 15

-- The theorem to prove
theorem bald_eagle_dive_time : (speed_bald_eagle * 30) = (speed_peregrine_falcon * time_peregrine_falcon) := by
  sorry

end NUMINAMATH_GPT_bald_eagle_dive_time_l723_72394


namespace NUMINAMATH_GPT_cindy_first_to_get_five_l723_72344

def probability_of_five : ℚ := 1 / 6

def anne_turn (p: ℚ) : ℚ := 1 - p
def cindy_turn (p: ℚ) : ℚ := p
def none_get_five (p: ℚ) : ℚ := (1 - p)^3

theorem cindy_first_to_get_five : 
    (∑' n, (anne_turn probability_of_five * none_get_five probability_of_five ^ n) * 
                cindy_turn probability_of_five) = 30 / 91 := by 
    sorry

end NUMINAMATH_GPT_cindy_first_to_get_five_l723_72344


namespace NUMINAMATH_GPT_equation_solutions_equivalence_l723_72325

theorem equation_solutions_equivalence {n k : ℕ} (hn : 1 < n) (hk : 1 < k) (hnk : n > k) :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^n + y^n = z^(n - k)) :=
by
  sorry

end NUMINAMATH_GPT_equation_solutions_equivalence_l723_72325


namespace NUMINAMATH_GPT_length_of_garden_l723_72302

variables (w l : ℕ)

-- Definitions based on the problem conditions
def length_twice_width := l = 2 * w
def perimeter_eq_900 := 2 * l + 2 * w = 900

-- The statement to be proved
theorem length_of_garden (h1 : length_twice_width w l) (h2 : perimeter_eq_900 w l) : l = 300 :=
sorry

end NUMINAMATH_GPT_length_of_garden_l723_72302


namespace NUMINAMATH_GPT_example_calculation_l723_72331

theorem example_calculation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end NUMINAMATH_GPT_example_calculation_l723_72331


namespace NUMINAMATH_GPT_ratio_fifth_terms_l723_72381

-- Define the arithmetic sequences and their sums
variables {a b : ℕ → ℕ}
variables {S T : ℕ → ℕ}

-- Assume conditions of the problem
axiom sum_condition (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom sum_condition2 (n : ℕ) : T n = n * (b 1 + b n) / 2
axiom ratio_condition : ∀ n, S n / T n = (2 * n - 3) / (3 * n - 2)

-- Prove the ratio of fifth terms a_5 / b_5
theorem ratio_fifth_terms : (a 5 : ℚ) / b 5 = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_fifth_terms_l723_72381


namespace NUMINAMATH_GPT_cost_of_pencils_and_pens_l723_72321

theorem cost_of_pencils_and_pens (p q : ℝ) 
  (h₁ : 3 * p + 2 * q = 3.60) 
  (h₂ : 2 * p + 3 * q = 3.15) : 
  3 * p + 3 * q = 4.05 :=
sorry

end NUMINAMATH_GPT_cost_of_pencils_and_pens_l723_72321


namespace NUMINAMATH_GPT_fan_rotation_is_not_translation_l723_72379

def phenomenon := Type

def is_translation (p : phenomenon) : Prop := sorry

axiom elevator_translation : phenomenon
axiom drawer_translation : phenomenon
axiom fan_rotation : phenomenon
axiom car_translation : phenomenon

axiom elevator_is_translation : is_translation elevator_translation
axiom drawer_is_translation : is_translation drawer_translation
axiom car_is_translation : is_translation car_translation

theorem fan_rotation_is_not_translation : ¬ is_translation fan_rotation := sorry

end NUMINAMATH_GPT_fan_rotation_is_not_translation_l723_72379


namespace NUMINAMATH_GPT_evaluate_expression_l723_72303

theorem evaluate_expression (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 2 * x ^ y + 5 * y ^ x - z ^ 2 = 42 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l723_72303


namespace NUMINAMATH_GPT_gcd_polynomial_correct_l723_72323

noncomputable def gcd_polynomial (b : ℤ) := 5 * b^3 + b^2 + 8 * b + 38

theorem gcd_polynomial_correct (b : ℤ) (h : 342 ∣ b) : Int.gcd (gcd_polynomial b) b = 38 := by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_correct_l723_72323


namespace NUMINAMATH_GPT_cone_volume_l723_72350

theorem cone_volume (l : ℝ) (θ : ℝ) (h r V : ℝ)
  (h_l : l = 5)
  (h_θ : θ = (8 * Real.pi) / 5)
  (h_arc_length : 2 * Real.pi * r = l * θ)
  (h_radius: r = 4)
  (h_height : h = Real.sqrt (l^2 - r^2))
  (h_volume_eq : V = (1 / 3) * Real.pi * r^2 * h) :
  V = 16 * Real.pi :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cone_volume_l723_72350


namespace NUMINAMATH_GPT_min_distance_origin_to_line_l723_72372

theorem min_distance_origin_to_line 
  (x y : ℝ) 
  (h : x + y = 4) : 
  ∃ P : ℝ, P = 2 * Real.sqrt 2 ∧ 
    (∀ Q : ℝ, Q = Real.sqrt (x^2 + y^2) → P ≤ Q) :=
by
  sorry

end NUMINAMATH_GPT_min_distance_origin_to_line_l723_72372


namespace NUMINAMATH_GPT_number_of_cows_is_six_l723_72310

variable (C H : Nat) -- C for cows and H for chickens

-- Number of legs is 12 more than twice the number of heads.
def cows_count_condition : Prop :=
  4 * C + 2 * H = 2 * (C + H) + 12

theorem number_of_cows_is_six (h : cows_count_condition C H) : C = 6 :=
sorry

end NUMINAMATH_GPT_number_of_cows_is_six_l723_72310


namespace NUMINAMATH_GPT_soccer_ball_purchase_l723_72364

theorem soccer_ball_purchase (wholesale_price retail_price profit remaining_balls final_profit : ℕ)
  (h1 : wholesale_price = 30)
  (h2 : retail_price = 45)
  (h3 : profit = retail_price - wholesale_price)
  (h4 : remaining_balls = 30)
  (h5 : final_profit = 1500) :
  ∃ (initial_balls : ℕ), (initial_balls - remaining_balls) * profit = final_profit ∧ initial_balls = 130 :=
by
  sorry

end NUMINAMATH_GPT_soccer_ball_purchase_l723_72364


namespace NUMINAMATH_GPT_triangle_area_l723_72371

open Real

def line1 (x y : ℝ) : Prop := y = 6
def line2 (x y : ℝ) : Prop := y = 2 + x
def line3 (x y : ℝ) : Prop := y = 2 - x

def is_vertex (x y : ℝ) (l1 l2 : ℝ → ℝ → Prop) : Prop := l1 x y ∧ l2 x y

def vertices (v1 v2 v3 : ℝ × ℝ) : Prop :=
  is_vertex v1.1 v1.2 line1 line2 ∧
  is_vertex v2.1 v2.2 line1 line3 ∧
  is_vertex v3.1 v3.2 line2 line3

def area_triangle (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2) -
             (v2.1 * v1.2 + v3.1 * v2.2 + v1.1 * v3.2))

theorem triangle_area : vertices (4, 6) (-4, 6) (0, 2) → area_triangle (4, 6) (-4, 6) (0, 2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l723_72371


namespace NUMINAMATH_GPT_megan_average_speed_l723_72383

theorem megan_average_speed :
  ∃ s : ℕ, s = 100 / 3 ∧ ∃ (o₁ o₂ : ℕ), o₁ = 27472 ∧ o₂ = 27572 ∧ o₂ - o₁ = 100 :=
by
  sorry

end NUMINAMATH_GPT_megan_average_speed_l723_72383


namespace NUMINAMATH_GPT_people_at_first_table_l723_72333

theorem people_at_first_table (N x : ℕ) 
  (h1 : 20 < N) 
  (h2 : N < 50)
  (h3 : (N - x) % 42 = 0)
  (h4 : N % 8 = 7) : 
  x = 5 :=
sorry

end NUMINAMATH_GPT_people_at_first_table_l723_72333


namespace NUMINAMATH_GPT_simplify_fraction_l723_72314

theorem simplify_fraction:
  ((1/2 - 1/3) / (3/7 + 1/9)) * (1/4) = 21/272 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l723_72314


namespace NUMINAMATH_GPT_votes_for_eliot_l723_72330

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end NUMINAMATH_GPT_votes_for_eliot_l723_72330


namespace NUMINAMATH_GPT_order_of_magnitudes_l723_72312

variable (x : ℝ)
variable (a : ℝ)

theorem order_of_magnitudes (h1 : x < 0) (h2 : a = 2 * x) : x^2 < a * x ∧ a * x < a^2 := 
by
  sorry

end NUMINAMATH_GPT_order_of_magnitudes_l723_72312


namespace NUMINAMATH_GPT_tetrahedron_cube_volume_ratio_l723_72348

theorem tetrahedron_cube_volume_ratio (s : ℝ) (h_s : s > 0):
    let V_cube := s ^ 3
    let a := s * Real.sqrt 3
    let V_tetrahedron := (Real.sqrt 2 / 12) * a ^ 3
    (V_tetrahedron / V_cube) = (Real.sqrt 6 / 4) := by
    sorry

end NUMINAMATH_GPT_tetrahedron_cube_volume_ratio_l723_72348


namespace NUMINAMATH_GPT_outfits_count_l723_72363

theorem outfits_count (shirts ties : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 6) : 
  (shirts * (ties + 1) = 49) :=
by
  sorry

end NUMINAMATH_GPT_outfits_count_l723_72363


namespace NUMINAMATH_GPT_allowable_rectangular_formations_count_l723_72352

theorem allowable_rectangular_formations_count (s t f : ℕ) 
  (h1 : s * t = 240)
  (h2 : Nat.Prime s)
  (h3 : 8 ≤ t ∧ t ≤ 30)
  (h4 : f ≤ 8)
  : f = 0 :=
sorry

end NUMINAMATH_GPT_allowable_rectangular_formations_count_l723_72352


namespace NUMINAMATH_GPT_rose_paid_after_discount_l723_72362

-- Define the conditions as given in the problem statement
def original_price : ℕ := 10
def discount_rate : ℕ := 10

-- Define the theorem that needs to be proved
theorem rose_paid_after_discount : 
  original_price - (original_price * discount_rate / 100) = 9 :=
by
  -- Here we skip the proof with sorry
  sorry

end NUMINAMATH_GPT_rose_paid_after_discount_l723_72362


namespace NUMINAMATH_GPT_find_width_of_lot_l723_72347

noncomputable def volume_of_rectangular_prism (l w h : ℝ) : ℝ := l * w * h

theorem find_width_of_lot
  (l h v : ℝ)
  (h_len : l = 40)
  (h_height : h = 2)
  (h_volume : v = 1600)
  : ∃ w : ℝ, volume_of_rectangular_prism l w h = v ∧ w = 20 := by
  use 20
  simp [volume_of_rectangular_prism, h_len, h_height, h_volume]
  sorry

end NUMINAMATH_GPT_find_width_of_lot_l723_72347


namespace NUMINAMATH_GPT_latest_time_temperature_84_l723_72305

noncomputable def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem latest_time_temperature_84 :
  ∃ t_max : ℝ, temperature t_max = 84 ∧ ∀ t : ℝ, temperature t = 84 → t ≤ t_max ∧ t_max = 11 :=
by
  sorry

end NUMINAMATH_GPT_latest_time_temperature_84_l723_72305


namespace NUMINAMATH_GPT_final_toy_count_correct_l723_72337

def initial_toy_count : ℝ := 5.3
def tuesday_toys_left (initial: ℝ) : ℝ := initial * 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_toys_left (tuesday_total: ℝ) : ℝ := tuesday_total * 0.498
def wednesday_new_toys : ℝ := 2.4
def thursday_toys_left (wednesday_total: ℝ) : ℝ := wednesday_total * 0.692
def thursday_new_toys : ℝ := 4.5

def total_toys (initial: ℝ) : ℝ :=
  let after_tuesday := tuesday_toys_left initial + tuesday_new_toys
  let after_wednesday := wednesday_toys_left after_tuesday + wednesday_new_toys
  let after_thursday := thursday_toys_left after_wednesday + thursday_new_toys
  after_thursday

def toys_lost_tuesday (initial: ℝ) (left: ℝ) : ℝ := initial - left
def toys_lost_wednesday (tuesday_total: ℝ) (left: ℝ) : ℝ := tuesday_total - left
def toys_lost_thursday (wednesday_total: ℝ) (left: ℝ) : ℝ := wednesday_total - left
def total_lost_toys (initial: ℝ) : ℝ :=
  let tuesday_left := tuesday_toys_left initial
  let tuesday_total := tuesday_left + tuesday_new_toys
  let wednesday_left := wednesday_toys_left tuesday_total
  let wednesday_total := wednesday_left + wednesday_new_toys
  let thursday_left := thursday_toys_left wednesday_total
  let lost_tuesday := toys_lost_tuesday initial tuesday_left
  let lost_wednesday := toys_lost_wednesday tuesday_total wednesday_left
  let lost_thursday := toys_lost_thursday wednesday_total thursday_left
  lost_tuesday + lost_wednesday + lost_thursday

def final_toy_count (initial: ℝ) : ℝ :=
  let current_toys := total_toys initial
  let lost_toys := total_lost_toys initial
  current_toys + lost_toys

theorem final_toy_count_correct :
  final_toy_count initial_toy_count = 15.8 := sorry

end NUMINAMATH_GPT_final_toy_count_correct_l723_72337


namespace NUMINAMATH_GPT_find_angle_A_l723_72327

theorem find_angle_A (a b c A B C : ℝ)
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) :
  A = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l723_72327


namespace NUMINAMATH_GPT_weight_of_dry_grapes_l723_72380

def fresh_grapes : ℝ := 10 -- weight of fresh grapes in kg
def fresh_water_content : ℝ := 0.90 -- fresh grapes contain 90% water by weight
def dried_water_content : ℝ := 0.20 -- dried grapes contain 20% water by weight

theorem weight_of_dry_grapes : 
  (fresh_grapes * (1 - fresh_water_content)) / (1 - dried_water_content) = 1.25 := 
by 
  sorry

end NUMINAMATH_GPT_weight_of_dry_grapes_l723_72380


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l723_72376

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > |b|) → (a^3 > b^3) ∧ ¬((a^3 > b^3) → (a > |b|)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l723_72376


namespace NUMINAMATH_GPT_probability_math_majors_consecutive_l723_72301

theorem probability_math_majors_consecutive :
  (5 / 12) * (4 / 11) * (3 / 10) * (2 / 9) * (1 / 8) * 12 = 1 / 66 :=
by
  sorry

end NUMINAMATH_GPT_probability_math_majors_consecutive_l723_72301


namespace NUMINAMATH_GPT_workman_problem_l723_72338

theorem workman_problem (A B : ℝ) (h1 : A = B / 2) (h2 : (A + B) * 10 = 1) : B = 1 / 15 := by
  sorry

end NUMINAMATH_GPT_workman_problem_l723_72338


namespace NUMINAMATH_GPT_geometric_sequence_product_l723_72387

theorem geometric_sequence_product (b : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, b (n+1) = b n * r)
  (h_b9 : b 9 = (3 + 5) / 2) : b 1 * b 17 = 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l723_72387


namespace NUMINAMATH_GPT_rackets_packed_l723_72313

theorem rackets_packed (total_cartons : ℕ) (cartons_3 : ℕ) (cartons_2 : ℕ) 
  (h1 : total_cartons = 38) 
  (h2 : cartons_3 = 24) 
  (h3 : cartons_2 = total_cartons - cartons_3) :
  3 * cartons_3 + 2 * cartons_2 = 100 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_rackets_packed_l723_72313


namespace NUMINAMATH_GPT_shorter_piece_length_l723_72334

-- Definitions according to conditions in a)
variables (x : ℝ) (total_length : ℝ := 140)
variables (ratio : ℝ := 5 / 2)

-- Statement to be proved
theorem shorter_piece_length : x + ratio * x = total_length → x = 40 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l723_72334


namespace NUMINAMATH_GPT_greatest_integer_x_l723_72319

theorem greatest_integer_x (x : ℤ) : (5 - 4 * x > 17) → x ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_x_l723_72319


namespace NUMINAMATH_GPT_cupcakes_sold_l723_72398

theorem cupcakes_sold (initial_made sold additional final : ℕ) (h1 : initial_made = 42) (h2 : additional = 39) (h3 : final = 59) :
  (initial_made - sold + additional = final) -> sold = 22 :=
by
  intro h
  rw [h1, h2, h3] at h
  sorry

end NUMINAMATH_GPT_cupcakes_sold_l723_72398


namespace NUMINAMATH_GPT_percentage_le_29_l723_72342

def sample_size : ℕ := 100
def freq_17_19 : ℕ := 1
def freq_19_21 : ℕ := 1
def freq_21_23 : ℕ := 3
def freq_23_25 : ℕ := 3
def freq_25_27 : ℕ := 18
def freq_27_29 : ℕ := 16
def freq_29_31 : ℕ := 28
def freq_31_33 : ℕ := 30

theorem percentage_le_29 : (freq_17_19 + freq_19_21 + freq_21_23 + freq_23_25 + freq_25_27 + freq_27_29) * 100 / sample_size = 42 :=
by
  sorry

end NUMINAMATH_GPT_percentage_le_29_l723_72342


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l723_72396

variable (a b c : ℝ)

theorem problem1 : a^4 * (a^2)^3 = a^10 :=
by
  sorry

theorem problem2 : 2 * a^3 * b^2 * c / (1 / 3 * a^2 * b) = 6 * a * b * c :=
by
  sorry

theorem problem3 : 6 * a * (1 / 3 * a * b - b) - (2 * a * b + b) * (a - 1) = -5 * a * b + b :=
by
  sorry

theorem problem4 : (a - 2)^2 - (3 * a + 2 * b) * (3 * a - 2 * b) = -8 * a^2 - 4 * a + 4 + 4 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l723_72396


namespace NUMINAMATH_GPT_max_tickets_jane_can_buy_l723_72389

-- Define ticket prices and Jane's budget
def ticket_price := 15
def discounted_price := 12
def discount_threshold := 5
def jane_budget := 150

-- Prove that the maximum number of tickets Jane can buy is 11
theorem max_tickets_jane_can_buy : 
  ∃ (n : ℕ), n ≤ 11 ∧ (if n ≤ discount_threshold then ticket_price * n ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (n - discount_threshold)) ≤ jane_budget)
  ∧ ∀ m : ℕ, (if m ≤ 11 then (if m ≤ discount_threshold then ticket_price * m ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (m - discount_threshold)) ≤ jane_budget) else false)  → m ≤ 11 := 
by
  sorry

end NUMINAMATH_GPT_max_tickets_jane_can_buy_l723_72389


namespace NUMINAMATH_GPT_greatest_visible_unit_cubes_from_single_point_l723_72300

-- Define the size of the cube
def cube_size : ℕ := 9

-- The total number of unit cubes in the 9x9x9 cube
def total_unit_cubes (n : ℕ) : ℕ := n^3

-- The greatest number of unit cubes visible from a single point
def visible_unit_cubes (n : ℕ) : ℕ := 3 * n^2 - 3 * (n - 1) + 1

-- The given cube size is 9
def given_cube_size : ℕ := cube_size

-- The correct answer for the greatest number of visible unit cubes from a single point
def correct_visible_cubes : ℕ := 220

-- Theorem stating the visibility calculation for a 9x9x9 cube
theorem greatest_visible_unit_cubes_from_single_point :
  visible_unit_cubes cube_size = correct_visible_cubes := by
  sorry

end NUMINAMATH_GPT_greatest_visible_unit_cubes_from_single_point_l723_72300


namespace NUMINAMATH_GPT_scientific_notation_1300000_l723_72345

theorem scientific_notation_1300000 :
  1300000 = 1.3 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_1300000_l723_72345


namespace NUMINAMATH_GPT_difference_of_squares_divisibility_l723_72375

theorem difference_of_squares_divisibility (a b : ℤ) :
  ∃ m : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 8 * m ∧ 
           ¬∃ n : ℤ, (2 * a + 3) ^ 2 - (2 * b + 1) ^ 2 = 16 * n :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_divisibility_l723_72375


namespace NUMINAMATH_GPT_combined_return_percentage_l723_72399

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return1_percent return2_percent : ℝ) (total_investment total_return : ℝ) :
  investment1 = 500 → 
  return1_percent = 0.07 → 
  investment2 = 1500 → 
  return2_percent = 0.09 → 
  total_investment = investment1 + investment2 → 
  total_return = investment1 * return1_percent + investment2 * return2_percent → 
  (total_return / total_investment) * 100 = 8.5 :=
by 
  sorry

end NUMINAMATH_GPT_combined_return_percentage_l723_72399


namespace NUMINAMATH_GPT_monomial_sum_l723_72332

theorem monomial_sum (m n : ℤ) (h1 : n = 2) (h2 : m + 2 = 1) : m + n = 1 := by
  sorry

end NUMINAMATH_GPT_monomial_sum_l723_72332


namespace NUMINAMATH_GPT_proposition_2_proposition_4_l723_72395

variable {m n : Line}
variable {α β : Plane}

-- Define predicates for perpendicularity, parallelism, and containment
axiom line_parallel_plane (n : Line) (α : Plane) : Prop
axiom line_perp_plane (n : Line) (α : Plane) : Prop
axiom plane_perp_plane (α β : Plane) : Prop
axiom line_in_plane (m : Line) (β : Plane) : Prop

-- State the correct propositions
theorem proposition_2 (m n : Line) (α β : Plane)
  (h1 : line_perp_plane m n)
  (h2 : line_perp_plane n α)
  (h3 : line_perp_plane m β) :
  plane_perp_plane α β := sorry

theorem proposition_4 (n : Line) (α β : Plane)
  (h1 : line_perp_plane n β)
  (h2 : plane_perp_plane α β) :
  line_parallel_plane n α ∨ line_in_plane n α := sorry

end NUMINAMATH_GPT_proposition_2_proposition_4_l723_72395


namespace NUMINAMATH_GPT_difference_of_digits_is_three_l723_72316

def tens_digit (n : ℕ) : ℕ :=
  n / 10

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem difference_of_digits_is_three :
  ∀ n : ℕ, n = 63 → tens_digit n + ones_digit n = 9 → tens_digit n - ones_digit n = 3 :=
by
  intros n h1 h2
  sorry

end NUMINAMATH_GPT_difference_of_digits_is_three_l723_72316


namespace NUMINAMATH_GPT_num_possible_bases_l723_72369

theorem num_possible_bases (b : ℕ) (h1 : b ≥ 2) (h2 : b^3 ≤ 256) (h3 : 256 < b^4) : ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_possible_bases_l723_72369


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l723_72361

theorem repeating_decimal_as_fraction :
  ∃ x : ℝ, x = 7.45 ∧ (100 * x - x = 738) → x = 82 / 11 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l723_72361


namespace NUMINAMATH_GPT_find_b_if_lines_parallel_l723_72359

-- Definitions of the line equations and parallel condition
def first_line (x y : ℝ) (b : ℝ) : Prop := 3 * y - b = -9 * x + 1
def second_line (x y : ℝ) (b : ℝ) : Prop := 2 * y + 8 = (b - 3) * x - 2

-- Definition of parallel lines (their slopes are equal)
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

-- Given conditions and the conclusion to prove
theorem find_b_if_lines_parallel :
  ∃ b : ℝ, (∀ x y : ℝ, first_line x y b → ∃ m1 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m1 * x + c) ∧ 
           (∀ x y : ℝ, second_line x y b → ∃ m2 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m2 * x + c) ∧ 
           parallel_lines (-3) ((b - 3) / 2) →
           b = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_if_lines_parallel_l723_72359


namespace NUMINAMATH_GPT_area_of_large_rectangle_l723_72346

-- Define the given areas for the sub-shapes
def shaded_square_area : ℝ := 4
def bottom_rectangle_area : ℝ := 2
def right_rectangle_area : ℝ := 6

-- Prove the total area of the large rectangle EFGH is 12 square inches
theorem area_of_large_rectangle : shaded_square_area + bottom_rectangle_area + right_rectangle_area = 12 := 
by 
sorry

end NUMINAMATH_GPT_area_of_large_rectangle_l723_72346


namespace NUMINAMATH_GPT_approximate_number_of_fish_in_pond_l723_72386

theorem approximate_number_of_fish_in_pond :
  (∃ N : ℕ, 
  (∃ tagged1 tagged2 : ℕ, tagged1 = 50 ∧ tagged2 = 10) ∧
  (∃ caught1 caught2 : ℕ, caught1 = 50 ∧ caught2 = 50) ∧
  ((tagged2 : ℝ) / caught2 = (tagged1 : ℝ) / (N : ℝ)) ∧
  N = 250) :=
sorry

end NUMINAMATH_GPT_approximate_number_of_fish_in_pond_l723_72386


namespace NUMINAMATH_GPT_problem_statement_l723_72339

variable (a b c d : ℝ)

-- Definitions for the conditions
def condition1 := a + b + c + d = 100
def condition2 := (a / (b + c + d)) + (b / (a + c + d)) + (c / (a + b + d)) + (d / (a + b + c)) = 95

-- The theorem which needs to be proved
theorem problem_statement (h1 : condition1 a b c d) (h2 : condition2 a b c d) :
  (1 / (b + c + d)) + (1 / (a + c + d)) + (1 / (a + b + d)) + (1 / (a + b + c)) = 99 / 100 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l723_72339


namespace NUMINAMATH_GPT_range_of_a_ineq_l723_72356

noncomputable def range_of_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ x₁ * x₁ + (a * a - 1) * x₁ + (a - 2) = 0 ∧
                x₂ * x₂ + (a * a - 1) * x₂ + (a - 2) = 0

theorem range_of_a_ineq (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧
    x₁^2 + (a^2 - 1) * x₁ + (a - 2) = 0 ∧
    x₂^2 + (a^2 - 1) * x₂ + (a - 2) = 0) → -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_ineq_l723_72356


namespace NUMINAMATH_GPT_iron_needed_for_hydrogen_l723_72378

-- Conditions of the problem
def reaction (Fe H₂SO₄ FeSO₄ H₂ : ℕ) : Prop :=
  Fe + H₂SO₄ = FeSO₄ + H₂

-- Given data
def balanced_equation : Prop :=
  reaction 1 1 1 1
 
def produced_hydrogen : ℕ := 2
def produced_from_sulfuric_acid : ℕ := 2
def needed_iron : ℕ := 2

-- Problem statement to be proved
theorem iron_needed_for_hydrogen (H₂SO₄ H₂ : ℕ) (h1 : produced_hydrogen = H₂) (h2 : produced_from_sulfuric_acid = H₂SO₄) (balanced_eq : balanced_equation) :
  needed_iron = 2 := by
sorry

end NUMINAMATH_GPT_iron_needed_for_hydrogen_l723_72378


namespace NUMINAMATH_GPT_missing_fraction_correct_l723_72315

theorem missing_fraction_correct : 
  (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-2 / 15) + (3 / 5) = 0.13333333333333333 :=
by sorry

end NUMINAMATH_GPT_missing_fraction_correct_l723_72315


namespace NUMINAMATH_GPT_prob1_part1_prob1_part2_l723_72340

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 5}
noncomputable def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2 * a}

theorem prob1_part1 (a : ℝ) (ha : a = 3) :
  A ∪ B a = {x | -2 < x ∧ x < 7} ∧ A ∩ B a = {x | -1 < x ∧ x < 5} :=
by {
  sorry
}

theorem prob1_part2 (h : ∀ x, x ∈ A → x ∈ B a) :
  ∀ a : ℝ, a ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_prob1_part1_prob1_part2_l723_72340


namespace NUMINAMATH_GPT_volume_ratio_of_cubes_l723_72388

-- Given conditions
def edge_length_smaller_cube : ℝ := 6
def edge_length_larger_cube : ℝ := 12

-- Problem statement
theorem volume_ratio_of_cubes : 
  (edge_length_smaller_cube / edge_length_larger_cube) ^ 3 = (1 / 8) := 
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_cubes_l723_72388


namespace NUMINAMATH_GPT_problem_statement_l723_72322

-- Definitions from the problem conditions
variable (r : ℝ) (A B C : ℝ)

-- Problem condition that A, B are endpoints of the diameter of the circle
-- Defining the length AB being the diameter -> length AB = 2r
def AB := 2 * r

-- Condition that ABC is inscribed in a circle and AB is the diameter implies the angle ACB = 90°
-- Using Thales' theorem we know that A, B, C satisfy certain geometric properties in a right triangle
-- AC and BC are the other two sides with H right angle at C.

-- Proving the target equation
theorem problem_statement (h : C ≠ A ∧ C ≠ B) : (AC + BC)^2 ≤ 8 * r^2 := 
sorry


end NUMINAMATH_GPT_problem_statement_l723_72322


namespace NUMINAMATH_GPT_equation_of_line_l723_72328

theorem equation_of_line 
  (slope : ℝ)
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h_slope : slope = 2)
  (h_line1 : a1 = 3 ∧ b1 = 4 ∧ c1 = -5)
  (h_line2 : a2 = 3 ∧ b2 = -4 ∧ c2 = -13) 
  : ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = -7) ∧ 
    (∀ x y : ℝ, (a1 * x + b1 * y + c1 = 0) ∧ (a2 * x + b2 * y + c2 = 0) → (a * x + b * y + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l723_72328


namespace NUMINAMATH_GPT_packs_of_yellow_bouncy_balls_l723_72351

/-- Maggie bought 4 packs of red bouncy balls, some packs of yellow bouncy balls (denoted as Y), and 4 packs of green bouncy balls. -/
theorem packs_of_yellow_bouncy_balls (Y : ℕ) : 
  (4 + Y + 4) * 10 = 160 -> Y = 8 := 
by 
  sorry

end NUMINAMATH_GPT_packs_of_yellow_bouncy_balls_l723_72351


namespace NUMINAMATH_GPT_compute_expression_l723_72329

theorem compute_expression : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l723_72329


namespace NUMINAMATH_GPT_train_passing_time_l723_72385

theorem train_passing_time
  (length_A : ℝ) (length_B : ℝ) (time_A : ℝ) (speed_B : ℝ) 
  (Dir_opposite : true) 
  (passenger_on_A_time : time_A = 10)
  (length_of_A : length_A = 150)
  (length_of_B : length_B = 200)
  (relative_speed : speed_B = length_B / time_A) :
  ∃ x : ℝ, length_A / x = length_B / time_A ∧ x = 7.5 :=
by
  -- conditions stated
  sorry

end NUMINAMATH_GPT_train_passing_time_l723_72385


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l723_72365

theorem arithmetic_sequence_geometric_condition :
  ∃ d : ℝ, d ≠ 0 ∧ (∀ (a_n : ℕ → ℝ), (a_n 1 = 1) ∧ 
    (a_n 3 = a_n 1 + 2 * d) ∧ (a_n 13 = a_n 1 + 12 * d) ∧ 
    (a_n 3 ^ 2 = a_n 1 * a_n 13) ↔ d = 2) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l723_72365


namespace NUMINAMATH_GPT_cost_price_of_watch_l723_72384

theorem cost_price_of_watch :
  ∃ (CP : ℝ), (CP * 1.07 = CP * 0.88 + 250) ∧ CP = 250 / 0.19 :=
sorry

end NUMINAMATH_GPT_cost_price_of_watch_l723_72384


namespace NUMINAMATH_GPT_calculate_total_payment_l723_72306

theorem calculate_total_payment
(adult_price : ℕ := 30)
(teen_price : ℕ := 20)
(child_price : ℕ := 15)
(num_adults : ℕ := 4)
(num_teenagers : ℕ := 4)
(num_children : ℕ := 2)
(num_activities : ℕ := 5)
(has_coupon : Bool := true)
(soda_price : ℕ := 5)
(num_sodas : ℕ := 5)

(total_admission_before_discount : ℕ := 
  num_adults * adult_price + num_teenagers * teen_price + num_children * child_price)
(discount_on_activities : ℕ := if num_activities >= 7 then 15 else if num_activities >= 5 then 10 else if num_activities >= 3 then 5 else 0)
(admission_after_activity_discount : ℕ := 
  total_admission_before_discount - total_admission_before_discount * discount_on_activities / 100)
(additional_discount : ℕ := if has_coupon then 5 else 0)
(admission_after_all_discounts : ℕ := 
  admission_after_activity_discount - admission_after_activity_discount * additional_discount / 100)

(total_cost : ℕ := admission_after_all_discounts + num_sodas * soda_price) :
total_cost = 22165 := 
sorry

end NUMINAMATH_GPT_calculate_total_payment_l723_72306


namespace NUMINAMATH_GPT_students_first_day_l723_72341

-- Definitions based on conditions
def total_books : ℕ := 120
def books_per_student : ℕ := 5
def students_second_day : ℕ := 5
def students_third_day : ℕ := 6
def students_fourth_day : ℕ := 9

-- Main goal
theorem students_first_day (total_books_eq : total_books = 120)
                           (books_per_student_eq : books_per_student = 5)
                           (students_second_day_eq : students_second_day = 5)
                           (students_third_day_eq : students_third_day = 6)
                           (students_fourth_day_eq : students_fourth_day = 9) :
  let books_given_second_day := students_second_day * books_per_student
  let books_given_third_day := students_third_day * books_per_student
  let books_given_fourth_day := students_fourth_day * books_per_student
  let total_books_given_after_first_day := books_given_second_day + books_given_third_day + books_given_fourth_day
  let books_first_day := total_books - total_books_given_after_first_day
  let students_first_day := books_first_day / books_per_student
  students_first_day = 4 :=
by sorry

end NUMINAMATH_GPT_students_first_day_l723_72341


namespace NUMINAMATH_GPT_Tim_total_score_l723_72307

/-- Given the following conditions:
1. A single line is worth 1000 points.
2. A tetris is worth 8 times a single line.
3. If a single line and a tetris are made consecutively, the score of the tetris doubles.
4. If two tetrises are scored back to back, an additional 5000-point bonus is awarded.
5. If a player scores a single, double and triple line consecutively, a 3000-point bonus is awarded.
6. Tim scored 6 singles, 4 tetrises, 2 doubles, and 1 triple during his game.
7. He made a single line and a tetris consecutively once, scored 2 tetrises back to back, 
   and scored a single, double and triple consecutively.
Prove that Tim’s total score is 54000 points.
-/
theorem Tim_total_score :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6 * single_points
  let tetrises := 4 * tetris_points
  let base_score := singles + tetrises
  let consecutive_tetris_bonus := tetris_points
  let back_to_back_tetris_bonus := 5000
  let consecutive_lines_bonus := 3000
  let total_score := base_score + consecutive_tetris_bonus + back_to_back_tetris_bonus + consecutive_lines_bonus
  total_score = 54000 := by
  sorry

end NUMINAMATH_GPT_Tim_total_score_l723_72307


namespace NUMINAMATH_GPT_calculate_E_l723_72391

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end NUMINAMATH_GPT_calculate_E_l723_72391


namespace NUMINAMATH_GPT_average_age_is_35_l723_72317

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end NUMINAMATH_GPT_average_age_is_35_l723_72317


namespace NUMINAMATH_GPT_ball_distribution_l723_72370

theorem ball_distribution (basketballs volleyballs classes balls : ℕ) 
  (h1 : basketballs = 2) 
  (h2 : volleyballs = 3) 
  (h3 : classes = 4) 
  (h4 : balls = 4) :
  (classes.choose 3) + (classes.choose 2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_ball_distribution_l723_72370


namespace NUMINAMATH_GPT_price_difference_l723_72304

def P := ℝ

def Coupon_A_savings (P : ℝ) := 0.20 * P
def Coupon_B_savings : ℝ := 40
def Coupon_C_savings (P : ℝ) := 0.30 * (P - 120) + 20

def Coupon_A_geq_Coupon_B (P : ℝ) := Coupon_A_savings P ≥ Coupon_B_savings
def Coupon_A_geq_Coupon_C (P : ℝ) := Coupon_A_savings P ≥ Coupon_C_savings P

noncomputable def x : ℝ := 200
noncomputable def y : ℝ := 300

theorem price_difference (P : ℝ) (h1 : P > 120)
  (h2 : Coupon_A_geq_Coupon_B P)
  (h3 : Coupon_A_geq_Coupon_C P) :
  y - x = 100 := by
  sorry

end NUMINAMATH_GPT_price_difference_l723_72304


namespace NUMINAMATH_GPT_find_magnitude_a_l723_72336

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c (m : ℝ) : ℝ × ℝ := (2, m)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem find_magnitude_a (m : ℝ) (h : dot_product (vector_add (vector_a m) (vector_c m)) (vector_b m) = 0) :
  magnitude (vector_a (-1 / 2)) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_magnitude_a_l723_72336


namespace NUMINAMATH_GPT_solution_set_f_x_leq_x_range_of_a_l723_72392

-- Definition of the function f
def f (x : ℝ) : ℝ := |2 * x - 7| + 1

-- Proof Problem for Question (1):
-- Given: f(x) = |2x - 7| + 1
-- Prove: The solution set of the inequality f(x) <= x is {x | 8/3 <= x <= 6}
theorem solution_set_f_x_leq_x :
  { x : ℝ | f x ≤ x } = { x : ℝ | 8 / 3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := f x - 2 * |x - 1|

-- Proof Problem for Question (2):
-- Given: f(x) = |2x - 7| + 1 and g(x) = f(x) - 2 * |x - 1|
-- Prove: If ∃ x, g(x) <= a, then a >= -4
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 :=
sorry

end NUMINAMATH_GPT_solution_set_f_x_leq_x_range_of_a_l723_72392


namespace NUMINAMATH_GPT_rectangle_area_l723_72354

theorem rectangle_area (area_square : ℝ) 
  (width_rectangle : ℝ) (length_rectangle : ℝ)
  (h1 : area_square = 16)
  (h2 : width_rectangle^2 = area_square)
  (h3 : length_rectangle = 3 * width_rectangle) :
  width_rectangle * length_rectangle = 48 := by sorry

end NUMINAMATH_GPT_rectangle_area_l723_72354


namespace NUMINAMATH_GPT_product_of_integers_l723_72311

theorem product_of_integers (a b : ℚ) (h1 : a / b = 12) (h2 : a + b = 144) :
  a * b = 248832 / 169 := 
sorry

end NUMINAMATH_GPT_product_of_integers_l723_72311


namespace NUMINAMATH_GPT_black_area_remaining_after_changes_l723_72377

theorem black_area_remaining_after_changes :
  let initial_fraction_black := 1
  let change_factor := 8 / 9
  let num_changes := 4
  let final_fraction_black := (change_factor ^ num_changes)
  final_fraction_black = 4096 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_black_area_remaining_after_changes_l723_72377


namespace NUMINAMATH_GPT_inverse_of_g_compose_three_l723_72324

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Assuming g(x) is defined only for x in {1, 2, 3, 4, 5}

noncomputable def g_inv (y : ℕ) : ℕ :=
  match y with
  | 4 => 1
  | 3 => 2
  | 1 => 3
  | 5 => 4
  | 2 => 5
  | _ => 0  -- Assuming g_inv(y) is defined only for y in {1, 3, 1, 5, 2}

theorem inverse_of_g_compose_three : g_inv (g_inv (g_inv 3)) = 4 := by
  sorry

end NUMINAMATH_GPT_inverse_of_g_compose_three_l723_72324


namespace NUMINAMATH_GPT_determine_fake_coin_weight_l723_72374

theorem determine_fake_coin_weight
  (coins : Fin 25 → ℤ) 
  (fake_coin : Fin 25) 
  (all_same_weight : ∀ (i j : Fin 25), i ≠ fake_coin → j ≠ fake_coin → coins i = coins j)
  (fake_diff_weight : ∃ (x : Fin 25), (coins x ≠ coins fake_coin)) :
  ∃ (is_heavy : Bool), 
    (is_heavy = true ↔ coins fake_coin > coins (Fin.ofNat 0)) ∨ 
    (is_heavy = false ↔ coins fake_coin < coins (Fin.ofNat 0)) :=
  sorry

end NUMINAMATH_GPT_determine_fake_coin_weight_l723_72374


namespace NUMINAMATH_GPT_circle_tangent_l723_72397

variables {O M : ℝ} {R : ℝ}

theorem circle_tangent
  (r : ℝ)
  (hOM_pos : O ≠ M)
  (hO : O > 0)
  (hR : R > 0)
  (h_distinct : ∀ (m n : ℝ), m ≠ n → abs (m - n) ≠ 0) :
  (r = abs (O - M) - R) ∨ (r = abs (O - M) + R) ∨ (r = R - abs (O - M)) →
  (abs ((O - M)^2 + r^2 - R^2) = 2 * R * r) :=
sorry

end NUMINAMATH_GPT_circle_tangent_l723_72397


namespace NUMINAMATH_GPT_triangle_area_l723_72343

theorem triangle_area (l1 l2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, l1 x y ↔ 3 * x - y + 12 = 0)
  (h2 : ∀ x y, l2 x y ↔ 3 * x + 2 * y - 6 = 0) :
  ∃ A : ℝ, A = 9 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l723_72343


namespace NUMINAMATH_GPT_sum_of_abc_l723_72382

theorem sum_of_abc (a b c : ℝ) (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 = 0) :
  a + b + c = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_abc_l723_72382


namespace NUMINAMATH_GPT_mean_home_runs_per_game_l723_72308

variable (home_runs : Nat) (games_played : Nat)

def total_home_runs : Nat := 
  (5 * 4) + (6 * 5) + (4 * 7) + (3 * 9) + (2 * 11)

def total_games_played : Nat :=
  (5 * 5) + (6 * 6) + (4 * 8) + (3 * 10) + (2 * 12)

theorem mean_home_runs_per_game :
  (total_home_runs : ℚ) / total_games_played = 127 / 147 :=
  by 
    sorry

end NUMINAMATH_GPT_mean_home_runs_per_game_l723_72308


namespace NUMINAMATH_GPT_aqua_park_earnings_l723_72357

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_aqua_park_earnings_l723_72357


namespace NUMINAMATH_GPT_liza_final_balance_l723_72393

theorem liza_final_balance :
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries (balance : ℝ) := 0.2 * balance
  let friday_interest (balance : ℝ) := 0.02 * balance
  let saturday_phone_bill := 70
  let saturday_additional_deposit := 300
  let tuesday_balance := monday_balance - tuesday_rent
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let thursday_balance_before_groceries := wednesday_balance - thursday_electric_bill - thursday_internet_bill
  let thursday_balance_after_groceries := thursday_balance_before_groceries - thursday_groceries thursday_balance_before_groceries
  let friday_balance := thursday_balance_after_groceries + friday_interest thursday_balance_after_groceries
  let saturday_balance_after_phone := friday_balance - saturday_phone_bill
  let final_balance := saturday_balance_after_phone + saturday_additional_deposit
  final_balance = 1562.528 :=
by
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries := 0.2 * (800 - 450 + 1500 - 117 - 100)
  let friday_interest := 0.02 * (800 - 450 + 1500 - 117 - 100 - 0.2 * (800 - 450 + 1500 - 117 - 100))
  let final_balance := 800 - 450 + 1500 - 117 - 100 - thursday_groceries + friday_interest - 70 + 300
  sorry

end NUMINAMATH_GPT_liza_final_balance_l723_72393
