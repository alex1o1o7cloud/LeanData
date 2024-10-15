import Mathlib

namespace NUMINAMATH_GPT_circle_form_eq_standard_form_l779_77914

theorem circle_form_eq_standard_form :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 11 := 
by
  intro x y
  sorry

end NUMINAMATH_GPT_circle_form_eq_standard_form_l779_77914


namespace NUMINAMATH_GPT_real_solution_four_unknowns_l779_77987

theorem real_solution_four_unknowns (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x * (y + z + t) ↔ (x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) :=
by
  sorry

end NUMINAMATH_GPT_real_solution_four_unknowns_l779_77987


namespace NUMINAMATH_GPT_Kylie_US_coins_left_l779_77923

-- Define the given conditions
def initial_US_coins : ℝ := 15
def Euro_coins : ℝ := 13
def Canadian_coins : ℝ := 8
def US_coins_given_to_Laura : ℝ := 21
def Euro_to_US_rate : ℝ := 1.18
def Canadian_to_US_rate : ℝ := 0.78

-- Define the conversions
def Euro_to_US : ℝ := Euro_coins * Euro_to_US_rate
def Canadian_to_US : ℝ := Canadian_coins * Canadian_to_US_rate
def total_US_before_giving : ℝ := initial_US_coins + Euro_to_US + Canadian_to_US
def US_left_with : ℝ := total_US_before_giving - US_coins_given_to_Laura

-- Statement of the problem to be proven
theorem Kylie_US_coins_left :
  US_left_with = 15.58 := by
  sorry

end NUMINAMATH_GPT_Kylie_US_coins_left_l779_77923


namespace NUMINAMATH_GPT_max_f_value_l779_77958

open Real

noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

theorem max_f_value : ∃ (x₀ y₀ : ℝ), (0 < x₀) ∧ (0 < y₀) ∧ (∀ (x y : ℝ), (0 < x) → (0 < y) → f x y ≤ f x₀ y₀) ∧ f x₀ y₀ = 1 / sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_max_f_value_l779_77958


namespace NUMINAMATH_GPT_solve_for_x_l779_77983

theorem solve_for_x (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l779_77983


namespace NUMINAMATH_GPT_max_square_plots_l779_77954
-- Lean 4 statement for the equivalent math problem

theorem max_square_plots (w l f s : ℕ) (h₁ : w = 40) (h₂ : l = 60) 
                         (h₃ : f = 2400) (h₄ : s ≠ 0) (h₅ : 2400 - 100 * s ≤ 2400)
                         (h₆ : w % s = 0) (h₇ : l % s = 0) :
  (w * l) / (s * s) = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_square_plots_l779_77954


namespace NUMINAMATH_GPT_max_value_when_a_zero_range_of_a_for_one_zero_l779_77981

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_when_a_zero :
  ∃ x : ℝ, x > 0 ∧ f 0 x = -1 :=
by sorry

theorem range_of_a_for_one_zero :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a > 0 :=
by sorry

end NUMINAMATH_GPT_max_value_when_a_zero_range_of_a_for_one_zero_l779_77981


namespace NUMINAMATH_GPT_value_of_x7_plus_64x2_l779_77921

-- Let x be a real number such that x^3 + 4x = 8.
def x_condition (x : ℝ) : Prop := x^3 + 4 * x = 8

-- We need to determine the value of x^7 + 64x^2.
theorem value_of_x7_plus_64x2 (x : ℝ) (h : x_condition x) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x7_plus_64x2_l779_77921


namespace NUMINAMATH_GPT_circle_area_conversion_l779_77913

-- Define the given diameter
def diameter (d : ℝ) := d = 8

-- Define the radius calculation
def radius (r : ℝ) := r = 4

-- Define the formula for the area of the circle in square meters
def area_sq_m (A : ℝ) := A = 16 * Real.pi

-- Define the conversion factor from square meters to square centimeters
def conversion_factor := 10000

-- Define the expected area in square centimeters
def area_sq_cm (A : ℝ) := A = 160000 * Real.pi

-- The theorem to prove
theorem circle_area_conversion (d r A_cm : ℝ) (h1 : diameter d) (h2 : radius r) (h3 : area_sq_cm A_cm) :
  A_cm = 160000 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_conversion_l779_77913


namespace NUMINAMATH_GPT_three_digit_number_441_or_882_l779_77979

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  n = 100 * a + 10 * b + c ∧
  n / (100 * c + 10 * b + a) = 3 ∧
  n % (100 * c + 10 * b + a) = a + b + c

theorem three_digit_number_441_or_882:
  ∀ n : ℕ, is_valid_number n → (n = 441 ∨ n = 882) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_441_or_882_l779_77979


namespace NUMINAMATH_GPT_convex_polyhedron_inequality_l779_77977

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end NUMINAMATH_GPT_convex_polyhedron_inequality_l779_77977


namespace NUMINAMATH_GPT_inequality_proof_l779_77935

theorem inequality_proof (a b : ℤ) (ha : a > 0) (hb : b > 0) : a + b ≤ 1 + a * b :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l779_77935


namespace NUMINAMATH_GPT_eight_is_100_discerning_nine_is_not_100_discerning_l779_77948

-- Define what it means to be b-discerning
def is_b_discerning (n b : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧ (∀ (U V : Finset ℕ), U ≠ V ∧ U ⊆ S ∧ V ⊆ S → U.sum id ≠ V.sum id)

-- Prove that 8 is 100-discerning
theorem eight_is_100_discerning : is_b_discerning 8 100 :=
sorry

-- Prove that 9 is not 100-discerning
theorem nine_is_not_100_discerning : ¬is_b_discerning 9 100 :=
sorry

end NUMINAMATH_GPT_eight_is_100_discerning_nine_is_not_100_discerning_l779_77948


namespace NUMINAMATH_GPT_solve_trig_equation_l779_77964
open Real

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (1 / 2) * abs (cos (2 * x) + (1 / 2)) = (sin (3 * x))^2 - (sin x) * (sin (3 * x))

-- Define the correct solution set 
def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (π / 6) + (k * (π / 2)) ∨ x = -(π / 6) + (k * (π / 2))

-- The theorem we need to prove
theorem solve_trig_equation : ∀ x : ℝ, original_equation x ↔ solution_set x :=
by sorry

end NUMINAMATH_GPT_solve_trig_equation_l779_77964


namespace NUMINAMATH_GPT_x_intercept_of_line_l779_77915

theorem x_intercept_of_line : ∃ x : ℚ, (6 * x, 0) = (35 / 6, 0) :=
by
  use 35 / 6
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l779_77915


namespace NUMINAMATH_GPT_marnie_eats_chips_l779_77982

theorem marnie_eats_chips (total_chips : ℕ) (chips_first_batch : ℕ) (chips_second_batch : ℕ) (daily_chips : ℕ) (remaining_chips : ℕ) (total_days : ℕ) :
  total_chips = 100 →
  chips_first_batch = 5 →
  chips_second_batch = 5 →
  daily_chips = 10 →
  remaining_chips = total_chips - (chips_first_batch + chips_second_batch) →
  total_days = remaining_chips / daily_chips + 1 →
  total_days = 10 :=
by
  sorry

end NUMINAMATH_GPT_marnie_eats_chips_l779_77982


namespace NUMINAMATH_GPT_jane_performance_l779_77942

theorem jane_performance :
  ∃ (p w e : ℕ), 
  p + w + e = 15 ∧ 
  2 * p + 4 * w + 6 * e = 66 ∧ 
  e = p + 4 ∧ 
  w = 11 :=
by
  sorry

end NUMINAMATH_GPT_jane_performance_l779_77942


namespace NUMINAMATH_GPT_best_approximation_of_x_squared_l779_77943

theorem best_approximation_of_x_squared
  (x : ℝ) (A B C D E : ℝ)
  (h1 : -2 < -1)
  (h2 : -1 < 0)
  (h3 : 0 < 1)
  (h4 : 1 < 2)
  (hx : -1 < x ∧ x < 0)
  (hC : 0 < C ∧ C < 1) :
  x^2 = C :=
sorry

end NUMINAMATH_GPT_best_approximation_of_x_squared_l779_77943


namespace NUMINAMATH_GPT_kelly_initial_sony_games_l779_77900

def nintendo_games : ℕ := 46
def sony_games_given_away : ℕ := 101
def sony_games_left : ℕ := 31

theorem kelly_initial_sony_games :
  sony_games_given_away + sony_games_left = 132 :=
by
  sorry

end NUMINAMATH_GPT_kelly_initial_sony_games_l779_77900


namespace NUMINAMATH_GPT_find_omega_l779_77946

noncomputable def omega_solution (ω : ℝ) : Prop :=
  ω > 0 ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) > 2 * Real.cos (ω * y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) ≥ 1)

theorem find_omega : omega_solution (1 / 2) :=
sorry

end NUMINAMATH_GPT_find_omega_l779_77946


namespace NUMINAMATH_GPT_find_a_l779_77909

def F (a b c : ℝ) : ℝ := a * b^3 + c

theorem find_a (a : ℝ) (h : F a 3 8 = F a 5 12) : a = -2 / 49 := by
  sorry

end NUMINAMATH_GPT_find_a_l779_77909


namespace NUMINAMATH_GPT_transport_capacity_l779_77920

-- Declare x and y as the amount of goods large and small trucks can transport respectively
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := 2 * x + 3 * y = 15.5
def condition2 : Prop := 5 * x + 6 * y = 35

-- The goal to prove
def goal : Prop := 3 * x + 5 * y = 24.5

-- Main theorem stating that given the conditions, the goal follows
theorem transport_capacity (h1 : condition1 x y) (h2 : condition2 x y) : goal x y :=
by sorry

end NUMINAMATH_GPT_transport_capacity_l779_77920


namespace NUMINAMATH_GPT_number_of_workers_in_original_scenario_l779_77917

-- Definitions based on the given conditions
def original_days := 70
def alternative_days := 42
def alternative_workers := 50

-- The statement we want to prove
theorem number_of_workers_in_original_scenario : 
  (∃ (W : ℕ), W * original_days = alternative_workers * alternative_days) → ∃ (W : ℕ), W = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_workers_in_original_scenario_l779_77917


namespace NUMINAMATH_GPT_failed_students_calculation_l779_77968

theorem failed_students_calculation (total_students : ℕ) (percentage_passed : ℕ)
  (h_total : total_students = 840) (h_passed : percentage_passed = 35) :
  (total_students * (100 - percentage_passed) / 100) = 546 :=
by
  sorry

end NUMINAMATH_GPT_failed_students_calculation_l779_77968


namespace NUMINAMATH_GPT_geometric_sum_S30_l779_77972

theorem geometric_sum_S30 (S : ℕ → ℝ) (h1 : S 10 = 10) (h2 : S 20 = 30) : S 30 = 70 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sum_S30_l779_77972


namespace NUMINAMATH_GPT_correct_propositions_l779_77967

-- Definitions based on the propositions
def prop1 := 
"Sampling every 20 minutes from a uniformly moving production line is stratified sampling."

def prop2 := 
"The stronger the correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1."

def prop3 := 
"In the regression line equation hat_y = 0.2 * x + 12, the forecasted variable hat_y increases by 0.2 units on average for each unit increase in the explanatory variable x."

def prop4 := 
"For categorical variables X and Y, the smaller the observed value k of their statistic K², the greater the certainty of the relationship between X and Y."

-- Mathematical statements for propositions
def p1 : Prop := false -- Proposition ① is incorrect
def p2 : Prop := true  -- Proposition ② is correct
def p3 : Prop := true  -- Proposition ③ is correct
def p4 : Prop := false -- Proposition ④ is incorrect

-- The theorem we need to prove
theorem correct_propositions : (p2 = true) ∧ (p3 = true) :=
by 
  -- Details of the proof here
  sorry

end NUMINAMATH_GPT_correct_propositions_l779_77967


namespace NUMINAMATH_GPT_negation_of_universal_l779_77939

theorem negation_of_universal {x : ℝ} : ¬ (∀ x > 0, x^2 - x ≤ 0) ↔ ∃ x > 0, x^2 - x > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l779_77939


namespace NUMINAMATH_GPT_remaining_sum_avg_l779_77908

variable (a b : ℕ → ℝ)
variable (h1 : 1 / 6 * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 2.5)
variable (h2 : 1 / 2 * (a 1 + a 2) = 1.1)
variable (h3 : 1 / 2 * (a 3 + a 4) = 1.4)

theorem remaining_sum_avg :
  1 / 2 * (a 5 + a 6) = 5 :=
by
  sorry

end NUMINAMATH_GPT_remaining_sum_avg_l779_77908


namespace NUMINAMATH_GPT_amount_paid_per_person_is_correct_l779_77990

noncomputable def amount_each_person_paid (total_bill : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) (num_people : ℕ) : ℝ := 
  let tip_amount := tip_rate * total_bill
  let tax_amount := tax_rate * total_bill
  let total_amount := total_bill + tip_amount + tax_amount
  total_amount / num_people

theorem amount_paid_per_person_is_correct :
  amount_each_person_paid 425 0.18 0.08 15 = 35.7 :=
by
  sorry

end NUMINAMATH_GPT_amount_paid_per_person_is_correct_l779_77990


namespace NUMINAMATH_GPT_tangents_parallel_l779_77973

variable {R : Type*} [Field R]

-- Let f be a function from ratios to slopes
variable (φ : R -> R)

-- Given points (x, y) and (x₁, y₁) with corresponding conditions
variable (x x₁ y y₁ : R)

-- Conditions
def corresponding_points := y / x = y₁ / x₁
def homogeneous_diff_eqn := ∀ x y, (y / x) = φ (y / x)

-- Prove that the tangents are parallel
theorem tangents_parallel (h_corr : corresponding_points x x₁ y y₁)
  (h_diff_eqn : ∀ (x x₁ y y₁ : R), y' = φ (y / x) ∧ y₁' = φ (y₁ / x₁)) :
  y' = y₁' :=
by
  sorry

end NUMINAMATH_GPT_tangents_parallel_l779_77973


namespace NUMINAMATH_GPT_square_of_negative_eq_square_l779_77956

theorem square_of_negative_eq_square (a : ℝ) : (-a)^2 = a^2 :=
sorry

end NUMINAMATH_GPT_square_of_negative_eq_square_l779_77956


namespace NUMINAMATH_GPT_max_value_is_one_l779_77961

noncomputable def max_value (x y z : ℝ) : ℝ :=
  (x^2 - 2 * x * y + y^2) * (x^2 - 2 * x * z + z^2) * (y^2 - 2 * y * z + z^2)

theorem max_value_is_one :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 3 →
  max_value x y z ≤ 1 :=
by sorry

end NUMINAMATH_GPT_max_value_is_one_l779_77961


namespace NUMINAMATH_GPT_find_k_l779_77971

variable (k : ℝ)
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, 2)

theorem find_k 
  (h : (k * a.1 - b.1, k * a.2 - b.2) = (k - 1, k - 2)) 
  (perp_cond : (k * a.1 - b.1, k * a.2 - b.2).fst * (b.1 + a.1) + (k * a.1 - b.1, k * a.2 - b.2).snd * (b.2 + a.2) = 0) :
  k = 8 / 5 :=
sorry

end NUMINAMATH_GPT_find_k_l779_77971


namespace NUMINAMATH_GPT_sum_of_squares_iff_double_sum_of_squares_l779_77901

theorem sum_of_squares_iff_double_sum_of_squares (n : ℕ) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ a b : ℤ, 2 * n = a^2 + b^2) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_iff_double_sum_of_squares_l779_77901


namespace NUMINAMATH_GPT_problem_solution_l779_77907

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l779_77907


namespace NUMINAMATH_GPT_curve_intersection_four_points_l779_77962

theorem curve_intersection_four_points (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 * a^2 ∧ y = a * x^2 - 2 * a) ∧ 
  (∃! (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ), 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧
    x1^2 + y1^2 = 4 * a^2 ∧ y1 = a * x1^2 - 2 * a ∧
    x2^2 + y2^2 = 4 * a^2 ∧ y2 = a * x2^2 - 2 * a ∧
    x3^2 + y3^2 = 4 * a^2 ∧ y3 = a * x3^2 - 2 * a ∧
    x4^2 + y4^2 = 4 * a^2 ∧ y4 = a * x4^2 - 2 * a) ↔ 
  a > 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_curve_intersection_four_points_l779_77962


namespace NUMINAMATH_GPT_cyclist_wait_time_l779_77995

noncomputable def hiker_speed : ℝ := 5 / 60
noncomputable def cyclist_speed : ℝ := 25 / 60
noncomputable def wait_time : ℝ := 5
noncomputable def distance_ahead : ℝ := cyclist_speed * wait_time
noncomputable def catching_time : ℝ := distance_ahead / hiker_speed

theorem cyclist_wait_time : catching_time = 25 := by
  sorry

end NUMINAMATH_GPT_cyclist_wait_time_l779_77995


namespace NUMINAMATH_GPT_half_angle_quadrant_l779_77986

theorem half_angle_quadrant
  (α : ℝ)
  (h1 : ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)
  (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
  ∃ k : ℤ, k * Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi * 3 / 4 ∧ Real.cos (α / 2) ≤ 0 := sorry

end NUMINAMATH_GPT_half_angle_quadrant_l779_77986


namespace NUMINAMATH_GPT_total_highlighters_l779_77969

-- Define the number of highlighters of each color
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

-- Prove the total number of highlighters
theorem total_highlighters : pink_highlighters + yellow_highlighters + blue_highlighters = 33 :=
by
  sorry

end NUMINAMATH_GPT_total_highlighters_l779_77969


namespace NUMINAMATH_GPT_Samanta_points_diff_l779_77940

variables (Samanta Mark Eric : ℕ)

/-- In a game, Samanta has some more points than Mark, Mark has 50% more points than Eric,
Eric has 6 points, and Samanta, Mark, and Eric have a total of 32 points. Prove that Samanta
has 8 more points than Mark. -/
theorem Samanta_points_diff 
    (h1 : Mark = Eric + Eric / 2) 
    (h2 : Eric = 6) 
    (h3 : Samanta + Mark + Eric = 32)
    : Samanta - Mark = 8 :=
sorry

end NUMINAMATH_GPT_Samanta_points_diff_l779_77940


namespace NUMINAMATH_GPT_width_of_larger_cuboid_l779_77975

theorem width_of_larger_cuboid
    (length_larger : ℝ)
    (width_larger : ℝ)
    (height_larger : ℝ)
    (length_smaller : ℝ)
    (width_smaller : ℝ)
    (height_smaller : ℝ)
    (num_smaller : ℕ)
    (volume_larger : ℝ)
    (volume_smaller : ℝ)
    (divided_into : Real) :
    length_larger = 12 → height_larger = 10 →
    length_smaller = 5 → width_smaller = 3 → height_smaller = 2 →
    num_smaller = 56 →
    volume_smaller = length_smaller * width_smaller * height_smaller →
    volume_larger = num_smaller * volume_smaller →
    volume_larger = length_larger * width_larger * height_larger →
    divided_into = volume_larger / (length_larger * height_larger) →
    width_larger = 14 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_width_of_larger_cuboid_l779_77975


namespace NUMINAMATH_GPT_inverse_proportion_k_value_l779_77930

theorem inverse_proportion_k_value (k m : ℝ) 
  (h1 : m = k / 3) 
  (h2 : 6 = k / (m - 1)) 
  : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_k_value_l779_77930


namespace NUMINAMATH_GPT_al_initial_portion_l779_77984

theorem al_initial_portion (a b c : ℝ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 200 + 2 * b + 1.5 * c = 1800) : 
  a = 600 :=
sorry

end NUMINAMATH_GPT_al_initial_portion_l779_77984


namespace NUMINAMATH_GPT_diagonal_length_of_regular_hexagon_l779_77911

-- Define a structure for the hexagon with a given side length
structure RegularHexagon (s : ℝ) :=
(side_length : ℝ := s)

-- Prove that the length of diagonal DB in a regular hexagon with side length 12 is 12√3
theorem diagonal_length_of_regular_hexagon (H : RegularHexagon 12) : 
  ∃ DB : ℝ, DB = 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_length_of_regular_hexagon_l779_77911


namespace NUMINAMATH_GPT_find_certain_number_l779_77959

open Real

noncomputable def certain_number (x : ℝ) : Prop :=
  0.75 * x = 0.50 * 900

theorem find_certain_number : certain_number 600 :=
by
  dsimp [certain_number]
  -- We need to show that 0.75 * 600 = 0.50 * 900
  sorry

end NUMINAMATH_GPT_find_certain_number_l779_77959


namespace NUMINAMATH_GPT_sin_180_degree_l779_77997

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end NUMINAMATH_GPT_sin_180_degree_l779_77997


namespace NUMINAMATH_GPT_graphs_symmetric_l779_77936

noncomputable def exp2 : ℝ → ℝ := λ x => 2^x
noncomputable def log2 : ℝ → ℝ := λ x => Real.log x / Real.log 2

theorem graphs_symmetric :
  ∀ (x y : ℝ), (y = exp2 x) ↔ (x = log2 y) := sorry

end NUMINAMATH_GPT_graphs_symmetric_l779_77936


namespace NUMINAMATH_GPT_fish_ratio_bobby_sarah_l779_77950

-- Defining the conditions
variables (bobby sarah tony billy : ℕ)

-- Condition: Billy has 10 fish.
def billy_has_10_fish : billy = 10 := by sorry

-- Condition: Tony has 3 times as many fish as Billy.
def tony_has_3_times_billy : tony = 3 * billy := by sorry

-- Condition: Sarah has 5 more fish than Tony.
def sarah_has_5_more_than_tony : sarah = tony + 5 := by sorry

-- Condition: All 4 people have 145 fish together.
def total_fish : bobby + sarah + tony + billy = 145 := by sorry

-- The theorem we want to prove
theorem fish_ratio_bobby_sarah : (bobby : ℚ) / sarah = 2 / 1 := by
  -- You can write out the entire proof step by step here, but initially, we'll just put sorry.
  sorry

end NUMINAMATH_GPT_fish_ratio_bobby_sarah_l779_77950


namespace NUMINAMATH_GPT_price_of_second_oil_l779_77999

theorem price_of_second_oil : 
  ∃ x : ℝ, 
    (10 * 50 + 5 * x = 15 * 56) → x = 68 := by
  sorry

end NUMINAMATH_GPT_price_of_second_oil_l779_77999


namespace NUMINAMATH_GPT_line_equation_M_l779_77989

theorem line_equation_M (x y : ℝ) :
  (∃ (m c : ℝ), y = m * x + c ∧ m = -5/4 ∧ c = -3)
  ∧ (∃ (slope intercept : ℝ), slope = 2 * (-5/4) ∧ intercept = (1/2) * -3 ∧ (y - 2 = slope * (x + 4)))
  → ∃ (a b : ℝ), y = a * x + b ∧ a = -5/2 ∧ b = -8 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_M_l779_77989


namespace NUMINAMATH_GPT_determine_k_for_intersection_l779_77925

theorem determine_k_for_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 3 = 2 * x + 5) ∧ 
  (∀ x₁ x₂ : ℝ, (k * x₁^2 + 2 * x₁ + 3 = 2 * x₁ + 5) ∧ 
                (k * x₂^2 + 2 * x₂ + 3 = 2 * x₂ + 5) → 
              x₁ = x₂) ↔ k = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_for_intersection_l779_77925


namespace NUMINAMATH_GPT_determine_range_of_a_l779_77934

theorem determine_range_of_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) : 
  a < 1 / 8 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_GPT_determine_range_of_a_l779_77934


namespace NUMINAMATH_GPT_line_length_l779_77924

theorem line_length (n : ℕ) (d : ℤ) (h1 : n = 51) (h2 : d = 3) : 
  (n - 1) * d = 150 := sorry

end NUMINAMATH_GPT_line_length_l779_77924


namespace NUMINAMATH_GPT_zero_descriptions_l779_77985

-- Defining the descriptions of zero satisfying the given conditions.
def description1 : String := "The number corresponding to the origin on the number line."
def description2 : String := "The number that represents nothing."
def description3 : String := "The number that, when multiplied by any other number, equals itself."

-- Lean statement to prove the validity of the descriptions.
theorem zero_descriptions : 
  description1 = "The number corresponding to the origin on the number line." ∧
  description2 = "The number that represents nothing." ∧
  description3 = "The number that, when multiplied by any other number, equals itself." :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_zero_descriptions_l779_77985


namespace NUMINAMATH_GPT_Matthias_fewer_fish_l779_77927

-- Define the number of fish Micah has
def Micah_fish : ℕ := 7

-- Define the number of fish Kenneth has
def Kenneth_fish : ℕ := 3 * Micah_fish

-- Define the total number of fish
def total_fish : ℕ := 34

-- Define the number of fish Matthias has
def Matthias_fish : ℕ := total_fish - (Micah_fish + Kenneth_fish)

-- State the theorem for the number of fewer fish Matthias has compared to Kenneth
theorem Matthias_fewer_fish : Kenneth_fish - Matthias_fish = 15 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Matthias_fewer_fish_l779_77927


namespace NUMINAMATH_GPT_remainder_of_sum_of_primes_is_eight_l779_77978

-- Define the first eight primes and their sum
def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sumFirstEightPrimes : ℕ := 77

-- Define the ninth prime
def ninthPrime : ℕ := 23

-- Theorem stating the equivalence
theorem remainder_of_sum_of_primes_is_eight :
  (sumFirstEightPrimes % ninthPrime) = 8 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_primes_is_eight_l779_77978


namespace NUMINAMATH_GPT_proposition_a_proposition_b_proposition_c_proposition_d_l779_77951

variable (a b c : ℝ)

-- Proposition A: If ac^2 > bc^2, then a > b
theorem proposition_a (h : a * c^2 > b * c^2) : a > b := sorry

-- Proposition B: If a > b, then ac^2 > bc^2
theorem proposition_b (h : a > b) : ¬ (a * c^2 > b * c^2) := sorry

-- Proposition C: If a > b, then 1/a < 1/b
theorem proposition_c (h : a > b) : ¬ (1/a < 1/b) := sorry

-- Proposition D: If a > b > 0, then a^2 > ab > b^2
theorem proposition_d (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 := sorry

end NUMINAMATH_GPT_proposition_a_proposition_b_proposition_c_proposition_d_l779_77951


namespace NUMINAMATH_GPT_find_starting_number_of_range_l779_77931

theorem find_starting_number_of_range :
  ∃ x, (∀ n, 0 ≤ n ∧ n < 10 → 65 - 5 * n = x + 5 * (9 - n)) ∧ x = 15 := 
by
  sorry

end NUMINAMATH_GPT_find_starting_number_of_range_l779_77931


namespace NUMINAMATH_GPT_sum_A_B_C_zero_l779_77947

noncomputable def poly : Polynomial ℝ := Polynomial.X^3 - 16 * Polynomial.X^2 + 72 * Polynomial.X - 27

noncomputable def exists_real_A_B_C 
  (p q r: ℝ) (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) :
  ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r)))) := sorry

theorem sum_A_B_C_zero 
  {p q r: ℝ} (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) 
  (hABC: ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))))) :
  ∀ A B C, A + B + C = 0 := sorry

end NUMINAMATH_GPT_sum_A_B_C_zero_l779_77947


namespace NUMINAMATH_GPT_g_zero_eq_zero_l779_77941

noncomputable def g : ℝ → ℝ :=
  sorry

axiom functional_equation (a b : ℝ) :
  g (3 * a + 2 * b) + g (3 * a - 2 * b) = 2 * g (3 * a) + 2 * g (2 * b)

theorem g_zero_eq_zero : g 0 = 0 :=
by
  let a := 0
  let b := 0
  have eqn := functional_equation a b
  sorry

end NUMINAMATH_GPT_g_zero_eq_zero_l779_77941


namespace NUMINAMATH_GPT_jordan_field_area_l779_77952

theorem jordan_field_area
  (s l : ℕ)
  (h1 : 2 * (s + l) = 24)
  (h2 : l + 1 = 2 * (s + 1)) :
  3 * s * 3 * l = 189 := 
by
  sorry

end NUMINAMATH_GPT_jordan_field_area_l779_77952


namespace NUMINAMATH_GPT_tangent_planes_of_surface_and_given_plane_l779_77949

-- Define the surface and the given plane
def surface (x y z : ℝ) := (x^2 + 4 * y^2 + 9 * z^2 = 1)
def given_plane (x y z : ℝ) := (x + y + 2 * z = 1)

-- Define the tangent plane equations to be proved
def tangent_plane_1 (x y z : ℝ) := (x + y + 2 * z - (109 / (6 * Real.sqrt 61)) = 0)
def tangent_plane_2 (x y z : ℝ) := (x + y + 2 * z + (109 / (6 * Real.sqrt 61)) = 0)

-- The statement to be proved
theorem tangent_planes_of_surface_and_given_plane :
  ∀ x y z, surface x y z ∧ given_plane x y z →
    tangent_plane_1 x y z ∨ tangent_plane_2 x y z :=
sorry

end NUMINAMATH_GPT_tangent_planes_of_surface_and_given_plane_l779_77949


namespace NUMINAMATH_GPT_sum_of_edges_l779_77974

-- Define the properties of the rectangular solid
variables (a b c : ℝ)
variables (V : ℝ) (S : ℝ)

-- Set the conditions
def geometric_progression := (a * b * c = V) ∧ (2 * (a * b + b * c + c * a) = S) ∧ (∃ k : ℝ, k ≠ 0 ∧ a = b / k ∧ c = b * k)

-- Define the main proof statement
theorem sum_of_edges (hV : V = 1000) (hS : S = 600) (hg : geometric_progression a b c V S) : 
  4 * (a + b + c) = 120 :=
sorry

end NUMINAMATH_GPT_sum_of_edges_l779_77974


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l779_77991

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 - 8 * x - 3 > 0 ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l779_77991


namespace NUMINAMATH_GPT_remainder_when_divided_l779_77965

theorem remainder_when_divided (P D Q R D'' Q'' R'' : ℕ) (h1 : P = Q * D + R) (h2 : Q = D'' * Q'' + R'') :
  P % (2 * D * D'') = D * R'' + R := sorry

end NUMINAMATH_GPT_remainder_when_divided_l779_77965


namespace NUMINAMATH_GPT_simplify_product_l779_77960

theorem simplify_product (x y : ℝ) : 
  (x - 3 * y + 2) * (x + 3 * y + 2) = (x^2 + 4 * x + 4 - 9 * y^2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_product_l779_77960


namespace NUMINAMATH_GPT_remaining_pencils_l779_77966

-- Define the initial conditions
def initial_pencils : Float := 56.0
def pencils_given : Float := 9.0

-- Formulate the theorem stating that the remaining pencils = 47.0
theorem remaining_pencils : initial_pencils - pencils_given = 47.0 := by
  sorry

end NUMINAMATH_GPT_remaining_pencils_l779_77966


namespace NUMINAMATH_GPT_reciprocal_of_one_fifth_l779_77944

theorem reciprocal_of_one_fifth : (∃ x : ℚ, (1/5) * x = 1 ∧ x = 5) :=
by
  -- The proof goes here, for now we assume it with sorry
  sorry

end NUMINAMATH_GPT_reciprocal_of_one_fifth_l779_77944


namespace NUMINAMATH_GPT_bens_car_costs_l779_77922

theorem bens_car_costs :
  (∃ C_old C_2nd : ℕ,
    (2 * C_old = 4 * C_2nd) ∧
    (C_old = 1800) ∧
    (C_2nd = 900) ∧
    (2 * C_old = 3600) ∧
    (4 * C_2nd = 3600) ∧
    (1800 + 900 = 2700) ∧
    (3600 - 2700 = 900) ∧
    (2000 - 900 = 1100) ∧
    (900 * 0.05 = 45) ∧
    (45 * 2 = 90))
  :=
sorry

end NUMINAMATH_GPT_bens_car_costs_l779_77922


namespace NUMINAMATH_GPT_mean_transformation_l779_77903

variable {x1 x2 x3 : ℝ}
variable (s : ℝ)
variable (h_var : s^2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12))

theorem mean_transformation :
  (x1 + 1 + x2 + 1 + x3 + 1) / 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_transformation_l779_77903


namespace NUMINAMATH_GPT_student_percentage_to_pass_l779_77918

/-- A student needs to obtain 50% of the total marks to pass given the conditions:
    1. The student got 200 marks.
    2. The student failed by 20 marks.
    3. The maximum marks are 440. -/
theorem student_percentage_to_pass : 
  ∀ (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ),
  student_marks = 200 → failed_by = 20 → max_marks = 440 →
  (student_marks + failed_by) / max_marks * 100 = 50 := 
by
  intros student_marks failed_by max_marks h1 h2 h3
  sorry

end NUMINAMATH_GPT_student_percentage_to_pass_l779_77918


namespace NUMINAMATH_GPT_efficiency_ratio_l779_77910

theorem efficiency_ratio (A B : ℝ) (h1 : A ≠ B)
  (h2 : A + B = 1 / 7)
  (h3 : B = 1 / 21) :
  A / B = 2 :=
by
  sorry

end NUMINAMATH_GPT_efficiency_ratio_l779_77910


namespace NUMINAMATH_GPT_regular_octahedron_vertices_count_l779_77912

def regular_octahedron_faces := 8
def regular_octahedron_edges := 12
def regular_octahedron_faces_shape := "equilateral triangle"
def regular_octahedron_vertices_meet := 4

theorem regular_octahedron_vertices_count :
  ∀ (F E V : ℕ),
    F = regular_octahedron_faces →
    E = regular_octahedron_edges →
    (∀ (v : ℕ), v = regular_octahedron_vertices_meet) →
    V = 6 :=
by
  intros F E V hF hE hV
  sorry

end NUMINAMATH_GPT_regular_octahedron_vertices_count_l779_77912


namespace NUMINAMATH_GPT_line_eq_x_1_parallel_y_axis_l779_77905

theorem line_eq_x_1_parallel_y_axis (P : ℝ × ℝ) (hP : P = (1, 0)) (h_parallel : ∀ y : ℝ, (1, y) = P ∨ P = (1, y)) :
  ∃ x : ℝ, (∀ y : ℝ, P = (x, y)) → x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_line_eq_x_1_parallel_y_axis_l779_77905


namespace NUMINAMATH_GPT_even_expressions_l779_77988

theorem even_expressions (x y : ℕ) (hx : Even x) (hy : Even y) :
  Even (x + 5 * y) ∧
  Even (4 * x - 3 * y) ∧
  Even (2 * x^2 + 5 * y^2) ∧
  Even ((2 * x * y + 4)^2) ∧
  Even (4 * x * y) :=
by
  sorry

end NUMINAMATH_GPT_even_expressions_l779_77988


namespace NUMINAMATH_GPT_regular_price_per_can_l779_77937

variable (P : ℝ) -- Regular price per can

-- Condition: The regular price per can is discounted 15 percent when the soda is purchased in 24-can cases
def discountedPricePerCan (P : ℝ) : ℝ :=
  0.85 * P

-- Condition: The price of 72 cans purchased in 24-can cases is $18.36
def priceOf72CansInDollars : ℝ :=
  18.36

-- Predicate describing the condition that the price of 72 cans is 18.36
axiom h : (72 * discountedPricePerCan P) = priceOf72CansInDollars

theorem regular_price_per_can (P : ℝ) (h : (72 * discountedPricePerCan P) = priceOf72CansInDollars) : P = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_regular_price_per_can_l779_77937


namespace NUMINAMATH_GPT_composite_function_evaluation_l779_77926

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem composite_function_evaluation : f (g 3) = 25 := by
  sorry

end NUMINAMATH_GPT_composite_function_evaluation_l779_77926


namespace NUMINAMATH_GPT_problem_statement_l779_77932

-- We begin by stating the variables x and y with the given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : x - 2 * y = 3
axiom h2 : (x - 2) * (y + 1) = 2

-- The theorem to prove
theorem problem_statement : (x^2 - 2) * (2 * y^2 - 1) = -9 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l779_77932


namespace NUMINAMATH_GPT_no_integers_abc_for_polynomial_divisible_by_9_l779_77904

theorem no_integers_abc_for_polynomial_divisible_by_9 :
  ¬ ∃ (a b c : ℤ), ∀ x : ℤ, 9 ∣ (x + a) * (x + b) * (x + c) - x ^ 3 - 1 :=
by
  sorry

end NUMINAMATH_GPT_no_integers_abc_for_polynomial_divisible_by_9_l779_77904


namespace NUMINAMATH_GPT_fraction_zero_implies_x_is_minus_5_l779_77955

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_is_minus_5_l779_77955


namespace NUMINAMATH_GPT_sum_coeff_eq_neg_two_l779_77916

theorem sum_coeff_eq_neg_two (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ) :
  (1 - 2*x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_coeff_eq_neg_two_l779_77916


namespace NUMINAMATH_GPT_walking_time_difference_at_slower_speed_l779_77945

theorem walking_time_difference_at_slower_speed (T : ℕ) (v_s: ℚ) (h1: T = 32) (h2: v_s = 4/5) : 
  (T * (5/4) - T) = 8 :=
by
  sorry

end NUMINAMATH_GPT_walking_time_difference_at_slower_speed_l779_77945


namespace NUMINAMATH_GPT_buyers_cake_and_muffin_l779_77902

theorem buyers_cake_and_muffin (total_buyers cake_buyers muffin_buyers neither_prob : ℕ) :
  total_buyers = 100 →
  cake_buyers = 50 →
  muffin_buyers = 40 →
  neither_prob = 26 →
  (cake_buyers + muffin_buyers - neither_prob) = 74 →
  90 - cake_buyers - muffin_buyers = neither_prob :=
by
  sorry

end NUMINAMATH_GPT_buyers_cake_and_muffin_l779_77902


namespace NUMINAMATH_GPT_wire_cutting_l779_77953

theorem wire_cutting : 
  ∃ (n : ℕ), n = 33 ∧ (∀ (x y : ℕ), 3 * x + y = 100 → x > 0 ∧ y > 0 → ∃ m : ℕ, m = n) :=
by {
  sorry
}

end NUMINAMATH_GPT_wire_cutting_l779_77953


namespace NUMINAMATH_GPT_find_abc_sum_l779_77957

theorem find_abc_sum :
  ∃ (a b c : ℤ), 2 * a + 3 * b = 52 ∧ 3 * b + c = 41 ∧ b * c = 60 ∧ a + b + c = 25 :=
by
  use 8, 12, 5
  sorry

end NUMINAMATH_GPT_find_abc_sum_l779_77957


namespace NUMINAMATH_GPT_probability_six_distinct_numbers_l779_77928

theorem probability_six_distinct_numbers :
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  probability = (35 / 648) := 
by
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  have h : favorable_outcomes = 15120 := by sorry
  have h2 : outcomes = 279936 := by sorry
  have prob : probability = (15120 / 279936) := by sorry
  have gcd_calc : gcd 15120 279936 = 432 := by sorry
  have simplified_prob : (15120 / 279936) = (35 / 648) := by sorry
  exact simplified_prob

end NUMINAMATH_GPT_probability_six_distinct_numbers_l779_77928


namespace NUMINAMATH_GPT_factorize_negative_quadratic_l779_77906

theorem factorize_negative_quadratic (x y : ℝ) : 
  -4 * x^2 + y^2 = (y - 2 * x) * (y + 2 * x) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_negative_quadratic_l779_77906


namespace NUMINAMATH_GPT_equivalent_resistance_is_15_l779_77963

-- Definitions based on conditions
def R : ℝ := 5 -- Resistance of each resistor in Ohms
def num_resistors : ℕ := 4

-- The equivalent resistance due to the short-circuit path removing one resistor
def simplified_circuit_resistance : ℝ := (num_resistors - 1) * R

-- The statement to prove
theorem equivalent_resistance_is_15 :
  simplified_circuit_resistance = 15 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_resistance_is_15_l779_77963


namespace NUMINAMATH_GPT_solve_equation_l779_77992

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end NUMINAMATH_GPT_solve_equation_l779_77992


namespace NUMINAMATH_GPT_garden_area_increase_l779_77993

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_increase_l779_77993


namespace NUMINAMATH_GPT_find_four_digit_number_l779_77938

theorem find_four_digit_number :
  ∃ (N : ℕ), 1000 ≤ N ∧ N < 10000 ∧ 
    (N % 131 = 112) ∧ 
    (N % 132 = 98) ∧ 
    N = 1946 :=
by
  sorry

end NUMINAMATH_GPT_find_four_digit_number_l779_77938


namespace NUMINAMATH_GPT_square_perimeter_is_44_8_l779_77996

noncomputable def perimeter_of_congruent_rectangles_division (s : ℝ) (P : ℝ) : ℝ :=
  let rectangle_perimeter := 2 * (s + s / 4)
  if rectangle_perimeter = P then 4 * s else 0

theorem square_perimeter_is_44_8 :
  ∀ (s : ℝ) (P : ℝ), P = 28 → 4 * s = 44.8 → perimeter_of_congruent_rectangles_division s P = 44.8 :=
by intros s P h1 h2
   sorry

end NUMINAMATH_GPT_square_perimeter_is_44_8_l779_77996


namespace NUMINAMATH_GPT_black_balls_probability_both_black_l779_77998

theorem black_balls_probability_both_black (balls_total balls_black balls_gold : ℕ) (prob : ℚ) 
  (h1 : balls_total = 11)
  (h2 : balls_black = 7)
  (h3 : balls_gold = 4)
  (h4 : balls_total = balls_black + balls_gold)
  (h5 : prob = (21 : ℚ) / 55) :
  balls_total.choose 2 * prob = balls_black.choose 2 :=
sorry

end NUMINAMATH_GPT_black_balls_probability_both_black_l779_77998


namespace NUMINAMATH_GPT_trajectory_of_Q_l779_77929

variables {P Q M : ℝ × ℝ}

-- Define the conditions as Lean predicates
def is_midpoint (M P Q : ℝ × ℝ) : Prop :=
  M = (0, 4) ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 2 = 0

-- Define the theorem that needs to be proven
theorem trajectory_of_Q :
  (∃ P Q M : ℝ × ℝ, is_midpoint M P Q ∧ point_on_line P) →
  ∃ Q : ℝ × ℝ, (∀ P : ℝ × ℝ, point_on_line P → is_midpoint (0,4) P Q → Q.1 + Q.2 - 6 = 0) :=
by sorry

end NUMINAMATH_GPT_trajectory_of_Q_l779_77929


namespace NUMINAMATH_GPT_matrix_multiplication_correct_l779_77970

-- Define the matrices
def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, 0, -3],
    ![1, 3, -2],
    ![0, 2, 4]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![1, -1, 0],
    ![0, 2, -1],
    ![3, 0, 1]
  ]

def C : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![-7, -2, -3],
    ![-5, 5, -5],
    ![12, 4, 2]
  ]

-- Proof statement that multiplication of A and B gives C
theorem matrix_multiplication_correct : A * B = C := 
by
  sorry

end NUMINAMATH_GPT_matrix_multiplication_correct_l779_77970


namespace NUMINAMATH_GPT_arcsin_inequality_l779_77980

theorem arcsin_inequality (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  (Real.arcsin x + Real.arcsin y > Real.pi / 2) ↔ (x ≥ 0 ∧ y ≥ 0 ∧ (y^2 + x^2 > 1)) := by
sorry

end NUMINAMATH_GPT_arcsin_inequality_l779_77980


namespace NUMINAMATH_GPT_relationship_between_M_and_N_l779_77976

variable (x y : ℝ)

theorem relationship_between_M_and_N (h1 : x ≠ 3) (h2 : y ≠ -2)
  (M : ℝ) (hm : M = x^2 + y^2 - 6 * x + 4 * y)
  (N : ℝ) (hn : N = -13) : M > N :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_M_and_N_l779_77976


namespace NUMINAMATH_GPT_factorize_cubed_sub_four_l779_77994

theorem factorize_cubed_sub_four (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_cubed_sub_four_l779_77994


namespace NUMINAMATH_GPT_max_zoo_area_l779_77919

theorem max_zoo_area (length width x y : ℝ) (h1 : length = 16) (h2 : width = 8 - x) (h3 : y = x * (8 - x)) : 
  ∃ M, ∀ x, 0 < x ∧ x < 8 → y ≤ M ∧ M = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_zoo_area_l779_77919


namespace NUMINAMATH_GPT_inequality1_inequality2_l779_77933

noncomputable def f (x : ℝ) := abs (x + 1 / 2) + abs (x - 3 / 2)

theorem inequality1 (x : ℝ) : 
  (f x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 2) := by
sorry

theorem inequality2 (a : ℝ) :
  (∀ x, f x ≥ 1 / 2 * abs (1 - a)) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end NUMINAMATH_GPT_inequality1_inequality2_l779_77933
