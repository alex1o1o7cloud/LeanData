import Mathlib

namespace NUMINAMATH_GPT_parabola_hyperbola_focus_l666_66638

theorem parabola_hyperbola_focus (p : ℝ) :
  let parabolaFocus := (p / 2, 0)
  let hyperbolaRightFocus := (2, 0)
  (parabolaFocus = hyperbolaRightFocus) → p = 4 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_l666_66638


namespace NUMINAMATH_GPT_chef_dressing_total_volume_l666_66650

theorem chef_dressing_total_volume :
  ∀ (V1 V2 : ℕ) (P1 P2 : ℕ) (total_amount : ℕ),
    V1 = 128 →
    V2 = 128 →
    P1 = 8 →
    P2 = 13 →
    total_amount = V1 + V2 →
    total_amount = 256 :=
by
  intros V1 V2 P1 P2 total_amount hV1 hV2 hP1 hP2 h_total
  rw [hV1, hV2, add_comm, add_comm] at h_total
  exact h_total

end NUMINAMATH_GPT_chef_dressing_total_volume_l666_66650


namespace NUMINAMATH_GPT_John_distance_proof_l666_66630

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_John_distance_proof_l666_66630


namespace NUMINAMATH_GPT_radian_measure_of_240_degrees_l666_66600

theorem radian_measure_of_240_degrees : (240 * (π / 180) = 4 * π / 3) := by
  sorry

end NUMINAMATH_GPT_radian_measure_of_240_degrees_l666_66600


namespace NUMINAMATH_GPT_find_f_80_l666_66645

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_relation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  f (x * y) = f x / y^2

axiom f_40 : f 40 = 50

-- Proof that f 80 = 12.5
theorem find_f_80 : f 80 = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_find_f_80_l666_66645


namespace NUMINAMATH_GPT_simplify_expression_l666_66678

variable {x y z : ℝ}

theorem simplify_expression (h : x^2 - y^2 ≠ 0) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l666_66678


namespace NUMINAMATH_GPT_max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l666_66643

namespace Geometry

variables {x y : ℝ}

-- Given condition
def satisfies_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * y + 1 = 0

-- Proof problems
theorem max_x_plus_y (h : satisfies_circle x y) : 
  x + y ≤ 2 + Real.sqrt 6 :=
sorry

theorem range_y_plus_1_over_x (h : satisfies_circle x y) : 
  -Real.sqrt 2 ≤ (y + 1) / x ∧ (y + 1) / x ≤ Real.sqrt 2 :=
sorry

theorem extrema_x2_minus_2x_plus_y2_plus_1 (h : satisfies_circle x y) : 
  8 - 2 * Real.sqrt 15 ≤ x^2 - 2 * x + y^2 + 1 ∧ x^2 - 2 * x + y^2 + 1 ≤ 8 + 2 * Real.sqrt 15 :=
sorry

end Geometry

end NUMINAMATH_GPT_max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l666_66643


namespace NUMINAMATH_GPT_appropriate_sampling_method_l666_66696

-- Definitions and conditions
def total_products : ℕ := 40
def first_class_products : ℕ := 10
def second_class_products : ℕ := 25
def defective_products : ℕ := 5
def samples_needed : ℕ := 8

-- Theorem statement
theorem appropriate_sampling_method : 
  (first_class_products + second_class_products + defective_products = total_products) ∧ 
  (2 ≤ first_class_products ∧ 2 ≤ second_class_products ∧ 1 ≤ defective_products) → 
  "Stratified Sampling" = "The appropriate sampling method for quality analysis" :=
  sorry

end NUMINAMATH_GPT_appropriate_sampling_method_l666_66696


namespace NUMINAMATH_GPT_trapezoid_upper_side_length_l666_66629

theorem trapezoid_upper_side_length (area base1 height : ℝ) (h1 : area = 222) (h2 : base1 = 23) (h3 : height = 12) : 
  ∃ base2, base2 = 14 :=
by
  -- The proof will be provided here.
  sorry

end NUMINAMATH_GPT_trapezoid_upper_side_length_l666_66629


namespace NUMINAMATH_GPT_minimum_number_of_guests_l666_66607

theorem minimum_number_of_guests :
  ∀ (total_food : ℝ) (max_food_per_guest : ℝ), total_food = 411 → max_food_per_guest = 2.5 →
  ⌈total_food / max_food_per_guest⌉ = 165 :=
by
  intros total_food max_food_per_guest h1 h2
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_minimum_number_of_guests_l666_66607


namespace NUMINAMATH_GPT_steak_entree_cost_l666_66683

theorem steak_entree_cost
  (total_guests : ℕ)
  (steak_factor : ℕ)
  (chicken_entree_cost : ℕ)
  (total_budget : ℕ)
  (H1 : total_guests = 80)
  (H2 : steak_factor = 3)
  (H3 : chicken_entree_cost = 18)
  (H4 : total_budget = 1860) :
  ∃ S : ℕ, S = 25 := by
  -- Proof steps omitted
  sorry

end NUMINAMATH_GPT_steak_entree_cost_l666_66683


namespace NUMINAMATH_GPT_sum_of_numbers_l666_66666

noncomputable def mean (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem sum_of_numbers (a b c : ℕ) (h1 : mean a b c = a + 8)
  (h2 : mean a b c = c - 20) (h3 : b = 7) (h_le1 : a ≤ b) (h_le2 : b ≤ c) :
  a + b + c = 57 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_numbers_l666_66666


namespace NUMINAMATH_GPT_polynomial_multiplication_l666_66679

theorem polynomial_multiplication :
  (5 * X^2 + 3 * X - 4) * (2 * X^3 + X^2 - X + 1) = 
  (10 * X^5 + 11 * X^4 - 10 * X^3 - 2 * X^2 + 7 * X - 4) := 
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_multiplication_l666_66679


namespace NUMINAMATH_GPT_find_y_l666_66685

theorem find_y (x y : ℕ) (h1 : x % y = 7) (h2 : (x : ℚ) / y = 86.1) (h3 : Nat.Prime (x + y)) : y = 70 :=
sorry

end NUMINAMATH_GPT_find_y_l666_66685


namespace NUMINAMATH_GPT_average_first_21_multiples_of_17_l666_66689

theorem average_first_21_multiples_of_17:
  let n := 21
  let a1 := 17
  let a21 := 17 * n
  let sum := n / 2 * (a1 + a21)
  (sum / n = 187) :=
by
  sorry

end NUMINAMATH_GPT_average_first_21_multiples_of_17_l666_66689


namespace NUMINAMATH_GPT_Sharik_cannot_eat_all_meatballs_within_one_million_flies_l666_66622

theorem Sharik_cannot_eat_all_meatballs_within_one_million_flies:
  (∀ n: ℕ, ∃ i: ℕ, i > n ∧ ((∀ j < i, ∀ k: ℕ, ∃ m: ℕ, (m ≠ k) → (∃ f, f < 10^6) )) → f > 10^6 ) :=
sorry

end NUMINAMATH_GPT_Sharik_cannot_eat_all_meatballs_within_one_million_flies_l666_66622


namespace NUMINAMATH_GPT_johns_total_payment_l666_66639

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end NUMINAMATH_GPT_johns_total_payment_l666_66639


namespace NUMINAMATH_GPT_circles_intersect_l666_66611

theorem circles_intersect :
  ∀ (x y : ℝ),
    ((x^2 + y^2 - 2 * x + 4 * y + 1 = 0) →
    (x^2 + y^2 - 6 * x + 2 * y + 9 = 0) →
    (∃ c1 c2 r1 r2 d : ℝ,
      (x - 1)^2 + (y + 2)^2 = r1 ∧ r1 = 4 ∧
      (x - 3)^2 + (y + 1)^2 = r2 ∧ r2 = 1 ∧
      d = Real.sqrt ((3 - 1)^2 + (-1 + 2)^2) ∧
      d > abs (r1 - r2) ∧ d < (r1 + r2))) :=
sorry

end NUMINAMATH_GPT_circles_intersect_l666_66611


namespace NUMINAMATH_GPT_sealed_envelope_problem_l666_66680

theorem sealed_envelope_problem :
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) →
  ((n = 12 ∧ (n % 10 ≠ 2) ∧ n ≠ 35 ∧ (n % 10 ≠ 5)) ∨
   (n ≠ 12 ∧ (n % 10 ≠ 2) ∧ n = 35 ∧ (n % 10 = 5))) →
  ¬(n % 10 ≠ 5) :=
by
  sorry

end NUMINAMATH_GPT_sealed_envelope_problem_l666_66680


namespace NUMINAMATH_GPT_probability_of_friends_in_same_lunch_group_l666_66608

theorem probability_of_friends_in_same_lunch_group :
  let groups := 4
  let students := 720
  let group_size := students / groups
  let probability := (1 / groups) * (1 / groups) * (1 / groups)
  students % groups = 0 ->  -- Students can be evenly divided into groups
  groups > 0 ->             -- There is at least one group
  probability = (1 : ℝ) / 64 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_of_friends_in_same_lunch_group_l666_66608


namespace NUMINAMATH_GPT_felicity_gasoline_usage_l666_66659

def gallons_of_gasoline (G D: ℝ) :=
  G = 2 * D

def combined_volume (M D: ℝ) :=
  M = D - 5

def ethanol_consumption (E M: ℝ) :=
  E = 0.35 * M

def biodiesel_consumption (B M: ℝ) :=
  B = 0.65 * M

def distance_relationship_F_A (F A: ℕ) :=
  A = F + 150

def distance_relationship_F_Bn (F Bn: ℕ) :=
  F = Bn + 50

def total_distance (F A Bn: ℕ) :=
  F + A + Bn = 1750

def gasoline_mileage : ℕ := 35

def diesel_mileage : ℕ := 25

def ethanol_mileage : ℕ := 30

def biodiesel_mileage : ℕ := 20

theorem felicity_gasoline_usage : 
  ∀ (F A Bn: ℕ) (G D M E B: ℝ),
  gallons_of_gasoline G D →
  combined_volume M D →
  ethanol_consumption E M →
  biodiesel_consumption B M →
  distance_relationship_F_A F A →
  distance_relationship_F_Bn F Bn →
  total_distance F A Bn →
  G = 56
  := by
    intros
    sorry

end NUMINAMATH_GPT_felicity_gasoline_usage_l666_66659


namespace NUMINAMATH_GPT_continuous_iff_integral_condition_l666_66688

open Real 

noncomputable section

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def integral_condition (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (∫ x in a..(a + a_seq n), f x) + (∫ x in (a - a_seq n)..a, f x) ≤ (a_seq n) / n

theorem continuous_iff_integral_condition (a : ℝ) (f : ℝ → ℝ)
  (h_nondec : is_non_decreasing f) :
  ContinuousAt f a ↔ ∃ (a_seq : ℕ → ℝ), (∀ n, 0 < a_seq n) ∧ integral_condition f a a_seq := sorry

end NUMINAMATH_GPT_continuous_iff_integral_condition_l666_66688


namespace NUMINAMATH_GPT_units_digit_G1000_l666_66612

def units_digit (n : ℕ) : ℕ :=
  n % 10

def power_cycle : List ℕ := [3, 9, 7, 1]

def G (n : ℕ) : ℕ :=
  3^(2^n) + 2

theorem units_digit_G1000 : units_digit (G 1000) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_G1000_l666_66612


namespace NUMINAMATH_GPT_product_of_consecutive_integers_even_l666_66695

theorem product_of_consecutive_integers_even (n : ℤ) : Even (n * (n + 1)) :=
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_even_l666_66695


namespace NUMINAMATH_GPT_new_area_of_rectangle_l666_66644

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 600) :
  let new_length := 0.8 * L
  let new_width := 1.05 * W
  new_length * new_width = 504 :=
by 
  sorry

end NUMINAMATH_GPT_new_area_of_rectangle_l666_66644


namespace NUMINAMATH_GPT_rectangle_area_l666_66667

theorem rectangle_area (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x * y = 5 :=
by
  -- Conditions given to us:
  -- 1. (h1) The sum of the sides is 5.
  -- 2. (h2) The sum of the squares of the sides is 15.
  -- We need to prove that the product of the sides is 5.
  sorry

end NUMINAMATH_GPT_rectangle_area_l666_66667


namespace NUMINAMATH_GPT_average_words_per_hour_l666_66697

/-- Prove that given a total of 50,000 words written in 100 hours with the 
writing output increasing by 10% each subsequent hour, the average number 
of words written per hour is 500. -/
theorem average_words_per_hour 
(words_total : ℕ) 
(hours_total : ℕ) 
(increase : ℝ) :
  words_total = 50000 ∧ hours_total = 100 ∧ increase = 0.1 →
  (words_total / hours_total : ℝ) = 500 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_average_words_per_hour_l666_66697


namespace NUMINAMATH_GPT_average_score_l666_66686

variable (score : Fin 5 → ℤ)
variable (actual_score : ℤ)
variable (rank : Fin 5)
variable (average : ℤ)

def students_scores_conditions := 
  score 0 = 10 ∧ score 1 = -5 ∧ score 2 = 0 ∧ score 3 = 8 ∧ score 4 = -3 ∧
  actual_score = 90 ∧ rank.val = 2

theorem average_score (h : students_scores_conditions score actual_score rank) :
  average = 92 :=
sorry

end NUMINAMATH_GPT_average_score_l666_66686


namespace NUMINAMATH_GPT_half_angle_in_first_quadrant_l666_66632

theorem half_angle_in_first_quadrant {α : ℝ} (h : 0 < α ∧ α < π / 2) : 
  0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end NUMINAMATH_GPT_half_angle_in_first_quadrant_l666_66632


namespace NUMINAMATH_GPT__l666_66626

noncomputable def t_value_theorem (a b x d t y : ℕ) (h1 : a + b = x) (h2 : x + d = t) (h3 : t + a = y) (h4 : b + d + y = 16) : t = 8 :=
by sorry

end NUMINAMATH_GPT__l666_66626


namespace NUMINAMATH_GPT_water_left_in_bucket_l666_66623

theorem water_left_in_bucket :
  ∀ (original_poured water_left : ℝ),
    original_poured = 0.8 →
    water_left = 0.6 →
    ∃ (poured : ℝ), poured = 0.2 ∧ original_poured - poured = water_left :=
by
  intros original_poured water_left ho hw
  apply Exists.intro 0.2
  simp [ho, hw]
  sorry

end NUMINAMATH_GPT_water_left_in_bucket_l666_66623


namespace NUMINAMATH_GPT_draw_probability_l666_66618

variable (P_lose_a win_a : ℝ)
variable (not_lose_a : ℝ := 0.8)
variable (win_prob_a : ℝ := 0.6)

-- Given conditions
def A_not_losing : Prop := not_lose_a = win_prob_a + win_a

-- Main theorem to prove
theorem draw_probability : P_lose_a = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_draw_probability_l666_66618


namespace NUMINAMATH_GPT_population_of_missing_village_l666_66641

theorem population_of_missing_village 
  (p1 p2 p3 p4 p5 p6 : ℕ) 
  (h1 : p1 = 803) 
  (h2 : p2 = 900) 
  (h3 : p3 = 1100) 
  (h4 : p4 = 1023) 
  (h5 : p5 = 945) 
  (h6 : p6 = 1249) 
  (avg_population : ℕ) 
  (h_avg : avg_population = 1000) :
  ∃ p7 : ℕ, p7 = 980 ∧ avg_population * 7 = p1 + p2 + p3 + p4 + p5 + p6 + p7 :=
by
  sorry

end NUMINAMATH_GPT_population_of_missing_village_l666_66641


namespace NUMINAMATH_GPT_fractional_eq_solve_simplify_and_evaluate_l666_66647

-- Question 1: Solve the fractional equation
theorem fractional_eq_solve (x : ℝ) (h1 : (x / (x + 1) = (2 * x) / (3 * x + 3) + 1)) : 
  x = -1.5 := 
sorry

-- Question 2: Simplify and evaluate the expression for x = -1
theorem simplify_and_evaluate (x : ℝ)
  (h2 : x ≠ 0) (h3 : x ≠ 2) (h4 : x ≠ -2) :
  (x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4) / ((x+2) / (x^3 - 4*x)) = 
  (x - 4) / (x - 2) ∧ 
  (x = -1) → ((x - 4) / (x - 2) = (5 / 3)) := 
sorry

end NUMINAMATH_GPT_fractional_eq_solve_simplify_and_evaluate_l666_66647


namespace NUMINAMATH_GPT_shakes_sold_l666_66620

variable (s : ℕ) -- the number of shakes sold

-- conditions
def shakes_ounces := 4 * s
def cone_ounces := 6
def total_ounces := 14

-- the theorem to prove
theorem shakes_sold : shakes_ounces + cone_ounces = total_ounces → s = 2 := by
  intros h
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_shakes_sold_l666_66620


namespace NUMINAMATH_GPT_prove_river_improvement_l666_66676

def river_improvement_equation (x : ℝ) : Prop :=
  4800 / x - 4800 / (x + 200) = 4

theorem prove_river_improvement (x : ℝ) (h : x > 0) : river_improvement_equation x := by
  sorry

end NUMINAMATH_GPT_prove_river_improvement_l666_66676


namespace NUMINAMATH_GPT_bike_ride_time_l666_66614

theorem bike_ride_time (y : ℚ) : 
  let speed_fast := 25
  let speed_slow := 10
  let total_distance := 170
  let total_time := 10
  (speed_fast * y + speed_slow * (total_time - y) = total_distance) 
  → y = 14 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_bike_ride_time_l666_66614


namespace NUMINAMATH_GPT_rtl_to_conventional_notation_l666_66604

theorem rtl_to_conventional_notation (a b c d e : ℚ) :
  (a / (b - (c * (d + e)))) = a / (b - c * (d + e)) := by
  sorry

end NUMINAMATH_GPT_rtl_to_conventional_notation_l666_66604


namespace NUMINAMATH_GPT_total_digits_in_numbering_pages_l666_66658

theorem total_digits_in_numbering_pages (n : ℕ) (h : n = 100000) : 
  let digits1 := 9 * 1
  let digits2 := (99 - 10 + 1) * 2
  let digits3 := (999 - 100 + 1) * 3
  let digits4 := (9999 - 1000 + 1) * 4
  let digits5 := (99999 - 10000 + 1) * 5
  let digits6 := 6
  (digits1 + digits2 + digits3 + digits4 + digits5 + digits6) = 488895 :=
by
  sorry

end NUMINAMATH_GPT_total_digits_in_numbering_pages_l666_66658


namespace NUMINAMATH_GPT_amount_received_from_mom_l666_66653

-- Defining the problem conditions
def receives_from_dad : ℕ := 5
def spends : ℕ := 4
def has_more_from_mom_after_spending (M : ℕ) : Prop := 
  (receives_from_dad + M - spends = receives_from_dad + 2)

-- Lean theorem statement
theorem amount_received_from_mom (M : ℕ) (h : has_more_from_mom_after_spending M) : M = 6 := 
by
  sorry

end NUMINAMATH_GPT_amount_received_from_mom_l666_66653


namespace NUMINAMATH_GPT_cos_pi_over_3_plus_2theta_l666_66627

theorem cos_pi_over_3_plus_2theta 
  (theta : ℝ)
  (h : Real.sin (Real.pi / 3 - theta) = 3 / 4) : 
  Real.cos (Real.pi / 3 + 2 * theta) = 1 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_cos_pi_over_3_plus_2theta_l666_66627


namespace NUMINAMATH_GPT_ratio_rate_down_to_up_l666_66642

noncomputable def rate_up (r_up t_up: ℕ) : ℕ := r_up * t_up
noncomputable def rate_down (d_down t_down: ℕ) : ℕ := d_down / t_down
noncomputable def ratio (r_down r_up: ℕ) : ℚ := r_down / r_up

theorem ratio_rate_down_to_up :
  let r_up := 6
  let t_up := 2
  let d_down := 18
  let t_down := 2
  rate_up 6 2 = 12 ∧ rate_down 18 2 = 9 ∧ ratio 9 6 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_rate_down_to_up_l666_66642


namespace NUMINAMATH_GPT_beads_to_remove_l666_66616

-- Definitions for the conditions given in the problem
def initial_blue_beads : Nat := 49
def initial_red_bead : Nat := 1
def total_initial_beads : Nat := initial_blue_beads + initial_red_bead
def target_blue_percentage : Nat := 90 -- percentage

-- The goal to prove
theorem beads_to_remove (initial_blue_beads : Nat) (initial_red_bead : Nat)
    (target_blue_percentage : Nat) : Nat :=
    let target_total_beads := (initial_red_bead * 100) / target_blue_percentage
    total_initial_beads - target_total_beads
-- Expected: beads_to_remove 49 1 90 = 40

example : beads_to_remove initial_blue_beads initial_red_bead target_blue_percentage = 40 := by 
    sorry

end NUMINAMATH_GPT_beads_to_remove_l666_66616


namespace NUMINAMATH_GPT_find_y_l666_66615

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hrem : x % y = 5) (hdiv : (x : ℝ) / y = 96.2) : y = 25 := by
  sorry

end NUMINAMATH_GPT_find_y_l666_66615


namespace NUMINAMATH_GPT_find_x_values_l666_66682

noncomputable def tan_inv := Real.arctan (Real.sqrt 3 / 2)

theorem find_x_values (x : ℝ) :
  (-Real.pi < x ∧ x ≤ Real.pi) ∧ (2 * Real.tan x - Real.sqrt 3 = 0) ↔
  (x = tan_inv ∨ x = tan_inv - Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l666_66682


namespace NUMINAMATH_GPT_tip_calculation_correct_l666_66669

noncomputable def calculate_tip (total_with_tax : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let bill_before_tax := total_with_tax / (1 + tax_rate)
  bill_before_tax * tip_rate

theorem tip_calculation_correct :
  calculate_tip 226 0.13 0.15 = 30 := 
by
  sorry

end NUMINAMATH_GPT_tip_calculation_correct_l666_66669


namespace NUMINAMATH_GPT_mb_range_l666_66677
-- Define the slope m and y-intercept b
def m : ℚ := 2 / 3
def b : ℚ := -1 / 2

-- Define the product mb
def mb : ℚ := m * b

-- Prove the range of mb
theorem mb_range : -1 < mb ∧ mb < 0 := by
  unfold mb
  sorry

end NUMINAMATH_GPT_mb_range_l666_66677


namespace NUMINAMATH_GPT_find_g_of_conditions_l666_66663

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end NUMINAMATH_GPT_find_g_of_conditions_l666_66663


namespace NUMINAMATH_GPT_initial_sheep_count_l666_66655

theorem initial_sheep_count 
    (S : ℕ)
    (initial_horses : ℕ := 100)
    (initial_chickens : ℕ := 9)
    (gifted_goats : ℕ := 37)
    (male_animals : ℕ := 53)
    (total_animals_half : ℕ := 106) :
    ((initial_horses + S + initial_chickens) / 2 + gifted_goats = total_animals_half) → 
    S = 29 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_sheep_count_l666_66655


namespace NUMINAMATH_GPT_sum_base9_to_base9_eq_l666_66646

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_base9_to_base9_eq_l666_66646


namespace NUMINAMATH_GPT_not_function_age_height_l666_66617

theorem not_function_age_height (f : ℕ → ℝ) :
  ¬(∀ (a b : ℕ), a = b → f a = f b) := sorry

end NUMINAMATH_GPT_not_function_age_height_l666_66617


namespace NUMINAMATH_GPT_area_of_triangle_ABF_l666_66635

theorem area_of_triangle_ABF (A B F : ℝ × ℝ) (hF : F = (1, 0)) (hA_parabola : A.2^2 = 4 * A.1) (hB_parabola : B.2^2 = 4 * B.1) (h_midpoint_AB : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) : 
  ∃ area : ℝ, area = 2 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABF_l666_66635


namespace NUMINAMATH_GPT_geometric_sequence_properties_l666_66631

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ) (h : r ≠ 0)
  (h1 : a = r * (-1))
  (h2 : b = r * a)
  (h3 : c = r * b)
  (h4 : -9 = r * c) :
  b = -3 ∧ a * c = 9 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l666_66631


namespace NUMINAMATH_GPT_value_of_expression_l666_66681

theorem value_of_expression (x y : ℕ) (h₁ : x = 12) (h₂ : y = 7) : (x - y) * (x + y) = 95 := by
  -- Here we assume all necessary conditions as given:
  -- x = 12 and y = 7
  -- and we prove that (x - y)(x + y) = 95
  sorry

end NUMINAMATH_GPT_value_of_expression_l666_66681


namespace NUMINAMATH_GPT_Carl_typing_words_l666_66668

variable (typingSpeed : ℕ) (hoursPerDay : ℕ) (days : ℕ)

theorem Carl_typing_words (h1 : typingSpeed = 50) (h2 : hoursPerDay = 4) (h3 : days = 7) :
  (typingSpeed * 60 * hoursPerDay * days) = 84000 := by
  sorry

end NUMINAMATH_GPT_Carl_typing_words_l666_66668


namespace NUMINAMATH_GPT_stream_speed_l666_66656

variables (v_s t_d t_u : ℝ)
variables (D : ℝ) -- Distance is not provided in the problem but assumed for formulation.

theorem stream_speed (h1 : t_u = 2 * t_d) (h2 : v_s = 54 + t_d / t_u) :
  v_s = 18 := 
by
  sorry

end NUMINAMATH_GPT_stream_speed_l666_66656


namespace NUMINAMATH_GPT_S_8_arithmetic_sequence_l666_66649

theorem S_8_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : a 4 = 18 - a 5):
  S 8 = 72 :=
by
  sorry

end NUMINAMATH_GPT_S_8_arithmetic_sequence_l666_66649


namespace NUMINAMATH_GPT_base3_sum_l666_66633

theorem base3_sum : 
  (1 * 3^0 - 2 * 3^1 - 2 * 3^0 + 2 * 3^2 + 1 * 3^1 - 1 * 3^0 - 1 * 3^3) = (2 * 3^2 + 1 * 3^1 + 0 * 3^0) := 
by 
  sorry

end NUMINAMATH_GPT_base3_sum_l666_66633


namespace NUMINAMATH_GPT_hats_in_shipment_l666_66601

theorem hats_in_shipment (H : ℝ) (h_condition : 0.75 * H = 90) : H = 120 :=
sorry

end NUMINAMATH_GPT_hats_in_shipment_l666_66601


namespace NUMINAMATH_GPT_first_day_of_month_l666_66691

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end NUMINAMATH_GPT_first_day_of_month_l666_66691


namespace NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l666_66610

variable (x : ℝ)
def p := |x| ≤ 2
def q := 0 ≤ x ∧ x ≤ 2

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬ q x := by
  sorry

end NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l666_66610


namespace NUMINAMATH_GPT_smallest_positive_b_l666_66699

theorem smallest_positive_b (b : ℤ) :
  b % 5 = 1 ∧ b % 4 = 2 ∧ b % 7 = 3 → b = 86 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_b_l666_66699


namespace NUMINAMATH_GPT_line_equation_l666_66652

theorem line_equation {x y : ℝ} (h : (x = 1) ∧ (y = -3)) :
  ∃ c : ℝ, x - 2 * y + c = 0 ∧ c = 7 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l666_66652


namespace NUMINAMATH_GPT_lcm_18_24_l666_66609

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end NUMINAMATH_GPT_lcm_18_24_l666_66609


namespace NUMINAMATH_GPT_odd_n_cubed_plus_23n_divisibility_l666_66603

theorem odd_n_cubed_plus_23n_divisibility (n : ℤ) (h1 : n % 2 = 1) : (n^3 + 23 * n) % 24 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_odd_n_cubed_plus_23n_divisibility_l666_66603


namespace NUMINAMATH_GPT_rate_of_rainfall_is_one_l666_66671

variable (R : ℝ)
variable (h1 : 2 + 4 * R + 4 * 3 = 18)

theorem rate_of_rainfall_is_one : R = 1 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_rainfall_is_one_l666_66671


namespace NUMINAMATH_GPT_coins_remainder_l666_66624

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end NUMINAMATH_GPT_coins_remainder_l666_66624


namespace NUMINAMATH_GPT_period_of_f_l666_66692

noncomputable def f (x : ℝ) : ℝ := sorry

theorem period_of_f (a : ℝ) (h : a ≠ 0) (H : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 4 * |a| :=
by
  sorry

end NUMINAMATH_GPT_period_of_f_l666_66692


namespace NUMINAMATH_GPT_angle_C_max_perimeter_l666_66673

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def circumradius_2 (r : ℝ) : Prop :=
  r = 2

def satisfies_condition (a b c A B C : ℝ) : Prop :=
  (a - c)*(Real.sin A + Real.sin C) = b*(Real.sin A - Real.sin B)

theorem angle_C (A B C a b c : ℝ) (h₁ : triangle_ABC A B C a b c) 
                 (h₂ : satisfies_condition a b c A B C)
                 (h₃ : circumradius_2 (2 : ℝ)) : 
  C = Real.pi / 3 :=
sorry

theorem max_perimeter (A B C a b c r : ℝ) (h₁ : triangle_ABC A B C a b c)
                      (h₂ : satisfies_condition a b c A B C)
                      (h₃ : circumradius_2 r) : 
  4 * Real.sqrt 3 + 2 * Real.sqrt 3 = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_angle_C_max_perimeter_l666_66673


namespace NUMINAMATH_GPT_find_cost_per_pound_of_mixture_l666_66637

-- Problem Definitions and Conditions
variable (x : ℝ) -- the variable x represents the pounds of Spanish peanuts used
variable (y : ℝ) -- the cost per pound of the mixture we're trying to find
def cost_virginia_pound : ℝ := 3.50
def cost_spanish_pound : ℝ := 3.00
def weight_virginia : ℝ := 10.0

-- Formula for the cost per pound of the mixture
noncomputable def cost_per_pound_of_mixture : ℝ := (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x)

-- Proof Problem Statement
theorem find_cost_per_pound_of_mixture (h : cost_per_pound_of_mixture x = y) : 
  y = (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x) := sorry

end NUMINAMATH_GPT_find_cost_per_pound_of_mixture_l666_66637


namespace NUMINAMATH_GPT_profit_equation_example_l666_66634

noncomputable def profit_equation (a b : ℝ) (x : ℝ) : Prop :=
  a * (1 + x) ^ 2 = b

theorem profit_equation_example :
  profit_equation 250 360 x :=
by
  have : 25 * (1 + x) ^ 2 = 36 := sorry
  sorry

end NUMINAMATH_GPT_profit_equation_example_l666_66634


namespace NUMINAMATH_GPT_find_g7_l666_66670

-- Given the required functional equation and specific value g(6) = 7
theorem find_g7 (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x + g y) (H2 : g 6 = 7) : g 7 = 49 / 6 := by
  sorry

end NUMINAMATH_GPT_find_g7_l666_66670


namespace NUMINAMATH_GPT_product_of_ab_l666_66619

theorem product_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 7) : a * b = -10 :=
by
  sorry

end NUMINAMATH_GPT_product_of_ab_l666_66619


namespace NUMINAMATH_GPT_round_robin_matches_l666_66684

-- Define the number of players in the tournament
def numPlayers : ℕ := 10

-- Define a function to calculate the number of matches in a round-robin tournament
def calculateMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

-- Theorem statement to prove that the number of matches in a 10-person round-robin chess tournament is 45
theorem round_robin_matches : calculateMatches numPlayers = 45 := by
  sorry

end NUMINAMATH_GPT_round_robin_matches_l666_66684


namespace NUMINAMATH_GPT_mt_product_l666_66690

def g : ℝ → ℝ := sorry

axiom func_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

axiom g3_value : g 3 = 6

def m : ℕ := 1

def t : ℝ := 6

theorem mt_product : m * t = 6 :=
by 
  sorry

end NUMINAMATH_GPT_mt_product_l666_66690


namespace NUMINAMATH_GPT_larger_angle_is_99_l666_66654

theorem larger_angle_is_99 (x : ℝ) (h1 : 2 * x + 18 = 180) : x + 18 = 99 :=
by
  sorry

end NUMINAMATH_GPT_larger_angle_is_99_l666_66654


namespace NUMINAMATH_GPT_number_of_players_l666_66693

theorem number_of_players (x y z : ℕ) 
  (h1 : x + y + z = 10)
  (h2 : x * y + y * z + z * x = 31) : 
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 3) ∨ (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 3) ∨ (x = 5 ∧ y = 3 ∧ z = 2) :=
sorry

end NUMINAMATH_GPT_number_of_players_l666_66693


namespace NUMINAMATH_GPT_move_left_is_negative_l666_66672

theorem move_left_is_negative (movement_right : ℝ) (h : movement_right = 3) : -movement_right = -3 := 
by 
  sorry

end NUMINAMATH_GPT_move_left_is_negative_l666_66672


namespace NUMINAMATH_GPT_find_f4_l666_66665

-- Let f be a function from ℝ to ℝ with the following properties:
variable (f : ℝ → ℝ)

-- 1. f(x + 1) is an odd function
axiom f_odd : ∀ x, f (-(x + 1)) = -f (x + 1)

-- 2. f(x - 1) is an even function
axiom f_even : ∀ x, f (-(x - 1)) = f (x - 1)

-- 3. f(0) = 2
axiom f_zero : f 0 = 2

-- Prove that f(4) = -2
theorem find_f4 : f 4 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f4_l666_66665


namespace NUMINAMATH_GPT_find_t_l666_66602

-- Definitions of the vectors involved
def vector_AB : ℝ × ℝ := (2, 3)
def vector_AC (t : ℝ) : ℝ × ℝ := (3, t)
def vector_BC (t : ℝ) : ℝ × ℝ := ((vector_AC t).1 - (vector_AB).1, (vector_AC t).2 - (vector_AB).2)

-- Condition for orthogonality
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Main statement to be proved
theorem find_t : ∃ t : ℝ, is_perpendicular vector_AB (vector_BC t) ∧ t = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l666_66602


namespace NUMINAMATH_GPT_value_of_F_l666_66687

theorem value_of_F (D E F : ℕ) (hD : D < 10) (hE : E < 10) (hF : F < 10)
    (h1 : (8 + 5 + D + 7 + 3 + E + 2) % 3 = 0)
    (h2 : (4 + 1 + 7 + D + E + 6 + F) % 3 = 0) : 
    F = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_F_l666_66687


namespace NUMINAMATH_GPT_exponent_multiplication_l666_66674

-- Define the core condition: the base 625
def base := 625

-- Define the exponents
def exp1 := 0.08
def exp2 := 0.17
def combined_exp := exp1 + exp2

-- The mathematical goal to prove
theorem exponent_multiplication (b : ℝ) (e1 e2 : ℝ) (h1 : b = 625) (h2 : e1 = 0.08) (h3 : e2 = 0.17) :
  (b ^ e1 * b ^ e2) = 5 :=
by {
  -- Sorry is added to skip the actual proof steps.
  sorry
}

end NUMINAMATH_GPT_exponent_multiplication_l666_66674


namespace NUMINAMATH_GPT_mark_lloyd_ratio_l666_66664

theorem mark_lloyd_ratio (M L C : ℕ) (h1 : M = L) (h2 : M = C - 10) (h3 : C = 100) (h4 : M + L + C + 80 = 300) : M = L :=
by {
  sorry -- proof steps go here
}

end NUMINAMATH_GPT_mark_lloyd_ratio_l666_66664


namespace NUMINAMATH_GPT_parabola_focus_distance_l666_66621

theorem parabola_focus_distance (p : ℝ) (h_pos : p > 0) (A : ℝ × ℝ)
  (h_A_on_parabola : A.2 = 5 ∧ A.1^2 = 2 * p * A.2)
  (h_AF : abs (A.2 - (p / 2)) = 8) : p = 6 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l666_66621


namespace NUMINAMATH_GPT_no_extreme_value_at_5_20_l666_66694

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 4 * x ^ 2 - k * x - 8

theorem no_extreme_value_at_5_20 (k : ℝ) :
  ¬ (∃ (c : ℝ), (forall (x : ℝ), f k x = f k c + (4 * (x - c) ^ 2 - 8 - 20)) ∧ c = 5) ↔ (k ≤ 40 ∨ k ≥ 160) := sorry

end NUMINAMATH_GPT_no_extreme_value_at_5_20_l666_66694


namespace NUMINAMATH_GPT_array_sum_remainder_mod_9_l666_66636

theorem array_sum_remainder_mod_9 :
  let sum_terms := ∑' r : ℕ, ∑' c : ℕ, (1 / (4 ^ r)) * (1 / (9 ^ c))
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ sum_terms = m / n ∧ (m + n) % 9 = 5 :=
by
  sorry

end NUMINAMATH_GPT_array_sum_remainder_mod_9_l666_66636


namespace NUMINAMATH_GPT_sum_of_cubes_l666_66661

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l666_66661


namespace NUMINAMATH_GPT_area_constant_k_l666_66628

theorem area_constant_k (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diagonal : d = Real.sqrt (l^2 + w^2)) :
  ∃ k : ℝ, (k = 10 / 29) ∧ (l * w = k * d^2) :=
by
  sorry

end NUMINAMATH_GPT_area_constant_k_l666_66628


namespace NUMINAMATH_GPT_not_divisible_by_5_for_4_and_7_l666_66660

-- Define a predicate that checks if a given number is not divisible by another number
def notDivisibleBy (n k : ℕ) : Prop := ¬ (n % k = 0)

-- Define the expression we are interested in
def expression (b : ℕ) : ℕ := 3 * b^3 - b^2 + b - 1

-- The theorem we want to prove
theorem not_divisible_by_5_for_4_and_7 :
  notDivisibleBy (expression 4) 5 ∧ notDivisibleBy (expression 7) 5 :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_5_for_4_and_7_l666_66660


namespace NUMINAMATH_GPT_find_r_l666_66605

noncomputable def f (r a : ℝ) (x : ℝ) : ℝ := (x - r - 1) * (x - r - 8) * (x - a)
noncomputable def g (r b : ℝ) (x : ℝ) : ℝ := (x - r - 2) * (x - r - 9) * (x - b)

theorem find_r
  (r a b : ℝ)
  (h_condition1 : ∀ x, f r a x - g r b x = r)
  (h_condition2 : f r a (r + 2) = r)
  (h_condition3 : f r a (r + 9) = r)
  : r = -264 / 7 := sorry

end NUMINAMATH_GPT_find_r_l666_66605


namespace NUMINAMATH_GPT_inequality_solution_l666_66606

theorem inequality_solution (x : ℝ) (h : 3 * x - 5 > 11 - 2 * x) : x > 16 / 5 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l666_66606


namespace NUMINAMATH_GPT_inconsistent_mixture_volume_l666_66640

theorem inconsistent_mixture_volume :
  ∀ (diesel petrol water total_volume : ℚ),
    diesel = 4 →
    petrol = 4 →
    total_volume = 2.666666666666667 →
    diesel + petrol + water = total_volume →
    false :=
by
  intros diesel petrol water total_volume diesel_eq petrol_eq total_volume_eq volume_eq
  rw [diesel_eq, petrol_eq] at volume_eq
  sorry

end NUMINAMATH_GPT_inconsistent_mixture_volume_l666_66640


namespace NUMINAMATH_GPT_integer_solutions_l666_66648

theorem integer_solutions (m : ℤ) :
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) → 
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + 1 / 2 = 0) →
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l666_66648


namespace NUMINAMATH_GPT_unique_solution_h_l666_66613

theorem unique_solution_h (h : ℝ) (hne_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_h_l666_66613


namespace NUMINAMATH_GPT_largest_n_satisfying_conditions_l666_66662

theorem largest_n_satisfying_conditions : 
  ∃ n : ℤ, 200 < n ∧ n < 250 ∧ (∃ k : ℤ, 12 * n = k^2) ∧ n = 243 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_satisfying_conditions_l666_66662


namespace NUMINAMATH_GPT_boys_and_girls_in_class_l666_66698

theorem boys_and_girls_in_class (m d : ℕ)
  (A : (m - 1 = 10 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)) ∨ 
       (m - 1 = 14 - 4 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)))
  (B : (m - 1 = 13 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)) ∨ 
       (m - 1 = 11 - 4 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)))
  (C : (m - 1 = 13 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4)) ∨ 
       (m - 1 = 19 - 4 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4))) : 
  m = 14 ∧ d = 15 := 
sorry

end NUMINAMATH_GPT_boys_and_girls_in_class_l666_66698


namespace NUMINAMATH_GPT_area_difference_of_tablets_l666_66657

theorem area_difference_of_tablets 
  (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 6) (h2 : d2 = 5) 
  (hs1 : d1^2 = 2 * s1^2) (hs2 : d2^2 = 2 * s2^2) 
  (A1 : ℝ) (A2 : ℝ) (hA1 : A1 = s1^2) (hA2 : A2 = s2^2)
  : A1 - A2 = 5.5 := 
sorry

end NUMINAMATH_GPT_area_difference_of_tablets_l666_66657


namespace NUMINAMATH_GPT_product_of_three_numbers_l666_66675

theorem product_of_three_numbers 
  (a b c : ℕ) 
  (h1 : a + b + c = 300) 
  (h2 : 9 * a = b - 11) 
  (h3 : 9 * a = c + 15) : 
  a * b * c = 319760 := 
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l666_66675


namespace NUMINAMATH_GPT_jerry_claim_percentage_l666_66651

theorem jerry_claim_percentage
  (salary_years : ℕ)
  (annual_salary : ℕ)
  (medical_bills : ℕ)
  (punitive_multiplier : ℕ)
  (received_amount : ℕ)
  (total_claim : ℕ)
  (percentage_claim : ℕ) :
  salary_years = 30 →
  annual_salary = 50000 →
  medical_bills = 200000 →
  punitive_multiplier = 3 →
  received_amount = 5440000 →
  total_claim = (annual_salary * salary_years) + medical_bills + (punitive_multiplier * ((annual_salary * salary_years) + medical_bills)) →
  percentage_claim = (received_amount * 100) / total_claim →
  percentage_claim = 80 :=
by
  sorry

end NUMINAMATH_GPT_jerry_claim_percentage_l666_66651


namespace NUMINAMATH_GPT_logarithmic_expression_l666_66625

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression :
  let log2 := lg 2
  let log5 := lg 5
  log2 + log5 = 1 →
  (log2^3 + 3 * log2 * log5 + log5^3 = 1) :=
by
  intros log2 log5 h
  sorry

end NUMINAMATH_GPT_logarithmic_expression_l666_66625
