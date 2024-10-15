import Mathlib

namespace NUMINAMATH_GPT_find_number_l1705_170526

theorem find_number (n : ℝ) (h : 3 / 5 * ((2 / 3 + 3 / 8) / n) - 1 / 16 = 0.24999999999999994) : n = 48 :=
  sorry

end NUMINAMATH_GPT_find_number_l1705_170526


namespace NUMINAMATH_GPT_rotated_angle_540_deg_l1705_170548

theorem rotated_angle_540_deg (θ : ℝ) (h : θ = 60) : 
  (θ - 540) % 360 % 180 = 60 :=
by
  sorry

end NUMINAMATH_GPT_rotated_angle_540_deg_l1705_170548


namespace NUMINAMATH_GPT_problem_statement_l1705_170516

theorem problem_statement (a b c : ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) : 
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1705_170516


namespace NUMINAMATH_GPT_standard_eq_of_largest_circle_l1705_170532

theorem standard_eq_of_largest_circle 
  (m : ℝ)
  (hm : 0 < m) :
  ∃ r : ℝ, 
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 8) ↔ 
      (x^2 + (y - 1)^2 = r)) :=
sorry

end NUMINAMATH_GPT_standard_eq_of_largest_circle_l1705_170532


namespace NUMINAMATH_GPT_set_intersection_complement_l1705_170585

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {3, 4, 5}

-- State the theorem
theorem set_intersection_complement :
  (U \ A) ∩ B = {4, 5} := by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1705_170585


namespace NUMINAMATH_GPT_probability_X_Y_Z_problems_l1705_170534

-- Define the success probabilities for Problem A
def P_X_A : ℚ := 1 / 5
def P_Y_A : ℚ := 1 / 2

-- Define the success probabilities for Problem B
def P_Y_B : ℚ := 3 / 5

-- Define the negation of success probabilities for Problem C
def P_Y_not_C : ℚ := 5 / 8
def P_X_not_C : ℚ := 3 / 4
def P_Z_not_C : ℚ := 7 / 16

-- State the final probability theorem
theorem probability_X_Y_Z_problems :
  P_X_A * P_Y_A * P_Y_B * P_Y_not_C * P_X_not_C * P_Z_not_C = 63 / 2048 := 
sorry

end NUMINAMATH_GPT_probability_X_Y_Z_problems_l1705_170534


namespace NUMINAMATH_GPT_eighty_five_squared_l1705_170512

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end NUMINAMATH_GPT_eighty_five_squared_l1705_170512


namespace NUMINAMATH_GPT_maria_total_cost_l1705_170545

variable (pencil_cost : ℕ)
variable (pen_cost : ℕ)

def total_cost (pencil_cost pen_cost : ℕ) : ℕ :=
  pencil_cost + pen_cost

theorem maria_total_cost : pencil_cost = 8 → pen_cost = pencil_cost / 2 → total_cost pencil_cost pen_cost = 12 := by
  sorry

end NUMINAMATH_GPT_maria_total_cost_l1705_170545


namespace NUMINAMATH_GPT_ethanol_in_fuel_A_l1705_170530

def fuel_tank_volume : ℝ := 208
def fuel_A_volume : ℝ := 82
def fuel_B_volume : ℝ := fuel_tank_volume - fuel_A_volume
def ethanol_in_fuel_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem ethanol_in_fuel_A 
  (x : ℝ) 
  (H_fuel_tank_capacity : fuel_tank_volume = 208) 
  (H_fuel_A_volume : fuel_A_volume = 82) 
  (H_fuel_B_volume : fuel_B_volume = 126) 
  (H_ethanol_in_fuel_B : ethanol_in_fuel_B = 0.16) 
  (H_total_ethanol : total_ethanol = 30) 
  : 82 * x + 0.16 * 126 = 30 → x = 0.12 := by
  sorry

end NUMINAMATH_GPT_ethanol_in_fuel_A_l1705_170530


namespace NUMINAMATH_GPT_temperature_representation_l1705_170554

-- Defining the temperature representation problem
def posTemp := 10 -- $10^\circ \mathrm{C}$ above zero
def negTemp := -10 -- $10^\circ \mathrm{C}$ below zero
def aboveZero (temp : Int) : Prop := temp > 0
def belowZero (temp : Int) : Prop := temp < 0

-- The proof statement to be proved using the given conditions
theorem temperature_representation : 
  (aboveZero posTemp → posTemp = 10) ∧ (belowZero negTemp → negTemp = -10) := 
  by
    sorry -- Proof would go here

end NUMINAMATH_GPT_temperature_representation_l1705_170554


namespace NUMINAMATH_GPT_numbers_combination_to_24_l1705_170584

theorem numbers_combination_to_24 :
  (40 / 4) + 12 + 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_numbers_combination_to_24_l1705_170584


namespace NUMINAMATH_GPT_sin_70_given_sin_10_l1705_170509

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_70_given_sin_10_l1705_170509


namespace NUMINAMATH_GPT_quadratic_solutions_l1705_170538

theorem quadratic_solutions (x : ℝ) : x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solutions_l1705_170538


namespace NUMINAMATH_GPT_parabola_equation_l1705_170507

theorem parabola_equation (p : ℝ) (hp : 0 < p)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (A B : ℝ × ℝ)
  (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (h_intersect : y1^2 = 2*p*x1 ∧ y2^2 = 2*p*x2)
  (M : ℝ × ℝ) (hM : M = ((x1 + x2) / 2, (y1 + y2) / 2))
  (hM_coords : M = (3, 2)) :
  p = 2 ∨ p = 4 :=
sorry

end NUMINAMATH_GPT_parabola_equation_l1705_170507


namespace NUMINAMATH_GPT_mean_of_y_and_18_is_neg1_l1705_170582

theorem mean_of_y_and_18_is_neg1 (y : ℤ) : 
  ((4 + 6 + 10 + 14) / 4) = ((y + 18) / 2) → y = -1 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_mean_of_y_and_18_is_neg1_l1705_170582


namespace NUMINAMATH_GPT_expand_product_l1705_170510

theorem expand_product (x : ℝ) : (x + 5) * (x - 16) = x^2 - 11 * x - 80 :=
by sorry

end NUMINAMATH_GPT_expand_product_l1705_170510


namespace NUMINAMATH_GPT_transition_to_modern_population_reproduction_l1705_170597

-- Defining the conditions as individual propositions
def A : Prop := ∃ (m b : ℝ), m < 0 ∧ b = 0
def B : Prop := ∃ (m b : ℝ), m < 0 ∧ b < 0
def C : Prop := ∃ (m b : ℝ), m > 0 ∧ b = 0
def D : Prop := ∃ (m b : ℝ), m > 0 ∧ b > 0

-- Defining the question as a property marking the transition from traditional to modern types of population reproduction
def Q : Prop := B

-- The proof problem
theorem transition_to_modern_population_reproduction :
  Q = B :=
by
  sorry

end NUMINAMATH_GPT_transition_to_modern_population_reproduction_l1705_170597


namespace NUMINAMATH_GPT_average_of_numbers_l1705_170502

theorem average_of_numbers : 
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1 + 1252140 + 2345) / 11 = 114391 :=
by
  sorry

end NUMINAMATH_GPT_average_of_numbers_l1705_170502


namespace NUMINAMATH_GPT_fraction_flower_beds_l1705_170529

theorem fraction_flower_beds (length1 length2 height triangle_area yard_area : ℝ) (h1 : length1 = 18) (h2 : length2 = 30) (h3 : height = 10) (h4 : triangle_area = 2 * (1 / 2 * (6 ^ 2))) (h5 : yard_area = ((length1 + length2) / 2) * height) : 
  (triangle_area / yard_area) = 3 / 20 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_flower_beds_l1705_170529


namespace NUMINAMATH_GPT_jerry_total_logs_l1705_170555

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_jerry_total_logs_l1705_170555


namespace NUMINAMATH_GPT_combined_wattage_l1705_170599

theorem combined_wattage (w1 w2 w3 w4 : ℕ) (h1 : w1 = 60) (h2 : w2 = 80) (h3 : w3 = 100) (h4 : w4 = 120) :
  let nw1 := w1 + w1 / 4
  let nw2 := w2 + w2 / 4
  let nw3 := w3 + w3 / 4
  let nw4 := w4 + w4 / 4
  nw1 + nw2 + nw3 + nw4 = 450 :=
by
  sorry

end NUMINAMATH_GPT_combined_wattage_l1705_170599


namespace NUMINAMATH_GPT_maximum_possible_median_l1705_170587

theorem maximum_possible_median
  (total_cans : ℕ)
  (total_customers : ℕ)
  (min_cans_per_customer : ℕ)
  (alt_min_cans_per_customer : ℕ)
  (exact_min_cans_count : ℕ)
  (atleast_min_cans_count : ℕ)
  (min_cans_customers : ℕ)
  (alt_min_cans_customer: ℕ): 
  (total_cans = 300) → 
  (total_customers = 120) →
  (min_cans_per_customer = 2) →
  (alt_min_cans_per_customer = 4) →
  (min_cans_customers = 59) →
  (alt_min_cans_customer = 61) →
  (min_cans_per_customer * min_cans_customers + alt_min_cans_per_customer * (total_customers - min_cans_customers) = total_cans) →
  max (min_cans_per_customer + 1) (alt_min_cans_per_customer - 1) = 3 :=
sorry

end NUMINAMATH_GPT_maximum_possible_median_l1705_170587


namespace NUMINAMATH_GPT_match_piles_l1705_170569

theorem match_piles (a b c : ℕ) (h : a + b + c = 96)
    (h1 : 2 * b = a + c) (h2 : 2 * c = b + a) (h3 : 2 * a = c + b) : 
    a = 44 ∧ b = 28 ∧ c = 24 :=
  sorry

end NUMINAMATH_GPT_match_piles_l1705_170569


namespace NUMINAMATH_GPT_trajectory_midpoints_l1705_170523

variables (a b c x y : ℝ)

def arithmetic_sequence (a b c : ℝ) : Prop := c = 2 * b - a

def line_eq (b a c x y : ℝ) : Prop := b * x + a * y + c = 0

def parabola_eq (x y : ℝ) : Prop := y^2 = -0.5 * x

theorem trajectory_midpoints
  (hac : arithmetic_sequence a b c)
  (line_cond : line_eq b a c x y)
  (parabola_cond : parabola_eq x y) :
  (x + 1 = -(2 * y - 1)^2) ∧ (y ≠ 1) :=
sorry

end NUMINAMATH_GPT_trajectory_midpoints_l1705_170523


namespace NUMINAMATH_GPT_circle_equation_l1705_170564

theorem circle_equation (x y : ℝ) : (3 * x - 4 * y + 12 = 0) → (x^2 + 4 * x + y^2 - 3 * y = 0) :=
sorry

end NUMINAMATH_GPT_circle_equation_l1705_170564


namespace NUMINAMATH_GPT_find_b_num_days_worked_l1705_170593

noncomputable def a_num_days_worked := 6
noncomputable def b_num_days_worked := 9  -- This is what we want to verify
noncomputable def c_num_days_worked := 4

noncomputable def c_daily_wage := 105
noncomputable def wage_ratio_a := 3
noncomputable def wage_ratio_b := 4
noncomputable def wage_ratio_c := 5

-- Helper to find daily wages for a and b given the ratio and c's wage
noncomputable def x := c_daily_wage / wage_ratio_c
noncomputable def a_daily_wage := wage_ratio_a * x
noncomputable def b_daily_wage := wage_ratio_b * x

-- Calculate total earnings
noncomputable def a_total_earning := a_num_days_worked * a_daily_wage
noncomputable def c_total_earning := c_num_days_worked * c_daily_wage
noncomputable def total_earning := 1554
noncomputable def b_total_earning := b_num_days_worked * b_daily_wage

theorem find_b_num_days_worked : total_earning = a_total_earning + b_total_earning + c_total_earning → b_num_days_worked = 9 := by
  sorry

end NUMINAMATH_GPT_find_b_num_days_worked_l1705_170593


namespace NUMINAMATH_GPT_fourth_number_is_57_l1705_170504

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def sum_list (l : List ℕ) : ℕ :=
  l.foldr (.+.) 0

theorem fourth_number_is_57 : 
  ∃ (N : ℕ), N < 100 ∧ 177 + N = 4 * (33 + digit_sum N) ∧ N = 57 :=
by {
  sorry
}

end NUMINAMATH_GPT_fourth_number_is_57_l1705_170504


namespace NUMINAMATH_GPT_slower_speed_l1705_170553

theorem slower_speed (f e d : ℕ) (h1 : f = 14) (h2 : e = 20) (h3 : d = 50) (x : ℕ) : 
  (50 / x : ℚ) = (50 / 14 : ℚ) + (20 / 14 : ℚ) → x = 10 := by
  sorry

end NUMINAMATH_GPT_slower_speed_l1705_170553


namespace NUMINAMATH_GPT_range_of_a_l1705_170557

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| - |x - 3| ≤ a) → a ≥ -5 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1705_170557


namespace NUMINAMATH_GPT_negation_of_universal_quadratic_l1705_170505

theorem negation_of_universal_quadratic (P : ∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  ¬(∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ ∃ a b c : ℝ, a ≠ 0 ∧ ¬(∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_quadratic_l1705_170505


namespace NUMINAMATH_GPT_shaded_area_l1705_170520

-- Defining the conditions
def small_square_side := 4
def large_square_side := 12
def half_large_square_side := large_square_side / 2

-- DG is calculated as (12 / 16) * small_square_side = 3
def DG := (large_square_side / (half_large_square_side + small_square_side)) * small_square_side

-- Calculating area of triangle DGF
def area_triangle_DGF := (DG * small_square_side) / 2

-- Area of the smaller square
def area_small_square := small_square_side * small_square_side

-- Area of the shaded region
def area_shaded_region := area_small_square - area_triangle_DGF

-- The theorem stating the question
theorem shaded_area : area_shaded_region = 10 := by
  sorry

end NUMINAMATH_GPT_shaded_area_l1705_170520


namespace NUMINAMATH_GPT_product_modulo_10_l1705_170519

-- Define the numbers involved
def a := 2457
def b := 7623
def c := 91309

-- Define the modulo operation we're interested in
def modulo_10 (n : Nat) : Nat := n % 10

-- State the theorem we want to prove
theorem product_modulo_10 :
  modulo_10 (a * b * c) = 9 :=
sorry

end NUMINAMATH_GPT_product_modulo_10_l1705_170519


namespace NUMINAMATH_GPT_devin_biked_more_l1705_170527

def cyra_distance := 77
def cyra_time := 7
def cyra_speed := cyra_distance / cyra_time
def devin_speed := cyra_speed + 3
def marathon_time := 7
def devin_distance := devin_speed * marathon_time
def distance_difference := devin_distance - cyra_distance

theorem devin_biked_more : distance_difference = 21 := 
  by
    sorry

end NUMINAMATH_GPT_devin_biked_more_l1705_170527


namespace NUMINAMATH_GPT_greatest_integer_gcd_l1705_170546

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_gcd_l1705_170546


namespace NUMINAMATH_GPT_intersection_P_Q_l1705_170571

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the problem statement as a theorem
theorem intersection_P_Q : P ∩ Q = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1705_170571


namespace NUMINAMATH_GPT_smallest_integer_odd_sequence_l1705_170547

/-- Given the median of a set of consecutive odd integers is 157 and the greatest integer in the set is 171,
    prove that the smallest integer in the set is 149. -/
theorem smallest_integer_odd_sequence (median greatest : ℤ) (h_median : median = 157) (h_greatest : greatest = 171) :
  ∃ smallest : ℤ, smallest = 149 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_odd_sequence_l1705_170547


namespace NUMINAMATH_GPT_smallest_multiple_of_seven_l1705_170515

/-- The definition of the six-digit number formed by digits a, b, and c followed by "321". -/
def form_number (a b c : ℕ) : ℕ := 100000 * a + 10000 * b + 1000 * c + 321

/-- The condition that a, b, and c are distinct and greater than 3. -/
def valid_digits (a b c : ℕ) : Prop := a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_multiple_of_seven (a b c : ℕ)
  (h_valid : valid_digits a b c)
  (h_mult_seven : form_number a b c % 7 = 0) :
  form_number a b c = 468321 :=
sorry

end NUMINAMATH_GPT_smallest_multiple_of_seven_l1705_170515


namespace NUMINAMATH_GPT_line_bisects_circle_and_perpendicular_l1705_170531

   def line_bisects_circle_and_is_perpendicular (x y : ℝ) : Prop :=
     (∃ (b : ℝ), ((2 * x - y + b = 0) ∧ (x^2 + y^2 - 2 * x - 4 * y = 0))) ∧
     ∀ b, (2 * 1 - 2 + b = 0) → b = 0 → (2 * x - y = 0)

   theorem line_bisects_circle_and_perpendicular :
     line_bisects_circle_and_is_perpendicular 1 2 :=
   by
     sorry
   
end NUMINAMATH_GPT_line_bisects_circle_and_perpendicular_l1705_170531


namespace NUMINAMATH_GPT_transformed_circle_eq_l1705_170580

theorem transformed_circle_eq (x y : ℝ) (h : x^2 + y^2 = 1) : x^2 + 9 * (y / 3)^2 = 1 := by
  sorry

end NUMINAMATH_GPT_transformed_circle_eq_l1705_170580


namespace NUMINAMATH_GPT_area_of_given_parallelogram_l1705_170581

def parallelogram_base : ℝ := 24
def parallelogram_height : ℝ := 16
def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem area_of_given_parallelogram : parallelogram_area parallelogram_base parallelogram_height = 384 := 
by sorry

end NUMINAMATH_GPT_area_of_given_parallelogram_l1705_170581


namespace NUMINAMATH_GPT_five_coins_no_105_cents_l1705_170513

theorem five_coins_no_105_cents :
  ¬ ∃ (a b c d e : ℕ), a + b + c + d + e = 5 ∧
    (a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 105) :=
sorry

end NUMINAMATH_GPT_five_coins_no_105_cents_l1705_170513


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1705_170541

-- Define α as an interior angle of triangle ABC
def is_interior_angle_of_triangle (α : ℝ) : Prop :=
  0 < α ∧ α < 180

-- Define the sine condition
def sine_condition (α : ℝ) : Prop :=
  Real.sin α = Real.sqrt 2 / 2

-- Define the main theorem
theorem necessary_but_not_sufficient (α : ℝ) (h1 : is_interior_angle_of_triangle α) (h2 : sine_condition α) :
  (sine_condition α) ↔ (α = 45) ∨ (α = 135) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1705_170541


namespace NUMINAMATH_GPT_polynomial_multiplication_l1705_170514

theorem polynomial_multiplication (x y : ℝ) : 
  (2 * x - 3 * y + 1) * (2 * x + 3 * y - 1) = 4 * x^2 - 9 * y^2 + 6 * y - 1 := by
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_l1705_170514


namespace NUMINAMATH_GPT_cows_grazed_by_C_l1705_170549

-- Define the initial conditions as constants
def cows_grazed_A : ℕ := 24
def months_grazed_A : ℕ := 3
def cows_grazed_B : ℕ := 10
def months_grazed_B : ℕ := 5
def cows_grazed_D : ℕ := 21
def months_grazed_D : ℕ := 3
def share_rent_A : ℕ := 1440
def total_rent : ℕ := 6500

-- Define the cow-months calculation for A, B, D
def cow_months_A : ℕ := cows_grazed_A * months_grazed_A
def cow_months_B : ℕ := cows_grazed_B * months_grazed_B
def cow_months_D : ℕ := cows_grazed_D * months_grazed_D

-- Let x be the number of cows grazed by C
variable (x : ℕ)

-- Define the cow-months calculation for C
def cow_months_C : ℕ := x * 4

-- Define rent per cow-month
def rent_per_cow_month : ℕ := share_rent_A / cow_months_A

-- Proof problem statement
theorem cows_grazed_by_C : 
  (6500 = (cow_months_A + cow_months_B + cow_months_C x + cow_months_D) * rent_per_cow_month) →
  x = 35 := by
  sorry

end NUMINAMATH_GPT_cows_grazed_by_C_l1705_170549


namespace NUMINAMATH_GPT_dog_weights_l1705_170573

structure DogWeightProgression where
  initial: ℕ   -- initial weight in pounds
  week_9: ℕ    -- weight at 9 weeks in pounds
  month_3: ℕ  -- weight at 3 months in pounds
  month_5: ℕ  -- weight at 5 months in pounds
  year_1: ℕ   -- weight at 1 year in pounds

theorem dog_weights :
  ∃ (golden_retriever labrador poodle : DogWeightProgression),
  golden_retriever.initial = 6 ∧
  golden_retriever.week_9 = 12 ∧
  golden_retriever.month_3 = 24 ∧
  golden_retriever.month_5 = 48 ∧
  golden_retriever.year_1 = 78 ∧
  labrador.initial = 8 ∧
  labrador.week_9 = 24 ∧
  labrador.month_3 = 36 ∧
  labrador.month_5 = 72 ∧
  labrador.year_1 = 102 ∧
  poodle.initial = 4 ∧
  poodle.week_9 = 16 ∧
  poodle.month_3 = 32 ∧
  poodle.month_5 = 32 ∧
  poodle.year_1 = 52 :=
by 
  have golden_retriever : DogWeightProgression := { initial := 6, week_9 := 12, month_3 := 24, month_5 := 48, year_1 := 78 }
  have labrador : DogWeightProgression := { initial := 8, week_9 := 24, month_3 := 36, month_5 := 72, year_1 := 102 }
  have poodle : DogWeightProgression := { initial := 4, week_9 := 16, month_3 := 32, month_5 := 32, year_1 := 52 }
  use golden_retriever, labrador, poodle
  repeat { split };
  { sorry }

end NUMINAMATH_GPT_dog_weights_l1705_170573


namespace NUMINAMATH_GPT_inequality_of_abc_l1705_170542

theorem inequality_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
sorry

end NUMINAMATH_GPT_inequality_of_abc_l1705_170542


namespace NUMINAMATH_GPT_units_digit_of_7_pow_2500_l1705_170518

theorem units_digit_of_7_pow_2500 : (7^2500) % 10 = 1 :=
by
  -- Variables and constants can be used to formalize steps if necessary, 
  -- but focus is on the statement itself.
  -- Sorry is used to skip the proof part.
  sorry

end NUMINAMATH_GPT_units_digit_of_7_pow_2500_l1705_170518


namespace NUMINAMATH_GPT_polygon_sides_from_interior_angles_l1705_170537

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_from_interior_angles_l1705_170537


namespace NUMINAMATH_GPT_find_square_divisible_by_four_l1705_170577

/-- There exists an x such that x is a square number, x is divisible by four, 
and 39 < x < 80, and that x = 64 is such a number. --/
theorem find_square_divisible_by_four : ∃ (x : ℕ), (∃ (n : ℕ), x = n^2) ∧ (x % 4 = 0) ∧ (39 < x ∧ x < 80) ∧ x = 64 :=
  sorry

end NUMINAMATH_GPT_find_square_divisible_by_four_l1705_170577


namespace NUMINAMATH_GPT_cricket_player_average_increase_l1705_170591

theorem cricket_player_average_increase
  (average : ℕ) (n : ℕ) (next_innings_runs : ℕ) 
  (x : ℕ) 
  (h1 : average = 32)
  (h2 : n = 20)
  (h3 : next_innings_runs = 200)
  (total_runs := average * n)
  (new_total_runs := total_runs + next_innings_runs)
  (new_average := (average + x))
  (new_total := new_average * (n + 1)):
  new_total_runs = 840 →
  new_total = 840 →
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_cricket_player_average_increase_l1705_170591


namespace NUMINAMATH_GPT_vectors_collinear_has_solution_l1705_170556

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x^2 - 1, 2 + x)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinearity condition (cross product must be zero) as a function
def collinear (x : ℝ) : Prop := (a x).1 * (b x).2 - (b x).1 * (a x).2 = 0

-- The proof statement
theorem vectors_collinear_has_solution (x : ℝ) (h : collinear x) : x = -1 / 2 :=
sorry

end NUMINAMATH_GPT_vectors_collinear_has_solution_l1705_170556


namespace NUMINAMATH_GPT_geometrical_shapes_OABC_l1705_170503

/-- Given distinct points A(x₁, y₁), B(x₂, y₂), and C(2x₁ - x₂, 2y₁ - y₂) on a coordinate plane
    and the origin O(0,0), determine the possible geometrical shapes that the figure OABC can form
    among these three possibilities: (1) parallelogram (2) straight line (3) rhombus.
    
    Prove that the figure OABC can form either a parallelogram or a straight line,
    but not a rhombus.
-/
theorem geometrical_shapes_OABC (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂) ∧ (x₂, y₂) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂)) :
  (∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁) ∨
  (2 * x₁ = x₁ + x₂ ∧ 2 * y₁ = y₁ + y₂) :=
sorry

end NUMINAMATH_GPT_geometrical_shapes_OABC_l1705_170503


namespace NUMINAMATH_GPT_collective_earnings_l1705_170522

theorem collective_earnings:
  let lloyd_hours := 10.5
  let mary_hours := 12.0
  let tom_hours := 7.0
  let lloyd_normal_hours := 7.5
  let mary_normal_hours := 8.0
  let tom_normal_hours := 9.0
  let lloyd_rate := 4.5
  let mary_rate := 5.0
  let tom_rate := 6.0
  let lloyd_overtime_rate := 2.5 * lloyd_rate
  let mary_overtime_rate := 3.0 * mary_rate
  let tom_overtime_rate := 2.0 * tom_rate
  let lloyd_earnings := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours - lloyd_normal_hours) * lloyd_overtime_rate)
  let mary_earnings := (mary_normal_hours * mary_rate) + ((mary_hours - mary_normal_hours) * mary_overtime_rate)
  let tom_earnings := (tom_hours * tom_rate)
  let total_earnings := lloyd_earnings + mary_earnings + tom_earnings
  total_earnings = 209.50 := by
  sorry

end NUMINAMATH_GPT_collective_earnings_l1705_170522


namespace NUMINAMATH_GPT_find_x_l1705_170508

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1705_170508


namespace NUMINAMATH_GPT_value_of_f_log_20_l1705_170588

variable (f : ℝ → ℝ)
variable (h₁ : ∀ x : ℝ, f (-x) = -f x)
variable (h₂ : ∀ x : ℝ, f (x - 2) = f (x + 2))
variable (h₃ : ∀ x : ℝ, x > -1 ∧ x < 0 → f x = 2^x + 1/5)

theorem value_of_f_log_20 : f (Real.log 20 / Real.log 2) = -1 := sorry

end NUMINAMATH_GPT_value_of_f_log_20_l1705_170588


namespace NUMINAMATH_GPT_no_integer_root_l1705_170575

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7 * x - 14 * (q^2 + 1) = 0 := sorry

end NUMINAMATH_GPT_no_integer_root_l1705_170575


namespace NUMINAMATH_GPT_one_positive_zero_l1705_170574

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x - 1

theorem one_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ f x a = 0 :=
sorry

end NUMINAMATH_GPT_one_positive_zero_l1705_170574


namespace NUMINAMATH_GPT_problem_statement_l1705_170535

noncomputable def f (x k : ℝ) : ℝ :=
  (1/5) * (x - k + 4500 / x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ :=
  100 / x * f x k

theorem problem_statement (x k : ℝ)
  (hx1 : 60 ≤ x) (hx2 : x ≤ 120)
  (hk1 : 60 ≤ k) (hk2 : k ≤ 100)
  (H : f 120 k = 11.5) :

  (∀ x, 60 ≤ x ∧ x ≤ 100 → f x k ≤ 9 ∧ 
  (if 75 ≤ k ∧ k ≤ 100 then fuel_consumption_100km (9000 / k) k = 20 - k^2 / 900
   else fuel_consumption_100km 120 k = 105 / 4 - k / 6)) :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1705_170535


namespace NUMINAMATH_GPT_rational_reciprocal_pow_2014_l1705_170566

theorem rational_reciprocal_pow_2014 (a : ℚ) (h : a = 1 / a) : a ^ 2014 = 1 := by
  sorry

end NUMINAMATH_GPT_rational_reciprocal_pow_2014_l1705_170566


namespace NUMINAMATH_GPT_min_length_M_inter_N_l1705_170501

def setM (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3 / 4}
def setN (n : ℝ) : Set ℝ := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
def setP : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem min_length_M_inter_N (m n : ℝ) 
  (hm : 0 ≤ m ∧ m + 3 / 4 ≤ 1) 
  (hn : 1 / 3 ≤ n ∧ n ≤ 1) : 
  let I := (setM m ∩ setN n)
  ∃ Iinf Isup : ℝ, I = {x | Iinf ≤ x ∧ x ≤ Isup} ∧ Isup - Iinf = 1 / 12 :=
  sorry

end NUMINAMATH_GPT_min_length_M_inter_N_l1705_170501


namespace NUMINAMATH_GPT_ages_of_residents_l1705_170579

theorem ages_of_residents (a b c : ℕ)
  (h1 : a * b * c = 1296)
  (h2 : a + b + c = 91)
  (h3 : ∀ x y z : ℕ, x * y * z = 1296 → x + y + z = 91 → (x < 80 ∧ y < 80 ∧ z < 80) → (x = 1 ∧ y = 18 ∧ z = 72)) :
  (a = 1 ∧ b = 18 ∧ c = 72 ∨ a = 1 ∧ b = 72 ∧ c = 18 ∨ a = 18 ∧ b = 1 ∧ c = 72 ∨ a = 18 ∧ b = 72 ∧ c = 1 ∨ a = 72 ∧ b = 1 ∧ c = 18 ∨ a = 72 ∧ b = 18 ∧ c = 1) :=
by
  sorry

end NUMINAMATH_GPT_ages_of_residents_l1705_170579


namespace NUMINAMATH_GPT_fractions_product_simplified_l1705_170533

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end NUMINAMATH_GPT_fractions_product_simplified_l1705_170533


namespace NUMINAMATH_GPT_probability_diff_colors_l1705_170583

-- Definitions based on the conditions provided.
-- Total number of chips
def total_chips := 15

-- Individual probabilities of drawing each color first
def prob_green_first := 6 / total_chips
def prob_purple_first := 5 / total_chips
def prob_orange_first := 4 / total_chips

-- Probabilities of drawing a different color second
def prob_not_green := 9 / total_chips
def prob_not_purple := 10 / total_chips
def prob_not_orange := 11 / total_chips

-- Combined probabilities for each case
def prob_green_then_diff := prob_green_first * prob_not_green
def prob_purple_then_diff := prob_purple_first * prob_not_purple
def prob_orange_then_diff := prob_orange_first * prob_not_orange

-- Total probability of drawing two chips of different colors
def total_prob_diff_colors := prob_green_then_diff + prob_purple_then_diff + prob_orange_then_diff

-- Theorem statement to be proved
theorem probability_diff_colors : total_prob_diff_colors = 148 / 225 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_probability_diff_colors_l1705_170583


namespace NUMINAMATH_GPT_extended_ohara_triple_example_l1705_170576

theorem extended_ohara_triple_example : 
  (2 * Real.sqrt 49 + Real.sqrt 64 = 22) :=
by
  -- We are stating the conditions and required proof here.
  sorry

end NUMINAMATH_GPT_extended_ohara_triple_example_l1705_170576


namespace NUMINAMATH_GPT_friends_count_l1705_170568

theorem friends_count (n : ℕ) (average_rent : ℝ) (new_average_rent : ℝ) (original_rent : ℝ) (increase_percent : ℝ)
  (H1 : average_rent = 800)
  (H2 : new_average_rent = 870)
  (H3 : original_rent = 1400)
  (H4 : increase_percent = 0.20) :
  n = 4 :=
by
  -- Define the initial total rent
  let initial_total_rent := n * average_rent
  -- Define the increased rent for one person
  let increased_rent := original_rent * (1 + increase_percent)
  -- Define the new total rent
  let new_total_rent := initial_total_rent - original_rent + increased_rent
  -- Set up the new average rent equation
  have rent_equation := new_total_rent = n * new_average_rent
  sorry

end NUMINAMATH_GPT_friends_count_l1705_170568


namespace NUMINAMATH_GPT_ratio_kid_to_adult_ticket_l1705_170551

theorem ratio_kid_to_adult_ticket (A : ℝ) : 
  (6 * 5 + 2 * A = 50) → (5 / A = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_kid_to_adult_ticket_l1705_170551


namespace NUMINAMATH_GPT_f_neg2_minus_f_neg3_l1705_170598

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end NUMINAMATH_GPT_f_neg2_minus_f_neg3_l1705_170598


namespace NUMINAMATH_GPT_expression_simplification_l1705_170559

theorem expression_simplification : (4^2 * 7 / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11)) = 44 / 9 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l1705_170559


namespace NUMINAMATH_GPT_area_of_rectangle_is_correct_l1705_170500

-- Given Conditions
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- Question: Find the area of the rectangle
def area := length * width

-- The theorem to prove
theorem area_of_rectangle_is_correct : area = 588 :=
by
  -- Proof steps can go here.
  sorry

end NUMINAMATH_GPT_area_of_rectangle_is_correct_l1705_170500


namespace NUMINAMATH_GPT_problem_l1705_170543

variable (a b : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end NUMINAMATH_GPT_problem_l1705_170543


namespace NUMINAMATH_GPT_relationship_among_m_n_k_l1705_170552

theorem relationship_among_m_n_k :
  (¬ ∃ x : ℝ, |2 * x - 3| + m = 0) → 
  (∃! x: ℝ, |3 * x - 4| + n = 0) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 5| + k = 0 ∧ |4 * x₂ - 5| + k = 0) →
  (m > n ∧ n > k) :=
by
  intros h1 h2 h3
  -- Proof part will be added here
  sorry

end NUMINAMATH_GPT_relationship_among_m_n_k_l1705_170552


namespace NUMINAMATH_GPT_min_value_expression_l1705_170558

theorem min_value_expression (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) : 
  ∃ z, z = (x + y) / x ∧ z = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1705_170558


namespace NUMINAMATH_GPT_Diego_more_than_half_Martha_l1705_170589

theorem Diego_more_than_half_Martha (M D : ℕ) (H1 : M = 90)
  (H2 : D > M / 2)
  (H3 : M + D = 145):
  D - M / 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_Diego_more_than_half_Martha_l1705_170589


namespace NUMINAMATH_GPT_shelves_needed_l1705_170565

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end NUMINAMATH_GPT_shelves_needed_l1705_170565


namespace NUMINAMATH_GPT_part_a_part_b_l1705_170506

-- Part (a)
theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : ¬(a % 10 = b % 10) :=
by sorry

-- Part (b)
theorem part_b (a b c : ℕ)
  (h1 : (2 * a + b) % 10 = (2 * b + c) % 10)
  (h2 : (2 * b + c) % 10 = (2 * c + a) % 10)
  (h3 : (2 * c + a) % 10 = (2 * a + b) % 10) :
  (a % 10 = b % 10) ∧ (b % 10 = c % 10) ∧ (c % 10 = a % 10) :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l1705_170506


namespace NUMINAMATH_GPT_row_sum_1005_equals_20092_l1705_170567

theorem row_sum_1005_equals_20092 :
  let row := 1005
  let n := row
  let first_element := n
  let num_elements := 2 * n - 1
  let last_element := first_element + (num_elements - 1)
  let sum_row := num_elements * (first_element + last_element) / 2
  sum_row = 20092 :=
by
  sorry

end NUMINAMATH_GPT_row_sum_1005_equals_20092_l1705_170567


namespace NUMINAMATH_GPT_initial_amount_l1705_170544

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_initial_amount_l1705_170544


namespace NUMINAMATH_GPT_alice_flips_heads_probability_l1705_170572

def prob_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem alice_flips_heads_probability :
  prob_heads 8 3 (1/3 : ℚ) (2/3 : ℚ) = 1792 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_alice_flips_heads_probability_l1705_170572


namespace NUMINAMATH_GPT_problem_statement_l1705_170517

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 2013 a b α β = 5) :
  f 2014 a b α β = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1705_170517


namespace NUMINAMATH_GPT_functional_relationship_and_point_l1705_170560

noncomputable def directly_proportional (y x : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x

theorem functional_relationship_and_point :
  (∀ x y, directly_proportional y x → y = 2 * x) ∧ 
  (∀ a : ℝ, (∃ (y : ℝ), y = 3 ∧ directly_proportional y a) → a = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_functional_relationship_and_point_l1705_170560


namespace NUMINAMATH_GPT_solution_in_Quadrant_III_l1705_170524

theorem solution_in_Quadrant_III {c x y : ℝ} 
    (h1 : x - y = 4) 
    (h2 : c * x + y = 5) 
    (hx : x < 0) 
    (hy : y < 0) : 
    c < -1 := 
sorry

end NUMINAMATH_GPT_solution_in_Quadrant_III_l1705_170524


namespace NUMINAMATH_GPT_correct_answer_l1705_170563

-- Definition of the correctness condition
def indicates_number (phrase : String) : Prop :=
  (phrase = "Noun + Cardinal Number") ∨ (phrase = "the + Ordinal Number + Noun")

-- Example phrases to be evaluated
def class_first : String := "Class First"
def the_class_one : String := "the Class One"
def class_one : String := "Class One"
def first_class : String := "First Class"

-- The goal is to prove that "Class One" meets the condition
theorem correct_answer : indicates_number "Class One" :=
by {
  -- Insert detailed proof steps here, currently omitted
  sorry
}

end NUMINAMATH_GPT_correct_answer_l1705_170563


namespace NUMINAMATH_GPT_candy_probability_l1705_170562

/-- 
A jar has 15 red candies, 15 blue candies, and 10 green candies. Terry picks three candies at random,
then Mary picks three of the remaining candies at random. Calculate the probability that they get 
the same color combination, irrespective of order, expressed as a fraction $m/n,$ where $m$ and $n$ 
are relatively prime positive integers. Find $m+n.$ -/
theorem candy_probability :
  let num_red := 15
  let num_blue := 15
  let num_green := 10
  let total_candies := num_red + num_blue + num_green
  let Terry_picks := 3
  let Mary_picks := 3
  let prob_equal_comb := (118545 : ℚ) / 2192991
  let m := 118545
  let n := 2192991
  m + n = 2310536 := sorry

end NUMINAMATH_GPT_candy_probability_l1705_170562


namespace NUMINAMATH_GPT_average_weight_of_Arun_l1705_170540

def arun_opinion (w : ℝ) : Prop := 66 < w ∧ w < 72
def brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def mother_opinion (w : ℝ) : Prop := w ≤ 69

theorem average_weight_of_Arun :
  (∀ w, arun_opinion w → brother_opinion w → mother_opinion w → 
    (w = 67 ∨ w = 68 ∨ w = 69)) →
  avg_weight = 68 :=
sorry

end NUMINAMATH_GPT_average_weight_of_Arun_l1705_170540


namespace NUMINAMATH_GPT_total_sales_correct_l1705_170586

def maries_newspapers : ℝ := 275.0
def maries_magazines : ℝ := 150.0
def total_sales := maries_newspapers + maries_magazines

theorem total_sales_correct :
  total_sales = 425.0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_sales_correct_l1705_170586


namespace NUMINAMATH_GPT_circle_area_from_intersection_l1705_170539

-- Statement of the problem
theorem circle_area_from_intersection (r : ℝ) (A B : ℝ × ℝ)
  (h_circle : ∀ x y, (x + 2) ^ 2 + y ^ 2 = r ^ 2 ↔ (x, y) = A ∨ (x, y) = B)
  (h_parabola : ∀ x y, y ^ 2 = 20 * x ↔ (x, y) = A ∨ (x, y) = B)
  (h_axis_sym : A.1 = -5 ∧ B.1 = -5)
  (h_AB_dist : |A.2 - B.2| = 8) : π * r ^ 2 = 25 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_from_intersection_l1705_170539


namespace NUMINAMATH_GPT_calculation_correct_l1705_170511

theorem calculation_correct : (5 * 7 + 9 * 4 - 36 / 3 : ℤ) = 59 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1705_170511


namespace NUMINAMATH_GPT_sara_remaining_red_balloons_l1705_170592

-- Given conditions
def initial_red_balloons := 31
def red_balloons_given := 24

-- Statement to prove
theorem sara_remaining_red_balloons : (initial_red_balloons - red_balloons_given = 7) :=
by
  -- Proof can be skipped
  sorry

end NUMINAMATH_GPT_sara_remaining_red_balloons_l1705_170592


namespace NUMINAMATH_GPT_minimal_disks_needed_l1705_170550

-- Define the capacity of one disk
def disk_capacity : ℝ := 2.0

-- Define the number of files and their sizes
def num_files_0_9 : ℕ := 5
def size_file_0_9 : ℝ := 0.9

def num_files_0_8 : ℕ := 15
def size_file_0_8 : ℝ := 0.8

def num_files_0_5 : ℕ := 20
def size_file_0_5 : ℝ := 0.5

-- Total number of files
def total_files : ℕ := num_files_0_9 + num_files_0_8 + num_files_0_5

-- Proof statement: the minimal number of disks needed to store all files given their sizes and the disk capacity
theorem minimal_disks_needed : 
  ∀ (d : ℕ), 
    d = 18 → 
    total_files = 40 → 
    disk_capacity = 2.0 → 
    ((num_files_0_9 * size_file_0_9 + num_files_0_8 * size_file_0_8 + num_files_0_5 * size_file_0_5) / disk_capacity) ≤ d
  :=
by
  sorry

end NUMINAMATH_GPT_minimal_disks_needed_l1705_170550


namespace NUMINAMATH_GPT_evaluate_expression_l1705_170578

theorem evaluate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 2 → (5 * x^(y + 1) + 6 * y^(x + 1) = 231) := by 
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1705_170578


namespace NUMINAMATH_GPT_find_chemistry_marks_l1705_170590

theorem find_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℤ)
    (average_marks total_subjects : ℤ)
    (h1 : marks_english = 36)
    (h2 : marks_math = 35)
    (h3 : marks_physics = 42)
    (h4 : marks_biology = 55)
    (h5 : average_marks = 45)
    (h6 : total_subjects = 5) :
    (225 - (marks_english + marks_math + marks_physics + marks_biology)) = 57 :=
by
  sorry

end NUMINAMATH_GPT_find_chemistry_marks_l1705_170590


namespace NUMINAMATH_GPT_value_of_adams_collection_l1705_170595

theorem value_of_adams_collection (num_coins : ℕ) (coins_value : ℕ) (total_value_4coins : ℕ) (h1 : num_coins = 20) (h2 : total_value_4coins = 16) (h3 : ∀ k, k = 4 → coins_value = total_value_4coins / k) : 
  num_coins * coins_value = 80 := 
by {
  sorry
}

end NUMINAMATH_GPT_value_of_adams_collection_l1705_170595


namespace NUMINAMATH_GPT_sin_five_pi_over_six_l1705_170521

theorem sin_five_pi_over_six : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
  sorry

end NUMINAMATH_GPT_sin_five_pi_over_six_l1705_170521


namespace NUMINAMATH_GPT_collections_in_bag_l1705_170561

noncomputable def distinct_collections : ℕ :=
  let vowels := ['A', 'I', 'O']
  let consonants := ['M', 'H', 'C', 'N', 'T', 'T']
  let case1 := Nat.choose 3 2 * Nat.choose 6 3 -- when 0 or 1 T falls off
  let case2 := Nat.choose 3 2 * Nat.choose 5 1 -- when both T's fall off
  case1 + case2

theorem collections_in_bag : distinct_collections = 75 := 
  by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_collections_in_bag_l1705_170561


namespace NUMINAMATH_GPT_mrs_li_actual_birthdays_l1705_170596
   
   def is_leap_year (year : ℕ) : Prop :=
     (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
   
   def num_leap_years (start end_ : ℕ) : ℕ :=
     (start / 4 - start / 100 + start / 400) -
     (end_ / 4 - end_ / 100 + end_ / 400)
   
   theorem mrs_li_actual_birthdays : num_leap_years 1944 2011 = 16 :=
   by
     -- Calculation logic for the proof
     sorry
   
end NUMINAMATH_GPT_mrs_li_actual_birthdays_l1705_170596


namespace NUMINAMATH_GPT_three_solutions_no_solutions_2891_l1705_170536

theorem three_solutions (n : ℤ) (hpos : n > 0) (hx : ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3 * x1 * y1^2 + y1^3 = n ∧ 
    x2^3 - 3 * x2 * y2^2 + y2^3 = n ∧ 
    x3^3 - 3 * x3 * y3^2 + y3^3 = n := 
sorry

theorem no_solutions_2891 : ¬ ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end NUMINAMATH_GPT_three_solutions_no_solutions_2891_l1705_170536


namespace NUMINAMATH_GPT_joes_current_weight_l1705_170525

theorem joes_current_weight (W : ℕ) (R : ℕ) : 
  (W = 222 - 4 * R) →
  (W - 3 * R = 180) →
  W = 198 :=
by
  intros h1 h2
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_joes_current_weight_l1705_170525


namespace NUMINAMATH_GPT_geometric_sequence_inserted_product_l1705_170594

theorem geometric_sequence_inserted_product :
  ∃ (a b c : ℝ), a * b * c = 216 ∧
    (∃ (q : ℝ), 
      a = (8/3) * q ∧ 
      b = a * q ∧ 
      c = b * q ∧ 
      (8/3) * q^4 = 27/2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_inserted_product_l1705_170594


namespace NUMINAMATH_GPT_cos_B_eq_find_b_eq_l1705_170570

variable (A B C a b c : ℝ)

-- Given conditions
axiom sin_A_plus_C_eq : Real.sin (A + C) = 8 * Real.sin (B / 2) ^ 2
axiom a_plus_c : a + c = 6
axiom area_of_triangle : 1 / 2 * a * c * Real.sin B = 2

-- Proving cos B
theorem cos_B_eq :
  Real.cos B = 15 / 17 :=
sorry

-- Proving b given the area and sides condition
theorem find_b_eq :
  Real.cos B = 15 / 17 → b = 2 :=
sorry

end NUMINAMATH_GPT_cos_B_eq_find_b_eq_l1705_170570


namespace NUMINAMATH_GPT_correct_weights_swapped_l1705_170528

theorem correct_weights_swapped 
  (W X Y Z : ℝ) 
  (h1 : Z > Y) 
  (h2 : X > W) 
  (h3 : Y + Z > W + X) :
  (W, Z) = (Z, W) :=
sorry

end NUMINAMATH_GPT_correct_weights_swapped_l1705_170528
