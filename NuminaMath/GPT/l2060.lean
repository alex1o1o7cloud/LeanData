import Mathlib

namespace NUMINAMATH_GPT_slower_train_speed_l2060_206075

-- Defining the conditions

def length_of_each_train := 80 -- in meters
def faster_train_speed := 52 -- in km/hr
def time_to_pass := 36 -- in seconds

-- Main statement: 
theorem slower_train_speed (v : ℝ) : 
    let relative_speed := (faster_train_speed - v) * (1000 / 3600) -- converting relative speed from km/hr to m/s
    let total_distance := 2 * length_of_each_train
    let speed_equals_distance_over_time := total_distance / time_to_pass 
    (relative_speed = speed_equals_distance_over_time) -> v = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_slower_train_speed_l2060_206075


namespace NUMINAMATH_GPT_equation_of_parallel_line_l2060_206026

theorem equation_of_parallel_line 
  (l : ℝ → ℝ) 
  (passes_through : l 0 = 7) 
  (parallel_to : ∀ x : ℝ, l x = -4 * x + (l 0)) :
  ∀ x : ℝ, l x = -4 * x + 7 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l2060_206026


namespace NUMINAMATH_GPT_A_beats_B_by_40_meters_l2060_206061

-- Definitions based on conditions
def distance_A := 1000 -- Distance in meters
def time_A := 240      -- Time in seconds
def time_diff := 10      -- Time difference in seconds

-- Intermediate calculations
def velocity_A : ℚ := distance_A / time_A
def time_B := time_A + time_diff
def velocity_B : ℚ := distance_A / time_B

-- Distance B covers in 240 seconds
def distance_B_in_240 : ℚ := velocity_B * time_A

-- Proof goal
theorem A_beats_B_by_40_meters : (distance_A - distance_B_in_240 = 40) :=
by
  -- Insert actual steps to prove here
  sorry

end NUMINAMATH_GPT_A_beats_B_by_40_meters_l2060_206061


namespace NUMINAMATH_GPT_hourly_rate_is_7_l2060_206037

-- Define the fixed fee, the total payment, and the number of hours
def fixed_fee : ℕ := 17
def total_payment : ℕ := 80
def num_hours : ℕ := 9

-- Define the function calculating the hourly rate based on the given conditions
def hourly_rate (fixed_fee total_payment num_hours : ℕ) : ℕ :=
  (total_payment - fixed_fee) / num_hours

-- Prove that the hourly rate is 7 dollars per hour
theorem hourly_rate_is_7 :
  hourly_rate fixed_fee total_payment num_hours = 7 := 
by 
  -- proof is skipped
  sorry

end NUMINAMATH_GPT_hourly_rate_is_7_l2060_206037


namespace NUMINAMATH_GPT_sarah_age_l2060_206002

variable (s m : ℕ)

theorem sarah_age (h1 : s = m - 18) (h2 : s + m = 50) : s = 16 :=
by {
  -- The proof will go here
  sorry
}

end NUMINAMATH_GPT_sarah_age_l2060_206002


namespace NUMINAMATH_GPT_find_number_l2060_206027

theorem find_number (x : ℝ) (h : 7 * x = 50.68) : x = 7.24 :=
sorry

end NUMINAMATH_GPT_find_number_l2060_206027


namespace NUMINAMATH_GPT_min_value_collinear_l2060_206016

theorem min_value_collinear (x y : ℝ) (h₁ : 2 * x + 3 * y = 3) (h₂ : 0 < x) (h₃ : 0 < y) : 
  (3 / x + 2 / y) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_collinear_l2060_206016


namespace NUMINAMATH_GPT_cost_of_fencing_correct_l2060_206034

noncomputable def cost_of_fencing (d : ℝ) (r : ℝ) : ℝ :=
  Real.pi * d * r

theorem cost_of_fencing_correct : cost_of_fencing 30 5 = 471 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_correct_l2060_206034


namespace NUMINAMATH_GPT_log_product_eq_one_sixth_log_y_x_l2060_206072

variable (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

theorem log_product_eq_one_sixth_log_y_x :
  (Real.log x ^ 2 / Real.log (y ^ 5)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 4)) *
  (Real.log (x ^ 4) / Real.log (y ^ 3)) *
  (Real.log (y ^ 5) / Real.log (x ^ 3)) *
  (Real.log (x ^ 3) / Real.log (y ^ 4)) = 
  (1 / 6) * (Real.log x / Real.log y) := 
sorry

end NUMINAMATH_GPT_log_product_eq_one_sixth_log_y_x_l2060_206072


namespace NUMINAMATH_GPT_min_pos_solution_eqn_l2060_206099

theorem min_pos_solution_eqn (x : ℝ) (h : (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 25) : x = 7 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_pos_solution_eqn_l2060_206099


namespace NUMINAMATH_GPT_total_worksheets_l2060_206080

theorem total_worksheets (worksheets_graded : ℕ) (problems_per_worksheet : ℕ) (problems_remaining : ℕ)
  (h1 : worksheets_graded = 7)
  (h2 : problems_per_worksheet = 2)
  (h3 : problems_remaining = 14): 
  worksheets_graded + (problems_remaining / problems_per_worksheet) = 14 := 
by 
  sorry

end NUMINAMATH_GPT_total_worksheets_l2060_206080


namespace NUMINAMATH_GPT_tile_ratio_l2060_206032

theorem tile_ratio (original_black_tiles : ℕ) (original_white_tiles : ℕ) (original_width : ℕ) (original_height : ℕ) (border_width : ℕ) (border_height : ℕ) :
  original_black_tiles = 10 ∧ original_white_tiles = 22 ∧ original_width = 8 ∧ original_height = 4 ∧ border_width = 2 ∧ border_height = 2 →
  (original_black_tiles + ( (original_width + 2 * border_width) * (original_height + 2 * border_height) - original_width * original_height ) ) / original_white_tiles = 19 / 11 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_tile_ratio_l2060_206032


namespace NUMINAMATH_GPT_maximized_area_using_squares_l2060_206005

theorem maximized_area_using_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
  by sorry

end NUMINAMATH_GPT_maximized_area_using_squares_l2060_206005


namespace NUMINAMATH_GPT_red_apples_count_l2060_206096

-- Definitions based on conditions
def green_apples : ℕ := 2
def yellow_apples : ℕ := 14
def total_apples : ℕ := 19

-- Definition of red apples as a theorem to be proven
theorem red_apples_count :
  green_apples + yellow_apples + red_apples = total_apples → red_apples = 3 :=
by
  -- You would need to prove this using Lean
  sorry

end NUMINAMATH_GPT_red_apples_count_l2060_206096


namespace NUMINAMATH_GPT_dice_sum_impossible_l2060_206098

theorem dice_sum_impossible (a b c d : ℕ) (h1 : a * b * c * d = 216)
  (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) 
  (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) : 
  a + b + c + d ≠ 18 :=
sorry

end NUMINAMATH_GPT_dice_sum_impossible_l2060_206098


namespace NUMINAMATH_GPT_avg_displacement_per_man_l2060_206042

-- Problem definition as per the given conditions
def num_men : ℕ := 50
def tank_length : ℝ := 40  -- 40 meters
def tank_width : ℝ := 20   -- 20 meters
def rise_in_water_level : ℝ := 0.25  -- 25 cm -> 0.25 meters

-- Given the conditions, we need to prove the average displacement per man
theorem avg_displacement_per_man :
  (tank_length * tank_width * rise_in_water_level) / num_men = 4 := by
  sorry

end NUMINAMATH_GPT_avg_displacement_per_man_l2060_206042


namespace NUMINAMATH_GPT_total_population_increase_l2060_206059
-- Import the required library

-- Define the conditions for Region A and Region B
def regionA_births_0_14 (time: ℕ) := time / 20
def regionA_births_15_64 (time: ℕ) := time / 30
def regionB_births_0_14 (time: ℕ) := time / 25
def regionB_births_15_64 (time: ℕ) := time / 35

-- Define the total number of people in each age group for both regions
def regionA_population_0_14 := 2000
def regionA_population_15_64 := 6000
def regionB_population_0_14 := 1500
def regionB_population_15_64 := 5000

-- Define the total time in seconds
def total_time := 25 * 60

-- Proof statement
theorem total_population_increase : 
  regionA_population_0_14 * regionA_births_0_14 total_time +
  regionA_population_15_64 * regionA_births_15_64 total_time +
  regionB_population_0_14 * regionB_births_0_14 total_time +
  regionB_population_15_64 * regionB_births_15_64 total_time = 227 := 
by sorry

end NUMINAMATH_GPT_total_population_increase_l2060_206059


namespace NUMINAMATH_GPT_toothpicks_15_l2060_206043

def toothpicks (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Not used, placeholder for 1-based indexing.
  | 1 => 3
  | k+1 => let p := toothpicks k
           2 + if k % 2 = 0 then 1 else 0 + p

theorem toothpicks_15 : toothpicks 15 = 38 :=
by
  sorry

end NUMINAMATH_GPT_toothpicks_15_l2060_206043


namespace NUMINAMATH_GPT_A_share_of_annual_gain_l2060_206090

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end NUMINAMATH_GPT_A_share_of_annual_gain_l2060_206090


namespace NUMINAMATH_GPT_roots_square_sum_eq_l2060_206066

theorem roots_square_sum_eq (r s t p q : ℝ) 
  (h1 : r + s + t = p) 
  (h2 : r * s + r * t + s * t = q) 
  (h3 : r * s * t = r) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
by
  sorry

end NUMINAMATH_GPT_roots_square_sum_eq_l2060_206066


namespace NUMINAMATH_GPT_students_basketball_cricket_l2060_206014

theorem students_basketball_cricket (A B: ℕ) (AB: ℕ):
  A = 12 →
  B = 8 →
  AB = 3 →
  (A + B - AB) = 17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_basketball_cricket_l2060_206014


namespace NUMINAMATH_GPT_monotonically_increasing_f_l2060_206071

open Set Filter Topology

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem monotonically_increasing_f : MonotoneOn f (Ioi 0) :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_f_l2060_206071


namespace NUMINAMATH_GPT_fgf_3_is_299_l2060_206000

def f (x : ℕ) : ℕ := 5 * x + 4
def g (x : ℕ) : ℕ := 3 * x + 2
def h : ℕ := 3

theorem fgf_3_is_299 : f (g (f h)) = 299 :=
by
  sorry

end NUMINAMATH_GPT_fgf_3_is_299_l2060_206000


namespace NUMINAMATH_GPT_more_oil_l2060_206023

noncomputable def original_price (P : ℝ) :=
  P - 0.3 * P = 70

noncomputable def amount_of_oil_before (P : ℝ) :=
  700 / P

noncomputable def amount_of_oil_after :=
  700 / 70

theorem more_oil (P : ℝ) (h1 : original_price P) :
  (amount_of_oil_after - amount_of_oil_before P) = 3 :=
  sorry

end NUMINAMATH_GPT_more_oil_l2060_206023


namespace NUMINAMATH_GPT_smallest_number_ending_in_9_divisible_by_13_l2060_206047

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end NUMINAMATH_GPT_smallest_number_ending_in_9_divisible_by_13_l2060_206047


namespace NUMINAMATH_GPT_cos_value_l2060_206082

theorem cos_value (A : ℝ) (h : Real.sin (π + A) = 1/2) : Real.cos (3*π/2 - A) = 1/2 :=
sorry

end NUMINAMATH_GPT_cos_value_l2060_206082


namespace NUMINAMATH_GPT_positive_real_inequality_l2060_206094

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_real_inequality_l2060_206094


namespace NUMINAMATH_GPT_equivalent_statements_l2060_206053

variables (P Q : Prop)

theorem equivalent_statements (h : P → Q) : 
  ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) ↔ (P → Q) := by
sorry

end NUMINAMATH_GPT_equivalent_statements_l2060_206053


namespace NUMINAMATH_GPT_parabola_constant_l2060_206033

theorem parabola_constant (b c : ℝ)
  (h₁ : -20 = 2 * (-2)^2 + b * (-2) + c)
  (h₂ : 24 = 2 * 2^2 + b * 2 + c) : 
  c = -6 := 
by 
  sorry

end NUMINAMATH_GPT_parabola_constant_l2060_206033


namespace NUMINAMATH_GPT_Anne_is_15_pounds_heavier_l2060_206008

def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52

theorem Anne_is_15_pounds_heavier : Anne_weight - Douglas_weight = 15 := by
  sorry

end NUMINAMATH_GPT_Anne_is_15_pounds_heavier_l2060_206008


namespace NUMINAMATH_GPT_positive_divisors_of_x_l2060_206077

theorem positive_divisors_of_x (x : ℕ) (h : ∀ d : ℕ, d ∣ x^3 → d = 1 ∨ d = x^3 ∨ d ∣ x^2) : (∀ d : ℕ, d ∣ x → d = 1 ∨ d = x ∨ d ∣ p) :=
by
  sorry

end NUMINAMATH_GPT_positive_divisors_of_x_l2060_206077


namespace NUMINAMATH_GPT_exponent_property_l2060_206048

theorem exponent_property : 3000 * 3000^2500 = 3000^2501 := 
by sorry

end NUMINAMATH_GPT_exponent_property_l2060_206048


namespace NUMINAMATH_GPT_isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l2060_206035

def isosceles_right_triangle_initial_leg_length (x : ℝ) (h : ℝ) : Prop :=
  x + 4 * ((x + 4) / 2) ^ 2 = x * x / 2 + 112 

def isosceles_right_triangle_legs_correct (a b : ℝ) (h : ℝ) : Prop :=
  a = 26 ∧ b = 26 * Real.sqrt 2

theorem isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm :
  ∃ (x : ℝ) (h : ℝ), isosceles_right_triangle_initial_leg_length x h ∧ 
                       isosceles_right_triangle_legs_correct x (x * Real.sqrt 2) h := 
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_legs_are_26_cm_and_hypotenuse_is_26_sqrt2_cm_l2060_206035


namespace NUMINAMATH_GPT_evaluate_expression_l2060_206056

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2060_206056


namespace NUMINAMATH_GPT_n_cubed_plus_two_not_divisible_by_nine_l2060_206095

theorem n_cubed_plus_two_not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ n^3 + 2) :=
sorry

end NUMINAMATH_GPT_n_cubed_plus_two_not_divisible_by_nine_l2060_206095


namespace NUMINAMATH_GPT_josie_animal_counts_l2060_206065

/-- Josie counted 80 antelopes, 34 more rabbits than antelopes, 42 fewer hyenas than 
the total number of antelopes and rabbits combined, some more wild dogs than hyenas, 
and the number of leopards was half the number of rabbits. The total number of animals 
Josie counted was 605. Prove that the difference between the number of wild dogs 
and hyenas Josie counted is 50. -/
theorem josie_animal_counts :
  ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
    antelopes = 80 ∧
    rabbits = antelopes + 34 ∧
    hyenas = (antelopes + rabbits) - 42 ∧
    leopards = rabbits / 2 ∧
    (antelopes + rabbits + hyenas + wild_dogs + leopards = 605) ∧
    wild_dogs - hyenas = 50 := 
by
  sorry

end NUMINAMATH_GPT_josie_animal_counts_l2060_206065


namespace NUMINAMATH_GPT_find_x_l2060_206092

variables (a b c d x y : ℚ)

noncomputable def modified_fraction (a b x y : ℚ) := (a + x) / (b + y)

theorem find_x (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : modified_fraction a b x y = c / d) :
  x = (b * c - a * d + y * c) / d :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2060_206092


namespace NUMINAMATH_GPT_simplify_fraction_l2060_206088

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)

theorem simplify_fraction : (1 / a) + (1 / b) - (2 * a + b) / (2 * a * b) = 1 / (2 * a) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2060_206088


namespace NUMINAMATH_GPT_negation_of_all_students_are_punctual_l2060_206091

variable (Student : Type)
variable (student : Student → Prop)
variable (punctual : Student → Prop)

theorem negation_of_all_students_are_punctual :
  ¬ (∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) := by
  sorry

end NUMINAMATH_GPT_negation_of_all_students_are_punctual_l2060_206091


namespace NUMINAMATH_GPT_penny_dime_halfdollar_same_probability_l2060_206057

def probability_same_penny_dime_halfdollar : ℚ :=
  let total_outcomes := 2 ^ 5
  let successful_outcomes := 2 * 2 * 2
  successful_outcomes / total_outcomes

theorem penny_dime_halfdollar_same_probability :
  probability_same_penny_dime_halfdollar = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_penny_dime_halfdollar_same_probability_l2060_206057


namespace NUMINAMATH_GPT_number_of_paths_l2060_206063

-- Define the coordinates and the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

def E := (0, 7)
def F := (4, 5)
def G := (9, 0)

-- Define the number of steps required for each path segment
def steps_to_F := 6
def steps_to_G := 10

-- Capture binomial coefficients for the calculated path segments
def paths_E_to_F := binomial steps_to_F 4
def paths_F_to_G := binomial steps_to_G 5

-- Prove the total number of paths from E to G through F
theorem number_of_paths : paths_E_to_F * paths_F_to_G = 3780 :=
by rw [paths_E_to_F, paths_F_to_G]; sorry

end NUMINAMATH_GPT_number_of_paths_l2060_206063


namespace NUMINAMATH_GPT_part1_part2_l2060_206054

theorem part1 : 2 * (-1)^3 - (-2)^2 / 4 + 10 = 7 := by
  sorry

theorem part2 : abs (-3) - (-6 + 4) / (-1 / 2)^3 + (-1)^2013 = -14 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2060_206054


namespace NUMINAMATH_GPT_polynomial_coeffs_identity_l2060_206055

theorem polynomial_coeffs_identity : 
  (∀ a b c : ℝ, (2 * x^4 + x^3 - 41 * x^2 + 83 * x - 45 = 
                (a * x^2 + b * x + c) * (x^2 + 4 * x + 9))
                  → a = 2 ∧ b = -7 ∧ c = -5) :=
by
  intros a b c h
  have h₁ : a = 2 := 
    sorry-- prove that a = 2
  have h₂ : b = -7 := 
    sorry-- prove that b = -7
  have h₃ : c = -5 := 
    sorry-- prove that c = -5
  exact ⟨h₁, h₂, h₃⟩

end NUMINAMATH_GPT_polynomial_coeffs_identity_l2060_206055


namespace NUMINAMATH_GPT_minimum_value_m_l2060_206064

theorem minimum_value_m (x0 : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 1| ≤ m) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_m_l2060_206064


namespace NUMINAMATH_GPT_max_ratio_of_mean_70_l2060_206041

theorem max_ratio_of_mean_70 (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hmean : (x + y) / 2 = 70) : (x / y ≤ 99 / 41) :=
sorry

end NUMINAMATH_GPT_max_ratio_of_mean_70_l2060_206041


namespace NUMINAMATH_GPT_paint_replacement_l2060_206010

theorem paint_replacement :
  ∀ (original_paint new_paint : ℝ), 
  original_paint = 100 →
  new_paint = 0.10 * (original_paint - 0.5 * original_paint) + 0.20 * (0.5 * original_paint) →
  new_paint / original_paint = 0.15 :=
by
  intros original_paint new_paint h_orig h_new
  sorry

end NUMINAMATH_GPT_paint_replacement_l2060_206010


namespace NUMINAMATH_GPT_locus_of_point_C_l2060_206015

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_isosceles_triangle (A B C : Point) : Prop := 
  let AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let AC := (A.x - C.x)^2 + (A.y - C.y)^2
  AB = AC

def circle_eqn (C : Point) : Prop :=
  C.x^2 + C.y^2 - 3 * C.x + C.y = 2

def not_points (C : Point) : Prop :=
  (C ≠ {x := 3, y := -2}) ∧ (C ≠ {x := 0, y := 1})

theorem locus_of_point_C :
  ∀ (A B C : Point),
    A = {x := 3, y := -2} →
    B = {x := 0, y := 1} →
    is_isosceles_triangle A B C →
    circle_eqn C ∧ not_points C :=
by
  intros A B C hA hB hIso
  sorry

end NUMINAMATH_GPT_locus_of_point_C_l2060_206015


namespace NUMINAMATH_GPT_range_of_a_l2060_206001

noncomputable def f (a x : ℝ) := (Real.exp x - a * x^2) 

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 ≤ x → f a x ≥ x + 1) ↔ a ∈ Set.Iic (1/2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2060_206001


namespace NUMINAMATH_GPT_tonya_payment_l2060_206087

def original_balance : ℝ := 150.00
def new_balance : ℝ := 120.00

noncomputable def payment_amount : ℝ := original_balance - new_balance

theorem tonya_payment :
  payment_amount = 30.00 :=
by
  sorry

end NUMINAMATH_GPT_tonya_payment_l2060_206087


namespace NUMINAMATH_GPT_area_of_rectangle_is_270_l2060_206083

noncomputable def side_of_square := Real.sqrt 2025

noncomputable def radius_of_circle := side_of_square

noncomputable def length_of_rectangle := (2/5 : ℝ) * radius_of_circle

noncomputable def initial_breadth_of_rectangle := (1/2 : ℝ) * length_of_rectangle + 5

noncomputable def breadth_of_rectangle := if (length_of_rectangle + initial_breadth_of_rectangle) % 3 = 0 
                                          then initial_breadth_of_rectangle 
                                          else initial_breadth_of_rectangle + 1

noncomputable def area_of_rectangle := length_of_rectangle * breadth_of_rectangle

theorem area_of_rectangle_is_270 :
  area_of_rectangle = 270 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_is_270_l2060_206083


namespace NUMINAMATH_GPT_correct_sentence_l2060_206038

-- Define an enumeration for different sentences
inductive Sentence
| A : Sentence
| B : Sentence
| C : Sentence
| D : Sentence

-- Define a function stating properties of each sentence
def sentence_property (s : Sentence) : Bool :=
  match s with
  | Sentence.A => false  -- "The chromosomes from dad are more than from mom" is false
  | Sentence.B => false  -- "The chromosomes in my cells and my brother's cells are exactly the same" is false
  | Sentence.C => true   -- "Each pair of homologous chromosomes is provided by both parents" is true
  | Sentence.D => false  -- "Each pair of homologous chromosomes in my brother's cells are the same size" is false

-- The theorem to prove that Sentence.C is the correct one
theorem correct_sentence : sentence_property Sentence.C = true :=
by
  unfold sentence_property
  rfl

end NUMINAMATH_GPT_correct_sentence_l2060_206038


namespace NUMINAMATH_GPT_perimeter_to_side_ratio_l2060_206085

variable (a b c h_a r : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < h_a ∧ 0 < r ∧ a + b > c ∧ a + c > b ∧ b + c > a)

theorem perimeter_to_side_ratio (P : ℝ) (hP : P = a + b + c) :
  P / a = h_a / r := by
  sorry

end NUMINAMATH_GPT_perimeter_to_side_ratio_l2060_206085


namespace NUMINAMATH_GPT_difference_of_areas_l2060_206078

-- Defining the side length of the square
def square_side_length : ℝ := 8

-- Defining the side lengths of the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 5

-- Defining the area functions
def area_of_square (side_length : ℝ) : ℝ := side_length * side_length
def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ := length * width

-- Stating the theorem
theorem difference_of_areas :
  area_of_square square_side_length - area_of_rectangle rectangle_length rectangle_width = 14 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_areas_l2060_206078


namespace NUMINAMATH_GPT_sqrt_of_8_l2060_206018

-- Definition of square root
def isSquareRoot (x : ℝ) (a : ℝ) : Prop := x * x = a

-- Theorem statement: The square root of 8 is ±√8
theorem sqrt_of_8 :
  ∃ x : ℝ, isSquareRoot x 8 ∧ (x = Real.sqrt 8 ∨ x = -Real.sqrt 8) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_8_l2060_206018


namespace NUMINAMATH_GPT_find_a_b_and_tangent_lines_l2060_206024

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1

theorem find_a_b_and_tangent_lines (a b : ℝ) :
  (3 * (-2 / 3)^2 + 2 * a * (-2 / 3) + b = 0) ∧
  (3 * 1^2 + 2 * a * 1 + b = 0) →
  a = -1 / 2 ∧ b = -2 ∧
  (∀ t : ℝ, f t a b = (t^3 + (a - 1 / 2) * t^2 - 2 * t + 1) → 
     (f t a b - (3 * t^2 - t - 2) * (0 - t) = 1) →
       (3 * t^2 - t - 2 = (t * (3 * (t - t))) ) → 
          ((2 * 0 + f 0 a b) = 1) ∨ (33 * 0 + 16 * 1 - 16 = 1)) :=
sorry

end NUMINAMATH_GPT_find_a_b_and_tangent_lines_l2060_206024


namespace NUMINAMATH_GPT_minimize_side_length_of_triangle_l2060_206009

-- Define a triangle with sides a, b, and c and angle C
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ) -- angle C in radians
  (area : ℝ) -- area of the triangle

-- Define the conditions for the problem
def conditions (T : Triangle) : Prop :=
  T.area > 0 ∧ T.C > 0 ∧ T.C < Real.pi

-- Define the desired result
def min_side_length (T : Triangle) : Prop :=
  T.a = T.b ∧ T.a = Real.sqrt ((2 * T.area) / Real.sin T.C)

-- The theorem to be proven
theorem minimize_side_length_of_triangle (T : Triangle) (h : conditions T) : min_side_length T :=
  sorry

end NUMINAMATH_GPT_minimize_side_length_of_triangle_l2060_206009


namespace NUMINAMATH_GPT_find_q_revolutions_per_minute_l2060_206052

variable (p_rpm : ℕ) (q_rpm : ℕ) (t : ℕ)

def revolutions_per_minute_q : Prop :=
  (p_rpm = 10) → (t = 4) → (q_rpm = (10 / 60 * 4 + 2) * 60 / 4) → (q_rpm = 120)

theorem find_q_revolutions_per_minute (p_rpm q_rpm t : ℕ) :
  revolutions_per_minute_q p_rpm q_rpm t :=
by
  unfold revolutions_per_minute_q
  sorry

end NUMINAMATH_GPT_find_q_revolutions_per_minute_l2060_206052


namespace NUMINAMATH_GPT_pradeep_passing_percentage_l2060_206070

-- Define the constants based on the conditions
def totalMarks : ℕ := 550
def marksObtained : ℕ := 200
def marksFailedBy : ℕ := 20

-- Calculate the passing marks
def passingMarks : ℕ := marksObtained + marksFailedBy

-- Define the percentage calculation as a noncomputable function
noncomputable def requiredPercentageToPass : ℚ := (passingMarks / totalMarks) * 100

-- The theorem to prove
theorem pradeep_passing_percentage :
  requiredPercentageToPass = 40 := 
sorry

end NUMINAMATH_GPT_pradeep_passing_percentage_l2060_206070


namespace NUMINAMATH_GPT_smallest_natural_number_l2060_206022

open Nat

theorem smallest_natural_number (n : ℕ) :
  (n + 1) % 4 = 0 ∧ (n + 1) % 6 = 0 ∧ (n + 1) % 10 = 0 ∧ (n + 1) % 12 = 0 →
  n = 59 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_l2060_206022


namespace NUMINAMATH_GPT_factorize_poly_l2060_206013

theorem factorize_poly (x : ℝ) : 4 * x^3 - x = x * (2 * x + 1) * (2 * x - 1) := by
  sorry

end NUMINAMATH_GPT_factorize_poly_l2060_206013


namespace NUMINAMATH_GPT_min_value_frac_sin_cos_l2060_206073

open Real

theorem min_value_frac_sin_cos (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∃ m : ℝ, (∀ x : ℝ, x = (1 / (sin α)^2 + 3 / (cos α)^2) → x ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
by
  have h_sin_cos : sin α ≠ 0 ∧ cos α ≠ 0 := sorry -- This is an auxiliary lemma in the process, a proof is required.
  sorry

end NUMINAMATH_GPT_min_value_frac_sin_cos_l2060_206073


namespace NUMINAMATH_GPT_pelicans_among_non_egrets_is_47_percent_l2060_206019

-- Definitions for the percentage of each type of bird.
def pelican_percentage : ℝ := 0.4
def cormorant_percentage : ℝ := 0.2
def egret_percentage : ℝ := 0.15
def osprey_percentage : ℝ := 0.25

-- Calculate the percentage of pelicans among the non-egret birds.
theorem pelicans_among_non_egrets_is_47_percent :
  (pelican_percentage / (1 - egret_percentage)) * 100 = 47 :=
by
  -- Detailed proof goes here
  sorry

end NUMINAMATH_GPT_pelicans_among_non_egrets_is_47_percent_l2060_206019


namespace NUMINAMATH_GPT_minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l2060_206029

-- Definition for Problem Part (a)
def box_dimensions := (3, 5, 7)
def initial_cockchafers := 3 * 5 * 7 -- or 105

-- Defining the theorem for part (a)
theorem minimum_empty_cells_face_move (d : (ℕ × ℕ × ℕ)) (n : ℕ) :
  d = box_dimensions →
  n = initial_cockchafers →
  ∃ k ≥ 1, k = 1 :=
by
  intros hdim hn
  sorry

-- Definition for Problem Part (b)
def row_odd_cells := 2 * 5 * 7  
def row_even_cells := 1 * 5 * 7  

-- Defining the theorem for part (b)
theorem minimum_empty_cells_diagonal_move (r_odd r_even : ℕ) :
  r_odd = row_odd_cells →
  r_even = row_even_cells →
  ∃ m ≥ 35, m = 35 :=
by
  intros ho he
  sorry

end NUMINAMATH_GPT_minimum_empty_cells_face_move_minimum_empty_cells_diagonal_move_l2060_206029


namespace NUMINAMATH_GPT_range_of_function_l2060_206097

theorem range_of_function : 
  ∀ y : ℝ, (∃ x : ℝ, y = x / (1 + x^2)) ↔ (-1 / 2 ≤ y ∧ y ≤ 1 / 2) := 
by sorry

end NUMINAMATH_GPT_range_of_function_l2060_206097


namespace NUMINAMATH_GPT_f_even_function_l2060_206074

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even_function : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  show f x = f (-x)
  sorry

end NUMINAMATH_GPT_f_even_function_l2060_206074


namespace NUMINAMATH_GPT_cost_of_milkshake_is_correct_l2060_206028

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end NUMINAMATH_GPT_cost_of_milkshake_is_correct_l2060_206028


namespace NUMINAMATH_GPT_find_value_x_y_cube_l2060_206021

variables (x y k c m : ℝ)

theorem find_value_x_y_cube
  (h1 : x^3 * y^3 = k)
  (h2 : 1 / x^3 + 1 / y^3 = c)
  (h3 : x + y = m) :
  (x + y)^3 = c * k + 3 * k^(1/3) * m :=
by
  sorry

end NUMINAMATH_GPT_find_value_x_y_cube_l2060_206021


namespace NUMINAMATH_GPT_find_y_plus_one_over_y_l2060_206044

variable (y : ℝ)

theorem find_y_plus_one_over_y (h : y^3 + (1/y)^3 = 110) : y + 1/y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_plus_one_over_y_l2060_206044


namespace NUMINAMATH_GPT_solution_set_for_a1_find_a_if_min_value_is_4_l2060_206036

noncomputable def f (a x : ℝ) : ℝ := |2 * x - 1| + |a * x - 5|

theorem solution_set_for_a1 : 
  { x : ℝ | f 1 x ≥ 9 } = { x : ℝ | x ≤ -1 ∨ x > 5 } :=
sorry

theorem find_a_if_min_value_is_4 :
  ∃ a : ℝ, (0 < a ∧ a < 5) ∧ (∀ x : ℝ, f a x ≥ 4) ∧ (∃ x : ℝ, f a x = 4) ∧ a = 2 :=
sorry

end NUMINAMATH_GPT_solution_set_for_a1_find_a_if_min_value_is_4_l2060_206036


namespace NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_7_l2060_206003

theorem remainder_of_3_pow_2023_mod_7 :
  3 ^ 2023 % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_7_l2060_206003


namespace NUMINAMATH_GPT_harold_car_payment_l2060_206025

variables (C : ℝ)

noncomputable def harold_income : ℝ := 2500
noncomputable def rent : ℝ := 700
noncomputable def groceries : ℝ := 50
noncomputable def remaining_after_retirement : ℝ := 1300

-- Harold's utility cost is half his car payment
noncomputable def utilities (C : ℝ) : ℝ := C / 2

-- Harold's total expenses.
noncomputable def total_expenses (C : ℝ) : ℝ := rent + C + utilities C + groceries

-- Proving that Harold’s car payment \(C\) can be calculated with the remaining money
theorem harold_car_payment : (2500 - total_expenses C = 1300) → (C = 300) :=
by 
  sorry

end NUMINAMATH_GPT_harold_car_payment_l2060_206025


namespace NUMINAMATH_GPT_largest_angle_measures_203_l2060_206004

-- Define the angles of the hexagon
def angle1 (x : ℚ) : ℚ := x + 2
def angle2 (x : ℚ) : ℚ := 2 * x + 1
def angle3 (x : ℚ) : ℚ := 3 * x
def angle4 (x : ℚ) : ℚ := 4 * x - 1
def angle5 (x : ℚ) : ℚ := 5 * x + 2
def angle6 (x : ℚ) : ℚ := 6 * x - 2

-- Define the sum of interior angles for a hexagon
def hexagon_angle_sum : ℚ := 720

-- Prove that the largest angle is equal to 203 degrees given the conditions
theorem largest_angle_measures_203 (x : ℚ) (h : angle1 x + angle2 x + angle3 x + angle4 x + angle5 x + angle6 x = hexagon_angle_sum) :
  (6 * x - 2) = 203 := by
  sorry

end NUMINAMATH_GPT_largest_angle_measures_203_l2060_206004


namespace NUMINAMATH_GPT_balance_pitcher_with_saucers_l2060_206045

-- Define the weights of the cup (C), pitcher (P), and saucer (S)
variables (C P S : ℝ)

-- Conditions provided in the problem
axiom cond1 : 2 * C + 2 * P = 14 * S
axiom cond2 : P = C + S

-- The statement to prove
theorem balance_pitcher_with_saucers : P = 4 * S :=
by
  sorry

end NUMINAMATH_GPT_balance_pitcher_with_saucers_l2060_206045


namespace NUMINAMATH_GPT_find_abc_l2060_206062

theorem find_abc (a b c : ℝ) 
  (h1 : 2 * b = a + c)  -- a, b, c form an arithmetic sequence
  (h2 : a + b + c = 12) -- The sum of a, b, and c is 12
  (h3 : (b + 2)^2 = (a + 2) * (c + 5)) -- a+2, b+2, and c+5 form a geometric sequence
: (a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2) :=
sorry

end NUMINAMATH_GPT_find_abc_l2060_206062


namespace NUMINAMATH_GPT_fraction_transformation_l2060_206031

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : 
  (-a) / (a - b) = a / (b - a) :=
sorry

end NUMINAMATH_GPT_fraction_transformation_l2060_206031


namespace NUMINAMATH_GPT_initial_blue_balls_proof_l2060_206020

-- Define the main problem parameters and condition
def initial_jars (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :=
  total_balls = 18 ∧
  removed_blue = 3 ∧
  remaining_balls = total_balls - removed_blue ∧
  probability = 1/5 → 
  (initial_blue_balls - removed_blue) / remaining_balls = probability

-- Define the proof problem
theorem initial_blue_balls_proof (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :
  initial_jars total_balls initial_blue_balls removed_blue probability remaining_balls →
  initial_blue_balls = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_blue_balls_proof_l2060_206020


namespace NUMINAMATH_GPT_number_of_possible_outcomes_l2060_206012

theorem number_of_possible_outcomes : 
  ∃ n : ℕ, n = 30 ∧
  ∀ (total_shots successful_shots consecutive_hits : ℕ),
  total_shots = 8 ∧ successful_shots = 3 ∧ consecutive_hits = 2 →
  n = 30 := 
by
  sorry

end NUMINAMATH_GPT_number_of_possible_outcomes_l2060_206012


namespace NUMINAMATH_GPT_triangular_region_area_l2060_206040

theorem triangular_region_area :
  let x_intercept := 4
  let y_intercept := 6
  let area := (1 / 2) * x_intercept * y_intercept
  area = 12 :=
by
  sorry

end NUMINAMATH_GPT_triangular_region_area_l2060_206040


namespace NUMINAMATH_GPT_g_at_10_l2060_206086

noncomputable def g : ℕ → ℝ := sorry

axiom g_zero : g 0 = 2
axiom g_one : g 1 = 1
axiom g_func_eq (m n : ℕ) (h : m ≥ n) : 
  g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2 + 2

theorem g_at_10 : g 10 = 102 := sorry

end NUMINAMATH_GPT_g_at_10_l2060_206086


namespace NUMINAMATH_GPT_minimum_participants_l2060_206006

theorem minimum_participants (x y z n : ℕ) 
  (hx : x + 1 + 2 * x = n)
  (hy : y + 1 + 3 * y = n)
  (hz : z + 1 + 4 * z = n) :
  n = 61 :=
by sorry

end NUMINAMATH_GPT_minimum_participants_l2060_206006


namespace NUMINAMATH_GPT_onlyD_is_PythagoreanTriple_l2060_206058

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def validTripleA := ¬ isPythagoreanTriple 12 15 18
def validTripleB := isPythagoreanTriple 3 4 5 ∧ (¬ (3 = 3 ∧ 4 = 4 ∧ 5 = 5)) -- Since 0.3, 0.4, 0.5 not integers
def validTripleC := ¬ isPythagoreanTriple 15 25 30 -- Conversion of 1.5, 2.5, 3 to integers
def validTripleD := isPythagoreanTriple 12 16 20

theorem onlyD_is_PythagoreanTriple : validTripleA ∧ validTripleB ∧ validTripleC ∧ validTripleD :=
by {
  sorry
}

end NUMINAMATH_GPT_onlyD_is_PythagoreanTriple_l2060_206058


namespace NUMINAMATH_GPT_rational_sum_p_q_l2060_206030

noncomputable def x := (Real.sqrt 5 - 1) / 2

theorem rational_sum_p_q :
  ∃ (p q : ℚ), x^3 + p * x + q = 0 ∧ p + q = -1 := by
  sorry

end NUMINAMATH_GPT_rational_sum_p_q_l2060_206030


namespace NUMINAMATH_GPT_atomic_weight_of_nitrogen_l2060_206067

-- Definitions from conditions
def molecular_weight := 53.0
def hydrogen_weight := 1.008
def chlorine_weight := 35.45
def hydrogen_atoms := 4
def chlorine_atoms := 1

-- The proof goal
theorem atomic_weight_of_nitrogen : 
  53.0 - (4.0 * 1.008) - 35.45 = 13.518 :=
by
  sorry

end NUMINAMATH_GPT_atomic_weight_of_nitrogen_l2060_206067


namespace NUMINAMATH_GPT_product_of_roots_l2060_206068

theorem product_of_roots :
  (∃ r s t : ℝ, (r + s + t) = 15 ∧ (r*s + s*t + r*t) = 50 ∧ (r*s*t) = -35) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - r) * (x - s) * (x - t)) :=
sorry

end NUMINAMATH_GPT_product_of_roots_l2060_206068


namespace NUMINAMATH_GPT_find_c_l2060_206079

theorem find_c (a b c : ℝ) (h1 : ∃ x y : ℝ, x = a * (y - 2)^2 + 3 ∧ (x,y) = (3,2))
  (h2 : (1 : ℝ) = a * ((4 : ℝ) - 2)^2 + 3) : c = 1 :=
sorry

end NUMINAMATH_GPT_find_c_l2060_206079


namespace NUMINAMATH_GPT_multiply_add_square_l2060_206017

theorem multiply_add_square : 15 * 28 + 42 * 15 + 15^2 = 1275 :=
by
  sorry

end NUMINAMATH_GPT_multiply_add_square_l2060_206017


namespace NUMINAMATH_GPT_total_protest_days_l2060_206007

theorem total_protest_days (d1 : ℕ) (increase_percent : ℕ) (d2 : ℕ) (total_days : ℕ) (h1 : d1 = 4) (h2 : increase_percent = 25) (h3 : d2 = d1 + (d1 * increase_percent / 100)) : total_days = d1 + d2 → total_days = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_protest_days_l2060_206007


namespace NUMINAMATH_GPT_base10_to_base7_of_804_l2060_206093

def base7 (n : ℕ) : ℕ :=
  let d3 := n / 343
  let r3 := n % 343
  let d2 := r3 / 49
  let r2 := r3 % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

theorem base10_to_base7_of_804 :
  base7 804 = 2226 :=
by
  -- Proof to be filled in.
  sorry

end NUMINAMATH_GPT_base10_to_base7_of_804_l2060_206093


namespace NUMINAMATH_GPT_sequence_problem_l2060_206076

theorem sequence_problem
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : 1 + a1 + a1 = a1 + a1)
  (h2 : b1 * b1 = b2)
  (h3 : 4 = b2 * b2):
  (a1 + a2) / b2 = 2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_sequence_problem_l2060_206076


namespace NUMINAMATH_GPT_find_parallel_and_perpendicular_lines_through_A_l2060_206081

def point_A : ℝ × ℝ := (2, 2)

def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 20 = 0

def parallel_line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 14 = 0

def perpendicular_line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y - 2 = 0

theorem find_parallel_and_perpendicular_lines_through_A :
  (∀ x y, line_l x y → parallel_line_l1 x y) ∧
  (∀ x y, line_l x y → perpendicular_line_l2 x y) :=
by
  sorry

end NUMINAMATH_GPT_find_parallel_and_perpendicular_lines_through_A_l2060_206081


namespace NUMINAMATH_GPT_coffee_consumption_l2060_206069

variables (h w g : ℝ)

theorem coffee_consumption (k : ℝ) 
  (H1 : ∀ h w g, h * g = k * w)
  (H2 : h = 8 ∧ g = 4.5 ∧ w = 2)
  (H3 : h = 4 ∧ w = 3) : g = 13.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_coffee_consumption_l2060_206069


namespace NUMINAMATH_GPT_angus_caught_4_more_l2060_206049

theorem angus_caught_4_more (
  angus ollie patrick: ℕ
) (
  h1: ollie = angus - 7
) (
  h2: ollie = 5
) (
  h3: patrick = 8
) : (angus - patrick) = 4 := 
sorry

end NUMINAMATH_GPT_angus_caught_4_more_l2060_206049


namespace NUMINAMATH_GPT_square_side_length_l2060_206011

theorem square_side_length (A : ℝ) (π : ℝ) (s : ℝ) (area_circle_eq : A = 100)
  (area_circle_eq_perimeter_square : A = 4 * s) : s = 25 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l2060_206011


namespace NUMINAMATH_GPT_modular_units_l2060_206051

theorem modular_units (U N S : ℕ) 
  (h1 : N = S / 4)
  (h2 : (S : ℚ) / (S + U * N) = 0.14285714285714285) : 
  U = 24 :=
by
  sorry

end NUMINAMATH_GPT_modular_units_l2060_206051


namespace NUMINAMATH_GPT_simplified_equation_equivalent_l2060_206060

theorem simplified_equation_equivalent  (x : ℝ) :
    (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end NUMINAMATH_GPT_simplified_equation_equivalent_l2060_206060


namespace NUMINAMATH_GPT_age_of_b_l2060_206039

theorem age_of_b (A B C : ℕ) (h₁ : (A + B + C) / 3 = 25) (h₂ : (A + C) / 2 = 29) : B = 17 := 
by
  sorry

end NUMINAMATH_GPT_age_of_b_l2060_206039


namespace NUMINAMATH_GPT_find_5_minus_a_l2060_206089

-- Define the problem conditions as assumptions
variable (a b : ℤ)
variable (h1 : 5 + a = 6 - b)
variable (h2 : 3 + b = 8 + a)

-- State the theorem we want to prove
theorem find_5_minus_a : 5 - a = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_5_minus_a_l2060_206089


namespace NUMINAMATH_GPT_line_point_relation_l2060_206084

theorem line_point_relation (x1 y1 x2 y2 a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * x1 + b1 * y1 = c1)
  (h2 : a2 * x2 + b2 * y2 = c2)
  (h3 : a1 + b1 = c1)
  (h4 : a2 + b2 = 2 * c2)
  (h5 : dist (x1, y1) (x2, y2) ≥ (Real.sqrt 2) / 2) :
  c1 / a1 + a2 / c2 = 3 := 
sorry

end NUMINAMATH_GPT_line_point_relation_l2060_206084


namespace NUMINAMATH_GPT_jon_buys_2_coffees_each_day_l2060_206046

-- Define the conditions
def cost_per_coffee : ℕ := 2
def total_spent : ℕ := 120
def days_in_april : ℕ := 30

-- Define the total number of coffees bought
def total_coffees_bought : ℕ := total_spent / cost_per_coffee

-- Prove that Jon buys 2 coffees each day
theorem jon_buys_2_coffees_each_day : total_coffees_bought / days_in_april = 2 := by
  sorry

end NUMINAMATH_GPT_jon_buys_2_coffees_each_day_l2060_206046


namespace NUMINAMATH_GPT_probability_of_same_length_segments_l2060_206050

-- Definitions directly from the conditions
def total_elements : ℕ := 6 + 9

-- Probability calculations
def probability_same_length : ℚ :=
  ((6 / total_elements) * (5 / (total_elements - 1))) + 
  ((9 / total_elements) * (8 / (total_elements - 1)))

-- Proof statement
theorem probability_of_same_length_segments : probability_same_length = 17 / 35 := by
  unfold probability_same_length
  sorry

end NUMINAMATH_GPT_probability_of_same_length_segments_l2060_206050
