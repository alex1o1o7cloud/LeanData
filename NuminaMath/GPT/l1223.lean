import Mathlib

namespace John_paid_total_l1223_122321

def vet_cost : ℝ := 400
def num_appointments : ℕ := 3
def insurance_cost : ℝ := 100
def coverage_rate : ℝ := 0.8

def discount : ℝ := vet_cost * coverage_rate
def discounted_visits : ℕ := num_appointments - 1
def discounted_cost : ℝ := vet_cost - discount
def total_discounted_cost : ℝ := discounted_visits * discounted_cost
def J_total : ℝ := vet_cost + total_discounted_cost + insurance_cost

theorem John_paid_total : J_total = 660 := by
  sorry

end John_paid_total_l1223_122321


namespace original_fraction_is_one_third_l1223_122302

theorem original_fraction_is_one_third (a b : ℕ) 
  (coprime_ab : Nat.gcd a b = 1) 
  (h : (a + 2) * b = 3 * a * b^2) : 
  (a = 1 ∧ b = 3) := 
by 
  sorry

end original_fraction_is_one_third_l1223_122302


namespace altered_solution_ratio_l1223_122314

theorem altered_solution_ratio (initial_bleach : ℕ) (initial_detergent : ℕ) (initial_water : ℕ) :
  initial_bleach / initial_detergent = 2 / 25 ∧
  initial_detergent / initial_water = 25 / 100 →
  (initial_detergent / initial_water) / 2 = 1 / 8 →
  initial_water = 300 →
  (300 / 8) = 37.5 := 
by 
  sorry

end altered_solution_ratio_l1223_122314


namespace sum_x_y_eq_l1223_122359

noncomputable def equation (x y : ℝ) : Prop :=
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0

theorem sum_x_y_eq (x y : ℝ) (h : equation x y) : x + y = -9 / 2 :=
by sorry

end sum_x_y_eq_l1223_122359


namespace fourth_grade_students_l1223_122368

theorem fourth_grade_students (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ) 
  (h_initial : initial_students = 35) (h_left : students_left = 10) (h_new : new_students = 10) :
  initial_students - students_left + new_students = 35 :=
by
  -- The proof goes here
  sorry

end fourth_grade_students_l1223_122368


namespace b_has_infinite_solutions_l1223_122384

noncomputable def b_value_satisfies_infinite_solutions : Prop :=
  ∃ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21 / 4

theorem b_has_infinite_solutions : b_value_satisfies_infinite_solutions :=
  sorry

end b_has_infinite_solutions_l1223_122384


namespace man_l1223_122385

theorem man's_speed_against_current :
  ∀ (V_down V_c V_m V_up : ℝ),
    (V_down = 15) →
    (V_c = 2.8) →
    (V_m = V_down - V_c) →
    (V_up = V_m - V_c) →
    V_up = 9.4 :=
by
  intros V_down V_c V_m V_up
  intros hV_down hV_c hV_m hV_up
  sorry

end man_l1223_122385


namespace fill_time_l1223_122341

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end fill_time_l1223_122341


namespace correct_expression_l1223_122377

theorem correct_expression (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : a / b = 4 / 3 := by
  sorry

end correct_expression_l1223_122377


namespace prime_pairs_l1223_122380

theorem prime_pairs (p q : ℕ) : 
  p < 2005 → q < 2005 → 
  Prime p → Prime q → 
  (q ∣ p^2 + 4) → 
  (p ∣ q^2 + 4) → 
  (p = 2 ∧ q = 2) :=
by sorry

end prime_pairs_l1223_122380


namespace combined_weight_is_correct_l1223_122392

-- Define the conditions
def elephant_weight_tons : ℕ := 3
def ton_in_pounds : ℕ := 2000
def donkey_weight_percentage : ℕ := 90

-- Convert elephant's weight to pounds
def elephant_weight_pounds : ℕ := elephant_weight_tons * ton_in_pounds

-- Calculate the donkeys's weight
def donkey_weight_pounds : ℕ := elephant_weight_pounds - (elephant_weight_pounds * donkey_weight_percentage / 100)

-- Define the combined weight
def combined_weight : ℕ := elephant_weight_pounds + donkey_weight_pounds

-- Prove the combined weight is 6600 pounds
theorem combined_weight_is_correct : combined_weight = 6600 :=
by
  sorry

end combined_weight_is_correct_l1223_122392


namespace feifei_reaches_school_at_828_l1223_122373

-- Definitions for all conditions
def start_time : Nat := 8 * 60 + 10  -- Feifei starts walking at 8:10 AM in minutes since midnight
def dog_delay : Nat := 3             -- Dog starts chasing after 3 minutes
def catch_up_200m_time : ℕ := 1      -- Time for dog to catch Feifei at 200 meters
def catch_up_400m_time : ℕ := 4      -- Time for dog to catch Feifei at 400 meters
def school_distance : ℕ := 800       -- Distance from home to school
def feifei_speed : ℕ := 2            -- assumed speed of Feifei where distance covered uniformly
def dog_speed : ℕ := 6               -- dog speed is three times Feifei's speed
def catch_times := [200, 400, 800]   -- Distances (in meters) where dog catches Feifei

-- Derived condition:
def total_travel_time : ℕ := 
  let time_for_200m := catch_up_200m_time + catch_up_200m_time;
  let time_for_400m_and_back := 2* catch_up_400m_time ;
  (time_for_200m + time_for_400m_and_back + (school_distance - 400))

-- The statement we wish to prove:
theorem feifei_reaches_school_at_828 : 
  (start_time + total_travel_time - dog_delay/2) % 60 = 28 :=
sorry

end feifei_reaches_school_at_828_l1223_122373


namespace find_absolute_value_l1223_122339

theorem find_absolute_value (h k : ℤ) (h1 : 3 * (-3)^3 - h * (-3) + k = 0) (h2 : 3 * 2^3 - h * 2 + k = 0) : |3 * h - 2 * k| = 27 :=
by
  sorry

end find_absolute_value_l1223_122339


namespace senior_ticket_cost_l1223_122309

theorem senior_ticket_cost (total_tickets : ℕ) (adult_ticket_cost : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) (senior_ticket_cost : ℕ) :
  total_tickets = 510 →
  adult_ticket_cost = 21 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  senior_ticket_cost = 15 :=
by
  sorry

end senior_ticket_cost_l1223_122309


namespace rational_includes_integers_and_fractions_l1223_122340

def is_integer (x : ℤ) : Prop := true
def is_fraction (x : ℚ) : Prop := true
def is_rational (x : ℚ) : Prop := true

theorem rational_includes_integers_and_fractions : 
  (∀ x : ℤ, is_integer x → is_rational (x : ℚ)) ∧ 
  (∀ x : ℚ, is_fraction x → is_rational x) :=
by {
  sorry -- Proof to be filled in
}

end rational_includes_integers_and_fractions_l1223_122340


namespace chip_cost_l1223_122312

theorem chip_cost 
  (calories_per_chip : ℕ)
  (chips_per_bag : ℕ)
  (cost_per_bag : ℕ)
  (desired_calories : ℕ)
  (h1 : calories_per_chip = 10)
  (h2 : chips_per_bag = 24)
  (h3 : cost_per_bag = 2)
  (h4 : desired_calories = 480) : 
  cost_per_bag * (desired_calories / (calories_per_chip * chips_per_bag)) = 4 := 
by 
  sorry

end chip_cost_l1223_122312


namespace tom_finishes_in_6_years_l1223_122367

/-- Combined program years for BS and Ph.D. -/
def BS_years : ℕ := 3
def PhD_years : ℕ := 5

/-- Total combined program time -/
def total_program_years : ℕ := BS_years + PhD_years

/-- Tom's time multiplier -/
def tom_time_multiplier : ℚ := 3 / 4

/-- Tom's total time to finish the program -/
def tom_total_time : ℚ := tom_time_multiplier * total_program_years

theorem tom_finishes_in_6_years : tom_total_time = 6 := 
by 
  -- implementation of the proof is to be filled in here
  sorry

end tom_finishes_in_6_years_l1223_122367


namespace sin_180_eq_0_l1223_122318

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l1223_122318


namespace peter_pizza_total_l1223_122320

theorem peter_pizza_total (total_slices : ℕ) (whole_slice : ℕ) (shared_slice : ℚ) (shared_parts : ℕ) :
  total_slices = 16 ∧ whole_slice = 1 ∧ shared_parts = 3 ∧ shared_slice = 1 / (total_slices * shared_parts) →
  whole_slice / total_slices + shared_slice = 1 / 12 :=
by
  sorry

end peter_pizza_total_l1223_122320


namespace find_first_offset_l1223_122388

theorem find_first_offset
  (area : ℝ)
  (diagonal : ℝ)
  (offset2 : ℝ)
  (first_offset : ℝ)
  (h_area : area = 225)
  (h_diagonal : diagonal = 30)
  (h_offset2 : offset2 = 6)
  (h_formula : area = (diagonal * (first_offset + offset2)) / 2)
  : first_offset = 9 := by
  sorry

end find_first_offset_l1223_122388


namespace line_passes_through_fixed_point_l1223_122366

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), y = k * x - k ∧ x = 1 ∧ y = 0 :=
by
  use 1
  use 0
  sorry

end line_passes_through_fixed_point_l1223_122366


namespace range_of_m_l1223_122353

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m) ^ 2 - 1

-- State the main theorem
theorem range_of_m (m : ℝ) :
  (∀ x ≤ 3, quadratic_function x m ≥ quadratic_function (x + 1) m) ↔ m ≥ 3 :=
by
  sorry

end range_of_m_l1223_122353


namespace candles_shared_equally_l1223_122357

theorem candles_shared_equally :
  ∀ (Aniyah Ambika Bree Caleb : ℕ),
  Aniyah = 6 * Ambika → Ambika = 4 → Bree = 0 → Caleb = 0 →
  (Aniyah + Ambika + Bree + Caleb) / 4 = 7 :=
by
  intros Aniyah Ambika Bree Caleb h1 h2 h3 h4
  sorry

end candles_shared_equally_l1223_122357


namespace box_volume_l1223_122396

theorem box_volume
  (L W H : ℝ)
  (h1 : L * W = 120)
  (h2 : W * H = 72)
  (h3 : L * H = 60) :
  L * W * H = 720 :=
by
  -- The proof goes here
  sorry

end box_volume_l1223_122396


namespace replacement_fraction_l1223_122364

variable (Q : ℝ) (x : ℝ)

def initial_concentration : ℝ := 0.70
def new_concentration : ℝ := 0.35
def replacement_concentration : ℝ := 0.25

theorem replacement_fraction (h1 : 0.70 * Q - 0.70 * x * Q + 0.25 * x * Q = 0.35 * Q) :
  x = 7 / 9 :=
by
  sorry

end replacement_fraction_l1223_122364


namespace solution_set_of_inequality_l1223_122333

theorem solution_set_of_inequality :
  ∀ (x : ℝ), abs (2 * x + 1) < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_of_inequality_l1223_122333


namespace sequence_general_term_l1223_122393

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

theorem sequence_general_term (h : ∀ n : ℕ, S n = 2 * n - a n) :
  ∀ n : ℕ, a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_general_term_l1223_122393


namespace smallest_positive_integer_n_l1223_122304

theorem smallest_positive_integer_n :
  ∃ (n: ℕ), n = 4 ∧ (∀ x: ℝ, (Real.sin x)^n + (Real.cos x)^n ≤ 2 / n) :=
sorry

end smallest_positive_integer_n_l1223_122304


namespace donut_selection_l1223_122350

theorem donut_selection :
  ∃ (ways : ℕ), ways = Nat.choose 8 3 ∧ ways = 56 :=
by
  sorry

end donut_selection_l1223_122350


namespace find_a_b_l1223_122351

def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

def f_derivative (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 10) (h2 : f_derivative 1 a b = 0) : a = 4 ∧ b = -11 :=
sorry

end find_a_b_l1223_122351


namespace distinct_primes_sum_reciprocal_l1223_122337

open Classical

theorem distinct_primes_sum_reciprocal (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (hineq: (1 / p : ℚ) + (1 / q) + (1 / r) ≥ 1) 
  : (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨
    (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) := 
sorry

end distinct_primes_sum_reciprocal_l1223_122337


namespace nine_digit_not_perfect_square_l1223_122355

theorem nine_digit_not_perfect_square (D : ℕ) (h1 : 100000000 ≤ D) (h2 : D < 1000000000)
  (h3 : ∀ c : ℕ, (c ∈ D.digits 10) → (c ≠ 0)) (h4 : D % 10 = 5) :
  ¬ ∃ A : ℕ, D = A ^ 2 := 
sorry

end nine_digit_not_perfect_square_l1223_122355


namespace back_wheel_revolutions_l1223_122328

theorem back_wheel_revolutions
  (front_diameter : ℝ) (back_diameter : ℝ) (front_revolutions : ℝ) (back_revolutions : ℝ)
  (front_diameter_eq : front_diameter = 28)
  (back_diameter_eq : back_diameter = 20)
  (front_revolutions_eq : front_revolutions = 50)
  (distance_eq : ∀ {d₁ d₂}, 2 * Real.pi * d₁ / 2 * front_revolutions = back_revolutions * (2 * Real.pi * d₂ / 2)) :
  back_revolutions = 70 :=
by
  have front_circumference : ℝ := 2 * Real.pi * front_diameter / 2
  have back_circumference : ℝ := 2 * Real.pi * back_diameter / 2
  have total_distance : ℝ := front_circumference * front_revolutions
  have revolutions : ℝ := total_distance / back_circumference 
  sorry

end back_wheel_revolutions_l1223_122328


namespace simplify_expression_l1223_122336

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 18 = 45 * w + 18 := by
  sorry

end simplify_expression_l1223_122336


namespace range_of_slope_exists_k_for_collinearity_l1223_122365

def line_equation (k x : ℝ) : ℝ := k * x + 1

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x + 3

noncomputable def intersect_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry  -- Assume a function that computes the intersection points (x₁, y₁) and (x₂, y₂)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v2 = (c * v1.1, c * v1.2)

theorem range_of_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0) :
  -4/3 < k ∧ k < 0 := 
sorry

theorem exists_k_for_collinearity (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0)
  (h5 : -4/3 < k ∧ k < 0) :
  collinear (2 - x₁ - x₂, -(y₁ + y₂)) (-2, 1) ↔ k = -1/2 :=
sorry


end range_of_slope_exists_k_for_collinearity_l1223_122365


namespace number_of_mandatory_questions_correct_l1223_122345

-- Definitions and conditions
def num_mandatory_questions (x : ℕ) (k : ℕ) (y : ℕ) (m : ℕ) : Prop :=
  (3 * k - 2 * (x - k) + 5 * m = 49) ∧
  (k + m = 15) ∧
  (y = 25 - x)

-- Proof statement
theorem number_of_mandatory_questions_correct :
  ∃ x k y m : ℕ, num_mandatory_questions x k y m ∧ x = 13 :=
by
  sorry

end number_of_mandatory_questions_correct_l1223_122345


namespace calculate_expression_l1223_122370

variable (x : ℝ)

theorem calculate_expression : ((3 * x)^2) * (x^2) = 9 * (x^4) := 
sorry

end calculate_expression_l1223_122370


namespace product_last_digit_l1223_122317

def last_digit (n : ℕ) : ℕ := n % 10

theorem product_last_digit :
  last_digit (3^65 * 6^59 * 7^71) = 4 :=
by
  sorry

end product_last_digit_l1223_122317


namespace center_of_circle_l1223_122381

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 10) (h4 : y2 = 7) :
  (x1 + x2) / 2 = 6 ∧ (y1 + y2) / 2 = 2 :=
by
  rw [h1, h2, h3, h4]
  constructor
  · norm_num
  · norm_num

end center_of_circle_l1223_122381


namespace stans_average_speed_l1223_122394

noncomputable def average_speed (distance1 distance2 distance3 : ℝ) (time1_hrs time1_mins time2 time3_hrs time3_mins : ℝ) : ℝ :=
  let total_distance := distance1 + distance2 + distance3
  let total_time := time1_hrs + time1_mins / 60 + time2 + time3_hrs + time3_mins / 60
  total_distance / total_time

theorem stans_average_speed  :
  average_speed 350 420 330 5 40 7 5 30 = 60.54 :=
by
  -- sorry block indicates missing proof
  sorry

end stans_average_speed_l1223_122394


namespace total_marbles_l1223_122346

-- Define the number of marbles Mary has
def marblesMary : Nat := 9 

-- Define the number of marbles Joan has
def marblesJoan : Nat := 3 

-- Theorem to prove the total number of marbles
theorem total_marbles : marblesMary + marblesJoan = 12 := 
by sorry

end total_marbles_l1223_122346


namespace smallest_digit_divisible_by_9_l1223_122379

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (0 ≤ d ∧ d < 10) ∧ (∃ k : ℕ, 26 + d = 9 * k) ∧ d = 1 :=
by
  sorry

end smallest_digit_divisible_by_9_l1223_122379


namespace gcd_A_B_l1223_122311

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B : Int.gcd A B = 1 := 
by
  -- Conditions
  have h1 : A = 1989^1990 - 1988^1990 := rfl
  have h2 : B = 1989^1989 - 1988^1989 := rfl
  -- Conclusion
  sorry

end gcd_A_B_l1223_122311


namespace length_of_DE_l1223_122378

-- Given conditions
variables (AB DE : ℝ) (area_projected area_ABC : ℝ)

-- Hypotheses
def base_length (AB : ℝ) : Prop := AB = 15
def projected_area_ratio (area_projected area_ABC : ℝ) : Prop := area_projected = 0.25 * area_ABC
def parallel_lines (DE AB : ℝ) : Prop := ∀ x : ℝ, DE = 0.5 * AB

-- The theorem to prove
theorem length_of_DE (h1 : base_length AB) (h2 : projected_area_ratio area_projected area_ABC) (h3 : parallel_lines DE AB) : DE = 7.5 :=
by
  sorry

end length_of_DE_l1223_122378


namespace number_of_factors_27648_l1223_122349

-- Define the number in question
def n : ℕ := 27648

-- State the prime factorization
def n_prime_factors : Nat := 2^10 * 3^3

-- State the theorem to be proven
theorem number_of_factors_27648 : 
  ∃ (f : ℕ), 
  (f = (10+1) * (3+1)) ∧ (f = 44) :=
by
  -- Placeholder for the proof
  sorry

end number_of_factors_27648_l1223_122349


namespace probability_approx_l1223_122395

noncomputable def circumscribed_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * R^3

noncomputable def single_sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R / 3)^3

noncomputable def total_spheres_volume (R : ℝ) : ℝ :=
  6 * single_sphere_volume R

noncomputable def probability_inside_spheres (R : ℝ) : ℝ :=
  total_spheres_volume R / circumscribed_sphere_volume R

theorem probability_approx (R : ℝ) (hR : R > 0) : 
  abs (probability_inside_spheres R - 0.053) < 0.001 := sorry

end probability_approx_l1223_122395


namespace balance_balls_l1223_122372

noncomputable def green_weight := (9 : ℝ) / 4
noncomputable def yellow_weight := (7 : ℝ) / 3
noncomputable def white_weight := (3 : ℝ) / 2

theorem balance_balls (B : ℝ) : 
  5 * green_weight * B + 4 * yellow_weight * B + 3 * white_weight * B = (301 / 12) * B :=
by
  sorry

end balance_balls_l1223_122372


namespace pond_water_after_evaporation_l1223_122397

theorem pond_water_after_evaporation 
  (I R D : ℕ) 
  (h_initial : I = 250)
  (h_evaporation_rate : R = 1)
  (h_days : D = 50) : 
  I - (R * D) = 200 := 
by 
  sorry

end pond_water_after_evaporation_l1223_122397


namespace price_reduction_correct_eqn_l1223_122371

theorem price_reduction_correct_eqn (x : ℝ) :
  120 * (1 - x)^2 = 85 :=
sorry

end price_reduction_correct_eqn_l1223_122371


namespace rate_of_mixed_oil_l1223_122323

/-- If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 67 per litre,
    then the rate of the mixed oil per litre is Rs. 55.67. --/
theorem rate_of_mixed_oil : 
  let volume1 := 10
  let price1 := 50
  let volume2 := 5
  let price2 := 67
  let total_cost := (volume1 * price1) + (volume2 * price2)
  let total_volume := volume1 + volume2
  (total_cost / total_volume : ℝ) = 55.67 :=
by
  sorry

end rate_of_mixed_oil_l1223_122323


namespace find_x_for_parallel_vectors_l1223_122301

-- Definitions for the given conditions
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The proof statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : parallel a (b x)) : x = 6 :=
  sorry

end find_x_for_parallel_vectors_l1223_122301


namespace area_T_is_34_l1223_122310

/-- Define the dimensions of the large rectangle -/
def width_rect : ℕ := 10
def height_rect : ℕ := 4
/-- Define the dimensions of the removed section -/
def width_removed : ℕ := 6
def height_removed : ℕ := 1

/-- Calculate the area of the large rectangle -/
def area_rect : ℕ := width_rect * height_rect

/-- Calculate the area of the removed section -/
def area_removed : ℕ := width_removed * height_removed

/-- Calculate the area of the "T" shape -/
def area_T : ℕ := area_rect - area_removed

/-- To prove that the area of the T-shape is 34 square units -/
theorem area_T_is_34 : area_T = 34 := 
by {
  sorry
}

end area_T_is_34_l1223_122310


namespace employee_pay_l1223_122358

variable (X Y : ℝ)

theorem employee_pay (h1: X + Y = 572) (h2: X = 1.2 * Y) : Y = 260 :=
by
  sorry

end employee_pay_l1223_122358


namespace range_of_m_l1223_122374

open Real

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m
def q (m : ℝ) : Prop := (2 - m) > 0

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → 1 ≤ m ∧ m < 2 :=
by
  sorry

end range_of_m_l1223_122374


namespace initial_southwards_distance_l1223_122325

-- Define a structure that outlines the journey details
structure Journey :=
  (southwards : ℕ) 
  (westwards1 : ℕ := 10)
  (northwards : ℕ := 20)
  (westwards2 : ℕ := 20) 
  (home_distance : ℕ := 30)

-- Main theorem statement without proof
theorem initial_southwards_distance (j : Journey) : j.southwards + j.northwards = j.home_distance → j.southwards = 10 := by
  intro h
  sorry

end initial_southwards_distance_l1223_122325


namespace compare_neg_one_neg_sqrt_two_l1223_122315

theorem compare_neg_one_neg_sqrt_two : -1 > -Real.sqrt 2 :=
  by
    sorry

end compare_neg_one_neg_sqrt_two_l1223_122315


namespace midpoint_sum_eq_six_l1223_122360

theorem midpoint_sum_eq_six :
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  (midpoint_x + midpoint_y) = 6 :=
by
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  sorry

end midpoint_sum_eq_six_l1223_122360


namespace calculate_value_l1223_122399

theorem calculate_value (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 :=
by
  sorry

end calculate_value_l1223_122399


namespace heaviest_lightest_difference_l1223_122354

-- Define 4 boxes' weights
variables {a b c d : ℕ}

-- Define given pairwise weights
axiom w1 : a + b = 22
axiom w2 : a + c = 23
axiom w3 : c + d = 30
axiom w4 : b + d = 29

-- Define the inequality among the weights
axiom h1 : a < b
axiom h2 : b < c
axiom h3 : c < d

-- Prove the heaviest box is 7 kg heavier than the lightest
theorem heaviest_lightest_difference : d - a = 7 :=
by sorry

end heaviest_lightest_difference_l1223_122354


namespace plates_added_before_topple_l1223_122335

theorem plates_added_before_topple (init_plates add_first add_total : ℕ) (h : init_plates = 27) (h1 : add_first = 37) (h2 : add_total = 83) : 
  add_total - (init_plates + add_first) = 19 :=
by
  -- proof goes here
  sorry

end plates_added_before_topple_l1223_122335


namespace different_distributions_l1223_122344

def arrangement_methods (students teachers: Finset ℕ) : ℕ :=
  students.card.factorial * (students.card - 1).factorial * ((students.card - 1) - 1).factorial

theorem different_distributions :
  ∀ (students teachers : Finset ℕ), 
  students.card = 3 ∧ teachers.card = 3 →
  arrangement_methods students teachers = 72 :=
by sorry

end different_distributions_l1223_122344


namespace necessarily_true_statement_l1223_122383

-- Define the four statements as propositions
def Statement1 (d : ℕ) : Prop := d = 2
def Statement2 (d : ℕ) : Prop := d ≠ 3
def Statement3 (d : ℕ) : Prop := d = 5
def Statement4 (d : ℕ) : Prop := d % 2 = 0

-- The main theorem stating that given one of the statements is false, Statement3 is necessarily true
theorem necessarily_true_statement (d : ℕ) 
  (h1 : Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ ¬ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ ¬ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ ¬ Statement2 d ∧ Statement3 d ∧ Statement4 d)):
  Statement2 d :=
sorry

end necessarily_true_statement_l1223_122383


namespace six_lines_regions_l1223_122306

def number_of_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1) / 2)

theorem six_lines_regions (h1 : 6 > 0) : 
    number_of_regions 6 = 22 :=
by 
  -- Use the formula for calculating number of regions:
  -- number_of_regions n = 1 + n + (n * (n - 1) / 2)
  sorry

end six_lines_regions_l1223_122306


namespace tan_alpha_sub_pi_over_8_l1223_122348

theorem tan_alpha_sub_pi_over_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (Real.pi / 8)) :
  Real.tan (α - Real.pi / 8) = (5 * Real.sqrt 2 + 1) / 49 :=
by sorry

end tan_alpha_sub_pi_over_8_l1223_122348


namespace fifty_percent_of_number_l1223_122322

-- Define the given condition
def given_condition (x : ℝ) : Prop :=
  0.6 * x = 42

-- Define the statement we need to prove
theorem fifty_percent_of_number (x : ℝ) (h : given_condition x) : 0.5 * x = 35 := by
  sorry

end fifty_percent_of_number_l1223_122322


namespace quadratic_root_l1223_122343

theorem quadratic_root (k : ℝ) (h : (1:ℝ)^2 - 3 * (1 : ℝ) - k = 0) : k = -2 :=
sorry

end quadratic_root_l1223_122343


namespace correct_statement_l1223_122324

-- Definitions
def certain_event (P : ℝ → Prop) : Prop := P 1
def impossible_event (P : ℝ → Prop) : Prop := P 0
def uncertain_event (P : ℝ → Prop) : Prop := ∀ p, 0 < p ∧ p < 1 → P p

-- Theorem to prove
theorem correct_statement (P : ℝ → Prop) :
  (certain_event P ∧ impossible_event P ∧ uncertain_event P) →
  (∀ p, P p → p = 1)
:= by
  sorry

end correct_statement_l1223_122324


namespace tomatoes_on_each_plant_l1223_122307

/-- Andy harvests all the tomatoes from 18 plants that have a certain number of tomatoes each.
    He dries half the tomatoes and turns a third of the remainder into marinara sauce. He has
    42 tomatoes left. Prove that the number of tomatoes on each plant is 7.  -/
theorem tomatoes_on_each_plant (T : ℕ) (h1 : ∀ n, n = 18 * T)
  (h2 : ∀ m, m = (18 * T) / 2)
  (h3 : ∀ k, k = m / 3)
  (h4 : ∀ final, final = m - k ∧ final = 42) : T = 7 :=
by
  sorry

end tomatoes_on_each_plant_l1223_122307


namespace range_of_a_l1223_122319

-- Assuming all necessary imports and definitions are included

variable {R : Type} [LinearOrderedField R]

def satisfies_conditions (f : R → R) (a : R) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x y, 1 ≤ x → x < y → f x < f y) ∧
  (∀ x, (1/2 : R) ≤ x ∧ x ≤ 1 → f (a * x) < f (x - 1))

theorem range_of_a (f : R → R) (a : R) :
  satisfies_conditions f a → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l1223_122319


namespace range_of_a_l1223_122326

theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0) → a < 5 := 
by sorry

end range_of_a_l1223_122326


namespace sum_max_min_interval_l1223_122391

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem sum_max_min_interval (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1) :
  let M := max (f a) (f b)
  let m := min (f a) (f b)
  M + m = 6 :=
by
  rw [h₁, h₂]
  let M := max (f (-1)) (f 1)
  let m := min (f (-1)) (f 1)
  sorry

end sum_max_min_interval_l1223_122391


namespace marble_problem_l1223_122361

variable (A V M : ℕ)

theorem marble_problem
  (h1 : A + 5 = V - 5)
  (h2 : V + 2 * (A + 5) = A - 2 * (A + 5) + M) :
  M = 10 :=
sorry

end marble_problem_l1223_122361


namespace constant_term_expansion_l1223_122362

theorem constant_term_expansion :
  (∃ c : ℤ, ∀ x : ℝ, (2 * x - 1 / x) ^ 4 = c * x^0) ∧ c = 24 :=
by
  sorry

end constant_term_expansion_l1223_122362


namespace number_is_375_l1223_122375

theorem number_is_375 (x : ℝ) (h : (40 / 100) * x = (30 / 100) * 50) : x = 37.5 :=
sorry

end number_is_375_l1223_122375


namespace range_of_t_l1223_122376

theorem range_of_t (a b t : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) 
    (h_ineq : 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1 / 2):
    t = Real.sqrt 2 / 2 :=
sorry

end range_of_t_l1223_122376


namespace cubic_sum_identity_l1223_122300

theorem cubic_sum_identity (x y z : ℝ) (h1 : x + y + z = 15) (h2 : xy + yz + zx = 34) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 1845 :=
by
  sorry

end cubic_sum_identity_l1223_122300


namespace find_k_l1223_122390

-- Define the lines as given in the problem
def line1 (k : ℝ) (x y : ℝ) : Prop := k * x + (1 - k) * y - 3 = 0
def line2 (k : ℝ) (x y : ℝ) : Prop := (k - 1) * x + (2 * k + 3) * y - 2 = 0

-- Define the condition for perpendicular lines
def perpendicular (k : ℝ) : Prop :=
  let slope1 := -k / (1 - k)
  let slope2 := -(k - 1) / (2 * k + 3)
  slope1 * slope2 = -1

-- Problem statement: Prove that the lines are perpendicular implies k == 1 or k == -3
theorem find_k (k : ℝ) : perpendicular k → (k = 1 ∨ k = -3) :=
sorry

end find_k_l1223_122390


namespace chewbacca_pack_size_l1223_122334

/-- Given Chewbacca has 20 pieces of cherry gum and 30 pieces of grape gum,
if losing one pack of cherry gum keeps the ratio of cherry to grape gum the same
as when finding 5 packs of grape gum, determine the number of pieces x in each 
complete pack of gum. We show that x = 14. -/
theorem chewbacca_pack_size :
  ∃ (x : ℕ), (20 - x) * (30 + 5 * x) = 20 * 30 ∧ ∀ (y : ℕ), (20 - y) * (30 + 5 * y) = 600 → y = 14 :=
by
  sorry

end chewbacca_pack_size_l1223_122334


namespace number_of_red_pencils_l1223_122329

theorem number_of_red_pencils (B R G : ℕ) (h1 : B + R + G = 20) (h2 : B = 6 * G) (h3 : R < B) : R = 6 :=
by
  sorry

end number_of_red_pencils_l1223_122329


namespace base_of_parallelogram_l1223_122338

theorem base_of_parallelogram (A h b : ℝ) (hA : A = 960) (hh : h = 16) :
  A = h * b → b = 60 :=
by
  sorry

end base_of_parallelogram_l1223_122338


namespace group_A_percentage_l1223_122316

/-!
In an examination, there are 100 questions divided into 3 groups A, B, and C such that each group contains at least one question. 
Each question in group A carries 1 mark, each question in group B carries 2 marks, and each question in group C carries 3 marks. 
It is known that:
- Group B contains 23 questions
- Group C contains 1 question.
Prove that the percentage of the total marks that the questions in group A carry is 60.8%.
-/

theorem group_A_percentage :
  ∃ (a b c : ℕ), b = 23 ∧ c = 1 ∧ (a + b + c = 100) ∧ ((a * 1) + (b * 2) + (c * 3) = 125) ∧ ((a : ℝ) / 125 * 100 = 60.8) :=
by
  sorry

end group_A_percentage_l1223_122316


namespace hyperbola_foci_coords_l1223_122332

theorem hyperbola_foci_coords :
  ∀ x y, (x^2) / 8 - (y^2) / 17 = 1 → (x, y) = (5, 0) ∨ (x, y) = (-5, 0) :=
by
  sorry

end hyperbola_foci_coords_l1223_122332


namespace tan_arctan_five_twelfths_l1223_122342

theorem tan_arctan_five_twelfths : Real.tan (Real.arctan (5 / 12)) = 5 / 12 :=
by
  sorry

end tan_arctan_five_twelfths_l1223_122342


namespace original_costs_l1223_122308

theorem original_costs (P_old P_second_oldest : ℝ) (h1 : 0.9 * P_old = 1800) (h2 : 0.85 * P_second_oldest = 900) :
  P_old + P_second_oldest = 3058.82 :=
by sorry

end original_costs_l1223_122308


namespace smallest_n_common_factor_l1223_122313

theorem smallest_n_common_factor :
  ∃ n : ℤ, n > 0 ∧ (gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 10 :=
by
  sorry

end smallest_n_common_factor_l1223_122313


namespace original_candle_length_l1223_122363

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (h_current : current_length = 48) (h_factor : factor = 1.33) :
  (current_length * factor = 63.84) :=
by
  -- The proof goes here
  sorry

end original_candle_length_l1223_122363


namespace largest_integer_is_222_l1223_122347

theorem largest_integer_is_222
  (a b c d : ℤ)
  (h_distinct : a < b ∧ b < c ∧ c < d)
  (h_mean : (a + b + c + d) / 4 = 72)
  (h_min_a : a ≥ 21) 
  : d = 222 :=
sorry

end largest_integer_is_222_l1223_122347


namespace max_tan2alpha_l1223_122387

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 2)
variable (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.tan (α + β) = 2 * Real.tan β)

theorem max_tan2alpha : 
    Real.tan (2 * α) = 4 * Real.sqrt 2 / 7 := 
by 
  sorry

end max_tan2alpha_l1223_122387


namespace simplify_and_evaluate_expr_l1223_122327

variables (a b : Int)

theorem simplify_and_evaluate_expr (ha : a = 1) (hb : b = -2) : 
  2 * (3 * a^2 * b - a * b^2) - 3 * (-a * b^2 + a^2 * b - 1) = 1 :=
by
  sorry

end simplify_and_evaluate_expr_l1223_122327


namespace point_in_fourth_quadrant_l1223_122352

theorem point_in_fourth_quadrant (m n : ℝ) (h₁ : m < 0) (h₂ : n > 0) : 
  2 * n - m > 0 ∧ -n + m < 0 := by
  sorry

end point_in_fourth_quadrant_l1223_122352


namespace curve_is_line_l1223_122386

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b c : ℝ), a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 ∧
  (a, b, c) = (-1, 2, -1) := sorry

end curve_is_line_l1223_122386


namespace jack_reads_books_in_a_year_l1223_122305

/-- If Jack reads 9 books per day, how many books can he read in a year (365 days)? -/
theorem jack_reads_books_in_a_year (books_per_day : ℕ) (days_per_year : ℕ) (books_per_year : ℕ) (h1 : books_per_day = 9) (h2 : days_per_year = 365) : books_per_year = 3285 :=
by
  sorry

end jack_reads_books_in_a_year_l1223_122305


namespace integer_solution_unique_l1223_122389

theorem integer_solution_unique (n : ℤ) : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 3) ↔ n = 5 :=
by
  sorry

end integer_solution_unique_l1223_122389


namespace children_got_on_bus_l1223_122369

theorem children_got_on_bus (initial_children total_children children_added : ℕ) 
  (h_initial : initial_children = 64) 
  (h_total : total_children = 78) : 
  children_added = total_children - initial_children :=
by
  sorry

end children_got_on_bus_l1223_122369


namespace fractions_are_integers_l1223_122303

theorem fractions_are_integers (a b c : ℤ) (h : ∃ k : ℤ, (a * b / c) + (a * c / b) + (b * c / a) = k) :
  ∃ k1 k2 k3 : ℤ, (a * b / c) = k1 ∧ (a * c / b) = k2 ∧ (b * c / a) = k3 :=
by
  sorry

end fractions_are_integers_l1223_122303


namespace min_x_given_conditions_l1223_122356

theorem min_x_given_conditions :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (100 : ℚ) / 151 < y / x ∧ y / x < (200 : ℚ) / 251 ∧ x = 3 :=
by
  sorry

end min_x_given_conditions_l1223_122356


namespace alcohol_water_ratio_l1223_122331

theorem alcohol_water_ratio
  (V p q : ℝ)
  (hV : V > 0)
  (hp : p > 0)
  (hq : q > 0) :
  let total_alcohol := 3 * V * (p / (p + 1)) + V * (q / (q + 1))
  let total_water := 3 * V * (1 / (p + 1)) + V * (1 / (q + 1))
  total_alcohol / total_water = (3 * p * (q + 1) + q * (p + 1)) / (3 * (q + 1) + (p + 1)) :=
sorry

end alcohol_water_ratio_l1223_122331


namespace keith_picked_p_l1223_122382

-- Definitions of the given conditions
def p_j : ℕ := 46  -- Jason's pears
def p_m : ℕ := 12  -- Mike's pears
def p_t : ℕ := 105 -- Total pears picked

-- The theorem statement
theorem keith_picked_p : p_t - (p_j + p_m) = 47 := by
  -- Proof part will be handled later
  sorry

end keith_picked_p_l1223_122382


namespace parallelogram_perimeter_l1223_122398

def perimeter_of_parallelogram (a b : ℝ) : ℝ :=
  2 * (a + b)

theorem parallelogram_perimeter
  (side1 side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_side2 : side2 = 12) :
  perimeter_of_parallelogram side1 side2 = 60 := 
by
  sorry

end parallelogram_perimeter_l1223_122398


namespace area_in_square_yards_l1223_122330

/-
  Given:
  - length of the classroom in feet
  - width of the classroom in feet

  Prove that the area required to cover the classroom in square yards is 30. 
-/

def classroom_length_feet : ℕ := 15
def classroom_width_feet : ℕ := 18
def feet_to_yard (feet : ℕ) : ℕ := feet / 3

theorem area_in_square_yards :
  let length_yards := feet_to_yard classroom_length_feet
  let width_yards := feet_to_yard classroom_width_feet
  length_yards * width_yards = 30 :=
by
  sorry

end area_in_square_yards_l1223_122330
