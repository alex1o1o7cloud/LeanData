import Mathlib

namespace largest_k_value_l1843_184378

theorem largest_k_value (a b c d : ℕ) (k : ℝ)
  (h1 : a + b = c + d)
  (h2 : 2 * (a * b) = c * d)
  (h3 : a ≥ b) :
  (∀ k', (∀ a b (h1_b : a + b = c + d)
              (h2_b : 2 * a * b = c * d)
              (h3_b : a ≥ b), (a : ℝ) / (b : ℝ) ≥ k') → k' ≤ k) → k = 3 + 2 * Real.sqrt 2 :=
sorry

end largest_k_value_l1843_184378


namespace factorization_correct_l1843_184362

theorem factorization_correct (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 :=
by
  sorry

end factorization_correct_l1843_184362


namespace minimum_value_f_on_neg_ab_l1843_184386

theorem minimum_value_f_on_neg_ab
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : a < b)
  (h2 : b < 0)
  (odd_f : ∀ x : ℝ, f (-x) = -f (x))
  (decreasing_f : ∀ x y : ℝ, 0 < x ∧ x < y → f y < f x)
  (range_ab : ∀ y : ℝ, a ≤ y ∧ y ≤ b → -3 ≤ f y ∧ f y ≤ 4) :
  ∀ x : ℝ, -b ≤ x ∧ x ≤ -a → -4 ≤ f x ∧ f x ≤ 3 := 
sorry

end minimum_value_f_on_neg_ab_l1843_184386


namespace jim_needs_more_miles_l1843_184364

-- Define the conditions
def totalMiles : ℕ := 1200
def drivenMiles : ℕ := 923

-- Define the question and the correct answer
def remainingMiles : ℕ := totalMiles - drivenMiles

-- The theorem statement
theorem jim_needs_more_miles : remainingMiles = 277 :=
by
  -- This will contain the proof which is to be done later
  sorry

end jim_needs_more_miles_l1843_184364


namespace second_dog_miles_per_day_l1843_184334

-- Definitions describing conditions
section DogWalk
variable (total_miles_week : ℕ)
variable (first_dog_miles_day : ℕ)
variable (days_in_week : ℕ)

-- Assert conditions given in the problem
def condition1 := total_miles_week = 70
def condition2 := first_dog_miles_day = 2
def condition3 := days_in_week = 7

-- The theorem to prove
theorem second_dog_miles_per_day
  (h1 : condition1 total_miles_week)
  (h2 : condition2 first_dog_miles_day)
  (h3 : condition3 days_in_week) :
  (total_miles_week - days_in_week * first_dog_miles_day) / days_in_week = 8 :=
sorry
end DogWalk

end second_dog_miles_per_day_l1843_184334


namespace michael_total_earnings_l1843_184317

-- Define the cost of large paintings and small paintings
def large_painting_cost : ℕ := 100
def small_painting_cost : ℕ := 80

-- Define the number of large and small paintings sold
def large_paintings_sold : ℕ := 5
def small_paintings_sold : ℕ := 8

-- Calculate Michael's total earnings
def total_earnings : ℕ := (large_painting_cost * large_paintings_sold) + (small_painting_cost * small_paintings_sold)

-- Prove: Michael's total earnings are 1140 dollars
theorem michael_total_earnings : total_earnings = 1140 := by
  sorry

end michael_total_earnings_l1843_184317


namespace no_such_functions_exist_l1843_184398

theorem no_such_functions_exist (f g : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1) :=
sorry

end no_such_functions_exist_l1843_184398


namespace sin_double_angle_l1843_184397

theorem sin_double_angle (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end sin_double_angle_l1843_184397


namespace perpendicular_k_parallel_k_l1843_184303

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the scalar multiple operations and vector operations
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 + v₂.1, v₂.2 + v₂.2)
def sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 - v₂.1, v₂.2 - v₂.2)
def dot (v₁ v₂ : ℝ × ℝ) : ℝ := (v₁.1 * v₂.1 + v₁.2 * v₂.2)

-- Problem 1: If k*a + b is perpendicular to a - 3*b, then k = 19
theorem perpendicular_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  dot vak amb = 0 → k = 19 := sorry

-- Problem 2: If k*a + b is parallel to a - 3*b, then k = -1/3 and they are in opposite directions
theorem parallel_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  ∃ m : ℝ, vak = smul m amb ∧ m < 0 → k = -1/3 := sorry

end perpendicular_k_parallel_k_l1843_184303


namespace m_div_x_l1843_184371

variable (a b k : ℝ)
variable (ha : a = 4 * k)
variable (hb : b = 5 * k)
variable (k_pos : k > 0)

def x := a * 1.25
def m := b * 0.20

theorem m_div_x : m / x = 1 / 5 := by
  sorry

end m_div_x_l1843_184371


namespace most_likely_wins_l1843_184346

theorem most_likely_wins {N : ℕ} (h : N > 0) :
  let p := 1 / 2
  let n := 2 * N
  let E := n * p
  E = N := 
by
  sorry

end most_likely_wins_l1843_184346


namespace total_copies_in_half_hour_l1843_184355

-- Definitions of the machine rates and their time segments.
def machine1_rate := 35 -- copies per minute
def machine2_rate := 65 -- copies per minute
def machine3_rate1 := 50 -- copies per minute for the first 15 minutes
def machine3_rate2 := 80 -- copies per minute for the next 15 minutes
def machine4_rate1 := 90 -- copies per minute for the first 10 minutes
def machine4_rate2 := 60 -- copies per minute for the next 20 minutes

-- Time intervals for different machines
def machine3_time1 := 15 -- minutes
def machine3_time2 := 15 -- minutes
def machine4_time1 := 10 -- minutes
def machine4_time2 := 20 -- minutes

-- Proof statement
theorem total_copies_in_half_hour : 
  (machine1_rate * 30) + 
  (machine2_rate * 30) + 
  ((machine3_rate1 * machine3_time1) + (machine3_rate2 * machine3_time2)) + 
  ((machine4_rate1 * machine4_time1) + (machine4_rate2 * machine4_time2)) = 
  7050 :=
by 
  sorry

end total_copies_in_half_hour_l1843_184355


namespace average_age_of_4_students_l1843_184349

theorem average_age_of_4_students (avg_age_15 : ℕ) (num_students_15 : ℕ)
    (avg_age_10 : ℕ) (num_students_10 : ℕ) (age_15th_student : ℕ) :
    avg_age_15 = 15 ∧ num_students_15 = 15 ∧ avg_age_10 = 16 ∧ num_students_10 = 10 ∧ age_15th_student = 9 → 
    (56 / 4 = 14) := by
  sorry

end average_age_of_4_students_l1843_184349


namespace max_sum_of_four_integers_with_product_360_l1843_184391

theorem max_sum_of_four_integers_with_product_360 :
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ a * b * c * d = 360 ∧ a + b + c + d = 66 :=
sorry

end max_sum_of_four_integers_with_product_360_l1843_184391


namespace smoke_diagram_total_height_l1843_184345

theorem smoke_diagram_total_height : 
  ∀ (h1 h2 h3 h4 h5 : ℕ),
    h1 < h2 ∧ h2 < h3 ∧ h3 < h4 ∧ h4 < h5 ∧ 
    (h2 - h1 = 2) ∧ (h3 - h2 = 2) ∧ (h4 - h3 = 2) ∧ (h5 - h4 = 2) ∧ 
    (h5 = h1 + h2) → 
    h1 + h2 + h3 + h4 + h5 = 50 := 
by 
  sorry

end smoke_diagram_total_height_l1843_184345


namespace count_rhombuses_in_large_triangle_l1843_184320

-- Definitions based on conditions
def large_triangle_side_length : ℕ := 10
def small_triangle_side_length : ℕ := 1
def small_triangle_count : ℕ := 100
def rhombuses_of_8_triangles := 84

-- Problem statement in Lean 4
theorem count_rhombuses_in_large_triangle :
  ∀ (large_side small_side small_count : ℕ),
  large_side = large_triangle_side_length →
  small_side = small_triangle_side_length →
  small_count = small_triangle_count →
  (∃ (rhombus_count : ℕ), rhombus_count = rhombuses_of_8_triangles) :=
by
  intros large_side small_side small_count h_large h_small h_count
  use 84
  sorry

end count_rhombuses_in_large_triangle_l1843_184320


namespace omega_value_l1843_184314

noncomputable def f (ω : ℝ) (k : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x - Real.pi / 6) + k

theorem omega_value (ω k : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, f ω k x ≤ f ω k (Real.pi / 3)) → ω = 8 :=
by sorry

end omega_value_l1843_184314


namespace bags_needed_l1843_184360

-- Definitions for the condition
def total_sugar : ℝ := 35.5
def bag_capacity : ℝ := 0.5

-- Theorem statement to solve the problem
theorem bags_needed : total_sugar / bag_capacity = 71 := 
by 
  sorry

end bags_needed_l1843_184360


namespace ratio_of_segments_l1843_184351

theorem ratio_of_segments (a b : ℕ) (ha : a = 200) (hb : b = 40) : a / b = 5 :=
by sorry

end ratio_of_segments_l1843_184351


namespace translation_of_civilisation_l1843_184390

def translation (word : String) (translation : String) : Prop :=
translation = "civilization"

theorem translation_of_civilisation (word : String) :
  word = "civilisation" → translation word "civilization" :=
by sorry

end translation_of_civilisation_l1843_184390


namespace hyperbola_asymptotes_l1843_184306

noncomputable def eccentricity_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b = Real.sqrt 15 * a) : Prop :=
  ∀ (x y : ℝ), (y = (Real.sqrt 15) * x) ∨ (y = -(Real.sqrt 15) * x)

theorem hyperbola_asymptotes (a : ℝ) (h₁ : a > 0) :
  eccentricity_asymptotes a (Real.sqrt 15 * a) h₁ (by simp) :=
sorry

end hyperbola_asymptotes_l1843_184306


namespace A_and_B_mutually_exclusive_l1843_184335

-- Definitions of events based on conditions
def A (a : ℕ) : Prop := a = 3
def B (a : ℕ) : Prop := a = 4

-- Define mutually exclusive
def mutually_exclusive (P Q : ℕ → Prop) : Prop :=
  ∀ a, P a → Q a → false

-- Problem statement: Prove A and B are mutually exclusive.
theorem A_and_B_mutually_exclusive :
  mutually_exclusive A B :=
sorry

end A_and_B_mutually_exclusive_l1843_184335


namespace time_spent_watching_tv_excluding_breaks_l1843_184365

-- Definitions based on conditions
def total_hours_watched : ℕ := 5
def breaks : List ℕ := [10, 15, 20, 25]

-- Conversion constants
def minutes_per_hour : ℕ := 60

-- Derived definitions
def total_minutes_watched : ℕ := total_hours_watched * minutes_per_hour
def total_break_minutes : ℕ := breaks.sum

-- The main theorem
theorem time_spent_watching_tv_excluding_breaks :
  total_minutes_watched - total_break_minutes = 230 := by
  sorry

end time_spent_watching_tv_excluding_breaks_l1843_184365


namespace geometric_product_is_geometric_l1843_184318

theorem geometric_product_is_geometric (q : ℝ) (a : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  ∀ n, (a n) * (a (n + 1)) = (q^2) * (a (n - 1) * a n) := by
  sorry

end geometric_product_is_geometric_l1843_184318


namespace solve_quadratic_completing_square_l1843_184332

theorem solve_quadratic_completing_square (x : ℝ) :
  (2 * x^2 - 4 * x - 1 = 0) ↔ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
by
  sorry

end solve_quadratic_completing_square_l1843_184332


namespace skillful_hands_wire_cut_l1843_184301

theorem skillful_hands_wire_cut :
  ∃ x : ℕ, (1000 = 15 * x) ∧ (1040 = 15 * x) ∧ x = 66 :=
by
  sorry

end skillful_hands_wire_cut_l1843_184301


namespace Ann_end_blocks_l1843_184350

-- Define blocks Ann initially has and finds
def initialBlocksAnn : ℕ := 9
def foundBlocksAnn : ℕ := 44

-- Define blocks Ann ends with
def finalBlocksAnn : ℕ := initialBlocksAnn + foundBlocksAnn

-- The proof goal
theorem Ann_end_blocks : finalBlocksAnn = 53 := by
  -- Use sorry to skip the proof
  sorry

end Ann_end_blocks_l1843_184350


namespace missing_number_l1843_184382

theorem missing_number (m x : ℕ) (h : 744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + m + x = 750 * 10)  
  (hx : x = 755) : m = 805 := by 
  sorry

end missing_number_l1843_184382


namespace common_difference_is_3_l1843_184340

variable {a : ℕ → ℤ} {d : ℤ}

-- Definitions of conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition_1 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 11 = 24

def condition_2 (a : ℕ → ℤ) : Prop :=
  a 4 = 3

-- Theorem statement to prove
theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ)
  (ha : is_arithmetic_sequence a d)
  (hc1 : condition_1 a d)
  (hc2 : condition_2 a) :
  d = 3 := by
  sorry

end common_difference_is_3_l1843_184340


namespace combined_pumps_fill_time_l1843_184383

theorem combined_pumps_fill_time (small_pump_time large_pump_time : ℝ) (h1 : small_pump_time = 4) (h2 : large_pump_time = 1/2) : 
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  (1 / combined_rate) = 4 / 9 :=
by
  -- Definitions of rates
  let small_pump_rate := 1 / small_pump_time
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := small_pump_rate + large_pump_rate
  
  -- Using placeholder for the proof.
  sorry

end combined_pumps_fill_time_l1843_184383


namespace find_breadth_of_plot_l1843_184372

-- Define the conditions
def length_of_plot (breadth : ℝ) := 3 * breadth
def area_of_plot := 2028

-- Define what we want to prove
theorem find_breadth_of_plot (breadth : ℝ) (h1 : length_of_plot breadth * breadth = area_of_plot) : breadth = 26 :=
sorry

end find_breadth_of_plot_l1843_184372


namespace find_actual_number_of_children_l1843_184379

theorem find_actual_number_of_children (B C : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 420)) : C = 840 := 
by
  sorry

end find_actual_number_of_children_l1843_184379


namespace ae_length_l1843_184313

theorem ae_length (AB CD AC AE : ℝ) (h: 2 * AE + 3 * AE = 34): 
  AE = 34 / 5 := by
  -- Proof steps will go here
  sorry

end ae_length_l1843_184313


namespace prime_iff_satisfies_condition_l1843_184329

def satisfies_condition (n : ℕ) : Prop :=
  if n = 2 then True
  else if 2 < n then ∀ k : ℕ, 2 ≤ k ∧ k < n → ¬ (k ∣ n)
  else False

theorem prime_iff_satisfies_condition (n : ℕ) : Prime n ↔ satisfies_condition n := by
  sorry

end prime_iff_satisfies_condition_l1843_184329


namespace equation_of_line_l1843_184380

theorem equation_of_line (P : ℝ × ℝ) (A : ℝ) (m : ℝ) (hP : P = (-3, 4)) (hA : A = 3) (hm : m = 1) :
  ((2 * P.1 + 3 * P.2 - 6 = 0) ∨ (8 * P.1 + 3 * P.2 + 12 = 0)) :=
by 
  sorry

end equation_of_line_l1843_184380


namespace sqrt_x_minus_5_meaningful_iff_x_ge_5_l1843_184344

theorem sqrt_x_minus_5_meaningful_iff_x_ge_5 (x : ℝ) : (∃ y : ℝ, y^2 = x - 5) ↔ (x ≥ 5) :=
sorry

end sqrt_x_minus_5_meaningful_iff_x_ge_5_l1843_184344


namespace proof1_proof2_proof3_proof4_l1843_184316

noncomputable def calc1 : ℝ := 3.21 - 1.05 - 1.95
noncomputable def calc2 : ℝ := 15 - (2.95 + 8.37)
noncomputable def calc3 : ℝ := 14.6 * 2 - 0.6 * 2
noncomputable def calc4 : ℝ := 0.25 * 1.25 * 32

theorem proof1 : calc1 = 0.21 := by
  sorry

theorem proof2 : calc2 = 3.68 := by
  sorry

theorem proof3 : calc3 = 28 := by
  sorry

theorem proof4 : calc4 = 10 := by
  sorry

end proof1_proof2_proof3_proof4_l1843_184316


namespace sum_of_two_digit_numbers_with_gcd_lcm_l1843_184315

theorem sum_of_two_digit_numbers_with_gcd_lcm (x y : ℕ) (h1 : Nat.gcd x y = 8) (h2 : Nat.lcm x y = 96)
  (h3 : 10 ≤ x ∧ x < 100) (h4 : 10 ≤ y ∧ y < 100) : x + y = 56 :=
sorry

end sum_of_two_digit_numbers_with_gcd_lcm_l1843_184315


namespace solve_for_x_l1843_184352

theorem solve_for_x (x : ℝ) (h : (x+10) / (x-4) = (x-3) / (x+6)) : x = -48 / 23 :=
by
  sorry

end solve_for_x_l1843_184352


namespace gcd_greatest_possible_value_l1843_184330

noncomputable def Sn (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_greatest_possible_value (n : ℕ) (hn : 0 < n) : 
  Nat.gcd (3 * Sn n) (n + 1) = 1 :=
sorry

end gcd_greatest_possible_value_l1843_184330


namespace fraction_speed_bus_train_l1843_184356

theorem fraction_speed_bus_train :
  let speed_train := 16 * 5
  let speed_bus := 480 / 8
  let speed_train_prop := speed_train = 80
  let speed_bus_prop := speed_bus = 60
  speed_bus / speed_train = 3 / 4 :=
by
  sorry

end fraction_speed_bus_train_l1843_184356


namespace nell_baseball_cards_l1843_184310

theorem nell_baseball_cards 
  (ace_cards_now : ℕ) 
  (extra_baseball_cards : ℕ) 
  (B : ℕ) : 
  ace_cards_now = 55 →
  extra_baseball_cards = 123 →
  B = ace_cards_now + extra_baseball_cards →
  B = 178 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end nell_baseball_cards_l1843_184310


namespace horizontal_asymptote_l1843_184357

def numerator (x : ℝ) : ℝ :=
  15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2

def denominator (x : ℝ) : ℝ :=
  5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1

noncomputable def rational_function (x : ℝ) : ℝ :=
  numerator x / denominator x

theorem horizontal_asymptote :
  ∃ y : ℝ, (∀ x : ℝ, x ≠ 0 → rational_function x = y) ↔ y = 3 :=
by
  sorry

end horizontal_asymptote_l1843_184357


namespace complex_ratio_identity_l1843_184368

variable {x y : ℂ}

theorem complex_ratio_identity :
  ( (x + y) / (x - y) - (x - y) / (x + y) = 3 ) →
  ( (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600) :=
by
  sorry

end complex_ratio_identity_l1843_184368


namespace rainfall_on_Monday_l1843_184358

theorem rainfall_on_Monday (rain_on_Tuesday : ℝ) (difference : ℝ) (rain_on_Tuesday_eq : rain_on_Tuesday = 0.2) (difference_eq : difference = 0.7) :
  ∃ rain_on_Monday : ℝ, rain_on_Monday = rain_on_Tuesday + difference := 
sorry

end rainfall_on_Monday_l1843_184358


namespace four_digit_numbers_divisible_by_11_and_5_with_sum_12_l1843_184367

theorem four_digit_numbers_divisible_by_11_and_5_with_sum_12:
  ∀ a b c d : ℕ, (a + b + c + d = 12) ∧ ((a + c) - (b + d)) % 11 = 0 ∧ (d = 0 ∨ d = 5) →
  false :=
by
  intro a b c d
  intro h
  sorry

end four_digit_numbers_divisible_by_11_and_5_with_sum_12_l1843_184367


namespace average_marks_all_students_l1843_184309

theorem average_marks_all_students
  (n1 n2 : ℕ)
  (avg1 avg2 : ℕ)
  (h1 : avg1 = 40)
  (h2 : avg2 = 80)
  (h3 : n1 = 30)
  (h4 : n2 = 50) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 65 :=
by
  sorry

end average_marks_all_students_l1843_184309


namespace find_days_l1843_184326

theorem find_days
  (wages1 : ℕ) (workers1 : ℕ) (days1 : ℕ)
  (wages2 : ℕ) (workers2 : ℕ) (days2 : ℕ)
  (h1 : wages1 = 9450) (h2 : workers1 = 15) (h3 : wages2 = 9975)
  (h4 : workers2 = 19) (h5 : days2 = 5) :
  days1 = 6 := 
by
  -- Insert proof here
  sorry

end find_days_l1843_184326


namespace function_has_one_root_l1843_184393

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem function_has_one_root : ∃! x : ℝ, f x = 0 :=
by
  -- Indicate that we haven't included the proof
  sorry

end function_has_one_root_l1843_184393


namespace pictures_per_album_l1843_184377

theorem pictures_per_album (phone_pics camera_pics albums pics_per_album : ℕ)
  (h1 : phone_pics = 7) (h2 : camera_pics = 13) (h3 : albums = 5)
  (h4 : pics_per_album * albums = phone_pics + camera_pics) :
  pics_per_album = 4 :=
by
  sorry

end pictures_per_album_l1843_184377


namespace necessary_but_not_sufficient_l1843_184312

variables (P Q : Prop)
variables (p : P) (q : Q)

-- Propositions
def quadrilateral_has_parallel_and_equal_sides : Prop := P
def is_rectangle : Prop := Q

-- Necessary but not sufficient condition
theorem necessary_but_not_sufficient (h : P → Q) : ¬(Q → P) :=
by sorry

end necessary_but_not_sufficient_l1843_184312


namespace P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l1843_184374

open ProbabilityTheory

section
/-- Probability of not getting three consecutive heads -/
def P (n : ℕ) : ℚ := sorry

theorem P_3_eq_seven_eighths : P 3 = 7 / 8 := sorry

theorem P_4_ne_fifteen_sixteenths : P 4 ≠ 15 / 16 := sorry

theorem P_decreasing (n : ℕ) (h : 2 ≤ n) : P (n + 1) < P n := sorry

theorem P_recurrence (n : ℕ) (h : 4 ≤ n) : P n = (1 / 2) * P (n - 1) + (1 / 4) * P (n - 2) + (1 / 8) * P (n - 3) := sorry
end

end P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l1843_184374


namespace second_coloring_book_pictures_l1843_184333

theorem second_coloring_book_pictures (P1 P2 P_colored P_left : ℕ) (h1 : P1 = 23) (h2 : P_colored = 44) (h3 : P_left = 11) (h4 : P1 + P2 = P_colored + P_left) :
  P2 = 32 :=
by
  rw [h1, h2, h3] at h4
  linarith

end second_coloring_book_pictures_l1843_184333


namespace find_x_given_ratio_constant_l1843_184375

theorem find_x_given_ratio_constant (x y : ℚ) (k : ℚ)
  (h1 : ∀ x y, (2 * x - 5) / (y + 20) = k)
  (h2 : (2 * 7 - 5) / (6 + 20) = k)
  (h3 : y = 21) :
  x = 499 / 52 :=
by
  sorry

end find_x_given_ratio_constant_l1843_184375


namespace game_last_at_most_moves_l1843_184388

theorem game_last_at_most_moves
  (n : Nat)
  (positions : Fin n → Fin (n + 1))
  (cards : Fin n → Fin (n + 1))
  (move : (k l : Fin n) → (h1 : k < l) → (h2 : k < cards k) → (positions l = cards k) → Fin n)
  : True :=
sorry

end game_last_at_most_moves_l1843_184388


namespace actual_price_of_good_l1843_184392

theorem actual_price_of_good (P : ℝ) 
  (hp : 0.684 * P = 6500) : P = 9502.92 :=
by 
  sorry

end actual_price_of_good_l1843_184392


namespace initial_candies_proof_l1843_184366

noncomputable def initial_candies (n : ℕ) := 
  ∃ c1 c2 c3 c4 c5 : ℕ, 
    c5 = 1 ∧
    c5 = n * 1 / 6 ∧
    c4 = n * 5 / 6 ∧
    c3 = n * 4 / 5 ∧
    c2 = n * 3 / 4 ∧
    c1 = n * 2 / 3 ∧
    n = 2 * c1

theorem initial_candies_proof (n : ℕ) : initial_candies n → n = 720 :=
  by
    sorry

end initial_candies_proof_l1843_184366


namespace jar_size_is_half_gallon_l1843_184328

theorem jar_size_is_half_gallon : 
  ∃ (x : ℝ), (48 = 3 * 16) ∧ (16 + 16 * x + 16 * 0.25 = 28) ∧ x = 0.5 :=
by
  -- Implementation goes here
  sorry

end jar_size_is_half_gallon_l1843_184328


namespace parallelogram_side_lengths_l1843_184381

theorem parallelogram_side_lengths (x y : ℝ) (h1 : 3 * x + 6 = 12) (h2 : 5 * y - 2 = 10) : x + y = 22 / 5 :=
by
  sorry

end parallelogram_side_lengths_l1843_184381


namespace coeff_x2_in_PQ_is_correct_l1843_184327

variable (c : ℝ)

def P (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2 - 3 * x + 1
def Q (x : ℝ) : ℝ := 3 * x^3 + c * x^2 - 8 * x - 5

def coeff_x2 (x : ℝ) : ℝ := -20 - 2 * c

theorem coeff_x2_in_PQ_is_correct :
  (4 : ℝ) * (-5) + (-3) * c + c = -20 - 2 * c := by
  sorry

end coeff_x2_in_PQ_is_correct_l1843_184327


namespace simplify_fraction_l1843_184304

theorem simplify_fraction :
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
sorry

end simplify_fraction_l1843_184304


namespace x_minus_y_options_l1843_184323

theorem x_minus_y_options (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) :
  (x - y ≠ 2013) ∧ (x - y ≠ 2014) ∧ (x - y ≠ 2015) ∧ (x - y ≠ 2016) := 
sorry

end x_minus_y_options_l1843_184323


namespace triangle_B_is_right_triangle_l1843_184395

theorem triangle_B_is_right_triangle :
  let a := 1
  let b := 2
  let c := Real.sqrt 3
  a^2 + c^2 = b^2 :=
by
  sorry

end triangle_B_is_right_triangle_l1843_184395


namespace sum_of_three_numbers_l1843_184376

theorem sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : b = 10) 
  (h2 : (a + b + c) / 3 = a + 20) 
  (h3 : (a + b + c) / 3 = c - 25) : 
  a + b + c = 45 := 
by 
  sorry

end sum_of_three_numbers_l1843_184376


namespace inequality_solution_l1843_184354

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x - 3) ≥ 1 ↔ (x > 3 ∨ x ≤ -2) :=
by 
  sorry

end inequality_solution_l1843_184354


namespace average_weight_increase_l1843_184300

theorem average_weight_increase (A : ℝ) :
  let initial_total_weight := 10 * A
  let new_total_weight := initial_total_weight - 65 + 97
  let new_average := new_total_weight / 10
  let increase := new_average - A
  increase = 3.2 :=
by
  sorry

end average_weight_increase_l1843_184300


namespace roots_of_quadratic_l1843_184396

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic (m : ℝ) :
  let a := 1
  let b := (3 * m - 1)
  let c := (2 * m^2 - m)
  discriminant a b c ≥ 0 :=
by
  sorry

end roots_of_quadratic_l1843_184396


namespace ferris_wheel_seats_l1843_184373

theorem ferris_wheel_seats (total_people seats_capacity : ℕ) (h1 : total_people = 8) (h2 : seats_capacity = 3) : 
  Nat.ceil ((total_people : ℚ) / (seats_capacity : ℚ)) = 3 := 
by
  sorry

end ferris_wheel_seats_l1843_184373


namespace kids_all_three_activities_l1843_184342

-- Definitions based on conditions
def total_kids : ℕ := 40
def kids_tubing : ℕ := total_kids / 4
def kids_tubing_rafting : ℕ := kids_tubing / 2
def kids_tubing_rafting_kayaking : ℕ := kids_tubing_rafting / 3

-- Theorem statement: proof of the final answer
theorem kids_all_three_activities : kids_tubing_rafting_kayaking = 1 := by
  sorry

end kids_all_three_activities_l1843_184342


namespace value_of_a_plus_b_l1843_184363

-- Define the given nested fraction expression
def nested_expr := 1 + 1 / (1 + 1 / (1 + 1))

-- Define the simplified form of the expression
def simplified_form : ℚ := 13 / 8

-- The greatest common divisor condition
def gcd_condition : ℕ := Nat.gcd 13 8

-- The ultimate theorem to prove
theorem value_of_a_plus_b : 
  nested_expr = simplified_form ∧ gcd_condition = 1 → 13 + 8 = 21 := 
by 
  sorry

end value_of_a_plus_b_l1843_184363


namespace sum_of_money_l1843_184399

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_value : C = 32) :
  A + B + C = 164 :=
by
  sorry

end sum_of_money_l1843_184399


namespace ratio_a_to_c_l1843_184338

theorem ratio_a_to_c {a b c : ℚ} (h1 : a / b = 4 / 3) (h2 : b / c = 1 / 5) :
  a / c = 4 / 5 := 
sorry

end ratio_a_to_c_l1843_184338


namespace nth_term_is_4037_l1843_184385

noncomputable def arithmetic_sequence_nth_term (n : ℕ) : ℤ :=
7 + (n - 1) * 6

theorem nth_term_is_4037 {n : ℕ} : arithmetic_sequence_nth_term 673 = 4037 :=
by
  sorry

end nth_term_is_4037_l1843_184385


namespace largest_x_undefined_largest_solution_l1843_184370

theorem largest_x_undefined (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0) → x = 10 ∨ x = 1 / 10 :=
by
  sorry

theorem largest_solution (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end largest_x_undefined_largest_solution_l1843_184370


namespace _l1843_184325

noncomputable def probability_event_b_given_a : ℕ → ℕ → ℕ → ℕ × ℕ → ℚ
| zeros, ones, twos, (1, drawn_label) =>
  if drawn_label = 1 then
    (ones * (ones - 1)) / (zeros + ones + twos).choose 2
  else 0
| _, _, _, _ => 0

lemma probability_theorem :
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  (1 - 1) * (ones - 1)/(total.choose 2) = 1/7 :=
by
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  let draw_label := 1
  let event_b_given_a := probability_event_b_given_a zeros ones twos (1, draw_label)
  have pos_cases : (ones * (ones - 1))/(total.choose 2) = 1 / 7 := by sorry
  exact pos_cases

end _l1843_184325


namespace average_interest_rate_l1843_184308

theorem average_interest_rate (total_investment : ℝ) (rate1 rate2 : ℝ) (annual_return1 annual_return2 : ℝ) 
  (h1 : total_investment = 6000) 
  (h2 : rate1 = 0.035) 
  (h3 : rate2 = 0.055) 
  (h4 : annual_return1 = annual_return2) :
  (annual_return1 + annual_return2) / total_investment * 100 = 4.3 :=
by
  sorry

end average_interest_rate_l1843_184308


namespace solution_set_empty_range_l1843_184339

theorem solution_set_empty_range (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 < 0 → false) ↔ (0 ≤ a ∧ a ≤ 12) := 
sorry

end solution_set_empty_range_l1843_184339


namespace arithmetic_sequence_general_term_l1843_184347

theorem arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  (∃ d : ℚ, d = -1/2 ∧ ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = d) :=
sorry

theorem general_term (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h₁ : a 1 = 3)
  (h₂ : ∀ n ≥ 2, 2 * a n = S n * S (n - 1)) :
  ∀ n, a n = if n = 1 then 3 else 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end arithmetic_sequence_general_term_l1843_184347


namespace ladder_distance_from_wall_l1843_184387

theorem ladder_distance_from_wall (h a b : ℕ) (h_hyp : h = 13) (h_wall : a = 12) :
  a^2 + b^2 = h^2 → b = 5 :=
by
  intros h_eq
  sorry

end ladder_distance_from_wall_l1843_184387


namespace arctan_sum_eq_half_pi_l1843_184305

theorem arctan_sum_eq_half_pi (y : ℚ) :
  2 * Real.arctan (1 / 3) + Real.arctan (1 / 10) + Real.arctan (1 / 30) + Real.arctan (1 / y) = Real.pi / 2 →
  y = 547 / 620 := by
  sorry

end arctan_sum_eq_half_pi_l1843_184305


namespace radius_of_smaller_circle_l1843_184341

theorem radius_of_smaller_circle (R : ℝ) (n : ℕ) (r : ℝ) 
  (hR : R = 10) 
  (hn : n = 7) 
  (condition : 2 * R = 2 * r * n) :
  r = 10 / 7 :=
by
  sorry

end radius_of_smaller_circle_l1843_184341


namespace combined_area_percentage_l1843_184324

theorem combined_area_percentage (D_S : ℝ) (D_R : ℝ) (D_T : ℝ) (A_S A_R A_T : ℝ)
  (h1 : D_R = 0.20 * D_S)
  (h2 : D_T = 0.40 * D_R)
  (h3 : A_R = Real.pi * (D_R / 2) ^ 2)
  (h4 : A_T = Real.pi * (D_T / 2) ^ 2)
  (h5 : A_S = Real.pi * (D_S / 2) ^ 2) :
  ((A_R + A_T) / A_S) * 100 = 4.64 := by
  sorry

end combined_area_percentage_l1843_184324


namespace factor_expression_l1843_184359

theorem factor_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) :=
by
  sorry

end factor_expression_l1843_184359


namespace sum_of_reciprocals_of_squares_l1843_184389

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 11) :
  (1 / (a:ℚ)^2) + (1 / (b:ℚ)^2) = 122 / 121 :=
sorry

end sum_of_reciprocals_of_squares_l1843_184389


namespace percentage_problem_l1843_184311

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.4 * x = 45) : 0.4 * 0.3 * x = 45 :=
by
  sorry

end percentage_problem_l1843_184311


namespace range_of_a_l1843_184322

open Real

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, otimes x (x + a) < 1) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l1843_184322


namespace alloy_ratio_proof_l1843_184307

def ratio_lead_to_tin_in_alloy_a (x y : ℝ) (ha : 0 < x) (hb : 0 < y) : Prop :=
  let weight_tin_in_a := (y / (x + y)) * 170
  let weight_tin_in_b := (3 / 8) * 250
  let total_tin := weight_tin_in_a + weight_tin_in_b
  total_tin = 221.25

theorem alloy_ratio_proof (x y : ℝ) (ha : 0 < x) (hb : 0 < y) (hc : ratio_lead_to_tin_in_alloy_a x y ha hb) : y / x = 3 :=
by
  -- Proof is omitted
  sorry

end alloy_ratio_proof_l1843_184307


namespace closure_of_A_range_of_a_l1843_184353

-- Definitions for sets A and B
def A (x : ℝ) : Prop := x < -1 ∨ x > -0.5
def B (x a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

-- 1. Closure of A
theorem closure_of_A :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ -0.5) ↔ (∀ x : ℝ, A x) :=
sorry

-- 2. Range of a when A ∪ B = ℝ
theorem range_of_a (B_condition : ∀ x : ℝ, B x a) :
  (∀ a : ℝ, -1 ≤ x ∨ x ≥ -0.5) ↔ (-1.5 ≤ a ∧ a ≤ 0) :=
sorry

end closure_of_A_range_of_a_l1843_184353


namespace compare_exp_sin_ln_l1843_184302

theorem compare_exp_sin_ln :
  let a := Real.exp 0.1 - 1
  let b := Real.sin 0.1
  let c := Real.log 1.1
  c < b ∧ b < a :=
by
  sorry

end compare_exp_sin_ln_l1843_184302


namespace find_a_value_l1843_184394

theorem find_a_value (a x y : ℝ) (h1 : x = 4) (h2 : y = 5) (h3 : a * x - 2 * y = 2) : a = 3 :=
by
  sorry

end find_a_value_l1843_184394


namespace earnings_per_widget_l1843_184343

-- Defining the conditions as constants
def hours_per_week : ℝ := 40
def hourly_wage : ℝ := 12.50
def total_weekly_earnings : ℝ := 700
def widgets_produced : ℝ := 1250

-- We need to prove earnings per widget
theorem earnings_per_widget :
  (total_weekly_earnings - (hours_per_week * hourly_wage)) / widgets_produced = 0.16 := by
  sorry

end earnings_per_widget_l1843_184343


namespace muffs_bought_before_december_correct_l1843_184331

/-- Total ear muffs bought by customers in December. -/
def muffs_bought_in_december := 6444

/-- Total ear muffs bought by customers in all. -/
def total_muffs_bought := 7790

/-- Ear muffs bought before December. -/
def muffs_bought_before_december : Nat :=
  total_muffs_bought - muffs_bought_in_december

/-- Theorem stating the number of ear muffs bought before December. -/
theorem muffs_bought_before_december_correct :
  muffs_bought_before_december = 1346 :=
by
  unfold muffs_bought_before_december
  unfold total_muffs_bought
  unfold muffs_bought_in_december
  sorry

end muffs_bought_before_december_correct_l1843_184331


namespace three_point_one_two_six_as_fraction_l1843_184348

theorem three_point_one_two_six_as_fraction : (3126 / 1000 : ℚ) = 1563 / 500 := 
by 
  sorry

end three_point_one_two_six_as_fraction_l1843_184348


namespace problem_solution_l1843_184384

noncomputable def x : ℝ := 3 / 0.15
noncomputable def y : ℝ := 3 / 0.25
noncomputable def z : ℝ := 0.30 * y

theorem problem_solution : x - y + z = 11.6 := sorry

end problem_solution_l1843_184384


namespace difference_in_lengths_l1843_184361

def speed_of_first_train := 60 -- in km/hr
def time_to_cross_pole_first_train := 3 -- in seconds
def speed_of_second_train := 90 -- in km/hr
def time_to_cross_pole_second_train := 2 -- in seconds

noncomputable def length_of_first_train : ℝ := (speed_of_first_train * (5 / 18)) * time_to_cross_pole_first_train
noncomputable def length_of_second_train : ℝ := (speed_of_second_train * (5 / 18)) * time_to_cross_pole_second_train

theorem difference_in_lengths : abs (length_of_second_train - length_of_first_train) = 0.01 :=
by
  -- The full proof would be placed here.
  sorry

end difference_in_lengths_l1843_184361


namespace number_of_valid_n_l1843_184321

theorem number_of_valid_n : 
  ∃ (c : Nat), (∀ n : Nat, (n + 9) * (n - 4) * (n - 13) < 0 → n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12) ∧ c = 11 :=
by
  sorry

end number_of_valid_n_l1843_184321


namespace gas_price_l1843_184337

theorem gas_price (x : ℝ) (h1 : 10 * (x + 0.30) = 12 * x) : x + 0.30 = 1.80 := by
  sorry

end gas_price_l1843_184337


namespace one_fourth_of_8_point_8_is_fraction_l1843_184369

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l1843_184369


namespace gcd_of_factors_l1843_184319

theorem gcd_of_factors (a b : ℕ) (h : a * b = 360) : 
    ∃ n : ℕ, n = 19 :=
by
  sorry

end gcd_of_factors_l1843_184319


namespace total_difference_proof_l1843_184336

-- Definitions for the initial quantities
def initial_tomatoes : ℕ := 17
def initial_carrots : ℕ := 13
def initial_cucumbers : ℕ := 8

-- Definitions for the picked quantities
def picked_tomatoes : ℕ := 5
def picked_carrots : ℕ := 6

-- Definitions for the given away quantities
def given_away_tomatoes : ℕ := 3
def given_away_carrots : ℕ := 2

-- Definitions for the remaining quantities 
def remaining_tomatoes : ℕ := initial_tomatoes - (picked_tomatoes - given_away_tomatoes)
def remaining_carrots : ℕ := initial_carrots - (picked_carrots - given_away_carrots)

-- Definitions for the difference quantities
def difference_tomatoes : ℕ := initial_tomatoes - remaining_tomatoes
def difference_carrots : ℕ := initial_carrots - remaining_carrots

-- Definition for the total difference
def total_difference : ℕ := difference_tomatoes + difference_carrots

-- Lean Theorem Statement
theorem total_difference_proof : total_difference = 6 := by
  -- Proof is omitted
  sorry

end total_difference_proof_l1843_184336
