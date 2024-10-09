import Mathlib

namespace log_cosine_range_l2147_214790

noncomputable def log_base_three (a : ℝ) : ℝ := Real.log a / Real.log 3

theorem log_cosine_range (x : ℝ) (hx : x ∈ Set.Ioo (Real.pi / 2) (7 * Real.pi / 6)) :
    ∃ y, y = log_base_three (1 - 2 * Real.cos x) ∧ y ∈ Set.Ioc 0 1 :=
by
  sorry

end log_cosine_range_l2147_214790


namespace abs_pos_of_ne_zero_l2147_214728

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end abs_pos_of_ne_zero_l2147_214728


namespace cost_per_sqft_is_6_l2147_214714

-- Define the dimensions of the room
def room_length : ℕ := 25
def room_width : ℕ := 15
def room_height : ℕ := 12

-- Define the dimensions of the door
def door_height : ℕ := 6
def door_width : ℕ := 3

-- Define the dimensions of the windows
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3

-- Define the total cost of whitewashing
def total_cost : ℕ := 5436

-- Calculate areas
def area_one_pair_of_walls : ℕ :=
  (room_length * room_height) * 2

def area_other_pair_of_walls : ℕ :=
  (room_width * room_height) * 2

def total_wall_area : ℕ :=
  area_one_pair_of_walls + area_other_pair_of_walls

def door_area : ℕ :=
  door_height * door_width

def window_area : ℕ :=
  window_height * window_width

def total_window_area : ℕ :=
  window_area * number_of_windows

def area_to_be_whitewashed : ℕ :=
  total_wall_area - (door_area + total_window_area)

def cost_per_sqft : ℕ :=
  total_cost / area_to_be_whitewashed

-- The theorem statement proving the cost per square foot is 6
theorem cost_per_sqft_is_6 : cost_per_sqft = 6 := 
  by
  -- Proof goes here
  sorry

end cost_per_sqft_is_6_l2147_214714


namespace solve_for_t_l2147_214797

variables (V0 V g a t S : ℝ)

-- Given conditions
def velocity_eq : Prop := V = (g + a) * t + V0
def displacement_eq : Prop := S = (1/2) * (g + a) * t^2 + V0 * t

-- The theorem to prove
theorem solve_for_t (h1 : velocity_eq V0 V g a t)
                    (h2 : displacement_eq V0 g a t S) :
  t = 2 * S / (V + V0) :=
sorry

end solve_for_t_l2147_214797


namespace perpendicular_lines_l2147_214736

theorem perpendicular_lines (a : ℝ) :
  (∃ l₁ l₂ : ℝ, 2 * l₁ + l₂ + 1 = 0 ∧ l₁ + a * l₂ + 3 = 0 ∧ 2 * l₁ + 1 * l₂ + 1 * a = 0) → a = -2 :=
by
  sorry

end perpendicular_lines_l2147_214736


namespace sequence_formulas_range_of_k_l2147_214763

variable {a b : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {k : ℝ}

-- (1) Prove the general formulas for {a_n} and {b_n}
theorem sequence_formulas (h1 : ∀ n, a n + b n = 2 * n - 1)
  (h2 : ∀ n, S n = 2 * n^2 - n)
  (hS : ∀ n, a (n + 1) = S (n + 1) - S n)
  (hS1 : a 1 = S 1) :
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, b n = -2 * n + 2) :=
sorry

-- (2) Prove the range of k
theorem range_of_k (h3 : ∀ n, a n = k * 2^(n - 1))
  (h4 : ∀ n, b n = 2 * n - 1 - k * 2^(n - 1))
  (h5 : ∀ n, b (n + 1) < b n) :
  k > 2 :=
sorry

end sequence_formulas_range_of_k_l2147_214763


namespace surface_area_spherical_segment_l2147_214783

-- Definitions based on given conditions
variables {R h : ℝ}

-- The theorem to be proven
theorem surface_area_spherical_segment (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_not_intersect_sphere : h < 2 * R) :
  S = 2 * π * R * h := by
  sorry

end surface_area_spherical_segment_l2147_214783


namespace acceptable_outfits_l2147_214739

-- Definitions based on the given conditions
def shirts : Nat := 8
def pants : Nat := 5
def hats : Nat := 7
def pant_colors : List String := ["red", "black", "blue", "gray", "green"]
def shirt_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]
def hat_colors : List String := ["red", "black", "blue", "gray", "green", "purple", "white"]

-- Axiom that ensures distinct colors for pants, shirts, and hats.
axiom distinct_colors : ∀ color ∈ pant_colors, color ∈ shirt_colors ∧ color ∈ hat_colors

-- Problem statement
theorem acceptable_outfits : 
  let total_outfits := shirts * pants * hats
  let monochrome_outfits := List.length pant_colors
  let acceptable_outfits := total_outfits - monochrome_outfits
  acceptable_outfits = 275 :=
by
  sorry

end acceptable_outfits_l2147_214739


namespace initial_candies_l2147_214706

-- Define the conditions
def candies_given_older_sister : ℕ := 7
def candies_given_younger_sister : ℕ := 6
def candies_left : ℕ := 15

-- Conclude the initial number of candies
theorem initial_candies : (candies_given_older_sister + candies_given_younger_sister + candies_left) = 28 := by
  sorry

end initial_candies_l2147_214706


namespace find_n_satisfies_equation_l2147_214753

-- Definition of the problem:
def satisfies_equation (n : ℝ) : Prop := 
  (2 / (n + 1)) + (3 / (n + 1)) + (n / (n + 1)) = 4

-- The statement of the proof problem:
theorem find_n_satisfies_equation : 
  ∃ n : ℝ, satisfies_equation n ∧ n = 1/3 :=
by
  sorry

end find_n_satisfies_equation_l2147_214753


namespace minyoung_yoojung_flowers_l2147_214716

theorem minyoung_yoojung_flowers (m y : ℕ) 
(h1 : m = 4 * y) 
(h2 : m = 24) : 
m + y = 30 := 
by
  sorry

end minyoung_yoojung_flowers_l2147_214716


namespace find_t_l2147_214767

variables {m n : ℝ}
variables (t : ℝ)
variables (mv nv : ℝ)
variables (dot_m_m dot_m_n dot_n_n : ℝ)
variables (cos_theta : ℝ)

-- Define the basic assumptions
axiom non_zero_vectors : m ≠ 0 ∧ n ≠ 0
axiom magnitude_condition : mv = 2 * nv
axiom cos_condition : cos_theta = 1 / 3
axiom perpendicular_condition : dot_m_n = (mv * nv * cos_theta) ∧ (t * dot_m_n + dot_m_m = 0)

-- Utilize the conditions and prove the target
theorem find_t : t = -6 :=
sorry

end find_t_l2147_214767


namespace complex_magnitude_problem_l2147_214751

open Complex

theorem complex_magnitude_problem
  (z w : ℂ)
  (hz : abs z = 1)
  (hw : abs w = 2)
  (hzw : abs (z + w) = 3) :
  abs ((1 / z) + (1 / w)) = 3 / 2 :=
by {
  sorry
}

end complex_magnitude_problem_l2147_214751


namespace Sperner_theorem_example_l2147_214768

theorem Sperner_theorem_example :
  ∀ (S : Finset (Finset ℕ)), (S.card = 10) →
  (∀ (A B : Finset ℕ), A ∈ S → B ∈ S → A ⊆ B → A = B) → S.card = 252 :=
by sorry

end Sperner_theorem_example_l2147_214768


namespace equation_has_unique_integer_solution_l2147_214730

theorem equation_has_unique_integer_solution:
  ∀ m n : ℤ, (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n → m = 0 ∧ n = 0 := by
  intro m n
  -- The proof is omitted
  sorry

end equation_has_unique_integer_solution_l2147_214730


namespace second_player_can_form_palindrome_l2147_214762

def is_palindrome (s : List Char) : Prop :=
  s = s.reverse

theorem second_player_can_form_palindrome :
  ∀ (moves : List Char), moves.length = 1999 →
  ∃ (sequence : List Char), sequence.length = 1999 ∧ is_palindrome sequence :=
by
  sorry

end second_player_can_form_palindrome_l2147_214762


namespace buddy_thursday_cards_l2147_214703

-- Definitions from the given conditions
def monday_cards : ℕ := 30
def tuesday_cards : ℕ := monday_cards / 2
def wednesday_cards : ℕ := tuesday_cards + 12
def thursday_extra_cards : ℕ := tuesday_cards / 3
def thursday_cards : ℕ := wednesday_cards + thursday_extra_cards

-- Theorem to prove the total number of baseball cards on Thursday
theorem buddy_thursday_cards : thursday_cards = 32 :=
by
  -- Proof steps would go here, but we just provide the result for now
  sorry

end buddy_thursday_cards_l2147_214703


namespace ceiling_is_multiple_of_3_l2147_214775

-- Given conditions:
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1
axiom exists_three_real_roots : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧
  polynomial x1 = 0 ∧ polynomial x2 = 0 ∧ polynomial x3 = 0

-- Goal:
theorem ceiling_is_multiple_of_3 (x1 x2 x3 : ℝ) (h1 : x1 < x2) (h2 : x2 < x3)
  (hx1 : polynomial x1 = 0) (hx2 : polynomial x2 = 0) (hx3 : polynomial x3 = 0):
  ∀ n : ℕ, n > 0 → ∃ k : ℤ, k * 3 = ⌈x3^n⌉ := by
  sorry

end ceiling_is_multiple_of_3_l2147_214775


namespace mary_initial_nickels_l2147_214747

variable {x : ℕ}

theorem mary_initial_nickels (h : x + 5 = 12) : x = 7 := by
  sorry

end mary_initial_nickels_l2147_214747


namespace solve_for_y_l2147_214792

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end solve_for_y_l2147_214792


namespace sum_of_integers_is_27_24_or_20_l2147_214726

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end sum_of_integers_is_27_24_or_20_l2147_214726


namespace mark_profit_l2147_214717

def initialPrice : ℝ := 100
def finalPrice : ℝ := 3 * initialPrice
def salesTax : ℝ := 0.05 * initialPrice
def totalInitialCost : ℝ := initialPrice + salesTax
def transactionFee : ℝ := 0.03 * finalPrice
def profitBeforeTax : ℝ := finalPrice - totalInitialCost
def capitalGainsTax : ℝ := 0.15 * profitBeforeTax
def totalProfit : ℝ := profitBeforeTax - transactionFee - capitalGainsTax

theorem mark_profit : totalProfit = 147.75 := sorry

end mark_profit_l2147_214717


namespace sum_of_ammeter_readings_l2147_214776

def I1 := 4 
def I2 := 4
def I3 := 2 * I2
def I5 := I3 + I2
def I4 := (5 / 3) * I5

theorem sum_of_ammeter_readings : I1 + I2 + I3 + I4 + I5 = 48 := by
  sorry

end sum_of_ammeter_readings_l2147_214776


namespace largest_three_digit_multiple_of_4_and_5_l2147_214784

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∃ (n : ℕ), n < 1000 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n = 980 :=
by
  sorry

end largest_three_digit_multiple_of_4_and_5_l2147_214784


namespace minimum_employees_needed_l2147_214779

theorem minimum_employees_needed
  (n_W : ℕ) (n_A : ℕ) (n_S : ℕ)
  (n_WA : ℕ) (n_AS : ℕ) (n_SW : ℕ)
  (n_WAS : ℕ)
  (h_W : n_W = 115)
  (h_A : n_A = 92)
  (h_S : n_S = 60)
  (h_WA : n_WA = 32)
  (h_AS : n_AS = 20)
  (h_SW : n_SW = 10)
  (h_WAS : n_WAS = 5) :
  n_W + n_A + n_S - (n_WA - n_WAS) - (n_AS - n_WAS) - (n_SW - n_WAS) + 2 * n_WAS = 225 :=
by
  sorry

end minimum_employees_needed_l2147_214779


namespace chairs_left_after_selling_l2147_214799

-- Definitions based on conditions
def chairs_before_selling : ℕ := 15
def difference_after_selling : ℕ := 12

-- Theorem statement based on the question
theorem chairs_left_after_selling : (chairs_before_selling - 3 = difference_after_selling) → (chairs_before_selling - difference_after_selling = 3) := by
  intro h
  sorry

end chairs_left_after_selling_l2147_214799


namespace parametric_to_ordinary_eq_l2147_214733

variable (t : ℝ)

theorem parametric_to_ordinary_eq (h1 : x = Real.sqrt t + 1) (h2 : y = 2 * Real.sqrt t - 1) (h3 : t ≥ 0) :
    y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end parametric_to_ordinary_eq_l2147_214733


namespace find_number_l2147_214743

variable (x : ℝ)

theorem find_number (h : 20 * (x / 5) = 40) : x = 10 := by
  sorry

end find_number_l2147_214743


namespace walkway_area_correct_l2147_214749

-- Define the dimensions of one flower bed
def flower_bed_width := 8
def flower_bed_height := 3

-- Define the number of flower beds and the width of the walkways
def num_flowers_horizontal := 3
def num_flowers_vertical := 4
def walkway_width := 2

-- Calculate the total dimension of the garden including both flower beds and walkways
def total_garden_width := (num_flowers_horizontal * flower_bed_width) + ((num_flowers_horizontal + 1) * walkway_width)
def total_garden_height := (num_flowers_vertical * flower_bed_height) + ((num_flowers_vertical + 1) * walkway_width)

-- Calculate the total area of the garden and the total area of the flower beds
def total_garden_area := total_garden_width * total_garden_height
def total_flower_bed_area := (flower_bed_width * flower_bed_height) * (num_flowers_horizontal * num_flowers_vertical)

-- Calculate the total area of the walkways in the garden
def total_walkway_area := total_garden_area - total_flower_bed_area

-- The statement to be proven:
theorem walkway_area_correct : total_walkway_area = 416 := by
  sorry

end walkway_area_correct_l2147_214749


namespace angle_bisector_triangle_inequality_l2147_214780

theorem angle_bisector_triangle_inequality (AB AC D BD CD x : ℝ) (hAB : AB = 10) (hCD : CD = 3) (h_angle_bisector : BD = 30 / x)
  (h_triangle_inequality_1 : x + (BD + CD) > AB)
  (h_triangle_inequality_2 : AB + (BD + CD) > x)
  (h_triangle_inequality_3 : AB + x > BD + CD) :
  (3 < x) ∧ (x < 15) ∧ (3 + 15 = (18 : ℝ)) :=
by
  sorry

end angle_bisector_triangle_inequality_l2147_214780


namespace fraction_of_time_riding_at_15mph_l2147_214700

variable (t_5 t_15 : ℝ)

-- Conditions
def no_stops : Prop := (t_5 ≠ 0 ∧ t_15 ≠ 0)
def average_speed (t_5 t_15 : ℝ) : Prop := (5 * t_5 + 15 * t_15) / (t_5 + t_15) = 10

-- Question to be proved
theorem fraction_of_time_riding_at_15mph (h1 : no_stops t_5 t_15) (h2 : average_speed t_5 t_15) :
  t_15 / (t_5 + t_15) = 1 / 2 :=
sorry

end fraction_of_time_riding_at_15mph_l2147_214700


namespace probability_non_expired_bags_l2147_214765

theorem probability_non_expired_bags :
  let total_bags := 5
  let expired_bags := 2
  let selected_bags := 2
  let total_combinations := Nat.choose total_bags selected_bags
  let non_expired_bags := total_bags - expired_bags
  let favorable_outcomes := Nat.choose non_expired_bags selected_bags
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end probability_non_expired_bags_l2147_214765


namespace average_of_numbers_l2147_214774

theorem average_of_numbers (a b c d e : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) (h₄ : d = 11) (h₅ : e = 12) :
  (a + b + c + d + e) / 5 = 10 :=
by
  sorry

end average_of_numbers_l2147_214774


namespace percentage_above_wholesale_cost_l2147_214764

def wholesale_cost : ℝ := 200
def paid_price : ℝ := 228
def discount_rate : ℝ := 0.05

theorem percentage_above_wholesale_cost :
  ∃ P : ℝ, P = 20 ∧ 
    paid_price = (1 - discount_rate) * (wholesale_cost + P/100 * wholesale_cost) :=
by
  sorry

end percentage_above_wholesale_cost_l2147_214764


namespace domain_width_p_l2147_214748

variable (f : ℝ → ℝ)
variable (h_dom_f : ∀ x, -12 ≤ x ∧ x ≤ 12 → f x = f x)

noncomputable def p (x : ℝ) : ℝ := f (x / 3)

theorem domain_width_p : (width : ℝ) = 72 :=
by
  let domain_p : Set ℝ := {x | -36 ≤ x ∧ x ≤ 36}
  have : width = 72 := sorry
  exact this

end domain_width_p_l2147_214748


namespace man_l2147_214788

theorem man's_salary (S : ℝ)
  (h1 : S * (1/5 + 1/10 + 3/5) = 9/10 * S)
  (h2 : S - 9/10 * S = 14000) :
  S = 140000 :=
by
  sorry

end man_l2147_214788


namespace trapezoid_problem_l2147_214769

theorem trapezoid_problem (b h x : ℝ) 
  (h1 : x = (12500 / (x - 75)) - 75)
  (h_cond : (b + 75) / (b + 25) = 3 / 2)
  (b_solution : b = 75) :
  (⌊(x^2 / 100)⌋ : ℤ) = 181 :=
by
  -- The statement only requires us to assert the proof goal
  sorry

end trapezoid_problem_l2147_214769


namespace bisection_method_third_interval_l2147_214734

noncomputable def bisection_method_interval (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : (ℝ × ℝ) :=
  sorry  -- Definition of the interval using bisection method, but this is not necessary.

theorem bisection_method_third_interval (f : ℝ → ℝ) :
  (bisection_method_interval f (-2) 4 3) = (-1/2, 1) :=
sorry

end bisection_method_third_interval_l2147_214734


namespace percent_neither_condition_l2147_214787

namespace TeachersSurvey

variables (Total HighBloodPressure HeartTrouble Both: ℕ)

theorem percent_neither_condition :
  Total = 150 → HighBloodPressure = 90 → HeartTrouble = 50 → Both = 30 →
  (HighBloodPressure + HeartTrouble - Both) = 110 →
  ((Total - (HighBloodPressure + HeartTrouble - Both)) * 100 / Total) = 2667 / 100 :=
by
  intros hTotal hBP hHT hBoth hUnion
  sorry

end TeachersSurvey

end percent_neither_condition_l2147_214787


namespace expected_potato_yield_l2147_214729

-- Definitions based on the conditions
def steps_length : ℕ := 3
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def yield_rate : ℚ := 3 / 4

-- Calculate the dimensions in feet
def garden_length_feet : ℕ := garden_length_steps * steps_length
def garden_width_feet : ℕ := garden_width_steps * steps_length

-- Calculate the area in square feet
def garden_area_feet : ℕ := garden_length_feet * garden_width_feet

-- Calculate the expected yield in pounds
def expected_yield_pounds : ℚ := garden_area_feet * yield_rate

-- The theorem to prove the expected yield
theorem expected_potato_yield :
  expected_yield_pounds = 3037.5 := by
  sorry  -- Proof is omitted as per the instructions.

end expected_potato_yield_l2147_214729


namespace cost_of_book_first_sold_at_loss_l2147_214754

theorem cost_of_book_first_sold_at_loss (C1 C2 C3 : ℝ) (h1 : C1 + C2 + C3 = 810)
    (h2 : 0.88 * C1 = 1.18 * C2) (h3 : 0.88 * C1 = 1.27 * C3) : 
    C1 = 333.9 := 
by
  -- Conditions given
  have h4 : C2 = 0.88 * C1 / 1.18 := by sorry
  have h5 : C3 = 0.88 * C1 / 1.27 := by sorry

  -- Substituting back into the total cost equation
  have h6 : C1 + 0.88 * C1 / 1.18 + 0.88 * C1 / 1.27 = 810 := by sorry

  -- Simplifying and solving for C1
  have h7 : C1 = 333.9 := by sorry

  -- Conclusion
  exact h7

end cost_of_book_first_sold_at_loss_l2147_214754


namespace ball_returns_velocity_required_initial_velocity_to_stop_l2147_214727

-- Define the conditions.
def distance_A_to_wall : ℝ := 5
def distance_wall_to_B : ℝ := 2
def distance_AB : ℝ := 9
def initial_velocity_v0 : ℝ := 5
def acceleration_a : ℝ := -0.4

-- Hypothesize that the velocity when the ball returns to A is 3 m/s.
theorem ball_returns_velocity (t : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  initial_velocity_v0 * t + (1 / 2) * acceleration_a * t^2 = distance_AB + distance_A_to_wall →
  initial_velocity_v0 + acceleration_a * t = 3 := sorry

-- Hypothesize that to stop exactly at A, the initial speed should be 4 m/s.
theorem required_initial_velocity_to_stop (t' : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  (0.4 * t') * t' + (1 / 2) * acceleration_a * t'^2 = distance_AB + distance_A_to_wall →
  0.4 * t' = 4 := sorry

end ball_returns_velocity_required_initial_velocity_to_stop_l2147_214727


namespace sum_of_fractions_eq_13_5_l2147_214746

noncomputable def sumOfFractions : ℚ :=
  (1/10 + 2/10 + 3/10 + 4/10 + 5/10 + 6/10 + 7/10 + 8/10 + 9/10 + 90/10)

theorem sum_of_fractions_eq_13_5 :
  sumOfFractions = 13.5 := by
  sorry

end sum_of_fractions_eq_13_5_l2147_214746


namespace inequality_abc_l2147_214755

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 :=
by
  sorry

end inequality_abc_l2147_214755


namespace gross_pay_is_450_l2147_214710

def net_pay : ℤ := 315
def taxes : ℤ := 135
def gross_pay : ℤ := net_pay + taxes

theorem gross_pay_is_450 : gross_pay = 450 := by
  sorry

end gross_pay_is_450_l2147_214710


namespace combinations_15_3_l2147_214777

def num_combinations (n k : ℕ) : ℕ := n.choose k

theorem combinations_15_3 :
  num_combinations 15 3 = 455 :=
sorry

end combinations_15_3_l2147_214777


namespace ak_divisibility_l2147_214735

theorem ak_divisibility {a k m n : ℕ} (h : a ^ k % (m ^ n) = 0) : a ^ (k * m) % (m ^ (n + 1)) = 0 :=
sorry

end ak_divisibility_l2147_214735


namespace problem_statement_l2147_214745

theorem problem_statement (x y z : ℤ) (h1 : x = z - 2) (h2 : y = x + 1) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := 
by
  sorry

end problem_statement_l2147_214745


namespace solve_equation_l2147_214718

theorem solve_equation (x : ℝ) : x * (x-3)^2 * (5+x) = 0 ↔ (x = 0 ∨ x = 3 ∨ x = -5) := 
by
  sorry

end solve_equation_l2147_214718


namespace solve_x_l2147_214701

theorem solve_x 
  (x : ℝ) 
  (h : (2 / x) + (3 / x) / (6 / x) = 1.25) : 
  x = 8 / 3 := 
sorry

end solve_x_l2147_214701


namespace true_proposition_l2147_214789

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l2147_214789


namespace isosceles_triangle_perimeter_l2147_214744

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 11) (h3 : a = b ∨ b = b) :
  (5 + 11 + 11 = 27) := 
by {
  sorry
}

end isosceles_triangle_perimeter_l2147_214744


namespace clock_strike_time_l2147_214709

theorem clock_strike_time (t : ℕ) (n m : ℕ) (I : ℕ) : 
  t = 12 ∧ n = 3 ∧ m = 6 ∧ 2 * I = t → (m - 1) * I = 30 := by 
  sorry

end clock_strike_time_l2147_214709


namespace math_problem_l2147_214711

theorem math_problem :
  ((-1)^2023 - (27^(1/3)) - (16^(1/2)) + (|1 - Real.sqrt 3|)) = -9 + Real.sqrt 3 :=
by
  sorry

end math_problem_l2147_214711


namespace Dawn_hourly_earnings_l2147_214795

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end Dawn_hourly_earnings_l2147_214795


namespace car_value_proof_l2147_214772

-- Let's define the variables and the conditions.
def car_sold_value : ℝ := 20000
def sticker_price_new_car : ℝ := 30000
def percent_sold : ℝ := 0.80
def percent_paid : ℝ := 0.90
def out_of_pocket : ℝ := 11000

theorem car_value_proof :
  (percent_paid * sticker_price_new_car - percent_sold * car_sold_value = out_of_pocket) →
  car_sold_value = 20000 := 
by
  intros h
  -- Introduction of any intermediate steps if necessary should just invoke the sorry to indicate the need for proof later
  exact sorry

end car_value_proof_l2147_214772


namespace number_of_grade12_students_selected_l2147_214758

def total_students : ℕ := 1500
def grade10_students : ℕ := 550
def grade11_students : ℕ := 450
def total_sample_size : ℕ := 300
def grade12_students : ℕ := total_students - grade10_students - grade11_students

theorem number_of_grade12_students_selected :
    (total_sample_size * grade12_students / total_students) = 100 := by
  sorry

end number_of_grade12_students_selected_l2147_214758


namespace sequence_general_formula_l2147_214782

theorem sequence_general_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (a 5 = 16) → ∀ n : ℕ, n > 0 → a n = 2^(n-1) :=
by
  intros h n hn
  sorry

end sequence_general_formula_l2147_214782


namespace students_in_both_clubs_l2147_214702

variables (Total Students RoboticClub ScienceClub EitherClub BothClubs : ℕ)

theorem students_in_both_clubs
  (h1 : Total = 300)
  (h2 : RoboticClub = 80)
  (h3 : ScienceClub = 130)
  (h4 : EitherClub = 190)
  (h5 : EitherClub = RoboticClub + ScienceClub - BothClubs) :
  BothClubs = 20 :=
by
  sorry

end students_in_both_clubs_l2147_214702


namespace triangle_cosine_identity_l2147_214791

open Real

variables {A B C a b c : ℝ}

theorem triangle_cosine_identity (h : b = (a + c) / 2) : cos (A - C) + 4 * cos B = 3 :=
sorry

end triangle_cosine_identity_l2147_214791


namespace gridPolygon_side_longer_than_one_l2147_214742

-- Define the structure of a grid polygon
structure GridPolygon where
  area : ℕ  -- Area of the grid polygon
  perimeter : ℕ  -- Perimeter of the grid polygon
  no_holes : Prop  -- Polyon does not contain holes

-- Definition of a grid polygon with specific properties
def specificGridPolygon : GridPolygon :=
  { area := 300, perimeter := 300, no_holes := true }

-- The theorem we want to prove that ensures at least one side is longer than 1
theorem gridPolygon_side_longer_than_one (P : GridPolygon) (h_area : P.area = 300) (h_perimeter : P.perimeter = 300) (h_no_holes : P.no_holes) : ∃ side_length : ℝ, side_length > 1 :=
  by
  sorry

end gridPolygon_side_longer_than_one_l2147_214742


namespace coprime_count_l2147_214786

theorem coprime_count (n : ℕ) (h : n = 56700000) : 
  ∃ m, m = 12960000 ∧ ∀ i < n, Nat.gcd i n = 1 → i < m :=
by
  sorry

end coprime_count_l2147_214786


namespace speed_of_second_car_l2147_214770

theorem speed_of_second_car
  (t : ℝ)
  (distance_apart : ℝ)
  (speed_first_car : ℝ)
  (speed_second_car : ℝ)
  (h_total_distance : distance_apart = t * speed_first_car + t * speed_second_car)
  (h_time : t = 2.5)
  (h_distance_apart : distance_apart = 310)
  (h_speed_first_car : speed_first_car = 60) :
  speed_second_car = 64 := by
  sorry

end speed_of_second_car_l2147_214770


namespace cos_shifted_alpha_l2147_214771

theorem cos_shifted_alpha (α : ℝ) (h1 : Real.tan α = -3/4) (h2 : α ∈ Set.Ioc (3*Real.pi/2) (2*Real.pi)) :
  Real.cos (Real.pi/2 + α) = 3/5 :=
sorry

end cos_shifted_alpha_l2147_214771


namespace max_viewing_area_l2147_214732

theorem max_viewing_area (L W: ℝ) (h1: 2 * L + 2 * W = 420) (h2: L ≥ 100) (h3: W ≥ 60) : 
  (L = 105) ∧ (W = 105) ∧ (L * W = 11025) :=
by
  sorry

end max_viewing_area_l2147_214732


namespace people_joined_l2147_214741

theorem people_joined (total_left : ℕ) (total_remaining : ℕ) (Molly_and_parents : ℕ)
  (h1 : total_left = 40) (h2 : total_remaining = 63) (h3 : Molly_and_parents = 3) :
  ∃ n, n = 100 := 
by
  sorry

end people_joined_l2147_214741


namespace min_value_of_reciprocals_l2147_214720

theorem min_value_of_reciprocals (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 2) : 
  ∃ (x : ℝ), x = 2 * a + b ∧ ∃ (y : ℝ), y = 2 * b + c ∧ ∃ (z : ℝ), z = 2 * c + a ∧ (1 / x + 1 / y + 1 / z = 27 / 8) :=
sorry

end min_value_of_reciprocals_l2147_214720


namespace select_two_integers_divisibility_l2147_214773

open Polynomial

theorem select_two_integers_divisibility
  (F : Polynomial ℤ)
  (m : ℕ)
  (a : Fin m → ℤ)
  (H : ∀ n : ℤ, ∃ i : Fin m, a i ∣ F.eval n) :
  ∃ i j : Fin m, i ≠ j ∧ ∀ n : ℤ, ∃ k : Fin m, k = i ∨ k = j ∧ a k ∣ F.eval n :=
by
  sorry

end select_two_integers_divisibility_l2147_214773


namespace pyramid_volume_l2147_214761

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end pyramid_volume_l2147_214761


namespace Sara_snow_volume_l2147_214759

theorem Sara_snow_volume :
  let length := 30
  let width := 3
  let first_half_length := length / 2
  let second_half_length := length / 2
  let depth1 := 0.5
  let depth2 := 1.0 / 3.0
  let volume1 := first_half_length * width * depth1
  let volume2 := second_half_length * width * depth2
  volume1 + volume2 = 37.5 :=
by
  sorry

end Sara_snow_volume_l2147_214759


namespace complement_A_in_U_l2147_214778

-- Define the universal set as ℝ
def U : Set ℝ := Set.univ

-- Define the set A as given in the conditions
def A : Set ℝ := {y | ∃ x : ℝ, 2^(Real.log x) = y}

-- The main statement based on the conditions and the correct answer
theorem complement_A_in_U : (U \ A) = {y | y ≤ 0} := by
  sorry

end complement_A_in_U_l2147_214778


namespace fraction_spent_toy_store_l2147_214781

noncomputable def weekly_allowance : ℚ := 2.25
noncomputable def arcade_fraction_spent : ℚ := 3 / 5
noncomputable def candy_store_spent : ℚ := 0.60

theorem fraction_spent_toy_store :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction_spent)
  let spent_toy_store := remaining_after_arcade - candy_store_spent
  spent_toy_store / remaining_after_arcade = 1 / 3 :=
by
  sorry

end fraction_spent_toy_store_l2147_214781


namespace one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l2147_214723

-- Definitions from conditions
def cubic_eq (x p q : ℝ) := x^3 + p * x + q

-- Correct answers in mathematical proofs
theorem one_real_root (p q : ℝ) : 4 * p^3 + 27 * q^2 > 0 → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem multiple_coinciding_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem three_distinct_real_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 < 0 → ∃ x₁ x₂ x₃ : ℝ, 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ cubic_eq x₁ p q = 0 ∧ cubic_eq x₂ p q = 0 ∧ cubic_eq x₃ p q = 0 := sorry

theorem three_coinciding_roots_at_origin : ∃ x : ℝ, cubic_eq x 0 0 = 0 := sorry

end one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l2147_214723


namespace additional_chicken_wings_l2147_214737

theorem additional_chicken_wings (friends : ℕ) (wings_per_friend : ℕ) (initial_wings : ℕ) (H1 : friends = 9) (H2 : wings_per_friend = 3) (H3 : initial_wings = 2) : 
  friends * wings_per_friend - initial_wings = 25 := by
  sorry

end additional_chicken_wings_l2147_214737


namespace min_value_of_2a7_a11_l2147_214721

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the sequence terms

-- Conditions
axiom geometric_sequence (n m : ℕ) (r : ℝ) (h : ∀ k, a k > 0) : a n = a 0 * r^n
axiom geometric_mean_condition : a 4 * a 14 = 8

-- Theorem to Prove
theorem min_value_of_2a7_a11 : ∀ n : ℕ, (∀ k, a k > 0) → 2 * a 7 + a 11 ≥ 8 :=
by
  intros
  sorry

end min_value_of_2a7_a11_l2147_214721


namespace circle_line_distance_condition_l2147_214738

theorem circle_line_distance_condition :
  ∀ (c : ℝ), 
    (∃ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 8 = 0 ∧ (x - y + c = 2 ∨ x - y + c = -2)) →
    -2*Real.sqrt 2 ≤ c ∧ c ≤ 2*Real.sqrt 2 := 
sorry

end circle_line_distance_condition_l2147_214738


namespace total_cost_proof_l2147_214752

noncomputable def cost_proof : Prop :=
  let M := 158.4
  let R := 66
  let F := 22
  (10 * M = 24 * R) ∧ (6 * F = 2 * R) ∧ (F = 22) →
  (4 * M + 3 * R + 5 * F = 941.6)

theorem total_cost_proof : cost_proof :=
by
  sorry

end total_cost_proof_l2147_214752


namespace egor_last_payment_l2147_214719

theorem egor_last_payment (a b c d : ℕ) (h_sum : a + b + c + d = 28)
  (h1 : b ≥ 2 * a) (h2 : c ≥ 2 * b) (h3 : d ≥ 2 * c) : d = 18 := by
  sorry

end egor_last_payment_l2147_214719


namespace product_units_digit_mod_10_l2147_214794

theorem product_units_digit_mod_10
  (u1 u2 u3 : ℕ)
  (hu1 : u1 = 2583 % 10)
  (hu2 : u2 = 7462 % 10)
  (hu3 : u3 = 93215 % 10) :
  ((2583 * 7462 * 93215) % 10) = 0 :=
by
  have h_units1 : u1 = 3 := by sorry
  have h_units2 : u2 = 2 := by sorry
  have h_units3 : u3 = 5 := by sorry
  have h_produce_units : ((3 * 2 * 5) % 10) = 0 := by sorry
  exact h_produce_units

end product_units_digit_mod_10_l2147_214794


namespace linear_function_solution_l2147_214704

open Function

theorem linear_function_solution (f : ℝ → ℝ)
  (h_lin : ∃ k b, k ≠ 0 ∧ ∀ x, f x = k * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x - 1) :
  (∀ x, f x = 2 * x - 1 / 3) ∨ (∀ x, f x = -2 * x + 1) :=
by
  sorry

end linear_function_solution_l2147_214704


namespace sqrt_eq_sum_seven_l2147_214713

open Real

theorem sqrt_eq_sum_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
    sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
by
  sorry

end sqrt_eq_sum_seven_l2147_214713


namespace hyperbola_center_l2147_214766

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x ^ 2 + 54 * x - 16 * y ^ 2 - 128 * y - 200 = 0) : 
  (x = -3) ∧ (y = -4) := 
sorry

end hyperbola_center_l2147_214766


namespace solve_for_b_l2147_214731

theorem solve_for_b (x y b : ℝ) (h1: 4 * x + y = b) (h2: 3 * x + 4 * y = 3 * b) (hx: x = 3) : b = 39 :=
sorry

end solve_for_b_l2147_214731


namespace min_num_stamps_is_17_l2147_214740

-- Definitions based on problem conditions
def initial_num_stamps : ℕ := 2 + 5 + 3 + 1
def initial_cost : ℝ := 2 * 0.10 + 5 * 0.20 + 3 * 0.50 + 1 * 2
def remaining_cost : ℝ := 10 - initial_cost
def additional_stamps : ℕ := 2 + 2 + 1 + 1
def total_stamps : ℕ := initial_num_stamps + additional_stamps

-- Proof that the minimum number of stamps bought is 17
theorem min_num_stamps_is_17 : total_stamps = 17 := by
  sorry

end min_num_stamps_is_17_l2147_214740


namespace power_of_binomials_l2147_214750

theorem power_of_binomials :
  (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 :=
by
  -- This is a placeholder for the actual proof steps.
  -- We use 'sorry' to indicate that the proof is omitted here.
  sorry

end power_of_binomials_l2147_214750


namespace original_number_of_movies_l2147_214798

theorem original_number_of_movies (x : ℕ) (dvd blu_ray : ℕ)
  (h1 : dvd = 17 * x)
  (h2 : blu_ray = 4 * x)
  (h3 : 17 * x / (4 * x - 4) = 9 / 2) :
  dvd + blu_ray = 378 := by
  sorry

end original_number_of_movies_l2147_214798


namespace intersection_of_sets_l2147_214722

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

def B : Set ℝ := Ico 0 4  -- Ico stands for interval [0, 4)

theorem intersection_of_sets : A ∩ B = Ico 2 4 :=
by 
  sorry

end intersection_of_sets_l2147_214722


namespace range_of_f_l2147_214760

def f (x : ℕ) : ℤ := 2 * x - 3

def domain := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

def range (f : ℕ → ℤ) (s : Set ℕ) : Set ℤ :=
  {y : ℤ | ∃ x ∈ s, f x = y}

theorem range_of_f :
  range f domain = {-1, 1, 3, 5, 7} :=
by
  sorry

end range_of_f_l2147_214760


namespace geometric_sequence_a3_l2147_214715

variable {a : ℕ → ℝ} (h1 : a 1 > 0) (h2 : a 2 * a 4 = 25)
def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (h_geom : geometric_sequence a) : 
  a 3 = 5 := 
by
  sorry

end geometric_sequence_a3_l2147_214715


namespace minimize_triangle_area_minimize_product_PA_PB_l2147_214707

-- Define the initial conditions and geometry setup
def point (x y : ℝ) := (x, y)
def line_eq (a b : ℝ) := ∀ x y : ℝ, x / a + y / b = 1

-- Point P
def P := point 2 1

-- Condition: the line passes through point P and intersects the axes
def line_through_P (a b : ℝ) := line_eq a b ∧ (2 / a + 1 / b = 1) ∧ a > 2 ∧ b > 1

-- Prove that the line minimizing the area of triangle AOB is x + 2y - 4 = 0
theorem minimize_triangle_area (a b : ℝ) (h : line_through_P a b) :
  a = 4 ∧ b = 2 → line_eq 4 2 := 
sorry

-- Prove that the line minimizing the product |PA||PB| is x + y - 3 = 0
theorem minimize_product_PA_PB (a b : ℝ) (h : line_through_P a b) :
  a = 3 ∧ b = 3 → line_eq 3 3 := 
sorry

end minimize_triangle_area_minimize_product_PA_PB_l2147_214707


namespace foci_on_x_axis_l2147_214793

theorem foci_on_x_axis (k : ℝ) : (∃ a b : ℝ, ∀ x y : ℝ, (x^2)/(3 - k) + (y^2)/(1 + k) = 1) ↔ -1 < k ∧ k < 1 :=
by
  sorry

end foci_on_x_axis_l2147_214793


namespace find_a_for_extraneous_roots_find_a_for_no_solution_l2147_214756

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end find_a_for_extraneous_roots_find_a_for_no_solution_l2147_214756


namespace unattainable_value_of_y_l2147_214712

theorem unattainable_value_of_y (x : ℚ) (h : x ≠ -5/4) :
  ¬ ∃ y : ℚ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3/4 :=
by
  sorry

end unattainable_value_of_y_l2147_214712


namespace num_ways_distribute_balls_l2147_214757

-- Definitions of the conditions
def balls := 6
def boxes := 4

-- Theorem statement
theorem num_ways_distribute_balls : 
  ∃ n : ℕ, (balls = 6 ∧ boxes = 4) → n = 8 :=
sorry

end num_ways_distribute_balls_l2147_214757


namespace intersection_count_l2147_214708

theorem intersection_count :
  ∀ {x y : ℝ}, (2 * x - 2 * y + 4 = 0 ∨ 6 * x + 2 * y - 8 = 0) ∧ (y = -x^2 + 2 ∨ 4 * x - 10 * y + 14 = 0) → 
  (x ≠ 0 ∨ y ≠ 2) ∧ (x ≠ -1 ∨ y ≠ 1) ∧ (x ≠ 1 ∨ y ≠ -1) ∧ (x ≠ 2 ∨ y ≠ 2) → 
  ∃! (p : ℝ × ℝ), (p = (0, 2) ∨ p = (-1, 1) ∨ p = (1, -1) ∨ p = (2, 2)) := sorry

end intersection_count_l2147_214708


namespace cycle_selling_price_l2147_214725

theorem cycle_selling_price
  (cost_price : ℝ)
  (gain_percentage : ℝ)
  (profit : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 930)
  (h2 : gain_percentage = 30.107526881720432)
  (h3 : profit = (gain_percentage / 100) * cost_price)
  (h4 : selling_price = cost_price + profit)
  : selling_price = 1210 := 
sorry

end cycle_selling_price_l2147_214725


namespace simplify_abs_expression_l2147_214796

theorem simplify_abs_expression (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2 * b + 5| + |-3 * a + 2 * b - 2| = 4 * a - 4 * b + 7 := by
  sorry

end simplify_abs_expression_l2147_214796


namespace number_of_subsets_of_set_l2147_214724

theorem number_of_subsets_of_set {n : ℕ} (h : n = 2016) :
  (2^2016) = 2^2016 :=
by
  sorry

end number_of_subsets_of_set_l2147_214724


namespace problem_2_8_3_4_7_2_2_l2147_214785

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end problem_2_8_3_4_7_2_2_l2147_214785


namespace sqrt_mixed_number_l2147_214705

theorem sqrt_mixed_number :
  (Real.sqrt (8 + 9/16)) = (Real.sqrt 137) / 4 :=
by
  sorry

end sqrt_mixed_number_l2147_214705
