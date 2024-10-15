import Mathlib

namespace NUMINAMATH_GPT_books_remaining_correct_l1612_161262

-- Define the initial number of book donations
def initial_books : ℕ := 300

-- Define the number of people donating and the number of books each donates
def num_people : ℕ := 10
def books_per_person : ℕ := 5

-- Calculate total books donated by all people
def total_donation : ℕ := num_people * books_per_person

-- Define the number of books borrowed by other people
def borrowed_books : ℕ := 140

-- Calculate the total number of books after donations and then subtract the borrowed books
def total_books_remaining : ℕ := initial_books + total_donation - borrowed_books

-- Prove the total number of books remaining is 210
theorem books_remaining_correct : total_books_remaining = 210 := by
  sorry

end NUMINAMATH_GPT_books_remaining_correct_l1612_161262


namespace NUMINAMATH_GPT_Brians_trip_distance_l1612_161265

theorem Brians_trip_distance (miles_per_gallon : ℕ) (gallons_used : ℕ) (distance_traveled : ℕ) 
  (h1 : miles_per_gallon = 20) (h2 : gallons_used = 3) : 
  distance_traveled = 60 :=
by
  sorry

end NUMINAMATH_GPT_Brians_trip_distance_l1612_161265


namespace NUMINAMATH_GPT_percentage_difference_l1612_161245

theorem percentage_difference (G P R : ℝ) (h1 : P = 0.9 * G) (h2 : R = 1.125 * G) :
  ((1 - P / R) * 100) = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1612_161245


namespace NUMINAMATH_GPT_original_price_of_sarees_l1612_161216

theorem original_price_of_sarees (P : ℝ) (h : 0.75 * 0.85 * P = 306) : P = 480 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_sarees_l1612_161216


namespace NUMINAMATH_GPT_find_raspberries_l1612_161244

def total_berries (R : ℕ) : ℕ := 30 + 20 + R

def fresh_berries (R : ℕ) : ℕ := 2 * total_berries R / 3

def fresh_berries_to_keep (R : ℕ) : ℕ := fresh_berries R / 2

def fresh_berries_to_sell (R : ℕ) : ℕ := fresh_berries R - fresh_berries_to_keep R

theorem find_raspberries (R : ℕ) : fresh_berries_to_sell R = 20 → R = 10 := 
by 
sorry

-- To ensure the problem is complete and solvable, we also need assumptions on the domain:
example : ∃ R : ℕ, fresh_berries_to_sell R = 20 := 
by 
  use 10 
  sorry

end NUMINAMATH_GPT_find_raspberries_l1612_161244


namespace NUMINAMATH_GPT_total_dolls_l1612_161258

def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

theorem total_dolls : rene_dolls + sister_dolls + grandmother_dolls = 258 :=
by {
  -- Required proof steps would be placed here, 
  -- but are omitted as per the instructions.
  sorry
}

end NUMINAMATH_GPT_total_dolls_l1612_161258


namespace NUMINAMATH_GPT_intersection_complement_eq_find_a_l1612_161246

-- Proof Goal 1: A ∩ ¬B = {x : ℝ | x ∈ (-∞, -3] ∪ [14, ∞)}

def setA : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def setB : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def negB : Set ℝ := {x | x ≤ -2 ∨ x ≥ 14}

theorem intersection_complement_eq :
  setA ∩ negB = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Proof Goal 2: The range of a such that E ⊆ B

def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

theorem find_a (a : ℝ) :
  (∀ x, E a x → setB x) → a ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_find_a_l1612_161246


namespace NUMINAMATH_GPT_trajectory_equation_l1612_161214

noncomputable def circle1_center := (-3, 0)
noncomputable def circle2_center := (3, 0)

def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

def is_tangent_internally (x y : ℝ) : Prop := 
  ∃ (P : ℝ × ℝ), circle1 P.1 P.2 ∧ circle2 P.1 P.2

theorem trajectory_equation :
  ∀ (x y : ℝ), is_tangent_internally x y → (x^2 / 16 + y^2 / 7 = 1) :=
sorry

end NUMINAMATH_GPT_trajectory_equation_l1612_161214


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_conversion_l1612_161299

theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (h1 : r = 10) 
  (h2 : θ = Real.pi / 3) 
  (h3 : z = -2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5, 5 * Real.sqrt 3, -2) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_conversion_l1612_161299


namespace NUMINAMATH_GPT_rolling_circle_trace_eq_envelope_l1612_161228

-- Definitions for the geometrical setup
variable {a : ℝ} (C : ℝ → ℝ → Prop)

-- The main statement to prove
theorem rolling_circle_trace_eq_envelope (hC : ∀ t : ℝ, C (a * t) a) :
  ∃ P : ℝ × ℝ → Prop, ∀ t : ℝ, C (a/2 * t + a/2 * Real.sin t) (a/2 + a/2 * Real.cos t) :=
by
  sorry

end NUMINAMATH_GPT_rolling_circle_trace_eq_envelope_l1612_161228


namespace NUMINAMATH_GPT_servant_leaving_months_l1612_161259

-- The given conditions
def total_salary_year : ℕ := 90 + 110
def monthly_salary (months: ℕ) : ℕ := (months * total_salary_year) / 12
def total_received : ℕ := 40 + 110

-- The theorem to prove
theorem servant_leaving_months (months : ℕ) (h : monthly_salary months = total_received) : months = 9 :=
by {
    sorry
}

end NUMINAMATH_GPT_servant_leaving_months_l1612_161259


namespace NUMINAMATH_GPT_find_natural_numbers_l1612_161270

theorem find_natural_numbers (x y z : ℕ) (hx : x ≤ y) (hy : y ≤ z) : 
    (1 + 1 / x) * (1 + 1 / y) * (1 + 1 / z) = 3 
    → (x = 1 ∧ y = 3 ∧ z = 8) 
    ∨ (x = 1 ∧ y = 4 ∧ z = 5) 
    ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
sorry

end NUMINAMATH_GPT_find_natural_numbers_l1612_161270


namespace NUMINAMATH_GPT_shortest_distance_parabola_to_line_l1612_161275

open Real

theorem shortest_distance_parabola_to_line :
  ∃ (d : ℝ), 
    (∀ (P : ℝ × ℝ), (P.1 = (P.2^2) / 8) → 
      ((2 * P.1 - P.2 - 4) / sqrt 5 ≥ d)) ∧ 
    (d = 3 * sqrt 5 / 5) :=
sorry

end NUMINAMATH_GPT_shortest_distance_parabola_to_line_l1612_161275


namespace NUMINAMATH_GPT_find_value_of_m_l1612_161218

/-- Given the parabola y = 4x^2 + 4x + 5 and the line y = 8mx + 8m intersect at exactly one point,
    prove the value of m^{36} + 1155 / m^{12} is 39236. -/
theorem find_value_of_m (m : ℝ) (h: ∃ x, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m ∧
  ∀ x₁ x₂, 4 * x₁^2 + 4 * x₁ + 5 = 8 * m * x₁ + 8 * m →
  4 * x₂^2 + 4 * x₂ + 5 = 8 * m * x₂ + 8 * m → x₁ = x₂) :
  m^36 + 1155 / m^12 = 39236 := 
sorry

end NUMINAMATH_GPT_find_value_of_m_l1612_161218


namespace NUMINAMATH_GPT_snowballs_made_by_brother_l1612_161290

/-- Janet makes 50 snowballs and her brother makes the remaining snowballs. Janet made 25% of the total snowballs. 
    Prove that her brother made 150 snowballs. -/
theorem snowballs_made_by_brother (total_snowballs : ℕ) (janet_snowballs : ℕ) (fraction_janet : ℚ)
  (h1 : janet_snowballs = 50) (h2 : fraction_janet = 25 / 100) (h3 : janet_snowballs = fraction_janet * total_snowballs) :
  total_snowballs - janet_snowballs = 150 :=
by
  sorry

end NUMINAMATH_GPT_snowballs_made_by_brother_l1612_161290


namespace NUMINAMATH_GPT_walking_speed_l1612_161223

theorem walking_speed 
  (D : ℝ) 
  (V_w : ℝ) 
  (h1 : D = V_w * 8) 
  (h2 : D = 36 * 2) : 
  V_w = 9 :=
by
  sorry

end NUMINAMATH_GPT_walking_speed_l1612_161223


namespace NUMINAMATH_GPT_percentage_increase_l1612_161250

theorem percentage_increase (x : ℝ) (h : x = 77.7) : 
  ((x - 70) / 70) * 100 = 11 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1612_161250


namespace NUMINAMATH_GPT_total_gumballs_l1612_161289

-- Define the count of red, blue, and green gumballs
def red_gumballs := 16
def blue_gumballs := red_gumballs / 2
def green_gumballs := blue_gumballs * 4

-- Prove that the total number of gumballs is 56
theorem total_gumballs : red_gumballs + blue_gumballs + green_gumballs = 56 := by
  sorry

end NUMINAMATH_GPT_total_gumballs_l1612_161289


namespace NUMINAMATH_GPT_find_angle_B_find_a_plus_c_l1612_161272

variable (A B C a b c S : Real)

-- Conditions
axiom h1 : a = (1 / 2) * c + b * Real.cos C
axiom h2 : S = Real.sqrt 3
axiom h3 : b = Real.sqrt 13

-- Questions (Proving the answers from the problem)
theorem find_angle_B (hA : A = Real.pi - (B + C)) : 
  B = Real.pi / 3 := by
  sorry

theorem find_a_plus_c (hac : (1 / 2) * a * c * Real.sin (Real.pi / 3) = Real.sqrt 3) : 
  a + c = 5 := by
  sorry

end NUMINAMATH_GPT_find_angle_B_find_a_plus_c_l1612_161272


namespace NUMINAMATH_GPT_beta_greater_than_alpha_l1612_161203

theorem beta_greater_than_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : Real.sin (α + β) = 2 * Real.sin α) : β > α := 
sorry

end NUMINAMATH_GPT_beta_greater_than_alpha_l1612_161203


namespace NUMINAMATH_GPT_range_of_a_l1612_161286

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 16 * Real.log x

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, a - 1 ≤ x ∧ x ≤ a + 2 → (fderiv ℝ f x) x < 0)
  ↔ (1 < a) ∧ (a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1612_161286


namespace NUMINAMATH_GPT_jail_time_calculation_l1612_161292

-- Define conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def arrests_per_day : ℕ := 10
def pre_trial_days : ℕ := 4
def half_two_week_sentence_days : ℕ := 7 -- 1 week is half of 2 weeks

-- Define the calculation of the total combined weeks of jail time
def total_combined_weeks_jail_time : ℕ :=
  let total_arrests := arrests_per_day * number_of_cities * days_of_protest
  let total_days_jail_per_person := pre_trial_days + half_two_week_sentence_days
  let total_combined_days_jail_time := total_arrests * total_days_jail_per_person
  total_combined_days_jail_time / 7

-- Theorem statement
theorem jail_time_calculation : total_combined_weeks_jail_time = 9900 := by
  sorry

end NUMINAMATH_GPT_jail_time_calculation_l1612_161292


namespace NUMINAMATH_GPT_problem1_problem2_l1612_161252

noncomputable def f (x : ℝ) : ℝ :=
  |x - 2| - |2 * x + 1|

theorem problem1 (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ -1/3 ≤ x :=
sorry

theorem problem2 (a : ℝ) (b : ℝ) :
  (∀ x, |a + b| - |a - b| ≥ f x) → (a ≥ 5 / 4 ∨ a ≤ -5 / 4) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1612_161252


namespace NUMINAMATH_GPT_same_number_of_friends_l1612_161236

-- Definitions and conditions
def num_people (n : ℕ) := true   -- Placeholder definition to indicate the number of people
def num_friends (person : ℕ) (n : ℕ) : ℕ := sorry -- The number of friends a given person has (needs to be defined)
def friends_range (n : ℕ) := ∀ person, 0 ≤ num_friends person n ∧ num_friends person n < n

-- Theorem statement
theorem same_number_of_friends (n : ℕ) (h1 : num_people n) (h2 : friends_range n) : 
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ num_friends p1 n = num_friends p2 n :=
by
  sorry

end NUMINAMATH_GPT_same_number_of_friends_l1612_161236


namespace NUMINAMATH_GPT_solution_set_quadratic_inequality_l1612_161207

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} :=
sorry

end NUMINAMATH_GPT_solution_set_quadratic_inequality_l1612_161207


namespace NUMINAMATH_GPT_find_ratio_l1612_161242

-- Given conditions
variable (x y a b : ℝ)
variable (h1 : 2 * x - y = a)
variable (h2 : 4 * y - 8 * x = b)
variable (h3 : b ≠ 0)

theorem find_ratio (a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 4 := by
  sorry

end NUMINAMATH_GPT_find_ratio_l1612_161242


namespace NUMINAMATH_GPT_sum_first_9000_terms_l1612_161248

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
a * ((1 - r^n) / (1 - r))

theorem sum_first_9000_terms (a r : ℝ) (h1 : geom_sum a r 3000 = 1000) 
                              (h2 : geom_sum a r 6000 = 1900) : 
                              geom_sum a r 9000 = 2710 := 
by sorry

end NUMINAMATH_GPT_sum_first_9000_terms_l1612_161248


namespace NUMINAMATH_GPT_exponents_multiplication_exponents_power_exponents_distributive_l1612_161233

variables (x y m : ℝ)

theorem exponents_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 :=
by sorry

theorem exponents_power (m : ℝ) : (m^2)^4 = m^8 :=
by sorry

theorem exponents_distributive (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 :=
by sorry

end NUMINAMATH_GPT_exponents_multiplication_exponents_power_exponents_distributive_l1612_161233


namespace NUMINAMATH_GPT_solution_of_system_l1612_161266

theorem solution_of_system :
  ∃ x y z : ℚ,
    x + 2 * y = 12 ∧
    y + 3 * z = 15 ∧
    3 * x - z = 6 ∧
    x = 54 / 17 ∧
    y = 75 / 17 ∧
    z = 60 / 17 :=
by
  exists 54 / 17, 75 / 17, 60 / 17
  repeat { sorry }

end NUMINAMATH_GPT_solution_of_system_l1612_161266


namespace NUMINAMATH_GPT_incorrect_value_at_x5_l1612_161282

theorem incorrect_value_at_x5 
  (f : ℕ → ℕ) 
  (provided_values : List ℕ) 
  (h_f : ∀ x, f x = 2 * x ^ 2 + 3 * x + 5)
  (h_provided_values : provided_values = [10, 18, 29, 44, 63, 84, 111, 140]) : 
  ¬ (f 5 = provided_values.get! 4) := 
by
  sorry

end NUMINAMATH_GPT_incorrect_value_at_x5_l1612_161282


namespace NUMINAMATH_GPT_infinitely_many_good_pairs_l1612_161238

def is_triangular (t : ℕ) : Prop :=
  ∃ n : ℕ, t = n * (n + 1) / 2

theorem infinitely_many_good_pairs :
  ∃ (a b : ℕ), (0 < a) ∧ (0 < b) ∧ 
  ∀ t : ℕ, is_triangular t ↔ is_triangular (a * t + b) :=
sorry

end NUMINAMATH_GPT_infinitely_many_good_pairs_l1612_161238


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1612_161257

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (2 * x + Real.pi / 4))

theorem monotonic_decreasing_interval :
  ∀ (x1 x2 : ℝ), (-Real.pi / 8) < x1 ∧ x1 < Real.pi / 8 ∧ (-Real.pi / 8) < x2 ∧ x2 < Real.pi / 8 ∧ x1 < x2 →
  f x1 > f x2 :=
sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1612_161257


namespace NUMINAMATH_GPT_correct_calculation_l1612_161278

theorem correct_calculation (x a : Real) :
  (3 * x^2 - x^2 ≠ 3) → 
  (-3 * a^2 - 2 * a^2 ≠ -a^2) →
  (x^3 / x ≠ 3) → 
  ((-x)^3 = -x^3) → 
  true :=
by
  intros _ _ _ _
  trivial

end NUMINAMATH_GPT_correct_calculation_l1612_161278


namespace NUMINAMATH_GPT_robot_Y_reaches_B_after_B_reaches_A_l1612_161294

-- Definitions for the setup of the problem
def time_J_to_B (t_J_to_B : ℕ) := t_J_to_B = 12
def time_J_catch_up_B (t_J_catch_up_B : ℕ) := t_J_catch_up_B = 9

-- Main theorem to be proved
theorem robot_Y_reaches_B_after_B_reaches_A : 
  ∀ t_J_to_B t_J_catch_up_B, 
    (time_J_to_B t_J_to_B) → 
    (time_J_catch_up_B t_J_catch_up_B) →
    ∃ t : ℕ, t = 56 :=
by 
  sorry

end NUMINAMATH_GPT_robot_Y_reaches_B_after_B_reaches_A_l1612_161294


namespace NUMINAMATH_GPT_find_first_train_length_l1612_161222

namespace TrainProblem

-- Define conditions
def speed_first_train_kmph := 42
def speed_second_train_kmph := 48
def length_second_train_m := 163
def time_clear_s := 12
def relative_speed_kmph := speed_first_train_kmph + speed_second_train_kmph

-- Convert kmph to m/s
def kmph_to_mps(kmph : ℕ) : ℕ := kmph * 5 / 18
def relative_speed_mps := kmph_to_mps relative_speed_kmph

-- Calculate total distance covered by the trains in meters
def total_distance_m := relative_speed_mps * time_clear_s

-- Define the length of the first train to be proved
def length_first_train_m := 137

-- Theorem statement
theorem find_first_train_length :
  total_distance_m = length_first_train_m + length_second_train_m :=
sorry

end TrainProblem

end NUMINAMATH_GPT_find_first_train_length_l1612_161222


namespace NUMINAMATH_GPT_allan_correct_answers_l1612_161220

theorem allan_correct_answers (x y : ℕ) (h1 : x + y = 120) (h2 : x - (0.25 : ℝ) * y = 100) : x = 104 :=
by
  sorry

end NUMINAMATH_GPT_allan_correct_answers_l1612_161220


namespace NUMINAMATH_GPT_a_plus_b_eq_zero_l1612_161267

-- Define the universal set and the relevant sets
def U : Set ℝ := Set.univ
def M (a : ℝ) : Set ℝ := {x | x^2 + a * x ≤ 0}
def C_U_M (b : ℝ) : Set ℝ := {x | x > b ∨ x < 0}

-- Define the proof theorem
theorem a_plus_b_eq_zero (a b : ℝ) (h1 : ∀ x, x ∈ M a ↔ -a < x ∧ x < 0 ∨ 0 < x ∧ x < -a)
                         (h2 : ∀ x, x ∈ C_U_M b ↔ x > b ∨ x < 0) : a + b = 0 := 
sorry

end NUMINAMATH_GPT_a_plus_b_eq_zero_l1612_161267


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1612_161260

noncomputable def geometric_sum (a q : ℕ) (n : ℕ) : ℕ :=
  a * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem (a : ℕ) (q : ℕ) (n : ℕ) (h_q : q = 2) (h_n : n = 4) :
  (geometric_sum a q 4) / (a * q) = 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1612_161260


namespace NUMINAMATH_GPT_fraction_simplification_l1612_161232

theorem fraction_simplification :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) /
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1612_161232


namespace NUMINAMATH_GPT_decimal_representation_of_7_div_12_l1612_161239

theorem decimal_representation_of_7_div_12 : (7 / 12 : ℚ) = 0.58333333 := 
sorry

end NUMINAMATH_GPT_decimal_representation_of_7_div_12_l1612_161239


namespace NUMINAMATH_GPT_intersection_point_l1612_161202

/-- Coordinates of points A, B, C, and D -/
def pointA : Fin 3 → ℝ := ![3, -2, 4]
def pointB : Fin 3 → ℝ := ![13, -12, 9]
def pointC : Fin 3 → ℝ := ![1, 6, -8]
def pointD : Fin 3 → ℝ := ![3, -1, 2]

/-- Prove the intersection point of the lines AB and CD is (-7, 8, -1) -/
theorem intersection_point :
  let lineAB (t : ℝ) := pointA + t • (pointB - pointA)
  let lineCD (s : ℝ) := pointC + s • (pointD - pointC)
  ∃ t s : ℝ, lineAB t = lineCD s ∧ lineAB t = ![-7, 8, -1] :=
sorry

end NUMINAMATH_GPT_intersection_point_l1612_161202


namespace NUMINAMATH_GPT_sum_of_angles_is_90_l1612_161237

variables (α β γ : ℝ)
-- Given angles marked on squared paper, which imply certain geometric properties
axiom angle_properties : α + β + γ = 90

theorem sum_of_angles_is_90 : α + β + γ = 90 := 
by
  apply angle_properties

end NUMINAMATH_GPT_sum_of_angles_is_90_l1612_161237


namespace NUMINAMATH_GPT_all_equal_l1612_161240

theorem all_equal (n : ℕ) (a : ℕ → ℝ) (h1 : 3 < n)
  (h2 : ∀ k : ℕ, k < n -> (a k)^3 = (a (k + 1 % n))^2 + (a (k + 2 % n))^2 + (a (k + 3 % n))^2) : 
  ∀ i j : ℕ, i < n -> j < n -> a i = a j :=
by
  sorry

end NUMINAMATH_GPT_all_equal_l1612_161240


namespace NUMINAMATH_GPT_cherries_per_pound_l1612_161256

-- Definitions from conditions in the problem
def total_pounds_of_cherries : ℕ := 3
def pitting_time_for_20_cherries : ℕ := 10 -- in minutes
def total_pitting_time : ℕ := 2 * 60  -- in minutes (2 hours to minutes)

-- Theorem to prove the question equals the correct answer
theorem cherries_per_pound : (total_pitting_time / pitting_time_for_20_cherries) * 20 / total_pounds_of_cherries = 80 := by
  sorry

end NUMINAMATH_GPT_cherries_per_pound_l1612_161256


namespace NUMINAMATH_GPT_problem1_l1612_161281

theorem problem1 (a b : ℝ) : (a - b)^3 + 3 * a * b * (a - b) + b^3 - a^3 = 0 :=
sorry

end NUMINAMATH_GPT_problem1_l1612_161281


namespace NUMINAMATH_GPT_walk_direction_east_l1612_161234

theorem walk_direction_east (m : ℤ) (h : m = -2023) : m = -(-2023) :=
by
  sorry

end NUMINAMATH_GPT_walk_direction_east_l1612_161234


namespace NUMINAMATH_GPT_find_equation_of_line_midpoint_find_equation_of_line_vector_l1612_161291

-- Definition for Problem 1
def equation_of_line_midpoint (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (x / (-6) + y / 2 = 1) ∧ l (-3) = 1)

-- Proof Statement for Problem 1
theorem find_equation_of_line_midpoint : equation_of_line_midpoint (-6) 2 :=
sorry

-- Definition for Problem 2
def equation_of_line_vector (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (y - 1) / (-1) = (x + 3) / (-6) ∧ l (-3) = 1)

-- Proof Statement for Problem 2
theorem find_equation_of_line_vector : equation_of_line_vector (-9) (3 / 2) :=
sorry

end NUMINAMATH_GPT_find_equation_of_line_midpoint_find_equation_of_line_vector_l1612_161291


namespace NUMINAMATH_GPT_cos_b_eq_one_div_sqrt_two_l1612_161263

variable {a b c : ℝ} -- Side lengths
variable {A B C : ℝ} -- Angles in radians

-- Conditions of the problem
variables (h1 : c = 2 * a) 
          (h2 : b^2 = a * c) 
          (h3 : a^2 + b^2 = c^2 - 2 * a * b * Real.cos C)
          (h4 : A + B + C = Real.pi)

theorem cos_b_eq_one_div_sqrt_two
    (h1 : c = 2 * a)
    (h2 : b = a * Real.sqrt 2)
    (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
    (h4 : A + B + C = Real.pi )
    : Real.cos B = 1 / Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_cos_b_eq_one_div_sqrt_two_l1612_161263


namespace NUMINAMATH_GPT_carter_drum_stick_sets_l1612_161269

theorem carter_drum_stick_sets (sets_per_show sets_tossed_per_show nights : ℕ) :
  sets_per_show = 5 →
  sets_tossed_per_show = 6 →
  nights = 30 →
  (sets_per_show + sets_tossed_per_show) * nights = 330 := by
  intros
  sorry

end NUMINAMATH_GPT_carter_drum_stick_sets_l1612_161269


namespace NUMINAMATH_GPT_diagonals_of_angle_bisectors_l1612_161226

theorem diagonals_of_angle_bisectors (a b : ℝ) (BAD ABC : ℝ) (hBAD : BAD = ABC) :
  ∃ d : ℝ, d = |a - b| :=
by
  sorry

end NUMINAMATH_GPT_diagonals_of_angle_bisectors_l1612_161226


namespace NUMINAMATH_GPT_coefficient_c_nonzero_l1612_161283

-- We are going to define the given polynomial and its conditions
def P (x : ℝ) (a b c d e : ℝ) : ℝ :=
  x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

-- Given conditions
def five_x_intercepts (P : ℝ → ℝ) (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0 ∧ P x4 = 0 ∧ P x5 = 0

def double_root_at_zero (P : ℝ → ℝ) : Prop :=
  P 0 = 0 ∧ deriv P 0 = 0

-- Equivalent proof problem
theorem coefficient_c_nonzero (a b c d e : ℝ)
  (h1 : P 0 a b c d e = 0)
  (h2 : deriv (P · a b c d e) 0 = 0)
  (h3 : ∀ x, P x a b c d e = x^2 * (x - 1) * (x - 2) * (x - 3))
  (h4 : ∀ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) : 
  c ≠ 0 := 
sorry

end NUMINAMATH_GPT_coefficient_c_nonzero_l1612_161283


namespace NUMINAMATH_GPT_product_of_differences_of_squares_is_diff_of_square_l1612_161298

-- Define when an integer is a difference of squares of positive integers
def diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ n = x^2 - y^2

-- State the main theorem
theorem product_of_differences_of_squares_is_diff_of_square 
  (a b c d : ℕ) (h₁ : diff_of_squares a) (h₂ : diff_of_squares b) (h₃ : diff_of_squares c) (h₄ : diff_of_squares d) : 
  diff_of_squares (a * b * c * d) := by
  sorry

end NUMINAMATH_GPT_product_of_differences_of_squares_is_diff_of_square_l1612_161298


namespace NUMINAMATH_GPT_arithmetic_sequence_a13_l1612_161219

theorem arithmetic_sequence_a13 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 5 = 3) (h2 : a 9 = 6) 
  (h3 : ∀ n, a n = a1 + (n - 1) * d) : 
  a 13 = 9 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a13_l1612_161219


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1612_161231

theorem right_triangle_hypotenuse (a b : ℕ) (a_val : a = 4) (b_val : b = 5) :
    ∃ c : ℝ, c^2 = (a:ℝ)^2 + (b:ℝ)^2 ∧ c = Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1612_161231


namespace NUMINAMATH_GPT_largest_study_only_Biology_l1612_161297

-- Let's define the total number of students
def total_students : ℕ := 500

-- Define the given conditions
def S : ℕ := 65 * total_students / 100
def M : ℕ := 55 * total_students / 100
def B : ℕ := 50 * total_students / 100
def P : ℕ := 15 * total_students / 100

def MS : ℕ := 35 * total_students / 100
def MB : ℕ := 25 * total_students / 100
def BS : ℕ := 20 * total_students / 100
def MSB : ℕ := 10 * total_students / 100

-- Required to prove that the largest number of students who study only Biology is 75
theorem largest_study_only_Biology : 
  (B - MB - BS + MSB) = 75 :=
by 
  sorry

end NUMINAMATH_GPT_largest_study_only_Biology_l1612_161297


namespace NUMINAMATH_GPT_bruce_paid_correct_amount_l1612_161208

-- Define the conditions
def kg_grapes : ℕ := 8
def cost_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 8
def cost_per_kg_mangoes : ℕ := 55

-- Calculate partial costs
def cost_grapes := kg_grapes * cost_per_kg_grapes
def cost_mangoes := kg_mangoes * cost_per_kg_mangoes
def total_paid := cost_grapes + cost_mangoes

-- The theorem to prove
theorem bruce_paid_correct_amount : total_paid = 1000 := 
by 
  -- Merge several logical steps into one
  -- sorry can be used for incomplete proof
  sorry

end NUMINAMATH_GPT_bruce_paid_correct_amount_l1612_161208


namespace NUMINAMATH_GPT_factorize_1_factorize_2_l1612_161243

-- Define the variables involved
variables (a x y : ℝ)

-- Problem (1): 18a^2 - 32 = 2 * (3a + 4) * (3a - 4)
theorem factorize_1 (a : ℝ) : 
  18 * a^2 - 32 = 2 * (3 * a + 4) * (3 * a - 4) :=
sorry

-- Problem (2): y - 6xy + 9x^2y = y * (1 - 3x) ^ 2
theorem factorize_2 (x y : ℝ) : 
  y - 6 * x * y + 9 * x^2 * y = y * (1 - 3 * x) ^ 2 :=
sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_l1612_161243


namespace NUMINAMATH_GPT_position_of_2010_is_correct_l1612_161211

-- Definition of the arithmetic sequence and row starting points
def first_term : Nat := 1
def common_difference : Nat := 2
def S (n : Nat) : Nat := (n * (2 * first_term + (n - 1) * common_difference)) / 2

-- Definition of the position where number 2010 appears
def row_of_number (x : Nat) : Nat :=
  let n := (Nat.sqrt x) + 1
  if (n - 1) * (n - 1) < x && x <= n * n then n else n - 1

def column_of_number (x : Nat) : Nat :=
  let row := row_of_number x
  x - (S (row - 1)) + 1

-- Main theorem
theorem position_of_2010_is_correct :
  row_of_number 2010 = 45 ∧ column_of_number 2010 = 74 :=
by
  sorry

end NUMINAMATH_GPT_position_of_2010_is_correct_l1612_161211


namespace NUMINAMATH_GPT_candies_total_l1612_161284

theorem candies_total (N a S : ℕ) (h1 : S = 2 * a + 7) (h2 : S = N * a) (h3 : a > 1) (h4 : N = 3) : S = 21 := 
sorry

end NUMINAMATH_GPT_candies_total_l1612_161284


namespace NUMINAMATH_GPT_sheila_weekly_earnings_l1612_161227

-- Defining the conditions
def hourly_wage : ℕ := 12
def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def hours_tt : ℕ := 6
def days_tt : ℕ := 2

-- Defining Sheila's total weekly earnings
noncomputable def weekly_earnings := (hours_mwf * hourly_wage * days_mwf) + (hours_tt * hourly_wage * days_tt)

-- The statement of the proof
theorem sheila_weekly_earnings : weekly_earnings = 432 :=
by
  sorry

end NUMINAMATH_GPT_sheila_weekly_earnings_l1612_161227


namespace NUMINAMATH_GPT_least_clock_equivalent_l1612_161229

theorem least_clock_equivalent (x : ℕ) : 
  x > 3 ∧ x % 12 = (x * x) % 12 → x = 12 := 
by
  sorry

end NUMINAMATH_GPT_least_clock_equivalent_l1612_161229


namespace NUMINAMATH_GPT_probability_not_passing_l1612_161209

theorem probability_not_passing (P_passing : ℚ) (h : P_passing = 4/7) : (1 - P_passing = 3/7) :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_probability_not_passing_l1612_161209


namespace NUMINAMATH_GPT_points_on_quadratic_l1612_161276

theorem points_on_quadratic (c y₁ y₂ : ℝ) 
  (hA : y₁ = (-1)^2 - 6*(-1) + c) 
  (hB : y₂ = 2^2 - 6*2 + c) : y₁ > y₂ := 
  sorry

end NUMINAMATH_GPT_points_on_quadratic_l1612_161276


namespace NUMINAMATH_GPT_n_divisibility_and_factors_l1612_161271

open Nat

theorem n_divisibility_and_factors (n : ℕ) (h1 : 1990 ∣ n) (h2 : ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n):
  n = 4 * 5 * 199 ∨ n = 2 * 25 * 199 ∨ n = 2 * 5 * 39601 := 
sorry

end NUMINAMATH_GPT_n_divisibility_and_factors_l1612_161271


namespace NUMINAMATH_GPT_handshake_count_l1612_161205

-- Define the number of team members, referees, and the total number of handshakes
def num_team_members := 7
def num_referees := 3
def num_coaches := 2

-- Calculate the handshakes
def team_handshakes := num_team_members * num_team_members
def player_refhandshakes := (2 * num_team_members) * num_referees
def coach_handshakes := num_coaches * (2 * num_team_members + num_referees)

-- The total number of handshakes
def total_handshakes := team_handshakes + player_refhandshakes + coach_handshakes

-- The proof statement
theorem handshake_count : total_handshakes = 125 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_handshake_count_l1612_161205


namespace NUMINAMATH_GPT_negation_example_l1612_161204

theorem negation_example (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x0 : ℝ, x0 > 0 ∧ (x0 + 1) * Real.exp x0 ≤ 1 :=
sorry

end NUMINAMATH_GPT_negation_example_l1612_161204


namespace NUMINAMATH_GPT_base_length_of_parallelogram_l1612_161206

theorem base_length_of_parallelogram 
  (area : ℝ) (base altitude : ℝ) 
  (h_area : area = 242)
  (h_altitude : altitude = 2 * base) :
  base = 11 :=
by
  sorry

end NUMINAMATH_GPT_base_length_of_parallelogram_l1612_161206


namespace NUMINAMATH_GPT_convert_13_to_binary_l1612_161295

theorem convert_13_to_binary : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by sorry

end NUMINAMATH_GPT_convert_13_to_binary_l1612_161295


namespace NUMINAMATH_GPT_sum_series_eq_half_l1612_161212

theorem sum_series_eq_half :
  ∑' n : ℕ, (3^(n+1) / (9^(n+1) - 1)) = 1/2 := 
sorry

end NUMINAMATH_GPT_sum_series_eq_half_l1612_161212


namespace NUMINAMATH_GPT_dilation_image_l1612_161253

open Complex

theorem dilation_image (z₀ : ℂ) (c : ℂ) (k : ℝ) (z : ℂ)
    (h₀ : z₀ = 0 - 2*I) (h₁ : c = 1 + 2*I) (h₂ : k = 2) :
    z = -1 - 6*I :=
by
  sorry

end NUMINAMATH_GPT_dilation_image_l1612_161253


namespace NUMINAMATH_GPT_original_price_of_shoes_l1612_161287

noncomputable def original_price (final_price : ℝ) (sales_tax : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  final_price / sales_tax / (discount1 * discount2)

theorem original_price_of_shoes :
  original_price 51 1.07 0.40 0.85 = 140.18 := by
    have h_pre_tax_price : 47.66 = 51 / 1.07 := sorry
    have h_price_relation : 47.66 = 0.85 * 0.40 * 140.18 := sorry
    sorry

end NUMINAMATH_GPT_original_price_of_shoes_l1612_161287


namespace NUMINAMATH_GPT_no_integer_solution_l1612_161296

theorem no_integer_solution (x y : ℤ) : 2 * x + 6 * y ≠ 91 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_l1612_161296


namespace NUMINAMATH_GPT_find_quartic_polynomial_l1612_161268

noncomputable def p (x : ℝ) : ℝ := -(1 / 9) * x^4 + (40 / 9) * x^3 - 8 * x^2 + 10 * x + 2

theorem find_quartic_polynomial :
  p 1 = -3 ∧
  p 2 = -1 ∧
  p 3 = 1 ∧
  p 4 = -7 ∧
  p 0 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_quartic_polynomial_l1612_161268


namespace NUMINAMATH_GPT_center_of_circle_l1612_161217

theorem center_of_circle (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y = 4 → (x, y) = (4, 2) :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l1612_161217


namespace NUMINAMATH_GPT_f_of_7_l1612_161285

theorem f_of_7 (f : ℝ → ℝ) (h : ∀ (x : ℝ), f (4 * x - 1) = x^2 + 2 * x + 2) :
    f 7 = 10 := by
  sorry

end NUMINAMATH_GPT_f_of_7_l1612_161285


namespace NUMINAMATH_GPT_Amanda_ticket_sales_goal_l1612_161249

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end NUMINAMATH_GPT_Amanda_ticket_sales_goal_l1612_161249


namespace NUMINAMATH_GPT_small_pizza_slices_l1612_161225

-- Definitions based on conditions
def large_pizza_slices : ℕ := 16
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Statement to prove
theorem small_pizza_slices (S : ℕ) (H : num_large_pizzas * large_pizza_slices + num_small_pizzas * S = total_slices_eaten) : S = 8 :=
by
  sorry

end NUMINAMATH_GPT_small_pizza_slices_l1612_161225


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1612_161247

def n_weight : ℝ := 14.01
def h_weight : ℝ := 1.01
def br_weight : ℝ := 79.90

def molecular_weight : ℝ := (1 * n_weight) + (4 * h_weight) + (1 * br_weight)

theorem molecular_weight_of_compound :
  molecular_weight = 97.95 :=
by
  -- proof steps go here if needed, but currently, we use sorry to complete the theorem
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1612_161247


namespace NUMINAMATH_GPT_shopkeeper_profit_percent_l1612_161215

noncomputable def profit_percent : ℚ := 
let cp_each := 1       -- Cost price of each article
let sp_each := 1.2     -- Selling price of each article without discount
let discount := 0.05   -- 5% discount
let tax := 0.10        -- 10% sales tax
let articles := 30     -- Number of articles
let cp_total := articles * cp_each      -- Total cost price
let sp_after_discount := sp_each * (1 - discount)    -- Selling price after discount
let revenue_before_tax := articles * sp_after_discount   -- Total revenue before tax
let tax_amount := revenue_before_tax * tax   -- Sales tax amount
let revenue_after_tax := revenue_before_tax + tax_amount -- Total revenue after tax
let profit := revenue_after_tax - cp_total -- Profit
(profit / cp_total) * 100 -- Profit percent

theorem shopkeeper_profit_percent : profit_percent = 25.4 :=
by
  -- Here follows the proof based on the conditions and steps above
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percent_l1612_161215


namespace NUMINAMATH_GPT_find_initial_amount_l1612_161261

-- defining conditions
def compound_interest (A P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  A - P

-- main theorem to prove the principal amount
theorem find_initial_amount 
  (A P : ℝ) (r : ℝ)
  (n t : ℕ)
  (h_P : A = P * (1 + r / n)^t)
  (compound_interest_eq : A - P = 1785.98)
  (r_eq : r = 0.20)
  (n_eq : n = 1)
  (t_eq : t = 5) :
  P = 1200 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_amount_l1612_161261


namespace NUMINAMATH_GPT_slices_per_friend_l1612_161210

theorem slices_per_friend (n : ℕ) (h1 : n > 0)
    (h2 : ∀ i : ℕ, i < n → (15 + 18 + 20 + 25) = 78 * n) :
    78 = (15 + 18 + 20 + 25) / n := 
by
  sorry

end NUMINAMATH_GPT_slices_per_friend_l1612_161210


namespace NUMINAMATH_GPT_sin_alpha_plus_pi_over_4_tan_double_alpha_l1612_161224

-- Definitions of sin and tan 
open Real

variable (α : ℝ)

-- Given conditions
axiom α_in_interval : 0 < α ∧ α < π / 2
axiom sin_alpha_def : sin α = sqrt 5 / 5

-- Statement to prove
theorem sin_alpha_plus_pi_over_4 : sin (α + π / 4) = 3 * sqrt 10 / 10 :=
by
  sorry

theorem tan_double_alpha : tan (2 * α) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_plus_pi_over_4_tan_double_alpha_l1612_161224


namespace NUMINAMATH_GPT_percent_defective_units_shipped_l1612_161273

variable (P : Real)
variable (h1 : 0.07 * P = d)
variable (h2 : 0.0035 * P = s)

theorem percent_defective_units_shipped (h1 : 0.07 * P = d) (h2 : 0.0035 * P = s) : 
  (s / d) * 100 = 5 := sorry

end NUMINAMATH_GPT_percent_defective_units_shipped_l1612_161273


namespace NUMINAMATH_GPT_dishonest_dealer_profit_l1612_161293

theorem dishonest_dealer_profit (cost_weight actual_weight : ℝ) (kg_in_g : ℝ) 
  (h1 : cost_weight = 1000) (h2 : actual_weight = 920) (h3 : kg_in_g = 1000) :
  ((cost_weight - actual_weight) / actual_weight) * 100 = 8.7 := by
  sorry

end NUMINAMATH_GPT_dishonest_dealer_profit_l1612_161293


namespace NUMINAMATH_GPT_range_of_m_l1612_161280

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0 
def neg_q_sufficient_for_neg_p (m : ℝ) : Prop :=
  ∀ x : ℝ, p x → q x m

theorem range_of_m (m : ℝ) : neg_q_sufficient_for_neg_p m → m ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1612_161280


namespace NUMINAMATH_GPT_max_b_for_integer_solutions_l1612_161264

theorem max_b_for_integer_solutions (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
sorry

end NUMINAMATH_GPT_max_b_for_integer_solutions_l1612_161264


namespace NUMINAMATH_GPT_degree_of_g_l1612_161200

theorem degree_of_g (f g : Polynomial ℝ) (h : Polynomial ℝ) (H1 : h = f.comp g + g) 
  (H2 : h.natDegree = 6) (H3 : f.natDegree = 3) : g.natDegree = 2 := 
sorry

end NUMINAMATH_GPT_degree_of_g_l1612_161200


namespace NUMINAMATH_GPT_tomTotalWeightMoved_is_525_l1612_161254

-- Tom's weight
def tomWeight : ℝ := 150

-- Weight in each hand
def weightInEachHand : ℝ := 1.5 * tomWeight

-- Weight vest
def weightVest : ℝ := 0.5 * tomWeight

-- Total weight moved
def totalWeightMoved : ℝ := (weightInEachHand * 2) + weightVest

theorem tomTotalWeightMoved_is_525 : totalWeightMoved = 525 := by
  sorry

end NUMINAMATH_GPT_tomTotalWeightMoved_is_525_l1612_161254


namespace NUMINAMATH_GPT_g_neg_3_eq_neg_9_l1612_161277

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Given functions and values
variables (f g : ℝ → ℝ) (h_even : is_even_function f) (h_f_g : ∀ x, f x = g x - 2 * x)
variables (h_g3 : g 3 = 3)

-- Goal: Prove that g (-3) = -9
theorem g_neg_3_eq_neg_9 : g (-3) = -9 :=
sorry

end NUMINAMATH_GPT_g_neg_3_eq_neg_9_l1612_161277


namespace NUMINAMATH_GPT_cafe_table_count_l1612_161235

theorem cafe_table_count (cafe_seats_base7 : ℕ) (seats_per_table : ℕ) (cafe_seats_base10 : ℕ)
    (h1 : cafe_seats_base7 = 3 * 7^2 + 1 * 7^1 + 2 * 7^0) 
    (h2 : seats_per_table = 3) : cafe_seats_base10 = 156 ∧ (cafe_seats_base10 / seats_per_table) = 52 := 
by {
  sorry
}

end NUMINAMATH_GPT_cafe_table_count_l1612_161235


namespace NUMINAMATH_GPT_degree_equality_l1612_161251

theorem degree_equality (m : ℕ) :
  (∀ x y z : ℕ, 2 + 4 = 1 + (m + 2)) → 3 * m - 2 = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_degree_equality_l1612_161251


namespace NUMINAMATH_GPT_range_of_a_l1612_161255

theorem range_of_a (x a : ℝ) 
  (h₁ : ∀ x, |x + 1| ≤ 2 → x ≤ a) 
  (h₂ : ∃ x, x > a ∧ |x + 1| ≤ 2) 
  : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1612_161255


namespace NUMINAMATH_GPT_matrix_power_101_l1612_161241

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

theorem matrix_power_101 :
  B ^ (101 : ℕ) = B := sorry

end NUMINAMATH_GPT_matrix_power_101_l1612_161241


namespace NUMINAMATH_GPT_total_wristbands_proof_l1612_161279

-- Definitions from the conditions
def wristbands_per_person : ℕ := 2
def total_wristbands : ℕ := 125

-- Theorem statement to be proved
theorem total_wristbands_proof : total_wristbands = 125 :=
by
  sorry

end NUMINAMATH_GPT_total_wristbands_proof_l1612_161279


namespace NUMINAMATH_GPT_find_ac_pair_l1612_161213

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_ac_pair_l1612_161213


namespace NUMINAMATH_GPT_probability_black_white_l1612_161201

structure Jar :=
  (black_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)

def total_balls (j : Jar) : ℕ :=
  j.black_balls + j.white_balls + j.green_balls

def choose (n k : ℕ) : ℕ := n.choose k

theorem probability_black_white (j : Jar) (h_black : j.black_balls = 3) (h_white : j.white_balls = 3) (h_green : j.green_balls = 1) :
  (choose 3 1 * choose 3 1) / (choose (total_balls j) 2) = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_black_white_l1612_161201


namespace NUMINAMATH_GPT_distance_problem_l1612_161274

-- Define the problem
theorem distance_problem
  (x y : ℝ)
  (h1 : x + y = 21)
  (h2 : x / 60 + 21 / 60 = 10 / 60 + y / 4) :
  x = 19 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_problem_l1612_161274


namespace NUMINAMATH_GPT_mary_max_earnings_l1612_161221

theorem mary_max_earnings
  (max_hours : ℕ)
  (regular_rate : ℕ)
  (overtime_rate_increase_percent : ℕ)
  (first_hours : ℕ)
  (total_max_hours : ℕ)
  (total_hours_payable : ℕ) :
  max_hours = 60 →
  regular_rate = 8 →
  overtime_rate_increase_percent = 25 →
  first_hours = 20 →
  total_max_hours = 60 →
  total_hours_payable = 560 →
  ((first_hours * regular_rate) + ((total_max_hours - first_hours) * (regular_rate + (regular_rate * overtime_rate_increase_percent / 100)))) = total_hours_payable :=
by
  intros
  sorry

end NUMINAMATH_GPT_mary_max_earnings_l1612_161221


namespace NUMINAMATH_GPT_ab_value_l1612_161288

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := 
by
  sorry

end NUMINAMATH_GPT_ab_value_l1612_161288


namespace NUMINAMATH_GPT_find_mean_of_two_l1612_161230

-- Define the set of numbers
def numbers : List ℕ := [1879, 1997, 2023, 2029, 2113, 2125]

-- Define the mean of the four selected numbers
def mean_of_four : ℕ := 2018

-- Define the sum of all numbers
def total_sum : ℕ := numbers.sum

-- Define the sum of the four numbers with a given mean
def sum_of_four : ℕ := 4 * mean_of_four

-- Define the sum of the remaining two numbers
def sum_of_two (total sum_of_four : ℕ) : ℕ := total - sum_of_four

-- Define the mean of the remaining two numbers
def mean_of_two (sum_two : ℕ) : ℕ := sum_two / 2

-- Define the condition theorem to be proven
theorem find_mean_of_two : mean_of_two (sum_of_two total_sum sum_of_four) = 2047 := 
by
  sorry

end NUMINAMATH_GPT_find_mean_of_two_l1612_161230
