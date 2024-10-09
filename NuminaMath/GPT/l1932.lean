import Mathlib

namespace children_in_circle_l1932_193252

theorem children_in_circle (n m : ℕ) (k : ℕ) 
  (h1 : n = m) 
  (h2 : n + m = 2 * k) :
  ∃ k', n + m = 4 * k' :=
by
  sorry

end children_in_circle_l1932_193252


namespace no_solution_in_natural_numbers_l1932_193257

theorem no_solution_in_natural_numbers (x y z : ℕ) : ¬((2 * x) ^ (2 * x) - 1 = y ^ (z + 1)) := 
  sorry

end no_solution_in_natural_numbers_l1932_193257


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l1932_193275

variable (U A B : Set ℝ)
variable (x : ℝ)

def universal_set := { x | x ≤ 4 }
def set_A := { x | -2 < x ∧ x < 3 }
def set_B := { x | -3 < x ∧ x ≤ 3 }

theorem complement_U_A : (U \ A) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem intersection_A_B : (A ∩ B) = { x | -2 < x ∧ x < 3 } := sorry

theorem complement_U_intersection_A_B : (U \ (A ∩ B)) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem complement_U_A_intersection_B : ((U \ A) ∩ B) = { x | -3 < x ∧ x ≤ -2 ∨ x = 3 } := sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l1932_193275


namespace find_k_l1932_193296

theorem find_k (k : ℝ) (h : (-3 : ℝ)^2 + (-3 : ℝ) - k = 0) : k = 6 :=
by
  sorry

end find_k_l1932_193296


namespace one_eq_one_of_ab_l1932_193284

variable {a b : ℝ}

theorem one_eq_one_of_ab (h : a * b = a^2 - a * b + b^2) : 1 = 1 := by
  sorry

end one_eq_one_of_ab_l1932_193284


namespace number_of_pens_bought_l1932_193241

theorem number_of_pens_bought 
  (P : ℝ) -- Marked price of one pen
  (N : ℝ) -- Number of pens bought
  (discount : ℝ := 0.01)
  (profit_percent : ℝ := 29.130434782608695)
  (Total_Cost := 46 * P)
  (Selling_Price_per_Pen := P * (1 - discount))
  (Total_Revenue := N * Selling_Price_per_Pen)
  (Profit := Total_Revenue - Total_Cost)
  (actual_profit_percent := (Profit / Total_Cost) * 100) :
  actual_profit_percent = profit_percent → N = 60 := 
by 
  intro h
  sorry

end number_of_pens_bought_l1932_193241


namespace determine_right_triangle_l1932_193218

variable (A B C : ℝ)
variable (AB BC AC : ℝ)

-- Conditions as definitions
def condition1 : Prop := A + C = B
def condition2 : Prop := A = 30 ∧ B = 60 ∧ C = 90 -- Since ratio 1:2:3 means A = 30, B = 60, C = 90

-- Proof problem statement
theorem determine_right_triangle (h1 : condition1 A B C) (h2 : condition2 A B C) : (B = 90) :=
sorry

end determine_right_triangle_l1932_193218


namespace owl_cost_in_gold_l1932_193225

-- Definitions for conditions
def spellbook_cost_gold := 5
def potionkit_cost_silver := 20
def num_spellbooks := 5
def num_potionkits := 3
def silver_per_gold := 9
def total_payment_silver := 537

-- Function to convert gold to silver
def gold_to_silver (gold : ℕ) : ℕ := gold * silver_per_gold

-- Function to compute total cost in silver for spellbooks and potion kits
def total_spellbook_cost_silver : ℕ :=
  gold_to_silver spellbook_cost_gold * num_spellbooks

def total_potionkit_cost_silver : ℕ :=
  potionkit_cost_silver * num_potionkits

-- Function to calculate the cost of the owl in silver
def owl_cost_silver : ℕ :=
  total_payment_silver - (total_spellbook_cost_silver + total_potionkit_cost_silver)

-- Function to convert the owl's cost from silver to gold
def owl_cost_gold : ℕ :=
  owl_cost_silver / silver_per_gold

-- The proof statement
theorem owl_cost_in_gold : owl_cost_gold = 28 :=
  by
    sorry

end owl_cost_in_gold_l1932_193225


namespace find_c1_in_polynomial_q_l1932_193266

theorem find_c1_in_polynomial_q
  (m : ℕ)
  (hm : m ≥ 5)
  (hm_odd : m % 2 = 1)
  (D : ℕ → ℕ)
  (hD_q : ∃ (c3 c2 c1 c0 : ℤ), ∀ (m : ℕ), m % 2 = 1 ∧ m ≥ 5 → D m = (c3 * m^3 + c2 * m^2 + c1 * m + c0)) :
  ∃ (c1 : ℤ), c1 = 11 :=
sorry

end find_c1_in_polynomial_q_l1932_193266


namespace sum_of_digits_of_special_number_l1932_193253

theorem sum_of_digits_of_special_number :
  ∀ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ (100 * x + 10 * y + z = x.factorial + y.factorial + z.factorial) →
  (x + y + z = 10) :=
by
  sorry

end sum_of_digits_of_special_number_l1932_193253


namespace locus_of_centers_of_tangent_circles_l1932_193202

noncomputable def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25
noncomputable def locus (a b : ℝ) : Prop := 4 * a^2 + 4 * b^2 - 6 * a - 25 = 0

theorem locus_of_centers_of_tangent_circles :
  (∃ (a b r : ℝ), a^2 + b^2 = (r + 1)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2) →
  (∃ a b : ℝ, locus a b) :=
sorry

end locus_of_centers_of_tangent_circles_l1932_193202


namespace trumpet_cost_l1932_193208

variable (total_amount : ℝ) (book_cost : ℝ)

theorem trumpet_cost (h1 : total_amount = 151) (h2 : book_cost = 5.84) :
  (total_amount - book_cost = 145.16) :=
by
  sorry

end trumpet_cost_l1932_193208


namespace joanie_loan_difference_l1932_193244

theorem joanie_loan_difference:
  let P := 6000
  let r := 0.12
  let t := 4
  let n_quarterly := 4
  let n_annually := 1
  let A_quarterly := P * (1 + r / n_quarterly)^(n_quarterly * t)
  let A_annually := P * (1 + r / n_annually)^t
  A_quarterly - A_annually = 187.12 := sorry

end joanie_loan_difference_l1932_193244


namespace range_of_a_l1932_193283

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (3 / 2)^x = (2 + 3 * a) / (5 - a)) ↔ a ∈ Set.Ioo (-2 / 3) (3 / 4) :=
by
  sorry

end range_of_a_l1932_193283


namespace solve_quadratic_eq_l1932_193299

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2 * x = 1) : x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 :=
sorry

end solve_quadratic_eq_l1932_193299


namespace xy_sum_values_l1932_193217

theorem xy_sum_values (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end xy_sum_values_l1932_193217


namespace percentage_increase_l1932_193294

def originalPrice : ℝ := 300
def newPrice : ℝ := 390

theorem percentage_increase :
  ((newPrice - originalPrice) / originalPrice) * 100 = 30 := by
  sorry

end percentage_increase_l1932_193294


namespace num_undef_values_l1932_193250

theorem num_undef_values : 
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, (x^2 + 4 * x - 5) * (x - 4) = 0 → x = -5 ∨ x = 1 ∨ x = 4 :=
by
  -- We are stating that there exists a natural number n such that n = 3
  -- and for all real numbers x, if (x^2 + 4*x - 5)*(x - 4) = 0,
  -- then x must be one of -5, 1, or 4.
  sorry

end num_undef_values_l1932_193250


namespace smallest_k_condition_l1932_193224

theorem smallest_k_condition (n k : ℕ) (h_n : n ≥ 2) (h_k : k = 2 * n) :
  ∀ (f : Fin n → Fin n → Fin k), (∀ i j, f i j < k) →
  (∃ a b c d : Fin n, a ≠ c ∧ b ≠ d ∧ f a b ≠ f a d ∧ f a b ≠ f c b ∧ f a b ≠ f c d ∧ f a d ≠ f c b ∧ f a d ≠ f c d ∧ f c b ≠ f c d) :=
sorry

end smallest_k_condition_l1932_193224


namespace min_value_of_reciprocals_l1932_193247

theorem min_value_of_reciprocals (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  ∃ x, x = (1 / a) + (1 / b) ∧ x ≥ 4 := 
sorry

end min_value_of_reciprocals_l1932_193247


namespace avg_speed_is_65_l1932_193251

theorem avg_speed_is_65
  (speed1: ℕ) (speed2: ℕ) (time1: ℕ) (time2: ℕ)
  (h_speed1: speed1 = 85)
  (h_speed2: speed2 = 45)
  (h_time1: time1 = 1)
  (h_time2: time2 = 1) :
  (speed1 + speed2) / (time1 + time2) = 65 := by
  sorry

end avg_speed_is_65_l1932_193251


namespace find_function_expression_find_range_of_m_l1932_193219

-- Statement for Part 1
theorem find_function_expression (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) : 
  y = -1/2 * x - 2 := 
sorry

-- Statement for Part 2
theorem find_range_of_m (m x : ℝ) (hx : x > -2) (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) :
  (-x + m < -1/2 * x - 2) ↔ (m ≤ -3) := 
sorry

end find_function_expression_find_range_of_m_l1932_193219


namespace additional_payment_each_friend_l1932_193278

theorem additional_payment_each_friend (initial_cost : ℕ) (earned_amount : ℕ) (total_friends : ℕ) (final_friends : ℕ) 
(h_initial_cost : initial_cost = 1700) (h_earned_amount : earned_amount = 500) 
(h_total_friends : total_friends = 6) (h_final_friends : final_friends = 5) : 
  ((initial_cost - earned_amount) / total_friends) / final_friends = 40 :=
sorry

end additional_payment_each_friend_l1932_193278


namespace post_height_l1932_193222

-- Conditions
def spiral_path (circuit_per_rise rise_distance : ℝ) := ∀ (total_distance circ_circumference height : ℝ), 
  circuit_per_rise = total_distance / circ_circumference ∧ 
  height = circuit_per_rise * rise_distance

-- Given conditions
def cylinder_post : Prop := 
  ∀ (total_distance circ_circumference rise_distance : ℝ), 
    spiral_path (total_distance / circ_circumference) rise_distance ∧ 
    circ_circumference = 3 ∧ 
    rise_distance = 4 ∧ 
    total_distance = 12

-- Proof problem: Post height
theorem post_height : cylinder_post → ∃ height : ℝ, height = 16 := 
by sorry

end post_height_l1932_193222


namespace denny_followers_l1932_193265

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end denny_followers_l1932_193265


namespace path_shorter_factor_l1932_193281

-- Declare variables
variables (x y z : ℝ)

-- Define conditions as hypotheses
def condition1 := x = 3 * (y + z)
def condition2 := 4 * y = z + x

-- State the proof statement
theorem path_shorter_factor (condition1 : x = 3 * (y + z)) (condition2 : 4 * y = z + x) :
  (4 * y) / z = 19 :=
sorry

end path_shorter_factor_l1932_193281


namespace range_of_a_l1932_193280

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x = 1) ↔ a ≠ 0 := by
sorry

end range_of_a_l1932_193280


namespace triangle_area_l1932_193254

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x^2 - sqrt 3

theorem triangle_area
  (A : ℝ) (b c : ℝ)
  (h1 : f A = 1)
  (h2 : b * c = 2) 
  (h3 : (b * cos A) * (c * cos A) = sqrt 2) : 
  (1 / 2 * b * c * sin A = sqrt 2 / 2) := 
sorry

end triangle_area_l1932_193254


namespace parabola_opens_upward_l1932_193297

theorem parabola_opens_upward (a : ℝ) (b : ℝ) (h : a > 0) : ∀ x : ℝ, 3*x^2 + 2 = a*x^2 + b → a = 3 ∧ b = 2 → ∀ x : ℝ, 3 * x^2 + 2 ≤ a * x^2 + b := 
by
  sorry

end parabola_opens_upward_l1932_193297


namespace B_catches_up_with_A_l1932_193200

-- Define the conditions
def speed_A : ℝ := 10 -- A's speed in kmph
def speed_B : ℝ := 20 -- B's speed in kmph
def delay : ℝ := 6 -- Delay in hours after A's start

-- Define the total distance where B catches up with A
def distance_catch_up : ℝ := 120

-- Statement to prove B catches up with A at 120 km from the start
theorem B_catches_up_with_A :
  (speed_A * delay + speed_A * (distance_catch_up / speed_B - delay)) = distance_catch_up :=
by
  sorry

end B_catches_up_with_A_l1932_193200


namespace digit_for_multiple_of_9_l1932_193221

theorem digit_for_multiple_of_9 : 
  -- Condition: Sum of the digits 4, 5, 6, 7, 8, and d must be divisible by 9.
  (∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (4 + 5 + 6 + 7 + 8 + d) % 9 = 0) →
  -- Result: The digit d that makes 45678d a multiple of 9 is 6.
  d = 6 :=
by
  sorry

end digit_for_multiple_of_9_l1932_193221


namespace problem_solution_l1932_193274

-- Lean 4 statement of the proof problem
theorem problem_solution (m : ℝ) (U : Set ℝ := Univ) (A : Set ℝ := {x | x^2 + 3*x + 2 = 0}) 
  (B : Set ℝ := {x | x^2 + (m + 1)*x + m = 0}) (h : ∀ x, x ∈ (U \ A) → x ∉ B) : 
  m = 1 ∨ m = 2 :=
by 
  -- This is where the proof would normally go
  sorry

end problem_solution_l1932_193274


namespace least_value_r_minus_p_l1932_193213

theorem least_value_r_minus_p (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 5) :
  ∃ r p, r = 5 ∧ p = 1/2 ∧ r - p = 9 / 2 :=
by
  sorry

end least_value_r_minus_p_l1932_193213


namespace number_of_chairs_borrowed_l1932_193288

-- Define the conditions
def red_chairs := 4
def yellow_chairs := 2 * red_chairs
def blue_chairs := yellow_chairs - 2
def total_initial_chairs : Nat := red_chairs + yellow_chairs + blue_chairs
def chairs_left_in_the_afternoon := 15

-- Define the question
def chairs_borrowed_by_Lisa : Nat := total_initial_chairs - chairs_left_in_the_afternoon

-- The theorem to state the proof problem
theorem number_of_chairs_borrowed : chairs_borrowed_by_Lisa = 3 := by
  -- Proof to be added
  sorry

end number_of_chairs_borrowed_l1932_193288


namespace ria_number_is_2_l1932_193263

theorem ria_number_is_2 
  (R S : ℕ) 
  (consecutive : R = S + 1 ∨ S = R + 1) 
  (R_positive : R > 0) 
  (S_positive : S > 0) 
  (R_not_1 : R ≠ 1) 
  (Sylvie_does_not_know : S ≠ 1) 
  (Ria_knows_after_Sylvie : ∃ (R_known : ℕ), R_known = R) :
  R = 2 :=
sorry

end ria_number_is_2_l1932_193263


namespace cos_2alpha_zero_l1932_193295

theorem cos_2alpha_zero (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
(h : Real.sin (2 * α) = Real.cos (Real.pi / 4 - α)) : 
  Real.cos (2 * α) = 0 :=
by
  sorry

end cos_2alpha_zero_l1932_193295


namespace find_initial_solution_liters_l1932_193228

-- Define the conditions
def percentage_initial_solution_alcohol := 0.26
def added_water := 5
def percentage_new_mixture_alcohol := 0.195

-- Define the initial amount of the solution
def initial_solution_liters (x : ℝ) : Prop :=
  0.26 * x = 0.195 * (x + 5)

-- State the proof problem
theorem find_initial_solution_liters : initial_solution_liters 15 :=
by
  sorry

end find_initial_solution_liters_l1932_193228


namespace last_integer_in_sequence_div3_l1932_193298

theorem last_integer_in_sequence_div3 (a0 : ℤ) (sequence : ℕ → ℤ)
  (h0 : a0 = 1000000000)
  (h_seq : ∀ n, sequence n = a0 / (3^n)) :
  ∃ k, sequence k = 2 ∧ ∀ m, sequence m < 2 → sequence m < 1 := 
sorry

end last_integer_in_sequence_div3_l1932_193298


namespace find_square_subtraction_l1932_193268

theorem find_square_subtraction (x y : ℝ) (h1 : x = Real.sqrt 5) (h2 : y = Real.sqrt 2) : (x - y)^2 = 7 - 2 * Real.sqrt 10 :=
by
  sorry

end find_square_subtraction_l1932_193268


namespace possible_strings_after_moves_l1932_193209

theorem possible_strings_after_moves : 
  let initial_string := "HHMMMMTT"
  let moves := [("HM", "MH"), ("MT", "TM"), ("TH", "HT")]
  let binom := Nat.choose 8 4
  binom = 70 := by
  sorry

end possible_strings_after_moves_l1932_193209


namespace regular_polygon_sides_l1932_193220

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ m : ℕ, m = 360 / n → n ≠ 0 → m = 30) : n = 12 :=
  sorry

end regular_polygon_sides_l1932_193220


namespace compare_numbers_l1932_193259

theorem compare_numbers :
  2^27 < 10^9 ∧ 10^9 < 5^13 :=
by {
  sorry
}

end compare_numbers_l1932_193259


namespace partOneCorrectProbability_partTwoCorrectProbability_l1932_193272

noncomputable def teachers_same_gender_probability (mA fA mB fB : ℕ) : ℚ :=
  let total_outcomes := mA * mB + mA * fB + fA * mB + fA * fB
  let same_gender := mA * mB + fA * fB
  same_gender / total_outcomes

noncomputable def teachers_same_school_probability (SA SB : ℕ) : ℚ :=
  let total_teachers := SA + SB
  let total_outcomes := (total_teachers * (total_teachers - 1)) / 2
  let same_school := (SA * (SA - 1)) / 2 + (SB * (SB - 1)) / 2
  same_school / total_outcomes

theorem partOneCorrectProbability : teachers_same_gender_probability 2 1 1 2 = 4 / 9 := by
  sorry

theorem partTwoCorrectProbability : teachers_same_school_probability 3 3 = 2 / 5 := by
  sorry

end partOneCorrectProbability_partTwoCorrectProbability_l1932_193272


namespace total_wet_surface_area_correct_l1932_193205

namespace Cistern

-- Define the dimensions of the cistern and the depth of the water
def length : ℝ := 10
def width : ℝ := 8
def depth : ℝ := 1.5

-- Calculate the individual surface areas
def bottom_surface_area : ℝ := length * width
def longer_side_surface_area : ℝ := length * depth * 2
def shorter_side_surface_area : ℝ := width * depth * 2

-- The total wet surface area is the sum of all individual wet surface areas
def total_wet_surface_area : ℝ := 
  bottom_surface_area + longer_side_surface_area + shorter_side_surface_area

-- Prove that the total wet surface area is 134 m^2
theorem total_wet_surface_area_correct : 
  total_wet_surface_area = 134 := 
by sorry

end Cistern

end total_wet_surface_area_correct_l1932_193205


namespace rebus_solution_l1932_193206

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l1932_193206


namespace difference_between_Annette_and_Sara_l1932_193203

-- Define the weights of the individuals
variables (A C S B E : ℝ)

-- Conditions given in the problem
def condition1 := A + C = 95
def condition2 := C + S = 87
def condition3 := A + S = 97
def condition4 := C + B = 100
def condition5 := A + C + B = 155
def condition6 := A + S + B + E = 240
def condition7 := E = 1.25 * C

-- The theorem that we want to prove
theorem difference_between_Annette_and_Sara (A C S B E : ℝ)
  (h1 : condition1 A C)
  (h2 : condition2 C S)
  (h3 : condition3 A S)
  (h4 : condition4 C B)
  (h5 : condition5 A C B)
  (h6 : condition6 A S B E)
  (h7 : condition7 C E) :
  A - S = 8 :=
by {
  sorry
}

end difference_between_Annette_and_Sara_l1932_193203


namespace debby_bought_bottles_l1932_193292

def bottles_per_day : ℕ := 109
def days_lasting : ℕ := 74

theorem debby_bought_bottles : bottles_per_day * days_lasting = 8066 := by
  sorry

end debby_bought_bottles_l1932_193292


namespace find_tangent_line_l1932_193214

theorem find_tangent_line (k : ℝ) :
  (∃ k : ℝ, ∀ (x y : ℝ), y = k * (x - 1) + 3 ∧ k^2 + 1 = 1) →
  (∃ k : ℝ, k = 4 / 3 ∧ (k * x - y + 3 - k = 0) ∨ (x = 1)) :=
sorry

end find_tangent_line_l1932_193214


namespace lemonade_problem_l1932_193216

theorem lemonade_problem (L S W : ℕ) (h1 : W = 4 * S) (h2 : S = 2 * L) (h3 : L = 3) : L + S + W = 24 :=
by
  sorry

end lemonade_problem_l1932_193216


namespace range_of_a_l1932_193240

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 := by
  sorry

end range_of_a_l1932_193240


namespace min_value_a2_plus_b2_l1932_193238

theorem min_value_a2_plus_b2 (a b : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 2 * b = 0 -> x = -2) : (∃ a b, a = 1 ∧ b = -1 ∧ ∀ a' b', a^2 + b^2 ≥ a'^2 + b'^2) := 
by {
  sorry
}

end min_value_a2_plus_b2_l1932_193238


namespace cos_double_angle_l1932_193229

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi - θ) = 1 / 3) : 
  Real.cos (2 * θ) = 7 / 9 :=
by 
  sorry

end cos_double_angle_l1932_193229


namespace train_speed_is_144_kmph_l1932_193237

noncomputable def length_of_train : ℝ := 130 -- in meters
noncomputable def time_to_cross_pole : ℝ := 3.249740020798336 -- in seconds
noncomputable def speed_m_per_s : ℝ := length_of_train / time_to_cross_pole -- in m/s
noncomputable def conversion_factor : ℝ := 3.6 -- 1 m/s = 3.6 km/hr

theorem train_speed_is_144_kmph : speed_m_per_s * conversion_factor = 144 :=
by
  sorry

end train_speed_is_144_kmph_l1932_193237


namespace walking_rate_on_escalator_l1932_193270

/-- If the escalator moves at 7 feet per second, is 180 feet long, and a person takes 20 seconds to cover this length, then the rate at which the person walks on the escalator is 2 feet per second. -/
theorem walking_rate_on_escalator 
  (escalator_rate : ℝ)
  (length : ℝ)
  (time : ℝ)
  (v : ℝ)
  (h_escalator_rate : escalator_rate = 7)
  (h_length : length = 180)
  (h_time : time = 20)
  (h_distance_formula : length = (v + escalator_rate) * time) :
  v = 2 :=
by
  sorry

end walking_rate_on_escalator_l1932_193270


namespace square_assembly_possible_l1932_193243

theorem square_assembly_possible (Area1 Area2 Area3 : ℕ) (h1 : Area1 = 29) (h2 : Area2 = 18) (h3 : Area3 = 10) (h_total : Area1 + Area2 + Area3 = 57) : 
  ∃ s : ℝ, s^2 = 57 ∧ true :=
by
  sorry

end square_assembly_possible_l1932_193243


namespace final_building_height_l1932_193261

noncomputable def height_of_final_building 
    (Crane1_height : ℝ)
    (Building1_height : ℝ)
    (Crane2_height : ℝ)
    (Building2_height : ℝ)
    (Crane3_height : ℝ)
    (Average_difference : ℝ) : ℝ :=
    Crane3_height / (1 + Average_difference)

theorem final_building_height
    (Crane1_height : ℝ := 228)
    (Building1_height : ℝ := 200)
    (Crane2_height : ℝ := 120)
    (Building2_height : ℝ := 100)
    (Crane3_height : ℝ := 147)
    (Average_difference : ℝ := 0.13)
    (HCrane1 : 1 + (Crane1_height - Building1_height) / Building1_height = 1.14)
    (HCrane2 : 1 + (Crane2_height - Building2_height) / Building2_height = 1.20)
    (HAvg : (1.14 + 1.20) / 2 = 1.13) :
    height_of_final_building Crane1_height Building1_height Crane2_height Building2_height Crane3_height Average_difference = 130 := 
sorry

end final_building_height_l1932_193261


namespace fraction_multiplication_l1932_193291

theorem fraction_multiplication :
  (2 / (3 : ℚ)) * (4 / 7) * (5 / 9) * (11 / 13) = 440 / 2457 :=
by
  sorry

end fraction_multiplication_l1932_193291


namespace total_fruits_l1932_193246

-- Define the given conditions
variable (a o : ℕ)
variable (ratio : a = 2 * o)
variable (half_apples_to_ann : a / 2 - 3 = 4)
variable (apples_to_cassie : a - a / 2 - 3 = 0)
variable (oranges_kept : 5 = o - 3)

theorem total_fruits (a o : ℕ) (ratio : a = 2 * o) 
  (half_apples_to_ann : a / 2 - 3 = 4) 
  (apples_to_cassie : a - a / 2 - 3 = 0) 
  (oranges_kept : 5 = o - 3) : a + o = 21 := 
sorry

end total_fruits_l1932_193246


namespace range_of_m_l1932_193248

theorem range_of_m (m : ℝ) :
  (∃ x0 : ℝ, m * x0^2 + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m * x + 1 > 0) → -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l1932_193248


namespace factory_produces_6400_toys_per_week_l1932_193239

-- Definition of worker productivity per day
def toys_per_day : ℝ := 2133.3333333333335

-- Definition of workdays per week
def workdays_per_week : ℕ := 3

-- Definition of total toys produced per week
def toys_per_week : ℝ := toys_per_day * workdays_per_week

-- Theorem stating the total number of toys produced per week
theorem factory_produces_6400_toys_per_week : toys_per_week = 6400 :=
by
  sorry

end factory_produces_6400_toys_per_week_l1932_193239


namespace star_points_number_l1932_193236

-- Let n be the number of points in the star
def n : ℕ := sorry

-- Let A and B be the angles at the star points, with the condition that A_i = B_i - 20
def A (i : ℕ) : ℝ := sorry
def B (i : ℕ) : ℝ := sorry

-- Condition: For all i, A_i = B_i - 20
axiom angle_condition : ∀ i, A i = B i - 20

-- Total sum of angle differences equal to 360 degrees
axiom angle_sum_condition : n * 20 = 360

-- Theorem to prove
theorem star_points_number : n = 18 := by
  sorry

end star_points_number_l1932_193236


namespace min_m_n_sum_l1932_193230

theorem min_m_n_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 108 * m = n^3) : m + n = 8 :=
  sorry

end min_m_n_sum_l1932_193230


namespace trigonometric_expression_value_l1932_193258

variable {α : ℝ}
axiom tan_alpha_eq : Real.tan α = 2

theorem trigonometric_expression_value :
  (1 + 2 * Real.cos (Real.pi / 2 - α) * Real.cos (-10 * Real.pi - α)) /
  (Real.cos (3 * Real.pi / 2 - α) ^ 2 - Real.sin (9 * Real.pi / 2 - α) ^ 2) = 3 :=
by
  have h_tan_alpha : Real.tan α = 2 := tan_alpha_eq
  sorry

end trigonometric_expression_value_l1932_193258


namespace relation_between_abc_l1932_193227

theorem relation_between_abc (a b c : ℕ) (h₁ : a = 3 ^ 44) (h₂ : b = 4 ^ 33) (h₃ : c = 5 ^ 22) : a > b ∧ b > c :=
by
  -- Proof goes here
  sorry

end relation_between_abc_l1932_193227


namespace statements_evaluation_l1932_193260

-- Define the statements A, B, C, D, E as propositions
def A : Prop := ∀ (A B C D E : Prop), (A → ¬B ∧ ¬C ∧ ¬D ∧ ¬E)
def B : Prop := sorry  -- Assume we have some way to read the statement B under special conditions
def C : Prop := ∀ (A B C D E : Prop), (A ∧ B ∧ C ∧ D ∧ E)
def D : Prop := sorry  -- Assume we have some way to read the statement D under special conditions
def E : Prop := A

-- Prove the conditions
theorem statements_evaluation : ¬ A ∧ ¬ C ∧ ¬ E ∧ B ∧ D :=
by
  sorry

end statements_evaluation_l1932_193260


namespace ratio_singers_joined_second_to_remaining_first_l1932_193267

-- Conditions
def total_singers : ℕ := 30
def singers_first_verse : ℕ := total_singers / 2
def remaining_after_first : ℕ := total_singers - singers_first_verse
def singers_joined_third_verse : ℕ := 10
def all_singing : ℕ := total_singers

-- Definition for singers who joined in the second verse
def singers_joined_second_verse : ℕ := all_singing - singers_joined_third_verse - singers_first_verse

-- The target proof
theorem ratio_singers_joined_second_to_remaining_first :
  (singers_joined_second_verse : ℚ) / remaining_after_first = 1 / 3 :=
by
  sorry

end ratio_singers_joined_second_to_remaining_first_l1932_193267


namespace area_difference_l1932_193255

theorem area_difference (T_area : ℝ) (omega_area : ℝ) (H1 : T_area = (25 * Real.sqrt 3) / 4) 
  (H2 : omega_area = 4 * Real.pi) (H3 : 3 * (X - Y) = T_area - omega_area) :
  X - Y = (25 * Real.sqrt 3) / 12 - (4 * Real.pi) / 3 :=
by 
  sorry

end area_difference_l1932_193255


namespace frac_two_over_x_values_l1932_193242

theorem frac_two_over_x_values (x : ℝ) (h : 1 - 9 / x + 20 / (x ^ 2) = 0) :
  (2 / x = 1 / 2 ∨ 2 / x = 0.4) :=
sorry

end frac_two_over_x_values_l1932_193242


namespace average_people_added_each_year_l1932_193232

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end average_people_added_each_year_l1932_193232


namespace determine_x_l1932_193212

theorem determine_x (x : ℝ) (h : x^2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
by
  sorry

end determine_x_l1932_193212


namespace exists_pairwise_coprime_product_of_two_consecutive_integers_l1932_193211

theorem exists_pairwise_coprime_product_of_two_consecutive_integers (n : ℕ) (h : 0 < n) :
  ∃ (a : Fin n → ℕ), (∀ i, 2 ≤ a i) ∧ (Pairwise (IsCoprime on fun i => a i)) ∧ (∃ k : ℕ, (Finset.univ.prod a) - 1 = k * (k + 1)) := 
sorry

end exists_pairwise_coprime_product_of_two_consecutive_integers_l1932_193211


namespace polygon_sides_eq_eight_l1932_193293

theorem polygon_sides_eq_eight (x : ℕ) (h : x ≥ 3) 
  (h1 : 2 * (x - 2) = 180 * (x - 2) / 90) 
  (h2 : ∀ x, x + 2 * (x - 2) = x * (x - 3) / 2) : 
  x = 8 :=
by
  sorry

end polygon_sides_eq_eight_l1932_193293


namespace product_of_two_numbers_l1932_193223
noncomputable def find_product (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : ℝ :=
x * y

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : find_product x y h1 h2 = 200 :=
sorry

end product_of_two_numbers_l1932_193223


namespace binary_to_decimal_and_octal_l1932_193231

theorem binary_to_decimal_and_octal (binary_input : Nat) (h : binary_input = 0b101101110) :
    binary_input == 366 ∧ (366 : Nat) == 0o66 :=
by
  sorry

end binary_to_decimal_and_octal_l1932_193231


namespace system_unique_solution_l1932_193245

theorem system_unique_solution 
  (x y z : ℝ) 
  (h1 : x + y + z = 3 * x * y) 
  (h2 : x^2 + y^2 + z^2 = 3 * x * z) 
  (h3 : x^3 + y^3 + z^3 = 3 * y * z) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) 
  (hz : 0 ≤ z) : 
  (x = 1 ∧ y = 1 ∧ z = 1) := 
sorry

end system_unique_solution_l1932_193245


namespace parabola_translation_eq_l1932_193285

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -x^2 + 2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := - (x - 2)^2 - 1

-- State the theorem to prove the translated function
theorem parabola_translation_eq :
  ∀ x : ℝ, translated_parabola x = - (x - 2)^2 - 1 :=
by
  sorry

end parabola_translation_eq_l1932_193285


namespace hearty_total_beads_l1932_193204

-- Definition of the problem conditions
def blue_beads_per_package (r : ℕ) : ℕ := 2 * r
def red_beads_per_package : ℕ := 40
def red_packages : ℕ := 5
def blue_packages : ℕ := 3

-- Define the total number of beads Hearty has
def total_beads (r : ℕ) (rp : ℕ) (bp : ℕ) : ℕ :=
  (rp * red_beads_per_package) + (bp * blue_beads_per_package red_beads_per_package)

-- The theorem to be proven
theorem hearty_total_beads : total_beads red_beads_per_package red_packages blue_packages = 440 := by
  sorry

end hearty_total_beads_l1932_193204


namespace compound_oxygen_atoms_l1932_193256

theorem compound_oxygen_atoms 
  (C_atoms : ℕ)
  (H_atoms : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_O : ℝ) :
  C_atoms = 4 →
  H_atoms = 8 →
  total_molecular_weight = 88 →
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  atomic_weight_O = 16.00 →
  (total_molecular_weight - (C_atoms * atomic_weight_C + H_atoms * atomic_weight_H)) / atomic_weight_O = 2 := 
by 
  intros;
  sorry

end compound_oxygen_atoms_l1932_193256


namespace ratio_of_areas_is_one_ninth_l1932_193289

-- Define the side lengths of Square A and Square B
variables (x : ℝ)
def side_length_a := x
def side_length_b := 3 * x

-- Define the areas of Square A and Square B
def area_a := side_length_a x * side_length_a x
def area_b := side_length_b x * side_length_b x

-- The theorem to prove the ratio of areas
theorem ratio_of_areas_is_one_ninth : (area_a x) / (area_b x) = (1 / 9) :=
by sorry

end ratio_of_areas_is_one_ninth_l1932_193289


namespace problem1_problem2_problem3_problem4_l1932_193271

theorem problem1 : 9 - 5 - (-4) + 2 = 10 := by
  sorry

theorem problem2 : (- (3 / 4) + 7 / 12 - 5 / 9) / (-(1 / 36)) = 26 := by
  sorry

theorem problem3 : -2^4 - ((-5) + 1 / 2) * (4 / 11) + (-2)^3 / (abs (-3^2 + 1)) = -15 := by
  sorry

theorem problem4 : (100 - 1 / 72) * (-36) = -(3600) + (1 / 2) := by
  sorry

end problem1_problem2_problem3_problem4_l1932_193271


namespace calculate_exponent_product_l1932_193233

theorem calculate_exponent_product : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end calculate_exponent_product_l1932_193233


namespace bookstore_shoe_store_common_sales_l1932_193290

-- Define the conditions
def bookstore_sale_days (d: ℕ) : Prop := d % 4 = 0 ∧ d >= 4 ∧ d <= 28
def shoe_store_sale_days (d: ℕ) : Prop := (d - 2) % 6 = 0 ∧ d >= 2 ∧ d <= 26

-- Define the question to be proven as a theorem
theorem bookstore_shoe_store_common_sales : 
  ∃ (n: ℕ), n = 2 ∧ (
    ∀ (d: ℕ), 
      ((bookstore_sale_days d ∧ shoe_store_sale_days d) → n = 2) 
      ∧ (d < 4 ∨ d > 28 ∨ d < 2 ∨ d > 26 → n = 2)
  ) :=
sorry

end bookstore_shoe_store_common_sales_l1932_193290


namespace solve_for_a_b_l1932_193276

open Complex

theorem solve_for_a_b (a b : ℝ) (h : (mk 1 2) / (mk a b) = mk 1 1) : 
  a = 3 / 2 ∧ b = 1 / 2 :=
sorry

end solve_for_a_b_l1932_193276


namespace trigonometric_simplification_l1932_193207

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.cos α ^ 2 - 1) /
  (2 * Real.tan (π / 4 - α) * Real.sin (π / 4 + α) ^ 2) = 1 :=
sorry

end trigonometric_simplification_l1932_193207


namespace prop1_prop2_prop3_prop4_exists_l1932_193282

variable {R : Type*} [LinearOrderedField R]
def f (b c x : R) : R := abs x * x + b * x + c

theorem prop1 (b c x : R) (h : b > 0) : 
  ∀ {x y : R}, x ≤ y → f b c x ≤ f b c y := 
sorry

theorem prop2 (b c : R) (h : b < 0) : 
  ¬ ∃ a : R, ∀ x : R, f b c x ≥ f b c a := 
sorry

theorem prop3 (b c x : R) : 
  f b c (-x) = f b c x + 2*c := 
sorry

theorem prop4_exists (c : R) : 
  ∃ b : R, ∃ x y z : R, f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z := 
sorry

end prop1_prop2_prop3_prop4_exists_l1932_193282


namespace sum_medians_less_than_perimeter_l1932_193273

noncomputable def median_a (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * b^2 + 2 * c^2 - a^2).sqrt

noncomputable def median_b (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * c^2 - b^2).sqrt

noncomputable def median_c (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * b^2 - c^2).sqrt

noncomputable def sum_of_medians (a b c : ℝ) : ℝ :=
  median_a a b c + median_b a b c + median_c a b c

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  perimeter a b c / 2

theorem sum_medians_less_than_perimeter (a b c : ℝ) :
  semiperimeter a b c < sum_of_medians a b c ∧ sum_of_medians a b c < perimeter a b c :=
by
  sorry

end sum_medians_less_than_perimeter_l1932_193273


namespace area_triangle_DEF_l1932_193215

noncomputable def triangleDEF (DE EF DF : ℝ) (angleDEF : ℝ) : ℝ :=
  if angleDEF = 60 ∧ DF = 3 ∧ EF = 6 / Real.sqrt 3 then
    1 / 2 * DE * EF * Real.sin (Real.pi / 3)
  else
    0

theorem area_triangle_DEF :
  triangleDEF (Real.sqrt 3) (6 / Real.sqrt 3) 3 60 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end area_triangle_DEF_l1932_193215


namespace find_weight_of_first_new_player_l1932_193210

variable (weight_of_first_new_player : ℕ)
variable (weight_of_second_new_player : ℕ := 60) -- Second new player's weight is a given constant
variable (num_of_original_players : ℕ := 7)
variable (avg_weight_of_original_players : ℕ := 121)
variable (new_avg_weight : ℕ := 113)
variable (num_of_new_players : ℕ := 2)

def total_weight_of_original_players : ℕ := 
  num_of_original_players * avg_weight_of_original_players

def total_weight_of_new_players : ℕ :=
  num_of_new_players * new_avg_weight

def combined_weight_without_first_new_player : ℕ := 
  total_weight_of_original_players + weight_of_second_new_player

def weight_of_first_new_player_proven : Prop :=
  total_weight_of_new_players - combined_weight_without_first_new_player = weight_of_first_new_player

theorem find_weight_of_first_new_player : weight_of_first_new_player = 110 :=
by 
  sorry

end find_weight_of_first_new_player_l1932_193210


namespace complex_magnitude_problem_l1932_193277

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l1932_193277


namespace geometric_sequence_common_ratio_l1932_193279

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 1 / 4) : 
  q = 1 / 2 :=
  sorry

end geometric_sequence_common_ratio_l1932_193279


namespace total_cost_of_books_and_pencils_l1932_193269

variable (a b : ℕ)

theorem total_cost_of_books_and_pencils (a b : ℕ) : 5 * a + 2 * b = 5 * a + 2 * b := by
  sorry

end total_cost_of_books_and_pencils_l1932_193269


namespace total_combinations_l1932_193286

/-- Tim's rearrangement choices for the week -/
def monday_choices : Nat := 1
def tuesday_choices : Nat := 2
def wednesday_choices : Nat := 3
def thursday_choices : Nat := 2
def friday_choices : Nat := 1

theorem total_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 12 :=
by
  sorry

end total_combinations_l1932_193286


namespace car_travel_distance_l1932_193264

-- Definitions based on the conditions
def car_speed : ℕ := 60  -- The actual speed of the car
def faster_speed : ℕ := car_speed + 30  -- Speed if the car traveled 30 km/h faster
def time_difference : ℚ := 0.5  -- 30 minutes less in hours

-- The distance D we need to prove
def distance_traveled : ℚ := 90

-- Main statement to be proven
theorem car_travel_distance : ∀ (D : ℚ),
  (D / car_speed) = (D / faster_speed) + time_difference →
  D = distance_traveled :=
by
  intros D h
  sorry

end car_travel_distance_l1932_193264


namespace train_probability_correct_l1932_193287

/-- Define the necessary parameters and conditions --/
noncomputable def train_arrival_prob (train_start train_wait max_time_Alex max_time_train : ℝ) : ℝ :=
  let total_possible_area := max_time_Alex * max_time_train
  let overlap_area := (max_time_train - train_wait) * train_wait + (train_wait) * max_time_train / 2
  overlap_area / total_possible_area

/-- Main theorem stating that the probability is 3/10 --/
theorem train_probability_correct :
  train_arrival_prob 0 15 75 60 = 3 / 10 :=
by sorry

end train_probability_correct_l1932_193287


namespace find_floor_l1932_193201

-- Define the total number of floors
def totalFloors : ℕ := 9

-- Define the total number of entrances
def totalEntrances : ℕ := 10

-- Each floor has the same number of apartments
-- The claim we are to prove is that for entrance 10 and apartment 333, Petya needs to go to the 3rd floor.

theorem find_floor (apartment_number : ℕ) (entrance_number : ℕ) (floor : ℕ)
  (h1 : entrance_number = 10)
  (h2 : apartment_number = 333)
  (h3 : ∀ (f : ℕ), 0 < f ∧ f ≤ totalFloors)
  (h4 : ∃ (n : ℕ), totalEntrances * totalFloors * n >= apartment_number)
  : floor = 3 :=
  sorry

end find_floor_l1932_193201


namespace initial_sand_in_bucket_A_l1932_193226

theorem initial_sand_in_bucket_A (C : ℝ) : 
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  x / C = 1 / 4 := by
  let bucketB_capacity := C / 2
  let sand_in_B := (3 / 8) * bucketB_capacity
  let after_pour := (7 / 16) * C
  let x := after_pour - sand_in_B
  show x / C = 1 / 4
  sorry

end initial_sand_in_bucket_A_l1932_193226


namespace OC_eq_l1932_193235

variable {V : Type} [AddCommGroup V]

-- Given vectors a and b
variables (a b : V)

-- Conditions given in the problem
def OA := a + b
def AB := 3 • (a - b)
def CB := 2 • a + b

-- Prove that OC = 2a - 3b
theorem OC_eq : (a + b) + (3 • (a - b)) + (- (2 • a + b)) = 2 • a - 3 • b :=
by
  -- write your proof here
  sorry

end OC_eq_l1932_193235


namespace fraction_product_correct_l1932_193262

theorem fraction_product_correct : (3 / 5) * (4 / 7) * (5 / 9) = 4 / 21 :=
by
  sorry

end fraction_product_correct_l1932_193262


namespace fraction_value_l1932_193234

theorem fraction_value (a b : ℚ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 :=
by
  -- The proof goes here.
  sorry

end fraction_value_l1932_193234


namespace union_of_A_and_B_l1932_193249

def setA : Set ℝ := { x | -3 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3 }
def setB : Set ℝ := { x | 1 < x }

theorem union_of_A_and_B :
  setA ∪ setB = { x | -1 ≤ x } := sorry

end union_of_A_and_B_l1932_193249
